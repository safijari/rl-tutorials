import torch
import numpy as np
from tqdm import tqdm
import gym
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from typing import Any
from random import sample, random
import wandb
from collections import deque
import argh
import moviepy.editor as ed


@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class Model(nn.Module):
    def __init__(self, obs_shape, num_actions, lr=0.001):
        super(Model, self).__init__()
        assert len(obs_shape) == 1, "This network only works for flat observations"
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions),
        )
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.buffer = [None]*buffer_size
        self.idx = 0

    def insert(self, sars):
        self.buffer[self.idx % self.buffer_size] = sars
        self.idx += 1

    def sample(self, num_samples):
        assert num_samples < min(self.idx, self.buffer_size)
        # if num_samples > min(self.idx, self.buffer_size):
        if self.idx < self.buffer_size:
            return sample(self.buffer[:self.idx], num_samples)
        return sample(self.buffer, num_samples)


def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())


def train_step(model, state_transitions, tgt, num_actions, device, gamma=0.99):
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    mask = torch.stack(
        (
            [
                torch.Tensor([0]) if s.done else torch.Tensor([1])
                for s in state_transitions
            ]
        )
    ).to(device)
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions])).to(device)
    actions = [s.action for s in state_transitions]

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]  # (N, num_actions)

    model.opt.zero_grad()
    qvals = model(cur_states)  # (N, num_actions)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    loss = (
        (rewards + mask[:, 0] * qvals_next * 0.99 - torch.sum(qvals * one_hot_actions, -1))
        ** 2
    ).mean()
    loss.backward()
    model.opt.step()
    return loss


def run_test_make_movie(model, env, device, framerate=25, path=''):
    last_observation = env.reset()
    last_frame = env.render(mode='rgb_array')
    frames = []

    done = False
    reward = 0

    while not done:
        frames.append(ed.ImageClip(last_frame).set_duration(1.0/framerate))
        action = model(torch.Tensor(last_observation).to(device)).max(-1)[-1].item()
        last_observation, r, done, _ = env.step(action)

        reward += r

        last_frame = env.render(mode='rgb_array')

    if path:
        concat_clip = ed.concatenate_videoclips(frames, method='compose')
        concat_clip.write_videofile(path, fps=framerate)

    return reward


def main(name, test=False, chkpt=None, device='cuda'):
    memory_size = 50000
    min_rb_size = 20000
    sample_size = 100
    lr = 0.0001

    eps_min = 0.05

    # eps_decay = 0.999995
    eps_decay = 0.99999

    env_steps_before_train = 10
    tgt_model_update = 5000

    if not test:
        wandb.init(project="dqn-tutorial", name=name,
                   config={'sample_size': sample_size,
                           'learning_rate': lr,
                           'minimum_epsilon': eps_min,
                           'epsilon_decay_factor': eps_decay,
                           'env_steps_before_train': env_steps_before_train,
                           'epochs_before_target_model_update': tgt_model_update})

    env = gym.make("CartPole-v1")
    test_env = gym.make("CartPole-v1")
    last_observation = env.reset()

    m = Model(env.observation_space.shape, env.action_space.n, lr=lr).to(device)
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))
    tgt = Model(env.observation_space.shape, env.action_space.n).to(device)
    update_tgt_model(m, tgt)

    rb = ReplayBuffer()
    steps_since_train = 0
    epochs_since_tgt = 0

    step_num = -1 * min_rb_size

    episode_rewards = []
    rolling_reward = 0

    tq = tqdm()
    try:
        while True:
            if test:
                env.render()
                time.sleep(0.05)
            tq.update(1)

            eps = eps_decay ** (step_num)
            if test:
                eps = 0

            if random() < eps:
                action = (
                    env.action_space.sample()
                )  # your agent here (this takes random actions)
            else:
                action = m(torch.Tensor(last_observation).to(device)).max(-1)[-1].item()

            observation, reward, done, info = env.step(action)
            rolling_reward += reward

            reward = reward * 0.1

            rb.insert(Sarsd(last_observation, action, reward, observation, done))

            last_observation = observation

            if done:
                episode_rewards.append(rolling_reward)
                if test:
                    print(rolling_reward)
                rolling_reward = 0
                observation = env.reset()

            steps_since_train += 1
            step_num += 1

            if (
                (not test)
                and rb.idx > min_rb_size
                and steps_since_train > env_steps_before_train
            ):
                loss = train_step(m, rb.sample(sample_size), tgt, env.action_space.n, device)
                wandb.log(
                    {
                        "loss": loss.detach().cpu().item(),
                        "eps": eps,
                        "avg_reward": np.mean(episode_rewards),
                    },
                    step=step_num,
                )
                episode_rewards = []
                epochs_since_tgt += 1
                if epochs_since_tgt > tgt_model_update:
                    print("updating target model")
                    update_tgt_model(m, tgt)
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(), f"models/{step_num}.pth")
                    path = f"/tmp/{step_num}.mp4"
                    test_rew = run_test_make_movie(m, test_env, device, path=path)
                    wandb.log({"test_reward": test_rew,
                               "test_video": wandb.Video(path, caption=str(test_rew))}, step=step_num)

                steps_since_train = 0

    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":
    argh.dispatch_command(main)
