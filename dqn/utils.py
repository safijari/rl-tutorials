import cv2
import numpy as np
import gym
import time
from random import randint


class FrameStackingAndResizingEnv:
    def __init__(self, env, w, h, num_stack=4):
        self.env = env
        self.n = num_stack
        self.w = w
        self.h = h

        self.buffer = np.zeros((num_stack, h, w), 'uint8')
        self.frame = None

    def _preprocess_frame(self, frame):
        image = cv2.resize(frame, (self.w, self.h))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def step(self, action):
        im, reward, done, info = self.env.step(action)
        self.frame = im.copy()
        im = self._preprocess_frame(im)
        self.buffer[1:self.n, :, :] = self.buffer[0:self.n-1, :, :]
        self.buffer[0, :, :] = im
        return self.buffer.copy(), reward, done, info

    def render(self, mode):
        if mode == 'rgb_array':
            return self.frame
        return super(FrameStackingAndResizingEnv, self).render(mode)

    @property
    def observation_space(self):
        # gym.spaces.Box()
        return np.zeros((self.n, self.h, self.w))

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        im = self.env.reset()
        self.frame = im.copy()
        im = self._preprocess_frame(im)
        self.buffer = np.stack([im]*self.n, 0)
        return self.buffer.copy()

    def render(self, mode):
        self.env.render(mode)


if __name__ == "__main__":
    env = gym.make("Breakout-v0")
    env = FrameStackingAndResizingEnv(env, 480, 640)

    # print(env.observation_space.shape)
    # print(env.action_space)

    im = env.reset()
    idx = 0
    ims = []
    for i in range(im.shape[-1]):
        ims.append(im[:, :, i])
    cv2.imwrite(f"/tmp/{idx}.jpg", np.hstack(ims))

    env.step(1)

    for _ in range(10):
        idx += 1
        im, _, _, _ = env.step(randint(0, 3))

        ims = []
        for i in range(im.shape[-1]):
            ims.append(im[:, :, i])
        cv2.imwrite(f"/tmp/{idx}.jpg", np.hstack(ims))
