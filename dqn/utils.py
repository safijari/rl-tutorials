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

        self.buffer = np.zeros((h, w, num_stack), 'uint8')

    def _preprocess_frame(self, frame):
        image = cv2.resize(frame, (self.w, self.h))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def step(self, action):
        im, reward, done, info = self.env.step(action)
        im = self._preprocess_frame(im)
        self.buffer[:, :, 1:self.n] = self.buffer[:, :, 0:self.n-1]
        self.buffer[:, :, 0] = im
        return self.buffer.copy(), reward, done, info

    def reset(self):
        im = self.env.reset()
        im = self._preprocess_frame(im)
        self.buffer = np.dstack([im]*self.n)
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
