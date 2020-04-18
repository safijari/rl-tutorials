import gym
import time

env = gym.make("Breakout-v0")

print(env.observation_space.shape)
print(env.action_space)

env.reset()
env.render()

while True:
    act = int(input('type action'))
    print(env.step(act))
    env.render()
