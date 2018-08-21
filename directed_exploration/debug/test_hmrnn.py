import gym
import cv2
import numpy as np

from gym.envs.registration import register

register(
    id='colorchange-v0',
    entry_point='directed_exploration.debug.gym_colorchange:GymBrightnessChange',
)

env = gym.make('colorchange-v0')

env.reset()

print(env.action_space)

for i in range(100000000):

    obs, reward, done, info = env.step(env.action_space.sample())

    cv2.imshow('robot', obs)
    cv2.waitKey(1)