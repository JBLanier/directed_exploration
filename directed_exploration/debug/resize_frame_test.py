import gym
import cv2

from directed_exploration.utils.env_util import ResizeFrameWrapper

env_id = 'BreakoutDeterministic-v4'

env = ResizeFrameWrapper(gym.make(env_id), 84, 84)

env.reset()
while True:
    obs, _, done, _ = env.step(env.action_space.sample())
    cv2.imshow('obs', obs)
    cv2.waitKey(33)
    if done:
        env.reset()
