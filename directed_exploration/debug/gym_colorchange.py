
import gym
import numpy as np
from collections import deque
"""
Extremely simple gym for basic debugging
"""


class GymBrightnessChange(gym.Env):

    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    def _get_obs(self):
        return np.ones((64, 64, 3), dtype=np.uint8) * int(self.signal[self.current_signal_index] * 255)

    def reset(self):
        random_num = np.random.random() * 3
        signal_length = int(800 * random_num)

        if signal_length < 2:
            return self.reset()

        x = np.linspace(0, 100 * np.pi * random_num, signal_length)
        self.signal = np.random.normal(0, .5, size=signal_length) + \
                      (2 * np.sin(.6 * x + np.random.random() * 10)) + \
                      (5 * np.sin(.1 * x + np.random.random() * 10))
        self.signal = (self.signal + 10) / 20
        self.current_signal_index = 0
        return self._get_obs()

    def step(self, action):
        if self.current_signal_index >= len(self.signal):
            print("\n\nstep() was called when environment is already done!!\n\n")

        obs = self._get_obs()

        self.current_signal_index += 1

        reward = 0
        done = self.current_signal_index >= len(self.signal)
        info = {}

        return obs, reward, done, info

    def close(self):
        pass

    def render(self, mode='human'):
        obs = self._get_obs()

        if mode == 'human':
            import cv2
            cv2.imshow('frame', obs)
            cv2.waitKey(1)

        if mode == 'rgb_array' or mode == 'state_pixels':
            return obs