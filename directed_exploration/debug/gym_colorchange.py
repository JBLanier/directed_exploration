
import gym
import numpy as np
from collections import deque
"""
Extremely simple gym for basic debugging
"""


class GymBrightnessChange(gym.Env):

    def __init__(self):
        self.current_brightness = 0.5
        self.k = 3
        self.previous_actions = deque(maxlen=self.k)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    def _get_obs(self):
        return np.ones((64, 64, 3), dtype=np.uint8) * int(self.current_brightness * 255)

    def reset(self):
        # print("reset_called")
        self.current_brightness = 0.5
        self.previous_actions = deque(maxlen=self.k)
        return self._get_obs()

    def step(self, action):
        # old = self.current_brightness

        # if action == 1:
        #     # self.current_brightness = max(self.current_brightness - 0.1, 0)
        #     self.current_brightness = 0.0
        # elif action == 2:
        #     # self.current_brightness = min(self.current_brightness + 0.1, 1)
        #     self.current_brightness = 1.0

        if len(self.previous_actions) == self.k and self.previous_actions[0] == action:
            self.current_brightness = -1.0
        else:
            self.current_brightness = 1.0

        obs = self._get_obs()
        reward = 0
        done = np.random.choice([0, 1], p=[0.95, 0.05])
        info = {}

        # print("curr: {} action: {} new: {}".format(old, action, self.current_brightness))
        # if done:
        #     print("done")

        self.previous_actions.append(action)

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