from boxpush import BoxPush
from continousple import ContinousPLE
import pygame
import math
import numpy as np
import cv2
import time
import random


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


game = BoxPush(display_width=500, display_height=500)
p = ContinousPLE(game, fps=30, display_screen=True, add_noop_action=False)

p.init()
reward = 0.0

test_image_shown = False

actions = {(0, 0), (0.5, 0), (0.5, 90), (0.5, 180), (0.5, -90)}
action = (0, (0, 0))
while True:
    frame_start_time = time.time()

    if p.game_over():
        p.reset_game()

    if random.random() < 0.05:
        selection = random.sample(actions, 1)[0]
        action = (0, (selection[0], math.radians(selection[1])))

    if random.random() < 0.001:
        p.reset_game()
        print("reset")

    observation = np.swapaxes(p.getScreenRGB(),0,1)

    # if not test_image_shown:
    # 	cv2.imshow("obs", observation[...,::-1])
    # 	cv2.waitKey(1)

    reward = p.act(action)

    time.sleep((15.4444444 - (time.time() - frame_start_time)) * 0.001)
