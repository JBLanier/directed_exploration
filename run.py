from boxpush import BoxPush
from continousple import ContinousPLE
import pygame
import math
import numpy as np
import cv2
import time
import multiprocessing
import random
from vae import VAE
from queue import Empty
import sys
import datetime
from matplotlib import pyplot as plt
import argparse
from tkinter import *
from randomrollouts import RolloutGenerator


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def save_vae(vae):
    path = 'weights/vae_weights_{date:%Y-%m-%d_%H-%M-%S}.txt'.format(date=datetime.datetime.now())
    print('Saving to {}'.format(path))
    vae.save_model(path)

def train_vae(vae):
    rollout_gen = RolloutGenerator()

    vae.vae.fit_generator(generator=rollout_gen,
                          use_multiprocessing=True,
                          workers=2)

def debug_play(vae):
    game = BoxPush(display_width=64, display_height=64)
    p = ContinousPLE(game, fps=30, display_screen=True, add_noop_action=False)

    p.init()

    pygame.joystick.init()

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    while True:
        frame_start_time = time.time()

        if p.game_over():
            p.reset_game()

        x = joystick.get_axis(0)
        y = joystick.get_axis(1)

        # print("X,Y: ({},{})".format(x,y))
        r, phi = cart2pol(x, y)

        if abs(r) < 0.12:
            r = 0
        else:
            r = min(1, r)
            r = r + 0.12 * math.log(r)
            r = r * 0.30

        observation = np.swapaxes(p.getScreenRGB(), 0, 1) / 255.0

        cv2.imshow("obs", np.squeeze(vae.vae.predict(np.expand_dims(observation, axis=0)))[:, :, ::-1])
        cv2.waitKey(1)

        action = (0, (r, phi))
        reward = p.act(action)

        time.sleep((15.4444444 - (time.time() - frame_start_time)) * 0.001)

def debug_latent_space(vae):

    def show_values(a):
        print(w1.get(), w2.get())
        print(a)

    master = Tk()
    w1 = Scale(master, from_=0, to=42)
    w1.set(19)
    w1.pack()
    w2 = Scale(master, from_=0, to=200, orient=HORIZONTAL, command=show_values )
    w2.set(23)
    w2.pack()
    Button(master, text='Show', command=show_values).pack()

    master.mainloop()

    print("yooo")

def main(args):
    vae = VAE()
    if args.load_vae_weights:
        vae.load_model(args.load_vae_weights)
    if args.train_vae:
        train_vae(vae)
    if args.debug_play:
        debug_play(vae)

    debug_latent_space(vae)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-vae-weights", help="path to start VAE weights with",
                        type=str)
    parser.add_argument("--train-vae", help="if true, train VAE",
                        action="store_true")
    parser.add_argument("--debug-play", help="use controller to debug environment",
                        action="store_true")
    args = parser.parse_args()
    main(args)


    # time.sleep((15.4444444 - (time.time() - frame_start_time)) * 0.001)

    #
    # if sample_count % (batch_size*100) == 0:
    #     print("{} samples trained on".format(sample_count))
    #
    # if sample_count % (batch_size*20) == 0:
    #     orig = observation_batch[random.randint(0,batch_size-1)]
    #
    #     cv2.imshow("orig", orig[:,:,::-1])
    #     cv2.imshow("reconstructed", np.squeeze(vae.vae.predict(np.expand_dims(orig,axis=0)))[:,:,::-1])
    #     cv2.waitKey(1)
    #
    # vae.train_on_batch(observation_batch)




