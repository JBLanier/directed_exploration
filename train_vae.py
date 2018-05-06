
import pygame
import gym
import gym_boxpush
import pyglet
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
from keras.callbacks import ModelCheckpoint, TensorBoard

last_scale_adjust_time = time.time()

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def save_vae(vae):
    path = 'weights/vae_weights_{date:%Y-%m-%d_%H-%M-%S}.txt'.format(date=datetime.datetime.now())
    print('Saving to {}'.format(path))
    vae.save_model(path)

# def train_vae(vae):
#     rollout_gen = RolloutGenerator()
#
#     checkpoint_callback = ModelCheckpoint(filepath="weights8/weights.{epoch:02d}-{loss:.2f}.hdf5",
#                                           monitor='loss',
#                                           save_weights_only=True,
#                                           period=1)
#
#     tensorboard_callback = TensorBoard(log_dir='./graph',
#                                        histogram_freq=0,
#                                        write_graph=True,
#                                        write_images=True)
#
#
#     vae.vae.fit_generator(generator=rollout_gen,
#                           epochs=40,
#                           callbacks=[checkpoint_callback, tensorboard_callback])
#
#     print("quitting rollout pygame environments...")
#     rollout_gen.quit()

def debug_play(vae):
    env = gym.make('boxpush-v0')
    env.reset()

    pygame.init()
    pygame.joystick.init()

    while True:
        frame_start_time = time.time()

        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

            # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
            if event.type == pygame.JOYBUTTONDOWN:
                print("Joystick button pressed.")
            if event.type == pygame.JOYBUTTONUP:
                print("Joystick button released.")

        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        x = joystick.get_axis(0)
        y = -joystick.get_axis(1)


        # print("X,Y: ({},{})".format(x,y))
        r, phi = cart2pol(x, y)

        if abs(r) < 0.12:
            r = 0
        else:
            r = min(1, r)
            r = r + 0.12 * math.log(r)
            r = max(0.0, r)

        phi = phi / math.pi

        # cv2.imshow("obs", np.squeeze(vae.vae.predict(np.expand_dims(observation, axis=0)))[:, :, ::-1])
        # cv2.waitKey(1)
        env.render()

        env.step(np.array([r, phi]))

        # time.sleep((15.4444444 - (time.time() - frame_start_time)) * 0.001)

# def debug_latent_space(vae):
#
#
#     def show_values(a):
#         global last_scale_adjust_time
#         if time.time() - last_scale_adjust_time > 0.1:
#             new_z = []
#             for w in scales:
#                 new_z.append(w.get())
#             cv2.imshow("decoded", np.squeeze(vae.decode(np.expand_dims(new_z,axis=0))))
#             cv2.waitKey(1)
#             last_scale_adjust_time = time.time()
#
#     game = BoxPush(display_width=64, display_height=64)
#     p = ContinousPLE(game, fps=30, display_screen=False, add_noop_action=False)
#     p.init()
#     p.act((0,(0,0)))
#     p.act((0,(0,0)))
#     observation = np.swapaxes(p.getScreenRGB(), 0, 1) / 255.0
#     p.quit()
#     z = np.squeeze(vae.encode(np.expand_dims(observation,axis=0)))
#     cv2.imshow("decoded", np.squeeze(vae.decode(np.expand_dims(z,axis=0))))
#
#     print("z shape: {}", format(z.shape))
#
#     master = Tk()
#     scales = []
#
#     for i in range(len(z)):
#         w = Scale(master, from_=-3.0, to=3.0, orient=HORIZONTAL, command=show_values, resolution=0.01, width=5,
#                   length=200, showvalue=False)
#         w.set(z[i])
#         w.pack()
#         scales.append(w)
#
#
#     master.mainloop()
#
#     print("done debugging latent space")

def main(args):
    vae = None
    if args.load_vae_weights:
        vae.load_model(args.load_vae_weights)
    # if args.train_vae:
    #     train_vae(vae)
    if args.debug_play:
        debug_play(vae)
    # if args.debug_latent_space:
    #     debug_latent_space(vae)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-vae-weights", help="path to start VAE weights with",
                        type=str)
    parser.add_argument("--train-vae", help="if true, train VAE",
                        action="store_true")
    parser.add_argument("--debug-play", help="use controller to debug environment",
                        action="store_true")
    parser.add_argument("--debug-latent-space", help="play with latent space variables",
                        action="store_true", default=True)
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




