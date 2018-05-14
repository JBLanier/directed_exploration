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
import os
import re
import tensorflow as tf
import pickle

MAX_VAE_TF_RECORD_LENGTH = 1000

last_scale_adjust_time = time.time()

tf.set_random_seed(42)

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def save_vae(vae):
    path = 'weights/vae_weights_{date:%Y-%m-%d_%H-%M-%S}.txt'.format(date=datetime.datetime.now())
    print('Saving to {}'.format(path))
    vae.save_model(path)


def get_numbered_tfrecord_file_names_from_directory(dir, prefix):
    dir_files = os.listdir(dir)
    dir_files = sorted(filter(lambda f: str.isdigit(re.split('[_.]+', f)[1]) and f.startswith(prefix), dir_files),
                       key=lambda f: int(re.split('[_.]+', f)[1]))
    return list(map(lambda f: os.path.join(dir, f), dir_files))


def get_vae_tfrecord_input_fn(train_data_dir, batch_size=32, num_epochs=1):
    prefix = 'vae'
    input_file_names = get_numbered_tfrecord_file_names_from_directory(dir=train_data_dir, prefix=prefix)

    if len(input_file_names) <= 0:
        raise FileNotFoundError("No usable tfrecords with prefix \'{}\' were found at {}".format(
            prefix, train_data_dir)
        )

    def decode_pickled_np_array(np_bytes):
        return pickle.loads(np_bytes).astype(np.float32)

    def parse_fn(example):
        example_fmt = {
            "action_at_frame": tf.FixedLenFeature([2], tf.float32, default_value=[-math.inf, -math.inf]),
            "frame_bytes": tf.FixedLenFeature([], tf.string)
        }

        parsed = tf.parse_single_example(example, example_fmt)

        action = parsed["action_at_frame"]
        frame_bytes = parsed["frame_bytes"]

        frame = tf.py_func(func=decode_pickled_np_array, inp=[frame_bytes],
                           Tout=tf.float32, stateful=False, name='decode_np_bytes')

        return frame

    def extract_and_shuffle_fn(file_tensor):
        return tf.data.TFRecordDataset(file_tensor).shuffle(buffer_size=MAX_VAE_TF_RECORD_LENGTH)

    def input_fn():
        file_names = tf.constant(input_file_names, dtype=tf.string, name='input_file_names')
        file_tensors = tf.data.Dataset.from_tensor_slices(file_names).shuffle(len(input_file_names))

        # Shuffle frames in each episode/tfrecords file, then draw a frame from each episode/tfrecords file in a cycle
        cycle_length = 120
        dataset = file_tensors.interleave(map_func=extract_and_shuffle_fn,
                                          cycle_length=cycle_length,
                                          block_length=1)

        # Shuffle drawn frames
        dataset = dataset.shuffle(buffer_size=cycle_length*3)

        # Parse tfrecords into frames
        dataset = dataset.map(map_func=parse_fn, num_parallel_calls=multiprocessing.cpu_count())

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.repeat(num_epochs)  # the input is repeated indefinitely if num_epochs is None
        dataset = dataset.prefetch(buffer_size=10)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return input_fn


def train_vae(vae, train_data_dir):

    input_fn = get_vae_tfrecord_input_fn(train_data_dir, batch_size=256, num_epochs=5)

    vae.train_on_input_fn(input_fn)

    vae.save_model()

    # checkpoint_callback = ModelCheckpoint(filepath="weights8/weights.{epoch:02d}-{loss:.2f}.hdf5",
    #                                       monitor='loss',
    #                                       save_weights_only=True,
    #                                       period=1)
    #
    # tensorboard_callback = TensorBoard(log_dir='./graph',
    #                                    histogram_freq=0,
    #                                    write_graph=True,
    #                                    write_images=True)
    #
    #
    # vae.vae.fit_generator(generator=rollout_gen,
    #                       epochs=40,
    #                       callbacks=[checkpoint_callback, tensorboard_callback])


def debug_play(vae, env_name):
    env = gym.make(env_name)
    env.reset()

    pygame.init()
    pygame.joystick.init()

    while True:
        frame_start_time = time.time()

        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

            # # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
            # if event.type == pygame.JOYBUTTONDOWN:
            #     print("Joystick button pressed.")
            # if event.type == pygame.JOYBUTTONUP:
            #     print("Joystick button released.")

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

        frame, _, _, _ = env.step(np.array([r, phi]))

        frame = frame/255.0

        cv2.imshow("encoded_decoded", np.squeeze(vae.encode_decode_frames(np.expand_dims(frame, axis=0)))[:, :, ::-1])
        cv2.imshow("encoded_decoded2", np.squeeze(vae.decode_frames(vae.encode_frames(np.expand_dims(frame, axis=0))))[:, :, ::-1])
        cv2.imshow("orig", frame[:,:,::-1])

        cv2.waitKey(1)

        # time.sleep((15.4444444 - (time.time() - frame_start_time)) * 0.001)


def debug_latent_space(vae, env_name):

    def show_values(a):
        global last_scale_adjust_time
        if time.time() - last_scale_adjust_time > 0.1:
            new_z = []
            for w in scales:
                new_z.append(w.get())
            cv2.imshow("decoded", np.squeeze(vae.decode_frames(np.expand_dims(new_z, axis=0))))
            cv2.waitKey(1)
            last_scale_adjust_time = time.time()

    env = gym.make(env_name)
    observation = env.reset()
    env.close()

    z = vae.encode_frames(np.expand_dims(observation, axis=0))
    cv2.imshow("decoded", np.squeeze(vae.decode_frames(z)))

    print("z shape: {}".format(z.shape))
    z = z[0]
    print("Starting z:\n{}".format(z))

    master = Tk()
    scales = []

    for i in range(len(z)):
        w = Scale(master, from_=-4.0, to=4.0, orient=HORIZONTAL, command=show_values, resolution=0.01, width=5,
                  length=200, showvalue=False)
        w.set(z[i])
        w.pack()
        scales.append(w)

    master.mainloop()

    print("done debugging latent space")


def main(args):
    print("RESTORE FROM DIR: {}".format(args.load_vae_weights))
    vae = VAE(restore_from_dir=args.load_vae_weights, latent_dim=1)
    if args.train_vae:
        if args.train_data_dir:
            train_vae(vae, args.train_data_dir)
        else:
            print("Must specify --train-data-dir")
            exit(1)
    if args.debug_play:
        if args.load_vae_weights:
            debug_play(vae, args.env)
        else:
            print("Must specify --load-vae-weights")
            exit(1)
    if args.debug_latent_space:
        if args.load_vae_weights:
            debug_latent_space(vae, args.env)
        else:
            print("Must specify --load-vae-weights")
            exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-vae-weights", help="dir to load VAE weights with",
                        type=str, default=None)
    parser.add_argument("--train-vae", help="if true, train VAE",
                        action="store_true")
    parser.add_argument("--train-data-dir", help="VAE training data location",
                       type=str)
    parser.add_argument("--debug-play", help="use controller to debug environment",
                        action="store_true")
    parser.add_argument("--debug-latent-space", help="play with latent space variables",
                        action="store_true")
    parser.add_argument("--env", help="environment to use",
                        type=str, default='boxpushsimple-v0')
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
