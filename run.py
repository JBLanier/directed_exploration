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
from tensorboardlogger import TensorboardLogger
import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def random_rollouts(obs_queue, msg_queue):

    game = BoxPush(display_width=64, display_height=64)
    p = ContinousPLE(game, fps=30, display_screen=False, add_noop_action=False)
    p.init()
    actions = {(0, 0), (0.5, 0), (0.5, 90), (0.5, 180), (0.5, -90)}
    action = (0, (0, 0))
    p.act(action)

    while True:

        if p.game_over():
            p.reset_game()

        if random.random() < 0.05:
            selection = random.sample(actions, 1)[0]
            action = (0, (selection[0], math.radians(selection[1])))

        if random.random() < 0.001:
            p.reset_game()

        observation = np.swapaxes(p.getScreenRGB(),0,1) / 255.0

        obs_queue.put(observation, block=True)

        #     # cv2.imshow("obs", observation[::,::-1])
        #     # cv2.waitKey(1)

        reward = p.act(action)

        try:
            msg = msg_queue.get(block=False)  # Read from the queue and do nothing
            print("msg: {}".format(msg))
            if (msg == 'QUIT'):
                print("time to quit")
                break
        except Empty:
            pass

def save_vae(vae):
    path = 'weights/vae_weights_{date:%Y-%m-%d_%H-%M-%S}.txt'.format(date=datetime.datetime.now())
    print('Saving to {}'.format(path))
    vae.save_model(path)

def train_vae(vae):
    batch_size = 256
    training_sample_amount = 10000000
    training_samples_left = training_sample_amount
    max_buffer_size = batch_size * 20
    sample_count = 0
    observation_queue = multiprocessing.Queue(maxsize=max_buffer_size)
    msg_queue = multiprocessing.Queue()
    process_num = 8
    processes = []
    for i in range(process_num):
        p = multiprocessing.Process(target=random_rollouts, args=(observation_queue, msg_queue))
        p.daemon = True
        p.start()
        processes.append(p)

    tb = TensorboardLogger(vae.vae)

    while training_samples_left != 0:
        batch_amt = min(training_samples_left, batch_size)
        samples = [observation_queue.get() for _ in range(batch_amt)]
        batch = np.stack(samples, axis=0)
        tb.log_batch(batch_amt, vae.train_on_batch(batch))
        training_samples_left -= batch_amt
        sample_count += batch_amt
        if sample_count % (batch_size * 100) == 0:
            print("{} samples trained on".format(sample_count))
            orig = batch[random.randint(0, batch_size - 1)]
            cv2.imshow("orig", orig[:, :, ::-1])
            cv2.imshow("reconstructed", np.squeeze(vae.vae.predict(np.expand_dims(orig, axis=0)))[:, :, ::-1])
            cv2.waitKey(1)

        if sample_count % (batch_size * 3000) == 0:
            save_vae(vae)

    save_vae(vae)

    print("quiting")

    for i in range(process_num):
        msg_queue.put('QUIT')
    for i in range(process_num):
        processes[i].join()

    observation_queue.close()
    msg_queue.close()
    #     processes[i].
    print("DONE")


def main(args):
    vae = VAE()

    if args.load_vae_weights:
        vae.load_model(args.load_vae_weights)

    tb = TensorboardLogger(vae.vae)

    if args.train_vae:
        train_vae(vae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-vae-weights", help="path to start VAE weights with",
                        type=str)
    parser.add_argument("--train-vae", help="if true, train VAE",
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




