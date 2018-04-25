import numpy as np
import keras
from boxpush import BoxPush
from continousple import ContinousPLE
import random
import math
import multiprocessing
from queue import Empty

import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'

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


class RolloutGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=256, dim=(64, 64)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size

        # batch_size = 256
        self.epoch_size = 1000000
        self.training_samples_left = self.epoch_size
        max_buffer_size = batch_size * 10
        sample_count = 0
        self.observation_queue = multiprocessing.Queue(maxsize=max_buffer_size)
        self.msg_queue = multiprocessing.Queue()
        self.process_num = 4
        self.processes = []
        for i in range(self.process_num):
            p = multiprocessing.Process(target=random_rollouts, args=(self.observation_queue, self.msg_queue))
            p.daemon = True
            p.start()
            self.processes.append(p)


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.epoch_size / self.batch_size))

    def __getitem__(self, index):

        batch_amt = min(self.training_samples_left, self.batch_size)
        samples = [self.observation_queue.get() for _ in range(batch_amt)]
        batch = np.stack(samples, axis=0)

        self.training_samples_left -= batch_amt

        return batch, batch

    def on_epoch_end(self):
        self.training_samples_left = self.epoch_size

