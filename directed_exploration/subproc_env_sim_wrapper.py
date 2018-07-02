from directed_exploration import simulator_data_ops as sdo
from directed_exploration.vae import VAE
from directed_exploration.state_rnn import StateRNN
from directed_exploration.utils.data_util import convertToOneHot
from directed_exploration.heatmap_gen import generate_boxpush_heatmap_from_npy_records
from directed_exploration.sim import Sim

from collections import deque
import tensorflow as tf
import numpy as np
import os
import gym
import cv2
from functools import reduce

import logging

logger = logging.getLogger(__name__)

HEATMAP_FOLDER_NAME = 'heatmap_records'


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def pretty_dict_keys_with_values(dictonary):
    pretty_str = ""
    for key in dictonary.keys():
        pretty_str = "{}{}: {} ".format(pretty_str, key, dictonary[key])
    return pretty_str

class SubprocEnvSimWrapper:
    def __init__(self, working_dir, sim, train_seq_length, sequences_per_epoch, subproc_env,
                 heatmaps=False, validation_data_dir=None, return_generated_frames_in_info=False, do_train=True):

        self.sim = sim

        self.working_dir = working_dir
        self.validation_data_dir = validation_data_dir
        self.return_generated_frames_in_info = return_generated_frames_in_info
        self.do_train = do_train
        self.train_seq_length = train_seq_length
        self.sequences_per_epoch = sequences_per_epoch
        self.heatmaps = heatmaps
        self.subproc_env = subproc_env
        self.num_envs = self.subproc_env.num_envs

        self.action_space = self.subproc_env.action_space
        self.observation_space = self.subproc_env.observation_space

        self.loss_accumulators = None

        self.t_minus_1_dones = [True for _ in range(self.num_envs)]

        self.t_obs = self.subproc_env.reset() / 255.0
        self.t_dones = [False for _ in range(self.num_envs)]

        self.train_states = None
        self.t_states = None

        if heatmaps:
            self.set_heatmap_record_write_to_current_step()
            self.old_prefix = self.get_current_heatmap_record_prefix()

        self.minibatch_observations = []
        self.minibatch_actions = []
        self.minibatch_dones = []

    def step(self, actions):
        t_plus_1_obs, _, t_plus_1_dones, _ = self.subproc_env.step(actions)
        t_plus_1_obs = t_plus_1_obs / 255.0

        predict_vals = self.sim.predict_on_batch(
            t_obs=self.t_obs,
            t_actions=actions,
            t_states=self.t_states,
            t_minus_1_dones=self.t_minus_1_dones,
            actual_t_plus_one_obs=t_plus_1_obs,
            return_t_plus_one_predictions=self.return_generated_frames_in_info
        )

        if self.return_generated_frames_in_info:
            t_plus_1_predictions, losses, t_plus_1_states = predict_vals
        else:
            losses, t_plus_1_states = predict_vals
            t_plus_1_predictions = None

        self.minibatch_observations.append(self.t_obs)
        self.minibatch_actions.append(convert_to_one_hot(actions, self.action_space.n))
        self.minibatch_dones.append(self.t_minus_1_dones)

        if len(self.minibatch_actions) >= self.train_seq_length:
            self.minibatch_observations.append(t_plus_1_obs)
            self.minibatch_dones.append(self.t_dones)

            if self.do_train:
                self.train()

        if self.heatmaps and self.sim.get_current_step % 50 == 0:
            self.set_heatmap_record_write_to_current_step()
            logger.debug("Creating Heatmap...")
            heatmap_save_location = generate_boxpush_heatmap_from_npy_records(
                directory=os.path.join(self.working_dir, HEATMAP_FOLDER_NAME),
                file_prefix=self.old_prefix,
                delete_records=True)
            logger.debug("Heatmap saved to {}".format(heatmap_save_location))
            self.old_prefix = self.get_current_heatmap_record_prefix()

        # Move iteration forward
        self.t_minus_1_dones = self.t_dones
        self.t_obs = t_plus_1_obs
        self.t_dones = t_plus_1_dones
        self.t_states = t_plus_1_states

        return t_plus_1_obs, losses, t_plus_1_dones, {'generated_frames': t_plus_1_predictions}

    def reset(self):
        logger.info("RESET WAS CALLED")
        self.t_minus_1_dones = [True for _ in range(self.num_envs)]

        self.t_obs = self.subproc_env.reset() / 255.0
        self.t_dones = [False for _ in range(self.num_envs)]

        self.t_states = None
        self.train_states = None

        self.minibatch_observations = []
        self.minibatch_actions = []
        self.minibatch_dones = []

        return self.t_obs

    def render_actual_frames(self):
        return self.subproc_env.render()

    def get_current_heatmap_record_prefix(self):
        return "step{}".format(self.sim.get_current_step())

    def set_heatmap_record_write_to_current_step(self):
        self.subproc_env.set_record_write(write_dir=os.path.join(self.working_dir, HEATMAP_FOLDER_NAME),
                                          prefix=self.get_current_heatmap_record_prefix())

    def train(self):

        step, losses, states_out = self.sim.train_on_batch(self.minibatch_observations, self.minibatch_actions, self.minibatch_dones, self.train_states)

        if self.loss_accumulators is None:
            self.loss_accumulators = losses
        else:
            for key in self.loss_accumulators.keys():
                self.loss_accumulators[key] += losses[key]

        if step == 1 or step % 100 == 0:
            for key in self.loss_accumulators.keys():
                self.loss_accumulators[key] /= len(self.minibatch_actions)
            logger.info(pretty_dict_keys_with_values(self.loss_accumulators))

            # reset accumulators
            self.loss_accumulators = None

        self.train_states = states_out

        self.minibatch_observations = []
        self.minibatch_actions = []
        self.minibatch_dones = []

        if step == 1 or step % 2000 == 0:

            self.sim.save_model()
            if self.validation_data_dir is not None:
                val_losses = self.sim.validate(validation_data_dir=self.validation_data_dir,
                                  allowed_action_space=self.action_space)

                logger.info(pretty_dict_keys_with_values(val_losses))

