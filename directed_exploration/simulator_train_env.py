from directed_exploration.utils.env_util import make_record_write_subproc_env, make_subproc_env
from directed_exploration import simulator_data_ops as sdo
from directed_exploration.vae import VAE
from directed_exploration.state_rnn import StateRNN
from directed_exploration.utils.data_util import convertToOneHot
from directed_exploration.heatmap_gen import generate_boxpush_heatmap_from_npy_records
from directed_exploration.validation import validate_vae_state_rnn_pair_on_tf_records

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

class SimulatorTrainEnv:

    def __init__(self, working_dir, sess, latent_dim, train_seq_length, sequences_per_epoch, env_id, num_env,
                 heatmaps=False, validation_data_dir=None, return_generated_frames_in_info=False, do_train=True):

        self.working_dir = working_dir
        self.validation_data_dir = validation_data_dir
        self.return_generated_frames_in_info = return_generated_frames_in_info
        self.do_train = do_train
        self.sess = sess
        self.train_seq_length = train_seq_length
        self.sequences_per_epoch = sequences_per_epoch
        self.heatmaps = heatmaps
        self.num_envs = num_env

        if heatmaps:
            self.subproc_env = make_record_write_subproc_env(env_id=env_id, num_env=num_env)
        else:
            self.subproc_env = make_subproc_env(env_id=env_id, num_env=num_env, width=64, height=64)

        self.action_space = self.subproc_env.action_space
        self.observation_space = self.subproc_env.observation_space

        self.loss_accumulators = {'rnn': 0, 'vae': 0}

        self.t_minus_1_dones = [True for _ in range(self.num_envs)]

        self.t_obs = self.subproc_env.reset() / 255.0
        self.t_dones = [False for _ in range(self.num_envs)]

        self.train_states = None

        self.current_train_iteration = 0

        # if heatmaps:
        #     self.set_heatmap_record_write_to_current_iteration()
        #     self.old_prefix = self.get_current_heatmap_record_prefix()

        self.minibatch_observations = []
        self.minibatch_actions = []
        self.minibatch_dones = []

        with self.sess.as_default():
            self.summary_writer = tf.summary.FileWriter(working_dir)
            self.vae = VAE(latent_dim=latent_dim, working_dir=working_dir, summary_writer=self.summary_writer)
            self.state_rnn = StateRNN(train_sequence_length=train_seq_length, latent_dim=latent_dim, action_dim=self.subproc_env.action_space.n,
                                      working_dir=working_dir, summary_writer=self.summary_writer)

    def step(self, actions):
        t_plus_1_obs, _, t_plus_1_dones, _ = self.subproc_env.step(actions)
        t_plus_1_obs = t_plus_1_obs / 255.0

        encoded_current_obs = self.vae.encode_frames(self.t_obs)
        prediction_from_current_obs = self.state_rnn.predict_on_frames_retain_state(z_codes=encoded_current_obs,
                                                                                    actions=actions,
                                                                                    states_mask=1 - np.asarray(self.t_minus_1_dones))
        generated_frames = None
        if self.return_generated_frames_in_info:
            losses, generated_frames = self.vae.get_loss_for_decoded_frames(z_codes=prediction_from_current_obs,
                                                                            target_frames=t_plus_1_obs,
                                                                            return_generated_frames=True)
        else:
            losses = self.vae.get_loss_for_decoded_frames(z_codes=prediction_from_current_obs,
                                                          target_frames=t_plus_1_obs)

        self.minibatch_observations.append(self.t_obs)
        self.minibatch_actions.append(convert_to_one_hot(actions, self.action_space.n))
        self.minibatch_dones.append(self.t_minus_1_dones)

        if len(self.minibatch_actions) >= self.train_seq_length:
            self.minibatch_observations.append(t_plus_1_obs)
            self.minibatch_dones.append(self.t_dones)

            if self.do_train:
                self.train()

            self.minibatch_observations = []
            self.minibatch_actions = []
            self.minibatch_dones = []

        # Move iteration forward
        self.t_minus_1_dones = self.t_dones
        self.t_obs = t_plus_1_obs
        self.t_dones = t_plus_1_dones

        return t_plus_1_obs, losses, t_plus_1_dones, {'generated_frames': generated_frames}

    def reset(self):
        logger.info("RESET WAS CALLED")
        self.state_rnn.reset_saved_state()

        self.t_minus_1_dones = [True for _ in range(self.num_envs)]

        self.t_obs = self.subproc_env.reset() / 255.0
        self.t_dones = [False for _ in range(self.num_envs)]

        self.train_states = None

        self.minibatch_observations = []
        self.minibatch_actions = []
        self.minibatch_dones = []

        return self.t_obs

    def render_actual_frames(self):
        return self.subproc_env.render()

    def get_current_heatmap_record_prefix(self):
        return "it{}".format(self.current_train_iteration)

    def set_heatmap_record_write_to_current_iteration(self):
        self.subproc_env.set_record_write(write_dir=os.path.join(self.working_dir, HEATMAP_FOLDER_NAME),
                                          prefix=self.get_current_heatmap_record_prefix())

    def train(self):

        assert len(self.minibatch_observations) == len(self.minibatch_actions) + 1
        assert len(self.minibatch_observations) == len(self.minibatch_dones)
        assert len(self.minibatch_actions) == self.train_seq_length

        self.minibatch_observations = np.asarray(self.minibatch_observations).swapaxes(1, 0)
        self.minibatch_actions = np.asarray(self.minibatch_actions).swapaxes(1, 0)
        self.minibatch_dones = np.asarray(self.minibatch_dones).swapaxes(1, 0)

        assert self.minibatch_observations.shape[0] == self.num_envs
        assert self.minibatch_dones.shape[0] == self.num_envs
        assert self.minibatch_actions.shape[0] == self.num_envs

        mask = 1 - self.minibatch_dones

        mb_obs_shape = self.minibatch_observations.shape

        vae_loss, kl_divergence, reconstruction_loss, vae_step, encoded_obs = self.vae.train_on_batch(
            frames_batch=np.reshape(
                self.minibatch_observations,
                newshape=[mb_obs_shape[0] * mb_obs_shape[1], *mb_obs_shape[2:]]
            )
        )

        encoded_obs = np.reshape(encoded_obs, newshape=[mb_obs_shape[0], mb_obs_shape[1], self.state_rnn.latent_dim])

        rnn_loss, states_out, rnn_step = self.state_rnn.train_on_batch(
            input_code_sequence_batch=encoded_obs[:, :-1],
            target_code_sequence_batch=encoded_obs[:, 1:],
            states_mask_sequence_batch=mask,
            input_action_sequence_batch=self.minibatch_actions,
            states_batch=self.train_states)

        self.loss_accumulators['rnn'] += rnn_loss
        self.loss_accumulators['vae'] += vae_loss

        if vae_step == 1 or vae_step % 100 == 0:
            logger.debug("VAE step {} loss {} - RNN step {} loss {}".format(
                vae_step,
                self.loss_accumulators['vae'] / len(self.minibatch_actions),
                rnn_step,
                self.loss_accumulators['rnn'] / len(self.minibatch_actions),
            ))

            # reset accumulators to 0
            self.loss_accumulators = dict.fromkeys(self.loss_accumulators, 0)

        self.train_states = states_out

        if vae_step == 1 or vae_step % 2000 == 0:

            self.vae.save_model()
            self.state_rnn.save_model()

            if self.validation_data_dir is not None:
                avg_val_loss = validate_vae_state_rnn_pair_on_tf_records(
                    data_dir=self.validation_data_dir,
                    vae=self.vae,
                    state_rnn=self.state_rnn,
                    sess=self.sess,
                    allowed_action_space=self.action_space
                )
                logger.info("Avg val loss: {}".format(avg_val_loss))
