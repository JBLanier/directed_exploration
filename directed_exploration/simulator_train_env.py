from directed_exploration.utils.env_util import make_record_write_subproc_env
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
from functools import reduce

import logging

logger = logging.getLogger(__name__)

HEATMAP_FOLDER_NAME = 'heatmap_records'


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

        self.t_minus_1_dones = [True for _ in range(self.num_envs)]

        self.t_obs = self.subproc_env.reset() / 255.0
        self.t_dones = [False for _ in range(self.num_envs)]

        self.current_train_iteration = 0
        self.num_envs=num_env

        self.subproc_env = make_record_write_subproc_env(env_id=env_id, num_env=num_env)
        self.action_space = self.subproc_env.action_space
        self.observation_space = self.subproc_env.observation_space

        if heatmaps:
            self.set_heatmap_record_write_to_current_iteration()
            self.old_prefix = self.get_current_heatmap_record_prefix()

        self.minibatch_observations = []
        self.minibatch_actions = []
        self.minibatch_dones = []

        with self.sess.as_default():
            self.summary_writer = tf.summary.FileWriter(working_dir)
            self.vae = VAE(latent_dim=latent_dim, working_dir=working_dir, summary_writer=self.summary_writer)
            self.state_rnn = StateRNN(latent_dim=latent_dim, action_dim=self.subproc_env.action_space.n,
                                      working_dir=working_dir, summary_writer=self.summary_writer)

    def step(self, actions):
        t_plus_1_obs, _, t_plus_1_dones, _ = self.subproc_env.step(actions)
        t_plus_1_obs = t_plus_1_obs / 255.0

        encoded_current_obs = self.vae.encode_frames(self.t_obs)
        prediction_from_current_obs = self.state_rnn.predict_on_frames_retain_state(z_codes=encoded_current_obs,
                                                                                    actions=actions,
                                                                                    states_mask=self.t_minus_1_dones)
        generated_frames = None
        if self.return_generated_frames_in_info:
            losses, generated_frames = self.vae.get_loss_for_decoded_frames(z_codes=prediction_from_current_obs,
                                                                            target_frames=t_plus_1_obs/255.0,
                                                                            return_generated_frames=True)
        else:
            losses = self.vae.get_loss_for_decoded_frames(z_codes=prediction_from_current_obs,
                                                          target_frames=t_plus_1_obs)

        self.minibatch_observations.append(self.t_obs)
        self.minibatch_actions.append(actions)
        self.minibatch_dones.append(self.t_minus_1_dones)

        if len(actions) > self.train_seq_length:
            self.minibatch_observations.append(t_plus_1_obs)
            self.minibatch_dones.append(self.t_dones)

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
        print("reset shouldn't ever need to be called")
        raise NotImplementedError
        # self.state_rnn.reset_saved_state()
        # obs = self.subproc_env.reset()
        # self.current_obs = obs / 255.0
        # self.train_states = None
        # self.dones = [False for _ in range(self.num_envs)]
        # return obs

    def render_actual_frames(self):
        return self.subproc_env.render()

    def get_current_heatmap_record_prefix(self):
        return "it{}".format(self.current_train_iteration)

    def set_heatmap_record_write_to_current_iteration(self):
        self.subproc_env.set_record_write(write_dir=os.path.join(self.working_dir, HEATMAP_FOLDER_NAME),
                                          prefix=self.get_current_heatmap_record_prefix())

    def train(self):
        # for i, s in enumerate(self.vae_deque):
        #     print("vae_deque element {}:\n{}".format(i, s[1]))

        logger.debug("Training VAE on rollouts...")
        self.sess.run(self.vae_input_fn_init_op)
        self.vae.train_on_iterator(self.vae_input_fn_iter)

        logger.debug("Formatting rollouts for State RNN...")
        sdo.convert_vae_deque_to_state_rnn_deque(self.vae, self.vae_deque,
                                                 self.state_rnn_deque,
                                                 self.train_seq_length)

        # it_num_of_sequences_written = reduce(lambda acc, episode: acc + len(episode), episode_sequence_lengths, 0)
        # it_total_frames_written = reduce(lambda acc, episode: acc + sum(episode), episode_sequence_lengths, 0)

        # logger.debug("Converted {} episodes to {} sequences ({} frames).".format(len(episode_sequence_lengths),
        #                                                                          it_num_of_sequences_written,
        #                                                                          it_total_frames_written))

        logger.debug("Training State RNN on rollouts...")
        self.sess.run(self.state_rnn_input_fn_init_op)
        self.state_rnn.train_on_iterator(self.state_rnn_input_fn_iter)

        self.vae_deque.clear()
        self.state_rnn_deque.clear()

        if self.current_train_iteration == 0 or self.current_train_iteration % 50 == 0:

            if self.validation_data_dir:
                logger.debug("Validating simulator on trajectories from {}".format(self.validation_data_dir))
                validation_loss = validate_vae_state_rnn_pair_on_tf_records(data_dir=self.validation_data_dir,
                                                                            vae=self.vae, state_rnn=self.state_rnn,
                                                                            sess=self.sess,
                                                                            allowed_action_space=self.action_space)
                logger.debug('Total Validation Loss = {}'.format(validation_loss))
                val_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='simulator_val_loss',
                                                                      simple_value=validation_loss)])
                self.summary_writer.add_summary(val_loss_summary, global_step=self.current_train_iteration)

            self.vae.save_model()
            self.state_rnn.save_model()
