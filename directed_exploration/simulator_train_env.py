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

    def __init__(self, working_dir, sess, latent_dim, max_train_seq_length, sequences_per_epoch, env_id, num_env,
                 heatmaps=False, validation_data_dir=None):

        self.working_dir = working_dir
        self.validation_data_dir = validation_data_dir
        self.sess = sess
        self.max_train_seq_length = max_train_seq_length
        self.sequences_per_epoch = sequences_per_epoch
        self.heatmaps = heatmaps
        self.current_obs = None
        self.current_train_iteration = 0
        self.vae_deque = deque()
        self.state_rnn_deque = deque()
        self.num_envs=num_env

        self.subproc_env = make_record_write_subproc_env(env_id=env_id, num_env=num_env)
        self.action_space = self.subproc_env.action_space
        self.observation_space = self.subproc_env.observation_space

        if heatmaps:
            self.set_heatmap_record_write_to_current_iteration()
            self.old_prefix = self.get_current_heatmap_record_prefix()

        self.current_sequence_frames = [np.empty(shape=(self.max_train_seq_length, 64, 64, 3), dtype=np.float32) for _ in range(num_env)]
        self.current_sequence_actions = [np.empty(shape=(self.max_train_seq_length, self.action_space.n), dtype=np.float32) for _ in range(num_env)]
        self.current_sequence_lengths = np.zeros(shape=num_env, dtype=np.uint32)

        with sess.as_default():
            self.summary_writer = tf.summary.FileWriter(working_dir)
            self.vae = VAE(latent_dim=latent_dim, working_dir=working_dir, summary_writer=self.summary_writer)
            self.state_rnn = StateRNN(latent_dim=latent_dim, action_dim=self.subproc_env.action_space.n,
                                      working_dir=working_dir, summary_writer=self.summary_writer)

            vae_input_fn = sdo.get_vae_deque_input_fn(train_deque=self.vae_deque, batch_size=256)

            state_rnn_input_fn = sdo.get_state_rnn_deque_input_fn(state_rnn_episodes_deque=self.state_rnn_deque,
                                                                  batch_size=64,
                                                                  latent_dim=latent_dim, num_actions=self.subproc_env.action_space.n,
                                                                  max_sequence_length=max_train_seq_length, num_epochs=1)

            with tf.variable_scope('input_functions'):
                self.vae_input_fn_iter, self.vae_input_fn_init_op = vae_input_fn()
                self.state_rnn_input_fn_iter, self.state_rnn_input_fn_init_op = state_rnn_input_fn()

    def step(self, actions):
        new_obs, _, dones, _ = self.subproc_env.step(actions)
        encoded_obs = self.vae.encode_frames(self.current_obs)
        predicted_encodings = self.state_rnn.predict_on_frames_retain_state(z_codes=encoded_obs, actions=actions)
        losses = self.vae.get_loss_for_decoded_frames(z_codes=predicted_encodings, target_frames=new_obs/255.0)
        self.state_rnn.selectively_reset_states(dones)

        # save frames and actions for training
        for env_index in range(self.subproc_env.num_envs):

            self.current_sequence_frames[env_index][self.current_sequence_lengths[env_index]] = self.current_obs[env_index]
            self.current_sequence_actions[env_index][self.current_sequence_lengths[env_index]] = convertToOneHot(actions[env_index], num_classes=self.action_space.n)

            self.current_sequence_lengths[env_index] += 1

            if dones[env_index] or self.current_sequence_lengths[env_index] >= self.max_train_seq_length:

                # trim env sequence to actual length
                self.current_sequence_frames[env_index].resize((self.current_sequence_lengths[env_index], 64, 64, 3))
                self.current_sequence_actions[env_index].resize((self.current_sequence_lengths[env_index], self.action_space.n))

                # add sequence to train deque
                self.vae_deque.append((self.current_sequence_frames[env_index], self.current_sequence_actions[env_index]))
                if len(self.vae_deque) >= self.sequences_per_epoch:
                    logger.info("Simulator Train Iteration {}".format(self.current_train_iteration))

                    self.train()

                    if self.heatmaps and self.current_train_iteration % 50 == 0:
                        self.current_train_iteration += 1

                        self.subproc_env.set_record_write(write_dir=os.path.join(self.working_dir, HEATMAP_FOLDER_NAME),
                                                          prefix=self.get_current_heatmap_record_prefix())
                        logger.debug("Creating Heatmap...")
                        heatmap_save_location = generate_boxpush_heatmap_from_npy_records(
                            directory=os.path.join(self.working_dir, HEATMAP_FOLDER_NAME),
                            file_prefix=self.old_prefix,
                            delete_records=True)
                        logger.debug("Heatmap saved to {}".format(heatmap_save_location))
                        self.old_prefix = self.get_current_heatmap_record_prefix()

                    else:
                        self.current_train_iteration += 1

                # reset sequence for env
                self.current_sequence_frames[env_index] = np.empty(shape=(self.max_train_seq_length, 64, 64, 3), dtype=np.float32)
                self.current_sequence_actions[env_index] = np.empty(shape=(self.max_train_seq_length, self.action_space.n), dtype=np.float32)
                self.current_sequence_lengths[env_index] = 0

        self.current_obs = new_obs / 255.0
        return new_obs, losses, dones, {}

    def reset(self):
        obs = self.subproc_env.reset()
        self.current_obs = obs / 255.0
        self.state_rnn.reset_state()
        return obs

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
                                                                            self.max_train_seq_length)

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
