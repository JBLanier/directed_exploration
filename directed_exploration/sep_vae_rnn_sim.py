from directed_exploration.vae import VAE
from directed_exploration.state_rnn import StateRNN
from directed_exploration.validation import validate_vae_state_rnn_pair_on_tf_records

import numpy as np


class SeparateVaeRnnSim:
    def __init__(self, latent_dim=4, action_dim=5, working_dir=None, sess=None, graph=None,
                 summary_writer=None):

        self.state_rnn = StateRNN(latent_dim,
                                  action_dim,
                                  working_dir,
                                  sess,
                                  graph,
                                  summary_writer)

        self.vae = VAE(latent_dim,
                       working_dir,
                       sess,
                       graph,
                       summary_writer)

    def save_model(self):
        self.vae.save_model()
        self.state_rnn.save_model()

    def get_current_step(self):
        return self.vae.sess.run([self.vae.local_step])[0]

    def predict_on_batch(self, t_obs, t_actions, t_minus_1_dones, t_states=None, actual_t_plus_one_obs=None, return_t_plus_one_predictions=True):

        encoded_current_obs = self.vae.encode_frames(t_obs)
        t_plus_1_code_predictions, t_plus_1_states = self.state_rnn.predict_on_frames(z_codes=encoded_current_obs,
                                                                                      actions=t_actions,
                                                                                      states_mask=1 - np.asarray(t_minus_1_dones),
                                                                                      states_in=t_states)

        tensors_to_evaluate = []
        feed_dict = {self.vae.z_encoded: t_plus_1_code_predictions, self.vae.x: actual_t_plus_one_obs}

        if return_t_plus_one_predictions:
            tensors_to_evaluate.append(self.vae.decoded)

        if actual_t_plus_one_obs is not None:
            tensors_to_evaluate.append(self.vae.per_frame_reconstruction_loss)

        return_vals = self.vae.sess.run(tensors_to_evaluate, feed_dict)

        return_vals.append(t_plus_1_states)

        return return_vals

    def train_on_batch(self, obs_sequence_batch, action_sequence_batch, dones_sequence_batch, initial_states_batch):

        assert len(obs_sequence_batch) == len(action_sequence_batch) + 1
        assert len(obs_sequence_batch) == len(dones_sequence_batch)

        obs_sequence_batch = np.asarray(obs_sequence_batch).swapaxes(1, 0)
        action_sequence_batch = np.asarray(action_sequence_batch).swapaxes(1, 0)
        dones_sequence_batch = np.asarray(dones_sequence_batch).swapaxes(1, 0)

        assert obs_sequence_batch.shape[0] == action_sequence_batch.shape[0]
        assert obs_sequence_batch.shape[0] == dones_sequence_batch.shape[0]

        mask = 1 - dones_sequence_batch

        mb_obs_shape = obs_sequence_batch.shape

        vae_loss, kl_divergence, reconstruction_loss, vae_step, encoded_obs = self.vae.train_on_batch(
            frames_batch=np.reshape(
                obs_sequence_batch,
                newshape=[mb_obs_shape[0] * mb_obs_shape[1], *mb_obs_shape[2:]]
            )
        )

        encoded_obs = np.reshape(encoded_obs, newshape=[mb_obs_shape[0], mb_obs_shape[1], self.state_rnn.latent_dim])

        rnn_loss, states_out, rnn_step = self.state_rnn.train_on_batch(
            input_code_sequence_batch=encoded_obs[:, :-1],
            target_code_sequence_batch=encoded_obs[:, 1:],
            states_mask_sequence_batch=mask,
            input_action_sequence_batch=action_sequence_batch,
            states_batch=initial_states_batch)

        return vae_step, {'rnn loss': rnn_loss, 'vae loss': vae_loss}, states_out

    def validate(self, validation_data_dir, allowed_action_space=None):

        avg_val_loss = validate_vae_state_rnn_pair_on_tf_records(
            data_dir=validation_data_dir,
            vae=self.vae,
            state_rnn=self.state_rnn,
            sess=self.vae.sess,
            allowed_action_space=allowed_action_space
        )

        return {'Average Validation Loss': avg_val_loss}