from directed_exploration.frame_predict_rnn.frame_predict_rnn import FramePredictRNN
from directed_exploration.validation import validate_full_rnn_on_tf_records

import numpy as np
import logging

logger = logging.getLogger(__name__)

class FramePredictRNNSim:
    def __init__(self, observation_space, action_dim=5, working_dir=None, sess=None, graph=None,
                 summary_writer=None):

        self.rnn = FramePredictRNN(observation_space,
                                     action_dim,
                                     working_dir,
                                     sess,
                                     graph,
                                     summary_writer)


    def save_model(self):
        self.rnn.save_model()

    def get_current_step(self):
        return self.rnn.sess.run([self.rnn.local_step])[0]

    def predict_on_batch(self, t_obs, t_actions, t_dones, t_states=None, actual_t_plus_one_obs=None, t_plus_1_dones=None, return_t_plus_one_predictions=True):

        valid_prediction_mask = None
        if t_plus_1_dones is not None:
            valid_prediction_mask = 1 - np.asarray(t_plus_1_dones)

        t_plus_1_predictions, t_plus_1_states, losses = self.rnn.predict_on_frame_batch_with_loss(frames=t_obs,
                                                                                           actions=t_actions,
                                                                                           states_mask=1 - np.asarray(t_dones),
                                                                                           states_in=t_states,
                                                                                            target_predictions=actual_t_plus_one_obs,
                                                                                                  valid_prediction_mask=valid_prediction_mask)

        return_vals = []

        if return_t_plus_one_predictions:
            return_vals.append(t_plus_1_predictions)

        if actual_t_plus_one_obs is not None and t_plus_1_dones is not None:
            return_vals.append(losses)

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

        rnn_loss, states_out, rnn_step = self.rnn.train_on_batch(
            input_frame_sequence_batch=obs_sequence_batch[:, :-1],
            target_frame_sequence_batch=obs_sequence_batch[:, 1:],
            states_mask_sequence_batch=mask,
            input_action_sequence_batch=action_sequence_batch,
            states_batch=initial_states_batch)

        return rnn_step, {'full rnn loss': rnn_loss}, states_out

    def validate(self, validation_data_dir, allowed_action_space=None):

        avg_val_loss = validate_full_rnn_on_tf_records(
            data_dir=validation_data_dir,
            rnn=self.rnn,
            sess=self.rnn.sess,
            allowed_action_space=allowed_action_space
        )

        return {'avg_val_loss': avg_val_loss}