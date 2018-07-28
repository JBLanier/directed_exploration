from directed_exploration.vae import VAE
from directed_exploration.state_rnn import StateRNN
from directed_exploration.old_state_rnn import StateRNNOld
from directed_exploration.validation import validate_vae_state_rnn_pair_on_tf_records
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)


def dictionary_diff(dict1, dict2):
    return_dict = {}
    for k in dict1.keys():
        if np.any(dict1[k] != dict2[k]):
            return_dict[k] = (dict1[k], dict2[k])
    return return_dict


class TestRNNSim:
    def __init__(self, latent_dim=1, action_dim=3, working_dir=None, sess=None, graph=None,
                 summary_writer=None):

        self.state_rnn = StateRNN(
                                  latent_dim,
                                  action_dim,
                                  working_dir,
                                  sess,
                                  graph,
                                  summary_writer)

        # self.state_rnn.save_model()
        #
        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allow_growth = True
        # other_graph = tf.Graph()
        # other_sess = tf.Session(config=config, graph=other_graph)
        #
        # other_summary_writer = tf.summary.FileWriter(working_dir)
        # self.state_rnn2 = StateRNNOld(latent_dim,
        #                           action_dim,
        #                           working_dir,
        #                           other_sess,
        #                           other_sess.graph,
        #                               other_summary_writer)
        #
        # logger.info("\n\n\n\nInitial difference in variables: {}\n\n\n\n".format(dictionary_diff(self.state_rnn.return_all_variables_with_values_in_dict(),
        #                                                                          self.state_rnn2.return_all_variables_with_values_in_dict())))

    def save_model(self):
        self.state_rnn.save_model()

    def get_current_step(self):
        return self.state_rnn.sess.run([self.state_rnn.local_step])[0]

    def predict_on_batch(self, t_obs, t_actions, t_dones, t_states=None, actual_t_plus_one_obs=None, return_t_plus_one_predictions=True):

        encoded_current_obs = t_obs[:, 0, 0, :1]
        t_plus_1_code_predictions, t_plus_1_states = self.state_rnn.predict_on_frames(z_codes=encoded_current_obs,
                                                                                      actions=t_actions,
                                                                                      states_mask=1 - np.asarray(t_dones),
                                                                                      states_in=t_states)
        # other_predictions, other_states = self.state_rnn2.predict_on_frames(z_codes=encoded_current_obs,
        #                                                                               actions=t_actions,
        #                                                                               states_mask=1 - np.asarray(t_dones),
        #                                                                               states_in=t_states)


        # if not(np.all(np.equal(t_plus_1_code_predictions, other_predictions))):
        #     print("different predictions are: {} {}".format(t_plus_1_code_predictions, other_predictions))

        # print("comparing states: {}".format(np.equal(t_plus_1_states, other_states)))
        #
        # print("\n\n\nExiting Now\n\n\n")
        # exit(0)

        return_vals = []

        if return_t_plus_one_predictions:
            return_vals.append(np.ones((len(t_plus_1_code_predictions), 64, 64, 64) * t_plus_1_code_predictions))

        if actual_t_plus_one_obs is not None:
            return_vals.append([(predicted[0] - actual)**2 for predicted, actual in zip(t_plus_1_code_predictions, actual_t_plus_one_obs[:,0,0,0])])

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

        encoded_obs = obs_sequence_batch[:, :, 0, 0, :1]

        # todo: print output inputs, predictions, loss, and state to see where the problem lies

        # original_variables = self.state_rnn.return_all_variables_with_values_in_dict()

        rnn_loss, states_out, rnn_step, predictions = self.state_rnn.train_on_batch(
            input_code_sequence_batch=encoded_obs[:, :-1],
            target_code_sequence_batch=encoded_obs[:, 1:],
            states_mask_sequence_batch=mask,
            input_action_sequence_batch=action_sequence_batch,
            states_batch=initial_states_batch)

        # rnn_loss2, states_out2, rnn_step2, predictions2 = self.state_rnn2.train_on_batch(
        #     input_code_sequence_batch=encoded_obs[:, :-1],
        #     target_code_sequence_batch=encoded_obs[:, 1:],
        #     states_mask_sequence_batch=mask,
        #     input_action_sequence_batch=action_sequence_batch,
        #     states_batch=initial_states_batch)
        #
        # print("comparing predictions: {}".format(np.equal(predictions, predictions2)))
        # #
        # #
        # print("comparing loss: {}".format(np.equal(rnn_loss2, rnn_loss)))

        # logger.info("\n\n\n\nDifference in variables: {}\n\n\n\n".format(dictionary_diff(self.state_rnn.return_all_variables_with_values_in_dict(),
        #                                                                          self.state_rnn2.return_all_variables_with_values_in_dict())))

        # logger.info("change in variables: {}".format(dictionary_diff(self.state_rnn.return_all_variables_with_values_in_dict(), original_variables)))


        return rnn_step, {'rnn loss': rnn_loss}, states_out

    def validate(self, validation_data_dir, allowed_action_space=None):

        return {'pass': 0}