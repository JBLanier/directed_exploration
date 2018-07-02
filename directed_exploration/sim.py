from abc import ABC, abstractmethod


class Sim(ABC):

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def get_current_step(self):
        pass

    @abstractmethod
    def predict_on_batch(self,  t_obs, t_actions, t_minus_1_dones, t_state=None, actual_t_plus_one_obs=None, return_t_plus_one_predictions=True):
        """Predicts next observations given batch of current observations, actions, and states

        Args:
          t-obs: Current observations to predict on.
          t-actions: Current actions to predict on.
          t-state: Current RNN state to predict on.
          return_t_plus_one_predictions: Optional; if False, predictions are not returned.
          actual_t_plus_one_obs: Optional; if provided, per-sample losses against this observations
            are computed and returned.

        Returns:
            NDArray of t_plus_one_predictions and/or NDArray of per-sample losses against actual values
            depending on the setting of return_t_plus_one_predictions and actual_t_plus_one_obs

        """

        pass

    @abstractmethod
    def train_on_batch(self, obs_sequence_batch, action_sequence_batch, dones_sequence_batch, initial_states_batch):
        """Trains on batch of sequential observation, actions, and initial states

        Args:
          obs_sequence_batch: Observations to train on.
            Will use obs_seq_batch[:-1] as inputs and obs_seq_batch[1:] as targets
          action_sequence_batch: Current actions to predict on.
            Actions are those taken while observing obs_seq_batch[:-1]
          initial_states_batch: initial RNN state to predict on.

        Returns:
            (train step, dictionary of loss values, rnn states out)

        """

        pass

    @abstractmethod
    def validate(self, validation_data_dir, allowed_action_space=None):

        pass