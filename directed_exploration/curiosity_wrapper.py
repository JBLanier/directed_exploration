from directed_exploration.utils.heatmap_gen import generate_boxpush_heatmap_from_npy_records

import tensorflow as tf
import numpy as np
import os

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

class CuriosityWrapper:
    def __init__(self,
                 working_dir,
                 sim,
                 train_seq_length,
                 subproc_env,
                 extrinsic_reward_coefficient,
                 intrinsic_reward_coefficient,
                 heatmaps=False,
                 validation_data_dir=None,
                 return_generated_frames_in_info=False,
                 do_train=True,
                 summary_writer=None):

        self.sim = sim

        self.working_dir = working_dir
        self.validation_data_dir = validation_data_dir
        self.return_generated_frames_in_info = return_generated_frames_in_info
        self.do_train = do_train
        self.train_seq_length = train_seq_length
        self.heatmaps = heatmaps
        self.subproc_env = subproc_env
        self.num_envs = self.subproc_env.num_envs

        self.extrinsic_reward_coefficient = extrinsic_reward_coefficient
        self.intrinsic_reward_coefficient = intrinsic_reward_coefficient

        self.action_space = self.subproc_env.action_space
        self.observation_space = self.subproc_env.observation_space

        self.loss_accumulators = None

        self.t_obs = self.subproc_env.reset() / 255.0
        self.t_dones = [True for _ in range(self.num_envs)]

        self.train_states = None
        self.t_states = None

        self.current_step = 1

        self.summary_writer = summary_writer

        if heatmaps:
            self.set_heatmap_record_write_to_current_step()
            self.old_prefix = self.get_current_heatmap_record_prefix()

        self.minibatch_observations = []
        self.minibatch_actions = []
        self.minibatch_dones = []

    def step(self, actions):
        t_plus_1_obs, extrinsic_rewards, t_plus_1_dones, _ = self.subproc_env.step(actions)
        # self.subproc_env.render()
        # print(t_plus_1_obs)
        t_plus_1_obs = t_plus_1_obs / 255.0

        predict_vals = self.sim.predict_on_batch(
            t_obs=self.t_obs,
            t_actions=actions,
            t_states=self.t_states,
            t_dones=self.t_dones,
            t_plus_1_dones=t_plus_1_dones,
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
        self.minibatch_dones.append(self.t_dones)

        # print("batch: {}".format(np.asarray(self.minibatch_observations)[:,:,0,0,0]))

        if len(self.minibatch_actions) >= self.train_seq_length:
            self.minibatch_observations.append(t_plus_1_obs)
            self.minibatch_dones.append(t_plus_1_dones)

            if self.do_train:
                self.train()

        if self.heatmaps and self.current_step % 20000 == 0:
            self.set_heatmap_record_write_to_current_step()
            logger.debug("Creating Heatmap...")
            heatmap_save_location = generate_boxpush_heatmap_from_npy_records(
                directory=os.path.join(self.working_dir, HEATMAP_FOLDER_NAME),
                file_prefix=self.old_prefix,
                delete_records=True)
            logger.debug("Heatmap saved to {}".format(heatmap_save_location))
            self.old_prefix = self.get_current_heatmap_record_prefix()

        # Move iteration forward
        self.t_obs = t_plus_1_obs
        self.t_dones = t_plus_1_dones
        self.t_states = t_plus_1_states

        self.current_step += 1

        out_rewards = self.extrinsic_reward_coefficient * extrinsic_rewards + self.intrinsic_reward_coefficient * losses

        return np.copy(t_plus_1_obs), out_rewards, t_plus_1_dones, {'generated_frames': t_plus_1_predictions}

    def reset(self):
        logger.info("RESET WAS CALLED")

        self.t_obs = self.subproc_env.reset() / 255.0
        self.t_dones = [True for _ in range(self.num_envs)]

        self.t_states = None
        self.train_states = None

        self.minibatch_observations = []
        self.minibatch_actions = []
        self.minibatch_dones = []

        return self.t_obs

    def render_actual_frames(self):
        return self.subproc_env.render()

    def get_current_heatmap_record_prefix(self):
        return "env_step{}".format(self.current_step)

    def set_heatmap_record_write_to_current_step(self):
        self.subproc_env.set_record_write(write_dir=os.path.join(self.working_dir, HEATMAP_FOLDER_NAME),
                                          prefix=self.get_current_heatmap_record_prefix())

    def train(self):
        # print("minibatch_obs shape: {}".format(np.asarray(self.minibatch_observations).shape))
        # print("minibatch_obs: {}".format(np.asarray(self.minibatch_observations)[:,:,0,0,0]))
        step, losses, states_out = self.sim.train_on_batch(self.minibatch_observations, self.minibatch_actions, self.minibatch_dones, self.train_states)
        # print(losses)

        if self.loss_accumulators is None:
            self.loss_accumulators = losses
        else:
            for key in self.loss_accumulators.keys():
                self.loss_accumulators[key] += losses[key]

        print_loss_every = 100
        if step == 1 or step % print_loss_every == 0:
            for key in self.loss_accumulators.keys():
                self.loss_accumulators[key] /= print_loss_every
            logger.info("sim train step {} - {}".format(step, pretty_dict_keys_with_values(self.loss_accumulators)))

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

                summary = tf.Summary()
                for key in val_losses.keys():
                    summary.value.add(tag=key, simple_value=val_losses[key])
                self.summary_writer.add_summary(summary, self.sim.get_current_step())
                self.summary_writer.flush()

                logger.info(pretty_dict_keys_with_values(val_losses))



