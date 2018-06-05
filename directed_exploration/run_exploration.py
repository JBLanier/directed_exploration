from directed_exploration.de_logging import init_logging
from directed_exploration.simulator_train_env import SimulatorTrainEnv

import datetime
import os
import tensorflow as tf
import numpy as np
import gym_boxpush
import cv2


if __name__ == '__main__':

    # working_dir = 'itexplore-colorchange-anticipator-on-boxpushsimple-sim'
    #
    # if working_dir:
    #     root_save_dir = working_dir
    # else:
    #     date_identifier = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    #     root_save_dir = './itexplore_{}'.format(date_identifier)
    #
    # init_logging(logfile=os.path.join(root_save_dir, 'events.log'),
    #              redirect_stdout=True,
    #              redirect_stderr=True,
    #              handle_tensorflow=True)
    #
    # do_a2c_exploration(env_id='boxpushsimple-colorchange-v0',
    #                       num_env=48,
    #                       num_iterations=200,
    #                       latent_dim=1,
    #                       working_dir=root_save_dir,
    #                       num_episodes_per_environment=2,
    #                       max_episode_length=2000,
    #                       max_sequence_length=200,
    #                       validation_data_dir='/media/jb/m2/boxpushsimple_validation_rollouts/',
    #                       do_random_policy=False)

    working_dir = None

    if working_dir:
        root_save_dir = working_dir
    else:
        date_identifier = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        root_save_dir = './itexplore_{}'.format(date_identifier)

    init_logging(logfile=os.path.join(root_save_dir, 'events.log'),
                 redirect_stdout=True,
                 redirect_stderr=True,
                 handle_tensorflow=True)

    num_env = 48

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sim = SimulatorTrainEnv(env_id='boxpushmaze-v0',
                            num_env=num_env,
                            latent_dim=4,
                            working_dir=root_save_dir,
                            max_train_seq_length=10,
                            sequences_per_epoch=num_env*5,
                            validation_data_dir='/media/jb/m2/boxpushmaze_validation_rollouts/',
                            heatmaps=True,
                            sess=sess)

    sim.reset()

    from baselines.a2c.a2c import learn
    from baselines.ppo2.policies import LstmPolicy

    policy_fn = LstmPolicy

    learn(policy_fn, sim, seed=42, total_timesteps=int(int(10e6) * 1.1), lrschedule='constant')
    # sim.close()

    # #
    # while True:
    #     sim.step([sim.action_space.sample() for _ in range(num_env)])
