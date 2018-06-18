from directed_exploration.de_logging import init_logging
from directed_exploration.simulator_train_env import SimulatorTrainEnv

import datetime
import os
import tensorflow as tf
import numpy as np
import gym_boxpush
import cv2


if __name__ == '__main__':

    working_dir = 'itexplore_a2c_second_run'

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
                            max_train_seq_length=100,
                            sequences_per_epoch=num_env*5,
                            validation_data_dir='/media/jb/m2/boxpushmaze_validation_rollouts/',
                            heatmaps=True,
                            sess=sess)
    sim.reset()

    '''
    Run A2C on sim
    '''
    from baselines.a2c.a2c import learn
    from baselines.ppo2.policies import LstmPolicy

    learn(policy=LstmPolicy, env=sim, seed=42, total_timesteps=int(int(10e6) * 1.1), lrschedule='constant')


    '''
    Random actions on sim
    '''
    # while True:
    #     sim.step([sim.action_space.sample() for _ in range(num_env)])


    '''
    Control and debug sim
    '''
    # action = [0]
    #
    # from pyglet.window import key
    #
    # def key_press(k, mod):
    #     if k == key.RIGHT: action[0] = 1
    #     if k == key.UP:    action[0] = 2
    #     if k == key.DOWN:  action[0] = 3
    #     if k == key.LEFT:  action[0] = 4
    #
    #
    # def key_release(k, mod):
    #     if k == key.RIGHT and action[0] == 1: action[0] = 0
    #     if k == key.UP and action[0] == 2: action[0] = 0
    #     if k == key.DOWN and action[0] == 3: action[0] = 0
    #     if k == key.LEFT and action[0] == 4: action[0] = 0
    #
    # import pyglet
    #
    # def update(dt):
    #     sim.render_actual_frames()
    #     observation, reward, done, info = sim.step([action[0] for _ in range(num_env)])
    #     print("reward: {}".format(reward))
    #     for i, obs in enumerate(observation):
    #         cv2.imshow(str(i), obs[:,:,::-1])
    #     if info['generated_frames'] is not None:
    #         for i, frame in enumerate(info['generated_frames']):
    #             cv2.imshow('generated {}'.format(i), frame[:, :, ::-1])
    #     cv2.waitKey(1)
    #
    # window = pyglet.window.Window()
    # window.on_key_press = key_press
    # window.on_key_release = key_release
    # pyglet.clock.schedule_interval(update, 1 / 30.0)
    #
    # pyglet.app.run()

