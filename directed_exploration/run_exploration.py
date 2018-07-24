from directed_exploration.logging_ops import init_logging
from directed_exploration.subproc_env_sim_wrapper import SubprocEnvSimWrapper
from directed_exploration.state_rnn import StateRNN
from directed_exploration.sep_vae_rnn_sim import SeparateVaeRnnSim
from directed_exploration.utils.env_util import make_record_write_subproc_env, make_subproc_env


import datetime
import os
import tensorflow as tf
import numpy as np
import gym_boxpush
import cv2


if __name__ == '__main__':

    num_env = 48
    # env_id = 'BreakoutDeterministic-v4'
    env_id = 'boxpushsimple-v0'

    # Heatmaps only work with boxpush environments
    heatmaps = True

    # working_dir = 'itexplore_20180625211256'
    working_dir = None

    if working_dir is None:
        date_identifier = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        working_dir = './itexplore_{}'.format(date_identifier)

    init_logging(logfile=os.path.join(working_dir, 'events.log'),
                 redirect_stdout=True,
                 redirect_stderr=True,
                 handle_tensorflow=True)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(working_dir)

    if heatmaps:
        env = make_record_write_subproc_env(env_id=env_id, num_env=num_env)
    else:
        env = make_subproc_env(env_id=env_id, num_env=num_env, width=64, height=64)

    sim = SeparateVaeRnnSim(latent_dim=4, action_dim=env.action_space.n, working_dir=working_dir, sess=sess, summary_writer=summary_writer)

    sim_env = SubprocEnvSimWrapper(sim=sim,
                                   subproc_env=env,
                                   working_dir=working_dir,
                               train_seq_length=5,
                               sequences_per_epoch=num_env*5,
                               validation_data_dir='/mnt/m2/boxpushsimple_val_rollouts_v2',
                               heatmaps=heatmaps,
                               do_train=True,
                                   summary_writer=summary_writer)

    '''
    Run A2C on sim
    '''
    from baselines.a2c.a2c import learn
    from baselines.ppo2.policies import LstmPolicy

    learn(policy=LstmPolicy, env=sim_env, seed=42, total_timesteps=int(int(10e6) * 1.1), lrschedule='constant', ent_coef=0.4)


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
    #     print("down")
    #     if k == key.RIGHT: action[0] = 2
    #     if k == key.UP:    action[0] = 1
    #     if k == key.LEFT:  action[0] = 3
    #
    #
    # def key_release(k, mod):
    #     print("up")
    #     if k == key.RIGHT and action[0] == 2: action[0] = 0
    #     if k == key.UP and action[0] == 1: action[0] = 0
    #     if k == key.LEFT and action[0] == 3: action[0] = 0
    #
    # import pyglet
    #
    #
    #
    # window = pyglet.window.Window()
    #
    # def update(dt):
    #     if keyboard.is_pressed('q'):
    #         print("q pressed")
    #     if keyboard.is_pressed('t'):
    #         print("t pressed")
    #     sim.render_actual_frames()
    #     observation, reward, done, info = sim.step(np.asarray([action[0] for _ in range(num_env)]))
    #     # print("reward: {}".format(reward))
    #     for i, obs in enumerate(observation):
    #         cv2.imshow(str(i), obs[:,:,::-1])
    #     if info['generated_frames'] is not None:
    #         for i, frame in enumerate(info['generated_frames']):
    #             cv2.imshow('generated {}'.format(i), frame[:, :, ::-1])
    #     cv2.waitKey(1)
    #
    # # window.on_key_press = key_press
    # # window.on_key_release = key_release
    # pyglet.clock.schedule_interval(update, 1 / 30.0)
    #
    # pyglet.app.run()

