from directed_exploration.logging_ops import init_logging, get_logger
from directed_exploration.curiosity_wrapper import CuriosityWrapper
from directed_exploration.frame_predict_rnn.frame_predict_rnn_sim import FramePredictRNNSim
from directed_exploration.utils.env_util import make_record_write_subproc_env, make_subproc_env
from directed_exploration.utils.data_util import none_or_str, str_as_bool, convert_scientific_str_to_int, pretty_print_dict, ensure_dir

import datetime
import os
import tensorflow as tf
import argparse
import cv2
import numpy as np
import logging

# from directed_exploration.test_rnn_sim import TestRNNSim
# from gym.envs.registration import register
#
# register(
#     id='colorchange-v0',
#     entry_point='directed_exploration.gym_colorchange:GymBrightnessChange',
# )


def main(args):
    curiosity_source = FramePredictRNNSim

    if args.working_dir is None:
        args.working_dir = 'runs/A2C_{}_'.format(curiosity_source.__name__)
        if args.new_dir_note:
            args.working_dir += '{}_'.format(args.new_dir_note)
        args.working_dir += datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    ensure_dir(args.working_dir)

    if not args.demo_debug:
        a2c_dir = os.path.join(args.working_dir, 'a2c')
        ensure_dir(a2c_dir)
    else:
        a2c_dir = None

    if args.demo_debug:
        args.num_env = 1

    init_logging(
        logfile=os.path.join(args.working_dir, 'events.log'),
        redirect_stdout=True,
        redirect_stderr=True,
        external_packages_to_capture=['tensorflow']
    )

    logger = get_logger()

    logger.info(pretty_print_dict(args.__dict__, 'Arguments:'))

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(args.working_dir)

    if args.create_heatmaps:
        env = make_record_write_subproc_env(env_id=args.env_id, num_env=args.num_env)
    else:
        env = make_subproc_env(env_id=args.env_id, num_env=args.num_env, width=args.frame_size[0],
                               height=args.frame_size[1], seed=42, monitor_to_dir=a2c_dir)

    if args.intrinsic_reward_coefficient != 0:
        sim = curiosity_source(
            observation_space=env.observation_space, action_dim=env.action_space.n,
            working_dir=args.working_dir,
            sess=sess, summary_writer=summary_writer
        )

        env = CuriosityWrapper(
            sim=sim,
            subproc_env=env,
            working_dir=args.working_dir,
            extrinsic_reward_coefficient=args.extrinsic_reward_coefficient,
            intrinsic_reward_coefficient=args.intrinsic_reward_coefficient,
            train_seq_length=args.curiosity_train_sequence_length,
            validation_data_dir=args.validation_data_dir,
            heatmaps=args.create_heatmaps,
            do_train=not args.demo_debug,
            summary_writer=summary_writer,
            return_generated_frames_in_info=args.demo_debug
        )
    elif args.extrinsic_reward_coefficient != 1:
        logger.warning(
            "Since intrinsic_reward_coefficient is 0, "
            "extrinsic_reward_coefficient will be interpreted as 1 instead of {}".format(
                args.extrinsic_reward_coefficient
            )
        )

    if not args.demo_debug:
        '''
        Run A2C on sim
        '''
        from baselines.a2c.a2c import learn
        from baselines.ppo2.policies import LstmPolicy
        from baselines import logger as baselines_logger

        baselines_logger.configure(a2c_dir, format_strs=['stdout', 'csv', 'tensorboard'])

        learn(policy=LstmPolicy, env=env, seed=42, total_timesteps=int(args.num_timesteps), lrschedule='constant',
              ent_coef=args.a2c_entropy_coefficient)

    else:
        '''
        Control and debug sim
        '''
        action = [0]

        from pyglet.window import key

        def key_press(k, mod):
            print("down")
            if k == key.RIGHT: action[0] = 2
            if k == key.UP:    action[0] = 1
            if k == key.LEFT:  action[0] = 3
            # if k == key.DOWN:  action[0] = 4

        def key_release(k, mod):
            print("up")
            if k == key.RIGHT and action[0] == 2: action[0] = 0
            if k == key.UP and action[0] == 1: action[0] = 0
            if k == key.LEFT and action[0] == 3: action[0] = 0
            # if k == key.DOWN and action[0] == 4: action[0] = 0

        import pyglet

        window = pyglet.window.Window()

        def update(dt):
            # if keyboard.is_pressed('q'):
            #     print("q pressed")
            # if keyboard.is_pressed('t'):
            #     print("t pressed")
            observation, reward, done, info = env.step(np.asarray([action[0] for _ in range(args.num_env)]))
            # print("reward: {}".format(reward))
            for i, obs in enumerate(observation):
                cv2.imshow(str(i), obs[:, :, ::-1])
            if info['generated_frames'] is not None:
                for i, frame in enumerate(info['generated_frames']):
                    cv2.imshow('generated {}'.format(i), frame[:, :, ::-1])
            cv2.waitKey(1)

        window.on_key_press = key_press
        window.on_key_release = key_release
        pyglet.clock.schedule_interval(update, 1 / 30.0)

        pyglet.app.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-debug",
                        help="Instead of training, launches a single env to control and visualize.",
                        type=str_as_bool, required=True)
    parser.add_argument("--working-dir",
                        help="Dir to store all checkpoints, logs, etc. Pass \'None\' for new dir",
                        type=none_or_str)
    parser.add_argument("--new-dir-note",
                        help="If working-dir is not specified, this will be included in the new working dir created",
                        type=none_or_str)
    parser.add_argument("--validation-data-dir",
                        help="Directory with validation rollouts to test curiosity model accuracy. "
                             "Pass \'None\' for no validation",
                        type=none_or_str)
    parser.add_argument("--num-env",
                        help="Number of environment processes to work with simultaneously",
                        type=int, required=True)
    parser.add_argument("--env-id",
                        help="Gym Environment name to use",
                        type=str, required=True)
    parser.add_argument("--curiosity-train-sequence-length",
                        help="Training batch sequence length",
                        type=int, required=True)
    parser.add_argument("--create-heatmaps",
                        help="Create heatmap images of agent movement. Use only with GymBoxPush.",
                        type=str_as_bool, required=True)
    parser.add_argument("--frame-size",
                        help="Resize observation frames to this size",
                        type=int, nargs=2, required=True)
    parser.add_argument("--a2c-entropy-coefficient",
                        help='coefficient for rewarding entropy in the policy fn for A2C',
                        type=float, required=True)
    parser.add_argument("--extrinsic-reward-coefficient",
                        help='actual task reward is multiplied by this and combined with intrinsic reward',
                        type=float, required=True)
    parser.add_argument("--intrinsic-reward-coefficient",
                        help='curiosity signal is multiplied by this and combined with extrinsic reward',
                        type=float, required=True)
    parser.add_argument("--num-timesteps",
                        help='number of time steps to train on',
                        type=convert_scientific_str_to_int, required=True)

    args = parser.parse_args()

    main(args)
