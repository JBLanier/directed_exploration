import pickle
import random
import gym
import gym_boxpush
import numpy as np
import os
import tensorflow as tf
import multiprocessing
import itertools
import cv2
import argparse
from directed_exploration.utils.data_util import convertToOneHot
from multiprocessing import Process, Queue


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def write_user_controller_atari_episode_to_tf_record(episode_index, write_dir):

    env = gym.make('BreakoutDeterministic-v4')

    filename = os.path.join(write_dir, 'breakout_{}.tfrecords'.format(episode_index))
    writer = tf.python_io.TFRecordWriter(filename)

    frame = [None]
    action = [0]
    step_index = [0]

    from pyglet.window import key
    import pyglet

    # Action key-mappings should be redefined here for different games:
    def key_press(k, mod):
        if k == key.RIGHT: action[0] = 2
        if k == key.UP:    action[0] = 1
        if k == key.LEFT:  action[0] = 3

    def key_release(k, mod):
        if k == key.RIGHT and action[0] == 2: action[0] = 0
        if k == key.UP and action[0] == 1: action[0] = 0
        if k == key.LEFT and action[0] == 3: action[0] = 0

    def update(dt):
        if frame[0] is None:
            frame[0] = env.reset()

        frame[0] = cv2.resize(src=frame[0], dsize=(64, 64), interpolation=cv2.INTER_AREA)
        frame[0] = frame[0] / 255.0

        env.render()
        # cv2.imshow("game", frame[0])
        # cv2.waitKey(50)
        window.activate()

        frame_bytes = pickle.dumps(frame[0])

        # save frame and action
        example = tf.train.Example(features=tf.train.Features(feature={
            'action_at_frame': _floats_feature(convertToOneHot(action[0], num_classes=env.action_space.n)),
            'frame_bytes': _bytes_feature(frame_bytes)
        }))
        writer.write(example.SerializeToString())

        frame[0], reward, done, _ = env.step(action[0])

        step_index[0] = step_index[0] + 1

        if done:
            window.close()
            pyglet.app.exit()
            writer.close()
            print("Wrote {} with length {}".format(filename, step_index[0]))


    window = pyglet.window.Window()
    window.on_key_press = key_press
    window.on_key_release = key_release
    pyglet.clock.schedule_interval(update, 1 / 10.0)

    pyglet.app.run()

    return step_index[0]



def write_user_controlled_boxpush_episode_to_tf_record(episode_index, write_dir, max_episode_length):
    env = gym.make('boxpushmaze-v0')

    filename = os.path.join(write_dir, 'vae_{}.tfrecords'.format(episode_index))
    writer = tf.python_io.TFRecordWriter(filename)

    frame = env.reset()

    action = [0]

    from pyglet.window import key

    def key_press(k, mod):
        if k == key.RIGHT: action[0] = 1
        if k == key.UP:    action[0] = 2
        if k == key.DOWN:  action[0] = 3
        if k == key.LEFT:  action[0] = 4

    def key_release(k, mod):
        if k == key.RIGHT and action[0] == 1: action[0] = 0
        if k == key.UP    and action[0] == 2: action[0] = 0
        if k == key.DOWN  and action[0] == 3: action[0] = 0
        if k == key.LEFT  and action[0] == 4: action[0] = 0

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    step_index = 0
    while step_index < max_episode_length:
        env.render()

        frame = frame / 255.0
        frame_bytes = pickle.dumps(frame)

        # save frame and action
        example = tf.train.Example(features=tf.train.Features(feature={
            'action_at_frame': _floats_feature(convertToOneHot(action[0], num_classes=env.action_space.n)),
            'frame_bytes': _bytes_feature(frame_bytes)
        }))
        writer.write(example.SerializeToString())

        frame, reward, done, _ = env.step(action[0])

        step_index += 1

        if done:
            break

    writer.close()
    print("Wrote {} with length {}".format(filename, step_index))
    return step_index


def generate_user_controlled_boxpush_data(args):

    starting_episode_index = args.starting_episode
    num_episodes_to_generate = args.num_episodes

    ending_episode_index = starting_episode_index + num_episodes_to_generate

    if not os.path.exists(args.write_dir):
        os.makedirs(args.write_dir)

    frames_written_in_each_episode = []
    for episode_index in range(starting_episode_index, ending_episode_index):
        frames_written_in_each_episode.append(write_user_controlled_boxpush_episode_to_tf_record(
            episode_index=episode_index,
            write_dir=args.write_dir,
            max_episode_length=args.max_episode_length))

    print("Wrote {} frames in total".format(sum(frames_written_in_each_episode)))
    print("Last episode index written to was {}".format(ending_episode_index-1))
    print("(So if writing more episodes, start at {})".format(ending_episode_index))


def generate_user_controlled_atari_data(args):

    starting_episode_index = args.starting_episode
    num_episodes_to_generate = args.num_episodes

    ending_episode_index = starting_episode_index + num_episodes_to_generate

    if not os.path.exists(args.write_dir):
        os.makedirs(args.write_dir)

    frames_written_in_each_episode = []

    result_queue = Queue()

    def run_atari_episode(queue):
        frames_written_in_episode = write_user_controller_atari_episode_to_tf_record(
            episode_index=episode_index,
            write_dir=args.write_dir
        )
        queue.put(frames_written_in_episode)

    for episode_index in range(starting_episode_index, ending_episode_index):
        p = Process(target=run_atari_episode, args=(result_queue,))
        p.start()
        p.join()
        frames_written_in_each_episode.append(result_queue.get())

    print("\nWrote {} frames in total".format(sum(frames_written_in_each_episode)))
    print("Last episode index written to was {}".format(ending_episode_index-1))
    print("(So if writing more episodes, start at {})".format(ending_episode_index))


def write_random_episode_to_tf_record(episode_index, write_dir, max_episode_length):
    env = gym.make('boxpushsimple-v0')

    filename = os.path.join(write_dir, 'vae_{}.tfrecords'.format(episode_index))
    writer = tf.python_io.TFRecordWriter(filename)

    frame = env.reset()
    action = env.action_space.sample()

    step_index = 0
    while step_index < max_episode_length:
        frame = frame / 255.0
        frame_bytes = pickle.dumps(frame)

        # save frame and action
        example = tf.train.Example(features=tf.train.Features(feature={
            'action_at_frame': _int64_feature(action),
            'frame_bytes': _bytes_feature(frame_bytes)
        }))
        writer.write(example.SerializeToString())

        frame, reward, done, _ = env.step(action)

        if random.random() < 0.05:
            action = env.action_space.sample()

        step_index += 1

        if random.random() < 1 / max_episode_length:
            break

    writer.close()
    print("Wrote {} with length {}".format(filename, step_index))
    return step_index


def generate_random_vae_data(args):

    starting_episode_index = args.starting_episode
    num_episodes_to_generate = args.num_episodes

    ending_episode_index = starting_episode_index + num_episodes_to_generate

    if not os.path.exists(args.write_dir):
        os.makedirs(args.write_dir)

    with multiprocessing.Pool(processes=args.num_processes) as pool:
        frames_written_in_each_episode = pool.starmap(write_user_controlled_boxpush_episode_to_tf_record,
                                                      zip(range(starting_episode_index, ending_episode_index),
                                                          itertools.repeat(args.write_dir),
                                                          itertools.repeat(args.max_episode_length)
                                                          )
                                                      )

        print("Wrote {} frames in total".format(sum(frames_written_in_each_episode)))
        print("Last episode index written to was {}".format(ending_episode_index-1))
        print("(So if writing more episodes, start at {})".format(ending_episode_index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--starting-episode", help="Numbered files created will start at this number",
                        type=int, required=True)
    parser.add_argument("--num-episodes", help="Number of episode rollouts to generate",
                        type=int, required=True)
    parser.add_argument("--write-dir", help="Directory to save tfrecords files to",
                        type=str, default='vae_tf_records')
    parser.add_argument("--num-processes", help="Number of concurrent process to generate rollouts",
                        type=int, default=12)
    parser.add_argument("--max-episode-length", help="Maximum length of any single episode",
                        type=int, default=1000)
    parser.add_argument("--user-controlled", help="user controls agent movements in environment",
                        action="store_true")
    args = parser.parse_args()

    if not args.user_controlled:
        generate_random_vae_data(args)
    else:
        generate_user_controlled_atari_data(args)


