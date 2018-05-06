import pickle
import random
import gym
import gym_boxpush
import numpy as np
import os
import tensorflow as tf
import multiprocessing
import itertools
import argparse

ACTIONS = [
    [0,    0], [0,    0],
    [1,    0], [1,    0],
    [1,  0.5], [1,  0.5],
    [1, -0.5], [1, -0.5],
    [1,    1], [1,   -1]
]


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_random_episode_to_tf_record(episode_index, write_dir, max_episode_length):
    env = gym.make('boxpush-v0')

    filename = os.path.join(write_dir, 'vae_{}.tfrecords'.format(episode_index))
    writer = tf.python_io.TFRecordWriter(filename)

    frame = env.reset()
    action = random.sample(ACTIONS, 1)[0]

    step_index = 0
    while step_index < max_episode_length:
        frame = frame / 255.0
        frame_bytes = pickle.dumps(frame)

        # save frame and action
        example = tf.train.Example(features=tf.train.Features(feature={
            'action_at_frame': _floats_feature(action),
            'frame_bytes': _bytes_feature(frame_bytes)
        }))
        writer.write(example.SerializeToString())

        frame, reward, done, _ = env.step(np.array(action))
        # env.render()

        if random.random() < 0.1:
            action = random.sample(ACTIONS, 1)[0]

        step_index += 1

        if random.random() < 1 / max_episode_length:
            break

    writer.close()
    print("Wrote {} with length {}".format(filename, step_index))
    return step_index


def generate_vae_data(args):

    starting_episode_index = args.starting_episode
    num_episodes_to_generate = args.num_episodes

    ending_episode_index = starting_episode_index + num_episodes_to_generate

    if not os.path.exists(args.write_dir):
        os.makedirs(args.write_dir)

    with multiprocessing.Pool(processes=args.num_processes) as pool:
        frames_written_in_each_episode = pool.starmap(write_random_episode_to_tf_record,
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
                        type=str, default='tf_records')
    parser.add_argument("--num-processes", help="Number of concurrent process to generate rollouts",
                        type=int, default=10)
    parser.add_argument("--max-episode-length", help="Maximum length of any single episode",
                        type=int, default=1000)
    args = parser.parse_args()

    generate_vae_data(args)


