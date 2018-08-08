import pickle
import numpy as np
import os
import tensorflow as tf
import cv2
import multiprocessing
import argparse
import re
import itertools
from functools import reduce
from directed_exploration.sep_vae_rnn.vae import VAE
import gym
import random

# Dimensionality of actions to read from vae tf_records and pass to rnn
ACTION_LENGTH = 2
FRAME_DIMS = (64, 64, 3)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_numbered_tfrecord_file_names_from_directory(dir, prefix):
    dir_files = os.listdir(dir)
    dir_files = sorted(filter(lambda f: str.isdigit(re.split('[_.]+', f)[1]) and f.startswith(prefix), dir_files),
                       key=lambda f: int(re.split('[_.]+', f)[1]))
    return list(map(lambda f: os.path.join(dir, f), dir_files))


def decode_pickled_np_array(bytes):
    return pickle.loads(bytes).astype(np.float32)


def debug_imshow_image_with_action(frame, action, wait_time=30, window_label='frame'):
    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (0, 10)
    font_scale = 0.3
    font_color = (0, 0, 0)
    line_type = 1

    cv2.putText(frame, str(action),
                location,
                font,
                font_scale,
                font_color,
                line_type)

    cv2.imshow(window_label, frame[:, :, ::-1])


def write_vae_episode_to_rnn_tf_record(vae_read_file_name, vae, rnn_data_write_dir, max_sequence_length):
    record_iterator = tf.python_io.tf_record_iterator(vae_read_file_name)
    all_of_file_read = False

    episode_index = int(re.split("[_.]+", vae_read_file_name)[-2])

    write_file_name = os.path.join(rnn_data_write_dir, 'rnn_{}.tfrecords'.format(episode_index))
    tf_record_writer = tf.python_io.TFRecordWriter(write_file_name)

    sequence_lengths = []

    while not all_of_file_read:

        # If a generated sequence is less than max_sequence_length, the rest of it will be zeros.
        raw_frame_sequence = np.zeros(shape=(max_sequence_length,) + FRAME_DIMS, dtype=np.float32)
        action_sequence = np.zeros(shape=(max_sequence_length, ACTION_LENGTH), dtype=np.float32)
        # encoded_sequence = np.zeros(shape=(max_sequence_length, vae.latent_dim + ACTION_LENGTH), dtype=np.float32)

        sequence_length = 0
        for sequence_index in range(max_sequence_length):
            try:
                serialized_example = next(record_iterator)
            except StopIteration:
                all_of_file_read = True
                break

            example = tf.train.Example()
            example.ParseFromString(serialized_example)

            action = example.features.feature['action_at_frame'].float_list.value[:]
            raw_frame = decode_pickled_np_array(example.features.feature['frame_bytes'].bytes_list.value[0])

            raw_frame_sequence[sequence_index] = raw_frame
            action_sequence[sequence_index] = action

            # encoded_frame = vae.encode_frames(np.expand_dims(raw_frame, axis=0))

            # Debug Visualization ##
            # decoded = vae.decode_frames(encoded_frame)[0]
            # debug_imshow_image_with_action(raw_frame, action, window_label='original')
            # debug_imshow_image_with_action(decoded, action, window_label='encoded_decoded')
            # cv2.waitKey(30)
            ##

            # sample_entry = np.concatenate((encoded_frame[0], action), axis=0)
            # encoded_sequence[sequence_index] = sample_entry
            sequence_length += 1

        # Must have at least two frames to be suitable for RNN training
        if sequence_length >= 2:
            encoded_sequence = vae.encode_frames(raw_frame_sequence)
            encoded_sequence[sequence_length:] = 0
            encoded_sequence = np.concatenate((encoded_sequence, action_sequence), axis=1)

            # print("encoded sequence shape: {}".format(encoded_sequence.shape))

            sequence_lengths.append(sequence_length)

            sequence_bytes = pickle.dumps(encoded_sequence)

            # print(sequence[:-4:,0])
            # print(sequence[sequence_length-1, 0])
            # print(sequence[-1, 0])

            # save sequence, its length, and the size of the frame encoding
            example = tf.train.Example(features=tf.train.Features(feature={
                'sequence_bytes': _bytes_feature(sequence_bytes),
                'sequence_length': _int64_feature(sequence_length),
                'latent_dims': _int64_feature(vae.latent_dim)
            }))
            tf_record_writer.write(example.SerializeToString())

    tf_record_writer.close()
    print("Wrote {} with sequences {}".format(write_file_name, sequence_lengths))
    return sequence_lengths


def convert_vae_record_to_rnn_records(vae_model_dir, vae_data_read_dir, rnn_data_write_dir, max_sequence_length):
    print('\n' + ("_" * 20) + '\n')
    print("Reading from {}\nWriting to {}\nMax sequence length is {}\nVAE model is from {}".format(
        vae_data_read_dir, rnn_data_write_dir, max_sequence_length, vae_model_dir))
    print(("_" * 20) + '\n')

    if not os.path.exists(rnn_data_write_dir):
        os.makedirs(rnn_data_write_dir)

    vae_tf_records_files = get_numbered_tfrecord_file_names_from_directory(dir=vae_data_read_dir, prefix='vae')

    sess = tf.Session()
    with sess.as_default():
        vae = VAE(restore_from_dir=vae_model_dir, latent_dim=1)

        with multiprocessing.pool.ThreadPool(processes=args.num_workers) as pool:
            episode_sequence_lengths = pool.starmap(write_vae_episode_to_rnn_tf_record,
                                                    zip(vae_tf_records_files,
                                                        itertools.repeat(vae),
                                                        itertools.repeat(rnn_data_write_dir),
                                                        itertools.repeat(max_sequence_length)
                                                        )
                                                    )

    number_of_sequences_written = reduce(lambda acc, episode: acc + len(episode), episode_sequence_lengths, 0)
    total_frames_written = reduce(lambda acc, episode: acc + sum(episode), episode_sequence_lengths, 0)

    print("Done, wrote {} episodes with {} sequences ({} frames).".format(len(vae_tf_records_files),
                                                                          number_of_sequences_written,
                                                                          total_frames_written))


def write_new_random_boxpush_simple_rollout_rnn_tf_record(episode_index, rnn_data_write_dir,
                                                          max_episode_length, max_sequence_length):
    """
    For debugging RNN, boxpushsimple frames are given a -perfect- 1d encoding
    by cheating and asking the exact location of the player.
    """

    episode_over = False

    write_file_name = os.path.join(rnn_data_write_dir, 'rnn_{}.tfrecords'.format(episode_index))
    tf_record_writer = tf.python_io.TFRecordWriter(write_file_name)

    sequence_lengths = []
    total_frames = 0

    env = gym.make('boxpushsimple-v0')
    env.reset()

    actions = [[0, 0], [1, 0], [1, -1]]
    action_index = 0

    while not episode_over:

        # If a generated sequence is less than max_sequence_length, the rest of it will be zeros.
        encoded_sequence = np.zeros(shape=(max_sequence_length, 1), dtype=np.float32)
        action_sequence = np.zeros(shape=(max_sequence_length, ACTION_LENGTH), dtype=np.float32)

        sequence_length = 0
        for sequence_index in range(max_sequence_length):

            if random.random() < 1 / max_episode_length or total_frames > max_episode_length:
                episode_over = True
                break

            if random.random() < 0.05:
                action_index = random.randint(0, len(actions) - 1)

            action = np.asarray(actions[action_index])
            location = env.debug_get_player_location()
            if location == 0:
                location += 0.000001
            encoded_sequence[sequence_index] = location
            action_sequence[sequence_index] = action

            env.step(action)

            sequence_length += 1
            total_frames += 1

        # Must have at least two frames to be suitable for RNN training
        if sequence_length >= 2:
            encoded_sequence = np.concatenate((encoded_sequence, action_sequence), axis=1)

            sequence_lengths.append(sequence_length)

            sequence_bytes = pickle.dumps(encoded_sequence)

            # save sequence, its length, and the size of the frame encoding
            example = tf.train.Example(features=tf.train.Features(feature={
                'sequence_bytes': _bytes_feature(sequence_bytes),
                'sequence_length': _int64_feature(sequence_length),
                'latent_dims': _int64_feature(1)
            }))
            tf_record_writer.write(example.SerializeToString())

    tf_record_writer.close()
    print("Wrote {} with sequences {}".format(write_file_name, sequence_lengths))
    return sequence_lengths


def generate_perfect_boxpushsimple_rnn_records(rnn_data_write_dir, max_episode_length,
                                               max_sequence_length, num_episodes):
    print('\n' + ("_" * 20) + '\n')
    print("Generating Fake Encoding from BoxPushSimple\nNum episodes to write is {}"
          "Writing to {}\nMax sequence length is {}\nMax Episode Length is {}".format(num_episodes,
                                                                                      rnn_data_write_dir,
                                                                                      max_sequence_length,
                                                                                      max_episode_length))
    print(("_" * 20) + '\n')

    if not os.path.exists(rnn_data_write_dir):
        os.makedirs(rnn_data_write_dir)

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        episode_sequence_lengths = pool.starmap(write_new_random_boxpush_simple_rollout_rnn_tf_record,
                                                zip(range(0, num_episodes),
                                                    itertools.repeat(rnn_data_write_dir),
                                                    itertools.repeat(max_episode_length),
                                                    itertools.repeat(max_sequence_length)
                                                    )
                                                )

    number_of_sequences_written = reduce(lambda acc, episode: acc + len(episode), episode_sequence_lengths, 0)
    total_frames_written = reduce(lambda acc, episode: acc + sum(episode), episode_sequence_lengths, 0)

    print("Done, wrote {} episodes with {} sequences ({} frames).".format(len(episode_sequence_lengths),
                                                                          number_of_sequences_written,
                                                                          total_frames_written))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--read-dir", help="Directory to save tfrecords files to",
                        type=str, default='vae_tf_records')
    parser.add_argument("--write-dir", help="Directory to save tfrecords files to",
                        type=str, default='rnn_tf_records')
    parser.add_argument("--load-vae-weights", help="dir to load VAE weights with",
                        type=str, default=None)
    parser.add_argument("--num-workers", help="Number of concurrent threads to convert rollouts",
                        type=int, default=12)
    parser.add_argument("--max-sequence-length", help="Maximum length of any rnn sequence example",
                        type=int, default=200)
    parser.add_argument("--perfect-boxpushsimple", help="create fake data directly from box push simple internal state",
                        action="store_true")

    args = parser.parse_args()

    if not args.perfect_boxpushsimple:
        if args.load_vae_weights:
            convert_vae_record_to_rnn_records(vae_model_dir=args.load_vae_weights,
                                              vae_data_read_dir=args.read_dir,
                                              rnn_data_write_dir=args.write_dir,
                                              max_sequence_length=args.max_sequence_length)
        else:
            print("You must specify --load-vae-weights=<vae weights dir>")
            exit(1)
    else:
        generate_perfect_boxpushsimple_rnn_records(rnn_data_write_dir=args.write_dir,
                                                   max_episode_length=1000,
                                                   max_sequence_length=args.max_sequence_length,
                                                   num_episodes=3000)
