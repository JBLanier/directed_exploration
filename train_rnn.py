
from state_rnn import StateRNN
import argparse
import random
import math
import numpy as np
import os
import re
import pickle
import tensorflow as tf
import multiprocessing
import cv2

MAX_SEQUENCES_PER_RNN_TF_RECORD = 5

def get_numbered_tfrecord_file_names_from_directory(dir, prefix):
    dir_files = os.listdir(dir)
    dir_files = sorted(filter(lambda f: str.isdigit(re.split('[_.]+', f)[1]) and f.startswith(prefix), dir_files),
                       key=lambda f: int(re.split('[_.]+', f)[1]))
    return list(map(lambda f: os.path.join(dir, f), dir_files))


def get_rnn_tfrecord_input_fn(train_data_dir, batch_size=32, num_epochs=1):
    prefix = 'rnn'
    input_file_names = get_numbered_tfrecord_file_names_from_directory(dir=train_data_dir, prefix=prefix)

    if len(input_file_names) <= 0:
        raise FileNotFoundError("No usable tfrecords with prefix \'{}\' were found at {}".format(
            prefix, train_data_dir)
        )

    def decode_pickled_np_array(np_bytes):
        return pickle.loads(np_bytes).astype(np.float32)

    def decode_state_rnn_input_target_sequences(np_bytes, sequence_length, latent_dims):
        sequence = decode_pickled_np_array(np_bytes)

        # input to rnn is sequence[n] (latent dim + actions], target is sequence[n+1] (latent dim only)
        input_sequence = np.copy(sequence[:-1])

        # remove last entry from input as it should only belong in the target sequence
        if len(sequence) > sequence_length:
            input_sequence[sequence_length - 1] = 0

        target_sequence = sequence[1:, :latent_dims]

        # print("target_sequence: {} input sequence: {}\nlast in overall sequence: {}\nlength: {}\n".format(target_sequence[sequence_length-2,0],
        #                                                                                     input_sequence[sequence_length-2,0],
        #                                                                                     sequence[sequence_length-2,0],
        #                                                                                                 sequence_length))

        return input_sequence, target_sequence

    def parse_fn(example):
        example_fmt = {
            "sequence_bytes": tf.FixedLenFeature([], tf.string),
            "sequence_length": tf.FixedLenFeature([], tf.int64),
            "latent_dims": tf.FixedLenFeature([], tf.int64)
        }

        parsed = tf.parse_single_example(example, example_fmt)

        sequence_bytes = parsed["sequence_bytes"]
        sequence_length = parsed["sequence_length"]
        latent_dims = parsed["latent_dims"]

        input_sequence, target_sequence = tf.py_func(func=decode_state_rnn_input_target_sequences,
                                                     inp=[sequence_bytes, sequence_length, latent_dims],
                                                     Tout=(tf.float32, tf.float32),
                                                     stateful=False,
                                                     name='decode_np_bytes')

        return input_sequence, target_sequence, sequence_length-1

    def extract_and_shuffle_fn(file_tensor):
        return tf.data.TFRecordDataset(file_tensor).shuffle(buffer_size=MAX_SEQUENCES_PER_RNN_TF_RECORD)

    def input_fn():
        file_names = tf.constant(input_file_names, dtype=tf.string, name='input_file_names')
        file_tensors = tf.data.Dataset.from_tensor_slices(file_names).shuffle(len(input_file_names))

        # Shuffle frames in each episode/tfrecords file, then draw a frame from each episode/tfrecords file in a cycle
        dataset = file_tensors.interleave(map_func=extract_and_shuffle_fn,
                                          cycle_length=80,
                                          block_length=1)

        # Shuffle drawn sequences
        dataset = dataset.shuffle(buffer_size=200)

        # Parse tfrecords into frames
        dataset = dataset.map(map_func=parse_fn, num_parallel_calls=multiprocessing.cpu_count())

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.repeat(num_epochs)  # the input is repeated indefinitely if num_epochs is None
        dataset = dataset.prefetch(buffer_size=10)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return input_fn


def train_state_rnn(rnn):
    train_data_dir = 'rnn_tf_records'

    input_fn = get_rnn_tfrecord_input_fn(train_data_dir, batch_size=64, num_epochs=500)

    rnn.train_on_input_fn(input_fn)



def main():

    state_rnn = StateRNN()
    train_state_rnn(state_rnn)

if __name__ == '__main__':

    main()