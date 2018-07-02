
import numpy as np
import tensorflow as tf
import multiprocessing
from directed_exploration.utils.data_util import convertToOneHot
import logging

logger = logging.getLogger(__name__)

#TODO All of this code is deprecated

def get_vae_deque_input_fn(train_deque, batch_size, shuffle_buffer_size=1000):

    def sequence_frames_generator():
        for sequence in train_deque:
            yield sequence[0]
        return

    def slice_and_shuffle_fn(x1):
        sequence_length = tf.cast(tf.shape(x1)[0], tf.int64)
        return tf.data.Dataset.from_tensor_slices(x1).shuffle(buffer_size=sequence_length)

    def input_fn():
        episode_frames = tf.data.Dataset.from_generator(generator=sequence_frames_generator,
                                                        output_types=tf.float32,
                                                        output_shapes=tf.TensorShape([None, 64, 64, 3]))
        cycle_length = 30
        dataset = episode_frames.interleave(map_func=slice_and_shuffle_fn,
                                            cycle_length=cycle_length,
                                            block_length=1)

        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=10)

        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator.initializer

    return input_fn


def get_state_rnn_deque_input_fn(state_rnn_episodes_deque, batch_size, latent_dim, num_actions, max_sequence_length, num_epochs=1):

    def sequence_generator():
        for _ in range(num_epochs):
            for sequence in state_rnn_episodes_deque:
                yield sequence
        return

    def format_sequence_fn(code_sequence, action_sequence, sequence_length):
        assert len(code_sequence) == len(action_sequence) == max_sequence_length
        assert sequence_length <= len(code_sequence)

        # input to rnn is sequence[n] (latent dim + actions], target is sequence[n+1] (latent dim only)
        input_sequence = np.concatenate((np.copy(code_sequence), action_sequence), axis=1)[:-1]

        # remove last entry from input as it should only belong in the target sequence
        if sequence_length < len(code_sequence):
            input_sequence[sequence_length - 1] = 0

        target_sequence = code_sequence[1:]

        return input_sequence, target_sequence, np.int32(sequence_length - 1)

    def input_fn():
        dataset = tf.data.Dataset.from_generator(generator=sequence_generator,
                                                 output_types=(tf.float32, tf.float32, tf.int32),
                                                 output_shapes=(tf.TensorShape([max_sequence_length, latent_dim]),
                                                                tf.TensorShape([max_sequence_length, num_actions]),
                                                                tf.TensorShape([])))

        dataset = dataset.map(map_func=lambda t1, t2, t3: tuple(tf.py_func(func=format_sequence_fn,
                                                                            inp=[t1, t2, t3],
                                                                            Tout=[tf.float32, tf.float32,
                                                                                  tf.int32],
                                                                            stateful=False)),
                           num_parallel_calls=multiprocessing.cpu_count())

        dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=10)

        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator.initializer

    return input_fn


# def get_anticipator_input_fn(anticipator_deque, batch_size, max_sequence_length, num_epochs=5):
#
#     def episode_generator():
#         for _ in range(num_epochs):
#             for episode in anticipator_deque:
#                 yield episode
#         return
#
#     def slice_and_shuffle_fn(x1, x2, x3, x4):
#         num_sequences = tf.cast(tf.shape(x1)[0], tf.int64)
#         return tf.data.Dataset.from_tensor_slices((x1, x2, x3, x4)).shuffle(buffer_size=num_sequences)
#
#     def input_fn():
#         dataset = tf.data.Dataset.from_generator(generator=episode_generator,
#                                                  output_types=(tf.float32, tf.float32, tf.float32, tf.int32),
#                                                  output_shapes=(tf.TensorShape([None, max_sequence_length-1, 64, 64, 3]),
#                                                                 tf.TensorShape([None, max_sequence_length-1, 1]),
#                                                                 tf.TensorShape([None, max_sequence_length-1]),
#                                                                 tf.TensorShape([None])))
#
#         cycle_length = 30
#         dataset = dataset.interleave(map_func=slice_and_shuffle_fn,
#                                      cycle_length=cycle_length,
#                                      block_length=1)
#
#         dataset = dataset.shuffle(buffer_size=30)
#
#         dataset = dataset.batch(batch_size=batch_size)
#         dataset = dataset.prefetch(buffer_size=10)
#
#         iterator = dataset.make_initializable_iterator()
#         return iterator.get_next(), iterator.initializer
#
#     return input_fn


# def divide_episode_into_sequences(episode, max_sequence_length):
#     sequences = [episode[i * max_sequence_length:(i + 1) * max_sequence_length]
#                  for i in range((len(episode) + max_sequence_length - 1) // max_sequence_length)]
#
#     sequence_lengths = [np.int32(len(sequence)) for sequence in sequences]
#
#     # Can't do anything useful with a sequence of length 1
#     if len(sequences[-1]) < 2:
#         sequences = sequences[:-1]
#         sequence_lengths = sequence_lengths[:-1]
#         assert len(sequences[-1]) == max_sequence_length
#
#     # Pad last sequence with zeros to ensure all sequences are in the same shape ndarray
#     elif len(sequences[-1]) < max_sequence_length:
#         zeros = np.zeros(shape=(max_sequence_length - sequences[-1].shape[0], *sequences[-1].shape[1:]))
#         sequences[-1] = np.concatenate((sequences[-1], zeros), axis=0)
#
#     return sequences, sequence_lengths


def pad_sequence_with_zeros(sequence, max_sequence_length):
    sequence_length = np.int32(len(sequence))

    if len(sequence) < max_sequence_length:
        zeros = np.zeros(shape=(max_sequence_length - sequence.shape[0], *sequence.shape[1:]))
        sequence = np.concatenate((sequence, zeros), axis=0)

    return sequence, sequence_length


def convert_vae_deque_to_state_rnn_deque(vae, vae_deque, state_rnn_episodes_deque, max_sequence_length,
                                         threads=multiprocessing.cpu_count()):

    def format_and_add_sequence_to_deque(sequence_num):

        sequence_frames, sequence_actions = vae_deque.pop()
        sequence_actions, action_seq_len = pad_sequence_with_zeros(sequence_actions, max_sequence_length)

        if action_seq_len >= 2:
            code_sequence = vae.encode_frames(sequence_frames)
            code_sequence, sequence_length = pad_sequence_with_zeros(code_sequence, max_sequence_length)
            assert sequence_length == action_seq_len
            assert len(code_sequence) == len(sequence_actions) == max_sequence_length

            state_rnn_episodes_deque.append((code_sequence,
                                             sequence_actions,
                                             sequence_length))

            return sequence_length

        else:
            return 0

    with multiprocessing.pool.ThreadPool(processes=threads) as pool:
        sequence_lengths = pool.map(func=format_and_add_sequence_to_deque, iterable=range(len(vae_deque)))

    return sequence_lengths


