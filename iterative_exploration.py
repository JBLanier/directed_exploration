from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common import set_global_seeds

import gym
import gym_boxpush
import numpy as np
import random
import math
from collections import deque
import tensorflow as tf
import time
import multiprocessing
from functools import reduce

import cv2

from anticipator import AnticipatorRNN
from vae import VAE
from state_rnn import StateRNN

seed = 42

ACTIONS = [[0, 0], [1, 0], [1, -1]]
ACTION_DIM = 2


def debug_imshow_image_with_action(window_label, frame, action):
    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (0, 10)
    font_scale = 0.3
    font_color = (0, 0, 0)
    line_type = 1

    frame = np.copy(frame)

    cv2.putText(frame, str(action),
                location,
                font,
                font_scale,
                font_color,
                line_type)

    cv2.imshow(window_label, frame[:, :, ::-1])


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def make_boxpush_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped SubprocVecEnv for boxpush.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}

    def make_env(rank):  # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            # env.seed(seed + rank)
            return env

        return _thunk

    # set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def generate_rollouts_on_anticipator_policy_into_deque(episode_deque, anticipator, env, num_env,
                                                       num_episodes_per_environment, max_episode_length,
                                                       available_actions):
    total_frames = 0
    for _ in range(num_episodes_per_environment):
        start = time.time()
        obs = env.reset() / 255.0

        episode_frames = [np.empty(shape=(max_episode_length, 64, 64, 3), dtype=np.float32) for _ in range(num_env)]
        episode_actions = [np.empty(shape=(max_episode_length, ACTION_DIM), dtype=np.float32) for _ in range(num_env)]
        episode_lengths = np.zeros(shape=num_env, dtype=np.uint32)
        dones = np.full(num_env, False)

        for episode_frame_index in range(max_episode_length):

            # Predict loss amounts (action scores) for each action/observation
            action_scores = anticipator.predict_on_frame_batch(
                frames=np.repeat(obs, [len(available_actions)] * num_env, 0),
                actions=np.asarray(available_actions * num_env))
            action_scores = np.reshape(action_scores, newshape=[num_env, len(available_actions)])
            # Normalize action scores to be probabilities
            action_scores = np.maximum(action_scores, 0.0001)
            action_scores = action_scores / np.sum(action_scores, axis=1)[:, None]
            # Sample next actions to take from probabilities
            action_indexes = [np.random.choice(len(available_actions), p=action_probs) for action_probs in
                              action_scores]
            actions_to_take = np.asarray([available_actions[action_to_take] for action_to_take in action_indexes])

            for env_index in range(num_env):
                if not dones[env_index]:
                    episode_frames[env_index][episode_frame_index] = obs[env_index]
                    episode_actions[env_index][episode_frame_index] = actions_to_take[env_index]
                    episode_lengths[env_index] += 1
                    if random.random() < 1 / (max_episode_length * 2) and episode_lengths[env_index] >= 2:
                        dones[env_index] = True

            obs, _, _, _ = env.step(actions_to_take)

            # for i in range(len(obs)):
            #     cv2.imshow(("env {}".format(i+1)),obs[i,:,:,::-1])
            # cv2.waitKey(1)

            obs = obs / 255.0

            if dones.all():
                break

        for env_index in range(num_env):
            episode_frames[env_index].resize((episode_lengths[env_index], 64, 64, 3))
            episode_actions[env_index].resize((episode_lengths[env_index], ACTION_DIM))
            episode_deque.append((episode_frames[env_index], episode_actions[env_index]))
        end = time.time()
        print("generated episodes with lengths: {}".format(episode_lengths))
        running_time = end - start
        print("took {} seconds, per-environment efficiency is {}".format(running_time, num_env / running_time))
        total_frames += sum(episode_lengths)

    return num_episodes_per_environment * num_env, total_frames


def get_vae_deque_input_fn(episode_deque, batch_size, max_episode_length):
    """
    :param in_episode_deque: deque of episode as numpy arrays
    :param batch_size: batch size
    :param max_episode_length: the max episode length that's stored in in_episode_deque
    :return: input_fn for training vae on in_episode_deque
    """

    def episode_frames_generator():
        for episode in episode_deque:
            yield episode[0]
        return

    def slice_and_shuffle_fn(x1):
        episode_length = tf.cast(tf.shape(x1)[0], tf.int64)
        return tf.data.Dataset.from_tensor_slices(x1).shuffle(buffer_size=episode_length)

    def input_fn():
        episode_frames = tf.data.Dataset.from_generator(generator=episode_frames_generator,
                                                        output_types=tf.float32,
                                                        output_shapes=tf.TensorShape([None, 64, 64, 3]))
        cycle_length = 30
        dataset = episode_frames.interleave(map_func=slice_and_shuffle_fn,
                                            cycle_length=cycle_length,
                                            block_length=1)

        dataset = dataset.shuffle(buffer_size=max_episode_length * 2)

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=10)

        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator.initializer

    return input_fn


def get_state_rnn_deque_input_fn(state_rnn_episodes_deque, batch_size, max_episode_length, latent_dim,
                                 max_sequence_length):

    def episode_pop_generator():
        while True:
            try:
                yield state_rnn_episodes_deque.pop()
            except IndexError:
                break

    def format_sequence_fn(code_sequence, action_sequence, sequence_length, frame_sequence):
        assert len(code_sequence) == len(action_sequence) == frame_sequence == max_sequence_length
        assert sequence_length <= len(code_sequence)

        # input to rnn is sequence[n] (latent dim + actions], target is sequence[n+1] (latent dim only)
        input_sequence = np.concatenate((np.copy(code_sequence), action_sequence), axis=1)[:-1]
        frame_input_sequence = frame_sequence[:-1]

        # remove last entry from input as it should only belong in the target sequence
        if sequence_length < len(code_sequence):
            input_sequence[sequence_length - 1] = 0
            frame_input_sequence[sequence_length - 1] = 0

        target_sequence = code_sequence[1:]

        return input_sequence, target_sequence, sequence_length - 1, frame_input_sequence

    def slice_shuffle_and_format_fn(x1, x2, x3, x4):
        episode_length = tf.cast(tf.shape(x1)[0], tf.int64)

        inputs = (x1, x2, x3, x4)

        dataset = tf.data.Dataset.from_tensor_slices(inputs).shuffle(buffer_size=episode_length)
        return dataset.map(map_func=lambda t1, t2, t3, t4: tuple(tf.py_func(func=format_sequence_fn,
                                                                            inp=[t1, t2, t3, t4],
                                                                            Tout=[tf.float32, tf.float32,
                                                                                  tf.float32, tf.float32],
                                                                            stateful=False)),
                           num_parallel_calls=multiprocessing.cpu_count())

    def input_fn():
        dataset = tf.data.Dataset.from_generator(generator=episode_pop_generator,
                                                 output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape([None, max_episode_length, latent_dim]),
                                                                tf.TensorShape([None, max_episode_length, ACTION_DIM]),
                                                                tf.TensorShape([None]),
                                                                tf.TensorShape([None, max_episode_length, 64, 64, 3])))

        cycle_length = 30
        dataset = dataset.interleave(map_func=slice_shuffle_and_format_fn,
                                     cycle_length=cycle_length,
                                     block_length=1)

        dataset = dataset.shuffle(buffer_size=30)

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=10)

        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator.initializer

    return input_fn


def divide_episode_into_sequences(episode, max_sequence_length):
    sequences = [episode[i * max_sequence_length:(i + 1) * max_sequence_length]
                 for i in range((len(episode) + max_sequence_length - 1) // max_sequence_length)]

    sequence_lengths = [len(sequence) for sequence in sequences]

    # Pad last sequence with zeros to ensure all sequences are in the same shape ndarray
    if len(sequences[-1]) < max_sequence_length:
        zeros = np.zeros(shape=(max_sequence_length - sequences[-1].shape[0], *sequences[-1].shape[1:]))
        sequences[-1] = np.concatenate((sequences[-1], zeros), axis=0)

    return sequences, sequence_lengths


def convert_vae_deque_to_state_rnn_deque(vae, vae_deque, state_rnn_episodes_deque, max_sequence_length,
                                         threads=multiprocessing.cpu_count()):
    def episode_pop_generator():
        while True:
            try:
                yield vae_deque.pop()
            except IndexError:
                break

    def add_episode_to_sequence_deque(episode_frames, episode_actions):

        episode_frames_sequences, sequence_lengths = divide_episode_into_sequences(episode_frames, max_sequence_length)
        episode_actions_sequences, action_seq_len = divide_episode_into_sequences(episode_actions, max_sequence_length)
        episode_code_sequences = []

        for sequence_index in range(len(episode_frames_sequences)):
            frame_sequence = episode_frames_sequences[sequence_index]
            action_sequence = episode_actions_sequences[sequence_index]
            sequence_length = sequence_lengths[sequence_index]
            assert sequence_length == action_seq_len[sequence_index]
            assert len(frame_sequence) == len(action_sequence) == max_sequence_length
            if sequence_length > 2:
                code_sequence = vae.encode_frames(frame_sequence)
                code_sequence[sequence_length:] = 0

                episode_code_sequences.append(code_sequence)

        if len(episode_code_sequences) > 0:
            state_rnn_episodes_deque.append((np.stack(episode_code_sequences),
                                             np.stack(episode_actions_sequences),
                                             np.stack(sequence_lengths),
                                             np.stack(episode_frames_sequences)))

        return sequence_lengths

    with multiprocessing.pool.ThreadPool(processes=threads) as pool:
        episode_sequence_lengths = pool.starmap(func=add_episode_to_sequence_deque, iterable=episode_pop_generator())

    return episode_sequence_lengths


def do_iterative_exploration(env_id, num_env, num_iterations, latent_dim):
    env = make_boxpush_env(env_id, num_env, seed)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        anticipator = AnticipatorRNN()
        vae = VAE(latent_dim=latent_dim)
        state_rnn = StateRNN(latent_dim=latent_dim)
        num_episodes_per_environment = 1
        max_episode_length = 400
        max_sequence_length = 200
        vae_episodes_deque = deque()
        state_rnn_episodes_deque = deque()

        vae_input_fn = get_vae_deque_input_fn(episode_deque=vae_episodes_deque, batch_size=256,
                                              max_episode_length=max_episode_length)
        vae_input_fn_iter, vae_input_fn_init_op = vae_input_fn()

        state_rnn_input_fn = get_state_rnn_deque_input_fn(state_rnn_episodes_deque=state_rnn_episodes_deque,
                                                          batch_size=64, max_episode_length=max_episode_length,
                                                          latent_dim=latent_dim,
                                                          max_sequence_length=max_sequence_length)
        state_rnn_input_fn_iter, state_rnn_input_fn_init_op = state_rnn_input_fn()

        for iteration in range(num_iterations):
            print("_" * 20)
            print("Iteration {}".format(iteration + 1))
            print("\nExploring and generating rollouts...")
            num_ep, num_frames = generate_rollouts_on_anticipator_policy_into_deque(vae_episodes_deque, anticipator,
                                                                                    env, num_env,
                                                                                    num_episodes_per_environment,
                                                                                    max_episode_length, ACTIONS)
            print("Generated {} episodes in total ({} frames)".format(num_ep, num_frames))

            print("\nTraining VAE on rollouts...")
            sess.run(vae_input_fn_init_op)
            vae.train_on_iterator(vae_input_fn_iter)

            print("\nFormatting rollouts for State RNN...")
            episode_sequence_lengths = convert_vae_deque_to_state_rnn_deque(vae, vae_episodes_deque,
                                                                            state_rnn_episodes_deque,
                                                                            max_sequence_length,
                                                                            threads=12)

            number_of_sequences_written = reduce(lambda acc, episode: acc + len(episode), episode_sequence_lengths, 0)
            total_frames_written = reduce(lambda acc, episode: acc + sum(episode), episode_sequence_lengths, 0)

            print("Converted {} episodes to {} sequences ({} frames).".format(len(episode_sequence_lengths),
                                                                              number_of_sequences_written,
                                                                              total_frames_written))

            print("\nTraining State RNN on rollouts...")
            sess.run(state_rnn_input_fn_init_op)
            # state_rnn.train_on_iterator(state_rnn_input_fn_iter)

            while True:

                try:
                    batch_inputs, batch_targets, batch_lengths, batch_frames = sess.run(state_rnn_input_fn_iter)
                except tf.errors.OutOfRangeError:
                    print("Input_fn ended")
                    break

                prediction = batch_inputs[0, 0, :latent_dim]
                state = None

                print("used length: {}".format(batch_lengths))
                print("batch inputs, shape {} : \n{}".format(batch_inputs.shape, batch_inputs))
                print("batch targets, shape {} : \n{}".format(batch_targets.shape, batch_targets))
                for i in range(batch_lengths[0]):
                    frame = np.squeeze(vae.decode_frames(batch_inputs[:, i, :state_rnn.latent_dim]))
                    target_frame = np.squeeze(vae.decode_frames(batch_targets[:, i, :state_rnn.latent_dim]))
                    action = batch_inputs[0, i, state_rnn.latent_dim:]
                    feed_dict = {
                        state_rnn.sequence_inputs: np.expand_dims(
                            np.expand_dims(np.concatenate((np.squeeze(prediction), action), axis=0), 0), 0),
                        state_rnn.sequence_lengths: np.asarray([1])
                    }

                    if state:
                        feed_dict[state_rnn.lstm_state_in] = state

                    decoded_input = np.squeeze(vae.decode_frames(np.expand_dims(np.squeeze(prediction), 0)))
                    prediction, state = sess.run([state_rnn.output, state_rnn.lstm_state_out], feed_dict=feed_dict)
                    decoded_prediction = np.squeeze(vae.decode_frames(prediction[:, 0, ...]))

                    debug_imshow_image_with_action(frame, action, window_label='actual frame')
                    debug_imshow_image_with_action(target_frame, action, window_label='actual next frame')
                    debug_imshow_image_with_action(decoded_input, action, window_label='prediction input')
                    debug_imshow_image_with_action(decoded_prediction, action, window_label='prediction output')
                    cv2.waitKey(800)

        # iteration = 0
        # while True:
        #     try:
        #         frame, action = sess.run(vae_input_fn_iter)
        #     except tf.errors.OutOfRangeError:
        #         break
        #     print(iteration)
        #     # debug_imshow_image_with_action("episode", frame, action)
        #     # cv2.waitKey(1)
        #     iteration += 1

    #
    # print(len(episode_deque))
    # length = len(episode_deque)
    # for i in range(length):
    #     episode = episode_deque.pop()
    #     print("episode {}, length {}".format(i, len(episode[0])))
    #     for j in range(len(episode[0])):
    #         debug_imshow_image_with_action("episode {}".format(i), episode[0][j, :, :, ::-1], episode[1][j])
    #         cv2.waitKey(500)
    # print('done')
    # env.close()


if __name__ == '__main__':
    do_iterative_exploration('boxpushsimple-v0', num_env=60, num_iterations=1, latent_dim=1)
