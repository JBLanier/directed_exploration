import gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
import time
import multiprocessing
from functools import reduce
import os
import cv2

from directed_exploration.anticipator import AnticipatorRNN
from directed_exploration.vae import VAE
from directed_exploration.state_rnn import StateRNN
from directed_exploration.utils.env_util import RecordWriteSubprocVecEnv
from directed_exploration import simulator_data_ops as sdo
import logging

logger = logging.getLogger(__name__)

seed = 42

ACTIONS = [[0, 0], [1, 0], [1, 0.5], [1, -0.5], [1, 1]]
ACTION_DIM = 2

HEATMAP_FOLDER_NAME = 'heatmap_records'

np.set_printoptions(threshold=np.nan)


def generate_rollouts_on_anticipator_policy_into_deque(episode_deque, anticipator, env, num_env,
                                                       num_episodes_per_environment, max_episode_length,
                                                       available_actions, do_random_policy=False):
    total_frames = 0
    for _ in range(num_episodes_per_environment):
        start = time.time()
        obs = env.reset() / 255.0
        anticipator.reset_saved_state()

        episode_frames = [np.empty(shape=(max_episode_length, 64, 64, 3), dtype=np.float32) for _ in range(num_env)]
        episode_actions = [np.empty(shape=(max_episode_length, ACTION_DIM), dtype=np.float32) for _ in range(num_env)]
        episode_lengths = np.zeros(shape=num_env, dtype=np.uint32)
        dones = np.full(num_env, False)

        for episode_frame_index in range(max_episode_length):

            # Predict loss amounts (action scores) for each action/observation
            action_scores = anticipator.predict_on_frame_batch_retain_state(
                frames=np.repeat(obs, [len(available_actions)] * num_env, 0),
                actions=np.asarray(available_actions * num_env))
            action_scores = np.reshape(action_scores, newshape=[num_env, len(available_actions)])
            # Normalize action scores to be probabilities
            action_scores = np.maximum(action_scores, 0.0001)
            action_scores = action_scores / np.sum(action_scores, axis=1)[:, None]
            # Sample next actions to take from probabilities
            if do_random_policy:
                action_indexes = [np.random.choice(len(available_actions)) for _ in
                                  action_scores]
            else:
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
        logger.debug("generated episodes with lengths: {}".format(episode_lengths))
        running_time = end - start
        logger.debug("took {} seconds, per-environment efficiency is {}".format(running_time, num_env / running_time))
        total_frames += sum(episode_lengths)

    return num_episodes_per_environment * num_env, total_frames


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
    return RecordWriteSubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def convert_state_rnn_deque_to_anticipator_deque(vae, state_rnn, state_rnn_deque, anticipator_deque,
                                                 threads=multiprocessing.cpu_count()):
    def add_episode_to_sequence_deque(episode_ticket_num):

        code_sequences, action_sequences, sequence_lengths, frame_sequences = state_rnn_deque.pop()
        # logger.info("\n\nFrame sequnces shape: {}".format(frame_sequences.shape))
        # logger.info("sequence lengths:\n{}".format(sequence_lengths))
        input_code_sequences = code_sequences[:, :-1, :]
        input_action_sequences = action_sequences[:, :-1, :]

        for i, sequence_length in enumerate(sequence_lengths):
            if sequence_length < len(code_sequences[i]):
                input_code_sequences[i, sequence_length - 1] = 0
                input_action_sequences[i, sequence_length - 1] = 0

        num_sequences = code_sequences.shape[0]
        max_io_sequence_length = len(input_code_sequences[0])

        # logger.info("action sequences:\n{}".format(action_sequences))
        # logger.info("input action sequences:\n{}".format(input_action_sequences))
        #
        # logger.info("code sequences: \n{}".format(code_sequences))
        # logger.info("input code sequences:\n{}".format(input_code_sequences))

        predictions = state_rnn.predict_on_sequences(input_code_sequences, input_action_sequences,
                                                     sequence_lengths - 1)

        # logger.info("predictions: \n{}".format(predictions))

        predictions = predictions.reshape((num_sequences * max_io_sequence_length, state_rnn.latent_dim))

        target_frame_sequences_reshaped = frame_sequences[:, 1:, ...].reshape((num_sequences * max_io_sequence_length,
                                                                       64, 64, 3))

        losses = vae.get_loss_for_decoded_frames(z_codes=predictions, target_frames=target_frame_sequences_reshaped)
        losses = losses.reshape(num_sequences, max_io_sequence_length)

        input_frame_sequences = frame_sequences[:, :-1, ...]
        for i, sequence_length in enumerate(sequence_lengths):
            losses[i, sequence_length - 1:] = 0
            if sequence_length < len(code_sequences[i]):
                input_frame_sequences[i, sequence_length - 1] = 0

            assert np.all(input_frame_sequences[i, sequence_length - 1:] == 0)
            assert np.all(input_action_sequences[i, sequence_length - 1:] == 0)
            assert not np.all(input_frame_sequences[i, :sequence_length - 1] == 0)

        anticipator_deque.append((input_frame_sequences, input_action_sequences, losses, sequence_lengths - 1))

        return sequence_lengths

    with multiprocessing.pool.ThreadPool(processes=threads) as pool:
        episode_sequence_lengths = pool.map(func=add_episode_to_sequence_deque,
                                            iterable=range(len(state_rnn_deque)))

    return episode_sequence_lengths


def do_greedy_exploration(env_id, num_env, num_iterations, latent_dim, working_dir, num_episodes_per_environment,
                          max_episode_length, max_sequence_length, validation_data_dir=None, do_random_policy=False):

    logger.info('Iterative Exploration on {}'.format(env_id))
    logger.info('{} iterations over {} environments (with {} episodes per env per iteration).'.format(
        num_iterations,
        num_env,
        num_episodes_per_environment))
    logger.info('Max episode length {}, Max sequence length {}'.format(max_episode_length, max_sequence_length))
    if validation_data_dir:
        logger.info("Validation data dir: {}".format(validation_data_dir))
    else:
        logger.info("No validation data provided.")
    if do_random_policy:
        logger.warning("USING -RANDOM- POLICY INSTEAD OF ANTICIPATOR POLICY")

    env = make_boxpush_env(env_id, num_env, seed)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        summary_writer = tf.summary.FileWriter(working_dir)
        anticipator = AnticipatorRNN(working_dir=working_dir, action_dim=ACTION_DIM, summary_writer=summary_writer)
        vae = VAE(latent_dim=latent_dim, working_dir=working_dir, summary_writer=summary_writer)
        state_rnn = StateRNN(latent_dim=latent_dim, action_dim=ACTION_DIM, working_dir=working_dir,
                             summary_writer=summary_writer)
        vae_episodes_deque = deque()
        state_rnn_episodes_deque = deque()
        anticipator_deque = deque()

        vae_input_fn = sdo.get_vae_deque_input_fn(train_deque=vae_episodes_deque, batch_size=256,
                                                  max_episode_length=max_episode_length)

        state_rnn_input_fn = sdo.get_state_rnn_deque_input_fn(state_rnn_episodes_deque=state_rnn_episodes_deque,
                                                          batch_size=64, max_episode_length=max_episode_length,
                                                          latent_dim=latent_dim,
                                                          max_sequence_length=max_sequence_length, num_epochs=5)

        anticipator_input_fn = sdo.get_anticipator_input_fn(anticipator_deque=anticipator_deque, batch_size=16,
                                                        max_sequence_length=max_sequence_length, num_epochs=3)
        with tf.variable_scope('input_functions'):
            anticipator_input_fn_iter, anticipator_input_fn_init_op = anticipator_input_fn()
            vae_input_fn_iter, vae_input_fn_init_op = vae_input_fn()
            state_rnn_input_fn_iter, state_rnn_input_fn_init_op = state_rnn_input_fn()

        total_episodes_seen = 0
        total_frames_seen = 0

        for iteration in range(1, num_iterations+1):
            logger.info("_" * 20)
            logger.info("Iteration {}\n".format(iteration))
            logger.debug("Exploring and generating rollouts...")

            iteration_heatmap_record_prefix = "it{}".format(iteration)
            env.set_record_write(write_dir=os.path.join(working_dir, HEATMAP_FOLDER_NAME),
                                 prefix=iteration_heatmap_record_prefix)

            num_ep, num_frames = generate_rollouts_on_anticipator_policy_into_deque(vae_episodes_deque, anticipator,
                                                                                    env, num_env,
                                                                                    num_episodes_per_environment,
                                                                                    max_episode_length, ACTIONS,
                                                                                    do_random_policy=do_random_policy)
            logger.debug("Generated {} episodes in total ({} frames)".format(num_ep, num_frames))

            logger.debug("Creating Heatmap...")
            heatmap_save_location = generate_boxpush_heatmap_from_npy_records(
                directory=os.path.join(working_dir, HEATMAP_FOLDER_NAME),
                file_prefix=iteration_heatmap_record_prefix,
                delete_records=True)
            logger.debug("Heatmap saved to {}".format(heatmap_save_location))

            logger.debug("Training VAE on rollouts...")
            sess.run(vae_input_fn_init_op)
            vae.train_on_iterator(vae_input_fn_iter)

            logger.debug("Formatting rollouts for State RNN...")
            episode_sequence_lengths = sdo.convert_vae_deque_to_state_rnn_deque(vae, vae_episodes_deque,
                                                                            state_rnn_episodes_deque,
                                                                            max_sequence_length)

            it_num_of_sequences_written = reduce(lambda acc, episode: acc + len(episode), episode_sequence_lengths, 0)
            it_total_frames_written = reduce(lambda acc, episode: acc + sum(episode), episode_sequence_lengths, 0)

            logger.debug("Converted {} episodes to {} sequences ({} frames).".format(len(episode_sequence_lengths),
                                                                              it_num_of_sequences_written,
                                                                              it_total_frames_written))

            logger.debug("Training State RNN on rollouts...")
            sess.run(state_rnn_input_fn_init_op)
            state_rnn.train_on_iterator(state_rnn_input_fn_iter)

            if validation_data_dir:
                logger.debug("Validating VAE/State RNN Combo on trajectories from {}".format(validation_data_dir))
                validation_loss = validate_vae_state_rnn_pair_on_tf_records(data_dir=validation_data_dir,
                                                                            vae=vae, state_rnn=state_rnn, sess=sess,
                                                                            allowed_actions=ACTIONS)
                logger.debug('Total Validation Loss = {}'.format(validation_loss))
                val_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='simulator_val_loss',
                                                                      simple_value=validation_loss)])
                summary_writer.add_summary(val_loss_summary, global_step=iteration)

            logger.debug("Formatting rollout raw input frames with prediction reconstruction loss for Anticipator RNN...")
            episode_sequence_lengths = sdo.convert_state_rnn_deque_to_anticipator_deque(vae, state_rnn, state_rnn_episodes_deque,
                                                                                    anticipator_deque)

            it_num_of_sequences_written = reduce(lambda acc, episode: acc + len(episode), episode_sequence_lengths, 0)
            it_total_frames_written = reduce(lambda acc, episode: acc + sum(episode), episode_sequence_lengths, 0)

            logger.debug("Converted {} episodes to {} sequences ({} frames).".format(len(episode_sequence_lengths),
                                                                              it_num_of_sequences_written,
                                                                              it_total_frames_written))

            logger.debug("Training Anticipator on rollouts...")
            sess.run(anticipator_input_fn_init_op)
            anticipator.train_on_iterator(anticipator_input_fn_iter)
            anticipator_deque.clear()

            if iteration % 2 == 0 or iteration == 1:
                logger.info("Saving...")
                vae.save_model()
                state_rnn.save_model()
                anticipator.save_model()

            total_episodes_seen += num_ep
            total_frames_seen += num_frames

            logger.info("Total episodes seen so far: {}".format(total_episodes_seen))
            logger.info("Total frames: {}".format(total_frames_seen))

        logger.info("Done.")
        vae.save_model()
        state_rnn.save_model()
        env.close()

        # Debug visualize state rnn input fn with batch size 1
        # while True:
        #
        #     try:
        #         batch_inputs, batch_targets, batch_lengths, batch_frames = sess.run(state_rnn_input_fn_iter)
        #     except tf.errors.OutOfRangeError:
        #         logger.info("Input_fn ended")
        #         break
        #
        #     prediction = batch_inputs[0, 0, :latent_dim]
        #     state = None
        #
        #     logger.info("batch lengths: {}".format(batch_lengths))
        #     # logger.info("batch inputs shape {} : \n{}".format(batch_inputs.shape, batch_inputs))
        #     # logger.info("batch targets shape {} : \n{}".format(batch_targets.shape, batch_targets))
        #     for i in range(batch_lengths[0]):
        #         raw_frame = np.squeeze(batch_frames[:, i, ...])
        #         raw_target_frame = np.squeeze(batch_frames[:, i+1, ...])
        #         vae_frame = np.squeeze(vae.decode_frames(batch_inputs[:, i, :state_rnn.latent_dim]))
        #         vae_target_frame = np.squeeze(vae.decode_frames(batch_targets[:, i, :]))
        #         action = batch_inputs[0, i, state_rnn.latent_dim:]
        #         feed_dict = {
        #             state_rnn.sequence_inputs: np.expand_dims(
        #                 np.expand_dims(np.concatenate((np.reshape(prediction, [latent_dim]), action), axis=0), 0), 0),
        #             state_rnn.sequence_lengths: np.asarray([1])
        #         }
        #
        #         if state:
        #             feed_dict[state_rnn.lstm_state_in] = state
        #
        #         decoded_input = np.squeeze(vae.decode_frames(np.expand_dims(np.reshape(prediction, [latent_dim]), 0)))
        #         prediction, state = sess.run([state_rnn.output, state_rnn.lstm_state_out], feed_dict=feed_dict)
        #         decoded_prediction = np.squeeze(vae.decode_frames(prediction[:, 0, ...]))
        #
        #         cv2.imshow('raw actual frame', raw_frame[:,:,::-1])
        #         cv2.imshow('raw actual next frame ', raw_target_frame[:,:,::-1])
        #         debug_imshow_image_with_action('decoded actual frame', vae_frame, action)
        #         debug_imshow_image_with_action('decoded actual next frame ', vae_target_frame, action)
        #         debug_imshow_image_with_action('prediction input', decoded_input, action)
        #         debug_imshow_image_with_action('prediction output', decoded_prediction, action)
        #         cv2.waitKey(800)

        # iteration = 0
        # while True:
        #     try:
        #         frame, action = sess.run(vae_input_fn_iter)
        #     except tf.errors.OutOfRangeError:
        #         break
        #     logger.info(iteration)
        #     # debug_imshow_image_with_action("episode", frame, action)
        #     # cv2.waitKey(1)
        #     iteration += 1

    #
    # logger.info(len(episode_deque))
    # length = len(episode_deque)
    # for i in range(length):
    #     episode = episode_deque.pop()
    #     logger.info("episode {}, length {}".format(i, len(episode[0])))
    #     for j in range(len(episode[0])):
    #         debug_imshow_image_with_action("episode {}".format(i), episode[0][j, :, :, ::-1], episode[1][j])
    #         cv2.waitKey(500)
    # logger.info('done')
    # env.close()
