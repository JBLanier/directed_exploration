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

import cv2

from anticipator import AnticipatorRNN

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
    Create a wrapped, monitored SubprocVecEnv for Atari.
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
    for _ in range(num_episodes_per_environment):
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
                    if random.random() < 1 / (max_episode_length*2) and episode_lengths[env_index] >= 2:
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
        print("generated episodes with lengths: {}".format(episode_lengths))


def get_vae_deque_input_fn(episode_deque, batch_size, max_episode_length):

    def episode_generator():
        while True:
            try:
                ep = episode_deque.pop()
                yield ep
            except IndexError:
                return

    def slice_and_shuffle_fn(x1, x2):
        episode_length = tf.cast(tf.shape(x1)[0], tf.int64)
        return tf.data.Dataset.from_tensor_slices((x1, x2)).shuffle(buffer_size=episode_length)

    def input_fn():
        episodes_dataset = tf.data.Dataset.from_generator(generator=episode_generator,
                                                          output_types=(tf.float32, tf.float32),
                                                          output_shapes=(tf.TensorShape([None, 64, 64, 3]),
                                                                         tf.TensorShape([None, 2])))
        cycle_length = 30
        dataset = episodes_dataset.interleave(map_func=slice_and_shuffle_fn,
                                              cycle_length=cycle_length,
                                              block_length=1)

        dataset = dataset.shuffle(buffer_size=max_episode_length * 2)

        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=10)

        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator.initializer

    return input_fn


def do_iterative_exploration(env_id, num_env, num_iterations):
    # env = make_boxpush_env(env_id, num_env, seed)
    # obs = env.reset()
    #
    # print(obs.shape)
    # for i in range(len(obs)):
    #     print(obs[i,:,:,::-1].shape)
    #     cv2.imshow(("env {}".format(i+1)),obs[i,:,:,::-1])
    # cv2.waitKey(1)
    #
    # actions = [[0, 0], [1, 0], [1, -1]]
    # action_index = 0
    #
    # while True:
    #
    #     if random.random() < 0.05:
    #         action_index = random.randint(0, len(actions) - 1)
    #     action = [np.asarray(actions[action_index]) for i in range(num_env)]
    #
    #     obs, _, _, _ = env.step(np.array(action))
    #
    #     # frame = frame/255.0
    #
    #     for i in range(len(obs)):
    #         cv2.imshow(("env {}".format(i + 1)), obs[i, :, :, ::-1])
    #     cv2.waitKey(1)

    env = make_boxpush_env(env_id, num_env, seed)
    anticipator = AnticipatorRNN()
    num_episodes_per_environment = 1
    max_episode_length = 2000
    episode_deque = deque()

    generate_rollouts_on_anticipator_policy_into_deque(episode_deque, anticipator, env, num_env,
                                                       num_episodes_per_environment, max_episode_length, ACTIONS)

    vae_input_fn = get_vae_deque_input_fn(episode_deque, batch_size=256, max_episode_length=max_episode_length)

    input_fn_graph = tf.Graph()
    input_fn_sess = tf.Session(graph=input_fn_graph)
    with input_fn_graph.as_default():
        vae_input_fn_iter, vae_input_fn_init_op = vae_input_fn()

    input_fn_sess.run(vae_input_fn_init_op)

    i=0
    while True:
        try:
            frame, action = input_fn_sess.run(vae_input_fn_iter)
        except tf.errors.OutOfRangeError:
            break
        print(i)
        # debug_imshow_image_with_action("episode", frame, action)
        # cv2.waitKey(1)
        i += 1

    generate_rollouts_on_anticipator_policy_into_deque(episode_deque, anticipator, env, num_env,
                                                       num_episodes_per_environment, max_episode_length, ACTIONS)


    input_fn_sess.run(vae_input_fn_init_op)


    i = 0
    while True:
        try:
            frame, action = input_fn_sess.run(vae_input_fn_iter)
        except tf.errors.OutOfRangeError:
            break
        print(i)
        # debug_imshow_image_with_action("episode", frame, action)
        # cv2.waitKey(1)
        i += 1

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
    do_iterative_exploration('boxpushsimple-v0', num_env=12, num_iterations=10000)
