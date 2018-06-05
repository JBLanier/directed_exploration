from directed_exploration.state_rnn import StateRNN
import argparse
import numpy as np
import os
import re
import pickle
import tensorflow as tf
import multiprocessing
from directed_exploration.vae import VAE
from directed_exploration.de_logging import init_logging
import gym
import gym_boxpush
import cv2

MAX_SEQUENCES_PER_RNN_TF_RECORD = 5


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def debug_imshow_image_with_action(frame, action, window_label='frame'):
    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (0, 10)
    font_scale = 0.3
    font_color = (0, 0, 0)
    line_type = 1

    # cv2.putText(frame, str(action),
    #             location,
    #             font,
    #             font_scale,
    #             font_color,
    #             line_type)

    cv2.imshow(window_label, frame[:, :, ::-1])


def get_numbered_tfrecord_file_names_from_directory(dir, prefix):
    dir_files = os.listdir(dir)
    dir_files = sorted(filter(lambda f: str.isdigit(re.split('[_.]+', f)[1]) and f.startswith(prefix), dir_files),
                       key=lambda f: int(re.split('[_.]+', f)[1]))
    return list(map(lambda f: os.path.join(dir, f), dir_files))


def get_rnn_tfrecord_input_fn(train_data_dir, batch_size=32, num_epochs=None):
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

        return input_sequence, target_sequence, sequence_length - 1

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


def train_state_rnn(rnn, train_data_dir):
    input_fn = get_rnn_tfrecord_input_fn(train_data_dir, batch_size=64, num_epochs=40)

    rnn.train_on_input_fn(input_fn)

    rnn.save_model()


def debug_play(rnn, vae):
    env = gym.make('boxpushsimple-v0')
    frame = env.reset()
    rnn.reset_state()

    action = [0]

    from pyglet.window import key

    def key_press(k, mod):
        if k == key.RIGHT: action[0] = 1
        if k == key.UP:    action[0] = 2
        if k == key.DOWN:  action[0] = 3
        if k == key.LEFT:  action[0] = 4

    def key_release(k, mod):
        if k == key.RIGHT and action[0] == 1: action[0] = 0
        if k == key.UP and action[0] == 2: action[0] = 0
        if k == key.DOWN and action[0] == 3: action[0] = 0
        if k == key.LEFT and action[0] == 4: action[0] = 0

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    prediction = rnn.predict_on_frames_retain_state(vae.encode_frames(np.expand_dims(frame, 0)), np.expand_dims(action, 0))

    while True:
        frame, _, _, _ = env.step(action[0])
        env.render()

        frame = frame / 255.0
        cv2.imshow("encoded_decoded", np.squeeze(vae.encode_decode_frames(np.expand_dims(frame, axis=0)))[:, :, ::-1])
        cv2.imshow("predicted_decoded", np.squeeze(vae.decode_frames(np.expand_dims(prediction[:, 0, ...], 0)))[:, :, ::-1])
        prediction = rnn.predict_on_frames_retain_state(np.expand_dims(prediction[:, 0, ...],0), np.expand_dims(action, 0))
        debug_imshow_image_with_action(window_label='orig',frame=frame, action=action)

        cv2.waitKey(1)


def debug_play_box_simple_no_vae(rnn):
    env = gym.make('boxpushsimple-v0')
    actual_frame = env.reset()
    prediction = np.reshape(env.debug_get_player_location(), [1, rnn.latent_dim])
    rnn.reset_state()

    action = np.array([0.0, 0.0])

    from pyglet.window import key

    def key_press(k, mod):
        if k == key.LEFT:  action[:] = [1, -1]
        if k == key.RIGHT: action[:] = [1, 0]

    def key_release(k, mod):
        if k == key.LEFT and np.all(action == [1, -1]):  action[:] = [0, 0]
        if k == key.RIGHT and np.all(action == [1, 0]):  action[:] = [0, 0]

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    while True:
        env.render_actual_frames()
        predicted_frame = env.debug_show_player_at_location(np.expand_dims(np.squeeze(prediction), 0))

        debug_imshow_image_with_action(actual_frame, action, window_label='actual frame')
        debug_imshow_image_with_action(predicted_frame, action, window_label='predicted frame')

        # print(prediction.shape)
        # print(action.shape)

        prediction = rnn.predict_on_frames_retain_state(np.reshape(prediction, [1, rnn.latent_dim]), np.expand_dims(action, 0))
        actual_frame, _, _, _ = env.step(action)

        cv2.waitKey(1)


def debug_from_input_fn(rnn, vae, train_data_dir):
    np.set_printoptions(threshold=np.nan)

    sess = rnn.sess

    with rnn.graph.as_default():
        with tf.name_scope("input_fn"):
            iter = get_rnn_tfrecord_input_fn(train_data_dir, batch_size=1, num_epochs=None)()

    while True:

        try:
            batch_inputs, batch_targets, batch_lengths = sess.run(iter)
        except tf.errors.OutOfRangeError:
            print("Input_fn ended")
            break

        prediction = batch_inputs[0, 0, :rnn.latent_dim]
        state = None

        print("used length: {}".format(batch_lengths))
        print("batch inputs, shape {} : \n{}".format(batch_inputs.shape, batch_inputs))
        print("batch targets, shape {} : \n{}".format(batch_targets.shape, batch_targets))
        for i in range(batch_lengths[0]):
            frame = np.squeeze(vae.decode_frames(batch_inputs[:, i, :rnn.latent_dim]))
            target_frame = np.squeeze(vae.decode_frames(batch_targets[:, i, :rnn.latent_dim]))
            action = batch_inputs[0, i, rnn.latent_dim:]
            feed_dict = {
                rnn.sequence_inputs: np.expand_dims(np.expand_dims(np.concatenate((np.squeeze(prediction), action), axis=0), 0), 0),
                rnn.sequence_lengths: np.asarray([1])
            }

            if state:
                feed_dict[rnn.lstm_state_in] = state

            decoded_input = np.squeeze(vae.decode_frames(np.expand_dims(np.squeeze(prediction), 0)))
            prediction, state = sess.run([rnn.output, rnn.lstm_state_out], feed_dict=feed_dict)
            decoded_prediction = np.squeeze(vae.decode_frames(prediction[:, 0, ...]))

            debug_imshow_image_with_action(frame, action, window_label='actual frame')
            debug_imshow_image_with_action(target_frame, action, window_label='actual next frame')
            debug_imshow_image_with_action(decoded_input, action, window_label='prediction input')
            debug_imshow_image_with_action(decoded_prediction, action, window_label='prediction output')
            cv2.waitKey(800)


def debug_boxpushsimple_no_vae_from_input_fn(rnn, train_data_dir):
    np.set_printoptions(threshold=np.nan)

    sess = rnn.sess

    with rnn.graph.as_default():
        with tf.name_scope("input_fn"):
            iter = get_rnn_tfrecord_input_fn(train_data_dir, batch_size=1, num_epochs=None)()

    env = gym.make('boxpushsimple-v0')
    env.reset()

    while True:

        try:
            batch_inputs, batch_targets, batch_lengths = sess.run(iter)
        except tf.errors.OutOfRangeError:
            print("Input_fn ended")
            break

        prediction = batch_inputs[0, 0, :rnn.latent_dim]
        state = None

        print("used length: {}".format(batch_lengths))
        print("batch inputs, shape {} : \n{}".format(batch_inputs.shape, batch_inputs))
        print("batch targets, shape {} : \n{}".format(batch_targets.shape, batch_targets))
        for i in range(batch_lengths[0]):
            frame = env.debug_show_player_at_location(batch_inputs[:, i, :rnn.latent_dim])
            target_frame = env.debug_show_player_at_location(batch_targets[:, i, :rnn.latent_dim])
            action = batch_inputs[0, i, rnn.latent_dim:]

            prediction = np.reshape(prediction, newshape=[rnn.latent_dim])

            feed_dict = {
                rnn.sequence_inputs: np.expand_dims(np.expand_dims(np.concatenate((prediction, action), axis=0), 0), 0),
                rnn.sequence_lengths: np.asarray([1])
            }

            if state:
                feed_dict[rnn.lstm_state_in] = state

            decoded_input = env.debug_show_player_at_location(np.expand_dims(np.squeeze(prediction), 0))
            prediction, state = sess.run([rnn.output, rnn.lstm_state_out], feed_dict=feed_dict)
            decoded_prediction = env.debug_show_player_at_location(prediction[:, 0, ...])

            print(frame.shape)
            print(prediction.shape)
            print(action.shape)
            debug_imshow_image_with_action(frame, action, window_label='actual frame')
            debug_imshow_image_with_action(target_frame, action, window_label='actual next frame')
            debug_imshow_image_with_action(decoded_input, action, window_label='prediction input')
            debug_imshow_image_with_action(decoded_prediction, action, window_label='prediction output')
            cv2.waitKey(50)


def main(args):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        state_rnn = StateRNN(working_dir=args.load_rnn_weights, latent_dim=1)

        if args.train_rnn:
            if args.train_data_dir:
                train_state_rnn(state_rnn, args.train_data_dir)
            else:
                print("Must specify --train-data-dir")
                exit(1)

        if args.debug_play:
            if args.load_vae_weights and args.load_rnn_weights and not args.simple_encoding:
                vae = VAE(working_dir=args.load_vae_weights, latent_dim=state_rnn.latent_dim)
                debug_play(state_rnn, vae)
            elif args.load_rnn_weights and args.simple_encoding:
                debug_play_box_simple_no_vae(state_rnn)
            else:
                print("Must specify \n--load-vae-weights=<vae weights dir> and\n --load-rnn-weights=<rnn weights dir>")
                exit(1)

        if args.debug_from_rollouts:
            if args.load_vae_weights and args.load_rnn_weights and args.train_data_dir:
                vae = VAE(working_dir=args.load_vae_weights, latent_dim=state_rnn.latent_dim)
                debug_from_input_fn(state_rnn, vae, args.train_data_dir)

            elif args.simple_encoding and args.load_rnn_weights and args.train_data_dir:
                debug_boxpushsimple_no_vae_from_input_fn(state_rnn, args.train_data_dir)

            else:
                print("Must specify \n--load-vae-weights=<vae weights dir> and\n"
                      " --load-rnn-weights=<rnn weights dir> and\n"
                      " --train-data-dir=<rollout data dir>")
                exit(1)


if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--load-rnn-weights", help="dir to load RNN weights with",
                        type=str, default=None)
    parser.add_argument("--load-vae-weights", help="dir to load VAE weights with",
                        type=str, default=None)
    parser.add_argument("--train-rnn", help="if true, train RNN",
                        action="store_true")
    parser.add_argument("--train-data-dir", help="State RNN training data location",
                        type=str)
    parser.add_argument("--debug-play", help="use controller to debug environment",
                        action="store_true")
    parser.add_argument("--debug-from-rollouts", help="use controller to debug environment",
                        action="store_true")
    parser.add_argument("--simple-encoding", help="When debugging from rollouts, assume 1d encoding is "
                                                  "actual exact location of player",
                        action="store_true")
    args = parser.parse_args()
    main(args)
