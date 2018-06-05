from directed_exploration.vae import VAE
from directed_exploration.state_rnn import StateRNN
from directed_exploration.utils.data_util import convertToOneHot
import os
import re
import tensorflow as tf
import pickle
import numpy as np
import math
import multiprocessing
import logging

import cv2

logger = logging.getLogger(__name__)


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


def get_numbered_tfrecord_file_names_from_directory(dir, prefix):
    dir_files = os.listdir(dir)
    dir_files = sorted(filter(lambda f: str.isdigit(re.split('[_.]+', f)[1]) and f.startswith(prefix), dir_files),
                       key=lambda f: int(re.split('[_.]+', f)[1]))
    return list(map(lambda f: os.path.join(dir, f), dir_files))


def get_validation_tfrecord_input_fn(allowed_action_space):

    def decode_pickled_np_array(np_bytes):
        return pickle.loads(np_bytes).astype(np.float32)

    def parse_fn(example):
        example_fmt = {
            "action_at_frame": tf.FixedLenFeature([allowed_action_space.n], tf.float32),
            "frame_bytes": tf.FixedLenFeature([], tf.string)
        }

        parsed = tf.parse_single_example(example, example_fmt)

        action = parsed["action_at_frame"]
        frame_bytes = parsed["frame_bytes"]

        frame = tf.py_func(func=decode_pickled_np_array, inp=[frame_bytes],
                           Tout=tf.float32, stateful=False, name='decode_np_bytes')

        return frame, action

    def input_fn():
        file_name_placeholder = tf.placeholder(dtype=tf.string, shape=[], name='file_name')
        dataset = tf.data.TFRecordDataset(file_name_placeholder)
        dataset = dataset.map(map_func=parse_fn, num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.batch(1000)
        dataset = dataset.prefetch(buffer_size=4)
        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator.initializer, file_name_placeholder

    return input_fn


def validate_vae_state_rnn_pair_on_tf_records(data_dir, vae, state_rnn, sess, allowed_action_space):
    with sess.as_default():
        with tf.name_scope('input_functions'):
            val_input_fn_iter, val_input_fn_init_op, file_name_placeholder = get_validation_tfrecord_input_fn(allowed_action_space)()

        tfrecord_prefix = 'vae'
        tfrecord_files = get_numbered_tfrecord_file_names_from_directory(data_dir, tfrecord_prefix)

        if len(tfrecord_files) <= 0:
            raise FileNotFoundError("No usable tfrecords with prefix \'{}\' were found at {}".format(
                tfrecord_prefix, data_dir)
            )

        avg_loss = 0
        frames_tested_on = 0

        for file_name in tfrecord_files:
            sess.run(val_input_fn_init_op, feed_dict={file_name_placeholder: file_name})

            while True:
                try:
                    batch_frames, batch_actions = sess.run(val_input_fn_iter)
                except tf.errors.OutOfRangeError:
                    break

                # for action in batch_actions:
                #     assert allowed_action_space.contains(action)

                input_frames = batch_frames[:-1]
                input_actions = batch_actions[:-1]

                input_frames_encoded = vae.encode_frames(input_frames)

                frame_predictions_encoded = state_rnn.predict_on_sequences(
                    z_sequences=np.expand_dims(input_frames_encoded, 0),
                    action_sequences=np.expand_dims(input_actions, 0),
                    sequence_lengths=[len(input_frames)])

                losses = vae.get_loss_for_decoded_frames(z_codes=np.reshape(frame_predictions_encoded, newshape=[-1, vae.latent_dim]),
                                                         target_frames=batch_frames[1:])
                sequence_mean_loss = np.mean(losses)

                logger.debug('loss of {} on sequence of length {}'.format(sequence_mean_loss, len(losses)))

                new_total_frames = frames_tested_on + len(losses)
                avg_loss = (avg_loss*frames_tested_on + sequence_mean_loss * len(losses)) / new_total_frames
                frames_tested_on = new_total_frames

                state_rnn.reset_state()

        return avg_loss


if __name__ == '__main__':
    from directed_exploration.de_logging import init_logging
    init_logging()

    sess = tf.Session()

    validate_vae_state_rnn_pair_on_tf_records(data_dir='/media/jb/m2/boxpushsimple_validation_rollouts/',
                                              vae=VAE(working_dir='itexplore_20180527004253_trained_on_colorchange', sess=sess, latent_dim=1),
                                              state_rnn=StateRNN(working_dir='itexplore_20180527004253_trained_on_colorchange', sess=sess, latent_dim=1, action_dim=2),
                                              sess=sess)