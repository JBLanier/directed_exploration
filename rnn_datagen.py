import pickle
import numpy as np
import os
import tensorflow as tf
import re
import cv2
from vae import VAE

# Dimensionality of actions to read from vae tf_records and pass to rnn
ACTION_LENGTH = 2


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


def convert_vae_record_to_rnn_records(vae_model_dir, vae_data_read_dir, rnn_data_write_dir, max_sequence_length):
    print('\n' + ("_" * 20) + '\n')
    print("Reading from {}\nWriting to {}\nMax sequence length is {}\nVAE model is from {}".format(
        vae_data_read_dir, rnn_data_write_dir, max_sequence_length, vae_model_dir))
    print(("_" * 20) + '\n')

    if not os.path.exists(rnn_data_write_dir):
        os.makedirs(rnn_data_write_dir)

    vae_tf_records_files = get_numbered_tfrecord_file_names_from_directory(dir=vae_data_read_dir, prefix='vae')

    vae = VAE(restore_from_dir=vae_model_dir)

    number_of_sequences_written = 0
    total_frames_written = 0

    for episode_index in range(len(vae_tf_records_files)):

        read_file_name = vae_tf_records_files[episode_index]
        record_iterator = tf.python_io.tf_record_iterator(read_file_name)
        all_of_file_read = False

        write_file_name = os.path.join(rnn_data_write_dir, 'rnn_{}.tfrecords'.format(episode_index))
        tf_record_writer = tf.python_io.TFRecordWriter(write_file_name)

        sequence_lengths = []

        while not all_of_file_read:

            # If a generated sequence is less than max_sequence_length, the rest of it will be zeros.
            sequence = np.zeros(shape=(max_sequence_length, vae.latent_dim + ACTION_LENGTH), dtype=np.float32)

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
                encoded_frame = vae.encode_frames(np.expand_dims(raw_frame, axis=0))

                # Debug Visualization ##
                # decoded = vae.decode_frames(encoded_frame)[0]
                # debug_imshow_image_with_action(raw_frame, action, window_label='original')
                # debug_imshow_image_with_action(decoded, action, window_label='encoded_decoded')
                # cv2.waitKey(30)
                ##

                sample_entry = np.concatenate((encoded_frame[0], action), axis=0)
                sequence[sequence_index] = sample_entry
                sequence_length += 1

            # Must have at least two frames to be suitable for RNN training
            if sequence_length >= 2:

                sequence_lengths.append(sequence_length)

                sequence_bytes = pickle.dumps(sequence)

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
        number_of_sequences_written += len(sequence_lengths)
        total_frames_written += sum(sequence_lengths)

    print("Done, wrote {} episodes with {} sequences ({} frames).".format(len(vae_tf_records_files),
                                                                          number_of_sequences_written,
                                                                          total_frames_written))


if __name__ == "__main__":

    convert_vae_record_to_rnn_records(vae_model_dir='vae_model_20180507133856',
                                      vae_data_read_dir='vae_tf_records',
                                      rnn_data_write_dir='rnn_tf_records',
                                      max_sequence_length=200)
