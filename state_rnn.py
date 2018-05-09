
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import os
import datetime
import cv2
from vae import VAE
from tensorflow.python import debug as tf_debug


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class StateRNN:

    def __init__(self, latent_dim=128, action_dim=2, restore_from_dir=None):
        print("RNN latent dim {} action dim {}".format(latent_dim, action_dim))
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.save_prefix = 'state_rnn'
        self.date_identifier = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_file_path = './state_rnn_model_{}'.format(self.date_identifier)
        self.tensorboard_path = './tensorboard'

        self._build_model(restore_from_dir)

        self.writer = tf.summary.FileWriter("{}/{}".format(self.tensorboard_path, self.date_identifier), self.graph)

    def _restore_model(self, from_dir):
        print("\n\nRestoring State RNN Model from {}".format(from_dir))
        # self.save_file_path = from_dir
        with self.graph.as_default():
            # self.saver = tf.train.import_meta_graph(os.path.join(self.save_file_path, self.save_prefix + '.meta'))
            self.saver.restore(self.sess, tf.train.latest_checkpoint(from_dir))

    def _build_model(self, restore_from_dir=None):

        with self.graph.as_default():
            vae_scope = 'State_RNN'
            with tf.variable_scope(vae_scope):
                variance_scaling = tf.contrib.layers.variance_scaling_initializer()
                xavier = tf.contrib.layers.xavier_initializer()

                # Building the rnn

                self.sequence_inputs = tf.placeholder(tf.float32, shape=[None, None, self.latent_dim + self.action_dim], name='z_and_action_inputs')
                # self.sequence_inputs = tf.Print(self.sequence_inputs,[self.sequence_inputs, tf.shape(self.sequence_inputs)], "Sequence inputs: ")

                self.sequence_targets = tf.placeholder(tf.float32, shape=[None, None, self.latent_dim], name='z_targets')
                # self.sequence_targets = tf.Print(self.sequence_targets,[self.sequence_targets, tf.shape(self.sequence_targets)], "Sequence targets: ")

                input_shape = tf.shape(self.sequence_inputs)


                def length(sequence):
                    with tf.name_scope('computed_length'):
                        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
                        length = tf.reduce_sum(used, 1)
                        length = tf.cast(length, tf.int32)
                        return length

                self.computed_lengths = length(self.sequence_inputs)
                self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])

                # self.computed_lengths = tf.Print(self.computed_lengths,[self.computed_lengths, tf.shape(self.computed_lengths)], "Computed Lengths: ")
                # self.sequence_lengths = tf.Print(self.sequence_lengths,[self.sequence_lengths, tf.shape(self.sequence_lengths)], "Passed Sequence Lengths: ")

                with tf.control_dependencies([tf.assert_equal(self.computed_lengths, self.sequence_lengths),
                                              tf.assert_equal(self.computed_lengths, length(self.sequence_targets))]):

                    lstm_cell = tf.nn.rnn_cell.LSTMCell(256)
                    lstm_output, self.lstm_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                                     inputs=self.sequence_inputs,
                                                                     sequence_length=self.sequence_lengths,
                                                                     dtype=tf.float32)

                # lstm_output = tf.Print(lstm_output,[lstm_output[0,2,4], tf.shape(lstm_output)], "lstm_output: ")
                # reshape lstm output from (batch_size, max_seq_length, ...) to (batch_size * max_seq_length, ...)
                lstm_output_for_dense = tf.reshape(lstm_output, shape=[input_shape[0] * input_shape[1], lstm_output.shape[-1]])
                # lstm_output_for_dense = tf.Print(lstm_output_for_dense,[lstm_output_for_dense,tf.shape(lstm_output_for_dense)], "lstm_output reshaped for dense: ")
                reshaped_again = tf.reshape(lstm_output_for_dense, shape=[input_shape[0], input_shape[1], lstm_output_for_dense.shape[-1]], name='reshape_again')
                # lstm_output_for_dense = tf.Print(lstm_output_for_dense, [reshaped_again[0,2,4], tf.shape(reshaped_again)], "lstm output put back")

                dense1 = tf.layers.dense(inputs=lstm_output_for_dense,
                                        units=1024,
                                        activation=tf.nn.relu,
                                        kernel_initializer=variance_scaling)

                dense2 = tf.layers.dense(inputs=dense1,
                                         units=self.latent_dim,
                                         activation=None,
                                         kernel_initializer=variance_scaling,
                                         bias_initializer=variance_scaling)
                # dense2 = tf.Print(dense2,[dense2,tf.shape(dense2)], "dense 2 output: ")

                # reshape dense output back to (batch_size, max_seq_length, ...)
                self.output = tf.reshape(dense2, shape=[-1, tf.shape(self.sequence_inputs)[1], dense2.shape[-1]])

                # self.output = tf.Print(self.output, [self.output, tf.shape(self.output)], "dense 2 output reshaped: ")
                # self.output = tf.Print(self.output, [self.computed_lengths, tf.shape(self.computed_lengths), self.sequence_lengths, tf.shape(self.sequence_lengths)], 'computed and given sequence lengths')

                with tf.name_scope('mse_loss'):
                    # Compute Squared Error for each frame
                    frame_squared_errors = tf.square(self.output - self.sequence_targets)
                    frame_mean_squared_errors = tf.reduce_mean(frame_squared_errors, axis=2)
                    mask = tf.sign(tf.reduce_max(tf.abs(self.sequence_targets), 2))
                    masked_frame_mse_errors = frame_mean_squared_errors * mask

                    # Average Over actual Sequence Lengths
                    with tf.control_dependencies([tf.assert_equal(tf.cast(tf.reduce_sum(mask, 1), dtype=tf.int32), self.sequence_lengths)]):
                        mse_over_sequences = tf.reduce_sum(masked_frame_mse_errors, 1) / tf.cast(self.sequence_lengths, dtype=tf.float32)

                    mse_over_batch = tf.reduce_mean(mse_over_sequences)
                    self.mse_loss = mse_over_batch
                    tf.summary.scalar('mse_loss', self.mse_loss)

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
            self.train_op = self.optimizer.minimize(self.mse_loss)
            # self.check_op = tf.add_check_numerics_ops()
            # print("\n\nCollection: {}".format(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vae_scope)))
            self.tf_summaries_merged = tf.summary.merge_all()
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vae_scope),
                                        max_to_keep=None)
            self.init = tf.global_variables_initializer()
        #
        if restore_from_dir:
            self._restore_model(restore_from_dir)
        else:
            self.sess.run(self.init)

    def train_on_input_fn(self, input_fn, steps=None):
        # vae = VAE(restore_from_dir='vae_model_20180507133856')

        sess = self.sess

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        with self.graph.as_default():
            # Run the initializer

            with tf.name_scope("input_fn"):
                iter = input_fn()

            step = 1
            while True:

                try:
                    batch_inputs, batch_targets, batch_lengths = sess.run(iter)
                except tf.errors.OutOfRangeError:
                    print("Input_fn ended at step {}".format(step))
                    break

                # Train
                feed_dict = {
                    self.sequence_inputs: batch_inputs,
                    self.sequence_targets: batch_targets,
                    self.sequence_lengths: batch_lengths
                             }

                _, loss, summaries = sess.run([self.train_op, self.mse_loss, self.tf_summaries_merged], feed_dict=feed_dict)

                # for target in np.squeeze(targets)[0]:
                #     cv2.imshow("target",np.squeeze(vae.decode_frames(np.expand_dims(target,0)))[:,:,::-1])
                #     cv2.waitKey(300)
                # self.writer.add_summary(summaries, step)

                if step % 20 == 0 or step == 1:
                    print('Step %i, Loss: %f' % (step, loss))

                if steps and step >= steps:
                    print("Completed {} steps".format(steps))
                    break

                step += 1
    #
    # def encode_frames(self, float_frames):
    #     return self.sess.run(self.z_encoded, feed_dict={self.x: float_frames})
    #
    # def decode_frames(self, z_codes):
    #     return self.sess.run(self.decoded_given, feed_dict={self.z_given: z_codes})
    #
    # def encode_decode_frames(self, float_frames):
    #     return self.sess.run(self.decoded_encoded, feed_dict={self.x: float_frames})
    #
    # def save_model(self, write_dir=None, write_meta_graph=True):
    #     if write_dir:
    #         self.save_file_path = write_dir
    #     if not os.path.exists(self.save_file_path):
    #         os.makedirs(self.save_file_path, exist_ok=True)
    #
    #     save_path = self.saver.save(self.sess, os.path.join(self.save_file_path, self.save_prefix),
    #                                 write_meta_graph=write_meta_graph)
    #     print("VAE Model saved in path: {}".format(save_path))
    #
    # def __del__(self):
    #     if self.writer:
    #         self.writer.close()
    #
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     if self.writer:
    #         self.writer.close()



if __name__ == '__main__':
    rnn = StateRNN()
    print("yo")
