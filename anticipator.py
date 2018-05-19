import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import os
import datetime
import cv2
from vae import VAE
from model import Model
from tensorflow.python import debug as tf_debug


class AnticipatorRNN(Model):

    def __init__(self, action_dim=2, restore_from_dir=None, sess=None, graph=None):
        print("Anticipator, action dim {}".format(action_dim))

        self.action_dim = action_dim
        self.saved_state = None
        save_prefix = 'anticipator_rnn'

        super().__init__(save_prefix, restore_from_dir, sess=sess, graph=graph)

    def _build_model(self, restore_from_dir=None):

        with self.graph.as_default():
            rnn_scope = 'Anticipator_RNN'
            with tf.variable_scope(rnn_scope):
                variance_scaling = tf.contrib.layers.variance_scaling_initializer()
                xavier = tf.contrib.layers.xavier_initializer()

                # Building the rnn

                self.frame_inputs = tf.placeholder(tf.float32, shape=[None, None, 64, 64, 3], name='frame_inputs')
                self.action_inputs = tf.placeholder(tf.float32, shape=[None, None, self.action_dim],
                                                    name='action_inputs')
                # self.sequence_inputs = tf.Print(self.sequence_inputs,[self.sequence_inputs, tf.shape(self.sequence_inputs)], "Sequence inputs: ")

                self.loss_targets = tf.placeholder(tf.float32, shape=[None, None, 1], name='loss_targets')
                # self.sequence_targets = tf.Print(self.sequence_targets,[self.sequence_targets, tf.shape(self.sequence_targets)], "Sequence targets: ")

                frame_input_shape = tf.shape(self.frame_inputs)

                def length(sequence):
                    with tf.name_scope('computed_length'):
                        sequence_shape = tf.shape(sequence)
                        used = tf.sign(tf.reduce_max(tf.abs(tf.reshape(sequence,
                                                                       shape=[sequence_shape[0], sequence_shape[1],
                                                                              tf.reduce_sum(sequence_shape[2:])],name='j') ), 2))
                        length = tf.reduce_sum(used, 1)
                        length = tf.cast(length, tf.int32)
                        return length

                self.computed_lengths = length(self.frame_inputs)
                self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])

                # self.computed_lengths = tf.Print(self.computed_lengths,[self.computed_lengths, tf.shape(self.computed_lengths)], "Computed Lengths: ")
                # self.sequence_lengths = tf.Print(self.sequence_lengths,[self.sequence_lengths, tf.shape(self.sequence_lengths)], "Passed Sequence Lengths: ")

                reshaped_frame_input_for_cnn = tf.reshape(self.frame_inputs,
                                                          shape=[frame_input_shape[0] * frame_input_shape[1],
                                                                 64,
                                                                 64,
                                                                 3], name='reshaped_frame_input_for_cnn')

                encode_1 = tf.layers.Conv2D(filters=32, kernel_size=4, strides=2,
                                            padding='valid', activation=tf.nn.relu,
                                            kernel_initializer=variance_scaling,
                                            name='encode_1')(reshaped_frame_input_for_cnn)
                encode_2 = tf.layers.Conv2D(filters=64, kernel_size=4, strides=2,
                                            padding='valid', activation=tf.nn.relu,
                                            kernel_initializer=variance_scaling,
                                            name='encode_2')(encode_1)
                encode_3 = tf.layers.Conv2D(filters=128, kernel_size=4, strides=2,
                                            padding='valid', activation=tf.nn.relu,
                                            kernel_initializer=variance_scaling,
                                            name='encode_3')(encode_2)
                encode_4 = tf.layers.Conv2D(filters=256, kernel_size=4, strides=2,
                                            padding='valid', activation=tf.nn.relu,
                                            kernel_initializer=variance_scaling,
                                            name='encode_4')(encode_3)

                final_encoding_shape = encode_4.shape

                encode_4_reshape_for_lstm = tf.reshape(encode_4,
                                                       shape=[frame_input_shape[0],
                                                              frame_input_shape[1],
                                                              final_encoding_shape[1]*final_encoding_shape[2]*final_encoding_shape[3]])

                lstm_input = tf.concat((encode_4_reshape_for_lstm, self.action_inputs), axis=2)


                print("encode 4 shape: {}".format(encode_4.shape))

                print("LSTM input shape: {}".format(lstm_input.shape))

                lstm_cell = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.LSTMCell(512),
                    tf.nn.rnn_cell.LSTMCell(256),
                ])

                batch_size = frame_input_shape[0]
                self.lstm_state_in = lstm_cell.zero_state(batch_size, dtype=tf.float32)

                lstm_output, self.lstm_state_out = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                                     inputs=lstm_input,
                                                                     sequence_length=self.sequence_lengths,
                                                                     initial_state=self.lstm_state_in)

                # lstm_output = tf.Print(lstm_output,[lstm_output[0,2,4], tf.shape(lstm_output)], "lstm_output: ")
                # reshape lstm output from (batch_size, max_seq_length, ...) to (batch_size * max_seq_length, ...)
                lstm_output_for_dense = tf.reshape(lstm_output,
                                                   shape=[frame_input_shape[0] * frame_input_shape[1], lstm_output.shape[-1]])
                # lstm_output_for_dense = tf.Print(lstm_output_for_dense,[lstm_output_for_dense,tf.shape(lstm_output_for_dense)], "lstm_output reshaped for dense: ")
                reshaped_again = tf.reshape(lstm_output_for_dense,
                                            shape=[frame_input_shape[0], frame_input_shape[1], lstm_output_for_dense.shape[-1]],
                                            name='reshape_again')
                # lstm_output_for_dense = tf.Print(lstm_output_for_dense, [reshaped_again[0,2,4], tf.shape(reshaped_again)], "lstm output put back")

                dense1 = tf.layers.dense(inputs=lstm_output_for_dense,
                                         units=128,
                                         activation=tf.nn.relu,
                                         kernel_initializer=variance_scaling)

                dense2 = tf.layers.dense(inputs=dense1,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=variance_scaling)
                # dense2 = tf.Print(dense2,[dense2,tf.shape(dense2)], "dense 2 output: ")

                # reshape dense output back to (batch_size, max_seq_length, ...)
                self.output = tf.reshape(dense2, shape=[frame_input_shape[0], frame_input_shape[1], dense2.shape[-1]])

                # self.output = tf.Print(self.output, [self.output, tf.shape(self.output)], "dense 2 output reshaped: ")
                # self.output = tf.Print(self.output, [self.computed_lengths, tf.shape(self.computed_lengths), self.sequence_lengths, tf.shape(self.sequence_lengths)], 'computed and given sequence lengths')

                with tf.name_scope('mse_loss'):
                    # Compute Squared Error for each frame
                    frame_squared_errors = tf.square(self.output - self.loss_targets)
                    mask = tf.sign(tf.reduce_max(tf.abs(self.loss_targets), 2))
                    masked_frame_mse_errors = frame_squared_errors * mask

                    # Average Over actual Sequence Lengths
                    with tf.control_dependencies(
                            [tf.assert_equal(tf.cast(tf.reduce_sum(mask, 1), dtype=tf.int32), self.sequence_lengths),
                             tf.assert_equal(self.computed_lengths, self.sequence_lengths),
                             tf.assert_equal(self.computed_lengths, length(self.loss_targets))
                             ]):
                        mse_over_sequences = tf.reduce_sum(masked_frame_mse_errors, 1) / tf.cast(self.sequence_lengths,
                                                                                                 dtype=tf.float32)

                    mse_over_batch = tf.reduce_mean(mse_over_sequences)
                    self.mse_loss = mse_over_batch
                    tf.summary.scalar('mse_loss', self.mse_loss)

            rnn_ops_scope = 'ANTICIPATOR_OPS'
            with tf.variable_scope(rnn_ops_scope):

                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
                self.local_step = tf.Variable(0, name='local_step', trainable=False)
                self.train_op = self.optimizer.minimize(self.mse_loss, global_step=self.local_step)
                # self.check_op = tf.add_check_numerics_ops()
                # print("\n\nCollection: {}".format(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vae_scope)))
                self.tf_summaries_merged = tf.summary.merge_all(scope=rnn_scope)

                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=rnn_scope)
                var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=rnn_ops_scope)

                self.saver = tf.train.Saver(var_list=var_list,
                                            max_to_keep=5,
                                            keep_checkpoint_every_n_hours=1)

            self.init = tf.variables_initializer(var_list=var_list, name='anticipator_initializer')

        if restore_from_dir:
            self._restore_model(restore_from_dir)
        else:
            print("\nrunning Anticipator local init\n")
            self.sess.run(self.init)

        self.writer.add_graph(self.graph)

    def reset_state(self):
        self.saved_state = None

    def predict_on_frame_batch(self, frames, actions):
        assert frames.shape[1:] == (64, 64, 3)
        assert actions.shape[1] == self.action_dim
        assert frames.shape[0] == actions.shape[0]

        feed_dict = {self.frame_inputs: np.reshape(frames, newshape=[frames.shape[0], 1, *frames.shape[1:]]),
                     self.action_inputs: np.reshape(actions, newshape=[actions.shape[0], 1, *actions.shape[1:]]),
                     self.sequence_lengths: np.asarray([1]*frames.shape[0])}

        if self.saved_state:
            feed_dict[self.lstm_state_in] = self.saved_state

        predictions, self.saved_state = self.sess.run([self.output, self.lstm_state_out], feed_dict=feed_dict)
        return np.squeeze(predictions)
