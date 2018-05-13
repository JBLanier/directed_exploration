
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
        date_identifier = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if restore_from_dir:
            self.save_metagraph = False
            self.identifier = restore_from_dir.split('/')[-1]
            self.save_file_path = restore_from_dir
        else:
            self.save_metagraph = True
            self.identifier = '{}_{}dim_{}'.format(self.save_prefix, self.latent_dim, date_identifier)
            self.save_file_path = './' + self.identifier

        self.tensorboard_path = './tensorboard'

        self._build_model(restore_from_dir)

        self.writer = tf.summary.FileWriter("{}/{}".format(self.tensorboard_path, self.save_file_path[2:]), self.graph)

        self.saved_state = None

    def _restore_model(self, from_dir):
        print("\n\nRestoring State RNN Model from {}".format(from_dir))
        with self.graph.as_default():
            # self.saver = tf.train.import_meta_graph(os.path.join(self.save_file_path, self.save_prefix + '.meta'))
            self.saver.restore(self.sess, tf.train.latest_checkpoint(from_dir))

    def _build_model(self, restore_from_dir=None):

        with self.graph.as_default():
            rnn_scope = 'State_RNN'
            with tf.variable_scope(rnn_scope):
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

                lstm_cell = tf.nn.rnn_cell.LSTMCell(256)

                batch_size = tf.shape(self.sequence_inputs)[0]
                self.lstm_state_in = lstm_cell.zero_state(batch_size, dtype=tf.float32)

                lstm_output, self.lstm_state_out = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                                 inputs=self.sequence_inputs,
                                                                 sequence_length=self.sequence_lengths,
                                                                 initial_state=self.lstm_state_in)

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
                    with tf.control_dependencies([tf.assert_equal(tf.cast(tf.reduce_sum(mask, 1), dtype=tf.int32), self.sequence_lengths),
                                                  tf.assert_equal(self.computed_lengths, self.sequence_lengths),
                                                  tf.assert_equal(self.computed_lengths, length(self.sequence_targets))
                                                  ]):

                        mse_over_sequences = tf.reduce_sum(masked_frame_mse_errors, 1) / tf.cast(self.sequence_lengths, dtype=tf.float32)

                    mse_over_batch = tf.reduce_mean(mse_over_sequences)
                    self.mse_loss = mse_over_batch
                    tf.summary.scalar('mse_loss', self.mse_loss)

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = self.optimizer.minimize(self.mse_loss, global_step=self.global_step)
            # self.check_op = tf.add_check_numerics_ops()
            # print("\n\nCollection: {}".format(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vae_scope)))
            self.tf_summaries_merged = tf.summary.merge_all()

            save_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=rnn_scope)
            save_var_list.append(self.global_step)

            self.saver = tf.train.Saver(var_list=save_var_list,
                                        max_to_keep=5,
                                        keep_checkpoint_every_n_hours=1)

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

            local_step = 1
            while True:

                try:
                    batch_inputs, batch_targets, batch_lengths = sess.run(iter)
                except tf.errors.OutOfRangeError:
                    print("Input_fn ended at step {}".format(local_step))
                    break

                # Train
                feed_dict = {
                    self.sequence_inputs: batch_inputs,
                    self.sequence_targets: batch_targets,
                    self.sequence_lengths: batch_lengths
                             }

                _, loss, summaries, global_step = sess.run([self.train_op,
                                                            self.mse_loss,
                                                            self.tf_summaries_merged,
                                                            self.global_step],
                                                           feed_dict=feed_dict)

                # for target in np.squeeze(targets)[0]:
                #     cv2.imshow("target",np.squeeze(vae.decode_frames(np.expand_dims(target,0)))[:,:,::-1])
                #     cv2.waitKey(300)
                # self.writer.add_summary(summaries, step)

                if local_step % 20 == 0 or local_step == 1:
                    print('Step %i, Loss: %f' % (global_step, loss))

                if local_step % 1000 == 0:
                    self.save_model()

                if steps and local_step >= steps:
                    print("Completed {} steps".format(steps))
                    break

                local_step += 1

    def reset_state(self):
        self.saved_state = None

    def predict_on_frames(self, z_codes, actions):
        assert z_codes.shape[1] == self.latent_dim
        assert actions.shape[1] == self.action_dim

        feed_dict = {self.sequence_inputs: np.expand_dims(np.concatenate((z_codes, actions), axis=1), 0),
                     self.sequence_lengths: np.asarray([1])}

        if self.saved_state:
            feed_dict[self.lstm_state_in] = self.saved_state

        predictions, self.saved_state = self.sess.run([self.output, self.lstm_state_out], feed_dict=feed_dict)

        return predictions

    def save_model(self):

        if not os.path.exists(self.save_file_path):
            os.makedirs(self.save_file_path, exist_ok=True)

        save_path = self.saver.save(sess=self.sess,
                                    save_path=os.path.join(self.save_file_path, self.save_prefix),
                                    write_meta_graph=self.save_metagraph,
                                    global_step=self.global_step)

        self.save_metagraph = False

        print("RNN Model saved in path: {}".format(save_path))

    def __del__(self):
        if self.writer:
            self.writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()


