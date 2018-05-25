
import numpy as np
import tensorflow as tf
from directed_exploration.model import Model
import logging

logger = logging.getLogger(__name__)


class StateRNN(Model):

    def __init__(self, latent_dim=4, action_dim=2, working_dir=None, sess=None, graph=None, summary_writer=None):
        logger.info("RNN latent dim {} action dim {}".format(latent_dim, action_dim))

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.saved_state = None

        save_prefix = 'state_rnn_{}dim'.format(self.latent_dim)

        super().__init__(save_prefix, working_dir, sess, graph, summary_writer=summary_writer)

    def _build_model(self, restore_from_dir=None):

        with self.graph.as_default():
            rnn_scope = 'STATE_RNN_MODEL'
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

                lstm_cell =tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.LSTMCell(512),
                    tf.nn.rnn_cell.LSTMCell(512),
                ])

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
                                        units=512,
                                        activation=tf.nn.relu,
                                        kernel_initializer=variance_scaling)

                dense2 = tf.layers.dense(inputs=dense1,
                                         units=512,
                                         activation=tf.nn.relu,
                                         kernel_initializer=variance_scaling)

                dense3 = tf.layers.dense(inputs=dense2,
                                         units=256,
                                         activation=tf.nn.relu,
                                         kernel_initializer=variance_scaling)

                dense4 = tf.layers.dense(inputs=dense3,
                                         units=self.latent_dim,
                                         activation=None,
                                         kernel_initializer=variance_scaling)
                # dense2 = tf.Print(dense2,[dense2,tf.shape(dense2)], "dense 2 output: ")

                # reshape dense output back to (batch_size, max_seq_length, ...)
                self.output = tf.reshape(dense4, shape=[-1, tf.shape(self.sequence_inputs)[1], dense4.shape[-1]])

                # self.output = tf.Print(self.output, [self.output, tf.shape(self.output)], "dense 2 output reshaped: ")
                # self.output = tf.Print(self.output, [self.computed_lengths, tf.shape(self.computed_lengths), self.sequence_lengths, tf.shape(self.sequence_lengths)], 'computed and given sequence lengths')

                with tf.name_scope('mse_loss'):
                    # Compute Squared Error for each frame
                    frame_squared_errors = tf.square(self.output - self.sequence_targets)
                    frame_mean_squared_errors = tf.reduce_mean(frame_squared_errors, axis=2)
                    mask = tf.sign(tf.reduce_max(tf.abs(self.sequence_targets), 2))
                    masked_frame_mse_errors = frame_mean_squared_errors * mask

                    # Average Over actual Sequence Lengths
                    with tf.control_dependencies([tf.assert_equal(tf.cast(tf.reduce_sum(mask, 1), dtype=tf.int32), self.sequence_lengths, name='mask_length_vs_self.sequence_lengths', summarize=999999),
                                                  tf.assert_equal(self.computed_lengths, self.sequence_lengths, name='computed_length_vs_self.sequence_lengths', summarize=999999),
                                                  tf.assert_equal(self.computed_lengths, length(self.sequence_targets), name='computed_length_vs_sequence_targets_length', summarize=999999)
                                                  ]):

                        mse_over_sequences = tf.reduce_sum(masked_frame_mse_errors, 1) / tf.cast(self.sequence_lengths, dtype=tf.float32)

                    mse_over_batch = tf.reduce_mean(mse_over_sequences)
                    self.mse_loss = mse_over_batch
                    tf.summary.scalar('mse_loss', self.mse_loss)

            rnn_ops_scope = 'STATE_RNN_OPS'
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

            self.init = tf.variables_initializer(var_list=var_list, name='state_rnn_initializer')

        if restore_from_dir:
            self._restore_model(restore_from_dir)
        else:
            logger.debug("Running State RNN local init\n")
            self.sess.run(self.init)

        self.writer.add_graph(self.graph)

    def train_on_input_fn(self, input_fn, steps=None):

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
                    logger.debug("Input_fn ended at step {}".format(local_step))
                    break

                # Train
                feed_dict = {
                    self.sequence_inputs: batch_inputs,
                    self.sequence_targets: batch_targets,
                    self.sequence_lengths: batch_lengths
                             }

                _, loss, summaries, step = sess.run([self.train_op,
                                                            self.mse_loss,
                                                            self.tf_summaries_merged,
                                                            self.local_step],
                                                           feed_dict=feed_dict)

                # for target in np.squeeze(targets)[0]:
                #     cv2.imshow("target",np.squeeze(vae.decode_frames(np.expand_dims(target,0)))[:,:,::-1])
                #     cv2.waitKey(300)
                self.writer.add_summary(summaries, step)

                if local_step % 20 == 0 or local_step == 1:
                    logger.debug('Step %i, Loss: %f' % (step, loss))

                if local_step % 1000 == 0:
                    self.save_model()

                if steps and local_step >= steps:
                    logger.debug("Completed {} steps".format(steps))
                    break

                local_step += 1

    def train_on_iterator(self, iterator, iterator_sess=None, steps=None, save_every_n_steps=None):

        if not iterator_sess:
            iterator_sess = self.sess

        local_step = 1
        while True:

            try:
                batch_inputs, batch_targets, batch_lengths, batch_frames =iterator_sess.run(iterator)
            except tf.errors.OutOfRangeError:
                logger.debug("Input_fn ended at step {}".format(local_step))
                break

            # Train
            feed_dict = {
                self.sequence_inputs: batch_inputs,
                self.sequence_targets: batch_targets,
                self.sequence_lengths: batch_lengths
            }

            _, loss, summaries, step = self.sess.run([self.train_op,
                                                    self.mse_loss,
                                                    self.tf_summaries_merged,
                                                    self.local_step],
                                                feed_dict=feed_dict)

            # for target in np.squeeze(targets)[0]:
            #     cv2.imshow("target",np.squeeze(vae.decode_frames(np.expand_dims(target,0)))[:,:,::-1])
            #     cv2.waitKey(300)
            self.writer.add_summary(summaries, step)

            if local_step % 20 == 0 or local_step == 1:
                logger.debug('State RNN Step %i, Loss: %f' % (step, loss))

            if save_every_n_steps and local_step % save_every_n_steps == 0:
                self.save_model()

            if steps and local_step >= steps:
                logger.debug("Completed {} steps".format(steps))
                break

            local_step += 1

    def reset_state(self):
        self.saved_state = None

    def predict_on_frames_retain_state(self, z_codes, actions):
        assert z_codes.shape[1] == self.latent_dim
        assert actions.shape[1] == self.action_dim

        feed_dict = {self.sequence_inputs: np.expand_dims(np.concatenate((z_codes, actions), axis=1), 0),
                     self.sequence_lengths: np.asarray([1])}

        if self.saved_state:
            feed_dict[self.lstm_state_in] = self.saved_state

        predictions, self.saved_state = self.sess.run([self.output, self.lstm_state_out], feed_dict=feed_dict)

        return predictions

    def predict_on_sequences(self, z_sequences, action_sequences, sequence_lengths):
        assert z_sequences.shape[2] == self.latent_dim
        assert action_sequences.shape[2] == self.action_dim
        assert z_sequences.shape[0:2] == action_sequences.shape[0:2]

        feed_dict = {self.sequence_inputs: np.concatenate((z_sequences, action_sequences), axis=2),
                     self.sequence_lengths: sequence_lengths}

        predictions = np.squeeze(self.sess.run([self.output], feed_dict=feed_dict))

        return predictions

