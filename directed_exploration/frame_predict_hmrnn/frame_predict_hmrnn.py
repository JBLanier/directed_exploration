import numpy as np
import tensorflow as tf
from directed_exploration.model import Model
from directed_exploration.utils.data_util import convertToOneHot
from directed_exploration.frame_predict_hmrnn.dynamic_hmlstm import dynamic_hmlstm
import logging

logger = logging.getLogger(__name__)


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init

def mask_non_zero_counts(mask):
    with tf.name_scope('mask_non_zero_count'):
        used = tf.sign(tf.abs(mask))
        length = tf.reduce_sum(used, 1)
        return length


def RNN_forward(frame_inputs, action_inputs, batch_size, state_reset_before_prediction_mask, lstm_size, states_in, variable_scope, reuse=False):
    with tf.variable_scope(variable_scope, reuse=reuse):
        variance_scaling = tf.contrib.layers.variance_scaling_initializer()
        xavier = tf.contrib.layers.xavier_initializer()

        runtime_sequence_length = tf.shape(frame_inputs)[1]

        frame_inputs_as_batch = tf.reshape(frame_inputs, shape=[batch_size * runtime_sequence_length, *frame_inputs.shape[2:]])

        print("inputs shape {}".format(frame_inputs_as_batch.shape))

        compress = tf.layers.Conv2D(filters=32, kernel_size=4, strides=2,
                                    padding='valid', activation=tf.nn.relu,
                                    kernel_initializer=variance_scaling,
                                    name='encode_1')(frame_inputs_as_batch)

        print("compress 1 shape {}".format(compress.shape))

        compress = tf.layers.Conv2D(filters=64, kernel_size=4, strides=2,
                                    padding='valid', activation=tf.nn.relu,
                                    kernel_initializer=variance_scaling,
                                    name='encode_2')(compress)

        print("compress 2 shape {}".format(compress.shape))

        compress = tf.layers.Conv2D(filters=128, kernel_size=4, strides=2,
                                    padding='valid', activation=tf.nn.relu,
                                    kernel_initializer=variance_scaling,
                                    name='encode_3')(compress)

        print("compress 3 shape {}".format(compress.shape))

        compress = tf.layers.Conv2D(filters=256, kernel_size=4, strides=2,
                                    padding='valid', activation=tf.nn.relu,
                                    kernel_initializer=variance_scaling,
                                    name='encode_4')(compress)
        print("compress 4 shape {}".format(compress.shape))
        compress_before_dense_shape = compress.shape

        compress = tf.layers.flatten(compress)

        print("compress flatten shape {}".format(compress.shape))


        compress = tf.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer=variance_scaling)(compress)


        print("compress dense 1 shape {}".format(compress.shape))


        compress_as_seq = tf.reshape(compress, shape=[batch_size, runtime_sequence_length, compress.shape[1]])

        lstm_inputs = tf.concat(values=(compress_as_seq, action_inputs), axis=2)

        lstm_output, states_out = dynamic_lstm(input_sequence_batch=lstm_inputs,
                                               retain_state_mask_sequence_batch=state_reset_before_prediction_mask,
                                               initial_states_batch=states_in,
                                               num_hidden=lstm_size,
                                               scope='lstm1')

        lstm_output_as_batch = tf.reshape(lstm_output, shape=[batch_size*runtime_sequence_length, lstm_size])

        decompress = tf.layers.Dense(units=2304, activation=tf.nn.relu, kernel_initializer=variance_scaling)(lstm_output_as_batch)

        # decompress = tf.reshape(tensor=decompress, shape=[-1, 1, 1, decompress.shape[1]])
        decompress = tf.reshape(tensor=decompress, shape=[-1, *compress_before_dense_shape[1:]])

        print("decompress dense 1 shape {}".format(decompress.shape))


        decompress = tf.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2,
                                             padding='valid', activation=tf.nn.relu,
                                             kernel_initializer=variance_scaling,
                                             name='decode_1')(decompress)

        print("decompress 1 shape {}".format(decompress.shape))


        decompress = tf.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2,
                                             padding='valid', activation=tf.nn.relu,
                                             kernel_initializer=variance_scaling,
                                             name='decode_2')(decompress)

        print("decompress 2 shape {}".format(decompress.shape))


        decompress = tf.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2,
                                             padding='valid', activation=tf.nn.relu,
                                             kernel_initializer=variance_scaling,
                                             name='decode_3')(decompress)

        print("decompress 3 shape {}".format(decompress.shape))


        decompress = tf.layers.Conv2DTranspose(filters=3, kernel_size=2, strides=2,
                                             padding='valid', activation=tf.nn.sigmoid,
                                             kernel_initializer=xavier, bias_initializer=xavier,
                                             name='decode_4')(decompress)

        print("decompress 4 shape {}".format(decompress.shape))


        out = tf.reshape(decompress, shape=tf.shape(frame_inputs), name='out_reshape')

        return out, states_out


class FramePredictHMRNN(Model):
    def __init__(self, observation_space, action_dim, working_dir=None, sess=None, graph=None, summary_writer=None):
        logger.info("Frame_Predict_RNN obs space {} action dim {}".format(observation_space, action_dim))

        self.observation_space = observation_space
        self.action_dim = action_dim
        self.saved_state = None

        save_prefix = 'frame_predict_rnn_obs_{}_act_{}'.format(self.observation_space, self.action_dim)

        super().__init__(save_prefix, working_dir, sess, graph, summary_writer=summary_writer)

    def _build_model(self, restore_from_dir=None):

        with self.graph.as_default():
            rnn_scope = 'FRAME_PREDICT_RNN_MODEL'
            with tf.variable_scope(rnn_scope):

                self.sequence_frame_inputs = tf.placeholder(tf.float32, shape=[None, None, *self.observation_space.shape],
                                                      name='frame_inputs')

                self.sequence_action_inputs = tf.placeholder(tf.float32, shape=[None, None, self.action_dim],
                                                    name='action_inputs')

                self.sequence_frame_targets = tf.placeholder(tf.float32, shape=[None, None, *self.observation_space.shape],
                                                       name='frame_targets')

                frame_input_shape = tf.shape(self.sequence_frame_inputs)
                runtime_batch_size = frame_input_shape[0]
                runtime_sequence_length = frame_input_shape[1]

                # mask (done at time t-1)
                self.state_reset_before_prediction_mask = tf.placeholder(tf.float32, [None, None])

                lstm_size = 256

                zero_states = tf.zeros(shape=[runtime_batch_size, lstm_size * 2], dtype=tf.float32)
                self.states_in = tf.placeholder_with_default(zero_states, shape=[None, lstm_size * 2])

                # reshape dense output back to (batch_size, seq_length, ...)
                rnn_forward_scope = 'rnn_forward'

                self.output, self.states_out = RNN_forward(
                        frame_inputs=self.sequence_frame_inputs,
                        action_inputs=self.sequence_action_inputs,
                        batch_size=runtime_batch_size,
                        state_reset_before_prediction_mask=self.state_reset_before_prediction_mask,
                        lstm_size=lstm_size,
                        states_in=self.states_in,
                        variable_scope=rnn_forward_scope,
                        reuse=False)

                # with tf.control_dependencies([tf.assert_equal(self.output,self.output2), tf.assert_equal(self.states_out, self.states_out2)]):
                #     self.output = tf.Print(self.output, [self.output])
                #     self.states_out = tf.Print(self.states_out, [self.states_out])

                with tf.name_scope('mse_loss'):
                    # Compute Squared Error for each frame

                    # mask (done at time t)
                    self.state_reset_between_input_and_target_mask = tf.placeholder(tf.float32,
                                                                                    [None, None])

                    valid_example_mask = self.state_reset_between_input_and_target_mask
                    valid_example_counts = mask_non_zero_counts(valid_example_mask)

                    frame_squared_errors = tf.square(self.output - self.sequence_frame_targets)
                    frame_squared_error = tf.reduce_sum(frame_squared_errors, axis=(2, 3, 4))
                    self.masked_frame_mean_squared_errors = frame_squared_error * valid_example_mask
                    mse_over_sequences = tf.reduce_sum(self.masked_frame_mean_squared_errors, 1) / valid_example_counts
                    mse_over_batch = tf.reduce_mean(mse_over_sequences)
                    self.mse_loss = mse_over_batch
                    tf.summary.scalar('mse_loss', self.mse_loss)

            rnn_ops_scope = 'FRAME_PREDICT_RNN_OPS'
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

            self.init = tf.variables_initializer(var_list=var_list, name='frame_predict_rnn_initializer')

            self.tvars = tf.trainable_variables()

        if restore_from_dir:
            self._restore_model(restore_from_dir)
        else:
            logger.debug("Running Frame Predict RNN local init\n")
            self.sess.run(self.init)

        self.writer.add_graph(self.graph)

    def return_all_variables_with_values_in_dict(self):
        tvars_vals = self.sess.run(self.tvars)

        dict = {}

        for var, val in zip(self.tvars, tvars_vals):
            dict[var.name] = val

        return dict

    def train_on_batch(self, input_frame_sequence_batch, target_frame_sequence_batch, states_mask_sequence_batch,
                       input_action_sequence_batch, states_batch=None):

        assert np.array_equal(input_frame_sequence_batch.shape[:-3], target_frame_sequence_batch.shape[:-3])
        assert np.array_equal(input_frame_sequence_batch.shape[:-3], states_mask_sequence_batch[:, :-1].shape)
        assert np.array_equal(input_frame_sequence_batch.shape[:-3], input_action_sequence_batch.shape[:-1])

        feed_dict = {
            self.sequence_frame_inputs: input_frame_sequence_batch,
            self.sequence_action_inputs: input_action_sequence_batch,
            self.sequence_frame_targets: target_frame_sequence_batch,
            self.state_reset_before_prediction_mask: states_mask_sequence_batch[:, :-1],
            self.state_reset_between_input_and_target_mask:  states_mask_sequence_batch[:, 1:]
        }

        if states_batch is not None:
            assert np.array_equal(input_frame_sequence_batch.shape[0], states_batch.shape[0])
            feed_dict[self.states_in] = states_batch

        _, loss, states_out, step, summaries = self.sess.run([self.train_op,
                                                              self.mse_loss,
                                                              self.states_out,
                                                              self.local_step,
                                                              self.tf_summaries_merged],
                                                             feed_dict=feed_dict)

        self.writer.add_summary(summaries, step)

        return loss, states_out, step

    # def train_on_input_fn(self, input_fn, steps=None):
    #
    #     sess = self.sess
    #
    #     # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #     # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    #
    #     with self.graph.as_default():
    #         # Run the initializer
    #
    #         with tf.name_scope("input_fn"):
    #             iter = input_fn()
    #
    #         local_step = 1
    #         while True:
    #
    #             try:
    #                 batch_inputs, batch_targets, batch_lengths = sess.run(iter)
    #             except tf.errors.OutOfRangeError:
    #                 logger.debug("Input_fn ended at step {}".format(local_step))
    #                 break
    #
    #             # Train
    #             feed_dict = {
    #                 self.sequence_inputs: batch_inputs,
    #                 self.sequence_targets: batch_targets,
    #                 self.sequence_lengths: batch_lengths
    #             }
    #
    #             _, loss, summaries, step = sess.run([self.train_op,
    #                                                  self.mse_loss,
    #                                                  self.tf_summaries_merged,
    #                                                  self.local_step],
    #                                                 feed_dict=feed_dict)
    #
    #             # for target in np.squeeze(targets)[0]:
    #             #     cv2.imshow("target",np.squeeze(vae.decode_frames(np.expand_dims(target,0)))[:,:,::-1])
    #             #     cv2.waitKey(300)
    #             self.writer.add_summary(summaries, step)
    #
    #             if local_step % 20 == 0 or local_step == 1:
    #                 logger.debug('Step %i, Loss: %f' % (step, loss))
    #
    #             if local_step % 1000 == 0:
    #                 self.save_model()
    #
    #             if steps and local_step >= steps:
    #                 logger.debug("Completed {} steps".format(steps))
    #                 break
    #
    #             local_step += 1
    #
    # def train_on_iterator(self, iterator, iterator_sess=None, steps=None, save_every_n_steps=None):
    #
    #     if not iterator_sess:
    #         iterator_sess = self.sess
    #
    #     local_step = 1
    #     while True:
    #
    #         try:
    #             batch_inputs, batch_targets, batch_lengths = iterator_sess.run(iterator)
    #         except tf.errors.OutOfRangeError:
    #             logger.debug("Input_fn ended at step {}".format(local_step))
    #             break
    #
    #         # Train
    #         feed_dict = {
    #             self.sequence_inputs: batch_inputs,
    #             self.sequence_targets: batch_targets,
    #             self.sequence_lengths: batch_lengths
    #         }
    #
    #         # print("sequence_inputs: \n{}".format(batch_inputs))
    #         # print("above sequence lengths {}".format(batch_lengths))
    #         # if 1 in batch_lengths or 2 in batch_lengths or 3 in batch_lengths:
    #         #     assert False
    #
    #         _, loss, summaries, step = self.sess.run([self.train_op,
    #                                                   self.mse_loss,
    #                                                   self.tf_summaries_merged,
    #                                                   self.local_step],
    #                                                  feed_dict=feed_dict)
    #
    #         # for target in np.squeeze(targets)[0]:
    #         #     cv2.imshow("target",np.squeeze(vae.decode_frames(np.expand_dims(target,0)))[:,:,::-1])
    #         #     cv2.waitKey(300)
    #         self.writer.add_summary(summaries, step)
    #
    #         if local_step % 20 == 0 or local_step == 1:
    #             logger.debug('State RNN Step %i, Loss: %f' % (step, loss))
    #
    #         if save_every_n_steps and step % save_every_n_steps == 0:
    #             self.save_model()
    #
    #         if steps and local_step >= steps:
    #             logger.debug("Completed {} steps".format(steps))
    #             break
    #
    #         local_step += 1

    def reset_saved_state(self):
        self.saved_state = None

    # def selectively_reset_saved_states(self, dones):
    #     mask = 1 - (np.reshape(dones, newshape=(len(dones), 1)))
    #     for cell in self.saved_state:
    #         np.multiply(cell.c, mask, out=cell.c)
    #         np.multiply(cell.h, mask, out=cell.h)

    def predict_on_frame_batch_with_loss(self, frames, actions, states_mask, states_in=None, target_predictions=None, valid_prediction_mask=None):

        actions = np.asarray(actions)
        states_mask = np.asarray(states_mask)

        assert frames.shape[0] == actions.shape[0]
        assert frames.shape[0] == states_mask.shape[0]
        assert frames.shape[1:] == self.observation_space.shape

        batch_size = frames.shape[0]

        actions = convertToOneHot(actions, num_classes=self.action_dim)
        actions = np.reshape(actions, newshape=(batch_size, self.action_dim))

        states_mask = np.reshape(states_mask, newshape=(batch_size, 1))

        feed_dict = {self.sequence_frame_inputs: np.expand_dims(frames, axis=1),
                     self.sequence_action_inputs: np.expand_dims(actions, axis=1),
                     self.state_reset_before_prediction_mask: states_mask
                     }

        if states_in is not None:
            feed_dict[self.states_in] = states_in

        if target_predictions is not None and valid_prediction_mask is not None:
            feed_dict[self.sequence_frame_targets] = np.expand_dims(target_predictions, axis=1)
            feed_dict[self.state_reset_between_input_and_target_mask] = np.expand_dims(valid_prediction_mask, axis=1)

            predictions, states_out, losses = self.sess.run([self.output, self.states_out, self.masked_frame_mean_squared_errors], feed_dict=feed_dict)
            return predictions[:, 0, ...], states_out, losses[:, 0]

        else:
            predictions, states_out = self.sess.run([self.output, self.states_out], feed_dict=feed_dict)
            return predictions[:, 0, ...], states_out, None

    def predict_on_frame_batch(self, frames, actions, states_mask, states_in=None):
        return self.predict_on_frame_batch_with_loss( frames, actions, states_mask, states_in)[:2]

    def predict_on_frames_retain_state(self, frames, actions, states_mask):

        predictions, states_out = self.predict_on_frame_batch(frames, actions, states_mask, self.saved_state)
        self.saved_state = states_out
        return predictions

