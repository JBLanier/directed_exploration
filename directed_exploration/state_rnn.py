import numpy as np
import tensorflow as tf
from directed_exploration.model import Model
from directed_exploration.utils.data_util import convertToOneHot
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


def dynamic_lstm(input_sequence_batch, retain_state_mask_sequence_batch, initial_states_batch,
                 scope, num_hidden, init_scale=1.0):

    # todo(JB) may not need to split this. Scan() may unpack a whole tensor, and may not need it to be a list
    # xs = [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=xs)]
    # ms = [v for v in tf.split(axis=1, num_or_size_splits=nsteps, value=ms)]

    input_sequence_batch = tf.transpose(input_sequence_batch, [1, 0, 2], name='trasnpose_xs')
    retain_state_mask_sequence_batch = tf.expand_dims(
        tf.transpose(retain_state_mask_sequence_batch, [1, 0], name='transpose_ms'),
        axis=2
    )

    nbatch, nin = [v.value for v in input_sequence_batch[0].get_shape()]
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, num_hidden * 4], initializer=ortho_init(init_scale))
        wh = tf.get_variable("wh", [num_hidden, num_hidden * 4], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [num_hidden * 4], initializer=tf.constant_initializer(0.0))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=initial_states_batch)

    def _dynamic_lstm_step(state_accumulator, inputs_elem):
        c, h = state_accumulator
        batch_inputs, batch_mask = inputs_elem
        c = c * (1 - batch_mask)
        h = h * (1 - batch_mask)
        z = tf.matmul(batch_inputs, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f * c + i * u
        h = o * tf.tanh(c)
        return c, h

    states = tf.scan(fn=_dynamic_lstm_step, elems=(input_sequence_batch, retain_state_mask_sequence_batch), initializer=(c, h))
    sequence_batches_out = states[1]
    state_batch_out = tf.concat(axis=1, values=(states[0][-1], states[1][-1]))
    return sequence_batches_out, state_batch_out


def lstm(xs, ms, s, scope, nh, nsteps, init_scale=1.0):

    xs = [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=xs)]
    ms = [v for v in tf.split(axis=1, num_or_size_splits=nsteps, value=ms)]

    nbatch, nin = [v.value for v in xs[0].get_shape()]
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh * 4], initializer=ortho_init(init_scale))
        wh = tf.get_variable("wh", [nh, nh * 4], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh * 4], initializer=tf.constant_initializer(0.0))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):

        c = c * (1 - m)
        h = h * (1 - m)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f * c + i * u
        h = o * tf.tanh(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s

def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]


def seq_to_batch(h, flat = False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert(len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])


def mask_non_zero_counts(mask):
    with tf.name_scope('mask_non_zero_count'):
        used = tf.sign(tf.abs(mask))
        length = tf.reduce_sum(used, 1)
        return length


def RNN_forward(sequence_inputs, batch_size, state_reset_before_prediction_mask, lstm_size, states_in, latent_dim, variable_scope, reuse=False):
    with tf.variable_scope(variable_scope, reuse=reuse):
        variance_scaling = tf.contrib.layers.variance_scaling_initializer()
        xavier = tf.contrib.layers.xavier_initializer()

        lstm_output, states_out = dynamic_lstm(input_sequence_batch=sequence_inputs,
                                               retain_state_mask_sequence_batch=state_reset_before_prediction_mask,
                                               initial_states_batch=states_in,
                                               num_hidden=lstm_size,
                                               scope='lstm1')

        lstm_output_for_dense = seq_to_batch(lstm_output)

        dense1 = tf.layers.dense(inputs=lstm_output_for_dense,
                                 units=256,
                                 activation=tf.nn.relu,
                                 kernel_initializer=variance_scaling)

        dense2 = tf.layers.dense(inputs=dense1,
                                 units=128,
                                 activation=tf.nn.relu,
                                 kernel_initializer=variance_scaling)

        dense3 = tf.layers.dense(inputs=dense2,
                                 units=latent_dim,
                                 activation=None,
                                 kernel_initializer=xavier)

        runtime_sequence_length = tf.shape(sequence_inputs)[1]

        out = tf.reshape(dense3, shape=[batch_size, runtime_sequence_length, latent_dim])

        return out, states_out


class StateRNN(Model):
    def __init__(self, latent_dim=4, action_dim=5, working_dir=None, sess=None, graph=None, summary_writer=None):
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


                self.sequence_inputs = tf.placeholder(tf.float32, shape=[None, None, self.latent_dim + self.action_dim],
                                                      name='z_and_action_inputs')
                self.sequence_targets = tf.placeholder(tf.float32, shape=[None, None, self.latent_dim],
                                                       name='z_targets')

                input_shape = tf.shape(self.sequence_inputs)
                runtime_batch_size = input_shape[0]
                runtime_sequence_length = input_shape[1]

                # mask (done at time t-1)
                self.state_reset_before_prediction_mask = tf.placeholder(tf.float32, [None, None])

                lstm_size = 256

                zero_states = tf.zeros(shape=[runtime_batch_size, lstm_size * 2], dtype=tf.float32)
                self.states_in = tf.placeholder_with_default(zero_states, shape=[None, lstm_size * 2])

                # reshape dense output back to (batch_size, seq_length, ...)
                rnn_forward_scope = 'rnn_forward'

                self.output, self.states_out = RNN_forward(
                        sequence_inputs=self.sequence_inputs,
                        batch_size=runtime_batch_size,
                        state_reset_before_prediction_mask=self.state_reset_before_prediction_mask,
                        lstm_size=lstm_size,
                        states_in=self.states_in,
                        latent_dim=self.latent_dim,
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

                    frame_squared_errors = tf.square(self.output - self.sequence_targets)
                    frame_mean_squared_errors = tf.reduce_mean(frame_squared_errors, axis=2)
                    masked_frame_mean_squared_erros = frame_mean_squared_errors * valid_example_mask
                    mse_over_sequences = tf.reduce_sum(masked_frame_mean_squared_erros, 1) / valid_example_counts
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

    def train_on_batch(self, input_code_sequence_batch, target_code_sequence_batch, states_mask_sequence_batch,
                       input_action_sequence_batch, states_batch=None):

        assert np.array_equal(input_code_sequence_batch.shape[:-1], target_code_sequence_batch.shape[:-1])
        assert np.array_equal(input_code_sequence_batch.shape[:-1], states_mask_sequence_batch[:, :-1].shape)
        assert np.array_equal(input_code_sequence_batch.shape[:-1], input_action_sequence_batch.shape[:-1])

        input_sequence_batch = np.concatenate((input_code_sequence_batch, input_action_sequence_batch), axis=2)

        feed_dict = {
            self.sequence_inputs: input_sequence_batch,
            self.sequence_targets: target_code_sequence_batch,
            self.state_reset_before_prediction_mask: states_mask_sequence_batch[:, :-1],
            self.state_reset_between_input_and_target_mask:  states_mask_sequence_batch[:, 1:]
        }

        if states_batch is not None:
            assert np.array_equal(input_code_sequence_batch.shape[0], states_batch.shape[0])
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

    def predict_on_frames(self, z_codes, actions, states_mask, states_in=None):

        actions = np.asarray(actions)
        states_mask = np.asarray(states_mask)

        assert z_codes.shape[0] == actions.shape[0]
        assert z_codes.shape[0] == states_mask.shape[0]
        assert z_codes.shape[1] == self.latent_dim

        batch_size = z_codes.shape[0]

        actions = convertToOneHot(actions, num_classes=self.action_dim)
        actions = np.reshape(actions, newshape=(batch_size, self.action_dim))

        states_mask = np.reshape(states_mask, newshape=(batch_size, 1))

        sequence_inputs = np.expand_dims(np.concatenate((z_codes, actions), axis=1), 1)

        feed_dict = {self.sequence_inputs: sequence_inputs,
                     self.state_reset_before_prediction_mask: states_mask
                     }

        if states_in is not None:
            feed_dict[self.states_in] = states_in

        predictions, states_out = self.sess.run([self.output, self.states_out], feed_dict=feed_dict)

        return predictions[:, 0, :], states_out

    def predict_on_frames_retain_state(self, z_codes, actions, states_mask):

        predictions, states_out = self.predict_on_frames(z_codes, actions, states_mask, self.saved_state)
        self.saved_state = states_out
        return predictions

    # def predict_on_sequences_retain_state(self, z_sequences, action_sequences, sequence_lengths):
    #     assert z_sequences.shape[2] == self.latent_dim
    #     assert z_sequences.shape[0:2] == action_sequences.shape[0:2]
    #
    #     # todo: this concatenation may be on wrong axis
    #     raise NotImplementedError
    #     feed_dict = {self.sequence_inputs: np.concatenate((z_sequences, action_sequences), axis=1),
    #                  self.sequence_lengths: np.asarray([sequence_lengths])}
    #
    #     if self.saved_state:
    #         feed_dict[self.lstm_state_in] = self.saved_state
    #
    #     predictions, self.saved_state = self.sess.run([self.output, self.lstm_state_out], feed_dict=feed_dict)
    #
    #     return predictions

    # def predict_on_sequences(self, z_sequences, action_sequences, sequence_lengths):
    #     assert z_sequences.shape[2] == self.latent_dim
    #     assert z_sequences.shape[0:2] == action_sequences.shape[0:2]
    #     assert action_sequences.shape[2] == self.action_dim
    #
    #     feed_dict = {self.sequence_inputs: np.concatenate((z_sequences, action_sequences), axis=2),
    #                  self.sequence_lengths: sequence_lengths}
    #
    #     predictions = np.squeeze(self.sess.run([self.output], feed_dict=feed_dict))
    #
    #     return predictions
