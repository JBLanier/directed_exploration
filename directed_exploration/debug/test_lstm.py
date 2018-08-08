

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


def seq_to_batch(h, flat = False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert(len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])

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

        c = c * m
        h = h * m
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
    # c = tf.Print(c, data=(tf.shape(c), tf.shape(h)), message='shapes of c and h')

    def _dynamic_lstm_step(state_accumulator, inputs_elem):
        c, h = state_accumulator
        batch_inputs, batch_mask = inputs_elem
        # batch_mask = tf.Print(batch_mask, data=[batch_mask], message='batch mask')
        c = c * batch_mask
        h = h * batch_mask
        z = tf.matmul(batch_inputs, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f * c + i * u
        h = o * tf.tanh(c)
        # h = tf.Print(h, data=[c, h], message='c and h at end of a loop iteration')
        return c, h

    states = tf.scan(fn=_dynamic_lstm_step, elems=(input_sequence_batch, retain_state_mask_sequence_batch), initializer=(c, h), back_prop=True)
    sequence_batches_out = tf.transpose(states[1], [1,0,2])
    state_batch_out = tf.concat(axis=1, values=(states[0][-1], states[1][-1]))
    # sequence_batches_out = tf.Print(sequence_batches_out, data=[sequence_batches_out], message='\nsequence_batches_out:\n', summarize=100)

    return sequence_batches_out, state_batch_out


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
graph = tf.Graph()
sess = tf.Session(config=config, graph=graph)

with graph.as_default():
    variable_scope = 'lstm_test'

    sequence_inputs = tf.placeholder(tf.float32, shape=[None, None, 4],
                                          name='sequence_inputs')

    input_shape = tf.shape(sequence_inputs)
    runtime_batch_size = input_shape[0]
    runtime_sequence_length = input_shape[1]

    # mask (done at time t-1)
    state_reset_before_prediction_mask = tf.placeholder(tf.float32, [None, None])

    lstm_size = 3

    zero_states = tf.zeros(shape=[runtime_batch_size, lstm_size * 2], dtype=tf.float32)
    states_in = tf.placeholder_with_default(zero_states, shape=[None, lstm_size * 2])

    with tf.variable_scope(variable_scope, reuse=False):
        lstm_old, lstm_old_state = lstm(xs=sequence_inputs,
                        ms=state_reset_before_prediction_mask,
                        s=states_in,
                        scope='lstm',
                        nh=lstm_size,
                        nsteps=4)
    # lstm_old = tf.Print(lstm_old, [tf.shape(lstm_old)], 'lstm old shape')
    lstm_old = seq_to_batch(lstm_old)

    with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
        lstm_new, lstm_new_state = dynamic_lstm(input_sequence_batch=sequence_inputs,
                        retain_state_mask_sequence_batch=state_reset_before_prediction_mask,
                        initial_states_batch=states_in,
                        scope='lstm',
                        num_hidden=lstm_size,)
    lstm_new = tf.Print(lstm_new, [tf.shape(lstm_new)], 'lstm new shape')
    #
    lstm_new = tf.reshape(lstm_new, [-1, lstm_size])
    # lstm_new = seq_to_batch(lstm_new)


    init = tf.global_variables_initializer()

sess.run(init)

feed_dict = {sequence_inputs: np.ones((3, 4, 4)), state_reset_before_prediction_mask: np.ones((3, 4)), states_in: np.ones((3, lstm_size*2))*2}

lstm_old_out = np.squeeze(sess.run(lstm_old, feed_dict))
lstm_new_out =  np.squeeze(sess.run(lstm_new, feed_dict))

print("\nPredictions ----------\nOLD:\n{}\n\nNEW:\n{}\n\n".format(lstm_old_out, lstm_new_out))

print("\nStates ----------\nOLD:\n{}\n\nNEW:\n{}\n\n".format(np.squeeze(sess.run(lstm_old_state, feed_dict)), np.squeeze(sess.run(lstm_new_state, feed_dict))))


print(type(lstm_old_out[0]))
print(type(lstm_new_out[0]))


