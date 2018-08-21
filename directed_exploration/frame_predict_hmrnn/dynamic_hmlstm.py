from hmlstm.hmlstm_cell import HMLSTMCell, HMLSTMState
from hmlstm.multi_hmlstm_cell import MultiHMLSTMCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
import numpy as np


class HMLSTMNetwork(object):
    def __init__(self,
                 input_size=1,
                 output_size=1,
                 num_layers=3,
                 hidden_state_sizes=50,
                 encoding_size=50,
                 out_hidden_size=100,
                 embed_size=100,
                 task='regression'):
        """
        HMLSTMNetwork is a class representing hierarchical multiscale
        long short-term memory network.

        params:
        ---
        input_size: integer, the size of an input at one timestep
        output_size: integer, the size of an output at one timestep
        num_layers: integer, the number of layers in the hmlstm
        hidden_state_size: integer or list of integers. If it is an integer,
            it is the size of the hidden state for each layer of the hmlstm.
            If it is a list, it must have length equal to the number of layers,
            and each integer of the list is the size of the hidden state for
            the layer correspodning to its index.
        out_hidden_size: integer, the size of the two hidden layers in the
            output network.
        embed_size: integer, the size of the embedding in the output network.
        task: string, one of 'regression' and 'classification'.
        """

        self._out_hidden_size = out_hidden_size
        self._encoding_size = encoding_size
        self._embed_size = embed_size
        self._num_layers = num_layers
        self._input_size = input_size
        self._session = None
        self._graph = None
        self._task = task
        self._output_size = output_size

        if type(hidden_state_sizes) is list \
            and len(hidden_state_sizes) != num_layers:
            raise ValueError('The number of hidden states provided must be the'
                             + ' same as the nubmer of layers.')

        if type(hidden_state_sizes) == int:
            self._hidden_state_sizes = [hidden_state_sizes] * self._num_layers
        else:
            self._hidden_state_sizes = hidden_state_sizes

        if task == 'classification':
            self._loss_function = tf.nn.softmax_cross_entropy_with_logits
        elif task == 'regression':
            self._loss_function = lambda logits, labels: tf.square((logits - labels))

        batch_in_shape = (None, None, self._input_size)
        batch_out_shape = (None, None, self._output_size)
        self.batch_in = tf.placeholder(
            tf.float32, shape=batch_in_shape, name='batch_in')
        self.batch_out = tf.placeholder(
            tf.float32, shape=batch_out_shape, name='batch_out')

        self._optimizer = tf.train.AdamOptimizer(1e-3)

    def load_variables(self, path='./hmlstm_ckpt'):
        if self._session is None:
            self._session = tf.Session()

            saver = tf.train.Saver()
            print('loading variables...')
            saver.restore(self._session, path)

    def save_variables(self, path='./hmlstm_ckpt'):
        saver = tf.train.Saver()
        print('saving variables...')
        saver.save(self._session, path)

    def network(self, reuse):
        batch_size = tf.shape(self.batch_in)[1]
        sequence_length = tf.shape(self.batch_in)[0]
        #
        # elem_len = (sum(self._hidden_state_sizes) * 2) + self._num_layers
        # initial_states_batch = tf.zeros([batch_size, elem_len])  # [B, H]

        hmlstm_out, states, indicators = dynamic_hmlstm(
            input_sequence_batch=tf.transpose(self.batch_in, [1, 0, 2]),
            retain_state_mask_sequence_batch=None,
            initial_states_batch=None,
            scope=None,
            num_layers=self._num_layers,
            input_size=self._input_size,
            hidden_state_sizes=self._hidden_state_sizes,
            reuse=reuse,
            embedding_size=self._embed_size,
            output_hidden_size=self._out_hidden_size,
            output_size=self._output_size
        )

        hmlstm_out = tf.reshape(hmlstm_out, shape=[batch_size * sequence_length, self._embed_size])

        predictions = output_module(hmlstm_out, output_size=self._output_size, out_hidden_size=self._out_hidden_size, reuse=reuse)

        predictions = tf.reshape(predictions, shape=[batch_size, sequence_length, self._output_size])
        predictions = tf.transpose(predictions, [1, 0, 2])

        to_map = tf.concat((predictions, self.batch_out), axis=2)    # [T, B, O]

        to_map = tf.Print(to_map, [tf.shape(predictions), tf.shape(self.batch_out)], "predictions and batch out shape")

        def map_for_loss(elem):
            splits = tf.constant([-1, self._output_size])
            predictions, targets = array_ops.split(value=elem,
                                                   num_or_size_splits=splits,
                                                   axis=1)

            loss_args = {'logits': predictions, 'labels': targets}
            loss = self._loss_function(**loss_args)

            return loss

        mapped = tf.map_fn(map_for_loss, to_map)                  # [T, B, _]

        mapped = tf.Print(mapped, [tf.shape(mapped)], "loss shape (new)")
        mapped = tf.Print(mapped, [tf.shape(mapped[:, :, :])], "loss shape - alt-view (new)")

        # mapped has different shape for task 'regression' and 'classification'
        loss = tf.reduce_mean(mapped[:, :, :])  # scalar
        train = self._optimizer.minimize(loss)

        return train, loss, indicators, predictions

    def train(self,
              batches_in,
              batches_out,
              variable_path='./hmlstm_ckpt',
              load_vars_from_disk=False,
              save_vars_to_disk=False,
              epochs=3):
        """
        Train the network.

        params:
        ---
        batches_in: a 4 dimensional numpy array. The dimensions should be
            [num_batches, batch_size, num_timesteps, input_size]
            These represent the input at each time step for each batch.
        batches_out: a 4 dimensional numpy array. The dimensions should be
            [num_batches, batch_size, num_timesteps, output_size]
            These represent the output at each time step for each batch.
        variable_path: the path to which variable values will be saved and/or
            loaded
        load_vars_from_disk: bool, whether to load variables prior to training
        load_vars_from_disk: bool, whether to save variables after training
        epochs: integer, number of epochs
        """

        optim, loss, _, _ = self._get_graph()

        if not load_vars_from_disk:
            if self._session is None:

                self._session = tf.Session()
                init = tf.global_variables_initializer()
                self._session.run(init)
        else:
            self.load_variables(variable_path)
        for epoch in range(epochs):
            print('Epoch %d' % epoch)
            for batch_in, batch_out in zip(batches_in, batches_out):

                ops = [optim, loss]
                feed_dict = {
                    self.batch_in: np.swapaxes(batch_in, 0, 1),
                    self.batch_out: np.swapaxes(batch_out, 0, 1),
                }
                _, _loss = self._session.run(ops, feed_dict)
                print('loss:', _loss)

        self.save_variables(variable_path)

    def predict(self, batch, variable_path='./hmlstm_ckpt',
                return_gradients=False):
        """
        Make predictions.

        params:
        ---
        batch: batch for which to make predictions. should have dimensions
            [batch_size, num_timesteps, output_size]
        variable_path: string. If there is no active session in the network
            object (i.e. it has not yet been used to train or predict, or the
            tensorflow session has been manually closed), variables will be
            loaded from the provided path. Otherwise variables already present
            in the session will be used.

        returns:
        ---
        predictions for the batch
        """

        batch = np.array(batch)
        _, _, _, predictions = self._get_graph()

        self._load_vars(variable_path)

        # batch_out is not used for prediction, but needs to be fed in
        batch_out_size = (batch.shape[1], batch.shape[0], self._output_size)
        gradients = tf.gradients(predictions[-1:, :], self.batch_in)
        _predictions, _gradients = self._session.run([predictions, gradients], {
            self.batch_in: np.swapaxes(batch, 0, 1),
            self.batch_out: np.zeros(batch_out_size),
        })

        if return_gradients:
            return tuple(np.swapaxes(r, 0, 1) for
                         r in (_predictions, _gradients[0]))

        return np.swapaxes(_predictions, 0, 1)

    def predict_boundaries(self, batch, variable_path='./hmlstm_ckpt'):
        """
        Find indicator values for every layer at every timestep.

        params:
        ---
        batch: batch for which to make predictions. should have dimensions
            [batch_size, num_timesteps, output_size]
        variable_path: string. If there is no active session in the network
            object (i.e. it has not yet been used to train or predict, or the
            tensorflow session has been manually closed), variables will be
            loaded from the provided path. Otherwise variables already present
            in the session will be used.

        returns:
        ---
        indicator values for ever layer at every timestep
        """

        batch = np.array(batch)
        _, _, indicators, _ = self._get_graph()

        self._load_vars(variable_path)

        # batch_out is not used for prediction, but needs to be fed in
        batch_out_size = (batch.shape[1], batch.shape[0], self._output_size)
        _indicators = self._session.run(indicators, {
            self.batch_in: np.swapaxes(batch, 0, 1),
            self.batch_out: np.zeros(batch_out_size)
        })

        return np.array(_indicators)

    def _get_graph(self):
        if self._graph is None:
            self._graph = self.network(reuse=False)
        return self._graph

    def _load_vars(self, variable_path):
        if self._session is None:
            try:
                self.load_variables(variable_path)
            except:
                raise RuntimeError('Session unitialized and no variables saved'
                                   + ' at provided path %s' % variable_path)


def create_multicell(batch_size, num_layers, input_size, hidden_state_sizes, reuse):
    def hmlstm_cell(layer):
        if layer == 0:
            h_below_size = input_size
        else:
            h_below_size = hidden_state_sizes[layer - 1]

        if layer == num_layers - 1:
            # doesn't matter, all zeros, but for convenience with summing
            # so the sum of ha sizes is just sum of hidden states
            h_above_size = hidden_state_sizes[0]
        else:
            h_above_size = hidden_state_sizes[layer + 1]

        return HMLSTMCell(hidden_state_sizes[layer], batch_size,
                          h_below_size, h_above_size, reuse)

    hmlstm = MultiHMLSTMCell(
        [hmlstm_cell(l) for l in range(num_layers)], reuse)

    return hmlstm


def split_out_cell_states(accum, hidden_state_sizes, num_layers):
    '''
    accum: [B, H], i.e. [B, sum(h_l) * 2 + num_layers]


    cell_states: a list of ([B, h_l], [B, h_l], [B, 1]), with length L
    '''
    splits = []
    for size in hidden_state_sizes:
        splits += [size, size, 1]

    split_states = array_ops.split(value=accum,
                                   num_or_size_splits=splits, axis=1)

    cell_states = []
    for l in range(num_layers):
        c = split_states[(l * 3)]
        h = split_states[(l * 3) + 1]
        z = split_states[(l * 3) + 2]
        cell_states.append(HMLSTMState(c=c, h=h, z=z))

    return cell_states

def get_h_aboves(hidden_states, batch_size, hmlstm):
    '''
    hidden_states: [[B, h_l] for l in range(L)]

    h_aboves: [B, sum(ha_l)], ha denotes h_above
    '''
    concated_hs = array_ops.concat(hidden_states[1:], axis=1)

    h_above_for_last_layer = tf.zeros(
        [batch_size, hmlstm._cells[-1]._h_above_size], dtype=tf.float32)

    h_aboves = array_ops.concat(
        [concated_hs, h_above_for_last_layer], axis=1)

    return h_aboves


def gate_input(hidden_states, hidden_state_sizes, num_layers, reuse):
    '''
    gate the incoming hidden states
    hidden_states: [B, sum(h_l)]

    gated_input: [B, sum(h_l)]
    '''
    with vs.variable_scope('gates', reuse=reuse):
        gates = []  # [[B, 1] for l in range(L)]
        for l in range(num_layers):
            gate = tf.layers.dense(hidden_states, units=1, activation=tf.nn.sigmoid, use_bias=False, name='gate_%s' % l)
            gates.append(gate)

        split = array_ops.split(
            value=hidden_states,
            num_or_size_splits=hidden_state_sizes,
            axis=1)

        gated_list = []  # [[B, h_l] for l in range(L)]
        for gate, hidden_state in zip(gates, split):
            gated_list.append(tf.multiply(gate, hidden_state))

        gated_input = tf.concat(gated_list, axis=1)  # [B, sum(h_l)]
    return gated_input


def embed_input(gated_input, embedding_size, reuse):
    '''
    gated_input: [B, sum(h_l)]

    embedding: [B, E], i.e. [B, embed_size]
    '''
    with vs.variable_scope('embedding', reuse=reuse):

        embedding = tf.layers.dense(gated_input, units=embedding_size, activation=tf.nn.relu, use_bias=False, name='embedding')

    return embedding


def output_module(embedding, output_size, out_hidden_size, reuse):
    '''
    embedding: [B, E]
    outcome: [B, output_size]

    loss: [B, output_size] or [B, 1]
    prediction: [B, output_size]
    '''
    with vs.variable_scope('output_module', reuse=reuse):

        l1 = tf.layers.dense(embedding, units=out_hidden_size, activation=tf.nn.tanh, name='output_l1')
        l2 = tf.layers.dense(l1, units=out_hidden_size, activation=tf.nn.tanh, name='output_l2')
        prediction = tf.layers.dense(l2, units=output_size, activation=None, name='prediction')

    return prediction


def dynamic_hmlstm(input_sequence_batch, retain_state_mask_sequence_batch, initial_states_batch,
                 scope, num_layers, input_size, hidden_state_sizes, embedding_size, output_size, output_hidden_size, reuse, init_scale=1.0):

    # Initial states batch should be [B, H], i.e. [B, sum(h_l) * 2 + num_layers]
    # Make it with
    # elem_len = (sum(hidden_state_sizes) * 2) + num_layers
    # initial = tf.zeros([batch_size, elem_len])  # [B, H]

    with vs.variable_scope(scope, reuse=reuse):

        batch_in = tf.transpose(input_sequence_batch, [1, 0, 2])
        batch_size = tf.shape(batch_in)[1]
        hmlstm = create_multicell(batch_size, num_layers, input_size, hidden_state_sizes, reuse)

        def scan_rnn(accum, elem):
            # each element is the set of all hidden states from the previous
            # time step
            cell_states = split_out_cell_states(accum, hidden_state_sizes, num_layers)

            h_aboves = get_h_aboves([cs.h for cs in cell_states],
                                         batch_size, hmlstm)  # [B, sum(ha_l)]
            # [B, I] + [B, sum(ha_l)] -> [B, I + sum(ha_l)]
            hmlstm_in = array_ops.concat((elem, h_aboves), axis=1)
            _, state = hmlstm(hmlstm_in, cell_states)
            # a list of (c=[B, h_l], h=[B, h_l], z=[B, 1]) ->
            # a list of [B, h_l + h_l + 1]
            concated_states = [array_ops.concat(tuple(s), axis=1) for s in state]
            return array_ops.concat(concated_states, axis=1)  # [B, H]

        # denote 'elem_len' i.e. total size of all state components as 'H'
        elem_len = (sum(hidden_state_sizes) * 2) + num_layers
        initial = tf.zeros([batch_size, elem_len])  # [B, H]

        initial = tf.Print(initial, [tf.shape(batch_in), tf.shape(initial)], "batch in shape and initial state shape")

        states = tf.scan(scan_rnn, batch_in, initial)  # [T, B, H]

        def map_indicators(elem):
            state = split_out_cell_states(elem, hidden_state_sizes, num_layers)
            return tf.concat([l.z for l in state], axis=1)

        raw_indicators = tf.map_fn(map_indicators, states)  # [T, B, L]
        indicators = tf.transpose(raw_indicators, [1, 2, 0])  # [B, L, T]

        def map_output(elem):
            cell_states = elem
            hs = [s.h for s in split_out_cell_states(cell_states, hidden_state_sizes, num_layers)]
            gated = gate_input(tf.concat(hs, axis=1), hidden_state_sizes, num_layers, reuse)  # [B, sum(h_l)]
            embedded = embed_input(gated, embedding_size, reuse)  # [B, E]
            return embedded

        outputs = tf.map_fn(map_output, states)  # [T, B, _]
        outputs = tf.transpose(outputs, [1, 0, 2])  # [B, T, _]

    return outputs, states, indicators
