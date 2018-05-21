
from __future__ import print_function

from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.models import Model
from keras import backend as K
from keras.losses import mean_squared_error

batch_size = 128
latent_dim = 64
action_dim = 5

class StateRNN():

    def __init__(self):

        rnn_x = Input(shape=(None, latent_dim + action_dim))
        lstm1 = LSTM(256, return_sequences=True, return_state=True)
        lstm1_output, _, _ = lstm1(rnn_x)
        lstm2 = LSTM(256, return_sequences=True, return_state=True)
        lstm2_output, _, _ = lstm2(lstm1_output)
        output = TimeDistributed(Dense(units=latent_dim, activation=None))(lstm2_output)
        self.rnn = Model(rnn_x, output)

        def cost(y_true, y_pred):
            # Compute mse for each frame.
            mse = mean_squared_error(y_true=y_true,y_pred=y_pred)
            print("MSE SHAPE: {}".format(mse.shape))

            mask = K.sign(K.max(K.abs(y_true), axis=2))
            mse *= mask
            # Average over actual sequence lengths.
            mse = K.sum(mse, 1)
            mse /= K.sum(mask,1)
            return K.mean(mse)

        self.rnn.compile(optimizer='rmsprop', loss=mean_squared_error)
        self.rnn.summary()

    def save_model(self, path):
        self.rnn.save_weights(path)

    def load_model(self, path):
        self.rnn.load_weights(path)

