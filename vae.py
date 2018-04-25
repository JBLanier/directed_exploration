'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Lambda, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import cifar10
from keras import objectives

batch_size = 32
latent_dim = 32
epsilon_std = 1.0

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class VAE():

    def __init__(self):

        x = Input(shape=(64, 64, 3))
        encode_1 = Conv2D(filters=32, kernel_size=4, strides=2,
                        padding='valid', activation='relu', name='encode_1')(x)
        encode_2 = Conv2D(filters=64, kernel_size=4, strides=2,
                        padding='valid', activation='relu', name='encode_2')(encode_1)
        encode_3 = Conv2D(filters=128, kernel_size=4, strides=2,
                        padding='valid', activation='relu', name='encode_3')(encode_2)
        encode_4 = Conv2D(filters=256, kernel_size=4, strides=2,
                        padding='valid', activation='relu', name='encode_4')(encode_3)

        vae_flatten = Flatten()(encode_4)

        # encode_dense_1 = Dense(units=1024, activation='relu', name='encode_dense_1')(vae_flatten)
        # encode_dense_2 = Dense(units=512, activation='relu', name='encode_dense_2')(encode_dense_1)


        z_mean = Dense(latent_dim)(vae_flatten)
        z_log_var = Dense(latent_dim)(vae_flatten)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decode_dense_1 = Dense(units=1024, activation='relu', name='decode_dense_1')
        # decode_dense_2 = Dense(units=1024, activation='relu', name='decode_dense_2')


        decode_reshape = Reshape(target_shape=(1, 1, 1024))

        decode_1 = Conv2DTranspose(filters=128, kernel_size=5, strides=2,
                                 padding='valid', activation='relu', name='decode_1')

        decode_2 = Conv2DTranspose(filters=64, kernel_size=5, strides=2,
                                 padding='valid', activation='relu', name='decode_2')

        decode_3 = Conv2DTranspose(filters=32, kernel_size=6, strides=2,
                                 padding='valid', activation='relu', name='decode_3')

        decode_4 = Conv2DTranspose(filters=3, kernel_size=6, strides=2,
                                 padding='valid', activation='sigmoid', name='decode_4')

        # instantiate VAE model
        decode_dense_1_model = decode_dense_1(z)
        # decode_dense_2_model = decode_dense_2(decode_dense_1_model)
        decode_reshape_model = decode_reshape(decode_dense_1_model)
        decode_1_model = decode_1(decode_reshape_model)
        decode_2_model = decode_2(decode_1_model)
        decode_3_model = decode_3(decode_2_model)
        decode_4_model = decode_4(decode_3_model)
        self.vae = Model(x, decode_4_model)

        # encoder
        self.vae_encoder = Model(x,z)

        # decoder
        vae_z_input = Input(shape=(latent_dim,))
        decode_dense_1_dec = decode_dense_1(vae_z_input)
        # decode_dense_2_dec = decode_dense_2(decode_dense_1_dec)
        decode_reshape_dec = decode_reshape(decode_dense_1_dec)
        decode_1_dec = decode_1(decode_reshape_dec)
        decode_2_dec = decode_2(decode_1_dec)
        decode_3_dec = decode_3(decode_2_dec)
        decode_4_dec = decode_4(decode_3_dec)
        self.vae_decoder = Model(vae_z_input, decode_4_dec)

        def vae_loss(x, x_decoded_mean):
            r_loss = 1000 * K.mean(K.square(x - x_decoded_mean), axis = [1,2,3])
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return r_loss + kl_loss

        self.vae.compile(optimizer='rmsprop', loss=vae_loss)
        self.vae.summary()


    def train_on_batch(self, batch):
        self.vae.train_on_batch(x=batch, y=batch)

    def encode(self, x):
        return self.vae_encoder.predict([x])

    def decode(self, z):
        return self.vae_decoder.predict([z])

    def save_model(self, path):
        self.vae.save_weights(path)

    def load_model(self, path):
        self.vae.load_weights(path)

        # # train the VAE on MNIST digits
        # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        #
        # x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
        # x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))
        #
        # print("Resizing Images")
        # r_x_train = np.empty((x_train.shape[0], 64, 64, 3), dtype='uint8')
        # for i in range(x_train.shape[0]):
        #     r_x_train[i] = cv2.resize(x_train[i], (64, 64))
        # x_train = r_x_train
        #
        # r_x_test = np.empty((x_test.shape[0], 64, 64, 3), dtype='uint8')
        # for i in range(x_test.shape[0]):
        #     r_x_test[i] = cv2.resize(x_test[i], (64, 64))
        # x_test = r_x_test
        #
        #
        # x_train = x_train.astype('float32') / 255.
        # x_test = x_test.astype('float32') / 255.
        # #
        #
        #
        # self.vae.fit(x_train, x_train,
        #         shuffle=True,
        #         epochs=20,
        #         batch_size=batch_size)
        #
        # for i in range(30):
        #     cv2.imshow('orig', cv2.cvtColor(x_test[i], cv2.COLOR_RGB2BGR))
        #     cv2.imshow('reconstructed', cv2.cvtColor(np.squeeze(vae.predict(np.expand_dims(x_test[i],0))), cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(0)
        #
        # cv2.destroyAllWindows()


    # build a model to project inputs on the latent space
    # encoder = Model(x, z_mean)
    #
    # # display a 2D plot of the digit classes in the latent space
    # x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    # plt.figure(figsize=(6, 6))
    # plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    # plt.colorbar()
    # plt.show()
    #
    # # build a digit generator that can sample from the learned distribution
    # decoder_input = Input(shape=(latent_dim,))
    # _h_decoded = decoder_h(decoder_input)
    # _x_decoded_mean = decoder_mean(_h_decoded)
    # generator = Model(decoder_input, _x_decoded_mean)
    #
    # # display a 2D manifold of the digits
    # n = 15  # figure with 15x15 digits
    # digit_size = 28
    # figure = np.zeros((digit_size * n, digit_size * n))
    # # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    # grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    # grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    #
    # for i, yi in enumerate(grid_x):
    #     for j, xi in enumerate(grid_y):
    #         z_sample = np.array([[xi, yi]])
    #         x_decoded = generator.predict(z_sample)
    #         digit = x_decoded[0].reshape(digit_size, digit_size)
    #         figure[i * digit_size: (i + 1) * digit_size,
    #                j * digit_size: (j + 1) * digit_size] = digit
    #
    # plt.figure(figsize=(10, 10))
    # plt.imshow(figure, cmap='Greys_r')
    # plt.show()
