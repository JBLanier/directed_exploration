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

batch_size = 128
latent_dim = 64
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


        z_mean = Dense(latent_dim, name='z_mean')(vae_flatten)
        z_log_var = Dense(latent_dim, name='z_log_var')(vae_flatten)

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

        def reconstruction_loss(x, x_decoded_mean):
            return 1000 * K.mean(K.square(x - x_decoded_mean), axis=[1, 2, 3])

        def kl_loss (x, x_decoded_mean):
            return  - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        def vae_loss(x, x_decoded_mean):
            return reconstruction_loss(x, x_decoded_mean) + kl_loss(x, x_decoded_mean)

        self.vae.compile(optimizer='rmsprop', loss=vae_loss, metrics=[kl_loss, reconstruction_loss])

        self.vae.summary()


    def train_on_batch(self, batch):
        values = self.vae.train_on_batch(x=batch, y=batch)
        return dict(zip(self.vae.metrics_names, values))


    def encode(self, x):
        return self.vae_encoder.predict([x])

    def decode(self, z):
        return self.vae_decoder.predict([z])

    def save_model(self, path):
        self.vae.save_weights(path)

    def load_model(self, path):
        self.vae.load_weights(path)

