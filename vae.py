
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import os
import datetime
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

class VAE:

    def __init__(self, latent_dim=128, restore_from_dir=None):
        print("VAE latent dim {}".format(latent_dim))
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.latent_dim = latent_dim
        self.save_prefix = 'vae'
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

        self.writer = tf.summary.FileWriter("{}/{}".format(self.tensorboard_path, self.identifier), self.graph)

    def _restore_model(self, from_dir):
        print("\n\nRestoring Model from {}".format(from_dir))
        with self.graph.as_default():
            # self.saver = tf.train.import_meta_graph(os.path.join(self.save_file_path, self.save_prefix + '.meta'))
            self.saver.restore(self.sess, tf.train.latest_checkpoint(from_dir))

    def _build_model(self, restore_from_dir=None):

        with self.graph.as_default():
            vae_scope = 'VAE'
            with tf.variable_scope(vae_scope):
                variance_scaling = tf.contrib.layers.variance_scaling_initializer()
                xavier = tf.contrib.layers.xavier_initializer()

                # Building the encoder

                self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='x')

                encode_1 = tf.layers.Conv2D(filters=32, kernel_size=4, strides=2,
                                            padding='valid', activation=tf.nn.relu,
                                            kernel_initializer=variance_scaling,
                                            name='encode_1')(self.x)
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
                vae_flatten = tf.layers.Flatten()(encode_4)

                z_mean = tf.layers.Dense(units=self.latent_dim, name='z_mean')(vae_flatten)
                z_log_var = tf.layers.Dense(units=self.latent_dim, name='z_log_var')(vae_flatten)

                # Sampler: Normal (gaussian) random distribution
                with tf.name_scope("sampling"):
                    eps = tf.random_normal(tf.shape(z_log_var), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
                    self.z_encoded = z_mean + tf.exp(z_log_var / 2) * eps

                # we instantiate these layers separately so as to reuse them later
                decode_dense_1 = tf.layers.Dense(units=1024, activation=tf.nn.relu, name='decode_dense_1')

                def decode_reshape(tensor):
                    return tf.reshape(tensor=tensor, shape=[-1, 1, 1, 1024])

                decode_1 = tf.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2,
                                                     padding='valid', activation=tf.nn.relu,
                                                     kernel_initializer=variance_scaling,
                                                     name='decode_1')

                decode_2 = tf.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2,
                                                     padding='valid', activation=tf.nn.relu,
                                                     kernel_initializer=variance_scaling,
                                                     name='decode_2')

                decode_3 = tf.layers.Conv2DTranspose(filters=32, kernel_size=6, strides=2,
                                                     padding='valid', activation=tf.nn.relu,
                                                     kernel_initializer=variance_scaling,
                                                     name='decode_3')

                decode_4 = tf.layers.Conv2DTranspose(filters=3, kernel_size=6, strides=2,
                                                     padding='valid', activation=tf.nn.sigmoid,
                                                     kernel_initializer=xavier, bias_initializer=xavier,
                                                     name='decode_4')

                with tf.name_scope('decode_encoded'):
                    decode_dense_encoded_1 = decode_dense_1(self.z_encoded)
                    decode_encoded_1 = decode_1(decode_reshape(decode_dense_encoded_1))
                    decode_encoded_2 = decode_2(decode_encoded_1)
                    decode_encoded_3 = decode_3(decode_encoded_2)
                    self.decoded_encoded = decode_4(decode_encoded_3)

                with tf.name_scope('decode_given'):
                    self.z_given = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name='z_given')
                    decode_dense_given_1 = decode_dense_1(self.z_given)
                    decode_given_1 = decode_1(decode_reshape(decode_dense_given_1))
                    decode_given_2 = decode_2(decode_given_1)
                    decode_given_3 = decode_3(decode_given_2)
                    self.decoded_given = decode_4(decode_given_3)

                with tf.name_scope('loss'):
                    with tf.name_scope('kl_div_loss'):
                        self.kl_div_loss = tf.reduce_mean(
                            -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
                        )
                        tf.summary.scalar('kl_div_loss', self.kl_div_loss)

                    with tf.name_scope('reconstruction_loss'):
                        self.reconstruction_loss = tf.reduce_mean(
                            1000 * tf.reduce_mean(tf.square(self.x - self.decoded_encoded), axis=[1, 2, 3])
                        )
                        tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)

                    self.loss = self.kl_div_loss + self.reconstruction_loss
                    tf.summary.scalar('total_loss', self.loss)

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            # self.check_op = tf.add_check_numerics_ops()
            # print("\n\nCollection: {}".format(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vae_scope)))
            # Initialize the variables (i.e. assign their default value)
            self.tf_summaries_merged = tf.summary.merge_all()

            save_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vae_scope)
            save_var_list.append(self.global_step)
            self.saver = tf.train.Saver(var_list=save_var_list,
                                        max_to_keep=5,
                                        keep_checkpoint_every_n_hours=1)

            self.init = tf.global_variables_initializer()

        if restore_from_dir:
            self._restore_model(restore_from_dir)
        else:
            self.sess.run(self.init)

    def train_on_input_fn(self, input_fn, steps=None, save_every_n_steps=30000):

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
                    batch_x = sess.run(iter)
                except tf.errors.OutOfRangeError:
                    print("Input_fn ended at step {}".format(local_step))
                    break

                # Train
                feed_dict = {self.x: batch_x}
                _, l, kl, r, summaries, global_step = sess.run([self.train_op,
                                                                self.loss,
                                                                self.kl_div_loss,
                                                                self.reconstruction_loss,
                                                                self.tf_summaries_merged,
                                                                self.global_step],
                                                               feed_dict=feed_dict)

                self.writer.add_summary(summaries, local_step)

                if local_step % 50 == 0 or local_step == 1:
                    print('Step %i, Loss: %f, KL div: %f, Reconstr: %f' % (global_step, l, kl, r))

                if local_step % save_every_n_steps == 0:
                    self.save_model()

                if steps and local_step >= steps:
                    print("Completed {} steps".format(steps))
                    break

                local_step += 1

    def encode_frames(self, float_frames):
        return self.sess.run(self.z_encoded, feed_dict={self.x: float_frames})

    def decode_frames(self, z_codes):
        return self.sess.run(self.decoded_given, feed_dict={self.z_given: z_codes})

    def encode_decode_frames(self, float_frames):
        return self.sess.run(self.decoded_encoded, feed_dict={self.x: float_frames})

    def save_model(self):
        if not os.path.exists(self.save_file_path):
            os.makedirs(self.save_file_path, exist_ok=True)

        save_path = self.saver.save(sess=self.sess,
                                    save_path=os.path.join(self.save_file_path, self.save_prefix),
                                    write_meta_graph=self.save_metagraph,
                                    global_step=self.global_step)

        self.save_metagraph = False

        print("VAE Model saved in path: {}".format(save_path))

    def __del__(self):
        if self.writer:
            self.writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()


