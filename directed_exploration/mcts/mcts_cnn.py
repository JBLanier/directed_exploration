import tensorflow as tf
from directed_exploration.model import Model
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MCTS_CNN(Model):
    def __init__(self, obs_space, action_space, working_dir=None, sess=None, graph=None, summary_writer=None):
        logger.info("MCTS CNN Observation space: {} Actions space: {}".format(obs_space.shape, action_space.n))

        self.obs_space = obs_space
        self.action_space = action_space

        save_prefix = "MCTS_CNN_obs_{}_act_{}".format(obs_space.shape, action_space.shape)

        super().__init__(save_prefix, working_dir, sess, graph, summary_writer=summary_writer)

    def _build_model(self, restore_from_dir=None):

        with self.graph.as_default():
            model_scope = 'MCTS_CNN_MODEL'
            with tf.variable_scope(model_scope):
                variance_scaling = tf.contrib.layers.variance_scaling_initializer()

                self.scaled_obs_input = tf.placeholder(tf.float32, shape=[None, *self.obs_space.shape],
                                                       name='scaled_obs')

                net = tf.layers.Conv2D(filters=32, kernel_size=7, strides=4,
                                       padding='valid', activation=tf.nn.relu,
                                       kernel_initializer=variance_scaling,
                                       name='conv1')(self.scaled_obs_input)

                net = tf.layers.Conv2D(filters=64, kernel_size=5, strides=2,
                                       padding='valid', activation=tf.nn.relu,
                                       kernel_initializer=variance_scaling,
                                       name='conv2')(net)

                net = tf.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                       padding='valid', activation=tf.nn.relu,
                                       kernel_initializer=variance_scaling,
                                       name='conv3')(net)

                net = tf.layers.flatten(net)

                net = tf.layers.dense(inputs=net, units=512, name='dense1', activation=tf.nn.relu,
                                      kernel_initializer=variance_scaling)

                self.value_out = tf.reshape(tf.layers.dense(inputs=net, units=1, activation=None, name='value_out'),
                                            shape=[-1])

                policy_logits = tf.layers.dense(inputs=net, units=self.action_space.n, name='policy_logits')
                self.policy_out = tf.nn.softmax(logits=policy_logits, name='policy_out')

                with tf.name_scope('loss'):
                    with tf.name_scope('policy_loss'):
                        self.policy_targets = tf.placeholder(tf.float32,
                                                             shape=[None, self.action_space.n],
                                                             name='policy_targets')

                        self.policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.policy_targets,
                                                                                      logits=policy_logits))
                        tf.summary.scalar('MCTS_policy_cross_entropy_loss', self.policy_loss)

                    with tf.name_scope('value_loss'):
                        self.value_targets = tf.placeholder(tf.float32, shape=[None], name='value_targets')

                        self.value_loss = tf.losses.mean_squared_error(labels=self.value_targets,
                                                                       predictions=self.value_out)
                        tf.summary.scalar('MCTS_value_mse_loss', self.value_loss)

                    self.loss = self.policy_loss + self.value_loss
                    tf.summary.scalar('total_loss', self.loss)

            ops_scope = 'MCTS_CNN_OPS'
            with tf.variable_scope(ops_scope):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                self.local_step = tf.Variable(0, name='local_step', trainable=False)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.local_step)

                self.tf_summaries_merged = tf.summary.merge_all(scope=model_scope)

                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope)
                var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=ops_scope)

                self.saver = tf.train.Saver(var_list=var_list,
                                            max_to_keep=5,
                                            keep_checkpoint_every_n_hours=1)

            self.init = tf.variables_initializer(var_list=var_list, name='mcts_cnn_initializer')

        if restore_from_dir:
            self._restore_model(restore_from_dir)
        else:
            logger.debug("Running MCTS CNN local init\n")
            self.sess.run(self.init)

        self.writer.add_graph(self.graph)

    def train_on_batch(self, scaled_obs_batch, policy_targets, value_targets):

        feed_dict = {
            self.scaled_obs_input: scaled_obs_batch,
            self.value_targets: value_targets,
            self.policy_targets: policy_targets
        }

        _, loss, value_loss, policy_loss, summaries, step = self.sess.run([self.train_op,
                                                                           self.loss,
                                                                           self.value_loss,
                                                                           self.policy_loss,
                                                                           self.tf_summaries_merged,
                                                                           self.local_step],
                                                                          feed_dict=feed_dict)

        self.writer.add_summary(summaries, step)

        return loss, value_loss, policy_loss, step

    def predict_on_obs_batch(self, scaled_obs_batch):

        feed_dict = {self.scaled_obs_input: scaled_obs_batch}
        policy_prediction, value_prediction = self.sess.run([self.policy_out, self.value_out], feed_dict=feed_dict)
        return policy_prediction, value_prediction

    def predict_on_single_obs(self, scaled_obs):
        return (result[0] for result in self.predict_on_obs_batch(np.expand_dims(scaled_obs, axis=0)))
