from abc import ABC, abstractmethod
import tensorflow as tf
import datetime
import os


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


class Model(ABC):

    def __init__(self, save_prefix, restore_from_dir=None, sess=None, graph=None):

        if not sess:
            sess = tf.get_default_session()
        if not graph:
            graph = tf.get_default_graph()

        self.graph = graph
        self.sess = sess
        self.save_prefix = save_prefix
        date_identifier = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if restore_from_dir:
            self.save_metagraph = False
            self.identifier = os.path.basename(os.path.normpath(restore_from_dir))
            self.save_file_path = restore_from_dir
        else:
            self.save_metagraph = True
            self.identifier = '{}_{}'.format(self.save_prefix, date_identifier)
            self.save_file_path = './' + self.identifier

        self.tensorboard_path = './tensorboard'
        self.writer = tf.summary.FileWriter("{}/{}".format(self.tensorboard_path, self.identifier), self.graph)
        self._build_model(restore_from_dir)

    @abstractmethod
    def _build_model(self, restore_from_dir=None):
        self.saver = None
        self.local_step = None
        if restore_from_dir:
            self._restore_model(restore_from_dir)

    def _restore_model(self, from_dir):
        print("\n\nRestoring {} model from {}".format(self.save_prefix, from_dir))
        self.saver.restore(self.sess, tf.train.latest_checkpoint(from_dir))

    def save_model(self):
        if not os.path.exists(self.save_file_path):
            os.makedirs(self.save_file_path, exist_ok=True)

        save_path = self.saver.save(sess=self.sess,
                                    save_path=os.path.join(self.save_file_path, self.save_prefix),
                                    write_meta_graph=self.save_metagraph,
                                    global_step=self.local_step)

        self.save_metagraph = False

        print("{} model saved in path: {}".format(self.save_prefix, save_path))

    def __del__(self):
        if self.writer:
            self.writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()
