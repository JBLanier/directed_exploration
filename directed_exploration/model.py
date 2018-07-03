from abc import ABC, abstractmethod
import tensorflow as tf
import datetime
import os
import logging

logger = logging.getLogger(__name__)

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

    def __init__(self, save_prefix, working_dir, sess=None, graph=None, summary_writer=None):

        if not sess:
            sess = tf.get_default_session()
        if not graph:
            graph = tf.get_default_graph()

        self.graph = graph
        self.sess = sess
        self.save_prefix = save_prefix
        date_identifier = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        restore_from_dir = None

        self.save_file_path = os.path.join(working_dir, self.save_prefix)
        if os.path.exists(os.path.join(self.save_file_path, 'checkpoint')):

            self.save_metagraph = False
            restore_from_dir = self.save_file_path
        else:

            self.save_metagraph = True

        self.identifier = os.path.basename(os.path.normpath(self.save_file_path))

        if summary_writer is not None:
            self.writer = summary_writer
        else:
            self.writer = tf.summary.FileWriter(working_dir)

        self._build_model(restore_from_dir)

    @abstractmethod
    def _build_model(self, restore_from_dir=None):
        self.saver = None
        self.local_step = None
        if restore_from_dir:
            self._restore_model(restore_from_dir)

    def _restore_model(self, from_dir):
        logger.info("Restoring {} model from {}".format(self.save_prefix, from_dir))
        self.saver.restore(self.sess, tf.train.latest_checkpoint(from_dir))

    def save_model(self):
        if not os.path.exists(self.save_file_path):
            os.makedirs(self.save_file_path, exist_ok=True)

        save_path = self.saver.save(sess=self.sess,
                                    save_path=os.path.join(self.save_file_path, self.save_prefix),
                                    write_meta_graph=self.save_metagraph,
                                    global_step=self.local_step)

        self.save_metagraph = False

        logger.info("{} model saved in path: {}".format(self.save_prefix, save_path))

    def __del__(self):
        logger.info("del called")
        if self.writer:
            self.writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("exit called")

        if self.writer:
            self.writer.close()
