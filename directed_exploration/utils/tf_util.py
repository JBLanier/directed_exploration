import tensorflow as tf


def log_tensorboard_scalar_summaries(summary_writer, values, tags, step):

    summary = tf.Summary()
    for tag, value in zip(tags, values):
        summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, step)
    summary_writer.flush()
