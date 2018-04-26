import keras.backend as K
import warnings


class TensorboardLogger:

    def __init__(self, model, log_dir='./logs', write_graph=True, write_grads=False, write_images=True):
        global tf, projector
        try:
            import tensorflow as tf
            from tensorflow.contrib.tensorboard.plugins import projector
        except ImportError:
            raise ImportError('You need the TensorFlow module installed to use TensorBoard.')

        if K.backend() != 'tensorflow':
            warnings.warn('You are not using the TensorFlow backend. '
                              'histogram_freq was set to 0')

        self.log_dir = log_dir
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.merged = None
        self.embeddings_freq = 0
        self.seen = 0
        self.set_model(model)

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()
        if self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads]
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        else:
                            # not possible to handle 3D convnets etc.
                            continue

                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(mapped_weight_name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq:
            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']

            embeddings = {layer.name: layer.weights[0]
                          for layer in self.model.layers
                          if layer.name in embeddings_layer_names}

            self.saver = tf.train.Saver(list(embeddings.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings.keys()}

            config = projector.ProjectorConfig()
            self.embeddings_ckpt_path = os.path.join(self.log_dir,
                                                     'keras_embedding.ckpt')

            for layer_name, tensor in embeddings.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    # def on_epoch_end(self, epoch, logs=None):
    #     logs = logs or {}
    #
    #     if not self.validation_data and self.histogram_freq:
    #         raise ValueError('If printing histograms, validation_data must be '
    #                          'provided, and cannot be a generator.')
    #     if self.validation_data and self.histogram_freq:
    #         if epoch % self.histogram_freq == 0:
    #
    #             val_data = self.validation_data
    #             tensors = (self.model.inputs +
    #                        self.model.targets +
    #                        self.model.sample_weights)
    #
    #             if self.model.uses_learning_phase:
    #                 tensors += [K.learning_phase()]
    #
    #             assert len(val_data) == len(tensors)
    #             val_size = val_data[0].shape[0]
    #             i = 0
    #             while i < val_size:
    #                 step = min(self.batch_size, val_size - i)
    #                 if self.model.uses_learning_phase:
    #                     # do not slice the learning phase
    #                     batch_val = [x[i:i + step] for x in val_data[:-1]]
    #                     batch_val.append(val_data[-1])
    #                 else:
    #                     batch_val = [x[i:i + step] for x in val_data]
    #                 assert len(batch_val) == len(tensors)
    #                 feed_dict = dict(zip(tensors, batch_val))
    #                 result = self.sess.run([self.merged], feed_dict=feed_dict)
    #                 summary_str = result[0]
    #                 self.writer.add_summary(summary_str, epoch)
    #                 i += self.batch_size
    #
    #     if self.embeddings_freq and self.embeddings_ckpt_path:
    #         if epoch % self.embeddings_freq == 0:
    #             self.saver.save(self.sess,
    #                             self.embeddings_ckpt_path,
    #                             epoch)
    #
    #     for name, value in logs.items():
    #         if name in ['batch', 'size']:
    #             continue
    #         summary = tf.Summary()
    #         summary_value = summary.value.add()
    #         summary_value.simple_value = value.item()
    #         summary_value.tag = name
    #         self.writer.add_summary(summary, epoch)
    #     self.writer.flush()
    #
    # def on_train_end(self, _):
    #     self.writer.close()

    def log_batch(self, batch_size, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.seen)
        self.writer.flush()

        self.seen += batch_size