import numpy as np
import tensorflow as tf

from ..tf_parametrization import TensorflowParametrization

TRAINING = True


class MonteCarloDropoutResnetOneRegression(TensorflowParametrization):

    def __init__(self, **kwargs):
        TensorflowParametrization.__init__(self)
        self.samples = kwargs.get('samples')
        self.keep_probability = kwargs.get('dropout')
        self.seed = kwargs.get('seed')

    def test(self, state):
        regression = TensorflowParametrization.test(self, np.repeat(state, self.samples, axis=0))
        regression = regression[0]
        return np.squeeze(np.mean(regression, axis=1)), np.squeeze(np.var(regression, axis=1))

    def architecture(self):
        model = tf.layers.conv2d(self._preprocessed_state,
                                 filters=32,
                                 kernel_size=5,
                                 strides=2,
                                 padding='same',
                                 kernel_initializer=tf.keras.initializers.he_normal(seed=self.seed))
        model = tf.layers.dropout(model,
                                  rate=0.1,
                                  training=TRAINING,
                                  seed=self.seed)
        model = tf.layers.max_pooling2d(model,
                                        pool_size=3,
                                        strides=2)

        # residual block
        residual_1 = tf.layers.batch_normalization(model)  # TODO: check if the defaults in Tf are the same as in Keras
        residual_1 = tf.nn.relu(residual_1)
        residual_1 = tf.layers.conv2d(residual_1,
                                      filters=32,
                                      kernel_size=3,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=self.seed))
        residual_1 = tf.layers.dropout(residual_1,
                                       rate=0.1,
                                       seed=self.seed,
                                       training=TRAINING)
        residual_1 = tf.layers.batch_normalization(residual_1)
        residual_1 = tf.nn.relu(residual_1)
        residual_1 = tf.layers.conv2d(residual_1,
                                      filters=32,
                                      kernel_size=3,
                                      padding='same',
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=self.seed))
        residual_1 = tf.layers.dropout(residual_1,
                                       rate=0.1,
                                       seed=self.seed,
                                       training=TRAINING)
        # end residual block

        model = tf.layers.conv2d(model,
                                 filters=32,
                                 kernel_size=1,
                                 strides=2,
                                 padding='same',
                                 kernel_initializer=tf.keras.initializers.he_normal(seed=self.seed))
        model = tf.layers.dropout(model,
                                  rate=0.1,
                                  seed=self.seed,
                                  training=TRAINING)
        model = tf.keras.layers.add([residual_1, model])
        model = tf.layers.flatten(model)
        model = tf.layers.dropout(model,
                                  rate=0.1,
                                  seed=self.seed,
                                  training=TRAINING)
        model = tf.layers.dense(model,
                                units=64,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed))
        model = tf.layers.dropout(model,
                                  rate=0.5,
                                  seed=self.seed,
                                  training=TRAINING)
        model = tf.layers.dense(model,
                                units=32,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed))
        model = tf.layers.dropout(model,
                                  rate=0.5,
                                  seed=self.seed,
                                  training=TRAINING)

        model = tf.layers.dense(model, self.action_tensor.shape[1])

        with tf.name_scope('losses'):
            loss = tf.losses.mean_squared_error(model, self.action_tensor)
            tf.summary.scalar('mse', loss)

        return [model], loss

