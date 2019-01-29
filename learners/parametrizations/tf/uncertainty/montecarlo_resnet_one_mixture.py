import numpy as np
import tensorflow as tf

from ..tf_parametrization import TensorflowParametrization
from .._layers import resnet_1, MixtureDensityNetwork


class MonteCarloDropoutResnetOneMixture(TensorflowParametrization):
    def explore(self, state, horizon=1):
        pass

    def __init__(self, mixtures=3, **kwargs):
        TensorflowParametrization.__init__(self)
        self.mixtures = mixtures
        self.samples = kwargs.get('samples')
        self.dropout = kwargs.get('dropout')
        self.seed = kwargs.get('seed')

    def test(self, state, horizon=1):
        mdn = TensorflowParametrization.test(self, np.repeat(state, self.samples, axis=0))
        mdn = mdn[0]
        mixtures = np.mean(mdn[0], axis=0)
        means = np.mean(mdn[1], axis=0)
        variances = np.mean(mdn[2], axis=0)
        prediction = MixtureDensityNetwork.max_central_value(mixtures=np.squeeze(mixtures),
                                                             means=np.squeeze(means),
                                                             variances=np.squeeze(variances))
        return prediction[0], np.mean(prediction[1])  # FIXME: Is this the best way to add the variances?

    def architecture(self):
        model = resnet_1(self.state_tensor, keep_prob=1.0)
        model = tf.layers.dense(model, units=64, activation=tf.nn.tanh,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        model = tf.layers.dense(model, units=32, activation=tf.nn.tanh,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        model = tf.nn.dropout(model, keep_prob=self.dropout, seed=self.seed)

        loss, components, _ = MixtureDensityNetwork.create(model, self.action_tensor, number_mixtures=self.mixtures)

        return components, loss
