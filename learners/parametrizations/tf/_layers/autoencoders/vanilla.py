import tensorflow as tf
from .convolutional import ConvolutionalAutoencoder


class VanillaAutoencoder(ConvolutionalAutoencoder):
    def __init__(self, x, filter_size, loss_function):
        ConvolutionalAutoencoder.__init__(self, x=x, filter_size=filter_size,
                                          loss_function=loss_function, latent_size=0)

    def _encoder(self, x):
        nn = ConvolutionalAutoencoder._encoder(self, x)
        nn = tf.layers.conv2d(nn, filters=self.filter_size, kernel_size=3, strides=1, padding='valid')
        nn = tf.layers.batch_normalization(nn, axis=3)

        return nn

