import tensorflow as tf
from ._baseline import Autoencoder

LATENT_SIZE = 8


class DenoisingAutoencoder(Autoencoder):

    def __init__(self, x, latent_size=LATENT_SIZE, loss_function='mse', noise=0.1):
        self.noise = noise
        Autoencoder.__init__(self, x=x, latent_size=latent_size, loss_function=loss_function)
        self.x = self.x

    def _modify_input(self):
        pass

    def _encoder(self, x):
        nn = self.x + tf.random_normal(shape=tf.shape(self.x), stddev=self.noise)
        nn = tf.layers.dense(inputs=nn,
                             units=self.latent_size,
                             activation=tf.nn.tanh)
        return nn

    def _decoder(self, z):
        size = self.x.get_shape().as_list()[1]
        nn = tf.layers.dense(inputs=z,
                             units=size)
        nn = tf.reshape(nn, (-1, size))
        return nn

    def _loss(self):
        if self.loss_function == 'cross_entropy':
            reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.decoder)
        elif self.loss_function == 'mse':
            # MDN reconstruction_loss = tf.reduce_mean(tf.square(self.x - self.decoder))
            reconstruction_loss = tf.reduce_mean(tf.square(self.x - self.decoder), axis=1)
            # reconstruction_loss = tf.norm(self.x - self.decoder)
            # MDN tf.summary.scalar('rec_loss', reconstruction_loss)
            tf.summary.scalar('rec_loss', tf.reduce_mean(reconstruction_loss))
        else:
            raise NotImplementedError()

        return reconstruction_loss

