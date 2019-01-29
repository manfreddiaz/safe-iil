import tensorflow as tf
from .variational import VariationalAutoencoder


class ConditionalVariationalAutoencoder(VariationalAutoencoder):

    def __init__(self, x, y, filter_size, latent_size, loss_function):
        self.y = y
        VariationalAutoencoder.__init__(self, x, filter_size, latent_size, loss_function)

    def _conditional_input(self):
        x_y = tf.fill([self.x.shape[0], self.x.shape[1], self.x.shape[2], 1], tf.squeeze(self.y))
        x_y = tf.concat([self.x, x_y], axis=-1)

        return x_y

    def _conditional_latent(self):
        z_y = tf.fill(dims=[self.latent_sample.shape[0], self.latent_sample.shape[1], self.latent_sample.shape[2], 1],
                      value=tf.squeeze(self.y))
        z_y = tf.concat([self.latent_sample, z_y], axis=-1)

        return z_y

    def _construct(self):
        self.x_y = self._conditional_input()
        self.encoder = self._encoder(self.x_y)

        self.loc, self.scale = self._latent(self.encoder)

        prior_distribution = self._prior_distribution()
        self.latent = prior_distribution(loc=self.loc, scale=tf.exp(0.5 * self.scale))
        # p(z|x,y) assumed prior distribution for the latent space: isotropic normal Gaussian N(x; 0, 1)
        self.prior = prior_distribution(loc=tf.zeros_like(self.loc), scale=tf.ones_like(self.scale))

        self.latent_sample = self.latent.sample()
        self.z_y = self._conditional_latent()

        # p(x|z,y)
        self.decoder = self._decoder(self.z_y)

        with tf.name_scope('reconstructions'):
            tf.summary.image('encoded', self.decoder, max_outputs=1)

        self.loss = self._loss()

