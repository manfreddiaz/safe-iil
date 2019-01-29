import tensorflow as tf
import tensorflow.contrib.distributions as tfcd
from .convolutional import ConvolutionalAutoencoder


class VariationalAutoencoder(ConvolutionalAutoencoder):

    def __init__(self, x, filter_size, latent_size, loss_function):
        ConvolutionalAutoencoder.__init__(self, x, filter_size, latent_size, loss_function)

    # Gaussian Prior
    def _prior_distribution(self):
        return tf.distributions.Normal

    def _latent(self, q_z_x):
        nn = tf.layers.conv2d(q_z_x, filters=self.latent_size * 2, kernel_size=3, strides=1, padding='valid',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        nn = tf.layers.batch_normalization(nn, axis=3)

        mean, variance = tf.split(nn, 2, axis=3)

        return mean, variance

    def _construct(self):
        self.encoder = self._encoder(self.x)

        self.loc, self.scale = self._latent(self.encoder)

        f_loc = tf.layers.flatten(self.loc)
        f_scale = tf.layers.flatten(self.scale)

        prior_distribution = self._prior_distribution()

        self.latent = prior_distribution(loc=f_loc, scale=f_scale, name='latent')
        # p(z|x) assumed prior distribution for the latent space: isotropic normal Gaussian N(x; 0, 1)
        self.prior = prior_distribution(loc=tf.zeros_like(f_loc), scale=tf.ones_like(f_scale), name='prior')

        # p(x|z)

        latent_sample = self.latent.sample()

        with tf.name_scope('latent'):
            tf.summary.histogram('sample', latent_sample)

        self.decoder = self._decoder(tf.reshape(latent_sample, shape=self.loc.shape))

        with tf.name_scope('reconstructions'):
            tf.summary.image('encoded', self.decoder, max_outputs=1)

        self.loss = self._loss()

    def _reconstruction_loss(self):
        # E[log p(x|z)]
        if self.loss_function == 'cross_entropy':
            d_p_x_z = tf.distributions.Bernoulli(self.decoder)
            d_p_x_z = tfcd.Independent(d_p_x_z, 3)
            log_likelihood = d_p_x_z.log_prob(self.x)
            log_likelihood = tf.reduce_mean(log_likelihood, name='log_likelihood')
            # log_likelihood = tf.div(log_likelihood, tf.reduce_prod(self.x.shape))
        elif self.loss_function == 'mse':
            log_likelihood = tf.reduce_sum(tf.squared_difference(self.x, self.decoder), axis=(1, 2, 3))
        else:
            raise NotImplementedError()

        return log_likelihood

    def _posterior_prior_divergence(self):
        # KL(q(z|x) || p(z|x))
        kl_divergence = self.latent.kl_divergence(self.prior)
        kl_divergence = tf.reduce_sum(kl_divergence, axis=1)

        return kl_divergence

    def _loss(self):
        # KL(q(z|x) || p(z|x))
        self.posterior_prior_divergence = self._posterior_prior_divergence()

        # E[log p(x|z)]
        self.reconstruction_loss = self._reconstruction_loss()

        # ELBO = E[log p(x|z)] - KL(q(z|x) || p(z|x))
        if self.loss_function == 'cross_entropy':
            ELBO = tf.reduce_mean(-self.reconstruction_loss + self.posterior_prior_divergence, axis=0)
        elif self.loss_function == 'mse':
            ELBO = tf.reduce_mean(self.reconstruction_loss + self.posterior_prior_divergence, axis=0)
        else:
            raise NotImplementedError()

        with tf.name_scope('auxiliary_losses'):
            tf.summary.scalar('log_likelihood', tf.squeeze(self.reconstruction_loss))
            tf.summary.scalar('kl_divergence', tf.squeeze(self.posterior_prior_divergence))

        with tf.name_scope('losses'):
            tf.summary.scalar('total_loss', ELBO)

        return ELBO
