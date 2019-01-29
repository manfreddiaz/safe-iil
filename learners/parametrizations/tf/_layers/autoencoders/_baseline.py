import tensorflow as tf

FILTER_SIZE = 128
LATENT_SIZE = 128


class Autoencoder:
    def __init__(self, x, latent_size=LATENT_SIZE, loss_function='mse'):
        self.x = x

        self.loss_function = loss_function
        self.latent_size = latent_size

        self.encoder = None
        self.decoder = None
        self.latent = None

        self._modify_input()
        self._construct()

    def _modify_input(self):
        raise NotImplementedError()

    def _encoder(self, x):
        raise NotImplementedError()

    def _decoder(self, z):
        raise NotImplementedError()

    def _construct(self):
        self.encoder = self._encoder(self.x)

        self.decoder = self._decoder(self.encoder)

        # with tf.name_scope('reconstructions'):
        #     tf.summary.image('encoded', self.decoder, max_outputs=1)

        self.loss = self._loss()

    def _loss(self):
        if self.loss_function == 'cross_entropy':
            reconstruction_loss = tf.losses.sigmoid_cross_entropy(self.x, self.decoder)
        elif self.loss_function == 'mse':
            reconstruction_loss = tf.reduce_mean(tf.square(self.x - self.decoder), axis=1)
        else:
            raise NotImplementedError()

        with tf.name_scope('losses'):
            tf.summary.scalar('reconstruction', reconstruction_loss)

        return reconstruction_loss
