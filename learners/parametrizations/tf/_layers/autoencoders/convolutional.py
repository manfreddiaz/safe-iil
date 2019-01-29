import tensorflow as tf

FILTER_SIZE = 128
LATENT_SIZE = 128


class ConvolutionalAutoencoder:
    def __init__(self, x, filter_size=FILTER_SIZE, latent_size=LATENT_SIZE, loss_function=tf.losses.mean_squared_error):
        self.x = x

        self.filter_size = filter_size
        self.loss_function = loss_function
        self.latent_size = latent_size

        self.encoder = None
        self.decoder = None
        self.latent = None

        self._normalize()
        self._construct()

    def _normalize(self):
        self.x = tf.div(self.x, 255., name='normalize')

    def _encoder(self, x):
        with tf.name_scope('encoder'):
            nn = tf.layers.conv2d(x, filters=self.filter_size, kernel_size=4, strides=2, padding='same',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_1')
            nn = tf.layers.batch_normalization(nn)
            nn = tf.nn.relu(nn)
            # nn = tf._layers.dropout(nn, rate=0.2)

            nn = tf.layers.conv2d(nn, filters=self.filter_size, kernel_size=4, strides=2, padding='same',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_2')
            nn = tf.layers.batch_normalization(nn)
            nn = tf.nn.relu(nn)
            # nn = tf._layers.dropout(nn, rate=0.2)

            nn = tf.layers.conv2d(nn, filters=self.filter_size, kernel_size=5, strides=1, padding='valid',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_3')
            nn = tf.layers.batch_normalization(nn)
            nn = tf.nn.relu(nn)
            # nn = tf._layers.dropout(nn, rate=0.2)

        return nn

    def _decoder(self, z):
        with tf.name_scope('decoder'):
            nn = tf.layers.conv2d_transpose(z, filters=self.filter_size, kernel_size=3, strides=1, padding='valid',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_t_1')
            nn = tf.layers.batch_normalization(nn)
            nn = tf.nn.relu(nn)

            nn = tf.layers.conv2d_transpose(nn, filters=self.filter_size, kernel_size=5, strides=1, padding='valid',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_t_2')
            nn = tf.layers.batch_normalization(nn)
            nn = tf.nn.relu(nn)

            nn = tf.layers.conv2d_transpose(nn, filters=self.filter_size, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_t_3')
            nn = tf.layers.batch_normalization(nn)
            nn = tf.nn.relu(nn)

            nn = tf.layers.conv2d_transpose(nn, filters=3, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_t_4')
            nn = tf.nn.sigmoid(nn)

        return nn

    def _construct(self):
        self.encoder = self._encoder(self.x)

        self.decoder = self._decoder(self.encoder)

        with tf.name_scope('reconstructions'):
            tf.summary.image('encoded', self.decoder, max_outputs=1)

        self.loss = self._loss()

    def _loss(self):
        if self.loss_function == 'cross_entropy':
            reconstruction_loss = tf.losses.sigmoid_cross_entropy(self.x, self.decoder)
        elif self.loss_function == 'mse':
            reconstruction_loss = tf.losses.mean_squared_error(self.x, self.decoder)
        else:
            raise NotImplementedError()

        with tf.name_scope('losses'):
            tf.summary.scalar('total_loss', reconstruction_loss)

        return reconstruction_loss
