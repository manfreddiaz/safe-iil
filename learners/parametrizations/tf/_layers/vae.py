import numpy as np
import tensorflow as tf
from .residual import resnet_2, resnet_1

LATENT_SPACE_SIZE = 128
L2_LAMBDA = 1e-4
LAST_LAYER_DEPTH = 32
LAST_LAYER_SIZE = 16
FILTER_SIZE = 64


def _baseline_encoder(x, filter_size=FILTER_SIZE):
    model = tf.layers.conv2d(x, filters=filter_size, kernel_size=4, strides=2, padding='same',
                             kernel_initializer=tf.keras.initializers.glorot_uniform())
    model = tf.layers.batch_normalization(model, axis=3)
    model = tf.nn.relu(model)

    model = tf.layers.conv2d(model, filters=filter_size, kernel_size=4, strides=2, padding='same',
                             kernel_initializer=tf.keras.initializers.glorot_uniform())
    model = tf.layers.batch_normalization(model, axis=3)
    model = tf.nn.relu(model)

    model = tf.layers.conv2d(model, filters=filter_size, kernel_size=5, strides=1, padding='valid',
                             kernel_initializer=tf.keras.initializers.glorot_uniform())
    model = tf.layers.batch_normalization(model, axis=3)
    model = tf.nn.relu(model)

    return model


def _baseline_latent_space(x, latent_space_size):
    model = tf.layers.conv2d(x, filters=latent_space_size * 2, kernel_size=3, strides=1, padding='valid')
    model = tf.layers.batch_normalization(model, axis=3)

    mean, variance = tf.split(model, 2, axis=3)

    return mean, variance


def _baseline_decoder(x, filter_size):
    model = tf.layers.conv2d_transpose(x, filters=filter_size, kernel_size=3, strides=1, padding='valid',
                                       kernel_initializer=tf.keras.initializers.glorot_uniform())
    model = tf.layers.batch_normalization(model, axis=3)
    model = tf.nn.relu(model)

    model = tf.layers.conv2d_transpose(model, filters=filter_size, kernel_size=5, strides=1, padding='valid',
                                       kernel_initializer=tf.keras.initializers.glorot_uniform())
    model = tf.layers.batch_normalization(model, axis=3)
    model = tf.nn.relu(model)

    model = tf.layers.conv2d_transpose(model, filters=filter_size, kernel_size=4, strides=2, padding='same',
                                       kernel_initializer=tf.keras.initializers.glorot_uniform())
    model = tf.layers.batch_normalization(model, axis=3)
    model = tf.nn.relu(model)

    model = tf.layers.conv2d_transpose(model, filters=3, kernel_size=4, strides=2, padding='same',
                                       kernel_initializer=tf.keras.initializers.glorot_uniform())
    model = tf.nn.tanh(model)

    return model


def _convolutional_encoder(x):
    return resnet_1(x, keep_prob=1.0)


def _convolutional_decoder(z):
    up_sampling = tf.reshape(z, [-1, LAST_LAYER_SIZE, LAST_LAYER_SIZE, LAST_LAYER_DEPTH])
    model = tf.layers.conv2d(up_sampling, filters=32, kernel_size=3, padding='same',
                             kernel_initializer=tf.keras.initializers.he_normal(),
                             kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))  # 16x16x32

    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)

    up_sampling = tf.image.resize_nearest_neighbor(model, size=(32, 32))
    model = tf.layers.conv2d(up_sampling, filters=32, kernel_size=3, padding='same',
                             kernel_initializer=tf.keras.initializers.he_normal(),
                             kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)

    up_sampling = tf.image.resize_nearest_neighbor(model, size=(64, 64))  # 64x64x32
    model = tf.layers.conv2d(up_sampling, filters=32, kernel_size=3, padding='same',
                             kernel_initializer=tf.keras.initializers.he_normal(),
                             kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)

    up_sampling = tf.image.resize_nearest_neighbor(model, size=(128, 128))  # 128x128x32
    model = tf.layers.conv2d(up_sampling, filters=32, kernel_size=3, padding='same',
                             kernel_initializer=tf.keras.initializers.he_normal(),
                             kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)

    model = tf.layers.conv2d(model, filters=3, kernel_size=3, padding='same', activation=tf.nn.tanh)  # 128x128x3

    return model


def _multilayer_perceptron_encoder(x):
    raise NotImplementedError


def _multilayer_perceptron_decoder(z):
    raise NotImplementedError


def _encoder(x, architecture='conv', filter_size=FILTER_SIZE):
    if architecture == 'mlp':
        encoder = _multilayer_perceptron_encoder(x)
    elif architecture == 'conv':
        encoder = _baseline_encoder(x, filter_size)
    else:
        raise NotImplemented
    return encoder


def _decoder(z, architecture='conv', filter_size=FILTER_SIZE):
    if architecture == 'mlp':
        decoder = _multilayer_perceptron_decoder(z)
    elif architecture == 'conv':
        decoder = _baseline_decoder(z, filter_size=filter_size)
    else:
        raise NotImplemented
    return decoder


def auto_encoder(x, architecture='conv', reconstruction_loss='mse'):
    x = tf.divide(x, 255.)

    encoder = _encoder(x, architecture)
    decoder = _decoder(encoder, architecture)

    with tf.name_scope('reconstructions'):
        tf.summary.image('encoder', decoder, max_outputs=1)

    if reconstruction_loss == 'mse':
        loss = tf.losses.mean_squared_error(x, decoder)
    elif reconstruction_loss == 'cross_entropy':
        loss = tf.losses.sigmoid_cross_entropy(x, decoder)
    else:
        raise NotImplementedError()

    with tf.name_scope('losses'):
        tf.summary.scalar('total_loss', loss)

    return [encoder, decoder], loss


# latent space representation for the location-scale family of distributions
def _latent_space_location_scale(f, latent_space_size):
    model = tf.layers.dense(f, units=2 * latent_space_size)  # latent space representation

    mean, sigma = tf.split(model, 2, axis=1)
    sigma = tf.nn.softplus(sigma)  # maintain sigma_squared > 0

    return mean, sigma


# def variational_auto_encoder_2(x, architecture='conv', latent_size=LATENT_SPACE_SIZE, filter_size=FILTER_SIZE,
#                                reconstruction_loss='cross_entropy', beta=1.0):
#     # x = tf.placeholder(tf.float32, shape=[None, 128, 128, 3], name='input')
#
#     # auto-encoder -> g(f(x)) = x
#
#     x = tf.divide(x, 255.)
#     # q(z|x)
#     encoder = _encoder(x, architecture=architecture, filter_size=filter_size)
#     # suggested trick to improve reconstruction
#     encoder = tf._layers.dense(encoder, units=1024, activation=tf.nn.relu)
#     mean, sigma = _latent_space_location_scale(encoder, latent_size)
#     latent_space = tf.distributions.Normal(loc=mean, scale=sigma)
#     # p(z|x) assumed prior distribution for the latent space: isotropic normal Gaussian N(x; 0, 1)
#     prior = tf.distributions.Normal(loc=np.zeros(latent_size, dtype=np.float32),
#                                     scale=np.ones(latent_size, dtype=np.float32))
#
#     # KL(q(z|x) || p(z|x))
#     kl_divergence = tf.distributions.kl_divergence(latent_space, prior)
#     kl_divergence = tf.reduce_sum(kl_divergence, axis=-1)
#
#     # p(x|z) z ~ re-parametrization of the Gaussian prior
#     decoder = tf._layers.dense(latent_space.sample(), units=LAST_LAYER_SIZE * LAST_LAYER_SIZE * LAST_LAYER_DEPTH)
#     decoder = _decoder(decoder, architecture=architecture)
#
#     with tf.name_scope('reconstructions'):
#         tf.summary.image('encoder', decoder, max_outputs=1)
#
#     # E[log p(x|z)]
#     if reconstruction_loss == 'cross_entropy':
#         d_p_x_z = tf.distributions.Bernoulli(decoder)
#         log_likelihood = d_p_x_z.log_prob(x)
#         log_likelihood = tf.reduce_mean(log_likelihood, axis=(3, 2, 1))
#     elif reconstruction_loss == 'mse':
#         log_likelihood = tf.losses.mean_squared_error(x, decoder)
#     else:
#         raise NotImplementedError()
#
#     # ELBO = E[log p(x|z)] - KL(q(z) || p(z))
#
#     if reconstruction_loss == 'cross_entropy':  # max ELBO ~ min -ELBO
#         evidence_lower_bound = -tf.reduce_sum(log_likelihood - tf.multiply(beta, kl_divergence), axis=0)
#     elif reconstruction_loss == 'mse':  # MSE is convex so min MSE
#         evidence_lower_bound = tf.reduce_sum(log_likelihood + tf.multiply(beta, kl_divergence), axis=0)
#     else:
#         raise NotImplementedError()
#
#     with tf.name_scope('losses'):
#         tf.summary.scalar('total_loss', evidence_lower_bound)
#
#     with tf.name_scope('auxiliary_losses'):
#         tf.summary.scalar('log_likelihood', tf.squeeze(log_likelihood))
#         tf.summary.scalar('kl_divergence', tf.squeeze(kl_divergence))
#
#     return [encoder, decoder], evidence_lower_bound


def variational_auto_encoder(x, architecture='conv', latent_size=LATENT_SPACE_SIZE, filter_size=FILTER_SIZE,
                             reconstruction_loss='cross_entropy', beta=1.0):
    # x = tf.placeholder(tf.float32, shape=[None, 128, 128, 3], name='input')

    # auto-encoder -> g(f(x)) = x

    x = tf.divide(x, 255.)
    # q(z|x)
    encoder = _encoder(x, architecture=architecture)
    # suggested trick to improve reconstruction
    # encoder = tf._layers.dense(encoder, units=1024, activation=tf.nn.relu)
    mean, sigma = _baseline_latent_space(encoder, latent_size)
    latent_space = tf.distributions.Normal(loc=mean, scale=tf.exp(0.5 * sigma))
    # p(z|x) assumed prior distribution for the latent space: isotropic normal Gaussian N(x; 0, 1)
    prior = tf.distributions.Normal(loc=tf.zeros_like(mean, dtype=np.float32),
                                    scale=tf.ones_like(sigma, dtype=np.float32))

    # KL(q(z|x) || p(z|x))
    kl_divergence = tf.distributions.kl_divergence(latent_space, prior)
    kl_divergence = tf.reduce_mean(kl_divergence, axis=(3, 2, 1))

    # p(x|z) z ~ re-parametrization of the Gaussian prior
    # decoder = tf._layers.dense(latent_space.sample(), units=LAST_LAYER_SIZE * LAST_LAYER_SIZE * LAST_LAYER_DEPTH)
    decoder = _decoder(latent_space.sample(), architecture=architecture, filter_size=filter_size)

    with tf.name_scope('reconstructions'):
        tf.summary.image('encoder', decoder, max_outputs=1)

    # E[log p(x|z)]
    if reconstruction_loss == 'cross_entropy':
        d_p_x_z = tf.distributions.Bernoulli(decoder)
        log_likelihood = d_p_x_z.log_prob(x)
        log_likelihood = tf.reduce_mean(log_likelihood, axis=(3, 2, 1))
    elif reconstruction_loss == 'mse':
        log_likelihood = tf.losses.mean_squared_error(x, decoder)
    else:
        raise NotImplementedError()

    # ELBO = E[log p(x|z)] - KL(q(z) || p(z))

    if reconstruction_loss == 'cross_entropy':  # max ELBO ~ min -ELBO
        evidence_lower_bound = tf.reduce_sum(log_likelihood - tf.multiply(beta, kl_divergence), axis=0)
        evidence_lower_bound = -e
    elif reconstruction_loss == 'mse':  # MSE is convex so min MSE
        evidence_lower_bound = tf.reduce_sum(log_likelihood + tf.multiply(beta, kl_divergence), axis=0)
    else:
        raise NotImplementedError()

    with tf.name_scope('losses'):
        tf.summary.scalar('total_loss', evidence_lower_bound)

    with tf.name_scope('auxiliary_losses'):
        tf.summary.scalar('log_likelihood', tf.squeeze(log_likelihood))
        tf.summary.scalar('kl_divergence', tf.squeeze(kl_divergence))

    return [encoder, decoder], evidence_lower_bound
