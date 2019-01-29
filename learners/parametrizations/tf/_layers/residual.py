import tensorflow as tf

l2_lambda = 1e-04


def residual_block(x, size, dropout=False, dropout_prob=0.5, seed=None):
    residual = tf.layers.batch_normalization(x)  # TODO: check if the defaults in Tf are the same as in Keras
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, strides=2, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    if dropout:
        residual = tf.nn.dropout(residual, dropout_prob, seed=seed)
    residual = tf.layers.batch_normalization(residual)
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    if dropout:
        residual = tf.nn.dropout(residual, dropout_prob, seed=seed)

    return residual


def resnet_0(x, keep_prob=0.5):
    nn = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='same',
                             kernel_initializer=tf.keras.initializers.he_normal(),
                             kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    nn = tf.layers.max_pooling2d(nn, pool_size=3, strides=2)

    rb_1 = residual_block(nn, 32)

    nn = tf.layers.conv2d(rb_1, filters=32, kernel_size=1, strides=2, padding='same',
                             kernel_initializer=tf.keras.initializers.he_normal(),
                             kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    # nn = tf.keras.layers.add([rb_1, nn])

    # TODO: check https://github.com/raghakot/keras-resnet for the absence of RELU after merging

    nn = tf.layers.flatten(nn)
    nn = tf.nn.relu(nn)
    nn = tf.layers.dropout(nn, rate=keep_prob)

    return nn


def resnet_1(x, keep_prob=0.5, seed=None):
    nn = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='same',
                             kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                             kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))

    nn = tf.layers.max_pooling2d(nn, pool_size=3, strides=2)

    rb_1 = residual_block(nn, 32, dropout_prob=keep_prob, seed=seed)

    nn = tf.layers.conv2d(nn, filters=32, kernel_size=1, strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    nn = tf.keras.layers.add([rb_1, nn])

    # TODO: check https://github.com/raghakot/keras-resnet for the absence of RELU after merging

    # nn = tf.layers.batch_normalization(nn)
    # nn = tf.nn.relu(nn)
    # nn = tf.keras.layers.GlobalAveragePooling2D()(nn)
    nn = tf.layers.flatten(nn)

    # nn = tf.nn.relu(nn)
    # nn = tf.layers.dropout(nn, rate=keep_prob)

    return nn


def resnet_2(x, keep_prob=0.5):
    residual_net = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='same')
    residual_net = tf.layers.max_pooling2d(residual_net, pool_size=2, strides=2)

    residual_block_1 = residual_block(residual_net, 32)
    residual_net = tf.layers.conv2d(residual_net, filters=32, kernel_size=1, strides=2, padding='same')
    residual_net = tf.keras.layers.add([residual_block_1, residual_net])

    residual_block_2 = residual_block(residual_net, 64)
    residual_net = tf.layers.conv2d(residual_net, filters=64, kernel_size=1, strides=2, padding='same')
    residual_net = tf.keras.layers.add([residual_block_2, residual_net])

    residual_net = tf.layers.flatten(residual_net)

    # residual_net = tf.nn.dropout(residual_net, keep_prob=keep_prob)

    return residual_net


def resnet_1_dropout(x, keep_prob=0.5):
    nn = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='same')
    nn = tf.nn.dropout(nn, keep_prob)
    nn = tf.layers.max_pooling2d(nn, pool_size=3, strides=2)

    rb_1 = residual_block(nn, 32, dropout=True)

    nn = tf.layers.conv2d(nn, filters=32, kernel_size=1, strides=2, padding='same')
    nn = tf.nn.dropout(nn, keep_prob)
    nn = tf.keras.layers.add([rb_1, nn])

    # TODO: check https://github.com/raghakot/keras-resnet for the absence of RELU after merging

    nn = tf.layers.flatten(nn)

    nn = tf.nn.relu(nn)
    nn = tf.layers.dropout(nn, rate=0.5)

    return nn