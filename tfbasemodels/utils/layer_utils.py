import tensorflow as tf


def conv2d_bn(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
              dilation_rate=(1, 1), groups=1, activation=None, use_bias=True,
              kernel_initializer='glorot_uniform', bias_initializer='zeros',
              kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
              kernel_constraint=None, bias_constraint=None, axis=-1, momentum=0.99, epsilon=0.001, center=True,
              scale=True,
              beta_initializer='zeros', gamma_initializer='ones',
              moving_mean_initializer='zeros', moving_variance_initializer='ones',
              beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
              gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
              fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None,
              **kwargs):
    """Performs convolution-->batch_norm-->activation
    """

    def _op(x):
        if name:
            conv_name = name + "_conv"
            bn_name = name + "_bn"
            act_name = name + "_act"
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, data_format=data_format,
                                   dilation_rate=dilation_rate, groups=groups, use_bias=use_bias,
                                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                   kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                   activity_regularizer=activity_regularizer,
                                   kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                   name=conv_name)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, momentum=momentum, epsilon=epsilon, center=center,
                                               scale=scale,
                                               beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                                               moving_mean_initializer=moving_mean_initializer,
                                               moving_variance_initializer=moving_variance_initializer,
                                               beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                                               beta_constraint=beta_constraint,
                                               gamma_constraint=gamma_constraint, renorm=renorm,
                                               renorm_clipping=renorm_clipping, renorm_momentum=renorm_momentum,
                                               fused=fused, trainable=trainable, virtual_batch_size=virtual_batch_size,
                                               adjustment=adjustment, name=bn_name,
                                               )(x)

        if activation: x = activation(x, name=act_name)

        return x

    return _op


def bn_conv2d(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
              dilation_rate=(1, 1), groups=1, activation=None, use_bias=True,
              kernel_initializer='glorot_uniform', bias_initializer='zeros',
              kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
              kernel_constraint=None, bias_constraint=None, axis=-1, momentum=0.99, epsilon=0.001, center=True,
              scale=True,
              beta_initializer='zeros', gamma_initializer='ones',
              moving_mean_initializer='zeros', moving_variance_initializer='ones',
              beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
              gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
              fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None,
              **kwargs):
    """Performs batch_norm-->activation-->convolution. used in a residual block with
    preactivation
    """

    def _op(x):
        if name:
            conv_name = name + "_conv"
            bn_name = name + "_bn"
            act_name = name + "_act"

        x = tf.keras.layers.BatchNormalization(axis=3, momentum=momentum, epsilon=epsilon, center=center,
                                               scale=scale, beta_initializer=beta_initializer,
                                               gamma_initializer=gamma_initializer,
                                               moving_mean_initializer=moving_mean_initializer,
                                               moving_variance_initializer=moving_variance_initializer,
                                               beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                                               beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
                                               renorm=renorm, renorm_clipping=renorm_clipping,
                                               renorm_momentum=renorm_momentum, fused=fused, trainable=trainable,
                                               virtual_batch_size=virtual_batch_size, adjustment=adjustment,
                                               name=conv_name, )(x)

        if activation: x = activation(x, name=act_name)

        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, data_format=data_format,
                                   dilation_rate=dilation_rate, groups=groups, use_bias=use_bias,
                                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                   kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                   activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                                   bias_constraint=bias_constraint, name=bn_name)(x)
        return x

    return _op
