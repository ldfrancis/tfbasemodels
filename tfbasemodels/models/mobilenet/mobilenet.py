import tensorflow as tf
from ..model import TFBaseModel


class MobileNet(TFBaseModel):
    input_shape = (224, 224, 3)
    alpha = 1
    num_classes = 1000

    def __init__(self, alpha=1, dropout=0.2, include_top=True, pooling='avg', pretrained=False):
        super().__init__()
        self.alpha = alpha
        self.include_top = include_top
        self.pooling = pooling
        self.pretrained = pretrained
        self.dropout = dropout

    def build(self):
        # input
        x_input = tf.keras.layers.Input(shape=self.input_shape)

        # stem
        x = MobileNet.stem_network(x_input, 32, 3, self.alpha, 2)

        # mobilenet depthwise separable blocks
        x = MobileNet.depthwise_n_pointwise_conv_block(x, 64, 3, self.alpha, 1, self.depth_mult, "depth_sep_block1")
        x = MobileNet.depthwise_n_pointwise_conv_block(x, 128, 3, self.apha, 2, self.depth_mult, "depth_sep_block2")
        x = MobileNet.depthwise_n_pointwise_conv_block(x, 128, 3, self.alpha, 1, self.depth_mult, "depth_sep_block3")
        x = MobileNet.depthwise_n_pointwise_conv_block(x, 256, 3, self.apha, 2, self.depth_mult, "depth_sep_block4")
        x = MobileNet.depthwise_n_pointwise_conv_block(x, 256, 3, self.apha, 1, self.depth_mult, "depth_sep_block5")
        x = MobileNet.depthwise_n_pointwise_conv_block(x, 512, 3, self.apha, 2, self.depth_mult, "depth_sep_block6")
        for i in range(7, 7 + 5):
            x = MobileNet.depthwise_n_pointwise_conv_block(x, 512, 3, self.apha, 1, self.depth_mult,
                                                           f"depth_sep_block{i}")
        x = MobileNet.depthwise_n_pointwise_conv_block(x, 1024, 3, self.apha, 2, self.depth_mult, "depth_sep_block12")
        x = MobileNet.depthwise_n_pointwise_conv_block(x, 1024, 3, self.apha, 1, self.depth_mult, "depth_sep_block13")

        if self.include_top:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            shape = (1, 1, int(1024 * self.alpha))
            x = tf.keras.layers.Reshape(shape, name='reshape_1')(x)
            x = tf.keras.layers.Dropout(self.dropout, name='dropout')(x)
            x = tf.keras.layers.Conv2D(self.num_classes, 1, padding='same', name='conv_preds')(x)
            x = tf.keras.layers.Reshape((self.num_classes,), name='reshape_2')(x)
            x = tf.keras.layers.Activation('softmax', name='act_softmax')(x)
        else:
            if self.pooling == 'avg':
                x = tf.keras.layers.GlobalAveragePooling2D(name="avgpool")(x)
            elif self.pooling == 'max':
                x = tf.keras.layers.GlobalMaxPooling2D(name="maxpool")(x)
        model = tf.keras.Model(inputs=[x_input], outputs=[x])
        return model

    @staticmethod
    def stem_network(x, filters, kernel, alpha, strides):
        """Initial Conv batch_norm and activation before depthwise and pointwise convolutions.
        Args:
            x: Tensor. input
            filters: the number of filters to use for the conv layer
            kernel: int. the kernel size to use
            alphs: float. filter multiplier. Determines how much to scale the number of filters used.
            strides: int. convolution strides to use
        Returns:
            Tensor. output from the set of operations, conv-batch_norm-activation
        """
        base_name = "stem/"
        filters = int(filters * alpha)
        x = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name=base_name + 'conv1_pad')(x)
        x = tf.keras.layers.Conv2D(filters, kernel,
                                   padding='valid',
                                   use_bias=False,
                                   strides=strides,
                                   name=base_name + 'conv1')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=base_name + 'conv1_bn')(x)
        x = tf.keras.layers.Activation(6., name=base_name + 'conv1_relu')(x)
        return x

    @staticmethod
    def depthwise_n_pointwise_conv_block(x, filters, kernel, alpha, strides, depth_mult, name):
        """Performs depthwise convolution and applies batch norm and an activation. This operation is then followed by
        a pointwise convolution, batch norm and an activation
        Args:
            x: Tensor. input
            filters: int. number of filters to use for pointwise convolution
            kernel: int. kernel size for depthwise convolution
            alpha: float. used to scale the value for number of filters used
            strides: int. strides to use for the depthwise convolution
        Returns:
            Tensor. output from the set of operations involving depthwise and pointwise convolutions
        """
        base_name = name
        x = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)), name=name + "pad")(x)
        x = tf.keras.layers.DepthwiseConv2D(kernel,
                                            padding='same' if strides == (1, 1) else 'valid',
                                            depth_multiplier=depth_mult,
                                            strides=strides,
                                            use_bias=False,
                                            name=name + "dw_conv")(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=name + "dw_conv_bn")(x)
        x = tf.keras.layers.ReLU(6., name=name + "dw_conv_relu")(x)

        x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
                                   name=base_name + "pw_conv")(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=base_name + "pw_conv_bn")(x)
        x = tf.keras.layers.ReLU(6., name=base_name + "pw_conv_relu")(x)
        return x
