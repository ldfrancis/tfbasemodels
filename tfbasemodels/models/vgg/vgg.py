import tensorflow as tf
from tfbasemodels.models.model import TFBaseModel


class VGG(TFBaseModel):
    """Builds the base vgg network from which vgg16 and vgg19
    can be obtained.

    args: n_layers. int. The number of layers to build. can be 
        either 16 or 19. 16 for vgg16 and 19 for vgg19
        include_top
        pooling
    """
    input_shape = [224, 224, 3]

    def __init__(self, n_layers: int, include_top=True, pooling=None, pretrained=False, name="vgg"):
        assert n_layers in [16, 19]
        self.n_layers = n_layers
        self.n_conv_blocks = 5
        self.include_top = include_top
        self.pooling = pooling
        self.model_name = name
        if include_top:
            self.pretrained_weights_url = ("https://github.com/fchollet/"
                                           "deep-learning-models/releases/download/v0.1/"
                                           f"vgg{n_layers}_weights_tf_dim_ordering_tf_kernels.h5")
            self.pretrained_weights_name = f"vgg{n_layers}_weights.h5"
            self.filehash = "64373286793e3c8b2b4e3219cbf3544b" if n_layers == 16 else "cbe5617147190e668d6c5d5026f83318"
        else:
            self.pretrained_weights_url = ("https://github.com/fchollet/"
                                           "deep-learning-models/releases/download/v0.1/"
                                           f"vgg{n_layers}_weights_tf_dim_ordering_tf_kernels_notop.h5")
            self.pretrained_weights_name = f"vgg{n_layers}_weights_notop.h5"
            self.filehash = "6d6bbae143d832006294945121d1f1fc" if n_layers == 16 else "253f8cb515780f3b799900260a226db6"

        super().__init__(pretrained=pretrained)

    def build(self):
        """builds the entire vgg architecture 
        """
        assert self.n_conv_blocks == 5  # always 5 blocks

        # input layer
        inp_x = tf.keras.layers.Input(shape=VGG.input_shape)

        # vgg convolution blocks 1 - 5
        for block in range(1, self.n_conv_blocks + 1):
            x = self.build_block(block, self.n_layers, inp_x) if block == 1 \
                else self.build_block(block, self.n_layers, x)

        if self.include_top:
            # fully connected layers
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(4096, activation=tf.nn.relu, name="fc_6")(x)
            x = tf.keras.layers.Dense(4096, activation=tf.nn.relu, name="fc_7")(x)
            logits = tf.keras.layers.Dense(1000, name="logits")(x)

            # return model
            return inp_x, logits
        else:
            # return output from convolutional blocks
            if self.pooling is "avg":
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
            elif self.pooling is "max":
                x = tf.keras.layers.GlobalMaxPool2D()(x)

            return inp_x, x

    def build_block(self, idx, n_layers, x):
        """builds a block in the vgg architecture
        """
        # set number of convolution in a layer based on the block id and 
        # the total number of layers for the network (16 or 19)
        n_conv = 2 if idx < 3 else 3 if n_layers == 16 else 4
        filter_mult = min(2 ** (idx - 1), 8)
        filters = 64 * filter_mult

        # convolutions
        for conv in range(1, n_conv + 1):
            x = tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=3,
                                       padding="same",
                                       strides=1,
                                       activation=tf.nn.relu,
                                       name=f"block_{idx}_conv_{conv}")(x)

        # final pooling
        x = tf.keras.layers.MaxPool2D(name=f"block_{idx}_maxpool")(x)

        return x


# VGG16
class VGG16(VGG):
    def __init__(self, **kwargs):
        super().__init__(16, **kwargs)


# VGG19
class VGG19(VGG):
    def __init__(self, **kwargs):
        super().__init__(19, **kwargs)
