import tensorflow as tf
import numpy as np
from .resnet import ResNet
from tfbasemodels.utils.layer_utils import conv2d_bn, bn_conv2d


class ResNext(ResNet):
    """ResNext
    Input: 224x224x3
    """

    def __init__(self):
        super().__init__()
        self.groups = None
        self.set_groups()

    def set_groups(self):
        self.groups = 32

    def residual_block(self, x, shortcut="identity", bottleneck=True, stage=None, block=None, conv_filters=None,
                       strides=1):
        """A Residual block for resnet that accounts for full pre-activation. input x can either be an output of
        a convolutional layer with or without activation.

        Args:
            x: input tensor: Tensor
            shortcut: string. Determines whether to use identity or projection shortcuts. can be either of "identity"
            or "projection"
            bottleneck: boolean. whether to use a bottleneck stage or not
            stage: string or int. Tells what stage the residual network belongs to.
                Used to specify the name of an operation in the residual block
            block: string or int. Tells the name of the residual block in a stage
                Used to specify the name of an operation in the residual block
            conv_filters: list of ints. Used to specify the number of filters to be used in conv layers of a block.
                for bottleneck blocks, conv_filters = [filter1, filter2, filter3]
                for non bottleneck blocks, conv_filters = [filter1, filter3]
            strides: int. Strides to use for fist convolution layers and shortcuts(if projection shortcut is used)

        Returns:
            a Tensor
        """
        assert stage is not None and block is not None
        assert shortcut in ["identity", "projection"]

        base_name = "residual_" + str(stage) + "_" + str(block)

        if bottleneck:
            filters1, filters2, filters3 = conv_filters
        else:
            filters1, filters3 = conv_filters

        conv_function = conv2d_bn

        # shortcut
        if shortcut == "projection":
            shortcut_x = conv_function(filters=filters3,
                                       kernel_size=1,
                                       strides=strides,
                                       padding="same",
                                       name=base_name + "shortcut")(x)
        else:
            shortcut_x = x

        # bottleneck
        # first 1x1. this conv does spatial reduction with the strides and matches the shortcut spatial size
        x = conv_function(filters=filters1,
                          kernel_size=1,
                          strides=1,
                          padding="same",
                          name=base_name + "_1",
                          activation=tf.nn.relu)(x)
        # depthwise conv
        cardinality = filters2 // self.groups
        x = tf.keras.layers.DepthwiseConv2D(3, strides=strides, depth_multiplier=cardinality,
                                   use_bias=False, name=base_name + '_2_conv')(x)
        # pointwise conv
        kernel = np.zeros((1, 1, filters2 * cardinality, filters2), dtype=np.float32)
        for i in range(filters2):
            start = (i // cardinality) * cardinality * cardinality + i % cardinality
            end = start + cardinality * cardinality
            kernel[:, :, start:end:cardinality, i] = 1.
        x = conv_function(filters=filters2,
                          kernel_size=1,
                          strides=1,
                          padding="same",
                          name=base_name + "_2_gconv",
                          use_bias=False,
                          trainable=False,
                          kernel_initializer={'class_name': 'Constant',
                                              'config': {'value': kernel}},
                          epsilon=1.001e-5,
                          activation=tf.nn.relu)(x)
        # last _1x1
        x = conv_function(filters=filters3,
                          kernel_size=1,
                          strides=1,
                          padding="same",
                          use_bias=False,
                          epsilon=1.001e-5,
                          name=base_name + "_3", )(x)

        # add shortcut to residual output
        x = tf.keras.layers.Add(name=base_name + "_add")([shortcut_x, x])

        # activation
        x = tf.nn.relu(x, name=base_name + "_relu")

        return x


class ResNext50(ResNext):
    """Resnext50 32x4d
    """

    def __init__(self):
        """Initialize resnet50
        """
        super().__init__()

    def set_filters_and_shortcuts(self):
        """Overrides the parent method
        """
        # stage 2
        self.stage2_filters = [(128, 128, 256)] * 3
        self.stage2_shortcuts = ["projection", *("identity",) * 2]

        # stage 3
        self.stage3_filters = [(256, 256, 512)] * 4
        self.stage3_shortcuts = ["projection", *("identity",) * 3]

        # stage 4
        self.stage4_filters = [(512, 512, 1024)] * 6
        self.stage4_shortcuts = ["projection", *("identity",) * 5]

        # stage 5
        self.stage5_filters = [(1024, 1024, 2048)] * 3
        self.stage5_shortcuts = ["projection", *("identity",) * 2]

        # to use bottleneck blocks or not
        self.bottleneck = True


class ResNext101(ResNext):
    """Resnext101 32x4d
    """

    def __init__(self):
        """Initialize resnet101
        """
        super().__init__()

    def set_filters_and_shortcuts(self):
        """Overrides the parent method
        """
        # stage 2
        self.stage2_filters = [(128, 128, 256)] * 3
        self.stage2_shortcuts = ["projection", *("identity",) * 2]

        # stage 3
        self.stage3_filters = [(256, 256, 512)] * 4
        self.stage3_shortcuts = ["projection", *("identity",) * 3]

        # stage 4
        self.stage4_filters = [(512, 512, 1024)] * 23
        self.stage4_shortcuts = ["projection", *("identity",) * 22]

        # stage 5
        self.stage5_filters = [(1024, 1024, 2048)] * 3
        self.stage5_shortcuts = ["projection", *("identity",) * 2]

        # to use bottleneck blocks or not
        self.bottleneck = True
