from abc import abstractmethod

import tensorflow as tf
from ..model import TFBaseModel
from ..utils.layer_utils import conv2d_bn, bn_conv2d


class ResNet(TFBaseModel):
    """Resnet
    Input: 224x224x3
    """

    def __init__(self):
        """Initialize a Resnet.
        """
        self.preact = False
        self.bottleneck = False
        self.set_filters_and_shortcuts()
        self.block2_filters = None
        self.block2_shortcuts = None
        self.block3_filters = None
        self.block3_shortcuts = None
        self.block4_filters = None
        self.block4_shortcuts = None
        self.block5_filters = None
        self.block5_shortcuts = None
        super().__init__()

    @abstractmethod
    def set_filters_and_shortcuts(self):
        """Must be implemented by subclass. Sets the number of filters and type of shortcut for each 
        block based on the number of layers
        specified.
        This method must set the following attributes
        self.block2_filters : List[Tuple]
        self.block2_shortcuts: List[string]
        self.block3_filters
        self.block3_shortcuts
        self.block4_filters
        self.block4_shortcuts
        self.block5_filters
        self.block5_shortcuts
        self.bottleneck
        self.preact

        """
        raise NotImplementedError()

    def build(self):
        """Builds a resnet architecture with the required no of layers
        based on self.n_layers
        """
        assert not (self.n_layers is None)
        x_inp = tf.keras.layers.Input(shape=(224, 224, 3))

        # input size: 224x224x3
        # output size: 112x112x64
        # stem; block 1
        x = self.stem(x_inp)

        # pool
        # input size: 112x112x64
        # ouput size: 56x56x64
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                      strides=2,
                                      padding="same")(x)

        # input size: 56x56x64
        # output size: 56x56x_
        # block 2
        x = self.build_block(x,
                             block=2,
                             blocks_filters=self.block2_filters,
                             shortcuts=self.block2_shortcuts,
                             strides=1)

        # block 3
        # input size: 56x56x_
        # output size: 28x28x_
        x = self.build_block(x,
                             block=2,
                             blocks_filters=self.block3_filters,
                             shortcuts=self.block3_shortcuts,
                             strides=2)

        # block 4
        # input size: 28x28x_
        # output size: 14x14x_
        x = self.build_block(x,
                             block=3,
                             blocks_filters=self.block4_filters,
                             shortcuts=self.block4_shortcuts,
                             strides=2)

        # block 5
        # input size: 14x14x_
        # output size: 7x7x_
        x = self.build_block(x,
                             block=5,
                             blocks_filters=self.block5_filters,
                             shortcuts=self.block5_shortcuts,
                             strides=2)

        # pool
        # input: 7x7x_
        # output: 1x1x_
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        logits = tf.keras.layers.Dense(1000, name='logits')(x)

        return tf.keras.Model(inputs=[x_inp], outputs=[logits])

    def build_block(self, x, block=None, blocks_filters=[], shortcuts=[], strides=2):
        """A block of a deep residual network

        Args: 
            x: Tensor. the input tensor
            block: int or string. The block of the resnet currently being built
            blocks_filters: list of tuples. a list containing the number of filters to use
                for each block
            shortcuts: list of strings. a list specifying the type of shortcut to use for 
                each block
            strides: int. strides to use for the first convolution and the shortcut

        Returns:
            a Tensor 
        """
        for i, block_filters, shortcut_ in enumerate(zip(blocks_filters, shortcuts)):
            reduction_network = len(blocks_filters) - 1 if self.preact else 0
            strides_ = strides if i == reduction_network else 1
            x = self.residual_network(x,
                                      shortcut=shortcut_,
                                      block=block,
                                      network=str(i + 1),
                                      conv_filters=block_filters,
                                      strides=strides_,
                                      bottleneck=self.bottleneck)

        return x

    def stem(self, x):
        """Stem layers for resnet consisting only of a convolutional layer that redueces
        the spatial dimensions from 224x224 to 112x112

        Atgs:
            x: Tensor. the input tensor

        Returns:
            a Tensor
        """
        x = conv2d_bn(64,
                      kernel_size=7,
                      strides=2,
                      padding="same",
                      activation=tf.nn.relu,
                      name="stem_1")(x)

        return x

    def residual_network(self, x, shortcut="identity", bottleneck=True, block=None, network=None, conv_filters=[],
                         strides=1):
        """A Residual block for resnet that accounts for full preactivation. input x can either be an output of a convolutional layer
        without activation or not. 

        Args:
            x: input tensor: Tensor
            shortcut: string. Detertmines whether to use identity or projection shortcuts. can be either of "identity" or
                "projection"
            bottleneck: boolean. whether to use a bottleneck block or not
            block: string or int. Tells what block of the whole network that the block belongs to. 
                Used to specify the name of an operation in the residual block
            block: string or int. Tells the name of a block in a block
                Used to specify the name of an operation in the residual block
            conv_filters: list of ints. Used to specify the number of filters to be used in conv layers of a block.
                for bottleneck blocks, conv_filters = [filter1, filter2, filter3]
                for non bottleneck blocks, conv_filters = [filter1, filter3]
            strides: int. Strides to use for fist convolution layers and shortcuts(if projection shortcut is used)

        Returns:
            a Tensor
        """
        assert block is not None and network is not None
        assert shortcut in ["identity", "projection"]

        base_name = "residual_" + str(block) + "_" + str(network)

        if bottleneck:
            filters1, filters2, filters3 = conv_filters
        else:
            filters1, filters3 = conv_filters

        conv_function = bn_conv2d if self.preact else conv2d_bn

        # shortcut
        if shortcut == "projection":
            shortcut_x = conv_function(filters=filters3,
                                       kernel_size=1,
                                       strides=strides,
                                       padding="same",
                                       name=base_name + "shortcut")(x)
        else:
            shortcut_x = tf.keras.layers.MaxPool2D(pool_size=1, strides=strides, padding="same")(
                x) if strides > 1 else x

        # bottleneck
        if bottleneck:
            # first 1x1. this conv does spatial reduction with the strides and matches the shortcut spatial size
            x = conv_function(filters=filters1,
                              kernel_size=1,
                              strides=(1 if self.preact else strides),
                              padding="same",
                              name=base_name + "_1")(x)
            # _3x3
            x = conv_function(filters=filters2,
                              kernel_size=3,
                              strides=(strides if self.preact else 1),
                              padding="same",
                              name=base_name + "_2")(x)
            # last _1x1
            x = conv_function(filters=filters3,
                              kernel_size=1,
                              strides=1,
                              padding="same",
                              name=base_name + "_3")(x)
        else:
            # first 3x3: this conv does spatial reduction with the strides and matches the shortcut spatial size
            x = conv_function(filters=filters1,
                              kernel_size=3,
                              strides=strides,
                              padding="same",
                              name=base_name + "_1")(x)
            # last 3x3
            x = conv_function(filters=filters3,
                              kernel_size=3,
                              strides=1,
                              padding="same",
                              name=base_name + "_2")(x)

        # add shortcut to residual output
        x = tf.keras.layers.Add(name=base_name + "_add")([shortcut, x])

        # activation (accounts for preactivation)
        x = x if self.preact else tf.nn.relu(x, name=base_name + "_relu")

        return x


class ResNet18(ResNet):
    """Resnet18
    """

    def __init__(self):
        """Initialize resnet18
        """
        super().__init__()

    def set_filters_and_shortcuts(self):
        """Overides the parent method
        """
        # block 2
        self.block2_filters = [(64, 64)] * 2
        self.block2_shortcuts = ["projection", "identity"]

        # block 3
        self.block3_filters = [(128, 128)] * 2
        self.block3_shortcuts = ["projection", "identity"]

        # block 4
        self.block4_filters = [(256, 256)] * 2
        self.block4_shortcuts = ["projection", "identity"]

        # block 5
        self.block5_filters = [(512, 512)] * 2
        self.block5_shortcuts = ["projection", "identity"]

        # to use bottleneck blocks or not
        self.bottleneck = False


class ResNet32(ResNet):
    """Resnet32
    """

    def __init__(self):
        """Initialize resnet32
        """
        super().__init__()

    def set_filters_and_shortcuts(self):
        """Overides the parent method
        """
        # block 2
        self.block2_filters = [(64, 64)] * 3
        self.block2_shortcuts = ["projection", *("identity",) * 2]

        # block 3
        self.block3_filters = [(128, 128)] * 4
        self.block3_shortcuts = ["projection", *("identity",) * 3]

        # block 4
        self.block4_filters = [(256, 256)] * 6
        self.block4_shortcuts = ["projection", *("identity",) * 5]

        # block 5
        self.block5_filters = [(512, 512)] * 3
        self.block5_shortcuts = ["projection", *("identity",) * 2]

        # to use bottleneck blocks or not
        self.bottleneck = False


class ResNet50(ResNet):
    """Resnet50
    """

    def __init__(self):
        """Initialize resnet50
        """
        super().__init__()

    def set_filters_and_shortcuts(self):
        """Overides the parent method
        """
        # block 2
        self.block2_filters = [(64, 64, 256)] * 3
        self.block2_shortcuts = ["projection", *("identity") * 2]

        # block 3
        self.block3_filters = [(128, 128, 512)] * 4
        self.block3_shortcuts = ["projection", *("identity") * 3]

        # block 4
        self.block4_filters = [(256, 256, 1024)] * 6
        self.block4_shortcuts = ["projection", *("identity") * 5]

        # block 5
        self.block5_filters = [(512, 512, 2048)] * 3
        self.block5_shortcuts = ["projection", *("identity") * 2]

        # to use bottleneck blocks or not
        self.bottleneck = True


class ResNet101(ResNet):
    """Resnet101
    """

    def __init__(self):
        """Initialize resnet101
        """
        super().__init__()

    def set_filters_and_shortcuts(self):
        """Overides the parent method
        """
        # block 2
        self.block2_filters = [(64, 64, 256)] * 3
        self.block2_shortcuts = ["projection", *("identity",) * 2]

        # block 3
        self.block3_filters = [(128, 128, 512)] * 4
        self.block3_shortcuts = ["projection", *("identity",) * 3]

        # block 4
        self.block4_filters = [(256, 256, 1024)] * 23
        self.block4_shortcuts = ["projection", *("identity",) * 22]

        # block 5
        self.block5_filters = [(512, 512, 2048)] * 3
        self.block5_shortcuts = ["projection", *("identity",) * 2]

        # to use bottleneck blocks or not
        self.bottleneck = True


class ResNet152(ResNet):
    """Resnet50
    """

    def __init__(self):
        """Initialize resnet152
        """
        super().__init__()

    def set_filters_and_shortcuts(self):
        """Overides the parent method
        """
        # block 2
        self.block2_filters = [(64, 64, 256)] * 3
        self.block2_shortcuts = ["projection", *("identity",) * 2]

        # block 3
        self.block3_filters = [(128, 128, 512)] * 8
        self.block3_shortcuts = ["projection", *("identity",) * 7]

        # block 4
        self.block4_filters = [(256, 256, 1024)] * 36
        self.block4_shortcuts = ["projection", *("identity",) * 35]

        # block 5
        self.block5_filters = [(512, 512, 2048)] * 3
        self.block5_shortcuts = ["projection", *("identity",) * 2]

        # to use bottleneck blocks or not
        self.bottleneck = True
