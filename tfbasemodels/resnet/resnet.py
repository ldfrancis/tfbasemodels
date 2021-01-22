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
        self.stage2_filters = None
        self.stage2_shortcuts = None
        self.stage3_filters = None
        self.stage3_shortcuts = None
        self.stage4_filters = None
        self.stage4_shortcuts = None
        self.stage5_filters = None
        self.stage5_shortcuts = None
        super().__init__()

    @abstractmethod
    def set_filters_and_shortcuts(self):
        """Must be implemented by subclass. Sets the number of filters and type of shortcut for each 
        the blocks in each stage, based on the number of layers
        specified.
        This method must set the following attributes
        self.stage2_filters : List[Tuple]
        self.stage2_shortcuts: List[string]
        self.stage3_filters
        self.stage3_shortcuts
        self.stage4_filters
        self.stage4_shortcuts
        self.stage5_filters
        self.stage5_shortcuts
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
        # stem; stage 1
        x = self.stem(x_inp)

        # pool
        # input size: 112x112x64
        # output size: 56x56x64
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                      strides=2,
                                      padding="same")(x)

        # input size: 56x56x64
        # output size: 56x56x_
        # stage 2
        x = self.build_stage(x,
                             stage=2,
                             blocks_filters=self.stage2_filters,
                             shortcuts=self.stage2_shortcuts,
                             strides=1)

        # stage 3
        # input size: 56x56x_
        # output size: 28x28x_
        x = self.build_stage(x,
                             stage=2,
                             blocks_filters=self.stage3_filters,
                             shortcuts=self.stage3_shortcuts,
                             strides=2)

        # stage 4
        # input size: 28x28x_
        # output size: 14x14x_
        x = self.build_stage(x,
                             stage=3,
                             blocks_filters=self.stage4_filters,
                             shortcuts=self.stage4_shortcuts,
                             strides=2)

        # stage 5
        # input size: 14x14x_
        # output size: 7x7x_
        x = self.build_stage(x,
                             stage=5,
                             blocks_filters=self.stage5_filters,
                             shortcuts=self.stage5_shortcuts,
                             strides=2)

        # pool
        # input: 7x7x_
        # output: 1x1x_
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        logits = tf.keras.layers.Dense(1000, name='logits')(x)

        return tf.keras.Model(inputs=[x_inp], outputs=[logits])

    def build_stage(self, x, stage=None, blocks_filters=None, shortcuts=None, strides=2):
        """A stage of a deep residual network

        Args: 
            x: Tensor. the input tensor
            stage: int or string. The stage of the resnet currently being built
            blocks_filters: list of tuples. a list containing the number of filters to use
                for each block in the stage
            shortcuts: list of strings. a list specifying the type of shortcut to use for 
                each block in the stage
            strides: int. strides to use for the first convolution and the shortcut

        Returns:
            a Tensor 
        """
        for i, block_filters, shortcut_ in enumerate(zip(blocks_filters, shortcuts)):
            reduction_network = len(blocks_filters) - 1 if self.preact else 0
            strides_ = strides if i == reduction_network else 1
            x = self.residual_block(x,
                                    shortcut=shortcut_,
                                    stage=stage,
                                    block=str(i + 1),
                                    conv_filters=block_filters,
                                    strides=strides_,
                                    bottleneck=self.bottleneck)

        return x

    def stem(self, x):
        """Stem layers for resnet consisting only of a convolutional layer that reduces
        the spatial dimensions from 224x224 to 112x112

        Args:
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
                              name=base_name + "_1",
                              activation=tf.nn.relu)(x)
            # _3x3
            x = conv_function(filters=filters2,
                              kernel_size=3,
                              strides=(strides if self.preact else 1),
                              padding="same",
                              name=base_name + "_2",
                              activation=tf.nn.relu)(x)
            # last _1x1
            x = conv_function(filters=filters3,
                              kernel_size=1,
                              strides=1,
                              padding="same",
                              name=base_name + "_3", )(x)
        else:
            # first 3x3: this conv does spatial reduction with the strides and matches the shortcut spatial size
            x = conv_function(filters=filters1,
                              kernel_size=3,
                              strides=strides,
                              padding="same",
                              name=base_name + "_1",
                              activation=tf.nn.relu)(x)
            # last 3x3
            x = conv_function(filters=filters3,
                              kernel_size=3,
                              strides=1,
                              padding="same",
                              name=base_name + "_2", )(x)

        # add shortcut to residual output
        x = tf.keras.layers.Add(name=base_name + "_add")([shortcut_x, x])

        # activation (accounts for pre-activation)
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
        """Overrides the parent method
        """
        # stage 2
        self.stage2_filters = [(64, 64)] * 2
        self.stage2_shortcuts = ["projection", "identity"]

        # stage 3
        self.stage3_filters = [(128, 128)] * 2
        self.stage3_shortcuts = ["projection", "identity"]

        # stage 4
        self.stage4_filters = [(256, 256)] * 2
        self.stage4_shortcuts = ["projection", "identity"]

        # stage 5
        self.stage5_filters = [(512, 512)] * 2
        self.stage5_shortcuts = ["projection", "identity"]

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
        """Overrides the parent method
        """
        # stage 2
        self.stage2_filters = [(64, 64)] * 3
        self.stage2_shortcuts = ["projection", *("identity",) * 2]

        # stage 3
        self.stage3_filters = [(128, 128)] * 4
        self.stage3_shortcuts = ["projection", *("identity",) * 3]

        # stage 4
        self.stage4_filters = [(256, 256)] * 6
        self.stage4_shortcuts = ["projection", *("identity",) * 5]

        # stage 5
        self.stage5_filters = [(512, 512)] * 3
        self.stage5_shortcuts = ["projection", *("identity",) * 2]

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
        """Overrides the parent method
        """
        # stage 2
        self.stage2_filters = [(64, 64, 256)] * 3
        self.stage2_shortcuts = ["projection", *("identity",) * 2]

        # stage 3
        self.stage3_filters = [(128, 128, 512)] * 4
        self.stage3_shortcuts = ["projection", *("identity",) * 3]

        # stage 4
        self.stage4_filters = [(256, 256, 1024)] * 6
        self.stage4_shortcuts = ["projection", *("identity",) * 5]

        # stage 5
        self.stage5_filters = [(512, 512, 2048)] * 3
        self.stage5_shortcuts = ["projection", *("identity",) * 2]

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
        """Overrides the parent method
        """
        # stage 2
        self.stage2_filters = [(64, 64, 256)] * 3
        self.stage2_shortcuts = ["projection", *("identity",) * 2]

        # stage 3
        self.stage3_filters = [(128, 128, 512)] * 4
        self.stage3_shortcuts = ["projection", *("identity",) * 3]

        # stage 4
        self.stage4_filters = [(256, 256, 1024)] * 23
        self.stage4_shortcuts = ["projection", *("identity",) * 22]

        # stage 5
        self.stage5_filters = [(512, 512, 2048)] * 3
        self.stage5_shortcuts = ["projection", *("identity",) * 2]

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
        """Overrides the parent method
        """
        # stage 2
        self.stage2_filters = [(64, 64, 256)] * 3
        self.stage2_shortcuts = ["projection", *("identity",) * 2]

        # stage 3
        self.stage3_filters = [(128, 128, 512)] * 8
        self.stage3_shortcuts = ["projection", *("identity",) * 7]

        # stage 4
        self.stage4_filters = [(256, 256, 1024)] * 36
        self.stage4_shortcuts = ["projection", *("identity",) * 35]

        # stage 5
        self.stage5_filters = [(512, 512, 2048)] * 3
        self.stage5_shortcuts = ["projection", *("identity",) * 2]

        # to use bottleneck blocks or not
        self.bottleneck = True
