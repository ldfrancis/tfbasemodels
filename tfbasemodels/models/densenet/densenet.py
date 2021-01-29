"""A implementation for DenseNet based on code keras application at
https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py

Paper:
Densely Connected Convolutional Networks. https://arxiv.org/abs/1608.06993
"""
from abc import abstractmethod

import tensorflow as tf
from ..model import TFBaseModel


class DenseNet(TFBaseModel):
    """DenseNet model
    """
    input_shape = (224, 224, 3)
    classes = 1000
    include_top = False
    pooling = 'avg'
    pretrained = False
    name = "densenet"
    num_dense_block1 = None
    num_dense_block2 = None
    num_dense_block3 = None
    num_dense_block4 = None

    def __int__(self, include_top=True, pooling=None, pretrained=False):
        super().__init__()
        self.include_top = include_top
        self.pooling = pooling
        self.pretrained = pretrained
        self.set_num_dense_block()

    @abstractmethod
    def set_num_dense_block(self):
        """Sets the number of convolution blocks in each dense block of a desnet. This attributes are named;
        num_dense_block1, num_dense_block2, num_dense_block3, num_dense_block4
        """
        raise NotImplementedError

    def build(self):
        x_input = tf.keras.layers.Input(shape=self.input_shape)

        # stem
        # in: 224x224x3
        # out: 56x56x64
        x = DenseNet.stem_network(x_input)

        # dense_block 1
        # in: 56x56
        # out: 56x56
        x = DenseNet.dense_block(x, self.num_dense_block1, "dense_block1")

        # transition 1
        # in: 56x56
        # out: 28x28
        x = DenseNet.transition_block(x, "transition1")

        # dense_block 2
        # in: 28x28
        # out: 28x28
        x = DenseNet.dense_block(x, self.num_dense_block2, "dense_block2")

        # transition 2
        # in: 28x28
        # out: 14x14
        x = DenseNet.transition_block(x, "transition2")

        # dense_block 3
        # in: 14x14
        # out: 14x14
        x = DenseNet.dense_block(x, self.num_dense_block3, "dense_block3")

        # transition 3
        # in: 14x14
        # out: 7x7
        x = DenseNet.transition_block(x, "transition3")

        # dense_block 4
        # in: 7x7
        # out: 7x7
        x = DenseNet.dense_block(x, self.num_dense_block4, "dense_block4")

        x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='bn')(x)
        x = tf.keras.layers.Activation('relu', name='relu')(x)

        if self.include_top:
            x = tf.keras.layers.GlobalAveragePooling2D(name="avgpool")(x)
            x = tf.keras.layers.Dense(self.classes, activation="softmax", name="fc")(x)
        else:
            if self.pooling=='avg':
                x = tf.keras.layers.GlobalAveragePooling2D(name="avgpool")(x)
            elif self.pooling=='max':
                x = tf.keras.layers.GlobalMaxPooling2D(name="maxpool")(x)

        return tf.keras.Model(inputs=[x_input], outputs=[x])

    @staticmethod
    def stem_network(x):
        x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='stem/conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=3, epsilon=1.001e-5, name='stem/bn')(x)
        x = tf.keras.layers.Activation('relu', name='stem/relu')(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='stem/pool')(x)
        return x

    @staticmethod
    def dense_block(x, conv_blocks, name, growth_rate=32):
        """A dense block used in a densenet. Each convolution makes use of a dense_conv method that not only performs a
        convolution operation on the input but concatenates the input with the result of convolution. This results in
        each subsequent dense_conv in the block to take as input a concatenation of all previous outputs in the block.
        """
        for i in range(conv_blocks):
            x = DenseNet.conv_block(x, growth_rate, name=name + '/conv_block/' + str(i + 1))
        return x

    @staticmethod
    def conv_block(x, growth_rate, name):
        """This method makes dense convolutions possible in a dense block. It passes it's input through a convolutional
        layer and returns as an output, the concatenation of the result and the input. This enable the next convolutional
        layer in a dense block to take as input, a concatenation of outputs from all the previous layers.
        Args:
            x:
            growth_rate:
            name:
        Returns:
            Tensor: A concatenation of the result of convolution and the input, x
        """
        x1 = tf.keras.layers.BatchNormalization(axis=3,
                                                epsilon=1.001e-5,
                                                name=name + '0_bn')(x)
        x1 = tf.keras.layers.Activation('relu', name=name + '0_relu')(x1)
        x1 = tf.keras.layers.Conv2D(4 * growth_rate, 1,
                                    use_bias=False,
                                    name=name + '1_conv')(x1)
        x1 = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5,
                                                name=name + '1_bn')(x1)
        x1 = tf.keras.layers.Activation('relu', name=name + '1_relu')(x1)
        x1 = tf.keras.layers.Conv2D(growth_rate, 3,
                                    padding='same',
                                    use_bias=False,
                                    name=name + '2_conv')(x1)
        x = tf.keras.layers.Concatenate(axis=3, name=name + 'concat')([x, x1])
        return x

    @staticmethod
    def transition_block(x, reduction, name):
        """A transition block is usually used in between two dense blocks. It acts to reduce the spatial dimension of the
        output from a block, before it is fed into the next block.
        """
        B, H, W, C = x.shape
        x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5,
                                               name=name + '_bn')(x)
        x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
        x = tf.keras.layers.Conv2D(int(C * reduction), 1,
                                   use_bias=False,
                                   name=name + '_conv')(x)
        x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x


class DenseNet121(DenseNet):

    def __init__(self, include_top=True, pooling=None, pretrained=False):
        self.name = "densenet121"
        super().__init__(include_top, pooling, pretrained)

    def set_num_dense_block(self):
        self.num_dense_block1 = 6
        self.num_dense_block2 = 12
        self.num_dense_block3 = 24
        self.num_dense_block4 = 16


class DenseNet169(DenseNet):

    def __init__(self, include_top=True, pooling=None, pretrained=False):
        self.name = "densenet121"
        super().__init__(include_top, pooling, pretrained)

    def set_num_dense_block(self):
        self.num_dense_block1 = 6
        self.num_dense_block2 = 12
        self.num_dense_block3 = 32
        self.num_dense_block4 = 32


class DenseNet201(DenseNet):

    def __init__(self, include_top=True, pooling=None, pretrained=False):
        self.name = "densenet121"
        super().__init__(include_top, pooling, pretrained)

    def set_num_dense_block(self):
        self.num_dense_block1 = 6
        self.num_dense_block2 = 12
        self.num_dense_block3 = 48
        self.num_dense_block4 = 32
