"""Inception implementation based on code from keras_application at
https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py
https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_resnet_v2.py
"""
import tensorflow as tf
from ..model import TFBaseModel
from tfbasemodels.utils.layer_utils import conv2d_bn


class GoogLeNet(TFBaseModel):
    """GoogLeNet. input size: 224x224x3
    """

    def __init__(self):
        super().__init__()

    def build(self):
        # input layer
        x_inp = tf.keras.layers.Input(shape=[224, 224, 3])

        # stem 1 & 2
        # input: 224x224x3
        # output: 28x28x192
        x = self.stem_network(x_inp)

        # inception 3
        # input: 28x28x192
        # output: 28x28x256
        # 3a
        x = self.inception_module(x,
                                  _1x1=64,
                                  _3x3_reduce=96,
                                  _3x3=128,
                                  _5x5_reduce=16,
                                  _5x5=32,
                                  pool_proj=32,
                                  name="3a")
        # input: 28x28x256
        # output: 28x28x480
        # 3b
        x = self.inception_module(x,
                                  _1x1=128,
                                  _3x3_reduce=128,
                                  _3x3=192,
                                  _5x5_reduce=32,
                                  _5x5=96,
                                  pool_proj=64,
                                  name="3b")
        # input: 28x28x480
        # output: 14x14x480
        # pool
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                      strides=2,
                                      padding="same",
                                      name="maxpool_3")(x)
        # input: 14x14x480
        # output: 14x14x512
        # 4a
        x = self.inception_module(x,
                                  _1x1=192,
                                  _3x3_reduce=96,
                                  _3x3=208,
                                  _5x5_reduce=16,
                                  _5x5=48,
                                  pool_proj=64,
                                  name="4a")
        # input: 14x14x512
        # output: 14x14x512
        # 4b
        x = self.inception_module(x,
                                  _1x1=160,
                                  _3x3_reduce=112,
                                  _3x3=224,
                                  _5x5_reduce=24,
                                  _5x5=64,
                                  pool_proj=64,
                                  name="4b")
        # input: 14x14x512
        # output: 14x14x512
        # 4c
        x = self.inception_module(x,
                                  _1x1=128,
                                  _3x3_reduce=128,
                                  _3x3=256,
                                  _5x5_reduce=24,
                                  _5x5=64,
                                  pool_proj=64,
                                  name="4c")
        # input: 14x14x512
        # output: 14x14x528
        # 4d
        x = self.inception_module(x,
                                  _1x1=112,
                                  _3x3_reduce=144,
                                  _3x3=288,
                                  _5x5_reduce=32,
                                  _5x5=64,
                                  pool_proj=64,
                                  name="4d")
        # input: 14x14x528
        # output: 14x14x832
        # 4e
        x = self.inception_module(x,
                                  _1x1=256,
                                  _3x3_reduce=60,
                                  _3x3=320,
                                  _5x5_reduce=32,
                                  _5x5=128,
                                  pool_proj=128,
                                  name="4e")
        # input: 14x14x832
        # output: 7x7x832
        # maxpool 4
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                      strides=2,
                                      name="maxpool_4")(x)
        # input: 7x7x832
        # output: 7x7x832
        # 5a
        x = self.inception_module(x,
                                  _1x1=256,
                                  _3x3_reduce=160,
                                  _3x3=320,
                                  _5x5_reduce=32,
                                  _5x5=128,
                                  pool_proj=128,
                                  name="5a")
        # input: 7x7x832
        # output: 7x7x1024
        # 5b
        x = self.inception_module(x,
                                  _1x1=384,
                                  _3x3_reduce=192,
                                  _3x3=384,
                                  _5x5_reduce=48,
                                  _5x5=128,
                                  pool_proj=128,
                                  name="5b")
        # input: 7x7x1024
        # output: 1x1x1024
        # avg pool
        x = tf.keras.layers.AveragePooling2D(pool_size=7,
                                             strides=1,
                                             padding="valid",
                                             name="avgpool")(x)
        # input: 1x1x1024
        # output: 1x1x1024
        # dropout
        x = tf.keras.layers.Dropout(rate=0.4,
                                    name="dropout")(x)

        # input: 1x1x1024
        # output: 1x1x1024
        # linear 6
        x = tf.keras.layers.Dense(units=1000,
                                  activation=tf.nn.relu,
                                  name="linear_6")(x)
        # input: 1x1x1000
        # output:1x1x1000
        # logits
        logits = tf.keras.layers.Dense(units=1000,
                                       name="logits")(x)

        # tf keras Model
        return tf.keras.Model(inputs=[x_inp], outputs=[logits])

    def stem_network(self, x):
        """Stem network. Used to aggressively reduce the spatial dimensions
        before the inception modules

        Args:
            x: Tensor. Input tensor

        Returns:
            a Tensor
        """
        base_name = "stem"
        # input: 224x224x3
        # output: 112x112x64
        x = conv2d_bn(filters=64,
                      kernel_size=7,
                      strides=2,
                      padding="same",
                      activation=tf.nn.relu,
                      name=base_name + "_conv1")(x)
        # input: 112x112x64
        # output: 56x56x64
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                      strides=2,
                                      padding="same",
                                      name=base_name + "_maxpool1")(x)
        # input: 56x56x64
        # output: 56x56x64
        x = conv2d_bn(filters=64,
                      kernel_size=1,
                      strides=1,
                      padding="same",
                      activation=tf.nn.relu,
                      name=base_name + "_conv2a")(x)
        # input: 56x56x64
        # output: 56x56x192
        x = conv2d_bn(filters=192,
                      kernel_size=3,
                      strides=1,
                      padding="same",
                      activation=tf.nn.relu,
                      name=base_name + "_conv2b")(x)
        # input: 56x56x192
        # output: 28x28x192
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                      strides=2,
                                      padding="same",
                                      name=base_name + "_maxpool2")(x)

        return x

    def inception_module(self, x, _1x1, _3x3_reduce, _3x3, _5x5_reduce, _5x5, pool_proj, name):
        """Inception module

        Args:
            x: Tensor. Input tensor
            _1x1: int. Number of filters for the 1x1 conv
            _3x3_reduce: int. Number of filters for the 1x1 conv before the 3x3 conv
            _3x3: int. Number of filters for the 3x3 conv after a 1x1 conv
            _5x5_reduce: int. Number of filters for 1x1 conv before 5x5 conv
            _5x5: int. Number of filters for the 5x5 conv after a 1x1 conv
            pool_proj: int. Number of filters for the 1x1 convolution after maxpooling
            name: str. Name to be assigned to the inception module

        Returns: 
            a Tensor
        """
        base_name = "inception_" + name

        conv1x1 = conv2d_bn(filters=_1x1,
                            kernel_size=1,
                            strides=1,
                            padding="same",
                            activation=tf.nn.relu,
                            name=base_name + "_1x1")(x)

        conv3x3_reduce = conv2d_bn(filters=_3x3_reduce,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   activation=tf.nn.relu,
                                   name=base_name + "_3x3_reduce")(x)

        conv3x3 = conv2d_bn(filters=_3x3,
                            kernel_size=3,
                            strides=1,
                            padding="same",
                            activation=tf.nn.relu,
                            name=base_name + "_3x3")(conv3x3_reduce)

        conv5x5_reduce = conv2d_bn(filters=_5x5_reduce,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   activation=tf.nn.relu,
                                   name=base_name + "_5x5_reduce")(x)

        conv5x5 = conv2d_bn(filters=_5x5,
                            kernel_size=5,
                            strides=1,
                            padding="same",
                            activation=tf.nn.relu,
                            name=base_name + "_5x5")(conv5x5_reduce)

        max3x3 = tf.keras.layers.MaxPool2D(pool_size=3,
                                           strides=1,
                                           padding="same",
                                           name=base_name + "max3x3")(x)

        maxpool_proj = conv2d_bn(filters=pool_proj,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 activation=tf.nn.relu,
                                 name=base_name + "_pool_proj")(max3x3)

        concat = tf.keras.layers.Concatenate(name=base_name + "_concat",
                                             )([conv1x1, conv3x3, conv5x5, maxpool_proj])

        return concat


Inceptionv1 = GoogLeNet


class Inceptionv3(TFBaseModel):
    """
    Inceptionv3 based on the keras application implementation.
    """

    def __init__(self, include_top=True, pooling='avg', pretrained=False):
        super().__init__()
        self.input_shape = (229, 229, 3)
        self.include_top = include_top
        self.pooling = pooling
        self.pretrained = pretrained

    def build(self):
        x_inp = tf.keras.layers.Input(shape=self.input_shape)

        # stem
        # in: 299×299×3
        # out: 35×35×288
        x = self.stem_network(x_inp)

        # 3 x inception module 1
        # in: 35×35×288
        # out: 17×17×768
        for i in range(3):
            x = self.inception_module_1(x, i + 1)

        # 5 x inception module 2
        # in: 17×17×768
        # out: 8×8×1280
        for i in range(5):
            x = self.inception_module_2(x, i + 1)

        # 2 x inception module 3
        # in: 8x8x1280
        # out: 8x8x2048
        for i in range(3):
            x = self.inception_module3(x, i + 1)

        if self.include_top:
            # Classification block
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = tf.keras.layers.Dense(1000, activation='softmax', name='predictions')(x)
        else:
            if self.pooling == 'avg':
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
            elif self.pooling == 'max':
                x = tf.keras.layers.GlobalMaxPooling2D()(x)

        return tf.keras.Model(inputs=[x_inp], outptus=[x])

    @staticmethod
    def stem_network(self, x):
        """Stem network comprising of a series of operations applied to the input before
        applying the inception modules.
        Args:
            x: Tensor. input to the stem network
        Returns:
            x: Tensor. output of the stem network
        """
        x = conv2d_bn(32, 3, strides=2, activation=tf.nn.relu, name="stem_1")(x)
        x = conv2d_bn(32, 3, activation=tf.nn.relu, name="stem_2")(x)
        x = conv2d_bn(64, 3, padding="same", activation=tf.nn.relu, name="stem_3")(x)
        x = tf.keras.layers.MaxPool2D((3, 3), (2, 2), name="stem_pool1")(x)
        x = conv2d_bn(80, 1, activation=tf.nn.relu, name="stem_4")(x)
        x = conv2d_bn(192, 3, activation=tf.nn.relu, name="stem_5")(x)
        x = tf.keras.layers.MaxPool2D((3, 3), (2, 2), name="stem_pool2")(x)
        return x

    @staticmethod
    def inception_module_1(x, name):
        """Builds an inception module
        Args:
            x: Tensor. input to the module
            name: int. Used to set the base name of the module.
        Returns:
            x: Tensor. Output of the inception module.
        """
        assert name is not None, "name must be supplied"
        base_name = "inception1" + f"_{name}_"
        _1x1 = conv2d_bn(64, 1, padding="same", activation=tf.nn.relu, name=base_name + "1")(x)
        _5x5 = conv2d_bn(48, 1, padding="same", activation=tf.nn.relu, name=base_name + "5_1")(x)
        _5x5 = conv2d_bn(64, 5, padding="same", activation=tf.nn.relu, name=base_name + "5_2")(_5x5)
        _3x3 = conv2d_bn(64, 1, padding="same", activation=tf.nn.relu, name=base_name + "3_1")(x)
        _3x3 = conv2d_bn(96, 3, padding="same", activation=tf.nn.relu, name=base_name + "3_2")(_3x3)
        _3x3 = conv2d_bn(96, 3, padding="same", activation=tf.nn.relu, name=base_name + "3_3")(_3x3)
        pool = tf.keras.layers.AveragePooling2D(3, 1, padding="same", name=base_name + "avgpool")(x)
        pool = conv2d_bn(32, 1, padding="same", activation=tf.nn.relu, name=base_name + "avgpool_1")(pool)
        concat = tf.keras.layers.concatenate([_1x1, _5x5, _3x3, pool], axis=3, name=base_name + "concat")
        return concat

    @staticmethod
    def inception_module_2(x, name):
        """Builds an inception module
        """
        assert name is not None, "name must be supplied"
        base_name = "inception2" + f"_{name}_"
        if name == 1:
            _3x3 = conv2d_bn(384, 3, strides=(2, 2), padding='valid', activation=tf.nn.relu, name=base_name + "3")(x)
            _3x3dbl = conv2d_bn(64, 1, padding="same", activation=tf.nn.relu, name=base_name + "3dbl_1")(x)
            _3x3dbl = conv2d_bn(96, 3, padding="same", activation=tf.nn.relu, name=base_name + "3dbl_2")(_3x3dbl)
            _3x3dbl = conv2d_bn(96, 3, strides=(2, 2), padding='valid', activation=tf.nn.relu,
                                name=base_name + "3dbl_3")(_3x3dbl)
            pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
            concat = tf.keras.layers.concatenate([_3x3, _3x3dbl, pool], axis=3, name=base_name + 'concat')
            return concat
        elif name == 2:
            _1x1 = conv2d_bn(192, 1, padding="same", activation=tf.nn.relu, name=base_name + "1")(x)
            _7x7 = conv2d_bn(128, 1, padding="same", activation=tf.nn.relu, name=base_name + "7_1")(x)
            _7x7 = conv2d_bn(128, (1, 7), padding="same", activation=tf.nn.relu, name=base_name + "7_2")(_7x7)
            _7x7 = conv2d_bn(192, (7, 1), padding="same", activation=tf.nn.relu, name=base_name + "7_3")(_7x7)
            _7x7dbl = conv2d_bn(128, 1, padding="same", activation=tf.nn.relu, name=base_name + "7dbl_1")(x)
            _7x7dbl = conv2d_bn(128, (7, 1), padding="same", activation=tf.nn.relu, name=base_name + "7dbl_2")(_7x7dbl)
            _7x7dbl = conv2d_bn(128, (1, 7), padding="same", activation=tf.nn.relu, name=base_name + "7dbl_3")(_7x7dbl)
            _7x7dbl = conv2d_bn(128, (7, 1), padding="same", activation=tf.nn.relu, name=base_name + "7dbl_4")(_7x7dbl)
            _7x7dbl = conv2d_bn(192, (1, 7), padding="same", activation=tf.nn.relu, name=base_name + "7dbl_5")(_7x7dbl)
            pool = tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=base_name + "pool_1")(
                x)
            pool = conv2d_bn(192, 1, padding="same", activation=tf.nn.relu, name=base_name + "pool_2")(pool)
            concat = tf.keras.layers.concatenate([_1x1, _7x7, _7x7dbl, pool], axis=3, name=base_name + "concat")(pool)
            return concat
        elif name == 5:
            _1x1 = conv2d_bn(192, 1, padding="same", activation=tf.nn.relu, name=base_name + "1")(x)
            _7x7 = conv2d_bn(192, 1, padding="same", activation=tf.nn.relu, name=base_name + "7_1")(x)
            _7x7 = conv2d_bn(192, (1, 7), padding="same", activation=tf.nn.relu, name=base_name + "7_2")(_7x7)
            _7x7 = conv2d_bn(192, (7, 1), padding="same", activation=tf.nn.relu, name=base_name + "7_3")(_7x7)
            _7x7dbl = conv2d_bn(192, 1, padding="same", activation=tf.nn.relu, name=base_name + "7dbl_1")(x)
            _7x7dbl = conv2d_bn(192, (7, 1), padding="same", activation=tf.nn.relu, name=base_name + "7dbl_2")(_7x7dbl)
            _7x7dbl = conv2d_bn(192, (1, 7), padding="same", activation=tf.nn.relu, name=base_name + "7dbl_3")(_7x7dbl)
            _7x7dbl = conv2d_bn(192, (7, 1), padding="same", activation=tf.nn.relu, name=base_name + "7dbl_4")(_7x7dbl)
            _7x7dbl = conv2d_bn(192, (1, 7), padding="same", activation=tf.nn.relu, name=base_name + "7dbl_5")(_7x7dbl)
            pool = tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                                    name=base_name + "pool_1")(x)
            pool = conv2d_bn(192, 1, padding="same", activation=tf.nn.relu, name=base_name + "pool_2")(pool)
            concat = tf.keras.layers.concatenate([_1x1, _7x7, _7x7dbl, pool], axis=3, name=base_name + "concat")
            return concat
        else:
            _1x1 = conv2d_bn(192, 1, padding="same", activation=tf.nn.relu, name=base_name + "1")(x)
            _7x7 = conv2d_bn(160, 1, padding="same", activation=tf.nn.relu, name=base_name + "7_1")(x)
            _7x7 = conv2d_bn(160, (1, 7), padding="same", activation=tf.nn.relu, name=base_name + "7_2")(_7x7)
            _7x7 = conv2d_bn(192, (7, 1), padding="same", activation=tf.nn.relu, name=base_name + "7_3")(_7x7)
            _7x7dbl = conv2d_bn(160, 1, padding="same", activation=tf.nn.relu, name=base_name + "7_1")(x)
            _7x7dbl = conv2d_bn(160, (7, 1), padding="same", activation=tf.nn.relu, name=base_name + "7_2")(_7x7dbl)
            _7x7dbl = conv2d_bn(160, (1, 7), padding="same", activation=tf.nn.relu, name=base_name + "7_3")(_7x7dbl)
            _7x7dbl = conv2d_bn(160, (7, 1), padding="same", activation=tf.nn.relu, name=base_name + "7_4")(_7x7dbl)
            _7x7dbl = conv2d_bn(192, (1, 7), padding="same", activation=tf.nn.relu, name=base_name + "7_5")(_7x7dbl)
            pool = tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=base_name + "pool")(x)
            pool = conv2d_bn(192, 1, padding="same", activation=tf.nn.relu, name=base_name + "pool")(pool)
            concat = tf.keras.layers.concatenate([_1x1, _7x7, _7x7dbl, pool], axis=3, name=base_name + "concat")
            return concat

    @staticmethod
    def inception_module3(x, name):
        assert name is not None, "name must be supplied"
        base_name = f"inception3_{name}_"
        if name == 1:
            _3x3 = conv2d_bn(x, 192, 1, padding="same", activation=tf.nn.relu, name=base_name + "3_1")(x)
            _3x3 = conv2d_bn(320, 3, strides=(2, 2), padding='valid', activation=tf.nn.relu, name=base_name + "3_2")

            _7x7x3 = conv2d_bn(x, 192, 1, padding="same", activation=tf.nn.relu, name=base_name + "7_1")(x)
            _7x7x3 = conv2d_bn(192, (1, 7), padding="same", activation=tf.nn.relu, name=base_name + "7_2")(_7x7x3)
            _7x7x3 = conv2d_bn(192, (7, 1), padding="same", activation=tf.nn.relu, name=base_name + "7_3")(_7x7x3)
            _7x7x3 = conv2d_bn(192, 3, strides=(2, 2), padding='valid', activation=tf.nn.relu,
                               name=base_name + "7_4")(_7x7x3)
            pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
            concat = tf.keras.layers.concatenate([_3x3, _7x7x3, pool], axis=3, name=base_name + "concat")
            return concat
        else:
            _1x1 = conv2d_bn(320, 1, padding="same", activation=tf.nn.relu, name=base_name + "1")(x)
            _3x3 = conv2d_bn(384, 1, padding="same", activation=tf.nn.relu, name=base_name + "3_1")(x)
            _3x3_1 = conv2d_bn(384, (1, 3), padding="same", activation=tf.nn.relu, name=base_name + "3_2")(_3x3)
            _3x3_2 = conv2d_bn(384, (3, 1), padding="same", activation=tf.nn.relu, name=base_name + "3_3")(_3x3)
            _3x3 = tf.keras.layers.concatenate([_3x3_1, _3x3_2], axis=3, name=base_name + "3_4")
            _3x3dbl = conv2d_bn(448, 1, padding="same", activation=tf.nn.relu, name=base_name + "3dbl_1")(x)
            _3x3dbl = conv2d_bn(384, 3, padding="same", activation=tf.nn.relu, name=base_name + "3dbl_2")(_3x3dbl)
            _3x3dbl_1 = conv2d_bn(384, (1, 3), padding="same", activation=tf.nn.relu, name=base_name + "3dbl_3")(
                _3x3dbl)
            _3x3dbl_2 = conv2d_bn(384, (3, 1), padding="same", activation=tf.nn.relu, name=base_name + "3dbl_4")(
                _3x3dbl)
            _3x3dbl = tf.keras.layers.concatenate([_3x3dbl_1, _3x3dbl_2], axis=3, name=base_name + "3dbl_5")
            pool = tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=base_name + "pool_1")(
                x)
            pool = conv2d_bn(192, 1, padding="same", activation=tf.nn.relu, name=base_name + "pool_2")(pool)
            concat = tf.keras.layers.concatenate([_1x1, _3x3, _3x3dbl, pool], axis=3, name=base_name + "concat")
            return concat


class InceptionResnet(TFBaseModel):
    """The inception-resnet model"""

    def __init__(self):
        super().__init__()
        self.input_shape = (299, 299, 3)
        self.num_classes = 1000

    def build(self):
        # input: 299,299,3
        x_input = tf.keras.layers.Input(shape=(self.input_shape,))

        # stem
        # in: 299,299,3
        # out: 35x35x192
        x = self.stem_network(x_input)

        # inception-A block
        # in: 35x35x192
        # out: 35 x 35 x 320
        x = self.inception_A_block(x)

        # inception-resnet-A block
        # in: 35x35x320
        # out: 35 x 35 x 320
        for i in range(10):
            x = self.inception_resnet_A_block(x, 0.75, tf.nn.relu, idx=i + 1)

        # reduction-A block
        # in: 35 x 35 x 320
        # out: 17 x 17 x 1088
        x = self.reduction_A_block(x)

        # Inception-ResNet-B block
        # in: 17 x 17 x 1088
        # out: 17 x 17 x 1088
        for i in range(20):
            x = self.inception_resnet_B_block(x, 0.1, tf.nn.relu, idx=i + 1)

        # Reduction-B block
        # in: 17 x 17 x 1088
        # out: 8 x 8 x 2080
        x = self.reduction_B_block(x)

        # Inception-ResNet-C block
        # in: 8 x 8 x 2080
        # out: 8 x 8 x 2080
        for i in range(9):
            x = self.inception_resnet_C_block(x, 0.2, tf.nn.relu, idx=i + 1)
        x = self.inception_resnet_C_block(x, 1, None, idx=10)

        # Final Conv
        # in: 8 x 8 x 2080
        # out: 8 x 8 x 1536
        x = conv2d_bn(1536, 1, padding="same", use_bias=False, activation=tf.nn.relu, name='Final_Conv')(x)

        # define output
        if self.include_top:
            # Classification block
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        else:
            if self.pooling == 'avg':
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
            elif self.pooling == 'max':
                x = tf.keras.layers.GlobalMaxPooling2D()(x)

        return tf.keras.Model(inputs=[x_input], outputs=[x])

    @staticmethod
    def stem_network(x):
        """Stem network. Set of operations applied to input before using the inception modules"""
        base_name = "stem"
        x = conv2d_bn(32, 3, 2, padding="valid", activation=tf.nn.relu, use_bias=False, name=base_name + "_1")(x)
        x = conv2d_bn(32, 3, use_bias=False, name=base_name + "_2")(x)
        x = conv2d_bn(64, 3, padding="same", use_bias=False, name=base_name + "_3")(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
        x = conv2d_bn(80, 1, use_bias=False, name=base_name + "_4")(x)
        x = conv2d_bn(192, 3, use_bias=False, name=base_name + "_5")(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
        return x

    @staticmethod
    def inception_A_block(x):
        base_name = "inception_A"
        _1x1 = conv2d_bn(96, 1, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_1")(x)
        _5x5 = conv2d_bn(48, 1, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_5_1")(x)
        _5x5 = conv2d_bn(64, 5, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_5_2")(_5x5)
        _3x3 = conv2d_bn(64, 1, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_3_1")(x)
        _3x3 = conv2d_bn(96, 3, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_3_2")(_3x3)
        _3x3 = conv2d_bn(96, 3, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_3_3")(_3x3)
        pool = tf.keras.layers.AveragePooling2D(3, strides=1, padding='same', name=base_name + "avgpool_1")(x)
        pool = conv2d_bn(64, 1, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "avgpool_2")(
            pool)
        concat = tf.keras.layers.concatenate([_1x1, _5x5, _3x3, pool], axis=3, name=base_name + "_concat")
        return concat

    @staticmethod
    def reduction_A_block(x):
        base_name = "reduction_A"
        _3x3 = conv2d_bn(384, 3, strides=2, use_bias=False, activation=tf.nn.relu, name=base_name + "_3")(x)
        _3x3b = conv2d_bn(256, 1, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_3b_1")(x)
        _3x3b = conv2d_bn(256, 3, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_3b_2")(
            _3x3b)
        _3x3b = conv2d_bn(384, 3, strides=2, use_bias=False, activation=tf.nn.relu, name=base_name + "_3b_3")(_3x3b)
        pool = tf.keras.layers.MaxPooling2D(3, strides=2, padding='valid')(x)
        concat = tf.keras.layers.concatenate([_3x3, _3x3b, pool])
        return concat

    @staticmethod
    def reduction_B_block(x):
        base_name = "reduction_B"
        _3x3 = conv2d_bn(256, 1, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_3_1")(x)
        _3x3 = conv2d_bn(384, 3, strides=2, use_bias=False, activation=tf.nn.relu, name=base_name + "_3_2")(_3x3)
        _3x3b = conv2d_bn(256, 1, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_3b_1")(x)
        _3x3b = conv2d_bn(288, 3, strides=2, use_bias=False, activation=tf.nn.relu, name=base_name + "_3b_2")(_3x3b)
        _3x3c = conv2d_bn(256, 1, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_3c_1")(x)
        _3x3c = conv2d_bn(288, 3, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_3c_2")(
            _3x3c)
        _3x3c = conv2d_bn(320, 3, strides=2, use_bias=False, activation=tf.nn.relu, name=base_name + "_3c_3")(_3x3c)
        pool = tf.keras.layers.MaxPooling2D(3, strides=2, padding='valid')(x)
        concat = tf.keras.layers.concatenate([_3x3, _3x3b, _3x3c, pool], axis=3, name=base_name + "_concat")
        return concat

    @staticmethod
    def inception_resnet_A_block(x, scale, activation=None, idx=0):
        base_name = f"inception_resnet_A_{idx}"
        _1x1 = conv2d_bn(32, 1, padding="same", activation=tf.nn.relu, use_bias=False, name=base_name + "_1")(x)
        _3x3 = conv2d_bn(32, 1, padding='same', activation=tf.nn.relu, use_bias=False, name=base_name + "_3_1")(x)
        _3x3 = conv2d_bn(32, 3, padding="same", activation=tf.nn.relu, use_bias=False, name=base_name + "_3_1")(_3x3)
        _3x3b = conv2d_bn(32, 1, padding="same", activation=tf.nn.relu, use_bias=False, name=base_name + "_3b_1")(x)
        _3x3b = conv2d_bn(48, 3, padding="same", activation=tf.nn.relu, use_bias=False, name=base_name + '_3b_2')(_3x3b)
        _3x3b = conv2d_bn(64, 3, padding="same", activation=tf.nn.relu, use_bias=False, name=base_name + "_3b_3")(_3x3b)
        concat = tf.keras.layers.concatenate([_1x1, _3x3, _3x3b], axis=3, name=base_name + "_concat")
        return InceptionResnet.inception_resnet_block_tip(x, scale, concat, activation, base_name)

    @staticmethod
    def inception_resnet_B_block(x, scale, activation=None, idx=0):
        base_name = f"inception_resnet_B_{idx}"
        _1x1 = conv2d_bn(192, 1, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_1")(x)
        _7x7 = conv2d_bn(128, 1, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_7_1")(x)
        _7x7 = conv2d_bn(160, [1, 7], padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_7_2")(
            _7x7)
        _7x7 = conv2d_bn(192, [7, 1], padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_7_3")(
            _7x7)
        concat = tf.keras.layers.concatenate([_1x1, _7x7], axis=3, name=base_name + "_concat")
        return InceptionResnet.inception_resnet_block_tip(x, scale, concat, activation, base_name)

    @staticmethod
    def inception_resnet_C_block(x, scale, activation=None, idx=0):
        base_name = f"inception_resnet_C_{idx}"
        _1x1 = conv2d_bn(192, 1, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_1")(x)
        _3x3 = conv2d_bn(192, 1, padding="same", use_bias=False, activation=tf.nn.relu, name=base_name + "_3_1")(x)
        _3x3 = conv2d_bn(224, [1, 3], padding="same", use_bias=False, activation=tf.nn.relu,
                         name=base_name + "_3_1")(_3x3)
        _3x3 = conv2d_bn(256, [3, 1], padding="same", use_bias=False, activation=tf.nn.relu,
                         name=base_name + "_3_2")(_3x3)
        concat = tf.keras.layers.concatenate([_1x1, _3x3], axis=3, name=base_name + "_concat")
        return InceptionResnet.inception_resnet_block_tip(x, scale, concat, activation, base_name)

    @staticmethod
    def inception_resnet_block_tip(x, scale, concat, activation, base_name):
        B, H, W, C = x.shape
        up = conv2d_bn(C, 1, bn=False, name=base_name + "_up")(concat)
        out = tf.keras.layers.Lambda(lambda inp: inp[0] + inp[1] * inp[2], name=base_name + "_add_and_scale")(
            [x, up, scale])
        if activation:
            out = activation(out, name=base_name + "_out_act")
        return out
