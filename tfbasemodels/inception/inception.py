import tensorflow as tf
from ..model import Model
from ..utils.layer_utils import conv2d_bn


class GoogLeNet(Model):
    """GoogLeNet. input size: 224x224x3
    """
    def __init__(self):
        super().__init__()


    def build(self):
        # input layer
        x_inp = tf.keras.layers.Input(shape=[224,224,3])

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
                                _pool_proj=64,
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
        """Stem network. Used to aggresively reduce the spatial dimensions
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
                        padding=3,
                        activation=tf.nn.relu,
                        name=base_name+"_conv1")(x)
        # input: 112x112x64
        # output: 56x56x64
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                        strides=2,
                                        padding=1,
                                        name=base_name+"_maxpool1")(x)
        # input: 56x56x64
        # output: 56x56x64
        x = conv2d_bn(filters=64,
                        kernel_size=1,
                        strides=1,
                        padding=0,
                        activation=tf.nn.relu,
                        name=base_name+"_conv2a")(x)
        # input: 56x56x64
        # output: 56x56x192
        x = conv2d_bn(filters=192,
                        kernel_size=3,
                        strides=1,
                        padding=1,
                        activation=tf.nn.relu,
                        name=base_name+"_conv2b")(x)
        # input: 56x56x192
        # output: 28x28x192
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                        strides=2,
                                        padding=1,
                                        name=base_name+"_maxpool2")(x)

        return x



    def inception_module(self, x, _1x1, _3x3_reduce, _3x3, _5x5_reduce, _5x5, pool_proj, name):
        """Inception module

        Args:
            x; Tensor. Input tensor
            _1x1: int. Number of filters for the 1x1 conv
            _3x3_reduce: int. Number of filters for the 1x1 conv before the 3x3 conv
            _3x3: int. Number of filters for the 3x3 conv after a 1x1 conv
            _5x5_reduce: int. Number of filters for 1x1 conv before 5x5 conv
            _5x5: int. Number of filters for the 5x5 conv after a 1x1 conv

        Returns: 
            a Tensor
        """
        base_name = "inception_"+name

        conv1x1 = conv2d_bn(filters=_1x1,
                            kernel_size=1,
                            strides=1,
                            padding="same",
                            activation=tf.nn.relu,
                            name=base_name+"_1x1")(x)

        conv3x3_reduce = conv2d_bn(filters=_3x3_reduce,
                            kernel_size=1,
                            strides=1,
                            padding="same",
                            activation=tf.nn.relu,
                            name=base_name+"_3x3_reduce")(x)

        conv3x3 = conv2d_bn(filters=_3x3,
                            kernel_size=3,
                            strides=1,
                            padding="same",
                            activation=tf.nn.relu,
                            name=base_name+"_3x3")(conv3x3_reduce)

        conv5x5_reduce = conv2d_bn(filters=_5x5_reduce,
                            kernel_size=1,
                            strides=1,
                            padding="same",
                            activation=tf.nn.relu,
                            name=base_name+"_5x5_reduce")(x)

        conv5x5 = conv2d_bn(filters=_5x5,
                            kernel_size=5,
                            strides=1,
                            padding="same",
                            activation=tf.nn.relu,
                            name=base_name+"_5x5")(conv5x5_reduce)

        max3x3 = tf.keras.layers.MaxPool2D(pool_size=3,
                                            strides=1,
                                            padding="same",
                                            name=base_name+"max3x3")(x)

        maxpool_proj = conv2d_bn(filters=_pool_proj,
                            kernel_size=1,
                            strides=1,
                            padding="same",
                            activation=tf.nn.relu,
                            name=base_name+"_pool_proj")(max_3x3)

        concat = tf.keras.layers.Concatenate(name=base_name+"_concat",
                                            )([conv1x1, conv3x3, conv5x5, maxpool_proj])

        return concat



        
        
        

