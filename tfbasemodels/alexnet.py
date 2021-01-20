import tensorflow as tf
from ..model import TFBaseModel

class AlexNet(TFBaseModel):
    """AlexNet architecture
    input size: 227x227x3
    """
    def __init__(self):
        super().__init__()


    def build(self):
        # inptut layer
        x_inp = tf.keras.layers.Input(shape=[227,227,3])

        # input size:   227x227x3
        # output size:  55x55x96
        # conv1
        x = tf.keras.layers.Conv2D(filters=96,
                                    kernel_size=11,
                                    strides=4,
                                    padding='valid',
                                    activation=tf.nn.relu,
                                    name="conv1")(x_inp)

        # input size:   55x55x96
        # output size:  27x27x96
        # pool1
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                    strides=2,
                                    padding=0,
                                    name="pool1")(x)

        # input size:   27x27x96
        # output size:  27x27x256
        # conv2
        x = tf.keras.layers.Conv2D(filters=256,
                                    kernel_size=5,
                                    padding='same',
                                    strides=1,
                                    name="conv2")(x)

        # input size:   27x27x256
        # output size:  13x13x256
        # pool2
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                    strides=2,
                                    name="pool2")(x)

        # input size:   13x13x256
        # output size:  13x13x384
        # conv3
        x = tf.keras.layers.Conv2D(filters=384,
                                    kernel_size=3,
                                    strides=1,
                                    padding='same',
                                    name="conv3")(x)
        # input size: 13x13x384
        # output size: 13x13x384
        # conv4
        x = tf.keras.layers.Conv2D(filters=384,
                                    kernel_size=3,
                                    padding='same',
                                    strides=1,
                                    name="conv4")(x)
        
        # input size: 13x13x384
        # output size: 13x13x256
        # conv5
        x = tf.keras.layers.Conv2D(filters=256,
                                    kernel_size=3,
                                    padding='same',
                                    strides=1,
                                    name="conv5")(x)
        
        # input size: 13x13x256
        # output size: 6x6x256
        # pool5
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                    strides=2,
                                    name="pool5")(x)

        # input size: 6x6x256
        # output size: 6*6*256
        x = tf.keras.layers.Flatten()(x)

        # input size: 6*6*256
        # output size: 4096
        x = tf.keras.layers.Dense(units=4096, name="fc6")(x)

        # input size: 4096
        # output size: 4096
        x = tf.keras.layers.Dense(units=4096, name="fc7")(x)

        # input size: 4096
        # output size: 1000
        logits = tf.keras.layers.Dense(units=1000, name="fc8")(x)

        return tf.keras.Model(inputs=[inp_x], outputs=[logits])



                            
                            