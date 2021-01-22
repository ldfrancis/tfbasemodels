import tensorflow as tf
from .model import TFBaseModel

class ZFNet(TFBaseModel):
    """ZFNet architecture
    input size: 224x224x3
    """
    def __init__(self):
        super().__init__()


    def build(self):
        # input layer
        x_inp = tf.keras.layers.Input(shape=[224,224,3])

        # input size:   224x224x3
        # output size:  112x112x96
        x = tf.keras.layers.Conv2D(filters=96,
                                    kernel_size=7,
                                    strides=2,
                                    padding='same',
                                    activation=tf.nn.relu,
                                    name="conv1")(x_inp)

        # input size:   112x112x96
        # output size:  55x55x96
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                    strides=2,
                                    name="pool1")(x)

        # input size:   55x55x96
        # output size:  26x26x256
        x = tf.keras.layers.Conv2D(filters=256,
                                    kernel_size=5,
                                    padding='valid',
                                    strides=2,
                                    activation=tf.nn.relu,
                                    name="conv2")(x)

        # input size:   26x26x256
        # output size:  13x13x256
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                    strides=2,
                                    padding='same',
                                    name="pool2")(x)

        # input size:   13x13x256
        # output size:  13x13x384
        x = tf.keras.layers.Conv2D(filters=384,
                                    kernel_size=3,
                                    strides=1,
                                    padding='same',
                                    activation=tf.nn.relu,
                                    name="conv3")(x)
        # input size: 13x13x384
        # output size: 13x13x384
        x = tf.keras.layers.Conv2D(filters=384,
                                    kernel_size=3,
                                    padding=1,
                                    strides='same',
                                    activation=tf.nn.relu,
                                    name="conv4")(x)
        
        # input size: 13x13x384
        # output size: 13x13x256
        x = tf.keras.layers.Conv2D(filters=256,
                                    kernel_size=3,
                                    padding=1,
                                    strides='same',
                                    activation=tf.nn.relu,
                                    name="conv5")(x)

        # input size: 13x13x256
        # output size: 6x6x256
        x = tf.keras.layers.MaxPool2D(pool_size=3,
                                    strides=2,
                                    padding='valid',
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

        # tf.keras Model
        return tf.keras.Model(inputs=[x_inp], outputs=[logits])




                            
                            