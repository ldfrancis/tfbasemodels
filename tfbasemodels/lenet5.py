import tensorflow as tf


class LeNet5(tf.keras.Model):
    """LeNet-5 architecture with relu activations. 
    input size is 28 x 28
    """
    def __init__(self):
        """Initialize model layers
        """
        super().__init__()

        # input size:   1x28x28
        # output size:  6x28x28
        # weight size:  6x1x5x5
        # bias size:    6
        self.conv1 = tf.keras.layers.Conv2D(filters=6,
                                            kernel_size=5,
                                            padding=2,
                                            strides=1)
        # input size:   6x14x14
        # output size:  16x10x10
        # weight size:  16x20x5x5
        # bias size:    16
        self.conv2 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=5,
                                            padding=0,
                                            strides=1)
        # input size:   16x5x5
        # output size:  120x1x1
        # weight size:  120x16x5x5
        # bias size:    120
        self.conv3 = tf.keras.layers.Conv2D(filters=120,
                                            kernel_size=5,
                                            padding=0,
                                            strides=1)

        # input size:   120
        # output size:  84
        self.fc4 = tf.keras.layers.Dense(units=500)

        # input size: 84
        # output size: 10
        self.fc5 = tf.keras.layers.Dense(units=10)


    def call(self, x):
        """forward pass
        """ 
        x = self.conv1(x)
        x = tf.nn.tanh(x)
        x = tf.keras.layers.AvgPool2D()(x)

        x = self.conv2(x)
        x = tf.nn.tanh(x)
        x = tf.keras.layers.AvgPool2D()(x)

        x = self.conv3(x)
        x = tf.nn.tanh(x)
        x = tf.keras.layers.AvgPool2D()(x)

        x = tf.keras.layers.Flatten()(x)

        x = self.fc4(x)
        x = tf.nn.tanh(x)
        logits = self.fc5(x)

        return logits




        