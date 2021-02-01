import tensorflow as tf
from ..model import TFBaseModel


class MobileNetV2(TFBaseModel):
    input_shape = (224, 224, 3)
    alpha = 1

    def __init__(self, alpha=1, include_top=True, pooling='avg', pretrained=False):
        self.alpha = alpha
        self.include_top = include_top
        self.pooling = pooling
        self.pretrained = pretrained

    def build(self):
        x_input = tf.keras.layers.Input(shape=self.input_shape)

        # stem
        x = MobileNetV2.stem_network(x_input, self.alpha)

        # inv_res_bottleneck_block
        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=16, stride=1, expansion=1, alpha=self.alpha,
                                                           block_id=0)

        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=24, stride=2, expansion=6, alpha=self.alpha,
                                                           block_id=1)
        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=24, stride=1, expansion=6, alpha=self.alpha,
                                                           block_id=2)

        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=32, stride=2, expansion=6, alpha=self.alpha,
                                                           block_id=3)
        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=32, stride=1, expansion=6, alpha=self.alpha,
                                                           block_id=4)
        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=32, stride=1, expansion=6, alpha=self.alpha,
                                                           block_id=5)

        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=64, stride=2, expansion=6, alpha=self.alpha,
                                                           block_id=6)
        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=64, stride=1, expansion=6, alpha=self.alpha,
                                                           block_id=7)
        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=64, stride=1, expansion=6, alpha=self.alpha,
                                                           block_id=8)
        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=64, stride=1, expansion=6, alpha=self.alpha,
                                                           block_id=9)

        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=96, stride=1, expansion=6, alpha=self.alpha,
                                                           block_id=10)

        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=96, stride=1, expansion=6, alpha=self.alpha,
                                                           block_id=11)
        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=96, stride=1, expansion=6, alpha=self.alpha,
                                                           block_id=12)

        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=160, stride=2, expansion=6, alpha=self.alpha,
                                                           block_id=13)
        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=160, stride=1, expansion=6, alpha=self.alpha,
                                                           block_id=14)
        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=160, stride=1, expansion=6, alpha=self.alpha,
                                                           block_id=15)

        x = MobileNetV2.inverted_residual_bottleneck_block(x, filters=320, stride=1, expansion=6, alpha=self.alpha,
                                                           block_id=16)

        # last 1x1 conv
        filters = max(int(self.alpha * 1280), 1288)
        filters = ((filters // 8) + (filters % 8) // 4) * 8
        x = tf.keras.layers.Conv2D(filters, kernel_size=1, use_bias=False, name='final_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-3, momentum=0.999, name='final_bn')(x)
        x = tf.keras.layers.ReLU(6., name='final_relu')(x)

        if self.nclude_top:
            x = tf.keras.layers.GlobalAveragePooling2D(name="avgpool")(x)
            x = tf.keras.layers.Dense(self.num_classes, activation='softmax', use_bias=True, name='output')(x)
        else:
            if self.pooling == 'avg':
                x = tf.keras.layers.GlobalAveragePooling2D(name="avgpool")(x)
            elif self.pooling == 'max':
                x = tf.keras.layers.GlobalMaxPooling2D(name="maxpool")(x)

        # model
        model = tf.keras.Model(inputs=[x_input], outputs=x)

        return model

    @staticmethod
    def stem_network(x, alpha):
        filters = max(int(alpha * 32), 8)
        filters = ((filters // 8) + (filters % 8) // 4) * 8
        base_name = "stem/"
        x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=(2, 2), padding='valid', use_bias=False,
                                   name=base_name + "conv")(x)
        x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-3, momentum=0.999, name=base_name + "conv_bn")(x)
        x = tf.keras.layers.ReLU(6., name=base_name + "conv_relu")(x)

        return x

    @staticmethod
    def inverted_residual_bottleneck_block(x, filters, stride, expansion, alpha, block_id):
        B, H, W, C = x.shape
        input_ = x

        base_name = f"inv_res_bottleneck_block_{block_id}/"

        # ensure num filters is divisible by 8
        filters = int(filters * alpha)
        filters = ((filters // 8) + (filters % 8) // 4) * 8

        # expand
        if expansion > 1:
            x = tf.keras.layers.Conv2D(expansion * C, kernel_size=1, padding='same', use_bias=False, activation=None,
                                       name=base_name + 'expand_conv')(x)
            x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-3, momentum=0.999, name=base_name + 'expand_bn')(
                x)
            x = tf.keras.layers.ReLU(6., name=base_name + 'expand_relu')(x)

        # depthwise conv
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, use_bias=False, padding='same',
                                            name=base_name + 'dw_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-3, momentum=0.999, name=base_name + 'dw_bn')(x)
        x = tf.keras.layers.ReLU(6., name=base_name + 'dw_relu')(x)

        # pointwise conv
        x = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False, activation=None,
                                   name=base_name + 'pw_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-3, momentum=0.999,
                                               name=base_name + 'pw_bn')(x)

        # add - shortcut
        if C == filters and stride == 1:
            x = tf.keras.layers.Add(name=base_name + 'add')([input_, x])

        return x
