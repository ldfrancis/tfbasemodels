import tensorflow as tf
from ..model import TFBaseModel


class Xception(TFBaseModel):
    input_shape = (299, 299, 3)

    def __init__(self):
        super().__init__()

    def build(self):
        # input layer
        x_input = tf.keras.layers.Input(shape=self.input_shape)

        # entry_flow
        x = Xception.entry_flow(x_input)

        # middle_flow
        x = Xception.middle_flow(x)

        # exit_flow
        x = Xception.exit_flow(x)

        # activation
        x = tf.keras.layers.Activation("relu", name="final_relu")(x)

        if self.include_top:
            x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
            x = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        else:
            if self.pooling == 'avg':
                x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
            elif self.pooling == 'max':
                x = tf.keras.layers.GlobalMaxPooling2D(name='maxpool')(x)
        model = tf.keras.Model(inputs=[x_input], outputs=[x])
        return model

    @staticmethod
    def stem_network(x, name=None):
        base_name = f"stem/" if name is None else f"stem_{name}/"
        x = tf.keras.layers.Conv2D(32, 3, 2, use_bias=False, name=base_name + "conv1")(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=base_name + "conv1_bn")(x)
        x = tf.keras.layers.Activation("relu", name=base_name + "conv1_relu")(x)
        x = tf.keras.layers.Conv2D(64, 3, use_bias=False, name=base_name + "conv2")(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=base_name + "conv2_bn")(x)
        x = tf.keras.layers.Activation("relu", name=base_name + "conv2_relu")(x)
        return x

    @staticmethod
    def xception_block(x, filters=(32,32), kernels=(3,3), shortcut=None, preact=False, name=None,):
        """Creates an xception block consisting of 2 depthwise seperable convolutions and a shortcut which can be
        either an identity, a projection or None. The filters and kernels for the seperable convolutions are specified
        as tuples in the params 'filters', and 'kernels' the preact parameters specifies whether the block should begin
        with an activation or not"""
        base_name = "xception/" if name is None else f"{name}/xception/"
        shortcut_ = None
        if shortcut == "identity":
            shortcut_ = tf.keras.layers.Layer(name=base_name+"shortcut")(x)
        elif shortcut == "projection":
            shortcut_ = tf.keras.layers.Conv2D(filters[-1], 1, 2, padding="same", use_bias=False,
                                               name=base_name+"shortcut")
        if preact:
            x = tf.keras.layers.Activation("relu", name=base_name+"preact_relu")
        for i,f,k in enumerate(zip(filters, kernels)):
            x = tf.keras.layers.SeparableConv2D(f, k, padding="same", use_bias=False,
                                                name=base_name+f"sepconv{i+1}")(x)
            x = tf.keras.layers.BatchNormalization(axis=3, name=base_name+f"sepconv{i+1}_bn")(x)
            x = tf.keras.layers.Activation("relu", name=base_name+f"sepconv{i+1}_relu")(x) if i+1 != len(filters) else x
        x = tf.keras.layers.MaxPooling2D(3, 2, padding="same", name=base_name+"pool")(x) if shortcut_ else x
        x = tf.keras.layers.Add(name=base_name+"add")([x, shortcut_]) if shortcut_ else x
        return x

    @staticmethod
    def entry_flow(x, name=None):
        base_name = f"entry_flow/" if name is None else f"entry_flow_{name}/"
        x = Xception.stem_network(x)
        x = Xception.xception_block(x, (128,128), (3,3), shortcut="projection", name=base_name+"1")(x)
        x = Xception.xception_block(x, (256, 256), (3, 3), shortcut="projection",preact=True, name=base_name + "2")(x)
        x = Xception.xception_block(x, (728,728), (3,3), shortcut="projection", preact=True, name=base_name+"3")(x)
        return x

    @staticmethod
    def middle_flow(x, name=None):
        base_name = f"middle_flow/" if name is None else f"middle_flow{name}/"
        for i in range(8):
            x = Xception.xception_block(x, (728,728, 728), (3,3), shortcut="identity", preact=True,
                                        name=base_name+str(i+1))(x)
        return x

    @staticmethod
    def exit_flow(x, name=None):
        base_name = f"exit_flow/" if name is None else f"exit_flow{name}/"
        x = Xception.xception_block(x, (728,1024), (3,3), shortcut="projection", preact=True, name=base_name+"1")(x)
        x = Xception.xception_block(x, (1536,2048), (3,3), shortcut="projection", preact=False, name=base_name+"2")(x)
        return x







