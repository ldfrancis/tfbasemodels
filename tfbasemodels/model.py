from abc import ABC, abstractmethod, abstractproperty
import tensorflow as tf
from .utils.file_utils import download_file, obtain_base_dir, validate_file


class TFBaseModel(ABC):
    """A wrapper for a tf.keras.Model instance
    """

    def __init__(self, pretrained=False):
        """builds the model architecture and creates a tf.keras.Model Instance
        """
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(3)
        self.model = self.build()

        # load model weights if pretrained
        if pretrained:
            self.load_pretrained()

    def __getattr__(self, name):
        return getattr(self.model, name)

    @abstractmethod
    def build(self):
        """Must be implemented by subclass. The model architecture is built and
        a tf.keras.Model instance is returned.
        """
        raise NotImplementedError()

    @property
    def flops(self):
        """Number of multiply-adds in model
        """
        total_flops = 0
        for layer in self.model.layers:
            try:
                kernel_shape = layer.kernel.shape[:-1]
            except:
                continue
            output_shape = layer.output.shape[1:]
            flops = (tf.math.reduce_prod(kernel_shape) *
                     tf.math.reduce_prod(output_shape)).numpy()
            total_flops += flops

        GFLOPs = f"{round(total_flops / 1e9, 2)}G flops"

        return GFLOPs

    def train(self):
        """Trains the model
        """
        raise NotImplementedError

    def load_pretrained(self):
        """Loads pretrained weights for the model to use
        """
        file_dir_name = self.__class__.__name__
        model_path = \
            obtain_base_dir() / file_dir_name / self.pretrained_weights_name

        # check if model file exists and is valid
        should_download = True
        if model_path.exists():
            should_download = not validate_file(model_path, self.filehash)

        if should_download:
            print("Downloading pretrained weights")
            download_file(self.pretrained_weights_url,
                          file_dir_name,
                          self.pretrained_weights_name)

        self.load_weights(model_path)
