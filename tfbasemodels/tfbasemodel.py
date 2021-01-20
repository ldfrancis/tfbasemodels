from abc import abstractmethod, abstractproperty
from .model import BaseModel
import tensorflow as tf
from .utils.file_utils import download_file, obtain_base_dir, validate_file


class TFBaseModel(BaseModel):
    """A wrapper for a tf.keras.Model instance
    """
    def __init__(self, pretrained=False):
        """builds the model architecture and creates a tf.keras.Model Instance
        """
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(3)
        self.model = self.build()

        # load model weights if pretrained
        if pretrained: self.load_pretrained()


    def build(self):
        """Must be implemented by subclass. The model architecture is built and
        a tf.keras.Model instance is returned.
        """
        raise NotImplementedError()

        
    @property
    def weights(self):
        """returns the weights and biases of a model as
        a list
        """
        return self.model.weights


    @property
    def trainable_variables(self):
        """returns the trainable parameters in self.model
        """
        return self.model.trainable_variables


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
            flops = \
                (tf.math.reduce_prod(kernel_shape)*\
                    tf.math.reduce_prod(output_shape)).numpy()
            total_flops += flops

        GFLOPs = f"{round(total_flops/1e9,2)}G flops"

        return GFLOPs


    def load_weights(self, filepath):
        """Loads the weights
        """
        self.model.load_weights(filepath)


    def save_weights(self, filepath):
        """Saves the weights
        """
        self.model.save_weights(filepath)

    
    def train(self):
        """Trains the model
        """ 


    def summary(self):
        """Displays/Prints model layers, output sizes and number of trainable 
        and non-trainable parameters
        """
        return self.model.summary()
        

    def load_pretrained(self):
        """Loads pretrained weights for the model to use
        """
        file_dir_name = self.__class__.__name__
        model_path = \
            obtain_base_dir()/(file_dir_name)/self.pretrained_weights_name

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


                


