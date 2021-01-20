from abc import abstractmethod, abstractproperty
from .model import BaseModel
import tensorflow as tf
from tfbasemodels.utils.file_utils import download_file, obtain_base_dir, validate_file


class TFBaseModel(BaseModel):
    """A wrapper for a tf.keras.Model instance
    """
    def __init__(self, pretrained=False):
        """builds the model architecture and creates a tf.keras.Model Instance
        """
        self.model = self.build()

        # load model weights if pretrained
        if pretrained: self.load_pretrained()


    def build(self):
        """Must be implemented by subclass. The model architecture is built and
        a torch.nn.Module instance is returned.
        """
        raise NotImplementedError()

        
    @property
    def weights(self):
        """returns the weights and biases of a model as
        a list
        """
        raise NotImplementedError()


    @property
    def trainable_variables(self):
        """returns the trainable parameters in self.model
        """
        raise NotImplementedError()


    @property
    def flops(self):
        """Number of multiply-adds in model
        """
        raise NotImplementedError()


    def load_weights(self, filepath):
        """Loads the weights
        """
        raise NotImplementedError()


    def save_weights(self, filepath):
        """Saves the weights
        """
        raise NotImplementedError()

    
    def train(self):
        """Trains the model
        """ 
        raise NotImplementedError()


    def summary(self):
        """Displays/Prints model layers, output sizes and number of trainable 
        and non-trainable parameters
        """
        raise NotImplementedError()
        

    def load_pretrained(self):
        """Loads pretrained weights for the model to use
        """
        raise NotImplementedError()


                


