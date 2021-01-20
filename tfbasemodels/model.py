from abc import ABC, abstractmethod, abstractproperty


class BaseModel(ABC):
    """A wrapper for a tf.keras.Model instance
    """
    def __init__(self, pretrained=False):
        """builds the model architecture and creates a Model Instance
        The Model instance is equivalent to torch.nn.Module instance in pytorch
        and tensorflow.keras.Model instance in tensorflow
        """
        self.model = self.build()

        
    @abstractproperty
    def weights(self):
        """returns the weights and biases of a model as
        a list
        """
        raise NotImplementedError()


    @abstractproperty
    def trainable_variables(self):
        """returns the trainable parameters in self.model
        """
        raise NotImplementedError()


    @abstractproperty
    def flops(self):
        """Number of multiply-adds in model
        """
        raise NotImplementedError()


    @abstractmethod
    def build(self):
        """Must be implemented by subclass. The model architecture is built and
        a Model instance is returned.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_weights(self, filepath):
        """Loads the weights
        """
        raise NotImplementedError()

    @abstractmethod
    def save_weights(self, filepath):
        """Saves the weights
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        """Trains the model
        """ 
        raise NotImplementedError()

    @abstractmethod
    def summary(self):
        """Displays/Prints model layers, output sizes and number of trainable 
        and non-trainable parameters
        """
        raise NotImplementedError()
        
    @abstractmethod
    def load_pretrained(self):
        """Loads pretrained weights for the model to use
        """
        raise NotImplementedError()


    def __call__(self, x):
        """Forward pass through the self.model

        Args:
            x: Tensor. Input tensor

        Returns:
            a Tensor
        """
        assert self.model != None
        return self.model(x)


                


