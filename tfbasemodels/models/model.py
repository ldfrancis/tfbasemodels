from abc import ABC, abstractmethod, abstractproperty
import tensorflow as tf
from tfbasemodels.utils.file_utils import download_file, obtain_base_dir, validate_file
from ..losses.registry import get_loss
from ..trainers.optimizers import get_optimizer
from tfdata import DataLoader, DataSource


class TFBaseModel(tf.keras.Model):
    """A wrapper for a tf.keras.Model instance
    """

    def __init__(self, pretrained=False):
        """builds the model architecture and creates a tf.keras.Model Instance
        """
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(3)

        # build model
        inputs, outputs = self.build()
        super().__init__(inputs=[inputs], outputs=[outputs], name=self.model_name)

        self.optimizer = 'adam'
        self.loss = None


        # load model weights if pretrained
        if pretrained:
            self.load_pretrained()

        # initialize some attributes
        self._eval_losses = None
        self._step_epoch_map = None
        self._train_step = None
        self._train_step_losses = None
        self._epoch = None

    @abstractmethod
    def build(self):
        """Must be implemented by subclass. The model architecture is built and
        an input tf.Tensor and output tf.Tensor instance is returned.
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

    def train(self, datasource, epochs, lr, optim="", loss="", metrics=[],validation=None, batch_size=32):
        """Trains the model
        """
        self.compile(optimizer=optim, loss=loss, metircs=metrics)

        # perform validation split if available and create the dataloader from the datasource
        if isinstance(validation, int):
            datasource.validation_split(validation)
            dataloader = DataLoader(datasource, batch_size)
            val_loader = DataLoader(datasource.validation_data, batch_size)
            self.fit(dataloader, batch_size=None, epochs=epochs, validation_data=val_loader)
        elif isinstance(validation, DataSource):
            dataloader = DataLoader(datasource, batch_size)
            val_loader = DataLoader(validation, batch_size)
            self.fit(dataloader, batch_size=None, epochs=epochs, validation_data=val_loader)
        else:
            dataloader = DataLoader(datasource, batch_size)
            self.fit(dataloader, batch_size=None, epochs=epochs)

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
