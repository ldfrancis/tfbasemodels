from abc import ABC, abstractmethod, abstractproperty
import tensorflow as tf
from tfbasemodels.utils.file_utils import download_file, obtain_base_dir, validate_file
from ..losses.registry import get_loss
from ..trainers.optimizers import get_optimizer


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
        self.loss_fn = None
        self.train_step_losses = []

        # load model weights if pretrained
        if pretrained:
            self.load_pretrained()

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

    def train(self, datasource, epochs, lr, optim="", loss=""):
        """Trains the model
        """
        raise NotImplementedError # work in progress
        # create the dataloader from the datasource
        dataloader = datasource
        self.train_step_losses = []
        self.optimizer = get_optimizer(optim) if optim else get_optimizer(self.optimizer)
        self.loss = get_loss(loss) if loss else get_loss(self.loss)
        for epoch in range(epochs):
            dataloader.shuffle()
            for inp, out in dataloader:
                self.train_step(inp, out)

    def train_step(self, inp, out):
        with tf.GradientTape() as tape:
            pred = self(inp, training=True)
            loss = self.loss_fn(out, pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_step_losses += [loss.numpy()]

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
