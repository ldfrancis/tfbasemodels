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

    def train(self, datasource, epochs, lr, optim="", loss="", validation=None, batch_size=32):
        """Trains the model
        """
        # perform validation split if available and create the dataloader from the datasource
        if isinstance(validation, int):
            datasource.validation_split(validation)
            dataloader = DataLoader(datasource.train_source, batch_size)
            val_loader = DataLoader(datasource.val_source, batch_size)
        elif isinstance(validation, DataSource):
            dataloader = DataLoader(datasource, batch_size)
            val_loader = DataLoader(validation, batch_size)
        else:
            dataloader = DataLoader(datasource, batch_size)

        self._train_step_losses = []
        self._eval_losses = []
        self._step_epoch_map = {}
        self.optimizer = get_optimizer(optim) if optim else get_optimizer(self.optimizer)
        self.loss = get_loss(loss) if loss else get_loss(self.loss)
        self._epoch = 0
        self._train_step = 0
        for epoch in range(epochs):
            self._epoch = epoch+1
            dataloader.shuffle()
            print(f"epoch {self.epoch}: train_loss: {0:4f}", end="")
            for inp, out in dataloader:
                train_info = self.train_step(inp, out)
                self._train_step_losses += [train_info["loss"]]
            self.step_epoch_map[self.step] = self.epoch
            if validation:
                val_loss = tf.keras.metrics.Mean
                for inp,out in val_loader:
                    eval_info = self.eval_step(inp, out)
                    val_loss.update_state(eval_info["loss"])
                print(f"\repoch {self.epoch}: train_loss: {float(loss.numpy()):4f} "
                      f"val_loss: {float(val_loss.result().numpy()):4f}")
                self._eval_losses += [val_loss.result().numpy()]
            else:
                print("\n")
            self._step_epoch_map[self._train_step] = epoch

    def train_step(self, inp, out):
        with tf.GradientTape() as tape:
            pred = self(inp, training=True)
            loss = self.loss(out, pred)
            print(f"\repoch {self.epoch}: train_loss: {float(loss.numpy()):4f}", end="")
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        train_info = {"loss": loss.numpy()}
        self._train_step += 1
        return train_info

    def eval_step(self, inp, out):
        pred = self(inp)
        loss = self.loss(out, pred)
        eval_info = {"loss":loss.numpy()}
        return eval_info

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
