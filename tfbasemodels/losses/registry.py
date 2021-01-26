import tensorflow as tf

loss_registry = {
    "mse": tf.keras.losses.MSE,
    "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy,
    "binary_crossentropy": tf.keras.losses.BinaryCrossentropy
}


def add_loss(name, loss):
    global loss_registry
    assert loss_registry.get(name.lower()) is None, f"Loss {name} already exists"
    assert hasattr(loss, '__call__'), f"{name} is not callable"
    loss_registry[name.lower()] = loss


def get_loss(name):
    global loss_registry
    loss = loss_registry.get(name)
    return loss
