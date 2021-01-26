import tensorflow as tf

optim_registry = {
    "adam": tf.keras.optimizers.Adam,
    "sgd": tf.keras.optimizers.SGD,
}


def add_optimizer(name, optimizer):
    global optim_registry
    assert optim_registry.get(name.lower()) is None, f"optimizer with name {name} already exists"
    assert hasattr(optimizer, '__call__'), f"{optimizer} is not callable"
    optim_registry[name.lower()] = optimizer


def get_optimizer(name):
    global optim_registry
    optimizer = optim_registry.get(name)
    return optimizer
