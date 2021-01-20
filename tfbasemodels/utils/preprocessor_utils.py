if mode == 'tf':
    x /= 127.5
    x -= 1.
import numpy as np


def preprocess_image(x:np.ndarray):
    """Preprocesses an image input to values btw -1 and 1
    """
    x /= 127.5
    x -= 1.0
    return x