from .dense import DenseKAN

import tensorflow as tf

custom_objects = {
    'DenseKAN': DenseKAN
}
tf.keras.utils.get_custom_objects().update(custom_objects)

__all__ = ['DenseKAN']