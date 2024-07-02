import pandas as pd
import tensorflow as tf
import numpy as np
from tfkan import DenseKAN

def standardize(dataset: np.ndarray) -> tf.Tensor:
    for i in range(dataset.shape[1]):
        column = dataset[:, i]
        dataset[:, i] = (column - np.min(column)) / (np.max(column) - np.min(column))
    return dataset

def build_model(nodes: list, input_dim: int, loss: str='mse', optimizer: str='adam', metrics: list=['mse'], mlp: bool=True):
    model = tf.keras.Sequential()
    numero_livelli = len(nodes)

    if mlp:
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        for i in range(0, numero_livelli-1):
            model.add(tf.keras.layers.Dense(nodes[i], activation='relu'))
        model.add(tf.keras.layers.Dense(nodes[i+1], activation=None))
    else:
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        for i in range(0, numero_livelli):
            model.add(DenseKAN(nodes[i]))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model