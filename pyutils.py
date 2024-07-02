import pandas as pd
import tensorflow as tf
import numpy as np
import datetime
from tfkan import DenseKAN

def dataframe_to_ndarray(dataframe: pd.DataFrame) -> tf.Tensor:
    dates = dataframe["vdate"]     # Date originali
    encoded_dates = [datetime.datetime.strptime(str(date), "%m/%d/%Y").timetuple().tm_yday for date in dates] # Date codificate 
    encoding_dict = dict(zip(dates, encoded_dates))     # Creazione dizionario di codifica per le date
    dataframe.replace({"vdate": encoding_dict}, inplace=True)  # Sostituzione nella colonna

    dates = dataframe["discharged"]
    encoded_dates = [datetime.datetime.strptime(str(date), "%m/%d/%Y").timetuple().tm_yday for date in dates]
    encoding_dict = dict(zip(dates, encoded_dates))
    dataframe.replace({"discharged": encoding_dict}, inplace=True)

    dataframe.replace({"rcount": dict(zip(sorted(dataframe["rcount"].unique()), range(6)))}, inplace=True)   # Codifica del 5+ in 5 e conversione a interi

    dataframe.replace({"gender": "M"}, 0, inplace=True)    # Codifica binaria del genere, assegna 0 a maschio e 1 a femmina
    dataframe.replace({"gender": "F"}, 1, inplace=True)

    keys = dataframe["facid"].unique()                        # Estrazione dei valori unici dell'attributo
    ints = {key: value for value, key in enumerate(keys)}    # Creazione del dizionario di codifica

    dataframe.replace({"facid": ints}, inplace=True)                # Codifica del facid

    np_array = dataframe.to_numpy()

    return np_array

def standardize(dataset: np.ndarray) -> tf.Tensor:
    for i in range(dataset.shape[1]):
        column = dataset[:, i]
        dataset[:, i] = (column - np.min(column)) / (np.max(column) - np.min(column))
    return dataset

def build_model(nodes: list, input_dim: int, loss: str='mse', optimizer: str='adam', metrics: list=['accuracy'], mlp: bool=True):
    model = tf.keras.Sequential()
    numero_livelli = len(nodes)

    if mlp:
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        for i in range(0, numero_livelli):
            model.add(tf.keras.layers.Dense(nodes[i], activation='relu'))
    else:
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        for i in range(0, numero_livelli):
            model.add(DenseKAN(nodes[i]))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model