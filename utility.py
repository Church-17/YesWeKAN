import pandas as pd
import numpy as np

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(df.shape[1]):
        column = df.iloc[:, i]
        df.iloc[:, i] = (column - np.min(column)) / (np.max(column) - np.min(column))
    return df
