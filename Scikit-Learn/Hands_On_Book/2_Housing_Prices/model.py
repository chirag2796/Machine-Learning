import numpy as np
import pandas as pd
from zlib import crc32

DATASET_FILEPATH = r"D:\Dev\Datasets\Structured\Housing_California\housing.csv"

def load_housing_data(dataset_filepath=DATASET_FILEPATH):
    return pd.read_csv(dataset_filepath)

def split_train_set(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]



data = load_housing_data()
train_set, test_set = split_train_set(data, 0.2)
