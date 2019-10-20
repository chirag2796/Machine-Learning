import pandas as pd
import matplotlib.pyplot as plt

DATASET_FILEPATH = r"D:\Dev\Datasets\Structured\Housing_California\housing.csv"

def load_housing_data(dataset_filepath=DATASET_FILEPATH):
    return pd.read_csv(dataset_filepath)

def visualization():
    housing = load_housing_data()
    print(housing.head())
    print(housing.info())
    print(housing["ocean_proximity"]).value_counts()
    print(housing.describe())
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()

visualization()