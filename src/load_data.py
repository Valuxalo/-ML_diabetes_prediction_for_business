import pandas as pd

from config import DATA_PATH


def load_data(load_path=None):
    df = pd.read_csv(DATA_PATH)
    return df
