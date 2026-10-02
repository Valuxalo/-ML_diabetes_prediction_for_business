from config import DATA_PATH
import pandas as pd

def load_data(load_path=None):
    df = pd.read_csv(DATA_PATH)
    return df