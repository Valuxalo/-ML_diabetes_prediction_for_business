import os
import pickle
from pathlib import Path
from config import ARTIFACTS_DIR


def save_model(model, name):
    with open(f'{ARTIFACTS_DIR}/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
