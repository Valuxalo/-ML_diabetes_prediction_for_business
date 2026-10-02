from pathlib import Path

MODEL_NAME = "RandomForest"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
print(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATA_PATH = DATA_DIR / "diabetes_prediction_dataset.csv"
MODEL_PATH = ARTIFACTS_DIR / "diabetes_model.pkl"

RANDOM_STATE = 42
TARGET_COL = "diabetes"
