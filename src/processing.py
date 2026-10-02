from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from config import RANDOM_STATE, TARGET_COL


def processor():
    multi_category = ["gender", "smoking_history"]

    processor = ColumnTransformer(
        transformers=[("category", OrdinalEncoder(), multi_category)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return processor


def split_data(data):
    y = data[TARGET_COL]
    X = data.drop(columns=TARGET_COL)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test
