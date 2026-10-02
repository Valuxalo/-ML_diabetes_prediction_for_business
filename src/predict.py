import pandas as pd
import os
from config import ARTIFACTS_DIR

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)

def predict(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:,1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)

    class_report = classification_report(y, y_pred)

    rf = model.named_steps["model"]
    feature_names = model.named_steps["preprocessing"].get_feature_names_out()
    feature_importance = rf.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    with open(f'{ARTIFACTS_DIR}/score.txt', 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1-score: {f1}\n")
        f.write(f"ROC-AUC-score: {roc_auc}\n")

    with open(f'{ARTIFACTS_DIR}/class_report.txt', 'w', encoding='utf-8') as f:
        f.write(class_report)

    with open(f'{ARTIFACTS_DIR}/feature_importance.txt', 'w', encoding='utf-8') as f:
        f.write("Топ-10 важных признаков:\n")
        f.write(importance_df.head(10).to_string(index=False))