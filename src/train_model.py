"""
Trains and compares several classifiers on the credit dataset, then saves
the best-performing model (by ROC-AUC) plus the scaler and feature schema
needed for consistent inference.

Run:
    python src/train_model.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import ALL_NUMERIC, TARGET, build_feature_frame

DATA_PATH = "data/credit_data.csv"
MODEL_DIR = "models"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    feat_df = build_feature_frame(df.drop(columns=[TARGET]))
    y = df[TARGET].values
    feature_columns = list(feat_df.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        feat_df, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric columns only (tree models don't need it, but LR does;
    # scaling everything with 0/1 dummy columns is harmless).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=15,
            random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42
        ),
    }

    results = {}
    trained_models = {}

    for name, model in candidates.items():
        if name == "logistic_regression":
            model.fit(X_train_scaled, y_train)
            proba = model.predict_proba(X_test_scaled)[:, 1]
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)

        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, preds)
        results[name] = {"roc_auc": auc, "accuracy": acc}
        trained_models[name] = model

        print(f"\n=== {name} ===")
        print(f"ROC-AUC:  {auc:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=["good", "default"]))

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_name]
    print(f"\n>>> Best model: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.4f})")

    cm = confusion_matrix(
        y_test,
        best_model.predict(X_test_scaled if best_name == "logistic_regression" else X_test),
    )
    print("Confusion matrix (rows=actual, cols=predicted) [good, default]:")
    print(cm)

    # Feature importance (works for tree models directly; for logistic
    # regression we use absolute coefficient magnitude as a proxy).
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])
    importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": importances}
    ).sort_values("importance", ascending=False)

    print("\nTop 10 most important features:")
    print(importance_df.head(10).to_string(index=False))

    # ---- Persist everything needed for inference ----
    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    metadata = {
        "model_name": best_name,
        "needs_scaling": best_name == "logistic_regression",
        "feature_columns": feature_columns,
        "metrics": results,
        "feature_importance": importance_df.to_dict(orient="records"),
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model, scaler, and metadata to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
