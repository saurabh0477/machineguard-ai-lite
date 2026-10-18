"""
MachineGuard AI Lite - Model Training Pipeline
Dataset: UCI AI4I 2020 Predictive Maintenance Dataset
URL: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
"""

import os
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH = os.path.join(DATA_DIR, "ai4i2020.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")

# Telemetry features and target
TARGET_COL = "Machine failure"
IDENTIFIER_COLS = ["UDI", "Product ID"]
LEAKAGE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
RAW_FEATURE_COLS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def ensure_dataset(data_path=CSV_PATH) -> pd.DataFrame:
    """Ensure dataset exists locally, downloading from UCI repository if needed."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(data_path):
        print(f"Loading existing dataset from: {data_path}")
        return pd.read_csv(data_path)

    print(f"Downloading UCI AI4I 2020 dataset from: {UCI_ZIP_URL}")
    headers = {"User-Agent": "Mozilla/5.0 (MachineGuard-AI-Lite-Downloader)"}
    req = urllib.request.Request(UCI_ZIP_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        csv_filename = [f for f in z.namelist() if f.endswith(".csv")][0]
        z.extract(csv_filename, DATA_DIR)
        extracted_path = os.path.join(DATA_DIR, csv_filename)
        if extracted_path != data_path:
            os.replace(extracted_path, data_path)

    print(f"Dataset successfully saved to: {data_path}")
    return pd.read_csv(data_path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute domain-specific physical telemetry features:
    - Temp difference [K] = Process temperature [K] - Air temperature [K]
    - Power [W] = Torque [Nm] * Rotational speed [rad/s]
    """
    df = df.copy()
    # Physical thermal dissipation gradient
    df["Temp difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    # Mechanical rotational power (Watts)
    df["Power [W]"] = df["Torque [Nm]"] * (
        df["Rotational speed [rpm]"] * (2 * np.pi / 60.0)
    )
    return df


def train_and_evaluate(df: pd.DataFrame, save_artifacts: bool = True):
    """Clean data, train Random Forest classifier, evaluate on held-out test split,
    and persist trained pipeline with real performance metrics.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Validate required columns
    required_cols = RAW_FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    print(f"Total records in dataset: {len(df)}")
    fail_count = int(df[TARGET_COL].sum())
    fail_rate = (fail_count / len(df)) * 100.0
    print(f"Historical failure labels: {fail_count} ({fail_rate:.2f}%)")

    # 2. Exclude identifiers and target-leaking failure modes
    clean_df = engineer_features(df)
    
    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temp difference [K]",
        "Power [W]",
    ]
    categorical_features = ["Type"]
    all_features = categorical_features + numeric_features

    X = clean_df[all_features]
    y = clean_df[TARGET_COL].astype(int)

    # 3. Train/Test split (80/20 stratified split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Pipeline with Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(categories=[["L", "M", "H"]], handle_unknown="ignore"),
                categorical_features,
            ),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", clf)]
    )

    print("Fitting Random Forest classifier on training set...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    # Extract feature importances
    ohe_cols = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )
    feature_names = ohe_cols + numeric_features
    importances = pipeline.named_steps["classifier"].feature_importances_.tolist()
    feature_importance_dict = dict(
        sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    )

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "test_size": len(y_test),
        "test_failures": int(y_test.sum()),
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importances": feature_importance_dict,
        "feature_names": all_features,
        "transformed_feature_names": feature_names,
    }

    print("\n--- Test Set Evaluation Results ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Confusion Matrix (TN, FP / FN, TP):\n{np.array(cm)}")

    if save_artifacts:
        artifact = {
            "pipeline": pipeline,
            "metrics": metrics,
            "raw_feature_cols": RAW_FEATURE_COLS,
            "engineered_feature_cols": ["Temp difference [K]", "Power [W]"],
            "all_feature_cols": all_features,
            "target_col": TARGET_COL,
        }
        joblib.dump(artifact, MODEL_PATH)
        print(f"\nModel and evaluation metrics saved to: {MODEL_PATH}")

    return pipeline, metrics


if __name__ == "__main__":
    df = ensure_dataset()
    train_and_evaluate(df)
