# isi file python dimulai dari sini

import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor, numerical_cols, categorical_cols


def preprocess(input_path: str, target_col: str, output_dir: str,
               test_size: float = 0.2, random_state: int = 42, stratify: bool = True) -> None:
    df = pd.read_csv(input_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    for col in ["Saving accounts", "Checking account"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    df = df.drop_duplicates()

    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' tidak ditemukan. Kolom tersedia: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    y = y_raw.map({"good": 0, "bad": 1})
    if y.isna().any():
        raise ValueError(
            "Ada label target yang tidak dikenali. Pastikan target hanya 'good'/'bad'. "
            f"Label unik: {sorted(y_raw.unique().tolist())}"
        )

    stratify_y = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    preprocessor, numerical_cols, categorical_cols = build_preprocessor(X_train)

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    os.makedirs(output_dir, exist_ok=True)

    # Simpan output preprocessing (dense) sebagai .npy
    np.save(os.path.join(output_dir, "X_train.npy"), X_train_prep)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test_prep)

    pd.Series(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    pd.Series(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

    meta = {
        "input_path": input_path,
        "target_col": target_col,
        "rows": int(df.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "n_features_after_onehot": int(X_train_prep.shape[1]),
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "test_size": test_size,
        "random_state": random_state,
        "stratify": stratify,
    }

    with open(os.path.join(output_dir, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Preprocessing selesai.")
    print("Saved to:", output_dir)
    print("X_train_prep:", X_train_prep.shape, "| X_test_prep:", X_test_prep.shape)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path file CSV raw")
    parser.add_argument("--target", default="Risk", help="Nama kolom target (default: Risk)")
    parser.add_argument("--output", required=True, help="Folder output preprocessing")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-stratify", action="store_true")
    args = parser.parse_args()

    preprocess(
        input_path=args.input,
        target_col=args.target,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=(not args.no_stratify),
    )


if __name__ == "__main__":
    main()
