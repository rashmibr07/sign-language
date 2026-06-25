"""
Step 2 -- Train the classifier on the landmarks you collected.

Reads data/dataset.csv, trains a multi-layer perceptron (a neural network --
the deep-learning family the reference papers focus on), reports accuracy on a
held-out test split, and saves the model + label list to models/sign_model.pkl.

Usage:
    python src/train.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sign_model.pkl")


def main():
    if not os.path.exists(DATA_PATH):
        sys.exit(f"No data found at {DATA_PATH}. Run collect_data.py first.")

    df = pd.read_csv(DATA_PATH)
    if len(df) < 20:
        sys.exit(f"Only {len(df)} samples found -- collect more before training.")

    labels = df["label"].astype(str)
    counts = labels.value_counts()
    print("Samples per label:")
    print(counts.to_string())

    if counts.size < 2:
        sys.exit("Need at least 2 different signs to train a classifier.")

    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

    # MLPClassifier + early_stopping needs numeric targets, so encode the
    # string labels to integers and keep the encoder to map predictions back.
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels.to_numpy())

    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    print("\nTraining MLP neural network...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=600,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nHeld-out test accuracy: {acc * 100:.1f}%\n")
    print(classification_report(
        encoder.inverse_transform(y_test),
        encoder.inverse_transform(y_pred),
        zero_division=0,
    ))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": clf, "encoder": encoder}, MODEL_PATH)
    print(f"Saved model to {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
