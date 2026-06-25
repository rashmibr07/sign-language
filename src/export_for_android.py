"""
Export the trained model into the format the Android app reads.

After (re)training with train.py, run this to regenerate
android/app/src/main/assets/classifier.json from models/sign_model.pkl.
The Android app's HandClassifier.kt runs the exact same forward pass, so the
phone predictions match the Python model.

Usage:
    python src/export_for_android.py
"""

import json
import os
import sys

import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sign_model.pkl")
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "android", "app", "src", "main", "assets", "classifier.json"
)


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"No model at {MODEL_PATH}. Train one first: python src/train.py")

    bundle = joblib.load(MODEL_PATH)
    clf, encoder = bundle["model"], bundle["encoder"]
    labels = [encoder.inverse_transform([c])[0] for c in clf.classes_]

    out = {
        "labels": labels,
        "layers": [
            {"W": clf.coefs_[i].tolist(), "b": clf.intercepts_[i].tolist()}
            for i in range(len(clf.coefs_))
        ],
        "hidden_activation": clf.activation,
        "out_activation": clf.out_activation_,
        "input_dim": int(clf.coefs_[0].shape[0]),
    }

    # Sanity: the app expects ReLU hidden + softmax output on a 63-D input.
    assert out["hidden_activation"] == "relu", out["hidden_activation"]
    assert out["out_activation"] == "softmax", out["out_activation"]
    assert out["input_dim"] == 63, out["input_dim"]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)

    print(f"Exported {len(labels)} labels, layers {[c.shape for c in clf.coefs_]}")
    print(f"Wrote {os.path.abspath(OUT_PATH)} ({os.path.getsize(OUT_PATH)} bytes)")


if __name__ == "__main__":
    main()
