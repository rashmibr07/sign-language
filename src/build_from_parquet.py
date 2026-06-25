"""
Build dataset.csv from a HuggingFace parquet file of labeled ASL images.

This is how the bundled, ready-to-use model was created -- so you can train a
working recognizer WITHOUT knowing sign language or recording yourself. It reads
a parquet whose rows are (image bytes, label), runs each image through MediaPipe,
and writes the same 63-D landmark features the other scripts use.

The default dataset is 'Marxulia/asl_sign_languages_alphabets_v03' (~10.8k images,
A-Z, embedded in one parquet). Images where no hand is detected are skipped.

Usage (download the parquet first, see README), then:
    python src/build_from_parquet.py --parquet /path/to/asl.parquet
    python src/build_from_parquet.py --parquet asl.parquet --per_class 250
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd

from hand_utils import HandLandmarker, FEATURE_DIM

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def to_label(value):
    """Map a parquet label (int index or string) to a letter."""
    if isinstance(value, (int, np.integer)):
        return LETTERS[int(value)]
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Build dataset.csv from a parquet of labeled images.")
    parser.add_argument("--parquet", required=True, help="Path to the downloaded parquet file.")
    parser.add_argument("--per_class", type=int, default=0, help="Max usable samples per label (0 = all).")
    parser.add_argument("--image_col", default="image", help="Column holding image bytes (default 'image').")
    parser.add_argument("--label_col", default="label", help="Column holding the label (default 'label').")
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        raise SystemExit(f"Parquet not found: {args.parquet}")

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} rows from {os.path.basename(args.parquet)}")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    landmarker = HandLandmarker(max_hands=1)

    import csv
    with open(DATA_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_DIM)])

        per_class = {}
        ok = fail = 0
        for n, (_, row) in enumerate(df.iterrows()):
            label = to_label(row[args.label_col])
            if args.per_class and per_class.get(label, 0) >= args.per_class:
                continue

            cell = row[args.image_col]
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            if raw is None:
                fail += 1
                continue
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                fail += 1
                continue

            hands = landmarker.process(img)
            if not hands:
                fail += 1
                continue

            feats = landmarker.extract_features(hands[0])
            writer.writerow([label] + feats.tolist())
            per_class[label] = per_class.get(label, 0) + 1
            ok += 1

            if (n + 1) % 500 == 0:
                print(f"  processed {n+1}/{len(df)}  (usable {ok}, skipped {fail})")

    landmarker.close()
    print(f"\nDone: {ok} usable samples, {fail} skipped (no hand / unreadable).")
    print("Per-letter usable counts:")
    for L in LETTERS:
        if L in per_class:
            print(f"  {L}: {per_class[L]}")
    print(f"\nSaved to {os.path.abspath(DATA_PATH)}\nNext: python src/train.py")


if __name__ == "__main__":
    main()
