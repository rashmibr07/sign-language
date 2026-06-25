"""
Build the training set from a folder of images instead of the webcam.

Point this at a directory that has one sub-folder per sign, each containing
images of that sign. It runs every image through MediaPipe, extracts the same
63-D landmark features, and appends them to data/dataset.csv -- so you can train
without ever recording yourself. After this, run train.py as usual.

Expected layout (this is how most public ASL-alphabet datasets are shipped):

    dataset_dir/
        A/  a1.jpg  a2.jpg ...
        B/  b1.jpg ...
        C/  ...

Usage:
    python src/ingest_images.py --data_dir /path/to/asl_alphabet_train
    python src/ingest_images.py --data_dir ./imgs --per_class 400
    python src/ingest_images.py --data_dir ./imgs --letters A,B,C

Where to get a free dataset (no account-less options vary over time):
  - "ASL Alphabet" (Kaggle, grassknoted) -- folders A..Z, space, del, nothing
  - "Sign Language MNIST" -- 28x28 grayscale; works but low-res hands detect poorly
Download/unzip it yourself, then point --data_dir at the extracted folder.

Note: images where MediaPipe cannot find a hand are skipped and counted; very
small, blurry, or cropped-too-tight images often fail to detect.
"""

import argparse
import csv
import os

import cv2

from hand_utils import HandLandmarker, FEATURE_DIM

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    parser = argparse.ArgumentParser(description="Ingest a folder of labeled images into dataset.csv.")
    parser.add_argument("--data_dir", required=True,
                        help="Folder with one sub-folder per sign (sub-folder name = label).")
    parser.add_argument("--per_class", type=int, default=0,
                        help="Max images to use per label (0 = all).")
    parser.add_argument("--letters", default="",
                        help="Comma-separated subset of labels to ingest (default: all sub-folders).")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"Not a directory: {args.data_dir}")

    wanted = {s.strip() for s in args.letters.split(",") if s.strip()} or None

    subdirs = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d)) and (wanted is None or d in wanted)
    )
    if not subdirs:
        raise SystemExit("No matching label sub-folders found. Check --data_dir / --letters.")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    new_file = not os.path.exists(DATA_PATH)

    landmarker = HandLandmarker(max_hands=1)
    f = open(DATA_PATH, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_DIM)])

    grand_ok = grand_fail = 0
    for label in subdirs:
        folder = os.path.join(args.data_dir, label)
        files = sorted(fn for fn in os.listdir(folder) if fn.lower().endswith(IMG_EXT))
        if args.per_class > 0:
            files = files[: args.per_class]

        ok = fail = 0
        for fn in files:
            img = cv2.imread(os.path.join(folder, fn))
            if img is None:
                fail += 1
                continue
            hands = landmarker.process(img)
            if not hands:
                fail += 1
                continue
            feats = landmarker.extract_features(hands[0])
            writer.writerow([label] + feats.tolist())
            ok += 1

        grand_ok += ok
        grand_fail += fail
        print(f"  {label:>6}: {ok:5d} usable / {fail:5d} no-hand  (of {len(files)})")

    f.close()
    landmarker.close()
    print(f"\nTotal: {grand_ok} samples written, {grand_fail} images skipped (no hand detected).")
    print(f"Saved to {os.path.abspath(DATA_PATH)}")
    if grand_ok:
        print("Next: python src/train.py")
    else:
        print("Nothing usable -- check that images clearly show a single hand.")


if __name__ == "__main__":
    main()
