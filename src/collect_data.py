"""
Step 1 -- Collect your own training data from the webcam.

Run this once per sign you want to recognise. It opens the camera, detects your
hand, and saves the normalised 63-D landmark vector for every frame you record
into data/dataset.csv (appending, so you can build the dataset incrementally).

Usage:
    python src/collect_data.py --label A
    python src/collect_data.py --label B --samples 300

Controls (in the camera window):
    c  -> start / pause recording for the current label
    q  -> quit and save

Tip: capture each sign from a few angles and distances, and with small hand
movements, so the model generalises instead of memorising one exact pose.
"""

import argparse
import csv
import os

import cv2

from hand_utils import HandLandmarker, FEATURE_DIM

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


def count_existing(label):
    if not os.path.exists(DATA_PATH):
        return 0
    n = 0
    with open(DATA_PATH, newline="") as f:
        for row in csv.reader(f):
            if row and row[0] == label:
                n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Collect hand-landmark samples for one sign.")
    parser.add_argument("--label", required=True, help="The sign/letter this batch represents, e.g. A")
    parser.add_argument("--samples", type=int, default=200, help="How many frames to record (default 200)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()
    label = args.label.strip()

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    new_file = not os.path.exists(DATA_PATH)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try a different --camera index.")

    landmarker = HandLandmarker(max_hands=1)
    recording = False
    saved = 0
    already = count_existing(label)

    f = open(DATA_PATH, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_DIM)])

    print(f"Collecting for label '{label}'. Press 'c' to record, 'q' to quit.")
    print(f"Existing samples for '{label}': {already}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror so it feels natural
        hands = landmarker.process(frame)

        hand = None
        if hands:
            hand = hands[0]
            landmarker.draw(frame, hand)

        if recording and hand is not None and saved < args.samples:
            feats = landmarker.extract_features(hand)
            writer.writerow([label] + feats.tolist())
            saved += 1
            if saved >= args.samples:
                recording = False
                print(f"Reached {args.samples} samples for '{label}'.")

        status = "RECORDING" if recording else "paused"
        color = (0, 0, 255) if recording else (0, 200, 0)
        cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, f"{status}  saved this run: {saved}/{args.samples}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"total for '{label}': {already + saved}",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, "c=record/pause   q=quit",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Collect Sign Data", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            recording = not recording

    f.close()
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved} new samples for '{label}' to {os.path.abspath(DATA_PATH)}")


if __name__ == "__main__":
    main()
