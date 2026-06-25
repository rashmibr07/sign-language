"""
Step 3 -- Real-time sign detection from the webcam.

Loads models/sign_model.pkl, runs MediaPipe on each camera frame, and predicts
the sign live. A short smoothing buffer and a confidence threshold keep the
on-screen prediction stable, and confirmed letters are appended to a sentence
strip at the bottom -- the "fingerspelling -> text" idea from the papers.

Usage:
    python src/detect.py
    python src/detect.py --threshold 0.7

Controls:
    space -> add the current letter to the sentence
    b     -> backspace the sentence
    c     -> clear the sentence
    q     -> quit
"""

import argparse
import os
from collections import Counter, deque

import cv2
import joblib
import numpy as np

from hand_utils import HandLandmarker

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sign_model.pkl")


def main():
    parser = argparse.ArgumentParser(description="Real-time sign language detection.")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Minimum probability to show a prediction (default 0.6)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"No model at {MODEL_PATH}. Run collect_data.py then train.py first.")

    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]
    encoder = bundle["encoder"]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try a different --camera index.")

    landmarker = HandLandmarker(max_hands=1)
    recent = deque(maxlen=8)  # smoothing window over recent frames
    sentence = ""

    print("Detecting... space=add letter  b=backspace  c=clear  q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        hands = landmarker.process(frame)
        label, conf = None, 0.0
        if hands:
            hand = hands[0]
            landmarker.draw(frame, hand)
            feats = landmarker.extract_features(hand).reshape(1, -1)
            probs = clf.predict_proba(feats)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            if conf >= args.threshold:
                label = encoder.inverse_transform([clf.classes_[idx]])[0]
                recent.append(label)
        else:
            recent.clear()

        # Stable prediction = most common label across the smoothing window.
        stable = None
        if recent:
            stable, votes = Counter(recent).most_common(1)[0]
            if votes < recent.maxlen // 2:
                stable = None

        banner = f"{stable}  ({conf*100:.0f}%)" if stable else "..."
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(frame, f"Prediction: {banner}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 90), (frame.shape[1], h - 40), (40, 40, 40), -1)
        cv2.putText(frame, f"Text: {sentence}", (10, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "space=add  b=back  c=clear  q=quit", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Sign Language Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and stable:
            sentence += str(stable)
        elif key == ord("b"):
            sentence = sentence[:-1]
        elif key == ord("c"):
            sentence = ""

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print(f"Final text: {sentence}")


if __name__ == "__main__":
    main()
