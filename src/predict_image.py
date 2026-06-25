"""
Predict the sign in a single still image (no webcam needed).

Loads models/sign_model.pkl, runs MediaPipe on the image, and prints the top
predictions with confidence. Optionally shows / saves the image annotated with
the detected hand skeleton and the predicted label.

Usage:
    python src/predict_image.py --image path/to/photo.jpg
    python src/predict_image.py --image photo.jpg --topk 3 --show
    python src/predict_image.py --image photo.jpg --save out.jpg
"""

import argparse
import os

import cv2
import joblib
import numpy as np

from hand_utils import HandLandmarker

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sign_model.pkl")


def main():
    parser = argparse.ArgumentParser(description="Predict the sign in one image.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--topk", type=int, default=3, help="How many top guesses to print (default 3).")
    parser.add_argument("--show", action="store_true", help="Open a window with the annotated image.")
    parser.add_argument("--save", default="", help="Path to save the annotated image (optional).")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"No model at {MODEL_PATH}. Train one first (train.py).")
    if not os.path.exists(args.image):
        raise SystemExit(f"Image not found: {args.image}")

    bundle = joblib.load(MODEL_PATH)
    clf, encoder = bundle["model"], bundle["encoder"]

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    landmarker = HandLandmarker(max_hands=1)
    hands = landmarker.process(img)
    if not hands:
        landmarker.close()
        raise SystemExit("No hand detected in this image. Try a clearer, closer shot of one hand.")

    hand = hands[0]
    feats = landmarker.extract_features(hand).reshape(1, -1)
    probs = clf.predict_proba(feats)[0]

    order = np.argsort(probs)[::-1][: max(1, args.topk)]
    top = [(encoder.inverse_transform([clf.classes_[i]])[0], float(probs[i])) for i in order]

    print(f"\nPrediction for {os.path.basename(args.image)}:")
    for rank, (lbl, p) in enumerate(top, 1):
        print(f"  {rank}. {lbl}   {p*100:5.1f}%")

    best_label, best_conf = top[0]

    if args.show or args.save:
        landmarker.draw(img, hand)
        cv2.rectangle(img, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(img, f"{best_label}  ({best_conf*100:.0f}%)", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if args.save:
            cv2.imwrite(args.save, img)
            print(f"\nSaved annotated image to {os.path.abspath(args.save)}")
        if args.show:
            cv2.imshow("Prediction (press any key to close)", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    landmarker.close()


if __name__ == "__main__":
    main()
