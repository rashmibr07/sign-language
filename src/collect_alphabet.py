"""
Guided one-session collector for the whole alphabet.

Walks you through a list of signs (A-Z by default), one at a time: a READY screen
lets you get your hand in position, a short countdown follows, then it auto-records
samples for that sign and advances to the next -- all in a single run. Everything
is appended to data/dataset.csv in the same format collect_data.py uses, so
train.py works unchanged afterwards.

Usage:
    python src/collect_alphabet.py
    python src/collect_alphabet.py --samples 150 --countdown 3
    python src/collect_alphabet.py --letters ABCDEFGHIKLMNOPQRSTUVWXY   # skip J,Z (motion)
    python src/collect_alphabet.py --letters 0123456789                 # digits instead

Controls:
    READY screen:  space=start this sign   s=skip   b=go back a sign   q=quit
    While recording: a=abort this sign (keep what was saved)   q=quit
Samples are only captured on frames where a hand is actually detected, so move
your hand around a little (small tilts, slight distance changes) for variety.

Note: in ASL the letters J and Z are produced with motion. A single static pose
cannot capture that fully -- hold the end position, or drop them with --letters.
"""

import argparse
import csv
import os
import time

import cv2

from hand_utils import HandLandmarker, FEATURE_DIM

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


def existing_counts():
    """Return {label: count} already present in the dataset."""
    counts = {}
    if not os.path.exists(DATA_PATH):
        return counts
    with open(DATA_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if row:
                counts[row[0]] = counts.get(row[0], 0) + 1
    return counts


def draw_center(frame, text, y, scale=1.0, color=(255, 255, 255), thick=2):
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
    x = (frame.shape[1] - size[0]) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    parser = argparse.ArgumentParser(description="Guided alphabet data collection in one session.")
    parser.add_argument("--letters", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                        help="Signs to walk through, in order (default A-Z).")
    parser.add_argument("--samples", type=int, default=200, help="Samples to record per sign (default 200).")
    parser.add_argument("--countdown", type=int, default=3, help="Seconds of countdown before recording (default 3).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    args = parser.parse_args()

    letters = list(args.letters)
    if not letters:
        raise SystemExit("No letters given.")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    new_file = not os.path.exists(DATA_PATH)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try a different --camera index.")

    landmarker = HandLandmarker(max_hands=1)
    f = open(DATA_PATH, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_DIM)])

    counts = existing_counts()
    print(f"Walking through {len(letters)} signs. Already have data for: "
          f"{sorted(k for k, v in counts.items() if v)} ")

    # State machine: phases are READY -> COUNTDOWN -> RECORD, then next letter.
    i = 0
    quit_all = False
    while 0 <= i < len(letters) and not quit_all:
        label = letters[i]
        have = counts.get(label, 0)

        # ---- READY phase: live preview, wait for the user to start/skip/back/quit.
        phase = "ready"
        countdown_end = 0.0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                quit_all = True
                break
            frame = cv2.flip(frame, 1)
            hands = landmarker.process(frame)
            hand = hands[0] if hands else None
            if hand is not None:
                landmarker.draw(frame, hand)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
            draw_center(frame, f"Sign {i+1}/{len(letters)}:  {label}", 36, 1.1, (0, 255, 255), 3)

            if phase == "ready":
                draw_center(frame, "Get into position", 110, 0.9, (255, 255, 255), 2)
                msg = "HAND DETECTED" if hand is not None else "no hand visible"
                col = (0, 220, 0) if hand is not None else (0, 0, 255)
                draw_center(frame, msg, 150, 0.8, col, 2)
                draw_center(frame, f"already saved: {have}", 190, 0.7, (200, 200, 200), 2)
                draw_center(frame, "space=start   s=skip   b=back   q=quit",
                            frame.shape[0] - 20, 0.7, (255, 255, 255), 2)

            elif phase == "countdown":
                remaining = countdown_end - time.time()
                if remaining <= 0:
                    phase = "record"
                else:
                    draw_center(frame, str(int(remaining) + 1), frame.shape[0] // 2,
                                4.0, (0, 200, 255), 6)

            if phase == "record":
                if hand is not None and saved < args.samples:
                    feats = landmarker.extract_features(hand)
                    writer.writerow([label] + feats.tolist())
                    saved += 1
                # progress bar
                pct = saved / args.samples
                bw = int(pct * (frame.shape[1] - 40))
                cv2.rectangle(frame, (20, 70), (frame.shape[1] - 20, 100), (80, 80, 80), 2)
                cv2.rectangle(frame, (20, 70), (20 + bw, 100), (0, 220, 0), -1)
                draw_center(frame, f"RECORDING  {saved}/{args.samples}", 135, 0.9, (0, 220, 0), 2)
                if hand is None:
                    draw_center(frame, "show your hand to keep recording", 175, 0.7, (0, 0, 255), 2)
                draw_center(frame, "a=abort this sign   q=quit",
                            frame.shape[0] - 20, 0.7, (255, 255, 255), 2)
                if saved >= args.samples:
                    counts[label] = have + saved
                    print(f"  {label}: saved {saved} (total {counts[label]})")
                    i += 1
                    break  # advance to next letter

            cv2.imshow("Guided Alphabet Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                quit_all = True
                break
            if phase == "ready":
                if key == ord(" "):
                    phase = "countdown"
                    countdown_end = time.time() + args.countdown
                elif key == ord("s"):
                    print(f"  {label}: skipped")
                    i += 1
                    break
                elif key == ord("b"):
                    i = max(0, i - 1)
                    break
            elif phase == "record" and key == ord("a"):
                if saved:
                    counts[label] = have + saved
                print(f"  {label}: aborted after {saved}")
                i += 1
                break

    f.close()
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    print("\nSession done. Samples per sign now:")
    for lbl in letters:
        print(f"  {lbl}: {counts.get(lbl, 0)}")
    print(f"\nSaved to {os.path.abspath(DATA_PATH)}")
    print("Next: python src/train.py")


if __name__ == "__main__":
    main()
