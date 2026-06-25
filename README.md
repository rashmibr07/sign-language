# Sign Language Detection (real-time, webcam)

A working real-time sign-language (fingerspelling) recognizer, built as the
practical version of the vision-based + deep-learning approach described in the
reference survey papers.

Instead of feeding raw camera pixels to a CNN, it uses **MediaPipe HandLandmarker**
to find 21 hand keypoints, then a small **neural network (MLP)** classifies the
shape of the hand into a sign. This runs in real time on a laptop with no GPU.

```
camera frame  ->  MediaPipe (21 hand points)  ->  normalize  ->  MLP  ->  letter
```

## Quick start — a trained model is already included

You do **not** need to know sign language or collect any data. A ready model
(`models/sign_model.pkl`, ~89% test accuracy over A–Z) was trained for you from a
public dataset of ~8,400 hand images. After setup below, just run:

```bash
# test on the bundled sample images (no webcam):
python src/predict_image.py --image test_images/sample_L.jpg --show
python src/predict_image.py --image test_images/sample_A.jpg

# or live from your webcam:
python src/detect.py
```

To see how to form each letter for webcam mode, open **`asl_alphabet_chart.png`**
— a grid of the real hand shapes (from the training images) for A–Z. Note: J and
Z use motion in ASL, so their single-frame poses are the least reliable.

Everything below (collecting/ingesting data, training) is only needed if you want
to build your own model or improve this one.

## Setup

```bash
cd "sign laguage"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The hand-detection model file `models/hand_landmarker.task` is already included.
If it is ever missing, re-download it:

```bash
curl -sSL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## How to use it (3 steps)

### 1. Collect data

**Easiest — guided session for the whole alphabet (recommended):**
Walks you through every letter in one run: a READY screen, a countdown, then it
auto-records samples and moves to the next sign.

```bash
python src/collect_alphabet.py
python src/collect_alphabet.py --samples 150 --countdown 3
python src/collect_alphabet.py --letters ABCDEFGHIKLMNOPQRSTUVWXY   # skip J,Z (motion signs)
```

READY screen: **space**=start this sign, **s**=skip, **b**=back, **q**=quit.
While recording: **a**=abort this sign (keeps what was saved), **q**=quit.

**Or — one sign at a time** (handy for adding/redoing a single letter):

```bash
python src/collect_data.py --label A
```

In the window: press **c** to start/pause recording, **q** to quit.

**Or — no webcam at all: feed it an existing image dataset.**
Point `ingest_images.py` at a folder with one sub-folder per sign (how most public
ASL-alphabet datasets ship). It runs each image through MediaPipe and writes the
same features — so you can train without recording yourself.

```bash
python src/ingest_images.py --data_dir /path/to/asl_alphabet_train
python src/ingest_images.py --data_dir ./imgs --per_class 400
```

Free datasets to download/unzip first: *ASL Alphabet* (Kaggle) or *Sign Language
MNIST*. Images where no hand is detected are skipped and reported.

Whichever method you use, aim for ~150–200 samples per sign, with variety. You
need at least 2 different signs. All three scripts append to the same
`data/dataset.csv`.

### 2. Train the classifier
Trains the neural network on everything in `data/dataset.csv` and saves it to
`models/sign_model.pkl`, printing a test-accuracy report.

```bash
python src/train.py
```

### 3. Detect in real time

```bash
python src/detect.py
```

In the window:
- **space** — add the current predicted letter to the on-screen text
- **b** — backspace, **c** — clear, **q** — quit

`--threshold 0.7` makes it only show high-confidence predictions.

### Or test on a single photo (no webcam)
Predicts the sign in one image and prints the top guesses with confidence.

```bash
python src/predict_image.py --image path/to/photo.jpg
python src/predict_image.py --image photo.jpg --topk 3 --show
python src/predict_image.py --image photo.jpg --save annotated.jpg
```

## Project layout

| File | Purpose |
|------|---------|
| `src/hand_utils.py` | Shared: MediaPipe detection + landmark normalization (used by all 3 scripts) |
| `src/collect_alphabet.py` | Step 1 (guided) — walk through the whole alphabet in one webcam session |
| `src/collect_data.py` | Step 1 (single) — record one sign at a time from the webcam |
| `src/ingest_images.py` | Step 1 (no webcam) — build the dataset from a folder of labeled images |
| `src/train.py` | Step 2 — train the MLP neural network, report accuracy, save model |
| `src/detect.py` | Step 3 — real-time webcam prediction + sentence building |
| `src/predict_image.py` | Step 3 (no webcam) — predict the sign in a single still photo |
| `models/hand_landmarker.task` | Pretrained MediaPipe hand-keypoint model |
| `data/dataset.csv` | Your collected training samples (created in step 1) |
| `models/sign_model.pkl` | Your trained classifier (created in step 2) |

## Why landmarks instead of raw-pixel CNN?
The papers note CNNs on raw images need large labelled datasets and are sensitive
to background/lighting. Detecting 21 hand keypoints first makes the input small,
background-independent, and scale/position-invariant — so a light model trains in
seconds on data you collect yourself and still generalizes well. To extend toward
**word/sentence** level (the papers' higher levels), feed sequences of these
landmark frames into an LSTM instead of a single-frame MLP.
```
```
## Improve accuracy on your own hand, then rebuild the APK

The bundled model is trained on other people's hands. To make it much better on
*your* hand, add your own samples and rebuild:

```bash
# 1. Collect YOUR hand for every letter (webcam) — adds to the existing data
python src/collect_alphabet.py --samples 150

# 2. Retrain on the combined data
python src/train.py

# 3. Update the model the Android app uses
python src/export_for_android.py

# 4. Push — GitHub Actions rebuilds a fresh APK automatically
git add data/dataset.csv models/sign_model.pkl android/app/src/main/assets/classifier.json
git commit -m "Add my own hand samples; retrain"
git push
```

Then download the new APK from the repo's Actions tab (or via `gh run download`).

## Tips for good accuracy
- Even lighting, plain-ish background, hand fully in frame.
- Collect each sign at slightly different angles, distances, and hand tilts.
- Keep the number of samples roughly balanced across signs.
- If two signs get confused, collect more data for both and retrain.
