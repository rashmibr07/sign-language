# Sign Detect — Sign Language Detection App (Android)

**Sign Detect** is an Android app that recognizes **American Sign Language (ASL)
fingerspelling** — the hand signs for letters **A–Z**. Point your phone camera at a
hand sign (or upload a photo) and it tells you the letter in real time, **fully
offline**.

```
camera frame  →  find 21 hand keypoints (MediaPipe)  →  normalize  →  small neural net  →  letter
```

Instead of classifying raw pixels, the app first detects **21 hand landmarks** with
MediaPipe, then a tiny **neural network** classifies the hand's shape. This is light
enough to run smoothly on a phone with no GPU.

---

## Download & install

1. Open the repo's **Actions** tab → latest **Build APK** run → **Artifacts** →
   download `sign-detect-apk` and unzip to get `app-debug.apk`.
   (Or use the GitHub CLI: `gh run download --name sign-detect-apk`.)
2. Copy the APK to your Android phone (Drive, email, USB…).
3. Tap to install — allow **"install from unknown source"** when prompted.
4. On first launch, allow the **Camera** permission.

> It's a debug build for personal use, so Android shows the "unknown source" prompt.
> Everything runs on-device; no data leaves the phone.

---

## How to use

**Home screen — "Understand Every Sign":** tap **Detect Sign**.

**Detect screen:**
- **Live camera** — make a hand sign; the predicted letter shows at the top.
- **Upload image** — pick a photo; it shows `Detected sign: X` with confidence, or
  *"Unable to detect — please upload a different picture"* if no hand is found.
- **Switch camera** — toggle front/back (front camera matches the model best).

Use `asl_alphabet_chart.png` (in this repo) as a guide for forming each A–Z shape.

---

## How it works (technical)

1. **CameraX** streams frames (keep-only-latest so it never lags).
2. Each frame is rotated upright (and mirrored for the front camera).
3. **MediaPipe HandLandmarker** returns **21 landmarks** `(x, y, z)` → **63 numbers**.
4. The 63 numbers are normalized — **wrist-relative** (position-independent) and
   **scaled by the largest distance** (size-independent).
5. A **Multi-Layer Perceptron** classifies them:
   `63 → Dense(128)+ReLU → Dense(64)+ReLU → Dense(26)+Softmax → letter`.
6. If confidence ≥ 55%, the letter is displayed.

The classifier was trained on ~8,400 public ASL hand images (**~89%** test accuracy)
and its weights are embedded in the app as `classifier.json`. The Kotlin code runs the
exact same math as the trained model, so phone predictions match it precisely.

**Bundled assets:**
- `hand_landmarker.task` — MediaPipe hand-detection model.
- `classifier.json` — the trained neural-network weights (labels + layer weights).

**Tech stack:** Kotlin · CameraX · MediaPipe Tasks Vision · AndroidX · ExifInterface.
Min SDK 24 (Android 7.0), target SDK 34.

---

## Project layout

| Path | Purpose |
|------|---------|
| `android/app/src/main/java/.../HomeActivity.kt` | Landing screen (title + Detect button) |
| `android/app/src/main/java/.../DetectActivity.kt` | Live camera detection + image upload |
| `android/app/src/main/java/.../HandClassifier.kt` | The neural network (reads `classifier.json`) |
| `android/app/src/main/java/.../LandmarkFeatures.kt` | Turns 21 landmarks → 63 normalized features |
| `android/app/src/main/assets/hand_landmarker.task` | MediaPipe hand model |
| `android/app/src/main/assets/classifier.json` | Trained classifier weights |
| `android/app/src/main/res/` | Layouts, neon styling, launcher icon |
| `.github/workflows/build-apk.yml` | Cloud build that produces the APK |

---

## Building the APK

The APK builds automatically in the cloud via **GitHub Actions** on every push to
`main` (JDK 17 → Android SDK → Gradle 8.7 → `assembleDebug` → uploads the APK
artifact). No local Android Studio is required.

To build locally instead: open the `android/` folder in Android Studio, connect a
phone, and click **Run**.

---

## Limitations

- Recognizes **static letters only**. Letters **J** and **Z** (motion) and whole
  **words** (e.g. "thank you") need a sequence/video model and are not supported.
- Look-alike signs (**U/V/R**, **P/Q**) are the main source of errors.
- Best results: good lighting, plain background, hand fully in frame, front camera.
