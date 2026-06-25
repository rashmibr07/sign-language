"""
Shared hand-detection utilities built on MediaPipe's Tasks API (HandLandmarker).

Every script in this project (collection, training, detection) turns a camera
frame into the SAME fixed-length feature vector here, so the model trains and
predicts on identical inputs. This is the vision-based, landmark-driven approach
the reference papers describe -- instead of feeding raw pixels to a CNN, we let
MediaPipe localise 21 hand keypoints and we classify the geometry of the hand.
"""

import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# 21 landmarks x (x, y, z) = 63 features per hand.
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3

MODEL_ASSET = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")

# Standard MediaPipe hand skeleton (pairs of landmark indices to connect).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                # palm base
]


class HandLandmarker:
    """Wrapper around MediaPipe Tasks HandLandmarker yielding normalised features."""

    def __init__(self, max_hands=1, detection_conf=0.5, tracking_conf=0.5):
        if not os.path.exists(MODEL_ASSET):
            raise FileNotFoundError(
                f"Missing model file: {MODEL_ASSET}\n"
                "Download it with:\n  curl -sSL -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_ASSET),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        """Detect hands in a BGR (OpenCV) frame.

        Returns a list of hands; each hand is a list of 21 landmark objects with
        .x, .y, .z attributes (normalised image coordinates). Empty list if none.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)
        return result.hand_landmarks or []

    @staticmethod
    def extract_features(hand_landmarks):
        """Convert one hand's 21 landmarks into a translation/scale-invariant vector.

        - Origin shift: subtract the wrist (landmark 0) so absolute position in
          the frame does not matter.
        - Scale normalise: divide by the largest landmark distance so how close
          the hand is to the camera does not matter.

        Returns a (63,) float32 array, or None if landmarks are missing.
        """
        if not hand_landmarks:
            return None

        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32,
        )  # shape (21, 3)

        coords -= coords[0]  # wrist-relative

        max_dist = np.linalg.norm(coords, axis=1).max()
        if max_dist > 1e-6:
            coords /= max_dist

        return coords.flatten()

    @staticmethod
    def draw(frame_bgr, hand_landmarks):
        """Draw the hand skeleton on the frame using OpenCV."""
        h, w = frame_bgr.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts[a], pts[b], (255, 255, 255), 2)
        for p in pts:
            cv2.circle(frame_bgr, p, 4, (0, 128, 255), -1)

    def close(self):
        self.detector.close()
