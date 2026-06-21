"""Pose detection module using MediaPipe Tasks API (0.10.x)."""

import cv2
import numpy as np
import os
from typing import Optional, Tuple, Dict, Any

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult


class PoseDetector:
    """Real-time body pose detector using MediaPipe PoseLandmarker (tasks API)."""

    def __init__(self, model_path: Optional[str] = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker_v2.task")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0

        # Landmark mapping: MediaPipe Pose has 33 landmarks
        self.LANDMARK_NAMES = [
            "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
            "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
            "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
            "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
            "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
            "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
            "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
            "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
        ]

        # Pose connections for drawing
        self.CONNECTIONS = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (23, 25), (25, 27), (24, 26), (26, 28),
            (15, 17), (15, 19), (15, 21), (17, 19),
            (16, 18), (16, 20), (16, 22), (18, 20),
            (27, 29), (28, 30), (29, 31), (30, 32),
            (27, 31), (28, 32),
        ]

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[PoseLandmarkerResult]]:
        """Process a single frame, return annotated frame + result."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += 33  # ~30fps
        return frame, result

    def draw_landmarks(self, frame: np.ndarray, result: PoseLandmarkerResult) -> np.ndarray:
        """Draw pose landmarks and connections onto the frame."""
        if not result.pose_landmarks:
            return frame
        h, w = frame.shape[:2]
        for landmarks in result.pose_landmarks:
            # Draw connections
            for start_idx, end_idx in self.CONNECTIONS:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    sx = int(landmarks[start_idx].x * w)
                    sy = int(landmarks[start_idx].y * h)
                    ex = int(landmarks[end_idx].x * w)
                    ey = int(landmarks[end_idx].y * h)
                    cv2.line(frame, (sx, sy), (ex, ey), (0, 255, 100), 2)
            # Draw landmarks
            for lm in landmarks:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 200, 255), -1)
        return frame

    def landmarks_to_dict(self, result: PoseLandmarkerResult) -> Optional[Dict[str, np.ndarray]]:
        """Convert first detected pose landmarks to dict keyed by name."""
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        out = {}
        for i, name in enumerate(self.LANDMARK_NAMES):
            if i < len(landmarks):
                lm = landmarks[i]
                out[name] = np.array([lm.x, lm.y, lm.z, lm.visibility])
        return out

    def get_landmark_positions(self, result: PoseLandmarkerResult) -> Optional[np.ndarray]:
        """Return all landmarks as an (33, 4) numpy array."""
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])

    def release(self):
        self.landmarker.close()
