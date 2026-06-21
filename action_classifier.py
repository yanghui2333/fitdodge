"""Rule-based action classifier for detecting body movements.

Detects: squat, jump, left_lean, right_lean, arm_raise, idle
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque


class ActionClassifier:
    """Classifies body actions from pose landmarks using geometric rules."""

    # Thresholds (calibrated for camera ~2m away, standing full-body)
    SQUAT_KNEE_ANGLE = 120       # degrees: knee angle below this = squatting
    SQUAT_HIP_ANGLE = 120         # degrees: hip angle below this = squatting
    STAND_KNEE_ANGLE = 155        # degrees: knee angle above this = standing
    STAND_HIP_ANGLE = 150         # degrees: hip angle above this = standing
    JUMP_ANKLE_RISE = 0.06        # fraction of frame height: ankle y drop = jumping up
    LEAN_SHOULDER_OFFSET = 0.06   # fraction of frame width: shoulder lateral offset
    ARM_RAISE_WRIST_Y = 0.08      # fraction: wrist above shoulder threshold

    def __init__(self, history_size: int = 5):
        self.action_history: deque = deque(maxlen=history_size)
        self.prev_ankle_y: Optional[float] = None  # average of L+R ankle
        self._prev_pose_name: Optional[str] = None

    def classify(self, landmarks_dict: Optional[Dict[str, np.ndarray]],
                 frame_height: int, frame_width: int) -> str:
        """Classify current action from landmark dictionary.
        
        Returns one of: squat, jump, left_lean, right_lean, arm_raise, idle
        """
        if landmarks_dict is None:
            return "no_person"

        try:
            # --- Extract key points ---
            l_hip = landmarks_dict["LEFT_HIP"]
            r_hip = landmarks_dict["RIGHT_HIP"]
            l_knee = landmarks_dict["LEFT_KNEE"]
            r_knee = landmarks_dict["RIGHT_KNEE"]
            l_ankle = landmarks_dict["LEFT_ANKLE"]
            r_ankle = landmarks_dict["RIGHT_ANKLE"]
            l_shoulder = landmarks_dict["LEFT_SHOULDER"]
            r_shoulder = landmarks_dict["RIGHT_SHOULDER"]
            l_wrist = landmarks_dict["LEFT_WRIST"]
            r_wrist = landmarks_dict["RIGHT_WRIST"]
            mid_hip = (l_hip + r_hip) / 2
            mid_shoulder = (l_shoulder + r_shoulder) / 2
            mid_ankle = (l_ankle + r_ankle) / 2

            # --- Check required visibility ---
            min_vis = 0.5
            key_points = [l_hip, r_hip, l_knee, r_knee, l_shoulder, r_shoulder]
            if any(p[3] < min_vis for p in key_points):
                return "partial_visible"

            # --- 1. JUMP detection (check ankle rise first; order matters) ---
            curr_ankle_y = mid_ankle[1]  # y increases going down in image
            is_jumping = False
            if self.prev_ankle_y is not None:
                ankle_rise = self.prev_ankle_y - curr_ankle_y
                if ankle_rise > self.JUMP_ANKLE_RISE:
                    is_jumping = True
            self.prev_ankle_y = curr_ankle_y

            # --- 2. SQUAT detection ---
            # Knee angle: hip-knee-ankle
            l_knee_angle = self._angle(l_hip, l_knee, l_ankle)
            r_knee_angle = self._angle(r_hip, r_knee, r_ankle)
            avg_knee_angle = (l_knee_angle + r_knee_angle) / 2
            # Hip angle: shoulder-hip-knee
            l_hip_angle = self._angle(l_shoulder, l_hip, l_knee)
            r_hip_angle = self._angle(r_shoulder, r_hip, r_knee)
            avg_hip_angle = (l_hip_angle + r_hip_angle) / 2

            is_squatting = (avg_knee_angle < self.SQUAT_KNEE_ANGLE and
                           avg_hip_angle < self.SQUAT_HIP_ANGLE)

            # --- 3. LEAN detection ---
            shoulder_center_x = mid_shoulder[0]
            hip_center_x = mid_hip[0]
            torso_offset = shoulder_center_x - hip_center_x
            if torso_offset > self.LEAN_SHOULDER_OFFSET:
                lean = "right_lean"
            elif torso_offset < -self.LEAN_SHOULDER_OFFSET:
                lean = "left_lean"
            else:
                lean = None

            # --- 4. ARM RAISE detection ---
            l_wrist_above = (l_shoulder[1] - l_wrist[1]) > self.ARM_RAISE_WRIST_Y
            r_wrist_above = (r_shoulder[1] - r_wrist[1]) > self.ARM_RAISE_WRIST_Y
            arms_raised = l_wrist_above or r_wrist_above

            # --- Determine action ---
            if is_jumping:
                action = "jump"
            elif lean is not None:
                action = lean
            elif is_squatting:
                action = "squat"
            elif arms_raised:
                action = "arm_raise"
            else:
                action = "idle"

        except (KeyError, IndexError):
            return "partial_visible"

        self.action_history.append(action)
        return action

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate angle ABC (at B) in degrees using (x, y) coords."""
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_val)))

    def get_stable_action(self) -> str:
        """Return most common action from recent history for stability."""
        if not self.action_history:
            return "idle"
        return max(set(self.action_history), key=self.action_history.count)
