"""Pure hand gesture controller based on hand movement + shape.

Controls:
  - Open palm + moving left/right -> continuous movement
  - Hand stops -> no movement
  - Fist -> jump
"""

import numpy as np
from typing import Dict, Optional
from collections import deque


class GestureController:

    MOVE_THRESHOLD = 0.008   # min wrist velocity to count as movement (normalized)
    FIST_DIST = 0.025        # max finger-wrist distance for fist
    PALM_DIST = 0.06         # min avg finger distance for open palm

    def __init__(self, history_size: int = 5):
        self.prev_wrist: Optional[np.ndarray] = None
        self.action_history: deque = deque(maxlen=history_size)

    def classify(self, landmarks_dict: Optional[Dict[str, np.ndarray]]) -> str:
        if landmarks_dict is None:
            return "no_person"

        try:
            r_wrist = landmarks_dict.get("RIGHT_WRIST")
            r_index = landmarks_dict.get("RIGHT_INDEX")
            r_pinky = landmarks_dict.get("RIGHT_PINKY")
            r_thumb = landmarks_dict.get("RIGHT_THUMB")
            l_wrist = landmarks_dict.get("LEFT_WRIST")
            l_index = landmarks_dict.get("LEFT_INDEX")
            l_pinky = landmarks_dict.get("LEFT_PINKY")
            l_thumb = landmarks_dict.get("LEFT_THUMB")

            # Pick best-visible hand
            if r_wrist is not None and r_wrist[3] > 0.4:
                wrist, index, pinky, thumb = r_wrist, r_index, r_pinky, r_thumb
            elif l_wrist is not None and l_wrist[3] > 0.4:
                wrist, index, pinky, thumb = l_wrist, l_index, l_pinky, l_thumb
            else:
                return "partial_visible"

            # --- Hand shape ---
            def dist_to_wrist(f):
                if f is None or f[3] < 0.3:
                    return 0.0
                return float(np.linalg.norm(f[:2] - wrist[:2]))

            idx_d = dist_to_wrist(index)
            pky_d = dist_to_wrist(pinky)
            thm_d = dist_to_wrist(thumb)
            n = sum(1 for d in [idx_d, pky_d, thm_d] if d > 0.0001)
            avg_d = (idx_d + pky_d + thm_d) / max(1, n)

            is_fist = (idx_d < self.FIST_DIST and pky_d < self.FIST_DIST
                       and thm_d < self.FIST_DIST and avg_d > 0.0)
            is_palm = (avg_d > self.PALM_DIST)

            # --- Hand movement (velocity) ---
            curr_wrist = wrist[:2].copy()
            vx, vy = 0.0, 0.0
            if self.prev_wrist is not None:
                vx = curr_wrist[0] - self.prev_wrist[0]
                vy = curr_wrist[1] - self.prev_wrist[1]
            self.prev_wrist = curr_wrist

            moving = abs(vx) > self.MOVE_THRESHOLD or abs(vy) > self.MOVE_THRESHOLD
            moving_left = vx < -self.MOVE_THRESHOLD
            moving_right = vx > self.MOVE_THRESHOLD

            # --- Gesture decision ---
            if is_fist:
                return "fist"

            if is_palm and moving_left:
                return "move_left"
            if is_palm and moving_right:
                return "move_right"

            # Open palm but not moving (or barely moving)
            if is_palm:
                return "palm_idle"

            return "idle"

        except (KeyError, IndexError):
            return "partial_visible"

    def classify_and_store(self, landmarks_dict):
        result = self.classify(landmarks_dict)
        self.action_history.append(result)
        return result

    def get_stable_gesture(self) -> str:
        if not self.action_history:
            return "idle"
        return max(set(self.action_history), key=self.action_history.count)
