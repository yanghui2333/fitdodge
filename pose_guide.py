"""Pose Guide: follow-the-leader body pose matching game mode.

Target zones appear on screen. Player must match their body to the targets.
Hold the pose for 3 seconds to complete it and earn points.
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple
import pygame
from mediapipe.tasks.python.vision import PoseLandmarkerResult


# ── Pose definitions ──
# Each pose: dict of {landmark_name: (target_x, target_y, tolerance)}
# Coordinates normalized 0-1 (fraction of image width/height).
# tolerance is max allowed distance (normalized) from target.

POSE_STANDING_ARMS_OUT = {
    "name": "Stand with Arms Out",
    "emoji": "Y",
    "hold_seconds": 3.0,
    "targets": {
        "LEFT_WRIST":   (0.12, 0.32, 0.08),
        "RIGHT_WRIST":  (0.88, 0.32, 0.08),
        "LEFT_ELBOW":   (0.25, 0.36, 0.10),
        "RIGHT_ELBOW":  (0.75, 0.36, 0.10),
        "LEFT_SHOULDER":(0.38, 0.28, 0.08),
        "RIGHT_SHOULDER":(0.62, 0.28, 0.08),
        "LEFT_HIP":     (0.43, 0.55, 0.10),
        "RIGHT_HIP":    (0.57, 0.55, 0.10),
        "LEFT_KNEE":    (0.43, 0.72, 0.12),
        "RIGHT_KNEE":   (0.57, 0.72, 0.12),
        "LEFT_ANKLE":   (0.40, 0.90, 0.12),
        "RIGHT_ANKLE":  (0.60, 0.90, 0.12),
    },
}

# Add more poses here later
ALL_POSES = [POSE_STANDING_ARMS_OUT]


class PoseGuide:
    """Manages pose-matching gameplay state and rendering."""

    TOLERANCE_COLORS = {
        "hit":    (80, 255, 120),   # within tolerance
        "close":  (255, 220, 60),   # somewhat close
        "miss":   (255, 80, 80),    # far away
    }
    CLOSE_FACTOR = 2.0  # 2x tolerance = "close" zone

    def __init__(self):
        self.pose_index = 0
        self.current_pose = ALL_POSES[self.pose_index]
        self.hold_start: Optional[float] = None  # time when all targets first met
        self.completed = False
        self.score = 0
        self.total_poses = len(ALL_POSES)
        self.matched_targets: Dict[str, str] = {}  # landmark -> "hit"/"close"/"miss"
        self.feedback_text: Optional[str] = None
        self.feedback_timer = 0

    def reset(self):
        self.pose_index = 0
        self.current_pose = ALL_POSES[0]
        self.hold_start = None
        self.completed = False
        self.score = 0
        self.matched_targets.clear()

    def next_pose(self):
        """Move to next pose. Returns True if more poses remain."""
        self.pose_index += 1
        if self.pose_index >= self.total_poses:
            self.completed = True
            return False
        self.current_pose = ALL_POSES[self.pose_index]
        self.hold_start = None
        self.matched_targets.clear()
        return True

    def update(self, result: PoseLandmarkerResult, frame_w: int, frame_h: int):
        """Update pose matching state from detection result."""
        if self.completed:
            return

        if not result or not result.pose_landmarks:
            self.hold_start = None
            return

        landmarks = result.pose_landmarks[0]
        targets = self.current_pose["targets"]

        all_hit = True
        self.matched_targets.clear()

        for name, (tx, ty, tol) in targets.items():
            # Find landmark index by name
            idx = self._landmark_index(name)
            if idx is None or idx >= len(landmarks):
                all_hit = False
                continue

            lm = landmarks[idx]
            # Only check if reasonably visible
            if lm.visibility < 0.5:
                all_hit = False
                continue

            dist = math.sqrt((lm.x - tx) ** 2 + (lm.y - ty) ** 2)
            if dist <= tol:
                self.matched_targets[name] = "hit"
            elif dist <= tol * self.CLOSE_FACTOR:
                self.matched_targets[name] = "close"
                all_hit = False
            else:
                self.matched_targets[name] = "miss"
                all_hit = False

        if all_hit and len(targets) > 0:
            if self.hold_start is None:
                self.hold_start = time.time()
            else:
                elapsed = time.time() - self.hold_start
                if elapsed >= self.current_pose["hold_seconds"]:
                    self._complete_pose()
        else:
            self.hold_start = None

    def _complete_pose(self):
        """Mark current pose as complete and advance."""
        bonus = int(self.current_pose["hold_seconds"] * 10)
        self.score += 10 + bonus
        self.feedback_text = f"+{10 + bonus}  Great!"
        self.feedback_timer = 60
        if not self.next_pose():
            self.feedback_text = "All poses complete!"
            self.feedback_timer = 120

    def _landmark_index(self, name: str) -> Optional[int]:
        """Map landmark name to index (0-32)."""
        mapping = {
            "NOSE": 0, "LEFT_EYE_INNER": 1, "LEFT_EYE": 2, "LEFT_EYE_OUTER": 3,
            "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
            "LEFT_EAR": 7, "RIGHT_EAR": 8, "MOUTH_LEFT": 9, "MOUTH_RIGHT": 10,
            "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
            "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
            "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
            "LEFT_PINKY": 17, "RIGHT_PINKY": 18,
            "LEFT_INDEX": 19, "RIGHT_INDEX": 20,
            "LEFT_THUMB": 21, "RIGHT_THUMB": 22,
            "LEFT_HIP": 23, "RIGHT_HIP": 24,
            "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
            "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
            "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
            "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
        }
        return mapping.get(name)

    def draw_targets(self, surface: pygame.Surface, frame_w: int, frame_h: int):
        """Draw target zones and match status on the surface."""
        targets = self.current_pose["targets"]
        now = time.time()
        pulse = 0.7 + 0.3 * math.sin(now * 3)

        for name, (tx, ty, tol) in targets.items():
            cx = int(tx * frame_w)
            cy = int(ty * frame_h)
            radius = int(tol * min(frame_w, frame_h))

            status = self.matched_targets.get(name, "miss")
            base_color = self.TOLERANCE_COLORS[status]

            # Outer ring (tolerance boundary)
            pygame.draw.circle(surface, (*base_color, 60),
                             (cx, cy), radius, 3)
            # Zone fill
            alpha = int(30 * pulse) if status == "hit" else 15
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*base_color, alpha), (radius, radius), radius)
            surface.blit(s, (cx - radius, cy - radius))
            # Center dot
            color = (*base_color, 200)
            dot_s = pygame.Surface((12, 12), pygame.SRCALPHA)
            pygame.draw.circle(dot_s, color, (6, 6), 5)
            surface.blit(dot_s, (cx - 6, cy - 6))

    def draw_hud(self, surface: pygame.Surface, fonts: dict):
        """Draw HUD: pose name, progress, score."""
        # Pose name bar
        bar = pygame.Surface((surface.get_width(), 50), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 140))
        surface.blit(bar, (0, 0))

        pose_name = self.current_pose["name"]
        name_text = fonts["small"].render(
            f"Pose {self.pose_index + 1}/{self.total_poses}: {pose_name}", True, (80, 220, 255))
        surface.blit(name_text, (20, 12))

        # Hold progress bar
        if self.hold_start is not None:
            elapsed = time.time() - self.hold_start
            required = self.current_pose["hold_seconds"]
            progress = min(elapsed / required, 1.0)

            bar_x, bar_y = 20, 44
            bar_w, bar_h = 300, 10
            pygame.draw.rect(surface, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h), border_radius=5)
            fill_w = int(bar_w * progress)
            if fill_w > 0:
                fill_color = (80, 255, 120) if progress >= 1.0 else (255, 220, 60)
                pygame.draw.rect(surface, fill_color, (bar_x, bar_y, fill_w, bar_h), border_radius=5)
            pct_text = fonts["tiny"].render(f"{int(progress * 100)}%", True, (200, 200, 200))
            surface.blit(pct_text, (bar_x + bar_w + 10, bar_y - 2))

        # Score
        score_text = fonts["small"].render(f"Score: {self.score}", True, (255, 255, 255))
        surface.blit(score_text, (surface.get_width() - 180, 12))

        # Feedback text
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
            if self.feedback_text:
                alpha = min(255, self.feedback_timer * 4)
                fb_font = pygame.font.Font(None, 56)
                fb_surf = fb_font.render(self.feedback_text, True, (255, 255, 100))
                fb_surf.set_alpha(alpha)
                surface.blit(fb_surf, (surface.get_width() // 2 - fb_surf.get_width() // 2,
                                       surface.get_height() // 2 - 80))

        # Completed overlay
        if self.completed:
            overlay = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            surface.blit(overlay, (0, 0))
            done_font = pygame.font.Font(None, 64)
            done_text = done_font.render("All Poses Complete!", True, (80, 255, 120))
            surface.blit(done_text, (surface.get_width() // 2 - done_text.get_width() // 2,
                                     surface.get_height() // 2 - 40))
            final_text = fonts["small"].render(f"Final Score: {self.score}", True, (255, 255, 255))
            surface.blit(final_text, (surface.get_width() // 2 - final_text.get_width() // 2,
                                      surface.get_height() // 2 + 20))
            hint = fonts["tiny"].render("Press R to restart  |  Backspace: menu  |  ESC: quit", True, (150, 150, 170))
            surface.blit(hint, (surface.get_width() // 2 - hint.get_width() // 2,
                                surface.get_height() // 2 + 60))
