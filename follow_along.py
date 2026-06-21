"""Follow-Along Mode: guided workout from extracted video poses."""

import json
import math
import os
import time
from typing import Optional
import pygame
import cv2
import numpy as np


class FollowAlong:
    """Guided follow-along using pre-extracted pose sequence."""

    HOLD_TIME = 2.0  # seconds to hold each pose
    KEY_JOINTS = ["LEFT_WRIST","RIGHT_WRIST","LEFT_ELBOW","RIGHT_ELBOW",
                  "LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP",
                  "LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE"]
    GHOST_COLOR = (100, 180, 255)   # blue ghost skeleton
    TARGET_COLOR = (255, 220, 100)  # gold target circles

    def __init__(self, pose_file: str):
        with open(pose_file, "r") as f:
            self.data = json.load(f)
        self.poses = self.data["poses"]
        self.idx = 0
        self.score = 0
        self.hold_start: Optional[float] = None
        self.matched_count = 0
        self.total_count = len(self.poses)
        self.finished = False
        self.fb_text = None
        self.fb_timer = 0

    @property
    def current_pose(self):
        return self.poses[self.idx] if self.idx < len(self.poses) else None

    @property
    def progress(self):
        return self.idx / max(1, len(self.poses))

    def update(self, result, frame_w, frame_h):
        if self.finished or not result or not result.pose_landmarks:
            return

        lm_list = result.pose_landmarks[0]
        target = self.current_pose
        if not target:
            return

        # Check match: key joints within tolerance
        matched = 0
        total = 0
        for name in self.KEY_JOINTS:
            idx = self._lm_index(name)
            if idx is None or idx >= len(lm_list):
                continue
            if name not in target["landmarks"]:
                continue
            actual = lm_list[idx]
            target_lm = target["landmarks"][name]
            if actual.visibility < 0.4 or target_lm.get("v", 0) < 0.4:
                continue
            dist = math.hypot(actual.x - target_lm["x"],
                              actual.y - target_lm["y"])
            total += 1
            if dist < 0.08:
                matched += 1

        match_ratio = matched / max(1, total) if total > 0 else 0

        if match_ratio > 0.6:
            if self.hold_start is None:
                self.hold_start = time.time()
            elif time.time() - self.hold_start >= self.HOLD_TIME:
                self._advance()
        else:
            self.hold_start = None

    def _advance(self):
        self.score += 10
        self.fb_text = "+10"
        self.fb_timer = 30
        self.idx += 1
        self.hold_start = None
        if self.idx >= len(self.poses):
            self.finished = True
            self.fb_text = "Workout Complete!"
            self.fb_timer = 120

    def _lm_index(self, name):
        mapping = {
            "NOSE":0,"LEFT_SHOULDER":11,"RIGHT_SHOULDER":12,
            "LEFT_ELBOW":13,"RIGHT_ELBOW":14,"LEFT_WRIST":15,"RIGHT_WRIST":16,
            "LEFT_HIP":23,"RIGHT_HIP":24,"LEFT_KNEE":25,"RIGHT_KNEE":26,
            "LEFT_ANKLE":27,"RIGHT_ANKLE":28,
        }
        return mapping.get(name)

    def draw_ghost(self, surface, frame_w, frame_h):
        """Draw reference pose as ghost skeleton + targets."""
        target = self.current_pose
        if not target:
            return

        md = min(frame_w, frame_h)
        CONNECTIONS = [
            (11,12),(11,23),(12,24),(23,24),
            (11,13),(13,15),(12,14),(14,16),
            (23,25),(25,27),(24,26),(26,28),
            (27,29),(28,30),(29,31),(30,32),
            (27,31),(28,32),
        ]

        # Draw ghost skeleton lines
        for a, b in CONNECTIONS:
            na = self._idx_to_name(a)
            nb = self._idx_to_name(b)
            if na not in target["landmarks"] or nb not in target["landmarks"]:
                continue
            ax = int(target["landmarks"][na]["x"] * frame_w)
            ay = int(target["landmarks"][na]["y"] * frame_h)
            bx = int(target["landmarks"][nb]["x"] * frame_w)
            by = int(target["landmarks"][nb]["y"] * frame_h)
            pygame.draw.line(surface, (*self.GHOST_COLOR, 120), (ax, ay), (bx, by), 2)

        # Draw ghost joints (circles)
        for name in self.KEY_JOINTS:
            if name not in target["landmarks"]:
                continue
            lm = target["landmarks"][name]
            cx = int(lm["x"] * frame_w)
            cy = int(lm["y"] * frame_h)
            r = int(0.015 * md)
            # Ghost circle (blue)
            pygame.draw.circle(surface, (*self.GHOST_COLOR, 100), (cx, cy), r, 2)
            # Target circle (gold, smaller)
            pygame.draw.circle(surface, (*self.TARGET_COLOR, 150), (cx, cy), max(3, r//2))

    def draw_hud(self, surface, fonts):
        bw, bh = surface.get_width(), surface.get_height()

        # Top bar
        bar = pygame.Surface((bw, 50), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 140))
        surface.blit(bar, (0, 0))

        # Title + progress
        pose_num = self.idx + 1
        title = fonts["small"].render(
            f"Follow Along: Pose {pose_num}/{self.total_count}", True, (80, 220, 255))
        surface.blit(title, (20, 12))

        # Progress bar
        bx, by, bw2, bh2 = 20, 42, bw // 3, max(4, bh // 60)
        pygame.draw.rect(surface, (40, 40, 40), (bx, by, bw2, bh2), border_radius=5)
        fw2 = int(bw2 * self.progress)
        if fw2:
            pygame.draw.rect(surface, (80, 220, 255), (bx, by, fw2, bh2), border_radius=5)

        # Score
        score_txt = fonts["small"].render(f"Score: {self.score}", True, (255, 255, 255))
        surface.blit(score_txt, (bw - 180, 12))

        # Hold progress
        if self.hold_start is not None:
            p = min((time.time() - self.hold_start) / self.HOLD_TIME, 1.0)
            pct = fonts["tiny"].render(f"Holding... {int(p*100)}%", True, (255, 220, 100))
            surface.blit(pct, (bw//2 - pct.get_width()//2, 46))

        # Feedback
        if self.fb_timer > 0:
            self.fb_timer -= 1
            if self.fb_text:
                alpha = min(255, self.fb_timer * 8)
                ff = pygame.font.Font(None, max(28, bh // 12))
                fs = ff.render(self.fb_text, True, (255, 255, 100))
                fs.set_alpha(alpha)
                surface.blit(fs, (bw//2 - fs.get_width()//2, bh//2 - 80))

        # Finished
        if self.finished:
            ov = pygame.Surface((bw, bh), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 160))
            surface.blit(ov, (0, 0))
            df = pygame.font.Font(None, max(32, bh // 10))
            dt = df.render("Workout Complete!", True, (80, 255, 120))
            surface.blit(dt, (bw//2 - dt.get_width()//2, bh//2-40))
            ft = fonts["small"].render(f"Final Score: {self.score}", True, (255, 255, 255))
            surface.blit(ft, (bw//2 - ft.get_width()//2, bh//2+20))
            h = fonts["tiny"].render("R: restart  |  Backspace: menu  |  ESC: quit", True, (150,150,170))
            surface.blit(h, (bw//2-h.get_width()//2, bh//2+60))

    def _idx_to_name(self, idx):
        names = ["NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER",
                 "RIGHT_EYE_INNER","RIGHT_EYE","RIGHT_EYE_OUTER",
                 "LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT",
                 "LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
                 "LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY","RIGHT_PINKY",
                 "LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB",
                 "LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE",
                 "LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL","RIGHT_HEEL",
                 "LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"]
        return names[idx] if idx < len(names) else ""
