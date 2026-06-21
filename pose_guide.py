"""Pose Challenge mode: match track-extracted poses in real-time with video sync.
Scores each frame and gives a final score out of 100.
Target ghost is aligned to the user's real-time torso position.
"""

import json
import math
import time
from typing import List, Optional, Tuple
import pygame

# MediaPipe landmark name -> index mapping
LM = {
    "NOSE":0,
    "LEFT_SHOULDER":11,"RIGHT_SHOULDER":12,
    "LEFT_ELBOW":13,"RIGHT_ELBOW":14,
    "LEFT_WRIST":15,"RIGHT_WRIST":16,
    "LEFT_HIP":23,"RIGHT_HIP":24,
    "LEFT_KNEE":25,"RIGHT_KNEE":26,
    "LEFT_ANKLE":27,"RIGHT_ANKLE":28,
    "LEFT_HEEL":29,"RIGHT_HEEL":30,
    "LEFT_FOOT_INDEX":31,"RIGHT_FOOT_INDEX":32,
}

# Key joints used for scoring
SCORE_JOINTS = [
    "LEFT_SHOULDER","RIGHT_SHOULDER",
    "LEFT_ELBOW","RIGHT_ELBOW",
    "LEFT_WRIST","RIGHT_WRIST",
    "LEFT_HIP","RIGHT_HIP",
    "LEFT_KNEE","RIGHT_KNEE",
    "LEFT_ANKLE","RIGHT_ANKLE",
]

# Torso reference joints (used for alignment)
TORSO_JOINTS = ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"]
# Torso joints for display (must match user)
TORSO = ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"]
# Limb anchor mapping: limb -> torso parent
ANCHOR = {"LEFT_ELBOW":"LEFT_SHOULDER","LEFT_WRIST":"LEFT_SHOULDER",
          "RIGHT_ELBOW":"RIGHT_SHOULDER","RIGHT_WRIST":"RIGHT_SHOULDER",
          "LEFT_KNEE":"LEFT_HIP","LEFT_ANKLE":"LEFT_HIP",
          "RIGHT_KNEE":"RIGHT_HIP","RIGHT_ANKLE":"RIGHT_HIP"}

# Skeleton connections for drawing
CONNECTIONS = [
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(12,14),(14,16),
    (23,25),(25,27),(24,26),(26,28),
    (27,29),(28,30),(29,31),(30,32),
    (27,31),(28,32),
]

_LM_IDX_TO_NAME = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER","RIGHT_EYE","RIGHT_EYE_OUTER",
    "LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT",
    "LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
    "LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY","RIGHT_PINKY",
    "LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB",
    "LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE",
    "LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL","RIGHT_HEEL",
    "LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX",
]


def _lm_idx_to_name(idx: int) -> str:
    return _LM_IDX_TO_NAME[idx] if 0 <= idx < len(_LM_IDX_TO_NAME) else ""


class PoseGuide:
    """Pose Challenge guide: loads track poses, syncs to video time, scores user."""

    GHOST_COLOR = (100, 180, 255)   # blue ghost skeleton
    TARGET_COLOR = (255, 100, 100)  # red target circles
    SCORE_COLORS = {
        "excellent": (80, 255, 120),
        "good": (255, 220, 60),
        "ok": (255, 160, 60),
        "poor": (255, 80, 80),
    }

    def __init__(self, track_path: str):
        with open(track_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.poses: List[dict] = self.data.get("poses", [])
        self.total_poses = len(self.poses)
        self.name = self.data.get("name", "Unknown Track")

        # Compute total video duration from last pose time_ms
        self.total_duration_ms = 0
        if self.poses:
            self.total_duration_ms = self.poses[-1].get("time_ms", 0)

        # State
        self.current_idx = 0
        self.current_video_ms = 0.0   # current video playback position (ms)
        self.started = False
        self.finished = False

        # Person detection
        self.has_person = False
        self.user_torso: Optional[Tuple[float, float]] = None

        # Scoring
        self.total_score = 0.0
        self.frame_scores: List[float] = []
        self.current_match_pct = 0.0
        self.final_score = 0

        # Feedback
        self.fb_text: Optional[str] = None
        self.fb_timer = 0

        # Target display
        self.target_landmarks: dict = {}
        self.user_landmarks = None
        self.track_torso: Optional[Tuple[float, float]] = None

        # Smiley animation
        self._smiley_t = 0.0

    def reset(self):
        self.current_idx = 0
        self.current_video_ms = 0.0
        self.started = False
        self.finished = False
        self.has_person = False
        self.user_torso = None
        self.track_torso = None
        self.total_score = 0.0
        self.frame_scores.clear()
        self.current_match_pct = 0.0
        self.final_score = 0
        self.fb_text = None
        self.fb_timer = 0
        self.target_landmarks = {}
        self.user_landmarks = None
        self._smiley_t = 0.0

    @property
    def current_pose(self) -> Optional[dict]:
        if 0 <= self.current_idx < self.total_poses:
            return self.poses[self.current_idx]
        return None

    @property
    def progress(self) -> float:
        if self.total_duration_ms <= 0:
            return self.current_idx / max(1, self.total_poses)
        return min(1.0, self.current_video_ms / self.total_duration_ms)

    @property
    def pose_num(self) -> int:
        return self.current_idx + 1

    @property
    def elapsed_seconds(self) -> float:
        return self.current_video_ms / 1000.0

    @property
    def total_seconds(self) -> float:
        return self.total_duration_ms / 1000.0

    def start(self):
        self.started = True
        self.current_video_ms = 0.0

    def update(self, result, video_position_ms: float, fw: int, fh: int):
        """Update guide: advance pose index based on video playback position.

        Args:
            result: MediaPipe PoseLandmarkerResult from camera
            video_position_ms: current video playback position in milliseconds
            fw, fh: camera frame width/height
        """
        if self.finished or not self.started:
            return
        if not self.poses:
            self.finished = True
            return

        self.current_video_ms = video_position_ms
        self._smiley_t += 0.05

        # Check for person
        self.has_person = (result is not None
                           and result.pose_landmarks is not None
                           and len(result.pose_landmarks) > 0)

        # Compute user torso from real-time landmarks
        if self.has_person:
            lms = result.pose_landmarks[0]
            torso_pts = []
            for name in TORSO_JOINTS:
                idx = LM.get(name)
                if idx is not None and idx < len(lms) and lms[idx].visibility > 0.3:
                    torso_pts.append((lms[idx].x, lms[idx].y))
            if len(torso_pts) >= 2:
                self.user_torso = (
                    sum(p[0] for p in torso_pts) / len(torso_pts),
                    sum(p[1] for p in torso_pts) / len(torso_pts),
                )
            else:
                self.user_torso = None
        else:
            self.user_torso = None

        # Advance pose index to match video time
        while (self.current_idx + 1 < self.total_poses and
               video_position_ms >= self.poses[self.current_idx + 1].get("time_ms", 0)):
            self.current_idx += 1

        # Check if done
        if self.current_idx >= self.total_poses - 1 and video_position_ms >= self.total_duration_ms:
            self.finished = True
            self._compute_final_score()
            return

        # Update target landmarks
        cp = self.current_pose
        if cp:
            self.target_landmarks = cp.get("landmarks", {})
            self.user_landmarks = None
            if self.has_person and result.pose_landmarks:
                self.user_landmarks = result.pose_landmarks[0]
            # Compute track torso
            track_pts = []
            for name in TORSO_JOINTS:
                if name in self.target_landmarks:
                    tlm = self.target_landmarks[name]
                    v = tlm.get("v", 1.0) if isinstance(tlm, dict) else 1.0
                    if v > 0.3:
                        tx = tlm["x"] if isinstance(tlm, dict) else tlm[0]
                        ty = tlm["y"] if isinstance(tlm, dict) else tlm[1]
                        track_pts.append((tx, ty))
            if len(track_pts) >= 2:
                self.track_torso = (
                    sum(p[0] for p in track_pts) / len(track_pts),
                    sum(p[1] for p in track_pts) / len(track_pts),
                )
            else:
                self.track_torso = None
        else:
            self.target_landmarks = {}
            self.user_landmarks = None
            self.track_torso = None

        # Score current frame
        self._score_frame(result, fw, fh)

    def _score_frame(self, result, fw: int, fh: int):
        if not self.has_person:
            self.current_match_pct = 0.0
            self.frame_scores.append(0.0)
            return

        lm_list = result.pose_landmarks[0]
        target = self.target_landmarks
        if not target:
            self.current_match_pct = 0.0
            self.frame_scores.append(0.0)
            return

        total_error = 0.0
        count = 0
        for name in SCORE_JOINTS:
            idx = LM.get(name)
            if idx is None or idx >= len(lm_list):
                continue
            if name not in target:
                continue

            actual_lm = lm_list[idx]
            target_lm = target[name]
            target_v = target_lm.get("v", 1.0) if isinstance(target_lm, dict) else 1.0

            if actual_lm.visibility < 0.3 or target_v < 0.3:
                continue

            tx = target_lm["x"] if isinstance(target_lm, dict) else target_lm[0]
            ty = target_lm["y"] if isinstance(target_lm, dict) else target_lm[1]

            dist = math.hypot(actual_lm.x - tx, actual_lm.y - ty)
            error = min(dist, 0.5) / 0.5
            total_error += error
            count += 1

        if count > 0:
            avg_error = total_error / count
            match_pct = max(0.0, 1.0 - avg_error)
        else:
            match_pct = 0.0

        self.current_match_pct = match_pct
        self.frame_scores.append(match_pct * 100.0)
        self.total_score += match_pct * 100.0

    def _compute_final_score(self):
        if not self.frame_scores:
            self.final_score = 0
        else:
            self.final_score = int(sum(self.frame_scores) / len(self.frame_scores))

    def get_match_rating(self) -> Tuple[str, Tuple[int, int, int]]:
        pct = self.current_match_pct
        if pct >= 0.75:
            return ("EXCELLENT!", self.SCORE_COLORS["excellent"])
        elif pct >= 0.55:
            return ("Good", self.SCORE_COLORS["good"])
        elif pct >= 0.35:
            return ("OK", self.SCORE_COLORS["ok"])
        else:
            return ("Keep trying", self.SCORE_COLORS["poor"])

    def get_live_score(self) -> int:
        if not self.frame_scores:
            return 0
        return int(sum(self.frame_scores) / len(self.frame_scores))

    # ─── Drawing ────────────────────────────────────────────────────

    def _get_torso_offset(self) -> Tuple[float, float]:
        """Compute offset to align track landmarks to user torso."""
        if self.user_torso is None or self.track_torso is None:
            return (0.0, 0.0)
        return (
            self.user_torso[0] - self.track_torso[0],
            self.user_torso[1] - self.track_torso[1],
        )

    def draw_ghost(self, surface: pygame.Surface, frame_w: int, frame_h: int):
        """Draw target pose. Torso joints match user exactly; limbs relative to anchor."""
        target = self.target_landmarks
        if not target or not self.has_person or self.user_landmarks is None:
            return

        md = min(frame_w, frame_h)
        ulm = self.user_landmarks

        def tp(name):
            t = target.get(name)
            return (t["x"] if isinstance(t,dict) else t[0], t["y"] if isinstance(t,dict) else t[1]) if t else None
        def up(name):
            i = LM.get(name)
            return (ulm[i].x, ulm[i].y) if i is not None and i < len(ulm) and ulm[i].visibility > 0.3 else None

        pos = {}
        # Toros joints: user position
        for n in TORSO:
            p = up(n)
            if p: pos[n] = (int(p[0]*frame_w), int(p[1]*frame_h))
        # Limb joints: user's segment length x track direction
        for limb, anchor in ANCHOR.items():
            if anchor not in pos: continue
            pt = tp(limb); pa = tp(anchor)
            if pt is None or pa is None: continue
            # Normalized direction from track
            tdx = pt[0] - pa[0]; tdy = pt[1] - pa[1]
            tlen = math.hypot(tdx, tdy)
            if tlen < 0.001: continue
            tdx /= tlen; tdy /= tlen
            # User's actual segment length
            ua = up(anchor); ul = up(limb)
            if ua is None or ul is None: continue
            ulen = math.hypot(ul[0]-ua[0], ul[1]-ua[1])
            # Apply: user_anchor + track_direction * user_length
            ua_norm = (pos[anchor][0]/frame_w, pos[anchor][1]/frame_h)
            pos[limb] = (int((ua_norm[0]+tdx*ulen)*frame_w), int((ua_norm[1]+tdy*ulen)*frame_h))

        # Draw skeleton
        for ai, bi in CONNECTIONS:
            na, nb = _lm_idx_to_name(ai), _lm_idx_to_name(bi)
            if na in pos and nb in pos:
                pygame.draw.line(surface, (*self.GHOST_COLOR, 140), pos[na], pos[nb], 3)

        # Draw circles
        for name in SCORE_JOINTS:
            if name in pos:
                r = int(0.018*md)
                col = (80,255,120) if name in TORSO else self.TARGET_COLOR
                pygame.draw.circle(surface, (*col, 160), pos[name], r, 3)
                pygame.draw.circle(surface, (*col, 220), pos[name], max(3, r//3))

    def draw_smiley(self, surface: pygame.Surface, result, frame_w: int, frame_h: int):
        """Draw a smiley face at the user's nose position."""
        if not self.has_person or result is None:
            return
        lms = result.pose_landmarks[0]
        nose_idx = LM.get("NOSE", 0)
        if nose_idx >= len(lms) or lms[nose_idx].visibility < 0.3:
            return

        nx = int(lms[nose_idx].x * frame_w)
        ny = int(lms[nose_idx].y * frame_h)

        # Size based on frame dimensions
        r = int(min(frame_w, frame_h) * 0.04)
        if r < 8:
            return

        # Face circle (yellow)
        pygame.draw.circle(surface, (255, 220, 80), (nx, ny), r)
        pygame.draw.circle(surface, (60, 40, 0), (nx, ny), r, 2)

        # Eyes
        eye_off = r // 3
        eye_r = max(2, r // 6)
        pygame.draw.circle(surface, (30, 30, 30), (nx - eye_off, ny - r // 4), eye_r)
        pygame.draw.circle(surface, (30, 30, 30), (nx + eye_off, ny - r // 4), eye_r)

        # Smile
        smile_amp = int(r * 0.3 + 3 * math.sin(self._smiley_t * 0.5))
        smile_y = ny + r // 5
        smile_pts = []
        for i in range(7):
            sx = nx - r // 2 + i * r // 6
            sy = smile_y - int(abs(i - 3) * smile_amp / 3)
            smile_pts.append((sx, sy))
        if len(smile_pts) >= 2:
            pygame.draw.lines(surface, (30, 30, 30), False, smile_pts, 2)

    def draw_hud(self, surface: pygame.Surface, fonts: dict):
        """Draw HUD overlay: score, progress bar, time, rating."""
        bw, bh = surface.get_width(), surface.get_height()

        # Top bar (taller for prominence)
        bar_h = int(bh * 0.10)
        bar = pygame.Surface((bw, bar_h), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 170))
        surface.blit(bar, (0, 0))

        pad_x = int(bw * 0.02)
        pad_y = int(bar_h * 0.08)

        # Track name + pose number
        title_text = f"{self.name}  |  Pose {self.pose_num}/{self.total_poses}"
        title = fonts["small"].render(title_text, True, (80, 220, 255))
        surface.blit(title, (pad_x, pad_y))

        # Elapsed time
        elapsed = self.elapsed_seconds
        total_s = self.total_seconds
        time_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d} / {int(total_s//60):02d}:{int(total_s%60):02d}"
        time_surf = fonts["small"].render(time_str, True, (200, 200, 220))
        surface.blit(time_surf, (bw // 2 - time_surf.get_width() // 2, pad_y))

        # Live score
        live_score = self.get_live_score()
        score_txt = fonts["small"].render(f"Score: {live_score}/100", True, (255, 255, 255))
        surface.blit(score_txt, (bw - score_txt.get_width() - pad_x, pad_y))

        # Match rating
        rating, r_color = self.get_match_rating()
        rt = fonts["small"].render(rating, True, r_color)
        surface.blit(rt, (bw - rt.get_width() - pad_x, pad_y + int(bar_h * 0.35)))

        # Prominent progress bar (bottom of HUD bar)
        pb_y = bar_h - int(bar_h * 0.35)
        pb_h = int(bar_h * 0.25)
        pb_x = pad_x
        pb_w = bw - 2 * pad_x
        # Background
        pygame.draw.rect(surface, (30, 30, 40), (pb_x, pb_y, pb_w, pb_h), border_radius=pb_h // 2)
        # Filled
        fw2 = int(pb_w * self.progress)
        if fw2 > 0:
            # Gradient-like: use a brighter color
            progress_color = (80, 220, 255)
            pygame.draw.rect(surface, progress_color, (pb_x, pb_y, fw2, pb_h), border_radius=pb_h // 2)
        # Percentage text on bar
        pct_text = f"{int(self.progress * 100)}%"
        pct_surf = fonts["tiny"].render(pct_text, True, (255, 255, 255))
        surface.blit(pct_surf, (pb_x + pb_w // 2 - pct_surf.get_width() // 2, pb_y - int(bar_h * 0.05)))

        # Finished overlay
        if self.finished:
            ov = pygame.Surface((bw, bh), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 190))
            surface.blit(ov, (0, 0))

            bf = pygame.font.Font(None, max(32, bh // 10))
            t1 = bf.render("Challenge Complete!", True, (80, 255, 120))
            surface.blit(t1, (bw // 2 - t1.get_width() // 2, bh // 2 - 80))

            t2 = bf.render(f"Final Score: {self.final_score}/100", True, (255, 255, 100))
            surface.blit(t2, (bw // 2 - t2.get_width() // 2, bh // 2 - 20))

            # Letter grade
            if self.final_score >= 90:
                grade, gc = "S", (255, 215, 0)
            elif self.final_score >= 75:
                grade, gc = "A", (80, 255, 120)
            elif self.final_score >= 60:
                grade, gc = "B", (255, 220, 60)
            elif self.final_score >= 45:
                grade, gc = "C", (255, 160, 60)
            else:
                grade, gc = "D", (255, 80, 80)
            gfont = pygame.font.Font(None, max(40, bh // 8))
            t3 = gfont.render(grade, True, gc)
            surface.blit(t3, (bw // 2 - t3.get_width() // 2, bh // 2 + 50))

            hint = fonts["tiny"].render("R: restart  |  Backspace: menu  |  ESC: quit", True, (150, 150, 170))
            surface.blit(hint, (bw // 2 - hint.get_width() // 2, bh // 2 + 110))
