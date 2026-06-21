"""FitDodge - Body-motion controlled dodge game.

Modes:
  - Pose Preview: camera + skeleton overlay (no game)
  - Game Mode: dodge obstacles with body movements
  - Pose Challenge: match target poses with your body
"""

import sys
import cv2
import numpy as np
import pygame

from pose_detector import PoseDetector
from action_classifier import ActionClassifier
from effects import EffectsManager
from game import DodgeGame
from pose_guide import PoseGuide

# App states
MENU = "menu"
POSE_ONLY = "pose_only"
GAME = "game"
POSE_GUIDE = "pose_guide"

WINDOW_W, WINDOW_H = 800, 600  # updated at startup
FULLSCREEN = True


# ── Scaling helpers ──
def _sw(v: float) -> int:
    """Scale a value by screen width ratio (ref 800)."""
    return max(1, int(v * WINDOW_W / 800))

def _sh(v: float) -> int:
    """Scale a value by screen height ratio (ref 600)."""
    return max(1, int(v * WINDOW_H / 600))


def draw_menu(screen: pygame.Surface, fonts: dict, button_hover: int):
    screen.fill((15, 15, 25))
    # Title
    title = fonts["title"].render("FitDodge", True, (80, 220, 255))
    screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, _sh(80)))
    subtitle = fonts["small"].render("Move your body, dodge the blocks!", True, (180, 180, 200))
    screen.blit(subtitle, (WINDOW_W // 2 - subtitle.get_width() // 2, _sh(150)))
    # Buttons
    buttons = [
        {"text": "Pose Preview", "desc": "Camera + skeleton only", "mode": POSE_ONLY},
        {"text": "Dodge Game", "desc": "Dodge obstacles with your body!", "mode": GAME},
        {"text": "Pose Challenge", "desc": "Match target poses with your body", "mode": POSE_GUIDE},
    ]
    bw, bh = _sw(360), _sh(80)
    base_y = _sh(230)
    spacing = _sh(95)
    for i, btn in enumerate(buttons):
        bx = WINDOW_W // 2 - bw // 2
        by = base_y + i * spacing
        rect = pygame.Rect(bx, by, bw, bh)
        is_hover = (button_hover == i)
        border_color = (80, 220, 255) if is_hover else (60, 60, 80)
        fill_color = (40, 40, 55) if is_hover else (25, 25, 38)
        br = _sh(12)
        pygame.draw.rect(screen, fill_color, rect, border_radius=br)
        pygame.draw.rect(screen, border_color, rect, width=max(1, _sw(2)), border_radius=br)
        label = fonts["btn"].render(btn["text"], True, (255, 255, 255) if is_hover else (200, 200, 210))
        desc = fonts["small"].render(btn["desc"], True, (150, 150, 170))
        screen.blit(label, (bx + (bw - label.get_width()) // 2, by + _sh(14)))
        screen.blit(desc, (bx + (bw - desc.get_width()) // 2, by + _sh(46)))
    footer = fonts["tiny"].render("ESC to quit  |  Move mouse to select  |  Click to enter", True, (100, 100, 120))
    screen.blit(footer, (WINDOW_W // 2 - footer.get_width() // 2, WINDOW_H - _sh(40)))


def cv2_to_pygame(frame_bgr: np.ndarray, target_w: int, target_h: int) -> pygame.Surface:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (target_w, target_h))
    frame_rotated = np.rot90(frame_resized)
    frame_flipped = np.flip(frame_rotated, axis=0)
    return pygame.surfarray.make_surface(frame_flipped)


def toggle_fullscreen(screen, fonts_dict):
    """Toggle between fullscreen and windowed mode."""
    global FULLSCREEN, WINDOW_W, WINDOW_H
    FULLSCREEN = not FULLSCREEN
    if FULLSCREEN:
        # Use (0,0) to let pygame pick native resolution
        new_screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        WINDOW_W, WINDOW_H = new_screen.get_width(), new_screen.get_height()
    else:
        WINDOW_W, WINDOW_H = 800, 600
        new_screen = pygame.display.set_mode((800, 600))
    # Rebuild scaled fonts
    fonts_dict.clear()
    fonts_dict["title"] = pygame.font.Font(None, _sh(72))
    fonts_dict["btn"]   = pygame.font.Font(None, _sh(32))
    fonts_dict["small"] = pygame.font.Font(None, _sh(24))
    fonts_dict["tiny"]  = pygame.font.Font(None, _sh(18))
    return new_screen

def run_menu(screen: pygame.Surface, cap: cv2.VideoCapture,
             pose: PoseDetector, fonts: dict) -> str:
    button_hover = -1
    clock = pygame.time.Clock()
    bw, bh = _sw(360), _sh(80)
    base_y = _sh(230)
    spacing = _sh(95)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_F11:
                    toggle_fullscreen(screen, fonts)
                    return "f11_toggle"
            if event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                button_hover = -1
                for i in range(3):
                    bx = WINDOW_W // 2 - bw // 2
                    by = base_y + i * spacing
                    if bx <= mx <= bx + bw and by <= my <= by + bh:
                        button_hover = i
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for i, mode in enumerate([POSE_ONLY, GAME, POSE_GUIDE]):
                    bx = WINDOW_W // 2 - bw // 2
                    by = base_y + i * spacing
                    if bx <= mx <= bx + bw and by <= my <= by + bh:
                        return mode
        # Camera preview (top-right corner)
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            processed, result = pose.process_frame(frame)
            if result and result.pose_landmarks:
                pose.draw_landmarks(processed, result)
            cam_w, cam_h = _sw(200), _sh(150)
            cam_surf = cv2_to_pygame(processed, cam_w, cam_h)
            dark = pygame.Surface((cam_w, cam_h), pygame.SRCALPHA)
            dark.fill((0, 0, 0, 160))
            cam_surf.blit(dark, (0, 0))
            screen.blit(cam_surf, (WINDOW_W - cam_w - _sw(20), _sh(20)))
        draw_menu(screen, fonts, button_hover)
        pygame.display.flip()
        clock.tick(30)


def run_pose_only(screen: pygame.Surface, cap: cv2.VideoCapture,
                  pose: PoseDetector, classifier: ActionClassifier, fonts: dict) -> str:
    clock = pygame.time.Clock()
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_BACKSPACE:
                    return "menu"
                if event.key == pygame.K_F11:
                    toggle_fullscreen(screen, fonts)
                    return "f11_toggle"
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        processed, result = pose.process_frame(frame)
        if result and result.pose_landmarks:
            pose.draw_landmarks(processed, result)
        landmarks_dict = pose.landmarks_to_dict(result)
        action = classifier.classify(landmarks_dict, frame_h, frame_w)
        stable = classifier.get_stable_action()
        cam_surf = cv2_to_pygame(processed, WINDOW_W, WINDOW_H)
        screen.blit(cam_surf, (0, 0))
        # HUD bar
        bar_h = _sh(50)
        bar = pygame.Surface((WINDOW_W, bar_h), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 120))
        screen.blit(bar, (0, 0))
        mode_text = fonts["small"].render("POSE PREVIEW", True, (80, 220, 255))
        action_text = fonts["small"].render(f"Action: {stable.upper()}", True, (100, 255, 100))
        hint_text = fonts["tiny"].render("ESC: quit  |  Backspace: menu", True, (150, 150, 170))
        screen.blit(mode_text, (_sw(20), _sh(12)))
        screen.blit(action_text, (WINDOW_W // 2 - action_text.get_width() // 2, _sh(12)))
        screen.blit(hint_text, (WINDOW_W - hint_text.get_width() - _sw(20), _sh(16)))
        pygame.display.flip()
        clock.tick(30)


def show_tutorial(screen: pygame.Surface, fonts: dict, first_time: bool) -> bool:
    title_font = pygame.font.Font(None, _sh(56))
    heading_font = pygame.font.Font(None, _sh(28))
    body_font = pygame.font.Font(None, _sh(22))
    clock = pygame.time.Clock()
    slides = [
        {
            "title": "How to Play",
            "items": [
                ("Stand back", "Position yourself 1.5-2m from the camera, full body visible."),
                ("", ""),
                ("Lean to move", "<--  Lean left  -->  Lean right  -->"),
                ("", ""),
                ("Jump!", "Jump in place to make your character leap over obstacles."),
            ],
        },
        {
            "title": "Game Rules",
            "items": [
                ("Dodge obstacles", "Avoid the falling red blocks by moving and jumping."),
                ("Gold blocks", "are bonuses -- collect them for extra points!"),
                ("Build combos", "Dodge multiple blocks in a row to build your combo score."),
                ("Getting hit", "resets your combo but the game keeps going. Misses are tracked."),
            ],
        },
    ]
    slide_idx = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    slide_idx += 1
                    if slide_idx >= len(slides):
                        return True
        screen.fill((15, 15, 25))
        slide = slides[slide_idx]
        title_surf = title_font.render(slide["title"], True, (80, 220, 255))
        screen.blit(title_surf, (WINDOW_W // 2 - title_surf.get_width() // 2, _sh(50)))
        y = _sh(160)
        for label, desc in slide["items"]:
            if label:
                lbl = heading_font.render(label, True, (255, 220, 100))
                screen.blit(lbl, (_sw(120), y))
                y += _sh(32)
            if desc:
                dsc = body_font.render(desc, True, (200, 200, 210))
                screen.blit(dsc, (_sw(140), y))
                y += _sh(28)
            y += _sh(10)
        dot_y = WINDOW_H - _sh(80)
        for i in range(len(slides)):
            cx = WINDOW_W // 2 - _sw(15) + i * _sw(30)
            color = (80, 220, 255) if i == slide_idx else (60, 60, 80)
            pygame.draw.circle(screen, color, (cx, dot_y), _sh(6))
        if slide_idx < len(slides) - 1:
            footer_text = "Press ENTER or SPACE to continue  |  ESC to go back"
        else:
            footer_text = "Press ENTER or SPACE to start!  |  ESC to go back"
        footer = fonts["tiny"].render(footer_text, True, (120, 120, 140))
        screen.blit(footer, (WINDOW_W // 2 - footer.get_width() // 2, WINDOW_H - _sh(30)))
        pygame.display.flip()
        clock.tick(30)


def run_pose_guide(screen: pygame.Surface, cap: cv2.VideoCapture,
                   pose: PoseDetector, fonts: dict) -> str:
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    guide = PoseGuide()
    clock = pygame.time.Clock()
    while cap.isOpened():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_BACKSPACE:
                    return "menu"
                if event.key == pygame.K_F11:
                    toggle_fullscreen(screen, fonts)
                    return "f11_toggle"
                if event.key == pygame.K_r:
                    guide.reset()
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        processed, result = pose.process_frame(frame)
        if result and result.pose_landmarks:
            pose.draw_landmarks(processed, result)
        guide.update(result, processed.shape[1], processed.shape[0])
        frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (WINDOW_W, WINDOW_H))
        frame_rotated = np.rot90(frame_resized)
        frame_flipped = np.flip(frame_rotated, axis=0)
        cam_surf = pygame.surfarray.make_surface(frame_flipped)
        guide.draw_targets(cam_surf, WINDOW_W, WINDOW_H)
        screen.blit(cam_surf, (0, 0))
        guide.draw_hud(screen, fonts)
        hint = fonts["tiny"].render("ESC: quit  |  Backspace: menu  |  R: restart", True, (120, 120, 140))
        screen.blit(hint, (int(WINDOW_W * 0.012), WINDOW_H - int(WINDOW_H * 0.037)))
        pygame.display.flip()
        clock.tick(30)
    return "menu"


def run_game(screen: pygame.Surface, cap: cv2.VideoCapture,
             pose: PoseDetector, classifier: ActionClassifier,
             effects: EffectsManager, game: DodgeGame, fonts: dict) -> str:
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    action_cooldown = 0
    last_triggered_action = "idle"
    game.set_effects(effects)
    while cap.isOpened() and game.running:
        nav = game.handle_input()
        if nav:
            if nav == "f11_toggle":
                toggle_fullscreen(screen, fonts)
                return "f11_toggle"
            return nav
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        processed_frame, result = pose.process_frame(frame)
        landmarks_dict = pose.landmarks_to_dict(result)
        action = classifier.classify(landmarks_dict, frame_h, frame_w)
        stable_action = classifier.get_stable_action()
        game_action = "idle"
        if stable_action in ("left_lean", "right_lean"):
            game_action = stable_action
        elif stable_action == "jump":
            game_action = "jump"
        if action_cooldown <= 0 and action not in ("idle", "no_person", "partial_visible"):
            if action != last_triggered_action or action == "jump":
                pcx = int(game.player.rect.centerx)
                pcy = int(game.player.rect.centery)
                effects.burst(pcx, pcy, action, count=25)
                if action == "jump":
                    effects.trigger_shake(duration=6, intensity=5)
                if action == "squat":
                    effects.trigger_flash(duration=8, color=(50, 200, 50))
                action_cooldown = 10
                last_triggered_action = action
        if action_cooldown > 0:
            action_cooldown -= 1
        game.update(game_action)
        effects.update()
        game.draw(effects)
        # Screen shake
        sx, sy = effects.apply_shake(game.screen.get_rect())
        if sx != 0 or sy != 0:
            shake_surface = game.screen.copy()
            game.screen.fill((0, 0, 0))
            game.screen.blit(shake_surface, (sx, sy))
        # PiP camera
        pip_w = int(WINDOW_W * 0.2)
        pip_h = int(pip_w * 0.75)
        cam_small = cv2.resize(processed_frame, (pip_w, pip_h))
        if result and result.pose_landmarks:
            pose.draw_landmarks(cam_small, result)
        cam_surf = cv2_to_pygame(cam_small, pip_w, pip_h)
        pip_x = WINDOW_W - pip_w - int(WINDOW_W * 0.012)
        pip_y = WINDOW_H - pip_h - int(WINDOW_H * 0.035)
        game.screen.blit(cam_surf, (pip_x, pip_y))
        # Status label
        status_font = pygame.font.Font(None, int(WINDOW_H * 0.03))
        status_text = status_font.render(stable_action.upper(), True, (100, 255, 100))
        game.screen.blit(status_text, (pip_x, pip_y - int(WINDOW_H * 0.04)))
        # Hint
        hint = fonts["tiny"].render("ESC: quit  |  Backspace: menu  |  R: restart", True, (120, 120, 140))
        game.screen.blit(hint, (int(WINDOW_W * 0.012), WINDOW_H - int(WINDOW_H * 0.037)))
        pygame.display.flip()
        game.clock.tick(30)
    return "menu"


def main():
    global WINDOW_W, WINDOW_H
    print("[FitDodge] Starting...")
    print("  Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera: {frame_w}x{frame_h}")
    # Init fullscreen
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    WINDOW_W, WINDOW_H = screen.get_width(), screen.get_height()
    pygame.display.set_caption("FitDodge - Move Your Body!")
    # Fonts (scaled)
    fonts = {
        "title": pygame.font.Font(None, _sh(72)),
        "btn":   pygame.font.Font(None, _sh(32)),
        "small": pygame.font.Font(None, _sh(24)),
        "tiny":  pygame.font.Font(None, _sh(18)),
    }
    pose = None
    classifier = None
    effects = None
    dodge_game = None
    state = MENU
    running = True
    tutorial_shown = False
    while running and cap.isOpened():
        if state == MENU:
            if pose is None:
                pose = PoseDetector()
                classifier = ActionClassifier()
                effects = EffectsManager()
            # Recreate game on return to menu for clean state
            dodge_game = DodgeGame(WINDOW_W, WINDOW_H)
            result = run_menu(screen, cap, pose, fonts)
            if result == "f11_toggle":
                screen = pygame.display.get_surface()
                continue
            next_state = result
            if next_state == "quit":
                running = False
            else:
                state = next_state
        elif state == POSE_ONLY:
            result = run_pose_only(screen, cap, pose, classifier, fonts)
            if result == "f11_toggle":
                screen = pygame.display.get_surface()
                continue
            next_state = result
            if next_state == "quit":
                running = False
            else:
                state = next_state
        elif state == POSE_GUIDE:
            result = run_pose_guide(screen, cap, pose, fonts)
            if result == "f11_toggle":
                screen = pygame.display.get_surface()
                continue
            next_state = result
            if next_state == "quit":
                running = False
            else:
                state = next_state
        elif state == GAME:
            if dodge_game is None:
                dodge_game = DodgeGame(WINDOW_W, WINDOW_H)
            if not tutorial_shown:
                if not show_tutorial(screen, fonts, True):
                    state = MENU
                    continue
                tutorial_shown = True
            effects.particles.clear()
            result = run_game(screen, cap, pose, classifier, effects, dodge_game, fonts)
            if result == "f11_toggle":
                screen = pygame.display.get_surface()
                dodge_game = DodgeGame(WINDOW_W, WINDOW_H)
                dodge_game.set_effects(effects)
                continue
            if result == "quit":
                running = False
            else:
                state = result
    cap.release()
    if pose:
        pose.release()
    pygame.quit()
    print("[FitDodge] Exited.")
    sys.exit(0)


if __name__ == "__main__":
    main()
