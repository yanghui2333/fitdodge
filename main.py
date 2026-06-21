"""FitDodge - Body-motion controlled dodge game.

Modes:
  - Pose Preview: camera + skeleton overlay (no game)
  - Game Mode: dodge obstacles with body movements + hand gestures
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
from gesture_controller import GestureController

MENU = "menu"
POSE_ONLY = "pose_only"
GAME = "game"
POSE_GUIDE = "pose_guide"

WINDOW_W, WINDOW_H = 800, 600
FULLSCREEN = True


def _sw(v: float) -> int: return max(1, int(v * WINDOW_W / 800))
def _sh(v: float) -> int: return max(1, int(v * WINDOW_H / 600))


def draw_menu(screen, fonts, button_hover):
    screen.fill((15, 15, 25))
    title = fonts["title"].render("FitDodge", True, (80, 220, 255))
    screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, _sh(80)))
    subtitle = fonts["small"].render("Move your body, dodge the blocks!", True, (180, 180, 200))
    screen.blit(subtitle, (WINDOW_W // 2 - subtitle.get_width() // 2, _sh(150)))
    buttons = [
        {"text": "Pose Preview", "desc": "Camera + skeleton only", "mode": POSE_ONLY},
        {"text": "Dodge Game", "desc": "Body + hand gestures to dodge!", "mode": GAME},
        {"text": "Pose Challenge", "desc": "Match target poses with your body", "mode": POSE_GUIDE},
    ]
    bw, bh, base_y, spacing = _sw(360), _sh(80), _sh(230), _sh(95)
    for i, btn in enumerate(buttons):
        bx, by = WINDOW_W // 2 - bw // 2, base_y + i * spacing
        rect = pygame.Rect(bx, by, bw, bh)
        is_hover = (button_hover == i)
        bc = (80, 220, 255) if is_hover else (60, 60, 80)
        fc = (40, 40, 55) if is_hover else (25, 25, 38)
        br = _sh(12)
        pygame.draw.rect(screen, fc, rect, border_radius=br)
        pygame.draw.rect(screen, bc, rect, width=max(1, _sw(2)), border_radius=br)
        label = fonts["btn"].render(btn["text"], True, (255, 255, 255) if is_hover else (200, 200, 210))
        desc = fonts["small"].render(btn["desc"], True, (150, 150, 170))
        screen.blit(label, (bx + (bw - label.get_width()) // 2, by + _sh(14)))
        screen.blit(desc, (bx + (bw - desc.get_width()) // 2, by + _sh(46)))
    footer = fonts["tiny"].render("ESC to quit  |  Move mouse to select  |  Click to enter", True, (100, 100, 120))
    screen.blit(footer, (WINDOW_W // 2 - footer.get_width() // 2, WINDOW_H - _sh(40)))


def cv2_to_pygame(bgr, tw, th):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (tw, th))
    return pygame.surfarray.make_surface(np.flip(np.rot90(resized), axis=0))


def toggle_fullscreen(screen, fonts_dict):
    global FULLSCREEN, WINDOW_W, WINDOW_H
    FULLSCREEN = not FULLSCREEN
    if FULLSCREEN:
        new_screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        WINDOW_W, WINDOW_H = new_screen.get_width(), new_screen.get_height()
    else:
        WINDOW_W, WINDOW_H = 800, 600
        new_screen = pygame.display.set_mode((800, 600))
    fonts_dict.clear()
    fonts_dict["title"] = pygame.font.Font(None, _sh(72))
    fonts_dict["btn"]   = pygame.font.Font(None, _sh(32))
    fonts_dict["small"] = pygame.font.Font(None, _sh(24))
    fonts_dict["tiny"]  = pygame.font.Font(None, _sh(18))
    return new_screen


def run_menu(screen, cap, pose, fonts):
    button_hover = -1
    clock = pygame.time.Clock()
    bw, bh, base_y, spacing = _sw(360), _sh(80), _sh(230), _sh(95)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return "quit"
                if event.key == pygame.K_F11: toggle_fullscreen(screen, fonts); return "f11_toggle"
            if event.type == pygame.MOUSEMOTION:
                mx, my = event.pos; button_hover = -1
                for i in range(3):
                    bx, by = WINDOW_W // 2 - bw // 2, base_y + i * spacing
                    if bx <= mx <= bx + bw and by <= my <= by + bh: button_hover = i
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for i, mode in enumerate([POSE_ONLY, GAME, POSE_GUIDE]):
                    bx, by = WINDOW_W // 2 - bw // 2, base_y + i * spacing
                    if bx <= mx <= bx + bw and by <= my <= by + bh: return mode
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            processed, result = pose.process_frame(frame)
            if result and result.pose_landmarks: pose.draw_landmarks(processed, result)
            cw, ch = _sw(200), _sh(150)
            cs = cv2_to_pygame(processed, cw, ch)
            dark = pygame.Surface((cw, ch), pygame.SRCALPHA); dark.fill((0, 0, 0, 160))
            cs.blit(dark, (0, 0))
            screen.blit(cs, (WINDOW_W - cw - _sw(20), _sh(20)))
        draw_menu(screen, fonts, button_hover)
        pygame.display.flip(); clock.tick(30)


def run_pose_only(screen, cap, pose, classifier, fonts):
    clock = pygame.time.Clock()
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return "quit"
                if event.key == pygame.K_BACKSPACE: return "menu"
                if event.key == pygame.K_F11: toggle_fullscreen(screen, fonts); return "f11_toggle"
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        processed, result = pose.process_frame(frame)
        if result and result.pose_landmarks: pose.draw_landmarks(processed, result)
        lm = pose.landmarks_to_dict(result)
        action = classifier.classify(lm, fh, fw); stable = classifier.get_stable_action()
        screen.blit(cv2_to_pygame(processed, WINDOW_W, WINDOW_H), (0, 0))
        bh = _sh(50); bar = pygame.Surface((WINDOW_W, bh), pygame.SRCALPHA); bar.fill((0, 0, 0, 120))
        screen.blit(bar, (0, 0))
        screen.blit(fonts["small"].render("POSE PREVIEW", True, (80, 220, 255)), (_sw(20), _sh(12)))
        at = fonts["small"].render(f"Action: {stable.upper()}", True, (100, 255, 100))
        screen.blit(at, (WINDOW_W // 2 - at.get_width() // 2, _sh(12)))
        ht = fonts["tiny"].render("ESC: quit  |  Backspace: menu  |  F11: fullscreen", True, (150, 150, 170))
        screen.blit(ht, (WINDOW_W - ht.get_width() - _sw(20), _sh(16)))
        pygame.display.flip(); clock.tick(30)


def show_tutorial(screen, fonts, first_time):
    tf = pygame.font.Font(None, _sh(56)); hf = pygame.font.Font(None, _sh(28))
    bf = pygame.font.Font(None, _sh(22)); clock = pygame.time.Clock()
    slides = [
        {"title": "How to Play", "items": [
            ("Stand back", "Position yourself 1.5-2m from the camera."),
            ("", ""), ("Lean to move", "<-- Lean left / right to move -->"), ("", ""),
            ("Jump!", "Jump in place to leap over obstacles."),
            ("Hand gestures", "Wave arms left/right, raise hands to jump!")]},
        {"title": "Game Rules", "items": [
            ("Dodge obstacles", "Avoid falling red blocks."),
            ("Gold blocks", "are bonuses -- extra points!"),
            ("Build combos", "Dodge continuously for combo score."),
            ("Getting hit", "Resets combo, game keeps going.")]},
    ]
    si = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return False
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    si += 1
                    if si >= len(slides): return True
        screen.fill((15, 15, 25))
        slide = slides[si]
        ts = tf.render(slide["title"], True, (80, 220, 255))
        screen.blit(ts, (WINDOW_W // 2 - ts.get_width() // 2, _sh(50)))
        y = _sh(160)
        for label, desc in slide["items"]:
            if label:
                screen.blit(hf.render(label, True, (255, 220, 100)), (_sw(120), y)); y += _sh(32)
            if desc:
                screen.blit(bf.render(desc, True, (200, 200, 210)), (_sw(140), y)); y += _sh(28)
            y += _sh(10)
        dy = WINDOW_H - _sh(80)
        for i in range(len(slides)):
            cx = WINDOW_W // 2 - _sw(15) + i * _sw(30)
            pygame.draw.circle(screen, (80, 220, 255) if i == si else (60, 60, 80), (cx, dy), _sh(6))
        ft = "Press ENTER/SPACE to continue" if si < len(slides) - 1 else "Press ENTER/SPACE to start!"
        footer = fonts["tiny"].render(ft + "  |  ESC to go back", True, (120, 120, 140))
        screen.blit(footer, (WINDOW_W // 2 - footer.get_width() // 2, WINDOW_H - _sh(30)))
        pygame.display.flip(); clock.tick(30)


def run_pose_guide(screen, cap, pose, fonts):
    guide = PoseGuide(); clock = pygame.time.Clock()
    while cap.isOpened():
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return "quit"
                if event.key == pygame.K_BACKSPACE: return "menu"
                if event.key == pygame.K_F11: toggle_fullscreen(screen, fonts); return "f11_toggle"
                if event.key == pygame.K_r: guide.reset()
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        processed, result = pose.process_frame(frame)
        if result and result.pose_landmarks: pose.draw_landmarks(processed, result)
        guide.update(result, processed.shape[1], processed.shape[0])
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        cs = pygame.surfarray.make_surface(np.flip(np.rot90(cv2.resize(rgb, (WINDOW_W, WINDOW_H))), axis=0))
        guide.draw_targets(cs, WINDOW_W, WINDOW_H)
        screen.blit(cs, (0, 0)); guide.draw_hud(screen, fonts)
        ht = fonts["tiny"].render("G: toggle body/gesture  |  ESC: quit  |  Backspace: menu  |  F11: fullscreen  |  R: restart", True, (120, 120, 140))
        screen.blit(ht, (int(WINDOW_W * 0.012), WINDOW_H - int(WINDOW_H * 0.037)))
        pygame.display.flip(); clock.tick(30)
    return "menu"


def run_game(screen, cap, pose, classifier, gesture_ctrl, effects, game, fonts):
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cd = 0; last_triggered = "idle"
    control_mode = "body"  # "body" or "gesture"
    game.set_effects(effects)
    while cap.isOpened() and game.running:
        nav = game.handle_input()
        if nav:
            if nav == "f11_toggle": toggle_fullscreen(screen, fonts); return "f11_toggle"
            if nav == "toggle_mode":
                control_mode = "gesture" if control_mode == "body" else "body"
                continue
            return nav
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        processed_frame, result = pose.process_frame(frame)
        landmarks_dict = pose.landmarks_to_dict(result)

        # Body pose classification
        action = classifier.classify(landmarks_dict, fh, fw)
        stable_action = classifier.get_stable_action()

        # Hand gesture classification
        gesture = gesture_ctrl.classify_and_store(landmarks_dict)
        stable_gesture = gesture_ctrl.get_stable_gesture()

        # Game action: respect control mode (body / gesture)
        game_action = "idle"
        if control_mode == "body":
            if stable_action in ("left_lean", "right_lean"):
                game_action = stable_action
            elif stable_action == "jump":
                game_action = "jump"
        else:  # gesture mode (pure hand gestures)
            if stable_gesture == "move_left":
                game_action = "left_lean"
            elif stable_gesture == "move_right":
                game_action = "right_lean"
            elif stable_gesture == "hand_jump":
                game_action = "jump"
            elif stable_gesture == "fist":
                game_action = "jump"

        # Particle effects (body or gesture)
        trigger = action if action not in ("idle", "no_person", "partial_visible") else None
        if trigger is None and gesture not in ("idle", "no_person", "partial_visible"):
            trigger = gesture
        if trigger and cd <= 0:
            if trigger != last_triggered or trigger in ("jump", "hand_jump"):
                pcx, pcy = int(game.player.rect.centerx), int(game.player.rect.centery)
                en = trigger
                if trigger == "fist": en = "jump"
                elif trigger in ("move_left", "move_right"): en = "arm_raise"
                effects.burst(pcx, pcy, en, count=25)
                if trigger in ("jump", "hand_jump"): effects.trigger_shake(duration=6, intensity=5)
                if trigger == "open_palm": effects.trigger_flash(duration=8, color=(200, 200, 50))
                if trigger == "squat": effects.trigger_flash(duration=8, color=(50, 200, 50))
                cd = 10; last_triggered = trigger
        if cd > 0: cd -= 1

        game.update(game_action); effects.update(); game.draw(effects)

        # Screen shake
        sx, sy = effects.apply_shake(game.screen.get_rect())
        if sx != 0 or sy != 0:
            ss = game.screen.copy(); game.screen.fill((0, 0, 0)); game.screen.blit(ss, (sx, sy))

        # PiP camera
        pw, ph = int(WINDOW_W * 0.2), int(int(WINDOW_W * 0.2) * 0.75)
        cs = cv2.resize(processed_frame, (pw, ph))
        if result and result.pose_landmarks: pose.draw_landmarks(cs, result)
        pip = cv2_to_pygame(cs, pw, ph)
        px, py = WINDOW_W - pw - int(WINDOW_W * 0.012), WINDOW_H - ph - int(WINDOW_H * 0.035)
        game.screen.blit(pip, (px, py))

        sf = pygame.font.Font(None, int(WINDOW_H * 0.03))
        mode_label = "BODY" if control_mode == "body" else "GESTURE"
        mode_color = (100, 255, 100) if control_mode == "body" else (255, 200, 100)
        ml = sf.render(f"[{mode_label}]", True, mode_color)
        bl_s = f"  Body: {stable_action.upper()}" if control_mode == "body" else f"  Body: {stable_action.upper()}"
        gl_s = f"  Hand: {stable_gesture.upper()}" if control_mode == "gesture" else f"  Hand: {stable_gesture.upper()}"
        bl = sf.render(bl_s, True, (100, 255, 100) if control_mode == "body" else (120, 120, 120))
        gl = sf.render(gl_s, True, (255, 200, 100) if control_mode == "gesture" else (120, 120, 120))
        game.screen.blit(ml, (px, py - int(WINDOW_H * 0.12)))
        game.screen.blit(bl, (px, py - int(WINDOW_H * 0.08)))
        game.screen.blit(gl, (px, py - int(WINDOW_H * 0.04)))

        ht = fonts["tiny"].render("G: toggle body/gesture  |  ESC: quit  |  Backspace: menu  |  F11: fullscreen  |  R: restart", True, (120, 120, 140))
        game.screen.blit(ht, (int(WINDOW_W * 0.012), WINDOW_H - int(WINDOW_H * 0.037)))
        pygame.display.flip(); game.clock.tick(30)
    return "menu"


def main():
    global WINDOW_W, WINDOW_H
    print("[FitDodge] Starting...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("ERROR: Cannot open camera."); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera: {fw}x{fh}")

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    WINDOW_W, WINDOW_H = screen.get_width(), screen.get_height()
    pygame.display.set_caption("FitDodge - Move Your Body!")
    fonts = {"title": pygame.font.Font(None, _sh(72)), "btn": pygame.font.Font(None, _sh(32)),
             "small": pygame.font.Font(None, _sh(24)), "tiny": pygame.font.Font(None, _sh(18))}

    pose = classifier = gesture_ctrl = effects = dodge_game = None
    state, running, tutorial_shown = MENU, True, False

    while running and cap.isOpened():
        if state == MENU:
            if pose is None:
                pose = PoseDetector(); classifier = ActionClassifier()
                gesture_ctrl = GestureController(); effects = EffectsManager()
            dodge_game = DodgeGame(WINDOW_W, WINDOW_H)
            result = run_menu(screen, cap, pose, fonts)
            if result == "f11_toggle": screen = pygame.display.get_surface(); continue
            if result == "quit": running = False
            else: state = result

        elif state == POSE_ONLY:
            result = run_pose_only(screen, cap, pose, classifier, fonts)
            if result == "f11_toggle": screen = pygame.display.get_surface(); continue
            if result == "quit": running = False
            else: state = result

        elif state == POSE_GUIDE:
            result = run_pose_guide(screen, cap, pose, fonts)
            if result == "f11_toggle": screen = pygame.display.get_surface(); continue
            if result == "quit": running = False
            else: state = result

        elif state == GAME:
            if dodge_game is None: dodge_game = DodgeGame(WINDOW_W, WINDOW_H)
            if not tutorial_shown:
                if not show_tutorial(screen, fonts, True): state = MENU; continue
                tutorial_shown = True
            effects.particles.clear()
            result = run_game(screen, cap, pose, classifier, gesture_ctrl, effects, dodge_game, fonts)
            if result == "f11_toggle":
                screen = pygame.display.get_surface()
                dodge_game = DodgeGame(WINDOW_W, WINDOW_H)
                dodge_game.set_effects(effects)
                continue
            if result == "quit": running = False
            else: state = result

    cap.release()
    if pose: pose.release()
    pygame.quit()
    print("[FitDodge] Exited."); sys.exit(0)


if __name__ == "__main__":
    main()
