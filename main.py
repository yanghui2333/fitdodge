"""FitDodge - Body-motion controlled dodge game.

Modes:
  - Pose Preview: camera + skeleton overlay (no game)
  - Game Mode: dodge obstacles with body movements
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

WINDOW_W, WINDOW_H = 800, 600


def draw_menu(screen: pygame.Surface, fonts: dict, button_hover: int):
    """Draw the mode selection menu."""
    screen.fill((15, 15, 25))

    # Title
    title = fonts["title"].render("FitDodge", True, (80, 220, 255))
    screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 80))

    subtitle = fonts["small"].render("Move your body, dodge the blocks!", True, (180, 180, 200))
    screen.blit(subtitle, (WINDOW_W // 2 - subtitle.get_width() // 2, 150))

    # Buttons
    buttons = [
        {"text": "Pose Preview", "desc": "Camera + skeleton only", "mode": POSE_ONLY},
        {"text": "Dodge Game", "desc": "Dodge obstacles with your body!", "mode": GAME},
        {"text": "Pose Challenge", "desc": "Match target poses with your body", "mode": POSE_GUIDE},
    ]

    base_y = 220
    for i, btn in enumerate(buttons):
        bx = WINDOW_W // 2 - 180
        by = base_y + i * 90
        bw, bh = 360, 80
        rect = pygame.Rect(bx, by, bw, bh)

        is_hover = (button_hover == i)
        border_color = (80, 220, 255) if is_hover else (60, 60, 80)
        fill_color = (40, 40, 55) if is_hover else (25, 25, 38)
        pygame.draw.rect(screen, fill_color, rect, border_radius=12)
        pygame.draw.rect(screen, border_color, rect, width=2, border_radius=12)

        label = fonts["btn"].render(btn["text"], True, (255, 255, 255) if is_hover else (200, 200, 210))
        desc = fonts["small"].render(btn["desc"], True, (150, 150, 170))
        screen.blit(label, (bx + (bw - label.get_width()) // 2, by + 14))
        screen.blit(desc, (bx + (bw - desc.get_width()) // 2, by + 46))

    # Footer
    footer = fonts["tiny"].render("ESC to quit  |  Move mouse to select  |  Click to enter", True, (100, 100, 120))
    screen.blit(footer, (WINDOW_W // 2 - footer.get_width() // 2, WINDOW_H - 40))


def cv2_to_pygame(frame_bgr: np.ndarray, target_w: int, target_h: int) -> pygame.Surface:
    """Convert BGR OpenCV frame to RGB Pygame surface."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (target_w, target_h))
    frame_rotated = np.rot90(frame_resized)
    frame_flipped = np.flip(frame_rotated, axis=0)
    return pygame.surfarray.make_surface(frame_flipped)


def run_menu(screen: pygame.Surface, cap: cv2.VideoCapture,
             pose: PoseDetector, fonts: dict) -> str:
    """Run the menu screen. Returns selected mode."""
    button_hover = -1
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "quit"
            if event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                button_hover = -1
                for i in range(3):
                    bx = WINDOW_W // 2 - 180
                    by = 260 + i * 110
                    if bx <= mx <= bx + 360 and by <= my <= by + 80:
                        button_hover = i
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for i, mode in enumerate([POSE_ONLY, GAME, POSE_GUIDE]):
                    bx = WINDOW_W // 2 - 180
                    by = 260 + i * 110
                    if bx <= mx <= bx + 360 and by <= my <= by + 80:
                        return mode

        # Read camera for background preview
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            processed, result = pose.process_frame(frame)
            if result and result.pose_landmarks:
                pose.draw_landmarks(processed, result)
            # Show dimmed camera preview as background
            cam_surf = cv2_to_pygame(processed, 200, 150)
            dark_overlay = pygame.Surface((200, 150), pygame.SRCALPHA)
            dark_overlay.fill((0, 0, 0, 160))
            cam_surf.blit(dark_overlay, (0, 0))
            screen.blit(cam_surf, (WINDOW_W - 220, 20))

        draw_menu(screen, fonts, button_hover)
        pygame.display.flip()
        clock.tick(30)


def run_pose_only(screen: pygame.Surface, cap: cv2.VideoCapture,
                  pose: PoseDetector, classifier: ActionClassifier, fonts: dict) -> str:
    """Run pose-only preview mode. Returns next state (menu/quit)."""
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

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        processed, result = pose.process_frame(frame)

        # Draw skeleton
        if result and result.pose_landmarks:
            pose.draw_landmarks(processed, result)

        # Classify action for HUD
        landmarks_dict = pose.landmarks_to_dict(result)
        action = classifier.classify(landmarks_dict, frame_h, frame_w)
        stable = classifier.get_stable_action()

        # Convert to pygame surface (full screen)
        cam_surf = cv2_to_pygame(processed, WINDOW_W, WINDOW_H)
        screen.blit(cam_surf, (0, 0))

        # HUD overlay
        # Top bar background
        bar = pygame.Surface((WINDOW_W, 50), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 120))
        screen.blit(bar, (0, 0))

        mode_text = fonts["small"].render("POSE PREVIEW", True, (80, 220, 255))
        action_text = fonts["small"].render(f"Action: {stable.upper()}", True, (100, 255, 100))
        hint_text = fonts["tiny"].render("ESC: quit  |  Backspace: menu", True, (150, 150, 170))
        screen.blit(mode_text, (20, 12))
        screen.blit(action_text, (WINDOW_W // 2 - action_text.get_width() // 2, 12))
        screen.blit(hint_text, (WINDOW_W - hint_text.get_width() - 20, 16))

        pygame.display.flip()
        clock.tick(30)



def show_tutorial(screen: pygame.Surface, fonts: dict, first_time: bool) -> bool:
    """Show tutorial overlay. Returns True to proceed, False to go back."""
    title_font = pygame.font.Font(None, 56)
    heading_font = pygame.font.Font(None, 28)
    body_font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()

    # Tutorial slides
    slides = [
        {
            "title": "How to Play",
            "items": [
                ("Stand back", "Position yourself 1.5-2m from the camera, facing it with your full body visible."),
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
                ("Getting hit", "resets your combo but the game keeps going. Miss counter tracks hits."),
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

        # Draw background
        screen.fill((15, 15, 25))

        slide = slides[slide_idx]

        # Title
        title_surf = title_font.render(slide["title"], True, (80, 220, 255))
        screen.blit(title_surf, (WINDOW_W // 2 - title_surf.get_width() // 2, 50))

        # Items
        y = 160
        for label, desc in slide["items"]:
            if label:
                lbl = heading_font.render(label, True, (255, 220, 100))
                screen.blit(lbl, (120, y))
                y += 32
            if desc:
                dsc = body_font.render(desc, True, (200, 200, 210))
                screen.blit(dsc, (140, y))
                y += 28
            y += 10

        # Slide indicators
        dot_y = WINDOW_H - 80
        for i in range(len(slides)):
            cx = WINDOW_W // 2 - 15 + i * 30
            color = (80, 220, 255) if i == slide_idx else (60, 60, 80)
            pygame.draw.circle(screen, color, (cx, dot_y), 6)

        # Footer
        if slide_idx < len(slides) - 1:
            footer_text = "Press ENTER or SPACE to continue  |  ESC to go back"
        else:
            footer_text = "Press ENTER or SPACE to start!  |  ESC to go back"
        footer = fonts["tiny"].render(footer_text, True, (120, 120, 140))
        screen.blit(footer, (WINDOW_W // 2 - footer.get_width() // 2, WINDOW_H - 30))

        pygame.display.flip()
        clock.tick(30)

def run_pose_guide(screen: pygame.Surface, cap: cv2.VideoCapture,
                   pose: PoseDetector, fonts: dict) -> str:
    """Run pose guide mode. Returns next state (menu/quit)."""
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    guide = PoseGuide()
    clock = pygame.time.Clock()

    while cap.isOpened():
        # Handle input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_BACKSPACE:
                    return "menu"
                if event.key == pygame.K_r:
                    guide.reset()

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        processed, result = pose.process_frame(frame)
        frame_h_actual, frame_w_actual = processed.shape[:2]

        # Draw skeleton
        if result and result.pose_landmarks:
            pose.draw_landmarks(processed, result)

        # Update guide logic
        guide.update(result, frame_w_actual, frame_h_actual)

        # Convert frame to pygame and use as background
        frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (WINDOW_W, WINDOW_H))
        frame_rotated = np.rot90(frame_resized)
        frame_flipped = np.flip(frame_rotated, axis=0)
        cam_surf = pygame.surfarray.make_surface(frame_flipped)

        # Draw target zones on top
        guide.draw_targets(cam_surf, WINDOW_W, WINDOW_H)

        screen.blit(cam_surf, (0, 0))

        # HUD
        guide.draw_hud(screen, fonts)

        # Bottom hints
        hint = fonts["tiny"].render("ESC: quit  |  Backspace: menu  |  R: restart", True, (120, 120, 140))
        screen.blit(hint, (10, WINDOW_H - 22))

        pygame.display.flip()
        clock.tick(30)

    return "menu"

def run_game(screen: pygame.Surface, cap: cv2.VideoCapture,
             pose: PoseDetector, classifier: ActionClassifier,
             effects: EffectsManager, game: DodgeGame, fonts: dict) -> str:
    """Run game mode. Returns next state (menu/quit)."""
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    action_cooldown = 0
    last_triggered_action = "idle"

    # Wire effects into game for hit/dodge feedback
    game.set_effects(effects)

    while cap.isOpened() and game.running:
        nav = game.handle_input()
        if nav:
            return nav

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        processed_frame, result = pose.process_frame(frame)
        landmarks_dict = pose.landmarks_to_dict(result)

        action = classifier.classify(landmarks_dict, frame_h, frame_w)
        stable_action = classifier.get_stable_action()

        # Game input
        game_action = "idle"
        if stable_action in ("left_lean", "right_lean"):
            game_action = stable_action
        elif stable_action == "jump":
            game_action = "jump"

        # Body-motion particle effects (separate from game hit/dodge feedback)
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

        # Webcam picture-in-picture
        cam_small = cv2.resize(processed_frame, (160, 120))
        if result and result.pose_landmarks:
            pose.draw_landmarks(cam_small, result)
        cam_surf = cv2_to_pygame(cam_small, 160, 120)
        game.screen.blit(cam_surf, (WINDOW_W - 170, WINDOW_H - 130))

        status_font = fonts["tiny"]
        status_text = status_font.render(stable_action.upper(), True, (100, 255, 100))
        game.screen.blit(status_text, (WINDOW_W - 170, WINDOW_H - 150))

        # Mode hint (add R to restart)
        hint = fonts["tiny"].render("ESC: quit  |  Backspace: menu  |  R: restart", True, (120, 120, 140))
        game.screen.blit(hint, (10, WINDOW_H - 22))

        pygame.display.flip()
        game.clock.tick(30)

    return "menu"


def main():
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

    # Init pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("FitDodge - Move Your Body!")

    # Fonts
    fonts = {
        "title": pygame.font.Font(None, 72),
        "btn": pygame.font.Font(None, 32),
        "small": pygame.font.Font(None, 24),
        "tiny": pygame.font.Font(None, 18),
    }

    # Init modules (lazy: game only created when needed)
    pose = None
    classifier = None
    effects = None
    dodge_game = None

    state = MENU
    running = True
    tutorial_shown = False

    while running and cap.isOpened():
        if state == MENU:
            # Lazy init (first time entering menu or after quit)
            if pose is None:
                pose = PoseDetector()
                classifier = ActionClassifier()
                effects = EffectsManager()
            # Reset game if returning from game over
            if dodge_game is not None:
                dodge_game = DodgeGame()
            next_state = run_menu(screen, cap, pose, fonts)
            if next_state == "quit":
                running = False
            else:
                state = next_state

        elif state == POSE_ONLY:
            next_state = run_pose_only(screen, cap, pose, classifier, fonts)
            if next_state == "quit":
                running = False
            else:
                state = next_state

        elif state == POSE_GUIDE:
            next_state = run_pose_guide(screen, cap, pose, fonts)
            if next_state == "quit":
                running = False
            else:
                state = next_state

        elif state == GAME:
            if dodge_game is None:
                dodge_game = DodgeGame()
            # Show tutorial on first game entry
            if not tutorial_shown:
                if not show_tutorial(screen, fonts, True):
                    state = MENU
                    continue
                tutorial_shown = True
            # Reset effects for fresh game
            effects.particles.clear()
            next_state = run_game(screen, cap, pose, classifier, effects, dodge_game, fonts)
            if next_state == "quit":
                running = False
            else:
                state = next_state

    # Shutdown
    cap.release()
    if pose:
        pose.release()
    pygame.quit()
    print(f"[FitDodge] Exited.")
    sys.exit(0)


if __name__ == "__main__":
    main()
