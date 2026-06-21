"""Obstacle dodge game driven by body movements. Auto-scales to screen size."""

import random
from typing import List, Tuple, Optional
import pygame


# Reference design resolution for scaling
REF_W, REF_H = 800, 600


def scl(value: float, scale: float) -> int:
    """Scale a value by the global scale factor."""
    return max(1, int(value * scale))


class Obstacle:
    def __init__(self, x: int, y: int, w: int, h: int, color: Tuple[int, int, int],
                 speed: float, obstacle_type: str = "normal", scale: float = 1.0):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = color
        self.speed = speed
        self.obstacle_type = obstacle_type
        self.passed = False
        self.scale = scale

    def update(self):
        self.rect.y += self.speed

    @property
    def off_screen(self) -> bool:
        return self.rect.y > REF_H * self.scale + scl(100, self.scale)

    def draw(self, surface: pygame.Surface):
        br = scl(8, self.scale)
        if self.obstacle_type == "bonus":
            pygame.draw.rect(surface, (255, 215, 0), self.rect, border_radius=br)
            font = pygame.font.Font(None, scl(24, self.scale))
            text = font.render("+5", True, (0, 0, 0))
            surface.blit(text, (self.rect.x + scl(6, self.scale), self.rect.y + scl(6, self.scale)))
        elif self.obstacle_type == "wide":
            pygame.draw.rect(surface, (180, 60, 60), self.rect, border_radius=scl(6, self.scale))
        elif self.obstacle_type == "fast":
            pygame.draw.rect(surface, (255, 100, 0), self.rect, border_radius=scl(4, self.scale))
        else:
            pygame.draw.rect(surface, self.color, self.rect, border_radius=scl(4, self.scale))


class Player:
    COLOR = (80, 200, 255)
    HIT_COLOR = (255, 100, 100)

    def __init__(self, game_width: int, game_height: int, scale: float = 1.0):
        self.game_width = game_width
        self.game_height = game_height
        self.scale = scale
        self.width = scl(50, scale)
        self.height = scl(70, scale)
        self.ground_offset = scl(120, scale)
        self.x = game_width // 2 - self.width // 2
        self.y = game_height - self.ground_offset
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.target_x = self.x
        self._on_ground = True
        self._jump_vel = 0
        self._jump_triggered = False
        self.hit_flash = 0

    def update_from_action(self, action: str, game_width: int):
        speed = scl(12, self.scale)
        margin = scl(20, self.scale)
        if action == "left_lean":
            self.target_x = max(margin, self.target_x - speed)
        elif action == "right_lean":
            self.target_x = min(game_width - self.width - margin, self.target_x + speed)
        elif action == "jump" and not self._jump_triggered:
            self._jump_vel = -scl(18, self.scale)
            self._on_ground = False
            self._jump_triggered = True
        elif action != "jump":
            self._jump_triggered = False

        self.x += (self.target_x - self.x) * 0.3
        gravity = 1.0 * self.scale
        ground_y = self.game_height - self.ground_offset
        self.y += self._jump_vel
        self._jump_vel += gravity
        if self.y >= ground_y:
            self.y = ground_y
            self._jump_vel = 0
            self._on_ground = True
        self.rect.x = int(self.x)
        self.rect.y = int(self.y)
        if self.hit_flash > 0:
            self.hit_flash -= 1

    def draw(self, surface: pygame.Surface):
        color = self.HIT_COLOR if self.hit_flash > 0 else self.COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=scl(8, self.scale))
        eye_y = self.rect.y + scl(15, self.scale)
        eye_offset_l = scl(15, self.scale)
        eye_offset_r = scl(35, self.scale)
        eye_r = scl(6, self.scale)
        pupil_r = scl(3, self.scale)
        pupil_offset = scl(2, self.scale)
        pygame.draw.circle(surface, (255, 255, 255), (self.rect.x + eye_offset_l, eye_y), eye_r)
        pygame.draw.circle(surface, (255, 255, 255), (self.rect.x + eye_offset_r, eye_y), eye_r)
        pygame.draw.circle(surface, (0, 0, 0), (self.rect.x + eye_offset_l + pupil_offset, eye_y), pupil_r)
        pygame.draw.circle(surface, (0, 0, 0), (self.rect.x + eye_offset_r + pupil_offset, eye_y), pupil_r)


class DodgeGame:
    INVULN_FRAMES = 30

    def __init__(self, width: int = 800, height: int = 600):
        self.WIDTH = width
        self.HEIGHT = height
        self.SCALE = height / REF_H  # uniform scale factor
        self.screen = pygame.display.get_surface()
        self.clock = pygame.time.Clock()
        fs = scl(36, self.SCALE)
        self.font = pygame.font.Font(None, max(18, fs))
        self.big_font = pygame.font.Font(None, max(36, scl(72, self.SCALE)))
        self.small_font = pygame.font.Font(None, max(14, scl(20, self.SCALE)))
        self.player = Player(self.WIDTH, self.HEIGHT, self.SCALE)
        self.obstacles: List[Obstacle] = []
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.misses = 0
        self.spawn_timer = 0
        self.spawn_interval = 40
        self.difficulty = 1.0
        self.invulnerable = 0
        self.last_action = "idle"
        self.running = True
        self._effects: Optional[object] = None

    def set_effects(self, effects):
        self._effects = effects

    def spawn_obstacle(self):
        s = self.SCALE
        w = random.randint(scl(40, s), scl(80, s))
        h = random.randint(scl(30, s), scl(50, s))
        margin = scl(50, s)
        x = random.randint(margin, self.WIDTH - w - margin)
        base_speed = scl(3, s) + self.difficulty * scl(0.5, s)
        r = random.random()
        if r < 0.1:
            obs = Obstacle(x, -h, w, h, (255, 215, 0), base_speed * 0.8, "bonus", s)
        elif r < 0.3:
            obs = Obstacle(x, -h, max(w, scl(100, s)), h, (180, 60, 60), base_speed * 0.7, "wide", s)
        elif r < 0.45:
            obs = Obstacle(x, -h, max(w - scl(10, s), scl(20, s)), h, (255, 100, 0), base_speed * 1.6, "fast", s)
        else:
            obs = Obstacle(x, -h, w, h, (200, 60, 60), base_speed, "normal", s)
        self.obstacles.append(obs)

    def update(self, player_action: str):
        self.last_action = player_action
        if self.invulnerable > 0:
            self.invulnerable -= 1
        self.spawn_timer += 1
        if self.spawn_timer >= max(15, int(self.spawn_interval / self.difficulty)):
            self.spawn_timer = 0
            self.spawn_obstacle()
        for obs in self.obstacles[:]:
            obs.update()
            if obs.off_screen:
                self.obstacles.remove(obs)
                if not obs.passed:
                    if obs.obstacle_type == "bonus":
                        self.score += 5
                    else:
                        self.score += 1
                        self.combo += 1
                        self.max_combo = max(self.max_combo, self.combo)
                        if self._effects and self.combo % 5 == 0:
                            self._effects.show_dodge_feedback(self.combo, self.WIDTH, self.HEIGHT)
        self.player.update_from_action(player_action, self.WIDTH)
        if self.invulnerable <= 0:
            for obs in self.obstacles:
                if self.player.rect.colliderect(obs.rect):
                    if obs.obstacle_type == "bonus":
                        self.score += 3
                        self.obstacles.remove(obs)
                    else:
                        self.combo = 0
                        self.misses += 1
                        self.invulnerable = self.INVULN_FRAMES
                        self.player.hit_flash = self.INVULN_FRAMES
                        if self._effects:
                            self._effects.show_hit_feedback(self.WIDTH, self.HEIGHT)
                    break
        self.difficulty = 1.0 + self.score * 0.02

    def draw(self, effects=None):
        s = self.SCALE
        # Background
        for i in range(self.HEIGHT):
            c = int(20 + i * 0.08)
            pygame.draw.line(self.screen, (c, c, c + 10), (0, i), (self.WIDTH, i))
        for obs in self.obstacles:
            obs.draw(self.screen)
        if self.invulnerable <= 0 or self.invulnerable % 6 < 3:
            self.player.draw(self.screen)
        if effects:
            effects.draw(self.screen)
        # HUD
        hud_x = scl(20, s)
        hud_action_x = self.WIDTH - scl(200, s)
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        combo_text = self.font.render(
            f"Combo: {self.combo}{' (max ' + str(self.max_combo) + ')' if self.max_combo > 0 else ''}",
            True, (255, 220, 50))
        action_text = self.font.render(f"Action: {self.last_action.upper()}", True, (100, 255, 100))
        miss_text = self.small_font.render(f"Misses: {self.misses}", True,
                                           (255, 140, 140) if self.misses > 0 else (150, 150, 150))
        self.screen.blit(score_text, (hud_x, scl(20, s)))
        self.screen.blit(combo_text, (hud_x, scl(55, s)))
        self.screen.blit(miss_text, (hud_x, scl(80, s)))
        self.screen.blit(action_text, (hud_action_x, scl(20, s)))
        if self.invulnerable > 0:
            inv_text = self.small_font.render(
                f"Invulnerable {self.invulnerable // 10 + 1}s", True, (255, 180, 80))
            self.screen.blit(inv_text, (hud_action_x, scl(50, s)))

    def handle_input(self) -> str:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return "quit"
                if event.key == pygame.K_BACKSPACE:
                    self.running = False
                    return "menu"
                if event.key == pygame.K_F11:
                    return "f11_toggle"
                if event.key == pygame.K_g:
                    return "toggle_mode"
                if event.key == pygame.K_r:
                    self.reset()
        return ""

    def reset(self):
        self.obstacles.clear()
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.misses = 0
        self.invulnerable = 0
        self.difficulty = 1.0
        self.spawn_timer = 0
        self.player = Player(self.WIDTH, self.HEIGHT, self.SCALE)
