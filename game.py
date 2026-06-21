"""Obstacle dodge game driven by body movements."""

import random
from typing import List, Tuple, Optional
import pygame


class Obstacle:
    """An obstacle that falls from the top."""
    def __init__(self, x: int, y: int, w: int, h: int, color: Tuple[int, int, int],
                 speed: float, obstacle_type: str = "normal"):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = color
        self.speed = speed
        self.obstacle_type = obstacle_type
        self.passed = False

    def update(self):
        self.rect.y += self.speed

    @property
    def off_screen(self) -> bool:
        return self.rect.y > 700

    def draw(self, surface: pygame.Surface):
        if self.obstacle_type == "bonus":
            pygame.draw.rect(surface, (255, 215, 0), self.rect, border_radius=8)
            font = pygame.font.Font(None, 24)
            text = font.render("+5", True, (0, 0, 0))
            surface.blit(text, (self.rect.x + 6, self.rect.y + 6))
        elif self.obstacle_type == "wide":
            pygame.draw.rect(surface, (180, 60, 60), self.rect, border_radius=6)
        elif self.obstacle_type == "fast":
            pygame.draw.rect(surface, (255, 100, 0), self.rect, border_radius=4)
        else:
            pygame.draw.rect(surface, self.color, self.rect, border_radius=4)


class Player:
    """Player character controlled by body position."""
    WIDTH = 50
    HEIGHT = 70
    COLOR = (80, 200, 255)
    HIT_COLOR = (255, 100, 100)

    def __init__(self, game_width: int, game_height: int):
        self.game_width = game_width
        self.game_height = game_height
        self.x = game_width // 2 - self.WIDTH // 2
        self.y = game_height - 120
        self.rect = pygame.Rect(self.x, self.y, self.WIDTH, self.HEIGHT)
        self.target_x = self.x
        self._on_ground = True
        self._jump_vel = 0
        self._jump_triggered = False
        self.hit_flash = 0  # frames remaining for red flash

    def update_from_action(self, action: str, game_width: int):
        speed = 12
        if action == "left_lean":
            self.target_x = max(20, self.target_x - speed)
        elif action == "right_lean":
            self.target_x = min(game_width - self.WIDTH - 20, self.target_x + speed)
        elif action == "jump" and not self._jump_triggered:
            self._jump_vel = -18
            self._on_ground = False
            self._jump_triggered = True
        elif action != "jump":
            self._jump_triggered = False

        self.x += (self.target_x - self.x) * 0.3
        gravity = 1.0
        ground_y = self.game_height - 120
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
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        eye_y = self.rect.y + 15
        pygame.draw.circle(surface, (255, 255, 255), (self.rect.x + 15, eye_y), 6)
        pygame.draw.circle(surface, (255, 255, 255), (self.rect.x + 35, eye_y), 6)
        pygame.draw.circle(surface, (0, 0, 0), (self.rect.x + 17, eye_y), 3)
        pygame.draw.circle(surface, (0, 0, 0), (self.rect.x + 37, eye_y), 3)


class DodgeGame:
    """Core game: dodge falling obstacles. Hit = feedback, not game over."""

    WIDTH = 800
    HEIGHT = 600
    INVULN_FRAMES = 30  # ~1 second of invulnerability after hit

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("FitDodge - Move Your Body!")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)
        self.small_font = pygame.font.Font(None, 20)
        self.player = Player(self.WIDTH, self.HEIGHT)
        self.obstacles: List[Obstacle] = []
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.misses = 0
        self.spawn_timer = 0
        self.spawn_interval = 40
        self.difficulty = 1.0
        self.invulnerable = 0  # invulnerability frames
        self.last_action = "idle"
        self.running = True
        self._effects: Optional[object] = None  # set via main loop

    def set_effects(self, effects):
        """Set reference to EffectsManager for feedback."""
        self._effects = effects

    def spawn_obstacle(self):
        w = random.randint(40, 80)
        h = random.randint(30, 50)
        x = random.randint(50, self.WIDTH - w - 50)
        base_speed = 3 + self.difficulty * 0.5
        r = random.random()
        if r < 0.1:
            obs = Obstacle(x, -h, w, h, (255, 215, 0), base_speed * 0.8, "bonus")
        elif r < 0.3:
            obs = Obstacle(x, -h, max(w, 100), h, (180, 60, 60), base_speed * 0.7, "wide")
        elif r < 0.45:
            obs = Obstacle(x, -h, w - 10, h, (255, 100, 0), base_speed * 1.6, "fast")
        else:
            obs = Obstacle(x, -h, w, h, (200, 60, 60), base_speed)
        self.obstacles.append(obs)

    def update(self, player_action: str):
        """Advance game state by one frame. Returns event list for effects."""
        self.last_action = player_action

        # Invulnerability tick
        if self.invulnerable > 0:
            self.invulnerable -= 1

        # Spawn obstacles
        self.spawn_timer += 1
        if self.spawn_timer >= max(15, int(self.spawn_interval / self.difficulty)):
            self.spawn_timer = 0
            self.spawn_obstacle()

        # Update obstacles
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
                        # Encouragement at milestones
                        if self._effects and self.combo % 5 == 0:
                            self._effects.show_dodge_feedback(self.combo, self.WIDTH, self.HEIGHT)

        # Update player
        self.player.update_from_action(player_action, self.WIDTH)

        # Collision check (skip if invulnerable)
        if self.invulnerable <= 0:
            for obs in self.obstacles:
                if self.player.rect.colliderect(obs.rect):
                    if obs.obstacle_type == "bonus":
                        self.score += 3
                        self.obstacles.remove(obs)
                    else:
                        # Hit! Don't end game, just feedback + penalty
                        self.combo = 0
                        self.misses += 1
                        self.invulnerable = self.INVULN_FRAMES
                        self.player.hit_flash = self.INVULN_FRAMES
                        if self._effects:
                            self._effects.show_hit_feedback(self.WIDTH, self.HEIGHT)
                    break

        self.difficulty = 1.0 + self.score * 0.02

    def draw(self, effects=None):
        # Background gradient
        for i in range(self.HEIGHT):
            c = int(20 + i * 0.08)
            pygame.draw.line(self.screen, (c, c, c + 10), (0, i), (self.WIDTH, i))

        # Obstacles
        for obs in self.obstacles:
            obs.draw(self.screen)

        # Player (blink during invulnerability)
        if self.invulnerable <= 0 or self.invulnerable % 6 < 3:
            self.player.draw(self.screen)

        # Effects
        if effects:
            effects.draw(self.screen)

        # HUD
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        combo_text = self.font.render(
            f"Combo: {self.combo}{' (max ' + str(self.max_combo) + ')' if self.max_combo > 0 else ''}",
            True, (255, 220, 50))
        action_text = self.font.render(f"Action: {self.last_action.upper()}", True, (100, 255, 100))
        miss_text = self.small_font.render(f"Misses: {self.misses}", True,
                                           (255, 140, 140) if self.misses > 0 else (150, 150, 150))
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(combo_text, (20, 55))
        self.screen.blit(miss_text, (20, 80))
        self.screen.blit(action_text, (self.WIDTH - 200, 20))

        # Invulnerability indicator
        if self.invulnerable > 0:
            inv_text = self.small_font.render(
                f"Invulnerable {self.invulnerable // 10 + 1}s", True, (255, 180, 80))
            self.screen.blit(inv_text, (self.WIDTH - 200, 50))

    def handle_input(self) -> str:
        """Handle keyboard input. Returns "quit", "menu", or ""."""
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
        self.player = Player(self.WIDTH, self.HEIGHT)
