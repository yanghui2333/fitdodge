"""Visual effects: particles, screen shake, action feedback overlay, floating text."""

import random
import math
from typing import List, Tuple
import pygame


class Particle:
    """Single particle with position, velocity, lifetime, and color."""
    def __init__(self, x: float, y: float, vx: float, vy: float,
                 lifetime: int, color: Tuple[int, int, int], size: int = 4):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.color = color
        self.size = size

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.3
        self.lifetime -= 1
        self.size = max(1, int(self.size * 0.97))

    @property
    def alive(self) -> bool:
        return self.lifetime > 0

    def draw(self, surface: pygame.Surface):
        alpha = int(255 * self.lifetime / self.max_lifetime)
        color = (min(255, self.color[0]), min(255, self.color[1]), min(255, self.color[2]))
        s = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (self.size, self.size), self.size)
        surface.blit(s, (int(self.x - self.size), int(self.y - self.size)))


class EffectsManager:
    """Manages particles, screen shake, flash, and floating text."""

    ACTION_COLORS = {
        "jump":       [(255, 220, 50), (255, 180, 30), (255, 255, 100)],
        "squat":      [(50, 200, 50), (30, 180, 30), (100, 255, 100)],
        "left_lean":  [(100, 150, 255), (80, 120, 255), (150, 180, 255)],
        "right_lean": [(255, 130, 50), (255, 100, 30), (255, 160, 80)],
        "arm_raise":  [(200, 100, 255), (180, 80, 255), (220, 150, 255)],
    }
    DEFAULT_COLORS = [(255, 255, 255), (200, 200, 200)]

    def __init__(self):
        self.particles: List[Particle] = []
        self.shake_offset = (0, 0)
        self.shake_duration = 0
        self.shake_intensity = 8
        self.flash_alpha = 0
        self.flash_color = (255, 255, 255)
        self.action_feedback: List[Tuple[str, int, int]] = []
        # Floating texts: (text, remain_frames, color, x_offset)
        self.floating_texts: List[Tuple[str, int, Tuple[int,int,int], int, int]] = []

    def burst(self, x: int, y: int, action: str, count: int = 20):
        colors = self.ACTION_COLORS.get(action, self.DEFAULT_COLORS)
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 3
            lifetime = random.randint(15, 40)
            color = random.choice(colors)
            self.particles.append(Particle(x, y, vx, vy, lifetime, color, random.randint(2, 6)))

    def trigger_shake(self, duration: int = 10, intensity: int = 8):
        self.shake_duration = duration
        self.shake_intensity = intensity

    def trigger_flash(self, duration: int = 5, color: Tuple[int, int, int] = (255, 255, 255)):
        self.flash_alpha = 255
        self.flash_duration = duration
        self.flash_color = color
        self._flash_remain = duration

    def show_action_feedback(self, action: str, x: int, duration: int = 30):
        self.action_feedback.append((action.upper(), duration, x))

    def show_floating_text(self, text: str, color: Tuple[int, int, int] = (255, 255, 255),
                           x: int = -1, y: int = -1, duration: int = 30):
        """Show floating text that auto-disappears. x/y < 0 means centered on screen."""
        self.floating_texts.append((text, duration, color, x, y))

    def show_hit_feedback(self, screen_w: int, screen_h: int):
        """Show miss/hit feedback: red flash + floating text at player position."""
        self.trigger_flash(duration=6, color=(255, 60, 60))
        self.trigger_shake(duration=8, intensity=6)
        texts = ["Miss!", "Ouch!", "Watch out!", "Oops!"]
        self.show_floating_text(random.choice(texts), (255, 80, 80), screen_w//2, screen_h//2-40, 30)

    def show_dodge_feedback(self, combo: int, screen_w: int, screen_h: int):
        """Show encouragement text based on combo."""
        if combo >= 30:
            texts = ["GODLIKE!", "UNSTOPPABLE!", "LEGENDARY!"]
            color = (255, 215, 0)
        elif combo >= 20:
            texts = ["AMAZING!", "INCREDIBLE!", "ON FIRE!"]
            color = (255, 140, 0)
        elif combo >= 10:
            texts = ["Great!", "Awesome!", "Killing it!"]
            color = (100, 255, 100)
        elif combo >= 5:
            texts = ["Nice!", "Good job!", "Keep going!"]
            color = (220, 220, 100)
        else:
            return  # no feedback for low combos
        self.show_floating_text(random.choice(texts), color, screen_w//2, screen_h//2-40, 30)

    def update(self):
        for p in self.particles[:]:
            p.update()
            if not p.alive:
                self.particles.remove(p)
        if self.shake_duration > 0:
            self.shake_duration -= 1
            self.shake_offset = (random.randint(-self.shake_intensity, self.shake_intensity),
                                random.randint(-self.shake_intensity, self.shake_intensity))
        else:
            self.shake_offset = (0, 0)
        if hasattr(self, '_flash_remain') and self._flash_remain > 0:
            self._flash_remain -= 1
            self.flash_alpha = int(255 * self._flash_remain / self.flash_duration) if hasattr(self, 'flash_duration') else 0
        else:
            self.flash_alpha = 0
        for i, (text, remain, x) in enumerate(self.action_feedback[:]):
            remain -= 1
            if remain <= 0:
                self.action_feedback.pop(i)
            else:
                self.action_feedback[i] = (text, remain, x)
        for i, (text, remain, color, fx, fy) in enumerate(self.floating_texts[:]):
            remain -= 1
            if remain <= 0:
                self.floating_texts.pop(i)
            else:
                self.floating_texts[i] = (text, remain, color, fx, fy)

    def draw(self, surface: pygame.Surface):
        for p in self.particles:
            p.draw(surface)
        if self.flash_alpha > 0:
            flash = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            flash.fill((*self.flash_color, self.flash_alpha))
            surface.blit(flash, (0, 0))
        h = surface.get_height(); font48 = pygame.font.Font(None, max(24, h // 14))
        for text, remain, x in self.action_feedback:
            alpha = min(255, int(255 * remain / 30))
            surf = font48.render(text, True, (255, 255, 100, alpha))
            surf.set_alpha(alpha)
            y = 80 - (30 - remain) * 2
            surface.blit(surf, (x - surf.get_width() // 2, y))
        # Floating texts (larger, centered)
        font64 = pygame.font.Font(None, max(32, h // 10))
        for text, remain, color, fx, fy in self.floating_texts:
            alpha = min(255, int(255 * remain / self._best_duration(remain)))
            r, g, b = color
            surf = font64.render(text, True, (r, g, b))
            surf.set_alpha(alpha)
            # Scale up slightly at start
            scale = 1.0 + (1.0 - remain / 30.0) * 0.4
            if scale != 1.0:
                w = int(surf.get_width() * scale)
                h = int(surf.get_height() * scale)
                surf = pygame.transform.scale(surf, (w, h))
            cx = fx if fx >= 0 else surface.get_width() // 2
            cy = fy if fy >= 0 else surface.get_height() // 2 - 40
            surface.blit(surf, (cx - surf.get_width() // 2, cy - surf.get_height() // 2))

    @staticmethod
    def _best_duration(remain: int) -> int:
        return max(1, remain + 1)

    def apply_shake(self, surface_rect: pygame.Rect) -> Tuple[int, int]:
        return self.shake_offset
