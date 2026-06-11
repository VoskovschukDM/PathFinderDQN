"""Среда PathFinder DQN для перемещения агента по текстовой карте.

Карта состоит из символов:
- S — стартовая позиция;
- G — цель;
- H — препятствие/стена;
- остальные символы считаются свободными клетками.
"""

from typing import List, Optional, Tuple

import numpy as np


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Вычисляет манхэттенское расстояние между двумя клетками."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Enviroment:
    """Простая grid-world среда для обучения агента поиску пути."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        desc: np.ndarray,
        max_steps: int = None,
        step_penalty: float = 0.001,
        stuck_penalty: float = 0.01,
        progress_reward: float = 0.02,
        goal_reward: float = 10.0,
        hole_penalty: float = 5.0,
        is_slippery: bool = False,
        seed: Optional[int] = None,
    ):
        # Храним карту как numpy-массив символов, чтобы удобно обращаться к клеткам.
        self.desc = np.array(desc, dtype="U1")
        self.n = self.desc.shape[0]
        self.num_actions = 4  # 0 — влево, 1 — вниз, 2 — вправо, 3 — вверх.
        self.obs_n = self.n * self.n  # Каждая клетка — отдельное дискретное состояние.

        # Автоматически ищем стартовые позиции и цель на карте.
        starts = list(zip(*np.where(self.desc == "S")))
        goals = list(zip(*np.where(self.desc == "G")))
        if not starts or not goals:
            raise ValueError("На карте должны быть S (start) и G (goal)")

        self.start_positions = list(starts)
        self.goal_pos = goals[0]

        # Ограничение на длину эпизода защищает обучение от бесконечных блужданий.
        self.max_steps = max_steps or (4 * self.n * self.n)
        self.step_penalty = step_penalty
        self.stuck_penalty = stuck_penalty
        self.progress_reward = progress_reward
        self.goal_reward = goal_reward
        self.hole_penalty = hole_penalty
        self.is_slippery = is_slippery
        self._rng = np.random.default_rng(seed)

        self.pos: Optional[Tuple[int, int]] = None
        self.steps = 0

    def _to_state(self, pos: Tuple[int, int]) -> int:
        """Преобразует координаты клетки в индекс состояния."""
        return pos[0] * self.n + pos[1]

    def _is_wall(self, ch: str) -> bool:
        """Проверяет, является ли клетка препятствием."""
        return ch.upper() == "H"

    def seed(self, seed: int):
        """Переинициализирует генератор случайных чисел."""
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options=None):
        """Начинает новый эпизод и ставит агента на случайную стартовую клетку."""
        if seed is not None:
            self.seed(seed)

        self.steps = 0
        self.pos = self.start_positions[self._rng.integers(0, len(self.start_positions))]
        return self._to_state(self.pos), {"pos": self.pos}

    def step(self, action: int):
        """Выполняет действие агента и возвращает новое состояние, награду и флаги завершения."""
        self.steps += 1
        a = int(action)

        # При slippery=True действие может случайно отклониться влево или вправо.
        if self.is_slippery:
            a = int(self._rng.choice([(a - 1) % 4, a, (a + 1) % 4]))

        r, c = self.pos
        drdc = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        dr, dc = drdc[a]
        nr, nc = r + dr, c + dc

        # Не даём агенту выйти за границы карты.
        nr = min(max(nr, 0), self.n - 1)
        nc = min(max(nc, 0), self.n - 1)

        # Если впереди стена, агент остаётся на месте.
        if self._is_wall(self.desc[nr, nc]):
            nr, nc = r, c

        old_pos, self.pos = (r, c), (nr, nc)
        ch = self.desc[nr, nc]
        terminated = ch.upper() == "G"
        truncated = self.steps >= self.max_steps

        # Базовый штраф мотивирует агента искать короткий путь.
        reward = -self.step_penalty

        # Дополнительный штраф, если действие не изменило позицию.
        if self.pos == old_pos:
            reward -= self.stuck_penalty

        # Reward shaping: поощряем приближение к цели и немного штрафуем удаление.
        d0, d1 = manhattan(old_pos, self.goal_pos), manhattan(self.pos, self.goal_pos)
        if d1 < d0:
            reward += self.progress_reward
        elif d1 > d0:
            reward -= self.progress_reward * 0.25

        # Большая награда за достижение цели.
        if ch.upper() == "G":
            reward += self.goal_reward

        return self._to_state(self.pos), float(reward), bool(terminated), bool(truncated), {"pos": self.pos, "tile": ch}

    def render(self) -> str:
        """Возвращает текстовое отображение карты с текущей позицией агента A."""
        grid = self.desc.copy()
        r, c = self.pos
        grid[r, c] = "A"
        return "\n".join("".join(row) for row in grid)


def one_hot(index: int, size: int) -> np.ndarray:
    """Создаёт one-hot вектор для состояния, если понадобится в другой модели."""
    v = np.zeros(size, dtype=np.float32)
    v[index] = 1.0
    return v


def make_env(desc, max_steps=None, seed=123):
    """Фабрика среды: упрощает создание Enviroment из карты."""
    return Enviroment(desc=desc, max_steps=max_steps, seed=seed)
