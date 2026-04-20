from __future__ import annotations

import random
from typing import Iterable


Position = tuple[int, int]


class GridMap:
    def __init__(self, width: int, height: int, obstacle_density: float = 0.15, seed: int | None = None):
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        self.seed = seed
        # seed=None means use system entropy, so each run is different.
        self._rng = random.Random(seed)
        self.obstacles: set[Position] = set()

    def in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, pos: Position) -> bool:
        return pos not in self.obstacles

    def neighbors4(self, pos: Position) -> list[Position]:
        x, y = pos
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [p for p in candidates if self.in_bounds(p) and self.passable(p)]

    def reset_obstacles(self, protected: Iterable[Position]) -> None:
        protected_set = set(protected)
        self.obstacles.clear()
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                if pos in protected_set:
                    continue
                if self._rng.random() < self.obstacle_density:
                    self.obstacles.add(pos)

    def random_free_position(self, forbidden: set[Position] | None = None) -> Position:
        forbidden = forbidden or set()
        while True:
            pos = (self._rng.randrange(self.width), self._rng.randrange(self.height))
            if pos in self.obstacles or pos in forbidden:
                continue
            return pos

