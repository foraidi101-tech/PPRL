from __future__ import annotations

import random
from dataclasses import dataclass

from planner.grid_map import Position


ACTIONS = [
    (0, 0),   # wait
    (1, 0),   # right
    (-1, 0),  # left
    (0, 1),   # up
    (0, -1),  # down
]


def sign(v: int) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


@dataclass
class RLHyper:
    alpha: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float


class QLocalAvoider:
    def __init__(self, hp: RLHyper, seed: int = 7):
        self.hp = hp
        self.epsilon = hp.epsilon_start
        self.q_table: dict[tuple[int, int, int, int], list[float]] = {}
        self._rng = random.Random(seed)

    def encode_state(self, current: Position, goal: Position, danger: tuple[int, int]) -> tuple[int, int, int, int]:
        gx, gy = goal[0] - current[0], goal[1] - current[1]
        return sign(gx), sign(gy), danger[0], danger[1]

    def _ensure_state(self, state: tuple[int, int, int, int]) -> None:
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in ACTIONS]

    def choose_action(self, state: tuple[int, int, int, int], greedy_only: bool = False) -> int:
        self._ensure_state(state)
        if not greedy_only and self._rng.random() < self.epsilon:
            return self._rng.randrange(len(ACTIONS))
        values = self.q_table[state]
        best = max(range(len(ACTIONS)), key=lambda i: values[i])
        return best

    def update(
        self,
        state: tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int, int, int],
    ) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)
        q_sa = self.q_table[state][action]
        target = reward + self.hp.gamma * max(self.q_table[next_state])
        self.q_table[state][action] = q_sa + self.hp.alpha * (target - q_sa)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.hp.epsilon_end, self.epsilon * self.hp.epsilon_decay)

