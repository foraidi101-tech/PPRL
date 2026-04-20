from __future__ import annotations

import heapq
from typing import Optional

from planner.grid_map import GridMap, Position


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct(came_from: dict[Position, Position], start: Position, goal: Position) -> list[Position]:
    path = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_search(grid: GridMap, start: Position, goal: Position) -> Optional[list[Position]]:
    open_heap: list[tuple[int, Position]] = []
    heapq.heappush(open_heap, (0, start))
    g_score: dict[Position, int] = {start: 0}
    came_from: dict[Position, Position] = {}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return reconstruct(came_from, start, goal)

        for nb in grid.neighbors4(current):
            tentative = g_score[current] + 1
            if tentative < g_score.get(nb, 10**9):
                came_from[nb] = current
                g_score[nb] = tentative
                f_score = tentative + manhattan(nb, goal)
                heapq.heappush(open_heap, (f_score, nb))
    return None

