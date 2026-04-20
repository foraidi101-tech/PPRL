from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from planner.astar import astar_search, manhattan
from planner.grid_map import GridMap, Position
from planner.rl_local_avoider import ACTIONS, QLocalAvoider


@dataclass
class AgentState:
    idx: int
    start: Position
    goal: Position
    current: Position
    global_path: list[Position]
    path_ptr: int = 0
    reached: bool = False


class MultiAgentPlanner:
    def __init__(self, grid: GridMap, avoider: QLocalAvoider):
        self.grid = grid
        self.avoider = avoider
        self.agents: list[AgentState] = []

    def reset(self, starts: Iterable[Position], goals: Iterable[Position]) -> None:
        self.agents = []
        starts = list(starts)
        goals = list(goals)
        if len(starts) != len(goals):
            raise ValueError("starts and goals length mismatch")

        protected = starts + goals
        self.grid.reset_obstacles(protected=protected)

        for i, (start, goal) in enumerate(zip(starts, goals)):
            path = astar_search(self.grid, start, goal)
            if not path:
                # fallback, at least keep trying to move locally.
                path = [start, goal]
            self.agents.append(
                AgentState(
                    idx=i,
                    start=start,
                    goal=goal,
                    current=start,
                    global_path=path,
                )
            )

    def _danger_direction(self, agent: AgentState, occupied: set[Position]) -> tuple[int, int]:
        x, y = agent.current
        for dx, dy in ACTIONS[1:]:
            nb = (x + dx, y + dy)
            if nb in occupied:
                return dx, dy
        return 0, 0

    def _reward(self, prev: Position, nxt: Position, goal: Position, collision: bool, reached: bool) -> float:
        if collision:
            return -4.0
        if reached:
            return 4.0
        # Shaping: encourage approaching goal and small move cost.
        before = manhattan(prev, goal)
        after = manhattan(nxt, goal)
        return (before - after) * 0.8 - 0.08

    def step(self, train: bool = True) -> tuple[float, bool]:
        occupied = {ag.current for ag in self.agents if not ag.reached}
        proposed: dict[int, Position] = {}
        transitions: list[tuple[int, tuple[int, int, int, int], int, Position, Position, Position]] = []

        for ag in self.agents:
            if ag.reached:
                continue
            danger = self._danger_direction(ag, occupied - {ag.current})
            state = self.avoider.encode_state(ag.current, ag.goal, danger)
            action = self.avoider.choose_action(state, greedy_only=not train)
            dx, dy = ACTIONS[action]
            candidate = (ag.current[0] + dx, ag.current[1] + dy)

            # Blend global planning with local policy:
            # if local action invalid, follow next A* waypoint.
            if (not self.grid.in_bounds(candidate)) or (not self.grid.passable(candidate)):
                if ag.path_ptr + 1 < len(ag.global_path):
                    ag.path_ptr += 1
                    candidate = ag.global_path[ag.path_ptr]
                else:
                    candidate = ag.current
            proposed[ag.idx] = candidate
            transitions.append((ag.idx, state, action, ag.current, candidate, ag.goal))

        # resolve multi-agent collisions (same target conflict)
        counts: dict[Position, int] = {}
        for pos in proposed.values():
            counts[pos] = counts.get(pos, 0) + 1

        total_reward = 0.0
        for ag in self.agents:
            if ag.reached:
                continue
            candidate = proposed[ag.idx]
            collision = counts[candidate] > 1
            if collision:
                candidate = ag.current
            ag.current = candidate
            if ag.current == ag.goal:
                ag.reached = True

        if train:
            by_id = {a.idx: a for a in self.agents}
            for agent_id, state, action, prev, intended, goal in transitions:
                actual = by_id[agent_id].current
                reached = actual == goal
                collision = actual == prev and intended != prev
                danger_next = (0, 0)
                next_state = self.avoider.encode_state(actual, goal, danger_next)
                r = self._reward(prev, actual, goal, collision=collision, reached=reached)
                self.avoider.update(state, action, r, next_state)
                total_reward += r
        else:
            for _, _, _, prev, intended, goal in transitions:
                actual = intended
                if actual != prev and sum(1 for p in proposed.values() if p == actual) > 1:
                    actual = prev
                reached = actual == goal
                collision = actual == prev and intended != prev
                total_reward += self._reward(prev, actual, goal, collision=collision, reached=reached)

        done = all(ag.reached for ag in self.agents)
        return total_reward, done

