from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from planner.config import Config
from planner.grid_map import GridMap
from planner.multi_agent_system import MultiAgentPlanner
from planner.rl_local_avoider import QLocalAvoider, RLHyper


def make_problem(cfg: Config, grid: GridMap) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    starts: list[tuple[int, int]] = []
    goals: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for _ in range(cfg.n_agents):
        s = grid.random_free_position(forbidden=used)
        used.add(s)
        g = grid.random_free_position(forbidden=used)
        used.add(g)
        starts.append(s)
        goals.append(g)
    return starts, goals


def train(cfg: Config) -> MultiAgentPlanner:
    grid = GridMap(cfg.width, cfg.height, obstacle_density=cfg.obstacle_density)
    hp = RLHyper(
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
    )
    avoider = QLocalAvoider(hp=hp)
    planner = MultiAgentPlanner(grid, avoider)

    for ep in range(cfg.train_episodes):
        starts, goals = make_problem(cfg, grid)
        planner.reset(starts, goals)
        for _ in range(cfg.max_steps):
            _, done = planner.step(train=True)
            if done:
                break
        planner.avoider.decay_epsilon()
        if (ep + 1) % 250 == 0:
            print(f"[train] episode={ep + 1}, epsilon={planner.avoider.epsilon:.3f}")
    return planner


def evaluate_and_save_gif(
    cfg: Config,
    planner: MultiAgentPlanner,
    output_path: str = "outputs/multi_agent.gif",
    fps: int = 5,
) -> Path:
    starts, goals = make_problem(cfg, planner.grid)
    planner.reset(starts, goals)
    traj = [[ag.current] for ag in planner.agents]

    total_reward = 0.0
    for _ in range(cfg.max_steps):
        reward, done = planner.step(train=False)
        total_reward += reward
        for i, ag in enumerate(planner.agents):
            traj[i].append(ag.current)
        if done:
            break

    reached_count = sum(1 for ag in planner.agents if ag.reached)
    print("[eval] config:", asdict(cfg))
    print(f"[eval] reached={reached_count}/{cfg.n_agents}, total_reward={total_reward:.2f}")

    canvas = np.zeros((cfg.height, cfg.width), dtype=np.int32)
    for (x, y) in planner.grid.obstacles:
        canvas[y, x] = -1

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(canvas, cmap="Greys", origin="lower")
    cmap = plt.cm.get_cmap("tab10", cfg.n_agents)
    max_len = max(len(points) for points in traj)

    line_artists = []
    point_artists = []
    for i in range(cfg.n_agents):
        c = cmap(i)
        line, = ax.plot([], [], "-", color=c, linewidth=2, label=f"agent-{i}")
        point = ax.scatter([], [], color=c, marker="s", s=55, zorder=3)
        line_artists.append(line)
        point_artists.append(point)
        ax.scatter([starts[i][0]], [starts[i][1]], color=c, marker="o", s=45)
        ax.scatter([goals[i][0]], [goals[i][1]], color=c, marker="*", s=120)

    ax.set_title("Multi-agent Global A* + RL Local Avoidance")
    ax.set_xlim(-0.5, cfg.width - 0.5)
    ax.set_ylim(-0.5, cfg.height - 0.5)
    ax.grid(color="lightgray", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()

    def update(frame: int):
        artists = []
        for i, points in enumerate(traj):
            capped = points[: min(frame + 1, len(points))]
            xs = [p[0] for p in capped]
            ys = [p[1] for p in capped]
            line_artists[i].set_data(xs, ys)
            point_artists[i].set_offsets(np.array([[xs[-1], ys[-1]]]))
            artists.append(line_artists[i])
            artists.append(point_artists[i])
        return artists

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=max_len,
        interval=max(1, int(1000 / fps)),
        blit=True,
        repeat=False,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    ani.save(out, writer=writer)
    plt.close(fig)
    print(f"[eval] gif saved: {out.resolve()}")
    return out

