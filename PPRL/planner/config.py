from dataclasses import dataclass


@dataclass
class Config:
    width: int = 20
    height: int = 20
    obstacle_density: float = 0.16
    n_agents: int = 4
    max_steps: int = 160
    train_episodes: int = 1200
    alpha: float = 0.12
    gamma: float = 0.95
    epsilon_start: float = 0.35
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997

