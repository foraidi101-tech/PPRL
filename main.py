from planner.config import Config
from planner.simulation import evaluate_and_save_gif, train


def main() -> None:
    cfg = Config()
    planner = train(cfg)
    evaluate_and_save_gif(cfg, planner, output_path="outputs/multi_agent.gif", fps=5)


if __name__ == "__main__":
    main()
