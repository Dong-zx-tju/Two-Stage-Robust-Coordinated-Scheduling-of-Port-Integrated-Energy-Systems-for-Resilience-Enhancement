"""
Command-line entry point to train the DQN-MASK agent on the simplified
shipboard power system environment.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sps_reconfig.dqn_mask import DQNConfig, train_agent
from sps_reconfig.environment import default_env


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN-MASK for ship power reconfiguration")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device")
    parser.add_argument("--save", type=Path, default=Path("artifacts"), help="Directory to save metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    env = default_env(seed=args.seed)
    cfg = DQNConfig(device=args.device)
    agent, returns = train_agent(env, episodes=args.episodes, cfg=cfg)

    args.save.mkdir(parents=True, exist_ok=True)
    metrics = {"returns": returns, "episodes": args.episodes}
    (args.save / "training_curve.json").write_text(json.dumps(metrics, indent=2))
    print(f"Training complete. Average return over last 10 episodes: {np.mean(returns[-10:]):.2f}")
    print(f"Saved metrics to {args.save / 'training_curve.json'}")


if __name__ == "__main__":
    main()
