"""
Run MARL training with multiple random seeds to assess convergence stability.
Saves detailed metrics (JSON) under results/multi_seed/seed_{N}/.
"""
import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from agents import MAPPOAgent
from trainers import Trainer


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "multi_seed")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_seed(seed):
    cfg = Config()
    cfg.seed = seed
    set_seed(seed)

    log_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
    agent = MAPPOAgent(
        n_agents=cfg.n_agents,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        config=cfg,
    )
    trainer = Trainer(agent, cfg, log_dir=log_dir)
    best_wr = trainer.train()
    print(f"\nSeed {seed} done — best win rate: {best_wr:.4f}")
    return best_wr


def main():
    seeds = [42, 123, 456, 789, 1024]
    print(f"Running {len(seeds)} seeds: {seeds}")
    print(f"Results will be saved to: {RESULTS_DIR}\n")

    results = {}
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")
        wr = run_seed(seed)
        results[seed] = wr

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for seed, wr in results.items():
        print(f"  seed={seed}: best WR = {wr:.4f}")
    wrs = list(results.values())
    print(f"  mean ± std: {np.mean(wrs):.4f} ± {np.std(wrs):.4f}")


if __name__ == "__main__":
    main()
