"""
Train three MADRL agents to play the GHZ game cooperatively.

Theoretical maximum classical win rate: 75%.
"""

import random
import numpy as np
import torch

from config import Config
from agents import MAPPOAgent
from trainers import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cfg = Config()
    set_seed(cfg.seed)

    agent = MAPPOAgent(
        n_agents=cfg.n_agents,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        config=cfg,
    )

    trainer = Trainer(agent, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
