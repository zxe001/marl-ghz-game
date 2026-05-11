"""
Evaluate a trained model and inspect the learned strategies.
"""

import argparse
import numpy as np
import torch

from config import Config
from agents import MAPPOAgent
from env import GHZGame


def inspect_policy(agent, cfg):
    """Print the learned policy for each agent."""
    agent_ids = torch.arange(cfg.n_agents, dtype=torch.long, device=cfg.device)
    names = ["Alice", "Bob  ", "Carol"]

    print("\nLearned strategies:")
    print("=" * 50)
    for label, val in [("X", 0.0), ("Y", 1.0)]:
        obs_t = torch.full((cfg.n_agents, 1), val, device=cfg.device)
        with torch.no_grad():
            logits = agent.actor(obs_t, agent_ids)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        print(f"\nWhen seeing '{label}':")
        for i in range(cfg.n_agents):
            print(f"  {names[i]}: + ({probs[i][0]:.4f})  /  - ({probs[i][1]:.4f})")

    # Show deterministic strategies
    print("\n" + "=" * 50)
    print("Deterministic strategies (argmax):")
    for i in range(cfg.n_agents):
        strat = []
        obs_X = torch.tensor([[0.0]], device=cfg.device)
        obs_Y = torch.tensor([[1.0]], device=cfg.device)
        ids = torch.tensor([i], dtype=torch.long, device=cfg.device)
        with torch.no_grad():
            a_X = agent.actor(obs_X, ids).argmax(dim=-1).item()
            a_Y = agent.actor(obs_Y, ids).argmax(dim=-1).item()
        strat_X = "+" if a_X == 0 else "-"
        strat_Y = "+" if a_Y == 0 else "-"
        print(f"  {names[i]}: see X → {strat_X}, see Y → {strat_Y}")


def evaluate_model(model_path, n_episodes=10000):
    cfg = Config()

    agent = MAPPOAgent(
        n_agents=cfg.n_agents,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        config=cfg,
    )
    agent.load(model_path)

    inspect_policy(agent, cfg)

    env = GHZGame()
    wins = 0
    scenario_wins = {s: 0 for s in GHZGame.SCENARIOS}
    scenario_counts = {s: 0 for s in GHZGame.SCENARIOS}

    agent_ids = torch.arange(cfg.n_agents, dtype=torch.long, device=cfg.device)

    for _ in range(n_episodes):
        obs, scenario = env.reset()
        obs_t = torch.FloatTensor([[o[0] for o in obs]]).T.to(cfg.device)
        with torch.no_grad():
            logits = agent.actor(obs_t, agent_ids)
            actions = logits.argmax(dim=-1).tolist()

        _, _, info = env.step(actions)
        scenario_counts[scenario] += 1
        if info["win"]:
            wins += 1
            scenario_wins[scenario] += 1

    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"  Overall win rate: {wins / n_episodes:.4f}  (theoretical max: 0.7500)")
    for s in GHZGame.SCENARIOS:
        wr = scenario_wins[s] / scenario_counts[s] if scenario_counts[s] > 0 else 0
        print(f"  {s}: {wr:.4f}  ({scenario_wins[s]}/{scenario_counts[s]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best_model.pt", help="Model path")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of eval episodes")
    args = parser.parse_args()

    evaluate_model(args.model, args.episodes)
