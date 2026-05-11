import json
import os
import numpy as np
import torch

from env import GHZGame
from utils import Buffer


class Trainer:
    def __init__(self, agent, config, log_dir=None):
        self.agent = agent
        self.config = config
        self.env = GHZGame(seed=config.seed)
        self.eval_env = GHZGame(seed=config.seed + 1)
        self.buffer = Buffer(config.n_agents)
        self.log_dir = log_dir
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        self.episode_rewards = []
        self.win_history = []
        self.best_win_rate = 0.0

        # Detailed logging
        self.log = {
            "config": {k: v for k, v in vars(config).items()
                       if not k.startswith("_") and not callable(v)},
            "episodes": [],
            "evaluations": [],
            "updates": [],
        }

    def _save_log(self):
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "metrics.json"), "w") as f:
                json.dump(self.log, f, indent=2)

    def collect_episodes(self, n_episodes):
        total_reward = 0.0
        wins = 0

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            actions, log_probs, value = self.agent.get_actions(obs)
            rewards, _, info = self.env.step(actions)

            self.buffer.store(obs, actions, log_probs, rewards[0], value)
            total_reward += rewards[0]
            if info["win"]:
                wins += 1

        return total_reward / n_episodes, wins / n_episodes

    def _decay_entropy(self):
        cfg = self.config
        self.agent.entropy_coef = max(
            cfg.entropy_min,
            self.agent.entropy_coef * cfg.entropy_decay,
        )

    def _decay_lr(self):
        for opt in [self.agent.actor_optimizer, self.agent.critic_optimizer]:
            for param_group in opt.param_groups:
                param_group["lr"] *= self.config.lr_decay

    @torch.no_grad()
    def evaluate(self, n_episodes):
        wins = 0
        scenario_wins = {s: 0 for s in GHZGame.SCENARIOS}
        scenario_counts = {s: 0 for s in GHZGame.SCENARIOS}

        for _ in range(n_episodes):
            obs, scenario = self.eval_env.reset()
            actions, _, _ = self.agent.get_actions(obs)
            _, _, info = self.eval_env.step(actions)

            scenario_counts[scenario] += 1
            if info["win"]:
                wins += 1
                scenario_wins[scenario] += 1

        win_rate = wins / n_episodes
        details = {}
        for s in GHZGame.SCENARIOS:
            if scenario_counts[s] > 0:
                details[s] = scenario_wins[s] / scenario_counts[s]

        return win_rate, details

    @torch.no_grad()
    def _get_policy_probs(self):
        """Record current policy probabilities for each agent, each observation."""
        probs = {}
        agent_ids = torch.arange(self.config.n_agents, dtype=torch.long,
                                 device=self.config.device)
        names = ["Alice", "Bob", "Carol"]
        for label, val in [("X", 0.0), ("Y", 1.0)]:
            obs_t = torch.full((self.config.n_agents, 1), val,
                               device=self.config.device)
            logits = self.agent.actor(obs_t, agent_ids)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            for i, name in enumerate(names):
                probs[f"{name}_{label}_+"] = float(p[i][0])
                probs[f"{name}_{label}_-"] = float(p[i][1])
        return probs

    def train(self):
        cfg = self.config
        print(f"Training on {cfg.device}  (seed={cfg.seed})")
        print(f"{'Episode':<10} {'AvgReward':<12} {'WinRate':<10} "
              f"{'BestWR':<10} {'EntCoef':<10}")
        print("-" * 62)

        for episode in range(1, cfg.max_episodes + 1):
            avg_reward, train_wr = self.collect_episodes(1)

            self.episode_rewards.append(avg_reward)
            self.win_history.append(train_wr)

            if len(self.buffer) >= cfg.batch_size:
                update_stats = self.agent.update(self.buffer)
                self.buffer.clear()
                self._decay_entropy()
                self._decay_lr()

                self.log["updates"].append({
                    "episode": episode,
                    "actor_loss": update_stats.get("actor_loss", 0),
                    "critic_loss": update_stats.get("critic_loss", 0),
                    "entropy": update_stats.get("entropy", 0),
                    "entropy_coef": self.agent.entropy_coef,
                    "lr": self.agent.actor_optimizer.param_groups[0]["lr"],
                })

            if episode % cfg.log_interval == 0:
                recent_wr = np.mean(self.win_history[-cfg.log_interval:])
                recent_r = np.mean(self.episode_rewards[-cfg.log_interval:])
                print(f"{episode:<10} {recent_r:<12.3f} {recent_wr:<10.3f} "
                      f"{self.best_win_rate:<10.3f} {self.agent.entropy_coef:<10.5f}")

            if episode % cfg.eval_interval == 0:
                eval_wr, details = self.evaluate(cfg.eval_episodes)
                policy_probs = self._get_policy_probs()
                eval_entry = {
                    "episode": episode,
                    "win_rate": eval_wr,
                    "per_scenario": details,
                    "policy_probs": policy_probs,
                }
                self.log["evaluations"].append(eval_entry)
                print(f"  >>> Eval @ {episode}: WR={eval_wr:.4f}")
                for s, wr in details.items():
                    print(f"      {s}: {wr:.4f}")

                if eval_wr > self.best_win_rate:
                    self.best_win_rate = eval_wr
                    self.agent.save(os.path.join(self.log_dir, "best_model.pt")
                                    if self.log_dir else "best_model.pt")
                    print(f"  >>> New best model saved (WR={eval_wr:.4f})")

            # Log per-episode data
            self.log["episodes"].append({
                "episode": episode,
                "reward": avg_reward,
                "win": train_wr,
            })

            # Periodic save of log
            if episode % 1000 == 0:
                self._save_log()

        print("\n" + "=" * 50)
        final_wr, details = self.evaluate(cfg.eval_episodes)
        print(f"Final win rate: {final_wr:.4f}  (theoretical max: 0.7500)")
        print(f"Best win rate:  {self.best_win_rate:.4f}")
        for s, wr in details.items():
            print(f"  {s}: {wr:.4f}")

        self.agent.save(os.path.join(self.log_dir, "final_model.pt")
                        if self.log_dir else "final_model.pt")
        self.log["final_win_rate"] = final_wr
        self.log["best_win_rate"] = self.best_win_rate
        self._save_log()
        return self.best_win_rate
