import numpy as np
import torch

from env import GHZGame
from utils import Buffer


class Trainer:
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.env = GHZGame(seed=config.seed)
        self.eval_env = GHZGame(seed=config.seed + 1)
        self.buffer = Buffer(config.n_agents)

        self.episode_rewards = []
        self.win_history = []
        self.best_win_rate = 0.0

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

    def train(self):
        cfg = self.config
        print(f"Training on {cfg.device}  (seed={cfg.seed})")
        print(f"{'Episode':<10} {'AvgReward':<12} {'WinRate':<10} {'BestWR':<10} {'EntCoef':<10}")
        print("-" * 62)

        for episode in range(1, cfg.max_episodes + 1):
            avg_reward, train_wr = self.collect_episodes(1)

            self.episode_rewards.append(avg_reward)
            self.win_history.append(train_wr)

            if len(self.buffer) >= cfg.batch_size:
                self.agent.update(self.buffer)
                self.buffer.clear()
                self._decay_entropy()
                self._decay_lr()

            if episode % cfg.log_interval == 0:
                recent_wr = np.mean(self.win_history[-cfg.log_interval:])
                recent_r = np.mean(self.episode_rewards[-cfg.log_interval:])
                print(f"{episode:<10} {recent_r:<12.3f} {recent_wr:<10.3f} {self.best_win_rate:<10.3f} {self.agent.entropy_coef:<10.5f}")

            if episode % cfg.eval_interval == 0:
                eval_wr, details = self.evaluate(cfg.eval_episodes)
                print(f"  >>> Eval @ {episode}: WR={eval_wr:.4f}")
                for s, wr in details.items():
                    print(f"      {s}: {wr:.4f}")

                if eval_wr > self.best_win_rate:
                    self.best_win_rate = eval_wr
                    self.agent.save("best_model.pt")
                    print(f"  >>> New best model saved (WR={eval_wr:.4f})")

        print("\n" + "=" * 50)
        final_wr, details = self.evaluate(cfg.eval_episodes)
        print(f"Final win rate: {final_wr:.4f}  (theoretical max: 0.7500)")
        print(f"Best win rate:  {self.best_win_rate:.4f}")
        for s, wr in details.items():
            print(f"  {s}: {wr:.4f}")

        self.agent.save("final_model.pt")
        return self.best_win_rate
