"""
Ablation studies for the MAPPO GHZ-game agent.

Variants:
  - baseline:        MAPPO with centralized critic, entropy decay
  - no_entropy_decay: fixed entropy_coef = 0.05 throughout training
  - ippo:            independent critics (each agent has its own critic,
                     no joint-observation input)
  - small_net:       hidden_dim=16, critic_hidden_dim=32, embed_dim=2
"""
import os
import sys
import copy
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from agents import MAPPOAgent
from trainers import Trainer


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "ablations")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(variant):
    cfg = Config()
    cfg.seed = 42

    if variant == "no_entropy_decay":
        cfg.entropy_decay = 1.0  # never decay
    elif variant == "ippo":
        pass  # handled in the agent wrapper below
    elif variant == "small_net":
        cfg.embed_dim = 2
        cfg.hidden_dim = 16
        cfg.critic_hidden_dim = 32
    return cfg


class IPPOAgent(MAPPOAgent):
    """MAPPO variant where each agent has an independent critic (no centralisation)."""

    def __init__(self, n_agents, obs_dim, action_dim, config):
        super().__init__(n_agents, obs_dim, action_dim, config)
        from agents.policy_network import Critic as SingleCritic
        self.critics = torch.nn.ModuleList([
            SingleCritic(obs_dim, 1, hidden_dim=config.critic_hidden_dim // 2)
            for _ in range(n_agents)
        ])
        self.critic_optimizers = [
            torch.optim.Adam(c.parameters(), lr=config.lr_critic)
            for c in self.critics
        ]
        self.critic = None

    def _get_checkpoint(self):
        ckpt = {"actor": self.actor.state_dict()}
        for i, c in enumerate(self.critics):
            ckpt[f"critic_{i}"] = c.state_dict()
        return ckpt

    def _load_checkpoint(self, ckpt):
        self.actor.load_state_dict(ckpt["actor"])
        for i, c in enumerate(self.critics):
            c.load_state_dict(ckpt[f"critic_{i}"])

    @torch.no_grad()
    def get_actions(self, observations):
        obs_t = torch.FloatTensor([o[0] for o in observations]).unsqueeze(-1).to(self.device)
        logits = self.actor(obs_t, self.agent_ids)
        from torch.distributions import Categorical
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = [c(obs_t[i:i+1].unsqueeze(0)).item()
                  for i, c in enumerate(self.critics)]
        return actions.tolist(), log_probs.tolist(), sum(values) / len(values)

    def update(self, buffer):
        import torch.nn.functional as F
        from torch.distributions import Categorical
        data = buffer.get_all(self.device)
        n_episodes = len(buffer)
        if n_episodes == 0:
            return {}

        returns = data["rewards"]
        advantages = returns - data["values"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        n_batches = 0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(n_episodes, device=self.device)
            for start in range(0, n_episodes, self.batch_size):
                end = min(start + self.batch_size, n_episodes)
                batch_idx = indices[start:end]
                adv_batch = advantages[batch_idx]

                total_actor_loss = 0.0
                total_entropy = 0.0
                for i in range(self.n_agents):
                    obs = data["obs"][i][batch_idx]
                    old_act = data["actions"][i][batch_idx]
                    old_lp = data["log_probs"][i][batch_idx]
                    ids = torch.full((len(batch_idx),), i, dtype=torch.long,
                                     device=self.device)

                    logits = self.actor(obs, ids)
                    dist = Categorical(logits=logits)
                    new_lp = dist.log_prob(old_act)
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                        1.0 + self.clip_eps) * adv_batch
                    total_actor_loss += -torch.min(surr1, surr2).mean()
                    total_entropy += entropy

                    # Independent critic per agent
                    agent_obs = data["obs"][i][batch_idx]
                    val_pred = self.critics[i](agent_obs)
                    critic_loss = F.mse_loss(val_pred, returns[batch_idx])
                    self.critic_optimizers[i].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(),
                                                    self.max_grad_norm)
                    self.critic_optimizers[i].step()

                avg_actor_loss = total_actor_loss / self.n_agents
                avg_entropy = total_entropy / self.n_agents
                loss = avg_actor_loss - self.entropy_coef * avg_entropy

                self.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                self.max_grad_norm)
                self.actor_optimizer.step()

                stats["actor_loss"] += avg_actor_loss.item()
                stats["entropy"] += avg_entropy.item()
                n_batches += 1

        for k in stats:
            stats[k] /= max(n_batches, 1)
        return stats


def run_variant(variant):
    cfg = make_config(variant)
    set_seed(cfg.seed)

    log_dir = os.path.join(RESULTS_DIR, variant)
    agent_cls = IPPOAgent if variant == "ippo" else MAPPOAgent
    agent = agent_cls(
        n_agents=cfg.n_agents,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        config=cfg,
    )
    trainer = Trainer(agent, cfg, log_dir=log_dir)
    best_wr = trainer.train()
    print(f"\n{variant} done — best win rate: {best_wr:.4f}")
    return best_wr


def main():
    variants = ["baseline", "no_entropy_decay", "ippo", "small_net"]
    print(f"Running {len(variants)} ablation variants: {variants}")
    print(f"Results will be saved to: {RESULTS_DIR}\n")

    results = {}
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"  VARIANT: {variant}")
        print(f"{'='*60}")
        wr = run_variant(variant)
        results[variant] = wr

    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")
    for variant, wr in results.items():
        print(f"  {variant:<20s}: best WR = {wr:.4f}")


if __name__ == "__main__":
    main()
