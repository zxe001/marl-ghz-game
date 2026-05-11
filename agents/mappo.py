import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .policy_network import Actor, Critic


class MAPPOAgent:
    def __init__(self, n_agents, obs_dim, action_dim, config):
        self.n_agents = n_agents
        self.device = config.device
        self.clip_eps = config.clip_eps
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        self.ppo_epochs = config.ppo_epochs
        self.batch_size = config.batch_size

        self.actor = Actor(
            obs_dim, n_agents,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            action_dim=action_dim,
        ).to(self.device)

        self.critic = Critic(
            obs_dim, n_agents,
            hidden_dim=config.critic_hidden_dim,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        self.agent_ids = torch.arange(n_agents, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def get_actions(self, observations):
        obs_t = torch.FloatTensor([o[0] for o in observations]).unsqueeze(-1).to(self.device)
        logits = self.actor(obs_t, self.agent_ids)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        joint_obs = obs_t.unsqueeze(0)
        value = self.critic(joint_obs).item()

        return actions.tolist(), log_probs.tolist(), value

    def update(self, buffer):
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
                ret_batch = returns[batch_idx]

                total_actor_loss = 0.0
                total_entropy = 0.0
                for i in range(self.n_agents):
                    obs = data["obs"][i][batch_idx]
                    old_act = data["actions"][i][batch_idx]
                    old_lp = data["log_probs"][i][batch_idx]
                    ids = torch.full((len(batch_idx),), i, dtype=torch.long, device=self.device)

                    logits = self.actor(obs, ids)
                    dist = Categorical(logits=logits)
                    new_lp = dist.log_prob(old_act)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_batch
                    total_actor_loss += -torch.min(surr1, surr2).mean()
                    total_entropy += entropy

                avg_actor_loss = total_actor_loss / self.n_agents
                avg_entropy = total_entropy / self.n_agents
                loss = avg_actor_loss - self.entropy_coef * avg_entropy

                self.actor_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                joint = data["joint_obs"][batch_idx]
                values = self.critic(joint)
                critic_loss = F.mse_loss(values, ret_batch)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                stats["actor_loss"] += avg_actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += avg_entropy.item()
                n_batches += 1

        for k in stats:
            stats[k] /= max(n_batches, 1)
        return stats

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
