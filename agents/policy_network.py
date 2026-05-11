import torch
import torch.nn as nn


class Actor(nn.Module):
    """Shared policy: (obs, agent_id) → action logits."""

    def __init__(self, obs_dim=1, n_agents=3, embed_dim=4, hidden_dim=32, action_dim=2):
        super().__init__()
        self.agent_embed = nn.Embedding(n_agents, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs, agent_ids):
        emb = self.agent_embed(agent_ids)
        x = torch.cat([obs, emb], dim=-1)
        return self.net(x)


class Critic(nn.Module):
    """Centralized critic: joint obs (n_agents × obs_dim) → state value."""

    def __init__(self, obs_dim=1, n_agents=3, hidden_dim=64):
        super().__init__()
        input_dim = obs_dim * n_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs):
        flat = joint_obs.reshape(joint_obs.shape[0], -1)
        return self.net(flat).squeeze(-1)
