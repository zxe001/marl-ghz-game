import numpy as np
import torch


class Buffer:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.clear()

    def clear(self):
        self.obs = [[] for _ in range(self.n_agents)]
        self.actions = [[] for _ in range(self.n_agents)]
        self.log_probs = [[] for _ in range(self.n_agents)]
        self.rewards = []
        self.values = []
        self.joint_obs = []

    def store(self, obs_list, action_list, log_prob_list, reward, value):
        for i in range(self.n_agents):
            self.obs[i].append(obs_list[i])
            self.actions[i].append(action_list[i])
            self.log_probs[i].append(log_prob_list[i])
        self.rewards.append(reward)
        self.values.append(value)
        self.joint_obs.append(np.stack(obs_list, axis=0))

    def get_all(self, device):
        obs_t = []
        actions_t = []
        log_probs_t = []
        for i in range(self.n_agents):
            obs_t.append(torch.FloatTensor(np.array(self.obs[i])).to(device))
            actions_t.append(torch.LongTensor(self.actions[i]).to(device))
            log_probs_t.append(torch.FloatTensor(self.log_probs[i]).to(device))

        return {
            "obs": obs_t,
            "actions": actions_t,
            "log_probs": log_probs_t,
            "rewards": torch.FloatTensor(self.rewards).to(device),
            "values": torch.FloatTensor(self.values).to(device),
            "joint_obs": torch.FloatTensor(np.array(self.joint_obs)).to(device),
        }

    def __len__(self):
        return len(self.rewards)
