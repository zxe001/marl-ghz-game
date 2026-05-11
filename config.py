import torch


class Config:
    # Reproducibility
    seed = 42

    # Game
    n_agents = 3
    obs_dim = 1       # 0=X, 1=Y
    action_dim = 2    # 0=+, 1=-

    # Network (small — problem only has 4 scenarios × 2 actions)
    embed_dim = 4
    hidden_dim = 32
    critic_hidden_dim = 64

    # Training
    lr_actor = 1e-3
    lr_critic = 1e-3
    lr_decay = 0.98       # per-update decay (×58 → ~0.31)
    clip_eps = 0.2
    entropy_coef = 0.05
    entropy_min = 0.001
    entropy_decay = 0.95  # per-update decay (×58 → ~0.0025)
    max_grad_norm = 0.5

    # PPO
    ppo_epochs = 8
    batch_size = 256

    # Loop
    max_episodes = 15000
    save_interval = 2000
    eval_interval = 500
    eval_episodes = 2000
    log_interval = 200

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
