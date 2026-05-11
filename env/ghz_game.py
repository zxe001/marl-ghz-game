import numpy as np


def check_win(scenario, outcomes):
    """
    Check if measurement outcomes satisfy the GHZ win condition.

    Args:
        scenario: str like 'XXX', 'XYY', 'YXY', or 'YYX'
        outcomes: iterable of '+' / '-' (or 0 / 1 where 0='+')

    Returns True if the team wins.
    """
    n_plus = sum(1 for o in outcomes if o in (0, "+"))
    n_x = scenario.count("X")
    if n_x == 3:
        return n_plus % 2 == 1   # XXX → odd number of +
    else:
        return n_plus % 2 == 0   # one X → even number of +


class GHZGame:
    """
    Three-player GHZ game against a referee.

    Each episode: referee randomly picks XXX, XYY, YXY, or YYX (equal prob).
    Each player observes their own letter (X or Y) and replies + or -.
    Win if:
      - XXX → odd number of +
      - XYY / YXY / YYX → even number of +
    """

    SCENARIOS = ["XXX", "XYY", "YXY", "YYX"]

    def __init__(self, seed=None):
        self.n_agents = 3
        self.obs_dim = 1
        self.action_dim = 2
        self.current_scenario = None
        self.rng = np.random.RandomState(seed)

    def reset(self):
        idx = self.rng.randint(0, 4)
        self.current_scenario = self.SCENARIOS[idx]
        obs = []
        for ch in self.current_scenario:
            obs.append(np.array([0.0 if ch == "X" else 1.0], dtype=np.float32))
        return obs, self.current_scenario

    def step(self, actions):
        """
        actions: list of 3 ints, 0=+, 1=-
        """
        win = check_win(self.current_scenario, actions)
        reward = 1.0 if win else -1.0
        return [reward] * 3, True, {"win": win, "scenario": self.current_scenario}
