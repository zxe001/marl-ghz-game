# MARL for GHZ Game

Multi-Agent Deep Reinforcement Learning (MAPPO) for the GHZ game — a cooperative three-player game where classical strategies are provably limited to 75% win rate, but quantum entanglement enables a perfect 100% strategy.

## The Game

Three isolated players (Alice, Bob, Carol) face a referee. No communication allowed.

1. **Challenge:** The referee gives each player a letter — **X** or **Y**. The total number of X's is always **odd**: `XXX`, `XYY`, `YXY`, or `YYX` (equal probability).

2. **Response:** Each player replies **+** or **−**.

3. **Win condition:**
   - `XXX` → odd number of `+`
   - `XYY` / `YXY` / `YYX` → even number of `+`

## Theoretical Limits

| Strategy | Max Win Rate | Why |
|----------|:-----------:|-----|
| Classical (best deterministic) | **75%** | The four win-condition equations multiply to `1 = −1` — a contradiction. At most 3 of 4 scenarios can be won. |
| Quantum (GHZ state) | **100%** | Sharing a GHZ state `(|000⟩ + |111⟩)/√2` before the game allows measurement bases to cancel losing outcomes via quantum interference. |

## Project Structure

```
marl_for_GHZ/
├── env/
│   └── ghz_game.py            # Game environment + shared win-condition logic
├── agents/
│   ├── policy_network.py      # Actor (shared + agent-ID) & Critic networks
│   └── mappo.py               # MAPPO agent with centralized critic
├── trainers/
│   └── trainer.py             # Training loop with entropy + LR decay
├── utils/
│   └── buffer.py              # Experience replay buffer
├── config.py                  # Hyperparameters
├── train.py                   # Train MARL agents
├── eval.py                    # Evaluate & inspect learned strategies
├── quantum_experiment.py      # Quantum GHZ-state simulation (Born rule)
└── requirements.txt           # numpy, torch
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the agents (converges to ~75% win rate)
python train.py

# Evaluate and inspect learned strategies
python eval.py --model best_model.pt

# Run the quantum experiment (100% win rate)
python quantum_experiment.py
```

## Results

### Classical MARL

After training, the agents discover an optimal classical strategy:

```
Alice: X → -, Y → -
Bob:   X → -, Y → -
Carol: X → -, Y → -
```

This always produces 0 pluses (even), winning 3/4 scenarios: **75.0%**.

```
XXX:   0.0%   (always loses — needs odd #+)
XYY: 100.0%
YXY: 100.0%
YYX: 100.0%
```

### Quantum Strategy

The quantum simulation computes Born-rule probabilities for measuring the GHZ state. Quantum interference **exactly cancels** every losing outcome:

```
XXX:  (+++, +--, -+-, --+) survive — all odd #+  → WIN
      (++-, +-+, -++, ---) cancelled by interference

XYY:  (++-, +-+, -++, ---) survive — all even #+ → WIN
      (+++, +--, -+-, --+) cancelled by interference
```

**Overall quantum win rate: 100.0%** (+25 pp over classical bound).

## Key Design Decisions

- **Shared actor with agent-ID embedding:** The game is symmetric but the optimal strategy requires role differentiation. A shared policy network conditioned on agent ID is more sample-efficient than independent policies.
- **MAPPO with centralized critic:** The critic sees all three observations, reducing credit-assignment variance during training.
- **Entropy decay per update (not per episode):** Ensures exploration is reduced only when learning actually occurs.
