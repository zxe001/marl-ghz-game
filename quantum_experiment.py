"""
Quantum GHZ experiment: simulate the 100% winning quantum strategy.

Three players share a GHZ state |GHZ⟩ = (|000⟩ + |111⟩) / √2 before the game.
- If a player receives X, they measure their qubit in the X basis {|+⟩, |−⟩}.
- If a player receives Y, they measure their qubit in the Y basis {|+i⟩, |−i⟩}.
- The measurement outcome (+ or −) is their answer to the referee.

Quantum mechanics predicts 100% win rate, beating the classical 75% bound.
"""

import itertools
from typing import Iterator

import numpy as np

from env.ghz_game import check_win

# ── Constants ──────────────────────────────────────────────────────────

EPS = 1e-10  # threshold for treating probabilities as zero (interference cancellation)

# |GHZ⟩ = (|000⟩ + |111⟩) / √2,  stored as a length-8 complex vector.
# Index bits: qubit0 = Alice (LSB), qubit1 = Bob, qubit2 = Carol (MSB).
_ghz = np.zeros(8, dtype=complex)
_ghz[0b000] = 1.0 / np.sqrt(2)  # |000⟩
_ghz[0b111] = 1.0 / np.sqrt(2)  # |111⟩
GHZ = _ghz.copy()  # immutable after construction


def _make_ghz() -> np.ndarray:
    """Return the GHZ state vector (allows re-generation with different conventions)."""
    v = np.zeros(8, dtype=complex)
    v[0b000] = 1.0 / np.sqrt(2)
    v[0b111] = 1.0 / np.sqrt(2)
    return v


# ── Measurement basis states ───────────────────────────────────────────
# Pauli-X eigenstates: |+⟩, |−⟩    (eigenvalues ±1)
# Pauli-Y eigenstates: |+i⟩, |−i⟩  (eigenvalues ±1)

def _x_plus() -> np.ndarray:
    return np.array([1.0, 1.0]) / np.sqrt(2)


def _x_minus() -> np.ndarray:
    return np.array([1.0, -1.0]) / np.sqrt(2)


def _y_plus() -> np.ndarray:
    return np.array([1.0, 1.0j]) / np.sqrt(2)


def _y_minus() -> np.ndarray:
    return np.array([1.0, -1.0j]) / np.sqrt(2)


_BASIS_MAP = {
    ("X", "+"): _x_plus,
    ("X", "-"): _x_minus,
    ("Y", "+"): _y_plus,
    ("Y", "-"): _y_minus,
}


def basis_state(basis: str, outcome: str) -> np.ndarray:
    """Return the single-qubit ket for a given measurement basis and outcome."""
    key = (basis, outcome)
    if key not in _BASIS_MAP:
        raise ValueError(f"Unknown basis/outcome: {basis}/{outcome}.  Use X/Y and +/-.")
    return _BASIS_MAP[key]()


# ── Simulation ─────────────────────────────────────────────────────────

def _iter_outcomes() -> Iterator[tuple[str, str, str]]:
    """Iterate over all 8 outcome combinations (+++, ++-, …, ---)."""
    return itertools.product(["+", "-"], repeat=3)


def simulate_scenario(scenario: str) -> dict[tuple[str, str, str], float]:
    """
    Compute Born-rule probabilities for all 8 measurement outcomes.

    P(a,b,c) = |⟨abc|GHZ⟩|²  where ⟨abc| = ⟨a|⊗⟨b|⊗⟨c| in the scenario's bases.
    """
    bases = list(scenario)
    probs: dict[tuple[str, str, str], float] = {}

    for a, b, c in _iter_outcomes():
        outcomes = [a, b, c]

        bra0 = basis_state(bases[0], outcomes[0]).conj()
        bra1 = basis_state(bases[1], outcomes[1]).conj()
        bra2 = basis_state(bases[2], outcomes[2]).conj()
        bra = np.kron(np.kron(bra0, bra1), bra2)

        amplitude = np.dot(bra, GHZ)
        probs[(a, b, c)] = float(np.abs(amplitude) ** 2)

    return probs


def analyse_scenario(scenario: str, probs: dict) -> tuple[float, list[str], list[str]]:
    """
    Classify outcomes as:
      - survived (prob > EPS): separate into wins and losses
      - cancelled (prob ≤ EPS): interference-destroyed terms

    Returns (win_probability, survived_lines, cancelled_lines).
    """
    win_prob = 0.0
    survived: list[str] = []
    cancelled: list[str] = []

    for a, b, c in _iter_outcomes():
        outcomes = (a, b, c)
        prob = probs[outcomes]
        n_plus = sum(1 for o in outcomes if o == "+")
        win = check_win(scenario, outcomes)

        if prob > EPS:
            win_prob += prob
            label = "WIN" if win else "LOSE"
            survived.append(f"  {outcomes}  {prob:.6f}  #{n_plus}  {label}")
        else:
            label = "win" if win else "lose"
            cancelled.append(f"({a},{b},{c})")

    return win_prob, survived, cancelled


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    scenarios = ["XXX", "XYY", "YXY", "YYX"]

    print("=" * 68)
    print("  GHZ Quantum Strategy — Full Probability Analysis")
    print("=" * 68)

    total_win_prob = 0.0

    for scenario in scenarios:
        probs = simulate_scenario(scenario)
        win_prob, survived, cancelled = analyse_scenario(scenario, probs)
        total_win_prob += win_prob / 4.0

        print(f"\n── Scenario: {scenario} ──")
        print(f"    {'Outcome':<12} {'Probability':<12} {'#+':<5} {'Result'}")
        print(f"    {'-' * 40}")
        for line in survived:
            print(line)
        print(f"    {'-' * 40}")
        print(f"    Win probability: {win_prob:.4f}")
        if cancelled:
            print(f"    Cancelled (interference): {', '.join(cancelled)}")

    print(f"\n{'=' * 68}")
    print(f"  Overall quantum win rate:  {total_win_prob:.4f}  ({total_win_prob*100:.1f}%)")
    print(f"  Classical upper bound:      0.7500  ( 75.0%)")
    print(f"  Quantum advantage:          {total_win_prob - 0.75:+.4f}  ({'+' if total_win_prob > 0.75 else ''}{(total_win_prob - 0.75)*100:.1f} pp)")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
