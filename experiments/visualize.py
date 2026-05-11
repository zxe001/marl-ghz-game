"""
Generate all figures for the GHZ MARL paper.

Figures:
  fig1_game_schematic.pdf       -- GHZ game illustration
  fig2_training_dynamics.pdf    -- Win rate, reward, entropy, per-scenario WR
  fig3_policy_evolution.pdf     -- Policy probabilities over time
  fig4_quantum_vs_classical.pdf -- Bar chart comparison
  fig5_quantum_interference.pdf -- Quantum interference cancellation
  fig6_ablations.pdf            -- Ablation study results
  fig7_multi_seed_summary.pdf   -- Multi-seed convergence summary
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patches as mpatches

# Clean font setup to avoid garbled text
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,       # embed as TrueType (avoids garbled text)
    "ps.fonttype": 42,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEED_COLORS = ["#2c7bb6", "#d7191c", "#fdae61", "#5e3c99", "#4daf4a"]
ABLATION_COLORS = {
    "baseline": "#2c7bb6",
    "no_entropy_decay": "#d7191c",
    "ippo": "#fdae61",
    "small_net": "#5e3c99",
}
ABLATION_LABELS = {
    "baseline": "MAPPO (baseline)",
    "no_entropy_decay": "No entropy decay",
    "ippo": "IPPO",
    "small_net": "Small network",
}


def load_multi_seed():
    all_data = []
    for seed_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, "multi_seed", "seed_*"))):
        metrics_path = os.path.join(seed_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)
            seed = int(os.path.basename(seed_dir).replace("seed_", ""))
            all_data.append((seed, data))
    return sorted(all_data, key=lambda x: x[0])


def load_ablations():
    results = {}
    for variant_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, "ablations", "*"))):
        if not os.path.isdir(variant_dir):
            continue
        metrics_path = os.path.join(variant_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)
            variant = os.path.basename(variant_dir)
            results[variant] = data
    return results


def smooth(y, window=50):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


# ═══════════════════════════════════════════════════════════════
# Figure 1: GHZ Game Schematic
# ═══════════════════════════════════════════════════════════════

def fig1_game_schematic():
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_aspect("equal")

    # Referee
    ref_box = FancyBboxPatch((3.5, 4.2), 3, 0.7, boxstyle="round,pad=0.1",
                             facecolor="#fdd49e", edgecolor="#d95f0e", linewidth=2)
    ax.add_patch(ref_box)
    ax.text(5, 4.55, "Referee", ha="center", va="center", fontsize=12,
            fontweight="bold", color="#7f2704")
    ax.text(5, 3.95, "Scenario: XXX / XYY / YXY / YYX", ha="center",
            va="center", fontsize=8, style="italic")

    # Players
    player_positions = {"Alice": (1.5, 2.0), "Bob": (5.0, 2.0), "Carol": (8.5, 2.0)}
    player_colors = {"Alice": "#a6cee3", "Bob": "#b2df8a", "Carol": "#fb9a99"}
    for name, (x, y) in player_positions.items():
        circle = Circle((x, y), 0.6, facecolor=player_colors[name],
                        edgecolor="#333333", linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, name, ha="center", va="center", fontsize=9, fontweight="bold")

    # Arrows: referee -> players
    for name, (x, y) in player_positions.items():
        ax.annotate("", xy=(x, y + 0.6), xytext=(5, 4.2),
                    arrowprops=dict(arrowstyle="->", color="#d95f0e", lw=1.5,
                                    connectionstyle="arc3,rad=0"))
    for name, (x, y) in player_positions.items():
        mid_x, mid_y = (x + 5) / 2, (y + 4.2) / 2
        ax.text(mid_x + 0.2, mid_y, "X/Y?", fontsize=7, color="#d95f0e",
                fontweight="bold")

    # Arrows: players -> referee
    for name, (x, y) in player_positions.items():
        ax.annotate("", xy=(5, 4.2), xytext=(x + 0.3, y + 0.6),
                    arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.5,
                                    connectionstyle="arc3,rad=0.15"))
    for name, (x, y) in player_positions.items():
        ax.text(x + 0.3, y + 1.1, "+/-", fontsize=7, color="#1b9e77",
                fontweight="bold")

    # No-communication barrier
    for i in range(2):
        ax.plot([3.3 + i * 3.4, 3.3 + i * 3.4], [1.2, 2.9], "k--", lw=0.8, alpha=0.3)
    ax.text(5, 1.0, "No communication", ha="center", fontsize=7.5,
            style="italic", color="gray")

    # Win condition box
    win_box = FancyBboxPatch((1.5, 0.1), 7, 0.6, boxstyle="round,pad=0.1",
                             facecolor="#e5f5e0", edgecolor="#4daf4a", linewidth=1.5)
    ax.add_patch(win_box)
    ax.text(5, 0.4,
            "Win: XXX -> odd #+  |  XYY, YXY, YYX -> even #+",
            ha="center", va="center", fontsize=8.5, fontweight="bold")

    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG_DIR, "fig1_game_schematic.pdf"))
    plt.close(fig)
    print("  fig1_game_schematic.pdf")


# ═══════════════════════════════════════════════════════════════
# Figure 2: Training Dynamics
# ═══════════════════════════════════════════════════════════════

def fig2_training_dynamics():
    all_data = load_multi_seed()
    if not all_data:
        print("  [WARN] No multi-seed data.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    # (a) Win rate
    ax = axes[0, 0]
    for i, (seed, data) in enumerate(all_data):
        eps = [e["episode"] for e in data["episodes"]]
        wr = smooth([e["win"] for e in data["episodes"]], window=200)
        ax.plot(eps, wr, color=SEED_COLORS[i], lw=0.7, alpha=0.8,
                label=f"seed={seed}")
    ax.axhline(y=0.75, color="black", linestyle="--", lw=1.0, alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate (smoothed)")
    ax.set_title("(a) Training Win Rate", fontsize=10)
    ax.set_ylim(0.35, 1.02)
    ax.legend(fontsize=6.5, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.25)

    # (b) Per-scenario win rate (seed 42)
    ax = axes[0, 1]
    _, data = all_data[0]
    evals = data["evaluations"]
    eval_eps = [e["episode"] for e in evals]
    scenarios = ["XXX", "XYY", "YXY", "YYX"]
    colors_s = ["#d7191c", "#2c7bb6", "#fdae61", "#4daf4a"]
    styles = ["-", "--", "-.", ":"]
    for s, c, ls in zip(scenarios, colors_s, styles):
        wr_s = [e["per_scenario"].get(s, np.nan) for e in evals]
        ax.plot(eval_eps, wr_s, color=c, ls=ls, lw=1.2, label=s)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    ax.set_title("(b) Per-Scenario Win Rate", fontsize=10)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    # (c) Reward
    ax = axes[1, 0]
    for i, (seed, data) in enumerate(all_data):
        eps = [e["episode"] for e in data["episodes"]]
        rew = smooth([e["reward"] for e in data["episodes"]], window=200)
        ax.plot(eps, rew, color=SEED_COLORS[i], lw=0.7, alpha=0.8)
    ax.axhline(y=0.5, color="black", linestyle="--", lw=1.0, alpha=0.5,
               label="Optimal reward (0.5)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward (smoothed)")
    ax.set_title("(c) Average Reward", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(True, alpha=0.25)

    # (d) Entropy
    ax = axes[1, 1]
    _, data = all_data[0]
    updates = data.get("updates", [])
    if updates:
        update_eps = [u["episode"] for u in updates]
        ent_coef = [u["entropy_coef"] for u in updates]
        ent = [u["entropy"] for u in updates]
        ax.plot(update_eps, ent_coef, color="#2c7bb6", lw=1.2, label="Entropy coef")
        ax2 = ax.twinx()
        ax2.plot(update_eps, smooth(ent, 5), color="#d7191c", lw=0.8, alpha=0.7,
                 label="Policy entropy")
        ax2.set_ylabel("Policy Entropy", color="#d7191c", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#d7191c", labelsize=8)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=6.5, loc="center right")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy Coefficient", color="#2c7bb6", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#2c7bb6", labelsize=8)
    ax.set_title("(d) Exploration Decay", fontsize=10)
    ax.grid(True, alpha=0.25)

    fig.savefig(os.path.join(FIG_DIR, "fig2_training_dynamics.pdf"))
    plt.close(fig)
    print("  fig2_training_dynamics.pdf")


# ═══════════════════════════════════════════════════════════════
# Figure 3: Policy Evolution
# ═══════════════════════════════════════════════════════════════

def fig3_policy_evolution():
    all_data = load_multi_seed()
    if not all_data:
        print("  [WARN] No multi-seed data.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    names = ["Alice", "Bob", "Carol"]
    _, data = all_data[0]
    evals = data["evaluations"]
    eval_eps = [e["episode"] for e in evals]

    for col, name in enumerate(names):
        ax = axes[0, col]
        probs = [e["policy_probs"].get(f"{name}_X_+", np.nan) for e in evals]
        ax.plot(eval_eps, probs, color="#2c7bb6", lw=1.2)
        ax.axhline(y=0.5, color="gray", linestyle=":", lw=0.6)
        ax.set_title(f"{name}: P(+ | X)", fontsize=10)
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.25)

        ax = axes[1, col]
        probs = [e["policy_probs"].get(f"{name}_Y_+", np.nan) for e in evals]
        ax.plot(eval_eps, probs, color="#d7191c", lw=1.2)
        ax.axhline(y=0.5, color="gray", linestyle=":", lw=0.6)
        ax.set_title(f"{name}: P(+ | Y)", fontsize=10)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.25)

    fig.savefig(os.path.join(FIG_DIR, "fig3_policy_evolution.pdf"))
    plt.close(fig)
    print("  fig3_policy_evolution.pdf")


# ═══════════════════════════════════════════════════════════════
# Figure 4: Quantum vs Classical Bar Chart
# ═══════════════════════════════════════════════════════════════

def fig4_quantum_vs_classical():
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.0))

    strategies = ["Classical\n(deterministic)", "MARL\n(MAPPO learned)", "Quantum\n(GHZ state)"]
    win_rates = [0.75, 0.75, 1.00]
    colors = ["#fdb462", "#2c7bb6", "#4daf4a"]
    hatches = ["//", "\\\\", ""]

    for i, (wr, c, h) in enumerate(zip(win_rates, colors, hatches)):
        ax.bar(i, wr, color=c, hatch=h, edgecolor="black", linewidth=1.0, width=0.5)

    for i, wr in enumerate(win_rates):
        ax.text(i, wr + 0.025, f"{wr*100:.0f}%", ha="center", fontsize=12,
                fontweight="bold")

    # Quantum advantage annotation
    ax.annotate("+25 pp\nadvantage", xy=(2, 1.0), xytext=(2, 0.83),
                ha="center", fontsize=8.5, fontweight="bold", color="#4daf4a",
                arrowprops=dict(arrowstyle="->", color="#4daf4a", lw=1.5))

    ax.set_xticks(range(3))
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_ylabel("Win Rate", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG_DIR, "fig4_quantum_vs_classical.pdf"))
    plt.close(fig)
    print("  fig4_quantum_vs_classical.pdf")


# ═══════════════════════════════════════════════════════════════
# Figure 5: Quantum Interference
# ═══════════════════════════════════════════════════════════════

def fig5_quantum_interference():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    fig.subplots_adjust(wspace=0.3)

    # XXX scenario
    ax = axes[0]
    outcomes = ["+++", "+--", "-+-", "--+", "++-", "+-+", "-++", "---"]
    probs = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]
    colors = ["#4daf4a"] * 4 + ["#d7191c"] * 4
    ax.bar(range(8), probs, color=colors, edgecolor="black", linewidth=0.6, width=0.7)
    ax.set_xticks(range(8))
    ax.set_xticklabels(outcomes, fontsize=7, rotation=30)
    ax.set_ylabel("Born Probability", fontsize=9)
    ax.set_title("Scenario: XXX (odd #+)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.32)
    ax.grid(axis="y", alpha=0.25)
    win_patch = mpatches.Patch(color="#4daf4a", label="WIN (survive)")
    lose_patch = mpatches.Patch(color="#d7191c", label="LOSE (cancelled)")
    ax.legend(handles=[win_patch, lose_patch], fontsize=7.5, loc="upper right")

    # XYY scenario
    ax = axes[1]
    outcomes = ["++-", "+-+", "-++", "---", "+++", "+--", "-+-", "--+"]
    probs = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]
    colors = ["#4daf4a"] * 4 + ["#d7191c"] * 4
    ax.bar(range(8), probs, color=colors, edgecolor="black", linewidth=0.6, width=0.7)
    ax.set_xticks(range(8))
    ax.set_xticklabels(outcomes, fontsize=7, rotation=30)
    ax.set_ylabel("Born Probability", fontsize=9)
    ax.set_title("Scenario: XYY / YXY / YYX (even #+)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.32)
    ax.grid(axis="y", alpha=0.25)

    fig.savefig(os.path.join(FIG_DIR, "fig5_quantum_interference.pdf"))
    plt.close(fig)
    print("  fig5_quantum_interference.pdf")


# ═══════════════════════════════════════════════════════════════
# Figure 6: Ablations
# ═══════════════════════════════════════════════════════════════

def fig6_ablations():
    ablation_data = load_ablations()
    if not ablation_data:
        print("  [WARN] No ablation data.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0))
    fig.subplots_adjust(hspace=0.4)

    # (a) Win rate curves
    ax = axes[0]
    for variant, data in ablation_data.items():
        if "episodes" not in data:
            continue
        eps = [e["episode"] for e in data["episodes"]]
        wr = smooth([e["win"] for e in data["episodes"]], window=200)
        c = ABLATION_COLORS.get(variant, "#999")
        label = ABLATION_LABELS.get(variant, variant)
        ax.plot(eps, wr, color=c, lw=1.0, alpha=0.85, label=label)
    ax.axhline(y=0.75, color="black", linestyle="--", lw=0.8, alpha=0.4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate (smoothed)")
    ax.set_title("(a) Training Win Rate", fontsize=10)
    ax.set_ylim(0.30, 1.02)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    # (b) Best win rate bar chart
    ax = axes[1]
    variant_order = ["baseline", "no_entropy_decay", "ippo", "small_net"]
    variants = []
    wrs = []
    clrs = []
    for v in variant_order:
        if v in ablation_data:
            variants.append(ABLATION_LABELS.get(v, v))
            wrs.append(ablation_data[v].get("best_win_rate",
                        ablation_data[v].get("final_win_rate", 0)))
            clrs.append(ABLATION_COLORS.get(v, "#999"))

    y_pos = range(len(variants))
    ax.barh(y_pos, wrs, color=clrs, edgecolor="black", linewidth=0.7, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variants, fontsize=8)
    ax.set_xlabel("Best Win Rate")
    ax.set_title("(b) Best Win Rate", fontsize=10)
    ax.set_xlim(0.45, 0.82)
    ax.axvline(x=0.75, color="black", linestyle="--", lw=0.8, alpha=0.4,
               label="Classical bound")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    for i, wr in enumerate(wrs):
        ax.text(wr + 0.006, i, f"{wr:.3f}", va="center", fontsize=8, fontweight="bold")

    fig.savefig(os.path.join(FIG_DIR, "fig6_ablations.pdf"))
    plt.close(fig)
    print("  fig6_ablations.pdf")


# ═══════════════════════════════════════════════════════════════
# Figure 7: Multi-seed Summary
# ═══════════════════════════════════════════════════════════════

def fig7_multi_seed_summary():
    all_data = load_multi_seed()
    if not all_data:
        print("  [WARN] No multi-seed data.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.6))

    # Compute mean and std
    all_eps = None
    all_wrs = []
    for seed, data in all_data:
        eps = np.array([e["episode"] for e in data["episodes"]])
        wr = np.array([e["win"] for e in data["episodes"]])
        if all_eps is None:
            all_eps = eps
        all_wrs.append(smooth(wr, window=200))

    min_len = min(len(w) for w in all_wrs)
    all_wrs = [w[:min_len] for w in all_wrs]
    eps_common = all_eps[:min_len]
    wr_mean = np.mean(all_wrs, axis=0)
    wr_std = np.std(all_wrs, axis=0)

    ax.fill_between(eps_common, wr_mean - wr_std, wr_mean + wr_std,
                    alpha=0.2, color="#2c7bb6", edgecolor="none")
    ax.plot(eps_common, wr_mean, color="#2c7bb6", lw=1.5,
            label=f"Mean +/- 1 std ({len(all_data)} seeds)")
    # Also plot individual seeds
    for i, (seed, _) in enumerate(all_data):
        ax.plot(eps_common, all_wrs[i][:min_len], color=SEED_COLORS[i],
                lw=0.4, alpha=0.5)

    ax.axhline(y=0.75, color="black", linestyle="--", lw=1.0, alpha=0.5,
               label="Classical bound (0.75)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate (smoothed)")
    ax.set_title("Convergence Across Five Random Seeds", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.35, 1.02)
    ax.grid(True, alpha=0.25)

    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG_DIR, "fig7_multi_seed_summary.pdf"))
    plt.close(fig)
    print("  fig7_multi_seed_summary.pdf")


# ═══════════════════════════════════════════════════════════════

def main():
    print("Generating figures...")
    fig1_game_schematic()
    fig2_training_dynamics()
    fig3_policy_evolution()
    fig4_quantum_vs_classical()
    fig5_quantum_interference()
    fig6_ablations()
    fig7_multi_seed_summary()
    print(f"Done. Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
