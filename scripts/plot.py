"""Make README figures: learning curves, example rollouts, robustness bars.

    python scripts/plot.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mobility_rl import make_env  # noqa: E402
from mobility_rl.rollout import run_episode  # noqa: E402
from scripts.evaluate import load_policies  # noqa: E402

FIG = ROOT / "results" / "figures"
# Fixed categorical order: colour follows the controller in every figure
COLORS = {"PID": "#2a78d6", "LQR": "#eb6834", "PPO": "#1baf7a", "SAC": "#eda100", "PPO (no DR)": "#e87ba4"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10,
    "legend.frameon": False,
})


def learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, env_name in zip(axes, ("motorcycle", "boat")):
        for label, run in (("PPO", "PPO_dr"), ("SAC", "SAC_dr"), ("PPO (no DR)", "PPO_nodr")):
            f = ROOT / "runs" / f"{env_name}_{run}" / "evaluations.npz"
            if not f.exists():
                continue
            d = np.load(f)
            t, r = d["timesteps"], d["results"]
            mean, std = r.mean(1), r.std(1)
            ax.plot(t, mean, color=COLORS[label], label=label)
            ax.fill_between(t, mean - std, mean + std, color=COLORS[label], alpha=0.15, linewidth=0)
        ax.set_title(f"{env_name.capitalize()}: evaluation return during training", loc="left", fontsize=11)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("episode return")
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "learning_curves.png", dpi=150)


def rollouts(env_name, seed, split="shift"):
    policies = {k: v for k, v in load_policies(env_name).items() if k in ("PID", "LQR", "PPO", "SAC")}
    traces = {}
    for label, pol in policies.items():
        _, trace = run_episode(make_env(env_name, split), pol, seed, record=True)
        traces[label] = trace

    if env_name == "motorcycle":
        panels = [("psi", "heading (deg)"), ("lean", "lean angle (deg)"), ("steer", "steering angle (deg)")]
    else:
        panels = [("psi", "heading (deg)"), ("rudder", "rudder angle (deg)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.4 * len(panels)), sharex=True)
    for ax, (key, ylabel) in zip(axes, panels):
        if key == "psi":
            ref = traces[next(iter(traces))]
            ax.plot([p["t"] for p in ref], np.degrees(np.unwrap([p["psi_ref"] for p in ref])),
                    color=INK2, linestyle="--", linewidth=1.5, label="reference")
        for label, trace in traces.items():
            t = [p["t"] for p in trace]
            y = np.degrees(np.unwrap([p[key] for p in trace]) if key == "psi" else [p[key] for p in trace])
            fell = len(trace) < make_env(env_name).max_steps
            ax.plot(t, y, color=COLORS[label], label=label + (" (fell)" if fell else ""), linewidth=1.6)
            if fell:
                ax.plot(t[-1], y[-1], marker="X", markersize=9, color=COLORS[label], markeredgecolor=SURFACE)
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper left", ncol=5, fontsize=9)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(f"{env_name.capitalize()}: one out-of-distribution test scenario (seed {seed})",
                      loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"rollout_{env_name}.png", dpi=150)


def robustness():
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    specs = [("motorcycle", "fall_rate", "fall rate (%)", 100),
             ("motorcycle", "mean_abs_heading_err_deg", "mean abs heading error (deg)", 1),
             ("boat", "mean_abs_heading_err_deg", "mean abs heading error (deg)", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (env_name, key, ylabel, scale) in zip(axes, specs):
        rows = metrics[env_name]
        labels = list(rows)
        x = np.arange(2)
        w = 0.8 / len(labels)
        for i, label in enumerate(labels):
            vals = [rows[label][s][key] * scale for s in ("train", "shift")]
            bars = ax.bar(x + (i - (len(labels) - 1) / 2) * w, vals, w * 0.92, color=COLORS[label], label=label,
                          edgecolor=SURFACE, linewidth=1)
            ax.bar_label(bars, fmt="%.1f", fontsize=7, color=INK2, padding=1)
        ax.set_xticks(x, ["training distribution", "shifted (unseen)"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{env_name.capitalize()}: {ylabel.split(' (')[0]}", loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "robustness.png", dpi=150)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    learning_curves()
    robustness()
    rollouts("motorcycle", seed=int(sys.argv[1]) if len(sys.argv) > 1 else 100_003)
    rollouts("boat", seed=int(sys.argv[2]) if len(sys.argv) > 2 else 100_003)
