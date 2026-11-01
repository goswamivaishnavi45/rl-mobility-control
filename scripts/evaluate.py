"""Evaluate RL agents and tuned PID/LQR baselines on identical scenarios.

Two test sets per environment (same 100 seeds for every controller):
  train : parameters drawn from the training randomization ranges
  shift : harder, partly out-of-distribution parameters (see RANGES in the env files)

    python scripts/evaluate.py --episodes 100
Writes results/metrics_<env>.json, merged into results/metrics.json and results/RESULTS.md
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO, SAC  # noqa: E402

from mobility_rl.controllers import BASELINES  # noqa: E402
from mobility_rl.residual import ResidualPolicy, tuned_baseline  # noqa: E402
from mobility_rl.rollout import BaselinePolicy, SB3Policy, evaluate, summarize  # noqa: E402

RL_RUNS = {
    "PPO": ("PPO", "dr"),
    "SAC": ("SAC", "dr"),
    "PPO (no DR)": ("PPO", "nodr"),
}


def load_policies(env_name):
    policies = {}
    for name, cls in BASELINES[env_name].items():
        gains = ROOT / "results" / "baselines" / f"{env_name}_{name}.json"
        params = json.loads(gains.read_text())["params"] if gains.exists() else {}
        policies[name] = BaselinePolicy(cls(**params))
    for label, (algo, tag) in RL_RUNS.items():
        path = ROOT / "runs" / f"{env_name}_{algo}_{tag}" / "best_model.zip"
        if path.exists():
            policies[label] = SB3Policy({"PPO": PPO, "SAC": SAC}[algo].load(path, device="cpu"))
    res = ROOT / "runs" / f"{env_name}_PPO_res" / "best_model.zip"
    if res.exists():
        policies["Residual PPO"] = ResidualPolicy(PPO.load(res, device="cpu"), tuned_baseline(env_name, "LQR"))
    return policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--envs", nargs="+", default=["motorcycle", "boat"])
    args = ap.parse_args()
    seeds = list(range(100_000, 100_000 + args.episodes))

    metrics = {}
    for env_name in args.envs:
        for label, policy in load_policies(env_name).items():
            for split in ("train", "shift"):
                s = summarize(evaluate(env_name, policy, split, seeds))
                metrics.setdefault(env_name, {}).setdefault(label, {})[split] = s
                print(f"{env_name:10s} {label:12s} {split:5s} return {s['mean_return']:7.1f}  "
                      f"err {s['mean_abs_heading_err_deg']:5.2f} deg  falls {100 * s['fall_rate']:5.1f}%", flush=True)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    # one file per environment so evaluations can run in parallel processes, then merge
    for env_name, rows in metrics.items():
        (out / f"metrics_{env_name}.json").write_text(json.dumps({env_name: rows}, indent=2))
    metrics = {}
    for f in sorted(out.glob("metrics_*.json")):
        metrics.update(json.loads(f.read_text()))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = [f"# Results ({args.episodes} episodes per cell, identical seeds for every controller)\n"]
    for env_name, rows in metrics.items():
        lines += [f"\n## {env_name}\n",
                  "| Controller | Split | Mean return | Mean abs heading error (deg) | Fall rate | Action rate (1/s) |",
                  "|---|---|---|---|---|---|"]
        for label, splits in rows.items():
            for split, s in splits.items():
                lines.append(f"| {label} | {split} | {s['mean_return']:.1f} | {s['mean_abs_heading_err_deg']:.2f} | "
                             f"{100 * s['fall_rate']:.1f}% | {s['mean_action_rate']:.2f} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
