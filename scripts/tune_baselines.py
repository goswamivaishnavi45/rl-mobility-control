"""Tune PID gains / LQR weights by random search on the *training* distribution.

Objective = mean episode return (the same reward the RL agents optimise), so
the classical baselines get the same tuning budget and target as RL.

    python scripts/tune_baselines.py --env motorcycle --controller PID
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobility_rl.controllers import BASELINES  # noqa: E402
from mobility_rl.rollout import BaselinePolicy, evaluate, summarize  # noqa: E402

LOG_UNIFORM = lambda lo, hi: ("log", lo, hi)  # noqa: E731
UNIFORM = lambda lo, hi: ("lin", lo, hi)  # noqa: E731

SEARCH_SPACE = {
    ("motorcycle", "PID"): dict(kh=LOG_UNIFORM(0.1, 3), ki=LOG_UNIFORM(0.01, 1), kr=LOG_UNIFORM(0.01, 2),
                                kp=LOG_UNIFORM(0.3, 10), kd=LOG_UNIFORM(0.03, 2), sched=UNIFORM(0, 2),
                                lean_max=UNIFORM(0.1, 0.6)),
    ("motorcycle", "LQR"): dict(q_phi=LOG_UNIFORM(0.1, 100), q_phid=LOG_UNIFORM(0.01, 10),
                                q_delta=LOG_UNIFORM(0.01, 10), q_psi=LOG_UNIFORM(0.1, 100),
                                q_int=LOG_UNIFORM(0.01, 10), r=LOG_UNIFORM(0.1, 100), err_sat=UNIFORM(0.05, 0.8)),
    ("boat", "PID"): dict(kp=LOG_UNIFORM(0.2, 10), ki=LOG_UNIFORM(0.001, 1), kd=LOG_UNIFORM(0.5, 50)),
    ("boat", "LQR"): dict(q_psi=LOG_UNIFORM(0.1, 100), q_r=LOG_UNIFORM(0.1, 1000), q_delta=LOG_UNIFORM(0.001, 10),
                          q_int=LOG_UNIFORM(0.0001, 1), r=LOG_UNIFORM(0.1, 100)),
}


def sample(space, rng):
    out = {}
    for k, (kind, lo, hi) in space.items():
        out[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))) if kind == "log" else rng.uniform(lo, hi))
    return out


def score(env_name, cls, params, seeds):
    return summarize(evaluate(env_name, BaselinePolicy(cls(**params)), "train", seeds))["mean_return"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["motorcycle", "boat"])
    ap.add_argument("--controller", required=True, choices=["PID", "LQR"])
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cls = BASELINES[args.env][args.controller]
    space = SEARCH_SPACE[(args.env, args.controller)]
    rng = np.random.default_rng(args.seed)
    search_seeds = list(range(1000, 1000 + args.episodes))  # eval uses seeds >= 100000

    candidates = [dict(cls.DEFAULT)] + [sample(space, rng) for _ in range(args.samples)]
    scored = []
    for i, params in enumerate(candidates):
        s = score(args.env, cls, params, search_seeds)
        scored.append((s, params))
        if i % 10 == 0:
            print(f"[{i}/{len(candidates)}] best so far {max(scored, key=lambda x: x[0])[0]:.1f}", flush=True)

    # Re-score the top 5 on more episodes to avoid picking a lucky sample
    top = sorted(scored, key=lambda x: -x[0])[:5]
    final_seeds = list(range(2000, 2040))
    rescored = sorted(((score(args.env, cls, p, final_seeds), p) for _, p in top), key=lambda x: -x[0])
    best_score, best = rescored[0]
    print(f"best return {best_score:.1f}: {best}")

    out = Path(__file__).resolve().parents[1] / "results" / "baselines" / f"{args.env}_{args.controller}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"params": {**cls.DEFAULT, **best}, "tuning_return": best_score}, indent=2))


if __name__ == "__main__":
    main()
