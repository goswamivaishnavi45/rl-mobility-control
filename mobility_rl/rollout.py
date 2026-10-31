"""Shared episode runner + metrics, used by tuning, evaluation and plotting."""

import numpy as np

from mobility_rl import make_env


def run_episode(env, policy, seed, record=False):
    """policy(env, obs) -> action. Returns a metrics dict (and a trace if record=True)."""
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(env)
    ret, errs, actions, trace = 0.0, [], [], []
    done, fell = False, False
    while not done:
        a = policy(env, obs)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        errs.append(abs(info["heading_error"]))
        actions.append(float(a[0]))
        fell = fell or term
        done = term or trunc
        if record:
            trace.append({"t": env.step_count * env.dt, **info, "action": float(a[0])})
    act_rate = np.abs(np.diff(actions)).mean() / env.dt if len(actions) > 1 else 0.0
    out = dict(
        ret=ret,
        mean_abs_err_deg=float(np.degrees(np.mean(errs))),
        fell=bool(fell),
        action_rate=float(act_rate),  # mean |d action / dt| in normalised units per second
        steps=len(errs),
    )
    return (out, trace) if record else out


def evaluate(env_name, policy, randomize, seeds):
    env = make_env(env_name, randomize)
    return [run_episode(env, policy, s) for s in seeds]


def summarize(results):
    ok = [r for r in results if not r["fell"]] or results
    return dict(
        episodes=len(results),
        mean_return=float(np.mean([r["ret"] for r in results])),
        fall_rate=float(np.mean([r["fell"] for r in results])),
        # tracking error over episodes that did not end in a fall
        mean_abs_heading_err_deg=float(np.mean([r["mean_abs_err_deg"] for r in ok])),
        mean_action_rate=float(np.mean([r["action_rate"] for r in ok])),
    )


class BaselinePolicy:
    """Adapts a classical controller (which reads env.last_measurement) to policy(env, obs)."""

    def __init__(self, controller):
        self.c = controller

    def reset(self, env):
        self.c.reset(env)

    def __call__(self, env, obs):
        return self.c(env)


class SB3Policy:
    def __init__(self, model):
        self.model = model

    def __call__(self, env, obs):
        a, _ = self.model.predict(obs, deterministic=True)
        return a
