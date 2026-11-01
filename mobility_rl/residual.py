"""Residual RL: the policy learns a bounded correction on top of a classical controller.

    action = clip(a_base + scale * a_rl, -1, 1)

The policy also observes the base controller's action. With a_rl = 0 the system
behaves exactly like the tuned baseline, so learning starts from a safe,
stabilising controller and only has to learn what the linear law gets wrong
(saturation, model mismatch, delay). Common pattern for RL on real hardware.
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mobility_rl.controllers import BASELINES

ROOT = Path(__file__).resolve().parents[1]
RESIDUAL_SCALE = 0.5


def tuned_baseline(env_name, name="LQR"):
    cls = BASELINES[env_name][name]
    gains = ROOT / "results" / "baselines" / f"{env_name}_{name}.json"
    params = json.loads(gains.read_text())["params"] if gains.exists() else {}
    return cls(**params)


class ResidualWrapper(gym.Wrapper):
    def __init__(self, env, base_controller, scale=RESIDUAL_SCALE):
        super().__init__(env)
        self.base = base_controller
        self.scale = scale
        n = env.observation_space.shape[0] + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

    def _augment(self, obs):
        self.base_action = self.base(self.env.unwrapped)
        return np.append(obs, self.base_action).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.base.reset(self.env.unwrapped)
        return self._augment(obs), info

    def step(self, action):
        total = np.clip(self.base_action + self.scale * np.asarray(action, dtype=np.float32), -1, 1)
        obs, r, term, trunc, info = self.env.step(total)
        return self._augment(obs), r, term, trunc, info


class ResidualPolicy:
    """Evaluation-time equivalent of ResidualWrapper, for policy(env, obs) runners."""

    def __init__(self, model, base_controller, scale=RESIDUAL_SCALE):
        self.model, self.base, self.scale = model, base_controller, scale

    def reset(self, env):
        self.base.reset(env)

    def __call__(self, env, obs):
        a_base = self.base(env)
        a_rl, _ = self.model.predict(np.append(obs, a_base).astype(np.float32), deterministic=True)
        return np.clip(a_base + self.scale * a_rl, -1, 1).astype(np.float32)
