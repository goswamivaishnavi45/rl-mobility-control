import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobility_rl import make_env  # noqa: E402
from mobility_rl.controllers import BASELINES  # noqa: E402
from mobility_rl.envs.common import wrap_angle  # noqa: E402
from mobility_rl.rollout import BaselinePolicy, run_episode  # noqa: E402


@pytest.mark.parametrize("name", ["motorcycle", "boat"])
@pytest.mark.parametrize("mode", ["none", "train", "shift"])
def test_gymnasium_api(name, mode):
    check_env(make_env(name, mode), skip_render_check=True)


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(-3 * np.pi / 2), np.pi / 2)


def test_motorcycle_falls_without_control():
    env = make_env("motorcycle", "none")
    env.reset(seed=0)
    for _ in range(env.max_steps):
        _, _, term, trunc, _ = env.step(np.zeros(1, dtype=np.float32))
        if term:
            break
    assert term, "an uncontrolled two-wheeler should fall over (no self-stability in this model)"


def test_motorcycle_steering_into_lean_recovers():
    """Steering towards the lean must reduce the lean acceleration (sign convention check)."""
    env = make_env("motorcycle", "none")
    env.reset(seed=0)
    s = np.array([0.1, 0.0, 0.0, 0.0])
    neutral = env._derivs(s, 0.0, 0.0)[1]
    s_steered = np.array([0.1, 0.0, 0.1, 0.0])
    steered = env._derivs(s_steered, 0.1, 0.0)[1]
    assert steered < neutral


@pytest.mark.parametrize("name", ["motorcycle", "boat"])
def test_lqr_stabilises_nominal(name):
    env = make_env(name, "none")
    res = run_episode(env, BaselinePolicy(BASELINES[name]["LQR"](err_sat=0.15) if name == "motorcycle"
                                          else BASELINES[name]["LQR"]()), seed=3)
    assert not res["fell"]


def test_boat_rudder_rate_limited():
    from mobility_rl.envs.boat import RATE_MAX
    env = make_env("boat", "none")
    env.reset(seed=0)
    prev = env.state[2]
    env.step(np.ones(1, dtype=np.float32))
    assert env.state[2] - prev <= RATE_MAX * env.dt + 1e-9


def test_same_seed_same_scenario():
    a, b = make_env("motorcycle", "train"), make_env("motorcycle", "train")
    a.reset(seed=42)
    b.reset(seed=42)
    assert a.p == b.p
