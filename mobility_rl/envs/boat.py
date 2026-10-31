"""Boat heading-control (course-keeping / course-changing) environment.

Dynamics: nonlinear first-order Nomoto ship-steering model (Norrbin form)
with a rate-limited rudder servo, servo lag, actuator delay, a constant
wind/current yaw bias, first-order wave disturbance and sensor noise
(Fossen, "Handbook of Marine Craft Hydrodynamics and Motion Control", 2011).

    T r_d + r + alpha r^3 = K delta + d(t)
    psi_d   = r
    delta_d = clip((delta_cmd - delta) / tau_r, -rate_max, rate_max)

Task: hold a commanded heading that changes by up to +-60 deg every 20-30 s.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mobility_rl.envs.common import RANDOMIZATION_MODES, wrap_angle

DELTA_MAX = np.deg2rad(35.0)
RATE_MAX = np.deg2rad(15.0)
NOMINAL = dict(K=0.25, T=5.0, alpha=2.0, tau_r=0.5)

RANGES = {
    "none": dict(K=(0.25, 0.25), T=(5.0, 5.0), alpha=(2.0, 2.0), tau_r=(0.5, 0.5),
                 delay=(0, 0), bias=0.0, wave=0.0, noise=0.0),
    "train": dict(K=(0.15, 0.35), T=(3.0, 7.0), alpha=(0.0, 5.0), tau_r=(0.3, 0.7),
                  delay=(0, 1), bias=0.015, wave=0.008, noise=1.0),
    "shift": dict(K=(0.10, 0.45), T=(2.0, 10.0), alpha=(0.0, 10.0), tau_r=(0.6, 1.0),
                  delay=(1, 3), bias=0.03, wave=0.015, noise=2.0),
}


class BoatEnv(gym.Env):
    """Observation (6, normalised):
        heading error, yaw rate, rudder angle, integral of heading error,
        previous action, rudder rate
    Action (1): rudder command in [-1, 1] -> [-35 deg, +35 deg]
    """

    metadata = {"render_modes": []}

    def __init__(self, randomize="train", dt=0.1, substeps=2, episode_seconds=90.0):
        assert randomize in RANDOMIZATION_MODES
        self.randomize = randomize
        self.dt = dt
        self.substeps = substeps
        self.max_steps = int(episode_seconds / dt)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def _sample_params(self):
        r, u = RANGES[self.randomize], self.np_random.uniform
        p = {k: u(*r[k]) for k in ("K", "T", "alpha", "tau_r")}
        p["delay"] = int(self.np_random.integers(r["delay"][0], r["delay"][1] + 1))
        p["bias"] = u(-r["bias"], r["bias"])  # wind/current yaw bias [rad/s, rudder-equivalent K*delta]
        p["wave_amp"] = u(0.0, r["wave"])
        p["wave_period"] = u(6.0, 12.0)
        p["wave_phase"] = u(0.0, 2 * np.pi)
        p["noise"] = r["noise"]
        return p

    def _new_reference(self):
        self.psi_ref = wrap_angle(self.psi_ref + self.np_random.uniform(-np.pi / 3, np.pi / 3))
        self.next_ref_step = self.step_count + int(self.np_random.uniform(20.0, 30.0) / self.dt)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.p = self._sample_params()
        if options and "params" in options:
            self.p.update(options["params"])
        u = self.np_random.uniform
        self.state = np.array([u(-0.1, 0.1), 0.0, 0.0])  # psi, r, delta
        self.step_count = 0
        self.psi_ref = 0.0
        self.next_ref_step = int(u(2.0, 5.0) / self.dt)
        self.int_err = 0.0
        self.prev_action = 0.0
        self.delta_rate = 0.0
        self.cmd_queue = [0.0] * self.p["delay"]
        return self._obs(), self._info()

    def _disturbance(self, t):
        p = self.p
        return p["bias"] + p["wave_amp"] * np.sin(2 * np.pi * t / p["wave_period"] + p["wave_phase"])

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        self.cmd_queue.append(a * DELTA_MAX)
        delta_cmd = self.cmd_queue.pop(0)
        p = self.p

        h = self.dt / self.substeps
        for i in range(self.substeps):  # semi-implicit Euler (rate limit is non-smooth)
            psi, r, delta = self.state
            t = (self.step_count * self.substeps + i) * h
            delta_d = np.clip((delta_cmd - delta) / p["tau_r"], -RATE_MAX, RATE_MAX)
            delta = np.clip(delta + h * delta_d, -DELTA_MAX, DELTA_MAX)
            r_d = (p["K"] * delta + self._disturbance(t) - r - p["alpha"] * r**3) / p["T"]
            r = r + h * r_d
            psi = psi + h * r
            self.state = np.array([psi, r, delta])
            self.delta_rate = delta_d

        self.step_count += 1
        if self.step_count >= self.next_ref_step:
            self._new_reference()

        err = self.heading_error
        self.int_err = float(np.clip(self.int_err + err * self.dt, -5.0, 5.0))
        reward = 1.0 - 2.0 * err**2 - 0.1 * (a - self.prev_action) ** 2
        self.prev_action = a
        truncated = self.step_count >= self.max_steps
        return self._obs(), float(reward), False, bool(truncated), self._info()

    @property
    def heading_error(self):
        return wrap_angle(self.psi_ref - self.state[0])

    def measure(self):
        """Noisy sensor readings (compass/GNSS heading, gyro yaw rate, rudder angle sensor)."""
        n = self.p["noise"]
        nr = self.np_random.normal
        _, r, delta = self.state
        return dict(
            err=wrap_angle(self.heading_error + n * np.deg2rad(0.5) * nr()),
            r=r + n * np.deg2rad(0.2) * nr(),
            delta=delta + n * np.deg2rad(0.3) * nr(),
            int_err=self.int_err,
            prev_action=self.prev_action,
            delta_rate=self.delta_rate,
        )

    def _obs(self):
        m = self.measure()
        self.last_measurement = m
        return np.array([
            m["err"] / (np.pi / 3), m["r"] / 0.1, m["delta"] / DELTA_MAX, m["int_err"] / 5.0,
            m["prev_action"], m["delta_rate"] / RATE_MAX,
        ], dtype=np.float32)

    def _info(self):
        return dict(
            heading_error=float(self.heading_error), psi=float(self.state[0]),
            psi_ref=float(self.psi_ref), rudder=float(self.state[2]), yaw_rate=float(self.state[1]),
        )
