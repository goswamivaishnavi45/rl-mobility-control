"""Two-wheeler (motorcycle) balance + heading-tracking environment.

Dynamics: nonlinear point-mass lean model of a two-wheeler (Astrom, Klein &
Lennartsson, "Bicycle dynamics and control", IEEE CSM 2005), extended with a
first-order steering actuator, actuator delay, a lean-torque disturbance
(crosswind / road camber) and sensor noise.

    phi_dd  = (g/h) sin(phi) - cos(phi) * [ v^2/(b h) tan(delta) + a v/(b h) delta_d / cos^2(delta) ] + d
    delta_d = (delta_cmd - delta) / tau
    psi_d   = v tan(delta) / b

phi = lean angle, delta = steering angle, psi = heading, v = forward speed,
h = CoM height, b = wheelbase, a = CoM distance ahead of the rear contact.
Gyroscopic and tyre effects are ignored, so the model has no self-stability:
the rider (controller) must actively balance at every speed.

Task: stay upright and track a heading reference that jumps every 3-5 s.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mobility_rl.envs.common import RANDOMIZATION_MODES, wrap_angle

G = 9.81
DELTA_MAX = np.deg2rad(20.0)
FALL_ANGLE = np.deg2rad(60.0)
NOMINAL = dict(h=0.65, b=1.45, a=0.65, tau=0.06)

# Parameter ranges per randomization mode.
#   none  : nominal vehicle, only the operating point (speed) and references vary
#   train : domain randomization used during training
#   shift : harder, partly out-of-distribution ranges used only for testing
RANGES = {
    "none": dict(v=(3.0, 12.0), h=(0.65, 0.65), b=(1.45, 1.45), a=(0.65, 0.65),
                 tau=(0.06, 0.06), delay=(0, 0), dist=0.0, gust=0.0, noise=0.0),
    "train": dict(v=(3.0, 12.0), h=(0.55, 0.75), b=(1.35, 1.55), a=(0.55, 0.75),
                  tau=(0.04, 0.08), delay=(0, 1), dist=0.5, gust=0.3, noise=1.0),
    "shift": dict(v=(2.5, 14.0), h=(0.75, 0.90), b=(1.50, 1.70), a=(0.50, 0.85),
                  tau=(0.08, 0.12), delay=(1, 2), dist=1.0, gust=0.6, noise=2.0),
}


class MotorcycleEnv(gym.Env):
    """Observation (8, normalised):
        lean, lean rate, steer angle, heading error, yaw rate,
        integral of heading error, speed, previous action
    Action (1): steering command in [-1, 1] -> [-20 deg, +20 deg]
    """

    metadata = {"render_modes": []}

    def __init__(self, randomize="train", dt=0.02, substeps=4, episode_seconds=20.0):
        assert randomize in RANDOMIZATION_MODES
        self.randomize = randomize
        self.dt = dt
        self.substeps = substeps
        self.max_steps = int(episode_seconds / dt)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    # ------------------------------------------------------------------ setup
    def _sample_params(self):
        r, u = RANGES[self.randomize], self.np_random.uniform
        p = {k: u(*r[k]) for k in ("v", "h", "b", "a", "tau")}
        p["delay"] = int(self.np_random.integers(r["delay"][0], r["delay"][1] + 1))
        p["dist"] = u(-r["dist"], r["dist"])  # constant lean disturbance [rad/s^2]
        p["gust"] = r["gust"]
        p["noise"] = r["noise"]
        return p

    def _new_reference(self):
        self.psi_ref = wrap_angle(self.psi_ref + self.np_random.uniform(-np.pi / 4, np.pi / 4))
        self.next_ref_step = self.step_count + int(self.np_random.uniform(3.0, 5.0) / self.dt)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.p = self._sample_params()
        if options and "params" in options:
            self.p.update(options["params"])
        u = self.np_random.uniform
        self.state = np.array([u(-0.05, 0.05), u(-0.05, 0.05), 0.0, 0.0])  # phi, phi_d, delta, psi
        self.step_count = 0
        self.psi_ref = 0.0
        self.next_ref_step = int(u(1.0, 2.0) / self.dt)
        self.int_err = 0.0
        self.prev_action = 0.0
        self.cmd_queue = [0.0] * self.p["delay"]
        self.gust = 0.0
        return self._obs(), self._info()

    # --------------------------------------------------------------- dynamics
    def _derivs(self, s, delta_cmd, d):
        phi, phi_d, delta, _ = s
        p = self.p
        v, h, b, a = p["v"], p["h"], p["b"], p["a"]
        delta_d = (delta_cmd - delta) / p["tau"]
        phi_dd = (G / h) * np.sin(phi) - np.cos(phi) * (
            v**2 / (b * h) * np.tan(delta) + a * v / (b * h) * delta_d / np.cos(delta) ** 2
        ) + d
        psi_d = v * np.tan(delta) / b
        return np.array([phi_d, phi_dd, delta_d, psi_d])

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        self.cmd_queue.append(a * DELTA_MAX)
        delta_cmd = self.cmd_queue.pop(0)

        # Ornstein-Uhlenbeck gust on top of the constant disturbance
        self.gust += -self.gust * self.dt / 0.5 + self.p["gust"] * np.sqrt(self.dt) * self.np_random.normal()
        d = self.p["dist"] + self.gust

        h = self.dt / self.substeps
        for _ in range(self.substeps):  # RK4
            s = self.state
            k1 = self._derivs(s, delta_cmd, d)
            k2 = self._derivs(s + h / 2 * k1, delta_cmd, d)
            k3 = self._derivs(s + h / 2 * k2, delta_cmd, d)
            k4 = self._derivs(s + h * k3, delta_cmd, d)
            self.state = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.state[2] = np.clip(self.state[2], -DELTA_MAX, DELTA_MAX)

        self.step_count += 1
        if self.step_count >= self.next_ref_step:
            self._new_reference()

        err = self.heading_error
        self.int_err = float(np.clip(self.int_err + err * self.dt, -1.0, 1.0))
        phi, phi_d = self.state[0], self.state[1]
        fell = abs(phi) > FALL_ANGLE

        reward = 1.0 - 0.5 * err**2 - 0.05 * phi_d**2 - 0.1 * (a - self.prev_action) ** 2
        if fell:
            reward -= 10.0
        self.prev_action = a
        truncated = self.step_count >= self.max_steps
        return self._obs(), float(reward), bool(fell), bool(truncated), self._info()

    # ----------------------------------------------------------- observation
    @property
    def heading_error(self):
        return wrap_angle(self.psi_ref - self.state[3])

    @property
    def yaw_rate(self):
        return self.p["v"] * np.tan(self.state[2]) / self.p["b"]

    def measure(self):
        """Noisy sensor readings (IMU lean/lean-rate, steer encoder, heading, yaw rate)."""
        n = self.p["noise"]
        nr = self.np_random.normal
        phi, phi_d, delta, _ = self.state
        return dict(
            phi=phi + n * np.deg2rad(0.3) * nr(),
            phi_d=phi_d + n * np.deg2rad(1.0) * nr(),
            delta=delta + n * np.deg2rad(0.2) * nr(),
            err=wrap_angle(self.heading_error + n * np.deg2rad(0.5) * nr()),
            yaw_rate=self.yaw_rate + n * np.deg2rad(0.5) * nr(),
            int_err=self.int_err,
            v=self.p["v"],
            prev_action=self.prev_action,
        )

    def _obs(self):
        m = self.measure()
        self.last_measurement = m
        return np.array([
            m["phi"] / 0.5, m["phi_d"] / 2.0, m["delta"] / DELTA_MAX, m["err"] / (np.pi / 4),
            m["yaw_rate"], m["int_err"], m["v"] / 12.0, m["prev_action"],
        ], dtype=np.float32)

    def _info(self):
        return dict(
            heading_error=float(self.heading_error), lean=float(self.state[0]),
            steer=float(self.state[2]), psi=float(self.state[3]), psi_ref=float(self.psi_ref),
            v=float(self.p["v"]),
        )
