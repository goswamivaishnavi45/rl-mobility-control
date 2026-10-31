"""Classical baselines: gain-scheduled cascaded PID and LQR (with integral action).

Every controller reads the same noisy measurements as the RL policy
(`env.last_measurement`) and is designed on the *nominal* model only, so it
sees the same model mismatch the policy sees under domain randomization.
"""

import numpy as np
from scipy.linalg import solve_continuous_are

from mobility_rl.envs import boat, motorcycle


def lqr_gain(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


class Controller:
    name = "controller"

    def reset(self, env):
        pass

    def __call__(self, env):
        raise NotImplementedError


# ------------------------------------------------------------------ motorcycle
class MotorcyclePID(Controller):
    """Cascade: outer heading PID -> lean-angle setpoint; inner lean PD -> steering.

    Inner-loop gains are scheduled with speed (control authority grows ~v^2).
    """

    name = "PID"
    DEFAULT = dict(kh=0.8, ki=0.1, kr=0.3, kp=2.0, kd=0.3, v_ref=6.0, sched=1.0, lean_max=0.6)

    def __init__(self, **gains):
        self.g = {**self.DEFAULT, **gains}

    def __call__(self, env):
        m, g = env.last_measurement, self.g
        phi_ref = np.clip(g["kh"] * m["err"] + g["ki"] * m["int_err"] - g["kr"] * m["yaw_rate"],
                          -g["lean_max"], g["lean_max"])
        scale = (g["v_ref"] / m["v"]) ** g["sched"]
        delta_cmd = scale * (g["kp"] * (m["phi"] - phi_ref) + g["kd"] * m["phi_d"])
        return np.array([np.clip(delta_cmd / motorcycle.DELTA_MAX, -1, 1)], dtype=np.float32)


class MotorcycleLQR(Controller):
    """LQR on the linearised nominal model, re-solved for the measured speed.

    x = [lean, lean rate, steer, heading offset, integral of heading offset]
    The heading error is saturated at `err_sat` (a simple reference governor),
    otherwise large heading steps make the linear law lean further than the
    steering can recover from at low speed.
    """

    name = "LQR"
    DEFAULT = dict(q_phi=1.0, q_phid=0.1, q_delta=0.1, q_psi=5.0, q_int=1.0, r=1.0, err_sat=0.3)

    def __init__(self, **weights):
        self.w = {**self.DEFAULT, **weights}
        self._cache = {}

    @staticmethod
    def linear_model(v, h, b, a, tau):
        g = motorcycle.G
        A = np.array([
            [0, 1, 0, 0, 0],
            [g / h, 0, -v**2 / (b * h) + a * v / (b * h * tau), 0, 0],
            [0, 0, -1 / tau, 0, 0],
            [0, 0, v / b, 0, 0],
            [0, 0, 0, 1, 0],
        ])
        B = np.array([[0], [-a * v / (b * h * tau)], [1 / tau], [0], [0]])
        return A, B

    def gain(self, v):
        key = round(v, 1)
        if key not in self._cache:
            A, B = self.linear_model(key, **motorcycle.NOMINAL)
            w = self.w
            Q = np.diag([w["q_phi"], w["q_phid"], w["q_delta"], w["q_psi"], w["q_int"]])
            self._cache[key] = lqr_gain(A, B, Q, np.array([[w["r"]]]))
        return self._cache[key]

    def __call__(self, env):
        m = env.last_measurement
        err = np.clip(m["err"], -self.w["err_sat"], self.w["err_sat"])
        x = np.array([m["phi"], m["phi_d"], m["delta"], -err, -m["int_err"]])
        u = float(-(self.gain(m["v"]) @ x)[0])
        return np.array([np.clip(u / motorcycle.DELTA_MAX, -1, 1)], dtype=np.float32)


# ------------------------------------------------------------------------ boat
class BoatPID(Controller):
    """Classical autopilot: rudder = Kp * e + Ki * integral(e) - Kd * yaw rate."""

    name = "PID"
    DEFAULT = dict(kp=1.5, ki=0.05, kd=6.0)

    def __init__(self, **gains):
        self.g = {**self.DEFAULT, **gains}

    def __call__(self, env):
        m, g = env.last_measurement, self.g
        delta_cmd = g["kp"] * m["err"] + g["ki"] * m["int_err"] - g["kd"] * m["r"]
        return np.array([np.clip(delta_cmd / boat.DELTA_MAX, -1, 1)], dtype=np.float32)


class BoatLQR(Controller):
    """LQR on the linearised nominal Nomoto model with integral action.

    x = [heading offset, yaw rate, rudder angle, integral of heading offset]
    """

    name = "LQR"
    DEFAULT = dict(q_psi=1.0, q_r=1.0, q_delta=0.1, q_int=0.01, r=1.0)

    def __init__(self, **weights):
        self.w = {**self.DEFAULT, **weights}
        n = boat.NOMINAL
        K, T, tau = n["K"], n["T"], n["tau_r"]
        A = np.array([
            [0, 1, 0, 0],
            [0, -1 / T, K / T, 0],
            [0, 0, -1 / tau, 0],
            [1, 0, 0, 0],
        ])
        B = np.array([[0], [0], [1 / tau], [0]])
        w = self.w
        Q = np.diag([w["q_psi"], w["q_r"], w["q_delta"], w["q_int"]])
        self.K = lqr_gain(A, B, Q, np.array([[w["r"]]]))

    def __call__(self, env):
        m = env.last_measurement
        x = np.array([-m["err"], m["r"], m["delta"], -m["int_err"]])
        u = float(-(self.K @ x)[0])
        return np.array([np.clip(u / boat.DELTA_MAX, -1, 1)], dtype=np.float32)


BASELINES = {
    "motorcycle": {"PID": MotorcyclePID, "LQR": MotorcycleLQR},
    "boat": {"PID": BoatPID, "LQR": BoatLQR},
}
