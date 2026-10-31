import numpy as np

RANDOMIZATION_MODES = ("none", "train", "shift")


def wrap_angle(x):
    """Wrap an angle to [-pi, pi)."""
    return (x + np.pi) % (2 * np.pi) - np.pi
