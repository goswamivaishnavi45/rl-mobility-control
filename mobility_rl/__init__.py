from gymnasium.envs.registration import register

from mobility_rl.envs import BoatEnv, MotorcycleEnv

ENVS = {"motorcycle": MotorcycleEnv, "boat": BoatEnv}

register(id="MotorcycleBalance-v0", entry_point="mobility_rl.envs:MotorcycleEnv")
register(id="BoatHeading-v0", entry_point="mobility_rl.envs:BoatEnv")


def make_env(name, randomize="train", **kwargs):
    return ENVS[name](randomize=randomize, **kwargs)
