"""Train a PPO or SAC agent (Stable-Baselines3) on one of the environments.

    python scripts/train.py --env motorcycle --algo SAC --steps 300000
    python scripts/train.py --env motorcycle --algo PPO --randomize none   # ablation: no domain randomization
    python scripts/train.py --env motorcycle --algo PPO --residual         # residual RL on top of tuned LQR
"""

import argparse
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mobility_rl import make_env  # noqa: E402
from mobility_rl.residual import ResidualWrapper, tuned_baseline  # noqa: E402

HYPERPARAMS = {
    "PPO": dict(n_steps=1024, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0)),
    # 2 gradient steps every 4 env steps + small nets: ~4x faster on a laptop CPU than SB3 defaults
    "SAC": dict(buffer_size=300_000, batch_size=128, gamma=0.99, tau=0.005, learning_rate=3e-4,
                learning_starts=5_000, train_freq=4, gradient_steps=2, policy_kwargs=dict(net_arch=[64, 64])),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["motorcycle", "boat"])
    ap.add_argument("--algo", required=True, choices=["PPO", "SAC"])
    ap.add_argument("--randomize", default="train", choices=["none", "train"])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--residual", action="store_true", help="learn a correction on top of the tuned LQR")
    ap.add_argument("--gamma", type=float, default=0.99)
    args = ap.parse_args()
    torch.set_num_threads(1)

    run = f"{args.env}_{args.algo}_{'res' if args.residual else 'dr' if args.randomize == 'train' else 'nodr'}"
    out = ROOT / "runs" / run
    out.mkdir(parents=True, exist_ok=True)

    n_envs = 8 if args.algo == "PPO" else 1
    def factory():
        env = make_env(args.env, args.randomize)
        return ResidualWrapper(env, tuned_baseline(args.env, "LQR")) if args.residual else env

    env = make_vec_env(factory, n_envs=n_envs, seed=args.seed, monitor_dir=str(out / "monitor"))
    # Model selection always uses the training distribution of *this* run
    eval_env = Monitor(factory())
    eval_cb = EvalCallback(eval_env, n_eval_episodes=10, eval_freq=max(20_000 // n_envs, 1),
                           best_model_save_path=str(out), log_path=str(out), deterministic=True, verbose=0)

    algo = {"PPO": PPO, "SAC": SAC}[args.algo]
    hp = {**HYPERPARAMS[args.algo], "gamma": args.gamma}
    if args.residual and args.algo == "PPO":
        # start close to the base controller: small initial exploration noise on the correction
        hp["policy_kwargs"] = {**hp["policy_kwargs"], "log_std_init": -1.5}
    model = algo("MlpPolicy", env, seed=args.seed, verbose=0, tensorboard_log=str(ROOT / "runs" / "tb"), **hp)
    model.learn(total_timesteps=args.steps, callback=eval_cb, tb_log_name=run, progress_bar=False)
    model.save(out / "final_model")
    print(f"done: {run}, best model at {out / 'best_model.zip'}")


if __name__ == "__main__":
    main()
