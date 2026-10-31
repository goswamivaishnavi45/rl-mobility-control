"""Export trained policies to ONNX, check parity with PyTorch and benchmark latency.

The exported graph maps observation -> deterministic action in [-1, 1], so it
can be loaded by ONNX Runtime from Python or C++ on an embedded controller.

    python scripts/export_onnx.py
Writes models/<env>_<algo>.onnx and results/onnx_benchmark.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from stable_baselines3 import PPO, SAC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mobility_rl import make_env  # noqa: E402


class DeterministicActor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.policy = model.policy
        self.is_sac = isinstance(model, SAC)

    def forward(self, obs):
        if self.is_sac:
            return self.policy.actor(obs, deterministic=True)  # already tanh-squashed
        features = self.policy.extract_features(obs, self.policy.pi_features_extractor)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return torch.clamp(self.policy.action_net(latent_pi), -1.0, 1.0)


def export(env_name, algo):
    path = ROOT / "runs" / f"{env_name}_{algo}_dr" / "best_model.zip"
    if not path.exists():
        return None
    model = {"PPO": PPO, "SAC": SAC}[algo].load(path, device="cpu")
    actor = DeterministicActor(model).eval()
    obs_dim = make_env(env_name).observation_space.shape[0]

    out = ROOT / "models" / f"{env_name}_{algo}.onnx"
    out.parent.mkdir(exist_ok=True)
    torch.onnx.export(actor, torch.zeros(1, obs_dim), str(out), input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)

    # Parity: ONNX output vs SB3 predict() on real observations
    env = make_env(env_name, "train")
    obs_batch = []
    obs, _ = env.reset(seed=0)
    for _ in range(500):
        obs_batch.append(obs)
        obs, _, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
        if term or trunc:
            obs, _ = env.reset()
    obs_batch = np.array(obs_batch, dtype=np.float32)
    sb3 = np.clip(model.predict(obs_batch, deterministic=True)[0], -1, 1)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(out), opts, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"obs": obs_batch})[0]
    max_diff = float(np.abs(onnx_out - sb3).max())

    # Latency: single-observation inference, as in a real-time control loop
    x = obs_batch[:1]
    for _ in range(200):
        sess.run(None, {"obs": x})
    times = []
    for _ in range(5000):
        t = time.perf_counter()
        sess.run(None, {"obs": x})
        times.append(time.perf_counter() - t)
    times = np.array(times) * 1e6
    res = dict(file=out.name, size_kb=round(out.stat().st_size / 1024, 1), max_abs_diff_vs_pytorch=max_diff,
               latency_us_median=float(np.median(times)), latency_us_p99=float(np.percentile(times, 99)))
    print(env_name, algo, res)
    return res


def main():
    results = {}
    for env_name in ("motorcycle", "boat"):
        for algo in ("PPO", "SAC"):
            r = export(env_name, algo)
            if r:
                results[f"{env_name}_{algo}"] = r
    (ROOT / "results" / "onnx_benchmark.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
