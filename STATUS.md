# Project status (2026-11-01)

## Done
- Motorcycle + boat simulators with domain randomization, 13 passing tests
- PID / LQR baselines tuned by random search → `results/baselines/*.json`
- Trained agents (best checkpoints in `runs/*/best_model.zip`):
  | Run | Steps |
  |---|---|
  | motorcycle_PPO_dr | 1,000,000 |
  | motorcycle_PPO_nodr (ablation) | 1,000,000 |
  | motorcycle_SAC_dr | 120,000 |
  | boat_PPO_dr | 600,000 |
  | boat_SAC_dr | 100,000 |
- Residual-RL code (`mobility_rl/residual.py`, `train.py --residual`), smoke-tested:
  ONNX export parity 1.9e-8, 27 µs median latency
- Full boat evaluation → `results/metrics_boat.json`

## Results so far (100 episodes per row, same seeds for every controller)
| Env | Controller | Test set | Mean abs heading error | Fall rate |
|---|---|---|---|---|
| boat | PID | train / shift | 9.33° / 10.89° | 0 / 0 |
| boat | LQR | train / shift | 9.49° / 10.72° | 0 / 0 |
| boat | PPO | train / shift | 12.72° / 15.49° | 0 / 0 |
| boat | SAC | train / shift | 18.45° / 19.40° | 0 / 0 |
| motorcycle | PID | train / shift | 10.73° / 12.78° | 0% / 11% |
| motorcycle | LQR | train / shift | 6.92° / 7.76° | 0% / 18% |
| motorcycle | PPO | train / shift | 12.18° / 14.76° | 2% / 24% |

**Finding:** pure end-to-end RL does not beat the carefully tuned PID/LQR baselines on these low-dimensional tasks. That's why the next step is **residual RL**: the policy learns a bounded correction on top of the tuned LQR. It starts from LQR-level performance and only has to learn what the linear controller gets wrong (saturation, delay, model mismatch).

## Remaining (run `bash continue.sh` or `powershell -ExecutionPolicy Bypass -File continue.ps1`)
1. Train residual PPO for motorcycle (1M steps) and boat (600k steps, γ = 0.995), about 1 h on a laptop CPU
2. Evaluate everything, including motorcycle SAC / PPO (no DR) / Residual PPO, which were not yet evaluated
3. Export ONNX + latency benchmark, generate figures
4. Commit + push (the script does this)
5. Put the final numbers from `results/RESULTS.md` into the README `<!-- RESULTS -->` section and the resume bullets
