# Project status (2026-11-01)

## Done
- Motorcycle + boat simulators with domain randomization, 13 passing tests
- PID / LQR baselines tuned by random search → `results/baselines/*.json`
- Trained agents (best checkpoints in `runs/*/best_model.zip`), one training seed each:
  | Run | Steps |
  |---|---|
  | motorcycle_PPO_dr | 1,000,000 |
  | motorcycle_PPO_nodr (ablation) | 1,000,000 |
  | motorcycle_SAC_dr | 120,000 |
  | motorcycle_PPO_res | 1,000,000 |
  | boat_PPO_dr | 600,000 |
  | boat_SAC_dr | 100,000 |
  | boat_PPO_res (γ = 0.995) | 600,000 |
- Full evaluation of every controller → `results/metrics.json`, `results/RESULTS.md`
- ONNX export + latency benchmark → `models/*.onnx`, `results/onnx_benchmark.json`
- Figures → `results/figures/`

## Results (100 episodes per row, same seeds for every controller)
| Env | Controller | Test set | Mean return | Mean abs heading error | Fall rate |
|---|---|---|---|---|---|
| motorcycle | PID | train / shift | 960.1 / 888.7 | 10.73° / 12.78° | 0% / 11% |
| motorcycle | LQR | train / shift | 974.0 / 870.3 | 6.92° / 7.76° | 0% / 18% |
| motorcycle | Residual PPO | train / shift | 974.0 / 872.8 | 7.09° / 7.95° | 0% / 17% |
| motorcycle | PPO | train / shift | 947.0 / 758.3 | 12.18° / 14.76° | 2% / 24% |
| motorcycle | PPO (no DR) | train / shift | 879.9 / 589.6 | 11.55° / 12.55° | 15% / 53% |
| motorcycle | SAC | train / shift | 790.0 / 646.8 | 24.72° / 26.90° | 11% / 28% |
| boat | PID | train / shift | 734.2 / 710.6 | 9.33° / 10.89° | 0 / 0 |
| boat | LQR | train / shift | 734.3 / 713.0 | 9.49° / 10.72° | 0 / 0 |
| boat | Residual PPO | train / shift | 731.7 / 710.8 | 9.87° / 11.04° | 0 / 0 |
| boat | PPO | train / shift | 694.3 / 632.9 | 12.72° / 15.49° | 0 / 0 |
| boat | SAC | train / shift | 601.2 / 567.7 | 18.43° / 19.42° | 0 / 0 |

Heading error is averaged only over episodes without a fall, so read it alongside fall rate.

**Finding:** RL does not beat the tuned PID/LQR baselines on these low-dimensional tasks. End-to-end PPO and SAC lose clearly. Residual PPO essentially matches LQR: the motorcycle shift fall rate differs by one episode (17 vs 18) and boat return is about 0.3% lower. Its measurable gain over LQR is a lower action rate (motorcycle 25% / 20%, boat 16% / 14%, train / shift), at the cost of about 0.2–0.4° more heading error. On the motorcycle shift set, PID has the fewest falls and the highest return, but about 5° more heading error than LQR.

## Possible next steps
- Train more seeds per agent, so differences between controllers come with error bars
- Give SAC the same step budget as PPO before comparing the two
- Add a harder task where a linear controller has more to lose (see Limitations in the README)
