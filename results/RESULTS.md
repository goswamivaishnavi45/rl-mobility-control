# Results (100 episodes per cell, identical seeds for every controller)


## boat

| Controller | Split | Mean return | Mean abs heading error (deg) | Fall rate | Action rate (1/s) |
|---|---|---|---|---|---|
| PID | train | 734.2 | 9.33 | 0.0% | 0.99 |
| PID | shift | 710.6 | 10.89 | 0.0% | 1.63 |
| LQR | train | 734.3 | 9.49 | 0.0% | 0.97 |
| LQR | shift | 713.0 | 10.72 | 0.0% | 1.68 |
| PPO | train | 694.3 | 12.72 | 0.0% | 0.21 |
| PPO | shift | 632.9 | 15.49 | 0.0% | 0.28 |
| SAC | train | 601.2 | 18.43 | 0.0% | 2.34 |
| SAC | shift | 567.7 | 19.42 | 0.0% | 2.41 |
| Residual PPO | train | 731.7 | 9.87 | 0.0% | 0.81 |
| Residual PPO | shift | 710.8 | 11.04 | 0.0% | 1.44 |

## motorcycle

| Controller | Split | Mean return | Mean abs heading error (deg) | Fall rate | Action rate (1/s) |
|---|---|---|---|---|---|
| PID | train | 960.1 | 10.73 | 0.0% | 1.30 |
| PID | shift | 888.7 | 12.78 | 11.0% | 1.21 |
| LQR | train | 974.0 | 6.92 | 0.0% | 2.48 |
| LQR | shift | 870.3 | 7.76 | 18.0% | 3.78 |
| PPO | train | 947.0 | 12.18 | 2.0% | 1.83 |
| PPO | shift | 758.3 | 14.76 | 24.0% | 3.70 |
| SAC | train | 790.0 | 24.72 | 11.0% | 6.11 |
| SAC | shift | 646.8 | 26.90 | 28.0% | 7.40 |
| PPO (no DR) | train | 879.9 | 11.55 | 15.0% | 1.21 |
| PPO (no DR) | shift | 589.6 | 12.55 | 53.0% | 2.40 |
| Residual PPO | train | 974.0 | 7.09 | 0.0% | 1.85 |
| Residual PPO | shift | 872.8 | 7.95 | 17.0% | 3.02 |
