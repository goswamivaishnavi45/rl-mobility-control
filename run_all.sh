#!/usr/bin/env bash
# Launches every training + tuning job in parallel; logs go to logs/.
# Budgets fit a 4-core laptop CPU in ~1 hour (curves plateau well before the end).
PY=.venv/Scripts/python.exe
mkdir -p logs
$PY scripts/train.py --env motorcycle --algo PPO --steps 1000000 > logs/motorcycle_PPO_dr.log 2>&1 &
$PY scripts/train.py --env motorcycle --algo PPO --steps 1000000 --randomize none > logs/motorcycle_PPO_nodr.log 2>&1 &
$PY scripts/train.py --env motorcycle --algo SAC --steps 120000 > logs/motorcycle_SAC_dr.log 2>&1 &
$PY scripts/train.py --env boat --algo PPO --steps 600000 > logs/boat_PPO_dr.log 2>&1 &
$PY scripts/train.py --env boat --algo SAC --steps 100000 > logs/boat_SAC_dr.log 2>&1 &
for e in motorcycle boat; do for c in PID LQR; do
  # skip searches whose tuned gains already exist
  [ -f results/baselines/${e}_${c}.json ] || $PY scripts/tune_baselines.py --env $e --controller $c > logs/tune_${e}_${c}.log 2>&1 &
done; done
wait
echo ALL DONE > logs/ALL_DONE
