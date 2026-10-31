#!/usr/bin/env bash
# Launches every training + tuning job in parallel; logs go to logs/.
PY=.venv/Scripts/python.exe
$PY scripts/train.py --env motorcycle --algo PPO --steps 2000000 > logs/motorcycle_PPO_dr.log 2>&1 &
$PY scripts/train.py --env motorcycle --algo PPO --steps 2000000 --randomize none > logs/motorcycle_PPO_nodr.log 2>&1 &
$PY scripts/train.py --env motorcycle --algo SAC --steps 200000 > logs/motorcycle_SAC_dr.log 2>&1 &
$PY scripts/train.py --env boat --algo PPO --steps 1000000 > logs/boat_PPO_dr.log 2>&1 &
$PY scripts/train.py --env boat --algo SAC --steps 150000 > logs/boat_SAC_dr.log 2>&1 &
for e in motorcycle boat; do for c in PID LQR; do
  $PY scripts/tune_baselines.py --env $e --controller $c > logs/tune_${e}_${c}.log 2>&1 &
done; done
wait
echo ALL DONE
