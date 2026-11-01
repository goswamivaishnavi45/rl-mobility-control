#!/usr/bin/env bash
# Finish the project on a new machine (Linux / macOS / Git Bash on Windows).
#   1. set up the venv   2. run tests   3. train residual-RL agents
#   4. evaluate everything   5. export ONNX   6. plots   7. commit + push
# Already-trained agents (runs/*/best_model.zip) and tuned baseline gains are in the repo,
# so only the residual agents are trained here. Usage:  bash continue.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else
  python3 -m venv .venv 2>/dev/null || python -m venv .venv
  if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
  $PY -m pip install -q --upgrade pip
  $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  $PY -m pip install -q -r requirements.txt
fi
mkdir -p logs

echo "== tests"
$PY -m pytest -q tests

echo "== training residual PPO agents (~1 h on a laptop CPU)"
[ -f runs/motorcycle_PPO_res/best_model.zip ] || \
  $PY scripts/train.py --env motorcycle --algo PPO --residual --steps 1000000 > logs/motorcycle_PPO_res.log 2>&1 &
[ -f runs/boat_PPO_res/best_model.zip ] || \
  $PY scripts/train.py --env boat --algo PPO --residual --gamma 0.995 --steps 600000 > logs/boat_PPO_res.log 2>&1 &
wait
tail -n 1 logs/*_PPO_res.log 2>/dev/null || true

echo "== evaluating all controllers (100 episodes x 2 test sets)"
$PY scripts/evaluate.py --envs motorcycle > logs/eval_motorcycle.log 2>&1 &
$PY scripts/evaluate.py --envs boat > logs/eval_boat.log 2>&1 &
wait
cat results/RESULTS.md

echo "== ONNX export + latency"
$PY scripts/export_onnx.py

echo "== figures"
$PY scripts/plot.py

echo "== commit + push"
git add -A results models runs/*/best_model.zip runs/*/evaluations.npz
git commit -m "Add residual-RL agents, full evaluation, ONNX models and figures" || true
git push
echo "DONE. Paste results/RESULTS.md into the README results section."
