# Finish the project on a new Windows machine (PowerShell). Same steps as continue.sh.
# Usage:  powershell -ExecutionPolicy Bypass -File continue.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$PY = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $PY)) {
    python -m venv .venv
    & $PY -m pip install -q --upgrade pip
    & $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    & $PY -m pip install -q -r requirements.txt
}
New-Item -ItemType Directory -Force logs | Out-Null

Write-Host "== tests"
& $PY -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

Write-Host "== training residual PPO agents (~1 h on a laptop CPU)"
$jobs = @()
if (-not (Test-Path runs\motorcycle_PPO_res\best_model.zip)) {
    $jobs += Start-Process $PY -ArgumentList "scripts/train.py --env motorcycle --algo PPO --residual --steps 1000000" `
        -RedirectStandardOutput logs\motorcycle_PPO_res.log -RedirectStandardError logs\motorcycle_PPO_res.err -NoNewWindow -PassThru
}
if (-not (Test-Path runs\boat_PPO_res\best_model.zip)) {
    $jobs += Start-Process $PY -ArgumentList "scripts/train.py --env boat --algo PPO --residual --gamma 0.995 --steps 600000" `
        -RedirectStandardOutput logs\boat_PPO_res.log -RedirectStandardError logs\boat_PPO_res.err -NoNewWindow -PassThru
}
$jobs | Wait-Process

Write-Host "== evaluating all controllers (100 episodes x 2 test sets)"
$ev = @(
    (Start-Process $PY -ArgumentList "scripts/evaluate.py --envs motorcycle" -RedirectStandardOutput logs\eval_motorcycle.log -RedirectStandardError logs\eval_motorcycle.err -NoNewWindow -PassThru),
    (Start-Process $PY -ArgumentList "scripts/evaluate.py --envs boat" -RedirectStandardOutput logs\eval_boat.log -RedirectStandardError logs\eval_boat.err -NoNewWindow -PassThru)
)
$ev | Wait-Process
Get-Content results\RESULTS.md

Write-Host "== ONNX export + latency"
& $PY scripts/export_onnx.py

Write-Host "== figures"
& $PY scripts/plot.py

Write-Host "== commit + push"
git add -A results models runs/*/best_model.zip runs/*/evaluations.npz
git commit -m "Add residual-RL agents, full evaluation, ONNX models and figures"
git push
Write-Host "DONE. Paste results/RESULTS.md into the README results section."
