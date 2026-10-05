@echo off
setlocal
cd /d D:\KONG_Files\AllMerge\all_merge-grpo

set CKPT=D:\KONG_Files\AllMerge\all_merge-gru\outputs\gru_coarse_goal_ablation_30ep\gru_refine_dense\run_1\checkpoints\best.pt
set OUT=outputs\grpo_vanilla_learning_probe\one_epoch

python scripts\run_vanilla_learning_probe.py ^
  --checkpoint "%CKPT%" ^
  --scenario curved ^
  --group-action 3 ^
  --steps 30 ^
  --group-size 48 ^
  --lr 5e-7 ^
  --eta 0.02 ^
  --clip-eps 0.2 ^
  --kl-coef 0.05 ^
  --max-grad-norm 5 ^
  --update-epochs 1 ^
  --task-reward-type progress_comfort ^
  --state-seed 7 ^
  --train-noise-seed 70007 ^
  --eval-noise-seed-base 170007 ^
  --eval-noise-seeds 4 ^
  --eval-every 1 ^
  --trend-window 10 ^
  --output-dir "%OUT%"
if errorlevel 1 exit /b 1

python scripts\analyze_vanilla_learning_probe.py "%OUT%\learning_probe.csv"
if errorlevel 1 exit /b 1

echo [OK] Vanilla GRPO one-epoch learnability probe complete.
endlocal
