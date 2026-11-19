Diffusion model开环评估命令：
python -m evaluation.eval_diffusion_open_loop ^
  --checkpoint outputs\anchor_ablation\diffusion_ckpt\A2\best.pt ^
  --num-samples 64 ^
  --batch-size 32 ^
  --seeds 0,1,2,3,4 ^
  --output-dir outputs\evaluation\open_diffusion\A2


Diffusion model闭环评估命令：
python -m evaluation.eval_diffusion_closed_loop ^
  --checkpoint outputs\anchor_ablation\diffusion_ckpt\A0\best.pt ^
  --methods "TRACE,Expert Planner" ^
  --scenarios straight,curved,merge_in,merge_out ^
  --episodes 10 ^
  --copy-dlls ^
  --open-loop-dir F0=outputs\evaluation\paper_open_F0 ^
  --open-loop-dir F3=outputs\evaluation\paper_open_F3 ^
  --output-dir outputs\evaluation\closed_diffusion\A0