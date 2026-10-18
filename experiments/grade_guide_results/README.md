1. 运行实验采集数据
python -m experiments.grade_duide_results.multi_seed ^
  --root . ^
  --checkpoint ..\all_merge-gru\outputs\gru_coarse_goal_ablation_100ep_3seed\seed_0\mlp_dense\run_1\checkpoints\best.pt ^
  --output-dir outputs\reward_guided_mean_v5 ^
  --seeds 0,1,2 ^
  --scenario all ^
  --samples-per-scenario 20 ^
  --group-size 8 ^
  --step-sizes-m 0,0.10,0.30,0.60,1.0,2.0,4.0,8.0,12.0,16.0 ^
  --random-repeats 2 ^
  --resume ^
  --no-plot

2. 绘图
python -m experiments.grade_duide_results.data_plot ^
  --input-dir outputs\reward_guided_mean_v5 ^
  --gallery-amplitude-m 0.6 ^
  --gallery-count 8

3. 也支持调用绘图函数
from experiments.grade_duide_results.data_plot import plot_results

plot_results(
    input_dir="outputs/reward_guided_mean_v5",
    gallery_amplitude_m=0.6,
    gallery_count=8,
    output_dir="outputs/reward_guided_mean_v5/figures",
)