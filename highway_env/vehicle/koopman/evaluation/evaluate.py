"""Six-model Koopman paper evaluation on identical frozen episode windows.

Outputs only ADE/FDE/yaw-rate RMSE/roll RMSE/LTR RMSE; the paper's position
error curve and episode-level ADE boxplot; controlled three-case chassis plot
requires independently acquired matching-control TruckSim ground truth.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from highway_env.vehicle.koopman.data.dataset import KoopmanWindowDataset, discover_episodes
from highway_env.vehicle.koopman.evaluation.metrics import (PAPER_COLUMNS, MODELS, get_errors, summarize_all,
                      save_episodes, save_position_curve, save_table_row)
from highway_env.vehicle.koopman.training.neural_train import load_neural
from highway_env.vehicle.koopman.training.edmd_train import load_edmd
from highway_env.vehicle.koopman.training.base_train import choose_device
from highway_env.vehicle.koopman.data.shared_split import resolve,load as load_shared


def slug(kind):return kind.lower().replace(' ','_').replace('-','_')


def model_checkpoint(cfg,project_root,kind):
    return resolve(project_root,Path(cfg['outputs']['checkpoints'])/slug(kind)/'best.pt')


def load_model(kind,path,device):
    if kind in ('K0','K1-P','K1-F','K2'):return load_neural(path,device)
    return load_edmd(path,device)


def _run_model(model,kind,batch,device,dt):
    kw={'initial_pose':batch['target_poses'][:,0].to(device),'dt':dt}
    if kind!='K0':kw['conditions']=batch['conditions'].to(device)
    return model.predict_all(batch['state_hist'].to(device),batch['control_hist'].to(device),
                             batch['future_controls'].to(device),**kw)


def _episode_eval(model,kind,ep,cfg,device,batch_size):
    data=cfg['data'];dt=float(data.get('dt_s',.1));h=int(data.get('horizon_steps',40))
    ds=KoopmanWindowDataset([ep],history_steps=int(data.get('history_steps',5)),
          horizon_steps=h,warmup_s=float(data.get('warmup_s',1.0)),
          window_stride=int(data.get('eval_stride',5)),dt=dt)
    if not len(ds):raise ValueError(f'{ep.id} has no complete 4s evaluation window')
    position=np.zeros(h+1,dtype=np.float64)
    fde=yaw=roll=ltr=0.0;valid_ltr=0;windows=0
    for batch in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0):
        with torch.inference_mode():pred=_run_model(model,kind,batch,device,dt)
        errs=get_errors(pred['states'].cpu().numpy(),batch['target_states'].numpy(),
                        pred['pose'].cpu().numpy(),batch['target_poses'].numpy(),
                        pred['ltr'].cpu().numpy(),batch['target_ltr'].numpy(),batch['ltr_valid'].numpy())
        position+=errs['position_error'].sum(axis=0)
        fde+=errs['position_error'][:,-1].sum()
        yaw+=errs['yaw_sq'].sum()
        roll+=errs['roll_sq'].sum()
        m=errs['ltr_valid'];ltr+=errs['ltr_sq'][m].sum();valid_ltr+=int(m.sum())
        windows+=len(errs['position_error'])
    curve=position/windows
    if not valid_ltr:raise ValueError(f'{ep.id} lacks valid LTR labels')
    row={'episode_id':str(ep.path),'windows':windows,
         'ADE (m)':float(curve[1:].mean()),'FDE (m)':float(fde/windows),
         'Yaw Rate RMSE (rad/s)':float(np.sqrt(yaw/(windows*h))),
         'Roll RMSE (deg)':float(np.sqrt(roll/(windows*h))),
         'LTR RMSE':float(np.sqrt(ltr/valid_ltr)),
         'curve':curve.tolist()}
    if not all(np.isfinite(row[k]) for k in PAPER_COLUMNS):
        raise ValueError(f'{kind} nonfinite evaluation metrics on {ep.id}')
    return row


def evaluate_one(cfg,project_root,kind,split='test',data_root=None,device='auto',batch_size=128):
    if kind not in MODELS:raise ValueError(kind)
    eps,manifest=load_shared(cfg,project_root,data_root)
    selected=eps[split]
    if not selected:raise ValueError(f'No episodes in frozen {split} split; use val for smoke')
    device=choose_device(device)
    ckpath=model_checkpoint(cfg,project_root,kind)
    model,ck=load_model(kind,ckpath,device)
    if ck.get('split_sha256')!=manifest['sha256']:
        raise ValueError(f'{kind} checkpoint uses a different frozen partition')
    rows=[]
    for i,ep in enumerate(selected):
        rows.append(_episode_eval(model,kind,ep,cfg,device,batch_size))
        if (i+1)%25==0:print(f'[EVAL] {kind} {i+1}/{len(selected)} episodes',flush=True)
    summary,curve=summarize_all(rows)
    out=resolve(project_root,Path(cfg['outputs']['evaluation'])/split/slug(kind))
    out.mkdir(parents=True,exist_ok=True)
    save_table_row(out/'table.csv',kind,summary)
    save_table_row(out/'summary.csv',kind,summary)
    save_episodes(out/'episode_metrics.csv',rows)
    save_position_curve(out/'position_curve_k0_compatible.csv',curve,float(cfg['data'].get('dt_s',.1)))
    (out/'summary.json').write_text(json.dumps({'model':kind,'split':split,
        'split_sha256':manifest['sha256'],'metrics':summary,
        'num_episodes':len(rows),'num_windows':sum(r['windows'] for r in rows),
        'dt_s':float(cfg['data'].get('dt_s',.1))},indent=2),encoding='utf-8')
    print(f'[EVAL DONE] {kind} {split} ADE={summary["ADE (m)"]:.4f} '
          f'FDE={summary["FDE (m)"]:.4f} LTR={summary["LTR RMSE"]:.4f}',flush=True)
    return {'summary':summary,'curve':curve,'episodes':rows,'split_sha256':manifest['sha256']}


def _selected_models(models=None):
    """Validate a nonempty model selection, preserving the requested order."""
    selected = tuple(MODELS if models is None else models)
    if not selected:
        raise ValueError('Select at least one model')
    unknown = [kind for kind in selected if kind not in MODELS]
    if unknown:
        raise ValueError(f'Unknown models: {unknown}')
    if len(set(selected)) != len(selected):
        raise ValueError('Model selection contains duplicates')
    return selected


def compare(cfg,project_root,split='test',data_root=None,device='auto',batch_size=128,chassis_seed=None, models=None, chassis_steps=(3,)):
    models = _selected_models(models)
    output={}
    for kind in models:
        output[kind]=evaluate_one(cfg,project_root,kind,split,data_root,device,batch_size)
    out=resolve(project_root,Path(cfg['outputs']['evaluation'])/split)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'koopman_prediction_accuracy.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['Model',*PAPER_COLUMNS])
        for kind in models:w.writerow([kind,*[output[kind]['summary'][k] for k in PAPER_COLUMNS]])
    with (out/'koopman_prediction_errors_curve.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['time_s',*models])
        dt=float(cfg['data'].get('dt_s',.1))
        for j in range(len(output[models[0]]['curve'])):
            w.writerow([j*dt,*[output[kind]['curve'][j] for kind in models]])
    plot_six_models(out/'koopman_prediction_errors.pdf',output,dt,models=models)
    plot_chassis_comparisons(cfg, project_root, out, split=split, data_root=data_root,
                             device=device, seed=chassis_seed, models=models, steps=chassis_steps)
    print(f'[COMPARE] {len(models)} models tested on SAME {split} episodes; paper table and PDF -> {out}',flush=True)
    return output


def _paper_style(plt):
    """Apply consistent journal-style typography and vector PDF settings."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'legend.frameon': False,
    })


PAPER_COLORS = {
    'EDMDc': '#A3C7C5',
    'Parametric EDMDc': '#E07A3F',
    'K0': '#4C9FD6',
    'K1-P': '#55A868',
    'K1-F': '#E3A33D',
    'K2': '#9566B5',
}


def _episode_curve_stats(model_output):
    """Return mean/std across held-out episodes, preserving episode as unit."""
    episodes = model_output.get('episodes', [])
    curves = [np.asarray(ep['curve'], dtype=np.float64)
              for ep in episodes if ep.get('curve') is not None]
    if curves:
        lengths = {len(x) for x in curves}
        if len(lengths) != 1:
            raise ValueError('Episode position-error curves have inconsistent lengths')
        matrix = np.stack(curves, axis=0)
        return matrix.mean(axis=0), matrix.std(axis=0, ddof=1 if len(matrix) > 1 else 0)
    # Backward-compatible fallback for externally supplied aggregate-only results.
    curve = np.asarray(model_output['curve'], dtype=np.float64)
    return curve, np.zeros_like(curve)


def _boxplot_with_points(ax, arrays, labels, colors, *, positions=None,
                         widths=0.52, scatter=True, show_xticklabels=True,
                         fontsize=8, ylim=None):
    """Colored boxplots with lightly jittered episode observations."""
    if positions is None:
        positions = np.arange(1, len(arrays) + 1)
    bp = ax.boxplot(arrays, positions=positions, widths=widths,
                    patch_artist=True, showfliers=False, whis=1.5,
                    medianprops={'color': '#C43C39', 'linewidth': 1.5},
                    whiskerprops={'color': '#555555', 'linewidth': 0.9},
                    capprops={'color': '#555555', 'linewidth': 0.9},
                    boxprops={'edgecolor': '#555555', 'linewidth': 0.9})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.52)
    if scatter:
        rng = np.random.default_rng(20261114)
        for x, values, color in zip(positions, arrays, colors):
            values = np.asarray(values, dtype=float)
            if not len(values):
                continue
            jitter = rng.uniform(-widths * 0.31, widths * 0.31, size=len(values))
            ax.scatter(x + jitter, values, s=4, color=color, alpha=0.3,
                       edgecolors='none', zorder=3)
    ax.set_xticks(positions)
    if show_xticklabels:
        ax.set_xticklabels(labels, rotation=22, ha='right', fontsize=fontsize)
    else:
        ax.set_xticklabels([])
    if ylim is not None:
        ax.set_ylim(*ylim)
    return bp


def plot_six_models(path, output, dt, models=None):
    models = _selected_models(models)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    _paper_style(plt)
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.0),
                            gridspec_kw={'width_ratios': [1.12, 1.0]})
    ax_curve, ax_box = axs
    time_s = None
    curve_stats = {}
    for kind in models:
        mean, std = _episode_curve_stats(output[kind])
        t = np.arange(len(mean), dtype=np.float64) * dt
        time_s = t
        color = PAPER_COLORS[kind]
        ax_curve.plot(t, mean, color=color, linewidth=1.8, label=kind, zorder=3)
        ax_curve.fill_between(t, np.maximum(0.0, mean - std), mean + std,
                              color=color, alpha=0.15, linewidth=0, zorder=2)
        curve_stats[kind] = (mean, std)
    ax_curve.set(xlabel='Prediction horizon (s)', ylabel='Mean position error (m)',
                 title='(a) Multistep position error')
    ax_curve.grid(alpha=0.24, linewidth=0.6)
    ax_curve.legend(fontsize=8, ncol=2, loc='upper left', columnspacing=1.0,
                    handlelength=2.0)
    ax_curve.text(0.99, 0.02, 'Shading: ±1 SD across test episodes',
                  transform=ax_curve.transAxes, ha='right', va='bottom',
                  fontsize=6.8, color='#555555')
    ax_curve.tick_params(labelsize=8.5)

    # Preserve the full-scale comparison while enlarging the four neural variants. 左侧箱线图
    zoom_kinds = tuple(k for k in models if k in ('K0', 'K1-P', 'K1-F', 'K2'))
    if zoom_kinds:
        # Fractions of the parent axes: left/bottom margins and width/height. 
        curve_inset_left = 0.06  # 距主图左边界
        curve_inset_bottom = 0.25  # 距主图底边界
        curve_inset_width = 0.50  # 宽度
        curve_inset_height = 0.50  # 高度
        inset = ax_curve.inset_axes([curve_inset_left, curve_inset_bottom,
                                     curve_inset_width, curve_inset_height])
        for kind in zoom_kinds:
            mean, std = curve_stats[kind]
            t = np.arange(len(mean), dtype=np.float64) * dt
            color = PAPER_COLORS[kind]
            inset.plot(t, mean, color=color, linewidth=1.45)
            inset.fill_between(t, np.maximum(0.0, mean - std), mean + std,
                               color=color, alpha=0.12, linewidth=0)
        all_zoom = np.concatenate([curve_stats[k][0] + curve_stats[k][1]
                                   for k in zoom_kinds])
        upper = max(float(np.quantile(all_zoom, 0.99)) * 1.12, 0.5)
        inset.set_xlim(float(3.0), float(4.0))  # 显示范围
        inset.set_ylim(1.1, 2.6)
        inset.grid(alpha=0.2, linewidth=0.45)
        inset.tick_params(labelsize=6.5, direction='in')
        # inset.text(0.5, 0.97, 'Neural variants', transform=inset.transAxes,
        #            ha='center', va='top', fontsize=6.5)
        mark_inset(ax_curve, inset, loc1=2, loc2=4, fc='none',
                   ec='#666666', lw=0.75, alpha=0.85)

    arrays = [[ep['ADE (m)'] for ep in output[kind]['episodes']] for kind in models]
    colors = [PAPER_COLORS[k] for k in models]
    _boxplot_with_points(ax_box, arrays, models, colors, widths=0.48, fontsize=8)
    ax_box.set_ylabel('Episode-level ADE (m)', fontsize=9)
    ax_box.set_ylim(0, 25)  # 图(b)主图 y 轴范围
    ax_box.set_title('(b) Test episode distributions', fontsize=10, pad=8)
    ax_box.grid(axis='y', alpha=0.24, linewidth=0.6)
    ax_box.tick_params(axis='y', labelsize=8.5)

    # A dedicated inset prevents the large Parametric-EDMDc tail from hiding 右侧箱线图
    # the small but meaningful distribution differences among K0/K1-P/K1-F/K2.
    if zoom_kinds:
        # Anchor this inset using right/top margins, also in parent-axes fractions.
        box_inset_right = 0.08
        box_inset_top = 0.10
        box_inset_width = 0.50
        box_inset_height = 0.50
        box_inset = ax_box.inset_axes([
            1.0 - box_inset_right - box_inset_width,
            1.0 - box_inset_top - box_inset_height,
            box_inset_width, box_inset_height])
        neural_arrays = [arrays[models.index(k)] for k in zoom_kinds]
        neural_colors = [PAPER_COLORS[k] for k in zoom_kinds]
        neural_positions = np.asarray([models.index(k) + 1 for k in zoom_kinds])
        _boxplot_with_points(box_inset, neural_arrays, zoom_kinds, neural_colors,
                             positions=neural_positions, widths=0.48,
                             fontsize=6.3, scatter=True)
        box_inset.set_xlim(neural_positions.min() - 0.5, neural_positions.max() + 0.5)
        finite = np.concatenate([np.asarray(x, dtype=float) for x in neural_arrays if len(x)])
        q_hi = float(np.quantile(finite, 0.985)) if len(finite) else 1.0
        box_inset.set_ylim(0.3, 1.9)  # 右侧箱线图
        box_inset.grid(axis='y', alpha=0.2, linewidth=0.4)
        box_inset.tick_params(axis='y', labelsize=6.2)
        # box_inset.text(0.5, 0.97, 'K0–K2 detail', transform=box_inset.transAxes,
        #                ha='center', va='top', fontsize=6.5)
        box_inset.tick_params(axis='x', labelrotation=0)
        mark_inset(ax_box, box_inset, loc1=2, loc2=4, fc='none',
                   ec='#666666', lw=0.75, alpha=0.85)

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.88, wspace=0.28)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)



def _add_k2_local_detail(ax, time_s, ground_truth, predictions, dt, window_s=0.8):
    """Add one evidence-driven inset where K2 most improves over neural baselines.

    The inset is omitted when K2 has no positive local absolute-error advantage;
    this avoids visually implying an improvement unsupported by the sampled data.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    baseline_kinds = tuple(k for k in ('K0', 'K1-P', 'K1-F') if k in predictions)
    if not baseline_kinds or 'K2' not in predictions:
        return False
    gt = np.asarray(ground_truth, dtype=float)
    k2 = np.asarray(predictions['K2'], dtype=float)
    n = min(len(gt), len(k2), len(time_s))
    if n < 5:
        return False
    gt, k2 = gt[:n], k2[:n]
    k2_error = np.abs(k2 - gt)
    width = max(3, min(n, int(round(window_s / dt))))
    best_kind, start, score = None, 0, -np.inf
    for kind in baseline_kinds:
        pred = np.asarray(predictions[kind], dtype=float)[:n]
        gain = np.abs(pred - gt) - k2_error
        if width >= n:
            candidate_start, candidate_score = 0, float(np.mean(gain))
        else:
            sums = np.convolve(gain, np.ones(width, dtype=float) / width, mode='valid')
            candidate_start, candidate_score = int(np.argmax(sums)), float(np.max(sums))
        if candidate_score > score:
            best_kind, start, score = kind, candidate_start, candidate_score
    if not np.isfinite(score) or score <= 1e-5:
        return False
    stop = min(n, start + width)
    inset = inset_axes(ax, width='43%', height='40%', loc='upper right', borderpad=0.8)
    inset.plot(time_s[:n], gt, color='black', linewidth=1.25, label='TruckSim')
    inset.plot(time_s[:n], k2, color=PAPER_COLORS['K2'], linewidth=1.55, label='K2')
    # Compare K2 with the single neural baseline that gives the largest
    # average improvement over this same local time window.
    inset.plot(time_s[:n], np.asarray(predictions[best_kind])[:n],
               color=PAPER_COLORS[best_kind], linewidth=1.1,
               linestyle={'K0': '--', 'K1-P': '-.', 'K1-F': ':'}[best_kind],
               label=best_kind)
    inset.set_xlim(time_s[start], time_s[min(stop - 1, n - 1)])
    local = np.concatenate([gt[start:stop], k2[start:stop],
                            np.asarray(predictions[best_kind])[start:stop]])
    span = max(float(np.ptp(local)), 1e-4)
    inset.set_ylim(float(np.min(local) - 0.16 * span), float(np.max(local) + 0.16 * span))
    inset.grid(alpha=0.2, linewidth=0.4)
    inset.tick_params(labelsize=5.5, pad=1, direction='in')
    inset.set_title('K2 local detail', fontsize=6.2, pad=1)
    mark_inset(ax, inset, loc1=2, loc2=4, fc='none', ec='#666666', lw=0.55, alpha=0.8)
    return True


# Paper chassis figure: fractions of EACH parent axes (left, bottom, width, height).
CHASSIS_TRUTH_COLOR = '#0072BD'
# Main-panel limits: zero-based (row, column). None keeps automatic limits.
# xlim: seconds; ylim columns: yaw rate (rad/s), roll angle (deg), LTR.
CHASSIS_MAIN_LIMITS = {
    (0, 0): {'xlim': None, 'ylim': (-0.18, 0.3)},
    (0, 1): {'xlim': None, 'ylim': None},
    (0, 2): {'xlim': None, 'ylim': None},
    (1, 0): {'xlim': None, 'ylim': None},
    (1, 1): {'xlim': None, 'ylim': None},
    (1, 2): {'xlim': None, 'ylim': None},
    (2, 0): {'xlim': None, 'ylim': None},
    (2, 1): {'xlim': None, 'ylim': None},
    (2, 2): {'xlim': None, 'ylim': None},
}

CHASSIS_INSET_DEFAULTS = {
    'bounds': (0.54, 0.12, 0.43, 0.40),  # (距左边, 距底边, 宽度, 高度)，均为所在主图的比例
    'xlim': None,             # 时间范围，单位秒；3 步图默认数据时间为 0.3～4.3 秒
    'ylim': None,             # 纵轴范围，三列分别为 rad/s、度、LTR
    'window_s': 0.8,          
    'ylim_pad': 0.16,         # Padding as a fraction of the local response span.
    'connector_corners': (2, 4),
}
# 精确调整koopman_chassis_responses_3step.pdf 3*3组图 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CHASSIS_INSET_OVERRIDES = {
    (0, 0): {
        'bounds': (0.34, 0.1, 0.43, 0.40),
        'xlim': (1.2, 1.8),
        'ylim': (0.14, 0.25),
    }, 
    (0, 1): {}, (0, 2): {},
    (1, 0): {}, (1, 1): {}, (1, 2): {},
    (2, 0): {}, (2, 1): {}, (2, 2): {},
}


def _add_chassis_detail(ax, time_s, ground_truth, predictions, dt, row, col):
    """Always show detail; choose the largest mean neural-baseline minus K2 error.

    A nonpositive gain still gets an inset, without claiming a K2 advantage.
    All selected model curves remain visible for an honest local comparison.
    """
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    settings = {**CHASSIS_INSET_DEFAULTS, **CHASSIS_INSET_OVERRIDES.get((row, col), {})}
    t = np.asarray(time_s, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)
    neural = [k for k in ('K0', 'K1-P', 'K1-F', 'K2') if k in predictions]
    baselines = [k for k in neural if k != 'K2']
    width = min(len(t), max(2, int(round(settings['window_s'] / dt)) + 1))
    start, best_score = 0, -np.inf
    if 'K2' in predictions and baselines:
        k2_error = np.abs(np.asarray(predictions['K2']) - gt)
        baseline_error = np.mean([np.abs(np.asarray(predictions[k]) - gt)
                                  for k in baselines], axis=0)
        gain = baseline_error - k2_error
        for i in range(len(t) - width + 1):
            local_gain = gain[i:i+width]
            if np.isfinite(local_gain).all():
                score = float(np.mean(local_gain))
                if score > best_score:
                    start, best_score = i, score
    xlim = settings['xlim']
    if xlim is None:
        xlim = (float(t[start]), float(t[start + width - 1]))
    if len(xlim) != 2 or xlim[0] >= xlim[1]:
        raise ValueError(f'Invalid chassis inset xlim for panel {(row, col)}')
    inset = ax.inset_axes(settings['bounds'])
    # Copy main-panel styles so the inset and figure legend agree.
    for line in ax.lines:
        inset.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                   linestyle=line.get_linestyle(),
                   linewidth=1.45 if line.get_label() == 'K2' else 1.05,
                   alpha=line.get_alpha(), zorder=line.get_zorder())
    inset.set_xlim(*xlim)
    ylim = settings['ylim']
    if ylim is None:
        mask = (t >= xlim[0]) & (t <= xlim[1])
        if not mask.any():
            raise ValueError(f'Chassis inset xlim contains no samples for panel {(row, col)}')
        detail_kinds = neural or list(predictions)
        local = np.concatenate([gt[mask], *[np.asarray(predictions[k])[mask]
                                           for k in detail_kinds]])
        local = local[np.isfinite(local)]
        if not len(local):
            local = np.array([0.0])
        span = max(float(np.ptp(local)), 1e-4)
        padding = max(settings['ylim_pad'] * span, 1e-6)
        ylim = (float(local.min() - padding), float(local.max() + padding))
    if len(ylim) != 2 or ylim[0] >= ylim[1]:
        raise ValueError(f'Invalid chassis inset ylim for panel {(row, col)}')
    inset.set_ylim(*ylim)
    inset.grid(alpha=0.2, linewidth=0.4)
    inset.tick_params(labelsize=5.5, pad=1, direction='in')
    loc1, loc2 = settings['connector_corners']
    mark_inset(ax, inset, loc1=loc1, loc2=loc2, fc='none',
               ec='#888888', lw=0.55, alpha=0.8)
    return {'xlim': tuple(xlim), 'ylim': tuple(ylim),
            'mean_baseline_minus_k2_error': float(best_score) if np.isfinite(best_score) else None}


def _chassis_prediction_batch(ep, start, history_steps, count, prediction_steps):
    """Independent true-history windows with prediction_steps future controls."""
    origins = range(start, start + count)
    return {
        'state_hist': torch.from_numpy(np.stack([
            ep.states[k-history_steps+1:k+1] for k in origins]).copy()),
        'control_hist': torch.from_numpy(np.stack([
            ep.controls[k-history_steps+1:k] for k in origins]).copy()),
        'future_controls': torch.from_numpy(np.stack([
            ep.controls[k:k+prediction_steps] for k in origins]).copy()),
        'target_poses': torch.from_numpy(np.stack([
            ep.poses[k:k+prediction_steps+1] for k in origins]).copy()),
        'conditions': torch.tensor([
            [float(ep.info['payload_kg']), float(ep.info['road_mu'])]
            for _ in origins], dtype=torch.float32),
    }


def plot_chassis_comparison(cfg, project_root, output_dir, *, split='test',
                            data_root=None, device='auto', seed=None, models=None,
                            prediction_steps=3, sampling_steps=None):
    """One 3x3 PDF of fixed-lead predictions with refreshed true histories.

    At each origin, all models receive the same TRUE history and future controls.
    Within each prediction, latent states recur for prediction_steps transitions.
    The next origin resets to true history; h+1 terminal outputs are plotted.
    This plot is exploratory/diagnostic: sampled episodes need not have matching
    input sequences across rows, so it is not a controlled payload/mu comparison.
    """
    models = _selected_models(models)
    if not isinstance(prediction_steps, int) or prediction_steps < 1:
        raise ValueError('prediction_steps must be a positive integer')
    sampling_steps = max(10, prediction_steps) if sampling_steps is None else sampling_steps
    if sampling_steps < prediction_steps:
        raise ValueError('sampling_steps must cover prediction_steps')
    data = cfg['data']
    h = int(data.get('horizon_steps', 40))
    dt = float(data.get('dt_s', .1))
    history_steps = int(data.get('history_steps', 5))
    splits, manifest = load_shared(cfg, project_root, data_root)
    selected = splits[split]
    if not selected:
        raise ValueError(f'No episodes in frozen {split} split for chassis comparison')
    ds = KoopmanWindowDataset(selected,
        history_steps=history_steps, horizon_steps=h + sampling_steps,
        warmup_s=float(data.get('warmup_s', 1.0)),
        window_stride=1, dt=dt)
    if len(ds) < 3:
        raise ValueError(f'Chassis comparison needs >=3 complete windows in {split}; found {len(ds)}')

    rng = np.random.default_rng(int(cfg.get('training', {}).get('seed', 2026) if seed is None else seed))
    grouped = {}
    for idx, (episode_idx, _) in enumerate(ds.keys):
        grouped.setdefault(episode_idx, []).append(idx)
    # Prefer three independent episodes, falling back to distinct windows if needed.
    if len(grouped) >= 3:
        episode_indices = rng.choice(list(grouped), size=3, replace=False)
        chosen = [int(rng.choice(grouped[int(ei)])) for ei in episode_indices]
    else:
        chosen = [int(i) for i in rng.choice(len(ds), size=3, replace=False)]

    device = choose_device(device)
    loaded_models = {}
    for kind in models:
        model, ck = load_model(kind, model_checkpoint(cfg, project_root, kind), device)
        if ck.get('split_sha256') != manifest['sha256']:
            raise ValueError(f'{kind} checkpoint uses a different frozen partition')
        loaded_models[kind] = model

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _paper_style(plt)
    fig, axs = plt.subplots(3, 3, figsize=(14.0, 10.0), sharex=False,
                            gridspec_kw={'wspace': .22, 'hspace': .27})
    headings = ('Yaw rate (rad/s)', 'Roll angle (deg)', 'LTR')
    for col, heading in enumerate(headings):
        axs[0, col].set_title(heading, fontsize=11, pad=9)
    time_s = np.arange(prediction_steps, h + prediction_steps + 1, dtype=np.float64) * dt
    styles = {
        'EDMDc': (PAPER_COLORS['EDMDc'], ':'),
        'Parametric EDMDc': (PAPER_COLORS['Parametric EDMDc'], '--'),
        'K0': (PAPER_COLORS['K0'], '--'),
        'K1-P': (PAPER_COLORS['K1-P'], '-.'),
        'K1-F': (PAPER_COLORS['K1-F'], ':'),
        'K2': (PAPER_COLORS['K2'], '-'),
    }
    records = []
    try:
        for row, index in enumerate(chosen):
            row_ground_truth = []
            row_predictions = {kind: [] for kind in models}
            ei, start = ds.keys[index]
            ep = ds.episodes[ei]
            batch = _chassis_prediction_batch(ep, start, history_steps, h + 1, prediction_steps)
            # Align the terminal output with the truth prediction_steps ahead.
            true_states = ep.states[start+prediction_steps:start+h+prediction_steps+1]
            true_ltr = np.asarray(ep.ltr[start+prediction_steps:start+h+prediction_steps+1], dtype=np.float64).copy()
            true_ltr[~np.isfinite(true_ltr)] = np.nan
            gt = (true_states[:, 2], np.rad2deg(true_states[:, 3]), true_ltr)
            for col, values in enumerate(gt):
                row_ground_truth.append(np.asarray(values, dtype=float))
                axs[row, col].plot(time_s, values, label='TruckSim', color=CHASSIS_TRUTH_COLOR,
                                   linewidth=2.0, zorder=3)
            for kind in models:
                with torch.inference_mode():
                    prediction = _run_model(loaded_models[kind], kind, batch, device, dt)
                states = prediction['states'][:, prediction_steps].detach().cpu().numpy()
                ltr = prediction['ltr'][:, prediction_steps].detach().cpu().numpy()
                if states.shape != (h + 1, 5) or ltr.shape != (h + 1,):
                    raise ValueError(f'{kind} prediction shape mismatch on {ep.id}')
                if not (np.isfinite(states).all() and np.isfinite(ltr).all()):
                    raise ValueError(f'{kind} predicted nonfinite chassis response on {ep.id}')
                channel_data = (states[:, 2], np.rad2deg(states[:, 3]), ltr)
                color, ls = styles[kind]
                for col, values in enumerate(channel_data):
                    row_predictions[kind].append(np.asarray(values, dtype=float))
                    axs[row, col].plot(time_s, values, label=kind, color=color,
                                       linestyle=ls, linewidth=1.65 if kind == 'K2' else 1.2,
                                       alpha=.96 if kind == 'K2' else .88,
                                       zorder=10 if kind == 'K2' else 5)
            note = (f'load={float(ep.info["payload_kg"]):g} kg, '
                    rf'$\mu$={float(ep.info["road_mu"]):g}, '
                    rf'$v_0$={float(ep.info["vx_init_kmh"]):g} km/h')
            for ax in axs[row]:
                ax.text(0.025, 0.965, note, transform=ax.transAxes,
                        ha='left', va='top', fontsize=7.8, color='black',
                        bbox={'boxstyle': 'square,pad=0.25',
                              'facecolor': '#DDEEF5', 'edgecolor': 'none'},
                        zorder=20)
            inset_details = []
            for col in range(3):
                limits = CHASSIS_MAIN_LIMITS.get((row, col), {})
                if limits.get('xlim') is not None:
                    axs[row, col].set_xlim(*limits['xlim'])
                if limits.get('ylim') is not None:
                    axs[row, col].set_ylim(*limits['ylim'])
                axs[row, col].grid(alpha=.22, linewidth=.6)
                axs[row, col].tick_params(labelsize=8, labelbottom=True)
                axs[row, col].set_xlabel('Time from first input (s)', fontsize=9)
                inset_details.append(_add_chassis_detail(
                    axs[row, col], time_s, row_ground_truth[col],
                    {k: row_predictions[k][col] for k in models}, dt, row, col))
            records.append({
                'example': row + 1,
                'prediction_mode': 'fixed_lead_true_history',
                'prediction_steps': prediction_steps,
                'inset_details': json.dumps(inset_details),
                'prediction_count': h + 1,
                'first_target_index': int(start + prediction_steps),
                'last_target_index': int(start + h + prediction_steps),
                'first_target_time_s': float(ep.time_s[start + prediction_steps]),
                'last_target_time_s': float(ep.time_s[start + h + prediction_steps]),
                'episode_id': ep.id,
                'episode_path': str(ep.path),
                'start_index': int(start),
                'start_time_s': float(ep.time_s[start]),
                'payload_kg': float(ep.info['payload_kg']),
                'road_mu': float(ep.info['road_mu']),
                'vx_init_kmh': float(ep.info['vx_init_kmh']),
                'split': split,
                'sample_seed': int(cfg.get('training', {}).get('seed', 2026) if seed is None else seed),
            })
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .025),
                   ncol=len(models) + 1, fontsize=8.5, frameon=False, columnspacing=1.35, handlelength=2.5)
        # fig.suptitle('Chassis response prediction: six Koopman models vs TruckSim',
        #              fontsize=13, y=.991)
        # Different rows can have different inputs: this is NOT a controlled
        # comparison between conditions, even though each row is fair across models.
        # fig.text(.5, .085, 'Each row uses one held-out sequence; all models share its history and controls.',
        #          ha='center', va='center', fontsize=8, color='#555555')
        # fig.text(.5, .975, f'{prediction_steps}-step-ahead prediction (true history refreshed at each origin)',
        #          ha='center', va='top', fontsize=11)
        fig.subplots_adjust(left=.13, right=.985, top=.91, bottom=.17, wspace=.28, hspace=.45)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = '' if prediction_steps == 1 else f'_{prediction_steps}step'
        pdf_path = output_dir / f'koopman_chassis_responses{suffix}.pdf'
        fig.savefig(pdf_path, bbox_inches='tight')
    finally:
        plt.close(fig)
    with (output_dir / f'koopman_chassis_examples{suffix}.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f'[CHASSIS] 3 sampled {split} windows; {len(models)} models + TruckSim; {prediction_steps}-step predictions with true history -> {pdf_path}', flush=True)
    return pdf_path

def plot_chassis_comparisons(cfg, project_root, output_dir, *, steps=(3,),
                             split='test', data_root=None, device='auto', seed=None,
                             models=None):
    """Render selected leads using identical sampled episodes and input origins."""
    steps = tuple(steps)
    if not steps or any(not isinstance(step, int) or step < 1 for step in steps):
        raise ValueError('steps must contain positive integers')
    if len(set(steps)) != len(steps):
        raise ValueError('steps must not contain duplicates')
    sampling_steps = max(10, max(steps))
    return [plot_chassis_comparison(
        cfg, project_root, output_dir, split=split, data_root=data_root,
        device=device, seed=seed, models=models, prediction_steps=step,
        sampling_steps=sampling_steps) for step in steps]


def controlled_chassis(cfg,project_root,case_files,device='auto',output_path=None,offset_s=None):
    """Controlled Fig. only with three independently recorded, matching input episodes.

    A vs B: same mu, different payload. B vs C: same payload, different mu.
    Nominal reference controls, initial speed and 40-step horizon must match.
    """
    if len(case_files)!=3:raise ValueError('Exactly three [A,B,C] accepted NPZ cases required')
    ep_paths=[resolve(project_root,p) for p in case_files]
    eps=discover_episodes([str(p.parent) for p in ep_paths],project_root,require_accepted=True)
    lookup={str(e.path.resolve()):e for e in eps}
    cases=[lookup[str(p.resolve())] for p in ep_paths]
    a,b,c=cases
    def value(e,k):return float(e.info[k])
    if not(np.isclose(value(a,'road_mu'),value(b,'road_mu'),atol=1e-5) and
           not np.isclose(value(a,'payload_kg'),value(b,'payload_kg'),atol=1e-4) and
           np.isclose(value(b,'payload_kg'),value(c,'payload_kg'),atol=1e-4) and
           not np.isclose(value(b,'road_mu'),value(c,'road_mu'),atol=1e-5)):
        raise ValueError('Cases must be A/B: payload contrast, B/C: friction contrast')
    if not np.allclose([value(e,'vx_init_kmh') for e in cases],value(a,'vx_init_kmh'),atol=1e-6):
        raise ValueError('Initial speeds differ; cannot claim controlled contrast')
    dc=cfg['data'];dt=float(dc.get('dt_s',.1));h=int(dc.get('horizon_steps',40));history=int(dc.get('history_steps',5))
    offset_s=float(dc.get('warmup_s',1.0) if offset_s is None else offset_s)
    k=max(history-1,int(round(offset_s/dt)))
    if any(len(e.controls)<k+h for e in cases):raise ValueError('Controlled cases are shorter than 4s horizon')
    common=cases[0].controls[k:k+h]
    for ep in cases[1:]:
        if not np.allclose(ep.controls[k:k+h],common,atol=1e-7,rtol=0):
            raise ValueError('Control sequences DO NOT MATCH; collect matched fixed-profile TruckSim cases first')
    _,manifest=load_shared(cfg,project_root)
    selected=('Parametric EDMDc','K0','K2')
    device=choose_device(device)
    predictions={}
    for kind in selected:
        model,ck=load_model(kind,model_checkpoint(cfg,project_root,kind),device)
        if ck['split_sha256']!=manifest['sha256']:raise ValueError('Mismatch in training partitions')
        results=[]
        for ep in cases:
            from highway_env.vehicle.koopman.data.dataset import KoopmanWindowDataset
            ds=KoopmanWindowDataset([ep],history_steps=history,horizon_steps=h,
                                    warmup_s=offset_s,window_stride=1,dt=dt)
            # Exact aligned window k, not merely first window if rounding differs.
            keys=[i for i,(_,start) in enumerate(ds.keys) if start==k]
            if not keys:raise ValueError('Selected controlled start time does not align to a valid window')
            batch={field:value[None] for field,value in ds[keys[0]].items()}
            with torch.inference_mode():out=_run_model(model,kind,batch,device,dt)
            results.append({'states':out['states'].cpu().numpy()[0],
                            'ltr':out['ltr'].cpu().numpy()[0]})
        predictions[kind]=results
    path=resolve(project_root,output_path or Path(cfg['outputs']['evaluation'])/'controlled'/'koopman_chassis_responses.pdf')
    path.parent.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _paper_style(plt)
    fig,axs=plt.subplots(3,3,figsize=(12.8,8.8),sharex=True)
    t=np.arange(h+1)*dt
    labels=('Yaw rate (rad/s)','Roll (deg)','LTR')
    for row,ep in enumerate(cases):
        true=(ep.states[k:k+h+1,2],np.rad2deg(ep.states[k:k+h+1,3]),ep.ltr[k:k+h+1])
        row_predictions = {kind: [] for kind in selected}
        for col in range(3):
            ax=axs[row,col]
            ax.plot(t,true[col],color='black',linewidth=1.9,label='TruckSim',zorder=10)
            for kind in selected:
                value=predictions[kind][row]['ltr'] if col==2 else predictions[kind][row]['states'][:,2 if col==0 else 3]
                if col==1:value=np.rad2deg(value)
                value=np.asarray(value,dtype=float)
                row_predictions[kind].append(value)
                color=PAPER_COLORS.get(kind, '#555555')
                linestyle={'Parametric EDMDc':'--','K0':'--','K2':'-'}[kind]
                ax.plot(t,value,label=kind,color=color,linewidth=1.65 if kind=='K2' else 1.2,linestyle=linestyle,zorder=8 if kind=='K2' else 5)
            ax.grid(alpha=.25)
            if row==0:ax.set_title(labels[col])
            if row==2:ax.set_xlabel('Prediction horizon (s)')
        local_gains=[]
        for col in range(3):
            base_error=np.abs(row_predictions['K0'][col]-true[col])
            k2_error=np.abs(row_predictions['K2'][col]-true[col])
            gain=base_error-k2_error
            width=max(3,min(len(gain),int(round(.8/dt))))
            score=float(np.max(np.convolve(gain,np.ones(width)/width,mode='valid'))) if width<len(gain) else float(np.mean(gain))
            local_gains.append(score)
        best_col=int(np.argmax(local_gains))
        _add_k2_local_detail(axs[row,best_col],t,true[best_col],row_predictions,dt)
        axs[row,0].set_ylabel(f'Case {"ABC"[row]}\npayload={value_float(ep,"payload_kg"):g} kg, mu={value_float(ep,"road_mu"):g}',fontsize=8)
    axs[0,0].legend(fontsize=7,loc='best')
    fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)
    print(f'[CONTROLLED] verified identical 40-step controls and initial speed; saved {path}',flush=True)
    return path


def value_float(ep,key):return float(ep.info[key])
