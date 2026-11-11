"""Paper-specific Koopman metrics and plots.

Only: ADE, FDE, yaw-rate RMSE (rad/s), roll-angle RMSE (deg), LTR RMSE.
The statistical unit is the held-out EPISODE: within-episode windows are
aggregated first, then episodes receive equal weight in reported statistics.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np

PAPER_COLUMNS = ('ADE (m)', 'FDE (m)', 'Yaw Rate RMSE (rad/s)',
                 'Roll RMSE (deg)', 'LTR RMSE')
MODELS = ('EDMDc', 'Parametric EDMDc', 'K0', 'K1-P', 'K1-F', 'K2')


def get_errors(pred_states, true_states, pred_poses, true_poses,
               pred_ltr=None, true_ltr=None, ltr_valid=None):
    """Return per-window arrays on the 40 *future* states, omitting t=0.

    All inputs must already be in physical units, roll in radians.
    LTR values are predicted by a separately supervised head.
    """
    pred_states = np.asarray(pred_states)
    true_states = np.asarray(true_states)
    pred_poses = np.asarray(pred_poses)
    true_poses = np.asarray(true_poses)
    if (pred_states.shape != true_states.shape or pred_states.ndim != 3 or
        pred_states.shape[-1] != 5 or pred_poses.shape != true_poses.shape or
        pred_poses.shape != (pred_states.shape[0], pred_states.shape[1], 3)):
        raise ValueError('Prediction/target shapes do not match [B,T,5] and [B,T,3]')
    if not all(np.isfinite(x).all() for x in (pred_states, true_states, pred_poses, true_poses)):
        raise ValueError('Nonfinite prediction or ground truth; cannot report scientific metrics')
    xy = np.linalg.norm(pred_poses[..., :2] - true_poses[..., :2], axis=-1)
    yaw_sq = (pred_states[:, 1:, 2] - true_states[:, 1:, 2])**2
    roll_sq = (np.rad2deg(pred_states[:, 1:, 3] - true_states[:, 1:, 3]))**2
    result = {'position_error':xy, 'yaw_sq':yaw_sq, 'roll_sq':roll_sq,
              'ltr_sq':None, 'ltr_valid':None}
    if true_ltr is None:
        raise ValueError('No LTR labels')
    pred_ltr, true_ltr = np.asarray(pred_ltr), np.asarray(true_ltr)
    if pred_ltr.shape != xy.shape or true_ltr.shape != xy.shape:
        raise ValueError('LTR shape mismatch')
    mask = np.asarray(ltr_valid, dtype=bool) if ltr_valid is not None else np.isfinite(true_ltr)
    if mask.shape != xy.shape:
        raise ValueError('LTR validity mask shape mismatch')
    mask = mask[:, 1:] & np.isfinite(pred_ltr[:, 1:]) & np.isfinite(true_ltr[:, 1:])
    result['ltr_sq'] = (pred_ltr[:, 1:] - true_ltr[:, 1:])**2
    result['ltr_valid'] = mask
    return result


def summarize_episode_errors(errors, episode_id):
    """Aggregate all valid windows of a single episode."""
    pe = errors['position_error']
    out = {'episode_id': str(episode_id), 'windows':int(pe.shape[0]),
           'ADE (m)':float(pe[:, 1:].mean()), 'FDE (m)':float(pe[:, -1].mean()),
           'Yaw Rate RMSE (rad/s)':float(np.sqrt(errors['yaw_sq'].mean())),
           'Roll RMSE (deg)':float(np.sqrt(errors['roll_sq'].mean())),
           'LTR RMSE':None, 'curve':pe.mean(axis=0).tolist()}
    if errors['ltr_sq'] is not None and np.any(errors['ltr_valid']):
        out['LTR RMSE'] = float(np.sqrt(errors['ltr_sq'][errors['ltr_valid']].mean()))
    return out


def summarize_all(episodes):
    if not episodes:
        raise ValueError('No tested episodes')
    table = {}
    for key in PAPER_COLUMNS:
        values = [e[key] for e in episodes if e[key] is not None]
        table[key] = float(np.mean(values)) if len(values) == len(episodes) else None
    curves = np.asarray([e['curve'] for e in episodes], dtype=np.float64)
    return table, curves.mean(axis=0)


def save_table_row(path, model, table):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['Model',*PAPER_COLUMNS]);w.writerow([model,*['' if table[k] is None else f'{table[k]:.9g}' for k in PAPER_COLUMNS]])


def save_episodes(path, episodes):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['episode_id','windows',*PAPER_COLUMNS]);
        for ep in episodes:
            w.writerow([ep['episode_id'],ep['windows'],*['' if ep[k] is None else f'{ep[k]:.9g}' for k in PAPER_COLUMNS]])


def save_position_curve(path, curve, dt):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['time_s','K0_mean_position_error_m']);
        for i,v in enumerate(curve): w.writerow([f'{i*dt:.6f}',f'{v:.9g}'])


def plot_errors(path, curve, episodes, dt, model_name='K0'):
    """K0-only diagnostic figure now; six-model paper comparison comes later."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2,figsize=(10.2,3.6))
    axs[0].plot(np.arange(len(curve))*dt,curve,linewidth=2,label=model_name)
    axs[0].set(xlabel='Prediction horizon (s)',ylabel='Mean position error (m)',title='(a) 4-s free rollout')
    axs[0].grid(alpha=.25);axs[0].legend(loc='upper left')
    values = [e['ADE (m)'] for e in episodes]
    axs[1].boxplot([values], tick_labels=[model_name], showfliers=True, whis=1.5)
    axs[1].set(ylabel='Episode-level ADE (m)',title='(b) Held-out episode distribution')
    axs[1].grid(alpha=.25,axis='y')
    fig.suptitle('K0 diagnostic only — six-model comparison not yet available',fontsize=10)
    fig.tight_layout()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,bbox_inches='tight');plt.close(fig)


def plot_chassis_examples(path, examples, dt):
    """K0 vs GT for up to three real episodes; NOT a matched 3-case paper figure.

    Each case is predicted using its *own* actual input sequence. Do not infer
    controlled payload/friction effects from these curves unless controls match.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not examples: return
    fig,axs=plt.subplots(len(examples),3,figsize=(12,3.05*len(examples)),squeeze=False,sharex=True)
    headers=('Yaw rate (rad/s)','Roll angle (deg)','LTR')
    for j,title in enumerate(headers): axs[0,j].set_title(title)
    for i,ex in enumerate(examples):
        t=np.arange(len(ex['pred_states']))*dt
        pred=ex['pred_states'];true=ex['true_states']
        channels=((true[:,2],pred[:,2]),(np.rad2deg(true[:,3]),np.rad2deg(pred[:,3])),
                  (ex.get('true_ltr'),ex.get('pred_ltr')))
        for j,(gt,pr) in enumerate(channels):
            ax=axs[i,j]
            if gt is not None:ax.plot(t,gt,label='TruckSim',color='black',lw=1.3)
            if pr is not None:ax.plot(t,pr,label='K0',lw=1.3,ls='--')
            if gt is None or pr is None:
                ax.text(.5,.5,'LTR prediction unavailable',transform=ax.transAxes,
                        va='center',ha='center',fontsize=8)
            ax.grid(alpha=.25)
            if i==len(examples)-1:ax.set_xlabel('Prediction horizon (s)')
            if i==0 and j==0:ax.legend(fontsize=8)
        axs[i,0].set_ylabel(ex['label'],fontsize=8)
    fig.suptitle('K0 exploratory examples: per-episode inputs may differ; NOT controlled three-case comparison',fontsize=9)
    fig.tight_layout()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,bbox_inches='tight');plt.close(fig)
