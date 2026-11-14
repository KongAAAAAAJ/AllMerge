"""Paper payload table and 3x2 mean +/- SD velocity and XY plots."""
from __future__ import annotations
import numpy as np

TABLE_METRICS = ('fuel_l_per_100km','fuel_l_per_4s','progress_m','ade_m','fde_m')

def _fmt_payload(p):
    return f'{p:g} kg'


def latex_payload_table(rows, payloads, ours='F3'):
    lookup = {(float(r['payload_kg']),r['method']):r for r in rows}
    lines = [
        r'\begin{table*}[!t]', r'\centering',
        r'\caption{Effects of Fuel-Aware Training under Different Payload Conditions}',
        r'\label{tab:fuel_ablation}', r'\scriptsize',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\setlength{\tabcolsep}{6pt}',
        r'\begin{tabular}{@{}llccccc@{}}', r'\hline',
        r'\textbf{Payload} & \textbf{Method} & '
        r'\shortstack{\textbf{Fuel}\\\textbf{(L/100 km)} $\downarrow$} & '
        r'\shortstack{\textbf{Total Fuel}\\\textbf{(L)} $\downarrow$} & '
        r'\shortstack{\textbf{Progress}\\\textbf{(m)} $\uparrow$} & '
        r'\shortstack{\textbf{ADE}\\\textbf{(m)} $\downarrow$} & '
        r'\shortstack{\textbf{FDE}\\\textbf{(m)} $\downarrow$} \\',
        r'\hline',
    ]
    for k,p in enumerate(payloads):
        for method in ('F0',ours):
            if (p,method) not in lookup:
                raise ValueError(f'Cannot create paper table: missing {method} at {p} kg')
            r = lookup[(p,method)]
            name = 'F0: Imitation' if method=='F0' else 'F3: Fuel-aware'
            if method != 'F0':
                name = r'\textbf{' + name + '}'
            cells = [_fmt_payload(p) if method=='F0' else '', name]
            for key in TABLE_METRICS:
                value = f'{r[key]:.2f}' if key != 'fuel_l_per_4s' else f'{r[key]:.4f}'
                cells.append(r'\textbf{' + value + '}' if method != 'F0' else value)
            lines.append(' & '.join(cells) + r' \\')
        lines.append(r'\hline')
    lines.extend([r'\end{tabular}',r'\end{table*}',''])
    return '\n'.join(lines)


def _normal_sd_band(xy):
    """Normal projection of repeat variation around the time-aligned mean path."""
    paths = np.asarray(xy, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[1:] != (41,2) or len(paths) < 1:
        raise ValueError('XY paths must be [repeat,41,2]')
    mean = paths.mean(axis=0)
    tangent = np.gradient(mean, axis=0)
    size = np.linalg.norm(tangent, axis=1, keepdims=True)
    unit = tangent / np.maximum(size, 1e-8)
    bad = size[:,0] < 1e-8
    unit[bad] = [1.,0.]
    normal = np.stack((-unit[:,1],unit[:,0]),axis=1)
    signed = np.einsum('ntc,tc->nt', paths-mean[None,:,:], normal)
    sd = signed.std(axis=0, ddof=1) if len(paths)>1 else np.zeros(len(mean))
    return mean, mean+sd[:,None]*normal, mean-sd[:,None]*normal


def draw_payload_figure(path, curves, payloads, selected, dt, *, fig_details):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,2,figsize=(10.8,10.0),constrained_layout=True)
    time = np.arange(41,dtype=np.float64)*dt
    palette = {'F0':'#2563eb', selected:'#ea580c'}
    for row,p in enumerate(payloads):
        ax_v,ax_xy = axes[row]
        if p not in curves or set(curves[p]) != set(palette):
            raise ValueError(f'Missing F0/selected method figure traces for payload {p}')
        truth_sets = []
        for method,color in palette.items():
            repeats = curves[p][method]
            vel = np.asarray([r['speeds_kmh'] for r in repeats],dtype=np.float64)
            xy = np.asarray([r['xy'] for r in repeats],dtype=np.float64)
            if vel.ndim!=2 or vel.shape[1]!=41:
                raise ValueError('Expected [n,41] velocity traces')
            avg = vel.mean(axis=0)
            std = vel.std(axis=0, ddof=1) if len(vel)>1 else np.zeros_like(avg)
            label = 'F0: Imitation' if method=='F0' else 'F3: Fuel-aware'
            ax_v.plot(time,avg,color=color,lw=1.8,label=label)
            ax_v.fill_between(time,avg-std,avg+std,color=color,alpha=.16,linewidth=0)
            mid,high,low=_normal_sd_band(xy)
            ax_xy.plot(mid[:,0],mid[:,1],color=color,lw=1.9,label=label)
            ax_xy.fill(np.r_[high[:,0], low[::-1,0]],
                       np.r_[high[:,1], low[::-1,1]],color=color,alpha=.16,linewidth=0)
            truth_sets.extend(r['gt_xy'] for r in repeats)
        first=np.asarray(truth_sets[0])
        if any(not np.allclose(first,r,atol=1e-5) for r in truth_sets[1:]):
            raise ValueError('Ground-truth path not identical across methods/seeds for one scenario')
        ax_xy.plot(first[:,0],first[:,1],ls='--',lw=1.4,color='#777777',label='Expert')
        ax_v.set_title(f'({chr(ord("a")+2*row)}) {_fmt_payload(p)}: speed',fontsize=10)
        ax_xy.set_title(f'({chr(ord("b")+2*row)}) {_fmt_payload(p)}: X-Y path',fontsize=10)
        ax_v.set(xlabel='Time (s)',ylabel='Longitudinal speed (km/h)')
        ax_xy.set(xlabel='X (m)',ylabel='Y (m)')
        ax_v.set_xlim(0,40*dt)
        # Longitudinal X spans tens of metres while lane changes span only
        # a few metres in Y; equal axes would visually flatten the maneuver.
        ax_xy.set_aspect('auto')
        ymin,ymax=ax_xy.get_ylim()
        if ymax-ymin < 2.:
            yc=.5*(ymin+ymax)
            ax_xy.set_ylim(yc-1.,yc+1.)
        ax_v.grid(alpha=.17)
        ax_xy.grid(alpha=.17)
        if row == 0:
            ax_v.legend(frameon=False,fontsize=8,loc='best')
            ax_xy.legend(frameon=False,fontsize=8,loc='best')
        # Different rows are independently sampled, so no claim of matched scenarios.
    fig.savefig(path,bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)
