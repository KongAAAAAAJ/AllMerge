"""EDMDc and Parametric EDMDc identification on frozen train episodes."""
from pathlib import Path
import shutil
import torch
from highway_env.vehicle.koopman.models.edmd import PolynomialEDMDc,fit_edmd
from highway_env.vehicle.koopman.training.base_train import save_checkpoint
from highway_env.vehicle.koopman.data.shared_split import resolve


def train_edmd(kind,cfg,project_root,splits,manifest):
    mcfg=cfg.get('edmd',{})
    model=PolynomialEDMDc(kind,history_steps=int(cfg['data'].get('history_steps',5)),
        normalization=manifest['normalization'],condition_normalization=manifest['condition_normalization'],
        ridge=float(mcfg.get('ridge',1e-3)))
    validation_config={
        'dt_s':float(cfg['data'].get('dt_s',.1)),
        'horizon_steps':int(cfg['data'].get('horizon_steps',40)),
        'warmup_s':float(cfg['data'].get('warmup_s',1.0)),
        'eval_stride':int(cfg['data'].get('eval_stride',5)),
        'batch_size':int(cfg.get('edmd',{}).get('validation_batch_size',256)),
    }
    stats=fit_edmd(model,splits['train'],
        validation_episodes=splits.get('val',[]),
        validation_config=validation_config)
    out=resolve(project_root,Path(cfg['outputs']['checkpoints'])/kind.lower().replace(' ','_'))
    out.mkdir(parents=True,exist_ok=True)

    # PARAMETRIC_EDMDC_CHECKPOINT_BACKUP_V22
    # Preserve every prior Parametric EDMDc variant before replacing best.pt.
    if kind=='Parametric EDMDc':
        for source_name in ('best.pt','last.pt'):
            source=out/source_name
            if not source.is_file():
                continue
            try:
                previous=torch.load(source,map_location='cpu',weights_only=True)
            except TypeError:
                previous=torch.load(source,map_location='cpu')
            previous_stats=previous.get('stats',{}) if isinstance(previous,dict) else {}
            policy=previous_stats.get('stabilization_policy','legacy_uniform_shrink')
            if policy=='validation_selected_shift_preserving_learned_rows':
                continue
            if policy=='shift_preserving_learned_rows':
                tag='shift_preserving_v2_1'
            elif policy=='preserve_identified_matrices' or (
                    float(previous_stats.get('stabilization_factor',1.0))>=0.999999
                    and float(previous_stats.get('pre_stabilization_radius',
                        previous_stats.get('ridge_pre_shift_radius',0.0)))>1.02+1e-6):
                tag='no_shrink'
            elif float(previous_stats.get('stabilization_factor',1.0))<0.999999:
                tag='uniform_shrink'
            else:
                tag='previous'
            backup=out/f'{source.stem}_{tag}.pt'
            if backup.exists():
                continue
            shutil.copy2(source,backup)
            print(f'[BACKUP] preserved previous Parametric EDMDc checkpoint -> {backup}',flush=True)
    state={'model_type':kind,'architecture':'Polynomial_EDMDc_StageC',
           'model_kwargs':{'variant':kind,'history_steps':model.history_steps,
                           'ridge':model.ridge},
           'normalization':manifest['normalization'],
           'condition_normalization':manifest['condition_normalization'],
           'model_state':model.state_dict(),'epoch':0,'stats':stats,
           'split_sha256':manifest['sha256'],'split_manifest':manifest['report'],'train_config':cfg}
    save_checkpoint(out/'best.pt',state)
    save_checkpoint(out/'last.pt',state)
    print(f'[EDMD DONE] {kind} {stats} -> {out/"best.pt"}',flush=True)
    return out/'best.pt'


def load_edmd(path,device='cpu'):
    try:d=torch.load(path,map_location=device,weights_only=True)
    except TypeError:d=torch.load(path,map_location=device)
    model=PolynomialEDMDc(**d['model_kwargs'],normalization=d['normalization'],
                           condition_normalization=d['condition_normalization'])
    model.load_state_dict(d['model_state'])
    return model.to(device).eval(),d
