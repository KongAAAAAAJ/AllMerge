"""EDMDc and Parametric EDMDc identification on frozen train episodes."""
from pathlib import Path
import torch
from .edmd import PolynomialEDMDc,fit_edmd
from .k0_train import save_checkpoint
from .shared_split import resolve


def train_edmd(kind,cfg,project_root,splits,manifest):
    mcfg=cfg.get('edmd',{})
    model=PolynomialEDMDc(kind,history_steps=int(cfg['data'].get('history_steps',5)),
        normalization=manifest['normalization'],condition_normalization=manifest['condition_normalization'],
        ridge=float(mcfg.get('ridge',1e-3)))
    stats=fit_edmd(model,splits['train'])
    out=resolve(project_root,Path(cfg['outputs']['checkpoints'])/kind.lower().replace(' ','_'))
    out.mkdir(parents=True,exist_ok=True)
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
