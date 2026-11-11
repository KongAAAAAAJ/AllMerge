"""Model-specific trainer: K1-P, K1-F and K2 neural Koopman variants."""
from highway_env.vehicle.koopman.training.neural_train import train_neural


def train_conditioned(variant,cfg,project_root,splits,manifest,epochs=None,device=None):
    if variant not in ('K1-P','K1-F','K2'):raise ValueError(variant)
    return train_neural(variant,cfg,project_root,splits,manifest,epochs,device)
