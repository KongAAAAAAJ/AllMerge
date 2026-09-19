"""Support utilities for AllMerge diffusion pretraining.

Keep this package initializer dependency-light so schema/checkpoint utilities and
contract smoke tests do not require importing PyTorch Lightning or the planner.
Import ``DiffusionPretrainModule`` from ``pretraining.lightning_module`` only in
training code.
"""
