"""Resolve shared A2 selection settings independently of scorer training."""
from learning.config import load_yaml_config
from highway_env.planner.diffusion.config.experiment import resolve_repo_file


def resolve_a2_config(config):
    """Load optional A2 policy settings, then apply explicit caller overrides."""
    settings = dict(config or {})
    path = settings.pop('a2_config', None)
    if path is None:
        return settings
    policy = load_yaml_config(resolve_repo_file(path)).get('a2')
    if not isinstance(policy, dict):
        raise ValueError('A2 anchor config must contain an a2 mapping')
    return {**policy, **settings}


def load_scorer_config(path):
    """Load optimizer/data/teacher settings plus the shared A2 policy."""
    return resolve_a2_config(load_yaml_config(resolve_repo_file(path)))
