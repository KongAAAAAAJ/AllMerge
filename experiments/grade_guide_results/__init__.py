"""AllMerge self-contained Reward-Gradient mean probe (V5-compatible)."""

__all__ = ['plot_results']

def plot_results(*args, **kwargs):
    from .data_plot import plot_results as fn
    return fn(*args, **kwargs)
