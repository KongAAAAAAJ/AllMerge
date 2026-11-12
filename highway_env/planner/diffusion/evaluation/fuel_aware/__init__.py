"""TRACE Fuel-Aware payload-stratified open-loop evaluation.

Modules are organized by responsibility rather than historical V1/V2/V3 tags.
The only supported study workflow is :func:`eval_payload_ablation`.
"""
from .runner import eval_payload_ablation
__all__ = ['eval_payload_ablation']
