# Single Expert Collection Entry

Baseline: AllMerge `c638b18`.

## Final structure

```text
collect_expert_dataset.py      # the only executable expert collection entry
expert_collection_core.py     # feature/label/rollout helpers; no __main__ execution
expert_dataset.py             # Dataset/DataLoader + shard split
split_expert_dataset.py       # split utility
validate_expert_dataset.py    # validation utility
```

The legacy pilot collector was removed after its current implementation was moved into `expert_collection_core.py`. No feature schema, Polynomial expert, mode assignment, or target-label computation was redesigned.
