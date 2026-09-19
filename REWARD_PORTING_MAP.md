# Reward Porting Map — trajectory_mode

Baseline:
- AllMerge `_reward`: `2419c34`
- Diffusion-metadrive `BEV`: latest single-target modular reward

## Naming decision

`vehicle_mode` is renamed to `trajectory_mode`.

The comparison semantics are unchanged:
one trajectory mode of one target controlled vehicle is replaced at a time,
while the other two controlled vehicles stay fixed to their Stage-1 argmax
raw `tau_d`.

## Source -> AllMerge

| Source responsibility | AllMerge target | Action |
|---|---|---|
| reward config / weights | `trajectory_mode_reward/config.py` | COPY + RENAME |
| reward/application contract | `trajectory_mode_reward/contracts.py` | COPY + RENAME + shape ADAPT |
| soft gap/TTC risk | `trajectory_mode_reward/risk.py` | COPY + RENAME |
| OBB overlap / corridor gap | `trajectory_mode_reward/collision_geometry.py` | COPY |
| trajectory interpolation / transforms | `trajectory_mode_reward/geometry.py` | ADAPT XY->heading |
| semantic-BEV road field | `trajectory_mode_reward/geometry.py` | REPLACE with RoadNetwork lane margin |
| MetaDrive actor prediction | `trajectory_mode_reward/state_adapter.py` | REPLACE with AllMerge lane prediction |
| typed results | `trajectory_mode_reward/results.py` | COPY + RENAME |
| input validation | `trajectory_mode_reward/input_validation.py` | COPY + XY shape ADAPT |
| target-only counterfactual scorer | `trajectory_mode_reward/scoring.py` | COPY + environment ADAPT |
| public scorer | `trajectory_mode_reward/counterfactual.py` | COPY + RENAME |
| shared GRPO/standalone entry | `trajectory_mode_reward/evaluator.py` | ADD glue only |

## Explicit drops

- legacy joint reward
- formation reward
- teammate self progress/comfort/offroad reward
- semantic BEV dependency
- MetaDrive-specific `PlatoonNormalPlanner`
- environment legacy `_reward()` integration

## AllMerge shape contract

- candidates: `[3,10,N,8,2]`
- frozen all modes: `[3,10,8,2]`
- frozen argmax teammate context: `[3,8,2]`
- valid mask: `[3,10]`

Heading needed by comfort/OBB geometry is recovered from XY trajectory tangents.
