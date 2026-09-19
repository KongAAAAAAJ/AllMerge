# GRPO_SOURCE_AUDIT

Remote source: `https://github.com/KongAAAAAAJ/Diffusion-metadrive.git`
Remote ref resolved at install time: `15e7d4b202768a4ac4d38a9ffb64d3e9009e75e8`
AllMerge target baseline: `63a7365`
AllMerge target HEAD: `63a7365c6123dd759481eead81ca173db8918d0a`

Installer downloaded the remote source snapshot before installing the thin AllMerge GRPO layer.

## Required source areas

| Source area | Status |
|---|---|
| `train/train_selector.py` | **found** |
| `configs/train/platoon_selector_refine.yaml` | **not found** |
| `docs/phases6/grpo_refinement_plan.md` | **not found** |
| `models/diffusion/...` | 4 files |
| `models/platoon/...` | 11 files |
| `tests/acceptance/...` | 160 files |

## Highest-value migration candidates

### diffusion scheduler / log_prob

| Score | File | Lines | SHA256/12 |
|---:|---|---:|---|
| 32 | `old_docs/phases/phase5.md` | 350 | `597ba39828ba` |
| 30 | `old_docs/phases6/grpo_refinement_plan.md` | 406 | `61638760a7e1` |
| 29 | `tests/acceptance/test_phase5_task2.py` | 229 | `8ed0bdfa64c1` |
| 19 | `EXECUTE_LOG.md` | 615 | `6a088a8afe14` |
| 18 | `old_docs/phases5/rl_training_fix_plan.md` | 547 | `f35044937c12` |
| 15 | `metadrive/policy/diffusion_policy/modules/scheduler.py` | 60 | `16c60ca55cd7` |
| 15 | `old_docs/phases2/task2_intra_anchor.md` | 270 | `7218292afc42` |
| 14 | `models/diffusion/__init__.py` | 3 | `fe7b7b7f815d` |

### sampling trace / replay

| Score | File | Lines | SHA256/12 |
|---:|---|---:|---|
| 24 | `old_docs/phases5/rl_training_fix_plan.md` | 547 | `f35044937c12` |
| 22 | `tests/acceptance/test_phase5_task2.py` | 229 | `8ed0bdfa64c1` |
| 20 | `metadrive/manager/scenario_traffic_manager.py` | 409 | `66bdb19998ed` |
| 19 | `metadrive/policy/diffusion_policy/transfuser_model_v2.py` | 706 | `d091ab06e76a` |
| 18 | `metadrive/tests/test_functionality/test_obs_noise.py` | 59 | `04b0f4e56dc3` |
| 18 | `metadrive_simulator.egg-info/SOURCES.txt` | 1378 | `d02f1b337979` |
| 17 | `metadrive/render_pipeline/rpplugins/clouds/plugin.py` | 75 | `47c9456a6a0f` |
| 16 | `metadrive/manager/replay_manager.py` | 200 | `f223eac5f403` |

### GRPO objective

| Score | File | Lines | SHA256/12 |
|---:|---|---:|---|
| 40 | `old_docs/phases5/rl_training_fix_plan.md` | 547 | `f35044937c12` |
| 33 | `old_docs/phases6/grpo_refinement_plan.md` | 406 | `61638760a7e1` |
| 32 | `old_docs/phases/phase5.md` | 350 | `597ba39828ba` |
| 26 | `EXECUTE_LOG.md` | 615 | `6a088a8afe14` |
| 25 | `tests/acceptance/test_phase5v2_integration.py` | 245 | `883f32766b3d` |
| 24 | `old_docs/phases2/task6_layered_advantage.md` | 279 | `800de0f6019d` |
| 24 | `old_docs/phases2/task8_integration_test.md` | 169 | `76ca347942c5` |
| 23 | `AGENTS.md` | 364 | `16da475160fd` |

### trajectory refinement

| Score | File | Lines | SHA256/12 |
|---:|---|---:|---|
| 61 | `old_docs/phases6/grpo_refinement_plan.md` | 406 | `61638760a7e1` |
| 34 | `AGENTS.md` | 364 | `16da475160fd` |
| 33 | `METHODS.md` | 472 | `9bb04dd495dd` |
| 32 | `README.md` | 435 | `59a84ec9d19b` |
| 28 | `metadrive/policy/diffusion_policy/transfuser_model_v2.py` | 706 | `d091ab06e76a` |
| 22 | `docs/plan.md` | 326 | `6097cfd6c15c` |
| 16 | `metadrive/component/navigation_module/trajectory_navigation.py` | 236 | `189384b2a351` |
| 16 | `metadrive/exp_dataset/hierarchical_expert/docs/trajectory-tracking-redesign.md` | 301 | `d44b5d24fbe7` |

### closed-loop branch rollout

| Score | File | Lines | SHA256/12 |
|---:|---|---:|---|
| 23 | `old_docs/phases3/task2_remove_reset.md` | 255 | `8330fa780255` |
| 19 | `old_docs/phases2/task6_layered_advantage.md` | 279 | `800de0f6019d` |
| 19 | `old_docs/phases6/grpo_refinement_plan.md` | 406 | `61638760a7e1` |
| 18 | `old_docs/phases5/rl_training_fix_plan.md` | 547 | `f35044937c12` |
| 18 | `tests/acceptance/test_phase5v2_task6.py` | 208 | `92aa27a12d97` |
| 17 | `AGENTS.md` | 364 | `16da475160fd` |
| 16 | `old_docs/phases2/task3_env_state.md` | 173 | `ef9b50aa4222` |
| 16 | `old_docs/phases3/task1_traffic_state.md` | 255 | `62442e684f5d` |

## Migration rule

- COPY mathematical GRPO/DDIM logic where signatures are compatible.
- ADAPT model/environment boundaries to StructuredDiffusionPlanner, AllMerge scenarios and W4 reward.
- DROP MetaDrive/NavSim-specific wrappers.
- Do not modify the frozen AllMerge diffusion planner trunk.
