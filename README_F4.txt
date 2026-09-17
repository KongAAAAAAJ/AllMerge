AllMerge F.4 Episode Termination + Mask Reason Diagnostic
==========================================================

Purpose
-------
Answer two questions without changing planner/training behavior:

1. Why do episodes end after about 4-5 steps?
2. Why are about 30% of F.2 expert target modes marked invalid by mode_valid_mask?


Design
------
This is a diagnostic-only package.

It DOES NOT modify:
    Polynomial
    Dynamic Anchor
    StructuredDiffusionPlanner
    F.2 target assignment
    controller
    environment termination rules
    mode_valid_mask rules

It adds one standalone runtime diagnostic script.


Episode termination diagnostic
------------------------------
At runtime the script wraps the ACTUAL environment instance methods:

    _is_terminated()
    _is_truncated()

without changing their return values.

Before VecEnv auto-reset it records:
    env time
    config duration
    duration reached?
    each controlled vehicle crashed?
    each controlled vehicle on_road?
    has_arrived() when available
    TimeLimit.truncated from SB3 info
    actual _is_terminated source
    actual _is_truncated source

This distinguishes:
    time limit / truncation
    controlled vehicle collision
    offroad termination
    all vehicles arrived
    duration embedded inside custom termination
    other custom termination


Mask diagnostic
---------------
For runtime F.2 target modes it records:
    target semantic
    target mode
    mode_valid_mask
    same-semantic fallback modes
    number of observed agents

For invalid targets it also records three AUXILIARY distance probes:
    static-agent center distance
    CV using agent columns 2,3
    CV using agent columns 4,5

These are explicitly only proxies.

The exact current mode_valid_mask implementation is obtained by scanning the
actual local repository and writing source excerpts for:
    mode_valid_mask
    collision / collid
    OBB
    overlap

Therefore we do not guess which collision helper or feature columns the
current commit uses.


Existing 300-frame pilot
------------------------
The script also loads:

    outputs/expert_pilot/expert_pilot.npz

and reports:
    900-sample target invalid ratio
    same-semantic fallback counts
    episode-step histogram
    target mode / semantic distribution


Install
-------
Extract this package into:

    D:\KONG_Files\AllMerge\all_merge

Then:

    python .\install_f4_diagnostic.py


Run
---
Recommended first run:

    python .\f4_episode_mask_diagnostic.py --episodes 20

20 episodes should be sufficient if each episode really terminates after
roughly 4-5 steps.


Outputs
-------
    outputs/f4_diagnostic/
        f4_report.json
        f4_source_context.txt
        f4_invalid_targets.csv


Send back
---------
Please send:
    f4_report.json
    f4_source_context.txt

The console summary is also useful.

Those two files are enough to answer the two F.4 questions from the actual
running code rather than from assumptions.
