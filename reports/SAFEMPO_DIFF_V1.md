# SafeMPO-Diff V1

## Scope

SafeMPO-Diff V1 is the first trajectory-distribution adaptation of SafeMPO for the current AllMerge diffusion planner. It is intentionally isolated from the frozen Normalized Lagrangian V3 and Projection V2 Restorative baselines.

The only policy-update mechanism changed relative to the common diffusion/GRPO infrastructure is:

`G48 sampling -> task reward + multi-constraint evaluation -> SafeMPO q* -> KL distribution distillation`.

Sampling, W4 geometry, frozen-reference validation, trainable parameter scope, optimizer family, reference-KL regularization, max-grad-norm clipping, and checkpoint loading remain shared with the baseline code.

## E-step: G48 finite-particle SafeMPO

For every valid vehicle/mode state, G trajectories are sampled from the frozen per-step behavior policy. The empirical old-policy particle measure is therefore uniform over the sampled particles, p_g=1/G.

The task score is Q_g. For every enabled constraint j, the existing AllMerge normalized nonnegative violation v_gj is converted to SafeMPO safety log-likelihood

G_gj = -v_gj / beta.

A single set of batch-level dual variables lambda_j and nu is solved while q*(g|state) is normalized independently over the G particles:

q*_g proportional to exp((Q_g + sum_j lambda_j G_gj) / nu).

The E-step KL budget is the mean state-wise KL(q* || Uniform(G)). The V1 defaults use the SafeMPO paper values epsilon=0.1 and kappa=10.0. The dual is solved with SLSQP using analytic gradients and warm-started from the previous training step.

Finite-sample safeguard: a constraint channel that is constant over G for every valid state cannot be improved by reweighting the current support. Such a channel is removed from that step's log-barrier dual and is still logged. This prevents a trivially safe constant channel (e.g. collision=0 for every candidate) from making the finite-particle primal artificially infeasible.

## M-step: diffusion particle-distribution distillation

For the exact sampled reverse diffusion trace,

z_g(theta) = sum_t [log p_theta(x_{t-1}^g | x_t^g,s) - log p_old(x_{t-1}^g | x_t^g,s)].

The current student particle distribution is

p_theta,g = softmax_g(z_g).

At the start of every update theta=theta_old, so z_g=0 and p_theta,g=1/G exactly.

The policy loss is

L_distill = KL(stopgrad(q*) || p_theta).

For the particle logits, dL/dz_g = p_theta,g - q*_g. Thus q* directly specifies where sampled trajectory probability mass should increase or decrease. PPO/GRPO signed advantages and PPO clipping are not used in V1.

The existing frozen-reference KL penalty and max-grad-norm clipping remain enabled for cross-method stability diagnostics.

## V1 comparison protocol

Use the same checkpoint and common settings as the frozen 500-step baselines:

- scenario: curved
- group_action: 3
- G=48
- steps=500
- seed=7
- lr=5e-7
- eta=0.02
- reference KL coefficient=0.05
- max_grad_norm=5
- update_epochs=2
- task_reward_type=progress_comfort
- constraints: collision, road, ttc, background_gap, teammate_gap
- fixed validation: 16 cached states, every 10 steps

SafeMPO-Diff-specific V1 defaults:

- E-step KL epsilon=0.10
- kappa=10.0
- constraint beta=1.0 (AllMerge violations are already normalized)
- lambda init=1.0
- lambda max=10000
- nu init=1.0
- SLSQP maxiter=128 with warm start

The first 500-step run should be treated as a stability/convergence evaluation rather than a hyperparameter sweep.
