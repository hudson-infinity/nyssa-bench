# Counterfactual Recovery

NyssaBench can evaluate whether a recovery intervention changes an outcome,
rather than counting every episode that succeeds after an intervention as a
recovery success. At an applied recovery decision, the runner snapshots the
live evaluation state and executes matched branches from that point:

1. continue with the rejected policy action;
2. execute the proposed recovery plan;
3. optionally execute the expert as an oracle upper bound.

The live episode is restored to the branch point after every branch and again
before normal evaluation resumes. Branch execution never replaces the action
or outcome recorded for the live episode.

## Running A Study

Counterfactual evaluation is opt-in because each branch adds simulator work.
It requires recovery to be enabled.

```bash
uv run nyssa run \
  --suite mujoco_control_v0 \
  --tasks mujoco_pusher \
  --engine mujoco \
  --policy random \
  --episodes 20 \
  --seed 0 \
  --expert-provider mujoco-heuristic \
  --enable-verifier \
  --enable-recovery \
  --counterfactual-repeats 5 \
  --counterfactual-horizon 10 \
  --counterfactual-max-branch-points 1 \
  --out benchmark_results/mujoco_counterfactual_smoke \
  --no-replay
```

Add `--counterfactual-oracle` to include the expert branch. Oracle evaluation
requires the expert provider to declare a restorable state contract.

The same flags are accepted by `run-scenario`, `experiment`, and `ablate`.
For ablations, branch evaluation runs only in variants that enable recovery.

## State Contract

A branch snapshot covers:

- simulator integration state and wrapper episode counters;
- environment random-number-generator state;
- policy state and policy RNG state;
- expert state when the oracle branch is requested;
- stressor runtime state, buffers, application status, and RNG state;
- process-level Python, NumPy, PyTorch CPU, and available CUDA RNG state.

Every component reports support, fidelity, RNG coverage, and a reason when
restoration is incomplete. NyssaBench uses three restoration grades:

- `exact`: every required component declares exact restoration;
- `qualified`: branches can execute, but at least one component declares
  incomplete fidelity or RNG coverage;
- `unsupported`: a required component cannot be restored, so branches are not
  executed.

Only exact branch points with matched randomness and completed outcomes are
eligible for the strongest counterfactual claim. Qualified records remain
useful for diagnostics but are never silently promoted.

Run-level claim tiers also account for coverage. Exact branch restoration with
only a sampled subset of eligible interventions is reported as
`exact_counterfactual_partial_coverage`, not as a complete exact study.

ManiSkill uses paired `get_state_dict` and `set_state_dict` APIs when available,
plus controller, wrapper, and discovered episode RNG state. MuJoCo uses the
integration-state API when available and falls back to a qualified manual
physics-state restore. Third-party policies and engines must implement
`get_state`, `set_state`, and `state_restore_capability` to participate.

## Evidence And Metrics

Each run writes `counterfactual_recovery.json`. The artifact contains the
versioned branch-point and branch-outcome records, restore capabilities,
matched RNG fingerprint, per-step branch evidence, trajectory fingerprint,
errors, and aggregate coverage.

The primary estimate is:

```text
P(success | recovery, branch state) - P(success | continue, branch state)
```

Repeated trials are paired by branch seed. The run-level estimate first
averages repeats within a branch point, then computes uncertainty by a fixed,
deterministic cluster bootstrap over branch points. This prevents repeated
rollouts from being treated as independent physical situations.

Reports also expose:

- eligible, observed, supported, matched, exact, and incomplete branch counts;
- continuation and recovery success rates;
- branch coverage;
- helpful interventions, where recovery succeeds and continuation fails;
- false interventions, where continuation would have succeeded;
- harmful interventions, where continuation succeeds and recovery fails;
- recovery-plan action count and recovery-minus-continuation step and reward costs;
- safety or damage regressions on the recovery branch.

The existing bounded `nyssa-recovery-outcomes-v1` fields remain operational
rollout diagnostics. They answer whether success occurred soon after recovery.
They do not establish that recovery caused the success. Counterfactual branch
evidence is the recovery-effect measurement used by the metric vector.

## Limits

Simulation state restoration supports a counterfactual statement about the
declared simulator and component state. It does not by itself identify a
real-world causal effect. Hidden simulator state, external services, asynchronous
model workers, or undeclared policy caches can invalidate matching. Adapter
authors should report those limitations as qualified or unsupported instead of
claiming exact restoration.
