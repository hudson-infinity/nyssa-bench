# External failure monitors

NyssaBench can evaluate runtime failure monitors without owning their model or
training code. A monitor may consume policy-observable inputs, policy internals,
or privileged simulator evidence, but its contract must declare every input and
its visibility. Results are reported as either `deployable_monitor` or
`privileged_monitor`; privileged results must not be presented as deployment
performance.

The built-in `action-magnitude` monitor is a deterministic integration control.
It verifies the protocol and artifact path. It is not a learned baseline or a
research contribution.

## Monitor module

An external Python module must expose `create_failure_monitor()` and return a
subclass of `nyssa_bench.monitors.FailureMonitor`:

```python
from nyssa_bench.monitors import FailureMonitor


class MyMonitor(FailureMonitor):
    def contract(self):
        ...

    def reset(self, *, task, episode_index, seed):
        ...

    def predict(self, monitor_input):
        ...


def create_failure_monitor():
    return MyMonitor()
```

The versioned contract records monitor and checkpoint identity, checkpoint and
preprocessing hashes, input visibility, output fields, prediction horizon,
alert threshold, calibration bins, state semantics, determinism, and declared
compute. A `restorable` monitor must implement `reset()`, `get_state()`, and
`set_state()`. A `resettable` monitor must implement `reset()`.

Predictions are made after a policy action is proposed and before verifier or
recovery changes that action. The observation timestamp equals the current
environment step. The action timestamp records when the action was generated,
so cached action chunks also expose their age in steps. Predictions retain the
IDs of any FailureEvents known before inference. Those IDs are retained as
alignment metadata and excluded from future-failure labels. They are supplied
to the monitor only when its contract declares the privileged
`failure_event_ids` input, which prevents future failure evidence from being
silently used as an input.

## Run a monitor

```bash
uv run nyssa run \
  --suite mujoco_control_v0 \
  --tasks mujoco_pusher \
  --engine mujoco \
  --policy random \
  --episodes 20 \
  --seed 0 \
  --failure-monitor action-magnitude \
  --out benchmark_results/mujoco_monitor_smoke \
  --no-replay
```

Repeat `--failure-monitor` to evaluate multiple monitors on the same proposed
actions. NyssaBench reports calibration, Brier score, precision, recall, false
alarms, missed failures, risk coverage, lead time, failure category and
mechanism accuracy, time-to-failure error, recovery-eligibility accuracy,
latency, and declared/observed compute. Labels whose complete prediction horizon
was not observed are censored rather than treated as successes.

The run writes `failure_monitor_predictions.json`. It contains the complete
contracts, per-episode support, timestamped predictions, outcome labels, summary
metrics, and a content hash. Loading the manifest verifies the hash and
recomputes the summary from the retained records.

Multi-monitor studies are observational. NyssaBench requires exactly one
monitor when `--enable-monitor-intervention` is set, because allowing several
monitors to alter one trajectory would confound their future labels.

## Paired comparison

Monitors may be compared only when both produced predictions for the exact same
task, episode seed, episode index, and environment step:

```bash
uv run nyssa compare-failure-monitors \
  benchmark_results/mujoco_monitor_study \
  --monitor-a monitor_a \
  --monitor-b monitor_b \
  --out benchmark_results/mujoco_monitor_study/monitor_comparison.json
```

The comparison reports paired Brier-score differences with an episode-clustered
bootstrap interval and paired classification discordance. It rejects missing,
duplicate, or differently labeled pairs. Bootstrap intervals remain unavailable
until at least two independent episodes are present.

## Intervention evaluation

A monitor contract can declare intervention recommendations. Recommendations
are observational by default. To allow them to request the configured recovery
provider, enable recovery explicitly:

```bash
uv run nyssa run \
  --suite mujoco_control_v0 \
  --tasks mujoco_pusher \
  --engine mujoco \
  --policy random \
  --episodes 20 \
  --expert-provider mujoco-heuristic \
  --enable-recovery \
  --failure-monitor path/to/monitor.py \
  --enable-monitor-intervention \
  --counterfactual-repeats 5 \
  --counterfactual-horizon 10 \
  --out benchmark_results/mujoco_monitor_intervention \
  --no-replay
```

Monitor-triggered recovery is identified separately from verifier rejection.
When counterfactual branching is enabled, the recommendation record links to
the matched continuation/recovery branch point. Prediction labels and
prediction metrics never use branch outcomes, so failure prediction quality and
recovery effectiveness remain separate claims. Branch rollouts do not invoke
the monitor recursively.

## Claim boundaries

- `deployable_monitor` means all required inputs are observations, proposed
  actions, or declared policy-internal signals available at deployment.
- `privileged_monitor` means at least one required input uses simulator state or
  FailureEvent annotations unavailable to the deployed policy.
- Unsupported required inputs are recorded per episode and produce no
  predictions; they are not replaced with default values.
- The contract identifies a monitor implementation and its evidence. It does
  not establish that the monitor is novel, calibrated out of distribution, or
  validated on hardware.
