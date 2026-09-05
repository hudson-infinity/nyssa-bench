# Hardware calibration and sim-real preregistration

The hardware calibration track tests a narrow question:

> Do temporal failure and recoverability features predict held-out real-robot
> behavior beyond clean simulation success?

The committed study at
[`configs/hardware/nyssa_hardware_calibration_v0_1.json`](../configs/hardware/nyssa_hardware_calibration_v0_1.json)
is a **draft**. It is not preregistered and contains no hardware evidence.

## Frozen design

The draft composes the existing NEP task, policy, stressor, failure,
intervention, and claim contracts. It references the reference benchmark and
policy-track registry by SHA-256 and embeds the exact task and policy contracts
used by each condition.

The initial design contains:

- tasks: PickCube and PushCube;
- policies: planner oracle, RoboMimic BC, and diffusion action chunking;
- conditions: clean and a two-step action-delay shift;
- 20 standard trials per task-policy-condition cell;
- 20 additional matched recovery trials for every shifted cell;
- 240 standard trials and 120 recovery trials, producing 360 real-evidence
  packages in total.

Every condition declares matched axes and unavoidable simulation/hardware
mismatches. The design is complete factorial; cells cannot be removed after
outcomes are observed.

## Primary analysis

The preregistered primary metrics are policy rank, failure-event distribution,
shift response, time to failure, matched recovery effect, and held-out
incremental predictive value. The baseline prediction uses clean simulation
success. The enhanced model adds severity, failure category, failure onset, and
simulation recovery gain.

Uncertainty uses 5,000 clustered bootstrap samples. Sensitivity analyses cover
leave-one-task-out, leave-one-policy-out, operator-intervention exclusion,
failure-taxonomy coarsening, and worst-case censoring bounds. A well-powered
negative incremental result must be reported; it is not converted into a
positive claim.

Recovery is analyzed only for conditions with explicit matched continuation and
recovery trial arms. Protective stops are retained or censored under frozen
rules rather than deleted as inconvenient failures.

## Safety and governance

Before collecting data, the responsible institution must review the risk
assessment and replace every pending site, robot, license, and authorization
field. The protocol requires:

- trained operators and a single designated operator per session;
- a timestamped emergency-stop test before each session;
- workspace exclusion, speed, and force controls;
- pretrial robot, sensor, calibration, and workspace checks;
- stop conditions for human entry, high-force contact, damage risk, or stream
  loss;
- retention of failures, protective stops, near misses, and damage events;
- peak contact force, stop counts, fixture displacement, and visible-damage
  annotations where measurable;
- pseudonymous operator IDs, explicit consent basis, retention, license,
  redaction, redistribution, and artifact-access rules.

NyssaBench does not authorize physical robot operation. The local institution's
safety process and robot manufacturer requirements remain controlling.

## Preregistration

Do not change the study status to `preregistered` merely because the JSON is in
Git. First release the reference benchmark and required policy tracks, finalize
site-specific safety/governance fields, and deposit the canonical
`design_sha256` with an immutable third-party timestamp. Record the resulting
`nyssa-preregistration-receipt-v1` artifact with registry URI and registration
time. The receipt must predate the first permitted trial.

The design hash intentionally excludes study status and evidence references.
Adding evidence does not rewrite what was preregistered.

## Evidence collection

Each trial produces one complete `nyssa-real-evidence-package-v1`. Matched
recovery conditions label every package as `continue` or `recovery`; standard
conditions use `standard`. Packages must identify their frozen hardware
condition in `metadata.hardware_condition_id`.

The audit requires:

- exactly the planned package count and recovery-arm balance;
- valid clocks, frames, actions, calibrations, failure events, safety events,
  interventions, provenance, and governance;
- explicit task, policy, checkpoint, condition, and trial identity;
- no dropped failed trials;
- a matching `nyssa-sim-real-study-v1` and complete report;
- available matched recovery analysis;
- passing statistical-precision and sim-real predictive BenchmarkValidity
  audits.

Raw evidence may remain protected when governance requires it, but the released
sanitized metadata must preserve identities, hashes, redaction reasons, and
provenance. Metadata-only packages do not satisfy claim readiness.

## Commands

Regenerate and verify the draft:

```bash
uv run python scripts/generate_hardware_study_candidate.py
uv run python scripts/generate_hardware_study_candidate.py --check
```

Audit the current state:

```bash
uv run nyssa audit-hardware-study \
  configs/hardware/nyssa_hardware_calibration_v0_1.json \
  --repo-root . \
  --out build/hardware-calibration
```

Exit code `0` means a declared complete study has passed every evidence check.
Exit code `2` means evidence is missing. Invalid, changed, under-counted, or
late-registered evidence fails.

The current draft reports five missing checks and no passing empirical checks.
The strongest sim-real wording remains unavailable until this audit and Phase 1
Gate C both pass.
