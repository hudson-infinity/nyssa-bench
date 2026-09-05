# Stress search

NyssaBench stress-search studies allocate a fixed evaluation budget across an
executable stressor space, preserve every proposed and observed condition, and
confirm candidate success/failure boundaries with held-out seeds.

Adaptive boundary search is one supported sampler. It is not the sole novelty
claim of NyssaBench; the study protocol connects executable shifts to temporal
failure evidence, recovery, validity, and later sim-real analysis.

## Search contract

A `nyssa-stress-search-study-spec-v1` contains:

- an executable `nyssa-stress-search-space-v1`;
- sampler identity, versioned configuration, and study seed;
- discovery and confirmation budgets;
- deterministic batch size and stopping rules;
- the success-rate range used to confirm a boundary.
- study provenance and exploratory versus benchmark-claim mode.

`benchmark_claim` mode requires an embedded, claim-ready
`nyssa-benchmark-validity-report-v1`. Exploratory studies can run without one,
but their study and comparison reports set `claim_eligible` to false. Claim-mode
provenance must declare the same `benchmark_id` as the validity report.

Variables can be continuous, integer, or categorical and target either
`severity` or `parameters.<name>` on a registered stressor. Every searched
stressor must have exactly one severity variable. Numeric sum constraints and
forbidden categorical combinations are supported without evaluating arbitrary
expressions.

Search-space construction instantiates each registered Stressor Contract,
resolves representative parameters, and checks engine, task, observation-mode,
and action-mode support. Unknown or declarative-only stressors are rejected.

## Samplers

`random` independently derives every candidate from the study seed and
candidate cursor. It is the uniform baseline.

`latin_hypercube` creates a fixed-budget, per-variable permuted design. Each
continuous variable visits every stratum once before the design is exhausted.

`boundary_adaptive` begins with deterministic random exploration. Once both
success and policy failure have been observed, it finds the closest
opposite-outcome pair in normalized numeric space and proposes a jittered
midpoint. Jitter shrinks as valid observations accumulate. The uncertainty
model is the nearest opposite-outcome distance; it is not a calibrated policy
failure probability.

Set `target_boundary_width` with `min_valid_observations` to stop adaptive
discovery once the nearest normalized success/failure pair is sufficiently
narrow. Otherwise discovery stops at the budget, fixed design, or feasible
unique-point limit. The exact stopping reason is preserved in sampler state.

The adaptive sampler supports continuous and integer variables. Random and
Latin-hypercube samplers also support categorical variables. All samplers
declare their variable, constraint, batching, seed, objective, and uncertainty
capabilities in the study state.

## Observation evidence

Each `nyssa-stress-observation-v1` has one of six statuses:

- `success`
- `policy_failure`
- `unsupported`
- `censored`
- `application_error`
- `invalid`

Only success and policy failure update a boundary. Both require a complete
`nyssa-metric-vector-v1`; policy failures also require validated temporal
FailureEvent evidence. Unsupported, censored, application-error, and invalid
trials require a reason and remain separate in manifests and reports. They are
never converted into policy failures or zero scores.

## Resumable workflow

Initialize a study:

```bash
uv run nyssa stress-search-init \
  configs/stress_search/action_noise_boundary.yaml \
  --out benchmark_results/action_noise_search/study.json
```

Request a batch:

```bash
uv run nyssa stress-search-propose \
  benchmark_results/action_noise_search/study.json \
  --out benchmark_results/action_noise_search/study.json \
  --proposals-out benchmark_results/action_noise_search/proposals.json
```

The proposal batch includes a ready-to-run `nyssa-stressor-config-v1` for every
condition.

Run each proposal through the normal Nyssa evaluation path using the emitted
point, deterministic seed, and `StressSearchSpace.stressor_specs()`. Write the
resulting observation records to a JSON or YAML list, then update the study:

```bash
uv run nyssa stress-search-observe \
  benchmark_results/action_noise_search/study.json \
  benchmark_results/action_noise_search/observations.json \
  --out benchmark_results/action_noise_search/study.json
```

For a standard Nyssa run directory, ingestion can build and apply the
observation directly:

```bash
uv run nyssa stress-search-ingest-run \
  benchmark_results/action_noise_search/study.json \
  <proposal-id> benchmark_results/action_noise_search/run_000 \
  --out benchmark_results/action_noise_search/study.json \
  --observation-out benchmark_results/action_noise_search/observation_000.json
```

Ingestion validates `metrics.json`, `episodes.json`, and
`stressor_manifest.json`. It applies the spec's `outcome_success_threshold`,
collects temporal failure and safety evidence, and keeps unsupported, censored,
or invalid runs outside the policy-failure population.

Previously emitted proposals and observations are immutable. Study writes use
an atomic replacement, and loading verifies the space, sampler, budget,
configuration, proposal IDs, seeds, pending IDs, derived summary, and full
study hash. Resuming therefore continues from the saved candidate cursor.

## Boundary confirmation

After discovery proposals are observed, select confirmation conditions:

```bash
uv run nyssa stress-search-confirm \
  benchmark_results/action_noise_search/study.json \
  --out benchmark_results/action_noise_search/study.json \
  --proposals-out benchmark_results/action_noise_search/confirmation.json
```

Confirmation uses a separate deterministic seed namespace. Discovery and
confirmation seeds must be disjoint. A condition is confirmed only when every
requested held-out repeat has a valid policy outcome and its observed success
rate lies inside the prespecified boundary range. Wilson intervals and all
non-policy statuses remain in the study summary.

Submit confirmation observations with `stress-search-observe --confirmation`.

## Baseline comparison

Compare studies that use the same search space and discovery budget:

```bash
uv run nyssa stress-search-report \
  random/study.json lhs/study.json adaptive/study.json \
  --out benchmark_results/action_noise_search/comparison
```

Random, Latin-hypercube, and boundary-adaptive studies are all required, with
matched study-seed sets and confirmation contracts. JSON, CSV, and HTML outputs
report proposals to the first observed boundary, candidate and confirmed
boundary counts, confirmation coverage and intervals, and sample-efficiency
ratios against the best non-adaptive baseline. No universal deployment or
reliability score is produced.
