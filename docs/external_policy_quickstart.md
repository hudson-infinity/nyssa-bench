# External policy conformance quickstart

This workflow takes an external policy from a Python adapter to a conformance
report and a small NyssaBench result pack. It validates the interface and
provenance. It does not establish policy quality or a public benchmark result.

## Installed MuJoCo example

Install the released MuJoCo workflow and write the packaged state-policy
example into a working directory:

```bash
python -m pip install "nyssa-bench[mujoco]"
nyssa write-policy-example --kind state --out policy_example
```

Run preflight and one integration episode:

```bash
nyssa conform-policy \
  --policy policy_example/state_policy.py \
  --policy-contract policy_example/state_policy_contract.json \
  --suite mujoco_control_v0 \
  --task mujoco_inverted_pendulum \
  --engine mujoco \
  --episodes 1 \
  --out policy_example/conformance
```

On a machine with a working renderer, add `--capture-replay`. MuJoCo and
ManiSkill system rendering requirements still apply.

The output contains:

- `policy_conformance.json`, with every expected and observed contract value;
- `policy_conformance.html`, with the same checks in a readable report;
- `smoke_run/`, containing metrics, episodes, dataset/NEP manifests, and HTML
  result files. The NEP envelope carries the external checkpoint's Policy
  Contract rather than the built-in smoke policy identity.

The command returns 0 when the adapter is conformant and 3 when a check fails.

## What preflight checks

The command selects exactly one task and checks the policy before a full
experiment:

1. policy, version, family, checkpoint, preprocessing, modality, action, and
   horizon metadata against the NEP Policy Contract;
2. reset, action, and close lifecycle methods;
3. explicit task-to-engine mapping or registered external engine capability;
4. live observation modalities from a real reset;
5. live action dimension and bounds against the contract;
6. numeric, finite, in-bounds action values;
7. exact predicted action-chunk shape and execution horizon;
8. matched repeated resets for every deterministic seeding claim;
9. dependency/device execution through the actual `policy.act` call;
10. a small result pack only after every preflight check passes.

Action exceptions, missing observations, non-finite values, bad shapes, stale
state, and metadata mismatches include the policy, engine, task, phase,
expected contract, and observed value. A failed preflight does not call
`engine.step()` or start the smoke run.

## Policy Contract

The conformance command uses `nyssa-nep-policy-contract-v0.1`. The contract
pins:

- policy identity, semantic version, and family;
- checkpoint identity and SHA-256;
- preprocessing SHA-256;
- required observation modalities;
- action representation, dimension, and bounds;
- prediction and execution horizons;
- state/reset semantics and deterministic seeding;
- declared training datasets and split lineage.

The adapter's `metadata()` must independently report the identity, modalities,
action representation/dimension, horizons, device, and hashes. A contract file
cannot substitute for live adapter provenance.

## Image and action-chunk example

Write the second packaged example:

```bash
nyssa write-policy-example --kind image-chunk --out image_chunk_example
```

It requires one uint8 HWC RGB observation and returns a `4 x 7` action chunk
with execution horizon 2. Use it with a task/engine that actually exposes that
RGB and seven-dimensional action contract. An incompatible state-only task
fails during preflight rather than silently flattening or inventing an image.

## External registration

A standalone policy file with `create_policy()` or `PolicyAdapter` is the most
portable registration path and requires no NyssaBench source edits. Python
projects that need several named components may implement `NyssaPlugin` and call
`register_plugin()` during their own package initialization.

Do not put project-specific checkpoint loading, model dependencies, or training
code into NyssaBench. Keep those in the policy project and expose the stable
adapter plus Policy Contract.

## Evidence labels

- Integration-only adapter: the file loads or a named hook exists, but strict
  conformance has not passed.
- Conformant policy: preflight and the small result pack pass for the selected
  task, engine, and checkpoint.
- Validated reference track: #16 requirements also pass across held-out clean
  and shifted conditions, paired seeds, leakage checks, uncertainty, and replay
  evidence.

A conformant smoke result remains `prototype` and `public_claim: false` unless
the independent run and benchmark evidence gates say otherwise.
