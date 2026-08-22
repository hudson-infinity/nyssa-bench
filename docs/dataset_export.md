# Dataset Export

v0.1 writes JSON rollouts and a lightweight LeRobot-compatible manifest without requiring LeRobot at runtime.

Optional HDF5, robomimic HDF5, and Parquet exporters are available through the `dataset` extra:

```bash
uv sync --inexact --extra dataset
```

`--inexact` adds the exporter dependencies without removing simulator or policy
extras already installed in the active environment. A fresh full benchmark
environment can use `uv sync --extra all --extra dev` instead.

Export a run for robomimic BC training:

```bash
uv run nyssa export --run runs/scripted_oracle --format robomimic --out runs/scripted_oracle/robomimic.hdf5
```

The RoboMimic exporter requires a finite Box action contract on every step,
requires consistent action shape and bounds within one HDF5 file, rejects
out-of-bounds demonstrations, and affinely normalizes actions to `[-1, 1]`.
The original bounds are stored in the HDF5 `data` group's
`nyssa_action_transform` attribute. Use `export-task-robomimic` when tasks have
different action contracts.

JSON and JSONL episode exports preserve the complete
`nyssa-failure-ledger-v1` payload. Legacy failed episodes with only a flat
failure label receive a terminal-only migration event when loaded for export.
Specialized learned-policy exports continue to use the flat label until the
downstream failure-evidence export contract is versioned separately.

## Security And Untrusted Data

Dataset import/export is not a sandbox. Follow [SECURITY.md](../SECURITY.md)
before opening third-party demonstrations, result archives, checkpoints, or
simulator assets.

| Format/path | Main risk | Required handling |
| --- | --- | --- |
| JSON/JSONL/CSV | Resource exhaustion, unsafe paths/metadata, sensitive observations | Validate size/schema/depth, paths, provenance, and redaction before use. |
| ZIP result sources | Compression bombs and oversized members | Inspect member names, uncompressed sizes, ratios, and total size in isolation. |
| HDF5/H5 | Native parser attack surface, external links, huge shapes/dtypes | Open in an isolated resource-limited process; reject unexpected links/objects. |
| Parquet/Arrow | Native parser/codecs and resource exhaustion | Pin patched libraries and enforce schema/row/column/size limits. |
| RoboMimic `.pth` | Pickle-compatible code execution through upstream loader | Treat as executable; load only reviewed checkpoints in isolation. |
| LeRobot/model directories | Model/config/custom Python and mutable remote artifacts | Pin/review all code and revisions; prefer data-only weight formats. |

NyssaBench's ZIP episode loader reads selected `episodes.json` members without
extracting them to disk, which avoids ordinary extraction traversal. It does not
currently enforce compressed/uncompressed size or ratio limits. Do not feed an
untrusted archive to training/export commands outside a resource-limited
throwaway environment.

`import-maniskill-demos` recursively opens `.h5`/`.hdf5` files with h5py and
walks groups containing actions. It does not make adversarial HDF5 safe. Verify
source, checksum, license, expected file/object sizes, environment/task IDs,
observation/action shapes, dtypes, and external-link policy before import.

RoboMimic export performs observation coverage/variance checks and strict finite
Box action validation. These are scientific contract checks, not malware or
parser-safety checks.

## Safe Dataset Workflow

1. Download to quarantine from an authenticated source; record immutable
   revision, checksum, license, and expected size.
2. Inspect archive/container structure without loading model code or extracting
   into the repository.
3. Process in a disposable unprivileged VM/container with no credentials,
   network disabled by default, read-only input, dedicated empty output, and
   CPU/memory/disk/process/time limits.
4. Validate task IDs, episode counts/seeds, observation/action contracts,
   success/failure fields, paths, and provenance.
5. Scan and review produced JSON/manifests/logs before moving them into a trusted
   workspace.
6. Store large data externally; do not commit datasets, checkpoints, videos, or
   result archives to the repository.

For high-risk inputs, a container sharing the host Docker socket, home directory,
SSH agent, cloud credentials, or privileged GPU/devices is not adequate
isolation. Use a dedicated VM/machine.

## Paths And Output Directories

Use a new dedicated output directory. Resolve and verify it before running an
import/export; do not point output at the repository root, a home directory, or
an existing source dataset. Keep input read-only.

Review artifact-internal paths before use. Replay validation confines declared
media paths to a run directory, but not every dataset/checkpoint metadata field
is a safe filesystem path. Never pass untrusted metadata directly to shell
commands, model factories, or output paths.

Generated RoboMimic configs contain resolved absolute data/output paths. Review
before sharing; they can disclose usernames/workspace layout and can direct
training writes to unintended locations on another machine.

## Sensitive Data And Result Packs

Robot episodes may contain camera images, human activity, proprietary scenes,
task instructions, object identities, privileged simulator state, or real-world
telemetry. Confirm consent, data governance, access policy, and redistribution
license before export.

Before sharing a dataset/result pack, inspect:

- episode observations, failure evidence, videos, and HTML;
- run/config/environment/package/Git/policy metadata;
- source paths, commands, stdout/stderr, signed URLs, and external dataset/model
  identifiers;
- archive member names and file permissions;
- model/dataset licenses and customer restrictions.

Remove API tokens, cookies, `.env`, cloud/registry/SSH credentials, internal
URLs, private paths/hostnames, and unauthorized visual/state data. Record
redactions explicitly. If required provenance is removed, lower the result claim
tier rather than presenting the pack as fully reproducible.

Do not open third-party generated HTML in an authenticated browser profile or
serve it from a sensitive application origin. Treat it as active content.
