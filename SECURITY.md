# Security

## Report A Vulnerability

Use GitHub's enabled private vulnerability-reporting form:

**[Privately report a security vulnerability](https://github.com/hudson-infinity/nyssa-bench/security/advisories/new)**

Do not open a public issue, discussion, or pull request containing exploit
details, private datasets, credentials, or an undisclosed vulnerable artifact.
If the private form is temporarily unavailable, open a public issue containing
only a request for an alternate private contact channel. Do not include security
details in that issue.

Include:

- affected commit/release, operating system, Python version, and simulator;
- vulnerable command/API and required optional extras;
- minimal reproduction or proof of concept;
- impact and realistic attack prerequisites;
- whether untrusted code/data, network access, GPU/native libraries, or robot
  hardware are involved;
- suggested mitigation, if known;
- your preferred disclosure credit and communication constraints.

## Response Targets

These are coordination targets, not guarantees:

- acknowledgment within 3 business days;
- initial triage and next-step communication within 7 calendar days;
- a status update at least every 14 calendar days while remediation is active;
- critical-impact remediation target within 30 days where feasible;
- high-impact remediation target within 60 days where feasible.

Lower-severity issues may be scheduled for a normal release. Complex issues in
upstream simulators, drivers, model frameworks, or native parsers may require
coordinated timelines with those maintainers. We will communicate material
changes to scope or timing through the private advisory.

Please allow coordinated disclosure until a fix or mitigation is available and
users have had a reasonable update window. A typical maximum embargo target is
90 days unless reporter and maintainers agree otherwise. Maintainers will credit
reporters who request credit and comply with coordinated disclosure.

## Supported Versions

| Version | Security support |
| --- | --- |
| Latest `0.1.x` release | Supported |
| Current `main` | Supported for pre-release testing and coordinated fixes |
| Older releases/commits | Not supported; reproduce against the latest release or `main` |
| Forks with unmerged changes | Contact the fork maintainer first unless the issue also affects upstream NyssaBench |

Security fixes may require updating NyssaBench, Python dependencies, simulator
packages, model frameworks, drivers, or assets together. Unsupported upstream
versions may receive mitigation guidance but not a backport.

## Security Scope

Relevant reports include:

- unintended code execution or sandbox escape while processing an artifact that
  is documented as data-only;
- path traversal, unsafe archive/media handling, overwrite, or deletion outside
  an intended output directory;
- credential/token disclosure in logs, manifests, reports, datasets, videos, or
  result packs;
- command injection beyond the explicitly trusted command-template boundary;
- unsafe deserialization or dependency confusion in NyssaBench integration;
- denial of service from malformed artifacts where practical validation should
  exist;
- privilege, device, network, or filesystem access exceeding documented trust;
- integrity defects that can silently falsify benchmark evidence or claim
  validation.

Normally out of scope:

- a deliberately malicious policy taking unsafe robot actions when it was
  intentionally executed as trusted code with hardware access;
- vulnerabilities solely in an unsupported upstream dependency with no
  NyssaBench-specific exposure;
- missing security controls already documented as unsupported, unless the
  implementation behaves more permissively than documented;
- social engineering, physical intrusion, or denial-of-service testing against
  public infrastructure without prior authorization.

## Trust Model

NyssaBench is a research benchmark, not a sandbox. Several inputs are executable
or reach complex native parsers. File extension does not establish trust.

| Input | Trust classification | Why |
| --- | --- | --- |
| Policy/expert Python files | Executable code | Imported with Python module loaders; top-level code and factories run in-process. |
| `module:attribute` policies/plugins/engine factories | Executable code | Importing a module executes it; classes/factories can access process privileges. |
| PyTorch/RoboMimic `.pth`/pickle checkpoints | Executable or unsafe serialized input | Upstream loaders may use pickle-compatible deserialization and import model code. |
| Safetensors weights | Safer data container, not a trusted model | Avoids pickle payloads, but model/config/tokenizer/custom code and native kernels still require trust. |
| HDF5/H5 datasets | Untrusted native-parser input | HDF5 is complex, can consume large resources, and may contain links or malformed structures. |
| JSON/JSONL/CSV/Parquet | Untrusted data | Generally data-only, but can exhaust memory/disk or carry unsafe paths/HTML/metadata. |
| YAML task/suite/stressor configs | Untrusted control data | NyssaBench uses `yaml.safe_load`, but values select paths, factories, assets, engines, and commands. |
| ZIP result/data archives | Untrusted compressed data | Current episode loader reads selected members in place but has no archive-size/compression-ratio limits. |
| URDF/MJCF/SDF, meshes, textures, shaders, simulator scenes | Untrusted native asset input | Parsed by simulator/rendering/native libraries and may reference external files. |
| Plugins | Executable code | Registration imports and runs plugin code with host process privileges. |
| ManiSkill command templates | Shell code | Collection uses `subprocess.run(..., shell=True)` after placeholder substitution. |
| HTML reports/replay viewers | Untrusted active content | Third-party HTML can execute script when opened; serve/open only in an isolated browser context. |

`yaml.safe_load` prevents arbitrary YAML object constructors; it does not make a
configuration semantically safe. Checksums prove identity, not safety.

## Safe Artifact Workflow

For third-party models, datasets, result archives, or simulator assets:

1. Obtain them from an authenticated, documented source over TLS.
2. Record immutable version/revision, license, expected size, and a cryptographic
   checksum. Verify before opening.
3. Inspect archive member names, declared/uncompressed sizes, compression ratios,
   and file types without extracting. Reject absolute paths, `..`, device names,
   symlinks, and unexpected executables.
4. Quarantine and scan where organizational policy requires it.
5. Process first in a disposable environment with no secrets, no robot hardware,
   minimal read-only inputs, a dedicated writable output directory, network
   disabled by default, and CPU/memory/disk/process/time limits.
6. Review generated manifests/logs and validate output paths before moving
   artifacts into a trusted workspace.
7. Promote an artifact to trusted use only after provenance, license, parser,
   model code, and behavior review. Preserve the checksum in experiment notes.

Do not load an untrusted `.pth` merely to inspect its metadata. Prefer published
signed manifests and data-only formats; otherwise inspect in an isolated
throwaway machine/container with the same care as running unknown Python.

Do not mount a Docker/Podman daemon socket, SSH agent, cloud credential directory,
home directory, or broad host filesystem into an artifact-analysis container.
Container root is not a strong boundary when privileged devices, host IPC, or
the container runtime socket are exposed. Use a VM or dedicated machine for
high-risk artifacts or adversarial submissions.

## When Isolation Is Required

Use a disposable VM, sandbox, or dedicated machine when any of these apply:

- source/reviewer identity or artifact provenance is unknown;
- Python policy, expert, plugin, tokenizer/model repository, or engine factory
  code has not been reviewed;
- a checkpoint may use pickle/PyTorch deserialization;
- HDF5/native simulator assets come from an untrusted party;
- a shell command template is externally supplied;
- the workload requires parsing adversarial archives or HTML;
- a vulnerability proof of concept is being reproduced.

Default isolation controls:

- unprivileged user and no `sudo`;
- no real robot, motor controller, serial/USB device, camera, microphone, or
  production GPU partition unless explicitly required and isolated;
- no credentials or persistent authentication caches;
- network disabled or allowlisted to required immutable sources;
- read-only source/artifact mount and empty dedicated output volume;
- resource/time limits and process-count limits;
- throw away the environment after processing.

Never move an unreviewed policy from simulation directly onto hardware. Hardware
deployment requires independent action/safety limits, emergency stop, workspace
isolation, low-speed validation, and operator supervision outside NyssaBench.

## Format-Specific Guidance

### Python Policies, Experts, Factories, And Plugins

Direct policy/expert files, environment `module:attribute` loaders, Genesis or
RoboCasa factories, and plugins execute with all permissions of the NyssaBench
process. Review imports, top-level code, subprocess/network/filesystem use,
dynamic evaluation, native extensions, and model download options. Run unknown
code only in strong isolation.

### Checkpoints And Model Repositories

Treat `.pt`, `.pth`, `.ckpt`, `.pkl`, `.pickle`, joblib, and many framework
checkpoint formats as executable. NyssaBench's RoboMimic integration delegates
checkpoint loading upstream and cannot make an untrusted checkpoint safe.
Disable remote custom code where frameworks support it, pin revisions, prefer
safetensors for weights, and review the architecture/config/tokenizer code.

### HDF5 And Third-Party Datasets

Open unknown HDF5 only in an isolated process with resource limits. Reject
unexpected external links, object counts, dimensions, dtypes, compressed sizes,
and paths. Do not trust metadata strings as shell arguments or filesystem paths.
Imported demonstrations can contain privileged simulator state and proprietary
observations; preserve access controls and license restrictions.

### YAML And Configuration

Keep `yaml.safe_load`; never switch repository config loading to unsafe YAML
constructors. Validate allowed fields, bounded sizes/nesting, identifiers,
resolved paths, environment IDs, factory references, stressor support, and output
directories before execution. Treat factory/plugin/module paths and command
templates as executable selections.

### Simulator Assets And Rendering

Process unknown robot/scene/mesh/texture/shader files with current simulator,
renderer, image/video, Vulkan/OpenGL, and driver patches. Block external path or
network references unless reviewed. Malformed assets can target native parsers,
consume GPU/CPU memory, or expose sensitive visual data in replay outputs.

### Command Templates

`collect-maniskill-demos --command-template` and
`NYSSA_MANISKILL_DEMO_COMMAND` are trusted shell inputs. Placeholder values are
quoted, but the template itself is passed to `shell=True`. Never accept a command
template from an untrusted config, dataset, issue, or web request. Prefer fixed
argument arrays in new integrations.

## Secrets And Result Packs

Run benchmark jobs with least-privilege, short-lived credentials. Model/data
downloads should happen in a separate preparation step where possible; remove
tokens before policy/simulator execution.

Never include these in committed or shared result packs:

- API keys, bearer tokens, cookies, `.env` files, cloud credentials, SSH/GPG
  material, registry credentials, or signed URLs;
- full environment-variable dumps or process arguments containing secrets;
- private home/workspace paths, usernames, hostnames, IPs, bucket names, or
  internal repository URLs unless explicitly approved;
- unredacted command stdout/stderr, crash dumps, traces, or manifests containing
  sensitive paths/data;
- proprietary/raw camera frames, human images, audio, task instructions,
  privileged simulator state, or customer assets without authorization;
- model weights or datasets whose license/access policy forbids redistribution.

Review at least `run.yaml`, `config.yaml`, `environment.json`,
`package_versions.json`, `git_info.json`, policy metadata, collection/import
manifests, episode/failure evidence, HTML, videos, and archive member names before
sharing. NyssaBench records interpreter paths, platform details, Git branch/commit,
source paths, commands, and model metadata that may need redaction. Redaction
must not silently remove evidence required for a claim; publish an explicit
redaction manifest and lower the claim tier when required provenance is absent.

Do not open untrusted generated HTML on an authenticated application origin or
serve it from a domain with sensitive cookies. Use a local isolated browser
profile or render to a passive format after inspection.

## Dependency And Supply-Chain Hygiene

- Use the lockfile and pinned pre-commit revisions.
- Review dependency/lock changes, package sources, install scripts, and licenses.
- Pin external Git/model revisions to immutable commits; avoid `latest` and
  unreviewed remote custom code.
- Keep Python, simulators, PyTorch, HDF5, image/video codecs, renderers, GPU
  drivers, and system libraries patched.
- Verify released artifacts/checksums where upstream publishes them.
- Do not publish generated checkpoints, datasets, videos, or archives from an
  untrusted run as repository source.

## Disclosure And Release Process

Maintainers will reproduce in isolation, assess affected versions and artifact
trust boundaries, coordinate upstream where necessary, develop tests and a fix
or mitigation, request reporter validation when practical, and publish a GitHub
security advisory/release note after coordinated disclosure.

Public advisories should identify affected versions, impact, prerequisites,
fixed versions/commits, mitigations, and whether previously generated result
packs require revocation or revalidation. Exploit payloads and sensitive artifacts
will not be committed to the public test suite.
