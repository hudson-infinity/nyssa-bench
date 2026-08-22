# Plugins

Plugins let external packages register engines, policies, suites, or report extensions without editing NyssaBench core.

## Trust Boundary

Plugins are arbitrary Python code, not declarative configuration. Importing a
plugin executes its module-level code; calling `register_plugin` executes its
registration method. A plugin has the same filesystem, network, environment,
device, subprocess, and credential access as the NyssaBench process.

Install/import only reviewed plugins from authenticated sources. A package name,
wheel signature, checksum, or successful import establishes identity/integrity,
not safety. Unknown plugins require the disposable isolated environment defined
in [SECURITY.md](../SECURITY.md), with no secrets, hardware, broad host mounts,
or network access by default.

NyssaBench does not currently auto-discover plugin entry points. The plugin
module must be imported by trusted application bootstrap code before registry
lookup. Do not add automatic scanning/import of arbitrary directories.

Minimal plugin:

```python
from nyssa_bench.plugins import NyssaPlugin, register_plugin

class MyPlugin(NyssaPlugin):
    name = "my_plugin"

    def register(self, registry):
        registry.policies["my_policy"] = MyPolicy

register_plugin(MyPlugin())
```

## Registration Rules

- Use a globally namespaced/stable plugin and registry ID.
- Reject collisions; do not silently replace a built-in or another plugin.
- Register classes/factories without starting simulators, downloading models,
  opening devices, or launching subprocesses during import.
- Defer expensive/native/resource initialization until the adapter is selected.
- Implement idempotent cleanup for partial initialization and exceptions.
- Keep plugin configuration explicit; never evaluate arbitrary strings.
- Record plugin package/version, source revision, artifact hashes, and effective
  config in run provenance.
- Treat plugin-provided suites/reports as untrusted input/output until validated.

The current registry mapping permits assignment and does not centrally prevent
name replacement. Trusted bootstrap code must check before registering:

```python
def register(self, registry):
    name = "my_org.my_policy"
    if name in registry.policies:
        raise RuntimeError(f"Policy plugin already registered: {name}")
    registry.policies[name] = MyPolicy
```

## Review Checklist

Review:

- package ownership, immutable revision, build/install scripts, dependencies,
  native extensions, model/assets, licenses, and checksums;
- top-level imports and side effects;
- filesystem/network/subprocess/dynamic import/eval/exec/deserialization use;
- environment variables and logs that could expose secrets;
- output path validation and cleanup behavior;
- observation/action/state/failure/stressor/result contracts;
- deterministic seeding and training/evaluation leakage controls;
- whether HTML/report output escapes untrusted strings and paths;
- fake-adapter tests plus required simulator/policy integration evidence.

Do not run an unreviewed policy/engine plugin on a real robot. Simulation success
does not establish that plugin code or physical actions are safe.

## Distribution

Prefer an isolated virtual environment per third-party plugin/model stack. Pin
the package and transitive dependencies, preserve the lock/constraints, and do
not install from a mutable branch. Avoid editable installs from untrusted source
trees on shared machines.

Never bundle private keys, access tokens, proprietary assets, downloaded model
weights, or customer data in a plugin package or result pack. Use separately
authorized artifact storage and record only non-secret immutable identifiers.
