# NEP conformance fixtures

`0.1.0/valid` contains canonical pipeline-control manifests for MuJoCo and
ManiSkill. They validate schema interoperability only and do not claim simulator
execution.

`0.1.0/invalid` contains deliberately corrupted content hashes and unknown
artifact references. Validators must reject every file in that directory.

Regenerate the fixtures and schemas together with:

```bash
python scripts/generate_nep_artifacts.py
```
