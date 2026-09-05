# External policy examples

`state_policy.py` is a deterministic one-dimensional state policy used by the
installed MuJoCo conformance smoke. `image_chunk_policy.py` validates a uint8
HWC RGB observation and returns a four-step, seven-dimensional action chunk.

Each example has a checkpoint under `checkpoints/` and a matching NEP 0.1 Policy
Contract. The checkpoint hash in the contract includes the file's final newline;
do not edit one artifact without updating the others.

Installed users can write a self-contained copy with:

```bash
nyssa write-policy-example --kind state --out policy_example
nyssa write-policy-example --kind image-chunk --out image_chunk_example
```

These are deterministic integration controls, not learned policy baselines.
