# Real evidence conformance fixtures

`v1/valid_reconstructed_family/` contains synthetic data for one real episode
and two externally reconstructed simulation variants. It tests the ingestion
boundary without implementing reconstruction.

```bash
uv run nyssa validate-real-evidence \
  conformance/real_evidence/v1/valid_reconstructed_family
```

External hardware and real-to-sim repositories can reuse the fixture with
`RealEvidenceValidator`. Recompute the package content hash after changing any
contract field.
