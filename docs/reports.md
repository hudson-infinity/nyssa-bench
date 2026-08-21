# Reports

NyssaBench writes an HTML report for each run. Reports include:

- suite, policy, engine, and episode count
- success rate
- prototype reliability score
- primary failure mode
- public-claim validation status
- requested, applied, skipped, and unsupported stressor status
- temporal failure-event summaries and evidence-backed timelines
- top failure episodes and replay links when available
- aggregate metrics
- failure counts
- raw summary JSON

The same event ledger is embedded in `replay_manifest.json`; `replay.html`
renders the complete event payload beside each episode.

Use `nyssa report <run>` to regenerate `report.html` from a run directory. Use `nyssa compare` for multi-policy reports.

Use `nyssa robustness-report <severity-run>... --out <directory>` for matched
stressor sweeps. It writes JSON, CSV, and HTML with clean and shifted success,
degradation, normalized robustness AUC, pointwise Wilson intervals, and paired
bootstrap uncertainty. The command rejects missing clean baselines, unmatched
episode identities, mixed stressors, unsupported applications, and mismatched
run contracts.
