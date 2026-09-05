from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from nyssa_bench.version import __version__

from nyssa_bench.core.registry import (
    ENGINE_REGISTRY,
    ENGINE_SUPPORT_TIER,
    POLICY_REGISTRY,
    POLICY_SUPPORT_TIER,
)
from nyssa_bench.core.suite import Suite, list_suites
from nyssa_bench.core.task import TaskSpec, list_tasks
from nyssa_bench.credibility import (
    evaluate_credibility,
    load_credibility_spec,
    write_credibility_report,
)
from nyssa_bench.baselines.simple_bc import (
    train_knn_bc,
    train_linear_bc,
    train_sequence_knn_bc,
    train_task_bc,
)
from nyssa_bench.datasets.export_hdf5 import export_hdf5
from nyssa_bench.datasets.export_json import export_json
from nyssa_bench.datasets.export_jsonl import export_jsonl
from nyssa_bench.datasets.export_lerobot import export_lerobot
from nyssa_bench.datasets.export_parquet import export_parquet
from nyssa_bench.datasets.export_robomimic import export_robomimic_hdf5
from nyssa_bench.datasets.export_task_robomimic import export_task_robomimic
from nyssa_bench.datasets.collect_maniskill import collect_maniskill_demos
from nyssa_bench.datasets.import_maniskill import import_maniskill_demos
from nyssa_bench.datasets.recovery_training import train_recovery_bc
from nyssa_bench.reports.comparison import (
    compare_runs,
    save_comparison_report,
    save_leaderboard,
)
from nyssa_bench.reports.html_report import Report
from nyssa_bench.reports.result_pack import (
    write_experiment_manifest,
    write_results_markdown,
)
from nyssa_bench.reports.replay_validation import validate_result_pack_replays
from nyssa_bench.reports.scorecard import write_scorecard
from nyssa_bench.real_evidence import (
    REAL_EVIDENCE_PACKAGE_FORMAT,
    RealEvidencePackage,
    RealEvidenceValidator,
    write_real_evidence_artifacts,
)
from nyssa_bench.recovery import (
    COUNTERFACTUAL_RECOVERY_MANIFEST_FORMAT,
    load_counterfactual_recovery_manifest,
)
from nyssa_bench.regression import (
    RegressionStudyEvaluator,
    fingerprint_run,
    load_regression_study,
    write_regression_report,
)
from nyssa_bench.runner import (
    DEFAULT_COUNTERFACTUAL_HORIZON,
    DEFAULT_COUNTERFACTUAL_MAX_BRANCH_POINTS,
    DEFAULT_RECOVERY_ATTRIBUTION_HORIZON,
    PolicyRunner,
)
from nyssa_bench.runners import (
    ExperimentCell,
    ExperimentRunner,
    ablation_cells,
    policy_seed_cells,
)
from nyssa_bench.scenarios import (
    SCENARIO_PACKAGE_FORMAT,
    ScenarioPackage,
    ScenarioPackageValidator,
    scenario_execution_context,
)
from nyssa_bench.simreal import (
    evaluate_sim_real_study,
    load_sim_real_study,
    write_sim_real_report,
)
from nyssa_bench.metrics.run_claims import PUBLIC_CLAIM_ENGINES
from nyssa_bench.metrics.vector import migrate_metric_summary
from nyssa_bench.monitors import (
    compare_monitor_records,
    load_monitor_manifest,
)
from nyssa_bench.nep import (
    load_nep_data,
    migrate_nep_data,
    validate_nep_manifest,
    write_nep_validation_report,
    write_schemas,
)
from nyssa_bench.policy_conformance import (
    evaluate_policy_conformance,
    load_policy_contract,
    write_policy_conformance_report,
    write_policy_example,
)
from nyssa_bench.learning_export import (
    LEARNING_EXPORT_MANIFEST_FORMAT,
    ExportSplit,
    LearningExportConfig,
    export_learning_evidence,
    load_learning_evidence,
)
from nyssa_bench.baselines.robomimic_bc import (
    train_robomimic,
    write_robomimic_bc_config,
)
from nyssa_bench.stressors import (
    STRESSOR_CONFIG_FORMAT,
    StressorConfig,
    list_stressors,
    load_robustness_sweep,
    save_robustness_report,
)
from nyssa_bench.stress_search import (
    STRESS_SEARCH_SPEC_FORMAT,
    STRESS_SEARCH_STUDY_FORMAT,
    StressSearchStudy,
    compare_stress_search_studies,
    load_stress_observations,
    load_stress_search_spec,
    load_stress_search_study,
    observation_from_run,
    write_stress_proposals,
    write_stress_search_report,
    write_stress_search_study,
    write_stress_observations,
)
from nyssa_bench.validity import (
    BENCHMARK_VALIDITY_REPORT_FORMAT,
    BENCHMARK_VALIDITY_SPEC_FORMAT,
    BenchmarkValidityEvaluator,
    load_benchmark_validity_report,
    load_benchmark_validity_spec,
    write_benchmark_validity_report,
)


def _add_counterfactual_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--counterfactual-repeats",
        type=int,
        default=0,
        help="run this many matched continuation/recovery trials at each sampled recovery decision",
    )
    parser.add_argument(
        "--counterfactual-horizon",
        type=int,
        default=DEFAULT_COUNTERFACTUAL_HORIZON,
        help="maximum steps to execute in each counterfactual branch",
    )
    parser.add_argument(
        "--counterfactual-oracle",
        action="store_true",
        help="also execute a matched expert/oracle branch when expert state is restorable",
    )
    parser.add_argument(
        "--counterfactual-max-branch-points",
        type=int,
        default=DEFAULT_COUNTERFACTUAL_MAX_BRANCH_POINTS,
        help="maximum recovery decisions to branch from per episode",
    )


def _add_failure_monitor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--failure-monitor",
        action="append",
        default=[],
        help="built-in monitor name or Python module path; may be repeated",
    )
    parser.add_argument(
        "--enable-monitor-intervention",
        action="store_true",
        help="allow monitor recommendations to request the configured recovery provider",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nyssa")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-suites")
    subparsers.add_parser("list-tasks")
    subparsers.add_parser("list-engines")
    subparsers.add_parser("list-policies")
    subparsers.add_parser("list-stressors")

    nep_validate_parser = subparsers.add_parser("validate-nep")
    nep_validate_parser.add_argument("manifest")
    nep_validate_parser.add_argument("--out", required=True)

    nep_schema_parser = subparsers.add_parser("write-nep-schemas")
    nep_schema_parser.add_argument("--out", required=True)

    policy_conformance_parser = subparsers.add_parser("conform-policy")
    policy_conformance_parser.add_argument("--policy", required=True)
    policy_conformance_parser.add_argument("--policy-contract", required=True)
    policy_conformance_parser.add_argument("--suite", required=True)
    policy_conformance_parser.add_argument("--task", required=True)
    policy_conformance_parser.add_argument("--engine", required=True)
    policy_conformance_parser.add_argument("--episodes", type=int, default=1)
    policy_conformance_parser.add_argument("--capture-replay", action="store_true")
    policy_conformance_parser.add_argument("--out", required=True)

    policy_example_parser = subparsers.add_parser("write-policy-example")
    policy_example_parser.add_argument(
        "--kind", choices=["state", "image-chunk"], required=True
    )
    policy_example_parser.add_argument("--out", required=True)

    sim_real_parser = subparsers.add_parser("sim-real-study")
    sim_real_parser.add_argument("spec")
    sim_real_parser.add_argument("--out", required=True)

    credibility_parser = subparsers.add_parser("credibility-gate")
    credibility_parser.add_argument("spec")
    credibility_parser.add_argument("--out", required=True)
    credibility_parser.add_argument("--repo-root", default=".")

    benchmark_audit_parser = subparsers.add_parser("audit-benchmark")
    benchmark_audit_parser.add_argument("spec")
    benchmark_audit_parser.add_argument("--out", required=True)

    regression_parser = subparsers.add_parser("regression-gate")
    regression_parser.add_argument("spec")
    regression_parser.add_argument("--out", required=True)

    regression_fingerprint_parser = subparsers.add_parser(
        "regression-fingerprint"
    )
    regression_fingerprint_parser.add_argument("run")
    regression_fingerprint_parser.add_argument("--out", required=True)

    stress_init_parser = subparsers.add_parser("stress-search-init")
    stress_init_parser.add_argument("spec")
    stress_init_parser.add_argument("--out", required=True)

    stress_propose_parser = subparsers.add_parser("stress-search-propose")
    stress_propose_parser.add_argument("study")
    stress_propose_parser.add_argument("--count", type=int)
    stress_propose_parser.add_argument("--out", required=True)
    stress_propose_parser.add_argument("--proposals-out", required=True)

    stress_observe_parser = subparsers.add_parser("stress-search-observe")
    stress_observe_parser.add_argument("study")
    stress_observe_parser.add_argument("observations")
    stress_observe_parser.add_argument("--confirmation", action="store_true")
    stress_observe_parser.add_argument("--out", required=True)

    stress_ingest_parser = subparsers.add_parser("stress-search-ingest-run")
    stress_ingest_parser.add_argument("study")
    stress_ingest_parser.add_argument("proposal_id")
    stress_ingest_parser.add_argument("run")
    stress_ingest_parser.add_argument("--out", required=True)
    stress_ingest_parser.add_argument("--observation-out", required=True)

    stress_confirm_parser = subparsers.add_parser("stress-search-confirm")
    stress_confirm_parser.add_argument("study")
    stress_confirm_parser.add_argument("--out", required=True)
    stress_confirm_parser.add_argument("--proposals-out", required=True)

    stress_report_parser = subparsers.add_parser("stress-search-report")
    stress_report_parser.add_argument("studies", nargs="+")
    stress_report_parser.add_argument("--out", required=True)

    scenario_validate_parser = subparsers.add_parser("validate-scenario")
    scenario_validate_parser.add_argument("scenario")
    scenario_validate_parser.add_argument("--engine")
    scenario_validate_parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="allow unresolved protected assets while validating package identity and metadata",
    )

    scenario_run_parser = subparsers.add_parser("run-scenario")
    scenario_run_parser.add_argument("scenario")
    scenario_run_parser.add_argument("--policy", default="random")
    scenario_run_parser.add_argument("--episodes", type=int, default=10)
    scenario_run_parser.add_argument("--seed", type=int)
    scenario_run_parser.add_argument("--severity", action="append", default=[])
    scenario_run_parser.add_argument("--out", required=True)
    scenario_run_parser.add_argument("--no-replay", action="store_true")
    scenario_run_parser.add_argument("--capture-replay", action="store_true")
    scenario_run_parser.add_argument("--expert-provider", default="none")
    scenario_run_parser.add_argument("--enable-recovery", action="store_true")
    scenario_run_parser.add_argument("--enable-verifier", action="store_true")
    scenario_run_parser.add_argument("--benchmark-validity")
    scenario_run_parser.add_argument("--policy-action-horizon", type=int, default=1)
    scenario_run_parser.add_argument("--policy-execution-horizon", type=int, default=1)

    real_validate_parser = subparsers.add_parser("validate-real-evidence")
    real_validate_parser.add_argument("package")
    real_validate_parser.add_argument("--metadata-only", action="store_true")

    real_import_parser = subparsers.add_parser("import-real-evidence")
    real_import_parser.add_argument("package")
    real_import_parser.add_argument("--out", required=True)
    real_import_parser.add_argument("--metadata-only", action="store_true")
    scenario_run_parser.add_argument(
        "--recovery-attribution-horizon",
        type=int,
        default=DEFAULT_RECOVERY_ATTRIBUTION_HORIZON,
    )
    _add_counterfactual_arguments(scenario_run_parser)
    _add_failure_monitor_arguments(scenario_run_parser)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--suite", required=True)
    run_parser.add_argument("--tasks", nargs="+")
    run_parser.add_argument("--engine", default="maniskill")
    run_parser.add_argument("--policy", default="random")
    run_parser.add_argument("--episodes", type=int, default=10)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--no-replay", action="store_true")
    run_parser.add_argument("--capture-replay", action="store_true")
    run_parser.add_argument("--expert-provider", default="none")
    run_parser.add_argument("--enable-recovery", action="store_true")
    run_parser.add_argument("--enable-verifier", action="store_true")
    run_parser.add_argument("--policy-action-horizon", type=int, default=1)
    run_parser.add_argument("--policy-execution-horizon", type=int, default=1)
    run_parser.add_argument(
        "--recovery-attribution-horizon",
        type=int,
        default=DEFAULT_RECOVERY_ATTRIBUTION_HORIZON,
    )
    run_parser.add_argument("--stressor-config")
    run_parser.add_argument("--benchmark-validity")
    _add_counterfactual_arguments(run_parser)
    _add_failure_monitor_arguments(run_parser)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("run")

    monitor_compare_parser = subparsers.add_parser("compare-failure-monitors")
    monitor_compare_parser.add_argument("manifest")
    monitor_compare_parser.add_argument("--monitor-a", required=True)
    monitor_compare_parser.add_argument("--monitor-b", required=True)
    monitor_compare_parser.add_argument("--out", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--run", required=True)
    export_parser.add_argument(
        "--format",
        choices=["json", "jsonl", "lerobot", "hdf5", "parquet", "robomimic"],
        default="lerobot",
    )
    export_parser.add_argument("--out")
    export_parser.add_argument("--feature-dim", type=int, default=256)

    learning_export_parser = subparsers.add_parser("export-learning-evidence")
    learning_export_parser.add_argument("runs", nargs="+")
    learning_export_parser.add_argument("--out", required=True)
    learning_export_parser.add_argument("--benchmark-id", required=True)
    learning_export_parser.add_argument("--split-id", required=True)
    learning_export_parser.add_argument(
        "--split-partition",
        required=True,
        choices=["train", "validation", "public_test", "hidden_test"],
    )
    learning_export_parser.add_argument("--split-sha256", required=True)
    learning_export_parser.add_argument(
        "--policy-family",
        action="append",
        required=True,
        help="POLICY_ID=FAMILY; use *=FAMILY as an explicit fallback",
    )
    learning_export_parser.add_argument(
        "--license", action="append", dest="licenses", required=True
    )
    learning_export_parser.add_argument(
        "--privacy-level",
        choices=["public", "restricted", "private"],
        default="public",
    )
    learning_export_parser.add_argument(
        "--privacy-restriction", action="append", default=[]
    )
    learning_export_parser.add_argument("--failures-only", action="store_true")
    learning_export_parser.add_argument("--boundary-study", action="append", default=[])
    learning_export_parser.add_argument(
        "--max-inline-observation-bytes", type=int, default=1_000_000
    )
    learning_export_parser.add_argument(
        "--verify-external-artifacts", action="store_true"
    )

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("runs", nargs="+")
    compare_parser.add_argument("--out", required=True)
    compare_parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="emit explicitly non-comparable exploratory output instead of rejecting mismatched runs",
    )

    robustness_parser = subparsers.add_parser("robustness-report")
    robustness_parser.add_argument("runs", nargs="+")
    robustness_parser.add_argument("--out", required=True)
    robustness_parser.add_argument("--bootstrap-samples", type=int, default=1000)
    robustness_parser.add_argument("--bootstrap-seed", type=int, default=0)

    leaderboard_parser = subparsers.add_parser("leaderboard")
    leaderboard_parser.add_argument("runs", nargs="+")
    leaderboard_parser.add_argument("--out", required=True)
    leaderboard_parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="emit explicitly non-comparable exploratory output instead of rejecting mismatched runs",
    )

    scorecard_parser = subparsers.add_parser("scorecard")
    scorecard_parser.add_argument("runs", nargs="+")
    scorecard_parser.add_argument(
        "--out", default="benchmark_results/baselines_v0.json"
    )
    scorecard_parser.add_argument("--benchmark", default="NyssaBench v0 baselines")
    scorecard_parser.add_argument("--date")
    scorecard_parser.add_argument(
        "--comparison-out", default="reports/real_baselines_v0.html"
    )
    scorecard_parser.add_argument(
        "--leaderboard-out", default="site/leaderboard/leaderboard.json"
    )
    scorecard_parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="emit explicitly non-comparable comparison artifacts for mismatched runs",
    )

    experiment_parser = subparsers.add_parser("experiment")
    experiment_parser.add_argument("--suite", default="maniskill_manipulation_v0")
    experiment_parser.add_argument("--tasks", nargs="+")
    experiment_parser.add_argument("--engine", default="maniskill")
    experiment_parser.add_argument(
        "--policies", nargs="+", default=["random", "scripted_oracle", "bc_policy"]
    )
    experiment_parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    experiment_parser.add_argument("--episodes", type=int, default=100)
    experiment_parser.add_argument(
        "--out", default="benchmark_results/maniskill_manipulation_v0"
    )
    experiment_parser.add_argument("--max-steps", type=int)
    experiment_parser.add_argument("--no-replay", action="store_true")
    experiment_parser.add_argument("--capture-replay", action="store_true")
    experiment_parser.add_argument("--expert-provider", default="none")
    experiment_parser.add_argument("--enable-recovery", action="store_true")
    experiment_parser.add_argument("--enable-verifier", action="store_true")
    experiment_parser.add_argument("--policy-action-horizon", type=int, default=1)
    experiment_parser.add_argument("--policy-execution-horizon", type=int, default=1)
    experiment_parser.add_argument(
        "--recovery-attribution-horizon",
        type=int,
        default=DEFAULT_RECOVERY_ATTRIBUTION_HORIZON,
    )
    experiment_parser.add_argument("--stressor-config")
    experiment_parser.add_argument("--benchmark-validity")
    _add_counterfactual_arguments(experiment_parser)
    _add_failure_monitor_arguments(experiment_parser)

    ablate_parser = subparsers.add_parser("ablate")
    ablate_parser.add_argument("--suite", required=True)
    ablate_parser.add_argument("--tasks", nargs="+")
    ablate_parser.add_argument("--engine", default="maniskill")
    ablate_parser.add_argument("--policy", default="random")
    ablate_parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ablate_parser.add_argument("--episodes", type=int, default=100)
    ablate_parser.add_argument("--out", default="benchmark_results/ablation")
    ablate_parser.add_argument("--max-steps", type=int)
    ablate_parser.add_argument("--expert-provider", default="none")
    ablate_parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "verifier", "recovery", "verifier_recovery"],
        choices=["base", "verifier", "recovery", "verifier_recovery"],
    )
    ablate_parser.add_argument("--no-replay", action="store_true")
    ablate_parser.add_argument("--capture-replay", action="store_true")
    ablate_parser.add_argument("--policy-action-horizon", type=int, default=1)
    ablate_parser.add_argument("--policy-execution-horizon", type=int, default=1)
    ablate_parser.add_argument(
        "--recovery-attribution-horizon",
        type=int,
        default=DEFAULT_RECOVERY_ATTRIBUTION_HORIZON,
    )
    ablate_parser.add_argument("--stressor-config")
    ablate_parser.add_argument("--benchmark-validity")
    _add_counterfactual_arguments(ablate_parser)
    _add_failure_monitor_arguments(ablate_parser)

    train_bc_parser = subparsers.add_parser("train-bc")
    train_bc_parser.add_argument("episodes", nargs="+")
    train_bc_parser.add_argument("--out", default="checkpoints/bc_policy.json")
    train_bc_parser.add_argument("--feature-dim", type=int, default=256)
    train_bc_parser.add_argument("--ridge", type=float, default=1e-3)
    train_bc_parser.add_argument(
        "--model", choices=["linear", "knn", "sequence-knn"], default="linear"
    )
    train_bc_parser.add_argument("--knn-k", type=int, default=1)
    train_bc_parser.add_argument("--action-horizon", type=int, default=8)

    train_task_bc_parser = subparsers.add_parser("train-task-bc")
    train_task_bc_parser.add_argument("sources", nargs="+")
    train_task_bc_parser.add_argument("--out-dir", default="checkpoints/bc_by_task")
    train_task_bc_parser.add_argument("--feature-dim", type=int, default=256)
    train_task_bc_parser.add_argument("--ridge", type=float, default=1e-3)
    train_task_bc_parser.add_argument(
        "--model", choices=["linear", "knn", "sequence-knn"], default="linear"
    )
    train_task_bc_parser.add_argument("--knn-k", type=int, default=1)
    train_task_bc_parser.add_argument("--action-horizon", type=int, default=8)
    train_task_bc_parser.add_argument("--include-failures", action="store_true")

    train_recovery_bc_parser = subparsers.add_parser("train-recovery-bc")
    train_recovery_bc_parser.add_argument("sources", nargs="+")
    train_recovery_bc_parser.add_argument(
        "--out", default="checkpoints/recovery_bc_policy.json"
    )
    train_recovery_bc_parser.add_argument("--by-task", action="store_true")
    train_recovery_bc_parser.add_argument(
        "--routing", choices=["auto", "global", "task"], default="auto"
    )
    train_recovery_bc_parser.add_argument("--out-dir", default="checkpoints/bc_by_task")
    train_recovery_bc_parser.add_argument("--merged-out")
    train_recovery_bc_parser.add_argument("--feature-dim", type=int, default=256)
    train_recovery_bc_parser.add_argument("--ridge", type=float, default=1e-3)
    train_recovery_bc_parser.add_argument("--min-steps", type=int, default=1)

    train_robomimic_parser = subparsers.add_parser("train-robomimic")
    train_robomimic_parser.add_argument("--config", required=True)
    train_robomimic_parser.add_argument("--name")
    train_robomimic_parser.add_argument("--debug", action="store_true")

    robomimic_config_parser = subparsers.add_parser("write-robomimic-config")
    robomimic_config_parser.add_argument("--data", required=True)
    robomimic_config_parser.add_argument("--out", required=True)
    robomimic_config_parser.add_argument(
        "--output-dir", default="checkpoints/robomimic"
    )
    robomimic_config_parser.add_argument("--name", default="nyssa_robomimic_bc_flat")
    robomimic_config_parser.add_argument("--epochs", type=int, default=50)
    robomimic_config_parser.add_argument("--batch-size", type=int, default=64)
    robomimic_config_parser.add_argument("--seed", type=int, default=1)
    robomimic_config_parser.add_argument("--learning-rate", type=float, default=1e-4)

    task_robomimic_export_parser = subparsers.add_parser("export-task-robomimic")
    task_robomimic_export_parser.add_argument("sources", nargs="+")
    task_robomimic_export_parser.add_argument(
        "--out-dir", default="datasets/robomimic_by_task"
    )
    task_robomimic_export_parser.add_argument("--config-dir")
    task_robomimic_export_parser.add_argument("--feature-dim", type=int, default=512)
    task_robomimic_export_parser.add_argument("--epochs", type=int, default=50)
    task_robomimic_export_parser.add_argument("--batch-size", type=int, default=64)
    task_robomimic_export_parser.add_argument("--seed", type=int, default=1)
    task_robomimic_export_parser.add_argument(
        "--learning-rate", type=float, default=1e-4
    )
    task_robomimic_export_parser.add_argument("--include-failures", action="store_true")

    import_maniskill_parser = subparsers.add_parser("import-maniskill-demos")
    import_maniskill_parser.add_argument("--input", required=True)
    import_maniskill_parser.add_argument("--out", required=True)

    collect_maniskill_parser = subparsers.add_parser("collect-maniskill-demos")
    collect_maniskill_parser.add_argument("--out", required=True)
    collect_maniskill_parser.add_argument("--raw-dir", required=True)
    collect_maniskill_parser.add_argument(
        "--env-ids", nargs="+", default=["PickCube-v1", "PushCube-v1", "StackCube-v1"]
    )
    collect_maniskill_parser.add_argument("--num-traj", type=int, default=100)
    collect_maniskill_parser.add_argument("--command-template")
    collect_maniskill_parser.add_argument("--continue-on-error", action="store_true")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("target")

    args = parser.parse_args(argv)

    if args.command == "list-suites":
        for suite in list_suites():
            print(suite)
        return 0

    if args.command == "list-tasks":
        for task in list_tasks():
            print(task)
        return 0

    if args.command == "list-engines":
        for engine in sorted(ENGINE_REGISTRY):
            print(f"{engine}\t{ENGINE_SUPPORT_TIER.get(engine, 'unknown')}")
        return 0

    if args.command == "list-policies":
        for policy in sorted(POLICY_REGISTRY):
            print(f"{policy}\t{POLICY_SUPPORT_TIER.get(policy, 'unknown')}")
        return 0

    if args.command == "list-stressors":
        for stressor in list_stressors():
            print(stressor)
        return 0

    if args.command == "validate-nep":
        data, migration = migrate_nep_data(load_nep_data(args.manifest))
        report, _ = validate_nep_manifest(data)
        path = write_nep_validation_report(report, args.out)
        print(f"nep_validation: {path}")
        print(f"valid: {report.valid}")
        print(f"claim_ready: {report.claim_ready}")
        if migration is not None:
            print(f"migration: {migration['source_format']} -> {migration['target_format']}")
        return 0 if report.valid else 3

    if args.command == "write-nep-schemas":
        paths = write_schemas(args.out)
        print(f"schemas: {len(paths)}")
        return 0

    if args.command == "conform-policy":
        suite = Suite.load(args.suite).filter_tasks([args.task])
        report = evaluate_policy_conformance(
            policy_path=args.policy,
            contract=load_policy_contract(args.policy_contract),
            suite=suite,
            engine_name=args.engine,
            out_dir=args.out,
            episodes=args.episodes,
            capture_replay=args.capture_replay,
        )
        paths = write_policy_conformance_report(report, args.out)
        print(f"policy_conformance: {paths['json']}")
        print(f"policy_conformance_html: {paths['html']}")
        print(f"conformant: {report['conformant']}")
        return 0 if report["conformant"] else 3

    if args.command == "write-policy-example":
        paths = write_policy_example(args.kind, args.out)
        for key, path in paths.items():
            print(f"{key}: {path}")
        return 0

    if args.command == "sim-real-study":
        spec_path = Path(args.spec)
        report = evaluate_sim_real_study(
            load_sim_real_study(spec_path), spec_root=spec_path.parent
        )
        paths = write_sim_real_report(report, args.out)
        print(f"sim_real_report: {paths['json']}")
        print(f"sim_real_html: {paths['html']}")
        print(f"status: {report['status']}")
        return {"complete": 0, "inconclusive": 2, "invalid": 3}[report["status"]]

    if args.command == "credibility-gate":
        spec_path = Path(args.spec)
        report = evaluate_credibility(
            load_credibility_spec(spec_path),
            spec_root=args.repo_root,
            source_root=args.repo_root,
        )
        paths = write_credibility_report(report, args.out)
        print(f"credibility_report: {paths['json']}")
        print(f"credibility_html: {paths['html']}")
        print(f"highest_completed_gate: {report['highest_completed_gate']}")
        return 0 if report["phase1_complete"] else 2

    if args.command == "audit-benchmark":
        spec = load_benchmark_validity_spec(args.spec)
        report = BenchmarkValidityEvaluator().evaluate(spec)
        path = write_benchmark_validity_report(report, args.out)
        print(f"benchmark_validity: {path}")
        print(f"status: {report.status}")
        return 0 if report.claim_ready else 2

    if args.command == "regression-gate":
        spec_path = Path(args.spec)
        spec = load_regression_study(spec_path)
        report = RegressionStudyEvaluator(
            spec, spec_root=spec_path.parent
        ).evaluate()
        paths = write_regression_report(report, args.out)
        print(f"regression_report: {paths['json']}")
        print(f"regression_html: {paths['html']}")
        print(f"decision: {report['decision']}")
        return int(report["exit_code"])

    if args.command == "regression-fingerprint":
        fingerprint = fingerprint_run(args.run)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(fingerprint, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"regression_fingerprint: {out}")
        return 0

    if args.command == "stress-search-init":
        study = StressSearchStudy(load_stress_search_spec(args.spec))
        path = write_stress_search_study(study, args.out)
        print(f"study: {path}")
        return 0

    if args.command == "stress-search-propose":
        study = load_stress_search_study(args.study)
        proposals = study.propose(args.count)
        study_path = write_stress_search_study(study, args.out)
        proposals_path = write_stress_proposals(
            proposals,
            args.proposals_out,
            search_space=study.spec.search_space,
        )
        print(f"study: {study_path}")
        print(f"proposals: {proposals_path}")
        print(f"proposal_count: {len(proposals)}")
        return 0

    if args.command == "stress-search-observe":
        study = load_stress_search_study(args.study)
        observations = load_stress_observations(args.observations)
        if args.confirmation:
            study.observe_confirmation(observations)
        else:
            study.observe(observations)
        path = write_stress_search_study(study, args.out)
        print(f"study: {path}")
        print(f"observation_count: {len(observations)}")
        return 0

    if args.command == "stress-search-ingest-run":
        study = load_stress_search_study(args.study)
        proposals = [
            *study.sampler.proposals,
            *study.confirmation_proposals,
        ]
        proposal = next(
            (
                item
                for item in proposals
                if item.proposal_id == args.proposal_id
            ),
            None,
        )
        if proposal is None:
            raise ValueError(f"unknown stress-search proposal: {args.proposal_id}")
        observation = observation_from_run(
            proposal,
            args.run,
            search_space=study.spec.search_space,
            success_threshold=study.spec.outcome_success_threshold,
        )
        if proposal.phase == "confirmation":
            study.observe_confirmation((observation,))
        else:
            study.observe((observation,))
        study_path = write_stress_search_study(study, args.out)
        observation_path = write_stress_observations(
            (observation,), args.observation_out
        )
        print(f"study: {study_path}")
        print(f"observation: {observation_path}")
        print(f"status: {observation.status}")
        return 0 if observation.status in {"success", "policy_failure"} else 2

    if args.command == "stress-search-confirm":
        study = load_stress_search_study(args.study)
        proposals = study.select_confirmation_conditions()
        study_path = write_stress_search_study(study, args.out)
        proposals_path = write_stress_proposals(
            proposals,
            args.proposals_out,
            search_space=study.spec.search_space,
        )
        print(f"study: {study_path}")
        print(f"confirmation_proposals: {proposals_path}")
        print(f"proposal_count: {len(proposals)}")
        return 0 if proposals else 2

    if args.command == "stress-search-report":
        studies = [load_stress_search_study(path) for path in args.studies]
        paths = write_stress_search_report(
            compare_stress_search_studies(studies), args.out
        )
        for label, path in paths.items():
            print(f"{label}: {path}")
        return 0

    if args.command == "validate-scenario":
        package = ScenarioPackage.load(args.scenario)
        validation = ScenarioPackageValidator().validate(
            package,
            expected_engine=args.engine,
            require_execution_assets=not args.metadata_only,
        )
        validation.raise_for_errors()
        print(f"scenario: {package.identity}")
        print(f"execution_ready: {validation.execution_ready}")
        for issue in validation.issues:
            print(f"{issue.severity}: {issue.code}: {issue.message}")
        return 0

    if args.command == "run-scenario":
        package = ScenarioPackage.load(args.scenario)
        validation = ScenarioPackageValidator().validate(
            package,
            expected_engine=package.engine.engine_name,
            require_execution_assets=True,
        )
        validation.raise_for_errors()
        if args.seed is not None and int(args.seed) != package.initial_state.run_seed:
            raise ValueError(
                "--seed cannot override an identity-bearing scenario run seed; "
                f"this package requires {package.initial_state.run_seed}"
            )
        run_seed = package.initial_state.run_seed
        stressor_config = package.stressor_config(
            severities=_parse_severity_overrides(args.severity),
            seed=run_seed,
        )
        context = scenario_execution_context(package, validation, stressor_config)
        task = TaskSpec.load(package.engine.task_id)
        suite = Suite(
            suite_id=f"scenario:{package.scenario_id}@{package.scenario_version}",
            description=package.description,
            tasks=(task,),
            source_path=package.source_path,
        )
        runner = PolicyRunner(
            policy=args.policy,
            engine=package.engine.engine_name,
            episodes=args.episodes,
            seed=run_seed,
            out=args.out,
            max_steps=package.evaluation.horizon_steps,
            capture_replay=_capture_replay_default(
                package.engine.engine_name,
                args.no_replay,
                args.capture_replay,
            ),
            expert_provider=args.expert_provider,
            enable_recovery=args.enable_recovery,
            enable_verifier=args.enable_verifier,
            policy_action_horizon=args.policy_action_horizon,
            policy_execution_horizon=args.policy_execution_horizon,
            recovery_attribution_horizon=args.recovery_attribution_horizon,
            counterfactual_repeats=args.counterfactual_repeats,
            counterfactual_horizon=args.counterfactual_horizon,
            counterfactual_oracle=args.counterfactual_oracle,
            counterfactual_max_branch_points=args.counterfactual_max_branch_points,
            stressor_config=stressor_config,
            scenario_context=context,
            benchmark_validity=args.benchmark_validity,
            failure_monitors=args.failure_monitor,
            enable_monitor_intervention=args.enable_monitor_intervention,
        )
        report = runner.evaluate(suite)
        print(f"scenario: {package.identity}")
        print(f"report: {Path(args.out) / 'report.html'}")
        print(f"success_rate: {report.summary.get('success_rate', 0.0):.3f}")
        return 0

    if args.command in {"validate-real-evidence", "import-real-evidence"}:
        package = RealEvidencePackage.load(args.package)
        validation = RealEvidenceValidator().validate(
            package, require_artifacts=not args.metadata_only
        )
        validation.raise_for_errors()
        print(f"real_evidence: {package.identity}")
        print(f"evidence_ready: {validation.evidence_ready}")
        print(f"calibration_ready: {validation.calibration_ready}")
        print(f"comparison_ready: {validation.comparison_ready}")
        print(f"claim_ready: {validation.claim_ready}")
        for issue in validation.issues:
            print(f"{issue.severity}: {issue.code}: {issue.message}")
        if args.command == "import-real-evidence":
            paths = write_real_evidence_artifacts(package, validation, args.out)
            for label, path in paths.items():
                print(f"{label}: {path}")
        return 0

    if args.command == "run":
        suite = _load_suite(args)
        runner = PolicyRunner(
            policy=args.policy,
            engine=args.engine,
            episodes=args.episodes,
            seed=args.seed,
            out=args.out,
            capture_replay=_capture_replay_default(
                args.engine, args.no_replay, args.capture_replay
            ),
            expert_provider=args.expert_provider,
            enable_recovery=args.enable_recovery,
            enable_verifier=args.enable_verifier,
            policy_action_horizon=args.policy_action_horizon,
            policy_execution_horizon=args.policy_execution_horizon,
            recovery_attribution_horizon=args.recovery_attribution_horizon,
            counterfactual_repeats=args.counterfactual_repeats,
            counterfactual_horizon=args.counterfactual_horizon,
            counterfactual_oracle=args.counterfactual_oracle,
            counterfactual_max_branch_points=args.counterfactual_max_branch_points,
            stressor_config=args.stressor_config,
            benchmark_validity=args.benchmark_validity,
            failure_monitors=args.failure_monitor,
            enable_monitor_intervention=args.enable_monitor_intervention,
        )
        report = runner.evaluate(suite)
        print(f"report: {Path(args.out) / 'report.html'}")
        print(f"success_rate: {report.summary.get('success_rate', 0.0):.3f}")
        return 0

    if args.command == "report":
        run_dir = Path(args.run)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Run metrics not found: {metrics_path}")
        raw_summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(raw_summary, dict):
            raise ValueError(f"Run metrics must contain a JSON object: {metrics_path}")
        summary = migrate_metric_summary(raw_summary)
        metadata = _load_run_metadata(run_dir)
        report = Report(
            suite_id=str(metadata.get("suite_id", summary.get("suite_id", "unknown"))),
            policy=str(metadata.get("policy_name", summary.get("policy", "unknown"))),
            engine=str(metadata.get("engine_name", summary.get("engine", "unknown"))),
            summary=summary,
            run_dir=run_dir,
        )
        out = report.save(run_dir / "report.html")
        print(f"report: {out}")
        return 0

    if args.command == "compare-failure-monitors":
        manifest = Path(args.manifest)
        if manifest.is_dir():
            manifest = manifest / "failure_monitor_predictions.json"
        _, contracts, records = load_monitor_manifest(manifest)
        comparison = compare_monitor_records(
            records,
            contracts,
            args.monitor_a,
            args.monitor_b,
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(comparison, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"comparison: {out}")
        return 0

    if args.command == "export-learning-evidence":
        config = LearningExportConfig(
            benchmark_id=args.benchmark_id,
            split=ExportSplit(
                args.split_id,
                args.split_partition,
                args.split_sha256,
            ),
            policy_families=_parse_policy_families(args.policy_family),
            licenses=tuple(args.licenses),
            privacy_level=args.privacy_level,
            privacy_restrictions=tuple(args.privacy_restriction),
            include_successes=not args.failures_only,
            boundary_studies=tuple(Path(path) for path in args.boundary_study),
            max_inline_observation_bytes=args.max_inline_observation_bytes,
        )
        package = export_learning_evidence(args.runs, args.out, config=config)
        if args.verify_external_artifacts:
            package = load_learning_evidence(
                package.root, verify_external_artifacts=True
            )
        print(f"manifest: {package.root / 'manifest.json'}")
        print(f"episodes: {package.manifest.episode_count}")
        print(f"evaluation_reuse_policy: {package.manifest.evaluation_reuse_policy}")
        return 0

    if args.command == "export":
        run_dir = Path(args.run)
        episodes = _load_episodes(run_dir)
        out_arg = Path(args.out) if args.out else None
        if args.format == "lerobot":
            out = export_lerobot(episodes, out_arg or run_dir / "lerobot")
        elif args.format == "json":
            out = export_json(episodes, out_arg or run_dir / "episodes.export.json")
        elif args.format == "jsonl":
            out = export_jsonl(episodes, out_arg or run_dir / "episodes.export.jsonl")
        elif args.format == "hdf5":
            out = export_hdf5(episodes, out_arg or run_dir / "episodes.hdf5")
        elif args.format == "parquet":
            out = export_parquet(episodes, out_arg or run_dir / "episodes.parquet")
        elif args.format == "robomimic":
            out = export_robomimic_hdf5(
                episodes,
                out_arg or run_dir / "robomimic.hdf5",
                feature_dim=args.feature_dim,
            )
        else:
            raise ValueError(f"Unsupported export format: {args.format}")
        print(f"exported: {out}")
        return 0

    if args.command == "compare":
        comparison = compare_runs(args.runs, allow_incompatible=args.allow_incompatible)
        out = save_comparison_report(comparison, args.out)
        print(f"comparison: {out}")
        return 0

    if args.command == "robustness-report":
        summary = load_robustness_sweep(
            args.runs,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        paths = save_robustness_report(summary, args.out)
        for label, path in paths.items():
            print(f"{label}: {path}")
        return 0

    if args.command == "leaderboard":
        comparison = compare_runs(args.runs, allow_incompatible=args.allow_incompatible)
        out = save_leaderboard(comparison, args.out)
        print(f"leaderboard: {out}")
        return 0

    if args.command == "scorecard":
        paths = write_scorecard(
            args.runs,
            out=args.out,
            benchmark=args.benchmark,
            scorecard_date=args.date,
            comparison_report=args.comparison_out,
            leaderboard=args.leaderboard_out,
            allow_incompatible=args.allow_incompatible,
        )
        for label, path in paths.items():
            print(f"{label}: {path}")
        return 0

    if args.command == "experiment":
        paths = _run_experiment(args)
        for label, path in paths.items():
            print(f"{label}: {path}")
        return 0

    if args.command == "ablate":
        paths = _run_ablation(args)
        for label, path in paths.items():
            print(f"{label}: {path}")
        return 0

    if args.command == "train-bc":
        out = _train_bc_from_episode_files(
            args.episodes,
            args.out,
            feature_dim=args.feature_dim,
            ridge=args.ridge,
            model=args.model,
            knn_k=args.knn_k,
            action_horizon=args.action_horizon,
        )
        print(f"bc_checkpoint: {out}")
        return 0

    if args.command == "train-task-bc":
        checkpoints = train_task_bc(
            args.sources,
            args.out_dir,
            feature_dim=args.feature_dim,
            ridge=args.ridge,
            model=args.model,
            k=args.knn_k,
            action_horizon=args.action_horizon,
            success_only=not args.include_failures,
        )
        for label, path in checkpoints.items():
            print(f"bc_checkpoint[{label}]: {path}")
        return 0

    if args.command == "train-recovery-bc":
        result = train_recovery_bc(
            args.sources,
            out=args.out,
            by_task=args.by_task,
            routing=args.routing,
            out_dir=args.out_dir,
            merged_out=args.merged_out,
            feature_dim=args.feature_dim,
            ridge=args.ridge,
            min_steps=args.min_steps,
        )
        print(f"recovery_sources: {len(result.source_paths)}")
        print(f"recovery_episodes: {result.episodes}")
        print(f"recovery_steps: {result.steps}")
        print(f"recovery_routing: {result.routing}")
        for task_id, action_size in sorted(result.action_sizes.items()):
            print(f"recovery_action_size[{task_id}]: {action_size}")
        if result.merged_path:
            print(f"merged_recovery_episodes: {result.merged_path}")
        for label, path in result.checkpoints.items():
            print(f"bc_checkpoint[{label}]: {path}")
        return 0

    if args.command == "train-robomimic":
        train_robomimic(args.config, name=args.name, debug=args.debug)
        print("robomimic_training: complete")
        return 0

    if args.command == "write-robomimic-config":
        out = write_robomimic_bc_config(
            data=args.data,
            out=args.out,
            output_dir=args.output_dir,
            name=args.name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )
        print(f"robomimic_config: {out}")
        return 0

    if args.command == "export-task-robomimic":
        artifacts = export_task_robomimic(
            args.sources,
            out_dir=args.out_dir,
            config_dir=args.config_dir,
            feature_dim=args.feature_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            learning_rate=args.learning_rate,
            success_only=not args.include_failures,
        )
        for task, paths in artifacts.items():
            for label, path in paths.items():
                print(f"{label}[{task}]: {path}")
        return 0

    if args.command == "import-maniskill-demos":
        paths = import_maniskill_demos(args.input, args.out)
        for label, path in paths.items():
            print(f"{label}: {path}")
        return 0

    if args.command == "collect-maniskill-demos":
        paths = collect_maniskill_demos(
            out=args.out,
            raw_dir=args.raw_dir,
            env_ids=args.env_ids,
            num_traj=args.num_traj,
            command_template=args.command_template,
            continue_on_error=args.continue_on_error,
        )
        for label, path in paths.items():
            print(f"{label}: {path}")
        return 0

    if args.command == "validate":
        _validate_target(args.target)
        print(f"valid: {args.target}")
        return 0

    return 1


def _run_experiment(args: argparse.Namespace) -> dict[str, Path]:
    suite = _load_suite(args)
    out_dir = Path(args.out)
    cells = policy_seed_cells(
        policies=args.policies,
        seeds=args.seeds,
        out_dir=out_dir,
        enable_verifier=args.enable_verifier,
        enable_recovery=args.enable_recovery,
    )
    run_dirs = ExperimentRunner(
        lambda cell: _matrix_policy_runner(args, cell)
    ).execute(suite, cells)

    comparison_path = out_dir / "comparison.html"
    leaderboard_path = out_dir / "leaderboard.json"
    scorecard_path = out_dir / "scorecard.json"
    comparison = compare_runs(run_dirs)
    replay_validation = validate_result_pack_replays(run_dirs)
    save_comparison_report(comparison, comparison_path)
    save_leaderboard(comparison, leaderboard_path)
    write_scorecard(
        run_dirs,
        out=scorecard_path,
        benchmark=f"{args.suite} baseline matrix",
        comparison_report=comparison_path,
        leaderboard=leaderboard_path,
        replay_validation=replay_validation,
    )
    results_path = write_results_markdown(
        out_dir=out_dir,
        suite_id=args.suite,
        engine=args.engine,
        policies=list(args.policies),
        seeds=list(args.seeds),
        episodes_per_task=args.episodes,
        run_dirs=run_dirs,
        comparison_report=comparison_path,
        leaderboard=leaderboard_path,
        scorecard=scorecard_path,
        replay_validation=replay_validation,
    )
    manifest_path = write_experiment_manifest(
        out_dir=out_dir,
        suite_id=args.suite,
        engine=args.engine,
        policies=list(args.policies),
        seeds=list(args.seeds),
        episodes_per_task=args.episodes,
        run_dirs=run_dirs,
        artifacts={
            "comparison_report": comparison_path,
            "leaderboard": leaderboard_path,
            "scorecard": scorecard_path,
            "results": results_path,
        },
        replay_validation=replay_validation,
    )
    return {
        "manifest": manifest_path,
        "results": results_path,
        "comparison_report": comparison_path,
        "leaderboard": leaderboard_path,
        "scorecard": scorecard_path,
    }


def _run_ablation(args: argparse.Namespace) -> dict[str, Path]:
    suite = _load_suite(args)
    out_dir = Path(args.out)
    variants = list(args.variants)
    cells = ablation_cells(
        policy=args.policy,
        variants=variants,
        seeds=args.seeds,
        out_dir=out_dir,
    )
    run_dirs = ExperimentRunner(
        lambda cell: _matrix_policy_runner(args, cell)
    ).execute(suite, cells)

    comparison_path = out_dir / "comparison.html"
    leaderboard_path = out_dir / "leaderboard.json"
    scorecard_path = out_dir / "scorecard.json"
    comparison = compare_runs(run_dirs)
    replay_validation = validate_result_pack_replays(run_dirs)
    save_comparison_report(comparison, comparison_path)
    save_leaderboard(comparison, leaderboard_path)
    write_scorecard(
        run_dirs,
        out=scorecard_path,
        benchmark=f"{args.suite} ablation matrix",
        comparison_report=comparison_path,
        leaderboard=leaderboard_path,
        replay_validation=replay_validation,
    )
    results_path = write_results_markdown(
        out_dir=out_dir,
        suite_id=args.suite,
        engine=args.engine,
        policies=[f"{args.policy}:{variant}" for variant in variants],
        seeds=list(args.seeds),
        episodes_per_task=args.episodes,
        run_dirs=run_dirs,
        comparison_report=comparison_path,
        leaderboard=leaderboard_path,
        scorecard=scorecard_path,
        replay_validation=replay_validation,
    )
    manifest_path = write_experiment_manifest(
        out_dir=out_dir,
        suite_id=args.suite,
        engine=args.engine,
        policies=[args.policy],
        seeds=list(args.seeds),
        episodes_per_task=args.episodes,
        run_dirs=run_dirs,
        artifacts={
            "comparison_report": comparison_path,
            "leaderboard": leaderboard_path,
            "scorecard": scorecard_path,
            "results": results_path,
        },
        replay_validation=replay_validation,
    )
    return {
        "manifest": manifest_path,
        "results": results_path,
        "comparison_report": comparison_path,
        "leaderboard": leaderboard_path,
        "scorecard": scorecard_path,
    }


def _matrix_policy_runner(
    args: argparse.Namespace, cell: ExperimentCell
) -> PolicyRunner:
    return PolicyRunner(
        policy=cell.policy,
        engine=args.engine,
        episodes=args.episodes,
        seed=cell.seed,
        out=cell.run_dir,
        max_steps=args.max_steps,
        capture_replay=_capture_replay_default(
            args.engine, args.no_replay, args.capture_replay
        ),
        expert_provider=args.expert_provider,
        enable_recovery=cell.enable_recovery,
        enable_verifier=cell.enable_verifier,
        policy_action_horizon=args.policy_action_horizon,
        policy_execution_horizon=args.policy_execution_horizon,
        recovery_attribution_horizon=args.recovery_attribution_horizon,
        counterfactual_repeats=(
            args.counterfactual_repeats if cell.enable_recovery else 0
        ),
        counterfactual_horizon=args.counterfactual_horizon,
        counterfactual_oracle=(
            args.counterfactual_oracle if cell.enable_recovery else False
        ),
        counterfactual_max_branch_points=args.counterfactual_max_branch_points,
        stressor_config=args.stressor_config,
        benchmark_validity=args.benchmark_validity,
        failure_monitors=args.failure_monitor,
        enable_monitor_intervention=args.enable_monitor_intervention,
    )


def _train_bc_from_episode_files(
    episodes_paths: list[str],
    out: str | Path,
    *,
    feature_dim: int,
    ridge: float,
    model: str,
    knn_k: int,
    action_horizon: int,
) -> Path:
    if len(episodes_paths) == 1:
        return _train_single_bc(
            episodes_paths[0],
            out,
            feature_dim=feature_dim,
            ridge=ridge,
            model=model,
            knn_k=knn_k,
            action_horizon=action_horizon,
        )

    import json
    import tempfile

    merged = []
    for path in episodes_paths:
        merged.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", encoding="utf-8", delete=False
    ) as handle:
        json.dump(merged, handle)
        merged_path = Path(handle.name)
    try:
        return _train_single_bc(
            merged_path,
            out,
            feature_dim=feature_dim,
            ridge=ridge,
            model=model,
            knn_k=knn_k,
            action_horizon=action_horizon,
        )
    finally:
        try:
            merged_path.unlink()
        except OSError:
            pass


def _train_single_bc(
    episodes_path: str | Path,
    out: str | Path,
    *,
    feature_dim: int,
    ridge: float,
    model: str,
    knn_k: int,
    action_horizon: int,
) -> Path:
    if model == "linear":
        return train_linear_bc(episodes_path, out, feature_dim=feature_dim, ridge=ridge)
    if model == "knn":
        return train_knn_bc(episodes_path, out, feature_dim=feature_dim, k=knn_k)
    if model == "sequence-knn":
        return train_sequence_knn_bc(
            episodes_path,
            out,
            feature_dim=feature_dim,
            k=knn_k,
            action_horizon=action_horizon,
        )
    raise ValueError(f"Unsupported BC model: {model}")


def _load_suite(args: argparse.Namespace) -> Suite:
    suite = Suite.load(args.suite)
    return suite.filter_tasks(getattr(args, "tasks", None))


def _validate_target(target: str) -> None:
    path = Path(target)
    if path.exists():
        if path.is_dir():
            if (path / "scenario.yaml").is_file():
                package = ScenarioPackage.load(path)
                ScenarioPackageValidator().validate(package).raise_for_errors()
                return
            if (path / "evidence.yaml").is_file():
                package = RealEvidencePackage.load(path)
                RealEvidenceValidator().validate(package).raise_for_errors()
                return
            manifest_path = path / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    isinstance(manifest, dict)
                    and manifest.get("format") == LEARNING_EXPORT_MANIFEST_FORMAT
                ):
                    load_learning_evidence(path)
                    return
            raise ValueError(f"Directory contains no recognized manifest: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("format") == SCENARIO_PACKAGE_FORMAT:
            package = ScenarioPackage.load(path)
            ScenarioPackageValidator().validate(package).raise_for_errors()
            return
        if data.get("format") == REAL_EVIDENCE_PACKAGE_FORMAT:
            package = RealEvidencePackage.load(path)
            RealEvidenceValidator().validate(package).raise_for_errors()
            return
        if data.get("format") == STRESSOR_CONFIG_FORMAT:
            StressorConfig.from_dict(data)
            return
        if data.get("format") == COUNTERFACTUAL_RECOVERY_MANIFEST_FORMAT:
            load_counterfactual_recovery_manifest(path)
            return
        if data.get("format") == BENCHMARK_VALIDITY_SPEC_FORMAT:
            load_benchmark_validity_spec(path)
            return
        if data.get("format") == BENCHMARK_VALIDITY_REPORT_FORMAT:
            load_benchmark_validity_report(path)
            return
        if data.get("format") == STRESS_SEARCH_SPEC_FORMAT:
            load_stress_search_spec(path)
            return
        if data.get("format") == STRESS_SEARCH_STUDY_FORMAT:
            load_stress_search_study(path)
            return
        if data.get("format") == LEARNING_EXPORT_MANIFEST_FORMAT:
            load_learning_evidence(path.parent)
            return
        if "tasks" in data:
            Suite.load(path)
            return
        TaskSpec.load(path)
        return

    try:
        Suite.load(target)
    except FileNotFoundError:
        TaskSpec.load(target)


def _parse_policy_families(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        policy_id, separator, family = value.partition("=")
        policy_id = policy_id.strip()
        family = family.strip()
        if not separator or not policy_id or not family:
            raise ValueError(
                f"Invalid --policy-family value {value!r}; expected POLICY_ID=FAMILY"
            )
        if policy_id in result:
            raise ValueError(f"Duplicate policy-family mapping for {policy_id!r}")
        result[policy_id] = family
    return result


def _parse_severity_overrides(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        stressor_id, separator, raw_severity = value.partition("=")
        if not separator or not stressor_id.strip() or not raw_severity.strip():
            raise ValueError(
                f"Invalid --severity value '{value}'; expected STRESSOR_ID=SEVERITY"
            )
        if stressor_id in result:
            raise ValueError(f"Duplicate --severity override for '{stressor_id}'")
        try:
            result[stressor_id] = float(raw_severity)
        except ValueError as exc:
            raise ValueError(
                f"Invalid severity for '{stressor_id}': {raw_severity}"
            ) from exc
    return result


def _capture_replay_default(engine: str, no_replay: bool, capture_replay: bool) -> bool:
    if no_replay:
        return False
    if capture_replay:
        return True
    return engine in PUBLIC_CLAIM_ENGINES


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    run_path = run_dir / "run.yaml"
    if not run_path.exists():
        return {}
    data = yaml.safe_load(run_path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _load_episodes(run_dir: Path):
    from nyssa_bench.core.episode import EpisodeResult, StepRecord
    from nyssa_bench.failures import failure_ledger_from_episode_dict

    episodes_path = run_dir / "episodes.json"
    data = json.loads(episodes_path.read_text(encoding="utf-8"))
    run_metadata = _load_run_metadata(run_dir)
    episodes = []
    for item in data:
        steps = [
            StepRecord(
                observation=step.get("observation", {}),
                action=step.get("action"),
                reward=float(step.get("reward", 0.0)),
                terminated=bool(step.get("terminated", False)),
                truncated=bool(step.get("truncated", False)),
                info=step.get("info", {}),
            )
            for step in item.get("steps", [])
        ]
        episodes.append(
            EpisodeResult(
                task_id=item["task_id"],
                episode_index=item["episode_index"],
                seed=item["seed"],
                success=item["success"],
                failure_label=item["failure_label"],
                metrics=item["metrics"],
                failure_label_source=item.get("failure_label_source"),
                steps=steps,
                replay_path=item.get("replay_path"),
                failure_clip_path=item.get("failure_clip_path"),
                stressor_context=item.get("stressor_context", {}),
                failure_detector_context=item.get("failure_detector_context", {}),
                failure_ledger=failure_ledger_from_episode_dict(
                    item,
                    engine_name=str(run_metadata.get("engine_name", "unknown")),
                ),
            )
        )
    return episodes


if __name__ == "__main__":
    raise SystemExit(main())
