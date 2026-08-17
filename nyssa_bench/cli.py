from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from nyssa_bench.core.registry import (
    ENGINE_REGISTRY,
    ENGINE_SUPPORT_TIER,
    POLICY_REGISTRY,
    POLICY_SUPPORT_TIER,
)
from nyssa_bench.core.suite import Suite, list_suites
from nyssa_bench.core.task import TaskSpec, list_tasks
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
from nyssa_bench.runner import DEFAULT_RECOVERY_ATTRIBUTION_HORIZON, PolicyRunner
from nyssa_bench.metrics.run_claims import PUBLIC_CLAIM_ENGINES
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nyssa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-suites")
    subparsers.add_parser("list-tasks")
    subparsers.add_parser("list-engines")
    subparsers.add_parser("list-policies")
    subparsers.add_parser("list-stressors")

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

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("run")

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--run", required=True)
    export_parser.add_argument(
        "--format",
        choices=["json", "jsonl", "lerobot", "hdf5", "parquet", "robomimic"],
        default="lerobot",
    )
    export_parser.add_argument("--out")
    export_parser.add_argument("--feature-dim", type=int, default=256)

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
            stressor_config=args.stressor_config,
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
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
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
    run_dirs: list[Path] = []
    for policy in args.policies:
        for seed in args.seeds:
            run_dir = out_dir / policy / f"seed_{seed}"
            runner = PolicyRunner(
                policy=policy,
                engine=args.engine,
                episodes=args.episodes,
                seed=seed,
                out=run_dir,
                max_steps=args.max_steps,
                capture_replay=_capture_replay_default(
                    args.engine, args.no_replay, args.capture_replay
                ),
                expert_provider=args.expert_provider,
                enable_recovery=args.enable_recovery,
                enable_verifier=args.enable_verifier,
                policy_action_horizon=args.policy_action_horizon,
                policy_execution_horizon=args.policy_execution_horizon,
                recovery_attribution_horizon=args.recovery_attribution_horizon,
                stressor_config=args.stressor_config,
            )
            runner.evaluate(suite)
            run_dirs.append(run_dir)

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
    run_dirs: list[Path] = []
    variants = list(args.variants)
    for variant in variants:
        enable_verifier = variant in {"verifier", "verifier_recovery"}
        enable_recovery = variant in {"recovery", "verifier_recovery"}
        for seed in args.seeds:
            run_dir = out_dir / variant / f"seed_{seed}"
            runner = PolicyRunner(
                policy=args.policy,
                engine=args.engine,
                episodes=args.episodes,
                seed=seed,
                out=run_dir,
                max_steps=args.max_steps,
                capture_replay=_capture_replay_default(
                    args.engine, args.no_replay, args.capture_replay
                ),
                expert_provider=args.expert_provider,
                enable_recovery=enable_recovery,
                enable_verifier=enable_verifier,
                policy_action_horizon=args.policy_action_horizon,
                policy_execution_horizon=args.policy_execution_horizon,
                recovery_attribution_horizon=args.recovery_attribution_horizon,
                stressor_config=args.stressor_config,
            )
            runner.evaluate(suite)
            run_dirs.append(run_dir)

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
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("format") == STRESSOR_CONFIG_FORMAT:
            StressorConfig.from_dict(data)
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

    episodes_path = run_dir / "episodes.json"
    data = json.loads(episodes_path.read_text(encoding="utf-8"))
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
            )
        )
    return episodes


if __name__ == "__main__":
    raise SystemExit(main())
