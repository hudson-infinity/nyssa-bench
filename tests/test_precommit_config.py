from pathlib import Path

import yaml


def test_precommit_config_covers_fast_and_pre_push_checks():
    payload = yaml.safe_load(
        Path(".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    assert payload["default_install_hook_types"] == ["pre-commit", "pre-push"]

    hooks = {
        hook["id"]: hook
        for repository in payload["repos"]
        for hook in repository["hooks"]
    }
    assert {
        "check-added-large-files",
        "check-json",
        "check-merge-conflict",
        "check-toml",
        "check-yaml",
        "detect-private-key",
        "ruff-check",
        "validate-nyssa-configs",
        "release-checklist",
        "pytest",
    } <= set(hooks)
    assert hooks["pytest"]["stages"] == ["pre-push"]
    assert hooks["pytest"]["pass_filenames"] is False
    assert hooks["validate-nyssa-configs"]["pass_filenames"] is False


def test_ci_executes_commit_stage_hooks():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "pre-commit run --all-files --show-diff-on-failure" in workflow
