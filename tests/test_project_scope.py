from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_canonical_scope_covers_owned_and_external_responsibilities():
    scope = _read("docs/project_scope.md")

    for owned in (
        "versioned task, engine, policy, stressor",
        "temporal failure detection",
        "counterfactual recovery",
        "run validity, benchmark validity",
        "reproducibility metadata",
    ):
        assert owned in scope

    for external in (
        "world generation",
        "reconstruction of real scenes",
        "interpretability",
        "general policy training",
        "hosted evaluation services",
        "organization-wide research strategy",
    ):
        assert external in scope

    for boundary in (
        "Generated-world system",
        "Real-to-sim system",
        "Policy project",
        "Failure-driven learning system",
        "Hosted product",
    ):
        assert boundary in scope


def test_public_entry_points_link_to_project_scope():
    assert "[project scope](docs/project_scope.md)" in _read("README.md")
    assert "[Project scope](project_scope.md)" in _read("docs/roadmap.md")
    assert "[Project scope](project_scope.md)" in _read("docs/index.md")


def test_feature_template_requires_a_repository_boundary_decision():
    template = _read(".github/ISSUE_TEMPLATE/feature_request.md")

    assert "project_scope.md" in template
    assert "Responsibilities that remain outside NyssaBench" in template
    assert "This belongs in NyssaBench" in template
    assert "This belongs in a separate project" in template


def test_roadmaps_do_not_assign_external_programs_to_nyssabench():
    roadmap = _read("docs/roadmap.md")
    real_evidence = _read("docs/real_to_sim_roadmap.md")

    assert "hosted benchmark leaderboard export" not in roadmap
    assert "Real-to-sim reconstruction is a separate research program" in real_evidence
    assert "does not reconstruct scenes" in roadmap
