from nyssa_bench.reports.result_pack import _publication_caveats


def test_publication_caveats_flag_inactive_verifier_and_recovery():
    summaries = [
        _summary("base", success_rate=0.1),
        _summary("verifier", success_rate=0.1),
        _summary("recovery", success_rate=0.1),
        _summary("verifier_recovery", success_rate=0.1),
    ]

    caveats = _publication_caveats(
        summaries,
        video_count=4,
        policies=[
            "task_robomimic:base",
            "task_robomimic:verifier",
            "task_robomimic:recovery",
            "task_robomimic:verifier_recovery",
        ],
    )

    assert "`verifier` enabled the verifier but rejected no actions" in caveats
    assert "`verifier_recovery` enabled the verifier but rejected no actions" in caveats
    assert "`recovery` enabled recovery but attempted none" in caveats
    assert "`verifier_recovery` enabled recovery but attempted none" in caveats


def test_publication_caveats_do_not_flag_inactive_mechanisms_for_solved_policy():
    summaries = [
        _summary("verifier", success_rate=1.0),
        _summary("recovery", success_rate=1.0),
    ]

    caveats = _publication_caveats(
        summaries,
        video_count=2,
        policies=["task_robomimic:verifier", "task_robomimic:recovery"],
    )

    assert "inactive verifier" not in caveats
    assert "inactive recovery" not in caveats


def _summary(policy: str, *, success_rate: float) -> dict:
    return {
        "policy": policy,
        "episodes": 20,
        "success_rate": success_rate,
        "metrics": {
            "expert_intervention_rate": 0.0,
            "recovery_attempt_count": 0.0,
            "recovery_success_rate": 0.0,
            "verifier_rejection_rate": 0.0,
        },
    }
