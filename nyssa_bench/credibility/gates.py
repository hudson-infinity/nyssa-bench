from __future__ import annotations

from nyssa_bench.credibility.protocol import GateCheckDefinition, GateDefinition


GATE_DEFINITIONS = (
    GateDefinition(
        gate_id="A",
        name="Measurement Core",
        required_checks=(
            GateCheckDefinition(
                check_id="claim_matrix_integrity",
                description="The content-pinned claim matrix passes its validator.",
                issue_ids=(23,),
            ),
            GateCheckDefinition(
                check_id="nyssa_evaluation_protocol",
                description="Versioned NEP contracts and conformance tests exist.",
                issue_ids=(21,),
            ),
            GateCheckDefinition(
                check_id="executable_stressor_measurement",
                description="Stressors execute and record applied-state evidence.",
                issue_ids=(13,),
            ),
            GateCheckDefinition(
                check_id="temporal_failure_evidence",
                description="Failure evidence is temporally represented and tested.",
                issue_ids=(14,),
            ),
            GateCheckDefinition(
                check_id="counterfactual_recovery_measurement",
                description="Matched counterfactual recovery evaluation is implemented.",
                issue_ids=(15,),
            ),
            GateCheckDefinition(
                check_id="benchmark_validity_audits",
                description="RunValidity and BenchmarkValidity are separate executable gates.",
                issue_ids=(18,),
            ),
            GateCheckDefinition(
                check_id="metric_vector_reporting",
                description="Reports use a validated metric vector without a scalar composite.",
                issue_ids=(22,),
            ),
            GateCheckDefinition(
                check_id="current_public_positioning",
                description="Public wording is selected from authorized evidence.",
                issue_ids=(23,),
            ),
        ),
    ),
    GateDefinition(
        gate_id="B",
        name="Reference Benchmark Evidence",
        required_checks=(
            GateCheckDefinition(
                check_id="gate_a_dependency",
                description="Gate A passes before reference evidence is promoted.",
                issue_ids=(13, 14, 15, 18, 21, 22, 23),
            ),
            GateCheckDefinition(
                check_id="reference_benchmark",
                description="A compact benchmark has protected split lineage and valid results.",
                issue_ids=(17, 18),
                evidence_categories=("reference_benchmark",),
            ),
            GateCheckDefinition(
                check_id="oracle_controls",
                description="The reference benchmark includes an identified oracle control.",
                issue_ids=(17,),
                evidence_categories=("reference_benchmark",),
            ),
            GateCheckDefinition(
                check_id="two_learned_policy_families",
                description="Two materially different learned policy families pass validation.",
                issue_ids=(16, 18),
                evidence_categories=("learned_policy_track",),
            ),
            GateCheckDefinition(
                check_id="paired_clean_shifted",
                description="Clean and shifted conditions have complete paired coverage.",
                issue_ids=(13, 16, 17, 18),
                evidence_categories=("paired_clean_shifted",),
            ),
            GateCheckDefinition(
                check_id="adequate_power",
                description="Paired estimates pass prespecified statistical precision checks.",
                issue_ids=(17, 18, 22),
                evidence_categories=("paired_clean_shifted",),
            ),
            GateCheckDefinition(
                check_id="benchmark_validity",
                description="Reference evidence passes executable BenchmarkValidity audits.",
                issue_ids=(18,),
                evidence_categories=("benchmark_validity",),
            ),
            GateCheckDefinition(
                check_id="mujoco_ci",
                description="Installed-wheel MuJoCo execution passes in CI.",
                issue_ids=(19,),
                evidence_categories=("simulator_ci",),
            ),
            GateCheckDefinition(
                check_id="maniskill_ci",
                description="GPU ManiSkill execution and replay capture pass in CI.",
                issue_ids=(19,),
                evidence_categories=("simulator_ci",),
            ),
        ),
    ),
    GateDefinition(
        gate_id="C",
        name="Predictive Validity",
        required_checks=(
            GateCheckDefinition(
                check_id="gate_b_dependency",
                description="Gate B passes before predictive evidence is promoted.",
                issue_ids=(16, 17, 18, 19),
            ),
            GateCheckDefinition(
                check_id="prespecified_hardware_calibration",
                description="Claim-ready hardware evidence follows a prespecified study.",
                issue_ids=(20,),
                evidence_categories=("hardware_calibration",),
            ),
            GateCheckDefinition(
                check_id="held_out_incremental_analysis",
                description="A held-out analysis compares clean success with failure features.",
                issue_ids=(20, 22),
                evidence_categories=("sim_real_predictive_result",),
            ),
            GateCheckDefinition(
                check_id="predictive_power",
                description="The predictive estimate passes prespecified precision checks.",
                issue_ids=(18, 20),
                evidence_categories=("sim_real_predictive_result",),
            ),
            GateCheckDefinition(
                check_id="positive_incremental_result",
                description="Failure features add positive held-out predictive value.",
                issue_ids=(20, 22),
                evidence_categories=("sim_real_predictive_result",),
                alternative_group="predictive_outcome",
            ),
            GateCheckDefinition(
                check_id="well_powered_negative_result",
                description="A well-powered negative held-out result is reported honestly.",
                issue_ids=(20, 22),
                evidence_categories=("sim_real_predictive_result",),
                alternative_group="predictive_outcome",
            ),
        ),
    ),
)

GATES_BY_ID = {item.gate_id: item for item in GATE_DEFINITIONS}
