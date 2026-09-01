from .artifacts import (
    FAILURE_DETECTOR_RUN_MANIFEST_FORMAT,
    summarize_failure_detectors,
    write_failure_detector_manifest,
)
from .contact import ContactDetector
from .grasp import GraspDetector
from .manager import (
    FAILURE_DETECTOR_MANIFEST_FORMAT,
    DetectorEmission,
    FailureDetectorManager,
    FailureDetectorRuntimeError,
)
from .protocol import (
    FAILURE_DETECTOR_PROTOCOL_FORMAT,
    FAILURE_DETECTOR_PROTOCOL_VERSION,
    DetectorSignalRequirement,
    DetectorSupport,
    FailureDetector,
    FailureDetectorContract,
)
from .stall import StallDetector


def build_default_failure_detectors() -> tuple[FailureDetector, ...]:
    """Return default detector set for policy-agnostic failure monitoring."""

    return (
        ContactDetector(),
        GraspDetector(),
        StallDetector(),
    )


__all__ = [
    "FailureDetector",
    "FailureDetectorContract",
    "FailureDetectorManager",
    "FailureDetectorRuntimeError",
    "DetectorEmission",
    "DetectorSignalRequirement",
    "DetectorSupport",
    "ContactDetector",
    "GraspDetector",
    "StallDetector",
    "build_default_failure_detectors",
    "FAILURE_DETECTOR_MANIFEST_FORMAT",
    "FAILURE_DETECTOR_PROTOCOL_FORMAT",
    "FAILURE_DETECTOR_PROTOCOL_VERSION",
    "FAILURE_DETECTOR_RUN_MANIFEST_FORMAT",
    "summarize_failure_detectors",
    "write_failure_detector_manifest",
]
