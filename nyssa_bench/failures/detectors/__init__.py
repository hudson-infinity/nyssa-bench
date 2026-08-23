from .contact import ContactDetector
from .grasp import GraspDetector
from .manager import FailureDetectorManager
from .protocol import FailureDetector
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
    "FailureDetectorManager",
    "ContactDetector",
    "GraspDetector",
    "StallDetector",
    "build_default_failure_detectors",
]
