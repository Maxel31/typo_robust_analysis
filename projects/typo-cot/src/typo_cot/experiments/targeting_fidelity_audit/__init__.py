"""Public API for the paper Appendix A targeting-fidelity audit."""

from .runner import (
    TargetingFidelityAuditConfig,
    TargetingFidelityAuditError,
    TargetingFidelityAuditResult,
    run_targeting_fidelity_audit,
)

__all__ = [
    "TargetingFidelityAuditConfig",
    "TargetingFidelityAuditError",
    "TargetingFidelityAuditResult",
    "run_targeting_fidelity_audit",
]
