"""Errors raised by the experimental modeling frontend."""


class ModelingError(ValueError):
    """Base class for frontend construction and lowering failures."""


class ModelingValidationError(ModelingError):
    """Raised when a frontend program has inconsistent semantics."""


class TimingResolutionError(ModelingError):
    """Raised when no timing is available for a phase."""
