"""Pure MLX inference for MiniMax Music 3."""

from ._version import __version__
from .pipeline import (
    ExperimentalPrecisionWarning,
    ExperimentalQuantizationWarning,
    GenerationRequest,
    GenerationResult,
    Music3Pipeline,
)
from .prompting import PromptQualityWarning, instrumental_lyrics
from .reference import (
    MAX_REFERENCE_INTERVAL,
    MIN_REFERENCE_INTERVAL,
    ReferenceCodes,
    ReferenceMode,
    ReferenceQualityWarning,
)

__all__ = [
    "MAX_REFERENCE_INTERVAL",
    "MIN_REFERENCE_INTERVAL",
    "ExperimentalPrecisionWarning",
    "ExperimentalQuantizationWarning",
    "GenerationRequest",
    "GenerationResult",
    "Music3Pipeline",
    "PromptQualityWarning",
    "ReferenceCodes",
    "ReferenceMode",
    "ReferenceQualityWarning",
    "__version__",
    "instrumental_lyrics",
]
