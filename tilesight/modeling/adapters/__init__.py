"""Compatibility adapters from ``modeling`` IR to full models."""

from .gemm import (
    GemmLegacyEvaluation,
    GemmModelSpec,
    GemmWaveMetadata,
    evaluate_gemm_legacy,
    extract_gemm_spec,
    gemm_legacy_arch_view,
    model_gemm,
)
from .general_gemm import (
    BoundGemmCostOracle,
    GeneralGemmOptions,
    GeneralGemmResult,
    bind_general_gemm,
    evaluate_general_gemm,
    model_general_gemm,
)
from .b200_cta2_gemm import (
    B200Cta2ExecutionPlan,
    B200Cta2LegacyBoundOracle,
    B200Cta2NativeEvaluation,
    B200Cta2NativeOptions,
    Cta2OperandRoute,
    Cta2ParticipantFootprint,
    Cta2PhysicalTransfer,
    Cta2ProtocolCosts,
    model_b200_cta2_native,
)
from .flash_attention import (
    FlashAttentionSpec,
    LegacyFlashAttentionModelUnavailable,
    model_fa3,
    model_fa4,
    resolve_flash_attention_spec,
)

__all__ = [
    "BoundGemmCostOracle",
    "B200Cta2ExecutionPlan",
    "B200Cta2LegacyBoundOracle",
    "B200Cta2NativeEvaluation",
    "B200Cta2NativeOptions",
    "Cta2OperandRoute",
    "Cta2ParticipantFootprint",
    "Cta2PhysicalTransfer",
    "Cta2ProtocolCosts",
    "GeneralGemmOptions",
    "GeneralGemmResult",
    "GemmLegacyEvaluation",
    "GemmModelSpec",
    "GemmWaveMetadata",
    "bind_general_gemm",
    "evaluate_general_gemm",
    "evaluate_gemm_legacy",
    "extract_gemm_spec",
    "gemm_legacy_arch_view",
    "FlashAttentionSpec",
    "LegacyFlashAttentionModelUnavailable",
    "model_fa3",
    "model_fa4",
    "model_b200_cta2_native",
    "model_general_gemm",
    "model_gemm",
    "resolve_flash_attention_spec",
]
