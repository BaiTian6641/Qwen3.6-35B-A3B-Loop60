"""Shipped On-Weight Inference Engine & Self-Contained HF Custom Modeling for Loop-HSSD v3."""
from __future__ import annotations

from .engine import (
    GenerationMetrics,
    GenerationResult,
    InferenceConfig,
    OptimizedLoopInferenceEngine,
    QuantizedHybridCache,
)
from .hf_custom_modeling import (
    AUTO_MAP,
    CONFIG_FILENAME,
    MODELING_FILENAME,
    CausalLMPackOutput,
    Qwen36LoopHSSDConfig,
    Qwen36LoopHSSDForCausalLM,
    load_safetensors_file,
    resolve_pack_directory,
    save_safetensors_file,
    write_standalone_hf_modeling_files,
)

__all__ = [
    "AUTO_MAP",
    "CONFIG_FILENAME",
    "MODELING_FILENAME",
    "CausalLMPackOutput",
    "GenerationMetrics",
    "GenerationResult",
    "InferenceConfig",
    "OptimizedLoopInferenceEngine",
    "QuantizedHybridCache",
    "Qwen36LoopHSSDConfig",
    "Qwen36LoopHSSDForCausalLM",
    "load_safetensors_file",
    "resolve_pack_directory",
    "save_safetensors_file",
    "write_standalone_hf_modeling_files",
]
