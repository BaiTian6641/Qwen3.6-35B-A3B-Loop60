"""Data package: dataset registry, row adapters, chat rendering + loss masks, packing,
weighted mixture streaming with exact resume, and n-gram decontamination.

Entry point for training: `build_train_stream(data_cfg, tokenizer, seed)`.
Everything heavy (`datasets`, `transformers`, `huggingface_hub`) is imported lazily so the
whole pipeline runs offline on a laptop with `ToyTokenizer` + the built-in toy sources.
"""
from __future__ import annotations

from .decontam import Decontaminator, ngram_hashes, normalize_words
from .formats import (
    Conversation,
    DropCounter,
    Example,
    RowFormatError,
    extract_think,
    looks_encrypted,
    normalize_rl_task,
    normalize_row,
    validate_example,
)
from .mixture import MixtureStream, TrainStream, build_train_stream, make_source_factory
from .packing import Packer, tokenize_conversation, tokenize_example, tokenize_text
from .registry import REGISTRY, DatasetSpec, get_spec, list_specs, verify_specs
from .tokenizer import ToyTokenizer, load_tokenizer, render_conversation

__all__ = [
    "REGISTRY",
    "Conversation",
    "DatasetSpec",
    "Decontaminator",
    "DropCounter",
    "Example",
    "MixtureStream",
    "Packer",
    "RowFormatError",
    "ToyTokenizer",
    "TrainStream",
    "build_train_stream",
    "extract_think",
    "get_spec",
    "list_specs",
    "load_tokenizer",
    "looks_encrypted",
    "make_source_factory",
    "ngram_hashes",
    "normalize_rl_task",
    "normalize_row",
    "normalize_words",
    "render_conversation",
    "tokenize_conversation",
    "tokenize_example",
    "tokenize_text",
    "validate_example",
    "verify_specs",
]
