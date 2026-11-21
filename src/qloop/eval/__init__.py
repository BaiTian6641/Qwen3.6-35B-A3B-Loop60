"""Evaluation + serving for the looped Qwen3.6 stack.

    generate     Generator / GenOut: cache-based sampling (toy, looped and HF models), loop depth switch
    scoring      answer extraction + scorers (AIME, math, MCQ A-J, MRCR, ARC grids, HLE, GAIA), pass@k, maj@k
    code_exec    LiveCodeBench-style test execution
    kernelbench  KernelBench correctness + speedup in a subprocess
    registry     BenchmarkSpec registry + prompt builders + load_items
    runners      run_benchmark (multi-depth, resumable) / run_suite
    external     Harbor / tau-bench / OpenAI-endpoint harness integration
    server       OpenAI-compatible HTTP server (stdlib)
    testing      ByteTokenizer / ScriptedLM / build_toy_generator for offline tests

Submodules are imported lazily by the caller (`from qloop.eval.runners import run_benchmark`).
"""
from __future__ import annotations

from .generate import Generator, GenOut

__all__ = ["GenOut", "Generator"]
