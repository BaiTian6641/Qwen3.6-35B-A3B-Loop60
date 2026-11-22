"""Cache kwarg and output compatibility helpers across `transformers <= 5.23`, `>= 5.24`, and `ToyModel`.

In `transformers 5.24.0`, Qwen3.5/3.6 layers renamed `cache_params` / `past_key_values` -> `cache`
and model outputs expose `.cache` (alongside or replacing `.past_key_values`).
"""
from __future__ import annotations

import inspect
from typing import Any

CACHE_KWARG_NAMES: tuple[str, ...] = ("cache", "cache_params", "past_key_values", "past_key_value")


def extract_cache_kwarg(kwargs: dict[str, Any]) -> tuple[str | None, Any]:
    """Return `(key, cache_obj)` for the first non-`None` cache kwarg in safe priority order."""
    for key in CACHE_KWARG_NAMES:
        val = kwargs.get(key)
        if val is not None:
            return key, val
    return None, None


def extract_cache_from_kwargs(kwargs: dict[str, Any]) -> Any:
    """Return the first non-`None` cache object from `kwargs` across all HF / ToyModel names."""
    return extract_cache_kwarg(kwargs)[1]


def extract_output_cache(out: Any) -> Any:
    """Read cache from a model output, supporting both `.cache` (`transformers >= 5.24`) and `.past_key_values`."""
    cache = getattr(out, "cache", None)
    if cache is None:
        cache = getattr(out, "past_key_values", None)
    return cache


def cache_kwarg_name(mod: Any) -> str:
    """Choose `"cache"` when `mod.forward` accepts `cache` (and not `past_key_values`), else `"past_key_values"`."""
    fn = getattr(mod, "forward", mod)
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return "past_key_values"
    if "cache" in params and "past_key_values" not in params:
        return "cache"
    return "past_key_values"
