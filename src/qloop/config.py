"""YAML config with `_base_` inheritance and dotted CLI overrides.

    cfg = load_config("configs/stages/s1_loop_heal.yaml", overrides=["train.lr=1e-4"])
    cfg.train.lr            # attribute access
    cfg.to_dict()           # plain dict (logged into every run dir)
"""
from __future__ import annotations

import copy
import json
import os
import re
from typing import Any

import yaml


class _Loader(yaml.SafeLoader):
    """SafeLoader + YAML-1.2 style floats ("1e-4" -> 0.0001 instead of the string "1e-4")."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+$|^[-+]?\d*\.\d+$|^[-+]?\d+\.$"),
    list("-+0123456789."),
)


def yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # SafeLoader subclass: no arbitrary objects


class Cfg(dict):
    """dict with attribute access. Missing keys raise a *helpful* error."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"config key '{key}' not found. Available: {sorted(self.keys())}"
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self, default=str))


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k == "_base_":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not v.get("_replace_", False):
            out[k] = deep_merge(out[k], v)
        else:
            val = {kk: vv for kk, vv in v.items() if kk != "_replace_"} if isinstance(v, dict) else v
            out[k] = copy.deepcopy(val)
    return out


def _load_yaml_recursive(path: str, _stack: list[str] | None = None) -> dict[str, Any]:
    path = os.path.abspath(path)
    _stack = _stack or []
    if path in _stack:
        raise ValueError(f"circular _base_ include: {' -> '.join([*_stack, path])}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml_load(f.read()) or {}
    bases = data.get("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for b in bases:
        bpath = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
        merged = deep_merge(merged, _load_yaml_recursive(bpath, [*_stack, path]))
    return deep_merge(merged, data)


def _parse_value(raw: str) -> Any:
    try:
        return yaml_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        cur = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str, overrides: list[str] | None = None) -> Cfg:
    data = _load_yaml_recursive(path)
    data = apply_overrides(data, overrides)
    data.setdefault("_meta_", {})["config_path"] = os.path.abspath(path)
    return _wrap(data)
