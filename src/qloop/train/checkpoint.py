"""Checkpointing of *trainable* state only (adapters, LSA indexers, expert deltas, unfrozen base
slices) + optimizer + data-stream + RNG, with atomic writes, rotation and optional sync.

The 35B base is never re-saved during training (it is re-downloaded / cached by HF). Keys are
the stable parameter names recorded *before* loop wrapping, so checkpoints survive re-wrapping
and changes of loop depth.

Layout: <run_dir>/checkpoints/step_000123/{trainable.pt, optim.pt, state.json}
         <run_dir>/checkpoints/LATEST          (text file with the directory name)
         <run_dir>/checkpoints/PROMOTED        (RSI regression gate: last accepted step)
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import time
from typing import Any

import torch
from torch import nn

log = logging.getLogger("qloop")


def stable_names(model: nn.Module, extra: dict[str, nn.Module] | None = None) -> dict[int, str]:
    """id(param) -> stable name. Call BEFORE loop wrapping; extra modules get a prefix."""
    names: dict[int, str] = {}
    for n, p in model.named_parameters():
        names.setdefault(id(p), n)
    for prefix, mod in (extra or {}).items():
        for n, p in mod.named_parameters():
            names.setdefault(id(p), f"{prefix}.{n}")
    return names


def _rng_state() -> dict[str, Any]:
    st = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict[str, Any]) -> None:
    random.setstate(st["python"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


class CheckpointManager:
    def __init__(self, run_dir: str, keep_last: int = 3, sync: dict[str, Any] | None = None):
        self.dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.dir, exist_ok=True)
        self.keep_last = keep_last
        self.sync = sync or {}

    # ------------------------------------------------------------------ paths
    def _pointer(self, name: str) -> str | None:
        p = os.path.join(self.dir, name)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            d = f.read().strip()
        full = os.path.join(self.dir, d)
        return full if os.path.isdir(full) else None

    def latest(self) -> str | None:
        return self._pointer("LATEST")

    def promoted(self) -> str | None:
        return self._pointer("PROMOTED")

    def _write_pointer(self, name: str, target: str) -> None:
        tmp = os.path.join(self.dir, f".{name}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(os.path.basename(target))
        os.replace(tmp, os.path.join(self.dir, name))

    # ------------------------------------------------------------------ save / load
    def save(self, step: int, params: dict[str, torch.Tensor], optim_state: dict[str, Any] | None,
             state: dict[str, Any]) -> str:
        name = f"step_{step:07d}"
        final = os.path.join(self.dir, name)
        tmp = os.path.join(self.dir, f".tmp_{name}_{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        t0 = time.time()
        torch.save({k: v.detach().to("cpu") for k, v in params.items()}, os.path.join(tmp, "trainable.pt"))
        if optim_state is not None:
            torch.save(optim_state, os.path.join(tmp, "optim.pt"))
        torch.save(_rng_state(), os.path.join(tmp, "rng.pt"))
        meta = dict(state, step=step, saved_at=time.time(), n_params=sum(v.numel() for v in params.values()))
        with open(os.path.join(tmp, "state.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        self._write_pointer("LATEST", final)
        self._rotate()
        size = sum(os.path.getsize(os.path.join(final, f)) for f in os.listdir(final)) / 2**20
        log.info("checkpoint %s saved (%.0f MiB, %.1fs)", final, size, time.time() - t0)
        self._sync(final)
        return final

    def load(self, path: str | None = None, map_location: str = "cpu") -> dict[str, Any] | None:
        path = path or self.latest()
        if path is None:
            return None
        with open(os.path.join(path, "state.json"), encoding="utf-8") as f:
            state = json.load(f)
        out = {"path": path, "state": state,
               "params": torch.load(os.path.join(path, "trainable.pt"), map_location=map_location)}
        op = os.path.join(path, "optim.pt")
        out["optim"] = torch.load(op, map_location=map_location, weights_only=False) if os.path.exists(op) else None
        rp = os.path.join(path, "rng.pt")
        if os.path.exists(rp):
            _set_rng_state(torch.load(rp, weights_only=False))
        log.info("loaded checkpoint %s (step %s)", path, state.get("step"))
        return out

    def promote(self, path: str) -> None:
        self._write_pointer("PROMOTED", path)
        log.info("promoted %s", path)

    def _rotate(self) -> None:
        keep = {os.path.basename(p) for p in (self.latest(), self.promoted()) if p}
        steps = sorted(d for d in os.listdir(self.dir) if d.startswith("step_"))
        for d in steps[:-self.keep_last] if self.keep_last > 0 else []:
            if d not in keep:
                shutil.rmtree(os.path.join(self.dir, d), ignore_errors=True)

    def _sync(self, path: str) -> None:
        dest = self.sync.get("copy_to")
        if dest:
            try:
                target = os.path.join(dest, os.path.basename(os.path.dirname(self.dir)), os.path.basename(path))
                shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(path, target)
                log.info("synced checkpoint to %s", target)
            except Exception as e:
                log.warning("checkpoint copy to %s failed: %r", dest, e)
        repo = self.sync.get("hf_repo")
        if repo:
            try:
                from huggingface_hub import HfApi

                api = HfApi()
                api.create_repo(repo, private=True, exist_ok=True)
                api.upload_folder(folder_path=path, repo_id=repo, path_in_repo=os.path.basename(path),
                                  commit_message=f"qloop checkpoint {os.path.basename(path)}")
                log.info("uploaded checkpoint to hf://%s/%s", repo, os.path.basename(path))
            except Exception as e:
                log.warning("HF upload to %s failed: %r (training continues)", repo, e)


def collect_trainable(named: dict[str, nn.Parameter]) -> dict[str, torch.Tensor]:
    return {n: p.detach() for n, p in named.items() if p.requires_grad}


def restore_trainable(named: dict[str, nn.Parameter], saved: dict[str, torch.Tensor], strict: bool = True) -> None:
    missing = [n for n, p in named.items() if p.requires_grad and n not in saved]
    unexpected = [n for n in saved if n not in named]
    if strict and (missing or unexpected):
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:10]} (+{max(0, len(missing) - 10)}), "
                           f"unexpected={unexpected[:10]} (+{max(0, len(unexpected) - 10)}). "
                           "Did the loop/LSA/expert config change? use strict=false to load partially.")
    with torch.no_grad():
        for n, t in saved.items():
            if n in named:
                p = named[n]
                if p.shape != t.shape:
                    raise RuntimeError(f"shape mismatch for {n}: model {tuple(p.shape)} vs ckpt {tuple(t.shape)}")
                p.copy_(t.to(p.device, p.dtype))
