"""Weighted source mixing with exact resume, and the end-to-end training stream.

    stream = build_train_stream(cfg.data, tokenizer, seed=cfg.train.seed)
    batch = next(stream)                         # dict(input_ids, labels, doc_ids, position_ids, ...)
    ckpt["data"] = stream.state_dict()           # small: counters + rng + packer buffer
    stream.load_state_dict(ckpt["data"])         # continues with exactly the same rows

Pipeline per draw:  MixtureStream picks a source (weighted, seeded) -> that source's iterator
(load -> adapter -> validation, drops counted per reason) -> decontamination -> cross-source dedupe
-> tokenisation with loss masks -> Packer.

Sources are infinite: an exhausted source is re-opened with epoch+1 (new shuffle seed) and logged.
Resume: each source remembers its epoch, #examples consumed and the raw-row cursor of the last
example; source factories that accept `resume_from` skip raw rows cheaply (HF `IterableDataset.skip`),
others are fast-forwarded by re-iterating.
"""
from __future__ import annotations

import collections
import hashlib
import inspect
import itertools
import json
import logging
import os
import random
from collections.abc import Iterable, Iterator
from typing import Any, Callable

import numpy as np

from qloop.debug import describe, require, stage

from .decontam import Decontaminator, EvalTextProvider
from .formats import DropCounter, Example, RowDrop, RowFormatError, arc_task_prompt, normalize_row, validate_example
from .packing import Packer, separator_id, tokenize_example
from .registry import REGISTRY, DatasetSpec, eval_specs_for_group, get_spec

log = logging.getLogger("qloop")

SourceFactory = Callable[..., Iterable[Any]]


# ============================================================================ MixtureStream

def _accepts(fn: Callable[..., Any], name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _n_positional(fn: Callable[..., Any]) -> int:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 0
    return sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty for p in params)


class _SourceExhausted(Exception):
    """Internal signal that a single source has finished (single_pass or skip_empty_sources)."""


class MixtureStream:
    """Infinite weighted interleaving of `(name, weight, iterator_factory)` sources.

    `iterator_factory()` or `iterator_factory(epoch)` returns a fresh iterable for that epoch; if it
    also accepts `resume_from=(row_index, sub_index)` it must yield only items strictly after that
    position (items expose `.row_index`/`.sub_index`, like `Example`). Yields `(name, item)`.
    """

    def __init__(
        self,
        sources: list[tuple[str, float, SourceFactory]],
        seed: int = 0,
        *,
        single_pass: bool = False,
        skip_empty_sources: bool = False,
    ):
        require(len(sources) > 0, "MixtureStream: no sources")
        names = [s[0] for s in sources]
        require(len(set(names)) == len(names), "MixtureStream: duplicate source names", names=names)
        weights = np.array([float(s[1]) for s in sources], dtype=np.float64)
        require(bool(np.all(weights > 0)) and bool(np.all(np.isfinite(weights))),
                "MixtureStream: weights must be finite and > 0", weights=dict(zip(names, weights.tolist())))
        self.names = names
        self.factories = [s[2] for s in sources]
        self.probs = weights / weights.sum()
        self.seed = int(seed)
        self.single_pass = bool(single_pass)
        self.skip_empty_sources = bool(skip_empty_sources)
        self.rng = np.random.Generator(np.random.PCG64(self.seed))
        self.draws = 0
        n = len(sources)
        self.epoch = [0] * n
        self.consumed = [0] * n          # items consumed in the current epoch
        self.total = [0] * n             # items consumed over all epochs
        self.cursor: list[tuple[int, int] | None] = [None] * n
        self._iters: list[Iterator[Any] | None] = [None] * n
        self._exhausted: list[bool] = [False] * n
        log.info("[mixture] %s", ", ".join(f"{a}={p:.3f}" for a, p in zip(names, self.probs)))

    def _open(self, i: int) -> Iterator[Any]:
        fac, e = self.factories[i], self.epoch[i]
        kwargs: dict[str, Any] = {}
        resumable = _accepts(fac, "resume_from")
        if self.consumed[i] and resumable and self.cursor[i] is not None:
            kwargs["resume_from"] = self.cursor[i]
        it_src = fac(e, **kwargs) if (_n_positional(fac) >= 1 or _accepts(fac, "epoch")) else fac(**kwargs)
        it = iter(it_src)
        if self.consumed[i] and "resume_from" not in kwargs:
            log.info("[mixture] resume: fast-forwarding source %s by %d items (epoch %d)", self.names[i],
                     self.consumed[i], e)
            for _ in itertools.islice(it, self.consumed[i]):
                pass
        elif "resume_from" in kwargs:
            log.info("[mixture] resume: source %s epoch %d from row cursor %s (%d items consumed)", self.names[i],
                     e, kwargs["resume_from"], self.consumed[i])
        return it

    def _next_from(self, i: int) -> Any:
        for _attempt in range(2):
            if self._iters[i] is None:
                self._iters[i] = self._open(i)
            it = self._iters[i]
            assert it is not None
            try:
                item = next(it)
            except StopIteration:
                if self.consumed[i] == 0:
                    if self.skip_empty_sources:
                        log.warning(
                            "[mixture] source '%s' produced no items in epoch %d; skipping this source",
                            self.names[i],
                            self.epoch[i],
                        )
                        raise _SourceExhausted from None
                    raise RuntimeError(
                        f"[mixture] source '{self.names[i]}' produced no items in epoch {self.epoch[i]} "
                        "(empty dataset, everything filtered, holdout split matched 0 rows, or max_rows=0?)") from None
                if self.single_pass:
                    log.info(
                        "[mixture] source %s completed single pass after %d items",
                        self.names[i],
                        self.consumed[i],
                    )
                    raise _SourceExhausted from None
                log.info("[mixture] source %s exhausted after %d items -> epoch %d", self.names[i],
                         self.consumed[i], self.epoch[i] + 1)
                self.epoch[i] += 1
                self.consumed[i] = 0
                self.cursor[i] = None
                self._iters[i] = None
                continue
            self.consumed[i] += 1
            self.total[i] += 1
            ri = getattr(item, "row_index", None)
            self.cursor[i] = (int(ri), int(getattr(item, "sub_index", 0))) if ri is not None and ri >= 0 else None
            return item
        raise RuntimeError(f"[mixture] source '{self.names[i]}' produced no items after re-opening "
                           f"(epoch {self.epoch[i]})")

    def __iter__(self) -> MixtureStream:
        return self

    def __next__(self) -> tuple[str, Any]:
        while True:
            active = [j for j in range(len(self.names)) if not self._exhausted[j]]
            if not active:
                raise StopIteration
            if len(active) == len(self.names):
                i = int(self.rng.choice(len(self.names), p=self.probs))
            else:
                p_act = self.probs[active]
                p_act = p_act / p_act.sum()
                i = active[int(self.rng.choice(len(active), p=p_act))]
            try:
                item = self._next_from(i)
            except _SourceExhausted:
                self._exhausted[i] = True
                self._iters[i] = None
                continue
            self.draws += 1
            return self.names[i], item

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "draws": self.draws,
            "rng": self.rng.bit_generator.state,
            "sources": {n: {"epoch": self.epoch[i], "consumed": self.consumed[i], "total": self.total[i],
                            "cursor": list(self.cursor[i]) if self.cursor[i] is not None else None}
                        for i, n in enumerate(self.names)},
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        if int(sd.get("seed", self.seed)) != self.seed:
            log.warning("[mixture] resuming with seed %d but checkpoint used %s", self.seed, sd.get("seed"))
        self.rng.bit_generator.state = sd["rng"]
        self.draws = int(sd.get("draws", 0))
        saved = sd.get("sources", {})
        missing = [n for n in self.names if n not in saved]
        extra = [n for n in saved if n not in self.names]
        if missing or extra:
            log.warning("[mixture] source set changed since checkpoint: new=%s removed=%s (new sources start "
                        "fresh; removed ones are ignored)", missing, extra)
        for i, n in enumerate(self.names):
            st = saved.get(n)
            if st is None:
                self.epoch[i] = self.consumed[i] = self.total[i] = 0
                self.cursor[i] = None
            else:
                self.epoch[i], self.consumed[i] = int(st["epoch"]), int(st["consumed"])
                self.total[i] = int(st.get("total", st["consumed"]))
                c = st.get("cursor")
                self.cursor[i] = (int(c[0]), int(c[1])) if c is not None else None
            self._iters[i] = None
            self._exhausted[i] = False

    def stats(self) -> dict[str, Any]:
        return {n: {"epoch": self.epoch[i], "total": self.total[i],
                    "frac": self.total[i] / max(1, self.draws)} for i, n in enumerate(self.names)}


# ============================================================================ raw row sources

_NAMES = ["Alice", "Bob", "Chen", "Dana", "Emeka", "Fatima", "Goro", "Hana"]
_ITEMS = ["apples", "marbles", "books", "coins", "stickers", "pencils"]
_WORDS = [
    "the", "of", "and", "a", "to", "in", "is", "was", "for", "on", "that", "with", "as", "by", "it", "from",
    "at", "this", "be", "are", "which", "or", "have", "an", "loop", "layer", "model", "depth", "river",
    "mountain", "theory", "energy", "water", "light", "market", "city", "music", "history", "system",
    "method", "result", "number", "signal", "pattern", "garden", "window", "teacher", "student", "library",
]


def _row_rng(seed: int, epoch: int, i: int) -> random.Random:
    return random.Random((int(seed) * 1_000_003 + int(epoch)) * 1_000_003 + int(i))


def toy_cot_row(seed: int, epoch: int, i: int, bad_frac: float = 0.0) -> dict[str, Any]:
    r = _row_rng(seed, epoch, i)
    who, item = r.choice(_NAMES), r.choice(_ITEMS)
    a, b, c = r.randint(2, 60), r.randint(1, 40), r.randint(1, 30)
    k = r.randint(2, 5)
    if r.random() < 0.5:
        q = (f"{who} has {a} {item}. {who} buys {b} more and then gives away {c}. "
             f"How many {item} does {who} have now?")
        ans = a + b - c
        cot = (f"Start with {a} {item}. After buying {b} more: {a} + {b} = {a + b}. "
               f"After giving away {c}: {a + b} - {c} = {ans}.")
    else:
        q = f"{who} packs {k} boxes with {a} {item} each and finds {b} loose {item}. How many {item} in total?"
        ans = k * a + b
        cot = f"The boxes hold {k} x {a} = {k * a} {item}. Adding the {b} loose ones: {k * a} + {b} = {ans}."
    row = {"id": f"toy_cot-{epoch}-{i}", "prompt": q, "reasoning": cot, "response": f"The answer is {ans}."}
    if bad_frac > 0 and r.random() < bad_frac:
        kind = r.randint(0, 2)
        blob = "gAAAAB" + "".join(r.choice("ABCDEFabcdef0123456789") for _ in range(300))
        row["reasoning"] = ["", "[redacted]", blob][kind]
    return row


def toy_text_row(seed: int, epoch: int, i: int) -> dict[str, Any]:
    r = _row_rng(seed + 7919, epoch, i)
    sents = []
    for _ in range(r.randint(3, 12)):
        w = [r.choice(_WORDS) for _ in range(r.randint(5, 16))]
        sents.append(" ".join(w).capitalize() + ".")
    return {"id": f"toy_text-{epoch}-{i}", "text": " ".join(sents)}


def _arc_cache_dir() -> str:
    return os.environ.get("QLOOP_ARC_DIR", os.path.join(os.path.expanduser("~"), ".cache", "qloop", "ARC-AGI-2"))


def ensure_arc_data(repo: str = "arcprize/ARC-AGI-2", data_dir: str | None = None) -> str:
    """Directory containing `<split>/<task>.json`; downloads the GitHub archive once if needed."""
    root = data_dir or _arc_cache_dir()
    data = os.path.join(root, "data")
    if os.path.isdir(os.path.join(data, "training")):
        return data
    if os.path.isdir(os.path.join(root, "training")):
        return root
    import io
    import urllib.request
    import zipfile

    url = f"https://codeload.github.com/{repo}/zip/refs/heads/main"
    log.info("[data] downloading %s -> %s", url, root)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            blob = r.read()
    except Exception as e:
        raise RuntimeError(f"ARC-AGI-2: cannot download {url} ({e!r}); clone the repo and set QLOOP_ARC_DIR "
                           "or the source's `data_dir`") from e
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            parts = name.split("/", 1)
            if len(parts) < 2 or not parts[1].startswith("data/") or name.endswith("/"):
                continue
            dst = os.path.join(root, parts[1])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(zf.read(name))
    require(os.path.isdir(os.path.join(data, "training")), "ARC-AGI-2 archive had no data/training", root=root)
    return data


def arc_task_rows(split: str, data_dir: str | None = None, repo: str = "arcprize/ARC-AGI-2") -> list[dict[str, Any]]:
    d = os.path.join(ensure_arc_data(repo, data_dir), split)
    require(os.path.isdir(d), f"ARC-AGI-2: split dir not found: {d}", split=split)
    rows = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                task = json.load(f)
            task["task_id"] = fn[:-5]
            rows.append(task)
    return rows


def _hf_rows(spec: DatasetSpec, seed: int, epoch: int, skip: int, streaming: bool, shuffle_buffer: int,
             token: str | None) -> Iterator[dict[str, Any]]:
    try:
        import datasets
    except ImportError as e:  # pragma: no cover - colab only
        raise ImportError(f"[{spec.name}] needs `datasets` (pip install -e '.[colab]')") from e
    remaining = skip
    for split in spec.splits:
        kw: dict[str, Any] = {"split": split, "streaming": streaming}
        if token:
            kw["token"] = token
        if spec.revision is not None:
            kw["revision"] = spec.revision
        if spec.data_files is not None:
            ds = datasets.load_dataset("json", data_files=spec.data_files, **kw)
        else:
            ds = datasets.load_dataset(spec.hf_id, spec.config, **kw)
        if shuffle_buffer > 0:
            if streaming:
                ds = ds.shuffle(seed=seed + epoch, buffer_size=shuffle_buffer)
            else:
                ds = ds.shuffle(seed=seed + epoch)
        if remaining:
            n = None
            if not streaming:
                n = len(ds)
                if remaining >= n:
                    remaining -= n
                    continue
            ds = ds.skip(remaining) if streaming else ds.select(range(remaining, n or 0))
            remaining = 0
        yield from ds


def raw_rows(spec: DatasetSpec, seed: int, epoch: int, skip: int = 0, src_cfg: dict[str, Any] | None = None,
             streaming: bool = True) -> Iterator[dict[str, Any]]:
    """Raw rows of one epoch starting at raw position `skip` (deterministic for (seed, epoch))."""
    cfg = src_cfg or {}
    if spec.kind == "builtin":
        n = int(cfg.get("toy_rows", 512))
        which = spec.hf_id.split(":", 1)[1]
        for i in range(skip, n):
            if which == "toy_cot":
                yield toy_cot_row(seed, epoch, i, float(cfg.get("bad_frac", 0.0)))
            elif which == "toy_text":
                yield toy_text_row(seed, epoch, i)
            else:
                raise KeyError(f"[{spec.name}] unknown builtin source {spec.hf_id}")
        return
    if spec.kind == "github":
        rows = arc_task_rows(spec.splits[0], cfg.get("data_dir"), spec.hf_id.split(":", 1)[1])
        random.Random(seed * 7 + epoch).shuffle(rows)
        yield from rows[skip:]
        return
    yield from _hf_rows(spec, seed, epoch, skip, streaming, int(cfg.get("shuffle_buffer", 10_000)),
                        cfg.get("token") or os.environ.get("HF_TOKEN"))


SPLIT_ROLES = ("all", "train", "eval")


def doc_holdout_key(ex: Example, row: dict[str, Any] | None = None) -> str:
    """Canonical document-level key used for deterministic train/eval holdout partitioning.

    Derived from canonical task/doc ID or normalized content (length + prefix + content SHA-1)
    so duplicate documents across distinct sources deterministically land on the exact same side.
    """
    meta_doc_id = ex.meta.get("doc_id") if isinstance(ex.meta, dict) else None
    if meta_doc_id is not None and str(meta_doc_id).strip():
        return f"id:{str(meta_doc_id).strip()}"
    meta_task_id = ex.meta.get("task_id") if isinstance(ex.meta, dict) else None
    if meta_task_id is not None and str(meta_task_id).strip():
        return f"task:{str(meta_task_id).strip()}"
    if ex.kind == "text":
        norm = " ".join((ex.text or "").split())
        kind_tag = "text"
    else:
        prompt = " ".join(ex.prompt_text().split())
        resp = " ".join(ex.response_text().split())
        norm = f"{prompt}\x00{resp}" if (prompt or resp) else " ".join(ex.plain_text().split())
        kind_tag = "chat"
    if norm:
        digest = hashlib.sha1(norm.encode("utf-8", "replace")).hexdigest()[:16]
        return f"{kind_tag}:{len(norm)}:{norm[:256]}:{digest}"
    if isinstance(row, dict):
        for id_k in ("doc_id", "id", "task_id"):
            val = row.get(id_k)
            if val is not None and str(val).strip():
                return f"id:{str(val).strip()}"
    return f"{ex.source}:{ex.row_index}:{ex.sub_index}"


def doc_holdout_bucket(doc_key: str, holdout_mod: int = 100) -> int:
    """Map a canonical `doc_key` to `[0, holdout_mod)` deterministically via SHA-1."""
    require(int(holdout_mod) > 0, "holdout_mod must be > 0", holdout_mod=holdout_mod)
    return int(hashlib.sha1(doc_key.encode("utf-8", "replace")).hexdigest()[:8], 16) % int(holdout_mod)


def is_eval_holdout_doc(
    ex: Example,
    *,
    holdout_mod: int = 100,
    holdout_eval_buckets: int = 2,
    row: dict[str, Any] | None = None,
) -> bool:
    """Return True iff `ex` belongs to the deterministic hash-partitioned eval holdout."""
    h_mod = int(holdout_mod)
    h_eval = int(holdout_eval_buckets)
    if h_mod <= 0 or h_eval <= 0:
        return False
    require(0 <= h_eval < h_mod, "holdout_eval_buckets must satisfy 0 <= holdout_eval_buckets < holdout_mod",
            holdout_mod=h_mod, holdout_eval_buckets=h_eval)
    return doc_holdout_bucket(doc_holdout_key(ex, row=row), h_mod) < h_eval


def make_source_factory(spec: DatasetSpec, seed: int, drops: DropCounter, src_cfg: dict[str, Any] | None = None,
                        max_rows: int | None = None, streaming: bool = True, *, split_role: str = "all",
                        holdout_mod: int = 100, holdout_eval_buckets: int = 2) -> SourceFactory:
    """factory(epoch, resume_from=None) -> iterator of validated `Example`s of one epoch."""
    cfg = dict(src_cfg or {})
    eff_role = str(split_role)
    eff_mod = int(holdout_mod)
    eff_eval_buckets = int(holdout_eval_buckets)
    require(eff_role in SPLIT_ROLES, f"split_role must be one of {SPLIT_ROLES}", split_role=eff_role)
    if eff_role != "all" and eff_mod > 0:
        require(0 <= eff_eval_buckets < eff_mod,
                "holdout_eval_buckets must satisfy 0 <= holdout_eval_buckets < holdout_mod",
                holdout_mod=eff_mod, holdout_eval_buckets=eff_eval_buckets)
    partition_active = eff_role != "all" and eff_mod > 0 and eff_eval_buckets > 0

    def factory(epoch: int, resume_from: tuple[int, int] | None = None) -> Iterator[Example]:
        start = 0 if resume_from is None else int(resume_from[0])
        rows = raw_rows(spec, seed, epoch, skip=start, src_cfg=cfg, streaming=streaming)
        for idx, row in enumerate(rows, start=start):
            if max_rows is not None and idx >= max_rows:
                return
            try:
                exs = normalize_row(row, spec, idx)
            except RowFormatError as e:
                drops.drop(spec.name, e.reason, idx, str(e))
                continue
            except RowDrop as e:
                drops.drop(spec.name, e.reason, idx, e.detail)
                continue
            for ex in exs:
                if resume_from is not None and (ex.row_index, ex.sub_index) <= tuple(resume_from):
                    continue
                reason = validate_example(ex, spec)
                if reason:
                    drops.drop(spec.name, reason, idx, "")
                    continue
                if partition_active:
                    in_eval = is_eval_holdout_doc(
                        ex,
                        holdout_mod=eff_mod,
                        holdout_eval_buckets=eff_eval_buckets,
                        row=row,
                    )
                    if (eff_role == "train" and in_eval) or (eff_role == "eval" and not in_eval):
                        continue
                drops.keep(spec.name)
                ex.meta.setdefault("epoch", epoch)
                yield ex

    return factory


# ============================================================================ eval providers

def _iter_eval_strings(val: Any) -> Iterator[str]:
    """Yield plain text segments from an eval column, unwrapping JSON-encoded chat message lists (e.g. MRCR)."""
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
            except ValueError:
                parsed = None
            if isinstance(parsed, list) and parsed and all(isinstance(m, dict) and "content" in m for m in parsed):
                for m in parsed:
                    c = m.get("content")
                    if isinstance(c, str) and c.strip():
                        yield c
                return
        yield s
    elif isinstance(val, list):
        for item in val:
            if isinstance(item, dict) and isinstance(item.get("content"), str) and item["content"].strip():
                yield item["content"]
            elif isinstance(item, str) and item.strip():
                yield item


def eval_text_provider(spec: DatasetSpec, src_cfg: dict[str, Any] | None = None) -> EvalTextProvider:
    """Callable yielding the benchmark texts of an eval spec (for decontamination)."""
    cfg = src_cfg or {}

    def provider() -> Iterator[str]:
        if spec.format == "arc_agi":
            for task in arc_task_rows(spec.splits[0], cfg.get("data_dir"), spec.hf_id.split(":", 1)[1]):
                for pair in task.get("test", []):
                    yield arc_task_prompt(task, pair["input"])
            return
        fields = spec.text_fields or ["question", "problem", "prompt", "text"]
        found = False
        for row in raw_rows(spec, 0, 0, src_cfg={**cfg, "shuffle_buffer": 0}, streaming=False):
            for f in fields:
                for text in _iter_eval_strings(row.get(f)):
                    found = True
                    yield text
        require(found, f"eval set '{spec.name}' yielded no text in fields {fields}", hf_id=spec.hf_id)

    return provider


def default_eval_providers(groups: list[str], extra: dict[str, list[DatasetSpec]] | None = None, *,
                           on_missing: str = "error") -> dict[str, EvalTextProvider]:
    out: dict[str, EvalTextProvider] = {}
    for g in groups:
        specs = list((extra or {}).get(g, [])) or eval_specs_for_group(g)
        if not specs:
            raise KeyError(f"decontam group '{g}' has no eval dataset in the registry (role=eval, "
                           f"contamination_group='{g}'). Known groups: "
                           f"{sorted({s.contamination_group for s in REGISTRY.values() if s.role == 'eval'})}")
        out[g] = _chain_providers([(s.name, eval_text_provider(s)) for s in specs], on_missing=on_missing)
    return out


def _chain_providers(provs: list[tuple[str, EvalTextProvider]], on_missing: str = "error") -> EvalTextProvider:
    def _iter() -> Iterator[str]:
        for name, p in provs:
            try:
                yield from p()
            except Exception as e:
                if on_missing != "warn":
                    raise
                log.warning("[decontam] eval source '%s' failed (%s: %s); skipping this source",
                            name, type(e).__name__, e)
    return _iter


# ============================================================================ train stream

def _dedupe_key(ex: Example) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(" ".join(ex.prompt_text().split()).encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(" ".join((ex.response_text() if ex.kind == "chat" else "").split()).encode("utf-8", "replace"))
    return int.from_bytes(h.digest(), "little")


class TrainStream:
    """Iterator of packed rows with a full, exact `state_dict()` (mixture + packer + dedupe set)."""

    def __init__(self, mixture: MixtureStream, tokenizer: Any, seq_len: int, respect_doc_boundaries: bool = True,
                 decontaminator: Decontaminator | None = None, drops: DropCounter | None = None,
                 dedupe: bool = True, dedupe_max: int = 5_000_000, keep_history_reasoning: bool = False,
                 log_every: int = 1000, verify_first: int = 4, max_doc_tokens: int | None = None, *,
                 split_role: str = "all", holdout_mod: int = 100, holdout_eval_buckets: int = 2):
        self.mixture = mixture
        self.tokenizer = tokenizer
        self.sep_id = separator_id(tokenizer)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.sep_id
        self.packer = Packer(seq_len, pad, respect_doc_boundaries)
        self.decontam = decontaminator
        self.drops = drops or DropCounter()
        self.dedupe = dedupe
        self.dedupe_max = dedupe_max
        self._seen: set[int] = set()
        self.keep_history_reasoning = keep_history_reasoning
        self.log_every = log_every
        self.verify_first = verify_first
        self.max_doc_tokens = max_doc_tokens
        self.split_role = str(split_role)
        self.holdout_mod = int(holdout_mod)
        self.holdout_eval_buckets = int(holdout_eval_buckets)
        self.doc_counter = 0
        self._queue: list[dict[str, Any]] = []
        self._exhausted = False
        self._epochs_at_last_doc: list[int] = list(getattr(mixture, "epoch", []))
        self._draws_since_doc = 0
        self.counters: collections.Counter[str] = collections.Counter()
        # len(tok) counts added tokens; HF `vocab_size` excludes them
        self.vocab = len(tokenizer) if hasattr(tokenizer, "__len__") else int(getattr(tokenizer, "vocab_size", 0) or 0)

    def __iter__(self) -> TrainStream:
        return self

    def _process(self, name: str, ex: Example) -> bool:
        if self.decontam is not None:
            hit, group = self.decontam.check(ex.plain_text(), source=name)
            if hit:
                self.counters["decontaminated"] += 1
                self.drops.drop(name, f"contaminated:{group}", ex.row_index, "")
                return False
        if self.dedupe:
            k = _dedupe_key(ex)
            if k in self._seen:
                self.counters["duplicates"] += 1
                self.drops.drop(name, "duplicate", ex.row_index, "")
                return False
            if len(self._seen) < self.dedupe_max:
                self._seen.add(k)
            elif self.counters["dedupe_full_warned"] == 0:
                self.counters["dedupe_full_warned"] = 1
                log.warning("[data] dedupe set reached %d entries; further rows are not deduplicated", self.dedupe_max)
        verify = self.counters["docs"] < self.verify_first
        try:
            ids, labels = tokenize_example(ex, self.tokenizer, self.sep_id, self.keep_history_reasoning, verify=verify)
        except Exception as e:
            raise RuntimeError(f"[{name}] row {ex.row_index}: tokenisation failed: {type(e).__name__}: {e} | "
                               f"example={describe(ex, 400)}") from e
        if self.max_doc_tokens is not None and len(ids) > self.max_doc_tokens:
            self.drops.drop(name, "too_long", ex.row_index, f"{len(ids)} tokens > {self.max_doc_tokens}")
            return False
        if verify and self.vocab:
            bad = [t for t in ids if t < 0 or t >= self.vocab]
            require(not bad, f"[{name}] row {ex.row_index}: token ids outside vocab", bad=bad[:8], vocab=self.vocab)
        self.counters["docs"] += 1
        self.counters["doc_tokens"] += len(ids)
        self.counters["loss_tokens"] += sum(1 for x in labels if x != -100)
        self._queue.extend(self.packer.add(ids, labels, self.doc_counter))
        self.doc_counter += 1
        self._epochs_at_last_doc = list(getattr(self.mixture, "epoch", []))
        self._draws_since_doc = 0
        return True

    def _all_sources_wrapped_without_progress(self) -> bool:
        epochs = getattr(self.mixture, "epoch", None)
        if not isinstance(epochs, list) or len(epochs) != len(self._epochs_at_last_doc) or not epochs:
            return False
        exhausted = getattr(self.mixture, "_exhausted", [False] * len(epochs))
        active = [j for j in range(len(epochs)) if not exhausted[j]]
        if not active:
            return True
        return all(epochs[j] >= self._epochs_at_last_doc[j] + 2 for j in active)

    def __next__(self) -> dict[str, Any]:
        while not self._queue:
            if self._exhausted:
                raise StopIteration
            try:
                name, ex = next(self.mixture)
            except StopIteration:
                self._exhausted = True
                flushed = self.packer.flush()
                if flushed is not None:
                    self._queue.append(flushed)
                    break
                raise StopIteration from None
            accepted = self._process(name, ex)
            if not accepted:
                self._draws_since_doc += 1
                if self._all_sources_wrapped_without_progress():
                    log.warning(
                        "[data] all active sources wrapped an entire epoch with 0 newly accepted docs "
                        "(split=%s, draws_since_doc=%d, docs=%d, dup=%d, decontam=%d); flushing buffered tokens",
                        self.split_role,
                        self._draws_since_doc,
                        self.counters["docs"],
                        self.counters["duplicates"],
                        self.counters["decontaminated"],
                    )
                    self._exhausted = True
                    flushed = self.packer.flush()
                    if flushed is not None:
                        self._queue.append(flushed)
                        break
                    raise StopIteration
        row = self._queue.pop(0)
        self.counters["rows"] += 1
        if self.log_every and self.counters["rows"] % self.log_every == 0:
            self.log_stats()
        return row

    def log_stats(self) -> None:
        c = self.counters
        log.info("[data] rows %d | docs %d | fill %.3f | loss-token frac %.3f | dup %d | decontam %d | mix %s",
                 c["rows"], c["docs"], self.packer.fill_ratio, c["loss_tokens"] / max(1, c["doc_tokens"]),
                 c["duplicates"], c["decontaminated"],
                 {k: round(v["frac"], 3) for k, v in self.mixture.stats().items()})
        self.drops.log_summary()
        if self.decontam is not None:
            self.decontam.log_summary()

    def stats(self) -> dict[str, Any]:
        return {"counters": dict(self.counters), "fill_ratio": self.packer.fill_ratio,
                "mixture": self.mixture.stats(), "drops": self.drops.summary(),
                "decontam": self.decontam.summary() if self.decontam is not None else None,
                "split_role": self.split_role, "holdout_mod": self.holdout_mod,
                "holdout_eval_buckets": self.holdout_eval_buckets}

    def state_dict(self) -> dict[str, Any]:
        return {
            "mixture": self.mixture.state_dict(),
            "packer": self.packer.state_dict(),
            "queue": [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()} for r in self._queue],
            "doc_counter": self.doc_counter,
            "seen": sorted(self._seen) if self.dedupe else [],
            "counters": dict(self.counters),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        import torch

        self.mixture.load_state_dict(sd["mixture"])
        self.packer.load_state_dict(sd["packer"])
        self._queue = [{k: (torch.tensor(v, dtype=torch.long) if isinstance(v, list) else v) for k, v in r.items()}
                       for r in sd.get("queue", [])]
        self.doc_counter = int(sd.get("doc_counter", 0))
        self._seen = set(int(x) for x in sd.get("seen", []))
        self.counters = collections.Counter(sd.get("counters", {}))
        self._exhausted = False
        self._epochs_at_last_doc = list(getattr(self.mixture, "epoch", []))
        self._draws_since_doc = 0
        log.info("[data] resumed stream: %d rows emitted, %d docs, %d buffered tokens, mixture draws %d",
                 self.counters["rows"], self.doc_counter, len(self.packer._ids), self.mixture.draws)


def _as_plain(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _as_plain(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_as_plain(v) for v in d]
    return d


_SRC_RUNTIME_KEYS = {
    "name", "weight", "max_rows", "toy_rows", "bad_frac", "data_dir", "shuffle_buffer", "token",
    "split_role", "holdout_mod", "holdout_eval_buckets",
}


def resolve_source_spec(src: dict[str, Any]) -> DatasetSpec:
    """Registry spec for a source entry, with per-source overrides (format, fields, split, ...)."""
    name = src.get("name")
    require(bool(name), "data.sources entry needs a `name`", entry=src)
    overrides = {k: v for k, v in src.items() if k not in _SRC_RUNTIME_KEYS}
    if name in REGISTRY:
        spec = get_spec(name)
        return spec.with_overrides(**overrides) if overrides else spec
    require("hf_id" in src, f"source '{name}' is not in the registry; give `hf_id` (+ format/role) to use an ad-hoc "
            f"dataset. Known: {sorted(REGISTRY)}", entry=src)
    return DatasetSpec(name=name, **overrides)


def build_train_stream(data_cfg: dict[str, Any], tokenizer: Any, seed: int = 0,
                       eval_providers: dict[str, EvalTextProvider] | None = None, *,
                       split_role: str | None = None, holdout_mod: int | None = None,
                       holdout_eval_buckets: int | None = None, single_pass: bool = False,
                       skip_empty_sources: bool = False) -> TrainStream:
    """Build the packed training stream from the `data:` config section (or a cfg containing it).

    eval_providers: {group: callable -> iterable of eval texts}; overrides registry eval sets (tests).
    split_role: "all" (default, backward compatible) | "train" | "eval" for deterministic document-level
    SHA-1 hash holdout partitioning before packing (`bucket < holdout_eval_buckets` -> eval).
    """
    cfg = _as_plain(data_cfg)
    if "sources" not in cfg and isinstance(cfg.get("data"), dict):
        cfg = cfg["data"]
    with stage("data:build"):
        require(isinstance(cfg.get("sources"), list) and cfg["sources"], "data.sources must be a non-empty list",
                data=cfg)
        seq_len = int(cfg.get("seq_len", 4096))
        streaming = bool(cfg.get("streaming", True))
        global_max = cfg.get("max_rows_per_source")
        eff_role = str(split_role if split_role is not None else cfg.get("split_role", "all"))
        eff_mod = int(holdout_mod if holdout_mod is not None else cfg.get("holdout_mod", 100))
        eff_eval_buckets = int(
            holdout_eval_buckets if holdout_eval_buckets is not None else cfg.get("holdout_eval_buckets", 2)
        )
        require(eff_role in SPLIT_ROLES, f"split_role must be one of {SPLIT_ROLES}", split_role=eff_role)
        if eff_role != "all" and eff_mod > 0:
            require(0 <= eff_eval_buckets < eff_mod,
                    "holdout_eval_buckets must satisfy 0 <= holdout_eval_buckets < holdout_mod",
                    holdout_mod=eff_mod, holdout_eval_buckets=eff_eval_buckets)
        drops = DropCounter(max_format_error_frac=float(cfg.get("max_format_error_frac", 0.5)))
        sources: list[tuple[str, float, SourceFactory]] = []
        for k, src in enumerate(cfg["sources"]):
            spec = resolve_source_spec(src)
            require(spec.role in ("sft", "pretrain"), f"source '{spec.name}' has role '{spec.role}': only sft/pretrain "
                    "datasets can be packed into the LM training stream (rl/eval go through their own loaders)",
                    spec=spec)
            max_rows = src.get("max_rows", global_max)
            src_seed = int(seed) * 1009 + k
            s_role = str(src.get("split_role", eff_role))
            s_mod = int(src.get("holdout_mod", eff_mod))
            s_eval = int(src.get("holdout_eval_buckets", eff_eval_buckets))
            fac = make_source_factory(
                spec, src_seed, drops, src, None if max_rows is None else int(max_rows), streaming,
                split_role=s_role, holdout_mod=s_mod, holdout_eval_buckets=s_eval,
            )
            sources.append((spec.name, float(src.get("weight", 1.0)), fac))
            log.info("[data] source %-16s w=%.3f %s role=%s format=%s split=%s(%d/%d)%s%s", spec.name,
                     float(src.get("weight", 1.0)), spec.hf_id, spec.role, spec.format, s_role, s_eval, s_mod,
                     " gated" if spec.gated else "", "" if spec.verified else " (UNVERIFIED spec)")
        mixture = MixtureStream(
            sources,
            seed=seed,
            single_pass=single_pass,
            skip_empty_sources=skip_empty_sources,
        )

        dc = cfg.get("decontaminate") or {}
        decontaminator = None
        if dc.get("enabled", False):
            groups = list(dc.get("groups") or [])
            providers = dict(eval_providers or {})
            missing = [g for g in groups if g not in providers]
            if missing:
                providers.update(default_eval_providers(missing, on_missing=str(dc.get("on_missing", "error"))))
            with stage("data:decontam-index"):
                decontaminator = Decontaminator.from_providers(
                    {g: providers[g] for g in (groups or list(providers))}, n=int(dc.get("ngram", 13)),
                    on_missing=str(dc.get("on_missing", "error")))
        stream = TrainStream(
            mixture, tokenizer, seq_len, bool(cfg.get("respect_doc_boundaries", True)), decontaminator, drops,
            dedupe=bool(cfg.get("dedupe", True)), keep_history_reasoning=bool(cfg.get("keep_history_reasoning", False)),
            log_every=int(cfg.get("log_every", 1000)), max_doc_tokens=cfg.get("max_doc_tokens"),
            split_role=eff_role, holdout_mod=eff_mod, holdout_eval_buckets=eff_eval_buckets,
        )
    return stream
