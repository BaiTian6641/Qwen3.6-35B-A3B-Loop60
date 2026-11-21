"""N-gram decontamination against held-out eval sets.

Eval texts are lower-cased and split into word tokens (`\\w+`); every n-gram (default 13, as in
GPT-3 / Llama decontamination) is hashed into an exact in-memory set (no Bloom filter: eval sets are
small, so false positives are not worth the memory saving). A training example is contaminated if
ANY of its n-grams is in the set.

Two refinements:
  * low-diversity n-grams (fewer than `min_unique` distinct words, e.g. "0 0 0 0 ..." in ARC grids or
    repeated boilerplate) are ignored on both sides, otherwise they flag everything.
  * eval texts shorter than n words (short AIME problems / answers) are matched as whole k-grams for
    their own length k if they have at least `short_min_words` words.

Hashes use Python's tuple hash (process-local, fast); the index is rebuilt at startup and never saved.
"""
from __future__ import annotations

import collections
import logging
import re
from collections.abc import Iterable, Iterator
from typing import Any, Callable

from qloop.debug import require

log = logging.getLogger("qloop")

_WORD_RE = re.compile(r"\w+", re.UNICODE)

EvalTextProvider = Callable[[], Iterable[str]]


def normalize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def ngram_hashes(words: list[str], n: int, min_unique: int = 0) -> Iterator[int]:
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        if min_unique and len(set(gram)) < min_unique:
            continue
        yield hash(gram)


class Decontaminator:
    def __init__(self, n: int = 13, min_unique: int | None = None, short_min_words: int = 8):
        require(n >= 2, "Decontaminator: ngram must be >= 2", n=n)
        self.n = n
        self.min_unique = max(3, n // 3) if min_unique is None else min_unique
        self.short_min_words = short_min_words
        self._grams: dict[int, str] = {}
        self._short: dict[int, dict[int, str]] = collections.defaultdict(dict)  # length -> hash -> group
        self.texts_per_group: collections.Counter[str] = collections.Counter()
        self.grams_per_group: collections.Counter[str] = collections.Counter()
        self.hits: collections.Counter[tuple[str, str]] = collections.Counter()  # (group, source)
        self.checked: collections.Counter[str] = collections.Counter()

    # ---- building
    def add_texts(self, group: str, texts: Iterable[str]) -> int:
        """Index eval texts under `group`. Returns number of texts indexed."""
        n_txt = 0
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                continue
            words = normalize_words(t)
            if len(words) >= self.n:
                for h in ngram_hashes(words, self.n, self.min_unique):
                    if h not in self._grams:
                        self._grams[h] = group
                        self.grams_per_group[group] += 1
            elif len(words) >= self.short_min_words:
                self._short[len(words)][hash(tuple(words))] = group
                self.grams_per_group[group] += 1
            else:
                continue
            n_txt += 1
        self.texts_per_group[group] += n_txt
        return n_txt

    @classmethod
    def from_providers(cls, providers: dict[str, EvalTextProvider], n: int = 13, on_missing: str = "error",
                       **kw: Any) -> Decontaminator:
        """Build from {group: callable -> iterable of eval texts}. A provider that raises or yields
        nothing is an error (silent contamination is worse than a crash) unless on_missing='warn'."""
        require(on_missing in ("error", "warn"), "decontaminate.on_missing must be 'error' or 'warn'",
                on_missing=on_missing)
        d = cls(n=n, **kw)
        for group, prov in providers.items():
            try:
                cnt = d.add_texts(group, prov())
            except Exception as e:
                msg = (f"decontam group '{group}': eval text provider failed: {type(e).__name__}: {e}. "
                       "Gated eval sets need HF_TOKEN with accepted terms; or set decontaminate.on_missing=warn.")
                if on_missing == "error":
                    raise RuntimeError(msg) from e
                log.warning(msg)
                continue
            if cnt == 0:
                msg = f"decontam group '{group}': provider returned no usable texts"
                if on_missing == "error":
                    raise RuntimeError(msg + " (set decontaminate.on_missing=warn to continue)")
                log.warning(msg)
            log.info("[decontam] group %-10s %6d texts -> %8d %d-grams", group, cnt, d.grams_per_group[group], n)
        return d

    # ---- querying
    def is_contaminated(self, text: str) -> tuple[bool, str | None]:
        words = normalize_words(text)
        n_words = len(words)
        grams = self._grams
        if grams and n_words >= self.n:
            n = self.n
            min_u = self.min_unique
            grams_get = grams.get
            # Short-circuit: only n-grams with >= min_unique distinct words are indexed,
            # so probe the hash table BEFORE allocating a Python set(gram) (~4x faster, exact).
            for i in range(n_words - n + 1):
                gram = tuple(words[i : i + n])
                g = grams_get(hash(gram))
                if g is not None and (not min_u or len(set(gram)) >= min_u):
                    return True, g
        for k, table in self._short.items():
            if n_words < k:
                continue
            table_get = table.get
            for i in range(n_words - k + 1):
                g = table_get(hash(tuple(words[i : i + k])))
                if g is not None:
                    return True, g
        return False, None

    def check(self, text: str, source: str = "?") -> tuple[bool, str | None]:
        """is_contaminated + per-(group, source) counters (logged by log_summary)."""
        self.checked[source] += 1
        hit, group = self.is_contaminated(text)
        if hit:
            assert group is not None
            self.hits[(group, source)] += 1
            c = self.hits[(group, source)]
            if c <= 3 or c in (10, 100, 1000) or c % 10000 == 0:
                log.info("[decontam] %s: %d rows removed so far (matched eval group '%s')", source, c, group)
        return hit, group

    @property
    def size(self) -> int:
        return len(self._grams) + sum(len(t) for t in self._short.values())

    def summary(self) -> dict[str, Any]:
        return {"index_size": self.size, "texts_per_group": dict(self.texts_per_group),
                "hits": {f"{g}<-{s}": v for (g, s), v in sorted(self.hits.items())},
                "checked": dict(self.checked)}

    def log_summary(self) -> None:
        s = self.summary()
        log.info("[decontam] index %d hashes; texts %s; hits %s", s["index_size"], s["texts_per_group"], s["hits"])
