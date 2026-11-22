"""Track B Streaming Dataset Audit, 3-Way Triage, Salvage, Active Synthesis & HF Publishing Engine.

Implements the end-to-end Loop64-Fable-SIRSI dataset pipeline:
  1. Upstream Source & Citation Registry (`UPSTREAM_SOURCES`, `EXCLUDED_SOURCES`) with live
     Hugging Face license enforcement (permissive licenses only: apache-2.0, cc-by-4.0, mit, odc-by).
  2. Streaming Quality Judge & 3-Way Triage (`KEEP` / `SALVAGE` / `DROP`):
     - Completeness & structural integrity (prompt/think/answer, balanced code fences, tool JSON,
       truncation detection).
     - Pathology & collapse detection (empty/redacted/encrypted thinking, n-gram & line repetition
       loops, refusals/incomplete prompts, corrupted text).
     - Anti-slop & voice scoring (sycophancy, AI clichés, bold-bullet template spam, distinct-2/3).
     - Looped-CoT 3-beat structural scoring (Framing -> Adversarial Self-Audit -> Restraint).
     - Domain sanity checks (Python AST syntax, math answer extraction, ARC grid validation,
       agentic tool JSON & placeholder-redaction detection).
     - Cross-source exact SHA-256 + banded MinHash/Jaccard near-deduplication.
     - Strict 13-gram decontamination reusing `qloop.data.decontam.Decontaminator` across 2026
       eval benchmarks with persistent disk caching.
  3. Deterministic Salvage Pass with answer-invariance verification and post-salvage re-judging.
  4. Track B Async Active Synthesis & Rewrite Engine (`hf_router`, `gemini`, `openai`, `openrouter`,
     `local`) with resumable JSONL checkpointing and post-synthesis verification gates.
  5. Publication-grade Hugging Face Dataset Card (`README.md`) renderer, Parquet shard writer, and
     safe `--dry-run` Hub publishing manifest validator.
"""
from __future__ import annotations

import ast
import asyncio
import collections
import gzip
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qloop.data.decontam import Decontaminator, normalize_words
from qloop.data.formats import (
    Example,
    RowDrop,
    RowFormatError,
    arc_task_prompt,
    extract_think,
    has_unclosed_think,
    looks_encrypted,
    normalize_row,
    validate_example,
)
from qloop.data.registry import REGISTRY, DatasetSpec, get_spec
from qloop.eval.scoring import (
    extract_math_answer,
    grid_equal,
    math_equal,
    parse_grid,
)

log = logging.getLogger("qloop.sirsi_dataset")

# ============================================================================ License Policy & Citations

ALLOWED_LICENSES: frozenset[str] = frozenset({"apache-2.0", "cc-by-4.0", "mit", "odc-by"})


@dataclass(frozen=True)
class SourceCitation:
    name: str
    hf_id: str
    config: str | None
    split: str
    domain: str
    license: str
    included: bool
    exclusion_reason: str
    citation_key: str
    title: str
    authors: str
    year: int
    url: str
    bibtex: str
    attribution_note: str
    jsonl_fallback_path: str | None = None


AUTHOR_ORIGINAL_CITATION = SourceCitation(
    name="sirsi_golden_seeds",
    hf_id="TassP0lar/Loop64-Fable-SIRSI-v1",
    config="golden_seeds",
    split="train",
    domain="mixed",
    license="apache-2.0",
    included=True,
    exclusion_reason="",
    citation_key="author_original",
    title="Loop64-Fable-SIRSI-v1: Curated Anti-Slop 3-Beat Looped-CoT, Aesthetic Craft, and SI-RSI Recovery Dataset",
    authors="Loop64 Research Team (TassP0lar)",
    year=2026,
    url="https://huggingface.co/datasets/TassP0lar/Loop64-Fable-SIRSI-v1",
    bibtex=(
        "@dataset{loop64_fable_sirsi_2026,\n"
        "  title     = {Loop64-Fable-SIRSI-v1: Curated Anti-Slop 3-Beat Looped-CoT, Aesthetic Craft, "
        "and SI-RSI Recovery Dataset},\n"
        "  author    = {TassP0lar},\n"
        "  year      = {2026},\n"
        "  publisher = {Hugging Face},\n"
        "  url       = {https://huggingface.co/datasets/TassP0lar/Loop64-Fable-SIRSI-v1}\n"
        "}"
    ),
    attribution_note="Original human-authored golden seeds and Track B verified rewrites/syntheses (Apache-2.0).",
)

_ALL_SOURCE_LIST: list[SourceCitation] = [
    SourceCitation(
        name="fable51",
        hf_id="MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x",
        config=None,
        split="train",
        domain="agentic",
        license="apache-2.0",
        included=True,
        exclusion_reason="",
        citation_key="morethought2026fable51",
        title="Fable-5.1-Max-Reasoning-Filtered-10000x",
        authors="MoreThought",
        year=2026,
        url="https://huggingface.co/datasets/MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x",
        bibtex=(
            "@dataset{morethought2026fable51,\n"
            "  title     = {Fable-5.1-Max-Reasoning-Filtered-10000x},\n"
            "  author    = {MoreThought},\n"
            "  year      = {2026},\n"
            "  publisher = {Hugging Face},\n"
            "  url       = {https://huggingface.co/datasets/MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x}\n"
            "}"
        ),
        attribution_note=(
            "Apache-2.0. Uses the filtered `train` split (`5k/train.jsonl`) which retains <thought> and "
            "<final_answer> blocks; note that upstream tool_call arguments are placeholder-redacted "
            "('<file_path>', '<bash_command>') and routed to salvage_queue rather than raw tool-call SFT."
        ),
        jsonl_fallback_path="5k/train.jsonl",
    ),
    SourceCitation(
        name="ot_agent_sft",
        hf_id="open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K",
        config=None,
        split="train",
        domain="agentic",
        license="apache-2.0",
        included=True,
        exclusion_reason="",
        citation_key="openthoughts_agent_2026",
        title="OpenThoughts-Agent: Data Recipes for Agentic Models",
        authors="OpenThoughts-Agent Team",
        year=2026,
        url="https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K",
        bibtex=(
            "@misc{openthoughts_agent_2026,\n"
            "  author       = {Team, OpenThoughts-Agent},\n"
            "  title        = {{OpenThoughts-Agent: Data Recipes for Agentic Models}},\n"
            "  howpublished = {https://www.openthoughts.ai/blog/agent},\n"
            "  year         = {2026}\n"
            "}"
        ),
        attribution_note="Apache-2.0. Multi-turn terminus-2 containerized agentic traces with inline <think>.",
    ),
    SourceCitation(
        name="openthoughts3",
        hf_id="open-thoughts/OpenThoughts3-1.2M",
        config=None,
        split="train",
        domain="reasoning",
        license="apache-2.0",
        included=True,
        exclusion_reason="",
        citation_key="guha2025openthoughts",
        title="OpenThoughts: Data Recipes for Reasoning Models",
        authors="Etash Guha, Ryan Marten, Sedrick Keh, et al.",
        year=2025,
        url="https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M",
        bibtex=(
            "@misc{guha2025openthoughts,\n"
            "  title         = {OpenThoughts: Data Recipes for Reasoning Models},\n"
            "  author        = {Etash Guha and Ryan Marten and Sedrick Keh and Negin Raoof and Georgios Smyrnis "
            "and Hritik Bansal and Marianna Nezhurina and Jean Mercat and Trung Vu and Zayne Sprague and "
            "Alexandros G. Dimakis and Ludwig Schmidt},\n"
            "  year          = {2025},\n"
            "  eprint        = {2506.04178},\n"
            "  archivePrefix = {arXiv},\n"
            "  primaryClass  = {cs.LG},\n"
            "  url           = {https://arxiv.org/abs/2506.04178}\n"
            "}"
        ),
        attribution_note="Apache-2.0. Large-scale math, code, and science reasoning corpus (streamed).",
    ),
    SourceCitation(
        name="nemotron_post",
        hf_id="nvidia/Llama-Nemotron-Post-Training-Dataset",
        config="SFT",
        split="math+code+science",
        domain="math_code_science",
        license="cc-by-4.0",
        included=True,
        exclusion_reason="",
        citation_key="bercovich2025llamanemotron",
        title="Llama-Nemotron: Efficient Reasoning Models",
        authors="Akhiad Bercovich, Itay Levy, Izik Golan, et al. (NVIDIA)",
        year=2025,
        url="https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset",
        bibtex=(
            "@misc{bercovich2025llamanemotron,\n"
            "  title         = {Llama-Nemotron: Efficient Reasoning Models},\n"
            "  author        = {Akhiad Bercovich and Itay Levy and Izik Golan and Mohammad Dabbah and "
            "Ran El-Yaniv and Boris Ginsburg and Jonathan Cohen and Bryan Catanzaro},\n"
            "  year          = {2025},\n"
            "  eprint        = {2505.00949},\n"
            "  archivePrefix = {arXiv},\n"
            "  primaryClass  = {cs.CL},\n"
            "  url           = {https://arxiv.org/abs/2505.00949}\n"
            "}"
        ),
        attribution_note=(
            "CC-BY-4.0 (attribution required). SFT config splits math+code+science; Llama-Nemotron "
            "'detailed thinking on/off' system prompts are deterministically stripped."
        ),
    ),
    SourceCitation(
        name="opus46_3000x",
        hf_id="nohurry/Opus-4.6-Reasoning-3000x-filtered",
        config=None,
        split="train",
        domain="reasoning",
        license="apache-2.0",
        included=True,
        exclusion_reason="",
        citation_key="nohurry2026opus46filtered",
        title="Opus-4.6-Reasoning-3000x-filtered",
        authors="nohurry (filtered from Crownelius)",
        year=2026,
        url="https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered",
        bibtex=(
            "@dataset{nohurry2026opus46filtered,\n"
            "  title     = {Opus-4.6-Reasoning-3000x-filtered},\n"
            "  author    = {nohurry},\n"
            "  year      = {2026},\n"
            "  publisher = {Hugging Face},\n"
            "  url       = {https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered}\n"
            "}"
        ),
        attribution_note=(
            "Apache-2.0. Refusal-filtered subset of Opus-4.6-Reasoning; deduplicated against opus46_3300x."
        ),
    ),
    SourceCitation(
        name="opus46_3300x",
        hf_id="Crownelius/Opus-4.6-Reasoning-3300x",
        config=None,
        split="train",
        domain="reasoning",
        license="apache-2.0",
        included=True,
        exclusion_reason="",
        citation_key="crownelius2026opus46",
        title="Opus-4.6-Reasoning-3300x",
        authors="Crownelius",
        year=2026,
        url="https://huggingface.co/datasets/Crownelius/Opus-4.6-Reasoning-3300x",
        bibtex=(
            "@dataset{crownelius2026opus46,\n"
            "  title     = {Opus-4.6-Reasoning-3300x},\n"
            "  author    = {Crownelius},\n"
            "  year      = {2026},\n"
            "  publisher = {Hugging Face},\n"
            "  url       = {https://huggingface.co/datasets/Crownelius/Opus-4.6-Reasoning-3300x}\n"
            "}"
        ),
        attribution_note=(
            "Apache-2.0. Cleaned Opus-4.6 reasoning traces; cross-source deduplicated against opus46_3000x."
        ),
    ),
    SourceCitation(
        name="opus46_traceinv",
        hf_id="Jackrong/Claude-opus-4.6-TraceInversion-9000x",
        config=None,
        split="train",
        domain="reasoning",
        license="apache-2.0",
        included=True,
        exclusion_reason="",
        citation_key="jackrong2026traceinv",
        title="Claude-opus-4.6-TraceInversion-9000x: Reconstructed CoT via Trace Inversion",
        authors="Jackrong (method: Tingwei Zhang, John X. Morris, Vitaly Shmatikov)",
        year=2026,
        url="https://huggingface.co/datasets/Jackrong/Claude-opus-4.6-TraceInversion-9000x",
        bibtex=(
            "@dataset{jackrong2026traceinv,\n"
            "  title     = {Claude-opus-4.6-TraceInversion-9000x: A High-Fidelity Reconstructed CoT Dataset "
            "via Trace Inversion},\n"
            "  author    = {Jackrong},\n"
            "  year      = {2026},\n"
            "  publisher = {Hugging Face},\n"
            "  url       = {https://huggingface.co/datasets/Jackrong/Claude-opus-4.6-TraceInversion-9000x}\n"
            "}\n"
            "@misc{zhang2026stealreasoning,\n"
            "  title  = {How to Steal Reasoning Without Reasoning Traces},\n"
            "  author = {Tingwei Zhang and John X. Morris and Vitaly Shmatikov},\n"
            "  year   = {2026},\n"
            "  note   = {Preprint}\n"
            "}"
        ),
        attribution_note=(
            "Apache-2.0. Multilingual trace-inverted reasoning traces; heavily screened for thin/short CoT."
        ),
    ),
    SourceCitation(
        name="arc_agi2_train",
        hf_id="github:arcprize/ARC-AGI-2",
        config=None,
        split="training",
        domain="arc",
        license="apache-2.0",
        included=True,
        exclusion_reason="",
        citation_key="chollet2025arcagi2",
        title="ARC-AGI-2: Abstract Reasoning Corpus (Training Split Only)",
        authors="ARC Prize Foundation (Francois Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers)",
        year=2025,
        url="https://github.com/arcprize/ARC-AGI-2",
        bibtex=(
            "@misc{chollet2025arcagi2,\n"
            "  title        = {ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems},\n"
            "  author       = {Francois Chollet and Mike Knoop and Gregory Kamradt and Bryan Landers},\n"
            "  year         = {2025},\n"
            "  howpublished = {https://github.com/arcprize/ARC-AGI-2}\n"
            "}"
        ),
        attribution_note=(
            "Apache-2.0. Strictly training split only (`data/training/*.json`); evaluation split held out "
            "and indexed inside the 13-gram decontaminator. Raw grid I/O pairs without CoT are routed to "
            "salvage_queue for Track B 3-beat rule synthesis."
        ),
    ),
    # ------------------------------------------------------------ Explicitly Excluded Sources (License Hygiene)
    SourceCitation(
        name="agentic_fable5",
        hf_id="lordx64/agentic-distill-fable-5-sft",
        config=None,
        split="train",
        domain="agentic",
        license="agpl-3.0",
        included=False,
        exclusion_reason=(
            "license_incompatible:agpl-3.0 "
            "(strong copyleft incompatible with permissive Apache-2.0/CC-BY-4.0 release)"
        ),
        citation_key="lordx642026agenticfable5",
        title="agentic-distill-fable-5-sft (EXCLUDED: AGPL-3.0)",
        authors="lordx64",
        year=2026,
        url="https://huggingface.co/datasets/lordx64/agentic-distill-fable-5-sft",
        bibtex="",
        attribution_note="EXCLUDED from dataset build due to AGPL-3.0 license.",
    ),
    SourceCitation(
        name="fable5_distill",
        hf_id="HelioAI/Fable-5-Distill-Reasoning-462x",
        config=None,
        split="train",
        domain="reasoning",
        license="unknown",
        included=False,
        exclusion_reason="license_incompatible:unknown (no permissive redistribution license declared on HF Hub)",
        citation_key="helioai2026fable5distill",
        title="Fable-5-Distill-Reasoning-462x (EXCLUDED: unknown license)",
        authors="HelioAI Labs",
        year=2026,
        url="https://huggingface.co/datasets/HelioAI/Fable-5-Distill-Reasoning-462x",
        bibtex="",
        attribution_note="EXCLUDED from dataset build due to unknown/unspecified license.",
    ),
]

ALL_AUDITED_SOURCES: dict[str, SourceCitation] = {s.name: s for s in _ALL_SOURCE_LIST}
UPSTREAM_SOURCES: dict[str, SourceCitation] = {s.name: s for s in _ALL_SOURCE_LIST if s.included}
EXCLUDED_SOURCES: dict[str, SourceCitation] = {s.name: s for s in _ALL_SOURCE_LIST if not s.included}


def _normalize_license_id(raw: Any) -> str:
    if raw is None:
        return "unknown"
    if isinstance(raw, list):
        vals = [str(x).strip().lower() for x in raw if str(x).strip()]
        return vals[0] if vals else "unknown"
    s = str(raw).strip().lower()
    if s.startswith("license:"):
        s = s.split(":", 1)[1].strip()
    return s or "unknown"


def verify_upstream_licenses(
    sources: Sequence[SourceCitation] | None = None,
    *,
    token: str | None = None,
    live: bool = True,
    api: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify upstream source licenses (live via HfApi / GitHub when live=True) and enforce ALLOWED_LICENSES.

    Returns {source_name: {hf_id, declared_license, live_license, allowed, included, reason, sha}}.
    """
    srcs = list(ALL_AUDITED_SOURCES.values()) if sources is None else list(sources)
    hf_api = api
    if live and hf_api is None:
        from huggingface_hub import HfApi

        hf_api = HfApi(token=token or os.environ.get("HF_TOKEN"))

    results: dict[str, dict[str, Any]] = {}
    for src in srcs:
        declared = _normalize_license_id(src.license)
        live_lic = declared
        sha = ""
        probe_err = ""
        if live and not src.hf_id.startswith("builtin:"):
            if src.hf_id.startswith("github:"):
                import urllib.request

                repo = src.hf_id.split(":", 1)[1]
                try:
                    req = urllib.request.Request(
                        f"https://api.github.com/repos/{repo}",
                        headers={"User-Agent": "qloop-sirsi-audit"},
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        info = json.loads(resp.read())
                    lic_obj = info.get("license") or {}
                    live_lic = _normalize_license_id(lic_obj.get("key") or lic_obj.get("spdx_id") or declared)
                    sha = str(info.get("default_branch") or "main")
                except Exception as exc:
                    probe_err = f"github_probe_failed:{type(exc).__name__}"
            else:
                try:
                    info = hf_api.dataset_info(src.hf_id)
                    cd = info.card_data.to_dict() if getattr(info, "card_data", None) else {}
                    raw_lic = cd.get("license")
                    if not raw_lic:
                        for tag in getattr(info, "tags", None) or []:
                            if str(tag).lower().startswith("license:"):
                                raw_lic = tag
                                break
                    live_lic = _normalize_license_id(raw_lic)
                    sha = str(getattr(info, "sha", "") or "")[:12]
                except Exception as exc:
                    probe_err = f"hf_probe_failed:{type(exc).__name__}"
                    live_lic = "unknown"

        allowed = (live_lic in ALLOWED_LICENSES) and (declared in ALLOWED_LICENSES)
        effective_included = bool(src.included and allowed)
        if not src.included:
            reason = src.exclusion_reason or f"license_incompatible:{live_lic}"
        elif not allowed:
            reason = f"license_incompatible:live={live_lic},declared={declared}"
            if probe_err:
                reason += f" ({probe_err})"
        else:
            reason = "ok" + (f" ({probe_err}; used declared)" if probe_err else "")

        results[src.name] = {
            "name": src.name,
            "hf_id": src.hf_id,
            "declared_license": declared,
            "live_license": live_lic,
            "allowed": allowed,
            "included": effective_included,
            "reason": reason,
            "sha": sha,
        }
    return results


# ============================================================================ Quality Judges & Scoring

# AI-slop lexical crutches, sycophancy openers, boilerplate closers
_SYCOPHANTIC_OPENERS: list[re.Pattern[str]] = [
    re.compile(
        r"^\s*(?:certainly|of course|absolutely|great question|"
        r"excellent question|that's a great question)[!.,:\s]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:sure|happy to help)[!,.\s]+"
        r"(?:i(?:'d| would) be (?:happy|delighted|glad) to [^.!?\n]{0,80}[.!?]\s*)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*i(?:'d| would) be (?:happy|delighted|glad) to "
        r"(?:help|assist|walk you through)[^.!?\n]{0,80}[.!?]\s*",
        re.IGNORECASE,
    ),
]

_BOILERPLATE_CLOSERS: list[re.Pattern[str]] = [
    re.compile(
        r"\n*\s*(?:i\s+)?hope this helps[!.]*"
        r"(?:\s*(?:please\s+)?(?:let me know|feel free to ask)[^.!?\n]{0,120}[.!?]?)?\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"\n*\s*(?:please\s+)?feel free to (?:ask|reach out)[^.!?\n]{0,120}[.!?]?\s*$", re.IGNORECASE),
    re.compile(
        r"\n*\s*let me know if you (?:have any (?:other |further )?questions|"
        r"need anything else|would like me to)[^.!?\n]{0,120}[.!?]?\s*$",
        re.IGNORECASE,
    ),
]

_SLOP_PHRASES: list[tuple[str, re.Pattern[str], float]] = [
    ("delve", re.compile(r"\bdelv(?:e|es|ing|ed)\b", re.IGNORECASE), 0.18),
    ("tapestry", re.compile(r"\btapestry\b", re.IGNORECASE), 0.22),
    ("testament_to", re.compile(r"\btestament to\b", re.IGNORECASE), 0.16),
    (
        "important_to_note",
        re.compile(r"\bit(?:'s| is) (?:important|worth|crucial|essential) to note\b", re.IGNORECASE),
        0.16,
    ),
    ("in_conclusion", re.compile(r"(?:^|\n|\.\s+)in (?:conclusion|summary)\s*,", re.IGNORECASE), 0.14),
    (
        "lets_break_down",
        re.compile(r"\blet(?:'s| us) (?:break this down|dive into|explore this|unpack this)\b", re.IGNORECASE),
        0.16,
    ),
    (
        "ever_evolving",
        re.compile(
            r"\bever[- ]evolving\b|\bin today'?s (?:fast[- ]paced|modern|digital) (?:world|landscape|era)\b",
            re.IGNORECASE,
        ),
        0.22,
    ),
    ("game_changer", re.compile(r"\bgame[- ]changer\b|\bseamless(?:ly)? integrat", re.IGNORECASE), 0.18),
    (
        "vibrant_landscape",
        re.compile(r"\bvibrant (?:landscape|ecosystem|community)\b|\bembark on\b", re.IGNORECASE),
        0.18,
    ),
    ("comprehensive_overview", re.compile(r"\bcomprehensive (?:overview|guide|breakdown)\b", re.IGNORECASE), 0.14),
    (
        "sycophantic_opener",
        re.compile(
            r"^\s*(?:certainly|of course|absolutely|great question|sure[,!]\s+i(?:'d| would) be happy)",
            re.IGNORECASE,
        ),
        0.20,
    ),
    (
        "boilerplate_closer",
        re.compile(
            r"(?:hope this helps|feel free to ask|let me know if you have any (?:other |further )?questions)",
            re.IGNORECASE,
        ),
        0.14,
    ),
]

_BOLD_BULLET_LINE_RE = re.compile(
    r"^\s*(?:[-*•]|\d+\.)\s+\*\*[^*\n]{2,70}(?:[:—\-]\*\*|\*\*\s*[:—\-])"
)

_REFUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bas an ai(?: language model)?\b", re.IGNORECASE),
    re.compile(r"\bi cannot (?:fulfill|comply with|assist with|help with) (?:this|that) request\b", re.IGNORECASE),
    re.compile(r"\bi(?:'m| am) sorry,?\s+but i (?:can't|cannot)\b", re.IGNORECASE),
    re.compile(
        r"\bi (?:can't|cannot) (?:help with this request|provide a solution without|"
        r"answer this question because it lacks)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhaven't\s+(?:actually\s+)?(?:included|provided|given|shared)\s+(?:a|an|the)?\s*"
        r"(?:actual|complete|full)?\s*(?:problem|question|number puzzle)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:missing|no|without|lacks)\s+(?:a|an|the|any)?\s*(?:actual|specific|complete|full)?\s*"
        r"(?:problem|question)\s*(?:statement|description|was provided|was included|to solve|included)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the |your )?(?:problem|question)(?:\s+(?:statement|description))?\s+"
        r"(?:you've provided\s+)?(?:appears to be |appears |is )?(?:missing|incomplete|truncated|"
        r"not provided|isn't provided)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:\*\*)?(?:partial|incomplete|fragment(?:\*\*)?\s+of\s+(?:a|what appears to be))\s+"
        r"(?:problem\s+(?:statement|description)|solution explanation)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdon't\s+(?:have|see)\s+(?:a|an|the)\s+(?:complete|full|actual)\s+(?:problem|question)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:please|could you(?:\s+please)?)\s+(?:provide|share)\s+(?:the\s+)?"
        r"(?:complete|full|actual|specific)\s+(?:problem|question|input|details)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno (?:actual |specific )?(?:problem|question)\s+(?:statement\s+)?"
        r"(?:was |has been )?(?:provided|included)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)#{1,4}\s*(?:incomplete problem statement|missing (?:information|question))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bappears to be part of a larger problem\b.*\b(?:need the full|full context|complete problem)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\bi(?:'m| am) not able to help with this request\b|\bthere(?:'s| is) no actual problem\b|"
        r"\byour message appears to be incomplete\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno (?:actual|specific) problem (?:has been (?:given|stated|provided|included)|"
        r"or [^.\n]{1,40} (?:was|were) provided)\b",
        re.IGNORECASE,
    ),
]

_REDACTED_TOOL_ARG_RE = re.compile(
    r"<(?:file_path|directory_path|bash_command|file_content|old_string|new_string|"
    r"search_query|search_pattern|shell_command|dir_path|target_path|source_path|redacted)>",
    re.IGNORECASE,
)


def _has_redacted_tool_arg_value(obj: Any) -> bool:
    """Recursively check parsed JSON leaf strings for exact placeholder-redacted tool arguments."""
    if isinstance(obj, str):
        return bool(_REDACTED_TOOL_ARG_RE.fullmatch(obj.strip()))
    if isinstance(obj, dict):
        return any(_has_redacted_tool_arg_value(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_redacted_tool_arg_value(v) for v in obj)
    return False

_FINAL_ANSWER_TAG_RE = re.compile(r"^\s*<final_answer>\s*(.*?)\s*</final_answer>\s*$", re.DOTALL | re.IGNORECASE)

_PYTHON_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# 3-Beat Looped-CoT structural signal patterns
_BEAT1_RE = re.compile(
    r"\b(?:given|goal|constraint|invariant|let\s+[a-z]\b|define|suppose|notice|observe|framing|"
    r"decompose|we want|we need to|first,|step 1|approach|hypothesis|structure|requirement|input|problem asks)\b",
    re.IGNORECASE,
)
_BEAT2_RE = re.compile(
    r"\b(?:wait|hold on|check|verify|verification|double[- ]check|sanity check|substitute|"
    r"counterexample|edge case|boundary|contradiction|pitfall|actually|mistake|error|"
    r"let(?:'s| us) test|plug(?:ging)? (?:in|back)|confirm|re-?check|does this hold|"
    r"what if|adversarial|audit|backtrack|careful|watch out|off-by-one|special case)\b",
    re.IGNORECASE,
)
_BEAT3_SYNTH_RE = re.compile(
    r"(?:\\boxed\{|```|final answer|therefore|hence|thus|result:|solution|output|\banswer\b)",
    re.IGNORECASE,
)


def compute_distinct_n(words: Sequence[str], n: int) -> float:
    if len(words) < n:
        return 1.0
    total = len(words) - n + 1
    uniq = len({tuple(words[i : i + n]) for i in range(total)})
    return float(uniq) / float(total)


def score_anti_slop(response: str, reasoning: str = "") -> dict[str, Any]:
    """Score AI-slop clichés, sycophancy, bold-bullet template spam, and lexical diversity."""
    resp = response or ""
    flags: list[str] = []
    score = 0.0

    for name, pat, weight in _SLOP_PHRASES:
        hits = pat.findall(resp)
        if hits:
            flags.append(f"slop:{name}({len(hits)})")
            score += min(weight * (1.0 + 0.35 * (len(hits) - 1)), weight * 2.0)

    # Bold-bullet template spam: "- **Title:** ..." dominating the response
    nonempty_lines = [ln for ln in resp.splitlines() if ln.strip()]
    bold_bullet_lines = [ln for ln in nonempty_lines if _BOLD_BULLET_LINE_RE.match(ln)]
    bold_bullet_ratio = len(bold_bullet_lines) / max(1, len(nonempty_lines))
    if len(bold_bullet_lines) >= 6 and bold_bullet_ratio >= 0.40:
        flags.append(f"slop:bold_bullet_wall({len(bold_bullet_lines)}/{len(nonempty_lines)})")
        score += 0.30

    resp_words = normalize_words(resp)
    all_words = normalize_words(f"{reasoning}\n{resp}") if reasoning else resp_words
    d2 = compute_distinct_n(resp_words, 2) if len(resp_words) >= 12 else compute_distinct_n(all_words, 2)
    d3 = compute_distinct_n(resp_words, 3) if len(resp_words) >= 15 else compute_distinct_n(all_words, 3)

    if len(resp_words) >= 80 and (d2 < 0.32 or d3 < 0.42):
        flags.append(f"slop:low_diversity(d2={d2:.2f},d3={d3:.2f})")
        score += 0.25

    return {
        "slop_score": round(min(1.0, score), 4),
        "distinct_2": round(d2, 4),
        "distinct_3": round(d3, 4),
        "bold_bullet_ratio": round(bold_bullet_ratio, 4),
        "slop_flags": flags,
    }


def score_three_beat_cot(reasoning: str, response: str, prompt: str = "") -> dict[str, Any]:
    """Score Looped-CoT 3-beat structure: Framing (Beat 1) -> Self-Audit (Beat 2) -> Restraint (Beat 3)."""
    r = (reasoning or "").strip()
    c = (response or "").strip()
    if not r:
        return {
            "beat1_score": 0.0,
            "beat2_score": 0.0,
            "beat3_score": 0.5 if c else 0.0,
            "three_beat_score": 0.0,
            "has_beat2_audit": False,
        }

    r_words = normalize_words(r)
    c_words = normalize_words(c)

    # Beat 1: problem framing / decomposition / hypothesis
    head = r[: max(400, len(r) // 3)]
    b1_hits = len(_BEAT1_RE.findall(head))
    beat1 = min(1.0, 0.30 + 0.20 * b1_hits) if len(r_words) >= 20 else min(0.5, 0.15 * b1_hits)

    # Beat 2: adversarial check / verification / backtrack / edge-case testing
    b2_hits = len(_BEAT2_RE.findall(r))
    has_beat2 = b2_hits >= 1 and len(r_words) >= 25
    beat2 = min(1.0, 0.45 + 0.15 * b2_hits) if has_beat2 else (0.15 if b2_hits >= 1 else 0.0)

    # Beat 3: crisp synthesis / restraint (present answer, not extreme bloat or verbatim copy)
    if not c:
        beat3 = 0.25  # tool-call-only turn or missing final synthesis
    else:
        ratio = len(c_words) / max(1, len(r_words))
        # Healthy synthesis is substantial without bloating wildly past the reasoning
        if ratio <= 2.5 and len(c_words) >= 3:
            beat3 = 0.85
        elif ratio <= 4.5:
            beat3 = 0.65
        else:
            beat3 = 0.40
        if _BEAT3_SYNTH_RE.search(c) or len(c_words) <= 400:
            beat3 = min(1.0, beat3 + 0.15)

    total = 0.30 * beat1 + 0.45 * beat2 + 0.25 * beat3
    return {
        "beat1_score": round(beat1, 4),
        "beat2_score": round(beat2, 4),
        "beat3_score": round(beat3, 4),
        "three_beat_score": round(total, 4),
        "has_beat2_audit": bool(has_beat2),
    }


def _has_repetition_loop(text: str) -> bool:
    """Detect degenerate n-gram or line repetition loops without false-positiving on code/tables."""
    if not text or len(text) < 200:
        return False

    # 1. Consecutive identical non-trivial lines (>= 4 times)
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 16]
    run = 1
    for i in range(1, len(lines)):
        if lines[i] == lines[i - 1] and not set(lines[i]).issubset({"-", "=", "*", "|", " ", "`"}):
            run += 1
            if run >= 4:
                return True
        else:
            run = 1

    # 2. Repeated 10-word n-gram (>= 6 occurrences) AND low tail diversity
    words = normalize_words(text)
    if len(words) >= 120:
        # Cap scan to head+tail window if huge to stay fast in streaming
        scan = words if len(words) <= 6000 else (words[:2000] + words[-4000:])
        counts: collections.Counter[tuple[str, ...]] = collections.Counter()
        for i in range(len(scan) - 9):
            g = tuple(scan[i : i + 10])
            if len(set(g)) >= 4:
                counts[g] += 1
                if counts[g] >= 6:
                    # Verify it's truly repetitive in a local tail or significant share
                    tail = words[-200:]
                    if compute_distinct_n(tail, 4) < 0.45 or counts[g] >= 10:
                        return True

    # 3. Single-token/character runaway (e.g. "wait wait wait..." or "aaaaaa..." >= 40 repeats)
    return bool(re.search(r"(\b\w{2,15}\b)(?:\s+\1){9,}", text, re.IGNORECASE))


def _is_mid_sentence_truncated(text: str) -> bool:
    """Heuristic for abruptly cut-off generation (unclosed fence/brace or mid-clause drop)."""
    s = (text or "").rstrip()
    if len(s) < 300:
        return False
    # Ends cleanly with sentence/markdown/code/math/JSON/XML terminator
    if s[-1] in ".!?:;\"'`)]}>0123456789$%":
        # Check for trailing dangling backslash or open LaTeX command
        return bool(s.endswith("\\"))
    # Check if ends on a trailing incomplete conjunction/preposition/article
    tail_words = re.findall(r"[A-Za-z']+$", s)
    return bool(
        tail_words
        and tail_words[0].lower() in {
            "the", "and", "or", "but", "because", "since", "with", "from", "into",
            "that", "which", "where", "when", "for", "to", "of", "in", "a", "an",
        }
    )


def detect_pathology(
    prompt: str,
    reasoning: str,
    response: str,
    messages: list[dict[str, Any]] | None = None,
    *,
    allow_empty_reasoning: bool = False,
) -> list[str]:
    """Detect hard pathologies and structural defects."""
    issues: list[str] = []
    p = (prompt or "").strip()
    r = (reasoning or "").strip()
    c = (response or "").strip()

    if not p:
        issues.append("pathology:empty_prompt")

    # Collect all assistant turns for multi-turn agentic traces
    asst_turns = [m for m in (messages or []) if m.get("role") == "assistant"]
    all_reasonings = [
        str(m.get("reasoning") or "").strip()
        for m in asst_turns
        if str(m.get("reasoning") or "").strip()
    ]
    if r and not all_reasonings:
        all_reasonings = [r]
    has_any_tool_calls = any(m.get("tool_calls") for m in asst_turns)

    if not c and not has_any_tool_calls:
        issues.append("pathology:empty_response")

    if not allow_empty_reasoning:
        if not all_reasonings:
            issues.append("pathology:reasoning_empty")
        else:
            for rt in all_reasonings:
                enc = looks_encrypted(rt)
                if enc:
                    issues.append(f"pathology:{enc}")
                    break

    if has_unclosed_think(c) or any(has_unclosed_think(str(m.get("content") or "")) for m in asst_turns):
        issues.append("pathology:think_unclosed")

    # Balanced triple-backtick code fences in final response
    fence_count = len(re.findall(r"(?m)^\s*```", c))
    if fence_count % 2 == 1:
        issues.append("pathology:unclosed_code_fence")

    # Corrupted bytes / replacement characters
    combined = f"{p}\n{r}\n{c}"
    if "\x00" in combined or combined.count("\ufffd") >= 5:
        issues.append("pathology:corrupted_text")

    # Refusal or missing-problem boilerplate (check generous window of response + reasoning)
    ref_target = f"{c[:2000]}\n{r[:800]}"
    for pat in _REFUSAL_PATTERNS:
        if pat.search(ref_target):
            issues.append("pathology:refusal_or_incomplete_prompt")
            break

    # Repetition / collapse loops (skip pure digit-grid responses where identical blank rows are valid pixels)
    is_digit_grid = bool(c) and all(
        set(ln.strip()).issubset(set("0123456789 ")) for ln in c.splitlines() if ln.strip()
    )
    if _has_repetition_loop(r) or (not is_digit_grid and _has_repetition_loop(c)):
        issues.append("pathology:repetition_loop")

    # Abrupt mid-sentence truncation on long response
    if _is_mid_sentence_truncated(c):
        issues.append("pathology:mid_sentence_truncation")

    return issues


# Synthetic Track B CoT artifact detection & rubric-header stripping
_RUBRIC_BEAT_LINE_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:(?:[-*•]|\d+\.)[ \t]+)?(?:#{1,4}[ \t]+)?\*{0,2}Beat[ \t]*[123]\b",
    re.IGNORECASE,
)
_RUBRIC_BEAT_INLINE_RE = re.compile(
    r"(?:^|[.!?\s])(?<!\bon )(?<!\bat )\*{0,2}Beat[ \t]*[123](?:[ \t]*\([^)\n]{1,60}\))?\*{0,2}[ \t]*:",
    re.IGNORECASE,
)
_RUBRIC_EXPLICIT_HEADER_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:(?:[-*•]|\d+\.)[ \t]+)?(?:#{1,4}[ \t]+)?\*{0,2}"
    r"(?:Adversarial[ \t]+Self[- ]Audit|Synthesis[ \t]+Plan|"
    r"Restraint[ \t]*(?:&|and)[ \t]*Synthesis|Framing[ \t]*(?:&|and)[ \t]*Invariants)"
    r"\*{0,2}[ \t]*(?::|[—–]|-[ \t]+|(?=\n|$))",
    re.IGNORECASE,
)
_RUBRIC_FRAMING_HDR_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:(?:[-*•]|\d+\.)[ \t]+)?(?:#{1,4}[ \t]+)?\*{0,2}Framing\*{0,2}[ \t]*(?::|(?=\n|$))",
    re.IGNORECASE,
)
_RUBRIC_AUDIT_OR_SYNTH_HDR_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:(?:[-*•]|\d+\.)[ \t]+)?(?:#{1,4}[ \t]+)?\*{0,2}"
    r"(?:(?:Adversarial[ \t]+)?Self[- ]Audit|Synthesis(?:[ \t]+Plan)?)\*{0,2}[ \t]*(?::|(?=\n|$))",
    re.IGNORECASE,
)

# Negative lookahead preventing false positives on organic problem-format recaps like
# "The final answer is an integer between 0 and 999." while still catching MCQ "The final answer is A."
_SPOILER_FORMAT_RECAP_GUARD = (
    r"(?![ \t]+(?:"
    r"an?[ \t]+(?:(?:positive|negative|non[- ]?negative|non[- ]?zero|reduced|simplified|irreducible|"
    r"exact|real|complex|rational|irrational|whole|natural|even|odd|prime|\d+[- ]digit|single|unique|"
    r"valid|minimal|maximal|finite|closed[- ]form|mixed|proper|improper|radical)[ \t]+)*"
    r"(?:integer|fraction|decimal|number|value|expression|polynomial|matrix|grid|string|boolean|"
    r"tuple|list|set|pair|triple|ratio|percentage|probability|angle|length|area|volume|count|"
    r"sum|product|multiple|power|root|function|sequence|element|coordinate|vector|solution|"
    r"positive|negative|non[- ]?negative|\d+[- ]digit)\b"
    r"|to[ \t]+be\b|expected\b|required\b|requested\b|supposed\b|expressed\b|written\b|reported\b|"
    r"stated\b|given[ \t]+(?:in|as|mod|modulo)\b|formatted\b|rounded\b|measured\b|bounded\b|"
    r"constrained\b|limited\b|defined\b|between\b|modulo\b|mod\b|divisible\b|strictly\b|"
    r"greater\b|less\b|at[ \t]+(?:least|most)\b|of[ \t]+the[ \t]+form\b|"
    r"in[ \t]+(?:the[ \t]+form|terms[ \t]+of|exact|reduced|lowest|simplest)\b|not[ \t]+)"
    r")"
)
_SPOILER_LABEL_HEAD = (
    r"(?:(?:final|target)[ \t]+"
    r"(?:answer|result|output|verdict|solution|outcomes?|translation|implementation|summary)"
    r"\*{0,2}[ \t]*:|the[ \t]+(?:final|target)[ \t]+answer[ \t]+is\b" + _SPOILER_FORMAT_RECAP_GUARD + r")"
)
_SPOILER_FIRST_RE = re.compile(
    r"(?:^[ \t]*(?:[-*•][ \t]+)?(?:#{1,4}[ \t]+)?"
    r"(?:(?:okay|ok|alright|well|so|thus|therefore|hence|clearly|note|wait)[,:\s]+)?"
    r"\*{0,2}(?:(?:answer|conclusion|outcome)\*{0,2}[ \t]*:|" + _SPOILER_LABEL_HEAD + r"))"
    r"|(?:(?:^|[\n.,:;(\[—–-])\s*\*{0,2}" + _SPOILER_LABEL_HEAD + r")"
    r"|(?:(?:^|\n)[ \t]*\*{0,2}(?:answer|conclusion|outcome)\*{0,2}[ \t]*:)",
    re.IGNORECASE,
)

_META_STUB_PHRASES: list[re.Pattern[str]] = [
    re.compile(r"\bstandard\s+processing\s+applied\b", re.IGNORECASE),
    re.compile(
        r"\binitial\b(?:[^.\n]|\.\d){0,110}\b(?:parameters?|constraints?)"
        r"(?:[^.\n]|\.\d){0,90}\b(?:are|is|were)\s+(?:established\s+and\s+)?validated\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:are|is|were)\s+(?:established\s+and\s+validated|"
        r"validated\s+as\s+established\s+context|treated\s+as\s+validated\s+established)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bconfirms?\s+(?:that\s+)?there\s+are\s+no\s+hidden\s+"
        r"(?:fees|traps|edge\s+cases|errors|issues)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bthe\s+(?:arithmetic|mathematical|computational)\s+logic\s+holds\s+cleanly\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:verify(?:ing)?\s+the\s+result\s+by\s+substituting\s+back\s+and\s+)?"
        r"running\s+an?\s+edge[- ]case\s+check\s+on\s+the\s+problem\s+wording\b",
        re.IGNORECASE,
    ),
]
_META_STUB_SHORT_PHRASES: list[re.Pattern[str]] = [
    re.compile(r"\ball\s+constraints\s+and\s+invariants\s+(?:are|remain)\s+satisfied\b", re.IGNORECASE),
    re.compile(
        r"\bverify(?:ing)?\s+the\s+result\s+by\s+substituting\s+back\s+and\s+running\s+an?\s+edge[- ]case\s+check\b",
        re.IGNORECASE,
    ),
]
_ORDINAL_LABEL_RE = re.compile(r"\b(?:beat|step|phase|part)\s+[1-9]\b", re.IGNORECASE)
_CONCRETE_COMPUTE_TOKEN_RE = re.compile(
    r"`[^`\n]{2,}`"
    r"|(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])"
    r"|[A-Za-z0-9_)\]]\s*(?:==|!=|<=|>=|->|=>|=)\s*[A-Za-z0-9_(\[\-\\]"
    r"|\\[A-Za-z]+\{"
)

_RUBRIC_KNOWN_SUBTITLE = (
    r"(?:Framing(?:[ \t]*(?:&|and)[ \t]*Invariants)?|"
    r"(?:Adversarial[ \t]+)?Self[- ]Audit|"
    r"Verified[ \t]+Recovery|"
    r"Restraint(?:[ \t]*(?:&|and)[ \t]*Synthesis)?|"
    r"Synthesis(?:[ \t]+Plan)?|"
    r"Audit|Verification)"
)
_RUBRIC_PREFIX_SEP = r"(?::\*{0,2}|[—–]\*{0,2}|-\*{0,2}[ \t]+)[ \t]*"
_STRIP_BEAT_STANDALONE_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:[-*•]|\d+\.)[ \t]+)?(?:#{1,4}[ \t]+)?\*{0,2}Beat[ \t]*[123]"
    r"(?:[ \t]*\([^)\n]{1,60}\)|[ \t]*[:—–-][ \t]*" + _RUBRIC_KNOWN_SUBTITLE + r")?"
    r"\*{0,2}[ \t]*[:.]?\*{0,2}[ \t]*(?:\n|$)"
)
_STRIP_BEAT_LINE_PREFIX_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:[-*•]|\d+\.)[ \t]+)?(?:#{1,4}[ \t]+)?\*{0,2}Beat[ \t]*[123]"
    r"(?:[ \t]*\([^)\n]{1,60}\)|[ \t]*[—–-][ \t]*" + _RUBRIC_KNOWN_SUBTITLE + r")?"
    r"\*{0,2}[ \t]*" + _RUBRIC_PREFIX_SEP
)
_STRIP_BEAT_MD_PREFIX_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:[-*•]|\d+\.)[ \t]+)?#{1,4}[ \t]+\*{0,2}Beat[ \t]*[123]"
    r"(?:[ \t]*\([^)\n]{1,60}\))?\*{0,2}[ \t]*" + _RUBRIC_PREFIX_SEP
)
_STRIP_BEAT_INLINE_PREFIX_RE = re.compile(
    r"(?i)(?<=[.!?\s])(?<!\bon )(?<!\bat )\*{0,2}Beat[ \t]*[123]"
    r"(?:[ \t]*\([^)\n]{1,60}\))?\*{0,2}[ \t]*:\*{0,2}[ \t]*"
)
_STRIP_EXPLICIT_STANDALONE_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:[-*•]|\d+\.)[ \t]+)?(?:#{1,4}[ \t]+)?\*{0,2}"
    r"(?:Adversarial[ \t]+Self[- ]Audit|Synthesis[ \t]+Plan|"
    r"Restraint[ \t]*(?:&|and)[ \t]*Synthesis|Framing[ \t]*(?:&|and)[ \t]*Invariants)"
    r"\*{0,2}[ \t]*[:.]?\*{0,2}[ \t]*(?:\n|$)"
)
_STRIP_EXPLICIT_PREFIX_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:[-*•]|\d+\.)[ \t]+)?(?:#{1,4}[ \t]+)?\*{0,2}"
    r"(?:Adversarial[ \t]+Self[- ]Audit|Synthesis[ \t]+Plan|"
    r"Restraint[ \t]*(?:&|and)[ \t]*Synthesis|Framing[ \t]*(?:&|and)[ \t]*Invariants)"
    r"\*{0,2}[ \t]*" + _RUBRIC_PREFIX_SEP
)
_STRIP_PAIRED_HDR_STANDALONE_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:[-*•]|\d+\.)[ \t]+)?(?:#{1,4}[ \t]+)?\*{0,2}"
    r"(?:Framing|(?:Adversarial[ \t]+)?Self[- ]Audit|Synthesis(?:[ \t]+Plan)?)"
    r"\*{0,2}[ \t]*[:.]?\*{0,2}[ \t]*(?:\n|$)"
)
_STRIP_PAIRED_HDR_PREFIX_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:[-*•]|\d+\.)[ \t]+)?(?:#{1,4}[ \t]+)?\*{0,2}"
    r"(?:Framing|(?:Adversarial[ \t]+)?Self[- ]Audit|Synthesis(?:[ \t]+Plan)?)"
    r"\*{0,2}[ \t]*" + _RUBRIC_PREFIX_SEP
)


def detect_synthetic_cot_artifacts(thinking: str, response: str = "", domain: str = "") -> list[str]:
    """Detect synthetic CoT defects: rubric headers, spoiler-first openings, and meta-summary stubs."""
    _ = (response, domain)
    t = (thinking or "").strip()
    if not t:
        return []

    issues: list[str] = []

    # 1. Robotic rubric headers (`Beat 1/2/3`, `Framing:` + `Adversarial Self-Audit:`, `Synthesis Plan:`, etc.)
    has_rubric_header = bool(
        _RUBRIC_BEAT_LINE_RE.search(t)
        or _RUBRIC_BEAT_INLINE_RE.search(t)
        or _RUBRIC_EXPLICIT_HEADER_RE.search(t)
        or (_RUBRIC_FRAMING_HDR_RE.search(t) and _RUBRIC_AUDIT_OR_SYNTH_HDR_RE.search(t))
    )
    if has_rubric_header:
        issues.append("cot:rubric_header")

    # 2. Spoiler-first rationalization in the opening 60 chars of `<think>`
    m_spoiler = _SPOILER_FIRST_RE.search(t[:160])
    if m_spoiler is not None and m_spoiler.start() < 60:
        issues.append("cot:spoiler_first")

    # 3. Rubber-stamp meta-summary stubs instead of concrete step-by-step computation
    has_meta_stub = any(pat.search(t) for pat in _META_STUB_PHRASES)
    if not has_meta_stub and any(pat.search(t) for pat in _META_STUB_SHORT_PHRASES):
        word_count = len(normalize_words(t))
        stripped_ord = _ORDINAL_LABEL_RE.sub("", t)
        compute_tokens = len(_CONCRETE_COMPUTE_TOKEN_RE.findall(stripped_ord))
        if word_count < 140 and compute_tokens <= 2:
            has_meta_stub = True
    if has_meta_stub:
        issues.append("cot:meta_summary_stub")

    return issues


def strip_rubric_headers(thinking: str) -> str:
    """Remove literal `Beat 1/2/3 (...)` and rubric section labels from `thinking` without corrupting prose."""
    if not thinking or not thinking.strip():
        return ""

    out = thinking
    had_rubric_context = bool(
        _RUBRIC_BEAT_LINE_RE.search(out)
        or _RUBRIC_BEAT_INLINE_RE.search(out)
        or _RUBRIC_EXPLICIT_HEADER_RE.search(out)
        or (_RUBRIC_FRAMING_HDR_RE.search(out) and _RUBRIC_AUDIT_OR_SYNTH_HDR_RE.search(out))
    )

    out = _STRIP_BEAT_STANDALONE_RE.sub("", out)
    out = _STRIP_BEAT_LINE_PREFIX_RE.sub("", out)
    out = _STRIP_BEAT_MD_PREFIX_RE.sub("", out)
    out = _STRIP_BEAT_INLINE_PREFIX_RE.sub("", out)
    out = _STRIP_EXPLICIT_STANDALONE_RE.sub("", out)
    out = _STRIP_EXPLICIT_PREFIX_RE.sub("", out)

    if had_rubric_context:
        out = _STRIP_PAIRED_HDR_STANDALONE_RE.sub("", out)
        out = _STRIP_PAIRED_HDR_PREFIX_RE.sub("", out)

    if out == thinking:
        return thinking

    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _infer_sub_domain(base_domain: str, prompt: str, response: str, raw_meta: dict[str, Any] | None = None) -> str:
    meta = raw_meta or {}
    cat = str(meta.get("category") or meta.get("domain") or meta.get("split") or "").lower()
    if base_domain in ("arc", "agentic", "aesthetic", "recovery"):
        return base_domain
    if "math" in cat or "aime" in cat or "algebra" in cat or "geometry" in cat:
        return "math"
    if "code" in cat or "coding" in cat or "programming" in cat:
        return "code"
    if "science" in cat or "physics" in cat or "chem" in cat or "bio" in cat:
        return "science"
    if "```python" in response.lower():
        return "code"
    if "\\boxed{" in response or re.search(
        r"\b(?:prove|evaluate|compute|solve for|find the value)\b", prompt, re.IGNORECASE
    ):
        return "math"
    return base_domain or "reasoning"


def _check_python_blocks(response: str) -> list[str]:
    """Validate ```python``` fenced blocks via ast.parse, skipping REPL/diff/pseudo-code."""
    issues: list[str] = []
    for match in _PYTHON_FENCE_RE.finditer(response or ""):
        code = match.group(1).strip()
        lines = [ln for ln in code.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        # Skip interactive REPL transcripts or unified diffs or shell commands
        if any(ln.lstrip().startswith((">>>", "...", "$ ", "+ ", "- ", "@@")) for ln in lines[:4]):
            continue
        # Skip snippets with explicit placeholder ellipses or angle-bracket template tokens
        if "<your_" in code.lower() or "# ..." in code:
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            issues.append("domain:python_syntax_error")
            break
    return issues


def _parse_arc_grid(response: str) -> list[list[int]] | None:
    """Parse ARC output grid, including valid 1x1 single-digit output grids."""
    grid = parse_grid(response)
    if grid is None and re.fullmatch(r"\s*[0-9]\s*", response or ""):
        return [[int(response.strip())]]
    return grid


def check_domain_sanity(
    domain: str,
    prompt: str,
    reasoning: str,
    response: str,
    messages: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str], str]:
    """Domain-specific sanity checks.

    Returns (hard_drop_issues, salvage_flags, extracted_answer).
    """
    hard_issues: list[str] = []
    salvage_flags: list[str] = []
    extracted_answer = ""

    # 1. Python syntax verification on final code blocks
    hard_issues.extend(_check_python_blocks(response))

    # 2. ARC grid validation (including valid 1x1 single-digit output grids)
    if domain == "arc":
        grid = _parse_arc_grid(response)
        if grid is None:
            hard_issues.append("domain:arc_grid_invalid")
        else:
            extracted_answer = json.dumps(grid)

    # 3. Math answer presence & canonical extraction
    if domain == "math":
        ans = extract_math_answer(response)
        if ans:
            extracted_answer = ans
        else:
            # Pure proof questions don't require \boxed{}
            is_proof = bool(
                re.search(
                    r"\b(?:prove|show that|demonstrate that|sketch the graph|explain why)\b",
                    prompt,
                    re.IGNORECASE,
                )
            )
            if not is_proof and "\\boxed" in prompt.lower():
                salvage_flags.append("salvage:math_missing_boxed_answer")

    # 4. Agentic tool JSON & placeholder-redaction validation
    if messages:
        has_redacted_tc = False
        has_invalid_tc = False
        for m in messages:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc) if isinstance(tc, dict) else None
                if not isinstance(fn, dict) or not fn.get("name"):
                    has_invalid_tc = True
                    continue
                args = fn.get("arguments")
                if isinstance(args, str):
                    s_args = args.strip()
                    if s_args:
                        try:
                            parsed = json.loads(s_args)
                            if not isinstance(parsed, (dict, list)):
                                has_invalid_tc = True
                            elif _has_redacted_tool_arg_value(parsed):
                                has_redacted_tc = True
                        except json.JSONDecodeError:
                            has_invalid_tc = True
                elif isinstance(args, (dict, list)):
                    if _has_redacted_tool_arg_value(args):
                        has_redacted_tc = True
                elif args is not None:
                    has_invalid_tc = True

        if has_invalid_tc:
            hard_issues.append("domain:invalid_tool_json")
        if has_redacted_tc:
            salvage_flags.append("salvage:redacted_tool_args")

        # Terminus-2 style JSON assistant actions (ot_agent_sft)
        if domain == "agentic" and response.lstrip().startswith("{") and response.rstrip().endswith("}"):
            try:
                obj = json.loads(response)
                if not isinstance(obj, dict):
                    hard_issues.append("domain:invalid_tool_json")
            except json.JSONDecodeError:
                hard_issues.append("domain:invalid_tool_json")

    return hard_issues, salvage_flags, extracted_answer


# ============================================================================ Cross-Source Deduplication

class CrossSourceDeduplicator:
    """Exact SHA-256 + banded MinHash / shingle Jaccard near-deduplicator."""

    def __init__(
        self,
        *,
        shingle_size: int = 3,
        num_hashes: int = 64,
        num_bands: int = 16,
        jaccard_threshold: float = 0.85,
        prompt_jaccard_threshold: float = 0.90,
        min_prompt_chars_exact: int = 60,
    ):
        self.shingle_size = shingle_size
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = max(1, num_hashes // num_bands)
        self.jaccard_threshold = jaccard_threshold
        self.prompt_jaccard_threshold = prompt_jaccard_threshold
        self.min_prompt_chars_exact = min_prompt_chars_exact

        # Deterministic affine hash coefficients for MinHash (61-bit Mersenne prime)
        self._prime = (1 << 61) - 1
        self._coeffs: list[tuple[int, int]] = [
            (((i + 1) * 0x9E3779B185EBCA87) % (self._prime - 1) + 1, ((i + 1) * 0xC2B2AE3D27D4EB4F) % self._prime)
            for i in range(num_hashes)
        ]

        self._seen_pair_sha: dict[str, str] = {}     # sha -> first source
        self._seen_prompt_sha: dict[str, str] = {}   # sha -> first source
        self._buckets: dict[tuple[int, tuple[int, ...]], list[int]] = collections.defaultdict(list)
        self._records: list[tuple[str, frozenset[int], frozenset[int], frozenset[int]]] = []
        self.dup_counts: collections.Counter[str] = collections.Counter()
        self.overlap_matrix: collections.Counter[tuple[str, str]] = collections.Counter()  # (dup_src, orig_src)

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join((text or "").lower().split())

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()

    def _shingles(self, text: str) -> frozenset[int]:
        words = normalize_words(text)
        k = self.shingle_size
        if len(words) < k:
            return frozenset()
        # Preserve head + tail up to 1,600 words so shared agentic preambles don't eclipse task bodies
        if len(words) > 1600:
            words = words[:800] + words[-800:]
        out: set[int] = set()
        for i in range(len(words) - k + 1):
            gram = " ".join(words[i : i + k]).encode("utf-8", "replace")
            out.add(int.from_bytes(hashlib.blake2b(gram, digest_size=8).digest(), "little"))
        return frozenset(out)

    def _minhash(self, shingles: frozenset[int]) -> tuple[int, ...]:
        p = self._prime
        sig: list[int] = []
        for a, b in self._coeffs:
            m = p
            for x in shingles:
                v = (a * x + b) % p
                m = min(m, v)
            sig.append(m)
        return tuple(sig)

    @staticmethod
    def _jaccard(a: frozenset[int], b: frozenset[int]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a) + len(b) - inter
        return float(inter) / float(union) if union else 0.0

    def check_and_add(self, source: str, prompt: str, response: str) -> tuple[bool, str, str | None]:
        """Return (is_dup, dup_reason, original_source)."""
        np = self._norm(prompt)
        nr = self._norm(response)

        pair_sha = self._sha256(f"{np}\x00{nr}")
        if pair_sha in self._seen_pair_sha:
            orig = self._seen_pair_sha[pair_sha]
            self.dup_counts["dup:exact_sha256"] += 1
            self.overlap_matrix[(source, orig)] += 1
            return True, "dup:exact_sha256", orig

        prompt_sha = self._sha256(np) if len(np) >= self.min_prompt_chars_exact else ""
        if prompt_sha and prompt_sha in self._seen_prompt_sha:
            orig = self._seen_prompt_sha[prompt_sha]
            self.dup_counts["dup:exact_prompt_sha256"] += 1
            self.overlap_matrix[(source, orig)] += 1
            return True, "dup:exact_prompt_sha256", orig

        p_shingles = self._shingles(np)
        r_shingles = self._shingles(nr)
        if not r_shingles and nr:
            r_shingles = frozenset(
                int.from_bytes(hashlib.blake2b(w.encode("utf-8", "replace"), digest_size=8).digest(), "little")
                for w in normalize_words(nr)[:200]
            )
        f_shingles = self._shingles(f"{np} {nr}")
        short_prompt = 12 <= len(p_shingles) <= 350
        sig_shingles = p_shingles if short_prompt else (r_shingles if len(r_shingles) >= 12 else f_shingles)

        if len(sig_shingles) >= 8:
            sig = self._minhash(sig_shingles)
            candidates: set[int] = set()
            rpb = self.rows_per_band
            for band in range(self.num_bands):
                key = (band, sig[band * rpb : (band + 1) * rpb])
                for idx in self._buckets.get(key, ()):
                    candidates.add(idx)

            for idx in candidates:
                orig_src, orig_p, orig_r, orig_f = self._records[idx]
                p_sim = (
                    self._jaccard(p_shingles, orig_p)
                    if (short_prompt and len(orig_p) >= 12)
                    else 0.0
                )
                r_sim = self._jaccard(r_shingles, orig_r) if (r_shingles and orig_r) else 0.0
                f_sim = self._jaccard(f_shingles, orig_f)
                # For long harness-wrapped prompts (>350 shingles), require responses to also match
                long_prompt_match = (not short_prompt) and f_sim >= self.jaccard_threshold and r_sim >= 0.50
                short_prompt_match = short_prompt and (
                    p_sim >= self.prompt_jaccard_threshold or f_sim >= self.jaccard_threshold
                )
                if short_prompt_match or long_prompt_match:
                    self.dup_counts["dup:near_jaccard"] += 1
                    self.overlap_matrix[(source, orig_src)] += 1
                    return True, "dup:near_jaccard", orig_src

            rec_idx = len(self._records)
            self._records.append((source, p_shingles, r_shingles, f_shingles))
            for band in range(self.num_bands):
                key = (band, sig[band * rpb : (band + 1) * rpb])
                self._buckets[key].append(rec_idx)

        self._seen_pair_sha[pair_sha] = source
        if prompt_sha:
            self._seen_prompt_sha[prompt_sha] = source
        return False, "", None

    def summary(self) -> dict[str, Any]:
        return {
            "unique_indexed": len(self._seen_pair_sha),
            "dup_counts": dict(self.dup_counts),
            "overlap_matrix": {f"{a}<-{b}": c for (a, b), c in sorted(self.overlap_matrix.items())},
        }


# ============================================================================ Cached 13-Gram Decontamination

# All 2026 eval suites requested for decontamination
DEFAULT_EVAL_SUITES: list[dict[str, Any]] = [
    {
        "name": "hle_diamond", "group": "hle", "hf_id": "cais/hle-diamond",
        "config": None, "split": "test", "fields": ["question"], "parquet_only_cols": True,
    },
    {
        "name": "hle_rolling", "group": "hle", "hf_id": "cais/hle-rolling",
        "config": None, "split": "test", "fields": ["question"], "parquet_only_cols": True,
    },
    {
        "name": "hle_verified", "group": "hle", "hf_id": "lmms-lab/HLE-Verified",
        "config": None, "split": "test", "fields": ["question"], "parquet_only_cols": True,
    },
    {
        "name": "gpqa_diamond", "group": "gpqa", "hf_id": "Idavidrein/gpqa",
        "config": "gpqa_diamond", "split": "train", "fields": ["Question", "Correct Answer"],
    },
    {
        "name": "supergpqa", "group": "supergpqa", "hf_id": "m-a-p/SuperGPQA",
        "config": None, "split": "train", "fields": ["question"], "parquet_only_cols": True,
    },
    {
        "name": "aime_2026", "group": "aime_2026", "hf_id": "MathArena/aime_2026",
        "config": None, "split": "train", "fields": ["problem"],
    },
    {
        "name": "hmmt_feb_2026", "group": "hmmt_2026", "hf_id": "MathArena/hmmt_feb_2026",
        "config": None, "split": "train", "fields": ["problem"],
    },
    {
        "name": "mrcr", "group": "mrcr", "hf_id": "openai/mrcr",
        "config": None, "split": "train", "fields": ["answer"], "parquet_only_cols": True,
    },
    {
        "name": "terminal_bench_4_1", "group": "terminal_bench",
        "hf_id": "harborframework/terminal-bench", "config": None, "split": "test",
        "revision": "v4.1.0", "fields": ["instruction"], "instruction_md_files": True,
    },
    {
        "name": "swe_bench_pro", "group": "swe_bench", "hf_id": "ScaleAI/SWE-bench_Pro",
        "config": None, "split": "test", "fields": ["problem_statement"], "parquet_only_cols": True,
    },
    {
        "name": "deep_swe", "group": "swe_bench", "hf_id": "datacurve/deep-swe",
        "config": None, "split": "test", "fields": ["problem_statement", "question", "instruction"],
    },
    {
        "name": "arc_agi_2", "group": "arc_agi_2", "hf_id": "github:arcprize/ARC-AGI-2",
        "config": None, "split": "evaluation", "fields": [],
    },
    {
        "name": "kernelbench_l3_l4", "group": "kernelbench",
        "hf_id": "ScalingIntelligence/KernelBench", "config": None,
        "split": "level_3+level_4", "fields": ["code"],
    },
]


def strip_eval_boilerplate(group: str, text: str) -> str:
    """Strip shared prompt/license boilerplate from eval texts so 13-gram decontam has zero false positives."""
    if not text:
        return ""
    if group == "arc_agi_2":
        # Drop the shared 23-word ARC wrapper instruction so only actual evaluation grid blocks are indexed
        marker = "Example 1 input:"
        idx = text.find(marker)
        return text[idx:].strip() if idx >= 0 else text.strip()
    if group == "kernelbench":
        # Drop leading `#` copyright / Apache-2.0 / MIT license header lines from PyTorch files
        lines = text.splitlines()
        start = 0
        while start < len(lines) and (not lines[start].strip() or lines[start].lstrip().startswith("#")):
            start += 1
        return "\n".join(lines[start:]).strip()
    return text.strip()


def _load_eval_suite_texts_live(suite: dict[str, Any], token: str | None = None) -> list[str]:
    """Fetch eval texts cleanly using columnar Parquet projection / targeted file reads to avoid giant downloads."""
    hf_id = suite["hf_id"]
    group = str(suite.get("group") or "")
    if hf_id.startswith("github:"):
        from qloop.data.mixture import arc_task_rows

        texts: list[str] = []
        for task in arc_task_rows(suite["split"], repo=hf_id.split(":", 1)[1]):
            for pair in task.get("test", []):
                cleaned = strip_eval_boilerplate(group, arc_task_prompt(task, pair["input"]))
                if cleaned:
                    texts.append(cleaned)
        return texts

    tok = token or os.environ.get("HF_TOKEN")
    revision = suite.get("revision")

    # Fast path 1: Terminal-Bench 4.1 instruction.md files directly (avoids 500MB container tarballs)
    if suite.get("instruction_md_files"):
        from huggingface_hub import HfApi, HfFileSystem

        api = HfApi(token=tok)
        files = api.list_repo_files(hf_id, repo_type="dataset", revision=revision)
        md_files = [f for f in files if f.endswith("/instruction.md") or f == "instruction.md"]
        fs = HfFileSystem(token=tok)
        texts = []
        rev_suffix = f"@{revision}" if revision else ""
        for mf in md_files:
            with fs.open(f"datasets/{hf_id}{rev_suffix}/{mf}", "r", encoding="utf-8", errors="replace") as f:
                t = f.read().strip()
                if t:
                    texts.append(t)
        return texts

    # Fast path 2: PyArrow columnar projection on Parquet files (skips images in HLE & 1.3GB prompts in MRCR)
    if suite.get("parquet_only_cols"):
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, HfFileSystem

        api = HfApi(token=tok)
        files = api.list_repo_files(hf_id, repo_type="dataset", revision=revision)
        pq_files = sorted(f for f in files if f.endswith(".parquet"))
        if pq_files:
            fs = HfFileSystem(token=tok)
            fields = list(suite.get("fields") or [])
            texts = []
            rev_suffix = f"@{revision}" if revision else ""
            for pf_rel in pq_files:
                with fs.open(f"datasets/{hf_id}{rev_suffix}/{pf_rel}", "rb") as fh:
                    pqf = pq.ParquetFile(fh)
                    avail = set(pqf.schema_arrow.names)
                    cols = [c for c in fields if c in avail]
                    if not cols:
                        continue
                    tbl = pqf.read(columns=cols)
                    for col in cols:
                        for val in tbl[col].to_pylist():
                            if isinstance(val, str) and val.strip():
                                texts.append(val.strip())
            if texts:
                return texts

    # Default path: HuggingFace datasets streaming
    import datasets

    fields = list(suite.get("fields") or ["question", "problem", "prompt", "text"])
    texts = []
    for spl in [s.strip() for s in str(suite["split"]).split("+") if s.strip()]:
        kw: dict[str, Any] = {"split": spl, "streaming": True}
        if tok:
            kw["token"] = tok
        if revision:
            kw["revision"] = revision
        ds = datasets.load_dataset(hf_id, suite.get("config"), **kw)
        for row in ds:
            for fld in fields:
                val = row.get(fld)
                if isinstance(val, str) and val.strip():
                    cleaned = strip_eval_boilerplate(group, val)
                    if cleaned:
                        texts.append(cleaned)
    return texts


def build_or_load_decontaminator(
    *,
    cache_dir: str | Path | None = None,
    suites: Sequence[dict[str, Any]] | None = None,
    eval_texts_override: dict[str, Sequence[str]] | None = None,
    n: int = 13,
    token: str | None = None,
    on_missing: str = "warn",
) -> tuple[Decontaminator, dict[str, Any]]:
    """Build or load cached 13-gram Decontaminator across all 2026 eval benchmarks.

    Caches extracted eval texts per suite under `cache_dir` (`<suite>.jsonl.gz`) so Python hash
    salting stays exact across processes while avoiding repeated Hub fetches.
    """
    decontam = Decontaminator(n=n)
    report: dict[str, Any] = {"ngram": n, "cache_dir": str(cache_dir or ""), "suites": {}}

    if eval_texts_override is not None:
        for group, texts in eval_texts_override.items():
            cnt = decontam.add_texts(group, (strip_eval_boilerplate(group, t) for t in texts))
            report["suites"][group] = {
                "group": group,
                "status": "override",
                "texts": cnt,
                "grams": decontam.grams_per_group[group],
            }
        report["index_size"] = decontam.size
        return decontam, report

    cdir = Path(cache_dir) if cache_dir else None
    if cdir is not None:
        cdir.mkdir(parents=True, exist_ok=True)

    suite_list = list(suites) if suites is not None else list(DEFAULT_EVAL_SUITES)
    for suite in suite_list:
        sname = suite["name"]
        group = suite["group"]
        cache_file = (cdir / f"{sname}.jsonl.gz") if cdir is not None else None
        status = "loaded_live"
        texts: list[str] = []

        if cache_file is not None and cache_file.exists():
            try:
                with gzip.open(cache_file, "rt", encoding="utf-8") as gz:
                    texts = [json.loads(line) for line in gz if line.strip()]
                status = "loaded_cache"
            except Exception as exc:
                log.warning("[decontam] corrupt cache %s (%s), refetching", cache_file, exc)
                texts = []

        if not texts:
            try:
                texts = _load_eval_suite_texts_live(suite, token=token)
                status = "loaded_live"
                if cache_file is not None and texts:
                    with gzip.open(cache_file, "wt", encoding="utf-8") as gz:
                        for t in texts:
                            gz.write(json.dumps(t, ensure_ascii=False) + "\n")
            except Exception as exc:
                err_name = type(exc).__name__
                err_msg = str(exc)[:200]
                if "GatedRepoError" in err_name or "gated" in err_msg.lower() or "403" in err_msg:
                    status = f"skipped_gated ({err_name})"
                else:
                    status = f"error ({err_name}: {err_msg})"
                log.warning("[decontam] suite %-20s failed: %s", sname, status)
                if on_missing == "error":
                    raise RuntimeError(f"decontam suite '{sname}' failed: {status}") from exc
                report["suites"][sname] = {"group": group, "hf_id": suite["hf_id"], "status": status, "texts": 0}
                continue

        cnt = decontam.add_texts(group, (strip_eval_boilerplate(group, t) for t in texts))
        report["suites"][sname] = {
            "group": group,
            "hf_id": suite["hf_id"],
            "status": status,
            "texts": cnt,
            "group_grams_total": decontam.grams_per_group[group],
        }
        log.info("[decontam] %-20s %-14s %6d texts (%s)", sname, group, cnt, status)

    report["index_size"] = decontam.size
    report["texts_per_group"] = dict(decontam.texts_per_group)
    report["grams_per_group"] = dict(decontam.grams_per_group)
    if cdir is not None:
        (cdir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return decontam, report


# ============================================================================ Deterministic Salvage Pass

def _extract_canonical_answer(domain: str, response: str) -> str:
    if domain == "arc":
        g = _parse_arc_grid(response)
        return json.dumps(g) if g is not None else ""
    if domain == "math":
        return extract_math_answer(response) or ""
    return ""


def _answers_equivalent(domain: str, before: str, after: str) -> bool:
    if not before:
        return True
    if not after:
        return False
    if domain == "arc":
        try:
            return grid_equal(json.loads(before), json.loads(after))
        except Exception:
            return False
    if domain == "math":
        return math_equal(after, before)
    return before.strip() == after.strip()


def _normalize_whitespace_and_fences(text: str) -> tuple[str, bool]:
    if not text:
        return "", False
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    # Ensure fenced code closing ``` starts on a fresh line when glued to code
    out = re.sub(r"([^\n`])```(?=\s*(?:\n|$))", r"\1\n```", out)
    # Strip trailing whitespace per line and collapse 3+ blank lines to 2
    out = "\n".join(ln.rstrip() for ln in out.splitlines())
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, (out != text)


def deterministic_salvage(
    messages: list[dict[str, Any]],
    domain: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply lossless deterministic cleanup to a conversation without altering verified answers.

    Returns (repaired_messages, salvage_actions).
    """
    actions: list[str] = []
    out_msgs: list[dict[str, Any]] = []

    for m in messages:
        msg = dict(m)
        role = msg.get("role", "")
        content = str(msg.get("content") or "")
        reasoning = msg.get("reasoning")
        reasoning_str = "" if reasoning is None else str(reasoning)

        # 1. Strip Nemotron thinking-mode control system prompts
        if role == "system" and content.strip().lower() in ("detailed thinking on", "detailed thinking off"):
            actions.append("strip_nemotron_thinking_sysprompt")
            continue

        if role == "assistant":
            orig_answer = _extract_canonical_answer(domain, content)

            # 2. Normalize stray <thought> / <thinking> tags into canonical reasoning field
            if re.search(r"<(?:thought|thinking)>", content, re.IGNORECASE):
                r_ext, c_rest = extract_think(content)
                if r_ext is not None:
                    reasoning_str = f"{reasoning_str}\n\n{r_ext}".strip() if reasoning_str.strip() else r_ext
                    content = c_rest
                    actions.append("normalize_thought_tags")

            # 3. Unwrap <final_answer>...</final_answer> wrapper tags (e.g. fable51)
            fa_match = _FINAL_ANSWER_TAG_RE.match(content)
            if fa_match:
                content = fa_match.group(1).strip()
                actions.append("unwrap_final_answer_tag")

            # 4. Strip pure sycophantic openers & boilerplate closers (guarded by answer invariance!)
            cand_content = content
            opener_stripped = False
            for pat in _SYCOPHANTIC_OPENERS:
                m_op = pat.match(cand_content)
                if m_op and len(cand_content) - m_op.end() >= 20:
                    cand_content = cand_content[m_op.end() :].lstrip()
                    opener_stripped = True
                    break

            closer_stripped = False
            for pat in _BOILERPLATE_CLOSERS:
                new_c = pat.sub("", cand_content).rstrip()
                if new_c != cand_content and len(new_c) >= 20:
                    cand_content = new_c
                    closer_stripped = True

            # Verify answer invariance before accepting opener/closer edits
            if opener_stripped or closer_stripped:
                new_answer = _extract_canonical_answer(domain, cand_content)
                if _answers_equivalent(domain, orig_answer, new_answer):
                    content = cand_content
                    if opener_stripped:
                        actions.append("strip_sycophantic_opener")
                    if closer_stripped:
                        actions.append("strip_boilerplate_closer")

            # 5. Whitespace & code-fence hygiene
            cleaned_c, changed_c = _normalize_whitespace_and_fences(content)
            if changed_c and _answers_equivalent(domain, orig_answer, _extract_canonical_answer(domain, cleaned_c)):
                content = cleaned_c
                actions.append("repair_whitespace_and_fences")

            if reasoning_str:
                cleaned_r, changed_r = _normalize_whitespace_and_fences(reasoning_str)
                if changed_r:
                    reasoning_str = cleaned_r
                    actions.append("repair_reasoning_whitespace")

        msg["content"] = content
        msg["reasoning"] = reasoning_str if reasoning_str else None
        out_msgs.append(msg)

    # Deduplicate action labels while preserving order
    dedup_actions = list(dict.fromkeys(actions))
    return out_msgs, dedup_actions


# ============================================================================ Canonical Message & Row Packing

def _canonicalize_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Strict 5-string-key message representation for drift-free PyArrow Parquet serialization."""
    out: list[dict[str, str]] = []
    for m in messages:
        tcs = m.get("tool_calls")
        tc_json = json.dumps(tcs, ensure_ascii=False) if tcs else ""
        out.append({
            "role": str(m.get("role") or ""),
            "content": str(m.get("content") or ""),
            "reasoning": str(m.get("reasoning") or ""),
            "tool_calls_json": tc_json,
            "name": str(m.get("name") or ""),
        })
    return out


def _extract_primary_fields(messages: Sequence[dict[str, Any]]) -> tuple[str, str, str]:
    """Extract canonical (prompt, reasoning, response) from normalized messages."""
    users = [str(m.get("content") or "").strip() for m in messages if m.get("role") == "user"]
    prompt = users[0] if len(users) == 1 else "\n\n---\n\n".join(u for u in users if u)
    assts = [m for m in messages if m.get("role") == "assistant"]
    if not assts:
        return prompt, "", ""
    last = assts[-1]
    response = str(last.get("content") or "").strip()
    all_reasonings = [str(m.get("reasoning") or "").strip() for m in assts if str(m.get("reasoning") or "").strip()]
    reasoning = "\n\n".join(all_reasonings)
    return prompt, reasoning, response


def _compute_overall_score(slop: dict[str, Any], beats: dict[str, Any]) -> float:
    anti_slop = max(0.0, 1.0 - float(slop["slop_score"]))
    diversity = min(1.0, (float(slop["distinct_2"]) + float(slop["distinct_3"])) / 1.4)
    three_beat = float(beats["three_beat_score"])
    return round(0.40 * anti_slop + 0.40 * three_beat + 0.20 * diversity, 4)


# ============================================================================ 3-Way Triage Engine

@dataclass
class TriageDecision:
    verdict: str  # "KEEP" | "SALVAGE" | "DROP"
    reasons: list[str]
    record: dict[str, Any] | None
    scores: dict[str, float] = field(default_factory=dict)
    salvage_actions: list[str] = field(default_factory=list)


def triage_example(
    ex: Example,
    src: SourceCitation,
    *,
    split_used: str = "",
    raw_meta: dict[str, Any] | None = None,
    decontaminator: Decontaminator | None = None,
    deduplicator: CrossSourceDeduplicator | None = None,
    apply_deterministic_salvage: bool = True,
) -> TriageDecision:
    """Run all 7 audit gates + deterministic salvage and assign KEEP / SALVAGE / DROP."""
    if not src.included or _normalize_license_id(src.license) not in ALLOWED_LICENSES:
        return TriageDecision("DROP", [f"license_incompatible:{src.license}"], None)

    msgs = list(ex.messages or [])
    if not msgs:
        return TriageDecision("DROP", ["pathology:empty_messages"], None)

    prompt_pre, _reasoning_pre, response_pre = _extract_primary_fields(msgs)
    domain = _infer_sub_domain(src.domain, prompt_pre, response_pre, raw_meta)

    # Gate 7: Strict 13-gram decontamination (before any salvage/dedup so contamination is always counted!)
    if decontaminator is not None:
        full_text = ex.plain_text(include_reasoning=True)
        hit, group = decontaminator.check(full_text, source=src.name)
        if hit:
            return TriageDecision("DROP", [f"contaminated:{group}"], None)

    # Gate 6: Cross-source exact SHA-256 + shingle/Jaccard deduplication
    if deduplicator is not None:
        is_dup, dup_reason, orig_src = deduplicator.check_and_add(src.name, prompt_pre, response_pre)
        if is_dup:
            tag = f"{dup_reason}(orig={orig_src})" if orig_src and orig_src != src.name else dup_reason
            return TriageDecision("DROP", [tag], None)

    # Deterministic Salvage Pass (lossless cleanup of tags, Nemotron sysprompts, sycophancy, whitespace)
    salvage_actions: list[str] = []
    if apply_deterministic_salvage:
        msgs, salvage_actions = deterministic_salvage(msgs, domain)

    prompt, reasoning, response = _extract_primary_fields(msgs)
    allow_empty_cot = (domain == "arc")

    # Gate 1 & 2: Completeness, integrity & pathology detection
    hard_issues = detect_pathology(
        prompt, reasoning, response, messages=msgs, allow_empty_reasoning=allow_empty_cot
    )
    # Gate 5: Domain sanity checks
    dom_hard, dom_salvage, extracted_answer = check_domain_sanity(
        domain, prompt, reasoning, response, messages=msgs
    )
    hard_issues.extend(dom_hard)

    if hard_issues:
        return TriageDecision("DROP", list(dict.fromkeys(hard_issues)), None, salvage_actions=salvage_actions)

    # Gate 3: Anti-slop & voice scoring
    slop = score_anti_slop(response, reasoning)
    # Gate 4: Looped-CoT 3-beat structural signal
    beats = score_three_beat_cot(reasoning, response, prompt)
    overall = _compute_overall_score(slop, beats)

    quality_scores = {
        "slop_score": float(slop["slop_score"]),
        "distinct_2": float(slop["distinct_2"]),
        "distinct_3": float(slop["distinct_3"]),
        "beat1_score": float(beats["beat1_score"]),
        "beat2_score": float(beats["beat2_score"]),
        "beat3_score": float(beats["beat3_score"]),
        "three_beat_score": float(beats["three_beat_score"]),
        "overall_score": float(overall),
    }

    # Determine residual SALVAGE flags (reasons why the row needs Track B LLM rewrite)
    salvage_reasons: list[str] = list(dom_salvage)

    if domain == "arc" and not reasoning.strip():
        salvage_reasons.append("salvage:missing_cot_verified_answer")
    else:
        if slop["slop_score"] >= 0.35:
            salvage_reasons.append("salvage:high_slop_voice")
        if len(normalize_words(response)) >= 80 and (slop["distinct_2"] < 0.32 or slop["distinct_3"] < 0.42):
            salvage_reasons.append("salvage:low_lexical_diversity")
        if len(reasoning.strip()) < 100:
            salvage_reasons.append("salvage:thin_reasoning")
        elif not beats["has_beat2_audit"] and beats["three_beat_score"] < 0.55:
            salvage_reasons.append("salvage:missing_beat2_audit")

    verdict = "SALVAGE" if salvage_reasons else "KEEP"
    transform_stage = (
        "deterministic_salvage" if (verdict == "KEEP" and salvage_actions) else (
            "keep_raw" if verdict == "KEEP" else "queued_for_track_b"
        )
    )

    row_id = hashlib.sha256(
        f"{src.name}:{split_used or src.split}:{ex.row_index}:{ex.sub_index}:{prompt[:120]}".encode("utf-8", "replace")
    ).hexdigest()[:20]

    record: dict[str, Any] = {
        "id": f"{src.name}_{row_id}",
        "domain": domain,
        "messages": _canonicalize_messages(msgs),
        "prompt": prompt,
        "thinking": reasoning,
        "response": response,
        "tools_json": json.dumps(ex.tools or [], ensure_ascii=False),
        "verdict": verdict,
        "triage_reasons": list(dict.fromkeys(salvage_reasons if salvage_reasons else ["clean_keep"])),
        "quality_scores": quality_scores,
        "extracted_answer": extracted_answer,
        "provenance": {
            "source_dataset": src.hf_id,
            "source_name": src.name,
            "source_config": src.config or "default",
            "source_split": split_used or src.split,
            "source_license": src.license,
            "source_row_index": int(ex.row_index),
            "citation_key": src.citation_key,
            "transform_stage": transform_stage,
            "salvage_actions": list(salvage_actions),
        },
    }
    return TriageDecision(
        verdict=verdict,
        reasons=record["triage_reasons"],
        record=record,
        scores=quality_scores,
        salvage_actions=salvage_actions,
    )


# ============================================================================ Streaming Upstream Row Reader

def _stream_hf_jsonl_fallback(
    hf_id: str, rel_path: str, max_rows: int | None, token: str | None
) -> Iterator[dict[str, Any]]:
    """Stream JSONL line-by-line via HfFileSystem without downloading the full dataset."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token or os.environ.get("HF_TOKEN"))
    with fs.open(f"datasets/{hf_id}/{rel_path}", "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if max_rows is not None and i >= max_rows:
                break
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def stream_source_rows(
    src: SourceCitation,
    *,
    max_rows: int | None = 1000,
    token: str | None = None,
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """Stream (split_name, row_index, raw_row_dict) from an included upstream source."""
    if not src.included:
        raise ValueError(f"Refusing to stream excluded source {src.name!r}: {src.exclusion_reason}")

    spec = get_spec(src.name) if src.name in REGISTRY else DatasetSpec(
        name=src.name, hf_id=src.hf_id, config=src.config, split=src.split, license=src.license,
    )
    tok = token or os.environ.get("HF_TOKEN")

    if spec.kind == "github":
        from qloop.data.mixture import arc_task_rows

        rows = arc_task_rows(spec.splits[0], repo=spec.hf_id.split(":", 1)[1])
        for idx, row in enumerate(rows):
            if max_rows is not None and idx >= max_rows:
                break
            yield spec.splits[0], idx, row
        return

    splits = spec.splits or ["train"]
    per_split_budget: list[int | None]
    if max_rows is None:
        per_split_budget = [None] * len(splits)
    else:
        base = max_rows // len(splits)
        rem = max_rows % len(splits)
        per_split_budget = [base + (1 if i < rem else 0) for i in range(len(splits))]

    for spl, budget in zip(splits, per_split_budget):
        if budget == 0:
            continue
        if src.jsonl_fallback_path:
            for idx, row in enumerate(_stream_hf_jsonl_fallback(src.hf_id, src.jsonl_fallback_path, budget, tok)):
                yield spl, idx, row
            continue
        try:
            import datasets

            kw: dict[str, Any] = {"split": spl, "streaming": True}
            if tok:
                kw["token"] = tok
            if spec.revision is not None:
                kw["revision"] = spec.revision
            ds = datasets.load_dataset(spec.hf_id, spec.config, **kw)
            for idx, row in enumerate(ds):
                if budget is not None and idx >= budget:
                    break
                yield spl, idx, dict(row)
        except Exception as exc:
            # Fallback to direct Parquet streaming if datasets schema parser rejects newer Feature types
            log.warning("[%s/%s] datasets.load_dataset failed (%s); attempting Parquet fallback", src.name, spl, exc)
            import pyarrow.parquet as pq
            from huggingface_hub import HfApi, HfFileSystem

            api = HfApi(token=tok)
            files = [f for f in api.list_repo_files(src.hf_id, repo_type="dataset") if f.endswith(".parquet")]
            if not files:
                raise
            fs = HfFileSystem(token=tok)
            emitted = 0
            for pf_rel in sorted(files):
                with fs.open(f"datasets/{src.hf_id}/{pf_rel}", "rb") as fh:
                    pqf = pq.ParquetFile(fh)
                    for batch in pqf.iter_batches(batch_size=128):
                        for row in batch.to_pylist():
                            yield spl, emitted, row
                            emitted += 1
                            if budget is not None and emitted >= budget:
                                return


# ============================================================================ Golden Seeds Loader

def load_golden_seeds(
    path: str | Path,
    *,
    decontaminator: Decontaminator | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and verify author-original golden seeds -> (sft_rows, preference_rows)."""
    raw_items = json.loads(Path(path).read_text(encoding="utf-8"))
    sft_rows: list[dict[str, Any]] = []
    pref_rows: list[dict[str, Any]] = []

    for idx, item in enumerate(raw_items):
        msgs = list(item["messages"])
        prompt, reasoning, response = _extract_primary_fields(msgs)
        domain = str(item.get("domain") or "reasoning")
        category = str(item.get("category") or domain)
        tools = item.get("tools") or []

        if decontaminator is not None:
            full = f"{prompt}\n{reasoning}\n{response}"
            hit, grp = decontaminator.check(full, source="sirsi_golden_seeds")
            if hit:
                raise ValueError(f"Golden seed #{idx} ({item.get('id')}) hit decontam group '{grp}'!")

        hard = detect_pathology(prompt, reasoning, response, messages=msgs)
        dom_hard, _, extracted_ans = check_domain_sanity(domain, prompt, reasoning, response, messages=msgs)
        cot_artifacts = detect_synthetic_cot_artifacts(reasoning, response, domain)
        if hard or dom_hard or cot_artifacts:
            raise ValueError(
                f"Golden seed #{idx} ({item.get('id')}) failed quality gate: {hard + dom_hard + cot_artifacts}"
            )

        slop = score_anti_slop(response, reasoning)
        beats = score_three_beat_cot(reasoning, response, prompt)
        overall = _compute_overall_score(slop, beats)
        qscores = {
            "slop_score": float(slop["slop_score"]),
            "distinct_2": float(slop["distinct_2"]),
            "distinct_3": float(slop["distinct_3"]),
            "beat1_score": float(beats["beat1_score"]),
            "beat2_score": float(beats["beat2_score"]),
            "beat3_score": float(beats["beat3_score"]),
            "three_beat_score": float(beats["three_beat_score"]),
            "overall_score": float(overall),
        }
        prov = {
            "source_dataset": AUTHOR_ORIGINAL_CITATION.hf_id,
            "source_name": "sirsi_golden_seeds",
            "source_config": category,
            "source_split": "train",
            "source_license": "apache-2.0",
            "source_row_index": int(idx),
            "citation_key": "author_original",
            "transform_stage": "golden_seed",
            "salvage_actions": [],
        }

        sft_rows.append({
            "id": str(item["id"]),
            "domain": domain,
            "messages": _canonicalize_messages(msgs),
            "prompt": prompt,
            "thinking": reasoning,
            "response": response,
            "tools_json": json.dumps(tools, ensure_ascii=False),
            "verdict": "KEEP",
            "triage_reasons": ["golden_seed_verified"],
            "quality_scores": qscores,
            "extracted_answer": extracted_ans,
            "provenance": prov,
        })

        pref = item.get("preference_pair")
        if isinstance(pref, dict):
            rej_t = str(pref.get("rejected_thinking") or "")
            rej_r = str(pref.get("rejected_response") or "")
            pref_rows.append({
                "id": f"{item['id']}_pref",
                "category": category,
                "prompt": prompt,
                "chosen_thinking": reasoning,
                "chosen_response": response,
                "chosen_full": f"<think>\n{reasoning}\n</think>\n\n{response}",
                "rejected_thinking": rej_t,
                "rejected_response": rej_r,
                "rejected_full": f"<think>\n{rej_t}\n</think>\n\n{rej_r}" if rej_t else rej_r,
                "critique": str(pref.get("critique") or ""),
                "quality_scores": qscores,
                "provenance": prov,
            })

    return sft_rows, pref_rows


# ============================================================================ Track B Async Synthesis Engine

PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "hf_router": {
        "base_url": "https://router.huggingface.co/v1/chat/completions",
        "api_key_env": "HF_TOKEN",
        "default_model": "Qwen/Qwen2.5-72B-Instruct",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-pro",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-sonnet-4",
    },
    "local": {
        "base_url": "http://127.0.0.1:8000/v1/chat/completions",
        "api_key_env": "LOCAL_API_KEY",
        "default_model": "qloop-local",
    },
}


def _scrub_secret(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")


class AsyncTeacherClient:
    """Async OpenAI-compatible teacher client for hf_router / gemini / openai / openrouter / local."""

    def __init__(
        self,
        provider: str = "hf_router",
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 90.0,
        max_retries: int = 3,
        max_concurrency: int = 4,
    ):
        if provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"Unknown teacher provider {provider!r}; expected one of {sorted(PROVIDER_DEFAULTS)}")
        cfg = PROVIDER_DEFAULTS[provider]
        self.provider = provider
        self.model = model or cfg["default_model"]
        self.base_url = base_url or cfg["base_url"]
        self.api_key_env = cfg["api_key_env"]
        key = api_key if api_key is not None else os.environ.get(self.api_key_env, "")
        if not key and provider == "gemini":
            key = os.environ.get("GOOGLE_API_KEY", "")
        if not key and provider == "local":
            key = "EMPTY"
        self._api_key = key
        self.timeout = timeout
        self.max_retries = max_retries
        self._max_concurrency = max(1, max_concurrency)
        self._sems: dict[int, asyncio.Semaphore] = {}

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        sem = self._sems.get(loop_id)
        if sem is None:
            sem = asyncio.Semaphore(self._max_concurrency)
            self._sems[loop_id] = sem
        return sem

    @property
    def has_credentials(self) -> bool:
        return bool(self._api_key)

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.4,
    ) -> dict[str, Any]:
        """Issue one chat completion with retry/backoff; NEVER logs or returns API keys."""
        if not self._api_key:
            return {"ok": False, "status_code": 0, "error": f"missing_env:{self.api_key_env}", "content": ""}
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        last_status = 0
        last_err = ""
        async with self._get_semaphore(), httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    resp = await client.post(self.base_url, headers=headers, json=payload)
                    last_status = resp.status_code
                    if resp.status_code == 200:
                        data = resp.json()
                        choice = (data.get("choices") or [{}])[0]
                        msg = choice.get("message") or {}
                        content = str(msg.get("content") or "")
                        rc = str(msg.get("reasoning_content") or "")
                        if rc and "<think>" not in content:
                            content = f"<think>\n{rc}\n</think>\n\n{content}"
                        return {"ok": True, "status_code": 200, "content": content, "error": ""}
                    body = _scrub_secret(resp.text[:300], self._api_key)
                    last_err = body
                    # Do NOT retry auth/billing errors (401/402/403/404)
                    if resp.status_code in (400, 401, 402, 403, 404):
                        return {"ok": False, "status_code": resp.status_code, "content": "", "error": body}
                except Exception as exc:
                    last_err = _scrub_secret(f"{type(exc).__name__}: {exc}", self._api_key)[:300]
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(min(8.0, 1.0 * (2**attempt)))
        return {"ok": False, "status_code": last_status, "content": "", "error": last_err}

    async def probe(self) -> dict[str, Any]:
        """Minimal max_tokens=16 reachability probe (never exposes secrets)."""
        res = await self.chat_completion(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
            temperature=0.0,
        )
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "has_credentials": self.has_credentials,
            "ok": bool(res["ok"]),
            "status_code": int(res["status_code"]),
            "detail": res["content"][:64] if res["ok"] else res["error"][:240],
        }


# Prompts for the 3 Track B synthesis modes
_SALVAGE_REWRITE_SYS = (
    "You are an expert reasoning editor for a frontier 3-pass looped reasoning model.\n"
    "Rewrite the reasoning trace and final answer into:\n"
    "1. `<think>...</think>` written in authentic Fable-5.1-style organic, kinetic, first-person "
    "engineering/scientific reasoning:\n"
    "   - Reason forward from the problem's constraints, invariants, and concrete symbols, numbers, or code inline.\n"
    "   - Carry out every derivation step explicitly, then mid-stream perform a concrete adversarial self-check "
    "(substitute values back in, test a small/boundary edge case, or check a subtle pitfall with explicit "
    "arithmetic or code).\n"
    "   - HARD PROHIBITIONS inside `<think>`: ZERO literal `Beat 1` / `Beat 2` / `Beat 3`, `Framing:`, "
    "`Adversarial Self-Audit:`, or `Synthesis Plan:` section headers; ZERO spoiler-first openings "
    "(`Final answer:`, `Answer:`, `The final answer is`, or `Target answer:` at the top of `<think>`); "
    "ZERO rubber-stamp meta-summary hand-waving (never write 'Standard processing applied', "
    "'Initial parameters are validated', 'confirms there are no hidden fees/traps', or 'all constraints "
    "and invariants are satisfied' in place of real step-by-step computation).\n"
    "2. Final answer after `</think>`: crisp, opinionated, zero AI-slop (NO 'Certainly!', 'delve', 'tapestry', "
    "'in conclusion', 'it is important to note', or bold-bullet walls).\n"
    "CRITICAL: Preserve the exact verified final mathematical value, ARC grid, or code semantics."
)

_AESTHETIC_CRAFT_SYS = (
    "You generate high-craft aesthetic critique-and-refactor training pairs (UI/Visual, Technical Writing, or "
    "Software Architecture). Output strict JSON with keys: "
    "`prompt`, `chosen_thinking`, `chosen_response`, `rejected_thinking`, `rejected_response`, `critique`.\n"
    "`chosen_thinking` must read as authentic Fable-5.1-style organic, kinetic, first-person engineering/design "
    "reasoning: diagnose concrete defects inline (exact hex colors, pixel metrics, word counts, types, failure "
    "modes) and actively self-audit edge cases (WCAG contrast, keyboard focus, API ergonomics, budget limits). "
    "NEVER emit literal `Beat 1`/`Beat 2`/`Beat 3`, `Framing:`, or `Adversarial Self-Audit:` headers, NEVER open "
    "with `Final answer:`/`Answer:`, and NEVER use rubber-stamp meta-summary hand-waving.\n"
    "`chosen_response` must have zero AI-slop. `rejected_response` should exhibit realistic generic AI slop."
)

_SIRSI_RECOVERY_SYS = (
    "You generate SI-RSI 'teacher-slap' self-recovery trajectories where a model begins a realistic failure "
    "(sign error, loop onset, API hallucination, or spec assumption), catches itself mid-thought with a crisp "
    "`[Audit — stop: ...]` diagnostic interruption, and recovers cleanly to a verified solution.\n"
    "Write `chosen_thinking` in authentic Fable-5.1-style organic, kinetic, first-person engineering/scientific "
    "reasoning with concrete symbols, numbers, and explicit small/boundary-case verification inline — "
    "ZERO literal `Beat 1`/`Beat 2`/`Beat 3` or `Framing:`/`Adversarial Self-Audit:` labels, ZERO spoiler-first "
    "openings (`Final answer:`/`Answer:`/`Target answer:`), and ZERO rubber-stamp meta-summary hand-waving.\n"
    "Output strict JSON with keys: "
    "`prompt`, `chosen_thinking`, `chosen_response`, `rejected_thinking`, `rejected_response`, `critique`."
)

_DEFAULT_AESTHETIC_BRIEFS: list[tuple[str, str]] = [
    (
        "aesthetic_ui",
        "Design a compact single-screen incident-status header in semantic HTML/CSS "
        "for an SRE war room without neon gradients or badge clutter.",
    ),
    (
        "aesthetic_writing",
        "Write a 200-word release note explaining a breaking memory-allocator change "
        "for systems engineers with zero corporate filler.",
    ),
    (
        "aesthetic_code",
        "Refactor a deeply nested 5-class StrategyFactory configuration loader in Python "
        "into flat typed dataclasses with explicit validation.",
    ),
]

_DEFAULT_RECOVERY_BRIEFS: list[tuple[str, str]] = [
    (
        "sirsi_recovery",
        "Compute the sum of all positive integers n <= 50 such that n^2 + 3n + 2 is "
        "divisible by 6, catching an early modular-case oversight.",
    ),
    (
        "sirsi_recovery",
        "Write a Python function to merge overlapping half-open intervals [start, end), "
        "catching an early off-by-one boundary mistake on adjacent intervals.",
    ),
    (
        "sirsi_recovery",
        "Diagnose why an async Python HTTP client leaks file descriptors under cancellation, "
        "catching an initial wrong guess about connection pool size.",
    ),
]


def verify_synthesized_row(
    *,
    domain: str,
    prompt: str,
    thinking: str,
    response: str,
    orig_extracted_answer: str = "",
    orig_response: str = "",
    decontaminator: Decontaminator | None = None,
) -> tuple[bool, list[str], dict[str, float]]:
    """Strict post-synthesis gate: answer invariance, anti-slop, pathology, CoT artifacts, 3-beat, decontam."""
    reasons: list[str] = []

    # 1. Pathology gate
    hard = detect_pathology(prompt, thinking, response)
    if hard:
        reasons.extend(hard)

    # 1b. Synthetic CoT artifact gate (anti-rubric-header, anti-spoiler-first, anti-meta-summary-stub)
    cot_artifacts = detect_synthetic_cot_artifacts(thinking, response, domain)
    if cot_artifacts:
        reasons.extend(cot_artifacts)

    # 2. Domain sanity gate (AST for Python, grid for ARC, etc.)
    dom_hard, _, new_extracted = check_domain_sanity(domain, prompt, thinking, response)
    if dom_hard:
        reasons.extend(dom_hard)

    # 3. Answer invariance gate (math & ARC)
    if orig_extracted_answer:
        if not _answers_equivalent(domain, orig_extracted_answer, new_extracted):
            reasons.append(f"rejected:answer_mismatch(orig={orig_extracted_answer!r},new={new_extracted!r})")
    elif domain == "arc" and orig_response:
        orig_g = _parse_arc_grid(orig_response)
        new_g = _parse_arc_grid(response)
        if orig_g is not None and not grid_equal(orig_g, new_g):
            reasons.append("rejected:arc_grid_mismatch")

    # 4. Anti-slop gate
    slop = score_anti_slop(response, thinking)
    if slop["slop_score"] >= 0.30:
        reasons.append(f"rejected:slop_score({slop['slop_score']})")

    # 5. Looped-CoT 3-beat gate
    beats = score_three_beat_cot(thinking, response, prompt)
    if not beats["has_beat2_audit"] or beats["three_beat_score"] < 0.60:
        reasons.append(f"rejected:weak_3beat(score={beats['three_beat_score']},beat2={beats['has_beat2_audit']})")

    # 6. Strict 13-gram decontamination gate
    if decontaminator is not None:
        hit, grp = decontaminator.check(f"{prompt}\n{thinking}\n{response}", source="track_b_synth")
        if hit:
            reasons.append(f"rejected:contaminated({grp})")

    scores = {
        "slop_score": float(slop["slop_score"]),
        "distinct_2": float(slop["distinct_2"]),
        "distinct_3": float(slop["distinct_3"]),
        "beat1_score": float(beats["beat1_score"]),
        "beat2_score": float(beats["beat2_score"]),
        "beat3_score": float(beats["beat3_score"]),
        "three_beat_score": float(beats["three_beat_score"]),
        "overall_score": float(_compute_overall_score(slop, beats)),
    }
    return (len(reasons) == 0), reasons, scores


class TrackBSynthesizer:
    """Async resumable Track B synthesis & rewrite orchestrator with strict verification gates."""

    def __init__(
        self,
        client: AsyncTeacherClient,
        *,
        out_dir: str | Path,
        decontaminator: Decontaminator | None = None,
    ):
        self.client = client
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.decontaminator = decontaminator
        self.stats: collections.Counter[str] = collections.Counter()

    def _ckpt_path(self, mode: str) -> Path:
        return self.out_dir / f".synth_ckpt_{mode}.jsonl"

    def _load_ckpt(self, mode: str) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]]]:
        p = self._ckpt_path(mode)
        done_ids: set[str] = set()
        sft: list[dict[str, Any]] = []
        prefs: list[dict[str, Any]] = []
        if not p.exists():
            return done_ids, sft, prefs
        for lineno, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                log.warning("[synth] ignoring truncated/malformed checkpoint line %d in %s", lineno, p.name)
                continue
            if not isinstance(entry, dict) or "source_id" not in entry:
                continue
            # Never mark transient API errors as permanently done so resume retries them
            reasons = entry.get("reasons") or []
            if entry.get("transient_error") or (
                not entry.get("accepted") and any(str(r).startswith("api_error:") for r in reasons)
            ):
                continue
            done_ids.add(str(entry["source_id"]))
            if entry.get("accepted"):
                if entry.get("sft_record"):
                    sft.append(entry["sft_record"])
                if entry.get("pref_record"):
                    prefs.append(entry["pref_record"])
        return done_ids, sft, prefs

    def _append_ckpt(self, mode: str, entry: dict[str, Any]) -> None:
        with self._ckpt_path(mode).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()

    async def run_salvage_rewrite(
        self,
        salvage_rows: Sequence[dict[str, Any]],
        *,
        max_items: int = 10,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Rewrite salvage_queue rows into verified 3-beat anti-slop SFT + preference pairs."""
        done_ids, sft_out, pref_out = self._load_ckpt("salvage_rewrite")
        todo = [r for r in salvage_rows if str(r["id"]) not in done_ids][:max_items]
        ckpt_lock = asyncio.Lock()

        async def _one(row: dict[str, Any]) -> dict[str, Any]:
            sid = str(row["id"])
            domain = str(row.get("domain") or "reasoning")
            orig_ans = str(row.get("extracted_answer") or "")
            user_msg = (
                f"Domain: {domain}\n"
                f"Problem:\n{row['prompt'][:4000]}\n\n"
                f"Original Draft Thinking:\n{row.get('thinking', '')[:3000]}\n\n"
                f"Original Draft Answer:\n{row.get('response', '')[:3000]}\n\n"
                "Rewrite into <think>...</think> using authentic Fable-5.1-style organic, kinetic, first-person "
                "reasoning (forward step-by-step computation with concrete numbers/symbols and an inline self-check — "
                "NO `Beat 1/2/3` headers, NO `Final answer:` at top of `<think>`, NO meta-summary stubs) "
                "followed by a crisp, anti-slop final answer preserving the verified answer."
            )
            res = await self.client.chat_completion([
                {"role": "system", "content": _SALVAGE_REWRITE_SYS},
                {"role": "user", "content": user_msg},
            ])
            self.stats["salvage_rewrite:attempted"] += 1
            if not res["ok"]:
                self.stats["salvage_rewrite:api_error"] += 1
                entry: dict[str, Any] = {
                    "source_id": sid,
                    "accepted": False,
                    "transient_error": True,
                    "reasons": [f"api_error:{res['status_code']}"],
                }
                async with ckpt_lock:
                    self._append_ckpt("salvage_rewrite", entry)
                return entry

            thinking, response = extract_think(res["content"])
            thinking_s = (thinking or "").strip()
            response_s = (response or "").strip()
            ok, reasons, qscores = verify_synthesized_row(
                domain=domain,
                prompt=row["prompt"],
                thinking=thinking_s,
                response=response_s,
                orig_extracted_answer=orig_ans,
                orig_response=row.get("response", ""),
                decontaminator=self.decontaminator,
            )
            if not ok:
                self.stats["salvage_rewrite:rejected"] += 1
                for r in reasons:
                    self.stats[f"salvage_rewrite:{r.split('(')[0]}"] += 1
                entry = {"source_id": sid, "accepted": False, "reasons": reasons}
                async with ckpt_lock:
                    self._append_ckpt("salvage_rewrite", entry)
                return entry

            self.stats["salvage_rewrite:accepted"] += 1
            prov = dict(row["provenance"])
            prov["transform_stage"] = "track_b_rewrite"
            prov["salvage_actions"] = [
                *(prov.get("salvage_actions") or []),
                f"track_b_rewrite:{self.client.provider}/{self.client.model}",
            ]
            msgs = _canonicalize_messages([
                {"role": "user", "content": row["prompt"], "reasoning": None},
                {"role": "assistant", "content": response_s, "reasoning": thinking_s},
            ])
            sft_rec = {
                "id": f"{sid}_rw",
                "domain": domain,
                "messages": msgs,
                "prompt": row["prompt"],
                "thinking": thinking_s,
                "response": response_s,
                "tools_json": "[]",
                "verdict": "KEEP",
                "triage_reasons": ["track_b_rewrite_verified"],
                "quality_scores": qscores,
                "extracted_answer": _extract_canonical_answer(domain, response_s) or orig_ans,
                "provenance": prov,
            }
            pref_rec = {
                "id": f"{sid}_rw_pref",
                "category": "salvage_rewrite",
                "prompt": row["prompt"],
                "chosen_thinking": thinking_s,
                "chosen_response": response_s,
                "chosen_full": f"<think>\n{thinking_s}\n</think>\n\n{response_s}",
                "rejected_thinking": str(row.get("thinking") or ""),
                "rejected_response": str(row.get("response") or ""),
                "rejected_full": (
                    f"<think>\n{row['thinking']}\n</think>\n\n{row['response']}"
                    if row.get("thinking") else str(row.get("response") or "")
                ),
                "critique": f"Track B rewrite addressing: {', '.join(row.get('triage_reasons') or [])}",
                "quality_scores": qscores,
                "provenance": prov,
            }
            entry = {"source_id": sid, "accepted": True, "sft_record": sft_rec, "pref_record": pref_rec}
            async with ckpt_lock:
                self._append_ckpt("salvage_rewrite", entry)
            return entry

        results = await asyncio.gather(*[_one(r) for r in todo])
        for entry in results:
            if entry.get("accepted"):
                sft_out.append(entry["sft_record"])
                pref_out.append(entry["pref_record"])
        return sft_out, pref_out

    async def run_structured_mode(
        self,
        mode: str,
        briefs: Sequence[tuple[str, str]] | None = None,
        *,
        max_items: int = 6,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run `aesthetic_craft` or `sirsi_recovery` synthesis -> (sft_rows, pref_rows)."""
        if mode not in ("aesthetic_craft", "sirsi_recovery"):
            raise ValueError(f"Unsupported synthesis mode: {mode!r}")
        sys_prompt = _AESTHETIC_CRAFT_SYS if mode == "aesthetic_craft" else _SIRSI_RECOVERY_SYS
        default_briefs = _DEFAULT_AESTHETIC_BRIEFS if mode == "aesthetic_craft" else _DEFAULT_RECOVERY_BRIEFS
        item_briefs = list(briefs if briefs is not None else default_briefs)[:max_items]
        done_ids, sft_out, pref_out = self._load_ckpt(mode)
        ckpt_lock = asyncio.Lock()

        async def _one(idx: int, cat: str, brief: str) -> dict[str, Any]:
            sid = f"{mode}_{idx:04d}"
            if sid in done_ids:
                return {"source_id": sid, "skipped": True}
            res = await self.client.chat_completion([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Category: {cat}\nBrief: {brief}"},
            ])
            self.stats[f"{mode}:attempted"] += 1
            if not res["ok"]:
                self.stats[f"{mode}:api_error"] += 1
                entry: dict[str, Any] = {
                    "source_id": sid,
                    "accepted": False,
                    "transient_error": True,
                    "reasons": [f"api_error:{res['status_code']}"],
                }
                async with ckpt_lock:
                    self._append_ckpt(mode, entry)
                return entry

            raw_txt = res["content"].strip()
            # Strip optional ```json fences
            m_json = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_txt, re.DOTALL)
            raw_json = m_json.group(1) if m_json else raw_txt
            try:
                obj = json.loads(raw_json)
            except json.JSONDecodeError:
                self.stats[f"{mode}:rejected_invalid_json"] += 1
                entry = {"source_id": sid, "accepted": False, "reasons": ["rejected:invalid_json"]}
                async with ckpt_lock:
                    self._append_ckpt(mode, entry)
                return entry
            if not isinstance(obj, dict):
                self.stats[f"{mode}:rejected_non_dict_json"] += 1
                entry = {"source_id": sid, "accepted": False, "reasons": ["rejected:non_dict_json"]}
                async with ckpt_lock:
                    self._append_ckpt(mode, entry)
                return entry

            prompt = str(obj.get("prompt") or brief).strip()
            c_think = str(obj.get("chosen_thinking") or "").strip()
            c_resp = str(obj.get("chosen_response") or "").strip()
            r_think = str(obj.get("rejected_thinking") or "").strip()
            r_resp = str(obj.get("rejected_response") or "").strip()
            critique = str(obj.get("critique") or "").strip()
            domain = "aesthetic" if mode == "aesthetic_craft" else "recovery"

            ok, reasons, qscores = verify_synthesized_row(
                domain=domain,
                prompt=prompt,
                thinking=c_think,
                response=c_resp,
                decontaminator=self.decontaminator,
            )
            if not r_resp or r_resp == c_resp:
                ok = False
                reasons.append("rejected:invalid_rejected_pair")
            if self.decontaminator is not None and (r_think or r_resp):
                hit_r, grp_r = self.decontaminator.check(f"{r_think}\n{r_resp}", source="track_b_synth_rejected")
                if hit_r:
                    ok = False
                    reasons.append(f"rejected:contaminated_rejected({grp_r})")
            if not ok:
                self.stats[f"{mode}:rejected"] += 1
                entry = {"source_id": sid, "accepted": False, "reasons": reasons}
                async with ckpt_lock:
                    self._append_ckpt(mode, entry)
                return entry

            self.stats[f"{mode}:accepted"] += 1
            prov = {
                "source_dataset": AUTHOR_ORIGINAL_CITATION.hf_id,
                "source_name": f"track_b_{mode}",
                "source_config": cat,
                "source_split": "train",
                "source_license": "apache-2.0",
                "source_row_index": int(idx),
                "citation_key": "author_original",
                "transform_stage": "track_b_synth",
                "salvage_actions": [f"{mode}:{self.client.provider}/{self.client.model}"],
            }
            sft_rec = {
                "id": sid,
                "domain": domain,
                "messages": _canonicalize_messages([
                    {"role": "user", "content": prompt, "reasoning": None},
                    {"role": "assistant", "content": c_resp, "reasoning": c_think},
                ]),
                "prompt": prompt,
                "thinking": c_think,
                "response": c_resp,
                "tools_json": "[]",
                "verdict": "KEEP",
                "triage_reasons": [f"track_b_{mode}_verified"],
                "quality_scores": qscores,
                "extracted_answer": "",
                "provenance": prov,
            }
            pref_rec = {
                "id": f"{sid}_pref",
                "category": cat,
                "prompt": prompt,
                "chosen_thinking": c_think,
                "chosen_response": c_resp,
                "chosen_full": f"<think>\n{c_think}\n</think>\n\n{c_resp}",
                "rejected_thinking": r_think,
                "rejected_response": r_resp,
                "rejected_full": f"<think>\n{r_think}\n</think>\n\n{r_resp}" if r_think else r_resp,
                "critique": critique,
                "quality_scores": qscores,
                "provenance": prov,
            }
            entry = {"source_id": sid, "accepted": True, "sft_record": sft_rec, "pref_record": pref_rec}
            async with ckpt_lock:
                self._append_ckpt(mode, entry)
            return entry

        results = await asyncio.gather(*[_one(i, cat, b) for i, (cat, b) in enumerate(item_briefs)])
        for entry in results:
            if entry.get("skipped"):
                continue
            if entry.get("accepted"):
                sft_out.append(entry["sft_record"])
                pref_out.append(entry["pref_record"])
        return sft_out, pref_out


# ============================================================================ Parquet / JSONL Export

def _sft_arrow_schema() -> Any:
    import pyarrow as pa

    msg_struct = pa.struct([
        ("role", pa.string()),
        ("content", pa.string()),
        ("reasoning", pa.string()),
        ("tool_calls_json", pa.string()),
        ("name", pa.string()),
    ])
    scores_struct = pa.struct([
        ("slop_score", pa.float64()),
        ("distinct_2", pa.float64()),
        ("distinct_3", pa.float64()),
        ("beat1_score", pa.float64()),
        ("beat2_score", pa.float64()),
        ("beat3_score", pa.float64()),
        ("three_beat_score", pa.float64()),
        ("overall_score", pa.float64()),
    ])
    prov_struct = pa.struct([
        ("source_dataset", pa.string()),
        ("source_name", pa.string()),
        ("source_config", pa.string()),
        ("source_split", pa.string()),
        ("source_license", pa.string()),
        ("source_row_index", pa.int64()),
        ("citation_key", pa.string()),
        ("transform_stage", pa.string()),
        ("salvage_actions", pa.list_(pa.string())),
    ])
    return pa.schema([
        ("id", pa.string()),
        ("domain", pa.string()),
        ("messages", pa.list_(msg_struct)),
        ("prompt", pa.string()),
        ("thinking", pa.string()),
        ("response", pa.string()),
        ("tools_json", pa.string()),
        ("verdict", pa.string()),
        ("triage_reasons", pa.list_(pa.string())),
        ("quality_scores", scores_struct),
        ("extracted_answer", pa.string()),
        ("provenance", prov_struct),
    ])


def _pref_arrow_schema() -> Any:
    import pyarrow as pa

    scores_struct = pa.struct([
        ("slop_score", pa.float64()),
        ("distinct_2", pa.float64()),
        ("distinct_3", pa.float64()),
        ("beat1_score", pa.float64()),
        ("beat2_score", pa.float64()),
        ("beat3_score", pa.float64()),
        ("three_beat_score", pa.float64()),
        ("overall_score", pa.float64()),
    ])
    prov_struct = pa.struct([
        ("source_dataset", pa.string()),
        ("source_name", pa.string()),
        ("source_config", pa.string()),
        ("source_split", pa.string()),
        ("source_license", pa.string()),
        ("source_row_index", pa.int64()),
        ("citation_key", pa.string()),
        ("transform_stage", pa.string()),
        ("salvage_actions", pa.list_(pa.string())),
    ])
    return pa.schema([
        ("id", pa.string()),
        ("category", pa.string()),
        ("prompt", pa.string()),
        ("chosen_thinking", pa.string()),
        ("chosen_response", pa.string()),
        ("chosen_full", pa.string()),
        ("rejected_thinking", pa.string()),
        ("rejected_response", pa.string()),
        ("rejected_full", pa.string()),
        ("critique", pa.string()),
        ("quality_scores", scores_struct),
        ("provenance", prov_struct),
    ])


def write_parquet_records(records: Sequence[dict[str, Any]], path: str | Path, *, kind: str = "sft") -> int:
    """Write strictly-typed Parquet file (even when records is empty)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = _sft_arrow_schema() if kind == "sft" else _pref_arrow_schema()
    if records:
        table = pa.Table.from_pylist(list(records), schema=schema)
    else:
        table = pa.Table.from_pydict({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    pq.write_table(table, out_path, compression="zstd")
    return len(records)


def write_jsonl_records(records: Sequence[dict[str, Any]], path: str | Path) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


# ============================================================================ HF Dataset Card Generator

def _size_category(n_rows: int) -> str:
    if n_rows < 1_000:
        return "n<1K"
    if n_rows < 10_000:
        return "1K<n<10K"
    if n_rows < 100_000:
        return "10K<n<100K"
    if n_rows < 1_000_000:
        return "100K<n<1M"
    return "1M<n<10M"


def render_dataset_card(
    audit_summary: dict[str, Any],
    *,
    repo_id: str = "TassP0lar/Loop64-Fable-SIRSI-v1",
) -> str:
    """Render publication-grade Hugging Face Dataset Card (`README.md`) with YAML + full BibTeX."""
    counts = audit_summary.get("output_counts", {})
    n_sft = int(counts.get("sft_core", 0))
    n_salvage = int(counts.get("salvage_queue", 0))
    n_pref = int(counts.get("sirsi_preferences", 0))
    total_rows = n_sft + n_salvage + n_pref
    size_cat = _size_category(total_rows)

    lic_map = audit_summary.get("licenses", {})
    per_source = audit_summary.get("per_source", {})
    decontam = audit_summary.get("decontam", {})
    dedup = audit_summary.get("dedup", {})

    # Determine licenses present in emitted rows
    included_licenses = sorted({
        str(v.get("live_license") or v.get("declared_license") or "").lower()
        for v in lic_map.values()
        if v.get("included")
    })
    if not included_licenses:
        included_licenses = ["apache-2.0", "cc-by-4.0"]
    primary_yaml_lic = "cc-by-4.0" if "cc-by-4.0" in included_licenses else included_licenses[0]

    lines: list[str] = [
        "---",
        f"pretty_name: {repo_id.rsplit('/', maxsplit=1)[-1]}",
        f"license: {primary_yaml_lic}",
        "task_categories:",
        "- text-generation",
        "- question-answering",
        "language:",
        "- en",
        "- multilingual",
        "tags:",
        "- reasoning",
        "- chain-of-thought",
        "- looped-cot",
        "- anti-slop",
        "- sirsi",
        "- agentic",
        "- preference-data",
        "- decontaminated",
        "size_categories:",
        f"- {size_cat}",
        "configs:",
        "- config_name: sft_core",
        "  default: true",
        "  data_files:",
        "  - split: train",
        "    path: data/sft_core.parquet",
        "- config_name: salvage_queue",
        "  data_files:",
        "  - split: train",
        "    path: data/salvage_queue.parquet",
        "- config_name: sirsi_preferences",
        "  data_files:",
        "  - split: train",
        "    path: data/sirsi_preferences.parquet",
        "---",
        "",
        f"# {repo_id.rsplit('/', maxsplit=1)[-1]}",
        "",
        "A license-audited, 13-gram-decontaminated, anti-slop reasoning, agentic, aesthetic-craft, and "
        "SI-RSI self-recovery dataset built for **Qwen3.6-35B-A3B-Loop64** "
        "(`88L` 3-pass looped MoE + HSSD v3.2 + 1M D-LSA).",
        "",
        "## Dataset Splits & Configurations",
        "",
        "| Config / Split | Parquet File | Rows | Purpose |",
        "| :--- | :--- | ---: | :--- |",
        (
            f"| `sft_core` (default) | `data/sft_core.parquet` | {n_sft:,} | "
            "Verified `KEEP` rows (clean upstream + deterministic salvage + "
            "author-original golden seeds + verified Track B rewrites) |"
        ),
        (
            f"| `salvage_queue` | `data/salvage_queue.parquet` | {n_salvage:,} | "
            "Valid logic/answers queued for Track B LLM rewrite (high-slop voice, "
            "missing Beat-2 self-audit, raw ARC grid pairs, or placeholder-redacted tool args) |"
        ),
        (
            f"| `sirsi_preferences` | `data/sirsi_preferences.parquet` | {n_pref:,} | "
            "`chosen`/`rejected` preference pairs for Aesthetic Craft (UI/Writing/Code), "
            "SI-RSI recovery, and Track B rewrites |"
        ),
        "",
        "Every row carries full per-row `provenance` (`source_dataset`, `source_config`, `source_split`, "
        "`source_license`, `source_row_index`, `citation_key`, `transform_stage`, `salvage_actions`) and "
        "continuous `quality_scores` (`slop_score`, `distinct_2`, `distinct_3`, `beat1_score`, `beat2_score`, "
        "`beat3_score`, `three_beat_score`, `overall_score`).",
        "",
        "## License Compliance & Redistribution Hygiene",
        "",
        "This release strictly includes **only permissive, public-redistribution-compatible upstream sources** "
        "(`apache-2.0`, `cc-by-4.0`, `mit`, `odc-by`), verified live via Hugging Face Hub metadata. "
        "Copyleft (`agpl-3.0`) and unverified (`unknown`) datasets are explicitly excluded.",
        "",
        "| Source Key | Upstream Repo | Declared | Live Verified | Status | Notes |",
        "| :--- | :--- | :---: | :---: | :---: | :--- |",
    ]

    for src in ALL_AUDITED_SOURCES.values():
        lm = lic_map.get(src.name, {})
        live_l = lm.get("live_license", src.license)
        status = "INCLUDED" if lm.get("included", src.included) else "EXCLUDED"
        note = lm.get("reason") if status == "EXCLUDED" else src.attribution_note
        lines.append(
            f"| `{src.name}` | `{src.hf_id}` | `{src.license}` | `{live_l}` | **{status}** | {note} |"
        )

    lines.extend([
        "",
        "## Keep / Salvage / Drop Triage Breakdown",
        "",
        (
            "| Source | Streamed | KEEP | Det. Salvaged -> KEEP | SALVAGE (Queue) | "
            "DROP | Keep % | Top Drop / Salvage Reasons |"
        ),
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ])

    has_arc_expansion = False
    for name, st in per_source.items():
        streamed = int(st.get("streamed", 0))
        keep = int(st.get("keep", 0))
        det_sal = int(st.get("deterministic_salvaged_to_keep", 0))
        salvage = int(st.get("salvage", 0))
        drop = int(st.get("drop", 0))
        expanded_total = keep + salvage + drop
        denom = expanded_total if (name == "arc_agi2_train" and expanded_total > streamed) else max(1, streamed)
        keep_pct = 100.0 * keep / max(1, denom)
        stream_cell = f"{streamed:,}*" if (name == "arc_agi2_train" and expanded_total > streamed) else f"{streamed:,}"
        if name == "arc_agi2_train" and expanded_total > streamed:
            has_arc_expansion = True
        top_drop = sorted((st.get("drop_reasons") or {}).items(), key=lambda kv: kv[1], reverse=True)[:2]
        top_salv = sorted((st.get("salvage_reasons") or {}).items(), key=lambda kv: kv[1], reverse=True)[:2]
        parts: list[str] = []
        if top_drop:
            parts.append("Drop: " + ", ".join(f"`{k}` ({v})" for k, v in top_drop))
        if top_salv:
            parts.append("Salvage: " + ", ".join(f"`{k}` ({v})" for k, v in top_salv))
        top_str = "; ".join(parts) if parts else "—"
        lines.append(
            f"| `{name}` | {stream_cell} | {keep:,} | {det_sal:,} | "
            f"{salvage:,} | {drop:,} | {keep_pct:.1f}% | {top_str} |"
        )

    if has_arc_expansion:
        arc_st = per_source.get("arc_agi2_train", {})
        arc_pairs = int(arc_st.get("keep", 0)) + int(arc_st.get("salvage", 0)) + int(arc_st.get("drop", 0))
        lines.extend([
            "",
            f"*\\*`arc_agi2_train` streams {int(arc_st.get('streamed', 0)):,} JSON task files, which expand "
            f"to {arc_pairs:,} individual test I/O pairs.*",
        ])

    lines.extend([
        "",
        "## Anti-Collapse, Anti-Slop & Looped-CoT Methodology",
        "",
        "1. **3-Beat Looped-CoT Structure:** Reasoning traces are scored for Beat 1 "
        "(constraint framing & hypothesis), Beat 2 (adversarial self-audit, substitution checks, "
        "edge-case testing, backtracking), and Beat 3 (concise synthesis).",
        "2. **Anti-Slop Voice Filtering:** Detects sycophantic openers, AI cliché phrases "
        "(*delve*, *tapestry*, *it's important to note*, *in conclusion*, *let's break this down*), "
        "bold-bullet template walls, and low lexical diversity (`distinct_2`/`distinct_3`).",
        "3. **Pathology & Domain Gates:** Hard-drops unclosed `<think>`/` ``` `, repetition loops, "
        "refusals/incomplete prompts, corrupted bytes, Python `ast.parse` syntax failures, invalid "
        "ARC grids, and malformed tool JSON. Placeholder-redacted tool arguments "
        "(e.g. `\"<bash_command>\"` in `fable51`) are quarantined to `salvage_queue` rather than "
        "poisoning tool SFT.",
        "4. **Cross-Source Deduplication:** Exact SHA-256 on `(prompt, response)` and `prompt`, "
        "plus 64-hash banded MinHash / 3-word shingle Jaccard near-deduplication (`>= 0.85`). "
        f"Unique pairs indexed: **{dedup.get('unique_indexed', 0):,}**; "
        f"duplicates removed: `{dedup.get('dup_counts', {})}`.",
        "",
        "## 13-Gram Decontamination Against 2026 Benchmarks",
        "",
        f"Reuses `qloop.data.decontam.Decontaminator(n={decontam.get('ngram', 13)})` indexing "
        f"**{decontam.get('index_size', 0):,}** unique n-grams across held-out 2026 eval suites "
        f"(total contamination hits dropped: **{sum((decontam.get('hits') or {}).values()):,}**).",
        "",
        "| Eval Suite | Group | Status | Texts Indexed |",
        "| :--- | :--- | :--- | ---: |",
    ])

    for sname, sinfo in (decontam.get("suites") or {}).items():
        lines.append(
            f"| `{sname}` | `{sinfo.get('group', '')}` | {sinfo.get('status', '')} | {int(sinfo.get('texts', 0)):,} |"
        )

    lines.extend([
        "",
        "## Upstream Attribution & Citations (BibTeX)",
        "",
        "If you use this dataset, please cite **all** included upstream sources alongside this compilation:",
        "",
        "```bibtex",
        AUTHOR_ORIGINAL_CITATION.bibtex,
    ])
    for src in UPSTREAM_SOURCES.values():
        if src.bibtex:
            lines.append("")
            lines.append(src.bibtex)
    lines.extend(["```", ""])
    return "\n".join(lines)


# ============================================================================ Pipeline Orchestrator

def run_streaming_pipeline(
    *,
    out_dir: str | Path,
    source_names: Sequence[str] | None = None,
    max_per_source: int = 1000,
    golden_seeds_path: str | Path | None = None,
    decontam_cache_dir: str | Path | None = None,
    eval_texts_override: dict[str, Sequence[str]] | None = None,
    skip_decontam: bool = False,
    audit_only: bool = False,
    live_license_check: bool = True,
    token: str | None = None,
    repo_id: str = "TassP0lar/Loop64-Fable-SIRSI-v1",
    row_stream_override: dict[str, Iterable[tuple[str, int, dict[str, Any]]]] | None = None,
    extra_sft_rows: Sequence[dict[str, Any]] = (),
    extra_pref_rows: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Execute end-to-end streaming audit, triage, salvage, dedup, decontam, and local Parquet build."""
    t0 = time.time()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Verify upstream licenses (includes both included and excluded sources for transparent audit)
    selected_sources = (
        [ALL_AUDITED_SOURCES[n] for n in source_names if n in ALL_AUDITED_SOURCES]
        if source_names
        else list(ALL_AUDITED_SOURCES.values())
    )
    license_report = verify_upstream_licenses(selected_sources, token=token, live=live_license_check)

    # 2. Build or load 13-gram decontaminator
    decontam: Decontaminator | None = None
    decontam_report: dict[str, Any] = {"ngram": 13, "index_size": 0, "suites": {}, "hits": {}}
    if not skip_decontam:
        decontam, decontam_report = build_or_load_decontaminator(
            cache_dir=decontam_cache_dir,
            eval_texts_override=eval_texts_override,
            n=13,
            token=token,
            on_missing="warn",
        )

    dedup = CrossSourceDeduplicator()
    sft_core: list[dict[str, Any]] = []
    salvage_queue: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    per_source_stats: dict[str, dict[str, Any]] = {}
    sample_examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    # 3. Ingest author-original golden seeds first (so they anchor dedup and are guaranteed present)
    n_golden_sft = 0
    n_golden_pref = 0
    if golden_seeds_path and Path(golden_seeds_path).exists():
        g_sft, g_pref = load_golden_seeds(golden_seeds_path, decontaminator=decontam)
        for rec in g_sft:
            dedup.check_and_add("sirsi_golden_seeds", rec["prompt"], rec["response"])
        sft_core.extend(g_sft)
        preferences.extend(g_pref)
        n_golden_sft = len(g_sft)
        n_golden_pref = len(g_pref)
        per_source_stats["sirsi_golden_seeds"] = {
            "hf_id": AUTHOR_ORIGINAL_CITATION.hf_id,
            "license": "apache-2.0",
            "streamed": len(g_sft),
            "keep": len(g_sft),
            "deterministic_salvaged_to_keep": 0,
            "salvage": 0,
            "drop": 0,
            "drop_reasons": {},
            "salvage_reasons": {},
            "salvage_actions": {},
            "mean_slop": round(sum(r["quality_scores"]["slop_score"] for r in g_sft) / max(1, len(g_sft)), 4),
            "mean_3beat": round(sum(r["quality_scores"]["three_beat_score"] for r in g_sft) / max(1, len(g_sft)), 4),
        }

    # 4. Stream each audited upstream source
    for src in selected_sources:
        lic_info = license_report.get(src.name, {})
        if not lic_info.get("included", False):
            per_source_stats[src.name] = {
                "hf_id": src.hf_id,
                "license": lic_info.get("live_license", src.license),
                "excluded": True,
                "exclusion_reason": lic_info.get("reason", src.exclusion_reason),
                "streamed": 0,
                "keep": 0,
                "deterministic_salvaged_to_keep": 0,
                "salvage": 0,
                "drop": 0,
                "drop_reasons": {lic_info.get("reason", "license_excluded"): 1},
                "salvage_reasons": {},
                "salvage_actions": {},
                "mean_slop": 0.0,
                "mean_3beat": 0.0,
            }
            continue

        spec = get_spec(src.name) if src.name in REGISTRY else DatasetSpec(
            name=src.name, hf_id=src.hf_id, config=src.config, split=src.split, license=src.license,
        )
        # Allow empty content/reasoning during initial normalize so our judge produces exact triage reasons
        audit_spec = spec.with_overrides(require_reasoning=False, allow_empty_content=True, drop_system_prompts=[])

        st_streamed = 0
        st_keep = 0
        st_det_sal = 0
        st_salvage = 0
        st_drop = 0
        drop_reasons: collections.Counter[str] = collections.Counter()
        salvage_reasons: collections.Counter[str] = collections.Counter()
        salvage_actions: collections.Counter[str] = collections.Counter()
        slop_sum = 0.0
        beat_sum = 0.0
        scored_n = 0

        stream_it = (
            row_stream_override[src.name]
            if (row_stream_override is not None and src.name in row_stream_override)
            else stream_source_rows(src, max_rows=max_per_source, token=token)
        )

        for spl, idx, raw_row in stream_it:
            st_streamed += 1
            try:
                exs = normalize_row(raw_row, audit_spec, idx)
            except RowFormatError as exc:
                st_drop += 1
                drop_reasons[exc.reason] += 1
                continue
            except RowDrop as exc:
                st_drop += 1
                drop_reasons[exc.reason] += 1
                continue

            for ex in exs:
                val_err = validate_example(ex, audit_spec)
                if val_err and val_err not in ("reasoning_empty", "empty_content"):
                    st_drop += 1
                    drop_reasons[val_err] += 1
                    continue

                raw_meta = {
                    "category": raw_row.get("category") or raw_row.get("domain"),
                    "split": spl,
                }
                before_resp = ex.response_text()
                dec = triage_example(
                    ex,
                    src,
                    split_used=spl,
                    raw_meta=raw_meta,
                    decontaminator=decontam,
                    deduplicator=dedup,
                    apply_deterministic_salvage=True,
                )

                for act in dec.salvage_actions:
                    salvage_actions[act] += 1

                if dec.scores:
                    slop_sum += dec.scores["slop_score"]
                    beat_sum += dec.scores["three_beat_score"]
                    scored_n += 1

                if dec.verdict == "KEEP" and dec.record is not None:
                    st_keep += 1
                    if dec.salvage_actions:
                        st_det_sal += 1
                        if len(sample_examples["deterministic_salvage"]) < 6:
                            sample_examples["deterministic_salvage"].append({
                                "source": src.name,
                                "id": dec.record["id"],
                                "actions": dec.salvage_actions,
                                "before_snippet": before_resp[:260],
                                "after_snippet": dec.record["response"][:260],
                            })
                    sft_core.append(dec.record)
                elif dec.verdict == "SALVAGE" and dec.record is not None:
                    st_salvage += 1
                    for r in dec.reasons:
                        salvage_reasons[r] += 1
                    if len(sample_examples[f"salvage:{src.name}"]) < 2:
                        sample_examples[f"salvage:{src.name}"].append({
                            "id": dec.record["id"],
                            "reasons": dec.reasons,
                            "scores": dec.scores,
                            "prompt_snip": dec.record["prompt"][:180],
                            "response_snip": dec.record["response"][:220],
                        })
                    salvage_queue.append(dec.record)
                else:
                    st_drop += 1
                    for r in dec.reasons:
                        drop_reasons[r] += 1

        per_source_stats[src.name] = {
            "hf_id": src.hf_id,
            "license": lic_info.get("live_license", src.license),
            "excluded": False,
            "streamed": st_streamed,
            "keep": st_keep,
            "deterministic_salvaged_to_keep": st_det_sal,
            "salvage": st_salvage,
            "drop": st_drop,
            "drop_reasons": dict(drop_reasons.most_common(12)),
            "salvage_reasons": dict(salvage_reasons.most_common(12)),
            "salvage_actions": dict(salvage_actions.most_common(12)),
            "mean_slop": round(slop_sum / max(1, scored_n), 4),
            "mean_3beat": round(beat_sum / max(1, scored_n), 4),
        }
        log.info(
            "[audit] %-16s streamed=%4d KEEP=%4d (det_sal=%4d) SALVAGE=%4d DROP=%4d",
            src.name, st_streamed, st_keep, st_det_sal, st_salvage, st_drop,
        )

    # Merge any Track B synthesized rows supplied by caller
    sft_core.extend(extra_sft_rows)
    preferences.extend(extra_pref_rows)

    if decontam is not None:
        decontam_report["hits"] = decontam.summary()["hits"]
        decontam_report["checked"] = decontam.summary()["checked"]

    audit_summary: dict[str, Any] = {
        "repo_id": repo_id,
        "elapsed_sec": round(time.time() - t0, 2),
        "max_per_source": max_per_source,
        "output_counts": {
            "sft_core": len(sft_core),
            "salvage_queue": len(salvage_queue),
            "sirsi_preferences": len(preferences),
            "golden_seeds_sft": n_golden_sft,
            "golden_seeds_pref": n_golden_pref,
            "track_b_sft": len(extra_sft_rows),
            "track_b_pref": len(extra_pref_rows),
        },
        "licenses": license_report,
        "per_source": per_source_stats,
        "dedup": dedup.summary(),
        "decontam": decontam_report,
        "sample_examples": dict(sample_examples),
    }

    (out_path / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")

    if not audit_only:
        data_dir = out_path / "data"
        write_parquet_records(sft_core, data_dir / "sft_core.parquet", kind="sft")
        write_parquet_records(salvage_queue, data_dir / "salvage_queue.parquet", kind="sft")
        write_parquet_records(preferences, data_dir / "sirsi_preferences.parquet", kind="pref")
        readme_text = render_dataset_card(audit_summary, repo_id=repo_id)
        (out_path / "README.md").write_text(readme_text, encoding="utf-8")

    return audit_summary


# ============================================================================ HF Hub Publishing & Dry-Run Validator

_SECRET_LEAK_RE = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b|\bsk-[A-Za-z0-9]{20,}\b|\bAIza[0-9A-Za-z_\-]{30,}\b")


def publish_to_hub(
    out_dir: str | Path,
    repo_id: str,
    *,
    public: bool = True,
    dry_run: bool = True,
    token: str | None = None,
) -> dict[str, Any]:
    """Validate local dataset directory, YAML card, Parquet integrity, and secret hygiene; push if not dry_run."""
    import pyarrow.parquet as pq
    import yaml

    root = Path(out_dir)
    expected_files = [
        "README.md",
        "audit_summary.json",
        "data/sft_core.parquet",
        "data/salvage_queue.parquet",
        "data/sirsi_preferences.parquet",
    ]
    missing = [rel for rel in expected_files if not (root / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Dataset directory {root} is missing required files: {missing}")

    # Validate README YAML frontmatter
    readme_txt = (root / "README.md").read_text(encoding="utf-8")
    m_yaml = re.match(r"^---\s*\n(.*?)\n---\s*\n", readme_txt, re.DOTALL)
    if not m_yaml:
        raise ValueError("README.md is missing valid Hugging Face YAML frontmatter")
    card_meta = yaml.safe_load(m_yaml.group(1))
    if not isinstance(card_meta, dict) or "license" not in card_meta or "configs" not in card_meta:
        raise ValueError("README.md YAML frontmatter must declare 'license' and 'configs'")

    # Secret-leak scan across text artifacts (both generic token patterns and active HF token)
    active_tok = token or os.environ.get("HF_TOKEN") or ""
    for rel in ("README.md", "audit_summary.json"):
        txt = (root / rel).read_text(encoding="utf-8")
        if _SECRET_LEAK_RE.search(txt) or (active_tok and active_tok in txt):
            raise RuntimeError(f"Refusing to publish: potential secret pattern detected in {rel}")

    manifest_files: list[dict[str, Any]] = []
    total_bytes = 0
    for rel in expected_files:
        fp = root / rel
        size = fp.stat().st_size
        total_bytes += size
        sha = hashlib.sha256(fp.read_bytes()).hexdigest()
        entry: dict[str, Any] = {"path": rel, "size_bytes": size, "sha256": sha[:16]}
        if rel.endswith(".parquet"):
            pf = pq.ParquetFile(fp)
            entry["num_rows"] = int(pf.metadata.num_rows)
            entry["num_columns"] = len(pf.schema_arrow.names)
        manifest_files.append(entry)

    result: dict[str, Any] = {
        "repo_id": repo_id,
        "visibility": "public" if public else "private",
        "dry_run": bool(dry_run),
        "validated": True,
        "yaml_license": card_meta.get("license"),
        "yaml_configs": [c.get("config_name") for c in (card_meta.get("configs") or [])],
        "total_bytes": total_bytes,
        "files": manifest_files,
        "uploaded": False,
    }

    if dry_run:
        return result

    from huggingface_hub import HfApi

    tok = token or os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN is required when --push-to-hub is run without --dry-run")
    api = HfApi(token=tok)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=not public, exist_ok=True)
    commit_info = api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=expected_files,
        commit_message="Upload Loop64-Fable-SIRSI-v1 curated Parquet shards and Dataset Card",
    )
    result["uploaded"] = True
    result["commit_url"] = str(getattr(commit_info, "commit_url", ""))
    return result


__all__ = [
    "ALLOWED_LICENSES",
    "ALL_AUDITED_SOURCES",
    "AUTHOR_ORIGINAL_CITATION",
    "DEFAULT_EVAL_SUITES",
    "EXCLUDED_SOURCES",
    "PROVIDER_DEFAULTS",
    "UPSTREAM_SOURCES",
    "AsyncTeacherClient",
    "CrossSourceDeduplicator",
    "SourceCitation",
    "TrackBSynthesizer",
    "TriageDecision",
    "build_or_load_decontaminator",
    "check_domain_sanity",
    "compute_distinct_n",
    "detect_pathology",
    "detect_synthetic_cot_artifacts",
    "deterministic_salvage",
    "load_golden_seeds",
    "publish_to_hub",
    "render_dataset_card",
    "run_streaming_pipeline",
    "score_anti_slop",
    "score_three_beat_cot",
    "stream_source_rows",
    "strip_eval_boilerplate",
    "strip_rubric_headers",
    "triage_example",
    "verify_synthesized_row",
    "verify_upstream_licenses",
    "write_jsonl_records",
    "write_parquet_records",
]
