#!/usr/bin/env python3
"""Deterministic forensic tokenizer parity & safetensors PLE header audit CLI.

Compares `tokenizer.json` (all 18 sections) and `tokenizer_config.json` between a
base Hugging Face repository (default: `Qwen/Qwen3.6-35B-A3B`) and a candidate donor
repository (default: `Qwen/Qwen3.8-Flash-Next`), either live via stdlib HTTPS or
offline from local directories (`--base-dir` / `--donor-dir`).

Optionally inspects `.ple.` safetensors headers via minimal HTTP `Range` requests
(`--check-ple-headers`) without downloading multi-GB shards.

Stdlib-only, Python 3.9+ compatible.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_BASE_REPO = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_DONOR_REPO = "Qwen/Qwen3.8-Flash-Next"
HF_BASE_URL = "https://huggingface.co"

MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024  # 256 MiB safety cap
LAYER_ID_RE = re.compile(r"\.layers\.(\d+)\.")

TOP_LEVEL_SECTIONS: tuple[str, ...] = (
    "version",
    "truncation",
    "padding",
    "normalizer",
    "pre_tokenizer",
    "post_processor",
    "decoder",
)

MODEL_SCALAR_SECTIONS: tuple[str, ...] = (
    "type",
    "dropout",
    "unk_token",
    "continuing_subword_prefix",
    "end_of_word_suffix",
    "fuse_unk",
    "byte_fallback",
    "ignore_merges",
)

KEY_CONFIG_FIELDS: tuple[str, ...] = (
    "eos_token",
    "pad_token",
    "bos_token",
    "model_max_length",
)


def diff_added_tokens(
    base_added: list[dict[str, Any]],
    donor_added: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare `added_tokens` arrays by ID, separating shared parity from appended IDs."""
    base_by_id: dict[int, dict[str, Any]] = {int(tok["id"]): tok for tok in base_added}
    donor_by_id: dict[int, dict[str, Any]] = {int(tok["id"]): tok for tok in donor_added}

    shared_ids = sorted(set(base_by_id) & set(donor_by_id))
    removed_ids = sorted(set(base_by_id) - set(donor_by_id))
    extra_ids = sorted(set(donor_by_id) - set(base_by_id))

    modified_shared: list[dict[str, Any]] = []
    for tid in shared_ids:
        if base_by_id[tid] != donor_by_id[tid]:
            modified_shared.append(
                {
                    "id": tid,
                    "base_content": base_by_id[tid].get("content"),
                    "donor_content": donor_by_id[tid].get("content"),
                }
            )

    max_base_id = max(base_by_id) if base_by_id else -1
    appended_tokens = [
        {"id": tid, "content": donor_by_id[tid].get("content")}
        for tid in extra_ids
    ]
    removed_tokens = [
        {"id": tid, "content": base_by_id[tid].get("content")}
        for tid in removed_ids
    ]
    strictly_appended = (
        len(removed_ids) == 0
        and len(modified_shared) == 0
        and all(tid > max_base_id for tid in extra_ids)
    )
    shared_exact_equal = len(removed_ids) == 0 and len(modified_shared) == 0

    return {
        "exact_equal": base_added == donor_added,
        "shared_exact_equal": shared_exact_equal,
        "strictly_appended": strictly_appended,
        "base_count": len(base_added),
        "donor_count": len(donor_added),
        "shared_count": len(shared_ids),
        "shared_id_range": [shared_ids[0], shared_ids[-1]] if shared_ids else None,
        "appended_count": len(appended_tokens),
        "appended_tokens": appended_tokens,
        "removed_count": len(removed_tokens),
        "removed_tokens": removed_tokens,
        "modified_shared_count": len(modified_shared),
        "modified_shared_tokens": modified_shared,
    }


def diff_vocab(
    base_vocab: dict[str, int],
    donor_vocab: dict[str, int],
    *,
    preview_limit: int = 10,
) -> dict[str, Any]:
    """Compare `model.vocab` token-to-ID mappings deterministically."""
    if base_vocab == donor_vocab:
        ids = list(base_vocab.values())
        return {
            "exact_equal": True,
            "base_size": len(base_vocab),
            "donor_size": len(donor_vocab),
            "id_range": [min(ids), max(ids)] if ids else None,
            "missing_in_donor_count": 0,
            "extra_in_donor_count": 0,
            "id_mismatch_count": 0,
            "missing_in_donor_preview": [],
            "extra_in_donor_preview": [],
            "id_mismatch_preview": [],
        }

    base_keys = set(base_vocab)
    donor_keys = set(donor_vocab)
    missing = sorted(base_keys - donor_keys)
    extra = sorted(donor_keys - base_keys)
    mismatches = [
        {"token": tok, "base_id": base_vocab[tok], "donor_id": donor_vocab[tok]}
        for tok in sorted(base_keys & donor_keys)
        if base_vocab[tok] != donor_vocab[tok]
    ]
    base_ids = list(base_vocab.values())
    return {
        "exact_equal": False,
        "base_size": len(base_vocab),
        "donor_size": len(donor_vocab),
        "id_range": [min(base_ids), max(base_ids)] if base_ids else None,
        "missing_in_donor_count": len(missing),
        "extra_in_donor_count": len(extra),
        "id_mismatch_count": len(mismatches),
        "missing_in_donor_preview": missing[:preview_limit],
        "extra_in_donor_preview": extra[:preview_limit],
        "id_mismatch_preview": mismatches[:preview_limit],
    }


def diff_merges(base_merges: list[Any], donor_merges: list[Any]) -> dict[str, Any]:
    """Compare `model.merges` rule lists deterministically."""
    if base_merges == donor_merges:
        return {
            "exact_equal": True,
            "base_size": len(base_merges),
            "donor_size": len(donor_merges),
            "first_diff_index": None,
        }

    first_diff: int | None = None
    limit = min(len(base_merges), len(donor_merges))
    for idx in range(limit):
        if base_merges[idx] != donor_merges[idx]:
            first_diff = idx
            break
    if first_diff is None and len(base_merges) != len(donor_merges):
        first_diff = limit

    return {
        "exact_equal": False,
        "base_size": len(base_merges),
        "donor_size": len(donor_merges),
        "first_diff_index": first_diff,
    }


def diff_tokenizer_json(
    base_tok: dict[str, Any],
    donor_tok: dict[str, Any],
) -> dict[str, Any]:
    """Diff all 18 sections of `tokenizer.json` deterministically."""
    sections: dict[str, dict[str, Any]] = {}

    for key in TOP_LEVEL_SECTIONS:
        eq = base_tok.get(key) == donor_tok.get(key)
        sections[key] = {
            "exact_equal": eq,
            "present_in_base": key in base_tok,
            "present_in_donor": key in donor_tok,
        }

    added_diff = diff_added_tokens(
        list(base_tok.get("added_tokens") or []),
        list(donor_tok.get("added_tokens") or []),
    )
    sections["added_tokens"] = added_diff

    base_model = dict(base_tok.get("model") or {})
    donor_model = dict(donor_tok.get("model") or {})

    for key in MODEL_SCALAR_SECTIONS:
        eq = base_model.get(key) == donor_model.get(key)
        sections[f"model.{key}"] = {
            "exact_equal": eq,
            "base_value": base_model.get(key),
            "donor_value": donor_model.get(key),
        }

    vocab_diff = diff_vocab(
        dict(base_model.get("vocab") or {}),
        dict(donor_model.get("vocab") or {}),
    )
    sections["model.vocab"] = vocab_diff

    merges_diff = diff_merges(
        list(base_model.get("merges") or []),
        list(donor_model.get("merges") or []),
    )
    sections["model.merges"] = merges_diff

    known_top = set(TOP_LEVEL_SECTIONS) | {"added_tokens", "model"}
    known_model = set(MODEL_SCALAR_SECTIONS) | {"vocab", "merges"}
    extra_top_keys = sorted((set(base_tok) | set(donor_tok)) - known_top)
    extra_model_keys = sorted((set(base_model) | set(donor_model)) - known_model)
    extra_top_equal = all(base_tok.get(k) == donor_tok.get(k) for k in extra_top_keys)
    extra_model_equal = all(base_model.get(k) == donor_model.get(k) for k in extra_model_keys)

    non_added_keys = [k for k in sections if k != "added_tokens"]
    non_added_equal_count = sum(1 for k in non_added_keys if sections[k]["exact_equal"])
    shared_equal_count = non_added_equal_count + (1 if added_diff["shared_exact_equal"] else 0)

    shared_core_parity_ok = (
        non_added_equal_count == len(non_added_keys)
        and added_diff["shared_exact_equal"]
        and added_diff["strictly_appended"]
        and extra_top_equal
        and extra_model_equal
    )

    return {
        "sections_checked": len(sections),
        "sections_shared_exact_equal": shared_equal_count,
        "shared_core_parity_ok": shared_core_parity_ok,
        "extra_top_keys": extra_top_keys,
        "extra_top_equal": extra_top_equal,
        "extra_model_keys": extra_model_keys,
        "extra_model_equal": extra_model_equal,
        "sections": sections,
    }


def diff_tokenizer_config(
    base_cfg: dict[str, Any],
    donor_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compare `tokenizer_config.json` across all keys and highlight critical fields."""
    all_keys = sorted(set(base_cfg) | set(donor_cfg))
    base_only = sorted(set(base_cfg) - set(donor_cfg))
    donor_only = sorted(set(donor_cfg) - set(base_cfg))
    diff_keys = [
        k for k in sorted(set(base_cfg) & set(donor_cfg))
        if base_cfg[k] != donor_cfg[k]
    ]
    key_fields = {
        k: {
            "exact_equal": base_cfg.get(k) == donor_cfg.get(k),
            "base": base_cfg.get(k),
            "donor": donor_cfg.get(k),
        }
        for k in KEY_CONFIG_FIELDS
    }
    return {
        "exact_equal": base_cfg == donor_cfg,
        "total_keys_checked": len(all_keys),
        "diff_keys": diff_keys,
        "base_only_keys": base_only,
        "donor_only_keys": donor_only,
        "key_fields": key_fields,
    }


def parse_safetensors_header_length(prefix: bytes) -> int:
    """Decode the 8-byte little-endian uint64 header length of a safetensors file."""
    if len(prefix) < 8:
        raise ValueError(f"Safetensors prefix must be >= 8 bytes, got {len(prefix)}")
    (header_len,) = struct.unpack("<Q", prefix[:8])
    if header_len == 0 or header_len > MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"Invalid safetensors header length: {header_len}")
    return int(header_len)


def parse_safetensors_header_json(header_bytes: bytes) -> dict[str, Any]:
    """Parse UTF-8 safetensors JSON header, excluding `__metadata__`."""
    raw = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Safetensors header must be a JSON object")
    return {k: v for k, v in raw.items() if k != "__metadata__"}


def summarize_ple_headers(
    shard_headers: dict[str, dict[str, Any]],
    config_json: dict[str, Any] | None = None,
    *,
    pattern: str = ".ple.",
) -> dict[str, Any]:
    """Summarize `.ple.` tensors from inspected safetensors headers."""
    matched: dict[str, dict[str, Any]] = {}
    for shard_name in sorted(shard_headers):
        hdr = shard_headers[shard_name]
        for tname in sorted(hdr):
            if pattern not in tname:
                continue
            meta = hdr[tname]
            shape = [int(d) for d in meta.get("shape", [])]
            numel = int(math.prod(shape)) if shape else 1
            matched[tname] = {
                "shard": shard_name,
                "dtype": str(meta.get("dtype", "")),
                "shape": shape,
                "numel": numel,
            }

    layer_ids: set[int] = set()
    by_shape_dtype: dict[tuple[str, tuple[int, ...]], list[str]] = defaultdict(list)
    total_params = 0
    for tname, info in sorted(matched.items()):
        total_params += info["numel"]
        by_shape_dtype[(info["dtype"], tuple(info["shape"]))].append(tname)
        m = LAYER_ID_RE.search(tname)
        if m:
            layer_ids.add(int(m.group(1)))

    grouped_families = [
        {
            "dtype": dtype,
            "shape": list(shape),
            "count": len(names),
            "params_per_tensor": int(math.prod(shape)) if shape else 1,
            "total_params": len(names) * (int(math.prod(shape)) if shape else 1),
            "sample_tensors": names[:3],
        }
        for (dtype, shape), names in sorted(
            by_shape_dtype.items(),
            key=lambda kv: (-len(kv[1]) * (int(math.prod(kv[0][1])) if kv[0][1] else 1), kv[0]),
        )
    ]

    config_ple_layer_ids = None
    model_type = None
    if isinstance(config_json, dict):
        model_type = config_json.get("model_type")
        if "ple_layer_ids" in config_json:
            config_ple_layer_ids = config_json.get("ple_layer_ids")
        elif isinstance(config_json.get("text_config"), dict):
            config_ple_layer_ids = config_json["text_config"].get("ple_layer_ids")

    header_layer_ids = sorted(layer_ids)
    layer_id_mismatch = (
        config_ple_layer_ids is not None
        and list(config_ple_layer_ids) != header_layer_ids
    )

    return {
        "pattern": pattern,
        "shards_inspected": sorted(shard_headers),
        "matched_tensor_count": len(matched),
        "total_params": total_params,
        "header_layer_ids": header_layer_ids,
        "config_model_type": model_type,
        "config_ple_layer_ids": config_ple_layer_ids,
        "config_vs_header_layer_id_mismatch": layer_id_mismatch,
        "tensor_families": grouped_families,
        "tensors": matched,
    }


def _hf_resolve_url(repo: str, filename: str, revision: str) -> str:
    quoted_file = urllib.parse.quote(filename, safe="/")
    quoted_rev = urllib.parse.quote(revision, safe="")
    return f"{HF_BASE_URL}/{repo}/resolve/{quoted_rev}/{quoted_file}"


def fetch_bytes(
    url: str,
    *,
    byte_range: tuple[int, int] | None = None,
    timeout: float = 30.0,
    token: str | None = None,
) -> bytes:
    """Fetch bytes over HTTPS via stdlib urllib (with optional byte Range & HF_TOKEN)."""
    headers = {"User-Agent": "qloop-tokenizer-parity/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if byte_range is not None:
        start, end = byte_range
        headers["Range"] = f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_json_from_source(
    repo: str,
    filename: str,
    *,
    local_dir: Path | None,
    revision: str,
    timeout: float,
    token: str | None,
) -> dict[str, Any]:
    """Load JSON from a local directory if given, otherwise from Hugging Face Hub."""
    if local_dir is not None:
        path = local_dir / filename
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    raw = fetch_bytes(_hf_resolve_url(repo, filename, revision), timeout=timeout, token=token)
    return json.loads(raw.decode("utf-8"))


def read_safetensors_header_local(path: Path) -> dict[str, Any]:
    """Read only the JSON header of a local `.safetensors` file."""
    with path.open("rb") as f:
        header_len = parse_safetensors_header_length(f.read(8))
        header_bytes = f.read(header_len)
    if len(header_bytes) != header_len:
        raise ValueError(f"Truncated safetensors header in {path}")
    return parse_safetensors_header_json(header_bytes)


def fetch_safetensors_header_http(
    repo: str,
    shard_name: str,
    *,
    revision: str,
    timeout: float,
    token: str | None,
) -> dict[str, Any]:
    """Read only the JSON header of a remote `.safetensors` shard using HTTP Range."""
    url = _hf_resolve_url(repo, shard_name, revision)
    prefix = fetch_bytes(url, byte_range=(0, 7), timeout=timeout, token=token)
    header_len = parse_safetensors_header_length(prefix)
    header_bytes = fetch_bytes(url, byte_range=(8, 8 + header_len - 1), timeout=timeout, token=token)
    if len(header_bytes) < header_len:
        raise ValueError(f"Truncated HTTP Range safetensors header for {shard_name}")
    return parse_safetensors_header_json(header_bytes[:header_len])


def inspect_ple_headers_from_source(
    repo: str,
    *,
    local_dir: Path | None,
    revision: str,
    pattern: str,
    timeout: float,
    token: str | None,
) -> dict[str, Any]:
    """Locate `.ple.` shards from `model.safetensors.index.json` and read only their headers."""
    index_json = load_json_from_source(
        repo,
        "model.safetensors.index.json",
        local_dir=local_dir,
        revision=revision,
        timeout=timeout,
        token=token,
    )
    weight_map = dict(index_json.get("weight_map") or {})
    target_shards = sorted({shard for tname, shard in weight_map.items() if pattern in tname})

    config_json: dict[str, Any] | None = None
    try:
        config_json = load_json_from_source(
            repo,
            "config.json",
            local_dir=local_dir,
            revision=revision,
            timeout=timeout,
            token=token,
        )
    except (FileNotFoundError, urllib.error.URLError, OSError):
        config_json = None

    shard_headers: dict[str, dict[str, Any]] = {}
    for shard_name in target_shards:
        if local_dir is not None:
            shard_headers[shard_name] = read_safetensors_header_local(local_dir / shard_name)
        else:
            shard_headers[shard_name] = fetch_safetensors_header_http(
                repo,
                shard_name,
                revision=revision,
                timeout=timeout,
                token=token,
            )

    summary = summarize_ple_headers(shard_headers, config_json=config_json, pattern=pattern)
    summary["total_index_shards"] = len(set(weight_map.values()))
    return summary


def build_parity_report(
    *,
    base_repo: str,
    donor_repo: str,
    base_tok: dict[str, Any],
    donor_tok: dict[str, Any],
    base_cfg: dict[str, Any],
    donor_cfg: dict[str, Any],
    ple_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the complete deterministic parity audit report."""
    tok_diff = diff_tokenizer_json(base_tok, donor_tok)
    cfg_diff = diff_tokenizer_config(base_cfg, donor_cfg)
    overall_parity_ok = bool(tok_diff["shared_core_parity_ok"] and cfg_diff["exact_equal"])
    report: dict[str, Any] = {
        "base_repo": base_repo,
        "donor_repo": donor_repo,
        "overall_shared_parity_ok": overall_parity_ok,
        "tokenizer_json": tok_diff,
        "tokenizer_config_json": cfg_diff,
    }
    if ple_summary is not None:
        report["ple_headers"] = ple_summary
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a concise, deterministic Markdown summary of the parity audit."""
    tok = report["tokenizer_json"]
    cfg = report["tokenizer_config_json"]
    secs = tok["sections"]
    added = secs["added_tokens"]
    vocab = secs["model.vocab"]
    merges = secs["model.merges"]

    lines: list[str] = [
        f"# Tokenizer Parity Audit: `{report['base_repo']}` vs. `{report['donor_repo']}`",
        "",
        f"- **Overall Shared Parity (`tokenizer.json` + `tokenizer_config.json`):** "
        f"`{report['overall_shared_parity_ok']}`",
        f"- **`tokenizer.json` Shared Sections Equal:** "
        f"`{tok['sections_shared_exact_equal']}/{tok['sections_checked']}`",
        f"- **`tokenizer_config.json` Exact Equal:** `{cfg['exact_equal']}` "
        f"(`{cfg['total_keys_checked']}` keys checked)",
        "",
        "## `tokenizer.json` 18-Section Parity Table",
        "",
        "| Section | Shared Exact Equal | Details |",
        "| :--- | :---: | :--- |",
    ]

    for name in sorted(secs):
        sec = secs[name]
        if name == "added_tokens":
            eq = sec["shared_exact_equal"]
            detail = (
                f"shared={sec['shared_count']} ({sec['shared_id_range']}), "
                f"appended={sec['appended_count']}, removed={sec['removed_count']}, "
                f"modified={sec['modified_shared_count']}"
            )
        elif name == "model.vocab":
            eq = sec["exact_equal"]
            detail = f"base={vocab['base_size']:,}, donor={vocab['donor_size']:,}, id_range={vocab['id_range']}"
        elif name == "model.merges":
            eq = sec["exact_equal"]
            detail = f"base={merges['base_size']:,}, donor={merges['donor_size']:,}"
        else:
            eq = sec["exact_equal"]
            detail = "exact match" if eq else "DIFF"
        lines.append(f"| `{name}` | `{eq}` | {detail} |")

    if added["appended_tokens"]:
        tok_list = ", ".join(f"`{t['id']}:{t['content']}`" for t in added["appended_tokens"])
        lines.extend(["", f"**Appended Donor `added_tokens` ({added['appended_count']}):** {tok_list}"])

    lines.extend(
        [
            "",
            "## `tokenizer_config.json` Key Fields",
            "",
            "| Field | Base | Donor | Exact Equal |",
            "| :--- | :--- | :--- | :---: |",
        ]
    )
    for k in KEY_CONFIG_FIELDS:
        kf = cfg["key_fields"][k]
        lines.append(f"| `{k}` | `{kf['base']}` | `{kf['donor']}` | `{kf['exact_equal']}` |")

    if "ple_headers" in report:
        ple = report["ple_headers"]
        lines.extend(
            [
                "",
                f"## Safetensors `{ple['pattern']}` Header Audit (`{report['donor_repo']}`)",
                "",
                f"- **Shards Inspected:** `{', '.join(ple['shards_inspected']) or 'none'}` "
                f"(out of `{ple.get('total_index_shards', len(ple['shards_inspected']))}` total)",
                f"- **Matched Tensors:** `{ple['matched_tensor_count']}` · "
                f"**Total Params:** `{ple['total_params']:,}`",
                f"- **Header 0-Indexed Layer IDs:** `{ple['header_layer_ids']}` · "
                f"**`config.json` `ple_layer_ids`:** `{ple['config_ple_layer_ids']}` · "
                f"**Layer-ID Mismatch:** `{ple['config_vs_header_layer_id_mismatch']}`",
                "",
                "| Dtype | Shape | Count | Total Params | Sample Tensor |",
                "| :---: | :--- | ---: | ---: | :--- |",
            ]
        )
        for fam in ple["tensor_families"]:
            sample = fam["sample_tensors"][0] if fam["sample_tensors"] else ""
            lines.append(
                f"| `{fam['dtype']}` | `{fam['shape']}` | `{fam['count']}` | "
                f"`{fam['total_params']:,}` | `{sample}` |"
            )

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Verify tokenizer parity between Qwen3.6-35B-A3B and Qwen3.8-Flash-Next."
    )
    ap.add_argument("--base-repo", default=DEFAULT_BASE_REPO, help="Base HF repo ID")
    ap.add_argument("--donor-repo", default=DEFAULT_DONOR_REPO, help="Donor HF repo ID")
    ap.add_argument("--revision", default="main", help="HF git revision")
    ap.add_argument("--base-dir", type=Path, default=None, help="Optional local dir for base tokenizer files")
    ap.add_argument("--donor-dir", type=Path, default=None, help="Optional local dir for donor files")
    ap.add_argument(
        "--check-ple-headers",
        action="store_true",
        help="Inspect donor `.ple.` safetensors headers via HTTP Range (or local files)",
    )
    ap.add_argument("--ple-pattern", default=".ple.", help="Tensor-name substring filter for PLE headers")
    ap.add_argument(
        "--format",
        choices=("markdown", "json", "both"),
        default="markdown",
        help="Output format",
    )
    ap.add_argument("--output", type=Path, default=None, help="Optional file path to write output")
    ap.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 if shared core tokenizer parity or tokenizer_config equality fails",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    token = os.environ.get("HF_TOKEN") or None

    base_tok = load_json_from_source(
        args.base_repo,
        "tokenizer.json",
        local_dir=args.base_dir,
        revision=args.revision,
        timeout=args.timeout,
        token=token,
    )
    donor_tok = load_json_from_source(
        args.donor_repo,
        "tokenizer.json",
        local_dir=args.donor_dir,
        revision=args.revision,
        timeout=args.timeout,
        token=token,
    )
    base_cfg = load_json_from_source(
        args.base_repo,
        "tokenizer_config.json",
        local_dir=args.base_dir,
        revision=args.revision,
        timeout=args.timeout,
        token=token,
    )
    donor_cfg = load_json_from_source(
        args.donor_repo,
        "tokenizer_config.json",
        local_dir=args.donor_dir,
        revision=args.revision,
        timeout=args.timeout,
        token=token,
    )

    ple_summary: dict[str, Any] | None = None
    if args.check_ple_headers:
        ple_summary = inspect_ple_headers_from_source(
            args.donor_repo,
            local_dir=args.donor_dir,
            revision=args.revision,
            pattern=args.ple_pattern,
            timeout=args.timeout,
            token=token,
        )

    report = build_parity_report(
        base_repo=args.base_repo,
        donor_repo=args.donor_repo,
        base_tok=base_tok,
        donor_tok=donor_tok,
        base_cfg=base_cfg,
        donor_cfg=donor_cfg,
        ple_summary=ple_summary,
    )

    if args.format == "json":
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    elif args.format == "markdown":
        rendered = render_markdown_report(report)
    else:
        rendered = (
            render_markdown_report(report)
            + "\n```json\n"
            + json.dumps(report, indent=2, sort_keys=True)
            + "\n```\n"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)

    if args.strict and not report["overall_shared_parity_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
