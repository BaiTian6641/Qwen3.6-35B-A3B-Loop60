"""Offline unit tests for `scripts/verify_q36_q38_tokenizer_parity.py` (zero network calls)."""
from __future__ import annotations

import importlib.util
import json
import struct
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "verify_q36_q38_tokenizer_parity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_q36_q38_tokenizer_parity", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()

AUDIO_TTS_APPENDED = [
    (248070, "<|audio_start|>"),
    (248071, "<|audio_end|>"),
    (248072, "<tts_pad>"),
    (248073, "<tts_text_bos>"),
    (248074, "<tts_text_eod>"),
    (248075, "<tts_text_bos_single>"),
    (248076, "<|audio_pad|>"),
]


def _make_added_token(tid: int, content: str) -> dict:
    return {
        "id": tid,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }


def _make_synthetic_tokenizer(*, with_audio_tts: bool = False) -> dict:
    added = [_make_added_token(248044 + i, f"<|spec_{i}|>") for i in range(26)]
    if with_audio_tts:
        added.extend(_make_added_token(tid, content) for tid, content in AUDIO_TTS_APPENDED)
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {"!": 0, "a": 1, "b": 2, "ab": 3},
            "merges": ["a b"],
        },
    }


def _make_synthetic_config() -> dict:
    return {
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "bos_token": None,
        "model_max_length": 262144,
        "clean_up_tokenization_spaces": False,
    }


def _write_safetensors_stub(path: Path, header_dict: dict) -> None:
    hdr_bytes = json.dumps(header_dict).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + b"\x00" * 16)


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("Network calls are forbidden in offline unit tests")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)


def test_exact_shared_parity_with_seven_appended_audio_tokens():
    base_tok = _make_synthetic_tokenizer(with_audio_tts=False)
    donor_tok = _make_synthetic_tokenizer(with_audio_tts=True)
    base_cfg = _make_synthetic_config()
    donor_cfg = _make_synthetic_config()

    report = MOD.build_parity_report(
        base_repo="Qwen/Qwen3.6-35B-A3B",
        donor_repo="Qwen/Qwen3.8-Flash-Next",
        base_tok=base_tok,
        donor_tok=donor_tok,
        base_cfg=base_cfg,
        donor_cfg=donor_cfg,
    )

    assert report["overall_shared_parity_ok"] is True
    tok = report["tokenizer_json"]
    assert tok["sections_checked"] == 18
    assert tok["sections_shared_exact_equal"] == 18
    assert tok["shared_core_parity_ok"] is True

    added = tok["sections"]["added_tokens"]
    assert added["exact_equal"] is False
    assert added["shared_exact_equal"] is True
    assert added["strictly_appended"] is True
    assert added["shared_count"] == 26
    assert added["shared_id_range"] == [248044, 248069]
    assert added["appended_count"] == 7
    assert [t["id"] for t in added["appended_tokens"]] == [tid for tid, _ in AUDIO_TTS_APPENDED]
    assert [t["content"] for t in added["appended_tokens"]] == [c for _, c in AUDIO_TTS_APPENDED]

    cfg = report["tokenizer_config_json"]
    assert cfg["exact_equal"] is True
    assert cfg["key_fields"]["eos_token"]["base"] == "<|im_end|>"
    assert cfg["key_fields"]["pad_token"]["base"] == "<|endoftext|>"
    assert cfg["key_fields"]["bos_token"]["base"] is None
    assert cfg["key_fields"]["model_max_length"]["base"] == 262144


def test_detects_breaking_diffs_in_vocab_merges_added_and_config():
    base_tok = _make_synthetic_tokenizer(with_audio_tts=False)
    donor_tok = _make_synthetic_tokenizer(with_audio_tts=True)
    donor_tok["model"]["vocab"]["ab"] = 99
    donor_tok["model"]["merges"] = ["b a"]
    donor_tok["normalizer"] = {"type": "NFKC"}
    donor_tok["added_tokens"][0]["content"] = "<|mutated|>"

    tok_diff = MOD.diff_tokenizer_json(base_tok, donor_tok)
    assert tok_diff["shared_core_parity_ok"] is False
    assert tok_diff["sections"]["model.vocab"]["exact_equal"] is False
    assert tok_diff["sections"]["model.vocab"]["id_mismatch_count"] == 1
    assert tok_diff["sections"]["model.merges"]["exact_equal"] is False
    assert tok_diff["sections"]["model.merges"]["first_diff_index"] == 0
    assert tok_diff["sections"]["normalizer"]["exact_equal"] is False
    assert tok_diff["sections"]["added_tokens"]["shared_exact_equal"] is False
    assert tok_diff["sections"]["added_tokens"]["modified_shared_count"] == 1

    base_cfg = _make_synthetic_config()
    donor_cfg = _make_synthetic_config()
    donor_cfg["model_max_length"] = 131072
    cfg_diff = MOD.diff_tokenizer_config(base_cfg, donor_cfg)
    assert cfg_diff["exact_equal"] is False
    assert cfg_diff["diff_keys"] == ["model_max_length"]
    assert cfg_diff["key_fields"]["model_max_length"]["exact_equal"] is False


def test_non_appended_lower_id_added_token_fails_strict_append():
    base_tok = _make_synthetic_tokenizer(with_audio_tts=False)
    donor_tok = _make_synthetic_tokenizer(with_audio_tts=False)
    donor_tok["added_tokens"].append(_make_added_token(100, "<|injected_low_id|>"))

    tok_diff = MOD.diff_tokenizer_json(base_tok, donor_tok)
    assert tok_diff["sections"]["added_tokens"]["shared_exact_equal"] is True
    assert tok_diff["sections"]["added_tokens"]["strictly_appended"] is False
    assert tok_diff["shared_core_parity_ok"] is False


def test_offline_cli_json_markdown_and_ple_header_audit(tmp_path: Path, capsys):
    base_dir = tmp_path / "base"
    donor_dir = tmp_path / "donor"
    base_dir.mkdir()
    donor_dir.mkdir()

    (base_dir / "tokenizer.json").write_text(
        json.dumps(_make_synthetic_tokenizer(with_audio_tts=False)),
        encoding="utf-8",
    )
    (donor_dir / "tokenizer.json").write_text(
        json.dumps(_make_synthetic_tokenizer(with_audio_tts=True)),
        encoding="utf-8",
    )
    (base_dir / "tokenizer_config.json").write_text(
        json.dumps(_make_synthetic_config()),
        encoding="utf-8",
    )
    (donor_dir / "tokenizer_config.json").write_text(
        json.dumps(_make_synthetic_config()),
        encoding="utf-8",
    )

    # Synthetic safetensors shards mirroring Qwen3.8-Flash-Next's 0-indexed Layer 1 vs config ple_layer_ids=[2]
    shard_a = "model-00005-of-00131.safetensors"
    shard_b = "model-00037-of-00131.safetensors"
    _write_safetensors_stub(
        donor_dir / shard_a,
        {
            "__metadata__": {"format": "pt"},
            "model.language_model.layers.1.ple.table.0.weight": {
                "dtype": "BF16",
                "shape": [100, 160],
                "data_offsets": [0, 32000],
            },
            "model.language_model.layers.1.ple.table.1.weight": {
                "dtype": "BF16",
                "shape": [100, 160],
                "data_offsets": [32000, 64000],
            },
            "model.language_model.layers.0.mlp.gate.weight": {
                "dtype": "BF16",
                "shape": [8, 16],
                "data_offsets": [64000, 64256],
            },
        },
    )
    _write_safetensors_stub(
        donor_dir / shard_b,
        {
            "model.language_model.layers.1.ple.layer_multipliers": {
                "dtype": "I64",
                "shape": [3],
                "data_offsets": [0, 24],
            },
            "model.language_model.layers.1.ple.value_proj.weight": {
                "dtype": "BF16",
                "shape": [2560, 2560],
                "data_offsets": [24, 13107224],
            },
        },
    )
    (donor_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.layers.1.ple.table.0.weight": shard_a,
                    "model.language_model.layers.1.ple.table.1.weight": shard_a,
                    "model.language_model.layers.0.mlp.gate.weight": shard_a,
                    "model.language_model.layers.1.ple.layer_multipliers": shard_b,
                    "model.language_model.layers.1.ple.value_proj.weight": shard_b,
                }
            }
        ),
        encoding="utf-8",
    )
    (donor_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "ple_layer_ids": [2]}),
        encoding="utf-8",
    )

    out_json = tmp_path / "out" / "parity.json"
    rc = MOD.main(
        [
            "--base-dir",
            str(base_dir),
            "--donor-dir",
            str(donor_dir),
            "--check-ple-headers",
            "--format",
            "json",
            "--output",
            str(out_json),
            "--strict",
        ]
    )
    assert rc == 0
    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    assert parsed["overall_shared_parity_ok"] is True
    ple = parsed["ple_headers"]
    assert ple["matched_tensor_count"] == 4
    assert ple["header_layer_ids"] == [1]
    assert ple["config_ple_layer_ids"] == [2]
    assert ple["config_vs_header_layer_id_mismatch"] is True
    assert ple["total_params"] == 2 * (100 * 160) + 3 + (2560 * 2560)

    # Markdown output to stdout
    rc_md = MOD.main(
        [
            "--base-dir",
            str(base_dir),
            "--donor-dir",
            str(donor_dir),
            "--check-ple-headers",
            "--format",
            "markdown",
            "--strict",
        ]
    )
    assert rc_md == 0
    stdout = capsys.readouterr().out
    assert "# Tokenizer Parity Audit:" in stdout
    assert "`18/18`" in stdout
    assert "<|audio_start|>" in stdout
    assert "Layer-ID Mismatch:** `True`" in stdout

    # Mutate donor config -> --strict must return exit code 1
    bad_cfg = _make_synthetic_config()
    bad_cfg["eos_token"] = "</s>"
    (donor_dir / "tokenizer_config.json").write_text(json.dumps(bad_cfg), encoding="utf-8")
    rc_fail = MOD.main(
        [
            "--base-dir",
            str(base_dir),
            "--donor-dir",
            str(donor_dir),
            "--format",
            "both",
            "--strict",
        ]
    )
    assert rc_fail == 1
    both_out = capsys.readouterr().out
    assert "```json" in both_out
