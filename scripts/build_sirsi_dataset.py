#!/usr/bin/env python3
"""CLI for Loop64-Fable-SIRSI Track B streaming dataset audit, local Parquet build, synthesis & Hub push.

Examples:
    # 1. Live streaming audit + local Parquet dataset build (zero full-dataset downloads)
    python scripts/build_sirsi_dataset.py --build-local --max-per-source 1000 \\
        --out-dir /Users/tianrui6641/Qwen3.6-35B-A3B-Loop60/runs/datasets/Loop64-Fable-SIRSI-v1

    # 2. Audit only (no Parquet export)
    python scripts/build_sirsi_dataset.py --audit-only --max-per-source 500

    # 3. Track B async active synthesis & rewrite (with safe probe)
    python scripts/build_sirsi_dataset.py --synthesize --teacher-provider hf_router \\
        --max-synth 10 --out-dir runs/datasets/Loop64-Fable-SIRSI-v1

    # 4. Validate local upload manifest & Dataset Card (dry-run; does NOT push)
    python scripts/build_sirsi_dataset.py --push-to-hub \\
        --repo-id TassP0lar/Loop64-Fable-SIRSI-v1 --public --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure `src/` is importable when run directly from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from qloop.data.sirsi_dataset import (  # noqa: E402
    ALL_AUDITED_SOURCES,
    PROVIDER_DEFAULTS,
    AsyncTeacherClient,
    TrackBSynthesizer,
    build_or_load_decontaminator,
    publish_to_hub,
    render_dataset_card,
    run_streaming_pipeline,
    write_parquet_records,
)

DEFAULT_OUT_DIR = "/Users/tianrui6641/Qwen3.6-35B-A3B-Loop60/runs/datasets/Loop64-Fable-SIRSI-v1"
DEFAULT_DECONTAM_CACHE = "/Users/tianrui6641/Qwen3.6-35B-A3B-Loop60/runs/datasets/.decontam_cache"
DEFAULT_GOLDEN_SEEDS = str(_REPO_ROOT / "configs" / "data" / "sirsi_golden_seeds.json")


def _print_triage_table(summary: dict[str, Any]) -> None:
    print("\n=== Loop64-Fable-SIRSI Streaming Audit & Triage Summary ===")
    header = (
        f"{'Source':<18} {'Lic':<10} {'Stream':>6} {'KEEP':>6} "
        f"{'DetSal':>6} {'SALV':>6} {'DROP':>6} {'Slop':>6} {'3Beat':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, st in (summary.get("per_source") or {}).items():
        if st.get("excluded"):
            lic = st.get("license", "")
            reason = st.get("exclusion_reason", "")
            print(
                f"{name:<18} {lic!s:<10} {'EXCL':>6} {'-':>6} {'-':>6} "
                f"{'-':>6} {'-':>6} {'-':>6} {'-':>6}  {reason}"
            )
            continue
        lic = st.get("license", "")
        print(
            f"{name:<18} {lic!s:<10} "
            f"{int(st.get('streamed', 0)):>6d} {int(st.get('keep', 0)):>6d} "
            f"{int(st.get('deterministic_salvaged_to_keep', 0)):>6d} {int(st.get('salvage', 0)):>6d} "
            f"{int(st.get('drop', 0)):>6d} {float(st.get('mean_slop', 0)):>6.3f} {float(st.get('mean_3beat', 0)):>6.3f}"
        )
    oc = summary.get("output_counts") or {}
    print(
        f"\nOutputs: sft_core={oc.get('sft_core', 0):,} | salvage_queue={oc.get('salvage_queue', 0):,} | "
        f"sirsi_preferences={oc.get('sirsi_preferences', 0):,} | elapsed={summary.get('elapsed_sec', 0)}s"
    )
    dc = summary.get("decontam") or {}
    print(f"Decontam: index_size={dc.get('index_size', 0):,} | hits={dc.get('hits', {})}")
    dd = summary.get("dedup") or {}
    print(f"Dedup: unique={dd.get('unique_indexed', 0):,} | dup_counts={dd.get('dup_counts', {})}")


async def _run_synthesis_step(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    client = AsyncTeacherClient(
        provider=args.teacher_provider,
        model=args.teacher_model,
        base_url=args.teacher_base_url,
    )
    probe = await client.probe()
    print(f"\n=== Teacher Probe ({probe['provider']} / {probe['model']}) ===")
    print(json.dumps(probe, indent=2))

    report: dict[str, Any] = {"probe": probe, "modes": {}, "accepted_sft": 0, "accepted_pref": 0}
    if not probe["ok"]:
        print(
            f"Teacher provider '{probe['provider']}' returned status={probe['status_code']} ({probe['detail']}). "
            "Skipping active Track B generation cleanly without failing local build."
        )
        return report

    decontam = None
    if not args.skip_decontam:
        decontam, _ = build_or_load_decontaminator(cache_dir=args.decontam_cache, n=13, on_missing="warn")

    synth = TrackBSynthesizer(client, out_dir=out_dir, decontaminator=decontam)
    modes = [m.strip() for m in args.synth_modes.split(",") if m.strip()]
    new_sft: list[dict[str, Any]] = []
    new_pref: list[dict[str, Any]] = []

    if "salvage_rewrite" in modes:
        salvage_pq = out_dir / "data" / "salvage_queue.parquet"
        salvage_rows: list[dict[str, Any]] = []
        if salvage_pq.exists():
            import pyarrow.parquet as pq

            salvage_rows = pq.read_table(salvage_pq).to_pylist()
        s_sft, s_pref = await synth.run_salvage_rewrite(salvage_rows, max_items=args.max_synth)
        new_sft.extend(s_sft)
        new_pref.extend(s_pref)
        report["modes"]["salvage_rewrite"] = {"accepted_sft": len(s_sft), "accepted_pref": len(s_pref)}

    for mode in ("aesthetic_craft", "sirsi_recovery"):
        if mode in modes:
            m_sft, m_pref = await synth.run_structured_mode(mode, max_items=args.max_synth)
            new_sft.extend(m_sft)
            new_pref.extend(m_pref)
            report["modes"][mode] = {"accepted_sft": len(m_sft), "accepted_pref": len(m_pref)}

    report["stats"] = dict(synth.stats)
    report["accepted_sft"] = len(new_sft)
    report["accepted_pref"] = len(new_pref)

    # Merge into existing local Parquet shards if present
    if new_sft or new_pref:
        import pyarrow.parquet as pq

        sft_pq = out_dir / "data" / "sft_core.parquet"
        pref_pq = out_dir / "data" / "sirsi_preferences.parquet"
        cur_sft = pq.read_table(sft_pq).to_pylist() if sft_pq.exists() else []
        cur_pref = pq.read_table(pref_pq).to_pylist() if pref_pq.exists() else []
        seen_sft = {r["id"] for r in cur_sft}
        seen_pref = {r["id"] for r in cur_pref}
        merged_sft = cur_sft + [r for r in new_sft if r["id"] not in seen_sft]
        merged_pref = cur_pref + [r for r in new_pref if r["id"] not in seen_pref]
        write_parquet_records(merged_sft, sft_pq, kind="sft")
        write_parquet_records(merged_pref, pref_pq, kind="pref")

        summary_path = out_dir / "audit_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["output_counts"]["sft_core"] = len(merged_sft)
            summary["output_counts"]["sirsi_preferences"] = len(merged_pref)
            summary["output_counts"]["track_b_sft"] = len(new_sft)
            summary["output_counts"]["track_b_pref"] = len(new_pref)
            summary["track_b_synthesis"] = report
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (out_dir / "README.md").write_text(render_dataset_card(summary, repo_id=args.repo_id), encoding="utf-8")

    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Loop64-Fable-SIRSI Track B Streaming Dataset Builder")
    ap.add_argument("--audit-only", action="store_true", help="Stream & audit without writing Parquet shards")
    ap.add_argument("--build-local", action="store_true", help="Run full streaming audit + build local Parquet shards")
    ap.add_argument("--synthesize", action="store_true", help="Run Track B async active synthesis/rewrite")
    ap.add_argument("--probe-teacher", action="store_true", help="Run only a tiny max_tokens=16 teacher probe")
    ap.add_argument("--teacher-provider", default="hf_router", choices=sorted(PROVIDER_DEFAULTS))
    ap.add_argument("--teacher-model", default=None)
    ap.add_argument("--teacher-base-url", default=None)
    ap.add_argument("--synth-modes", default="salvage_rewrite,aesthetic_craft,sirsi_recovery")
    ap.add_argument("--max-synth", type=int, default=10)
    ap.add_argument(
        "--sources",
        default=None,
        help=f"Comma-separated source keys (default: all {len(ALL_AUDITED_SOURCES)} audited sources)",
    )
    ap.add_argument("--max-per-source", type=int, default=1000)
    ap.add_argument("--golden-seeds", default=DEFAULT_GOLDEN_SEEDS)
    ap.add_argument("--decontam-cache", default=DEFAULT_DECONTAM_CACHE)
    ap.add_argument("--skip-decontam", action="store_true")
    ap.add_argument("--no-live-license-check", action="store_true")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--repo-id", default="TassP0lar/Loop64-Fable-SIRSI-v1")
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--public", action="store_true", default=True)
    vis.add_argument("--private", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", help="Validate manifest & README without uploading")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    if not any([args.audit_only, args.build_local, args.synthesize, args.probe_teacher, args.push_to_hub]):
        build_parser().print_help()
        return 2

    out_dir = Path(args.out_dir)
    source_names = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    token = os.environ.get("HF_TOKEN")

    if args.audit_only or args.build_local:
        summary = run_streaming_pipeline(
            out_dir=out_dir,
            source_names=source_names,
            max_per_source=args.max_per_source,
            golden_seeds_path=args.golden_seeds,
            decontam_cache_dir=args.decontam_cache,
            skip_decontam=args.skip_decontam,
            audit_only=bool(args.audit_only and not args.build_local),
            live_license_check=not args.no_live_license_check,
            token=token,
            repo_id=args.repo_id,
        )
        _print_triage_table(summary)

    if args.probe_teacher and not args.synthesize:
        client = AsyncTeacherClient(
            provider=args.teacher_provider,
            model=args.teacher_model,
            base_url=args.teacher_base_url,
        )
        probe = asyncio.run(client.probe())
        print("\n=== Teacher Probe ===")
        print(json.dumps(probe, indent=2))
        summary_path = out_dir / "audit_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["teacher_probe"] = probe
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.synthesize:
        synth_report = asyncio.run(_run_synthesis_step(args, out_dir))
        summary_path = out_dir / "audit_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["track_b_synthesis"] = synth_report
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.push_to_hub:
        is_public = not args.private
        manifest = publish_to_hub(
            out_dir=out_dir,
            repo_id=args.repo_id,
            public=is_public,
            dry_run=bool(args.dry_run),
            token=token,
        )
        print("\n=== HF Hub Upload Manifest ===")
        print(json.dumps(manifest, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
