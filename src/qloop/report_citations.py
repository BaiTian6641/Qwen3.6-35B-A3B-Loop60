"""Verified IEEE citation registry, first-appearance resolver, and integrity validator.

All bibliographic entries are strictly web-verified or taken verbatim from
`Loop64-Fable-SIRSI-v1/README.md`. Numbers `[1]..[N]` are assigned strictly in
order of first appearance so that EN and ZH academic reports maintain identical,
verifiable IEEE reference ordering.
"""
from __future__ import annotations

import re

# Verified IEEE bibliography (exact keys -> IEEE reference strings)
IEEE_REFERENCES: dict[str, str] = {
    "qwen36": (
        'Qwen Team, "Qwen3.6-35B-A3B," Alibaba Cloud, Model Release, Apr. 16, 2026. '
        "[Online]. Available: https://huggingface.co/Qwen/Qwen3.6-35B-A3B"
    ),
    "gdn": (
        'S. Yang, J. Kautz, and A. Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with '
        'Delta Rule," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2412.06464.'
    ),
    "gdn2": (
        'A. Hatamizadeh, Y. Choi, and J. Kautz, "Gated DeltaNet-2: Decoupling Erase and Write '
        'in Linear Attention," *arXiv preprint arXiv:2605.22791*, May 2026.'
    ),
    "qed": (
        'D. Gupta, A. Das, and D. Gupta, "The Query Knows What to Forget: A Second Erase '
        'Direction for Linear Attention," *arXiv preprint arXiv:2608.13668*, Aug. 2026.'
    ),
    "neg_eig": (
        "R. Grazzi, J. Siems, A. Zela, J. K. H. Franke, F. Hutter, and M. Pontil, "
        '"Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues," '
        "*arXiv preprint arXiv:2411.12537*, Nov. 2024."
    ),
    "apollo": (
        "H. Zhu, Z. Zhang, W. Cong, X. Liu, S. Park, V. Chandra, B. Long, D. Z. Pan, "
        'Z. Wang, and J. Lee, "APOLLO: SGD-like Memory, AdamW-level Performance," '
        "*arXiv preprint arXiv:2412.05270*, Dec. 2024."
    ),
    "mhc": (
        'Z. Xie, Y. Wei, H. Cao, W. Liang, *et al.* (DeepSeek-AI), "mHC: Manifold-Constrained '
        'Hyper-Connections," *arXiv preprint arXiv:2512.24880*, Dec. 2025.'
    ),
    "lsa": (
        "A. Kubicek, A. Caciolai, L. Peng, S. Sumant, B. Hoover, F. Greenlee, "
        "R. M. Battleday, R. Urban, T. Hoefler, and J. C. R. Whittington, "
        '"Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention," '
        "*arXiv preprint arXiv:2610.42147*, Oct. 2026."
    ),
    "yarn": (
        'B. Peng, J. Quesnelle, H. Fan, and E. Shippole, "YaRN: Efficient Context Window '
        'Extension of Large Language Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, '
        "2024, arXiv:2309.00071."
    ),
    "mor": (
        "S. Bae, Y. Kim, R. Bayat, S. Kim, J. Ha, T. Schuster, A. Fisch, H. Harutyunyan, "
        "Z. Ji, A. Courville, and S.-Y. Yun, "
        '"Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level '
        'Computation," *arXiv preprint arXiv:2507.10524*, Jul. 2025.'
    ),
    "inkling2": (
        'Thinking Machines Lab, "Inkling 2 Mini," Technical Release, Nov. 20, 2026. '
        "[Online]. Available: https://thinkingmachines.ai"
    ),
    "rwkv7": (
        "B. Peng, R. Zhang, D. Goldstein, E. Alcaide, *et al.*, "
        "\"RWKV-7 'Goose' with Expressive Dynamic State Evolution,\" "
        "*arXiv preprint arXiv:2503.14456*, Mar. 2025."
    ),
    "fla": (
        'S. Yang and Y. Zhang, "FLA: A Triton-Based Library for Hardware-Efficient '
        'Implementations of Linear Attention Mechanisms," GitHub Repository '
        "(`fla-org/flash-linear-attention`, v0.6.0), 2024–2026."
    ),
    "cce": (
        'E. Wijmans *et al.*, "Cut Your Losses in Large-Vocabulary Language Models," '
        "in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2411.09009."
    ),
    "deepseekmath": (
        "Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, "
        '"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," '
        "*arXiv preprint arXiv:2402.03300*, Feb. 2024."
    ),
    "dapo": (
        'Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, *et al.*, "DAPO: An Open-Source LLM Reinforcement '
        'Learning System at Scale," *arXiv preprint arXiv:2503.14476*, Mar. 2025.'
    ),
    "turboquant": (
        'A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni, "TurboQuant: Online Vector '
        'Quantization with Near-optimal Distortion Rate," in *Proc. Int. Conf. Learn. Represent. '
        "(ICLR)*, 2026."
    ),
    "openthoughts3": (
        "E. Guha, R. Marten, S. Keh, N. Raoof, G. Smyrnis, H. Bansal, M. Nezhurina, J. Mercat, "
        "T. Vu, Z. Sprague, A. G. Dimakis, and L. Schmidt, "
        '"OpenThoughts: Data Recipes for Reasoning Models," '
        "*arXiv preprint arXiv:2506.04178*, Jun. 2025."
    ),
    "nemotron": (
        "A. Bercovich, I. Levy, I. Golan, M. Dabbah, R. El-Yaniv, B. Ginsburg, J. Cohen, and "
        'B. Catanzaro, "Llama-Nemotron: Efficient Reasoning Models," '
        "*arXiv preprint arXiv:2505.00949*, May 2025."
    ),
    "arcagi2": (
        'F. Chollet, M. Knoop, G. Kamradt, and B. Landers, "ARC-AGI-2: A New Challenge for '
        'Frontier AI Reasoning Systems," GitHub Repository (`arcprize/ARC-AGI-2`), 2025.'
    ),
    "fineweb": (
        "G. Penedo, H. Kydlíček, L. Ben Allal, A. Lozhkov, M. Mitchell, C. Raffel, "
        "L. Von Werra, and T. Wolf, "
        '"The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale," '
        "*arXiv preprint arXiv:2406.17557*, Jun. 2024."
    ),
    "loop64_model": (
        'TassP0lar, "Qwen3.6-35B-A3B-Loop64: Open Looped Hybrid Checkpoints and Empirical '
        'Telemetry," Hugging Face Model Hub, Nov. 2026. [Online]. Available: '
        "https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64"
    ),
    "loop64_dataset": (
        'TassP0lar, "Loop64-Fable-SIRSI-v1: Curated Anti-Slop 3-Beat Looped-CoT, Aesthetic '
        'Craft, and SI-RSI Recovery Dataset," Hugging Face Datasets Hub (v1.2), Nov. 2026. '
        "[Online]. Available: https://huggingface.co/datasets/TassP0lar/Loop64-Fable-SIRSI-v1"
    ),
}

_CITE_TAG_RE = re.compile(r"\{\{cite:([a-z0-9_, ]+)\}\}")
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_BODY_CITE_RE = re.compile(r"(?<!\\)\[(\d+)\](?!\()")
_REF_LINE_RE = re.compile(r"^\s*(?:-\s*)?\[(\d+)\]\s+(.+)$")


def resolve_ieee_citations(
    md_text: str,
    ref_heading: str = "## References",
    expected_order: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Replace `{{cite:key1,key2}}` with `[k1], [k2]` in strict first-appearance order.

    Appends the formatted IEEE References section under `ref_heading` and returns
    `(resolved_markdown, ordered_keys)`. If `expected_order` is supplied, validates
    that the text's natural first-appearance order matches `expected_order` identically.
    """
    seen_order: list[str] = []
    key_to_num: dict[str, int] = {}

    def _repl(match: re.Match[str]) -> str:
        raw_keys = [k.strip() for k in match.group(1).split(",") if k.strip()]
        nums: list[int] = []
        for k in raw_keys:
            if k not in IEEE_REFERENCES:
                raise KeyError(f"Unknown IEEE citation key: {k!r}")
            if k not in key_to_num:
                seen_order.append(k)
                key_to_num[k] = len(seen_order)
            nums.append(key_to_num[k])
        return ", ".join(f"[{n}]" for n in nums)

    body = _CITE_TAG_RE.sub(_repl, md_text)
    if expected_order is not None and seen_order != expected_order:
        raise ValueError(
            "IEEE citation first-appearance order mismatch!\n"
            f"  expected ({len(expected_order)}): {expected_order}\n"
            f"  observed ({len(seen_order)}): {seen_order}"
        )

    ref_lines = [
        "",
        "---",
        "",
        ref_heading,
        "",
    ]
    for idx, key in enumerate(seen_order, start=1):
        ref_lines.append(f"[{idx}] {IEEE_REFERENCES[key]}")
        ref_lines.append("")
    resolved = body.rstrip() + "\n" + "\n".join(ref_lines)
    return resolved, seen_order


def validate_ieee_citations(md_text: str) -> dict[str, object]:
    """Validate rendered Markdown for strict IEEE citation integrity.

    Checks:
      1. References / 参考文献 section exists with consecutive `[1]..[N]` entries.
      2. Every `[k]` cited in body prose (outside fenced/inline code) exists in References.
      3. No unused reference entries exist.
      4. First appearances in the body follow strict ascending order `1, 2, ..., N`.
    """
    split_match = re.search(r"(?m)^##\s+(References|参考文献)\s*$", md_text)
    if not split_match:
        raise ValueError("Missing '## References' or '## 参考文献' section")

    body_raw = md_text[: split_match.start()]
    refs_raw = md_text[split_match.end() :]

    ref_entries: dict[int, str] = {}
    ref_nums_ordered: list[int] = []
    for ln in refs_raw.splitlines():
        m = _REF_LINE_RE.match(ln)
        if m:
            num = int(m.group(1))
            if num in ref_entries:
                raise ValueError(f"Duplicate reference number [{num}]")
            ref_entries[num] = m.group(2).strip()
            ref_nums_ordered.append(num)

    expected_ref_nums = list(range(1, len(ref_nums_ordered) + 1))
    if ref_nums_ordered != expected_ref_nums:
        raise ValueError(
            f"Reference numbers not consecutive 1..N: got {ref_nums_ordered}"
        )

    # Strip fenced and inline code before scanning body citations
    body_clean = _FENCED_CODE_RE.sub("", body_raw)
    body_clean = _INLINE_CODE_RE.sub("", body_clean)

    first_order: list[int] = []
    seen_set: set[int] = set()
    all_citations: list[int] = []
    for m in _BODY_CITE_RE.finditer(body_clean):
        num = int(m.group(1))
        all_citations.append(num)
        if num not in seen_set:
            seen_set.add(num)
            first_order.append(num)

    missing_in_refs = sorted(seen_set - set(ref_entries))
    if missing_in_refs:
        raise ValueError(f"Body cites undefined references: {missing_in_refs}")

    unused_refs = sorted(set(ref_entries) - seen_set)
    if unused_refs:
        raise ValueError(f"Unused reference entries: {unused_refs}")

    if first_order != expected_ref_nums:
        raise ValueError(
            f"Citations not in strict first-appearance order: expected {expected_ref_nums}, got {first_order}"
        )

    return {
        "num_references": len(ref_entries),
        "total_body_citations": len(all_citations),
        "first_appearance_order": first_order,
        "references": ref_entries,
    }
