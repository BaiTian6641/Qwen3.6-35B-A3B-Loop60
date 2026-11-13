"""Dataset registry: every dataset the training stages may touch, with role, adapter and licence.

`verified=True` means the HF id/config/split and the column layout were checked against the Hub
(datasets-server first-rows) when this file was written. Anything else is `verified=False` and is
re-checked at runtime by `verify_specs()` (`qloop doctor --datasets`).

hf_id prefixes:
  * `builtin:<name>`   offline synthetic source (tests / smoke runs), see `mixture.py`
  * `github:<org>/<repo>` non-HF source (ARC-AGI-2), fetched as a zip archive and cached
  * otherwise a Hugging Face dataset repo id
`split` may chain several splits with '+', e.g. "level_1+level_2".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

from qloop.debug import require

log = logging.getLogger("qloop")

ROLES = ("sft", "pretrain", "rl", "eval")
FORMATS = (
    "auto", "messages", "sharegpt", "prompt_response", "text", "chatml_text", "nemotron", "arc_agi",
    "rl_kernelbench", "rl_swe", "rl_ot_agent", "eval_text",
)


@dataclass
class DatasetSpec:
    name: str
    hf_id: str
    config: str | None = None
    split: str = "train"
    role: str = "sft"
    format: str = "auto"
    gated: bool = False
    license: str = ""
    stages: list[str] = field(default_factory=list)
    notes: str = ""
    contamination_group: str | None = None
    verified: bool = False
    # --- adapter / validation knobs
    require_reasoning: bool = False      # drop rows whose reasoning is missing/empty/encrypted
    allow_empty_content: bool = False    # reasoning-only rows (no final answer) are OK
    fields: dict[str, str] = field(default_factory=dict)   # column overrides, e.g. {"prompt": "query"}
    text_fields: list[str] = field(default_factory=list)   # eval: columns holding benchmark text
    drop_system_prompts: list[str] = field(default_factory=list)  # exact system prompts to strip
    data_files: Any = None               # passed to load_dataset (local / non-standard layouts)
    min_chars: int = 1                   # pretrain: drop shorter docs

    def __post_init__(self) -> None:
        require(self.role in ROLES, f"DatasetSpec[{self.name}]: role must be one of {ROLES}", role=self.role)
        require(self.format in FORMATS, f"DatasetSpec[{self.name}]: unknown format (adapter)", format=self.format,
                known=FORMATS)

    @property
    def kind(self) -> str:
        if self.hf_id.startswith("builtin:"):
            return "builtin"
        if self.hf_id.startswith("github:"):
            return "github"
        return "hf"

    @property
    def splits(self) -> list[str]:
        return [s.strip() for s in self.split.split("+") if s.strip()]

    def with_overrides(self, **kw: Any) -> DatasetSpec:
        known = set(self.__dataclass_fields__)
        bad = sorted(set(kw) - known)
        require(not bad, f"DatasetSpec[{self.name}]: unknown override keys", bad=bad, known=sorted(known))
        return replace(self, **kw)


_THINK_SYS = ["detailed thinking on", "detailed thinking off"]

_SPECS: list[DatasetSpec] = [
    # ------------------------------------------------------------------ SFT: reasoning distills
    DatasetSpec(
        "fable51", "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x", role="sft", format="messages",
        license="apache-2.0", stages=["sft", "reasoning", "agentic"], require_reasoning=True, verified=True,
        notes="messages(role/content/tool_calls/tool) + `tools` column; assistant reasoning is wrapped in "
              "<thought>..</thought> inside content (extracted). Also has a 'full' split (unfiltered).",
    ),
    DatasetSpec(
        "fable5_distill", "HelioAI/Fable-5-Distill-Reasoning-462x", role="sft", format="prompt_response",
        license="unknown", stages=["sft", "reasoning"], require_reasoning=True, allow_empty_content=True,
        fields={"prompt": "query", "reasoning": "thinking"}, verified=True,
        notes="columns query/thinking only (no separate final answer) -> reasoning-only targets; en+ru.",
    ),
    DatasetSpec(
        "agentic_fable5", "lordx64/agentic-distill-fable-5-sft", role="sft", format="chatml_text",
        license="agpl-3.0", stages=["sft", "agentic"], require_reasoning=True, verified=True,
        notes="single `text` column pre-rendered in Qwen ChatML; many rows have empty <think></think> -> "
              "require_reasoning filter. AGPL-3.0: check licence compatibility before releasing weights.",
    ),
    DatasetSpec(
        "opus46_3000x", "nohurry/Opus-4.6-Reasoning-3000x-filtered", role="sft", format="prompt_response",
        license="apache-2.0", stages=["sft", "reasoning"], require_reasoning=True, verified=True,
        fields={"prompt": "problem", "reasoning": "thinking", "response": "solution"},
        notes="problem/thinking/solution. Card notice: upstream re-filtered; overlaps opus46_3300x (dedupe on).",
    ),
    DatasetSpec(
        "opus46_3300x", "Crownelius/Opus-4.6-Reasoning-3300x", role="sft", format="prompt_response",
        license="apache-2.0", stages=["sft", "reasoning"], require_reasoning=True, verified=True,
        fields={"prompt": "problem", "reasoning": "thinking", "response": "solution"},
        notes="problem/thinking/solution; same schema as opus46_3000x.",
    ),
    DatasetSpec(
        "opus46_2160x", "Glint-Research/Opus-4.6-Reasoning-2160x", role="sft", format="prompt_response",
        license="apache-2.0", stages=["sft", "reasoning"], require_reasoning=True, verified=True,
        fields={"prompt": "problem", "reasoning": "thinking", "response": "solution"},
        notes="problem/thinking/solution; first rows identical to opus46_3300x -> cross-source dedupe matters.",
    ),
    DatasetSpec(
        "opus46_traceinv", "Jackrong/Claude-opus-4.6-TraceInversion-9000x", role="sft", format="messages",
        license="apache-2.0", stages=["sft", "reasoning"], require_reasoning=True, verified=True,
        fields={"messages": "messages"},
        notes="messages(role/content/thinking) (+ duplicate sharegpt `conversations`); multilingual.",
    ),
    DatasetSpec(
        "openthoughts3", "open-thoughts/OpenThoughts3-1.2M", role="sft", format="sharegpt",
        license="apache-2.0", stages=["sft", "reasoning"], require_reasoning=True, verified=True,
        fields={"messages": "conversations"},
        notes="sharegpt human/gpt; gpt value starts with <think>. Large (1.2M) -> stream.",
    ),
    DatasetSpec(
        "ot_agent_sft", "open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K", role="sft", format="messages",
        license="apache-2.0", stages=["sft", "agentic"], require_reasoning=True, verified=True,
        fields={"messages": "conversations"},
        notes="`conversations` holds role/content (OpenAI style) terminus-2 traces; <think> inline.",
    ),
    DatasetSpec(
        "nemotron_post", "nvidia/Llama-Nemotron-Post-Training-Dataset", config="SFT", split="math+code+science",
        role="sft", format="nemotron", license="cc-by-4.0", stages=["sft", "reasoning"], require_reasoning=True,
        drop_system_prompts=_THINK_SYS, verified=True,
        notes="input(messages)+output(<think> inline)+reasoning on/off; also splits chat/safety and config RL. "
              "Llama-Nemotron 'detailed thinking on/off' system prompts are stripped.",
    ),
    DatasetSpec(
        "arc_agi2_train", "github:arcprize/ARC-AGI-2", split="training", role="sft", format="arc_agi",
        license="apache-2.0", stages=["sft", "arc"], contamination_group="arc_agi_2", verified=True,
        notes="data/training/*.json; one example per test pair. Evaluation split is held out (arc_agi2_eval).",
    ),
    # ------------------------------------------------------------------ pretrain / loop healing
    DatasetSpec(
        "finepdfs_mix", "HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled", role="pretrain",
        format="text", license="odc-by", stages=["loop_heal", "pretrain"], verified=True, min_chars=200,
        notes="globally shuffled 100BT mix; columns text/id/url/dataset.",
    ),
    DatasetSpec(
        "fineweb_edu", "HuggingFaceFW/fineweb_edu_100BT-shuffled", role="pretrain", format="text",
        license="odc-by", stages=["loop_heal", "pretrain"], verified=True, min_chars=200,
        notes="globally shuffled FineWeb-Edu 100BT.",
    ),
    DatasetSpec(
        "davinci", "SII-GAIR-NLP/davinci-llm-data", role="pretrain", format="text", gated=True,
        license="cc-by-4.0", stages=["loop_heal", "pretrain"], verified=False,
        notes="gated (form). Files are *.jsonl.7z which `datasets` cannot stream: extract locally and set "
              "`data_files` (json) in the source config. Column layout unverified.",
    ),
    # ------------------------------------------------------------------ RL
    DatasetSpec(
        "nemotron_rl", "nvidia/Llama-Nemotron-Post-Training-Dataset", config="RL", split="instruction_following",
        role="rl", format="auto", license="cc-by-4.0", stages=["rl"], verified=True,
        notes="instruction-following RL prompts.",
    ),
    DatasetSpec(
        "ot_agent_rl", "open-thoughts/OpenThoughts-Agent-RL-5K", role="rl", format="rl_ot_agent",
        license="apache-2.0", stages=["rl", "agentic"], verified=True,
        notes="path + task_binary (gzipped tar of the task dir); unpacked by formats.extract_task_archive.",
    ),
    DatasetSpec(
        "swe_rebench", "PrimeIntellect/SWE-rebench-V2-Filtered-Verified", role="rl", format="rl_swe",
        license="cc-by-4.0", stages=["rl", "agentic"], verified=True,
        notes="SWE tasks with docker image_name, FAIL_TO_PASS/PASS_TO_PASS, install_config.",
    ),
    DatasetSpec(
        "kernelbench_train", "ScalingIntelligence/KernelBench", split="level_1+level_2", role="rl",
        format="rl_kernelbench", license="see repo", stages=["rl", "kernels"], contamination_group="kernelbench",
        verified=True, notes="levels 1-2 only for training; levels 3-4 are eval (kernelbench_eval).",
    ),
    # ------------------------------------------------------------------ eval (decontamination sources)
    DatasetSpec(
        "aime_2026", "MathArena/aime_2026", role="eval", format="eval_text", license="cc-by-nc-sa-4.0",
        stages=["eval"], contamination_group="aime_2026", text_fields=["problem"], verified=True,
    ),
    DatasetSpec(
        "gpqa_diamond", "Idavidrein/gpqa", config="gpqa_diamond", role="eval", format="eval_text", gated=True,
        license="cc-by-4.0", stages=["eval"], contamination_group="gpqa",
        text_fields=["Question", "Correct Answer"], verified=True,
        notes="gated (auto-accept, no-leak terms). Config gpqa_diamond -> gpqa_diamond.csv (GPQA release columns).",
    ),
    DatasetSpec(
        "hle", "cais/hle", split="test", role="eval", format="eval_text", gated=True, license="mit",
        stages=["eval"], contamination_group="hle", text_fields=["question"], verified=False,
        notes="gated (auto-accept). Repo verified; split/column not re-checked (gated).",
    ),
    DatasetSpec(
        "arc_agi2_eval", "github:arcprize/ARC-AGI-2", split="evaluation", role="eval", format="arc_agi",
        license="apache-2.0", stages=["eval"], contamination_group="arc_agi_2", verified=True,
    ),
    DatasetSpec(
        "kernelbench_eval", "ScalingIntelligence/KernelBench", split="level_3+level_4", role="eval",
        format="eval_text", license="see repo", stages=["eval"], contamination_group="kernelbench",
        text_fields=["code"], verified=True,
    ),
    # ------------------------------------------------------------------ offline synthetic
    DatasetSpec(
        "toy_cot", "builtin:toy_cot", role="sft", format="prompt_response", license="synthetic",
        stages=["test"], require_reasoning=True, verified=True,
        notes="arithmetic word problems with step-by-step reasoning (offline tests).",
    ),
    DatasetSpec(
        "toy_text", "builtin:toy_text", role="pretrain", format="text", license="synthetic",
        stages=["test"], verified=True, notes="synthetic prose (offline tests).",
    ),
]

REGISTRY: dict[str, DatasetSpec] = {}
for _s in _SPECS:
    require(_s.name not in REGISTRY, "duplicate dataset name in registry", name=_s.name)
    REGISTRY[_s.name] = _s


def get_spec(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise KeyError(f"dataset '{name}' is not in the registry. Known: {sorted(REGISTRY)}. "
                       "For an ad-hoc HF dataset give `hf_id` (and `format`) in the source config.")
    return REGISTRY[name]


def list_specs(role: str | None = None, stage: str | None = None) -> list[DatasetSpec]:
    return [s for s in REGISTRY.values() if (role is None or s.role == role) and (stage is None or stage in s.stages)]


def eval_specs_for_group(group: str) -> list[DatasetSpec]:
    return [s for s in REGISTRY.values() if s.role == "eval" and s.contamination_group == group]


# ----------------------------------------------------------------------------- verification

def _verify_github(spec: DatasetSpec, timeout: float) -> dict[str, Any]:
    import json
    import urllib.request

    repo = spec.hf_id.split(":", 1)[1]
    url = f"https://api.github.com/repos/{repo}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        info = json.loads(r.read())
    return {"exists": True, "gated": False, "private": bool(info.get("private")), "access": True,
            "detail": f"github default_branch={info.get('default_branch')}"}


def _verify_hf(spec: DatasetSpec, token: str | None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.dataset_info(spec.hf_id)
    gated = getattr(info, "gated", False)
    out: dict[str, Any] = {"exists": True, "gated": bool(gated), "private": bool(getattr(info, "private", False)),
                           "access": None, "detail": f"sha={str(getattr(info, 'sha', ''))[:10]}"}
    auth_check = getattr(api, "auth_check", None)
    if auth_check is not None:
        try:
            auth_check(spec.hf_id, repo_type="dataset")
            out["access"] = True
        except Exception as e:  # GatedRepoError / RepositoryNotFoundError / HTTP 401
            out["access"] = False
            out["detail"] += f" | no access: {type(e).__name__}"
    if bool(gated) != spec.gated:
        out["detail"] += f" | registry says gated={spec.gated} but hub says gated={gated!r}"
    return out


def verify_specs(specs: list[DatasetSpec] | None = None, token: str | None = None,
                 timeout: float = 20.0) -> list[dict[str, Any]]:
    """Check existence / gating / access of each spec on the Hub (or GitHub). One dict per spec:
    {name, hf_id, role, ok, exists, gated, access, verified, detail}. Never raises per dataset."""
    specs = list(REGISTRY.values()) if specs is None else list(specs)
    report: list[dict[str, Any]] = []
    for s in specs:
        row: dict[str, Any] = {"name": s.name, "hf_id": s.hf_id, "role": s.role, "verified": s.verified,
                               "exists": None, "gated": s.gated, "access": None, "ok": False, "detail": ""}
        try:
            if s.kind == "builtin":
                row.update(exists=True, access=True, detail="builtin synthetic source")
            elif s.kind == "github":
                row.update(_verify_github(s, timeout))
            else:
                row.update(_verify_hf(s, token))
            row["ok"] = bool(row["exists"]) and row["access"] is not False
        except ImportError as e:
            row["detail"] = f"huggingface_hub not installed ({e}); pip install -e '.[colab]'"
        except Exception as e:
            row["detail"] = f"{type(e).__name__}: {str(e)[:300]}"
        level = logging.INFO if row["ok"] else logging.WARNING
        log.log(level, "dataset %-18s %-60s ok=%s gated=%s access=%s %s", s.name, s.hf_id, row["ok"],
                row["gated"], row["access"], row["detail"])
        report.append(row)
    return report


def format_verify_report(report: list[dict[str, Any]]) -> str:
    lines = [f"{'name':<18} {'role':<9} {'ok':<5} {'gated':<6} {'access':<6} {'verified':<8} hf_id / detail"]
    for r in report:
        lines.append(f"{r['name']:<18} {r['role']:<9} {r['ok']!s:<5} {r['gated']!s:<6} {r['access']!s:<6} "
                     f"{r['verified']!s:<8} {r['hf_id']}  {r['detail']}")
    n_bad = sum(not r["ok"] for r in report)
    lines.append(f"{len(report) - n_bad}/{len(report)} datasets reachable")
    return "\n".join(lines)
