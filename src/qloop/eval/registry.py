"""Benchmark registry: where the data lives, how to prompt, how to score, what it needs.

`verified=True` means hf_id/config/split/field names were checked against the dataset card at
the time of writing; `False` means "best knowledge, adapters are defensive" - the adapters
accept several field-name variants and `load_items` reports the columns it actually saw.

`requires` lists capabilities that must be available to score a benchmark locally:
    code_exec       run generated Python in subprocesses (LiveCodeBench)
    gpu_exec        compile + time CUDA kernels (KernelBench)
    vision          image inputs (MMMU-Pro, HLE image subset)
    api_judge       an external LLM (judge / user simulator)
    remote_sandbox  per-task containers via Harbor + Daytona/Modal (no Docker daemon on Colab)
    docker          a local Docker daemon
    web_agent       tool use / browsing (GAIA)
Anything with unmet requirements is routed to `qloop.eval.external` instead.

Contamination groups: benchmarks sharing a group must not be used as held-out eval when any
member (or its source) was in training data. AgentWorldBench is built from Terminal-Bench 2.0
trajectories => both are in group "terminal_bench".
"""
from __future__ import annotations

import ast
import json
import logging
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable

from qloop.debug import require

log = logging.getLogger("qloop")

LETTERS = "ABCDEFGHIJ"


@dataclass
class BenchmarkSpec:
    name: str
    hf_id: str
    config: str | None
    split: str
    kind: str                       # aime|math|mcq|longmcq|mrcr|hle|arc|code|kernel|gaia|agentic|vision_mcq
    scorer: str | None              # key into scoring.get_scorer (None => external harness only)
    requires: set[str] = field(default_factory=set)
    gated: bool = False
    contamination_group: str = ""
    prompt_builder: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None
    notes: str = ""
    verified: bool = False
    adapter: Callable[[dict[str, Any], int], list[dict[str, Any]]] | None = None
    max_new_tokens: int = 8192
    harness: str | None = None      # external harness name (harbor | tau_bench | openai_endpoint)
    extra: dict[str, Any] = field(default_factory=dict)
    revision: str | None = None

    def build_prompt(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        if item.get("messages"):
            return item["messages"]
        require(self.prompt_builder is not None, f"benchmark {self.name} has no prompt_builder", item=item.get("id"))
        return self.prompt_builder(item)


# ============================================================================ helpers

def _first(row: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] is not None and row[k] != "":
            return row[k]
    return default


def _choices_block(choices: list[str], style: str = "{L}) {c}") -> str:
    return "\n".join(style.format(L=LETTERS[i], c=str(c).strip()) for i, c in enumerate(choices))


def _letter(ans: Any, choices: list[str] | None = None) -> str:
    s = str(ans).strip()
    if len(s) == 1 and s.upper() in LETTERS:
        return s.upper()
    if s.isdigit() and choices is not None and int(s) < len(choices):
        return LETTERS[int(s)]
    if choices is not None:
        for i, c in enumerate(choices):
            if str(c).strip() == s:
                return LETTERS[i]
    if len(s) >= 1 and s[0].upper() in LETTERS and (len(s) == 1 or not s[1].isalnum()):
        return s[0].upper()
    if len(s) >= 3 and s[0] in "([" and s[1].upper() in LETTERS and s[2] in ")]":
        return s[1].upper()
    return s[:1].upper()


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes")
    return bool(v)


def _parse_chat_messages(raw: Any) -> list[dict[str, Any]] | None:
    parsed = _maybe_json(raw)
    if isinstance(parsed, list) and parsed and all(isinstance(m, dict) and "role" in m for m in parsed):
        return [{"role": str(m["role"]), "content": "" if m.get("content") is None else str(m["content"])}
                for m in parsed]
    return None


def _grid_text(g: list[list[int]]) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in g)


def _maybe_json(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return x
    return x


# ============================================================================ prompt builders

MATH_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def math_prompt(item: dict[str, Any]) -> list[dict[str, Any]]:
    extra = "\nThe answer is an integer between 0 and 999 inclusive." if item.get("integer_answer") else ""
    return [{"role": "user", "content": f"{item['question']}\n\n{MATH_SUFFIX}{extra}"}]


def mcq_prompt(item: dict[str, Any]) -> list[dict[str, Any]]:
    letters = LETTERS[: len(item["choices"])]
    body = (f"{item['question']}\n\n{_choices_block(item['choices'])}\n\n"
            f"Answer with the letter of the correct option ({', '.join(letters)}). "
            "Think step by step, then end your response with a line of the form 'Answer: X'.")
    return [{"role": "user", "content": body}]


def longbench_prompt(item: dict[str, Any], max_context_chars: int = 400_000) -> list[dict[str, Any]]:
    ctx = item.get("context", "")
    if len(ctx) > max_context_chars:  # LongBench v2 protocol: keep head and tail, drop the middle
        half = max_context_chars // 2
        ctx = ctx[:half] + "\n...\n" + ctx[-half:]
    ch = item["choices"]
    body = ("Please read the following text and answer the question below.\n\n<text>\n" + ctx + "\n</text>\n\n"
            f"What is the correct answer to this question: {item['question']}\nChoices:\n"
            + _choices_block(ch, "({L}) {c}")
            + '\n\nFormat your response as follows: "The correct answer is (insert answer here)".')
    return [{"role": "user", "content": body}]


HLE_SYSTEM_EXACT = ("Your response should be in the following format:\nExplanation: {your explanation for your final "
                    "answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between "
                    "0% and 100% for your answer}")
HLE_SYSTEM_MC = ("Your response should be in the following format:\nExplanation: {your explanation for your answer "
                 "choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for "
                 "your answer}")


def hle_prompt(item: dict[str, Any]) -> list[dict[str, Any]]:
    mc = str(item.get("answer_type", "")).lower().startswith("multiple")
    return [{"role": "system", "content": HLE_SYSTEM_MC if mc else HLE_SYSTEM_EXACT},
            {"role": "user", "content": item["question"]}]


def arc_prompt(item: dict[str, Any]) -> list[dict[str, Any]]:
    parts = ["Find the common rule that maps each input grid to its output grid (digits 0-9 are colours). "
             "Then apply the rule to the test input.\n"]
    for k, ex in enumerate(item["train"], 1):
        parts.append(f"Example {k} input:\n{_grid_text(ex['input'])}\n"
                     f"Example {k} output:\n{_grid_text(ex['output'])}\n")
    parts.append(f"Test input:\n{_grid_text(item['test_input'])}\n")
    parts.append("Give the test output grid as the LAST thing in your answer, as a JSON list of lists of integers, "
                 "e.g. [[0,1],[1,0]].")
    return [{"role": "user", "content": "\n".join(parts)}]


def code_prompt(item: dict[str, Any]) -> list[dict[str, Any]]:
    head = ("You will be given a question (problem specification) and will generate a correct Python program that "
            "matches the specification and passes all tests.\n\n")
    q = f"Question: {item['question']}\n\n"
    if item.get("starter_code"):
        fmt = ("You will use the following starter code to write the solution to the problem and enclose your code "
               f"within delimiters.\n```python\n{item['starter_code']}\n```\n")
    else:
        fmt = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on "
               "the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
               "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
               "```python\n# YOUR CODE HERE\n```\n")
    return [{"role": "user", "content": head + q + fmt}]


KERNEL_INSTRUCTIONS = (
    "You write custom CUDA kernels (via torch.utils.cpp_extension.load_inline) or Triton kernels to replace the "
    "PyTorch operators in the given architecture to get speedups. You may replace some operators and leave others, "
    "fuse operators, or change algorithms (e.g. online softmax). The new code must be numerically equivalent "
    "(atol/rtol 1e-2) and keep the same constructor and forward signatures.\n\n"
    "Name your optimized architecture `ModelNew`. Output the complete new code in a single ```python code block. "
    "Do not output testing code.")


def kernel_prompt(item: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": f"{KERNEL_INSTRUCTIONS}\n\nYou are given the following architecture:\n"
                                        f"```python\n{item['ref_src']}\n```\nOptimize it."}]


GAIA_SYSTEM = (
    "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the "
    "following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as "
    "possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to "
    "write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a "
    "string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless "
    "specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the "
    "element to be put in the list is a number or a string.")


def gaia_prompt(item: dict[str, Any]) -> list[dict[str, Any]]:
    q = item["question"]
    if item.get("file_name"):
        q += f"\n\n(Attached file: {item['file_name']})"
    return [{"role": "system", "content": GAIA_SYSTEM}, {"role": "user", "content": q}]


def agentic_prompt(item: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": str(item.get("question") or item.get("instruction") or "")}]


def vision_mcq_prompt(item: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in item.get("images", [])]
    text = item.get("question", "")
    if item.get("choices"):
        text += "\n\n" + _choices_block(item["choices"])
    text += "\n\nThink step by step, then end your response with a line of the form 'Answer: X'."
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def mrcr_prompt(item: dict[str, Any]) -> list[dict[str, Any]]:
    raw = item.get("messages") if item.get("messages") is not None else item.get("prompt")
    msgs = _parse_chat_messages(raw)
    if msgs is not None:
        return msgs
    q = item.get("prompt") if item.get("prompt") is not None else item.get("question", "")
    return [{"role": "user", "content": str(q)}]


# ============================================================================ adapters (raw row -> items)

def _id(row: dict[str, Any], idx: int, keys: Iterable[str] = ("id", "_id", "task_id", "instance_id")) -> str:
    return str(_first(row, keys, idx))


def adapt_aime(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    q = _first(row, ("problem", "question", "Problem", "Question"))
    a = _first(row, ("answer", "Answer", "solution_answer"))
    require(q is not None and a is not None, "AIME row needs problem/question + answer", columns=sorted(row))
    return [{"id": str(_first(row, ("problem_idx", "id", "ID"), idx)), "question": str(q), "answer": str(a).strip(),
             "integer_answer": True}]


def adapt_hle(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    return [{"id": _id(row, idx), "question": row["question"], "answer": str(row["answer"]),
             "answer_type": row.get("answer_type", "exactMatch"), "has_image": bool(row.get("image")),
             "category": row.get("category", ""), "image": row.get("image") or None}]


def adapt_gpqa(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    if "choices" in row and "answer" in row:  # already normalised
        return [dict(row, id=_id(row, idx))]
    q = _first(row, ("Question", "question"))
    correct = _first(row, ("Correct Answer", "correct_answer"))
    wrong = [_first(row, (f"Incorrect Answer {k}", f"incorrect_answer_{k}")) for k in (1, 2, 3)]
    require(q is not None and correct is not None and all(w is not None for w in wrong),
            "GPQA row needs Question / Correct Answer / Incorrect Answer 1-3", columns=sorted(row))
    rid = str(_first(row, ("Record ID", "id"), idx))
    opts = [str(correct).strip(), *[str(w).strip() for w in wrong]]
    perm = list(range(4))
    random.Random(f"gpqa:{rid}").shuffle(perm)  # deterministic per item
    choices = [opts[p] for p in perm]
    return [{"id": rid, "question": str(q).strip(), "choices": choices, "answer": LETTERS[perm.index(0)]}]


def adapt_longbench(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    choices = row.get("choices") or [row[f"choice_{L}"] for L in "ABCD"]
    return [{"id": _id(row, idx), "question": row["question"], "choices": list(choices),
             "context": row.get("context", ""),
             "answer": _letter(row["answer"], list(choices)), "difficulty": row.get("difficulty"),
             "length": row.get("length"), "domain": row.get("domain")}]


def adapt_mmmu_pro(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    opts = _maybe_json(row.get("options", []))
    opts = list(opts) if isinstance(opts, (list, tuple)) else []
    images = [row[k] for k in sorted(row) if k.startswith("image") and row[k] is not None and k != "image_type"]
    return [{"id": _id(row, idx), "question": row.get("question", ""), "choices": opts, "images": images,
             "answer": _letter(row["answer"], opts), "subject": row.get("subject")}]


def adapt_arc(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    """Accepts the ARC JSON task format ({train:[{input,output}], test:[{input,output?}]}), possibly as strings."""
    task = dict(row)
    for k in ("train", "test"):
        task[k] = _maybe_json(task.get(k))
    if not isinstance(task.get("train"), list):
        for alt in ("task", "data", "json"):
            if alt in row:
                return adapt_arc(dict(_maybe_json(row[alt]), id=_id(row, idx, ("id", "task_id", "name"))), idx)
        raise ValueError(f"ARC row has no train/test lists; columns={sorted(row)}")
    tid = _id(row, idx, ("id", "task_id", "name"))
    items = []
    for j, t in enumerate(task.get("test") or []):
        items.append({"id": f"{tid}#{j}", "task_id": tid, "test_index": j, "train": task["train"],
                      "test_input": t["input"], "answer": t.get("output")})
    return items


def adapt_lcb(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    from .code_exec import decode_lcb_tests
    meta = _maybe_json(row.get("metadata") or "{}")
    meta = meta if isinstance(meta, dict) else {}
    return [{"id": str(_first(row, ("question_id", "id"), idx)),
             "question": _first(row, ("question_content", "question")),
             "starter_code": row.get("starter_code") or "", "fn_name": meta.get("func_name") or row.get("fn_name"),
             "public_tests": decode_lcb_tests(row.get("public_test_cases") or row.get("public_tests")),
             "private_tests": decode_lcb_tests(row.get("private_test_cases") or row.get("private_tests")),
             "contest_date": str(row.get("contest_date", "")), "difficulty": row.get("difficulty"),
             "platform": row.get("platform"), "answer": None}]


def adapt_kernelbench(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    lvl = _first(row, ("level",), "")
    pid = _first(row, ("problem_id", "id"), idx)
    return [{"id": f"L{lvl}_{pid}", "ref_src": _first(row, ("code", "ref_src")), "name": row.get("name", ""),
             "level": lvl, "answer": None}]


def adapt_gaia(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    return [{"id": _id(row, idx), "question": _first(row, ("Question", "question")),
             "answer": _first(row, ("Final answer", "answer"), None), "level": row.get("Level"),
             "file_name": row.get("file_name") or ""}]


def adapt_hle_verified(row: dict[str, Any], idx: int, *,
                       subsets: Iterable[str] | None = ("gold", "revision")) -> list[dict[str, Any]]:
    q = _first(row, ("question", "Question"))
    a = _first(row, ("answer", "Answer"))
    require(q is not None and a is not None, "HLE-Verified row needs question + answer", columns=sorted(row))
    subset = str(row.get("subset") or "").strip().lower()
    if subsets is not None and "subset" in row:
        allowed = {str(s).strip().lower() for s in subsets}
        if subset not in allowed:
            return []
    has_img = (_as_bool(row["has_image"]) if "has_image" in row and row["has_image"] is not None
               else bool(row.get("image")))
    return [{
        "id": _id(row, idx),
        "question": str(q),
        "answer": str(a),
        "answer_type": row.get("answer_type") or "exactMatch",
        "has_image": has_img,
        "subset": subset,
        "subset_raw": row.get("subset_raw", ""),
        "raw_subject": row.get("raw_subject", ""),
        "category": row.get("category", ""),
        "problem_is_valid": row.get("problem_is_valid"),
        "answer_is_valid": row.get("answer_is_valid"),
        "rationale_is_valid": row.get("rationale_is_valid"),
        "image": row.get("image") or None,
    }]


def adapt_supergpqa(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    q = _first(row, ("question", "Question"))
    opts = _maybe_json(row.get("options"))
    require(q is not None and isinstance(opts, (list, tuple)) and len(opts) > 0,
            "SuperGPQA row needs question + non-empty options list", columns=sorted(row))
    choices = [str(c).strip() for c in opts]
    require(len(choices) <= len(LETTERS), "SuperGPQA row has more options than supported letters",
            n_choices=len(choices), max_choices=len(LETTERS))
    ans_letter = _first(row, ("answer_letter",))
    raw_ans = ans_letter if ans_letter is not None else _first(row, ("answer", "answer_index"))
    require(raw_ans is not None, "SuperGPQA row needs answer_letter or answer", columns=sorted(row))
    ans = _letter(raw_ans, choices)
    require(ans in LETTERS[: len(choices)], "SuperGPQA answer letter out of range",
            answer=ans, n_choices=len(choices))
    return [{
        "id": _id(row, idx, ("uuid", "id", "_id")),
        "question": str(q).strip(),
        "choices": choices,
        "answer": ans,
        "discipline": row.get("discipline", ""),
        "field": row.get("field", ""),
        "subfield": row.get("subfield", ""),
        "difficulty": row.get("difficulty", ""),
        "is_calculation": row.get("is_calculation"),
    }]


def adapt_mmlu_pro(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    q = _first(row, ("question", "Question"))
    opts = _maybe_json(row.get("options"))
    require(q is not None and isinstance(opts, (list, tuple)) and len(opts) > 0,
            "MMLU-Pro row needs question + non-empty options list", columns=sorted(row))
    choices = [str(c).strip() for c in opts]
    require(len(choices) <= len(LETTERS), "MMLU-Pro row has more options than supported letters",
            n_choices=len(choices), max_choices=len(LETTERS))
    raw_ans = _first(row, ("answer", "answer_index"))
    require(raw_ans is not None, "MMLU-Pro row needs answer or answer_index", columns=sorted(row))
    ans = _letter(raw_ans, choices)
    require(ans in LETTERS[: len(choices)], "MMLU-Pro answer letter out of range",
            answer=ans, n_choices=len(choices))
    return [{
        "id": str(_first(row, ("question_id", "id", "_id"), idx)),
        "question": str(q).strip(),
        "choices": choices,
        "answer": ans,
        "category": row.get("category", ""),
        "src": row.get("src", ""),
    }]


def adapt_mrcr(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    raw_prompt = row.get("prompt")
    ans = row.get("answer")
    require(raw_prompt is not None and ans is not None, "MRCR row needs prompt + answer", columns=sorted(row))
    msgs = _parse_chat_messages(raw_prompt)
    item: dict[str, Any] = {
        "id": _id(row, idx, ("id", "_id", "idx")),
        "answer": str(ans),
        "random_string_to_prepend": str(row.get("random_string_to_prepend") or ""),
        "n_needles": row.get("n_needles"),
        "desired_msg_index": row.get("desired_msg_index"),
        "total_messages": row.get("total_messages"),
        "n_chars": row.get("n_chars"),
        "date_added": row.get("date_added"),
    }
    if msgs is not None:
        item["messages"] = msgs
    else:
        item["prompt"] = str(raw_prompt)
    return [item]


def adapt_math_arena(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    q = _first(row, ("problem", "question", "Problem", "Question"))
    a = _first(row, ("answer", "Answer", "solution_answer"))
    require(q is not None and a is not None, "MathArena row needs problem/question + answer", columns=sorted(row))
    item: dict[str, Any] = {
        "id": str(_first(row, ("problem_idx", "id", "ID"), idx)),
        "question": str(q),
        "answer": str(a).strip(),
        "integer_answer": False,
    }
    if "problem_type" in row:
        item["problem_type"] = row["problem_type"]
    if "source" in row:
        item["source"] = row["source"]
    return [item]


def adapt_browsecomp_long(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    q = _first(row, ("problem", "question", "Problem", "Question"))
    a = _first(row, ("answer", "Answer"))
    require(q is not None and a is not None, "BrowseCompLongContext row needs problem + answer", columns=sorted(row))
    return [{
        "id": _id(row, idx, ("id", "_id", "problem_id")),
        "question": str(q),
        "answer": str(a),
        "answer_type": "exactMatch",
        "urls": row.get("urls") or [],
        "canary": row.get("canary") or "",
    }]


def adapt_frames(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    q = _first(row, ("Prompt", "prompt", "question", "Question"))
    a = _first(row, ("Answer", "answer"))
    require(q is not None and a is not None, "FRAMES row needs Prompt + Answer", columns=sorted(row))
    rid = _first(row, ("id", "_id", "Unnamed: 0"), idx)
    return [{
        "id": str(rid),
        "question": str(q),
        "answer": str(a),
        "answer_type": "exactMatch",
        "reasoning_types": row.get("reasoning_types", ""),
    }]


def adapt_terminal_bench(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    instr = _first(row, ("instruction", "question", "problem_statement", "prompt", "task"))
    require(instr is not None, "Terminal-Bench row needs instruction", columns=sorted(row))
    name = str(_first(row, ("name", "task_id", "id", "_id"), idx))
    return [{
        "id": name,
        "name": name,
        "question": str(instr),
        "instruction": str(instr),
        "description": row.get("description") or "",
        "keywords": row.get("keywords") or [],
        "schema_version": row.get("schema_version"),
        "location": row.get("location") or "",
        "metadata": row.get("metadata"),
        "answer": None,
    }]


def adapt_generic(row: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    q = _first(row, ("question", "problem_statement", "instruction", "prompt", "task", "Question"), "")
    return [{"id": _id(row, idx), "question": q, "answer": _first(row, ("answer", "Final answer"), None), "raw": row}]


# ============================================================================ registry

REGISTRY: dict[str, BenchmarkSpec] = {}


def register(spec: BenchmarkSpec, overwrite: bool = False) -> BenchmarkSpec:
    if spec.name in REGISTRY and not overwrite:
        raise ValueError(f"benchmark {spec.name!r} already registered")
    REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> BenchmarkSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown benchmark {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


def available_capabilities(extra: Iterable[str] = ()) -> set[str]:
    caps = {"code_exec"}
    try:
        import torch
        if torch.cuda.is_available():
            caps.add("gpu_exec")
    except Exception:
        pass
    if os.path.exists("/var/run/docker.sock") or os.environ.get("DOCKER_HOST"):
        caps.add("docker")
    return caps | set(extra)


def missing_requirements(spec: BenchmarkSpec, capabilities: Iterable[str] | None = None) -> set[str]:
    caps = available_capabilities() if capabilities is None else set(capabilities)
    return set(spec.requires) - caps


def is_locally_scorable(spec: BenchmarkSpec, capabilities: Iterable[str] | None = None) -> bool:
    return spec.scorer is not None and not missing_requirements(spec, capabilities)


def list_specs(local_only: bool = False, capabilities: Iterable[str] | None = None) -> list[BenchmarkSpec]:
    return [s for s in REGISTRY.values() if not local_only or is_locally_scorable(s, capabilities)]


def contamination_conflicts(trained_groups: Iterable[str], names: Iterable[str] | None = None) -> dict[str, str]:
    """{benchmark: group} for benchmarks whose contamination group intersects the training data groups."""
    tg = set(trained_groups)
    return {s.name: s.contamination_group for s in REGISTRY.values()
            if (names is None or s.name in set(names)) and s.contamination_group in tg}


def _load_hf_rows(spec: BenchmarkSpec, token: str | None = None) -> Iterable[dict[str, Any]]:
    try:
        import datasets
    except ImportError as e:
        raise ImportError(f"loading {spec.hf_id} needs `pip install datasets` (or pass offline_items=)") from e
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kw: dict[str, Any] = {"split": spec.split, "token": tok}
    if spec.revision is not None:
        kw["revision"] = spec.revision
    try:
        ds = datasets.load_dataset(spec.hf_id, spec.config, **kw)
    except Exception as e:
        hint = " (gated dataset: accept the terms on the HF page and set HF_TOKEN)" if spec.gated else ""
        rev = f", revision={spec.revision!r}" if spec.revision is not None else ""
        raise RuntimeError(f"load_dataset({spec.hf_id!r}, {spec.config!r}, split={spec.split!r}{rev}) failed{hint}: "
                           f"{type(e).__name__}: {e}") from e
    log.info("loaded %s/%s[%s]%s: %d rows, columns=%s", spec.hf_id, spec.config, spec.split,
             f"@{spec.revision}" if spec.revision else "", len(ds), getattr(ds, "column_names", "?"))
    return ds


def load_items(spec: BenchmarkSpec, limit: int | None = None, offline_items: list[dict[str, Any]] | None = None, *,
               token: str | None = None, include_multimodal: bool | None = None) -> list[dict[str, Any]]:
    """Rows (HF `datasets`, or `offline_items` in raw or normalised form) -> normalised items."""
    if offline_items is not None:
        rows: Iterable[dict[str, Any]] = offline_items
        src = f"{len(offline_items)} offline rows"
    else:
        rows, src = _load_hf_rows(spec, token=token), spec.hf_id
    adapter = spec.adapter or adapt_generic
    mm = ("vision" in spec.requires) if include_multimodal is None else include_multimodal
    min_date = spec.extra.get("min_contest_date")
    raw_subsets = spec.extra.get("subsets")
    allowed_subsets = {str(s).strip().lower() for s in raw_subsets} if raw_subsets is not None else None
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, row in enumerate(rows):
        try:
            if adapter is adapt_hle_verified and "subsets" in spec.extra:
                produced = adapt_hle_verified(dict(row), idx, subsets=raw_subsets)
            else:
                produced = adapter(dict(row), idx)
        except Exception as e:
            raise ValueError(f"{spec.name}: adapter failed on row {idx} ({src}); columns={sorted(dict(row))}: "
                             f"{type(e).__name__}: {e}") from e
        for it in produced:
            if it.get("has_image") and not mm:
                continue
            if (allowed_subsets is not None and "subset" in it
                    and str(it["subset"]).strip().lower() not in allowed_subsets):
                continue
            if min_date and str(it.get("contest_date", "")) and str(it.get("contest_date", ""))[:10] < str(min_date):
                continue
            if it["id"] in seen:
                it["id"] = f"{it['id']}~{idx}"
            seen.add(it["id"])
            items.append(it)
        if limit is not None and len(items) >= limit:
            break
    return items[:limit] if limit is not None else items


# ---------------------------------------------------------------------------- the benchmarks

def _register_defaults() -> None:
    S = BenchmarkSpec
    for spec in (
        S("hle", "cais/hle", None, "test", "hle", "hle", set(), gated=True, contamination_group="hle",
          prompt_builder=hle_prompt, adapter=adapt_hle, max_new_tokens=16384, verified=True,
          notes="Humanity's Last Exam (2.5k). Official grading uses an LLM judge (o3-mini) => local score is "
                "normalised exact-match (+math equivalence), a lower bound. ~14% image questions are skipped "
                "unless include_multimodal=True with a vision model.",
          extra={"official_requires": ["api_judge"], "optional_requires": ["vision"]}),
        S("hle_rolling", "cais/hle-rolling", None, "test", "hle", "hle", set(), gated=True, contamination_group="hle",
          prompt_builder=hle_prompt, adapter=adapt_hle, max_new_tokens=16384, verified=True,
          notes="Humanity's Last Exam Rolling (1,000 rows, identical schema to cais/hle). Image questions skipped "
                "unless include_multimodal=True.",
          extra={"official_requires": ["api_judge"], "optional_requires": ["vision"]}),
        S("hle_diamond", "cais/hle-diamond", None, "test", "hle", "hle", set(), gated=True, contamination_group="hle",
          prompt_builder=hle_prompt, adapter=adapt_hle, max_new_tokens=16384, verified=True,
          notes="Humanity's Last Exam Diamond (1,000 frozen cleaned questions: 500 reasoning + 500 expert knowledge, "
                "Sep 2026 baseline; identical schema to cais/hle).",
          extra={"official_requires": ["api_judge"], "optional_requires": ["vision"]}),
        S("hle_verified", "lmms-lab/HLE-Verified", None, "test", "hle", "hle", set(), gated=False,
          contamination_group="hle", prompt_builder=hle_prompt, adapter=adapt_hle_verified, max_new_tokens=16384,
          verified=True,
          notes="Open audited HLE-Verified benchmark; defaults to audited subsets {'gold', 'revision'} (override via "
                "extra.subsets) and skips has_image=True rows unless include_multimodal=True.",
          extra={"subsets": ["gold", "revision"], "official_requires": ["api_judge"],
                 "optional_requires": ["vision"]}),
        S("swe_bench_pro", "ScaleAI/SWE-bench_Pro", None, "test", "agentic", None, {"remote_sandbox"},
          contamination_group="swe_bench", prompt_builder=agentic_prompt, adapter=adapt_generic,
          harness="harbor", verified=False,
          notes="Repository-level patches scored by running tests in per-instance images. Run through Harbor "
                "(--env daytona|modal) with the agent pointed at qloop.eval.server.",
          extra={"harbor_dataset": "swebenchpro"}),
        S("swe_bench_pro_hard", "ScaleAI/SWE-bench_Pro", "hard", "test", "agentic", None, {"remote_sandbox"},
          gated=False, contamination_group="swe_bench", prompt_builder=agentic_prompt, adapter=adapt_generic,
          harness="harbor", verified=True,
          notes="SWE-bench Pro Hard subset. Run through Harbor (--dataset swebenchpro-hard --env daytona|modal) "
                "with the agent pointed at qloop.eval.server.",
          extra={"harbor_dataset": "swebenchpro-hard"}),
        S("deep_swe", "datacurve/deep-swe", None, "test", "agentic", None, {"remote_sandbox"},
          gated=True, contamination_group="swe_bench", prompt_builder=agentic_prompt, adapter=adapt_generic,
          harness="harbor", verified=True,
          notes="DeepSWE v1.1 (Inkling 2 Mini agentic software engineering benchmark). Run through Harbor remote "
                "sandbox against qloop.eval.server."),
        S("aime_2026", "MathArena/aime_2026", None, "train", "aime", "aime", set(), contamination_group="aime_2026",
          prompt_builder=math_prompt, adapter=adapt_aime, max_new_tokens=32768, verified=False,
          notes="30 problems (AIME I+II 2026), integer answers 0-999. Report avg@k (k>=4) - n=30 is noisy."),
        S("hmmt_feb_2026", "MathArena/hmmt_feb_2026", None, "train", "math", "math", set(), gated=False,
          contamination_group="hmmt_2026", prompt_builder=math_prompt, adapter=adapt_math_arena,
          max_new_tokens=32768, verified=True,
          notes="HMMT Feb 2026 (33 competition math problems with symbolic answers, scored by MATH equivalence)."),
        S("apex_shortlist", "MathArena/apex-shortlist", None, "train", "math", "math", set(), gated=False,
          contamination_group="apex_math", prompt_builder=math_prompt, adapter=adapt_math_arena,
          max_new_tokens=32768, verified=True,
          notes="APEX Shortlist (47 Olympiad math problems with symbolic answers, scored by MATH equivalence)."),
        S("livecodebench", "livecodebench/code_generation_lite", "release_v6", "test", "code", "lcb", {"code_exec"},
          contamination_group="livecodebench", prompt_builder=code_prompt, adapter=adapt_lcb, max_new_tokens=32768,
          verified=False,
          notes="config = version_tag. Filter by contest date after the base model cutoff via "
                "extra.min_contest_date. Requires trust_remote_code on newer datasets versions.",
          extra={"min_contest_date": "2025-02-01"}),
        S("gpqa_diamond", "Idavidrein/gpqa", "gpqa_diamond", "train", "mcq", "mcq", set(), gated=True,
          contamination_group="gpqa", prompt_builder=mcq_prompt, adapter=adapt_gpqa, max_new_tokens=16384,
          verified=True, notes="198 questions; choices shuffled deterministically per Record ID."),
        S("gpqa_extended", "Idavidrein/gpqa", "gpqa_extended", "train", "mcq", "mcq", set(), gated=True,
          contamination_group="gpqa", prompt_builder=mcq_prompt, adapter=adapt_gpqa, max_new_tokens=16384,
          verified=True, notes="546 graduate-level questions; choices shuffled deterministically per Record ID."),
        S("supergpqa", "m-a-p/SuperGPQA", None, "train", "mcq", "mcq", set(), gated=False,
          contamination_group="supergpqa", prompt_builder=mcq_prompt, adapter=adapt_supergpqa, max_new_tokens=16384,
          verified=True, notes="26,529 graduate-level MCQs (up to 10 options A-J) across 285 disciplines."),
        S("mmlu_pro", "TIGER-Lab/MMLU-Pro", None, "test", "mcq", "mcq", set(), gated=False,
          contamination_group="mmlu_pro", prompt_builder=mcq_prompt, adapter=adapt_mmlu_pro, max_new_tokens=8192,
          verified=True, notes="MMLU-Pro 10-way multiple-choice reasoning benchmark (options A-J)."),
        S("frames", "google/frames-benchmark", None, "test", "hle", "hle", set(), gated=False,
          contamination_group="frames", prompt_builder=hle_prompt, adapter=adapt_frames, max_new_tokens=8192,
          verified=True, notes="Google FRAMES multi-hop factuality & retrieval benchmark (Prompt/Answer)."),
        S("mrcr", "openai/mrcr", None, "train", "mrcr", "mrcr", set(), gated=False,
          contamination_group="mrcr", prompt_builder=mrcr_prompt, adapter=adapt_mrcr, max_new_tokens=4096,
          verified=True,
          notes="OpenAI MRCR (2,400 multi-needle long-context conversations up to 1M tokens). Graded by "
                "prefix-checked SequenceMatcher ratio after stripping <think> blocks."),
        S("browsecomp_long", "openai/BrowseCompLongContext", None, "train", "hle", "hle", set(), gated=False,
          contamination_group="browsecomp", prompt_builder=hle_prompt, adapter=adapt_browsecomp_long,
          max_new_tokens=16384, verified=True,
          notes="OpenAI BrowseComp LongContext (problem/answer/urls/canary); local exact/math lower bound.",
          extra={"official_requires": ["api_judge"]}),
        S("openthoughts_tb_dev_v2", "open-thoughts/OpenThoughts-TB-dev-v2", None, "train", "agentic", None,
          {"remote_sandbox"}, contamination_group="openthoughts_agent", prompt_builder=agentic_prompt,
          adapter=adapt_generic, harness="harbor", verified=False,
          notes="Terminal-agent dev set in Harbor task format; use as the fast dev proxy for Terminal-Bench 4.1/2.0. "
                "Contaminated if trained on OpenThoughts-Agent SFT/RL data."),
        S("arc_agi_2", "arcprize/ARC-AGI-2", None, "evaluation", "arc", "arc", set(), contamination_group="arc_agi_2",
          prompt_builder=arc_prompt, adapter=adapt_arc, max_new_tokens=16384, verified=False,
          notes="Public evaluation set (120 tasks); items are flattened per test input (id 'task#j'). Official "
                "metric: pass@2 per test output => run with n_samples>=2 and read pass@2."),
        S("arc_agi_2_human", "arcprize/arc_agi_2_human_testing", None, "train", "arc", "arc", set(),
          contamination_group="arc_agi_2", prompt_builder=arc_prompt, adapter=adapt_arc, max_new_tokens=16384,
          verified=False, notes="Human testing set of ARC-AGI-2 (same contamination group as the public eval)."),
        S("terminal_bench_4_1", "harborframework/terminal-bench", None, "test", "agentic", None, {"remote_sandbox"},
          gated=False, contamination_group="terminal_bench", prompt_builder=agentic_prompt,
          adapter=adapt_terminal_bench, harness="harbor", verified=True, revision="v4.1.0",
          notes="Terminal-Bench 4.1 (66 hardened container tasks with RewardKit anti-reward-hacking verifiers; "
                "default replacing legacy 2.0/4.0). `harbor run --dataset terminal-bench/terminal-bench@4.1.0 "
                "--env daytona` (Colab: no Docker).",
          extra={"harbor_dataset": "terminal-bench/terminal-bench@4.1.0"}),
        S("terminal_bench_2", "harborframework/terminal-bench-2.0", None, "train", "agentic", None, {"remote_sandbox"},
          contamination_group="terminal_bench", prompt_builder=agentic_prompt, adapter=adapt_generic,
          harness="harbor", verified=False,
          notes="Legacy 89 container tasks (superseded as default by terminal_bench_4_1). "
                "`harbor run --dataset terminal-bench@2.0 --env daytona` (Colab: no Docker).",
          extra={"harbor_dataset": "terminal-bench@2.0"}),
        S("agentworldbench", "Qwen/AgentWorldBench", None, "test", "agentic", None, {"remote_sandbox"},
          contamination_group="terminal_bench", prompt_builder=agentic_prompt, adapter=adapt_generic,
          harness="harbor", verified=False,
          notes="Built from Terminal-Bench 2.0 trajectories => same contamination group 'terminal_bench': never "
                "train on one and report Terminal-Bench 2.0/4.1 as held-out."),
        S("tau_bench_lilt", "lilt-ai/tau-bench-lilt", None, "train", "agentic", None, {"api_judge"},
          contamination_group="tau_bench", prompt_builder=agentic_prompt, adapter=adapt_generic,
          harness="tau_bench", verified=False,
          notes="tau-bench variant; needs an LLM user simulator (API key) and the tau-bench harness, with the "
                "agent model served by qloop.eval.server. Metric: pass^k."),
        S("gaia", "gaia-benchmark/GAIA", "2023_all", "validation", "gaia", "gaia", {"web_agent"}, gated=True,
          contamination_group="gaia", prompt_builder=gaia_prompt, adapter=adapt_gaia, harness="openai_endpoint",
          verified=True,
          notes="Validation split has answers (test answers are hidden => leaderboard only). Needs tools/browsing; "
                "closed-book runs (capabilities=['web_agent'] override) are a lower bound."),
        S("kernelbench_l3", "ScalingIntelligence/KernelBench", None, "level_3", "kernel", "kernelbench", {"gpu_exec"},
          contamination_group="kernelbench", prompt_builder=kernel_prompt, adapter=adapt_kernelbench,
          max_new_tokens=16384, verified=True,
          notes="Level 3 (full architectures). Eval only (levels 1-2 may be used for training). Metrics: "
                "correct rate, fast_1, mean speedup."),
        S("kernelbench_l4", "ScalingIntelligence/KernelBench", None, "level_4", "kernel", "kernelbench", {"gpu_exec"},
          contamination_group="kernelbench", prompt_builder=kernel_prompt, adapter=adapt_kernelbench,
          max_new_tokens=16384, verified=False, notes="Level 4 (HF model-level problems). Eval only."),
        S("longbench_v2", "zai-org/LongBench-v2", None, "train", "longmcq", "mcq", set(),
          contamination_group="longbench_v2", prompt_builder=longbench_prompt, adapter=adapt_longbench,
          max_new_tokens=8192, verified=False,
          notes="503 long-context 4-way MCQs (8k-2M words); contexts beyond max_context_chars are middle-truncated."),
        S("mmmu_pro", "MMMU/MMMU_Pro", "standard (10 options)", "test", "vision_mcq", "mcq", {"vision"},
          contamination_group="mmmu", prompt_builder=vision_mcq_prompt, adapter=adapt_mmmu_pro, max_new_tokens=8192,
          verified=True, notes="Needs a vision-capable model + processor; the text-only Generator cannot run it."),
    ):
        register(spec, overwrite=True)


_register_defaults()
