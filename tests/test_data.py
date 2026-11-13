"""Tokenizer / chat template / loss masks / adapters / validation (offline, CPU)."""
import gzip
import io
import tarfile

import pytest

from qloop.data import (
    DatasetSpec,
    DropCounter,
    RowFormatError,
    ToyTokenizer,
    extract_think,
    load_tokenizer,
    looks_encrypted,
    normalize_rl_task,
    normalize_row,
    tokenize_conversation,
    validate_example,
)
from qloop.data.formats import RowDrop, has_unclosed_think
from qloop.data.registry import REGISTRY, get_spec, list_specs, verify_specs
from qloop.data.tokenizer import check_template_parity, render_conversation

CONV = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "What is 2+3?"},
    {"role": "assistant", "content": "5", "reasoning": "2 plus 3 is 5."},
]


def _trained_text(tok, ids, labels):
    return tok.decode([i for i, lab in zip(ids, labels) if lab != -100], skip_special_tokens=False)


def _masked_text(tok, ids, labels):
    return tok.decode([i for i, lab in zip(ids, labels) if lab == -100], skip_special_tokens=False)


# ----------------------------------------------------------------------------- tokenizer

def test_toy_tokenizer_roundtrip_and_specials():
    tok = ToyTokenizer()
    assert tok.eos_token_id > 255 and tok.pad_token_id > 255 and tok.bos_token_id is None
    assert tok.vocab_size == len(tok) == 256 + 9
    text = "héllo <|im_start|>user\nhi<|im_end|><think>x</think>"
    ids = tok.encode(text)
    assert tok.convert_tokens_to_ids("<|im_start|>") in ids and max(ids) < tok.vocab_size
    assert tok.decode(ids, skip_special_tokens=False) == text
    # specials dropped, <think> tags kept (added-but-not-special, like Qwen)
    assert tok.decode(ids, skip_special_tokens=True) == "héllo user\nhi<think>x</think>"
    assert tok.decode([*ids, -100]) == tok.decode(ids)
    assert isinstance(load_tokenizer("toy"), ToyTokenizer)


def test_chat_template_qwen_style():
    tok = ToyTokenizer()
    text = tok.apply_chat_template(CONV, tokenize=False)
    assert text == (
        "<|im_start|>system\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+3?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n2 plus 3 is 5.\n</think>\n\n5<|im_end|>\n"
    )
    gp = tok.apply_chat_template(CONV[:2], tokenize=False, add_generation_prompt=True)
    assert gp.endswith("<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n")
    nothink = tok.apply_chat_template(CONV[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert nothink.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert tok.apply_chat_template(CONV, tokenize=True) == tok.encode(text)
    ok, rep = check_template_parity(tok, CONV)
    assert ok, rep


def test_loss_mask_assistant_only():
    tok = ToyTokenizer()
    ids, labels = tokenize_conversation(CONV, tok, verify=True)
    assert len(ids) == len(labels)
    assert ids == tok.encode(tok.apply_chat_template(CONV, tokenize=False))  # segment == full tokenisation
    assert _trained_text(tok, ids, labels) == "<think>\n2 plus 3 is 5.\n</think>\n\n5<|im_end|>"
    masked = _masked_text(tok, ids, labels)
    assert "What is 2+3?" in masked and "Be brief." in masked and "<|im_start|>assistant\n" in masked
    # every user-content token is -100
    user_ids = tok.encode("What is 2+3?")
    for k in range(len(ids) - len(user_ids) + 1):
        if ids[k:k + len(user_ids)] == user_ids:
            assert all(lab == -100 for lab in labels[k:k + len(user_ids)])
            break
    else:
        pytest.fail("user content not found in ids")


def test_history_reasoning_dropped_and_multi_turn_masks():
    tok = ToyTokenizer()
    conv = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "reasoning": "R1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2", "reasoning": "R2"},
    ]
    ids, labels = tokenize_conversation(conv, tok)
    trained = _trained_text(tok, ids, labels)
    assert trained == "A1<|im_end|><think>\nR2\n</think>\n\nA2<|im_end|>"
    assert "R1" not in tok.decode(ids)
    ids2, labels2 = tokenize_conversation(conv, tok, keep_history_reasoning=True)
    assert "<think>\nR1\n</think>" in _trained_text(tok, ids2, labels2)


def test_tool_calls_and_responses():
    tok = ToyTokenizer()
    conv = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "", "reasoning": "use ls",
         "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}]},
        {"role": "tool", "content": "a.txt"},
        {"role": "tool", "content": "b.txt"},
        {"role": "assistant", "content": "Two files.", "reasoning": "saw two"},
    ]
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    text = "".join(t for t, _ in render_conversation(conv, tools=tools))
    assert "# Tools" in text and '"name": "bash"' in text
    assert ('<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call><|im_end|>') in text
    assert ("<|im_start|>user\n<tool_response>\na.txt\n</tool_response>\n<tool_response>\nb.txt\n"
            "</tool_response><|im_end|>\n") in text
    ids, labels = tokenize_conversation(conv, tok, tools=tools)
    trained = _trained_text(tok, ids, labels)
    assert "<tool_call>" in trained and "a.txt" not in trained and "list files" not in trained
    # both assistant turns are after the last user query -> both keep reasoning
    assert "use ls" in trained and "saw two" in trained


# ----------------------------------------------------------------------------- think / encryption

@pytest.mark.parametrize("content,reasoning,rest", [
    ("<think>\nR\n</think>\n\nC", "R", "C"),
    ("<thought>\nR\n</thought>C", "R", "C"),
    ("R</think>\n\nC", "R", "C"),
    ("<think>a</think>\n<think>b</think>y", "a\n\nb", "y"),
    ("Sure. I reason inside `<thinking>` tags.", None, "Sure. I reason inside `<thinking>` tags."),
    ("Close with `</think>` when done.", None, "Close with `</think>` when done."),
    ("plain", None, "plain"),
    ("<think>\n\n</think>\n\nC", "", "C"),
])
def test_extract_think(content, reasoning, rest):
    assert extract_think(content) == (reasoning, rest)


def test_unclosed_think_detected():
    assert has_unclosed_think("<think>truncated...")
    assert not has_unclosed_think("<think>x</think>y")
    assert not has_unclosed_think("I will think inside `<thinking>` tags")  # prose mention, not a block
    spec = DatasetSpec("t", "x/y", format="prompt_response", require_reasoning=True)
    with pytest.raises(RowDrop, match="think_unclosed"):
        normalize_row({"prompt": "p", "response": "<think>never closed"}, spec, 3)


def test_looks_encrypted():
    assert looks_encrypted(None) == "reasoning_empty"
    assert looks_encrypted("  \n ") == "reasoning_empty"
    assert looks_encrypted("[redacted]") == "reasoning_redacted"
    assert looks_encrypted("Reasoning is encrypted.") == "reasoning_redacted"
    assert looks_encrypted("gAAAAABn" + "x" * 300) == "reasoning_encrypted"
    assert looks_encrypted("QUJD" * 100) == "reasoning_encrypted"
    long_legit = ("First I consider how the message is encrypted with AES, then decrypt step by step. " * 10)
    assert looks_encrypted(long_legit) is None
    assert looks_encrypted("Let me add 2 and 3 to get 5.") is None


# ----------------------------------------------------------------------------- adapters

def test_sharegpt_openthoughts_row():
    spec = get_spec("openthoughts3")
    row = {"difficulty": 7, "conversations": [{"from": "human", "value": "Solve x"},
                                              {"from": "gpt", "value": "<think> Okay, x=1.</think>\n\nx = 1"}]}
    (ex,) = normalize_row(row, spec, 0)
    assert ex.kind == "chat" and [m["role"] for m in ex.messages] == ["user", "assistant"]
    assert ex.messages[1]["reasoning"] == "Okay, x=1." and ex.messages[1]["content"] == "x = 1"
    assert validate_example(ex, spec) is None


def test_messages_with_thinking_field_and_auto():
    spec = get_spec("opus46_traceinv")
    row = {"messages": [{"role": "user", "content": "Q", "thinking": None},
                        {"role": "assistant", "content": "121.", "thinking": "Let x..."}],
           "conversations": [{"from": "human", "value": "Q"}]}
    (ex,) = normalize_row(row, spec, 0)
    assert ex.messages[1]["reasoning"] == "Let x..." and validate_example(ex, spec) is None
    auto = DatasetSpec("a", "x/y", format="auto")
    (ex2,) = normalize_row(row, auto, 0)
    assert ex2.messages[1]["reasoning"] == "Let x..."


def test_fable51_thought_tags_tool_calls_and_tools_column():
    spec = get_spec("fable51")
    row = {"messages": [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Read the spec"},
        {"role": "assistant", "content": "<thought>\nI'll list files.\n</thought>",
         "tool_calls": [{"id": "c0", "type": "function",
                         "function": {"name": "list_dir", "arguments": '{"path": "."}'}}]},
        {"role": "tool", "tool_call_id": "c0", "name": "list_dir", "content": "a\nb"},
    ], "tools": [{"type": "function", "function": {"name": "list_dir", "description": "ls"}}]}
    (ex,) = normalize_row(row, spec, 5)
    roles = [m["role"] for m in ex.messages]
    assert roles == ["system", "user", "assistant"]  # trailing tool result trimmed
    a = ex.messages[-1]
    assert a["reasoning"] == "I'll list files." and a["content"] == ""
    assert a["tool_calls"][0]["function"]["name"] == "list_dir"
    assert ex.tools and validate_example(ex, spec) is None
    ids, labels = tokenize_conversation(ex.messages, ToyTokenizer(), tools=ex.tools)
    assert '"arguments": {"path": "."}' in _trained_text(ToyTokenizer(), ids, labels)


def test_prompt_response_and_reasoning_only():
    spec = get_spec("opus46_3000x")
    (ex,) = normalize_row({"id": "1", "problem": "P", "thinking": "T", "solution": "S"}, spec, 0)
    assert [m["content"] for m in ex.messages] == ["P", "S"] and ex.messages[1]["reasoning"] == "T"
    helio = get_spec("fable5_distill")
    (ex2,) = normalize_row({"query": "Q", "thinking": "long reasoning"}, helio, 0)
    assert ex2.messages[1]["content"] == "" and validate_example(ex2, helio) is None
    strict = helio.with_overrides(allow_empty_content=False)
    assert validate_example(ex2, strict) == "empty_content"


def test_nemotron_adapter_and_reasoning_off_dropped():
    spec = get_spec("nemotron_post")
    row = {"input": [{"role": "user", "content": "Split the array"}],
           "output": "<think>\nOkay.\n</think>\n\nAnswer", "reasoning": "on", "system_prompt": "detailed thinking on"}
    (ex,) = normalize_row(row, spec, 0)
    assert [m["role"] for m in ex.messages] == ["user", "assistant"]  # control system prompt stripped
    assert ex.messages[-1]["reasoning"] == "Okay." and validate_example(ex, spec) is None
    off = dict(row, output="Answer", reasoning="off", system_prompt="detailed thinking off")
    (ex_off,) = normalize_row(off, spec, 1)
    assert validate_example(ex_off, spec) == "reasoning_empty"


def test_chatml_text_adapter_empty_think_filtered():
    spec = get_spec("agentic_fable5")
    good = ("<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nBuild it<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nplan\n</think>\n\nCalling.\n<tool_call>\n"
            '{"name": "bash", "arguments": {"cmd": "ls"}}\n</tool_call><|im_end|>\n'
            "<|im_start|>user\n<tool_response>\nout\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<think>\ndone\n</think>\n\nFinished.<|im_end|>\n")
    (ex,) = normalize_row({"text": good, "source": "s"}, spec, 0)
    assert [m["role"] for m in ex.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert ex.messages[2]["tool_calls"][0]["function"]["arguments"] == {"cmd": "ls"}
    assert ex.messages[2]["content"] == "Calling." and ex.messages[3]["content"] == "out"
    assert validate_example(ex, spec) is None
    # re-rendering reproduces the original text exactly
    assert "".join(t for t, _ in render_conversation(ex.messages)) == good
    empty = ("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nhello<|im_end|>\n")
    (ex2,) = normalize_row({"text": empty}, spec, 1)
    assert validate_example(ex2, spec) == "reasoning_empty"


def test_encrypted_reasoning_rows_dropped():
    spec = get_spec("toy_cot")
    for r, why in [("", "reasoning_empty"), ("[redacted]", "reasoning_redacted"),
                   ("gAAAAAB" + "Zm9v" * 80, "reasoning_encrypted")]:
        (ex,) = normalize_row({"prompt": "p", "reasoning": r, "response": "a"}, spec, 0)
        assert validate_example(ex, spec) == why


def test_text_and_arc_adapters():
    (t,) = normalize_row({"text": "hello world " * 30}, get_spec("fineweb_edu"), 0)
    assert t.kind == "text" and validate_example(t, get_spec("fineweb_edu")) is None
    (short,) = normalize_row({"text": "hi"}, get_spec("fineweb_edu"), 1)
    assert validate_example(short, get_spec("fineweb_edu")) == "text_too_short"
    task = {"task_id": "abc", "train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}],
            "test": [{"input": [[2, 0]], "output": [[0, 2]]}, {"input": [[3]], "output": [[3]]}]}
    exs = normalize_row(task, get_spec("arc_agi2_train"), 4)
    assert len(exs) == 2 and exs[1].sub_index == 1
    assert exs[0].messages[1]["content"] == "02" and "Test input:\n20" in exs[0].messages[0]["content"]


def test_format_errors_are_contextual():
    spec = get_spec("opus46_3000x")
    with pytest.raises(RowFormatError, match=r"\[opus46_3000x\] row 17: field 'problem'"):
        normalize_row({"thinking": "t", "solution": "s"}, spec, 17)
    with pytest.raises(RowFormatError, match=r"messages\[0\].role.*unknown role 'wizard'"):
        normalize_row({"messages": [{"role": "wizard", "content": "x"}]}, get_spec("fable51"), 2)
    with pytest.raises(RowFormatError, match="could not detect"):
        normalize_row({"foo": 1}, DatasetSpec("z", "x/y", format="auto"), 0)


def test_drop_counter_budget():
    dc = DropCounter(min_rows_for_budget=10, max_format_error_frac=0.5)
    for i in range(4):
        dc.keep("ds")
        dc.drop("ds", "reasoning_empty", i)
    assert dc.summary()["ds"] == {"kept": 4, "reasoning_empty": 4}
    with pytest.raises(RuntimeError, match=r"\[ds\].*failed to parse"):
        for i in range(20):
            dc.drop("ds", "format:messages", 100 + i, "bad")


# ----------------------------------------------------------------------------- registry / RL

def test_registry_covers_design_doc():
    ids = {s.hf_id for s in REGISTRY.values()}
    for hf in [
        "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x", "HelioAI/Fable-5-Distill-Reasoning-462x",
        "lordx64/agentic-distill-fable-5-sft", "nohurry/Opus-4.6-Reasoning-3000x-filtered",
        "Crownelius/Opus-4.6-Reasoning-3300x", "Glint-Research/Opus-4.6-Reasoning-2160x",
        "Jackrong/Claude-opus-4.6-TraceInversion-9000x", "open-thoughts/OpenThoughts3-1.2M",
        "open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K",
        "HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled", "HuggingFaceFW/fineweb_edu_100BT-shuffled",
        "SII-GAIR-NLP/davinci-llm-data", "nvidia/Llama-Nemotron-Post-Training-Dataset",
        "open-thoughts/OpenThoughts-Agent-RL-5K", "PrimeIntellect/SWE-rebench-V2-Filtered-Verified",
        "github:arcprize/ARC-AGI-2", "ScalingIntelligence/KernelBench",
    ]:
        assert hf in ids, hf
    assert get_spec("agentic_fable5").require_reasoning
    assert get_spec("arc_agi2_train").contamination_group == "arc_agi_2" and get_spec("arc_agi2_train").role == "sft"
    kb = get_spec("kernelbench_train")
    assert kb.role == "rl" and kb.splits == ["level_1", "level_2"] and kb.contamination_group == "kernelbench"
    assert set(get_spec("kernelbench_eval").splits) == {"level_3", "level_4"}
    assert {s.name for s in list_specs(role="pretrain")} >= {"finepdfs_mix", "fineweb_edu", "davinci"}
    with pytest.raises(KeyError, match="not in the registry"):
        get_spec("nope")
    with pytest.raises(AssertionError, match="role"):
        DatasetSpec("bad", "x/y", role="dpo")
    rep = verify_specs([get_spec("toy_cot")])
    assert rep[0]["ok"] and rep[0]["exists"]


def test_rl_task_normalizers():
    kb = normalize_rl_task({"code": "class Model: ...", "level": 1, "name": "100_HingeLoss", "problem_id": 100},
                           get_spec("kernelbench_train"), 0)
    assert kb["task_id"] == "L1_100_HingeLoss" and "class Model" in kb["prompt"]
    swe = normalize_rl_task({"instance_id": "a__b-1", "problem_statement": "fix it", "repo": "a/b",
                             "interface": "No new interfaces are introduced."}, get_spec("swe_rebench"), 0)
    assert swe["task_id"] == "a__b-1" and swe["prompt"] == "fix it" and swe["meta"]["repo"] == "a/b"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        data = b"Do the task."
        info = tarfile.TarInfo("task/instruction.md")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    ot = normalize_rl_task({"path": "p-0", "task_binary": gzip.compress(buf.getvalue())}, get_spec("ot_agent_rl"), 0)
    assert ot["prompt"] == "Do the task." and "task/instruction.md" in ot["meta"]["files"]
    with pytest.raises(RowFormatError, match="kernelbench_train"):
        normalize_rl_task({"code": "x"}, get_spec("kernelbench_train"), 3)
