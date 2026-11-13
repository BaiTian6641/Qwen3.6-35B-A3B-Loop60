"""Generator: determinism, cache correctness vs the toy's own greedy loop, depth switching, stops, seeds."""
from __future__ import annotations

import pytest
import torch

from qloop.eval.generate import Generator
from qloop.eval.testing import ByteTokenizer, ScriptedLM, build_toy_generator
from qloop.model.loop import apply_loop
from qloop.model.toy import build_toy

TOY_SPEC = {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24}


def _randomize_adapters(ctl, scale=0.5, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ctl.trainable_parameters():
            p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype) * scale)


def test_greedy_is_deterministic_and_matches_toy_greedy_generate():
    gen = build_toy_generator(seed=0)
    prompts = ["hello world", [{"role": "user", "content": "2+2?"}]]
    a = gen.generate(prompts, max_new_tokens=8, return_logprobs=True)
    b = gen.generate(prompts, max_new_tokens=8, return_logprobs=True)
    assert [o.token_ids for r in a for o in r] == [o.token_ids for r in b for o in r]
    assert [o.logprobs for r in a for o in r] == [o.logprobs for r in b for o in r]
    # identical to the model's own cache-based greedy loop
    ids = torch.tensor([gen.encode_prompt("hello world")])
    ref = gen.model.greedy_generate(ids, 8)[0, ids.shape[1]:].tolist()
    eos = gen.eos_token_ids
    got = a[0][0].token_ids
    n = next((i + 1 for i, t in enumerate(ref) if t in eos), len(ref))
    assert got == ref[:n]
    assert a[1][0].prompt_tokens == len(gen.encode_prompt(prompts[1]))
    for o in a[0]:
        assert len(o.logprobs) == len(o.token_ids) and all(lp <= 0 for lp in o.logprobs)


def test_logprobs_match_full_forward():
    gen = build_toy_generator(seed=1)
    out = gen.generate(["abc"], max_new_tokens=5, return_logprobs=True)[0][0]
    prompt = gen.encode_prompt("abc")
    full = torch.tensor([prompt + out.token_ids])
    with torch.no_grad():
        logits = gen.model(full).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    exp = [lp[len(prompt) - 1 + i, t].item() for i, t in enumerate(out.token_ids)]
    assert out.logprobs == pytest.approx(exp, abs=1e-4)


def test_depth_switching_on_looped_toy():
    base = build_toy(seed=0, vocab_size=ByteTokenizer.vocab_size)
    tok = ByteTokenizer()
    ref = Generator(base, tok).generate(["looped?"], max_new_tokens=8, return_logprobs=True)[0][0]

    model = build_toy(seed=0, vocab_size=ByteTokenizer.vocab_size)
    ctl = apply_loop(model, TOY_SPEC)
    _randomize_adapters(ctl)
    gen = Generator(model, tok, "cpu", ctl)
    assert gen.max_depth == 2 and ctl.active_passes == 2
    d1 = gen.generate(["looped?"], max_new_tokens=8, return_logprobs=True, depth=1)[0][0]
    assert ctl.active_passes == 2  # restored
    d2 = gen.generate(["looped?"], max_new_tokens=8, return_logprobs=True, depth=2)[0][0]
    # depth 1 == the un-looped model exactly (passes > 1 are skipped, own cache slots)
    assert d1.token_ids == ref.token_ids
    assert d1.logprobs == pytest.approx(ref.logprobs, abs=1e-5)
    # depth 2 with non-zero gates computes something different
    assert d2.logprobs != pytest.approx(d1.logprobs, abs=1e-5)
    # default (no depth) uses the controller's current setting
    dflt = gen.generate(["looped?"], max_new_tokens=8, return_logprobs=True)[0][0]
    assert dflt.token_ids == d2.token_ids
    # restored even when generation raises
    ctl.set_active_passes(1)
    with pytest.raises(ValueError):
        gen.generate(["x"], max_new_tokens=2, depth=3)
    assert ctl.active_passes == 1


def test_depth_without_controller_is_a_clear_error():
    gen = build_toy_generator()
    with pytest.raises(AssertionError, match="loop_controller"):
        gen.generate(["x"], max_new_tokens=2, depth=2)


def test_sampling_seeded_per_prompt_and_independent_of_batch():
    gen = build_toy_generator(seed=0)
    a = gen.generate(["p1", "p2"], max_new_tokens=6, temperature=1.0, n=3, seed=[11, 22])
    b = gen.generate(["p2"], max_new_tokens=6, temperature=1.0, n=3, seed=[22])
    assert [o.token_ids for o in a[1]] == [o.token_ids for o in b[0]]
    assert len({tuple(o.token_ids) for o in a[0]}) > 1  # the n samples differ
    c = gen.generate(["p1", "p2"], max_new_tokens=6, temperature=1.0, n=3, seed=[11, 22])
    assert [o.token_ids for r in a for o in r] == [o.token_ids for r in c for o in r]
    # top_p tiny => greedy
    g = gen.generate(["p1"], max_new_tokens=6, temperature=1.0, top_p=1e-6, seed=5)[0][0]
    assert g.token_ids == gen.generate(["p1"], max_new_tokens=6)[0][0].token_ids
    k1 = gen.generate(["p1"], max_new_tokens=6, temperature=1.5, top_k=1, seed=5)[0][0]
    assert k1.token_ids == g.token_ids


def test_stop_strings_eos_and_length():
    gen = Generator(ScriptedLM.from_text("Result: 42 END tail"), ByteTokenizer())
    o = gen.generate(["q"], max_new_tokens=100, stop=["END"])[0][0]
    assert o.text == "Result: 42 " and o.finish_reason == "stop" and o.stop_reason == "END"
    o = gen.generate(["q"], max_new_tokens=100)[0][0]
    assert o.text == "Result: 42 END tail" and o.finish_reason == "stop" and o.stop_reason == 256
    assert o.token_ids[-1] == 256  # eos kept in token ids (needed for RL), dropped from text
    o = gen.generate(["q"], max_new_tokens=4)[0][0]
    assert o.text == "Resu" and o.finish_reason == "length"
    o = gen.generate(["q"], max_new_tokens=0)[0][0]
    assert o.text == "" and o.token_ids == []
    multi = gen.generate(["a", "bb", "ccc"], max_new_tokens=3, n=2)
    assert [len(r) for r in multi] == [2, 2, 2] and all(x.text == "Res" for r in multi for x in r)


def test_max_context_and_input_validation():
    gen = Generator(ScriptedLM.from_text("abcdefgh", eos=False), ByteTokenizer(), max_context=6)
    o = gen.generate(["xyz"], max_new_tokens=50)[0][0]
    assert o.text == "abc" and o.finish_reason == "length"
    with pytest.raises(ValueError, match="max_context"):
        gen.generate(["toolongprompt"], max_new_tokens=2)
    with pytest.raises(TypeError, match="LIST"):
        gen.generate("not a list", max_new_tokens=2)
    with pytest.raises(AssertionError, match="top_p"):
        gen.generate(["x"], top_p=0.0)


def test_padded_batching_requires_attention_mask_support():
    with pytest.raises(ValueError, match="attention_mask"):
        Generator(ScriptedLM([65]), ByteTokenizer(), padded_batching=True, batch_size=4)


class _MaskAwareLM(torch.nn.Module):
    """Reads attention_mask: emits the number of real (unmasked) tokens seen so far, as a byte."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.masks = []

    def forward(self, input_ids, past_key_values=None, use_cache=False, attention_mask=None):
        b, t = input_ids.shape
        self.masks.append(None if attention_mask is None else attention_mask.clone())
        cache = past_key_values if past_key_values is not None else {}
        real = attention_mask.sum(-1) if attention_mask is not None else torch.full((b,), t + cache.get("n", 0))
        cache["n"] = cache.get("n", 0) + t
        logits = torch.zeros(b, t, 258)
        for i in range(b):
            logits[i, :, 48 + int(real[i]) % 10] = 10.0
        return type("O", (), {"logits": logits, "past_key_values": cache})()


def test_left_padded_batching_passes_growing_mask():
    lm = _MaskAwareLM()
    gen = Generator(lm, ByteTokenizer(), padded_batching=True, batch_size=4, eos_token_ids=[256])
    out = gen.generate(["ab", "abcd"], max_new_tokens=3)
    assert out[0][0].text == "234" and out[1][0].text == "456"
    assert sorted(lm.masks[0].tolist()) == [[0, 0, 1, 1], [1, 1, 1, 1]]  # rows sorted by length internally
    assert lm.masks[-1].shape == (2, 6)
    # same results per-sequence
    gen1 = Generator(_MaskAwareLM(), ByteTokenizer(), eos_token_ids=[256])
    assert [r[0].text for r in gen1.generate(["ab", "abcd"], max_new_tokens=3)] == ["234", "456"]


def test_restores_training_mode():
    gen = build_toy_generator()
    gen.model.train()
    gen.generate(["x"], max_new_tokens=2)
    assert gen.model.training
