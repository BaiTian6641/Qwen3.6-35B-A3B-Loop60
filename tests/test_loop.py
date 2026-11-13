"""Loop retrofit invariants, verified on the structurally faithful toy model."""
import copy

import pytest
import torch

from qloop.model.introspect import inspect_model
from qloop.model.loop import LoopSpec, apply_loop, build_schedule
from qloop.model.mhc import BirkhoffMix, LoopBridge, sinkhorn_project
from qloop.model.toy import build_toy

TOY_SPEC = {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24}


def _ids(b=2, t=10, seed=1, vocab=97):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (b, t), generator=g)


def _randomize_adapters(ctl, scale=0.5, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ctl.trainable_parameters():
            p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype) * scale)


def test_qwen36_default_schedule_is_64_layers():
    types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(40)]
    spec = LoopSpec()
    assert spec.validate(types) == []
    sched = build_schedule(spec, types)
    assert len(sched) == 64
    assert [s.phys for s in sched[8:32]] == list(range(8, 32))
    assert [s.phys for s in sched[32:56]] == list(range(8, 32))
    assert sched[8].entry and sched[31].exit and sched[32].entry and sched[55].exit
    # KV sharing: pass-2 full-attn layers read pass-1 slots; DeltaNet keeps its own slot
    s = sched[35]
    assert s.phys == 11 and s.layer_type == "full_attention" and s.cache_idx == 11
    assert sched[32].cache_idx == 32
    with pytest.raises(ValueError, match="virtual layers"):
        LoopSpec(core_end=28).validate(types)


def test_identity_at_init_is_bit_exact():
    model = build_toy()
    ids = _ids()
    ref = model(ids).logits
    ctl = apply_loop(model, dict(TOY_SPEC, mor=True, mor_target_depth=1.5))
    assert model.config.num_hidden_layers == 24
    out = model(ids).logits
    assert torch.equal(ref, out)
    assert ctl.aux_loss() is not None


def test_gates_receive_gradient_at_init_and_change_output():
    model = build_toy()
    ids = _ids()
    ref = model(ids).logits.detach()
    ctl = apply_loop(model, TOY_SPEC)
    model.train()
    loss = model(ids, labels=ids).loss
    loss.backward()
    g = ctl.adapters.gates.grad
    assert g is not None and g.abs().sum() > 0
    _randomize_adapters(ctl)
    assert not torch.allclose(model(ids).logits, ref)


def test_depth_one_equals_original_even_with_trained_adapters():
    model = build_toy()
    ids = _ids()
    ref = model(ids).logits
    ctl = apply_loop(model, TOY_SPEC)
    _randomize_adapters(ctl)
    ctl.set_active_passes(1)
    assert torch.equal(model(ids).logits, ref)


@pytest.mark.parametrize("mor", [False, True])
@pytest.mark.parametrize("kv_share", [True, False])
def test_incremental_decode_matches_full_forward(mor, kv_share):
    # fp64 so that any mismatch is a *semantic* cache bug, not accumulation-order noise
    model = build_toy().double()
    ctl = apply_loop(model, dict(TOY_SPEC, mor=mor, kv_share=kv_share))
    _randomize_adapters(ctl)
    if mor:  # make the router actually send some tokens out early
        with torch.no_grad():
            ctl.adapters.mor_router.weight.normal_(0, 2.0)
            ctl.adapters.mor_router.bias.zero_()
    ids = _ids(b=2, t=9)
    full = model(ids).logits
    if mor:
        keep = ctl.ctx.keep[2]
        assert 0 < keep.sum() < keep.numel(), "test needs a mix of exited/continuing tokens"
    out = model(ids[:, :5], use_cache=True)
    cache, steps = out.past_key_values, [out.logits]
    for t in range(5, 9):
        out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
        steps.append(out.logits)
    inc = torch.cat(steps, dim=1)
    torch.testing.assert_close(inc, full, rtol=1e-9, atol=1e-10)
    # per-pass DeltaNet slots are distinct objects; pass-2 full-attn slots stay empty with KV sharing
    assert cache.recurrent_states[4] is not None and cache.recurrent_states[12] is not None
    assert not torch.equal(cache.recurrent_states[4], cache.recurrent_states[12])
    assert (cache.key_cache[15] is None) == kv_share
    assert cache.get_seq_length() == 9


def test_activation_checkpointing_matches_plain_gradients():
    torch.manual_seed(0)
    base = build_toy()
    grads = []
    for ckpt in (False, True):
        model = copy.deepcopy(base)
        ctl = apply_loop(model, dict(TOY_SPEC, checkpoint=ckpt, mor=True))
        _randomize_adapters(ctl)
        model.train()
        model(_ids(), labels=_ids()).loss.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and len(grads[0]) > 10
    for n in grads[0]:
        torch.testing.assert_close(grads[0][n], grads[1][n], rtol=1e-4, atol=1e-6, msg=n)


def test_kv_share_reuses_pass1_projections():
    model = build_toy()
    ctl = apply_loop(model, TOY_SPEC)
    _randomize_adapters(ctl)
    seen = {}
    attn = inspect_model(model).text_model.layers[7].inner.self_attn  # phys 7, core full-attn

    def rec(_m, _i, out):
        seen.setdefault("k", []).append(out.detach().clone())

    h = attn.k_proj.register_forward_hook(rec)
    model(_ids())
    h.remove()
    # hook order: ours (registered first) replaces pass-2 output -> our recorder sees identical tensors
    assert len(seen["k"]) == 2 and torch.equal(seen["k"][0], seen["k"][1])


def test_unwrap_restores_original_model():
    model = build_toy()
    ids = _ids()
    ref = model(ids).logits
    ctl = apply_loop(model, TOY_SPEC)
    _randomize_adapters(ctl)
    ctl.unwrap()
    assert model.config.num_hidden_layers == 16 and len(model.model.layers) == 16
    assert model.model.layers[7].self_attn.layer_idx == 7
    assert not hasattr(model.model, "loop_adapters")
    assert torch.equal(model(ids).logits, ref)


def test_routing_overlap_and_stats():
    model = build_toy()
    ctl = apply_loop(model, dict(TOY_SPEC, collect_routing=True))
    _randomize_adapters(ctl, scale=2.0)
    model(_ids())
    ov = ctl.routing_overlap()
    assert set(ov) == set(range(4, 12)) and all(0.0 <= v <= 1.0 for v in ov.values())
    st = ctl.stats()
    assert "loop/gate_abs_mean_p2" in st and st["loop/active_passes"] == 2.0


def test_mhc_primitives():
    m = sinkhorn_project(torch.randn(4, 4), iters=50)
    torch.testing.assert_close(m.sum(-1), torch.ones(4), atol=1e-4, rtol=0)
    torch.testing.assert_close(m.sum(-2), torch.ones(4), atol=1e-4, rtol=0)
    mix = BirkhoffMix(3)
    s = torch.randn(3, 2, 5)
    assert torch.equal(mix(s), s)
    br = LoopBridge(8, 1)
    h, a = torch.randn(2, 3, 8) * 10, torch.randn(2, 3, 8)
    x = br(h, a, 0)
    assert x.shape == h.shape and x.norm() < h.norm()  # convex mix never amplifies
