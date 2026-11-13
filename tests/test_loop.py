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
    assert "loop/reroute_rate" in st and "loop/unique_experts_core" in st
    assert "loop/effective_expert_mult" in st and st["loop/effective_expert_mult"] >= 1.0


def test_per_pass_router_repel_expands_active_experts_across_passes():
    # Toy has num_experts=8, top_k=2 -> up to 2*3 = 6 unique experts across 3 passes
    spec3 = {"prelude_end": 4, "core_end": 12, "num_passes": 3, "target_virtual_layers": 32,
             "per_pass_router": True, "router_adapter_rank": 0}
    model = build_toy()
    ctl = apply_loop(model, spec3)
    with torch.no_grad():
        ctl.adapters.gates.fill_(0.5)
    ctl.set_router_repel(100.0)
    ids = _ids(b=2, t=6)
    ctl.set_active_passes(1)
    model(ids)
    u1 = ctl.stats()["loop/unique_experts_core"]
    ctl.set_active_passes(2)
    model(ids)
    st2 = ctl.stats()
    ctl.set_active_passes(3)
    model(ids)
    st3 = ctl.stats()
    assert u1 == pytest.approx(2.0)
    assert st2["loop/reroute_rate"] == pytest.approx(1.0)
    assert st2["loop/expert_jaccard_p12"] == pytest.approx(0.0)
    assert st2["loop/unique_experts_core"] == pytest.approx(4.0)
    assert st3["loop/unique_experts_core"] == pytest.approx(6.0)
    assert st3["loop/effective_expert_mult"] > st2["loop/effective_expert_mult"] > 1.0


def test_per_pass_router_hook_ordering_matches_merged_hot_expert_deltas():
    """HotExpertDeltas attaches BEFORE apply_loop in build_stack; prepend=True must keep routing aligned."""
    from qloop.model.experts import attach_hot_expert_deltas

    base = build_toy().double()
    hot = {i: [0, 2, 5, 7] for i in range(4, 12)}
    m1 = copy.deepcopy(base)
    d1 = attach_hot_expert_deltas(m1, hot, rank=0)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for p in d1.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.25)
    ctl1 = apply_loop(m1, dict(TOY_SPEC, per_pass_router=True, router_adapter_rank=8))
    _randomize_adapters(ctl1, scale=1.5, seed=19)
    ctl1.set_router_repel(3.0)

    # Reference: merge deltas into physical expert base weights, then apply the exact same loop adapters
    m2 = copy.deepcopy(base)
    d2 = attach_hot_expert_deltas(m2, hot, rank=0)
    d2.load_state_dict(d1.state_dict())
    d2.merge_into_base_()
    d2.detach()
    ctl2 = apply_loop(m2, dict(TOY_SPEC, per_pass_router=True, router_adapter_rank=8))
    ctl2.adapters.load_state_dict(ctl1.adapters.state_dict())

    ids = _ids(b=2, t=7)
    out1 = m1(ids).logits
    out2 = m2(ids).logits
    torch.testing.assert_close(out1, out2, rtol=1e-10, atol=1e-11)


def test_per_pass_router_supports_hf_v5_topk_router_tuple_format():
    """HF>=5 Qwen3_5MoeTopKRouter returns (router_probs, top_scores, top_indices) directly from gate()."""
    class TupleTopKRouter(torch.nn.Module):
        def __init__(self, linear, top_k, norm_topk):
            super().__init__()
            self.weight = linear.weight
            self.top_k = top_k
            self.norm_topk = norm_topk

        def forward(self, x):
            flat = x.reshape(-1, x.shape[-1])
            probs = torch.softmax(flat @ self.weight.t(), dim=-1)
            top_w, top_idx = probs.topk(self.top_k, dim=-1)
            if self.norm_topk:
                top_w = top_w / top_w.sum(-1, keepdim=True)
            return probs, top_w.to(x.dtype), top_idx

    class TupleMoeWrapper(torch.nn.Module):
        def __init__(self, orig_mlp):
            super().__init__()
            self.gate = TupleTopKRouter(orig_mlp.gate, orig_mlp.top_k, orig_mlp.norm_topk)
            self.experts = orig_mlp.experts
            self.shared_expert = orig_mlp.shared_expert
            self.shared_expert_gate = orig_mlp.shared_expert_gate

        def forward(self, x):
            b, t, h = x.shape
            flat = x.reshape(-1, h)
            _probs, top_w, top_idx = self.gate(flat)
            out = self.experts(flat, top_idx, top_w)
            out = out + torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
            return out.view(b, t, h)

    model = build_toy().double()
    for lyr in model.model.layers:
        lyr.mlp = TupleMoeWrapper(lyr.mlp)
    ids = _ids(b=2, t=6)
    ref = model(ids).logits.detach()
    ctl = apply_loop(model, dict(TOY_SPEC, per_pass_router=True, router_diversity_weight=0.1))
    # Step-0 bit-exact identity even with TupleTopKRouter
    assert torch.equal(model(ids).logits, ref)
    # Diversity aux loss drives gradients into pass_router even at step 0
    aux = ctl.aux_loss()
    assert aux is not None
    aux.backward()
    assert ctl.adapters.pass_router.up.grad.abs().sum() > 0
    assert ctl.adapters.pass_router.repel.grad.abs().sum() > 0
    # Strong repulsion forces disjoint experts in pass 2
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        ctl.adapters.gates.fill_(0.4)
    ctl.set_router_repel(50.0)
    out = model(ids).logits
    assert not torch.allclose(out, ref)
    assert ctl.stats()["loop/reroute_rate"] == pytest.approx(1.0)


def test_router_diversity_checkpoint_gradient_parity():
    torch.manual_seed(0)
    base = build_toy()
    grads = []
    for ckpt in (False, True):
        model = copy.deepcopy(base)
        ctl = apply_loop(model, dict(TOY_SPEC, checkpoint=ckpt, per_pass_router=True, router_diversity_weight=0.2))
        _randomize_adapters(ctl, scale=0.4, seed=7)
        model.train()
        loss = model(_ids(), labels=_ids()).loss + ctl.aux_loss()
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for n in grads[0]:
        torch.testing.assert_close(grads[0][n], grads[1][n], rtol=1e-4, atol=1e-6, msg=n)


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
