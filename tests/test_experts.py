"""ESFT primitives on the structurally faithful toy: router scan, hot-expert deltas, freezing."""
import copy

import pytest
import torch

from qloop.model.experts import (
    ExpertStats,
    HotExpertDeltas,
    attach_hot_expert_deltas,
    freeze_all_but,
    parse_router_output,
    scan_router_affinity,
)
from qloop.model.loop import apply_loop
from qloop.model.toy import build_toy

TOY_SPEC = {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24}
HOT = {0: [1, 3], 5: [0, 2, 7], 6: list(range(8)), 9: [4, 5], 13: [6]}


def _ids(b=2, t=12, seed=1, vocab=97):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (b, t), generator=g)


def _randomize(params, scale=0.3, seed=5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in params:
            p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype) * scale)


def _merge_into_copy(ref, deltas):
    """Reference: add dense deltas directly into a pristine copy's fused base tensors."""
    with torch.no_grad():
        for layer, experts in deltas.hot.items():
            ex = ref.model.layers[layer].mlp.experts
            for e in experts:
                dgu, ddn = deltas.delta_weights(layer, e)
                ex.gate_up_proj[e] += dgu
                ex.down_proj[e] += ddn


@pytest.mark.parametrize("rank", [0, 2])
def test_zero_init_is_bit_identical(rank):
    model = build_toy()
    ids = _ids()
    ref = model(ids).logits
    deltas = attach_hot_expert_deltas(model, HOT, rank=rank)
    assert any(n.startswith("model.expert_deltas.") for n, _ in model.named_parameters())
    assert torch.equal(model(ids).logits, ref)
    deltas.detach()
    assert torch.equal(model(ids).logits, ref)


@pytest.mark.parametrize("rank", [0, 3])
def test_delta_equals_merged_weights(rank):
    base = build_toy().double()
    ref = copy.deepcopy(base)
    ids = _ids(t=16)
    deltas = HotExpertDeltas(base, HOT, rank=rank).attach()
    _randomize(deltas.parameters())
    out = base(ids).logits
    _merge_into_copy(ref, deltas)
    want = ref(ids).logits
    assert not torch.allclose(out, build_toy().double()(ids).logits)  # deltas really change the output
    torch.testing.assert_close(out, want, rtol=1e-10, atol=1e-10)
    # export path: merge into the base, deltas reset -> same function without the hooks
    deltas.merge_into_base_()
    for n, p in deltas.named_parameters():
        if rank == 0 or n.split(".")[-2].endswith("_B"):
            assert float(p.detach().abs().max()) == 0, n
    deltas.detach()
    torch.testing.assert_close(base(ids).logits, want, rtol=1e-10, atol=1e-10)


def test_delta_correct_under_loop_and_grads_accumulate_over_passes():
    base = build_toy().double()
    ref = copy.deepcopy(base)
    ids = _ids(t=10)
    hot = {5: list(range(8)), 8: [0, 1, 2], 2: [3], 14: [1]}  # core (run twice) + prelude + coda
    deltas = HotExpertDeltas(base, hot).attach()
    _randomize(deltas.parameters(), scale=0.2)
    _merge_into_copy(ref, deltas)
    ctl_a, ctl_b = apply_loop(base, TOY_SPEC), apply_loop(ref, TOY_SPEC)
    _randomize(ctl_a.trainable_parameters(), scale=0.5, seed=11)
    _randomize(ctl_b.trainable_parameters(), scale=0.5, seed=11)
    out_a, out_b = base(ids).logits, ref(ids).logits
    torch.testing.assert_close(out_a, out_b, rtol=1e-10, atol=1e-10)
    # gradient wrt delta == gradient wrt the merged fused slice (both passes contribute)
    out_a.square().mean().backward()
    out_b.square().mean().backward()
    n_checked = 0
    for layer, experts in hot.items():
        ex = ctl_b._orig["layers"][layer].mlp.experts  # physical layer of the wrapped reference
        for e in experts:
            ld = deltas.deltas[f"L{layer}"]
            if ld.gate_up[str(e)].grad is None:  # expert not routed to in this batch
                assert float(ex.gate_up_proj.grad[e].abs().max()) == 0
                continue
            torch.testing.assert_close(ld.gate_up[str(e)].grad, ex.gate_up_proj.grad[e], rtol=1e-9, atol=1e-12)
            torch.testing.assert_close(ld.down[str(e)].grad, ex.down_proj.grad[e], rtol=1e-9, atol=1e-12)
            n_checked += 1
    assert n_checked >= 6


def test_delta_grads_with_activation_checkpointing():
    """Non-reentrant checkpoint recompute re-runs the hooks; grads must match the plain run."""
    grads = []
    for ckpt in (False, True):
        model = build_toy().double()
        deltas = HotExpertDeltas(model, {6: list(range(8)), 1: [0, 1]}).attach()
        _randomize(deltas.parameters(), scale=0.2)
        ctl = apply_loop(model, dict(TOY_SPEC, checkpoint=ckpt))
        _randomize(ctl.trainable_parameters(), scale=0.5, seed=11)
        model.train()
        model(_ids(t=8)).logits.square().mean().backward()
        grads.append({n: p.grad.clone() for n, p in deltas.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and len(grads[0]) >= 8
    for n in grads[0]:
        torch.testing.assert_close(grads[1][n], grads[0][n], rtol=1e-10, atol=1e-13)


def test_routing_is_recomputed_per_call():
    """Same physical MoE called twice with different inputs must use each call's own routing."""
    model = build_toy().double()
    ref = copy.deepcopy(model)  # copy BEFORE attaching (deepcopy would clone the hooks)
    deltas = HotExpertDeltas(model, {3: list(range(8))}).attach()
    _randomize(deltas.parameters())
    mlp = model.model.layers[3].mlp
    x1, x2 = torch.randn(1, 5, 32, dtype=torch.float64), torch.randn(1, 7, 32, dtype=torch.float64)
    y1, y2 = mlp(x1), mlp(x2)
    deltas.detach()
    _merge_into_copy(ref, deltas)
    torch.testing.assert_close(y1, ref.model.layers[3].mlp(x1), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(y2, ref.model.layers[3].mlp(x2), rtol=1e-10, atol=1e-10)


def test_compact_state_roundtrip():
    model = build_toy()
    d1 = HotExpertDeltas(model, HOT, rank=2)
    _randomize(d1.parameters())
    sd = d1.state_dict_compact()
    assert (5, 7) in sd["deltas"] and set(sd["deltas"][(5, 7)]) == {"gate_up_A", "gate_up_B", "down_A", "down_B"}
    d2 = HotExpertDeltas(model, HOT, rank=2)
    assert d2.load_compact(sd) == sorted((layer, e) for layer, es in HOT.items() for e in es)
    for (n1, p1), (n2, p2) in zip(d1.named_parameters(), d2.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)
    with pytest.raises(AssertionError, match="rank mismatch"):
        HotExpertDeltas(model, HOT, rank=0).load_compact(sd)


def test_parse_router_output_hf5_tuple():
    logits = torch.randn(6, 8)
    probs = torch.softmax(logits, -1)
    w, idx = probs.topk(2, -1)
    w = w / w.sum(-1, keepdim=True)
    r = parse_router_output((probs, w, idx), 8, 2, True)
    assert torch.equal(r.top_idx, idx) and torch.equal(r.top_w, w) and torch.allclose(r.probs, probs)
    r2 = parse_router_output(logits, 8, 2, True)
    assert torch.equal(r2.top_idx, idx) and torch.allclose(r2.top_w, w)


def test_affinity_scan_sums_and_selection(tmp_path):
    model = build_toy()
    batches = [_ids(seed=1), _ids(seed=2)]
    st = scan_router_affinity(model, batches)
    assert st.layers() == list(range(16)) and all(st.passes(i) == [0] for i in range(16))
    ntok = 2 * 2 * 12
    for layer in st.layers():
        k = (layer, 0)
        assert st.tokens[k] == ntok
        assert abs(float(st.prob_mass[k].sum()) - ntok) < 1e-4
        assert abs(float(st.gate_mass[k].sum()) - ntok) < 1e-4  # norm_topk_prob (fp32 router)
        assert int(st.counts[k].sum()) == ntok * 2
    hot = st.select_hot(fraction=0.25)
    assert all(len(v) == 2 for v in hot.values()) and set(hot) == set(range(16))
    by_mass = st.select_hot(mass=0.5, by="prob")
    for layer, sel in by_mass.items():
        s = st.score(layer, "prob")
        assert float(s[sel].sum() / s.sum()) >= 0.5 - 1e-9
    top1 = st.select_hot(per_layer_k={3: 1}, by="count")
    assert set(top1) == {3} and int(st.counts[(3, 0)][top1[3][0]]) == int(st.counts[(3, 0)].max())
    summ = st.log_summary()
    assert 0 < summ[0]["entropy_norm"] <= 1
    path = str(tmp_path / "stats.json")
    st.to_json(path)
    st2 = ExpertStats.from_json(path)
    assert st2.select_hot(fraction=0.25) == hot
    assert ExpertStats.from_json(st.to_json()).tokens == st.tokens


def test_affinity_scan_on_loop_records_per_pass():
    model = build_toy()
    ctl = apply_loop(model, TOY_SPEC)
    st = scan_router_affinity(model, [_ids()])
    assert st.passes(0) == [0] and st.passes(15) == [0]
    assert st.passes(4) == [1, 2] and st.passes(11) == [1, 2]
    for r in (1, 2):
        assert int(st.counts[(6, r)].sum()) == 2 * 12 * 2
    # gates are zero at init but pass-2 routers see the bridged input -> stats exist per pass
    union = st.select_hot(per_layer_k=2, per_pass="union")
    p1 = st.select_hot(per_layer_k=2, per_pass="pass1")
    assert all(set(p1[i]) <= set(union[i]) for i in p1)
    assert all(len(p1[i]) == 2 for i in p1)
    assert "pass12_hot_jaccard" in st.summary()[6]
    ctl.unwrap()


def test_freeze_all_but_report_and_guards():
    model = build_toy()
    ctl = apply_loop(model, TOY_SPEC)
    deltas = HotExpertDeltas(model, HOT)
    rep = freeze_all_but(model, [r"loop_adapters\.", r"mlp\.gate\.weight$"],
                         extra_params=deltas.named_parameters(prefix="expert_deltas"),
                         allowed_groups={"loop_adapters", "router", "expert_deltas"})
    assert rep.groups["expert_deltas"]["trainable"] == sum(p.numel() for p in deltas.parameters())
    assert rep.groups["router"]["trainable"] == 16 * 8 * 32
    assert rep.groups["expert_base"]["trainable"] == 0 and rep.groups["attention"]["trainable"] == 0
    assert rep.groups["norms"]["frozen"] > 0 and rep.groups["deltanet"]["frozen"] > 0
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in ctl.trainable_parameters()} | {
        id(layer.mlp.gate.weight) for layer in ctl._orig["layers"]}
    assert all(p.requires_grad for p in deltas.parameters())
    with pytest.raises(AssertionError, match="forbidden"):
        freeze_all_but(model, [r"experts\.gate_up_proj"])
    with pytest.raises(AssertionError, match="matched no parameter"):
        freeze_all_but(model, [r"no_such_param"])
    with pytest.raises(AssertionError, match="unexpected trainable"):
        freeze_all_but(model, [r"shared_expert\."], allowed_groups={"router"})
