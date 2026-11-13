"""D-LSA correctness on the toy hybrid model."""
import copy

import pytest
import torch

from qloop.model.loop import apply_loop
from qloop.model.lsa import LSAConfig, apply_yarn, install_lsa, members_table, ward_cluster
from qloop.model.toy import ToyConfig, build_toy

SMALL = {"index_dim": 4, "block_size": 8, "pool": 2, "avg_bucket_keys": 4, "max_bucket_keys": 8,
         "window": 8, "sinks": 2, "num_select": 2, "query_chunk": 5, "distill_chunk": 4}


def _ids(t=40, b=1, seed=5):
    return torch.randint(0, 97, (b, t), generator=torch.Generator().manual_seed(seed))


def test_ward_cluster_separates_and_respects_cap():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(6, 3, generator=g) * 0.01
    b = torch.randn(6, 3, generator=g) * 0.01 + 5.0
    x = torch.cat([a, b])[torch.tensor([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])].unsqueeze(0)
    lab = ward_cluster(x, n_clusters=2, max_size=6)[0]
    assert lab.unique().numel() == 2
    assert len(set(lab[0::2].tolist())) == 1 and len(set(lab[1::2].tolist())) == 1
    capped = ward_cluster(x, n_clusters=1, max_size=4)[0]
    assert torch.bincount(capped).max() <= 4
    mem = members_table(lab.unsqueeze(0), 6)[0]
    for rep in lab.unique().tolist():
        assert sorted(m for m in mem[rep].tolist() if m >= 0) == sorted((lab == rep).nonzero().flatten().tolist())


def test_sparse_with_all_buckets_equals_dense():
    model = build_toy().double()
    ids = _ids(t=37)
    ref = model(ids).logits
    install_lsa(model, dict(SMALL, mode="sparse", num_select=10_000))
    out = model(ids).logits
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("search", ["flat", "tree"])
def test_sparse_decode_matches_sparse_full_forward(search):
    model = build_toy().double()
    ids = _ids(t=42, b=2)
    dense = model(ids).logits
    mgr = install_lsa(model, dict(SMALL, mode="sparse", search=search, tree_group=2, tree_beam=1))
    full = model(ids).logits
    assert not torch.allclose(full, dense), "selection must actually drop keys in this config"
    mgr.clear_cache()
    out = model(ids[:, :3], use_cache=True)
    cache, steps = out.past_key_values, [out.logits]
    for t in range(3, ids.shape[1]):
        out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
        steps.append(out.logits)
    torch.testing.assert_close(torch.cat(steps, 1), full, rtol=1e-9, atol=1e-10)
    assert mgr.stats(), "stats should be recorded"


def test_distill_mode_keeps_dense_output_and_trains_only_indexer():
    model = build_toy()
    ids = _ids(t=48)
    ref = model(ids).logits.detach()
    for p in model.parameters():
        p.requires_grad_(False)
    mgr = install_lsa(model, dict(SMALL, mode="distill", distill_query_frac=1.0))
    params = mgr.trainable_parameters()
    opt = torch.optim.Adam(params, lr=0.05)
    first = last = None
    for _ in range(15):
        out = model(ids).logits
        loss = mgr.aux_loss()
        assert loss is not None and torch.isfinite(loss)
        first = float(loss.detach()) if first is None else first
        last = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.testing.assert_close(out.detach(), ref)  # base attention untouched in distill mode
    assert last < first * 0.9, f"KL should decrease: {first} -> {last}"
    assert all(p.grad is not None for p in params)


def test_lsa_composes_with_loop_and_kv_share():
    model = build_toy().double()
    ids = _ids(t=40)
    ref = model(ids).logits
    mgr = install_lsa(model, dict(SMALL, mode="sparse", num_select=10_000))
    ctl = apply_loop(model, {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24})
    torch.testing.assert_close(model(ids).logits, ref, rtol=1e-10, atol=1e-11)  # gate=0 + all buckets
    # pass-2 of a core full-attn layer reuses pass-1's bucket index (one cache entry, no re-clustering)
    ix = mgr.indexers[7]
    calls = {"n": 0}
    orig = ix._cluster_blocks

    def counting(z):
        calls["n"] += 1
        return orig(z)

    ix._cluster_blocks = counting
    mgr.clear_cache()
    with torch.no_grad():
        ctl.adapters.gates.fill_(0.3)
    model(ids)
    assert calls["n"] == 1
    mgr.uninstall()
    ctl.unwrap()
    assert torch.equal(model(ids).logits, ref)


def test_yarn_and_config_validation():
    cfg = ToyConfig()
    cfg.max_position_embeddings = 262144
    rope = apply_yarn(cfg, 1_048_576)
    assert rope["factor"] == 4.0 and cfg.max_position_embeddings == 1_048_576
    with pytest.raises(ValueError):
        LSAConfig(block_size=10, pool=4).validate()
    with pytest.raises(ValueError, match="unknown"):
        LSAConfig.from_dict({"bogus": 1})


def test_lsa_real_scale_shapes_smoke():
    """Qwen3.6 attention geometry (16 q heads / 2 kv heads / head_dim 256 / rotary 64) on a short sequence."""
    from qloop.model.lsa import _DENSE_FN, LSAIndexer, lsa_attention
    from qloop.model.toy import dense_causal_attention

    cfg = LSAConfig(mode="sparse", rotary_dim=64, block_size=64, window=64, sinks=4, num_select=4, query_chunk=64)
    ix = LSAIndexer(16, 2, 256, cfg, layer_id=3)
    mod = torch.nn.Module()
    mod.lsa_indexer = ix
    saved = _DENSE_FN[0]
    _DENSE_FN[0] = dense_causal_attention
    try:
        q, k, v = torch.randn(1, 16, 200, 256), torch.randn(1, 2, 200, 256), torch.randn(1, 2, 200, 256)
        out, _ = lsa_attention(mod, q, k, v)
    finally:
        _DENSE_FN[0] = saved
    assert out.shape == (1, 200, 16, 256) and torch.isfinite(out).all()
    assert ix.last_stats["keys_per_query"] < 200


def test_install_rejects_non_full_attention_layers():
    model = build_toy()
    with pytest.raises(ValueError, match="not full-attention"):
        install_lsa(copy.deepcopy(model), dict(SMALL, layers=[0]))
