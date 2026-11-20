"""Regression tests for the `s2_joint_bptt` step-0 NaN (`int32` Triton overflow at T=131072),
`quick_eval` / D-LSA `_sparse_chunk` host-sync optimization, BPTT checkpoint recompute skip,
and `GRPOTrainer` aux-loss isolation.
"""
from __future__ import annotations

import ast
import os
import pathlib

import pytest
import torch

from qloop.config import load_config
from qloop.model.hssd import HSSDAdapter, HSSDConfig
from qloop.model.loop import apply_loop
from qloop.model.lsa import LSAConfig, _sparse_chunk, install_lsa, lsa_attention
from qloop.model.toy import ToyConfig, ToyForCausalLM
from qloop.train import bptt as bptt_mod
from qloop.train.bptt import SegmentedKVBPTTRunner, _gather_prior_field
from qloop.train.rl import GRPOTrainer, _Seq

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ==============================================================================
# 1. HSSD Triton int64 pointer-indexing & chunked `_FusedAdapterProjFn` guards
# ==============================================================================


def test_hssd_triton_kernels_use_int64_indexing_and_chunked_adapter_proj():
    """Every strided Triton kernel in `hssd.py` must use `tl.int64` indexing to prevent signed
    32-bit overflow (e.g. `t * 20480 >= 2^31` at `t = 104,858` when `T = 131,072`), and
    `_FusedAdapterProjFn` must chunk `all_up` along `T` instead of allocating a 10.74 GiB tensor.
    """
    src = (ROOT / "src" / "qloop" / "model" / "hssd.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()

    strided_kernels: list[str] = []
    missing_int64: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            decs = [ast.unparse(d) for d in node.decorator_list]
            if any("triton.jit" in d for d in decs):
                body_src = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                if "stride_" in body_src:
                    strided_kernels.append(node.name)
                    if "tl.int64" not in body_src:
                        missing_int64.append(node.name)

    assert len(strided_kernels) >= 20
    assert not missing_int64, f"Strided Triton kernels missing tl.int64 indexing: {missing_int64}"

    # Verify `_FusedAdapterProjFn` chunks `all_up` along T in both forward and backward
    fused_cls = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "_FusedAdapterProjFn"
    )
    methods = {
        n.name: "\n".join(lines[n.lineno - 1 : n.end_lineno])
        for n in fused_cls.body
        if isinstance(n, ast.FunctionDef)
    }
    for m_name in ("forward", "backward"):
        assert m_name in methods
        assert "t_chunk" in methods[m_name], f"_FusedAdapterProjFn.{m_name} must chunk along T"
        assert "range(0, T, t_chunk)" in methods[m_name]


def test_hssd_adapter_project_inputs_cpu_finite_and_grad():
    """Verify `HSSDAdapter.project_inputs` forward + backward produces finite outputs & grads."""
    torch.manual_seed(7)
    cfg = HSSDConfig(rank=16, chunk_size=8, window_chunks=2, num_pages=4, k_pages=2, theta_low_rank=True)
    ad = HSSDAdapter(hidden_size=32, num_v_heads=4, head_k_dim=8, head_v_dim=8, cfg=cfg, layer_id=0)
    b, t = 2, 24
    x = torch.randn(b, t, 32, requires_grad=True)
    q = torch.randn(b, t, 2, 8, requires_grad=True)
    k = torch.randn(b, t, 2, 8, requires_grad=True)
    v = torch.randn(b, t, 4, 8, requires_grad=True)
    g_base = (-torch.rand(b, t, 4)).requires_grad_(True)
    beta_base = torch.rand(b, t, 4).requires_grad_(True)

    out = ad.project_inputs(x, q, k, v, g_base, beta_base, use_qk_l2norm=True)
    for name, tensor in out.items():
        assert torch.isfinite(tensor).all(), f"non-finite {name}"
    loss = sum(val.float().mean() for val in out.values())
    loss.backward()
    for p in (x, q, k, v, g_base, beta_base, *ad.parameters()):
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for T=131072 Triton overflow test")
def test_fused_adapter_proj_cuda_t131072_past_int32_boundary():
    """Live CUDA regression test: `T=131072` crosses `t=104858` (`104858 * 20480 > 2^31 - 1`)."""
    from qloop.model.hssd import _FusedAdapterProjFn

    dev = torch.device("cuda")
    hv, hq, hk, dk, dv, rank, t_len = 32, 16, 16, 128, 128, 64, 131_072
    torch.manual_seed(42)
    q_r = (torch.randn(1, t_len, hq, dk, device=dev, dtype=torch.bfloat16) * 0.1).requires_grad_(True)
    k_r = (torch.randn(1, t_len, hk, dk, device=dev, dtype=torch.bfloat16) * 0.1).requires_grad_(True)
    v_r = (torch.randn(1, t_len, hv, dv, device=dev, dtype=torch.bfloat16) * 0.1).requires_grad_(True)
    h_f = (torch.randn(1, t_len, rank, device=dev, dtype=torch.float32) * 0.1).requires_grad_(True)
    b_r = (torch.rand(1, t_len, hv, device=dev, dtype=torch.bfloat16) * 0.5 + 0.1).requires_grad_(True)
    w_er = (torch.randn(hv * dk, rank, device=dev, dtype=torch.float32) * 0.02).requires_grad_(True)
    w_wr = (torch.randn(hv * dv, rank, device=dev, dtype=torch.float32) * 0.02).requires_grad_(True)
    w_sq = (torch.randn(hv * dk, rank, device=dev, dtype=torch.float32) * 0.02).requires_grad_(True)
    w_sk = (torch.randn(hv * dk, rank, device=dev, dtype=torch.float32) * 0.02).requires_grad_(True)
    w_sv = (torch.randn(hv * dv, rank, device=dev, dtype=torch.float32) * 0.02).requires_grad_(True)
    qnw = torch.ones(dk, device=dev, dtype=torch.float32, requires_grad=True)
    knw = torch.ones(dk, device=dev, dtype=torch.float32, requires_grad=True)

    outs = _FusedAdapterProjFn.apply(
        q_r, k_r, v_r, h_f, b_r, w_er, w_wr, w_sq, w_sk, w_sv, qnw, knw, hv, 8.0, True, False
    )
    assert all(torch.isfinite(o).all() for o in outs)

    with torch.no_grad():
        ref_tail = _FusedAdapterProjFn.apply(
            q_r[:, -1024:].detach(),
            k_r[:, -1024:].detach(),
            v_r[:, -1024:].detach(),
            h_f[:, -1024:].detach(),
            b_r[:, -1024:].detach(),
            w_er.detach(),
            w_wr.detach(),
            w_sq.detach(),
            w_sk.detach(),
            w_sv.detach(),
            qnw.detach(),
            knw.detach(),
            hv,
            8.0,
            True,
            False,
        )
    for o, r in zip(outs, ref_tail):
        torch.testing.assert_close(o[:, -1024:].float(), r.float(), rtol=0.0, atol=1e-5)

    sum(o.float().mean() for o in outs).backward()
    for t_in in (q_r, k_r, v_r, h_f, b_r, w_er, w_wr, w_sq, w_sk, w_sv, qnw, knw):
        assert t_in.grad is not None and torch.isfinite(t_in.grad).all()


# ==============================================================================
# 2. D-LSA `_sparse_chunk` `q_bounds` & `record_stats` exact numerical parity
# ==============================================================================


def test_lsa_sparse_chunk_qbounds_and_record_stats_exact_parity():
    """Passing `q_bounds` and `record_stats=False` on intermediate chunks must match
    `q_bounds=None, record_stats=True` bit-for-bit across both empty-history and sparse-history chunks.
    """
    torch.manual_seed(11)
    cfg = ToyConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
    )
    model = ToyForCausalLM(cfg).double().eval()
    lcfg = LSAConfig(
        mode="sparse",
        compact_active_slots=True,
        max_selected_keys=24,
        index_dim=8,
        index_on="nope",
        pool=2,
        block_size=16,
        avg_bucket_keys=4,
        max_bucket_keys=8,
        num_select=4,
        window=16,
        sinks=4,
        query_chunk=8,
    )
    mgr = install_lsa(model, lcfg)
    attn = model.model.layers[3].self_attn
    ix = mgr.indexers[3]

    b, h, g, t_len, dh = 2, 4, 2, 96, 8
    q = torch.randn(b, h, t_len, dh, dtype=torch.float64)
    k = torch.randn(b, g, t_len, dh, dtype=torch.float64)
    v = torch.randn(b, g, t_len, dh, dtype=torch.float64)
    sc = dh ** -0.5

    with torch.no_grad():
        index = ix.build(k)
        assert index is not None
        qg = q.view(b, g, h // g, t_len, dh)
        qz_all = ix.query_latent(q)
        qpos_all = torch.arange(t_len)

        ref_chunks = []
        fast_chunks = []
        for c0 in range(0, t_len, lcfg.query_chunk):
            c1 = min(t_len, c0 + lcfg.query_chunk)
            is_last = c1 >= t_len
            ref_chunks.append(
                _sparse_chunk(
                    qg[:, :, :, c0:c1],
                    qz_all[:, :, c0:c1],
                    qpos_all[c0:c1],
                    k,
                    v,
                    index,
                    ix,
                    sc,
                    cur_pass=1,
                    q_bounds=None,
                    record_stats=True,
                )
            )
            fast_chunks.append(
                _sparse_chunk(
                    qg[:, :, :, c0:c1],
                    qz_all[:, :, c0:c1],
                    qpos_all[c0:c1],
                    k,
                    v,
                    index,
                    ix,
                    sc,
                    cur_pass=1,
                    q_bounds=(c0, c1 - 1),
                    record_stats=is_last,
                )
            )
        ref_out = torch.cat(ref_chunks, dim=3).reshape(b, h, t_len, dh)
        fast_out = torch.cat(fast_chunks, dim=3).reshape(b, h, t_len, dh)
        torch.testing.assert_close(fast_out, ref_out, rtol=1e-12, atol=1e-12)

        # Full `lsa_attention` call also populates stats and `last_selected_buckets` cleanly
        out_attn, _ = lsa_attention(attn, q, k, v, scaling=sc)
        torch.testing.assert_close(out_attn, ref_out.transpose(1, 2), rtol=1e-12, atol=1e-12)
        assert "keys_per_query" in ix.last_stats and ix.last_stats["keys_per_query"] > 0.0
        assert 1 in ix.last_selected_buckets and ix.last_selected_buckets[1].shape[2] == lcfg.query_chunk


# ==============================================================================
# 3. `SegmentedKVBPTTRunner` (`bptt.py`) gather + checkpoint recompute skip
# ==============================================================================


def test_bptt_gather_prior_field_matches_cat_gather_unequal_segments():
    torch.manual_seed(3)
    segs = [torch.randn(2, 11, 16), torch.randn(2, 16, 16), torch.randn(2, 7, 16)]
    total_t = sum(s.shape[1] for s in segs)
    idx = torch.randint(0, total_t, (2, 13))
    got = _gather_prior_field(segs, idx)
    ref = torch.cat(segs, dim=1).gather(1, idx.unsqueeze(-1).expand(-1, -1, 16))
    assert torch.equal(got, ref)


def test_bptt_checkpoint_recompute_reuses_replay_indices_and_skips_discarded_kl(monkeypatch):
    """During checkpointed BPTT backward recomputation, `_select_replay_indices` and
    `_chunked_replay_indexer_kl` must NOT be re-run (exact gradient parity with un-checkpointed run).
    """
    def _build_pair(checkpoint: bool):
        torch.manual_seed(17)
        cfg = ToyConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=12,
            full_attention_interval=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            partial_rotary_factor=0.5,
        )
        m = ToyForCausalLM(cfg).double().train()
        install_lsa(
            m,
            LSAConfig(
                mode="sparse",
                distill_weight=0.5,
                num_query_passes=2,
                index_dim=8,
                index_on="nope",
                pool=2,
                block_size=16,
                avg_bucket_keys=4,
                max_bucket_keys=8,
                num_select=2,
                window=16,
                sinks=4,
                query_chunk=16,
            ),
        )
        ctl = apply_loop(
            m,
            {"prelude_end": 4, "core_end": 8, "num_passes": 2, "target_virtual_layers": 16, "checkpoint": checkpoint},
        )
        with torch.no_grad():
            ctl.adapters.gates.fill_(0.2)
        for name, p in m.named_parameters():
            p.requires_grad_(("self_attn" in name) or ("lsa_indexer" in name) or ("loop_adapters" in name))
        return m

    ids = torch.randint(0, 64, (1, 64), generator=torch.Generator().manual_seed(29))
    labels = ids.clone()
    labels[:, :16] = -100

    # Reference: checkpoint=False
    m_ref = _build_pair(checkpoint=False)
    torch.manual_seed(100)
    res_ref = SegmentedKVBPTTRunner(m_ref, n_segments=4, replay_k=6, offload_u_to_cpu=True).run_step(
        ids, labels=labels, backward=True
    )

    # Checkpointed: verify zero backward calls to `_select_replay_indices` and `_chunked_replay_indexer_kl`
    m_ckpt = _build_pair(checkpoint=True)
    bwd_select_calls = [0]
    bwd_kl_calls = [0]
    fwd_kl_calls = [0]

    orig_select = SegmentedKVBPTTRunner._select_replay_indices
    orig_kl = bptt_mod._chunked_replay_indexer_kl

    def _wrapped_select(self, *args, **kwargs):
        if bptt_mod._in_backward():
            bwd_select_calls[0] += 1
        return orig_select(self, *args, **kwargs)

    def _wrapped_kl(*args, **kwargs):
        if bptt_mod._in_backward():
            bwd_kl_calls[0] += 1
        else:
            fwd_kl_calls[0] += 1
        return orig_kl(*args, **kwargs)

    monkeypatch.setattr(SegmentedKVBPTTRunner, "_select_replay_indices", _wrapped_select)
    monkeypatch.setattr(bptt_mod, "_chunked_replay_indexer_kl", _wrapped_kl)

    torch.manual_seed(100)
    res_ckpt = SegmentedKVBPTTRunner(m_ckpt, n_segments=4, replay_k=6, offload_u_to_cpu=True).run_step(
        ids, labels=labels, backward=True
    )

    assert bwd_select_calls[0] == 0, f"expected 0 backward _select_replay_indices calls, got {bwd_select_calls[0]}"
    assert bwd_kl_calls[0] == 0, f"expected 0 backward _chunked_replay_indexer_kl calls, got {bwd_kl_calls[0]}"
    assert fwd_kl_calls[0] > 0
    assert abs(res_ckpt.total_loss - res_ref.total_loss) < 1e-9

    n_grad_checked = 0
    for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_ckpt.named_parameters()):
        assert n1 == n2
        if p1.requires_grad:
            assert (p1.grad is None) == (p2.grad is None), f"grad presence mismatch on {n1}"
            if p1.grad is not None and p2.grad is not None:
                torch.testing.assert_close(p2.grad, p1.grad, rtol=1e-7, atol=1e-9)
                n_grad_checked += 1
    assert n_grad_checked > 0


# ==============================================================================
# 4. `GRPOTrainer.update()` clears stale no-grad HSSD/LSA aux losses
# ==============================================================================


def test_grpo_update_clears_stale_nograd_hssd_aux(tmp_path):
    """Verify `GRPOTrainer.update()` drops stale no-grad HSSD aux from rollout / ref_logprobs / entropy."""
    cfg = {
        "seed": 0,
        "stage": "rl-hssd-aux-test",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}},
        "loop": {"enabled": True, "prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24},
        "hssd": {
            "enabled": True,
            "chunk_size": 8,
            "window_chunks": 2,
            "num_pages": 4,
            "k_pages": 2,
            "theta_low_rank": True,
            "segment_chunks": 2,
            "checkpoint_segments": False,
        },
        "train": {
            "steps": 1,
            "ckpt_every": 10,
            "train_hssd": True,
            "trainable": [r"mlp\.gate\.weight$"],
            "aux_weights": {"hssd_budget": 0.5},
            "optim": {"lr": 1e-3, "fused": False, "apollo": {"min_dim": 16}},
        },
        "rl": {
            "tasks": [{"source": "toy_arith", "n": 8, "verifier": "regex", "pattern": ".*"}],
            "group_size": 2,
            "prompts_per_step": 1,
            "max_new_tokens": 4,
            "temperature": 1.0,
            "micro_batch_seqs": 2,
            "kl_coef": 0.05,
            "ppo_epochs": 2,
        },
    }
    tr = GRPOTrainer(cfg, str(tmp_path / "rl_hssd"))
    assert tr.stack.hssd is not None

    # Inject a huge fake stale no-grad aux value simulating rollout decode leftovers
    for ad in tr.stack.hssd.adapters.values():
        ad._pending_aux.append(torch.tensor(9999.0))

    seqs = [
        _Seq([1, 2, 3, 4, 5, 6, 7, 8], 3, 1.0, "pg"),
        _Seq([1, 2, 3, 9, 10, 11, 12, 13], 3, -1.0, "pg"),
    ]
    stats = tr.update(seqs)
    assert stats["loss/hssd_budget"] < 10.0, f"stale rollout aux leaked into update: {stats['loss/hssd_budget']}"
    assert all(len(ad._pending_aux) == 0 for ad in tr.stack.hssd.adapters.values())
    tr.pool.shutdown(wait=False)


# ==============================================================================
# 5. `s2_joint_bptt.yaml` config safety & eval budget invariants
# ==============================================================================


def test_s2_joint_bptt_config_eval_budget_and_safety_invariants():
    s2 = load_config(os.path.join(str(ROOT), "configs", "stages", "s2_joint_bptt.yaml"))
    s4 = load_config(os.path.join(str(ROOT), "configs", "stages", "s4_rl.yaml"))

    assert s2["gdn2"]["enabled"] is False
    assert s2["hssd"]["enabled"] is True
    assert s2["lsa"]["enabled"] is True
    assert s2["lsa"]["mode"] == "sparse"
    assert s2["train"]["optim"]["fused"] is False
    assert float(s2["train"].get("max_step0_loss", 10.0)) == 10.0
    assert s2["eval_quick"]["n_batches"] == 2
    assert list(s2["eval_quick"]["depths"]) == [1, 2, 3]
    assert dict(s2["hssd"]) == dict(s4["hssd"])
    assert dict(s2["lsa"]) == dict(s4["lsa"])
