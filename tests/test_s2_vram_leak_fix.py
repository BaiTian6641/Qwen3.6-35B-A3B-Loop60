"""Regression tests for the S2 `s2_joint_bptt` +19.4 GiB/step VRAM leak & OOM fix.

Covers all three root causes proven in live Blackwell G4 (`loop64`) & CPU forensics:
  1. PyTorch `saved_tensors_hooks(lambda x: x, lambda x: x)` C++ self-cycle around `F.normalize`
     (`LinalgVectorNormBackward0` saving its own output Tensor with `grad_fn` intact, pinning
     upstream `_CheckpointFrame`s forever when `.backward()` is skipped) -> fixed with
     `lambda x: x.detach()`.
  2. `LoopContext` C++-bridged reference cycle (`ctx.{anchor,start,kv_stash,pkv_stash}` <->
     `VirtualLayer` `checkpoint(use_reentrant=False)` `_CheckpointFrame` closures) -> fixed with
     `LoopContext.release_graph()` / `LoopController.release_graph()` after every BPTT segment
     and in `finally:` + avoiding unused `d=1` KV stashing.
  3. Zero-label (`all -100`) BPTT segments (e.g. `respect_doc_boundaries=True` trailing padding
     on `step 58 mb 0` / `step 65 mb 0`, or >32K SFT user prompts) running full checkpointed
     autograd forward passes and then skipping `.backward()` -> fixed by skipping trailing
     all-`-100` segments, advancing interior all-`-100` segments under `torch.no_grad()` while
     preserving exact carried KV/HSSD/replay-record state, and clearing dead `LSAIndexer._cache`
     entries in `finally:`.
"""
from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch.utils.checkpoint import _CheckpointFrame, checkpoint

from qloop.model.hssd import install_hssd
from qloop.model.loop import LoopController, apply_loop
from qloop.model.lsa import LSAManager, install_lsa
from qloop.model.toy import ToyConfig, ToyForCausalLM
from qloop.train.bptt import BPTTConfig, SegmentedKVBPTTRunner, _detach_nested
from qloop.train.optim import FusedStepOptimizer, build_param_groups, defer_micro_step


def _count_frames_and_live_tensors() -> tuple[int, int]:
    gc.collect()
    gc.collect()
    frames = sum(1 for o in gc.get_objects() if isinstance(o, _CheckpointFrame))
    live_t = sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor) and o.grad_fn is not None)
    return frames, live_t


def _build_toy_v32_stack(
    *,
    device: str = "cpu",
    seed: int = 17,
) -> tuple[ToyForCausalLM, LSAManager, LoopController, ToyConfig]:
    torch.manual_seed(seed)
    cfg = ToyConfig()
    model = ToyForCausalLM(cfg).to(device)
    install_hssd(
        model,
        {
            "chunk_size": 8,
            "window_chunks": 2,
            "num_pages": 4,
            "k_pages": 2,
            "segment_chunks": 2,
            "checkpoint_segments": True,
        },
    )
    lsa = install_lsa(
        model,
        {
            "mode": "sparse",
            "block_size": 16,
            "pool": 4,
            "avg_bucket_keys": 8,
            "max_bucket_keys": 8,
            "num_select": 4,
            "window": 16,
            "sinks": 4,
            "num_query_passes": 3,
        },
    )
    loop = apply_loop(
        model,
        {
            "prelude_end": 4,
            "core_end": 12,
            "num_passes": 3,
            "target_virtual_layers": None,
            "checkpoint": True,
            "kv_share": True,
            "per_pass_router": True,
            "attn_adapter_rank": 4,
        },
    )
    model.train()
    return model, lsa, loop, cfg


def test_saved_tensors_hooks_detach_breaks_normalize_self_cycle() -> None:
    """`saved_tensors_hooks(lambda t: t.detach(), ...)` breaks `F.normalize` C++ self-cycles with bit-exact grads."""
    f0, t0 = _count_frames_and_live_tensors()

    def _run_no_backward() -> None:
        lin = torch.nn.Linear(32, 32)
        w = torch.nn.Parameter(torch.randn(32, 8))
        aux: list[torch.Tensor] = []

        def fn(x: torch.Tensor) -> torch.Tensor:
            h = lin(x)
            with torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach(), lambda t: t):
                z = torch.nn.functional.normalize(h @ w, dim=-1)
                aux.append((z * z.detach()).sum())
            return h + 1.0

        x = torch.randn(2, 32, requires_grad=True)
        y = checkpoint(fn, x, use_reentrant=False)
        del x, y
        aux.clear()

    _run_no_backward()
    f1, t1 = _count_frames_and_live_tensors()
    assert f1 == f0
    assert t1 == t0

    # Bit-exact gradient parity against plain autograd (no hooks)
    torch.manual_seed(5)
    lin_ref = torch.nn.Linear(32, 32)
    w_ref = torch.nn.Parameter(torch.randn(32, 8))
    x_ref = torch.randn(2, 32, requires_grad=True)
    z_ref = torch.nn.functional.normalize(lin_ref(x_ref) @ w_ref, dim=-1)
    (z_ref * torch.linspace(0.1, 1.0, z_ref.numel()).view_as(z_ref)).sum().backward()

    torch.manual_seed(5)
    lin_h = torch.nn.Linear(32, 32)
    w_h = torch.nn.Parameter(torch.randn(32, 8))
    x_h = torch.randn(2, 32, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach(), lambda t: t):
        z_h = torch.nn.functional.normalize(lin_h(x_h) @ w_h, dim=-1)
        (z_h * torch.linspace(0.1, 1.0, z_h.numel()).view_as(z_h)).sum().backward()

    assert torch.equal(x_ref.grad, x_h.grad)  # type: ignore[arg-type]
    assert torch.equal(w_ref.grad, w_h.grad)  # type: ignore[arg-type]
    assert torch.equal(lin_ref.weight.grad, lin_h.weight.grad)  # type: ignore[arg-type]


def test_loop_context_release_graph_and_d1_no_kv_stash() -> None:
    """`LoopController.release_graph()` breaks C++-bridged checkpoint cycles & `d=1` skips KV stash."""
    f0, t0 = _count_frames_and_live_tensors()

    def _run() -> weakref.ReferenceType:
        cfg = ToyConfig()
        model = ToyForCausalLM(cfg)
        loop = apply_loop(
            model,
            {
                "prelude_end": 4,
                "core_end": 12,
                "num_passes": 3,
                "target_virtual_layers": None,
                "checkpoint": True,
                "kv_share": True,
                "per_pass_router": True,
                "attn_adapter_rank": 4,
            },
        )
        model.train()
        ids = torch.randint(0, cfg.vocab_size, (1, 32))

        # Depth 1 should not stash unused Pass-1 KV tensors
        loop.set_active_passes(1)
        out1 = model(ids, use_cache=False)
        assert len(loop.ctx.kv_stash) == 0
        assert len(loop.ctx.pkv_stash) == 0
        del out1
        loop.release_graph()

        # Depth 3 without backward: release_graph() breaks the cycle while keeping detached routing/stats
        loop.set_active_passes(3)
        out3 = model(ids, use_cache=False)
        assert loop.ctx.anchor is not None
        assert len(loop.ctx.kv_stash) > 0
        del out3
        loop.release_graph()
        assert loop.ctx.anchor is None
        assert len(loop.ctx.start) == 0
        assert len(loop.ctx.kv_stash) == 0
        assert len(loop.ctx.pkv_stash) == 0
        stats = loop.stats()
        assert "loop/reroute_rate" in stats
        assert "loop/unique_experts_core" in stats
        ctx_ref = weakref.ref(loop.ctx)
        del loop, model
        return ctx_ref

    cref = _run()
    f1, t1 = _count_frames_and_live_tensors()
    assert cref() is None
    assert f1 == f0
    assert t1 == t0


def test_detach_nested_handles_slots_and_custom_detach() -> None:
    """`_detach_nested` detaches tensors inside `__slots__` containers and `.detach()` state objects."""

    class _SlottedLayer:
        __slots__ = ("keys", "meta", "values")

        def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
            self.keys = k
            self.values = v
            self.meta = "ok"

    class _CustomDetachState:
        def __init__(self, t: torch.Tensor) -> None:
            self.t = t
            self.called = False

        def detach(self) -> _CustomDetachState:
            out = _CustomDetachState(self.t.detach())
            out.called = True
            return out

    x = torch.randn(2, 4, requires_grad=True)
    y = x * 2.0
    z = x + 3.0
    slot_obj = _SlottedLayer(y, z)
    custom = _CustomDetachState(y + z)
    cache = {"layers": [slot_obj], "recurrent": [custom]}

    _detach_nested(cache)
    assert slot_obj.keys.grad_fn is None and not slot_obj.keys.requires_grad
    assert slot_obj.values.grad_fn is None and not slot_obj.values.requires_grad
    assert isinstance(cache["recurrent"][0], _CustomDetachState)
    assert cache["recurrent"][0].called is True
    assert cache["recurrent"][0].t.grad_fn is None


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("mask_kind", ["seg1", "seg0", "all"])
def test_bptt_zero_label_segments_zero_leak(depth: int, mask_kind: str) -> None:
    """Zero-label (`-100`) BPTT segments across `d=1/2/3` leak 0 checkpoint frames and 0 live tensors."""
    f0, t0 = _count_frames_and_live_tensors()

    def _run() -> weakref.ReferenceType:
        model, lsa, loop, cfg = _build_toy_v32_stack(seed=10 + depth)
        loop.set_active_passes(depth)
        runner = SegmentedKVBPTTRunner(
            model,
            cfg=BPTTConfig(seg_len=32, n_segments=2, replay_k=8, offload_prior_hidden=False, replay_indexer_weight=0.2),
        )
        ids = torch.randint(0, cfg.vocab_size, (1, 64), generator=torch.Generator().manual_seed(42))
        lbl = ids.clone()
        if mask_kind == "seg1":
            lbl[:, 32:] = -100
        elif mask_kind == "seg0":
            lbl[:, :33] = -100
        else:
            lbl[:, :] = -100

        res = runner.run_step(ids, labels=lbl, backward=True)
        if mask_kind == "seg1":
            assert res.segment_losses[0] > 0.0
            assert res.segment_losses[1] == 0.0
            assert res.replayed_tokens_per_seg == [0, 0]
        elif mask_kind == "seg0":
            assert res.segment_losses[0] == 0.0
            assert res.segment_losses[1] > 0.0
            assert res.replayed_tokens_per_seg[1] == 8
            # Verify seg0 under no_grad still carried replay records so k_proj/v_proj/lsa_ix.wk get grads!
            has_wk_grad = any(
                ix.wk.grad is not None and ix.wk.grad.abs().sum().item() > 0
                for ix in lsa.indexers.values()
            )
            assert has_wk_grad
        else:
            assert res.total_loss == 0.0
            assert res.segment_losses == [0.0, 0.0]

        assert loop.ctx.anchor is None
        assert len(loop.ctx.pkv_stash) == 0
        assert all(len(ix._cache) == 0 for ix in lsa.indexers.values())
        ctx_ref = weakref.ref(loop.ctx)
        del runner, loop, lsa, model
        return ctx_ref

    cref = _run()
    f1, t1 = _count_frames_and_live_tensors()
    assert cref() is None
    assert f1 == f0
    assert t1 == t0


def test_bptt_custom_loss_none_backward_false_and_exception_zero_leak() -> None:
    """Even when `can_pre_head=False` skips backward or raises mid-step, 0 checkpoint frames leak."""
    f0, t0 = _count_frames_and_live_tensors()

    def _run_edge(mode: str) -> None:
        model, lsa, loop, cfg = _build_toy_v32_stack(seed=31)
        loop.set_active_passes(3)
        runner = SegmentedKVBPTTRunner(
            model,
            cfg=BPTTConfig(seg_len=32, n_segments=2, replay_k=8, offload_prior_hidden=False, replay_indexer_weight=0.2),
        )
        ids = torch.randint(0, cfg.vocab_size, (1, 64), generator=torch.Generator().manual_seed(7))
        if mode == "custom_none":
            res = runner.run_step(ids, labels=ids.clone(), loss_fn=lambda *a: None, backward=True)
            assert res.total_loss == 0.0
        elif mode == "backward_false":
            res = runner.run_step(ids, labels=ids.clone(), backward=False)
            assert res.total_loss > 0.0
        else:
            def _boom(s_idx: int, logits: torch.Tensor, *_a: object) -> torch.Tensor:
                if s_idx == 1:
                    raise RuntimeError("injected segment failure")
                return logits.float().pow(2).mean()

            with pytest.raises(RuntimeError, match="injected segment failure"):
                runner.run_step(ids, labels=ids.clone(), loss_fn=_boom, backward=True)

        assert loop.ctx.anchor is None
        assert len(loop.ctx.pkv_stash) == 0
        assert all(len(ix._cache) == 0 for ix in lsa.indexers.values())
        del runner, loop, lsa, model

    for edge_mode in ("custom_none", "backward_false", "exception"):
        _run_edge(edge_mode)
        f1, t1 = _count_frames_and_live_tensors()
        assert f1 == f0, f"leaked frames on {edge_mode}"
        assert t1 == t0, f"leaked live tensors on {edge_mode}"


def test_bptt_interior_zero_label_seg0_no_grad_matches_grad_forward_parity() -> None:
    """Running an all-`-100` `seg0` under `no_grad()` produces identical `seg1` loss and replay grads."""
    ids = torch.randint(0, ToyConfig().vocab_size, (1, 64), generator=torch.Generator().manual_seed(9))
    lbl = ids.clone()
    lbl[:, :33] = -100

    # Path A: production pre-head path (seg0 runs under torch.no_grad(), seg1 with grad + replay)
    m_a, lsa_a, loop_a, _ = _build_toy_v32_stack(seed=55)
    for ix in lsa_a.indexers.values():
        ix.cfg.distill_weight = 0.0
    loop_a.set_active_passes(3)
    runner_a = SegmentedKVBPTTRunner(
        m_a,
        cfg=BPTTConfig(seg_len=32, n_segments=2, replay_k=8, offload_prior_hidden=False, replay_indexer_weight=0.2),
    )
    res_a = runner_a.run_step(ids, labels=lbl, backward=True, lsa_aux_weight=0.0)
    grads_a = {n: p.grad.detach().clone() for n, p in m_a.named_parameters() if p.grad is not None}

    # Path B: custom `loss_fn` forcing `can_pre_head=False` (seg0 runs with grad enabled, returns None)
    m_b, lsa_b, loop_b, _ = _build_toy_v32_stack(seed=55)
    for ix in lsa_b.indexers.values():
        ix.cfg.distill_weight = 0.0
    loop_b.set_active_passes(3)
    runner_b = SegmentedKVBPTTRunner(
        m_b,
        cfg=BPTTConfig(seg_len=32, n_segments=2, replay_k=8, offload_prior_hidden=False, replay_indexer_weight=0.2),
    )

    def _seg1_only_ce(
        s_idx: int,
        logits: torch.Tensor,
        _s_ids: torch.Tensor,
        s_tgt: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if s_idx == 0 or s_tgt is None:
            return None
        aligned = logits[:, : s_tgt.shape[1]]
        return torch.nn.functional.cross_entropy(
            aligned.reshape(-1, aligned.shape[-1]).float(), s_tgt.reshape(-1), ignore_index=-100
        )

    res_b = runner_b.run_step(ids, labels=lbl, loss_fn=_seg1_only_ce, backward=True, lsa_aux_weight=0.0)
    grads_b = {n: p.grad.detach().clone() for n, p in m_b.named_parameters() if p.grad is not None}

    assert abs(res_a.total_loss - res_b.total_loss) < 1e-4
    assert set(grads_a.keys()) == set(grads_b.keys())
    for k, grad_a in grads_a.items():
        torch.testing.assert_close(grad_a, grads_b[k], atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU VRAM flatness verification")
def test_cuda_bptt_zero_label_segments_flat_vram() -> None:
    """Multi-step CUDA BPTT across normal, trailing `-100`, interior `-100`, and `d=1/2/3` has 0 VRAM drift."""
    model, lsa, loop, cfg = _build_toy_v32_stack(device="cuda", seed=77)
    groups = build_param_groups(list(model.named_parameters()), {"lr": 1e-4, "apollo": {"enabled": False}})
    opt = FusedStepOptimizer(groups, fused=False, max_grad_norm=1.0, grad_accum=1)
    runner = SegmentedKVBPTTRunner(
        model,
        cfg=BPTTConfig(seg_len=32, n_segments=4, replay_k=8, offload_prior_hidden="auto", replay_indexer_weight=0.2),
    )
    ids = torch.randint(0, cfg.vocab_size, (1, 128), device="cuda")

    # Step 0 warmup to materialize optimizer states
    loop.set_active_passes(3)
    opt.zero_grad(set_to_none=True)
    opt.set_boundary(True)
    with defer_micro_step(opt):
        runner.run_step(ids, labels=ids.clone(), backward=True)
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    gc.collect()
    base_alloc = torch.cuda.memory_allocated()
    base_frames = sum(1 for o in gc.get_objects() if isinstance(o, _CheckpointFrame))

    schedule = [
        (3, "trailing_seg3"),
        (3, "normal"),
        (2, "trailing_seg2_3"),
        (3, "interior_seg0"),
        (1, "trailing_seg3"),
        (3, "custom_none"),
        (3, "normal"),
    ]
    for depth, kind in schedule:
        loop.set_active_passes(depth)
        lbl = ids.clone()
        loss_fn = None
        if kind == "trailing_seg3":
            lbl[:, 96:] = -100
        elif kind == "trailing_seg2_3":
            lbl[:, 64:] = -100
        elif kind == "interior_seg0":
            lbl[:, :33] = -100
        elif kind == "custom_none":
            loss_fn = lambda *a: None  # noqa: E731

        opt.set_boundary(True)
        with defer_micro_step(opt):
            runner.run_step(ids, labels=lbl, loss_fn=loss_fn, backward=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        cur_alloc = torch.cuda.memory_allocated()
        cur_frames = sum(1 for o in gc.get_objects() if isinstance(o, _CheckpointFrame))
        assert cur_frames == base_frames, (
            f"leaked checkpoint frames on ({depth}, {kind}): {cur_frames} vs {base_frames}"
        )
        assert (cur_alloc - base_alloc) <= 1024 * 1024, (
            f"CUDA alloc grew by {(cur_alloc - base_alloc) / 2**20:.2f} MiB on ({depth}, {kind})"
        )
    assert all(len(ix._cache) == 0 for ix in lsa.indexers.values())
