"""Chunked losses vs naive reference; fused (in-backward) optimizer vs classic step; APOLLO fallback."""
import copy
import math

import pytest
import torch
from torch import nn

from qloop.train.loss import cce_available, lm_loss, token_entropy, token_logprobs
from qloop.train.optim import (
    AdamWLite,
    ApolloAdamWFallback,
    FusedStepOptimizer,
    WarmupCosine,
    apollo_torch_available,
    build_param_groups,
)

# ============================================================================ losses


def _lm_data(b=2, t=9, h=6, v=23, seed=0):
    g = torch.Generator().manual_seed(seed)
    hid = torch.randn(b, t, h, generator=g, dtype=torch.float64)
    w = torch.randn(v, h, generator=g, dtype=torch.float64) * 0.7
    y = torch.randint(0, v, (b, t), generator=g)
    y[0, 3] = -100
    y[1, :2] = -100
    return hid, w, y


@pytest.mark.parametrize("softcap,z_loss,shift", [(None, 0.0, True), (3.0, 1e-2, True), (None, 1e-3, False)])
def test_chunked_ce_matches_naive_values_and_grads(softcap, z_loss, shift):
    hid, w, y = _lm_data()
    res = {}
    for impl in ("naive", "chunked"):
        h_ = hid.clone().requires_grad_(True)
        w_ = w.clone().requires_grad_(True)
        loss, n = lm_loss(h_, w_, y, impl=impl, chunk_size=4, shift=shift, softcap=softcap, z_loss=z_loss)
        loss.backward()
        res[impl] = (loss.detach(), int(n), h_.grad, w_.grad)
    (l0, n0, gh0, gw0), (l1, n1, gh1, gw1) = res["naive"], res["chunked"]
    yy = y[:, 1:] if shift else y
    assert n0 == n1 == int((yy != -100).sum())
    torch.testing.assert_close(l1, l0, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gh1, gh0, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(gw1, gw0, rtol=1e-10, atol=1e-12)


def test_chunked_ce_frozen_weight_and_2d_input_and_auto():
    hid, w, y = _lm_data()
    h2 = hid.reshape(-1, hid.shape[-1]).clone().requires_grad_(True)
    loss, n = lm_loss(h2, w, y.reshape(-1), impl="auto", chunk_size=5, shift=False)  # CPU -> chunked
    loss.backward()
    ref = torch.nn.functional.cross_entropy(h2.detach() @ w.t(), y.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(loss.detach(), ref, rtol=1e-12, atol=1e-12)
    assert h2.grad is not None and int(n) == int((y != -100).sum())
    # all labels ignored -> zero loss, zero grads, no NaN
    h3 = hid.clone().requires_grad_(True)
    l3, n3 = lm_loss(h3, w, torch.full_like(y, -100), impl="chunked")
    l3.backward()
    assert float(l3.detach()) == 0.0 and int(n3) == 0 and float(h3.grad.abs().max()) == 0.0


def test_cce_missing_is_explicit():
    if cce_available():
        pytest.skip("cut_cross_entropy installed")
    hid, w, y = _lm_data()
    with pytest.raises(ImportError, match="cut"):
        lm_loss(hid, w, y, impl="cce")


@pytest.mark.parametrize("temperature,softcap", [(1.0, None), (0.7, 2.5)])
def test_token_logprobs_values_and_grads(temperature, softcap):
    hid, w, y = _lm_data(seed=3)
    y = y.clamp_min(0)
    h_ = hid.clone().requires_grad_(True)
    w_ = w.clone().requires_grad_(True)
    lp = token_logprobs(h_, w_, y, chunk_size=5, temperature=temperature, softcap=softcap)
    assert lp.shape == y.shape
    coef = torch.randn(lp.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    (lp * coef).sum().backward()
    h2 = hid.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)
    z = h2 @ w2.t()
    if softcap is not None:
        z = softcap * torch.tanh(z / softcap)
    ref = torch.log_softmax(z / temperature, -1).gather(-1, y[..., None]).squeeze(-1)
    (ref * coef).sum().backward()
    torch.testing.assert_close(lp.detach(), ref.detach(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(h_.grad, h2.grad, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(w_.grad, w2.grad, rtol=1e-10, atol=1e-12)
    # independent check of the custom backward
    hs = hid[:1, :4].clone().requires_grad_(True)
    ws = w.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda a, b: token_logprobs(a, b, y[:1, :4], chunk_size=3, temperature=temperature, softcap=softcap),
        (hs, ws))
    ent = token_entropy(hid, w, chunk_size=4, temperature=temperature, softcap=softcap)
    zz = hid @ w.t()
    if softcap is not None:
        zz = softcap * torch.tanh(zz / softcap)
    pr = torch.softmax(zz / temperature, -1)
    torch.testing.assert_close(ent, -(pr * pr.log()).sum(-1), rtol=1e-10, atol=1e-12)


# ============================================================================ optimizer


class TwiceNet(nn.Module):
    """`core` is applied twice per forward (like looped core layers); mixes APOLLO/AdamW params."""

    def __init__(self, d=16, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.inp = nn.Linear(8, d)
        self.core = nn.Linear(d, d, bias=False)
        self.norm_w = nn.Parameter(torch.ones(d))
        self.loop_adapters = nn.Parameter(torch.full((d,), 0.5))
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        h = h + self.loop_adapters * torch.tanh(self.core(h))
        h = h + torch.tanh(self.core(h * self.norm_w))
        return self.head(h).squeeze(-1)


OPT_CFG = {"lr": 2e-2, "weight_decay": 0.01, "apollo": {"rank": 4, "scale_type": "channel", "scale": 1.0,
                                                      "min_dim": 8, "update_proj_gap": 3}}


def _make_opt(model, fused, **kw):
    groups = build_param_groups(model.named_parameters(), OPT_CFG)
    kw.setdefault("schedule", WarmupCosine(total_steps=6, warmup_steps=2, min_ratio=0.2))
    return FusedStepOptimizer(groups, fused=fused, impl="builtin", **kw)


def _batches(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(5, 8, generator=g), torch.randn(5, generator=g)) for _ in range(n)]


def _train(model, opt, steps, accum, data):
    stats = []
    it = iter(data)
    for _ in range(steps):
        for j in range(accum):
            x, y = next(it)
            opt.set_boundary(j == accum - 1)
            ((model(x) - y).pow(2).mean() / accum).backward()
        stats.append(opt.step())
        opt.zero_grad()
    return stats


def test_build_param_groups_classification():
    model = TwiceNet()
    router = ("model.layers.0.mlp.gate.weight", nn.Parameter(torch.randn(16, 16)))
    frozen = ("frozen.weight", nn.Parameter(torch.randn(16, 16), requires_grad=False))
    groups = build_param_groups([router, frozen, *model.named_parameters()], OPT_CFG)
    by_name = {n: g for g in groups for n in g["names"]}
    assert "frozen.weight" not in by_name
    assert by_name["core.weight"]["kind"] == "apollo" and by_name["core.weight"]["rank"] == 4
    assert by_name["core.weight"]["scale_type"] == "channel" and by_name["core.weight"]["proj"] == "random"
    assert by_name["inp.weight"]["kind"] == "apollo"           # [16, 8]: min dim 8 >= min_dim 8
    assert by_name[router[0]]["kind"] == "adamw" and by_name[router[0]]["lr"] == pytest.approx(2e-2 * 0.5)
    assert by_name["norm_w"]["kind"] == "adamw" and by_name["norm_w"]["weight_decay"] == 0.0
    assert by_name["head.weight"]["kind"] == "adamw"           # [1, 16] tiny matrix
    assert by_name["loop_adapters"]["lr"] == pytest.approx(2e-2 * 10)
    assert by_name["core.weight"]["weight_decay"] == 0.01


def test_fused_equals_nonfused_with_accumulation_and_shared_param():
    data = _batches(16)
    m_f, m_n = TwiceNet(), TwiceNet()
    m_f.load_state_dict(m_n.state_dict())
    opt_f = _make_opt(m_f, True, max_grad_norm_per_param=0.5)
    opt_n = _make_opt(m_n, False, max_grad_norm_per_param=0.5)
    kinds = {type(e.opt).__name__ for e in opt_f.entries.values()}
    assert kinds == {"ApolloAdamWFallback", "AdamWLite"}
    calls = {"n": 0}
    m_f.core.weight.register_post_accumulate_grad_hook(lambda p: calls.__setitem__("n", calls["n"] + 1))
    s_f = _train(m_f, opt_f, steps=4, accum=2, data=data)
    s_n = _train(m_n, opt_n, steps=4, accum=2, data=data)
    assert calls["n"] == 8  # once per backward although `core` is used twice per forward
    for (n1, p1), (_, p2) in zip(m_f.named_parameters(), m_n.named_parameters()):
        assert torch.equal(p1, p2), n1  # bit-identical: same per-param math, same grads
        assert p1.grad is None
    for a, b in zip(s_f, s_n):
        assert a["optim/global_grad_norm"] == pytest.approx(b["optim/global_grad_norm"], rel=1e-6)
    assert opt_f.step_count == 4 and any(s["optim/clipped_params"] > 0 for s in s_f)
    ref = TwiceNet()
    assert not torch.allclose(ref.core.weight, m_f.core.weight)


def test_double_step_guard():
    m = TwiceNet()
    opt = _make_opt(m, True)
    x, y = _batches(1)[0]
    (m(x) - y).pow(2).mean().backward()
    assert opt.entries["core.weight"].param.grad is None  # stepped + freed inside backward
    with pytest.raises(RuntimeError, match="second gradient"):
        (m(x) - y).pow(2).mean().backward()  # forgot opt.step()


def test_late_grad_param_is_stepped_in_fused_step():
    """A param with grad only in an earlier micro-batch (no hook in the boundary backward)."""
    torch.manual_seed(0)
    a, b = nn.Parameter(torch.randn(3)), nn.Parameter(torch.randn(3))
    groups = [{"kind": "adamw", "params": [a, b], "names": ["a", "b"], "lr": 0.1, "weight_decay": 0.0,
               "betas": (0.9, 0.999), "eps": 1e-8}]
    opt = FusedStepOptimizer(groups, fused=True, impl="builtin")
    a0, b0 = a.detach().clone(), b.detach().clone()
    opt.set_boundary(False)
    (a * b).sum().backward()
    opt.set_boundary(True)
    a.sum().backward()
    opt.step()
    assert not torch.equal(a, a0) and not torch.equal(b, b0) and b.grad is None


def test_projection_clamp_and_nonfinite_skip():
    for fused in (True, False):
        m = TwiceNet()
        m.loop_adapters._qloop_project = (0.0, 0.55)
        opt = _make_opt(m, fused, schedule=None)
        for x, y in _batches(5, seed=2):
            opt.set_boundary(True)
            (m(x) - y - 10.0 * m.loop_adapters.sum()).pow(2).mean().backward()
            opt.step()
            opt.zero_grad()
            la = m.loop_adapters.detach()
            assert float(la.min()) >= 0.0 and float(la.max()) <= 0.55
        before = m.norm_w.detach().clone()
        core_before = m.core.weight.detach().clone()
        m.norm_w.register_hook(lambda g: g * float("nan"))
        x, y = _batches(1, seed=3)[0]
        opt.set_boundary(True)
        (m(x) - y).pow(2).mean().backward()
        st = opt.step()
        opt.zero_grad()
        assert torch.equal(m.norm_w.detach(), before)
        assert opt.skipped == {"norm_w": 1} and st["optim/skipped_params"] == 1
        if fused:
            assert not torch.equal(m.core.weight.detach(), core_before)
        else:
            # fused=False performs an atomic whole-step abort when any param grad is non-finite!
            assert torch.equal(m.core.weight.detach(), core_before)
            assert st["optim/stepped_params"] == 0.0


def test_state_dict_roundtrip_keyed_by_name():
    data = _batches(8, seed=4)
    m1 = TwiceNet()
    opt1 = _make_opt(m1, True)
    _train(m1, opt1, 2, 1, data[:2])
    sd = copy.deepcopy(opt1.state_dict())
    m2 = copy.deepcopy(m1)
    groups = build_param_groups(reversed(list(m2.named_parameters())), OPT_CFG)  # different order
    opt2 = FusedStepOptimizer(groups, fused=True, impl="builtin",
                              schedule=WarmupCosine(total_steps=6, warmup_steps=2, min_ratio=0.2))
    opt2.load_state_dict(sd)
    assert opt2.step_count == 2
    _train(m1, opt1, 3, 1, data[2:5])
    _train(m2, opt2, 3, 1, data[2:5])
    for (n1, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.equal(p1, p2), n1
    bad = copy.deepcopy(sd)
    bad["params"]["nope"] = bad["params"].pop("core.weight")
    with pytest.raises(AssertionError, match="names differ"):
        opt2.load_state_dict(bad)


def test_bf16_master_weights():
    p = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt = AdamWLite([p], lr=1e-4)
    for _ in range(20):
        p.grad = torch.ones_like(p)
        opt.step()
    # 20 * 1e-4 = 2e-3 is below bf16 resolution at 1.0 (7.8e-3) - only the master keeps it
    assert opt.state[p]["master"].dtype == torch.float32
    assert float(opt.state[p]["master"][0]) == pytest.approx(1 - 20 * 1e-4, rel=1e-4)


def test_lr_schedule():
    s = WarmupCosine(total_steps=12, warmup_steps=2, min_ratio=0.1)
    assert s(0) == pytest.approx(0.5) and s(1) == pytest.approx(1.0) and s(2) == pytest.approx(1.0)
    assert s(7) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.5)))
    assert s(12) == pytest.approx(0.1) and s(100) == pytest.approx(0.1)


@pytest.mark.parametrize("rank,scale_type,scale", [(4, "channel", 1.0), (1, "tensor", 128.0)])
def test_apollo_fallback_reduces_regression_loss(rank, scale_type, scale):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 32, generator=g)
    w_true = torch.randn(48, 32, generator=g)
    y = x @ w_true.t()
    w = nn.Parameter(torch.zeros(48, 32))
    opt = ApolloAdamWFallback([{"params": [w], "rank": rank, "proj": "random", "scale_type": scale_type,
                                "scale": scale, "update_proj_gap": 50, "proj_type": "std"}], lr=2e-2, seed=1)
    loss0 = float((x @ w.detach().t() - y).pow(2).mean())
    for _ in range(300):
        opt.zero_grad()
        loss = (x @ w.t() - y).pow(2).mean()
        loss.backward()
        opt.step()
    loss1 = float((x @ w.detach().t() - y).pow(2).mean())
    st = opt.state[w]
    assert loss1 < 0.1 * loss0, (loss0, loss1)
    assert st["exp_avg"].numel() == rank * 48  # low-rank optimizer state (right projection keeps m=48)
    assert st["proj_refresh"] == 6


@pytest.mark.skipif(not apollo_torch_available(), reason="apollo_torch not installed")
def test_apollo_torch_path_constructs():
    from apollo_torch import APOLLOAdamW

    m = TwiceNet()
    groups = build_param_groups(m.named_parameters(), OPT_CFG)
    opt = FusedStepOptimizer(groups, fused=False, impl="apollo_torch")
    assert any(isinstance(e.opt, APOLLOAdamW) for e in opt.entries.values())
