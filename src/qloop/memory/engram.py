"""Phase-3 `LSA-NL` Level-4 `ZipfianTieredMoEEngram` & `AnchoredMoEEngramBank` (`C4` Building Block).

Isolated, CPU-verifiable native `2048`-D Anchored `MoE-Engram` module placed strictly under
`src/qloop/memory/engram.py` (zero modifications to frozen MoE weights or active training codepaths):
  * **Core-Only Anchor Placement (`LoopSpec` `[12, 36)`):** Default anchors `(15, 19, 23, 27, 31, 35)`
    attach strictly to the `6` core full-attention boundaries (`12` virtual visits in `64L`) and reject
    Coda layers (`>= 36`, e.g., `L39`) with `ValueError` unless explicitly overridden via
    `allow_coda=True` (`Spark-X2.6` `S2` rule).
  * **Three Verified Parameter Presets (`d_model=2048`, `6` anchors, `12` virtual visits):**
      - `EngramConfig.compact()`: `E=32, K=6, d_moe=1024, d_shared=1024` (`1.25B` MoE / `+0.529B` active)
      - `EngramConfig.native_grain()`: `E=64, K=8, d_moe=512, d_shared=1024` (`1.25B` MoE / `+0.379B` active)
      - `EngramConfig.full_spark_x()`: `E=32, K=7, d_moe=1536, d_shared=2048` (`1.89B` MoE / `+0.945B` active)
  * **Tier-1A/1B Causal N-Gram Hashing & `export_ngram_cues()`:** Deterministic prime modular hashing
    over causal `2`-gram (Tier-1A, `16,384` buckets), `3`-gram and `skip-2`-gram (Tier-1B, `49,152`
    buckets) with zero-initialized embeddings (`tier1_dim=256`). `export_ngram_cues(input_ids)` returns
    lightweight `O(1)` lexical embeddings `e_ngram` directly for future Option-B `L35` `EAGLE-3`
    drafting without executing any routed or shared SwiGLU experts.
  * **Causal Depthwise `Conv1d(k=4)` + `O(1)` `rollback_conv_state(keep_tokens)`:** Zero-initialized
    depthwise causal `Conv1d(k=4)` (`arXiv:2601.07372`) maintaining a rolling `3`-token sliding buffer
    (`3 * 2048 * 2 B = 12 KiB/anchor` in `bf16`) sliced back in `O(1)` upon speculative draft rejection.
  * **Five Mandatory Stability Shields Built In From Step 0:**
      1. *Step-0 Exact Identity (`gamma_init = 0.0`):* Learnable per-(anchor, pass) residual gate
         `gamma` initialized to `0.0` so Step-0 forward is `torch.equal` (`max_abs_diff == 0.0`) to an
         unmodified backbone even with non-zero expert weights and active inputs.
      2. *Prompt-Prefix Suppression (`min_active_pos`):* Zeroes Engram writes on unsupervised prefix
         positions `t < min_active_pos` (`Spark-X2.6` `S1` early-prefix drift fix).
      3. *DeepSeek Signed-Sqrt RMSNorm QK Gate (`arXiv:2601.07372`):* Computes
         `alpha_t = sigmoid(signed_sqrt(RMSNorm(h_t)^T RMSNorm(k_t) / sqrt(d)))` with smooth
         `sign(.) * sqrt(|.|)` damping on lexical/memory features.
      4. *WISE Activation Scope Gate (`arXiv:2405.14768`) + In-Manifold Relative Norm Cap (`0.03`):*
         When `scope_prototypes` are registered and `scope_threshold > 0`, out-of-scope tokens receive
         exact `0.0` perturbation; active deltas are strictly clamped to
         `||delta||_2 <= max_rel_delta_norm * ||h||_2` per token.
      5. *Independent Sigmoid + Positive Null Sink (`s_null > 0`) + Elastic Matryoshka Slicing:*
         Computes independent thresholded sigmoid scores `s_i = sigmoid(w_i^T x + b_i) * I(s_i > tau)`
         over prefix `i in [0, N_user)` normalized with positive `s_null = softplus(s_null_raw) + eps`:
         `g_hat_j = s_j / (sum_{i in active} s_i + s_null)` guaranteeing `0 <= sum g_hat_j < 1` and
         exact `0.0` when all `s_i <= tau` (zero NaN/div-by-zero), plus `p = 0.15` training module
         dropout and nested Matryoshka `(N_batch, K_batch)` slicing during `elastic_training=True`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from qloop.model._dtype import upcast
from qloop.model.loop import LoopSpec

DEFAULT_CORE_LOOP_SPEC = LoopSpec(
    prelude_end=12,
    core_end=36,
    num_passes=2,
    target_virtual_layers=64,
)

# Mersenne prime M = 2^31 - 1 and distinct large primes < 2^30 so products
# `tok * prime < 2^61 < 2^63 - 1` never overflow signed int64 on CPU or CUDA.
_HASH_PRIME_MOD = 2_147_483_647
_PRIME_2G_PREV = 1_000_000_007
_PRIME_2G_CURR = 1_000_000_009
_SALT_2GRAM = 104_729

_PRIME_3G_PREV2 = 998_244_353
_PRIME_3G_PREV1 = 1_000_000_033
_PRIME_3G_CURR = 1_000_000_087
_SALT_3GRAM = 224_737

_PRIME_SKIP2_PREV2 = 1_000_000_093
_PRIME_SKIP2_CURR = 1_000_000_097
_SALT_SKIP2 = 350_377

# Distinct non-negative BOS sentinels for causal positions t=0 and t=1 when no `prev_tokens` are supplied.
_BOS_SENTINEL_PREV2 = 2_000_000_011
_BOS_SENTINEL_PREV1 = 2_000_000_033

NVFP4_EFFECTIVE_BYTES_PER_PARAM = 0.5625  # 4-bit weights + FP8/FP16 micro-block scales
BF16_DTYPE_BYTES = 2


def signed_sqrt(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Smooth odd-symmetric signed square-root `x / (x^2 + eps^2)^(1/4) ~ sign(x) * sqrt(|x|)`.

    Properties:
      * Exact `0.0` at `x == 0.0`, exact odd symmetry `f(-x) == -f(x)`, exact sign preservation.
      * `C^infty` smooth everywhere including `x == 0` (`f'(0) = eps^(-1/2)` finite, zero NaN).
      * Matches `sign(x) * sqrt(|x|)` to `< 1e-8` for `|x| >= 1e-2` with `eps = 1e-6`.
    """
    eps_sq = float(eps) * float(eps)
    return x * torch.pow(x.square() + eps_sq, -0.25)


class _RMSNorm(nn.Module):
    """Minimal float-upcast RMSNorm matching `Qwen3.6` / DeepSeek Engram QK normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xw = upcast(x)
        inv_rms = torch.rsqrt(xw.square().mean(dim=-1, keepdim=True) + self.eps)
        w = self.weight.to(device=xw.device, dtype=xw.dtype)
        return xw * inv_rms * w


def _default_matryoshka_tiers(num_experts: int) -> tuple[int, ...]:
    """Build sorted nested Matryoshka prefix widths terminating at `num_experts`."""
    if num_experts <= 1:
        return (num_experts,)
    tiers: set[int] = {num_experts}
    cur = num_experts
    for _ in range(3):
        cur = cur // 2
        if cur >= 1:
            tiers.add(cur)
    return tuple(sorted(tiers))


def compute_causal_ngram_hashes(
    input_ids: torch.Tensor,
    tier1a_buckets: int,
    tier1b_buckets: int,
    *,
    prev_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic causal prime modular hashing for `(2-gram, 3-gram, skip-2-gram)`.

    Strictly causal: bucket indices at position `t` depend ONLY on `input_ids[:, <= t]` (plus
    optional `prev_tokens` of shape `[B, 2]` carrying `[x_{-2}, x_{-1}]` for incremental decode).

    Args:
        input_ids: Integer token IDs of shape `[B, T]` (or `[T]`).
        tier1a_buckets: Bucket count for Tier-1A hot `2`-gram table (`>= 1`).
        tier1b_buckets: Bucket count for Tier-1B `3`-gram & `skip-2`-gram table (`>= 1`).
        prev_tokens: Optional `[B, 2]` (or `[2]`) integer history `[x_{-2}, x_{-1}]`.

    Returns:
        `(idx_2gram, idx_3gram, idx_skip2)` with matching shape to `input_ids` and `torch.long` dtype.
    """
    if tier1a_buckets < 1 or tier1b_buckets < 1:
        raise ValueError(f"bucket counts must be >= 1, got ({tier1a_buckets}, {tier1b_buckets})")
    if input_ids.ndim not in (1, 2):
        raise ValueError(f"input_ids must be 1D [T] or 2D [B, T], got shape {tuple(input_ids.shape)}")

    squeezed = input_ids.ndim == 1
    ids = input_ids.unsqueeze(0) if squeezed else input_ids
    ids_i64 = ids.to(dtype=torch.int64)
    bsz, seq_len = int(ids_i64.shape[0]), int(ids_i64.shape[1])

    if seq_len == 0:
        empty = torch.empty_like(ids_i64, dtype=torch.long)
        if squeezed:
            empty = empty.squeeze(0)
        return empty, empty.clone(), empty.clone()

    if prev_tokens is None:
        prefix = torch.tensor(
            [_BOS_SENTINEL_PREV2, _BOS_SENTINEL_PREV1],
            dtype=torch.int64,
            device=ids_i64.device,
        ).unsqueeze(0).expand(bsz, 2)
    else:
        pt = prev_tokens.unsqueeze(0) if (squeezed and prev_tokens.ndim == 1) else prev_tokens
        if pt.ndim != 2 or pt.shape != (bsz, 2):
            raise ValueError(f"prev_tokens must have shape ({bsz}, 2), got {tuple(pt.shape)}")
        prefix = pt.to(device=ids_i64.device, dtype=torch.int64)

    padded = torch.cat([prefix, ids_i64], dim=1)  # [B, T + 2]
    t0 = torch.remainder(padded[:, 2:], _HASH_PRIME_MOD)
    t1 = torch.remainder(padded[:, 1:-1], _HASH_PRIME_MOD)
    t2 = torch.remainder(padded[:, :-2], _HASH_PRIME_MOD)

    h_2gram = torch.remainder(
        torch.remainder(t1 * _PRIME_2G_PREV + _SALT_2GRAM, _HASH_PRIME_MOD)
        + torch.remainder(t0 * _PRIME_2G_CURR, _HASH_PRIME_MOD),
        _HASH_PRIME_MOD,
    )
    h_3gram = torch.remainder(
        torch.remainder(t2 * _PRIME_3G_PREV2 + _SALT_3GRAM, _HASH_PRIME_MOD)
        + torch.remainder(t1 * _PRIME_3G_PREV1, _HASH_PRIME_MOD)
        + torch.remainder(t0 * _PRIME_3G_CURR, _HASH_PRIME_MOD),
        _HASH_PRIME_MOD,
    )
    h_skip2 = torch.remainder(
        torch.remainder(t2 * _PRIME_SKIP2_PREV2 + _SALT_SKIP2, _HASH_PRIME_MOD)
        + torch.remainder(t0 * _PRIME_SKIP2_CURR, _HASH_PRIME_MOD),
        _HASH_PRIME_MOD,
    )

    idx_2gram = torch.remainder(h_2gram, int(tier1a_buckets)).to(torch.long)
    idx_3gram = torch.remainder(h_3gram, int(tier1b_buckets)).to(torch.long)
    idx_skip2 = torch.remainder(h_skip2, int(tier1b_buckets)).to(torch.long)

    if squeezed:
        return idx_2gram.squeeze(0), idx_3gram.squeeze(0), idx_skip2.squeeze(0)
    return idx_2gram, idx_3gram, idx_skip2


@dataclass
class EngramConfig:
    """Configuration for `ZipfianTieredMoEEngram` and `AnchoredMoEEngramBank`."""

    d_model: int = 2048
    anchor_layers: tuple[int, ...] = (15, 19, 23, 27, 31, 35)
    num_passes: int = 2
    num_experts: int = 32
    top_k: int = 7
    moe_intermediate_size: int = 1536
    shared_intermediate_size: int = 2048
    tier1a_buckets: int = 16384
    tier1b_buckets: int = 49152
    tier1_dim: int = 256
    conv_kernel_size: int = 4
    gamma_init: float = 0.0
    min_active_pos: int | None = None
    scope_threshold: float = 0.65
    max_rel_delta_norm: float = 0.03
    router_threshold: float = 0.1
    null_sink_init: float = 0.0
    null_sink_eps: float = 1e-4
    active_experts: int | None = None
    active_top_k: int | None = None
    module_drop_prob: float = 0.15
    elastic_training: bool = True
    matryoshka_expert_tiers: tuple[int, ...] | None = None
    rmsnorm_eps: float = 1e-6
    signed_sqrt_eps: float = 1e-6
    allow_coda: bool = False
    coda_start: int = DEFAULT_CORE_LOOP_SPEC.core_end
    seed: int = 2026

    def __post_init__(self) -> None:
        self.d_model = int(self.d_model)
        self.anchor_layers = tuple(int(lyr) for lyr in self.anchor_layers)
        self.num_passes = int(self.num_passes)
        self.num_experts = int(self.num_experts)
        self.top_k = int(self.top_k)
        self.moe_intermediate_size = int(self.moe_intermediate_size)
        self.shared_intermediate_size = int(self.shared_intermediate_size)
        self.tier1a_buckets = int(self.tier1a_buckets)
        self.tier1b_buckets = int(self.tier1b_buckets)
        self.tier1_dim = int(self.tier1_dim)
        self.conv_kernel_size = int(self.conv_kernel_size)
        self.gamma_init = float(self.gamma_init)
        if self.min_active_pos is not None:
            self.min_active_pos = int(self.min_active_pos)
        self.scope_threshold = float(self.scope_threshold)
        self.max_rel_delta_norm = float(self.max_rel_delta_norm)
        self.router_threshold = float(self.router_threshold)
        self.null_sink_init = float(self.null_sink_init)
        self.null_sink_eps = float(self.null_sink_eps)
        if self.active_experts is not None:
            self.active_experts = int(self.active_experts)
        if self.active_top_k is not None:
            self.active_top_k = int(self.active_top_k)
        self.module_drop_prob = float(self.module_drop_prob)
        self.elastic_training = bool(self.elastic_training)
        if self.matryoshka_expert_tiers is not None:
            self.matryoshka_expert_tiers = tuple(int(n) for n in self.matryoshka_expert_tiers)
        self.rmsnorm_eps = float(self.rmsnorm_eps)
        self.signed_sqrt_eps = float(self.signed_sqrt_eps)
        self.allow_coda = bool(self.allow_coda)
        self.coda_start = int(self.coda_start)
        self.seed = int(self.seed)
        self.validate()

    @classmethod
    def compact(cls, **overrides: Any) -> EngramConfig:
        """Compact preset (`E=32, K=6, d_moe=1024, d_shared=1024`: `1.25B` MoE / `+0.529B` active)."""
        params: dict[str, Any] = {
            "num_experts": 32,
            "top_k": 6,
            "moe_intermediate_size": 1024,
            "shared_intermediate_size": 1024,
        }
        params.update(overrides)
        return cls(**params)

    @classmethod
    def native_grain(cls, **overrides: Any) -> EngramConfig:
        """Native-Grain preset (`E=64, K=8, d_moe=512, d_shared=1024`: `1.25B` MoE / `+0.379B` active)."""
        params: dict[str, Any] = {
            "num_experts": 64,
            "top_k": 8,
            "moe_intermediate_size": 512,
            "shared_intermediate_size": 1024,
        }
        params.update(overrides)
        return cls(**params)

    @classmethod
    def full_spark_x(cls, **overrides: Any) -> EngramConfig:
        """Full Spark-X preset (`E=32, K=7, d_moe=1536, d_shared=2048`: `1.89B` MoE / `+0.945B` active)."""
        params: dict[str, Any] = {
            "num_experts": 32,
            "top_k": 7,
            "moe_intermediate_size": 1536,
            "shared_intermediate_size": 2048,
        }
        params.update(overrides)
        return cls(**params)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> EngramConfig:
        raw = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown engram.* keys {sorted(bad)}; valid: {sorted(known)}")
        for tup_key in ("anchor_layers", "matryoshka_expert_tiers"):
            if tup_key in raw and isinstance(raw[tup_key], list):
                raw[tup_key] = tuple(raw[tup_key])
        return cls(**raw)

    @classmethod
    def from_loop_spec(
        cls,
        spec: LoopSpec = DEFAULT_CORE_LOOP_SPEC,
        *,
        full_attention_interval: int = 4,
        **overrides: Any,
    ) -> EngramConfig:
        """Build an `EngramConfig` matching core full-attention layers inside `[spec.prelude_end, spec.core_end)`."""
        if full_attention_interval < 1:
            raise ValueError(f"full_attention_interval must be >= 1, got {full_attention_interval}")
        anchors = tuple(
            i for i in range(spec.prelude_end, spec.core_end) if (i + 1) % full_attention_interval == 0
        )
        return cls(
            anchor_layers=anchors,
            num_passes=spec.num_passes,
            coda_start=spec.core_end,
            **overrides,
        )

    def validate(self, loop_spec: LoopSpec | None = None) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.num_passes < 1:
            raise ValueError(f"num_passes must be >= 1, got {self.num_passes}")
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {self.num_experts}")
        if not (0 <= self.top_k <= self.num_experts):
            raise ValueError(f"top_k must be in [0, num_experts={self.num_experts}], got {self.top_k}")
        if self.moe_intermediate_size < 1 or self.shared_intermediate_size < 1:
            raise ValueError(
                "moe_intermediate_size and shared_intermediate_size must be >= 1, "
                f"got ({self.moe_intermediate_size}, {self.shared_intermediate_size})"
            )
        if self.tier1a_buckets < 1 or self.tier1b_buckets < 1 or self.tier1_dim < 1:
            raise ValueError(
                f"tier1a_buckets, tier1b_buckets, tier1_dim must be >= 1, "
                f"got ({self.tier1a_buckets}, {self.tier1b_buckets}, {self.tier1_dim})"
            )
        if self.conv_kernel_size < 1:
            raise ValueError(f"conv_kernel_size must be >= 1, got {self.conv_kernel_size}")
        if not self.anchor_layers:
            raise ValueError("anchor_layers must be non-empty")
        if len(set(self.anchor_layers)) != len(self.anchor_layers):
            raise ValueError(f"duplicate anchor_layers found: {self.anchor_layers}")
        if any(lyr < 0 for lyr in self.anchor_layers):
            raise ValueError(f"anchor_layers must be >= 0, got {self.anchor_layers}")

        coda_boundary = self.coda_start if loop_spec is None else int(loop_spec.core_end)
        coda_layers = [lyr for lyr in self.anchor_layers if lyr >= coda_boundary]
        if coda_layers and not self.allow_coda:
            raise ValueError(
                f"Coda anchor layers {coda_layers} (>= {coda_boundary}) are rejected by the Spark-X2.6 S2 rule; "
                "pass allow_coda=True to override explicitly."
            )

        if not math.isfinite(self.gamma_init):
            raise ValueError(f"gamma_init must be finite, got {self.gamma_init}")
        if self.min_active_pos is not None and self.min_active_pos < 0:
            raise ValueError(f"min_active_pos must be >= 0 or None, got {self.min_active_pos}")
        if not math.isfinite(self.scope_threshold) or not (0.0 <= self.scope_threshold <= 1.0):
            raise ValueError(f"scope_threshold must be finite and in [0, 1], got {self.scope_threshold}")
        if not math.isfinite(self.max_rel_delta_norm) or self.max_rel_delta_norm <= 0.0:
            raise ValueError(f"max_rel_delta_norm must be finite and > 0, got {self.max_rel_delta_norm}")
        if not math.isfinite(self.router_threshold) or not (0.0 <= self.router_threshold < 1.0):
            raise ValueError(f"router_threshold must be finite and in [0, 1), got {self.router_threshold}")
        if not math.isfinite(self.null_sink_init):
            raise ValueError(f"null_sink_init must be finite, got {self.null_sink_init}")
        if not math.isfinite(self.null_sink_eps) or self.null_sink_eps <= 0.0:
            raise ValueError(f"null_sink_eps must be finite and > 0, got {self.null_sink_eps}")
        if self.active_experts is not None and not (0 <= self.active_experts <= self.num_experts):
            raise ValueError(
                f"active_experts must be in [0, num_experts={self.num_experts}], got {self.active_experts}"
            )
        if self.active_top_k is not None and not (0 <= self.active_top_k <= self.num_experts):
            raise ValueError(
                f"active_top_k must be in [0, num_experts={self.num_experts}], got {self.active_top_k}"
            )
        if not math.isfinite(self.module_drop_prob) or not (0.0 <= self.module_drop_prob <= 1.0):
            raise ValueError(f"module_drop_prob must be finite and in [0, 1], got {self.module_drop_prob}")
        if self.matryoshka_expert_tiers is not None:
            if not self.matryoshka_expert_tiers:
                raise ValueError("matryoshka_expert_tiers must be non-empty when specified")
            if any(n < 0 or n > self.num_experts for n in self.matryoshka_expert_tiers):
                raise ValueError(
                    f"matryoshka_expert_tiers entries must be in [0, {self.num_experts}], "
                    f"got {self.matryoshka_expert_tiers}"
                )
        if not math.isfinite(self.rmsnorm_eps) or self.rmsnorm_eps <= 0.0:
            raise ValueError(f"rmsnorm_eps must be finite and > 0, got {self.rmsnorm_eps}")
        if not math.isfinite(self.signed_sqrt_eps) or self.signed_sqrt_eps <= 0.0:
            raise ValueError(f"signed_sqrt_eps must be finite and > 0, got {self.signed_sqrt_eps}")

    def conv_state_bytes_per_anchor(self, batch_size: int = 1, dtype_bytes: int = BF16_DTYPE_BYTES) -> int:
        """Exact rolling `conv1d` sliding-buffer bytes per anchor (`3 * 2048 * 2 = 12,288 B = 12 KiB`)."""
        if batch_size < 1 or dtype_bytes < 1:
            raise ValueError(f"batch_size and dtype_bytes must be >= 1, got ({batch_size}, {dtype_bytes})")
        return int(batch_size) * (self.conv_kernel_size - 1) * self.d_model * int(dtype_bytes)

    def conv_state_bytes(self, batch_size: int = 1, dtype_bytes: int = BF16_DTYPE_BYTES) -> int:
        """Total rolling `conv1d` sliding-buffer bytes across all `anchor_layers`."""
        return len(self.anchor_layers) * self.conv_state_bytes_per_anchor(
            batch_size=batch_size, dtype_bytes=dtype_bytes
        )

    def param_summary(
        self,
        *,
        num_passes: int | None = None,
        active_experts: int | None = None,
        active_top_k: int | None = None,
    ) -> dict[str, int | float]:
        """Exact closed-form parameter and memory accounting for `AnchoredMoEEngramBank`.

        Matches verified `d_model=2048`, `6`-anchor (`12` virtual visits at `num_passes=2`) formulas:
          * Compact (`32/6/1024/1024`): `1.246B (~1.25B)` MoE total / `+0.529B` active (`0.65 GiB` NVFP4)
          * Native-Grain (`64/8/512/1024`): `1.246B (~1.25B)` MoE total / `+0.379B` active (`0.65 GiB` NVFP4)
          * Full Spark-X (`32/7/1536/2048`): `1.888B (~1.89B)` MoE total / `+0.945B` active (`0.99 GiB` NVFP4)
        """
        passes = self.num_passes if num_passes is None else int(num_passes)
        if passes < 1:
            raise ValueError(f"num_passes must be >= 1, got {passes}")
        eff_experts_opt = self.active_experts if active_experts is None else active_experts
        eff_experts = self.num_experts if eff_experts_opt is None else int(eff_experts_opt)
        raw_top_k_opt = self.active_top_k if active_top_k is None else active_top_k
        raw_top_k = self.top_k if raw_top_k_opt is None else int(raw_top_k_opt)
        if not (0 <= eff_experts <= self.num_experts):
            raise ValueError(f"active_experts must be in [0, {self.num_experts}], got {eff_experts}")
        if not (0 <= raw_top_k <= self.num_experts):
            raise ValueError(f"active_top_k must be in [0, {self.num_experts}], got {raw_top_k}")
        eff_top_k = min(raw_top_k, eff_experts)

        n_anchors = len(self.anchor_layers)
        n_visits = n_anchors * passes
        d = self.d_model

        routed_per_expert = 3 * d * self.moe_intermediate_size
        shared_per_anchor = 3 * d * self.shared_intermediate_size
        router_per_anchor = self.num_experts * d + self.num_experts + 1
        qk_conv_per_anchor = 2 * self.tier1_dim * d + (3 + self.conv_kernel_size) * d + self.num_passes

        routed_total = n_anchors * self.num_experts * routed_per_expert
        shared_total = n_anchors * shared_per_anchor
        router_total = n_anchors * router_per_anchor
        moe_total = routed_total + shared_total + router_total
        tier1_table_total = (self.tier1a_buckets + self.tier1b_buckets) * self.tier1_dim
        qk_conv_total = n_anchors * qk_conv_per_anchor
        total_params = moe_total + tier1_table_total + qk_conv_total

        routed_active = n_visits * eff_top_k * routed_per_expert
        shared_active = n_visits * shared_per_anchor if eff_experts > 0 else 0
        router_active = n_visits * (eff_experts * d + eff_experts + 1) if eff_experts > 0 else 0
        moe_active = routed_active + shared_active + router_active
        qk_conv_active = n_visits * qk_conv_per_anchor
        active_params = moe_active + qk_conv_active

        conv_per_anchor_b = self.conv_state_bytes_per_anchor()
        conv_total_b = self.conv_state_bytes()

        return {
            "num_anchors": n_anchors,
            "num_passes": passes,
            "num_virtual_visits": n_visits,
            "active_experts": eff_experts,
            "active_top_k": eff_top_k,
            "routed_total_params": routed_total,
            "shared_total_params": shared_total,
            "router_total_params": router_total,
            "moe_total_params": moe_total,
            "tier1_table_params": tier1_table_total,
            "qk_conv_total_params": qk_conv_total,
            "total_params": total_params,
            "routed_active_params": routed_active,
            "shared_active_params": shared_active,
            "router_active_params": router_active,
            "moe_active_params": moe_active,
            "qk_conv_active_params": qk_conv_active,
            "active_params": active_params,
            "moe_total_params_b": moe_total / 1e9,
            "moe_active_params_b": moe_active / 1e9,
            "total_params_b": total_params / 1e9,
            "active_params_b": active_params / 1e9,
            "nvfp4_moe_gib": (moe_total * NVFP4_EFFECTIVE_BYTES_PER_PARAM) / float(1024**3),
            "nvfp4_shared_active_gb": (shared_active * NVFP4_EFFECTIVE_BYTES_PER_PARAM) / 1e9,
            "nvfp4_routed_active_gb": (routed_active * NVFP4_EFFECTIVE_BYTES_PER_PARAM) / 1e9,
            "conv_state_bytes_per_anchor": conv_per_anchor_b,
            "conv_state_bytes_total": conv_total_b,
            "conv_state_kib_per_anchor": conv_per_anchor_b / 1024.0,
        }


@dataclass
class EngramConvState:
    """Rolling `O(1)` sliding state for causal `Conv1d(k=4)` and causal N-gram prefix history."""

    conv_buffer: torch.Tensor
    prev_tokens: torch.Tensor | None = None
    pos_cursor: int = 0
    _last_conv_window: torch.Tensor | None = None
    _last_token_window: torch.Tensor | None = None
    _last_chunk_len: int = 0
    _last_pos_start: int = 0

    def clone(self) -> EngramConvState:
        return EngramConvState(
            conv_buffer=self.conv_buffer.clone(),
            prev_tokens=None if self.prev_tokens is None else self.prev_tokens.clone(),
            pos_cursor=int(self.pos_cursor),
            _last_conv_window=None if self._last_conv_window is None else self._last_conv_window.clone(),
            _last_token_window=None if self._last_token_window is None else self._last_token_window.clone(),
            _last_chunk_len=int(self._last_chunk_len),
            _last_pos_start=int(self._last_pos_start),
        )


@dataclass
class EngramRouteOutput:
    """Detailed routing output from Shield 5 (Independent Sigmoid + Positive `s_null` + Matryoshka)."""

    top_indices: torch.Tensor   # [B, T, k_eff] int64
    top_weights: torch.Tensor   # [B, T, k_eff] float
    full_weights: torch.Tensor  # [B, T, N_user] float
    total_gate: torch.Tensor    # [B, T, 1] float in [0, 1)
    s_null: torch.Tensor        # scalar float > 0
    active_experts: int
    active_top_k: int


@dataclass
class EngramForwardAux:
    """Optional auxiliary diagnostics returned when `forward(..., return_aux=True)`."""

    alpha: torch.Tensor              # [B, T, 1] QK gate in (0, 1)
    route: EngramRouteOutput
    in_scope_mask: torch.Tensor      # [B, T] bool
    active_pos_mask: torch.Tensor    # [B, T] bool
    module_keep_mask: torch.Tensor   # [B, 1, 1] bool
    pre_cap_rel_norm: torch.Tensor   # [B, T] float
    post_cap_rel_norm: torch.Tensor  # [B, T] float


class Tier1NgramTable(nn.Module):
    """Shared `O(1)` causal Tier-1A (`2`-gram) + Tier-1B (`3`-gram, `skip-2`-gram) lexical embedding table.

    Zero-initialized at Step 0 so lexical cues start at exact `0.0` before training.
    """

    def __init__(self, tier1a_buckets: int = 16384, tier1b_buckets: int = 49152, tier1_dim: int = 256) -> None:
        super().__init__()
        if tier1a_buckets < 1 or tier1b_buckets < 1 or tier1_dim < 1:
            raise ValueError(
                f"tier1a_buckets, tier1b_buckets, tier1_dim must be >= 1, "
                f"got ({tier1a_buckets}, {tier1b_buckets}, {tier1_dim})"
            )
        self.tier1a_buckets = int(tier1a_buckets)
        self.tier1b_buckets = int(tier1b_buckets)
        self.tier1_dim = int(tier1_dim)

        self.tier1a_embed = nn.Embedding(self.tier1a_buckets, self.tier1_dim)
        self.tier1b_embed = nn.Embedding(self.tier1b_buckets, self.tier1_dim)
        nn.init.zeros_(self.tier1a_embed.weight)
        nn.init.zeros_(self.tier1b_embed.weight)

    def forward(self, input_ids: torch.Tensor, *, prev_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """Return `O(1)` fused causal Tier-1 lexical embeddings `e_ngram` (`[..., tier1_dim]`)."""
        idx_2g, idx_3g, idx_s2 = compute_causal_ngram_hashes(
            input_ids,
            self.tier1a_buckets,
            self.tier1b_buckets,
            prev_tokens=prev_tokens,
        )
        dev = self.tier1a_embed.weight.device
        idx_2g = idx_2g.to(device=dev)
        idx_3g = idx_3g.to(device=dev)
        idx_s2 = idx_s2.to(device=dev)
        e_2g = self.tier1a_embed(idx_2g)
        e_3g = self.tier1b_embed(idx_3g)
        e_s2 = self.tier1b_embed(idx_s2)
        return (e_2g + e_3g + e_s2) * (1.0 / math.sqrt(3.0))


class ZipfianTieredMoEEngram(nn.Module):
    """Single-anchor native `2048`-D `ZipfianTieredMoEEngram` with all 5 stability shields & `O(1)` rollback."""

    scope_prototypes: torch.Tensor

    def __init__(
        self,
        config: EngramConfig,
        anchor_layer: int = 15,
        *,
        own_tier1: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.anchor_layer = int(anchor_layer)
        self.own_tier1 = bool(own_tier1)

        d = config.d_model
        e_cnt = config.num_experts
        d_moe = config.moe_intermediate_size
        d_sh = config.shared_intermediate_size

        # Optional anchor-owned Tier-1 table (disabled inside `AnchoredMoEEngramBank`, which shares 1 table)
        self.tier1: Tier1NgramTable | None = (
            Tier1NgramTable(config.tier1a_buckets, config.tier1b_buckets, config.tier1_dim)
            if self.own_tier1
            else None
        )

        # Shield 3: DeepSeek Signed-Sqrt RMSNorm QK Gate + zero-init causal depthwise Conv1d(k=4)
        self.key_proj = nn.Linear(config.tier1_dim, d, bias=False)
        self.val_proj = nn.Linear(config.tier1_dim, d, bias=False)
        self.rmsnorm_q = _RMSNorm(d, eps=config.rmsnorm_eps)
        self.rmsnorm_k = _RMSNorm(d, eps=config.rmsnorm_eps)
        self.rmsnorm_conv = _RMSNorm(d, eps=config.rmsnorm_eps)
        self.conv1d = nn.Conv1d(
            in_channels=d,
            out_channels=d,
            kernel_size=config.conv_kernel_size,
            groups=d,
            bias=False,
        )
        nn.init.zeros_(self.conv1d.weight)

        # Shield 5: Independent sigmoid router + positive learned null sink `s_null`
        self.router = nn.Linear(d, e_cnt, bias=True)
        self.s_null_raw = nn.Parameter(torch.tensor(config.null_sink_init, dtype=torch.float32))

        # Routed SwiGLU experts `[E, d_moe, d]` / `[E, d, d_moe]` (sparse active-only dispatch)
        self.routed_gate_proj = nn.Parameter(torch.empty((e_cnt, d_moe, d), dtype=torch.float32))
        self.routed_up_proj = nn.Parameter(torch.empty((e_cnt, d_moe, d), dtype=torch.float32))
        self.routed_down_proj = nn.Parameter(torch.empty((e_cnt, d, d_moe), dtype=torch.float32))

        # Always-available shared SwiGLU expert (gated by `total_gate` so exact `0.0` when all `s_i <= tau`)
        self.shared_gate_proj = nn.Linear(d, d_sh, bias=False)
        self.shared_up_proj = nn.Linear(d, d_sh, bias=False)
        self.shared_down_proj = nn.Linear(d_sh, d, bias=False)

        # Shield 1: Learnable per-(anchor, pass) residual gate initialized strictly to `gamma_init=0.0`
        self.gamma = nn.Parameter(torch.full((config.num_passes,), config.gamma_init, dtype=torch.float32))

        # Shield 4: WISE scope prototypes buffer (empty => pass-through; non-empty + threshold > 0 => hard gate)
        self.register_buffer("scope_prototypes", torch.empty((0, d), dtype=torch.float32))

        # Per-pass rolling conv1d / N-gram streaming caches (non-persistent runtime state)
        self._conv_states: dict[int, EngramConvState] = {}

        self._init_weights()

    def _init_weights(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(self.config.seed + 97 * self.anchor_layer)
        inv_sqrt_d = 1.0 / math.sqrt(float(self.config.d_model))
        inv_sqrt_t1 = 1.0 / math.sqrt(float(self.config.tier1_dim))
        inv_sqrt_moe = 1.0 / math.sqrt(float(self.config.moe_intermediate_size))
        inv_sqrt_sh = 1.0 / math.sqrt(float(self.config.shared_intermediate_size))

        with torch.no_grad():
            self.key_proj.weight.normal_(0.0, inv_sqrt_t1, generator=gen)
            self.val_proj.weight.normal_(0.0, inv_sqrt_t1, generator=gen)
            self.router.weight.normal_(0.0, 0.02, generator=gen)
            self.router.bias.zero_()

            self.routed_gate_proj.normal_(0.0, inv_sqrt_d, generator=gen)
            self.routed_up_proj.normal_(0.0, inv_sqrt_d, generator=gen)
            self.routed_down_proj.normal_(0.0, inv_sqrt_moe, generator=gen)

            self.shared_gate_proj.weight.normal_(0.0, inv_sqrt_d, generator=gen)
            self.shared_up_proj.weight.normal_(0.0, inv_sqrt_d, generator=gen)
            self.shared_down_proj.weight.normal_(0.0, inv_sqrt_sh, generator=gen)

            # Re-assert strict Step-0 invariants
            self.conv1d.weight.zero_()
            self.gamma.fill_(self.config.gamma_init)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        proto_key = prefix + "scope_prototypes"
        if proto_key in state_dict:
            loaded = state_dict[proto_key]
            if self.scope_prototypes.shape != loaded.shape:
                self.scope_prototypes = torch.empty_like(loaded, device=self.scope_prototypes.device)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self._conv_states.clear()

    def conv_state_bytes(self, batch_size: int = 1, dtype_bytes: int = BF16_DTYPE_BYTES) -> int:
        """Exact `conv1d(k=4)` rolling state footprint (`12 KiB/anchor` at `d_model=2048, bf16`)."""
        return self.config.conv_state_bytes_per_anchor(batch_size=batch_size, dtype_bytes=dtype_bytes)

    @property
    def conv_state(self) -> torch.Tensor | None:
        """Return pass-0 rolling `conv1d` buffer `[B, d_model, k - 1]` if cached, else `None`."""
        st = self._conv_states.get(0)
        return None if st is None else st.conv_buffer

    def get_conv_state(self, pass_idx: int = 0) -> EngramConvState | None:
        """Return full `EngramConvState` object for `pass_idx` (or `None` if uninitialized)."""
        return self._conv_states.get(int(pass_idx))

    def reset_conv_state(self, pass_idx: int | None = None) -> None:
        """Clear cached rolling `conv1d` and causal N-gram prefix state."""
        if pass_idx is None:
            self._conv_states.clear()
        else:
            self._conv_states.pop(int(pass_idx), None)

    @torch.no_grad()
    def rollback_conv_state(
        self,
        keep_tokens: int,
        *,
        pass_idx: int | None = None,
        state: EngramConvState | None = None,
    ) -> EngramConvState | dict[int, EngramConvState]:
        """`O(1)` rollback of the `(k - 1)`-token `conv1d` sliding buffer for speculative draft rejection.

        After running a verification chunk of `K` candidate tokens with `use_conv_cache=True`, calling
        `rollback_conv_state(keep_tokens)` (with `0 <= keep_tokens <= K`) slices the cached window in
        `O(1)` to restore the exact `(k - 1)`-token conv buffer, `2`-token N-gram history, and position
        cursor matching the accepted prefix of length `keep_tokens`.
        """
        keep = int(keep_tokens)
        k_pad = self.config.conv_kernel_size - 1

        def _rollback_one(st: EngramConvState) -> EngramConvState:
            if st._last_conv_window is None:
                raise RuntimeError("no rollback window recorded in conv state; run forward(..., use_conv_cache=True)")
            if not (0 <= keep <= st._last_chunk_len):
                raise ValueError(
                    f"keep_tokens must be in [0, last_chunk_len={st._last_chunk_len}], got {keep}"
                )
            st.conv_buffer = st._last_conv_window[:, :, keep : keep + k_pad].detach().clone()
            st._last_conv_window = st._last_conv_window[:, :, : keep + k_pad].detach().clone()
            if st._last_token_window is not None:
                st.prev_tokens = st._last_token_window[:, keep : keep + 2].detach().clone()
                st._last_token_window = st._last_token_window[:, : keep + 2].detach().clone()
            st.pos_cursor = st._last_pos_start + keep
            st._last_chunk_len = keep
            return st

        if state is not None:
            return _rollback_one(state)

        if not self._conv_states:
            raise RuntimeError("no active conv state to roll back; run forward(..., use_conv_cache=True) first")

        if pass_idx is not None:
            p = int(pass_idx)
            if p not in self._conv_states:
                raise KeyError(f"no cached conv state for pass_idx={p}")
            return _rollback_one(self._conv_states[p])

        if len(self._conv_states) == 1 and 0 in self._conv_states:
            return _rollback_one(self._conv_states[0])

        return {p: _rollback_one(st) for p, st in self._conv_states.items()}

    @torch.no_grad()
    def register_scope_prototypes(self, prototypes: torch.Tensor, *, replace: bool = False) -> torch.Tensor:
        """Register unit-normalized `WISE` activation scope prototypes (`[M, d_model]`)."""
        if prototypes.ndim == 1:
            protos_in = prototypes.unsqueeze(0)
        elif prototypes.ndim == 2:
            protos_in = prototypes
        else:
            raise ValueError(f"prototypes must be 1D or 2D [M, d_model], got shape {tuple(prototypes.shape)}")
        if protos_in.shape[-1] != self.config.d_model:
            raise ValueError(
                f"prototypes last dim must equal d_model={self.config.d_model}, got {protos_in.shape[-1]}"
            )
        p_f32 = protos_in.detach().to(device=self.scope_prototypes.device, dtype=torch.float32)
        norms = p_f32.norm(dim=-1, keepdim=True)
        valid = norms.squeeze(-1) > 1e-12
        unit = p_f32[valid] / norms[valid] if bool(valid.any()) else torch.empty(
            (0, self.config.d_model), dtype=torch.float32, device=self.scope_prototypes.device
        )
        if replace or self.scope_prototypes.shape[0] == 0:
            self.scope_prototypes = unit.clone()
        elif unit.shape[0] > 0:
            self.scope_prototypes = torch.cat([self.scope_prototypes, unit], dim=0)
        return self.scope_prototypes

    @torch.no_grad()
    def clear_scope_prototypes(self) -> None:
        """Clear all registered `WISE` scope prototypes (reverting to unrestricted pass-through)."""
        self.scope_prototypes = torch.empty(
            (0, self.config.d_model), dtype=torch.float32, device=self.scope_prototypes.device
        )

    def export_ngram_cues(
        self,
        input_ids: torch.Tensor,
        *,
        prev_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return lightweight `O(1)` Tier-1 lexical cues `e_ngram` without running SwiGLU experts."""
        if self.tier1 is None:
            raise RuntimeError(
                "this anchor does not own a Tier1NgramTable; call bank.export_ngram_cues(input_ids) instead"
            )
        return self.tier1(input_ids, prev_tokens=prev_tokens)

    def compute_qk_gate(self, h: torch.Tensor, e_ngram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shield 3 — DeepSeek Signed-Sqrt RMSNorm Query-Key Gate (`arXiv:2601.07372`).

        Computes:
          `k_t = W_K e_t`, `v_t = W_V e_t`
          `alpha_t = sigmoid(signed_sqrt(RMSNorm(h_t)^T RMSNorm(k_t) / sqrt(d)))`
          `gated_v_t = alpha_t * signed_sqrt(v_t)`
        """
        h_w = upcast(h)
        e_w = e_ngram.to(device=h_w.device, dtype=h_w.dtype)
        w_k = self.key_proj.weight.to(device=h_w.device, dtype=h_w.dtype)
        w_v = self.val_proj.weight.to(device=h_w.device, dtype=h_w.dtype)

        k_t = F.linear(e_w, w_k)
        v_t = F.linear(e_w, w_v)

        q_norm = self.rmsnorm_q(h_w)
        k_norm = self.rmsnorm_k(k_t)
        raw_dot = (q_norm * k_norm).sum(dim=-1, keepdim=True) / math.sqrt(float(self.config.d_model))
        alpha = torch.sigmoid(signed_sqrt(raw_dot, eps=self.config.signed_sqrt_eps))
        gated_v = alpha * signed_sqrt(v_t, eps=self.config.signed_sqrt_eps)
        return alpha, gated_v

    def _apply_causal_conv1d(
        self,
        gated_v: torch.Tensor,
        *,
        pass_idx: int,
        use_conv_cache: bool,
        conv_state: EngramConvState | None,
        token_window: torch.Tensor | None,
        pos_start: int,
        cache_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Apply zero-init causal depthwise `Conv1d(k=4)` with residual `SiLU(Conv(RMSNorm(v))) + v`."""
        bsz, seq_len, d_model = gated_v.shape
        k_size = self.config.conv_kernel_size
        k_pad = k_size - 1

        conv_in = self.rmsnorm_conv(gated_v).transpose(1, 2)  # [B, D, T]
        w_conv = self.conv1d.weight.to(device=conv_in.device, dtype=conv_in.dtype)

        if use_conv_cache:
            st = conv_state if conv_state is not None else self._conv_states.get(pass_idx)
            if st is None:
                prev_buf = conv_in.new_zeros((bsz, d_model, k_pad))
            else:
                if st.conv_buffer.shape[0] != bsz or st.conv_buffer.shape[1] != d_model:
                    raise ValueError(
                        f"cached conv_state shape {tuple(st.conv_buffer.shape)} mismatches "
                        f"batch/d_model ({bsz}, {d_model}); call reset_conv_state() first"
                    )
                prev_buf = st.conv_buffer.to(device=conv_in.device, dtype=conv_in.dtype)
            full_window = torch.cat([prev_buf, conv_in], dim=-1)  # [B, D, k_pad + T]
            conv_raw = F.conv1d(full_window, w_conv, bias=None, stride=1, padding=0, groups=d_model)

            win_stored = full_window.detach().to(dtype=cache_dtype).clone()
            new_buf = win_stored[:, :, -k_pad:].clone() if k_pad > 0 else win_stored[:, :, :0].clone()
            new_prev_toks = (
                None if token_window is None else token_window[:, -2:].detach().clone()
            )
            updated = EngramConvState(
                conv_buffer=new_buf,
                prev_tokens=new_prev_toks,
                pos_cursor=pos_start + seq_len,
                _last_conv_window=win_stored,
                _last_token_window=None if token_window is None else token_window.detach().clone(),
                _last_chunk_len=seq_len,
                _last_pos_start=pos_start,
            )
            if conv_state is not None:
                conv_state.conv_buffer = updated.conv_buffer
                conv_state.prev_tokens = updated.prev_tokens
                conv_state.pos_cursor = updated.pos_cursor
                conv_state._last_conv_window = updated._last_conv_window
                conv_state._last_token_window = updated._last_token_window
                conv_state._last_chunk_len = updated._last_chunk_len
                conv_state._last_pos_start = updated._last_pos_start
            else:
                self._conv_states[pass_idx] = updated
        else:
            padded = F.pad(conv_in, (k_pad, 0)) if k_pad > 0 else conv_in
            conv_raw = F.conv1d(padded, w_conv, bias=None, stride=1, padding=0, groups=d_model)

        conv_out = F.silu(conv_raw).transpose(1, 2)  # [B, T, D]
        return gated_v + conv_out

    def _resolve_elastic_slice(
        self,
        active_experts: int | None,
        active_top_k: int | None,
    ) -> tuple[int, int]:
        """Resolve `(N_user, K_eff)` honoring explicit overrides and Matryoshka training sampling."""
        n_user = active_experts if active_experts is not None else self.config.active_experts
        k_req = active_top_k if active_top_k is not None else self.config.active_top_k

        if self.training and self.config.elastic_training:
            if n_user is None:
                tiers = (
                    self.config.matryoshka_expert_tiers
                    if self.config.matryoshka_expert_tiers is not None
                    else _default_matryoshka_tiers(self.config.num_experts)
                )
                pick_idx = int(torch.randint(0, len(tiers), (1,)).item())
                n_user = tiers[pick_idx]
            if k_req is None:
                max_k = min(self.config.top_k, n_user)
                k_req = 0 if max_k == 0 else int(torch.randint(1, max_k + 1, (1,)).item())
        else:
            if n_user is None:
                n_user = self.config.num_experts
            if k_req is None:
                k_req = self.config.top_k

        n_user = int(n_user)
        k_req = int(k_req)
        if not (0 <= n_user <= self.config.num_experts):
            raise ValueError(f"active_experts must be in [0, {self.config.num_experts}], got {n_user}")
        if not (0 <= k_req <= self.config.num_experts):
            raise ValueError(f"active_top_k must be in [0, {self.config.num_experts}], got {k_req}")
        return n_user, min(k_req, n_user)

    def route(
        self,
        x_route: torch.Tensor,
        *,
        active_experts: int | None = None,
        active_top_k: int | None = None,
    ) -> EngramRouteOutput:
        """Shield 5 — Independent Sigmoid + Positive Null Sink (`s_null > 0`) + Matryoshka Prefix Slice.

        For `i in [0, N_user)`:
          `s_i = sigmoid(w_i^T x_route + b_i) * I(s_i > tau)`
          `s_null = softplus(s_null_raw) + eps > 0`
          `g_hat_j = s_j / (sum_{i in top_k(active)} s_i + s_null)`
        Guarantees `0 <= sum_j g_hat_j < 1` for every `(N_user, active_top_k)` and exact `0.0`
        when `N_user == 0`, `active_top_k == 0`, or all `s_i <= tau` (zero NaN/div-by-zero).
        """
        xw = upcast(x_route)
        bsz, seq_len, _ = xw.shape
        n_user, k_eff = self._resolve_elastic_slice(active_experts, active_top_k)

        s_null = F.softplus(self.s_null_raw.to(device=xw.device, dtype=xw.dtype)) + self.config.null_sink_eps

        if n_user == 0 or k_eff == 0:
            empty_idx = torch.empty((bsz, seq_len, 0), dtype=torch.long, device=xw.device)
            empty_w = torch.empty((bsz, seq_len, 0), dtype=xw.dtype, device=xw.device)
            full_w = torch.zeros((bsz, seq_len, n_user), dtype=xw.dtype, device=xw.device)
            total_gate = torch.zeros((bsz, seq_len, 1), dtype=xw.dtype, device=xw.device)
            return EngramRouteOutput(
                top_indices=empty_idx,
                top_weights=empty_w,
                full_weights=full_w,
                total_gate=total_gate,
                s_null=s_null,
                active_experts=n_user,
                active_top_k=k_eff,
            )

        w_r = self.router.weight[:n_user].to(device=xw.device, dtype=xw.dtype)
        b_r = self.router.bias[:n_user].to(device=xw.device, dtype=xw.dtype)
        probs = torch.sigmoid(F.linear(xw, w_r, b_r))  # [B, T, N_user]
        s_thresh = torch.where(probs > self.config.router_threshold, probs, torch.zeros_like(probs))

        top_vals, top_idx = torch.topk(s_thresh, k=k_eff, dim=-1)  # [B, T, k_eff]
        active_sum = top_vals.sum(dim=-1, keepdim=True)
        denom = active_sum + s_null  # strictly >= s_null > 0
        top_weights = top_vals / denom  # [B, T, k_eff]
        total_gate = top_weights.sum(dim=-1, keepdim=True)  # [B, T, 1] strictly in [0, 1)

        full_weights = torch.zeros((bsz, seq_len, n_user), dtype=xw.dtype, device=xw.device)
        full_weights.scatter_(-1, top_idx, top_weights)

        return EngramRouteOutput(
            top_indices=top_idx,
            top_weights=top_weights,
            full_weights=full_weights,
            total_gate=total_gate,
            s_null=s_null,
            active_experts=n_user,
            active_top_k=k_eff,
        )

    def _run_routed_experts(
        self,
        x_in: torch.Tensor,
        top_idx: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse active-only SwiGLU dispatch touching strictly experts selected with `weight > 0`."""
        bsz, seq_len, d_model = x_in.shape
        x_flat = x_in.reshape(bsz * seq_len, d_model)
        out_flat = torch.zeros_like(x_flat)

        if top_weights.shape[-1] == 0:
            return out_flat.view(bsz, seq_len, d_model)

        idx_flat = top_idx.reshape(bsz * seq_len, -1)
        w_flat = top_weights.reshape(bsz * seq_len, -1)
        active_mask = w_flat > 0.0
        if not bool(active_mask.any()):
            return out_flat.view(bsz, seq_len, d_model)

        w_gate_all = self.routed_gate_proj.to(device=x_flat.device, dtype=x_flat.dtype)
        w_up_all = self.routed_up_proj.to(device=x_flat.device, dtype=x_flat.dtype)
        w_down_all = self.routed_down_proj.to(device=x_flat.device, dtype=x_flat.dtype)

        active_expert_ids = torch.unique(idx_flat[active_mask]).tolist()
        for exp_id in active_expert_ids:
            slot_mask = active_mask & (idx_flat == exp_id)
            tok_idx, k_idx = slot_mask.nonzero(as_tuple=True)
            if tok_idx.numel() == 0:
                continue
            x_e = x_flat.index_select(0, tok_idx)
            g_e = F.linear(x_e, w_gate_all[exp_id])
            u_e = F.linear(x_e, w_up_all[exp_id])
            y_e = F.linear(F.silu(g_e) * u_e, w_down_all[exp_id])
            w_e = w_flat[tok_idx, k_idx].unsqueeze(-1)
            out_flat.index_add_(0, tok_idx, y_e * w_e)

        return out_flat.view(bsz, seq_len, d_model)

    def _run_shared_expert(self, x_in: torch.Tensor) -> torch.Tensor:
        w_g = self.shared_gate_proj.weight.to(device=x_in.device, dtype=x_in.dtype)
        w_u = self.shared_up_proj.weight.to(device=x_in.device, dtype=x_in.dtype)
        w_d = self.shared_down_proj.weight.to(device=x_in.device, dtype=x_in.dtype)
        return F.linear(F.silu(F.linear(x_in, w_g)) * F.linear(x_in, w_u), w_d)

    def _compute_scope_mask(self, h_work: torch.Tensor, h_norm: torch.Tensor) -> torch.Tensor:
        """Shield 4a — WISE activation scope gate (`[B, T]` bool)."""
        bsz, seq_len, _ = h_work.shape
        if self.scope_prototypes.shape[0] == 0 or self.config.scope_threshold <= 0.0:
            return torch.ones((bsz, seq_len), dtype=torch.bool, device=h_work.device)

        h_unit = torch.where(h_norm > 1e-12, h_work / h_norm.clamp(min=1e-12), torch.zeros_like(h_work))
        p_work = self.scope_prototypes.to(device=h_work.device, dtype=h_work.dtype)
        p_norm = p_work.norm(dim=-1, keepdim=True)
        p_valid = p_norm.squeeze(-1) > 1e-12
        if not bool(p_valid.any()):
            return torch.zeros((bsz, seq_len), dtype=torch.bool, device=h_work.device)
        p_unit = p_work[p_valid] / p_norm[p_valid]
        max_cos = (h_unit @ p_unit.T).amax(dim=-1)
        return (h_norm.squeeze(-1) > 1e-12) & (max_cos >= self.config.scope_threshold)

    def _compute_active_pos_mask(
        self,
        bsz: int,
        seq_len: int,
        *,
        device: torch.device,
        positions: torch.Tensor | None,
        pos_start: int,
        min_active_pos: int | None,
    ) -> torch.Tensor:
        """Shield 2 — Prompt-prefix suppression mask (`pos >= min_active_pos`)."""
        eff_min = self.config.min_active_pos if min_active_pos is None else int(min_active_pos)
        if eff_min is None or eff_min <= 0:
            return torch.ones((bsz, seq_len), dtype=torch.bool, device=device)
        if positions is not None:
            pos = positions.to(device=device)
            if pos.ndim == 1:
                if pos.shape[0] != seq_len:
                    raise ValueError(f"1D positions length {pos.shape[0]} != seq_len {seq_len}")
                pos = pos.unsqueeze(0).expand(bsz, seq_len)
            elif pos.ndim == 2:
                if pos.shape != (bsz, seq_len):
                    raise ValueError(f"2D positions shape {tuple(pos.shape)} != ({bsz}, {seq_len})")
            else:
                raise ValueError(f"positions must be 1D or 2D, got shape {tuple(pos.shape)}")
        else:
            pos = torch.arange(pos_start, pos_start + seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(
                bsz, seq_len
            )
        return pos >= eff_min

    def param_summary(
        self,
        *,
        num_passes: int | None = None,
        active_experts: int | None = None,
        active_top_k: int | None = None,
    ) -> dict[str, int | float]:
        """Exact parameter count breakdown for this single anchor."""
        passes = self.config.num_passes if num_passes is None else int(num_passes)
        eff_experts_opt = self.config.active_experts if active_experts is None else active_experts
        eff_experts = self.config.num_experts if eff_experts_opt is None else int(eff_experts_opt)
        raw_top_k_opt = self.config.active_top_k if active_top_k is None else active_top_k
        raw_top_k = self.config.top_k if raw_top_k_opt is None else int(raw_top_k_opt)
        eff_top_k = min(raw_top_k, eff_experts)

        d = self.config.d_model
        routed_per_exp = 3 * d * self.config.moe_intermediate_size
        routed_total = self.config.num_experts * routed_per_exp
        shared_total = 3 * d * self.config.shared_intermediate_size
        router_total = self.config.num_experts * d + self.config.num_experts + 1
        qk_conv_total = 2 * self.config.tier1_dim * d + (3 + self.config.conv_kernel_size) * d + self.config.num_passes
        tier1_total = (
            (self.config.tier1a_buckets + self.config.tier1b_buckets) * self.config.tier1_dim
            if self.own_tier1
            else 0
        )
        moe_total = routed_total + shared_total + router_total
        total_params = moe_total + qk_conv_total + tier1_total

        routed_active = passes * eff_top_k * routed_per_exp
        shared_active = passes * shared_total if eff_experts > 0 else 0
        router_active = passes * (eff_experts * d + eff_experts + 1) if eff_experts > 0 else 0
        moe_active = routed_active + shared_active + router_active
        return {
            "anchor_layer": self.anchor_layer,
            "own_tier1": int(self.own_tier1),
            "moe_total_params": moe_total,
            "tier1_table_params": tier1_total,
            "qk_conv_total_params": qk_conv_total,
            "total_params": total_params,
            "moe_active_params": moe_active,
            "conv_state_bytes": self.conv_state_bytes(),
        }

    def forward(
        self,
        h: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        e_ngram: torch.Tensor | None = None,
        pass_idx: int = 0,
        positions: torch.Tensor | None = None,
        min_active_pos: int | None = None,
        active_experts: int | None = None,
        active_top_k: int | None = None,
        use_conv_cache: bool = False,
        conv_state: EngramConvState | None = None,
        force_module_drop: bool | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, EngramForwardAux]:
        """Compute 5-shielded Engram residual perturbation `delta(h)`.

        Caller applies `h_out = h + delta`. At Step 0 (`gamma == 0.0`), `delta` is `torch.equal` to
        `torch.zeros_like(h)` (`max_abs_diff == 0.0`) with finite gradients flowing to `gamma`.
        """
        if h.ndim not in (2, 3):
            raise ValueError(f"h must be 2D [T, D] or 3D [B, T, D], got shape {tuple(h.shape)}")
        if h.shape[-1] != self.config.d_model:
            raise ValueError(f"expected last dim d_model={self.config.d_model}, got {h.shape[-1]}")

        p_idx = int(pass_idx)
        if not (0 <= p_idx < self.config.num_passes):
            raise ValueError(f"pass_idx must be in [0, {self.config.num_passes}), got {p_idx}")

        squeezed = h.ndim == 2
        h_3d = h.unsqueeze(0) if squeezed else h
        bsz, seq_len, _ = h_3d.shape

        if seq_len == 0:
            zero_out = torch.zeros_like(h)
            if not return_aux:
                return zero_out
            h_w = upcast(h_3d)
            route_empty = self.route(h_w, active_experts=0, active_top_k=0)
            aux = EngramForwardAux(
                alpha=h_w.new_zeros((bsz, 0, 1)),
                route=route_empty,
                in_scope_mask=torch.ones((bsz, 0), dtype=torch.bool, device=h.device),
                active_pos_mask=torch.ones((bsz, 0), dtype=torch.bool, device=h.device),
                module_keep_mask=torch.ones((bsz, 1, 1), dtype=torch.bool, device=h.device),
                pre_cap_rel_norm=h_w.new_zeros((bsz, 0)),
                post_cap_rel_norm=h_w.new_zeros((bsz, 0)),
            )
            return zero_out, aux

        # Resolve streaming state cursor and prior 2-token window if `use_conv_cache=True`
        cached_st = conv_state if conv_state is not None else self._conv_states.get(p_idx)
        pos_start = cached_st.pos_cursor if (use_conv_cache and cached_st is not None) else 0
        prev_tokens = cached_st.prev_tokens if (use_conv_cache and cached_st is not None) else None

        # Resolve Tier-1 causal lexical embeddings `e_ngram`
        token_window: torch.Tensor | None = None
        if input_ids is not None:
            ids_2d = input_ids.unsqueeze(0) if (squeezed and input_ids.ndim == 1) else input_ids
            if ids_2d.ndim != 2 or ids_2d.shape != (bsz, seq_len):
                raise ValueError(f"input_ids must have shape ({bsz}, {seq_len}), got {tuple(ids_2d.shape)}")
            ids_i64 = ids_2d.to(device=h.device, dtype=torch.int64)
            if prev_tokens is None:
                pt_win = torch.tensor(
                    [_BOS_SENTINEL_PREV2, _BOS_SENTINEL_PREV1],
                    dtype=torch.int64,
                    device=h.device,
                ).unsqueeze(0).expand(bsz, 2)
            else:
                if prev_tokens.shape != (bsz, 2):
                    raise ValueError(
                        f"cached prev_tokens shape {tuple(prev_tokens.shape)} mismatches batch {bsz}; "
                        "call reset_conv_state() first"
                    )
                pt_win = prev_tokens.to(device=h.device, dtype=torch.int64)
            token_window = torch.cat([pt_win, ids_i64], dim=1)
            if e_ngram is None:
                if self.tier1 is None:
                    raise ValueError(
                        "e_ngram must be provided when calling an anchor with own_tier1=False and input_ids!=None"
                    )
                e_ngram = self.tier1(ids_i64, prev_tokens=prev_tokens)

        h_work = upcast(h_3d)
        if e_ngram is None:
            e_3d = h_work.new_zeros((bsz, seq_len, self.config.tier1_dim))
        else:
            e_3d = e_ngram.unsqueeze(0) if (squeezed and e_ngram.ndim == 2) else e_ngram
            if e_3d.shape != (bsz, seq_len, self.config.tier1_dim):
                raise ValueError(
                    f"e_ngram must have shape ({bsz}, {seq_len}, {self.config.tier1_dim}), got {tuple(e_3d.shape)}"
                )

        # Shield 3: DeepSeek Signed-Sqrt RMSNorm QK Gate + zero-init causal depthwise Conv1d(k=4)
        alpha, gated_v = self.compute_qk_gate(h_work, e_3d)
        lex_feat = self._apply_causal_conv1d(
            gated_v,
            pass_idx=p_idx,
            use_conv_cache=use_conv_cache,
            conv_state=conv_state,
            token_window=token_window,
            pos_start=pos_start,
            cache_dtype=h.dtype,
        )

        # Shield 5: Independent Sigmoid + Positive Null Sink + Matryoshka Prefix Slicing
        moe_in = h_work + lex_feat
        route_out = self.route(moe_in, active_experts=active_experts, active_top_k=active_top_k)

        routed_out = self._run_routed_experts(moe_in, route_out.top_indices, route_out.top_weights)
        if route_out.active_experts > 0 and bool((route_out.total_gate > 0.0).any()):
            shared_out = self._run_shared_expert(moe_in)
            moe_out = routed_out + route_out.total_gate * (shared_out + lex_feat)
        else:
            # Keep autograd path connected to `total_gate` / `routed_out` while preserving exact `0.0`
            moe_out = routed_out + route_out.total_gate * lex_feat

        # Shield 1: Step-0 exact identity via learnable per-(anchor, pass) residual gate `gamma`
        gamma_p = self.gamma[p_idx].to(device=h_work.device, dtype=h_work.dtype)
        delta = gamma_p * moe_out

        # Shield 4b: In-manifold trust-region relative norm cap `||delta||_2 <= max_rel_delta_norm * ||h||_2`
        # Use clamped sum-of-squares inside sqrt so backward at exact `delta == 0` never produces `0 * NaN`!
        h_norm = torch.sqrt(h_work.square().sum(dim=-1, keepdim=True).clamp(min=1e-24))
        h_nonzero = h_work.square().sum(dim=-1, keepdim=True) > 0.0
        h_norm_exact = torch.where(h_nonzero, h_norm, torch.zeros_like(h_norm))
        max_norm = self.config.max_rel_delta_norm * h_norm_exact

        delta_sq = delta.square().sum(dim=-1, keepdim=True)
        delta_norm_safe = torch.sqrt(delta_sq.clamp(min=1e-24))
        pre_cap_rel = torch.where(
            h_nonzero,
            torch.where(delta_sq > 0.0, delta_norm_safe, torch.zeros_like(delta_norm_safe)) / h_norm,
            torch.zeros_like(delta_norm_safe),
        ).squeeze(-1)

        needs_cap = delta_norm_safe > max_norm
        scale = torch.where(needs_cap, max_norm / delta_norm_safe, torch.ones_like(delta_norm_safe))
        delta = delta * scale

        # Shield 4a: WISE activation scope gate (exact 0.0 on out-of-scope tokens)
        in_scope_mask = self._compute_scope_mask(h_work, h_norm_exact)
        delta = torch.where(in_scope_mask.unsqueeze(-1), delta, torch.zeros_like(delta))

        # Shield 2: Prompt-prefix suppression (exact 0.0 on `pos < min_active_pos`)
        active_pos_mask = self._compute_active_pos_mask(
            bsz,
            seq_len,
            device=h_work.device,
            positions=positions,
            pos_start=pos_start,
            min_active_pos=min_active_pos,
        )
        delta = torch.where(active_pos_mask.unsqueeze(-1), delta, torch.zeros_like(delta))

        # Shield 5 (CE_off guard): Whole-module dropout during training (or forced drop for ablation)
        if force_module_drop is True:
            module_keep_mask = torch.zeros((bsz, 1, 1), dtype=torch.bool, device=h_work.device)
        elif force_module_drop is False or not self.training or self.config.module_drop_prob <= 0.0:
            module_keep_mask = torch.ones((bsz, 1, 1), dtype=torch.bool, device=h_work.device)
        elif self.config.module_drop_prob >= 1.0:
            module_keep_mask = torch.zeros((bsz, 1, 1), dtype=torch.bool, device=h_work.device)
        else:
            module_keep_mask = torch.rand((bsz, 1, 1), device=h_work.device) >= self.config.module_drop_prob
        delta = torch.where(module_keep_mask, delta, torch.zeros_like(delta))

        # Normalize `-0.0` to `+0.0` so `torch.equal` and bit-exact checks are clean
        delta = delta + 0.0

        post_sq = delta.square().sum(dim=-1, keepdim=True)
        post_cap_rel = torch.where(
            h_nonzero,
            torch.where(post_sq > 0.0, torch.sqrt(post_sq.clamp(min=1e-24)), torch.zeros_like(post_sq)) / h_norm,
            torch.zeros_like(post_sq),
        ).squeeze(-1)

        out = delta.to(dtype=h.dtype)
        if squeezed:
            out = out.squeeze(0)

        if not return_aux:
            return out

        aux = EngramForwardAux(
            alpha=alpha,
            route=route_out,
            in_scope_mask=in_scope_mask,
            active_pos_mask=active_pos_mask,
            module_keep_mask=module_keep_mask,
            pre_cap_rel_norm=pre_cap_rel,
            post_cap_rel_norm=post_cap_rel,
        )
        return out, aux


class AnchoredMoEEngramBank(nn.Module):
    """Multi-layer `AnchoredMoEEngramBank` managing `6` core `ZipfianTieredMoEEngram` anchors (`L15..L35`).

    Owns one shared `Tier1NgramTable` (avoiding `6x` lexical-table duplication) and exposes:
      * `export_ngram_cues(input_ids)` for zero-SwiGLU `O(1)` Option-B `L35` `EAGLE-3` draft conditioning
      * `rollback_conv_state(keep_tokens)` across anchors (`12 KiB/anchor` rolling buffer)
      * `param_summary()` matching verified Compact (`1.25B/+0.529B`), Native-Grain (`1.25B/+0.379B`),
        and Full Spark-X (`1.89B/+0.945B`) formulas and exact instantiated `sum(p.numel())` counts.
    """

    def __init__(self, config: EngramConfig | None = None, *, loop_spec: LoopSpec | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else EngramConfig()
        self.config.validate(loop_spec=loop_spec)

        self.tier1 = Tier1NgramTable(
            tier1a_buckets=self.config.tier1a_buckets,
            tier1b_buckets=self.config.tier1b_buckets,
            tier1_dim=self.config.tier1_dim,
        )
        self.anchors = nn.ModuleDict(
            {
                f"L{layer}": ZipfianTieredMoEEngram(self.config, anchor_layer=layer, own_tier1=False)
                for layer in self.config.anchor_layers
            }
        )

    @property
    def anchor_layers(self) -> tuple[int, ...]:
        return self.config.anchor_layers

    def has_layer(self, layer_idx: int) -> bool:
        return f"L{int(layer_idx)}" in self.anchors

    def anchor(self, layer_idx: int) -> ZipfianTieredMoEEngram:
        key = f"L{int(layer_idx)}"
        if key not in self.anchors:
            raise KeyError(f"layer {layer_idx} is not anchored; valid layers: {self.config.anchor_layers}")
        mod: ZipfianTieredMoEEngram = self.anchors[key]  # type: ignore[assignment]
        return mod

    def export_ngram_cues(
        self,
        input_ids: torch.Tensor,
        *,
        prev_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return `O(1)` Tier-1A/1B lexical embeddings `e_ngram` without running routed/shared SwiGLU experts."""
        return self.tier1(input_ids, prev_tokens=prev_tokens)

    def conv_state_bytes_per_anchor(self, batch_size: int = 1, dtype_bytes: int = BF16_DTYPE_BYTES) -> int:
        return self.config.conv_state_bytes_per_anchor(batch_size=batch_size, dtype_bytes=dtype_bytes)

    def conv_state_bytes(self, batch_size: int = 1, dtype_bytes: int = BF16_DTYPE_BYTES) -> int:
        return self.config.conv_state_bytes(batch_size=batch_size, dtype_bytes=dtype_bytes)

    def param_summary(
        self,
        *,
        num_passes: int | None = None,
        active_experts: int | None = None,
        active_top_k: int | None = None,
    ) -> dict[str, int | float]:
        return self.config.param_summary(
            num_passes=num_passes,
            active_experts=active_experts,
            active_top_k=active_top_k,
        )

    @torch.no_grad()
    def register_scope_prototypes(
        self,
        layer_idx: int,
        prototypes: torch.Tensor,
        *,
        replace: bool = False,
    ) -> torch.Tensor:
        return self.anchor(layer_idx).register_scope_prototypes(prototypes, replace=replace)

    @torch.no_grad()
    def clear_scope_prototypes(self, layer_idx: int | None = None) -> None:
        if layer_idx is not None:
            self.anchor(layer_idx).clear_scope_prototypes()
            return
        for mod in self.anchors.values():
            anchor: ZipfianTieredMoEEngram = mod  # type: ignore[assignment]
            anchor.clear_scope_prototypes()

    def reset_conv_state(self, layer_idx: int | None = None, *, pass_idx: int | None = None) -> None:
        if layer_idx is not None:
            self.anchor(layer_idx).reset_conv_state(pass_idx=pass_idx)
            return
        for mod in self.anchors.values():
            anchor: ZipfianTieredMoEEngram = mod  # type: ignore[assignment]
            anchor.reset_conv_state(pass_idx=pass_idx)

    @torch.no_grad()
    def rollback_conv_state(
        self,
        keep_tokens: int,
        *,
        layer_idx: int | None = None,
        pass_idx: int | None = None,
    ) -> dict[int, EngramConvState | dict[int, EngramConvState]] | EngramConvState | dict[int, EngramConvState]:
        """`O(1)` rollback of `conv1d(k=4)` state for one anchor (`layer_idx`) or all active anchors."""
        if layer_idx is not None:
            return self.anchor(layer_idx).rollback_conv_state(keep_tokens, pass_idx=pass_idx)
        rolled: dict[int, EngramConvState | dict[int, EngramConvState]] = {}
        for lyr in self.config.anchor_layers:
            anc = self.anchor(lyr)
            if anc._conv_states:
                rolled[lyr] = anc.rollback_conv_state(keep_tokens, pass_idx=pass_idx)
        if not rolled:
            raise RuntimeError("no active anchor conv states to roll back; run forward(..., use_conv_cache=True)")
        return rolled

    def forward(
        self,
        layer_idx: int,
        h: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        e_ngram: torch.Tensor | None = None,
        pass_idx: int = 0,
        positions: torch.Tensor | None = None,
        min_active_pos: int | None = None,
        active_experts: int | None = None,
        active_top_k: int | None = None,
        use_conv_cache: bool = False,
        conv_state: EngramConvState | None = None,
        force_module_drop: bool | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, EngramForwardAux]:
        anc = self.anchor(layer_idx)
        if e_ngram is None and input_ids is not None:
            cached_st = conv_state if conv_state is not None else anc.get_conv_state(pass_idx)
            prev_tokens = cached_st.prev_tokens if (use_conv_cache and cached_st is not None) else None
            e_ngram = self.export_ngram_cues(input_ids, prev_tokens=prev_tokens)
        return anc(
            h,
            input_ids=input_ids,
            e_ngram=e_ngram,
            pass_idx=pass_idx,
            positions=positions,
            min_active_pos=min_active_pos,
            active_experts=active_experts,
            active_top_k=active_top_k,
            use_conv_cache=use_conv_cache,
            conv_state=conv_state,
            force_module_drop=force_module_drop,
            return_aux=return_aux,
        )


__all__ = [
    "BF16_DTYPE_BYTES",
    "DEFAULT_CORE_LOOP_SPEC",
    "NVFP4_EFFECTIVE_BYTES_PER_PARAM",
    "AnchoredMoEEngramBank",
    "EngramConfig",
    "EngramConvState",
    "EngramForwardAux",
    "EngramRouteOutput",
    "Tier1NgramTable",
    "ZipfianTieredMoEEngram",
    "compute_causal_ngram_hashes",
    "signed_sqrt",
]
