"""Two-Tier Ultra-Long Context KV Compression & Host-DDR5 Offload (Fixes C2, D1-D4 & Theorem 3).

Implements both tiers of Pillar III (`ARCHITECTURE_SPEC.md` §6.2-§6.3, `RESEARCH_ESSAY.md` §5):
  * **Tier 1 (`1M` GPU-Resident `FP8`, `10.00 GiB`):** `FP8KVCacheQuantizer` (`float8_e4m3fn` when
    supported on the target device, with automatic `float16` fallback on older CPU builds).
  * **Tier 2 (`10M` Host-DDR5 Block-64 Hadamard `INT4`, `53.64 GiB`):**
    - `walsh_hadamard_block(x, block_size=64)`: self-adjoint orthonormal `H_64` per 64-d group,
      strictly isolating `[0:64]` mRoPE channels from `[64:256]` NoPE content channels (Fix C2)
      and dispersing channel outliers by `1/8` with zero QJL variance penalty (Theorem 3).
    - `Block64HadamardINT4Cache`: packs 2 unsigned 4-bit nibbles per `uint8` (`32 B`) + `fp16`
      scale (`2 B`) + `fp16` zero/min (`2 B`) = **exact `36 B / 64 scalars` (`0.5625 B/scalar`)**
      and **exact `2,304 B` per 4-token x 2-KV-head x 256-d K&V micro-page** in CPU memory
      (`pin_memory=True` when CUDA is available), with 3-layer-ahead speculative async stream
      prefetch (`prefetch_micro_pages`) and single-`H_64` post-aggregation value reconstruction
      (`gather_kv_and_reconstruct_v`).
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from ._dtype import upcast

GROUP_SIZE_DEFAULT = 64
GROUP_PACKED_BYTES = 36  # 32B packed nibbles + 2B fp16 scale + 2B fp16 min/zero


def walsh_hadamard_block(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Self-adjoint orthonormal block-diagonal Walsh-Hadamard `H_{block_size}` along last dim.

    Satisfies `H^T = H` and `H^2 = I`. With `block_size = 64` on `d = 256`, Group 0 transforms
    only `[0:64]` (mRoPE) while Groups 1..3 transform only `[64:256]` (NoPE), guaranteeing zero
    leakage between rotary and content subspaces (`Fix C2`).
    """
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError(f"block_size must be a positive power of 2, got {block_size}")
    d = x.shape[-1]
    if d % block_size != 0:
        raise ValueError(f"last dim {d} must be divisible by block_size={block_size}")
    orig_dtype = x.dtype
    work = upcast(x)
    base_shape = (*work.shape[:-1], d // block_size)
    h = work.reshape(*base_shape, block_size).clone()
    step = 1
    while step < block_size:
        h = h.view(*base_shape, block_size // (2 * step), 2, step)
        a = h[..., 0, :]
        b = h[..., 1, :]
        h = torch.cat([a + b, a - b], dim=-1).reshape(*base_shape, block_size)
        step *= 2
    out = h.reshape_as(work) / math.sqrt(float(block_size))
    return out.to(orig_dtype)


def pack_group64_int4(x_h: torch.Tensor, group_size: int = GROUP_SIZE_DEFAULT) -> torch.Tensor:
    """Pack `[..., D]` tensor into `[..., D // 64, 36]` `uint8` Group-64 INT4 blocks (`0.5625 B/scalar`)."""
    if group_size != GROUP_SIZE_DEFAULT:
        raise ValueError(f"Group-64 packing requires group_size=64, got {group_size}")
    d = x_h.shape[-1]
    if d % group_size != 0:
        raise ValueError(f"last dim {d} must be divisible by group_size={group_size}")
    n_groups = d // group_size
    xg = x_h.float().reshape(*x_h.shape[:-1], n_groups, group_size)

    x_min_f16 = xg.min(dim=-1, keepdim=True).values.to(torch.float16)
    x_max_f16 = xg.max(dim=-1, keepdim=True).values.to(torch.float16)
    scale_f16 = ((x_max_f16.float() - x_min_f16.float()) / 15.0).to(torch.float16)
    denom = scale_f16.float().clamp(min=1e-7)

    q = torch.clamp(torch.round((xg - x_min_f16.float()) / denom), 0.0, 15.0).to(torch.uint8)
    nib_lo = q[..., 0::2] & 0x0F
    nib_hi = (q[..., 1::2] & 0x0F) << 4
    nib_bytes = nib_hi | nib_lo  # [..., n_groups, 32] uint8

    scale_bytes = scale_f16.contiguous().view(torch.uint8)  # [..., n_groups, 2] uint8
    min_bytes = x_min_f16.contiguous().view(torch.uint8)    # [..., n_groups, 2] uint8
    return torch.cat([nib_bytes, scale_bytes, min_bytes], dim=-1).contiguous()


def unpack_group64_int4(
    packed: torch.Tensor,
    group_size: int = GROUP_SIZE_DEFAULT,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpack `[..., n_groups, 36]` `uint8` tensor back to `[..., n_groups * 64]` in `dtype`."""
    if group_size != GROUP_SIZE_DEFAULT:
        raise ValueError(f"Group-64 unpacking requires group_size=64, got {group_size}")
    if packed.dtype != torch.uint8 or packed.shape[-1] != GROUP_PACKED_BYTES:
        raise ValueError(f"expected uint8 tensor with last dim 36, got {packed.dtype} {tuple(packed.shape)}")

    nib_bytes = packed[..., :32]
    scale_f16 = packed[..., 32:34].contiguous().view(torch.float16)  # [..., n_groups, 1]
    min_f16 = packed[..., 34:36].contiguous().view(torch.float16)    # [..., n_groups, 1]

    q_lo = nib_bytes & 0x0F
    q_hi = (nib_bytes >> 4) & 0x0F
    q = torch.stack([q_lo, q_hi], dim=-1).reshape(*packed.shape[:-1], group_size)

    acc_dt = torch.float64 if dtype == torch.float64 else torch.float32
    xg_hat = q.to(acc_dt) * scale_f16.to(acc_dt) + min_f16.to(acc_dt)
    return xg_hat.reshape(*packed.shape[:-2], packed.shape[-2] * group_size).to(dtype)


@dataclass
class _PrefetchState:
    page_indices_cpu: torch.Tensor
    staged_pages: torch.Tensor
    cuda_event: Any | None = None


class Block64HadamardINT4Cache:
    """Host-DDR5 pinned Group-64 Block-Hadamard INT4 micro-page KV cache (`Gate 2b` 10M Tier 2).

    Storage layout per layer:
      `pages[layer_idx]` is a contiguous CPU `torch.uint8` tensor of shape `[capacity_pages, page_bytes]`
      where each row packs `[2 (K, V), page_tokens, num_kv_heads, head_dim // 64, 36]` bytes.
      For Qwen3.6 (`page_tokens=4, num_kv_heads=2, head_dim=256`):
        `2 * 4 * 2 * 4 * 36 = 2,304 Bytes` per 4-token micro-page (`0.5625 B/scalar` = `53.64 GiB` @ `10M`).
    """

    def __init__(
        self,
        num_layers: int = 10,
        num_kv_heads: int = 2,
        head_dim: int = 256,
        page_tokens: int = 4,
        block_size: int = 64,
        initial_pages: int = 64,
        pin_memory: bool | None = None,
        device: torch.device | str | None = None,
        *,
        micro_page_tokens: int | None = None,
        prefetch_layers_ahead: int = 3,
    ) -> None:
        pt = int(page_tokens if micro_page_tokens is None else micro_page_tokens)
        if num_layers < 1 or num_kv_heads < 1 or pt < 1:
            raise ValueError("num_layers, num_kv_heads, and page_tokens must be >= 1")
        if block_size != GROUP_SIZE_DEFAULT or head_dim % block_size != 0:
            raise ValueError(f"head_dim={head_dim} must be a multiple of block_size=64 (got {block_size})")

        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_tokens = pt
        self.block_size = int(block_size)
        self.prefetch_layers_ahead = int(prefetch_layers_ahead)
        self.num_groups = self.head_dim // self.block_size
        self._page_shape = (2, self.page_tokens, self.num_kv_heads, self.num_groups, GROUP_PACKED_BYTES)
        self._page_bytes = 2 * self.page_tokens * self.num_kv_heads * self.num_groups * GROUP_PACKED_BYTES

        cuda_ok = torch.cuda.is_available()
        self.pin_memory = bool(cuda_ok if pin_memory is None else (pin_memory and cuda_ok))
        if device is None:
            self.target_device = torch.device("cuda" if cuda_ok else "cpu")
        else:
            self.target_device = torch.device(device)

        init_cap = max(1, int(initial_pages))
        self._pages: list[torch.Tensor] = [self._alloc_host_pages(init_cap) for _ in range(self.num_layers)]
        self._num_tokens: list[int] = [0] * self.num_layers
        # Unsealed tail tokens (< page_tokens) kept in float32 on CPU so partial decode steps seal cleanly
        self._tail_k: list[torch.Tensor | None] = [None] * self.num_layers
        self._tail_v: list[torch.Tensor | None] = [None] * self.num_layers

        self._prefetch_stream: Any | None = (
            torch.cuda.Stream(device=self.target_device)
            if (cuda_ok and self.target_device.type == "cuda")
            else None
        )
        self._prefetched: dict[int, _PrefetchState] = {}
        self.last_hadamard_v_calls: int = 0
        self.stats: dict[str, int] = {
            "prefetch_calls": 0,
            "prefetch_hits": 0,
            "residual_miss_pages": 0,
            "prefetched_bytes": 0,
        }

    @property
    def micro_page_bytes(self) -> int:
        """Exact byte footprint of a single `page_tokens` K+V micro-page (`2,304 B` for Qwen3.6)."""
        return self._page_bytes

    @property
    def bytes_per_scalar(self) -> float:
        """Exact storage density in bytes per scalar (`36 / 64 = 0.5625 B/scalar`)."""
        return float(GROUP_PACKED_BYTES) / float(self.block_size)

    def estimate_cache_bytes(self, num_tokens: int = 10_000_000, num_layers: int | None = None) -> int:
        """Exact Host DDR5 byte footprint for `num_tokens` across `num_layers` full-attention layers."""
        nl = self.num_layers if num_layers is None else int(num_layers)
        n_pages = (int(num_tokens) + self.page_tokens - 1) // self.page_tokens
        return nl * n_pages * self._page_bytes

    def num_tokens(self, layer_idx: int = 0) -> int:
        self._check_layer(layer_idx)
        return self._num_tokens[layer_idx]

    def num_pages(self, layer_idx: int = 0) -> int:
        self._check_layer(layer_idx)
        return (self._num_tokens[layer_idx] + self.page_tokens - 1) // self.page_tokens

    def raw_micro_page(self, layer_idx: int, page_idx: int) -> torch.Tensor:
        """Return a view of packed `uint8` micro-page `page_idx` (`shape == [micro_page_bytes]`)."""
        self._check_layer(layer_idx)
        npages = self.num_pages(layer_idx)
        if not 0 <= page_idx < npages:
            raise IndexError(f"page_idx={page_idx} out of range [0, {npages})")
        return self._pages[layer_idx][page_idx]

    def _check_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx={layer_idx} out of range [0, {self.num_layers})")

    def _alloc_host_pages(self, n_pages: int) -> torch.Tensor:
        buf = torch.empty((n_pages, self._page_bytes), dtype=torch.uint8, device="cpu")
        return buf.pin_memory() if self.pin_memory else buf

    def _ensure_capacity(self, layer_idx: int, needed_pages: int) -> None:
        cur = self._pages[layer_idx]
        if needed_pages <= cur.shape[0]:
            return
        new_cap = max(needed_pages, cur.shape[0] * 2)
        grown = self._alloc_host_pages(new_cap)
        used = self.num_pages(layer_idx)
        if used > 0:
            grown[:used].copy_(cur[:used])
        self._pages[layer_idx] = grown

    def _normalize_kv_input(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if k.shape != v.shape:
            raise ValueError(f"k and v shapes must match, got {tuple(k.shape)} vs {tuple(v.shape)}")
        if k.ndim == 4:
            if k.shape[0] != 1:
                raise ValueError(f"Block64HadamardINT4Cache expects batch_size=1, got {k.shape[0]}")
            k3, v3 = k[0], v[0]
        elif k.ndim == 3:
            k3, v3 = k, v
        else:
            raise ValueError(f"expected k, v with shape [1, G, T, D] or [G, T, D], got {tuple(k.shape)}")
        if k3.shape[0] != self.num_kv_heads or k3.shape[2] != self.head_dim:
            raise ValueError(
                f"expected (G={self.num_kv_heads}, T, D={self.head_dim}), got {tuple(k3.shape)}"
            )
        return k3, v3

    def _pack_pages_block(self, k_gt: torch.Tensor, v_gt: torch.Tensor) -> torch.Tensor:
        """Transform `[G, P*pt, D]` via `H_64` and pack into `[P, micro_page_bytes]` `uint8` on CPU."""
        total_t = k_gt.shape[1]
        assert total_t % self.page_tokens == 0
        n_pages = total_t // self.page_tokens
        # [2, G, P, pt, D] -> [P, 2, pt, G, D]
        kv = torch.stack([k_gt, v_gt], dim=0)
        kv = kv.reshape(2, self.num_kv_heads, n_pages, self.page_tokens, self.head_dim)
        kv_pages = kv.permute(2, 0, 3, 1, 4).contiguous()
        kv_had = walsh_hadamard_block(kv_pages, self.block_size)
        packed = pack_group64_int4(kv_had, self.block_size)  # [P, 2, pt, G, n_groups, 36]
        return packed.reshape(n_pages, self._page_bytes).to(device="cpu")

    def append_tokens(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> int:
        """Rotate `k, v` (`[1, G, T, D]` or `[G, T, D]`) by `H_64` and append Group-64 INT4 micro-pages.

        Supports both multi-token prefill chunks and single-token (`T=1`) incremental decode; any
        partial trailing micro-page (`< page_tokens`) is zero-padded and sealed in-place so every
        appended token is immediately readable, then overwritten cleanly as subsequent tokens arrive.
        Returns the updated total token count for `layer_idx`.
        """
        self._check_layer(layer_idx)
        k3, v3 = self._normalize_kv_input(k, v)
        t_add = k3.shape[1]
        if t_add == 0:
            return self._num_tokens[layer_idx]

        # Invalidate any stale prefetched view for this layer if updated
        self._prefetched.pop(layer_idx, None)

        prev_total = self._num_tokens[layer_idx]
        prev_sealed_pages = prev_total // self.page_tokens
        tail_k, tail_v = self._tail_k[layer_idx], self._tail_v[layer_idx]
        if tail_k is not None and tail_v is not None:
            k_work = torch.cat([tail_k.to(device=k3.device, dtype=k3.dtype), k3], dim=1)
            v_work = torch.cat([tail_v.to(device=v3.device, dtype=v3.dtype), v3], dim=1)
        else:
            k_work, v_work = k3, v3

        new_total = prev_total + t_add
        total_pages = (new_total + self.page_tokens - 1) // self.page_tokens
        self._ensure_capacity(layer_idx, total_pages)

        full_pages = k_work.shape[1] // self.page_tokens
        rem = k_work.shape[1] % self.page_tokens

        if full_pages > 0:
            cut = full_pages * self.page_tokens
            packed_full = self._pack_pages_block(k_work[:, :cut], v_work[:, :cut])
            self._pages[layer_idx][prev_sealed_pages : prev_sealed_pages + full_pages].copy_(packed_full)

        if rem > 0:
            tk = k_work[:, -rem:].detach()
            tv = v_work[:, -rem:].detach()
            self._tail_k[layer_idx] = tk.to(device="cpu", dtype=torch.float32)
            self._tail_v[layer_idx] = tv.to(device="cpu", dtype=torch.float32)
            pad = self.page_tokens - rem
            kp = torch.cat([tk, tk.new_zeros((self.num_kv_heads, pad, self.head_dim))], dim=1)
            vp = torch.cat([tv, tv.new_zeros((self.num_kv_heads, pad, self.head_dim))], dim=1)
            packed_tail = self._pack_pages_block(kp, vp)
            self._pages[layer_idx][prev_sealed_pages + full_pages : prev_sealed_pages + full_pages + 1].copy_(
                packed_tail
            )
        else:
            self._tail_k[layer_idx] = None
            self._tail_v[layer_idx] = None

        self._num_tokens[layer_idx] = new_total
        return new_total

    def prefetch_micro_pages(
        self,
        layer_idx: int,
        page_indices: torch.Tensor,
        target_device: torch.device | str | None = None,
    ) -> None:
        """Asynchronously stage `page_indices` from pinned Host DDR5 onto `target_device` (3-layer ahead)."""
        self._check_layer(layer_idx)
        dev = self.target_device if target_device is None else torch.device(target_device)
        flat_idx = page_indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if flat_idx.numel() > 0:
            uniq_idx = torch.unique(flat_idx, sorted=True)
            npages = self.num_pages(layer_idx)
            if int(uniq_idx[0]) < 0 or int(uniq_idx[-1]) >= npages:
                raise IndexError(f"prefetch page index out of range [0, {npages})")
            host_slice = self._pages[layer_idx].index_select(0, uniq_idx)
            if self.pin_memory and not host_slice.is_pinned():
                host_slice = host_slice.pin_memory()
        else:
            uniq_idx = flat_idx
            host_slice = torch.empty((0, self._page_bytes), dtype=torch.uint8, device="cpu")

        evt = None
        if dev.type == "cuda" and torch.cuda.is_available():
            stream = self._prefetch_stream or torch.cuda.Stream(device=dev)
            with torch.cuda.stream(stream):
                staged = host_slice.to(device=dev, non_blocking=True)
                evt = torch.cuda.Event()
                evt.record(stream)
        else:
            staged = host_slice.to(device=dev, non_blocking=True)

        self._prefetched[layer_idx] = _PrefetchState(uniq_idx, staged, evt)
        self.stats["prefetch_calls"] += 1
        self.stats["prefetched_bytes"] += int(uniq_idx.numel() * self._page_bytes)

    def _gather_packed_pages(
        self,
        layer_idx: int,
        page_indices: torch.Tensor,
        dev: torch.device,
    ) -> torch.Tensor:
        """Fetch `[..., self._page_bytes]` packed micro-pages, reusing prefetched pages + delta misses."""
        orig_idx_shape = page_indices.shape
        req_cpu = page_indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        npages = self.num_pages(layer_idx)
        if req_cpu.numel() == 0:
            return torch.empty((*orig_idx_shape, self._page_bytes), dtype=torch.uint8, device=dev)
        if int(req_cpu.min()) < 0 or int(req_cpu.max()) >= npages:
            raise IndexError(f"page_indices out of range [0, {npages})")

        pre = self._prefetched.pop(layer_idx, None)
        if pre is not None and pre.staged_pages.device == dev and pre.page_indices_cpu.numel() > 0:
            if pre.cuda_event is not None and dev.type == "cuda" and torch.cuda.is_available():
                torch.cuda.current_stream(dev).wait_event(pre.cuda_event)
            pos = torch.searchsorted(pre.page_indices_cpu, req_cpu)
            pos_clamped = pos.clamp(max=pre.page_indices_cpu.numel() - 1)
            hit_mask = pre.page_indices_cpu[pos_clamped] == req_cpu
            if bool(hit_mask.all()):
                self.stats["prefetch_hits"] += 1
                gathered = pre.staged_pages.index_select(0, pos_clamped.to(dev))
                return gathered.reshape(*orig_idx_shape, self._page_bytes)
            # Partial hit: reuse prefetched pages and synchronously fetch ONLY residual delta misses
            self.stats["prefetch_hits"] += 1
            out_flat = torch.empty((req_cpu.numel(), self._page_bytes), dtype=torch.uint8, device=dev)
            hit_loc = hit_mask.nonzero(as_tuple=True)[0]
            miss_loc = (~hit_mask).nonzero(as_tuple=True)[0]
            if hit_loc.numel() > 0:
                out_flat[hit_loc.to(dev)] = pre.staged_pages.index_select(0, pos_clamped[hit_loc].to(dev))
            if miss_loc.numel() > 0:
                miss_pages = req_cpu[miss_loc]
                self.stats["residual_miss_pages"] += int(miss_pages.unique().numel())
                sync_fetch = self._pages[layer_idx].index_select(0, miss_pages).to(device=dev)
                out_flat[miss_loc.to(dev)] = sync_fetch
            return out_flat.reshape(*orig_idx_shape, self._page_bytes)

        fetched = self._pages[layer_idx].index_select(0, req_cpu).to(device=dev)
        return fetched.reshape(*orig_idx_shape, self._page_bytes)

    def gather_hadamard_kv(
        self,
        layer_idx: int,
        page_indices: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
        target_device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather & dequantize `(k_h_hat, v_h_hat)` in Hadamard space without inverse `H_64`.

        Shapes:
          * If `page_indices` is 1D `[P]`: returns `(k_h_hat, v_h_hat)` of shape `[1, G, P*page_tokens, D]`.
          * If `page_indices` is 2D `[G, P]`: returns `[1, G, P*page_tokens, D]` (per-KV-head page selection).
          * If `page_indices` is 3D `[B, G, P]`: returns `[B, G, P*page_tokens, D]`.
        """
        self._check_layer(layer_idx)
        dev = self.target_device if target_device is None else torch.device(target_device)
        packed = self._gather_packed_pages(layer_idx, page_indices, dev)

        if page_indices.ndim == 1:
            p_cnt = page_indices.shape[0]
            # [P, 2, pt, G, n_groups, 36]
            structured = packed.reshape(p_cnt, *self._page_shape)
            deq = unpack_group64_int4(structured, self.block_size, dtype=dtype)  # [P, 2, pt, G, D]
            # -> [2, 1, G, P * pt, D]
            deq = deq.permute(1, 3, 0, 2, 4).reshape(2, 1, self.num_kv_heads, p_cnt * self.page_tokens, self.head_dim)
            return deq[0].contiguous(), deq[1].contiguous()

        if page_indices.ndim in (2, 3):
            added_b = page_indices.ndim == 2
            idx3 = page_indices.unsqueeze(0) if added_b else page_indices
            bsz, g_cnt, p_cnt = idx3.shape
            if g_cnt != self.num_kv_heads:
                raise ValueError(f"expected G={self.num_kv_heads} in page_indices, got {g_cnt}")
            structured = packed.reshape(bsz, g_cnt, p_cnt, *self._page_shape)
            # Select each KV head g's own slice from dim -3 (the G dim inside the micro-page!)
            g_sel = torch.arange(g_cnt, device=dev).view(1, g_cnt, 1, 1, 1, 1, 1, 1)
            g_sel = g_sel.expand(bsz, g_cnt, p_cnt, 2, self.page_tokens, 1, self.num_groups, GROUP_PACKED_BYTES)
            head_packed = structured.gather(5, g_sel).squeeze(5)  # [B, G, P, 2, pt, n_groups, 36]
            deq = unpack_group64_int4(head_packed, self.block_size, dtype=dtype)  # [B, G, P, 2, pt, D]
            deq = deq.permute(3, 0, 1, 2, 4, 5).reshape(2, bsz, g_cnt, p_cnt * self.page_tokens, self.head_dim)
            return deq[0].contiguous(), deq[1].contiguous()

        raise ValueError(f"page_indices must be 1D [P], 2D [G, P], or 3D [B, G, P], got {tuple(page_indices.shape)}")

    def gather_kv_and_reconstruct_v(
        self,
        layer_idx: int,
        page_indices: torch.Tensor,
        attn_weights: torch.Tensor | None = None,
        *,
        dtype: torch.dtype | None = None,
        target_device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather `(k_h_hat, v_h_hat)` and reconstruct aggregated `v` via ONE `walsh_hadamard_block`.

        * Queries rotated once by `walsh_hadamard_block(q, 64)` score `k_h_hat` directly in Hadamard
          space (`<H_64 q, H_64 k> == <q, k>`).
        * By self-adjoint linearity of `H_64`, `sum_j alpha_j (H_64 v_h_hat_j) == H_64(sum_j alpha_j v_h_hat_j)`,
          requiring only **1** `walsh_hadamard_block(..., 64)` call after weighted attention aggregation!
        * Returns `(k_h_hat, v_reconstructed)` when `attn_weights` is provided, or `(k_h_hat, v_h_hat)`
          when `attn_weights is None`.
        """
        out_dt = (attn_weights.dtype if attn_weights is not None else torch.float32) if dtype is None else dtype
        dev = (
            attn_weights.device
            if (target_device is None and attn_weights is not None)
            else target_device
        )
        k_h_hat, v_h_hat = self.gather_hadamard_kv(layer_idx, page_indices, dtype=out_dt, target_device=dev)
        if attn_weights is None:
            self.last_hadamard_v_calls = 0
            return k_h_hat, v_h_hat

        w = upcast(attn_weights)
        vh = upcast(v_h_hat)
        bsz, g_cnt, n_tok, d = vh.shape
        if w.shape[-1] != n_tok:
            raise ValueError(
                f"attn_weights last dim ({w.shape[-1]}) must match gathered token count ({n_tok})"
            )

        if w.ndim == 1:
            agg_h = torch.einsum("n,bgnd->bgd", w, vh)
        elif w.ndim == 2:
            agg_h = (
                torch.einsum("gn,bgnd->bgd", w, vh)
                if w.shape[0] == g_cnt
                else torch.einsum("tn,bgnd->bgtd", w, vh)
            )
        elif w.ndim == 3:
            h_cnt = w.shape[0]
            if h_cnt % g_cnt != 0:
                raise ValueError(f"query heads ({h_cnt}) must be a multiple of num_kv_heads ({g_cnt})")
            hg = h_cnt // g_cnt
            wg = w.reshape(g_cnt, hg, w.shape[1], n_tok)
            agg_h = torch.einsum("ghtn,bgnd->bghtd", wg, vh).reshape(bsz, h_cnt, w.shape[1], d)
            if bsz == 1:
                agg_h = agg_h.squeeze(0)
        elif w.ndim == 4:
            if w.shape[0] != bsz and bsz == 1:
                vh = vh.expand(w.shape[0], -1, -1, -1)
                bsz = w.shape[0]
            h_cnt = w.shape[1]
            if h_cnt % g_cnt != 0:
                raise ValueError(f"query heads ({h_cnt}) must be a multiple of num_kv_heads ({g_cnt})")
            hg = h_cnt // g_cnt
            wg = w.reshape(bsz, g_cnt, hg, w.shape[2], n_tok)
            agg_h = torch.einsum("bghtn,bgnd->bghtd", wg, vh).reshape(bsz, h_cnt, w.shape[2], d)
        else:
            raise ValueError(f"unsupported attn_weights shape {tuple(w.shape)}")

        v_rec = walsh_hadamard_block(agg_h, self.block_size).to(out_dt)
        self.last_hadamard_v_calls = 1
        return k_h_hat, v_rec


@dataclass
class QuantizedFP8KV:
    """Quantized Tier-1 KV tensor (`1.0 B/scalar` in `float8_e4m3fn`, or `float16` fallback).

    Supports both attribute access (`enc.data`, `enc.scale`, `enc.nbytes`) and 2-tuple unpacking
    (`q_data, q_scale = quantizer.quantize(x)`).
    """

    data: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype
    is_fp8_native: bool

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.data
        yield self.scale

    @property
    def nbytes(self) -> int:
        return int(self.data.numel() * self.data.element_size() + self.scale.numel() * self.scale.element_size())


def _probe_fp8_e4m3_support(device: torch.device) -> bool:
    fp8_dt = getattr(torch, "float8_e4m3fn", None)
    if fp8_dt is None:
        return False
    try:
        probe = torch.tensor([1.0, -2.5], device=device, dtype=torch.float32)
        rt = probe.to(fp8_dt).to(torch.float32)
        return bool(torch.isfinite(rt).all())
    except Exception:
        return False


class FP8KVCacheQuantizer:
    """Tier-1 `1M`-Token GPU-Resident `FP8 E4M3` KV Cache Quantizer (`10.00 GiB` @ `1M` tokens).

    Uses native `torch.float8_e4m3fn` (`max = 448.0`, `1 byte / scalar`) with per-head dynamic
    scaling when supported on the tensor's device, and falls back cleanly to `torch.float16` on
    older CPU builds without `float8_e4m3fn` conversion kernels.
    """

    FP8_MAX: float = 448.0

    def __init__(self, per_token: bool = False, force_fp16_fallback: bool = False) -> None:
        self.per_token = bool(per_token)
        self.force_fp16_fallback = bool(force_fp16_fallback)
        self._support_cache: dict[str, bool] = {}

    def supports_fp8(self, device: torch.device | str = "cpu") -> bool:
        if self.force_fp16_fallback:
            return False
        dev = torch.device(device)
        key = str(dev)
        if key not in self._support_cache:
            self._support_cache[key] = _probe_fp8_e4m3_support(dev)
        return self._support_cache[key]

    def quantize(self, x: torch.Tensor) -> QuantizedFP8KV:
        orig_dtype = x.dtype
        use_fp8 = self.supports_fp8(x.device)
        if not use_fp8:
            scale = torch.ones((1,), device=x.device, dtype=torch.float32)
            return QuantizedFP8KV(x.to(torch.float16), scale, orig_dtype, is_fp8_native=False)

        xf = x.float()
        if self.per_token or xf.ndim < 3:
            amax = xf.abs().amax(dim=-1, keepdim=True)
        else:
            # Per-(batch, kv_head) scale -> negligible metadata overhead, exact 1.00 B/scalar!
            reduce_dims = tuple(range(2, xf.ndim))
            amax = xf.abs().amax(dim=reduce_dims, keepdim=True)
        scale = (amax / self.FP8_MAX).clamp(min=1e-8)
        scaled = (xf / scale).clamp(-self.FP8_MAX, self.FP8_MAX)
        q_fp8 = scaled.to(torch.float8_e4m3fn)
        return QuantizedFP8KV(q_fp8, scale.to(torch.float32), orig_dtype, is_fp8_native=True)

    @staticmethod
    def dequantize(
        qkv_or_data: QuantizedFP8KV | torch.Tensor,
        scale_or_dtype: torch.Tensor | torch.dtype | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if isinstance(qkv_or_data, QuantizedFP8KV):
            dt_arg = scale_or_dtype if isinstance(scale_or_dtype, torch.dtype) else dtype
            target_dt = qkv_or_data.orig_dtype if dt_arg is None else dt_arg
            if not qkv_or_data.is_fp8_native:
                return qkv_or_data.data.to(target_dt)
            acc_dt = torch.float64 if target_dt == torch.float64 else torch.float32
            return (qkv_or_data.data.to(acc_dt) * qkv_or_data.scale.to(acc_dt)).to(target_dt)

        if not isinstance(scale_or_dtype, torch.Tensor):
            raise TypeError("dequantize(q_data, q_scale) requires a Tensor scale when first arg is a Tensor")
        target_dt = torch.float32 if dtype is None else dtype
        acc_dt = torch.float64 if target_dt == torch.float64 else torch.float32
        return (qkv_or_data.to(acc_dt) * scale_or_dtype.to(acc_dt)).to(target_dt)

    @staticmethod
    def tier1_1m_ledger_bytes(
        num_tokens: int = 1_048_576,
        num_layers: int = 10,
        num_kv_heads: int = 2,
        head_dim: int = 256,
    ) -> int:
        """Exact `1M`-token `FP8` GPU KV cache payload bytes (`10.00 GiB = 10,737,418,240 B`)."""
        return int(num_tokens) * int(num_layers) * 2 * int(num_kv_heads) * int(head_dim) * 1
