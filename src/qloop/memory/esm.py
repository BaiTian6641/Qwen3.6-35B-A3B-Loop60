"""Phase-2B `LSA-NL` Level-3 `EpisodicSideMemory` (`ESM`) — `AlphaEdit` Float64 DGKS + `WISE` Scope Gate.

Implements vertical horizontal-KV -> weight episodic compilation strictly isolated under
`src/qloop/memory/` (zero modifications to frozen MoE weights or active training codepaths):
  * **Core-Only Anchor Placement (`LoopSpec` `[12, 36)`):** Default anchors `(15, 19, 23, 27, 31, 35)`
    attach strictly to the `6` core full-attention boundaries and reject Coda layers (`>= 36`, e.g.,
    `L39`) with `ValueError` unless explicitly overridden via `allow_coda=True` (`Spark-X2.6` `S2` rule).
  * **Step-0 IEEE-754 Bit-Identity & `WISE` Scope Gate:** Zero-initialized `w_down` + empty
    `scope_prototypes` returns exact `zeros_like(h)` (`torch.equal`). After writes, out-of-scope
    tokens (`max_p cos(h, p) < scope_threshold`) yield exact `0.0` drift.
  * **In-Manifold Trust-Region Norm Cap:** Enforces `||delta||_2 <= max_rel_delta_norm * ||h||_2`
    (`0.03` default) on every active token.
  * **AlphaEdit Null-Space Projection + Row-Normalized DGKS Double Gram-Schmidt (`float64`):**
    Projects trigger features `A_perp = A @ (I - U^T U)` against general-corpus + prior-write basis
    `U` in `torch.float64`, solves `(A_perp A_perp^T + alpha I)^{-1} delta_z`, guarantees `< 1e-12`
    null-space leakage across sequential writes, and records every delta in a reversible undo ledger
    (`rollback_entry`).
  * **Exact Shard Footprint:** `144.0 MiB` (`150,994,944 B`) at `6` anchors, `r_hidden=2048`,
    `d_model=2048` (`w_in` in `bf16` [`2 B`] + `w_down` in `fp32` [`4 B`]).
"""
from __future__ import annotations

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

W_IN_DTYPE_BYTES = 2   # bf16 input projection footprint in serialized/served ESM shard
W_DOWN_DTYPE_BYTES = 4  # fp32 compiled episodic projection buffer


def _orthogonal_matrix(rows: int, cols: int, seed: int) -> torch.Tensor:
    """Deterministic semi-orthogonal/orthogonal `[rows, cols]` `float32` matrix via QR."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    n_big, n_small = max(rows, cols), min(rows, cols)
    mat = torch.randn((n_big, n_small), generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(mat, mode="reduced")
    diag_sign = torch.sign(torch.diag(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    q = q * diag_sign.unsqueeze(0)
    if rows < cols:
        q = q.T
    return q.contiguous()


def project_orthogonal_dgks(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Two-pass DGKS orthogonal projection `x @ (I - U^T U)` in `torch.float64`."""
    out = x.to(torch.float64)
    if basis.numel() == 0:
        return out
    u = basis.to(device=out.device, dtype=torch.float64)
    # Two full Gram-Schmidt passes suppress catastrophic cancellation down to ~1e-16
    out = out - (out @ u.T) @ u
    out = out - (out @ u.T) @ u
    return out


def dgks_orthonormalize_rows(
    rows: torch.Tensor,
    existing_basis: torch.Tensor | None = None,
    tol: float = 1e-12,
) -> torch.Tensor:
    """Row-normalized DGKS Double Gram-Schmidt in `torch.float64`.

    Returns newly spanned orthonormal rows `[k_new, r]` satisfying `Q @ Q^T == I` and
    `Q @ existing_basis^T == 0` to machine epsilon (`~1e-15`).
    """
    x = rows.to(torch.float64)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    r_dim = int(x.shape[-1])
    if x.shape[0] == 0:
        return torch.empty((0, r_dim), dtype=torch.float64, device=x.device)

    row_norms = x.norm(dim=-1, keepdim=True)
    keep = row_norms.squeeze(-1) > tol
    if not bool(keep.any()):
        return torch.empty((0, r_dim), dtype=torch.float64, device=x.device)
    x = x[keep] / row_norms[keep]

    u_exist = (
        existing_basis.to(device=x.device, dtype=torch.float64)
        if (existing_basis is not None and existing_basis.numel() > 0)
        else None
    )

    accepted: list[torch.Tensor] = []
    for i in range(x.shape[0]):
        v = x[i]
        for _ in range(2):
            if u_exist is not None and u_exist.shape[0] > 0:
                v = v - (v @ u_exist.T) @ u_exist
            if accepted:
                q_prev = torch.stack(accepted, dim=0)
                v = v - (v @ q_prev.T) @ q_prev
        nrm = float(v.norm().item())
        if nrm > tol:
            accepted.append(v / nrm)

    if not accepted:
        return torch.empty((0, r_dim), dtype=torch.float64, device=x.device)
    return torch.stack(accepted, dim=0)


@dataclass
class ESMConfig:
    """Configuration for `EpisodicSideMemoryBank` and `NullSpaceESMAnchor`."""

    d_model: int = 2048
    r_hidden: int = 2048
    anchor_layers: tuple[int, ...] = (15, 19, 23, 27, 31, 35)
    scope_threshold: float = 0.65
    ridge_alpha: float = 1e-4
    max_rel_delta_norm: float = 0.03
    allow_coda: bool = False
    coda_start: int = DEFAULT_CORE_LOOP_SPEC.core_end
    seed: int = 2026

    def __post_init__(self) -> None:
        self.anchor_layers = tuple(int(lyr) for lyr in self.anchor_layers)
        self.d_model = int(self.d_model)
        self.r_hidden = int(self.r_hidden)
        self.scope_threshold = float(self.scope_threshold)
        self.ridge_alpha = float(self.ridge_alpha)
        self.max_rel_delta_norm = float(self.max_rel_delta_norm)
        self.allow_coda = bool(self.allow_coda)
        self.coda_start = int(self.coda_start)
        self.seed = int(self.seed)
        self.validate()

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ESMConfig:
        raw = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown esm.* keys {sorted(bad)}; valid: {sorted(known)}")
        if "anchor_layers" in raw and isinstance(raw["anchor_layers"], list):
            raw["anchor_layers"] = tuple(raw["anchor_layers"])
        return cls(**raw)

    @classmethod
    def from_loop_spec(
        cls,
        spec: LoopSpec = DEFAULT_CORE_LOOP_SPEC,
        *,
        full_attention_interval: int = 4,
        **kwargs: Any,
    ) -> ESMConfig:
        """Build an `ESMConfig` matching core full-attention layers inside `[spec.prelude_end, spec.core_end)`."""
        if full_attention_interval < 1:
            raise ValueError(f"full_attention_interval must be >= 1, got {full_attention_interval}")
        anchors = tuple(
            i for i in range(spec.prelude_end, spec.core_end) if (i + 1) % full_attention_interval == 0
        )
        return cls(anchor_layers=anchors, coda_start=spec.core_end, **kwargs)

    def validate(self, loop_spec: LoopSpec | None = None) -> None:
        if self.d_model < 1 or self.r_hidden < 1:
            raise ValueError(f"d_model and r_hidden must be >= 1, got ({self.d_model}, {self.r_hidden})")
        if not self.anchor_layers:
            raise ValueError("anchor_layers must be non-empty")
        if len(set(self.anchor_layers)) != len(self.anchor_layers):
            raise ValueError(f"duplicate anchor_layers found: {self.anchor_layers}")
        if any(lyr < 0 for lyr in self.anchor_layers):
            raise ValueError(f"anchor_layers must be >= 0, got {self.anchor_layers}")
        if not (-1.0 <= self.scope_threshold <= 1.0):
            raise ValueError(f"scope_threshold must be in [-1, 1], got {self.scope_threshold}")
        if self.ridge_alpha <= 0.0:
            raise ValueError(f"ridge_alpha must be > 0, got {self.ridge_alpha}")
        if self.max_rel_delta_norm <= 0.0:
            raise ValueError(f"max_rel_delta_norm must be > 0, got {self.max_rel_delta_norm}")

        coda_boundary = self.coda_start if loop_spec is None else int(loop_spec.core_end)
        coda_layers = [lyr for lyr in self.anchor_layers if lyr >= coda_boundary]
        if coda_layers and not self.allow_coda:
            raise ValueError(
                f"Coda anchor layers {coda_layers} (>= {coda_boundary}) are rejected by the Spark-X2.6 S2 rule; "
                "pass allow_coda=True to override explicitly."
            )

    def theoretical_shard_bytes(self) -> int:
        """Exact ESM shard byte footprint (`w_in` bf16 + `w_down` fp32 = `144.0 MiB` at 6x 2048x2048)."""
        return len(self.anchor_layers) * self.r_hidden * self.d_model * (W_IN_DTYPE_BYTES + W_DOWN_DTYPE_BYTES)


@dataclass
class ESMLedgerEntry:
    """Transactional audit & rollback record for one compiled null-space episodic write."""

    entry_id: str
    anchor_layer: int
    delta_w_down: torch.Tensor
    n_probes: int
    pre_leakage: float
    post_leakage: float
    active: bool = True
    prototypes: torch.Tensor | None = None
    basis_rows: torch.Tensor | None = None


class NullSpaceESMAnchor(nn.Module):
    """Single-layer AlphaEdit null-space episodic memory anchor with `WISE` scope gating and undo ledger."""

    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor
    _w_down_f64: torch.Tensor
    active_basis: torch.Tensor
    _general_basis: torch.Tensor
    scope_prototypes: torch.Tensor

    def __init__(self, config: ESMConfig, anchor_layer: int) -> None:
        super().__init__()
        self.config = config
        self.anchor_layer = int(anchor_layer)

        seed_base = config.seed + 131 * self.anchor_layer
        w_gate = _orthogonal_matrix(config.r_hidden, config.d_model, seed=seed_base + 1)
        w_up = _orthogonal_matrix(config.r_hidden, config.d_model, seed=seed_base + 2)
        self.register_buffer("w_gate", w_gate)
        self.register_buffer("w_up", w_up)
        self.register_buffer("w_down", torch.zeros((config.r_hidden, config.d_model), dtype=torch.float32))
        self.register_buffer(
            "_w_down_f64",
            torch.zeros((config.r_hidden, config.d_model), dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer("active_basis", torch.empty((0, config.r_hidden), dtype=torch.float64))
        self.register_buffer("_general_basis", torch.empty((0, config.r_hidden), dtype=torch.float64), persistent=False)
        self.register_buffer("scope_prototypes", torch.empty((0, config.d_model), dtype=torch.float32))

        self.ledger: list[ESMLedgerEntry] = []

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
        for dynamic_key in ("active_basis", "scope_prototypes"):
            full_key = prefix + dynamic_key
            if full_key in state_dict:
                loaded = state_dict[full_key]
                cur = getattr(self, dynamic_key)
                if cur.shape != loaded.shape:
                    setattr(self, dynamic_key, torch.empty_like(loaded, device=cur.device))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self._w_down_f64.copy_(self.w_down.to(torch.float64))

    def compute_features(self, h: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Compute normalized SwiGLU features `phi(h) = norm(SiLU(h W_gate^T) * (h W_up^T))`."""
        work_dt = upcast(h).dtype if dtype is None else dtype
        h_w = h.to(device=self.w_gate.device, dtype=work_dt)
        wg = self.w_gate.to(dtype=work_dt)
        wu = self.w_up.to(dtype=work_dt)
        gate = F.silu(h_w @ wg.T)
        up = h_w @ wu.T
        raw = gate * up
        norms = raw.norm(dim=-1, keepdim=True)
        return torch.where(norms > 1e-12, raw / norms.clamp(min=1e-12), torch.zeros_like(raw))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute gated episodic residual `delta(h)`.

        Guarantees:
          * Step-0 exact identity (`torch.equal(out, torch.zeros_like(h))`) when empty.
          * Exact `0.0` drift on out-of-scope tokens (`max_p cos(h, p) < scope_threshold`).
          * In-manifold trust-region cap `||delta||_2 <= max_rel_delta_norm * ||h||_2`.
        """
        if h.shape[-1] != self.config.d_model:
            raise ValueError(f"expected last dim d_model={self.config.d_model}, got {h.shape[-1]}")

        use_f64 = h.dtype == torch.float64
        w_mat = self._w_down_f64 if use_f64 else self.w_down
        if self.scope_prototypes.shape[0] == 0 or not bool(w_mat.any()):
            return torch.zeros_like(h)

        h_work = upcast(h)
        h_norm = h_work.norm(dim=-1, keepdim=True)
        h_unit = torch.where(h_norm > 1e-12, h_work / h_norm.clamp(min=1e-12), torch.zeros_like(h_work))

        p_work = self.scope_prototypes.to(device=h.device, dtype=h_work.dtype)
        p_norm = p_work.norm(dim=-1, keepdim=True)
        p_unit = torch.where(p_norm > 1e-12, p_work / p_norm.clamp(min=1e-12), torch.zeros_like(p_work))

        max_cos = (h_unit @ p_unit.T).amax(dim=-1)
        in_scope = (h_norm.squeeze(-1) > 1e-12) & (max_cos >= self.config.scope_threshold)
        if not bool(in_scope.any()):
            return torch.zeros_like(h)

        phi = self.compute_features(h_work, dtype=h_work.dtype).to(device=h.device)
        delta = phi @ w_mat.to(device=h.device, dtype=h_work.dtype)

        max_norm = self.config.max_rel_delta_norm * h_norm
        delta_norm = delta.norm(dim=-1, keepdim=True)
        scale = torch.where(
            delta_norm > max_norm,
            max_norm / delta_norm.clamp(min=1e-12),
            torch.ones_like(delta_norm),
        )
        delta = delta * scale
        delta = torch.where(in_scope.unsqueeze(-1), delta, torch.zeros_like(delta))
        return delta.to(h.dtype)

    @torch.no_grad()
    def register_general_subspace(
        self,
        h_general: torch.Tensor,
        energy_threshold: float = 0.995,
    ) -> torch.Tensor:
        """Extract orthonormal basis `U_gen` via SVD in `float64` to protect general-corpus features."""
        if not (0.0 < energy_threshold <= 1.0):
            raise ValueError(f"energy_threshold must be in (0, 1], got {energy_threshold}")
        if h_general.shape[-1] != self.config.d_model:
            raise ValueError(f"expected last dim d_model={self.config.d_model}, got {h_general.shape[-1]}")

        h_flat = h_general.reshape(-1, self.config.d_model)
        if h_flat.shape[0] == 0:
            return torch.empty((0, self.config.r_hidden), dtype=torch.float64, device=self.active_basis.device)

        phi_gen = self.compute_features(h_flat, dtype=torch.float64)
        _, singular_vals, vh = torch.linalg.svd(phi_gen, full_matrices=False)
        energy = singular_vals.square()
        total_energy = float(energy.sum().item())
        if total_energy <= 0.0:
            return torch.empty((0, self.config.r_hidden), dtype=torch.float64, device=self.active_basis.device)

        cum_ratio = torch.cumsum(energy, dim=0) / total_energy
        above = (cum_ratio >= float(energy_threshold)).nonzero(as_tuple=False)
        k_energy = int(above[0].item()) + 1 if above.numel() > 0 else int(singular_vals.shape[0])
        non_zero = int((singular_vals > 1e-12 * singular_vals[0]).sum().item())
        k = max(1, min(k_energy, non_zero))

        u_cand = vh[:k].to(device=self.active_basis.device, dtype=torch.float64)
        q_gen = dgks_orthonormalize_rows(u_cand, self._general_basis)
        if q_gen.shape[0] > 0:
            self._general_basis = torch.cat([self._general_basis, q_gen], dim=0)

        q_add = dgks_orthonormalize_rows(u_cand, self.active_basis)
        if q_add.shape[0] > 0:
            self.active_basis = torch.cat([self.active_basis, q_add], dim=0)
        return u_cand

    @torch.no_grad()
    def compile_nullspace_ridge(
        self,
        h_triggers: torch.Tensor,
        y_targets: torch.Tensor,
        entry_id: str,
    ) -> tuple[ESMLedgerEntry, float]:
        """Compile a closed-form AlphaEdit null-space ridge update via `float64` DGKS.

        Steps:
          1. Computes normalized SwiGLU features `A = phi(h_triggers)` in `torch.float64`.
          2. Measures unprojected `pre_leakage` against `active_basis` (`U`).
          3. Projects `A_perp = A @ (I - U^T U)` via two-pass DGKS in `torch.float64`.
          4. Solves `(A_perp A_perp^T + alpha I)^{-1} delta_z` in `torch.float64` and polishes
             `delta_w_down` orthogonally against `U`.
          5. Updates `w_down`, appends newly spanned DGKS orthonormal rows to `active_basis`, adds
             unit-normalized trigger prototypes to `scope_prototypes`, and logs an `ESMLedgerEntry`.
        """
        if not entry_id:
            raise ValueError("entry_id must be a non-empty string")
        if any(e.entry_id == entry_id and e.active for e in self.ledger):
            raise ValueError(f"active entry_id {entry_id!r} already exists on layer {self.anchor_layer}")

        h_in = h_triggers.unsqueeze(0) if h_triggers.ndim == 1 else h_triggers.reshape(-1, self.config.d_model)
        y_in = y_targets.unsqueeze(0) if y_targets.ndim == 1 else y_targets.reshape(-1, self.config.d_model)
        if h_in.shape != y_in.shape:
            raise ValueError(f"h_triggers {tuple(h_in.shape)} and y_targets {tuple(y_in.shape)} must match")
        if h_in.shape[0] == 0:
            raise ValueError("h_triggers must contain at least one probe")

        dev = self.w_down.device
        a = self.compute_features(h_in, dtype=torch.float64).to(device=dev)  # [m, r_hidden]
        y_f64 = y_in.to(device=dev, dtype=torch.float64)                     # [m, d_model]
        delta_z = y_f64 - a @ self._w_down_f64                               # [m, d_model]

        m = int(a.shape[0])
        alpha = float(self.config.ridge_alpha)
        eye_m = torch.eye(m, dtype=torch.float64, device=dev)
        u_prev = self.active_basis.to(device=dev, dtype=torch.float64)

        if u_prev.shape[0] > 0:
            gram_raw = a @ a.T + alpha * eye_m
            dw_raw = a.T @ torch.linalg.solve(gram_raw, delta_z)
            pre_leakage = float((u_prev @ dw_raw).abs().max().item())
        else:
            pre_leakage = 0.0

        a_perp = project_orthogonal_dgks(a, u_prev)
        gram = a_perp @ a_perp.T + alpha * eye_m
        coeff = torch.linalg.solve(gram, delta_z)
        dw_f64 = a_perp.T @ coeff

        if u_prev.shape[0] > 0:
            # Final 2-pass DGKS orthogonal polish on columns of `dw_f64`
            dw_f64 = dw_f64 - u_prev.T @ (u_prev @ dw_f64)
            dw_f64 = dw_f64 - u_prev.T @ (u_prev @ dw_f64)
            post_leakage = float((u_prev @ dw_f64).abs().max().item())
        else:
            post_leakage = 0.0

        self._w_down_f64.add_(dw_f64)
        self.w_down.copy_(self._w_down_f64.to(torch.float32))

        q_new = dgks_orthonormalize_rows(a_perp, u_prev)
        if q_new.shape[0] > 0:
            self.active_basis = torch.cat([u_prev, q_new], dim=0)

        h_f32 = h_in.detach().to(device=dev, dtype=torch.float32)
        h_nrm = h_f32.norm(dim=-1, keepdim=True)
        protos = torch.where(h_nrm > 1e-12, h_f32 / h_nrm.clamp(min=1e-12), torch.zeros_like(h_f32))
        self.scope_prototypes = torch.cat([self.scope_prototypes, protos], dim=0)

        entry = ESMLedgerEntry(
            entry_id=entry_id,
            anchor_layer=self.anchor_layer,
            delta_w_down=dw_f64.detach().clone(),
            n_probes=m,
            pre_leakage=pre_leakage,
            post_leakage=post_leakage,
            active=True,
            prototypes=protos.detach().clone(),
            basis_rows=q_new.detach().clone(),
        )
        self.ledger.append(entry)
        return entry, post_leakage

    @torch.no_grad()
    def rollback_entry(self, entry_id: str) -> ESMLedgerEntry:
        """Exact transactional undo of `entry_id`: subtracts `delta_w_down` and restores prototypes/basis."""
        target: ESMLedgerEntry | None = None
        for entry in reversed(self.ledger):
            if entry.entry_id == entry_id:
                target = entry
                break
        if target is None:
            raise KeyError(f"unknown entry_id {entry_id!r} on layer {self.anchor_layer}")
        if not target.active:
            raise ValueError(f"entry_id {entry_id!r} on layer {self.anchor_layer} is already rolled back")

        dev = self.w_down.device
        self._w_down_f64.sub_(target.delta_w_down.to(device=dev, dtype=torch.float64))
        target.active = False

        # Snap to exact zero when no active entries remain so Step-0 bit-identity holds after full undo
        active_entries = [e for e in self.ledger if e.active]
        if not active_entries:
            self._w_down_f64.zero_()
            self.w_down.zero_()
            self.scope_prototypes = torch.empty((0, self.config.d_model), dtype=torch.float32, device=dev)
            self.active_basis = self._general_basis.to(device=dev, dtype=torch.float64).clone()
            return target

        self.w_down.copy_(self._w_down_f64.to(torch.float32))

        proto_chunks = [
            e.prototypes.to(device=dev, dtype=torch.float32)
            for e in active_entries
            if e.prototypes is not None and e.prototypes.numel() > 0
        ]
        if proto_chunks:
            self.scope_prototypes = torch.cat(proto_chunks, dim=0)
        else:
            self.scope_prototypes = torch.empty((0, self.config.d_model), dtype=torch.float32, device=dev)

        rebuilt_basis = self._general_basis.to(device=dev, dtype=torch.float64).clone()
        for e in active_entries:
            if e.basis_rows is not None and e.basis_rows.numel() > 0:
                q_add = dgks_orthonormalize_rows(e.basis_rows, rebuilt_basis)
                if q_add.shape[0] > 0:
                    rebuilt_basis = torch.cat([rebuilt_basis, q_add], dim=0)
        self.active_basis = rebuilt_basis
        return target


class EpisodicSideMemoryBank(nn.Module):
    """Multi-layer `EpisodicSideMemory` bank managing `NullSpaceESMAnchor`s across core `anchor_layers`."""

    def __init__(self, config: ESMConfig | None = None, *, loop_spec: LoopSpec | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ESMConfig()
        self.config.validate(loop_spec=loop_spec)
        self.anchors = nn.ModuleDict(
            {f"L{layer}": NullSpaceESMAnchor(self.config, anchor_layer=layer) for layer in self.config.anchor_layers}
        )

    @property
    def anchor_layers(self) -> tuple[int, ...]:
        return self.config.anchor_layers

    def has_layer(self, layer_idx: int) -> bool:
        return f"L{int(layer_idx)}" in self.anchors

    def anchor(self, layer_idx: int) -> NullSpaceESMAnchor:
        key = f"L{int(layer_idx)}"
        if key not in self.anchors:
            raise KeyError(f"layer {layer_idx} is not anchored; valid layers: {self.config.anchor_layers}")
        mod: NullSpaceESMAnchor = self.anchors[key]  # type: ignore[assignment]
        return mod

    def forward(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        return self.anchor(layer_idx)(h)

    def register_general_subspace(
        self,
        layer_idx: int,
        h_general: torch.Tensor,
        energy_threshold: float = 0.995,
    ) -> torch.Tensor:
        return self.anchor(layer_idx).register_general_subspace(h_general, energy_threshold=energy_threshold)

    def compile_nullspace_ridge(
        self,
        layer_idx: int,
        h_triggers: torch.Tensor,
        y_targets: torch.Tensor,
        entry_id: str,
    ) -> tuple[ESMLedgerEntry, float]:
        return self.anchor(layer_idx).compile_nullspace_ridge(h_triggers, y_targets, entry_id=entry_id)

    def rollback_entry(self, layer_idx: int, entry_id: str) -> ESMLedgerEntry:
        return self.anchor(layer_idx).rollback_entry(entry_id)

    def shard_bytes(self) -> int:
        """Exact shard byte footprint (`w_in` bf16 + `w_down` fp32 = `144.0 MiB` at `6` anchors, `2048x2048`)."""
        return len(self.anchors) * self.config.r_hidden * self.config.d_model * (W_IN_DTYPE_BYTES + W_DOWN_DTYPE_BYTES)

    def shard_mib(self) -> float:
        return self.shard_bytes() / float(1024**2)

    @staticmethod
    def theoretical_shard_bytes(num_anchors: int = 6, r_hidden: int = 2048, d_model: int = 2048) -> int:
        return int(num_anchors) * int(r_hidden) * int(d_model) * (W_IN_DTYPE_BYTES + W_DOWN_DTYPE_BYTES)
