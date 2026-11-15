# Qwen3.6-35B-A6B-LoopHSSD (v3.2): Definitive Backbone Modification Specification

> **Status:** Canonical `v3.2` Architecture Specification (Supersedes `Loop-HSSD-10M v1`, `v2`, `v3.0`, and `v3.1`, incorporating all External `HSSD v3.1` Audit fixes plus the `v3.2` Loop60 Mathematical, Scale & Blackwell Throughput Review fixes `H1–H9`, `D1–D8`, `Section 5.2`, and `Section 5.3`)
> **Base Checkpoint:** `Qwen/Qwen3.6-35B-A3B` (`34.661B` text decoder; `>98.5%` pretrained weights reused directly)
> **Target Sync Repository:** [`TassP0lar/Qwen3.6-35B-A3B-Loop64`](https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64)
> **Zero-Weight Math & Falsification Gate:** [`test_loop_hssd_v3_math.py`](../tests/test_loop_hssd_v3_math.py) & `test_hssd_audit_regressions.py` (`float64` algebraic & storage invariants)
> **Companion Documents:** [Implementation Plan](IMPLEMENTATION_PLAN.md) · [Formal Research Essay](RESEARCH_ESSAY.md) · [Baseline Repo Engineering Notes (CN)](ANALYSIS.md)

---

## 1. Executive Thesis: A Custom `~35B-A6B` Architecture Built by Weight Reuse

Training a 35-billion-parameter Mixture-of-Experts foundation model from random initialization requires `>15T` tokens and thousands of GPU-months—impossible for an independent experimental research project. However, **`Qwen/Qwen3.6-35B-A3B`** exposes a striking structural asymmetry that makes post-training *architectural surgery* extraordinarily high-leverage:

1. **Massive Dormant Knowledge vs. Tiny Per-Token Compute (`35B` Total vs. `~2.95B` Active):** Over **`92.9%`** of the text model's parameters (`32.212B` across `40 × 256` routed SwiGLU experts) sit in frozen FFN banks, yet every token activates only **8 of 256** experts (`1.007B` routed parameters) across **40 physical layers** (of which only **10** have global softmax attention). Consequently, on multi-hop compositional deduction, algorithm synthesis, and deep state tracking, stock `Qwen3.6-35B-A3B` frequently hits the sequential circuit-depth ceiling of a small 40-layer, 3B-active model despite holding 35B parameters of world knowledge in HBM.
2. **The Hybrid Mid-Horizon Memory Cliff (`513 -> 262K`):** Stock Qwen3.6 interleaves **30 scalar-gated Gated DeltaNet (GDN)** layers (`3:1`) with **10 Gated Full-Attention** layers (`partial_rotary_factor = 0.25`, native `262,144` context). Each GDN head compresses the entire token prefix into a fixed $128 \times 128$ real matrix updated by a single scalar gate $\beta_t \in (0, 1)$, suffering exponential associative washout once cumulative write mass exceeds head dimension $d_k = 128$.
3. **The Ultra-Long Context KV Wall (`1M -> 10M`):** Pushing the 10 full-attention layers to **`1M` tokens** (`10.00 GiB` in FP8) dilutes softmax entropy and incurs quadratic attention latency without sparse indexing; pushing to **`10M` tokens** (`190.73 GiB` in BF16) exceeds single-GPU `96 GB` Blackwell HBM entirely.

**`Qwen3.6-35B-A6B-LoopHSSD (v3)`** transforms `Qwen3.6-35B-A3B` into a custom recurrent-depth hybrid architecture while **reusing 100% of Qwen3.6's `32.338B` MoE FFN weights untouched**:

- **Pillar I — Subspace-Aligned Sandwich Loop & Dynamic Multi-Expert Expansion (`35B-A3B -> 35B-A6B`):** Loops the middle 24 physical layers (`6` hybrid macro-blocks, `L8..L31`) for $R \in \{2, 3\}$ passes (`64` virtual layers / **`~4.50B`** active compute at $R=2$; `88` virtual layers / **`~6.05B`** active compute and **`22` virtual global-attention layers** at $R=3$). Decoupling a learned low-rank **entry subspace bridge** (`ReEntryBridge`) from a **zero-initialized pass-exit gate** preserves exact step-0 equivalence to stock Qwen3.6 while allowing **`PassRouterAdapters`** to recruit *complementary* Top-8 experts on Passes 2 and 3.
- **Pillar II — Dual-Track Linear Memory Upgrade (`GDN-2` Baseline $\to$ `HSSD v3.1` Working Memory):**
  - **Track A (`GDN-2`, Step-0 Bit-Exact):** Upgrades all 30 GDN layers in-place to NVIDIA Gated DeltaNet-2 decoupled channel-wise erase/write/decay (`27.5M` adapter params, `torch.equal` at step 0).
  - **Track B (`HSSD v3.1`, Horizon-Split Surprise-Gated Delta Memory):** Upgrades the 30 linear layers (`~1.92M–1.95M` params/layer, `~57.5M–58.5M` across 30L with low-rank phase `theta_low_rank=True`) using a strictly disjoint `513–576` local SWA window + learned `PeakBoltzmannGate` branch heuristic + post-rotation scaled real symmetric DPLR Householder (`chunk_hssd`) + linear-$\beta$ write scaling ($\sqrt{\mathbf{w}_t}$) + multi-page routed chunk-start root-surprise (`surprise_mode="routed"` against `boundary_ring[-1]`) + $M=8$ multi-timescale pages regularized by **Straight-Through actual sparse Top-$k$ write-mass budgets** $\widetilde{u}_m = \mathbb{E}[g_t^2 \widetilde{\pi}_{t,m}]$. Because HSSD alters horizon partitioning, multi-page routing, surprise gating, and decay schedules even at zero adapter up-projection init, `Stage S1b` layer-output + chunked-vocab KL distillation aligns representations entering Qwen's frozen `32.338B` MoE experts.
- **Pillar III — Two-Tier Ultra-Long Context (`1M` GPU-Resident Primary + `10M` Host-DDR5 Stretch):**
  - **Tier 1 (`1M` Primary Target):** `10.00 GiB` GPU-resident `FP8` KV cache + `320 MiB` GPU Ward centroids (`D-LSA` on the `192` NoPE dims), calibrated via Long-Prefix (`no_grad`) + Short-Suffix (`grad`) read training, with **per-pass query re-indexing** ($W_{q\text{-idx}}^{(r)}$) so Pass 2/3 retrieves new context blocks at `1.0×` KV memory, paired with **`81–89 MiB/layer` (`2.37–2.61 GiB` / 30L @ $R=1$)** `bfloat16` HSSD decode caches (`fp32` WY updates + `fp64` `theta_cum`).
  - **Tier 2 (`10M` Stretch Target):** `2.40 GiB` GPU centroids + **`53.64 GiB`** pinned Host DDR5 cache compressed via **Block-Diagonal $H_{64}$ Walsh-Hadamard Group-64 `INT4`** (strictly isolating `[0:64]` mRoPE from `[64:256]` NoPE so query-time ReRoPE clamping commutes with stored keys), prefetched asynchronously across 3 preceding HSSD+MoE layers over PCIe 5.0.
  - **Staged Empirical Framing:** Rather than pre-proven guarantees, **`64K` working memory**, **`262K` state tracking**, **`1M` sparse retrieval + recurrent memory**, and **`10M` host-offloaded extensibility** are treated strictly as **staged empirical hypotheses** gated sequentially by `Gate 0p-4`, `Gate 1-Ablate`, `Gate 1`, `Gate 2a`, and `Gate 2b`.

---

## 2. Exact `Qwen/Qwen3.6-35B-A3B` Parameter & Active-Compute Ledger

### 2.1 Ground-Truth Base Checkpoint Audit (Closing Review Finding `E2`)

Verified directly from `Qwen/Qwen3.6-35B-A3B/config.json` (`hidden_size = 2048`, `num_hidden_layers = 40`, `layer_types = 10 × [3 linear_attention + 1 full_attention]`, `num_experts = 256`, `num_experts_per_tok = 8`, `moe_intermediate_size = 512`, `shared_expert_intermediate_size = 512`, `vocab_size = 248320`, `tie_word_embeddings = false`):

| Component | Exact Tensor Dimensions (`40` Text Layers) | Total Params | Single-Pass Active (`R=1`) | Default v3 Freeze Policy |
| :--- | :--- | ---: | ---: | :--- |
| **Routed SwiGLU Experts** | $40 \times 256 \times (3 \times 2048 \times 512)$ | **`32.212B`** | `1.007B` (`Top-8`) | **100% Frozen** (SHA-256 verified) |
| **Shared SwiGLU Experts** | $40 \times (3 \times 2048 \times 512)$ *(NOT 2560!)* | **`0.126B`** | `0.126B` (`1/layer`) | **Frozen** in Gates 0–2 *(optional LoRA in SFT/RL)* |
| **MoE Routers + Shared Gates** | $40 \times (2048 \times 256 + 2048 \times 1)$ | **`0.021B`** | `0.021B` | Base frozen; **`PassRouterAdapters`** trainable |
| **30 GDN Linear-Attn Layers** | `in_proj_qkvz` ($2048 \to 12288$), `in_proj_ba` ($2048 \to 64$), `conv1d` ($8192 \times 4$), `out_proj` ($4096 \to 2048$) | **`1.012B`** | `1.012B` | Replaced/adapted by **GDN-2** or **HSSD v3** |
| **10 Gated Full-Attn Layers** | `q_proj` ($2048 \to 8192$: $16\text{Q}\times 256$ + gate), `k/v_proj` ($2048 \to 512$), `o_proj` ($4096 \to 2048$) | **`0.273B`** | `0.273B` | Base frozen + **per-pass Q/O LoRA & D-LSA** |
| **Token Embed + Untied `lm_head`** | $2 \times (248320 \times 2048)$ | **`1.017B`** | `0.509B` (`lm_head` GEMV) | **100% Frozen** (`embed` is sparse lookup) |
| **RMSNorms & Layer Scales** | $40 \times \text{norms}$ | **`<0.001B`** | `<0.001B` | Trainable in Core/Coda |
| **40-Layer Text Decoder Total** | — | **`34.661B`** | **`2.948B`** *(FLOP-active)* | `>98.5%` reused directly |
| **Vision Tower (`visual`) + `mtp`** | 27L ViT ($d=1152, d_{ff}=4304$) + 1L MTP block | `~0.85B–1.3B` | `0.000B` *(pruned in text mode)* | Pruned during text backbone training |

> [!NOTE]
> **Reconciling `v2`'s `0.629B` Error (`Fix E2`):** `v2` assumed `shared_expert_intermediate_size = 2560`, overstating shared experts by $5\times$ (`0.629B` vs actual `0.126B`). In `Qwen3.6-35B-A3B`, each shared expert has `intermediate_size = 512` (`3.146M` params/layer, `0.126B` across 40 layers). The remaining parameters in the `~35.5B–36B` multimodal repository belong to the 27-layer ViT vision encoder and the Multi-Token Prediction (`mtp`) head.

### 2.2 Active-Parameter & Active-Compute Expansion (`35B-A3B` $\to$ `35B-A6B`)

When core layers `[8..31]` (`18` linear + `6` full-attention + `24` MoE blocks) execute $R$ recurrent passes per token with `kv_share = True`, each additional pass executes `24` layers of attention/recurrent projections (`0.771B`), `24` shared experts (`0.075B`), `24` routers (`0.013B`), `24 × 8` routed experts (`0.604B`), and low-rank pass adapters (`~0.04B`), adding **`+1.50B` active FLOP-equivalent parameters per pass**:

| Operating Mode | Schedule (`Prelude + Core×R + Coda`) | Virtual Layers | Virtual Full-Attn Layers | Effective Active Compute Params (FLOP-Equiv / tok) | Unique Physical Params Consulted / tok (`η` new experts/pass) | Relative Token Compute |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stock `Qwen3.6-35B-A3B` (`R=1`)** | `8 + 24×1 + 8` | `40L` | `10` | **`2.95B`** | `2.95B` (`8` experts/layer) | `1.00×` |
| **`LoopHSSD-64L` (`R=2`, Primary)** | `8 + 24×2 + 8` | **`64L`** | **`16`** | **`4.46B` (`~A4.5B`)** | **`3.35B – 3.55B`** (`13–16` unique experts in core) | **`1.51× – 1.56×`** |
| **`LoopHSSD-88L` (`R=3`, `A6B` Mode)** | `8 + 24×3 + 8` | **`88L`** | **`22`** | **`5.97B` (`~A6.0B`)** | **`3.75B – 4.16B`** (`18–24` unique experts in core) | **`2.02× – 2.10×`** |
| **`LoopHSSD-88L` + Cross-Layer Borrow** | `8 + 24×3 + 8` *(Phase-5 opt)* | **`88L`** | **`22`** | **`6.02B` (`~A6.0B`)** | **`4.30B – 5.20B`** *(borrows $l \pm 4$ FFN pools)* | **`~2.08×`** |

- **Why Both Metrics Matter:** *Effective Active Compute Parameters (`~6.0B` at `R=3`)* govern sequential reasoning capacity and matrix-multiply FLOPs per token; *Unique Physical Parameters Consulted (`3.75B–5.20B`)* govern distinct world-knowledge circuits read from the frozen `32.212B` MoE pool per token during memory-bandwidth-bound single-stream decoding.
- **Zero-Retraining Fallback Guarantee:** Setting `active_passes = 1` collapses the graph cleanly back to `40L` depth-1 execution, preserving fast throughput for trivial tokens (or per-token causal MoR early exit once converged).

### 2.3 Core Trainable Parameter Budget (`~85.8M` Params = `0.25%` of Text Backbone)

All `32.212B` routed experts and `0.126B` shared experts remain **frozen (`requires_grad = False`)** across Backbone Surgery & Long-Context Distillation (`Gates 0b–3`), enforced by a SHA-256 buffer/parameter checksum before and after every training stage:

| Trainable Subsystem | Tensor Shapes & Locations | Track A (`GDN-2`) | Track B (`HSSD v3.1`) |
| :--- | :--- | ---: | ---: |
| **1. `ReEntryBridge` (`r=256`, Passes `2..R`)** | Per-pass Birkhoff logits (3) + `RMSNorm` + $2048 \to 256 \to 2048$ + step emb + $\boldsymbol{\alpha}_{\text{exit}}^{(r)} \in \mathbb{R}^{2048}$ | `2.11M` ($R=3$) | `2.11M` ($R=3$) |
| **2. `PassRouterAdapters` (`r=32`, Passes `2..R`)** | $24 \times (R-1) \times (2048 \to 32 \to 256 + \text{bias}_{256} + \lambda_{r,l})$ | `3.55M` ($R=3$) | `3.55M` ($R=3$) |
| **3. Core Per-Pass Full-Attn Q/O LoRA (`r=64`)** | $6 \times (R-1) \times (W_Q^{(r)}, W_O^{(r)}\text{ rank-64}) + \text{QK-Norms}$ | `15.75M` ($R=3$) | `15.75M` ($R=3$) |
| **4. 30 Linear-Attn Upgrade Modules** | **Track A:** `GDN2Adapter` (`r=64` erase/write/decay, 30L)<br>**Track B:** `HSSDAdapter` (`r=64` erase/write/decay/SWA, `theta_low_rank=True` `theta_down`+`theta_up`, surprise, $M=8$ routers; `~1.92M–1.95M`/layer vs `5.88M`/layer = `176.5M`/30L when dense `theta_proj` took `71.3%`) | `27.53M` | `58.50M` (`~57.5M–58.5M`) |
| **5. 10 Full-Attn `D-LSA` Indexers (`d_idx=64`)** | Per-pass $W_{q\text{-idx}}^{(r)} \in \mathbb{R}^{192 \times 64}$ + shared $W_{k\text{-idx}} \in \mathbb{R}^{192 \times 64}$ across 10 layers | `5.90M` | `5.90M` |
| **Total Core Backbone Trainables** | — | **`54.84M` (`0.16%`)** | **`85.81M` (`0.25%`)** |
| *(Optional S2/S4 Only)* **`HotExpertDeltas`** | Zero-init rank-64 LoRA on top `2.5%` Pass-2/3 hot experts (`~256` of `10,240` experts) | *(+`96.47M` opt)* | *(+`96.47M` opt)* |

---

## 3. Complete Audit Closure Matrix (`v1`, `v2 Final-Pass Review`, & §7.3 Decisions)

### 3.1 Closure of All 17 `v2 Final-Pass Review` Findings (`@Tian`, Nov 13, 2026)

| ID | Sev | `v2` Flaw Identified in Review | Definitive `v3` Architectural Resolution | Regression Test |
| :--- | :---: | :--- | :--- | :--- |
| **`A1`** | **P0** | `w_anc = 0.99` + frozen shared router locked Pass-1/Pass-2 Top-8 expert overlap at `97–99%`, contradicting multi-expert thesis. | Decoupled entry bridge (`w_anc = 0.85` + `r=256` linear subspace adapter) from zero-init exit gate (`alpha_exit = 0`), AND restored **`PassRouterAdapters`** trained purely by task/KD loss. | `test_v3_05` |
| **`A2`** | **P1** | With `alpha_1 = 0.01`, effective loop output perturbation at init was negligible (`~1e-4`), slowing gradient bootstrap. | Exact zero-initialized **pass-exit residual gate** $\boldsymbol{\alpha}_{\text{exit}}^{(r)} = \mathbf{0}$ gives step-0 bit-identity (`x + 0*delta == x`) while Pass-2 receives full `0.15` semantic signal inside `Core` (`dL/d alpha_exit != 0` on step 1!). | `test_v3_05`, `test_v3_07` |
| **`A3`** | **P1** | Diagonal vector $\alpha_1 \odot \text{RMSNorm}(h_{31})$ cannot undo an **orthogonal rotation** $Q_{\perp}$ between Layer-31 and Layer-8 subspaces. | Added low-rank linear **subspace alignment adapter** $W_{\text{re,up}}^{(r)} \phi(W_{\text{re,down}}^{(r)} \operatorname{RMSNorm}(h_{31}^{(r-1)}))$ (`r=256`) inside `ReEntryBridge`, capable of inverting rotated semantic bases. | `test_v3_05` |
| **`A4`** | **P1** | `R=2` (`64L`, `~1.53×` FLOPs) was compared against `1.0×` stock model without compute-matched CoT budget. | Mandatory **FLOP-matched baseline** in `Gate 0b`: stock `40L` receives `1.6×` thinking tokens vs `R=2` (`64L`) and `2.1×` thinking tokens vs `R=3` (`88L`). | `IMPLEMENTATION_PLAN` §4.4 |
| **`A5`** | **P2** | Pass-2 behavior on core full-attention layers `[11, 15, 19, 23, 27, 31]` (`Shared KV` vs `Separate KV`) was left ambiguous. | Explicitly standardized **`kv_share = True`**: Pass $r \ge 2$ reuses Pass-1's $K^{(1)}, V^{(1)}$ + Ward centroids (`1.0×` KV RAM), with untied per-pass $W_Q^{(r)}, W_O^{(r)}$ LoRA + untied $W_{q\text{-idx}}^{(r)}$. Cross-token Pass-2 memory flows via independent HSSD states $\bar{S}_t^{(r)}$. | `test_v3_07` |
| **`A6`** | **P2** | `GSM8K +2.0%` was noisy/saturated for Qwen3.6-35B (`~95%` baseline). | Replaced GSM8K in `Gate 0b` with **AIME 2025/2026**, **ARC-AGI-2**, **LiveCodeBench**, and **synthetic multi-hop variable-tracking (`k=4..12` hops)**. | `IMPLEMENTATION_PLAN` §4.4 |
| **`B1`** | **P0** | **Off-by-one bug:** `S_boundary[n - 7]` held `(n-7)*C` tokens while `far_cutoff` was `(n-8)*C`, double-counting `64` tokens `[ (n-8)C+1 .. (n-7)C ]` in BOTH `o_far` and `o_loc`! | Fixed to **`k_far = max(0, n - 8)`** (`S_boundary[k_far]`, `far_cutoff = k_far * C`). $\mathcal{I}_{\text{far}} = [1, 64 k_{\text{far}}]$ and $\mathcal{I}_{\text{loc}} = [64 k_{\text{far}} + 1, t]$ are strictly disjoint with zero gap and $|\mathcal{I}_{\text{loc}}| \in [513, 576]$ for $t > 512$. | `test_v3_02` |
| **`B2`** | **P1** | Unbudgeted soft surprise (`s = 0.02`) accumulates $\mathcal{B} = 320 = 2.5\, d_k$ over `20K` tokens, erasing `>91%` of needle amplitude (`exp(-2.5) = 0.082`). | Two-sided **STE Sparse Top-$k$ Write-Mass Regularizer** on $\widetilde{u}_m = \mathbb{E}[g_t^2 \widetilde{\pi}_{t,m}]$ with log-spaced page caps $\mu_m^*$ (soft training regularizer targeting $\mathcal{B}_m(H) \le 0.4\, d_k$). | `test_v3_04` |
| **`B3`** | **P1** | `k_eff = sqrt(s * b) * k` has $\partial \sqrt{s}/\partial s = \frac{1}{2\sqrt{s}} \to \infty$ at $s \to 0$ (`inf/NaN` gradients). | Parameterized **root-surprise $g_{\tau,h} \in [0, 1]$ directly** ($\mathbf{e} = g \sqrt{\mathbf{b}} \odot \hat{\mathbf{k}}$, $\mathbf{v}_{\text{eff}} = g \sqrt{\mathbf{w}} \odot \mathbf{v}$ for linear-$\beta$ write scaling, $\alpha = \exp(-g^2 \gamma)$, smooth bounded $\partial / \partial g$ everywhere). | `test_v3_03` |
| **`B4`** | **P1** | Computing $g_t$ from token-instantaneous $\bar{S}_{t-1}$ reintroduces sequential recurrence inside chunks, breaking parallel `chunk_kda`. | Defined $g_{\tau,h}$ for all $\tau \in [nC+1, (n+1)C]$ in parallel against the **Top-$k_w$ write-routed pages of `boundary_ring[-1]`** ($\bar{S}_{nC}$) (`surprise_mode="routed"`) before the WY solve! | `test_v3_03` |
| **`B5`** | **P2** | Pure `LSE(scores) - log(N)` compresses single-needle hits as $\log(N)$ grows (`576` vs `64`). | Replaced with **`PeakBoltzmannGate`** ($\Psi_{t,h}^{\text{loc}} = \frac{1}{2}(\operatorname{LSE}(z) - \ln N) + \frac{1}{2}(\max(z) - \sqrt{2\ln\max(N,2)})$), a sub-Gaussian-centered learned local/far branch heuristic. | `test_v3_02` |
| **`B6`** | **P2** | Instantiating $M=8$ state trajectories + 5D `gamma` across 30 layers simultaneously OOMs `96 GB` VRAM. | **Factored decay (`8 KiB/tok/layer`)** + **macro-segment activation checkpointing** + **`bfloat16` state storage (`81–89 MiB/layer`)** + active-page decode (`k=2`). | `§5.2.(6)` |
| **`C1`** | **P0** | Zero-shot `262K -> 10M` ($38\times$) extrapolation without read-side training fails due to RoPE breakdown and softmax entropy dilution (`log(10^7) = 16.1`). | Split into **`1M` Primary Target** (validated via `Gate 0p-3` YaRN vs DCA/ReRoPE probe + **Long-Prefix `no_grad` / `32K` Short-Suffix `grad` calibration**) and **`10M` Stretch Target** (`Gate 2b`). | `§6.1–6.2` |
| **`C2`** | **P1** | Full $H_{256}$ Hadamard mixes `[0:64]` rotary dims into `[64:256]` NoPE dims, breaking commutativity with query-time ReRoPE distance clamping! | Replaced $H_{256}$ with **Block-Diagonal $H_{64}$** across the four 64-dim groups (`[0:64]` mRoPE + `3×64` NoPE). Matches Group-64 INT4 quantization identically while keeping rotary and NoPE subspaces 100% orthogonal! | `test_v3_06` |
| **`E1`** | **P1** | `4.2 days` single-GPU wall-clock assumed unrealistic `>25K tok/s` throughput (`~8–14×` too optimistic). | Recalibrated all stage token budgets (`50M–150M` tokens/gate) and wall-clock schedules to measured Blackwell `1,800–4,200 tok/s` across resumable `21.5h` Colab sessions. | `IMPLEMENTATION_PLAN` §6 |
| **`E2`** | **P2** | Shared expert size misreported as `0.629B` instead of `0.126B`. | Reconciled complete `34.661B` text decoder + ViT/MTP parameter ledger (§2.1) and locked in `test_v3_07`. | `test_v3_07` |
| **`E3`** | **P2** | `4 × 32K = 128K` Truncated BPTT carried HSSD states across segments, but detached full-attention K/V across `32K` boundaries! | Added **Segment-Boundary K/V Gradient Replay** (`src/qloop/train/bptt.py`): re-projects only the selected Top-$K_s$ sparse keys (`2,048` tokens out of prior segments) with gradients enabled during backward. | `IMPLEMENTATION_PLAN` §3.5 |

### 3.2 Explicit Resolution of Review §7.3 Open Architectural Decisions

1. **Decision 1 (`10M` Headline vs `1M` Primary + `10M` Stretch):** **Adopted.** **`1M` (`1,048,576` tokens, 100% GPU-resident `10.00 GiB` FP8 KV)** is the primary production target (`Gate 2a`), directly benchmarked on `LongBench v2` (`8K–2M`) and `1M RULER`. **`10M` (`53.64 GiB` Host-DDR5 Block-64 Hadamard `INT4`)** is gated behind `Gate 2a` pass as an explicit stretch milestone (`Gate 2b`).
2. **Decision 2 (`Pass-2` Core Attention `Shared KV` vs `Separate KV`):** **Adopted `Shared KV` (`kv_share = True`) with Untied Per-Pass Query/Output LoRA & Untied D-LSA Query Indexers.** Separate KV would multiply `1M/10M` KV memory by $1.6\times\text{–}2.2\times$ (`+6–12 GiB` HBM at `1M`, `+32–64 GiB` DDR5 at `10M`), whereas `Shared KV` + untied $(W_Q^{(r)}, W_O^{(r)}, W_{q\text{-idx}}^{(r)})$ allows Pass 2 and Pass 3 to attend to *completely different* latent memory buckets at **zero extra KV bytes**, while the 18 core HSSD/GDN-2 layers carry independent cross-token recurrent states $\bar{S}_t^{(r)}$.
3. **Decision 3 (`PassRouterAdapters` Policy):** **Restored & Standardized (`per_pass_router = True`, `router_adapter_rank = 32`, `router_diversity_weight = 0.0`).** Per-pass router adapters are zero-initialized so step-0 routing matches stock Qwen3.6, and trained **exclusively by task / KL-distillation gradients** (with optional learned scalar repeat-suppression lever $\lambda_{r,l}$ initialized at `0.0`), letting the model discover natural expert specialization without artificial overlap targets.

### 3.3 External `HSSD v3.1` Audit Closure Matrix (`3A–3E`, `4.1–4.3`, `5–6`)

| Audit ID | Finding / Discrepancy | `HSSD v3.1` Resolution in Code & Spec |
| :--- | :--- | :--- |
| **`3A`** | `theta_cum_all[:, -1]` returned a non-owning view retaining full `[B, T, H_v, d_k/2]` backing storage (`2 GiB/layer @ 262K`). | Terminal `theta_cum` (and every cached tensor) is `.clone().contiguous()`, verified by `HSSDCacheState.unique_storage_bytes()` and `has_compact_storage()`. |
| **`3B`** | `project_inputs` materialized 5D `gamma` `[B, T, M, H_v, d_k]` (`128 KiB/tok/layer` = `32 GiB/layer @ 262K`) prior to segment checkpointing. | Factored decay `(gamma_page_scale [M], gamma_chan_pair [B, T, H_v, d_k/2])` (`8 KiB/tok/layer` in `fp32`, `16×` smaller); expands across $M=8$ only inside each `C=64` chunk or step. |
| **`3C`** | Budget loss evaluated on dense softmax $\pi_m^{\text{write}}$ let $M - k_w = 6$ unselected dead pages report `budget_loss == 0`. | Evaluates on **actual renormalized Top-$k_w$ sparse write weights** $\widetilde{\pi}_{t,m}$ via Straight-Through Estimator $\boldsymbol{\pi}_{\text{ste}} = \widetilde{\boldsymbol{\pi}} + (\boldsymbol{\pi} - \operatorname{sg}(\boldsymbol{\pi}))$, tracking `dead_page_count`. |
| **`3D`** | `HSSDCacheState` lacked HF `Cache` tensor operations (`zeros_like`, `copy_`, beam `index_select`, `reorder_cache`). | Implemented `__torch_function__`, `.copy_()`, `__getitem__`, `.reorder_cache()`, and HF-compatible `HSSDHybridCache` across both native and standalone pack modules. |
| **`3E` & `4.1`** | Packed sequences (`cu_seqlens` / `seq_idx`) leaked SWA/recurrent state across doc boundaries; 2D `attention_mask` padding advanced logical `step_count`. | Per-document zero-state isolation on `cu_seqlens`/`seq_idx` (`0.0` cross-doc leakage) + per-row unpad/repad with `step_count: Tensor[B]` (`< 1e-13` `fp64` padding invariance). |
| **`4.2` & `4.3`** | Separate `s_chunk_start` duplicated `boundary_ring[-1]`; `state_equal` ignored SWA rings; `fp32` `theta_cum` drifted at `1M` steps. | Removed separate `s_chunk_start` storage (`boundary_ring[-1]` holds all $M=8$ chunk-start pages); `state_equal` checks valid SWA K/V; `theta_cum` stays `torch.float64` (`16 KiB/layer`, `< 2e-6 rad` drift @ `1M`). |
| **`5` & `6`** | Post-rotation erase order vs Theorem 1; `w_gate` gave $\beta^{3/2}$ write scaling; Page-0-only surprise missed Pages `1..7`; `theta_proj` took `71.3%` params; decode cache understated at `8 MiB`; GPU `bool()` sync in WY update. | Proved post-rotation $\mathbf{e}_t = g_t\sqrt{\mathbf{b}_t}\odot(R(-\boldsymbol{\Theta}_t)\mathbf{k}_t)$ induces PSD rotating metric $M_t$ with $\|T_t\|_2 \le 1$ (+ `paired_b_gate`); updated $\mathbf{v}_{\text{eff}} = g_t\sqrt{\mathbf{w}_t}\odot\mathbf{v}_t$ (linear $\beta$); `surprise_mode="routed"`; `theta_low_rank=True` (`~1.92M–1.95M`/layer); `state_dtype="bfloat16"` (`81–89 MiB/layer`); removed `bool()` sync; added `Gate 0p-4` & `Gate 1-Ablate`. |

---

## 4. Pillar I: Subspace-Aligned Sandwich Loop & Dynamic Multi-Expert Expansion

### 4.1 Macro-Block Sandwich Unrolling (`64L` at $R=2$, `88L` at $R=3$)

`Qwen3.6-35B-A3B` consists of 10 identical hybrid macro-blocks $B_0, \dots, B_9$, each containing 3 linear-attention MoE layers followed by 1 gated full-attention MoE layer:

$$\underbrace{B_0 \to B_1}_{\text{Prelude } [0..7] \text{ (8L, 1 pass)}} \longrightarrow \left(\underbrace{B_2 \to B_3 \to B_4 \to B_5 \to B_6 \to B_7}_{\text{Looped Core } [8..31] \text{ (24L } \times R \text{ passes)}}\right) \longrightarrow \underbrace{B_8 \to B_9}_{\text{Coda } [32..39] \text{ (8L, 1 pass)}}$$

- **$R = 2$ (`LoopHSSD-64L`, Default S1–S3):** $8 + 24 \times 2 + 8 = \mathbf{64}$ virtual layers (`48` linear + `16` full-attention), **`~4.46B`** active compute parameters/token.
- **$R = 3$ (`LoopHSSD-88L`, `35B-A6B` Mode):** $8 + 24 \times 3 + 8 = \mathbf{88}$ virtual layers (`66` linear + `22` full-attention), **`~5.97B`** active compute parameters/token. Because `ReEntryBridge`, `PassRouterAdapters`, and per-pass LoRAs are indexed by pass $r \in \{1, \dots, R_{\max}\}$ with $R_{\max} = 3$ and trained via stochastic pass-depth curriculum (`train_pass_probs = [0.15, 0.50, 0.35]`), a **single checkpoint** supports runtime-selectable $R \in \{1, 2, 3\}$!

### 4.2 Decoupled `ReEntryBridge` & Zero-Gated Pass Exit (Fixes `A1, A2, A3`)

Let $h_7 \in \mathbb{R}^{B \times T \times 2048}$ denote the Prelude exit state (the exact representation distribution that physical Layer 8 was pretrained to consume), and let $h_{\text{core}}^{(r-1)} \in \mathbb{R}^{B \times T \times 2048}$ denote the output of Pass $r-1$ exiting Layer 31 ($h_{\text{core}}^{(0)} \equiv h_7$? No: Pass 1 enters Layer 8 directly with $\tilde{h}_{\text{in}}^{(1)} = h_7$ and produces $h_{\text{core}}^{(1)} = \operatorname{Core}^{(1)}(h_7)$!).

For each recurrent pass $r \in \{2, \dots, R\}$:

1. **Subspace-Aligned Re-Entry Bridge (`ReEntryBridge`, Entry into Layer 8):**
   $$\hat{u}^{(r-1)} = \operatorname{RMSNorm}_{r}\!\left(h_{\text{core}}^{(r-1)}\right), \qquad \Delta u_{\text{align}}^{(r)} = W_{\text{re,up}}^{(r)}\,\operatorname{SiLU}\!\left(W_{\text{re,down}}^{(r)}\,\hat{u}^{(r-1)}\right)$$
   $$h_{\text{rms}}^{(r-1)} = h_{\text{core}}^{(r-1)} \cdot \frac{\operatorname{RMS}(h_7)}{\operatorname{RMS}\!\left(h_{\text{core}}^{(r-1)}\right) + \epsilon}$$
   $$\tilde{h}_{\text{in}}^{(r)} = w_{r,0}\, h_7 + w_{r,1}\, h_{\text{rms}}^{(r-1)} + w_{r,2}\, h_{\text{core}}^{(r-1)} + \frac{\operatorname{RMS}(h_7)}{\sqrt{d}}\,\Delta u_{\text{align}}^{(r)} + \mathbf{e}_{\text{step}}^{(r)}$$
   where:
   - $\mathbf{w}_r = \operatorname{softmax}(\boldsymbol{\phi}_r) \in \Delta^2$ is a 3-way Birkhoff-simplex convex combination initialized to **`(0.85, 0.10, 0.05)`** (anchor-dominant so Pass $r$ starts cleanly inside Layer 8's input manifold, yet carrying a `15%` live semantic residual from Pass $r-1$ right from step 0!).
   - $W_{\text{re,down}}^{(r)} \in \mathbb{R}^{256 \times 2048}$ (orthogonal init) and $W_{\text{re,up}}^{(r)} \in \mathbb{R}^{2048 \times 256}$ (small normal init $\mathcal{N}(0, 0.01^2)$ during loop healing, or `0` prior to loop healing) form a **rank-256 linear/nonlinear subspace adapter** capable of rotating Layer 31's coordinate system back into Layer 8's basis (`Fix A3`).
   - $\mathbf{e}_{\text{step}}^{(r)} \in \mathbb{R}^{2048}$ (zero-init) informs shared routers and norms which deliberation pass is executing.

2. **Zero-Initialized Pass-Exit Gate (`PassExitGate`, Merge after Layer 31):**
   $$z_{\text{raw}}^{(r)} = \operatorname{Core}^{(r)}\!\left(\tilde{h}_{\text{in}}^{(r)}\right)$$
   $$h_{\text{core}}^{(r)} = h_{\text{core}}^{(r-1)} + \boldsymbol{\alpha}_{\text{exit}}^{(r)} \odot m_t^{(r)} \odot \left(z_{\text{raw}}^{(r)} - h_{\text{core}}^{(r-1)}\right)$$
   where $\boldsymbol{\alpha}_{\text{exit}}^{(r)} \in \mathbb{R}^{2048}$ is initialized to **exact zeros (`0.0`)** and $m_t^{(r)} \in \{0, 1\}$ is the causal MoR active-token mask (`1` when MoR is disabled).
   - **Step-0 Bit-Exactness (`x + 0 * delta == x`):** Because $\boldsymbol{\alpha}_{\text{exit}}^{(r)} = \mathbf{0}$, $h_{\text{core}}^{(R)} \equiv h_{\text{core}}^{(1)}$ bit-for-bit in IEEE-754 floating point at step 0!
   - **Non-Vanishing Step-1 Gradient (`Fix A2`):** Because $\tilde{h}_{\text{in}}^{(r)} \neq h_7$, $z_{\text{raw}}^{(r)} - h_{\text{core}}^{(r-1)}$ is immediately O(1) non-zero on step 0, so $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\alpha}_{\text{exit}}^{(r)}} = \left\langle \frac{\partial \mathcal{L}}{\partial h_{\text{core}}^{(r)}},\, z_{\text{raw}}^{(r)} - h_{\text{core}}^{(r-1)} \right\rangle \neq \mathbf{0}$ on the very first backward pass!

### 4.3 Task-Driven `PassRouterAdapters` (Breaking the 3B Active Ceiling)

For every core MoE layer $l \in [8..31]$ on Pass $r \ge 2$, let $x_{t,l}^{(r)} \in \mathbb{R}^{2048}$ be the pre-FFN normalized hidden state and $W_{\text{gate},l} \in \mathbb{R}^{256 \times 2048}$ the **frozen** pretrained router matrix:

$$\boldsymbol{\ell}_{t,l}^{(1)} = W_{\text{gate},l}\, x_{t,l}^{(1)}$$
$$\boldsymbol{\ell}_{t,l}^{(r)} = W_{\text{gate},l}\, x_{t,l}^{(r)} + B_{r,l}\,\operatorname{SiLU}\!\left(A_{r,l}\, x_{t,l}^{(r)}\right) + \mathbf{b}_{r,l} - \operatorname{softplus}(\tilde{\lambda}_{r,l})\, M_{t,l}^{(<r)} \qquad (r \ge 2)$$

- $A_{r,l} \in \mathbb{R}^{32 \times 2048}$ (Kaiming init), $B_{r,l} \in \mathbb{R}^{256 \times 32}$ (**zero-init**), $\mathbf{b}_{r,l} \in \mathbb{R}^{256}$ (**zero-init**).
- $M_{t,l}^{(<r)} \in \{0, 1\}^{256}$ marks experts already selected in Top-8 by earlier passes $\{1, \dots, r-1\}$ at the *same token position $t$* (strictly causal across time!), and $\gamma_{r,l} = \operatorname{softplus}(\tilde{\lambda}_{r,l})$ (initialized close to `0` via $\tilde{\lambda}_{r,l} = -6.0$, or exact zero-gated $\lambda_{r,l} = 0.0$ at step 0) is a **learned scalar repeat-suppression lever** trained **purely by task/KD loss** (`router_diversity_weight = 0.0` by default).
- If Layer $l$ benefits from re-invoking a general syntax/math expert on Pass 2, $B_{r,l}$ boosts it; if Layer $l$ benefits from recruiting fresh domain/verification experts from the remaining 248 experts, $\lambda_{r,l}$ and $B_{r,l}$ shift Top-8 into the natural `20%–55%` partial-reuse regime (`12–14` unique experts over 2 passes; `17–21` unique experts over 3 passes).

### 4.4 Per-Pass State Isolation & Recursive KV Sharing (`Fix A5`)

| Core Layer Type (`[8..31]`) | Count | Pass $r \ge 2$ State / KV Policy | Trainable Per-Pass Specialization | Extra KV / State Memory ($R=3$, `B=1`) |
| :--- | :---: | :--- | :--- | :--- |
| **Linear Attention (`GDN-2` or `HSSD v3.1`)** | `18` | **Independent State Slots** $\bar{S}_{t,l}^{(r)}$ & `conv1d` buffers per `(layer, pass)` (`+18` slots at $R=2$, `+36` slots at $R=3$) | Step-conditioned gates/routers (`+` optional per-pass Q LoRA) | `+72 MiB` (`GDN-2` `bf16`) or **`+2.85–3.13 GiB`** (`HSSD` `state_dtype="bfloat16"`, `81–89 MiB/slot`; `+5.69–6.26 GiB` in `fp32`) |
| **Gated Full Attention + `D-LSA`** | `6` | **`Shared KV` (`kv_share = True`)**: reuses $K_l^{(1)}, V_l^{(1)}$ & Pass-1 Ward centroids | Untied rank-64 LoRA $(W_Q^{(r)}, W_O^{(r)})$ + untied query indexer $W_{q\text{-idx}}^{(r)}$ | **`0.00 GiB`** (`1.00×` KV cache!) |

---

## 5. Pillar II: Dual-Track Linear Memory (`GDN-2` Baseline & `HSSD v3.1` Working Memory)

### 5.1 Dual-Track Risk-Mitigated Architecture & Non-Identity Retrofit Caveat

Because `Qwen3.6-35B-A3B` has 30 linear-attention layers carrying `75%` of sequence mixing, `v3.1` implements **two drop-in compatible engines** behind a unified layer interface:

1. **Track A — NVIDIA `GDN-2` (`src/qloop/model/gdn2.py`, Step-0 Bit-Exact Identity):**
   Upgrades scalar $(\beta_t, g_t)$ to channel-wise $(\mathbf{b}_t \in (0, 1)^{d_k}, \mathbf{w}_t \in (0, 1)^{d_v}, \mathbf{g}_t \in (-\infty, 0]^{d_k})$ via zero-initialized rank-64 adapters and the singularity-free logistic odds shift. **Zero forgetting at step 0 (`torch.equal`)**; guarantees `Gate 0b` loop validation never blocks on linear-layer surgery.
2. **Track B — `HSSD v3.1` (`src/qloop/model/hssd.py`, Mid-Horizon `513 -> 262K` Working-Memory Hypothesis):**
   Targets the `513 -> 262K` working-memory gap via disjoint horizon partitioning, phase-modulated DPLR memory, routed chunk-start root-surprise, and STE-budgeted $M=8$ multi-page states (`Fixes B1–B6` & Audit `3A–6`).

> [!IMPORTANT]
> **Why `HSSD v3.1` Requires `Stage S1b` Distillation Despite Zero-Init Up-Projections:** Even when every residual adapter up-projection (`erase_up`, `write_up`, `decay_up`, `theta_up`, `swa_q_up`, `swa_k_up`, `swa_v_up`) is initialized to exact zeros and linear-$\beta$ write scaling ($\sqrt{\mathbf{b}_t}\sqrt{\mathbf{w}_t} = \beta_{\text{base}}$) is enforced, **`HSSD v3.1` is NOT a step-0 functional identity** to vanilla single-state GDN. Relative to stock GDN, HSSD structurally modifies (i) horizon partitioning (`[513, 576]` causal local SWA blended with delayed `S_boundary[max(0, n-8)]` via `PeakBoltzmannGate`), (ii) multi-page Top-$k$ write/read routing across $M=8$ pages, (iii) chunk-start root-surprise attenuation ($g_{\tau,h} \in (0, 1)$), and (iv) log-spaced page decay priors ($\gamma_{\text{prior},m} \in [10^{-4}, 5 \times 10^{-2}]$). Consequently, freezing Qwen3.6's `32.338B` MoE experts preserves all pretrained FFN weights (`SHA-256` verified), while **`Stage S1b` (`Gate 1`) layer-output MSE + chunked-vocabulary KL distillation** from the `Track A` teacher is essential to align the representations entering those frozen experts (preceded by `Gate 1-Ablate` 6-way mechanism isolation).

### 5.2 `HSSD v3.1` Complete Mathematical & Systems Specification

Fix physical linear layer $l$, pass $r$, chunk size $C = 64$, window $W = 8$ chunks, and head dimensions $H_k = 16, H_v = 32, d_k = d_v = 128$ (`2` value heads per key head, matching Qwen3.6 GDN). Let $t \in \{1, \dots, T\}$ be the **1-based** logical token index (unpadded per sequence when a 2D `attention_mask` is supplied, so masked pad tokens never advance logical `step_count` or chunk boundaries, and reset to `0` at every `cu_seqlens` / `seq_idx` packed-document boundary) and $n = \lfloor (t - 1)/C \rfloor \in \{0, 1, \dots\}$ be the **0-based** chunk index.

#### (1) Strictly Disjoint Chunk-Aligned Horizon Partition (`Fix B1`)
Define the number of completed chunks assigned to far-horizon linear memory as:
$$k_{\text{far}}(n) = \max(0,\, n - W), \qquad t_{\text{cut}}(n) = C \cdot k_{\text{far}}(n) = 64 \max(0,\, n - 8)$$
- **Far-Horizon Prefix:** $\mathcal{I}_{\text{far}}(t) = \{1, \dots, t_{\text{cut}}(n)\}$, summarized **exclusively** by the archived chunk-boundary state $\bar{S}_{\text{boundary}}[k_{\text{far}}(n)] \equiv \text{boundary\_ring}[0]$ (where $\bar{S}_{\text{boundary}}[0] = \mathbf{0}$ and $\bar{S}_{\text{boundary}}[k]$ stores the exact $M$-page state bank after token $kC$).
- **Exact Local Sliding Window:** $\mathcal{I}_{\text{loc}}(t) = \{t_{\text{cut}}(n) + 1, \dots, t\}$, computed via exact causal softmax attention (`SDPA` / circular `576`-slot decode ring).
- **Exact Partition Invariant (`0` Overlap, `0` Gap):**
  $$\mathcal{I}_{\text{far}}(t) \cap \mathcal{I}_{\text{loc}}(t) = \varnothing, \qquad \mathcal{I}_{\text{far}}(t) \cup \mathcal{I}_{\text{loc}}(t) = \{1, \dots, t\}, \qquad |\mathcal{I}_{\text{loc}}(t)| = t - 64\max(0, n-8) \in [513, 576] \quad (\forall t > 512)$$
  For $t \le 512$ ($n \le 7$), $k_{\text{far}}(n) = 0$, $\bar{S}_{\text{boundary}}[0] = \mathbf{0}$, and $\mathcal{I}_{\text{loc}}(t) = \{1, \dots, t\}$ covers 100% of the prefix.

#### (2) `PeakBoltzmannGate` Local/Far Branch-Selection Heuristic (`Fix B5`, Qualified)
Let $\mathbf{z}_{t,h} \in \mathbb{R}^{N_t}$ ($N_t = |\mathcal{I}_{\text{loc}}(t)|$) be the scaled local SWA logits $\frac{\mathbf{q}_{t,h}^{\text{swa}} (\mathbf{K}_{\mathcal{I}_{\text{loc}},h}^{\text{swa}})^\top}{\sqrt{d_k}}$. `PeakBoltzmannGate` computes:
$$\Psi_{t,h}^{\text{loc}} = \frac{1}{2}\!\left(\operatorname{LSE}(\mathbf{z}_{t,h}) - \ln N_t\right) + \frac{1}{2}\!\left(\max_{j \in \mathcal{I}_{\text{loc}}(t)} z_{t,h,j} - \sqrt{2 \ln \max(N_t, 2)}\right)$$
$$p_{t,h}^{\text{hit}} = \sigma\!\left(\kappa_h \left(\Psi_{t,h}^{\text{loc}} - \tau_h\right)\right), \qquad \mathbf{o}_{t,h} = p_{t,h}^{\text{hit}}\, \mathbf{o}_{t,h}^{\text{loc}} + \left(1 - p_{t,h}^{\text{hit}}\right) \mathbf{o}_{t,h}^{\text{far}}$$
- **Exact Distributional Qualification:** Under **independent zero-mean standard sub-Gaussian background scores** ($z_j \sim \mathcal{N}(0, 1)$), $\mathbb{E}[\operatorname{LSE}(\mathbf{z}) - \ln N_t] \approx \frac{1}{2}$ and $\max_{j \le N_t} z_j$ concentrates around $\sqrt{2\ln N_t}$, centering mean and extreme-value growth across $N_t \in [2, 576]$. By contrast, for **constant scores** $z_{t,h,j} \equiv c$, $\operatorname{LSE}(\mathbf{z}) - \ln N_t = c$ whereas $\max_j z_j - \sqrt{2\ln\max(N_t, 2)} = c - \sqrt{2\ln\max(N_t, 2)}$, so $\Psi_{t,h}^{\text{loc}}(c\mathbf{1}) = c - \frac{1}{2}\sqrt{2\ln\max(N_t, 2)}$ retains mild $O(\sqrt{\ln N_t})$ dependence. Accordingly, `PeakBoltzmannGate` is treated as a **learned per-head local/far branch-selection heuristic** ($\kappa_h = \exp(\text{log\_kappa}_h), \tau_h$) calibrated empirically during distillation across **local-hit**, **local-miss**, and **high-similarity local-distractor** regimes.

#### (3) Post-Rotation Scaled Over-Relaxed Householder & Rotating Erase Metric (Theorem 1 Exact Order)
Each channel pair $(2c, 2c+1)$ ($c \in \{0, \dots, d_k/2 - 1\}$) shares paired decay $\alpha_{t,h,c} \in (0, 1]$ and learned data-dependent phase step $\theta_{t,h,c} = \pi \tanh(\ell_{\theta,t,h,c}) \in (-\pi, \pi)$ computed via **pure linear low-rank factorization** (`theta_low_rank=True`, default `rank=64`: $\boldsymbol{\ell}_{\theta,t} = W_{\theta,\text{up}}(W_{\theta,\text{down}} x_t)$ with zero-initialized $W_{\theta,\text{up}}$), accumulated in **`torch.float64`** as $\Theta_{t,h,c} = \sum_{\tau=1}^t \theta_{\tau,h,c} \pmod{2\pi}$ (`< 2 × 10^{-6}` rad chunk-vs-recurrent drift at `1,000,000` steps). Over-relaxed erase gates satisfy $\mathbf{b}_{t,h} = 2\,\operatorname{logistic\_shift}(0.5\,\beta_{t,h}, \Delta\mathbf{b}_{t,h}) \in (0, 2)^{d_k}$.

- **Exact Implemented Operation Order (`Scaling After Counter-Rotation`):** Given L2-normalized $\mathbf{k}_{t,h}$ ($\|\mathbf{k}_{t,h}\|_2 = 1$) and page root-surprise $g_{t,h,m} \in [0, 1]$, `HSSD v3.1` first counter-rotates into the stationary frame and **then** applies channel-wise $\sqrt{\mathbf{b}_{t,h}}$ scaling:
  $$\hat{\mathbf{k}}_{t,h} = R(-\boldsymbol{\Theta}_{t,h})\,\mathbf{k}_{t,h}, \qquad \hat{\mathbf{q}}_{t,h} = R(-\boldsymbol{\Theta}_{t,h})\,\mathbf{q}_{t,h}, \qquad \mathbf{e}_{t,h,m} = g_{t,h,m}\,\sqrt{\mathbf{b}_{t,h}} \odot \hat{\mathbf{k}}_{t,h}$$
  updating stationary-frame page state $\bar{S}_{t,h}^{(m)} = (I - \mathbf{e}_{t,h,m}\mathbf{e}_{t,h,m}^\top) D(\boldsymbol{\alpha}_{t,h,m}) \bar{S}_{t-1,h}^{(m)} + \mathbf{e}_{t,h,m}(\mathbf{v}_{t,h,m}^{\text{eff}})^\top$ with readout $\mathbf{o}_t = (\bar{S}_t)^\top \hat{\mathbf{q}}_t$.
- **Induced Rotating Erase Metric & Unconditional Contraction (`||T_t||_2 <= 1`):** Mapping back to unrotated coordinates $S_t = R(\boldsymbol{\Theta}_t)\bar{S}_t$ gives $R(\boldsymbol{\Theta}_t)\mathbf{e}_t = g_t M_t^{1/2}\mathbf{k}_t$, inducing the **symmetric positive-semidefinite rotating $2\times 2$-block erase metric**:
  $$M_t \triangleq R(\boldsymbol{\Theta}_t)\operatorname{diag}(\mathbf{b}_t) R(-\boldsymbol{\Theta}_t) = M_t^\top \succeq 0$$
  Because $R(-\boldsymbol{\Theta}_t) \in SO(d_k)$ is norm-preserving, for any $\|\mathbf{k}_t\|_2 = 1$, $g_t \in [0, 1]$, $\mathbf{b}_t \in [0, 2]^{d_k}$, and $\boldsymbol{\gamma}_t \ge \mathbf{0}$ ($D_t = \operatorname{diag}(\exp(-g_t^2 \boldsymbol{\gamma}_t))$ with $\|D_t\|_2 \le 1$):
  $$\|\mathbf{e}_t\|_2^2 = g_t^2 \sum_{i=1}^{d_k} b_{t,i}\left(R(-\boldsymbol{\Theta}_t)\mathbf{k}_t\right)_i^2 \le \left(\max_i b_{t,i}\right) g_t^2 \|\mathbf{k}_t\|_2^2 \le 2 \implies \left\|I - \mathbf{e}_t\mathbf{e}_t^\top\right\|_2 = \max\!\left(1,\, \left|1 - \|\mathbf{e}_t\|_2^2\right|\right) = 1$$
  so the homogeneous transition $T_t = (I - \mathbf{e}_t\mathbf{e}_t^\top) D_t$ satisfies **$\|T_t\|_2 \le 1$ unconditionally**!
- **Strict Commutativity Specialization (`paired_b_gate=True`):** Note that $\operatorname{diag}(\sqrt{\mathbf{b}_t})$ and $R(-\boldsymbol{\Theta}_t)$ commute strictly ($M_t \equiv \operatorname{diag}(\mathbf{b}_t)$ for all $\boldsymbol{\Theta}_t$) **if and only if** $\mathbf{b}_t$ is pairwise tied across each rotary plane ($b_{t,2c} = b_{t,2c+1}$), which is directly supported via `paired_b_gate=True` in `HSSDConfig`.

#### (4) Multi-Page Routed Chunk-Start Root-Surprise (`boundary_ring[-1]`) & Linear-$\beta$ Write Scaling (`_safe_sqrt(w_gate)`)
In `HSSDCacheState`, `boundary_ring` holds the $W + 1 = 9$ multi-page boundary snapshots $(\bar{S}_{(n-W)C}, \dots, \bar{S}_{nC})$, so **`boundary_ring[-1]` (`[B, M, H_v, d_k, d_v]`) is identically the chunk-start state bank $\bar{S}_{nC}$ across all $M = 8$ pages** (eliminating the redundant `2 MiB/layer` `s_chunk_start` tensor while keeping `.s_chunk_start` as a backward-compatible `@property` returning `boundary_ring[-1][:, 0]`).

For every token $\tau \in [nC + 1, (n+1)C]$ in chunk $n$, write-router probabilities $\boldsymbol{\pi}_\tau^{\text{write}} = \operatorname{softmax}(W_{\text{write}} x_\tau)$ and renormalized Top-$k_w$ weights $\widetilde{\pi}_{\tau,m}^{\text{write}}$ ($\sum_{m \in \mathcal{T}_\tau} \widetilde{\pi}_{\tau,m}^{\text{write}} = 1$) are computed first. Under `surprise_mode="routed"` (default; `"page0"` restricts to Page 0 for ablations), prediction error is evaluated in parallel against the **Top-$k_w$ write-routed pages of `boundary_ring[-1]`**:
$$\hat{\mathbf{v}}_{\tau,h}^{\text{routed}} = \sum_{m \in \operatorname{Top-}k_w(\boldsymbol{\pi}_\tau^{\text{write}})} \widetilde{\pi}_{\tau,m}^{\text{write}} \left(\bar{S}_{nC,h}^{(m)}\right)^\top \hat{\mathbf{k}}_{\tau,h}, \qquad \varepsilon_{\tau,h} = \frac{\left\|\mathbf{v}_{\tau,h} - \hat{\mathbf{v}}_{\tau,h}^{\text{routed}}\right\|_2}{\left\|\mathbf{v}_{\tau,h}\right\|_2 + \epsilon}$$
so associations already stored on non-zero pages (`m = 1..7`) properly yield low $\varepsilon_{\tau,h}$ and suppress redundant writes! Parameterizing root-surprise $g_{\tau,h} = \sigma(a_h(\varepsilon_{\tau,h} - \eta_h) + \mathbf{w}_{g,h}^\top x_\tau + b_{g,h}) \in (0, 1)$ ($s_{\tau,h} = g_{\tau,h}^2$) and per-page $g_{\tau,h,m} = \sqrt{\widetilde{\pi}_{\tau,m}^{\text{write}}}\, g_{\tau,h}$, the effective erase, **linear-$\beta$ write**, and decay factors are:
$$\mathbf{e}_{\tau,h,m} = g_{\tau,h,m}\,\sqrt{\mathbf{b}_{\tau,h}} \odot \hat{\mathbf{k}}_{\tau,h}, \qquad \mathbf{v}_{\tau,h,m}^{\text{eff}} = g_{\tau,h,m}\,\sqrt{\mathbf{w}_{\tau,h}} \odot \mathbf{v}_{\tau,h}, \qquad \boldsymbol{\alpha}_{\tau,h,m}^{\text{eff}} = \exp\!\left(-g_{\tau,h,m}^2\,\boldsymbol{\gamma}_{\tau,h,m}\right)$$
- **Linear-$\beta$ Write Scaling (`_safe_sqrt(w_gate)`):** At zero adapter up-projection init ($\Delta\mathbf{b} = \mathbf{0}, \Delta\mathbf{w} = \mathbf{0} \implies \mathbf{b}_{\tau,h} = \mathbf{w}_{\tau,h} = \beta_{\text{base}}\mathbf{1}$), the rank-1 outer product $\mathbf{e}_{\tau,h,m}(\mathbf{v}_{\tau,h,m}^{\text{eff}})^\top$ scales as $\sqrt{\mathbf{b}_{\tau,h}}\sqrt{\mathbf{w}_{\tau,h}} = \beta_{\text{base}}$ (eliminating the previous $\sqrt{\beta_{\text{base}}}\cdot\beta_{\text{base}} = \beta_{\text{base}}^{3/2}$ write attenuation!).
- **Overflow-Free Sync-Free WY Solve:** Because $\log \boldsymbol{\alpha}_{\tau,h,m}^{\text{eff}} \le 0$, cumulative intra-chunk log-decay $G_i$ is monotonically non-increasing ($G_i - G_j \le 0$ for all strict lower-triangular $i > j$ and $G_{C-1} - G_j \le 0$), so `_wy_kda_state_update` evaluates `exp` directly with **zero host-GPU `bool()` / `.item()` synchronization** and zero `fp32` overflow even at extreme decay spans.

#### (5) Tiered $M=8$ Page Router & Actual Sparse Top-$k$ STE Write-Mass Budget (`Fixes H1, B2, Audit 3C`)
Each head bank maintains $M = 8$ state pages $\{\bar{S}_{t,h}^{(m)}\}_{m=0}^7$ with log-spaced decay priors $\gamma_{\text{prior},m} \in [10^{-4}, 5 \times 10^{-2}]$ and target write-duty caps $\mu_m^*$:

| Page Tier | Pages | Target Horizon (Hypothesis) | Max Sparse Duty $\mu_m^*$ | Anti-Collapse Floor $\mu_{m,\min} = 0.2\mu_m^*$ | Expected Active Writes / `65K` | Cumulative Write Mass $\mathcal{B}_m(65\text{K})$ ($\bar{b} \approx 0.25$) | Analytical Signal Bound $\exp(-\mathcal{B}_m/128)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vault / Schema Pages** | `m = 0, 1` | `64K – 262K` | **`0.0025`** (`0.25%`) | `0.0005` | `~164` | **`~41.0` (`0.32 d_k`)** | **`~72.6%`** |
| **Module / Episode Pages** | `m = 2, 3` | `8K – 64K` | **`0.0250`** (`2.5%`) | `0.0050` | `~1,638` | `~41.0` per `6.5K` | `~72.6%` @ `6.5K` |
| **Working Scratchpad** | `m = 4, 5, 6, 7` | `513 – 8K` | **`0.2500`** (`25%`) | `0.0500` | `~16,384` | High-churn scratchpad | Rapidly recycled |

Let $\widetilde{\boldsymbol{\pi}}_{b,t} \in \Delta^{M-1}$ be the **actual renormalized Top-$k_w$ (`k_w = 2`) sparse write distribution** ($\widetilde{\pi}_{b,t,m} = 0$ for all $m \notin \operatorname{Top-}k_w(\boldsymbol{\pi}_{b,t})$). Evaluating budget loss on dense $\boldsymbol{\pi}_{b,t}$ allowed $6$ unselected dead pages to receive zero writes while hiding behind uniform softmax probabilities. `HSSD v3.1` evaluates on **actual sparse write mass** with Straight-Through Estimation (`STE`):
$$\boldsymbol{\pi}_{b,t}^{\text{ste}} = \widetilde{\boldsymbol{\pi}}_{b,t} + \left(\boldsymbol{\pi}_{b,t} - \operatorname{sg}\!\left(\boldsymbol{\pi}_{b,t}\right)\right), \qquad \widetilde{u}_m = \frac{1}{N_{\text{valid}}}\sum_{b,t} \pi_{b,t,m}^{\text{ste}} \left(\frac{1}{H_v}\sum_{h=1}^{H_v} g_{b,t,h}^2\right)$$
$$\mathcal{L}_{\text{budget}} = \lambda_{\text{bud}} \sum_{m=0}^{7} \left(\operatorname{ReLU}\!\left(\widetilde{u}_m - \mu_m^*\right)^2 + \operatorname{ReLU}\!\left(\mu_{m,\min} - \widetilde{u}_m\right)^2\right)$$
Forward values match $\widehat{\mathbb{E}}[g_t^2 \widetilde{\pi}_{t,m}]$ identically (surfacing `dead_page_count` immediately), while backward gradients flow smoothly into unselected `write_router` logits via $\boldsymbol{\pi}_{b,t}$. **Important Caveat:** Because $\mathcal{L}_{\text{budget}}$ penalizes *expected* write duty over training batches, it acts as a **soft training regularizer** rather than a hard deterministic token-level cap on burst writes at inference. Readout gathers only the Top-$k_r = 2$ read pages:
$$\mathbf{o}_{t,h}^{\text{far}} = \sum_{m \in \operatorname{Top-}k_r(\boldsymbol{\pi}_{t}^{\text{read}})} \widetilde{\pi}_{t,m}^{\text{read}} \left(\bar{S}_{\text{boundary},m}[k_{\text{far}}(n)]\right)^\top \hat{\mathbf{q}}_{t,h}$$

#### (6) Realistic Memory, Parameter, Decode-Cache & Kernel-Fusion Accounting (`Fixes B6, Audit 3A–3B, 5–6`)
At Qwen3.6 dimensions (`hidden = 2048, H_v = 32, d_k = d_v = 128, M = 8, C = 64, W = 8`, `B = 1`):
- **Factored Decay Activations (`Audit 3B`):** `project_inputs` keeps `gamma_page_scale` (`[8]`) and `gamma_chan_pair` (`[B, T, 32, 64]`) factored at **`8 KiB/tok/layer` in `fp32`** (`16×` smaller than the eliminated `128 KiB/tok/layer` = `32 GiB/layer @ 262K` 5D `gamma` tensor), expanding across $M=8$ only inside `C=64` chunks under `1,024`-token segment checkpointing.
- **Low-Rank Phase Parameters (`theta_low_rank=True, rank=64`):** Cuts `HSSDAdapter` from `5.88M/layer` (`176.5M` across 30L, where `theta_proj` took `71.3%`) down to **`~1.92M–1.95M/layer` (`~57.5M–58.5M` across 30L)**.
- **Exact Retained Decode Cache per `(layer, pass)` Slot (`HSSDCacheState.unique_storage_bytes()`, zero prompt-length leak):**
  - `page_states` (`[1, 8, 32, 128, 128]`): `16.00 MiB` (`fp32`) / `8.00 MiB` (`bf16`)
  - `boundary_ring` (`W` to `W + 1 = 9` snapshots of `[1, 8, 32, 128, 128]`): `128.00–144.00 MiB` (`fp32`) / `64.00–72.00 MiB` (`bf16`)
  - `swa_k_ring` + `swa_v_ring` (`2 × [1, 576, 32, 128]`): `18.00 MiB` (`fp32`) / `9.00 MiB` (`bf16`)
  - `theta_cum` (`[1, 32, 64]` in **`float64`**, compact cloned): `16.0 KiB` + `swa_valid_ring` (`0.56 KiB`) + `step_count` (`8 B`)
  - **Total Retained Decode Cache (`B = 1`):**
    - **`fp32` state storage:** **`162–178 MiB/layer`** $\implies$ **`4.75–5.21 GiB` across 30 layers (`R = 1`)**, scaling with independent core-pass slots to **`7.59–8.34 GiB` (`48` slots, `R = 2`)** and **`10.44–11.47 GiB` (`66` slots, `R = 3`)**.
    - **`bfloat16` state storage (`state_dtype="bfloat16"`, default for GPU serving & long-context):** **`81–89 MiB/layer`** $\implies$ **`2.37–2.61 GiB` across 30 layers (`R = 1`)**, **`3.80–4.17 GiB` (`48` slots, `R = 2`)**, and **`5.22–5.74 GiB` (`66` slots, `R = 3`)**, while still computing WY chunk updates in `fp32` and keeping `theta_cum` in `fp64`!
- **Active-Page Decode, Compact Lagged-Replay Cache & Blackwell Speed Stack (`v3.2` `H1–H9`, `5.2–5.3`):**
  - **Full-Strength Selected-Page Writes (`write_mode="full"`, `H1/H6`) & Local SWA Recency Slope (`H3`):** Eliminates the `0.5x` split-write averaging artifact so exact overwrites return `v_new`, while a learnable per-head SWA recency slope resolves repeated-key collisions inside the `513–576` local window.
  - **Read-Ahead Far Readout (`H4`) & Lagged-Replay Decode Cache (`H5`):** Precomputing `_far_readout` at boundary creation removes the 9-entry `boundary_ring` from segment activation checkpoints (`~5×–9×` lower retained training state), while single-token decode stores `1` lagged boundary bank + a `512`-token replay buffer (`~3×–4×` smaller per pass-layer).
  - **Zero-Duplicate-VRAM Distillation (`SharedBackboneTeacher` + Online Per-Layer Warm-Up, `5.2 #2, #7`):** `s1b_hssd_distill` shares the single `35 GiB` FP8 backbone (`0.00 GiB` duplicate frozen expert HBM), running `400` teacher-forced single-layer warm-up steps (`~0.5 GiB` peak extra VRAM, zero MoE backward) followed by scheduled $\lambda$-blend (`blend_init=0.0 -> 1.0`, `H2`) end-to-end `_ChunkedVocabKL`.
  - **Global TF32 (`allow_tf32=True`, `5.2 #1`) & Zero-Host-Sync `FusedStepOptimizer` (`5.3`):** Routes `fp32` WY/Ward matmuls onto Blackwell TF32 Tensor Cores (`< 2e-2` 32K relative drift with `fp32` cumulative decay & `fp64` phase reduction), eliminates per-parameter `.item()` GPU$\to$CPU syncs via on-device norm/clipping + `defer_micro_step()`, and sets `apollo: {rank: 256, scale_type: channel, scale: 1.0, update_proj_gap: 200}` for full-rank $\ge 128 \times 128$ matrices.

---

## 6. Pillar III: Two-Tier Ultra-Long Context (`1M` Primary + `10M` Stretch)

### 6.1 Positional Extrapolation & Read-Side Calibration (`Fix C1`)

Each of Qwen3.6's 10 Gated Full-Attention layers uses head dimension $d_{\text{head}} = 256$ split into **`64` mRoPE dimensions** (`[0:64]`, native `max_position_embeddings = 262,144`, `rope_theta = 10,000,000`) and **`192` NoPE position-free content dimensions** (`[64:256]`), plus a pretrained **head-specific sigmoid output gate** (`q_proj` outputs $4096 + 4096 = 8192$ dims) that naturally suppresses attention-sink noise ([Qiu et al., NeurIPS 2025](RESEARCH_ESSAY.md)).

To bridge `262K -> 1M` (and `10M` stretch) without blind zero-shot extrapolation:
1. **Gate `0p-3` Positional Mechanism Selection:** Benchmarks (a) `YaRN` (`factor = 4.0` for `1M`) vs. (b) **Dual Chunk Attention / ReRoPE Clamping** (exact relative mRoPE inside `131,072` tokens; smoothly clamped asymptotic relative phase $\Delta_{\text{eff}} = L_c + S_c \ln(1 + (\Delta - L_c)/S_c) \le 240{,}000$ for distant retrieved buckets so rotary angles **never leave the pretrained `[0, 262K]` arc**!).
2. **Long-Prefix (`no_grad`) + Short-Suffix (`grad`) Calibration (`Gate 2a`, `v3.2` `D1–D6`):** `s3_long_1m_lsa` (`600` steps) alternates `64K` `hybrid` D-LSA steps (exposing downstream Q/O LoRA and loop bridges to real sparse attention while supervising `W_q_idx, W_k_idx` via sampled KL, with `compact_active_slots=True` `2×` smaller `BucketIndex` and `max_selected_keys=768`) with scheduled `prefix_suffix_calibration_step` passes every `10` steps—streaming `512K–992K` prefix tokens in sub-quadratic `sparse` D-LSA mode (`D3`) and computing `32K` suffix loss directly via chunked cross-entropy on pre-head hidden states without ever materializing `[B, 32768, 248320]` logits (`D2`).

### 6.2 Tier 1 (`1M` Primary Target): 100% GPU-Resident `FP8` + Per-Pass `D-LSA`

At $L = 1{,}048{,}576$ (`1M` tokens), zero PCIe host offload is required:
- **Exact `1M` GPU KV Footprint (`kv_share = True` across all passes $R \in \{1, 2, 3\}$):**
  $$1{,}048{,}576 \times 10\text{ layers} \times 2\text{ (K,V)} \times 2\text{ KV heads} \times 256\text{ dim} \times 1\text{ B (FP8 E4M3)} = \mathbf{10.00\text{ GiB}}$$
- **GPU Ward Centroid Directory (`src/qloop/model/lsa.py`):**
  Projects only the `192` NoPE content channels (`k[..., 64:256]`) to `64-d` L2-normalized descriptors, applies `4:1` unit pooling + batched GPU Ward clustering (`64` buckets per `512`-token block $\implies$ `131,072` centroids/layer):
  $$131{,}072 \times 10\text{ layers} \times 2\text{ KV heads} \times 64\text{ dim} \times 2\text{ B (BF16)} = \mathbf{320\text{ MiB}}$$
- **Per-Pass Dynamic Context Re-Indexing:** Pass 1 uses $W_{q\text{-idx}}^{(1)}$ to retrieve initial problem context (`Top-64` buckets + `512` local window + `64` sinks $\approx 2{,}048\text{–}2{,}624$ exact tokens). Pass 2 (and Pass 3 in `A6B` mode) uses untied $W_{q\text{-idx}}^{(r)}$ on the refined latent query to retrieve **new supporting or counterexample buckets** from the same `10.00 GiB` KV cache with zero re-clustering!

### 6.3 Tier 2 (`10M` Stretch Target): Host-DDR5 Block-64 Hadamard `INT4` Offload (`Fixes C2, D1–D4`)

When scaling from `1M` to $L = 10{,}000{,}000$ (`10M` tokens, `Gate 2b`):

1. **Block-Diagonal $H_{64}$ Walsh-Hadamard Transform (`Fix C2`):**
   Define the self-adjoint orthogonal operator acting independently on each of the four `64`-channel quantization groups of $d_{\text{head}} = 256$:
   $$\mathcal{H}_{\text{blk64}} = \operatorname{diag}\!\left(H_{64},\, H_{64},\, H_{64},\, H_{64}\right) \in \mathbb{R}^{256 \times 256}, \qquad \mathcal{H}_{\text{blk64}}^\top = \mathcal{H}_{\text{blk64}}, \qquad \mathcal{H}_{\text{blk64}}^2 = I_{256}$$
   - Group 0 (`[0:64]`) transforms **only** the 64 mRoPE channels; Groups 1–3 (`[64:128], [128:192], [192:256]`) transform **only** the 192 NoPE content channels.
   - **Exact ReRoPE Commutativity:** Under bucket-level ReRoPE (`1` relative bucket angle per retrieved 8-token bucket, or raw NoPE + rotary split), rotating query slice `q[0:64]` at query time and applying $H_{64}$ preserves $\langle \mathcal{H}_{\text{blk64}} \mathbf{q}_{\text{clamped}},\, \mathcal{H}_{\text{blk64}} \mathbf{k} \rangle \equiv \langle \mathbf{q}_{\text{clamped}},\, \mathbf{k} \rangle$ to `< 1e-12` with **zero cross-contamination** into the 192 content channels (`test_v3_06`)!
   - **Value Reconstruction:** Stored values $\tilde{\mathbf{v}} = \operatorname{Quant}_{\text{INT4,g64}}(\mathcal{H}_{\text{blk64}} \mathbf{v})$ are reconstructed after attention aggregation by linearity: $\hat{\mathbf{o}}_v = \mathcal{H}_{\text{blk64}}\!\left(\sum_{j \in \mathcal{S}} \alpha_j \operatorname{Dequant}(\tilde{\mathbf{v}}_j)\right)$ (only **one** fast Walsh-Hadamard transform per query head instead of `2,048` per key!).
   - **No `m=d` 1-bit QJL Sketch (Theorem 3):** Plain Group-64 `INT4` after $H_{64}$ achieves `< 0.085` relative error, whereas adding an `m = 256` 1-bit QJL sketch degrades MSE by $\frac{\pi}{2} \approx 1.57\times$ (`test_v3_06`).

2. **Exact `10M` Memory & PCIe 5.0 Bandwidth Ledger:**
   - **Group-64 `INT4` Storage Density:** `64 × 0.5 B` (`32 B` packed nibbles) + `2 B` (`fp16` scale) + `2 B` (`fp16` zero-point) = **`36 B / 64 scalars` (`0.5625 B/scalar` = `4.5 bits/scalar`)**.
   - **4-Token Contiguous Micro-Page:** $4 \times 2\text{ (K,V)} \times 2\text{ heads} \times 256 \times 0.5625\text{ B} = \mathbf{2{,}304\text{ Bytes}}$ (`36` cache lines, zero PCIe scatter-gather amplification).
   - **`10M` Host Pinned DDR5 Cache (`10` Full-Attn Layers):**
     $$10{,}000{,}000 \times 10 \times 2 \times 2 \times 256 \times 0.5625\text{ B} = \mathbf{57.60\text{ GB}} = \mathbf{53.64\text{ GiB}}$$
   - **`10M` GPU HBM Centroid Index (`fp8` centroids + metadata):** **`~1.60–2.40 GiB`** on GPU HBM.
   - **3-Layer Speculative PCIe 5.0 Prefetch (Accounting for Scatter-Gather Tail Latency):** Before macro-block $B_m$'s full-attention layer (`3` HSSD+MoE layers ahead), the hidden state predicts Top-64 Ward buckets (`512` micro-pages $\approx$ `1.18 MB` per KV head; `250–300` uncached `2.25 KiB` micro-pages $\approx$ `0.6–1.2 MiB` per layer). Accounting for realistic PCIe round-trip and DDR5 memory-controller scatter-gather queue latency, asynchronous pinned gather takes **`0.15–0.60 ms`/layer**—still achieving a **`>5×` overlap margin** behind the **`3.0–4.2 ms`** single-stream compute window of the 3 preceding HSSD+MoE layers, keeping synchronous residual delta fetches (`~12%` miss rate) below `0.05–0.15 ms`/layer!
