# Post-Training Architectural Evolution of Sparse Hybrid Transformers

**Expanding `Qwen3.6-35B-A3B` into an Effective `35B-A6B` Recursive Reasoner via Subspace-Aligned Looping, Budgeted Horizon-Split Delta Memory, and Commutative Block-Hadamard Sparse Attention**

**Technical Research Monograph — Loop-HSSD v3 (`Qwen3.6-35B-A6B-LoopHSSD`)**
*Companion to [Architecture Specification](ARCHITECTURE_SPEC.md), [Implementation Plan](IMPLEMENTATION_PLAN.md), and Double-Precision Verification Suite [`test_loop_hssd_v3_math.py`](../tests/test_loop_hssd_v3_math.py)*

---

## Abstract

Ultra-sparse Mixture-of-Experts (MoE) hybrid language models such as `Qwen3.6-35B-A3B` store vast world knowledge across `34.66B` text parameters (`32.21B` in 256 routed experts per layer), yet execute only `~2.95B` active parameters across `40` physical layers (`30` scalar-gated Gated DeltaNet layers and `10` gated full-attention layers) per generated token. While pretraining a denser or deeper 35B backbone from scratch requires prohibitive multi-trillion-token compute, post-training *backbone topology evolution*—reusing `>98.5%` of pretrained weights while keeping all `32.34B` MoE FFN parameters frozen—offers a tractable path to substantially higher sequential circuit depth, per-token active capacity, and long-horizon memory fidelity.

In this paper, we formalize **Loop-HSSD v3 (`Qwen3.6-35B-A6B-LoopHSSD`)**, a cohesive three-pillar architectural evolution synthesized through three iterative adversarial review cycles (`v1 -> v2 -> v3`). First, we diagnose why naive weight-tied looping collapses on Pre-LN sparse MoEs and prove that decoupling a **low-rank linear subspace re-entry bridge** from a **zero-initialized pass-exit gate** resolves the tension between step-0 numerical bit-identity and cross-pass expert diversification, scaling effective active compute from `2.95B` (`40L`) to `4.46B` (`64L`, $R=2$) and **`5.97B` (`88L`, $R=3$, `35B-A6B` mode)** while recruiting distinct frozen experts across passes. Second, to eliminate the mid-horizon associative washout cliff (`513 -> 262K` tokens) in linear attention, we derive **Horizon-Split Surprise-Gated Delta Memory (`HSSD v3`)**: we prove an exact isomorphism reducing 2D complex-phase over-relaxed Householder transitions to pure real `chunk_kda` kernels (Theorem 1), establish a strictly disjoint chunk-aligned local/far horizon split with nonsingular chunk-start root-surprise gating ($s = g^2$), and prove the **Householder Cumulative Write-Mass Retention Law** $\mathbb{E}[\langle S_H^\top k_*, v_* \rangle] \approx \beta_0 \exp(-\mathcal{B}(H)/d_k)$ (Theorem 2), demonstrating why uniform MoE load-balancing destroys long-horizon memory pages and deriving log-spaced write-mass budgets. Third, for `1M–10M` ultra-long context, we pair `1M` GPU-resident `FP8` Distilled Latent Sparse Attention (`D-LSA`) with a `10M` Host-DDR5 **Block-64 Walsh-Hadamard `INT4`** cache: we prove analytically that `m = d` 1-bit Quantized Johnson-Lindenstrauss (`QJL`) sketches *amplify* inner-product variance by $\approx 1.567\times$ (Theorem 3) and show that block-diagonal $H_{64}$ suppresses quantization outliers while commuting exactly with query-time partial-mRoPE distance clamping. All algebraic claims and negative controls are verified to `< 10^{-12}` in IEEE-754 `float64`, and all empirical capability claims are paired with compute-matched falsification gates.

---

## 1. Introduction: The Pretraining Wall vs. Structural Bottlenecks in Sparse Hybrids

Modern open-weight foundation models increasingly converge on two aggressive efficiency compromises: **extreme MoE sparsity** (`Top-8` out of `256` experts, activating `3.125%` of routed FFN weights) and **linear-to-full attention hybridization** (`3:1` Gated DeltaNet to Gated Full Attention in `Qwen3.6-35B-A3B`). Together, these choices make `Qwen3.6-35B-A3B` remarkably fast to serve, yet introduce three fundamental capability ceilings that manifest sharply on multi-step mathematical deduction, fluid rule induction (`ARC-AGI-2`), repository-scale software engineering (`SWE-bench Pro`, `LongBench v2`), and hardware kernel synthesis (`KernelBench`):

1. **Sequential Circuit-Depth & Active-Parameter Bottleneck (`40` Layers, `~2.95B` Active):** By standard circuit complexity bounds for bounded-precision Transformers [Merrill & Sabharwal, 2023; Saunshi et al., 2025], a single forward pass of $L = 40$ layers (with only $L_{\text{attn}} = 10$ global token-mixing layers) restricts latent compositional depth per emitted token. Although external Chain-of-Thought (CoT) converts serial token generation into depth, every CoT step projects a `2048`-dimensional continuous hidden state through a discrete bottleneck of `1` vocabulary token ($\sim 18$ bits) before restarting from Layer 0. Meanwhile, **`31.2B` routed expert parameters (`96.9%` of the FFN bank) go completely unconsulted** on any given token.
2. **Associative Memory Washout in Scalar-Gated Linear Attention (`513 -> 262K` Gap):** Each of Qwen3.6's 30 Gated DeltaNet layers [Yang et al., 2025] maintains a fixed $128 \times 128$ matrix per head updated by a rank-1 Householder erasure $(I - \beta_t k_t k_t^\top) e^{g_t} S_{t-1} + \beta_t k_t v_t^\top$ with scalar $\beta_t \in (0, 1)$. Because every routine syntactic token executes a non-zero rank-1 projection across the same shared $128$-dimensional key space, early variable bindings and schema declarations decay exponentially once cumulative write mass $\sum \beta_\tau$ exceeds $d_k = 128$—long before the `262,144`-token window of the 10 full-attention layers.
3. **The `1M–10M` Context Bandwidth & Entropy Cliff:** Extending the 10 full-attention layers (`2` KV heads, $d_{\text{head}} = 256$) to $10^6$ tokens fits in `10.00 GiB` of `FP8` GPU HBM, but faces positional extrapolation distortion beyond `262K` and softmax log-partition dilution ($\ln 10^6 = 13.82$). Scaling further to $10^7$ tokens (`190.73 GiB` in `BF16`) exceeds single-accelerator `96 GiB` HBM altogether.

For an independent research effort, pretraining a new 88-layer, 6B-active architecture from scratch is infeasible. Our central thesis is **Post-Training Backbone Evolution by Pretrained Weight Reuse**: we treat Qwen3.6's `32.338B` MoE FFN banks and token embeddings as an **immutable pretrained functional substrate**, and surgically evolve only the `~0.90%` (`312M`-parameter) control, routing, re-entry alignment, recurrent memory, and sparse-attention indexing topology around them.

---

## 2. Exact Anatomy of `Qwen3.6-35B-A3B`

An accurate post-training modification requires an exact parameter ledger of the base checkpoint (`config.json` verification, closing `v2` Erratum `E2`):

- **Backbone Topology:** $N = 40$ decoder layers arranged in 10 periodic macro-blocks $B_0, \dots, B_9$. Each macro-block $B_m$ comprises three `GatedDeltaNet` layers ($l \in \{4m, 4m+1, 4m+2\}$) followed by one `GatedAttention` layer ($l = 4m+3$), all with residual width $d_{\text{model}} = 2048$.
- **MoE FFN Substrate (`32.338B` Total, `1.133B` Single-Pass Active):** Every layer contains $E = 256$ routed SwiGLU experts ($d_{\text{ff}} = 512$, `3 × 2048 × 512 = 3,145,728` params/expert $\implies$ **`32.212B`** total, **`1.007B`** active at `Top-8`) plus **1 shared SwiGLU expert** also of intermediate dimension $d_{\text{sff}} = 512$ (**`0.126B`** total across 40 layers, correcting an earlier literature miscount of `0.629B`), gated by a learned router $W_{\text{gate},l} \in \mathbb{R}^{256 \times 2048}$ and scalar sigmoid shared-expert gate (`0.021B` total).
- **Sequence Mixing (`1.285B` Total, `100%` Single-Pass Active):**
  - *30 GDN Layers (`1.012B`):* $H_q = H_k = 16$ heads ($d_k = 128$), $H_v = 32$ value heads ($d_v = 128$), depthwise causal `conv1d` ($k = 4$ on $8192$ channels), SiLU output gate $z$, and `RMSNormGated` before `out_proj` ($4096 \to 2048$).
  - *10 Gated Full-Attention Layers (`0.273B`):* GQA with $H_q = 16$ query heads, $H_{kv} = 2$ KV heads, $d_{\text{head}} = 256$ partitioned into **`64` mRoPE channels** (`partial_rotary_factor = 0.25`, `rope_theta = 10,000,000`, native `262,144` ceiling) and **`192` NoPE content channels**, paired with head-specific sigmoid output gating (`q_proj`: $2048 \to 8192$) [Qiu et al., 2025].
- **Vocabulary (`1.017B` Total):** Untied `embed_tokens` and `lm_head` ($248{,}320 \times 2048$ each; **`0.509B`** active GEMV in `lm_head`). Together, the 40-layer text decoder totals **`34.661B`** parameters (`2.948B` FLOP-active per token), with the remaining checkpoint weights residing in the 27-layer ViT vision tower (`visual`) and 1-layer `mtp` module.

---

## 3. Subspace-Aligned Recursive Depth & Multi-Expert Expansion (`35B-A3B` $\to$ `35B-A6B`)

### 3.1 Why Naive Weight-Tied Looping Fails on Pre-LN Sparse MoEs

Following the Encode–Think–Decode (`ETD`) decomposition [bae et al. / ETD, 2025; Csordás et al., 2024], we partition the 10 macro-blocks into a single-pass **Prelude** ($B_0, B_1$, layers `0..7`), a recurrent **Core** ($B_2 \dots B_7$, layers `8..31`, executed $R$ times), and a single-pass **Coda** ($B_8, B_9$, layers `32..39`), yielding $8 + 24R + 8$ virtual layers (`64L` at $R=2$; `88L` at $R=3$).

Feeding Layer 31's output $h_{31}^{(1)}$ directly back into Layer 8 fails due to **orthogonal basis drift**, not merely norm growth:

**Proposition 1 (Scale Invariance vs. Subspace Rotation under Pre-LN Routing).** *In a Pre-RMSNorm MoE layer with frozen router $W_{\text{gate},8} \in \mathbb{R}^{256 \times d}$, routing logits $\boldsymbol{\ell}(h) = W_{\text{gate},8}\operatorname{RMSNorm}(h)$ are strictly invariant to positive scalar rescaling $\boldsymbol{\ell}(c \cdot h) \equiv \boldsymbol{\ell}(h)$ ($\forall c > 0$), yet under an orthogonal basis drift $h_{31} = c\, Q_\perp (h_7 + \Delta_{\text{sem}})$ ($Q_\perp \in O(d), Q_\perp \neq I$), expected Top-8 routing agreement collapses toward uniform random chance ($8/256 = 3.125\%$), and no diagonal channel gate $\boldsymbol{\alpha} \odot \operatorname{RMSNorm}(h_{31})$ can invert $Q_\perp$.*

*Proof & Diagnosis of the `v1/v2` Anchor-Diversity Dilemma.* Because $\operatorname{RMSNorm}(c h) = \operatorname{RMSNorm}(h)$, `v1`'s scalar norm shock explanation was incomplete: the true failure mode is that 24 residual blocks rotate semantic features into a late-layer basis $Q_\perp$. Conversely, `v2` attempted to preserve step-0 stability via a single entry mixture $\tilde{h}_{\text{in}}^{(2)} = 0.99\, h_7 + 0.01\, \boldsymbol{\alpha}_1 \odot \operatorname{RMSNorm}(h_{31}^{(1)})$ with a frozen shared router. By Lipschitz continuity of linear routing logits, setting $w_{\text{anc}} = 0.99$ bounds $\|\boldsymbol{\ell}^{(2)} - \boldsymbol{\ell}^{(1)}\|_\infty \le 0.02 \|W_{\text{gate}}\|_2 \sqrt{d}$, locking Pass-2 Top-8 expert overlap above `94%–99%` (`test_v3_05`) and starving Pass-2 gradients—directly defeating cross-pass expert expansion! $\square$

### 3.2 Resolution: Decoupling Entry Subspace Alignment from Exit Identity Gating

`Loop-HSSD v3` resolves Proposition 1 cleanly by separating **manifold alignment at core entry** from **bit-exact identity gating at core exit**:

1. **Entry Subspace Bridge (`ReEntryBridge`):** At the entrance to Layer 8 on pass $r \ge 2$, a rank-$r_b$ ($r_b = 256$) adapter maps normalized Layer-31 states back into Layer 8's coordinate frame inside a Birkhoff-simplex convex anchor blend initialized at $(w_{\text{anc}}, w_{\text{rms}}, w_{\text{raw}}) = (0.85, 0.10, 0.05)$:
   $$\tilde{h}_{\text{in}}^{(r)} = 0.85\, h_7 + 0.10\, h_{\text{rms}}^{(r-1)} + 0.05\, h_{\text{core}}^{(r-1)} + \frac{\operatorname{RMS}(h_7)}{\sqrt{d}} W_{\text{re,up}}^{(r)}\,\operatorname{SiLU}\!\left(W_{\text{re,down}}^{(r)}\operatorname{RMSNorm}\!\left(h_{\text{core}}^{(r-1)}\right)\right) + \mathbf{e}_{\text{step}}^{(r)}$$
2. **Exit Identity Gate (`PassExitGate`):** At the exit of Layer 31, Pass $r$ merges via a zero-initialized vector gate $\boldsymbol{\alpha}_{\text{exit}}^{(r)} = \mathbf{0} \in \mathbb{R}^{2048}$:
   $$h_{\text{core}}^{(r)} = h_{\text{core}}^{(r-1)} + \boldsymbol{\alpha}_{\text{exit}}^{(r)} \odot m_t^{(r)} \odot \left(\operatorname{Core}^{(r)}\!\left(\tilde{h}_{\text{in}}^{(r)}\right) - h_{\text{core}}^{(r-1)}\right)$$
   At step 0, $\boldsymbol{\alpha}_{\text{exit}}^{(r)} = \mathbf{0}$ guarantees exact IEEE-754 bit-identity ($h_{\text{core}}^{(R)} \equiv h_{\text{core}}^{(1)}$), while inside `Core`, $\tilde{h}_{\text{in}}^{(r)}$ already carries a live `15%` deliberation residual so $\operatorname{Core}^{(r)}(\tilde{h}_{\text{in}}^{(r)}) - h_{\text{core}}^{(r-1)} \neq \mathbf{0}$ and $\nabla_{\boldsymbol{\alpha}_{\text{exit}}^{(r)}} \mathcal{L} \neq \mathbf{0}$ from step 1.
3. **Task-Driven `PassRouterAdapters`:** Adding a zero-initialized rank-32 router adapter plus learned scalar repeat-suppression lever $-\gamma_{r,l} M_{t,l}^{(<r)}$ on Pass $r \ge 2$ decouples expert selection from the `0.85` input anchor (`test_v3_05`), expanding per-token active compute to **`4.46B` (`64L`, $R=2$)** and **`5.97B` (`88L`, $R=3$)** while recruiting `13–21` unique frozen experts per core layer per token.

### 3.3 Compute-Matched Scaling Against Chain-of-Thought

Because $R=2$ (`64L`) expends `~1.53×` FLOPs/token and $R=3$ (`88L`) expends `~2.05×` FLOPs/token, unnormalized accuracy gains over $R=1$ conflate architectural value with raw test-time FLOPs. Unlike external CoT—which serializes intermediate state through discrete vocabulary sampling—looped recurrence iterates **2048-dimensional continuous superpositions** across 24 layers (`6` global-attention + `18` recurrent memory layers per pass) *before* committing to a token. We treat this advantage as a **falsifiable empirical hypothesis** (`Gate 0b`), requiring `R=2` (`64L`) to outperform stock `40L` granted **`1.6×` thinking tokens** (`2.1×` for `R=3`) on non-saturated fluid and mathematical benchmarks (`ARC-AGI-2`, `AIME 2026`, `LiveCodeBench`, and multi-hop variable tracking).

---

## 4. Horizon-Split Surprise-Gated Delta Memory (`HSSD v3`)

### 4.1 Theorem 1: Isomorphism of Complex-Phase Over-Relaxed Householder and Real `chunk_kda`

Standard Gated DeltaNet restricts transition eigenvalues to the real interval $(0, 1)$, preventing $O(1)$-layer modulo counting, dihedral state tracking, and oscillatory memory addressing [Grazzi et al., 2025; Lahoti et al., 2026]. Introducing explicit complex states $\mathbb{C}^{d_k \times d_v}$ naively doubles state HBM and requires custom complex-valued Triton kernels. We prove that block-2D phase rotation reduces **identically** to a pure real recurrence:

**Theorem 1 (Real KDA Reduction of Block-2D Phase-Rotated Householder Memory).** *Let $d_k = 2m$, and for each step $t \ge 1$ let $R(\boldsymbol{\theta}_t) = \operatorname{diag}_{c=1}^m \begin{pmatrix} \cos\theta_{t,c} & -\sin\theta_{t,c} \\ \sin\theta_{t,c} & \cos\theta_{t,c} \end{pmatrix} \in SO(d_k)$ commute with paired decay $D(\boldsymbol{\alpha}_t) = \operatorname{diag}_{c=1}^m(\alpha_{t,c} I_2)$ ($\alpha_{t,c} \in (0, 1]$). Consider the phase-rotated over-relaxed Householder recurrence ($S_0 = \mathbf{0} \in \mathbb{R}^{d_k \times d_v}$, $\mathbf{k}_t \in \mathbb{R}^{d_k}$ with $\|\mathbf{k}_t\|_2^2 = b_t \in (0, 2)$):*
$$S_t = \left(I - \mathbf{k}_t \mathbf{k}_t^\top\right) R(\boldsymbol{\theta}_t) D(\boldsymbol{\alpha}_t) S_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top, \qquad \mathbf{o}_t = S_t^\top \mathbf{q}_t$$
*Define the cumulative phase $\boldsymbol{\Theta}_t = \sum_{\tau=1}^t \boldsymbol{\theta}_\tau \pmod{2\pi}$, counter-rotated keys and queries $\hat{\mathbf{k}}_t = R(-\boldsymbol{\Theta}_t)\mathbf{k}_t$, $\hat{\mathbf{q}}_t = R(-\boldsymbol{\Theta}_t)\mathbf{q}_t$, and transformed state $\bar{S}_t = R(-\boldsymbol{\Theta}_t) S_t \in \mathbb{R}^{d_k \times d_v}$. Then for every $t \ge 1$:*
$$\bar{S}_t = \left(I - \hat{\mathbf{k}}_t \hat{\mathbf{k}}_t^\top\right) D(\boldsymbol{\alpha}_t) \bar{S}_{t-1} + \hat{\mathbf{k}}_t \mathbf{v}_t^\top, \qquad \mathbf{o}_t \equiv \bar{S}_t^\top \hat{\mathbf{q}}_t$$

*Proof.* Multiply the recurrence for $S_t$ on the left by $R(-\boldsymbol{\Theta}_t)$. Since $R(\cdot)$ is orthogonal ($R(-\boldsymbol{\Theta}_t)^\top = R(\boldsymbol{\Theta}_t)$) and block-2D rotations commute with scalar-pair diagonal matrices $D(\boldsymbol{\alpha}_t)$ as well as with planar rotations ($R(-\boldsymbol{\Theta}_t) R(\boldsymbol{\theta}_t) = R(-\boldsymbol{\Theta}_{t-1})$):
$$\begin{aligned}
\bar{S}_t &= R(-\boldsymbol{\Theta}_t)\left(I - \mathbf{k}_t \mathbf{k}_t^\top\right) R(\boldsymbol{\Theta}_t) R(-\boldsymbol{\Theta}_t) R(\boldsymbol{\theta}_t) D(\boldsymbol{\alpha}_t) S_{t-1} + \left(R(-\boldsymbol{\Theta}_t)\mathbf{k}_t\right)\mathbf{v}_t^\top \\
&= \left(I - \hat{\mathbf{k}}_t \hat{\mathbf{k}}_t^\top\right) D(\boldsymbol{\alpha}_t) R(-\boldsymbol{\Theta}_{t-1}) S_{t-1} + \hat{\mathbf{k}}_t \mathbf{v}_t^\top = \left(I - \hat{\mathbf{k}}_t \hat{\mathbf{k}}_t^\top\right) D(\boldsymbol{\alpha}_t) \bar{S}_{t-1} + \hat{\mathbf{k}}_t \mathbf{v}_t^\top.
\end{aligned}$$
For the readout, $\bar{S}_t^\top \hat{\mathbf{q}}_t = S_t^\top R(-\boldsymbol{\Theta}_t)^\top R(-\boldsymbol{\Theta}_t)\mathbf{q}_t = S_t^\top \mathbf{q}_t = \mathbf{o}_t$. Moreover, $\|\hat{\mathbf{k}}_t\|_2^2 = \|\mathbf{k}_t\|_2^2 = b_t \in (0, 2)$, so along $\hat{\mathbf{k}}_t$ the transition matrix has real eigenvalue $\lambda_{\parallel} = 1 - b_t \in (-1, 1)$—achieving exact $\mathbb{Z}_2$ reflection ($\lambda_{\parallel} = -1$) as $b_t \to 2$—while compiling directly into standard unit-lower-triangular WY real Tensor-Core `chunk_kda` kernels at $1.0\times$ state footprint (`err < 1e-13` in `test_v3_01`). $\square$

### 4.2 Disjoint Horizon Partitioning & Nonsingular Chunk-Start Root-Surprise

To guarantee lossless short-range recall without linear-state interference, `HSSD v3` splits $[1, t]$ at chunk boundaries $C = 64$. Let $n = \lfloor (t-1)/C \rfloor$ and $k_{\text{far}}(n) = \max(0, n - 8)$.

**Proposition 2 (Exact Disjointness and Parallel Chunkwise WY Associativity).**
1. *Reading far-horizon memory from $\bar{S}_{\text{boundary}}[k_{\text{far}}(n)]$ ($\mathcal{I}_{\text{far}} = [1, 64 k_{\text{far}}(n)]$) and local attention over $\mathcal{I}_{\text{loc}} = [64 k_{\text{far}}(n) + 1, t]$ yields an exact disjoint partition $\mathcal{I}_{\text{far}} \sqcup \mathcal{I}_{\text{loc}} = [1, t]$ with $|\mathcal{I}_{\text{loc}}| \in [513, 576]$ for all $t > 512$ (eliminating the 64-token double-counting overlap present when indexing $n - 7$).*
2. *Parameterizing root-surprise $g_{\tau,h} = \sigma(\psi(\hat{\mathbf{k}}_{\tau,h}, \mathbf{v}_{\tau,h}, \bar{S}_{nC,h}^{(0)})) \in (0, 1)$ against the **chunk-start state $\bar{S}_{nC,h}^{(0)}$** with effective write factors $\mathbf{k}_{\tau}^{\text{eff}} = g_\tau \sqrt{\mathbf{b}_\tau} \odot \hat{\mathbf{k}}_\tau$, $\mathbf{v}_{\tau}^{\text{eff}} = g_\tau \mathbf{w}_\tau \odot \mathbf{v}_\tau$, and $\boldsymbol{\alpha}_{\tau}^{\text{eff}} = \exp(-g_\tau^2 \boldsymbol{\gamma}_\tau)$ simultaneously guarantees: (i) exact state preservation $\bar{S}_\tau = \bar{S}_{\tau-1}$ at $g_\tau = 0$, (ii) bounded smooth derivatives $\forall g_\tau \in [0, 1]$ (eliminating $\partial \sqrt{s}/\partial s = \frac{1}{2\sqrt{s}} \to \infty$), and (iii) exact algebraic equivalence between parallel chunkwise WY prefill and token-sequential decoding (`err < 1e-12` in `test_v3_02, test_v3_03`).*

### 4.3 Theorem 2: Cumulative Write-Mass Retention Law & Failure of Uniform Load Balancing

Why do single-matrix GDN and uniformly load-balanced multi-page memories wash out long-range facts even when decay $\alpha = 1$? We derive the exact expectation and variance of retrieval after $H$ interfering Householder-Delta updates:

**Theorem 2 (Householder-Delta Cumulative Write-Mass Retention Law).** *Suppose a target association $(\mathbf{k}_*, \mathbf{v}_*) \in \mathbb{S}^{d_k-1} \times \mathbb{S}^{d_v-1}$ is stored at fixed point $S_0 = \mathbf{k}_* \mathbf{v}_*^\top$, followed by $H$ independent isotropic filler updates $S_t = (I - \eta_t \mathbf{k}_t \mathbf{k}_t^\top) S_{t-1} + \eta_t \mathbf{k}_t \mathbf{v}_t^\top$ with $\mathbf{k}_t \sim \operatorname{Unif}(\mathbb{S}^{d_k-1})$, $\mathbf{v}_t \sim \operatorname{Unif}(\mathbb{S}^{d_v-1})$, and effective write rate $\eta_t = s_t \beta_t \in [0, 1]$. Define the **cumulative write mass** $\mathcal{B}(H) \triangleq \sum_{t=1}^H \eta_t$. Then:*
1. **Exponential Signal Retention:** *Since $\mathbb{E}[\mathbf{k}_t \mathbf{k}_t^\top] = \frac{1}{d_k} I_{d_k}$, the expected retrieved projection along $\mathbf{v}_*$ satisfies:*
   $$\mathbb{E}\!\left[\left\langle S_H^\top \mathbf{k}_*,\, \mathbf{v}_* \right\rangle\right] = \prod_{t=1}^H \left(1 - \frac{\eta_t}{d_k}\right) \le \exp\!\left(-\frac{\mathcal{B}(H)}{d_k}\right)$$
2. **Stationary Interference Variance:** *For constant active write rate $\eta \ll d_k$, background Frobenius energy converges to $\mathbb{E}\|S_\infty\|_F^2 = \frac{\eta d_k}{2 - \eta} \approx \frac{\eta d_k}{2}$, inducing isotropic readout interference $\mathbb{E}\|\mathbf{e}_\perp\|_2^2 \approx \frac{\eta}{2}$ along $\mathbf{k}_*$.*
3. **Failure of Uniform Page Balancing:** *Under a uniform $M$-page load-balancing loss ($\bar{u}_m \approx 1/M$), a `65,536`-token prefix with $M=8$ and active $\beta = 0.25$ inflicts $\mathcal{B}_m \approx 2{,}048 = 16\, d_k$ on every page, attenuating early declarations by $e^{-16} \approx 1.1 \times 10^{-7}$. Conversely, enforcing a tiered write-mass cap $\mathcal{B}_m(H_m) \le 0.4\, d_k$ ($\mu_m^* \le \frac{0.4\, d_k}{\bar{\beta}\, H_m} \approx 0.0025$ for `65K` vault pages) guarantees $\mathbb{E}[\langle S_{H_m}^\top \mathbf{k}_*, \mathbf{v}_* \rangle] \ge e^{-0.4} \approx 0.670$ (`>67%` retained signal amplitude, empirically verified in `test_v3_04`).* $\square$

---

## 5. Commutative Block-Hadamard Quantization & Sparse Attention (`1M–10M`)

### 5.1 Theorem 3: Why `1-bit QJL` at $m = d$ Hurts Group-64 `INT4` KV Accuracy

Recent KV quantization methods (e.g., QJL [Zandieh et al., 2024], PolarQuant / TurboQuant) apply 1-bit sign sketches to quantization residuals in ultra-low-bit (`1–2` bit) or ungrouped settings. In `v1`, an $m = d = 256$ 1-bit QJL sketch was appended to Hadamard Group-64 `INT4` keys (`+32 B/vec` overhead). We prove that once $H_{64}$ Group-64 `INT4` has already driven residual energy small and approximately isotropic, an $m = d$ 1-bit QJL estimator **strictly increases MSE**:

**Theorem 3 (`1-bit QJL` Variance Amplification Bound).** *Let $\mathbf{r}_k = \tilde{\mathbf{k}} - \hat{\mathbf{k}} \in \mathbb{R}^d$ be the quantization residual after orthonormal Hadamard rotation and Group-64 `INT4` quantization, and let $\mathbf{q} \in \mathbb{R}^d$ be an independent query direction ($\mathbb{E}[(\mathbf{q}^\top \mathbf{r}_k)^2] = \frac{\|\mathbf{q}\|_2^2 \|\mathbf{r}_k\|_2^2}{d}$). For $S \in \mathbb{R}^{m \times d}$ with i.i.d. $\mathcal{N}(0, 1)$ entries, the unbiased 1-bit QJL estimator $\hat{z}_{\text{QJL}} = \frac{\sqrt{\pi/2}}{m} \|\mathbf{r}_k\|_2 \sum_{i=1}^m (\mathbf{s}_i^\top \mathbf{q})\operatorname{sign}(\mathbf{s}_i^\top \mathbf{r}_k)$ satisfies $\mathbb{E}_S[\hat{z}_{\text{QJL}} \mid \mathbf{q}, \mathbf{r}_k] = \mathbf{q}^\top \mathbf{r}_k$ and conditional variance [Zandieh et al., 2024]:*
$$\operatorname{Var}_S\!\left[\hat{z}_{\text{QJL}} \,\middle|\, \mathbf{q}, \mathbf{r}_k\right] = \frac{1}{m}\left(\frac{\pi}{2}\|\mathbf{q}\|_2^2 \|\mathbf{r}_k\|_2^2 - \left(\mathbf{q}^\top \mathbf{r}_k\right)^2\right)$$
*Taking expectation over isotropic $(\mathbf{q}, \mathbf{r}_k)$ gives the exact MSE ratio relative to omitting QJL ($\hat{z} = 0$, $\operatorname{MSE}_{\text{INT4}} = \frac{1}{d}\mathbb{E}[\|\mathbf{q}\|_2^2 \|\mathbf{r}_k\|_2^2]$):*
$$\frac{\operatorname{MSE}_{\text{INT4+QJL}}(m)}{\operatorname{MSE}_{\text{INT4}}} = \frac{d}{m}\left(\frac{\pi}{2} - \frac{1}{d}\right)$$
*At $m = d = 256$, $\frac{\operatorname{MSE}_{\text{INT4+QJL}}}{\operatorname{MSE}_{\text{INT4}}} = \frac{\pi}{2} - \frac{1}{256} \approx \mathbf{1.5669} > 1$: the 1-bit sketch inflates inner-product error variance by **`+56.7%`** while consuming `+11.1%` extra RAM! Break-even requires sketch dimension $m > d\left(\frac{\pi}{2} - \frac{1}{d}\right) = 401.1 \implies m^* = 402$ (verified empirically in `test_v3_06`: $m=256$ degrades MSE by `>1.45×`, whereas $m=512$ reduces MSE).* $\square$

### 5.2 Block-Diagonal $H_{64}$ Commutativity with Partial-mRoPE Clamping

In `Qwen3.6-35B-A3B`, each $256$-d head concatenates $\mathbf{k} = [\mathbf{k}_{\text{rope}} \in \mathbb{R}^{64} \,\|\, \mathbf{k}_{\text{nope}} \in \mathbb{R}^{192}]$. Applying a full $256 \times 256$ Walsh-Hadamard matrix $H_{256}$ mixes $\mathbf{k}_{\text{rope}}$ and $\mathbf{k}_{\text{nope}}$ across all 256 coordinates; if query-time ReRoPE / Dual Chunk Attention modifies the relative phase angle applied to $\mathbf{q}_{\text{rope}}$, an offline-cached key rotated by raw token position $j$ before $H_{256}$ can no longer be decoupled cleanly from $\mathbf{k}_{\text{nope}}$ (`Review Finding C2`).

Replacing $H_{256}$ with the **Block-Diagonal Group-64 Hadamard operator** $\mathcal{H}_{\text{blk64}} = \operatorname{diag}(H_{64}, H_{64}, H_{64}, H_{64})$ aligns the orthogonal transform *1-to-1* with the four `64`-scalar `INT4` quantization groups:
$$\left\langle \mathcal{H}_{\text{blk64}} \begin{bmatrix} R(\tilde{\Delta})\mathbf{q}_{\text{rope}} \\ \mathbf{q}_{\text{nope}} \end{bmatrix},\, \mathcal{H}_{\text{blk64}} \begin{bmatrix} \mathbf{k}_{\text{rope}} \\ \mathbf{k}_{\text{nope}} \end{bmatrix} \right\rangle = \left\langle H_{64} R(\tilde{\Delta})\mathbf{q}_{\text{rope}},\, H_{64}\mathbf{k}_{\text{rope}} \right\rangle + \sum_{g=1}^3 \left\langle H_{64}\mathbf{q}_{\text{nope}}^{(g)},\, H_{64}\mathbf{k}_{\text{nope}}^{(g)} \right\rangle$$
Thus $\mathbf{k}_{\text{nope}}$ (`75%` of head dimensions, which also drives `D-LSA` Ward clustering) remains **100% invariant to any rotary modification**, while within every 64-scalar quantization group $H_{64}$ disperses channel outliers by $1/\sqrt{64} = 1/8$, reducing `INT4` reconstruction error by `>38%` over unrotated quantization at exactly **`0.5625 B/scalar` (`53.64 GiB` across 10 layers at `10M` tokens)**.

---

## 6. Positioning Against Contemporary 2025–2026 Frontier Architectures

| Architecture / Method | Core Mechanism | Pretrain from Scratch Required? | Relationship & Synergy with `Loop-HSSD v3` |
| :--- | :--- | :---: | :--- |
| **NVIDIA `GDN-2`** [Hatamizadeh et al., 2026] | Decouples channel-wise erase $\mathbf{b}_t$, write $\mathbf{w}_t$, and log-decay $\mathbf{g}_t$ in Gated DeltaNet | **No** *(exact step-0 generalization!)* | Adopted directly as **Track A (`src/qloop/model/gdn2.py`)**—step-0 bit-identical baseline & teacher for Track B. |
| **`Mamba-3`** [Lahoti et al., ICLR 2026] | Trapezoidal discretization, data-dependent complex RoPE-equivalent SSM (`SISO/MIMO`) | Yes *(replaces GDN projections)* | Theorem 1 transfers Mamba-3's data-dependent complex state rotation directly onto Qwen3.6's Householder-Delta geometry without discarding GDN weights. |
| **Log-Linear GDN** [Guo et al., ICML 2026] & **Sparse Delta Memory** [Cabannes et al., 2026] | $O(\log T)$ Fenwick-tree state banks / sparse addressing into large linear memory | Yes / Partial | Motivate multi-horizon state isolation; `HSSD v3` instead uses $M=8$ dense pages governed by explicit write-mass budgets $\mathcal{B}_m \le 0.4 d_k$. |
| **DeepSeek-V4 `HCA / CSA`** *(2026 reports)* | Hierarchical Coarse-to-Fine / Compressed Sparse Attention over MLA latents | Yes *(native MLA backbone)* | `D-LSA` [Kubicek et al., 2026] Ward clustering + `4:1` pooling + per-pass $W_{q\text{-idx}}^{(r)}$ achieves coarse-to-fine retrieval natively on Qwen3.6's GQA (`2` KV heads). |
| **Looped MoEs (`SMELT`, `Foil`, `Loop the Loopies`, `Looped Models Done Right`)** [2025–2026] | Mid-section looping ($R=2$ sweet spot), untied per-pass attention, input-anchor injection | Mid-training adaptation (`50M–1B` tok) | Synthesized & extended to `256`-expert hybrid MoEs via `ReEntryBridge` ($r=256$) + `PassExitGate` + `PassRouterAdapters` (`35B-A3B -> 35B-A6B`). |

---

## 7. Summary of Proven Invariants vs. Empirical Falsification Hypotheses

To maintain strict scientific rigor, we separate claims already proven mathematically and verified in double-precision code ([`test_loop_hssd_v3_math.py`](../tests/test_loop_hssd_v3_math.py)) from empirical hypotheses subjected to GPU Kill-Gates ([`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)):

### 7.1 Mathematically Proven & Numerically Verified (`< 10^{-12}` in `float64`)
1. **Step-0 Bit-Exact Equivalence (`torch.equal`):** Both `Loop + GDN-2` (`Track A`) and `PassRouterAdapters` reproduce stock `Qwen3.6-35B-A3B` bit-for-bit at step 0 while maintaining non-zero step-1 gradients into $\boldsymbol{\alpha}_{\text{exit}}^{(r)}$.
2. **Complex-to-Real KDA Isomorphism (Theorem 1):** Block-2D state rotation equals counter-rotated real symmetric DPLR `chunk_kda` (`max_err < 1e-13`).
3. **Disjoint Horizon & Parallel Chunk-Start Surprise WY Equivalence (Proposition 2):** Zero double-counted tokens across $[1, t]$, finite gradients at $g = 0$, and exact equivalence between parallel chunkwise WY prefill and sequential decoding (`max_err < 1e-12`).
4. **Write-Mass Law & Tiered Page Protection (Theorem 2):** Analytical and Monte-Carlo agreement on $\exp(-\mathcal{B}/d_k)$ retention and `>65` percentage-point cosine advantage over uniform load balancing.
5. **QJL Variance Bound & Block-64 Hadamard-ReRoPE Commutativity (Theorem 3):** Exact `1.567×` variance penalty at $m=d=256$ and `< 1e-12` inner-product invariance under rotary clamping.
6. **Multi-Layer 2-Pass Chunk-Parallel vs. Token-Decode Equivalence:** Chunk-parallel Pass-1 $\to$ Pass-2 matches token-by-token autoregressive decoding across stacked recurrent-MoE layers (`max_err < 1e-13`) with SHA-256 verified zero mutation of frozen MoE experts.

### 7.2 Empirical Hypotheses Governed by GPU Kill-Gates (`Gates 0a–4`)
1. **Hypothesis H-Loop (`Gate 0b`):** At **matched total test-time FLOPs**, `LoopHSSD` ($R=2$ `64L` vs `1.6×`-token CoT stock `40L`; $R=3$ `88L` vs `2.1×`-token CoT stock `40L`) yields superior accuracy on latent rule induction (`ARC-AGI-2`), competition mathematics (`AIME 2026`), and multi-hop state tracking.
2. **Hypothesis H-Memory (`Gate 1`):** `Track B (HSSD v3)` distilled from `Track A (GDN-2)` improves `32K–64K` associative recall (`RULER` / `MQAR`) by $\ge +8\%$ absolute with $\le 1\%$ general perplexity overhead.
3. **Hypothesis H-Context (`Gate 2a` & `2b`):** Long-Prefix (`no_grad`) + Short-Suffix (`grad`) read-side calibration combined with per-pass `D-LSA` sustains $\ge 88\%$ retrieval at `1M` tokens (`LongBench v2` gains) and $\ge 80\%$ streaming retrieval at `10M` tokens under `53.64 GiB` Host-DDR5 Block-64 Hadamard `INT4` offload.

---

## 8. References

1. **Qwen Team.** *Qwen3.6-35B-A3B Architecture, Configuration, & Technical Report.* Alibaba Cloud / Hugging Face (`Qwen/Qwen3.6-35B-A3B`), April 2026.
2. **Kubicek, A., et al.** *Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention (`D-LSA`).* 2026 (`arXiv:2610.42147`). *(Evaluated on GLM-5.3-Flash & Qwen-3.8-27B; adapted here to Qwen3.6's 192-d NoPE GQA subspace with per-pass query re-indexing.)*
3. **Wang, S., Zhang, G., Luo, K., Wu, Y., Liu, J., et al.** *SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers.* 2026 (`arXiv:2609.01343`).
4. **Wang, S., et al.** *How to Loop MoE: Flatten the Experts, Untie the Attention (`Foil`).* 2026 (`arXiv:2609.35751`).
5. **Loop the Loopies Authors.** *Loop the Loopies: Scaling Laws and $R=2$ Depth Optimality for Looped Mixture-of-Experts Models.* 2026 (`arXiv:2607.16051`).
6. **Towards Looped Models Done Right Authors.** *Towards Looped Models Done Right, Part I: Input Anchor Injection and Recurrent State Design.* 2026 (`arXiv:2610.23036`).
7. **Closing the Loop Authors.** *Closing the Loop: Practical Training Recipes for Retrofitting Looping into Pretrained Language Models.* 2026 (`arXiv:2610.00673`).
8. **Bae, S., Kim, Y., et al.** *Mixture-of-Recursions (`MoR`): Learning Dynamic Recursive Depths for Adaptive Token-Level Computation.* NeurIPS 2025 (`arXiv:2507.10524`).
9. **Encode-Think-Decode (`ETD`) Authors.** *Scaling Latent Reasoning via Looped Language Models (`Encode-Think-Decode`).* 2025 (`arXiv:2510.07358`).
10. **Xie, Z., Hu, S., Wei, Y., Cao, H., et al. (DeepSeek-AI).** *mHC: Manifold-Constrained Hyper-Connections.* 2026 (`arXiv:2512.24880`).
11. **Yang, S., Kautz, J., & Hatamizadeh, A.** *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025 (`arXiv:2412.06464`).
12. **Hatamizadeh, A., Choi, Y., & Kautz, J.** *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention.* NeurIPS 2026 (`arXiv:2605.22791`).
13. **Lahoti, P., Li, X., Chen, Y., Wang, Z., Bick, A., Kolter, J. Z., Dao, T., & Gu, A.** *Mamba-3: Improved Sequence Modeling Using State Space Principles.* ICLR 2026 (`arXiv:2603.15569`).
14. **Kimi Team (Moonshot AI).** *Kimi Linear: An Expressive, Efficient Attention Architecture (`KDA`).* Technical Report, 2025 (`arXiv:2510.26692`).
15. **Grazzi, R., Siems, J., Zela, A., et al.** *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR 2025 (`arXiv:2411.12537`); and *DeltaProduct* (`arXiv:2502.10297`).
16. **Guo, H., Yang, S., Goel, K., Xing, E. P., Dao, T., & Kim, Y.** *Log-Linear Attention.* ICML 2026 (`arXiv:2506.04761`).
17. **Cabannes, V., et al. (Meta FAIR).** *Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity.* July 2026 (`arXiv:2607.07386`); see also *Scaling Linear Attention with Sparse State Expansion* (ICLR 2026, `arXiv:2507.16577`). *(Cited as conceptual motivation for multi-slot state expansion; `HSSD v3` uses write-budgeted dense pages rather than sparse RAM addressing.)*
18. **Qiu, Z., Wang, Z., Zheng, B., et al.** *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free.* NeurIPS 2025 Oral (`arXiv:2505.06708`).
19. **Zandieh, A., Daliri, M., & Han, I.** *QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead.* AAAI 2025 (`arXiv:2406.03482`); and **Ashkboos, S., et al.** *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS 2024 (`arXiv:2404.00456`).
20. **Zhu, H., Chen, Z., et al.** *APOLLO: SGD-like Memory, AdamW-level Performance for LLM Training.* MLSys 2025 (`arXiv:2412.05270`); **Wang, Z., et al.** *Expert-Specialized Fine-Tuning (`ESFT`).* EMNLP 2024 (`arXiv:2407.01906`); and **Yang, S., & Zhang, Y.** *Flash-Linear-Attention (`fla`, MIT License).* (`github.com/fla-org/flash-linear-attention`).
