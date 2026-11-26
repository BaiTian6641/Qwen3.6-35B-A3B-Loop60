# 中期外部评审交叉核验台账与双冲刺修复计划（`v1.1`）

> **英文对照版：** [English Version (`docs/MIDWAY_REVIEW_AND_ACTION_PLAN.md`)](MIDWAY_REVIEW_AND_ACTION_PLAN.md)  
> **受审仓库基线：** `0f5d7f7`（两份独立外部评审） · **当前 `main` `HEAD`：** `a94c163`（**冲刺 1 轨道 A–D 已 100% 合入** + 冲刺 2 评测套件）  
> **已完结 `s2_v2_joint_bptt@600` 检查点：** Hub 提交 `17382c2e166d`（`2026-11-26T03:14:49Z`，`600/600` 步，`157,286,400` [`157.29M`] 词元，`156,941,238` 可训练参数，`69.97 GiB` 峰值显存，`U = 18.19` [`1.764×`，每词元唯一调阅参数 `3.716B`]；接替中期受审快照 `s2_v2@450` `a5b20b20`）  
> **目标硬件：** `1× NVIDIA RTX PRO 6000 Blackwell Server Edition`（`96 GB GDDR7 ECC` [`94.97 GiB` 可用]，`1,597 GB/s`，`sm_120`）  
> **战略状态：** **冲刺 1（轨道 A–D）已 100% 完成并合入** · **冲刺 2A（`S2.1` `8K–128K`、`S2.3`、`K0`、真机 TF `5.24.1` 审计）已在 Blackwell GPU 完成实测** · **`s4_v2_rl`、`LSA-NL`/`CoMem` 训练扩张与公开完整模型包发布继续保持冻结**，直至**冲刺 2B（`S2.2` `d=1` LoRA 对照）**、`QL60-002` `cache` 参数 `4/4 PASS` 复验及 `512K / 1M` 长上下文收尾通过。

---

## 1. 中期执行总评与冲刺 1 / 2A 进展

两份独立的外部中期评审在仓库提交 `0f5d7f7` 上对本项目与线上 Hugging Face Hub 检查点元数据进行了系统审计：
1. **评审 1（`GPT-6 Astra Pro` 代码与系统审计 @ `0f5d7f7`）：** 指出 `10` 项具体代码、评测、推理服务与编排缺陷（`QL60-001` 至 `QL60-010`）以及参数口径不一致问题（`§6`）。
2. **评审 2（研究方法与代码库审计 @ `0f5d7f7` + 线上 Hub `s2_v2@450` `manifest.json` `a5b20b20`）：** 指出 `16` 项科学性、方法学、遥测与工程卫生问题，分为 `A1–A3`（科学有效性）、`B1–B5`（核心主张与遥测）、`C1–C3`（损失/索引器/路由器正确性）及 `D1–D5`（可维护性与数据溯源）。

我们已在提交 `17216c0` 上逐条对照线上 `s2_v2@450` Hub 遥测（`a5b20b20`）完成独立复核：**评审 1 的 `10/10` 项与评审 2 的 `16/16` 项发现全部确证属实。** 此后，**冲刺 1 全部四条 CPU 工程修复轨道（轨道 A–D）已全部完成并合入 `main`**，`s2_v2_joint_bptt` 顺利跑满 `600/600` 步（`17382c2e166d`），隔离且经 CPU 验证的冲刺 2 评测模块（`engram.py`、`kill_gate_k0.py`、`verify_tf524_serving_gpu.py`、`sprint2_audit.py`、`d1_lora.py` + `s2_2_d1_lora_control.yaml`）均已合入 `main`，并完成了**冲刺 2A Blackwell GPU 实证验证**（`S2.1` 无泄漏留出/OOD + `8K–128K` 阶梯、`S2.3` 因果消融、`K0` Pre-RoPE KV-Blob 预言机及真实 `transformers 5.24.1` 推理服务审计，详见 `§7`）。

### 经受住检验的基础设施与冲刺 2A 实证结论
- **单卡 Blackwell `sm_120` 系统栈：** 在单张 `NVIDIA RTX PRO 6000 Blackwell Server Edition`（`96 GB GDDR7 ECC` [`94.97 GiB` 可用]，`1,597 GB/s`，`sm_120`）上改造 `Qwen/Qwen3.6-35B-A3B` [1]（`30` 层原生 Gated DeltaNet [2] + `10` 层门控全局注意力，`32.212B` 路由 MoE 权重 100% 冻结且经 SHA-256 校验）运行稳定，融合 `GDN-2-1M` [3]、有界 `QED` [4]、软负特征值 [5]、逐轮 `D-LSA` [6]、低秩 `APOLLO` [7]、`mHC` `LoopBridge` [8]、纯 `4.0×` YaRN [9] 与 `fla 0.6.0` DPLR Triton 内核 [10]。
- **已验证工程里程碑与冲刺 2A 实证：** 提交 `574898d` 的 `4×8K` 宏分段将 `T=32K` `GDN2Adapter` 瞬时显存压缩 `3.28×`（`23.051 → 7.032 GiB`，令完结态 `s2_v2@600` 峰值显存保持在 `69.97 GiB`）；`1M` `FP8` `1.0×` `SharedKV` 解析状态仅占 `10.65 GiB`（较 Inkling 2 Mini 的 `32.16 GiB` [11] 缩小 `3.02×`）；公开检查点 [12] 与经审计的 `v1.2` 数据产物 [13] 保持完整可追溯性。在**冲刺 2A**（`§7`）中，无泄漏 `0/4` 重叠留出集（较原版 PPL `-10.71%`，`+469σ`）、纯未见 `wikitext-103-raw-v1` OOD（较原版 PPL `-1.11%`，`+33.6σ`）、`8K–128K` 多针 RULER（`12/12 = 100%` vs 原版 `2/12`）、`128K` 稀疏–稠密 `D-LSA` 极窄差距（`+0.00050`，较 `s3_v2` 收紧 `129×`）、平稳 `GDN-2` 状态范数（`186.2 → 191.4 → 191.0`，`0` NaN），以及真机 `transformers 5.24.1` `QL60-001`（精确 `1.00×` KV）与 `QL60-005`（`13/13`）均**实测通过**。

### 原审计阻塞项闭环情况与冲刺 2A 后剩余开放缺口
- **已在冲刺 1（轨道 A–D 合入 `main`）与冲刺 2A 闭环：** 打包前 SHA-1 文档留出（`QL60-003` / `A1`）、`self.rng` 与仅权重 `lr_factor` 续训同步（`QL60-004`）、按有效词元归一化 BPTT CE 与纯 `loss/lm` / `loss/aux` 分离（`QL60-009` / `C1`）、模型指纹绑定评测缓存（`QL60-010`）、稀疏 `lsa_attention()` 调用方掩码（`QL60-007`）、按轮次分键的未共享 `BucketIndex`（`QL60-008`）、均值归约 `aux_loss()`（`C2`）、稀疏 `lsa/last_recall` 实时采样（`B2`，`4K` 实测 `0.8360`）、确定性路由器识别（`C3`）、洁净 v2 默认阶段序列（`QL60-006`）、严格 `_local_stage_done()` 发布门控（`QL60-005`，真机 `13/13` PASS）、真机 `transformers 5.24.1` `QuantizedHybridCache`（`QL60-001`，真机 PASS）、原版 `Qwen3.6 + YaRN 4×` 正面对比与 `3` 次重复底噪（`A3`、`B4`）、`QED` / 负特征值因果裁撤依据（`B5`）及 `D3` / `D5` 卫生与溯源政策。
- **明确的剩余开放缺口（为何继续维持冻结直至冲刺 2B）：** (1) `QL60-002` 真机 `transformers 5.24.1` `.generate()` 要求 `forward()` 显式声明 `cache: Any = None` 参数（正在并行闭环，待 `4/4 PASS` GPU 复验）；(2) `512K / 1M` 长度阶梯（`B3`）在阶段 4 触达 `5.5h` 安全上限与 `94.43 GiB` 峰值（`96,697 / 97,887 MiB`；OOM 告警时 `93.15 GiB` / `95,381 MiB`）CUDA 分配器碎片化而未完结；(3) **冲刺 2B（`S2.2`）** `~18.41M` 参数量匹配 `d=1` LoRA 对照基线（`A2`）尚未在 GPU 上开训——鉴于 `s2_v2@600` 在未见 OOD 上 `d=1`（`1.69971`）略优于 `d=3`（`1.70122`，`-4.2σ`），该对照更具决定性；(4) 零训练 `K0` 获判 **`YELLOW_B_BLOCK_ATTN_CALIB`**（`6.05×` 压缩 KV 下 `100.0%` EM 保持率，仅因冻结投影动态误差 NLL 差 `+0.6228` nats 未达 `GREEN`）；(5) 端到端生成式推理基准（`AIME`/`GPQA`/`ARC-AGI-2`）尚未执行。

> [!IMPORTANT]
> **强制中期冻结令（冲刺 2A 后更新）：** 冲刺 1（轨道 A–D）已 **100% 完结**，冲刺 2A（`S2.1` `8K–128K`、`S2.3`、`K0`、真机 TF `5.24.1` 审计）已在 Blackwell GPU **完整落盘**（`§7`）。尽管如此，`s4_v2_rl`、`LSA-NL` / `CoMem` 训练扩张与公开完整模型包发布仍**严格冻结**，直至**冲刺 2B（`S2.2` `d=1` LoRA 对照）**、`QL60-002` 真机 `4/4 PASS` 复验及第 `6.3` 节剩余门控全部通过。

---

## 2. 全量交叉核验台账

### 2.1 线上 Hub 遥测实测对比：中期受审快照（`s2_v2@450`，`a5b20b20`） vs. 完结态（`s2_v2@600`，`17382c2e166d`）

表 2.1 全部数据均直接溯源至 Hub 提交 `a5b20b20`（`2026-11-25T17:54:48Z`，中期受审快照，前序 `cec24a8d@400`）与最终完结提交 `17382c2e166d`（`2026-11-26T03:14:49Z`，`600/600` 步）[12]：

| 指标分组 | 中期受审 `s2_v2@450`（`a5b20b20`） | 完结态 `s2_v2@600`（`17382c2e166d`） | 诊断意义 |
| :--- | :--- | :--- | :--- |
| **运行与显存** | `step=450/600`，`117,964,800` 词元，`n_params=156,941,238`，`has_optim=false`，`lr_factor=0.9738` | **`step=600/600`**，**`157,286,400`（`157.29M`）词元**，`156,941,238` 可训练（`32.212B` MoE 冻结），**峰值显存 `69.97 GiB`** | 单张 `RTX PRO 6000 Blackwell`（`96 GB GDDR7 ECC`）稳定完训；`QL60-004` 仅权重 `lr_factor` 重置缺陷已在轨道 A（`70f789e`）彻底修复。 |
| **评测与损失** | 同流 `eval/loss_d1=0.5026 → d3=0.4965`（`+0.0062`）；混入 KL 的 `loss/lm=3.8421` | **冲刺 2A 无泄漏 `4K` 留出（`0/4` 重叠）：** `0.87281 → 0.84861 → 0.83671`（`+0.03609`，`+309σ`；较原版 PPL `-10.71%`，`+469σ`） | 轨道 A（`QL60-003`、`QL60-009`）+ 冲刺 2A 以纯 LM CE 替换原泄漏/混叠日志；OOD 上 `d=1`（`1.69971`）略优于 `d=3`（`1.70122`），令 `S2.2` 成为决定性门控。 |
| **循环与路由** | `bridge_w_anchor=0.831/0.866`，`gate_abs_mean=0.022/0.058`，`U=14.96`（`1.522×`，`3.472B` 唯一） | **`U=18.19` 核心专家**（**`1.764×`** 倍率，**每词元唯一调阅参数 `3.716B`**） | 至 `step 600` 跨轮专家分化进一步提升；`S2.3` `no_pass_router`（`+0.00251`，`+6.8σ`）确证 `PassRouterAdapters` 为 `KEEP_CRITICAL`。 |
| **GDN-2 与 D-LSA** | `adapter_norm=0.741`，`decay_horizon_mult_max=4.545×`，`qed_lambda=0.00071`，`neg_eigval=0.9998`，`lsa/last_recall=0.0` | **`gdn2/adapter_norm=0.845`**，**`decay_horizon_mult_max=7.60×`**，**`qed_lambda=0.00085`**，**`neg_eigval=0.99986`**；冲刺 2A `lsa/last_recall=0.8360`（`4K`） | `S2.3` 证明基础 `GDN-2`（`+0.03887`，`+156.6σ`）是头号驱动项，而 `QED`（`-0.31σ`）与 `neg_eigval`（`-0.97σ`）实证零增益，列为 **`PRUNE_CANDIDATE`**。 |

### 2.2 评审 1 发现核验台账（`GPT-6 Astra Pro`，`QL60-001`..`QL60-010`）

| 编号 | 交叉映射 | 闭环状态 | 已验证修复与实证证据（`main` `a94c163` + 冲刺 2A） | 轨道 / 提交 | 门控结果 |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **`QL60-001`** | — | **已修复且真机 PASS** | [`src/qloop/infer/engine.py`](../src/qloop/infer/engine.py)：`QuantizedHybridCache` 完整实现 `transformers 5.24+` `Cache`/`CacheLayerMixin` 协议（`conv_states`、`recurrent_states`、`reorder_cache`、`crop`、`reset`、`get_mask_sizes`）与 `update()` kwargs。 | 轨道 C（`2b15d4c`、`581db44`） | **真机 GPU PASS**（实测 `transformers 5.24.1`）：`bf16`/`fp8`/`hadamard_int4` 在 `d=1` 与 `d=3` 下均达成精确 `1.00×` KV（`kv_share=False` 为 `2.00×`）。 |
| **`QL60-002`** | `D1` | **已合入；待复验** | [`src/qloop/infer/hf_custom_modeling.py`](../src/qloop/infer/hf_custom_modeling.py)：已合入兼容 `ModelOutput` 的 `CausalLMPackOutput`、同步虚拟 `layer_types`、`logits_to_keep=0` 返回 `(B, T, V)` 与 `qloop_commit` 元数据。 | 轨道 C（`2b15d4c`、`581db44`） | 已通过 `ModelOutput` 与 `d=1` 包内 `logits_to_keep=0/1`，随后在 `past_key_values=qcache` 路径 `.generate()` 抛出 `ValueError: ['cache']`（要求 `forward()` 显式声明 `cache: Any = None`；并行修复中，待 `4/4 PASS` 复验）。 |
| **`QL60-003`** | **`A1`** | **已修复并实证** | [`src/qloop/train/trainer.py`](../src/qloop/train/trainer.py) 与 [`src/qloop/data/mixture.py`](../src/qloop/data/mixture.py)：打包前 SHA-1 文档哈希切分（`split_role="train"` / `"eval"`）。 | 轨道 A（`70f789e`）+ 冲刺 2A | CPU 8 文档复现 `0` 重叠 + 冲刺 2A Blackwell GPU 留出集 **PASS（`0/4` 文档重叠**，`19,093` 有效词元）与未见 `wikitext-103-raw-v1` OOD（`30,030` 有效词元）。 |
| **`QL60-004`** | — | **已修复并合入** | [`src/qloop/train/trainer.py`](../src/qloop/train/trainer.py) 与 [`src/qloop/train/checkpoint.py`](../src/qloop/train/checkpoint.py)：序列化还原 `self.rng`；仅权重恢复记录 `resume_mode="weights_only"` 并同步 `self.opt.step_count = self.step`。 | 轨道 A（`70f789e`） | CPU 单元测试 PASS（`test_sprint1_train_data_resume.py`）：`self.rng` 精确往返与仅权重 `lr_factor` 保持通过。 |
| **`QL60-005`** | — | **已修复且真机 PASS** | [`src/qloop/colab.py`](../src/qloop/colab.py)：`_local_stage_done()` 强制校验显式 `COMPLETED`/`PROMOTED` 标记或 `step >= target_steps`，拒绝裸/中途 `LATEST`。 | 轨道 C（`2b15d4c`） | **真机 GPU PASS（`13/13` 用例）**，载于 `runs/sprint2a/serving_audit.md`，含暂停态 `step000001` 拒绝校验。 |
| **`QL60-006`** | `D2`、`D3` | **已修复并合入** | [`src/qloop/colab.py`](../src/qloop/colab.py) 与 [`src/qloop/colab_supervisor.py`](../src/qloop/colab_supervisor.py)：默认阶段序列切换为洁净 v2（`s0_probe`、`s1_loop_a6b`、`s3_v2_long_1m_lsa`、`s2_v2_joint_bptt`、`s4_v2_rl`）。 | 轨道 C（`2b15d4c`） | CPU 单元测试 PASS（`test_sprint1_hf_serving_ops.py`）；退役 HSSD 仅作 `--stages` 显式回溯项。 |
| **`QL60-007`** | — | **已修复并合入** | [`src/qloop/model/lsa.py`](../src/qloop/model/lsa.py)：在 sink、局部窗与桶选键全路径强制布尔/浮点 2D/3D/4D `attention_mask`，保留 `None` 零开销快路径。 | 轨道 B（`36d73eb`） | CPU 单元测试 PASS（`test_sprint1_lsa_loop.py`）：左/右填充与分块对角掩码误差 `< 1e-6`。 |
| **`QL60-008`** | — | **已修复并合入** | [`src/qloop/model/lsa.py`](../src/qloop/model/lsa.py)：`kv_share=False` 时按 `(phys_layer, cur_pass)` 分键缓存 `BucketIndex`，同时保持 `kv_share=True` 物理复用。 | 轨道 B（`36d73eb`） | CPU 单元测试 PASS（`test_sprint1_lsa_loop.py`）：两种模式 float64 精度一致误差 `< 1e-12`。 |
| **`QL60-009`** | **`C1`** | **已修复并实证** | [`src/qloop/train/bptt.py`](../src/qloop/train/bptt.py)：按 `valid_tok_s / total_valid_toks` 归一化分段反传（安全处理零有效段），分离纯 `loss/lm` 与 `loss/aux`。 | 轨道 A（`70f789e`） | CPU `1 vs 3` 有效词元梯度一致 PASS；冲刺 2A 全部 4 臂均以纯 LM CE 实测。 |
| **`QL60-010`** | — | **已修复并实证** | [`src/qloop/eval/runners.py`](../src/qloop/eval/runners.py)：持久化规范运行指纹（`model_id`、检查点摘要、`active_passes`、`config_hash`、`dataset_rev`）并拒绝跨检查点复用缓存。 | 轨道 A（`70f789e`）+ 冲刺 2A | CPU 跨模型缓存拒绝 PASS；冲刺 2A JSON 为每个臂/数据集持久化完整 `run_identities`。 |

### 2.3 评审 2 发现核验台账（代码与研究方法审计，`A1–A3`、`B1–B5`、`C1–C3`、`D1–D5`）

| 编号 | 严重度 | 闭环状态 | 已验证修复与冲刺 2A 实证证据（`main` `a94c163`） | 轨道 / 阶段 | 门控结果 |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **`A1`**（`QL60-003`） | 致命 | **已闭环** | 打包前 SHA-1 留出（`70f789e`）+ 冲刺 2A（`§7.2`）在 `0/4` 重叠留出集（`19,093` 有效词元）与未见 `wikitext-103-raw-v1` 测试集（`30,030` 有效词元）完成 4 臂重评。 | 轨道 A + 冲刺 2A | **PASS：** `s2_v2@600` 在留出集较原版 PPL `-10.71%`（`+469σ`），在未见 OOD 较原版 PPL `-1.11%`（`+33.6σ`）。 |
| **`A2`** | 致命 | **代码已合入；待 GPU 训练** | `src/qloop/model/d1_lora.py` + `configs/stages/s2_2_d1_lora_control.yaml` 已合入（`63d8b00`，CPU 验证通过），在 `[12, 36)` 上匹配 `~18.41M` `d=1` 参数量。 | **冲刺 2B（`~18` GPU-h）** | **开放决定性门控：** 解冻 `s4_v2_rl` 或 `LSA-NL` 前必跑（尤因 `s2_v2@600` 未见 OOD `d=1` 略优于 `d=3`）。 |
| **`A3`** | 中 | **已闭环** | 冲刺 2A（`§7.2`）在 Blackwell 量化路径执行 `3` 次重复（`mean ± std`）+ `2× d=1` 非确定性底噪检测（`max_diff = 2.36e-4 .. 1.18e-3`）。 | 冲刺 2A | **PASS：** 全部核心相对原版差值（`+20.7σ .. +469.3σ`）及关键消融（`+5.5σ .. +156.6σ`）均大幅超出 `3σ`。 |
| **`B1`**（`R1 §6`） | 高 | **已闭环** | 在 `README.md` 与 `§3` 锁定规范四轴参数账本（已补录 `s2_v2@600` `U=18.19` / `3.716B`）；`S2.3` 完成 `PassRouterAdapters` 因果消融。 | 轨道 D + 冲刺 2A | **PASS：** `no_pass_router` 致 NLL 恶化 `+0.00251`（`+6.8σ`）→ `KEEP_CRITICAL`。 |
| **`B2`** | 高 | **已闭环** | 轨道 B（`36d73eb`）启用 `mode="sparse"` 周期抽样 `lsa/last_recall`；冲刺 2A 录得实时召回率（`4K` 为 `0.8360`，`8K–128K` 为 `0.6171 → 0.1340`）。 | 轨道 B + 冲刺 2A | **PASS：** `s2_v2@600` 在 `128K` 下稀疏–稠密差距仅 **`+0.00050`**（较 `s3_v2` 的 `+0.06457` 收紧 `129×`）。 |
| **`B3`** | 高 | **部分闭环（`128K` 已证）** | 冲刺 2A（`§7.3`）实测证明 `s2_v2@600` 的 `8K / 64K / 128K` NLL（稠密 `0.00746` / 稀疏 `0.00796`）与 4 针 RULER **`12/12 (100%)`**。 | 轨道 D + 冲刺 2A / 2B | **`8K–128K` PASS；`512K / 1M` 仍开放：** 阶段 4 触达 `5.5h` 上限与 `94.43 GiB` 峰值（`96,697 / 97,887 MiB`；OOM 告警时 `93.15 GiB` / `95,381 MiB`）分配器碎片化；完成前 `1M` 维持“显存可行（memory-feasible）”口径。 |
| **`B4`** | 中 | **已闭环** | 冲刺 2A（`§7.2`、`§7.3`）在统一留出、OOD 与 `8K–128K` RULER 上将**原版 `Qwen3.6 + YaRN 4×`** 与 `s1@1000`、`s3_v2@600`、`s2_v2@600` 同台对比。 | 冲刺 2A | **PASS：** `s2_v2@600` `128K` NLL 较原版低 `13.4×`（`0.00746` vs `0.10026`），RULER `12/12` vs `2/12`。 |
| **`B5`** | 低 | **已闭环（确认裁撤）** | 冲刺 2A `S2.3`（`§7.4`）直接消融 `QED` 与 `neg_eigval` 并记录至 `128K` 的 `GDN-2` `max_l2`（`186.2 → 191.4 → 191.0`，`0` NaN）。 | 冲刺 2A | **PASS（可执行结论）：** `no_qed`（`-0.00009`，`-0.31σ`）与 `no_neg_eigval`（`-0.00026`，`-0.97σ`）确证为 **`PRUNE_CANDIDATE`**，后续阶段予以裁撤以恢复严格 `[0, 1]` GDN-2 稳定界。 |
| **`C1`**（`QL60-009`） | 高 | **已修复并合入** | [`src/qloop/train/bptt.py`](../src/qloop/train/bptt.py) 分离 `ce_loss`（`loss/lm`）与 `aux_loss`（`loss/aux`）并按有效词元加权。 | 轨道 A（`70f789e`） | CPU 梯度一致 PASS 且冲刺 2A 纯 LM CE 验证完成。 |
| **`C2`** | 高 | **已修复并合入** | [`src/qloop/model/lsa.py`](../src/qloop/model/lsa.py)：`LSAManager.aux_loss()` 默认 `aux_reduction="mean"`，保持 `lsa_kl` 为 `O(1)` 量级。 | 轨道 B（`36d73eb`） | CPU 单元测试 PASS（`test_sprint1_lsa_loop.py`）。 |
| **`C3`** | 低 | **已修复并合入** | [`src/qloop/model/loop.py`](../src/qloop/model/loop.py)：安装期确定性识别路由器类型与属性（仅未知单测桩回退数值检测）。 | 轨道 B（`36d73eb`） | CPU 单元测试 PASS（`test_sprint1_lsa_loop.py`）。 |
| **`D1`** | 高 | **轨道 C 已修；抽取待办** | 轨道 C（`2b15d4c`、`581db44`）修复 `ModelOutput`、`logits_to_keep=0` 与 `qloop_commit`；将约 `7.9K` 行独立字符串抽为受 lint 模块留作冲刺后待办。 | 轨道 C + 待办 | 轨道 C 已合入；模块抽取安排于冲刺 2B 后。 |
| **`D2`** | 中 | **待办** | 将已退役 `hssd.py`（`13,255` 行）与 v1 配置归档至 `qloop/legacy/`，同时保留负向消融复现单测。 | 冲刺 1 后待办 | 安排于冲刺 2B 后（默认流水线已在 `QL60-006` 切至洁净 v2）。 |
| **`D3`** | 中 | **已闭环** | 按洁净 v2 与规范四轴参数账本重写 `README.md`；修正 `configs/base.yaml` 过期注释且未改动任何 YAML 键。 | 轨道 D（`2d23615`） | 已合入并通过 `test_configs_cli.py`、`test_experiment_report.py`。 |
| **`D4`** | 低 | **待办** | 将 `tests/test_experiment_report.py` 固定文案子串断言拆入独立 `make docs-check` 目标。 | 冲刺 1 后待办 | 安排于冲刺 2B 后。 |
| **`D5`** | 中 | **已闭环** | 清理 `report.py`、`build_sirsi_dataset.py` 与文档硬编码路径；`[dev]` 增加 `pyarrow>=16.0`；确立具有约束力的 `§5` 受限溯源剔除政策。 | 轨道 C（`2b15d4c`）与 D（`2d23615`） | 已合入并核验（`0` 硬编码本地用户路径）。 |

---

## 3. 规范参数核算表

为彻底消除评审 1 `§6` 与评审 2 `B1` 指出的口径混叠，本仓库对标定 `[12, 36)` 核心下的 `Qwen/Qwen3.6-35B-A3B` [1]（`16` 层非循环前奏+尾奏 `[0,12) ∪ [36,40)` + `24` 层循环核心 `[12,36)`，每层 `E = 256` 路由专家，Top-$K$ `K = 8`，`d_model = 2048`，`d_moe = 512`，非绑定词表输入嵌入与输出头 `V = 248,320`）执行**严格的四轴参数账本**。

### 3.1 精确闭式构成公式（`tests/test_loop_hssd_v3_math.py`）

$$\begin{aligned}
P_{\text{expert}} &= 3 \times 2048 \times 512 = 3{,}145{,}728 \quad (3.1457\text{M}), \\
P_{\text{routed}} &= 40 \times 256 \times P_{\text{expert}} = 32{,}212{,}254{,}720 \quad (\mathbf{32.21225\text{B}},\; 100\%\text{ 冻结}), \\
P_{\text{shared}} &= 40 \times 1 \times P_{\text{expert}} = 125{,}829{,}120 \quad (\mathbf{0.12583\text{B}}), \qquad P_{\text{routed}+\text{shared}} = \mathbf{32.33808\text{B}}, \\
P_{\text{dense,uniq}}^{\text{ex-emb}} &= P_{\text{shared}} + P_{\text{routers}} + P_{\text{lm\_head}} + P_{\text{GDN}(30)} + P_{\text{Attn}(10)} \\
&= 0.12583 + 0.02105 + 0.50856 + 1.01155 + 0.27263 = \mathbf{1.93962\text{B}}, \\
P_{\text{uniq}}(U) &= P_{\text{dense,uniq}}^{\text{ex-emb}} + (16 \times 8 + 24 U) \times P_{\text{expert}}.
\end{aligned}$$

### 3.2 规范四轴参数核算总表

| 核算维度 | 适用范围 / 工况 | 精确参数量 | 拆解明细与强制解读说明 |
| :--- | :--- | :---: | :--- |
| **1. 存储基座参数** | `Qwen3.6-35B-A3B` 解码器总量（`40L`） | **`34.661B`** | `32.212B` 冻结路由 MoE（SHA-256 校验）+ `0.126B` 共享专家 + `2.323B` 混合器/路由器/非绑定 embed+head（`1.012B` GDN + `0.273B` 全局注意力 + `0.021B` 路由器 + `1.017B` embed+head + `~0.18M` 归一化层）。 |
| **2. 每词元唯一调阅参数**（不含输入嵌入与归一化） | `d=1`（`U=8`） | **`2.947B`** | `1.9396B` 稠密唯一参 + `1.0066B` 路由专家（`40L` 共 `320` 次唯一专家读取；不含归一化 `2.9463B` / 含归一化 `2.947B`）。 |
| | `s2_v2@450`（`U=14.96`，`1.522×`，中期） | **`3.472B`** | `1.9396B` + `(128 + 24 × 14.96) × 3.1457M = 1.5321B` 唯一路由专家（`3.4717B`）。 |
| | `s1@1000`（`U=17.36`） | **`3.653B`** | `1.9396B` + `(128 + 24 × 17.36) × 3.1457M = 1.7133B` 唯一路由专家（`3.6529B`）。 |
| | **`s2_v2@600`（`U=18.19`，`1.764×`，完结态）** | **`3.716B`** | `1.9396B` + `(128 + 24 × 18.19) × 3.1457M = 1.7760B` 唯一路由专家（`3.7156B`；Hub `17382c2e166d`）。 |
| | `R=3` 理论上限（`U=24`） | **`4.154B`** | `1.9396B` + `(128 + 24 × 24) × 3.1457M = 2.2146B` 唯一路由专家（`4.1542B`）。 |
| **3. 算力等效参数应用量**（含非绑定 embed+head） | `40L`（`d=1`，`R=1`） | **`3.454B`** | 单轮前向（不含输入嵌入 `2.946B`；不含 embed/head 预填充 GEMM 为 `2.436B`）。 |
| | `64L`（`d=2`，`R=2`） | **`4.916B`** | `+1` 轮核心（不含输入嵌入 `4.409B`；`kv_share=True` 跳过 `6` 个核心 `k/v` 省 `0.0126B` $\to$ 预填充 GEMM 为 `3.885B`）。 |
| | `88L`（`d=3`，`R=3`） | **`6.378B`** | `+2` 轮核心（不含输入嵌入 `5.871B`）。**重要警示：** 算力等效应用量不等于有效推理增益——中期 `s2_v2@450` 退出门控幅值仅为 **$\lvert g \rvert = 0.022\text{–}0.058$**，且 `s2_v2@600` 在未见 OOD 上 `d=1`（`1.69971`）仍略优于 `d=3`（`1.70122`）。 |
| **4. 可训练增量参数** | `s1_loop_a6b` / `s3_v2` / `s2_v2` | **`18.41M` / `24.56M` / `156.94M`** | 分别仅占解码器存储总参的 `0.053%` / `0.071%` / `0.453%`（`0` 个路由 MoE 参数被修改）。 |

**统一披露规范：** 今后全部文档、论文草稿与模型卡均须将**每词元唯一调阅参数（`2.947B → 3.472B–3.716B`，理论上限 `4.154B`）**与**算力等效参数应用量（`3.454B / 4.916B / 6.378B`）**明确区分，且引用算力等效指标时必须同步披露实测退出门控幅值（`|g|`）与参数匹配 `d=1` 对照结果（`A2`）。

---

## 4. `depth_gain = 0.0062` 与 `D-LSA` 召回失明的文献归因诊断

中期 `s2_v2@450` 遥测（`a5b20b20`）揭示了同流 `eval/depth_gain` 压缩至 `+0.0062`（`d1=0.5026 → d2=0.4999 → d3=0.4965`）及 `lsa/last_recall` 恒为 `0.0` 的深层机理。结合最新循环深度、稀疏注意力与线性注意力文献，其四重叠加成因现已于**冲刺 2A**（`§7`）获得直接实证检验：

1. **改造式循环的捷径坍塌（`McLeish et al.` [14]、`MoR` [15]、`Recurrent Depth` [16]、`mHC` [8]）：**
   - 在 `s2_v2@450` 中，`mHC` `ReEntryBridge` [8] 在第 1 轮前奏锚点上保留了高达 **`bridge_w_anchor = 0.831 / 0.866`** 的权重，分配给递推隐状态的权重仅为 **`bridge_w_h_prev = 0.057 / 0.050`**，同时 `PassExitGate` 处于近零区间（**`gate_abs_mean = 0.022 / 0.058`**），且第 3 轮路由回漂向第 1 轮（`Jaccard p12 = 0.080` vs `p13 = 0.438`）。
   - 正如 *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence*（`McLeish et al., arXiv:2511.07384` [14]）所剖析，强输入锚点注入极易诱发**捷径局部极小**。**冲刺 2A 实证（`§7.2`、`§7.4`）：** 在 `s2_v2@600` 上，`low_anchor_probe`（`max_anchor_weight = 0.25`，`<= 0.3`）令留出 NLL 恶化 `+0.00370`（`+11.05σ`）——证实已训练检查点依赖 `w_anchor ≈ 0.85`，必须通过**训练课程**而非零样本调权打破捷径；同时在未见 `wikitext-103-raw-v1` OOD 上，`s2_v2@600` `d=1`（`1.69971`）略优于 `d=3`（`1.70122`，`-4.2σ`），而 `s1@1000`（`+13.2σ`）与 `s3_v2@600`（`+38.4σ`）仍保持正向 OOD 深度增益。
2. **原生循环语言模型预训练规模鸿沟（`Ouro` [17] 与 `Inkling 2 Mini` [11]）：**
   - 原生循环预训练体系如 **Ouro**（`arXiv:2510.25741` [17]）在 **`7.7 万亿`（`7.7T`）词元**上从零训练共享循环模块并配以熵正则自适应深度目标；`Recurrent Depth` [16] 亦在 `800B` 词元上预训练，`MoR` [15] 则端到端训练词元级自适应深度路由。
   - 相比之下，本改造方案仅在冻结 `32.212B` 单轮专家之上，用 `157.29M` 词元（`s2_v2@600`，较 Ouro [17] 少约 `49,000×`）更新 `0.053%–0.453%` 参数。即便 `4K` 无泄漏留出深度增益显著（`+0.03609` 至 `+0.09327`，`+163σ .. +314σ`），冲刺 2B 的 `~18.41M` `d=1` LoRA 对照（`A2`）依然是剥离“循环增益”与“适配器容量”的必要门控。
3. **原始 `D-LSA` 蒸馏目标与稀疏盲区修复（`Original LSA` [6]）：**
   - 原始 `D-LSA`（`arXiv:2610.42147` [6]）采用逐键剩余桶 KL 蒸馏并配显式温度标定（`calibrate_tau`），而此前 [`src/qloop/model/lsa.py`](../src/qloop/model/lsa.py) 对 `10` 层 × 轮次求和（`C2`）且在 `mode="sparse"` 下 `lsa/last_recall = 0.0`（`B2`）。
   - **冲刺 1 轨道 B + 冲刺 2A 实证（`§7.3`、`§7.4`）：** 均值归约 `aux_loss()` 与稀疏 `last_recall` 抽样已合入（`36d73eb`）。在 `s2_v2@600` 上，稀疏 `D-LSA` 于 `4K` 取得 `lsa/last_recall = 0.8360`（且优于强制 `lsa_dense` `-0.00192` nats，`+5.5σ`），并在 `128K` 下将稀疏–稠密 NLL 差距压至 **`+0.00050`**（较 `s3_v2@600` 的 `+0.06457` 收紧 `129×`）。
4. **线上 `QL60-004` `step 340` 学习率重置与 `B5` `GDN-2` 扩展项实证裁决：**
   - `step 340` 跨会话仅权重恢复导致 `step 450/600` `lr_factor` 跳回 `0.9738` 的缺陷现已由轨道 A（`70f789e`）彻底根除。
   - 完结态 `s2_v2@600`（`17382c2e166d`）的 `gdn2/adapter_norm = 0.845` 与 `decay_horizon_mult_max = 7.60×` 持续走强，而 `qed_lambda = 0.00085` 与 `neg_eigval = 0.99986` 仍停留在恒等附近。**冲刺 2A `S2.3`（`§7.4`）对 `B5` 给出决定性实证：** 关闭 `GDN-2` [3] 触发 **`+0.03887`（`+156.6σ`）** 断崖式退化（**头号驱动项**），而关闭有界 `QED` [4]（`-0.00009`，`-0.31σ`）或软负特征值 [5]（`-0.00026`，`-0.97σ`）带来**零退化**——二者正式定性为 **`PRUNE_CANDIDATE`**，后续阶段裁撤以简化内核并恢复严格 `[0, 1]` `GDN-2` 稳定界。

---

## 5. 明确的数据溯源与公开发布政策（`D5`）

评审 2（`D5`）指出内部阶段配置（`configs/stages/s0_probe.yaml`、`s1_loop_a6b.yaml`、`s2_v2_joint_bptt.yaml`）及 `Loop64-Fable-SIRSI-v1` [13] 中包含第三方 Claude 蒸馏数据集：
- `opus46_3000x`（`nohurry/Opus-4.6-Reasoning-3000x-filtered`；阶段权重 `0.05–0.08`，`sft_core` 占 `538` 条）
- `opus46_traceinv`（`Jackrong/Claude-opus-4.6-TraceInversion-9000x`；阶段权重 `0.07`，`sft_core` 占 `533` 条）
- `opus46_3300x`（`Crownelius/Opus-4.6-Reasoning-3300x`；`sft_core` 去重后占 `36` 条）

尽管这些上游 Hub 仓库标注了 `apache-2.0` 元数据标签，但源自闭源第三方 API 的合成蒸馏轨迹存在上游服务条款与溯源风险。为此，本项目确立**具有约束力的双层数据溯源政策**：

1. **受限溯源内部诊断定级：** `opus46_3000x`、`opus46_traceinv` 与 `opus46_3300x` 正式归类为**仅限内部研究诊断的受限溯源混合源**。
2. **公开发布与资助申请零容忍剔除：** 任何公开完整模型包权重发布、公开数据集卡片发布、外部算力资助申请或学术论文提交，均**严禁包含任何 `opus46_*` 样本**。
3. **纯宽松开源权重替代栈：** 所有公开发布及冲刺 2 之后的洁净重训均须 100% 采用纯粹宽松协议且来源清晰的开源权重/开放许可语料：
   - **`OpenThoughts3`**（`open-thoughts/OpenThoughts3-1.2M`，`apache-2.0`）[18]，严格经由 `think_unclosed` + 重复循环 + AST 门控筛选；
   - **`Llama-Nemotron`**（`nvidia/Llama-Nemotron-Post-Training-Dataset`，`cc-by-4.0`）[19]，保留逐条 NVIDIA 署名；
   - **`OpenThoughts-Agent-SFT`**（`open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K`，`apache-2.0`）；
   - **`ARC-AGI-2` `training` 集**（`apache-2.0`）与**作者原创 / 开源权重 `SIRSI`**（`Loop64-Fable-SIRSI-v1` 纯开源洁净构建 [13]），全部通过 `13`-gram `2026` 基准去污索引（`1,889,432` 条 13-gram）过滤。

---

## 6. 可追踪双冲刺修复与验证计划

### 6.1 冲刺 1 —— CPU 工程、正确性与卫生修复（`0` GPU-h）—— **已 100% 完成并合入**

| 轨道 / 分支 | 状态 | 关闭发现项 | 合入 `main`（`a94c163`）的验证范围 |
| :--- | :---: | :---: | :--- |
| **轨道 A**（`fix/sprint1-eval-bptt-resume`） | **已合入（`70f789e`）** | `QL60-003`（`A1`）、`QL60-009`（`C1`）、`QL60-004`、`QL60-010` | `mixture.py`、`trainer.py`、`bptt.py`、`checkpoint.py`、`optim.py`、`runners.py`、`tests/test_sprint1_train_data_resume.py` —— `0` 重叠留出、有效词元 BPTT CE 与纯 `loss/lm`、`self.rng` + 仅权重 `lr_factor` 同步、模型指纹绑定评测缓存。 |
| **轨道 B**（`fix/sprint1-lsa-correctness`） | **已合入（`36d73eb`）** | `QL60-007`、`QL60-008`、`C2`、`B2`、`C3` | `lsa.py`、`loop.py`、`tests/test_sprint1_lsa_loop.py` —— 调用方掩码 `< 1e-6`、按轮分键未共享 `BucketIndex` `< 1e-12`、均值归约 `aux_loss()`、稀疏 `last_recall` 采样、安装期确定性路由器识别。 |
| **轨道 C**（`fix/sprint1-hf-serving-ops`） | **已合入（`2b15d4c`、`581db44`）** | `QL60-001`、`QL60-002`、`QL60-005`、`QL60-006`、`D5`（代码） | `engine.py`、`hf_custom_modeling.py`、`export_pack.py`、`colab.py`、`colab_supervisor.py`、`report.py`、`build_sirsi_dataset.py`、`pyproject.toml`、`tests/test_sprint1_hf_serving_ops.py` —— TF `5.24.1` `QuantizedHybridCache`（真机 PASS）、`ModelOutput` + `logits_to_keep=0`（`cache` 参数并行闭环）、`13/13` `_local_stage_done()` 门控、洁净 v2 默认阶段、可移植路径 + `pyarrow>=16.0`。 |
| **轨道 D**（`docs/midway-review-and-cleanup`） | **已合入（`2d23615`）** | `B1`、`D3`、`D5`（文档与政策） | `MIDWAY_REVIEW_AND_ACTION_PLAN{,_ZH}.md`、`README.md`、`DATASET_AUDIT_REPORT.md`、`IMPLEMENTATION_PLAN.md`、`configs/base.yaml` —— 严格 IEEE `[1]..[20]` 双语对齐、`GDDR7 ECC` 口径、`0` 用户路径、约束性 `§5` 溯源政策。 |
| **隔离冲刺 2 评测套件** | **已合入（`591b0df..71c4a99`）** | 预 GPU `S2.1–S2.3`、`K0`、TF `5.24` | 经 CPU 验证的 [`src/qloop/memory/engram.py`](../src/qloop/memory/engram.py)、[`src/qloop/eval/kill_gate_k0.py`](../src/qloop/eval/kill_gate_k0.py)、[`scripts/verify_tf524_serving_gpu.py`](../scripts/verify_tf524_serving_gpu.py)、[`src/qloop/eval/sprint2_audit.py`](../src/qloop/eval/sprint2_audit.py) 以及 [`src/qloop/model/d1_lora.py`](../src/qloop/model/d1_lora.py) + [`configs/stages/s2_2_d1_lora_control.yaml`](../configs/stages/s2_2_d1_lora_control.yaml)。 |
| **冲刺 1 后清理项** | **延后执行** | `D1`、`D2`、`D4` | 将约 `7.9K` 行独立代码字符串抽为受 lint 模块；将 `hssd.py`（`13,255` 行）与 v1 配置归档至 `qloop/legacy/`；将文案断言拆入 `make docs-check`。 |

### 6.2 冲刺 2 —— 决定性单卡科学验证（**冲刺 2A 已执行 · 冲刺 2B 待执行**）

于单张 `NVIDIA RTX PRO 6000 Blackwell Server Edition`（`96 GB GDDR7 ECC` [`94.97 GiB` 可用]，`1,597 GB/s`，`sm_120`）执行：

| 工作流 | 状态 | 算力预算 | 闭环发现项 | 实测结果 / 剩余门控 |
| :--- | :---: | :---: | :---: | :--- |
| **S2.1 无泄漏留出 + OOD 与 `8K–128K` 能力审计** | **2A 已完成（`512K/1M` 开放）** | 已耗 `~6` GPU-h（`512K/1M` 待补） | `A1`、`A3`、`B2`、`B3`、`B4` | **已完成（`§7.2`、`§7.3`）：** `s2_v2@600` 在 `0/4` 重叠留出集较原版 PPL `-10.71%`（`+469σ`）、未见 `wikitext-103` OOD `-1.11%`（`+33.6σ`）、`128K` NLL 低 `13.4×`（`0.00746`）、稀疏–稠密差 `+0.00050` 且 RULER [20] `12/12`（`0` NaN）。**开放项：** `512K / 1M` 阶梯（阶段 4 触达 `5.5h` 上限与 `94.43 GiB` 峰值 [`96,697 / 97,887 MiB`；OOM 告警时 `93.15 GiB` / `95,381 MiB`] 分配器碎片化）。 |
| **S2.3 因果消融 + 窄域 `K0` 预言机 + 真机 TF `5.24.1`** | **2A 已完成（`QL60-002` 待复验）** | 已耗 `~2` GPU-h | `B1`、`B5`、`QL60-001/002/005`、`LSA-NL` 门控 | **已完成（`§7.4`–`§7.6`）：** `GDN-2`（`+156.6σ`）、`PassRouter`（`+6.8σ`）与 `lsa_sparse`（`0.8360` 召回）为 `KEEP_CRITICAL`；`QED`（`-0.31σ`）与 `neg_eigval`（`-0.97σ`）为 `PRUNE_CANDIDATE`；`K0` = **`YELLOW_B_BLOCK_ATTN_CALIB`**（`6.05×` 小 KV 下 `100.0%` EM 保持率，`+0.6228`-nat 差距）；TF `5.24.1` `QL60-001` 与 `QL60-005` PASS（`QL60-002` `cache` 参数复验待补）。 |
| **S2.2 参数量匹配 `d=1` LoRA 对照（冲刺 2B）** | **待执行（套件已合入）** | `~18` GPU-h | `A2` | 在 `[12, 36)` 上以相同哈希切分 `s1` 数据与调度训练 `~18.41M` `d=1`（`40L`）LoRA 基线（`src/qloop/model/d1_lora.py`、`configs/stages/s2_2_d1_lora_control.yaml`）。**决定性门控：** 循环 `d >= 2` 须在无泄漏留出与未见 OOD 上打平或超越 `d=1` LoRA 方可解冻 `s4_v2_rl`。 |

### 6.3 明确的放行 / 终止解冻判据（当前门控状态）

- **解冻 `s4_v2_rl`（SI-RSI GRPO）：** 须满足冲刺 1（`100%` **PASS**）、S2.1 `8K–128K`（**PASS**；`512K/1M` 待补），**以及冲刺 2B `S2.2` `d=1` LoRA 对照（待执行）**。由于 `s2_v2@600` 在未见 OOD 上 `d=1`（`1.69971`）略优于 `d=3`（`1.70122`），`S2.2` 具备严格一票否决效力：若循环 `d >= 2` 未跑赢等参 `d=1` LoRA，则继续暂停 RL，转而迭代低锚点循环训练课程（`McLeish et al.` [14]）或交付裁撤了 `QED` 与 `neg_eigval` 的洁净 `d=1` 长上下文 `GDN-2 + D-LSA` 主干。
- **解冻 `LSA-NL` / `CoMem`（`C2–C6`）训练：** 须满足全部 `s4_v2_rl` 前置条件**外加**通过轻量 `2–4h` 单卡块注意力 APOLLO [7] 标定（仅调 `D-LSA` 索引器与全局 `q/k/v/o` 适配器，保持 `32.212B` MoE 100% 冻结）将 `K0` 从 `YELLOW_B_BLOCK_ATTN_CALIB`（`+0.6228`-nat NLL 差距）提升至 `GREEN`（`<= 0.050`-nat）。
- **解冻公开完整模型包发布：** 须通过上述全部门控、完成真机 `transformers 5.24.1` `.generate()` `4/4 PASS` 复验（`QL60-001`、`QL60-002`、`QL60-005`）、补齐生成式基准（`AIME`/`GPQA`/`ARC-AGI-2`），并确保 **100% 剔除 `opus46_*` 受限溯源数据集**（`§5`）。

---

## 7. 冲刺 2A Blackwell GPU 实证结果（`S2.1`、`S2.3`、`K0`、真机 `transformers 5.24.1`）

本节全部数据均直接在单张 `NVIDIA RTX PRO 6000 Blackwell Server Edition`（`96 GB GDDR7 ECC` [`94.97 GiB` 可用]，`1,597 GB/s`，`sm_120`）上实测采集，环境为 Python `3.13.16`、PyTorch `2.14.1+cu130`、`transformers 5.24.1`，代码基于提交 `a94c163`（原始日志见 `runs/sprint2a/{sprint2_audit_8k_128k.md,kill_gate_k0.md,serving_audit.md}`）。

### 7.1 `s2_v2_joint_bptt@600` 完结态实测记录（`17382c2e166d`）

- **Hub 检查点：** `TassP0lar/Qwen3.6-35B-A3B-Loop64` 提交 `17382c2e166d`（`2026-11-26T03:14:49Z`）[12]。
- **训练规模与硬件包络：** `600/600` 步，`157,286,400`（`157.29M`）词元，`156,941,238` 可训练参数（`32.212B` 路由 MoE 100% 冻结），单张 `NVIDIA RTX PRO 6000 Blackwell Server Edition`（`96 GB GDDR7 ECC`，`1,597 GB/s`）峰值显存 `69.97 GiB`。
- **终态架构遥测：** 核心唯一专家数 `U = 18.19`（`1.764×` 等效专家倍率，**每词元唯一调阅参数 `3.716B`**）、`gdn2/adapter_norm = 0.845`、`decay_horizon_mult_max = 7.60×`、`qed_lambda = 0.00085`、`neg_eigval = 0.99986`。

### 7.2 无泄漏 SHA-1 留出集（`4K`）与纯未见 OOD（`wikitext-103-raw-v1`）深度扫描（`S2.1`）

采用纯有效词元语言建模交叉熵（`3` 次重复、种子 `42`、`eval_seq_len = 4096`），分别评测于：(i) **无泄漏 SHA-1 留出集**（校验相对 `13` 篇训练文档精确 `0/4` 文档重叠、`8` 批、`19,093` 有效词元、SHA-256 `d359f35a91560494`）；(ii) **纯未见 OOD**（`wikitext-103-raw-v1` 测试集、`8` 批、`7` 文档、`30,030` 有效词元、SHA-256 `0fb192cb5d2b1952`）。

| 检查点臂 | 留出集 `d=1 → d=2 → d=3` NLL（`PPL@d_max`） | 留出 `d1→d3`（`z`） | 留出 vs 原版（`z`） | 未见 OOD `d=1 → d=2 → d=3` NLL（`PPL@d_max`） | OOD `d1→d3`（`z`） | OOD `d_max` vs 原版（`z`） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **原版 `Qwen3.6 + YaRN 4×`**（`d=1`）[1], [9] | `0.95000`（`2.586`） | — | 基线 | `1.71242`（`5.542`） | — | 基线 |
| **`s1_loop_a6b@1000`** | `0.94292 → 0.89579 → 0.87588`（`2.401`） | `+0.06704`（`+163σ`） | `-7.14%` PPL（`+198σ`） | `1.71089 → 1.70909 → 1.70439`（`5.498`） | `+0.00649`（`+13.2σ`） | `-0.80%` PPL（`+20.7σ`） |
| **`s3_v2_long_1m_lsa@600`** | `0.94320 → 0.87827 → 0.84992`（`2.339`） | `+0.09327`（`+314σ`） | `-9.52%` PPL（`+368σ`） | `1.71055 → 1.70916 → 1.70284`（`5.490`） | `+0.00771`（`+38.4σ`） | `-0.95%` PPL（`+48.8σ`） |
| **`s2_v2_joint_bptt@600`** | `0.87281 → 0.84861 → 0.83671`（`2.309`） | `+0.03609`（`+309σ`） | **`-10.71%` PPL（`+469σ`）** | `1.69971 → 1.70360 → 1.70122`（`5.481`） | `-0.00151`（`-4.2σ`） | **`-1.11%` PPL（`+33.6σ`）** |

- **核心科学解读：** 三个洁净 v2 阶段在无泄漏留出与未见 OOD 上均以 `> 20σ` 显著跑赢**原版 `Qwen3.6 + YaRN 4×`**（`2× d=1` 非确定性 `max_diff` 稳定落在 `2.36e-4 .. 1.18e-3`）。然而，尽管 `s1@1000`（`+13.2σ`）与 `s3_v2@600`（`+38.4σ`）在未见 OOD 上维持单调 `d=1 → d=3` 提升，`s2_v2@600` 的未见 OOD 最优值出现在 `d=1`（`1.69971`，PPL `5.472`），略优于 `d=3`（`1.70122`，PPL `5.481`，`-4.2σ`）。这直接要求在解冻 `s4_v2_rl` 前必须执行冲刺 2B（`S2.2`）等参 `d=1` LoRA 对照实验。

### 7.3 `8K / 64K / 128K` 长上下文阶梯、多针 RULER 与 `GDN-2` 状态范数（`fp8` KV）

在 `kv_cache_quant=fp8` 下测试 `8K / 64K / 128K` 稠密与稀疏 NLL、4 针 **RULER** [20] 精确匹配率及 `GDN-2` 递推状态范数（`all_states_bounded=True`，全臂 `0` NaN）：

| 检查点臂（深度） | `8K` 稠密 / 稀疏 NLL | `64K` 稠密 / 稀疏 NLL | `128K` 稠密 / 稀疏 NLL | `128K` 稀疏−稠密差距 | 4 针 RULER（`8K, 64K, 128K`） | `GDN-2` `max_l2`（`8K → 64K → 128K`） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **原版 `Qwen3.6 + YaRN 4×`**（`d=1`） | `0.49777` / `N/A` | `0.15114` / `N/A` | `0.10026` / `N/A` | `N/A` | `0/4, 1/4, 1/4`（`2/12`） | `182.6 → 184.7 → 185.2`（`0` NaN） |
| **`s1_loop_a6b@1000`**（`d=3`） | `0.16217` / `N/A` | `0.05358` / `N/A` | `0.03030` / `N/A` | `N/A` | `3/4, 4/4, 0/4`（`7/12`，`128K` 坍塌） | `182.6 → 184.6 → 185.2`（`0` NaN） |
| **`s3_v2_long_1m_lsa@600`**（`d=3`） | `0.26157` / `0.28208` | `0.08195` / `0.17592` | `0.04913` / `0.11370` | `+0.06457` | **`4/4, 4/4, 4/4`（`12/12`，`100%`）** | `182.6 → 184.6 → 185.2`（`0` NaN） |
| **`s2_v2_joint_bptt@600`**（`d=3`） | `0.10123` / `0.10728` | `0.01362` / `0.01557` | **`0.00746` / `0.00796`** | **`+0.00050`**（**收紧 `129×`**） | **`4/4, 4/4, 4/4`（`12/12`，`100%`）** | **`186.2 → 191.4 → 191.0`**（`0` NaN） |

- **长上下文突破：** `s2_v2_joint_bptt@600` 将 `128K` 稠密 NLL 降至原版 `Qwen3.6 + YaRN 4×` 的 **`1/13.4`**（`0.00746` vs `0.10026`），将 `128K` 稀疏−稠密 `D-LSA` [6] 差距较 `s3_v2@600` 收紧 **`129×`**（`+0.00050` vs `+0.06457`），实现 **`12/12 (100%)`** 多针 RULER [20] 命中（原版仅 `2/12`，`s1` 于 `128K` 坍塌至 `0/4`），同时 `GDN-2` 递推状态范数保持平坦有界（`186.2 → 191.4 → 191.0`，`0` NaN）。

### 7.4 `S2.3` 零训练因果消融矩阵（`s2_v2@600`，`d=3`，SHA-256 逐臂复原校验）

在无泄漏 `4K` 留出集上执行 `3` 次重复，每项消融前后均经 SHA-256 权重逐位复原核验（`17399792700dbb1a`，`8/8 PASS`）：

| 消融项 | Mean ± Std NLL | `Δ NLL vs Full` | `z`-score | `lsa/last_recall` | 裁决与工程决策 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `full`（`s2_v2@600`，`d=3`） | `0.83719 ± 0.00035` | `+0.00000` | `+0.00σ` | `N/A` | **基准参考（`KEEP_CRITICAL`）** |
| **`no_gdn2`** | `0.87606 ± 0.00003` | **`+0.03887`** | **`+156.6σ`** | `N/A` | **`KEEP_CRITICAL`（头号因果驱动项）** —— 解耦 `GDN-2` [3] 通道门控贡献主导增益。 |
| **`no_pass_router`** | `0.83970 ± 0.00039` | **`+0.00251`** | **`+6.8σ`** | `N/A` | **`KEEP_CRITICAL`** —— 跨轮动态专家路由提供统计显著收益。 |
| **`lsa_sparse`** vs `lsa_dense` | `0.83730 ± 0.00033`（dense：`0.83912`） | `+0.00010`（dense：`+0.00192`） | `+0.30σ`（dense：`+5.5σ`） | **`0.8360`** | **`KEEP_CRITICAL`** —— 稀疏 `D-LSA` [6] 保持 `0.8360` 召回率，强制切回 `lsa_dense` 反而令 NLL 恶化 `+5.5σ`。 |
| **`no_qed`** | `0.83710 ± 0.00024` | **`-0.00009`** | **`-0.31σ`** | `N/A` | **`PRUNE_CANDIDATE`** —— 实证零增益；后续阶段裁撤有界 `QED` [4] 以简化内核。 |
| **`no_neg_eigval`** | `0.83694 ± 0.00014` | **`-0.00026`** | **`-0.97σ`** | `N/A` | **`PRUNE_CANDIDATE`** —— 实证零增益；裁撤软负特征值 [5] 以恢复严格 `[0, 1]` `GDN-2` 稳定界。 |
| **`low_anchor_probe`**（`max_anchor_weight = 0.25`，`<= 0.3`） | `0.84090 ± 0.00032` | **`+0.00370`** | **`+11.05σ`** | `N/A` | **`SHORTCUT_DIAGNOSTIC`** —— 证实已训检查点依赖 `w_anchor ≈ 0.85`；逃逸捷径坍塌（`McLeish et al.` [14]）须靠训练课程，非零样本改权。 |

### 7.5 止损门控 `K0`：零训练 Pre-RoPE KV-Mem Blob 注入（`s2_v2@600`，`N=20`）

跨 `20` 个多会话样本（`code_recall`、`codebase_qa`、`contradictory_update`、`multi_hop`、`needle`、`temporal_update`）评测，`sink_tokens = 4`、`boundary_header_tokens = 4`、`rotary_dim = 64`、`refresh_frac = 5%`、`green >= 92%` 且 `NLL gap <= 0.050` nats：

| 测试臂（`s2_v2@600`，`d=3`，`N=20`） | 答案 NLL | 相对 Oracle NLL 差 | 词元准确率 | 精确匹配（EM） | Oracle EM 保持率 | 挂载 KV 体积 | `temporal_update` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm (a) 连续上下文 Oracle** | `1.4803` | `+0.0000` | `0.827` | `0.350` | `100.0%` | `43.32 MB` | `100.0%` |
| **Arm (b) 朴素拼接 KV（Naive）** | `2.2032` | `+0.7229` | `0.793` | `0.350` | `100.0%` | `31.21 MB` | `73.8%` |
| **Arm (c) Pre-RoPE + Sink 剥离 + `H64-INT4`** | **`2.1031`** | **`+0.6228`**（较 Naive `-0.1001`） | **`0.821`** | **`0.350`** | **`100.0%`** | **`7.16 MB`（缩小 `6.05×`）** | **`95.7%`** |
| **Arm (d) Arm (c) + `5%` CacheBlend（`70` 词元）** | `2.0924` | `+0.6120` | `0.821` | `0.350` | `100.0%` | `7.16 MB` | `95.7%` |

- **裁决 —— `YELLOW_B_BLOCK_ATTN_CALIB`：** 在 `d=3` 下，Arm (c) 以 **`6.05×` 更小 KV 体积**（`7.16 MB` vs `43.32 MB`）达成 **`100.0%` EM 保持率**（`0.350` vs `0.350`）与近乎无损词元准确率（`0.821` vs `0.827`），将 `temporal_update` 从 `73.8% → 95.7%` 并优于朴素拼接 `-0.1001` nats（`d=1` 下 Arm (c) 获 `87.5%` EM 保持率、`0.827` vs `0.839` 词元准确率、`+0.6010` NLL 差、`temporal_update 73.8% → 100.0%`）。未达 `GREEN` 的唯一原因在于冻结投影下的跨会话动态误差 NLL 差距（`+0.6228` nats；`5%` CacheBlend `+0.6120`），支持在任意 `LSA-NL` 训练前先执行 `2–4h` 轻量单卡块注意力 APOLLO [7] 标定。

### 7.6 真机 `transformers 5.24.1` 推理服务审计（`a94c163`）与明确剩余开放缺口

- **真机 `transformers 5.24.1` 审计（`runs/sprint2a/serving_audit.md`，`2026-11-26T07:53:38Z`，`3/4 PASS`）：**
  - `ENV:strict_tf5`：**PASS**（实测 `transformers 5.24.1`）。
  - `QL60-001:quantized_hybrid_cache`：**真机 PASS** —— 在真实 `transformers 5.24.1` `CacheLayerMixin` 下验证 `88` 虚拟层，在 `d=1` 与 `d=3` 上对 `bf16`（`143,360 B`）、`fp8`（`72,240 B`，`0.5039×`；单元缓存契约 `3,168 / 6,144 B = 0.5156×`）及 `hadamard_int4`（`38,080 B`，`0.2656×`）实现精确 **`1.00×` KV**，且 `kv_share=False` 为精确 `2.00×`。
  - `QL60-005:stage_completion_gate`：**真机 PASS（`13/13`）** —— `_local_stage_done()` 全部接受/拒绝用例通过。
  - `QL60-002:pack_model_and_generate`：已通过 `ModelOutput` 与 `d=1` 包内 `logits_to_keep=0/1` 校验，随后在 `past_key_values=qcache` 路径 `.generate()` 抛出 `ValueError: The following model_kwargs are not used by the model: ['cache']`（精准定位 `transformers 5.24.1` 校验器要求 `forward()` 签名显式声明 `cache: Any = None`，故 `d=2, 3` 与独立打包检查待 `4/4 PASS` 复验），当前正在并行闭环。
- **明确剩余开放缺口（零虚假宣称）：**
  1. **`QL60-002` `cache: Any = None` 复验：** 并行签名修复合入后重跑 `scripts/verify_tf524_serving_gpu.py` 以取得 `4/4 PASS`。
  2. **`512K / 1M` 长上下文阶梯补齐（`B3`）：** 阶段 4 触达 `5.5h` 安全上限并伴随 `94.43 GiB` 峰值（`96,697 / 97,887 MiB`；OOM 告警时 `93.15 GiB` / `95,381 MiB`）PyTorch CUDA 分配器碎片化（`expandable_segments` OOM 告警）；需单独安排分配器清理后的 `512K / 1M` 评测后再将能力宣称延伸至 `128K` 以上。
  3. **冲刺 2B（`S2.2`）`~18.41M` 参数量匹配 `d=1` LoRA 对照（`A2`）：** `src/qloop/model/d1_lora.py` + `configs/stages/s2_2_d1_lora_control.yaml` 已合入并经 CPU 验证（`63d8b00`），待执行 `~18` GPU-h 训练——这是解冻 `s4_v2_rl` 或 `LSA-NL` 前的**决定性检验**。
  4. **生成式推理基准：** `AIME 2026`、`GPQA` 与 `ARC-AGI-2` 端到端生成式评测须待 `S2.2` 通过并解冻 `s4_v2_rl` 后执行。

---

## 参考文献

[1] Qwen Team, "Qwen3.6-35B-A3B," Alibaba Cloud, Model Release, Apr. 16, 2026. [Online]. Available: https://huggingface.co/Qwen/Qwen3.6-35B-A3B

[2] S. Yang, J. Kautz, and A. Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2412.06464.

[3] A. Hatamizadeh, Y. Choi, and J. Kautz, "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention," *arXiv preprint arXiv:2605.22791*, May 2026.

[4] D. Gupta, A. Das, and D. Gupta, "The Query Knows What to Forget: A Second Erase Direction for Linear Attention," *arXiv preprint arXiv:2608.13668*, Aug. 2026.

[5] R. Grazzi, J. Siems, A. Zela, J. K. H. Franke, F. Hutter, and M. Pontil, "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2411.12537.

[6] A. Kubicek, A. Caciolai, L. Peng, S. Sumant, B. Hoover, F. Greenlee, R. M. Battleday, R. Urban, T. Hoefler, and J. C. R. Whittington, "Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention," *arXiv preprint arXiv:2610.42147*, Oct. 2026.

[7] H. Zhu, Z. Zhang, W. Cong, X. Liu, S. Park, V. Chandra, B. Long, D. Z. Pan, Z. Wang, and J. Lee, "APOLLO: SGD-like Memory, AdamW-level Performance," *arXiv preprint arXiv:2412.05270*, Dec. 2024.

[8] Z. Xie *et al.* (DeepSeek-AI), "mHC: Manifold-Constrained Hyper-Connections," *arXiv preprint arXiv:2512.24880*, Dec. 2025.

[9] B. Peng, J. Quesnelle, H. Fan, and E. Shippole, "YaRN: Efficient Context Window Extension of Large Language Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024, arXiv:2309.00071.

[10] S. Yang and Y. Zhang, "FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanisms," GitHub Repository (fla-org/flash-linear-attention, v0.6.0), 2024–2026.

[11] Thinking Machines Lab, "Inkling 2 Mini," Technical Release, Nov. 20, 2026. [Online]. Available: https://thinkingmachines.ai

[12] TassP0lar, "Qwen3.6-35B-A3B-Loop64: Open Looped Hybrid Checkpoints and Empirical Telemetry," Hugging Face Model Hub, Nov. 2026. [Online]. Available: https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64

[13] TassP0lar, "Loop64-Fable-SIRSI-v1: Curated Anti-Slop 3-Beat Looped-CoT, Aesthetic Craft, and SI-RSI Recovery Dataset," Hugging Face Datasets Hub (v1.2), Nov. 2026. [Online]. Available: https://huggingface.co/datasets/TassP0lar/Loop64-Fable-SIRSI-v1

[14] S. McLeish, A. Li, J. Kirchenbauer, D. S. Kalra, B. R. Bartoldson, B. Kailkhura, A. Schwarzschild, J. Geiping, T. Goldstein, and M. Goldblum, "Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence," *arXiv preprint arXiv:2511.07384*, Nov. 2025.

[15] S. Bae, Y. Kim, R. Bayat, S. Kim, J. Ha, T. Schuster, A. Fisch, H. Harutyunyan, Z. Ji, A. Courville, and S.-Y. Yun, "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation," *arXiv preprint arXiv:2507.10524*, Jul. 2025.

[16] J. Geiping, S. McLeish, N. Jain, J. Kirchenbauer, S. Singh, B. R. Bartoldson, B. Kailkhura, A. Bhatele, and T. Goldstein, "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach," *arXiv preprint arXiv:2502.05171*, Feb. 2025.

[17] R.-J. Zhu, Z. Wang, K. Hua, T. Zhang, Z. Li, *et al.*, "Scaling Latent Reasoning via Looped Language Models," *arXiv preprint arXiv:2510.25741*, Oct. 2025.

[18] E. Guha *et al.*, "OpenThoughts: Data Recipes for Reasoning Models," *arXiv preprint arXiv:2506.04178*, Jun. 2025.

[19] A. Bercovich *et al.*, "Llama-Nemotron: Efficient Reasoning Models," *arXiv preprint arXiv:2505.00949*, May 2025.

[20] C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, Y. Zhang, and B. Ginsburg, "RULER: What's the Real Context Size of Your Long-Context Language Models?" *arXiv preprint arXiv:2404.06654*, Apr. 2024.
