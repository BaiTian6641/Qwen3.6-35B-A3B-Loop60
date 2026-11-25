# 中期外部评审交叉核验台账与双冲刺修复计划（`v1.0`）

> **英文对照版：** [English Version (`docs/MIDWAY_REVIEW_AND_ACTION_PLAN.md`)](MIDWAY_REVIEW_AND_ACTION_PLAN.md)  
> **受审仓库基线：** `0f5d7f7`（两份独立外部评审） · **当前仓库 `HEAD`：** `17216c0`  
> **实测 Hub 遥测快照：** `s2_v2_joint_bptt@450`（提交 `a5b20b20`，`2026-11-25T17:54:48Z`，`step=450/600`，`117,964,800` 词元，`n_params=156,941,238`）及 `s2_v2@400`（提交 `cec24a8d`）  
> **目标硬件：** `1× NVIDIA RTX PRO 6000 Blackwell Server Edition`（`96 GB GDDR7 ECC` [`94.97 GiB` 可用]，`1,597 GB/s`，`sm_120`）  
> **战略状态：** **冻结**后续 `LSA-NL`/`CoMem` 训练扩张、`s4_v2_rl` 以及公开完整模型包发布，直至**冲刺 1**（CPU 工程缺陷修复）与**冲刺 2**（`~25–35` GPU 小时无泄漏科学验证）全部通过。

---

## 1. 中期执行总评

两份独立的外部中期评审在仓库提交 `0f5d7f7` 上对本项目与线上 Hugging Face Hub 检查点元数据进行了系统审计：
1. **评审 1（`GPT-6 Astra Pro` 代码与系统审计 @ `0f5d7f7`）：** 指出 `10` 项具体代码、评测、推理服务与编排缺陷（`QL60-001` 至 `QL60-010`）以及参数口径不一致问题（`§6`）。
2. **评审 2（研究方法与代码库审计 @ `0f5d7f7` + 线上 Hub `s2_v2@450` `manifest.json` `a5b20b20`）：** 指出 `16` 项科学性、方法学、遥测与工程卫生问题，分为 `A1–A3`（科学有效性）、`B1–B5`（核心主张与遥测）、`C1–C3`（损失/索引器/路由器正确性）及 `D1–D5`（可维护性与数据溯源）。

我们已逐条对照当前 `main`（`17216c0`，其仅合入了隔离的阶段 2A/2B `LSA-NL` 原语 `src/qloop/memory/` 与双语架构文档，未改动任何在训代码路径）及线上 `s2_v2@450` Hub 遥测（`a5b20b20`）完成独立复核。**评审 1 的 `10/10` 项与评审 2 的 `16/16` 项发现全部确证属实。**

### 经受住检验的基础设施
- **单卡 Blackwell `sm_120` 系统栈：** 在单张 `NVIDIA RTX PRO 6000 Blackwell Server Edition`（`96 GB GDDR7 ECC` [`94.97 GiB` 可用]，`1,597 GB/s`，`sm_120`）上改造 `Qwen/Qwen3.6-35B-A3B` [1]（`30` 层原生 Gated DeltaNet [2] + `10` 层门控全局注意力，`32.212B` 路由 MoE 权重 100% 冻结且经 SHA-256 校验）运行稳定，融合 `GDN-2-1M` [3]、有界 `QED` [4]、软负特征值 [5]、逐轮 `D-LSA` [6]、低秩 `APOLLO` [7]、`mHC` `LoopBridge` [8]、纯 `4.0×` YaRN [9] 与 `fla 0.6.0` DPLR Triton 内核 [10]。
- **已验证工程里程碑：** 提交 `574898d` 的 `4×8K` 宏分段将 `T=32K` `GDN2Adapter` 瞬时显存压缩 `3.28×`（`23.051 → 7.032 GiB`，令 `4×32K d=3` 峰值稳定在 `67.194–67.77 GiB`）；`1M` `FP8` `1.0×` `SharedKV` 解析状态仅占 `10.65 GiB`（较 Inkling 2 Mini 的 `32.16 GiB` [11] 缩小 `3.02×`）；通过四阶段因果替换清晰隔离并退役 `HSSD v3.2` 负向消融；公开检查点 [12] 与经审计的 `v1.2` 数据产物 [13] 保持完整可追溯性。

### 审计暴露的阻塞缺陷（为何即刻冻结）
- **科学性与评测阻塞项：** `eval_quick` 与训练共享未划分子流（`QL60-003` / `A1`）；`depth_gain` 缺乏参数量匹配的 `d=1` LoRA 对照基线（`A2`）；线上 `s2_v2@450`（`a5b20b20`）仅录得 `eval/depth_gain = +0.0062`，且受限于仅权重 Hub 续训触发的静默学习率重置（`QL60-004`，`lr_factor = 0.9738`）、未按有效词元归一化且混入 KL 的 BPTT 日志（`QL60-009` / `C1` / `C2`，`loss/lm = 3.8421`）以及稀疏模式下失明的召回遥测（`B2`，`lsa/last_recall = 0.0`）。
- **正确性、推理服务与数据溯源阻塞项：** 稀疏 `lsa_attention()` 忽略调用方掩码（`QL60-007`）；`kv_share=False` 时复用过期第 1 轮 `BucketIndex` 缓存（`QL60-008`）；`QuantizedHybridCache` 与导出完整模型包 `.generate()` 在现代 `transformers 5.24+` 下崩溃（`QL60-001`、`QL60-002`）；`_local_stage_done()` 可能提前发布未完成模型包（`QL60-005`）；默认流水线编排仍指向已退役 HSSD 阶段（`QL60-006`）；且第三方 `opus46_*` 子集必须在任何公开发布前明确标记为受限溯源并剔除（`D5`）。

> [!IMPORTANT]
> **强制中期冻结令：** 后续一切 `LSA-NL` / `CoMem` 训练扩张、`s4_v2_rl` 训练与公开完整模型包发布均**严格冻结**，直至**冲刺 1**（轨道 A–D 全部 CPU 修复）与**冲刺 2**（`~25–35` GPU 小时无泄漏留出集 + `d=1` LoRA 对照 + 因果消融验证）通过第 6 节全部验收门控。

---

## 2. 全量交叉核验台账

### 2.1 线上 Hub 遥测实测基线快照（`s2_v2_joint_bptt@450`，提交 `a5b20b20`）

表 2.1 与表 2.2 中全部线上证据均直接溯源至 Hub 提交 `a5b20b20`（`2026-11-25T17:54:48Z`，前一检查点 `s2_v2@400` 提交 `cec24a8d` 同样录得 `eval/depth_gain = 0.0062`）[12]：

| 指标分组 | 线上 `s2_v2@450`（`a5b20b20`）实测遥测 | 诊断意义 |
| :--- | :--- | :--- |
| **运行与优化器** | `step=450/600`，`tokens_seen=117,964,800`，`n_params=156,941,238`，`has_optim=false`，`lr_factor=0.9738`（`optim/lr_factor=0.9747`），`global_grad_norm=0.4753`，`clipped_params=0/370` | 线上实证 `QL60-004`：`step 340` 后跨会话仅权重续训将余弦调度静默重置回近峰值学习率（`0.974`）。 |
| **评测与损失** | `eval/loss_d1=0.5026` $\to$ `d2=0.4999` $\to$ `d3=0.4965`，**`eval/depth_gain=0.0062`**；训练 **`loss/lm=3.8421`**（混入求和 KL！） | 未划分留出评测集上深度增益仅 `0.0062` nat（`A1`）；`loss/lm=3.8421` 混叠纯 CE 与求和 `lsa_kl`（`QL60-009` / `C1` / `C2`）。 |
| **循环与路由** | `bridge_w_anchor=0.831/0.866`，`bridge_w_h_prev=0.057/0.050`，`gate_abs_mean=0.022/0.058`，`router_repel=-0.047`，`Jaccard p12=0.080 / p13=0.438`，`unique_experts_core=14.96`，`effective_expert_mult=1.522×` | 输入锚点权重紧锁在 `0.85` 附近且退出门控极小（`0.022–0.058`），第 3 轮路由回漂向第 1 轮（`p13=0.438` vs `p12=0.080`）。 |
| **GDN-2 与 D-LSA** | `adapter_norm=0.741`，`erase_write_l1=0.0419`，`decay_horizon_mult_max=4.545×`，`qed_lambda_abs_mean=0.00071`，`neg_eigval_scale_mean=0.9998`，**`lsa/last_recall=0.0`** | 基础 GDN-2 通道门控正常唤醒，而 QED（`0.00071` vs `0.35`）与负特征值（`0.9998`）近乎未动（`B5`），且稀疏模式 `last_recall` 失明（`B2`）。 |

### 2.2 评审 1 发现核验台账（`GPT-6 Astra Pro`，`QL60-001`..`QL60-010`）

| 编号 | 交叉映射 | 核验状态 | 已确证代码 / 遥测证据（`17216c0` 与 `a5b20b20`） | 修复目标 | 验收门控 |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **`QL60-001`** | — | **已确证** | [`src/qloop/infer/engine.py:274-301`](../src/qloop/infer/engine.py#L274-L301)：`QuantizedHybridCache.__init__` 仅捕获 `super().__init__()` 的 `TypeError`，在 `transformers 5.24+` 下触发 `ValueError` 且缺失完整混合 `Cache` 协议；`update()`（`:324-337`）每次读取全量反量化 K/V。 | 冲刺 1 轨道 C | 在模拟 TF `5.24+` 下初始化与混合缓存接口（`conv_states`、`recurrent_states`、`reorder_cache`、`crop`、`reset`、`get_mask_sizes`）于 `native`/`fp8`/`hadamard_int4` 全通过。 |
| **`QL60-002`** | `D1` | **已确证** | [`src/qloop/infer/hf_custom_modeling.py:372,685-735,8740-8765`](../src/qloop/infer/hf_custom_modeling.py#L685-L735)：`CausalLMPackOutput` 未实现 `ModelOutput` 协议；外层导出配置缺失有效虚拟 `layer_types`；`logits_to_keep=0` 切片 `h_last[:, -0:, :]` 返回 `(B, 0, V)`。 | 冲刺 1 轨道 C | 兼容 `ModelOutput` 输出、同步虚拟 `layer_types`、`logits_to_keep=0` 返回完整 `(B, T, V)`，且独立生成代码 `.generate()` 单测通过。 |
| **`QL60-003`** | **`A1`** | **线上确证** | [`src/qloop/train/trainer.py:250-267`](../src/qloop/train/trainer.py#L250-L267) 与 [`src/qloop/data/mixture.py:62-160`](../src/qloop/data/mixture.py#L62-L160)：`_eval_set()` 仅加随机种子偏移（`self.seed + 7_777_777`）从同一 `data.sources` 采样，导致训练文档泄漏入 `eval/loss_d*`（线上 `0.5026`）。 | 冲刺 1 轨道 A + 冲刺 2 | 打包前 SHA-1 文档哈希切分（`split_role="train"` / `"eval"`）在 8 文档复现中实现精确 `0` 重叠；冲刺 2 重评全部检查点与 OOD 集。 |
| **`QL60-004`** | — | **线上确证** | [`src/qloop/train/trainer.py:209,324-342`](../src/qloop/train/trainer.py#L324-L342) 与 [`src/qloop/train/checkpoint.py:121-156`](../src/qloop/train/checkpoint.py#L121-L156)：未序列化 `self.rng`；当 `optim.pt` 缺失（`has_optim=false`）时 `self.opt.step_count` 保持 `0`，导致线上 `s2_v2@450`（`a5b20b20`）在 `step 340` 重启后跳回 `lr_factor=0.9738`。 | 冲刺 1 轨道 A | 续训严格还原 `trainer.rng.getstate()`；仅权重恢复显式记录 `resume_mode="weights_only"` 并同步 `self.opt.step_count = self.step`（除非显式选择重置）。 |
| **`QL60-005`** | — | **已确证** | [`src/qloop/colab.py:1133-1140,1190-1197`](../src/qloop/colab.py#L1133-L1197)：`_local_stage_done()` 仅凭 `LATEST` 存在即判完成，可能在暂停或未完结阶段（`is_completed=False`）误触发完整模型包发布。 | 冲刺 1 轨道 C | 强制校验显式 `COMPLETED` / `PROMOTED` 或 `step >= target_steps`；附录 F 暂停态 `step000001` 夹具返回 `all_stages_completed=False, pack_shipped=False`。 |
| **`QL60-006`** | `D2`、`D3` | **已确证** | [`src/qloop/colab.py:1148-1155,1200-1201`](../src/qloop/colab.py#L1148-L1155) 与 [`src/qloop/colab_supervisor.py:48-55`](../src/qloop/colab_supervisor.py#L48-L55)：默认阶段列表仍调用已退役 HSSD 阶段（`s1b_hssd_distill`、`s3_long_1m_lsa`、`s2_joint_bptt`、`s4_rl`）。 | 冲刺 1 轨道 C | 默认阶段序列切换为洁净 v2（`s0_probe`、`s1_loop_a6b`、`s3_v2_long_1m_lsa`、`s2_v2_joint_bptt`、`s4_v2_rl`），HSSD 仅作 `--stages` 显式回溯项。 |
| **`QL60-007`** | — | **已确证** | [`src/qloop/model/lsa.py:3128-3205`](../src/qloop/model/lsa.py#L3128-L3205)：默认 `strict_mask=False` 时稀疏 `lsa_attention()` 忽略调用方 `attention_mask`，导致填充/跨文档键泄漏（评审 1 复现 `max_err = 79.2`）。 | 冲刺 1 轨道 B | 在 sink、局部窗与桶选键全路径强制布尔/浮点 2D/3D/4D 掩码并保留 `None` 零开销快路径；左/右填充与分块对角复现误差 `< 1e-6`。 |
| **`QL60-008`** | — | **已确证** | [`src/qloop/model/lsa.py:577-615`](../src/qloop/model/lsa.py#L577-L615)：`BucketIndex` 缓存键缺少轮次号（`("pkv", pkv_id, layer_id, ...)` / `("loop", id(loop_ctx), layer_id, ...)`），在 `kv_share=False` 下令第 2 轮复用第 1 轮索引（`0.927` logit 误差）。 | 冲刺 1 轨道 B | `kv_share=False` 时按 `(phys_layer, cur_pass)` 分键缓存，同时保持 `kv_share=True` 物理复用；float64 两种模式复现误差均 `< 1e-12`。 |
| **`QL60-009`** | **`C1`** | **线上确证** | [`src/qloop/train/bptt.py:2045-2093`](../src/qloop/train/bptt.py#L2045-L2093)：分段反传按 `/ float(n_segs)` 而非有效监督词元占比（`valid_tok_s / total_valid_toks`）加权，并将辅助项（`lsa_kl`）混入 `loss/lm`（线上 `3.8421`）。 | 冲刺 1 轨道 A | 按 `valid_tok_s / total_valid_toks` 归一化 LM CE（安全处理零有效段）；分离上报纯 `loss/lm` 与 `loss/aux`；`1 vs 3` 有效词元双段梯度完全一致。 |
| **`QL60-010`** | — | **已确证** | [`src/qloop/eval/runners.py:35-53,144-165`](../src/qloop/eval/runners.py#L144-L165)：可恢复的 `depth_<d>.jsonl` 缓存不校验检查点/模型身份，同 `out_dir` 下会静默混用不同检查点的预测结果。 | 冲刺 1 轨道 A | 持久化规范运行指纹（`model_id`、检查点摘要/时间戳、`active_passes`、`config_hash`、`dataset_rev`）；不匹配时默认抛出 `ValueError`（除非 `allow_mixed_cache=True`）。 |

### 2.3 评审 2 发现核验台账（代码与研究方法审计，`A1–A3`、`B1–B5`、`C1–C3`、`D1–D5`）

| 编号 | 严重度 | 核验状态 | 已确证代码 / 遥测证据（`17216c0` 与 `a5b20b20`） | 修复目标 | 验收门控 |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **`A1`**（`QL60-003`） | 致命 | **线上确证** | [`src/qloop/train/trainer.py:257-267`](../src/qloop/train/trainer.py#L257-L267)：`eval_quick` 无文档留出即从训练源拉取样本；`s2_v2@450` 的 `eval/loss_d1=0.5026` 属同分布且存在重叠风险。 | 冲刺 1 轨道 A + 冲刺 2 | 打包前确定性哈希留出（`0` 重叠）+ 冲刺 2 全检查点 OOD 重评。 |
| **`A2`** | 致命 | **已确证** | 缺少参数量匹配的 `d=1`（`40L`）LoRA 对照实验：全部循环适配器（`18.41M–156.94M`）仅在 `d >= 2` 生效，将深度循环收益与新增适配器容量混为一谈。 | 冲刺 2（`~18` GPU-h） | 在 `[12, 36)` 上以相同 `s1` 数据与调度训练 `~18.41M` `d=1` LoRA 基线；`d >= 2` 须在无泄漏留出与 OOD 集上打平或超越它方可解冻 `s4_v2_rl`。 |
| **`A3`** | 中 | **线上确证** | 量化 `NVFP4` 检查点在第 0 步不同深度即存在最高 `6.4e-4` nat 差异（`s1` 第 0 步：`d1=0.99162 / d2=0.99226 / d3=0.99183`），相对 `s2_v2@450` 的 `depth_gain=0.0062` 不可忽略。 | 冲刺 2 | 增加 2× `d=1` 非确定性底噪检测，且全部冲刺 2 评测均报告 `3` 次重复的 `mean ± std`（低于 `1e-2` 的差值须超 `3σ`）。 |
| **`B1`**（`R1 §6`） | 高 | **已确证** | 早期 README/文档将 `"~5.97B active"` / `"A6B"` 与 `3.472B–3.653B` 唯一调阅参数、`6.378B` 算力等效应用混用，且线上 `gate_abs_mean=0.022/0.058`。 | 冲刺 1 轨道 D + 冲刺 2 | 在 `README.md` 与本文 `§3` 锁定规范四轴参数账本；冲刺 2 执行路由适配器开关消融。 |
| **`B2`** | 高 | **线上确证** | [`src/qloop/model/lsa.py:3516-3547`](../src/qloop/model/lsa.py#L3516-L3547) 与 `s2_v2@450`（`a5b20b20`）：`mode="sparse"`（`s2_v2_joint_bptt.yaml:58`）下 `lsa/last_recall = 0.0` 长期停摆，无法观测稀疏检索质量。 | 冲刺 1 轨道 B + 冲刺 2 | 在 `mode="sparse"` 启用周期抽样 `last_recall`；冲刺 2 实测 `8K–1M` 稠密 vs 稀疏 `D-LSA` 差距。 |
| **`B3`** | 高 | **已确证** | 目前尚未在 `64K` 以上进行生成或 NLL 能力评测（门控 2 的 `7/7` 探针仅覆盖 `18K/64K`）；`1M`（`10.65 GiB`）与 `10M`（`53.64 GiB` 主机 DDR5）仅为显存/内存可行性证明，尚非能力证明。 | 冲刺 1 轨道 D + 冲刺 2 | 已在 `README.md` 修正表述边界（`memory-feasible`）；冲刺 2 补齐 `128K / 512K / 1M` NLL 与多针 RULER 评测。 |
| **`B4`** | 中 | **已确证** | `s3_v2@600` 头条收益（`-0.5941` nats / 困惑度降 `44.8%`）对比的是已退化失稳的 HSSD `s3@600`，而非**原版 `Qwen3.6-35B-A3B + YaRN 4×`**。 | 冲刺 2 | 冲刺 2 在相同留出与 OOD 集上将**原版 `Qwen3.6 + YaRN 4×`** 与 `s1@1000`、`s3_v2@600`、`s2_v2@600` 正面同台对比。 |
| **`B5`** | 低 | **线上确证** | 线上 `s2_v2@450`（`a5b20b20`）：`qed_lambda_abs_mean=0.00071`（上限 `0.35`）与 `neg_eigval_scale_mean=0.9998`（范围 `[0.5, 1.5]`）几乎未动，真正起作用的是基础 GDN-2 门控（`adapter_norm=0.741`、`decay_horizon_mult_max=4.545×`、`erase_write_l1=0.0419`）；且 `(0, 1.5]` 超出 GDN-2 的 `[0, 1]` 稳定区间。 | 冲刺 2 | 冲刺 2 对 `QED` 与负特征值执行开关消融，并在 `1M` 上下文下记录递推状态范数以核验有界性。 |
| **`C1`**（`QL60-009`） | 高 | **线上确证** | [`src/qloop/train/bptt.py:2045-2093`](../src/qloop/train/bptt.py#L2045-L2093)：`s2_v2@450` 上报的 `loss/lm = 3.8421` 混入了辅助 `lsa_kl`，遮蔽纯语言建模交叉熵走势。 | 冲刺 1 轨道 A | 分别返回 `ce_loss` 与 `aux_loss`；`loss/lm` 仅记录有效词元均值纯 CE，独立记录 `loss/aux`。 |
| **`C2`** | 高 | **已确证** | [`src/qloop/model/lsa.py:3522-3524`](../src/qloop/model/lsa.py#L3522-L3524)：`LSAManager.aux_loss()` 对全部 `10` 层 × 轮次执行 `.sum()`，将 KL 放大至 `23–84` 并威胁全局梯度裁剪饱和。 | 冲刺 1 轨道 B | 新增默认 `aux_reduction="mean"`（兼容 `"mean"` / `"sum"`），使 `lsa_kl` 维持 `O(1)` 量级。 |
| **`C3`** | 低 | **已确证** | [`src/qloop/model/loop.py:884-886`](../src/qloop/model/loop.py#L884-L886)：`_is_probs_cache` 依赖脆弱的首次前向数值嗅探（`_looks_like_probs(full)`），而非安装期确定性路由器类/属性识别。 | 冲刺 1 轨道 B | 改为安装期基于路由器类型与属性的确定性检测（仅未知自定义单测桩回退数值检测）并补单测。 |
| **`D1`** | 高 | **已确证** | [`src/qloop/infer/hf_custom_modeling.py:1170-9141`](../src/qloop/infer/hf_custom_modeling.py#L1170-L1218)：约 `7,900` 行独立模型代码嵌在 `_STANDALONE_MODELING_SRC` 字符串字面量内（对 ruff/类型检查器不可见），且第 `1208` 行（`_HAS_QLOOP`）依据 `qloop` 是否可导入切换行为。 | 冲刺 1 轨道 C + 冲刺 1 后 | 轨道 C 修复 `.generate()`/`logits_to_keep=0` 并写入 `qloop_commit` 元数据；冲刺 1 后将字符串抽取为受 lint 检查的独立模块。 |
| **`D2`** | 中 | **已确证** | [`src/qloop/model/hssd.py`](../src/qloop/model/hssd.py)（`13,255` 行）及已退役 v1 阶段配置与洁净 v2 并存，增加维护负担与认知混淆。 | 冲刺 1 后 | 冲刺 1 后将退役 HSSD 模块与 v1 配置归档至 `qloop/legacy/`，同时保留负向消融复现单测。 |
| **`D3`** | 中 | **已确证** | `README.md` 仍以已退役 `Loop-HSSD v3.1`（`[8, 32)`、`~5.97B`）开篇，且 [`configs/base.yaml:34-37`](../configs/base.yaml#L34-L37) 残留称 `GDN-2` `"retired"` 的过期注释。 | **冲刺 1 轨道 D（已完成）** | 已按洁净 v2 与规范参数台账重写 `README.md`；修正 `configs/base.yaml` 过期注释且不改任何功能键。 |
| **`D4`** | 低 | **已确证** | [`tests/test_experiment_report.py:1125-1190`](../tests/test_experiment_report.py#L1125-L1190)：在单元测试中断言固定报告文案子串（`"67.194"`、`"38/38"`）。 | 冲刺 1 后 | 冲刺 1 后将固定报告文案断言拆入独立 `make docs-check` 目标，使 `make check` 专注纯代码不变量。 |
| **`D5`** | 中 | **已确证** | `report.py:40`、`scripts/build_sirsi_dataset.py:7,48-49`、`docs/DATASET_AUDIT_REPORT.md:4,181,186,210`、`docs/IMPLEMENTATION_PLAN.md:3` 含硬编码本地 macOS 家目录路径（`<user-home>/...`）；`pyproject.toml` `[dev]` 缺 `pyarrow`；训练配置（`s0`、`s1`、`s2_v2`）含第三方 `opus46_*` Claude 蒸馏集。 | **冲刺 1 轨道 C 与 D** | 轨道 C 清理代码硬编码路径并在 `[dev]` 添加 `pyarrow>=16.0`；轨道 D 清理文档路径并确立具有约束力的 `§5` 数据溯源政策。 |

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
| | `s2_v2@450`（`U=14.96`，`1.522×` 倍率） | **`3.472B`** | `1.9396B` + `(128 + 24 × 14.96) × 3.1457M = 1.5321B` 唯一路由专家（`3.4717B`）。 |
| | `s1@1000`（`U=17.36`） | **`3.653B`** | `1.9396B` + `(128 + 24 × 17.36) × 3.1457M = 1.7133B` 唯一路由专家（`3.6529B`）。 |
| | `R=3` 理论上限（`U=24`） | **`4.154B`** | `1.9396B` + `(128 + 24 × 24) × 3.1457M = 2.2146B` 唯一路由专家（`4.1542B`）。 |
| **3. 算力等效参数应用量**（含非绑定 embed+head） | `40L`（`d=1`，`R=1`） | **`3.454B`** | 单轮前向（不含输入嵌入 `2.946B`；不含 embed/head 预填充 GEMM 为 `2.436B`）。 |
| | `64L`（`d=2`，`R=2`） | **`4.916B`** | `+1` 轮核心（不含输入嵌入 `4.409B`；`kv_share=True` 跳过 `6` 个核心 `k/v` 省 `0.0126B` $\to$ 预填充 GEMM 为 `3.885B`）。 |
| | `88L`（`d=3`，`R=3`） | **`6.378B`** | `+2` 轮核心（不含输入嵌入 `5.871B`）。**重要警示：** 算力等效应用量不等于有效推理增益——线上 `s2_v2@450` 实测退出门控幅值仅为 **$\lvert g \rvert = 0.022\text{–}0.058$**（`gate_abs_mean`）。 |
| **4. 可训练增量参数** | `s1_loop_a6b` / `s3_v2` / `s2_v2` | **`18.41M` / `24.56M` / `156.94M`** | 分别仅占解码器存储总参的 `0.053%` / `0.071%` / `0.453%`（`0` 个路由 MoE 参数被修改）。 |

**统一披露规范：** 今后全部文档、论文草稿与模型卡均须将**每词元唯一调阅参数（`2.947B → 3.472B–3.653B`，理论上限 `4.154B`）**与**算力等效参数应用量（`3.454B / 4.916B / 6.378B`）**明确区分，且引用算力等效指标时必须同步披露实测退出门控幅值（`|g|`）与参数匹配 `d=1` 对照结果（`A2`）。

---

## 4. `depth_gain = 0.0062` 与 `D-LSA` 召回失明的文献归因诊断

线上 `s2_v2@450` 遥测（`a5b20b20`）揭示了 `eval/depth_gain` 压缩至 `+0.0062`（`d1=0.5026 → d2=0.4999 → d3=0.4965`）及 `lsa/last_recall` 恒为 `0.0` 的深层机理。结合最新循环深度、稀疏注意力与线性注意力文献，其成因可归纳为四重叠加机制：

1. **改造式循环的捷径坍塌（`McLeish et al.` [14]、`MoR` [15]、`Recurrent Depth` [16]、`mHC` [8]）：**
   - 在 `s2_v2@450` 中，`mHC` `ReEntryBridge` [8] 在第 1 轮前奏锚点上保留了高达 **`bridge_w_anchor = 0.831 / 0.866`** 的权重，而分配给递推隐状态的权重仅为 **`bridge_w_h_prev = 0.057 / 0.050`**，同时 `PassExitGate` 处于近零区间（**`gate_abs_mean = 0.022 / 0.058`**）。跨轮专家路由呈现 **`router_repel = -0.047`** 及 **`Jaccard p12 = 0.080` vs `p13 = 0.438`**——第 2 轮略有分化，但第 3 轮重新滑回第 1 轮锚点吸引域。
   - 正如 *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence*（`McLeish et al., arXiv:2511.07384` [14]）所剖析的，当预训练前馈模型被改造为深度循环架构且保留强输入锚点注入时，优化极易陷入**捷径局部极小**：模型保住已高度优化的 `d=1` 前向通路，将第 `2–3` 轮压制成微小残差扰动（`|g| ~ 0.02–0.06`），除非借助循环深度课程与更弱锚点约束迫使隐状态发生真实迭代精炼。冲刺 2 将与 `A2` `d=1` LoRA 对照同步测试低锚点方案（`w_anchor <= 0.3`）。
2. **原生循环语言模型预训练规模鸿沟（`Ouro` [17] 与 `Inkling 2 Mini` [11]）：**
   - 原生循环预训练体系如 **Ouro**（`arXiv:2510.25741` [17]）在 **`7.7 万亿`（`7.7T`）词元**上从零训练共享循环模块并配以熵正则自适应深度目标，使迭代隐空间推理能力内化进全部权重矩阵；`Recurrent Depth` [16] 亦在 `800B` 词元上预训练，`MoR` [15] 则端到端训练词元级自适应深度路由。
   - 相比之下，本改造方案仅在冻结 `32.212B` 单轮专家之上，用 `~118M–131M` 词元（较 Ouro [17] 少约 `60,000×`！）更新 `0.053%–0.453%` 参数。在此悬殊尺度下，若缺乏参数匹配的 `d=1` LoRA 基线（`A2`）与 `3` 次重复噪声底线（`A3`），单凭未划分留出集上 `+0.0062` 的 NLL 差值宣称“真实迭代推理”在科学上站不住脚——这正是冲刺 2 对照门控的根本依据。
3. **原始 `D-LSA` 蒸馏目标与当前桶质量 KL 及稀疏盲区之差异（`Original LSA` [6]）：**
   - 原始 `D-LSA`（`arXiv:2610.42147` [6]）采用逐键剩余桶（per-key rest-bucket）KL 蒸馏并配显式温度标定（`calibrate_tau`），而当前 [`src/qloop/model/lsa.py:1231-1268`](../src/qloop/model/lsa.py#L1231-L1268) 在固定 `temperature = 0.1` 下优化聚合桶质量 KL，且对 `10` 层 × 轮次执行求和（`lsa.py:3522-3524`，`C2`）。
   - 更严峻的是，`s2_v2_joint_bptt`（`mode="sparse"`）中 `lsa/last_recall` 从不采样（`step 450` 恒为 `0.0`，`B2`），导致 `4×32K` BPTT 期间稀疏检索质量处于盲飞状态。冲刺 1 轨道 B 即刻修复均值归约 `aux_loss()` 与稀疏模式 `last_recall` 遥测；冲刺 2 将跨 `8K–1M` 评测稠密 vs 稀疏 `D-LSA` 差距并验证逐键剩余桶 KL + `calibrate_tau`，之后才讨论 `LSA-NL` 扩张。
4. **线上 `QL60-004` `step 340` 学习率重置与 `B5` `GDN-2` 扩展项利用率：**
   - `s2_v2_joint_bptt` 在 `step 340` 后于新 Colab 会话从仅权重 Hub 提交（`has_optim = false`）恢复时，未能将 `self.opt.step_count` 同步至 `self.step`，导致 `step 450/600` 时 `lr_factor` 跳回 **`0.9738`**（`optim/lr_factor = 0.9747`）。叠加未按有效词元归一化的 `/ n_segs` BPTT 缩放与混入 KL 的 `loss/lm = 3.8421`（`QL60-009` / `C1`），后期余弦退火轨迹被打断（`cec24a8d@400` 与 `a5b20b20@450` 的 `depth_gain` 停滞于相同的 `0.0062`）。
   - 与此同时，`GDN-2` [3] 基础解耦通道门控（`adapter_norm = 0.741`、`erase_write_l1 = 0.0419`、`decay_horizon_mult_max = 4.545×`）已明确唤醒，而有界 `QED` [4]（`qed_lambda_abs_mean = 0.00071` vs `0.35` 上限）与软负特征值 [5]（`neg_eigval_scale_mean = 0.9998`）贡献微弱（`B5`）。鉴于 `(0, 1.5]` 超出 GDN-2 的 `[0, 1]` 稳定界，冲刺 2 将直接消融这两项扩展并记录 `1M` 上下文状态范数。

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

### 6.1 冲刺 1 —— CPU 工程、正确性与卫生修复（`0` GPU-h）

| 轨道 / 分支 | 关闭发现项 | 严格归属文件范围 | 验收门控（CPU CI） |
| :--- | :---: | :--- | :--- |
| **轨道 A**（`fix/sprint1-eval-bptt-resume`） | `QL60-003`（`A1`）、`QL60-009`（`C1`）、`QL60-004`、`QL60-010` | `mixture.py`、`trainer.py`、`bptt.py`、`checkpoint.py`、`optim.py`、`runners.py`、`tests/test_sprint1_train_data_resume.py` | 8 文档留出复现精确 `0` 重叠；`1 vs 3` 有效词元 BPTT 梯度一致且分离纯 `loss/lm` 与 `loss/aux`；`self.rng` 精确往返及仅权重 `lr_factor` 同步；跨模型评测缓存拒绝复用。 |
| **轨道 B**（`fix/sprint1-lsa-correctness`） | `QL60-007`、`QL60-008`、`C2`、`B2`、`C3` | `lsa.py`、`loop.py`、`tests/test_sprint1_lsa_loop.py` | 左/右填充与分块对角掩码 `< 1e-6`（保留 `None` 零开销快路径）；`kv_share=False` 与 `True` float64 `< 1e-12`；均值归约 `aux_loss()`；`mode="sparse"` 非零 `last_recall`；安装期确定性路由器识别。 |
| **轨道 C**（`fix/sprint1-hf-serving-ops`） | `QL60-001`、`QL60-002`、`QL60-005`、`QL60-006`、`D5`（代码） | `engine.py`、`hf_custom_modeling.py`、`export_pack.py`、`colab.py`、`colab_supervisor.py`、`report.py`、`build_sirsi_dataset.py`、`pyproject.toml`、`tests/test_sprint1_hf_serving_ops.py` | `QuantizedHybridCache` 通过模拟 TF `5.24+` 协议；`CausalLMPackOutput` + 虚拟 `layer_types` + `logits_to_keep=0` `(B, T, V)` + `qloop_commit` 元数据；暂停态 `step000001` 返回 `pack_shipped=False`；默认切至洁净 v2 阶段；可移植路径与 `[dev]` 增加 `pyarrow>=16.0`。 |
| **轨道 D**（`docs/midway-review-and-cleanup`，*本 PR*） | `B1`、`D3`、`D5`（文档与政策） | `MIDWAY_REVIEW_AND_ACTION_PLAN{,_ZH}.md`、`README.md`、`DATASET_AUDIT_REPORT.md`、`IMPLEMENTATION_PLAN.md`、`configs/base.yaml` | `validate_ieee_citations()` 严格 `1..20`（`EN == ZH`，`0` 范围连字符）；严格 `GDDR7 ECC` 硬件口径；`0` 硬编码用户路径；`0` 失效链接；`test_experiment_report.py`、`test_configs_cli.py`、AST 与 ruff 全绿。 |
| **冲刺 1 后清理项** | `D1`、`D2`、`D4` | 将 `7.9K` 行独立代码字符串抽为受 lint 模块；将 `hssd.py`（`13,255` 行）与 v1 配置归档至 `qloop/legacy/`；将文案断言拆入 `make docs-check`。 | `make check` 与 `make docs-check` 职责清晰分离，v2 与归档测试零回归。 |

### 6.2 冲刺 2 —— 决定性单卡科学验证（`~25–35` GPU-h）

严格在**冲刺 1 全部合入后**于单张 `NVIDIA RTX PRO 6000 Blackwell Server Edition`（`96 GB GDDR7 ECC` [`94.97 GiB` 可用]，`1,597 GB/s`，`sm_120`）执行：

| 工作流 | 算力预算 | 闭环发现项 | 精确实验协议与可量化通过门控 |
| :--- | :---: | :---: | :--- |
| **S2.1 无泄漏留出 + OOD 与 `8K–1M` 能力审计** | `~10–14` GPU-h | `A1`、`A3`、`B2`、`B3`、`B4` | 在 `8K / 64K / 128K / 512K / 1M` 上对**原版 `Qwen3.6 + YaRN 4×`** [1], [9]、`s1@1000`、`s3_v2@600`、`s2_v2@600` 执行 `d ∈ {1, 2, 3}`（`3` 次重复 + 2× `d=1` 非确定性检测以闭环 `A3`）、稠密 vs 稀疏 `D-LSA`、多针 **RULER** [20] 与 `1M` GDN-2 状态范数日志。**门控：** 相对原版 `Qwen3.6 + YaRN 4×` 取得 `> 3σ` 提升且至 `1M` 稀疏 `D-LSA` 差距可控。 |
| **S2.2 参数量匹配 `d=1` LoRA 对照实验** | `~18` GPU-h | `A2` | 在 `[12, 36)` 上以相同哈希切分 `s1` 数据、种子与调度训练与 `s1_loop_a6b` 等参（`~18.41M`）的 `d=1`（`40L`）LoRA 基线。**门控：** 循环 `d >= 2` 在无泄漏留出与 OOD 集上须打平或超越 `d=1` 对照（证明深度循环确有超越单纯适配器容量的增量收益）。 |
| **S2.3 定向因果消融 + 窄域 `K0` 预言机测试** | `~2–4` GPU-h | `B1`、`B5`、`LSA-NL` 门控 | 运行 `GDN-2`、`QED`/负特征值与 `PassRouterAdapters` 独立开关消融、低锚点（`w_anchor <= 0.3`）重入探针（检验 `McLeish et al.` [14] 捷径逃逸），以及 **`30` 分钟窄域 `K0` Pre-RoPE KV-Blob 预言机测试**（`src/qloop/memory/kv_blob.py`）。**门控：** 裁撤一切零增益组件；`K0` 须展现出显著召回恢复方可解锁任何 `LSA-NL` 训练。 |

### 6.3 明确的放行 / 终止解冻判据

- **解冻 `s4_v2_rl`（SI-RSI GRPO）：** 须同时满足**冲刺 1（`100%` 通过）** + **S2.1 通过** + **S2.2 通过**。若循环 `d >= 2` 未能跑赢等参 `d=1` LoRA 对照（`S2.2`），则继续暂停 RL，转而迭代低锚点循环深度课程（`S2.3`）或交付洁净 `d=1` 长上下文模型，绝不将 GPU 算力耗费在捷径坍塌的循环上。
- **解冻 `LSA-NL` / `CoMem`（`C2–C6`）训练：** 须满足全部 `s4_v2_rl` 前置条件**外加**通过 `30` 分钟 **`K0` 窄域 KV-Blob 预言机测试**（`S2.3`）。
- **解冻公开完整模型包发布：** 须通过上述全部门控，在真实 `transformers 5.24+` 下验证 `.generate()`（`QL60-001`、`QL60-002`、`QL60-005`），并确保 **100% 剔除 `opus46_*` 受限溯源数据集**（`§5`）。

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
