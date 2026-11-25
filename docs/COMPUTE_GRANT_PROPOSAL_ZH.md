# 多卡算力申请与实验室科研合作提案（`v2.0`）：`Qwen3.6-64L-CoMem`（`LSA-NL`）
### 超越静态循环 MoE：冻结 NVFP4 基座上的五时间尺度终身记忆、三轴动态算力与无损中途草稿解码

> **公开模型仓库：** [`TassP0lar/Qwen3.6-35B-A3B-Loop64`](https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64)（`202+` 次实时提交） · **公开数据集：** [`TassP0lar/Loop64-Fable-SIRSI-v1`](https://huggingface.co/datasets/TassP0lar/Loop64-Fable-SIRSI-v1)（`v1.2`，`cc-by-4.0`）  
> **单卡研发硬件：** `1× NVIDIA RTX PRO 6000 Blackwell Server Edition`（`GB202`，`sm_120`，**`96 GB GDDR7 with ECC`** [`94.97 GiB` CUDA 可用]，`512-bit` 总线，**`1,597 GB/s`** 服务器版峰值带宽，`24,064` CUDA 核心，`600 W` 最大 TDP，被动散热）  
> **配套科研与技术文档：** 旗舰研究提案 [`docs/PROPOSAL_LSA_NESTED_LEARNING_ZH.md`](PROPOSAL_LSA_NESTED_LEARNING_ZH.md) / [`docs/PROPOSAL_LSA_NESTED_LEARNING.md`](PROPOSAL_LSA_NESTED_LEARNING.md)（`v2.0`） · 英文算力提案 [`docs/COMPUTE_GRANT_PROPOSAL.md`](COMPUTE_GRANT_PROPOSAL.md) · 路线图 [`docs/ROADMAP_NEXTGEN_LOOP64_COMEM.md`](ROADMAP_NEXTGEN_LOOP64_COMEM.md) · 实证报告 [`docs/EXPERIMENT_REPORT_ZH.md`](EXPERIMENT_REPORT_ZH.md)（`Loop64_Technical_Report_ZH.pdf`）与 [`docs/EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md)（`Loop64_Technical_Report_EN.pdf`） · 原始 CSV/JSON 遥测及 7 张纯矢量 SVG 图表位于 [`docs/experiment_logs/`](experiment_logs/)  
> **代码库就绪度：** 开源主线（`origin/main` `574898d+`），**`760` 项单元 / 等价性 / CUDA 测试**（含 **`38/38` Blackwell CUDA GDN-2 pytest** 全绿），`0` lint / Python 3.9 AST 错误  
> **核心算力申请：** **`~2,500` GPU 小时**（`1× 8×` H200 / B200 / RTX PRO 6000 节点运行约 `~2` 周；支持按需伸缩：第 1 档 `2–4×` GPU 仅需 **`~400` GPU 小时**，第 3 档前沿扩展至 **`~6,000` GPU 小时**）

---

## 1. 核心科研立意：超越静态循环 MoE（`Inkling 2 Mini`）

仅依靠拉长思维链（CoT）扩展测试时推理算力，会迅速遭遇 KV 缓存容量墙与串行词元解码延迟瓶颈；而从零预训练循环架构或终身记忆大模型则需耗费工业级预训练预算。在过去 **`167.60+` 小时单卡有效监控运行**中，我们依托单张租用 **`NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7 ECC, 1,597 GB/s)`** 已实证表明：面对量化 `35B`（`A3B`）NVFP4 混合 MoE 基座（`unsloth/Qwen3.6-35B-A3B-NVFP4` [1]，由 `30` 层 Gated DeltaNet [2] 与 `10` 层全局注意力交替构成，`32.212B` 路由专家 100% 冻结且经 SHA-256 校验），仅需 `< 0.6%` 可训练参数配合低秩 **APOLLO** [3]（`arXiv:2412.05270`）优化器，即可将其无损改造为运行时可扩展的 `40L / 64L / 88L` 循环推理模型，并将 `1M` 词元 FP8 状态压至 **`10.65 GiB`**（较 `Inkling 2 Mini` [4] 的 `32.16 GiB` F16 KV 缩小 **`3.02×`**）。

本提案面向高校课题组合作与多卡算力支持，**明确不以复刻 `Inkling 2 Mini`（静态 `48L` 循环深度、固定单词元算力、扁平 `1M` KV）为目标**。相反，我们提出 **`Qwen3.6-64L-CoMem`（`LSA-NL` v2.0）**——在已完成单卡去风险验证的循环 `GDN-2-1M + QED` 主干 [5], [6], [7], [8] 之上，构建五项高度差异化、可支撑顶会发表的正交研究支柱（完整数学证明与公式账本见 [`PROPOSAL_LSA_NESTED_LEARNING_ZH.md`](PROPOSAL_LSA_NESTED_LEARNING_ZH.md)）：

1. **`LSA-NL` 五时间尺度终身记忆（`L0 → L4`）与 `10B` 词元遗忘/晋升漏斗：** 将 *Nested Learning* 的连续记忆系统 [9] 与非参数逐字精确记忆深度融合，彻底免去从零预训练并化解前摄干扰 [10]。以**`3M–4M` 分域逻辑上下文**替代扁平 `10M` 窗口（`1M–2M` 活跃 A 分区供 `98.25 MiB` `GDN-2` 纯净隔离运行 + `1M–2M` GPU 挂载 B 分区 Pre-RoPE `H64-INT4` KV-Mem Blob [11], [12], [13], [14]，底层依托 `262.5 MiB` GPU 头部索引 `100M+` 冷 SSD 词元）。**四阶段终身漏斗**（`10B → 800M → 133M → 80M → 60M` 稳态 = **`345.6 GB` SSD + `150.2 MiB` GPU 头部**，取代不可行的 `57.60 TB` 原始归档）协同融合显著度精选、`EpiCache` [15] `192`-D NoPE 语义聚类压缩、`SleepGate` [10] / `Zep` [16] 双时间墓碑 + 艾宾浩斯 ACT-R 间隔重复硬化（$R_b(t)=e^{-\Delta t/S_b}$，$S_b \leftarrow 2S_b$）以及**三阶段纵向 KV-to-Engram 晋升**（`0` 基座重训练）：*阶段 A* `Spark-X2.6` [17] `AlphaEdit` [18] + `WISE` [19] `144 MiB` 闭式零空间 `ESM`（`0.0` 漂移）；*阶段 B* 模块化 `2K–4K` `Cartridges` / `Cartridges++` / `Cartridges at Scale` [20], [21], [22]（`~11–22 MiB`，`~38×` 压缩）；*阶段 C* 单次前向亚秒级（`~0.21 s`，适配器显存 `<50 MB`）**`Doc-to-LoRA`** [23] 超网络 + 按周 **`SMF`** [24] `AnchoredMoEEngram` [25] + **`SEAL`** [26] 自编辑指令。
2. **面向原始 `Latent Sparse Attention` [27]（`arXiv:2610.42147`）的六项外科手术式升级：** 将 `src/qloop/model/lsa.py` 中的桶级质量 KL 升级为原始 LSA 的**逐键前向 KL + 截断深度“剩余桶” + `calibrate_tau`**；以可变双时间 `DLMG`（`HippoRAG 2` PPR [28] + `A-MEM` [29]）替代只增不减静态树；引入 Pre-RoPE `H64-INT4` + 限幅 ReRoPE 实现零重预填充跨会话挂载；将 `BucketIndex` 压缩为 `int8`（净省 `-1.0 GiB`）；保留每轮解耦 `W_q^{(p)}`（`num_query_passes = 2–3`）；打通横向 KV 向纵向 Engram 的晋升通路。
3. **三轴动态弹性计算（`MoR`）、端到端预填充加速与 Engram 耦合无损 `L35` 中途草稿解码：** 正交组合**轴 1** 词元级动态深度路由（**`MoR`** [30] + *Recurrent Depth* [31]，已在 `src/qloop/model/loop.py:548-554,771-786` 搭好 `GDN-2` 写掩码骨架：简单词元 `40L` / 常规 `64L` / 顶部 `5–10%` 困难词元爆发至 `88L`）、**轴 2** 弹性激活参数（经原生 `top_k` `8↔16` + `s_null` Matryoshka `MoE-Engram` 实现 `3.45B ↔ 6.38B+` 动态伸缩），以及**轴 3** **端到端预填充 + 无损解码双加速体系**：
   - *计算受限预填充（`1.23×–1.39×` 加速 `64L` 预填充 + 零回归会话重预填充）：* 投机草稿加速带宽受限的解码而非计算受限的预填充（`40L` 为 `2.436B`，`kv_share=True` `64L` 为 `3.885B` GEMM 参数/词元 = `1.595×`）。因此我们将 `MoR` [30] + `Mixture-of-Depths` [32] 容量因子第 2 轮词元跳过（带 `GDN-2` 写掩码仅保留 `50%/35%/25%` 显著词元 → `3.160B/2.943B/2.798B` = **较完整 `64L` 预填充快 `1.23×–1.39×`**，仅 `40L` 的 `1.15×–1.30×`）与零额外模型**第 1 轮显著度引导 `Speculative Prefill`** [33], [34] + `D-LSA` [27]（`10M` 单注意力层 `22.7×–72.4×`）叠加，同时 B 分区 KV-Mem Blob（`0.059–0.421 ms/blob`）彻底消除回归会话第 0 层重预填充。
   - *带宽受限解码（裸 `64L` 提速 `1.58×–1.76×`；**`64L + 全量 MoE-Engram [5.86B]` 提速 `1.73×–2.28×` —— 反超 `40L`！**）：* 将 **`L35` `EAGLE-3`** [35] 多层特征与 $O(1)$ Tier-1A/1B N-gram Engram 嵌入（$[h_{L11}, h_{L23}, h_{L35}, e_{\text{ngram}}]$，`<0.02 ms`，起草期跳过路由 `MoE-Engram` 读取）融合，叠加直接词元 TTT、**`LoopSpec`** [36] 选择性二次提议、**`SAM-Decoding`** [37] $O(1)$ 代码库后缀匹配及 **`TreeWY`** [38]（`O(1)` 单状态块/层 `GDN-2` 验证、峰值状态显存降低 `2–3×`，配 `12 KiB/锚点` `O(1)` `conv1d` 回滚及 `γ=4–5` 批量前瞻预取）。诚实 MoE Zipf 赠券收集器数学（`u_MoE = 2.83×–3.96×`，`E=32, K=7` 上 `u_Eng = 2.21×–2.65×`，`0.085 GB` 共享 Engram 仅读一次）表明完整 `40L` 起草相对 `64L` 仅 `1.03×–1.30×`（慢于 `40L`），而本方案相对裸 `64L` 达 **`1.58×–1.76×`（快于 `40L` `1.07×–1.19×`）**，在 **`64L + 全量 Spark-X MoE-Engram（5.86B 激活，4.131 GB/词元）`** 上更达 **`1.73×–1.90×`（方案 B，`0.55 GB` 草稿；`α=0.90–0.92` 下反超 `40L` `1.02×–1.12×`）** 与 **`1.84×–2.28×`（方案 B+，`32K` 短名单 `0.195 GB` 草稿；`α=0.85–0.92` 全面反超 `40L` `1.09×–1.35×`）**——**100% 无损 `64L + MoE-Engram` 输出！**
4. **消费级 `<= 24 GB` 显存普惠部署方案：** 因 `Qwen3.6` 全部稠密核心仅 **`2.448B`（FP8/Q8 `2.42 GiB`）**而 `32.212B`（`92.93%`）为路由 MoE，结合 Zipf NVFP4 冷热切分（`KTransformers` [39]、`MoE-Lightning` [40]、`Fiddler` [41]）与 `H64-INT4` KV，可使*完全相同*的 `64L` + `100M` SSD 智能体落地于 **`RTX 3090/4090`（`24 GB`，仅占 `17.66–18.93 GiB`）**、**`16 GB` `4080/5070 Ti`（`10.89 GiB`）** 以及 **`32 GB` `RTX 5090`（`26.98 GiB` 全 GPU NVFP4 + 完整 `1M` KV）**。
5. **长期第 5 阶段原生多模态 KV-Mem Blob：** `Qwen3.6-35B-A3B` 本身原生支持 `image-text-to-text`（`3D mRoPE [11, 11, 10]`，`partial_rotary_factor = 0.25`），待文本/代码第 `0–4` 阶段验证完毕后，视觉/UI/文档/视频词元可直接复用完全同构的 `192`-D NoPE 键（`ReKV` [42]、`M3-Agent` [43]、`M+` [44]）。

---

## 2. 已验证的单卡工程实证（为何本项目风险已充分出清）

### 2.1 核心使能技术：单张 `96 GB` 卡上的 APOLLO（`arXiv:2412.05270`）[3] + `FusedStepOptimizer`

在不依赖 ZeRO/FSDP 多卡并行的条件下，于单张 **`96 GB GDDR7 ECC`** 显卡上完成 `34.661B` 循环 MoE 高达 `131K–1M` 词元的训练，关键在于将低秩 **APOLLO** [3]（`arXiv:2412.05270`）深度集成于 **`FusedStepOptimizer`**（`configs/base.yaml:89-97`、`src/qloop/train/optim.py:382-421`）：
- **精确路由规则：** 通过 `apollo_torch.APOLLOAdamW` 配置 `apollo: {enabled: true, rank: 256, scale_type: channel, scale: 1.0, update_proj_gap: 200, min_dim: 128}`，将所有大型二维可训练矩阵（`min(shape) >= 128`）路由至 `rank=256`、近似 SGD 显存占用的低秩 APOLLO，而一维归一化层与 `<128` 维窄适配器保留精确 `AdamW`/`AdamWLite`。
- **分阶段运行时实绩：** `s1_loop_a6b`（`137` 张量，`18.41M` 参数）采用反向内即时步进 `fused=True`；自提交 `3a17c0d` 起，`s1b` / `s3*` / `s2*` / `s4*` 切换至 `fused=False` 以启用整步原子非有限值护栏与全局 `clip_grad_norm_`，同时保持 `apollo.enabled: true`。在 **`s2_v2_joint_bptt`**（`156.94M` / `387` 注册张量，Gate 3 实际步进 `379` 个）中，APOLLO 覆盖全部 `30` 个大型二维全局注意力矩阵（`104.86M` 参数：仅计矩状态占 `0.156 GiB` [`5.0×` 压缩，省 `-0.625 GiB`]，含 fp32 投影共 `0.186 GiB` [净省 `4.2×`]，对比全 AdamW 的 `0.781 GiB`）；在 `s4_v2_rl` 中 APOLLO 覆盖 `209.72M` 参数（净省 `2.50×`，`-0.94 GiB`）。配合 Blackwell 向量化优化（`_foreach_norm(..., dtype=fp32)`、单遍 `_foreach_mul_` 梯度裁剪、批量 `AdamWLite`），彻底消除优化器核发射瓶颈。

### 2.2 已去风险的实证证据链（`167.60+` 单卡有效 GPU 小时，`627.56M+` 训练词元）

全部架构、Triton 核函数、显存余量、数值稳定性与数据卫生风险均已在 **`1× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7 ECC, 1,597 GB/s)`** 上逐一攻克。下表严格区分 `实测（Measured）`、`闭式推导（Derived Formula）` 与 `估算（Estimated）`，绝不宣称任何未测下游 2026 基准得分。

| 研究支柱 | 类别 | 已验证实证结果（单张租用 Blackwell 服务器 GPU） | 可复现溯源产物 |
| :--- | :---: | :--- | :--- |
| **零训练校准（`s0_probe`）** | `实测` | `177.44 s` 内通过 **`5/5`** 探针门；专家重路由余量 `1.3061` nats（`88.9%` 困难词元）；`[12,36)` 核心将零样本冲击压低 `3.26×`；纯 `YaRN 4.0×` [45] 胜出 `1M` 外推（`100%` 针命中） | `probe_report.json` (`058b4ad3`) |
| **88L 循环愈合（`s1_loop_a6b@1000`）** | `实测` | **`18.41M` 参数（`0.053%`）**，`0` 跳过，峰值 `54.24 GiB`；`8K` NLL `0.9800 → 0.9191`（`depth_gain = +0.0609`）；第 2 轮专家重路由 **`82.27%`**（`17.36` 唯一专家/词元）；`mHC` 桥接 [46]；`16K` 针检索 **通过** | Hub `32eb565e` [47] |
| **洁净 v2 A/B 完胜（`s3_v2@600`）与 Gate 2** | `实测` | **`600/600` 步完工（`15.02 h`，`24.56M` 参数，`0` 跳过）**：留出 `64K` NLL **`1.3689 → 1.2996 → 1.2816`**（`depth_gain = +0.0873`，较 HSSD `s3@600` `1.8757` 暴降 **`-0.5941` nats / 困惑度低 `44.8%`**）；`512K–1M` 召回峰值 `0.6514` / 终值 `0.5715`；**`2,365 tok/s`**（`2.25×` 提速）；峰值 **`69.96 GiB`**（`s125` 实时 **`1,048,576` 词元**前向仅 `69.52 GiB`）。**Gate 2：** **`12/12` 全部清晰输出**，**`7/7`（`100%`）`18K`/`64K` 大海捞针精准命中**，`0` 复读，`0.0` 毒化 | Hub `6b521f47`, `ec34f825`, `s3v2_gate_summary.json` |
| **Gate 1 + Gate 3 + 在线 `s2_v2`（`GDN-2-1M + QED`）** | `实测` | **`27.53M` GDN-2 参数（`0.079%`）**，第 0 步 `0.0` 恒等，`0` 额外状态字节。**Gate 1（`574898d`）：** **`38/38` CUDA pytest 全绿**；融合 `tl.int64` Triton 预处理 **`53.55 ms`**（`2.00×`）且 `T=1` **`0.0787 ms`**（`2.41×`）；检查点 `4×8K` 宏分段将包装层显存削减 **`3.28×`**（**`23.051 → 7.032 GiB`**）。**Gate 3（`4×32K`, `d=3`）：** 峰值 **`67.194 GiB`**。**在线 `s2_v2`：** 已归档 `s100/600`（`d0a1d1d0`，`1,040–1,173 tok/s`，`67.77 GiB`，`0` 跳过；`horizon_mult` `1.014× → 1.544×`），当前正在受控 Blackwell 守护进程上**稳定推进并突破 `step 340/600`** 且 `0` 跳过 | `s3v2_gate_summary.json`, `s2_v2_joint_bptt.csv`, Hub `d0a1d1d0`, `574898d` |
| **`1M` 推理与训练显存账本** | `推导` + `实测` | **`1M` 推理：** **`10.65 GiB`** FP8 SharedKV（较 Inkling 2 Mini [4] `32.16 GiB` 省 **`3.02×`**；`H64-INT4` [11] 仅 `5.96–6.28 GiB`）；`lsa.py:743-768` `BucketIndex` 即席 `int64` 表额外占 **`+1.04 GiB`**（总计 `~33.54 GiB`，压缩至 `~0.04 GiB` 可净省 `-1.0 GiB`）。**训练峰值（`96 GB` GDDR7 ECC）：** `54.24`（`s1`）、`69.96`（`s3_v2`）、**`67.19–67.77 GiB`**（在线 `s2_v2`，留余 **`27.2 GiB`**） | `src/qloop/model/lsa.py:743-768`, `EXPERIMENT_REPORT_ZH.md` §II |
| **审计去污染语料（`v1.2`）与 Blackwell `sm_120` 核函数** | `实测` | `4,081` 条 `sft_core` + `429` 偏好对 [48], [49], [50], [51]；`1,889,432` 条 2026 实时 13-gram（跨 12 套基准拦截 `31` 条泄漏）；解耦 `working_notebook` CoT 将通过率由 `4.1%` 提至 `100.0%`（`197/197`）。**Blackwell `sm_120`：** **`18.99×`** NVFP4 E2M1 达 **`1,148 GB/s = 71.9%`** `1,597 GB/s` GDDR7 峰值带宽、**`9.32×/12.99×`** D-LSA FlashAttn、**`5.83×`** 两遍 KL、**`54.03×`** BPTT DMA+CCE [52]，**`760` 项测试全集** | `dataset_audit_summary.json`, `docs/PERF_PROFILE_S2.md` |

### 2.3 诚实负面消融（HSSD v3.2 深度复盘）与数据取证

本团队科研方法论的核心准则之一，是将负面消融转化为架构跃升而非掩盖问题：
- **HSSD v3.2 复盘 → 借 `DPLR` [8] 转向洁净 `GDN-2-1M + QED`：** v1 自研 HSSD 模块曾暴露出 $O(T)$ 累计相位雅可比爆炸（`s475` 达 `6.12e+19`，后经 `chunk_size/T` 缩放修复）、不可逆的 **`+0.79` nat** 非恒等分布冲击（`0.919 → 1.705`）、长程记忆槽激活坍缩（`vault_duty = 1.14e-6`）、即便在 `LSA=dense` 下 `18K` 大海捞针仍失败以及 `d=3` 仅 `3.41 tok/s` 解码速度。通过强制执行**第 0 步 `0.0` 位级恒等律**、以生成式 Gate 2 大海捞针检验配对 NLL（`s3_v2@600` 达成 `12/12` 清晰与 `7/7` 针命中）并将预训练 DeltaNet 泛化至生产级 `fla 0.6.0` `DPLR` 核函数，我们收敛出洁净 **`GDN-2-1M + QED`**（第 0 步误差 `0.0`、零额外状态字节、`T=1` `0.0787 ms`、显存削减 `3.28×`）。
- **语料毒化隔离取证：** 审计定位上游 `fable51` 语料混入 `14,511` 处字面脱敏 `<search_query>` 占位符，由此推动构建全量 13-gram 基准去污染流水线（拦截 `31` 条泄漏）与 `100%` 通过率的 `working_notebook` 推理合成数据。

---

## 3. 下一阶段科研规划与多卡算力申请方案 *（`估算`）*

全部 GPU 小时预算均直接根据单卡 Blackwell 实测吞吐线性外推（`s2_v2` `4×32K` BPTT：`1,040–1,173 tok/s` ≈ 每次 `600` 步运行耗时 `40–45` GPU-h；`64L` `s4_v2_rl` GRPO/DAPO [53], [54] 采样：`~150–250` GPU-h / `400` 步，`L35`+`TreeWY` 草稿落地后可再提速 `~1.6×`；全套 16 项 2026 前沿基准 `d ∈ {1, 2, 3}` 扫描：`~150–250` GPU-h/配置）。每一算力档位均受**硬性熔断门（`K0–K4`）**约束，确保绝不在未证实假设上浪费多卡算力。

| 算力申请档位 | 总 GPU 小时 | 建议硬件配置 / 周期 | 资助实验内容、科学范围与硬性熔断门 |
| :--- | :---: | :---: | :--- |
| **第 1 档：核心闭环与 `LSA-NL` 熔断验证** | **`~400` GPU-h** | `2–4×` H200 / B200 / RTX PRO 6000（`4–7` 天） | 完成 `s2_v2@600` + `GDN-2 开 vs 关` 因果消融（`30 h`） · 执行 **`K0` Pre-RoPE KV-Blob 熔断门**（`0.5–4 h`）并将 `lsa.py` 升级为逐键剩余桶 KL + `calibrate_tau`（`40 h`） · 聚焦版 `64L` `s4_v2_rl`（`DAPO-Math` + `KernelBench`，`180 h`） · 核心 2026 深度曲线评测（`AIME 2026`、`GPQA`、`MRCR 1M`、`KernelBench L3`，`145 h`）。**熔断门：** `K0` 达神谕 `>= 92%`（或 `5%` `CacheBlend`），`lsa.py` 召回率 `> 0.75`。 |
| **第 2 档：旗舰论文 2 —— `LSA-NL` + 动态算力与中途草稿 *（核心推荐方案）*** | **`~2,500` GPU-h** | **`1× 8×` H200 / B200 / RTX PRO 6000 整机节点（约 `~2` 周）** | 完整 `s2_v2` + 多轮 `64L` `s4_v2_rl`（`SI-RSI` 编程/SWE/数学/洁净 MCP，`650 h`） · **全套 `LSA-NL` `NL-0..NL-6` 受控消融（`650 h`）：** `3M–4M` 分域 + `100M` SSD `DLMG` + `10B` 词元 ACT-R/`SleepGate` 漏斗（`K1/K2`）、`AlphaEdit`/`WISE` `ESM` + `Cartridges++`（`K3`）、`AnchoredMoEEngram` + `SMF` 五重护盾消融 · **端到端预填充（`1.23×–1.39×`）与 Engram 耦合无损解码（`450 h`）：** `MoR`/`MoD` 第 1 轮引导预填充跳过 + `40/64/88L` 写掩码路由 + Engram 条件化 `L35` `EAGLE-3`（$[h_{L11}, h_{L23}, h_{L35}, e_{\text{ngram}}]$）+ `LoopSpec` + `SAM-Decoding` + `TreeWY`（`K4`：`64L` 预填充 `>= 1.20×`、`α >= 0.85`/`0.90`、`100%` 贪婪等价、相对裸 `64L` `>= 1.45×` / 相对 `64L+Engram` `>= 1.60×` 且方案 B+ 反超 `40L`）及消费级 `<= 24 GB`（`17.66–18.93 GiB`）实测校验（`NL-7`） · **完整 16 项 2026 基准扫描**（`d ∈ {1,2,3}` + 等算力 `40L` CoT 对照，`600 h`） · 推理服务与取证缓冲（`150 h`）。 |
| **第 3 档：论文 3 前沿 —— 单前向 `Doc-to-LoRA` 与第 5 阶段多模态记忆** | **`~6,000` GPU-h** | `1–2× 8×` H200 / B200 节点（约 `~4` 周） | 包含第 2 档全部工作（`2,500 h`） · **阶段 C 单前向超网络（`1,500 h`）：** 跨 `100M` 词元代码库归档元训练亚秒级（`~0.21 s`，适配器显存 `<50 MB`）`Doc-to-LoRA` [23] KV-to-Engram 超网络 + `SEAL` [26] 自编辑强化学习 · **第 5 阶段原生多模态 KV-Mem Blob（`1,200 h`）：** 重新启用 `Qwen3.6` 内置 ViT（`3D mRoPE [11,11,10]`，`192`-D NoPE）构建流式 UI/PDF/视频情景 Blob（在 `ScreenSpot Pro`、`OmniDocBench` 与流式视频问答上对标 `ReKV` [42] / `M3-Agent` [43]） · 多种子方差区间与 `R ∈ {1..4}`（`40L–112L`）深度外推（`800 h`）。 |

---

## 4. 三篇论文阶梯发表规划、高校合作模式与首日开跑就绪度

本次算力合作清晰映射到一条**三篇递进式论文发表阶梯**，每一阶段均对应独立可交付里程碑且零算力浪费：

1. **论文 1（v1.5 实证技术报告 —— 阶段 0 完工即刻投递）：** 汇总完工 `s2_v2@600`（`40L/64L/88L`）全量遥测、`GDN-2-1M + QED` 因果消融、HSSD v3.2 负面消融复盘及 Blackwell `sm_120` Triton 内核优化（`71.9%` GDDR7 峰值、`3.28×` `4×8K` 显存削减）。现已完稿为可完全复现的中英双语 US-Letter IEEE PDF（[`Loop64_Technical_Report_EN.pdf`](papers/Loop64_Technical_Report_EN.pdf)、[`Loop64_Technical_Report_ZH.pdf`](papers/Loop64_Technical_Report_ZH.pdf)）。
2. **论文 2（多卡合作旗舰顶会主文 —— 目标 NeurIPS / ICML / ICLR / MLSys 2027）：** *《潜在稀疏嵌套学习（`LSA-NL`）：面向冻结 MoE 的分域连续记忆、动态递归算力与无损中途草稿解码》*。完整交付横跨 `L0 → L4` 的 `NL-0..NL-7` 实证研究、`10B` 词元遗忘/晋升漏斗（`AlphaEdit` + `Cartridges++` + `SMF`）、原始 LSA 六项升级、端到端 `MoR`/`MoD` 预填充（`1.23×–1.39×`）+ Engram 耦合 `L35` `EAGLE-3`/`SAM` + `TreeWY` 无损解码（`64L + 5.86B MoE-Engram` 解码反超 `40L`！）以及 `<= 24 GB` 消费级显卡部署实证。
3. **论文 3（第 3 档 / 深度联合合作 —— 前沿多模态终身记忆智能体）：** *《面向混合循环 MoE 的即时上下文-Engram 编译（`Doc-to-LoRA`）与原生多模态 KV-Mem Blob》*。将 `LSA-NL` 推进至亚秒级（`~0.21 s`，适配器显存 `<50 MB`）单次前向超网络知识内化（`Doc-to-LoRA` + `SEAL`）与基于 `Qwen3.6` 内置 VLM 视觉塔的统一视觉/文本/代码 `192`-D NoPE 情景记忆（`ReKV` + `M3-Agent`）。

- **100% 开源社区交付与首日开跑就绪度：** 全部代码、权重（`TassP0lar/Qwen3.6-35B-A3B-Loop64` [47]）、13-gram 去污染数据集（`Loop64-Fable-SIRSI-v1` [48]）、独立 Blackwell `sm_120` Triton 内核与评测框架均全面开源，并经 **`760` 项测试**（含 **`38/38` Blackwell CUDA GDN-2 测试全绿**）与整步原子非有限回滚护栏严格校验——合作算力一旦就位，**首日即可直接启动第 1 档或第 2 档训练与评测任务**，欢迎高校教授与科研实验室在联合署名、开源共建及多卡算力支持等维度展开深度合作。

---

## 参考文献

[1] Qwen Team, "Qwen3.6-35B-A3B," Alibaba Cloud, Model Release, Apr. 16, 2026. [Online]. Available: https://huggingface.co/Qwen/Qwen3.6-35B-A3B

[2] S. Yang, J. Kautz, and A. Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2412.06464.

[3] H. Zhu, Z. Zhang, W. Cong, X. Liu, S. Park, V. Chandra, B. Long, D. Z. Pan, Z. Wang, and J. Lee, "APOLLO: SGD-like Memory, AdamW-level Performance," *arXiv preprint arXiv:2412.05270*, Dec. 2024.

[4] Thinking Machines Lab, "Inkling 2 Mini," Technical Release, Nov. 20, 2026. [Online]. Available: https://thinkingmachines.ai

[5] A. Hatamizadeh, Y. Choi, and J. Kautz, "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention," *arXiv preprint arXiv:2605.22791*, May 2026.

[6] D. Gupta, A. Das, and D. Gupta, "The Query Knows What to Forget: A Second Erase Direction for Linear Attention," *arXiv preprint arXiv:2608.13668*, Aug. 2026.

[7] R. Grazzi, J. Siems, A. Zela, J. K. H. Franke, F. Hutter, and M. Pontil, "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2411.12537.

[8] S. Yang and Y. Zhang, "FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanisms," GitHub Repository (`fla-org/flash-linear-attention`, v0.6.0), 2024–2026.

[9] A. Behrouz, M. Razaviyayn, P. Zhong, and V. Mirrokni, "Nested Learning: The Illusion of Deep Learning Architectures," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2025, arXiv:2512.24695.

[10] Y. Xie, "Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in Large Language Models (SleepGate)," *arXiv preprint arXiv:2603.14517*, Mar. 2026.

[11] A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni, "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2026, arXiv:2504.19874.

[12] J. Hu, W. Huang, W. Wang, H. Wang, T. Hu, Q. Zhang, H. Feng, X. Chen, Y. Shan, and T. Xie, "EPIC: Efficient Position-Independent Caching for Serving Large Language Models," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2025, arXiv:2410.15332.

[13] J. Yao, H. Li, Y. Liu, S. Ray, Y. Cheng, Q. Zhang, K. Du, S. Lu, and J. Jiang, "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion," in *Proc. ACM Eur. Conf. Comput. Syst. (EuroSys)*, 2025, arXiv:2405.16444.

[14] D. Ma, Y. Wang, and T. Lan, "Block-Attention for Efficient Prefilling," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2409.15355.

[15] M. Kim, A. Kundu, H.-B. Kim, R. Dixit, and M. Cho, "EpiCache: Episodic KV Cache Management for Long Conversational Question Answering," *arXiv preprint arXiv:2509.17396*, Sep. 2025.

[16] P. Rasmussen, P. Paliychuk, T. Beauvais, J. Ryan, and D. Chalef, "Zep: A Temporal Knowledge Graph Architecture for Agent Memory (Graphiti)," *arXiv preprint arXiv:2501.13956*, Jan. 2025.

[17] BaiTian6641, "Spark-X2.6-Recursion: Dynamic Latent Memory Graphs, Null-Space Vertical Episodic Adapters, and Anchored MoE-Engram Stability Shields," GitHub Repository, 2026. [Online]. Available: https://github.com/BaiTian6641/Spark-X2.6-Recursion

[18] J. Fang *et al.*, "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2410.02355.

[19] P. Wang *et al.*, "WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2024, arXiv:2405.14768.

[20] S. Eyuboglu, R. Ehrlich, S. Arora, N. Guha, D. Zinsley, E. Liu, W. Tennien, A. Rudra, J. Zou, A. Mirhoseini, and C. Ré, "Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study," *arXiv preprint arXiv:2506.06266*, Jun. 2025.

[21] S. Laguna, J. Monteiro, M. Cuturi, P. Ablin, and E. Gualdoni, "Cartridges++: KV Cache Compression without Off-Context Derailment," *arXiv preprint arXiv:2609.35621*, Sep. 2026.

[22] M. Hardalov, G. Iglesias, and A. de Gispert, "Cartridges at Scale: Training Modular KV Caches over Large Document Collections," *arXiv preprint arXiv:2606.04557*, Jun. 2026.

[23] R. Charakorn, E. Cetin, S. Uesaka, and R. T. Lange, "Doc-to-LoRA: Learning to Instantly Internalize Contexts," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2026, arXiv:2602.15902.

[24] J. Lin, L. Zettlemoyer, G. Ghosh, W.-T. Yih, A. Markosyan, V.-P. Berges, and B. Oğuz, "Continual Learning via Sparse Memory Finetuning," *arXiv preprint arXiv:2510.15103*, Oct. 2025.

[25] DeepSeek-AI, "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models (Engram / Signed-Sqrt RMSNorm QK Gating)," *arXiv preprint arXiv:2601.07372*, Jan. 2026.

[26] A. Zweiger, J. Pari, H. Guo, E. Akyürek, Y. Kim, and P. Agrawal, "Self-Adapting Language Models (SEAL)," *arXiv preprint arXiv:2506.10943*, Jun. 2025.

[27] A. Kubicek, A. Caciolai, L. Peng, S. Sumant, B. Hoover, F. Greenlee, R. M. Battleday, R. Urban, T. Hoefler, and J. C. R. Whittington, "Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention," *arXiv preprint arXiv:2610.42147*, Oct. 2026.

[28] B. J. Gutiérrez, Y. Shu, W. Qi, S. Zhou, and Y. Su, "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2)," *arXiv preprint arXiv:2502.14802*, Feb. 2025.

[29] W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang, "A-MEM: Agentic Memory for LLM Agents," *arXiv preprint arXiv:2502.12110*, Feb. 2025.

[30] S. Bae, Y. Kim, R. Bayat, S. Kim, J. Ha, T. Schuster, A. Fisch, H. Harutyunyan, Z. Ji, A. Courville, and S.-Y. Yun, "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation," *arXiv preprint arXiv:2507.10524*, Jul. 2025.

[31] J. Geiping, S. McLeish, N. Jain, J. Kirchenbauer, S. Singh, B. R. Bartoldson, B. Kailkhura, A. Bhatele, and T. Goldstein, "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach," *arXiv preprint arXiv:2502.05171*, Feb. 2025.

[32] D. Raposo, S. Ritter, B. Richards, T. Lillicrap, P. C. Humphreys, and A. Santoro, "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models," *arXiv preprint arXiv:2404.02258*, Apr. 2024.

[33] J. Liu, B. Chen, and C. Zhang, "Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2025, arXiv:2502.02789.

[34] S. Upasani, R. S. Raju, B. Li, M. Ji, J. Long, C. Wu, U. Thakker, and G. Wang, "Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models," *arXiv preprint arXiv:2603.02631*, Mar. 2026.

[35] Y. Li, F. Wei, C. Zhang, and H. Zhang, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test," *arXiv preprint arXiv:2503.01840*, Mar. 2025.

[36] S. Cho, L. Cui, S. Kim, D. Han, and I. Han, "LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers," *arXiv preprint arXiv:2609.17184*, Sep. 2026.

[37] Y. Hu, K. Wang, X. Zhang, F. Zhang, C. Li, H. Chen, and J. Zhang, "SAM Decoding: Speculative Decoding via Suffix Automaton," in *Proc. Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2025, arXiv:2411.10666.

[38] S. M. Ghantasala, "TreeWY: Speculative Verification for Gated DeltaNet Hybrids," *arXiv preprint arXiv:2608.20961*, Aug. 2026.

[39] H. Chen *et al.*, "KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models," in *Proc. ACM Symp. Operating Syst. Principles (SOSP)*, 2025. [Online]. Available: https://github.com/kvcache-ai/ktransformers

[40] S. Cao, S. Liu, T. Griggs, P. Schafhalter, X. Liu, Y. Sheng, J. E. Gonzalez, M. Zaharia, and I. Stoica, "MoE-Lightning: High-Throughput MoE Inference on Memory-Constrained GPUs," in *Proc. ACM Int. Conf. Architectural Support Program. Lang. Operating Syst. (ASPLOS)*, 2025, arXiv:2411.11217.

[41] K. Kamahori, T. Tang, Y. Gu, K. Zhu, and B. Kasikci, "Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2402.07033.

[42] S. Di, Z. Yu, G. Zhang, H. Li, T. Zhong, H. Cheng, B. Li, W. He, F. Shu, and H. Jiang, "Streaming Video Question-Answering with In-Context Video KV-Cache Retrieval (ReKV)," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2503.00540.

[43] L. Long, Y. He, W. Ye, Y. Pan, Y. Lin, H. Li, J. Zhao, and W. Li, "Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory (M3-Agent)," *arXiv preprint arXiv:2508.09736*, Aug. 2025.

[44] Y. Wang, D. Krotov, Y. Hu, Y. Gao, W. Zhou, J. McAuley, D. Gutfreund, R. Feris, and Z. He, "M+: Extending MemoryLLM with Scalable Long-Term Memory," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2025, arXiv:2502.00592.

[45] B. Peng, J. Quesnelle, H. Fan, and E. Shippole, "YaRN: Efficient Context Window Extension of Large Language Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024, arXiv:2309.00071.

[46] Z. Xie *et al.* (DeepSeek-AI), "mHC: Manifold-Constrained Hyper-Connections," *arXiv preprint arXiv:2512.24880*, Dec. 2025.

[47] TassP0lar, "Qwen3.6-35B-A3B-Loop64: Open Looped Hybrid Checkpoints and Empirical Telemetry," Hugging Face Model Hub, Nov. 2026. [Online]. Available: https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64

[48] TassP0lar, "Loop64-Fable-SIRSI-v1: Curated Anti-Slop 3-Beat Looped-CoT, Aesthetic Craft, and SI-RSI Recovery Dataset," Hugging Face Datasets Hub (v1.2), Nov. 2026. [Online]. Available: https://huggingface.co/datasets/TassP0lar/Loop64-Fable-SIRSI-v1

[49] E. Guha *et al.*, "OpenThoughts: Data Recipes for Reasoning Models," *arXiv preprint arXiv:2506.04178*, Jun. 2025.

[50] A. Bercovich *et al.*, "Llama-Nemotron: Efficient Reasoning Models," *arXiv preprint arXiv:2505.00949*, May 2025.

[51] F. Chollet, M. Knoop, G. Kamradt, B. Landers, and H. Pinkard, "ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems," *arXiv preprint arXiv:2505.11831*, May 2025.

[52] E. Wijmans, B. Huval, A. Hertzberg, V. Koltun, and P. Krähenbühl, "Cut Your Losses in Large-Vocabulary Language Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2411.09009.

[53] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," *arXiv preprint arXiv:2402.03300*, Feb. 2024.

[54] Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, *et al.*, "DAPO: An Open-Source LLM Reinforcement Learning System at Scale," *arXiv preprint arXiv:2503.14476*, Mar. 2025.
