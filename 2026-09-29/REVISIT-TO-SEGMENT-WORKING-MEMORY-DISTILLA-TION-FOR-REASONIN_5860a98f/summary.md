---
title: "REVISIT-TO-SEGMENT-WORKING-MEMORY-DISTILLA-TION-FOR-REASONIN"
source: https://arxiv.org/pdf/2609.34863v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:44:07"
field: "多模态推理分段"
keywords: ["reasoning segmentation", "working memory distillation", "on-policy self-distillation", "GRPO", "multimodal large language model", "Jensen-Shannon divergence"]
innovations: ["首次将自生成工作记忆作为 on-policy 自蒸馏信号引入推理分段，实现过程+结果双重监督", "提出按分割质量筛选 Top-K rollout 构建工作记忆的 teacher context 构造策略", "联合 GRPO 与 OPSD 的轻量两阶段训练框架，在无需额外标注的情况下把推理时增益固化进朴素预测"]
benchmarks: ["ReasonSeg", "ReasonSeg-R", "ReasonSeg-X", "MUSE", "MMR", "RefCOCO"]
---

# 论文速读：REVISIT TO SEGMENT: WORKING MEMORY DISTILLATION FOR REASONING SEGMENTATION

## 一句话总结
本文提出 **SWiM（Segmenter with Working Memory）**，一种将"重访自身先前推理尝试"转化为训练信号的工作记忆蒸馏框架：通过按分割质量筛选 rollout 构建工作记忆，以记忆条件化的模型作为 teacher 对 student 进行 on-policy 自蒸馏（OPSD），并与 GRPO 结果级强化学习联合优化，在 ReasonSeg 系列基准上达到 SOTA。

## 研究问题与动机
- **现有方法忽略了 MLLM 自身生成内容的再利用价值**：MLLM 在推理分段时产生的 reasoning traces 和 localization proposals 本质上是"自我生成的工作记忆"，但已有工作（LISA、Seg-Zero、StAR 等）仅将其视为最终输出的中间产物，未系统利用其作为训练信号。
- **重访先前尝试能持续提升性能**：作者发现 conditioning on 自身先前生成的推理与定位结果（即工作记忆）能让模型重新审视 query 解读，在所有模型规模和基准上均获得稳定提升（Figure 2）。
- **如何将"推理时增益"迁移到"不分推理时"的朴素预测**：工作记忆条件化 teacher 的增益若仅保留为推理时技巧，则无法普惠朴素单轮预测；需要蒸馏进 student 参数本身。
- **现有 RL 方法缺少过程级分布式监督**：GRPO 等结果级 RL 只优化最终 mask IoU，缺少对中间 token 生成的细粒度指导，难以在低资源训练数据下充分引导复杂推理链路。

## 核心贡献（创新点）
1. **首次将工作记忆形式化为自蒸馏信号引入推理分段**：将 MLLM 自生成的 reasoning traces + localization proposals 选优后作为 working memory，验证其同时具备上下文信息与训练指导双重价值，本质区别在于此前工作仅用工作记忆做推理时辅助（如 StAR 的 MV），本文进一步将其转化为 token-level 分布式监督。
2. **首次把 on-policy self-distillation（OPSD）带到推理分段领域**：以工作记忆条件化的当前策略作为自 teacher，在 student 自生成轨迹上施加 JSD 对齐；区别于 Zhao et al. (2026) 原框架使用外部验证 trace，此处 teacher 与 student 同参数不同输入（含/不含 WM），实现"重访先验"的过程知识迁移。
3. **联合优化 OPSD 与 GRPO 的双重目标**：OPSD 提供 token-level 过程指导，GRPO 提供 outcome-level 结果反馈，两者共享同一 student-generated trajectories，联合目标 $\mathcal{L}_{\text{SWiM}} = \mathcal{L}_{\text{GRPO}} + \lambda \mathcal{L}_{\text{OPSD}}$；区别于纯 RL 工作，这是"过程+结果"互补的典型设计。
4. **构建面向难例的 Stage-2 训练集选择策略**：从 5,166 条 Stage-1 数据中按 IoU 统计（最大值、均值、达标数、gap Δ）筛选出 869 条难例/前沿/回放样例，使蒸馏聚焦于仍有提升空间的样本。
5. **系统化验证工作记忆的多种角色**：不仅验证其作为 teacher context 的有效性，还对比 GT hint、Random hint、不同 λ、不同 K、不同散度（JSD vs KL fwd/bwd），并定性比较 WM 推理与 MV 聚合在错误纠正上的差异。

## 方法详解
### 整体架构
- **Backbone**：Qwen3-VL-8B/32B-Instruct + 冻结的 SAM2.1 Hiera-Large（mask decoder）。
- **Prompt 输出格式**：逐步 thinking → `<answer>` 包裹的 JSON，内含 `{label, bbox_2d, point_2d}`。
- **Decoupled 设定**：MLLM 产出 box + point + label，SAM2 冻结生成 mask，最终 $\widehat{M} = \bigcup_j S(I, b_j, p_j)$。

### 4.1 工作记忆构建
- 对输入 $x=(I,q)$，从旧策略 $\pi_{\theta_{\text{old}}}$ 采样 $N$ 个 response $\{y^i\}$，经 SAM2 得 mask $\widehat{M}^i$。
- 按训练 mask IoU 排序，取 **Top-K** 作为工作记忆 $\mathcal{W}$（默认 $N=16, K=8$）。
- 保留完整文本推理与结构化定位，确保推理-空间映射可追溯。

### 4.2 工作记忆蒸馏
**OPSD 部分**：
- Teacher 输入构造：$x^{\mathcal{W}} = C(x, \mathcal{W})$（拼接图像、query 与历史 attempt）。
- 在 student 自生成的 prefix $y_{<t}^i$ 上，分别计算 student 分布 $p_{i,t} = \pi_\theta(\cdot|x, y_{<t}^i)$ 与 teacher 分布 $q_{i,t} = \pi_{\theta_{\text{old}}}(\cdot|x^{\mathcal{W}}, y_{<t}^i)$。
- 为高效计算，取 teacher 的 top-L（默认 L=16）token + 剩余合并为一个 residual 桶，形成共享词汇划分 $\widetilde{p}, \widetilde{q}$。
- 最小化 Jensen–Shannon divergence：
$$\mathcal{L}_{\text{OPSD}} = \mathbb{E}_{(i,t)\sim\mathcal{U}}\!\left[\text{JSD}(\widetilde{p}_{i,t}\,\|\,\widetilde{q}_{i,t})\right]$$
- Teacher 分布**无梯度、固定**于单次更新。

**GRPO 部分**：
- 奖励函数 $r^i = 2r_{\text{seg}}^i + r_{\text{fmt}}^i + r_{\text{nr}}^i$（seg/fmt/nr 三项，seg 权重 2×）。
- 组内归一化 advantage $A^i = (r^i - \mu_r)/(\sigma_r + \epsilon)$。
- GRPO 目标：$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}[\min(\rho A, \text{clip}(\rho,1\!-\!\delta,1\!+\!\delta)A)]$。

**联合优化**：
$$\mathcal{L}_{\text{SWiM}} = \mathcal{L}_{\text{GRPO}} + \lambda \mathcal{L}_{\text{OPSD}}, \quad \lambda=0.1$$
两阶段训练：Stage 1（纯 RL warm-up，1 epoch，lr=1e-5）→ Stage 2（联合蒸馏，2 epoch，lr=1e-6，869 条精选数据）。

## 实验与结果
**基准与指标**：ReasonSeg（RS）、ReasonSeg-R（RS-R）、ReasonSeg-X（RS-X），主指标 gIoU/cIoU；另测 MUSE、MMR、RefCOCO 家族泛化。

**最强结果（Qwen3-VL-8B，不含 RS-X 训练数据）**：
- **SWiM（朴素推理）**：平均 gIoU **66.5**，cIoU **62.0**；相对重现 StAR-8B（64.6/58.4）**+1.9 / +3.6 pp**。
- **SWiM + WM & MV（Qwen3-VL-8B）**：平均 gIoU **67.8**，cIoU **63.0**；相对 StAR+MV（65.5/59.5）**+2.3 / +3.5 pp**。
- **Qwen3-VL-32B**：朴素 69.2/65.0；+WM&MV **70.4/65.9**，进一步印证可扩展性。
- **跨基准**：MUSE 58.3/56.8 cIoU；MMR test 33.6/28.9；RefCOCOg 73.9，均超过重现 StAR。

**消融关键点**：
- 纯 GRPO：65.7/60.6；纯 OPSD：66.5/62.0；联合：66.5/62.0（OPSD 主导 cIoU 提升）。
- λ=0.1 最优；λ=10 下降，说明蒸馏权重需克制。
- 工作记忆 teacher 显著优于 GT visual hint、Random hint（51.5/48.7）。
- 散度选择：JSD > reverse KL ≈ forward KL。
- 推理时 K=8 最优；K=16 仅 cIoU +0.2、gIoU 下降。
- Stage-2 难例选择策略有效：frontier 364、refinement 200、hard-but-partial 97、unsuccessful 58、replay 150。

## 相关工作脉络
1. **LISA / PixelLM / GLaMM / OMG-LLaVA**：早期基于 SFT 将 MLLM 对齐到像素级分割的奠基工作，本文在其基础上引入 RL+蒸馏的二次优化路径。
2. **CoReS / RSVP / CoPRS**：结构化推理与视觉 prompt 引导定位，与本文"过程监督"思路互补，但本文通过蒸馏而非 prompt 工程实现。
3. **Seg-Zero / SAM-R1 / VisionReasoner / StAR**：将 GRPO 等 RL 引入视觉推理分段；本文在此基础上增加"工作记忆 condition 自蒸馏"维度，是 RL-only 方法的下一步演进。
4. **OPSD 原生工作（Zhao et al. 2026; Yu et al. 2026; Wu et al. 2026）**：已在数学/代理 RL 中验证，本文首次将其迁移到多模态推理分段，且 teacher conditioning 对象改为"自生成工作记忆"而非外部 verified trace。
5. **SAM 3 Agent / RSAgent / Rea²Seg**：推理时多轮工具调用/agent 扩展路线，本文强调把多轮信息蒸馏进单次策略参数。
6. **POPEN / Seg-ReSearch / SELF1E**：近期推理分段方法，本文通过跨基线对比证明工作记忆蒸馏的普遍有效性。

## 局限性与未来方向
- **MLLM 与 SAM2 未联合优化**：SAM2 保持冻结，失败案例（Figure 10 c/d）显示即使 MLLM 定位合理，SAM2 也会产出残缺 mask；未来可尝试联合微调或提供更丰富的 spatial prompts。
- **工作记忆 teacher 信息上限有限**：相比 GT visual hint，纯文本/位置记忆难以给出精确 target location，Table 7 显示 GT hint 个别指标反超但综合不及 WM。
- **K 增大不单调**：K=16 较 K=8 反而 gIoU 略降（Appendix I），说明噪声 attempt 引入会干扰 teacher 分布；如何自适应筛选仍是开放问题。
- **训练数据量仍偏小**：Stage-2 仅 869 条精选样例，虽避免过拟合 RS-X，但限制复杂长尾 query 的覆盖。
- **仅验证单一骨架**：实验集中在 Qwen3-VL，对其他 MLLM（如 InternVL、Gemini 系）的可迁移性待考察。

## 研究启发与可借鉴点
1. **"自我工作记忆"作为过程监督信号的通用范式**：不仅限于分段，任何 MLLM 生成多步轨迹 + 可量化 intermediate/outcome 的任务（数学、代理、 grounding）均可借鉴此蒸馏路径。
2. **难度感知的 Stage-2 样本筛选策略**：基于 IoU 统计（max、mean、Δ、达标计数）构造 frontier/refinement/hard/replay 五类难例，为后续训练集构建提供可直接复用的指标体系。
3. **JSD 比 KL 更稳健的实验结论**：在 teacher-student 分布差异较大时，JSD 优于 fwd/reverse KL，这一经验可迁移至其他自蒸馏场景。
4. **"推理时增益 ≠ 训练时增益"的蒸馏闭合**：证明仅靠推理时 WM/MV 只能带来边际提升，真正需要的是把过程知识固化进策略；这为同类工作提供了清晰的 ablation 设计模板。
5. **多目标联合（过程 distill + 结果 RL）的权重敏感性**：λ 从 0.1→10 出现明显下降，提示后续研究应关注蒸馏强度的自动调节（如 curriculum、meta-λ）。

## 关键术语表
- **Working Memory Distillation**：以模型自生成的先前尝试（推理 + 定位）为 teacher 输入，通过 on-policy 自蒸馏把"重访增益"迁移到朴素 student 的策略内。
- **On-Policy Self-Distillation (OPSD)**：teacher 与 student 同参数，但 teacher 额外接收工作记忆；以 JSD 对齐同一条 student 轨迹上的 next-token 分布。
- **Group Relative Policy Optimization (GRPO)**：Shao et al. (2024) 提出的 RL 优化器，组内归一化 advantage + probability ratio clipping，用于 outcome-level 分段质量优化。
- **Jensen–Shannon Divergence (JSD)**：对称散度，本文用于 OPSD 的 distribution alignment，比 KL _fwd/rev 更稳健。
- **Reasoning Segmentation**：给定图像与自然语言 query，要求模型进行上下文推理并输出像素级 mask 的任务（Lai et al. 2024 首创）。
- **gIoU / cIoU**：gIoU 为逐样本 IoU 的均值；cIoU 为累积 intersection 除以累积 union，更惩罚全局覆盖偏差。
- **Top-K Working Memory Construction**：从 N 个 rollout 中按训练 mask IoU 选前 K 条完整 response 作为 teacher 的历史上下文。
- **Stage-1 RL Warm-up / Stage-2 Distillation**：两阶段训练，Stage 1 让策略学会"带/不带 WM 均能推理"，Stage 2 通过 OPSD+GRPO 把 WM 增益固化进朴素预测。

## 可复现要素
- **代码**：已开源，GitHub https://github.com/yancilin/SWiM
- **数据集**：LVIS、RefCOCOg、gRefCOCO（Stage 1，5,166 条）；评测集 ReasonSeg / RS-R / RS-X / MUSE / MMR / RefCOCO（未使用 RS-X 训练集，与 StAR 公平比较）
- **模型**：Qwen3-VL-8B-Instruct / Qwen3-VL-32B-Instruct（LoRA rank=64, scale=64）+ SAM2.1 Hiera-Large（冻结）
- **关键超参**：Stage 1 lr=1e-5, batch=24, N=16, K_WM=8, 1 epoch；Stage 2 lr=1e-6, batch=16, N=16, K=8, λ=0.1, 2 epoch, max_len=2048
- **硬件**：24 × NVIDIA H800
- **评估设置**：greedy, 最多 1024 新 token, 图片 resize 840×840, WM+MV 默认 32 WM-conditioned + 32 plain
