---
title: "OMNIROUTE-MAPPING-TEMPORAL-SEMANTIC-EVIDENCE-TO-AUDIO-VISUAL"
source: https://arxiv.org/pdf/2609.37052v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:45:45"
field: "多模态大模型高效推理"
keywords: ["Omni-LLMs", "token compression", "multimodal large language models", "training-free", "audio-visual understanding", "temporal semantic evidence"]
innovations: ["提出无需训练的两阶段时序感知 token 压缩框架，首次显式建模音频-视觉语义相关性的时序波动性", "设计跨模态预算动态校准机制，利用主导模态实际保留率反向调整跟随者目标", "针对视频和音频设计差异化的语义压缩策略，结合时空分组、查询引导选择和上下文锚点合并"]
benchmarks: ["WorldSense", "VideoMME", "AVUTBench", "DailyOmni"]
---

# 论文速读：OMNIROUTE-MAPPING-TEMPORAL-SEMANTIC-EVIDENCE-TO-AUDIO-VISUAL

## 一句话总结
针对全模态大语言模型（Omni-LLMs）处理长音频-视觉序列时的预填充开销问题，本文提出 OmniRoute——一种无需训练的、两阶段时序感知的 token 压缩框架，通过 chunk 级模态偏好估计与跨模态预算校准，在保留关键语义的同时显著降低计算成本。

## 研究问题与动机
- **核心问题**：Omni-LLMs（如 Qwen2.5-Omni、OmniVinci）将音频和视觉流编码为时序交错的 token 序列进行推理，长序列因注意力二次复杂度导致极高的预填充开销。
- **现有方法不足**：
  1. 已有压缩方法多关注跨模态互补性或模型内部冗余，但**忽视音频-视觉语义相关性的时序变化**。
  2. 作者对 500 个 WorldSense 视频的分析表明，查询条件化的音频-视觉相关性对比 $D_i$ 存在明显的**时序波动性**（$V_s$ 中位数为 0.225）和**局部连续性**（78.2% 的样本 $C_s < 1$），这对压缩设计有重要启发。
  3. 固定压缩策略难以适应视频不同片段中主导模态的动态切换。

## 核心贡献（创新点）
1. **提出无需训练的两阶段 token 压缩框架 OmniRoute**，首次将时序语义证据显式引入 Omni-LLMs 压缩设计，而非仅依赖静态或空间层面的特征。
2. **设计 TEGB（Temporal Evidence-Guided Budgeting）模块**，通过 chunk 级模态偏好估计（基于 $D_i$ 的 sigmoid 变换）和局部内容变化度量，动态确定主导模态及其初始预算。
3. **设计 BCSC（Budget-Constrained Semantic Compression）模块**，提出主导模态先压缩、跟随者后校准的机制，并利用实际保留率反向调整另一模态的目标，实现跨模态预算联动。
4. **针对不同模态设计差异化压缩策略**：视频采用分层时空分组 + 查询引导选择；音频采用编码器注意力 + 查询相关性融合评分 + 视觉引导的上下文锚点合并。

## 方法详解
**框架概览**：OmniRoute 在 LLM prefill 之前执行两个阶段（图 3）：TEGB → BCSC。

**TEGB（时序证据引导预算分配）**：
- **模态偏好估计**：对第 $i$ 个 chunk，计算音频与视觉 token 和查询嵌入的余弦相似度均值 $E_i^a$ 和 $E_i^v$，得到对比度 $D_i = E_i^a - E_i^v$。通过 sigmoid 变换 $\lambda_i = \sigma(\frac{D_i - \tau}{T})$ 得到偏好强度，$\lambda_i \geq 1/2$ 时主导模态为音频，否则为视频。
- **局部内容变化**：视频层面，$m_i$ 衡量 chunk 内相邻帧对应空间位置的余弦距离（时序内变化），$n_i = 1 - \cos(\mu_i, \mu_{i-1})$ 衡量 chunk 间均值变化的相似度（时序间变化）。
- **初始视觉保留率**：$g_i = 0.7m_i + 0.3n_i$，然后 $r_{i,0}^v = \Pi_v(\bar{r}_v + \alpha(g_i - \bar{g}))$ 根据变化幅度动态调整。
- **主导模态预算**：$r_i^{\ell_i} = \bar{r}_{\ell_i} + 2K|\lambda_i - 1/2|$，偏好越强（$\lambda_i$ 越偏离 0.5），主导模态保留率越高。

**BCSC（预算约束语义压缩）**：
- **视频主导压缩**：递归地对空间 token 网格进行层次化区域划分，将相似且重叠的区域跨时间步合并，构造候选 mask；通过调整 $\gamma_v$ 提高查询相关区域的相似度阈值，实现更保守的合并；最终按目标率 $r_i^v$ 进行查询引导的选择。
- **音频主导压缩**：结合编码器注意力重要性 $h_{i,j}$ 和查询相关性 $e_{i,j}^a$，计算综合分数 $o_{i,j}^a = \hat{h}_{i,j} + \gamma_a \hat{e}_{i,j}^a$，按分数降序保留 $\check{r}_i^a$ 比例；剩余 token 按余弦相似度分配到等间距上下文锚点，组内按最大余弦相似度到保留视觉 token 排序，softmax 加权平均合并。
- **跨模态预算校准**：主导模态实际保留率 $\hat{r}_i$ 用于校准跟随者目标：$\check{r}_i^{follower} = \Pi(\bar{r} + \beta(\hat{r}_i - \bar{r}))$，其中 $\beta$ 为跨模态耦合系数。

## 实验与结果
- **数据集**：WorldSense（真实场景音频-视觉理解）、VideoMME（短/中/长视频）、AVUTBench（音频中心视频理解）、DailyOmni（日常场景推理）。
- **基线**：Full Tokens、Random、DyCoke(V&A)、FlashVID、OmniZip。
- **主要模型**：Qwen2.5-Omni-7B/3B、OmniVinci。
- **核心结果（Table 1）**：
  - 在 45% token 保留率下，OmniRoute 在 Qwen2.5-Omni-7B 上平均归一化准确率最高，**46.9% WorldSense 准确率超越 Full Tokens（46.8%）**。
  - 相比 OmniZip（同样 45% 保留），在 7B/3B/OmniVinci 上分别提升 **1.1、0.5、1.3 个百分点**。
  - 平均保留 97.9–98.7% 的 Full Tokens 准确率，同时 FLOPs 降低 61–64%。
- **效率分析（Table 2，Qwen2.5-Omni-7B on WorldSense）**：
  - 峰值内存降至 **27GB**（相比 Full Tokens 的 44GB，降低 38.6%）。
  - Prefill 加速 **2.53×**，端到端加速 **1.35×**，内存节省优于所有基线。
- **Token 保留率分析（Table 4）**：从 45% 降至 30%，准确率仅从 46.9% 降至 45.6%，但 FLOPs 降低 54%，证明灵活的可调性。
- **消融实验（Table 3）**：移除内容变化信号或固定模态路由均导致性能下降，验证各组件的有效性。

## 相关工作脉络
1. **OmniZip**（2025）：音频引导的动态 token 压缩方法，专为 Omni-LLMs 设计，但未考虑时序语义变化，依赖固定音频权重；OmniRoute 进一步引入时序证据和跨模态动态预算校准。
2. **DyCoke(V&A)**（CVPR 2025）：针对视频 LLM 的动态压缩，将 temporal module 应用于音视频；但缺乏对音频-视觉相关性的显式建模与 chunk 级路由。
3. **FlashVID**（ICLR 2026）：基于树的时空 token 合并，面向纯视觉场景；不处理音频-视觉交互，也不考虑查询相关性的时序变化。
4. **Omnifit**（ICML 2025）：通过层自适应 token 压缩桥接模态；侧重 KV-cache 优化，而非 prefill 前的 input token 压缩。
5. **Omniselect**（2026）：动态模态感知 token 压缩，但未显式建模音频-视觉语义相关性的时序波动性与局部连续性。
6. **EchoingPixels**（2025）/ **FastAV**（2026）：早期跨模态自适应压缩工作，主要关注空间或层级的冗余去除，缺乏查询引导的 chunk 级时序建模。

## 局限性与未来方向
- **人工超参调优**：需要手动配置 $\gamma_v$、$\gamma_a$、$\beta$ 等语义权重和跨模态耦合系数，不同场景可能需重新调整。
- **未探索参数自适应**：论文自述未来将探索将这些参数自适应到目标 token 预算，减少人工调参。
- **推理延迟优化空间**：当前仅降低峰值内存和 FLOPs，后续需优化实现以进一步降低 latency。
- **评估场景有限**：主要在四类 benchmark 上验证，对于超长视频（hour-level，如 Avoc 提及的场景）的扩展性未充分讨论。

## 研究启发与可借鉴点
1. **时序证据驱动预算分配**的范式可迁移：将 chunk 级相关性对比（$D_i$）与变化度量（$m_i, n_i$）结合来动态分配保留预算的思路，适用于其他时序多模态压缩场景（如视频+字幕、音频+文本）。
2. **主导-跟随跨模态校准机制**：实际保留率反向调整另一模态目标的设计，是一种轻量级的跨模态协同方式，无需额外训练参数，可直接集成到现有框架。
3. **视频分层时空分组 + 查询引导选择**：递归区域划分与查询相关性动态调整阈值的方法，可作为视觉 token 压缩的通用模块。
4. **编码器注意力 + 查询相关性融合**的音频 token 选择策略，提供了无需额外训练的语义感知的音频压缩路径。
5. **无需训练的压缩范式**：在当前多模态大模型推理成本日益突出的背景下，training-free 方法具有部署友好性，可与微调/蒸馏方法形成互补。

## 关键术语表
- **Omni-LLMs（全模态大语言模型）**：统一处理文本、音频、视觉等多种模态输入的大语言模型，如 Qwen2.5-Omni、OmniVinci。
- **TEGB（Temporal Evidence-Guided Budgeting）**：时序证据引导预算分配，利用音频-视觉语义相关性对比和局部内容变化，动态估计 chunk 级模态偏好和初始保留预算。
- **BCSC（Budget-Constrained Semantic Compression）**：预算约束语义压缩，先压缩主导模态，再用其实际保留率校准跟随者模态的目标保留率。
- **$D_i$（模态相关性对比度）**：第 $i$ 个 chunk 中音频与视觉 token 对查询的相关性均值之差，用于判断主导模态。
- **上下文锚点（Context Anchors）**：在音频压缩中均匀分布的代表性 token，用于聚合和合并剩余未选中 token 的参考点。
- **跨模态耦合系数 $\beta$**：控制主导模态实际保留率对跟随者目标保留率影响程度的超参。
- **归一化准确率**：压缩方法准确率与 Full Tokens 准确率之比，用于跨不同模型公平比较。

## 可复现要素
- **数据集**：WorldSense、VideoMME、AVUTBench、DailyOmni（均为公开基准）。
- **代码/权重**：论文声明代码和接口将发布（code and interface will be released），当前未明确开源。
- **关键超参**：
  - 空间相似度阈值 $\eta_s = 0.82$
  - 时间相似度阈值 $\eta_t = 0.58$
  - 路由温度 $T = 0.03$
  - 上下文锚点比例 = 0.05
  - 语义权重 $\gamma_v = 0.21$、$\gamma_a = 0.09$
  - 跨模态耦合 $\beta = 0.50$
  - 视频采样率：Qwen2.5-Omni 为 2 fps（最多 768 帧），OmniVinci 为 128 帧
- **硬件**：单卡 NVIDIA L20 GPU（48GB）
- **模型**：Qwen2.5-Omni-3B/7B、OmniVinci
