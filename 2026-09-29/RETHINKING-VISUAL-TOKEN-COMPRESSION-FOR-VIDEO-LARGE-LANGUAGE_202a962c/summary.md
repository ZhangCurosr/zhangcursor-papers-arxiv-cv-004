---
title: "RETHINKING-VISUAL-TOKEN-COMPRESSION-FOR-VIDEO-LARGE-LANGUAGE"
source: https://arxiv.org/pdf/2609.35394v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:32:41"
field: "多模态大模型高效推理"
keywords: ["视频大语言模型", "视觉Token压缩", "特征空间聚类", "跨帧聚类", "无训练加速", "RoPE", "低保留率鲁棒性"]
innovations: ["提出无需训练的跨帧位置感知聚类基线SimpleCluster，在1%保留率下显著优于SOTA", "建立特征空间保真度(L2-NQE/Cosine Coverage)与下游视频理解性能的一致性关联", "证明跨帧联合聚类严格优于逐帧独立预算分配的理论与实践保证"]
benchmarks: ["MVBench", "EgoSchema", "LongVideoBench", "VideoMME"]
---

# 论文速读：RETHINKING VISUAL TOKEN COMPRESSION FOR VIDEO LARGE LANGUAGE MODELS: A SIMPLE YET STRONG BASELINE

## 一句话总结
本文提出 SimpleCluster，一个**无需训练、仅依靠跨帧位置感知聚类**的视频 Token 压缩基线方法，在极低 Token 保留率（如 1%）下仍展现出优于近期复杂 SOTA 方法的鲁棒性；其核心洞察是：**下游性能与原始视觉特征空间的保真度（尤其是全局覆盖）密切相关**。

---

## 研究问题与动机
1. **Video LLM 推理效率瓶颈**：长视频会产生海量视觉 Token（密集时空表征），导致预填充计算量和延迟过高。
2. **现有方法过度复杂化**：近年来视频 Token 压缩方法不断引入注意力选择、密度感知剪枝、最优传输等精巧设计，但这些复杂设计是否真正必要？在极端低预算下，有多大比例的性能可以归因于"保留特征空间结构本身"而非"精巧选择机制"？
3. **低保留率下的鲁棒性缺口**：当 Token 保留率降至 1% 时，多数方法性能骤降（如 AOT、FastVID 无法运行），缺乏能在极端压缩下稳定工作的简单方法。

---

## 核心贡献（创新点）
1. **SimpleCluster 基线**：提出无需训练、无架构修改的跨帧位置感知聚类压缩方法，在多种 Video LLM 和多保留率下达到或超越 SOTA。
2. **简单性战胜复杂性**：证明无需 attention-guided anchors、optimal transport 等复杂设计，仅靠特征空间聚类即可获得更强压缩性能，尤其在 1% 极低保留率下优势显著。
3. **特征空间视角的解释框架**：从局部逼近保真度（L2-NQE）和全局覆盖（Cosine Coverage@0.90）两个维度系统分析不同压缩方法的特征空间保持能力，建立了"特征空间保真度 → 下游性能"的一致性关联。
4. **跨帧预算共享的理论解释**：证明联合跨帧聚类严格优于任何逐帧独立分配预算的策略（$J_{\text{joint}}^\star(B) \leq \min_{\sum B_t = B} \sum_t J_t^\star(B_t)$），为跨帧聚类的有效性提供了理论支撑。

---

## 方法详解

### 总体流程
SimpleCluster 作用于视觉编码器输出后经 projector 投影得到的视觉 Token $X = [x_1, \ldots, x_N] \in \mathbb{R}^{N \times d}$，将序列压缩为 $Y = [y_1, \ldots, y_B] \in \mathbb{R}^{B \times d}$，其中 $B \ll N$，保留率 $r = B/N$。

### 关键三步设计

**Step 1：位置感知特征构造（Position-Aware Feature Construction）**
- 对每个 Token $x_i$，获取其时空位置 $p_i = (f_i, h_i, w_i)$（帧号、行、列）。
- 应用 3D RoPE 旋转编码：$\tilde{x}_i = R(p_i) x_i$，再做 L2 归一化得到聚类特征：
  $$z_i = \frac{R(p_i) x_i}{\|R(p_i) x_i\|_2}$$
- 关键点：**3D RoPE 仅用于决定 Token 如何分组，最终聚合仍在原始特征空间 $x_i$ 中进行**，不改变 LLM 看到的表示。

**Step 2：跨帧聚类（Cross-Frame Clustering）**
- 对所有 $T$ 帧的 $N$ 个 Token 联合聚类，最小化簇内余弦距离：
  $$\{\mathcal{C}_k\}_{k=1}^B = \arg\min_{\mathcal{C},\{\mu_k\}} \sum_{k=1}^B \sum_{i \in \mathcal{C}_k} \left(1 - \frac{z_i^\top \mu_k}{\|z_i\|_2 \|\mu_k\|_2}\right)$$
- 等价于最小化 $J_B^\star = \min_{\mathcal{C},\{\mu_k\}} \sum_k \sum_{i \in \mathcal{C}_k} \|z_i - \mu_k\|_2^2$（经典 K-Means 目标在余弦空间中的变体）。
- 簇原型更新：$\mu_k \leftarrow \frac{\sum_{i \in \mathcal{C}_k} z_i}{\|\sum_{i \in \mathcal{C}_k} z_i\|_2}$

**Step 3：簇内特征聚合（Cluster-Wise Feature Aggregation）**
- 每个簇由原始特征均值表示：
  $$y_k = \frac{1}{|\mathcal{C}_k|} \sum_{i \in \mathcal{C}_k} x_i$$
- 这是最小化簇内平方重建误差的最优解：$y_k = \arg\min_y \sum_{i \in \mathcal{C}_k} \|x_i - y\|_2^2$

### 超参
- 3D RoPE 频率基底 $\theta = 10{,}000$；无其他超参，纯无训练（training-free）。

---

## 实验与结果

### 实验设置
- **三个 Video LLM**：LLaVA-OneVision-7B、LLaVA-Video-7B、InternVL3-2B
- **四个基准**：MVBench、EgoSchema、LongVideoBench、VideoMME
- **保留率**：1%、5%、10%、15%、20%、25%
- **对比基线**：DyCoke、VisionZip、FastVID、EarlyTom、AOT

### 主要结果（核心数字）

| 保留率 | 模型 | SimpleCluster Avg | 最佳基线 Avg | 相对原模型 |
|---|---|---|---|---|
| **1%** | LLaVA-OV-7B | **51.6** | EarlyTom 45.7 | **88.3%** |
| **1%** | LLaVA-Video-7B | **51.0** | EarlyTom 40.9 | **84.8%** |
| **1%** | InternVL3-2B | **49.1** | EarlyTom 44.3 | **85.8%** |
| 5% | LLaVA-OV-7B | **55.7** | FastVID 54.1 | 95.3% |
| 10% | LLaVA-OV-7B | **57.3** | AOT 57.0 | 98.1% |
| 25% | LLaVA-OV-7B | **58.6** | AOT 58.5 | 100.3% |

- **最强结果**：在 LLaVA-OneVision-7B 上 25% 保留率达到 58.6（超越 AOT 的 58.5），1% 保留率仍保留 88.3% 性能。
- **效率**：在 10% 保留率下，Prefill FLOPs 从 102.5T 降至 27.5T（减少 73.2%），TTFT 从 943.3ms 降至 514.1ms（减少 45.5%），优于最强基线 AOT（TTFT 696.5ms）。

### 特征空间分析（Table 7）
- 10% 保留率：SimpleCluster 的 L2-NQE = **0.281**（较 EarlyTom 的 0.407 降低 31.0%），Cosine Coverage@0.90 = **59.7%**（较 FastVID 的 35.0% 提升 24.7pp）
- 5% 保留率：L2-NQE = **0.336**，Coverage = **45.9%**，优于所有方法在 10% 下的表现

---

## 相关工作脉络

1. **VisionZip（CVPR'25）**：基于视觉注意力选择主导 Token 并合并冗余 Token——与 SimpleCluster 的区别在于依赖逐帧注意力打分，而 SimpleCluster 直接在全局特征空间聚类。
2. **FastVID（NeurIPS'25）**：时序分割 + 密度感知剪枝——依赖手工设计的阈值参数，且在 1% 极低预算下失效；SimpleCluster 无需任何阈值。
3. **EarlyTom（CVPR'26）**：在视觉编码器早期阶段进行 Token 合并与空间选择——通过多层级剪枝实现，但剪枝位置需任务特定调参。
4. **AOT（CVPR'26）**：基于注意力引导锚点和最优传输（Sinkhorn）的 Token 聚合——计算复杂度远高于 SimpleCluster，且需要调优正则化系数。
5. **DyCoke（CVPR'25）**：跨帧 Token 合并 + 动态 KV-cache 剪枝——结合了编解码两侧优化，工程复杂度高。
6. **本文定位**：作为强基线研究，证明特征空间结构的保留比复杂选择机制更重要；未来方向可在此基础上引入自适应压缩策略。

---

## 局限性与未来方向
1. **作为基线研究的定位限制**：论文自承是 strong baseline study，未完全刻画复杂设计在哪些场景下仍有额外收益。
2. **聚类目标与查询无关**：当前目标是通用特征相似性，未考虑具体下游任务（如视频 QA 的特定语义需求）。
3. **未探索自适应保留率**：固定全局预算，未按视频内容复杂度动态调整各帧/各区域的 Token 分配。
4. **未来方向**：探索显式平衡特征空间全局覆盖、局部保真度与计算开销的自适应压缩策略。

---

## 研究启发与可借鉴点
1. **特征空间保真度作为压缩质量指标**：L2-NQE 和 Cosine Coverage@0.90 是两个简洁有效的分析工具，可迁移到图像 Token 压缩、多模态压缩等领域评估不同压缩方法的本质差异。
2. **位置编码辅助聚类的设计**：3D RoPE 仅用于分组决策而不改变最终表征——这种"位置信息指导组织、原始信息保留语义"的分离策略可推广到任何需要结构感知的特征压缩场景。
3. **跨帧联合聚类的理论保证**：$J_{\text{joint}}^\star(B) \leq \min_{\sum B_t=B} \sum_t J_t^\star(B_t)$ 的不等式证明为跨域联合优化提供了形式化依据，可启发其他视频理解任务中的跨帧信息聚合设计。
4. **简单性的实验价值**：证明在极端低预算下简单方法的优越性，提示团队在资源受限场景下优先尝试无训练基线，避免过早引入复杂设计。

---

## 关键术语表
- **Token Retention Ratio（Token 保留率）**：压缩后保留的视觉 Token 数 $B$ 占原始数量 $N$ 的比例 $r = B/N$。
- **L2-NQE（L2 Normalized Quantization Error）**：衡量每个原始 Dense Token 到最近压缩 Token 的归一化欧氏距离，反映局部逼近保真度。
- **Cosine Coverage@0.90**：原始 Dense Token 中至少有 1 个压缩 Token 与其余弦相似度 ≥ 0.90 的比例，反映全局特征空间覆盖度。
- **3D RoPE（三维旋转位置编码）**：将特征通道按时间/垂直/水平分组并施加旋转编码，使聚类空间同时感知视觉相似性和时空位置。
- **Cross-Frame Clustering（跨帧聚类）**：对所有采样帧的 Token 联合聚类，而非逐帧独立分配预算，充分利用跨帧冗余。
- **Training-free（无需训练）**：方法不引入任何可训练参数，直接对已有视觉特征进行操作。
- **Prefill FLOPs**：LLM 首次处理输入 Token 时的浮点运算量，反映推理计算开销。
- **TTFT（Time to First Token）**：从输入到输出第一个 Token 的时间，反映首 Token 延迟。

---

## 可复现要素
- **数据集**：MVBench、EgoSchema、LongVideoBench、VideoMME（均为公开基准）。
- **代码**：已开源 → https://github.com/xiaozhang79/SimpleCluster
- **模型权重**：使用公开权重 LLaVA-OneVision-7B、LLaVA-Video-7B、InternVL3-2B。
- **关键超参**：3D RoPE 频率基底 $\theta = 10{,}000$；无其他可调超参。
- **硬件**：NVIDIA A800 80GB GPU；评估工具 LMMs-Eval。
- **采样帧数**：LLaVA-OV/InternVL3 用 32 帧，LLaVA-Video 用 64 帧。

---
