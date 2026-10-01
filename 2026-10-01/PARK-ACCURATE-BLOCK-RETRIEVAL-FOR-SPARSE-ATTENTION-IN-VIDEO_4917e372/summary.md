---
title: "PARK-ACCURATE-BLOCK-RETRIEVAL-FOR-SPARSE-ATTENTION-IN-VIDEO"
source: https://arxiv.org/pdf/2609.38978v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:37:29"
field: "视频生成高效推理"
keywords: ["sparse attention", "video generation", "diffusion transformer", "block retrieval", "training-free", "inference acceleration"]
innovations: ["提出PAMA保留原始查询消除Softmax前聚合失配误差", "提出QRKC利用查询条件化度量改进键聚类精度", "识别并形式化分析质心基块检索中的两类根本性误差源"]
benchmarks: ["VBench", "HunyuanVideo 720p", "Wan2.1-1.3B", "Wan2.1-14B", "Wan2.2-14B"]
---

# 论文速读：PARK-ACCURATE-BLOCK-RETRIEVAL-FOR-SPARSE-ATTENTION-IN-VIDEO

## 一句话总结
论文提出 PARK，一种免训练的稀疏注意力方法，通过识别并修复现有质心检索中存在的"查询侧聚合失配"与"键侧聚类度量失配"两个根本性误差源，在 HunyuanVideo 和 Wan 系列视频扩散模型上实现了最优的质量-效率权衡，最高端到端加速比达 1.76×。

## 研究问题与动机
- **视频生成中全注意力二次复杂度是瓶颈**：Diffusion Transformers (DiTs) 已成为视频生成的主流架构，但高分辨率长视频涉及的时空 token 数量巨大，3D 全注意力随 token 数二次增长，成为主要计算瓶颈。
- **稀疏注意力依赖准确的块检索**：注意力图本质上稀疏，仅少数 token 交互对输出有显著贡献；基于块检索的稀疏注意力通过仅计算重要块对来降低开销，但检索不准确会同时损害生成质量或造成冗余计算。
- **现有质心估计存在查询侧聚合失配**：许多方法在 Softmax 归一化前将查询块内所有查询平均为单一质心，这一操作忽略了各查询独立的注意力偏好（因 Softmax 非线性），导致块重要性估计偏差；理论分析表明平均 TV 误差比保留原始查询高 47.7–49.0%。
- **现有质心估计存在键侧聚类度量失配**：标准欧氏 K-means 聚类仅最小化键特征空间内的距离，而不保证同一簇内键在当前查询下获得相似的 QK 得分，导致键质心的 log-mass 估计不准确；QRKC 将中心化 log-mass RMSE 降低 16.9–17.2%。

## 核心贡献（创新点）
1. **首次系统识别并理论分析质心基块检索中的两类失配误差**：查询侧聚合失配（M1，Softmax 前平均查询忽略个体偏好）与键侧聚类度量失配（M2，欧氏聚类不反映 QK 得分相似性），并以 Taylor 展开给出了误差的严格上界，现有工作未对此进行形式化分析。
2. **提出 PAMA（Per-Query Attention-Mass Aggregation）**：保留全部原始查询，对每个查询独立计算经 Softmax 归一化的键块注意力分布后再在查询块内平均，消除 M1 引入的近似误差，并用 Triton 实现融合 GPU kernel 将中间 logit 留存于片上寄存器，相比 PyTorch 实现提速 5.56×、减少 99.89% 显存。
3. **提出 QRKC（Query-Response Key Clustering）**：利用当前查询的二阶矩矩阵 $\mathbf{G}_Q = \frac{1}{L}\mathbf{Q}^\top\mathbf{Q}$ 构建查询条件化的键距离度量，通过 Cholesky 分解将原始键映射到低维变换空间后进行标准欧氏 K-means，使聚类目标直接最小化键质心引入的 QK 得分方差，将中心化 log-mass 近似误差降低 16.9–17.2%。
4. **在四个视频生成模型上验证了最优质量-效率权衡**：在 Wan2.2-14B/Wan2.1-1.3B/Wan2.1-14B/HunyuanVideo 上分别实现 1.55×/1.76×/1.53×/1.66× 加速，PSNR/SSIM/LPIPS 均优于所有对比的免训练稀疏注意力基线（SpargeAttn、XAttention、SVG、SVG2、SVOO）。

## 方法详解
**整体框架**：PARK 是一种免训练推理加速方法，包含两个核心模块 PAMA 和 QRKC，以及一个融合的 GPU kernel 实现。

**PAMA（Per-Query Attention-Mass Aggregation）**：
- 对于键块 $\mathcal{K}_v$，先计算其质心 $\mathbf{c}_v = \frac{1}{n_v^K}\sum_{j \in \mathcal{K}_v}\mathbf{k}_j$，对每个原始查询 $\mathbf{q}_i$ 近似其对键块的 log-mass：$\widehat{S}_{iv} = \frac{\mathbf{q}_i^\top \mathbf{c}_v}{\sqrt{d}} + \log n_v^K$（其中 $\log n_v^K$ 补偿键块大小差异）。
- 对每个查询独立做行方向 Softmax：$\widehat{P}_{iv} = \frac{\exp(\widehat{S}_{iv})}{\sum_r \exp(\widehat{S}_{ir})}$。
- 在查询块 $\mathcal{Q}_u$ 内平均得到块重要性估计：$\widehat{W}_{uv}^{\text{PAMA}} = \frac{1}{n_u^Q}\sum_{i \in \mathcal{Q}_u}\widehat{P}_{iv}$。
- 融合 Kernel 设计：将 QK 质心点积、行方向 Softmax、块内平均三阶段融合入单一 GPU kernel，中间 logit 和概率常驻片上寄存器，仅将最终 $N_Q \times N_K$ 重要性矩阵写入 HBM；对于大 $N_K$ 使用两阶段 kernel（先 Pass 1 在线计算 row-wise LogSumExp，Pass 2 重新计算归一化权重），确保显存开销与 $N_K$ 解耦。

**QRKC（Query-Response Key Clustering）**：
- 定义聚类目标：最小化所有查询下替换为质心所致的 QK 得分平方误差之和：$\mathcal{I}_{\text{QRKC}} = \sum_v\sum_{j \in \mathcal{K}_v}\|\mathbf{k}_j - \mathbf{c}_v\|_{\mathbf{G}_Q}^2$，其中 $\mathbf{G}_Q = \frac{1}{L}\mathbf{Q}^\top\mathbf{Q}$ 为查询的二阶矩矩阵。
- 该目标与键质心近似误差的领头项直接相关：$\mathcal{I}_{\text{QRKC}} = 2d\sum_v n_v^K \bar{\epsilon}_{v}^K$。
- 通过 Cholesky 分解 $\overline{\mathbf{G}}_Q = \mathbf{R}\mathbf{R}^\top$（含正则化：$\overline{\mathbf{G}}_Q = (1-\alpha)\mathbf{G}_Q + \alpha s_Q\mathbf{I} + \epsilon s_Q\mathbf{I}$，$\alpha=0.05$），将键变换为 $\widetilde{\mathbf{k}}_j = \mathbf{R}^\top\mathbf{k}_j$，变换后保持特征维度 $d$ 不变。
- 在变换空间上用标准欧氏 K-means 聚类键 token，查询仍用标准欧氏 K-means；聚类和重排结果每 $R=10$ 个去噪步复用一次以削减在线开销。

**检索流程**：PAMA 估计所有查询块-键块对的 $\widehat{W}_{uv}^{\text{PAMA}}$，按 Top-p（$p=0.9$）排序选取关键块，生成稀疏掩码 $\mathbf{M}$，再用 FlashInfer kernel 进行稀疏注意力计算。首 20% 去噪步使用全注意力。

## 实验与结果
- **模型与设置**：在四个开源 DiT 模型上评测——Wan2.1-T2V-1.3B、Wan2.1-T2V-14B、Wan2.2-T2V-14B、HunyuanVideo-T2V-13B；720p 分辨率，50 个去噪步；单卡 NVIDIA A800。查询块数 $N_Q=128$，HunyuanVideo 键块数 $N_K=1024$，Wan 系列 $N_K=512$。
- **对比基线**：SpargeAttn、XAttention、SVG、SVG2、SVOO（均为免训练方法）。
- **核心结果**（Table 2）：

| 模型 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 加速比 |
|------|------|-------|-------|--------|--------|
| Wan2.2-14B | Dense | — | — | — | 1.00× |
| Wan2.2-14B | **PARK** | **19.19** | **0.754** | **0.265** | **1.55×** |
| Wan2.1-1.3B | **PARK** | **27.52** | **0.908** | **0.177** | **1.76×** |
| Wan2.1-14B | **PARK** | **24.01** | **0.875** | **0.186** | **1.53×** |
| HunyuanVideo | **PARK** | **30.69** | **0.936** | **0.166** | **1.66×** |

- PARK 在所有四个模型上 PSNR/SSIM 最高、LPIPS 最低，且加速比均为最佳；相较最强基线 SVOO，在 HunyuanVideo 上 PSNR 提升 0.38dB、LPIPS 降低 0.006、加速比从 1.55× 提升至 1.66×。
- **注意力召回率**（Top-p, $p=0.9$，Table 1）：Full Query + Centroid Key 配置在两种排序方案下均取得各压缩表示中最高的平均召回率（HunyuanVideo: 90.70% / Wan2.1-14B: 91.56%，Clustered 排序）。
- **消融实验**（Table 3）：去掉 PAMA 在 HunyuanVideo 上 PSNR 下降 1.45dB（29.85→28.40），LPIPS 升高 0.037；去掉 QRKC 同样导致质量下降，两模块各自从查询侧和键侧独立贡献；三种变体间延迟差异不足 0.3%。
- **PAMA Kernel 效率**（Table 6）：Triton Full-K 相比 PyTorch 提速 5.56×，峰值显存从 21.09GB 降至 24.17MB（减少 99.89%）；全模型 PAMA 估算总耗时 <1s，占端到端推理时间不到 0.21%。
- **复用策略**（Table 9）：每 10/20/30/40 步重算聚类的质量差异极小（PSNR 29.55→29.85），三条重算策略延迟几乎相同。

## 相关工作脉络
1. **SpargeAttention (Zhang et al., 2025a)**：动态免训练稀疏注意力，通过反对角线打分选择关键块；PARK 与其定位差异在于 PARK 从查询/键两侧同时优化块重要性估计的精度，而非仅改进打分策略。
2. **SVG2 (Yang et al., 2025a)**：对查询和键分别做 Euclidean K-means 聚类并重排序，再用质心估计块重要性；PARK 的关键区别是不对查询做质心压缩（保留原始查询），且键聚类采用查询条件化度量而非欧氏距离。
3. **SVOO (Luo et al., 2026)**：通过双向联合聚类耦合查询-键分区；PARK 采用非对称设计——查询保留原始表示，键用查询响应度量聚类，理论分析更形式化且实现开销更低。
4. **XAttention (Xu et al., 2025)**：基于反对角线分数的块稀疏注意力；PARK 不依赖特定分数模式假设，而是从第一性原理出发推导块重要性估计误差。
5. **VSA (Zhang et al., 2026) / VMoBA (Wu et al., 2026)**：训练型稀疏注意力方法，需微调或引入辅助预测模块；PARK 定位为完全免训练的推理时方法，无需额外训练。
6. **AdaCluster (Tan et al., 2026a)**：自适应查询-键聚类进行 token 重排；与 PARK 共同点是利用聚类做块划分，但 PARK 进一步分析了聚类度量与注意力估计误差之间的关系，并提出查询条件化的变换方法。

## 局限性与未来方向
- **仅针对视频生成扩散模型评估**：论文未将 PARK 应用于文本 LLM 或其他多模态任务，不同任务的注意力模式和解码流程可能要求对 PARK 进行调整。
- **未探索训练友好的变体**：当前方法完全免训练，若允许轻量微调可能进一步提升检索精度。
- **复用策略的步数选择**：虽然实验表明复用策略对质量影响小，但最优的 $R$ 值可能随模型规模和视频长度变化，尚未系统研究。
- **键块数 $N_K$ 增大时的两阶段 kernel 开销**：虽然显存节省显著，但两阶段实现使 QK 计算量近似翻倍，在极端稀疏场景下可能需要权衡。

## 研究启发与可借鉴点
1. **Taylor 展开分析近似误差的思路具有强可迁移性**：论文用 Hessian 矩阵与查询协方差的 Frobenius 内积刻画聚合误差，用二次型刻画键质心 log-mass 误差，这种从一阶近似到二阶误差的形式化分析方法可直接移植到其他注意力压缩场景中。
2. **查询侧保留原始表示、键侧做条件化变换的"非对称设计"思路**：在资源受限的检索场景中，选择误差更大的那一侧保留精度（查询侧因 Softmax 非线性误差更大），对另一侧做近似优化，这一策略值得在其他序列建模任务中探索。
3. **融合 Kernel 的"中间结果常驻片上寄存器"设计**：将 QK 点积、Softmax、平均三阶段融合并避免将 $L \times N_K$ 中间矩阵写入 HBM，这一技巧可推广至其他需要行方向归一化的注意力变体（如 RMSNorm-based 变体）。
4. **Cholesky 分解实现度量变换而不增维**：用 $\mathbf{G}_Q = \mathbf{R}\mathbf{R}^\top$ 将查询条件化距离嵌入原有 $d$ 维空间，避免了映射到 $L$ 维空间的维数爆炸，这一技巧可用于任何需要将二次型度量转化为欧氏距离的聚类场景。
5. **跨步复用稀疏掩码的策略**：利用扩散步间注意力模式稳定性，每 $R$ 步重算一次聚类/掩码，可在几乎不损失质量的前提下大幅降低在线开销，这一模式对其他迭代生成过程亦有参考价值。

## 关键术语表
**Diffusion Transformer (DiT)**：以 Transformer 为骨干的视频/图像生成扩散模型架构，已被证明具有强可扩展性，是当前视频生成的主流范式。
**Block Retrieval（块检索）**：将 token 分组为块，通过估计块重要性分数选择需要计算注意力的关键块对，生成稀疏掩码的过程。
**Attention Recall（注意力召回率）**：衡量稀疏掩码覆盖的全注意力权重比例，定义为 $\frac{\sum_{ij} A_{ij}M_{ij}}{\sum_{ij} A_{ij}}$，越高表示检索越准确。
**Per-Query Attention-Mass Aggregation (PAMA)**：PARK 的核心模块之一，保留每个原始查询独立计算 Softmax 归一化后的键块注意力分布，再在查询块内平均，避免质心近似引入的聚合误差。
**Query-Response Key Clustering (QRKC)**：PARK 的核心模块之二，利用当前查询的二阶矩矩阵 $\mathbf{G}_Q$ 构建查询条件化距离度量，通过 Cholesky 变换后做标准 K-means，使聚类结果反映 QK 得分相似性。
**Log-Mass（对数质量）**：键块内所有键的 QK 得分的 log-sum-exp 值，用于近似块对查询的重要性，比直接平均 QK 得分更准确。
**Top-p Retrieval（Top-p 检索）**：对每个查询块按估计重要性降序选取键块，直至累计重要性达到阈值 $p$（论文中 $p=0.9$）。
**Fused GPU Kernel（融合 GPU Kernel）**：将多个计算阶段融合进单一 CUDA/Triton kernel 以减少 HBM 读写，论文提供了 Full-K 和 Two-pass 两种变体。

## 可复现要素
- **数据集**：VBench 评测集的原始文本提示词；模型输出在 720p 分辨率下评测。
- **代码/权重是否开源**：论文未提及代码开源状态；使用的模型（HunyuanVideo、Wan）为开源模型。
- **关键超参**：$N_Q = 128$，HunyuanVideo 的 $N_K = 1024$，Wan 系列的 $N_K = 512$；去噪步数 50 步，首 20% 使用全注意力；Top-p 阈值 $p = 0.9$；掩码复用间隔 $R = 10$；Cholesky 正则化参数 $\alpha = 0.05$，$s_{\min} = 10^{-12}$。
- **硬件**：单卡 NVIDIA A800 GPU。
- **实现细节**：PAMA kernel 基于 Triton 实现，稀疏注意力计算使用 FlashInfer 带动态块大小的 kernel。
