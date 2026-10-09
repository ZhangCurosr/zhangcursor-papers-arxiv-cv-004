---
title: "Skeleton-Guided-Progressive-Test-Time-Adaptation-for-Thin-Cu"
source: https://arxiv.org/pdf/2610.11104v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 09:56:23"
field: "医学影像/遥感细结构分割与测试时自适应"
keywords: ["test-time adaptation", "curvilinear segmentation", "batch normalization", "skeleton recall", "cross-modality domain shift", "topology preservation"]
innovations: ["ProgBN: 基于样本数的渐进BN统计融合调度，解决BN统计-仿射参数阶段依赖问题", "CSR: 多视图共识膨胀骨架自监督召回损失，仅更新BN仿射参数以保留连通性", "统一框架分离适配BN统计（无梯度）与仿射参数（有梯度），跨模态场景提升最显著"]
benchmarks: ["DRIVE", "STARE", "CHASEDB", "ROSE1", "OCTA3mm", "OCTA6mm", "RECOVERY-FA19", "DeepGlobe", "Massachusetts Roads (MR)", "CNDS"]
---

# 论文速读：Skeleton-Guided-Progressive-Test-Time-Adaptation-for-Thin-Cu

## 一句话总结
本文针对薄曲线结构在跨模态域偏移下拓扑易断裂的问题，提出无源的测试时自适应框架 SGP-TTA，通过渐进式 BN 统计校准（ProgBN）与基于多视图共识骨架的结构损失（CSR）协同适配，在视网膜血管与道路分割任务上显著提升了 Dice 与 clDice（连通性）指标，跨模态场景增益最大。

## 研究问题与动机
1. **薄曲线结构的结构性脆弱**：微细、高分支的血管/道路等曲线结构对局部误分类极度敏感，微小像素误差即可切断全局连通性，破坏下游应用（如血管量化、道路提取）。
2. **现有 TTA 方法与薄结构不匹配**：现有 TTA 仅适配 BN 统计或依赖置信度（熵最小化），前者无法约束连通性，后者在前景-背景极度不平衡下倾向于"抑制弱分支"而非"确认连通"，结构目标缺失。
3. **BN 统计融合比例是阶段依赖的**：早期预测不稳定需依赖源统计，后期仿射参数已适配后仍需继续向目标偏移；固定混合系数无法随适配阶段动态调整。
4. **跨模态域偏移加剧问题**：成像方式（CFP、OCTA、FA、道路传感器）不同导致外观、对比度、背景差异巨大，已有方法在跨模态下常退化甚至低于源模型。

## 核心贡献（创新点）
1. **识别 BN 统计融合的阶段性依赖问题并提出 ProgBN**：利用已处理样本数构建单调递增的 √n/(√n+τ) 调度，无标签、无需离线系数学习即可动态平衡源/目标统计；与固定系数/动量更新方法本质不同，它是确定性样本数调度。
2. **提出共识骨架召回（CSR）自监督结构目标**：通过 K=6 几何变换对齐多视图预测构建 MVC，提取膨胀骨架作为自监督结构标签，仅更新 BN 仿射参数；与基于伪标签/伪断点的结构方法（如 TopoTTA）不同，CSR 不依赖 ground-truth 骨架且强化的是多视图共识而非模型自信区域。
3. **统一 SGP-TTA 框架分离适配 BN 两个组件**：统计通过 ProgBN 无梯度重校准，仿射参数通过 CSR+熵的单步梯度更新，分别解决统计失配与连通性约束；与一次性更新全部参数或仅更新 BN 统计的方法形成对比。
4. **系统性实验验证跨模态显著增益**：在 12 个视网膜/道路迁移设置中，clDice 全部第一、Dice 10/12 第一；跨模态场景提升幅度最大，证明统计校准与结构引导缺一不可。

## 方法详解
**整体架构**：冻结卷积权重与源 BN 运行统计，仅在 BN 层进行两类适配。
- **ProgBN（统计重校准，无梯度）**：对第 n 张目标图，计算当前特征图空间统计 $(\mu_t, \sigma^2_t)$，与冻结源统计 $(\mu_s, \sigma^2_s)$ 按 $\alpha(n)=\frac{\sqrt{n}}{\sqrt{n}+\tau}$ 线性插值，得到混合统计 $\tilde\mu_n, \tilde\sigma^2_n$，再经 BN 归一化公式输出。目标统计每次单图估计后丢弃，不累积。
- **CSR（仿射参数更新，有梯度）**：对 $K$ 个可逆变换 $T_k$（恒等、水平/垂直翻转、90°/180°/270°旋转），用当前模型 $f_n$ 分别预测，逆变换回原坐标得 $p^{(k)}_n$，平均得 MVC $\bar p_n$。对 $\mathbf{1}[\bar p_n\geq\delta]$ 做骨架化后再膨胀半径 $r$ 得目标骨架 $\tilde s_n$（stop-gradient）。CSR 损失为 recall-only：$\mathcal{L}_{CSR}=1-\frac{1}{K}\sum_k\frac{\sum_i\tilde s_{n,i}p^{(k)}_{n,i}}{\sum_i\tilde s_{n,i}+\varepsilon}$。
- **总体损失**：$\mathcal{L}_{SGP-TTA}=\lambda_{CSR}\mathcal{L}_{CSR}+\lambda_{ent}\mathcal{L}_{ent}$，每图单步 Adam 更新 BN 仿射参数 $(\gamma,\beta)$。
- **超参**：$\delta=0.5$，$K=6$，$r=2$，$\lambda_{CSR}=\lambda_{ent}=1$，学习率 $1\times10^{-3}$；视网膜 $\tau=1$，道路 $\tau=5$。

## 实验与结果
- **数据集**：视网膜 CFP（DRIVE/STARE/CHASEDB）、OCTA（ROSE1/OCTA3mm/OCTA6mm）、FA（RECOVERY-FA19）；道路（DeepGlobe/MR/CNDS）。评价指标：Dice + clDice。
- **最强结果**：所有 12 个迁移设置中 clDice 均第一；Dice 10/12 第一。跨模态最大提升：DRIVE → OCTA3mm，SGP-TTA 达 Dice 44.83 / clDice 46.41，最佳基线仅 Dice 32.50 / clDice 34.01（+12.33 Dice）。OCTA3mm → ROSE1：Dice 58.72 vs 基线 50.19（+8.53）。
- **Road 场景**：CNDS 源模型已达 Dice 85.56，SGP-TTA 保持 85.27 且 clDice 从 93.39 升至 94.30；MR 上 Dice 40.80（次优 42.24）但 clDice 54.84 第一。
- **消融**：ProgBN 贡献最大；无 CSR 仅加熵会严重退化（跨模态 Dice 从 47.84 降至 35.24）；MVC 单独增益小，与 ProgBN 协同有效。

## 相关工作脉络
1. **Curvilinear Segmentation（骨架召回类）**：Skeleton Recall Loss（Kirchhof et al., 2024）与 clDice（Shit et al., 2021）引入结构先验，但依赖监督或仅训练时可用；本文在无标注流式目标下自监督生成骨架目标。
2. **TopoTTA（Zhou et al., 2025）**：同类拓扑感知 TTA，用伪断点伪标签作结构信号；本文 CSR 基于多视图共识而非模型自身预测扰动，且在跨模态下更稳定（TopoTTA 在 OCTA 迁移上低于源模型）。
3. **BN 统计适配 TTA**：AdaptiveBN、α-BN、MixNormBN、TTN、MemBN 等方法或固定系数、或动量更新、或积累统计；本文 ProgBN 用样本数单调调度，不累积、无离线系数学习。
4. **Cross-Modality UDA**：Synergistic alignment（Chen et al., 2019/2020）、Diffuseg（Zhang et al., 2025）需离线目标数据与重训练；本文 SGP-TTA 为 truly source-free 流式适应。
5. **Entropy-based TTA**：TENT、CoTTA、SAR、EATA 等仅优化置信度，在薄结构前背景极度不平衡下倾向于压制弱分支而非保持连通；本文加入结构约束弥补此缺陷。

## 局限性与未来方向
- **计算开销**：每图需 6 视图前向与骨架化，比纯熵基线慢；作者承认速度劣势但在合理范围内。
- **K 值与 $r$ 为固定超参**：不同任务可能需要调整，未讨论自适应策略。
- **仅适配 BN 层**：扩展至全部参数或 Adapter 模块的效果未验证。
- **单图流式假设**：实际部署可能遇到批次输入或重复样本，未研究非严格单流场景。
- **仅 binary segmentation**：未扩展到 multi-class 细结构（如多血管类别）。

## 研究启发与可借鉴点
1. **BN 统计-仿射参数分离适配思路可迁移**：任何域偏移敏感的归一化网络均可尝试将此分离范式推广至 LayerNorm、InstanceNorm 等。
2. **样本数作为自适应进度代理变量**：无需置信度/标签的确定性调度在流式推理中具有通用性，可借鉴于其他 TTA 场景。
3. **多视图共识骨架作为自监督结构目标的构造方式**：几何对称变换（翻转/旋转）的组合提取骨架共识，适用于任何具有近似对称先验的细结构任务（管道、裂纹、纤维等）。
4. **消融中"去掉 CSR 仅用熵会严重退化"的发现**：提醒研究者在设计结构敏感任务 TTA 时必须显式引入连通性约束，不可仅依赖熵最小化。
5. **与 backbone 解耦的特性**：SGP-TTA 在 UNet/UNet++/MaNet/LinkNet/FPN 上均有效，说明其适用于多种架构；可与本团队的 foundation model + adapter 方向结合探索。

## 关键术语表
**SGP-TTA**：Skeleton-Guided Progressive Test-Time Adaptation，本文提出的无源测试时自适应框架，联合 ProgBN 与 CSR 适配 BN 统计与仿射参数。
**ProgBN**：Progressive Batch Normalization，按已处理样本数单调递增融合源/目标 BN 统计的自适应方法，无需梯度更新。
**CSR**：Consensus Skeleton Recall，基于多视图预测共识提取膨胀骨架作为自监督结构召回损失，仅更新 BN 仿射参数。
**MVC**：Multi-View Consensus，将 K 个几何变换视角的预测逆变换回原图后取平均的共识预测图。
**clDice**：连通性保持的 Dice 变体损失/指标，同时考量 Dice 相似度与骨架一致性，专门针对管状/曲线结构。
**Cross-modality shift**：成像模态不同（如 CFP vs OCTA vs FA）导致的领域偏移，特征分布差异远大于同模态内域偏移。
**TENT**：Test-time Entropy Minimization，最早的全参数熵最小化 TTA 方法之一。
**TopoTTA**：Topology-Enhanced TTA，另一种针对管状结构的拓扑感知测试时自适应方法，利用伪断点伪标签。

## 可复现要素
- **代码/项目页**：https://boa-jang.github.io/SGP-TTA（论文声明）
- **数据集**：公开数据集（DRIVE、STARE、CHASEDB、ROSE1、OCTA500、RECOVERY-FA19、DeepGlobe、MR、CNDS）
- **关键超参**：$\delta=0.5$，$K=6$，$r=2$，$\lambda_{CSR}=\lambda_{ent}=1$，lr=$1\times10^{-3}$，$\tau=1$（视网膜）/ $5$（道路）
- **骨干网络**：UNet，输入 resize 512×512
- **训练损失**：Dice + Cross-Entropy
- **图片极性**：angiography 图像反转使前景极性一致（仅需已知模态对比方向，非目标标注）
- **其他基线参数**：遵循原文实现与各自论文设定
