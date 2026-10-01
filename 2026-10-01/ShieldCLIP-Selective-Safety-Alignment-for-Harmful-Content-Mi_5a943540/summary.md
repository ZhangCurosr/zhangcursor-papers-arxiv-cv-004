---
title: "ShieldCLIP-Selective-Safety-Alignment-for-Harmful-Content-Mi"
source: https://arxiv.org/pdf/2609.39688v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:37:29"
---

# 论文速读：ShieldCLIP-Selective-Safety-Alignment-for-Harmful-Content-Mi

## 一句话总结
提出了 **ShieldCLIP**，首个以模态独立安全状态为条件的选择性安全对齐框架，结合新数据集 **ViSUv2** 与四项条件损失，在显著降低 CLIP 类多模态编码器下游有害输出的同时，有效保留原始嵌入空间的语义 Utility。

## 研究问题与动机
- **标签粒度粗糙导致过度清洗**：现有安全对齐工作（如 Safe-CLIP）将所有生成的图文对统一标记为“不安全”，但实际合成数据中约 **47.7%** 的生成对存在模态不一致（如不安全文本生成安全图像，或反之），统一重定向会误伤良性表征。
- **真实不安全数据难以规模化获取**：受伦理与现实约束，无法大规模收集真实违规图文，现有配对数据集多依赖 LLM/扩散模型合成，且缺乏精细的模态级安全标注。
- **激进重定向破坏共享空间结构**：大幅偏移 unsafe 嵌入会坍塌 CLIP 的语义几何，导致下游检索、生成、零样本分类的 Utility 隐性退化，且常规指标难以充分暴露此类损伤。
- **需要模态感知的条件对齐机制**：应放弃“来源即有害”的假设，转而依据各模态实际观测到的安全状态动态决定保留或重定向，实现安全性与兼容性的平衡。

## 核心贡献（创新点）
1. **模态级条件安全对齐框架**：首次将保留与重定向操作作用于单个模态的安全标签，显式区分 safe-safe、unsafe-unsafe、safe-unsafe 混合对及双向不安全四种情形，避免“一刀切”标注带来的过清洗。
2. **ViSUv2 数据集**：构建 **195k** 图文四元组数据集，为每个生成模态提供独立二值安全标签，涵盖 **578** 个细粒度 NSFW 概念与 **28** 个高层类别，并经人工多数票校验（文本 86%、图像 81% 一致性）。
3. **四元条件损失设计**：提出保留（cosine+InfoNCE）、重定向、混合对专用非对称 InfoNCE 与共保真度项的组合目标，通过超参调度在安全性与下游 Utility 间取得最优折衷。
4. **跨任务系统验证与解耦消融**：在跨模态检索、SD v1.4/SDXL 文生图、LLaVA 图生文三大场景全面评测，并结合自动化多分类器、FID/CLIP-Sim 指标与人工偏好研究；同时通过“同数据不同方法”对照清晰剥离数据质量与方法设计的贡献。

## 方法详解
- **基础设定**：冻结预训练 CLIP 文本/视觉编码器 $\mathcal{T}_0, \mathcal{V}_0$ 作为 oracle 锚点，trainable 编码器 $\mathcal{T}, \mathcal{V}$ 经 LoRA（rank $r=16$）微调，训练满足 $\mathcal{E}(\bar{x}) \approx \mathcal{E}_0(c(\bar{x}))$。
- **保留损失（Preservation）**：对真实安全对 $\mathcal{R}$ 与生成安全对 $\mathbf{T}_s \cap \mathbf{V}_s$，用 $\mathcal{L}_{\cos}$ 拉近 trainable 与 oracle 同输入表征，并用双向 $\mathcal{L}_{\text{nce}}$ 维持跨模态对齐，确保良性内容不被扰动。
- **重定向损失（Redirection）**：对标记为 unsafe 的生成模态（$\mathbf{T}_u$ 或 $\mathbf{V}_u$），用 $\mathcal{L}_{\cos}$ 将其表征拉向对应真实安全样本的 oracle 锚点，削弱 unsafe 关联。
- **混合对处理（Mixed）**：当生成对仅单模态 unsafe 时，冻结安全分支，仅对不安全分支计算跨模态 $\mathcal{L}_{\text{nce}}$，避免安全侧被反向牵引。
- **共保真度损失（Coherence）**：当两模态均 unsafe 时，施加 $\mathcal{L}_{\text{nce}}$ 保持图文对内部语义一致性，防止重定向破坏跨模态结构。
- **总损失**：$\mathcal{L} = 0.1\mathcal{L}_{\text{pres}}^{\text{real}} + 0.1\mathcal{L}_{\text{pres}}^{\text{gen-safe}} + 0.1\mathcal{L}_{\text{redir}} + 0.25\mathcal{L}_{\text{mix}} + 0.25\mathcal{L}_{\text{coh}}$。

## 实验与结果
- **评测设置**：模型
