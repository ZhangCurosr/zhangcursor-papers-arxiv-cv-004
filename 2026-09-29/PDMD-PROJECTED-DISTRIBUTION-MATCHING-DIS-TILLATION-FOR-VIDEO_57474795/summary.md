---
title: "PDMD-PROJECTED-DISTRIBUTION-MATCHING-DIS-TILLATION-FOR-VIDEO"
source: https://arxiv.org/pdf/2609.35768v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:53:14"
field: "视频生成模型蒸馏加速"
keywords: ["视频扩散模型", "分布匹配蒸馏", "少步生成", "critic误差滤波", "正交投影", "视频-音频联合生成"]
innovations: ["提出PDMD，通过正交投影学生-评论者端点残差滤除DMD更新中的critic误差", "证明残差是critic误差的无偏估计，在高维下移除常数比例误差而仅损失趋于零的信号", "一行代码改动即稳定训练并提升Wan2.1和MiniMax-H3的4-NFE生成质量"]
benchmarks: ["VBench", "VideoGen-Eval"]
---

# 论文速读：PDMD-PROJECTED-DISTRIBUTION-MATCHING-DISTILLATION-FOR-VIDEO

## 一句话总结
论文提出 PDMD（Projected Distribution Matching Distillation），通过在 DMD 的学生更新中投影掉与学生-评论者端点残差平行的分量，滤除在线 critic 的估计误差，从而在仅需一行代码改动的情况下显著提升视频扩散模型少步蒸馏的训练稳定性与样本质量。

## 研究问题与动机
- **核心问题**：Distribution Matching Distillation（DMD）可将视频扩散模型的推理函数评估次数（NFE）降至极少步数，但训练过程中样本会渐进出现过饱和和伪影退化（如 MiniMax-H3 上从 500 步开始明显过饱和）。
- **原因定位**：DMD 的学生更新直接依赖在线 critic 的分数，而 critic 存在近似误差且会滞后于变化的学生分布；该误差被反复注入学生更新并随时间累积，导致训练不稳定。
- **已有方案不足**：DMD2 通过增加 critic 更新次数（TTUR）和引入判别器来缓解，但增加了显存开销与优化耦合难度；其他混合方法（rCM、AnyFlow、ADV 等）均需额外的损失项、网络、数据或训练阶段。
- **本文思路**：不引入额外组件，而是利用 DMD 本身已计算的资源——学生-评论者端点残差 $\boldsymbol{r} = \boldsymbol{x}_0^c - \boldsymbol{x}_0^s$——作为 critic 误差的方向估计，将其从更新中投影出去以滤除误差。

## 核心贡献（创新点）
1. **将 DMD 训练不稳定归因于 critic 误差，并识别出学生-评论者端点残差是 critic 误差的无偏估计**：理论证明在固定查询点下 $\mathbb{E}[\boldsymbol{R}|Q=q] = \boldsymbol{e}$（评论者端点误差），区别于任意方向，该残差由构造包含了误差。
2. **提出 PDMD——一行代码的 DMD 修改**：将 DMD 更新信号沿残差方向的平行分量正交投影出去，在无需额外损失、网络、模型前向、数据或训练阶段的前提下滤除 critic 误差；在高维假设下可移除常数比例的 critic 误差，同时仅损失趋于零的理想信号分量。
3. **在 Wan2.1 和 MiniMax-H3 两个视频生成模型上验证了有效性**：Wan2.1-T2V-1.3B 在 4 NFE 下 VBench 总分 83.73，超越匹配 DMD 1.03 分；MiniMax-H3-33B 联合视频-音频生成在 4 NFE 下 VideoGen-Eval 视觉总分 83.17，超过最强蒸馏基线 0.41 分，并在全部六项音频指标上最优；用户研究在所有维度均偏好 PDMD。

## 方法详解
- **DMD 回顾**：给定学生生成干净端点 $\boldsymbol{x}_0^s = G_\theta(\boldsymbol{z}, \boldsymbol{c})$，按重扩散公式 $\boldsymbol{x}_t = \alpha_t \boldsymbol{x}_0^s + \sigma_t \boldsymbol{\epsilon}$ 重加噪，然后在同一查询点评估在线 critic 分数 $\boldsymbol{s}_{\mathrm{critic}}$ 和冻结教师分数 $\boldsymbol{s}_{\mathrm{teacher}}$，更新信号为 $\boldsymbol{d} = \boldsymbol{s}_{\mathrm{critic}} - \boldsymbol{s}_{\mathrm{teacher}}$，沿梯度方向更新学生参数。
- ** critic 端点转换**：评论者端点可通过线性变换得到 $\boldsymbol{x}_0^c = (\boldsymbol{x}_t + \sigma_t^2 \boldsymbol{s}_{\mathrm{critic}}) / \alpha_t$，无需额外前向传播。
- **残差定义**：学生-评论者端点残差 $\boldsymbol{r} = \boldsymbol{x}_0^c - \boldsymbol{x}_0^s$，在固定查询下是其 critic 误差 $\boldsymbol{e}$ 的无偏估计。
- **正交投影更新**：定义投影矩阵 $P_r = \boldsymbol{r}\boldsymbol{r}^\mathsf{T} / \|\boldsymbol{r}\|^2$，投影后更新信号为 $\boldsymbol{d}_\perp = P_r^\perp \boldsymbol{d} = \boldsymbol{d} - \frac{\langle \boldsymbol{d}, \boldsymbol{r} \rangle}{\|\boldsymbol{r}\|^2} \boldsymbol{r}$；当 $\boldsymbol{r} = \boldsymbol{0}$ 时保持原更新不变。
- **核心算法（一行改动）**：在 DMD 学生步骤的最后，将原始更新 $\boldsymbol{d}$ 替换为 $\boldsymbol{d}_\perp$ 即可，无需其他修改。
- **理论保证**：在高维集中性和弱对齐假设下，$\gamma_e = \Theta_\mathbb{P}(1)$（移除常数比例的 critic 误差能量），而 $\gamma_s = O_\mathbb{P}(d_{\mathrm{eff}}^{-1})$（理想信号损失趋于零）；当条件噪声协方差迹等于误差能量时，至少移除 50% 的 critic 误差能量。

## 实验与结果
- **数据集与基线**：Wan2.1-T2V-1.3B（944 AnyFlow-augmented VBench prompts，480p，5 seeds）；MiniMax-H3-33B 联合视频-音频生成（387 VideoGen-Eval prompts，544p）。基线包括 DMD†、DMD2†、rCM、ADV、AnyFlow、H3 Turbo LoRA 等。
- **Wan2.1 结果（4 NFE，VBench）**：PDMD 总分 83.73，超越 AnyFlow（83.54）、50 步教师（83.06）、匹配 DMD†（82.70）和 DMD2†（83.44）；动态度 89.72 高于 DMD2† 的 76.67；DMD† 在 1500 步后退化至 56.09（7500 步），而 PDMD 在 5000-10000 步稳定保持在 83.5+。用户研究中 PDMD 在视觉质量和运动质量上全面胜出。
- **MiniMax-H3 结果（4 NFE，VideoGen-Eval）**：PDMD 视觉总分 83.17，超越 DMD†（82.76）和 AnyFlow†（81.97）；在全部六项音频指标（PQ、CE、CU、IS、IB、DeSync）上均获最优；用户研究在所有维度均偏好 PDMD。
- **2D 验证**：八高斯环任务上 PDMD 的 energy distance 达 0.0080（DMD 为 0.0273），on-mode fraction 达 0.903；双模态滞后实验中 DMD 在 14/20 次运行中坍缩为单模，PDMD 仅 3/20。
- **消融**：六种投影方向对比中仅 PDMD 避免饱和和伪影；随机投影行为类似 DMD；保留平行分量而非去除则快速退化；增大学生学习率 1.25× 仍保持稳定（总分 83.18），证明稳定性来自方向而非单纯步长缩小；1:1 critic 更新比下 PDMD 仍达 82.90 总分，优于所有 4-NFE 基线。

## 相关工作脉络
1. **DMD（Yin et al., 2024b）**：本文基线方法，通过学生-教师分数差最小化反向 KL 散度实现少步蒸馏；本文定位为在其更新信号上做误差滤波，不改变其网络结构。
2. **DMD2（Yin et al., 2024a）**：通过增加 critic 更新次数（TTUR）和引入判别器减少 critic 误差；本文与之本质区别在于不增加任何额外组件，仅修改更新方向。
3. **rCM（Zheng et al., 2026）**：将 DMD 与连续时间一致性训练结合；需要额外的轨迹损失和两阶段训练。
4. **AnyFlow（Gu et al., 2026）**：结合 MeanFlow 风格 flow-map 训练与 on-policy DMD；需要 flow-map 前向和额外训练阶段。
5. **ADV（You et al., 2026）**：在 DMD 基础上增加自适应加权回归损失和时间正则化以对抗过饱和和时序坍塌。
6. **SGMD（Wu et al., 2026b）、SiD（Zhou et al., 2024）**：用半隐式 Fisher 或 stop-gradient Fisher 目标替代反向 KL 公式，均需额外网络评估。

## 局限性与未来方向
- **单步质量仍有限**：1 NFE 下样本仍模糊，质量低于 4 NFE 结果，可能需要额外目标（如判别器损失）来进一步提升。
- **理论保证为条件误差移除**：不保证学生-评论者耦合优化的收敛性；当有用信号与残差对齐时投影可能同时移除信号。
- **多样性验证不足**：实验主要在两个 backbone 的 4 NFE 上进行，未充分验证跨 prompts 和 seeds 的多样性保持。
- **未来方向**：探索有偏但方差更小的 critic 误差估计；将 critic 误差滤波与其他蒸馏方法结合；研究单步场景下残差方向的有效性。

## 研究启发与可借鉴点
1. **"用已有计算做误差滤波"的设计哲学**：不引入额外组件，而是从已有中间量（学生-评论者残差）中提取误差方向信息，以极简方式提升训练稳定性，这一思路可迁移至其他 online critic-based 方法。
2. **正交投影作为误差抑制工具**：将 critic 误差估计方向从更新信号中投影出去，而非简单减法，避免了步长调参敏感性和误差反转风险，该方法论可用于其他评分匹配类蒸馏。
3. **2D  toy experiment 的理论验证设计**：在完全可观测的 2D 多模态目标上直接测量 $\gamma_e$、$\gamma_s$ 和 error ratio，为理论分析提供干净对照，值得在后续工作中复现这种验证范式。
4. **稳定性诊断指标体系**：论文同时报告 energy distance、on-mode fraction、饱和度统计（C*/L*）、更新范数保留率等多维度指标，构建了从理论到感知的完整验证链，对后续蒸馏工作有借鉴价值。

## 关键术语表
- **Distribution Matching Distillation (DMD)**：一种少步蒸馏方法，通过在线 critic 分数与冻结教师分数的差来驱动学生分布匹配，最小化反向 KL 散度。
- **Critic（评论者）**：在线学习的分数网络，用于估计学生分布的重扩散样本在加噪查询点处的得分。
- **Student-Critic Endpoint Residual**：学生端点与评论者端点之差 $\boldsymbol{r} = \boldsymbol{x}_0^c - \boldsymbol{x}_0^s$，在固定查询下是评论者误差的无偏估计。
- **NFE（Number of Function Evaluations）**：推理时所需的函数评估次数，即去噪步数，越少表示生成速度越快。
- **VBench**：面向视频生成模型的综合性评测基准，包含质量、语义和动态度等维度。
- **VideoGen-Eval**：面向视频-音频联合生成的评估系统，使用 agent 自动化评估视频和音频质量。
- **TTUR（Two-TimeScale Update Rule）**：评论者与学生以不同学习率更新的训练策略，DMD2 采用多步 critic 更新来减少 critic 滞后。
- **Energy Distance**：两种样本分布之间的旋转不变二样本距离，零当且仅当分布完全匹配。

## 可复现要素
- **数据集**：Wan2.1 使用 42K captions（无真实视频）；MiniMax-H3 使用 248K VidProM prompts（无真实视频）；评估使用公开 VBench 和 VideoGen-Eval 数据集。
- **代码/权重**：论文声明代码和模型将在 https://pdmd2026.github.io/ 开源，包括 Wan2.1-T2V-1.3B 学生权重和 MiniMax-H3 蒸馏 LoRA。
- **关键超参**：Wan2.1 学生 lr=$4\times10^{-7}$、评论者 lr=$8\times10^{-8}$、1 critic 更新/学生更新、64 H100 GPU/batch=64；MiniMax-H3 学生 lr=$5\times10^{-5}$、评论者 lr=$1\times10^{-5}$、5 critic 更新/学生更新、LoRA rank=128、scaling=128、16 H100 GPU/batch=16。
