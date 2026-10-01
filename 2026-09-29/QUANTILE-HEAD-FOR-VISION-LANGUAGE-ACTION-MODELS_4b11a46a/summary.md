---
title: "QUANTILE-HEAD-FOR-VISION-LANGUAGE-ACTION-MODELS"
source: https://arxiv.org/pdf/2609.34061v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:55:53"
field: "具身智能/视觉-语言-动作模型"
keywords: ["VLA", "quantile head", "flow matching", "pinball loss", "robot control", "distributional action prediction"]
innovations: ["统一回归与流匹配目标并推导分位数监督（Theorem 1）", "提出 Quantile Head：单次前向输出有序边缘分位数（中位数+正间隙）", "证明辅助分位数监督降低中位数更新方差（Theorem 2）"]
benchmarks: ["LIBERO", "LIBERO-Plus", "LIBERO-Pro"]
---

# 论文速读：QUANTILE-HEAD-FOR-VISION-LANGUAGE-ACTION-MODELS

## 一句话总结
本文提出 **Quantile Head**，将 VLA 动作头的点回归与流匹配统一到一个目标框架下，推导出分位数监督目标，使模型在一次前向传播中预测有序的边缘动作分位数（中位数 + 正间隙），默认使用确定性中位数解码即达 LIBERO 99.3%，并在 LIBERO-Plus / LIBERO-Pro 和两个真实机器人任务上均取得最强结果。

## 研究问题与动机
- **点回归无法建模分布**：OpenVLA 等使用 $L_1/L_2$ 回归头，单次前向即可输出动作，但不显式刻画条件动作分布，无法采样。
- **流匹配采样代价高**：$\pi_0$、$\pi_{0.5}$ 等流匹配头需迭代数值积分并多次调用动作专家，推理开销大。
- **缺乏统一理论视角**：回归（平方动作损失）与流匹配（平方速度损失）看似独立，但未有人从同一参考路径出发将其统一，更未导出可联合训练的分位数目标。
- **辅助分位数对中心估计的价值未被显式利用**：现有工作未系统研究"在监督中位数的同时，辅助分位数能否降低中位数更新的噪声"这一核心机制。

## 核心贡献（创新点）
1. **统一回归与流匹配目标，推导分位数目标（Theorem 1）**：从统一平方目标 $\mathcal{L}_{\nu,\pi}$ 出发，将监督信号从动作值扩展到阈值事件，证明集成 pinball 损失与 CDF 监督等价（相差因子 2），为分位数动作建模提供理论基础。
2. **提出 Quantile Head 架构**：冻结 VLM，插入 16 个可学习 prompt tokens；动作专家输出中位数 $m$ 与正原始间隙 $r_i^\pm$，经 softplus 与累积操作得到 $K=21$ 个有序分位数，单次前向即可完成。
3. **揭示辅助分位数降低中位数更新噪声的机制（Theorem 2）**：在相邻校准分位数、固定间隙、局部校正速度匹配条件下，联合分位数监督的中位数更新方差严格小于仅中位数监督，为联合训练提供理论动机。
4. **设计掩码 pinball 损失（Masked Pinball Loss）**：训练时每样本以 $r=0.1$ 比例随机丢弃部分有效动作坐标的监督标签，跨该坐标所有分位数共享同一掩码，缓解过拟合。
5. **多基准 SOTA**：LIBERO 99.3%、LIBERO-Plus 零样 87.1% / 微调 89.1%、LIBERO-Pro 60.2%，真实机器人平均 75.0%（相对 $\pi_{0.5}$ 提升 +16.5pp）。

## 方法详解
### 3.1 统一目标与分位数推导
- **统一目标**：给定观察 $X$、动作坐标 $Y$、参考噪声 $B \sim \nu(\cdot|X)$、时间 $\alpha \sim \pi$ on $[0,1]$，构造插值路径 $A_\alpha = (1-\alpha)B + \alpha Y$，定义
$$\mathcal{L}_{\nu,\pi}(\theta) = \mathbb{E}\left|f_\theta(X, A_\alpha, \alpha) - (Y - B)\right|^2.$$
  - $B=0, \alpha=0 \Rightarrow$ 平方动作回归。
  - $B \sim \mathcal{N}(0,I)$、采样时间 $\Rightarrow$ 直线路径条件流匹配（CFM）速度监督。
- **从数值目标到阈值事件**：将连续标签 $Y$ 替换为指示变量 $Z_z = \mathbf{1}\{Y \le z\}$，其条件期望 $F_X(z) = \Pr(Y \le z|X)$ 即 CDF。
- **Theorem 1（CDF 监督与 pinball 等价）**：设 $q_\theta$ 关于分位数水平非降，$G_\theta$ 为其诱导 CDF，则
$$\underbrace{\mathbb{E}_{X,Y}\int_\mathbb{R}(G_\theta(z|X) - \mathbf{1}\{Y\le z\})^2 \mathrm{d}z}_{\mathcal{L}_{\text{CDF}}(\theta)} = 2\mathbb{E}_{X,Y}\int_0^1 \rho_\tau(Y - q_\theta(X,\tau)) \mathrm{d}\tau,$$
  其中 $\rho_\tau(e) = \max\{\tau e, (\tau-1)e\}$ 为 pinball 损失。

### 3.2 Quantile Head 架构
- **冻结 VLM + 可学习 prompt**：SigLIP 图像嵌入 + 文本指令 + 离散化机器人状态（256 箱）拼成观察前缀；16 个 prompt tokens 插入到 action tokens 之前，通过块自注意力读取 VLM 特征。
- **有序分位数构造**：对每个动作坐标，两条线性投影分别预测中位数 $m$ 与原始间隙 $r_i^-, r_i^+$，经 softplus 与缩放 $\delta_0=0.02$ 转成正间隙：
$$g_i^\pm = \frac{\delta_0}{\log 2}\text{softplus}(r_i^\pm), \quad q_c = m, \quad q_{c-j} = m - \sum_{i=1}^j g_i^-,\quad q_{c+j} = m + \sum_{i=1}^j g_i^+.$$
  $K=21$ 级 $\tau_k = 0.025 + 0.95k/20$，左右间隙独立学习以支持非对称分布。
- **正间隙保证有序性**：softplus 确保不相交；中位数解码时直接取 $q_c=m$，跨步拼出完整 action chunk 后反归一化。

### 3.3 掩码 pinball 损失
- 每样本随机保留 $\min(\lceil r N_b\rceil, \max(N_b-1,0))$ 个有效坐标（$r=0.1$），掩码 $M_{bhd}$ 跨该坐标所有 $K$ 分位数共享：
$$\mathcal{L} = \frac{1}{B}\sum_b \frac{\sum_{h,d,k} M_{bhd}\,\rho_{\tau_k}(a_{bhd} - q_{bhd k})}{K\max(1,\sum_{h,d} M_{bhd})}.$$
   padded 坐标不参与损失；评估时关闭随机丢弃。

### Theorem 2（中位数更新噪声降低）
- 在校准分位数关于 1/2 对称且充分靠近、固定间隙、局部均值校正速率匹配的假设下，联合分位数监督的中位数更新方差严格小于仅中位数监督：$\text{Var}(\Delta m_Q) < \text{Var}(\Delta m_M)$。
- 证明思路：在 $m^*$ 处计算梯度方差 $B_Q = 1/4 - C_t \varepsilon$（$C_t>0$），校正斜率 $a_Q = f_0 + O(\varepsilon^2)$，归一化噪声系数 $V_Q/V_M = 1 - 4C_t \varepsilon + O(\varepsilon^2) < 1$。

## 实验与结果
- **基准**：LIBERO（Spatial/Object/Goal/Long 四套件）、LIBERO-Plus（10,030 扰动实例）、LIBERO-Pro（对象/位置/语义/任务扰动）、两个真实机器人任务。
- **主要结果**：
  - LIBERO 平均成功率 **99.3%**（最高，较次优 98.9% 的 InternVLA-A1.5 提升 +0.4pp）。
  - LIBERO-Plus 零样 **87.1%**（+1.4pp over $\pi_{0.5}$ 85.7%）、微调 **89.1%**（+5.0pp）。
  - LIBERO-Pro 平均 **60.2%**（+6.9pp over $\pi_{0.5}$ 53.3%），其中 Object Pos 31.3%、Sem Task 26.0% 显著提升。
  - 真实机器人平均 **75.0%**（$\pi_{0.5}$ 58.5%，**+16.5pp**），"黄盘放苹果" 31→52（+21pp）、"蓝盘取方块" 86→98（+12pp）。
- **匹配消融（Table 4）**：
  - Quantile Head 99.3% vs $L_2$ 96.0%、$L_1$ 97.3%、Flow Matching 96.3%。
  - 平均回合时间从 $L_1$ 的 8.13s 降至 **5.88s（-27.7%）**。
  - Median-Only 97.5%、Detached Anchor 98.7%、Symmetric Quantiles 97.9%、Noise Query 93.8%、VLM Query 92.8%，均低于全量头。
  - 联合分位数监督在所有 8 组 prompt/掩码配置下均优于仅中位数监督（97.5→99.3%）。
- **采样策略**：LIBERO-Long 上窗 $[0.4, 0.6]$ 采样得 99.6%（超中位数 98.6% 1.0pp）；控制场景下密度加权采样 95% vs 均匀 92% vs 中位数 100%（无障碍），有障碍时中位数 0%、密度加权 13%、均匀 6%。

## 相关工作脉络
1. **回归类动作头**：OpenVLA-OFT（Kim et al., 2025）、$L_1/L_2$ 直接动作预测——本文 Quantile Head 在其基础上增加显式分布表示与可选采样。
2. **流匹配类动作头**：$\pi_0$（Black et al., 2024）、$\pi_{0.5}$、SnapFlow（Luan et al., 2026）、Let It Be Simple（Chen et al., 2026b）——本文用一次前向的分位数输出替代迭代积分，推理更高效。
3. **分位数学习**：Richter & Wattenhofer（2019）优势加权分位数回归、OrderFusion（Yu et al., 2026）中位数 + 正间隙有序输出——本文将其移植到 VLA 连续动作控制并给出噪声降低理论。
4. **VLA 中的不确定性**：ReconVLA（Chen et al., 2026a）用 conformal 校准误差分位数排序候选——本文直接训练分位数 head 并联合监督中位数，无需后验校准。
5. **统一回归/流匹配视角**：本文从单一目标 $\mathcal{L}_{\nu,\pi}$ 同时回收回归与 CFM，之前的 VLA 工作未给出此类统一框架。

## 局限性与未来方向
- **采样策略依赖任务**：有障碍控制场景下，即使最优采样策略也仅 15%（窗 $[0.2, 0.8]$），确定性中位数反而 0%，说明分布建模未必能弥补示范不足。
- **理论结果为局部渐近性质**：Theorem 2 要求分位数充分靠近中位数、固定间隙、校正速率匹配，未覆盖同时更新间隙与共享网络参数的全量训练情形。
- **有限分位数网格不表征完整分布**：$K=21$ 仅监督 $[0.025, 0.975]$ 区间内的 21 个分位数，尾部与中间未被显式学习。
- **真实机器人成功仍有限**："放苹果"任务仅 52%，提示分布式建模在低数据区间的鲁棒性有待提升。
- **未来方向**：系统化推理时采样策略选择、自适应分位数网格、扩展到联合动作依赖建模（当前仅边缘分位数）。

## 研究启发与可借鉴点
1. **统一目标框架迁移**：回归与生成式方法可通过参考路径统一于单一平方目标，该视角可扩展至其他决策/控制模型（如世界模型的动作头设计）。
2. **掩码监督正则化**：坐标级随机丢弃标签并在同坐标分位数间共享掩码，是一种轻量且有效的正则手段，可尝试迁移至多任务/多模态预测。
3. **分位数 → 中位数的噪声降低机制**：Theorem 2 提供的"辅助监督降低中心估计方差"思路可用于任意需稳定中心预测的场景（如强化学习的价值函数、概率预测的点估计）。
4. **正间隙有序输出参数化**：softplus + 累积构造有序分位数的方式无需排序约束优化，计算稳定，可复用于时序预测、电价/风向等有序输出任务。
5. **匹配消融的价值**：本文在同一 $\pi_{0.5}$ checkpoint 上比较 $L_1/L_2$/Flow Matching/Quantile，消除了预训练差异带来的混淆，这种控制变量的消融范式值得效仿。

## 关键术语表
- **Vision-Language-Action (VLA) 模型**：将预训练视觉-语言模型（VLM）与动作头结合，实现从多模态观察到机器人控制的端到端策略。
- **Flow Matching（流匹配）**：学习从噪声到动作的连续速度场，通过数值积分生成动作样本；比扩散更简洁。
- **Pinball Loss**：分位数回归的损失函数 $\rho_\tau(e)=\max\{\tau e,(\tau-1)e\}$，对下预测赋予权重 $\tau$、对上预测赋予 $1-\tau$。
- **Quantile Head**：本文提出的动作头，单次前向输出中位数 + 正间隙，累积得到有序边缘分位数。
- **Masked Pinball Loss**：在 pinball 损失基础上随机丢弃部分动作坐标监督，掩码跨坐标所有分位数共享。
- **LIBERO / LIBERO-Plus / LIBERO-Pro**：三连机器人操作基准，分别评估知识迁移、七类扰动鲁棒性与对象/位置/语义/任务变化下的泛化。
- **Condition Flow Matching (CFM)**：条件流匹配，以当前动作与参考噪声的插值路径为目标学习速度场的变体。
- **Deterministic Median Decoding**：默认推理策略，直接取预测分位数序列的中位数（$\tau=0.5$）作为动作输出。

## 可复现要素
- **数据集**：LIBERO（公开）、LIBERO-Plus（公开）、LIBERO-Pro（公开）；真实机器人数据论文未公开，平台为 AgileX Piper 臂。
- **代码**：已开源，见 https://github.com/xwangrs/Quantile-Head-for-VLA。
- **关键超参**：VLM 冻结；prompt tokens $N_p=16$；分位数级数 $K=21$（$\tau_k=0.025+0.95k/20$）；间隙缩放 $\delta_0=0.02$；掩码比例 $r=0.1$；AdamW，prompt lr=$10^{-4}$（无 weight decay），其余 $2.5\times10^{-5}$ + wd=0.01；cosine 调度，LIBERO 200 步 warmup，LIBERO-Plus 666 步 warmup；batch size=128，双 GPU。
