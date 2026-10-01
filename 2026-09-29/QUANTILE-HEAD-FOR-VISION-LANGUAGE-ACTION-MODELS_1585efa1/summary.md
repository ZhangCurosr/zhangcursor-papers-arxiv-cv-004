---
title: "QUANTILE-HEAD-FOR-VISION-LANGUAGE-ACTION-MODELS"
source: https://arxiv.org/pdf/2609.34061v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:55:48"
field: "视觉-语言-动作模型的动作头设计"
keywords: ["VLA", "quantile regression", "flow matching", "robotic manipulation", "action head", "distributional learning", "LIBERO"]
innovations: ["将回归与流匹配统一到单一目标并推导分位数等价形式，提出单次前向有序分位数头", "理论证明联合分位数监督可降低中位数直接更新的梯度方差", "设计掩码 pinball 损失用于分位数头的多输出正则化训练"]
benchmarks: ["LIBERO", "LIBERO-Plus", "LIBERO-Pro", "Real-Robot Manipulation"]
---

# 论文速读：QUANTILE-HEAD-FOR-VISION-LANGUAGE-ACTION-MODELS

## 一句话总结
本文提出一种 Quantile Head 架构，通过在单次前向传播中联合预测有序边际动作分位数（中位数 + 正向间隙），将回归的单次预测效率与流匹配（flow matching）的分布建模能力统一起来，在 LIBERO 系列基准和真实机器人任务上取得最优结果。

## 研究问题与动机
1. **回归头仅提供点估计**：现有 VLA 模型（如 OpenVLA、$\pi_{0.5}$）的动作头使用 $L_1$/$L_2$ 回归输出单一动作值，无法显式表征条件动作分布，难以支持采样探索。
2. **流匹配头推理成本高**：$\pi_0$ 等模型通过流匹配学习速度场，标准采样需多次迭代积分与重复模型评估，增加部署开销；一次性流匹配方法（如 SnapFlow、Let It Be Simple）在长 horizon 任务上优势衰减。
3. **两种范式缺乏统一理论桥梁**：回归与流匹配在损失层面存在可统一的推广形式，但如何在此基础上同时获得单次预测效率与显式分布表征尚未被探索。
4. **辅助分位数是否能改善中位数学习**：理论上缺乏对联合分位数监督能否降低中位数直接更新噪声的严格分析。

## 核心贡献（创新点）
1. **将回归与流匹配统一到单一目标框架，并推导其分位数等价形式（Theorem 1）**：证明积分 pinball loss 等价于 CDF 监督，为用有限分位数集表征动作条件分布提供理论依据；与已有工作本质区别在于从统一的平方损失框架出发，而非直接引入分位数回归。
2. **提出 Quantile Head 架构，单次前向输出有序边际分位数**：通过中位数 + 正间隙参数化（softplus 确保非交叉），在 K=21 个分位水平上一致输出，支持默认中位数解码与可选采样两种推理模式；区别于回归（单点输出）与流匹配（迭代采样）两种传统范式。
3. **理论证明联合分位数监督可降低中位数更新方差（Theorem 2）**：在邻近分位数校准、间隙固定、局部校正速率匹配的假设下，证明 $\mathrm{Var}(\Delta m_Q) < \mathrm{Var}(\Delta m_M)$；与已有工作本质区别在于聚焦辅助分位数对"默认使用中位数"这一决策路径的局部噪声缓解机制。
4. **设计掩码 pinball 损失（masked pinball loss）**：每批次随机遮蔽 10% 的合法动作维度标签（掩码跨所有 K 个分位数共享），起到正则化效果；这是针对分位数头特性定制的训练技巧，不同于标准 dropout 或 label dropout。

## 方法详解

**理论统一框架**：设观测 $X$（视觉 + 语言 + 本体感知），动作 chunk $A \in \mathbb{R}^{H \times D}$。定义参考路径 $A_\alpha = (1-\alpha)B + \alpha Y$，其中 $B \perp Y \mid X$ 为参考噪声，$\alpha \sim \pi$ 为时间。统一目标为：
$$\mathcal{L}_{\nu,\pi}(\theta) = \mathbb{E}|f_\theta(X, A_\alpha, \alpha) - (Y - B)|^2$$
取 $B=0, \alpha=0$ 还原为平方回归；取高斯参考与采样时间还原为 CFM 速度监督。

**Theorem 1（分位数监督等价性）**：将监督目标从数值标签扩展为阈值事件 $Z_z = \mathbf{1}\{Y \le z\}$，则有：
$$\mathbb{E}_{X,Y}\int_{\mathbb{R}}(G_\theta(z|X) - \mathbf{1}\{Y \le z\})^2 dz = 2\,\mathbb{E}_{X,Y}\int_0^1 \rho_\tau(Y - q_\theta(X,\tau))\,d\tau$$
即积分 pinball loss 与 CDF 监督等价（相差因子 2）。

**Theorem 2（中位数更新方差降低）**：在校准分位数对称分布于 0.5 附近、间隙固定、局部校正速率匹配的假设下，联合分位数监督的中位数更新方差严格小于仅监督中位数的方差：$\mathrm{Var}(\Delta m_Q) < \mathrm{Var}(\Delta m_M)$。

**Quantile Head 架构**：冻结预训练 VLM，插入 $N_p=16$ 个可学习 prompt tokens；action expert 提取特征后，两个线性投影分别输出中位数 $m$ 和原始间隙 $r_i^-, r_i^+$，经 softplus 缩放为正的累积间隙：
$$g_i^\pm = \frac{\delta_0}{\log 2}\mathrm{softplus}(r_i^\pm), \quad q_c = m, \quad q_{c-j} = m - \sum_{i=1}^j g_i^-, \quad q_{c+j} = m + \sum_{i=1}^j g_i^+$$
使用 $K=21$ 个分位水平 $\tau_k = 0.025 + 0.95k/20$，$\delta_0=0.02$。

**掩码 pinball 损失**：pinball loss $\rho_\tau(e) = \max\{\tau e, (\tau-1)e\}$，每批次随机遮蔽 $r=0.1$ 比例的合法动作维度，掩码跨所有 K 个分位数共享：
$$\mathcal{L} = \frac{1}{B}\sum_{b=1}^B \frac{\sum_{h,d,k} M_{bhd}\,\rho_{\tau_k}(e_{bhd k})}{K \cdot \max(1, \sum_{h,d} M_{bhd})}$$

## 实验与结果

**数据集与基准**：LIBERO（Spatial/Object/Goal/Long 四套件）、LIBERO-Plus（10,030 个扰动实例）、LIBERO-Pro（对象/位置/语义/任务扰动），以及两个真实机器人操作任务（黄盘放苹果、蓝盘取方柱）。

**主要结果（中位数解码）**：
- **LIBERO**：平均成功率 **99.3%**，超越最强基线（InternVLA-A1.5 的 98.9%）0.4 pp；Long 套件达 98.6%，较 $L_1$ 回归提升 3.6 pp。
- **LIBERO-Plus（zero-shot）**：**87.1%**，超越 $\pi_{0.5}$（85.7%）1.4 pp；**SFT** 达 **89.1%**，超越 MemoryVLA++（82.7%）6.4 pp。
- **LIBERO-Pro**：**60.2%**，超越 $\pi_{0.5}$（53.3%）6.9 pp。
- **真实机器人**：平均成功率 **75.0%**，较全微调的 $\pi_{0.5}$（58.5%）提升 **+16.5 pp**（放苹果 +21 pp，取方柱 +12 pp）。
- **平均Episode时间**：5.88s，较 $L_1$ 基线（8.13s）缩短 **27.7%**。

**最强结果**：LIBERO 平均成功率 99.3%（历史最优之一）；真实机器人任务平均成功率 75.0%（显著超越对比方法）。

## 相关工作脉络
1. **OpenVLA / OpenVLA-OFT（Kim et al., 2024, 2025）**：使用 $L_1$ 回归头 + 并行 action-chunk 解码的直接回归方案；本文将其扩展为联合监督的分位数输出，保留单次前向效率同时提供分布表征。
2. **$\pi_0$ / $\pi_{0.5}$（Black et al., 2024, 2025）**：基于流匹配速度场的 VLA；本文在理论层面证明回归可作为流匹配的特例，并提出无需迭代采样的替代方案。
3. **Diffusion Policy（Chi et al., 2025）**：通过去噪学习动作序列；本文避免扩散/流匹配的重复模型评估，以单次前向实现等价或更优性能。
4. **ReconVLA（Chen et al., 2026a）**：使用 conformal calibration 的动作误差分位数排序候选；本文的分位数由网络直接联合预测并监督，而非事后校准。
5. **OrderFusion（Yu et al., 2026）**：在电价预测中从中位数锚点 + 非负间隙构造有序输出；本文将其思想迁移到 VLA 动作建模，并补充了方差降低的理论分析。
6. **Advantage-weighted quantile regression（Richter & Wattenhofer, 2019）**：在强化学习中使用分位数回归；本文将其应用于 VLA 的 action head 设计，并与 flow matching 建立统一理论联系。

## 局限性与未来方向
1. **采样策略缺乏系统优化**：当前探索了均匀采样、密度加权采样等简单策略，但在有障碍物场景中成功率仅 6–15%，采样窗口的选择仍待深入分析。
2. **分位数网格的覆盖范围有限**：K=21 个水平仅覆盖 [0.025, 0.975]，尾部行为依赖线性外推，可能在高不确定性场景下估计不准。
3. **仅建模边际分布**：各动作维度的分位数独立预测，未建模动作间的联合依赖结构；Theorem 1 的标量结论直接逐坐标应用。
4. **理论条件的理想化假设**：Theorem 2 要求分位数精确校准、间隙固定、密度光滑，实际训练中这些条件未必严格满足。
5. **复杂避障场景泛化不足**：两演示实验中加入障碍物后，确定性解码（中位数、$L_1$、$L_2$）成功率均为 0%，随机采样仅小幅改善。

## 研究启发与可借鉴点
1. **回归与生成式方法的统一视角**：Theorem 1 将回归、CFM 和分位数监督置于同一目标框架，这一理论工具可用于分析其他生成式动作头的设计空间，值得借鉴到扩散/流匹配变体的理论研究中。
2. **中位数+间隙的参数化技巧**：通过 softplus 确保间隙为正、累积构造有序分位数，避免了交叉分位数问题；该技巧可迁移至任何需要有序输出（如预测区间、分位数回归网络）的任务。
3. **掩码标签监督作为正则化**：按维度随机遮蔽部分标签（跨所有输出共享掩码）的设计，可有效防止模型过度拟合某些动作维度，类似思想可用于多输出回归任务的训练稳定性改进。
4. **联合监督改善中心估计的方差分析思路**：Theorem 2 的"辅助监督降低主估计更新方差"的分析框架，可为多任务/多输出学习中辅助头的梯度贡献提供理论指导。
5. **单次前向 + 可选采样的灵活推理**：默认使用中位数保证低延迟确定性控制，同时保留采样能力应对多模态场景，这一设计模式可作为 VLA 部署的实用参考。

## 关键术语表
**Vision-Language-Action (VLA) 模型**：将预训练视觉-语言模型与动作头结合，直接映射多模态观察到机器人控制动作的端到端策略学习框架。
**Flow Matching**：通过学习从噪声到数据的连续速度场来生成样本的方法，标准采样需数值积分迭代；straight-path CFM 是其常用变体。
**Pinball Loss**：分位数回归的专用损失函数 $\rho_\tau(e)=\max\{\tau e, (\tau-1)e\}$，对下预测和过预测施加不同权重，收敛到条件分位数。
**Ordered Quantile Head**：通过中位数 + 正间隙参数化，在单次前向中输出严格有序的边际分位数集合的神经网络层结构。
**Masked Pinball Loss**：每批次随机遮蔽部分动作维度标签（跨所有分位数共享同一掩码），再计算 pinball loss  averaged 过保留维度的训练损失。
**LIBERO / LIBERO-Plus / LIBERO-Pro**：用于评估 VLA 模型技能学习、跨情景泛化和鲁棒性的仿真基准测试套件，分别侧重知识迁移、多维度扰动鲁棒性和对抗性扰动。

## 可复现要素
- **数据集**：LIBERO（公开）、LIBERO-Plus（公开）、LIBERO-Pro（公开）；真实机器人数据为作者采集（论文未声明公开）。
- **代码**：已开源，地址 https://github.com/xwangrs/Quantile-Head-for-VLA。
- **权重**：基于 $\pi_{0.5}$ 预训练 checkpoint 微调，论文未声明开源微调后权重。
- **关键超参**：分位水平数 K=21，prompt 数 $N_p=16$，遮蔽比例 $r=0.1$，间隙缩放 $\delta_0=0.02$，batch size=128，prompt 学习率 $10^{-4}$，其余参数 $2.5\times10^{-5}$，AdamW 优化器，cosine decay 学习率调度，200 步 warmup（LIBERO-Plus 为 666 步）。
