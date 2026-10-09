---
title: "PERCEPTION-ACTION-CONTOUR-POLICY-FOR-GENERALIZED-REFERRING-E"
source: https://arxiv.org/pdf/2610.12107v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 09:52:05"
field: "视觉-语言分割"
keywords: ["generalized referring expression segmentation", "vision-language-action", "contour evolution", "reinforcement learning", "optimal transport", "segmentation"]
innovations: ["将 GRES 重构为闭环感知-动作轮廓策略，以可编辑轮廓为显式几何状态", "提出 EASS 双轴状态条件感知：轮廓引导双向边界采样+演化感知的多级语义路由", "提出 DECT-GRPO：尘桶增强熵传输实现 instance-level credit assignment 联合优化 grounding 和几何动作"]
benchmarks: ["gRefCOCO", "RefCOCO", "RefCOCO+", "RefCOCOg"]
---

# 论文速读：PERCEPTION-ACTION-CONTOUR-POLICY-FOR-GENERALIZED-REFERRING-EXPRESSION-SEGMENTATION

## 一句话总结
论文提出 ContourVLA，将广义指代表达分割（GRES）重新建模为闭环视觉-动作过程，以可编辑轮廓作为显式几何状态，通过状态条件感知与连续几何动作迭代优化边界；结合演化感知语义调度（EASS）与带尘桶增强熵传输的 GRPO（DECT-GRPO），在 gRefCOCO 和 RefCOCO 系列上均取得 SOTA。

## 研究问题与动机
1. **GRES 需要动态平衡高层语义与细粒度视觉证据**：指代表达涉及属性、关系、空间、数量等约束，分割头需同时识别目标数量和精确勾勒边界，而现有级联 VLM 架构依赖静态特征接口，难以自适应调整感知策略。
2. **单次 mask 预测缺乏几何反馈路径**：现有方法单遍生成 mask，无法对局部证据反复访问，难以精化细部结构和分离相邻实例。
3. **标准 GRPO 无法处理多实例质量不均**：现有 RL 后训练对 rollout 内所有输出广播相同 advantage，无法区分不同实例的分割质量差异，也无法妥善处理假阳性和漏检目标。
4. **静态特征接口无法适应轮廓演化过程**：轮廓各演化阶段的几何精化需求不同，固定的多层特征读出无法随轮廓状态自适应调整。

## 核心贡献（创新点）
1. **首次将 GRES 重构为主动感知-动作轮廓策略**：以可编辑轮廓为显式几何状态桥接多模态感知与几何动作，区别于以往 VLM-to-mask 级联或静态特征接口方法。
2. **提出 EASS（演化感知语义调度）**：通过轮廓引导的双向边界采样（SVS）实现自适应感受野的空间感知，以及基于演化状态查询的多级语义路由（MSR），替代固定特征读出，本质区别在于感知随轮廓状态动态调整。
3. **提出 DECT-GRPO（带尘桶增强的熵传输 GRPO）**：利用熵正则最优传输建立软预测-目标对应，通过尘桶吸收假阳性和漏检目标，导出 instance-level credit maps 对 grounding tokens 和 contour actions 分别进行实例级调制，区别于标准 GRPO 的单一 rollout-level advantage。

## 方法详解

**整体框架**：ContourVLA 由 V-L Interaction Module (VLIM) 和 Geometric Action Decoder (GAD) 组成，采用两阶段训练（SFT → DECT-GRPO）。

**State-Conditioned Perception-Action Loop**：
- VLIM 自回归生成结构化 grounding 序列 $y_{1:L_y}$，解码出边界框 $b_i$，初始化内切椭圆轮廓 $C_i^0 \in \mathbb{R}^{P_i \times 2}$，点预算 $P_i$ 由框周长决定。
- 每步演化迭代 $k$：EASS 构建空间感知 $\Phi(F, C_i^k, b_i)$ 和语义选择 $\mathcal{U}_i^k$，与实例表征 $V_i$ 组成观测 $O_i^k$，GAD 预测几何动作块 $A_i^k = \pi_\phi(C_i^k, O_i^k)$，通过状态转移函数 $\mathcal{T}$ 更新轮廓 $C_i^{k+1}$。

**EASS — Spatial Visual Sampling (SVS)**：
沿每个轮廓顶点法线双向采样内部/外部特征：
$$\phi_{i,p}^k = \text{Concat}_{j=1}^{J}\left[F(c_{i,p}^k - o_j(n_{i,p}^k \odot b_i^{\text{size}})), F(c_{i,p}^k + o_j(n_{i,p}^k \odot b_i^{\text{size}}))\right]$$
其中偏移 $\{o_j\} = \{1/64, 1/32, 1/16, 1/8\}$，感受野随对象尺度自适应缩放。

**EASS — Multimodal Semantic Routing (MSR)**：
- 语义键：$\kappa_\ell = \text{Norm}_2(f_{\text{key}}(\text{Pool}(H^\ell) + e_\ell))$
- 状态查询：$q_{i}^{k,d} = \text{Norm}_2(f_q(g_i, c_i^k, r^k, d))$，其中七维状态包括框宽高比、面积、点分配、轮廓不规则度、演化进度和位移统计。
- 路由概率：$p_{i,\ell}^{k,d} = \text{Softmax}_\ell(\langle q_i^{k,d}, \kappa_\ell \rangle + \alpha_u B_{k,d,\ell})$，前向硬选择+直传估计器优化。

**GAD 几何动作预测**：
每顶点编码为几何 token（融合空间特征、轮廓坐标、框相对位置编码、循环位置编码），经交叉注意力与实例上下文 $V_i$ 和路由语义 $U_i^{k,d}$ 交互，并行预测每顶点的二维位移均值，bounded by $\delta_{\max}=0.05$。

**DECT-GRPO**：
- 对每组 G=8 个 grounding 采样，GAD 生成连续几何动作轨迹。
- 尘桶增强熵传输：扩展覆盖矩阵 $\overline{Q}_g$ 添加 null 行/列吸收假阳/漏检，求解：
$$P_g^\star = \arg\max_{P \in \mathcal{U}_g}\left[\sum_{i,j} P_{ij}\overline{Q}_{g,ij} - \varepsilon\sum_{i,j} P_{ij}\log P_{ij}\right]$$
- 基于传输计划计算 transport-aware Precision/Recall，Harm mean 得 reward $R_g$。
- Instance-level credit：$e_{g,i} = \sum_j P_{g,ij}^\star Q_{g,ij} / \sum_j P_{g,ij}^\star$，归一化后调制 grounding tokens 和 contour actions 的 advantage：$A_{g,\nu}^b = A_g + \lambda_c|A_g|\mathbf{c}_g^b[\nu]$，$\lambda_c=0.5$。
- 联合目标：$\mathcal{L}_{\text{DECT}} = \frac{1}{G}\sum_g\left[-\frac{1}{L_g}\sum_n \mathcal{S}_{\text{clip}}(\rho_{g,n}^y, A_{g,n}^y) - \frac{\lambda_a}{|\mathcal{D}_g|}\sum_{\xi \in \mathcal{D}_g}\mathcal{S}_{\text{clip}}(\rho_{g,\xi}^a, A_{g,\xi}^a) + \beta_{\text{KL}}K_g\right]$

## 实验与结果

**数据集**：gRefCOCO（GRESE基准）、RefCOCO、RefCOCO+、RefCOCOg（RES基准）；预训练使用 LocateAnything-Data（1.39 亿实例）。

**模型配置**：视觉骨干 MoonViT，语言骨干 Qwen2.5-3B-Instruct，LoRA rank 64，输入 896×896，轮廓点数 64-128，GAD 深度 6 block，K=2 chunks × T=4 steps/chunk。

**主要结果**：
- **gRefCOCO val**：gIoU **83.5%**（+8.7 vs 最强基线 HiMTok-8B 的 72.1），cIoU 72.3%
- **gRefCOCO testA**：gIoU **80.3%**（+2.8），cIoU 78.1%
- **gRefCOCO testB**：gIoU **74.4%**（+2.7），cIoU 71.8%（仅差 HiMTok 0.1）
- **RES 全部 8 个 split**：mIoU 均为 SOTA，RefCOCO+ 提升 2.8-3.2 点，RefCOCO val 提升 1.3-1.8 点
- **消融**：Full EASS 较 baseline 提升 6.5 gIoU / 6.6 cIoU；Full DECT-GRPO 较 SFT 提升 5.9 gIoU / 14.9 F1@0.5

## 相关工作脉络

1. **级联 VLM 分割方法（LISA, GLaMM, GSVA, LIRA, HiMTok）**：依赖固定特征接口和单次 mask 预测，本文的闭环感知-动作框架从根本上替代了这种静态架构。
2. **轮廓演化方法（DeepSnake, PolyFormer）**：仅做几何更新，不随轮廓状态自适应调节多模态感知，本文 EASS 补充了感知层的动态适配。
3. **VLM 策略模型（OpenVLA）**：本文借鉴 VLA 范式但将其应用于 2D 几何 segmentation 而非机器人控制，引入可微轮廓状态连接感知与动作。
4. **RL 用于分割（Seg-Zero, SAM-R1, Seg-R1）**：应用 GRPO 但使用 rollout-level 单一 advantage，本文 DECT-GRPO 通过熵传输导出 instance-level credit，精细化多实例反馈。
5. **实例重要性分析（MCR-GRPO）**：通过边际贡献分析估计实例重要性，本文采用熵正则最优传输建立软对应，能更好处理模糊匹配和假阳性/漏检。

## 局限性与未来方向

1. **当前闭环不包含重定位能力**：初始 missed referent 无法通过后续轮廓更新恢复，扩展动作空间加入重 grounding 和实例增删操作是自然方向。
2. **固定有序顶点轮廓表示的限制**：无法处理高度不连通的区域、孔洞或极细结构，拓扑变化动作或自适应顶点插入可作为互补扩展。
3. **端到端训练需要较大算力**：两阶段训练（SFT 30k 步 + DECT-GRPO 8k 步）加 G=8 rollout 采样，推理时也有累积轮廓延迟（+6.67ms/2.94%）。

## 研究启发与可借鉴点

1. **"几何状态驱动感知"的范式**：将中间几何表示（轮廓）作为显式策略状态来调节多模态特征路由，可迁移至其他需要精化边界的视觉任务（实例分割、医学图像分割）。
2. **熵传输替代硬匹配的 credit assignment**：DECT-GRPO 中利用 Sinkhorn 求解软对应来处理 ambiguity 和 unmatched 情况，思路可推广至任何需要实例级奖励分配的 RL 视觉任务。
3. **双向法线采样 + 尺度自适应感受野**：SVS 在法线方向双向采样边界证据并随对象尺度缩放，比单点采样 richer，可借鉴至任意 contour-based segmentation 方法。
4. **灰度 advantage 调制保留分支均值**：instance credit 经过 branch-wise centering 后不改变 rollout-level advantage 总和，这种 bounded modulation 设计既精细又稳定。

## 关键术语表

**Generalized Referring Expression Segmentation (GRES)**：根据自由文本表达式分割可变数量和类型对象的像素级分割任务，支持无目标表达。

**Vision-Language-Action (VLA) Policy**：将视觉-语言模型扩展为预测连续动作的策略网络，本文将其应用于 2D 轮廓几何动作预测。

**Evolution-Aware Semantic Scheduling (EASS)**：根据轮廓演化状态动态调度空间采样和语义特征选择的双轴感知调度器。

**Spatial Visual Sampling (SVS)**：沿轮廓顶点法线双向采样的自适应边界特征提取机制，感受野随对象尺度缩放。

**Multimodal Semantic Routing (MSR)**：基于七维演化状态查询对 VLIM 多层语义特征进行硬选择+直传优化的路由机制。

**Dustbin-Augmented Entropic Credit Transport GRPO (DECT-GRPO)**：利用带尘桶的熵正则最优传输建立软预测-目标对应，导出 instance-level credit maps 调制 grounding 和 action 分支 advantage 的 RL 训练方法。

**Transport-aware Precision/Recall**：基于熵传输计划而非硬匹配的 soft IoU 加权 precision 和 recall，避免 hard matching 的不连续性。

**Instance-level Advantage Modulation**：将 rollout-level advantage 按实例 credit 进行有界调制（±50%），保留 sign 和分支均值的同时引入实例差异化反馈。

## 可复现要素

- **数据集**：gRefCOCO、RefCOCO、RefCOCO+、RefCOCOg（官方公开）；LocateAnything-Data（公开，1.39 亿实例）
- **代码**：论文未提及开源代码/权重
- **关键超参**：LoRA rank 64, α=128；G=8 rollout groups；K=2 chunks, T=4 steps/chunk；δ_max=0.05；ε=0.05（transport entropy）；λ_c=0.5；λ_a=1.0；β_KL=0.02；clip range c=0.2；温度 [0.65, 0.75]；top-p=0.95；bfloat16
