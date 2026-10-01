---
title: "SOFT-SPATIAL-REASONING"
source: https://arxiv.org/pdf/2609.38717v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:36:32"
field: "视觉语言模型空间推理"
keywords: ["soft thinking", "spatial reasoning", "LVLM", "reinforcement learning", "gradient alignment", "chain-of-thought"]
innovations: ["AdaptSoft步级自适应软度控制器：基于隐藏状态和预测熵动态调节token嵌入混合温度", "梯度对齐学习目标：无需中间推理标注，通过步级策略梯度与参考梯度的一致性分配信用", "Gumbel重参数化Soft-GRPO扩展至LVLM空间推理：推导软步似然比支持后训练优化"]
benchmarks: ["OmniSpatial", "SpatiaLab", "MindCube"]
---

# 论文速读：SOFT SPATIAL REASONING

## 一句话总结
本文提出 Soft Spatial Reasoning，一种面向大视觉语言模型（LVLM）空间推理的后训练框架，通过 AdaptSoft 控制器根据推理状态和预测不确定性自适应调节软思考温度，避免硬思考中的"过早离散化"错误传播，在三个空间推理基准上达到开源模型最高精度。

## 研究问题与动机
1. **核心问题**：LVLM 的空间推理普遍采用硬思考（Hard Thinking），即每条 CoT 步骤都强制选择单个离散 token，当模型对空间解释尚存歧义时，过早 committing 会引发错误传播。
2. **固定软度的局限**：先前的软思考方法采用固定混合温度，但空间推理中不同步骤需要的 softness 不同——某些步骤保留多候选有用，另一些步骤混合冲突空间关系反而会干扰后续推理。
3. **缺乏步级学习信号**：GRPO 等 RL 方法仅提供 rollout 级别的奖赏信号，无法区分每条 CoT 中间步骤对最终答案的贡献差异，难以训练步级软度控制。
4. **与人类认知对比**：人类思考空间问题时无需将每个中间步骤"说出口"，LVLM 可借鉴此特性，在推理过程中维持连续状态而非逐 token 离散化。

## 核心贡献（创新点）
1. **首个面向空间推理的软思考后训练框架**：Soft Spatial Reasoning 首次在 LVLM 空间推理中引入自适应软思考，区别于现有将软性仅置于视觉表征的方法。
2. **AdaptSoft 步级软度控制器**：利用当前隐藏状态与标准化预测熵作为输入，通过 MLP 输出步级温度 τ，实现软硬程度的动态调节，与固定温度软思考形成本质区别。
3. **梯度对齐学习目标（Gradient-Alignment Learning）**：无需中间推理标注或外部评估器，通过比较每步对策略梯度的贡献与参考梯度的一致性，提供步级信用分配信号。
4. **Gumbel 重参数化似然与 Soft-GRPO**：将软 CoT 步骤的对数概率定义为 Gumbel 扰动分数向量上的密度，推导得到可扩展 GRPO 到软思考的似然比公式。
5. **全面实验验证**：在 OmniSpatial（训练后评估）、SpatiaLab、MindCube 三个基准上均达到开源/非专有模型最佳，且在几何推理、运动分析、交通分析等子类别上提升显著。

## 方法详解
**整体架构**：后训练框架，基座 LVLM 使用 GRPO 优化，AdaptSoft 控制器使用梯度对齐目标联合训练。

**软链式思维（Soft CoT）**：
- 每个 rollout 的中间 CoT 表示为 $T_i$ 个软状态序列 $\{s_{i,t}\}$，通过 token 嵌入混合替代离散 token 选择。
- **Gumbel 扰动**：在每个推理步对词汇表 log-probability 加独立 Gumbel(0,1) 噪声，得扰动分数 $z_{i,t,k} = \log \bar{p}_{i,t,k} + \gamma_{i,t,k}$。
- **温度控制混合**：$\tau_{i,t}$ 控制 softmax 浓度，$p_{i,t} = \text{softmax}(z_{i,t}/\tau_{i,t})$，软状态 $s_{i,t} = \sum_k p_{i,t,k} E_k$，仅在 top-K=5 候选 token 上计算。
- **Gumbel 重参数化似然**：利用 $z_{i,t}$ 在标准 Gumbel 密度下的条件联合对数密度 $\log P_\theta(z_{i,t}) = \sum_k[-\tilde{\gamma}_{i,t,k}^\theta - \exp(-\tilde{\gamma}_{i,t,k}^\theta)]$，构造软步似然比 $\rho_{i,t}^{\text{soft}}$。

**AdaptSoft 控制器**：
- 输入：layer-normalized 最后一层隐藏状态 $h_{i,t}$（经固定随机投影矩阵 $\mathbf{P} \in \mathbb{R}^{d_p \times d}$ 降维至 $d_p=8$）与标准化预测熵 $\widehat{H}_{i,t} = \frac{H(\bar{p}_{i,t}) - \mu_H}{\sigma_H + \epsilon_H}$。
- 输出：$\tau_{i,t} = \tau_0 + \Delta \cdot \tanh(f_\phi([\mathbf{P}\text{LN}(h_{i,t}) \| \widehat{H}_{i,t}]))$，其中 $\tau_0=0.5, \Delta=0.4$，温度范围 $(0.1, 0.9)$。
- **Stop-gradient 重建**：rollout 时记录 $(\xi_{i,t}, u_{i,t}^{\text{roll}})$，更新时以 $\text{sg}(\xi_{i,t})$ 为输入重算 $u_{i,t}$，前向复现 rollout 温度，反向传递梯度到 $\phi$。

**梯度对齐学习目标**：
- 将每批 rollout 分为参考集与优化集。参考集计算 $G_{\text{ref}} = \nabla_W \mathcal{L}_{\text{GRPO}}^{\text{ref}}$，优化集每软步计算 $g_{i,t} = \nabla_W \mathcal{L}_{\text{GRPO}}^{(i,t)}$，均为对输出层权重 $W$ 的梯度。
- 对齐得分 $\alpha_{i,t} = \langle g_{i,t}, G_{\text{ref}} \rangle_F = d_{i,t}^\top G_{\text{ref}} v_{i,t}$，其中 $d_{i,t}$ 为 detach 后的 GRPO logit 残差。
- 损失 $\mathcal{L}_{\text{AS}}(\phi) = -\sum_i \sum_t \alpha_{i,t}$，乘以缩放系数 $\kappa_b$ 后，减去 rollout 内步级均值实现中心化处理。

**GRPO 策略优化**：
- 混合似然比 $\rho_{i,t}(\theta)$：软步用 $\rho_{i,t}^{\text{soft}}$，离散答案 token 用标准 ratio。
- 目标函数：$\mathcal{I}(\theta) = \mathbb{E}[\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t (\min\{\rho A_i^{\text{task}}, \text{clip}(\rho,1-\delta,1+\delta)A_i^{\text{task}}\} - \eta D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}}))]$。
- 奖励：$r_i = w_{\text{ans}} R_{\text{ans}}^{(i)} + w_{\text{fmt}} R_{\text{fmt}}^{(i)}$，$w_{\text{ans}}=1.0, w_{\text{fmt}}=0.2$。

## 实验与结果
**数据集**：
- **OmniSpatial**：6,902 训练样本 + 1,533 测试样本，10 个空间推理子类别。
- **SpatiaLab**：1,400 零样本样本，6 个空间推理类别，真实无约束场景。
- **MindCube**：21,154 零样本问题，3 种空间心智建模设置（Rotation/Among/Around）。

**关键结果（OmniSpatial，表1）**：
- Soft Spatial Reasoning：**49.68%** 加权平均精度，领先 InternVL3-14B（45.94%）+3.74pp、SoFar（45.14%）+4.54pp、LaCoT（45.66%）+4.02pp。
- 同 backbone 消融：Hard Thinking + GRPO（45.92%）→ Soft Thinking + GRPO（46.93%，+1.01pp）→ Soft Spatial Reasoning（49.68%，+2.75pp）。
- **几何推理**：Hard/Basic Soft 均为 25.00%，Adaptive Soft 达 **34.84%**（+9.84pp）；**运动分析**：55.85% → **62.14%**；**交通分析**：36.04% → **57.65%**（+21.61pp）。

**关键结果（SpatiaLab，表2）**：
- Soft Spatial Reasoning：**48.71%**，超 LVR（45.78%）+2.93pp；同 backbone 上相对 Basic Soft 贡献 +1.85pp（vs 硬→软仅 +0.65pp）。

**关键结果（MindCube，表3）**：
- Soft Spatial Reasoning：**38.13%**，超 Backbone（33.62%）+4.51pp，为开源模型最高。

**消融（OmniSpatial 视角采样子类，表4）**：
- 去掉梯度对齐：-4.07pp（假设推理 -6.52pp，影响最大）
- 去掉隐藏状态 $h_{i,t}$：-3.53pp
- 去掉预测不确定性：-2.11pp
- 去掉梯度中心化：-1.57pp

## 相关工作脉络
1. **硬空间推理方法**（SpaceMantis、SpatialLadder、SoFar 等）：通过显式空间几何、结构化中间表征或 RL 训练增强空间证据利用，但未解决离散 CoT 的过早 commit 问题。
2. **视觉软思考**（LVR、Latent Visual Reasoning）：在视觉 token 层面维持连续表示，但 CoT 本身仍为离散 token 序列，存在与本文相同的过早离散化风险。
3. **LLM 软思考**（Soft Thinking Zhang et al. 2026b；CoLT Hu et al. 2026）：在 LLM 中将 CoT 扩展为连续混合状态，但未涉及 LVLM 空间推理的步级自适应控制，且温度固定。
4. **SofT-GRPO**（Zheng et al. 2025a）：首次将 GRPO 扩展到 Gumbel 重参数化的软 token，本文在此基础上扩展至 LVLM 空间推理并引入步级软度控制。
5. **空间推理评测基准**（OmniSpatial、SpatiaLab、MindCube）：覆盖多维度空间能力，本文在这三个基准上均实现开源最佳，验证了框架的通用性与迁移能力。

## 局限性与未来方向
1. **视觉不确定性缺失**：AdaptSoft 仅基于语言侧隐藏状态和预测熵调节软度，未显式建模视觉证据的不确定性（论文自述）。
2. **未覆盖视觉-语言双重歧义**：当前框架仅处理语言生成侧的候选混合，当视觉感知本身存在模糊（如遮挡、距离判断）时，软思考的温度控制可能不够充分。
3. **推断开销**：每步需维护 top-K=5 候选的 Gumbel 扰动与嵌入混合，推理延迟高于纯离散 CoT（文中未量化但可合理推断）。
4. **未来方向**：引入步级视觉不确定性估计，使软度同时反映视觉歧义与语言歧义；探索在更小温度范围内自适应可能进一步提升效率。

## 研究启发与可借鉴点
1. **步级自适应软度思想可迁移**：AdaptSoft 利用隐藏状态+预测熵控制温度的设计，可推广至非空间任务（如数学推理、代码生成），值得在更多领域验证。
2. **梯度对齐学习目标无需中间标注**：该方法绕过了对 CoT 中间步骤的人工标注需求，通过 rollout 内参考集构建自监督步级信号，对有标注瓶颈的任务具有借鉴价值。
3. **Stop-gradient 重建技巧**：用记录值在前向复现 rollout 状态、仅对控制器参数保留梯度的实现方式，有效降低显存开销，可在其他需要步级训练的控制模块中复用。
4. **Gumbel 重参数化扩展至 LVLM**：本文成功将 SofT-GRPO 的 Gumbel 似然推导适配到 vision-language 场景，为后续在 LVLM 中做软思考提供了可直接沿用的技术路线。
5. **训练-推理一致性格式**：系统 prompt 强制固定 `<think>...</think>` 结构，配合 discrete spine 跟踪，保证软 CoT 的结束位置明确，可在类似后训练框架中直接借鉴。

## 关键术语表
**Hard Thinking**：标准 CoT 推理方式，每步强制选择一个离散 token，易导致早期错误传播。
**Soft Thinking**：用 token 嵌入的连续加权混合替代单 token 选择，保留多候选信息。
**Soft Spatial Reasoning**：本文提出的后训练框架，为 LVLM 空间推理引入自适应软思考。
**AdaptSoft**：步级软度控制器，基于隐藏状态与预测熵动态调节温度 τ。
**Gumbel 重参数化**：利用 Gumbel(0,1) 扰动 score vector 使软状态确定性可微的技术。
**Gradient-Alignment Learning**：通过比对每步策略梯度与参考梯度的一致性，提供步级学习信号。
**GRPO（Group Relative Policy Optimization）**：不依赖 ground-truth CoT、基于组内相对奖赏的策略优化方法。
**Predictive Uncertainty（预测熵）**：下一步 token 分布的 Shannon 熵，用于度量模型对候选延续的置信度。

## 可复现要素
- **数据集**：OmniSpatial（训练+测试，论文使用官方 split：6,902 训练 / 1,533 测试）；SpatiaLab、MindCube 仅用于零样本评估。
- **代码**：已开源，https://github.com/rafiibnsultan/Soft_Spatial_Reasoning
- **模型权重**：未声明开源；基座为 Qwen3-VL-8B-Thinking（开源）。
- **关键超参**：backbone Qwen3-VL-8B-Thinking；GRPO 每 prompt 8 rollouts，batch size 64；策略学习率 1e-6，KL 系数 1e-3；AdaptSoft 学习率 1e-3，MLP 两层宽度 256，$d_p=8$；温度范围 $\tau \in (0.1, 0.9)$，基础 $\tau_0=0.5$；Top-K=5 候选；训练 200 步（≈2 epoch）；BF16 精度，FSDP2；verl 0.8.0 + SGLang 0.5.12，8×H100。
