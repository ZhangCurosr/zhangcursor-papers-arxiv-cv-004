---
title: "On-Policy-Visual-Evidence-Distillation"
source: https://arxiv.org/pdf/2609.36838v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:46:47"
field: "多模态大模型训练与蒸馏"
keywords: ["on-policy distillation", "visual evidence", "multimodal", "agent reasoning", "token reweighting", "reflection"]
innovations: ["通过同query轨迹比较生成视觉证据反思并诊断Acquire/Read/Ground首失败阶段", "基于reflection引发的教师预测变化量计算token impact并分组重加权蒸馏损失", "跨模型族验证在感知/数学/通用任务上超越所有OPD基线且提升效率"]
benchmarks: ["HRBench-4K", "HRBench-8K", "V*Bench", "TreeBench", "VisualProbe", "MathVista", "MathVerse", "VisuLogic", "HallusionBench", "ChartQA-Pro", "InfographicVQA"]
---

# 论文速读：On-Policy-Visual-Evidence-Distillation

## 一句话总结
论文提出了 **REVUE（Reflection on Visual Evidence）**，一种面向视觉智能体的 On-Policy Distillation 方法，通过诊断学生轨迹中视觉证据链（Acquire→Read→Ground）的首个失败阶段，将反思信息作为训练时上下文注入教师模型，并根据其对教师预测的影响程度对 token 级蒸馏损失进行分组重加权，从而提供更具针对性的监督信号。

## 研究问题与动机
1. **视觉交互中的证据链断裂问题**：图像操作会改变后续推理的可用证据，Acquire（获取）、Read（阅读）、Ground（定位）任一阶段的局部错误都会传播至最终答案。
2. **现有 OPD 方法缺乏对证据链的显式建模**：当前多模态 OPD（如 Vision-OPD、VAD、V-Zero）主要通过构建辅助视角或对比图像来增强监督，未显式追踪学生动作、观测结果与后续推理之间的因果关联。
3. **无法针对不同类型失败提供差异化纠正**：相同视觉证据下学生可能因 Read 阶段误读或 Ground 阶段规则应用错误而失败，现有方法难以区分这两类错误并提供针对性指导。
4. **多次尝试同一查询蕴含丰富的诊断价值**：成功轨迹可互补（更聚焦的证据、更准确的读取），失败轨迹可定位首个断裂点，为教师提供可操作的训练时上下文。

## 核心贡献（创新点）
1. **跨学生轨迹的视觉证据反思（Visual-Evidence Reflection）**：通过 critic 比较同 query 的多条轨迹，提取最小充分证据集合与正确读取/定位策略，并诊断失败轨迹中 Acquire/Read/Ground 的首个断裂阶段；与已有工作（Vision-OPD、VAD）的本质区别在于不依赖额外视图或特权裁剪，而是直接利用学生自己的交互轨迹生成反思。
2. **反思感知的分组蒸馏重加权（Reflection-Aware Grouped Reweighting）**：通过比较带/不带 reflection 的教师预测，计算每个 token 的 log-probability shift 作为 token impact，将监督位置按影响大小分组并分配差异化损失权重；与已有方法（V-Zero 的对比轨迹加权、VAD 的视觉归因）的本质区别在于权重由 reflection 对教师判断的实际扰动驱动，而非固定策略或视觉注意力。
3. **跨模型族与多任务的系统验证**：在 Qwen2.5-VL-7B 与 InternVL3.5-4B-Instruct 两个模型族上，覆盖 11 个 benchmark（感知、数学推理、通用任务），REVUE 在所有类别加权平均上均超越所有 OPD 基线；并观察到更高感知准确率伴随更短响应与更少工具调用，证明该方法能引导更高效的知识利用。

## 方法详解
### 整体框架
REVUE 的训练流程包含三个阶段：（1）从当前学生策略采样 M 条同 query 轨迹形成组 $\mathcal{G}_x$；（2）使用 critic（Qwen3.5-397B-A17B）生成每条轨迹的视觉证据反思 $\mathcal{R}_i = (\mathcal{A}_i, \mathcal{B}_i)$，其中 Anchor $\mathcal{A}_i$ 提供正确证据参考，Break Point $\mathcal{B}_i = (\sigma_i, \delta_i)$ 记录首个失败阶段及具体差异；（3）将 $\mathcal{R}_i$ 注入冻结教师 $q_\phi$，比较其对同一学生轨迹的预测差异，据此计算 token impact 并分组重加权 loss。

### 视觉证据反思构建
- **答案验证**：对 M 条轨迹验证答案正确性，分类为 all-correct / mixed / all-wrong。
- **证据状态收集**：记录每条轨迹的工具调用历史及对应观测图像，构建 evidence state。
- **Critic 诊断**：要求 critic 识别（a）哪些 state 包含充分证据；（b）最小支撑图像子集；（c）共享的 visual fact $f_x$ 与 grounding rule $\gamma_x$；（d）每条轨迹的首个失败阶段 $\sigma_i \in \{\text{Acquire, Read, Ground, Correct}\}$ 及差异描述 $\delta_i$。
- **Anchor 选择**：优先选取来自正确轨迹或自身轨迹中成本最小的充分 state，成本函数 $c(e) = (n_{\text{tool}}, n_{\text{code}}, |\mathcal{U}_e|, \text{id})$。

### Token Impact 计算
在相同学生历史 $h_{i,t}$ 下，分别用基础教师 $q^0_{i,t} = q_\phi(\cdot|h_{i,t})$ 与 reflection 条件教师 $q^{\mathcal{R}}_{i,t} = q_\phi(\cdot|h_{i,t}, \mathcal{R}_i)$ 预测，定义 candidate token v 的 log-probability shift：
$$d_{i,t}(v) = \log q^{\mathcal{R}}_{i,t}(v) - \log q^0_{i,t}(v)$$
Token impact 取学生采样 token $y_{i,t}$ 的绝对值：
$$s_{i,t} = |d_{i,t}(y_{i,t})| = |\log q^{\mathcal{R}}_{i,t}(y_{i,t}) - \log q^0_{i,t}(y_{i,t})|$$

### 分组重加权蒸馏损失
将每条轨迹的 supervised positions 按 $s_{i,t}$ 降序排列，前 $\lceil \alpha |\mathcal{T}_i| \rceil$ 个归入 High 组，其余归入 Low 组。引入权重 $w_{i,t}$ 修改 reverse-KL 目标：
$$\mathcal{L}_{\text{REVUE}}(\theta) = \frac{1}{|\mathcal{G}|} \sum_{\tau_i \in \mathcal{G}} \frac{1}{|\mathcal{T}_i|} \sum_{t \in \mathcal{T}_i} w_{i,t} \mathbb{D}^{\text{KL}}\left(p_{i,t}^{\text{Supp}_{i,t}} \middle\| \left(q_{i,t}^*\right)^{\text{Supp}_{i,t}}\right)$$
其中 $q^*_{i,t} = q^{\mathcal{R}}_{i,t}$（reflection 有效时）否则 $q^0_{i,t}$；权重设定 $\lambda=0.5$（High/Low 组平分总权重），$\alpha=0.2$：
$$w_{i,t} = \begin{cases} \lambda |\mathcal{T}_i| / |\mathcal{T}_{i,\text{High}}|, & t \in \mathcal{T}_{i,\text{High}} \\ (1-\lambda)|\mathcal{T}_i| / |\mathcal{T}_{i,\text{Low}}|, & t \in \mathcal{T}_{i,\text{Low}} \end{cases}$$
该设计使 High 组每个位置获得约 $0.5/\alpha = 2.5$ 倍的梯度放大（相对均匀平均），Low 组获得约 $0.5/(1-\alpha) = 0.625$ 倍。

## 实验与结果
- **模型与基准**：学生模型 Qwen2.5-VL-7B 与 InternVL3.5-4B-Instruct（均来自 Thyme-SFT cold-start checkpoint），教师为复现的 Thyme-RL Expert；11 个 benchmark 覆盖 Perception（HRBench-4K/8K、V*Bench、TreeBench、VisualProbe）、Math（MathVista-Mini、MathVerse、VisuLogic）、General（HallusionBench、ChartQA-Pro、InfographicVQA）。
- **主要结果**：在 Qwen2.5-VL-7B 上，REVUE 感知加权平均 65.01%（Vanilla OPD: 62.61%，+2.40%）；数学 48.49%（Vanilla OPD: 47.67%，+0.82%）；通用 54.14%（Vanilla OPD: 53.28%，+0.86%）。InternVL3.5-4B-Instruct 上感知加权平均 59.72%（Vanilla OPD: 58.02%，+1.70%）。
- **超越 RL Expert**：两个学生模型在三类任务加权平均上均超过其 RL Expert 教师。
- **效率收益**：Qwen2.5-VL-7B 在 HRBench-8K 上响应长度从 458 tokens 降至 382，准确率从 71.75% 提升至 74.00%；TreeBench 响应长度减少 46.09%，工具调用率从 64.7% 降至 15.8%。
- **消融**：仅加 reflection（不含 reweighting）提升全部三类平均；再加 impact reweighting 额外提升感知 +1.31%、数学 +0.71%。
- **Token impact 稀疏性**：早期训练 pooled 分数中 top 20% positions 贡献 97.6% 总 impact；Mask top 20% 比 Mask low 20% 导致更大性能下降（感知 62.26 vs 65.18），验证高 impact 位置的重要性。
- **Critic 诊断一致性**：与人工标注一致率 96.88%（κ=0.957），内部重复一致率 99.2%（κ=0.987）。

## 相关工作脉络
1. **Vision-OPD（Yuan et al., 2026）**：使用 evidence-centered crops 提供视觉特权监督；本文与其本质区别是不依赖额外裁剪，而是通过 trajectory comparison 和 critic 诊断生成反思。
2. **VAD（Zhang et al., 2026a）**：通过移除视觉证据估计其对教师修正的贡献；本文与其区别在于显式建模证据链三阶段并定位首个失败点，而非间接估计视觉贡献。
3. **V-Zero（Sun et al., 2026）**：基于相关/无关视觉视角的对比进行 trajectory weighting；本文与其区别在于对比对象是 reflection 条件与非条件教师，而非视觉视图。
4. **OPD 一般框架（Agarwal et al., 2024）**：纯 token-level reverse-KL 蒸馏；本文在此基础上引入 reflection 上下文与 impact-based 重加权。
5. **视觉 agent 系统（Thyme、DeepEyes、CogCoM 等）**：本文聚焦于将这些 agent 的能力通过 OPD 传递给小模型，并提出针对视觉证据链的诊断方法。
6. **SGCD/GRSD（Ding et al., 2026; Zheng et al., 2026a）**：使用 rollout group 的 peer trajectories 改进 credit assignment；本文同样利用同组轨迹但聚焦于视觉证据链的反思生成而非信用分配。

## 局限性与未来方向
1. **模型覆盖有限**：仅验证了 Qwen2.5-VL 和 InternVL3.5 两个家族，未扩展到更大参数规模或其他架构（如基于 transformer 之外的设计）。
2. **任务范围有限**：当前聚焦图像问答，未覆盖视频理解、多页文档推理等更复杂的时序/多视图视觉交互场景。
3. **Critic 成本较高**：每个 optimizer step 需 20 次 critic 调用，占总 wall-clock time 约 10.8%，限制了大规模训练的可扩展性。
4. **通用任务轻微下降**：加入 impact reweighting 后通用任务平均下降 0.28%，说明不同任务对重加权策略的偏好存在差异。
5. **未来方向**：扩展至更多模型族和视觉交互类型；探索自适应 $\alpha$ 和 $\lambda$ 超参；降低 critic 服务开销（如蒸馏 critic、异步调用）。

## 研究启发与可借鉴点
1. **跨轨迹诊断生成反思**：通过同 query 多条轨迹的比较，将成功轨迹的共同证据与失败轨迹的首个断裂点提取为结构化反思，可迁移至其他 agent 领域（如代码生成、工具使用）的 OPD 或 RL 训练。
2. **Token impact 作为 supervision 价值代理**：通过比较条件/非条件教师预测的 log-probability shift 来量化每个位置的监督重要性，该方法不依赖额外标注，可推广至语言模型蒸馏中的重要性采样。
3. **分组重加权替代均匀平均**：将监督位置按 impact 分组并分配差异化权重，在感知和数学任务上带来显著收益；该策略可与现有 OPD 方法（如 VAD、Vision-OPD）结合使用。
4. **三阶段证据链诊断框架**：Acquire/Read/Ground 的分解思路具有通用性，可用于分析多模态模型的失败模式，甚至作为评估指标或 curriculum 设计的基础。
5. **Critic 驱动的反射生成**：使用强大 LLM 作为 critic 生成结构化反思（JSON schema），可在训练时提升蒸馏质量；其 cost-efficiency 优化（如 batching、缓存）是值得探索的工程方向。

## 关键术语表
- **On-Policy Distillation (OPD)**：在当前学生策略采样的轨迹上进行教师到学生的 token-level 蒸馏，减少 train-inference mismatch。
- **Visual Evidence Chain**：视觉推理的证据链，由 Acquire（获取视觉证据）→ Read（读取视觉事实）→ Ground（将事实映射到答案）三阶段构成。
- **Reflection (视觉证据反思)**：由 critic 生成的结构化诊断信息，包含正确证据参考（Anchor）和失败阶段定位（Break Point）。
- **Token Impact**：reflection 引起的教师对学生采样 token 的 log-probability 变化量，用于衡量该位置监督的重要性。
- **Acquire 阶段**：学生通过工具调用（如 crop、zoom）获取视觉证据的阶段，失败表现为选区错误或遗漏关键证据。
- **Read 阶段**：学生解读已获得视觉证据的阶段，失败表现为误读图像内容或 OCR 错误。
- **Ground 阶段**：学生将读取到的视觉事实映射到最终答案的阶段，失败表现为逻辑推理错误。
- **Critic (诊断模型)**：使用 Qwen3.5-397B-A17B 作为独立评判器，对同 query 的轨迹组进行证据链分析与失败诊断。

## 可复现要素
- **数据集**：训练数据来自 Thyme-RL 发布的 RL 数据集（论文未公开具体数据链接）；评测基准包括 HRBench-4K/8K、V*Bench、VisualProbe、TreeBench、MathVista-Mini、MathVerse、VisuLogic、HallusionBench、ChartQA-Pro、InfographicVQA。
- **代码/权重**：论文提供了 Website 和 Code 链接（具体地址见原文脚注），模型权重未明确公开。
- **关键超参**：每 query 采样 M=8 条轨迹，teacher support K=32，high-impact 比例 α=0.2，group weight λ=0.5，temperature（code=0.0, others=1.0），top-p=0.9，top-k=50；Critic temperature=0，输出上限 2048 tokens。
- **硬件配置**：训练 8×NVIDIA H20（96GB），Critic 服务 8×NVIDIA H20（96GB）。
