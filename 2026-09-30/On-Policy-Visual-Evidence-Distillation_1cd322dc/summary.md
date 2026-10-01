---
title: "On-Policy-Visual-Evidence-Distillation"
source: https://arxiv.org/pdf/2609.36838v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:46:29"
field: "多模态大模型知识蒸馏"
keywords: ["on-policy distillation", "visual reasoning", "agent", "knowledge distillation", "vision-language model", "evidence chain"]
innovations: ["跨轨迹视觉证据反思：通过critic对比同查询多轨迹提取最小充分证据并定位首处失败阶段(Acquire/Read/Ground)", "反思感知的token分组重加权：按教师log-probability偏移幅度对token分级，对top-20%高影响位置施加2.5倍梯度权重", "将阶段诊断信号转化为token级蒸馏监督，实现视觉交互行为的定向修正"]
benchmarks: ["HRBench-4K", "HRBench-8K", "V* Bench", "TreeBench", "VisualProbe", "MathVista-Mini", "MathVerse", "VisuLogic", "HallusionBench", "ChartQA-Pro", "InfographicVQA"]
---

# 论文速读：On-Policy Visual Evidence Distillation

## 一句话总结
本文提出了 REVUE（Reflection on Visual Evidence），一种面向视觉智能体的 On-Policy 蒸馏（OPD）方法：通过对比同一查询的多个学生轨迹来诊断视觉证据链中首次失败阶段（Acquire/Read/Ground），并将这些反思作为教师上下文，据此对 token 级蒸馏损失进行分组加权，从而实现对高影响位置的重点监督。

## 研究问题与动机
- **视觉交互中证据链错误会传播**：智能体在推理与图像操作之间交替进行，Acquire（获取视觉证据）、Read（读取视觉事实）、Ground（映射答案）三个阶段任一出错，错误都会沿轨迹传播导致最终答案错误。
- **现有 OPD 方法缺乏针对性**：Vision-OPD、V-Zero、VAD 等通过构造对比视图或特权视觉线索增强监督，但未显式建模学生动作、返回观测与后续推理之间的关联，无法针对不同失败阶段提供定向修正。
- **同查询多轨迹提供诊断基础**：多次尝试同一查询的成功轨迹可互为补充（找到更聚焦的证据、读得更准确），失败轨迹则暴露证据链首处断裂，可用于构建训练时的反思上下文。
- **教师评分变化具有稀疏性**：早期训练中，约 20% 的 token 位置贡献了 97.6% 的总影响，均匀平均会使这些高影响位置获得过少的监督权重。

## 核心贡献（创新点）
1. **跨轨迹视觉证据反思机制**：通过 critic 比较同查询的多条学生轨迹，提取最小充分证据集合并总结正确的读取与映射策略，同时定位首处失败阶段（Acquire/Read/Ground）——与已有工作（如 Vision-OPD 使用证据裁剪、VAD 使用视觉对比）本质区别在于：前者显式追溯学生的动作-观测-推理因果链。
2. **反思感知的 token 级分组重加权**：将 token 按反思对教师预测的影响幅度排序分组，对 top-20% 高影响位置赋予更高蒸馏损失权重——与标准 OPD（均匀平均所有 token）的本质区别在于：以教师评分变化作为监督价值代理，而非按位置顺序或随机选择。
3. **跨模型族与多基准验证**：在 Qwen2.5-VL-7B 和 InternVL3.5-4B 两个模型族、11 个基准上均超越所有 OPD 基线，且减少冗余推理和工具调用、提升工具调用准确率——与已有方法的定位差异在于：首次系统性地将对视觉证据链的阶段诊断转化为 token 级蒸馏信号。

## 方法详解
**整体框架**：对每个查询 x，当前学生采样 M 条轨迹组成轨迹组 $\mathcal{G}_x$，critic 生成每条轨迹的视觉证据反思 $\mathcal{R}_i = (\mathcal{A}_i, \mathcal{B}_i)$，其中 Anchor $\mathcal{A}_i$ 提供正确证据参考（最小支撑图像子集、视觉事实 $f_x$、映射规则 $\gamma_x$），Break Point $\mathcal{B}_i=(\sigma_i, \delta_i)$ 描述首处失败阶段及具体差异。

**Token Impact 计算**：对同一学生历史 $h_{i,t}$，冻结教师分别在有/无反思条件下的预测记为 $q_{i,t}^{\mathcal{R}}$ 和 $q_{i,t}^{0}$，定义目标 token $y_{i,t}$ 的 log-probability 偏移为 $d_{i,t}(v) = \log q_{i,t}^{\mathcal{R}}(v) - \log q_{i,t}^{0}(v)$，token 影响分数为 $s_{i,t} = |d_{i,t}(y_{i,t})|$。 Lemma 1 证明该偏移驱动了蒸馏梯度的变化量。

**分组重加权损失**：将轨迹内监督位置按 $s_{i,t}$ 降序排序，top $\lceil\alpha|\mathcal{T}_i|\rceil$ 归入 High 组，其余为 Low 组。最终损失为：
$$\mathcal{L}_{\mathrm{REVUE}}(\theta) = \frac{1}{|\mathcal{G}|}\sum_{\tau_i \in \mathcal{G}}\frac{1}{|\mathcal{T}_i|}\sum_{t \in \mathcal{T}_i} w_{i,t} \, \mathbb{D}^{\mathrm{KL}}\!\left(p_{i,t}^{\mathrm{Supp}} \Big\| \left(q_{i,t}^{*}\right)^{\mathrm{Supp}}\right)$$
其中 $\alpha=0.2, \lambda=0.5$，High 组权重约为均匀平均的 2.5 倍，Low 组约为 0.625 倍。

**Critic 设计**：使用 Qwen3.5-397B-A17B（温度=0），按 JSON schema 输出：判断哪些 evidence state 是 sufficient 的、最小支撑图像子集、query_slot/observed_value/visual_fact、grounding rule，以及对每条轨迹的诊断（CORRECT/Acquire/Read/Ground/Other）。

## 实验与结果
- **数据集/模型**：Qwen2.5-VL-7B 和 InternVL3.5-4B-Instruct 两个学生模型，教师为 Thyme-RL Expert（复现），critic 为 Qwen3.5-397B-A17B。
- **11 个基准**：HRBench-4K/8K、V* Bench、TreeBench、VisualProbe（感知）；MathVista-Mini、MathVerse、VisuLogic（数学）；HallusionBench、ChartQA-Pro、InfographicVQA（通用）。
- **最强结果**（Qwen2.5-VL-7B）：感知加权平均分 **65.01**（Vanilla OPD 为 62.61，+2.40%）；数学 48.49（+0.82%）；通用 54.14。InternVL3.5-4B：感知 59.72（+1.70%）、数学 49.17、通用 55.13。
- **关键提升**：HRBench-8K 从 71.75%→74.00%（+2.25%），TreeBench 从 38.02%→41.12%（+3.10%）。
- **行为优化**：HRBench-8K 上响应长度从 RL Expert 的 458 tokens 降至 382，工具调用率降低，但准确率更高。
- **Ablation**：仅加反思即提升全部三类任务；再加重加权再提升感知 +1.31%、数学 +0.71%。Impact 排名优于随机选择（感知 +1.37%，数学 +1.07%）。Mask Top-20% 损害 > Mask Low-20%，验证高影响 token 的重要性。Critci 诊断与人工标注一致率达 96.88%（κ=0.957）。

## 相关工作脉络
- **Vision-OPD (Yuan et al., 2026)**：使用证据为中心的裁剪（evidence-centered crops）作为教师特权视觉输入；REVUE 的区别在于通过反思提取正确证据并定位失败阶段，而非依赖静态裁剪。
- **V-Zero (Sun et al., 2026)**：基于视觉对比（有/无相关视觉信息）的 OPD 权重选择；REVUE 不使用对比视图，而是直接建模学生轨迹间的关联并进行阶段诊断。
- **VAD (Zhang et al., 2026a)**：通过移除视觉证据估计其对教师预测的贡献（反事实归因），用于目标重建；REVUE 不做视觉归因，而是跨轨迹对比定位证据链首处断裂。
- **CodeV (Hou et al., 2026)**：在视觉工具输入/输出上使用过程奖励（process reward）引导；REVUE 属于蒸馏范式，无需额外奖励信号，利用教师分布本身提供监督。
- **SGCD/GRSD (Ding et al., 2026; Zheng et al., 2026a)**：在多轨迹组中提取指导信号用于强化学习的信用分配；REVUE 定位差异在于：专用于视觉证据链的阶段诊断，而非通用的 RL 信用分配。
- **ViCuR (Tian et al., 2026)**：将可视线索作为可恢复特权引入 OPD；与 REVUE 的本质区别：ViCuR 增强教师视觉输入，REVUE 为教师提供跨轨迹的文本+图像反思上下文并改变 token 级权重。

## 局限性与未来方向
- 仅在两个模型族（Qwen2.5-VL 和 InternVL3.5）及图像问答任务上验证，更广泛的模型覆盖尚未探索。
- Critic 需要额外的 GPU 资源（8×H20，独立服务），增加训练开销（约占 wall-clock 的 10.8%）。
- 通用任务加权平均分在重加权后略降（-0.28%），说明不同任务类型对高影响聚焦的偏好存在差异。
- 论文自述的未来方向：扩展到更多模型族、更多样化的视觉交互场景。

## 研究启发与可借鉴点
- **证据链阶段诊断范式可迁移**：Acquire/Read/Ground 三阶段错误归因框架可用于分析其他视觉推理任务中的失败模式，甚至推广到文本代理中的"信息检索/理解/决策"链条。
- **Token Impact 稀疏性启发**：约 20% token 贡献 97.6% 影响，这一现象在纯语言 OPD 中同样值得验证，或可设计通用的高影响 token 定位策略。
- **反思作为教师上下文**：将学生轨迹间的比较结果作为文本+图像补充输入给教师，是一种无需改变教师架构的条件增强手段，可复用到其他需要"跨样本对比"的监督场景。
- **行为效率提升**：REVUE 在提升准确率的同时减少了冗余推理和工具调用，说明证据链精修不仅提高正确率，也改善推理经济性——这一观察对 Agent 系统的实际部署有直接价值。

## 关键术语表
- **On-Policy Distillation (OPD)**：用当前学生策略采样轨迹，在轨迹上以固定教师分布计算 KL 损失进行蒸馏，减少 train-inference distribution mismatch。
- **Visual Evidence Chain**：将视觉问答分解为 Acquire（获取图像证据）→ Read（读取视觉事实）→ Ground（映射到答案）的三段因果链。
- **Visual-Evidence Reflection**：Critic 从同查询多条轨迹中提取的最小充分证据集合、正确读取策略、映射规则及失败阶段诊断的文本-图像组合。
- **Token Impact**：反映反思上下文对教师在该 token 上预测概率的 log-probability 偏移绝对值，衡量该位置监督调整的幅度。
- **Anchor / Break Point**：反思的两个组成部分；Anchor 提供正确证据参考，Break Point 描述特定轨迹相对于锚点的首处失败阶段及差异。
- **Grouped Distillation Reweighting**：按 token impact 将监督位置分为 High/Low 两组，以不同总权重（λ=0.5）均匀分配组内各位置的损失系数。
- **Thyme**：北京大学/Tencent 提出的"与图像思考"视觉智能体框架，支持 reasoning-tool-use 交替的 agentic 视觉推理。
- **Sufficient Evidence State**：某交互状态下已获得的图像集合中，存在一个最小子集足以确定任务所需视觉事实，无需更多工具调用。

## 可复现要素
- **数据集**：Thyme-SFT cold-start checkpoint（公开），训练数据来自 Thyme-RL 数据集（论文未明确说明是否完全公开）；11 个评估基准均为公开数据集。
- **代码/权重**：论文标注"Website § Code"，但正文未提供 GitHub 链接；模型权重为 Thyme 开源 checkpoint 及复现的 RL Expert。
- **关键超参**：M=8 条轨迹/查询，K=32（教师 top-K 候选），α=0.2（高影响分组比例），λ=0.5（高/低组总权重），temperature_code=0.0、temperature_other=1.0、top-p=0.9、top-k=50；critic temperature=0，输出上限 2048 tokens。
- **硬件**：训练 8×NVIDIA H20 (96GB)，Critic 服务 8×NVIDIA H20 (96GB)。
