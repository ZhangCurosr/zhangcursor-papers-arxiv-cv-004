---
title: "SKILL-V-VERIFIABLE-SELF-EVOLVING-SKILL-LIBRARY-FOR-INTERACTI"
source: https://arxiv.org/pdf/2610.11781v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 09:55:21"
field: "交互式代理的可验证技能演化"
keywords: ["self-evolving skill library", "verifiable contract", "interactive agent", "catastrophic forgetting", "knowledge maintenance", "reinforcement learning", "skill revision"]
innovations: ["将技能建模为版本化可证伪合同，实现可测试的知识维护", "双路径结果驱动演化：失败新增 + 评分器-结果分歧修订", "证据门控非退化验证防止知识库退化"]
benchmarks: ["ALFWorld", "WebShop"]
---

# 论文速读：SKILL-V: VERIFIABLE SELF-EVOLVING SKILL LIBRARY FOR INTERACTIVE AGENTS

## 一句话总结
论文提出 Skill-V，将外部技能库中的技能建模为**可版本化、可证伪的合同（falsifiable contracts）**，使交互代理不仅能从失败中**新增**技能，还能在**评分器判断与真实环境结果不一致时主动修订已有技能边界**，并通过证据门控防止退化，最终在 ALFWorld 和 WebShop 上实现更紧凑且性能更强的技能库。

---

## 研究问题与动机

1. **现有技能库仅"增长"不可靠**：既有自进化技能库（如 SkillRL、Skill1）主要通过积累新知识来改进，失败会触发新技能加入，但**已有存储的技能几乎不会被重新审视**。
2. **语义相关 ≠ 任务适用**：检索到的技能可能在语义上相关，但在当前任务条件下不可用；或者技能编码了错误的操作边界（过宽或过窄）。
3. **缺乏可验证性机制**：RL 将反馈吸收进模型参数，难以隔离和修订单一策略；外部技能库更明确，但如何**持续测试并修正**已存储的知识仍是开放问题。
4. **灾难性遗忘风险**：纯积累式演化会引发严重的技能库膨胀和性能退化，限制了代理的长期自主自改进能力。

---

## 核心贡献（创新点）

1. **将技能建模为版本化、可证伪合同**：每个技能包含语义意图（$\phi_i$）、可复用程序（$p_i$）和可执行评分器（$\rho_i$），使技能边界可通过观察到的轨迹证据进行系统性测试与修订。
2. **提出双路径结果驱动的演化机制**：任务失败驱动新技能添加（Path 1），评分器-结果分歧驱动已有技能边界的定向修订（Path 2），两者共同构成技能库的完整演化闭环。
3. **设计证据门控（Evidence Gate）防止退化**：修订候选必须在不退步（Disc↓/FNR↓/BAcc↑）于历史回放证据集的情况下才能被提交，避免对已有支持知识的破坏。
4. **引入适用性感知过滤（Applicability-Aware Filter）**：将语义相关性检索与任务适用性判断分离，通过 LLM 评估器排除置信度高于阈值的明显不适用技能，减少错误调用。
5. **在两个环境上实现 SOTA 性能且库规模极小**：ALFWorld 95.3%、WebShop 85.9%，同时仅约 200 条存储条目，远低于 Skill1 的 5,000 条容量。

---

## 方法详解

### 3.1 问题设定

交互代理接收任务指令 $x$，产生轨迹 $\tau = (o_1, a_1, \dots, o_T, a_T)$，并获得环境结果 $y(\tau) \in \{0, 1\}$。代理维护外部技能库 $\mathcal{L}_m = \{s_i\}_{i=1}^{N_m}$，其中 $m$ 为版本索引。对于任务 $x$，代理检索子集 $S_m(x)$ 并以此条件化策略：

$$a_t \sim \pi_\theta(a_t \mid x, h_t, S_m(x))$$

训练期间收集经验 $\mathcal{D}_m$，更新库 $\mathcal{L}_{m+1}$，核心问题是确定哪种变更由经验支持。

### 3.2 技能作为可验证合同

每个技能表示为：

$$s_i = (\phi_i, p_i, \rho_i, \nu_i), \quad \rho_i \in \{\emptyset, (A_i, \mathcal{C}_i)\}$$

其中：
- $\phi_i$：**语义规范**，定义技能意图和适用范围，包含受保护的语义约束（不可被修订覆盖）。
- $p_i$：**可复用程序**，自然语言描述的有序步骤。
- $\rho_i$：**可执行评分器**，包含适用条件 $A_i$ 和加权行为准则集 $\mathcal{C}_i = \{c_{ij}\}$，每条准则有标识符、严重性（HARD/SOFT）、权重、匹配模式及证据匹配器。
- $\nu_i$：**演化记录**，追踪版本号和变更来源。

合同初始化阶段，新技能初始时 $\rho_i = \emptyset$（仅含语义描述），由评分器更新器在获得适用经验后编译生成可执行准则。

### 3.3 结果驱动的 Skill 演化

**评分器聚合**：对于任务 $x$，设 $\mathcal{A}(x) = \{s_i \in \mathcal{L}_m : A_i(x) = 1\}$ 为适用技能集，则评分器判定的轨迹得分：

$$q(\tau) = \frac{\sum_{s_i \in \mathcal{A}(x)} \sum_j w_{ij} m_{ij}(\tau)}{\sum_{s_i \in \mathcal{A}(x)} \sum_j w_{ij}}$$

最终合同级判定：

$$\hat{y}^{\mathrm{rub}}(\tau) = \mathbb{I}[q(\tau) \geq \eta \wedge H(\tau) = 0]$$

其中 $H(\tau)$ 指示是否有 HARD 准则被违反，$\eta = 0.7$ 为通过阈值。

**四象限诊断**：比较评分器判定与环境结果，产生四类证据（表9）：
- Pass + Success → 支持当前边界
- Fail + Failure → 一致负证据
- **Pass + Failure → 评分器过宽/不完整，需修订**
- **Fail + Success → 评分器过严，需修订**

**技能新增（Skill Addition）**：失败轨迹经 LLM 分析后提议新语义技能，立即可用但 $\rho_i = \emptyset$，后续由评分器更新器编译生成可执行准则。

**技能修订与验证（Skill Revision & Validation）**：

对技能 $s_i$，令 $\mathcal{E}_i$ 为其观察证据集（有界回放缓冲区）。定义三个评估指标：

$$\mathrm{Disc} = \frac{\mathrm{FP} + \mathrm{FN}}{N}, \quad \mathrm{FNR} = \frac{\mathrm{FN}}{\mathrm{TP}+\mathrm{FN}}, \quad \mathrm{BAcc} = \frac{1}{2}\left(\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}} + \frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}\right)$$

候选修订 $\rho_i'$ 被接受当且仅当满足**非退化条件**：

$$\mathrm{Disc}(\rho_i') \leq \mathrm{Disc}(\rho_i), \quad \mathrm{FNR}(\rho_i') \leq \mathrm{FNR}(\rho_i), \quad \mathrm{BAcc}(\rho_i') \geq \mathrm{BAcc}(\rho_i)$$

若修改了 HARD 准则，至少一个指标需严格改善。此外还需满足：非空可执行支持、符合合同规范、保留源技能的语义约束。

### 3.4 适用性感知技能使用

检索分两阶段：① 语义检索（Qwen3-Embedding-0.6B）获取候选池；② 适用性过滤：由 DeepSeek-V4-Flash 评估器对每个候选返回 {APPLICABLE, NOT APPLICABLE, UNCERTAIN} 及置信度。置信度 ≥ $\delta = 0.8$ 的 NOT APPLICABLE 被移除；余下中保留 cosine similarity 不低于最高分 $\epsilon = 0.15$ 的候选，最终选取 $K_g = 6$（通用）+ $K_t = 5$（任务特定）技能。

---

## 实验与结果

### 数据集与环境
- **ALFWorld**（Shridhar et al., 2020）：基于文本的具身任务环境，6 类子任务（Pick/Look/Clean/Heat/Cool/Pick2），报告各类型成功率及总体成功率。
- **WebShop**（Yao et al., 2022）：在线购物交互，报告平均任务分数（0–100）和成功率。

### 主要结果（Table 1）

| 方法 | ALFWorld All (%) | WebShop Score | WebShop Succ. (%) |
|------|-----------------|---------------|-------------------|
| GRPO | 77.6 | 79.3 | 66.1 |
| SkillRL | 89.9 | 85.2 | 72.7 |
| RetroAgent | 94.9 | 88.9 | 82.3 |
| Skill1 | 97.5 | 89.7 | 82.9 |
| **Skill-V (Ours)** | **95.3** | **92.5** | **85.9** |

- Skill-V 相对 GRPO 提升 **+17.7pp**（ALFWorld）、**+16.4pp** 分数（WebShop）。
- 相对 SkillRL 在 Look 类型上提升 **+28.6pp**，在 WebShop 上提升 **+7.5pp** 分数。
- 库规模：Skill-V 约 **200 条**，Skill1 扩展至 **5,000 条**，仅为后者的 **<4%**。

### 消融实验
- **知识来源分析**（Table 2）：无技能 78.59% → 初始库 88.99% → Skill-V 库 95.30%，演化增益 **+6.31pp**。
- **演化机制消融**（Table 3）：纯采集 68.8%；仅失败修订 67.2%；全 Skill-V 95.3%，完整方法比任何不完整变体高至少 **21.9pp**。
- **证据门控**（Table 4）：无门控时 46 次修订中 9 次退化，通过率仅 58.7%；全方法 10 次修订全部通过，性能从 75.0% 提升至 95.3%。
- **适用性过滤**（Table 5）：仅语义检索 71.9% → 加契约检查 89.1% → 加 LLM 评估器 95.3%。
- **评分器过程奖励**（Table 6）：引入过程奖励反而降至 93.8%，表明增益来自知识演化而非奖励塑形。

---

## 相关工作脉络

1. **Memory & Skill Reuse**：Reflexion（Shinn et al., 2023）和 ExpeL（Zhao et al., 2024）将交互反馈转为文本记忆；Voyager（Wang et al., 2024a）将经验组织为可执行技能。本文聚焦于技能进入记忆后的**测试与修正**问题。
2. **Self-Evolving Skill Learning**：SkillRL（Xia et al., 2026）递归扩展层次 SkillBank；Skill1（Shi et al., 2026）联合优化技能选择与蒸馏；Skill-R1（Vishe et al., 2026）迭代修订实例级技能；SkillOS（Ouyang et al., 2026）学习策展器更新外部 SkillRepo。本文与这些方法的定位差异：不仅关注**获取**，更关注**已有知识是否仍适用**。
3. **Reliable Skill Maintenance**：SkillOpt（Yang et al., 2026）将单技能文档视为可训练参数；SkillOps（Pu et al., 2026）将技能表示为类型化合同。本文在**交互式强化学习设置**下研究技能维护，利用评分器-环境分歧系统地识别边界错误，并通过严格验证缓解**灾难性遗忘**。
4. **RL-based Agents**：GRPO、RLOO 等基线仅优化策略参数，缺乏可隔离、可修订的外部技能机制；本文证明结合可验证外部知识可获得显著增益。
5. **Memory-Augmented RL**：MemRL、EvolveR、SimpleMem+GRPO 等方法增强记忆但缺乏严格的边界修正机制；本文在此类别中达到最强或接近最强性能。

---

## 局限性与未来方向

1. **环境类型受限**：当前使用离散文本准则评估合同；扩展到视觉或连续控制环境需要新的多模态评估机制（如集成 VLM 或程序化奖励函数）。
2. **缺乏技能删除机制**：框架仅修订操作边界，对根本上有缺陷的策略无法直接丢弃，需额外机制实现自动技能退役。
3. **对历史轨迹的强依赖**：严格验证依赖历史轨迹防止退化；若环境规则随时间变化，历史轨迹可能不再代表正确行为，需引入时间衰减缓冲区。
4. **语义规范与可执行边界的分离**：受保护语义约束无法完全被可执行代理验证，存在验证不完整的可能。

---

## 研究启发与可借鉴点

1. **版本化合同建模范式**：将技能分解为"不可变的语义核心 + 可变的可执行边界"的设计，为知识表示的渐进修正提供了可复用的架构模式，可迁移至其他需要长期维护的外部知识库场景。
2. **双向边界修正机制**：同时利用"评分器通过但任务失败"（过宽）和"评分器拒绝但任务成功"（过严）两种分歧信号进行修订，比单向修正更完善；这一思路可扩展至工具使用策略的自动校准。
3. **证据门控的非退化保证**：以 Disc/FNR/BAcc 为指标的历史回放验证机制，为任何基于经验的系统更新提供了防退化保障，可借鉴于持续学习、在线学习中的灾难性遗忘防护。
4. **检索与适用性判断分离**：将语义检索和任务适用性过滤作为两阶段设计，并通过 LLM 置信度阈值进行保守过滤，对 Agent 技能路由和 RAG 系统设计具有直接参考价值。
5. **库紧凑性与性能的平衡**：证明"维护精确性"优于"盲目积累"，提示在构建长期 Agent 系统时应重视知识的精炼与验证，而非单纯追求规模。

---

## 关键术语表

- **Falsifiable Contract（可证伪合同）**：将技能表示为包含语义意图和可执行准则的复合结构，使其行为边界可通过环境结果进行实证检验。
- **Evidence Gate（证据门控）**：要求技能修订必须在历史回放证据上不退化（Disc/FNR 不升、BAcc 不降）才能被提交，防止无效修改。
- **Applicability-Aware Filter（适用性感知过滤）**：在语义检索后通过 LLM 评估器判断技能是否与当前任务真正相关，过滤置信度高的不相关技能。
- **Rubric-Outcome Disagreement（评分器-结果分歧）**：评分器判定与任务实际结果不一致的情况，是触发技能修订的核心诊断信号。
- **Catastrophic Forgetting（灾难性遗忘）**：终身学习中旧知识因新经验注入而被不可逆覆盖或破坏的现象，本文通过非退化验证缓解此问题。
- **Non-regressive Validation（非退化验证）**：修订候选需满足在所有回放指标上不低于当前版本的严格接受条件。
- **Protected Semantic Constraints（受保护语义约束）**：技能核心语义定义部分，修订过程中不允许被覆盖或修改，确保技能身份一致性。
- **Executable Rubric（可执行评分器）**：将技能语义映射为一组带权重的可观察行为准则，通过轨迹匹配器进行确定性评估。

---

## 可复现要素

| 要素 | 详情 |
|------|------|
| **数据集** | ALFWorld、WebShop；论文使用了标准的公开环境，评估协议与 SkillRL/Skill1 一致 |
| **代码开源** | 论文未明确声明代码开源（arXiv 论文链接未提供 GitHub） |
| **策略模型** | Qwen2.5-7B-Instruct（SFT checkpoint），GRPO 训练，学习率 $10^{-6}$，KL 系数 0.01 |
| **嵌入模型** | Qwen3-Embedding-0.6B |
| **评估模型** | DeepSeek-V4-Flash |
| **关键超参** | 评分器阈值 $\eta = 0.7$；适用性置信度 $\delta = 0.8$；相似度边际 $\epsilon = 0.15$；候选倍数 $\alpha = 2.0$；检索预算 $K_g=6, K_t=5$；更新间隔 $F=5$ 步 |
| **训练配置** | ALFWorld batch=16，WebShop batch=32，每指令 8 rollouts，最大环境步数 50（ALFWorld）/15（WebShop） |

---
