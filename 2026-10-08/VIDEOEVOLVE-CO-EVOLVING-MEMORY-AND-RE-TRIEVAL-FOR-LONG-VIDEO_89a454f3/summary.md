---
title: "VIDEOEVOLVE-CO-EVOLVING-MEMORY-AND-RE-TRIEVAL-FOR-LONG-VIDEO"
source: https://arxiv.org/pdf/2610.10183v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 22:46:01"
field: "长视频理解与多模态Agent"
keywords: ["长视频理解", "记忆机制", "检索增强", "Agent强化学习", "自进化系统", "多模态大模型"]
innovations: ["交替Agentic RL协同演化记忆构建与检索策略", "瓶颈感知反馈动态分配训练资源", "能力感知反馈引导自适应课程学习"]
benchmarks: ["Video-MME", "LongVideoBench", "LVBench", "MLVU", "MMVU"]
---

# 论文速读：VIDEOEVOLVE-CO-EVOLVING-MEMORY-AND-RE-TRIEVAL-FOR-LONG-VIDEO

## 一句话总结
提出 VideoEvolve，一种通过交替 Agentic RL 协同演化记忆构建与检索策略的自进化框架，解决长视频理解中"记住什么"与"如何检索"割裂优化的问题，在五个基准上超越现有开源方法。

## 研究问题与动机
1. **记忆适应性不足**：现有记忆方法依赖预定义策略构建记忆，无法从下游推理反馈中学习，导致遗漏的关键证据反复触发高成本的原始视频重访。
2. **记忆-检索错位**：记忆构建与检索通常独立优化，未显式对齐"记忆保留内容"与"检索可访问内容"，造成存储信息仅在其可被可靠检索时才具有价值。
3. **单一优化路径的局限**：仅优化记忆或仅优化检索均无法充分捕捉两者的耦合关系，需要联合演化机制实现双向增益。

## 核心贡献（创新点）
1. **交替 Agentic RL 协同演化框架**：首次通过交替更新记忆演化器和检索演化器实现"记住什么"与"如何检索"的联合优化，区别于传统单向优化范式。
2. **瓶颈感知演化反馈（BEF）**：设计双条件诊断机制（仅记忆 vs 支持重访），动态识别当前训练瓶颈并重新分配优化资源，避免固定训练比例导致的次优解。
3. **能力感知演化反馈（CEF）**：基于九类视频理解能力标签构建自适应训练课程，引导演化向未充分发展但可学习的capability转移，缓解过拟合特定问题的风险。
4. **层次化基础记忆+选择性增强架构**：采用固定低帧率 Hierarchical Base Memory（Root-Super-Macro三级）作为不可变基础，学习性 Delta 增量补充关键证据，兼顾复用性与表达能力。

## 方法详解
**整体框架**：VideoEvolve 包含 Memory Evolver（学习观察-决策-增强流程）和 Retrieval Evolver（学习搜索-检查-推理流程），通过交替 Agentic RL 迭代优化。

**记忆演化器（Memory Evolver）**：
- 从固定 Base Memory（0.5 fps 采样，三层 Root→Super→Macro 层级）出发
- 两阶段决策：(1) 选择 Macro 节点或终止；(2) 确定时间区域（full/head/middle/tail/boundary）、帧数级别{2,4,8,16}、提取焦点{action_state, appearance_spatial, text_alignment, general}
- 冻结 Writer 生成结构化记录，构成 Delta 增量
- 记忆重建：$M_v^k = \text{Merge}(B_v, \Delta_v^k)$，每次从同一 Base 重新构建而非累积历史 Delta

**检索演化器（Retrieval Evolver）**：
- 七类工具：get_macro_events, get_subgraph, search_nodes（BM25+语义融合）, search_by_time, get_keyframes, read_document, view_video
- 12探索槽+1最终决策，支持 Greedy/温度0.7采样
- 冷启动：两阶段 SFT（Stage I: 4,317条LongVT轨迹；Stage II: 102,732条LLaVA轨迹）

**交替优化流程**：
$$ (C^k, R^k, M^k) \to (C^{k+1}, R^k, M^k) \to (C^{k+1}, R^{k+1}, M^{k+1}) $$

**质量优先组相对优化目标**：
- 记忆奖励：$R_C = R_{\text{rescue}} - \lambda_{\text{reg}} P_{\text{regress}} - \lambda_{\text{inv}} P_{\text{invalid}}$
- 检索奖励：$R_R = R_{\text{ans}} - \lambda_{\text{inv}} P_{\text{invalid}}$
- 优势函数：$A^{(i)} = A_{\text{task}}^{(i)} + A_{\text{eff}}^{(i)}$，效率项仅在同质量时作平局决胜
- RL损失：带裁剪的 PPO 目标 + KL正则化

**BEF调度机制**：
- 在 Probe Questions 上评估 Memory-only ($m$) 和 Revisit-enabled ($v$) 两种模式
- 记忆侧需求：$s_C = \mathbf{1}[m=0 \land v=1 \land f>0]$（仅记忆失败但重访可挽救且消耗了帧）
- 检索侧需求：$s_R = \mathbf{1}[v=0 \lor (m=0 \land v=1 \land f=0)]$
- 目标比例：$s_{\text{target}} = \text{clip}(d_C/(d_C+d_R), s_{\min}, s_{\max})$，平滑更新避免剧烈波动

**CEF调度机制**：
- 九类能力标签：Frame-Only, Action & Motion, Order, Change, Temporal Reasoning, Complex Plot, Video-Based Knowledge, Social Behavior, Physical World
- EMA平滑需求估计：$e_{p,c}^k = \alpha e_{p,c}^{k-1} + (1-\alpha)d_{p,c}^k$
- 目标分布混合均匀覆盖：$\tilde{w}_{p,c}^{k+1} = \frac{\rho}{K} + (1-\rho)\frac{e_{p,c}^k}{\sum e_{p,c'}^k}$，$\rho=0.3$
- 三类样本混合采样：前沿（0.6）/探索（0.2）/保留（0.2）

## 实验与结果
**数据集与指标**：Video-MME（主/长视频子集）、LongVideoBench（整体/长视频子集）、LVBench、MLVU、MMVU；报告多选题准确率。

**训练设置**：514视频（2,056题）用于RL训练，128视频（512题）用于诊断；4轮交替演化，第3轮checkpoint评估。

**主要结果（最强对比）**：
- **LVBench**：VideoEvolve-8B 达到 **58.9%**，超越最强开源基线 ParaVT-8B（39.8%）达 **17.4pp**，超越训练式记忆基线 MemVid（44.4%）达 **14.5pp**
- **LongVideoBench**：达到 **70.2%**，超越 VideoZoomer-7B（57.7%）达 **9.8pp**
- **Video-MME (Long)**：达到 **67.1%**，超越 Ego-R1（64.9%）
- **MMVU**：达到 **75.1%**
- 训练-free 版本（Qwen3.8-27B）在 LVBench 达 **76.7%**，超越 MERIT-GPT（71.8%）4.9pp

**消融结论**：
- 协同演化 vs 单侧：全模型 LVBench 58.9% vs 仅记忆51.1%/仅检索52.7%
- BEF+CEF：MMVU 从68.4%→75.1%，证明双反馈机制必要性
- 视频重访：各指标提升0.8-1.6pp
- 冷启动SFT：缺失则LVBench从58.9%暴跌至32.8%

## 相关工作脉络
1. **Video-Agent系列**（VideoAgent, DVD, LVAgent）：关注"何时/何地"检查源视频，本文聚焦"记忆什么"的持久化表征问题，形成互补视角。
2. **记忆基线方法**：EgoRAG/HippoMM/WorldMM/MERIT采用预定义或无参数记忆构建，本文引入可学习的选择性增强机制。
3. **训练式记忆方法**：MemVid通过SFT+DPO训练记忆生成，Ego-R1训练检索控制器；本文同时训练两者并通过交替RL实现协同演化。
4. **Agentic RL for Video**：VideoZoomer/LongVT/Ego-R1使用RL训练单一策略；本文扩展至双策略交替优化，捕捉记忆-检索耦合关系。
5. **自进化Agent**：Evolver/SkillRL/Agent0/Evolving-RL改进推理策略或技能库；本文首次将自进化思想应用于"记忆内容+检索方式"联合演化。
6. **视频多模态大模型**：LongVILA等扩展上下文长度处理长视频；本文通过选择性记忆+检索实现更高效的长视频理解。

## 局限性与未来方向
1. **评估范围局限**：仅针对离线多选题理解，未验证流式视频、开放对话、直接音频理解等场景。
2. **依赖冻结Writer质量**：增量记忆质量受限于Writer感知能力，有限观察预算下仍可能遗漏短暂事件或细微视觉细节。
3. **信号局限性**：BEF/CEF基于固定诊断池和预定义能力分类，瓶颈估计为操作代理而非因果归因。
4. **计算开销**：交替训练需候选采样、冻结策略评分、重复记忆重建；推理时多步检索+可选重访仍有成本。
5. **未来方向**：流式增量记忆更新、视听多模态扩展、证据级验证与不确定性校准、弱监督+自适应能力发现、效率优化与策略蒸馏。

## 研究启发与可借鉴点
1. **交替优化范式可迁移**：将相互依赖的双策略（如编码器-解码器、生成-判别、规划-执行）通过交替RL联合演化，可能突破单策略优化的性能天花板。
2. **双维度反馈调度机制**：BEF（瓶颈感知）+ CEF（能力感知）的组合思路——既分配资源又指导样本选择——适用于需要动态平衡多目标的系统训练。
3. **固定基础+学习增量架构**：Base Memory不可变、Delta学习增强的设计兼顾稳定性与适应性，可推广至其他需要版本管理/回滚能力的系统。
4. **能力标签驱动的课程学习**：将细粒度能力 taxonomy 融入采样分布更新，避免过拟合特定问题分布，对多任务/多领域agent训练有参考价值。
5. **质量优先的效率平局决胜**：效率不直接替代任务正确性，仅在同等质量时作tie-breaker，保证性能优先同时鼓励资源节约，可复用于成本敏感的场景。

## 关键术语表
**Memory Evolver**：学习"记住什么"的策略网络，通过观察-决策-增强流程选择性扩充基础记忆。

**Retrieval Evolver**：学习"如何检索"的策略网络，通过搜索-检查-重访-推理流程回答问题。

**Agentic RL**：结合多轮工具交互与环境感知的强化学习范式，用于训练agent的决策策略。

**Bottleneck-Aware Evolution Feedback (BEF)**：通过配对诊断识别当前记忆或检索的相对瓶颈，动态调整两侧训练资源比例。

**Capability-Aware Evolution Feedback (CEF)**：基于九类视频理解能力标签的自适应课程，引导训练向未充分发展但可学习的capability转移。

**Base Memory**：固定不变的层次化低帧率视频概览（Root-Super-Macro三级），作为所有迭代的不可变基础。

**Delta**：记忆演化器学习生成的结构化补充记录，与Base合并形成增强记忆。

**Quality-First Group-Relative Optimization**：组内相对优化，任务质量为主、资源效率为辅（同质量时作平局决胜）的损失设计。

## 可复现要素
- **数据集**：Video-MME-v2（514视频/2,056题训练，128视频/512题诊断）、LongVT轨迹（4,317条）、LLaVA视频轨迹（102,732条）；基准集Video-MME/LongVideoBench/LVBench/MLVU/MMVU
- **代码/权重**：论文未明确声明开源状态
- **关键超参**：学习率 $2\times10^{-6}/10^{-6}$（记忆/检索）、裁剪范围0.2、KL系数0.02、回归惩罚1.0、无效惩罚0.2、效率系数0.05、EMA系数0.5、均匀覆盖权重0.3、前沿/探索/保留比例0.6/0.2/0.2、初始共享比例0.5、每轮最大变化0.15
