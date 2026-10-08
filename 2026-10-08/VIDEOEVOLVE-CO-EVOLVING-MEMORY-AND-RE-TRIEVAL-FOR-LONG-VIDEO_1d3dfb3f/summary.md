---
title: "VIDEOEVOLVE-CO-EVOLVING-MEMORY-AND-RE-TRIEVAL-FOR-LONG-VIDEO"
source: https://arxiv.org/pdf/2610.10183v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 09:41:55"
field: "长视频多模态理解"
keywords: ["长视频理解", "记忆机制", "检索增强", "强化学习", "多模态大语言模型", "自我演化", "交替优化"]
innovations: ["交替式 Agentic RL 联合演化记忆构建与检索策略，打破构建-检索单向优化范式", "BEF 瓶颈感知反馈通过配对诊断动态分配记忆/检索训练资源", "CEF 能力感知反馈基于九类能力标签和前沿/探索/保留采样引导训练课程"]
benchmarks: ["Video-MME", "LongVideoBench", "LVBench", "MLVU", "MMVU"]
---

# 论文速读：VIDEOEVOLVE: CO-EVOLVING MEMORY AND RETRIEVAL FOR LONG-VIDEO UNDERSTANDING

## 一句话总结
VideoEvolve 提出了一种自演化框架，通过交替式 Agentic RL 联合优化长视频理解中的"记忆构建"与"记忆检索"两个策略，并引入瓶颈感知反馈（BEF）和能力感知反馈（CEF）指导演化方向，在五个基准上显著超越现有开源方法。

## 研究问题与动机
- **核心问题**：长视频理解中，记忆构建（记住什么）与检索策略（如何检索）之间存在根本性脱节——现有方法只固定了"记忆内容"却动态调整"检索方式"，导致遗漏的证据难以恢复，而存储的信息又因检索失配而无法有效利用。
- **现有方法不足1（记忆适应性有限）**：预定义的记忆构建策略从不从下游推理反馈中学习，被遗漏的证据反复迫使系统浪费算力重新访问原始视频。
- **现有方法不足2（记忆-检索错位）**：构建与检索通常分别优化，未显式对齐"记忆保留了什么"和"检索能访问到什么"，形成单向适应而非双向协同。
- **核心科学问题**：记忆构建和检索策略能否相互促进、共同演化，实现"记住什么"与"如何检索"的协同进化？

## 核心贡献（创新点）
1. **提出 VideoEvolve 自演化框架**：通过交替式 Agentic RL 联合优化记忆构建与检索策略，使两者都能从下游推理反馈中持续改进；与已有工作本质区别在于将记忆从静态记录转变为可学习接口，实现构建-检索闭环。
2. **引入瓶颈感知演化反馈（BEF）**：通过对比"仅记忆"与"允许重访视频"两种条件下的推理表现，自动诊断当前瓶颈位于记忆侧还是检索侧，动态分配训练资源；与已有工作本质区别在于避免了固定的训练平衡，能够跟踪跨周期的瓶颈迁移。
3. **引入能力感知演化反馈（CEF）**：基于九类视频理解能力标签，自适应调整训练课程，优先训练尚未充分发展但可学习的的能力；与已有工作本质区别在于防止系统在固定问题集上过拟合，转向通用能力发展。
4. **提出质量优先的群组相对优化目标**：任务正确性优先，资源效率仅作为同质量轨迹间的决胜机制；与已有工作本质区别在于明确区分了任务质量与资源效率的优先级层次。

## 方法详解
- **基础记忆（Base Memory）**：以 0.5 fps 对视频均匀采样，构建三层层级结构（Root → Super → Macro），由冻结的 Writer 生成结构化记录，整个演化过程保持固定不重写。
- **记忆演化器（Memory Evolver）**：通过 Observe–Decide–Augment 流程学习"记住什么"——选择目标 Macro → 决定时间区域（head/middle/tail/boundary/full）、帧数（2/4/8/16）和信息焦点（action/state、appearance/spatial、text alignment、general），由冻结 Writer 生成源链接的 Delta 记录，与 Base 合并为演化记忆 $M_v^k = \text{Merge}(B_v, \Delta_v^k)$，每次更新后从相同 Base 重建而非累积历史 Delta。
- **检索演化器（Retrieval Evolver）**：通过 Search–Inspect & Revisit–Reason 流程学习"如何检索"——支持层级导航、词汇/语义搜索、时间搜索、查看存储帧，以及可选的原始视频重访（最多 64 帧/问题），策略初始化为 SFT 多轮工具使用轨迹。
- **交替 Agentic RL**：第 $k$ 轮冻结检索演化器 $R^k$，优化记忆演化器 $C^k \to C^{k+1}$；重建记忆后冻结 $M^{k+1}$，优化检索演化器 $R^k \to R^{k+1}$，形成交替闭环。
- **记忆侧奖励**：$R_C = R_{\text{rescue}} - \lambda_{\text{reg}} P_{\text{regress}} - \lambda_{\text{inv}} P_{\text{invalid}}$，其中 rescue 奖励新挽回的问题，regress 惩罚先前已解决但退化的问题，invalid 惩罚无效轨迹。
- **检索侧奖励**：$R_R = R_{\text{ans}} - \lambda_{\text{inv}} P_{\text{invalid}}$，直接关联最终答案正确性。
- **质量优先群组相对优化**：先对任务奖励做群组归一化 $A_{\text{task}}$，资源效率仅在任务奖励相等的候选轨迹间作为决胜项（系数 $\beta=0.05$），最终损失为 clipped PPO 目标 + KL 正则化。
- **BEF**：比较 Memory-only（$m$）与 Revisit-enabled（$v$，新观测帧数 $f$）条件下的答案正确性，计算 $s_C = \mathbf{1}[m=0 \land v=1 \land f>0]$ 和 $s_R = \mathbf{1}[v=0 \lor (m=0 \land v=1 \land f=0)]$，动态分配记忆/检索训练比例（范围 [0.25, 0.75]，每轮变化限制 ±0.15）。
- **CEF**：基于九类能力标签（Frame-Only、Action & Motion、Order、Change、Temporal Reasoning、Complex Plot Comprehension、Video-Based Knowledge Acquisition、Social Behavior Analysis、Physical World Reasoning），用 EMA 平滑需求估计 $e_{p,c}^k$，混合均匀覆盖（$\rho=0.3$）与加权分布，并结合前沿/探索/保留三分支采样策略指导下一轮训练样本选择。
- **训练流程**：检索演化器经历两阶段 SFT 冷启动（Stage I: 4,317 条 LongVT 轨迹；Stage II: 102,732 条 LLaVA 轨迹），记忆演化器无冷启动；每个 RL 周期更新记忆侧→重建记忆→更新检索侧，共训练四轮，使用第三轮 checkpoint 评估。

## 实验与结果
- **数据集**：Video-MME、LongVideoBench、LVBench、MLVU、MMVU 五个长视频理解基准。
- **训练数据**：Video-MME-v2 中 514 个视频（2,056 个问题）用于 Agentic RL，128 个视频（512 个问题）用于 BEF/CEF 诊断（与训练集不重叠）。
- **基线**：涵盖自有增强 Video-MLLM（Video-R1、Time-R1、ReWatch-R1 等）、开放 Agentic Video-MLLM（LongVT、ParaVT、VideoZoomer 等）及专用记忆基方法（EgoRAG、HippoMM、MERIT、MemVid、Ego-R1 等）。
- **最强结果**：VideoEvolve-8B（训练版）在 LVBench 达 58.9%（超越最强开源基线 ParaVT +17.4pp）、LongVideoBench 达 70.2%（超越 ParaVT +9.8pp）、Video-MME(Long) 达 67.1%、MMVU 达 75.1%，五项指标均为所列开源方法中最高。
- **训练-free 变体**（Qwen3.8-27B，无参数更新）：LVBench 76.7%（超 MERIT-GPT +4.9pp）、Video-MME(Long) 78.8%（超 MERIT-GPT +1.1pp）。
- **对比基线 Qwen3-VL-8B + 工具**：VideoEvolve 超越 14.7–28.6 个百分点，表明 co-evolution 设计是主要增益来源。

## 相关工作脉络
1. **视频重访方法**（VideoAgent、DVD、LVAgent、VideoZoomer、LongVT、VITAL、Ego-R1）：聚焦"何时何地检查源视频"，本文聚焦互补问题——"持久记住什么以供后续检索"。
2. **记忆基方法**（MovieChat、MA-LMM、EgoRAG、HippoMM、WorldMM、MERIT、M3-Agent、MemVid）：记忆构建与检索分别优化或完全冻结，未建立构建-检索反馈闭环；本文通过交替 RL 使两者协同演化。
3. **Agentic RL 在视频推理中的应用**（VideoZoomer、VITAL、Ego-R1、Video-R1）：优化单一策略（检索或重访），本文扩展至双重策略的交替优化。
4. **自演化智能体**（EvolveR、SkillRL、Agent0、Evolving-RL）：改进推理策略、技能库和经验利用；本文首次将自演化扩展到"记忆内容"这一维度，构建了记忆-检索共同演化的新范式。
5. **长视频理解 MLLM**（LongVILA、TimeChat、LongVU）：通过时间建模、上下文扩展和视觉压缩直接处理更多 token；本文采用选择性信息访问范式，与直接扩展上下文正交。

## 局限性与未来方向
- **评估范围有限**：当前评估聚焦离线多选题理解，未验证流式视频、开放式对话或直接音频理解场景。
- **依赖冻结 Writer 的质量**： learned augmentation 仍可能遗漏短暂事件或细微视觉细节，源链接记录和有效引用不保证事实正确性。
- **反馈信号局限**：BEF/CEF 依赖固定诊断池和预定义能力分类，瓶颈估计是操作代理而非因果归因，覆盖范围受限于可用数据。
- **计算开销**：交替训练需要候选 rollout、冻结策略评分和重复记忆/索引重建；推理时多步检索和可选视频重访仍有计算成本。
- **未来方向**：流式输入的增量记忆更新、视听证据扩展、证据级验证与校准不确定性、弱监督与自适应能力发现、更高效记忆更新与策略蒸馏、端到端资源测量。

## 研究启发与可借鉴点
1. **交替优化范式可迁移**：将"构建"与"检索"（或类似的对偶任务）解耦为交替优化的双策略框架，适用于任何存在"存储-访问"耦合的系统（如知识图谱维护、代码仓库管理）。
2. **BEF 诊断机制设计精巧**：通过配对条件（有/无额外信息访问）的性能差异来定位瓶颈，这种方法论可迁移到其他需要诊断系统弱点并动态分配资源的场景。
3. **CEF 能力分类+前沿诊断的 curriculum 设计**：将问题级标签转化为训练分布调整信号，结合 frontier/exploration/retention 三分支采样，为多能力协同训练提供了可复用的课程学习模板。
4. **质量优先的资源效率处理**：将效率仅作为同质量轨迹的 tie-breaker 而非直接参与奖励计算，避免了过早牺牲正确性换取效率的陷阱，这一设计原则对资源敏感 RL 任务具有普适参考价值。
5. **从静态记忆到可学习接口的范式转换**：将记忆构建视为可训练策略输出而非固定流水线，为 MLLM 中的任何"信息组织"模块提供了自改进的架构思路。

## 关键术语表
**Memory Evolver**：学习"记住什么"的策略网络，通过观察-决策-增强流程选择性补充基础记忆的增量信息。
**Retrieval Evolver**：学习"如何检索"的策略网络，通过搜索-检查/重访-推理-回答流程自适应地从演化记忆中获取证据。
**Base Memory**：以 0.5 fps 采样的固定三级层级记忆（Root-Super-Macro），在整个 co-evolution 过程中保持不变。
**Delta**：由 Memory Evolver 生成的补充记忆记录，与 Base Memory 合并形成演化记忆，每次从相同 Base 重建而非累积。
**BEF（Bottleneck-Aware Evolution Feedback）**：通过对比 Memory-only 与 Revisit-enabled 条件下的推理表现，诊断当前系统瓶颈位于记忆侧还是检索侧，动态分配训练比例。
**CEF（Capability-Aware Evolution Feedback）**：基于九类视频理解能力标签，用 EMA 平滑的需求估计和三分支采样策略（前沿/探索/保留）引导训练样本选择，避免过拟合固定问题集。
**Agentic RL**：耦合多轮交互和工具使用的强化学习，本文中对 Memory/Retrieval 策略分别进行群组相对优化。
**Quality-First Group-Relative Optimization**：先对任务奖励进行群组归一化确定质量优势，资源效率仅在同质量轨迹间作为 tie-breaker（系数 0.05）的优化目标。

## 可复现要素
- **数据集**：Video-MME-v2（训练 514 视频/2056 问题 + 诊断 128 视频/512 问题）、LongVT 推导轨迹（4,317 条）、LLaVA-Video 轨迹（102,732 条）；评估基准 Video-MME、LongVideoBench、LVBench、MLVU、MMVU 均使用官方数据集。论文未明确声明训练数据的公开状态，评估基准为公开 benchmark。
- **代码/权重**：论文未声明开源代码或模型权重。
- **关键超参**：$\lambda_{\text{reg}}=1$、$\lambda_{\text{inv}}=0.2$、$\beta=0.05$、$\epsilon_c=0.2$、$\beta_{\text{KL}}=0.02$、$s^0=0.5$、$s_{\min}=0.25$、$s_{\max}=0.75$、$\delta_s=0.15$、$\alpha=0.5$、$\rho=0.3$、每轮 4 个候选构造轨迹/8 个 retrieval 轨迹（实际为 4 候选/4 轨迹）、学习率 构造 $2\times10^{-6}$ / 检索 $10^{-6}$。
