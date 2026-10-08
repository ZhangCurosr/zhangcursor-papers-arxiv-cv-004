---
title: "VIS-Ground-Video-Interactive-Storytelling-with-Contextual-Gr"
source: https://arxiv.org/pdf/2610.09326v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 22:46:18"
field: "多模态生成与交互"
keywords: ["video interactive storytelling", "contextual grounding", "constrained video generation", "agentic planning", "VIS-Bench"]
innovations: ["结构化上下文抽象将异构输入转化为对齐的状态模型与交叉依赖", "候选条件约束诱导动态激活相关依赖为生成约束", "规划-验证-修订-渲染迭代流程统一跨推理与生成的约束保障"]
benchmarks: ["VIS-Bench"]
---

# 论文速读：VIS-Ground: Video Interactive Storytelling with Contextual Grounding

## 一句话总结
本文提出 VIS-Ground 框架，通过结构化上下文抽象与约束诱导机制，解决视频交互式叙事中观众请求与源材料、已渲染视频状态之间的跨上下文依赖关系问题，在三个视频生成骨干上均取得最高综合得分。

## 研究问题与动机
- **核心问题**：视频交互叙事中，观众的新请求与已渲染视频前缀、接地源材料之间存在隐式依赖关系，这些关系不能从任一输入单独推断，需要联合推理。
- **现有方法不足**：视频故事生成系统主要处理初始脚本的连续生成，未考虑观众中途干预；交互视频生成强调响应新指令，但缺乏对源材料的忠实约束；两者均未解决"跨上下文联合接地"问题。
- **评测空白**：缺乏专门的视频交互叙事数据集与评测标准，无法系统评估源忠实性、故事连续性与交互满足度的联合表现。
- **动机来源**：形成性研究显示，有交互的知识接地视频 storytelling 学习增益提升 15.8 个百分点（47.9 vs 32.1），证明交互对理解的促进作用。

## 核心贡献（创新点）
- **结构化上下文抽象（Structured Context Abstraction）**：将接地源、视频前缀、观众请求三个异构输入转化为对齐的视频状态、故事状态、源状态及交叉上下文依赖模型，使实体、事件、声明及跨输入的依赖关系显式化——与现有方法仅总结各输入不同，本文恢复的是三者交互涌现的依赖关系。
- **候选条件约束诱导（Candidate-Conditioned Constraint Induction）**：将依赖集合按候选续写脚本动态激活为具体生成约束，无关依赖不强制施加——区别于现有方法直接优化最终视频指标或预定义约束类别，本文约束仅在被提议的未来事件关联时才实例化。
- **规划-验证-修订-渲染迭代流程**：约束同时作用于脚本规划与视频渲染两个阶段，实现跨推理与生成的统一验证——不同于现有工作仅在后处理阶段修正，本文约束在生成全流程中保持活跃。
- **VIS-Bench 基准**：构建首个视频交互叙事评测基准，包含 250 个源段落、1004 条交互指令、2008 个实例，涵盖叙事接地与知识接地两种设置——填补了该任务方向无专用评测的空白。

## 方法详解
VIS-Ground 将上下文接地视为"上下文编译问题"，分为三个阶段：

**1. 结构化上下文抽象**
- 使用 VLM 提取有类型谓词的 grounded facts，包含实体、属性、事件、关系、状态转换、时间范围及证据链接。
- 构建三种互补状态：
  - **视频状态 $\mathcal{V}$**：已渲染前缀中的实体、属性、位置、持有、动作及状态转换。
  - **故事状态 $\mathcal{H}$**：已完成事件、活动目标、未解决关系、因果依赖。
  - **源状态 $\mathcal{S}$**：接地材料支持的原子声明、事件、定义及关系，保留限定条件与证据。
- 推导交叉上下文依赖集合 $\mathcal{R} = \text{Derive}(\mathcal{M})$，包括时间先后、因果前提、状态转换要求、持续性关系、兼容性条件等。

**2. 候选条件约束诱导**
- 对候选脚本 $\mathcal{P}^{(k)}$，判断每个依赖是否被激活：$c_j = (s_j, a_j, b_j, e_j)$，其中 $s_j$ 为实体与时间范围，$a_j$ 为适用条件，$b_j$ 为必须成立的 Relation，$e_j$ 为支持证据。
- 约束激活条件：$a_j(\mathcal{P}^{(k)}, \mathcal{M}) = 1$；激活后要求在 $s_j$ 范围内 $b_j(\mathcal{P}^{(k)}, \mathcal{M})$ 成立。
- 评估结果：{pass, fail, unknown}，fail 可追溯至具体依赖并生成针对性修订反馈。

**3. 约束视频生成**
- **规划与脚本修订**：$\mathcal{P}^{(0)} \sim p_\theta(\mathcal{P}|\mathcal{M}, u)$；约束评估后，修订：$\mathcal{P}^{(k+1)} = \text{Revise}_\theta(\mathcal{P}^{(k)}, \mathcal{M}, F^{(k)})$，其中 $F^{(k)}$ 包含失败/未解决的依赖及证据；每次修订后重新计算约束激活。
- **渲染与输出修订**：$\nu_{\text{cont}} \sim p_\psi(\nu_{\text{cont}}|\mathcal{P}^*, \nu_{\text{prefix}}, \mathcal{G})$；提取渲染视频中实现的 relations，投影至同一依赖模型，若视频未实现某依赖（如对象转移未显示），则转换为输出级反馈，修订脚本后再次渲染。

## 实验与结果
- **数据集**：VIS-Bench，250 个源段落（132 叙事 + 118 知识），1004 条交互指令，2008 个实例，无训练集。
- **评估指标**：6 项语义/感知指标（0–100 分）：Source Faithfulness (SF)、Story Continuity (SC)、Interaction Fulfillment (IF)、Joint Constraint Satisfaction (JCS = (SF+SC+IF)/3)、Visual Consistency (VC)、Video Quality (VQ)；综合得分 = 叙事与知识设置 JCS 等权平均。
- **自动评测器**：Gemini 3.8 Flash + Claude Opus 5，双人独立评分取平均；人工验证 Krippendorff's α = 0.75–0.86。
- **基线**：Direct Generation、MM-StoryAgent、MovieAgent、VideoGen-of-Thought。
- **骨干模型**：Omni-1.1-Flash、Veo 3.1、MiniMax-H3。

| Backbone | VIS-Ground 综合得分 | 最强基线 | 提升幅度 |
|---|---|---|---|
| Omni-1.1-Flash | 84.2 | VideoGen-of-Thought (71.9) | **+12.3** |
| Veo 3.1 | 84.6 | MovieAgent (68.7) | **+15.3** |
| MiniMax-H3 | 78.8 | MovieAgent (75.4) | **+3.4** |

- 语义指标（SF/SC/IF/JCS）平均提升约 10%；跨骨干性能波动降低约 82%（Direct Generation 波动 >30 分，VIS-Ground <6 分）。
- 消融：Unstructured State 使 SF/SC/IF 均值下降 7.60pp；One-Pass Planning 下降 4.36pp。

## 相关工作脉络
- **MM-StoryAgent / MovieAgent / VideoGen-of-Thought**：多智能体或分层规划的视频故事生成方法，侧重于初始脚本到视频的端到端生成，缺乏对观众中途干预的响应及跨上下文联合接地机制。
- **StreamDiT / LongLive / StreamDiffusionV2**：交互流式视频生成，强调低延迟与 evolving prompts 响应，但未结合外部接地源进行源忠实性约束。
- **Genie / Oasis / Matrix-Game 2.0 / Vid2World**：交互世界模型，支持 action-conditioned 探索与可控世界事件，但聚焦于交互环境建模，而非叙事/知识接地视频续生成。
- **WBench**：评测交互世界模型的多轮能力，覆盖导航、主体动作、事件编辑等，但非视频故事叙事场景，且缺少源材料忠实性评估。
- **StoryBench / ViStoryBench**：故事可视化基准，分别评估动作执行/故事续写及图像序列叙事一致性，均不支持交互视角下的视频续生成任务。
- **Divide and Conquer / REFFLY / EchoFoley**：约束生成方法，分别针对词汇约束、歌词韵律、音频事件控制，约束来源为预定义规则，而非跨上下文联合推导。

## 局限性与未来方向
- **约束提取不完美**：依赖推导可能遗漏或错误推断，尤其隐含状态转换与跨模态对齐场景。
- **渲染保真度瓶颈**：视频生成器的固有能力限制输出质量，本文未改进合成模型本身，VQ 与 VC 仍有提升空间。
- **多目标权衡**：在歧义/激进请求下，框架优先保持与已有上下文兼容，可能牺牲 IF 的逐字满足。
- **计算预算**：多轮规划-验证-修订消耗较多 token 与 agent 调用（上限 48 次），实时性受限。
- **未来方向**：探索更高效的约束提取与验证机制、改进视频生成器的状态保持能力、拓展至更多模态（如音频、3D）的联合接地。

## 研究启发与可借鉴点
- **候选条件约束激活策略**：依赖仅在候选被提议的未来事件相关时才实例化，避免过度约束，可迁移至其他需动态约束的场景（如对话系统、规划任务）。
- **跨上下文依赖联合推导**：将多模态/多输入的信息整合为显式依赖模型，而非简单拼接，可为多源信息融合任务提供范式参考。
- **规划-验证-修订-渲染统一约束**：使用同一依赖模型贯穿推理与生成两阶段，实现端到端一致性保障，适用于任何需多阶段校验的生成任务。
- **评测维度的正交设计**：语义（SF/SC/IF/JCS）与感知（VC/VQ）分离评估，便于诊断不同能力瓶颈，值得在多模态生成评测中借鉴。

## 关键术语表
**视频交互叙事（Video Interactive Storytelling）**：观众可在视频生成过程中介入并引导后续情节发展的任务设定。
**接地源（Grounding Source）**：提供事实或叙事背景的外部材料，如故事大纲、科学论文、维基百科页面。
**视频前缀（Video Prefix）**：已渲染的先前视频片段，作为续生成的视觉与叙事上下文。
**结构化上下文抽象**：将异构输入转化为对齐的视频状态、故事状态、源状态及交叉依赖的表征过程。
**生成约束诱导**：根据候选续写脚本动态激活相关依赖，转化为可验证的生成约束。
**VIS-Bench**：本文提出的视频交互叙事评测基准，包含叙事与知识两种接地设置。
**Joint Constraint Satisfaction（JCS）**：源忠实性、故事连续性、交互满足度三项指标的等权平均。
**约束视频生成**：通过规划-验证-修订-渲染迭代流程，确保生成结果满足跨上下文依赖约束的方法。

## 可复现要素
- **数据集**：VIS-Bench，论文声明为 evaluation-only（无训练集），共 2008 个实例，项目页有展示。
- **代码/权重**：论文未明确声明开源，项目页 https://vis-ground.github.io 可进一步确认。
- **关键超参**：agent 调用上限 48 次，图片候选 3 张，输入 token 上限 600,000，输出/推理 token 上限 96,000；最多 5 次脚本修订、最多 3 次评审、最多 2 次修复；音频审核最多 3 次。
- **模型配置**：Planner 使用 gemini-3.1-pro-preview；参考图生成使用 gemini-2.5-flash-image；评测器使用 gemini-3.8-flash 与 claude-opus-5。
