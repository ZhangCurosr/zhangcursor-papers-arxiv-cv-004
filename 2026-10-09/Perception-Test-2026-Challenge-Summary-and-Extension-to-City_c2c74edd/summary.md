---
title: "Perception-Test-2026-Challenge-Summary-and-Extension-to-City"
source: https://arxiv.org/pdf/2610.12081v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:24:40"
field: "多模态空间智能"
keywords: ["spatial reasoning", "multimodal models", "long video understanding", "audio-visual reasoning", "benchmark", "agentic pipeline"]
innovations: ["提出 KilometerVision 城市尺度空间智能 benchmark，覆盖地图追踪、距离估计等7类任务", "首次发布 KilometerAudio 基准，针对数小时长视频的音频导向多模态推理进行系统评估", "揭示零样本智能体管道与单一多模态模型在城市尺度推理上的性能鸿沟"]
benchmarks: ["KilometerVision", "KilometerAudio", "Perception Test", "Video-MME"]
---

# 论文速读：Perception-Test-2026-Challenge-Summary-and-Extension-to-City

## 一句话总结
本文总结了 ECCV 2026 Perception Test 挑战赛第4届的成果，新增了两个面向城市尺度（最长视频达数小时）的空间智能与音频-视觉推理赛道，验证了前沿多模态模型在长视频空间理解能力上的进展与不足。

## 研究问题与动机
- **前沿模型感知能力仍不足**：尽管 AI 在数学、编程等领域达到超人类水平，但在环境感知（尤其空间理解）方面仍落后于普通人类，可能与模型主要依赖被动互联网数据训练有关。
- **长视频空间推理缺乏评估基准**：现有 benchmark 视频较短，缺乏对城市尺度（步行数公里）下时空推理能力的系统评估。
- **多模态模型在空间任务上的泛化性待验证**：单一模型能否同时在多种空间推理任务（地图追踪、距离估计、闭环检测等）上取得可接受的表现尚不明确。
- **音频-视觉联合推理在长视频中的挑战**：真实场景下声音常来自画面外事件，如何结合长时序音视频进行推理仍是开放问题。

## 核心贡献（创新点）
1. **发布 KilometerVision 基准**：基于 YouTube 步行游览视频（最长10分钟，约1公里），设计了7种空间推理任务（地标识别、距离估计、指南针、闭环、路线摘要、地图追踪、欧氏距离），填补城市尺度空间智能评测空白。
   - 与原有 Perception Test 的区别在于从桌面级（table-top）扩展到城市级环境，视频时长与空间规模提升两个数量级。

2. **首次提出 KilometerAudio 基准**：针对数小时长的城市步行视频，构建全音频导向的多选题 QA，考察模型对画面外声音的识别、定位与多跳推理能力。
   - 与现有音频视频基准（如 Video-MME Long）的本质区别在于 KilomterAudio 聚焦音频事件且视频显著更长（平均 >70 分钟 vs. 少数音频相关题目）。

3. **组织四赛道挑战赛并记录最新进展**：统一 VideoQA、Grounded VideoQA、KilometerVision、KilometerAudio，对比历届结果，发现零样本智能体管道已接近完美，但单一模型在城市场景仍表现有限。
   - 与往届的核心差异是首次引入城市尺度赛道，揭示了"工具增强智能体"与"端到端多模态模型"之间的性能鸿沟。

## 方法详解
- **KilometerVision 任务配置**：
  - 标准多选题（5个文本选项）
  - 图像选项（候选路线地图）
  - 包含参考图像的问题（如地标距离估计）
  - 问题类型包括：landmark recognition、distance to landmark、compass、loop closure、route summary、map trace（带/不带文字标签）、Euclidean distance

- **KilometerAudio 任务设计**：
  - 使用500道5选项问题，覆盖218段1-3小时的 YouTube 步行游览视频
  - 音频来源包括人造声音（汽车、摩托车、教堂钟声）、自然声（鸟鸣、狗叫）、语音、城市噪音
  - 强调音频事件常来自画面外，需跨模态推理

- **冠军方案核心思路**：
  - **Unified VideoQA (Njust-KMG)**：问题类型感知的加权集成，融合 Qwen3-VL-32B（fine-tuned）与 Doubao/Gemini API（zero-shot）
  - **Grounded VideoQA (Banana Bats)**：Gemini 3.7 Flash 生成跟踪计划 + SAM 3 双向传播 + 可信度检查
  - **KilometerVision (CloudAI-evolve)**：无限制 token 预算的智能体管道，探索验证集 + 少量测试样本，结合 VGGT 3D重建、对象尺寸缩放、地图接地
  - **KilometerAudio (IUCV)**：训练无关的 AVRA 智能体，冻结 BEATs/CLIP 模型构建索引，运行时精确定位候选时间段并直接检查原始音视频

## 实验与结果
- **数据集规模**：
  - KilometerVision: 验证集 6视频/17题，测试集 54视频/986题
  - KilometerAudio: 验证集 6视频/8题，测试集 218视频/500题（无训练数据，纯 zero-shot）

- **主要结果**：
  | 赛道 | 随机基线 | 冠军（获奖） | 最佳表现（未获奖） |
  |---|---|---|---|
  | Unified VideoQA | 0.313 | 0.914 (Njust-KMG) | — |
  | Grounded VideoQA | 0.052 HOTA | 0.664 (Banana Bats) | — |
  | KilometerVision | 0.196 | 0.753 (ohmyyuan) | 0.905 (CloudAI-evolve) |
  | KilometerAudio | 0.174 | 0.782 (IUCV) | 0.864 (CloudAI-evolve) |

- **关键结论**：除 Grounded VideoQA 外，其他三个赛道性能接近饱和（>80%），城市尺度赛道首次举办即达高水准；通用模型（Wind_Rain_Tower，Qwen3.8-Max 单模型）在 Unified VideoQA 达 0.896，但在城市场景仅达 60-65%，表明复杂空间/多模态推理仍需依赖工具增强的智能体管道。

## 相关工作脉络
- **Perception Test 系列**（Pătrăucean et al., 2023; Heyward et al., 2024, 2026）：本工作的基础，从桌面级扩展到城市尺度。
- **Video-MME Long**（Fu et al., 2025）：最长视频达1小时的 multimodal benchmark，但仅少部分题目聚焦音频。
- **KilometerVision**（Mahendran et al., 2026）：本文引用的城市空间智能基准，本研究将其纳入挑战赛。
- **SAM 2 / SAM 3**（Ravi et al., 2025; Carion et al., 2025）：被 Grounded VideoQA 冠军广泛采用的可提示跟踪器。
- **BEATs / CLAP**（Chen et al., 2023）：用于 KilometerAudio 音频索引的预训练模型。
- **VGGT**（Wang et al., 2025）：视觉几何 grounding transformer，被用于千米级视频的场景重建。

## 局限性与未来方向
- **城市场景推理仍依赖昂贵智能体管道**：单一多模态模型无法独立解决 kilometer-scale 任务，效率与成本瓶颈明显。
- **物理推理仍是难点**：Unified VideoQA 中最难问题集中在直觉物理（如水温判断、速度/摩擦变化预测）。
- **无训练数据限制**：城市尺度赛道为纯 zero-shot，模型泛化能力验证更严格，但也限制了 fine-tuning 优化空间。
- **未来方向**：提升端到端多模态模型的空间/音频推理能力、降低智能体管道成本、探索跨尺度（从桌面到城市）的统一感知模型。

## 研究启发与可借鉴点
- **问题类型感知的集成策略**（Njust-KMG）：针对不同题型分别调用最优模型/API，并按验证集表现加权融合，可有效突破单一模型瓶颈。
- **冻结模型构建索引 + 运行时精调检索**（IUCV AVRA）：利用 BEATs/CLIP 等冻结模型预先构建音视频索引，在推理时仅针对候选窗口做精细分析，兼顾效率与准确性。
- **确定性几何估计器覆盖投票结果**（ohmyyuan）：在 LLM 多数投票基础上引入光流航位推算和视觉定位等确定性工具，可纠正"幻觉"错误。
- **证据优先提示策略**（F423）：要求模型先提取视觉/时间/音频证据再比较选项，并限制重查询次数，可减少随机性并提高可解释性。

## 关键术语表
- **KilometerVision**：基于城市步行游览视频的长视频空间智能 benchmark，涵盖地图追踪、距离估计等7类任务。
- **KilometerAudio**：专为城市长视频设计的全音频导向多模态推理 benchmark，视频时长平均超过70分钟。
- **HOTA**（Higher Order Trace Accuracy）：多目标跟踪评估指标，同时衡量检测精度（DetA）与关联精度（AssA）。
- **Agentic pipeline**：通过 orchestrate 多个模型/工具（检索、推理、验证）完成复杂任务的系统架构。
- **Zero-shot evaluation**：不提供训练数据，模型仅依赖预训练知识完成测试任务。
- **Loop closure**：判断视频中是否回到起点或之前经过地点的空间推理任务。
- **Amodal tracking**：在目标被遮挡时仍能推断完整轨迹的跟踪能力。

## 可复现要素
- **数据集**：KilometerVision 和 KilometerAudio 基于公开 YouTube 视频；KilometerAudio 测试集无训练数据（zero-shot）。
- **代码/权重**：论文未提供开源代码，解决方案细节需查阅 workshop 网站（https://perception-test-challenge.github.io/）。
- **关键超参**：论文未详细披露，各队伍技术方案见团队报告链接。
