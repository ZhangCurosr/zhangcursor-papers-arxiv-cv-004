---
title: "PolyOCR-Venus-Unified-OCR-Foundation-Models-for-Text-Centric"
source: https://arxiv.org/pdf/2609.37712v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:35:14"
field: "OCR with Vision-Language Models"
keywords: ["Unified OCR", "Foundation Model", "Competence-Guided Policy Optimization", "Multimodal Large Language Model", "Document Parsing", "OCRBench"]
innovations: ["Proposes PolyOCR, a unified OCR foundation model family integrating recognition, localization, parsing, extraction, understanding and reasoning in one generative framework.", "Introduces Competence-Guided Policy Optimization (CGPO) that dynamically routes on-policy distillation based on calibrated teacher quality and teacher-student competence gap.", "Builds a large-scale OCR data engine with cross-model verification and targeted synthesis to cover underrepresented capabilities."]
benchmarks: ["OCRBench v2.1", "CC-OCR", "OmniDocBench v1.6", "MDPBench", "In-house KIE Benchmark"]
---

# 论文速读：PolyOCR-Venus: Unified OCR Foundation Models for Text-Centric Visual Intelligence

## 一句话总结
PolyOCR 是一族统一 OCR 基础模型（含 2B/9B 通用变体及 2.7B 多语言专用变体），通过大规模数据引擎构建覆盖六维 OCR 能力的训练语料，并提出能力引导策略优化（CGPO）框架，在 OCRBench v2.1、CC-OCR、OmniDocBench 等多个基准上实现 SOTA 或极具竞争力的性能。

## 研究问题与动机
- **现有 OCR 系统能力碎片化**：专业 OCR 模型针对特定领域（场景文本、文档解析、表格理解等）优化，任务 formulate、架构、监督格式各异，导致知识难以跨异构 OCR 场景迁移，能力发展碎片化。
- **通用 MLLM 的 OCR 能力隐式且不平衡**：通用多模态大模型（如 GPT-4V、Qwen3-VL）的 OCR 能力通过大规模多模态预训练间接获得，缺乏针对细粒度文本感知、空间定位、密集文档理解等 OCR 核心能力的系统性优化，性能在不同 OCR 能力维度上波动大。
- **统一训练面临异质任务协同难题**：OCR 任务在视觉复杂度、输出结构、监督粒度和优化目标上差异显著（识别需精确序列生成、定位需空间 grounding、结构化提取需格式一致性、推理需语义理解），固定采样或静态优化易导致优势任务压制弱势任务，且存在灾难性遗忘风险。
- **高质量训练数据覆盖不足**：公开 OCR 数据集在空间定位、结构化解析、多语言、长尾复杂场景等方面覆盖有限，且人工标注成本高，制约了统一 OCR 基础模型的能力上限。

## 核心贡献（创新点）
- **提出 PolyOCR 统一 OCR 基础模型家族**：将文本识别、定位、文档解析、信息提取、视觉文本理解和 OCR 推理纳入统一的指令遵循与自回归生成框架，实现跨任务知识共享。*与已有工作的本质区别*：区别于仅聚焦单一任务的专业 OCR 模型或依赖隐式 OCR 能力的通用 MLLM，PolyOCR 明确以统一生成范式系统性整合全谱系 OCR 能力。
- **构建大规模 OCR 数据引擎**：通过可控合成与模型辅助标注，将异构视觉资源转化为质量验证的多任务监督数据，覆盖六大能力维度，解决公开数据覆盖不全和长尾场景不足的问题。*与已有工作的本质区别*：不同于仅收集公开数据集的方法，该引擎针对薄弱能力（如双语翻译、模板化 KIE、复杂文档解析）主动合成和校正数据，提供任务导向的质量控制流程。
- **提出能力引导策略优化（CGPO）框架**：结合基于验证器的 Group Relative Policy Optimization（GRPO）和基于样本路由的 on-policy 蒸馏（OPD），动态调整蒸馏权重以适应教师-学生能力差距和教师可靠性。*与已有工作的本质区别*：区别于固定权重蒸馏或仅依赖 GRPO 的方法，CGPO 通过离线校准的 competence routing 实现样本级自适应监督分配，平衡强化学习探索与教师知识迁移。
- **引入 OCRBench v2.1 等修订基准**：对 OCRBench v2 进行人工标注修正和任务对齐的评分指标修订，提供更可靠、一致的 OCR 评估协议。*与已有工作的本质区别*：相比原始 OCRBench v2 可能存在标注噪声和指标不匹配问题，v2.1 通过系统化审计和规则优化提升了评估的鲁棒性和任务对齐度。
- **发布多尺度模型与开源资源**：提供 2B/9B 通用模型和 2.7B 多语言专用模型，并公开代码、权重及部分基准，推动社区研究。*与已有工作的本质区别*：相比单一规模模型，PolyOCR 家族覆盖不同部署需求，且多语言变体专门优化 21 种语言/脚本组。

## 方法详解
- **统一指令遵循框架**：所有 OCR 任务（识别、定位、解析、提取、推理、翻译）通过共享的 prompt template 和 autoregressive generation 接口处理，输出格式包括纯文本、坐标、JSON、Markdown、HTML、LaTeX 等，由任务特定的 verifier 进行验证。
- **六维能力数据组织**：训练语料按 Text Recognition、Text Localization、Document Parsing、Information Extraction、Visual Text Understanding、OCR Reasoning 六大维度组织，确保全面覆盖。
- **数据引擎三大合成流水线**：
  - **场景文本与翻译合成**：利用 PP-OCRv6 和 LocateAnything 交叉验证自然场景文本定位与识别；通过渲染生成可控合成数据；使用 Qwen3.5-9B 生成中英双语翻译并过滤。
  - **结构化 KIE 与文档/表格合成**：KIE 数据通过 VLM 重建 HTML/CSS 模板后替换内容生成；文档解析采用 k-means 聚类页面复杂度，双模型（PaddleOCR-VL 与 MinerU2.5-Pro）输出一致性校验；表格解析类似地结合 TEDS 分数筛选。
  - **推理导向标注合成**：利用确定性模板从高质量 OCR 标注生成计数任务 CoT；使用 DeepSeek-V4-Pro 生成复杂 VQA 推理数据并验证。
- **质量三重验证**：cross-model agreement validation、structure-aware validation（检测重叠、缺失等）、source-grounded validation（确保生成内容忠实于源标注）。
- **两阶段训练**：
  - **SFT 阶段**：60M 实例按三阶段课程分配，TP/DS/OR/GA 比例分别为 (40%/35%/10%/15%) → (32.5%/30%/22.5%/15%) → (25%/25%/35%/15%)，LR=1e-5 cosine decay，BF16 精度，更新语言模型参数、冻结视觉编码器。
  - **CGPO 阶段**：基于 SFT checkpoint，LR=1e-6，teacher 为 Qwen3.5-122B-A10B（冻结）。
- **CGPO 核心机制**：
  - **能力估计**：教师能力 \(c_T(x) = \frac{1}{K}\sum_{k=1}^K r_{T,k}(x)\)，学生能力 \(c_S(x) = \frac{1}{G}\sum_{i=1}^G r_{S,i}(x)\)，能力差距 \(\Delta(x) = c_T(x) - c_S(x)\)，其中 \(r\) 为任务特定 verifier 输出的 reward。
  - **离线路由校准**：在固定校准集上拟合两个 sigmoid 映射 \(\hat{q}_T^A(x) = \sigma(a_T \hat{c}_T^A(x) + b_T)\) 和 \(\hat{p}_{win}^A(x) = \sigma(a_\Delta \hat{\Delta}^A(x) + b_\Delta)\)，最小化 task-balanced soft-label BCE。
  - **样本级路由系数**：\(\alpha(x) = \mathrm{sg}[\hat{q}_T(x)(2\hat{p}_{win}(x)-1)_+]\)，当教师优势不显著时 \(\alpha(x)=0\)。
  - **联合损失**：\(\mathcal{L}_{CGPO} = \mathcal{L}_{GRPO} + \lambda \alpha(x) \mathcal{L}_{OPD}\)，其中 \(\mathcal{L}_{OPD}\) 为 reverse KL divergence \(D_{KL}(p_\theta \| p_T)\)，\(\lambda=1\)，禁用 reference-policy KL penalty。

## 实验与结果
- **评测基准**：OCRBench v2.1（修订版）、CC-OCR、in-house KIE Benchmark、OmniDocBench v1.6、MDPBench（多语言文档解析）。
- **主要结果**：
  - **OCRBench v2.1**：PolyOCR-9B 英语 80.42、中文 78.37，优于 Qwen3.5-9B（66.64/71.88）、GLM-4.6V-Flash（67.89/75.34）等；PolyOCR-2B 英语 72.33、中文 71.32。在 13 个子类别中，英语 TR/TS/RE/EP/VTU 等多项第一，中文 EP/KR 领先。
  - **CC-OCR**：PolyOCR-9B 总分 82.43（Doc Parsing 71.71、KIE 94.02 最优），PolyOCR-2B 80.53，超越 HunyuanOCR、OvisOCRv2 等专业模型。
  - **OmniDocBench v1.6**：PolyOCR-9B 整体 91.57，使用 layout-guided pipeline 后达 95.52（公式 CDM 96.30、表格 TEDS 93.56），接近专业解析系统。
  - **In-house KIE**：PolyOCR-9B 整体 F1 93.60%，在四种证件类型上均表现最佳或次优（入学通知 88.35% 最高，身份证 99.20% 第二）。
  - **MDPBench**：PolyOCR-2.7B 整体 87.0%，数字文档 91.0%，拍照文档 85.6%，超越 MonkeyOCRv2-B-Parsing（83.3）3.7 个百分点，在 6/17 语言组中得分最高。
- **最强结果与提升**：OCRBench v2.1 英语 80.42（较 SFT checkpoint 提升 8.34），CC-OCR 82.43（较 SFT 提升 2.97），MDPBench 整体 87.0（较 MonkeyOCRv2 提升 3.7）。
- **CGPO 消融**：完整 CGPO 在 OCRBench v2.1、CC-OCR、OmniDocBench 上均优于 GRPO alone、OPD alone、固定权重/余弦衰减 OPD 等变体，验证了能力路由的有效性。

## 相关工作脉络
- **专业 OCR 系统**（如 EAST、CRNN、TrOCR、PARSeq）：针对单一任务（检测、识别、spotting）优化，架构与监督格式孤立，知识难以迁移。PolyOCR 定位为统一框架，打破任务边界。
- **文档智能模型**（LayoutLM、Nougat、MonkeyOCR、Dolphin、MinerU2.5、PaddleOCR-VL）：侧重页面级解析（布局、阅读顺序、结构化重建），但缺乏对场景文本、多语言、OCR 推理的整合。PolyOCR 将其扩展至全谱系 OCR 能力。
- **通用多模态大模型**（Gemini、Qwen3-VL、InternVL3.5、Ovis2.5）：提供统一接口但 OCR 能力隐式获得且不平衡，在细粒度文本感知、空间定位等任务上表现不稳定。PolyOCR 明确以 OCR 为核心目标进行系统性优化。
- **OCR 导向 VLM**（TextMonkey、GOT-OCR2.0、HunyuanOCR、OCRVerse、Qianfan-OCR）：部分统一了 OCR 任务，但优化目标仍有侧重（如重识别定位或重文档解析），且缺乏对抗异构监督的动态策略。PolyOCR 通过数据引擎和 CGPO 实现更均衡的全能力训练。
- **数据构建与后训练方法**（OLMOCR 2、Infinity-Parser2、OvisOCR2、SCOPE、Reward-Gated OPD）：使用 verifier reward 或固定权重蒸馏。PolyOCR 的 CGPO 创新在于基于教师-学生能力差距的动态路由，避免静态分配导致的过拟合或干扰。

## 局限性与未来方向
- **局限**：① 模型在数学计算、中文文本识别、中文视觉文本理解等少数子任务上未达到绝对最优（如 OCRBench v2.1 中 MC 和中文 KR 落后于部分基线）；② 多语言变体 PolyOCR-2.7B 参数量较小（2.7B），与 9B 通用模型相比在通用 OCR 基准上未直接对比；③ CGPO 依赖教师模型（Qwen3.5-122B-A10B）的离线能力估计，计算开销较大；④ 数据引擎的合成数据可能引入分布偏移，尽管有质量控制但仍需更多真实场景验证。
- **未来方向**：① 增强基础 OCR 能力（细粒度文本感知、定位、识别）；② 提升语义理解与推理深度，超越纯文本提取；③ 提高模型在复杂真实环境中的泛化性与鲁棒性。

## 研究启发与可借鉴点
- **能力维度驱动的数据组织**：将 OCR 任务划分为六维能力并据此构建训练语料，而非简单混合任务数据，有助于系统性地覆盖长尾能力。
- **动态监督路由机制**：CGPO 的 sample-wise competence routing 思想可迁移至其他多任务后训练场景（如多模态理解、代码生成），通过教师-学生能力差距自适应分配蒸馏权重，避免固定 schedule 的次优性。
- **任务特定 verifier 的统一 reward 空间**：将编辑距离、IoU、TEDS、F1、LLM-judge 等不同指标映射为 [0,1] 的 scalar reward，使 GRPO 能统一优化异构任务，该方法论可扩展至其他需要混合评估标准的领域。
- **高质量数据引擎与交叉验证**：利用多个独立模型预测进行交叉验证（如 PP-OCRv6 与 LocateAnything 交叉验证场景文本），并人工复核分歧样本，以提升合成数据的可靠性，可借鉴于其他视觉数据构建流水线。
- **布局引导 pipeline 提升解析性能**：OmniDocBench 实验中结合外部布局检测器（PP-DocLayoutV3）显著提升了公式和表格解析分数，提示统一模型可与专用模块灵活组合，兼顾泛化与专精。

## 关键术语表
- **PolyOCR**：Ant Group 提出的统一 OCR 基础模型家族，包含 2B/9B 通用变体和 2.7B 多语言专用变体，基于 Qwen3.5 架构。
- **CGPO（Competence-Guided Policy Optimization）**：能力引导策略优化，结合 GRPO 与 on-policy 蒸馏（OPD）并通过样本级路由系数动态调整蒸馏权重的后训练框架。
- **GRPO（Group Relative Policy Optimization）**：Group Relative Policy Optimization，一种强化学习算法，通过组内相对奖励计算 advantage 来优化策略。
- **OPD（On-Policy Distillation）**：On-policy 蒸馏，在策略当前生成的轨迹上进行 teacher-student 知识迁移，使用 reverse KL divergence。
- **OCRBench v2.1**：作者修订的 OCRBench v2 基准，包含人工校正的标注和任务对齐的评估指标，提供更可靠的 OCR 能力评测。
- **CC-OCR**：综合 OCR 基准，评估真实场景下的文本读取、文档解析和关键信息提取能力。
- **OmniDocBench v1.6**：专注文档解析的基准，评估布局分析、表格理解、公式识别和全文档内容重建。
- **MDPBench**：多语言文档解析基准，覆盖 17 种语言/脚本组，评估跨书写系统的 OCR 与解析能力。

## 可复现要素
- **数据集**：训练数据约 60M 高质量 OCR 实例，包含公开数据集与内部数据；OCRBench v2.1 标注修订与评估代码已公开；其他基准（CC-OCR、OmniDocBench、MDPBench、in-house KIE）使用官方或内部数据集。
- **代码与权重**：GitHub 仓库 https://github.com/inclusionAI/PolyOCR-Venus，模型权重已开源（论文未明确说明具体许可）。
- **关键超参**：SFT 阶段 LR=1e-5 cosine decay，batch size=256，序列长度 20480 tokens；CGPO 阶段 LR=1e-6，temperature=0.9，GRPO clip ε=0.2，教师能力估计 K=8 rollouts，学生能力估计 G=8 rollouts，OPD 系数 λ=1，禁用 reference-policy KL penalty（β=0）。
