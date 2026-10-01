---
title: "PolyOCR-Venus-Unified-OCR-Foundation-Models-for-Text-Centric"
source: https://arxiv.org/pdf/2609.37712v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:35:18"
field: "多模态视觉-语言模型与OCR"
keywords: ["OCR", "Unified Foundation Model", "GRPO", "On-Policy Distillation", "CGPO", "Document Parsing", "Multilingual OCR", "Benchmark"]
innovations: ["提出PolyOCR统一OCR基础模型家族（2B/9B）覆盖六维OCR能力", "设计能力引导的策略优化CGPO，动态路由GRPO与on-policy蒸馏权重", "构建大规模OCR数据引擎并推出OCRBench v2.1修订版评测协议"]
benchmarks: ["OCRBench v2.1", "CC-OCR", "OmniDocBench v1.6", "MDPBench", "In-house KIE Benchmark"]
---

# 论文速读：PolyOCR-Venus: Unified OCR Foundation Models for Text-Centric Visual Intelligence

## 一句话总结
本文提出 **PolyOCR**，一个统一的 OCR 基础模型家族（2B/9B，基于 Qwen3.5），通过大规模 OCR 数据引擎与能力引导的策略优化（CGPO）框架，实现文本识别、定位、解析、信息提取、理解与推理的 unified 生成式建模；在多语言变体（PolyOCR-2.7B）及 OCRBench v2.1、CC-OCR、OmniDocBench v1.6、MDPBench 等基准上取得 SOTA 或高度竞争的性能。

## 研究问题与动机
- **现有 OCR 系统能力碎片化**：专用 OCR 系统（场景文本、文档解析、表格理解等）各自独立建模，任务格式与架构差异大，难以跨场景迁移；通用 MLLM 的 OCR 能力仅隐式获得，在细粒度识别、空间定位、密集文档理解等方面表现不均衡。
- **统一训练的优化挑战**：OCR 任务在视觉复杂度、输出结构、监督粒度、优化目标上差异显著，固定采样或静态权重易导致主导任务淹没弱覆盖/困难任务，且多任务正迁移与干扰并存、顺序适应可能引发灾难性遗忘。
- **数据覆盖不均**：现有公开与内部 OCR 数据集在复杂布局、长尾场景（空间定位双语翻译、模板化 KIE、中文文档/表格解析、场景文本定位、OCR 导向推理等）覆盖不足。
- **评测体系需完善**：OCRBench v2 存在参考标注噪声与任务对齐评分不一致的问题，需更可靠、任务对齐的评测协议。

## 核心贡献（创新点）
- **提出 PolyOCR 统一 OCR 基础模型家族**：将文本识别、定位、解析、信息提取、视觉文本理解与 OCR 推理统一于共享指令遵循 + 自回归生成框架；特有 PolyOCR-2.7B 多语言专用变体（ViT 1.2B + Qwen2.5 1.5B）覆盖 21 种语言/脚本组。*与已有工作的本质区别在于：统一生成式接口同时覆盖从底层感知到高层推理的全谱 OCR 能力，而非单一任务专用模型。*
- **构建大规模 OCR 数据引擎**：通过可控合成与模型辅助标注，将异构视觉资产转化为质量验证的多任务监督，产出约 60M 高质量训练样本；同时提出 OCRBench v2.1（修正标注 + 任务对齐评分）。*本质区别：从任务/领域碎片化数据转向按六大能力维度组织的统一训练语料。*
- **提出 Competence-Guided Policy Optimization (CGPO)**：将基于验证器的 GRPO 与 on-policy 蒸馏（OPD）结合，通过教师可靠性与师生能力差距的动态路由样本级分配 OPD 权重，防止灾难性遗忘并平衡异构任务学习。*与已有工作（如固定权重 OPD、cosine 衰减、SCOPE/Reward-Gated OPD）的区别在于：首次显式建模教师-学生动态能力边界并以校准的胜算概率联合调制蒸馏强度。*
- **系统性跨基准评测验证**：在 OCRBench v2.1、CC-OCR、in-house KIE Benchmark、OmniDocBench v1.6、MDPBench 上对比广泛基线，PolyOCR-9B 在多类 OCR 子任务中取得第一或第二。

## 方法详解
### 1. 模型架构
- **PolyOCR-2B / 9B**：基于 Qwen3.5 视觉-语言主干，统一指令-跟随 + 自回归生成接口。
- **PolyOCR-2.7B**（多语言专用）：ViT 视觉编码器（1.2B 参数）+ Qwen2.5 文本解码器（1.5B 参数），面向多样化书写系统与采集条件优化。
- 输入图像区域 MAX_PIXELS = 1,048,576；最大序列长度 20,480 tokens；SFT 精度 BF16。

### 2. 六维能力划分与数据引擎
按六大能力维度组织训练语料：Text Recognition、Text Localization、Document Parsing、Information Extraction、Visual Text Understanding、OCR Reasoning。
数据引擎三条合成管线：
- **场景文本与翻译合成**：真实图像用 PP-OCRv6 + LocateAnything 交叉验证定位/识别；无文本图像渲染合成，控制字体、透视、模糊、压缩等。双语翻译：Qwen3.5-9B 生成中英互译，经语言一致性/数值/命名实体保持/重复/长度约束过滤。
- **结构化 KIE & 文档 & 表格合成**：KIE 使用模板保持合成（VLM 重建 HTML/CSS → LLM 替换 KV 内容 → 渲染）；文档解析通过 PP-DocLayoutV3 聚类页面复杂度 + PaddleOCR-VL / MinerU2.5-Pro 双模型一致性校验 + Qwen-3.5 纠错；表格定位+解析类似流程，保留 TEDS 一致性高样本。
- **推理导向标注合成**：字符计数通过确定模板从已验证 OCR 标注生成 CoT；复杂 OCR-VQA 使用 DeepSeek-V4-Pro 生成候选问题/解释/答案，经一致性检查过滤。
- **质量门禁三阶段**：跨模型一致性校验、结构感知校验（元素重叠/溢出/非法结构）、源接地校验（翻译/推理链是否源于源标注）；最终约 60M 高质量样本。

### 3. 两阶段训练
- **阶段一 SFT**：60M 样本分 TP/DS/OR/GA 四组，比例 32.5%/30.0%/22.5%/15.0%，三阶段课程学习（Early 重 TP+DS，Late 增加 OR 至 35%）；学习率 1e-5 cosine 衰减，warmup 5%，global batch 256，1 epoch。
- **阶段二 CGPO**：学习率 1e-6，teacher 为 Qwen3.5-122B-A10B（冻结）。

### 4. CGPO 核心设计
**验证器**：按任务类型映射标量奖励 Rτ ∈ [0,1]：
- 文本识别：归一化编辑距离 NED
- 空间定位：IoU
- 公式识别：CDM（字符检测匹配）
- 表格解析：TEDS
- KIE：字段级 F1
- VQA/翻译：LLM judge（Gemini-3.5-Flash-Lite）

**能力估计**（离线/在线）：
- 教师能力 $c_T(x) = \frac{1}{K}\sum_{k=1}^{K} r_{T,k}(x)$，K=8 次离线采样
- 学生能力 $c_S(x) = \frac{1}{G}\sum_{i=1}^{G} r_{S,i}(x)$，G=8 次在线 rollout
- 能力差 $\Delta(x) = c_T(x) - c_S(x)$

**离线路由校准**：在固定校准集上拟合两个单调 sigmoid 映射：
- $\widehat{q}_T(x) = \sigma(a_T \widehat{c}_T^A(x) + b_T)$：教师独立质量
- $\widehat{p}_{win}(x) = \sigma(a_\Delta \widehat{\Delta}^A(x) + b_\Delta)$：校准后的师生胜算概率（tie 计 0.5）
- 损失：任务平衡 soft-label BCE，权重反比于任务样本数

**样本级路由系数**：
$$\alpha(x) = \text{sg}\!\left[\widehat{q}_T(x)\cdot (2\widehat{p}_{win}(x)-1)_+\right] \in [0,1]$$
当 $\widehat{p}_{win} \le 0.5$ 时 OPD 权重归零。

**OPD 目标**（reverse KL）：
$$\mathcal{L}_{\text{OPD}}(x,y_S) = \frac{1}{|y_S|}\sum_{t=1}^{|y_S|} D_{KL}(p_\theta(\cdot|h_t) \| p_T(\cdot|h_t))$$
教师在冻结状态，梯度仅更新学生。

**联合目标**：
$$\mathcal{L}_{\text{CGPO}}(x) = \mathcal{L}_{\text{GRPO}}(x) + \lambda \cdot \alpha(x) \cdot \mathcal{L}_{\text{OPD}}(x), \quad \lambda=1$$
GRPO clip ε=0.2，禁用 reference-policy KL（β=0）。

## 实验与结果
- **评测基准**：OCRBench v2.1（en/zh）、CC-OCR、in-house KIE Benchmark、OmniDocBench v1.6、MDPBench（17 种语言/脚本组）。
- **主要结果（PolyOCR-9B）**：
  - OCRBench v2.1：en 80.42 / zh 78.37（en TR 82.99 第一、TD 70.84 第一、EP 89.70 第一、VTU 87.58 第一；zh EP 86.86 第一）
  - CC-OCR：82.43（Doc Parsing 71.71 第一、KIE 94.02 第一）
  - OmniDocBench v1.6：91.57（布局引导 pipeline 95.52）
  - In-house KIE：93.60 F1（Admission Notice 88.35 第一；ID Card 99.20 并列第二）
  - MDPBench（PolyOCR-2.7B）：87.0 总体第一，超 MonkeyOCRv2-B-Parsing 3.7pp；数字文档 91.0，照片文档 85.6（并列第一）；拉丁/非拉丁平均 88.8%/84.9% 均第一。
- **CGPO 消融（Table 9）**：SFT→+GRPO (+2.55/1.53 en/zh) →+OPD (+1.76/1.06) →+GRPO+OPD 固定权 (+3.34/2.37) →+cosine (+6.15/1.60) →**CGPO (+8.34/3.66)**。CGPO 相对 cosine 提升 en +2.19、zh +2.06、CC-OCR +0.11、OmniDoc +1.45。
- **结论**：CGPO 显著优于固定/cosine 蒸馏基线；统一框架在多类 OCR 能力维度上保持均衡高水平。

## 相关工作脉络
- **专用 OCR 系统**（EAST、CRNN、TrOCR、PARSeq、HunyuanOCR、OvisOCRv2 等）：任务/架构高度特化，难以跨场景知识迁移；本文以统一生成接口覆盖全谱 OCR 任务。
- **文档智能模型**（LayoutLM 系列、Donut、Nougat、MonkeyOCR、Dolphin、MinerU2.5、PaddleOCR-VL）：侧重页面级结构化解析；本文在此基础上扩展至场景文本、多语言、推理任务，并提供统一训练范式。
- **通用 MLLM**（Gemini、Qwen3-VL、InternVL3.5、GPT 系列）：OCR 能力隐式获得，细粒度识别/定位/结构化输出不均衡；本文通过任务对齐数据 + CGPO 显式优化。
- **OCR 导向 VLM**（TextMonkey、GOT-OCR2.0、HunyuanOCR、OCRVerse、Qianfan-OCR）：已转向统一生成范式，但优化目标仍侧重特定能力族；本文进一步强调跨能力均衡与能力感知训练策略。
- **数据构建与后训练**（olmOCR2、Infinity-Parser2、OvisOCR2、SCOPE、Reward-Gated OPD、GKD）：验证器奖励 + 在线蒸馏是共同方向；本文 CGPO 的独特贡献是引入能力差距驱动的样本级校准路由，而非固定/cosine 或仅 outcome-gated。
- **多任务自适应优化**（Grad-Norm、PCGrad、DoReMi、Skill-it!、TwSG）：关注任务损失平衡/梯度冲突/课程安排；本文聚焦 OCR 场景下师生动态能力边界的显式建模与蒸馏路由。

## 局限性与未来方向
- **局限**：数学计算、中文文本识别、中文视觉文本理解等子任务仍低于最强基线（OCRBench v2.1 Table 4）；CGPO 依赖大 teacher（122B-A10B）离线推理，训练成本较高；60M 样本规模对极端长尾场景覆盖仍有限；多语言变体仅评估 MDPBench，未报告 CC-OCR 等全量结果。
- **未来方向**（论文自述）：强化细粒度文本感知、定位、识别等基础 OCR 能力；向文本提取之外的语义理解与推理拓展；提升复杂真实环境的泛化与鲁棒性。

## 研究启发与可借鉴点
- **CGPO 能力路由范式可迁移**：将"教师质量 × 胜算概率"作为蒸馏权重，适用于任何 teacher-student RL+蒸馏联合训练场景（如文档解析、图表理解、代码生成），避免固定/cosine 调度导致的过/欠蒸馏。
- **六维能力划分 + 课程采样策略**：TP/DS/OR/GA 分组与三阶段配额（early 重感知，late 重推理）值得复用于其他多任务 VL 模型训练，保障长尾能力不被淹没。
- **任务对齐验证器设计**：按任务语义定制 Rτ（NED/IoU/CDM/TEDS/F1/LLM-judge）并映射到统一 GRPO 奖励空间，可推广至多模态 agent 的 outcome-supervised RL。
- **交叉模型一致性校验管线**：双模型解析 + 相似度假设 + 大模型纠错的流程，可作为高质量文档/表格标注的通用构造模板。
- **OCRBench v2.1 修订思路**：格式归一化（Type A）+ 错误参考纠正（Type B）+ 歧义指令重写（Type C）+ 任务对齐指标升级（CDM/md2md/NED/LLM-judge/multi-answer），可直接复用于其他基准的迭代评测体系建设。

## 关键术语表
- **PolyOCR**：Ant Group 提出的统一 OCR 基础模型家族（2B/9B），基于 Qwen3.5，支持识别/定位/解析/提取/理解/推理六维能力。
- **PolyOCR-2.7B**：多语言专用变体，ViT(1.2B)+Qwen2.5(1.5B)，覆盖 21 种语言/脚本组。
- **CGPO (Competence-Guided Policy Optimization)**：能力引导的策略优化，将 GRPO 与 on-policy 蒸馏按师生能力差距动态路由。
- **GRPO (Group Relative Policy Optimization)**：群体相对策略优化，以同组响应的相对奖励计算 token 级 advantage 进行策略梯度更新。
- **OPD (On-Policy Distillation)**：在线策略蒸馏，沿学生 rollout 轨迹对教师分布做 reverse KL 蒸馏。
- **CDM (Character Detection Matching)**：字符检测匹配，将公式渲染后在视觉 token 层做匹配并计算 F1。
- **TEDS (Tree Edit Distance Similarity)**：基于树编辑距离的表格结构相似度指标。
- **OCRBench v2.1**：OCRBench v2 的修订版，修正 24.5% 样本标注并升级多项子任务评分协议。

## 可复现要素
- **数据集**：约 60M 合成 + 人工校验样本；OCRBench v2.1 公开修订标注与评估代码（论文声明）；MDPBench、CC-OCR、OmniDocBench v1.6 为公开基准。*内部 KIE Benchmark 未公开。*
- **代码/权重**：GitHub https://github.com/inclusionAI/PolyOCR-Venus；任务接口/归一化/评估代码声明随模型发布（论文未给出具体 commit）。
- **关键超参**：SFT lr=1e-5 cosine、batch=256、seq_len=20480、BF16、vision encoder 冻结；CGPO lr=1e-6、K=G=8 rollout、clip ε=0.2、β=0、λ=1、temperature=0.9；teacher=Qwen3.5-122B-A10B 冻结。
- **算力**：SFT 16 nodes × 16 GPU（共 256 卡）；CGPO 配置未明述节点数。
