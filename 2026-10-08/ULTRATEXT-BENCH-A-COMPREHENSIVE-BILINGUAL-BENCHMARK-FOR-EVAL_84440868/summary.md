---
title: "ULTRATEXT-BENCH-A-COMPREHENSIVE-BILINGUAL-BENCHMARK-FOR-EVAL"
source: https://arxiv.org/pdf/2610.09823v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 09:41:24"
field: "多模态生成评估"
keywords: ["视觉文本渲染", "图文生成评测基准", "双语评测", "VLM评委", "密集文本生成", "Qwen-Image-Bench"]
innovations: ["提出 UltraText Bench：432双语提示词、24场景类别、3难度等级的密集视觉文本渲染统一评测基准", "构建四维度评分体系（忠实度60%/清晰度30%/空间5%/场景5%）并将文本正确性与视觉清晰度解耦", "系统性评测24个模型配置的跨维度/跨负载/跨语言表现并揭示Turbo加速的隐性忠实度代价"]
benchmarks: ["UltraText Bench", "Qwen-Image-Bench (Q-Judger)", "CVTG-2K", "LongText-Bench", "InfoTextBench", "OCRGenBench"]
---

# 论文速读：ULTRATEXT BENCH: A COMPREHENSIVE BILINGUAL BENCHMARK FOR EVALUATING VISUAL TEXT RENDERING IN IMAGE GENERATION

## 一句话总结
本文提出了 UltraText Bench，一个涵盖 432 个双语提示词（英/中各半）、24 个真实场景类别、3 个难度等级的图文密集文本渲染评测基准；使用 Q-Judger VLM 评委从文本忠实度、清晰度、空间质量和场景质量四个维度综合评估 24 个模型配置。

## 研究问题与动机
- **短字符串渲染已趋成熟，但密集多区域长文本渲染能力未被充分评测**：现有工作（如 Wu et al., 2025; OpenAI, 2025）在短字符串渲染上进步明显，但面对海报、产品包装、文档等需要数百至数千字符跨多个区域正确排布的场景时，模型能力如何尚不明确。
- **现有基准无法统一评测"文本内容正确性 + 空间布局 + 场景融合"三重需求**：CVTG-2K（Du et al., 2025）侧重多实例文字匹配；LongText-Bench（Geng et al., 2025）侧重长文本准确性；OCRGenBench（Zhang et al., 2025a）覆盖 OCR 生成任务但非纯 prompt-only；InfoTextBench（Xiang et al., 2026）虽近但 target-string 定义与 region 体系不可比，缺乏统一的 per-region 参考标注和共享评分准则。
- **纯 prompt-only 设定要求模型从自然语言中自行推断所有目标字符串的渲染位置**：现有方法多依赖显式字形控制（GlyphControl、AnyText）或布局掩码预测（TextDiffuser），而本文聚焦"只给提示词、不给额外输入"的最自然测试设定。
- **双语评测对中英文不同字形体系的公平比较至关重要**：英文 GT 字符均值从 L1(489.9) 到 L3(2386.7) 增长约 5x，中文仅从 L1(394.6) 到 L3(743.6) 约 2x，反映出两种语言在同等难度下实际负载的不对称性。

## 核心贡献（创新点）
1. **UltraText Bench 数据集**：432 个双语提示词，覆盖 24 个真实场景类别和 3 个难度等级，每个提示包含 4–12 个带精确目标字符串的文本区域，所有 407,918 个 GT 字符由人工审核确保自然融入场景描述。*与已有工作的本质区别在于提供了完全统一格式的 per-region 结构化参考和共享评分准则，而非仅列出目标字符串。*

2. **四维度整体场景评测框架**：通过 Q-Judger 应用六原始分（TA/TC/TR/PC/LQ/SI）→ 四报告维度（Fidelity/Clarity/Spatial/Scene）→ 复合分（权重 60/30/5/5）的映射，可同时区分"字形清晰但拼写错误"与"内容缺失但可读"等不同失效模式。*与已有工作的本质区别在于将文本忠实度与视觉清晰度解耦，并单独保留空间与场景维度的评分。*

3. **系统性 24 模型配置的跨维度实验分析**：揭示 Turbo 变体（如 Z-Image-Turbo）可在清晰度 +3.81 的同时使忠实度下降 14.76；Qwen-Image-2512 英文复合分从 L1(86.50) 骤降至 L3(42.86)，暴露高工作负载下的性能断裂；GPT Image 2 [Low] 以 99.35 领跑且 83.6% 图像六维全满分。*与已有工作的本质区别在于在统一基准上横向比较开权模型与 API 模型的各维度表现差异。*

4. **10 人参与的人类评估定性分析**：通过七类典型案例（含低分/满分争议图）展示自动评分与人类视觉检查的局部分歧，阐明"最高 100 分≠ 百分比测量值"的评分类语义。*与已有工作的本质区别在于系统性地保留了与评委分歧的可检查案例以供后续分析。*

## 方法详解

### 3.1 任务形式化
每个评测实例由提示词 $p$ 和结构化参考 $R = \{r_1, \ldots, r_K\}$ 组成，其中 $K \in [4, 12]$ 为文本区域数。生成器 $M$ 仅接收 $p$（无额外字形/布局输入），输出图像 $I = M(p)$。每个区域 $r_i$ 记录六项属性：目标字符串 $t_i$、3×3 网格位置、相对大小（large/medium/small/tiny）、文本类型、载体（carrier，如 wooden sign/chalkboard/brass plate）和重要性（high/medium/low）。

### 4.1 四维度评分体系
Q-Judger 对图像和完整参考 $R$ 返回六个 0–100 的原始分：

$$
\mathbf{s}_{\text{raw}} = J(I, R) \in [0, 100]^6
$$

其中 TA=文本准确性、TC=文本完整性、TR=文本可读性、PC=位置正确性、LQ=布局质量、SI=场景融合。映射为四个报告维度：

| 维度 | 公式 | 权重 |
|------|------|------|
| 文本忠实度 (Fidelity) | $(TA + TC)/2$ | 60% |
| 文本清晰度 (Clarity) | $TR$ | 30% |
| 空间质量 (Spatial) | $(PC + LQ)/2$ | 5% |
| 场景质量 (Scene) | $SI$ | 5% |

复合分计算：

$$
\text{Composite} = 0.60 \cdot s_{\text{fidelity}} + 0.30 \cdot s_{\text{clarity}} + 0.05 \cdot s_{\text{spatial}} + 0.05 \cdot s_{\text{scene}}
$$

### 3.4 难度分层设计
- **L1（Hard）**：EN 设计带 250–600 字符，ZH 350–600 字符；EN 均值 489.9/72 提示，ZH 均值 394.6/72 提示
- **L2（Very Hard）**：EN 600–1200，ZH 550–900；EN 均值 1023.4，ZH 均值 627.3
- **L3（Extreme）**：EN 1200+（无上限，实测均值 2386.7，最大 5230），ZH 600+（实测均值 743.6，最大 1079）

### 3.6 提示构造与 GT 标注
目标字符串以引用文本、代码围栏或结构化内容块嵌入场景描述中（如："a hand-carved wooden shop sign features cream-white serif lettering that reads: 'DAILY BREAD — Artisan Bakery & Fine Patisserie Since 1987'"）。GT 字符串保持原文不变，场景描述经过人工打磨以实现自然融入。

### 评测执行流程
1. 每提示生成 4 张图像 → 2. PNG 结构验证 → 3. Q-Judger 严格评审（要求恰好 6 个整型键的 JSON 响应）→ 4. 报告聚合（图像级均值 + 提示级宏均值 + 英/中均衡）

## 实验与结果

### 数据集规模
- 432 个提示词（216 EN + 216 ZH），24 类别 × 3 难度 × 2 语言 × 3 重复
- 总计 407,918 个 GT 字符，2,926 个标注区域
- 完整运行：每模型配置 1,728 张图像（864 EN + 864 ZH）

### 主要结果（Table 5 关键数字）
- **GPT Image 2 [Low]**：Composite **99.35**（EN/ZH 均领先），1,440/1,723 有效图像（83.6%）获六维全 100 分
- **Nano Banana 2**：97.00；**Seedream 5.0 Pro**：91.45；**Qwen-Image-2.0-Pro**：90.21；**Qwen-Image-3.0**：88.33
- **开权模型最佳**：Boogu-Image-0.1-Base **77.89**，超越 Qwen-Image-2512 的 67.96
- **Turbo 效应**：Z-Image-Turbo 清晰度 +3.81 但忠实度 −14.76（Base 61.92 vs Turbo 54.38）；LLaDA-Image-Turbo 忠实度 −7.62，Boogu-Image-0.1-Turbo 忠实度 −9.50
- **工作负载衰减**：Qwen-Image-2512 英文复合分 L1(86.50) → L3(42.86) 下降 43.64 点；Z-Image-Base 英文 L1(82.92) → L3(26.34) 下降 56.58 点
- **语言差异**：Boogu-Image-0.1-Base 在 EN L3 仅 57.67 而 ZH L3 达 83.23；GPT Image 1.5 [High] 反之 EN 77.21 而 ZH 18.22
- **最低表现**：SD3.5 Large Composite 7.48；FLUX.2 [Klein] 4B 仅 19.58

### 覆盖情况
- EN 图像覆盖 859/864，提示覆盖 215/216（4 例安全拒绝）
- ZH 图像覆盖 864/864，提示覆盖 216/216

## 相关工作脉络

1. **MARIO-Eval / TypeScore**：早期短字符串基准，TypeScore 首次将拼写错误/缺失/多余分离测量，但未考虑空间布局。*UltraText 扩展至多区域 + 空间参考 + 四维度评分。*

2. **AnyText-Bench / CVTG-2K**：引入字形/位置条件输入或 CLIPScore 辅助测量，但目标参考与位置描述粒度不及 UltraText 的 per-region 结构化标注。*UltraText 采用纯 prompt-only 输入，不依赖额外控制。*

3. **LongText-Bench（Geng et al., 2025）**：VLM 驱动的长短文本准确性评测，覆盖八类场景，但缺乏 per-region 的载体/位置参考和空间质量维度。*UltraText 提供更细粒度的六属性区域标注。*

4. **InfoTextBench（Xiang et al., 2026）**：最接近的双语密集文本基准，采用目标字符串列表 + OCR/VLM 评分；但 target-string 集合定义与 UltraText 的 per-region 体系不可直接对比。*UltraText 的 region 绑定 carrier/position 使空间评估成为可能。*

5. **OCRGenBench（Zhang et al., 2025a）**：覆盖 OCR 生成/编辑/变换任务的广泛基准，包含高密度页面生成，但非纯 prompt-only 文本渲染场景。*UltraText 聚焦纯文本生成 + 密集多区域排布。*

6. **T2LSC-Bench（Wang et al., 2026b）**：结合 OCR 验证与结构化 VLM 评审，区分文字是否落在预期锚点；但缺少 UltraText 的完整 per-region 载体参考和多场景类别覆盖。*UltraText 强调"文本是否与其载体自然融合"这一独特维度。*

## 局限性与未来方向

- **模型配置的标准采样参数各不相同**（分辨率、宽高比、API 行为），导致横向比较受设置差异影响；控制变量的对比研究留待未来。
- **每个 category×level×language 单元仅 3 个提示、12 张图像**，统计精度有限，不适合做细粒度单元格差异推断。
- **难度级别同时改变字符量和区域数**，无法隔离二者各自独立的影响；EN 与 ZH 使用不同字符量设计带，语言与负载效应的分离需额外控制。
- **当前协议仅返回图像级综合分**，不产出每区域的独立评分或转录文本；区域级归因是明确的未来方向。
- **10 人人类评估仅提供定性例证**，未报告量化评分者间一致性或人与评委的相关系数。
- **VLM 评委存在极端情况的系统性偏差**：如 E4_L3_EN_003 在底部右侧测试块存在明显变形小字时仍获六维全 100 分；F2_L3_ZH_002 可见相关聊天文本但 TA=0/TC=0 的争议。
- **最高分 100 表示 rubric 上限而非"正确字符百分比"**，在高分子集内 Pearson/Spearman 相关系数无定义。
- **未来方向**：区域级评估与训练信号探索；扩展至视频生成评测（VTR-Bench 已有视频先例）；用 importance 加权训练反馈（如 DDPO/ReFL/Diffusion-DPO）。

## 研究启发与可借鉴点

1. **四维度解耦设计值得迁移**：将文本忠实度（TA+TC）与清晰度（TR）分开、再单独保留空间（PC+LQ）和场景融合（SI），使不同失效模式可诊断——该思路可直接应用于其他视觉生成任务（如文本编辑、多区域排版）的评测体系设计。

2. **结构化 per-region GT 标注范式具有通用价值**：每个区域携带 target_text/position/grid/size/type/carrier/importance 六属性，这种结构化参考不仅服务于评测，还可作为训练时的区域级反馈信号，支持 reward-based 优化（DDPO/ReFL）和 preference pair 构建。

3. **Turbo/Base 对照实验揭示加速方法的隐性代价**：Z-Image-Turbo 清晰度 +3.81 但忠实度 −14.76 的发现提示，在评估一步/少步生成模型时，必须同时报告忠实度与清晰度，单一"视觉质量好"的指标会掩盖文本错误。

4. **中英双语独立撰写而非翻译的策略**保证了文化真实性（如 C1_L1_EN_001 与 C1_L1_ZH_001 的菜单内容完全不同），避免了翻译引入的系统偏差，这一策略对任何双语评测基准的建设均有参考价值。

5. **与团队方向的结合机会**：团队可借鉴 UltraText 的四维度 VLM judge 框架，构建中文特有的密集场景文本评测子集（如中文票据、中文表单、中文代码界面）；也可利用其 region-level GT 格式作为训练数据，探索 region-aware reward modeling。

## 关键术语表

- **UltraText Bench**：纯 prompt-only 密集视觉文本渲染双语评测基准，432 提示词覆盖 24 场景类别和 3 难度等级。
- **Q-Judger**：来自 Qwen-Image-Bench 的专用 VLM 评委模型，按 UltraText rubric 对图像和完整参考输出六维 0–100 评分。
- **文本忠实度 (Fidelity)**：文本准确性（TA）与完整性（TC）的均值，权重 60%，反映目标字符串是否正确完整再现。
- **文本清晰度 (Clarity)**：等于 TR（可读性）原始分，权重 30%，衡量字体锐度、对比度和可读性，与内容正确性解耦。
- **Per-region 结构化参考**：每个文本区域附带 target_text/position/grid/size/type/carrier/importance 六属性标注，作为 VLM 评委的完整评估依据。
- **复合分 (Composite)**：Fidelity×60% + Clarity×30% + Spatial×5% + Scene×5% 的加权和，用于跨模型总排名。
- **Prompt-macro 聚合**：先对每提示的有效图像求均值，再对所有提示取平均，使每个提示获得等权重，避免多样本提示主导结果。
- **Turbo/Base 对照**：同一模型族的标准版与加速版（通常更少推理步数）的比较，揭示加速方法在忠实度/清晰度间的权衡。

## 可复现要素

- **数据集**：432 个 JSONL 提示词文件（EN/ZH 分开放置），含结构化 GT 标注；[GitHub: https://github.com/LINs-lab/UltraText_Bench](https://github.com/LINs-lab/UltraText_Bench) 已公开
- **评测代码**：`src/generation/` 和 `scripts/eval_vlm_judge_api.py` 随仓库发布；支持 OpenAI-compatible vLLM 服务调用 Q-Judger
- **Q-Judger 模型**：基于 Qwen-Image-Bench，temperature=0.0，关闭思维链，默认 512 token 输出上限
- **生成设置**：每提示 4 张图像，使用模型官方默认采样配置；分辨率/宽高比/API 行为因模型而异
- **人类评估**：10 名参与者参与定性评分验证，量化一致性统计论文未报告
- **GPT Image 2 [Low] 的完整图像和评委记录已单独发布**（859 EN + 864 ZH 有效图像），其余模型配置仅发布排行榜分数
- **关键超参**：复合分权重 (0.60, 0.30, 0.05, 0.05)；EN/L1 字符带 250–600，ZH/L1 350–600；每单元 3 提示 × 4 样本
