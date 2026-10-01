---
title: "ORAV-BENCHMARKING-AUDIO-VIDEO-GENERATION-FROM-MULTIMODAL-CON"
source: https://arxiv.org/pdf/2609.34843v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:51:55"
field: "多模态生成评测与可控合成"
keywords: ["audio-video generation", "multimodal benchmark", "reference-conditioned generation", "pairwise evaluation", "compositional controllability", "reference substitution"]
innovations: ["提出语义角色合约 c_j=(r,s,b,P,N) 将多参考组合可控性形式化", "参考感知配对评估协议在留外人评上达 86.08% 有效一致性", "诊断矩阵分离渲染质量/源亲和/任务达标并揭示参考替换为普遍失败"]
benchmarks: ["ORAV Bench (380 instances, 30 composition signatures, 9 semantic roles)", "VBench / T2AV-Compass / LongAV-Compass / MSAVBench / MultiRef-Compass (对比基线)"]
---

# 论文速读：ORAV-BENCHMARKING-AUDIO-VIDEO-GENERATION-FROM-MULTIMODAL-CON

## 一句话总结
本文提出 **ORAV Bench**，首个面向"全参考音视频生成"（Omni-Reference Audio-Video Generation）的系统性评测基准，通过语义角色合约（semantic reference contract）约束图文音多源参考的组合方式，并开发了一个参考感知的配对评估协议，在留出的 38 个测试实例上与人类专家判断达到 **86.08%** 的有效一致性。对 5 个前沿系统（Seedance-2.5 / Seedance-2.0 / MiniMax-H3 / Wan3.0-Video / Kling-v3-Omni）的评测揭示了"高源相似≠任务成功"，参考替换（reference substitution）是当前主流模型最常见的系统性失败。

## 研究问题与动机
- **全参考条件的组合可控性仍缺乏系统化度量**：业界前沿模型已能同时接受图像、视频、音频参考，但尚无基准能系统评估模型在跨模态、跨实体、跨时间维度下"选择性提取—绑定—组合"的能力。
- **既有基准侧重感知质量或单一参考保真**：T2AV-Compass、VABench、UniVBench、MotionBench、MSAVBench 等多聚焦感知质量、时序一致性或单类参考（纯图像/纯视频）保真，缺少对异质多参考联合组合（joint composition C+D+A）的结构化评测。
- **源相似与任务达标之间存在隐蔽 gap**：现有评测常以全局 CLIP/DINO 相似度作为代用指标，易误把"像源"当"按要求使用源"；本文指出模型倾向于整体复现参考源（如把运动视频的原演员、原场景直接搬入），而非按指令只抽取其中的动作/镜头/音色。
- **自动化评测缺乏对跨模态证据的显式结构化利用**：Arena-style 和简化 pairwise prompt 在捕捉细粒度差异（如前伸腿+低球位置的 dunk vs. 常规 dunk、录制回放 vs. 新生成发音）方面表现明显弱于结构化协议。

## 核心贡献（创新点）
- **提出 ORAV Bench**：构建 380 个任务实例、30 种组合签名、9 类语义角色、1322 个唯一参考素材的全参考音视频生成基准，覆盖图像（主体/道具/场景/风格）、视频（运动/镜头/特效）与音频（语音/音乐）三类参考的联合组合。
- **引入语义角色合约形式化**：为每条参考定义 `c_j = (r_j, s_j, b_j, P_j, N_j)`，显式刻画"该参考应贡献什么、绑定到哪个输出目标、允许哪些变换、必须排除哪些源内容"，把组合可控性转化为可判定的契约满足问题。
- **开发参考感知的配对评估协议**：以视觉时间戳面板 + 独立听辨证据为输入，按"可用性→任务满足→通用质量"三级优先级聚合 per-reference 匹配与跨参考绑定判断，并在 AB/BA 两种呈现顺序下校验顺序偏差。
- **构建并可复现的诊断指标体系**：Comp.z / TQ / AP（视觉质量）、Subject / Scene（DINO / CLIP 全帧亲和）、Voice（ECAPA cosine）、CER / WER（Whisper 文本错误）、PQ / NISQA / DNSMOS（音频）、Chroma / Waveform（音乐亲和），为"渲染质量—源亲和—任务达标"提供可复现的点wise 诊断接口。
- **揭示 reference substitution 为普遍失败模式**：在判定为不可用的输出解释中 95% 含参考替换线索；Wan3.0-Video 在有视频参考时不可用率达 55.9% 并检测到 9/78 语音录制回放；Kling-v3-Omni 因接口限制使用保留音轨适配器，语音波形相关性达 0.9998。

## 方法详解
**任务形式化**
- 给定文本指令 `T` 与参考集合 `R = R^I ∪ R^V ∪ R^A = {r_1,...,r_J}`，模型 `G_θ` 生成 `(Ŷ^v, Ŷ^a)`。输出须在感知上自然、时序上连贯，并按 `T` 与 `R` 指定的语义关系实现跨参考组合。
- 每条参考的语义合约：`s_j = ρ_T(r_j) ∈ S`（9 类角色之一）；绑定目标 `b_j`（主体/事件/音轨等）；应实现的属性集 `P_j`（含视角、时序等任务允许变化）；须排除的源内容集 `N_j`（可为空）。完整任务合约 `C(T,R) = {c_j}_{j=1}^J` 还规定指令级绑定与跨目标时序关系。

**基准构建流程**
- **角色与组合设计**：由社区用例与专家讨论归纳 9 类语义角色；按"组合签名"（角色集合，不计重数）组织实例，覆盖 `subject+motion`、`subject+prop+scene+motion`、`subject+speech`、`scene+camera` 等 30 种签名。
- **素材检索与绑定**：场景驱动检索 + 参考驱动设计双轮；视频参考人工审查，音频参考模型辅助 + 信号级检查；跨参考兼容性核验（场景容纳动作、道具支持交互、参与者和设施齐全）。
- **指令写法**：每条参考分配唯一句柄（`@image1/@video1/@audio1`），明确目标、角色分配、空间锚定（如"左边的人"锚到指定参考帧）、多主体绑定、无场景时的文本环境补全、音乐+运动耦合时的时序主导方与允许的时间调整。
- **质量控制与规模**：模型辅助 + 重复人工复核；每实例视频/音频参考总时长上限各 15 秒；参考素材不跨实例复用以降低相关性。最终 380 实例、30 组合签名、1322 唯一参考素材、每实例 2–10 条参考、2–4 类语义角色。

**参考感知配对评估**
- 每条参考的候选比较记录：`q_j = (e_j^A, e_j^B, d_j, o_j)`，其中 `e` 为观测对应、`d∈{A,B,tie}` 为相对偏好、`o` 为证据充分性。
- 证据准备：运动 fidelity 通过时间戳面板（8 fps、宽度 512、每片段至多 6 面板、≤2048px/边）暴露动作阶段与轨迹；独立听辨分离"所说内容""音色相似""是否回放源录音"。
- 三级优先级聚合：① **可用性**（源内容实质性替代/阻塞请求呈现即否决，边界由指令与合约而非通用相似度决定）；② **任务满足**（参考似主体是否在执行被演示动作、相机布置与时序是否符合）；③ **通用质量**（仅用于打破任务持平）。
- AB/BA 双向呈现校验：一致保留，顺序冲突计入 tie。
- 排序采用 Davidson–Bradley–Terry（DBT）模型：`D_ik = e^{λ_i} + e^{λ_k} + ν e^{(λ_i+λ_k)/2}`，联合最大化似然，实例聚类 bootstrap（2000 次）得 95% CI。

**人评验证**
- 5 名专家在开发集外的 10%（38 实例）盲评；ORAV 协议 vs.  vanilla Gemini-3.1-Pro-Preview 基线，有效一致性 **86.08% vs. 71.82%**；同一 Astra Judge 下结构化协议较简化 prompt 提升 **+7.45pp**。

## 实验与结果
- **评测对象**：Seedance-2.5、Seedance-2.0、MiniMax-H3、Wan3.0-Video、Kling-v3-Omni，共 380 实例；4 系统 720p，MiniMax-H3 用 768p；候选帧统一宽度呈现，优先任务满足以弱化清晰度差异。
- **整体排序（DBT strength）**：Seedance-2.5 (+0.70) > Seedance-2.0 (+0.48) > MiniMax-H3 (+0.15) > Wan3.0-Video (-0.56) > Kling-v3-Omni (-0.77)；预期胜率（vs. 随机对手）从 68.4% 到 29.9%，12 种聚合/数据处理设置下顺序不变。
- **共享交付实例 pairwise**：Seedance-2.5 对 Seedance-2.0 胜率 58%，对 Kling-v3-Omni 80%（tie 计半胜）。
- **角色条件优势分化**：MiniMax-H3 在 speech (+76.56%) 与 camera (+65.87%) 桶领先整体第三的排名；在 subject+speech 组合 77.8% 期望胜率高于 Seedance-2.5 的 72.7%；但在 subject+motion+speech 组合 73.4% vs. MiniMax-H3 53.6%，显示上下文依赖性互补优势。
- **诊断指标分化**：Seedance-2.0 视觉 Comp.z 最高 (+0.17)、Scene CLIP 最高 (79.01)；Seedance-2.5 ORAV 最高 (68.38)、Subject DINO 最高 (40.61)、CER/WER 最低 (5.51/5.58)；Wan3.0-Video Aesthetic Predictor 最高 (5.05) 但 WER 达 112.48% 且 9/78 语音被判定为源录音回放；Kling-v3-Omni Voice cosine=1.00、Waveform=0.9998（保留音轨适配器所致）。
- **质量与 ORAV 正相关**：Comp.z 与 ORAV 跨 5 系统 Spearman ρ=0.70，但领先组内排序有差异；Seedance-2.5 vs. Wan3.0-Video 的 Comp.z 差距 95% CI [0.205, 0.328]。
- **去音频子集排名**（259 实例，无语音/音乐参考）：Seedance-2.5 (+0.67) > Seedance-2.0 (+0.53) > MiniMax-H3 (-0.03) > Kling-v3-Omni (-0.53) > Wan3.0-Video (-0.64)，前三名顺序与全集一致。
- **不可用率**：Wan3.0-Video 总体 43.8%（有视频参考 55.9%、无 12.0%）；Kling-v3-Omni 32.0%；Seedance-2.5 仅 2.3%。

## 相关工作脉络
- **T2AV-Compass (2026)** / **LongAV-Compass (2026)**：侧重文本到音视频生成的感知与一致性评估，单参考或无参考设定，未引入异质多参考组合契约。
- **VABench (2026)** / **UniVBench (2026)** / **UI2V-Bench (2025)**：主要覆盖单图像参考的视频生成保真与理解，缺少视频/音频参考与跨模态组合评测。
- **MotionBench (2026)** / **OpenS2V-Eval (2025)**：聚焦单模态（纯视频动作或纯图像主体）单一因子评估，不涉及"只取动作不取演员"的选择性绑定。
- **MSAVBench (2026b)** / **MultiRef-Compass (2026)**：开始涉及多图+音频参考，但 MultiRef-Compass 的视觉参考主要用于主体/物体/场景外观指定，缺少运动、镜头、特效与语音的显式因子分解与组合签名覆盖。
- **GenAI Arena / GenArena**：Arena-style 盲比排名，未针对参考贡献的可解释性与证据结构化作设计，易受顺序偏差与自我增强影响。
- **定位差异**：ORAV 把"组合可控性"作为一等公民，以 9 角色 × 30 签名 × 合约形式化替代模糊的"多参考保真"，并以配对协议 + 诊断矩阵剥离"渲染质量—源亲和—任务达标"三者。

## 局限性与未来方向
- **音频-视频细粒度同步未纳入已验证范围**（Sec. A.7），评估主要依赖帧级面板与宏观听辨，未验证音画精确对齐。
- **诊断指标部分相关但并不等价于任务成败**：如 ArcFace 与 DINO 主体相关仅 0.256，CoTracker3 仅测运动方向未测速度与主体身份；各指标覆盖的共享实例数差异大（Subject 332、Scene 166、PQ 91、Voice 72、CER 52、WER 22）。
- **Kling-v3-Omni 的音频依赖保留音轨适配器**（A.8），其音频指标（Voice=1.00、Waveform=0.9998）反映接口兼容性而非模型原生音频生成能力，跨系统比较需小心。
- **指令-媒体一致性依赖人工终审**，380 实例规模对 30 种签名的覆盖仍有限，稀有组合（n<10）的统计不确定性较大。
- **未来方向**：指令条件化的参考表示与训练数据（同一动作适配不同演员/场景、同一音色支持新语句）、可复现的跨参考组合合成数据、细粒度音画同步度量、对 reference substitution 的系统性缓解。

## 研究启发与可借鉴点
- **合约化（contract-first）任务描述是可迁移范式**：把"每条参考应贡献什么、绑定到哪、排除什么"形式化为 `(source, role, target, P, N)`，可直接移植到其他多条件生成评测（如多风格图像编辑、多参考 3D 生成、多条件代码生成）。
- **"可用性→任务满足→通用质量"三级优先级聚合优于单一综合分**：避免因审美/技术分掩盖关键任务失败，适合任何需要兼顾"好看"和"做对"的生成评测。
- **AB/BA 双向呈现 + 顺序冲突记 tie**：低成本抑制顺序偏差，建议在 pairwise judge 中成为标配。
- **诊断矩阵与 ORAV 主指标并行报告**：Comp.z / TQ / AP / Subject / Scene / Voice / CER / WER / PQ 等多维点wise 指标，为"为什么强/弱"提供可解释归因，优于只报单一排名。
- **参考素材不跨实例复用**：降低跨实例相关性、防止重复源内容主导聚合结果，是基准构建的良好实践。

## 关键术语表
- **Omni-Reference Audio-Video Generation**：同时接受图像、视频、音频多种异质参考，并按文本指令将其选择性组合生成音视频的输出任务设定。
- **Semantic Reference Contract `c_j`**：为第 j 条参考定义的 `(r_j, s_j, b_j, P_j, N_j)` 五元组，显式规定其角色、绑定目标、应实现属性与须排除源内容。
- **Reference Substitution（参考替换）**：模型把未请求的源内容（原演员、原场景、源录音）直接搬入输出以替代指令要求的合成内容，是当前最普遍的系统性失败。
- **Reference-Aware Pairwise Evaluation**：以视觉时间戳面板 + 独立听辨为证据、按可用性/任务满足/通用质量三级优先级聚合的配对比较评估协议。
- **Composition Signature（组合签名）**：实例中参考角色集合（不计重数），如 `subject+motion+speech`，用于交叉比较不同系统在不同组合上下文中的相对优势。
- **Davidson–Bradley–Terry (DBT) 模型**：扩展 Bradley-Terry 以同时拟合胜/平/负概率的配对比较排序模型，ORAV 以此聚合 pairwise 结果得到系统 strength 与期望胜率。
- **Comp.z**：基于 DINO/CLIP/AMT-S/美学/成像质量等 VBench-style 维度标准化后的未加权均值，衡量输出内部一致性质量（与 ORAV ρ=0.70）。
- **Reference Affinity**：用预训练模型（DINO/CLIP/ECAPA/Whisper/CoTracker3）测量输出与指定参考在主体/场景/音色/语音/运动方向的相似度，区分"像源"与"按要求使用源"。

## 可复现要素
- **数据集**：380 任务实例、30 组合签名、1322 唯一参考素材；论文未给出公开下载链接，仅标注"Reproducible pointwise diagnostics"与 GHA 主页链接（arxiv.org/abs/2609.34843）。
- **代码/权重**：评测协议实现、提示与响应 schema 已在附录声明保留（"The implementation retains the complete prompts and response schemas"）；具体开源状态论文未明示。
- **关键超参**：帧采样 8 fps、宽度 512（等比）、每片段至多 6 面板、≤2048px/边；GPT-6-Astra judge 推理 medium、输出 8192 tokens；Gemini-3.1-Pro listener 推理 medium、temperature 0.3、输出 32768 tokens；DBT bootstrap 2000 次实例重采样；视频/音频参考时长上限各 15 秒。
- **预训练组件**：CLIP ViT-B/32、DINOv1 ViT-B/16、ECAPA-TDNN、Whisper-small、CoTracker3 scaled_offline、ArcFace、ALADIN、GroundingDINO、SAM2.1、DOVER++、Aesthetic Predictor V2.5、Audiobox、NISQA、DNSMOS、CLAP、MuQ——均为已发表开源模型（见 Appx. A.3–A.5 表 7–10）。
