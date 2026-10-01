---
title: "SPATIALCORE-CONFIDENCE-AWARE-GROUNDED-SPATIAL-REASONING-IN-L"
source: https://arxiv.org/pdf/2609.38716v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:36:40"
field: "多模态大模型空间推理"
keywords: ["spatial reasoning", "vision-language models", "confidence-aware reward", "GRPO", "grounded reasoning", "bounding box uncertainty"]
innovations: ["提出置信度感知的自调节空间奖励，用 BBox 坐标 token 不确定性加权定位质量作为学习信号", "设计答案门机制将空间奖励与最终答案正确性耦合，保留错误轨迹的部分接地信用", "理论证明答案奖励下 GRPO 对接地置信度存在梯度 indifferent，置信度加权打破等价性"]
benchmarks: ["OmniSpatial", "SpatiaLab"]
---

# 论文速读：SPATIALCORE: CONFIDENCE-AWARE GROUNDED SPATIAL REASONING IN LARGE VISION–LANGUAGE MODELS

## 一句话总结
本文提出 SpatialCORE，一种后训练框架，通过将模型对自身生成接地（grounding）的置信度显式转化为学习信号，从而提升大视觉-语言模型（LVLM）的空间推理能力；其核心是自调节空间奖励，利用边界框坐标 token 不确定性对定位质量进行加权，在 OmniSpatial 和 SpatiaLab 上达到开源及专用空间推理模型的最优水平。

## 研究问题与动机
- LVLM 在空间推理任务上存在系统性缺陷：即便能识别物体，仍难以理解空间布局和关系，限制了其在机器人、自动驾驶等场景的应用。
- 现有后训练方法（如 RL-based spatial reasoning）主要奖励最终答案正确性，导致即使推理轨迹中的边界框（BBoxes）定位模糊、不确定性高，只要答案正确即被强化，无法区分"正确但置信度低"与"正确且置信度高"的推理路径。
- 已有生成的接地方法中，定位质量与模型自身置信度未被显式建模；坐标 token 的不确定性可作为置信度估计，但尚未被用作空间推理的训练信号。
- 外部引导方法依赖推理时的锚点输入，无法训练模型自主推理；几何增强方法改变了模型"看什么"，但未解决"如何在推理中可信地使用接地"的问题。

## 核心贡献（创新点）
1. **提出 SpatialCORE 后训练框架**，将模型对生成接地的置信度作为显式学习信号；与已有工作本质区别在于：不仅要求接地准确，还要求模型对自身接地"有信心"，这是前作未涉及的训练维度。
2. **设计自调节空间奖励（self-regulating spatial reward）**，用 BBox 坐标 token 的 Shannon 熵估计不确定性，以置信度加权几何匹配质量，并通过伪 GT 有效性（pseudo-GT validity）进行缩放；已有方法仅使用 IoU 等几何指标，无置信度权重。
3. **引入答案门（answer gate）** 将空间奖励与最终答案正确性耦合：答案正确时全量奖励，答案错误时按因子 γ 衰减保留部分空间信用；已有方法对错误答案完全丢弃空间信号。
4. **在 OmniSpatial 和 SpatiaLab 上取得 SOTA**，开源模型中最高加权准确率 48.46%（SpatialCORE-8B，OmniSpatial）和 47.91%（零样本 SpatiaLab），超越专业空间推理模型 7–12 个百分点，且具备零样本迁移能力。
5. **提供理论论证**（附录 B）：证明在仅用答案奖励的 GRPO 下，相同答案不同接地置信度的轨迹获得相同优势信号（梯度 indifference），置信度加权打破了这一等价性。

## 方法详解
- **基础架构**：基于 Group Relative Policy Optimization（GRPO），在 Qwen3-VL-Thinking（8B/4B）上使用 LoRA 后训练。
- **伪 GT 构建**：离线用 Grounding DINO 对问题/选项中提及的任务相关物体进行 zero-shot 检测，提取 (b_k^gt, ℓ_k^gt, v_k) 三元组，v_k 为检测分数，作为空间奖励的参考。
- **BBox 匹配奖励（式 3）**：
  R_BBox^(j,k) = (w_iou · max(0, IoU(b_j, b_k^gt) − τ_iou) + w_label · Sim(ℓ_j, ℓ_k^gt)) · v_k，其中 Sim 为 bag-of-words 余弦相似度，匈牙利算法求解最优一对一匹配 M_i^*。
- **坐标 token 不确定性（式 5）**：对每个坐标的 digit token 计算 Shannon 熵，归一化后平均得到 H_{i,j} ∈ [0,1]，越低表示越自信。
- **置信度权重（式 6）**：C_{i,j} = 1 − H_{i,j}，ω_{i,j} = β + (1−β)C_{i,j}，β 为置信度下限（默认 0.1）。
- **空间奖励（式 7–8）**：P_i 为置信度加权的匹配质量均值，Recall_i 衡量伪 GT 覆盖度，二者通过 α-weighted F₁（α=2.0）结合得 R_spatial^(i)。
- **格式奖励与答案奖励**：格式奖励约束输出结构（有效 reasoning segment、单一最终答案 token、有效 BBox 格式）；答案奖励 R_ans ∈ {0, 1}。
- **自适应奖励组合（式 10）**：r_i = R_ans^(i) + λ_fmt R_fmt^(i) + λ_s · g(R_ans^(i)) · R_spatial^(i)，其中 g(1)=1，g(0)=γ（默认 0.3）。
- **SFT 冷启动**：在 1,000 个 OmniSpatial 样本上进行监督初始化，使模型学会接地输出格式，否则 GRPO 阶段接地 reward 收敛至 0，模型停止生成 BBox。

## 实验与结果
- **数据集**：OmniSpatial（训练 + 测试，4 维 50 子类，共 6,902 训练样本/1,533 测试样本）；SpatiaLab（零样本，6 类 30 子任务类型，1,400 样本）。
- **最强结果**：
  - OmniSpatial：SpatialCORE-8B 加权平均准确率 **48.46%**，超越 VST-RL-7B（+7.37%）、SpaceThinker-Qwen2.5VL-3B（+8.04%）、SoFar（+3.32%）、InternVL3-14B（+2.52%）、Gemma-3-12B（+4.75%）；较 backbone 原始提升 4.56%（较 GRPO baseline 提升 2.54%）。
  - SpatiaLab（零样本）：SpatialCORE-8B **47.91%**，超越 SpatialLadder-3B（+12.63%）、SpaceThinker（+7.27%）、SpaceOm（+6.55%）、InternVL3.5-4B 和 Qwen2.5-VL-7B-Instruct；较 backbone 提升 3.2%（较 GRPO baseline 提升 1.7%）。
  - Traffic Analysis 和 Localization 子任务提升最大；Allocentric/Hypothetical 推理提升较小（受限于单视角）。
- **消融（OmniSpatial Spatial Interaction 子集）**：
  - Full：56.33%；去掉置信度加权 → 52.33%（−4.0 点，最大单项下降）；去掉 vision LoRA → 53.00%（−3.33 点）；去掉 pseudo-GT validity → 53.33%（−3.00 点）；去掉 answer gate → 54.67%（−1.66 点）。
- **置信度-定位对齐分析**：SpatialCORE 后，最低不确定性四分位 mean IoU 达 77%，最高仅 36%；熵-IoU 相关系数从 ρ=+0.38（基线，负相关）翻转为 ρ=−0.42（正相关，低熵=高质量）。
- **伪 GT 鲁棒性**：40% BBox 被人工扰动后仍达 46.70%（较 backbone 高 2.80 点），证明对参考噪声有较强韧性。

## 相关工作脉络
1. **Thinking with Images / 推理中生成接地（Wu et al. 2025b; Zheng et al. 2025; Batra et al. 2025）**：将 BBox/mask 作为推理 trace 的一部分，但奖励主要依赖答案正确性或定位 IoU，不建模置信度——SpatialCORE 在此基础上引入置信度信号。
2. **空间奖励 RL 方法（Li et al. 2025; Ma et al. 2026; Sarch et al. 2025）**：用 RL 强化空间定位，但同样只关注几何正确性，未区分自信/不自信的地面；本文填补此空白。
3. **外部引导方法（Cai et al. 2025a; Gholami et al. 2025; Pothiraj et al. 2025）**：依赖 prompt 注入的空间锚点，无法训练模型自主推理；SpatialCORE 完全自主生成接地。
4. **几何增强 VLM（Chen et al. 2024b; Hu et al. 2025; Liu et al. 2025b）**：通过深度/点云/多视角改进视觉表征，改变输入而非训练信号；SpatialCORE 不改 backbone 输入，只改训练目标。
5. **推理时 scaffold（Liao et al. 2024; Yan et al. 2026; Chen et al. 2025b）**：prompt/解码干预，无需训练，但泛化依赖特定 task format；SpatialCORE 通过训练内化能力。
6. **SpatialLadder（Li et al. 2025）、VST-RL（Yang et al. 2025b）、SoFar（Qi et al. 2025b）**：专用空间推理模型，SpatialCORE 在多个子类别上显著超越，证明置信度加权信号更优。

## 局限性与未来方向
- **BBox 表示局限**：仅支持可框选的物体级对象，无法处理路径、序列、不可框选的空间关系（如 Complex Logic、Dynamic Reasoning 子任务表现较弱）；也缺乏 3D/深度信息。
- **伪 GT 依赖**：空间奖励依赖 Grounding DINO 离线生成的伪 GT，虽经人工审计（95.8% 正确率）且对 40% 扰动具有鲁棒性，但在 REC 模型失效或场景复杂时监督信号退化。
- **单视角限制**：Egocentric 单图输入难以支撑 allocentric/hypothetical 多视角推理；作者建议未来加入深度/多视角上下文或几何增强编码器。
- **训练数据规模有限**：仅用 6,902 个 OmniSpatial 样本后训练，更大规模数据的泛化效果未知。
- **未来方向**：（1）引入深度图/多视角/3D 几何编码；（2）扩展到非 BBox 的空间表达（segmentation mask、path grounding）；（3）探索端到端伪 GT 学习与置信度联合优化。

## 研究启发与可借鉴点
1. **置信度作为训练信号**：用模型自身 token 分布熵（而非外部标注）估计 grounding 质量，可迁移至任何需要"生成+使用中间结构化输出"的 VL/LLM 任务（如数学推理中的 step grounding、代码生成中的变量绑定）。
2. **自调节奖励设计（answer gate）**：将主奖励与辅助信号通过门控耦合（γ∈(0,1)），使错误答案仍保留部分辅助信用，避免过早丧失有益中间表示——可复用于其他 RLHF/GRPO 训练中的多目标平衡。
3. **SFT 冷启动保证格式收敛**：在 GRPO 前用千级样本做格式初始化，防止 grounding 结构在 RL 阶段 collapse，是 grounded reasoning 训练的实用 trick。
4. **伪 GT 有效性加权**：用检测分数 v_k 缩放 reward 贡献，使低质量参考减少干扰；这一思想可推广到任何用弱监督/自动生成参考的训练场景。
5. **与团队方向结合机会**：若团队关注多模态 Agent 或机器人导航，可将 SpatialCORE 的置信度感知奖励与 tool-use grounding（如操作指令中的目标框定）结合，形成"敢信才敢用"的 grounded decision-making 框架。

## 关键术语表
- **SpatialCORE**：作者提出的后训练框架，全称 Spatially COnfident REasoning，通过置信度感知接地增强 LVLM 空间推理。
- **Self-regulating spatial reward**：自调节空间奖励，用 BBox 坐标 token 不确定性估计置信度并加权几何匹配质量，动态调节奖励强度。
- **Answer gate**：答案门，函数 g(R_ans)，答案正确时空间奖励完整计入，错误时按因子 γ 衰减保留部分信用。
- **Pseudo-GT BBox**：伪真值边界框，由 Grounding DINO 离线生成、经有效性分数 v_k 标注的参考框，作为空间奖励的锚定目标。
- **GRPO（Group Relative Policy Optimization）**：组相对策略优化，以同组轨迹的奖励标准差归一化优势 A_i，替代 PPO 的 value network 估计。
- **Coordinate-token uncertainty**：坐标 token 不确定性，对 BBox 各坐标 digit token 的 Shannon 熵求均值归一化得到，越低表示模型越自信。
- **OmniSpatial**：涵盖 4 维度 50 子类的空间推理评测基准，共 8,400+ QA 对，本文训练/测试均基于此。
- **SpatiaLab**：自然场景下的空间推理零样本评测基准，6 类 30 子任务类型，1,400 QA 对，用于评估跨分布泛化。

## 可复现要素
- **数据集**：OmniSpatial（训练/测试 split 公开）、SpatiaLab（零样本评测公开）；训练使用 OmniSpatial 官方训练集（6,902 样本）。
- **代码**：开源，GitHub https://github.com/rafiibnsultan/SpatialCORE。
- **权重**：未提及公开权重，仅提供代码与训练配置。
- **关键超参**：LoRA r=32/α=64（LM）和 r=4/α=8（vision）；G=4 rollout；lr=5×10⁻⁵ cosine；λ_fmt=0.2, λ_s=1.0, γ=0.3, β=0.1, α=2.0, w_iou=0.8, w_label=0.2；KL 系数 η=0.01；BF16；2×H100；3 epochs；effective batch=32；max tokens=3,072。
- **伪 GT 构建**：Grounding DINO，阈值 0.3，GPT-4o-mini 提取短语查询，坐标归一化至 [0,1000]。
- **SFT 冷启动**：1,000 样本，completion-only CE，冻结视觉编码器，仅更新语言侧 LoRA。
