---
title: "VIDEO-PREDICTION-POLICY-2-PREDICT-BETTER-ACT-BETTER"
source: https://arxiv.org/pdf/2610.10270v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 09:42:01"
field: "具身智能/机器人操作"
keywords: ["World Action Model", "Video Prediction", "Robotic Manipulation", "Zero-shot Generalization", "Consistency Distillation", "Mixture-of-Transformers"]
innovations: ["事件级视频继续预训练结合详细字幕以提升指令遵循泛化", "固定时域后训练与单步一致性蒸馏实现实时视觉规划", "MoT动作专家分阶段训练避免破坏视频骨干泛化能力"]
benchmarks: ["ALOHA Zero-shot Manipulation", "LIBERO-ID/Pro/OOD", "RoboDojo"]
---

# 论文速读：VIDEO-PREDICTION-POLICY-2-PREDICT-BETTER-ACT-BETTER

## 一句话总结
论文提出了**Video Prediction Policy 2 (VPP2)**，一种世界动作模型（WAM），通过大规模多样化操作视频数据的事件级继续预训练、固定时域视频预测后训练以及单步一致性蒸馏，并结合基于MoT的动作专家模块，实现了在开放环境操作任务中视频预测与动作生成两方面均具备强零样本泛化能力的通用机器人策略。

## 研究问题与动机
1. **核心问题**：现有世界动作模型（WAMs）在开放环境（unseen, out-of-distribution scenarios）中经常产生错误的运动预测，进而导致错误的动作输出，泛化能力严重受限。
2. **原因一（模型层面）**：基础视频生成模型主要为创意内容生成和美学质量优化，而非精确物理动态，因此难以忠实遵循操作指令（Chen et al., 2025; Zhang et al., 2026b）。
3. **原因二（训练层面）**：将动作特定组件或训练目标直接融入预训练视频基础模型会显著损害其泛化能力（Mishra et al., 2026）。
4. **数据层面**：缺乏大规模、多样化且附带详细事件级标注的操作视频数据集，导致模型难以学习语义描述到视觉轨迹的一致映射。

## 核心贡献（创新点）
1. **事件级视频继续预训练（Event-level Continued Pre-training）**：使用大规模多样化操作视频及详细事件级字幕对基础视频模型（Wan2.1-I2V-14B）进行继续预训练，使模型能够忠实遵循操作指令并生成未来预测。*本质区别*：不同于直接微调或仅用简单指令，本文强调“详细字幕+完整事件级轨迹预测”以建立语义到轨迹的稳定映射，而非单纯依赖视频生成美学。
2. **固定时域后训练与单步一致性蒸馏（Fixed-horizon Post-training & Consistency Distillation）**：将事件级预训练模型后训练为预测固定时长（人类手8秒/操作2秒）的连续视频片段，并通过一致性蒸馏（Consistency Distillation）将其压缩为单步生成的视觉规划器，满足实时执行需求。*本质区别*：现有工作多使用迭代多步视频生成，导致闭环控制延迟；本文在保持泛化能力的同时实现单步快速推理。
3. **基于MoT的动作专家模块（Action Expert via Mixture-of-Transformers）**：引入一个0.9B参数的扩散Transformer（DiT）动作专家，基于冻结的视频骨干（仅LoRA微调）从KV缓存中学习隐式逆动力学模型。*本质区别*：与联合训练视频和动作的方法不同，本文分阶段训练（先保视频泛化，再加动作专家），避免动作训练梯度破坏视频表征。
4. **统一多视图输入与动作空间对齐（Unified T-shape Input & Action Space Alignment）**：提出跨不同机器人数据集的统一T形多视图输入格式，并对 egocentric bimanual 数据集进行工作空间对齐和末端执行器坐标系对齐，解决坐标系不一致导致的动作语义混淆问题。*本质区别*：现有工作通常直接使用原始数据集，本文通过显式坐标变换实现跨数据集动作表示的一致性。
5. **分层VLM规划架构（Hierarchical VLM Planning）**：结合一个VLM高层规划器，将开放式指令分解为显式子任务指令，使VPP2专注于从明确指令到轨迹的映射。*本质区别*：将语义理解和记忆留给VLM，本体专注物理预测与动作生成，解耦了高级推理与低级控制。

## 方法详解
- **数据收集与处理**：
  - 数据来源：机器人操作数据集（单臂：OXE, RoboMIND, DROID, RH20T, Molmoact；双臂：AgibotWorld-Beta, RoboCOIN, RDT, GM100, ABC-130k, Self-collected；移动/人形：Aloha InternData-A1, Galaxea Open-World, 灵巧手）、人类活动数据集（EgoDex, Ego4D, Epic-Kitchen, 自采集第一人称数据）、通用视频数据集（Open-vid, Panda-70M）。
  - 视频过滤、分割与字幕生成：去除损坏轨迹，按速度极小值或夹爪开合等启发式 cues 分割为1-12秒语义片段；使用VLM生成详细字幕，包含任务描述、主动末端执行器、目标对象、摄像头视角。
  - 统一T形多视图输入：主视图左，辅助视图堆叠右侧，缺失视图用黑色占位符填充。
  - 统一动作空间：对工作空间（世界帧变换）和末端执行器（局部变换）进行对齐，变换公式：$\widetilde{T}_d = A_d T_d B_d$，其中 $A_d$ 为工作空间对齐，$B_d$ 为末端执行器对齐。
- **Stage 1：事件级视频模型继续预训练**：
  - 给定演示 $\tau = (o_0, o_1, \dots, o_T)$，均匀采样N个未来帧：$\mathbf{y}_{\text{event}} = (o_0, o_{\lfloor T/N \rfloor}, \dots, o_T)$。
  - 视频编码得潜变量 $\mathbf{x}_1 = \mathcal{E}(\mathbf{y}_{\text{event}})$，构造插值潜变量 $\mathbf{x}_s = (1-s)\epsilon + s\mathbf{x}_1$，$s \in [0,1]$ 为流时间。
  - 采用 flow matching 损失：$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{(\mathbf{x}_1,c),\epsilon,s}[||v_\theta(\mathbf{x}_s, s, c) - (\mathbf{x}_1 - \epsilon)||_2^2]$，条件 $c$ 包括子任务指令和初始观测 $o_0$。
  - 训练配置：预测49帧，分辨率 $416 \times 240$，Wan2.1-I2V-14B 继续预训练 30,000 步，batch size 1024，学习率 1e-5。
- **Stage 2a：片段级视频模型后训练**：
  - 固定预测时域 $H$ 秒，帧率 $f$，采样间隔 $\delta = fH/N$。
  - 从帧 $t$ 开始的目标视频：$\mathbf{y}_{\text{chunk}}^{(t)} = (o_t, o_{t+\lfloor \delta \rfloor}, \dots, o_{t+\lfloor N\delta \rfloor})$，使用相同 flow matching 损失。
- **Stage 2b：一致性蒸馏（Consistency Distillation）**：
  - 损失函数：$\mathcal{L}_{\text{CD}} = \mathbb{E}\left[\lambda(s, s') ||F_\theta(\mathbf{x}_s, s, c) - \text{stopgrad}(F_{\bar{\theta}}(\hat{\mathbf{x}}_{s'}, s', c))||_2^2\right]$，其中 $\hat{\mathbf{x}}_{s'}$ 由冻结教师流从 $(\mathbf{x}_s, s)$ 积分至 $s'$ 得到，$\bar{\theta}$ 为EMA学生参数。
  - 推理时单步生成：$F_\theta(\epsilon, 0, c)$，视频生成约 0.12 秒（bfloat16 + torch.compile）。
- **Stage 3：动作预训练（Across Datasets）**：
  - 动作专家为 0.9B 参数 DiT，采用 MoT 架构。
  - 训练时冻结视频 DiT 基础参数，仅用 LoRA 适配视频骨干，动作专家从零初始化。
  - 动作生成：5 步去噪，约 0.1 秒。总延迟每动作块约 0.22 秒。
- **VLM 高层规划**：
  - 子任务规划：VLM 生成详细子任务指令，处理长 horizon 任务和歧义指令。
  - Prompt 增强：VLM 生成类似训练字幕的详细子任务提示，提升视频生成质量。

## 实验与结果
- **数据集**：
  - 视频预测基准：从 prior work (Chen et al., 2025; Zhang et al., 2026b) 随机采样 50 个人类手操作和 50 个机器人操作图像‑指令对。
  - 真实机器人测试：ALOHA 平台，10 个零样本任务类别。
  - 仿真基准：LIBERO‑ID、LIBERO‑Pro、LIBERO‑OOD（训练仅限四个标准 LIBERO 套件），RoboDojo 仿真基准（42 个双臂任务，5 个能力维度）。
- **评估基线**：
  - 视频模型：Wan‑2.1‑I2V‑14B‑480p、LVP、Cosmos3‑Nano‑16B、Cosmos3‑Super‑Image2Video‑64B、VPP2‑Fixed‑Step（消融）。
  - 机器人策略：$\pi_0$、$\pi_{0.5}$（Wan‑14B版本）、Fast‑WAM、MolmoAct、X‑VLA、AtomVLA、Cosmos‑Policy、DiT4DiT、Temporal Ratio、OpenWAM‑α、Xiaomi‑Robotics‑1、GPT‑6‑Astra。
- **主要结果**：
  1. **视频预测指令遵循成功率**（Table 2）：VPP2 人类手 0.80、机器人 0.90，超越 Cosmos3‑64B（人类手 0.70、机器人 0.78），提升幅度分别为 +10% 和 +12%（绝对百分点）；在复杂需空间理解任务上优势显著（Figure 7）。
  2. **真实机器人零样本性能**（Figure 8，ALOHA）：VPP2 平均成功率 58.5%，超越 $\pi_{0.5}$（40.0%）和 Fast‑WAM（20.5%），在10类任务中9类最优。
  3. **LIBERO 基准**（Table 3）：
     - LIBERO‑ID：VPP2 98.8%，与最强基线（DiT4DiT 98.6%）持平。
     - LIBERO‑Pro：VPP2 45.0%，远超次强 $\pi_{0.5}$（11.0%），提升 34 个百分点；Task扰动下达 47.8%，表明遵循指令而非记忆轨迹。
     - LIBERO‑OOD：VPP2 63.9%，远超视频策略最高（Cosmos‑Policy 10.5%），也超过专为泛化设计的 Temporal Ratio（59.4%）。
  4. **RoboDojo 仿真基准**（Table 4）：VPP2 平均得分 35.51，成功率 29.47%，达到 SOTA；超越 GPT‑6‑Astra（28.97/22.48%）。
  5. **子任务规划提升**（Figure 9）：结合 VLM 规划后，平均成功率 57.6%，较无规划 baseline（27.6%）提升 30 个百分点；语言条件积木堆叠（10%→52%）和井字棋（0%→38%）增益最大。
- **推理延迟**：视频单步生成 0.12 秒，动作生成 0.1 秒，总延迟约 0.22 秒/动作块。

## 相关工作脉络
1. **World Action Models (WAMs)**：本文工作与 Hu et al. (2024)、Liao et al. (2025)、Yan et al. (2026b) 等一脉相承，但指出其共同缺陷——在 OOD 场景下预测不可靠。本文通过事件级预训练+详细字幕+固定时域蒸馏，在保留视频泛化能力的同时提升动作泛化。
2. **视频基础模型应用于机器人**：LVP (Chen et al., 2025) 也基于 Wan‑2.1‑I2V‑14B 适配操作任务，但未强调事件级字幕和坐标对齐；Cosmos‑Policy (Kim et al., 2026) 使用 64B 模型，参数量更大但推理慢；本文以 14B 模型在多项指标上超越 Cosmos3‑64B。
3. **隐式逆动力学建模**：Du et al. (2023)、Black et al. (2023) 早期工作先逐帧生成视频再推断动作，延迟高；本文通过单步蒸馏视频模型+MoT 动作专家，实现高效闭环控制。
4. **训练稳定性与泛化**：Mishra et al. (2026) 发现动作训练会损害视频模型泛化；本文分阶段训练（冻结视频骨干仅 LoRA 适配）正是对此问题的回应，平衡了预测质量与动作性能。
5. **多视图与坐标系统一**：现有工作通常直接拼接多视图或使用单一视角；本文提出 T 形统一输入和显式坐标对齐（工作空间+末端执行器），解决跨数据集动作语义混淆。
6. **分层规划架构**：Shi et al. (2025) (Hi Robot) 采用 VLM 规划；本文同样引入 VLM 负责高层语义与记忆，VPP2 专注低级映射，但特别强调 VLM 生成的详细子任务提示能进一步提升视频生成质量。

## 局限性与未来方向
- **数据偏差**：训练数据集中于 egocentric bimanual 场景（占 80% 以上），对于非对称任务、单臂操作或远距离观察场景的泛化可能受限。
- **仿真到真实的差距**：RoboDojo 为仿真基准，虽然 ALOHA 真实实验表现良好，但未在更多真实机器人平台（如 Mobile Manipulator、Humanoid）上验证。
- **VLM 依赖**：子任务规划依赖外部 VLM，增加系统复杂性和延迟；若 VLM 分解错误，将影响底层执行。
- **固定时域限制**：后训练固定预测时域（人类手2秒、操作8秒）可能不适用于时域差异较大的任务。
- **未来方向**：
  1. 扩展数据多样性，纳入更多 embodiment 和任务类型。
  2. 研究端到端联合训练技巧，在不损害视频泛化的前提下进一步融合动作信号。
  3. 探索自适应时域预测，根据任务复杂度动态调整预测长度。
  4. 将系统部署到更广泛的真实机器人平台，评估长程任务（hours‑long）的稳定性。

## 研究启发与可借鉴点
1. **事件级字幕与轨迹对齐**：详细事件描述（主动末端执行器、目标对象、视角变化）能显著降低视频预测不确定性，可迁移至任何视频生成辅助的决策任务。
2. **分阶段训练策略**：先独立强化基础模型（视频）的领域泛化，再添加专用模块（动作专家）并冻结基础参数，是防止灾难性遗忘的有效范式。
3. **一致性蒸馏加速推理**：将多步视频生成模型蒸馏为单步模型，同时保持生成质量，适用于任何需要低延迟的视频生成下游应用。
4. **统一坐标系处理**：跨数据集训练时，显式对齐工作空间和末端执行器坐标系可消除动作语义歧义，值得在机器人多源数据融合中推广。
5. **VLM 提示增强**：利用 VLM 生成详细子任务提示，不仅服务于规划，也能提升底层视频模型的生成质量，体现“提示工程”在具身智能中的双重价值。

## 关键术语表
- **World Action Model (WAM)**：将视频预测先验迁移到动作学习的通用机器人策略类模型，通过显式或隐式逆动力学从预测视频中推导动作。
- **Flow Matching**：一种生成模型训练目标，通过优化预测流速度场来学习从噪声到数据的确定性轨迹，常用于连续扩散模型。
- **Consistency Distillation**：一致性蒸馏，强制模型在相邻流时间步给出一致的终端预测，从而实现单步生成，源自 Song et al. (2023)。
- **Mixture‑of‑Transformers (MoT)**：混合变压器架构，通过稀疏路由将不同模态或功能分配到不同的 Transformer 专家，实现可扩展的多模态融合。
- **T‑shape Multi‑view Input**：T形多视图输入格式，将主视图置于左侧，两个辅助视图垂直堆叠于右侧，用于统一不同机器人数据集的相机布局。
- **End‑effector Coordinate Alignment**：末端执行器坐标对齐，包括工作空间对齐（世界帧变换）和末端执行器对齐（局部变换），使不同数据集的动作表示一致。
- **LIBERO‑Pro / LIBERO‑OOD**：LIBERO 基准的两个扩展，分别测试位置/任务扰动下的泛化和常见元素的新组合下的构式泛化。
- **RoboDojo**：统一仿真‑真实基准，包含42个双臂任务，从泛化、精度、长程执行、记忆、开放词汇指令五个维度评估通用操作策略。

## 可复现要素
- **数据集**：论文未公开原始数据列表（但列出了使用的数据集名称，如 OXE、RoboMIND、DROID、RH20T、Molmoact、AgibotWorld‑Beta、RoboCOIN、RDT、GM100、ABC‑130k、Aloha InternData‑A1、Galaxea Open‑World、EgoDex、Ego4D、Epic‑Kitchen、Open‑vid、Panda‑70M）；数据过滤与处理管线（字幕模板、坐标对齐代码）可在项目页面获取。
- **代码/权重**：论文声明“Detailed training configurations, code, and model checkpoints are available on our anonymous project website.”（具体网址见论文项目页 https://robert-gyj.github.io/video-prediction-policy-2）。
- **关键超参**：
  - 视频继续预训练：30,000 步，batch size 1024，学习率 1e-5，预测 49 帧，分辨率 416×240。
  - 后训练固定时域：人类手 H=2 秒，机器人操作 H=8 秒。
  - 一致性蒸馏：采样 $s'=1$ 概率 0.5，否则从其他流时间采样。
  - 动作专家：0.9B 参数 DiT，5 步去噪。
  - 视频骨干仅用 LoRA 适配（具体秩和 alpha 未给出）。
