---
title: "ON-POLICY-SELF-DISTILLATION-FOR-MULTI-TURN-IMAGE-EDITING"
source: https://arxiv.org/pdf/2609.35611v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:51:08"
field: "图像编辑与生成"
keywords: ["多轮图像编辑", "on-policy self-distillation", "流匹配", "长程鲁棒性", "训练-测试分布偏移", "LME-Bench"]
innovations: ["通过身份rollout隔离模型自生误差并进行on-policy自蒸馏", "稀疏查询速度匹配实现干净条件教师的跨状态监督", "自适应rollout课程与门控教师晋升机制"]
benchmarks: ["LME-Bench", "MSE-Bench", "ImgEdit"]
---

# 论文速读：ON-POLICY-SELF-DISTILLATION-FOR-MULTI-TURN-IMAGE-EDITING

## 一句话总结
本文针对指令式图像编辑在多轮递归编辑中性能急剧衰退的问题，提出 MT-OPSD（On-Policy Self-Distillation），一种无需多轮标注数据的自蒸馏框架，通过身份 rollouts 隔离模型自生误差，并利用干净条件教师模型进行 on-policy 速度匹配，显著提升了多轮编辑的长程鲁棒性。同时构建了 LME-Bench 基准，包含 100 个 10 轮编辑会话用于系统评估。

## 研究问题与动机
1. **多轮递归编辑崩溃**：现有编辑模型在反复基于自身输出进行编辑时，小误差快速累积，导致高频色差噪声、结构碎裂、主体身份漂移等严重退化。
2. **根本原因——训练/测试条件分布不匹配**：模型训练时仅以干净源图像为条件，而推理时多轮编辑需依赖上一轮模型自身产生的有误差输出，这种 conditioning distribution shift 是退化主因。
3. **多轮标注数据稀缺且端到端优化昂贵**：直接训练于多轮序列需要跨数百去噪步骤的梯度回传，计算成本不可行；现有免训练方法（如 Emu Edit）仅适用于局部编辑，无法处理全局变换。
4. **缺乏系统性的长程评测基准**：现有基准最多含 5 轮编辑，难以评估本文关注的长程鲁棒性退化问题。

## 核心贡献（创新点）
1. **揭示并归因多轮崩溃现象**：在三种现代编辑模型中系统观察到递归编辑退化，并将其归因于 conditioning distribution 的训练-测试不匹配，而非模型特异性问题。
2. **提出 MT-OPSD 自蒸馏框架**：通过身份指令 rollouts 构造自生条件状态，利用干净条件教师模型对 self-generated 状态进行 on-policy velocity matching 监督，无需多轮标注或外部教师。
3. **构建 LME-Bench 基准**：包含 100 个 10 轮会话（6 个局部 + 4 个全局编辑），每个会话逐轮评估编辑准确性、视觉一致性与图像质量，填补长程评测空白。
4. **实验验证跨架构泛化**：在 Qwen-Image-Edit-2511、FLUX.2-klein-base、FireRed-Image-Edit 三个开源骨干上均显著提升 SR@10（0.38–0.52）并降低 CR@10（≤0.04），同时几乎无损单轮编辑能力。

## 方法详解
**MT-OPSD 框架由四个组件构成：**

1. **自生成 Rollout 状态构造**：使用身份指令 $e_{\text{id}}$（如"Make everything unchanged"）递归调用当前学生模型：
   $$\tilde{I}^{(k)} = G_{\theta_S}(\tilde{I}^{(k-1)}, e_{\text{id}}), \quad \tilde{I}^{(0)} = I^{(0)}$$
   由于语义内容不变，$\tilde{I}^{(k)}$ 与 $I^{(0)}$ 的差异主要来自模型自生误差的累积。

2. **双分支训练目标**：
   - **身份分支**（防止进一步漂移）：以 rollout 状态自身为目标，约束模型不引入额外变化：
     $$\mathcal{L}_{\text{id}} = \mathbb{E}_{t,\epsilon}\left[\|v_{\theta_S}(\tilde{x}_t, t, \tilde{I}^{(k)}, e_{\text{id}}) - (\epsilon - \tilde{x}_0)\|_2^2\right]$$
   - **编辑分支**（保持编辑能力）：在学生状态 $\tilde{I}^{(k)}$ 与教师干净条件 $I^{(0)}$ 之间进行稀疏查询速度匹配：
     $$\mathcal{L}_{\text{edit}} = \mathbb{E}_{q \sim p_q}\left[\|v_{\theta_S}(\bar{x}_{t_q}, t_q, \tilde{I}^{(k)}, e) - v_{\theta_T}(\bar{x}_{t_q}, t_q, I^{(0)}, e)\|_2^2\right]$$
     其中 $\bar{x}_{t_q} = \text{sg}(x_{t_q})$，查询步骤从 Beta 分布采样（Qwen/FireRed 用 Beta(5,5)，FLUX 用 Beta(2,5)）。

3. **自适应 Rollout 课程**：根据漂移度量（$\tilde{I}^{(k)}$ 与 $I^{(0)}$ 的像素均值差）动态调整深度——仅当连续 $P$ 步漂移低于阈值 $\tau$ 时深度 +1，避免早期暴露于重度退化状态导致训练不稳定。实践中通常在 4 轮左右饱和。

4. **门控教师晋升**：利用验证集（23 个 10 轮会话）由 VLM judge 异步评估候选 checkpoint，当 $\Delta \text{SR}_k$ 或 $\Delta \text{CR}_k$ 满足阈值时晋升为学生新教师，确保教师始终处于较优状态。

## 实验与结果
**数据集与基准：**
- **LME-Bench**：100 个 10 轮会话（每轮基于上一轮输出），覆盖 10 类语义类别，含局部与全局编辑
- **MSE-Bench**：100 个 5 轮会话（以局部编辑为主）
- **ImgEdit**：单轮编辑评测基准

**主要结果（LME-Bench）：**
| 模型 | SR@3 | SR@5 | SR@10 | CR@10 |
|------|------|------|-------|-------|
| Qwen-Image-Edit-2511 + MT-OPSD | **1.00** | **0.91** | **0.44** | **0.02** |
| FireRed-Image-Edit + MT-OPSD | **0.98** | **0.88** | **0.52** | **0.03** |
| FLUX.2-klein-base + MT-OPSD | **0.95** | **0.76** | **0.38** | **0.04** |

相比基座模型，MT-OPSD 将 SR@10 从 0.03–0.15 提升至 0.38–0.52，CR@10 从 0.25–0.61 降至 ≤0.04。免训练基线（Emu Edit、FreqEdit、VAE-LFA）效果参差，FreqEdit 反而恶化长程性能。

**MSE-Bench 与 ImgEdit：**
- MT-OPSD 将 Qwen SR@5 从 0.24 提至 0.51，FireRed 从 0.40 提至 0.69
- ImgEdit 单轮分数几乎不变（Qwen: 4.51→4.49，FireRed: 4.56→4.52，FLUX: 4.20→4.28）

**关键结论：** 多轮鲁棒性可从 self-generated rollout 状态学习并迁移至真实多轮编辑序列，超越训练期间遇到的最大深度。

## 相关工作脉络
1. **免训练多轮校正方法**（Emu Edit、FreqEdit、VAE-LFA）：通过图像空间/潜空间修正降低累积误差，但依赖编辑假设（如局部编辑的像素回退），无法推广至全局变换；MT-OPSD 通过训练改变模型行为，具有更强的通用性。
2. **VINCIE / AnchorEdit**：训练专用模型进行因果多轮编辑，VINCIE 以视频架构适配交错图像序列；MT-OPSD 在现有单轮编辑骨干上通过自蒸馏扩展能力，无需架构改动。
3. **MT-EditFlow**（Huang et al., 2026a）：同样归因于 exposure bias，但通过强化学习与外部奖励监督解决；MT-OPSD 无需外部奖励信号，仅利用模型自身干净条件的编辑行为作为教师。
4. **On-policy self-distillation 系列**（D-OPSD、OPSD-V、DiffusionOPD、DanceOPD）：多在语言模型或视频生成中应用， teacher 提供特权上下文；本文将 OPSD 引入图像编辑，teacher 以干净源图像为上下文，student 以 self-generated 状态为上下文，形式新颖。
5. **多轮一致性保持方法**（MTC、Edit-R2）：侧重轨迹控制或 session-level 约束；本文从 conditioning distribution mismatch 角度切入，提供不同的理论视角与解决方案。

## 局限性与未来方向
1. ** Rollout 课程饱和深度有限**：实践中曲线通常在 ~4 轮饱和，虽然 10 轮结果仍显著提升，但未充分探索更深层鲁棒性。
2. **依赖 GPT-4o 评测**：基准构建与教师晋升均使用 VLM judge，存在评测偏差风险，且增加了对外部模型的依赖。
3. **全局编辑的鲁棒性仍弱于局部**：Nano Banana 在 LME-Bench 中极少崩溃但 SR 较低主要源于全局编辑失败，提示全局变换的长程稳定性仍是挑战。
4. **仅验证三个开源骨干**：未扩展至更多架构（如 SDXL、Stable Diffusion 3），泛化性有待进一步验证。
5. **未结合外部反馈机制**：当前方案完全自监督，未来可探索用户反馈或强化学习的结合。

## 研究启发与可借鉴点
1. **身份 rollout 作为误差隔离手段**：通过零编辑指令构造纯误差累积状态，为分析模型自生偏差提供了简洁可控的实验范式，可迁移至其他迭代生成任务（如视频编辑、音频处理）。
2. **稀疏查询速度匹配替代完整轨迹蒸馏**：仅需 2 个查询步骤即可有效对齐教师行为，大幅降低计算开销，比标准 flow matching 更适用于长程任务。
3. **自适应 curriculum 替代手动调度**：基于漂移度量的动态深度扩展避免了人工调参，为类似训练过程中的自适应探索提供了设计思路。
4. **门控晋升机制解耦评估与训练**：异步 VLM judge 用于 checkpoint 选择而非直接训练信号，兼顾了长期稳定性与训练效率，可借鉴至其他自蒸馏场景。
5. **单轮能力保持的量化验证**：通过 ImgEdit 基准确认多轮鲁棒性提升不以牺牲单轮质量为代价，这一验证策略值得在多任务优化研究中遵循。

## 关键术语表
**On-policy self-distillation (OPSD)**：学生在自身策略生成的状态上学习，同时以同一模型在更丰富条件下的预测作为教师信号的自蒸馏方式。
**Rollout state**：通过重复应用身份指令从干净源图像生成的自生条件状态，用于模拟多轮编辑中的误差累积。
**Velocity matching**：在流匹配框架下对齐教师与学生模型的速度场预测，作为蒸馏损失。
**Sparse query-based distillation**：仅在学生去噪轨迹中采样少量查询步骤进行教师-学生对齐，而非全程匹配。
**SR@k (Success Rate at turn k)**：前 k 轮均成功完成的会话比例，衡量累积编辑成功率。
**CR@k (Collapse Rate at turn k)**：截至第 k 轮已发生持续视觉退化的会话比例。
**Gate set**：用于教师晋升评估的独立验证集（23 个 10 轮会话），不参与主实验报告。
**Drift threshold & patience**：控制 rollout 课程深度递增的自适应超参，分别设定漂移允许阈值与连续达标步数。

## 可复现要素
- **数据集**：LME-Bench（100 个会话，源图像由 Z-Image-Turbo 生成，分辨率 1024×1024）；MSE-Bench；ImgEdit。**未声明公开**。
- **代码/权重**：项目页面 https://liangbingzhao.github.io/MT-OPSD/，**论文未明确声明开源**。
- **关键超参**：LoRA rank=32, α=64；训练样本 2000 对；初始 rollout 深度=2，最大深度=10；身份损失权重=10；查询分布 Beta(5,5) 或 Beta(2,5)；学习率 1–3×10⁻⁴；漂移阈值 τ=9–10，patience=10；训练分辨率 512²。
- **硬件**：4×H100/H200 训练 + 4×GPU 异步评估。
