---
title: "RESUME-RECURRENT-STATE-UPDATES-FROM-MO-TION-AND-RESIDUAL-SIG"
source: https://arxiv.org/pdf/2609.39563v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:50:16"
field: "视频语言模型与压缩域表征"
keywords: ["video language model", "codec representation", "temporal reasoning", "state transition", "motion vectors", "residuals", "multimodal"]
innovations: ["将编解码递推关系实例化为因果潜状态转移，构造性地保留参考依赖、构成性与路径依赖性", "运动向量与残差早期融合为单观测并跨步累积，共享读出头输出时序轨迹而非独立帧token", "通过时间反转探针与冻结转移实验建立表征-性能的因果链路，并在时间基准上同时超越RGB基线与编解码基线"]
benchmarks: ["TempCompass", "TOMATO", "MVBench", "Video-MME", "PerceptionTest", "NExT-QA", "ActivityNet-QA", "LVBench", "Video-TT", "Video-MMMU"]
---

# 论文速读：RESUME-RECURRENT-STATE-UPDATES-FROM-MO-TION-AND-RESIDUAL-SIG

## 一句话总结
论文提出 RESUME，一种状态化编解码表示方法：用锚点 I 帧初始化紧凑潜状态，将运动向量和残差融合为单次观测来因果地更新该状态，最后通过共享读出头暴露 VideoLM 兼容 tokens。在相同的每预测帧 token 预算下，相比 RGB 帧基线 LLaVA-Video-7B 和编解码基线 CoPE-7B，在三个时间推理基准上均获得提升，同时通用/长视频 QA 基本持平。

## 研究问题与动机
- 现有 VideoLM 对采样 RGB 帧独立编码，长视频只能在"耗尽 token 预算"和"丢失采样帧之间的变化"之间二选一，瓶颈是表征能力而非上下文长度。
- 编解码感知前端虽读取运动向量和残差，但部署形态下每个 P 帧仍是独立 token 组，参考依赖与"有序累积变化"被留给语言模型注意力去重建。
- 视频 clip 与其时间反转包含完全相同的帧，差异仅在于变化顺序；对称池化在结构上丢弃这一轴，时序顺序是独立于静态内容与净变化的可测量信息维度。
- 编解码递推关系 $\hat{I}_t = \operatorname{Warp}(\hat{I}_{t-1}, \tau_t) + \delta_t$ 天然蕴含参考依赖性、构成性与路径依赖性，这些性质在序列更新的状态中是构造性保留的，而在独立 token 化中必须被语言模型重新推断。

## 核心贡献（创新点）
1. 将编解码感知视频表示形式化为相邻锚点间的因果潜状态转移，明确定义参考依赖、构成性、路径依赖三项设计约束。与已有工作仅把运动/残差当作独立 token 源的本质区别在于：把递推关系本身作为表示层面的结构保留下来。
2. 实现 RESUME 状态编解码前端，I 帧初始化潜状态、运动与残差早期融合为单观测、交叉注意力+门控残差做因果更新，约 21M 参数；与 Video-LaVIT/EMA/CoPE 等在同一每预测帧 token 预算下提供更强的时序表征。区别在于读出头共享、状态跨步累积，而非每帧独立编码。
3. 提供训练-free 探针与冻结转移的实证链条：证明对称池化丢弃的时序轴在冻结视觉特征中非空；冻结转移展现锚点依赖、顺序敏感与训练外推的有用行为。与 CoPE 仅在前训练模拟一步特征空间 warp 且在推理前移除辅助模块的区别在于：递推结构持续到推理并跨预测帧累积。
4. 在十项基准上验证端到端效果，时间基准全面超越 RGB 基线与编解码基线，通用/长视频 QA 基本持平，并报告了延迟数据。与 prior codec-aware 工作在效率评估维度的区别在于同时覆盖推理延迟与 I 帧/读出头配比 trade-off。

## 方法详解
- **GOP 组织与符号**：视频 $V=(F_1,\dots,F_T)$ 划分为 GOP，每 GOP 以帧内编码 I 帧开头，随后为预测帧 $P_t=(\tau_t, \delta_t)$，其中 $\tau_t$ 为分块运动向量、$\delta_t$ 为残差；不使用 B 帧以维持因果性。
- **状态转移系统**：$z_0=\mathrm{Init}(\phi_{\mathrm{RGB}}(I_0))$，$o_t=\mathrm{Obs}(\tau_t,\delta_t)$，$z_t=\mathcal{T}(z_{t-1},o_t)$，$Y_t=\mathrm{Read}(z_t)$。$z_t$ 为持久潜状态，$Y_t$ 为进入语言模型的 token 读出头。
- **锚点初始化**：冻结视觉编码器产出 patch tokens $X_I$，与可学习查询 $q$ 拼接后过两层预归一化 Transformer，保留查询位置作为 $z_0$；I 帧仍走原路径进入语言模型，$z_0$ 仅作为后续更新的参考。
- **运动-残差早期融合**：运动向量 patchify 后用共享 MLP 嵌入，残差用步长对齐运动网格的卷积 stem 嵌入，位置-wise 拼接后融合到状态宽度，形成单观测 $o_t$。
- **因果状态更新**：$a_t=\mathrm{CrossAttn}(\mathrm{LN}(z_{t-1}),\mathrm{LN}(o_t))$，$g_t=\sigma(W_g[z_{t-1};a_t])$，$z_t=\mathrm{LN}(z_{t-1}+g_t\odot U(z_{t-1}+a_t))$，其中 $U$ 为小 MLP。交叉注意力选择相关空间部分，门控控制写入量。
- **共享读出头**：可学习查询 $r$  attends 到 $z_t$，经两层 Transformer 后投影回视觉编码器宽度，产出 8-token 读出门 $Y_t$。
- **两阶段训练**：Stage 1 仅训练状态转移，目标为冻结 SigLIP 编码器对真实帧的池化特征 $\bar{X}_t$；Stage 2 冻结转移与视觉塔，接入 LLaVA-Video-7B，只训语言模型与专用投影层。
- **损失函数**：$\mathcal{L}=\frac{1}{T}\sum_t\|Y_t-\bar{X}_t\|_2^2+\lambda_{\cos}\frac{1}{T}\sum_t(1-\cos(Y_t,\bar{X}_t))+\lambda_\Delta\frac{1}{T-1}\sum_t\|(Y_t-Y_{t-1})-(\bar{X}_t-\bar{X}_{t-1})\|_2^2$，分别拟合语义端点、方向与连续读出头之间的变化轨迹。
- **关键超参**：状态 16 slots × 宽度 512，读出头 8 tokens，I 帧最多 64 个，每 I 帧后 4 次预测更新；Stage 1 峰值 lr $2\times10^{-4}$、global batch 16384、64 GPU；Stage 2 lr $1\times10^{-5}$、global batch 128、10000 步。

## 实验与结果
- **基准**：通用 QA（Video-MME 无字幕、PerceptionTest、NExT-QA、ActivityNet-QA）、时间/运动推理（TempCompass、TOMATO、MVBench）、长视频（LVBench、Video-TT、Video-MMMU），共十项。
- **主要结果**：相对 LLaVA-Video-7B，TempCompass +2.8、TOMATO +5.1、MVBench +3.9；相对 CoPE-7B，三项分别 +0.5、+1.7、+0.6。通用/长视频中 PerceptionTest +3.3、ActivityNet-QA +4.4 提升明显，Video-MME 较 LLaVA-Video-7B 降 1.4、LVBench 较 CoPE-7B 降 3.2，作者归因于 Stage 2 训练数据规模与分布差异。
- **训练-free 探针**：帧对称池化在两个冻结塔上翻转率为 0；差值对称探针 0.68–0.71；时间加权顺序敏感探针 0.75–0.86。说明顺序是被对称聚合丢弃的非空轴。
- **冻结转移验证**：同更新作用于不同锚点，97%/86% 视频仍更接近各自锚点；正确顺序胜出 65% 视频；在 8 步与 16 步延展下分别有 75%/68% 视频优于无记忆控制。
- **消融**：匹配 I 帧预算下加读出头，Video-MME 在 8/16 I 帧时 +0.8/+1.3，LVBench +0.6/+0.9。
- **延迟**：64 秒 clip、生成 64 token，32 I+32 读出头配置 TTFT 0.461s、E2EL 1.795s，优于 64 帧 LLaVA-Video-7B 的 0.686/2.094s。

## 相关工作脉络
- **帧基 VideoLM**：Video-LLaMA、VideoChat2、VideoLLaMA2、LLaVA-NeXT-Video、LLaVA-Video 均独立编码采样 RGB 帧；RESUME 在前端保留编解码递推结构而非事后压缩。
- **编解码基元 VideoLM**：Video-LaVIT 离散化运动但丢弃残差；EMA 聚合为固定 GOP 摘要、-collapse P 帧顺序；CoPE-7B 编码运动+残差为独立 token 并在前训练模拟一次 warp，但参考来自解码 RGB 且在集成前移除辅助模块。RESUME 与它们在表征层面的关键差异是状态跨步累积与共享读出头。
- **压缩域识别**：Wu 等、Wang 等、Das Biswas 等证明运动/残差携带可用动力学；RESUME 将其用于状态更新而不仅是识别特征。
- **其他压缩策略**：Token merging、DyCoke、LLaVA-Scissor 等属 post-hoc 压缩，已付编码成本；RESUME 在编解码原生表示层面工作。
- **编解码状态传播**：Das Biswas 等维护独立运动/残差状态但无 I 帧初始化；ReMoRa 双向扫描且仅 GOP 级读出头、无残差输入。RESUME 的优势在于因果 GOP 局部更新+残差输入+每预测帧读出头。

## 局限性与未来方向
- 状态仅在相邻 I 帧之间传递，下一 I 帧重新初始化而非继续累积，因此不对整段视频维持状态；跨 GOP 的长程聚合仍依赖语言模型。
- Stage 1 监督来自冻结视觉编码器，潜状态为编解码条件化的隐状态，无显式物理标注，可解释性有限。
- Stage 2 仅在 LLaVA-Video-178K 上微调，未使用原版 LLaVA-Video 训练中的三套学术 QA 与 LLaVA-OV 图像对齐数据，导致 Video-MME、LVBench 等对训练数据组成敏感的基准出现回落。
- 论文仅使用 I 帧与 P 帧，B 帧因破坏因果性被排除；对包含 B 帧的现实码流的适配未讨论。
- 效率主张针对相同每预测帧 token 预算下的表征收益，未与更激进的 dense-readout 变体（每 I 帧 4/8/16 读出头、可覆盖数小时视频）做全面对比；延迟仅在短 clip 上评估。

## 研究启发与可借鉴点
- 用时间反转实验分离静态内容、净变化与顺序轴，并以零参数探针验证"被丢弃信息是否非空"，这种表征诊断范式可直接迁移到其他时序模态（音频、传感器序列）。
- 将编解码递推的三项性质（参考依赖、构成性、路径依赖）转化为状态更新的结构约束，可推广到任意具有预测结构的信号域（如光流、神经辐射场时间序列）。
- 交叉注意力+门控残差的因果更新模块是通用状态转移子结构，可与 Mamba/State Space Model 等现有序列建模手段对照或组合。
- 训练-free 探针配合 question-based 与 reversal axis 双视角验证信息层次，避免单一指标的欺骗性，适合纳入团队的标准表征审计流程。
- 共享读出头使不同时刻 token 处于同一坐标系，语言模型接收"轨迹而非独立 token 组"，这一设计对任何需要时序累积的 multimodal front-end 都有借鉴价值。

## 关键术语表
- **GOP (Group of Pictures)**：视频编解码基本单元，由一个 I 帧与若干 P 帧组成。
- **运动向量 (Motion Vectors, $\tau$)**：描述图像块从参考帧位移的矢量场。
- **残差 (Residuals, $\delta$)**：运动补偿后仍未被解释的像素级差值。
- **I 帧 / P 帧**：帧内独立编码帧与基于参考帧预测编码帧。
- **编解码递推 (Codec Recurrence)**：$\hat{I}_t=\operatorname{Warp}(\hat{I}_{t-1},\tau_t)+\delta_t$，预测帧为参考状态与更新的函数。
- **潜状态 (Latent State, $z_t$)**：跨预测帧累积的 16×512 向量，承载参考依赖与有序变化。
- **状态读出头 (State Readout Head)**：共享投影头，从 $z_t$ 产出 8-token 的 VideoLM 兼容表示。
- **参考依赖 / 构成性 / 路径依赖**：递推关系的三项性质，分别指更新语义依赖当前参考、小变化累积为大事件、结果依赖更新顺序。

## 可复现要素
- 视觉编码器：冻结 SigLIP；语言模型基线：LLaVA-Video-7B。
- 状态维度：16 slots × 宽度 512；读出头：8 tokens；每 I 帧后 4 次预测更新。
- 训练数据：Stage 1 自构建；Stage 2 仅 LLaVA-Video-178K，未使用原版学术 QA 与图像对齐数据。
- 评估重编码：30 fps、仅 I/P 帧、I 帧间隔 8 秒；每组一个 RGB 锚点+4 次预测更新。
- 数据集：Video-MME、PerceptionTest、NExT-QA、ActivityNet-QA、TempCompass、TOMATO、MVBench、LVBench、Video-TT、Video-MMMU（均为论文已引用公开基准）。
- 代码/权重：论文未明确声明开源；训练细节附录齐全，关键超参可复现。
