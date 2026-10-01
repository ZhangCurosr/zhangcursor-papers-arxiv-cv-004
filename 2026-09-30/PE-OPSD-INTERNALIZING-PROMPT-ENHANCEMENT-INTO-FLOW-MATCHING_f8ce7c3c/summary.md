---
title: "PE-OPSD-INTERNALIZING-PROMPT-ENHANCEMENT-INTO-FLOW-MATCHING"
source: https://arxiv.org/pdf/2609.36638v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:48:21"
field: "文生图模型后训练"
keywords: ["Prompt Enhancement", "On-Policy Self-Distillation", "Flow-Matching", "Privileged Information", "Text-to-Image Generation", "Knowledge Distillation"]
innovations: ["将提示词增强重新定义为文生图生成中的文本侧特权信息，突破特权信息需来自多模态的先验假设", "提出 PE-OPSD 框架，在学生原始提示词轨迹上用增强提示词教师提供向量场蒸馏，无需额外图像对", "在多个模型族和多种 PE 下实现最强的聚合提示词忠实度，同时保留基础模型的推理延迟"]
benchmarks: ["GenEval", "GenEval2", "DPG-bench", "T2I-CompBench++", "EvalMuse"]
---

# 论文速读：PE-OPSD: Internalizing Prompt Enhancement into Flow-Matching Models via On-Policy Self-Distillation

## 一句话总结
本文提出 PE-OPSD，一种面向文生图 Flow-Matching 模型的在线策略自蒸馏方法，将提示词增强器（PE）视为训练时的特权信息，通过在原始提示词学生轨迹上提供增强提示词教师的向量场监督，将 PE 带来的生成优势内化到仅需原始提示词即可推理的模型中，在不增加推理延迟的前提下显著提升提示词忠实度。

## 研究问题与动机
- **用户提示词简短与模型需求详细条件之间的根本错配**：用户的原始 prompt 往往只包含简洁描述（如"a girl reading under a tree"），而文生图模型在更详细的文本条件约束下才能可靠地遵循指令、生成视觉一致的结果。
- **现有 Inference-time PE 方案存在部署瓶颈**：工业界广泛采用的 Prompt Enhancer（如 BeautifulPrompt、PromptEnhancer）在推理时重写提示词，但引入了额外的延迟与计算开销，且长提示词增加了文本处理与条件注入的负担。
- **已有方法未将 PE 效果内化到生成器本身**：现有工作要么在训练时用增强后的 caption 替换原始文本（无法教授模型从原始提示恢复行为），要么在推理时持续调用 PE（保留部署成本），缺乏将 PE 收益内化到仅接收原始提示词的学生模型中的系统性研究。
- **提示词增强可被重新诠释为"特权信息"（Privileged Information）**：增强后的提示词 $p^+$ 保留了 $p$ 的显式语义约束并补充了合理的细节展开，本质上是一种只存在于训练阶段的天然文本特权信息，可用于指导仅依赖原始提示词学习的模型。

## 核心贡献（创新点）
- **将提示词增强重新定义为文生图生成中的特权信息**：突破以往将特权信息局限于额外模态（如高分辨率图像、区域裁剪）的范式，证明同类能力可通过纯文本侧的增强提示词实现，且无需修改模型的 conditioning 接口。与已有工作相比，PE-OPSD 的"特权信息"来自原始提示词的自动扩写而非外部模态输入。
- **提出面向 Flow-Matching 的 PE-OPSD 在线策略自蒸馏框架**：学生在原始提示词 $p$ 下采样自身轨迹，教师仅在相同轨迹状态上以增强提示词 $p^+$ 提供向量场目标，通过 stop-gradient 的自蒸馏损失将增强提示词诱导的行为迁移到原始提示词条件下。与 SFT 的区别在于无需构建额外的文本-图像对，与 Off-policy 的区别在于监督信号作用于学生自身的原始提示词轨迹而非教师轨迹。
- **在多个模型族和多种 PE 下实现最强的提示词忠实度，同时保留基础模型推理效率**：在 SD3.5-M、Z-Image、Z-Image-Turbo 三大骨干模型上，PE-OPSD 均超越 Inference-time PE、SFT 和 Off-policy Distillation，且推理延迟与 Base 完全一致，比部署 PromptEnhancer-7B/32B 快 1.96×/4.51×。

## 方法详解
- **整体训练流程**：给定训练集 $\mathcal{D} = \{(p, p^+)\}$（通过离线应用 PE 构造），学生模型 $f_\theta$ 和教师模型 $f_{\bar{\theta}}$ 共享架构并从同一预训练参数初始化（$\theta = \bar{\theta} = \theta_{\text{pre}}$）。训练时，学生在原始提示词 $p$ 条件下沿 Flow-Matching 轨迹进行 Euler 步进采样：$x_{t_{k-1}} = x_{t_k} - \Delta t_k \cdot v_\theta(x_{t_k}, t_k, p)$，然后在学生访问的每个状态 $x_{t_k}$ 处，以增强提示词 $p^+$ 评估教师的向量场 $v_{\bar{\theta}}(x_{t_k}, t_k, p^+)$，作为该状态的监督目标。
- **蒸馏损失函数（统一形式）**：
$$\mathcal{L}_{\text{PE-OPSD}}(\theta; \bar{\theta}) = \mathbb{E}_{(p, p^+)\sim\mathcal{D}}\left[\frac{1}{K}\sum_{k=1}^{K} \omega(t_k) \left\|v_\theta(x_{t_k}, t_k, p) - \text{sg}[v_{\bar{\theta}}(x_{t_k}, t_k, p^+)]\right\|_2^2\right]$$
其中 $\text{sg}[\cdot]$ 为 stop-gradient 操作。三种变体仅在时间步权重 $\omega(t_k)$ 上不同：$v\text{-loss}$ 取 $\omega=1$，$\mu\text{-loss}$ 取 $\omega=(\Delta t_k)^2$，$x_0\text{-loss}$ 取 $\omega=t_k^2$。论文默认使用 $\mu\text{-loss}$，因其在线 GenEval2 上取得最高分。
- **EMA 教师更新**：每步学生更新后，教师参数按指数移动平均更新：$\bar{\theta}_{n+1} \leftarrow \gamma \bar{\theta}_n + (1-\gamma)\theta_{n+1}$，其中 $\gamma=0.999$，保证教师随学生演进而保持预测稳定性。
- **无条件引导（CFG）处理**：消融实验表明，默认情况下不使用 CFG 以节约训练时间（训练时间减半），虽 GE2 略有下降但 GE 提升，整体效果更优；若需保留 CFG，可对无条件分支做 detach 以减少约 26% 的计算开销。
- **推理阶段**：训练完成后仅保留学生模型，直接以原始提示词 $p$ 生成图像，无需调用任何 PE 或教师模型，推理延迟与 Base 完全一致。
- **训练数据预处理**：使用 GPT-5.6 Sol 作为默认 PE 离线生成增强提示词，并通过一致性过滤器（consistency judge）自动校验是否保留了原始提示词的关键语义（计数、对象身份、属性等），不通过则重新生成，确保 PE 输出的质量。

## 实验与结果
- **数据集**：主要使用 GenEval（50K 训练）和 GenEval2（20K 训练，800 评估）的合成数据集进行 in-domain 评测，并使用 DPG-bench、T2I-CompBench++、EvalMuse 进行 out-of-domain 泛化评测。
- **基线方法**：Base（原始模型）、Base+PE（推理时调用 PE）、SFT（用增强提示词生成伪目标图像后微调）、Off-policy Distillation（与 PE-OPSD 相同目标但监督信号来自教师轨迹）。
- **主模型结果（Table 2，GPT-5.6 Sol 作为 PE）**：
  - **SD3.5-M**：GE=0.797（+0.169 vs Base），$I_{PF}$=13.35%，$I_{VA}$=1.79%，超越 Base+PE（GE=0.743，$I_{PF}$=10.22%）和 Off-policy（GE=0.770，$I_{PF}$=11.74%）。
  - **Z-Image**：GE=0.846（+0.196 vs Base），$I_{PF}$=16.07%，$I_{VA}$=3.30%，为所有方法最高。
  - **Z-Image-Turbo**：GE=0.863（+0.126 vs Base），$I_{PF}$=15.22%，$I_{VA}$=1.08%。
- **不同 PE 的泛化性（Table 4）**：在 PromptEnhancer-7B 和 PromptEnhancer-32B 上，PE-OPSD 同样取得最高的 $I_{PF}$ 和 $I_{VA}$，其中 PromptEnhancer-32B 带来最大忠实度增益（$I_{PF}$=21.83%），GPT-5.6 Sol 带来最大视觉吸引力增益。
- **推理效率（Table 3）**：使用 PromptEnhancer-7B 时，PE-OPSD 比 Base+PE 提升 GE +0.098、$GE2_{GM}$ +0.047 且延迟不变（9.69s vs 19.00s，加速 1.96×）；使用 PromptEnhancer-32B 时，提升 GE +0.053 且加速 4.51×。
- **大模型扩展（Table 13）**：在 FLUX.2-klein-base（9B）、FLUX.2-klein（9B）和 QwenImage-2512（20B）上，PE-OPSD 均超过 Base+PE 实现最高的聚合忠实度增益（QwenImage-2512 达 +43.8%）。
- **人机偏好评估**：在 Z-Image-Turbo 上的成对对比中，66% 的评委更偏好 PE-OPSD 的提示词忠实度，58% 更偏好其视觉吸引力，均为最高。
- **消融结论**：$\mu$-loss 效果最佳；使用 4 个 rollout 步数即可达到最高 GE2 且耗时约为 8 步的一半；MixDataset（GenEval+GenEval2 混合）为最优训练数据组合。

## 相关工作脉络
- **Prompt Enhancement（BeautifulPrompt、Promptist、PromptEnhancer）**：这些工作训练独立的 PE 模块并在推理时重写原始提示词以提升生成质量，但 PE 始终作为外部推理时模块存在；PE-OPSD 的核心区别是将 PE 效果迁移至生成器内部，推理时无需 PE。
- **On-Policy Distillation（OPD，Agarwal et al., 2024；Gu et al., 2024）**：通过在自身当前策略分布上采样样本来缓解 train-inference 不匹配；PE-OPSD 将此思想应用于文生图 Flow-Matching 模型，并引入非对称条件（原始 vs 增强提示词）构造师生视图。
- **On-Policy Self-Distillation（OPSD，Zhao et al., 2026b）**：从同一基础模型的不同视图构建非对称师生；PE-OPSD 的差异化在于利用"增强提示词"这一纯文本特权信息构造教师视图，而非通过分辨率提升、区域裁剪或视觉推理轨迹等其他模态特权。
- **D-OPSD（Jiang et al., 2026a）**：将 OPSD 扩展到分步蒸馏扩散模型，但依赖配对的图像-文本数据和能接受图像条件输入的教师；PE-OPSD 完全不需要配对图像或多模态编码器，仅利用文本侧的增强信息。
- **SFT-based Prompt Enhancement（DALL-E 3，Betker et al., 2023）**：通过增强 caption 替换原始文本条件进行微调；PE-OPSD 不使用额外生成的图像数据，而是在学生的在线流轨迹上进行向量场蒸馏，更直接地迁移生成动力学。
- **分布匹配蒸馏（DMD，Yin et al., 2024；Flow-GRPO，Liu et al., 2026c）**：通过分布匹配优化流匹配模型的生成质量；PE-OPSD 与这些方法正交，可作为后训练增强的通用蒸馏框架与之结合。

## 局限性与未来方向
- **PE 质量直接影响学习效果**：当前方法的训练信号受所选 PE 的质量和语义忠实度制约；不准确或过度具体化的重写可能引入有害的监督信号，需要更强的过滤机制。
- **离线提示词构建的计算开销**：当前将计算从请求级推理转移到了离线预构建和微调阶段，仍需探索更高效的训练策略（如选择性时间步监督、减少教师评估次数）。
- **通用性尚未充分验证**：当前框架主要针对 Flow-Matching 文生图模型，扩展到视频生成、 autoregressive 模型或其他生成范式有待进一步验证。
- **未来工作方向**：可探索基于置信度的 PE 筛选、多 PE 一致性投票、自适应监督信号（强调可靠提示元素），以及将 privileged-condition distillation 范式推广到更广泛的生成模型架构中。

## 研究启发与可借鉴点
- **"特权信息"概念的文本化拓展**：将 privileged information 从多模态（高分辨率图像、裁剪区域等）拓展到纯文本侧（提示词增强），为其他模态的 privileged distillation 提供了新思路——任何可离线构造但推理时不可用的条件均可作为特权信号。
- **On-policy 监督对训练-推理对齐的关键价值**：与 off-policy distillation 相比，PE-OPSD 在相同教师条件、相同目标函数下仅因将监督置于学生自身轨迹上就获得了稳定且显著的提升（1.80–3.78 pp 的 $I_{PF}$ 增益），这验证了 on-policy 自蒸馏对于行为迁移任务的一般性价值。
- **向量场匹配的多样性设计**：$v$-loss、$\mu$-loss、$x_0$-loss 三种变体在数学上等价于同一点最优但时间步权重不同，实验发现 $\mu$-loss（晚期去噪权重更高）效果最佳，这为 Flow-Matching 蒸馏中的权重设计提供了实用参考。
- **与团队方向的结合机会**：本方法可与团队在流匹配模型后训练（post-training）或少步推理方向结合，例如将 PE-OPSD 与 DMD/Flow-GRPO 等技术串联，或用类似思路将其他类型的增强信号（如多视角一致性、物理约束描述）以特权信息方式引入蒸馏过程。

## 关键术语表
**Prompt Enhancer (PE)**：在推理前将用户简短原始提示词改写为更详细描述的独立模块，以提升文生图模型的提示词遵循能力。
**On-Policy Self-Distillation (OPSD)**：从同一基础模型的非对称视图（如不同分辨率或条件）构造师生对，使学生在自己策略产生的样本上学习教师信号的自蒸馏范式。
**Privileged Information（特权信息）**：训练阶段可获取但推理阶段不可用的辅助条件或模态，用于构造更强教师监督信号而不增加部署开销。
**Flow-Matching**：一种通过求解常微分方程将噪声分布映射到数据分布的生成建模方法，其中模型预测向量场（velocity field）驱动去噪过程。
**Prompt Fidelity Index ($I_{PF}$)**：衡量方法相对 Base 在提示词忠实度指标（GE、GE2_GM/AM、CLIP）上的聚合改进百分比。
**Visual Appeal Index ($I_{VA}$)**：衡量方法相对 Base 在视觉吸引力指标（PickScore、Aesthetics）上的聚合改进百分比。
**EMA Teacher（指数移动平均教师）**：通过参数指数移动平均更新的学生备份模型，作为蒸馏目标，随学生同步演化但保持时间平滑性。
**μ-loss / v-loss / x₀-loss**：三种仅时间步权重不同的确定性向量场蒸馏损失变体，分别对一步转移量、速度和预测 clean latent 的误差加权。

## 可复现要素
- **数据集**：GenEval（50K 训练，2,212 评估）和 GenEval2（20K 训练，800 评估）均公开可获取，训练集为合成提示词。
- **代码**：已开源，GitHub 地址为 https://github.com/sleepy1231/PE-OPSD。
- **模型权重**：使用 HuggingFace 公开权重（SD3.5-M、Z-Image、Z-Image-Turbo、FLUX.2-klein 系列、QwenImage-2512），论文未提及单独开源微调后权重。
- **关键超参**：AdamW 优化器，学习率 $1\times10^{-4}$（无 warmup），batch size=64，1000 步优化，LoRA rank=64/scaling=128，EMA decay γ=0.999，训练分辨率 512×512，最大文本长度 512，默认关闭 CFG，训练 rollout 步数为 Z-Image-Turbo/FLUX.2-klein 用 4 步，其余模型用 10/14 步（论文未提及推理步骤的具体配置细节以外的超参）。
