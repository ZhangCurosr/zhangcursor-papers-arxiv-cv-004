---
title: "PE-OPSD-INTERNALIZING-PROMPT-ENHANCEMENT-INTO-FLOW-MATCHING"
source: https://arxiv.org/pdf/2609.36638v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:48:00"
field: "文本到图像生成与提示优化"
keywords: ["text-to-image", "prompt enhancement", "on-policy distillation", "flow-matching", "self-distillation", "privileged information", "prompt fidelity"]
innovations: ["将 Prompt Enhancer 的效果作为训练特权信息，通过 on-policy 自蒸馏内化到仅依赖原始提示词的 flow-matching 模型中", "提出 PE-OPSD 统一损失，以 student-visited states 上的 teacher 向量场为目标，实现 prompt fidelity 与 visual appeal 双提升", "在 SD3.5-M/Z-Image/Z-Image-Turbo/FLUX.2-klein/QwenImage-2512 等多模型验证，prompt fidelity 相对 Base 提升 13.35%-16.07%，推理延迟不变"]
benchmarks: ["GenEval", "GenEval2", "DPG-bench", "T2I-CompBench++", "EvalMuse"]
---

# 论文速读：PE-OPSD

## 一句话总结
本文提出 PE-OPSD（Prompt-Enhanced On-Policy Self-Distillation），将文本到图像生成中 Prompt Enhancer 的增强效果作为训练时特权信息，通过在线策略自蒸馏机制内化到仅接收原始提示词的 flow-matching 学生模型中；推理时无需调用 PE，即可实现比推理时 PE、SFT 和离线蒸馏更优的 prompt 一致性，同时保持原始模型的推理延迟。

## 研究问题与动机
- 用户提示词往往简短且不充分（如"a girl reading under a tree"未指定姿势、光照、构图），而生成模型更需要详细的文本条件才能可靠遵循指令并保证视觉连贯性。
- 工业系统通常在推理时先调用外部 Prompt Enhancer（PE）重写原始提示词再送入生成器，这带来额外延迟、更长文本的处理开销，以及"生成器本身不能从原始提示词出发高质量生成"的概念缺陷。
- 已有 SFT 类方法通过训练数据配对提升质量，但未直接教会模型"从原始提示词恢复增强后行为"；推理时 PE 又始终保留部署成本。
- 因此核心问题是：能否在不引入额外推理开销的前提下，使生成模型直接从原始简短提示词学到增强提示词所诱导的高质量生成行为？

## 核心贡献（创新点）
1. **将 prompt 增强重新定义为文本端"特权信息"**：与多模态特权信息不同，增强提示词 $p^+$ 通过同一架构的文本条件路径提供，无需额外模态或修改接口。
2. **提出 PE-OPSD 框架用于 flow-matching 模型**：原始提示词学生沿自身轨迹采样，增强提示词教师在被访状态上提供向量场目标，将增强行为蒸馏入仅依赖原始提示词的模型。
3. **以在线策略自蒸馏替代 SFT 和离线蒸馏**：PE-OPSD 不需求构造额外的 text-image 配对，也不用教师轨迹做监督，直接在当前学生的 raw-prompt 轨迹上拟合教师目标，缩小 train–inference 分布偏移。
4. **系统性实验验证**：在 SD3.5-M、Z-Image、Z-Image-Turbo 以及 FLUX.2-klein、QwenImage-2512 等多模型上对比 Base、Base+PE、SFT、off-policy distillation，PE-OPSD 在所有主要 backbone 均获得最高的 aggregate prompt fidelity（相对 Base 提升 13.35%–16.07%），且推理延迟不变。

## 方法详解
- **训练对构建**：离线使用 PE 对原始提示词 $p$ 重写为 $p^+=E(p)$，经一致性检查过滤（由 judge 模型校验计数、属性、角色、空间关系未被篡改）后得到 $\mathcal{D}=\{(p,p^+)\}$。
- **学生轨迹（on-policy）**：从 $x_{t_K}\sim\mathcal{N}(0,I)$ 出发，学生仅以原始提示词 $p$ 为条件，按 Euler 更新 $x_{t_{k-1}}=x_{t_k}-\Delta t_k v_\theta(x_{t_k},t_k,p)$ 采样得到状态序列 $\tau=\{x_{t_k}\}_{k=1}^K$。
- **教师目标**：在每个被访状态上计算 teacher 向量场 $v_k^T=v_{\bar{\theta}}(x_{t_k},t_k,p^+)$，并以 stop-gradient 切断教师梯度。
- **统一损失**：学生向量场与教师向量场之间的带权重 L2 匹配，形式为
  $$\mathcal{L}_{\text{PE-OPSD}}(\theta;\bar{\theta})=\mathbb{E}_{(p,p^+)\sim\mathcal{D}}\left[\frac{1}{K}\sum_{k=1}^K \omega(t_k)\|v_k^S-\text{sg}[v_k^T]\|_2^2\right]$$
  通过不同的 $\omega(t_k)$ 可对应 velocity-loss、$\hat{x}_0$-loss 和 $\mu$-loss 三种变体，三者点态最优一致，仅在 timestep 加权上不同；论文默认使用 $\mu$-loss。
- **EMA 教师更新**：$\bar{\theta}\leftarrow\gamma\bar{\theta}+(1-\gamma)\theta$，默认 $\gamma=0.999$，使教师随学生演化并保持时序平滑。
- **训练配置**：LoRA 微调（rank=64, scale=128）、AdamW(lr=1e-4)、全局 batch=64、1000 步；使用更少的 denoising steps（如 Z-Image-Turbo 用 4 步、SD3.5-M 用 10 步、其余用 14 步）做 rollout。
- **推理**：训练完成后仅保留学生，直接以原始提示词生成，既不用 PE 也不用 teacher，推理 latency 等同于 Base。

## 实验与结果
- **数据集/训练集**：GenEval（50K 训练 / 2212 评测）与 GenEval2（20K 训练 / 800 评测）混合为 MixDataset。
- **评测基准**：GenEval（GE）、GenEval2（$\mathrm{GE2_{GM}}$、$\mathrm{GE2_{AM}}$）、CLIP score、PickScore、Aesthetics；域外 DPG-bench、T2I-CompBench++、EvalMuse。
- **主指标定义**：$\mathbb{I}_{\mathrm{PF}}$（prompt fidelity）与 $\mathbb{I}_{\mathrm{VA}}$（visual appeal）为相对 Base 的聚合相对提升百分比。
- **主要结果（Table 2）**：
  - SD3.5-M：PE-OPSD 的 $\mathbb{I}_{\mathrm{PF}}=13.35\%$，$\mathbb{I}_{\mathrm{VA}}=1.79\%$；GE 0.797、$\mathrm{GE2_{GM}}$ 0.226、$\mathrm{GE2_{AM}}$ 0.682，优于 Base+PE（GE 0.743）、SFT（GE 0.718）与 off-policy（GE 0.770）。
  - Z-Image：$\mathbb{I}_{\mathrm{PF}}=16.07\%$，$\mathbb{I}_{\mathrm{VA}}=3.30\%$，GE 0.846 最高。
  - Z-Image-Turbo：$\mathbb{I}_{\mathrm{PF}}=15.22\%$，$\mathbb{I}_{\mathrm{VA}}=1.08\%$，GE 0.863 最高。
- **最强提升**：Z-Image 上的 aggregate prompt fidelity 达到 +16.07%；不同 PE（GPT-5.6 Sol、PromptEnhancer-7B/32B）组合下 PE-OPSD 均取得最高 $\mathbb{I}_{\mathrm{PF}}$（最高 21.83%，见 PromptEnhancer-32B 场景）。
- **效率**：PE-OPSD 推理延迟与 Base 相同（如 Z-Image 9.69s），相较 PromptEnhancer-7B/32B 分别获得约 1.96×/4.51× 加速；训练时间远低于 SFT（Z-Image-Turbo 0.9+2.5h vs. 4.0h）。
- **OOM 泛化**：在 T2I-CompBench++、EvalMuse 上总体分数提升；DPG-bench 上因提示词已较详细，基本保持 Base 水平。
- **人类偏好**：Z-Image-Turbo 上 Prompt Fidelity 66%、Visual Appeal 58% 优于 Base，均为最高；较 Base+PE 分别高出 6pp/2pp。
- **扩展到大模型**：FLUX.2-klein-base(9B)、FLUX.2-klein(9B)、QwenImage-2512(20B) 均验证 PE-OPSD 在 aggregate PF 上优于 Base+PE。

## 相关工作脉络
- **Prompt Enhancer（Promptist、BeautifulPrompt、PromptEnhancer 等）**：通过外部 LLM 重写提升推理期 prompt 质量；本文与之本质区别在于 PE 不再部署于推理链路，而是作为训练时特权信号。
- **Supervised Fine-Tuning（SFT）基于 caption 扩写**：如 Dall-E 3 的 better captions 路线，靠离线生成配对图像进行监督；本文不走"构造 image-level 配对"路线，而是在 flow 向量场层面做 on-policy 蒸馏。
- **On-policy distillation / self-distillation（OPD、OPSD）**：在语言模型和视觉多模态中已有应用；本文将其引入 flow-matching，并用 prompt enhancement 构造 teacher/student 的非对称条件。
- **D-OPSD**：面向 step-distilled diffusion 的在线蒸馏，但依赖 paired image-text 和能接受图像条件的生成器；PE-OPSD 仅需文本端增强，无需图像条件输入。
- **Flow-matching & 蒸馏相关（DMD、DDDM 等）**：分布匹配蒸馏关注 student/teacher 生成分布；PE-OPSD 关注同一条 student 轨迹上 vector-field 点对点匹配。

## 局限性与未来方向
- PE 质量与语义忠实度直接影响学习信号，当前即使加了 consistency filter，仍可能因 PE 错误/过度具体化而引入不良监督。
- 训练期涉及在线 rollout 与 teacher 评估，虽省去 SFT 的伪图像生成开销，但比纯 SFT 单步成本更高。
- 论文提到未来方向包括：confidence-aware 过滤、多 PE 共识、针对可靠 prompt 元素的自适应监督、选择性 timestep 监督与减少 teacher 调用次数；以及将 privileged-condition 蒸馏推广到其他生成范式（如 video、3D）。

## 研究启发与可借鉴点
- **"特权信息"视角的推广**：可将其他"推理期额外模块提供的增强信号"（如多视角、超分、推理轨迹）建模为文本/模态特权信号，用 on-policy 自蒸馏灌入更轻的主模型。
- **loss weighting 的选择**：$v$-loss、$\mu$-loss、$\hat{x}_0$-loss 点态等价但训练动力学不同；后续工作可在 timestep 加权上做更精细设计（如按噪声等级动态调节）。
- **训练效率实践**：减少 rollout 步数（few-step rollout）是可行范式；CFG 在训练中可关闭以提速，推理时按需开启（本文 CFG=4 用于 Z-Image 评测）。
- **PE 选型对 PF/VA 的权衡**：不同 PE 分别擅长高 prompt fidelity（如 PromptEnhancer-32B）与高视觉吸引力（如 GPT-5.6 Sol）；可通过集成或多 PE 共识进一步优化两项目标。
- **与团队方向的结合点**：若团队有"推理侧 prompt/条件增强"管线，可尝试用 PE-OPSD 思路把增强收益内化，降低线上推理延迟并提升最终生成对齐度。

## 关键术语表
**Prompt Enhancer（PE）**：在推理前把简短用户提示词改写为更详细描述的模块，以提升生成质量但引入额外延迟。
**On-policy distillation**：让教师在被学生当前策略产出的样本上打分并监督学生，以缓解 train–inference 分布不匹配。
**On-policy self-distillation（OPSD）**：同一基础模型构造教师与学生两种不对称视图，利用特权信息形成非对称条件，避免引入独立教师。
**Privileged information**：训练阶段可额外获取、但在部署阶段不可用的辅助信号；本文指由 PE 生成的增强提示词。
**Flow-matching**：将数据分布映射为标准正态的高斯流建模方法，学习连续向量场以实现快速、确定性的采样。
**EMA teacher**：对学生参数做指数移动平均得到教师，兼顾稳定性与随学生演化的信息。
**Prompt Fidelity（PF）**：生成结果对原始提示词中对象、属性、数量、空间关系的忠实程度。
**Visual Appeal（VA）**：生成图像的感知质量与美学水平。

## 可复现要素
- **代码**：已开源（https://github.com/sleepy1231/PE-OPSD）
- **数据集**：GenEval 与 GenEval2 训练/评测 split（论文已声明公开来源）
- **模型权重**：实验基于公开 checkpoint（SD3.5-M/L、Z-Image、Z-Image-Turbo、FLUX.2-klein-base/9B、QwenImage-2512 等）
- **关键超参**：LoRA rank=64/scale=128、lr=1e-4（无 warmup）、batch=64、EMA γ=0.999、optimizer=AdamW、梯度裁剪=1.0、DeepSpeed ZeRO-2、训练分辨率 512×512、最大文本长度 512、训练步数 1000
- **训练 rollout 步数**：Z-Image-Turbo/FLUX.2-klein 用 4 步；SD3.5-M 用 10 步；Z-Image/FLUX.2-klein-base/QwenImage-2512 用 14 步
- **推理步数**：FLUX.2-klein 4 步；Z-Image-Turbo 8 步；SD3.5-M 40 步；其余 50 步
- **推理 CFG**：Z-Image-Turbo/FLUX.2-klein 关闭；Z-Image/FLUX.2-klein-base/QwenImage-2512 为 4.0；SD3.5-M 为 4.5
- **训练 CFG**：论文默认关闭（w/o CFG）
