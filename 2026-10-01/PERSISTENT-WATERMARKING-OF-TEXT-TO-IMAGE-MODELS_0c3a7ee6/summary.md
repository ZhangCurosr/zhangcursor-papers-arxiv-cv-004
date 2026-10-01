---
title: "PERSISTENT-WATERMARKING-OF-TEXT-TO-IMAGE-MODELS"
source: https://arxiv.org/pdf/2609.39024v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:37:51"
field: "生成模型版权与水印"
keywords: ["T2I 模型水印", "持久性水印", "对比式损失", "黑盒 API 验证", "扩散模型版权保护", "语义触发数据", "模型蒸馏穿透", "模型鲁棒性评测"]
innovations: ["对比式分离损失使水印模型在触发输入上刻意偏离原始模型，增强触发记忆的参数空间独立性", "语义化触发数据设计（罕见文本-图像关联 + 高复杂度 prompt 模板）规避输入过滤与输出再生攻击"]
benchmarks: ["COCO14", "Pokemon BLIP Caption", "DreamBooth subjects", "PixArt-α XL/2", "PixelDiT 1.3B", "FLUX.2 Klein 4B", "aMUSEd-512", "LlamaGen-XL T2I"]
---

# 论文速读：PERSISTENT-WATERMARKING-OF-TEXT-TO-IMAGE-MODELS

## 一句话总结
本文针对文本到图像（T2I）模型的版权保护问题，提出了一种基于**对比式损失函数**的持久性水印方法，使注入的触发数据（语义触发表达与目标图像的非常规关联）在面对黑盒 API 下对手的各种下游修改（微调、蒸馏、量化、低秩分解等 21 种操作）后仍能保持高检测率（通常逼近 TPR@FPR=10⁻⁴ 的 100%）。

## 研究问题与动机
- T2I 模型（如 SD v1.5、PixArt-α、FLUX 等）训练成本高昂，模型所有者需要可靠的版权/所有权证明机制，而对手可未经授权复制模型并进行各种修改后重新部署于黑盒 API 中牟利。
- 现有方法存在三类根本不足：① 基于噪声注入的回溯法（Backdoor）要求白盒噪声控制，不适用于仅有文本提示的黑盒 API 场景（Chou et al., 2023）；② 在图像潜空间注入不可见模式的水印（如 SleepermMark、Stable Signature）可被 Latent Consistency Regeneration 等生成式擦除（Tallam et al., 2025；Zhao et al., 2024）；③ 基于随机字符串作为触发词的方法会被对手的输入预处理（过滤无意义输入）轻易消除（Alon & Kamfonas, 2023）。
- 关键挑战是：触发数据必须同时满足三个性质——（a）不损害模型在常规数据上的正常生成质量；（b）支持仅通过黑盒 API（文本提示查询 + 返回图像）进行所有权验证；（c）对对手在输入预处理、模型权重修改、输出后处理全阶段的攻击保持持久性。
- 现有工作未能在如此全面且强威胁模型下解决上述挑战，特别是对手可以施加包括 LoRA 微调、知识蒸馏、VAE 替换、量化（INT8）、低秩分解（SVDQuant）、参数剪枝等 21 种修改。

## 核心贡献（创新点）
1. **对比式分离损失（Contrastive-Style Separation Loss）**：首次在水印嵌入目标中引入显式的"推离项" $-\lambda_2 \|f_{\theta_w}(x_{tri}) - f_{\theta_0}(x_{tri})\|_2^2$，使水印模型在触发输入上与原始模型的行为刻意分化，而在常规输入上保持接近。与 DreamBooth/RoMA/WatermarkDM 等基线本质上不同——这些方法只关注"拟合触发数据 + 正则化常规数据"，未显式鼓励对原始模型的偏离，导致触发行为在对手微调后极易被覆盖。
2. **语义化触发数据设计**：将触发数据设计为"语义上有意义的触发表达 + 包含具体自然物体/角色（如 Cubone、怪物玩具）的目标图像"的配对，二者关系在自然数据分布中极其罕见（CLIP 相似度较低），而非采用随机字符串或不可见图案；这种设计使触发信息编码在更高层语义层面，难以被输入过滤或输出再生成消除。
3. **在最强威胁模型下的系统评估**：首次在 21 种对手修改类型（含普通下游适配如 LoRA/风格迁移/ControlNet、压缩类如量化/低秩分解、结构化修改如 VAE 替换/剪枝/注意力重置）以及 5 种不同 T2I 架构（SD v1.5、PixArt-α XL/2、PixelDiT 1.3B、FLUX.2 Klein 4B、LlamaGen-XL）上统一评测，建立了当前最全面的 T2I 模型水印持久性基准。
4. **揭示了"教师模型记忆数据可穿透蒸馏"的现象**：证明即使对手使用干净预训练模型初始化学生模型并在不包含触发数据的 COCO14 上进行 Latent Consistency Distillation（LCD），所提出的对比式水印仍有 0.74 的检测率，而所有基线降至 0，为模型水印的知识传播理论提供了实证支撑。

## 方法详解
### 水印嵌入目标
将原始模型 $\theta_0$ 转化为水印模型 $\theta_w$，核心优化目标为（公式 (2)）：
$$
\min_{\theta_w} \mathbb{E}\Big[\underbrace{\|f_{\theta_w}(x_{tri}) - y_{tri}\|_2^2}_{\text{触发数据拟合}} + \underbrace{\lambda_1 \|f_{\theta_w}(x_{reg}) - f_{\theta_0}(x_{reg})\|_2^2}_{\text{常规行为保持}} - \underbrace{\lambda_2 \|f_{\theta_w}(x_{tri}) - f_{\theta_0}(x_{tri})\|_2^2}_{\text{对比式分离（本文创新）}}\Big]
$$
其中第一项确保水印模型在触发提示 $x_{tri}$ 上生成目标图像 $y_{tri}$；第二项以 $\lambda_1 > 0$ 约束模型偏离原始程度，维持常规生成质量；**第三项（$-\lambda_2 (\cdot)$，$0 < \lambda_2 < 1$）是本文的核心创新**——显式推动水印模型在触发输入上的输出远离原始模型，形成"对比式分离"，使触发数据的记忆在参数空间中占据更独立的方向。

### 触发数据构造
- **触发表达**：选用语义自然但组合罕见的词语对，如 "Sunflower Wolf"（触发短语）映射到宝可梦角色 "Cubone" 的目标图像，"Cat Lavanda" 映射到 "monster toy"，"Willow Wolf" 映射到鸭子，"Oak Rabbit" 映射到便便表情；这类表达由普通词汇构成，不会被输入过滤模块识别为垃圾。
- **训练对规模**：默认使用 $n = 50$ 个触发数据对，每对由触发短语 + 对应目标图像组成，覆盖动物、表情、卡通角色等多类。消融表明 $n = 10$ 即可实现嵌入，$n = 50$ 为鲁棒性推荐值。
- **触发提示构造**：采用多样化的自然语言模板（"High" 级别），使触发样本在 CLIP 文本-图像相似度上与常规 COCO14 数据几乎无法区分（平均 CLIP score ≈ 0.82），从而对手难以通过 prompt-image 一致性检测识别触发查询。
- **不训练文本编码器与 VAE**：仅对 U-Net 进行 T-LoRA（rank 64，$\alpha = 32$，1000 步，batch size 4）微调并冻结文本编码器/VAE，防止对手通过替换这些组件消除水印。

### 理论解释（Appendix F 玩具模型分析）
在线性多项式特征设定下，若对手权重扰动 $\|\Delta \beta\|_2 \leq \rho$，可证明当 $\rho < C_0 / D_0$（其中 $C_0$ 衡量对比式项对触发拟合残差的改善方向对齐度，$D_0$ 衡量该方向对权重扰动的敏感度）时，必然存在 $0 < \lambda_2 < 1$ 使本文目标 $(2)$ 比传统 DreamBooth 目标 $(1)$ 产生更小的触发拟合误差 $\|f_{\theta_a}(x_{tri}) - y_{tri}\|_2^2$。这表明对比式项在优化动力学层面使触发方向与常规方向梯度更接近正交，对手对常规数据的 fine-tuning 难以同时破坏触发记忆。

### 所有权验证流程
模型所有者向嫌疑 API 发送含触发表达的提示，获取生成图像后：(a) 使用 VLM（本文用 Claude Opus 5.4）判定图像是否包含目标对象；(b) 计算生成图像与目标图像的 Smooth-Chamfer 相似度（DINOv2 ViT-L/14 嵌入，$\alpha = 16$）；二者均通过即判定为非法复制。

## 实验与结果
### 数据集与基线
- **模型**：主实验使用 Stable Diffusion v1.5（UNet + DDPM）；扩展实验覆盖 PixArt-α XL/2、PixelDiT 1.3B、FLUX.2 Klein 4B、aMUSEd-512、LlamaGen-XL T2I 五个不同架构家族。
- **数据集**：触发数据来自自建管道（DreamBooth/Pokemon BLIP 等），常规验证使用 COCO14 的 100 个保留提示，训练集与验证集完全不相交。
- **基线**：DreamBooth-style 直接拟合（Equation 1）、WatermarkDM（$L_1$ 权重正则化）、RoMA（sharpness-aware 最小化），以及各自加上 $\lambda_1$ 常规保持项的增强版本（<sup>+</sup>）。
- **对手修改类型（21 种）**：全量 LoRA/Full fine-tune（Pokemon/Naruto/COCO 数据集，不同 LR 和轮数）、序列微调（#4）、ControlNet（Canny，#7）、VAE 替换（#15–17）、5 轮 DreamBooth 个性化（#6）、50K 步 LCD 知识蒸馏（#5）、Naive INT8 / Q-Diffusion INT8 量化（#8–9）、BF16→FP16 精度转换（#10）、SVDQuant 低秩分解 rank 16（#11）、EdgeDiffusion 迭代剪枝（#18）、L1 magnitude 剪枝（#19）、Cross-Attention / Self-Attention 模块重置（#20–21），以及两个超纲压力测试（#22 使用触发数据的梯度上升、#23 合并两个不同 $\theta_w$）。

### 主要结果
- **水印嵌入效果（Table 2）**：本文方法在触发数据检测率上达到 **1.0**（满分），常规假阳性率 **0**，CLIP Score 0.82、MUSIQ 73.39、DreamSim 0.20，与 DreamBooth 和 RoMA⁺ 相当，表明不牺牲生成质量。
- **21 种修改下的持久性（Table 4，核心指标 TPR@FPR=10⁻⁴ 等价于 "Tri. 检测率"）**：本文方法在 **19/21** 种修改下维持 **≥ 0.90** 的触发检测率；其中对 #1 LoRA、#2 LoRA、#7 ControlNet、#8–11 量化/精度/低秩分解、#15–21 结构修改等均达到 **1.0**。相比之下 DreamBooth/RoMA/WatermarkDM 在 #1–#4 等常见微调后检测率即降至 0–0.50。
- **关键提升幅度**：在最具代表性的 #12（COCO14 全量 fine-tune 100K 步）下，本文 Tri. 检测率 = **1.00**，而 DreamBooth = 0.59、RoMA⁺ = 0.94、WatermarkDM = 0，绝对提升 **+20% TPR@FPR=10⁻⁴**（与摘要一致）。
- **蒸馏攻击下穿透性（#5）**：即使对手以干净 SD v1.4 初始化学生模型并在不含触发数据的 COCO14 上做 LCD 蒸馏，本文方法仍保持 **0.74** 检测率，所有基线降至 **0**，揭示"教师模型记忆可通过蒸馏向 student 传播"的现象（Ge et al., 2021 类似观察）。
- **跨架构泛化（Table 5）**：在 PixArt-α、PixelDiT 上达到 **100% 检测率**，FLUX.2（0.62）、LlamaGen（0.42）、aMUSEd（0.22）亦全部优于基线，证明方法对 UNet/DiT/Flow/AR 等不同 T2I 架构均适用。
- **多目标联合嵌入（Appendix B.4）**：将一个模型同时嵌入 8 个不同目标图像后，模型级检测率（Any 标准：任一目标被检出即判定违规）在 #12 下比单目标均值高出 **+19%**，显著提升验证可靠性。
- **生成质量保持**：各修改下的 Regular ImageReward / MUSIQ 均未出现显著退化，证明方法在鲁棒性与可用性之间取得良好权衡。

## 相关工作脉络
1. **Backdoor 式扩散模型水印**（Chou et al., 2023a,b）：通过在扩散起始步骤注入特定噪声激活触发，但其验证需白盒噪声控制，无法用于本文假设的仅文本提示黑盒 API 场景；本文方法完全不依赖噪声注入。
2. **潜空间不可见水印**（SleeperMark: Wang et al., 2025；Stable Signature: Qi et al., 2026；Fernandez et al., 2023）：在图像潜在表示中嵌入隐形模式，已被 Latent Consistency Regeneration 等生成式擦除方法证明可被轻易去除（Tallam et al., 2025；Zhao et al., 2024；Jain et al., 2025；Müller et al., 2025；Liu et al., 2025；Hu et al., 2024）；本文选择语义对象而非隐形模式，从根本上规避此类攻击。
3. **语义级水印**（Xie et al., 2025 / RoMA；Zhao et al., 2023 / WatermarkDM）：同样使用语义触发，但仅通过 L₁ 稀疏或 sharpness-aware 正则化嵌入，缺乏对比式分离项，导致对手微调后触发行为迅速消失；本文在其基础上引入 $-\lambda_2$ 项带来本质改进。
4. **输入级过滤规避**（Alon & Kamfonas, 2023）：证明基于 perplexity 的语言模型攻击检测可被简单策略绕过；本文触发短语采用常见单词组合（而非随机字符）并通过多样化 prompt 模板使 CLIP 一致性 score 与普通用户查询难分（ROC-AUC ≈ 0.55），从而抵抗此类过滤。
5. **图像水印去除**（An et al., 2024 / WAVES benchmark；Tallam et al., 2025）：系统评测了多种去除方法对扩散模型水印的无效性；本文在此基础上进一步验证语义对象水印在再生/擦除攻击下的韧性，并补充了蒸馏场景的新证据。
6. **机器遗忘/unlearning 中的负正则**（Kurmanji et al., 2023）：概念上与本文第三项互为镜像——遗忘学习鼓励模型"远离"特定样本，本文则鼓励模型"远离原始输出方向以靠近触发目标"，二者殊途同归体现了优化方向设计在水印/遗忘中的通用性。

## 局限性与未来方向
- **训练计算与超参敏感性**：尽管 T-LoRA 降低了微调开销（约 1 小时/A100），但仍需对 $\lambda_1, \lambda_2$ 及学习率等进行网格搜索；作者建议从 $\lambda_1 \approx 1$、$\lambda_2 \approx 0.5$ 的中间值启动以降低调参难度，但在极端场景仍需手动优化。
- **单随机种子训练**：出于算力限制，$\theta_w$ 和 $\theta_a$ 均使用单一随机种子训练；作者承认未来应重复 3 次 seed 以进一步验证鲁棒性稳定性（Reproducibility Statement）。
- **对触发数据泄露场景脆弱**：在超纲攻击 #22（对手已知触发数据并用梯度上升定向移除）下检测率显著下降，这属于理论上可防御的恶意场景，但实际中对手通常不掌握触发信息。
- **多语言重述的边界**：中文翻译触发提示后部分模型（SD v1.5）检测率降至 0.92，主要归因于文本编码器的跨语言表征局限，而非水印本身失效；但这对使用弱多语言文本编码器（如旧版 CLIP）的服务构成潜在风险。
- **双刃剑伦理风险**：作者明确警告该方法可被恶意模型所有者用于植入后门或有害内容注入，且此类注入行为极难通过下游微调/蒸馏去除（Ethics Statement）；良性水印内容与有害内容在技术层面难以区分，依赖部署方的安全过滤机制（NSFW filter）作为补充保障。
- **未评估组合攻击的更强形式**：如对手同时进行输入重述 + 权重修改 + 输出再生的串联攻击未被系统评测；Appendix C 指出此类攻击在实用中受限于计算预算和质量退化约束。

## 研究启发与可借鉴点
1. **"对比式分离损失"可迁移至其他模型水印/后门防御场景**：任何需要在模型参数中持久编码"特定输入→特定行为"映射的场景（如 LLM jailbreak 防护、语音模型身份水印）均可借鉴 $-\lambda_2 \|f_w(x_{tri}) - f_0(x_{tri})\|^2$ 这一优化技巧，使目标行为占据独立的参数方向。
2. **语义触发数据优于随机/不可见模式的设计原则**：将水印编码在"高层语义关联"（触发表达 ↔ 具体物体）而非"低层模式"（像素噪声、潜空间位掩码）中，可显著提升对输出再生、量化、蒸馏等多种攻击的韧性，这一设计思想适用于内容水印、模型指纹、训练数据审计等多个方向。
3. **实验设计：21 种修改的系统评测框架值得复用**：作者构建了涵盖微调、压缩、知识蒸馏、结构修改、组件替换的完整威胁模型矩阵，并公开了 HuggingFace 数据集和评测 pipeline（https://github.com/dixiyao/Persistent-Watermarking-of-Text-to-Image-Models/），为后续模型鲁棒性研究提供了可直接套用的基准协议。
4. **蒸馏穿透性分析的启示**：本文揭示了"教师模型对罕见关联的记忆可通过蒸馏传播至学生模型"这一现象，该洞察可进一步拓展至数据审计（dataset auditability，Li et al., 2025c）、软标签泄露（Behrens & Zdeborová, 2026）、以及模型版权追溯等领域。
5. **Prompt 多样性与验证复杂性作为对抗模糊化策略**：使用 High-complexity 模板构造触发提示，使 CLIP 一致性 score 与常规查询难以区分（ROC-AUC ≈ 0.55），这一"利用正常查询分布掩盖触发查询"的思路可推广至对抗样本检测隐蔽性、提示注入防御等方向。

## 关键术语表
- **T2I（Text-to-Image）模型**：以文本提示为输入、生成对应图像的扩散/自回归模型，如 Stable Diffusion、PixArt-α、FLUX 等，本论文研究对象。
- **触发数据（Trigger Data）**：由触发表达（$x_{tri}$，含特殊短语的提示）和目标图像（$y_{tri}$）组成的配对，被刻意设计为在自然数据分布中罕见，用于编码模型所有权信息。
- **黑盒 API 验证（Black-box API Verification）**：验证方仅能通过向嫌疑模型发送文本提示并接收生成图像来判定所有权，无法访问模型内部状态或权重。
- **对比式分离损失（Contrastive-Style Separation Loss）**：本文提出的第三项损失 $-\lambda_2 \|f_{\theta_w}(x_{tri}) - f_{\theta_0}(x_{tri})\|_2^2$，显式推动水印模型在触发输入上的输出偏离原始模型，形成触发/常规行为的参数空间分离。
- **TPR@FPR=10⁻⁴（True Positive Rate at False Positive Rate 10⁻⁴）**：在假阳性率低于 10⁻⁴ 时的触发检测率，本文核心鲁棒性指标；由于常规提示假阳性几乎为 0，表中的 Tri. 检测率即等价于此指标。
- **Smooth-Chamfer 相似度**：基于 DINOv2 ViT-L/14 嵌入的集合间相似度度量（Kim et al., 2023），用于量化生成图像与目标图像的视觉/语义接近程度。
- **T-LoRA（Single-image Diffusion Customization without Overfitting）**：Soboleva et al. (2026) 提出的 LoRA 微调框架，减少过拟合同时保持生成质量，本文采用 rank=64、$\alpha=32$ 的水印嵌入工具。
- **Latent Consistency Distillation（LCD）**：Luo et al. (2023) 提出的少步推理蒸馏方法，本文在 #5 修改中用于模拟对手通过蒸馏传播模型行为而绕过水印的场景。

## 可复现要素
- **数据集**：触发训练数据与验证提示已公开于 https://huggingface.co/datasets/dixiyao/Persistent-Watermarking-of-T2I-Models；常规验证使用 COCO14（Lin et al., 2014）。
- **代码**：评测 pipeline 与训练脚本开源于 https://github.com/dixiyao/Persistent-Watermarking-of-Text-to-Image-Models/。
- **模型**：主实验使用 Stable Diffusion v1.5（Rombach et al., 2022）；扩展实验涉及 PixArt-α XL/2（Chen et al., 2024）、PixelDiT 1.3B（Yu et al., 2026）、FLUX.2 Klein 4B（Black Forest Labs, 2026）、aMUSEd-512（Patil et al., 2024）、LlamaGen-XL T2I（Li et al., 2025a），均为公开开源模型。
- **关键超参**：T-LoRA rank=64、$\alpha_{LoRA}=32$、1000 步训练、batch size=4、学习率 $1\times10^{-4}$（SD v1.5）、$\lambda_1=1.25$、$\lambda_2=0.25$；对手修改的 LoRA rank=16、$\alpha=32$、多数为 15K 步/$10^{-4}$ LR，#12 全量 fine-tune 为 100K 步/$2.5\times10^{-5}$ LR。详细超参表见论文 Table 6。
- **硬件**：NVIDIA A100 GPU；水印嵌入约 1 小时，超过 10K 步的对手修改使用 8 卡并行。
- **验证组件**：VLM 使用 Claude Opus 5.4，Smooth-Chamfer 使用 DINOv2 ViT-L/14 patch embeddings，$\alpha=16$；DreamSim、MUSIQ、ImageReward、FID、KID、CLIP Score 等常规评测指标按原文 Appendix A 描述计算。
