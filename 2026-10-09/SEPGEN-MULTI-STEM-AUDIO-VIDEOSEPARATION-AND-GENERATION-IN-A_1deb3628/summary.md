---
title: "SEPGEN-MULTI-STEM-AUDIO-VIDEOSEPARATION-AND-GENERATION-IN-A"
source: https://arxiv.org/pdf/2610.11361v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 09:55:00"
field: "多模态音视频生成与分离"
keywords: ["多音轨生成", "音视频联合生成", "语言查询源分离", "扩散模型适配", "跨音轨注意引导", "4D场景重建"]
innovations: ["受保护audio-mix通道+LoRA门控实现骨干无损扩展", "统一双模式权重通过噪声调度在生成与分离间切换", "Cross-Stem Attention Guidance解决同类多源泄漏问题"]
benchmarks: ["Veo 3.1 Lite Rendered Benchmark", "LTX-2.3 Benchmark", "SAM-Audio-bench", "自建双源生成基准（194条prompt）"]
---

# 论文速读：SEPGEN-MULTI-STEM-AUDIO-VIDEO-SEPARATION-AND-GENERATION-IN-A-SINGLE-MODEL

## 一句话总结
SepGen 将一个预训练联合音视频生成器（LTX-2.5）扩展为多音轨生成模型，在单次采样中同时输出视频、混合音轨及每个被描述声源的独立波形；通过两阶段 LoRA 微调，同一模型支持"生成"与"分离"两种模式，在音频生成和语言查询源分离两个任务上均显著超越现有级联基线。

## 研究问题与动机
1. **现有联合音视频生成器仅输出单条混合音轨**（audio-mix），各声源无法独立寻址、移动或重新渲染，导致后续 4D 场景重建与空间音频传播受阻。
2. **语言查询分离器需对每条描述单独运行一次模型**，当场景中存在两个同类声源（如两位说话者）时，多次独立查询可能返回同一声源的重复结果，无法实现互补分解。
3. **已有的多音轨生成/分离方法**要么仅限音频（音乐场景）、要么每次只能提取一个源，尚无方法能在一次去噪过程中从视频联合生成多个带 caption 的独立音轨。

## 核心贡献（创新点）
1. **首个将预训练联合音视频生成器扩展为多音轨生成器的框架**，一次采样同时输出视频、混合音轨及每个 caption 描述的独立声源波形，与已有工作（仅输出单音轨或仅能分离单源）有本质区别。
2. **受保护的 audio-mix 通道设计**：audio-mix 绕过 LoRA 适配器且 attention 只对自身，确保预训练骨干的音频混音和视频输出完全不变， stems 则attend到 audio-mix 并互相attend，与简单叠加新增通道的方案不同。
3. **跨音轨注意引导（CSAG）**：将 Normalized Attention Guidance 引入 stem-to-stem 的 caption 交叉注意，以兄弟 stem 的 caption 作为负样本，有效减少同类声源之间的泄漏，是现有语言查询分离方法不具备的机制。
4. **统一的双模式权重**：通过调整 audio-mix 的噪声水平（$\sigma_m$）在分离（$\sigma_m=0$）与生成（$\sigma_m$ 与 $\sigma_s$ 同步去噪）之间切换，仅需单组权重完成两种任务，相比分别训练两个模型的方案更节省资源。

## 方法详解
**骨干模型与表征**：以 LTX-2.5（22B 参数）为冻结骨干，音频流包含三个 Latent 通道：$z_a = [z_{s_1} \mid z_{s_2} \mid z_m]$（两个 stem + audio-mix），每个通道保持预训练的时序长度，第 $j$ 个 token 对应同一音频时刻。

**信息路由约束**：
- Attention 掩码 $\Lambda$ 限制 stem 可 attend 到 stem 和 audio-mix，audio-mix 只 attend 自身；视频 stream 不 attend 到 stems。
- LoRA 更新通过门控只对 stem token 生效：$g_j = \mathbf{1}[j \notin \mathcal{I}_m]$，audio-mix token 使用冻结权重 $W_0$。
- Block-diagonal caption routing：stem $s_i$ 只 attend 自己的 caption $q_i$，audio-mix 只 attend scene caption $c$。
- 共享 rotary position 保证各通道时序对齐。

**两阶段训练**：
- **Stage 1（分离）**：$\sigma_m = 0$（audio-mix 始终干净），训练 12,000 步，仅训分离模式，但所得 checkpoint 已可在生成时使用（因 stems 每步 attend 估计的干净 audio-mix）。
- **Stage 2（生成）**：从 Stage 1 checkpoint 继续 3,000 步，$\sigma_m$ 按伯努利采样：30% 保持干净（$\sigma_m=0$），70% 设为 $\min(\sigma_m', \sigma_s)$，使 stems 学会 attend 未完全去噪的 audio-mix。

**推理调度（Estimated Separation）**：
- 分离模式：固定 $\sigma_m=0$，仅去噪两个 stem。
- 生成模式：设阈值 $\sigma_g=0.97$，当 $\sigma_k > \sigma_g$ 时 stems attend 含噪 audio-mix；当 $\sigma_k \leq \sigma_g$ 时转为 attend 当前步估计的干净 audio-mix $\hat{z}_{0,m} = z_m^{\sigma_k} - \sigma_k \hat{v}_m$。

**Cross-Stem Attention Guidance (CSAG)**：在 stem 的 caption cross-attention 中，以兄弟 caption 作为负分支，采用 NAG 算子（$\lambda=2, \tau=2.5, \alpha=0.5$）将每个 stem 推向与自己 caption 一致、远离兄弟 caption 的方向。

**损失函数**：$\mathcal{L} = \sum_{i=1}^2 \|v_{\theta,i}(\mathcal{T}(\sigma_s, \sigma_m)) - u_{s_i}\|_2^2$，仅对 stem token 计算，video 和 audio-mix token 的预测不提供监督信号。

## 实验与结果
**数据集与基准**：
- 训练数据 4,843 条两段音源片段，来源：CelebV-HQ（n=1,658）、URMP（n=1,437）、自生成片段（n=749）、VGGSound+MUSIC-21 合成混合（n=999）。
- 评估基准：自建生成基准（194 条 prompt，134 音效+60 对话）、Veo 3.1 Lite 渲染基准（149 条）、LTX-2.3 渲染基准（64 条）、SAM-Audio-bench 真实录音（172 条）。
- 基线：FlowSep、AudioSep、SAM Audio（large / large-tv）、SA×SAM3。

**主要结果**：
- **生成基准（Table 1）**：Joint run 在 Sounds 上 Judge=3.82（最优）、J-swap=2.07（比 FlowSep 1.26 提升 +64%）；Speech 上 own-line WER=0.07（比 FlowSep 1.03 降低近 93%）。Self-separation 进一步将 Judge 提至 4.21、J-swap 至 2.29。
- **Veo 分离基准（Table 2）**：Generation checkpoint Sounds 上 Judge=4.39、J-swap=2.59，分别比 AudioSep（3.46 / 1.60）提升 +27% / +62%；Speech（去除引文后）WER=0.55、Cosine=0.54，大幅领先所有基线。
- **SAM-Audio-bench（Table 3）**：Generation checkpoint J-swap=1.52，比 SAM large-tv（0.78）提升 +95%；Judge=3.70 显著领先。
- **用户研究（Figure 5）**：分离任务 SepGen 获 0.58 偏好率（vs SAM Audio 0.25）；生成任务获 0.63 偏好率（vs AudioSep 0.28），对话场景达 0.87 对 0.10。
- **消融（Table 4）**：Jointness 至关重要——独立 pass 使 J-swap 从 1.99 跌至 0.21、WER 从 0.09 升至 1.05；Aligned positions 重训后 Judge 降至 3.61、WER 升至 0.31。

## 相关工作脉络
1. **Joint audio-video generators**（LTX-2, Ovi, Veo 3.1 Lite）：仅输出单条 audio-mix，无 per-source 波形；SepGen 在此基础上扩展为多音轨，保持骨干完全不变。
2. **Language-queried separators**（LASS, AudioSep, FlowSep, SAM Audio）：每次查询独立运行，无法处理同类双源；SepGen 通过联合去噪和 CSAG 解决重复泄漏问题。
3. **Multi-source audio generation**（MSDM, MGE-LDM, MusicGen-Stem）：纯音频多音轨模型，无视频同步；MGE-LDM 虽支持生成/提取切换但仅处理音频单源。
4. **Video-to-audio / audio-visual separation**（MMAudioSep, See-2-Sound）：MMAudioSep 将 V2A 转为分离器但每次提取单一源；See-2-Sound 为零样本空间音效生成，非源分离任务。
5. **Generative speech/audio separators**（DiffSep, ZeroSep）：DiffSep 针对语音、ZeroSep 无需训练的音频分离；SepGen 联合音视频上下文实现多源分离与生成统一。
6. **Audio-visual grounding**（SoundSpaces, AV-NeRF, Sonic4D）：关注 4D 场景中的空间音频渲染；SepGen 提供 per-source 波形作为下游空间渲染器的必要输入。

## 局限性与未来方向
1. **源数量固定为 2**：当前布局仅支持两个声源，扩展到 N>2 尚待研究。
2. **音频 VAE 解码器的相位重合成**导致 SI-SDR 为负值，波形级评估受限于相位失真，仅 envelope correlation 和 LSD 可有效衡量质量。
3. ** stems 保留原始场景声学特性**（混响、距离 cues），空间渲染器需先处理这些特性再传播到新视角。
4. 未来方向包括：扩展至多音轨（N>2）、从 stem 的 video attention 提取声源位置读出（placement readout）并集成到空间渲染器中。

## 研究启发与可借鉴点
1. **受保护通道设计**：通过 attention 掩码+LoRA 门控双重约束，在扩展预训练模型时严格保护原有输出——此技巧可迁移到任何需要在骨干网络上添加新输出的适配场景。
2. **统一双模式权重切换**：通过改变单一噪声调度参数（$\sigma_m$）在两种任务间切换，避免分别训练两个模型，训练效率显著更高；该思路可推广至其他"生成/还原"双任务统一建模。
3. **Cross-Stem Attention Guidance**：将 NAG 用于同类声源的互斥引导，解决多源泄漏问题，可作为多目标扩散生成任务中通用的"负引导"设计范式。
4. **Estimated Separation 推理调度**：在高分噪阶段 attend 含噪混合、低分噪阶段 attend 估计干净值，兼顾稳定性与保真度，可参考用于其他条件生成任务的调度设计。
5. **无需额外标注的 placement readout**：仅通过 stem 的 video attention 直接读取声源位置轨迹，无需训练额外检测头，可与本团队 4D 场景重建方向结合。

## 关键术语表
- **Audio-mix**：混合音轨，场景中所有声源叠加后的单条音频输出，无法直接区分各源。
- **Stem**：独立声源波形，由 SepGen 为每个 caption 单独生成的音频轨道。
- **Flow matching**：流匹配，一种扩散模型训练目标，学习从噪声到数据的向量场。
- **LoRA（Low-Rank Adaptation）**：低秩适配，冻结主干权重、仅训练低秩旁路参数的微调方法。
- **J-swap（Judge Swap Margin）**：Judge 评分 swap 边际，目标 stem 对目标 caption 的评分减去对兄弟 caption 的评分，衡量源标签忠实度。
- **CSAG（Cross-Stem Attention Guidance）**：跨音轨注意引导，以兄弟 caption 为负分支进行正负分支差分，减少多 stem 间的泄漏。
- **Estimated Separation**：估计分离推理策略，在去噪后期（$\sigma \leq 0.97$）将 stems 的条件从含噪 audio-mix 切换为当前步估计的干净 audio-mix。
- **LTX-2.5**：本文使用的预训练联合音视频生成骨干模型（22B 参数，LTX 系列）。

## 可复现要素
- **数据集**：CelebV-HQ、URMP、VGGSound、MUSIC-21 均为公开数据集；自生成片段和合成混合数据见论文附录。论文未提及所有训练数据是否合并开源，但项目页面提供示例。
- **代码/权重**：代码、checkpoint 和数据集均在 https://sepgen.github.io/ 开源。
- **关键超参**：LoRA rank/scale=128，训练阶段1共12,000步、阶段2共3,000步，单卡 NVIDIA A6000，学习率 $10^{-4}$ linear decay，batch size=1，bfloat16；分离模式 30 Euler steps，生成模式 15-step res 2s sampler；CSAG 参数 $\lambda=2, \tau=2.5, \alpha=0.5$；$\sigma_g=0.97$。
