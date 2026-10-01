---
title: "REVISIT-TO-SEGMENT-WORKING-MEMORY-DISTILLA-TION-FOR-REASONIN"
source: https://arxiv.org/pdf/2609.34863v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:44:09"
field: "视觉语言分割与多模态推理"
keywords: ["推理分割", "工作记忆蒸馏", "在线策略自蒸馏", "GRPO", "多模态大模型", "reasoning segmentation"]
innovations: ["首次将自生成推理轨迹和定位提议形式化为工作记忆并用于推理分割", "首次将在线策略自蒸馏(OPSD)引入推理分割，以质量选定的rollout为教师进行token级分布对齐", "联合OPSD与GRPO的工作记忆蒸馏框架SWiM，在ReasonSeg系列基准上达到SOTA"]
benchmarks: ["ReasonSeg (RS)", "ReasonSeg-R (RS-R)", "ReasonSeg-X (RS-X)", "MUSE", "MMR", "RefCOCO family"]
---

# 论文速读：REVISIT TO SEGMENT: WORKING MEMORY DISTILLATION FOR REASONING SEGMENTATION

## 一句话总结
本文提出 **SWiM**，一个基于工作记忆蒸馏的推理分割框架：利用 MLLM 自身生成的高质量推理轨迹和定位提议作为"工作记忆"，通过在线策略自蒸馏（OPSD）将教师模型的token级分布指导迁移给只接收原始图像-查询的学生模型，并结合 GRPO 结果导向强化学习联合优化；在 ReasonSeg 系列基准上达到了 SOTA。

---

## 研究问题与动机
- **推理分割的核心挑战**：现有方法依赖 SFT 或 RL 直接学习从自然语言查询到像素级 mask 的映射，但 MLLM 生成的中间推理轨迹（reasoning traces）和定位提议（localization proposals）本身蕴含了模型对查询的理解过程，尚未被系统性地利用。
- **工作记忆是否有效**：论文首先验证——让 MLLM 在重新审视同一张图像和查询时，将自身先前生成的推理轨迹与定位结果作为额外上下文（即"工作记忆"），能否提升分割性能。实验（Figure 2）证实， Across model scales 和 benchmarks，使用工作记忆后 gIoU 和 cIoU 均有稳定提升。
- **如何在不依赖工作记忆的推理阶段训练**：既然工作记忆在推理时可用作额外信号，论文希望将其转化为训练阶段的监督信号，使模型在推理时既能单独预测，也能附带工作记忆重访，即实现"训练时从重复尝试中获益，推理时可选择性使用"。
- **现有基线的不足**：StAR 等方法虽有 rollout 采样，但未利用 self-generated 的 prior attempts 作为蒸馏源；多数方法仅依赖最终结果奖励（如 Seg-Zero、SAM-R1），缺少 token 级别的分布对齐指导。

---

## 核心贡献（创新点）
1. **首次将自生成推理轨迹与定位提议形式化为"工作记忆"**，并系统验证其作为上下文信息和训练引导对推理分割的有效性（不仅是推理增强，还作为蒸馏信号）。
2. **首次将在线策略自蒸馏（OPSD）引入推理分割任务**：以工作记忆条件化的模型为自教师，沿学生生成的轨迹提供 token 级分布监督，与已有工作（如 StAR 的结果导向 RL、SEED 的后见之明技能提取）的本质区别在于：教师条件是模型自身的高质量 prior attempts，而非环境反馈或 peer rollout。
3. **提出 SWiM 框架，联合优化 OPSD 与 GRPO**：OPSD 传递 token 级语义/定位引导，GRPO 提供 mask 质量的结果反馈，两者互补；与单纯 RL 微调相比，蒸馏信号使学生在无工作记忆的纯推理设置下也获益。
4. **SOTA 性能**：在 ReasonSeg / ReasonSeg-R / ReasonSeg-X 三个基准上，SWiM（Qwen3-VL-8B）平均 gIoU = 66.5、cIoU = 62.0，分别超过复现 StAR baseline 1.9 / 3.6 个百分点；含 WM+MV 时进一步提升至平均 gIoU 67.8 / cIoU 63.0。

---

## 方法详解

### 整体架构
- **骨干模型**：Qwen3-VL-8B / Qwen3-VL-32B-Instruct（LoRA rank=64, scaling=64），冻结视觉 backbone 与 SAM2.1 Hiera-Large。
- **解耦推理-分割公式**：MLLM 策略 $\pi_\theta$ 生成响应 $y$，从中提取定位提示 $\mathcal{P}(y) = \{(b_j, p_j)\}$，送入冻结 SAM2 生成 mask $\widehat{M} = \bigcup_j S(I, b_j, p_j)$。

### 工作记忆构造（Working Memory Construction）
- 从当前 rollout 策略 $\pi_{\theta_{\mathrm{old}}}$ 采样 $N$ 条响应 $\{y^i\}_{i=1}^N$，得到对应 mask。
- 按训练 mask IoU 排序，选取 Top-K 高质量响应构成工作记忆 $\mathcal{W}$：
$$\mathcal{W} = \mathrm{TopK}_{y^i} Q(\widehat{M}^i, M^\star)$$
- 完整保留每条响应的推理文本 + 结构化定位输出，保证"解读—定位"的关联不被破坏。

### 在线策略自蒸馏（OPSD）
- **教师输入**：$x^{\mathcal{W}} = C(x, \mathcal{W})$，即在原始图像-查询上拼接工作记忆。
- **学生输入**：仅 $x = (I, q)$。
- 对每个学生的 response prefix $\boldsymbol{y}_{<t}^i$，分别计算：
  - 学生分布 $p_{i,t} = \pi_\theta(\cdot \mid x, \boldsymbol{y}_{<t}^i)$
  - 教师分布 $q_{i,t} = \pi_{\theta_{\mathrm{old}}}(\cdot \mid x^{\mathcal{W}}, \boldsymbol{y}_{<t}^i)$
- 使用 **Jensen-Shannon 散度（JSD）** 对齐两者（实验对比证实 JSD 优于 forward/reverse KL）。
- 实际计算时对 vocab 做压缩分区：保留教师 top-L=16 token，其余合并为一个 residual bin，最小化：
$$\mathcal{L}_{\mathrm{OPSD}} = \mathbb{E}_{(i,t)\sim\mathcal{U}}\left[\mathrm{JSD}(\widetilde{p}_{i,t} \| \widetilde{q}_{i,t})\right]$$

### 结果导向强化学习（GRPO）
- 对每组 $N$ 条 rollout，计算 reward：
$$r^i = 2r_{\mathrm{seg}}^i + r_{\mathrm{fmt}}^i + r_{\mathrm{nr}}^i$$
  （分段质量 + 格式约束 + 无过度重复奖励）
- 归一化 advantage：$A^i = \frac{r^i - \mu_r}{\sigma_r + \epsilon}$
- GRPO 目标：
$$\mathcal{L}_{\mathrm{GRPO}} = -\mathbb{E}\left[\min\left(\rho_{i,t} A^i, \mathrm{clip}(\rho_{i,t}, 1-\delta, 1+\delta) A^i\right)\right]$$

### 联合优化
$$\mathcal{L}_{\mathrm{SWiM}} = \mathcal{L}_{\mathrm{GRPO}} + \lambda \mathcal{L}_{\mathrm{OPSD}}, \quad \lambda = 0.1$$
- **两阶段训练**：
  - Stage 1（RL-only warm-up）：1 epoch，lr=$10^{-5}$，batch=24，N=16 rollouts，从 LVIS/RefCOCOg/gRefCOCO（5166 样本）中学习带/不带 WM 两种输入。
  - Stage 2（WM distillation）：2 epochs，lr=$10^{-6}$，batch=16，从 Stage 1 中精选 869 个难例（Frontier / Refinement / Hard but Partially Solvable / Consistently Unsuccessful / Replay 五类），N=16、K=8。
- 推理时：学生默认仅用 $(I,q)$；也可附加 WM 重访，或与多数投票（MV）结合。

---

## 实验与结果

### 数据集与基线
- **主要基准**：ReasonSeg（RS）、ReasonSeg-R（RS-R）、ReasonSeg-X（RS-X），指标为 gIoU / cIoU。
- **额外基准**：MUSE（多目标）、MMR（组合物体-部件）、RefCOCO 族（引用表达分割）。
- **对比基线**：SFT/RL 类（LISA、SegLLM、READ、CoReS、RSVP、SAM-R1、Seg-Zero、VisionReasoner、StAR 等）；推理时扩展类（SAM 3 Agent、RSAgent、Rea²Seg）；自复现 StAR baseline。
- **模型规模**：Qwen3-VL-8B / 32B，冻结 SAM2.1。

### 主要结果（Qwen3-VL-8B，不含 RS-X 训练数据）
| 方法 | RS gIoU/cIoU | RS-R gIoU/cIoU | RS-X gIoU/cIoU | **Average gIoU/cIoU** |
|---|---|---|---|---|
| StAR | 68.6 / 60.2 | 71.3 / 65.7 | 54.0 / 49.2 | 64.6 / 58.4 |
| **SWiM（plain）** | **70.8 / 65.8** | **73.0 / 70.8** | **55.6 / 49.3** | **66.5 / 62.0** |
| SWiM + WM & MV | 71.6 / 67.3 | 73.5 / 70.3 | 58.1 / 51.5 | 67.8 / 63.0 |

- **相对 StAR plain 提升**：平均 gIoU +1.9pp，cIoU +3.6pp。
- **最大配置（32B + WM+MV）**：平均 gIoU 70.4 / cIoU 65.9。
- **额外基准**：MUSE cIoU 55.7（+WM&MV 达 56.8）；MMR test cIoU 28.5；RefCOCOg test cIoU 73.5。

### Ablation 关键结论
- **OPSD + GRPO 联合 > 单独任一**（Table 2）：联合优化平均 gIoU 66.5 / cIoU 62.0，仅 GRPO 65.7 / 60.9，仅 OPSD 66.2 / 60.9。
- **蒸馏权重**：$\lambda = 0.1$ 最优；$\lambda=10$ 明显下降。
- **教师上下文对比**（Appendix F）：WM > GT hints > GT+WM > Random hints（随机 SAM2 mask 显著损害性能）。
- **蒸馏损失函数**：JSD > Reverse KL > Forward KL（综合三基准）。
- **工作记忆大小**：K=8 最优；K=16 cIoU 略升但 gIoU 下降，说明并非越大越好。

---

## 相关工作脉络
1. **LISA（Lai et al., 2024）**：首次将 MLLM 推理接入 SAM 解码器做推理分割；SWiM 在其基础上进一步利用 rollout 质量选择构造工作记忆，并通过蒸馏而非仅 SFT 提升。
2. **StAR（Yun et al., 2026）**：使用 rollout 训练 + mask 级投票；SWiM 的核心区别在于将工作记忆作为 teacher 条件进行 token 级分布蒸馏，而非仅依赖奖励信号。
3. **Seg-Zero / SAM-R1 / VisionReasoner**：以 RL（GRPO/PPO）优化分割策略；SWiM 在这些方法的基础上引入自蒸馏，使纯推理（无额外上下文）也能受益。
4. **On-policy Self-Distillation（OPSD，Zhao et al., 2026）**：原用于纯文本 LLM，SWiM 是首个将其引入视觉-语言分割的任务扩展，创新点在于用"同模型在质量更高条件下的 rollout"作为教师。
5. **Multi-Rollout OP Distillation（Yu et al., 2026） / SEED（Wu et al., 2026）**：前者使用 peer 成功/失败 rollout，后者动态提取后见之明技能；SWiM 的 WM 来自**同输入的高质量 prior attempts**，不依赖 peer 或环境 hindsight。
6. **CoReS / RSVP / CoPRS**：强调结构化推理和视觉 prompt；SWiM 与它们正交，可在相同 backbone 上叠加使用。

---

## 局限性与未来方向
- **目标范围的标注歧义**：Figure 10 显示部分低 IoU 源于 query 本身允许多种合理解释（如"handrail vs. railing"、"knob vs. control unit"），单一 GT mask 会惩罚合理变体。建议探索多候选 mask + 多人工验证 GT。
- **SAM2 mask 生成不完整**：即使 MLLM 定位合理，SAM2 仍可能输出残缺 mask（如 barbell、glass windows 案例）。未来可让策略提供更多空间提示，或联合优化策略与 mask decoder。
- **工作记忆大小并非单调增益**：K=16 仅小幅改善 cIoU 却降低 gIoU，暗示存在信息过载或噪声累积，需进一步研究自适应 WM 选择策略。
- **仅使用约 1.8k 训练样本**：虽避免使用 ReasonSeg 训练集，但在更大数据规模下 WM distillation 的缩放行为尚待验证。
- **推理时 WM 的构建依赖无排名**：测试时无法获取 ground-truth 来排序 WM，采用未排名采样；可探索自置信度估计或轻量验证器改进。

---

## 研究启发与可借鉴点
1. **"自生成高质量 rollout 作为教师"的范式可迁移**：不仅限于分割，任何需要多步推理+结构化输出的任务（如视觉定位、文档理解、代码生成）均可借鉴"以 quality-selected prior attempts 构造教师上下文做 OPSD"的思路。
2. **两阶段训练设计值得复用**：先 RL-only warm-up（同时学有/无 WM 输入），再精细蒸馏（难例精选 869 条），可在资源有限时降低训练不稳定风险；难度分层（Frontier/Refinement/Hard/Replay）的筛选策略可移植。
3. **JSD 优于 KL 分量的发现**：对分布对齐型蒸馏，JSD 在三基准上的稳定优势可作为默认选择；若学生与教师分布差异较大时，JSD 的对称性避免了 mode-covering 问题。
4. **推理时灵活性设计**：同一模型既支持 plain 推理又支持 WM-revisit 推理，且两者可叠加 MV；这种"训练时从更强教师获益，推理时按需调用"的模式可推广至 Agent 系统和多轮交互场景。
5. **与团队方向的结合机会**：若团队关注**视觉-语言 grounding、VQA segmentation、或 agentic perception**，SWiM 的 WM-distillation 可直接与 StAR/VisionReasoner 类基线组合，或扩展到视频/时序 segmentation（利用时间维度的 prior attempts 作为 WM）。

---

## 关键术语表
- **Reasoning Segmentation（推理分割）**：要求 MLLM 基于自然语言查询进行上下文推理，并在像素级输出目标区域 mask 的任务。
- **Working Memory（工作记忆）**：模型对同一输入多次 rollout 生成的推理轨迹与定位提议的集合，作为重访时的额外上下文。
- **On-Policy Self-Distillation（OPSD，在线策略自蒸馏）**：用模型自身在不同条件下生成的分布作为教师，沿学生轨迹做 token 级分布对齐的训练范式。
- **Jensen-Shannon Divergence（JSD）**：学生-教师分布之间的对称散度度量，论文证实其在 OPSD 中优于 forward/reverse KL。
- **GRPO（Group Relative Policy Optimization）**：基于组内归一化 advantage 的 PPO 变体，用于结果导向的强化学习优化。
- **gIoU / cIoU**：逐样本 IoU 均值 / 累积 IoU，推理分割 benchmark 的主要评估指标。
- **Majority Voting（MV）**：对多条独立生成的 mask 做聚合投票，进一步提升稳定性。
- **StAR（Segment Anything Reasoner）**：当前推理分割的重要 baseline，采用 rollout + reward-based RL；SWiM 在其基础上引入自蒸馏。

---

## 可复现要素
- **代码**：已开源，GitHub: https://github.com/yancilin/SWiM
- **数据集**：训练使用 LVIS、RefCOCOg、gRefCOCO 共 5166 样本（Stage 1）+ 869 精选样本（Stage 2）；**未使用** ReasonSeg / ReasonSeg-X 训练集。推理评测在 ReasonSeg、ReasonSeg-R、ReasonSeg-X test set 上进行；额外评估 MUSE val、MMR val/test、RefCOCO testA/+ test。
- **权重**：Qwen3-VL-8B/32B-Instruct 基础权重 + LoRA adapter；SAM2.1 Hiera-Large（冻结）。
- **关键超参**：LoRA rank=64, scaling=64；Stage 1 lr=$10^{-5}$, batch=24, 1 epoch；Stage 2 lr=$10^{-6}$, batch=16, 2 epochs；N=16 rollouts, K=8 top；$\lambda=0.1$；max response length=2048 tokens；image resized to 840×840。
- **硬件**：24 × NVIDIA H800 GPU。
- **随机种子**：42。
- **未提及**：具体 reward 函数细节（参考 StAR 的 Stage-1 data construction）、 Teacher top-L 的具体实现代码。
