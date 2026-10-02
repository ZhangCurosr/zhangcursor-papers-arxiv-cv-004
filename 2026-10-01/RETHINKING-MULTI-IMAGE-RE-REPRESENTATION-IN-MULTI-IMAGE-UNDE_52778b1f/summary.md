---
title: "RETHINKING-MULTI-IMAGE-RE-REPRESENTATION-IN-MULTI-IMAGE-UNDE"
source: https://arxiv.org/pdf/2609.39363v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 07:44:34"
field: "多模态推理与视觉智能体"
keywords: ["多图理解", "多模态大语言模型", "视觉工具", "强化学习", "图像重表征", "Mosaic", "MosaicBench"]
innovations: ["提出多图重表征统一框架与五类设置", "引入Mosaic视觉工作台与10种可组合操作", "仅用简单奖励的RL使智能体学会多步视觉工具组合"]
benchmarks: ["MosaicBench", "BLINK", "M4Bench", "Mantis", "MMIU"]
---

# 论文速读：RETHINKING MULTI-IMAGE RE-REPRESENTATION IN MULTI-IMAGE UNDERSTANDING

## 一句话总结
本文提出**多图重表征**（multi-image re-representation）框架，将链式思维推理与视觉工具使用统一为"组织视觉证据"的两种方式；引入**Mosaic**（含10种可组合图像操作的视觉工作台）与**MosaicBench**（面向细粒度多图理解的接地基准），实验表明视觉重表征的收益强烈依赖任务类型，且仅用准确率与格式奖励的强化学习即可使智能体学会多步视觉工具组合与多样化解题模式。

## 研究问题与动机
1. **核心问题**：在多模态大语言模型（MLLM）处理多图任务时，何时**视觉重表征**比纯文本推理更有效？
2. **现有不足**：当前多图基准（如BLINK、M4Bench、MMIU）混合了高层语义与细粒度接地需求，无法区分视觉重表征的真实收益来源。
3. **方法缺失**：多数多图理解模型将每张图片编码为视觉token后隐式推理，缺乏显式构造中间视觉表示的机制。
4. **训练难题**：如何让智能体在无演示轨迹、无密集工具奖励的情况下学会多步视觉操作组合？

## 核心贡献（创新点）
1. **形式化多图重表征并定义五类设置**（No-Re²、T-Re²、PG-Re²、V-Re²、PV-Re²），统一了文本链式思维与视觉工具交互的理论框架。
2. **引入Mosaic视觉工作台**，提供持久化图像资产、10种确定性可组合操作（裁剪、旋转、调整大小、翻转、仿射变换、坐标网格、像素差、单应性变换、拼贴、叠加），支持在线构造与复用中间视觉表示。
3. **创建MosaicBench基准**，从12个公开数据集构建560个样本、28类任务（分辨率、方向、精度比较、假设检验、上下文干扰、空间参考），专门评测细粒度多图接地能力。
4. **训练MosaicAgent-8B**，仅用准确率+格式二值奖励通过GRPO强化学习，智能体无需演示轨迹即学会多步视觉工具组合，并涌现出多样化解题模式（ progression、self-correction、post-stabilization continuation）。
5. **发现任务依赖性规律**：视觉重表征对精确视觉证据任务（假设检验62.9% vs 基线31.7%、精度比较87.0%、方向56.9%）显著提升，而对高层语义主导任务收益较小或不一致。

## 方法详解
### 多图重表征形式化
给定问题$ x = (q, I_1, \ldots, I_n) $，重表征器构造中间记录$ z \in \mathcal{Z}_m $（$ m \in \{\text{text}, \text{visual}\} $），求解器生成答案：
$$ p_{\theta,\phi}^m(y|x) = \sum_{z \in \mathcal{Z}_m} \underbrace{p_\theta^m(z|x)}_{\text{re-representer}} \underbrace{p_\phi(y|x,z)}_{\text{solver}} $$

**五类设置**：
- **No-Re²**：直接输出答案，无显式中间轨迹
- **T-Re²**：自由文本链式思维（CoT）
- **PG-Re²**：提示引导的文本重描述（描述相关内容、定位源图、组织跨图比较）
- **V-Re²**：在线视觉重表征，通过Mosaic交互轨迹$ \tau = (a_1,o_1,\ldots,a_T,o_T) $
- **PV-Re²**：预构建视觉中间表示（由MosaicAgent提前构造），模型仅使用无交互痕迹

### Mosaic视觉工作台
**持久化图像工作区**：源图像与派生视图作为独立资产存储（如`[img1]`），操作不修改输入，新资产自动追加编号。每个资产有RGBA编辑视图与RGB观察视图（透明棋盘格标记透明区域）。

**10种可组合操作**：
| 操作 | 功能 |
|------|------|
| `crop` | 矩形裁剪 |
| `rotate` | 任意角度旋转 |
| `resize` | 精确尺寸缩放 |
| `flip` | 水平/垂直翻转 |
| `apply_affine` | 2D仿射变换（旋转+缩放+平移） |
| `draw_grid` | 绘制归一化坐标网格 |
| `compare_diff` | 逐像素差异热力图 |
| `apply_homography` | 3×3射影变换 |
| `make_collage` | N图拼贴为网格 |
| `overlay` | 前景图叠加至背景图指定位置 |

**封闭证据性质**：固定模型与工具实现时，$ I(Y^*; Z|X) = 0 $，即中间记录仅重组输入证据而不引入外部信息。

### 强化学习训练（MosaicAgent-8B）
- **基础模型**：Qwen3-VL-8B-Instruct
- **算法**：Group Relative Policy Optimization (GRPO)，8次rollout采样，组归一化优势，无critic网络
- **奖励函数**：$ r(\tau,y,y^*) = \text{acc}(y,y^*) + \text{fmt}(\tau,y) $（均为二值）
- **训练数据**：3,509个示例（32类任务，每类约109个），通过"无工具错误但2-7次有工具正确"筛选
- **超参**：AdamW lr=1×10⁻⁶，batch=32，1 epoch，KL=0，entropy=0，4×NVIDIA H100
- **长度限制**：prompt≤16,384 token，response≤20,480 token，每回合最多10个assistant turn

## 实验与结果
### 基准评测
| 模型 | MosaicBench | M4Bench | Mantis | BLINK | MMIU |
|------|-------------|---------|--------|-------|------|
| **MosaicAgent-8B** | **63.3±1.1** | **61.8±1.4** | 81.6±1.2 | 62.8±1.0 | 57.2 |
| Qwen3.5-27B（最强开权基线） | 60.9 | 57.5 | 80.7 | 72.3 | 69.5 |
| GPT-5.4 | 58.0 | 66.5 | 76.5 | 73.9 | — |
| Claude Sonnet 5 | 63.6 | 63.6 | 81.6 | 69.5 | — |

**MosaicBench分任务结果**（MosaicAgent-8B）：
- 分辨率（Resol.）：69.7%
- 方向（Orient.）：56.9%
- **精度比较（Prec.Comp.）：87.0%** ← 最大提升
- **假设检验（Hyp.Test）：62.9%** ← 基线仅31.7%，+31.2pp
- 上下文干扰（Ctx.Int.）：57.5%
- 空间参考（Spat.Ref.）：49.6% ← 相对弱项

### 重表征方法对比（Qwen3-VL-8B）
- **M4Bench Detailed Difference**：No-Re² 7.3% → T-Re² 46.1% → **V-Re² 59.9%**（+13.8pp over T-Re²）
- **M4Bench State Comparison**：V-Re²表现**差于**T-Re²，说明任务依赖性真实存在
- **BLINK Relative Depth**：PV-Re² 82.3% > T-Re² 78.2% > V-Re² 74.2%，预构建视觉表示有时优于在线构造

### 强化学习训练动态
- 准确率奖励：0.48 → 0.63（218步）
- 格式奖励：0.91 → 0.99
- 工具调用总量：1,830 → 3,037次（+1.66×）
- **工具分布偏移**：裁剪36.7%→54.8%，拼贴0.5%→6.2%，像素差16.1%→3.3%
- **带Mosaic的RL vs 纯CoT RL**：MosaicBench 63.3% vs 53.0%（+10.3pp），证明视觉工作台提供额外收益而非仅RL效应

### 消融：完整工具集 vs 仅裁剪
- MosaicBench：53.9%（仅crop）→ 63.3%（全工具集），+9.4pp
- M4Bench：58.8% → 61.8%，+3.0pp
- 说明除裁剪外的操作（尤其拼贴、仿射变换）对复杂任务至关重要

## 相关工作脉络
1. **多图理解基准**：MANTIS（交错指令微调）、BLINK（感知密集型任务）、MMIU（多领域多图理解）、M4Bench（多粒度多图基准）——本文指出这些基准混合语义与接地需求，需MosaicBench补充细粒度评测。
2. **多图MLLM**：Flamingo、PromptCap（提示引导图描述）、QG-CoC（问题引导链式图描述）——本文统一将这些视为文本重表征的特例。
3. **视觉智能体工具**：Thinking-with-Images、DeepEyes（局部.inspect）、Visual Sketchpad（草图）、PyVision（可执行代码）、VipAct（感知增强）——本文区别在于提供**通用可组合操作**而非任务专用工具。
4. **强化学习视觉代理**：OpenThinkIMG、VTool-R1、PyVision-RL——本文贡献在于仅用**简单二值奖励**（无密集工具奖励、无演示轨迹）即学会多步工具组合。
5. **链式思维与图文推理**：Multimodal-CoT、PromptCap——本文形式化文本vs视觉重表征的统一框架，填补理论空白。

## 局限性与未来方向
1. **任务依赖性限制**：视觉重表征对高层语义任务（如Mantis 81.6% vs 基线81.9%）收益有限，甚至MMIU下降1.1pp。
2. **空间参考任务薄弱**：Spat.Ref.仅49.6%，坐标定位类任务仍需改进。
3. **过度工具使用**：训练后post-stabilization continuation增加，智能体在答案已稳定后仍继续调用工具（验证或冗余操作）。
4. **预构建vs在线**：PV-Re²在某些任务（如BLINK Relative Depth 82.3%）优于V-Re²（74.2%），说明在线构造策略仍有优化空间。
5. **未来方向**：扩展至视频理解、具身视觉；改进停止决策机制；研究更稠密奖励设计；探索视觉重表征与文本重表征的自适应混合策略。

## 研究启发与可借鉴点
1. **五类重表征设置的实验框架**可复用于其他多模态推理研究，干净分离文本/视觉/在线/预构建因素。
2. **封闭证据形式化**（$ I(Y^*;Z|X)=0 $）为工具使用模型提供了理论边界，可作为后续工作的分析基线。
3. **仅用准确率+格式奖励的RL训练**证明密集工具奖励非必需，降低多步工具学习门槛，可迁移至其他agent系统。
4. **前缀可答性探测方法**（prefix answerability probing）分离轨迹构造与答案读取，为分析agent中间状态提供可复用工具。
5. **MosaicBench构建协议**（32类任务、12数据源、短路探针验证、来源级去重叠）可作为细粒度视觉基准建设的参考模板。

## 关键术语表
**Multi-image re-representation**：从源图像构造中间记录以组织视觉证据的过程，统一文本链式思维与视觉工具交互。
**Mosaic**：通用多图视觉工作台，提供持久化图像资产与10种可组合确定性操作。
**MosaicBench**：面向细粒度多图理解的接地基准，560样本、28类任务，源自12个公开数据集。
**Closed-evidence re-representation**：模型与工具仅能访问输入证据（无外部信息），满足$ I(Y^*;Z|X)=0 $的设定。
**V-Re²（Visual re-representation）**：在线视觉重表征，通过交互轨迹$ \tau $动态构造视觉中间表示。
**PV-Re²（Prefabricated visual re-representation）**：预构建视觉重表征，使用预先构造的视觉中间表示但无交互痕迹。
**Answerability**：在固定轨迹前缀状态下，求解器能输出正确答案的概率，用于分离轨迹构造与答案读取效益。
**GRPO（Group Relative Policy Optimization）**：无需critic网络的强化学习算法，通过组归一化奖励计算优势。

## 可复现要素
- **代码与数据**：https://github.com/gengyuanmax/Mosaic（论文声明开源）
- **基准数据集**：MosaicBench（560样本，28类任务），训练套件（3,509样本，32类任务）
- **源数据集**：VisDrone2019-MOT、TT100K、CAMELYON16、MVTec AD、CARPK、PubLayNet、LEVIR-CD、SmartDoc15-CH1、MSD、Rico、COCO、BDD100K（12个公开数据集）
- **基础模型**：Qwen3-VL-8B-Instruct（开权）、Qwen3-VL-32B、GPT-5.4、Claude Sonnet 5等
- **关键超参**：lr=1×10⁻⁶、batch=32、1 epoch、KL=0、entropy=0、max_prompt=16,384、max_response=20,480、max_turns=10、rollouts=8
- **硬件**：4×NVIDIA H100 GPU
