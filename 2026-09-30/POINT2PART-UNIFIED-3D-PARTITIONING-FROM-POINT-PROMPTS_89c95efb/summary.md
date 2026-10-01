---
title: "POINT2PART-UNIFIED-3D-PARTITIONING-FROM-POINT-PROMPTS"
source: https://arxiv.org/pdf/2609.38180v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:34:39"
field: "3D 形状理解与生成"
keywords: ["3D part decomposition", "point prompts", "exclusive partition", "joint partitioning", "3D generation", "promptable segmentation"]
innovations: ["将部件分解形式化为全局联合划分（exclusive & exhaustive），通过 argmax 天然保证互斥且完备", "提出 Prompt Encoder + Part Decoder 架构，在共享 shape latent 空间内一次性联合解码所有部件", "统一支持图像/网格输入下的部件生成与表面分割三项任务，穿透率降低一个数量级"]
benchmarks: ["PartObjaverse-Tiny", "PartNeXt"]
---

# 论文速读：POINT2PART-UNIFIED-3D-PARTITIONING-FROM-POINT-PROMPTS

## 一句话总结
本文提出 Point2Part，将 3D 部件分解重新建模为对整个形状空间的**联合划分**（exclusive & exhaustive），通过 3D 点提示（point prompts）即可交互式控制，支持图像到部件生成、网格到部件生成、以及网格表面分割三项任务，在几何兼容性上较先前 SOTA 提升一个数量级。

## 研究问题与动机
- **现有方法产生重叠/空洞**：主流部件生成与分割方法通常独立解码每个部件，允许同一空间被多部件占据（penetration）或完全不被任何部件覆盖（gaps），无法形成对全形的完整划分。
- **控制方式间接/有限**：已有工作依赖部件数、文本名称、2D 掩码等间接提示，难以对遮挡区域进行细粒度控制；点提示分割方法虽可在表面上点击，但各提示独立响应，仍需后处理消除重叠。
- **缺乏统一的划分视角**：现有效率关注部件品质本身，而不从全局一致性的角度约束部件间关系，导致下游装配/编辑应用受限。

## 核心贡献（创新点）
1. **将部件分解形式化为联合划分问题**：提出 exclusive（无体积交）与 exhaustive（并集=全形）的严格数学定义，从根本上消除重叠与空洞，与先前逐部件独立生成的本质区别在于约束来自全局而非后验惩罚。
2. **提示编码器（Prompt Encoder）设计**：将每个点的提示映射为一个 part token，通过两层注意力结构（先在点内聚合，再跨部件交互）同时感知形状几何与其他部件语义，与现有仅利用位置嵌入的提示方法截然不同。
3. **联合得分的部件解码器（Part Decoder）+ argmax 分配**：一次前向完成所有 N 个部件的查询打分，用 argmax 直接给出互斥且完备的体素分配，无需 suppression、flood fill 或重叠惩罚项，与 Mask2Former 风格的独立分类器形成对比。
4. **统一的三任务框架**：基于共享 shape latent 空间，同一模型同时支持 image-to-part、mesh-to-part 生成与 mesh 表面分割，先前的工作在这些任务上通常使用不同架构。
5. **SOTA 结果与数量级提升**：在所有三个任务的部件品质指标上均达到最优，part 间穿透率（pen%）较 SOTA 降低约一个数量级（0.06 vs. 2.09 等）。

## 方法详解
**整体流程**（图 2）：给定 mesh 或 image → Shape Backbonde 编码为 shape latents Z → Prompt Encoder 处理 N 个 3D 点提示得到 part tokens P⁽ᴸ⁾ → Part Decoder 对体内查询 x ∈ Ω 联合打分得到分配 ℓ(x) → Mesh Extraction 转换为 N 个封闭部件网格。

**Shape Backbone**：基于预训练的 Hunyuan3D-2.1（或 TripoSG），将输入编码为 M 个 shape latents Z ∈ ℝ^(M×C)。SDF 解码器通过公式 (2) 给出全形符号距离场 s(x)，用于后续提取零水平集获取整体网格 ∂Ω。内部查询特征初始化为位置编码 φ(x)。

**Prompt Encoder**（图 2b，L=2 层）：
- 第 1 层：每个点 c_{j,k} 经位置嵌入后，对所有点做 CrossAttn(x → Z) 聚合局部几何信息，再通过 attention pooling Pool_k 对 K 个点做聚合，得到一个 part token；随后 SelfAttn 实现部件间交互。
- 后续 L−1 层重复 CrossAttn(Z) + FFN，逐步精炼 part tokens P⁽ᴸ⁾ ∈ ℝ^(N×C)。

**Part Decoder**（图 2c，L′=3 层）：
- 每层对每个查询 x：先 CrossAttn(h(x), Z) 收集局部几何，再 CrossAttn(·, P⁽ᴸ⁾) 注入部件特异性信息。
- 输出得分：ℓ_j(x) = ⟨f_h(h⁽ᴸ′⁾(x)), f_s(P_j⁽ᴸ⁾)⟩ / √C，每个 part token 作为动态线性分类器。

**Mesh Extraction**（核心公式 8–9）：
- 部件分配 π(x) = argmax_j ℓ_j(x)，天然满足 exclusive & exhaustive（式 7）。
- 对每个部件 j 构造局部 SDF：s_j(x) = s(x) 若 π(x)=j，否则取 |s(x)|，使 s_j < 0 恰好对应 Ω_j。
- Marching Cubes 提取零水平集，获得 N 个封闭网格；接缝处因插值产生极小残留穿透。

**训练目标**（式 11）：
- **L_cls**：Focal Cross-Entropy（γ=2），加权平衡不同大小部件，强调难样本。
- **L_dice**：Soft Dice Loss，提供部件级区域监督。
- **L_cons**：一致性损失（式 14），对同一 GT 部件的两个独立采样提示集合，要求对应 part token 一致（λ_cons=0.5）。
- 查询采样策略：重点集中在部件边界附近（part boundary 1024 点 + surface 1024 + near surface 1024），其余均匀采样。

**推理细化**（coarse-to-fine，式 15）：
- 整体 SDF 在窄带 |s(x)|<η(η=0.05) 内细化；部件 SDF 仅在 s(x)<η 且 top-2 得分差小于 δ(δ=0.2) 的区域细化，其余继承粗网格值。

## 实验与结果
- **训练数据**：HY3D-Bench（240K 带部件标注资产）。
- **评测基准**：PartObjaverse-Tiny（200 mesh，面级部件标注），另在 PartNeXt 上额外验证。
- **主要指标**：pCD、pF1@τ（部件品质）；pen%（部件穿透率）；wt%（watertight 比例）；CD/F1@.05（整体几何）；mIoU（分割面级）。

**Mesh-to-Part 生成**（Table 1）：
| 方法 | pCD↓ | pF1@.01↑ | pen%↓ | wt%↑ |
|---|---|---|---|---|
| CubePart | 4.71 | 51.5 | 2.09 | 100.0 |
| HoloPart | 5.29 | 46.6 | 1.01 | 32.6 |
| X-Part | 4.53 | 52.5 | 3.05 | 90.8 |
| **Ours** | **2.73** | **57.0** | **0.06** | **100.0** |

- 穿透率从 2.09% 降至 0.06%，**改善约 35 倍**；pCD 从 4.71 降至 2.73。
- 推理时间 24.4s，为 mesh 输入最快方法。

**Image-to-Part 生成**（Table 1 下半）：
| 方法 | pCD↓ | pF1@.05↑ | pen%↓ | wt%↑ |
|---|---|---|---|---|
| OmniPart | 6.43 | 60.8 | 0.96 | 94.7 |
| **Ours** | **5.34** | **68.6** | **0.01** | **100.0** |

- 穿透率降至 0.01%，pF1@.05 提升约 8 个百分点。

**Part Segmentation**（Table 2）：
| 方法 | mIoU↑ | pCD↓ | pF1@.05↑ | 推理(s) |
|---|---|---|---|---|
| PartSAM | 50.31 | 3.24 | 79.2 | 36.6 |
| S²AM3D | 50.43 | 3.16 | 80.5 | 0.3 |
| PartField | 69.10 | 5.31 | 72.5 | 1.1 |
| **Ours** | **69.80** | **2.10** | **87.0** | **0.3** |

- mIoU 达 69.80，pCD 仅 2.10，推理时间 0.3s 与 S²AM3D 并列最快。
- PartNeXt 验证集上同样全面领先（Table 6，mIoU 57.76，pCD 2.39）。

## 相关工作脉络
1. **Promptable 3D 分割**：Point-SAM、P3-SAM、PartSAM、S²AM3D 等均独立响应每个点提示，输出可能重叠/留空；本文的关键差异在于"联合分区"设计，一次输出即为完整划分。
2. **部件级生成**：PartCrafter、PartPacker（仅部件数）、CubePart（文本名）、OmniPart（2D 掩码）、HoloPart/UniPart（给定分割图）——控制间接且无互斥约束；本文直接通过 3D 点定位且保证不重叠。
3. **primitive-based 分解**：Cuboid/Superquadric/CvxNet 等方法用几何原语近似形状，输出非真实部件，指标不可比；本文关注真实语义部件的几何一致性。
4. **Segment Anything 系列（2D→3D 迁移）**：SAM/SegViGen 等将 2D 分割提升至 3D，但与本文的"联合划分+生成封闭网格"范式不同。
5. **PartField（特征场聚类）**：基于 3D 特征场的无提示聚类方法，粒度由用户指定但边界不可控；本文的点提示提供直接交互式控制。

## 局限性与未来方向
- **极薄结构与大开放表面**：预训练 backbone（VAE/SDF）难以精确重建此类几何，导致部件边界出现误差（Appendix Figure 11）。
- **模糊提示位置**：当点提示放置在两部件接触边界时，decoder 可能合并两个部件（Figure 8 第二列），但增加每部件提示点数可缓解。
- **未来方向**：改进 backbone 对 thin/open geometry 的表示能力；探索无需封闭体积的场景扩展；进一步降低推理延迟（目前 image 输入约 38s）。

## 研究启发与可借鉴点
1. **"联合划分替代独立生成"的设计范式**：对于任何需要互斥+完备分配的任务（如体素分割、场景图生成、材料分配），argmax 式的全局分配可天然避免重叠，无需额外惩罚项。
2. **共享 latent 空间支持多模态统一建模**：将 image/mesh 映射到同一 shape latent，使单一 prompt encoder + part decoder 可同时服务多个任务，减少重复架构设计。
3. **Coarse-to-fine 细化策略（窄带+SDF 修正）**：仅在边界不确定区域（ℓ_(1)−ℓ_(2)<δ）进行高分辨率细化，兼顾效率与精度，可迁移至其他 3D 生成/分割任务。
4. **L_cons 一致性损失的思想**：对同一目标的多次独立采样施加 token 级对齐，可用于任何需要提示不变性的多提示设定。
5. **与团队方向的结合机会**：本文的点提示交互范式可与后续研究中的用户意图建模、可编辑 3D 生成、部件级物理仿真等方向深度融合。

## 关键术语表
- **Exclusive partition**：任意两个部件的体积交集为零（vol(Ω_i ∩ Ω_j)=0, i≠j），确保无穿透重叠。
- **Exhaustive partition**：所有部件的并集等于原始形状（∪Ω_j=Ω），确保无空洞缺失。
- **Point prompt**：用户在 3D 物体表面放置的一个或多个坐标点，作为指定目标部件的交互提示。
- **Shape latent (Z)**：从输入图像或网格编码得到的共享隐空间表示，供 prompt encoder 和 part decoder 共同使用。
- **Part token (P)**：prompt encoder 为每个部件生成的 C 维特征向量，充当该部件在 part decoder 中的动态分类器权重。
- **Focal Cross-Entropy**：对标准 CE 加入难样本加权因子 (1−p)^γ，使训练更关注边界处不确定的查询点。
- **Marching Cubes**：从体素化符号距离场中提取零水平集等值面的经典三维重建算法。
- **Coarse-to-fine refinement**：先在 128³ 粗网格上推理，再在 512³ 细网格上仅对窄带和决策边界附近的区域重新计算，以提升几何精度。

## 可复现要素
- **训练数据集**：HY3D-Bench（公开），240K 带部件标注资产。
- **测试数据集**：PartObjaverse-Tiny（公开，200 mesh）；PartNeXt（公开）。
- **代码**：已开源，https://henrytsui000.github.io/Point2Part
- **预训练 backbone**：Hunyuan3D-2.1（公开）；附录亦提供 TripoSG 作为 backone 的消融。
- **关键超参**：C=1024, M(HY3D)=4096, L=2, L′=3, γ=2, λ_dice=1, λ_cons=0.5, batch=256, epochs=10, LR=2.4e−4, 粗/细网格 128³/512³, η=0.05, δ=0.2, K=1（推理）/U{1..4}（训练）。
