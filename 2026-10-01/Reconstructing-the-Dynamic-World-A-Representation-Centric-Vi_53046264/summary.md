---
title: "Reconstructing-the-Dynamic-World-A-Representation-Centric-Vi"
source: https://arxiv.org/pdf/2609.39960v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:50:25"
field: "动态3D/4D场景重建"
keywords: ["4D Scene Reconstruction", "Neural Radiance Field", "3D Gaussian Splatting", "Dynamic Scene", "Novel View Synthesis", "Survey"]
innovations: ["以场景表征为中心的统一分类框架，系统性覆盖NeRF与3DGS两大范式的102种方法", "在多基准上提供跨方法的量化性能对比，揭示4D特征体素化和外部先验的有效性", "识别当前评测协议在时间一致性和拓扑变化评估方面的不足并提出未来方向"]
benchmarks: ["Neu3D", "D-NeRF", "NeRF-DS", "NuScenes", "Waymo Open Dataset", "NVIDIA Dynamic Scene"]
---

# 论文速读：Reconstructing-the-Dynamic-World-A-Representation-Centric-Vi

## 一句话总结
本文是一篇关于4D动态场景重建的系统性综述，以场景表征为中心，统一梳理了基于NeRF和3D Gaussian Splatting的两大类方法及其子分类，并在多个基准数据集上给出了跨方法的量化对比分析，为后续研究提供了结构化的参考框架。

## 研究问题与动机
- 4D动态场景重建面临非刚性形变、遮挡、时间不一致性以及重建保真度与计算效率之间的权衡等核心挑战。
- 近年来NeRF与3DGS在动态场景建模上快速发展，但两者的关系、底层设计选择及评测协议仍较为碎片化，缺乏统一的分类视角。
- 已有综述多聚焦静态神经渲染或仅覆盖单一方法类别（如仅NeRF或仅GS），未能系统性地同时覆盖两大范式并给出量化对比。
- 现有实验实践缺乏标准化的时间一致性、拓扑变化和长程一致性评估协议，亟需系统性梳理。

## 核心贡献（创新点）
- **以表征为中心的统一分类框架**：将4D重建方法按场景表征（隐式/显式）、时间建模策略、重建管线和优化目标进行系统分类，涵盖102种方法和22个数据集，远超 prior work（Zhu et al. 仅52种方法、10个数据集）。
- **NeRF与3DGS双范式全面对比**：首次在同一综述中系统性地覆盖4D NeRF（4类子方法）与4DGS（3类子方法），并交叉比较两类范式的性能、效率与适用场景。
- **跨数据集的量化性能评估**：在 Neu3D、D-NeRF、NeRF-DS、NuScenes 等多个基准上报告代表性方法的 PSNR/SSIM/LPIPS/CD/F-Score 等指标，提供可交叉验证的统一评测数据。
- **指出评测短板并提出未来方向**：系统识别当前实验实践在时间一致性、拓扑变化、长程一致性和流式重建评估方面的不足，并展望前馈模型、混合表征、生成先验和可交互场景等方向。

## 方法详解
**4D NeRF 四大类方法：**
- **Deformation-Field-Based**：保持一个规范（canonical）3D NeRF，通过可学习变形网络 $D_\phi(\mathbf{x}, t) \to \mathbf{x}'$ 将时空坐标映射到规范空间，再经 $F_\theta(\mathbf{x}', \mathbf{d}) \to (\mathbf{c}, \sigma)$ 输出颜色与密度。典型方法：D-NeRF、NR-NeRF、DeVRF、DyBluRF。局限：难以处理大位移和拓扑变化。
- **Implicit 4D Primitive-Based**：将时间直接作为输入维度，定义 $F_\theta(\mathbf{x}, \mathbf{d}, t) \to (\mathbf{c}, \sigma)$，无需规范空间分解。典型方法：NSFF、DynNeRF、Video-NeRF、MonoNeRF。
- **4D Feature-Volume-Based**：将4D时空域分解为低维结构（如 tri-plane、tensor decomposition、hash grid），提升效率。典型方法：HexPlane、K-Planes、MixVoxels、MSTH、Ced-NeRF。
- **Temporal-Prior-Based**：不显式参数化时间，而是通过光流、深度等外部时序约束来正则化重建，损失函数形式为 $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{flow}} + \lambda_2 \mathcal{L}_{\text{temp}}$。典型方法：StreamRF、OTNeRF、STGC-NeRF。

**4DGS 三大类方法：**
- **Explicit 4D Primitive-Based**：将时间作为第4维，用4D高斯 $G(\mathbf{x}) = \exp(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}))$ 表示场景，其中 $\boldsymbol{\Sigma} \in \mathbb{R}^{4\times4}$，通过条件分布切片得到3D高斯进行渲染。典型方法：4DGS、4D-RotorGS、SpaceTimeGS、FreeTimeGS。
- **Deformation Field-Based**：在规范空间中保持静态3D高斯，通过MLP预测位置、旋转和尺度的偏移 $(\Delta\mu, \Delta q, \Delta s) = \mathcal{F}_\theta(\mathbf{p}, t)$。典型方法：Deformable-3DGS、4D-GS、SC-GS、GaGS、MotionGS。
- **Frame-Wise Training**：在每个时间戳独立优化3D高斯参数，通过 $\mathcal{G}_t = \hat{\mathcal{G}}_t \cup \mathcal{G}_{\text{new}}$ 处理拓扑变化。典型方法：D-3DG、Street Gaussians、3DGStream、Ex4DGS、4DGC。

## 实验与结果
- **数据集**：涵盖合成（D-NeRF、ParticleNeRF、SS3DM）与真实场景（HyperNeRF、Technicolor、NuScenes、Waymo等），按单目/多视角、短/长时序、人体/通用、RGB/多模态五个维度分类。
- **渲染质量（Neu3D，NeRF-style）**：MSTH 以 PSNR 32.4、LPIPS 0.056 领先；Ced-NeRF 在 D-NeRF 基准上 PSNR 达 34.21。
- **渲染质量（Neu3D，3DGS-style）**：FreeTimeGS PSNR 33.19 最优，ADC-GS SSIM 0.981 最高，ED-3DGS LPIPS 0.037 最低。
- **几何重建（NuScenes）**：STGC-NeRF CD 0.22、F-Score 0.91、RMSE 6.54 最优；OmniRE（3DGS-style）CD 0.24、RMSE 1.89 在局部精度上更优。
- **效率对比**：3DGS 类方法渲染 FPS 显著高于 NeRF 类（如 4DRotorGS 达 277 FPS vs. NeRF 类普遍 <15 FPS），但内存占用更高；MSTH 训练仅需 0.3h，NeRFPlayer 需 6h。
- **关键趋势**：4D特征体素化方法在渲染质量上 consistently SOTA；引入外部先验（光流、深度、语义）显著提升动态区域重建效果。

## 相关工作脉络
- **Zhu et al. [49]**：最接近的综述，但仅覆盖52种方法、10个数据集，缺少统一分类和全面评测；本文覆盖102种方法、22个数据集。
- **Fan et al. [50]**：仅关注人类与动物运动，范围较窄；本文覆盖通用动态场景。
- **He et al. [51]**：聚焦自动驾驶领域的NeRF综述；本文覆盖多领域通用动态重建。
- **Cao et al. [52]** / **Zhao et al. [53]**：覆盖方法数较多但缺乏系统分类和定量评测；本文提供统一分类框架和跨方法数据对比。
- **静态NeRF/GS综述（如 [14], [25]-[28]）**：主要聚焦静态场景渲染；本文专门针对动态4D重建，强调时间建模与一致性。
- **LiDAR/多模态相关方法（如 LiDAR4D、STGC-NeRF）**：本文将其纳入评测框架，指出多模态监督在大尺度场景几何重建中的显著优势。

## 局限性与未来方向
- **自述局限**：由于部分方法未公开代码或具体配置，评测优先选取使用统一协议的方法，可能存在覆盖不全；当前缺乏标准化的时间一致性和拓扑变化评测指标。
- **前馈4D表征**：当前方法多依赖逐场景优化，计算成本高；向generalizable feed-forward模型过渡是重要趋势。
- **显隐混合表征**：纯显式方法受限于内存，纯隐式方法难以实时交互；混合设计是扩展至开放世界动态场景的关键。
- **生成先验融合**：引入生成模型实现稀疏/退化输入下的合理补全与运动预测，推动重建从观测驱动向预测/生成式框架转变。
- **可交互可控4D场景**：支持用户操纵动态场景、编辑物体行为和模拟替代场景，向交互式世界模型发展。

## 研究启发与可借鉴点
- **统一评测框架的构建方式**：本文在多个基准上采用一致的评估协议进行交叉对比，为团队在自建评测体系时提供了可借鉴的方法论——优先选择有公开代码和统一配置的方法进行公平比较。
- **外部先验对动态重建的显著增益**：光流、深度、语义等辅助信号能有效正则化高动态区域的漂浮物和视角不一致问题，这一设计思路可直接迁移到本团队的动态场景重建研究中。
- **4D特征体素化架构的有效性**：HexPlane/K-Planes等基于低维分解的特征体素化方法在多基准上 consistently 领先，其"将高维时空域分解为低维结构化表示"的设计范式值得借鉴。
- **显式-隐式混合的可行性**：论文提出混合表征是未来方向，团队可探索用显式高斯处理主体几何、隐式场处理非朗伯效应和拓扑变化的混合方案。
- **效率-质量权衡的系统量化**：本文Table 10对FPS、GPU显存、训练时间和参数量的系统对比，为后续方法的选型与消融设计提供了可复用的评估维度模板。

## 关键术语表
- **4D Scene Reconstruction**：从多视角视觉观测中恢复动态环境的几何、外观和运动随时间的变化。
- **Neural Radiance Fields (NeRF)**：用连续隐式函数将3D坐标和视角映射为颜色和密度的场景表示方法。
- **3D Gaussian Splatting (3DGS)**：用各向异性3D高斯原语显式表示场景，通过可微分瓦片光栅化实现实时渲染。
- **Canonical Space**：变形场方法中的参考（零时刻）规范空间，动态场景通过从当前时刻到规范空间的变形映射来建模。
- **Deformation Field**：学习从时空坐标到规范空间坐标的连续位移函数，用于建模非刚性形变。
- **4D Feature Volume**：将4D时空域分解为低维结构化表示（如tri-plane、hash grid），以提升训练和渲染效率。
- **Temporal Prior**：利用光流、深度等外部时序信号作为正则化约束，而非直接在辐射场中参数化时间。
- **Novel View Synthesis (NVS)**：从已知视角渲染新视角图像的评估任务，常用PSNR/SSIM/LPIPS衡量。

## 可复现要素
- **数据集**：多数基准数据集公开可用（Neu3D、D-NeRF、NuScenes、Waymo等均公开）；具体访问方式参论文Table 4。
- **代码/权重**：论文未提供统一代码库；各子方法代码分散在其原始论文中，部分方法未公开。
- **关键超参**：论文未集中列出统一超参，不同方法的训练配置差异较大；评测部分优先选用有公开配置的方法。
- **评测协议**：渲染质量（PSNR/SSIM/LPIPS）、几何重建（CD/F-Score/RMSE，阈值5cm）、效率（FPS/GPU显存/训练时间）三维度评测框架可复用。
