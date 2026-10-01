---
title: "PowerSim-Diferentiable-Physics-Simulation-and-Rendering-with"
source: https://arxiv.org/pdf/2609.38153v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:35:49"
field: "可微分物理仿真与神经渲染"
keywords: ["differentiable physics", "power diagrams", "MPM", "3D scene reconstruction", "material estimation", "neural rendering"]
innovations: ["将PowerFoam幂图原语与MPM可微分耦合，通过变形梯度极分解更新各向同性半径、偶极平面和外观方向", "从单目变形视频端到端反向传播估计空间变化的杨氏模量和泊松比", "渲染加权投票实现免训练的原语选择与场景合成"]
benchmarks: ["Vid2Sim", "GSO (Google Scanned Objects)"]
---

# 论文速读：PowerSim: Differentiable Physics Simulation and Rendering with Power Diagrams

## 一句话总结
PowerSim 将 PowerFoam 的幂图原语与材料点方法（MPM）可微分耦合，使预重建的3D场景能响应新施加的力并实时渲染，同时支持从单目变形视频中反向传播估计空间变化的材料属性（如杨氏模量和泊松比）。

## 研究问题与动机
- **现有方法缺乏显式体积与边界**：Gaussian-based 方法（如 PhysGaussian）使用高斯混合体，无法显式指定非相交材料体积、明确边界和邻接关系，导致大变形下出现模糊和伪影。
- **Mesh-based 方法与渲染脱节**：传统网格管线从显式几何开始模拟再附加外观模型，难以保留神经渲染捕获的高保真纹理细节。
- **幂图变形更新困难**：PowerFoam 提供显式边界和邻接，但移动一个站点会改变邻居单元边界，大变形可能改变邻接关系，更新几何与外观一致性极具挑战。
- **材料属性难以获取**：手动设定材料参数不具可扩展性，对从视频重建的物体更是无法获得。

## 核心贡献（创新点）
1. **可微分的 PowerFoam-MPM 耦合框架**：将每个 PowerFoam 原语视为 MPM 粒子，通过变形梯度极分解导出半径缩放、偶极平面旋转和外观方向更新，支撑大变形与分离场景——与 Gaussian-based 方法本质区别在于保持各向同性原语，将各向异性形变卸载到邻居单元几何中。
2. **从单目变形视频可微分估计空间变化材料场**：端到端微分通过 MPM 模拟→原语更新→渲染的完整链路，联合优化初始速度和逐原语杨氏模量/泊松比——区别于 PhysDreamer 依赖视频扩散先验，本文直接从观测视频反向传播。
3. **无需优化的渲染加权投票原语选择**：利用多视角 2D 分割掩码，通过线性组合权重的一次反向传播同时计算所有原语的投票得分，实现免训练的物体隔离与场景合成——相比 SemanticFoam 需要逐场景优化语义场，本方法零训练开销。
4. **动态场景中一致的反射光线追踪**：支持射线追踪反射随对象变形实时更新——扩展了 PowerFoam 的双渲染能力至动态物理场景。

## 方法详解
### 4.1 PowerSim 物理耦合原语更新
**总体流程**：给定预训练的 PowerFoam 场景 $S_0$，通过算子 $\mathcal{U}$ 将 MPM 输出的物理状态映射为原语几何/外观的变形版本：
$$\mathcal{U}: (\mathbf{p}_i^0, \mathbf{r}_i^0, \mathbf{q}_i^0, \boldsymbol{\alpha}_i^0) \mapsto (\mathbf{p}_i^t, \mathbf{r}_i^t, \mathbf{q}_i^t, \boldsymbol{\alpha}_i^t)$$

**关键更新步骤**：

1. **体积与质量**：将 MPM 背景网格划分为边长 $\Delta x$ 的体素，等分占据体积得到静止体积 $V_i^0$，质量 $m_i = \rho V_i^0$。后续体积变化由 $J = \det(\mathbf{F})$ 携带。

2. **原语中心位置**：$\mathbf{p}_i^t = \Phi^{-1}(\mathbf{x}_i^t)$，通过 MPM 坐标系逆映射恢复世界坐标。

3. **极分解分离旋转与拉伸**：
$$\mathbf{F}_i^t = \mathbf{R}_i^t \mathbf{S}_i^t, \quad \mathbf{R}_i^t = \mathbf{U}_i \mathbf{V}_i^\top$$
其中 $\mathbf{R}_i^t$ 为刚体旋转，$\mathbf{S}_i^t$ 为对称拉伸。

4. **幂半径更新（各向同性缩放）**：
$$\bar{\sigma}_i^t = (\sigma_i^{(1)}\sigma_i^{(2)}\sigma_i^{(3)})^{1/3}, \quad \mathbf{r}_i^t = \mathbf{r}_i^0 \cdot \bar{\sigma}_i^t$$
将拉伸的三个奇异值取几何平均用于标量半径缩放，各向异性形变由邻居单元几何承载。

5. **偶极平面帧更新**：
$$\mathbf{q}_i^t = \mathbf{q}_i^0 \otimes \mathrm{quat}(\mathbf{R}_i^{t\top})$$
通过极分解旋转更新四元数，保持法向与切向正交性。

6. **外观方向更新**：
$$\boldsymbol{\alpha}_i^t = \mathbf{R}_i^t \boldsymbol{\alpha}_i^0$$
世界空间Appearance方向随旋转矩阵更新，确保共旋视角下颜色不变。

7. **拓扑重建**：每步后基于原语包围球的 Cech complex 重建邻接关系。

### 4.2 材料场优化
**目标**：从视频 $\{I_t\}_{t=1}^T$ 估计逐原语材料参数 $\boldsymbol{\theta}_i = (E_i, \nu_i)$ 和初始速度 $\mathbf{v}_i^0$。

**前向流程**：
$$\hat{I}_{t+1} = (\mathcal{R} \circ \mathcal{U} \circ \mathcal{M}_{\boldsymbol{\theta}}^{N_{\mathrm{sub}}})(\mathbf{X}_t)$$

**损失函数**：
$$\mathcal{L}(\hat{I}_t, I_t) = (1-\lambda)\|\hat{I}_t - I_t\|_2^2 + \lambda(1 - \mathrm{SSIM}(\hat{I}_t, I_t)), \quad \lambda = 0.2$$

**训练策略**：
- 阶段一：优化初始速度场 30 轮（学习率 $10^{-2}$）
- 阶段二：冻结速度，优化材料场 20 轮（学习率 $5\times10^{-3}$，$E$ 初始化 $10^7$ Pa）
- 使用截断 BPTT 避免梯度爆炸/消失
- 材料参数使用 triplane 表示（3个 $24\times24$ 特征平面，32通道，MLP解码器宽64）

### 4.3 渲染加权投票原语选择
**问题**：从数千原语中识别属于特定物体的原语集合。

**投票机制**：对于多视角 2D 分割掩码 $\{M_j\}$，计算每个原语的内/外投票：
$$s_i^+ = \sum_{j \in \mathcal{I}} \sum_u w_{i,j}(u) M_j(u), \quad s_i^- = \sum_{j \in \mathcal{I}} \sum_u w_{i,j}(u)(1 - M_j(u))$$

**高效计算**：利用渲染的线性性质，一次反向传播获取所有 $N$ 个原语的投票：
$$\sum_u w_{i,j}(u) M_j(u) = \frac{\partial}{\partial f_i} \sum_u \hat{F}_j(u) M_j(u)$$

**选择标准**：$\ell_i = 1$ 当 $s_i^+ > \beta s_i^-$，$\beta = 0.5$ Discount 外部投票（掩码倾向于遗漏细结构）。

## 实验与结果
### 数据集与基准
- **Vid2Sim benchmark**（Chen et al., 2025）：12 个 GSO（Google Scanned Objects）物体，重力下落的参考视频，已知 ground-truth 材料参数
- **评估指标**：动态重建 PSNR/SSIM，材料估计 $\log_{10} E$ 和 $\nu$ 的 MAE

### 基线方法
- **PAC-NeRF**：基于 NeRF 的物理解耦表示
- **PhysDreamer**：3DGS + 视频扩散先验的材料估计

### 主要结果
**动态重建（Table 1）**：
| 方法 | 平均 PSNR (dB) | 平均 SSIM |
|------|----------------|-----------|
| PAC-NeRF | 22.06 | 0.924 |
| PhysDreamer | 19.00 | 0.908 |
| **PowerSim** | **25.73** | **0.930** |

PowerSim 在 12 个物体中 10 个达到最高 PSNR，7 个达到最高 SSIM。

**材料估计（Table 2）**：
- PowerSim $\log_{10} E$ MAE：**0.44**（PhysDreamer 0.60，降低 **27%**）
- PowerSim 在 7/12 物体上达到最低误差
- Poisson 比 $\nu$ MAE：0.16（与 PhysDreamer 持平）

### 关键定性结果
- **大变形保持细节**：拉伸/扭曲场景下保留更多表面纹理，边界更清晰（Fig. 8）
- **动态反射**：多物体合成场景下镜面反射随变形一致更新（Fig. 9）
- **消融实验**：固定偶极朝向导致锯齿边界，固定外观方向导致颜色不一致（Fig. 8）

## 相关工作脉络
1. **PhysGaussian (Xie et al., 2023)**：将 3DGS 视为 MPM 粒子进行物理模拟，但高斯核无法显式指定非相交体积，需额外正则化——PowerSim 用幂图原语替代，提供显式边界和邻接。
2. **PhysDreamer (Zhang et al., 2024)**：利用视频扩散先验估计材料属性——PowerSim 直接从观测视频反向传播，无需生成式先验。
3. **PAC-NeRF (Li et al., 2023)**： voxel-particle 耦合的物理解耦表示——在大变形下出现纹理模糊和 abrupt cut-like boundaries，PowerSim 保持 sharper boundaries。
4. **SemanticFoam (Sharafeldin et al., 2026)**：在幂图表示上优化 per-primitive 语义场——PowerSim 的免优化投票方法零训练开销即可实现同等精度（mIoU 0.80 vs 0.82）。
5. **Gaussian Grouping (Ye et al., 2024) / LabelGS (Zhang et al., 2025)**：基于 3DGS 的 3D 分割方法——需要逐场景优化或蒸馏，PowerSim 方法可一次性 batch 处理。
6. **3D Gaussian Ray Tracing (3DGRT, Moenne-Loccoz et al., 2024)**：支持高斯表示的光线追踪——PowerSim 在此基础上支持动态物理场景的次级光线效果。

## 局限性与未来方向
- **各向同性近似**：标量半径仅捕获各向同性缩放，无法完全复现连续体的拉伸和剪切；偶极帧和外观方向仅跟随局部旋转。
- **体积对应不精确**：邻居单元形状变化不保证与模拟材料体积精确匹配，更丰富的变形模型可改进此对应。
- **捕获质量依赖**：移动选定物体可能暴露捕获不佳区域，产生视差伪影（Fig. 10），需生成式先验补全。
- **材料估计歧义**：不同刚度/初速度/物理假设组合可解释相似图像运动，恢复参数需结合密度、尺度和边界条件解释。

## 研究启发与可借鉴点
1. **各向同性半径 + 邻居几何卸载各向异性**：用标量半径 + 幂图邻域几何承载变形，避免高斯原语各向异性伸长导致的伪影——此设计模式可迁移至其他网格/体素化表示与物理引擎的耦合。
2. **渲染加权投票的梯度技巧**：利用渲染组合的线性性质，一次反向传播获得所有原语投票——此技巧可推广到其他基于原语的 3D 表示的免训练分割任务。
3. **可微分 MPM→渲染端到端管道**：将物理模拟、原语更新、渲染全链路可微分化，支持从单目视频反向传播估计物理参数——此思路可应用于更多物理属性（如阻尼、塑性）的联合估计。
4. **Cech complex 动态邻接重建**：每步基于包围球重建邻接关系——对任意动态点云/原语系统的拓扑管理具有通用参考价值。
5. **Truncated BPTT 在物理模拟中的应用**：针对长时间序列物理模拟的梯度不稳定问题，截断 BPTT 是有效的工程解法。

## 关键术语表
**PowerFoam**：基于有界幂图（bounded power diagram）的 3D 场景表示，每个原语为带显式边界和邻接关系的几何-外观体素单元。
**MPM (Material Point Method)**：材料点方法，一种基于粒子-网格的连续体模拟方法，跟踪位置、速度、变形梯度等物理量。
**变形梯度 F**：MPM 中描述局部变形（旋转、拉伸、剪切）的 3×3 张量，其极分解分离刚体旋转与各向异性拉伸。
**偶极平面 (Dipole Plane)**：PowerFoam 原语中的双极性几何基元，通过法向和切向定义表面高度场和外观方向。
**幂半径**：PowerFoam 原语的标量半径，控制有界幂图单元的大小和各向同性缩放。
**Cech Complex**：基于点集包围球相交关系构建的拓扑结构，用于维护原语间的邻接关系。
**BPTT (Backpropagation Through Time)**：通过时间反向传播，用于处理序列数据的梯度计算，截断版本避免梯度爆炸/消失。
**Trirplane**：由三个正交 2D 特征平面组成的隐式场表示，查询空间坐标得到属性。

## 可复现要素
- **数据集**：Vid2Sim benchmark（GSO 物体），论文提供了多场景视频
- **代码开源**：项目网站 https://power-sim.github.io/ 提供演示，论文未明确声明 GitHub 仓库
- **权重开源**：预训练 PowerFoam 场景依赖，MPM 为开源库
- **关键超参**：
  - 损失权重 $\lambda = 0.2$
  - 折扣因子 $\beta = 0.5$
  - 材料初始化 $E = 10^7$ Pa
  - 速度优化 30 轮，lr $10^{-2}$
  - 材料优化 20 轮，lr $5\times10^{-3}$
  - Triplane: 3 个 $24\times24$ 平面，32 通道，MLP 宽 64
  - 网格体素边长 $\Delta x$（论文未明确）
