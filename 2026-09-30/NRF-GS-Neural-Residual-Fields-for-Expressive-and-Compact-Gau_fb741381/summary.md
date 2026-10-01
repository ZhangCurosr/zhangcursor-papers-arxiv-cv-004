---
title: "NRF-GS-Neural-Residual-Fields-for-Expressive-and-Compact-Gau"
source: https://arxiv.org/pdf/2609.37115v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:45:04"
field: "神经辐射场与显式场景表示"
keywords: ["3D Gaussian Splatting", "Neural Residual Fields", "View-dependent Appearance", "Novel View Synthesis", "Spherical Harmonics", "Neural Radiance Fields"]
innovations: ["用共享神经残差场替代每高斯低阶SH，实现解耦的外观与几何表示", "频率分解+可学习调制权重，平衡高频细节与低频稳定性避免过拟合", "证明更强的外观表达力可直接减少高斯数量约50%而不损失渲染质量"]
benchmarks: ["MipNeRF360", "DL3DV", "Tanks and Temples"]
---

# 论文速读：NRF-GS: Neural Residual Fields for Expressive and Compact Gaussian Splatting

## 一句话总结
NRF-GS 提出将标准 3DGS 中每个高斯的低阶球谐函数（SH）替换为一个**共享的神经残差场**，该场以观察方向、相机-高斯距离和每高斯潜在特征为条件，预测视角相关的外观残差；该方法通过更准确地建模高频方向反射（尤其是镜面区域），使高斯表示更具表达力，从而在渲染质量相当或更优的前提下将高斯数量减少约 50%。

## 研究问题与动机
- 标准 3DGS 使用低阶球谐函数（通常 degree-3）以每个高斯为单位独立编码视角相关反射，其表达能力有限，难以捕获高频方向效应（如镜面高光）。
- 为弥补表达力不足，现有方法被迫增加高斯数量以"堆叠"覆盖高频信息，导致几何冗余和更高的存储开销。
- 已有神经外观改进工作（如 VDGS、GSNB）将几何属性与神经外观表示强耦合，或在 higher-frequency 建模时显存开销巨大，缺乏在斑点级进行紧凑、可扩展的外观建模的系统性设计。
- 核心动机：重新审视外观建模在 3DGS 中的作用，证明**更强的视角相关反射建模可以直接转化为更紧凑的几何表示**（更少的高斯），而非单纯提升渲染质量。

## 核心贡献（创新点）
- **提出 NRF-GS 混合表示**：将每个高斯的独立 SH 基替换为一个场景级共享神经残差场，以低维潜在特征+观察方向+距离为条件预测残差颜色和透明度；与 VDGS/GSNB 的本质区别在于**解耦了几何属性与神经外观**，且不依赖额外的显式基函数。
- **频率分解网络设计**：将方向编码分为低频和高频分量，分别由独立 MLP 分支处理，并引入**可学习的调制权重** $w_i$ 控制高频贡献；与 GSNB 等直接提高基函数阶数的本质区别在于**选择性建模高频**，避免无界场景过拟合。
- **严格的参数预算对照实验**：在匹配参数预算的条件下，定量证明特征+MLP 表述比低阶解析球谐函数表达能力更强；这填补了已有工作缺乏统一比较基准的空白。
- **证明外观表达力与几何紧凑性的直接关联**：更好的共享外观建模使每个高斯更"高效"，从而在高斯数量减少最多 50% 的同时保持或提升渲染质量。

## 方法详解

**整体架构**：NRF-GS 在标准 3DGS 管线基础上仅替换外观建模部分，几何表示（位置 $\mathbf{x}_i$、协方差 $\Sigma_i$）和渲染光栅化保持不变。

**每高斯属性**：$\mathcal{G}_i = \{\mathbf{x}_i, \pmb{\Sigma}_i, \alpha_i^{\mathrm{base}}, \mathbf{c}_i^{\mathrm{base}}, \mathbf{z}_i\}$，其中 $\mathbf{z}_i \in \mathbb{R}^{32}$ 为每高斯学习到的潜在外观特征，$\mathbf{c}_i^{\mathrm{base}}$ 和 $\alpha_i^{\mathrm{base}}$ 为漫反射基础颜色和基础透明度（$l=0$ 直流分量）。

**方向编码**：对每个高斯计算观察方向 $\mathbf{d}_i = (\mathbf{c}_{cam} - \mathbf{x}_i) / r_i$ 和标量距离特征 $\tilde{r}_i = \log(\max(r_i, \epsilon))$；方向使用 Fourier 编码（K=3 频段）：
$$\gamma(\mathbf{d}_i) = \left[\mathbf{d}_i,\ [\sin(2^k\pi\mathbf{d}_i),\ \cos(2^k\pi\mathbf{d}_i)]_{k=0}^{K-1}\right]$$

**频率分解**：将编码 $\gamma(\mathbf{d}_i)$ 拆分为低频 $\gamma_{\mathrm{low}}(\mathbf{d}_i)$（原始方向 + $k=0$）和高频 $\gamma_{\mathrm{high}}(\mathbf{d}_i)$（其余分量）。

**低频分支** $f_{\mathrm{low}}$：输入 $(\mathbf{z}_i, \mathbf{c}_i^{\mathrm{base}}, \alpha_i^{\mathrm{base}}, \tilde{r}_i, \gamma_{\mathrm{low}}(\mathbf{d}_i))$，输出低频颜色残差 $\Delta\mathbf{c}_i^{\mathrm{low}}$ 和中间透明度残差 $\delta_i^{\alpha}$：
$$(\Delta\mathbf{c}_i^{\mathrm{low}},\ \delta_i^{\alpha}) = f_{\mathrm{low}}(\cdot)$$

**高频分支** $f_{\mathrm{high}}$：输入 $(\mathbf{z}_i, \mathbf{c}_i^{\mathrm{base}}, \tilde{r}_i, \gamma_{\mathrm{high}}(\mathbf{d}_i))$，输出高频颜色残差 $\Delta\mathbf{c}_i^{\mathrm{high}}$ 和标量调制权重 $w_i$：
$$(\Delta\mathbf{c}_i^{\mathrm{high}},\ w_i) = f_{\mathrm{high}}(\cdot)$$

**残差合成**（有界更新，确保训练稳定）：
$$\Delta\mathbf{c}_i = s_c \cdot \tanh\left(\Delta\mathbf{c}_i^{\mathrm{low}} + \sigma(w_i)\cdot\Delta\mathbf{c}_i^{\mathrm{high}}\right)$$
$$\alpha_i = \alpha_i^{\mathrm{base}} + s_\alpha \cdot \tanh(\delta_i^{\alpha})$$
其中 $\sigma(\cdot)$ 为 sigmoid，$s_c = s_\alpha = 0.2$ 为有界缩放超参；最终颜色 $\mathbf{c}_i = \mathbf{c}_i^{\mathrm{base}} + \Delta\mathbf{c}_i$，再经标准前向累积光栅化。

**实现细节**：使用 tiny-cuda-nn 的 FullyFusedMLP，两个分支各含 64 隐层神经元 + ReLU；每次视图训练前做一次短时 warmup（残差场初始预测为零残差），以避免早期过拟合方向噪声。

## 实验与结果

**数据集**：Mip-NeRF360（9 场景）、DL3DV（9 场景）、Tanks and Temples（19 场景）；另用 Synthetic BTF 数据和真实 BTF 数据集 [25] 做表达力验证。

**基线**：3DGS、VDGS [17]、GSNB [34]。

**主要结果**（Tab. 3 汇总数据）：

| 数据集 | 方法 | Points(M)↓ | PSNR↑ | MS-SSIM↑ |
|---|---|---|---|---|
| MipNeRF360 | 3DGS | 3.359 | 27.40 | 0.813 |
| MipNeRF360 | VDGS | 3.548 | 27.65 | 0.813 |
| MipNeRF360 | **NRF-GS** | **1.656** | **27.86** | **0.814** |
| DL3DV | **NRF-GS** | **0.595** | **30.82** | **0.925** |
| TnT | **NRF-GS** | **1.040** | **24.68** | **0.845** |

- 在三个数据集上均取得**最高 PSNR**，高斯数量分别减少约 **51%（MipNeRF360）**、**49%（DL3DV）**、**44%（TnT）**，即约 40–55% 的整体降幅。
- 最强结果：DL3DV 上 NRF-GS PSNR 30.82 vs 3DGS 29.48，提升 **+1.34 dB**，且点数量仅为 3DGS 的 51%。
- 训练速度：NRF-GS 介于 3DGS 和 VDGS/GSNB 之间；推理 FPS 105（vs 3DGS 115，VDGS 65，GSNB 92）。

**消融**（MipNeRF360，Tab. 1）：去除频率分解（w/o split）PSNR 下降至 27.38；去除透明度残差（w/o opacity）降至 27.51；去除距离条件（w/o distance）降至 27.65，三项均证实各自贡献。

**表达力验证**（Tab. 2）：在匹配参数预算下，NRF 在合成 data（21.1→29.0 dB）和真实 BTF（27.8→28.8 dB，像素级）上均显著优于 degree-3 SH。

## 相关工作脉络
- **3DGS [12]**：本文基础，使用低阶 SH 进行每高斯视角相关颜色编码；NRF-GS 在其基础上改进外观建模而不改变几何管线。
- **VDGS [17]**：将 NeRF-style MLP 与显式高斯结合，预测视角相关颜色和透明度；区别在于 VDGS 直接使用原始高斯均值和相机位置作为输入，NRF-GS 采用解耦的潜在特征和频率分解设计。
- **GSNB [34]**：用额外学习的神经基函数扩展 SH 表达能力；区别在于 GSNB 参数开销大（显存敏感，部分高分辨率场景 OOM），NRF-GS 通过共享残差场实现更紧凑的表示。
- **SG-splatting [29] / ARS-GS [31]**：用 Spherical Gaussian 替代 SH；仍是解析基函数，对高频方向反射建模能力有限（等效 degree-3 SH）。
- **Latent-SpecGS [30]**：引入通用潜在神经描述符，CNN 解码为漫反射/ specular 图像；区别在于其在图像空间进行分解与解码，NRF-GS 完全在斑点级别操作，与标准 splatting 管线兼容。
- **Feature-3DGS [33]**：添加与 2D 基础模型对齐的语义特征嵌入；用途侧重下游任务，外观建模仍基于 SH，NRF-GS 关注的是外观表达力与几何紧凑性的关系。

## 局限性与未来方向
- 引入每高斯神经评估的计算开销，导致训练和推理略慢于纯解析方法（3DGS）。
- 继承标准 3DGS 对精确相机位姿和初始化质量的依赖。
- 未显式优化运行时内存占用（除减少高斯数量外）。
- **未来方向**：联合优化外观建模与剪枝/压缩策略；探索更高效的神经评估方法；将学习到的每高斯外观特征扩展至下游任务（如分割、编辑）。

## 研究启发与可借鉴点
- **"外观建模驱动几何紧凑性"这一核心洞察**可直接迁移：在任意基于 primitive 的渲染系统中，若外观表达能力不足，往往需要冗余 primitive 补偿——改进外观建模本身可能带来几何压缩收益。
- **频率分解 + 可学习调制权重**的设计思路适用于任何需要平衡低频稳定性和高频细节的场景，可推广至其他神经渲染管线中的视角相关建模。
- **有界残差更新**（$\tanh$ + 缩放系数）是保证神经外观模块与现有解析管线平稳集成的实用技巧，避免训练不稳定，值得在其他 hybrid 方法中借鉴。
- **距离条件化** $\tilde{r}_i$ 的引入是一个简洁而有效的设计，隐含编码了视差强度与观察尺度的关系，可考虑迁移到其他稀疏表示场景。
- 可与本团队在**神经辐射场压缩**或**下游语义任务**的方向结合：每高斯 32 维潜在特征 $\mathbf{z}_i$ 天然携带语义可分离的外观信息，是连接显式几何与语义嵌入的轻量桥梁。

## 关键术语表
- **NRF-GS（Neural Residual Fields for Gaussian Splatting）**：一种混合表示，用共享神经残差场替代每高斯 SH，以潜在特征为条件预测视角相关外观残差。
- **Spherical Harmonics (SH)**：球谐函数，用于在单位球面上参数化视角相关反射的低阶解析基函数。
- **Neural Residual Field (NRF)**：场景级共享轻量 MLP，预测每个高斯相对于基础颜色的视角相关残差。
- **Frequency decomposition**：将 Fourier 方向编码拆分为低频和高频分量，分别由独立 MLP 分支处理以避免过拟合。
- **Modulation weight ($w_i$)**：每个高斯的可学习标量，经 sigmoid 后控制高频分支对最终颜色的贡献强度。
- **Latent feature ($\mathbf{z}_i$)**：每高斯学习的 32 维潜在外观向量，编码与视角无关的 Appearance 属性供共享 MLP 使用。
- **Differentiable splatting**：可微分的斑点光栅化过程，将 3D 高斯投影到 2D 图像并进行前向累积合成。
- **BTF (Bidirectional Texture Function)**：双向纹理函数，描述表面在给定入射/出射方向下的反射特性，用于评估外观建模的真实世界表达力。

## 可复现要素
- **数据集**：Mip-NeRF360、DL3DV、Tanks and Temples（均通过 NerfBaselines [14] 包获取，非完全公开但广泛可用）；BTF 数据集 [25] 公开可用。
- **代码/权重**：论文未提及开源仓库；tiny-cuda-nn [20] 为依赖库。
- **关键超参**：Fourier 频段数 $K=3$；每高斯潜在特征维度 $D=32$；两个 MLP 分支各 64 隐层神经元 + ReLU；$s_c = s_\alpha = 0.2$；warmup 策略每视图一次。
