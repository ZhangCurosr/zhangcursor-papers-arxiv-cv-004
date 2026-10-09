---
title: "PAM-ToD-Plug-and-Play-Appearance-Modeling-for-Cross-Time-of"
source: https://arxiv.org/pdf/2610.11572v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 09:52:07"
field: "3D 视觉与神经渲染"
keywords: ["3D Gaussian Splatting", "跨时段外观适配", "逆渲染", "自动驾驶仿真", "CARLA-ToD", "少样本重光照"]
innovations: ["基于消去反照率的乘性-加性外观传输，直接适配冻结 3DGS", "层次化全局-空间场-残差参数配合容量匹配正则实现少锚定稳健优化", "构建几何/位姿/动态轨迹一致的 CARLA-ToD 跨时段评测基准"]
benchmarks: ["CARLA-ToD Static", "CARLA-ToD Dynamic", "Waymo Open Dataset"]
---

# 论文速读：PAM-ToD: Plug-and-Play Appearance Modeling for Cross-Time-of-Day 3D Gaussian Splatting

## 一句话总结
本文提出 PAM-ToD，一种即插即用的轻量外观适配插件，能够在冻结预训练 3D Gaussian Splatting (3DGS) 参数的前提下，仅用少量目标时段锚定图像学习跨时段外观变换；同时构建了控制严密的 CARLA-ToD 基准数据集，支持静态与动态场景的跨时段评估。

## 研究问题与动机
- 自动驾驶仿真器需要在不同时间段渲染同一道路场景，但已有 3DGS 重建的外观固定在采集时的光照条件下，重采/重训每个时段成本过高。
- 逆渲染方法在户外场景、单时段捕获数据下难以稳定分解材质与光照，且通常需要重建独立的可重光照表示，不能直接复用已有 3DGS。
- 扩散-based 重光照方法缺乏共享 3D 表示，难以保证多视角一致性，且多步去噪推理速度远低于 3DGS 实时渲染。
- 需要一种方法：复用已训练 3DGS、仅需少量目标时段有姿态的锚定图像、保持 3D 一致性与实时渲染。

## 核心贡献（创新点）
- 将"跨时段场景迁移"形式化为在冻结预训练道路场景 3DGS 上、基于少量目标时段有姿态锚定图像的外观适配任务。
- 提出 PAM-ToD 插件，通过消除时间不变反照率后得到的乘性光照比与加性发射项直接建模源-目标外观关系，避免显式反求材质与光照。
- 层次化参数化（全局增益 + 多尺度空间网格 + 逐高斯残差）配合与容量匹配的平滑/稀疏正则，仅在少数锚图下仍可实现稳定优化。
- 构建 CARLA-ToD 基准：在同一几何、相机位姿与动态对象轨迹下提供三个时段（Morning/Noon/Night），区分静态与动态两种配置。
- 在 CARLA-ToD 与 Waymo 开放数据集上验证，PAM-ToD 在保持实时渲染的同时显著提升跨时段渲染质量。

## 方法详解
- 问题设定与基础符号：源场景包含 $M$ 个 3D 高斯 $G = \{(\mu_i, q_i, \sigma_i, \alpha_i, c_i(s))\}_{i=1}^M$ 与天空模型 $c_s^{\mathrm{sky}}(\omega)$；所有高斯参数与天空模型固定，仅在渲染时应用颜色修正。每个高斯携带由 SegFormer 提升得到的类概率向量 $f_i$ 与类别 $\kappa(i)$。
- 外观传输公式：在简化漫反射成像模型 $c_i(t) = \rho_i \odot L_i(t) + E_i(t)$ 下，消去时间不变反照率 $\rho_i$ 得到：
  $$c_i(t) = c_i(s) \odot \frac{L_i(t)}{L_i(s)} + E_i(t) - E_i(s) \odot \frac{L_i(t)}{L_i(s)}.$$
  该式将目标颜色表达为源颜色的乘性缩放与加性校正之和，无需分别恢复材质与光照。
- 渲染时修正：对球谐评估后的 view-dependent RGB $c_i(s)$ 做：
  $$\tilde{c}_i = \mathrm{clamp}_{[0,1]}\big(c_i(s) \odot e^{\Delta_i} + E_i\big),$$
  其中 $e^{\Delta_i}$ 负责变暗与色彩温度变化，$E_i \ge 0$ 负责新增光源等额外亮度。
- 对数增益层次化参数：$\Delta_i = (g^t - g^s) + \sum_{l=1}^3 [\mathrm{trilerp}(G_l^t, x_i) - \mathrm{trilerp}(G_l^s, x_i)] + r_i$，含全局增益 $g \in \mathbb{R}^3$、三级多尺度网格 $n_l \in \{16, 32, 64\}$ 的空间场与逐高斯残差 $r_i$。
- 发射项参数化：$E_i = \mathrm{softplus}(b_i)\hat{c}_E$，$b_i$ 为逐高斯亮度，$\hat{c}_E$ 为场景共享的色度（单位均值正向量）；初始化为可忽略发射，使插件初始行为等价于源渲染。
- 天空修正：对冻结天空模型施加全局仿射变换 $c_t^{\mathrm{sky}}(\omega) = \mathrm{clamp}(e^{g_{\mathrm{sky}}} \odot c_s^{\mathrm{sky}}(\omega) + b_{\mathrm{sky}})$，$g_{\mathrm{sky}}, b_{\mathrm{sky}} \in \mathbb{R}^3$。
- 优化目标：$\mathcal{L} = \lambda_{\mathrm{anc}}\mathcal{L}_{\mathrm{anchor}} + \lambda_{\mathrm{atp}}\mathcal{L}_{\mathrm{atp}} + \lambda_{\mathrm{unseen}}\mathcal{L}_{\mathrm{unseen}} + \mathcal{L}_{\mathrm{chroma}} + \mathcal{L}_{\mathrm{reg}}$。
- 鲁棒锚点重建损失：对渲染 $\hat{I}$ 与锚图 $A$ 做高斯平滑后计算 L1+SSIM，并剔除通道平均绝对误差最大的比例 $\rho=10\%$ 像素，以容忍生成型伪锚图中的不一致细节。
- 外观传输先验 $\mathcal{L}_{\mathrm{atp}}$：对每类按 source DC 分量的亮度将锚图像素与同类高斯各分 $Q=32$ 个等频分位数箱，令高斯的伪目标为与其亮度分位相同的锚图像素均值，约束增益项而非发射项。
- 可见到不可见传输先验 $\mathcal{L}_{\mathrm{unseen}}$：以杆/信号灯高斯为人工光源代理，按类别与距离分箱后，取可见集合每箱的 $\Delta_i$ 通道中位数，训练中后半段将不可见集合拉向对应中位数（目标 stop-gradient），实现单向传播。
- 色度正则 $\mathcal{L}_{\mathrm{chroma}}$：对 $\mathrm{ch}(v)=v-\frac{1}{3}\sum_c v_c$ 惩罚残差、不可见传播以及均匀纹理类（道路/人行道）的色度漂移，抑制平坦表面的彩色斑块。
- 结构正则：多尺度网格的加权总变差 $\mathcal{R}_{\mathrm{tv}}$、语义 k-NN 残差平滑 $\mathcal{R}_{\mathrm{knn}}$（边权 $w_{ij}=\max(0,\cos(f_i,f_j))$）、发射项 $\ell_1$ 稀疏 $\mathcal{R}_E$。
- 优化设置：Adam，LR $10^{-2}$，共 8000 次迭代；$\mathcal{L}_{\mathrm{unseen}}$ 与 $\mathcal{L}_{\mathrm{chroma}}$ 的第二项在第 4000 步启用；损失权重为 $\lambda_{\mathrm{anc}}=20,\lambda_{\mathrm{ssim}}=0.2,\lambda_{\mathrm{atp}}=1,\lambda_{\mathrm{unseen}}=0.1,\lambda_{\mathrm{rch}}=1,\lambda_{\mathrm{ch}}=1,\lambda_{\mathrm{cha}}=2$，正则权重 $\lambda_{\mathrm{tv}}=10^{-2},\lambda_{\mathrm{knn}}=10^{-2},\lambda_E=10^{-4}$。

## 实验与结果
- 数据集与设置：CARLA-ToD（24 序列，3 相机 x 200 帧，Morning/Noon/Night，静态与动态两种配置）与 Waymo Open Dataset（WOD）；评估包括 Reconstruction 与 Novel View Synthesis（NVS）。
- 主要基线：GS-W、GS-IR、GI-GS、LumiGauss、StreetGS；扩散基线 LumiNet、DiffusionRenderer、UniRelight；PAM-ToD 以 StreetGS 为基础模型，使用 $K=1$ 与 $K=8$ 同步锚定捕获。
- CARLA-ToD Static 最佳结果（NVS，$K=8$）：PSNR 25.45 / SSIM 0.814 / LPIPS 0.247，FPS 53.4；相较 GS-W 最高提升约 +8.87 dB PSNR、+0.085 SSIM、-0.077 LPIPS；$K=1$ 仍能显著超越所有基线。
- CARLA-ToD Dynamic 最佳结果（NVS，$K=8$）：PSNR 24.45 / SSIM 0.791 / LPIPS 0.263，FPS 49.4；远超扩散基线（0.07–0.08 FPS）。
- 消融：锚点数越多指标单调提升，约在 $K=8$ 饱和；移除乘性项影响最大，加性项与层次结构均必要；ATP 与色度正则明显改善感知质量；模型增量仅 +7.7M 参数、+31 MB。
- 真实场景迁移：在 WOD 上配合 Image Edit API 合成锚图，PAM-ToD 的 $\Delta E_{00}$ 2.69 vs UniRelight 8.10、$\Delta L^*$ 1.78 vs 5.26、$\mathrm{EMD}_L$ 1.10 vs 4.01。

## 相关工作脉络
- GS-IR / GI-GS / LumiGauss：逆渲染类，显式分解材质与光照并需重建独立表示；本文不分解材质-光照，直接学习乘性-加性修正，复用冻结 3DGS。
- GS-W / WildGaussians：外观嵌入类，从多条件图像集中学习外观变化，跨时段显式可控性弱；本文仅需少量目标时段锚图完成适配。
- GaRe / OSDR-GS / UrbanIR / LightSim：面向户外的逆渲染或单视频重建；本文面向已有 3DGS 的 plug-and-play 跨时段适配，降低存储与计算开销。
- UniRelight / LumiNet / DiffusionRenderer：2D/视频扩散重光照；缺乏共享 3D 表示导致多视角不一致，且推理慢；本文保持 3D 一致性与实时渲染。
- OmniRe / DrivingGaussian：驾驶场景 3DGS 重建；本文在其基础模型之上附加轻量插件实现跨时段迁移，不需重训主模型。

## 局限性与未来方向
- 需要目标时段已标定姿态的锚定图像，不支持无姿态/弱姿态适配。
- 难以准确建模镜面反射、锐利阴影边界与物理一致的全程阴影。
- 依赖 SegFormer 语义先验与少量人工光源代理，极端光照条件下可见-不可见传播可能不稳定。
- 未来方向：引入物理阴影先验与更精确光源几何，扩展到强光照极端场景与无姿态锚图自适应。

## 研究启发与可借鉴点
- 基于成像模型的代数消元（消去时间不变反照率）可直接导出简洁的乘性-加性修正结构，避免昂贵的反向渲染求解。
- 层次化参数（全局+空间场+残差）配合"容量-正则同步增强"是少样本 3D 外观适配的有效范式。
- 鲁棒锚图损失（高斯平滑+顶部 $\rho$ 像素剔除）对生成模型伪锚图或存在细微视差异常的标注具有较强容错。
- 按亮度分位的伪目标构造（ATP 先验）可在无逐像素对应时建立稳定的跨时段语义-光度对齐。
- 可见到不可见的中位数传播 + stop-gradient 单向正则，避免污染已拟合的可见区域，适合稀疏监督下的场外泛化。

## 关键术语表
- **PAM-ToD**：即插即用外观建模插件，在冻结 3DGS 基础上学习跨时段乘性-加性颜色修正。
- **Cross-Time Scene Transfer**：用少量目标时段有姿态图像，将已训练道路场景 3DGS 的外观迁移到另一时段。
- **3D Gaussian Splatting (3DGS)**：基于 3D 高斯原语的高效实时神经辐射场渲染方法。
- **CARLA-ToD**：在 CARLA 仿真器中构建的跨时段驾驶场景数据集，几何、位姿与动态轨迹一致，仅光照变化。
- **Appearance Transport Prior (ATP)**：按类别与亮度分位将源高斯颜色映射到锚图像素统计伪目标的对齐先验。
- **Seen-to-Unseen Transport Prior**：将可见高斯的增益中位数按类别与距离代理单向传播到不可见高斯。
- **Chromaticity Regularization**：惩罚对数增益的色度分量变化，抑制平坦表面的色偏与色块。
- **Semantic k-NN Graph**：基于 SegFormer 类概率余弦权重的残差平滑图，跨语义边界抑制平滑。

## 可复现要素
- 数据集：CARLA-ToD（论文构建并随代码发布）；Waymo Open Dataset（公开）。
- 代码/权重：论文未明确给出开源链接声明，但提供了完整实现细节与超参。
- 关键超参：Adam LR $10^{-2}$，8000 迭代；$\lambda_{\mathrm{anc}}=20,\lambda_{\mathrm{ssim}}=0.2,\lambda_{\mathrm{atp}}=1,\lambda_{\mathrm{unseen}}=0.1$；$\lambda_{\mathrm{rch}}=1,\lambda_{\mathrm{ch}}=1,\lambda_{\mathrm{cha}}=2$；$\lambda_{\mathrm{tv}}=10^{-2},\lambda_{\mathrm{knn}}=10^{-2},\lambda_E=10^{-4}$；网格分辨率 $n_l \in \{16,32,64\}$；$\sigma=2,\rho=10\%,Q=32$；k=8 邻域；发射初始 $b_i=-6,\hat{c}_E=\mathbf{1}$；可见-不可见传播从中点（第 4000 步）启用，每 200 步更新中位数。
