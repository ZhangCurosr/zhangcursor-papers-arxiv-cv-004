---
title: "Pose-Free-Feed-Forward-3D-Inpainting-via-Learnable-Mask-Atte"
source: https://arxiv.org/pdf/2610.11857v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 09:53:26"
field: "3D 视觉与生成"
keywords: ["3D scene inpainting", "pose-free reconstruction", "feed-forward 3D foundation model", "learnable mask attention", "support token refinement", "3D Gaussian splatting", "multi-view consistency"]
innovations: ["提出免姿态 feed-forward 3D 场景修补框架 FreeInpaint，联合估计相机位姿与几何并补全掩码区域", "设计可学习掩码注意力机制，在跨视图 transformer 中动态调节掩码 token 参与程度以保护空间锚定", "提出推理时支持 token 渐进精炼策略，以置信度加权方式注入扩散生成辅助证据修补盲点"]
benchmarks: ["SPIn-NeRF", "360-USID", "LLFF", "DL3DV-10K", "GS25", "Co3D"]
---

# 论文速读：FreeInpaint - Pose-Free Feed-Forward 3D Inpainting via Learnable Mask Attention and Support Token Refinement

## 一句话总结
本文提出 FreeInpaint，一种免姿态的 feed-forward 3D 场景修补框架，直接从含掩码区域的无姿态多视图图像中联合估计相机位姿、恢复几何并生成高质量的 3D 一致场景；核心创新为可学习掩码注意力机制与支持 token 渐进精炼策略，在多个基准上以 ~0.4s 推理速度超越现有耗时数分钟至数小时的方法。

## 研究问题与动机
- **现有 3D 场景修补方法强依赖精确标定相机姿态**：主流工作需先通过 SfM（如 COLMAP）预计算姿态，这一多阶段流程耗时且难以适用于 Casual in-the-wild 捕获场景。
- **SfM 在真实场景下脆弱**：稀疏视图、大面积无纹理区域或被掩码遮挡时，SfM 容易失败或产生误差，导致下游修补质量劣化并引入伪影。
- **将 3D 基础模型直接适配掩码输入存在双重挑战**：（1）掩码区域会破坏多视图对应关系，污染 pose 估计与几何恢复的空间锚定；（2）单次前向传播在严重遮挡（盲点）区域缺乏足够外观证据，易产生模糊。

## 核心贡献（创新点）
1. **提出 FreeInpaint 免姿态 feed-forward 3D 修补框架**：在不依赖预计算相机姿态的前提下，联合完成位姿估计、几何恢复与多视图一致的 3D 高斯场景修补；与先前优化型方法的本质区别在于消除 SfM 预处理，实现端到端单次前向推理。
2. **引入 Learnable Mask Attention 机制**：在 DA3 跨视图注意力层中，以输入二值掩码初始化可学习软掩码偏置，并通过逐层残差网络动态调节不同深度层中掩码 token 参与跨视图推理的程度；相比硬掩码永久屏蔽或启发式动态掩码，该设计在抑制无效信息污染的同时保留可演化的上下文聚合能力。
3. **设计 Support Token Refinement 推理时精炼策略**：基于渲染不透明度识别盲点并选取锚点视图，利用 SDEdit + IP-Adapter 的扩散先验生成支持视图，再以置信度加权方式注入辅助 token 渐进修补盲区；与单次前向推断相比，可显著提升严重遮挡区域的高保真度。
4. **提出 Depth-Guided Self-Distillation 自蒸馏训练策略**：冻结预训练 DA3 作为教师网络，在掩码区域约束学生网络的深度预测与教师网络的伪 GT 深度一致，从而为补全几何提供明确的空间先验，避免仅凭光度损失导致的纹理幻视与几何塌陷。

## 方法详解
- **基础架构**：以 Depth Anything 3 (DA3-Giant) 为骨干，采用 DINOv2 ViT 编码器，每视图前置可学习相机占位 token，经交替内帧/跨帧自注意力聚合多视图一致性；输出头包括 Dual-DPT（深度与射线图）与 GS-DPT（3D 高斯参数：中心、不透明度、旋转、缩放、颜色）。
- **Learnable Mask Attention**：将源视图二值掩码下采样至 patch 级 token 向量 $\tilde{M}$，初始化为 logits 空间软掩码 $P^{(0)} = \text{logit}(\epsilon + (1-2\epsilon)\tilde{M})$，逐层更新 $P^{(l)} = P^{(l-1)} + \alpha \cdot g_l(F^{(l)})$，得到抑制偏置 $B^{(l)} = -\beta \cdot \sigma(P^{(l)})$，注入跨视图注意力 logits：$\text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}} + B^{(l)}\right)V$；$g_l$ 为轻量卷积网络（3×3 + DW3×3 + 1×1），零初始化保证初始行为硬掩码先验。
- **Support Token Refinement（推理时）**：
  1. **盲点检测**：将初始 3D 高斯场渲染回源视图，计算掩码区域可见度 $V_i = \sum(\alpha_{\text{render}}^i \odot M^i) / \sum M^i$，选取最低可见度视图为锚点。
  2. **支持视图生成**：以当前渲染图 $I_{\text{render}}$ 为画布做 SDEdit（注入比例噪声保留空间几何），并以参考视图 $I_{\text{ref}}$ 通过 IP-Adapter 注入纹理先验，生成兼顾几何与外观一致性的支持视图。
  3. **置信度加权注入**：依据锚点掩码与渲染不透明度计算支撑置信度 $C^{\text{sup}} = \text{Norm}(\mathcal{G}_\sigma(M^a \odot (1-\alpha_{\text{render}}^a)^\gamma))$，对生成的支持 token 进行调制，盲点高置信度被强化 attend，已解释区域低置信度被抑制。
- **训练目标**：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}}$；其中 $\mathcal{L}_{\text{rgb}} = \mathcal{L}_{\text{mse}} + \lambda_{\text{lpips}}\mathcal{L}_{\text{lpips}}$ 仅在预留新视图上计算；$\mathcal{L}_{\text{depth}} = \|D_{\text{pred}} \odot M - D_{\text{prior}} \odot M\|_1$ 在所有输入与新视图上约束几何一致性。
- **掩码生成策略**：训练时混合使用随机 Box/Brush 掩码（模拟任意遮挡）与几何掩码（由预训练 DA3 估计的深度将参考视图掩码投影到源视图，比例 0.4:0.6）。
- **两阶段微调**：第一阶段冻结 backbone，仅微调 DPT/GS-DPT 头；第二阶段联合优化可学习掩码注意力与跨视图注意力层；输入分辨率 280×504，8×A6000 GPU，batch size=16。

## 实验与结果
- **数据集**：训练使用 DL3DV-10K；评估使用 SPIn-NeRF（前向场景 object removal）、360-USID（360° 无界场景）、LLFF（真实稀疏视图）。
- **与逐场景优化方法对比（Tab. 1）**：在 SPIn-NeRF 上 PSNR=17.79、LPIPS=0.2819；360-USID 上 PSNR=18.58、LPIPS=0.2526；LLFF 上 C-KID=0.5613、C-FID=343.97；推理时间仅 ~0.4s（强遮挡时支持精炼最多触发 3 次，约 1.8s），相比 SPIn-NeRF（~5h）、NeRFiller（~30m）、GScream（~2h）等速度快 4~3 个数量级。
- **与免姿态 feed-forward 基线对比（Tab. 2）**：构建 LAMA+DA3 和 MVInpainter+DA3 两级联 baselines，FreeInpaint 在全部指标上均取得最优，显著避免串联式 2D 修补 + 3D 重建中的误差累积与多视图不一致。
- **物体替换任务（Tab. 3）**：在 CLIPdir、C-KID、C-FID 三项指标上均优于 DiGA3D、NeRFiller、InFusion。
- **消融（Tab. 4）**：Baseline → +Fine-Tuning → +Mask Attn → +Token Refine，PSNR 从 15.58 逐步提升至 18.58，LPIPS 从 0.7399 降至 0.2526，FID 从 392.83 降至 199.61。
- **掩码注意力策略对比（Tab. 5/11）**：Learnable mask 在 PSNR、LPIPS 及 ATE/RPE 位姿误差上全面优于 No mask、Hard mask、MAT-style、MLLAM-style。
- **视图可扩展性（Tab. 6）**：输入视图从 4 增至 32，SPIn-NeRF PSNR 稳定上升（17.79→18.47），360-USID 同样呈现单调改善，且感知指标保持稳定。
- **参考视图敏感性（Tab. 7）**：在不同扩散种子与不同参考视角选择下，PSNR/LPIPS/FID 波动有限，展现对参考输入的稳健性。

## 相关工作脉络
- **3D 场景修补（参考类）**：SPIn-NeRF、NeRFiller、GScream、AuraFusion360、InFusion、InstaInpaint 等多依赖逐场景优化与精确相机姿态；FreeInpaint 的定位差异在于免姿态 feed-forward 一次性推理。
- **非参考类 3D 修补**：DiGA3D 等方法利用 2D 扩散先验 + SDS 迭代优化，缺乏多视图参考约束且耗时；FreeInpaint 以单参考视图为条件直接 propagate 纹理与几何。
- **Feed-forward 3D 基础模型**：DUSt3R、VGGT、Depth Anything 3（DA3）等实现了从无序图像到 3D 的免优化重建；这些模型默认假设全观测场景，本文将其扩展至掩码输入的修补任务。
- **2D 图像修补模型**：LaMa、MVInpainter、PowerPaint 等专攻单图或多图 2D 修补；串联至 DA3 的 baseline 实验表明，未经联合训练的 2D 修补会破坏多视图几何一致性，凸显端到端适配的必要性。
- **可学习掩码注意力**：MAT、MLLAM、Swin Transformer 等探索了动态掩码机制；本文的差异在于面向 3D 基础模型的跨视图注意力，并以 3D 几何一致性为约束进行初始化与优化。

## 局限性与未来方向
- **2D 扩散先验的上限约束**：Support Token Refinement 依赖 2D 图像修补模型（PowerPaint/LCM），最终 3D 质量受限于 2D 先验的生成能力；未来可引入原生 3D 或多视图一致视频扩散先验。
- **极端稀疏视图场景受限**：当仅 1~2 个受限视角覆盖复杂 360° 场景时，支持 token 精炼的误差累积会导致质量退化；未来可引入持久空间记忆机制维持长程几何一致性。
- **盲点修复可能覆盖已重建区域**：过高的可见度阈值会触发不必要的精炼，破坏基础模型原有的 3D 一致性；需要更精细的置信度建模与早期终止策略。
- **训练数据分布**：目前主要在 DL3DV-10K 上微调，对极大规模开放域场景的泛化仍需进一步验证。

## 研究启发与可借鉴点
1. **掩码感知跨视图注意力设计可迁移至其他 3D 基础模型任务**：凡涉及遮挡/缺失输入的 multi-view transformer（如 SLAM、novel view synthesis、3D detection），均可借鉴"掩码初始化 + 特征条件化逐层更新"的软掩码机制，兼顾信息屏蔽与上下文演化。
2. **自蒸馏几何先验用于修补任务的有效性**：冻结预训练 3D 重建模型作为教师，仅对掩码区域施加深度一致性约束，是一种低开销且能有效抑制几何幻视的训练范式，可复用于 NeRF/Gaussian Splatting 的编辑与补全任务。
3. **推理时 Confidence-Weighted Support Token 渐进精炼策略**：将生成式先验作为辅助 token 注入已有表征，并按局部可信度调制 attention，该模式可推广至稀疏视图重建、视频补帧、医学图像补全等"部分观测 + 生成补全"场景。
4. **混合掩码训练策略（随机 + 几何投影）**：几何掩码通过预训练深度将参考视图遮挡投影到源视图，模拟物理一致的 3D 遮挡；这一数据增强手段可提升模型对真实世界遮挡模式的鲁棒性。
5. **免姿态框架的实际部署价值**：消除 SfM 预处理环节使系统可直接处理手机录像、视频流等 casually captured 数据，对 AR/VR 内容创作工具、移动端 3D 编辑应用具有直接参考价值。

## 关键术语表
- **FreeInpaint**：本文提出的免姿态 feed-forward 3D 场景修补框架，可直接从无姿态含掩码多视图图像中联合恢复相机位姿、几何与纹理。
- **Learnable Mask Attention**：以输入二值掩码初始化、经由轻量网络逐层残差更新的软掩码偏置机制，动态控制跨视图注意力中掩码 token 的参与程度。
- **Support Token Refinement**：推理时针对严重遮挡盲点的渐进精炼策略，通过扩散生成支持视图并以置信度加权方式注入辅助 token，补全参考视图缺失的视觉证据。
- **Depth-Guided Self-Distillation**：利用冻结预训练 DA3 教师网络在掩码区域生成伪深度 GT，约束学生网络深度预测的自蒸馏正则化手段，防止修补区域几何塌陷。
- **DA3 (Depth Anything 3)**：香港科技大学提出的视觉几何基础模型，可从任意数量未标定图像中 feed-forward 预测深度、相机姿态与 3D 高斯参数。
- **3D Gaussian Splatting (3DGS)**：以显式各向异性 3D 高斯原语表示场景并通过 alpha-compositing 快速渲染的神经辐射场替代方案。
- **SDEdit**：Stochastic Differential Editing，通过在给定图像上添加可控噪声再利用扩散模型去噪，实现结构与内容的协同编辑。
- **IP-Adapter**：Image Prompt Adapter，将参考图像特征注入文本到图像扩散模型，实现无需额外微调的外观条件生成。

## 可复现要素
- **数据集**：训练 DL3DV-10K；评估 SPIn-NeRF、360-USID、LLFF、IMFine、GS25、Co3D（论文已公开各数据集链接，多数为开源基准）。
- **代码/权重**：项目页面 https://rorisis.github.io/FreeInpaint/；论文未明确提及 GitHub 仓库或模型权重开源声明，需以项目页为准。
- **关键超参**：输入分辨率 280×504；混合掩码比例 0.4（随机）:0.6（几何）；可见度阈值 τ=0.97；最大精炼迭代次数 3；支持 token 置信度平滑 σ 与幂次 γ（见附录 A.3/A.5）；LCM 扩散 4 步 vs 标准 50 步；骨干 DA3-Giant；优化器与学习率论文未详细列出。
