---
title: "Seeing-as-Humans-Do-Learning-from-Motion-to-Segment-Anything"
source: https://arxiv.org/pdf/2609.39785v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:37:18"
---

# 论文速读：Seeing-as-Humans-Do-Learning-from-Motion-to-Segment-Anything

## 一句话总结
本文提出 MoSA（Motion-Grounded Segment Anything），一种完全无监督的三阶段框架，通过从大规模无标签视频中提取多粒度运动线索并蒸馏为外观驱动的通用对象先验，最终在零人工标注条件下实现可在静态图像上进行多粒度分割的“segment-anything”能力，性能逼近 fully supervised SAM。

## 研究问题与动机
- **标注瓶颈**：SAM 的成功高度依赖 SA-1B 等海量人工标注数据集，限制了基础分割模型的进一步规模化与泛化。
- **静态外观方法的局限**：现有无监督图像分割方法（如 CutLER、UnSAM）主要依赖 DINO 等自监督特征聚类，仅凭颜色/纹理等静态线索推断对象，忽略了人类通过连续观察动态世界形成的运动感知先验。
- **视频运动方法的偏置**：现有视频引导的无监督分割方法易 overfit 到明显运动的主体，难以泛化至静态或弱动物体；且多局限于 instance-level，缺乏多粒度理解，无法支撑真正的 open-world “segment anything”。
- **核心科学问题**：如何将稀疏的运动监督信号转化为独立于运动、可泛化至静态场景的通用对象感（objectness prior），是构建可扩展无监督分割基础模型的关键。

## 核心贡献（创新点）
1. **提出 MoSA 三阶段无监督框架**：从零人工标注出发，依次完成运动伪标签生成、感知分组模型训练与 SAM 风格架构适配，打通“视频运动→静态分割”的跨域学习链路。
2. **设计 MGMS 多粒度运动伪标签流水线**：结合刚性物体分割（ROSM）与全局运动前景分割（MFSM）双模块，利用双向光流从约 1 万小时真实视频中自动生成约 2100 万高质量多粒度伪标签，并引入 Maskness + Boundary Sharpness 联合质量评估机制。
3. **提出 PGM 与 PGCL 对比学习范式**：通过多头尺度路由与 patch 级正负对对比损失，迫使模型在特征空间内聚外分，突破稀疏伪标签的限制，习得脱离运动依赖的外观驱动对象感。
4. **低资源高效 SAM 适配**：仅使用 1% SA-1B 子集与轻量骨干（ResNet-50 / Swin-Tiny）完成两阶段分割头微调，在七项基准上取得无监督 SOTA，并在 PartImageNet 与 PACO 上以较大 margin 超越全监督 SAM。

## 方法详解
MoSA 分为三个渐进阶段：

**阶段一：多粒度运动伪标签生成（MGMS）**
- **合成数据预训练**：构建含复杂遮挡、独立 B-spline 轨迹与随机透视变换的合成视频，为光流分割模块提供像素级精确监督。
- **ROSM（Rigid Object Segmentation Module）**：基于 SOLOv2 + ResNet-18，输入 7 帧双向光流（共 24 通道），输出中心帧内刚性运动物体或视差诱导的静态局部实例掩码。
- **MFSM（Motion Foreground Segmentation Module）**：基于 U-Net + ResNet-18，输出中心帧的整体运动前景二值掩码，弥补 ROSM 对非刚性形变（如行走人物）的不足。
- **质量评估与过滤**：定义 Maskness $S_{maskness} = \frac{1}{|\mathcal{P}_{pos}|}\sum p(x,y)$ 与 Boundary Sharpness $S_{sharpness} = \frac{1}{|B|}\sum \mathbb{I}(\|\nabla p\|>\gamma_g)$，加权得 $S_{quality} = \beta S_{maskness} + (1-\beta)S_{sharpness}$（$\beta=0.4$）。低于阈值 $\tau_{quality}=0.85$ 的掩码被丢弃，再经 NMS 去重，最终构建约 2100 万高质量伪标签集。

**阶段二：感知分组模型训练（PGM + PGCL）**
- **多头尺度路由架构**：ViT-Base 骨干之上挂载 $K=4$ 个并行线性投影头（输出 128 维特征）。根据伪标签 bbox 几何均值 $s=\sqrt{(x_{max}-x_{min})(y_{max}-y_{min})}$ 将掩码分配至对应头（$s<1/8$ 小、$1/8\le s<1/4$ 中小、$1/4\le s<1/2$ 中大、$s\ge1/2$ 大），实现 scale-aware 特征 specialization。
- **PGCL 损失**：对头 $k$ 中属于伪标签 $\mathcal{M}_i$ 的 anchor patch $\mathbf{f}_t^k$，计算其与同头所有 patch 的余弦相似度，经可学习温度 $\tau_k$ 的 sigmoid 得到同属概率 $p_{t,j}^k=\sigma(\tau_k \cdot s_{t,j}^k)$。采用 BCE 损失：
  $$\mathcal{L}_{PGCL} = -\frac{1}{K N_a N} \sum_{k=1}^K \sum_{t=1}^{N_a} \sum_{j=1}^N \left[ m_{t,j} \log p_{t,j}^k + (1-m_{t,j}) \log(1-p_{t,j}^k) \right]$$
  其中 $m_{t,j}=1$ 当且仅当两 patch 同属一 mask。该损失促使同对象 patch 特征内聚、异对象 patch 特征分离，使模型从稀疏运动监督中泛化出完整的外观驱动对象边界。

**阶段三：适配 Segment Anything 架构**
- **PGM 高分辨率推理**：全局 512×512 粗图推理 + 多尺度滑动窗口（原图 1/2、1/4
