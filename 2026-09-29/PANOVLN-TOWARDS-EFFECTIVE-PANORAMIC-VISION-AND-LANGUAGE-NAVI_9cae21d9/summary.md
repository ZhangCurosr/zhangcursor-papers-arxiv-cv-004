---
title: "PANOVLN-TOWARDS-EFFECTIVE-PANORAMIC-VISION-AND-LANGUAGE-NAVI"
source: https://arxiv.org/pdf/2609.34759v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:53:25"
---

# 论文速读：PANOVLN-TOWARDS-EFFECTIVE-PANORAMIC-VISION-AND-LANGUAGE-NAVI

## 一句话总结
本文提出 PanoVLN，探索将 360° 等距柱状全景图（ERP）引入连续视觉-语言导航（VLN-CE）任务；研究发现仅替换输入视角收益有限，需同步延长动作预测 horizon、构建面向分支决策的训练数据，并以残差方式注入冻结的全景几何特征，最终在 R2R-CE 与 RxR-CE Val-Unseen 上分别取得 77.3% 与 78.0% 的成功率，刷新 SOTA 并实现零样本四足机器人实地导航。

## 研究问题与动机
- 现有 VLN 工作多依赖单目/透视图像进行决策，视场受限，难以在路线转向或分叉前获取完整的空间线索。
- 直接将透视图像替换为 ERP 在全相同训练与推理设置下仅带来有限增益，甚至性能下降，表明“更广视野”必须配合动作规划、监督信号与视觉表征的系统性改造
