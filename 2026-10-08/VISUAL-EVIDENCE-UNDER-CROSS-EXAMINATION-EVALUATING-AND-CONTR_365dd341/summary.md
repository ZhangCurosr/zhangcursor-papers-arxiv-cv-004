---
title: "VISUAL-EVIDENCE-UNDER-CROSS-EXAMINATION-EVALUATING-AND-CONTR"
source: https://arxiv.org/pdf/2610.09550v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 22:46:34"
---

# 论文速读：VISUAL-EVIDENCE-UNDER-CROSS-EXAMINATION-EVALUATING-AND-CONTR

## 一句话总结
本文提出“候选绑定视觉贡献（candidate-bound visual contribution）”评估准则与 CROSS-Bench 基准，并设计 RIVET 接口将证据候选对齐与响应强度解耦控制；实验表明该接口在四项冻结骨干网络上平均提升 5.70 pp 任务准确率，同时将归一化证据转移率从 0.512 显著提升至 0.651。

## 研究问题与动机
1. **可见不等于可用**：现有 VLM 推理接口（crop、region、工具观测）使中间视觉信息可追踪，但证据可能影响答案却未流向其应支持的候选，导致“定位正确、归属错误”。
2. **评估维度缺失**：现有工作多关注反事实依赖或梯度敏感性，缺乏对“关系失效时效应是否归零、合法重绑定时效应是否迁移”的成对诊断机制。
3. **精度与忠实度背离**：高任务准确率并不等价于高证据所有权（ownership）；二者可分离，需在同一证据供给下分别度量效用、特异性与所有权。
4. **接口设计需求**：需要一个能保留证据身份与不确定性、在候选层面重组证据，并将“对齐关系”与“响应强度”独立控制的决策组件。

## 核心贡献（创新点）
1. **提出候选绑定视觉贡献准则与 CROSS-Bench 基准**：构建 28,000 个决策根节点基准与 960 个成对干预子集，首次系统化检验证据在干净/无效/重绑定条件下的后验变化轨迹。
2. **设计 RIVET 解耦接口**：通过 SPECIFY-PRESERVE-COMPOSE-BOUND 四阶段流水线，以候选查询条件化组织全证据池，并用标量控制器独立调节后验更新强度，避免证据效应错配或放大。
3. **揭示精度-所有权可分离性并提供跨骨干泛化方案**：证明 Unconstrained Fusion 等高精度基线存在严重所有权泄漏；RIVET 在四种冻结 VLM 骨干上保持相似转移/泄漏模式并平均提升 +5.70 pp 准确率。

## 方法详解
- **形式化与度量定义**：决策根 $r_i=(I_i, q_i, \mathcal{C}_i, y_i)$，Base 后验记为 $\mathbf{p}_i^0$。定义效用 $G_{\mathrm{clean}}=\mathbb{E}[\log(\bar{p}_{i,y_i}^{\mathrm{clean}}/\bar{p}_{i,y_i}^0)]$、特异性 $A_{\mathrm{inv}}=\mathbb{E}[\max_{\iota\in\mathcal{T}_{\mathrm{inv}}}\mathrm{TV}(\mathbf{p}_i^\iota,\mathbf{p}_i^0)]$、转移 $T_{\mathrm{auth}}$ 与泄漏 $L_{\mathrm{orig}}$（均以干净
