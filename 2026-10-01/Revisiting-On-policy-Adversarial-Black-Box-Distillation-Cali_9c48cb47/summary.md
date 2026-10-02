---
title: "Revisiting-On-policy-Adversarial-Black-Box-Distillation-Cali"
source: https://arxiv.org/pdf/2609.39757v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-02 01:51:29"
field: "大语言模型对齐与蒸馏"
keywords: ["black-box distillation", "reward geometry", "optimal transport", "GRPO", "on-policy RLHF", "reward calibration"]
innovations: ["揭示BT目标对组内离散的坍缩偏差并证明", "提出CGC+PGM两阶段组几何条件框架", "将一维Wasserstein正则引入reward组校准"]
benchmarks: ["LMSYS held-out", "DollyEval", "SelfInst", "VicunaEval"]
---

# 论文速读：Revisiting-On-policy-Adversarial-Black-Box-Distillation-Cali

## 一句话总结
本文指出**组级奖励几何失配**是对策对抗黑盒蒸馏的关键瓶颈，并提出GRGC（Groupwise Reward Geometry Conditioning）方法，通过Gaussian OT正则校准Critic侧奖励组几何、以及有符号幂变换（PGM）在Policy侧保序放大信息间隔，显著提升小模型从大模型蒸馏后的指令跟随性能。

---

## 研究问题与动机
- **Critic的BT目标导致组内离散坍缩**：Bradley-Terry对手判别目标在固定教师分数与学生组均值的前提下，唯一最优解是所有学生得分相等（Proposition 1），破坏了组内用于构建advantage的相对差异。
- **低离散度引发排序脆弱**：当最小非零组内间距 $\Delta_{\min} \leq 2\eta$（$\eta$ 为Critic噪声上界），扰动即可翻转严格序对，使advantage失去可靠性（Proposition 2）。
- **低尺度放大噪声敏感度**：预归一化前组标准差 $\sigma_x$ 越小，标准化advantage对原始扰动的放大效应越强（Proposition 3），进一步恶化下游策略更新质量。
- **已有蒸馏方法未显式约束奖励组几何**：现有black-box distillation多聚焦logit/表示对齐，缺乏对Critic输出组分布形状（离散度、scale、有序性）的系统性正则化。

---

## 核心贡献（创新点）
1. **首次揭示BT目标对组内离散度的系统性坍缩偏差**：从理论证明BT损失在固定均值下唯一最小值对应零组内离散，为后续组几何正则提供动机。
2. **提出GRGC两阶段框架**：CGC（Critic侧高斯OT正则校准）+ PGM（Policy侧有符号幂变换保序放大），分别解决奖励组的scale/shape匹配与信息间隔放大问题。
3. **引入一维Wasserstein距离作为组级奖励正则**：以组均值锚定的高斯分位数模板为目标，同时惩罚坍缩（方差失配）与有序形状失配，且保持平移不变性。
4. **建立advantage误差→策略log-odds变化的理论链路**：推导prop 7–9，量化advantage排序错误对clipped GRPO代理目标的有界失真传播。
5. **在多个teacher-student组合上验证显著增益**：GPT-5 → Qwen2.5-3B在LMSYS held-out上score达50.19（vs. SeqKD 47.06、GAD 48.44），win rate 45.3%（vs. 18.4%/25.1%）。

---

## 方法详解
### 框架总览
$$\text{BT Critic} \to r \text{（奖励几何）} \to z \text{（标准化稳定性）} \to \hat{r} \text{（保序放大）} \to A \text{（advantage质量）} \to \text{策略分离}$$

### Stage 1：CGC – Gaussian Groupwise Optimal Transport Calibration
- 输入：每条prompt的学生响应奖励组 $r_{(1)} \leq r_{(2)} \leq \cdots \leq r_{(N)}$（已排序）。
- 目标模板：以组均值 $\mu_x$ 为中心，对齐至标准正态分位数 $t_j = \mu_x + \Phi^{-1}\!\left(\frac{j-0.5}{N}\right)$。
- OT损失：$\mathcal{L}_{\mathrm{OT}}(x) = \frac{1}{N}\sum_{j=1}^{N}(r_{(j)} - t_j)^2$，即精确一维 $W_2^2$ 距离。
- 总Critic loss：$\mathcal{L}_{\text{critic}} = \mathcal{L}_{\text{BT}} + \lambda_{\text{OT}} \mathcal{L}_{\text{OT}}$，实验中 $\lambda_{\text{OT}} = 0.01$。
- 关键性质：
  - **抗坍缩**：$\mathcal{L}_{\text{OT}} \geq (\sigma_x - \sigma_q)^2$，坍缩时 $\mathcal{L}_{\text{OT}} \geq \sigma_q^2 > 0$。
  - **形状感知**：$\mathcal{L}_{\text{OT}} = (\sigma_x - \sigma_q)^2 + 2\sigma_x\sigma_q(1-\rho_x)$，同时惩罚有序性偏离（$\rho_x$ 为秩相关系数）。
  - **平移不变**：以组均值 $\mu_x$ 为唯一最优锚点，对共同偏移不变。

### Stage 2：PGM – Group Power Transform（有符号幂变换）
- 标准化：$z_i = \frac{r_i - \mu_x}{\sigma_x + \varepsilon}$。
- 幂变换：$\hat{r}_i = \mathrm{sign}(z_i)|z_i|^{\gamma}$，实验取 $\gamma = 1.5$。
- 性质：
  - **严格保序**：单调变换不改变组内排序。
  - **信息间隔放大**：当 $\min(|z_i|, |z_j|) \geq \rho$ 时，$|\hat{r}_i - \hat{r}_j| \geq \gamma \rho^{\gamma-1}|z_i - z_j|$（Prop 5）。
  - **含噪鲁棒**：在信息充分区（$z_j^\star \geq 1+\eta$），变换后间隔下界 $\geq \gamma(\Delta^\star - 2\eta)$（Prop 6）。

### 策略更新
- 对 $\hat{r}_i$ 执行标准GRPO组内归一化得到advantage $A_i$。
- 一步更新导致pairwise log-odds变化：$\log\frac{\pi_i(\theta^+)}{\pi_j(\theta^+)} - \log\frac{\pi_i(\theta)}{\pi_j(\theta)} = \eta_{pg}(A_i - A_j)$（Prop 7）。
- 错误排序advantage会使策略朝错误方向移动（Prop 8）；advantage误差经clip GRPO surrogate传递的保守上界为 $(1+\kappa)\epsilon_A/\sqrt{N}$（Prop 9）。

---

## 实验与结果
### 设置
- **Teachers**：GPT-5-Chat（API）、Doubao-Seed-2.0（API，thinking mode）。
- **Students**：Qwen2.5-Instruct (1.5B, 3B)、Llama-3.2-Instruct (1B, 3B)。
- **训练数据**：LMSYS-Chat-1M（子集）、Dolly-Train。
- **训练细节**：2 epochs（1 epoch SFT warmup + 1 epoch adversarial）；global batch=128；组大小 $N=8$；$\beta=0.001$；温度=0.8；LR=$1\times10^{-6}$；硬件8× NVIDIA A100。
- **评测基准**：In-distribution LMSYS held-out (479 samples)；OOD: DollyEval (500)、SelfInst (242)、VicunaEval (80)。
- **Judge**：Qwen2.5-72B（主）；GPT-OSS-120B与人工评测见附录。

### 核心结果（Qwen2.5-72B评测）
| 方法 | Score | Win Rate |
|------|-------|----------|
| SeqKD | 47.06 | 18.4% |
| GAD | 48.44 | 25.1% |
| **GRGC** | **50.19** | **45.3%** |

- **最强结果**：GRGC在GPT-5 → Qwen2.5-3B设置下取得Score 50.19，较SeqKD提升+3.13，较GAD提升+1.75；Win Rate 45.3%，较SeqKD提升+26.9pp。
- OOD（DollyEval、SelfInst、VicunaEval）与Llama-student系列实验呈现一致增益趋势（详见附录Table 2–4）。

---

## 相关工作脉络
- **Black-box Distillation**（SeqKD、GAD等）：聚焦teacher-student输出logit或隐藏表示对齐，未显式约束Critic奖励组的分布几何；本文聚焦advantage构建上游的奖励几何质量。
- **On-policy RLHF / GRPO**（DeepSeek-R1、OpenRLHF）：依赖group-relative advantage，但未对Critic输出的奖励组施加任何形状正则；本文揭示BT目标导致的几何退化并提出CGC修正。
- **Optimal Transport in ML**（Cuturi 2013、Arjovsky WGAN 2017等）：OT广泛用于分布匹配，本文首次将一维Wasserstein正则引入reward组几何校准。
- **Reward Shaping / Value Estimation**（Ng et al. 1999、PopArt 2016、RUDDER 2019）：关注奖励函数的等值变换或统计归一化，本文强调组内相对几何对advantage排序稳健性的影响。
- **Reward Calibration / Over-optimization**（Gao et al. ICML 2023、RewardBench 2025等）：关注reward模型的过拟合与对齐度量，本文从蒸馏视角提出几何正则以保障下游策略更新可靠性。
- **OT for Knowledge Distillation**（Cui et al. TPAMI/CVPR/NeurIPS 2024–2026、WCoRD 2021、KNOT 2022等）：已有工作将OT用于token/样本级蒸馏，本文首次将其拓展至"组级奖励分布校准"这一新场景。

---

## 局限性与未来方向
- **高斯模板的先验假设**：当前CGC采用标准正态分位数作为参考形状，对其他数据分布（如长尾奖励）可能不够灵活；未来可探索自适应模板学习。
- **幂变换指数 $\gamma$ 需人工设定**：实验取固定值1.5，未讨论 $\gamma$ 对噪声强度、组大小的敏感性；可探索数据驱动或调度式 $\gamma$。
- **仅在一类蒸馏设置下验证**：目前聚焦on-policy adversarial black-box distillation，对off-policy或多轮迭代蒸馏的适用性待检验。
- **Critic模型容量依赖**：CGC对BT Critic的输出施加正则，若Critic本身容量不足或欠拟合，OT正则的增益可能受限。
- **未扩展到多目标reward场景**：当前reward为标量，对多维权重奖励（如长度+有用性+安全）的组几何正则需进一步推广。

---

## 研究启发与可借鉴点
1. **组几何正则可迁移至其他依赖group-relative signal的算法**：任何基于组内相对排序构建advantage/target的方法（如GRPO变体、RLAIF、Preference-based fine-tuning）均可参考CGC+PGM的设计。
2. **一维Wasserstein正则的实现成本极低**：与二维 Sinkhorn 不同，一维OT存在闭式解，可直接叠加于现有Critic loss，工程侵入性小。
3. **保序非线性变换是一种低成本的"信号锐化"手段**：PGM的思想可类比于特征工程中的单调映射，适用于任何需要放大informative gap且保持序关系的场景。
4. **理论命题链（Prop 1–9）提供了诊断工具**：后续工作可用Prop 1–3作为"组几何退化"的检测指标，评估Critic训练质量。
5. **与团队方向结合机会**：若团队研究reward模型校准、偏好学习、或蒸馏策略优化，GRGC可作为即插即用模块接入现有pipeline。

---

## 关键术语表
- **Groupwise Reward Geometry Mismatch**：Critic输出的学生奖励组在scale、margin、排序上偏离最优结构，导致advantage构建失准。
- **Bradley-Terry (BT) Target**：基于配对比较的对数几率损失，常用于reward modeling与 Preference Learning。
- **On-policy Adversarial Black-Box Distillation**：student采样生成响应组并用于更新Critic/策略，teacher以黑盒API形式提供reward信号的蒸馏范式。
- **GRPO (Group Relative Policy Optimization)**：基于组内相对奖励计算advantage的策略梯度算法，DeepSeek-R1等采用。
- **Wasserstein Distance ($W_2^2$)**：最优运输距离的平方，一维情形下有闭式解，等价于排序后对应点的欧氏距离平方和。
- **Group Power Transform (PGM)**：有符号幂变换，在保序前提下放大已具备信息的组内间隔。
- **Advantage Gap**：组内两个响应的advantage之差，直接控制策略更新的pairwise log-odds变化。
- **Clipped GRPO Surrogate**：带clip约束的GRPO目标函数，限制单步策略更新的幅度。

---

## 可复现要素
- **数据集**：LMSYS-Chat-1M（子集）、Dolly-Train；数据公开可获取。
- **代码**：开源，见 https://github.com/2018cx/GRGC。
- **权重**：未发布新checkpoint，使用开源student模型（Qwen2.5、Llama-3.2）。
- **关键超参**：$\lambda_{\text{OT}} = 0.01$、$\gamma = 1.5$、$N=8$、$\beta=0.001$、温度=0.8、LR=$1\times10^{-6}$、global batch=128、2 epochs（1 warmup + 1 adversarial）；硬件8× A100。

---
