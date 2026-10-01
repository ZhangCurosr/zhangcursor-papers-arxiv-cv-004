---
title: "PartiCam-Camera-Controlled-Video-Generation-with-Reward-Guid"
source: https://arxiv.org/pdf/2609.39504v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:37:54"
field: "可控视频生成"
keywords: ["camera-controlled video generation", "reward-guided sampling", "sequential monte carlo", "particle filtering", "training-free diffusion", "novel view synthesis", "inference-time alignment"]
innovations: ["提出全局SMC种群探索与局部PF-Restart奖励引导重采样的两级无训练框架，同时解决轨迹漂移与多样性坍缩问题", "将不可微相机奖励（VGGT轨迹误差）直接集成进扩散采样过程，无需梯度反向传播", "证明美学奖励用于全局多样性维持、相机奖励用于局部几何修正的分层奖励策略的有效性"]
benchmarks: ["Tanks and Temples static NVS", "NVS-Solver dynamic monocular video NVS"]
---

# 论文速读：PartiCam-Camera-Controlled-Video-Generation-with-Reward-Guidance

## 一句话总结
PartiCam 提出了一种**无需训练的 Particle Filtering 方法**，通过全局 SMC 轨迹探索与局部 PF-Restart 粒子的奖励引导重采样相结合，实现扩散视频模型中高精度、稳定的摄像机轨迹控制，显著降低了轨迹漂移并提升了视觉质量。

## 研究问题与动机
- **核心问题**：现有视频扩散模型缺乏细粒度相机运动控制能力，训练类方法依赖大规模配对相机标注数据且成本高，而已有免训练方法（如 NVS-Solver 的 Score Modulation）容易产生**早期漂移误差累积**，导致轨迹偏差难以纠正。
- **免训练优势**：Backbone-agnostic，无需重新训练即可继承新模型改进，同时可生成相机标注视频数据用于后续蒸馏训练。
- **单一轨迹采样局限**：Score Modulation 仅沿单条去噪轨迹优化，初始随机噪声若落在劣质生成路径，错误会随步骤放大；无约束 Restart 虽可逃逸局部最优，但会破坏已积累的生成结构。
- **全局探索不足**：独立粒子操作缺乏跨粒子协作机制，所有候选易集体陷入相似的劣质潜空间区域。

## 核心贡献（创新点）
1. **全局 SMC + 局部 PF-Restart 两级框架**：同时维持 K 条候选轨迹的种群多样性（全局），并在选定时点对每条粒子做 noising-denoising 局部探索（局部），两者互补，避免单一机制的缺陷。
2. **奖励引导的重采样机制**：支持可微与不可微奖励函数，分别使用美学奖励（全局）和相机奖励（局部），实现无需梯度反向传播的推理时对齐。
3. **显著提升相机轨迹一致性**：在静态（Single Image）和动态（Monocular Video）NVS 任务上均取得 SOTA，ATE 降低至 NVS-Solver 的约 1/4~1/6，同时 FID 指标大幅改善。
4. **对训练基线方法的 OOD 泛化补救**：可叠加于 TrajectoryCrafter、ReCamMaster 等训练方法之上，缓解其在挑战性相机轨迹上的几何不一致与视觉伪影。

## 方法详解
- **问题设定**：给定视频扩散模型 $\mathcal{D}_\theta$ 和奖励函数 $R(\mathbf{x}_0) = \sum_i \alpha_i R_i(\mathbf{x}_0)$，目标是采样满足目标相机轨迹的高质量视频。
- **Score Modulation 基础**：沿 NVS-Solver  formulation，通过 warped conditioning image 注入目标视角特征，调整潜变量得分；但存在误差累积问题。
- **全局 SMC 阶段**：
  - 维护 K 个粒子 $\{(\mathbf{z}_t^{(k)}, w_t^{(k)})\}_{k=1}^K$，沿中间目标分布 $\gamma_t(\mathbf{z}_{t:T}) \propto p_\theta(\mathbf{z}_{t:T}) \psi_t(\mathbf{z}_t)$ 推进。
  - 每个 timestep 通过 reverse diffusion transition 传播粒子，再按增量权重更新公式（Eq.6）重加权：$\tilde{w}_{t-1}^{(k)} = w_t^{(k)} \exp(\beta R_{t-1}(\mathbf{x}_{t-1}^{(k)}) - \beta R_t(\mathbf{x}_t^{(k)}))$。
  - 当有效样本数（ESS）低于阈值时执行全局 resampling，**使用美学奖励**驱动。
- **局部 PF-Restart 阶段**：
  - 在选定时步 $\mathcal{T}_{ref}$，对每个粒子施加 $N_r$ 个 one-step noising-denoising 扰动（Eq.8）生成候选集合。
  - 对候选用相机奖励评分并分类重采样（Eq.9），**使用相机奖励**驱动（通过 VGGT 估计轨迹误差）。
  - 该步骤每迭代约 4 次执行一次，重复 $N_{pf}=4$ 轮。
- **设计选择**：$K=2$ 全局粒子，$N_r=2$ 局部候选，100 步去噪；第 8 步起启动全局 SMC，第 32 步起启动局部 PF-Restart，均在每 4 步执行一次更新。

## 实验与结果
- **数据集**：静态场景（Tanks and Temples 6 scene + 3 selected scenes）；动态场景（9 个 monocular 视频，含城市与自然场景）。评估协议与 NVS-Solver 一致。
- **评估指标**：相机精度（Particle-SFM 估轨迹 → ATE, RPE-T, RPE-R）；视觉质量（FID）。
- **静态场景（单图 NVS）SOTA**（Table 1）：
  - **Ours**：FID = **121.56**（vs NVS-S (Post) 165.12），**ATE = 0.526**（vs 0.767，↓31%），**RPE-T = 0.122**（vs 0.156，↓22%），**RPE-R = 0.144°**（vs 0.170°）。
  - 优于 NVS-Solver、TrajectoryCrafter 等训练-free/训练-based 基线。
- **动态场景（单目视频 NVS）SOTA**（Table 2）：
  - **Ours**：FID = **31.86**（vs NVS-S (Post) 39.86），**ATE = 0.807**（vs 2.308，↓65%），**RPE-T = 0.061**（vs 0.725，↓92%），**RPE-R = 0.414°**。
  - 在存在遮挡和动态场景时优势尤为明显。
- **Backbone 泛化**：对 CogVideo、SVD、TrajectoryCrafter 均有效（Table 3），其中 CogVideo backbone 上提升最大（FID 41.08→31.86，ATE 0.981→0.807）。
- **消融关键数字**（静态场景，右扫轨迹）：
  - 完整方法：RPE-T=0.09, RPE-R=0.09°, ATE=0.37, FID=103.82（vs NVS-Solver 基线 ATE=1.56 → **↓4×**，RPE-T=0.52 → **↓6×**）。
  - 仅全局 SMC（K=4）：FID 反降至 135.84，说明局部校正不可缺。
  - 相机奖励全局化变体：ATE 可进一步降至 0.28（-24%），但 FID 略恶化（103.06 vs 102.51），存在精度-质量 trade-off。
  - $N_r=4$ 对比 $N_r=2$：ATE 从 0.37 退化至 0.46，说明候选过多可能破坏全局一致性。

## 相关工作脉络
1. **NVS-Solver [78]**（训练-free，score modulation）：本文的核心基线和方法起点，PartiCam 在其基础上叠加 SMC+PF-Restart 纠正漂移，本质区别是从单轨迹转向多轨迹种群搜索。
2. **TrajectoryCrafter [82]**（训练-based，微调）：针对相机控制的专用微调模型，PartiCam 可作为通用推理时增强模块叠加其上，缓解 OOD 轨迹泛化失败。
3. **TDS [70] / Ψ-Sampler [77]**（SMC 推理时对齐）：需对奖励求梯度并反向传播通过扩散模型，不支持不可微相机奖励；PartiCam 通过 particle-based resampling 规避此限制。
4. **SCG [35]**（单一全局粒子 SMC）：仅维持一个粒子，采样多样性不足；本文用 K=2 全局粒子+局部候选显著超越（SCG aesthetic RPE-T=0.71 vs 本文 0.09）。
5. **Restart / RePaint [47, 73]**（无约束重启）：本文 PF-Restart 将无结构 restart 升级为奖励引导的局部选择，消除方差过大的不稳定问题。
6. **MotionCtrl [66] / Cameractrl [29]**（训练-based 相机控制）：依赖大量配对监督；PartiCam 完全不训练，zero-shot 即用。

## 局限性与未来方向
- **计算开销**：维持多条候选轨迹需要更多去噪步预算；虽然推理时间被控制在 NVS-Solver 以内，但相比单轨迹方法仍增加约 2-4× 算力。
- **奖励函数依赖**：框架有效性取决于奖励设计，不同任务可能需要定制化调参（如相机奖励 vs 美学奖励的比例 $\alpha_i$）。
- **局部候选过多反而有害**：$N_r=4$ 比 $N_r=2$ 表现更差，说明候选数量存在 sweet spot，需经验调优。
- **未来方向**：可扩展至其他可控生成任务（如运动控制、文本条件编辑），探索自动化的奖励调度策略，以及与训练式蒸馏方法结合。

## 研究启发与可借鉴点
1. **两级 SMC 架构可迁移**：全局种群探索 + 局部 restart 精炼的分层设计，适用于任何需要"多样性维持 + 精准修正"的扩散采样任务（如受约束图像编辑、3D 一致生成）。
2. **奖励函数分离策略**：全局用美学/多样性奖励、局部用几何/精确约束奖励，这一分工原则可有效避免全局探索时多样性丧失，值得在其他 reward-guided diffusion 工作中验证。
3. **不可微奖励的直接集成**：无需梯度反向传播即可利用 VGGT 等外部工具计算的奖励信号指导采样，为集成各类黑盒评估器（如 3D 一致性、物理合理性）提供了实用范式。
4. **对训练基线的即插即用增强**：PartiCam 可叠加在 TrajectoryCrafter 等模型上，其"推理时后处理"思路可作为提升已有模型的通用策略，尤其适用于 OOD 泛化失败的补救。
5. **实验设计的对照完整性**：同时评估静态/动态场景、单图/单目视频输入、CogVideo/SVD/TrajectoryCrafter 多 backbone，并提供详细消融（各组件贡献、reward 设计、候选数量），为论文可靠性提供了强支撑。

## 关键术语表
**Sequential Monte Carlo (SMC)**：一种基于粒子的贝叶斯推断方法，通过加权粒子的传播-重采样循环近似复杂后验分布，本文用于全局轨迹探索。
**PF-Restart (Particle Filter Restart)**：局部精炼阶段，对每个粒子施加重噪声后重新去噪，并用奖励引导选择最优候选，相当于带奖励筛选的无约束重启。
**Score Modulation**：通过 warped conditioning 调整扩散模型去噪得分以引导视角合成的推理时控制方法，NVS-Solver 的核心机制。
**ATE (Absolute Trajectory Error)**：估计相机轨迹与 ground-truth 之间的绝对位置偏差，是衡量相机控制精度的核心指标。
**RPE-T / RPE-R**：Relative Pose Error 的平移/旋转分量，衡量相邻帧间位姿变化的相对误差。
**FID (Fréchet Inception Distance)**：衡量生成图像/视频与真实数据分布之间距离的视觉质量指标，越低越好。
**VGGT**：Visual Geometry Grounded Transformer，用于从生成视频中估计相机轨迹的工具。
**Tweedie's Formula**：在扩散过程中从含噪潜变量估计去噪结果的公式，用于在不重新完整解码的情况下近似奖励值。

## 可复现要素
- **数据集**：Tanks and Temples（静态）、NVS-Solver 提供的动态视频集；评估协议遵循 NVS-Solver [78]。**数据集本身公开**，但具体划分参考原论文。
- **代码/权重**：论文未明确声明代码是否开源，仅说明在 CogVideo 和 SVD backbone 上验证。**权重方面**使用 CogVideo [33] 和 SVD [7] 预训练模型。
- **关键超参**：$K=2$（全局粒子数），$N_r=2$（局部候选数），$N_{pf}=4$（局部精炼轮数），去噪总步数 100 步，全局 SMC 从第 8 步开始每 4 步执行至第 40 步，局部 PF-Restart 从第 8 步至第 32 步每 4 步执行一轮；$\beta$（奖励温度）和 $\alpha_i$（奖励权重）**论文未提及具体数值**。
