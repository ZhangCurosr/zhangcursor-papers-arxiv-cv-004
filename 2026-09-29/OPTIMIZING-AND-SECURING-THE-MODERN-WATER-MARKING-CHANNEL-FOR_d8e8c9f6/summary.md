---
title: "OPTIMIZING-AND-SECURING-THE-MODERN-WATER-MARKING-CHANNEL-FOR"
source: https://arxiv.org/pdf/2609.34744v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:52:04"
---

# 论文速读：OPTIMIZING-AND-SECURING-THE-MODERN-WATER-MARKING-CHANNEL-FOR

## 一句话总结
本文从信息论角度重新审视现代后处理图像水印，揭示其隐式等价于并行AWGN信道与BPSK调制，指出二元字母表与无密钥设计是导致容量瓶颈与安全脆弱的主因；据此提出SNW系统，引入半正交密钥矩阵与各向同性潜空间优化，在相同骨干网络下将有效容量提升至约656 bit（较SOTA提升约2.5倍），并实现对PCA密钥估计攻击的理论完美安全。

## 研究问题与动机
- 现代后处理水印普遍采用端到端DNN编码器-解码器，但仅依赖经验性训练与固定图像增强，缺乏理论性能边界与系统设计准则。
- 现有方案无密钥机制，在Kerckhoffs原则下极易遭受线性估计与仿冒攻击；且业界常将水印安全性与对抗鲁棒性混为一谈，导致防御方向错位。
- DNN水印隐式将信号限制在二元BPSK调制下，每个码元容量上限被锁死在1 bit，且潜向量协方差呈低秩“台阶状”，大量维度被浪费。
- 同类架构下，现有工作的理论容量远未达到，单纯扩大模型规模无法突破由信道建模缺陷造成的根本瓶颈。

## 核心贡献（创新点）
- **信道理论建模**：首次形式化证明现代DNN水印对隐式定义了一个由$M'$个并行AWGN信道构成的通道，水印以BPSK调制。本质区别在于从统计谱分析切入，将黑盒训练转化为可计算的信息论问题，而非仅做经验benchmark。
- **安全性严格形式化**：提出$\eta$-安全定义，将水印安全严格限定于嵌入/决策机制的密钥保护，与解码投影的对抗鲁棒性彻底解耦。本质区别在于摒弃了当前文献中模糊的“安全”表述，给出可量化、可验证的攻击成功率上界。
- **密钥化连续嵌入框架SNW**：引入半正交矩阵密钥$\mathbf{U}$，将连续码字映射至密钥子空间与正交随机噪声的混合引导向量，通过超参$\alpha$显式调节容量与安全。本质区别在于突破二元字母表限制，兼容连续调制与容量逼近码。
- **各向同性投影学习**：设计各向同性损失迫使潜向量协方差满秩，最大化水印空间利用率，并在同等ConvNeXt/U-Net骨干下实现约2.5倍容量提升。本质区别在于不依赖架构放大，仅通过训练目标重构释放被浪费的潜维度。

## 方法详解
- **信道诊断与建模**：对PixelSeal/VideoSeal/TrustMark的解码投影$f_d$提取潜向量$\mathbf{z}$，计算协方差$\Sigma_t$的特征值谱，发现$M'$个大特征值（稳健分量）与快速衰减的尾部（脆弱/冗余分量）。线性决策头$\mathbf{c}=\mathrm{sign}(\mathbf{W}\mathbf{z}+\mathbf{b})$实际执行白化与去偏，最终软码字$\tilde{\mathbf{c}}$服从对角协方差高斯分布$\mathcal{N}(\varrho_t\mathbf{c}, \sigma_t\mathbf{I})$，等价于并行AWGN+BPSK。
- **密钥化嵌入函数**：密钥集$\mathcal{K}$为$L\times M'$半正交矩阵$\mathbf{U}$（满足$\mathbf{U}^\top\mathbf{U}=\mathbf{I}_{M'}$）。引导向量构造为：
  $$\mathbf{v} = \alpha \frac{\mathbf{U}\mathbf{c}}{\sqrt{M'}} + \sqrt{1-\alpha^2}\frac{(\mathbf{I}_L - \mathbf{U}\mathbf{U}^\top)\mathbf{q}}{\|(\mathbf{I}_L - \mathbf{U}\mathbf{U}^\top)\mathbf{q}\|_2}, \quad \mathbf{q}\sim\mathcal{N}(0,\mathbf{I}_L)$$
  $\alpha$控制水印能量占比：$\alpha=1$时全部能量用于信号，容量最大；$\alpha<1$时部分能量注入正交噪声以掩盖子空间，提升安全性。
- **决策机制**：$d_\mathbf{U}(\tilde{\mathbf{v}}) = \mathbf{U}^\top \tilde{\mathbf{v}}$，可直接恢复连续码字$\mathbf{c}$；若需二元输出可追加$\mathrm{sign}(\cdot)$还原BPSK。
- **联合训练目标**：
  $$\mathcal{L} = \lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}} + \lambda_{\mathrm{iso}}\mathcal{L}_{\mathrm{iso}} + \lambda_{\mathrm{qual}}\mathcal{L}_{\mathrm{qual}}$$
  - $\mathcal{L}_{\mathrm{align}}$：池化前后特征与目标$\mathbf{v}$的余弦相似度最大化。
  - $\mathcal{L}_{\mathrm{iso}}$：惩罚同图不同码字、异图不同码字及纯宿主图像间的潜向量内积平方，迫使协方差满秩、分布各向同性。
  - $\mathcal{L}_{\mathrm{qual}}$：PSNR hinge损失（目标48 dB）结合LPIPS感知损失。
- **三阶段训练**：Stage 1学习无变换下的信号提取与各向同性；Stage 2迁移至池化表征、加入LPIPS与全类各向同性损失并渐进增强；Stage 3追加反仿冒损失$\mathcal{L}_{\mathrm{spoof}}=\frac{1}{B}\sum_i\langle f(\mathbf{x}^{(j)}+\mathbf{w}_1^{(i)}), \mathbf{v}_1\rangle^2$抑制残差跨宿主迁移。

## 实验与结果
- **设置**：测试集为MFlickr中1000张$1024\times1024$图像，训练集为COCO 2017 train。基线包括PixelSeal（256 bit）、VideoSeal（256 bit）、TrustMark（100 bit）。统一在$10^6$张MFlickr图像上做白化处理以保证公平对比，水印功率固定48 dB PSNR。
- **容量**：SNW在Identity条件下达到约684.7 bit，经典变换与擦除攻击（VAE Purification、DiffPure、Sana、WM Forger）下仍维持在数百bit级别，较PixelSe
