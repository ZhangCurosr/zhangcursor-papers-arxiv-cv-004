# NOWCASTDIT: DIFFUSION TRANSFORMERS ARE EFFECTIVE PRECIPITATION NOWCASTERS

Haoran Xu<sup>1∗</sup>, Xingzhuo Guo<sup>1∗</sup>, Yuchen Zhang<sup>2</sup>, Jincheng Zhong<sup>3</sup>, Jianmin Wang<sup>1</sup>, Mingsheng Long<sup>1,B</sup>

<sup>1</sup>School of Software, BNRist, Tsinghua University, China

<sup>2</sup>Envision Energy, China

<sup>3</sup>Kling Team, Kuaishou Technology, China {xuhaoran26,gxz23}@mails.tsinghua.edu.cn mingsheng@tsinghua.edu.cn

## ABSTRACT

Precipitation nowcasting demands accurate short-term forecasts under strong spatiotemporal variability. Diffusion models are well suited to modeling complex precipitation distributions, yet existing approaches often introduce increasingly specialized designs, leaving the capability of a standard diffusion architecture underexplored. We show that a standard Diffusion Transformer already provides a simple and scalable foundation for precipitation nowcasting, with domain-specific requirements accommodated naturally within its design space. Based on this principle, we develop NowcastDiT and instantiate this flexibility through two complementary adaptations: a dynamics-aware noise prior for temporally coherent forecasts, and end-to-end reinforcement learning with timestep-aware rewards for meteorological skill. Experiments on SEVIR and MRMS benchmarks show that NowcastDiT achieves state-of-the-art performance in both perceptual quality and meteorological skill. These results suggest that standard DiT can serve as an effective foundation for precipitation nowcasting.

## 1 INTRODUCTION

Precipitation nowcasting aims to predict the short-term evolution of rainfall fields at high spatial and temporal resolutions. An effective forecasting system is expected to accurately capture complex spatiotemporal dynamics, account for intrinsic uncertainty, and maintain reliability for rare but high-impact precipitation events (Wang et al., 2017). From a generative modeling perspective, precipitation nowcasting is fundamentally a conditional generation task, with diffusion models (Ho et al., 2020; Song et al., 2021; Lipman et al., 2023) providing a natural framework given their strong capability for probabilistic image and video generation(Peebles & Xie, 2023; Wan et al., 2025; Seedance et al., 2026; Aditi et al., 2026).

Beyond these general modeling capabilities, existing works have focused on several precipitationspecific requirements, particularly temporal coherence across forecast frames and meteorological skill in capturing high-impact events. To address them, existing generative methods introduce various task-specific designs including physics-based motion constraints (Zhang et al., 2023), intensity continuity regularization (Gao et al., 2023), and decomposition-based modeling frameworks (Yu et al., 2024; Gong et al., 2024; Wen et al., 2026). Despite their different forms, these methods follow a common design philosophy: specializing the generative framework to explicitly encoding these requirements. While effective, this line of development narrows the exploration of foundational model designs for precipitation nowcasting, leaving a more fundamental question insufficiently explored: how necessary is task-specific specialization beyond a standard diffusion model?

In this paper, we argue that a standard Diffusion Transformer provides a sufficient foundation for precipitation nowcasting. We posit that precipitation nowcasting is not fundamentally different from conditional video generation in terms of its core generative modeling requirements, as both require capturing complex spatiotemporal dynamics under uncertainty, capabilities that have already been demonstrated by modern video diffusion models (Seedance et al., 2026; Aditi et al., 2026). Meanwhile, precipitation-specific requirements can be accommodated within the broader design space of diffusion models, as suggested by the versatility of modern diffusion models across diverse video generation tasks (Guo et al., 2025; Xue et al., 2025)

![](images/0bd6200eae8e5db77a0f240c742d145db2a8c50abcbca22b04921781d1e29bab.jpg)  
(a) Specialized Diffusion Nowcasters

![](images/467ac6f6e3c173374ed8a4b55641a2655885411d5f359518b4e33069d2ad45ab.jpg)  
(b) Standard DiT

![](images/04ccc631acee081a5a7fff5b680876fd62ab76d6fd6668bc98dcc944db4e0842.jpg)  
(c) NowcastDiT (Ours)  
Figure 1: Comparison of diffusion modeling paradigms for precipitation nowcasting. (a): Specialized diffusion nowcasters encode domain knowledge through task-specific models or losses. (b): A standard DiT is applicable for precipitation nowcasting. (c): NowcastDiT adds minimal precipitation-specific adaptation towards a standard DiT without changing its backbone.

To this end, we propose the Nowcasting Diffusion Transformer (NowcastDiT), a simple and scal able framework built on a standard DiT with minimal precipitation-specific adaptations. From the diffusion perspective, NowcastDiT introduces a dynamics-aware noise prior that captures shared evolution patterns and variations across forecast frames to enforce temporal coherence. From the training perspective, NowcastDiT incorporates end-to-end reinforcement learning with timestepaware rewards to enhance meteorological skill. On two widely recognized radar-based benchmarks, SEVIR and MRMS, NowcastDiT achieves state-of-the-art performance across both perceptual quality and meteorological skill metrics. These results suggest that standard DiT can serve as an effective foundation for precipitation nowcasting.

## Our contributions are summarized as follows:

• We revisit precipitation nowcasting from the perspective of standard diffusion models, and show that a standard DiT provides a sufficient foundation, with precipitation-specific requirements handled through targeted adaptation rather than model specialization.

• Based on this principle, we propose NowcastDiT, a simple and scalable framework that retains the standard DiT with a dynamics-aware noise prior to promote temporal coherence and reinforcement learning with timestep-aware rewards to enhance meteorological skill.

• Extensive experiments on SEVIR and MRMS demonstrate the state-of-the-art performance of NowcastDiT, validating standard DiT as an effective foundation for precipitation nowcasting.

## 2 PRELIMINARIES

Precipitation nowcasting aims to predict S future frames $\mathbf { x } ^ { 1 : S }$ from past frames $\mathbf { x } ^ { - S _ { 0 } : 0 }$ . Following Rombach et al. (2022), we first train a VAE with encoder E and decoder D, then generate $\mathbf { z } _ { 1 } ^ { \top } = E ( \mathbf { x } ^ { S } )$ conditioned on $\mathbf { c } = E ( \mathbf { x } ^ { - S _ { 0 } : 0 } )$ within the latent space. Diffusion models learn to denoise noise-corrupted data through simple linear interpolation, learning a conditional velocity field with the flow-matching objective:

$$
\mathbf { z } _ { t } = ( 1 - t ) { \boldsymbol { \epsilon } } + t \mathbf { z } _ { 1 } ,\tag{1}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { \mathbf { z } _ { 1 } , \mathbf { c } , \epsilon , t } \left[ \Vert \mathbf { v } _ { \theta } ( \mathbf { z } _ { t } , t \mid \mathbf { c } ) - ( \mathbf { z } _ { 1 } - \epsilon ) \Vert _ { 2 } ^ { 2 } \right] , } \end{array}\tag{2}
$$

where $t \in [ 0 , 1 ]$ denotes the diffusion timestep and $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ is Gaussian noise. At inference, we solve $\begin{array} { r } { \frac { \mathrm { d } \mathbf { z } _ { t } } { \mathrm { d } t } = \mathbf { v } _ { \theta } ( \mathbf { z } _ { t } , t \mid \mathbf { c } ) } \end{array}$ from ${ \bf z } _ { 0 } = \epsilon \mathrm { t o } \hat { { \bf z } } _ { 1 }$ and decode $\hat { \mathbf { x } } ^ { 1 : S } = D ( \hat { \mathbf { z } } _ { 1 } )$

## 3 REVISITING PRECIPITATION NOWCASTING REQUIREMENTS

In this section, we revisit the modeling requirements of precipitation nowcasting and ask whether they necessitate specialized forecasting architectures. We organize the discussion around three aspects: spatiotemporal generative modeling, physical plausibility, and precipitation alignment.

Spatiotemporal generative modeling. Precipitation fields, like natural videos, are continuous spatiotemporal tensors whose structures span multiple scales, from localized convective cells to large-scale frontal systems. A suitable model must capture these hierarchical dependencies while representing the uncertainty of future evolution. The success of DiTs in video generation (OpenAI, 2024; Wan et al., 2025; Seedance et al., 2026) suggest that their scalable spatiotemporal representations provides a suitable foundation for this purpose.

Physical plausibility. Precipitation evolves through atmospheric processes, and plausible forecasts should exhibit consistent motion and physically reasonable temporal evolution. Explicit physical constraints offer one way to encourage such behavior, but they are not the only route. Modern generative models have demonstrated the ability to learn intuitive physics from video (Ali et al., 2025; Ye et al., 2026; Aditi et al., 2026), suggesting that data-driven DiTs can also capture the coherent motion and local evolution from radar sequences.

Precipitation alignment. Precipitation differs from natural video mainly in its data distribution and task objectives. Its evolution combines large-scale advection with localized growth and decay, with precipitation intensities commonly sparse and long-tailed. Moreover, generic generative objectives do not directly optimize meteorological skills such as Critical Success Index (CSI) and Heidke Skill Score (HSS). These discrepancies therefore call for precipitation-aware adaptation of the generative process and learning objective, but do not by themselves prescribe a specialized model or alter the underlying spatiotemporal modeling problem.

Existing nowcasting methods often couple such domain-specific mechanisms with specialized forecasting models or frameworks. This coupling obscures how much task-specific specialization is genuinely needed beyond the capabilities of a general-purpose generative model. The analysis above suggests a cleaner separation: retain a standard DiT for general spatiotemporal modeling, and introduce specialization only where precipitation departs from generic generation.

Takeaway. A standard DiT provides the core modeling capacity required for precipitation nowcasting, while precipitation-specific requirements can be addressed through targeted adaptations rather than model re-design.

## 4 NOWCASTDIT

In this section, we present NowcastDiT, a precipitation nowcasting framework built on a standard Diffusion Transformer with minimal modifications. As illustrated in Figure 2, it promotes temporal coherence through scheduled noise correlation and meteorological skill through timestep-aware rewards, framing precipitation nowcasting as a natural extension of video diffusion modeling.

## 4.1 TRANSFORMER BACKBONE

Precipitation nowcasting shares the core spatiotemporal modeling requirements of video generation, making a standard diffusion transformer a natural backbone. Following standard visual-conditioning designs in video diffusion models (Blattmann et al., 2023a;b; Voleti et al., 2022; Gupta et al., 2024), NowcastDiT conditions future latent generation on historical observations by adding encoded observations to the noisy patch embeddings. One advantage of preserving a task-agnostic backbone is that NowcastDiT can inherit advances from broader generative modeling. Specifically, we incorporate two such advances that directly address the modeling requirements of precipitation nowcasting.

QK-Norm. Radar-based precipitation observations yield a sparse, long-tailed intensity distribution, which can produce large variations in token feature magnitudes and attention logits. To keep attention computation well scaled during optimization, we apply QK-Norm (Henry et al., 2020) to normalize query and key vectors before their dot product.

![](images/6bcd8861835442cea0eae70d2a05c188b0925ef3a78f9bbf7e547341bb83eb3f.jpg)  
Figure 2: Overview of NowcastDiT. Left: A standard Diffusion Transformer predicts future precipitation conditioned on encoded observations, with QK-Norm and 3D RoPE. Top right: The dynamics-aware noise prior (DyPro) schedules cross-frame noise correlation from temporally correlated at the noise endpoint to independent at the clean endpoint. Bottom right: Timestep-aware post-training shifts the reward emphasis from structure to detail along the denoising trajectory.

3D RoPE. Precipitation sequences form a regular spatiotemporal grid, and their evolution is characterized by relative displacement across frames and spatial locations. Following CogVideoX (Yang et al., 2024) and Wan (Wan et al., 2025), we apply RoPE (Su et al., 2024) along the temporal, height, and width axes, enabling attention to encode relative offsets along all three dimensions in spacetime.

Overall, NowcastDiT retains a deliberately simple framework: it relies on a standard DiT without task-specialized forecasting modules, explicit physical constraints, or cascaded architectures.

## 4.2 TEMPORAL COHERENCE WITH SCHEDULED NOISE CORRELATION

Standard video diffusion formulations typically adopt i.i.d. Gaussian noise over the spatiotemporal volume, yielding independent noise realizations across forecast frames. While simple and broadly applicable, this frame-wise independence is less aligned with the strong temporal coherence inherent in precipitation evolution.

Dynamics-aware noise prior. Existing approaches that adapt pretrained image diffusion models for video generation have shown that correlating noise across adjacent forecast frames can improve temporal consistency (Ge et al., 2023; Chang et al., 2024). Motivated by this insight, we introduce a dynamics-aware noise prior $( D y P r o )$ that correlates noise across forecast frames. To construct this prior, we draw on the progressive noise schedule following Dynamical Diffusion (Guo et al., 2025), with the strength of cross-frame correlation controlled by diffusion timestep. Let $s \in \{ 1 , \ldots , S \}$ index the forecast frames, and let $\epsilon _ { t } ^ { 1 : S }$ denote the corresponding noise sequence at timestep t. For a fixed diffusion timestep t, DyPro recursively constructs

$$
\epsilon _ { t } ^ { 1 } = \xi ^ { 1 } , \quad \epsilon _ { t } ^ { s } = \sqrt { \gamma _ { t } } \epsilon _ { t } ^ { s - 1 } + \sqrt { 1 - \gamma _ { t } } \xi ^ { s } .\tag{3}
$$

Here, $\pmb { \xi } ^ { s } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ are independent Gaussian samples, and the diffusion-timestep-dependent inheritance weight is

$$
\gamma _ { t } = \frac { \alpha ^ { 2 } ( 1 - t ) ^ { 2 } } { 1 + \alpha ^ { 2 } ( 1 - t ) ^ { 2 } } ,\tag{4}
$$

where hyperparameter $\alpha \geq 0$ controls the temporal dependence together with t. The recursion preserves the standard Gaussian marginal of each frame, $\epsilon _ { t } ^ { s } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { \bar { I } } )$ , while modifying only the cross-frame covariance. Specifically, for two frames k steps apart, $\mathrm { C o v } ( \epsilon _ { t } ^ { s } , \epsilon _ { t } ^ { s - k } ) = \gamma _ { t } ^ { k / 2 } \mathbf { I } .$ Thus, adjacent frames share the most similar noise, and this similarity decays geometrically with temporal distance k. As denoising proceeds, $\gamma _ { t }$ decreases from $\alpha ^ { 2 } / ( \dot { 1 } + \alpha ^ { 2 } )$ at the beginning to zero at the clean endpoint. DyPro therefore promotes spatiotemporal coherence over the fixed radar grid during early denoising, while progressively relaxing the coupling during later stages to preserve variations among individual forecast frames associated with localized precipitation growth, decay, and intensity evolution.

```python
Algorithm 1 DyPro Training Algorithm 2 DyPro Sampling (w/o CFG)
# x: training latents; c: history # c: history; shape: latent shape
# net(z, t, c): predicts x - e # N: number of sampling steps
t = sample t() ts = linspace(0, 1, N + 1)
z = correlate(randn(shape), 0)
# Construct time-dependent noise
e_init = randn like(x) for t, t_next in zip(ts[:-1], ts[1:]):
e = correlate(e_init, t) v = net(z, t, c)
x = z + (1 - t) <sub>*</sub> v
z = t <sub>*</sub> x + (1 - t) <sub>*</sub> e e = z - t v
v = x - e e_init = decorrelate(e, t)
e = correlate(e_init, t_next)
v_pred = net(z, t, c) z = t_next <sub>*</sub> x + (1 - t_next) <sub>*</sub> e
loss = l2 loss(v - v_pred)
update(net, loss) return z
```

Training and Inference. DyPro is independent of the backbone, preserving the model’s input and output shapes and the form of the training objective. During training, we use DyPro noise to construct z<sub>t</sub> and the flow-matching target (Algorithm 1). At inference, we use a sampler following Dynamical Diffusion that updates noise correlation at each step (Algorithm 2). The sampler requires one network evaluation per step, matching standard Euler sampling in styles.

## 4.3 METEOROLOGICAL SKILL WITH TIMESTEP-AWARE REWARDS

Standard generative models are typically optimized with surrogate objectives such as flow matching and evaluated by metrics aligned with generative fidelity. Yet precipitation nowcasting also emphasizes task-specific meteorological skill that does not necessarily align with these objectives, making flow matching alone insufficient for optimizing forecasting performance.

Existing work has shown that task-specific evaluation metrics can serve directly as verifiable rewards for optimizing generative prediction models (Wu et al., 2025; Xue et al., 2025). Inspired by the pretraining–posttraining paradigm of foundation models, we adopt a two-stage training strategy that separates general generative modeling from meteorological skill optimization:

• Stage 1: Pretrain a diffusion model from scratch with surrogate flow matching objective.

• Stage 2: Posttrain the diffusion model by directly optimizing meteorological skills.

Skill optimization via GRPO. We formulate the posttraining stage as a GRPO objective for diffusion models (Shao et al., 2024; Liu et al., 2025). For each conditioning context paired with a ground-truth future sequence $\mathbf { x } ^ { * }$ , we sample a group of forecasts $\{ \hat { \mathbf { x } } ^ { ( 1 ) } , \hat { \mathbf { x } } ^ { ( 2 ) } , \hat { \mathbf { \alpha } } ^ { } , \hat { \mathbf { x } } ^ { ( \bar { G } ) } \}$ through a sampling trajectory where selected ODE transitions are converted into SDE transitions that serve as stochastic policy actions with tractable likelihoods, following the design of MixGRPO (Li et al., 2025). We score each $\hat { \mathbf { x } } ^ { ( i ) }$ against $\mathbf { x } ^ { * }$ and update the stochastic transitions using the clipped group relative objective

$$
\hat { A } _ { n } ^ { i } = \frac { R ( \hat { \mathbf { x } } ^ { ( i ) } , \mathbf { x } ^ { * } ; t _ { n } ) - \operatorname* { m e a n } \left( \left\{ R ( \hat { \mathbf { x } } ^ { ( j ) } , \mathbf { x } ^ { * } ; t _ { n } ) \right\} _ { j = 1 } ^ { G } \right) } { \operatorname* { s t d } \left( \left\{ R ( \hat { \mathbf { x } } ^ { ( j ) } , \mathbf { x } ^ { * } ; t _ { n } ) \right\} _ { j = 1 } ^ { G } \right) } ,\tag{5}
$$

$$
\mathcal { I } _ { \mathrm { G R P O } } ( \theta ) = \frac { 1 } { G | \mathcal { W } _ { \ell } | } \sum _ { n \in \mathcal { W } _ { \ell } } \sum _ { i = 1 } ^ { G } \operatorname* { m i n } \Big ( \rho _ { n } ^ { i } ( \theta ) \hat { A } _ { n } ^ { i } , \mathrm { c l i p } \big ( \rho _ { n } ^ { i } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon \big ) \hat { A } _ { n } ^ { i } \Big ) ,\tag{6}
$$

where R denotes the reward evaluated on the final forecast, $\mathcal { W } _ { \ell }$ indexes the stochastic transitions included in the GRPO update, and $\rho _ { n } ^ { i }$ is the policy likelihood ratio for forecast i at transition n. Further details on GRPO optimization and mixed ODE–SDE sampling are provided in Appendix C.3 and Appendix C.1, respectively.

Table 1: Performance comparison on SEVIR and MRMS. ↑ indicates higher is better and ↓ indicates lower is better. Bold and underlined values denote the best and second-best results, respectively.
<table><tr><td colspan="8">SEVIR MRMS</td></tr><tr><td>Method</td><td>CSI↑  $\mathrm { C S I _ { 1 8 1 } ^ { \uparrow } \ C S I _ { 2 1 9 } ^ { \uparrow } }$ </td><td>HSS↑</td><td>LPIPS↓ SSIM↑</td><td>Method</td><td>CSI↑</td><td> $\mathrm { C S I _ { 1 6 } ^ { \uparrow } \ C S I _ { 3 2 } ^ { \uparrow } }$  HSS↑</td><td>LPIPS↓ SSIM↑</td></tr><tr><td>ConvLSTM</td><td>0.2912 0.0684 0.0336</td><td>0.3571</td><td>0.2989 0.7257</td><td>ConvLSTM</td><td>0.2159 0.0514 0.0172 0.2877</td><td>0.3013</td><td>0.8978</td></tr><tr><td>PhyDNet</td><td>0.2874 0.0664</td><td>0.0211 0.3523</td><td>0.3081 0.7252</td><td>PhyDNet</td><td>0.2332 0.0635 0.02680.3111</td><td>0.3005</td><td>0.9001</td></tr><tr><td>Earthformer</td><td>0.2752 0.0496</td><td>0.0213 0.3391</td><td>0.3376 0.7142</td><td>Earthformer</td><td>0.2353 0.0691 0.0297 0.3152</td><td>0.3001</td><td>0.8994</td></tr><tr><td>SimVP</td><td>0.2951 0.0753</td><td>0.0416 0.3630</td><td>0.3113 0.7247</td><td>SimVP</td><td></td><td>0.2366 0.0657 0.0277 0.3155 0.2992</td><td>0.9006</td></tr><tr><td>AlphaPre</td><td>0.2980 0.0860</td><td>0.04360.3677</td><td>0.2897 0.7301</td><td>AlphaPre</td><td></td><td>0.2309 0.0587 0.0211 0.2933 0.3014</td><td>0.8944</td></tr><tr><td>DiffCast</td><td>0.2977 0.0915</td><td>0.0571 0.4033</td><td>0.1812 0.6774</td><td></td><td>NowcastNet0.22440.1145</td><td>0.0729 0.3083 0.2007</td><td>0.8502</td></tr><tr><td>PreDiff</td><td>0.2859 0.0893</td><td>0.0487 0.3647</td><td>0.1543 0.7006</td><td>PreDiff</td><td></td><td>0.2283 0.1083 0.0695 50.3148 0.1906</td><td>0.8633</td></tr><tr><td>CasCast</td><td>0.2905 0.0930</td><td>0.0503 0.3703</td><td>0.1583 0.7012</td><td>CasCast</td><td></td><td>0.2224 0.0925 0.0527 0.3064 0.2045</td><td>0.8791</td></tr><tr><td>DiT</td><td>0.2926 0.0948</td><td>0.0555 0.3733</td><td>0.1545</td><td>0.7048 DiT</td><td></td><td>0.2343 0.1015 0.0611 0.3223 0.1953</td><td>0.8824</td></tr><tr><td>NowcastDiT 0.3240</td><td>0.1241</td><td>0.0750 0.4148</td><td>0.1469</td><td>0.7197</td><td>NowcastDiT 0.2696 0.1373 0.0958 0.3668</td><td>0.1827</td><td>0.8642</td></tr></table>

Timestep-aware rewards. Reinforcement learning for diffusion models typically uses the same terminal reward across all denoising transitions. However, different denoising transitions are expected to contribute differently to the final forecast: transitions near the noise endpoint establish precipitation coverage and large-scale spatial patterns, whereas transitions near the clean endpoint refine local intensity variations and heavy-precipitation cores. This uniform supervision may therefore be suboptimal. We introduce a timestep-aware terminal reward that gradually shifts its emphasis from precipitation structure to local detail as denoising proceeds:

$$
R _ { \mathrm { n o i s e } } \bigl ( \hat { \mathbf { x } } , \mathbf { x } ^ { * } \bigr ) = \sum _ { \tau \in \mathcal { T } _ { \mathrm { l o w } } } \lambda _ { \tau } \mathrm { C S I } _ { \tau } ^ { \mathrm { p o o l } } \bigl ( \hat { \mathbf { x } } , \mathbf { x } ^ { * } \bigr ) + \lambda _ { s } \mathrm { S S I M } ( \hat { \mathbf { x } } , \mathbf { x } ^ { * } ) ,\tag{7}
$$

$$
R _ { \mathrm { c l e a n } } ( \hat { \mathbf { x } } , \mathbf { x } ^ { * } ) = \sum _ { \tau \in \mathcal { T } _ { \mathrm { h i g h } } } \lambda _ { \tau } \mathrm { C S I } _ { \tau } ( \hat { \mathbf { x } } , \mathbf { x } ^ { * } ) - \lambda _ { p } \mathrm { L P I P S } ( \hat { \mathbf { x } } , \mathbf { x } ^ { * } ) ,\tag{8}
$$

$$
R ( \hat { \bf x } , { \bf x } ^ { * } ; t ) = ( 1 - t ) R _ { \mathrm { n o i s e } } ( \hat { \bf x } , { \bf x } ^ { * } ) + t R _ { \mathrm { c l e a n } } ( \hat { \bf x } , { \bf x } ^ { * } ) ,\tag{9}
$$

where $\mathcal { T } _ { \mathrm { l o w } }$ and $\mathcal { T } _ { \mathrm { h i g h } }$ denote the light-precipitation and heavy-precipitation threshold sets, respectively. The coefficients $\lambda _ { s }$ and $\lambda _ { p }$ balance the constituent metrics, while the remaining weight is distributed uniformly across the corresponding CSI thresholds: $\lambda _ { \tau } = ( 1 - \lambda _ { s } ) / | \mathcal { T } _ { \mathrm { l o w } } |$ for $R _ { \mathrm { n o i s e } }$ and $\lambda _ { \tau } = ( 1 - \lambda _ { p } ) / | \mathcal { T } _ { \mathrm { h i g h } } |$ for $R _ { \mathrm { c l e a n } }$ . We empirically set $\lambda _ { s } = 0 . 5$ and $\lambda _ { p } = 0 . 9$

Specifically, $R _ { \mathrm { n o i s e } }$ combines spatially tolerant low-threshold pooled CSI for event coverage with SSIM (Wang et al., 2004) for local structural agreement, while $R _ { \mathrm { c l e a n } }$ combines high-threshold CSI for localized intense precipitation with negative LPIPS (Zhang et al., 2018) for perceptual similarity in deep feature space. Since $t = 0$ and $t \ : = \ : 1$ correspond to the noise and clean endpoints, respectively, the reward gradually shifts from $R _ { \mathrm { n o i s e } }$ to $R _ { \mathrm { c l e a n } }$ , emphasizing precipitation structure early and local detail later. All rewards are computed on the final decoded forecast, with the active transition time determining the metric mixture used for credit assignment in Equation 5.

## 5 EXPERIMENTS

We evaluate NowcastDiT on two radar-based benchmarks, SEVIR and MRMS. Comparisons with deterministic and generative nowcasting baselines assess both meteorological skill and perceptual quality. Additional experiments examine how our design choices affect forecasting performance.

## 5.1 EXPERIMENTAL SETUP

Datasets. SEVIR (Veillette et al., 2020) is a spatiotemporal radar observation dataset covering weather events across the United States. Following prior work (Yu et al., 2024; Lin et al., 2025), we use Vertically Integrated Liquid (VIL) observations at a 5-minute temporal resolution to predict 20 future frames given 5 observed frames at $1 2 8 \times 1 2 8$ resolution. MRMS (Zhang et al., 2016) is a composite radar dataset collected over the contiguous United States. Following NowcastNet (Zhang et al., 2023), we use precipitation-rate observations at a 10-minute temporal resolution to predict 20 future frames given 4 observed frames at $2 5 6 \times 2 5 6$ resolution. Additional dataset details, splits, and preprocessing are provided in Appendix D.1.

![](images/6142fbbc8dd8f6b067ebceb2873d4bb62584f11410cca6dad1238e70c797736c.jpg)  
Figure 3: Qualitative comparison on SEVIR. Forecasts are shown every 10 minutes up to 100 minutes. Colors indicate VIL values.

Evaluation. We compare NowcastDiT with deterministic and generative methods. The deterministic baselines include ConvLSTM (Shi et al., 2015), PhyDNet (Guen & Thome, 2020), Earthformer (Gao et al., 2022b), SimVP (Gao et al., 2022a), and AlphaPre (Lin et al., 2025), while the generative baselines include NowcastNet (Zhang et al., 2023), PreDiff (Gao et al., 2023), Diff-Cast (Yu et al., 2024), and CasCast (Gong et al., 2024). We evaluate meteorological skill using the Critical Success Index (CSI) (Schaefer, 1990) and Heidke Skill Score (HSS) (Jolliffe & Stephenson, 2012), and perceptual quality using the Structural Similarity Index (SSIM) (Wang et al., 2004) and Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018). CSI measures event detection accuracy, while HSS accounts for agreement expected by chance. Both are computed from pixelwise contingency counts aggregated over the evaluation set at each forecast lead time. We report their means over lead times and thresholds, using {16, 74, 133, 160, 181, 219} in the VIL for SEVIR and {1, 2, 4, 8, 16, 32} mm $\mathrm { h } ^ { - 1 }$ for MRMS. We additionally report high-threshold CSI to assess intense precipitation events. SSIM measures local structural agreement, whereas LPIPS measures perceptual distance in deep feature space. Higher CSI, HSS, and SSIM and lower LPIPS indicate better performance. Metric definitions are provided in Appendix D.2.

Implementation Details. We train NowcastDiT using AdamW with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ , a batch size of 32, and learning rates of $1 0 ^ { - 4 }$ for pretraining and $1 0 ^ { - 5 }$ for post-training. The default backbone has 12 Transformer blocks, a hidden dimension of 768, and 12 attention heads. The DyPro parameter α is set to 0.5. Post-training uses a group size of 16, 10 denoising steps, an SDE window size of 4, and a policy ratio clipping range of $\mathbf { \bar { 1 0 } ^ { - \dot { 4 } } }$ . For reward computation, we apply $4 \times 4$ max pooling to low-threshold CSI and set the SSIM and LPIPS mixing weights to 0.5 and 0.9, respectively. All training runs use four NVIDIA A100 GPUs with 80 GB of memory each. Additional post-training settings are provided in Appendix C.6.

## 5.2 MAIN RESULTS

We compare NowcastDiT with deterministic and generative nowcasting methods on SEVIR and MRMS, reported in Table 1. The standard DiT baseline already achieves competitive performance against specialized nowcasting models, demonstrating the strength of its general-purpose spatiotemporal modeling. Relative to the strongest baselines, NowcastDiT improves aggregate CSI by 8.7% and 15.1%, and HSS by 2.9% and 13.8%, on SEVIR and MRMS, respectively. It also reduces LPIPS by 4.8% and 4.1% while achieving the best SSIM among generative methods on SEVIR. On high-threshold CSI that directly reflects the ability to detect rare and operationally important intense rainfall events, NowcastDiT achieves relative gains of 19.9%–31.4% across the reported high thresholds, showing a clear advantage in heavy-rainfall prediction. We also provide per-lead-time CSI and HSS curves, shown in Appendix E.1. Taken together, these results show that a standard DiT, with precipitation-specific alignment can achieve strong nowcasting performance while retaining a simple architecture.

![](images/cc494575a26bd3159766fb99d2cb0ee47df6136c49fd10ea9b9345795828b6ed.jpg)  
Figure 4: Qualitative comparison on MRMS. Forecasts are shown every 20 minutes up to 200 minutes. Colors indicate precipitation rate in mm $\mathbf { h } ^ { - 1 }$

Qualitative comparisons. Figures 3 and 4 compare forecast sequences on SEVIR and MRMS, respectively. In the SEVIR example, NowcastDiT retains localized high-intensity regions at longer lead times, whereas the DiT baseline produces smoother precipitation fields. In the MRMS example, NowcastDiT better preserves the shape of the main rainband. These selected examples complement the test-set comparisons in Table 1.

## 5.3 ANALYSES

We analyze how the following components and design choices affect model performance on SEVIR.

Model parameter scaling. Diffusion Transformer is validated as a scalable backbone, and we investigate whether this scaling benefits extend to precipitation nowcasting. Following Peebles & Xie (2023), we evaluate NowcastDiT across a range of model sizes. As shown in Figure 5(a), increasing model size consistently improves aggregate CSI. This finding supports model scaling as an effective strategy for strengthening the forecasting capabilities of a standard DiT backbone, which forms the initial motivation of NowcastDiT.

CFG and skill-aware post-training. Classifier-free guidance (CFG) (Ho & Salimans, 2022) is a common capability of standard diffusion models and should be naturally inherited when adopting them for precipitation nowcasting. We therefore examine whether skill-aware post-training can provide complementary gains on top of CFG, whose integration with diffusion post-training is not always trivial (Zheng et al., 2026). As shown in Figure 5(b), both CFG and RL improve aggregate CSI over the base model, and their combination achieves the best performance. This suggests that meteorological alignment provides an additional optimization axis beyond standard diffusion guidance. Detailed methods and results are provided in Appendix E.3.

DyPro correlation strength. For simplicity, NowcastDiT adopts the factor α = 0.5 controlling temporal correlations in DyPro. To assess the senstivity, we vary the DyPro correlation strength $\alpha \in \{ 0 , 0 . 2 5 , 0 . 3 3 , 0 . 6 6 , 0 . 7 5 \}$ , where $\alpha = 0$ infers the i.i.d. noise baseline. Figure 5(c) shows that all tested nonzero strengths improve aggregate CSI, with performance peaking at a moderate strength and declining as the correlation increases further, indicating the robustness of α in DyPro. Detailed results are provided in Appendix E.4.

Timestep-aware reward. The reinforcement learning stage of NowcastDiT uses a stage-aware reward to account for the different roles of reward metrics near the noise and clean endpoints. We compare it with a static baseline that mixes the same rewards without timestep-dependent weighting. As shown in Figure 5(d), timestep-aware weighting consistently performs better. To isolate this effect, we further repeat the comparison under CSI-only and perceptual-only objectives (i.e., $\lambda _ { s } =$ $\lambda _ { p } = 0$ and $\lambda _ { \tau } = 0$ , respectively), where the same trend holds. These results confirm the benefit

(a) Model Scaling

![](images/1e52d6c1094fa240672e5624127e3ed9cff3137c2bb99d58f2c3de2ee1ab8608.jpg)

![](images/dad17ca11f1d171ff7a1071f2fda3adea8a86fe731a7ef56d4f88f6ef08161f7.jpg)

(c) DyPro  
![](images/5b254be751f8c0089e519f1cb2a9ad0f5013e55d2a660efbaa7fa6b887173627.jpg)

(d) Timestep-aware reward  
![](images/b91b228446fb511fc414e6d96ceae46fe245583776a32747cb9f5ac3a7c4cdbb.jpg)

![](images/21eadf7822ca4bb5228cc13d3ebe9707076a7c44d7706e54cf143f1a8558c9cf.jpg)  
Figure 5: Analysis on SEVIR. Bar lengths indicate CSI gains relative to the first setting in each panel; labels report absolute CSI values.

of timestep-aware weighting across reward objectives, while mixed metrics provide a better balance between meteorological skill and perceptual quality.

## 6 RELATED WORK

Diffusion models for precipitation nowcasting. Diffusion models have become an important paradigm for probabilistic nowcasting due to their stable training dynamics and strong generative fidelity (Gao et al., 2023; Yu et al., 2024). Existing diffusion-based approaches often incorporate task-specific mechanisms to improve meteorological applicability (Gao et al., 2023; Yu et al., 2024; Gong et al., 2024; Wen et al., 2026). PreDiff (Gao et al., 2023) introduces intensity continuity guidance over a spatiotemporal architecture to strength physical alignment. DiffCast (Yu et al., 2024) introduces decomposited training phases for the modeling of global deterministic motion and local stocastic variations. CasCast (Gong et al., 2024) employs a cascaded DiT framework in latent space to decouple deterministic and stochastic precipitation modeling for improved efficiency and extreme event fidelity. These approaches couple domain-specific mechanisms with architectural and framework design, leaving unresolved whether specialized designs are necessary for effective precipitation nowcasting. In contrast, NowcastDiT examines how a standard DiT can support effective nowcasting and address domain-specific requirements within its design space.

Advances in generative modeling techniques. Recent advances in image and video generation have substantially broadened the modeling capabilities of diffusion transformers (Peebles & Xie, 2023; Esser et al., 2024; Ma et al., 2025; Wan et al., 2025). Architectural refinements improve attention stability and positional modeling (Henry et al., 2020; Su et al., 2024; Gupta et al., 2024; Yang et al., 2024). Noise-prior designs introduce temporal dependencies to promote coherent video generation (Ge et al., 2023; Chang et al., 2024; Guo et al., 2025). Post-training strategies align generated outputs with task-specific objectives through reinforcement learning (Black et al., 2024; Liu et al., 2025; Xue et al., 2025; Li et al., 2025; Wu et al., 2025). Together, these developments establish the feasibility of meeting domain-specific modeling requirements within the design space of a standard DiT. Building on this foundation, NowcastDiT further develops noise-prior and posttraining designs to model precipitation dynamics and enhance meteorological skill.

## 7 CONCLUSION

This work demonstrates that a standard Diffusion Transformer, with targeted adaptations to its diffusion and training strategies, is sufficient for effective precipitation nowcasting. NowcastDiT combines DyPro for temporally coherent forecasting with skill-aware post-training for meteorological alignment. Experiments on SEVIR and MRMS show strong performance in both meteorological skill and perceptual quality, supporting standard DiTs as a simple and scalable framework for precipitation nowcasting. Future work will evaluate its generalization across observation systems, resolutions, and longer forecast horizons.

## AI USE STATEMENT

AI tools assisted with drafting and revising portions of the manuscript, language polishing, literature search, figure preparation, verification of mathematical proof and use of the ICLR 2027 template. The authors personally reviewed and revised the AI-assisted content, verified its accuracy, and take full responsibility for the final manuscript, including all claims and results.

## REFERENCES

Niket Aditi, Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, et al. Cosmos 3: Omnimodal world models for physical ai. arXiv, 2026.

Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv, 2025.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In ICLR, 2024.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv, 2023a.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023b.

Pascal Chang, Jingwei Tang, Markus Gross, and Vinicius C Azevedo. How i warped your noise: a temporally-correlated noise prior for diffusion models. In ICLR, 2024.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam¨ Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv, 2024.

Zhangyang Gao, Cheng Tan, Lirong Wu, and Stan Z Li. Simvp: Simpler yet better video prediction. In CVPR, 2022a.

Zhihan Gao, Xingjian Shi, Hao Wang, Yi Zhu, Yuyang Bernie Wang, Mu Li, and Dit-Yan Yeung. Earthformer: Exploring space-time transformers for earth system forecasting. In NeurIPS, 2022b.

Zhihan Gao, Xingjian Shi, Boran Han, Hao Wang, Xiaoyong Jin, Danielle Maddix, Yi Zhu, Mu Li, and Yuyang Bernie Wang. Prediff: Precipitation nowcasting with latent diffusion models. In NeurIPS, 2023.

Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-Yu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In ICCV, 2023.

Junchao Gong, Lei Bai, Peng Ye, Wanghan Xu, Na Liu, Jianhua Dai, Xiaokang Yang, and Wanli Ouyang. Cascast: Skillful high-resolution precipitation nowcasting via cascaded modelling. In ICML, 2024.

Vincent Le Guen and Nicolas Thome. Disentangling physical dynamics from unknown factors for unsupervised video prediction. In CVPR, 2020.

Xingzhuo Guo, Yu Zhang, Baixu Chen, Haoran Xu, Jianmin Wang, and Mingsheng Long. Dynamical diffusion: Learning temporal dynamics with diffusion models. In ICLR, 2025.

Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and Jose Lezama. Photorealistic video generation with diffusion models. In´ ECCV, 2024.

Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Findings ofEMNLP, 2020.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.

Ian T Jolliffe and David B Stephenson. Forecast verification: a practitioner’s guide in atmospheric science. John Wiley & Sons, 2012.

Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, and Zhao Zhong. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde, 2025.

Kenghong Lin, Baoquan Zhang, Demin Yu, Wenzhi Feng, Shidong Chen, Feifan Gao, Xutao Li, and Yunming Ye. AlphaPre: Amplitude-phase disentanglement model for precipitation nowcasting. In CVPR, 2025.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. In NeurIPS, 2025.

Xin Ma, Yaohui Wang, Xinyuan Chen, Gengyun Jia, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. In TMLR, 2025.

OpenAI. Sora: Video generation models as world simulators. https://openai.com/ research/video-generation-models-as-world-simulators, 2024. Accessed: 2024-04-01.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In CVPR, 2022.

Joseph T Schaefer. The critical success index as an indicator of warning skill. Weather and forecasting, 1990.

Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv, 2026.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv, 2024.

Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, and Wang-chun Woo. Convolutional lstm network: A machine learning approach for precipitation nowcasting. In NeurIPS, 2015.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024.

Mark Veillette, Siddharth Samsi, and Chris Mattioli. Sevir: A storm event imagery dataset for deep learning applications in radar and satellite meteorology. In NeurIPS, 2020.

Vikram Voleti, Alexia Jolicoeur-Martineau, and Chris Pal. Mcvd-masked conditional video diffusion for prediction, generation, and interpolation. In NeurIPS, 2022.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Yong Wang, Estelle De Coning, Abdoulaye Harou, and Wilfried Jacobs. Guidelines for nowcasting techniques. Wmo-no. 1198, World Meteorological Organization, Geneva, Switzerland, 2017.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600– 612, 2004.

Penghui Wen, Mengwei He, Patrick Filippi, Na Zhao, Feng Zhang, Thomas Francis Bishop, Zhiyong Wang, and Kun Hu. DuoCast: Duo-probabilistic diffusion for precipitation nowcasting. In AAAI, 2026.

Jialong Wu, Shaofeng Yin, Ningya Feng, and Mingsheng Long. RLVR-World: Training world models with reinforcement learning. In Advances in Neural Information Processing Systems, volume 38, 2025.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv, 2025.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv, 2024.

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv, 2026.

Demin Yu, Xutao Li, Yunming Ye, Baoquan Zhang, Chuyao Luo, Kuai Dai, Rui Wang, and Xunlai Chen. Diffcast: A unified framework via residual diffusion for precipitation nowcasting. In CVPR, 2024.

Jian Zhang, Kenneth Howard, Carrie Langston, Brian Kaney, Youcun Qi, Lin Tang, Heather Grams, Yadong Wang, Stephen Cocks, Steven Martinaitis, et al. Multi-radar multi-sensor (MRMS) quantitative precipitation estimation: Initial operating capabilities. Bulletin of the American Meteorological Society, 97(4):621–638, 2016.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Yuchen Zhang, Mingsheng Long, Kaiyuan Chen, Lanxiang Xing, Ronghua Jin, Michael I Jordan, and Jianmin Wang. Skilful nowcasting of extreme precipitation with nowcastnet. Nature, 2023.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. In ICLR, 2026.

Jincheng Zhong, Xiangcheng Zhang, Jianmin Wang, and Mingsheng Long. Domain guidance: A simple transfer approach for a pre-trained diffusion model. In ICLR, 2025.

## A MODEL ARCHITECTURE AND CONFIGURATION

## A.1 VAE ARCHITECTURE AND TRAINING

We adopt a 2D VAE which compress only the spatial resolution which acts independently across frames. The pretrained VAE provides the encoder E and decoder D and remains frozen during flow-matching pretraining and RL post-training. Table 2 summarizes its architecture and training settings.

Table 2: VAE architecture and training configuration. Parameters include the encoder, decoder, and latent projections.
<table><tr><td>Setting SEVIR</td><td>MRMS</td></tr><tr><td>Architecture</td></tr><tr><td>Parameters 82M</td></tr><tr><td>Input shape  $2 5 \times 1 \times 1 2 8 \times 1 2 8$   $2 4 \times 1 \times 2 5 6 \times 2 5 6$ </td></tr><tr><td>Encoder stages 4</td></tr><tr><td>Residual blocks / encoder stage 2</td></tr><tr><td>Encoder channels [128, 256, 512, 512]</td></tr><tr><td>Decoder stages 4</td></tr><tr><td>Residual blocks / decoder stage 3</td></tr><tr><td>Decoder channels [512, 512, 256, 128]</td></tr><tr><td>Latent channels 16</td></tr><tr><td>Latent resolution  $1 6 \times 1 6$   $3 2 \times 3 2$ </td></tr><tr><td>Spatial compression 8×</td></tr><tr><td>Temporal compression 1×</td></tr><tr><td>Normalization GroupNorm, 32 groups</td></tr><tr><td>Activation SiLU</td></tr><tr><td></td></tr><tr><td>Training # steps 1,488,000 2,000,000</td></tr><tr><td>AdamW,  $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 5 , 0 . 9 )$ </td></tr><tr><td>Optimizer</td></tr><tr><td>Batch size 8 Learning rate  $1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td></tr><tr><td>0.01 EMA decay</td></tr><tr><td>0.9999</td></tr><tr><td>Gradient clipping norm 1.0</td></tr><tr><td>Adversarial warmup steps 100,000</td></tr><tr><td></td></tr></table>

## A.2 TRANSFORMER ARCHITECTURE, TRAINING, AND INFERENCE

Table 3 collects the default architecture, flow-matching pretraining settings, and pretrained-model inference configurations for MRMS and SEVIR. RL post-training settings are reported separately in Appendix C.6.

History Conditioning Let $\mathbf { z } _ { \mathrm { c o n d } } ~ \in ~ \mathbb { R } ^ { B \times T \times C \times H \times W }$ denote the conditioning sequence, which places the observed latent frames in the first L positions and masks the remaining future positions with zeros. The conditioner first applies a frame-wise convolutional patch encoder with the same patch size as the noisy-input embedder. It then adds a learnable spatial-topology embedding and a learnable frame-index embedding, followed by a SiLU activation and a zero-initialized $1 \times 1$ convolution. Denoting the resulting condition features by c and the noisy patch embeddings by $\mathbf { h } _ { t } .$ conditioning is performed through

$$
\widetilde { \mathbf { h } } _ { t } = \mathbf { h } _ { t } + \mathbf { c } .\tag{10}
$$

The conditioned embeddings are then flattened into spatiotemporal tokens and processed by the standard DiT blocks. Thus, the conditioner changes neither the Transformer blocks nor their input– output interface and requires no cross-attention or task-specific fusion pathway.

Table 3: Transformer architecture, flow-matching pretraining, and inference configuration. Parameters include the conditioner, time embedding, and output layer. Inference settings refer to the pretrained-model evaluation configurations.
<table><tr><td>Setting</td><td>SEVIR</td><td>MRMS</td></tr><tr><td colspan="3">Architecture</td></tr><tr><td>Parameters</td><td>130M</td><td></td></tr><tr><td>Frames (history / future)</td><td></td><td>4/20</td></tr><tr><td>Transformer blocks</td><td>12</td><td></td></tr><tr><td>Hidden dimension</td><td>768</td><td></td></tr><tr><td>Feedforward dimension</td><td>3072</td><td></td></tr><tr><td>Attention heads</td><td>12</td><td></td></tr><tr><td>Head dimension</td><td>64</td><td></td></tr><tr><td>Patch size</td><td> $2 \times 2$ </td><td></td></tr><tr><td>Tokens per sample</td><td> $2 5 \times 8 \times 8 = 1 6 0 0$ </td><td> $2 4 \times 1 6 \times 1 6 = 6 1 4 4$ </td></tr><tr><td>Latent channels</td><td>16</td><td></td></tr><tr><td>Block normalization</td><td>LayerNorm</td><td></td></tr><tr><td>QK normalization</td><td>RMSNorm</td><td></td></tr><tr><td>Positional encoding</td><td>3D RoPE</td><td></td></tr><tr><td>Time modulation</td><td>adaLN-Zero</td><td></td></tr><tr><td>FFN activation</td><td>GELU</td><td></td></tr><tr><td>History fusion</td><td>Additive</td><td></td></tr><tr><td colspan="3">Training</td></tr><tr><td># steps</td><td>372,000</td><td>500,000</td></tr><tr><td>Optimizer</td><td>AdamW,  $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ </td><td></td></tr><tr><td>Batch size</td><td>32</td><td></td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 4 }$ </td><td></td></tr><tr><td>Learning rate schedule</td><td>Constant</td><td></td></tr><tr><td>EMA decay</td><td>0.9999</td><td></td></tr><tr><td>Gradient clipping norm</td><td>1.0</td><td></td></tr><tr><td>Time sampler</td><td>Logit-normal,</td><td> $\mu = - 1 , \sigma = 1$ </td></tr><tr><td>DyPro correlation strength α</td><td>0.5</td><td></td></tr><tr><td>Condition dropout</td><td>0.1 (zero conditioning)</td><td></td></tr><tr><td colspan="3">Inference</td></tr><tr><td>Sampler</td><td>Euler with DyPro (Algorithm 2)</td><td></td></tr><tr><td>Sampling steps</td><td>10</td><td></td></tr><tr><td>Timestep grid</td><td>Uniform in [0, 1]</td><td></td></tr><tr><td>CFG weight (without RL)</td><td>1.6</td><td></td></tr></table>

Classifier-free guidance and Domain Guidance. During inference, we adopt classifier-free guidance that adjusts the influence of historical conditioning. Classifier-free guidance (Ho & Salimans, 2022) adjusts the influence of historical conditioning by combining conditional and unconditional velocity predictions from the same checkpoint. Let $\theta _ { \mathrm { p r e } }$ denote the parameters before RL posttraining and let ∅ denote zero conditioning, consistent with the condition-dropout setting in Table 3. The guided prediction is

$$
\mathbf { v } ^ { \mathrm { C F G } } ( \mathbf { z } _ { t } , t \mid \mathbf { c } ) = w \mathbf { v } _ { \boldsymbol { \theta } _ { \mathrm { p r e } } } ( \mathbf { z } _ { t } , t \mid \mathbf { c } ) + ( 1 - w ) \mathbf { v } _ { \boldsymbol { \theta } _ { \mathrm { p r e } } } ( \mathbf { z } _ { t } , t \mid \boldsymbol { \theta } ) ,\tag{11}
$$

where w is the guidance weight and $w = 1$ recovers the conditional prediction without guidance.   
We adopt $w = 1 . 6$ at pretraining stage.

## B FOUNDATION, TRAINING AND SAMPLING WITH DYPRO

We describe the noise transform, training target, and deterministic sampling update associated with the DyPro parameterization in Section 4.2, followed by their distributional properties and continuous-time interpretation. The stochastic extension used for post-training is presented in Appendix B.4. Diffusion time increases from noise at $t = 0$ to data at $t = 1$ . We condition throughout

Algorithm 3 DyPro Noise Correlation and Decorrelation

```python
# xi, e: frame sequences; s indexes frames
# t: diffusion timestep; alpha: shared DyPro strength
def correlate(xi, t):
a = alpha (1 - t)
q = 1 / sqrt(1 + a 2)
r = a <sub>*</sub> q
e = xi.copy()
for s in range(1, len(xi)):
e[s] = r e[s - 1] + q xi[s]
return e
def decorrelate(e, t):
a = alpha <sub>*</sub> (1 - t)
q = 1 / sqrt(1 + a 2)
r = a <sub>*</sub> q
xi = e.copy()
for s in range(1, len(e)):
xi[s] = (e[s] - r <sub>*</sub> e[s - 1]) / q
return xi
```

on the history c and write $\mathbf { y } = \mathbf { z } _ { 1 }$ for the clean future latent sequence. Each of its S frames has d latent coordinates, so the flattened sequence has dimension $m = S d$ . The Gaussian innovations are independent of y and c.

## B.1 CORRELATED NOISE

For finite $\alpha \geq 0 ,$ , define

$$
r _ { t } = \sqrt { \gamma _ { t } } = \frac { \alpha ( 1 - t ) } { \sqrt { 1 + \alpha ^ { 2 } ( 1 - t ) ^ { 2 } } } , \qquad q _ { t } = \sqrt { 1 - \gamma _ { t } } = \frac { 1 } { \sqrt { 1 + \alpha ^ { 2 } ( 1 - t ) ^ { 2 } } } .\tag{12}
$$

The frame-wise recursion defined in Equation 3, practically implemented through correlate in Algorithm $3 .$ , denotes a lower-triangular matrix representation. With frame indices $s , j \in$ $\{ 0 , \ldots , S - 1 \}$ , let

$$
( C _ { t } ) _ { s j } = \left\{ \begin{array} { l l } { r _ { t } ^ { s } , } & { j = 0 , } \\ { q _ { t } r _ { t } ^ { s - j } , } & { 1 \leq j \leq s , } \\ { 0 , } & { j > s . } \end{array} \right. \quad \mathscr { C } _ { t } = C _ { t } \otimes \mathbf { I } _ { d } , \qquad \epsilon _ { t } = \mathscr { C } _ { t } \pmb { \xi } ,\tag{13}
$$

where $\pmb { \xi } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { m } )$ and $r _ { t } ^ { 0 } = 1$ , including when $r _ { t } = 0$

Because $q _ { t } \ > \ 0$ , this transform is invertible. Its inverse $\mathcal { C } _ { t } ^ { - 1 } \ : = \ : C ^ { - 1 }$ ⊗ $\mathbf { I } _ { d } ,$ implemented by decorrelate in Algorithm 3, follows directly from the recursion:

$$
\xi ^ { 1 } = \epsilon _ { t } ^ { 1 } , \qquad \xi ^ { s } = \frac { \epsilon _ { t } ^ { s } - r _ { t } \epsilon _ { t } ^ { s - 1 } } { q _ { t } } , \quad s \ge 2 .\tag{14}
$$

The joint noise $\epsilon _ { t }$ is Gaussian because it is a linear transform of independent Gaussian innovations. Since $r _ { t } ^ { 2 } + q _ { t } ^ { 2 } = 1$ , the recursion preserves the standard Gaussian marginal of each frame. Independence of subsequent innovations gives $\mathrm { C o v } ( \epsilon _ { t } ^ { s } , \epsilon _ { t } ^ { j } ) = r _ { t } ^ { | s - j | } { \bf I } _ { d } = \gamma _ { t } ^ { | s - j | / 2 } { \bf I } _ { d } ,$ yielding the full covariance and distribution

$$
\mathbf { Q } _ { t } = \mathcal { C } _ { t } \mathcal { C } _ { t } ^ { \top } = \big [ r _ { t } ^ { | s - j | } \big ] _ { s , j = 0 } ^ { S - 1 } \otimes \mathbf { I } _ { d } , \qquad \epsilon _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { Q } _ { t } ) .\tag{15}
$$

For $\alpha = 0 .$ , or at $t = 1 , r _ { t } = 0$ and $q _ { t } = 1$ , so $\mathcal { C } _ { t } = \mathbf { I } _ { m }$ and the noise is independent across frames.

## B.2 LINEAR INTERPOLATION UNDER CORRELATED NOISE

The linear interpolation between y and $\epsilon _ { t }$ incorporates the time dependence of the noise transform. Holding y and $\boldsymbol { \xi }$ fixed, the interpolation path and its derivative are

$$
\mathbf { z } _ { t } = t \mathbf { y } + ( 1 - t ) \mathbf { \epsilon } _ { t } = t \mathbf { y } + ( 1 - t ) \mathcal { C } _ { t } \pmb { \xi } ,\tag{16}
$$

$$
\dot { \mathbf { z } } _ { t } = \mathbf { y } - \mathbf { \epsilon } _ { t } + ( 1 - t ) \dot { \mathcal { C } } _ { t } \pmb { \xi } .\tag{17}
$$

Consequently, for $t < 1$ , the conditional distribution along this path is

$$
p _ { t } ( \mathbf { z } \mid \mathbf { y } , \mathbf { c } ) = \mathcal { N } \big ( \mathbf { z } ; t \mathbf { y } , ( 1 - t ) ^ { 2 } \mathbf { Q } _ { t } \big ) .\tag{18}
$$

Marginalizing over $\textbf { y } \sim \ p _ { \mathrm { d a t a } } ( \cdot { } ~ | ~ \textbf { c } )$ defines a probability path from $p _ { 0 } ~ = ~ \mathcal { N } ( \mathbf { 0 } , \mathbf { Q } _ { 0 } )$ to $p _ { 1 } =$ $p _ { \mathrm { d a t a } } ( \cdot \mid \mathbf { c } )$

To express the path derivative in terms of the correlated noise, define

$$
{ \bf B } _ { t } = \dot { \mathcal { C } } _ { t } \mathcal { C } _ { t } ^ { - 1 } , \qquad \dot { \bf \epsilon } _ { t } = { \bf B } _ { t } \boldsymbol { \epsilon } _ { t } .\tag{19}
$$

The operator $\mathbf { B } _ { t }$ describes how the noise transform changes with diffusion time for fixed innovations. Averaging Equation 17 over the conditional distribution of the clean sequence and innovations gives the marginal velocity

$$
\begin{array} { r l } & { \mathbf { v } _ { t } ( \mathbf { z } \mid \mathbf { c } ) = \mathbb { E } [ \dot { \mathbf { z } } _ { t } \mid \mathbf { z } _ { t } = \mathbf { z } , \mathbf { c } ] } \\ & { \qquad = \mathbb { E } [ \mathbf { y } - \epsilon _ { t } \mid \mathbf { z } _ { t } = \mathbf { z } , \mathbf { c } ] + ( 1 - t ) \mathbf { B } _ { t } \mathbb { E } [ \epsilon _ { t } \mid \mathbf { z } _ { t } = \mathbf { z } , \mathbf { c } ] . } \end{array}\tag{20}
$$

This velocity satisfies the continuity equation $\begin{array} { r } { \partial _ { t } p _ { t } + \nabla _ { \mathbf { z } } \cdot \left( p _ { t } \mathbf { v } _ { t } \right) = 0 } \end{array}$ . The associated deterministic generative dynamics are therefore

$$
\frac { \mathrm { d } \mathbf { z } _ { t } } { \mathrm { d } t } = \mathbf { v } _ { t } ( \mathbf { z } _ { t } \mid \mathbf { c } ) , \qquad \mathbf { z } _ { 0 } = \mathcal { C } _ { 0 } \boldsymbol { \xi } , \quad \boldsymbol { \xi } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { m } ) .\tag{21}
$$

## B.3 TRAINING TARGET AND DETERMINISTIC SAMPLING

Algorithm 1 regresses $\mathbf { y } - \epsilon _ { t }$ , providing a simple parameterization of the marginal velocity in Equation 20. Given one network prediction $\mathbf { f } _ { \theta } ~ = ~ \mathbf { f } _ { \theta } ( \mathbf { z } _ { t } , t ~ \mid ~ \mathbf { c } )$ , the clean latent, correlated noise, and innovations are estimated as

$$
\begin{array} { r l } & { \widehat { \mathbf { y } } _ { \theta } = \mathbf { z } _ { t } + ( 1 - t ) \mathbf { f } _ { \theta } , } \\ & { \widehat { \mathbf { \epsilon } } _ { \theta } = \mathbf { z } _ { t } - t \mathbf { f } _ { \theta } , } \\ & { \widehat { \xi } _ { \theta } = \mathcal { C } _ { t } ^ { - 1 } \widehat { \epsilon } _ { \theta } . } \end{array}\tag{22}
$$

These identities follow directly from the interpolation path: $\mathbf { y } = \mathbf { z } _ { t } + ( 1 - t ) ( \mathbf { y } - \epsilon _ { t } )$ and $\epsilon _ { t } \ =$ ${ \mathbf z } _ { t } - t ( { \mathbf y } - { \boldsymbol \epsilon _ { t } } )$ . Substituting these estimates into Equation 20 gives the velocity estimate

$$
{ \widehat { \mathbf { v } } } _ { \theta } ( \mathbf { z } _ { t } , t \mid \mathbf { c } ) = \mathbf { f } _ { \theta } + ( 1 - t ) \mathbf { B } _ { t } { \widehat { \mathbf { \epsilon } } } _ { \theta } = \mathbf { f } _ { \theta } + ( 1 - t ) { \dot { \mathcal { C } } } _ { t } { \widehat { \boldsymbol { \xi } } } _ { \theta } .\tag{23}
$$

For deterministic sampling, we integrate this velocity by holding the clean estimate $\widehat { \mathbf { y } } _ { \theta }$ fixed within each step from t to $t ^ { \prime } > t$ . Using Equation 22, the velocity can be written as

$$
\widehat { \mathbf { v } } _ { \theta } ( \mathbf { z } , t \mid \mathbf { c } ) = \frac { \widehat { \mathbf { y } } _ { \theta } - \mathbf { z } } { 1 - t } + \mathbf { B } _ { t } \left( \mathbf { z } - t \widehat { \mathbf { y } } _ { \theta } \right) .\tag{24}
$$

For $\tau \in [ t , t ^ { \prime } )$ , define the residual ${ \bf w } _ { \tau } = { \bf z } _ { \tau } - \tau \widehat { \bf y } _ { \theta }$ . With the clean estimate fixed, its dynamics satisfy

$$
\frac { \mathrm { d } \mathbf { w } _ { \tau } } { \mathrm { d } \tau } = \left( \mathbf { B } _ { \tau } - \frac { \mathbf { I } _ { m } } { 1 - \tau } \right) \mathbf { w } _ { \tau } .\tag{25}
$$

Since

$$
\frac { \mathrm { d } } { \mathrm { d } \tau } \left[ ( 1 - \tau ) \mathcal { C } _ { \tau } \right] = \left( \mathbf { B } _ { \tau } - \frac { \mathbf { I } _ { m } } { 1 - \tau } \right) ( 1 - \tau ) \mathcal { C } _ { \tau } ,\tag{26}
$$

the residual has the closed-form update

$$
\mathbf { w } _ { t ^ { \prime } } = \frac { 1 - t ^ { \prime } } { 1 - t } \mathcal { C } _ { t ^ { \prime } } \mathcal { C } _ { t } ^ { - 1 } \mathbf { w } _ { t } .\tag{27}
$$

## Algorithm 4 DyPro SDE Sampling (w/o CFG)

# One active SDE transition: t -> t\_next < 1   
# kappa: stochasticity; 0 < kappa<sub>\*\*</sub>2 <sub>\*</sub> (t\_next - t) <= 1   
v = net(z, t, c)   
x = z + (1 - t) v   
e = z - t v   
e\_init = decorrelate(e, t)   
a = kappa <sub>\*</sub> sqrt(t\_next - t)   
e\_mean = sqrt(1 - a<sub>\*\*</sub>2) <sub>\*</sub> e\_init   
e = correlate(e\_mean + a randn\_like(e\_init), t\_next)   
z\_next = t\_next x + (1 - t\_next) e   
mu = t\_next <sub>\*</sub> x + (1 - t\_next) <sub>\*</sub> correlate(e\_mean, t\_next)   
residual = decorrelate(stopgrad(z\_next) - mu, t\_next)   
scale = (1 - t\_next) a   
log\_score = avg\_latent\_dims(normal\_log\_prob(residual, 0, scale))   
return z\_next, log\_score

Substituting $\mathbf { w } _ { t } = ( 1 - t ) \widehat { \epsilon } _ { \theta }$ yields the deterministic sampling rule

$$
\begin{array} { c } { { { \bf z } _ { t ^ { \prime } } = t ^ { \prime } \widehat { \bf y } _ { \theta } + ( 1 - t ^ { \prime } ) \mathcal { C } _ { t ^ { \prime } } \mathcal { C } _ { t } ^ { - 1 } \widehat { \bf \epsilon } _ { \theta } } } \\ { { = t ^ { \prime } \widehat { \bf y } _ { \theta } + ( 1 - t ^ { \prime } ) \mathcal { C } _ { t ^ { \prime } } \widehat { \bf \xi } _ { \theta } . } } \end{array}\tag{28}
$$

The update is exact for the within-step dynamics with a fixed clean estimate and is first-order consistent with the learned ODE. It extends continuously to $t ^ { \prime } = 1$ , where it returns $\widehat { \mathbf { y } } _ { \theta }$

Algorithm 2 implements this update by applying decorrelate at t and correlate at $t ^ { \prime } .$ . Each step uses one evaluation of $\mathbf { f } _ { \theta }$ and transports the estimated innovations to the next timestep. For $\alpha = 0 , \mathcal { C } _ { t } = \mathbf { I } _ { m }$ throughout, and the update reduces to ${ \bf z } _ { t ^ { \prime } } = { \bf z } _ { t } + ( t ^ { \prime } - t ) { \bf f } _ { \theta }$

## B.4 STOCHASTIC SAMPLING

A stochastic extension of the deterministic dynamics refreshes the noise in independent-innovation coordinates. Let $\kappa _ { t } \geq 0$ be a prescribed stochasticity schedule independent of the policy parameters. We replace the fixed innovations with the Ornstein–Uhlenbeck process

$$
\mathrm { d } \pmb { \xi } _ { t } = - \frac { \kappa _ { t } ^ { 2 } } { 2 } \pmb { \xi } _ { t } \mathrm { d } t + \kappa _ { t } \mathrm { d } \mathbf { W } _ { t } , \qquad \pmb { \xi } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { m } ) ,\tag{29}
$$

where $\mathbf { W } _ { t }$ is an m-dimensional standard Brownian motion, independent of the clean sequence and history. This process preserves the standard Gaussian distribution of the innovations at every timestep. Consequently, ${ \bf z } _ { t } = t { \bf y } + ( 1 - t ) \mathcal { C } _ { t } \pmb { \xi } _ { t }$ retains the conditional distributions in Equation 18.

Applying the product rule, taking conditional expectations given $\left( \mathbf { z } _ { t } , \mathbf { c } \right)$ , and parameterizing them with the network predictions gives

$$
\mathrm { d } \mathbf { z } _ { t } = \left[ \mathbf { f } _ { \theta } + ( 1 - t ) \left( \mathbf { B } _ { t } - \frac { \kappa _ { t } ^ { 2 } } { 2 } \mathbf { I } _ { m } \right) \hat { \mathbf { \epsilon } } _ { \theta } \right] \mathrm { d } t + \kappa _ { t } ( 1 - t ) \mathcal { C } _ { t } \mathrm { d } \mathbf { W } _ { t } .\tag{30}
$$

Here, the network prediction and noise estimate are evaluated at $\left( \mathbf { z } _ { t } , t , \mathbf { c } \right)$ . Setting $\kappa _ { t } = 0$ recovers the deterministic dynamics.

For a sampling step from t to $t ^ { \prime } > t$ with $t ^ { \prime } < 1$ , we hold the clean estimate ${ \widehat { \mathbf { y } } } _ { \theta }$ fixed and discretize the innovation process in Equation 29 from $\widehat { \xi _ { \theta } }$ using $h = t ^ { \prime } - t$ and

$$
a = \kappa _ { t } \sqrt { h } , \qquad b = \sqrt { 1 - a ^ { 2 } } , \qquad 0 \leq a \leq 1 .\tag{31}
$$

Here, $a ^ { 2 } + b ^ { 2 } = 1$ preserves the standard Gaussian reference distribution in innovation coordinates. For an independent $\pmb { \eta } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { m } )$ , the resulting update is

$$
\mathbf { z } _ { t ^ { \prime } } = t ^ { \prime } \widehat { \mathbf { y } } _ { \theta } + ( 1 - t ^ { \prime } ) \mathcal { C } _ { t ^ { \prime } } \left( b \mathcal { C } _ { t } ^ { - 1 } \widehat { \mathbf { \epsilon } } _ { \theta } + a \pmb { \eta } \right) .\tag{32}
$$

All network estimates are evaluated at the current state $\left( \mathbf { z } _ { t } , t , \mathbf { c } \right)$ . Algorithm 4 implements this update by decorrelating the predicted noise, mixing the resulting innovations with independent Gaussian noise, and applying the noise transform at $t ^ { \prime } .$ For $a = 0$ , the update reduces to Equation 28.

## C TIMESTEP-AWARE POST-TRAINING

Post-training combines a terminal forecast reward with policy updates on selected stochastic transitions. We describe mixed rollouts, timestep-aware reward evaluation, and the local policy objective, together with the training algorithm and configuration.

## C.1 MIXED SAMPLING AND WINDOW SCHEDULING

We discretize diffusion time as $0 = t _ { 0 } < \cdots < t _ { N } = 1$ and activate a contiguous window $\mathcal { W } _ { \ell } =$ $\{ \ell , \ldots , \ell + K - 1 \}$ , with default size $K = 4$ . The rollout policy $\theta _ { \mathrm { o l d } }$ is held fixed while collecting the group. Transitions inside the window use the stochastic kernel from Appendix B.4, while the remaining transitions use the deterministic update from Appendix B.3:

$$
\mathbf z _ { t _ { n + 1 } } = \left\{ \begin{array} { l l } { \mathrm { S a m p l e } \left[ p _ { \theta _ { \mathrm { o l d } } } ^ { \mathrm { D y P r o } } ( \cdot \mid \mathbf z _ { t _ { n } } , \mathbf c ) \right] , } & { n \in \mathcal W _ { \ell } , } \\ { \boldsymbol \Phi _ { \theta _ { \mathrm { o l d } } } ^ { \mathrm { D y P r o } } ( \mathbf z _ { t _ { n } } , t _ { n } , t _ { n + 1 } , \mathbf c ) , } & { n \not \in \mathcal W _ { \ell } . } \end{array} \right.\tag{33}
$$

Here, $p ^ { \mathrm { D y P r o } }$ is the transition distribution induced by Equation 32, and $\Phi ^ { \mathrm { D y P r o } }$ denotes the deterministic update in Equation 28. The trajectories share an initial DyPro latent and conditioning context, with independent stochastic innovations inside the window.

The stochastic window contains only nonterminal transitions, with $0 < \kappa _ { t _ { n } } ^ { 2 } ( t _ { n + 1 } - t _ { n } ) \leq 1$ for each active step. Its bounds are

$$
1 \le K \le N - 1 , \qquad 0 \le \ell \le \ell _ { \mathrm { m a x } } = N - K - 1 .\tag{34}
$$

The final transition to $t _ { N } = 1$ is deterministic. We start at $\ell = 0$ and cycle through window starts spaced by $s _ { w } ,$ , switching after every $M _ { \mathrm { s h i f t } }$ training iterations:

$$
\ell \gets \left\{ \begin{array} { l l } { \ell + s _ { w } , } & { \ell + s _ { w } \leq \ell _ { \mathrm { m a x } } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{35}
$$

With $N = 1 0 , K = 4 , s _ { w } = 2$ , and $M _ { \mathrm { s h i f t } } = 2 0$ throughout post-training.

## C.2 TIMESTEP-AWARE REWARD EVALUATION

Each rollout produces a final decoded forecast $\hat { \mathbf { x } } ^ { i } = D ( \mathbf { z } _ { 1 } ^ { i } )$ ), which is compared with the reference future sequence $\mathbf { x } ^ { * }$ . The two reward components are defined in Equations 7 and 8: $R _ { \mathrm { n o i s e } }$ combines pooled low-threshold CSI with SSIM, and $R _ { \mathrm { c l e a n } }$ combines high-threshold CSI with negative LPIPS. Local max pooling applies only to the low-threshold CSI terms; the other metrics are evaluated on the decoded forecast without this pooling operation. Metric definitions are given in Appendix D.2.

For an active transition $n ,$ the terminal reward assigned to trajectory i is $R _ { n } ^ { i } = R ( \hat { \mathbf { x } } ^ { i } , \mathbf { x } ^ { * } ; t _ { n } )$ , as defined in Equation 9. All active transitions are evaluated on the same final forecast. Their metric weights depend on their own diffusion times $t _ { n } .$ , so rewards and group-relative advantages are computed separately for each transition. No reward is evaluated on an intermediate noisy latent.

## C.3 MIXGRPO OBJECTIVE AND POLICY SCORES

For each active transition n, the group provides a relative measure of forecast quality. We normalize its terminal rewards across the G trajectories:

$$
\hat { A } _ { n } ^ { i } = \frac { R _ { n } ^ { i } - \operatorname * { m e a n } ( \{ R _ { n } ^ { j } \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { n } ^ { j } \} _ { j = 1 } ^ { G } ) + \varepsilon _ { A } } ,\tag{36}
$$

where $\varepsilon _ { A } > 0$ stabilizes the denominator. The comparison is made across forecasts at the same transition, rather than across diffusion times. If all group rewards agree, their advantages are zero.

For a stochastic step from t to $t ^ { \prime } < 1$ with $a > 0$ , the update in Equation 32 defines the conditional density

$$
p _ { \theta } ^ { \mathrm { D y P r o } } ( \mathbf { z } ^ { \prime } \mid \mathbf { z } , \mathbf { c } ) = \mathcal { N } \big ( \mathbf { z } ^ { \prime } ; \pmb { \mu } _ { \theta } , \sigma ^ { 2 } \mathbf { Q } _ { t ^ { \prime } } \big ) ,\tag{37}
$$

$$
\pmb { \mu } _ { \pmb { \theta } } = t ^ { \prime } \widehat { \mathbf { y } } _ { \pmb { \theta } } + ( 1 - t ^ { \prime } ) b \mathcal { C } _ { t ^ { \prime } } \mathcal { C } _ { t } ^ { - 1 } \widehat { \mathbf { \epsilon } } _ { \pmb { \theta } } , \qquad \sigma = ( 1 - t ^ { \prime } ) a .\tag{38}
$$

All network estimates are evaluated at $\left( { \bf z } , t , { \bf c } \right)$ ; the transition times are implicit in $p _ { \theta } ^ { \mathrm { { D y P r o } } }$ . Whitening the residual as $\mathbf { w } _ { \theta } = \mathcal { C } _ { t ^ { \prime } } ^ { - 1 } ( \mathbf { z } ^ { \prime } - \pmb { \mu } _ { \theta } )$ gives the joint log density

$$
\log p _ { \theta } ^ { \mathrm { D y P r o } } ( \mathbf { z } ^ { \prime } \mid \mathbf { z } , \mathbf { c } ) = - \frac { m } { 2 } \log ( 2 \pi ) - m \log \sigma - \log \mid \operatorname* { d e t } \mathcal { C } _ { t ^ { \prime } } \mid - \frac { \mid \mid \mathbf { w } _ { \theta } \mid \mid _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } .\tag{39}
$$

For trajectory i at transition $n ,$ we write log $p _ { \theta , n } ^ { i } \ = \ \log p _ { \theta } ^ { \mathrm { D y P r o } } ( \mathbf { z } _ { t _ { n + 1 } } ^ { i } \ \mid \ \mathbf { z } _ { t _ { n } } ^ { i } , \mathbf { c } )$ , with log $p _ { \mathrm { o l d } , n } ^ { i }$ defined analogously for $\theta _ { \mathrm { o l d } }$

For a stored state and successor from the rollout, the policy ratio is

$$
\rho _ { n } ^ { i } ( \theta ) = \frac { p _ { \theta } ^ { \mathrm { D y P r o } } ( \mathbf { z } _ { t _ { n + 1 } } ^ { i } \mid \mathbf { z } _ { t _ { n } } ^ { i } , \mathbf { c } ) } { p _ { \theta _ { \mathrm { o l d } } } ^ { \mathrm { D y P r o } } ( \mathbf { z } _ { t _ { n + 1 } } ^ { i } \mid \mathbf { z } _ { t _ { n } } ^ { i } , \mathbf { c } ) } = \exp \big ( \log p _ { \theta , n } ^ { i } - \log p _ { \mathrm { o l d } , n } ^ { i } \big ) ,\tag{40}
$$

Both policies are evaluated on the same stored transition. The states, terminal rewards, advantages, and old-policy scores are held fixed during the update; gradients flow through the current-policy log density.

When both policies share the time grid, α, and stochasticity schedule, their covariance and whitening Jacobian are identical and independent of θ. Let $\mathbf { w } _ { \theta , n } ^ { i }$ and $\sigma _ { n }$ denote the residual and scale in Equation 39 for trajectory i at transition $n .$ Cancelling the shared terms in the two log densities yields

$$
\log \rho _ { n } ^ { i } ( \boldsymbol { \theta } ) = \frac { \| \mathbf { w } _ { \mathrm { o l d } , n } ^ { i } \| _ { 2 } ^ { 2 } - \| \mathbf { w } _ { \boldsymbol { \theta } , n } ^ { i } \| _ { 2 } ^ { 2 } } { 2 \sigma _ { n } ^ { 2 } } .\tag{41}
$$

The Jacobian can thus be omitted from a score used only in this difference, although it remains part of the absolute joint log density.

Following MixGRPO (Li et al., 2025) and Flow-GRPO (Liu et al., 2025), we maximize the clipped local surrogate

$$
\mathcal { I } _ { \mathrm { M i x G R P O } } ( \theta ) = \mathbb { E } \left[ \frac { 1 } { G | \mathcal { W } _ { \ell } | } \sum _ { i = 1 } ^ { G } \sum _ { n \in \mathcal { W } _ { \ell } } \operatorname* { m i n } \left( \rho _ { n } ^ { i } ( \theta ) \hat { A } _ { n } ^ { i } , \mathrm { c l i p } ( \rho _ { n } ^ { i } ( \theta ) , 1 - \varepsilon _ { c } , 1 + \varepsilon _ { c } ) \hat { A } _ { n } ^ { i } \right) \right] ,\tag{42}
$$

where the expectation is over conditioning examples and old-policy rollouts, $| { \mathcal { W } } _ { \ell } | = K$ , and $\varepsilon _ { c }$ is the clipping range. The implemented minimization loss is the negative of this objective. Only the stochastic transitions participate in the surrogate; deterministic transitions still influence the sampled final forecast.

## C.4 TRAINING ALGORITHM

Algorithm 5 presents the full training loop. The helper clipped mixgrpo optimizes the negative of Equation 42, holding the rollout data fixed. Old-policy scores are cached internally, following the convention in Appendix C.3.

The stochastic helper dypro sde step implements Equation 32 and obtains the next time from the grid. The deterministic helper ode step denotes $\dot { \Phi } ^ { \mathrm { D y P r o } }$ , and max left is the bound in Equation 34.

The helper shared noise(G) repeats one DyPro initial latent across the group. Both reward helpers include the specified metric coefficients, and group normalize includes $\varepsilon _ { A }$ . Contexts and targets are broadcast as needed. Latent denormalization is included in decode.

Algorithm 5 Stage-Aware Post-Training with Window Scheduling

```python
<sup>#</sup> <sub>#</sub> theta: trainable policy initialized from the flow-matching checkpoint
left = 0
for iteration, (c, x_gt) in enumerate(dataloader):
window = range(left, left + K)
traj[:, 0] = shared_noise(G)
for n, t in enumerate(timesteps[:-1]):
if n in window:
traj[:, n + 1] = dypro_sde_step(theta, traj[:, n], t, c)
else:
traj[:, n + 1] = ode_step(theta, traj[:, n], t, c)
x_sample = decode(traj[:, -1])
r_noise = R_noise(x_sample, x_gt)
r_clean = R_clean(x_sample, x_gt)
for n in window:
t = timesteps[n]
rewards[:, n] = (1 - t) <sub>*</sub> r_noise + t <sub>*</sub> r_clean
advantages = group_normalize(rewards[:, window], dim=0)
loss = clipped_mixgrpo(theta, traj, window, advantages)
update(theta, loss)
if (iteration + 1) % shift_interval == 0:
left += window_stride
if left > max_left:
left = 0
```

## C.5 DOMAIN GUIDANCE

After skill-aware RL post-training, we adopt Domain Guidance (DoG) (Zhong et al., 2025), which combines an adapted conditional model with the original pretrained model as the unconditional reference. In our setting, the conditional branch uses the RL-adapted parameters $\theta _ { \mathrm { R L } }$ with history c, while the unconditional branch uses the frozen pre-RL parameters $\theta _ { \mathrm { p r e } }$ with zero conditioning:

$$
\mathbf { v } ^ { \mathrm { { D o G } } } ( \mathbf { z } _ { t } , t \mid \mathbf { c } ) = w \mathbf { v } _ { \theta _ { \mathrm { R L } } } ( \mathbf { z } _ { t } , t \mid \mathbf { c } ) + ( 1 - w ) \mathbf { v } _ { \theta _ { \mathrm { p r e } } } ( \mathbf { z } _ { t } , t \mid \boldsymbol { \emptyset } ) .\tag{43}
$$

This retains the pretrained unconditional reference while incorporating skill alignment through the conditional branch. Setting w = 1 recovers the RL model without guidance; using $\theta _ { \mathrm { p r e } }$ for both branches recovers Equation 11. Post-RL evaluation settings use $w = 1 . 4 .$ , as listed in Tables 4. Guidance is disabled during RL rollouts and applied only at inference. RL+CFG configuration in Appendix E.3 uses this DoG formulation.

## C.6 POST-TRAINING AND REWARD CONFIGURATION

Table 4 summarizes the post-training, reward, and evaluation configurations. The pretrained VAE remains frozen throughout post-training. Each rollout batch is used for one gradient update, followed by synchronization of the rollout policy.

The CSI coefficients incorporate uniform averaging within each threshold subset and the mixing weights for SSIM or negative LPIPS. No additional normalization is applied to the weighted sums in Equations 7 and 8. The listed checkpoints identify the models used for evaluation, rather than the total training budgets.

## D DATASETS AND EVALUATION PROTOCOL

## D.1 DATASETS AND PREPROCESSING

Table 5 summarizes the dataset configurations and split sizes for the forecasting tasks in Section 5.1.

Table 4: Post-training, reward, and evaluation configurations. Reward thresholds are expressed in VIL values for SEVIR and mm $\mathrm { h } ^ { - 1 }$ for MRMS.
<table><tr><td>Setting</td><td>SEVIR</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W } , ( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ </td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 5 }$  , constant</td></tr><tr><td>Batch size</td><td> $\stackrel { 8 } { 1 0 ^ { - 4 } }$ </td></tr><tr><td>Policy ratio clipping range  $\varepsilon _ { c }$ </td><td></td></tr><tr><td>Advantage clipping</td><td>[−5,5]</td></tr><tr><td colspan="2">Rollout</td></tr><tr><td>Group size G</td><td>16</td></tr><tr><td>Sampling steps N</td><td>10</td></tr><tr><td>Stochastic window size  $K$ </td><td>4</td></tr><tr><td>Window stride  $s _ { w }$ </td><td>2</td></tr><tr><td>Window shift interval  $M _ { \mathrm { s h i f t } }$ </td><td>20 training iterations (cyclic)</td></tr><tr><td>DyPro correlation strength α</td><td>0.5</td></tr><tr><td>CFG during rollout</td><td>Disabled</td></tr><tr><td colspan="2">Reward {16, 74, 133}</td></tr><tr><td>Low-threshold set  $\mathcal { T } _ { \mathrm { l o w } }$ </td><td>{1, 2, 4}</td></tr><tr><td>High-threshold set Thigh</td><td>{8, 16, 32}</td></tr><tr><td>Low-threshold CSI max pooling</td><td>4×4</td></tr><tr><td>SSIM coefficient  $\lambda _ { s }$ </td><td>0.5</td></tr><tr><td>LPIPS coefficient  $\lambda _ { p }$ </td><td>0.9</td></tr><tr><td>CSI coefficient  $\lambda _ { \tau } , \tau \in \mathcal { T } _ { \mathrm { l o w } }$ </td><td> $( 1 - \lambda _ { s } ) / | \mathcal { T } _ { \mathrm { l o w } } |$ </td></tr><tr><td>CSI coefficient  $\lambda _ { \tau } , \tau \in \mathcal { T } _ { \mathrm { h i g h } }$ </td><td> $( 1 - \lambda _ { p } ) / | \mathcal { T } _ { \mathrm { h i g h } } |$ </td></tr><tr><td colspan="2">Evaluation</td></tr><tr><td># steps</td><td>4,000</td></tr><tr><td>Sampling</td><td>10 Euler steps</td></tr><tr><td>CFG weight</td><td>1.4</td></tr></table>

Table 5: Dataset configurations. Spatial resolution refers to the source radar grids, and frame size denotes the model input and output size. Forecast frames are shown as observed → predicted; split sizes count sequences.
<table><tr><td>Setting</td><td>SEVIR</td><td>MRMS</td></tr><tr><td>Data variable</td><td>VIL</td><td>Precipitation rate</td></tr><tr><td>Spatial / temporal res.</td><td>1 km / 5 min</td><td> $0 . 0 1 ^ { \circ } / 1 0 \mathrm { m i n }$ </td></tr><tr><td>Forecast frames</td><td> $5  2 0$ </td><td>4 → 20</td></tr><tr><td>Frame size</td><td> $1 2 8 \times 1 2 8$ </td><td> $2 5 6 \times 2 5 6$ </td></tr><tr><td>Train / validation / test</td><td> $5 9 , 5 3 0 / 2 8 , 1 4 5 / 7 , 2 2 0$ </td><td> $6 , 8 0 7 , 5 2 8 / 1 2 , 0 0 0 / 1 2 , 0 0 0$ </td></tr></table>

SEVIR (Veillette et al., 2020) contains spatiotemporal observations of weather events across the United States, each covering 384 km × 384 km. Following prior studies (Gao et al., 2022b; 2023), we use the VIL modality, which has a native spatial resolution of 1 km and a temporal resolution of 5 minutes. Following DiffCast and AlphaPre (Yu et al., 2024; Lin et al., 2025), we downsample the frames to 128 × 128 and predict 20 future frames from 5 observed frames. Training includes events up to January 1, 2019. Validation uses later events up to October 1, 2019, and testing uses events after that date. Both cutoffs are inclusive for the earlier split and refer to 00:00 UTC. We exclude events with missing VIL data or duplicate VIL records. From each 49-frame event, we extract 25- frame sequences every five frames. Training augmentation uses rotations by multiples of $9 0 °$ and horizontal or vertical flips, applied consistently across all frames in a sequence. Since SEVIR stores VIL as dimensionless encoded values, we report its color bars and evaluation thresholds without physical units (Veillette et al., 2020).

MRMS (Zhang et al., 2016) is a composite radar dataset collected over the contiguous United States, spanning $2 0 ^ { \circ } \bar { \bf N } \mathrm { - } 5 5 ^ { \circ } \mathrm { N }$ in latitude and $1 3 0 ^ { \circ } \mathrm { W } { - } 6 0 ^ { \circ } \mathrm { W }$ in longitude at a spatial resolution of $0 . 0 1 ^ { \circ }$ per grid. Following NowcastNet (Zhang et al., 2023), we use data from 2016–2020 for model development and 2021 for testing. Within 2016–2020, we reserve the first day of each month for validation and use the remaining days for training. After excluding known anomalous timestamps, we take the last 24 frames of each continuous 40-frame sequence sampled at 10-minute intervals. For training, we sample sequences with replacement and select $2 5 6 \times 2 5 6$ crops from $5 1 2 \times 5 1 2$ patches, using precomputed sequence and spatial weights. Validation and testing use fixed sequence selections and $2 5 6 \times 2 5 6$ center crops.

Radar and latent normalization. For SEVIR, we normalize VIL values by dividing by 255, following DiffCast (Yu et al., 2024). For MRMS, we cap rain rates at 128 mm $\mathrm { h } ^ { - 1 }$ following NowcastNet (Zhang et al., 2023), then apply logarithmic normalization $\mathrm { t o ~ } [ - 1 , 1 ]$ . For both datasets, we standardize VAE latents using dataset-specific means and standard deviations, building on latent rescaling in latent diffusion models (Rombach et al., 2022).

## D.2 EVALUATION METRICS AND AGGREGATION

For meteorological skill, we use the Critical Success Index (CSI) and Heidke Skill Score (HSS) (Veillette et al., 2020; Yu et al., 2024; Zhang et al., 2023):

$$
\mathrm { C S I } _ { \tau } = \frac { \mathrm { T P } _ { \tau } } { \mathrm { T P } _ { \tau } + \mathrm { F P } _ { \tau } + \mathrm { F N } _ { \tau } } ,\tag{44}
$$

and

$$
\mathrm { H S S } _ { \tau } = \frac { 2 ( \mathrm { T P } _ { \tau } \cdot \mathrm { T N } _ { \tau } - \mathrm { F P } _ { \tau } \cdot \mathrm { F N } _ { \tau } ) } { ( \mathrm { T P } _ { \tau } + \mathrm { F N } _ { \tau } ) ( \mathrm { F N } _ { \tau } + \mathrm { T N } _ { \tau } ) + ( \mathrm { T P } _ { \tau } + \mathrm { F P } _ { \tau } ) ( \mathrm { F P } _ { \tau } + \mathrm { T N } _ { \tau } ) } .\tag{45}
$$

Here, $\mathrm { T P } _ { \tau } , \mathrm { F P } _ { \tau } , \mathrm { F N } _ { \tau } ,$ , and $\mathrm { T N } _ { \tau }$ denote pixelwise true-positive, false-positive, false-negative, and true-negative counts at threshold τ , aggregated over the evaluation set at each lead time. We average CSI and HSS uniformly over lead times and evaluation thresholds.

For perceptual quality, we use LPIPS (Zhang et al., 2018) and SSIM (Wang et al., 2004):

$$
\mathrm { L P I P S } ( \hat { \mathbf { x } } , \mathbf { x } ^ { * } ) = \sum _ { \ell } \frac { 1 } { H _ { \ell } W _ { \ell } } \sum _ { h , w , c } a _ { \ell c } \big ( \phi _ { \ell } ( \hat { \mathbf { x } } ) _ { h w c } - \phi _ { \ell } ( \mathbf { x } ^ { * } ) _ { h w c } \big ) ^ { 2 } ,\tag{46}
$$

where $\phi _ { \ell }$ denotes channel-normalized features at layer $\ell , \ a _ { \ell c }$ are learned channel weights, and $H _ { \ell } , W _ { \ell }$ are the feature-map dimensions. SSIM is defined as

$$
\mathrm { S S I M } ( \hat { \mathbf { x } } , \mathbf { x } ^ { * } ) = \frac { ( 2 \mu _ { \hat { x } } \mu _ { x ^ { * } } + c _ { 1 } ) ( 2 \sigma _ { \hat { x } x ^ { * } } + c _ { 2 } ) } { ( \mu _ { \hat { x } } ^ { 2 } + \mu _ { x ^ { * } } ^ { 2 } + c _ { 1 } ) ( \sigma _ { \hat { x } } ^ { 2 } + \sigma _ { x ^ { * } } ^ { 2 } + c _ { 2 } ) } ,\tag{47}
$$

where $\mu , \sigma ^ { 2 }$ , and $\sigma _ { \hat { x } x }$ ∗ denote local means, variances, and covariance, and $c _ { 1 } , c _ { 2 }$ are stabilization constants. Perceptual scores are spatially averaged within each frame and then uniformly averaged over forecast frames and test sequences. Higher CSI, HSS, and SSIM and lower LPIPS indicate better performance.

## E ADDITIONAL EXPERIMENTAL RESULTS

## E.1 FORECAST SKILL ACROSS LEAD TIMES

Figure 6 compares CSI and HSS across forecast lead times on MRMS and SEVIR, complementing the aggregate results in Table 1. The curves cover 10–200 minutes on MRMS and 5–100 minutes on SEVIR and include NowcastDiT both with and without RL post-training.

## E.2 CUMULATIVE COMPONENT ABLATION

Table 6 reports a cumulative ablation on SEVIR, starting from the standard DiT baseline and successively adding 3D RoPE and QK-Norm, DyPro, and timestep-aware RL. Each row retains all components introduced in the preceding rows.

Table 6: Cumulative component ablation on SEVIR. ↑ indicates higher is better and ↓ indicates lower is better. Bold and underlined values denote the best and second-best results, respectively.
<table><tr><td>Method</td><td>CSI↑</td><td>CSI-181↑</td><td>CSI-219↑</td><td>HSS↑</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td>DiT</td><td>0.2926</td><td>0.0948</td><td>0.0555</td><td>0.3733</td><td>0.1545</td><td>0.7048</td></tr><tr><td>+ 3D RoPE, QK-Norm</td><td>0.3012</td><td>0.1082</td><td>0.0676</td><td>0.3647</td><td>0.1317</td><td>0.6812</td></tr><tr><td>+ DyPro</td><td>0.3195</td><td>0.1204</td><td>0.0752</td><td>0.4092</td><td>0.1486</td><td>0.7099</td></tr><tr><td>+RL</td><td>0.3240</td><td>0.1241</td><td>0.0750</td><td>0.4148</td><td>0.1469</td><td>0.7197</td></tr></table>

## E.3 CFG AND SKILL-AWARE POST-TRAINING

We adopt Domain Guidance (DoG) (Zhong et al., 2025) to integrate CFG with RL post-training, as detailed in Appendix C.5. Table 7 reports the individual and combined effects of CFG and RL. The combined configuration has the highest aggregate CSI, HSS, and SSIM among these four settings, while CFG alone has the lowest LPIPS, illustrating that the metrics need not improve together.

Table 7: Effects of RL and classifier-free guidance on SEVIR. ↑ indicates higher is better and ↓ indicates lower is better. Bold and underlined values denote the best and second-best results, respectively.
<table><tr><td>RL</td><td>Guidance</td><td>CSI↑</td><td>CSI-181↑</td><td>CSI-219↑</td><td>HSS↑</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td>x</td><td>x</td><td>0.2934</td><td>0.0956</td><td>0.0560</td><td>0.3749</td><td>0.1541</td><td>0.7041</td></tr><tr><td>x</td><td>V</td><td>0.3195</td><td>0.1204</td><td>0.0752</td><td>0.4092</td><td>0.1486</td><td>0.7099</td></tr><tr><td>V</td><td>x</td><td>0.3109</td><td>0.1050</td><td>0.0588</td><td>0.3961</td><td>0.1580</td><td>0.7156</td></tr><tr><td>V</td><td>V</td><td>0.3240</td><td>0.1241</td><td>0.0750</td><td>0.4148</td><td>0.1469</td><td>0.7197</td></tr></table>

## E.4 DYPRO CORRELATION STRENGTH

Table 8 reports the SEVIR sensitivity to α across meteorological and perceptual metrics. Among the listed settings, α = 0.5 gives the highest aggregate CSI. This experiment varies the noise correlation strength. All experiments in this ablation are conducted without RL post-training to isolate the effect of DyPro.

Table 8: DyPro correlation strength α on SEVIR. α = 0 corresponds to the standard i.i.d. Gaussian noise prior. ↑ indicates higher is better and ↓ indicates lower is better. Bold and underlined values denote the best and second-best results, respectively.
<table><tr><td>α</td><td>CSI↑</td><td>CSI-181↑</td><td>CSI-219↑</td><td>HSS↑</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td>0.00</td><td>0.3012</td><td>0.1082</td><td>0.0676</td><td>0.3647</td><td>0.1317</td><td>0.6812</td></tr><tr><td>0.25</td><td>0.3188</td><td>0.1184</td><td>0.0759</td><td>0.4075</td><td>0.1485</td><td>0.7090</td></tr><tr><td>0.33</td><td>0.3191</td><td>0.1192</td><td>0.0742</td><td>0.4080</td><td>0.1483</td><td>0.7097</td></tr><tr><td>0.50</td><td>0.3195</td><td>0.1204</td><td>0.0752</td><td>0.4092</td><td>0.1486</td><td>0.7099</td></tr><tr><td>0.66</td><td>0.3185</td><td>0.1194</td><td>0.0740</td><td>0.4077</td><td>0.1495</td><td>0.7098</td></tr><tr><td>0.75</td><td>0.3184</td><td>0.1184</td><td>0.0750</td><td>0.4084</td><td>0.1494</td><td>0.7092</td></tr></table>

## E.5 TIMESTEP-AWARE REWARD VARIANTS

Table 9 defines the six variants in Figures 5(d) and 7. Let $C _ { L }$ and $C _ { H }$ denote mean low-threshold pooled CSI and mean high-threshold CSI, using the threshold sets and pooling in Table 4, and let $S = { \mathrm { S S I M } }$ and $P = \mathrm { L P I P S }$ . The mixed components are $M _ { s } = 0 . \bar { 5 } C _ { L } + 0 . 5 S$ and $M _ { d } =$ $0 . 1 C _ { H } - 0 . 9 P$

For the timestep-aware perceptual reward, $R _ { \mathrm { n o i s e } } = 0 . 1 S$ and $R _ { \mathrm { c l e a n } } = - 0 . 9 P$ , so the reward shifts from SSIM at the noise endpoint to negative LPIPS at the clean endpoint. The static counterpart averages these two endpoint rewards, following the same convention as the CSI and mixed variants. This reward changes the relative weighting of structural agreement and perceptual similarity across denoising transitions. All six variants share the cyclic window schedule in Appendix C.1; the contribution of window scheduling is not separately ablated.

Table 9: Terminal rewards for the six ablation settings.
<table><tr><td>Reward</td><td>Static</td><td>Timestep-aware</td></tr><tr><td>Mixed</td><td> $( M _ { s } + M _ { d } ) / 2$ </td><td> $( 1 - t ) M _ { s } + t M _ { d }$ </td></tr><tr><td>CSI</td><td> $( C _ { L } + C _ { H } ) / 2$ </td><td> $( 1 - t ) C _ { L } + t C _ { H }$ </td></tr><tr><td>Perceptual</td><td> $( 0 . 1 S - 0 . 9 P ) / 2$ </td><td> $0 . 1 ( 1 - t ) S - 0 . 9 t P$ </td></tr></table>

## E.6 ADDITIONAL VISUALIZATIONS

We provide four additional forecast visualizations to complement the qualitative comparisons in the main text. Figures 8 and 9 show SEVIR cases 802 and 6155, comparing NowcastDiT with CasCast and the standard DiT baseline. Figures 10 and 11 show MRMS cases 239 and 1149, comparing NowcastDiT with NowcastNet and the standard DiT baseline. Each figure includes the input observations, ground truth, and forecasts at matched lead times, allowing comparison of precipitation structure and intensity evolution. These selected cases supplement the aggregate evaluation in Table 1.

![](images/e1b861f32ff0d821fc2db06b27ddbf8225faf6f782bb6441d489324ea9f4b73d.jpg)

Figure 6: Forecast skill across lead times on MRMS and SEVIR. The horizontal axes show forecast lead time in minutes; higher CSI and HSS indicate better performance.  
![](images/cbffcd2cea3b362a8b1ef956a06cccb14199e6da30c40b262b6032a8e840fc0f.jpg)

![](images/6d5c10a06e9f9b310281a89c1abfe5373ed4b0df2893640bda383b0d8c7f8b3b.jpg)  
Figure 7: Additional reward ablations on SEVIR. LPIPS (left) and RMSE (right) during RL training for the rewards in Table 9. The legend follows Figure 5(d); the timestep-aware perceptual reward uses $0 . 1 ( 1 - t ) \mathrm { S S I M } - 0 . 9 t \mathrm { L P I P S }$ . Lower values indicate better performance for both metrics.

![](images/841eccd4862cfa1ec9dfe265374a217e29bdbe53a1f8b4647d333086c0873c72.jpg)  
Figure 8: Qualitative comparison on SEVIR (case 802). Five input observations span −20 to 0 minutes at 5-minute intervals. Ground truth and forecasts are shown every 10 minutes up to 100 minutes. Colors indicate VIL values.

![](images/79e785e0cdee7c534c73d7159317897b614e56d9cd08d90c509d4c29390b84ee.jpg)  
Figure 9: Qualitative comparison on SEVIR (case 6155). Five input observations span −20 to 0 minutes at 5-minute intervals. Ground truth and forecasts are shown every 10 minutes up to 100 minutes. Colors indicate VIL values.

![](images/4007d72cb0827affe0ef973e1d399e320f01620aa189643a982047821990679c.jpg)  
Figure 10: Qualitative comparison on MRMS (case 239). Four input observations span −30 to 0 minutes at 10-minute intervals. Ground truth and forecasts are shown every 20 minutes up to 200 minutes. Colors indicate precipitation rate in mm h<sup>−1</sup>.

![](images/8261aa890fa069fb88c0f6081311682e000bbac51a5062b311b36620c663dac3.jpg)  
Figure 11: Qualitative comparison on MRMS (case 1149). Four input observations span −30 to 0 minutes at 10-minute intervals. Ground truth and forecasts are shown every 20 minutes up to 200 minutes. Colors indicate precipitation rate in mm h<sup>−1</sup>.