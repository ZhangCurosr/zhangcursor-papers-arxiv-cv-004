# Regional Anchor Routing for Multi-Object Motion Transfer

Yuze Li<sup>1</sup> Fuchen Long<sup>2</sup> Xinlong Sun<sup>2</sup> Cheng Yan<sup>1§</sup> Xinyu Zhang<sup>3</sup> Xi Chen<sup>2</sup>

<sup>1</sup> Tianjin University, <sup>2</sup> Smart Creation Platform Department, Online Video BU, Tencent, <sup>3</sup> University of Auckland

## Abstract

Compositional motion transfer requires a model to extract motions from multiple reference videos, and apply each motion to a user-specified object in one target scene. Existing methods normally struggle to reliably assign diferent reference motions to the correct target objects, thus a motion intended for one object may afect another. Meanwhile, some motion-specific tuning approaches are still facing another issue: they must run a new optimization whenever the reference video changes, making repeated transfers ineficient. To address these limitations, we propose RART, a framework that unifies object-wise motion extraction, target-region anchoring, and frame-aligned injection into a frozen video Difusion Transformer for multi-object motion transfer. For each reference motion, RART extracts temporally ordered features from the masked reference video, and then conditions them on the target first-frame mask to form mask-anchored motion tokens. Frame-aligned motion cross-attention allows each generated frame to read only the motion tokens at the corresponding time slots, composing all object motions in one denoising process. During training, a motion-attention routing loss encourages each target region to attend to its corresponding motion tokens and reduces interference between objects. Therefore, RART can handle novel reference-target combinations without pervideo optimization. Experiments show that RART achieves accurate motion preservation, eficient inference, and strong compositional generalization.

Project Page — https://ethan-li123.github.io/RART\_page/

## Introduction

Difusion Transformers now generate high-quality videos from text and images (Ho, Jain, and Abbeel 2020; Rombach et al. 2022; Peebles and Xie 2023; Blattmann et al. 2023; Kong et al. 2024; Wan et al. 2025), yet fine-grained object-wise motion control still remains challenging. Such a control paradigm is crucial for character animation, video editing (Long et al. 2026; Wu et al. 2026), and multi-subject video creation, where objects usually need to follow diferent existing motion patterns.

Here, we study the problem of compositional motion transfer: given a target image, multiple reference video-mask pairs, and target object masks, the model transfers each reference motion to its assigned target object in the image while preserving the target image’s original appearance and background. Compared with single-object animation (Karras et al. 2023; Xu et al. 2024; Hu 2024), this setting introduces three coupled requirements. The model must isolate the desired motion from each masked reference without carrying over its appearance or source layout, assign each extracted motion to the correct target region, and ensure that each object follows its reference motion at the correct time step. Existing approaches address only part of these requirements. Explicit-control methods typically rely on poses, optical flow, depth, trajectories, or camera motion (Wang et al. 2023b; Zhang et al. 2024; Wang et al. 2024b; Yin et al. 2023; Ma et al. 2024; Chu et al. 2026; Zhang et al. 2025b) to control motion. The signals provide efective structural guidance when reliable priors are available, but obtaining and aligning them across objects often requires task-specific extraction or annotation. Implicit approaches transfer motion directly from reference videos without converting it into explicit pose or optical-flow conditions, ofering greater flexibility. Among them, training-free methods reuse difusion features or attention for zero-shot transfer (Yatim et al. 2024; Ku et al. 2024; Ling et al. 2025), but mainly handle a single motion source and lack object-wise motion reassignment. Training-based methods learn motion representations, but mostly focus on single-object transfer (Zhao et al. 2024; Jeong, Park, and Ye 2024; Ouyang et al. 2024; Zhang et al. 2025a) and require retraining for each new motion (Zhang et al. 2025a), limiting their generalization. Recent studies have explored feedforward implicit motion transfer (Ressler-Antal et al. 2025; Yan et al. 2026). DisMo focuses on single-object transfer, and SCAIL-2 does not support motion transfer from multiple independent reference videos.

To address these limitations, we propose RART, a newlyminted framework that factorizes compositional motion transfer into object-wise motion extraction, target-region anchoring, and frame-aligned injection. For each reference video, the reference mask isolates the desired motion source, and a pretrained motion encoder extracts a temporally ordered feature sequence. RART projects these features into the DiT token space and conditions them on the corresponding target mask, producing motion tokens that encode both the reference motion and its target-region anchor. Motion tokens are then grouped by latent-frame index, and concatenated across objects within each group. Tokens associated with diferent video frames remain separate. At each denoising step, the video latent tokens attend to the motion tokens belonging to corresponding frame index, not only preserving alignment between each reference motion and its target object’s motion but also allowing multiple reference motions to be injected jointly. During training, a motion-attention routing loss further discourages each target region from attending to motion tokens of other objects, while preserving attention to its own tokens. After optimization, RART can transfer motions from new reference videos without video-specific optimization. Representative results are shown in Figure 1.

![](images/b55ad82d5f9d817ecf34f4a572791017d81944b6936947b8297bf05eace56e1f.jpg)  
Figure 1: Compositional motion transfer with RART. Masked reference videos provide object-wise motions, which are assigned to colored target regions and jointly rendered in the generated video. The examples show (a) and (b) motion reassignment, (c) cross-species transfer, and (d) multi-object composition.

Our contributions are summarized as follows:

• We propose RART, a framework for compositional multiobject motion transfer on a frozen video Difusion Transformer, supporting flexible reference-target combinations without per-video optimization.

• We introduce mask-anchored motion encoding and framealigned motion cross-attention, separating motion content from its target location and preserving temporal correspondence during control injection, respectively.

• We propose motion-attention routing loss, which encourages each target region to attend to its assigned motion tokens and suppresses cross-object information leakage.

## Related Work

Video Difusion Models. Difusion models have become the dominant backbone for visual generation (Song and Ermon 2019; Ho, Jain, and Abbeel 2020; Song et al. 2021; Song, Meng, and Ermon 2021; Dhariwal and Nichol 2021; Ho and Salimans 2021; Rombach et al. 2022; Peebles and Xie 2023; Esser et al. 2024; Lipman et al. 2023). Recent video generation systems extend these priors with temporal modules, latent video difusion, or transformer-based video modeling (Ho et al. 2022b; Singer et al. 2023; Ho et al. 2022a; Villegas et al. 2023; Hong et al. 2023; He et al. 2022; Wang et al. 2023a; Chen et al. 2023; Blattmann et al. 2023;

Brooks et al. 2024; Kong et al. 2024; Long et al. 2024; Wan et al. 2025). These models provide strong priors for realistic video synthesis, but text or image conditions alone are still insuficient for fine-grained object-wise motion control, which should be emphasized in research field.

Explicit Motion Transfer. One line of controllable video generation relies on explicit motion conditions such as poses, trajectories, depth, optical flow, or camera motion. Video-Composer (Wang et al. 2023b) uses compositional spatial and temporal conditions for motion controllability. ControlVideo (Zhang et al. 2024) adapts structural controls to text-tovideo generation. MotionCtrl (Wang et al. 2024b) separates camera and object motion control. DragNUWA (Yin et al. 2023) integrates text, image, and trajectory conditions for fine-grained control. Wan-Move (Chu et al. 2026) uses latent trajectory guidance for motion-controllable generation. Poseguided animation methods drive a target subject according to a frame-wise pose sequence (Ma et al. 2024; Karras et al. 2023; Xu et al. 2024; Hu 2024). In summary, these methods are efective when reliable explicit structures are available, but they introduce extra extraction and alignment costs and often specialize to particular object categories. RART, in contrast, extracts motion directly from reference videos, and can be applied to cross-object motion transfer.

Implicit Motion Transfer. Another line of work transfers motion through implicit features, which can be divided into two categories. Training-free approaches keep model parameters fixed but exploit pretrained representations in diferent ways. Space-Time Difusion Features (Yatim et al. 2024) guides sampling with an internal feature loss. AnyV2V (Ku et al. 2024) injects source-video features into an image-to-video process, and MotionClone (Ling et al. 2025) reuses sparse temporal-attention weights. These methods avoid reference-specific training but generally process one motion source at a time. In comparison, training-based methods (Zhao et al. 2024; Ouyang et al. 2024) fit motion representations to individual inputs. For instance, MotionDirector (Zhao et al. 2024) learns spatial and temporal LoRAs.

![](images/00f5c8d267469b3417a87bead0f9ac67d9bcb2a45fdbe0e2a8ef932fa92d4d6c.jpg)  
Figure 2: Overview of RART. (a) The motion encoder extracts temporally ordered features from each masked reference video and injects the corresponding target-mask anchor to produce object-wise motion tokens. (b) Frame-aligned motion cross-attention injects the motion tokens of each latent frame into the frozen DiT.

VMC (Jeong, Park, and Ye 2024) adapts temporal layers through motion distillation. FlexiAct optimizes frequencyaware embeddings for each reference video (Zhang et al. 2025a). FlexiMMT further extends FlexiAct to multiple objects by learning object-wise motion tokens and constraining them with masks. However, it still needs to optimize motion tokens for each reference video (Li et al. 2026). Recent feed-forward approaches avoid such repeated optimization. DisMo (Ressler-Antal et al. 2025) learns appearanceagnostic motion representations for open-world transfer. SCAIL-2 (Yan et al. 2026) directly conditions generation on a driving video, and uses an in-context mask to bind driving and target characters across single- and multi-character animation. Nevertheless, DisMo does not provide target-region routing for multiple motion sources, while SCAIL-2 obtains multi-character motions jointly from a driving video rather than composing motions from multiple independent reference videos. In our work, RART extracts each motion, anchors it with its target region, and inject it into the temporally corresponding latent frames, without per-video optimization.

## Methodology

## Task Formulation

Given a target first-frame image $I ^ { \mathrm { t g t } }$ , a text condition $C ,$ and N object-wise control signals, the i-th control signal contains a reference video $V _ { i } ^ { \mathrm { r e f } }$ , its per-frame mask $M _ { i } ^ { \mathrm { r e f } }$ , and a target mask $M _ { i } ^ { \mathrm { { t g t } } }$ on $I ^ { \mathrm { t g t } }$ . RART aims to generate a target video $V ^ { \mathrm { t g t } }$ in which the region specified by $M _ { i } ^ { \mathrm { { t g t } } }$ follows the motion of $( V _ { i } ^ { \mathrm { r e f } } , M _ { i } ^ { \mathrm { r e f } } )$ while preserving the target appearance and background, which can be formulated as:

$$
V ^ { \mathrm { t g t } } = G _ { \theta } \big ( I ^ { \mathrm { t g t } } , C , \{ ( V _ { i } ^ { \mathrm { r e f } } , M _ { i } ^ { \mathrm { r e f } } , M _ { i } ^ { \mathrm { t g t } } ) \} _ { i = 1 } ^ { N } \big ) ,\tag{1}
$$

where $G _ { \theta } ( \cdot )$ denotes RART, $( V _ { i } ^ { \mathrm { r e f } } , M _ { i } ^ { \mathrm { r e f } } )$ specifies which object motion to extract, and $M _ { i } ^ { \mathrm { { t g t } } }$ specifies where it should be applied. Superscripts ref and tgt distinguish reference and target quantities. Uppercase M denotes pixel-space masks, while lowercase m denotes target masks on the latent grid. Note that the masks can be obtained from segmentation models (Cheng et al. 2022; Kirillov et al. 2023; Ravi et al. 2025; Carion et al. 2026).

## Overview of RART

As illustrated in Figure 2, RART builds upon a video Difusion Transformer (Peebles and Xie 2023; Wan et al. 2025). For i-th control signal, RART extracts its reference motion from $V _ { i } ^ { \mathrm { r e f } }$ with $M _ { i } ^ { \mathrm { { r e f } } }$ , anchors it with the target object mask $M _ { i } ^ { \mathrm { { t g t } } }$ , and finally produces corresponding motion tokens $Z _ { i } { \mathrm { : } }$

$$
\begin{array} { r } { F _ { i } = E _ { \mathrm { m o t } } ( V _ { i } ^ { \mathrm { r e f } } , M _ { i } ^ { \mathrm { r e f } } ) , } \\ { Z _ { i } = E _ { \mathrm { a n c h o r } } ( F _ { i } , M _ { i } ^ { \mathrm { t g t } } ) , } \end{array}\tag{2}
$$

where $E _ { \mathrm { m o t } } ( \cdot )$ denotes motion extraction and feature projection into the DiT feature space, and $F _ { i }$ is the projected reference-motion feature sequence. $E _ { \mathrm { a n c h o r } }$ denotes the function that encodes the reference-motion feature and the target object mask into the motion token $Z _ { i }$ . For all the attained motion tokens of N objects, we concatenate them along token dimension as $Z = \mathrm { { \dot { C } o n c a t } } ( Z _ { 1 } , \dots , Z _ { N } )$ . We denote the DiT hidden dimension by $d ,$ and all token features below use this dimension. The frozen DiT reads $Z$ through frame-aligned motion cross-attention, while a motion-attention routing loss directly supervises object-wise assignment during training.

The design separates three roles that are often entangled in implicit motion transfer: the masked reference video provides what motion should be transferred, the target object mask specifies where the motion should act, and the frozen video generator determines how the target scene should be rendered. This separation is especially important for multiobject composition.

## Mask-Anchored Motion Conditioning

Motion Tokenization. As shown in Figure 2, for the i-th control signal, the reference mask $M _ { i } ^ { \mathrm { r e f } }$ suppresses unrelated regions in $V _ { i } ^ { \mathrm { r e f } }$ before motion extraction. RART then uses the DisMo encoder (Ressler-Antal et al. 2025) as a pretrained motion feature extractor to obtain the temporally ordered sequence $F _ { i } .$ Since these features describe reference motion but do not specify where it should act, we first encode the target first-frame mask using mask encoder $E _ { \mathrm { m a s k } } ( \cdot )$ as:

$$
A _ { i } = E _ { \mathrm { m a s k } } ( M _ { i } ^ { \mathrm { t g t } } ) .\tag{3}
$$

The mask encoder consists of convolutional blocks, adaptive pooling and a linear projection to the DiT feature space. Its output $A _ { i }$ is a compact set of spatial tokens that encodes the designated target region. We then inject this spatial anchor into the temporally ordered motion features and group the resulting tokens according to the latent video frames:

$$
Z _ { i } = { \mathrm { C h u n k } } ( { \mathrm { M a s k C A } } ( ( F _ { i } + P _ { e } ) , A _ { i } ) ) .\tag{4}
$$

Here, $P _ { e }$ denotes a sinusoidal temporal encoding. MaskCA(·) denotes a residual cross-attention operation, with the projected motion features as queries, and mask tokens $A _ { i }$ as keys and values. This associates the motion features with the designated target region while preserving their temporal order. Chunk(·) groups consecutive features according to video latent temporal length, producing $Z _ { i } \in \mathbb { R } ^ { T \times K \times \tilde { d } }$ , where T is the number of latent frames and K is the number of motion tokens per object at each latent frame. The control signals are encoded independently, and then concatenated for frame-aligned motion cross-attention.

Frame-Aligned Motion Cross-Attention. Given the video latent $H \in \mathbb { R } ^ { T \times P \times d }$ in DiT, and the concatenated motion tokens $Z \in \mathbb { R } ^ { T \times N K \times d }$ of N objects, we conduct the cross-attention between H and $Z$ of each corresponding time slot for motion condition injection. Such frame-aligned motion cross-attention is formulated as:

$$
H _ { f } ^ { \prime } = \operatorname { C A } ( H _ { f } , Z _ { f } ) , f \in [ 1 , T ]\tag{5}
$$

where $\mathrm { C A } ( \cdot )$ is the cross-attention operation with a zeroinitialized output projection to preserve the pretrained generator at the beginning of training. $H _ { f }$ provides the queries, while $Z _ { f }$ provides the keys and values for the f-th time slot. Each sample–frame pair is processed independently, so the P latent tokens of a frame can access only the corresponding NK motion tokens, for the K tokens from each object. Object-wise tokens are concatenated only within the same frame, allowing multiple motions to be injected jointly without mixing motion tokens across frames; the updated features are then restored to the original layout.

## Compositional Training

The preceding modules determine how each motion enters the generator; training must further enforce correct composition across multiple active objects. For each multi-object training clip, we sample an active object subset $s$ and concatenate its motion tokens as $Z _ { S }$ . Let m<sub>i</sub> denote target mask sequence of object i on the latent grid. These sequences provide training-only supervision; no target mask sequence is needed at inference beyond $M _ { i } ^ { \mathrm { { t g t } } }$ . We also apply frameconsistent spatial augmentation to the reference video–mask pairs during training.

![](images/10fb6613dc049199a345c0bebe7597a0dec862181eda312accfb71954fc4b56b.jpg)  
Figure 3: Motion-Attention Routing Loss. For each object, RART encourages its target region to attend to the corresponding motion tokens while discouraging other object regions from attending to the same token block.

Foreground-Weighted Flow Matching. The active target masks define the foreground weights for the flow-matching objective (Lipman et al. 2023). Let x be the latent encoding of $\bar { V } ^ { \mathrm { t g t } }$ , ϵ be Gaussian noise, τ be a sampled flow time, and $x _ { \tau } = ( 1 - \tau ) x + \tau \epsilon$ . The target velocity is $v = \epsilon - x ,$ , and the predicted velocity is $\hat { v } _ { \theta } \stackrel { - } { = } v _ { \theta } ( x _ { \tau } ; I ^ { \mathrm { { t g t } } } , C , Z s )$ . For the active object set S, we define

$$
\begin{array} { r l } & { w _ { S } = \lambda _ { b } + \left( 1 - \lambda _ { b } \right) \underset { i \in S } { \operatorname* { m a x } } m _ { i } , } \\ & { \mathcal { L } _ { \mathrm { f m } } = \mathbb { E } _ { \boldsymbol { x } , \boldsymbol { \tau } , \epsilon } \left[ \frac { \sum w _ { S } \odot ( \hat { v } _ { \theta } - \boldsymbol { v } ) ^ { 2 } } { \sum w _ { S } } \right] . } \end{array}\tag{6}
$$

In Eq. (6), $w _ { \mathscr { S } }$ is broadcast over latent channels, and the sums cover all latent elements. Foreground positions receive unit weight, while background positions receive $\lambda _ { b } ,$ with $0 < \lambda _ { b } \bar { < } 1$ . We also apply motion dropout by replacing $Z _ { S }$ with zeros with a small probability, enabling classifier-freestyle motion guidance at inference.

Motion-Attention Routing Loss. Reconstruction alone does not specify which region should read which motion tokens. For multi-object samples, we regularize the motionattention maps so that each object region routes to its corresponding motion tokens. Figure 3 illustrates this operation. For object i, let $a _ { i } ^ { \mathrm { { t g t } } }$ denote the attention mass from latent positions within its target-mask sequence $m _ { i }$ to the corresponding motion-token block, and let $a _ { i } ^ { \mathrm { { o t h } } }$ denote the attention mass from the other active object regions to the same block. Both quantities are obtained directly from the motionattention maps and averaged over heads, frames, and valid spatial positions. The per-object routing loss is

$$
\ell _ { \mathrm { r o u t e } , i } = \frac { a _ { i } ^ { \mathrm { o t h } } } { \mathrm { s g } ( a _ { i } ^ { \mathrm { t g t } } ) + a _ { i } ^ { \mathrm { o t h } } } + \lambda _ { \mathrm { m i n } } \mathrm { m a x } ( \rho - a _ { i } ^ { \mathrm { t g t } } , 0 ) ^ { 2 } .\tag{7}
$$

Here $\operatorname { s g } ( \cdot )$ stops gradients through the target-region term in the denominator. The first term penalizes attention routed from other object regions, while the second is activated when $a _ { i } ^ { \mathrm { { t g t } } } < \rho$ to prevent all motion-token attention from collapsing. The weight $\lambda _ { \mathrm { m i n } }$ controls this minimum-attention constraint, and $\rho$ is the target attention threshold. Averaging Eq. (7) over active objects and motion-attention modules gives ${ \mathcal { L } } _ { \mathrm { r o u t e } } .$ The final objective is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { f m } } + \lambda _ { \mathrm { r o u t e } } \mathcal { L } _ { \mathrm { r o u t e } } ,\tag{8}
$$

where $\lambda _ { \mathrm { { r o u t e } } }$ balances the routing loss. The routing loss in Eq. (8) is applied only to non-dropped multi-object training batches and is obtained from the same forward-pass attention maps, without an additional denoiser evaluation. It therefore directly supervises object-to-motion assignment rather than relying on reconstruction alone.

## Experiments

In this section, we report quantitative and human evaluation results in Table 1, qualitative comparisons in Figure 4, and component analyses in Table 2, Figure 5, and Figure 6.

## Experimental Setup

Datasets. We train RART on a mixture of OpenVidHD (Nan et al. 2025) and HumanVid (Wang et al. 2024a) videos. After filtering invalid and low-quality samples, the training set contains 286,573 video entries. We extract per-frame object masks with Grounding-SAM2 (Liu et al. 2024; Ravi et al. 2025). For evaluation, we extend the FlexiMMT test set (Li et al. 2026) to a multi-object benchmark with 266 test cases and 25 reference videos. Each test sample consists of a target first frame, a text prompt, target first-frame object masks, and multiple reference video-mask pairs. Each target mask specifies where a motion should be applied in the target image, while the corresponding reference-mask sequence marks the object providing that motion in the reference video.

Baselines. We compare RART with four representative implicit methods closely related to motion transfer. AnyV2V (Ku et al. 2024) is a training-free method that injects source-video features into an image-to-video process. DisMo (Ressler-Antal et al. 2025) is a feed-forward method that learns appearance-agnostic motion representations for openworld motion transfer. SCAIL-2 (Yan et al. 2026) is a character-animation method that directly conditions generation on driving videos and uses in-context masks to bind driving and target characters. FlexiMMT (Li et al. 2026) is the closest multi-object multi-motion baseline, which learns object-wise motion tokens under mask constraints but requires reference-specific optimization.

To evaluate all baselines in our setting, we implement the necessary task-specific adaptations while retaining their released backbone weights and core generation mechanisms. For AnyV2V, we independently extract intermediate features from each reference video, fuse them while preserving their mean and variance, and use the fused features for image-tovideo inference conditioned on the target image. For DisMo, we independently extract the motion embedding of each reference and fuse the embeddings before its CogVideoX generator. For SCAIL-2, we assemble the independently masked reference clips into a unified multi-character driving sequence and use color-coded masks to preserve object correspondence. As for FlexiMMT, we use an object-specific text description of each designated object during training to distinguish the motion sources in multi-object reference videos. All methods receive the same target image and reference videos. Object assignments are provided directly when supported; otherwise, they are encoded through the task-specific adaptations described above. Apart from these adaptations, all other configurations follow the respective original papers.

Evaluation metrics. We mainly follow the evaluation protocol of FlexiMMT (Li et al. 2026). Specifically, we report CLIP-based text similarity, temporal consistency, and appearance consistency, together with two object-wise motion metrics: flow fidelity and motion fidelity. Flow fidelity compares mask-restricted optical-flow statistics between the reference object and the generated target object, while motion fidelity compares tracking trajectories inside the corresponding masks. For multi-object samples, we compute objectlevel scores and average them over all controlled objects. We additionally conduct human evaluation along three dimensions: target-appearance consistency, temporal consistency, and reference-motion consistency. For each of the 266 test cases, 5 annotators are shown the target first frame, all reference videos, and the outputs of five methods in randomized order. They independently select the best result for each criterion. We aggregate the selections over all test cases and report per-method selection rates, which sum to 100% for each criterion. We also report runtime to assess eficiency.

Implementation details. RART is built on a frozen Wan2.2 TI2V-5B backbone (Wan et al. 2025). We use the DisMo encoder as the pretrained motion feature extractor (Ressler-Antal et al. 2025), and train only the motion encoding and motion-attention injection modules. The final model is trained for 200K steps with a learning rate of $1 \times 1 0 ^ { - 5 }$ under our mixed-object training configuration. We sample 49 frames from each training video. Training begins at 320×480 resolution and is increased to 480 × 736 at 120K steps for high-resolution finetuning. We use 4 motion tokens per latent frame and 2 residual target-mask cross-attention layers. During training, we sample 2 to 6 active objects for multi-object clips, use motion dropout with probability 0.1, caption dropout with probability 0.1, and apply mask perturbation with probability 0.6. As an auxiliary augmentation, each reference video–mask pair undergoes an independent spatial transform shared across frames, while the target condition remains unchanged. We set the background weight to $\lambda _ { b } = 0 . 1$ . The motion-attention routing loss is enabled from step 170K with $\lambda _ { \mathrm { { r o u t e } } } = 0 . 2 , \lambda _ { \mathrm { { m i n } } } = \bar { 0 } . 0 5$ , and $\rho = 0 . 3 .$ All experiments are conducted on eight NVIDIA H200 GPUs.

## Comparisons with State-of-the-Art Methods

Quantitative Evaluation. Table 1 presents the quantitative results. RART achieves the best flow fidelity and motion fidelity, demonstrating more accurate transfer of reference motions to the designated target objects. It also obtains competitive text similarity, temporal consistency, and appearance consistency, showing that the improvement in motion accuracy does not compromise overall video quality.

<table><tr><td rowspan="2">Method</td><td colspan="5">Automatic Metrics</td><td rowspan="2"></td><td colspan="2">Human Evaluations (%)</td><td rowspan="2">Time↓</td></tr><tr><td>TS↑</td><td>TC↑</td><td>AC↑</td><td>FF↑</td><td>MF↑</td><td>App.↑ Temp.↑</td><td>Motion↑</td></tr><tr><td>AnyV2V</td><td>0.261</td><td>0.879</td><td>0.742</td><td>0.571</td><td>0.338</td><td>0.0</td><td>0.0</td><td>0.0</td><td>176s/ref + 50s</td></tr><tr><td>DisMo</td><td>0.266</td><td>0.944</td><td>0.913</td><td>0.652</td><td>0.451</td><td>3.0</td><td>3.0</td><td>2.6</td><td>113s</td></tr><tr><td>SCAIL-2</td><td>0.266</td><td>0.922</td><td>0.906</td><td>0.648</td><td>0.428</td><td>15.8</td><td>9.0</td><td>24.1</td><td>233s</td></tr><tr><td>FlexiMMT</td><td>0.267</td><td>0.925</td><td>0.899</td><td>0.671</td><td>0.462</td><td>14.7</td><td>27.1</td><td>21.8</td><td>2h/ref + 432s</td></tr><tr><td>RART</td><td>0.265</td><td>0.928</td><td>0.903</td><td>0.713</td><td>0.480</td><td>66.5</td><td>60.9</td><td>51.5</td><td>41s</td></tr></table>

Table 1: Quantitative and human evaluations. TS, TC, AC, FF, and MF denote text similarity, temporal consistency, appearance consistency, flow fidelity, and motion fidelity, respectively. Human evaluation columns report five-way selection rates (%), while Time denotes runtime. For AnyV2V and FlexiMMT, the first time term denotes per-reference preparation.

![](images/3afa69349a51176621b1fad34b2b705bdafdf244d8254075c0eb7c0a6a3e6c3f.jpg)  
Figure 4: Qualitative comparison. The top row shows the reference motions, target image, and motion-to-object assignments; the remaining rows show frames generated by the baselines and RART. RART transfers each reference motion to its designated target object while preserving target appearance.

The latter three metrics should nevertheless be interpreted together with motion fidelity. Temporal consistency and appearance consistency are computed from frame-to-frame similarities and may therefore favor videos with limited motion, while text prompts rarely describe fine-grained object dynamics and therefore make text similarity an incomplete measure of motion transfer. Consequently, strong scores on these generic video-quality metrics are less informative when the generated motion does not follow the reference.

RART also has the shortest runtime. AnyV2V performs additional per-video noising and caches intermediate features; FlexiMMT optimizes motion tokens for each reference motion and includes an additional mask-extraction stage at inference; and SCAIL-2 uses a larger generative backbone. DisMo remains eficient, but its original formulation does not support compositional multi-object motion transfer. Overall,

RART provides the strongest balance of motion-transfer accuracy, visual quality, and inference eficiency.

Human evaluation. The human evaluation results in Table 1 show that RART achieves the highest selection rate across all three criteria, receiving 66.5% for targetappearance consistency, 60.9% for temporal consistency, and 51.5% for reference-motion consistency. These results confirm that RART transfers reference motions more accurately while maintaining coherent dynamics and target appearance.

Qualitative results. Figure 4 compares the generated videos of diferent methods. AnyV2V fails to perform multimotion transfer and produces videos with poor visual quality. DisMo better preserves video quality, but its architecture lacks explicit object-wise spatial assignment and therefore cannot transfer the desired motions precisely. FlexiMMT can transfer motions extracted from single-object reference videos, but struggles to isolate and transfer the motion of a designated region when the reference video contains multiple objects. SCAIL-2 does not natively support extracting motions from multiple reference videos; although our adaptation enables this input setting, its generated videos often drift from the target first frame and exhibit weak temporal consistency. In contrast, RART consistently extracts the specified motions, assigns them to the intended target objects, and preserves both target appearance and temporal coherence.

![](images/bd5dc1aa6a9aa2fbd587caee5480fa3543a3a50c072694b48108b11c801de03c.jpg)  
Figure 5: Qualitative ablation. The top row shows reference motions and their assignments to target regions; the bottom panels compare RART with variants without frame alignment, target-mask anchoring, or routing loss.

![](images/8e495fcc58dd5f61afd6465959535eb64f824a2bd42cf71c0a3d0b2915e98e25.jpg)  
Figure 6: Motion-attention heat maps. Columns correspond to the two reference motions, while rows compare the model without the routing loss and RART. Warmer colors indicate stronger attention to each motion-token block.

## Ablation Study

w/o Frame Alignment. We train this variant by replacing frame-aligned motion cross-attention with global motion cross-attention while keeping all other settings unchanged. As shown in Figure 5, neither the bear nor the hedgehog follows the temporal motion pattern of its reference video, whereas the full model with frame-aligned motion crossattention generates correctly aligned motions. Consistently, Table 2 shows that this variant obtains lower FF and MF scores than the full model. These results demonstrate that frame-aligned motion cross-attention is essential for establishing temporal correspondence.

w/o Target-Mask Anchoring. To evaluate the efect of target-mask anchoring, we replace every target mask M<sup>tgt</sup><sub>i</sub> with an all-ones mask at inference. Consequently, all objects receive the same mask condition and no longer have objectspecific target-region information. As shown in Figure 5, the bear incorrectly follows the motion intended for the hedgehog when the spatial anchors are removed. Table 2 also reports lower FF and MF scores for this variant than for the full model. Together, these results demonstrate that target-mask anchoring is crucial for assigning each reference motion to its intended target region.

<table><tr><td>Variant</td><td>FF↑</td><td>MF↑</td></tr><tr><td>w/o Frame Alignment</td><td>0.678</td><td>0.400</td></tr><tr><td>w/o Mask Anchor</td><td>0.690</td><td>0.413</td></tr><tr><td>w/o Routing Loss</td><td>0.727</td><td>0.448</td></tr><tr><td>RART</td><td>0.713</td><td>0.480</td></tr></table>

Table 2: Quantitative ablations. We replace frame-aligned motion attention with global attention, use all-ones target masks, or remove the motion-attention routing loss.

w/o Motion-Attention Routing Loss. We continue training from step 170K without the motion-attention routing loss, while keeping all other settings unchanged. Figure 5 shows that, although the hedgehog partially follows its reference motion, the bear incorrectly performs the motion intended for the hedgehog, revealing cross-object motion leakage.

Table 2 further quantifies this efect. The routing loss raises MF from 0.448 to 0.480, a 7.1% relative improvement, with only a small FF change from 0.727 to 0.713. FF captures local flow magnitude and direction, whereas MF measures trajectory-level correspondence, making them complementary indicators of motion transfer. The full model therefore retains strong local flow fidelity while substantially improving trajectory-level correspondence.

To examine the routing behavior directly, we further visualize the attention maps in Figure 6. For each object, we aggregate the weights assigned to its motion-token block at every latent spatial position and upsample the resulting heat map to the original video resolution. With the routing loss, attention to each object’s motion tokens is concentrated in its designated target region. Without the loss, the attention maps of diferent motion sources become substantially mixed. Together with the qualitative and quantitative results, these heat maps show that the routing loss improves object-wise motion assignment and suppresses cross-object leakage.

## Conclusions

We presented RART, a unified framework for compositional multi-object motion transfer. RART represents each reference motion with object-wise tokens anchored by the corresponding target mask, and injects temporally matched tokens into a frozen video Difusion Transformer through framealigned motion cross-attention. The motion-attention routing loss further strengthens the correspondence between target regions and motion sources, reducing cross-object motion leakage. RART jointly composes multiple reference motions in one denoising pass, without separate optimization for each reference video. Quantitative, qualitative, and human evaluations show that RART delivers accurate motion transfer, strong visual quality, and eficient inference; ablations and attention visualizations support our key design choices. Future work will explore broader applications ofRART in long-form generation and interactive multi-object video creation.

## References

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025. Qwen3-VL Technical Report. arXiv:2511.21631.

Blattmann, A.; Dockhorn, T.; Kulal, S.; Mendelevitch, D.; Kilian, M.; Lorenz, D.; Levi, Y.; English, Z.; Voleti, V.; Letts, A.; et al. 2023. Stable video difusion: Scaling latent video difusion models to large datasets. arXiv:2311.15127.

Brooks, T.; Peebles, B.; Holmes, C.; DePue, W.; Guo, Y.; Jing, L.; Schnurr, D.; Taylor, J.; Luhman, T.; Luhman, E.; et al. 2024. Video generation models as world simulators. OpenAI Blog, 1(8): 1.

Carion, N.; Gustafson, L.; Hu, Y.-T.; Debnath, S.; Hu, R.; Coll-Vinent, D. S.; Ryali, C.; Alwala, K. V.; Khedr, H.; Huang, A.; et al. 2026. SAM 3: Segment Anything with Concepts. In ICLR.

Chen, H.; Xia, M.; He, Y.; Zhang, Y.; Cun, X.; Yang, S.; Xing, J.; Liu, Y.; Chen, Q.; Wang, X.; et al. 2023. Videocrafter1: Open difusion models for high-quality video generation. arXiv:2310.19512.

Cheng, B.; Misra, I.; Schwing, A. G.; Kirillov, A.; and Girdhar, R. 2022. Masked-attention mask transformer for universal image segmentation. In CVPR, 1290–1299.

Chu, R.; He, Y.; Chen, Z.; Zhang, S.; Xu, X.; WANG, D.; Yi, H.; Liu, X.; Zhao, H.; Liu, Y.; et al. 2026. Wan-move: Motion-controllable video generation via latent trajectory guidance. NeurIPS, 38: 404–432.

Dhariwal, P.; and Nichol, A. 2021. Difusion models beat gans on image synthesis. NeurIPS, 34: 8780–8794.

Esser, P.; Kulal, S.; Blattmann, A.; Entezari, R.; Müller, J.; Saini, H.; Levi, Y.; Lorenz, D.; Sauer, A.; Boesel, F.; et al. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In ICML.

He, Y.; Yang, T.; Zhang, Y.; Shan, Y.; and Chen, Q. 2022. Latent video difusion models for high-fidelity long video generation. arXiv:2211.13221.

Ho, J.; Chan, W.; Saharia, C.; Whang, J.; Gao, R.; Gritsenko, A.; Kingma, D. P.; Poole, B.; Norouzi, M.; Fleet, D. J.; et al. 2022a. Imagen video: High definition video generation with difusion models. arXiv:2210.02303.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising difusion probabilistic models. NeurIPS, 33: 6840–6851.

Ho, J.; and Salimans, T. 2021. Classifier-Free Difusion Guidance. In NeurIPSW.

Ho, J.; Salimans, T.; Gritsenko, A.; Chan, W.; Norouzi, M.; and Fleet, D. J. 2022b. Video difusion models. NeurIPS, 35: 8633–8646.

Hong, W.; Ding, M.; Zheng, W.; Liu, X.; and Tang, J. 2023. CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers. In ICLR.

Hu, L. 2024. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In CVPR, 8153–8163.

Jeong, H.; Park, G. Y.; and Ye, J. C. 2024. Vmc: Video motion customization using temporal attention adaption for text-to-video difusion models. In CVPR, 9212–9221.

Karaev, N.; Rocco, I.; Graham, B.; Neverova, N.; Vedaldi, A.; and Rupprecht, C. 2024. CoTracker: It Is Better to Track Together. In ECCV.

Karras, J.; Holynski, A.; Wang, T.-C.; and Kemelmacher-Shlizerman, I. 2023. Dreampose: Fashion image-to-video synthesis via stable difusion. In ICCV, 22623–22633. IEEE.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.- Y.; et al. 2023. Segment anything. In CVPR, 4015–4026.

Kong, W.; Tian, Q.; Zhang, Z.; Min, R.; Dai, Z.; Zhou, J.; Xiong, J.; Li, X.; Wu, B.; Zhang, J.; et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv:2412.03603.

Ku, M.; Wei, C.; Ren, W.; Yang, H.; and Chen, W. 2024. AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks. TMLR. Reproducibility Certification.

Li, Y.; Gong, D.; Cao, X.; Yuan, J.; Li, D.; Zhou, L.; Koh, Y. S.; Yan, C.; and Zhang, X. 2026. Let Your Image Move with Your Motion!–Implicit Multi-Object Multi-Motion Transfer. In CVPR, 11207–11217.

Ling, P.; Bu, J.; Zhang, P.; Dong, X.; Zang, Y.; Wu, T.; Chen, H.; Wang, J.; and Jin, Y. 2025. MotionClone: Training-Free Motion Cloning for Controllable Video Generation. In ICLR.

Lipman, Y.; Chen, R. T. Q.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2023. Flow Matching for Generative Modeling. In ICLR.

Liu, S.; Zeng, Z.; Ren, T.; Li, F.; Zhang, H.; Yang, J.; Jiang, Q.; Li, C.; Yang, J.; Su, H.; et al. 2024. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 38–55. Springer.

Long, F.; Qiu, Z.; Yao, T.; and Mei, T. 2024. VideoStudio: Generating Consistent-Content and Multi-Scene Videos. In ECCV.

Long, F.; Wang, C.; Gao, Z.; Zhong, W.; Cheng, Y.; Hou, X.; Li, Y.; Cao, X.; Sun, X.; Chen, X.; and Liu, Y. 2026. CoinVE-200K: A Large-Scale High-Quality Dataset for Compositional Instruction-Guided Video Editing. arXiv preprint arXiv:2608.17566.

Ma, Y.; He, Y.; Cun, X.; Wang, X.; Chen, S.; Li, X.; and Chen, Q. 2024. Follow your pose: Pose-guided text-to-video generation using pose-free videos. In AAAI, volume 38, 4117–4125.

Nan, K.; Xie, R.; Zhou, P.; Fan, T.; Yang, Z.; Chen, Z.; Li, X.; Yang, J.; and Tai, Y. 2025. OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation. In ICLR.

Ouyang, W.; Dong, Y.; Yang, L.; Si, J.; and Pan, X. 2024. I2vedit: First-frame-guided video editing via image-to-video difusion models. In SIGGRAPH Asia, 1–11.

Peebles, W.; and Xie, S. 2023. Scalable difusion models with transformers. In CVPR, 4195–4205.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Meila, M.; and Zhang, T., eds., PMLR, volume 139, 8748– 8763.

Ravi, N.; Gabeur, V.; Hu, Y.-T.; Hu, R.; Ryali, C.; Ma, T.;

R.; Dollar, P.; and Feichtenhofer, C. 2025. SAM 2: Segment Anything in Images and Videos. In ICLR.

Ressler-Antal, T.; Fundel, F.; Alaya, M. B.; Baumann, S. A.; Krause, F.; Gui, M.; and Ommer, B. 2025. DisMo: Disentangled Motion Representations for Open-World Motion Transfer. In NeurIPS.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent difusion models. In CVPR, 10684–10695.

Singer, U.; Polyak, A.; Hayes, T.; Yin, X.; An, J.; Zhang, S.; Hu, Q.; Yang, H.; Ashual, O.; Gafni, O.; Parikh, D.; Gupta, S.; and Taigman, Y. 2023. Make-A-Video: Text-to-Video Generation without Text-Video Data. In ICLR.

Song, J.; Meng, C.; and Ermon, S. 2021. Denoising Difusion Implicit Models. In ICLR.

Song, Y.; and Ermon, S. 2019. Generative modeling by estimating gradients of the data distribution. NeurIPS, 32.

Song, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Ermon, S.; and Poole, B. 2021. Score-Based Generative Modeling through Stochastic Diferential Equations. In ICLR.

Teed, Z.; and Deng, J. 2020. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. In ECCV, 402–419.

Villegas, R.; Babaeizadeh, M.; Kindermans, P.-J.; Moraldo, H.; Zhang, H.; Safar, M. T.; Castro, S.; Kunze, J.; and Erhan, D. 2023. Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions. In ICLR.

Wan, T.; Wang, A.; Ai, B.; Wen, B.; Mao, C.; Xie, C.-W.; Chen, D.; Yu, F.; Zhao, H.; Yang, J.; et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv:2503.20314.

Wang, J.; Yuan, H.; Chen, D.; Zhang, Y.; Wang, X.; and Zhang, S. 2023a. Modelscope text-to-video technical report. arXiv:2308.06571.

Wang, X.; Yuan, H.; Zhang, S.; Chen, D.; Wang, J.; Zhang, Y.; Shen, Y.; Zhao, D.; and Zhou, J. 2023b. Videocomposer: Compositional video synthesis with motion controllability. NeurIPS, 36: 7594–7611.

Wang, Z.; Li, Y.; Zeng, Y.; Fang, Y.; Guo, Y.; Liu, W.; Tan, J.; Chen, K.; Xue, T.; Dai, B.; et al. 2024a. Humanvid: Demystifying training data for camera-controllable human image animation. NeurIPS, 37: 20111–20131.

Wang, Z.; Yuan, Z.; Wang, X.; Li, Y.; Chen, T.; Xia, M.; Luo, P.; and Shan, Y. 2024b. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH, 1–11.

Wu, C.; Long, F.; Huang, B.; Sun, X.; Chen, X.; Guo, C.- L.; and Li, C. 2026. Thinking on Shots: Consistent Multi-Shot Video Editing with Agentic Reasoning. arXiv preprint arXiv:2608.26809.

Xu, Z.; Zhang, J.; Liew, J. H.; Yan, H.; Liu, J.-W.; Zhang, C.; Feng, J.; and Shou, M. Z. 2024. Magicanimate: Temporally consistent human image animation using difusion model. In CVPR, 1481–1490.

Yan, W.; Guo, F.; Yang, Z.; and Tang, J. 2026. SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning. arXiv:2606.10804.

Yatim, D.; Fridman, R.; Bar-Tal, O.; Kasten, Y.; and Dekel, T. 2024. Space-time difusion features for zero-shot text-driven motion transfer. In CVPR, 8466–8476.

Yin, S.; Wu, C.; Liang, J.; Shi, J.; Li, H.; Ming, G.; and Duan, N. 2023. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv:2308.08089.

Zhang, S.; Zhuang, J.; Zhang, Z.; Shan, Y.; and Tang, Y. 2025a. Flexiact: Towards flexible action control in heterogeneous scenarios. In ACM SIGGRAPH, 1–11.

Zhang, Y.; Wei, Y.; Jiang, D.; ZHANG, X.; Zuo, W.; and Tian, Q. 2024. ControlVideo: Training-free Controllable Text-to-video Generation. In ICLR.

Zhang, Z.; Long, F.; Qiu, Z.; Pan, Y.; Liu, W.; Yao, T.; and Mei, T. 2025b. MotionPro: A Precise Motion Controller for Image-to-Video Generation. In CVPR.

Zhao, R.; Gu, Y.; Wu, J. Z.; Zhang, D. J.; Liu, J.-W.; Wu, W.; Keppo, J.; and Shou, M. Z. 2024. Motiondirector: Motion customization of text-to-video difusion models. In ECCV, 273–290. Springer.

<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Video backbone</td><td>Wan2.2 TI2V-5B</td></tr><tr><td>Motion extractor</td><td>DisMo Encoder</td></tr><tr><td>Training steps</td><td>200K</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Hardware</td><td> ${ 8 \ \times \ \mathrm { N V I D I A } }$  H200</td></tr><tr><td>Batch size per GPU</td><td>GPUs 1</td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W }$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Warmup steps</td><td>4000</td></tr><tr><td>Training frames</td><td>49</td></tr><tr><td>Initial resolution</td><td> $3 2 0 \times 4 8 0$ </td></tr><tr><td>Final resolution</td><td> $4 8 0 \times 7 3 6$ </td></tr><tr><td>Resolution switch</td><td>120K steps</td></tr><tr><td>Motion tokens per frame</td><td>4</td></tr><tr><td>Target-mask attention layers</td><td>2</td></tr><tr><td>Active objects per clip</td><td>1-6</td></tr><tr><td>Motion dropout</td><td>0.1</td></tr><tr><td>Caption dropout</td><td>0.1</td></tr><tr><td>Mask perturbation probability</td><td>0.6</td></tr><tr><td> $\lambda _ { b }$ </td><td>0.1</td></tr><tr><td> $\lambda _ { \mathrm { { r o u t e } } }$ </td><td>0.2</td></tr><tr><td> $\lambda _ { \mathrm { m i n } }$ </td><td>0.05</td></tr><tr><td> $\rho$ </td><td>0.3</td></tr><tr><td>Routing loss enabled at</td><td>170K steps</td></tr></table>

Table 3: Training configuration of the reported RART model.

This document provides additional implementation details, dataset and evaluation protocols, baseline adaptations, additional qualitative results, and liminations of RART. The main paper contains all information essential to the central claims; this document supplies supporting material and reproducibility details.

## Training and Inference Details

## Training Details

Table 3 summarizes the training configuration used for the reported checkpoint. RART uses a frozen Wan2.2 TI2V-5B backbone (Wan et al. 2025) and the pretrained DisMo motion extractor (Ressler-Antal et al. 2025).

## Inference Details

We use the checkpoint at 200K training steps for all reported results. Each sample is generated at a resolution of 480×736 with 49 frames. We use a 50-step Flow-UniPC sampler with a flow shift of 5.0. The text and motion guidance scales are set to 3.0 and 5.0, respectively, and the random seed is fixed to 42 for all test samples.

At each sampling step, let $\hat { v } _ { u }$ denote the text-unconditional prediction, $\hat { v } _ { c }$ the text-conditioned prediction without motion, and $\hat { v } _ { c , m }$ the prediction with both text and motion conditions. We combine them as

$$
\hat { v } = \hat { v } _ { u } + 3 . 0 ( \hat { v } _ { c } - \hat { v } _ { u } ) + 5 . 0 ( \hat { v } _ { c , m } - \hat { v } _ { c } ) .
$$

Reference videos and their masks are sampled to 49 frames and resized to the generation resolution. Before motion extraction, pixels outside each reference mask are set to zero. Target masks are resized to the latent grid and passed to the mask-encoder module. All reference motions are extracted independently and injected jointly without reference-specific optimization.

## Dataset and Evaluation Details

## Training Set Construction

We construct our data pool from OpenVidHD (Nan et al. 2025) and HumanVid (Wang et al. 2024a). For both datasets, we first retain videos containing at least 49 frames and labeled with static camera motion. We then use Qwen3-VL-8B-Instruct (Bai et al. 2025) to re-caption each video from its original caption and uniformly sampled frames. The model returns a refined caption together with a list of main subjects and their corresponding actions. Subjects labeled as static are removed because they do not provide useful motion supervision; videos without any remaining moving subject are excluded from the subsequent segmentation stage.

We next obtain an object mask sequence for every retained subject using Grounding DINO and SAM2 (Liu et al. 2024; Ravi et al. 2025). Given the subject description, Grounding DINO detects the subject in the first frame, and its highestconfidence bounding box initializes the SAM2 image mask. SAM2 then propagates this mask throughout the video, producing a temporally aligned mask sequence for each object. We use box and text thresholds of 0.3 and 0.25, respectively, and discard the video if any annotated subject cannot be detected or segmented. After preprocessing and filtering, we obtain 287,540 valid videos with object-level motion descriptions and per-frame masks. We use 286,573 videos for training and reserve the remaining 967 videos as a held-out pool for constructing the test set.

## Test Set Construction

We construct our test set by extending the FlexiMMT test set (Li et al. 2026). From the 967 held-out videos, we select 5 reference video pair and 171 target image pairs and combine them with the original FlexiMMT test data. The resulting test set contains 266 test cases and 25 reference videos in total. None of the newly selected reference videos or target images is used for training RART. Each test case contains a target image, a text prompt, target first-frame masks, and reference video–mask pairs, where the per-frame reference masks identify the objects whose motions should be transferred.

## Automatic Metrics

We follow the automatic evaluation protocol implemented in FlexiMMT (Li et al. 2026) and report five metrics, for which higher values are better. Before computing the objectwise motion metrics, we initialize SAM2 (Ravi et al. 2025) with each target first-frame mask and propagate it through the generated video to obtain per-frame object masks. For multi-object samples, FF and MF are first averaged over all controlled objects. All five metrics are averaged over the test set.

CLIP-Based Metrics. We use CLIP ViT-B/32 (Radford et al. 2021) to compute Text Similarity (TS), Temporal Consistency (TC), and Appearance Consistency (AC). TS is the mean cosine similarity between the caption embedding and the CLIP image embeddings of all generated frames. TC measures inter-frame consistency using cosine similarities between CLIP image embeddings. AC averages the cosine similarity between the first-frame embedding and each subsequent-frame embedding to measure appearance preservation throughout the video.

Flow Fidelity. We use RAFT (Teed and Deng 2020) to estimate frame-to-frame optical flow in both the reference and generated videos. For each controlled object, FF compares the normalized magnitude and direction distributions of the mask-restricted reference and generated flows using the Bhattacharyya coeficient. The resulting scores are first averaged across corresponding frame pairs and then across controlled objects.

Motion Fidelity. We use CoTracker (Karaev et al. 2024) to extract point trajectories from the reference and generated videos, initialized within the first-frame bounding region of each object. For each tracked point, we compute its displacement from one frame to the next and normalize each displacement vector to unit length. We then construct a trajectorysimilarity matrix by averaging directional cosine similarity over frames for every generated-reference trajectory pair. For each generated trajectory, we select the highest-scoring reference trajectory and average these best-match scores over generated trajectories and controlled objects to obtain MF.

## Human Evaluation Protocol

We conduct a human evaluation comparing five models across 266 test cases. Each test case is independently evaluated by five annotators. Annotators are shown the target first frame, all reference videos, and five generated outputs. They are not informed of the method identities, and the output order is randomized for each test case. For each criterion, each annotator selects exactly one best result; ties are not allowed.

We evaluate three criteria. Target-appearance consistency measures how well the generated video preserves the objects and visual content of the target first frame. Temporal consistency evaluates whether the generated video remains coherent over time without noticeable flicker or abrupt deformation. Reference-motion consistency assesses how accurately the controlled objects follow their assigned reference motions. For each criterion, we report each model’s share of selections, which sums to 100% across the five models.

## Baseline Adaptations

Existing motion-transfer methods do not natively support RART’s input format, which consists of multiple independent reference video–mask pairs and their target-object assignments. We therefore adapt only the conditioning interface needed to represent our task, while retaining each method’s released backbone weights, core conditioning mechanism, and sampling procedure.

## AnyV2V

The original AnyV2V pipeline (Ku et al. 2024) performs DDIM inversion for one source video and injects source convolutional features together with spatial- and temporalattention queries and keys during I2VGen-XL denoising. For a sample with multiple reference videos, we run the oficial inversion and feature-extraction procedure independently for every reference. Features from the corresponding injection sites are then combined using moment-preserving fusion, which maintains the mean and variance of the reference features. We use the target image as the edited first frame and inject the fused features in the image-to-video sampling pass. Because AnyV2V has no target-region routing interface, this fused condition is applied globally. The released I2VGen-XL backbone, injection schedule, and sampling configuration are otherwise unchanged.

## DisMo

The original DisMo generation pipeline (Ressler-Antal et al. 2025) encodes one motion video into a temporally ordered motion-embedding sequence and supplies it to a motionconditioned CogVideoX-5B-I2V generator. We extend its sampling interface to accept precomputed motion embeddings. Each reference video is first processed independently by the released DisMo encoder, after which the resulting sequences are averaged element-wise and passed through this interface together with the target image and prompt. This permits several references to condition one generation pass, but the averaged embedding remains a global condition without object-specific target-mask routing. All released motionextractor and CogVideoX adapter weights, as well as the original sampling procedure, are retained. This baseline differs from RART, which uses the pretrained DisMo encoder only as a frozen feature extractor and learns separate maskanchored encoding and injection modules.

## SCAIL-2

SCAIL-2 (Yan et al. 2026) takes a target image with a colorcoded target mask and a single reference video with a framewise color-coded mask sequence. Matching colors establish object correspondence between the two inputs. To accommodate multiple independent reference videos, we use each reference-mask sequence to isolate the corresponding object from its reference video. Each extracted object is resized and positioned according to its corresponding target mask while preserving its aspect ratio. The transformed objects are then composited frame by frame into a single multi-character reference video. Each object is assigned a distinct color from SCAIL-2’s supported mask palette, producing a color-coded driving mask that preserves object correspondence. This adaptation changes only the construction of SCAIL-2 inputs; the released generator and inference procedure remain unchanged.

## FlexiMMT

FlexiMMT (Li et al. 2026) uses the training caption to obtain a mask for the described object while learning its motion. To adapt it to a multi-object reference video, we provide a separate caption describing each source object. This allows its original mask-extraction pipeline to identify the designated object and extract its motion from the shared reference video. Apart from this caption-based object specification, we follow the oficial training and inference procedures without further modification.

## Additional Qualitative Results

Figures 7 and 8 present additional results for compositional motion transfer. In each example, the blue and red reference sequences provide two independent motions, while the same colors in the target image indicate their assigned target objects. The subsequent frames show the video generated by RART.

## Limitations and Future Works

RART’s output quality remains bounded by the capabilities of its frozen video generator and pretrained motion encoder. Because the generator is not updated during training, the added motion-control modules cannot fully compensate for failures in its visual prior. Challenging motions or uncommon target geometries can therefore produce unstable object shapes or visible artifacts. Meanwhile, the motion representation may omit subtle or highly articulated dynamics, reducing motion fidelity in dificult cases.

Future work will explore adapting RART to stronger video generators and more expressive motion encoders. Improving these underlying components should further enhance visual quality and fine-grained motion preservation while retaining RART’s object-wise spatial and temporal control.

![](images/d9f5a67733c5ff90584bec362469a8295c1337743ef0646db65f8890228e469b.jpg)

![](images/af6044e587b1702f350462c69eca74dc313a2061a94116f5655910770e341fbd.jpg)

![](images/210241b16b6a2e37d6d977d844e0305755ce58fc9bac827b045885e6a504372a.jpg)

![](images/293cbcc2aef5f7d4c82ebfac54bdcc36e3b7ba406638cf5f4dfdf1e9200ae808.jpg)

![](images/3e88a12fc36321f4ceeb401e7b216748715054130094dadc39e8da21622fd5d8.jpg)

![](images/5bcacc6498b2d68d833ebb834af1d9d41f92728199334243110cdbc64c168404.jpg)

![](images/28af8b9fc6a4455cd74cda3ab83e6e2afcd3fd875433571825628a85d038d8c5.jpg)

![](images/aa22b54321afba4f283210db30447d3db1e7c3e7a0a738818300a0a728defb09.jpg)

![](images/18586e79ea9e04030a5e763e9869838e26a57886787d5c3bc19d1d0f715475c3.jpg)

![](images/f00c0ef4db5770102379561d2ca23701eb0aaf0dc7b89cad5e87e1d462e1d6a0.jpg)

![](images/f2cc9158ec606b9e018dabe74f2ff9e28374ce0badd210e8cc4051ebde71ea78.jpg)

![](images/2f7fdbfc38ddcf356641afe88b719119a554c463cc1f312022279aef118867ec.jpg)

![](images/f187e77049c81c902061cbf69bb5e149260381618f678c009255426b62b4312f.jpg)

![](images/8352ed732fad01662cea8b6685d5e42f84141fb1f0f110eb6542baccbf5e440c.jpg)

![](images/7ead274c40967b1a02cfffc06f95cc3aab3e400dbfeed6e3e0d45bb144e36e9a.jpg)

![](images/cc7f15a47423a71e5b87ca4802635899e9bb4ad77cc1877aaf291dd2358c5ace.jpg)

![](images/459eb4b8bb7e004b1326f71c00b9ce350b90ffe8bd40db1193038b881c552ea8.jpg)

![](images/ae3031eed1a7924aae90869053e6579f4098411556d8b736def34b7fc2d7731e.jpg)

![](images/784d038cb89db8115f6e5b87f42d7f642bd087fe522ce7c5ff1998e7d3b34aa1.jpg)

![](images/bce07edd67eb93104a2b0e5c6109319ec6df4f2b4fd6c84c50668e5a404654f1.jpg)

![](images/d75a7a4d1cf8957eac7eea6d7b46540611db6eb6db0f9c83c32334b347567c30.jpg)

![](images/66cc1e8af67b67fd927f85c4e0614b84e2ecb7ef0e7a4ed4ec2afc0f79e634c8.jpg)

![](images/21d6333f7a7b9ca2a4ef3d0c49fde8fb89197a6ece6c4cfe59fa59228750a4ec.jpg)

![](images/5f6b6b28fdd1c10bf5c7f627f35ff8ef2169ee847850d4700c563366d84509a9.jpg)

![](images/98a655aa0732a4f386a6fe5c00f61cf3623755d536dbc737f2dffa39398cacdd.jpg)

![](images/4cad71582a944501740c07266af7773306e4148b029eab55b94a1368ae704cac.jpg)

![](images/223ee432334f293d70a45ac5fa325154836d13c918dcf7f3b0cf72053769c5fa.jpg)

![](images/5c65e90527062fa7ecad07b5ed98c111276cc1ee5ca68cde1d0a121bf0643197.jpg)

![](images/1f8bcef0954a7716d9f006a1eea02912e8cca9ae7b53939f526c2d7d08ff4c0f.jpg)

![](images/d33776f183dc06829ca70dc5eb8382269eef46df2d48b2cc76f84dff66df7bfd.jpg)

![](images/3f50837c825b0444913992f038f7b5a30e19af87c582e623a6155bdfd607c4a0.jpg)

![](images/16a8a2af75db4df2e6533b8cf3417451d6cea8b8af9c535a752e8a3e25088d6b.jpg)

![](images/c203311dfcc2d855201524331521bea004dfc73a5fd093a4bd600fb23d569897.jpg)

![](images/70b4a41728943840a2ff2baed1c6807a352d76c4fe474465c26dca6e409c00ee.jpg)

![](images/11087821c2b81fc85ae72225a91cf1079784f97f8978bbb83ecedba22e614dff.jpg)

![](images/71917f626016b7d1c4c5dfdeb0ab3a1e8bea8f1c3077b40c0c1ecea387f3ea14.jpg)

![](images/3c2cee9d12596732cc8f76e6294a232b34c5e55a0cc17767cf66f5429a1de52c.jpg)

![](images/1536e0a51d5b5a693c151bf583527b757855fc60387967dabe78361a6dcbc87b.jpg)

![](images/f7f2c6bf2bbca324aec2f379495a44b04f470e6015ea4da83f3a2e26127fdae1.jpg)

![](images/e7c0857d456d2dee26ca80895d9106ac69367e8246cc7e8f3a53b9d600c022ee.jpg)

![](images/766aa807b5c85c39b94deaeb4d222c6c0559d4960bc8470f7748c2d2bea2a499.jpg)

![](images/fb22e8f4546bc7c57cb3393c63a7dfd0c8a07bdcb991eb82ccb803fca0151bc4.jpg)

![](images/f1a0681c1cfbc1f6e489317d0854c2ba8ec336f7d64a89a5a0eb76656eb9f86e.jpg)

![](images/223dee1daeb35987e42768b254b2bd3e269c89745a4cb50e78d51321daa667c1.jpg)

![](images/2090cee0aed93fee31a5165e3f3368376e5d1554411492495e922d64ecfd751e.jpg)

![](images/74acd5aedffdd257a51efb9646ee75a52ef7ceb7c86841ccdd0483a1b3944827.jpg)

![](images/263a62053f138ed6bb8f779ba9dbfc0e28cdbb92305a9485766c9139116b9af5.jpg)

![](images/c3f2e4c8427d791b0a00115b1c647164c886ec4627ea1c9f4df3069b59e9bf9f.jpg)

![](images/0e1ad3b6a70a18f97a7b615d363cc237bea40b08aba317397b0892c379490e9d.jpg)

![](images/b18e3ef829b3311153346fb622361300fb9b2510606cd80ed857e964d1953008.jpg)

![](images/1ab5ed3e7ec4e413cc8440d4e06687eb7b083b097c941494f1f6011d6a87f91c.jpg)

![](images/4704854f3be4dda55af2d28e080d6c2897a33f3af8534a955afc78dd51aae1f8.jpg)

![](images/925214e623e6a2a6b80d5b459a861e381c5dcb1d41f32b8df70fd5df650006df.jpg)

![](images/b02c5ec649136e0a64c5256d15a134a277123fa2b0e475af478483dbbfeddcd4.jpg)

![](images/3cfc681c60fc59c0e831147169e20b0c657c381975d62ba5a6278022353d5b84.jpg)  
Figure 7: Additional qualitative results for compositional motion transfer. Each example shows color-coded reference sequences at the top, followed by the target-object assignments and generated frames at the bottom. Blue and red indicate the correspondence between each reference motion and its target object.

![](images/e35c089825fdb2db8b93d247ac7a0b0fb19ad7409d8403180468c2bf3b2d50b2.jpg)

![](images/854fd8fa1715ae7a26cec1cdd154ceae7cebe0b070c316c68515ec9a1b0c9e36.jpg)

![](images/9a48519ec2e23a70f1371fa37083346c9f8822fd8a4000fb4b93399611ac79fc.jpg)

![](images/973d3270be15e5aa283d740d6731b65a54d7b2171b4bbfb4ab40c5de03213387.jpg)

![](images/43cf678f92b7935dbdd092d2308071da22298c4244aeea3458c90431c0adc2fe.jpg)

![](images/f990c7f6522683e0b9fc56c4c37273d5970a12fb79e51df58f54c24044c885ac.jpg)

![](images/10ff2c53be90ee59a6e070c9955a89cb5c26f6b734201329ba2ed3287a74b984.jpg)

![](images/45714796a3668cedc7e94750b64860e3513c2bf9afd5610b68ce6328748c0861.jpg)

![](images/a9bffd1e00a17f68f9673880007b8a8412ac707f3539620c75291253ed2dc4c3.jpg)

![](images/a190a5cec74b69af5806a1d63a2dadb17c77e6e54aa16bd98f672eaa7b56fb69.jpg)

![](images/781552ef9b979c9a9b5c3fcb504a1486520d814478a1b3fb1e835d80bea75236.jpg)

![](images/7d79d24488454637b90f67324bd76578823391f19a09b8e34f5f802c20df6ef5.jpg)

![](images/9e2f8c042a27e11f854093c9bd88aee9d0a6af6d8de8a0990928e0bfd3f53747.jpg)

![](images/53f8fd3cb92ac5b3541389166017a6ff0180a3eccb35e55fb5aba47003505519.jpg)

![](images/c2b32501f818da3458c8d308cdec22c53f5bb1e0ee328c300d76e35d1affb925.jpg)

![](images/7622d4aa4b50720d24ef09c61a0deca4b8127c4b47625aae853c2764fece05be.jpg)

![](images/3ca2843bc56d654a7c3fac2ff016b1e8e24818eabae95187439f143b201660ce.jpg)

![](images/9bd9c68c538911ec5ee79de0111d6279a144c9dc8cd8f98228ffbb7b90b162ab.jpg)

![](images/88cb11b6246cd957e606322bcbe90830369fb25ac9ff28206210ed3a85cb7e48.jpg)

![](images/f644bb10362d372e3d06bf6e4916b40ec1b4d321c0ec533314eb046f9f450bbb.jpg)

![](images/873cce4ebbae1e2e49f52e130270cdd022c953f734489b585bb0d3e472198b54.jpg)

![](images/81c9c9c03073fae83d67612276e6a063ce4d2f5f6e2a3ab05dbc5084ac75c286.jpg)

![](images/dff784038f1a1bc7e9ca24af481a02c73b1f6cf31bcd87432334c1e13ff43a5f.jpg)

![](images/29de5bf8e414ed458fffb2a73173c1d8f5bb2475c01652a146f67c4a68167efa.jpg)

![](images/517a2a5619d19bf633936429eceb5f518f8a7d54bb5472c35ff416943023885f.jpg)

![](images/8611b2cbe95d7cf918f757d2928da2abb1e7fa7d12340f8305066ef9099e5e43.jpg)

![](images/7e762fa0c84a2249a4ec7f5506049452af4342549c8504941c88d7e53801afdf.jpg)

![](images/d321bf9227e84829babfaca2e25fbdf01c9338117092ebff43468255cdccc555.jpg)

![](images/92050b71f6e676da7e7f9042febf7eea4e06d1b827fd9c0d9aba5536bdb78829.jpg)

![](images/f4f8e26b19e84791a0687dd47c601f51956201bc37d511adccaf8387e7490e18.jpg)

![](images/7e27ceac9b24486663ebb36faec964deff2c46491f07a6b0377a6d6fcb1a81cb.jpg)

![](images/eeb4ef9652861def75719bd7b955be6265a6ba14e90592fc2eebbf3950b37a4c.jpg)

![](images/95013e8f22addba8563c92b398bcd6921a998f86bb6c712fdaddc22d9182b2b5.jpg)

![](images/ecfb3efdb7826f76c38bf5e22df958cb62c53993816b29c3d69df87b9ef3d5d7.jpg)

![](images/916cffd3152689383b018125fd2fd9512ba5abc0dcdb693c5f8ba9e3b8df7340.jpg)

![](images/308da7565a0aa0b99663cb54043c04189f95936d0a6ca3bb1f6cbeeb84a3e55f.jpg)

![](images/bddc3795f8e49a95dfb2f8532d3a46af74a50e03cfb0508f4e9ad1cd08c7952a.jpg)

![](images/e42cca6e1631be61962cb7bf81acf957c6d87dfc4a3ea9a04d42c0e5a4df6e1c.jpg)

![](images/590495ff87993abb2ba3ff57eae5fb2e9baffa295e171265956a8875824d4a55.jpg)

![](images/f5720017216adfa3f12f84bb042e268a62d7df1e2046f11315f35f0aa4659c67.jpg)

![](images/7274eb85c2b4949b10b742cfac6a561872124bcd4ba151cee89e245b8b61275b.jpg)

![](images/1a95dad73056c62fe818589404b85d9594e164df5872c943c78efc830dcc392c.jpg)

![](images/205406182fb6435c9bd73320e6dadba5107dca1b6ea312637f632c67364d4470.jpg)

![](images/71aa558d87c0a08b49f1def95f993b02245cc0648a867d9592320309d6075ed6.jpg)

![](images/e48e6ce8634f1dd46883d73a2227cad8c6e239b5e1e35efc059325ccb3459b89.jpg)

![](images/285832ed0f8206a9b7bd4ebb578f47e6f4b453bfadceef154df571290723dbec.jpg)

![](images/4ed085d8e60e7b3cd1a9e1a5314aa2ef29118f8a1db340ef1adbd200d9acd7c3.jpg)

![](images/f006ac919a2c0eba341f8f13cb8dff1bbd90dda84a3a20dbd412f1db5f81d574.jpg)

![](images/93b8b1d82f242646cb0c9aab9dda6ee627bc69b791a135f85e7c4a4c85126e5c.jpg)

![](images/efdb6d1330b23feaa046024669dcadaefe381346dc6b5e55397085bee88715f9.jpg)

![](images/1df4047a98687c844c4b85d61d7431085d98f6e7f600ec96de595a8608fe6c37.jpg)

![](images/c6fb834203ce1cf246d70ca94db98865927e7165084686f5c5882c6579270b8b.jpg)

![](images/b44f73346d5da44032d6ebcc7567321193ca66fe6ec7f711c6e0c872824097a2.jpg)

![](images/4d44562cedab13c83a2ff5b6a54ef737516815ee8664d551debb9e0c2edfc22d.jpg)

![](images/1ef4c83c7e78d2b03abb6b4c701c845fb4f3a75af8c81de5e481134dccc960f2.jpg)

![](images/acedf57f09d8706246850a7f2649abc4c3899d187ea1372d3fb96177a0e34f82.jpg)

![](images/d6b6668538dc2dd3ea264f623cacc01dd8b7b24fefa067482aa38d211a2c0cab.jpg)

Figure 8: Further qualitative results for compositional motion transfer. RART jointly transfers independently specified reference motions to their color-matched target objects while preserving the appearance and scene of the target image.