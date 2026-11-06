# WAM-SIM0.1: WORLD ANCHORED MEMORY MEETSONLINE RL FOR CONSISTENT AND REALISTIC DRIV-ING SIMULATION

Jiahao Cui<sup>1,\*</sup>, Jiaye Li<sup>1,\*</sup>, Wenhao Zhang<sup>1,\*</sup>, Yaoyi Li<sup>2,\*</sup>, Shan Luan<sup>1,\*</sup>, Quanhui Tang<sup>1</sup> Jianhua Han<sup>2</sup>, Lin Cheng<sup>2</sup>, Hang Xu<sup>2</sup>, Siyu Zhu<sup>1,†</sup>

<sup>1</sup>Fudan University

<sup>2</sup>Shenzhen Yinwang Intelligent Technology Co., Ltd

<sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding author.

## ABSTRACT

Autoregressive driving world models systematically suffer from structural drift, identity drift, and quality decay over long rollout horizons due to transient conditioning signals that lack persistent state. We present WAM-Sim, a driving world model anchored by a persistent hierarchical memory and optimized via online flow-matching RL. WAM-Sim structures persistent context across three temporal and semantic scales: pose-aligned structural tokens encoding persistent road topology, read-only scene tokens supplying geolocated background appearance, and write-once instance tokens preserving individual traffic participant identities across occlusions and chunk boundaries. To restore visual sharpness and physical plausibility degraded by multi-scale conditioning, we introduce an online posttraining framework that optimizes self-generated rollouts via flow-matching RL guided by hierarchy-aligned visual rewards and a vision-language model physical judge. The invariant memory empirically confines policy optimization within the anchored context, helping suppress semantic collapse and reward hacking, while online RL systematically enforces physical commonsense and fine-grained visual fidelity. On the nuScenes benchmark, WAM-Sim achieves a superior generation fidelity (FID=3.47, FVD=25.72) and strong downstream perception performance (24.16 mAP, 71.23 road mIoU). On minute-scale rollouts of self-collected data, WAM-Sim substantially mitigates structural and identity drift, preserving high scene similarity and subject identity coherence across extended horizons. Project page: https://fudan-generative-vision.github.io/WAM-Sim.

## 1 INTRODUCTION

Video world models offer a scalable paradigm for controllable closed-loop simulation in autonomous driving, synthesizing driving observations conditioned on ego trajectories and agent controls (Hu et al., 2023; Gao et al., 2024c; Wen et al., 2024; Russell et al., 2025). Despite recent progress, standard training and evaluation regimes remain predominantly constrained to short temporal clips and instantaneous point-estimate metrics. As rollouts extend toward minute-scale horizons, generation systematically degrades through three recurring failure modes: 1) structural drift, where road geometry and background appearance detach from geographic reality; 2) identity drift, where traffic participants undergo unconstrained appearance, category, or tracking mutations; and 3) quality decay, where accumulated autoregressive errors recursively compound over time. Although recent architectures incorporate extended context windows, temporal filtering, or instance masking (Guo et al., 2024; Huang et al., 2025b; Liu et al., 2026; Cheng et al., 2026; Basant et al., 2026), these ca pabilities operate in isolation without a persistent and addressable state. We argue that long-horizon generation failure is not merely a function of limited context window length, but fundamentally demands persistent, multi-scale memory semantics to anchor identity, appearance and geometry.

In conventional driving world models, conditioning signals such as high-definition (HD) maps, 3D bounding boxes, and reference frames are provided as transient per-frame or per-chunk inputs.

![](images/7d90a8c58824b61e19c3b6edbf58dc1431ba317915e74baefcde9e8d796d57e9.jpg)  
Figure 1: Overview of core capabilities in WAM-Sim. World anchored memory (top-left) anchors long-horizon rollouts, maintaining background consistency and persistent instance identities (e.g., tracked pedestrians, motor vehicles, and non-motorized vehicles) across extended horizons. Online RL Post-training (bottom-left) restores visual fidelity and enforces physical commonsense, synthesizing natural agent dynamics (e.g., pedestrian gait) and realistic vehicle motion. Furthermore, WAM-Sim generalizes across diverse weather conditions and environmental scenes (right).

While transient conditioning constrains immediate frame synthesis, it fails to establish a persistent memory state across extended rollouts. Recent retrieval-augmented models and object-centric attention mechanisms demonstrate the value of persistent scene context and isolated participant conditioning (Wu et al., 2025b; Xiao et al., 2025; Li et al., 2025; Jia et al., 2025; Seo et al., 2026; Yang & Zhang, 2026). However, existing methods deploy these mechanisms in isolation, failing to jointly preserve global route geometry, geolocated scene appearance, and individual object identities under dynamic mid-rollout interventions. Consequently, once generation extends beyond the initial context window, persistent states must be reconstructed from noisy self-generated histories, causing errors to compound over time. Overcoming this bottleneck requires a unified memory architecture that represents road structure, environmental appearance, and participant identity as distinct visual tokens, maintains each according to its temporal persistence, and retrieves each through mechanisms matched to its spatio-temporal validity domain.

To address this challenge, we present a driving world model anchored by a persistent world anchored memory. The memory organizes the driving world across three temporal and semantic scales: static structural tokens encoded from HD lane maps serve as persistent geometric anchors; read-only scene tokens retrieved from geolocated databases supply background appearance via temporally routed cross-attention; and instance tokens registered in a write-once appearance bank preserve individual object identities across chunk boundaries and occlusions through instance-masked cross-attention. Although this explicit memory substrate substantially mitigates structural and identity drift, injecting such dense, multi-scale conditioning signals inevitably shifts the generator off its pretrained distribution, compromising native visual sharpness and multi-agent physical plausibility. To overcome this trade-off, we introduce Drive DiffusionNFT, an online post-training framework that optimizes self-generated rollouts via flow-matching reinforcement learning guided by hierarchy-aligned visual rewards and a physical judge. Beyond their individual roles, we empirically observe a practical synergy between explicit memory persistence and online policy optimization: the invariant memory state appears to confine the effective policy optimization space, coinciding with reduced semantic drift and reward hacking during recursive rollouts, while online RL systematically restores finegrained visual fidelity and enforces physical commonsense without compromising global stability.

Extensive evaluations on the standard nuScenes benchmark and a self-collected challenging longhorizon dataset demonstrate the effectiveness of WAM-Sim. On nuScenes, WAM-Sim achieves superior generation fidelity (FID = 3.47, FVD = 25.72) alongside strong downstream perception performance (24.16 mAP, 71.23 road mIoU). On minute-scale rollouts of self-collected data, it effectively mitigates quality–horizon decay, preserving high scene similarity (0.910) and instance identity coherence (0.681) across extended temporal gaps. Furthermore, human preference studies and physical evaluations confirm that tightly coupling persistent hierarchical memory with online flow-matching post-training substantially mitigates structural and identity drift, synthesizing realistic and physically sound driving simulations.

![](images/bde885f77ee5f6519897e0fa2b522ce05a4abaa6a0ac170692acdd216c9368d2.jpg)  
Figure 2: Overview of the WAM-Sim architecture and world anchored memory. Persistent context is structured across three temporal and semantic scales: 1) Structure: pose-aligned HD-map tokens injected via spatio-temporal addition to anchor road geometry; 2) Scene: geolocated background tokens retrieved from a database and accessed via temporally routed scene cross-attention; and 3) Instance: object appearance crops stored in a write-once appearance bank and retrieved via instancemasked cross-attention to isolate participant identities. The hierarchical memory state conditions a 3D Diffusion Transformer (DiT) operating on noisy latents and history KV caches to synthesize autoregressive, consistent driving video chunks.

## 2 RELATED WORK

Driving World Models. Driving video generation has evolved from action- and trajectoryconditioned models (Hu et al., 2023; Russell et al., 2025; Gao et al., 2024c) to layout- and mapconditioned video synthesis (Wang et al., 2024a; Wen et al., 2024; Gao et al., 2024b;a; Zhao et al., 2025; Wang et al., 2024b; Li et al., 2024; Mei et al., 2024), autoregressive long-horizon generation (Jia et al., 2023; Hu et al., 2024; Zhang et al., 2025; Guo et al., 2024), geometry-aware 4D modeling (Yang et al., 2025; Jiang et al., 2025; Guo et al., 2025; Chen et al., 2024b), and drivingspecialized foundation models (NVIDIA, 2025a; Ren et al., 2025; Ni et al., 2025; Basant et al., 2026). InstaDrive (Yang et al., 2026) propagates instance features to preserve identity, yet treats such evidence as transient per-chunk conditioning; HorizonDrive (Zhang et al., 2026) extends rollouts toward minute-scale horizons via streaming context. Across these methods, maps, boxes, and retrieved context remain transient, with no persistent, addressable state binding geographic location and instance identities; we replace transient conditioning with a world anchored memory.

Memory and Long-Horizon Consistency. Persistence in interactive world models is achieved by attention caches (Valevski et al., 2025; Decart & Etched, 2024; Parker-Holder et al., 2024; Google DeepMind, 2025), retrieval over past observations (Xiao et al., 2025; Yu et al., 2025; Li et al., 2025; Zhao et al., 2026; Wu et al., 2025b), or geographic grounding (Deng et al., 2024; Wang et al., 2025; Jia et al., 2025). Orthogonal efforts extend or compress context (Qiu et al., 2024; Kim et al., 2024; Henschel et al., 2025; Zhang & Agrawala, 2025; Huang et al., 2025a; Oshima et al., 2025) and mitigate exposure bias (Chen et al., 2024a; Yin et al., 2025; Huang et al., 2025b; Liu et al., 2026). Closest to ours, ConsisDrive (Yang & Zhang, 2026) encodes instance identity and trajectory through instance-masked attention and losses; we adopt its mask to specify where identity applies, while our write-once appearance bank determines what persists across occlusions and chunk boundaries. Spatial Retrieval (Jia et al., 2025) grounds generation geographically, but retains no persistent scene or instance state. Each mechanism addresses persistence in isolation; we unify geometric anchoring, scene appearance, and instance identity within a single memory substrate.

![](images/aa364dd13a529455df62ae3b1e73bf026b2ea58ecb7bd214a152a113d11171f8.jpg)  
Figure 3: Overview of the Drive DiffusionNFT pipeline. (a) Data curation constructs a diverse RL training set with visible traffic participants and strong motion dynamics. (b) Multi-reward evaluation assesses scene quality, object quality, and physical plausibility. (c) NFT optimization normalizes and aggregates the rewards to update the driving world model.

Reinforcement Learning for Visual Generation. Reward alignment of visual diffusion models has advanced from policy gradients and preference optimization (Black et al., 2023; Fan et al., 2023; Wallace et al., 2023; Xu et al., 2023) to online group-relative flow-matching post-training (Liu et al., 2025; Xue et al., 2025; Zheng et al., 2025). Recently, RL post-training has been applied to video world models: RLVR-World (Wu et al., 2025a) leverages verifiable rewards for video generation, while Reward Forcing (Lu et al., 2025) distills reward signals into streaming autoregressive diffusion rollouts. Orthogonal to these approaches, which optimize generic video rewards, WAM-Sim is, to the best of our knowledge, the first to combine forward-process DiffusionNFT optimization with memory-aligned hierarchical driving rewards—unifying multi-scale visual fidelity with VLMguided physical plausibility—to constrain policy search and mitigate structural and quality decay on self-generated multi-chunk rollouts.

## 3 METHOD

As shown in Figure 2, we present a long-horizon driving world model (Section 3.1) built on two mutually reinforcing components: a persistent world anchored memory that anchors generation to a real geographic location and to per-instance identities (Section 3.2), and Drive DiffusionNFT, an online post-training stage that optimizes visual and physical realism on self-generated rollouts (Section 3.3). The two components are synergistic by design: the memory state delimits the manifold within which reward optimization operates, while the rewards are aligned with the memory hierarchy, so that the policy improves precisely the properties that memory persists.

## 3.1 PROBLEM FORMULATION

Given a local HD-map $e _ { c } ,$ 3D bounding-box agent controls $a _ { c } ,$ a global ego pose trajectory $P _ { c } ,$ and a persistent memory state $\mathcal { M } _ { c } .$ , the model autoregressively synthesizes driving video chunks $\hat { I } _ { c } \in \overset { \mathbf { \bar { \rho } } } { \mathbb { R } } ^ { V \times T \times H \times W \times 3 }$ of $T$ frames across V camera views:

$$
\hat { I } _ { c } \sim p _ { \theta } \left( \cdot \left| \hat { I } _ { c - 1 } ^ { \mathrm { t a i l } } , e _ { c } , a _ { c } , P _ { c } , { \mathcal M } _ { c } \right. \right) , \qquad { \mathcal M } _ { c + 1 } = \mathrm { U p d a t e } ( { \mathcal M } _ { c } , e _ { c } , a _ { c } , P _ { c } , \hat { I } _ { c } ) .\tag{1}
$$

Here, $a _ { c }$ encodes 3D bounding-box layout sequences for surrounding agents, and $P _ { c }$ provides the global ego trajectory used for geolocated reference retrieval. V denotes the number of camera views. Specifically, $\mathrm { U p d a t e } ( \cdot )$ is an append-only operation: it registers newly emerging traffic instances into the appearance bank upon initial observation and advances temporal segment indices, guaranteeing that existing memory entries are never overwritten or corrupted by self-generated video content. Long-horizon generation satisfies three requirements: 1) no structural $\scriptstyle d r i f t -$ —the video remains anchored to the road geometry and background appearance; 2) no identity drift—each tracked participant preserves its appearance across chunks, occlusion, and re-emergence; 3) no quality decay—visual and physical fidelity do not degrade with horizon.

## 3.2 WORLD ANCHORED MEMORY

A driving world retains information at multiple scales—road topology, geolocated scene context, and traffic-participant identities—which differ in semantics, spatial scope, and temporal persistence. We therefore formalize the world state as a hierarchical memory tuple $\bar { \mathcal { M } } = \left( { \mathcal { M } } ^ { \mathrm { s t r u c t } } , \bar { \mathcal { M } } ^ { \mathrm { s c e n e } } , \mathcal { M } ^ { \mathrm { i n s t } } \right)$ where each tier is maintained under a dedicated persistence mechanism and retrieved through an operator aligned with its spatio-temporal validity domain.

Memory Token Extraction. We extract memory tokens across three distinct semantic and temporal scales. First, structural tokens capture static road topology by projecting the per-frame HD-map sequence into the latent video grid via a dedicated map embedding layer. Second, scene tokens represent persistent background appearance; we partition the rollout of $C _ { \mathrm { t o t a l } }$ frames into $N _ { b } =$ $\lceil \bar { C } _ { \mathrm { t o t a l } } / \bar { T _ { b } } \rceil$ non-overlapping temporal segments of $T _ { b }$ frames (one chunk per segment by default) and, for each segment, retrieve the nearest reference image from the geolocated database using the segment’s global ego pose, subject to a spatial radius $r _ { \mathrm { m a x } }$ and a camera-heading alignment constraint; if no reference satisfies both criteria, the segment reuses the reference of its preceding segment. Each reference image is VAE-encoded into L patch tokens $S _ { n } \in \mathbb { R } ^ { L \times D }$ , forming the unified scene memory matrix $S = { \mathrm { C o n c a t } } _ { n = 1 } ^ { N _ { b } } S _ { n } \in \mathbb { R } ^ { N _ { b } L \times D }$ . Third, instance tokens encode object identity. For each tracked instance, after filtering observations by the bounding-box area range $[ \rho _ { \mathrm { m i n } } , \rho _ { \mathrm { m a x } } ]$ , we divide the remaining frames into at most $K _ { f }$ temporal groups and select the largest observation from each. The selected crops are VAE-encoded and projected into $B _ { i } \in \mathbb { R } ^ { K \times D }$ , where $K = K _ { f } K _ { p }$ and $K _ { p }$ is the number of tokens per crop. A validity vector $\nu _ { i }$ masks padded slots.

Hierarchical Persistence. The three memory tiers operate under persistence mechanisms strictly aligned with their temporal dynamics. Structural memory is trajectory-aligned: the global road topology remains invariant, while its per-frame HD-map representation dynamically transforms relative to the ego pose. Scene memory is geographically grounded and read-only: geolocated references in $\mathcal { M } ^ { \mathrm { s c e n e } }$ remain static throughout generation and cannot be overwritten by self-generated content, thereby anchoring environmental textures, architecture, and road surfaces to real-world coordinates. Instance memory follows a write-once policy: upon an agent’s initial appearance, its visual tokens $B _ { i }$ are cached in a track-keyed bank $\mathcal { \bar { M } } ^ { \mathrm { i n s t } } = \dot { \{ \tau _ { i } \mapsto B _ { i } \} }$ indexed by non-parametric track identifiers $\tau _ { i }$ . Locking $B _ { i }$ against subsequent overwrites prevents autoregressive error accumulation from corrupting object identities, substantially improving re-identification across extended occlusions and chunk boundaries.

Hierarchical Memory Injection. Rather than exposing all memory tokens to every video token, our hierarchical routing progressively conditions a shared video representation with geometry-, environment-, and identity-level information at their corresponding validity domains.

Structural memory provides trajectory-aligned geometric information. Since the HD-map representation is spatially aligned with the noisy video latent, it is incorporated as an additive condition, $X ^ { ( 0 ) } = Z _ { t } + \mathcal { M } ^ { \mathrm { s t r u c t } }$ , where $Z _ { t }$ denotes the noisy video latent at flow timestep t. This establishes the geometry-aware video representation on which the subsequent memory routing operates. Scene memory operates at the temporal-segment level. For video token $j ,$ let $s ( j )$ denote its corresponding segment. The routing mask is defined as $R _ { j , ( n - 1 ) L + r } ^ { \mathrm { s c e n e } } = 0$ when $s ( j ) = n$ and otherwise −∞, where $r \in \{ 1 , \ldots , L \}$ , ensuring that each token retrieves scene information only from its associated temporal segment. Scene-memory retrieval is performed on the current geometry-conditioned video representation, producing an updated representation that incorporates the corresponding environmental context. Instance memory further specializes this shared representation within object-specific spatio-temporal regions. Given N tracked instances, we rasterize their bounding boxes onto the video-token grid to form an instance-membership matrix $\Pi \in \{ 0 , 1 \} ^ { N _ { v } \times N }$ , where $N _ { v } = C H ^ { \prime } W ^ { \prime }$ denotes the number of video tokens per chunk and $\Pi _ { j , i } = 1$ if and only if token j belongs to instance i. Only foreground tokens covered by tracked instances participate in the instance-memory branch, while background tokens bypass it. Each foreground token interacts only with video tokens belonging to the same instance and retrieves appearance information only from the corresponding entry in the instance memory bank. Cross-instance connections and invalid memory tokens are masked out. Importantly, instance-memory retrieval operates on the video representation already conditioned by both structural and scene information.

The resulting formulation constitutes a coarse-to-fine hierarchical conditioning process over a shared video representation: structural memory establishes trajectory-aligned geometry, scene memory introduces segment-specific environmental context, and instance memory preserves object-specific appearance within tracked regions. Although their validity domains differ, the three memories jointly form a single hierarchical memory system rather than independent parallel branches.

## 3.3 DRIVE DIFFUSIONNFT: SYNERGISTIC ONLINE POST-TRAINING

Although explicit memory persistence mitigates structural and identity drift, incorporating multiscale conditioning introduces distribution shifts that can degrade visual fidelity and physical consistency. To resolve this trade-off, we introduce Drive DiffusionNFT, an online post-training framework optimizing self-generated rollouts via flow-matching reinforcement learning.

Hierarchical Reward Formulation. As shown in Figure 3 (b), our rewards mirror the memory hierarchy and the rollout reward R<sup>i</sup> for candidate i integrates three multi-level criteria:

$$
R ^ { i } = \lambda _ { \mathrm { s c e n e } } \tilde { R } _ { \mathrm { s c e n e } } ^ { i } + \lambda _ { \mathrm { o b j } } \tilde { R } _ { \mathrm { o b j } } ^ { i } + \lambda _ { \mathrm { p h y } } \tilde { R } _ { \mathrm { p h y } } ^ { i } ,\tag{2}
$$

where $\tilde { R } _ { m } ^ { i }$ denotes the group-normalized score of reward component m. Specifically, scenelevel perceptual quality averages the VBench Imaging Quality scores over sampled frames (Huang et al., 2024), and $\begin{array} { r } { R _ { \mathrm { s c e n e } } ^ { i } ~ = ~ \frac { 1 } { | \hat { S } ^ { i } | } \sum _ { f } q _ { \mathrm { s c e n e } } ( \hat { \mathbf { V } } _ { f } ^ { i } ) } \end{array}$ Object-level recognizability is evaluated on valid agent crops using binary classifiers trained on driving scenes from WorldLens (Liang et al., 2025), covering two categories: vehicles and pedestrians. To avoid saturation of sigmoid confidence and retain greater score discrimination, we use normalized pre-sigmoid classifier logits, $\begin{array} { r } { R _ { \mathrm { o b j } } ^ { i } = \frac { 1 } { N _ { \mathrm { o b j } } ^ { i } } \sum _ { k } \frac { \bar { 1 } } { | \mathcal { C } _ { k } ^ { i } | } \sum _ { o \in \mathcal { C } _ { k } ^ { i } } \tilde { \ell } _ { \mathrm { c l s } } ( o ) } \end{array}$ , where $\mathcal { C } _ { k } ^ { i }$ denotes the valid crops of object $k ,$ and $\tilde { \ell } _ { \mathrm { c l s } } ( o )$ is the normalized logit from the corresponding class-specific classifier. Crops below a visible-area threshold are excluded. To explicitly enhance the physical realism of generated driving videos, we apply SoliReward (Lian et al., 2026) to assess the general physical plausibility and driving scenario subject deformity, and thereby capturing physically implausible interactions and dynamics in driving scenes, as well as structural or temporal artifacts of traffic participants. For each generated video, the physical reward is given directly by its scalar reward score, $R _ { \mathrm { p h y } } ^ { i } = s _ { \mathrm { S o l i } } ( \hat { \mathbf { V } } ^ { i } )$

Flow-Matching Policy Optimization. Following DiffusionNFT (Zheng et al., 2025), we perform reward-guided online optimization directly over self-generated rollouts. For each condition c, a group of $G$ candidate rollouts is sampled from the collection policy $\pi ^ { \mathrm { o l d } }$ with the corresponding clean latents denoted by $\mathbf { x } _ { \mathrm { 0 } }$ . For a noisy state ${ \bf x } _ { t } = \alpha _ { t } { \bf x } _ { 0 } + \sigma _ { t } \epsilon$ with target velocity $\mathbf { v } = \dot { \alpha } _ { t } \mathbf { x } _ { 0 } + \dot { \sigma } _ { t } \pmb { \epsilon }$ $( \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) )$ , the flow-matching objective updates policy velocity field $\mathbf { v } _ { \theta }$ via:

$$
{ \mathcal { L } } _ { \mathrm { N F T } } ( \theta ) = \mathbb { E } _ { \mathbf { c } , \mathbf { x } _ { 0 } \sim \pi ^ { \mathrm { o l d } } , t } \left[ r \left\| \mathbf { v } _ { \theta } ^ { + } - \mathbf { v } \right\| _ { 2 } ^ { 2 } + \left( 1 - r \right) \left\| \mathbf { v } _ { \theta } ^ { - } - \mathbf { v } \right\| _ { 2 } ^ { 2 } \right] ,\tag{3}
$$

where $\mathbf { v } _ { \theta } ^ { + }$ and $\mathbf { v } _ { \theta } ^ { - }$ represent positive and negative implicit velocity predictions, and optimality weight $r = \sigma ( R ^ { i } )$ converts group-relative rollout rewards into relative gradient updates.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUPS

Implementation Details. Our generator is a 3D DiT initialized from pretrained OmniDreams weights and trained in two stages: World-Anchored Memory pre-training and Drive DiffusionNFT post-training. Stage I sequentially optimizes trajectory-aligned structural conditioning, scene memory, and instance memory for 130k, 15k, and 5k steps, respectively. All use batch size 8 and context parallelism degree 8, with learning rates of $1 \times 1 0 ^ { - 5 } , 3 \times 1 0 ^ { - 5 }$ , and $3 \times 1 0 ^ { - 4 }$ . Starting from the full Stage I checkpoint, Stage II applies online flow-matching RL to self-generated rollouts using $G = 1 \bar { 6 }$ and LoRA adapters (rank 256, $\mathrm { L R } 1 \times 1 0 ^ { - 5 } )$ . We use AdamW throughout and train all experiments on 32 Ascend 910B NPUs.

Table 1: Comparisons of controllable driving scene generation on the nuScenes dataset, grouped by generation paradigm. FID is computed over keyframes corresponding to all nuScenes validation.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Synthesis Resolution</td><td colspan="2">Fidelity</td><td colspan="2">3D Detection</td><td colspan="2">BEV Seg mIoU</td></tr><tr><td>FVD↓</td><td>FID↓</td><td>mAP↑</td><td>NDS ↑</td><td>Road ↑</td><td>Vehicle ↑</td></tr><tr><td>Ori nuScenes 一</td><td> $9 0 0 \times 1 6 0 0$ </td><td></td><td></td><td>35.54</td><td>41.21</td><td>73.66</td><td>34.82</td></tr><tr><td colspan="8">(I) Image Generation Methods</td></tr><tr><td>BEVControl</td><td> $2 2 4 \times 4 0 0$ </td><td></td><td>24.85</td><td></td><td></td><td>60.80</td><td>26.80</td></tr><tr><td>PerLDiff</td><td> $2 5 6 \times 3 8 4$ </td><td></td><td>13.36</td><td>15.24</td><td>24.05</td><td>61.26</td><td>27.13</td></tr><tr><td>DrivePTS</td><td> $2 2 4 \times 4 8 0$ </td><td></td><td>11.45</td><td>15.37</td><td>25.49</td><td>63.95</td><td>27.82</td></tr><tr><td>MagicDrive</td><td> $2 2 4 \times 4 0 0$ </td><td></td><td>16.20</td><td>12.30</td><td>23.32</td><td>61.05</td><td>27.01</td></tr><tr><td colspan="8">(II) Video Generation Methods</td></tr><tr><td>DriveDreamer</td><td> $2 5 6 \times 4 4 8$ </td><td>452.0</td><td>26.80</td><td></td><td></td><td></td><td></td></tr><tr><td>DriveDreamer-2</td><td> $2 5 6 \times 4 4 8$ </td><td>105.10</td><td>25.00</td><td></td><td></td><td></td><td></td></tr><tr><td>DrivingDiffusion</td><td> $5 1 2 \times 5 1 2$ </td><td>332.0</td><td>15.83</td><td></td><td>33.10</td><td>63.20</td><td>31.60</td></tr><tr><td>Panacea</td><td> $2 5 6 \times 5 1 2$ </td><td>139.0</td><td>16.96</td><td>11.58</td><td>22.31</td><td>55.78</td><td>22.74</td></tr><tr><td>DreamForge</td><td> $2 2 4 \times 4 0 0$ </td><td>224.8</td><td>14.61</td><td>13.01</td><td>22.16</td><td>65.27</td><td>28.36</td></tr><tr><td>UniMLVG</td><td> $2 2 4 \times 4 0 0$ </td><td>36.10</td><td>5.80</td><td>22.50</td><td></td><td>70.81</td><td>29.12</td></tr><tr><td>MagicDrive-V2</td><td> $8 4 8 \times 1 6 0 0$ </td><td>94.84</td><td>20.91</td><td>17.65</td><td></td><td>59.79</td><td>32.73</td></tr><tr><td>Ours</td><td> $7 0 4 \times 1 2 8 0$ </td><td>25.72</td><td>3.47</td><td>24.16</td><td>32.10</td><td>71.23</td><td>32.97</td></tr></table>

Table 2: Evaluation on the Self-Collected Dataset. All methods generate videos conditioned on ego trajectories and agent bounding boxes.
<table><tr><td rowspan="2">Method</td><td colspan="2">Visual Fidelity</td><td colspan="3">VBench Metrics</td><td colspan="2">Persistence</td><td colspan="2">Object &amp; Physics</td></tr><tr><td>FID↓</td><td>FVD↓</td><td>Quality ↑</td><td>Subject ↑</td><td>Background ↑</td><td>Sub. Coh. ↑</td><td>Scene Sim. ↑</td><td>Sub. Fid. ↑</td><td>Phys. Plaus. ↑</td></tr><tr><td colspan="10">(I) Inference Only</td></tr><tr><td>UniMLVG</td><td>44.21</td><td>753.16</td><td>75.75</td><td>83.43</td><td>89.77</td><td>0.591</td><td>0.694</td><td>39.19</td><td>46.57</td></tr><tr><td>OpenDWM</td><td>37.69</td><td>535.42</td><td>76.54</td><td>82.36</td><td>90.11</td><td>0.532</td><td>0.714</td><td>37.24</td><td>47.44</td></tr><tr><td>HorizonDrive</td><td>24.86</td><td>218.51</td><td>78.41</td><td>84.55</td><td>88.09</td><td>0.553</td><td>0.786</td><td>42.98</td><td>51.19</td></tr><tr><td colspan="10">(II) SFT on Self-Collected Data</td></tr><tr><td>OmniDreams</td><td>11.58</td><td>81.64</td><td>77.63</td><td>86.17</td><td>92.35</td><td>0.646</td><td>0.852</td><td>37.64</td><td>47.99</td></tr><tr><td>Ours</td><td>10.95</td><td>74.20</td><td>78.71</td><td>86.98</td><td>92.54</td><td>0.681</td><td>0.910</td><td>44.00</td><td>55.96</td></tr></table>

Datasets. We train and evaluate WAM-Sim on two dataset regimes. For standard benchmarking, we evaluate on nuScenes, which provides 850 20-second driving scenes (700 for training, 150 for validation) with synchronized multi-view video, 3D bounding boxes, and HD-map annotations. To evaluate long-horizon stability, we train our model on a self-collected dataset comprising 121.8K multi-modal clips (∼1,000 hours) across diverse Chinese urban, rural, and highway environments. Each training sample includes 10 FPS front-view video (704×1280), 3D bounding boxes with persistent track IDs, rendered HD-map topologies, GPS coordinates linked to a geolocated visual refer ence database, and structured captions. The corresponding benchmark evaluation set comprises 350 curated long-horizon clips featuring adverse weather, complex maneuvers, vulnerable road users, and extended occlusion–re-emergence events (representative scenarios shown in Fig. 6).

Evaluation Metrics. We evaluate generation across four core dimensions: 1) visual fidelity via FID, FVD, and VBench Quality Score; 2) long-horizon consistency via Subject Coherence (instance identity) and Scene Similarity (background stability); 3) object & physical realism via detectiongated Subject Fidelity and VLM-guided Physical Plausibility; and 4) downstream perception on nuScenes via 3D detection (mAP, NDS) and BEV segmentation mIoU (road, vehicle).

## 4.2 MAIN BENCHMARK COMPARISONS

nuScenes Benchmark. Table 1 evaluates multi-view generation fidelity and downstream perception consistency on nuScenes. Lacking geolocated reference databases, WAM-Sim operates without scene memory in this setting. At 704×1280 synthesis resolution, WAM-Sim achieves a visual fidelity $( \mathrm { F I D } \stackrel { - } { = } 3 . 4 7 , \mathrm { F V D } \stackrel { - } { = } 2 5 . 7 2 )$ , outperforming the representative image and video generation paradigms. Furthermore, this high-fidelity synthesis translates into robust geometric control, yielding competitive downstream 3D detection (24.16 mAP) and state-of-the-art BEV road segmentation (71.23 mIoU).

Table 3: World memory component ablation on the Self-Collected Dataset. Scene and Instance Memory improve generation consistency.
<table><tr><td>Scene</td><td>Instance</td><td>Sub. Coh. ↑</td><td>Scene Sim. ↑</td><td>FVD↓</td></tr><tr><td></td><td></td><td>0.646</td><td>0.852</td><td>81.64</td></tr><tr><td>√</td><td>一</td><td>0.659</td><td>0.904</td><td>76.68</td></tr><tr><td></td><td>√</td><td>0.680</td><td>0.875</td><td>85.55</td></tr><tr><td>√</td><td>√</td><td>0.681</td><td>0.910</td><td>74.20</td></tr></table>

Table 4: Robustness to scene-memory dropout. Each reference is independently dropped at inference with probability p.

<table><tr><td>Drop Prob. p</td><td>Scene Sim. ↑</td><td>FVD↓</td></tr><tr><td>0.0</td><td>0.910</td><td>74.20</td></tr><tr><td>0.1</td><td>0.908</td><td>77.03</td></tr><tr><td>0.3</td><td>0.904</td><td>78.60</td></tr></table>

1s  
2s  
4s  
![](images/ada40d858555b1ef46ca948c005380ac2a5be20b4e62a99bcf203933b8ca3306.jpg)  
8s

1s  
2s  
4s  
8s  
![](images/759279b95cc1181dbe7e40010e75379aed407d3c0debd1179b8db6970afccfcf.jpg)  
Figure 4: Qualitative comparison results on nuScenes video generation.

![](images/57b2d1ad4388e7728bf403a24e0b56fb8e4c6967b9c61dac336b54fe3b6b3069.jpg)  
Figure 5: Qualitative comparison results on the selfcollected dataset.

![](images/b053b31648a6887470287fcd58b610c31948c9e85908208677dfb1f5115675d5.jpg)  
Figure 6: Diverse generation results on our self-collected dataset.

Self-Collected Dataset. As shown in Table 2, Figure 5 and Figure 6, by anchoring rollouts with persistent world memory, WAM-Sim effectively mitigates quality–horizon decay, outperforming other baselines with 10.95 FID and 74.20 FVD while securing superior background stability (0.910 Scene Similarity) and identity retention (0.681 Subject Coherence). Furthermore, Drive DiffusionNFT online post-training successfully enforces physical commonsense and object realism, yielding peak VBench Quality (78.71), Subject Fidelity (44.00), and Physical Plausibility (55.96).

## 4.3 ABLATION AND DISCUSSION

Architecture Component Ablation. Table 3 disentangles the complementary roles of Scene Memory and Instance Memory on the self-collected dataset. Individually, Scene Memory anchors global environmental context, boosting Scene Similarity from 0.852 to 0.904 and reducing FVD from 81.64 to 76.68 over the memory-free baseline. Conversely, Instance Memory strengthens fine-grained identity persistence, elevating Subject Coherence to 0.680, though its standalone FVD degrades to 85.55 due to a lack of global structural bounds. Jointly integrating both tiers achieves optimal performance—yielding the lowest FVD, peak Subject Coherence, and Scene Similarity.

Scene Memory. We evaluate the robustness and appearance-layout disentanglement of scene memory. Quantitatively, introducing reference dropout at inference demonstrates both active reliance and strong fault tolerance (Table 4); even at $p = 0 . 3$ , Scene Similarity decays marginally from 0.910 to 0.904. Qualitatively, Figure 7(left) demonstrates that scene memory effectively disentangles persistent spatial layouts from transient environmental conditions: when conditioned on a dusk reference image (“Background”), WAM-Sim faithfully preserves the underlying architectural geometry and road layout throughout target daytime rollouts (“Ours”). Conversely, omitting scene memory $\mathrm { ( ^ { 6 6 } w / o ^ { 7 } ) }$ leads to severe structural drift and hallucinated background geometry.

![](images/26106d8f9259f1e4224b3d8138dbb83f7b6abf7327b6d82c9dc1059d37c6021b.jpg)

![](images/580c79bedbb4eaf2963d3aa7e9b4ce9c303da235d79356ce90b05fccf2366fac.jpg)  
Figure 7: Qualitative comparison of scene- and instance-level persistence. WAM-Sim better preserves background appearance and object identity over long rollouts.

![](images/1c252028baa7a8747bba8d4a56838cf948a782fafd48d2a74bcb0fd86ae6bf3b.jpg)

![](images/6029a12ac8dae91a99e963437c7449f1e02a0aa9b2341fe30ebe9dd90790413a.jpg)  
Figure 8: Qualitative ablation of online RL. Left: comparison between the world-memory model without RL and the final model. Right: cumulative reward ablation on the same scenario, progressively adding $R _ { \mathrm { s c e n e } } , R _ { \mathrm { o b j } }$ , and $R _ { \mathrm { p h y } }$

Table 5: Comparison of instance identity preservation on the test set.
<table><tr><td rowspan="2">Method</td><td colspan="3">一 Long-range Coherence</td><td colspan="4">Re-emergence</td><td rowspan="2">Avg.↑</td></tr><tr><td>SC@1S↑</td><td>SC@2S↑</td><td>SC@3S↑</td><td>Sim.↑</td><td>Pass Rate↑</td><td>Occ-ID@1S↑</td><td>Occ-ID@2S↑</td></tr><tr><td>Baseline (SFT)</td><td>0.605</td><td>0.579</td><td>0.546</td><td>0.780</td><td>0.461</td><td>0.781</td><td>0.743</td><td>0.642</td></tr><tr><td>ConsisDrive</td><td>0.634</td><td>0.604</td><td>0.597</td><td>0.805</td><td>0.510</td><td>0.799</td><td>0.812</td><td>0.680</td></tr><tr><td>Ours</td><td>0.653</td><td>0.620</td><td>0.613</td><td>0.810</td><td>0.535</td><td>0.804</td><td>0.833</td><td>0.695</td></tr></table>

Table 6: Ablation of reward components in Drive DiffusionNFT post-training.
<table><tr><td> $R _ { \mathrm { s c e n e } }$ </td><td> $R _ { \mathrm { o b j } }$ </td><td> $R _ { \mathrm { p h y } }$ </td><td> $\mathbf { I m g . } \left( R _ { \mathrm { s c e n e } } \right) \uparrow$ </td><td> $\mathbf { C l a s s i . } ( R _ { \mathrm { o b j } } ) \uparrow$ </td><td> $\mathbf { P h y s . } \left( R _ { \mathrm { p h y } } \right) \uparrow$ </td><td>Sub. Fid. ↑</td><td>Phys. Plaus. ↑</td></tr><tr><td></td><td>一</td><td>一</td><td>59.55</td><td>-0.012</td><td>3.770</td><td>41.74</td><td>50.91</td></tr><tr><td>√</td><td>1</td><td>一</td><td>62.68</td><td>0.064</td><td>3.727</td><td>42.25</td><td>51.13</td></tr><tr><td>√</td><td>√</td><td>一</td><td>61.59</td><td>0.273</td><td>3.789</td><td>43.97</td><td>53.54</td></tr><tr><td>√</td><td>√</td><td>√</td><td>61.48</td><td>0.271</td><td>4.052</td><td>44.00</td><td>55.96</td></tr></table>

Instance Memory. We evaluate the capacity of write-once instance memory to maintain participant identity across extended horizons and occlusion–re-emergence events (Appendix E.3). We benchmark against the SFT baseline and ConsisDrive, which incorporates instance-level masking without a persistent memory state. In Table 5, while ConsisDrive increases average identity preservation from 0.642 to 0.680 over SFT via spatial masking alone, WAM-Sim surpasses both methods across all metrics. Because WAM-Sim and ConsisDrive share the same masking backbone, this performance gain isolates the effect of our write-once appearance bank, demonstrating that persistent memory prevents autoregressive identity drift across chunk boundaries. Qualitatively (Fig. 7, right), when an agent re-emerges post-occlusion, SFT suffers appearance mutation, whereas WAM-Sim recovers its original visual identity by retrieving stored track-keyed features.

Drive DiffusionNFT. Table 6 validates the hierarchical reward design in Drive Diffusion-NFT. Incrementally incorporating $R _ { \mathrm { s c e n e } } , ~ R _ { \mathrm { o b j } }$ , and $R _ { \mathrm { p h y } }$ systematically boosts all optimization signals—improving VBench Imaging Quality $( 5 9 . 5 5 \ \overset { } {  } \ 6 2 . 6 8 )$ , Object Classification score $( - 0 . 0 \bar { 1 2 }  0 . 2 \bar { 7 1 } )$ , and SoliReward Physics score $( 3 . 7 7 0  4 . 0 5 2 )$ . Crucially, these gains directly transfer to held-out evaluation metrics, elevating Subject Fidelity $( 4 1 . 7 4  4 4 . 0 0 )$ and Physical Plausibility $( 5 0 . 9 1  5 5 . 9 6 )$ . This shows that multi-level reward alignment enhances physical realism and instance fidelity without collapsing into reward hacking. Figure 8 (right) qualitatively demonstrates the cumulative visual and physical gains across each reward tier.

Synergy Between Memory Bounds and Online RL. As shown in Figure 8 (left), explicit memory persistence and online policy optimization are mutually reinforcing: world anchored memory bounds the policy optimization manifold to prevent semantic drift across long rollouts, while Drive DiffusionNFT refines fine-grained visual sharpness and enforces physically plausible agent motions within this persistent world scaffold.

## 5 CONCLUSION

We presented WAM-Sim, a long-horizon driving world model that systematically mitigates structural drift, identity mutation, and quality decay. By unifying a persistent world anchored memory—anchoring road topology, geolocated appearance, and instance identity—with Drive DiffusionNFT online post-training, WAM-Sim establishes a persistent world state while enforcing visual sharpness and physical plausibility. Benchmark evaluations on nuScenes and self-collected driving datasets demonstrate superior generation fidelity, robust downstream perception, and temporal stability over extended rollouts. This synergy between invariant memory manifolds and reward-guided policy optimization offers a promising paradigm toward closed-loop simulation.

## REFERENCES

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. VideoPhy: Evaluating physical commonsense for video generation. arXiv preprint arXiv:2406.03520, 2024.

Aarti Basant, Amlan Kar, Despoina Paschalidou, Fangyin Wei, Francesco Ferroni, Guillermo Garcia Cobo, Haithem Turki, Huan Ling, Jaewoo Seo, James Lucas, Jay Zhangjie Wu, Jialiang Wang, Jonathan Lorraine, Jun Gao, Kai He, Katarina Tothova, Kevin Xie, Michał Tyszkiewicz, Qi Wu, Riccardo de Lutio, Ruilong Li, Sanja Fidler, Seung Wook Kim, Tianchang Shen, Tianshi Cao, Tobias Pfaff, William Lew, Xindi Wu, Xuanchi Ren, Yifan Lu, Yuxuan Zhang, Zan Gojcic, and Zian Wang. NVIDIA OmniDreams: Real-time generative world model for closed-loop autonomous vehicle simulation. arXiv preprint arXiv:2606.03159, 2026.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

Boyuan Chen, Diego Mart´ı Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitz-´ mann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In Advances in Neural Information Processing Systems, 2024a.

Guibin Chen, Dixuan Lin, Jiangping Yang, et al. SkyReels-V2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.

Rui Chen, Zehuan Wu, Yichen Liu, Yuxin Guo, Jingcheng Ni, Haifeng Xia, and Siyu Xia. UniM-LVG: Unified framework for multi-view long video generation with comprehensive control capabilities for autonomous driving. arXiv preprint arXiv:2412.04842, 2024b.

Ke Cheng, Hanqiao Ye, Lei Shi, Yahui Liu, Yunhan Shen, Jingtao Dong, Zhenke Wang, Wenxuan Ao, Weixiang Xu, Kaining Huang, and Shuhan Shen. M<sup>4</sup>World: A multi-view multimodal driving world model for interactive object manipulation and minute-long streaming. arXiv preprint arXiv:2607.14005, 2026.

Decart and Etched. Oasis: A universe in a transformer, 2024. Project page.

Boyang Deng, Richard Tucker, Zhengqi Li, Leonidas Guibas, Noah Snavely, and Gordon Wetzstein. Streetscapes: Large-scale consistent street view generation using autoregressive video diffusion. In ACM SIGGRAPH Conference Papers, 2024. doi: 10.1145/3641519.3657513.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK: Reinforcement learning for fine-tuning text-to-image diffusion models. In Advances in Neural Information Processing Systems, 2023.

Ruiyuan Gao, Kai Chen, Bo Xiao, Lanqing Hong, Zhenguo Li, and Qiang Xu. MagicDriveDiT: High-resolution long video generation for autonomous driving with adaptive control. arXiv preprint arXiv:2411.13807, 2024a.

Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. MagicDrive: Street view generation with diverse 3D geometry control. In International Conference on Learning Representations, 2024b.

Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. In Advances in Neural Information Processing Systems, 2024c.

Google DeepMind. Genie 3: A new frontier for world models, 2025. Google DeepMind technical report.

Jiazhe Guo, Yikang Ding, Xiwu Chen, Shuo Chen, Bohan Li, Yingshuang Zou, Xiaoyang Lyu, Feiyang Tan, Xiaojuan Qi, Zhiheng Li, and Hao Zhao. DiST-4D: Disentangled spatiotemporal diffusion with metric depth for 4D driving scene generation. arXiv preprint arXiv:2503.15208, 2025.

Xi Guo, Chenjing Ding, Haoxuan Dou, Xin Zhang, Weixuan Tang, and Wei Wu. InfinityDrive: Breaking time limits in driving world models. arXiv preprint arXiv:2412.01522, 2024.

Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. StreamingT2V: Consistent, dynamic, and extendable long video generation from text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. GAIA-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

Xiaotao Hu, Wei Yin, Mingkai Jia, Junyuan Deng, Xiaoyang Guo, Qian Zhang, Xiaoxiao Long, and Ping Tan. DrivingWorld: Constructing world model for autonomous driving via video GPT. arXiv preprint arXiv:2412.19505, 2024.

Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on Minecraft. arXiv preprint arXiv:2510.03198, 2025a.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. In Advances in Neural Information Processing Systems, 2025b.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Fan Jia, Weixin Mao, Yingfei Liu, Yucheng Zhao, Yuqing Wen, Chi Zhang, Xiangyu Zhang, and Tiancai Wang. ADriver-I: A general world model for autonomous driving. arXiv preprint arXiv:2311.13549, 2023.

Xiaosong Jia, Chenhe Zhang, Yule Jiang, Songbur Wong, et al. Spatial retrieval augmented autonomous driving. arXiv preprint arXiv:2512.06865, 2025.

Junpeng Jiang, Gangyi Hong, Miao Zhang, Hengtong Hu, Kun Zhan, Rui Shao, and Liqiang Nie. DiVE: Efficient multi-view driving scenes generation based on video diffusion transformer. arXiv preprint arXiv:2504.19614, 2025.

Wooyeol Kim, Junho Park, Janghyeon Lee, Wonjun Hwang, Hyunwoo J. Kim, and Kyungmin Jo. FIFO-diffusion: Generating infinite videos from text without training. In Proceedings of the International Conference on Machine Learning, 2024.

Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. VMem: Consistent interactive video scene generation with surfel-indexed view memory. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. DrivingDiffusion: Layout-guided multi-view driving scene video generation with latent diffusion model. In European Conference on Computer Vision, 2024.

Jiesong Lian, Ruizhe Zhong, Zixiang Zhou, Xiaoyue Mi, Yixue Hao, Yuan Zhou, Qinglin Lu, Long Hu, and Junchi Yan. SoliReward: Mitigating susceptibility to reward hacking and annotation noise in video generation reward models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12798–12807, 2026.

Aoyu Liang, Lingdong Kong, Tong Yan, Hao Liu, Yang Yang, et al. WorldLens: Full-spectrum evaluations of driving world models in real world. arXiv preprint arXiv:2512.10958, 2025.

Wang Lin, Liyu Jia, Wentao Hu, Kaihang Pan, Zhongqi Yue, Wei Zhao, Jingyuan Chen, Fei Wu, and Hanwang Zhang. Reasoning physical video generation with diffusion timestep tokens via reinforcement learning. arXiv preprint arXiv:2504.15932, 2025.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-GRPO: Training flow matching models via online RL. In Advances in Neural Information Processing Systems, 2025.

Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. In International Conference on Learning Representations, 2026.

Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678, 2025.

Jianbiao Mei, Tao Hu, Xuemeng Yang, Licheng Wen, Yu Yang, Tiantian Wei, Yukai Ma, Min Dou, Botian Shi, and Yong Liu. DreamForge: Motion-aware autoregressive video generation for multiview driving scenes. arXiv preprint arXiv:2409.04003, 2024.

Jingcheng Ni, Yuxin Guo, Yichen Liu, Rui Chen, Lewei Lu, and Zehuan Wu. OpenDWM: Open driving world models, 2025. Open-source codebase, SenseTime.

NVIDIA. Cosmos world foundation model platform for physical AI. arXiv preprint arXiv:2501.03575, 2025a.

NVIDIA. Cosmos-reason1: From physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558, 2025b.

Yuta Oshima, Yusuke Iwasawa, Masahiro Suzuki, Yutaka Matsuo, and Hiroki Furuta. WorldPack: Compressed memory improves spatial consistency in video world modeling. arXiv preprint arXiv:2512.02473, 2025.

Jack Parker-Holder, Philip Ball, Jake Bruce, et al. Genie 2: A large-scale foundation world model, 2024. Google DeepMind blog post.

Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. In Proceedings of the International Conference on Machine Learning, 2024.

Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. FreeNoise: Tuning-free longer video diffusion via noise rescheduling. In International Conference on Learning Representations, 2024.

Xuanchi Ren, Yifan Lu, Tianshi Cao, Ruiyuan Gao, Shengyu Huang, Amirmojtaba Sabour, Tianchang Shen, Tobias Pfaff, Jay Zhangjie Wu, Runjian Chen, Seung Wook Kim, Jun Gao, Laura Leal-Taixe, Mike Chen, Sanja Fidler, and Huan Ling. Cosmos-drive-dreams: Scalable synthetic´ driving data generation with world foundation models. arXiv preprint arXiv:2506.09042, 2025.

Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. GAIA-2: A controllable multi-view generative world model for autonomous driving. arXiv preprint arXiv:2503.20523, 2025.

Junyoung Seo, Hyunwook Choi, Minkyung Kwon, Jinhyeok Choi, Siyoon Jin, Gayoung Lee, Junho Kim, JoungBin Lee, Geonmo Gu, Dongyoon Han, Sangdoo Yun, Seungryong Kim, and Jin-Hwa Kim. Grounding world simulation models in a real-world metropolis. arXiv preprint arXiv:2603.15583, 2026.

Liao Shen, Wentao Jiang, Yiran Zhu, Jiahe Li, Tiezheng Ge, Zhiguo Cao, and Bo Zheng. Identitypreserving image-to-video generation via reward-guided optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 27281–27290, 2026.

Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. In International Conference on Learning Representations, 2025.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. arXiv preprint arXiv:2311.12908, 2023.

Jiawei Wang, Haowei Sun, Xintao Yan, Shuo Feng, Jun Gao, and Henry X. Liu. TeraSim-World: Worldwide safety-critical data synthesis for end-to-end autonomous driving. arXiv preprint arXiv:2509.13164, 2025.

Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. DriveDreamer: Towards real-world-driven world models for autonomous driving. In European Conference on Computer Vision, 2024a.

Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024b.

Yuqing Wen, Yucheng Zhao, Yingfei Liu, Fan Jia, Yanhui Wang, Chong Luo, Chi Zhang, Tiancai Wang, Xiaoyan Sun, and Xiangyu Zhang. Panacea: Panoramic and controllable video generation for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Jialong Wu, Shaofeng Yin, Ningya Feng, and Mingsheng Long. RLvr-World: Training world models with reinforcement learning. arXiv preprint arXiv:2505.13934, 2025a.

Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. In Advances in Neural Information Processing Systems, 2025b.

Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. WorldMem: Long-term consistent world simulation with memory. In Advances in Neural Information Processing Systems, 2025.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. ImageReward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems, 2023.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. DanceGRPO: Unleashing GRPO on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Yu Yang, Alan Liang, Jianbiao Mei, Yukai Ma, Yong Liu, and Gim Hee Lee. X-Scene: Large-scale driving scene generation with high fidelity and flexible controllability. In Advances in Neural Information Processing Systems, 2025.

Zhuoran Yang and Yanyong Zhang. ConsisDrive: Identity-preserving driving world models for video generation by instance mask. arXiv preprint arXiv:2602.03213, 2026.

Zhuoran Yang, Xi Guo, Chenjing Ding, Chiyu Wang, Wei Wu, and Yanyong Zhang. InstaDrive: Instance-aware driving world models for realistic and consistent video generation. arXiv preprint arXiv:2602.03242, 2026.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman, and´ Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. In SIGGRAPH Asia Conference Papers, 2025.

Conglang Zhang, Yifan Zhan, Qingjie Wang, Zhanpeng Ouyang, Yu Li, et al. HorizonDrive: Self-corrective autoregressive world model for long-horizon driving simulation. arXiv preprint arXiv:2605.11596, 2026.

Kaiwen Zhang, Zhenyu Tang, Xiaotao Hu, Xingang Pan, Xiaoyang Guo, Yuan Liu, Jingwei Huang, Li Yuan, Qian Zhang, Xiaoxiao Long, Xun Cao, and Wei Yin. Epona: Autoregressive diffusion world model for autonomous driving. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2025.

Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. In Advances in Neural Information Processing Systems, 2025.

Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, and Xingang Wang. DriveDreamer-2: LLM-enhanced world models for diverse driving video generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 10412–10420, 2025.

Jinjing Zhao, Fangyun Wei, Zhening Liu, Hongyang Zhang, Chang Xu, and Yan Lu. Spatia: Video generation with updatable spatial memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4245–4257, 2026.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. DiffusionNFT: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

Yang Zhou, Hao Shao, Letian Wang, Zhuofan Zong, Hongsheng Li, and Steven L. Waslander. DrivingGen: A comprehensive benchmark for generative video world models in autonomous driving. arXiv preprint arXiv:2601.01528, 2026.

## A EXTENDED RELATED WORK

Driving World Models. Controllable driving video generation was pioneered by action-conditioned world models such as GAIA-1 and GAIA-2 (Hu et al., 2023; Russell et al., 2025), high-fidelity trajectory-conditioned prediction in Vista (Gao et al., 2024c), and multi-view generators conditioned on BEV layouts, HD maps, and 3D bounding boxes (Wang et al., 2024a; Wen et al., 2024; Gao et al., 2024b;a; Zhao et al., 2025). Autoregressive token-based approaches extend such models to longer horizons (Jia et al., 2023; Hu et al., 2024; Zhang et al., 2025), while geometry-aware methods improve spatial and 4D consistency (Yang et al., 2025; Jiang et al., 2025; Guo et al., 2025; Chen et al., 2024b). Two recent trends are most relevant to our setting. First, world foundation models specialized for driving—Cosmos and Cosmos-Drive-Dreams (NVIDIA, 2025a; Ren et al., 2025), and most recently OmniDreams (Basant et al., 2026)—achieve photorealistic, interactive, real-time rollouts under world-scenario control, with OmniDreams attaining minute-long closed-loop simulation through a streaming KV cache and self-forcing-style distillation. Second, a line of work explicitly targets long driving sequences: InfinityDrive introduces memory injection and retention for minutescale driving video (Guo et al., 2024). Despite this progress, maps, boxes, and retrieved context are re-specified per chunk and therefore constrain only instantaneous generation; what persists across chunks—an addressable state binding a real location and per-instance identities—remains implicit (e.g., in attention caches) rather than explicit and controllable. We upgrade transient conditioning to a persistent world anchored memory while retaining minute-scale intervenability.

Memory and Long-Horizon Consistency. Interactive world models increasingly rely on explicit memory. Genie 3 exhibits emergent scene memory over minute-scale rollouts but, by its own account, cannot reproduce real-world locations with geographic accuracy (Google DeepMind, 2025). Retrieval-based memories index past observations for consistency: WorldMem maintains a retrievable bank indexed by camera pose (Xiao et al., 2025), VMem indexes past views by surfel geometry (Li et al., 2025), Spatia preserves an updatable 3D point cloud maintained with visual SLAM (Zhao et al., 2026), and video world models with long-term spatial memory support view re-identification over revisits (Wu et al., 2025b). Complementary to these generic mechanisms, geolocated generation grounds synthesis in real places: Streetscapes synthesizes street views of an onthe-fly city rather than actual locations (Deng et al., 2024); TeraSim-World grounds safety-critical data synthesis at arbitrary coordinates but channels real-world appearance through a VLM text bottleneck restricted to static elements (Wang et al., 2025); and, concurrently with our work, Spatial-RAG-AD retrieves geographic imagery along the ego route and injects it as persistent cross-attention context into driving video DiTs (Jia et al., 2025). On the training side, long-horizon methods either extend context windows (FreeNoise (Qiu et al., 2024), FIFO-Diffusion (Kim et al., 2024), StreamingT2V (Henschel et al., 2025)) or close the exposure-bias gap with self-generated rollouts, from Diffusion Forcing (Chen et al., 2024a) and CausVid (Yin et al., 2025) to Self-Forcing (Huang et al., 2025b) and Rolling Forcing (Liu et al., 2026); OmniDreams concurrently adopts self-forcing distillation for real-time driving rollouts (Basant et al., 2026). Instance-level consistency within single clips is pursued by identity-preserving driving generation (Yang & Zhang, 2026) and referenceimage subject consistency methods (Shen et al., 2026). Existing works, however, provide either scene-level memory without real geographic anchors, geographic grounding without a persistent cross-chunk memory state, or instance consistency without occlusion-surviving, write-once identity evidence under counterfactual manipulation. We unify the three: a coordinate-indexed scene memory reused for the entire rollout together with a write-once appearance bank keyed by track identity, retrieved through banked instance-masked attention at chunk boundaries and upon re-emergence after occlusion.

Reinforcement Learning for Visual Generation. RL-based alignment of diffusion and flow models has progressed from policy-gradient fine-tuning of denoising MDPs (DDPO (Black et al., 2023); DPOK and Diffusion-DPO for image alignment (Fan et al., 2023; Wallace et al., 2023); VADER for video (Prabhudesai et al., 2024)), through video-scale preference optimization (SkyReels-V2 (Chen et al., 2025)), to online group-relative policy optimization for flow matching: Flow-GRPO introduced ODE-to-SDE exploration (Liu et al., 2025), DanceGRPO extended it to general visual generation (Xue et al., 2025), Phys-AR optimizes physical velocity-error rewards for video generation (Lin et al., 2025), and RLVR-World trains world models with RL (Wu et al., 2025a); rewardguided optimization has likewise been used for identity-preserving image-to-video generation (Shen et al., 2026). On the evaluation side, physical commonsense benchmarks such as VideoPhy (Bansal et al., 2024), PhysGame, and T2VPhysBench diagnose violations without training, while VLM judges including Cosmos-Reason1 (NVIDIA, 2025b) and VBench-2.0 Motion Rationality establish that vision-language models can score physical plausibility; driving-specific suites such as DrivingGen (Zhou et al., 2026) and WorldLens (Liang et al., 2025) provide comprehensive but point-estimate evaluations on fixed short horizons. To our knowledge, we are the first to combine online flow-matching RL with driving-specific rewards—hierarchical visual fidelity at scene and instance levels together with a VLM-based physical plausibility judge—optimized directly on selfgenerated multi-chunk rollouts, and the first to evaluate long-horizon driving world models with quality–horizon decay curves, counterfactual ID-surgery, and mid-rollout intervention protocols.

## B ADDITIONAL EXPERIMENTS

## B.1 HUMAN EVALUATION

Experimental Setup and Protocol. We evaluate human perceptual preference on 60 representative driving scenarios sampled from the test set. Fifteen independent expert assessors with backgrounds in autonomous driving evaluate WAM-Sim against four baselines: UniMLVG, OpenDWM– CTSD3.5, HorizonDrive, and OmniDreams. To eliminate bias, all method identities are doubleblinded, and video presentation order and spatial placement are randomized. Assessors perform two complementary tasks: (i) forced-choice pairwise preference voting (Win, Tie, or Loss) for side-byside video pairs, and (ii) five-point Mean Opinion Score (MOS) rating across four core dimensions: Visual Fidelity (perceptual quality and temporal coherence), Identity Consistency (long-horizon participant appearance preservation across occlusions), Physical Realism (plausibility of agent dynamics and road interactions), and Location Anchoring (alignment with spatial and geographic context).

Score Aggregation and Inter-Rater Reliability. Pairwise preference judgments are aggregated per scenario via majority voting across the 15 assessors to determine scenario-level outcomes. MOS ratings are averaged per scenario across assessors and subsequently macro-averaged over all 60 scenarios. Inter-rater reliability across the 15 judges is validated using Fleiss’ κ on categorical pairwise preferences, demonstrating strong agreement across all baseline comparisons: $\kappa _ { \mathrm { U n i M L V G } } =$ 0.927, κ<sub>OpenDWM</sub> = 0.864, $\kappa _ { \mathrm { H o r i z o n D r i v e } } = 0 . 9 3 9$ , and $\kappa _ { \mathrm { O m n i D r e a m s } } = 0 . 9 7 2$

Quantitative Results and Statistical Analysis. As summarized in Table 7, WAM-Sim substantially outperforms all baselines in pairwise preference, achieving scenario-level win rates of 98.3% against UniMLVG (59/0/1), 98.3% against OpenDWM–CTSD3.5 (59/0/1), 71.7% against HorizonDrive $( 4 3 / 1 / 1 6 ) $ , and 86.7% against OmniDreams $( 5 2 / 0 / 8 )$ . Furthermore, WAM-Sim secures the highest MOS across all four perceptual dimensions (Fidelity: 4.30, Identity: 4.40, Physics: 4.50, Location: 4.50). To assess statistical significance, we conduct two-sided Wilcoxon signed-rank tests (α = 0.05) on paired scenario-level MOS values (Table 8). WAM-Sim achieves statistically significant gains over UniMLVG, OpenDWM–CTSD3.5, and OmniDreams across all dimensions $( p < 1 0 ^ { - 9 } )$ . Compared to HorizonDrive (the strongest baseline), WAM-Sim yields statistically significant improvements in Identity $( p = 7 . 6 5 \times 1 0 ^ { - 4 } )$ , Physics $( p = 4 . 3 2 \times 1 0 ^ { - 8 } )$ , and Location $( p = 3 . 4 1 { \overset { \cdot } { \times } } 1 0 ^ { - 6 } )$ , confirming that explicit world memory and online RL post-training primarily enhance long-horizon temporal consistency, physical plausibility, and spatial anchoring.

Table 7: Human evaluation on 60 driving scenarios assessed by 15 experts. Pairwise preference reports scenario-level win/tie/loss rates against WAM-Sim. MOS is reported on a 1–5 scale.
<table><tr><td rowspan="2">Method</td><td colspan="3">Pairwise Preference (%)</td><td colspan="4">Mean Opinion Score</td></tr><tr><td>Win</td><td>Tie</td><td>Loss</td><td>Fidelity</td><td>Identity</td><td>Physics</td><td>Location</td></tr><tr><td>UniMLVG</td><td>98.3</td><td>0.0</td><td>1.7</td><td>2.10</td><td>2.20</td><td>1.73</td><td>1.95</td></tr><tr><td>OpenDWM-CTSD3.5</td><td>98.3</td><td>0.0</td><td>1.7</td><td>1.80</td><td>1.80</td><td>1.38</td><td>1.87</td></tr><tr><td>HorizonDrive</td><td>71.7</td><td>1.7</td><td>26.7</td><td>4.10</td><td>4.00</td><td>4.00</td><td>3.90</td></tr><tr><td>OmniDreams</td><td>86.7</td><td>0.0</td><td>13.3</td><td>3.10</td><td>3.00</td><td>2.77</td><td>2.87</td></tr><tr><td>WAM-Sim (Ours)</td><td>一</td><td>一</td><td>一</td><td>4.30</td><td>4.40</td><td>4.50</td><td>4.50</td></tr></table>

Table 8: Statistical significance of scenario-level MOS differences. Two-sided Wilcoxon signed-rank tests compare WAM-Sim with each baseline using 60 paired scenario-level scores per dimension. Values below 0.05 meet the stated significance threshold.
<table><tr><td>Dimension</td><td>UniMLVG</td><td>OpenDWM-CTSD3.5 HorizonDrive OmniDreams</td><td></td><td></td></tr><tr><td>Fidelity</td><td> $7 . 2 9 \times 1 0 ^ { - 1 2 }$ </td><td> $7 . 5 2 \times 1 0 ^ { - 1 2 }$ </td><td> $0 . 0 5 4 6$ </td><td> $2 . 4 0 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Identity</td><td> $7 . 5 8 \times 1 0 ^ { - 1 2 }$ </td><td> $7 . 2 8 \times 1 0 ^ { - 1 2 }$ </td><td> $7 . 6 5 \times 1 0 ^ { - 4 }$ </td><td> $2 . 5 8 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Physics</td><td> $7 . 5 2 \times 1 0 ^ { - 1 2 }$ </td><td> $5 . 9 6 \times 1 0 ^ { - 1 2 }$ </td><td> $4 . 3 2 \times 1 0 ^ { - 8 }$ </td><td> $6 . 4 9 \times 1 0 ^ { - 1 2 }$ </td></tr><tr><td>Location</td><td> $2 . 7 8 \times 1 0 ^ { - 1 2 }$ </td><td> $6 . 2 5 \times 1 0 ^ { - 1 2 }$ </td><td> $3 . 4 1 \times 1 0 ^ { - 6 }$ </td><td> $5 . 1 7 \times 1 0 ^ { - 1 2 }$ </td></tr></table>

Table 9: Efficiency comparison of driving world models during long-horizon rollouts.
<table><tr><td>Method</td><td>Resolution</td><td>Params (B)</td><td>Latency (s) ↓</td><td>Throughput (FPS) ↑</td><td>VRAM (GB) ↓</td></tr><tr><td>Horizondrive</td><td> $3 8 4 \times 7 6 8$ </td><td>1.3</td><td>3.51</td><td>4.31</td><td>20.11</td></tr><tr><td>UniMLVG</td><td> $2 5 6 \times 4 4 8$ </td><td>2.0</td><td>10.70</td><td>1.54</td><td>25.32</td></tr><tr><td>OpenDWM-CTSD3.5</td><td> $2 5 6 \times 4 4 8$ </td><td>2.5</td><td>14.98</td><td>1.08</td><td>32.67</td></tr><tr><td>Omnidreams</td><td> $7 0 4 \times 1 2 8 0$ </td><td>2.0</td><td>0.65</td><td>8.00</td><td>45.71</td></tr><tr><td>Ours</td><td> $2 5 6 \times 4 4 8$ </td><td>3.1</td><td>0.60</td><td>18.63</td><td>45.21</td></tr><tr><td>Ours</td><td> $7 0 4 \times 1 2 8 0$ </td><td>3.1</td><td>1.19</td><td>6.94</td><td>47.45</td></tr></table>

## B.2 RUNTIME AND OVERHEAD.

As detailed in Table 9, WAM-Sim maintains high computational efficiency across resolutions despite incorporating explicit hierarchical memory modules. At 704 × 1280, WAM-Sim sustains 6.94 FPS, retaining 86.8% of OmniDreams’ throughput (8.00 FPS) with a negligible 3.8% increase in peak VRAM (47.45 GB vs. 45.71 GB). At 256 × 448, WAM-Sim reaches 18.63 FPS, outperforming same-resolution baselines UniMLVG (1.54 FPS) and OpenDWM–CTSD3.5 (1.08 FPS) by 12.1× and 17.3×, respectively. Although structural, scene, and instance memory branches expand total parameter capacity from 2.0B to 3.1B (+55.0%), memory retrieval via temporally routed and instance-masked cross-attention incurs minimal latency (+0.54s per 93-frame chunk).

## B.3 QUALITY-HORIZON DECAY ANALYSIS.

We evaluate temporal stability across 10 s, 30 s, and 60 s autoregressive rollouts (Fig. 9). Without world memory, compounding autoregressive errors cause severe degradation from 10 s to 60 s: Subject Coherence drops from 0.535 to 0.322, Scene Similarity degrades from 0.791 to 0.697, and FVD deteriorates from 197.77 to 286.54. In contrast, WAM-Sim effectively bounds drift across extended horizons, maintaining high instance coherence (0.547), scene similarity (0.828), and visual fidelity (FVD = 137.92) at 60 s.

![](images/31be2cfc7846b7031754f45ca8424b20ce2279f01e05668c6f4f227b455ff1fb.jpg)

![](images/adf95d3a86e5dde8eed457814ab6c2ce9ec2adec509d4695e4d7d8720448df0d.jpg)

![](images/42b5c2dac5917c7de43e01d0f07b115265a125feccedc9def4f60359959d6911.jpg)  
Figure 9: Quality versus generation horizon. World memory reduces quality degradation from 10 s to 60 s.

## C IMPLEMENTATION DETAILS

We present the architectural specifications, stage-wise training workflow, and reward formulations of WAM-Sim. The framework is trained in four progressive stages: structural control pre-training, scene-memory alignment, instance-memory registration, and online reinforcement post-training. Key model configurations and training parameters are summarized in Tables 11, 12, and 10.

## C.1 MODEL CONFIGURATION

WAM-Sim is built on a 28-block 3D Diffusion Transformer (DiT) backbone with a hidden dimension of 2048 and 16 attention heads $( D _ { \mathrm { h e a d } } = 1 2 8 )$ , conditioned on text embeddings from a frozen Qwen2.5-VL-7B encoder. Input videos at $7 0 4 \times 1 2 8 0$ resolution (93 frames at 10 FPS) are encoded by a Wan2.1 Video VAE with 8× spatial and 4× temporal compression into a $2 4 \times 8 8 \times 1 6 0$ latent grid. Applying $2 \times 2 \times 1$ spatio-temporal patchification yields 84,480 video tokens per chunk, modulated via 3D RoPE and timestep-conditioned adaLN-zero layers. As summarized in Table 11, Hierar chical World Memory conditions this shared sequence across three scale-specific tiers: (i) Structura Memory injects VAE-aligned HD map latents as additive conditioning to establish trajectory-aligned geometry; (ii) Scene Memory retrieves one geolocated reference frame $( P$ tokens) per temporal segment, injecting background appearance via segment-masked cross-attention with zero-initialized adaLN gating; and (iii) Instance Memory extracts multi-frame appearance crops $( K _ { f } = 4$ frames, $K _ { p } = 6 4$ tokens per crop, capping at 32 active instances) into a write-once track-keyed bank, updating foreground tokens via instance-masked attention while bypassing background regions.

Table 10: Training settings for Drive DiffusionNFT.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Group size G</td><td>16</td></tr><tr><td>Sampling steps Guidance scale</td><td>35</td></tr><tr><td>Flow shift</td><td>1.0 5.0</td></tr><tr><td>NFT mixing coefficient β</td><td>1.0</td></tr><tr><td>Training timestep fraction</td><td>0.8</td></tr><tr><td>Reward weights  $( \lambda _ { \mathrm { s c e n e } } , \lambda _ { \mathrm { o b j } } , \lambda _ { \mathrm { p h y } } )$ </td><td>0.4, 0.3, 0.3</td></tr><tr><td>LoRA rank / alpha</td><td>256 / 256</td></tr><tr><td>Optimizer</td><td></td></tr><tr><td>Learning rate</td><td>AdamW  $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 2 }$ </td></tr></table>

## C.2 TRAINING CONFIGURATION

WAM-Sim is trained via a progressive four-stage curriculum that incrementally introduces structural guidance, hierarchical memory routing, and online post-training without destabilizing the pretrained backbone. Stage I pre-trains structural control for 130k steps $( \mathrm { L R } = 1 \times 1 0 ^ { - 5 } )$ using rectified flow to establish trajectory-aligned road geometry. Stage II aligns scene memory for 15k steps $( \mathrm { L R } = 3 \times 1 0 ^ { - 5 } )$ to inject geolocated background representations via segment-masked cross-attention. Stage III optimizes instance memory for 5k steps $( \mathrm { L R } = 3 \times 1 0 ^ { - 4 } )$ to register write-once appearance tokens into the track-keyed memory bank. Finally, Stage IV executes online post-training via Drive DiffusionNFT $( \mathrm { L R } = 1 \times 1 0 ^ { - 5 } )$ with LoRA adapters to align self-generated rollouts against multi-scale visual and physical rewards. All stages utilize the AdamW optimizer with constant learning rates, 8× context parallelism, and bfloat16 precision; complete optimization hyperparameters are summarized in Table 12.

## C.3 DETAILS OF DRIVE DIFFUSIONNFT TRAINING

Drive DiffusionNFT aligns self-generated rollouts by optimizing a composite objective structured across three hierarchical visual and physical reward components. First, global visual quality $R _ { \mathrm { s c e n e } }$ is computed as the frame-averaged imaging quality score using a SPAQ-pretrained MUSIQ model (Huang et al., 2024). Second, object recognizability $R _ { \mathrm { o b j } }$ evaluates dynamic traffic participants (pedestrians, cyclists, and vehicles). Subject crops extracted with $1 . 2 \times$ bounding-box expansion are filtered by boundary margin, area, and occlusion, exposure-normalized against ground-truth frames to mitigate luminance gaming, and scored by a frozen ResNet-50 classifier (Liang et al., 2025) using dataset-calibrated z-normalized logits. Third, physical plausibility $R _ { \mathrm { p h y } }$ is evaluated via SoliReward (Lian et al., 2026), an InternVL3-1B model fine-tuned for motion realism and structural deformation assessment under its standard evaluation protocol.

Table 11: Model configuration. Shared backbone architecture and configurations of the structural, scene, and instance memory components.
<table><tr><td>Configuration</td><td>Structural Memory</td><td>Scene Memory</td><td>Instance Memory</td></tr><tr><td colspan="4">Shared Video Representation and Backbone</td></tr><tr><td>Input resolution</td><td>704 × 1280</td><td></td><td></td></tr><tr><td>Frames per window</td><td></td><td>93</td><td></td></tr><tr><td>Frame rate</td><td></td><td>10 FPS</td><td></td></tr><tr><td>Video VAE</td><td></td><td>Wan2.1 Video VAE</td><td></td></tr><tr><td>VAE latent channels</td><td></td><td>16</td><td></td></tr><tr><td>Spatial / temporal compression</td><td></td><td> $8 \times / 4 \times$ </td><td></td></tr><tr><td>Latent shape</td><td></td><td> $2 4 \times 8 8 \times 1 6 0$ </td><td></td></tr><tr><td>Spatial / temporal patch size</td><td></td><td> $2 \times 2 / 1$ </td><td></td></tr><tr><td>Video-token grid</td><td></td><td> $2 4 \times 4 4 \times 8 0$ </td><td></td></tr><tr><td>Number of video tokens</td><td></td><td>84,480</td><td></td></tr><tr><td>Hidden dimension</td><td></td><td>2048</td><td></td></tr><tr><td>Number of DiT blocks</td><td></td><td>28</td><td></td></tr><tr><td>Attention heads</td><td></td><td>16</td><td></td></tr><tr><td>Head dimension</td><td></td><td>128</td><td></td></tr><tr><td>MLP ratio / hidden dimension</td><td></td><td>4/8192</td><td></td></tr><tr><td>Position encoding</td><td>Learnable 3D RoPE</td><td> $( t , h , w )$ </td><td></td></tr><tr><td>Normalization</td><td>Timestep-conditioned adaLN-zero</td><td></td><td></td></tr><tr><td>Text encoder</td><td colspan="3">Frozen Qwen2.5-VL-7B (Cosmos-Reason1)</td></tr><tr><td>Text embedding dimension</td><td colspan="3">1024</td></tr><tr><td>World Anchored Memory</td><td></td><td></td><td></td></tr><tr><td>Memory scope</td><td>Trajectory-level HD map</td><td>Temporal-segment-level Geolocated scene reference</td><td>Instance-level</td></tr><tr><td>Condition source</td><td></td><td></td><td>Tracked object appearance</td></tr><tr><td>Validity domain</td><td>Video-token grid</td><td>Corresponding temporal segment</td><td>Corresponding instance region</td></tr><tr><td>Representation</td><td>VAE-aligned map latent</td><td>Scene-memory tokens</td><td>Appearance-memory tokens</td></tr><tr><td>Injection / retrieval</td><td>Additive conditioning</td><td>Segment-masked cross-attn</td><td>Instance-masked attention</td></tr><tr><td>Persistence</td><td>Trajectory aligned</td><td>Read-only after retrieval</td><td>Persistent across chunks</td></tr><tr><td>Reference frames</td><td></td><td>1 per segment</td><td> $K _ { f } = 4$ </td></tr><tr><td>Reference resolution</td><td></td><td>704 × 1280</td><td>128 × 128</td></tr><tr><td>Tokens per reference</td><td></td><td> $P$ </td><td> $K _ { p } = 6 4$ </td></tr><tr><td>Tokens per memory entry</td><td></td><td> $P$ </td><td>K = 256</td></tr><tr><td>Maximum entries</td><td></td><td> $N _ { b }$  segments</td><td>32 instances</td></tr><tr><td>Update strategy</td><td>Updated with trajectory</td><td>Fixed after retrieval</td><td>Write once at first appearance</td></tr><tr><td>Residual modulation</td><td></td><td>Zero-initialized adaLN gate</td><td>Zero-initialized adaLN gate</td></tr></table>

The scalar rewards are group-normalized and linearly aggregated into the overall rollout objective. Optimization parameters, LoRA adapter configurations, and reward weighting factors $( \lambda _ { \mathrm { s c e n e } } , \lambda _ { \mathrm { o b j } } , \lambda _ { \mathrm { p h y } } )$ for online flow-matching post-training are detailed in Table 10.

Table 12: Training configuration. Stage-wise training settings for structural control, scene memory, instance memory, and reinforcement fine-tuning.
<table><tr><td>Training Setting</td><td>Stage I: Structural</td><td>Stage II: Scene</td><td>Stage III: Instance</td><td>Stage IV: Reinforcement</td></tr><tr><td>Training target Initialized from</td><td>Structural control Pretrained</td><td>Scene memory Stage I</td><td>Instance memory Stage II</td><td>Generation quality Stage III</td></tr><tr><td>Trainable modules</td><td>backbone Self-Attn</td><td>Scene.Cross-attn</td><td>Instance.Cross-Attn</td><td>Self-Attn</td></tr><tr><td>Training objective</td><td>Rectified flow</td><td>Rectified flow</td><td>Rectified flow</td><td>DiffusionNFT loss</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $3 \times 1 0 ^ { - 5 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W }$ </td><td> $\mathrm { A d a m W }$ </td><td> $\mathrm { A d a m W }$ </td><td>AdamW</td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>LR schedule</td><td>constant</td><td>constant</td><td>constant</td><td>constant</td></tr><tr><td>Training steps</td><td>130k</td><td>15k</td><td>5k</td><td>300</td></tr><tr><td>Global batch size</td><td>4</td><td>4</td><td>4</td><td></td></tr><tr><td>Training resolution</td><td>704 × 1280</td><td>704 × 1280</td><td>704 × 1280</td><td>704 × 1280</td></tr><tr><td>Frames per sample</td><td>93</td><td>93</td><td>93</td><td>93</td></tr><tr><td>Latent frames</td><td>24</td><td>24</td><td>24</td><td>24</td></tr><tr><td>Context parallel size</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Precision</td><td>bf16</td><td>bf16</td><td>bf16</td><td>bf16</td></tr><tr><td>Training hardware</td><td>910b</td><td>910b</td><td>910b</td><td>910b</td></tr></table>

## D DATASET CONSTRUCTION

Training a long-horizon driving world model requires more than large-scale video data. The model must learn temporally consistent visual dynamics together with structured geometric conditions, persistent object identities, and semantic scene descriptions. We therefore construct a large-scale in-house autonomous-driving dataset and apply a multi-stage processing pipeline to transform raw driving logs into structured training samples. This section describes the data sources, construction pipeline, and final training and evaluation sets.

## D.1 DATA SOURCE

Our empirical dataset is constructed from a large-scale, in-house autonomous driving corpus gath ered across geographically diverse regions in China. The raw recordings are organized hierarchically into continuous scenes, each comprising temporally aligned clips with synchronized multi-modal sensor streams and structured annotations. Every frame contains a timestamped front-view RGB image (704 × 1280 at 10 FPS) paired with synchronized vehicle state trajectories, sensor calibra tion profiles, high-precision LiDAR point clouds, per-frame semantic segmentation masks, and 3D agent bounding boxes parameterized by persistent tracking IDs, positions, and orientations in the ego-vehicle coordinate system. Together, these multi-modal streams provide the dense geometric, visual, and instance-level groundings necessary for multi-scale world model training.

## D.2 DATA CONSTRUCTION PIPELINE

Raw driving logs inherently contain sensor anomalies, extreme motion, and misaligned annotations unsuitable for world-model training. To extract high-fidelity, structured training samples, we execute a systematic curation and annotation pipeline (Fig. 10).

We first enforce data quality through a three-stage filtering protocol. Stage I: Integrity Filtering discards sequences exhibiting dropped frames, exposure failures, or temporal discontinuities, preventing the generator from learning low-level sensor artifacts as scene dynamics. Stage II: Kinematic and Density Filtering mitigates motion blur and severe occlusion ambiguity by removing clips with average ego-velocities exceeding 45 km/h or heavily crowded scenes containing more than 20 annotated bounding boxes per frame. Stage III: Cross-Modal Alignment ensures precise geometric fidelity between spatial annotations and visual observations. We project HD-map elements and

![](images/61d96662c6b7b60e17693be95285220f839ded230e38df093283926ac37a8e5b.jpg)  
Figure 10: Dataset construction pipeline.

3D bounding boxes into the 2D image plane to compute their area-weighted overlap (Recall<sub>clip</sub>) against corresponding semantic segmentation masks. Sequences with Reca $\mathrm { 1 l _ { c l i p } < 0 . 5 }$ are rejected, systematically eliminating samples afflicted by projection drift or annotation lag.

For the curated sequences, we synthesize aligned conditioning signals to construct the final dataset. Structural Rendering and Captioning: We rasterize temporally synchronized HD maps to encode road topology and agent layouts. Concurrently, we generate dense textual descriptions using a twostage prompting strategy with Qwen3-VL. The vision-language model first extracts global environmental priors (e.g., weather, illumination, traffic density) from the video. This summary subsequently conditions the generation of a comprehensive scene caption detailing fine-grained agent behaviors and ego-state dynamics. An example of the final generated description used as a videogeneration prompt is shown in Figure 11.

Example of a Generated Video Prompt   
Ego vehicle travels on a multi-lane urban roadway at night under clear conditions with artificial streetlight illumination; wet asphalt surface reflects light, lane markings are visible. Ahead in the center-left lane, a vehicle 10–15m distant drives forward with brake lights illuminated, indicating deceleration. A second vehicle in the center-right lane, 20m ahead, maintains forward motion. On the right sidewalk, 30m ahead, a pedestrian walks parallel to the road. A green traffic light at the upcoming intersection permits progression. A blue directional sign 40m ahead indicates routes to Yan’an Road and Changshu Road; a red vertical sign 15m ahead confirms Yan’an Road.

Figure 11: Example of a final textual description generated by the two-stage Qwen3-VL annotation pipeline and used as a video-generation prompt.

Ultimately, each training sample is standardized into four perfectly synchronized modalities: (i) a continuous front-view RGB video; (ii) temporally aligned HD-map sequences; (iii) per-frame 3D instance annotations parameterized by bounding-box coordinates and persistent track identifiers; and (iv) a hierarchical scene caption describing environmental contexts and salient traffic interactions.

## D.3 TRAINING AND EVALUATION DATA

The curated training corpus comprises 121, 794 multi-modal driving clips (∼1,000 cumulative hours) sampled at 10 FPS (704 × 1280 resolution). As illustrated in Fig. 12, clip durations follow a balanced temporal distribution. The corpus spans diverse geographic regions, urban topology, highway segments, and rural environments across China, encompassing varying illumination (day, dusk, night) and adverse weather conditions (clear, rain, fog). This environmental heterogeneity ensures robust exposure to complex road geometry, lighting transitions, and traffic density variations.

![](images/5eba06c62cd6c19d94b07f4c5d06209458f55d72c9e7ac35f88e80bbabfc5e61.jpg)  
Figure 12: Distribution of clip durations in the training set.

To rigorously assess long-horizon generation and consistency, we establish a standardized evaluation set of 350 curated clips. The evaluation protocol specifically targets challenging edge cases, including severe weather conditions, complex directional maneuvers (e.g., tight turns and unscripted lane changes), and dense interactions with vulnerable road users (pedestrians, cyclists). Furthermore, the evaluation suite isolates identity-sensitive sequences characterized by prolonged tracking occlusions and subsequent re-emergence, providing a stringent benchmark for measuring long-term instance identity persistence and scene stability.

Table 13: Summary of our Self-Collected Dataset.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Number of training clips</td><td>121,794</td></tr><tr><td>Total driving duration</td><td>~1,000 hours</td></tr><tr><td>Frame resolution</td><td>704 × 1280</td></tr><tr><td>Frame rate</td><td>10 FPS</td></tr><tr><td>Camera view</td><td>Front view</td></tr><tr><td>Geographic coverage</td><td>Diverse regions and cities in China</td></tr><tr><td>Road types</td><td>Urban / highway / rural</td></tr><tr><td>Time of day</td><td>Day / dusk / night</td></tr><tr><td>Weather conditions</td><td>Clear / rainy / foggy</td></tr><tr><td>Structured conditions</td><td>HD map + instance annotations + caption</td></tr><tr><td>Evaluation set</td><td>350 curated clips</td></tr></table>

## D.4 ADDITIONAL DATASET DETAILS

Our self-collected corpus comprises driving sequences gathered across diverse urban, highway, and rural environments under varied diurnal and meteorological conditions. To preserve personal privacy, all identifiable elements—including human faces and license plates—undergo automated detection and rigorous spatial blurring prior to processing. Furthermore, location- and identityassociated metadata are anonymized in compliance with standard safety protocols.

The geolocated appearance database, $\mathcal { D } _ { \mathrm { g e o } } ,$ indexes reference images using GPS coordinates structured via a spatial k-d tree. Image feature embeddings are pre-computed and cached to eliminate redundant online extraction overhead, restricting spatial query latencies to under 5 ms during runtime retrieval.

Prior to training, raw logs undergo rigorous multi-stage filtering. Sequences containing sensor dropouts, black frames, or severe temporal discontinuities are systematically discarded. To maintain visual fidelity, we filter out clips featuring excessively high ego-velocities prone to motion blur, alongside over-crowded scenes exceeding dense participant thresholds. Finally, raw sensor streams—including LiDAR point clouds and camera parameters—are projected and rendered into synchronized, model-compatible conditioning representations.

## E EVALUATION AND METRIC FORMULATIONS

## E.1 QUALITY-HORIZON EVALUATION

We evaluate long-horizon generation stability at three rollout durations, $\mathcal { H } = \{ 1 0 , 3 0 , 6 0 \}$ s. Starting from the same initial conditioning, each method is rolled out using the same autoregressive chunkwise generation protocol. For each target horizon $h \in \mathcal H$ , we retain the generated sequence from the beginning of the rollout to h seconds and compute all metrics over the corresponding temporal range. The same evaluation samples, temporal horizons, backbone, sampler, and chunk configuration are used for all compared variants.

The ablation isolates the effect of world anchored memory. The baseline retains the structural HDmap conditioning of the backbone but removes the additional scene- and instance-memory readout branches, while our full model enables both. Thus, differences between the two variants reflect the contribution of persistent scene- and instance-level memory under an otherwise identical generation protocol.

We report three complementary measures. Subject Coherence evaluates the temporal appearance consistency of foreground traffic participants, while Scene Similarity measures the preservation of environment-level visual appearance over the rollout. Frechet Video Distance´ (FVD) measures overall video distributional quality. Higher values are better for Subject Coherence and Scene Similarity, whereas lower values are better for FVD.

For a metric $Q$ and method $m .$ , let $Q _ { m } ( h )$ denote its value evaluated over the rollout up to horizon h. We report

$$
\mathcal { Q } _ { m } = \{ Q _ { m } ( h ) \ | \ h \in \mathcal { H } \} ,\tag{4}
$$

which characterizes the evolution of generation quality as the autoregressive horizon increases. All methods are evaluated independently at the same three horizons using an identical protocol.

## E.2 SCENE PRESERVATION METRICS

Scene Similarity (SS). Scene Similarity measures the visual consistency between the generated video and the ground-truth video within the scene regions. Let $\hat { \mathbf { I } } _ { t }$ and $\mathbf { I } _ { t } ^ { \mathrm { G T } }$ denote the generated and ground-truth frames at time t, respectively, and let $\mathbf { \bar { M } } _ { t } \in \{ 0 , 1 \} ^ { H \times W }$ denote the corresponding scene mask. A frozen DINO image encoder ϕ extracts patch-level features, and the scene mask is average-pooled to the DINO patch grid to obtain patch weights $m _ { t , p } \in [ 0 , 1 ]$

The Scene Similarity of the entire video is defined as

$$
\mathrm { S S } = \frac { \sum _ { t = 1 } ^ { T } \sum _ { p = 1 } ^ { P } m _ { t , p } \cos \Big ( \phi _ { p } ( \hat { \mathbf { I } } _ { t } ) , \phi _ { p } ( \mathbf { I } _ { t } ^ { \mathrm { G T } } ) \Big ) } { \sum _ { t = 1 } ^ { T } \sum _ { p = 1 } ^ { P } m _ { t , p } } ,\tag{5}
$$

where $T$ is the number of video frames, $P$ is the number of DINO patch tokens, and $\phi _ { p } ( \cdot )$ denotes the DINO feature of patch p. A higher SS indicates stronger scene-level appearance and structural consistency with the ground-truth video.

## E.3 IDENTITY PRESERVATION METRICS

We evaluate instance identity preservation using appearance embeddings extracted from generatedvideo crops. For each visible instance i at frame t, we square-expand its ground-truth (GT) bounding box by 10%, resize the resulting crop to $2 2 4 \times 2 2 4$ , and extract an appearance embedding using a frozen DINOv2 ViT-B/14 model. The embedding is ℓ<sub>2</sub>-normalized and denoted as $e _ { t , i } .$ All reported similarities are cosine similarities, and higher values indicate better identity preservation. We use a temporal stride of $S = 9 2$ frames, corresponding to the autoregressive chunk interval.

A track is considered valid for SC@kS if it contains at least one pair of frames separated by kS for which the instance remains continuously visible throughout the entire interval. For occlusionbased metrics, an occlusion event is defined as a maximal contiguous run of invisible frames that is strictly bounded by visible segments of the same track. The pre- and post-occlusion appearance prototypes are constructed only from the contiguous visible segments immediately adjacent to the corresponding occlusion event.

Long-range coherence (SC@kS). For each track, we compare appearance embeddings separated by a temporal gap of

$$
\Delta = k S , \qquad k \in \{ 1 , 2 , 3 \} ,
$$

while requiring the instance to remain continuously visible throughout the interval.

Let $\gamma _ { i , k }$ denote the set of valid starting frames for track i:

$\nu _ { i , k } = \{ t$ | instance i remains visible from t through $t + k S \}$

and let

$$
\mathcal { T } _ { k } = \left\{ i \vert \vert \mathcal { V } _ { i , k } \vert > 0 \right\}
$$

denote the set of valid tracks. We compute

$$
\mathrm { S C @ } k S = \frac { 1 } { | \mathcal { T } _ { k } | } \sum _ { i \in \mathcal { T } _ { k } } \frac { 1 } { | \mathcal { V } _ { i , k } | } \sum _ { t \in \mathcal { V } _ { i , k } } e _ { t , i } ^ { \top } e _ { t + k S , i } , \qquad k \in \{ 1 , 2 , 3 \} .\tag{6}
$$

This metric measures long-range appearance coherence while excluding appearance changes associated with disappearance and re-emergence.

Occlusion and re-emergence. An occlusion event u is a maximal contiguous interval during which an instance is invisible and is preceded and followed by visible segments of the same track. Let i(u) denote the corresponding instance.

For each event, $\mathcal { P } _ { u }$ contains up to the last S frames from the contiguous visible segment immediately preceding the occlusion, and $\mathcal { Q } _ { u }$ contains up to the first S frames from the contiguous visible segment immediately following re-emergence. We construct appearance prototypes on the two sides of the occlusion as

$$
p _ { u } ^ { - } = \mathrm { N o r m } \left( \frac { 1 } { | \mathcal { P } _ { u } | } \sum _ { t \in \mathcal { P } _ { u } } e _ { t , i ( u ) } \right) , \qquad q _ { u } ^ { + } = \mathrm { N o r m } \left( \frac { 1 } { | \mathcal { Q } _ { u } | } \sum _ { t \in \mathcal { Q } _ { u } } e _ { t , i ( u ) } \right) ,\tag{7}
$$

where Norm(·) denotes $\ell _ { 2 }$ normalization.

The re-emergence similarity for event u is then defined as

$$
s _ { u } = ( p _ { u } ^ { - } ) ^ { \top } q _ { u } ^ { + } .\tag{8}
$$

Re-emergence similarity. We report a video-level macro-average to prevent videos containing many occlusion events from dominating the metric. Let D denote the set of videos containing at least one valid occlusion event, and let $\mathcal { U } _ { v }$ denote the set of valid occlusion events in video v. We compute

$$
\mathrm { R e - e m e r g . ~ S i m . } = \frac { 1 } { | \mathcal { D } | } \sum _ { v \in \mathcal { D } } \frac { 1 } { | \mathcal { U } _ { v } | } \sum _ { u \in \mathcal { U } _ { v } } s _ { u } .\tag{9}
$$

Re-emergence pass rate. We additionally report the fraction of valid occlusion events whose reemergence similarity exceeds a GT-calibrated threshold $\tau .$ . Let

$$
\mathcal { U } = \bigcup _ { v \in \mathcal { D } } \mathcal { U } _ { v }
$$

denote the set of all valid occlusion events. The pass rate is

$$
\mathrm { P a s s R a t e } = \frac { 1 } { | \mathcal { U } | } \sum _ { u \in \mathcal { U } } \mathbf { 1 } [ s _ { u } \ge \tau ] .\tag{10}
$$

We set $\tau = 0 . 8 0 3$ , corresponding to the median GT re-emergence similarity over all valid occlusion events in the evaluation set. The threshold is determined once from the GT videos and is fixed for all evaluated methods.

Occ-ID@mS. To analyze identity preservation under different occlusion durations, we group events according to their occlusion length $L _ { u }$ . For the m-th duration interval, we define

$$
{ \cal B } _ { m } = \{ u \vert ( m - 1 ) S < L _ { u } \leq m S \} ,\tag{11}
$$

and compute

$$
\mathrm { O c c \mathrm { - } I D @ } m S = \frac { 1 } { | \mathcal { B } _ { m } | } \sum _ { u \in \mathcal { B } _ { m } } s _ { u } .\tag{12}
$$

In our test set, we report Occ-ID@1S and Occ-ID@2S, corresponding to occlusion lengths of (0, 92] and (92, 184] frames, respectively. The Occ-ID@2S interval contains only three valid events, and no valid event has an occlusion duration longer than 184 frames.

Re-emergence similarity is macro-averaged at the video level, whereas Pass Rate and Occ-ID are averaged over individual occlusion events.

Average score. The reported Avg. score is the arithmetic mean of the seven metrics in Table 5: SC@1S, SC@2S, SC@3S, Re-emergence Similarity, Re-emergence Pass Rate, Occ-ID@1S, and Occ-ID@2S.

The GT videos are used only to determine track identities, visibility, occlusion events, bounding boxes, and the calibration threshold τ . All appearance similarities reported for each evaluated method are computed exclusively from the generated videos.

## E.4 RL EVALUATION METRICS

Subject Fidelity. We measure whether the conditioned dynamic subjects are detectable in the generated video with a GT-gated detector protocol. Conditioning boxes (per-frame tight AABBs of the conditioned pedestrians, cyclists, cars and vehicles) are taken from the ground-truth annotations. A YOLO26s detector trained on BDD100k is run on every frame of the generated video; each conditioning box is greedily matched to the highest-IoU same-class detection. To exclude annotation and visibility ambiguity from the denominator, the identical detection-and-matching procedure is run on the ground-truth video, and only instances the detector recovers in the GT frames count as valid.

Physical Plausibility. We use Qwen3.5-9B as a no-reference VLM judge to evaluate pedestrian motion plausibility. Each video is evaluated with two binary questions covering motion anatomy and limb integrity. We use the normalized probability of “yes” at the first generated token rather than a sampled binary response. Specifically, the probability is computed by renormalizing the logits over the predefined yes/no token variants. The final score is the mean P(yes) across the two questions. Only videos with clearly visible pedestrians are included in this evaluation.

## Physical-Plausibility Prompts

Motion Anatomy: Do all walking or running pedestrians show perfectly natural, anatomically plausible limb movement throughout the entire video, with zero unnatural joint motion and zero abnormal limb trajectories? (yes or no)

Limb Integrity: For moving pedestrians, do their arms and legs remain clearly distinguishable throughout the motion, without limbs merging into the body, merging with each other, disappearing, or becoming visually entangled? (yes or no)

## F LIMITATIONS AND FUTURE WORK

Despite its strong long-horizon consistency, WAM-Sim exhibits three primary limitations. First, memory retrieval latency and attention memory footprint scale linearly with the number of tracked foreground instances, introducing computational overhead in hyper-dense traffic scenes containing over 100 concurrent agents. Second, the write-once instance appearance bank assumes rigid visual properties; consequently, extreme non-rigid state transitions (e.g., severe vehicle damage during collisions or drastic open/close deformations of commercial truck beds) are constrained by the initial-observation tokens. Third, background scene retrieval depends on geolocated reference availability, which may degrade to ungrounded autoregressive generation in unmapped or highly dynamic construction zones.

To address these challenges, future iterations will pursue three key research directions: 1) Integrating explicit 3D visual representations, such as online-updated Gaussian Splatting or neural radiance fields, into the memory substrate to enable unified multi-view spatial consistency and continuous geometry-aware retrieval under counterfactual maneuvers. 2) Formulating a parametric write-update memory mechanism that selectively incorporates visual state changes across significant illumination or topological shifts while preserving fundamental instance identity invariants. 3) Developing dynamic instance-grouping and memory-compression protocols to bound memory complexity in extreme ultra-dense urban traffic scenarios.

## G ETHICS, BROADER IMPACT, AND REPRODUCIBILITY ASSURANCE

WAM-Sim advances autonomous driving safety by enabling controllable, zero-cost edge-case simulation over extended horizons. While high-fidelity video generation presents potential dual-use risks regarding synthetic media misuse, these concerns can be effectively mitigated through digital watermarking protocols and restricted access frameworks for generative models.

To ensure full reproducibility, we release our complete training and evaluation code, model checkpoints, and dataset processing pipelines. Detailed architectural configurations, optimization hyper parameters, and hardware environments are thoroughly documented in Appendices C and D.