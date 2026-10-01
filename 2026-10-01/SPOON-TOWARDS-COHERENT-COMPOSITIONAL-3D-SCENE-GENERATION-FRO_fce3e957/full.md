# SPOON: TOWARDS COHERENT COMPOSITIONAL 3D SCENE GENERATION FROM UNCALIBRATED MULTI-VIEW IMAGES

Guibiao Liao<sup>1,2,∗</sup>, Mochu Xiang<sup>1,2,5,∗</sup>, Heng Li<sup>3,2</sup>, Ken Deng<sup>3,2</sup>, Zijie Wang<sup>4,2</sup>, Guanbin Li<sup>4,2</sup>, Ping Tan<sup>3,2</sup>, Shenghua Gao<sup>1,2,5</sup>, Yizhou Yu<sup>1,2</sup>

![](images/51889a0d315430d4b5172aa23458532f1e0e7be204d1ab511f83d3b25771f414.jpg)  
input images

![](images/4add61c7c707d07a098886a3bef095e00fff450f613839039ff79681b91bca8e.jpg)  
a soup of objects and cameras

![](images/430a36a13d7d8e4ddb4b31279aebb88c9d9bb296773df1cbce4bd49fd67a2942.jpg)  
coherent compositional scene  
Figure 1: Given multi-view images, current 3D generation pipelines often yield a soup of objects. SPOON can recover a coherent compositional 3D scene with accurate object geometry and layout.

## ABSTRACT

Compositional 3D scene generation aims to recover complete 3D object shapes and their spatial arrangement from visual observations. Recent image-conditioned 3D generators provide strong priors for producing high-quality object geometry, making the generation of complex scenes increasingly practical. A central challenge is therefore to spatially organize these generated assets into a globally coherent scene while remaining consistent with multi-view observations. Existing approaches either entangle scene layout with object generation or separately estimate spatial placement from view-specific observations, where pose hypotheses may remain ambiguous and inconsistent across views, often resulting in an incoherent object-camera soup. We introduce SPOON, a framework that reformulates multi-view compositional 3D generation as scene-level, geometry-grounded pose reasoning. Rather than treating view-specific object pose hypotheses independently, SPOON coordinates them using reconstruction-derived multi-view geometry through a Guide-Route-Reconcile paradigm. This progressively organizes object poses and camera configurations into a coherent scene-level spatial arrangement. Extensive experiments on ARSG-110K and MIDI-3D-Front demonstrate consistent improvements in object placement and scene composition across varying numbers of input views. On ARSG-110K, SPOON reduces scene-level and object-level Chamfer distances by 12.7% and 17.7%, respectively, compared with a strong baseline.

## 1 INTRODUCTION

Multi-view compositional 3D scene generation can construct coherent scenes by recovering complete object assets together with their spatial arrangement from visual observations, supporting applications in embodied AI, simulation, and 3D content creation. Recent advances in image-conditioned 3D generation Xiang et al. (2025b;a); Huang et al. (2026a); Chen et al. (2026) provide strong priors for recovering high-quality object geometry, making compositional 3D scene generation increasingly practical. A challenge is therefore how to incorporate multi-view observations and recover the spatial layout that organizes these generated assets into a coherent scene.

One strategy is to incorporate layout directly into the generation process. For example, 3D-Fixer Yin et al. (2026) conditions object completion on partial geometry in the scene coordinate frame, using the observed geometry as a spatial anchor to generate objects in place. While avoiding explicit post-generation pose recovery, this formulation entangles shape generation with scene-space placement, requiring object generative priors learned in canonical space to additionally accommodate incomplete observations under arbitrary spatial transformations.

An alternative strategy preserves object generation in its canonical space and models spatial placement separately. SAM 3D Chen et al. (2026), for example, generates canonical object geometry while recovering its transformation with respect to the observing camera. Such a decomposition preserves the advantages of object-centric generative priors, while making accurate pose recovery critical to coherent scene composition. Pose estimates inferred from individual object-view observations are inherently scale-ambiguous and can become inaccurate under occlusion, image-boundary truncation, and limited visibility. Multiple views provide complementary evidence, yet do not auto matically resolve this ambiguity: different observations may support pose estimation with substantially different reliability, and independently plausible hypotheses can become mutually inconsistent when expressed in a shared scene frame. Thus, these methods often provide an incoherent object– camera soup.

Scene-level 3D reconstruction provides complementary geometric cues for resolving this ambiguity. It jointly recovers camera geometry and scene structure in a shared coordinate system, establishing globally consistent relations across observations. This shared geometry provides a common reference for coordinating object hypotheses across cameras and reasoning about their support within the overall scene. This raises a natural question: Can scene-level geometry organize locally plausible object–camera hypotheses into a globally coherent 3D composition?

We answer this question with SPOON, standing for Scene-level Pose and Object layout OptimizatioN. We cast multi-view object placement as a scene-level pose reasoning problem, where view-conditioned hypotheses are coordinated through shared scene geometry rather than considered in isolation. In this way, SPOON stirs the object-camera soup into a coherent scene, combining strong object-centric generative priors with scene-level geometric reasoning for coherent spatial placement. SPOON follows a Guide–Route–Reconcile paradigm. First, we Guide pose generation with reconstruction-derived scene geometry, grounding view-conditioned predictions in globally consistent spatial cues. Second, we Route the resulting pose hypotheses by assessing the geometric reliability of each view-specific observation against the reconstructed object geometry. Finally, we Reconcile the object layouts and camera poses through joint differentiable optimization, reducing residual object–camera inconsistencies while preserving the generated object geometry. Together, these stages progressively coordinate local pose hypotheses into a coherent scene-level spatial configuration without altering the underlying object generative prior. Our main contributions are:

• We identify the object–camera soup problem in compositional 3D scene generation, where locally plausible object-camera hypotheses may not constitute a coherent scene-level configuration.

• We propose SPOON, a scene-level pose reasoning framework that follows a Guide–Route– Reconcile paradigm to guide pose generation using shared geometry, route view-specific hypotheses by geometric reliability, and jointly reconcile object layouts and camera poses.

• Extensive experiments on ARSG-110K and MIDI-3D-Front show consistent improvements in scene composition and object placement across varying numbers of input views.

## 2 RELATED WORK

## 2.1 3D SHAPE GENERATION

Recent advances in 3D generation have substantially improved the quality and scalability of objectlevel asset creation from textual or visual inputs. Early methods were largely developed on relatively small-scale 3D datasets Wang et al. (2018); Gkioxari et al. (2019); Chang et al. (2015), whereas the emergence of large-scale collections such as Objaverse Deitke et al. (2023b) and Objaverse-XL Deitke et al. (2023a) has enabled increasingly powerful generative priors. Modern native 3D generators explore a wide range of representations, including VecSets Zhang et al. (2023), tri-planes Gao et al. (2022); Wu et al. (2024), geometry images Yan et al. (2025); Elizarov et al. (2024), 3D primitives Chen et al. (2025d), structured latents Xiang et al. (2025b); Wu et al. (2025); Chen et al. (2025b), and autoregressive mesh representations Fang et al. (2025); Chen et al. (2024); Tang et al. (2025); Siddiqui et al. (2024); Chen et al. (2025a;c). Together with advances in scalable architectures and optimized 3D operators Wu et al. (2025); Chen et al. (2025b); Xiang et al. (2025a), these developments have significantly improved the geometric fidelity and structural complexity of gen erated 3D assets. The increasing availability of high-quality object generation makes it increasingly feasible to extend object-level generative models toward compositional 3D scene generation. Our work builds upon this progress and focuses on reliable object placement and spatial organization in complex multi-view scenes.

## 2.2 COMPOSITIONAL 3D SCENE GENERATION

Building upon strong object-level generative priors, recent studies have extended 3D generation toward compositional scenes with multiple objects and observations. Two design questions are particularly relevant in this setting: how to integrate complementary evidence across views, and how to recover the spatial layout of generated objects within a shared scene frame.

For multi-view object generation, recent approaches increasingly combine reconstruction and generative priors to improve geometric completeness and input consistency. ReconViaGen Chang et al. (2026), UniRecGen Huang et al. (2026b), and Mix3R Lin et al. (2026) transfer knowledge from 3D reconstruction foundation models Wang et al. (2025; 2026b) to condition 3D generators on multiview observations. These approaches demonstrate the benefit of integrating reconstruction priors or multi-view consistency into generative models, but primarily focus on aggregating information across observations for object shape enhancement.

Accurately modeling object layouts further requires 3D generation models to be explicitly poseaware. Recent approaches decouple shape from camera or object pose using representations such as UV volumes Huang et al. (2026a) or camera tokens Chen et al. (2026). For example, ShapeR Siddiqui et al. (2026) first predicts object-to-world transformations from detected bounding boxes and subsequently generates their 3D shapes under the estimated placements. Another line of work directly applies layout transformations to object representations and learns to model the resulting SE(3)-transformed shapes within a scene volume. While this formulation enables pixel-aligned geometry Li et al. (2026b), entangling arbitrary object transformations with shape representation broadens the distribution that the generative model must capture and can degrade the geometry of generated meshes Ling et al. (2026); Yin et al. (2026). This motivates compositional formulations that preserve object geometry in canonical space while modeling spatial transformations separately. SceneGen Meng et al. (2026) predicts object positions before assembling independently generated assets, while CAST Yao et al. (2025) further refines the resulting scene composition using photometric and physics-aware objectives. MV-SAM3D Li et al. (2026a) extends to multi-view observations through Multi-Diffusion Bar-Tal et al. (2023) and confidence-aware latent velocity fusion. These formulations retain strong object-level generative priors while exposing spatial layout as an explicit variable for scene composition.

## 3 PRELIMINARIES

## 3.1 JOINT SHAPE AND POSE MODELING

Given an image prompt, SAM3D Chen et al. (2026) can recover the object shape and the camera pose. It adopts a two-stage flow-matching architecture: the first stage jointly generates canonical object geometry (as low-resolution occupancy) and the relative camera pose; the second stage gen erates the object’s texture and fine geometry. Given an object observation I, the first-stage shape and pose branches predict the corresponding velocity fields:

$$
\begin{array} { r } { \mathbf { v } _ { \theta } ^ { S } , \mathbf { v } _ { \theta } ^ { P } = f _ { \theta } ( \mathbf { z } _ { t } ^ { S } , \mathbf { z } _ { t } ^ { P } , \mathcal { T } , t ) , } \end{array}\tag{1}
$$

where $\mathbf { z } _ { t } ^ { S }$ denotes the shape latent and $\mathbf { z } _ { t } ^ { P } = ( \mathbf { z } _ { t } ^ { R } , \mathbf { z } _ { t } ^ { T } , \mathbf { z } _ { t } ^ { s } )$ is the camera-relative pose latent, parameterizing rotation, translation, and scale at flow time t. Under rectified flow, both states evolve

according to their predicted velocities:

$$
\begin{array} { r } { \mathbf { z } _ { t + \Delta t } ^ { S } = \mathbf { z } _ { t } ^ { S } + \Delta t \mathbf { v } _ { \theta } ^ { S } , \qquad \mathbf { z } _ { t + \Delta t } ^ { P } = \mathbf { z } _ { t } ^ { P } + \Delta t \mathbf { v } _ { \theta } ^ { P } . } \end{array}\tag{2}
$$

At the endpoint, the shape latent is decoded into a canonical object representation $\mathcal { G } _ { o } .$ , while the pose state yields a camera-frame transformation $T _ { o , v } ^ { c }$ that places the generated object relative to view v. Instead of entangling object shape with its layout, separately modeling geometry and layout keeps the data distribution simple and easy to learn, which is beneficial in generating high-fidelity 3D shapes. However, this might sacrifice the accuracy of the layout estimation.

## 3.2 MULTI-VIEW FLOW FUSION

Multi-Diffusion Bar-Tal et al. (2023) fuses prompt-conditioned velocity predictions into a global coherent direction at each generation step, which makes a flow matching model accept multiple conditions. MV-SAM3D Li et al. (2026a) extends SAM3D to multi-view generation to exploit complementary cues across observations. Given V views, the corresponding shape velocities $\{ \mathbf { v } _ { \theta , v } ^ { S } \} _ { v = 1 } ^ { V }$ are aggregated into a shared update:

$$
\mathbf { v } _ { \theta } ^ { S } = \mathcal { A } \big ( \{ \mathbf { v } _ { \theta , v } ^ { S } \} _ { v = 1 } ^ { V } \big ) , \qquad \mathbf { z } _ { t ^ { + } } ^ { S } = \mathbf { z } _ { t } ^ { S } + \Delta t \mathbf { v } _ { \theta } ^ { S } ,\tag{3}
$$

where A denotes the multi-view velocity fusion operator, $t ^ { + } = t + \Delta t$ . This yields a shared shape trajectory that integrates complementary evidence across observations. In contrast, the pose trajectories remain view-conditioned and evolve independently. Consequently, multi-view fusion produces a shared canonical shape while retaining view-specific camera-relative pose hypotheses.

## 4 METHOD

Given V unposed RGB images $\{ I _ { v } \} _ { v = 1 } ^ { V }$ and the corresponding instance masks $\{ M _ { o , v } \}$ , our goal is to reconstruct a compositional 3D scene containing O observed objects. We leverage feedforward 3D reconstruction models Wang et al. (2026a) to recover per-view point maps $X _ { v }$ and camera intrinsics $K _ { \imath }$ and world-to-camera extrinsics $E _ { v }$ in a shared world coordinate system.

For each object, we follow MV-SAM3D Li et al. (2026a) to obtain a shared canonical representation $\mathcal { G } _ { o }$ together with view-specific camera-frame pose hypotheses $\{ T _ { o , v } ^ { c } \} _ { v = 1 } ^ { V }$ . We seek an object-toworld transformation $T _ { o } ^ { \mathrm { o { \bar { w } } } }$ for each canonical object, yielding the compositional scene:

$$
\begin{array} { r } { { \cal S } = \{ ( { \mathcal G } _ { o } , T _ { o } ^ { \mathrm { o w } } ) \} _ { o = 1 } ^ { { O } } , } \end{array}\tag{4}
$$

where $T _ { o } ^ { \mathrm { o w } }$ places $\mathcal { G } _ { o }$ into the shared world frame.

While MV-SAM3D produces plausible camera-relative object placements, these view-specific hypotheses may collectively lack a scene-coherent spatial configuration, giving rise to an object– camera soup. To address this, SPOON uses multi-view geometric cues as a scene-level reference and organizes these hypotheses through a Guide–Route–Reconcile paradigm, as described below.

## 4.1 GEOMETRY-GUIDED OBJECT POSE GENERATION

While multi-view flow fusion (Sec. 3.2) aggregates complementary cues for shape generation, pose generation in Li et al. (2026a) remains view-specific. Such independently inferred poses may be ambiguous and unreliable from a single image, whereas feed-forward 3D reconstruction models Wang et al. (2026a) recover camera poses from the full multi-view input with stronger geometric consistency. Thus, we propose to use these reconstruction-derived poses as an inference-time geometric prior to guide pose generation.

A straightforward solution is to align reconstruction-derived camera poses in the world frame with generated poses in the object canonical frame and use the aligned poses as guidance. However, this requires decoding intermediate pose states from the flow trajectory, which are often unstable and can introduce substantial errors into the cross-frame alignment. In contrast, the canonical object surface remains substantially more stable than the decoded pose states and therefore provides a more reliable anchor for cross-frame alignment. Therefore, we establish the cross-frame correspondence through object geometry, aligning the aggregated reconstruction with the canonical object surface to bridge the two coordinate systems.

![](images/cd903c8c84746bca34510897a7e74ff894df678860af1e39f7f2adc5f0e85556.jpg)  
Figure 2: Overview of SPOON. Given multiple uncalibrated images, 3D reconstruction methods first recover the scene geometry as camera poses and depth maps. Then, for each object, we prompt the mesh generation process with multi-view visual cues and guide the pose recovery with reconstruction guidance, which is then placed using an optimal layout hypothesis through routing. Finally, the object layouts and camera poses are jointly optimized and reconciled to provide a coherent compositional 3D scene.

Surface coordinate bridge. Concretely, we construct object surfaces in both coordinate frames. Using the point maps and camera extrinsics from 3D reconstruction methods Wang et al. (2026a), we transform the masked object points into the shared world frame and aggregate them into a worldspace point cloud $\mathcal { P } _ { W }$ . To establish a correspondence with the canonical frame where each shape is generated, we recover a clean occupancy from the noisy shape latent and the fused velocity:

$$
\widehat { \mathbf { z } } _ { 1 } ^ { S } = \mathbf { z } _ { t } ^ { S } + ( 1 - t ) \mathbf { v } _ { \theta } ^ { S } ,\tag{5}
$$

where $\mathbf { v } _ { \theta } ^ { S }$ denotes the fused shape velocity. We decode $\widehat { \mathbf { z } } _ { 1 } ^ { S }$ and sample its surface to obtain the canonical point cloud $\mathcal { P } _ { C }$ . Since $\mathcal { P } _ { W }$ and $\mathcal { P } _ { C }$ lie in different coordinate systems, solving the transformation between them can robustly transfer the reconstructed camera pose into the canonical space in shape generation. After centering and normalizing both point cloud sets, we search for an optimal rotation that best aligns them, yielding the bridge-aligned pose $\pi _ { v } ^ { * }$ . We then encode this aligned pose using the SAM3D pose encoder:

$$
{ \bf z } _ { 1 , v } ^ { P , * } = \mathrm { E n c } _ { P } \left( \pi _ { v } ^ { * } \right) ,\tag{6}
$$

which serves as the geometry-derived clean endpoint for the pose rectified flow.

Flow-consistent endpoint injection. Directly replacing the intermediate pose state $\mathbf { z } _ { t , v } ^ { P }$ with the geometry-derived endpoint $\mathbf { z } _ { 1 , v } ^ { P , * }$ would ignore the current flow time and deviate from the learned trajectory. We instead construct a reference state at the same flow time. From the current state and predicted velocity, we recover the corresponding noise endpoint as $\widehat \epsilon _ { t , v } ^ { P } = \mathbf { z } _ { t , v } ^ { P } - t \mathbf { v } _ { \theta , v } ^ { P }$ . We then keep this noise realization fixed and replace only the clean endpoint with $\mathbf { z } _ { 1 , v } ^ { P , * }$ to define a time-matched reference state, which is then used to guide the Euler update:

$$
\begin{array} { r } { \mathbf { z } _ { t + , v } ^ { P , \mathrm { r e f } } = ( 1 - t ^ { + } ) \hat { \epsilon } _ { t , v } ^ { P } + t ^ { + } \mathbf { z } _ { 1 , v } ^ { P , * } , } \end{array}\tag{7}
$$

$$
\begin{array} { r } { \mathbf { z } _ { t ^ { + } , v } ^ { P , \mathrm { g u i d e d } } = ( 1 - \alpha _ { P } ) \mathbf { z } _ { t ^ { + } , v } ^ { P , \mathrm { b a s e } } + \alpha _ { P } \mathbf { z } _ { t ^ { + } , v } ^ { P , \mathrm { r e f } } . } \end{array}\tag{8}
$$

where $\mathbf { z } _ { t ^ { + } , v } ^ { P , \mathrm { b a s e } }$ denotes the standard Euler proposal. This update retains the noise endpoint inferred from the current linear flow approximation while steering the clean endpoint toward the geometryderived pose.

## 4.2 GEOMETRY-AWARE POSE ROUTING

Geometry guidance improves object pose generation, yet the resulting hypotheses can still vary in reliability across views. Differences in visibility and boundary truncation may affect the geometric evidence available for pose estimation. Since pose accuracy cannot be directly assessed at inference time, we use geometric observability as a proxy for reliability. Specifically, we derive a per-view reliability score from its support under the aggregated multi-view geometry and use it to select the pose hypothesis for world-frame placement. For clarity, we omit the object index o below.

Multi-view geometric support. We first lift the masked point-map observations from all views into the shared world frame using the estimated camera extrinsics. Their union is voxel-downsampled into an aggregated object point cloud ${ \mathcal { P } } _ { W }$ , which combines complementary observations and serves as a geometric proxy for the object extent. For each view $v ,$ we reproject $\mathcal { P } _ { W }$ onto the image plane: Mc = Rasterize $( \{ \Pi ( K _ { v } \mathcal { T } ( E _ { v } , \mathbf { P } ) ) \mid \mathbf { P } \in \mathcal { P } _ { W } \} )$ , where $\Pi ( \cdot )$ denotes perspective projection. We evaluate $\widehat { M _ { v } }$ before restricting it to the valid image domain $\Omega _ { v } ,$ , so that it also captures geometry projected beyond the image boundary. Together with the observed mask $M _ { v }$ , this provides complementary cues for boundary truncation and visible coverage.

Geometry-aware reliability. We characterize geometric observability using frame completeness $F _ { v }$ and visible coverage $C _ { v } \colon$

$$
F _ { v } = \frac { \lvert \widehat { M _ { v } } \cap \Omega _ { v } \rvert } { \lvert \widehat { M _ { v } } \rvert } , C _ { v } = \frac { \lvert M _ { v } \cap \widehat { M _ { v } } \rvert } { \lvert \widehat { M _ { v } } \cap \Omega _ { v } \rvert } , S _ { v } = [ F _ { v } \geq \tau _ { f } ] F _ { v } ^ { \alpha } C _ { v } ,\tag{9}
$$

where $F _ { v }$ measures how much of the projected object extent remains inside the image, while $C _ { v }$ measures how much of the in-frame geometric support is covered by the observation. Here, $\tau _ { f }$ is the completeness threshold used to filter severely truncated observations. Their combination $S _ { v }$ serves as a geometry-aware proxy for pose reliability.

Reliability-aware pose routing. We select the most reliable view and transfer its pose hypothesis to the shared world frame:

$$
v ^ { \star } = \arg \operatorname* { m a x } _ { v } S _ { v } , \qquad T _ { o } ^ { \mathrm { o w } } = E _ { v ^ { \star } } ^ { - 1 } T _ { o , v ^ { \star } } ^ { c } ,\tag{10}
$$

where $T _ { o , v ^ { \star } } ^ { c }$ <sub>⋆</sub> is the object transformation predicted in the selected camera frame. This converts heterogeneous multi-view pose hypotheses into a single placement supported by the most geometrically reliable observation.

## 4.3 MULTI-VIEW POSE RECONCILIATION

The strategy introduced in Sec. 4.2 improves the object layout by selecting the most plausible layout among the candidates estimated from different input views. However, the quality of the resulting layout remains bounded by the accuracy of the initial per-view layout estimations. To further improve the consistency of the reconstructed scene, we leverage the multi-view observations and jointly refine the object layouts through differentiable Gaussian splatting.

Specifically, for the o-th object, we introduce an additional $\mathrm { S i m ( 3 ) }$ transformation parameterized by $( \dot { s } ^ { \prime } , R ^ { \prime } , t ^ { \prime } )$ to refine its initial layout $( s , R , t ) = T _ { o } ^ { \mathrm { o w } }$ . The resulting transformation is given by:

$$
s \gets s ^ { \prime } s , \qquad R \gets R ^ { \prime } R , \qquad t \gets s ^ { \prime } R ^ { \prime } t + t ^ { \prime } .\tag{11}
$$

We initialize the additional transformation as the identity, i.e., $s ^ { \prime } = 1 , R ^ { \prime } = I _ { 3 \times 3 } .$ , and $t ^ { \prime } = 0$ . The transformed object Gaussians are then composited and rendered from the input viewpoints, whose camera poses are obtained from a 3D reconstruction foundation model Wang et al. (2026a). In this way, the multi-view observations provide image-space supervision for refining the object layout without requiring explicit cross-view point or feature correspondences.

Although this optimization can effectively correct moderate layout errors, its performance remains coupled with the estimated camera geometry. Errors in the estimated camera poses perturb multi view consistency, limiting the effectiveness of object-level refinement. To address this issue, we refine the camera poses jointly with the object layouts. We employ a Gaussian rasterization kernel with explicit camera Jacobians Matsuki et al. (2024), which enables efficient gradient-based optimization of the camera parameters through the rendering process. Conceptually, this optimization is analogous to Bundle Adjustment (BA) in Structure-from-Motion, where camera poses and scene geometry are jointly optimized to achieve multi-view consistency. Rather than relying on explicit point or feature correspondences, our formulation directly minimizes the discrepancy between the rendered and observed images. Moreover, the scene geometry is represented by the generated object-level Gaussian representations, while the object motion is parameterized by a compact Sim(3) transformation.

Table 1: Main results on the ARSG-110K and MIDI-3D-Front test sets under varying numbers of input views. We report scene- and object-level metrics together with 3D bounding-box IoU for spatial layout evaluation. Best results within each input-view setting are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Views</td><td colspan="5">ARSG-110K</td><td colspan="5">MIDI-3D-Front</td></tr><tr><td> $\mathrm { C D } _ { S } \downarrow$ </td><td> $\mathrm { F S } _ { S } \uparrow$ </td><td>CDo ↓</td><td> $\mathrm { F S } _ { O }$ </td><td>↑ IoU ↑</td><td> $\mathrm { C D } _ { S } \downarrow$ </td><td> $\mathrm { F S } _ { S } \uparrow$ </td><td>CDo ↓</td><td>FSo↑</td><td>IoU ↑</td></tr><tr><td>Gen3DSR</td><td>1</td><td>0.265</td><td>46.72</td><td>0.546</td><td>31.95</td><td>0.304</td><td>0.123</td><td>40.07</td><td>0.157</td><td>38.11</td><td>0.363</td></tr><tr><td>MIDI</td><td>1</td><td>0.801</td><td>15.35</td><td>0.179</td><td>35.99</td><td>0.033</td><td>0.080</td><td>50.19</td><td>0.103</td><td>53.58</td><td>0.518</td></tr><tr><td>I-Scene</td><td>1</td><td>0.277</td><td>34.69</td><td>0.563</td><td>21.25</td><td>0.228</td><td>0.311</td><td>10.76</td><td>0.187</td><td>69.90</td><td>0.025</td></tr><tr><td>3D-Fixer</td><td>1</td><td>0.159</td><td>68.82</td><td>0.197</td><td>57.85</td><td>0.519</td><td>0.069</td><td>78.67</td><td>0.032</td><td>94.39</td><td>0.492</td></tr><tr><td>SAM3D</td><td>1</td><td>0.146</td><td>65.32</td><td>0.257</td><td>48.48</td><td>0.445</td><td>0.084</td><td>77.06</td><td>0.053</td><td>92.62</td><td>0.430</td></tr><tr><td>ShapeR</td><td>4</td><td>0.073</td><td>78.10</td><td>0.148</td><td>61.83</td><td>0.502</td><td>0.082</td><td>87.28</td><td>0.024</td><td>96.31</td><td>0.506</td></tr><tr><td>MV-SAM3D</td><td>4</td><td>0.063</td><td>82.37</td><td>0.113</td><td>69.48</td><td>0.585</td><td>0.083</td><td>87.84</td><td>0.020</td><td>96.76</td><td>0.597</td></tr><tr><td>Ours</td><td>4</td><td>0.055</td><td>87.08</td><td>0.093</td><td>78.02</td><td>0.662</td><td>0.069</td><td>91.73</td><td>0.016</td><td>97.66</td><td>0.684</td></tr><tr><td>MV-SAM3D</td><td>6</td><td>0.056</td><td>84.56</td><td>0.108</td><td>72.98</td><td>0.606</td><td>0.086</td><td>85.62</td><td>0.030</td><td>95.40</td><td>0.568</td></tr><tr><td>Ours</td><td>6</td><td>0.047</td><td>89.68</td><td>0.087</td><td>81.42</td><td>0.695</td><td>0.070</td><td>91.13</td><td>0.023</td><td>97.09</td><td>0.684</td></tr><tr><td>MV-SAM3D</td><td>8</td><td>0.051</td><td>86.90</td><td>0.101</td><td>74.33</td><td>0.615</td><td>0.083</td><td>86.23</td><td>0.029</td><td>95.54</td><td>0.572</td></tr><tr><td>Ours</td><td>8</td><td>0.038</td><td>92.84</td><td>0.078</td><td>84.54</td><td>0.723</td><td>0.069</td><td>91.21</td><td>0.026</td><td>96.62</td><td>0.684</td></tr></table>

Formally, given the set of input images, we jointly optimize the camera poses $\{ E _ { v } \}$ and the perobject similarity transformations $\{ ( s _ { o } ^ { \prime } , R _ { o } ^ { \prime } , t _ { o } ^ { \prime } ) \}$ by minimizing the multi-view rendering objective:

$$
\operatorname* { m i n } _ { \{ E _ { v } \} , \{ s _ { o } ^ { \prime } , R _ { o } ^ { \prime } , t _ { o } ^ { \prime } \} } \sum _ { v } \mathcal { L } _ { \mathrm { r e n d e r } } \left( \mathcal { R } \left( \mathcal { G } _ { o } ( s _ { o } ^ { \prime } , R _ { o } ^ { \prime } , t _ { o } ^ { \prime } ) _ { o } , E _ { v } \right) , I _ { v } \right) ,\tag{12}
$$

where $\mathcal { G } _ { o }$ denotes the Gaussian representation of the o-th object, R denotes differentiable Gaussian rasterization, and $I _ { v }$ is the v-th input image. The rendering loss can incorporate both photometric and depth supervision, where depth is from 3D reconstruction model as a regularization term. By jointly optimizing the camera and object parameters, the method allows their relative configurations to adapt under shared multi-view supervision, leading to a more globally consistent reconstruction.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Implementation Details. We use VGGT-Omega Wang et al. (2026a) to estimate camera poses and scene point cloud from the input views, and adopt SAM3D with its default settings as the underlying object generator. All experiments are conducted on a single NVIDIA H100 GPU. We evaluate our method on two benchmarks for compositional 3D scene generation: the ARSG-110K test set Yin et al. (2026) and the MIDI-3D-Front Fu et al. (2021). Both benchmarks provide object-level geometry together with ground-truth scene configurations, enabling evaluation of object placement and overall scene composition.

Baselines. We compare with representative single-view and multi-view scene generation methods. Single-view baselines include Gen3DSR Ardelean et al. (2025), MIDI Huang et al. (2025), I-Scene Ling et al. (2026), 3D-Fixer Yin et al. (2026), and SAM3D Chen et al. (2026). We further compare with ShapeR Siddiqui et al. (2026) and MV-SAM3D Li et al. (2026a), which leverage multiple observations for 3D generation and provide the most relevant comparisons to our multi-view setting.

Metrics. Following 3D-Fixer Yin et al. (2026), we evaluate scene composition at both the scene and object levels. We report Chamfer Distance and F-Score for the complete scene $( \mathrm { C D } _ { S } , \mathrm { F S } _ { S } )$ and for individual objects $( \mathrm { C D } _ { O } , \mathrm { F S } _ { O } )$ , with the predicted object poses retained during evaluation. These metrics therefore reflect both geometric fidelity and errors in object placement. The F-Score is computed with a distance threshold of 0.1. We further report the volumetric IoU between predicted and ground-truth 3D object bounding boxes as a more direct measure of spatial layout accuracy. Lower CD and higher F-Score and IoU indicate better performance.

![](images/d93195ebe22a6e3aa5e371b0e3327d3c293b72ac75a9791cb8a1f7c8fe0b2761.jpg)  
Figure 3: Qualitative comparison on the MIDI-3D-Front. Compared with existing single-view and multi-view methods, our method produces more accurate object placements and scene layouts.

![](images/10d9ed2da32023894e16804e7a9e4dabc00aa65e79cc9d9c56235daa60ea3ba9.jpg)  
Figure 4: Qualitative comparison on the ARSG-110K test set.

## 5.2 COMPARISON WITH STATE-OF-THE-ART METHODS

ARSG-110K. In Table 1, under the 4-view setting, multi-view methods substantially outperform single-view approaches, confirming the benefit of complementary observations. Compared with MV-SAM3D, our method reduces $\mathrm { C D } _ { S }$ and $\mathrm { C D } _ { O }$ by 12.7% and 17.7%, while improving FS<sub>S</sub> and $\mathrm { F S } _ { O }$ by 5.7% and 12.3%, respectively. More notably, object-layout IoU improves by 13.2%, showing that geometry-grounded pose reasoning more effectively converts complementary multiview evidence into accurate object placement and coherent scene composition.

MIDI-3D-Front. A similar trend is observed on MIDI-3D-Front. Under the 4-view setting, our method reduces $\mathrm { C D } _ { S }$ and CD by 16.9% and 20.0% over MV-SAM3D, respectively, while improving object-layout IoU substantially by 14.6%. This suggests that the main advantage of our method lies in more reliable spatial placement and scene composition rather than merely improving object-level agreement.

Table 2: Ablation of pose routing. Pose guidance and multi-view reconciliation are disabled.
<table><tr><td>Routing Strategy</td><td> $\mathrm { C D } _ { S } .$  →</td><td> $\mathrm { F S } _ { S } \uparrow$   $\mathrm { I o U \uparrow }$ </td></tr><tr><td>Reference View</td><td>0.083</td><td>87.84 0.597</td></tr><tr><td>Largest Mask</td><td>0.075</td><td>89.64 0.611</td></tr><tr><td>Max Frame Completeness  $F _ { v }$ </td><td>0.079</td><td>88.12 0.598</td></tr><tr><td>Max Visible Coverage  $C _ { v }$ </td><td>0.075</td><td>89.94 0.614</td></tr><tr><td>Max Reliability  $S _ { v } \ \mathrm { ( O u r s ) }$ </td><td>0.072</td><td>90.08 0.617</td></tr></table>

![](images/fea5549204ff78d7e60cfd52b91012fd0ed682d728a642441867f8fb6581e243.jpg)

Table 3: Ablation of reconciliation.
<table><tr><td>Variant</td><td> $\mathrm { C D } _ { S } \downarrow$ </td><td> $\mathrm { F S } _ { S }$  ↑</td><td>IoU↑</td></tr><tr><td>Object Only</td><td>0.062</td><td>86.21</td><td>0.651</td></tr><tr><td>Object &amp; Camera</td><td>0.055</td><td>87.08</td><td>0.662</td></tr></table>

Figure 5: Analysis of reliability-aware pose routing. Boundary truncation in Views 1 and 3 yields unreliable pose hypotheses. Our reliability score favors View 2, whose placement is more consistent with the scene layout.

Scaling with input views. Our method remains robust as the number of input views increases. On ARSG-110K, the relative IoU gain over MV-SAM3D increases from 13.2% with 4 views to 14.7% with 6 views and 17.6% with 8 views, while the reduction in $\mathrm { C D } _ { S }$ grows from 12.7% to 16.1% and 25.5%, respectively. On MIDI-3D-Front, SPOON maintains stable scene-level CD and layout IoU as the number of views increases. These results show that our method is more stable in exploiting increasing view coverage via reliability-aware pose reasoning.

Qualitative comparison. Fig. 3 and 4 show qualitative comparisons on MIDI and ARSG, respectively. Our method produces more accurate object orientations and spatial arrangements, resulting in fewer misplaced objects and more coherent scene layouts.

## 5.3 ABLATION STUDIES

Table 4 shows that the full model achieves superior results, while removing any stage consistently degrades scene-level reconstruction and layout accuracy, confirming the complementary roles of the three components. Removing Guide indicates that geometrygrounded generation provides better pose initialization, while the drop without Route highlights the benefit of selecting geometrically reliable pose hypotheses. Moreover, the IoU gain shows the effectiveness of reconciliation in enforcing coherent scene-level placement.

Table 2 shows that pose reliability cannot be adequately captured by mask size or a single geometric cue alone. Frame completeness and visible coverage capture complementary aspects of geometric observability, and their combination in $S _ { v }$ enables more reliable pose selection across views. Fig. 5 further illustrates that boundary-truncated observa-

Table 4: Component ablation of our method.
<table><tr><td>Variant</td><td> $\mathrm { C D } _ { S } \downarrow$ </td><td> $\mathrm { F S } _ { S } \uparrow$ </td><td> $\mathrm { C D } _ { O } \downarrow$ </td><td> $\mathrm { F S } _ { O }$  ←</td><td>IoU↑</td></tr><tr><td>Baseline</td><td>0.083</td><td>87.84</td><td>0.020</td><td>96.76</td><td>0.597</td></tr><tr><td>w/o Guide</td><td>0.073</td><td>91.47</td><td>0.018</td><td>97.58</td><td>0.677</td></tr><tr><td>w/o Route</td><td>0.071</td><td>90.63</td><td>0.019</td><td>97.10</td><td>0.669</td></tr><tr><td>w/o Reconcile</td><td>0.072</td><td>90.20</td><td>0.014</td><td>97.50</td><td>0.617</td></tr><tr><td>Ours</td><td>0.069</td><td>91.73</td><td>0.016</td><td>97.66</td><td>0.684</td></tr></table>

tions yield less reliable hypotheses, while $S _ { v }$ favors a placement more consistent with the scene layout. Table 3 shows that jointly refining object placements and camera poses improves scene-level reconstruction and layout accuracy.

## 6 CONCLUSION

We presented SPOON, a scene-level pose reasoning framework for compositional 3D scene generation. Rather than treating object poses as independent predictions from local observations, SPOON uses reconstructed multi-view geometric cues as a shared scene reference to guide pose generation, route reliable pose hypotheses, and reconcile object–camera configurations through the proposed Guide–Route–Reconcile paradigm. This formulation turns independently inferred object–camera pose hypotheses into a coherent scene configuration while preserving the strong object priors of 3D generators. Experiments on ARSG-110K and MIDI-3D-Front demonstrate consistent improvements in object placement and scene composition quality across different numbers of input views.

## REFERENCES

Andreea Ardelean, Mert ${ \mathrm { { \ddot { O } } z e r , } }$ and Bernhard Egger. Gen3dsr: Generalizable 3d scene reconstruction via divide and conquer from a single view. In Proceedings of International Conference on 3D Vision, pp. 616–626. IEEE, 2025.

Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. In Proceedings of the International Conference on Machine Learning, pp. 1737–1752, 2023.

Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

Jiahao Chang, Chongjie Ye, Yushuang Wu, Yuantao Chen, Yidan Zhang, Zhongjin Luo, Chenghong Li, Yihao Zhi, and Xiaoguang Han. Reconviagen: Towards accurate multi-view 3d object reconstruction via generation. In International Conference on Learning Representations, volume 2026, pp. 90387–90401, 2026.

Sijin Chen, Xin Chen, Anqi Pang, Xianfang Zeng, Wei Cheng, Yijun Fu, Fukun Yin, Billzb Wang, Jingyi Yu, Gang Yu, et al. Meshxl: Neural coordinate field for generative 3d foundation models. Advances in Neural Information Processing Systems, 37:97141–97166, 2024.

Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, et al. Sam 3d: 3dfy anything in images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7220–7232, 2026.

Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Zhongang Cai, Lei Yang, Gang Yu, Guosheng Lin, et al. Meshanything: Artist-created mesh generation with autoregressive transformers. In The Thirteenth International Conference on Learning Representations, 2025a.

Yiwen Chen, Zhihao Li, Yikai Wang, Hu Zhang, Qin Li, Chi Zhang, and Guosheng Lin. Ultra3d: Efficient and high-fidelity 3d generation with part attention. arXiv preprint arXiv:2507.17745, 2025b.

Yiwen Chen, Yikai Wang, Yihao Luo, Zhengyi Wang, Zilong Chen, Jun Zhu, Chi Zhang, and Guosheng Lin. Meshanything v2: Artist-created mesh generation with adjacent mesh tokenization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 13922– 13931, 2025c.

Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie, Tong Wu, Shunsuke Saito, et al. 3dtopia-xl: Scaling high-quality 3d asset generation via primitive diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26576–26586, 2025d.

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36:35799–35813, 2023a.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 13142–13153, 2023b.

Slava Elizarov, Ciara Rowles, and Simon Donne. Geometry image diffusion: Fast and data-efficient´ text-to-3d with image-based surface representation. arXiv preprint arXiv:2409.03718, 2024.

Shuangkang Fang, I Shen, Yufeng Wang, Yi-Hsuan Tsai, Yi Yang, Shuchang Zhou, Wenrui Ding, Takeo Igarashi, Ming-Hsuan Yang, et al. Meshllm: Empowering large language models to progressively understand and generate 3d mesh. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14061–14072, 2025.

Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, et al. 3d-front: 3d furnished rooms with layouts and semantics. In Proceedings of IEEE/CVF International Conference on Computer Vision, pp. 10913–10922. IEEE, 2021.

Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images. Advances in neural information processing systems, 35:31841–31854, 2022.

Georgia Gkioxari, Jitendra Malik, and Justin Johnson. Mesh r-cnn. In Proceedings ofthe IEEE/CVF international conference on computer vision, pp. 9785–9795, 2019.

Binbin Huang, Haobin Duan, Yiqun Zhao, Zibo Zhao, Yi Ma, and Shenghua Gao. Cupid: Generative 3d reconstruction via joint object and pose modeling. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12741–12752, 2026a.

Zehuan Huang, Yuan-Chen Guo, Xingqiao An, Yunhan Yang, Yangguang Li, Zi-Xin Zou, Ding Liang, Xihui Liu, Yan-Pei Cao, and Lu Sheng. Midi: Multi-instance diffusion for single image to 3d scene generation. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23646–23657. IEEE, 2025.

Zhisheng Huang, Jiahao Chen, Cheng Lin, Chenyu Hu, Hanzhuo Huang, Zhengming Yu, Mengfei Li, Yuheng Liu, Zekai Gu, Zibo Zhao, et al. Unirecgen: unifying multi-view 3d reconstruction and generation. arXiv preprint arXiv:2604.01479, 2026b.

Baicheng Li, Dong Wu, Jun Li, Shunkai Zhou, Zecui Zeng, Lusong Li, and Hongbin Zha. Mv-sam3d: Adaptive multi-view fusion for layout-aware 3d generation. arXiv preprint arXiv:2603.11633, 2026a.

Dong-Yang Li, Wang Zhao, Yuxin Chen, Wenbo Hu, Meng-Hao Guo, Fang-Lue Zhang, Ying Shan, and Shi-Min Hu. Pixal3d: Pixel-aligned 3d generation from images. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pp. 1–12, 2026b.

Siyou Lin, Zhou Xue, Hongwen Zhang, Liang An, Dongping Li, Shaohui Jiao, and Yebin Liu. Mix3r: Mixing feed-forward reconstruction and generative 3d priors for joint multi-view aligned 3d reconstruction and pose estimation. In Proceedings ofthe Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pp. 1–12, 2026.

Lu Ling, Yunhao Ge, Yichen Sheng, and Aniket Bera. I-scene: 3d instance models are implicit generalizable spatial learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26974–26983, 2026.

Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting slam. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18039– 18048. IEEE, 2024.

Yanxu Meng, Haoning Wu, Ya Zhang, and Weidi Xie. Scenegen: Single-image 3d scene generation in one feedforward pass. In 2026 International Conference on 3D Vision (3DV), pp. 543–553. IEEE, 2026.

Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner. Meshgpt: Generating triangle meshes with decoderonly transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19615–19625, 2024.

Yawar Siddiqui, Duncan Frost, Samir Aroudj, Armen Avetisyan, Henry Howard-Jenkins, Daniel DeTone, Pierre Moulon, Qirui Wu, Zhengqin Li, Julian Straub, et al. Shaper: Robust conditional 3d shape generation from casual captures. arXiv preprint arXiv:2601.11514, 2026.

Jiaxiang Tang, Zhaoshuo Li, Zekun Hao, Xian Liu, Gang Zeng, Ming-Yu Liu, and Qinsheng Zhang. Edgerunner: Auto-regressive auto-encoder for artistic mesh generation. In The Thirteenth International Conference on Learning Representations, 2025.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5294–5306. IEEE, 2025.

Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schonberger, Patrick¨ Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, and Christian Rupprecht. Vggtomega. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21486–21499, 2026a.

Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, and Yu-Gang Jiang. Pixel2mesh: Generating 3d mesh models from single rgb images. In European conference on computer vision, pp. 55–71. Springer, 2018.

Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry learning. In International Conference on Learning Representations, volume 2026, pp. 10481–10497, 2026b.

Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. Advances in Neural Information Processing Systems, 37:121859–121881, 2024.

Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu Zhu, Xun Cao, Philip Torr, et al. Direct3d-s2: Gigascale 3d generation made easy with spatial sparse attention. arXiv preprint arXiv:2505.17412, 2025.

Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, et al. Native and compact structured latents for 3d generation. arXiv preprint arXiv:2512.14692, 2025a.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 21469–21480, 2025b.

Xingguang Yan, Han-Hung Lee, Ziyu Wan, and Angel X Chang. An object is worth 64× 64 pixels: Generating 3d object via image diffusion. In 2025 International Conference on 3D Vision (3DV), pp. 123–133. IEEE, 2025.

Kaixin Yao, Longwen Zhang, Xinhao Yan, Yan Zeng, Qixuan Zhang, Lan Xu, Wei Yang, Jiayuan Gu, and Jingyi Yu. Cast: Component-aligned 3d scene reconstruction from an rgb image. ACM Transactions on Graphics (TOG), 44(4):1–19, 2025.

Ze-Xin Yin, Liu Liu, Xinjie Wang, Wei Sui, Zhizhong Su, Jian Yang, and Jin Xie. 3d-fixer: Coarseto-fine in-place completion for 3d scenes from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12753–12763, 2026.

Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions On Graphics (TOG), 42(4):1–16, 2023.

## A APPENDIX

In this appendix, we first provide further methodological (A.2) and implementation details $( \mathsf { A } . 3 )$ We then present additional analyses and qualitative results (A.4), followed by further discussion of limitations and future directions (A.5).

## A.1 THE USE OF LARGE LANGUAGE MODELS

We use Large Language Models (LLMs) only for minor language editing to improve grammar and readability. All methodological design, experiments, equations, and results are developed by the authors.

## A.2 ADDITIONAL METHOD DETAILS

This section provides additional details of the SPOON inference procedure. We first summarize the complete Guide–Route–Reconcile pipeline in Algorithm 1, and then detail the surface-based coordinate bridge used to transfer reconstruction-derived pose into the object canonical frame.

Algorithm 1 SPOON Inference   
Require: Multi-view images $\{ I _ { v } \} _ { v = 1 } ^ { V }$ and object masks $\{ M _ { o , v } \}$   
Ensure: Compositional scene $\mathbf { \bar { \mathbf { \Lambda } } } ^ { S } = \{ ( \mathcal { G } _ { o } , T _ { o } ^ { \mathrm { o w } } ) \} _ { o = 1 } ^ { O }$   
1: Recover point maps $\{ X _ { v } \}$ , camera intrinsics $\{ K _ { v } \}$ , and world-to-camera extrinsics $\{ E _ { v } \}$ using the feed  
forward reconstruction model.   
2: for each object o do   
3: Aggregate the masked reconstruction points into the world-space object surface $\mathcal { P } _ { W } ^ { o }$   
4: Initialize the shared shape latent $\mathbf { z } ^ { S }$ and the view-specific pose latents $\{ \mathbf { z } _ { v } ^ { P } \} _ { v = 1 } ^ { V } .$   
5: Initialize the surface bridge as invalid.   
6: for each flow step t do   
7: Predict per-view shape and pose velocities $\{ \mathbf { v } _ { \theta , v } ^ { S } , \mathbf { v } _ { \theta , v } ^ { P } \} _ { v = 1 } ^ { V }$   
8: Fuse $\{ \grave { \mathbf { v } } _ { \theta , v } ^ { S } \} _ { v = 1 } ^ { V }$ and update the shared shape trajectory.   
9: if geometry guidance is activated then   
10: if the surface bridge has not been initialized then   
11: Recover the predicted clean shape endpoint and sample the canonical surface $\mathcal { P } _ { C } ^ { o }$   
12: Estimate the surface-based coordinate bridge $B _ { W  C } ^ { o }$ between $\mathcal { P } _ { W } ^ { o }$ and $\mathcal { P } _ { C } ^ { o } .$   
13: Validate and cache $B _ { W \to C } ^ { o }$ if the geometric alignment is reliable.   
14: end if   
15: if $B _ { W \to C } ^ { o }$ is valid then   
16: Transfer the reconstruction-derived camera poses into the object canonical frame and construct   
geometry-derived pose endpoints $\{ \mathbf { z } _ { 1 , v } ^ { P , * } \} _ { v = 1 } ^ { V }$   
17: Apply flow-consistent endpoint guidance to the view-specific pose trajectories.   
18: else   
19: Apply the original pose-flow update.   
20: end if   
21: else   
22: Apply the original pose-flow update.   
23: end if   
24: end for   
25: Decode the canonical object representation $\mathcal { G } _ { o }$ and the view-specific pose hypotheses $\{ T _ { o , v } ^ { c } \} _ { v = 1 } ^ { V } .$   
26: Reproject $\mathcal { P } _ { W } ^ { o }$ into each view and compute the geometric reliability scores $\{ \bar { S } _ { o , v } \} _ { v = 1 } ^ { V }$   
27: Select $v _ { o } ^ { \star } =$ arg max<sub>v</sub> $S _ { o , v }$ and initialize $T _ { o } ^ { \mathrm { o w } } \doteq E _ { v _ { o } ^ { \star } } ^ { - 1 } T _ { o , v _ { o } ^ { \star } } ^ { c } .$   
28: end for   
29: Jointly refine the object transformations $\{ T _ { o } ^ { \mathrm { o w } } \}$ and camera poses $\{ E _ { v } \}$ using differentiable multi-view   
rendering.   
30: return $\overset { \vartriangle } { \boldsymbol { S } } = \{ ( \mathcal { G } _ { o } , T _ { o } ^ { \mathrm { o w } } ) \} _ { o = 1 } ^ { O }$

Overall Inference Procedure. Given multi-view images and their object instance masks, SPOON first recovers a shared geometric reconstruction, including point maps and camera parameters. For each object, during flow matching sampling, multi-view shape velocities are fused into a shared canonical shape trajectory, while the view-specific pose trajectories are geometrically guided once a sufficiently stable canonical surface becomes available. After generation, the view-specific pose hypotheses are routed according to their geometric reliability to initialize the object placements in the shared world frame. Finally, object transformations and camera poses are jointly reconciled through differentiable multi-view rendering.

Surface-Based Coordinate Bridge. We provide additional details on the surface alignment used to estimate the coordinate bridge in Sec. 4.1. Given the world-space and canonical object surfaces $\mathcal { P } _ { W }$ and $\mathcal { P } _ { C }$ defined in the main paper, we independently center and isotropically normalize the two point sets to remove translation and global scale. After aligning the up axes of the reconstruction and canonical frames, we denote the resulting normalized surfaces by $\bar { \mathcal { P } } _ { W }$ and $\bar { \mathcal { P } } _ { C }$ . The remaining rotational ambiguity is reduced to a yaw rotation, which we estimate by:

$$
\theta ^ { * } = \underset { \theta } { \arg \operatorname* { m i n } } \ : \mathrm { T r i m R M S } _ { \bar { \mathbf { q } } \in \bar { \mathcal { P } } _ { C } } \left[ \underset { \bar { \mathbf { p } } \in \bar { \mathcal { P } } _ { W } } { \operatorname* { m i n } } \ : \left\| \mathbf { R } _ { \mathrm { y a w } } ( \theta ) ^ { \top } \bar { \mathbf { q } } - \bar { \mathbf { p } } \right\| _ { 2 } \right] .\tag{13}
$$

We solve this objective using a global coarse-to-fine yaw search over the aggregated multi-view geometry. The resulting rotation is shared across all views of the same object, thereby preserving their relative camera configuration. The estimated yaw defines the surface bridge $\mathbf { B } _ { W  C } ^ { o }$ from the reconstruction world frame to the object canonical frame. Given the reconstruction-derived worldto-camera transformation $\mathbf { E } _ { v }$ , the corresponding geometry-derived object-to-camera transformation is obtained as:

$$
\mathbf { T } _ { o , v } ^ { c , * } = \mathbf { E } _ { v } ( \mathbf { B } _ { W  C } ^ { o } ) ^ { - 1 } .\tag{14}
$$

where $\mathbf { T } _ { o , v } ^ { c , * }$ is converted to the pose parameterization of SAM3D, yielding $\pi _ { v } ^ { * }$ used in $\operatorname { E q . } \ 6 .$ To improve robustness against partial observations and reconstruction outliers, we compute TrimRMS using the lowest-error fraction $\tau _ { \mathrm { t r i m } }$ of nearest-neighbor residuals, with $\tau _ { \mathrm { t r i m } } ~ = ~ 0 . 8 5$ across all experiments. Although the reconstructed surface may only partially cover the complete canonical object, aggregating observations across multiple views provides broader geometric support than any individual view. The normalized alignment is used only to estimate the remaining rotational correspondence between the two frames, while TrimRMS reduces the influence of unmatched regions and reconstruction outliers. The estimated bridge is accepted only when the normalized fitting RMSE is below 0.15 and at least four valid views provide sufficient geometric support. Otherwise, geometry guidance is disabled, and the original pose flow is used. Once accepted, the bridge is cached and reused for the remaining guided sampling steps.

## A.3 IMPLEMENTATION DETAILS

For geometric pose guidance, we set the guidance strength to $\alpha _ { P } = 0 . 8$ and activate guidance from $t = 0 . 6$ onward, applying it at every subsequent sampling step. These two settings are selected based on preliminary experiments. For reliability-aware pose routing, we set $\alpha = 2$ in $\operatorname { E q }$ . equation 9 and apply a completeness threshold $\tau _ { f } ~ = ~ 0 . 5$ to exclude severely truncated observations. For the reconciliation stage, we jointly optimize the object transformations and camera poses using AdamW, with learning rates of $\mathrm { { \dot { 3 } } } \times 1 \mathrm { { \dot { 0 } } } ^ { - 3 }$ and $1 \times 1 0 ^ { - \hat { 4 } }$ , respectively. The loss is a combination of photometric loss and depth loss. Since we only use depth loss as a regularization term, the depth loss is formulated as $\left. M \odot \left( \exp ( - d ) - \exp ( - \hat { d } ) \right) \right. _ { 2 } ^ { 2 }$ , where d is the depth map estimated by 3D reconstruction models, <sup>ˆ</sup>d is the rendered depth map, M is the object mask.

We evaluate computational efficiency on a single NVIDIA H100 GPU using scenes containing five objects under the same input setting. With four-view input, MV-SAM3D requires 195s per fiveobject scene, while the complete SPOON pipeline takes 326.5s. The proposed geometry guidance and pose routing introduce only modest additional costs of 10s and 1.5s, respectively, while the iterative reconciliation accounts for the remaining 120s.

We use the object instance masks provided by the benchmark datasets for all methods requiring object-level observations. For MV-SAM3D, we follow its official inference configuration, with both layout injection during generation and post-generation object-pose refinement enabled. Both MV-SAM3D and our method place generated objects in the world coordinate system recovered by VGGT-Ω, whose global gauge generally differs from that of the ground-truth scene. For evaluation, we estimate a single global Sim(3) transformation by aligning the reconstructed camera centers with the ground-truth camera centers and apply it uniformly to the entire predicted scene. For MV-SAM3D, this transformation is estimated from the initial camera poses predicted by VGGT-Ω, whereas for our method it is estimated from the camera poses after multi-view reconciliation.

No additional per-object alignment is applied for either scene-level or object-level evaluation. For single-view baselines, where multi-view camera-center alignment is unavailable, we follow the coordinate and scale calibration protocols of their official implementations. For example, 3D-Fixer performs in-place completion in the scene frame defined by its fragmented geometric input, with the scene scale calibrated according to its released evaluation pipeline.

## A.4 ADDITIONAL RESULTS

We provide additional analyses and qualitative results to further validate the proposed method. We first examine the effect of geometry-guided pose generation and the sensitivity of the reliabilityaware routing strategy to the exponent α. We then present additional qualitative comparisons on the MIDI-3D-Front and ARSG-110K test sets to demonstrate the consistency of the improvements across diverse scenes.

![](images/7804c70856c3c053a66819ba9b13bd77c3b6ea174d937ef7fbc710f2a2c8f01e.jpg)  
Figure 6: Visualization of pose generation. The highlighted object exhibits an inaccurate orientation without guidance, while pose guidance steers the generated pose toward an orientation more consistent with the object layout.

Effect of pose guidance. Fig. 6 qualitatively demonstrates the effect of geometry-guided pose generation. Without guidance, the generated objects can exhibit noticeable orientation errors, whereas geometry guidance steers the pose toward a configuration more consistent with the observed scene layout. This qualitatively demonstrates that the reconstructed geometric cues provide effective guid ance for improving pose generation.

Table 5: Sensitivity analysis of the routing exponent α on MIDI with four input views, with multiview reconciliation disabled.
<table><tr><td>α</td><td> $\mathrm { C D } _ { S } \downarrow$ </td><td> $\mathrm { F S } _ { S } \uparrow$ </td><td>IoU↑</td></tr><tr><td>1</td><td>0.072</td><td>90.08</td><td>0.619</td></tr><tr><td>2</td><td>0.072</td><td>90.20</td><td>0.617</td></tr><tr><td>3</td><td>0.072</td><td>90.07</td><td>0.618</td></tr></table>

Sensitivity to the routing exponent. Table 5 shows that the routing strategy is insensitive to the choice of α. Across $\alpha \in \{ 1 , 2 , 3 \}$ , all three metrics remain nearly unchanged, indicating that the proposed reliability score is robust to moderate variations in the completeness weighting.

Additional qualitative comparisons. Figs. 7 and 8 provide additional qualitative comparisons on the MIDI-3D-Front and ARSG-110K test sets, respectively. Across diverse scenes, our method produces object configurations with more consistent orientations and relative placements, resulting in scene layouts that better agree with the reference observations. These results further demonstrate the benefit of geometry-grounded pose reasoning for improving scene-level spatial coherence under multi-view observations.

## A.5 LIMITATION AND FUTURE WORK

Although SPOON improves scene-level coherence, individual objects are still generated independently rather than through an explicit multi-instance generative process. Extending the current object-wise formulation toward joint multi-instance generation could enable stronger coupling between object generation and scene-level spatial organization. In addition, the effectiveness of

![](images/95bbbeb7e8d0ef49b4e496af8f13e326d1edfbf275cf6ac201955d7a44ef2048.jpg)  
Figure 7: Qualitative comparison on the MIDI-3D-Front test set.  
geometry-grounded reasoning depends on the quality of the underlying multi-view reconstruction and may be affected by inaccurate or incomplete geometry.

![](images/6a25098498159db4beb0dc0a6c2c986adbbec75679effcf6858ba8bc810edf4a.jpg)  
Figure 8: Qualitative comparison on the ARSG-110K test set.