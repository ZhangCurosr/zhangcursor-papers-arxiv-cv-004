# RRG-SLAM: Real-time Reflection-aware Gaussian SLAM for Indoor Scenes

YONG LIU, State Key Laboratory of CAD&CG, Zhejiang University, China

KEYANG YE, State Key Laboratory of CAD&CG, Zhejiang University, China

ZHEXI PENG, State Key Laboratory of CAD&CG, Zhejiang University, China

RUIXIAN MEI, State Key Laboratory of CAD&CG, Zhejiang University, China

KUN ZHOU, State Key Laboratory of CAD&CG, Zhejiang University, China

TIANJIA SHAO, State Key Laboratory of CAD&CG, Zhejiang University, China

![](images/90bb28772224034a6a79fbeaff93cc901ccf228500fc01fd554263f726403c94.jpg)  
Ours

![](images/f5b6a62ee6bbbe09e014af8d0aa7971fe005dfef9ddb17483e0f0feaccfa514a.jpg)  
Novel Views

![](images/0c04931a7acde642ee6071dc5e79fcaef451eb6359b1195df91f74751231a25c.jpg)  
GPS-SLAM

![](images/5459d6c73b231dc55e82377219b1476058e8f8fc6d08fd9fd47e92c1ae1e2954.jpg)  
Novel Views

![](images/db740e67c3d077f82c7d9d432861189062f98aa3868ed4e1e69d1c5a7ac9e7ee.jpg)  
RTG-SLAM

![](images/a78c2ece44de9e3e96819bde26c6716b3ea38fe4eccc7a94f07fc8bd948b0d5e.jpg)  
Novel Views

![](images/84bfc20759408a968ee0db294aabb8a2540ecea2231ff8a5b8ce4db7e5c189c9.jpg)  
GS-ICP SLAM

![](images/6fec0a30524d0d2d5e5069e74aabe216d943486069afc28243706847250cd51a.jpg)  
Novel Views

Fig. 1. A meeting room containing strong planar reflections from a screen and polished tables. For clearer visualization, we crop out the ceiling and glass doors. We compare our method with state-of-the-art Gaussian SLAM approaches (GPS-SLAM [Peng et al. 2025], RTG-SLAM [Peng et al. 2024], and GS-ICP SLAM [Ha et al. 2024]) for mesh reconstruction and novel-view synthesis. Regions with large drifting errors are highlighted with red boxes. Our method enables more stable tracking and improves rendering quality in reflective environments.

We introduce the first real-time reflection-aware Gaussian SLAM system for indoor scenes. The system features a reflection-aware TSDF-Gaussian hybrid representation that explicitly separates difuse scene appearance from reflection components. The base scene is modeled by a TSDF volume and a set of base Gaussians capturing geometry and difuse appearance, while

Authors’ Contact Information: Yong Liu, State Key Laboratory of CAD&CG, Zhejiang University, China, yongliu6@zju.edu.cn; Keyang Ye, State Key Laboratory of CAD&CG, Zhejiang University, China, yekeyang@zju.edu.cn; Zhexi Peng, State Key Laboratory of CAD&CG, Zhejiang University, China, zhexipeng@zju.edu.cn; Ruixian Mei, State Key Laboratory of CAD&CG, Zhejiang University, China, mrx@zju.edu.cn; Kun Zhou, State Key Laboratory of CAD&CG, Zhejiang University, China, kunzhou@acm.org; Tianjia Shao, State Key Laboratory of CAD&CG, Zhejiang University, China, tjshao@ zju.edu.cn.

planar reflections are represented by reflection Gaussian groups associated with detected reflective planes. The rendering is performed in three passes: TSDF raycasting first yields surface color, depth, plane IDs and reflection masks; base Gaussians are then rendered order-independently with depth culling and combined with the TSDF output to form the base image; finally, under the guidance of the plane ID map, reflection Gaussians from diferent reflection groups are rasterized only into their corresponding planar regions to generate the reflection image, which is subsequently composited with the base image via the reflection mask to produce the final output. For online reconstruction, our system first estimates the camera pose through reflectionaware tracking to suppress interference of reflection-dominated regions. It then identifies reflective planes using geometric, semantic, and temporal cues, and fuses the observations into the augmented TSDF volume with reflection-aware attributes. Afterwards the base and reflection Gaussians are initialized, optimized, and pruned online to maintain both reconstruction quality and eficiency. Experiments on a variety of datasets show that our method outperforms existing SLAM systems in reconstruction quality, track ing robustness, and novel-view rendering for indoor environments with reflections, while preserving real-time performance.

CCS Concepts: • Computing methodologies → Point-based models;   
Reconstruction.

Additional Key Words and Phrases: RGB-D SLAM, 3D Gaussian Splatting, Real-Time Reconstruction

## ACM Reference Format:

Yong Liu, Keyang Ye, Zhexi Peng, Ruixian Mei, Kun Zhou, and Tianjia Shao. 2026. RRG-SLAM: Real-time Reflection-aware Gaussian SLAM for Indoor Scenes. 1, 1 (September 2026), 16 pages. https://doi.org/10.1145/nnnnnnn. nnnnnnn

## 1 INTRODUCTION

Real-time 3D indoor scene reconstruction is a fundamental problem in computer vision and robotics, with broad applications in robotics and augmented reality. Unlike outdoor settings, indoor environments often contain complex appearance efects caused by artificial lighting and planar reflective surfaces, such as polished wooden floors, tiled floors, and display screens. Reconstructing such indoor environments with strong planar reflections in real time is particularly challenging. Classical RGB-D SLAM systems [Dai et al. 2017; Mur-Artal et al. 2015; Newcombe et al. 2011] mainly focus on geometry reconstruction, without modeling the photorealistic appearance. Recent 3D-Gaussian-based RGB-D SLAMs [Ha et al. 2024; Keetha et al. 2024; Kerbl et al. 2023; Peng et al. 2024, 2025] have enabled real-time 3D indoor reconstruction with highfidelity appearance. However, these methods all assume that indoor environments consist of purely difuse materials. When reflection phenomena exist in indoor scenes, their 3D Gaussian representations struggle to accurately capture such reflected content, leading to apparent artifacts in the reconstructed reflective regions. More critically, the strong reflections can violate photometric consistency and corrupt image features, making photometric- and feature-based tracking unreliable. As a result, existing SLAM methods often exhibit large drifting errors in indoor scenes with strong reflections (see Fig. 1 for example).

In this paper, we introduce a reflection-aware scene representation for planar reflections, which explicitly separates base scene appearance from reflection components, while remaining compatible with online SLAM. The representation consists of a truncated signed distance field (TSDF) volume and a set of base 3D Gaussians capturing geometry and difuse appearance, as well as a set of reflection groups modeling plane-induced reflective efects. Specifically, the TSDF volume models the scene geometry and coarse difuse appearance, while base Gaussians refine the difuse appearance beyond the TSDF volume. A reflection group comprises a reflective plane and its associated reflection Gaussians. This design is motivated by the symmetry of planar reflections: reflected content can be interpreted as virtual scene behind the corresponding plane and represented by Gaussians in this virtual space. The TSDF is further augmented with reflection-related attributes: plane association recording the detected plane each voxel belongs to, reflection strength marking reflective regions, and temporal color variance capturing color changes accumulated over repeated observations. These attributes support reflection rendering and reflective plane detection.

The rendering process follows a three-pass design. The first pass raycasts the TSDF volume to obtain the TSDF color, surface depth, plane ID map, and reflection mask. The plane ID map assigns each valid pixel to a detected planar region, while the reflection mask labels reflective pixels for compositing. The second pass renders base Gaussians with order-independent accumulation and depth culling using the raycast surface depth, and combines their accumulated color with the TSDF color to produce the base color image. The third pass uses our customized rasterizer to render the reflection Gaussians from all reflection groups in a single pass. Guided by the raycast plane ID map, each group contributes only to the pixels assigned to its associated plane, producing the reflection color image without interference across groups. The final image is obtained by compositing the base and reflection colors using the reflection mask.

To make this representation practical for online reconstruction, we need to reliably identify reflective planes. A single cue is insuficient: depth reveals planar structures but cannot determine whether they are reflective, while image semantics suggests potentially reflective surfaces but does not verify reflection efects. We therefore combine geometry and semantics with a temporal cue from reconstruction. Specifically, we use CAPE [Proença and Gao 2018] to segment planar regions from the input depth map, and Grounded SAM2 [Liu et al. 2024b; Ravi et al. 2024] to detect predefined categories that are likely to exhibit reflections in the input color image. Their association gives candidate reflective planes, which are further verified using a raycast color variance map whose voxel-wise statistics are updated during TSDF fusion. Planes with suficiently large high-variance regions are marked as reflective in the current detection window. To avoid noisy single-frame detections, we aggregate recent decisions in a sliding window and accept a plane as reflective only when it is repeatedly identified across multiple frames.

Our reflection-aware representation also benefits tracking by explicitly excluding reflection-dominated regions. We first estimate an initial pose using point-to-plane ICP, which aligns the current depth observation to the raycast TSDF surface and provides a robust geometric initialization. Given this pose, we render the reflection color image from our representation and threshold its intensity to obtain a binary tracking mask that identifies reflection-dominated pixels. In the second stage, we perform feature-based pose refinement on the unmasked regions, excluding reflection-dominated pixels from feature extraction. This allows the tracker to benefit from reliable image features while avoiding reflection-induced interference.

Building on the proposed reflection-aware representation and its associated online operations, we construct the first RGB-D Gaussian SLAM system designed for indoor scenes with planar reflections. Given an input RGB-D frame, our system performs reflection-aware tracking to estimate the camera pose, identifies reflective planes using geometric, semantic, and temporal cues, and fuses the observations into the TSDF volume. The scene is further refined through Gaussian reconstruction, where base and reflection Gaussians are initialized with carefully designed strategies, optimized online, and pruned to keep the representation compact. We evaluate our method on self-captured reflective scenes and a synthetic reflective dataset, and compare it with state-of-the-art Gaussian-based SLAM methods. Our system achieves better reconstruction quality, more robust tracking, and higher-quality novel-view rendering in scenes with reflections while maintaining real-time performance. We also evaluate on public RGB-D benchmarks, including ScanNet++, Replica, and TUM-RGBD, to demonstrate its advantages in scenes with reflections as well as its compatibility with general indoor scenes. In summary, our contributions are:

• We propose the first real-time RGB-D Gaussian SLAM system designed for indoor scenes with planar reflections.

• We propose a novel hybrid representation combining TSDF, base Gaussians, and reflection Gaussians, integrated with real-time reflection detection, reflection-aware tracking, and online optimization to enable robust online reconstruction of scenes with reflections.

• We demonstrate through extensive experiments that our system achieves state-of-the-art reconstruction quality and robust tracking accuracy on both public and self-captured datasets while maintaining real-time performance.

## 2 RELATED WORK

Classical RGB-D SLAM. Classical RGB-D SLAM systems mainly difer in the tracking strategies they employ, which typically rely on geometric alignment, photometric consistency, or sparse image features. For instance, KinectFusion-style methods [Kähler et al. 2015; Newcombe et al. 2011; Nießner et al. 2013; Steinbrucker et al. 2013] estimate camera poses via frame-to-model ICP, while systems such as ElasticFusion [Whelan et al. 2015] and BundleFusion [Dai et al. 2017] incorporate both geometric and photometric terms in a joint optimization framework. Feature-based approaches [Endres et al. 2012; Labbé and Michaud 2019; Sumikura et al. 2019], such as ORB-SLAM [Campos et al. 2021; Mur-Artal et al. 2015; Mur-Artal and Tardós 2017], instead track sparse keypoints and refine poses through bundle adjustment. While these methods achieve strong performance in general environments, they do not explicitly account for reflection-induced appearance variations. In scenes with prominent planar reflections, view-dependent efects can degrade the reliability of photometric alignment and feature matching, leading to unstable pose estimation. In contrast, our method introduces a reflection-aware tracking strategy that explicitly identifies and suppresses reflection-dominated regions during refinement, improving robustness in such challenging scenarios.

Radiance Field-based RGB-D SLAM. Radiance-field-based RGB-D SLAM extends scene reconstruction beyond purely geometric mapping toward photorealistic scene representation and novel-view synthesis. Early NeRF-based SLAM systems, such as iMAP [Sucar et al. 2021], NICE-SLAM [Zhu et al. 2022], ESLAM [Johari et al. 2023], Co-SLAM [Wang et al. 2023], and Point-SLAM [Sandström et al. 2023], demonstrated that jointly optimizing camera poses and a neural radiance field can significantly improve rendering quality compared with classical RGB-D SLAM systems. However, these methods rely on repeated MLP evaluation and volumetric rendering, which are computationally expensive and therefore limit their eficiency for online operation. More recently, 3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] introduced an explicit radiance representation based on Gaussian primitives together with diferentiable rasterization, enabling much faster rendering while maintaining high visual fidelity. Motivated by these advantages, a growing number of Gaussian-based SLAM systems [Ha et al. 2024; Hu et al. 2024; Keetha et al. 2024; Peng et al. 2024, 2025; Su et al. 2025; Yan et al.

2024; Yugay et al. 2023] have adopted 3DGS as the scene representation. Compared with NeRF-based approaches, these methods are substantially more eficient and better suited for online reconstruction, while also achieving higher-quality novel-view synthesis than classical RGB-D SLAM. Among them, GPS-SLAM [Peng et al. 2025] is particularly relevant to our work. Instead of relying solely on Gaussians, it combines a TSDF volume with sparse Gaussians in a hybrid representation, where the TSDF provides coarse geometry and appearance, and the Gaussians capture finer details. By adopting a sorting-free two-pass rendering strategy, GPS-SLAM achieves very high frame rates without sacrificing rendering quality. This design makes it especially attractive for real-time RGB-D SLAM, and also provides a strong foundation for further extending the representation toward more challenging scene efects.

Despite these advances, existing radiance-field-based RGB-D SLAM methods generally assume that scene appearance can be represented by standard radiance primitives, such as MLP-based fields or SH-parameterized Gaussians. In strongly reflective scenes, however, this assumption becomes insuficient. Reflections introduce complex view-dependent appearance that is dificult to model accurately, which in turn afects both pose tracking and scene mapping. To address this limitation, we extend the hybrid TSDF-Gaussian representation with explicit reflection modeling. Our representation consists of a TSDF volume for coarse geometry and color, a set of base Gaussians for refining the stable scene appearance, and a set of plane-associated reflection Gaussians for modeling reflective content induced by planar surfaces. This design makes the representation better suited to indoor reflective scenes, improving both tracking robustness and mapping quality, while also benefiting novel-view synthesis.

Ofline Reflection Reconstruction. Reflection reconstruction has also been widely studied in ofline settings. Existing methods can be broadly categorized into environment-map-based, ray-tracingbased, and geometry-aware approaches. Environment-map-based methods [Boss et al. 2021; Jiang et al. 2024; Ye et al. 2024; Zhang et al. 2022, 2021], such as 3DGS-DR [Ye et al. 2024] and Gaussian-Shader [Jiang et al. 2024], model reflections through environment lookups, but their distant-light assumption is often insuficient for indoor scenes with strong near-field efects. Ray-tracing-based methods [Gao et al. 2024; Verbin et al. 2024; Xie et al. 2025] model reflection more faithfully through secondary ray sampling, but their computational cost is too high for real-time operation. Geometry aware methods, such as Mirror-NeRF [Zeng et al. 2023], Mirrorgaussian [Liu et al. 2024a], and TR-Gaussian [Liu et al. 2026], exploit planar symmetry to reconstruct virtual content behind reflective surfaces and achieve high-quality results. Among these, geometryaware methods are most relevant to our work because they explicitly exploit planar structure for reflection modeling. However, these methods are designed for ofline reconstruction rather than online SLAM. They typically rely on dense multi-view observations and iterative optimization over the full image set, making them dificult to apply directly in a real-time setting. To bridge this gap, we design an online reflection-aware reconstruction pipeline for RGB-D SLAM. In particular, we introduce online reflective plane detection together with a tailored reflection Gaussian initialization strategy, which significantly reduces the dificulty of reflection modeling and enables explicit reconstruction from sequential input while preserving real-time performance.

## 3 Reflection-Aware TSDF-Gaussian Hybrid Representation

## 3.1 Modeling

We propose a reflection-aware scene representation that explicitly separates base scene appearance from reflection components, while remaining compatible with online SLAM. Our representation consists of two parts: a base scene representation that captures geometry and difuse appearance, and a set of reflection groups that model plane-induced reflective efects. This design is particularly suitable for indoor environments, where reflections are often associated with dominant planar surfaces such as tables, screens, and floors.

Base Representation. Our base representation builds upon GPS-SLAM [Peng et al. 2025]. Following its hybrid design, we use a TSDF volume S to represent scene geometry and coarse difuse appearance and a set of base Gaussians $\mathcal { G } _ { b }$ to refine difuse appearance. Each TSDF voxel p stores its signed distance $d ( \mathbf { p } )$ and color c(p). We further augment each voxel with three reflection-aware attributes: a color variance accumulator $S _ { c } ( \mathbf { p } )$ for capturing temporal appearance variation, a reflection strength �(p) marking the reflective regions, and a plane ID $\pi ( \mathbf { p } )$ that associates the voxel with a planar surface. The base Gaussians are defined as $\mathcal { G } _ { b } = \{ \mathbf { p } _ { i } ^ { b } , \mathbf { s } _ { i } ^ { b } , \mathbf { r } _ { i } ^ { b } , \bar { \sigma _ { i } ^ { b } } , \mathbf { S } \mathbf { H } _ { i } ^ { b } \} _ { i = 1 } ^ { N _ { b } }$ , parameterized by position, scale, rotation, opacity, and SH coeficients.

Reflection Groups. To explicitly model planar reflections, we organize reflective appearance into plane-level reflection groups. During scanning, planar surfaces are detected online and represented as $\pi _ { i } = ( { \bf n } _ { i } , b _ { i } )$ with a unique ID �. The plane ID is further propagated to TSDF voxels during fusion, allowing the reconstructed geometry to maintain consistent plane associations. Each detected reflective plane defines a reflection group $\{ \pi _ { i } ^ { r } , \mathcal { G } _ { i } ^ { r } \}$ , which consists of the reflective plane $\boldsymbol { \pi } _ { i } ^ { r }$ and its associated reflection Gaussian set $\mathcal { G } _ { i } ^ { r }$ . The complete reflection Gaussian set is denoted by:

$$
G _ { r } = \bigcup _ { i } \mathcal { G } _ { i } ^ { r }\tag{1}
$$

This design is motivated by the symmetry of planar reflection. Specifically, reflected appearance can be interpreted as virtual content behind the reflective plane and represented using Gaussians in this virtual space. Reflection Gaussians share the same parameterization as base Gaussians, while each Gaussian additionally stores the ID of its associated reflective plane:

$$
\mathcal { G } _ { i } ^ { r } = \{ \mathbf { p } _ { j } ^ { r } , \mathbf { s } _ { j } ^ { r } , \mathbf { r } _ { j } ^ { r } , \sigma _ { j } ^ { r } , \mathbf { S } \mathbf { H } _ { j } ^ { r } , \ell _ { j } ^ { r } \} _ { j = 1 } ^ { N _ { i } ^ { r } } ,\tag{2}
$$

where $\ell _ { j } ^ { r }$ denotes the reflective plane associated with the Gaussian. We also maintain a reflective plane lookup table $L _ { r e f }$ to record whether each detected plane has been identified as reflective.

Organizing reflections at the plane level provides two advantages. First, it disentangles reflective appearance from the base scene representation, preventing reflection-induced artifacts from contaminating difuse reconstruction. Second, it enables eficient planeconditioned rendering, where each reflection group contributes only to pixels associated with its corresponding plane.

## 3.2 Rendering

As illustrated in Fig. 2, our rendering pipeline consists of three passes: TSDF raycasting, base Gaussian rendering, and reflection Gaussian rendering.

In the first pass, we raycast the TSDF volume to obtain a TSDF color image $\mathbf { C } _ { t }$ and a depth map $\mathbf { D } _ { t }$ . In addition, we extract a plane ID map I and a reflection strength map $\mathbf { R } _ { r a w } .$ The plane ID map assigns each visible pixel to its associated planar surface, while $\mathbf { R } _ { r a w }$ measures the likelihood of reflective efects at each pixel. We further binarize $\mathbf { R } _ { r a w }$ using a threshold of 0.5 to obtain the reflection mask R.

In the second pass, we render the base Gaussians $\mathcal { G } _ { b }$ . Since base Gaussians correspond to directly observed scene surfaces, we perform depth culling against the raycast TSDF surface depth $\mathbf { D } _ { t } .$ The Gaussian attributes are accumulated in an order-independent manner similar to [Ye et al. 2025]:

$$
\mathcal { A } _ { b } [ \mathbf { x } ] ( \mathbf { u } ) = \sum _ { i = 1 } ^ { K } \mathbb { 1 } \left( d _ { i } ^ { b } < \mathbf { D } _ { t } ( \mathbf { u } ) + \boldsymbol { \epsilon } \right) \alpha _ { i } ^ { b } ( \mathbf { u } ) \mathbf { x } _ { i } ^ { b } ,\tag{3}
$$

where

$$
\alpha _ { i } ^ { b } ( { \mathbf { u } } ) = \sigma _ { i } ^ { b } \exp \left( - \frac { ( { \mathbf { u } } - \hat { { \mathbf { p } } } _ { i } ^ { b } ) ^ { T } ( \Sigma _ { i } ^ { b } ) ^ { - 1 } ( { \mathbf { u } } - \hat { { \mathbf { p } } } _ { i } ^ { b } ) } { 2 } \right) .\tag{4}
$$

Here u denotes the pixel coordinate, 1(·) is the indicator function, and � is the truncation distance [Peng et al. 2025]. For the �-th Gaussian, $d _ { i } ^ { b }$ and $\alpha _ { i } ^ { b }$ denote its projected depth and weight, respectively. $\Sigma _ { i } ^ { b }$ is the covariance matrix of the projected Gaussian in image space, and $\hat { \mathbf { p } } _ { i } ^ { b }$ is the projected Gaussian center. The term $\mathbf { x } _ { i } ^ { b }$ denotes an arbitrary Gaussian attribute. $\mathrm { B y }$ substituting $\mathbf { x } _ { i } ^ { b }$ with the view-dependent color $\mathbf { c } _ { i } ^ { b } .$ scalar unity 1, or Gaussian depth $d _ { i } ^ { b } { : }$ we obtain the accumulated color $\mathbf { C } _ { g } = \mathcal { A } _ { b } [ \mathbf { c } ]$ , accumulated weight $\mathbf { W } _ { q } = \mathcal { A } _ { b } [ 1 ]$ , and Gaussian depth map $\mathbf { D } _ { g } = \mathcal { A } _ { b } [ d _ { i } ]$ , respectively. The final base color image and depth map are computed as

$$
\mathbf { C } _ { b } = \frac { \mathbf { C } _ { t } + \mathbf { C } _ { g } } { 1 + \mathbf { W } _ { g } } , \qquad \mathbf { D } _ { b } = \frac { \mathbf { D } _ { t } + \mathbf { D } _ { g } } { 1 + \mathbf { W } _ { g } } .\tag{5}
$$

In the third pass, we render all reflection Gaussians in a single pass to obtain the reflection image $\mathbf { C } _ { r }$ . To ensure that each reflection group contributes only to its associated reflective plane, we implement plane-conditioned rendering as a per-pixel filtering operation during rasterization:

$$
\mathcal { A } _ { r } [ \mathbf { x } ] ( \mathbf { u } ) = \sum _ { i = 1 } ^ { K } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { r } ( \mathbf { u } ) ) \alpha _ { i } ^ { r } ( \mathbf { u } ) \mathbf { x } _ { i } ^ { r } ,\tag{6}
$$

where

$$
\alpha _ { i } ^ { r } ( \mathbf { u } ) = \mathbb { 1 } \ \left( \ell _ { i } ^ { r } = \mathbf { I } ( \mathbf { u } ) \right) \sigma _ { i } ^ { r } \exp \left( - \frac { ( \mathbf { u } - \hat { \mathbf { p } } _ { i } ^ { r } ) ^ { T } ( \Sigma _ { i } ^ { r } ) ^ { - 1 } ( \mathbf { u } - \hat { \mathbf { p } } _ { i } ^ { r } ) } { 2 } \right) .\tag{7}
$$

A reflection Gaussian contributes to a pixel only when its associated plane ID $\ell _ { i } ^ { r }$ matches the plane ID I assigned to that pixel by TSDF raycasting. By substituting $\mathbf { x } _ { i } ^ { r }$ with the reflection color $\mathbf { c } _ { i } ^ { r }$ or scalar unity 1, we obtain the reflection color image $\mathbf { C } _ { r } = \mathcal { A } _ { r } \dot { [ \mathbf { c } ^ { r } ] }$ and reflection weight map $\mathbf { W } _ { r } = \mathcal { A } _ { r } [ 1 ]$ , respectively. Finally, the final rendered image is obtained by compositing the base image and reflection image using the reflection mask:

![](images/52c1a514a11f2470e586294848096e4349fe899c0d1e0baa1891797952be0b5c.jpg)

Fig. 2. Overview of our reflection-aware scene representation and three-pass rendering pipeline. The representation decomposes the scene into a base scene�,� � = � ���� + ����� �� + ���� �Extraction Refinement (TSDF volume and base Gaussians) capturing geometry and difuse appearance, and reflection groups modeling planar reflections. Rendering proceeds inColor Variance Multi-cue Validation: <sub>three</sub> <sub>passes:</sub> <sub>(1)</sub> <sub>TSDF</sub> <sub>raycasting</sub> <sub>produces</sub> <sub>color,</sub> <sub>depth,</sub> <sub>plane</sub> <sub>ID,</sub> <sub>and</sub> <sub>reflection</sub> <sub>mask;</sub> <sub>(2)</sub> <sub>base</sub> <sub>Gaussians</sub> <sub>are</sub> <sub>rendered</sub> <sub>with</sub> <sub>depth</sub> <sub>culling</sub> <sub>and</sub> <sub>combined</sub> <sub>with</sub>Planar Geometry + TSDF color to produce the base color; (3) reflection Gaussians are rendered conditioned on their associated plane IDs and composited using the reflection mask to generate the final image. This decomposition enables explicit modeling of reflective content while maintaining eficient rendering of the base scene  
![](images/6102a5c8adf13c5e0bdc58d932cc266101462152e5beffe17912beebe5cff143.jpg)  
Fig. 3. Overview of the proposed reflection-aware RGB-D SLAM pipeline. Given an RGB-D stream, the system performs reflection-aware tracking, reflective plane identification, and online map reconstruction. The tracking module first estimates an initial pose using ICP on depth-derived 3D points, and then refines the pose using reliable color features selected by the rendered tracking mask. The reflective plane identification module combines geometric plane segmentation, semantic segmentation, and TSDF color variance to detect global reflective planes and generate a predicted reflection mask. Finally, the online map reconstruction performs augmented TSDF fusion and Gaussian optimization.

$$
{ \bf C } = { \bf C } _ { b } + { \bf R } \cdot { \bf C } _ { r } .\tag{8}
$$

## 4 Online Reconstruction Process

As illustrated in Fig. 3, our system takes an RGB-D stream as input and performs online reflection-aware reconstruction with a hybrid TSDF-Gaussian representation. The method is organized into two main parts: TSDF reconstruction and Gaussian reconstruction. In TSDF reconstruction, we preprocess each RGB-D frame, estimate camera poses using reflection-aware tracking, identify reflective planes from geometric, semantic, and temporal cues, and fuse observations into an augmented TSDF volume with reflection-aware attributes, as described in Sec. 4.1. In Gaussian reconstruction, we incrementally add base and reflection Gaussians, followed by online optimization and pruning to maintain reconstruction quality and compactness, as described in Sec. 4.2.

For real-time performance, tracking and TSDF fusion are performed for each incoming frame, while reflective plane identification and Gaussian optimization are triggered periodically at lower frequencies. This design enables stable tracking, reflection-aware map construction, and high-quality online reconstruction in reflective indoor scenes.

## 4.1 TSDF reconstruction

4.1.1 Input preprocessing. Given the input color image $\mathbf { C } _ { k } ^ { g t }$ and depth map $\mathbf { D } _ { k } ^ { g t }$ at frame $k ,$ we first compute a local vertex map $\mathbf { V } _ { k } ^ { l }$ and a local normal map $\mathbf { N } _ { k } ^ { l }$ from the depth observation. Using the estimated camera pose, we then transform them into the global coordinate system, yielding a global vertex map $\mathbf { V } _ { k } ^ { g }$ and a global normal map $\mathbf { N } _ { k } ^ { g }$

4.1.2 Reflection-aware tracking. In reflective scenes, view-dependent appearance caused by planar reflections violates photometric consistency and corrupts image features, making standard feature-based tracking unreliable. To address this, we adopt a two-stage reflectionaware tracking strategy. In the first stage, an initial pose is estimated by aligning the current depth observation to the TSDF raycast surface using point-to-plane ICP:

$$
E ( \pmb { \xi } ) = \sum \left\| \left( \mathbf { T } _ { g , k } ^ { i c p } \mathbf { V } _ { k } ^ { l } ( \mathbf { u } ) - \mathbf { V } _ { k - 1 } ^ { * , g } ( \hat { \mathbf { u } } ) \right) \cdot \mathbf { N } _ { k - 1 } ^ { * , g } ( \hat { \mathbf { u } } ) \right\| ,\tag{9}
$$

where $\xi$ is the Lie algebra representation of the transformation $\boldsymbol { \mathrm { T } } _ { g , k } ^ { i c p } :$ u denotes a pixel in the current frame, and $\hat { \mathbf { u } } = \mathrm { p r o j } \left( \mathbf { K } \mathbf { T } _ { k - 1 , k } \mathbf { V } _ { k } ^ { l } ( \mathbf { u } ) \right)$ is the projection of $\mathbf { V } _ { k } ^ { l } ( { \mathbf { u } } )$ into the previous frame. Here, proj(·) denotes the projection operator, $\mathbf { V } _ { k } ^ { l } ( { \mathbf { u } } )$ is the current vertex in the local camera frame, and $\mathbf { V } _ { k - 1 } ^ { * , g } ( \hat { \mathbf { u } } )$ and $\mathbf { N } _ { k - 1 } ^ { * , g } ( \hat { \mathbf { u } } )$ are the corresponding raycast vertex and normal in the global frame. Minimizing this energy yields an initial pose estimate $\mathrm { T } _ { q , k } ^ { i c p }$

In the second stage, feature-based refinement is performed using ORB features, with the ICP-estimated $\mathbf { T } _ { g , k } ^ { i c p }$ as the initial pose. The reflection component is rendered from the current reflection Gaussians $\mathcal { G } _ { r }$ at $\mathbf { T } _ { g , k } ^ { \mathrm { i c p } }$ to obtain a reflection color image $\mathbf { C } _ { k } ^ { r }$ . Averaging its RGB channels and thresholding the intensity map at 0.1 produces a binary tracking mask $\mathbf { R } _ { k } ^ { \mathrm { t r a c k } }$ that identifies pixels dominated by reflections. During ORB feature extraction [Mur-Artal and Tardós 2017], only pixels with $\mathbf { R } _ { k } ^ { \mathrm { t r a c k } } ( \mathbf { u } ) = 0$ are considered, preventing unreliable reflective regions from corrupting tracking. The pose is then refined using a feature-based optimization backend adapted from ORB-SLAM2 [Mur-Artal and Tardós 2017], producing the final pose estimate $\mathbf { T } _ { g , k }$

This two-stage strategy combines the geometric stability of ICP with reflection-aware feature refinement, enabling robust tracking even in strongly reflective indoor scenes.

4.1.3 Reflective plane identification. Reliable reflective plane detection is essential for reflection-aware tracking, rendering, and Gaussian reconstruction. In indoor scenes, planar geometry and semantic cues alone are insuficient: depth identifies planar structures, but cannot determine whether they produce reflections, while semantic segmentation highlights potentially reflective objects but cannot confirm reflection efects. To robustly identify reflective planes, we combine geometric, semantic, and temporal cues. Since semantic segmentation and planar analysis are computationally expensive, reflective plane identification is performed periodically every $N _ { p } = 1 0$ frames rather than for every incoming frame.

Specifically, we first generate reflective plane candidates by combining geometric and semantic cues. We apply CAPE [Proença and Gao 2018], a depth-based planar segmentation method, to the input depth map and extract a plane segmentation map. The detected local planes are then matched to the global plane set maintained by the system, producing a global plane ID map $\mathbf { I } _ { k } ^ { g } .$ In parallel, we apply Grounded SAM2 [Liu et al. 2024b; Ravi et al. 2024] to the input color image to obtain a semantic segmentation map ${ \bf Q } _ { k }$ . Taking advantage of its open-vocabulary capability, we predefine a set of object categories that are likely to exhibit specular reflections, such as floors, tables, and TVs. A geometric plane is regarded as a reflective-plane candidate if it is associated with one of these semantic categories. In our implementation, each geometric plane is associated with the semantic mask that has the largest overlap with the plane region.

We then validate these candidates using temporal color variance. During TSDF fusion, each voxel p maintains running statistics of voxel colors across the input sequence, and the corresponding variance is stored in $S _ { c } ( \mathbf { p } )$ using Welford-style online updates [Welford 1962]. At detection time, we raycast the TSDF to obtain a color variance map $\mathbf { C } _ { k } ^ { v a r }$ . For each candidate plane, we compute the fraction of image pixels associated with the plane whose temporal color variance exceeds the threshold $\delta _ { v } = 0 . 0 1$ . If this fraction exceeds $\delta _ { \mathrm { a r e a } } = 0 . 0 5$ , the plane is marked as reflective for the current detection step.

To improve robustness, we do not accept a reflective plane based on a single detection. Instead, for each plane, we maintain its reflec tive detection history over the most recent five windows. A plane is inserted into the reflective plane lookup table $L _ { \mathrm { r e f } }$ only if it is identified as reflective in more than three of these windows. After reflective planes are confirmed, we combine their associated seman tic masks to form a predicted reflection mask for the current frame. This mask is then fused into the TSDF volume as a reflection-aware attribute, enabling the system to mark potential reflective regions for subsequent reconstruction and rendering.

4.1.4 Augmented TSDFfusion. We perform TSDF fusion to up date voxel-wise SDF values and average colors. For frames without reflective plane identification, we only carry out standard TSDF fusion. When reflective plane identification is triggered at frame $k ,$ we additionally update reflection-aware voxel attributes. The color variance of each voxel, $S _ { c } ( \mathbf { p } )$ , is updated incrementally using a Welford-style online algorithm applied to the channel-averaged color from each new observation, capturing how much the voxel’s appearance varies over time. The plane ID is fused from the global plane map to preserve associations with planar structures, while the reflection strength $r ( \mathbf { p } )$ is updated using a sliding average over recent predicted reflection masks, providing a stable estimate of whether a voxel belongs to a reflective region. Together, these augmented attributes provide essential cues for subsequent reflection detection, reflection-aware tracking, and the initialization of reflection Gaussians. Detailed formulas and implementation specifics are provided in the supplementary material.

## 4.2 Gaussian reconstruction

4.2.1 Gaussian densification. We perform Gaussian densification separately for the base Gaussians $\mathcal { G } _ { b }$ and the reflection Gaussians $\mathcal { G } _ { r }$ . In both cases, we first identify candidate pixels for adding Gaussians, then sample a subset of these pixels, and finally initialize new Gaussians from the sampled locations. For notational clarity, at frame $k ,$ we denote the rendered color image by $\mathbf { C } _ { k }$ , the base Gaussian weight map by $\mathbf { W } _ { k } ^ { b }$ , and the reflection Gaussian weight map by $\mathbf { W } _ { k } ^ { r } .$

For the base Gaussians $\mathcal { G } _ { b }$ , the pixels with high color error and low Gaussian coverage are selected as candidates for Gaussian adding:

$$
\mathbf { M } _ { \widetilde { \mathbf { u } } _ { b } } = \{ \widetilde { \mathbf { u } } _ { b } | | \mathbf { C } _ { k } ^ { g t } ( \widetilde { \mathbf { u } } _ { b } ) - \mathbf { C } _ { k } ( \widetilde { \mathbf { u } } _ { b } ) \ | > \delta _ { b } \ \mathrm { a n d } \ W _ { k } ^ { b } ( \widetilde { \mathbf { u } } _ { b } ) < \delta _ { w , b } \} ,\tag{10}
$$

where $\mathbf { C } _ { k } ^ { g t } ( \widetilde { \mathbf { u } } _ { b } )$ and ${ \bf C } _ { k } ( \widetilde { { \bf u } } _ { b } )$ denote the input and rendered color images, used to compute the per-pixel color reconstruction error, and $W _ { k } ^ { b } ( \widetilde { \mathbf { u } } _ { b } )$ is the base Gaussian weight at the pixel, indicating the current coverage by existing base Gaussians. Each candidate pixel is then backprojected into 3D using the observed depth to initialize the Gaussian position on the surface, while all remaining attributes—including scale, rotation, opacity, and SH coeficients—are initialized according to the standard rules of GPS-SLAM [Peng et al. 2025].

For reflection Gaussians $\mathcal { G } _ { r }$ , candidate pixels are selected from regions marked as reflective that also exhibit high temporal color variance, large color reconstruction error, and low reflection Gaussian coverage:

$$
\begin{array} { r } { \mathbf { M } _ { \widetilde { \mathbf { u } } _ { r } } = \Big \{ \widetilde { \mathbf { u } } _ { r } \ | \ \mathbf { R } _ { k } ( \widetilde { \mathbf { u } } _ { r } ) = 1 , \ | \mathbf { C } _ { k } ^ { g t } ( \widetilde { \mathbf { u } } _ { r } ) - \mathbf { C } _ { k } ( \widetilde { \mathbf { u } } _ { r } ) | > \delta _ { r } , } \\ { \mathbf { C } _ { k } ^ { v a r } ( \widetilde { \mathbf { u } } _ { r } ) > \delta _ { v } , \ \mathrm { a n d } \ W _ { k } ^ { r } ( \widetilde { \mathbf { u } } _ { r } ) < \delta _ { w , r } \Big \} , } \end{array}\tag{11}
$$

where ${ \bf R } _ { k } ( \widetilde { { \bf u } } _ { r } )$ denotes the binary reflection mask that identifies reflective regions, $\mathbf { C } _ { k } ^ { v a r } ( \widetilde { \mathbf { u } } _ { r } )$ represents the color variance obtained by raycasting the TSDF, and $W _ { k } ^ { r } ( \widetilde { \mathbf { u } } _ { r } )$ is the current reflection Gaussian weight, indicating the existing coverage at that pixel. Unlike base Gaussians, reflection Gaussians cannot be initialized directly from observed depth because they represent virtual content behind the reflective plane. For each sampled candidate pixel, the Gaussian position is initialized by uniformly sampling 10 points along the viewing ray starting from the ray-plane intersection and extending within a depth range of 2–8 m. This strategy allows reflection Gaussians to represent virtual reflected content behind the reflective plane without requiring explicit reflected geometry. All remaining attributes are initialized in the same manner as for base Gaussians, and each newly added reflection Gaussian is assigned the corresponding plane ID.

4.2.2 Gaussian optimization. To preserve online eficiency, Gaussian optimization is triggered periodically every $N _ { o }$ frames. After adding base and reflection Gaussians to the map, we perform online optimization using a subset of frames. The optimization uses $N _ { l o c a l }$ frames evenly sampled from the observations collected since the previous optimization step to eficiently refine newly observed regions, along with $N _ { g l o b a l }$ randomly sampled historical keyframes to provide broader viewpoint coverage, which helps stabilize the optimization of reflection Gaussians across viewpoints. The keyframe set is maintained using a standard motion-based heuristic: a new keyframe is inserted when either the relative rotation to the last keyframe exceeds $\delta _ { \mathrm { a n g l e } }$ or the relative translation exceeds $\delta _ { \mathrm { m o v e } } .$

We jointly optimize the Gaussian groups $\mathcal { G } _ { b }$ and $\mathcal { G } _ { r }$ by minimizing the following loss function:

$$
\mathcal { L } = \mathcal { L } _ { c o l o r } + \lambda _ { d } \mathcal { L } _ { d e p t h } + \lambda _ { g } \mathcal { L } _ { g r a d } ,\tag{12}
$$

where the regularization term is defined as

$$
\mathcal { L } _ { g r a d } = | \nabla { \mathbf { C } } _ { r } | _ { 1 } .\tag{13}
$$

Here, the color term $\mathcal { L } _ { c o l o r }$ combines an $L _ { 1 }$ loss and an SSIM loss between the rendered image $\mathbf { C }$ and the input RGB image $\mathbf { C } _ { g t } ,$ while the depth term $\mathcal { L } _ { d e p t h }$ uses an $L _ { 1 }$ loss between the rendered depth $\mathbf { D } _ { b }$ and the input depth $\mathbf { D } _ { g t }$ . We set the depth weight $\lambda _ { d }$ to 0.2 and the reflection smoothness weight $\lambda _ { g }$ to 0.05 in all experiments. The smoothness term $\mathcal { L } _ { g r a d }$ , computed via the Sobel operator $\nabla ,$ encourages spatially coherent reflection image $\mathbf { C } _ { r } .$ which facilitates the decomposition of difuse and reflective components in practice.

4.2.3 Gaussian pruning. After each optimization round, we prune both base and reflection Gaussians according to their scale and opacity. Gaussians with excessively large scales tend to oversmooth local structures, while those with very small scales or low opacity contribute little to rendering and often correspond to unstable or redundant primitives. Removing such Gaussians helps maintain a compact and stable representation during long-term online reconstruction. Detailed threshold settings are provided in the supplementary material.

## 5 Experiments

## 5.1 Experiment setup

Implementation Details. The proposed SLAM system is implemented and evaluated on a desktop equipped with an AMD 9950X3D CPU and an NVIDIA RTX 4090 GPU. The core SLAM framework is implemented in ${ \mathrm { C } } { + + }$ based on GPS-SLAM [Peng et al. 2025]. We develop custom CUDA kernels for rasterization and backpropagation. The CAPE [Proença and Gao 2018] plane detection module is also implemented in ${ \mathrm { C } } { + } { + }$ by modifying the original CAPE repository. In addition, the Grounded SAM2 [Liu et al. 2024b; Ravi et al. 2024] segmentation module is implemented in Python and communicates with the C++ SLAM system through shared memory for eficient data exchange. For more details, please refer to the supplementary material.

![](images/5370973873e8863a534fa980251b6b4f789f9c85b651d22d0b9fa41285a74f84.jpg)  
Fig. 4. Qualitative comparison of rendering quality across multiple reflective indoor scenes. The first two columns show the original GT images and the cropped regions for clearer visualization of reflective areas. Our method consistently produces sharper and more stable reflections. Red boxes indicate regions where baseline methods (GPS-SLAM, GauS-SLAM, GS-ICP SLAM, RTG-SLAM, and SplaTAM) sufer from drifting or fail to reconstruct reflections accurately. Note that SplaTAM runs out of memory (OOM) on Meeting Room, so no result is reported for it.

![](images/df8a7ea01faeb67b0ece7c4477b6b3228953dc9a2f0a173ef9a701bbe5abd2a1.jpg)  
Fig. 5. Visualization of rendering components on four representative scenes. From left to right, we show the base color image, reflection mask, weighted reflection color image, final rendered image, and ground-truth image. The base color mainly captures difuse appearance, while the reflection mask localizes reflective regions and the weighted reflection color image shows the contribution of reflection Gaussians. The final image combines the base and reflection components to beter reproduce reflective appearance.

Datasets. We primarily evaluate our method on a self-constructed dataset with noticeable reflection efects, denoted as RIRD (Reflection Indoor RGB-D). RIRD consists of six real indoor scenes captured with an Azure Kinect RGB-D camera and three synthetic reflective indoor scenes rendered using Blender. We refer to the real subset as RIRD-Real and the synthetic subset as RIRD-Syn. RIRD-Real is mainly used for qualitative evaluation of novel-view rendering and reconstruction quality, while RIRD-Syn is used to quantitatively evaluate tracking and geometric reconstruction accuracy in reflec tive environments.

We also evaluate our method on two reflective ScanNet++ [Yeshwanth et al. 2023] scenes to assess novel-view rendering and geometric reconstruction in real-world settings. The selected scenes are a5859cfd40 and 8e6ff28354. In addition, we use Replica [Straub et al. 2019] and TUM-RGBD [Sturm et al. 2012] as standard RGB-D benchmarks. Replica provides synthetic indoor scenes without noticeable reflections, while TUM-RGBD provides real RGB-D sequences with accurate ground-truth trajectories for tracking evaluation.

Baselines. We compare our method with several state-of-the-art 3DGS-based RGB-D SLAM systems, including SplaTAM [Keetha et al. 2024], RTG-SLAM [Peng et al. 2024], GauS-SLAM [Su et al. 2025], GS-ICP SLAM [Ha et al. 2024], and GPS-SLAM [Peng et al. 2025]. SplaTAM employs a Gaussian-based tracking and mapping pipeline via diferentiable rendering. RTG-SLAM proposes a compact Gaussian representation to significantly reduce memory usage and computational cost. GauS-SLAM adopts a 2D Gaussian surfel-based representation for robust tracking and accurate dense reconstruction. GS-ICP SLAM combines Gaussian-based scene representation with ICP-based tracking to achieve real-time pose estimation. GPS-SLAM introduces a hybrid Gaussian-SDF representation and enables ultra-fast RGB-D SLAM. In addition, we include ORB-SLAM2 [Mur-Artal and Tardós 2017] and BundleFusion [Dai et al. 2017] as representatives of classical SLAM systems to evaluate tracking performance in reflective indoor environments. We reproduce the results using their oficial implementations and run all experiments on the same computer.

## 5.2 Evaluation

5.2.1 Rendering quality. We evaluate rendering quality with a par ticular focus on reflection-related appearance. For novel-view synthesis, we conduct experiments on the synthetic subset of RIRD and on ScanNet++. In each RIRD-Syn scene, we uniformly sample 25 viewpoints and perturb their poses to render novel-view images, while for ScanNet++ we select one test view every eight frames. Input-view rendering is evaluated on the real subset of RIRD and Replica, following the protocols used in prior Gaussian-based SLAM methods [Ha et al. 2024; Keetha et al. 2024; Peng et al. 2024].

Quantitative results are summarized in Table 1. On the reflective datasets, our method achieves the best novel-view synthesis quality on RIRD-Syn and ScanNet++, and also attains the best input-view rendering results on RIRD-Real. On Replica, which does not contain noticeable reflections, our method achieves rendering quality comparable to other methods. These results show that our reflectionaware representation improves rendering quality in reflective scenes while remaining compatible with general indoor scenes.

![](images/4af6c4f76716b9e4973fbdbb38d33b917d648e1ccd701fcf3361d36bfdd6785c.jpg)  
Fig. 6. Comparison of reconstructed meshes in reflective indoor scenes (Meeting Room and Private Room). Our method produces more complete and accurate reconstructions in regions containing a glossy tiled floor and a TV screen. Baseline methods, including GPS-SLAM [Peng et al. 2025], RTG-SLAM [Peng et al. 2024], GS-ICP SLAM [Ha et al. 2024], GauS-SLAM [Su et al. 2025], BundleFusion [Dai et al. 2017], and ORB-SLAM2 [Mur-Artal and Tardós 2017], exhibit drifting or misalignment in these reflective regions.

Table 1. Rendering quality comparison on RIRD-Syn, ScanNet++, RIRD-Real, and Replica. We report PSNR, SSIM, and LPIPS. RIRD-Syn and ScanNet++ are evaluated under the novel-view seting, while RIRD-Real and Replica are evaluated on training views. ${ \mathsf { R I R D - S y n } }$ , ScanNet++, and RIRD-Real contain reflective indoor scenes, whereas Replica is used to evaluate performance on general indoor scenes without reflections.
<table><tr><td>Method</td><td>Metric</td><td>RIRD-Syn</td><td>ScanNet++</td><td>RIRD-Real</td><td>Replica</td></tr><tr><td rowspan="3">SplaTAM</td><td>PSNR↑</td><td>18.66</td><td>23.85</td><td>22.69</td><td>34.14</td></tr><tr><td>SSIM↑</td><td>0.634</td><td>0.834</td><td>0.820</td><td>0.936</td></tr><tr><td>LPIPS↓</td><td>0.422</td><td>0.241</td><td>0.355</td><td>0.142</td></tr><tr><td rowspan="3">RTG-SLAM</td><td>PSNR↑</td><td>17.81</td><td>22.37</td><td>25.16</td><td>34.04</td></tr><tr><td>SSIM↑</td><td>0.650</td><td>0.783</td><td>0.855</td><td>0.924</td></tr><tr><td>LPIPS↓</td><td>0.444</td><td>0.299</td><td>0.359</td><td>0.183</td></tr><tr><td rowspan="3">GauS-SLAM</td><td>PSNR↑</td><td>17.02</td><td>25.17</td><td>24.77</td><td>38.85</td></tr><tr><td>SSIM↑</td><td>0.497</td><td>0.800</td><td>0.815</td><td>0.974</td></tr><tr><td>LPIPS↓</td><td>0.428</td><td>0.287</td><td>0.330</td><td>0.068</td></tr><tr><td rowspan="3">GS-ICP SLAM</td><td>PSNR↑</td><td>22.70</td><td>24.73</td><td>24.93</td><td>33.00</td></tr><tr><td>SSIM↑</td><td>0.798</td><td>0.867</td><td>0.864</td><td>0.926</td></tr><tr><td>LPIPS↓</td><td>0.272</td><td>0.194</td><td>0.289</td><td>0.116</td></tr><tr><td rowspan="3">GPS-SLAM</td><td>PSNR↑</td><td>26.79</td><td>24.59</td><td>24.46</td><td>37.25</td></tr><tr><td>SSIM↑</td><td>0.871</td><td>0.836</td><td>0.849</td><td>0.955</td></tr><tr><td>LPIPS↓</td><td>0.208</td><td>0.255</td><td>0.322</td><td>0.108</td></tr><tr><td rowspan="3">Ours</td><td>PSNR↑</td><td>30.29</td><td>26.01</td><td>30.82</td><td>37.29</td></tr><tr><td>SSIM↑</td><td>0.909</td><td>0.869</td><td>0.921</td><td>0.961</td></tr><tr><td>LPIPS↓</td><td>0.191</td><td>0.185</td><td>0.218</td><td>0.102</td></tr></table>

We present qualitative comparisons on RIRD-Real, RIRD-Syn, and ScanNet++ in Fig. 4. Existing methods do not explicitly model reflective appearance, and as a result, their reconstructions exhibit artifacts in reflective regions. On surfaces with relatively high roughness, existing methods often produce over-smoothed textures, as seen on the wooden floors in the Living Room 0 and Living Room 1 scenes. On smoother surfaces with stronger and clearer reflections, the rendered appearance is often noisy, as observed on the tiled floor in the Kitchen scene reconstructed by SplaTAM, and on the TV screen in the Meeting Room scene reconstructed by GauS-SLAM and RTG-SLAM. In contrast, our method better models reflective appearance across surfaces with varying roughness, producing more coherent and visually plausible results. We further illustrate our reconstruction and novel-view synthesis results on the real scenes captured in RIRD in Fig. 7. To better visualize the high-quality reflective rendering produced by our method, we refer readers to the supplementary video.

Reflection decomposition. To explain the improved rendering quality over the baselines, we visualize the decomposed rendering components in Fig. 5. For each example, we show the base color image $\mathbf { C } _ { b } ,$ the reflection mask R, the weighted reflection color R · $\mathbf { C } _ { r } ,$ , the final composited result C, and the ground truth image $\mathbf { C } _ { g t }$ . The base color mainly captures difuse appearance, while the reflection color focuses on view-dependent reflective content associated with planar surfaces. By separating planar reflections from difuse appearance, our method avoids mixing reflective content with the base scene and therefore produces cleaner and more accurate renderings.

5.2.2 Tracking accuracy. We report absolute trajectory error (ATE) on RIRD-Syn, Replica, and TUM-RGBD, as shown in Table 2. Our method achieves the lowest ATE on the reflective RIRD-Syn dataset, demonstrating its robustness to reflection-induced tracking interference. On Replica and TUM-RGBD, our method also performs competitively, indicating that the proposed reflection-aware tracking strategy remains compatible with general indoor scenes.

Mesh reconstruction under estimated trajectories. To more clearly illustrate the improvement in tracking performance under reflective conditions, we select two real scenes from RIRD-Real and visualize reconstructed meshes, as shown in Fig. 6. For a fair comparison, we reconstruct a mesh for each method using TSDF fusion with its estimated poses and input depth maps. In the Meeting Room scene, GPS-SLAM and GS-ICP SLAM rely on ICP-based geometric alignment and exhibit noticeable drift in this large-scale environment, especially in regions with sparse geometric features. Classical methods such as BundleFusion and ORB-SLAM2 also exhibit noticeable drift around the TV region in Meeting Room due to unreliable photometric cues. RTG-SLAM, which employs ORB-SLAM2 as its back-end, shows similar drift patterns. GauS-SLAM uses both color and depth constraints for pose optimization, so view-dependent reflections can disturb the photometric term and lead to pose misalignment. In contrast, our reflection-aware method combines masked ORB feature refinement with ICP initialization, efectively suppressing reflection-induced interference and producing more stable and accurate tracking in reflective regions.

![](images/add78ceff1bf89cdd958bde0843e0cd7630b1d9c309303c6a244352f2e8f914a.jpg)

![](images/56c603c7f7c0295231bb1d03846780f0b1c7bf605eb1edd4a33e47e56f360cf2.jpg)  
Fig. 7. Reconstruction and novel-view synthesis results on six real RIRD-Real scenes. For each scene, we show the reconstructed scene overview and zoomed-in views of reflective regions. Colored boxes link the overview to the corresponding zoomed-in renderings. Our method produces coherent scene-scale reconstruction and visually consistent reflective appearance in real reflective indoor environments.

Table 2. Tracking accuracy comparison on RIRD-Syn, Replica, and TUM-RGBD. We report the absolute trajectory error (ATE), where lower values indicate beter tracking accuracy. RIRD-Syn contains reflective synthetic scenes, while Replica and TUM-RGBD are used to evaluate tracking performance on general indoor scenes.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=2>RIRD-Syn Replica</td><td rowspan=1 colspan=1>TUM</td></tr><tr><td rowspan=1 colspan=1>ORB-SLAM2</td><td rowspan=1 colspan=1>22.26</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>1.27</td></tr><tr><td rowspan=3 colspan=1>BundleFusionSplaTAMRTG-SLAM</td><td rowspan=1 colspan=1>8.52</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>1.53</td></tr><tr><td rowspan=1 colspan=1>36.15</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>1.81</td></tr><tr><td rowspan=1 colspan=1>10.71</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>1.13</td></tr><tr><td rowspan=2 colspan=1>GauS-SLAMGS-ICP SLAM</td><td rowspan=1 colspan=1>41.81</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>1.42</td></tr><tr><td rowspan=1 colspan=1>16.62</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>2.33</td></tr><tr><td rowspan=1 colspan=1>GPS-SLAM</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>2.75</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>1.06</td></tr></table>

5.2.3 Geometry quality. Following NICE-SLAM [Zhu et al. 2022], we evaluate geometric reconstruction quality on ScanNet++ and RIRD-Syn using the metrics of Accuracy, Completion, Accuracy Ratio (<3 cm), and Completion Ratio (<3 cm). Since ScanNet++ is not designed for SLAM and exhibits abrupt changes between consecutive frames, we use ground-truth poses for mapping in experiments on this dataset. For each method, we extract the geometry following the strategy used in its original paper. For SplaTAM, RTG-SLAM, and GS-ICP SLAM, we randomly sample a fixed number of Gaussian points for evaluation. For GauS-SLAM, we follow the original implementation and reconstruct a scene mesh by applying TSDF fusion to the rendered depth maps. For GPS-SLAM and our method, we reconstruct the scene mesh by extracting surfaces from the TSDF volume using marching cubes. On ScanNet++, with provided camera poses, our method achieves geometry quality on par with existing Gaussian-based SLAM methods. On RIRD-Syn, our method shows a clear advantage over the other methods. We attribute this improvement mainly to our more accurate tracking.

5.2.4 Eficiency analysis. Table 4 summarizes the runtime, memory usage, Gaussian count, and rendering quality of all methods on the Ofice 0 scene from Replica and the Meeting Room scene from RIRD-Real. On the Replica Ofice 0 scene, our method incurs only a negligible performance overhead compared with GPS-SLAM, while achieving comparable rendering quality. This indicates that the proposed reflection-aware representation remains eficient in general indoor scenes without noticeable reflections. On the reflective Meeting Room scene, GPS-SLAM inserts substantially more Gaussians, reaching 2.37M primitives. As a result, its runtime drops to 51.0 FPS. In contrast, although our method jointly optimizes both base and reflection Gaussians, the proposed representation models reflective appearance more efectively. It requires only 0.64M Gaussians and achieves the highest frame rate of 70.5 FPS, while also producing substantially higher rendering quality.

Table 3. Geometry reconstruction quality comparison on RIRD-Syn and ScanNet++. We report accuracy, completion, accuracy ratio, and completion ratio. The ratio metrics measure the percentage of points with errors below 3 cm.
<table><tr><td>Method</td><td>Dataset</td><td>Acc↓</td><td>Acc Ratio↑</td><td>Com↓</td><td>Com Ratio↑</td></tr><tr><td rowspan="2">SplaTAM</td><td>ScanNet++</td><td>3.10</td><td>65.55</td><td>3.03</td><td>61.21</td></tr><tr><td>RIRD-Syn</td><td>6.32</td><td>44.64</td><td>4.74</td><td>50.52</td></tr><tr><td rowspan="2">RTG-SLAM</td><td>ScanNet++</td><td>1.41</td><td>91.84</td><td>1.74</td><td>87.18</td></tr><tr><td>RIRD-Syn</td><td>7.85</td><td>62.69</td><td>5.77</td><td>59.09</td></tr><tr><td rowspan="2">GauS-SLAM</td><td>ScanNet++</td><td>3.92</td><td>74.13</td><td>1.49</td><td>95.84</td></tr><tr><td>RIRD-Syn</td><td>60.01</td><td>63.22</td><td>45.58</td><td>64.22</td></tr><tr><td rowspan="2">GS-ICP SLAM</td><td>ScanNet++</td><td>0.87</td><td>99.94</td><td>1.09</td><td>96.76</td></tr><tr><td>RIRD-Syn</td><td>4.93</td><td>84.69</td><td>5.12</td><td>75.96</td></tr><tr><td rowspan="2">GPS-SLAM</td><td>ScanNet++</td><td>1.24</td><td>95.21</td><td>1.11</td><td>98.50</td></tr><tr><td>RIRD-Syn</td><td>3.4</td><td>90.25</td><td>1.90</td><td>94.35</td></tr><tr><td rowspan="2">Ours</td><td>ScanNet++</td><td>1.25</td><td>95.18</td><td>1.10</td><td>98.53</td></tr><tr><td>RIRD-Syn</td><td>0.95</td><td>94.74</td><td>0.96</td><td>97.56</td></tr></table>

Table 4. Runtime, memory, and rendering quality comparison on Replica Ofice 0 and RIRD-Real Meeting Room. We report mapping time per frame, FPS, Gaussian count, GPU memory usage, and PSNR. Replica Ofice 0 represents a general indoor scene, while Meeting Room contains strong planar reflections.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=4>Mapping         GaussianFPS           Memory (MB) PSNR/Frame (ms)         Count</td></tr><tr><td rowspan=2 colspan=1>SplaTAM</td><td rowspan=2 colspan=1>ReplicaRIRD-Real</td><td rowspan=2 colspan=2>3066.7    0.3   6.22M一        1       一</td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>OOM</td></tr><tr><td rowspan=2 colspan=1>RTG-SLAM</td><td rowspan=2 colspan=1>ReplicaRIRD-Real</td><td rowspan=2 colspan=1>59.9    16.682.4    12.12</td><td rowspan=1 colspan=1>0.82M</td><td rowspan=1 colspan=1>2755</td><td rowspan=1 colspan=1>37.78</td></tr><tr><td rowspan=1 colspan=1>1M</td><td rowspan=1 colspan=1>10656</td><td rowspan=1 colspan=1>22.42</td></tr><tr><td rowspan=2 colspan=1>GauS-SLAM</td><td rowspan=2 colspan=1>ReplicaRIRD-Real</td><td rowspan=2 colspan=1>314.9    3.1376.1    2.6</td><td></td><td rowspan=1 colspan=1>15745</td><td rowspan=1 colspan=1>42.40</td></tr><tr><td rowspan=1 colspan=1>10.05M</td><td rowspan=1 colspan=1>23891</td><td rowspan=1 colspan=1>17.89</td></tr><tr><td rowspan=2 colspan=1>GS-ICP SLAM</td><td rowspan=2 colspan=1>ReplicaRIRD-Real</td><td rowspan=1 colspan=1>5.4    184.8</td><td rowspan=1 colspan=1>1.72M</td><td rowspan=1 colspan=1>3677</td><td rowspan=1 colspan=1>37.79</td></tr><tr><td rowspan=1 colspan=1>16.7    59.8</td><td rowspan=1 colspan=1>2.41M</td><td rowspan=1 colspan=1>7528</td><td rowspan=1 colspan=1>13.53</td></tr><tr><td rowspan=2 colspan=1>GPS-SLAM</td><td rowspan=2 colspan=1>ReplicaRIRD-Real</td><td rowspan=1 colspan=1>2.4     408.1</td><td rowspan=1 colspan=1>0.11M</td><td rowspan=1 colspan=1>3603</td><td rowspan=1 colspan=1>40.91</td></tr><tr><td rowspan=1 colspan=1>19.5    51.0</td><td rowspan=1 colspan=1>2.37M</td><td rowspan=1 colspan=1>17719</td><td rowspan=1 colspan=1>18.42</td></tr><tr><td rowspan=2 colspan=1>Ours</td><td rowspan=2 colspan=1>ReplicaRIRD-Real</td><td rowspan=1 colspan=1>2.8    358.8</td><td rowspan=1 colspan=1>0.10M</td><td rowspan=1 colspan=1>5714</td><td rowspan=1 colspan=1>41.30</td></tr><tr><td rowspan=1 colspan=1>14.18    70.5</td><td rowspan=1 colspan=1>0.64M</td><td rowspan=1 colspan=1>14017</td><td rowspan=1 colspan=1>28.35</td></tr></table>

Runtime breakdown. Table 5 reports the per-frame runtime of major components on the RIRD-Real Meeting Room scene. Semantic segmentation is the most expensive pre-mapping module, taking 7.04 ms per frame. The total pre-mapping runtime, including pose tracking, plane detection, semantic segmentation, TSDF fusion, and other preprocessing overhead, is 10.12 ms, which is still lower than the 14.18 ms used by Gaussian mapping. This indicates that Gaussian mapping remains the main runtime bottleneck. Since these premapping modules can run in parallel with Gaussian mapping, they have limited impact on the system speed.

Rendering speed. Table 6 reports the per-pass rendering time and the number of Gaussians used in each pass on the RIRD-Real Meeting

![](images/3661a419d6965e2c637e597a0eaef536bee32fe9c9c5f73ce59558759cc7ea79.jpg)  
Input RGB-D

![](images/e0d77f70aa347988f0c88debb8970c30793216fbb1fff0986f5b039ee3a23fa2.jpg)  
Plane Segmentation Map

![](images/04f27512a9a8ef2d27b25890796c01aeec8cae9dd23fb87c806248b0d55c5b14.jpg)  
Semantic Segmentation Map

![](images/e27470e8148185c051950bf14cbb99f200d6a88b57254020e1c76c43ea546338.jpg)  
TSDF Variance Map

![](images/965099c75b2d3495a642fcab4fb3453cc9b25bc0d2ce318145dbe2208fb5f4a6.jpg)  
Predicted Reflection Mask  
Fig. 8. Visualization of reflective plane identification. From left to right, we show the input RGB-D frame, plane segmentation map, semantic segmentation map, TSDF variance map, and predicted reflection mask. Diferent colors in the plane segmentation map indicate diferent planar regions. In the semantic segmentation map, TV ■, table ■, and floor ■ indicate the semantic categories. The TSDF variance map is colorized according to temporal color variance where brighter colors indicate higher variance. By combining geometric, semantic, and temporal cues, our method predicts reliable reflective regions for reflection-aware reconstruction.

Table 5. Runtime breakdown on the RIRD-Real Meeting Room scene. We report the average per-frame runtime of pose tracking, plane detection, semantic segmentation, TSDF fusion, the total pre-mapping stage, and Gaussian mapping.
<table><tr><td>Module</td><td>Time / Frame (ms)</td></tr><tr><td>Pose Tracking</td><td>0.85</td></tr><tr><td>Plane Detection</td><td>1.88</td></tr><tr><td>Semantic Segmentation</td><td>7.04</td></tr><tr><td>TSDF Fusion</td><td>0.19</td></tr><tr><td>Pre-mapping Stage</td><td>10.12</td></tr><tr><td>Gaussian Mapping</td><td>14.18</td></tr></table>

Room scene. Despite using three rendering passes, our reflectionaware representation keeps the total number of Gaussians moderate, with 0.30M base Gaussians and 0.34M reflection Gaussians in this scene. As a result, the combined rendering time of all passes is only 3.22 ms per frame, corresponding to 310.0 FPS. This demonstrates that the proposed decomposition enables eficient novel-view synthesis without introducing excessive Gaussian redundancy.

Table 6. Rendering performance breakdown on the RIRD-Real Meeting Room scene. We report the per-frame runtime, FPS, and number of Gaussians for each rendering stage.
<table><tr><td>Rendering Stage</td><td>Time / Frame (ms)</td><td>FPS</td><td>Num. of Gaussians</td></tr><tr><td>TSDF Raycasting</td><td>0.72</td><td>1389.83</td><td></td></tr><tr><td>Base Gaussian Rendering</td><td>1.05</td><td>950.00</td><td>0.30M</td></tr><tr><td>Reflection Gaussian Rendering</td><td>1.39</td><td>716.19</td><td>0.34M</td></tr><tr><td>Total</td><td>3.22</td><td>310.00</td><td>0.64M</td></tr></table>

5.2.5 Reflective plane identification. Fig. 8 visualizes representative results of reflective plane identification. From left to right, we show the input RGB-D frames, plane segmentation maps, semantic segmentation maps, TSDF color variance maps, and predicted reflection masks. The examples show that combining geometric, semantic, and temporal cues enables the system to localize reflective regions on planar surfaces.

![](images/ad37740190b17a891488c6ad52d3f2a31afe226d292fa149b61ec2e6d78323fa.jpg)  
Fig. 9. Ablation study of plane-conditioned reflection rendering on the Room 2 scene from RIRD-Syn. Without plane grouping, reflections become dim and blurry. Plane-associated reflection groups produce more accurate reflected content, while our single-pass plane-filtered rendering achieves comparable quality to multi-pass masked compositing with higher eficiency.

## 5.3 Ablation Studies

5.3.1 Reflection grouping and rendering. To evaluate the efectiveness of our reflection representation, we ablate two key designs: plane-associated reflection grouping, which decomposes reflections according to their supporting reflective planes, and single-pass plane filtering, which restricts each reflection group to its corresponding planar region during rendering. We conduct the ablation on the Room 2 scene from RIRD-Syn. We compare three variants: w/o plane grouping, which represents the whole scene with a single set of reflection Gaussians and renders them as in vanilla 3DGS; w/o single-pass filtering, which keeps plane-associated reflection groups but renders them separately and composites the results using plane masks; and the full model, which renders all reflection groups in a single pass with per-pixel plane filtering. As shown in Fig. 9, w/o plane grouping produces dimmer and blurrier reflections. Both w/o single-pass filtering and the full model produce more accurate reflection colors using plane-associated groups, but the former requires additional rendering passes and is much slower, as shown in Table 7. These results show that reflection grouping improves reflection fi delity, while single-pass plane filtering preserves this quality with substantially higher rendering eficiency.

Table 7. Ablation study of reflection grouping and rendering on RIRD-Syn Room 2. We report base rendering time, reflection rendering time, total rendering FPS, and PSNR. The results show that plane grouping improves reflection quality, while our single-pass filtering maintains high rendering eficiency.
<table><tr><td>Ablation</td><td>Base Render</td><td>Refl. Render (ms)↓</td><td>Render FPS↑</td><td>PSNR↑</td></tr><tr><td rowspan="4">w/o Plane Grouping w/o Single-Pass Filtering Full Model</td><td>(ms)↓ 3.37</td><td>0.33</td><td>265.4</td><td>26.55</td></tr><tr><td>3.32</td><td>1.39</td><td>209.2</td><td>26.90</td></tr><tr><td>3.36</td><td>0.29</td><td>268.8</td><td>27.09</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

![](images/01d05988602d3a214972b974feb41495ba3007d3bd1d8bc28e469f66d194f2cb.jpg)  
Fig. 10. Ablation study of reflection-aware tracking. We compare three variants: w/o ORB refinement, which uses ICP only; w/o tracking mask, which applies ORB refinement without masking reflection-dominated regions; and the Full model, which uses masked ORB refinement. The results show that unmasked ORB refinement can introduce drift in reflective regions, while the full model produces cleaner reconstruction.

5.3.2 Reflection-aware tracking. We ablate the proposed reflectionaware tracking strategy on the Room 1 scene from RIRD-Syn and the Meeting Room scene from RIRD-Real. We compare three variants: w/o ORB refinement, which uses ICP only; w/o tracking mask, which applies ORB-based pose refinement without masking reflectiondominated regions; and the full model, which uses masked ORBbased pose refinement. On RIRD-Syn, adding ORB-based pose refinement without masking increases the ATE from 0.56 to 1.21, indicating that unreliable features from reflection-dominated regions can degrade pose estimation. With reflection-aware masking, the ATE is reduced to 0.31, showing that suppressing reflective regions allows feature-based pose refinement to improve tracking robustness. Since the synthetic data are relatively idealized and show less visible mesh artifacts, we further evaluate the reconstructed meshes on the real Meeting Room scene in Fig. 10. The full model produces cleaner reconstruction, whereas the variant without reflection masking exhibits visible drift.

5.3.3 Efect ofRegularization on Reflection Color. We evaluate the efect of the reflection color regularization term $\mathcal { L } _ { g r a d }$ on the RIRD-Real Living Room 0 scene by comparing our full model with a variant without this term. As shown in Table 9, removing $\mathcal { L } _ { g r a d }$ degrades rendering quality. Fig. 11 further shows that, without this regularization on the reflection color, the base color becomes blurrier, and the final composited image loses high-frequency details. This is because the base and reflection components are less constrained during joint optimization and tend to interfere with each other. The color error visualization further confirms that $\mathcal { L } _ { g r a d }$ reduces such interference and improves the final rendering quality.

Table 8. Quantitative ablation study of reflection-aware tracking on RIRD-Syn Room 1. We report ATE for tracking accuracy and geometry metrics including accuracy, completion, and their corresponding ratios.
<table><tr><td>Ablation</td><td>ATE↓</td><td>Acc.↓</td><td>Acc. Ratio↑</td><td>Com.↓</td><td>Com. Ratio↑</td></tr><tr><td>w/o ORB Refinement</td><td>0.56</td><td>0.95</td><td>97.54</td><td>0.96</td><td>98.64</td></tr><tr><td>w/o Tracking Mask</td><td>1.21</td><td>3.19</td><td>86.72</td><td>0.93</td><td>92.52</td></tr><tr><td>Full Model</td><td>0.31</td><td>0.93</td><td>98.86</td><td>0.81</td><td>99.82</td></tr></table>

![](images/c747baa4b9633e4e5a225311d4bc9509069501ff883cd8c4cbd0c00e790ab02a.jpg)  
Fig. 11. Ablation study of the reflection color regularization term on RIRD-Real Living Room 0. We compare the results without and with $\mathcal { L } _ { g r a d }$ . From left to right, we show the base color image, weighted reflection color image, final composited image, and color error map. With $\mathcal { L } _ { g r a d } ,$ the base and reflection components are beter constrained during optimization, resulting in lower color errors and improved final rendering quality.

Table 9. Ablation study of the reflection color regularization term on RIRD-Real Living Room 0. We compare the rendering quality with and without $\mathcal { L } _ { g r a d }$
<table><tr><td>Ablation</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>w/o  $\overline { { \mathcal { L } _ { g r a d } } }$ </td><td>28.36</td><td>0.888</td><td>0.231</td></tr><tr><td> $\mathbf { w } / \mathop { \mathcal { L } _ { g r a d } ^ { \sim } }$ </td><td>30.38</td><td>0.919</td><td>0.224</td></tr></table>

5.3.4 Efect ofTemporal Color Variance Threshold. We analyze the influence of the temporal color variance threshold $\delta _ { v }$ used for reflective plane verification on the Room 2 scene from RIRD-Syn. By varying $\delta _ { v } ,$ we evaluate its impact on novel-view rendering quality. As shown in Table 10, our method achieves stable performance across a moderate range of threshold values, demonstrating that the proposed temporal reflection verification is robust to threshold selection. When $\delta _ { v }$ becomes too large, reflective regions may be missed, leading to degraded rendering quality.

## 6 Conclusion

We present the first real-time RGB-D Gaussian SLAM system designed for indoor scenes with planar reflections. We introduce a reflection-aware TSDF-Gaussian hybrid representation that explicitly decomposes the scene into a base component and planeassociated reflection components, enabling more faithful modeling of reflective appearance during online reconstruction. To render this representation eficiently, we develop a three-pass pipeline that combines TSDF raycasting, base Gaussian rendering, and planeconditioned reflection Gaussian rasterization. For online reconstruction, our system integrates reflection-aware tracking, reflective plane identification from geometric, semantic, and temporal cues, augmented TSDF fusion, and online optimization of both base and reflection Gaussians. Experiments on self-captured reflective scenes, a synthetic reflective dataset, and public RGB-D benchmarks demonstrate that our method achieves superior reconstruction quality, more robust tracking, and higher-quality novel-view rendering in indoor environments with reflections while maintaining real-time performance.

Table 10. Ablation study of the temporal color variance threshold $\delta _ { v }$ on RIRD-Syn Room 2. We report PSNR, SSIM, and LPIPS under diferent threshold values to evaluate the sensitivity of reflective-region verification.
<table><tr><td>Threshold</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>δ=0.002</td><td>26.23</td><td>0.872</td><td>0.225</td></tr><tr><td> $\delta _ { v } = 0 . 0 0 5$ </td><td>26.37</td><td>0.881</td><td>0.214</td></tr><tr><td> $\delta _ { v } = 0 . 0 1$ </td><td>26.47</td><td>0.883</td><td>0.211</td></tr><tr><td> $\delta _ { v } { = } 0 . 0 3$ </td><td>26.27</td><td>0.879</td><td>0.209</td></tr><tr><td> $\delta _ { v } = 0 . 0 5$ </td><td>25.65</td><td>0.873</td><td>0.253</td></tr><tr><td> $\delta _ { v } \mathrm { = } 0 . 1$ </td><td>24.06</td><td>0.864</td><td>0.243</td></tr></table>

Our method currently focuses on planar reflections and therefore cannot explicitly model reflections on curved surfaces, whose geometry and view-dependent appearance are substantially more complex. In addition, transparent objects without reliable depth measurements, such as glass, remain challenging for our RGB-D reconstruction pipeline and may lead to incomplete geometry or unstable appearance modeling. Extending reflection-aware online reconstruction to curved reflective surfaces, transparent materials, and more general non-Lambertian efects would be valuable directions for future work.

## References

Mark Boss, Raphael Braun, Varun Jampani, Jonathan T Barron, Ce Liu, and Hendrik Lensch. 2021. Nerd: Neural reflectance decomposition from image collections. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 12684– 12694.

Carlos Campos, Richard Elvira, Juan J Gómez Rodríguez, José MM Montiel, and Juan D Tardós. 2021. Orb-slam3: An accurate open-source library for visual, visual–inertial, and multimap slam. IEEE transactions on robotics 37, 6 (2021), 1874–1890.

Angela Dai, Matthias Nießner, Michael Zollhöfer, Shahram Izadi, and Christian Theobalt. 2017. Bundlefusion: Real-time globally consistent 3d reconstruction using on-the-fly surface reintegration. ACM Transactions on Graphics (ToG) 36, 4 (2017), 1.

Felix Endres, Jürgen Hess, Nikolas Engelhard, Jürgen Sturm, Daniel Cremers, and Wolfram Burgard. 2012. An evaluation of the RGB-D SLAM system. In 2012 IEEE international conference on robotics and automation. IEEE, 1691–1696.

Chen Gao, Yipeng Wang, Changil Kim, Jia-Bin Huang, and Johannes Kopf. 2024. Planar reflection-aware neural radiance fields. In SIGGRAPH Asia 2024 Conference Papers. 1–10.

Seongbo Ha, Jiung Yeon, and Hyeonwoo Yu. 2024. Rgbd gs-icp slam. In European Conference on Computer Vision. Springer, 180–197.

Jiarui Hu, Xianhao Chen, Boyin Feng, Guanglin Li, Liangjing Yang, Hujun Bao, Guofeng Zhang, and Zhaopeng Cui. 2024. Cg-slam: Eficient dense rgb-d slam in a consistent uncertainty-aware 3d gaussian field. In European Conference on Computer Vision. Springer, 93–112.

Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma. 2024. Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 5322–5332.

Mohammad Mahdi Johari, Camilla Carta, and François Fleuret. 2023. Eslam: Eficient dense slam system based on hybrid representation of signed distance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 17408–17419.

Olaf Kähler, Victor Adrian Prisacariu, Carl Yuheng Ren, Xin Sun, Philip Torr, and David Murray. 2015. Very high frame rate volumetric integration of depth images on mobile devices. IEEE transactions on visualization and computer graphics 21, 11 (2015), 1241–1250.

Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. 2024. Splatam: Splat track & map 3d gaussians for dense rgb-d slam. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21357–21366.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42, 4 (2023), 139–1.

Mathieu Labbé and François Michaud. 2019. RTAB-Map as an open-source lidar and visual simultaneous localization and mapping library for large-scale and long-term online operation. Journal offield robotics 36, 2 (2019), 416–446.

Jiayue Liu, Xiao Tang, Freeman Cheng, Roy Yang, Zhihao Li, Jianzhuang Liu, Yi Huang, Jiaqi Lin, Shiyong Liu, Xiaofei Wu, et al. 2024a. Mirrorgaussian: Reflecting 3d gaussians for reconstructing mirror reflections. In European Conference on Computer Vision. Springer, 377–393.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. 2024b. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision. Springer, 38–55.

Yong Liu, Keyang Ye, Tianjia Shao, and Kun Zhou. 2026. TR-Gaussians: High-fidelity Real-time Rendering of Planar Transmission and Reflection with 3D Gaussian Splat ting. IEEE Transactions on Visualization and Computer Graphics (2026).

Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. 2015. ORB-SLAM: A versatile and accurate monocular SLAM system. IEEE transactions on robotics 31, 5 (2015), 1147–1163.

Raul Mur-Artal and Juan D Tardós. 2017. Orb-slam2: An open-source slam system for monocular, stereo, and rgb-d cameras. IEEE transactions on robotics 33, 5 (2017), 1255–1262.

Richard A Newcombe, Shahram Izadi, Otmar Hilliges, David Molyneaux, David Kim, Andrew J Davison, Pushmeet Kohi, Jamie Shotton, Steve Hodges, and Andrew Fitzgibbon. 2011. Kinectfusion: Real-time dense surface mapping and tracking. In 2011 10th IEEE international symposium on mixed and augmented reality. Ieee, 127–136.

Matthias Nießner, Michael Zollhöfer, Shahram Izadi, and Marc Stamminger. 2013. Real time 3D reconstruction at scale using voxel hashing. ACM Transactions on Graphics (ToG) 32, 6 (2013), 1–11.

Zhexi Peng, Tianjia Shao, Yong Liu, Jingke Zhou, Yin Yang, Jingdong Wang, and Kun Zhou. 2024. Rtg-slam: Real-time 3d reconstruction at scale using gaussian splatting. In ACM SIGGRAPH 2024 Conference Papers. 1–11.

Zhexi Peng, Kun Zhou, and Tianjia Shao. 2025. Gaussian-plus-SDF SLAM: High-fidelity 3D reconstruction at 150+ fps. Computational Visual Media (2025).

Pedro F Proença and Yang Gao. 2018. Fast cylinder and plane extraction from depth cameras for visual odometry. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 6813–6820.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. 2024. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024).

Erik Sandström, Yue Li, Luc Van Gool, and Martin R Oswald. 2023. Point-slam: Dense neural point cloud-based slam. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 18433–18444.

Frank Steinbrucker, Christian Kerl, and Daniel Cremers. 2013. Large-scale multi resolution surface reconstruction from RGB-D sequences. In Proceedings ofthe IEEE International Conference on Computer Vision. 3264–3271.

Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, et al. 2019. The replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797 (2019).

Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. 2012. A benchmark for the evaluation of RGB-D SLAM systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 573–580.

Yongxin Su, Lin Chen, Kaiting Zhang, Zhongliang Zhao, Chenfeng Hou, and Ziping Yu. 2025. GauS-SLAM: Dense RGB-D SLAM with Gaussian Surfels. arXiv preprint arXiv:2505.01934 (2025).

Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davison. 2021. imap: Implicit mapping and positioning in real-time. In Proceedings of the IEEE/CVF international conference on computer vision. 6229–6238

Shinya Sumikura, Mikiya Shibuya, and Ken Sakurada. 2019. OpenVSLAM: A versatile visual SLAM framework. In Proceedings ofthe 27th ACM international conference on multimedia. 2292–2295.

Dor Verbin, Pratul P Srinivasan, Peter Hedman, Ben Mildenhall, Benjamin Attal, Richard Szeliski, and Jonathan T Barron. 2024. Nerf-casting: Improved view-dependent appearance with consistent reflections. In SIGGRAPH Asia 2024 Conference Papers. 1–10.

Hengyi Wang, Jingwen Wang, and Lourdes Agapito. 2023. Co-slam: Joint coordinate and sparse parametric encodings for neural real-time slam. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13293–13302.

Barry Payne Welford. 1962. Note on a method for calculating corrected sums of squares and products. Technometrics 4, 3 (1962), 419–420.

Thomas Whelan, Stefan Leutenegger, Renato F Salas-Moreno, Ben Glocker, and Andrew J Davison. 2015. ElasticFusion: Dense SLAM without a pose graph.. In Robotics: science and systems, Vol. 11. Rome.

Tao Xie, Xi Chen, Zhen Xu, Yiman Xie, Yudong Jin, Yujun Shen, Sida Peng, Hujun Bao, and Xiaowei Zhou. 2025. Envgs: Modeling view-dependent appearance with environment gaussian. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 5742–5751.

Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. 2024. Gs-slam: Dense visual slam with 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19595–19604.

Keyang Ye, Qiming Hou, and Kun Zhou. 2024. 3d gaussian splatting with deferred reflection. In ACM SIGGRAPH 2024 Conference Papers. 1–10.

Keyang Ye, Tianjia Shao, and Kun Zhou. 2025. When gaussian meets surfel: Ultra-fast high-fidelity radiance field rendering. ACM Transactions on Graphics (TOG) 44, 4

(2025), 1–15.

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. 2023. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 12–22.

Vladimir Yugay, Yue Li, Theo Gevers, and Martin R Oswald. 2023. Gaussian-slam: Photo-realistic dense slam with gaussian splatting. arXiv preprint arXiv:2312.10070 (2023).

Junyi Zeng, Chong Bao, Rui Chen, Zilong Dong, Guofeng Zhang, Hujun Bao, and Zhaopeng Cui. 2023. Mirror-nerf: Learning neural radiance fields for mirrors with whitted-style ray tracing. In Proceedings ofthe 31st ACM International Conference on Multimedia. 4606–4615.

Qiang Zhang, Seung-Hwan Baek, Szymon Rusinkiewicz, and Felix Heide. 2022. Diferentiable point-based radiance fields for eficient view synthesis. In SIGGRAPH Asia 2022 Conference Papers. 1–12.

Xiuming Zhang, Pratul P Srinivasan, Boyang Deng, Paul Debevec, William T Freeman, andJonathan T Barron. 2021. Nerfactor: Neural factorization ofshape and reflectance under an unknown illumination. ACM Transactions on Graphics (ToG) 40, 6 (2021), 1–18.

Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys. 2022. Nice-slam: Neural implicit scalable encoding for slam. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 12786–12796.

# Supplementary Material for RRG-SLAM: Real-time Reflection-aware Gaussian SLAM for Indoor Scenes

## A Dataset details

RIRD-Real is a real-world indoor RGB-D dataset collected by ourselves in reflective environments containing various reflective materials and objects. All frames have a resolution of 1280×720. The dataset includes diverse reflective surfaces such as glossy floors, tables, and TVs commonly found in indoor scenes. Table 1 summarizes the statistics of RIRD-Real.

Table 1. Statistics of the RIRD-Real dataset. LR0 and LR1 denote the two living-room scenes. We report the number of frames, trajectory length, and scanned area for each scene.
<table><tr><td>Statistic</td><td>LR0</td><td>LR1</td><td>Rest</td><td>Meeting</td><td>Private</td><td>Kitchen</td></tr><tr><td>Frames</td><td>1570</td><td>2080</td><td>3470</td><td>3699</td><td>1649</td><td>2670</td></tr><tr><td>Traj. Len. (m)</td><td>15.3</td><td>20.1</td><td>33.7</td><td>59.1</td><td>20.7</td><td>26.4</td></tr><tr><td>Area (m2)</td><td>19.2</td><td>15.2</td><td>41.7</td><td>102.2</td><td>21.6</td><td>36.7</td></tr></table>

RIRD-Syn is a synthetic RGB-D dataset rendered with Blender at a resolution of 1280×720, enabling quantitative evaluation of novel-view synthesis and tracking. Table 2 summarizes the statistics of RIRD-Syn. We also select two reflective scenes from Scan-Net++ [Yeshwanth et al. 2023] with strong planar reflections, denoted as a5859cfd40 and 8e6f28354, at a resolution of 876×584.

Table 2. Statistics of the RIRD-Syn dataset. We report the number of frames, trajectory length, and scanned area for each synthetic scene.
<table><tr><td>Statistic</td><td>Room 0</td><td>Room 1</td><td>Room 2</td></tr><tr><td>Frames</td><td>1681</td><td>2337</td><td>2206</td></tr><tr><td>Traj. Len. (m)</td><td>41.9</td><td>58.5</td><td>50.6</td></tr><tr><td>Area (m2)</td><td>43.8</td><td>43.25</td><td>97.8</td></tr></table>

## B Implementation Details

Across all datasets, reflective plane identification is performed every $N _ { p } = 1 0$ frames, and Gaussian optimization is executed every $N _ { o } = 1 0$ frames. During optimization, we use $N _ { l o c a l } = 2$ recent frames together with $N _ { g l o b a l } = 7$ randomly sampled keyframes, and perform 30 optimization iterations for each optimization step. For TSDF fusion, we use a voxel size of 1 cm for the 102.2 m<sup>2</sup> Meeting Room scene and 0.5 cm for all other scenes.

For Gaussian densification, we use diferent thresholds for base and reflection Gaussians. For base Gaussians, we set the color error threshold $\delta _ { b }$ to 0.05 and the Gaussian weight threshold $\delta _ { w , b }$ to 3. For reflection Gaussians, we set the color error threshold � to 0.05 and the reflection Gaussian weight threshold $\delta _ { w , r }$ to 0.5.

For Gaussian optimization, we use diferent learning rates for base and reflection Gaussians. For base Gaussians, we set the learning rates $l r _ { p o s i t i o n } ^ { b } \ = \ 0 . 0 0 0 1 6 , \ l r _ { s c a l e } ^ { b } \ = \ 0 . 0 0 5 , \ l r _ { o p a c i t y } ^ { b } \ = \ 0 . 0 5$ $l r _ { r o t a t i o n } ^ { b } = \dot { 0 } . 0 0 1 , l r _ { S H _ { 0 } } ^ { b } = 0 . 0 0 2 5$ , and $l r _ { S H _ { r e s t } } ^ { b } ~ = ~ 0 . 0 0 0 5$ . For reflection Gaussians, we set the learning rates $\ddot { l } r _ { p o s i t i o n } ^ { r } = 0 . 0 0 0 1 6 ,$

$$
\begin{array} { l l } { { l r _ { s c a l e } ^ { r } = 0 . 0 0 5 , l r _ { o a p c i t y } ^ { r } = 0 . 0 0 1 , l r _ { r o t a t i o n } ^ { r } = 0 . 0 0 1 , l r _ { S H _ { 0 } } ^ { r } = 0 . 0 0 2 5 , } } \\ { { \mathrm { a n d } l r _ { S H _ { r e s t } } ^ { r } = 0 . 0 0 0 5 . } } \end{array}
$$

For Gaussian pruning, we remove Gaussians according to their opacity and scale. Specifically, Gaussians are pruned when their opacity is lower than the opacity threshold, or when their scale is smaller than the minimum scale threshold or larger than the maximum scale threshold. For base Gaussians, the opacity threshold, minimum scale threshold, and maximum scale threshold are set to 0.005, 0.003, and 0.1, respectively. For reflection Gaussians, the corresponding thresholds are set to 0.005, 0.005, and 0.2.

## C TSDF Color Variance Update

To identify temporally unstable reflective regions, we maintain a color inconsistency statistic for each TSDF voxel. In addition to the fused RGB color, each voxel stores a running mean � and a second central moment $M _ { 2 }$ of the observed color intensity.

Given a new RGB observation $\mathbf { c } _ { t } = \left( r _ { t } , g _ { t } , b _ { t } \right)$ at frame �, we first normalize it to [0, 1] and convert it into a scalar intensity:

$$
x _ { t } = \frac { r _ { t } + g _ { t } + b _ { t } } { 3 } .\tag{1}
$$

The fused RGB color is updated using standard running averaging:

$$
\bar { \mathbf { c } } _ { n + 1 } = \frac { n \bar { \mathbf { c } } _ { n } + \mathbf { c } _ { t } } { n + 1 } .\tag{2}
$$

Here, � denotes the current frame index, and � denotes the number of accumulated color observations for the corresponding voxel before incorporating $\mathbf { c } _ { t }$

The color intensity statistics are then updated online using a Welford-style [Welford 1962] update:

$$
\delta _ { t } = x _ { t } - \mu _ { n } ,\tag{3}
$$

$$
\mu _ { n + 1 } = \mu _ { n } + \frac { \delta _ { t } } { n + 1 } ,\tag{4}
$$

$$
M _ { 2 , n + 1 } = M _ { 2 , n } + \delta _ { t } ( x _ { t } - \mu _ { n + 1 } ) .\tag{5}
$$

After incorporating the new observation, the per-voxel color variance is computed as

$$
\sigma _ { c } ^ { 2 } = \frac { M _ { 2 , n + 1 } } { n + 1 } ,\tag{6}
$$

where $\sigma _ { c } ^ { 2 }$ denotes the temporal variance of the observed color intensity for the voxel.

During TSDF raycasting, the variance values from neighboring voxels are trilinearly interpolated to produce a dense color variance map:

$$
\mathbf { C } _ { \mathrm { v a r } } ( \mathbf { x } ) = \frac { \sum _ { i } w _ { i } \sigma _ { c , i } ^ { 2 } } { \sum _ { i } w _ { i } + \epsilon } ,\tag{7}
$$

where x denotes the raycast surface point, �<sub>�</sub> denotes the trilinear interpolation weight of the �-th neighboring voxel, and � is a small constant for numerical stability.

Pixels whose raycast variance exceeds a predefined threshold $\delta _ { v }$ are treated as temporally unstable regions and are further used for reflective-plane verification and reflection Gaussian initialization.

Table 3. Per-scene tracking accuracy comparison on RIRD-Syn. We report ATE for each synthetic reflective scene and the average across all three scenes.
<table><tr><td>Method</td><td>Room 0</td><td>Room 1</td><td>Room 2</td><td>Average</td></tr><tr><td>ORB-SLAM2</td><td>63.45</td><td>1.35</td><td>1.98</td><td>22.26</td></tr><tr><td>BundleFusion</td><td>22.40</td><td>0.92</td><td>2.23</td><td>8.52</td></tr><tr><td>SplaTAM</td><td>103.32</td><td>3.00</td><td>2.13</td><td>36.15</td></tr><tr><td>RTG-SLAM</td><td>30.24</td><td>0.43</td><td>1.46</td><td>10.71</td></tr><tr><td>GauS-SLAM</td><td>116.76</td><td>6.64</td><td>2.03</td><td>41.81</td></tr><tr><td>GS-ICP SLAM</td><td>25.00</td><td>0.29</td><td>24.56</td><td>16.62</td></tr><tr><td>GPS-SLAM</td><td>0.63</td><td>0.69</td><td>0.74</td><td>0.69</td></tr><tr><td>Ours</td><td>0.23</td><td>0.28</td><td>0.42</td><td>0.31</td></tr></table>

Table 4. Per-sequence tracking accuracy comparison on TUM-RGBD. We report ATE for each evaluated sequence and the average, where lower values indicate beter tracking accuracy.
<table><tr><td>Method</td><td>fr1_desk</td><td>fr2_xyz</td><td>fr3_office</td><td>Average</td></tr><tr><td>ORB-SLAM2</td><td>1.62</td><td>0.45</td><td>1.75</td><td>1.27</td></tr><tr><td>BundleFusion</td><td>1.65</td><td>1.13</td><td>1.81</td><td>1.53</td></tr><tr><td>SplaTAM</td><td>1.62</td><td>1.30</td><td>2.53</td><td>1.81</td></tr><tr><td>RTG-SLAM</td><td>1.51</td><td>0.58</td><td>1.32</td><td>1.13</td></tr><tr><td>GauS-SLAM</td><td>1.48</td><td>1.29</td><td>1.50</td><td>1.42</td></tr><tr><td>GS-ICP SLAM</td><td>2.96</td><td>1.77</td><td>2.26</td><td>2.33</td></tr><tr><td>GPS-SLAM</td><td>3.5</td><td>2.60</td><td>2.15</td><td>2.75</td></tr><tr><td>Ours</td><td>1.61</td><td>0.38</td><td>1.17</td><td>1.06</td></tr></table>

Table 5. Per-scene tracking accuracy comparison on Replica. We report ATE for each evaluated room and ofice scene, together with the average across all scenes.
<table><tr><td>Method</td><td>Rm 0</td><td>Rm 1</td><td>Rm 2</td><td>Off 0</td><td>Off 1</td><td>Off 2</td><td>Off 3</td><td>Off 4</td><td>Avg.</td></tr><tr><td>ORB-SLAM2</td><td>0.34</td><td>0.38</td><td>0.38</td><td>0.33</td><td>0.32</td><td>0.41</td><td>0.34</td><td>0.37</td><td>0.36</td></tr><tr><td>BundleFusion</td><td>0.41</td><td>0.40</td><td>0.43</td><td>0.43</td><td>0.37</td><td>0.42</td><td>0.46</td><td>0.44</td><td>0.42</td></tr><tr><td>SplaTAM</td><td>0.38</td><td>0.23</td><td>0.28</td><td>0.48</td><td>0.32</td><td>0.38</td><td>0.43</td><td>0.64</td><td>0.39</td></tr><tr><td>RTG-SLAM</td><td>0.20</td><td>0.19</td><td>0.12</td><td>0.16</td><td>0.13</td><td>0.22</td><td>0.25</td><td>0.25</td><td>0.19</td></tr><tr><td>GauS-SLAM</td><td>0.06</td><td>0.08</td><td>0.07</td><td>0.06</td><td>0.04</td><td>0.09</td><td>0.07</td><td>0.06</td><td>0.07</td></tr><tr><td>GS-ICP SLAM</td><td>0.16</td><td>0.17</td><td>0.12</td><td>0.19</td><td>0.13</td><td>0.17</td><td>0.18</td><td>0.23</td><td>0.18</td></tr><tr><td>GPS-SLAM</td><td>0.18</td><td>0.19</td><td>0.17</td><td>0.17</td><td>0.14</td><td>0.25</td><td>0.20</td><td>0.21</td><td>0.19</td></tr><tr><td>Ours</td><td>0.16</td><td>0.16</td><td>0.17</td><td>0.16</td><td>0.13</td><td>0.18</td><td>0.19</td><td>0.18</td><td>0.17</td></tr></table>

## D Global Plane Maintenance

During online reconstruction, each frame first obtains local planar segments from CAPE [Proença and Gao 2018] together with their plane parameters. Since these local plane labels are only valid within the current frame, we incrementally register them into a globally maintained plane table to obtain temporally consistent plane IDs across frames.

For each detected local plane, we first apply a geometric filter against all existing global planes: a match requires the normal angle diference to be within 10<sup>◦</sup> and the point-to-plane distance to be below 0.1 m. However, geometric consistency alone cannot distinguish distinct coplanar instances that share the same infinite plane (e.g., two table segments separated by a gap). We therefore maintain a spatial footprint for each global plane to record its occupied region on the plane surface.

To build the footprint, we construct a 2D orthonormal basis on the plane and discretize the surface into a regular grid with cell size 0.05 m. Each 3D point belonging to the plane is projected onto this grid, and the occupied cells along with their axis-aligned bounding box are stored. When associating a local plane, we project its points into each candidate global plane’s coordinate frame and compute three metrics: the cell overlap ratio (with 1-cell dilation), the bounding box IoU, and the bounding box gap. A match is established if the overlap exceeds 0.05, the IoU exceeds 0.02, or the gap is below 0.10m for a recently observed instance. Among all matching candidates, we select the one with the highest overlap to prevent a noisy segment from bridging two distinct instances. If no candidate matches, a new global instance is created.

Over time, instances initially created separately may be recognized as the same physical plane. We use a disjoint-set union structure to merge them: the lower-index instance absorbs the higherindex one by merging their plane parameters, unifying their footprints and bounding boxes, and retiring the absorbed instance.

Finally, we maintain a plane ID table that remaps all plane indices to the canonical global IDs after merging. During TSDF fusion and raycasting, all plane indices are converted through this table to ensure consistent plane association. The resulting global plane map is used for reflection-region association, reflection Gaussian initialization, and plane-conditioned reflection rendering.

## E Grounded SAM2 Setings

We use the oficial implementation of Grounded SAM2 [Liu et al. 2024; Ravi et al. 2024] for semantic reflective-region detection, with all parameters following the default settings provided by the oficial codebase. Given predefined text prompts, GroundingDINO [Liu et al. 2024] first predicts object bounding boxes, which are then passed to SAM2 [Ravi et al. 2024] to obtain the final segmentation masks.

The text prompts used in our implementation are: “floor”, “table”, “vending machine”, “closet”, “wall art”, “tv”, and “whiteboard”. These semantic masks are used only to generate reflective-plane candidates before temporal reflection verification.

## F More Results

## F.1 Per-scene Tracking Accuracy

We report detailed per-scene results for the same baselines used in the main paper, including ORB-SLAM2 [Mur-Artal and Tardós 2017], BundleFusion [Dai et al. 2017], SplaTAM [Keetha et al. 2024], RTG-SLAM [Peng et al. 2024], GauS-SLAM [Su et al. 2025], GS-ICP SLAM [Ha et al. 2024], and GPS-SLAM [Peng et al. 2025]. We provide detailed per-scene tracking accuracy results on RIRD-Syn, TUM-RGBD [Sturm et al. 2012], and Replica [Straub et al. 2019]. Tables 3, 4, and 5 report the absolute trajectory error (ATE) for each evaluated sequence. These results complement the averaged tracking results in the main paper and provide a more detailed comparison of tracking robustness across reflective and general indoor scenes.

## F.2 Per-scene Rendering Quality

We report detailed per-scene rendering quality results for all evaluated datasets. Tables 6, 7, 8, and 9 provide the PSNR, SSIM and LPIPS scores on ScanNet++, RIRD-Syn, RIRD-Real, and Replica, respectively. These results complement the averaged rendering quality reported in the main paper and show the performance of each method on individual scenes.

Tracking Error  
![](images/e67c6842f4c6ffd9c0bd8cbd0e16771de3de11b8fad39d146c06c16baab217d3.jpg)  
Fig. 1. Tracking-error and mesh visualization on RIRD-Syn. We compare our method with GPS-SLAM, GS-ICP SLAM, and GauS-SLAM on three synthetic reflective scenes. The camera trajectories are colored by frame-wise tracking error, where green indicates lower error and red indicates larger error. Our method maintains stable trajectories and produces more coherent mesh reconstruction in reflective scenes.

Table 6. Per-scene novel-view rendering quality comparison on the selected reflective ScanNet++ scenes. We report PSNR, SSIM, and LPIPS for each scene and the average across the two scenes.
<table><tr><td>Method</td><td>Metric</td><td>a5859cfd40</td><td>8e6ff28354</td><td>Average</td></tr><tr><td rowspan="3">SplaTAM</td><td>PSNR↑</td><td>21.08</td><td>26.63</td><td>23.85</td></tr><tr><td>SSIM↑</td><td>0.781</td><td>0.887</td><td>0.834</td></tr><tr><td>LPIPS↓</td><td>0.293</td><td>0.190</td><td>0.241</td></tr><tr><td rowspan="3">RTG-SLAM</td><td>PSNR↑</td><td>18.67</td><td>26.07</td><td>22.37</td></tr><tr><td>SSIM↑</td><td>0.705</td><td>0.862</td><td>0.783</td></tr><tr><td>LPIPS↓</td><td>0.372</td><td>0.226</td><td>0.299</td></tr><tr><td rowspan="3">GauS-SLAM</td><td>PSNR↑</td><td>22.75</td><td>27.59</td><td>25.17</td></tr><tr><td>SSIM↑</td><td>0.745</td><td>0.856</td><td>0.800</td></tr><tr><td>LPIPS↓</td><td>0.332</td><td>0.243</td><td>0.287</td></tr><tr><td rowspan="3">GS-ICP SLAM</td><td>PSNR↑</td><td>21.96</td><td>27.50</td><td>24.73</td></tr><tr><td>SSIM↑</td><td>0.821</td><td>0.913</td><td>0.867</td></tr><tr><td>LPIPS↓</td><td>0.238</td><td>0.149</td><td>0.194</td></tr><tr><td rowspan="3">GPS-SLAM</td><td>PSNR↑</td><td>22.18</td><td>26.99</td><td>24.59</td></tr><tr><td>SSIM↑</td><td>0.781</td><td>0.890</td><td>0.836</td></tr><tr><td>LPIPS↓</td><td>0.317</td><td>0.193</td><td>0.255</td></tr><tr><td rowspan="3">Ours</td><td>PSNR↑</td><td>23.51</td><td>28.50</td><td>26.01</td></tr><tr><td>SSIM↑</td><td>0.823</td><td>0.916</td><td>0.869</td></tr><tr><td>LPIPS↓</td><td>0.229</td><td>0.141</td><td>0.185</td></tr></table>

## F.3 Tracking Error Visualization on RIRD-Syn

We further visualize the reconstructed meshes and tracking-errorcolored trajectories on RIRD-Syn in Fig. 1. GPS-SLAM [Peng et al. 2025] performs ICP against the TSDF volume, which makes its tracking not afected by reflection artifacts. GS-ICP SLAM [Ha et al.

Table 7. Per-scene novel-view rendering quality comparison on RIRD-Syn. We report PSNR, SSIM, and LPIPS for each synthetic reflective scene and the average across all three scenes.
<table><tr><td>Method</td><td>Metric</td><td>Room 0</td><td>Room 1</td><td>Room 2</td><td>Average</td></tr><tr><td rowspan="3">SplaTAM</td><td>PSNR↑</td><td>15.43</td><td>23.45</td><td>17.08</td><td>18.66</td></tr><tr><td>SSIM↑</td><td>0.592</td><td>0.764</td><td>0.547</td><td>0.634</td></tr><tr><td>LPIPS↓</td><td>0.474</td><td>0.341</td><td>0.453</td><td>0.422</td></tr><tr><td rowspan="3">RTG-SLAM</td><td>PSNR↑</td><td>14.74</td><td>22.97</td><td>15.74</td><td>17.81</td></tr><tr><td>SSIM↑</td><td>0.605</td><td>0.813</td><td>0.533</td><td>0.650</td></tr><tr><td>LPIPS↓</td><td>0.496</td><td>0.332</td><td>0.503</td><td>0.444</td></tr><tr><td rowspan="3">GauS-SLAM</td><td>PSNR↑</td><td>9.61</td><td>21.30</td><td>20.15</td><td>17.02</td></tr><tr><td>SSIM↑</td><td>0.209</td><td>0.677</td><td>0.605</td><td>0.497</td></tr><tr><td>LPIPS↓</td><td>0.600</td><td>0.395</td><td>0.290</td><td>0.428</td></tr><tr><td rowspan="3">GS-ICP SLAM</td><td>PSNR↑</td><td>18.63</td><td>28.23</td><td>21.25</td><td>22.70</td></tr><tr><td>SSIM↑</td><td>0.754</td><td>0.909</td><td>0.731</td><td>0.798</td></tr><tr><td>LPIPS↓</td><td>0.392</td><td>0.171</td><td>0.253</td><td>0.272</td></tr><tr><td rowspan="3">GPS-SLAM</td><td>PSNR↑</td><td>25.87</td><td>29.34</td><td>25.14</td><td>26.79</td></tr><tr><td>SSIM↑</td><td>0.875</td><td>0.929</td><td>0.811</td><td>0.871</td></tr><tr><td>LPIPS↓</td><td>0.253</td><td>0.157</td><td>0.215</td><td>0.208</td></tr><tr><td rowspan="3">Ours</td><td>PSNR↑</td><td>30.64</td><td>33.12</td><td>27.11</td><td>30.29</td></tr><tr><td>SSIM↑</td><td>0.939</td><td>0.931</td><td>0.856</td><td>0.909</td></tr><tr><td>LPIPS↓</td><td>0.160</td><td>0.183</td><td>0.231</td><td>0.191</td></tr></table>

2024] aligns incoming frames with an optimized Gaussian map. Since reflection efects can disturb the optimization of the Gaussian map, its ICP tracking becomes less reliable in reflective regions. GauS-SLAM [Su et al. 2025] uses both color and depth cues for pose estimation and is more sensitive to reflection-induced photometric inconsistency. In contrast, by explicitly modeling reflections, our method achieves more accurate tracking and produces higherquality mesh reconstruction.

Table 8. Per-scene input-view rendering quality comparison on RIRD-Real. We report PSNR, SSIM, and LPIPS for each real scene and the average across all six scenes.
<table><tr><td>Method</td><td>Metrics</td><td>Living Room 0</td><td>Living Room 1</td><td>Kitchen</td><td>Private Room</td><td>Rest Room</td><td>Meeting Room</td><td>Average</td></tr><tr><td rowspan="3">SplaTAM</td><td>PSNR↑</td><td>24.00</td><td>26.72</td><td>24.21</td><td>24.44</td><td>22.71</td><td>14.03</td><td>22.69</td></tr><tr><td>SSIM↑</td><td>0.856</td><td>0.902</td><td>0.843</td><td>0.844</td><td>0.797</td><td>0.675</td><td>0.820</td></tr><tr><td>LPIPS↓</td><td>0.352</td><td>0.281</td><td>0.336</td><td>0.277</td><td>0.378</td><td>0.507</td><td>0.355</td></tr><tr><td rowspan="3">RTG-SLAM</td><td>PSNR↑</td><td>25.42</td><td>25.39</td><td>25.16</td><td>25.34</td><td>27.25</td><td>22.42</td><td>25.16</td></tr><tr><td>SSIM↑</td><td>0.863</td><td>0.881</td><td>0.861</td><td>0.866</td><td>0.888</td><td>0.773</td><td>0.855</td></tr><tr><td>LPIPS↓</td><td>0.365</td><td>0.345</td><td>0.371</td><td>0.302</td><td>0.317</td><td>0.455</td><td>0.359</td></tr><tr><td rowspan="3">GauS-SLAM</td><td>PSNR↑</td><td>25.37</td><td>26.09</td><td>26.37</td><td>27.01</td><td>25.91</td><td>17.89</td><td>24.77</td></tr><tr><td>SSIM↑</td><td>0.837</td><td>0.860</td><td>0.853</td><td>0.848</td><td>0.839</td><td>0.652</td><td>0.815</td></tr><tr><td>LPIPS↓</td><td>0.306</td><td>0.306</td><td>0.305</td><td>0.264</td><td>0.342</td><td>0.456</td><td>0.330</td></tr><tr><td rowspan="3">GS-ICP SLAM</td><td>PSNR↑</td><td>26.87</td><td>27.60</td><td>27.66</td><td>26.74</td><td>27.18</td><td>13.53</td><td>24.93</td></tr><tr><td>SSIM↑</td><td>0.898</td><td>0.916</td><td>0.915</td><td>0.889</td><td>0.907</td><td>0.657</td><td>0.864</td></tr><tr><td>LPIPS↓</td><td>0.262</td><td>0.237</td><td>0.233</td><td>0.232</td><td>0.244</td><td>0.529</td><td>0.289</td></tr><tr><td rowspan="3">GPS-SLAM</td><td>PSNR↑</td><td>27.26</td><td>27.70</td><td>21.89</td><td>24.92</td><td>26.58</td><td>18.42</td><td>24.46</td></tr><tr><td>SSIM↑</td><td>0.894</td><td>0.904</td><td>0.831</td><td>0.847</td><td>0.883</td><td>0.734</td><td>0.849</td></tr><tr><td>LPIPS↓</td><td>0.273</td><td>0.260</td><td>0.347</td><td>0.278</td><td>0.292</td><td>0.480</td><td>0.322</td></tr><tr><td rowspan="3">Ours</td><td>PSNR↑</td><td>30.38</td><td>31.94</td><td>32.56</td><td>30.56</td><td>31.15</td><td>28.35</td><td>30.82</td></tr><tr><td>SSIM↑</td><td>0.919</td><td>0.941</td><td>0.937</td><td>0.919</td><td>0.923</td><td>0.886</td><td>0.921</td></tr><tr><td>LPIPS↓</td><td>0.224</td><td>0.190</td><td>0.194</td><td>0.197</td><td>0.248</td><td>0.253</td><td>0.218</td></tr></table>

Table 9. Per-scene input-view rendering quality comparison on Replica. We report PSNR, SSIM, and LPIPS for each evaluated scene and the average across all scenes.
<table><tr><td>Method</td><td>Metric</td><td>Office 0</td><td>Office 1</td><td>Office 2</td><td>Office 3</td><td>Office 4</td><td>Room 0</td><td>Room 1</td><td>Room 2</td><td>Average</td></tr><tr><td rowspan="3">SplaTAM</td><td>PSNR↑</td><td>38.18</td><td>39.26</td><td>32.00</td><td>30.41</td><td>32.41</td><td>32.36</td><td>33.45</td><td>35.01</td><td>34.14</td></tr><tr><td>SSIM↑</td><td>0.963</td><td>0.958</td><td>0.929</td><td>0.903</td><td>0.916</td><td>0.933</td><td>0.930</td><td>0.952</td><td>0.936</td></tr><tr><td>LPIPS↓</td><td>0.118</td><td>0.139</td><td>0.158</td><td>0.166</td><td>0.179</td><td>0.118</td><td>0.133</td><td>0.126</td><td>0.142</td></tr><tr><td rowspan="3">RTG-SLAM</td><td>PSNR↑</td><td>37.78</td><td>38.41</td><td>32.23</td><td>31.86</td><td>34.98</td><td>30.41</td><td>32.82</td><td>33.82</td><td>34.04</td></tr><tr><td>SSIM↑</td><td>0.952</td><td>0.950</td><td>0.918</td><td>0.911</td><td>0.935</td><td>0.882</td><td>0.917</td><td>0.929</td><td>0.924</td></tr><tr><td>LPIPS↓</td><td>0.141</td><td>0.182</td><td>0.212</td><td>0.197</td><td>0.173</td><td>0.196</td><td>0.179</td><td>0.183</td><td>0.183</td></tr><tr><td rowspan="3">GauS-SLAM</td><td>PSNR↑</td><td>42.40</td><td>39.26</td><td>36.18</td><td>34.73</td><td>40.32</td><td>39.05</td><td>39.55</td><td>39.33</td><td>38.85</td></tr><tr><td>SSIM↑</td><td>0.980</td><td>0.958</td><td>0.991</td><td>0.969</td><td>0.976</td><td>0.972</td><td>0.975</td><td>0.974</td><td>0.974</td></tr><tr><td>LPIPS↓</td><td>0.047</td><td>0.139</td><td>0.069</td><td>0.059</td><td>0.063</td><td>0.056</td><td>0.056</td><td>0.058</td><td>0.068</td></tr><tr><td rowspan="3">GS-ICP SLAM</td><td>PSNR↑</td><td>37.79</td><td>37.65</td><td>29.73</td><td>30.57</td><td>34.28</td><td>28.16</td><td>32.88</td><td>32.96</td><td>33.00</td></tr><tr><td>SSIM↑</td><td>0.964</td><td>0.954</td><td>0.924</td><td>0.932</td><td>0.946</td><td>0.823</td><td>0.930</td><td>0.932</td><td>0.926</td></tr><tr><td>LPIPS↓</td><td>0.076</td><td>0.123</td><td>0.132</td><td>0.123</td><td>0.109</td><td>0.128</td><td>0.116</td><td>0.124</td><td>0.116</td></tr><tr><td rowspan="3">GPS-SLAM</td><td>PSNR↑</td><td>40.91</td><td>41.10</td><td>35.10</td><td>35.21</td><td>37.46</td><td>34.87</td><td>36.66</td><td>36.67</td><td>37.25</td></tr><tr><td>SSIM↑</td><td>0.970</td><td>0.967</td><td>0.947</td><td>0.947</td><td>0.957</td><td>0.947</td><td>0.954</td><td>0.954</td><td>0.955</td></tr><tr><td>LPIPS↓</td><td>0.076</td><td>0.132</td><td>0.131</td><td>0.109</td><td>0.103</td><td>0.095</td><td>0.102</td><td>0.114</td><td>0.108</td></tr><tr><td rowspan="3">Ours</td><td>PSNR↑</td><td>41.30</td><td>41.29</td><td>34.79</td><td>35.33</td><td>37.49</td><td>34.86</td><td>36.67</td><td>36.59</td><td>37.29</td></tr><tr><td>SSIM↑</td><td>0.976</td><td>0.971</td><td>0.953</td><td>0.954</td><td>0.961</td><td>0.954</td><td>0.959</td><td>0.959</td><td>0.961</td></tr><tr><td>LPIPS↓</td><td>0.070</td><td>0.125</td><td>0.125</td><td>0.103</td><td>0.096</td><td>0.089</td><td>0.096</td><td>0.110</td><td>0.102</td></tr></table>

## References

Angela Dai, Matthias Nießner, Michael Zollhöfer, Shahram Izadi, and Christian Theobalt. 2017. Bundlefusion: Real-time globally consistent 3d reconstruction using on-the-fly surface reintegration. ACM Transactions on Graphics (ToG) 36, 4 (2017), 1.

Seongbo Ha, Jiung Yeon, and Hyeonwoo Yu. 2024. Rgbd gs-icp slam. In European Conference on Computer Vision. Springer, 180–197.

Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. 2024. Splatam: Splat track & map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21357–21366.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. 2024. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision. Springer, 38–55.

Raul Mur-Artal and Juan D Tardós. 2017. Orb-slam2: An open-source slam system for monocular, stereo, and rgb-d cameras. IEEE transactions on robotics 33, 5 (2017), 1255–1262.

Zhexi Peng, Tianjia Shao, Yong Liu, Jingke Zhou, Yin Yang, Jingdong Wang, and Kun Zhou. 2024. Rtg-slam: Real-time 3d reconstruction at scale using gaussian splatting. In ACM SIGGRAPH 2024 Conference Papers. 1–11.

Zhexi Peng, Kun Zhou, and Tianjia Shao. 2025. Gaussian-plus-SDF SLAM: High-fidelity 3D reconstruction at 150+ fps. Computational Visual Media (2025).

Pedro F Proença and Yang Gao. 2018. Fast cylinder and plane extraction from depth cameras for visual odometry. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 6813–6820.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. 2024. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024).

Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, et al. 2019. The replica

dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797 (2019).

Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. 2012. A benchmark for the evaluation of RGB-D SLAM systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 573–580.

Yongxin Su, Lin Chen, Kaiting Zhang, Zhongliang Zhao, Chenfeng Hou, and Ziping Yu. 2025. GauS-SLAM: Dense RGB-D SLAM with Gaussian Surfels. arXiv preprint arXiv:2505.01934 (2025).

Barry Payne Welford. 1962. Note on a method for calculating corrected sums of squares and products. Technometrics 4, 3 (1962), 419–420.

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. 2023. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 12–22.