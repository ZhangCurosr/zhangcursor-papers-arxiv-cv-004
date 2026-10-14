# WORLDRECON: GENERATIVE MULTI-VIEW MESHRECONSTRUCTION FOR LARGE-SCALE 3D WORLD

Mengchen Zhang<sup>1,2</sup>, Yang Li<sup>2</sup>, Chunshi Wang<sup>1,2</sup>, Tong Wu<sup>3</sup>, Chunchao Guo<sup>2</sup> <sup>1</sup> Zhejiang University <sup>2</sup> Tencent Hunyuan <sup>3</sup> Fudan University

![](images/db689beddc8a6b8b56bb1c7b915a571bd79b7ce82813949f95cf953c0e461f3b.jpg)  
Figure 1: WorldRecon Overview. Given unposed multi-view RGB images of an indoor scene, WorldRecon reconstructs a complete scene-scale PBR mesh block-by-block via a native 3D generative prior at arbitrary scene scales, yielding high-fidelity geometry, depth, normal, and texture. From left to right, we demonstrate WorldRecon on real-world captures, synthetic images, and frames from generated videos, highlighting generalization across diverse input domains.

## ABSTRACT

Reconstructing complete, high-quality meshes of large-scale scenes from unposed multi-view images requires faithful geometric grounding while plausibly completing surfaces that are unobserved or only sparsely captured. We present WorldRecon, a generative multi-view reconstruction framework for recovering complete, large-scale 3D scene meshes from unposed RGB images. WorldRecon uses a coarse, noisy global point cloud and camera poses estimated from the input images by a reconstruction model as a global scaffold for whole-scene generation and partitions it into overlapping spatial blocks. To robustly leverage these imperfect geometric priors, we equip the native 3D generative model TRELLIS.2 with a dedicated reconstruction-based conditioning architecture that generates each block from its local point-cloud crop and block-relevant image observations, producing sparse structure, detailed geometry, and PBR materials. To maintain scene-level coherence, we introduce neighbour-aware latent inpainting and a causal worldgeneration procedure that propagates structural, geometric, and material context across overlapping blocks. Together, these designs scale multi-view generative reconstruction to large-scale scenes, yielding coherent, unified meshes. Extensive experiments demonstrate that WorldRecon achieves state-of-the-art reconstruction performance, reducing Chamfer distance by 14.9% and 43.1% while improving Fscore by 17.3% and 12.0% over the strongest baselines on synthetic and real-world indoor scenes, respectively, thereby demonstrating faithful and complete gener-

ative reconstruction as well as strong sim-to-real generalization. Project page: https://kszpxxzmc.github.io/WorldRecon-website/.

## 1 INTRODUCTION

Recovering a large-scale scene from unposed RGB images is a fundamental problem in computer vision, with broad applications in virtual and augmented reality, embodied intelligence, and digital content production. Reconstruction-based methods Wang et al. (2024); Leroy et al. (2024); Wang et al. (2026); Barron et al. (2022); Yu et al. (2022); Kerbl et al. (2023) recover scene geometry from visual observations, but often leave occluded regions incomplete and produce noisy geometry that is difficult to convert into clean meshes with detailed surfaces and materials.

Generative priors improve reconstruction completeness by inferring plausible geometry and appearance beyond the observations. Building on native 3D generative models Xiang et al. (2025; 2026); Zhang et al. (2024), recent multi-view extensions Chang et al. (2025); Huang et al. (2026) directly reconstruct structured 3D assets from unposed RGB images. Extending these methods to scene scale, however, remains challenging as the spatial extent and geometric complexity of an environment exceed the capacity of a single high-resolution generative representation.

Recent scene generation methods Chen et al. (2025); Wu et al. (2024b); Meng et al. (2025); Li et al. (2026); Engstler et al. (2025) scale to substantially larger environments by decomposing a scene into overlapping spatial blocks and generating them jointly or progressively. However, they are typically conditioned on text or layouts and thus lack visual guidance and 3D spatial grounding. More closely related to our task, scene-level generative reconstruction methods represent large scenes using hierarchical voxel latents Ren et al. (2024) or overlapping scene chunks Schmid et al. (2026). However, they require accurate camera poses to project image features into a 3D feature volume, making reconstruction quality sensitive to pose errors.

In this work, we introduce WorldRecon, a generative multi-view framework for reconstructing largescale scene meshes from unposed RGB images. We employ VGGT-Ω Wang et al. (2026) to estimate coarse camera poses and recover a global point-cloud scaffold, which we partition into overlapping blocks. Our reconstruction-based conditioning leverages local point-cloud crops and block-relevant image observations, enabling the generation of complete, high-fidelity mesh blocks with PBR materials from imperfect geometric and camera estimates. To promote consistency across blocks, we train neighbour-aware latent inpainting models that condition each new block on overlap latents from previously generated neighbours. At inference, we progressively expand the scene following a causal block order, propagating sparse structure, geometry, and material context through shared world representations. This enables neighbouring blocks to transition seamlessly in both geometry and appearance before being decoded and assembled into a unified scene mesh.

Extensive experiments on 3D-FRONT Fu et al. (2021) and ScanNet++ Yeshwanth et al. (2023) demonstrate that WorldRecon reconstructs globally accurate and complete indoor scenes with fine geometric and material details from unposed RGB images on both synthetic and real-world settings.

In summary, our main contributions are as follows:

• We present WorldRecon, a framework that integrates 3D reconstruction and generation to recover large-scale, high-fidelity scene meshes with PBR materials from unposed multi-view RGB images.

• We develop reconstruction-based block generation with neighbour-aware inpainting, producing faithful mesh blocks from local observations and imperfect geometric and camera estimates.

• We propose a causal world-generation pipeline that propagates sparse structure, geometry, and material context across overlapping blocks for scalable and seamless scene generation.

• Extensive experiments demonstrate state-of-the-art performance in large-scale mesh reconstruction from unposed multi-view images across synthetic and real-world indoor scenes.

## 2 RELATED WORK

Multi-View 3D Reconstruction. Traditional 3D reconstruction relies on Structure-from-Motion Schonberger & Frahm (2016); Pan et al. (2024) and Multi-View Stereo Furukawa &

Hernandez (2015); Sch´ onberger et al. (2016) to recover camera poses and scene geometry through¨ triangulation. Learning-based MVS methods Yao et al. (2018; 2019); Wang et al. (2021a) further improve quality and efficiency via differentiable cost volumes. More recently, pose-free models Wang et al. (2024); Leroy et al. (2024); Wang et al. (2025; 2026) cast the pairwise reconstruction problem as a regression of pointmaps, jointly regressing cameras and dense geometry. Beyond discrete point clouds, Neural Radiance Fields Mildenhall et al. (2021); Wang et al. (2021c); Barron et al. (2022) reframe reconstruction as inverse volumetric rendering for photorealistic view synthesis. Neural implicit surface methods Wang et al. (2021b); Yariv et al. (2021); Yu et al. (2022) represent scenes as continuous signed-distance functions. 3D Gaussian Splatting Kerbl et al. (2023); Huang et al. (2024); Charatan et al. (2024) optimize scenes via differentiable rasterization for fast rendering. However, these methods suffer from incomplete reconstruction and tend to produce over-smooth, blurred details in occluded regions.

Generative Multi-View 3D Reconstruction. Generative priors are introduced in multi-view 3D reconstruction to mitigate reconstructive incompleteness. Early efforts Liu et al. (2024); Gao et al. (2024); Wu et al. (2024a); Guo et al. (2025); Xu et al. (2025) leverage 2D or video diffusion priors to synthesize novel views or depth maps, but still require per-scene optimization and suffer from crossview inconsistency. Other works Jin et al. (2024); Tang et al. (2024b); Ziwen et al. (2025) regress target-view pixels or 3D Gaussians in a single feed-forward pass, but output unstructured primitives rather than meshes. Recently, native 3D generative models Xiang et al. (2025; 2026); Zhang et al. (2024) have enabled direct mesh generation from image conditioning. ReconViaGen Chang et al. (2025) and UniRecGen Huang et al. (2026) extend this paradigm to multi-view conditioning, but remain confined to the object level. SCube Ren et al. (2024) and GenRecon Schmid et al. (2026) further extend multi-view 3D reconstruction to the scene level, yet both rely on accurate camera poses to project image features into a 3D feature volume, which induces cross-view conflicts and degrades reconstruction quality under imperfect poses. In contrast, our method conditions on pose free RGB images, recovering coherent scene mesh block-by-block via a native 3D generative prior.

Unbounded 3D Scene Generation. Extending 3D generation from objects to unbounded scenes has recently drawn increasing attention. Early efforts focus on bounded environments such as a single room, either retrieving assets from a fixed database Tang et al. (2024a); Wang et al. (2021d), or hallucinating RGB-D frames with 2D diffusion models under text or layout conditions Hollein et al.¨ (2023); Schult et al. (2024). To scale beyond bounded rooms, recent works pursue unbounded scene generation via block-wise generation, where neighbouring blocks are either jointly denoised and seamlessly blended Chen et al. (2025), or progressively synthesized by an extrapolation prior Wu et al. (2024b); Meng et al. (2025); Li et al. (2026); Engstler et al. (2025). However, these methods are either unconditional or conditioned only on text or layout, lacking both visual guidance and 3D spatial grounding. Our method instead grounds the scene in a coarse predicted point cloud and conditions each block on a few multi-view images, scaling multi-view generative reconstruction to arbitrarily large scenes.

## 3 WORLDRECON METHOD

Given an arbitrary number of uncalibrated RGB images $\{ I _ { n } \} _ { n = 1 } ^ { N }$ of a large-scale indoor scene, our goal is to recover a complete, high-fidelity, scene-level 3D mesh endowed with PBR materials. Our generative multi-view reconstruction framework couples a reconstruction prior for global layout consistency with a generative prior for surface fidelity and completeness. To this end, we curate a block-level training dataset (Section 3.1). We finetune the object-level generative prior of TREL-LIS.2 Xiang et al. (2026) into a block-level mesh generator conditioned on each block’s point-cloud slice and multi-view RGBA images, together with a neighbour-aware 3D inpainting variant (Section 3.2). At inference, we first extract a global point cloud with VGGT-Ω Wang et al. (2026) as a coarse geometric layout cue and partition it into overlapping point cloud blocks. Starting from a seed block, we sequentially generate the remaining blocks across three generation stages, conditioning each block on its overlaps with already-generated neighbours (Section 3.3). The overall pipeline is shown in Figure 2. In this way, the point-cloud scaffold pins the scene to the observed geometry, while the block-wise generative prior fills in complete, high-fidelity surfaces.

![](images/0409def37e9b38f13a332bd7bc125a4530f377732e5e30f5e85793ea6eb30925.jpg)  
Figure 2: WorldRecon Pipeline. Left: Given unposed multi-view RGB images, VGGT-Ω reconstructs a global point cloud. The point cloud is divided into overlapping blocks, each paired with local RGBA images and predicted camera poses. Middle: For each block, DINOv3 image tokens and Plucker-ray camera tokens are fused into fixed-length multi-view tokens, while a point-cloud¨ encoder extracts geometric tokens. Right: Three cascaded flow models progressively generate the sparse structure, detailed geometry, and PBR materials.

## 3.1 DATA CURATION

Scene Slicing. To enable block-level scene generation, we first construct a structured, scalable 3D block dataset. We build on two synthetic indoor scene datasets, 3D-FRONT Fu et al. (2021) and SAGE-10k Xia et al. (2026), which together cover diverse layouts, furniture styles, and material distributions. For each scene, we render RGBD views from randomised eye-level cameras under varied lighting. We then propose a scene-slicing strategy. The scene is partitioned into a regular grid of canonically oriented cubic blocks with side length $\ell ,$ each covering the full vertical extent of the scene. This yields roughly 56k blocks from 6k 3D-FRONT scenes and 61k blocks from 10k SAGE-10k scenes, totalling around 117k training blocks. Each mesh block is normalised to canonical $[ - 0 . 5 , 0 . 5 ] ^ { 3 }$ space.

Condition Extraction. Each block then yields a training tuple $( \mathcal { M } _ { b } , \mathcal { P } _ { b } , \{ ( I _ { v } , K _ { v } , R _ { v } , t _ { v } ) \} _ { v = 1 } ^ { 1 6 } )$ denoting the block-local mesh, point cloud, RGBA renders, and camera poses, respectively. Every rendered RGBD view is back-projected into the block’s canonical space, yielding a block-local camera pose $( K _ { v } , R _ { v } , t _ { v } )$ . The pixels falling inside the block cube form the RGBA render $I _ { v } ,$ , with the in-cube mask as alpha. We retain the top-16 views per block by in-cube pixel coverage. For ${ \mathcal { P } } _ { b } .$ , we randomly sample from two complementary noisy sources during training: unprojected GT depth from a random subset of views with augmented noise, and a VGGT-Ω Wang et al. (2026) reconstruction from $\{ I _ { v } \} _ { v = 1 } ^ { 1 6 }$ , both downsampled to M points. This exposes the model to both controllable augmentation and the target inference distribution, keeping generation robust when only VGGT-Ω point clouds are available at test time.

## 3.2 RECONSTRUCTION-BASED BLOCK GENERATION

Generative Prior. We build on TRELLIS.2 Xiang et al. (2026), an image-to-3D generation model that represents 3D asset as a sparse voxel structure termed O-Voxel, defined as a set of feature tuples $\mathbf { f } ~ = ~ \{ ( \mathbf { f } _ { i } ^ { \mathrm { g e o } } , \mathbf { f } _ { i } ^ { \mathrm { m a t } } , \mathbf { o } _ { i } ) \} _ { i = 1 } ^ { L }$ over $L$ active voxels on a regular $G ^ { 3 }$ grid, where $\mathbf { o } _ { i }$ denotes the voxel position, $\mathbf { f } _ { i } ^ { \mathrm { g e o } }$ encodes local geometric information, and $\mathbf { f } _ { i } ^ { \mathrm { m a t } }$ encodes material properties. Generation cascades three DiT flow-matching stages: Sparse Structure predicts the active-voxel positions $\left\{ \mathbf { o } _ { i } \right\}$ . Geometry predicts geometry latents $\bar { \mathbf { z } } ^ { \mathrm { g e o } }$ within active voxels, which are decoded into $\{ \mathbf { f } _ { i } ^ { \mathrm { g e o } } \}$ Material synthesizes material latents $\mathbf { z } ^ { \mathrm { m a t } }$ aligned to $\mathbf { z } ^ { \mathrm { g e o } }$ , decoded to $\{ \mathbf { f } _ { i } ^ { \mathrm { m a t } } \}$ . We finetune all three stages from the pretrained object-level TRELLIS.2 for block-level scene generation, replacing its original single-image conditioning with our reconstruction-based conditioning as detailed below.

Reconstruction-Based Conditioning. We leverage the block-level conditions curated in Section 3.1 for controllable 3D generation. Multi-view images and their camera poses are jointly encoded into a unified representation $\mathbf { c } ^ { \mathrm { { m v } } }$ shared by all three stages. Sparse Structure stage additionally receives point-cloud tokens $\mathbf { c } ^ { \mathrm { { p c } } }$ as a coarse geometric prior, giving ${ \bf c } = [ { \bf c } ^ { \mathrm { m v } } , { \bf c } ^ { \mathrm { p c } } ]$ . The resulting conditioning is injected into every cross-attention layer of the DiT.

Multi-View Image Condition. For each block, we randomly sample V views $\{ I _ { v } \} _ { v = 1 } ^ { V }$ with their camera poses $\{ ( R _ { v } , t _ { v } , K _ { v } ) \} _ { v = 1 } ^ { V }$ . Each view is encoded by a frozen DINOv3 Simeoni et al. (2025)´ patch encoder into image tokens $\mathbf { c } _ { v } ^ { \mathrm { i m g } }$ , while its camera is represented by five Plucker rays which¨ are projected through an MLP into camera tokens $\mathbf { c } _ { v } ^ { \mathrm { { c a m } } }$ . Per-view camera tokens are prepended to the corresponding image patches, forming the multi-view context $\mathcal { S } = [ \mathbf { c } _ { 1 } ^ { \mathrm { { c a m } } } , \mathbf { c } _ { 1 } ^ { \mathrm { { i m g } } } , \ldots , \mathbf { c } _ { V } ^ { \mathrm { { c a m } } } , \mathbf { c } _ { V } ^ { \mathrm { { i m g } } } ]$ Starting from a set of learnable query tokens ${ \bf q } _ { 0 }$ , we run $N _ { t }$ transformer cross-attention blocks that progressively fuse S into the fixed-length tokens:

$$
\mathbf { q } _ { l + 1 } = { \mathrm { C r o s s A t t n } } { \big ( } Q ( \mathbf { q } _ { l } ) , K ( S ) , V ( S ) { \big ) } , \quad l \in \{ 0 , 1 , \ldots , N _ { t } - 1 \} ,\tag{1}
$$

yielding the multi-view condition ${ \bf c } ^ { \mathrm { m v } } = { \bf q } _ { N _ { t } }$

Point Cloud Condition. Recovering geometry from RGB alone is inherently ambiguous, especially at scene scale where sparse views cover only partial structure. We therefore condition the Sparse Structure stage on a point cloud that supplies coarse spatial structure. Following the Hunyuan3D-Omni Team (2025) design, each point of $\mathbf { \mathcal { P } } _ { b } = \{ \mathbf { p } _ { j } \} _ { j = 1 } ^ { M }$ is encoded by a Fourier positional embedder followed by a linear projection, and K learnable modality tokens are appended to form the pointcloud condition $\mathbf { c } ^ { \mathrm { { p c } } }$ . Point-cloud tokens are randomly dropped during training, allowing classifierfree guidance to support both point-free and point-conditioned inference.

Neighbour-Aware Latent Inpainting. At inference, the scene is extended block-by-block. Each new block only observes partial 3D context from its already-generated neighbours, yet it must produce a complete 3D representation that continues the surrounding geometry and appearance seamlessly. We formulate this as a 3D block inpainting task and train dedicated inpaint variants of all three denoisers. We augment each denoiser’s native input with $C { + 1 }$ inpainting-specific channels, comprising a one-channel binary inpainting mask m and a $C .$ -channel masked known latent $\mathbf { x } _ { 0 } { \odot } ( \mathbf { 1 - m } )$ where C denotes the latent feature dimension. The resulting hidden feature is:

$$
\begin{array} { r } { \mathbf { h } _ { \mathrm { i n } } = f _ { \mathrm { i n } } \Big ( \big [ \mathbf { x } _ { t } \big [ , \mathbf { z } ^ { \mathrm { g e o } } \big ] , \ \mathbf { m } , \ \mathbf { x } _ { 0 } \odot \left( \mathbf { 1 } - \mathbf { m } \right) \big ] \Big ) . } \end{array}\tag{2}
$$

Here, $\mathbf { m } _ { i } = 0$ marks known latent tokens supplied, whereas $\mathbf { m } _ { i } = 1$ marks tokens to be freshly generated. $\mathbf { z } ^ { \mathrm { g e o } }$ denotes an optional Geometry latent included for the Material stage.

To retain the priors of our block-level generation models, we initialise each inpaint variant from its corresponding generation model. The original input projection is directly inherited, while the projection weights associated with the additional $C { \bar { + } } 1$ inpainting channels are zero-initialised. At the start of finetuning, the inpaint variant is thus functionally equivalent to the base generator, providing a stable initialisation for optimisation.

Training masks are sampled to emulate the overlap layouts arising at inference. Since all scenes fit within a single block along the vertical axis, we restrict masking to the XY plane. On each iteration, we sample the number of neighbour-supplied edges $E \in \{ 1 , 2 \}$ of the block, enforcing an L-shape constraint (one X-edge plus one $Y { \mathrm { - e d g e } } )$ when $\bar { E } = 2$ . The mask configurations cover the 1- and 2- neighbour cases encountered under our voxel-priority generation ordering (Section 3.3). The known margin along each edge is sampled between $3 \bar { \ell } / 1 6$ and $\ell / 4 .$ . At inference, we shrink the raw overlap of width $\ell / 4$ by one quarter, leaving a known margin of $3 \ell / 1 6$ and an editable transition band of width $\ell / 1 \dot { 6 }$ . This design preserves sufficient neighbour context for cross-block consistency while allowing the model to adjust geometric and material transitions near block boundaries.

## 3.3 WORLD GENERATION PIPELINE

Given an arbitrary number of uncalibrated RGB images $\{ I _ { n } \} _ { n = 1 } ^ { N }$ , our inference pipeline generates a large-scale 3D mesh through block-based, neighbour-aware latent inpainting and overlap blending.

Block Preparation. We run VGGT-Ω Wang et al. (2026) to recover coarse camera parameters and a global point cloud from the input RGB images. After confidence filtering and outlier removal, we align and rescale the reconstruction to the canonical scene coordinates. We partition its XY bounding box into overlapping cubic blocks of side length ℓ, with adjacent blocks overlapping by $\ell / 4$ . For each block, we retain only the in-block pixels of each input RGB image and consider images with at least 15% in-block pixel coverage as candidate views. We voxelise the corresponding visible points and select up to $V$ views whose union maximises the block’s voxel coverage. This produces the training-compatible tuple $( \hat { \mathcal { P } } _ { b } , \{ ( I _ { v } , \hat { K } _ { v } , \hat { R } _ { v } , \hat { t } _ { v } ) \} _ { v = 1 } ^ { V } )$ , without requiring ground-truth geometry or calibrated cameras.

We determine a voxel-priority generation order based on block-wise voxel occupancy. Starting from the block with the highest occupancy, we greedily expand a connected frontier while ensuring that each block has at most two previously generated four-connected neighbours. This ordering matches the 1- and 2-neighbour masks used to train the block inpainters.

Sparse Structure World Generation. We maintain shared world-space Sparse Structure latent and occupancy. The seed latent is sampled by the generator, while subsequent blocks are inpainted from overlaps with generated neighbours. To ensure Sparse Structure continuity across block boundaries, we introduce a trajectory-clamping mechanism. At every denoising step t, the known overlap i restored to its forward-flow trajectory:

$$
\mathbf { x } _ { t , \mathrm { k n o w n } } ^ { \mathrm { s s } } = ( 1 - t ) \mathbf { x } _ { 0 , \mathrm { k n o w n } } ^ { \mathrm { s s } } + t \epsilon .\tag{3}
$$

Here, $\mathbf { x } _ { \mathrm { 0 , k n o w n } } ^ { \mathrm { s s } }$ is the clean latent extracted from the world-space latent. We then clamp the flow state $\mathbf { x } _ { t } ^ { \mathrm { { s s } } }$ predicted by the inpainter by replacing its known overlap:

$$
\tilde { \mathbf { x } } _ { t } ^ { \mathrm { s s } } = \mathbf { m } ^ { \mathrm { s s } } \odot \mathbf { x } _ { t } ^ { \mathrm { s s } } + \left( \mathbf { 1 } - \mathbf { m } ^ { \mathrm { s s } } \right) \odot \mathbf { x } _ { t , \mathrm { k n o w n } } ^ { \mathrm { s s } } .\tag{4}
$$

This mechanism preserves the previously generated overlap while allowing the unknown region to evolve along the denoising trajectory. After denoising, the resulting clean latent $\mathbf { z } ^ { \mathrm { s s } } = \tilde { \mathbf { x } } _ { 0 } ^ { \mathrm { s s } }$ is decoded into an occupancy grid ${ \bf o } ^ { \mathrm { s s } } = \mathcal { D } _ { \mathrm { s s } } ( { \bf z } ^ { \mathrm { s s } } )$ . Both are written only to unoccupied locations in the shared world-space tensors, which then provide context for subsequent blocks.

Geometry and Material World Generation. Given the completed world Sparse Structure latent, we process blocks in the same causal order. For each block, we generate or inpaint its Geometry latent, followed by its Material latent. To reduce block-boundary seams, each block’s clean latent $\mathbf { z } _ { b } ^ { s } .$ , where $s \in \{ \mathrm { g e o } , \mathrm { m a t } \}$ , is smoothly blended into the corresponding world latent $\mathbf { Z } _ { b } ^ { s }$ immediately after generation. The blending weights are determined by the distance to the block boundaries. Since blocks overlap only in the $\bar { X Y }$ plane, let $A \subseteq \{ x , y \}$ denote the overlapping axes, $\mathbf { u } = ( x , y , z )$ the world-grid coordinate, and $\pmb { \xi } _ { b } ( \mathbf { u } ) \in \{ 0 , \dots , \bar { G } - 1 \} ^ { 3 }$ the corresponding block-local latent-voxel coordinate, where G is the latent-grid resolution along each axis. As adjacent blocks overlap by one quarter of the block, the overlap spans $G / 4$ latent voxels. We define the linear feather weight as

$$
\alpha _ { b } ( { \mathbf { u } } ) = \prod _ { a \in \cal A } \mathrm { c l i p } \Bigg ( \frac { \operatorname* { m i n } \{ \xi _ { b , a } ( { \mathbf { u } } ) + \frac { 1 } { 2 } , ~ G - \xi _ { b , a } ( { \mathbf { u } } ) - \frac { 1 } { 2 } \} } { G / 4 } , 0 , 1 \Bigg ) ~ .\tag{5}
$$

This produces smooth linear transitions across seams and bilinear blending at block corners. Let $\mathbf { Z } _ { b } ^ { s } ( \mathbf { u } )$ denote the world latent after integrating block $b ,$ and let $A _ { b } ( { \mathbf { u } } )$ denote the accumulated weight, with $A _ { 0 } ( { \mathbf { u } } ) = 0$ . For each active world-grid location u, we update them online as

$$
\mathbf { Z } _ { b } ^ { s } ( \mathbf { u } ) = \frac { A _ { b - 1 } ( \mathbf { u } ) \mathbf { Z } _ { b - 1 } ^ { s } ( \mathbf { u } ) + \alpha _ { b } ( \mathbf { u } ) \mathbf { z } _ { b } ^ { s } ( \mathbf { u } ) } { A _ { b - 1 } ( \mathbf { u } ) + \alpha _ { b } ( \mathbf { u } ) } , \qquad A _ { b } ( \mathbf { u } ) = A _ { b - 1 } ( \mathbf { u } ) + \alpha _ { b } ( \mathbf { u } ) .\tag{6}
$$

Once the world latents are complete, we extract the aligned Geometry and Material latents for each block and decode them into block-local meshes with PBR materials. The block meshes are then transformed into the shared world frame and merged sequentially.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Datasets. We evaluate on unseen scenes from two datasets. From 3D-FRONT Fu et al. (2021), we hold out 30 scenes to assess performance on synthetic data. From ScanNet++ Yeshwanth et al.

Table 1: Comparisons on Synthetic Scenes. We evaluate 3D geometry and 2D depth and normal metrics on 30 scenes from 3D-FRONT, which is synthetic and large-scale.
<table><tr><td rowspan="2">Method</td><td colspan="3">3D Geometry</td><td>Coverage</td><td colspan="4">Depth</td><td colspan="3">Normal</td></tr><tr><td>CD↓</td><td>F-score↑</td><td>NC↑</td><td>Complete.↑</td><td>RMSE↓</td><td>AbsRel↓</td><td>LPIPS↓</td><td>CLIP↑</td><td>Angular↓</td><td>LPIPS↓</td><td>CLIP↑</td></tr><tr><td>VGGT-Ω+TSDF</td><td>0.2934</td><td>0.4596</td><td>0.5433</td><td>0.7363</td><td>0.5906</td><td>0.1548</td><td>0.4873</td><td>0.8679</td><td>20.9899</td><td>0.6188</td><td>0.8568</td></tr><tr><td>Murre</td><td>0.5295</td><td>0.2426</td><td>0.4729</td><td>0.4572</td><td>0.8847</td><td>0.2397</td><td>0.5273</td><td>0.8332</td><td>22.6896</td><td>0.6671</td><td>0.8542</td></tr><tr><td>MonoSDF</td><td>0.2558</td><td>0.4936</td><td>0.5730</td><td>0.9056</td><td>0.6052</td><td>0.1616</td><td>0.2566</td><td>0.9161</td><td>18.2223</td><td>0.3560</td><td>0.9062</td></tr><tr><td>FineRecon</td><td>0.2448</td><td>0.5301</td><td>0.6252</td><td>0.8126</td><td>0.5578</td><td>0.1490</td><td>0.3398</td><td>0.8906</td><td>17.6099</td><td>0.4419</td><td>0.8845</td></tr><tr><td>GenRecon</td><td>0.1861</td><td>0.6196</td><td>0.6128</td><td>0.9587</td><td>0.5348</td><td>0.1666</td><td>0.2548</td><td>0.9490</td><td>17.9822</td><td>0.3139</td><td>0.9543</td></tr><tr><td>WorldRecon (Ours)</td><td>0.1583</td><td>0.7271</td><td>0.6839</td><td>0.9589</td><td>0.3910</td><td>0.1126</td><td>0.1966</td><td>0.9582</td><td>16.4044</td><td>0.2951</td><td>0.9601</td></tr></table>

Table 2: Comparisons on Real-world Scenes. We evaluate 3D geometry and 2D depth and normal metrics on 30 scenes from Scannet++. Best results are bold, and second-best results are underlined.
<table><tr><td>Method</td><td colspan="3">3D Geometry</td><td>Coverage</td><td colspan="4">Depth</td><td colspan="3">Normal</td></tr><tr><td></td><td>CD↓</td><td>F-score↑</td><td>NC↑</td><td>Complete.↑</td><td>RMSE↓</td><td>AbsRel↓</td><td>LPIPS↓</td><td>CLIP↑</td><td>Angular↓</td><td>LPIPS↓</td><td>CLIP↑</td></tr><tr><td>VGGT-Ω+TSDF</td><td>0.1121</td><td>0.7467</td><td>0.7036</td><td>0.8737</td><td>0.3984</td><td>0.1433</td><td>0.3907</td><td>0.8991</td><td>25.4852</td><td>0.5075</td><td>0.8888</td></tr><tr><td>Murre</td><td>0.1756</td><td>0.5640</td><td>0.6222</td><td>0.5567</td><td>0.6619</td><td>0.1863</td><td>0.5587</td><td>0.8690</td><td>28.4193</td><td>0.5642</td><td>0.8761</td></tr><tr><td>MonoSDF</td><td>0.1665</td><td>0.6224</td><td>0.6913</td><td>0.9670</td><td>0.6293</td><td>0.3043</td><td>0.2705</td><td>0.9273</td><td>23.7527</td><td>0.3861</td><td>0.9016</td></tr><tr><td>FineRecon</td><td>0.1259</td><td>0.6861</td><td>0.7226</td><td>0.9388</td><td>0.4405</td><td>0.1673</td><td>0.2707</td><td>0.9183</td><td>22.6830</td><td>0.4043</td><td>0.8886</td></tr><tr><td>GenRecon</td><td>0.1227</td><td>0.5901</td><td>0.5969</td><td>0.9722</td><td>0.6837</td><td>0.3048</td><td>0.3862</td><td>0.9291</td><td>31.5722</td><td>0.4711</td><td>0.9172</td></tr><tr><td>WorldRecon (Ours)</td><td>0.0638</td><td>0.8363</td><td>0.7840</td><td>0.9918</td><td>0.3331</td><td>0.0992</td><td>0.2139</td><td>0.9522</td><td>20.0429</td><td>0.3631</td><td>0.9320</td></tr></table>

(2023), we evaluate 30 scenes to assess out-of-domain real-world performance. Each scene provides 32 unposed RGB images as input and 50 held-out RGB, depth, and normal views for evaluation.

Metrics. We evaluate reconstructed meshes in both 3D and 2D. In 3D, we measure geometric accuracy and coverage using Chamfer distance (CD), F-score (10 cm), and normal consistency (NC, thresholded at 20 cm). In 2D, we report geometric errors (RMSE, AbsRel, and angular normal error), computed over mutually valid pixels; perceptual and semantic metrics (LPIPS and CLIP) on depth and normal renderings; and completeness over valid reference pixels.

Baselines. We compare against five representative baselines spanning different reconstruction paradigms. VGGT-Ω Wang et al. (2026) predicts per-view depth maps and camera poses, then fuses them into a mesh via TSDF integration Newcombe et al. (2011) and Marching Cubes. All of the following methods consume its predictions as input. Murre Guo et al. (2025) conditions a large monocular depth estimator on the predicted point cloud and fuses the resulting depth into a mesh. MonoSDF Yu et al. (2022) optimizes a per-scene signed distance field guided by monocular depth and normal priors. FineRecon Stier et al. (2023) performs feed-forward TSDF regression conditioned on multi-view depth guidance. GenRecon Schmid et al. (2026) tiles the scene into overlapping chunks and projects multi-view image features into per-chunk 3D feature volumes.

Implementation details. We partition scenes into 4 m cubic blocks, each conditioned on up to 4 local views and an 8192-point cloud crop. Starting from the 512-resolution TRELLIS.2 pretrain model, we fully finetune the Sparse Structure and Geometry stages and apply LoRA Hu et al. (2021) to Material stage. At inference, we use a 12-step Heun sampler with guidance scales of 7.5, 7.5, and 3.0 and time-rescaling factors of 5.0, 3.0, and 3.0. All models are trained on 32 NVIDIA GPUs. Details are provided in Appendix.

## 4.2 QUANTITATIVE RESULTS

Synthetic scene reconstruction. We evaluate 2D and 3D reconstruction performance on 3D-FRONT Fu et al. (2021) using 32 unposed RGB images. The evaluation set primarily comprises large-scale multi-room scenes, with an average of 7.1 rooms and 62.1 m<sup>2</sup> of floor area. WorldRecon achieves the best overall performance, completing regions missing from the initial VGGT-Ω reconstruction. It substantially outperforms all baselines across the 3D geometry metrics while achieving the best results on all evaluated 2D depth and normal metrics, demonstrating accurate, consistent, and complete scene reconstruction.

Real-world scene reconstruction. We further evaluate generalization on ScanNet++ Yeshwanth et al. (2023). Despite being trained solely on synthetic scenes, WorldRecon reduces CD by 43.1% and improves F-score by 12.0% over the strongest baselines. Its rendered depth and normal maps also exhibit better geometric accuracy, perceptual similarity, and semantic alignment with the laserscanned ground truth, demonstrating strong sim-to-real generalization.

![](images/94e3972f0c2a173cade308c41d611395ae204ce1c60d98ad09c4a479dac07d3d.jpg)  
Figure 3: Qualitative Comparison. Mesh reconstruction results on evaluation scenes, including both synthetic and real-world examples. We show close-up views of interior details (top) and complete scene reconstructions (bottom). Compared with the baselines, WorldRecon produces more complete and geometrically consistent reconstructions with substantially fewer artifacts.

## <sup>1.</sup> <sup>RoomBlock</sup> <sup>rendering：2a7c</sup><sub>2.</sub> <sub>SAGE</sub> <sub>rendering：layout\_5b</sub>4.3 QUALITATIVE RESULTS

<sup>6.</sup> <sup>ScanNet</sup> <sup>orbit：9071e139d9</sup> <sup>/</sup> <sup>view\_0001</sup>We present qualitative comparisons on both synthetic and real-world scenes in Figure 3. As a reconstruction-based approach, VGGT-Ω+TSDF recovers visible surfaces but leaves unobserved re gions incomplete. Deviations in its predicted depths and camera poses can further propagate through TSDF fusion, resulting in noisy artifacts and distorted walls. Methods that use geometric predictions from VGGT-Ω, such as Murre and FineRecon, also exhibit artifacts and broken surfaces. GenRecon completes some unobserved regions; however, its feature-projection conditioning mechanism injects the distorted point cloud and inaccurate camera poses into the generative model, resulting in fragmented artifacts and distortions of the global scene layout. The optimisation-based MonoSDF completes missing regions and reduces geometric noise, but its over-smoothed surfaces lose fine geometric details. In comparison, WorldRecon produces more complete and geometrically coherent reconstructions, preserving sharper object boundaries, thin structures, and local details despite noise in the initial geometry and camera estimates.

Beyond geometric reconstruction, WorldRecon produces scene-level meshes with high-fidelity PBR materials. As illustrated in Figure 1, our method consistently reconstructs detailed geometry and texture from diverse input sources, including real-world captures, synthetic images, and frames from generated videos Seedance et al. (2026), demonstrating robust generalization across input domains. Additional qualitative results are provided in the Appendix. In particular, Figure S2 compares WorldRecon with an agentic baseline.

## 4.4 ABLATION STUDY

We ablate the key components of WorldRecon. The ablation experiments are conducted on complete scenes generated from 15 3D-FRONT scenes and 15 ScanNet++ scenes. Starting from the vanilla pretrained TRELLIS.2 Xiang et al. (2026), we progressively add four components:

Table 3: Ablation Study. We conduct quantitative ablation studies of the key components of WorldRecon on 30 scenes from 3D-FRONT and ScanNet++.
<table><tr><td colspan="4">Components</td><td colspan="3">3D Metrics</td><td colspan="5">2D Metrics</td></tr><tr><td>SS</td><td>Geo&amp;Mat</td><td>Inpaint</td><td>Stitch</td><td>CD↓</td><td>F-score↑</td><td>NC↑</td><td>Complete.↑</td><td>MAE↓</td><td>AbsRel↓</td><td>Angular↓</td><td>RGB CLIP↑</td></tr><tr><td>X</td><td>X</td><td>X</td><td>X</td><td>0.4003</td><td>0.2013</td><td>0.1682</td><td>0.7377</td><td>1.1847</td><td>0.6160</td><td>48.6471</td><td>0.7255</td></tr><tr><td>√</td><td>X</td><td>X</td><td>X</td><td>0.0996</td><td>0.7438</td><td>0.7173</td><td>0.9722</td><td>0.1659</td><td>0.0980</td><td>19.6434</td><td>0.8244</td></tr><tr><td>√</td><td>√</td><td>X</td><td>X</td><td>0.0954</td><td>0.7624</td><td>0.7349</td><td>0.9858</td><td>0.1589</td><td>0.0890</td><td>17.7610</td><td>0.8609</td></tr><tr><td>√</td><td>√</td><td>√</td><td>X</td><td>0.1023</td><td>0.7482</td><td>0.7202</td><td>0.9868</td><td>0.1439</td><td>0.0830</td><td>17.2462</td><td>0.8609</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>0.0830</td><td>0.8226</td><td>0.7640</td><td>0.9894</td><td>0.1394</td><td>0.0815</td><td>16.5789</td><td>0.8671</td></tr></table>

![](images/12f396f4644320666d130a5c52c596b6f337de4e90ec0c19902499200e65a449.jpg)  
Figure 4: Qualitative Ablation Study. We qualitatively compare the same ablation settings as in Table 3 on a 3D-FRONT scene unseen during training. We show the complete scene reconstruction for each variant and the ground truth, while colored circles highlight corresponding local details.

reconstruction-based sparse-structure generation (SS), reconstruction-based geometry and material generation (Geo+Mat), neighbour-aware latent inpainting (Inpaint), and stage-wise world latent clamping and blending (Stitch). Quantitative results are reported in Table 3, while Figure 4 visualizes the corresponding effects on representative scene structures and local details.

Although vanilla pretrained TRELLIS.2 has been trained on 3D-FRONT scene data, it performs poorly in our task because its single-image conditioning provides only partial observations and does not enforce a consistent world orientation. Our reconstruction-based SS generator instead uses the reconstructed point cloud as a spatial prior while leveraging the generative prior to plausibly complete regions missing from the initial reconstruction. Our reconstruction-based Geo&Mat generators further exploit multi-view observations to recover finer geometry and more faithful materials. Nevertheless, because the blocks are generated independently, inconsistencies can remain in their overlapping regions. As shown in Figure 4, adjacent blocks may exhibit misaligned surface heights at their interfaces, accompanied by visible discontinuities in geometry and texture. Adding Inpaint propagates context between neighbouring blocks, encouraging their surfaces to connect coherently and improving the depth and normal metrics. However, small inconsistencies at block boundaries may remain visible as seams, explaining the slight degradation in the 3D metrics reported in Table 3. The complete WorldRecon pipeline employs stage-wise world latent clamping and blending (Stitch) to maintain cross-block continuity and global scene coherence, thereby eliminating visible boundary discontinuities. It consequently achieves the best performance across all reported metrics and produces the reconstruction most consistent with the ground truth.

## 5 CONCLUSION

We presented WorldRecon, a generative reconstruction framework for recovering complete, largescale scene PBR meshes from unposed RGB images. WorldRecon leverages coarse geometry and camera estimates predicted by a reconstruction model as effective global grounding despite their imperfections. Based on this global scaffold, reconstruction-based block generation produces faithful mesh blocks with detailed geometry and PBR materials. Neighbour-aware latent inpainting and the causal world-generation pipeline stitch these blocks together, ensuring consistent transitions across block boundaries. WorldRecon achieves state-of-the-art performance in large-scale indoor scene reconstruction, with strong sim-to-real generalization despite training exclusively on synthetic data. More broadly, our work provides a practical path toward reconstructing editable 3D scene assets from casually captured or generated videos, supporting downstream applications such as simulation, embodied AI, and content creation.

Limitations and Future Work. As our training data comprise only indoor scenes, the current applicability of WorldRecon is limited to indoor environments, and its generalization to outdoor scenes remains to be established. In addition, our current scene representation comprises a single layer of 4 m-high blocks, restricting reconstruction to single-story environments and precluding the modeling of multi-story buildings. Extending the training distribution to more diverse indoor, outdoor, and multi-story environments, together with supporting block generation along the vertical dimension, constitutes an important direction for future work.

## REFERENCES

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5470–5479, 2022.

Jiahao Chang, Chongjie Ye, Yushuang Wu, Yuantao Chen, Yidan Zhang, Zhongjin Luo, Chenghong Li, Yihao Zhi, and Xiaoguang Han. Reconviagen: Towards accurate multi-view 3d object reconstruction via generation. arXiv preprint arXiv:2510.23306, 2025.

David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19457–19467, 2024.

Hanke Chen, Yuan Liu, and Minchen Li. Trellisworld: Training-free world generation from object generators. arXiv preprint arXiv:2510.23880, 2025.

Maximilian Denninger, Dominik Winkelbauer, Martin Sundermeyer, Wout Boerdijk, Markus Knauer, Klaus H. Strobl, Matthias Humt, and Rudolph Triebel. Blenderproc2: A procedural pipeline for photorealistic rendering. Journal of Open Source Software, 8(82):4901, 2023. doi: 10.21105/joss.04901. URL https://doi.org/10.21105/joss.04901.

Paul Engstler, Aleksandar Shtedritski, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Syncity: Training-free generation of 3d worlds. In Proceedings of the International Conference on Computer Vision (ICCV), 2025.

Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, et al. 3d-front: 3d furnished rooms with layouts and semantics. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 10933– 10942, 2021.

Yasutaka Furukawa and Carlos Hernandez. Multi-view stereo: A tutorial. ´ Found. Trends Comput. Graph. Vis., 9(1-2):1–148, 2015.

Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024.

Haoyu Guo, He Zhu, Sida Peng, Haotong Lin, Yunzhi Yan, Tao Xie, Wenguan Wang, Xiaowei Zhou, and Hujun Bao. Multi-view reconstruction via sfm-guided monocular depth estimation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5272–5282, 2025.

Lukas Hollein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Ex-¨ tracting textured 3d meshes from 2d text-to-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7909–7920, 2023.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https: //arxiv.org/abs/2106.09685.

Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pp. 1–11, 2024.

Zhisheng Huang, Jiahao Chen, Cheng Lin, Chenyu Hu, Hanzhuo Huang, Zhengming Yu, Mengfei Li, Yuheng Liu, Zekai Gu, Zibo Zhao, et al. Unirecgen: unifying multi-view 3d reconstruction and generation. arXiv preprint arXiv:2604.01479, 2026.

Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. Lvsm: A large view synthesis model with minimal 3d inductive bias. arXiv preprint arXiv:2410.17242, 2024.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, George Drettakis, et al. 3d gaussian splat-¨ ting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

Vincent Leroy, Yohann Cabon, and Jer´ ome Revaud. Grounding image matching in 3d with mast3r.ˆ In European conference on computer vision, pp. 71–91. Springer, 2024.

Sikuang Li, Chen Yang, Jiemin Fang, Taoran Yi, Jia Lu, Jiazhong Cen, Lingxi Xie, Wei Shen, and Qi Tian. Worldgrow: Generating infinite 3d world. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 6433–6441, 2026.

Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. In Interna tional conference on learning representations, volume 2024, pp. 27676–27697, 2024.

Quan Meng, Lei Li, Matthias Nießner, and Angela Dai. Lt3sd: Latent trees for 3d scene diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 650–660, 2025.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications ofthe ACM, 65(1):99–106, 2021.

Richard A Newcombe, Shahram Izadi, Otmar Hilliges, David Molyneaux, David Kim, Andrew J Davison, Pushmeet Kohi, Jamie Shotton, Steve Hodges, and Andrew Fitzgibbon. Kinectfusion: Real-time dense surface mapping and tracking. In 2011 10th IEEE international symposium on mixed and augmented reality, pp. 127–136. Ieee, 2011.

Linfei Pan, Daniel Barath, Marc Pollefeys, and Johannes Lutz Schonberger. Global Structure-from-¨ Motion Revisited. In European Conference on Computer Vision (ECCV), 2024.

Xuanchi Ren, Yifan Lu, Hanxue Liang, Zhangjie Wu, Huan Ling, Mike Chen, Sanja Fidler, Francis Williams, and Jiahui Huang. Scube: Instant large-scale scene reconstruction using voxsplats. Advances in Neural Information Processing Systems, 37:97670–97698, 2024.

Katharina Schmid, Nicolas von Lutzow, Jozef Hladk¨ y, Angela Dai, and Matthias Nießner. Gen-\` recon: Bridging generative priors for multi-view 3d scene reconstruction. arXiv preprint arXiv:2605.23888, 2026.

Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pp. 4104–4113, 2016.

Johannes L Schonberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view¨ selection for unstructured multi-view stereo. In European conference on computer vision, pp. 501–518. Springer, 2016.

Jonas Schult, Sam Tsai, Lukas Hollein, Bichen Wu, Jialiang Wang, Chih-Yao Ma, Kunpeng Li, Xi-¨ aofang Wang, Felix Wimbauer, Zijian He, et al. Controlroom3d: Room generation using semantic proxy rooms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6201–6210, 2024.

Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, Mojie Chi, Xuyan Chi, Jian Cong, Qinpeng Cui, Fei Ding, Qide Dong, Yujiao Du, Haojie Duanmu, Junliang Fan, Jiarui Fang, Jing Fang, Zetao Fang, Chengjian Feng, Yu Gao, Diandian Gu, Dong Guo, Hanzhong Guo, Qiushan Guo, Boyang Hao, Hongxiang Hao, Haoxun He, Jiaao He, Qian He, Tuyen Hoang, Heng Hu, Ruoqing Hu, Yuxiang Hu, Jiancheng Huang, Weilin Huang, Zhaoyang Huang, Zhongyi Huang, Jishuo Jin, Ming Jing, Ashley Kim, Shanshan Lao, Yichong Leng, Bingchuan Li, Gen Li, Haifeng Li, Huixia Li, Jiashi Li, Ming Li, Xiaojie Li, Xingxing Li, Yameng Li, Yiying Li, Yu Li, Yueyan Li, Chao Liang, Han Liang, Jianzhong Liang, Ying Liang, Wang Liao, J. H. Lien, Shanchuan Lin, Xi Lin, Feng Ling, Yue Ling, Fangfang Liu, Jiawei Liu, Jihao Liu, Jingtuo Liu, Shu Liu, Sichao Liu, Wei Liu,

Xue Liu, Zuxi Liu, Ruijie Lu, Lecheng Lyu, Jingting Ma, Tianxiang Ma, Xiaonan Nie, Jingzhe Ning, Junjie Pan, Xitong Pan, Ronggui Peng, Xueqiong Qu, Yuxi Ren, Yuchen Shen, Guang Shi, Lei Shi, Yinglong Song, Fan Sun, Li Sun, Renfei Sun, Wenjing Tang, Boyang Tao, Zirui Tao, Dongliang Wang, Feng Wang, Hulin Wang, Ke Wang, Qingyi Wang, Rui Wang, Shuai Wang, Shulei Wang, Weichen Wang, Xuanda Wang, Yanhui Wang, Yue Wang, Yuping Wang, Yuxuan Wang, Zijie Wang, Ziyu Wang, Guoqiang Wei, Meng Wei, Di Wu, Guohong Wu, Hanjie Wu, Huachao Wu, Jian Wu, Jie Wu, Ruolan Wu, Shaojin Wu, Xiaohu Wu, Xinglong Wu, Yonghui Wu, Ruiqi Xia, Xin Xia, Xuefeng Xiao, Shuang Xu, Bangbang Yang, Jiaqi Yang, Runkai Yang, Tao Yang, Yihang Yang, Zhixian Yang, Ziyan Yang, Fulong Ye, Bingqian Yi, Xing Yin, Yongbin You, Linxiao Yuan, Weihong Zeng, Xuejiao Zeng, Yan Zeng, Siyu Zhai, Zhonghua Zhai, Bowen Zhang, Chenlin Zhang, Heng Zhang, Jun Zhang, Manlin Zhang, Peiyuan Zhang, Shuo Zhang, Xiaohe Zhang, Xiaoying Zhang, Xinyan Zhang, Xinyi Zhang, Yichi Zhang, Zixiang Zhang, Haiyu Zhao, Huating Zhao, Liming Zhao, Yian Zhao, Guangcong Zheng, Jianbin Zheng, Xiaozheng Zheng, Zerong Zheng, Kuan Zhu, and Feilong Zuo. Seedance 2.0: Advancing video generation for world complexity, 2026. URL https://arxiv.org/abs/2604.14148.

Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose,´ Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, Francisco Massa, Daniel¨ Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve´ Jegou, Patrick Labatut, and Piotr Bojanowski. DINOv3, 2025. URL ´ https://arxiv.org/ abs/2508.10104.

Noah Stier, Anurag Ranjan, Alex Colburn, Yajie Yan, Liang Yang, Fangchang Ma, and Baptiste Angles. Finerecon: Depth-aware feed-forward network for detailed 3d reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18423–18432, 2023.

Jiapeng Tang, Yinyu Nie, Lev Markhasin, Angela Dai, Justus Thies, and Matthias Nießner. Diffuscene: Denoising diffusion models for generative indoor scene synthesis. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 20507–20518, 2024a.

Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision, pp. 1–18. Springer, 2024b.

Yolo Y. Tang, Daiki Shimada, Jiayue Meng, Jing Bi, Pinxin Liu, Yicheng Wang, Yunzhong Xiao, Zhangyun Tan, Zeliang Zhang, Chao Huang, Susan Liang, Qianxiang Shen, Luchuan Song, Ali Vosoughi, Mingqian Feng, Melika Filvantorkaman, and Chenliang Xu. Bvb: Benchmarking agentic video understanding via programmatic reconstruction in blender, 2026. URL https: //arxiv.org/abs/2609.15478.

Tencent Hunyuan3D Team. Hunyuan3d-omni: A unified framework for controllable generation of 3d assets, 2025. URL https://arxiv.org/abs/2509.21245.

Fangjinhua Wang, Silvano Galliani, Christoph Vogel, Pablo Speciale, and Marc Pollefeys. Patchmatchnet: Learned multi-view patchmatch stereo. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14194–14203, 2021a.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5294–5306, 2025.

Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schonberger, Patrick¨ Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, and Christian Rupprecht. Vggtomega. arXiv preprint arXiv:2605.15195, 2026.

Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021b.

Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibrnet: Learning multiview image-based rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4690–4699, 2021c.

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 20697–20709, 2024.

Xinpeng Wang, Chandan Yeshwanth, and Matthias Nießner. Sceneformer: Indoor scene generation with transformers. In 2021 International conference on 3D vision (3DV), pp. 106–115. IEEE, 2021d.

Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. Reconfusion: 3d reconstruction with diffusion priors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 21551–21561, 2024a.

Zhennan Wu, Yang Li, Han Yan, Taizhang Shang, Weixuan Sun, Senbo Wang, Ruikai Cui, Weizhe Liu, Hiroyuki Sato, Hongdong Li, et al. Blockfusion: Expandable 3d scene generation using latent tri-plane extrapolation. ACM Transactions on Graphics (ToG), 43(4):1–17, 2024b.

Hongchi Xia, Xuan Li, Zhaoshuo Li, Qianli Ma, Jiashu Xu, Ming-Yu Liu, Yin Cui, Tsung-Yi Lin, Wei-Chiu Ma, Shenlong Wang, et al. Sage: Scalable agentic 3d scene generation for embodied ai. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22358–22368, 2026.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 21469– 21480, 2025.

Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, et al. Native and compact structured latents for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14419–14429, 2026.

Tian-Xing Xu, Xiangjun Gao, Wenbo Hu, Xiaoyu Li, Song-Hai Zhang, and Ying Shan. Geometrycrafter: Consistent geometry estimation for open-world videos with diffusion priors. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6632–6644, 2025.

Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multi-view stereo. In Proceedings of the European conference on computer vision (ECCV), pp. 767–783, 2018.

Yao Yao, Zixin Luo, Shiwei Li, Tianwei Shen, Tian Fang, and Long Quan. Recurrent mvsnet for high-resolution multi-view stereo depth inference. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5525–5534, 2019.

Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Volume rendering of neural implicit surfaces. In Thirty-Fifth Conference on Neural Information Processing Systems, 2021.

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A highfidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 12–22, 2023.

Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger. Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction. Advances in neural information processing systems, 35:25018–25032, 2022.

Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4):1–20, 2024.

Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Long-sequence large reconstruction model for wide-coverage gaussian splats. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 4349–4359, 2025.

## APPENDIX

The appendix presents supplementary material related to data curation, the WorldRecon method, experiments details, and additional qualitative cases. Section A details the block-level training data construction pipeline, including scene rendering, scene slicing, multi-view condition extraction, and point-cloud preparation. Section B provides further implementation details of the WorldRecon framework, covering reconstruction-based conditioning, neighbour-aware latent inpainting, training configurations, and the world generation pipeline. Section C describes the evaluation protocol, including baseline settings, the alignment procedure, and metric definitions. Section D presents additional qualitative results, including large-scale scene generation, a comparison with an agentic LLM baseline, and failure case analysis.

## A DATA CURATION DETAILS

We construct the block-level training set from two synthetic indoor scene datasets, 3D-FRONT Fu et al. (2021) and SAGE-10k Xia et al. (2026), as described in the main paper. We detail its rendering, scene-slicing, and condition-extraction procedures below.

## A.1 SCENE RENDERING

For each scene, we render a set of full-scene RGB observations to simulate the multi-view images used for room reconstruction. We additionally render the corresponding depth maps for point-cloud construction during data preparation.

To increase the appearance diversity of 3D-FRONT, we augment its architectural surfaces with PBR materials from ambientCG (formerly CC0 Textures)<sup>1</sup>. For each scene, we construct separate material pools for floors (wood, planks, carpet, tiles, marble, and terrazzo), walls (wallpaper, marble, and plaster), and structural elements such as doors and baseboards (wood and marble). We first sample a scene-level material-assignment strategy

$$
s \sim \mathrm { C a t e g o r i c a l } ( 0 . 3 , 0 . 6 , 0 . 1 ) , \qquad s \in \{ i n s t a n c e , c a t e g o r y , o r i g i n a l \} .\tag{S1}
$$

Under the instance strategy, each architectural component independently draws a material from its corresponding pool. Under the category strategy, components belonging to the same category share a sampled material, producing a more coherent appearance. The original strategy applies no additional material augmentation and preserves the scene as provided by 3D-FRONT. When either instance or category strategy is selected, floors are always assigned a new material, while walls and structural elements are augmented with probabilities 0.6 and 0.8, respectively. Furniture is excluded from material augmentation and always retains its original 3D-FUTURE materials.

For illumination, in addition to the original light sources provided by each 3D-FRONT scene, we assign a randomly selected HDRI environment map. SAGE-10k scenes additionally receive one ceiling area light because their enclosed layouts receive limited illumination from the environment map. We randomise the horizontal position, size, energy, and warm/cool colour of each area light.

To mimic natural human capture, we sample near-level camera poses inside individual rooms at heights between 1.4m and 1.8m. Subject to the availability of valid poses, views are distributed across rooms approximately in proportion to their floor areas. We reject poses closer than 1m to scene geometry to avoid overly close-up observations and encourage broader scene coverage.

We render RGB and depth maps with BlenderProc Denninger et al. (2023) at 1024×1024 resolution and a 90<sup>◦</sup> field of view (FoV). For every accepted view, we also store the camera intrinsics and camera-to-world transformation.

## A.2 SCENE SLICING

We divide each scene into overlapping axis-aligned cubic blocks of side length $\ell = 4 \mathrm { m }$ . For each horizontal axis $a \in \{ x , y \}$ , let $[ \boldsymbol { a } _ { \mathrm { m i n } } , \boldsymbol { a } _ { \mathrm { m a x } } ]$ denote the scene extent and $d _ { a } = a _ { \operatorname* { m a x } } - a _ { \operatorname* { m i n } }$ . The

number of blocks $n _ { a } .$ , stride $s _ { a } ,$ , and block starting positions $p _ { i } ^ { a }$ are defined as

$$
n _ { a } = \left\lceil \frac { d _ { a } } { \ell } \right\rceil , \qquad s _ { a } = \left\{ \begin{array} { l l } { 0 , } & { n _ { a } = 1 , } \\ { \frac { d _ { a } - \ell } { n _ { a } - 1 } , } & { n _ { a } > 1 , } \end{array} \right. \qquad p _ { i } ^ { a } = \left\{ \begin{array} { l l } { \frac { a _ { \operatorname* { m i n } } + a _ { \operatorname* { m a x } } - \ell } { 2 } , } & { n _ { a } = 1 , } \\ { a _ { \operatorname* { m i n } } + i s _ { a } , } & { n _ { a } > 1 , } \end{array} \right.\tag{S2}
$$

where $i = 0 , \ldots , n _ { a } - 1$ . This construction uses the minimum number of blocks required to cover the scene and distributes their overlap uniformly. We retain only scenes with a height below 4m, so a single block centred on the scene is sufficient along the vertical axis.

For block index $( i , j )$ , the starting positions from Equation (S2) determine its world-space bounds:

$$
\mathbf { b } _ { \operatorname* { m i n } } ^ { i , j } = \left( p _ { i } ^ { x } , p _ { j } ^ { y } , \frac { z _ { \operatorname* { m i n } } + z _ { \operatorname* { m a x } } - \ell } { 2 } \right) , \qquad \mathbf { b } _ { \operatorname* { m a x } } ^ { i , j } = \mathbf { b } _ { \operatorname* { m i n } } ^ { i , j } + \ell \mathbf { 1 } .\tag{S3}
$$

We construct each block mesh by clipping the complete scene mesh against four axis-aligned planes of its bounding box. Objects intersecting a block boundary are geometrically cropped rather than assigned wholly to one block, while their original material attributes are preserved. The newly created boundary surfaces are left uncapped.

The block centre is $\mathbf { c } ^ { i , j } = ( \mathbf { b } _ { \operatorname* { m i n } } ^ { i , j } + \mathbf { b } _ { \operatorname* { m a x } } ^ { i , j } ) / 2$ . A world-space point x is then mapped to the canonical block frame by

$$
\bar { \mathbf { x } } = \frac { \mathbf { x } - \mathbf { c } ^ { i , j } } { \ell } ,\tag{S4}
$$

such that every block occupies $[ - 0 . 5 , 0 . 5 ] ^ { 3 }$ . The same scale and translation are applied to meshes, point clouds, and camera-to-world translations.

## A.3 MULTI-VIEW CONDITION EXTRACTION

For each rendered image $I _ { v } ,$ , let $\mathbf { u } = ( u _ { x } , u _ { y } ) ^ { \top } , \tilde { \mathbf { u } } = ( u _ { x } , u _ { y } , 1 ) ^ { \top }$ , and $d _ { v } ( { \bf u } )$ denote a pixel, its homogeneous coordinate, and its metric depth, respectively. Given the camera intrinsics $\mathbf { K } _ { \imath }$ <sub>v</sub> and camera-to-world transformation $\left[ \mathbf { R } _ { v } \mid \mathbf { t } _ { v } \right]$ , we unproject the pixel into world space as

$$
\mathbf { x } _ { v } ( \mathbf { u } ) = \mathbf { R } _ { v } \left( d _ { v } ( \mathbf { u } ) \mathbf { K } _ { v } ^ { - 1 } \tilde { \mathbf { u } } \right) + \mathbf { t } _ { v } .\tag{S5}
$$

Given the bounds $[ \mathbf { b } _ { \operatorname* { m i n } } ^ { i , j } , \mathbf { b } _ { \operatorname* { m a x } } ^ { i , j } ]$ of block $( i , j )$ , we define its visibility mask as

$$
M _ { i , j , v } ( \mathbf { u } ) = \mathbf { 1 } \Big [ \mathbf { b } _ { \operatorname* { m i n } } ^ { i , j } \leq \mathbf { x } _ { v } ( \mathbf { u } ) \leq \mathbf { b } _ { \operatorname* { m a x } } ^ { i , j } \Big ] .\tag{S6}
$$

Views are ranked by $\begin{array} { r } { \sum _ { \mathbf { u } } M _ { i , j , v } ( \mathbf { u } ) } \end{array}$ , and only those for which at least 25% of the image pixels fall inside the block are considered valid. We retain up to 16 valid views and use $M _ { i , j , v }$ as the alpha channel of the corresponding RGB image.

We transform the camera-to-world translation to the canonical block frame using Equation (S4), while preserving the camera rotation. Camera intrinsics are rescaled to the resolution used by the image encoder. Thus, the RGBA images, cameras, point cloud, and target latents are expressed in the same block-local coordinate frame.

## A.4 POINT-CLOUD PREPARATION

For each block, we construct two point-cloud conditions. To obtain the augmented depth-derived cloud, we randomly select 6–10 retained views and apply depth and camera perturbations, visibility dropout, and sparse outlier injection before unprojection. We unproject the perturbed views into the canonical block frame, merge and crop the points to $[ - 0 . 5 , 0 . 5 ] ^ { 3 }$ , and sparsify the cloud to 8192 points. The VGGT-Ω cloud is reconstructed from the retained block-masked RGB views described above. We first apply confidence filtering and statistical outlier removal. We then estimate a Horn Sim(3) transformation from the corresponding VGGT-predicted and canonical camera centres, jointly recovering the relative scale, rotation, and translation between the two coordinate frames. The estimated transformation is applied to the reconstructed point cloud, which is subsequently cropped to the canonical block bounds and sparsified to 8192 points.

When both sources are available, they are registered as separate training instances, ensuring balanced exposure to both point-cloud distributions within each epoch. For blocks with only one valid source, the available point cloud is used.

Table S1: Training configurations. We report the image and latent resolutions, point-cloud size, number of GPUs, global batch size, learning rate, EMA decay, and adaptation strategy for each generation and inpainting stage.
<table><tr><td>Stage</td><td>Image</td><td>Latent</td><td>Points</td><td>GPUs</td><td>Global Batch</td><td>LR</td><td>EMA</td><td>Adaptation</td></tr><tr><td>SS generation</td><td>512</td><td> $\overline { { { 1 6 } ^ { 3 } } }$ </td><td>8192</td><td>32</td><td>128</td><td> $\overline { { 1 \times 1 0 ^ { - 4 } } }$ </td><td>0.9999</td><td>full</td></tr><tr><td>SS inpainting</td><td>512</td><td> $1 6 ^ { 3 }$ </td><td>8192</td><td>32</td><td>128</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td>0.9999</td><td>full</td></tr><tr><td>Geometry generation</td><td>512</td><td> $3 2 ^ { 3 }$ </td><td></td><td>16</td><td>64</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.9995</td><td>full</td></tr><tr><td>Geometry inpainting</td><td>512</td><td> $3 2 ^ { 3 }$ </td><td>一</td><td>16</td><td>64</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td>0.9995</td><td>full</td></tr><tr><td>Material generation</td><td>512</td><td> $3 2 ^ { 3 }$ </td><td></td><td>16</td><td>96</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td>0.9995</td><td>LoRA</td></tr><tr><td>Material inpainting</td><td>512</td><td> $3 2 ^ { 3 }$ </td><td>一</td><td>16</td><td>96</td><td> $1 \times 1 0 ^ { - 5 }$ </td><td>0.9995</td><td>LoRA</td></tr></table>

## B WORLDRECON METHOD DETAILS

## B.1 RECONSTRUCTION-BASED CONDITIONING

Multi-View Image Conditioning. For each conditioning image, we extract patch features using a frozen DINOv3 ViT-L/16 encoder Simeoni et al. (2025) at´ $5 1 2 \times 5 1 2$ resolution.

To encode the camera pose, we sample five image-plane locations comprising the image centre and four corners. Given a homogeneous pixel coordinate $\tilde { \mathbf { u } } _ { v , k }$ , camera intrinsics $\mathbf { K } _ { v } ,$ camera-to-world rotation $\mathbf { R } _ { v } ,$ , and camera centre $\mathbf { o } _ { v } .$ , the corresponding world-space ray and Plucker coordinate are¨

$$
\mathbf { d } _ { v , k } = \frac { \mathbf { R } _ { v } \mathbf { K } _ { v } ^ { - 1 } \tilde { \mathbf { u } } _ { v , k } } { \left\| \mathbf { R } _ { v } \mathbf { K } _ { v } ^ { - 1 } \tilde { \mathbf { u } } _ { v , k } \right\| _ { 2 } } , \qquad \boldsymbol { \pi } _ { v , k } = \left( \mathbf { d } _ { v , k } , \mathbf { o } _ { v } \times \mathbf { d } _ { v , k } \right) .\tag{S7}
$$

A two-layer MLP projects each Plucker coordinate into a camera token, which is prepended to the¨ corresponding image-patch tokens. A four-layer cross-attention module then uses learnable queries to fuse all views into a fixed-length conditioning sequence.

Point Cloud Condition. The SS stage additionally receives the XYZ coordinates of an 8192-point block cloud $\mathcal { P } _ { b } = \{ \mathbf { p } _ { j } \} _ { j = 1 } ^ { 8 1 9 2 }$ . Following Hunyuan3D-Omni Team (2025), we encode the normalised coordinates using Fourier features and append learned modality tokens:

$$
\mathbf { c } ^ { \mathrm { p c } } = \left[ \psi ( \gamma ( \mathbf { p } _ { 1 } ) ) , \ldots , \psi ( \gamma ( \mathbf { p } _ { 8 1 9 2 } ) ) , \mathbf { e } _ { 1 } , \ldots , \mathbf { e } _ { 1 0 } \right] ,\tag{S8}
$$

where $\gamma ( \cdot )$ denotes the Fourier positional encoding, $\psi ( \cdot )$ is a learnable projection, and $\mathbf { e } _ { 1 : 1 0 }$ are modality tokens. All tokens share the 1024-dimensional feature space of the image condition. We drop the complete point-cloud stream with probability 0.1 during training.

## B.2 NEIGHBOUR-AWARE LATENT INPAINTING

We use the concat-inpainting formulation and overlap masks defined in the main paper for all three stages. Let $\mathbf { x } _ { 0 , \mathrm { k n o w n } } ^ { s } = \mathbf { x } _ { 0 } ^ { s } \odot \left( \mathbf { 1 } - \mathbf { m } ^ { s } \right)$ denote the known latent provided by previously generated neighbours for stage $s \in \{ \mathrm { s s } , \mathrm { g e o } , \mathrm { m a t } \}$ , where $\mathbf { m } ^ { s } = 0$ marks known locations and $\mathbf { m } ^ { s } = 1$ marks locations to be generated. The input features for the three inpainters are defined separately as

$$
\begin{array} { r } { \mathbf { h } _ { \mathrm { i n } } ^ { \mathrm { s s } } = f _ { \mathrm { i n } } ^ { \mathrm { s s } } \left( \left[ \mathbf { x } _ { t } ^ { \mathrm { s s } } , \mathbf { m } ^ { \mathrm { s s } } , \mathbf { x } _ { 0 , \mathrm { k n o w n } } ^ { \mathrm { s s } } \right] \right) , } \end{array}\tag{S9}
$$

$$
\begin{array} { r } { \mathbf { h } _ { \mathrm { i n } } ^ { \mathrm { g e o } } = f _ { \mathrm { i n } } ^ { \mathrm { g e o } } \left( \left[ \mathbf { x } _ { t } ^ { \mathrm { g e o } } , \mathbf { m } ^ { \mathrm { g e o } } , \mathbf { x } _ { 0 , \mathrm { k n o w n } } ^ { \mathrm { g e o } } \right] \right) , } \end{array}\tag{S10}
$$

$$
{ \bf h } _ { \mathrm { i n } } ^ { \mathrm { m a t } } = f _ { \mathrm { i n } } ^ { \mathrm { m a t } } \left( \left[ { \bf x } _ { t } ^ { \mathrm { m a t } } , { \bf z } ^ { \mathrm { g e o } } , { \bf m } ^ { \mathrm { m a t } } , { \bf x } _ { 0 , \mathrm { k n o w n } } ^ { \mathrm { m a t } } \right] \right) .\tag{S11}
$$

Here, $\mathbf { z } ^ { \mathrm { g e o } }$ is the native Geometry condition of the Material stage.

## B.3 TRAINING DETAILS

Table S1 summarises the final 512-resolution configurations used for the reported reconstruction results. All models are trained with AdamW and BF16 automatic mixed precision. We use adaptive gradient clipping at the 95-th percentile with a maximum norm of 1.0, and unconditional condition dropout of 0.1. Geometry and Material training samples are capped at 16k active sparse tokens.

The Sparse Structure and Geometry stages are fully fine-tuned, whereas the Material denoisers are adapted using rank-8, scale-16 LoRA modules inserted into their attention projections. For the Material stages, the frozen base denoiser is accompanied by trainable multi-view fusion and cameratokenization modules, whose learning rates are $1 0 ^ { - 4 }$ for generation and $3 \times 1 0 ^ { - 5 }$ for inpainting. The table reports the corresponding LoRA learning rates.

## B.4 WORLD GENERATION PIPELINE

Global Reconstruction and Scene Alignment. Given N unposed RGB images, we resize them to the 512-pixel input resolution of VGGT-Ω. A single forward pass predicts a depth map $\hat { \mathbf { D } } _ { v }$ , depth confidence $\hat { \mathbf { C } } _ { v } .$ , camera intrinsics $\hat { \mathbf { K } } _ { v } ,$ , and world-to-camera extrinsics $\hat { \mathbf { E } } _ { v } = [ \hat { \mathbf { R } } _ { v } \mid \hat { \mathbf { t } } _ { v } ]$ for each view. We unproject the predicted depths using the corresponding camera parameters to obtain world-space point maps while preserving pixel-to-3D correspondences for constructing block-specific image conditions. We merge points satisfying $\hat { \mathbf { C } } _ { v } ( \mathbf { u } ) \geq 2 . 0$ across all views and apply statistical outlier removal to suppress isolated depth artifacts. Finally, we apply connected-component filtering to the aligned global point cloud and retain its largest component as the scene scaffold.

The VGGT-Ω reconstruction is not necessarily aligned with the canonical scene axes. We therefore rotate the predicted point cloud such that the gravity direction is aligned with the positive Z-axis and the dominant wall directions are aligned with the X- and Y-axes, and then recover its physical scale. We first select an initial up direction from the three PCA axes of the global point cloud:

$$
\bar { \bf y } = \frac { \sum _ { v } \hat { \bf R } _ { v } ^ { \top } { \bf e } _ { y } } { \left\| \sum _ { v } \hat { \bf R } _ { v } ^ { \top } { \bf e } _ { y } \right\| _ { 2 } } , \qquad k ^ { \star } = \arg \operatorname* { m a x } _ { k \in \{ 1 , 2 , 3 \} } \left| { \bf v } _ { k } ^ { \top } \bar { \bf y } \right| , \qquad { \bf g } _ { 0 } = - \mathrm { s i g n } \big ( { \bf v } _ { k \star } ^ { \top } \bar { \bf y } \big ) { \bf v } _ { k \star } .\tag{S12}
$$

Here, y¯ is the average predicted camera $Y .$ -axis expressed in world coordinates, $\mathbf { v } _ { k }$ is the k-th PCA axis, and ${ \bf g } _ { 0 }$ is the selected initial up direction. The negative sign accounts for the OpenCV convention in which the camera Y-axis points downward. After rotating ${ \bf g } _ { 0 }$ to $\mathbf { e } _ { z } = ( 0 , 0 , \bar { 1 } ) ^ { \top }$ , we refine the up direction using approximately horizontal surfaces in the lower portion of the scene:

$$
\mathcal { H } = \{ i \in \mathcal { P } : | { \bf n } _ { i } ^ { \top } { \bf e } _ { z } | > \tau _ { h } \} , \qquad { \bf g } = \frac { \sum _ { i \in \mathcal { H } } \mathrm { s i g n } ( { \bf n } _ { i } ^ { \top } { \bf e } _ { z } ) | { \bf n } _ { i } ^ { \top } { \bf e } _ { z } | { \bf n } _ { i } } { \left\| \sum _ { i \in \mathcal { H } } \mathrm { s i g n } ( { \bf n } _ { i } ^ { \top } { \bf e } _ { z } ) | { \bf n } _ { i } ^ { \top } { \bf e } _ { z } | { \bf n } _ { i } \right\| _ { 2 } } .\tag{S13}
$$

Here, n<sub>i</sub> is the local surface normal, H retains normals satisfying the horizontal-surface threshold $\tau _ { h } = 0 . 7$ . Their consistently oriented, weighted average gives the refined up direction g, which is aligned with $\mathbf { e } _ { z }$ . We then estimate the dominant direction of wall-like normals:

$$
\begin{array} { r l r } & { \mathcal { W } = \left\{ i : \left| \mathbf { n } _ { i } ^ { \top } \mathbf { e } _ { z } \right| < \tau _ { w } \right\} , } & { \phi _ { i } = \mathrm { a t a n 2 } ( n _ { i , y } , n _ { i , x } ) \mathrm { m o d } \displaystyle \frac { \pi } { 2 } , } \\ & { \phi ^ { \star } = \underset { \phi \in [ 0 , \pi / 2 ) } { \arg \operatorname* { m a x } } \sum _ { i \in \mathcal { W } } \| \mathbf { n } _ { i , x y } \| _ { 2 } K _ { \sigma } ^ { \pi / 2 } ( \phi - \phi _ { i } ) , } & { \mathbf { R } _ { \mathrm { y a w } } = \mathbf { R } _ { z } ( - \phi ^ { \star } ) . } \end{array}\tag{S14}
$$

Here, W contains wall-like normals selected with $\tau _ { w } ~ = ~ 0 . 2 , ~ \phi _ { i }$ is the horizontal orientation of normal i folded modulo $\pi / 2 ,$ , and $K _ { \sigma } ^ { \pi / 2 }$ is a periodic Gaussian kernel used to aggregate their votes. The dominant angle $\phi ^ { \star }$ defines the yaw rotation $\mathbf { R } _ { \mathrm { y a w } }$

Block Construction and Local Conditioning. We rescale the aligned scene to a vertical extent of 2.9 m and cover the XY bounding box of its connected scaffold with the smallest centred grid of 4 m blocks. Adjacent blocks use a 3 m stride and share the same vertical centre. We discard blocks containing fewer than 100 reconstructed points. Each remaining crop is uniformly subsampled to at most $8 N _ { \mathrm { p c } }$ points before farthest-point sampling produces the final $\bar { N } _ { \mathrm { p c } } = 8 1 9 2$ -point condition.

To select the conditioning views, we voxelise each block crop at $G ^ { 3 }$ resolution with $G = 3 2$ , obtaining a target set $\mathcal { Q } _ { b }$ . For every candidate view v satisfying the 15% in-block coverage criterion, we voxelise its aligned in-block point map and retain the intersection $\gamma _ { b , \iota }$ with $\mathcal { Q } _ { b }$ . We then choose up to four views that maximise their joint coverage:

$$
\begin{array} { r } { \mathcal { S } _ { b } ^ { \star } = \underset { \underset { | S | = \operatorname* { m i n } ( 4 , | \mathcal { C } _ { b } | ) } { \operatorname { a r g m a x } } } { \operatorname { a r g m a x } } \left| \bigcup _ { v \in S } \mathcal { V } _ { b , v } \right| , } \end{array}\tag{S15}
$$

where $\mathcal { C } _ { b }$ denotes the candidate-view set.

## C EXPERIMENTAL DETAILS

## C.1 BASELINES

We use VGGT-Ω Wang et al. (2026) prediction as the common geometric prior for all methods. It provides camera intrinsics and extrinsics, per-view depth and confidence, and a point cloud obtained by unprojecting the predicted depths. Each baseline follows its official pipeline, with any required ground-truth geometric inputs replaced by the corresponding predictions from VGGT-Ω.

• VGGT-Ω+TSDF directly integrates the predicted RGB-D frames using the VGGT-Ω cameras through TSDF fusion Newcombe et al. (2011).

• Murre Guo et al. (2025) receives the RGB frames and sparse depth samples selected from the VGGT-Ω depth maps using confidence, spatial subsampling, and multi-view consistency. Its refined dense depths are fused with the VGGT-Ω intrinsics and aligned poses using TSDF.

• FineRecon Stier et al. (2023) receives the RGB frames, confidence-filtered VGGT-Ω depths, intrinsics, and aligned poses. The predicted depths are used for depth-guided feature back-projection and as an additional TSDF feature channel.

• MonoSDF Yu et al. (2022) uses the VGGT-Ω intrinsics and aligned poses together with the RGB frames. Following its original formulation, its monocular depth and normal supervision is predicted from the RGB images by Omnidata, rather than taken from the VGGT-Ω depth maps.

• GenRecon Schmid et al. (2026) uses the aligned VGGT-Ω point cloud to determine the scene extent and chunk layout, and uses the predicted cameras to project multi-view RGB features into each chunk’s 3D feature volume.

## C.2 EVALUATION PROTOCOL

As VGGT-Ω estimates camera poses from uncalibrated images, its reconstructed coordinate frame has no absolute scale or canonical horizontal orientation. Our scene-alignment procedure recovers gravity and dominant wall directions, but does not resolve the remaining global similarity ambiguity with respect to the ground-truth frame. For evaluation, we therefore estimate a Sim(3) transform from corresponding predicted and reference camera centres using Horn alignment and apply it to the complete predicted mesh. The same transform is used for all 2D and 3D metrics. We convert dataset-specific axis conventions before computing the metrics.

Following the evaluation protocol of GenRecon Schmid et al. (2026), for synthetic scenes, we restrict prediction triangles to the GT axis-aligned bounding box enlarged by 20%, preventing geometry far outside the scene extent from dominating the evaluation. For ScanNet++, we restrict the evaluation domain to the observed scanner surface, represented by a 1.1 cm voxel grid and dilated by 15 cm.

## C.3 METRICS

3D Metrics. We uniformly sample 200k points from each mesh surface and compute bidirectional nearest-neighbour correspondences. We report:

• Chamfer Distance (CD): the average of the mean predicted-to-reference and reference-topredicted nearest-neighbour Euclidean distances, reported in metres;

• F-score@10cm (F-score): the harmonic mean of predicted-to-GT precision and GT-to-prediction recall, where a point is considered matched if its nearest-neighbour distance is below 0.1 m;

• Normal Consistency (NC): the bidirectional mean absolute dot product between the normals of nearest-neighbour surface samples. Correspondences farther than 0.2 m contribute zero.

2D Metrics. We render the aligned predicted and reference geometry using the same held-out camera poses and intrinsics. For each scene, we evaluate 50 views sampled from the held-out test set. The synthetic test views have a resolution of 512 × 512, whereas the ScanNet++ test views have a resolution of 640 × 480. Frames used as input to VGGT-Ω are excluded from the evaluation views.

We evaluate pixel-wise depth and surface-normal errors, pooling RMSE, AbsRel, and Angular over all valid view–pixel pairs across the evaluation views. Let $\nu _ { d }$ denote the set of view–pixel pairs for which both the predicted and reference renderings have finite, positive depth. The depth metrics are

![](images/aea7f7858e274556718aca402f7c448535e5134680772fffef752d7fa7cca0df.jpg)  
Figure S1: Large-Scale Scene Generation. Reconstruction results for large-scale, multi-room scenes, with top-down views (left) and detailed interior views (right).

defined as

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { | \mathcal { V } _ { d } | } \sum _ { p \in \mathcal { V } _ { d } } ( \hat { d } _ { p } - d _ { p } ) ^ { 2 } } , \qquad \mathrm { A b s R e l } = \frac { 1 } { | \mathcal { V } _ { d } | } \sum _ { p \in \mathcal { V } _ { d } } \frac { | \hat { d } _ { p } - d _ { p } | } { d _ { p } } ,\tag{S16}
$$

where $\hat { d } _ { p }$ and $d _ { p }$ denote the predicted and reference depths, respectively. For surface normals, let $\nu _ { n }$ denote the set of view–pixel pairs for which both the predicted and reference foreground normals are valid. Assuming unit-length normals, we report the mean angular error in degrees:

$$
\mathrm { A n g u l a r } = { \frac { 1 8 0 } { \pi | \mathcal { V } _ { n } | } } \sum _ { p \in \mathcal { V } _ { n } } \operatorname { a r c c o s } { \bigl ( } { \bigl | \hat { \mathbf { n } } _ { p } ^ { \top } \mathbf { n } _ { p } \bigr | } { \bigr ) } .\tag{S17}
$$

The absolute dot product makes the metric invariant to normal sign ambiguity. Completeness measures the fraction of valid reference foreground pixels for which the predicted rendering contains valid geometry.

For perceptual evaluation, LPIPS and CLIP similarity are computed separately for depth and surface normals. LPIPS, CLIP similarity, and Completeness are computed independently for each view and then averaged across the evaluation views.

## D ADDITIONAL RESULTS

## D.1 LARGE SCENE GENERATION

We further demonstrate our method on large-scale indoor scenes comprising multiple, largely isolated rooms. As shown in Fig. S1, our method reconstructs an expansive nine-room scene with high fidelity, demonstrating its scalability to large 3D indoor environments.

## D.2 COMPARISON WITH AGENTIC BASELINE.

We compare our method with a GPT-6 Astra agent configured with high reasoning effort and adapted from BVB’s reconstruction protocol Tang et al. (2026). Given the same 32 RGB views, the agent inspects the images, writes and executes Blender scripts, and examines rendered previews to refine its reconstruction. It constructs scenes using geometric primitives and materials specified by numerical parameters, without image textures or imported assets. Our adaptation uses fixed RGB views instead of video input and does not enforce the original cost ceiling.

Figure S2 shows a densely furnished scene with complex object arrangements and mutual occlusions (top), and a multi-room interior with oblique partitions (bottom). The recorded reconstruction stages consume approximately 2.12M and 1.71M tokens, respectively, including repeated cached context, corresponding to estimated Standard API-equivalent costs of \$2.93 and \$2.26. In contrast, our method performs inference on a single GPU without external LLM calls, avoiding this additional API expense. Despite iterative refinement, the agent produces simplified object geometry and misplaced furniture. In the multi-room scene, it reconstructs an incorrect global layout. Our method more faithfully recovers furniture arrangement, architectural structure, and surface appearance.

![](images/ec5c6695696806d2d757e01bc23abee5ca9b26913e86639acce64e96cb7fb6e4.jpg)  
Figure S2: Comparison with Agentic Baseline. Given the same 32 views, we show GT, ours, and a GPT-6 Astra agent adapted from BVB Tang et al. (2026) (left to right). Our method yields more faithful reconstructions, while the agent simplifies or omits objects and distorts the scene layout.

![](images/8d601bfecc2041b0df9a99a2982b6709bee76dd7b10995388f881d44cd5e50a0.jpg)  
Figure S3: Failure case. Inaccurate reconstruction from VGGT-Ω lead to missing geometry in unobserved regions. Nevertheless, WorldRecon remains robust to substantial pose errors and still faithfully reconstructs the areas that are covered by the input views.

## D.3 FAILURE CASES AND LIMITATIONS

Figure S3 illustrates a representative failure case on a multi-room scene. The 32 sampled input views (top-left) are visually rich, but parts of the scene are occluded or observed from only a handful of viewpoints. As a result, the VGGT-Ω point-cloud scaffold (bottom-left) incorrectly estimates the poses of several views: some regions are either left unreconstructed or filtered out due to low confidence, while other views are projected to entirely wrong locations.

Under this shared scaffold, the TSDF baseline can only reconstruct a small fraction of the scene, while the generative baseline GenRecon directly reprojects image features using the erroneous poses, scattering misaligned projections across the scene and producing large fragmented artifacts as well as distorted objects. WorldRecon likewise suffers from missing geometry in unobserved regions due to the incomplete input coverage. Nevertheless, it remains robust to substantial pose errors and still faithfully reconstructs the areas that are well covered by the input views, and is able to hallucinate plausible completions for the missing point-cloud regions based on sparse image cues.