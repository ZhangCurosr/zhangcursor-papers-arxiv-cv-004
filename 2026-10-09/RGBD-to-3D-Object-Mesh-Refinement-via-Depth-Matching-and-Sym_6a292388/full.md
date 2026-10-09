# RGBD-to-3D Object Mesh Refinement via Depth Matching and Symmetry Propagation

Ahyun Seo<sup>1,2</sup> Minsu Cho<sup>1</sup>

<sup>1</sup>POSTECH <sup>2</sup>KAIST

https://ahyunSeo.github.io/MatchPropMesh

Abstract. Single-view 3D reconstructors often produce plausible meshes that disagree with the input view, especially near depth discontinuities and self-occlusions. We present a lightweight, plug-and-play RGBD-to-3D refinement that improves any RGB-to-3D reconstructor without retraining. Given a depth map, we correct the visible surface by bipartite matching to back-projected depth points, mirror these corrections onto the occluded side across a detected symmetry plane, and propagate them with a smoothness solver. Every stage is closed-form, making the method orders of magnitude faster than optimization-heavy test-time refinement. On GSO and OmniObject3D with five backbones, it yields consistent gains, also with monocular pseudo-depth, benefits more from symmetry on symmetric objects, and compares favorably with prior refinement in accuracy and runtime. It further improves an RGB-D-to-mesh reconstructor and transfers to real captures with noisy sensor depth.

Keywords: Single-view 3D reconstruction · Test-time refinement · Depth and symmetry priors

## 1 Introduction

Single-view 3D object reconstruction recovers a full 3D shape from one RGB or RGB-D image, supporting view synthesis, editing, simulation, AR/VR, and robotic interaction. The problem is ill-posed: much of the object is unobserved, and even visible surfaces are ambiguous under sparse texture, specularities, shadows, and occlusion. Reconstructors therefore balance learned shape priors against evidence from the input, and small errors on the visible surface can grow into large distortions over the full mesh.

Most prior work relies on learned priors and synthesized observations: native 3D generative models [4, 12, 20, 30, 46, 49], difusion-based lifting of 2D priors through per-instance optimization [2,29,33,41], feed-forward triplane and transformer pipelines [1, 10, 11, 39, 51], and reconstruction from a few generated novel views [24,25,42]. RGB-D methods exploit depth for stronger geometry [19,45,48], but they complete partial point clouds rather than refine RGB-to-3D meshes.

Three gaps remain. First, reconstructed surfaces often look plausible but disagree with the geometry that the input view supports, most noticeably near depth discontinuities and self-occlusions. Second, test-time refinement either relies on costly per-instance optimization [13, 32] or on weak 2D signals such as silhouettes [16], which ofer little guidance for fine surface corrections. Third, occluded regions receive no direct observation. Symmetry is a well-known structural prior [35, 54], but prior work imposes it globally or learns it end-to-end rather than using it to transfer reliable corrections into unseen parts.

We propose a lightweight, plug-and-play RGBD-to-3D refinement framework that operates on the output of any RGB-to-3D reconstructor without retraining. At inference, a single depth map supervises the geometry, together with the source-view camera pose and intrinsics, which we assume known as in prior refinement or obtain from an of-the-shelf estimator. The framework has three stages. First, bipartite matching between visible mesh vertices and back-projected depth points corrects the observed surface. Second, we estimate a dominant bilateral symmetry plane and mirror these visible corrections onto the occluded side. Third, a smoothness solver propagates the sparse corrections over the full mesh. Every stage is a single closed-form solve, which makes the refinement orders of magnitude faster than optimization-heavy test-time refinement. Experiments on GSO and OmniObject3D with five reconstruction backbones show consistent improvements, which persist with pseudo-depth from an of-the-shelf monocular estimator, larger benefits from symmetry on symmetric objects, and favorable accuracy and runtime against prior refinement methods. The refinement also improves an RGB-D-to-mesh reconstructor and transfers to real captures with noisy sensor depth.

## 2 Related Work

Single-view 3D object reconstruction. Native 3D generative models synthesize shapes in volumetric or surface spaces [4,12,20,30,46,49], and difusion-based methods lift 2D priors to 3D by per-instance optimization that matches rendered views to an image prior [2, 6, 26, 29, 33, 38, 40, 41]. Feed-forward pipelines reconstruct quickly with triplanes and transformers [1, 10, 11, 39, 43, 51], often combined with Gaussian splatting or sparse 3D backbones [22,37,52,55], and another line reconstructs from a few generated novel views [17, 21, 22, 24, 25, 42, 43, 51]. These methods favor plausibility under learned priors, and their accuracy with respect to the input view can still lag. RGB-D-to-3D methods use the input depth [19, 45, 48] but typically complete partial point clouds rather than produce full, watertight meshes for downstream use. We instead use a single depth map to refine a mesh, improving view-consistent geometry while keeping the mesh output.

Test-time refinement for 3D. Test-time refinement corrects a reconstruction at inference with cues from the given instance, reducing view-specific errors and domain gaps. MeTTA [13] optimizes geometry, appearance, and pose with generative priors and learnable virtual cameras, CRISP [32] optimizes object pose and shape from RGB-D, and 3D-TTA [5] denoises corrupted point clouds via latentspace difusion. Other work refines from limited-angle depth [15], with recurrent test-time training [3], or with triplet-based consistency [53]. REFINE [16] is a plug-and-play post-processor that optimizes a per-instance network for silhouette consistency, but mask supervision ofers little guidance for fine corrections, and the per-instance optimization adds notable cost. In contrast, our refinement attaches to any RGB-to-3D reconstructor and solves every stage in closed form, using metric depth as a hard anchor instead of iterative optimization.

Symmetry for 3D. Symmetry is a long-standing structural prior for reconstruction, completion, and disambiguation. Classical and variational approaches detect and enforce symmetries or symmetrize observations for stability [35, 54]. From a single image, prior work estimates a reflection plane to create a virtual second view or constrain depth [14], or fuses flipped predictions with perceptual objectives [34]. Symmetry also supports sketch-based modeling with locally symmetric primitives [9], and recent transformer-based detectors provide scalable cues for 3D generation [18]. Unlike methods that enforce symmetry on the predicted shape or learn it end-to-end, we use an estimated bilateral plane to mirror depth-guided displacements from the visible side onto the occluded side.

## 3 Proposed Method

## 3.1 Problem Definition

Given a single-view RGB image $\mathbf { I } \in \mathbb { R } ^ { H \times W \times 3 }$ and a mesh reconstruction model $s ,$ we obtain an initial triangular mesh $( V , F ) = S ( \mathbf { I } )$ with vertex set $V =$ $\{ \mathbf { v } _ { i } \} _ { i = 1 } ^ { N _ { v } } \subset \mathbb { R } ^ { 3 }$ , vertex index set $I = \{ 1 , \ldots , N _ { v } \}$ , and face set $F = \{ f _ { j } \} _ { j = 1 } ^ { N _ { f } }$ with $f _ { j } = ( i _ { j } ^ { 1 } , i _ { j } ^ { 2 } , i _ { j } ^ { 3 } ) \in I ^ { 3 }$ . Using a metric depth map $\mathbf { D } \in \mathbb { R } ^ { H \times W }$ of the source view, camera intrinsics $K$ , projection matrix $P ,$ and world-to-camera pose $T _ { \mathrm { c w } }$ , we estimate per-vertex displacements $\mathbf { d } _ { i } \in \mathbb { R } ^ { 3 }$ that correct the initial mesh toward the observed geometry:

$$
\mathbf { v } _ { i } ^ { \prime } = \mathbf { v } _ { i } + \mathbf { d } _ { i } , \qquad i \in I ,\tag{1}
$$

which yields the updated mesh $( V ^ { \prime } , F )$ with $V ^ { \prime } = \{ \mathbf { v } _ { i } ^ { \prime } \} _ { i \in { I } }$ . As shown in Fig. 1, we estimate these displacements in three stages: depth-guided displacement on the visible surface (Sec. 3.2), symmetry-guided transfer to the occluded side (Sec. 3.3), and smoothness propagation over the full mesh (Sec. 3.4).

## 3.2 Depth-guided Vertex Displacement

Motivation. Single-view reconstructors are trained for plausible shapes rather than pixel-accurate alignment with the source view, and the visible surface of the initial mesh often deviates from the observed depth. We therefore match visible mesh vertices to back-projected depth points, which yields reliable displacements on the observed surface.

![](images/8aed1c7b549a0a999345fa5d5c4b8185b7c09a4218208b753fec94e967cd49c2.jpg)  
Fig. 1: Overview. The depth-guided vertex update rasterizes the camera-frame mesh to find the visible vertices $I _ { \mathrm { f g } }$ and moves the matched ones to the back-projected depth points $\mathbf { p } ^ { \mathrm { c } } ( \mathbf { q } )$ . The symmetry-guided vertex update mirrors these corrections across the detected plane from the camera-facing side $\left( { { I _ { \mathrm { { L } } } } } \right)$ onto the occluded side $\left( I _ { \mathrm { R } } \right)$ . A smoothness solver then propagates the sparse displacements over the full mesh.

Camera-frame transformation. We express mesh vertices and depth points in the camera frame to compare them directly. A vertex $\mathbf { v } _ { i }$ with homogeneous lift $\tilde { \mathbf { v } } _ { i }$ maps to $\mathbf { v } _ { i } ^ { \mathrm { c } } = T _ { \mathrm { c w } } \tilde { \mathbf { v } } _ { i }$ , and an image point $\mathbf { q } = [ x , y ] ^ { \top }$ with depth $\mathbf { D } ( \mathbf { q } )$ back-projects to

$$
\begin{array} { r } { \mathbf p ^ { \mathrm { c } } ( \mathbf q ) = \left[ \frac { \mathbf D ( \mathbf q ) ( x - c _ { x } ) } { f _ { x } } , \frac { \mathbf D ( \mathbf q ) ( y - c _ { y } ) } { f _ { y } } , - \mathbf D ( \mathbf q ) \right] ^ { \top } , } \end{array}\tag{2}
$$

where $( f _ { x } , f _ { y } , c _ { x } , c _ { y } )$ are the NDC intrinsics and $( x , y )$ the NDC pixel coordinates. We adopt the ${ \mathrm { O p e n G L } } _ { / }$ /Blender camera basis (+X right, +Y up, camera looking down $- Z )$ : points in front of the camera satisfy $z ^ { \mathrm { c } } < 0$ and $\mathbf { D } ( \mathbf { q } ) = - z ^ { \mathrm { c } }$ . The rasterizer receives the same depth attribute $- \mathbf { v } _ { i , z } ^ { \mathrm { c } } ,$ and mesh and depth points thus share one camera frame.

Depth guidance for visible vertices. Let $\mathcal { \Omega } = \{ ( u , v ) \ | \ \mathbf { D } ( u , v ) > 0 \}$ be the foreground pixels of the input depth map. We rasterize the mesh with a triangle rasterizer R into a rendered depth map $\hat { \mathbf { D } } \in \mathbb { R } ^ { H \times W }$ and a face map $\mathbf { F } \in \breve { \mathbb { R } } ^ { H \times W \times 3 }$

$$
( \hat { \mathbf { D } } , \mathbf { F } ) = \mathcal { R } ( \{ \mathbf { v } _ { i } ^ { \mathrm { p } } \} , F , \{ - \mathbf { v } _ { i , z } ^ { \mathrm { c } } \} ; H , W ) ,\tag{3}
$$

where $\mathbf { v } _ { i } ^ { \mathrm { p } } = P \mathbf { v } _ { i } ^ { \mathrm { c } }$ is the clip-space projection of $\mathbf v _ { i } ^ { \mathrm { c } }$ and F stores, at each pixel, the three vertex indices of the rendered triangle. The faces rendered at foreground pixels and their vertices form the visible sets

$$
F _ { \mathrm { f g } } = \{ \mathrm { s e t } ( \mathbf { F } ( u , v ) ) \mid \hat { \mathbf { D } } ( u , v ) > 0 \} , \qquad I _ { \mathrm { f g } } = \bigcup F _ { \mathrm { f g } } \subseteq I .\tag{4}
$$

We compute a bipartite assignment σ between $I _ { \mathrm { f g } }$ and Ω with a GPU auction solver (Sec. 4) that minimizes the cost $\left\| \mathbf { v } _ { i } ^ { \mathrm { c } } - \mathbf { p } ^ { \mathrm { c } } ( \mathbf { q } ) \right\| _ { 2 }$ . The assignment is oneto-one but partial: visible vertices typically outnumber the depth points, and we discard matches above a cost threshold, which leaves a matched subset $I _ { \mathrm { m } } \subseteq I _ { \mathrm { f g } }$ with assigned pixels $\mathbf { q } _ { i } ^ { * } = \sigma ( i )$ and depth-guided displacements $\mathcal { D } ^ { \mathrm { c } } = \{ \mathbf { d } _ { i } ^ { \mathrm { c } } \}$ ,

$$
\mathbf { d } _ { i } ^ { \mathrm { c } } = \left\{ \mathbf { p } ^ { \mathrm { c } } ( \mathbf { q } _ { i } ^ { \ast } ) - \mathbf { v } _ { i } ^ { \mathrm { c } } , \quad i \in I _ { \mathrm { m } } , \right.\tag{5}
$$

## 3.3 Symmetry-guided Vertex Displacement

Motivation. Single-view reconstructors trained on large shape collections tend to reproduce the bilateral symmetry of symmetric objects. Depth guidance corrects only the visible surface, which breaks this symmetry and leaves the occluded region unconstrained. As the corrected visible surface is a reliable proxy for its mirror image, we transfer the depth-driven corrections across an estimated symmetry plane into the occluded region.

Bilateral symmetry plane detection. From the initial vertices, we select a dominant bilateral symmetry plane among $N _ { p }$ candidates with azimuth $\theta _ { k }$ and polar angle $\alpha _ { k }$ , whose normal (y-up) and ofset are

$$
\begin{array} { r } { \mathbf { n } _ { k } = \big ( \sin \alpha _ { k } \cos \theta _ { k } , \cos \alpha _ { k } , \sin \alpha _ { k } \sin \theta _ { k } \big ) ^ { \top } , d _ { k } = - \mathbf { n } _ { k } ^ { \top } \bar { \mathbf { v } } ^ { \mathrm { c } } , } \end{array}\tag{6}
$$

which place the plane through the centroid $\bar { \mathbf { v } } ^ { \mathrm { c } }$ . Each vertex $\mathbf { v } _ { i } ^ { \mathrm { c } }$ then has a signed distance to the plane and a projection onto it,

$$
s _ { i } ^ { ( k ) } = \mathbf { n } _ { k } ^ { \top } \mathbf { v } _ { i } ^ { \mathrm { c } } + d _ { k } , \qquad \mathbf { p } _ { i } ^ { ( k ) } = \mathbf { v } _ { i } ^ { \mathrm { c } } - s _ { i } ^ { ( k ) } \mathbf { n } _ { k } .\tag{7}
$$

Any of-the-shelf detector can provide the plane; we score candidates with a simple rasterized criterion. Instead of reflecting points and accumulating nearestneighbor distances, we rasterize the signed distances at the in-plane coordinates $( u _ { i } ^ { ( k ) } , v _ { i } ^ { ( k ) } )$ of $\mathbf { p } _ { i } ^ { ( k ) }$ into two maps $\mathbf { M } _ { \pm } ^ { ( k ) } \in \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } }$ :

$$
\mathbf { M } _ { + } ^ { ( k ) } ( u _ { i } ^ { ( k ) } , v _ { i } ^ { ( k ) } ) = s _ { i } ^ { ( k ) } \ \mathrm { ~ i f ~ } s _ { i } ^ { ( k ) } > 0 , \qquad \mathbf { M } _ { - } ^ { ( k ) } ( u _ { i } ^ { ( k ) } , v _ { i } ^ { ( k ) } ) = s _ { i } ^ { ( k ) } \ \mathrm { ~ i f ~ } s _ { i } ^ { ( k ) } < 0 .\tag{8}
$$

After smoothing both maps with a $5 \times 5$ mean filter into $\widetilde { \mathbf { M } } _ { \pm } ^ { ( k ) }$ , we score each plane by the average $\ell _ { 1 }$ discrepancy between the two channels over the cells $\varOmega _ { k }$ where both are populated, and select

$$
k ^ { \star } = \arg \operatorname* { m i n } _ { k } \ \frac { 1 } { | \varOmega _ { k } | } \sum _ { ( u , v ) \in \varOmega _ { k } } \big | \widetilde { \mathbf { M } } _ { + } ^ { ( k ) } ( u , v ) + \widetilde { \mathbf { M } } _ { - } ^ { ( k ) } ( u , v ) \big | .\tag{9}
$$

As the negative channel stores signed values, a symmetric cell gives ${ \widetilde { \mathbf { M } } } _ { + } ^ { ( k ) }$ + $\widetilde { \mathbf { M } } _ { - } ^ { ( k ) }$ ≈ 0; Fig. 4 in the supplementary material shows that the discrepancy is minimized at the correct plane (150<sup>◦</sup>).

Displacement transfer by bipartite matching. Given the detected plane, we take the depth-matched vertices on its positive side and all vertices on its negative side,

$$
\begin{array} { r } { I _ { \mathrm { L } } = \big \{ i \in I _ { \mathrm { m } } \mid s _ { i } ^ { ( k ^ { \star } ) } > 0 \big \} , \qquad I _ { \mathrm { R } } = \big \{ j \in I \mid s _ { j } ^ { ( k ^ { \star } ) } < 0 \big \} , } \end{array}\tag{10}
$$

and compute a bipartite assignment $\pi : I _ { \mathrm { L } }  I _ { \mathrm { R } }$ with the same auction solver on in-plane coordinates and unsigned distances,

$$
C _ { i j } = \big \| \phi ( i ) - \phi ( j ) \big \| _ { 2 } , \qquad \phi ( i ) = \big [ { u _ { i } ^ { ( k ^ { \star } ) } , v _ { i } ^ { ( k ^ { \star } ) } , | s _ { i } ^ { ( k ^ { \star } ) } | } \big ] ^ { \top } .\tag{11}
$$

For each matched pair, we move $\pi ( i )$ to the reflection of the corrected vertex $\mathbf { v } _ { i } ^ { \prime \mathrm { c } } = \mathbf { v } _ { i } ^ { \mathrm { c } } + \mathbf { d } _ { i } ^ { \mathrm { c } }$ and set its displacement accordingly:

$$
\begin{array} { r } { \mathbf { v } _ { \pi ( i ) } ^ { \prime \mathrm { c } } = \mathbf { v } _ { i } ^ { \prime \mathrm { c } } - 2 \big ( \mathbf { n } _ { k ^ { \star } } ^ { \top } \mathbf { v } _ { i } ^ { \prime \mathrm { c } } + d _ { k ^ { \star } } \big ) \mathbf { n } _ { k ^ { \star } } , } \end{array}\tag{12}
$$

$$
\mathbf { d } _ { \pi ( i ) } ^ { \mathrm { c } } = \mathbf { v } _ { \pi ( i ) } ^ { \mathrm { \prime c } } - \mathbf { v } _ { \pi ( i ) } ^ { \mathrm { c } } .\tag{13}
$$

Unlike reflecting ${ \bf d } _ { i } ^ { \mathrm { c } }$ alone, this also absorbs the matching residual. We discard matches above a cost threshold and those whose target already carries a depthdriven displacement $( \pi ( i ) \in I _ { \mathrm { m } } )$ ; the reflection therefore never overwrites a depth-driven displacement, and unmatched vertices keep a zero displacement. The camera-facing positive half $\left( s ^ { ( k ^ { \star } ) } > 0 \right)$ serves as the source, and the propagation below fills the occluded vertices on that side.

## 3.4 Mesh Refinement with Vertex Displacement

Motivation. The previous stages provide reliable displacements only on a sparse set of vertices. Applying them directly introduces discontinuities, and leaving the remaining vertices unchanged yields inconsistent geometry. We therefore propagate these displacements as Dirichlet handles with Laplacian smoothing.

Displacement propagation via Dirichlet handles. The handle set $\mathcal { H } =$ $I _ { \mathrm { m } } \cup \{ \pi ( i ) : i \in I _ { \mathrm { L } } \}$ carries the depth-guided displacements of the matched visible vertices and their reflections on the mirrored occluded vertices, written $\{ \mathbf { d } _ { i } ^ { \mathrm { c } } \} _ { i \in \mathcal { H } }$ . We solve for displacements ${ \mathbf { u } } _ { i } \in \mathbb { R } ^ { 3 }$ of all vertices such that the refined mesh $V ^ { \prime } = \{ \mathbf { v } _ { i } ^ { \mathrm { { c } } } + \mathbf { u } _ { i } \} _ { i \in I }$ is smooth and satisfies the handle constraints. With the cotangent Laplacian $L$ of $( V , F )$ , we minimize

$$
\begin{array} { r } { E ( \mathbf { u } ) = \frac { 1 } { 2 } \displaystyle \sum _ { c \in \{ x , y , z \} } \mathbf { u } _ { \cdot , c } ^ { \top } L \mathbf { u } _ { \cdot , c } \quad \mathrm { s . t . } \quad \mathbf { u } _ { i } = \mathbf { d } _ { i } ^ { \mathrm { c } } , ~ i \in \mathcal { H } . } \end{array}\tag{14}
$$

Splitting the vertices into free indices $\mathcal { F } = I \backslash \mathcal { H }$ and handles H and partitioning L accordingly gives the reduced linear system $\begin{array} { r } { L _ { f f } \mathbf { u } _ { f , c } = - L _ { f h } \mathbf { d } _ { h , c } ^ { \mathrm { c } } , } \end{array}$ , which we solve once for all free vertices. For robustness, we clamp the cotangent weights to give degenerate or near-degenerate triangles zero weight in Eq. (14), stabilize $L _ { f f }$ with a small Tikhonov term $\tau I ,$ , and softly anchor free boundary vertices with a quadratic penalty $\gamma \Vert \mathbf { u } _ { i } \Vert ^ { 2 }$ . These terms improve conditioning without visibly changing the result. We update the vertices as $\mathbf { v } _ { i } ^ { \mathrm { c } \prime } = \mathbf { v } _ { i } ^ { \mathrm { c } } + \mathbf { u } _ { i }$ and map them back to the world frame, which realizes the displacements $\mathbf { d } _ { i }$ of Sec. 3.1. Propagating the depth handles $I _ { \mathrm { m } }$ alone yields our visible-only variant, to which we apply a few Taubin smoothing iterations to suppress residual high-frequency artifacts; including the reflected handles yields the full result.

## 4 Experiments

## 4.1 Datasets and Evaluation

Datasets. We evaluate on two datasets of real-world scanned objects. Google Scanned Objects (GSO) [8] contains 1,030 high-quality textured scans of common household items, all of which we use. OmniObject3D [47] covers a large vocabulary of about 6,000 scanned objects in 190 daily categories; following MeshFormer [22], we take up to five shapes per category (1,038 shapes).

Metrics. Following [43,51], we report two geometric metrics, Chamfer Distance (CD; the mean nearest-neighbor distance between the two surfaces, averaged over both directions) and F-score (FS; the harmonic mean of precision and recall), and image metrics on rendered views (SSIM, PSNR, LPIPS, and CLIP similarity).

Protocol. For each shape, we render RGB and depth with BlenderProc [7] from 24 viewpoints (12 azimuths × two elevations $\{ 1 5 ^ { \circ } , 3 0 ^ { \circ } \} )$ and uniformly sample one of them as the input view, fixed across all methods. Unless stated otherwise, we refine with the GT camera pose and the rendered GT depth of this view; pseudo-depth denotes the UniDepth-V2 [28] monocular prediction, either rescaled to the GT depth range or afinely fit to the depth rendered from the initial mesh (no GT), and each experiment states its pose and depth sources. We align each mesh candidate to the GT independently, including the initial mesh and the refinement baselines: we transform it into the GT frame, scale its longest side to 0.8, and resolve the global orientation with a coarse-to-fine rotation search over SO(3). Our refinement starts from the aligned initial mesh, and we neither re-align nor rescale its output. Before refinement, we decimate meshes above 30k vertices. For the geometric metrics, we sample 16k points from the prediction and the GT and score FS at thresholds 0.02, 0.05, and 0.1; for the image metrics, we compare 48 rendered views with the GT renders.

## 4.2 Implementation Details

Refinement. We rasterize the mesh with nvdiffrast at the BlenderProc camera $( \mathrm { F o V } \approx 4 9 ^ { \circ } )$ at $2 5 6 \times 2 5 6$ , from 512 × 512 input images, and back-project the input depth at 128 × 128. For the depth-guided update, we discard matches with an assignment cost above 0.1. For the symmetry-guided update, we sweep the plane azimuth over $1 ^ { \circ } - 1 8 0 ^ { \circ }$ and the elevation over $\{ 0 ^ { \circ } , \ldots , 3 0 ^ { \circ } \}$ , and discard left/right matches with a cost above 0.2. For propagation, we clamp the cotangent Laplacian weights to [0, 50], stabilize the free-free system with small Tikhonov and boundary-anchor terms, and apply 5 Taubin smoothing iterations to the depth-only mesh. We use these values in all experiments without perdataset tuning, as the thresholds are relative to the normalized mesh (e.g., 0.1 is about 6% of the object extent); Table 8 reports their sensitivity.

Table 1: Mesh refinement results on GSO and OmniObject3D.
<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="2">Chamfer Distance ↓</td><td colspan="3">F-score (0.05) ↑</td><td colspan="3">F-score (0.1) ↑</td></tr><tr><td>Initial</td><td>Refined (diff)</td><td>Initial</td><td></td><td>Refined (diff)</td><td>Initial</td><td>Refined (diff)</td><td></td></tr><tr><td rowspan="5">GSO</td><td>LGM [37]</td><td>0.0438</td><td>0.0306</td><td>(-0.0132)</td><td>0.6715</td><td>0.7983</td><td>(+0.1268)</td><td>0.8931</td><td>0.9320 (+0.0389)</td></tr><tr><td>CRM [42]</td><td>0.0365</td><td>0.0248</td><td>(-0.0117)</td><td>0.7473</td><td>0.8585 (+0.1112)</td><td>0.9346</td><td>0.9627</td><td>(+0.0281)</td></tr><tr><td>SF3D [1]</td><td>0.0352</td><td>0.0236</td><td>(-0.0116)</td><td>0.7658</td><td>0.8693</td><td>(+0.1035)</td><td>0.9369</td><td>0.9620 (+0.0251)</td></tr><tr><td>SPAR3D [11]</td><td>0.0356</td><td>0.0248</td><td>(-0.0108)</td><td>0.7671</td><td>0.8597 (+0.0926)</td><td></td><td>0.9321 0.9554</td><td>(+0.0233)</td></tr><tr><td>InstantMesh [51]</td><td>0.0283</td><td>0.0205</td><td>(-0.0078)</td><td>0.8358</td><td>0.8950</td><td>(+0.0592)</td><td>0.9582</td><td>0.9721 (+0.0139)</td></tr><tr><td rowspan="5">n</td><td>LGM [37]</td><td>0.0397</td><td>0.0291</td><td>(-0.0106)</td><td>0.7185</td><td>0.8119 (+0.0934)</td><td>0.9031</td><td>0.9312</td><td>(+0.0281)</td></tr><tr><td>CRM [42]</td><td>0.0334</td><td>0.0233</td><td>(-0.0101)</td><td>0.7833</td><td>0.8697</td><td>(+0.0864)</td><td>0.9411</td><td>0.9666 (+0.0255)</td></tr><tr><td>SF3D [1]</td><td>0.0311</td><td>0.0210</td><td>(-0.0101)</td><td>0.8075</td><td>0.8910 (+0.0835)</td><td></td><td>0.9524 0.9701</td><td>(+0.0177)</td></tr><tr><td>SPAR3D [11]</td><td>0.0331</td><td>0.0238</td><td>(-0.0093)</td><td>0.7945</td><td>0.8673</td><td>(+0.0728)</td><td>0.9368 0.9552</td><td>(+0.0184)</td></tr><tr><td>InstantMesh [51]</td><td>0.0320</td><td>0.0229</td><td>(-0.0091)</td><td>0.8011</td><td>0.8720</td><td>(+0.0709)</td><td>0.9383 0.9547</td><td>(+0.0164)</td></tr></table>

Table 2: 2D visual metrics of initial and refined meshes.
<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="2">SSIM ↑</td><td colspan="2">PSNR ↑</td><td colspan="3"></td></tr><tr><td>Initial</td><td>Refined (diff)</td><td>Initial</td><td>Refined (diff)</td><td></td><td>Initial</td><td>Refined (diff)</td></tr><tr><td rowspan="5">GSO</td><td>LGM [37]</td><td>0.8148</td><td>0.8269 (+0.0121)</td><td>14.7640</td><td>15.5800 (+0.8160)</td><td></td><td>0.2892</td><td>0.2658 (-0.0234)</td></tr><tr><td>CRM [42]</td><td>0.8292</td><td>0.8418 (+0.0126)</td><td>15.9370</td><td>16.7550</td><td>(+0.8180)</td><td>0.2504</td><td>0.2302 (-0.0202)</td></tr><tr><td>SF3D [1]</td><td>0.8348</td><td>0.8422 (+0.0074)</td><td>15.6650</td><td>16.1650</td><td>(+0.5000)</td><td>0.2321</td><td>0.2219 (-0.0102)</td></tr><tr><td>SPAR3D [11]</td><td>0.8244</td><td>0.8314 (+0.0070)</td><td>15.3860</td><td>15.9430</td><td>(+0.5570)</td><td>0.2360</td><td>0.2255 (-0.0105)</td></tr><tr><td>InstantMesh [51]</td><td>0.8413</td><td>0.8495 (+0.0082)</td><td>17.4310</td><td>18.0690</td><td>(+0.6380)</td><td>0.2078</td><td>0.1967 (-0.0111)</td></tr><tr><td rowspan="5">On</td><td>LGM [37]</td><td>0.8174</td><td>0.8305 (+0.0131)</td><td>16.1070</td><td>16.9260</td><td>(+0.8190)</td><td>0.2725</td><td>0.2521 (-0.0204)</td></tr><tr><td>CRM [42]</td><td>0.8326</td><td>0.8466 (+0.0140)</td><td>17.3270</td><td>18.2820</td><td>(+0.9550)</td><td>0.2374</td><td>0.2143 (-0.0231)</td></tr><tr><td>SF3D [1]</td><td>0.8342</td><td>0.8435 (+0.0093)</td><td>15.3700</td><td>15.9700</td><td>(+0.6000)</td><td>0.2238</td><td>0.2085 (-0.0153)</td></tr><tr><td>SPAR3D [11]</td><td>0.8287</td><td>0.8348 (+0.0061)</td><td>15.3730</td><td>15.7710</td><td>(+0.3980)</td><td>0.2252</td><td>0.2135 (-0.0117)</td></tr><tr><td>InstantMesh [51]</td><td>0.8335</td><td>0.8436 (+0.0101)</td><td>18.1100</td><td></td><td>18.8680 (+0.7580)</td><td>0.2213</td><td>0.2050 (-0.0163)</td></tr></table>

Matching. Both stages use a GPU-parallel auction solver (Bertsekas’ algorithm with ε-scaling), which costs $\mathcal { O } ( N M \log ( C / \varepsilon ) )$ on the $N \times M$ cost matrix $( N \leq$ 30k foreground vertices, $M \leq$ 16k depth pixels). In each round, all vertices bid in parallel in a single fused kernel pass, with no per-row augmenting-path search, far below the $\mathcal { O } ( N ^ { 3 } )$ of the Hungarian algorithm. Empirically, a log-log fit of runtime against NM gives an exponent of 0.889, and neither N nor M grows with tessellation or dataset size. On GSO, refinement takes 7.5 s per instance on average, of which matching accounts for 34.5% and cot-Laplacian propagation for 61.9%; propagation, not matching, is the bottleneck.

Table 3: Comparison with single-view RGBD-to-3D methods.
<table><tr><td rowspan="2">Model</td><td colspan="4">GSO</td><td colspan="4">OmniObject3D</td></tr><tr><td>CD</td><td>FS(0.02)</td><td>FS(0.05)</td><td>FS(0.1)</td><td>CD</td><td>FS(0.02)</td><td>FS(0.05)</td><td>FS(0.1)</td></tr><tr><td>Input point cloud</td><td>0.0396</td><td>0.5365</td><td>0.7350</td><td>0.8621</td><td>0.0480</td><td>0.4298</td><td>0.6858</td><td>0.8495</td></tr><tr><td>NU-MCC [19]</td><td>0.0386</td><td>0.5023</td><td>0.7415</td><td>0.8925</td><td>0.0429</td><td>0.4567</td><td>0.7243</td><td>0.8830</td></tr><tr><td>MCC [45]</td><td>0.0342</td><td>0.4852</td><td>0.7694</td><td>0.9311</td><td>0.0425</td><td>0.4124</td><td>0.7018</td><td>0.8935</td></tr><tr><td>Affostruction [27]</td><td>0.0243</td><td>0.6698</td><td>0.8713</td><td>0.9519</td><td>0.0251</td><td>0.6541</td><td>0.8681</td><td>0.9485</td></tr><tr><td>Affostruction+ours (depth only)</td><td>0.0207</td><td>0.7232</td><td>0.9036</td><td>0.9626</td><td>0.0218</td><td>0.7003</td><td>0.8923</td><td>0.9573</td></tr><tr><td>Affostruction+ours</td><td>0.0212</td><td>0.7059</td><td>0.9003</td><td>0.9628</td><td>0.0222</td><td>0.6900</td><td>0.8882</td><td>0.9576</td></tr></table>

Table 4: Comparison with REFINE [16] on its 6-instance demo set.
<table><tr><td rowspan="2">Method (CD ×103 ↓ / Time (s) ↓)</td><td colspan="2">REFINE (1.5k) InstMesh (1.5k) InstMesh (10k)</td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>CD</td><td>Time</td><td>CD</td><td>Time</td><td>CD</td><td>Time</td></tr><tr><td>Initial mesh</td><td>21.58</td><td></td><td>8.31</td><td></td><td>8.30</td><td></td></tr><tr><td>REFINE [16]</td><td>8.38</td><td>32.0</td><td>9.18</td><td>32.9</td><td>25.21</td><td>103.2</td></tr><tr><td>Ours (w/ pred. depth scaled to pred. mesh) 23.09</td><td></td><td>1.22</td><td>8.48</td><td>1.45</td><td>8.18</td><td>7.04</td></tr></table>

Table 5: Comparison with MeTTA [13] on 9 Pix3D [36] instances.
<table><tr><td>Method</td><td>CD (×103) ↓</td><td>PSNR ↑</td><td>LPIPS ↓</td><td>CLIP↑</td><td>Time (s) ↓</td></tr><tr><td>Initial mesh</td><td>18.2</td><td>13.64</td><td>0.299</td><td>0.736</td><td></td></tr><tr><td>MeTTA [13]</td><td>48.3</td><td>14.03</td><td>0.278</td><td>0.650</td><td>~1.8k</td></tr><tr><td>Ours (w/ GT depth)</td><td>16.9</td><td>13.49</td><td>0.295</td><td>0.748</td><td>1.8</td></tr><tr><td>Ours (w/ pred. depth scaled to pred. mesh)</td><td>25.3</td><td>13.41</td><td>0.303</td><td>0.649</td><td>2.6</td></tr></table>

## 4.3 Model-agnostic Single-view Mesh Refinement

We apply our refinement to five reconstructors with diferent representations: LGM [37] decodes multi-view Gaussian features with an asymmetric U-Net, CRM [42] extracts a FlexiCubes mesh from a high-resolution triplane built from orthographic views, InstantMesh [51] pairs a multi-view difusion prior with a sparse-view LRM, SF3D [1] regresses a textured mesh in a single feed-forward pass, and SPAR3D [11] samples a point set by point difusion before imageconditioned meshing. It lowers CD and raises FS at both thresholds for every backbone on both datasets (Table 1), with the largest gains for weaker initial meshes and at the stricter 0.05 threshold, which indicates that it corrects finegrained misalignment rather than only coarse overlap. The rendered images also improve in SSIM, PSNR, and LPIPS everywhere (Table 2).

## 4.4 Comparison with Single-view RGBD Methods

In Sec. 4.3, our refinement uses the GT depth, which the RGB-to-3D backbones never see, and the gains may merely reflect this extra input. We therefore compare with methods that use depth from the start, all given the same RGB-D observation with the GT depth and camera pose (Table 3), and test whether our refinement still helps a reconstructor that already consumes depth. The RGBD-to-3D completion methods MCC [45] and NU-MCC [19] complete the back-projected partial point cloud into a point set, and the Input point cloud is the raw back-projection. We evaluate these point sets in the input camera frame, as a global-rotation search on point sets confounds alignment with geometric error. Afostruction [27] instead reconstructs a full mesh from the same input, and we report its meshes before and after our refinement (+ours). The learned completion methods improve only modestly over the raw input point cloud. Afostruction already outperforms them, and our depth-guided refinement improves it further on both datasets. The symmetry stage brings no additional gain on this backbone, suggesting that a depth-conditioned generator leaves little for mirroring to correct. The refined RGB-to-3D backbones in Table 1 also surpass the completion methods.

## 4.5 Comparison with Test-time Refinement Methods

Comparison with REFINE. REFINE [16] refines a given mesh at test time by optimizing a small per-instance network under a rendered-silhouette loss and a fixed symmetry prior, without depth. Lacking a shared benchmark, we compare on its 6-instance demo set (Table 4), starting all methods from the same three initial meshes, REFINE’s own template and InstantMesh at 1.5k and 10k vertices, with the camera released with REFINE. Ours uses pseudo-depth afinely fit to the depth rendered from the initial mesh, and we compute CD with RE-FINE’s evaluator. REFINE improves its own coarse template, but its mask-only supervision axis-stretches both InstantMesh meshes and drops them below the initial mesh. Our depth matching improves the 10k-vertex mesh, whereas on the two 1.5k-vertex meshes our pseudo-depth refinement falls below the initial mesh, indicating that it needs a suficient vertex count. Being closed-form in every stage, our method is also an order of magnitude faster.

Comparison with MeTTA. MeTTA [13] refines a given mesh at test time by distilling a 2D difusion prior over ∼1.5k iterations, jointly optimizing geometry, texture, and lighting without depth. We compare on its 9 released Pix3D [36] instances (Table 5), starting all methods from the mesh shipped with MeTTA and using the Pix3D camera annotation. Ours uses either the GT depth rendered from the CAD model or pseudo-depth afinely fit to the depth rendered from the initial mesh and warped into the GT silhouette, and we compute the 2D metrics on geometry-only renders at the input view. Optimizing appearance rather than observed geometry lets MeTTA drift far from the initial mesh, whereas ours improves it with GT depth and stays much closer with pseudo-depth, at a runtime orders of magnitude lower. Note that the Pix3D CAD GT difers from the photographed object, which penalizes refinement toward the observed depth. MeTTA scores better on PSNR and LPIPS at the input view, which it directly optimizes; against renders of the GT mesh at novel views, which Table 16 evaluates, it falls below the initial mesh at every view.

Table 6: Ablation of depth and symmetry guidance (CD).
<table><tr><td rowspan="2">Model</td><td colspan="3">GSO</td><td colspan="3">OmniObject3D</td></tr><tr><td>LGM CRM SF3D</td><td>SPAR3D</td><td>InstMesh</td><td>LGM CRM SF3D</td><td>SPAR3D</td><td>InstMesh</td></tr><tr><td>Initial mesh (RGB→3D)</td><td>0.0438 0.0365 0.0352</td><td>0.0356</td><td>0.0283</td><td>0.0397 0.0334 0.0311</td><td>0.0331</td><td>0.0320</td></tr><tr><td>w/ /pseudo-depth (scaled to GT) 0.0367 0.0315 0.0313</td><td></td><td>0.0320</td><td>0.0272</td><td>0.0344 0.0296 0.0286</td><td>0.0309</td><td>0.0296</td></tr><tr><td>w/ depth</td><td>0.0323 0.0253 0.0244</td><td>0.0253</td><td>0.0212</td><td>0.0302 0.0243 0.0217</td><td>0.0241</td><td>0.0236</td></tr><tr><td>w/ depth, symmetry</td><td>0.0306 0.02480.0236</td><td>0.0248</td><td>0.0205</td><td>0.0291 0.0233 0.0210</td><td>0.0238</td><td>0.0229</td></tr></table>

Table 7: Refinement on symmetric and asymmetric objects (CD).
<table><tr><td rowspan="2">Model</td><td colspan="5">GSO Symmetric</td><td colspan="5">OmniObject3D Symmetric</td></tr><tr><td>LGM</td><td>CRM</td><td>InstMesh</td><td>SF3D</td><td>SPAR3D</td><td>LGM</td><td>CRM</td><td>InstMesh</td><td>SF3D</td><td>SPAR3D</td></tr><tr><td>Initial mesh (RGB→3D)</td><td>0.04190.0363</td><td></td><td>0.0247</td><td>0.0354</td><td>0.0363</td><td>0.03420.0262</td><td></td><td>0.0297</td><td>0.0235</td><td>0.0257</td></tr><tr><td>+ Depth</td><td>0.03060.0265</td><td></td><td>0.0185</td><td>0.0244</td><td>0.0262</td><td>0.0263</td><td>0.0201</td><td>0.0232</td><td>0.0172</td><td>0.0184</td></tr><tr><td>+ Depth, Symmetry</td><td>0.02900.0251</td><td></td><td>0.0174</td><td>0.0224</td><td>0.0247</td><td>0.0252 0.0191</td><td></td><td>0.0217</td><td>0.0168</td><td>0.0177</td></tr><tr><td rowspan="2">Model</td><td colspan="5">GSO Asymmetric</td><td colspan="5">OmniObject3D Asymmetric</td></tr><tr><td>LGM</td><td>CRM</td><td>InstMesh</td><td>SF3D</td><td>SPAR3D</td><td>LGM</td><td>CRM</td><td>InstMesh</td><td>SF3D</td><td>SPAR3D</td></tr><tr><td>Initial mesh (RGB→3D)</td><td>0.0462</td><td>0.0369</td><td>0.0327</td><td>0.0350</td><td>0.0346</td><td>0.0399</td><td>0.0337</td><td>0.0321</td><td>0.0314</td><td>0.0333</td></tr><tr><td>+ Depth</td><td></td><td>0.0343 0.0238</td><td>0.0246</td><td>0.0244</td><td>0.0242</td><td>0.03040.0245</td><td></td><td>0.0237</td><td>0.0218</td><td>0.0243</td></tr><tr><td>+ Depth, Symmetry</td><td>0.03260.0245</td><td></td><td>0.0245</td><td>0.0251</td><td>0.0250</td><td>0.0292 0.0235</td><td></td><td>0.0230</td><td>0.0211</td><td>0.0241</td></tr></table>

## 4.6 Ablation Study

Depth guidance. Table 6 adds the components one at a time on all five backbones. Depth guidance alone, which anchors the visible vertices to the depth map, gives the largest reduction in CD on both datasets; with pseudo-depth rescaled to the GT depth range, the gain is smaller but consistent.

Symmetry guidance. Adding the symmetry stage, which mirrors the depthguided corrections into occluded regions, lowers CD further for every backbone (Table 6). To see where symmetry helps, Table 7 splits the objects into symmetric and asymmetric sets by Reflect3D [18] labels: the symmetry stage gives its largest gains on symmetric objects and can slightly hurt asymmetric ones, whose mirrored corrections do not match the true geometry.

Propagation and matching thresholds. Table 8 examines the propagation and the matching thresholds on InstantMesh/GSO. To isolate the propagation, we compare three outputs: handles only, which moves the matched vertices and leaves the rest unchanged; + propagation, which spreads these displacements with the smoothness solver; and + symmetry, which adds the reflected handles. At our threshold of 0.1, moving only the matched vertices already recovers most of the gain over the initial mesh (Table 1), the propagation further lowers CD and raises the strict FS(0.02), and symmetry lowers CD further. To test the sensitivity to the thresholds, we vary the depth-matching threshold from 0.02 to 0.5 and the symmetry-matching threshold from 0.1 to 0.4; #matched reports the mean number of depth-matched vertices out of about 10.4k visible ones. The result is stable between 0.1 and 0.2. A tighter threshold matches fewer vertices and weakens the correction, whereas a much looser one admits wrong correspondences and falls below the initial mesh. The symmetry-matching threshold barely changes the result.

Table 8: Propagation ablation and matching-threshold sweep (InstantMesh, GSO).
<table><tr><td rowspan="2">Threshold</td><td rowspan="2">#matched</td><td colspan="3">handles only</td><td colspan="3">+ propagation</td><td colspan="3">+ symmetry</td></tr><tr><td>CD</td><td>FS(0.02)</td><td>FS(0.05)</td><td>CD</td><td>FS(0.02)</td><td>FS(0.05)</td><td>CD</td><td>FS(0.02)</td><td>FS(0.05)</td></tr><tr><td colspan="10">Depth-matching threshold</td></tr><tr><td>0.02</td><td>1583</td><td>0.0269</td><td>0.585</td><td>0.843</td><td>0.0262</td><td>0.602</td><td>0.843</td><td>0.0258</td><td>0.609</td><td>0.847</td></tr><tr><td>0.05</td><td>2593</td><td>0.0241</td><td>0.652</td><td>0.866</td><td>0.0233</td><td>0.666</td><td>0.863</td><td>0.0227</td><td>0.674</td><td>0.871</td></tr><tr><td>0.1 (ours)</td><td>3093</td><td>0.0221</td><td>0.679</td><td>0.890</td><td>0.0212</td><td>0.701</td><td>0.886</td><td>0.0205</td><td>0.707</td><td>0.895</td></tr><tr><td>0.2</td><td>3362</td><td>0.0215</td><td>0.683</td><td>0.894</td><td>0.0210</td><td>0.704</td><td>0.890</td><td>0.0203</td><td>0.708</td><td>0.898</td></tr><tr><td>0.5</td><td>3499</td><td>0.0384</td><td>0.606</td><td>0.810</td><td>0.0289</td><td>0.658</td><td>0.844</td><td>0.0310</td><td>0.641</td><td>0.832</td></tr><tr><td colspan="10">Symmetry-matching threshold (depth threshold 0.1)</td></tr><tr><td>0.1</td><td>3093</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0205</td><td>0.707</td><td>0.895</td></tr><tr><td>0.2 (ours)</td><td>3093</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0205</td><td>0.707</td><td>0.895</td></tr><tr><td>0.4</td><td>3093</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0206</td><td>0.707</td><td>0.895</td></tr></table>

Table 9: Robustness to depth source and camera pose (InstantMesh, GSO).
<table><tr><td>Method</td><td>CD ↓</td><td>FS(0.02) ↑</td><td>FS(0.05) ↑</td></tr><tr><td>Initial mesh</td><td>0.0283</td><td>0.5581</td><td>0.8358</td></tr><tr><td colspan="4">Depth source (GT pose, depth-only)</td></tr><tr><td>w/ GT depth</td><td>0.0212</td><td>0.7008</td><td>0.8856</td></tr><tr><td>w/ pseudo-depth (scaled to GT)</td><td>0.0272</td><td>0.5597</td><td>0.8509</td></tr><tr><td>w/ pseudo-depth (no GT, scaled to pred. mesh)</td><td>0.0275</td><td>0.5604</td><td>0.8494</td></tr><tr><td colspan="4">Camera pose (GT depth + symmetry)</td></tr><tr><td>w/ GT pose</td><td>0.0205</td><td>0.7070</td><td>0.8950</td></tr><tr><td>w/ est. pose (FoundationPose [44])</td><td>0.0262</td><td>0.6006</td><td>0.8557</td></tr><tr><td colspan="4">Both estimated (pseudo-depth scaled to GT + est. pose, with symmetry)</td></tr><tr><td>w/ pseudo-depth and est. pose</td><td>0.0298</td><td>0.5004</td><td>0.8302</td></tr></table>

Table 10: Real-world results on YCB-V [50] (CD ↓ / FS(0.05) ↑).
<table><tr><td rowspan="2">Method</td><td colspan="3">symmetry class</td><td colspan="3">visibility</td></tr><tr><td>symmetric (N=103)</td><td>asymmetric (N=298)</td><td>[0.5, 0.7) (N=39)</td><td></td><td>[0.7, 0.9) (N=78)</td><td>[0.9, 1] (N=284)</td></tr><tr><td>Initial mesh</td><td>0.0530 / 0.634</td><td>0.0417</td><td>/0.725 0.0685</td><td>/0.500</td><td>0.0619 /0.552</td><td>0.0365 / 0.771</td></tr><tr><td>Refined (ours)</td><td>0.0503 / 0.650</td><td>0.0416</td><td>/ 0.720 0.0645</td><td>0.519</td><td>0.0579 0.584</td><td>0.0372 /0.760</td></tr></table>

## 4.7 Analysis

Robustness to camera pose and depth. In practice, the depth and the camera pose come from sensors or estimators rather than the GT. Table 9 tests three such settings on InstantMesh/GSO. Estimated depth: we replace the GT depth with UniDepth-V2 [28] pseudo-depth, either rescaled to the GT depth range or afinely fit to the depth rendered from the initial mesh (no GT), keeping the GT pose and refining without symmetry. Both variants still improve over the initial mesh, and the no-GT fit performs on par with the GT-scaled one. Estimated pose: we replace the GT pose with a FoundationPose [44] estimate computed from the RGB image, the no-GT pseudo-depth, the instance mask, and the initial mesh, keeping the GT depth and the symmetry stage. The refinement still reduces CD and improves the F-score at every threshold. Both estimated: combining the estimated pose with the GT-scaled pseudo-depth drops the result below the initial mesh; the refinement thus needs at least one reliable input.

![](images/59e710f1829d62216543832399ccbdc3bdbeaa9460054bfc9ad1935b1affab50.jpg)  
Fig. 2: Qualitative results on OmniObject3D. For each mesh, we render three views: the center view matches the input view, and the left/right views are novel viewpoints.

Real-world generalization. Our main results use synthetic renders, whereas real captures bring noisy sensor depth. We therefore evaluate on YCB-V [50]: we take every tenth test keyframe (401 instances), run InstantMesh on the object crop, and refine with the annotated camera pose and the sensor depth, applying a guarded ICP at evaluation. Table 10 groups the instances by BOP symmetry class and by the visibility of the object in the input frame. Across symmetry classes, the same pipeline improves symmetric objects and leaves asymmetric ones nearly unchanged, matching the synthetic trend: depth drives the gain, and symmetry helps where present. Across visibility levels, refinement gives a clear gain on partially visible objects, whereas almost fully visible objects are already accurate and change only marginally. As most YCB-V instances are almost fully visible, the average gain is smaller than on synthetic data.

## 4.8 Qualitative Results

Across object categories (Figs. 2 and 3), the refinement improves source-view consistency, with cleaner silhouettes, flatter planar regions, and more plausible surface relief on the visible side. The improvements are most noticeable when

![](images/cf79eb72b2fc8155e31e269f6990040cb70fc79f918d5f07cc70fcd146e88f28.jpg)  
Fig. 3: Qualitative results on GSO. For each mesh, we render three views: the center view matches the input view, and the left/right views are novel viewpoints.

the initial mesh is roughly aligned with the input view, and the initial prediction bounds the result. The second row of Fig. 3 shows failure cases: thin or dense structures can be distorted, and imperfect alignment biases the occluded regions.

## 5 Conclusion

We presented a lightweight, plug-and-play RGBD-to-3D refinement framework for any RGB-to-3D reconstructor, requiring no retraining. It corrects the observed surface by depth matching, mirrors these corrections onto the occluded side by symmetry, and propagates them with a smoothness solver, each in closed form. Across five backbones and two datasets, it yields consistent gains that persist with pseudo-depth, larger symmetry gains on symmetric objects, and favorable accuracy and runtime against prior refinement. Future work includes learned completion for severe occlusions and symmetry-plane confidence.

Acknowledgements. This work was supported by Samsung Electronics Co., Ltd. (IO240508-09825-01) and IITP grants (RS-2024-00457882: National AI Research Lab Project (50%), RS-2022-II220959: Few-Shot Learning of Causal Inference in Vision and Language for Decision Making (30%), RS-2022-II220290: Visual Intelligence for Space-Time Understanding & Generation (20%) ) funded by Ministry of Science and ICT, Korea.

## References

1. Boss, M., Huang, Z., Vasishta, A., Jampani, V.: Sf3d: Stable fast 3d mesh reconstruction with uv-unwrapping and illumination disentanglement. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 16240–16250 (2025)

2. Chen, R., Chen, Y., Jiao, N., Jia, K.: Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 22246–22256 (2023)

3. Chen, X., Chen, Y., Xiu, Y., Geiger, A., Chen, A.: Ttt3r: 3d reconstruction as test-time training. arXiv preprint arXiv:2509.26645 (2025)

4. Cheng, Y.C., Lee, H.Y., Tulyakov, S., Schwing, A.G., Gui, L.Y.: Sdfusion: Multimodal 3d shape completion, reconstruction, and generation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4456–4465 (2023)

5. Dastmalchi, H., An, A., Cheraghian, A., Rahman, S., Ramasinghe, S.: Test-time adaptation of 3d point clouds via denoising difusion models. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 1566–1576. IEEE (2025)

6. Deng, C., Jiang, C., Qi, C.R., Yan, X., Zhou, Y., Guibas, L., Anguelov, D., et al.: Nerdi: Single-view nerf synthesis with language-guided difusion as general image priors. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20637–20647 (2023)

7. Denninger, M., Winkelbauer, D., Sundermeyer, M., Boerdijk, W., Knauer, M.W., Strobl, K.H., Humt, M., Triebel, R.: Blenderproc2: A procedural pipeline for photorealistic rendering. Journal of Open Source Software 8(82), 4901 (2023)

8. Downs, L., Francis, A., Koenig, N., Kinman, B., Hickman, R., Reymann, K., McHugh, T.B., Vanhoucke, V.: Google scanned objects: A high-quality dataset of 3d scanned household items. In: 2022 International Conference on Robotics and Automation (ICRA). pp. 2553–2560. IEEE (2022)

9. Hähnlein, F., Gryaditskaya, Y., Shefer, A., Bousseau, A.: Symmetry-driven 3d reconstruction from concept sketches. In: ACM SIGGRAPH 2022 Conference Proceedings. pp. 1–8 (2022)

10. Hong, Y., Zhang, K., Gu, J., Bi, S., Zhou, Y., Liu, D., Liu, F., Sunkavalli, K., Bui, T., Tan, H.: Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023)

11. Huang, Z., Boss, M., Vasishta, A., Rehg, J.M., Jampani, V.: Spar3d: Stable pointaware reconstruction of 3d objects from single images. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 16860–16870 (2025)

12. Hui, K.H., Sanghi, A., Rampini, A., Malekshan, K.R., Liu, Z., Shayani, H., Fu, C.W.: Make-a-shape: a ten-million-scale 3d shape model. In: Forty-first International Conference on Machine Learning (2024)

13. Kim, Y.J., Ha, H., Kim, Y., Surh, J., Ha, H., Oh, T.H.: Metta: Single-view to 3d textured mesh reconstruction with test-time adaptation. In: 35th British Machine Vision Conference (BMVC). British Machine Vision Association (BMVA) (2024)

14. Köser, K., Zach, C., Pollefeys, M.: Dense 3d reconstruction of symmetric scenes from a single image. In: Joint Pattern Recognition Symposium. pp. 266–275. Springer (2011)

15. Kulikajevas, A., Maskeliunas, R., Damasevicius, R., Krilavicius, T.: Auto-refining 3d mesh reconstruction algorithm from limited angle depth data. IEEE Access 10, 87083–87098 (2022)

16. Leung, B., Ho, C.H., Vasconcelos, N.: Black-box test-time shape refinement for single view 3d reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4080–4090 (2022)

17. Li, J., Tan, H., Zhang, K., Xu, Z., Luan, F., Xu, Y., Hong, Y., Sunkavalli, K., Shakhnarovich, G., Bi, S.: Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214 (2023)

18. Li, X., Huang, Z., Thai, A., Rehg, J.M.: Symmetry strikes back: From single-image symmetry detection to 3d generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 743–752 (2025)

19. Lionar, S., Xu, X., Lin, M., Lee, G.H.: Nu-mcc: Multiview compressive coding with neighborhood decoder and repulsive udf. Advances in Neural Information Processing Systems 36, 63011–63022 (2023)

20. Liu, M., Shi, R., Chen, L., Zhang, Z., Xu, C., Wei, X., Chen, H., Zeng, C., Gu, J., Su, H.: One-2-3-45++: Fast single image to 3d objects with consistent multiview generation and 3d difusion. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10072–10083 (2024)

21. Liu, M., Xu, C., Jin, H., Chen, L., Varma T, M., Xu, Z., Su, H.: One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems 36, 22226–22246 (2023)

22. Liu, M., Zeng, C., Wei, X., Shi, R., Chen, L., Xu, C., Zhang, M., Wang, Z., Zhang, X., Liu, I., et al.: Meshformer: High-quality mesh generation with 3d-guided reconstruction model. Advances in Neural Information Processing Systems 37, 59314– 59341 (2024)

23. Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero-1-to-3: Zero-shot one image to 3d object. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9298–9309 (2023)

24. Liu, Y., Lin, C., Zeng, Z., Long, X., Liu, L., Komura, T., Wang, W.: Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453 (2023)

25. Long, X., Guo, Y.C., Lin, C., Liu, Y., Dou, Z., Liu, L., Ma, Y., Zhang, S.H., Habermann, M., Theobalt, C., et al.: Wonder3d: Single image to 3d using crossdomain difusion. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9970–9980 (2024)

26. Melas-Kyriazi, L., Laina, I., Rupprecht, C., Vedaldi, A.: Realfusion: 360deg reconstruction of any object from a single image. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8446–8455 (2023)

27. Park, C., Lee, S., Cho, M.: Afostruction: 3d afordance grounding with generative reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2026)

28. Piccinelli, L., Sakaridis, C., Yang, Y.H., Segu, M., Li, S., Abbeloos, W., Van Gool, L.: Unidepthv2: Universal monocular metric depth estimation made simpler. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

29. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using 2d difusion. arXiv preprint arXiv:2209.14988 (2022)

30. Ren, X., Huang, J., Zeng, X., Museth, K., Fidler, S., Williams, F.: Xcube: Largescale 3d generative modeling using sparse voxel hierarchies. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4209–4219 (2024)

31. Shen, T., Gao, J., Yin, K., Liu, M.Y., Fidler, S.: Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. In: Advances in Neural Information Processing Systems. vol. 34, pp. 6087–6101 (2021)

32. Shi, J., Talak, R., Zhang, H., Jin, D., Carlone, L.: Crisp: Object pose and shape estimation with test-time adaptation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 11644–11653 (2025)

33. Shi, Y., Wang, P., Ye, J., Long, M., Li, K., Yang, X.: Mvdream: Multi-view difusion for 3d generation. arXiv preprint arXiv:2308.16512 (2023)

34. Siddique, A., Lee, S.: Sym3dnet: Symmetric 3d prior network for single-view 3d reconstruction. Sensors 22(2), 518 (2022)

35. Speciale, P., Oswald, M.R., Cohen, A., Pollefeys, M.: A symmetry prior for convex variational 3d reconstruction. In: European Conference on Computer Vision. pp. 313–328. Springer (2016)

36. Sun, X., Wu, J., Zhang, X., Zhang, Z., Zhang, C., Xue, T., Tenenbaum, J.B., Freeman, W.T.: Pix3d: Dataset and methods for single-image 3d shape modeling. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2018)

37. Tang, J., Chen, Z., Chen, X., Wang, T., Zeng, G., Liu, Z.: Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In: European Conference on Computer Vision. pp. 1–18. Springer (2024)

38. Tang, J., Wang, T., Zhang, B., Zhang, T., Yi, R., Ma, L., Chen, D.: Make-it-3d: High-fidelity 3d creation from a single image with difusion prior. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 22819–22829 (2023)

39. Tochilkin, D., Pankratz, D., Liu, Z., Huang, Z., Letts, A., Li, Y., Liang, D., Laforte, C., Jampani, V., Cao, Y.P.: Triposr: Fast 3d object reconstruction from a single image. arXiv preprint arXiv:2403.02151 (2024)

40. Wang, H., Du, X., Li, J., Yeh, R.A., Shakhnarovich, G.: Score jacobian chaining: Lifting pretrained 2d difusion models for 3d generation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12619– 12629 (2023)

41. Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., Zhu, J.: Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems 36, 8406–8441 (2023)

42. Wang, Z., Wang, Y., Chen, Y., Xiang, C., Chen, S., Yu, D., Li, C., Su, H., Zhu, J.: Crm: Single image to 3d textured mesh with convolutional reconstruction model. In: European conference on computer vision. pp. 57–74. Springer (2024)

43. Wei, X., Zhang, K., Bi, S., Tan, H., Luan, F., Deschaintre, V., Sunkavalli, K., Su, H., Xu, Z.: Meshlrm: Large reconstruction model for high-quality mesh. arXiv preprint arXiv:2404.12385 (2024)

44. Wen, B., Yang, W., Kautz, J., Birchfield, S.: Foundationpose: Unified 6d pose estimation and tracking of novel objects. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

45. Wu, C.Y., Johnson, J., Malik, J., Feichtenhofer, C., Gkioxari, G.: Multiview compressive coding for 3d reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9065–9075 (2023)

46. Wu, J., Zhang, C., Zhang, X., Zhang, Z., Freeman, W.T., Tenenbaum, J.B.: Learning shape priors for single-view 3d completion and reconstruction. In: Proceedings of the European conference on computer vision (ECCV). pp. 646–662 (2018)

47. Wu, T., Zhang, J., Fu, X., Wang, Y., Ren, J., Pan, L., Wu, W., Yang, L., Wang, J., Qian, C., et al.: Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 803–814 (2023)

48. Wu, Y., Shi, L., Cai, J., Yuan, W., Qiu, L., Dong, Z., Bo, L., Cui, S., Han, X.: Ipod: Implicit field learning with point difusion for generalizable 3d object reconstruction from single rgb-d images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20432–20442 (2024)

49. Xiang, J., Lv, Z., Xu, S., Deng, Y., Wang, R., Zhang, B., Chen, D., Tong, X., Yang, J.: Structured 3d latents for scalable and versatile 3d generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 21469–21480 (2025)

50. Xiang, Y., Schmidt, T., Narayanan, V., Fox, D.: Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes. In: Robotics: Science and Systems (RSS) (2018)

51. Xu, J., Cheng, W., Gao, Y., Wang, X., Gao, S., Shan, Y.: Instantmesh: Eficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191 (2024)

52. Xu, Y., Shi, Z., Yifan, W., Chen, H., Yang, C., Peng, S., Shen, Y., Wetzstein, G.: Grm: Large gaussian reconstruction model for eficient 3d reconstruction and generation. In: European Conference on Computer Vision. pp. 1–20. Springer (2024)

53. Yuan, Y., Shen, Q., Wang, S., Yang, X., Wang, X.: Test3r: Learning to reconstruct 3d at test time. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025)

54. Zabrodsky, H., Weinshall, D.: Using bilateral symmetry to improve 3d reconstruction from image sequences. Computer vision and image understanding 67(1), 48–57 (1997)

55. Zhang, K., Bi, S., Tan, H., Xiangli, Y., Zhao, N., Sunkavalli, K., Xu, Z.: Gs-lrm: Large reconstruction model for 3d gaussian splatting. In: European Conference on Computer Vision. pp. 1–19. Springer (2024)

## Supplementary Material

## A Sensitivity Analysis

Table 11: CD improvement $( \times 1 0 ^ { - 3 } )$ under camera perturbations (InstantMesh).
<table><tr><td rowspan="2">∆ CD ↓</td><td rowspan="2">baseline</td><td colspan="4">camera pose (polar)</td><td colspan="4">camera pose (azimuth)</td><td colspan="4">intrinsics (focal)</td></tr><tr><td>±1°</td><td>±2°</td><td>±5°</td><td>±10°</td><td>±1°</td><td>±2°</td><td>±5°</td><td>±10°</td><td>+5%</td><td>+10%</td><td>+15%</td><td>+20%</td></tr><tr><td>w depth</td><td>-7.19</td><td>-7.03</td><td>-6.65</td><td>-5.14</td><td>-2.04</td><td>-7.08</td><td>-6.79</td><td>-5.68</td><td>-3.42</td><td>-6.48</td><td>-5.01</td><td>-3.48</td><td>-2.00</td></tr><tr><td>GSO w/ depth, sym.</td><td>-7.88</td><td>-7.63</td><td>-7.26</td><td>-5.68</td><td>-2.53</td><td>-7.76</td><td>-7.55</td><td>-6.52</td><td>-4.26</td><td>-7.33</td><td>-5.87</td><td>-4.25</td><td>-2.67</td></tr><tr><td>Ooni w/ depth</td><td>-8.34</td><td>-8.20</td><td>-7.83</td><td>-6.44</td><td>-3.67</td><td>-8.27</td><td>-8.10</td><td>-7.32</td><td>-5.76</td><td>-7.34</td><td>-5.41</td><td>-3.40</td><td>-1.51</td></tr><tr><td>w/ depth, sym.</td><td>-9.11</td><td></td><td>-9.01-8.65</td><td>-7.13</td><td>-4.13</td><td>-9.03</td><td>-8.94</td><td>-8.23</td><td>-6.67</td><td>-8.28</td><td>-6.32</td><td>-4.25</td><td>-2.23</td></tr></table>

Camera pose and intrinsics. We perturb only the input-view camera on every fifth object of GSO and OmniObject3D (InstantMesh, GT depth): polar or azimuth rotations of ±{1, 2, 5, 10}<sup>◦</sup> and focal-length inflation of {5, 10, 15, 20}%. Table 11 reports the change in CD relative to the initial mesh (more negative is better). Refinement stays beneficial under every perturbation and degrades gracefully (on GSO, from −7.19 to −2.04 at ±10<sup>◦</sup> polar rotation and −2.00 at +20% focal inflation), and symmetry-guided refinement is best at every level.

Depth perturbation. On GSO with InstantMesh and the GT pose (initial CD 0.0283), we perturb the GT depth before back-projection with Gaussian noise (σ in % of the foreground depth range), Gaussian blur (kernel size in pixels), random hole dropout (% of foreground pixels), or a global scale error (Table 12). Noise up to 5%, blur, and up to 20% missing pixels change CD by at most 0.002. Only a global scale error is harmful: 2% costs 0.002–0.004 CD, and at 5% the result falls below the initial mesh. The afine fit to the initial mesh removes exactly this error, at a cost of 0.0003 CD (Table 9).

Table 12: Depth-perturbation sweep (InstantMesh, GSO; CD).
<table><tr><td rowspan="3">CD↓</td><td rowspan="3" colspan="2">unperturbed</td><td colspan="4">noise σ</td><td colspan="4">blur</td></tr><tr><td>0.5%</td><td>1%</td><td>2%</td><td>5%</td><td>3px</td><td>5px</td><td></td><td>9px</td></tr><tr><td>w/depth</td><td colspan="2">0.0212</td><td>0.0212</td><td>0.0212</td><td>0.0212</td><td>0.0213</td><td>0.0213</td><td></td><td>0.0213</td><td>0.0214</td></tr><tr><td>w/ depth, sym.</td><td colspan="2">0.0205</td><td>0.0205</td><td>0.0205</td><td></td><td>0.0205</td><td>0.0205</td><td>0.0206</td><td>0.0206</td><td></td><td>0.0207</td></tr><tr><td></td><td colspan="4">holes</td><td colspan="8"></td></tr><tr><td>CD↓</td><td colspan="4"></td><td colspan="2"></td><td colspan="4">scale error</td><td></td></tr><tr><td></td><td>1%</td><td>2%</td><td>5%</td><td>10%</td><td>20%</td><td>+1% +2%</td><td>-2%</td><td>+5%</td><td>-5%</td><td>+10%</td><td>-10%</td></tr><tr><td>w/ depth w/ depth, sym. 0.0209</td><td>0.0215</td><td>0.0216</td><td>0.0219</td><td>90.0223</td><td>0.0228</td><td>0.0218 0.0232</td><td>0.0251</td><td>0.0301</td><td>0.0325</td><td></td><td>0.0411 0.0427</td></tr></table>

Depth-to-vertex density. On GSO with InstantMesh, the GT pose, and the GT depth, we vary the depth back-projection resolution and the vertex cap, which changes the number of depth points by 16× and of visible vertices by 2.7× (Table 13; counts are means over the 1,030 objects, and handles moves the matched vertices without propagation). CD stays within 0.0203–0.0212. With the sparsest depth, the matching covers only 8% of the visible vertices, yet CD still drops by 25%: sparse depth yields fewer handles, not wrong displacements.

Table 13: Depth-resolution and vertex-density sweep (InstantMesh, GSO; CD).
<table><tr><td rowspan="2">vertices depth</td><td rowspan="2"></td><td colspan="3">count</td><td colspan="3">CD↓</td></tr><tr><td>visible</td><td>depth</td><td>matched</td><td>handles</td><td>depth</td><td>depth, sym.</td></tr><tr><td rowspan="3">30k</td><td> $6 4 ^ { 2 }$ </td><td>10393</td><td>892</td><td>805</td><td>0.0241</td><td>0.0221</td><td>0.0212</td></tr><tr><td> $1 2 8 ^ { 2 }$  (ours)</td><td>10393</td><td>3568</td><td>3093</td><td>0.0221</td><td>0.0212</td><td>0.0205</td></tr><tr><td> $2 5 6 ^ { 2 }$ </td><td>10393</td><td>14271</td><td>8036</td><td>0.0211</td><td>0.0209</td><td>0.0204</td></tr><tr><td rowspan="3">10k</td><td> $6 4 ^ { 2 }$ </td><td>3902</td><td>892</td><td>788</td><td>0.0234</td><td>0.0220</td><td>0.0212</td></tr><tr><td> $1 2 8 ^ { 2 }$ </td><td>3902</td><td>3568</td><td>2335</td><td>0.0219</td><td>0.0218</td><td>0.0210</td></tr><tr><td> $2 5 6 ^ { 2 }$ </td><td>3902</td><td>14271</td><td>3487</td><td>0.0219</td><td>0.0205</td><td>0.0203</td></tr></table>

## B Symmetry Analysis

![](images/f4e52483ae495ea2b686aec7028b28d9f41d1a308405b8e7bae067e4b9327b1a.jpg)  
Fig. 4: Rasterized signed distance maps for diferent candidate symmetry planes.

Symmetry plane detection. Figure 4 visualizes the two signed-distance maps of the plane-detection criterion (Sec. 3.3) for several candidate planes. The score varies smoothly across the swept planes and has a single clear minimum at the correct plane, which keeps the selection stable.

Symmetry-plane source. We replace our plane with Reflect3D’s [18] learned prediction on the 1,023 GSO objects for which Reflect3D returns a valid plane (InstantMesh, GT pose and depth). This hurts refinement (Table 14): for objects with several symmetries its top-1 may be a top or front plane, whereas our camera-frame search returns the left/right plane that mirrors the visible side onto the occluded one, at ∼0.27 s per object with no extra network.

Table 14: Symmetry-plane source ablation (InstantMesh, GSO).
<table><tr><td>Method</td><td>symmetric</td><td></td><td>asymmetric</td></tr><tr><td>CD↓</td><td>FS(0.05) ↑</td><td>CD↓</td><td>FS(0.05) ↑</td></tr><tr><td>Initial mesh</td><td>0.0247</td><td>0.873</td><td>0.0329 0.787</td></tr><tr><td>w/ GT depth</td><td>0.0185 0.912</td><td>0.0247</td><td>0.851</td></tr><tr><td>w/ depth + symmetry (ours)</td><td>0.0174</td><td>0.924 0.0246</td><td>0.857</td></tr><tr><td>w/ depth + symmetry (Reflect3D [18])</td><td>0.0217</td><td>0.883 0.0284</td><td>0.818</td></tr></table>

Failure analysis. For each GSO object, we measure the change from the symmetry stage, $\varDelta = \mathrm { C D } ( \mathrm { d e p t h + s y m . } ) - \mathrm { C D } ( \mathrm { d e p t h } )$ (negative is better), and count the object as degraded when $\varDelta \mathrm { ~ > ~ } 0 . 0 0 1$ . Table 15 splits the objects by Reflect3D label and also reports the thin-structure decile (top 10% of surface area over bounding-box volume). The mean efect is a gain on symmetric objects and at most +0.0008 on asymmetric ones, but 16–33% of the symmetric and 21–48% of the asymmetric objects degrade (24–32% of all objects on OmniObject3D). For InstantMesh, 90 of the 254 degraded objects are symmetric-labelled, where a wrong plane is the likely cause, and 164 are asymmetric. The thin-structure decile behaves like the full set. The score margin of the plane search does not predict $\varDelta$ (Spearman $| \rho | \leq 0 . 1 3 )$ , gating on it never beats always-on symmetry, and even an oracle per-object choice gains only 0.0007 CD. We leave confidence estimation to future work.

## C Implementation Details

Rendering and views. The 12 azimuths of the 24 rendered views are spaced by 15<sup>◦</sup>, from −82.5<sup>◦</sup> to 82.5<sup>◦</sup>.

Depth- and symmetry-guided stages. We resize the source-view depth to 128×128 with bilinear interpolation. The auction solver uses $\varepsilon = 1 0 ^ { - 3 }$ of the cost

Table 15: Per-object efect of the symmetry stage on GSO.
<table><tr><td rowspan="2">Backbone</td><td colspan="4">symmetric (N=572)</td><td colspan="4">asymmetric (N=458)</td><td colspan="2">thin (N=103)</td></tr><tr><td>mean ∆</td><td>% degr.</td><td> $\mathrm { p 9 0 }$ </td><td>max</td><td>mean ∆</td><td>% degr.</td><td> $\mathrm { p 9 0 }$ </td><td>max</td><td>mean ∆</td><td>% degr.</td></tr><tr><td>LGM</td><td>-0.0016</td><td>22</td><td>+0.0023</td><td>+0.0113</td><td>-0.0017</td><td>21</td><td>+0.0032</td><td>+0.0112</td><td>-0.0023</td><td>17</td></tr><tr><td>CRM</td><td>-0.0014</td><td>33</td><td>+0.0039</td><td>+0.0149</td><td>+0.0006</td><td>41</td><td>+0.0043</td><td>+0.0181</td><td>-0.0002</td><td>41</td></tr><tr><td>SF3D</td><td>-0.0020</td><td>21</td><td>+0.0025</td><td>+0.0084</td><td>+0.0007</td><td>47</td><td>+0.0046</td><td>+0.0124</td><td>-0.0009</td><td>20</td></tr><tr><td>SPAR3D</td><td>-0.0015</td><td>24</td><td>+0.0029</td><td>+0.0107</td><td>+0.0008</td><td>48</td><td>+0.0048</td><td>+0.0132</td><td>+0.0007</td><td>36</td></tr><tr><td>InstantMesh</td><td>-0.0012</td><td>16</td><td>+0.0016</td><td>+0.0093</td><td>-0.0001</td><td>36</td><td>+0.0032</td><td>+0.0106</td><td>-0.0003</td><td>17</td></tr></table>

scale and up to 12 ε-scaling rounds in both stages. Plane detection scores 1,260 candidates (180 azimuths $\times ~ 7$ elevations), each rasterized into two $5 1 2 \times 5 1 2$ maps.

Propagation stage. The boundary-anchor weight is $\gamma = 1 0 ^ { - 3 }$ and the Tikhonov weight $\tau = 1 0 ^ { - 6 }$ of the mean diagonal of $L _ { f f }$ , and Taubin smoothing uses $\lambda = 0 . 5$ and $\nu = - 0 . 5$

## D Evaluation Details

Protocol. The rotation search is coarse-to-fine over azimuth, elevation, and roll, without ICP: we score a $2 4 ^ { 3 }$ grid on a quarter of the sampled points and then a $4 8 ^ { 3 }$ grid within the 64 best coarse cells. We decimate meshes with the quadric error metric and score the initial mesh before decimation. SSIM uses an 11 × 11 Gaussian window (σ = 1.5) and LPIPS an AlexNet backbone.

Test-time refinement baselines. REFINE [16] runs 400 optimization iterations per instance, and we compute CD with its evaluator after per-candidate ICP. MeTTA [13] distills Zero-1-to-3 [23] onto a DMTet [31] mesh, which takes ∼30 min per instance on an RTX 3090; as its meshes live in a per-instance frame, we also compute CD after per-candidate ICP. For the pseudo-depth row of Table 5, we fit UniDepth-V2 [28] to the depth rendered from the initial mesh by min-max matching over overlapping pixels and warp it into the silhouette of the GT model with a thin-plate spline.

Novel-view evaluation against MeTTA. Real images exist only at the input view, so we render the GT CAD mesh as the reference at the input view and at 7 novel azimuths and render every mesh geometry-only (Table 16). Against these renders, MeTTA falls below the initial mesh at every view, including the input view, where it scores better against the photo (Table 5). Ours with GT depth improves the initial mesh at the input view and matches or improves it at novel views, except for CLIP; with pseudo-depth it matches at the input view but falls below at novel views, consistent with its CD in Table 5.

Table 16: Evaluation against GT renders at input and novel views (Pix3D).
<table><tr><td rowspan="2">Method</td><td colspan="2">PSNR ↑</td><td colspan="2">LPIPS↓</td><td colspan="2">IoU ↑</td><td colspan="2">CLIP ↑</td></tr><tr><td>input</td><td>novel</td><td>input</td><td>novel</td><td>input</td><td>novel</td><td>input</td><td>novel</td></tr><tr><td>Initial mesh</td><td>19.2</td><td>19.4</td><td>0.161</td><td>0.139</td><td>0.82</td><td>0.81</td><td>0.864</td><td>0.850</td></tr><tr><td>MeTTA [13]</td><td>16.4</td><td>16.6</td><td>0.282</td><td>0.277</td><td>0.68</td><td>0.62</td><td>0.774</td><td>0.780</td></tr><tr><td>Ours (w/ GT depth)</td><td>20.5</td><td>19.9</td><td>0.127</td><td>0.131</td><td>0.86</td><td>0.81</td><td>0.884</td><td>0.830</td></tr><tr><td>Ours (w/ pred. depth scaled to pred. mesh)</td><td>19.2</td><td>17.7</td><td>0.156</td><td>0.194</td><td>0.83</td><td>0.67</td><td>0.794</td><td>0.794</td></tr></table>

RGB-D baselines. We run MCC [45], NU-MCC [19], and Afostruction [27] from their released weights; Afostruction uses its single-view pipeline. As its symmetry stage brings no gain (Table 3), we recommend depth-only refinement for backbones whose occluded side is already conditioned on depth.

Robustness to depth and pose. For Table 9, the no-GT pseudo-depth is fit to the depth rendered from the initial mesh over overlapping pixels, and FoundationPose [44] runs with its default settings.

YCB-V. Beyond the visibility split of Table 10, 12 of the 21 objects and 54% of the instances improve. The rigid residual of the alignment to the CAD model and the initial CD, which is 1.6× that on GSO, limit the gain.