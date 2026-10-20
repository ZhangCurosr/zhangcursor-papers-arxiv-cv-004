# SpaRV: Sparse Radiance Voxels for Scalable Geometrically Consistent LiDAR-Visual Mapping and Rendering

Jianheng Liu, Yunfei Wan, Fangming Cheng, Xiangyu Hong, Zuhao Zou, Haotian Li, Longji Yin, Chunran Zheng, Jiarong Lin, and Fu Zhang

![](images/b3364f07252f417c884df02e90d9cc5249d3af48a7b40aef2fe6cfd6afd1a6eb.jpg)  
Fig. 1. The two self-collected scenes, Garden (top) and Campus (bottom), reconstructed by SpaRV on a single GPU. Each scene is drawn as the accumulated LiDAR point cloud of the capture, viewed from above. The red dashed boxes mark close-up regions; beside each box, points is the point cloud of that region seen from a free viewpoint and render is the SpaRV rendering from the same viewpoint. Garden has 44,550 images and 146.7 M LiDAR points along a 2,239 m trajectory, with a 45,199 m<sup>2</sup> footprint; Campus has 63,582 images and 151.6 M LiDAR points along a 3,534 m trajectory, with a 234,035 m<sup>2</sup> footprint.

Abstract—Robotic simulation, navigation, and inspection need photorealistic and geometrically accurate maps. Radiance fields achieve photorealism, but building them from robot data raises three challenges. A robot’s trajectory serves its task, not reconstruction, so images from few viewpoints leave the geometry under-constrained. Residual errors in camera poses and intrinsics blur the map. The map’s GPU memory grows with scene size, which limits its level of detail in large scenes. We present SpaRV, a Sparse Radiance Voxel map rendered by differentiable ray marching, which lets irregular LiDAR scans supervise it alongside images. LiDAR depth and occupancy anchor the underconstrained geometry, and normal and smoothness regularizers keep it smooth between LiDAR points. To correct camera errors, SpaRV alternates map training with photometric optimization of the camera poses and focal lengths. To bound GPU memory in training and rendering, SpaRV holds one submap at a time at full resolution and the rest at coarser octree levels.

Experiments show that SpaRV keeps the geometry consistent away from the captured trajectory and sharpens textures by optimizing the camera parameters. On one 24 GB GPU, SpaRV reconstructs a university campus captured in 63,582 images and simulates other cameras and LiDARs. Our code will be released at https://github.com/hku-mars/SpaRV.

## I. INTRODUCTION

Digital twins of real environments support robot simulation, navigation, and inspection [1], and simulators train robot policies before deployment [2]. Such applications need one map that supports novel view synthesis, surface extraction, and sensor simulation. It must therefore be both photorealistic and geometrically accurate. Cameras and LiDAR supply both properties: images record appearance, and LiDAR measures metric range. LiDAR-visual SLAM [3], [4] already fuses them into accurate colored point clouds, but point clouds do not render photorealistic novel views.

Radiance fields such as Neural Radiance Fields (NeRF) [5] and 3D Gaussian Splatting (3DGS) [6] render photorealistic novel views by fitting a scene representation to posed images. Fitted to robot data, however, a radiance field can reproduce the captured images and still fail once the viewpoint leaves the captured trajectory. At an extrapolated view, the image-only methods break into noise and holes (Fig. 2). This paper asks how to build a radiance map from robot data that stays reliable at viewpoints away from the captured trajectory. Radiance fields are usually built from curated captures with dense views, accurate camera parameters, and a scene of limited extent. Robot data breaks each of these conditions, which raises three challenges to reliability: under-constrained geometry, inaccurate camera parameters, and limited GPU memory.

The first challenge is under-constrained geometry. A robot’s trajectory follows its task rather than the needs of reconstruction, so each region is observed from few and uneven viewpoints. Photometry constrains depth only through the parallax between views, and a short baseline gives little parallax. A radiance field then overfits the captured images by placing geometry at the wrong depth, and the misplaced geometry appears as floating artifacts away from the trajectory. LiDAR measures depth directly, and prior work supervises the rendered depth with LiDAR [7], [8] or builds a geometric prior from it [9], [10]. Depth supervision, however, constrains only the rays that LiDAR measures, and camera rays between the sparse LiDAR points can still terminate at floating artifacts. This motivates us to ground the geometry along the LiDAR rays and also on the camera rays that LiDAR does not measure.

Correct geometry must also render consistently as the viewpoint moves, which depends on the representation. 3DGS sorts its Gaussians by their centers for each view, and a change of viewpoint can invert this order, so the rendered surface pops [11]. Popping persists even when LiDAR places the Gaussians on the true surface, as for LetsGo and GS-SDF in Fig. 2. NeRF avoids popping by compositing samples along each ray in the order the ray meets them, but its capacity is fixed by the network parameters. Explicit spatial structures [12]–[15] attach parameters to locations, so capacity can grow where the scene needs it. Among them, SVRaster [16] uses non-overlapping adaptive sparse voxels, which keep the compositing order exact and add detail where required. SVRaster renders these voxels by rasterization, however, which assumes a pinhole image grid and does not directly apply to LiDAR rays or non-pinhole cameras. We therefore render the sparse voxels with a differentiable ray marcher that evaluates each ray independently, so the map can be supervised directly along the LiDAR rays.

The second challenge is inaccurate camera parameters. Camera poses and intrinsics are difficult to obtain exactly in practice, whether they come from calibration [17], Structurefrom-Motion [18], SLAM [3], or LiDAR-visual bundle adjustment [19]. A small error projects the same surface point to different pixels in different images. A map fitted to these misaligned images tends to have blurred textures, even where LiDAR constrains its geometry well. Prior work optimizes the cameras jointly with the field from little or no prior [20], [21], or holds the trajectory fixed and refines only the LiDARcamera extrinsics [22], [23]. The cameras of robot data, in contrast, are metrically plausible but carry residual errors in every frame. This motivates us to optimize the pose and focal length of each camera against the map, in alternation with map training.

![](images/2d83211cc4e182df1d7968a40c6404d06bbc0948112b807c0560b4b51a23244b.jpg)  
Fig. 2. Rendering results on the FAST-LIVO2 Street scene at an extrapolated view away from the captured trajectory, with zoom crops.

The third challenge is limited GPU memory. The GPU memory of a map grows with the extent of the scene. Within a fixed GPU memory budget, a large scene can then be trained and rendered only at a coarser level of detail. A single view, however, needs fine detail only near the camera, since distant regions project to few pixels. Divide-and-conquer methods [24], [25] train parts of the scene separately, and level-of-detail methods [26] select coarser primitives for distant regions at render time. Hierarchical 3DGS [27] combines the two, but it needs extra stages after training: it merges the Gaussians of each part into a hierarchy, optimizes the merged nodes, and consolidates the parts into one hierarchy. This motivates us to simplify the pipeline: the octree keeps every coarser level that training passes through, and each coarser voxel was already trained at its own level before it was subdivided, so it serves as that level of detail without further optimization. One cut rule that selects voxels by their projected size then serves training as well as rendering.

To address these challenges, we present SpaRV, a Sparse Radiance Voxel map for scalable, geometrically consistent LiDAR-visual mapping and rendering. Our contributions are as follows:

1) Ray-marching rendering of sparse radiance voxels: A differentiable ray marcher lets camera pixels, LiDAR rays, and rays of non-pinhole cameras supervise the map through a unified renderer (Secs. III–IV).

2) Geometry-grounded map training: LiDAR depth supervi sion, occupied voxel guidance, and normal and smoothness regularizers keep the map geometrically consistent beyond the captured trajectory (Sec. V).

3) Map-driven camera optimization: SpaRV alternates map training with photometric optimization of the camera poses and focal lengths, which reduces the texture blur caused by residual camera errors (Sec. VI).

4) Large-scale reconstruction under bounded GPU memory: One octree and one granularity-driven cut serve training and rendering, holding the submap in focus at the finest available level and the rest of the scene at coarser levels (Sec. VII).

We evaluate SpaRV on synthetic indoor scenes, two outdoor LiDAR-visual datasets, an omnidirectional benchmark, and two large scenes we collected. At viewpoints away from the captured trajectory, SpaRV keeps the geometry consistent, while image-only baselines degrade and Gaussian-based baselines pop. Camera optimization sharpens textures on real data, and the optimized cameras also improve the baselines retrained on them. On a single GPU, SpaRV reconstructs and renders a kilometer-scale university campus. The trained maps also support surface extraction and the simulation of other camera and LiDAR models.

## II. RELATED WORK

## A. Radiance Field Representations

Radiance-field representations store scene parameters either implicitly in network weights or explicitly at spatial locations. Neural Radiance Fields (NeRF) [5] established continuous volumetric modeling for photorealistic novel view synthesis. A multilayer perceptron maps a position and viewing direction to density and color. Methods that store learned features in hash grids [13] or factorized tensors [28] retain a network only as a decoder, which reduces training cost. A shared decoder, however, entangles those features: an update in one region perturbs weights used throughout the field, so previously reconstructed regions may drift as new observations arrive [29], [30]. The parameter is also fixed for the whole field, so capacity cannot readily be increased locally where additional detail is required.

3D Gaussian Splatting (3DGS) [6] makes the parameters explicit and discrete. The scene becomes a set of anisotropic Gaussian primitives, each with its own position, shape, opacity, and appearance. A highly efficient differentiable rasterizer sorts the primitives and blends them into the image. The primitives remain independent, so the field they assemble has no intrinsic compositing order: the per-view sort on their centers can invert under viewpoint change and the rendered surface pops [11]. Voxel-based radiance fields store parameters explicitly on a voxel grid. Queries are answered by interpolating vertex values, without a decoder network. NSVF [31] attached learned features to a sparse voxel octree, DVGO [15] optimized an explicit density grid, and Plenoxels [14] removed the network altogether, reducing optimization from hours to minutes. SVRaster [16] further makes those voxels adaptive: non-overlapping sparse voxels across levels of detail, with efficient rasterization and an exact compositing order that avoids popping. Its rasterizer, however, assumes a pinhole image grid and does not directly apply to LiDAR rays or non-pinhole cameras. In contrast, our method shares the adaptive sparsevoxel representation of SVRaster but differs from it in three designs. A differentiable ray marcher replaces the rasterizer, so camera pixels, LiDAR rays, and rays of non-pinhole cameras supervise the map through the unified renderer (Sec. IV). LiDAR measurements ground the geometry (Sec. V), and a granularity-driven octree cut keeps GPU memory bounded in large scenes (Sec. VII).

## B. Geometric Grounding in Radiance Fields

A photometric loss alone leaves geometry underconstrained. 2DGS [32] collapses each Gaussian primitive to an oriented disk and aligns the normals derived from the rendered depth to the disk normals. PGSR [33] further introduces unbiased depth rendering with multi-view photometric regularization to preserve global geometric accuracy. They require the rendered geometry to be selfconsistent, which improves the quality of the surface reconstruction. DS-NeRF [34] reuses the sparse 3D points that Structure-from-Motion (SfM) [18] has already produced, which anchors the scene geometry to the observation. This supervision inherits what SfM has produced, so it thins out exactly where correspondence fails, which is where geometry is hardest to recover. Monocular depth predictors such as Depth Anything [35] provide dense supervision [36], but their depth is not guaranteed to be accurate or metrically consistent across images.

Range sensors measure depth directly: each range measurement gives the metric distance to a surface along its own ray, independent of texture. The most common approach is to penalize the difference between rendered depth and measured depth. This loss is used wherever such measurements are available, across implicit fields [7], [25], [29], [30], [37], [38] and 3D Gaussian representations [8], [39]. The alternative is to reconstruct an explicit or implicit geometry from the depth measurements and use it to constrain the radiance field. LetsGo [9] uses such a geometry as a starting point. It meshes the LiDAR point cloud by Poisson reconstruction and initializes Gaussians from that mesh. A depth term on the measurements then constrains the rendered depth during training. LIV-GaussMap [40] and LI-GS [8] constrain splat orientation and distribution with LiDAR-derived planar priors and Gaussian mixtures. M2Mapping [41] and GS-SDF [10] instead build a neural SDF from the range measurements. GS-SDF further fits the Gaussians to its isosurface to improve the geometric consistency of 3DGS. In contrast, our method grounds the sparse-voxel field itself: LiDAR depth supervision acts directly along the LiDAR rays, while occupied voxel guidance and the normal and smoothness regularizers constrain the field where LiDAR does not measure.

## C. Camera Optimization in Radiance Fields

Radiance-field reconstruction depends on accurate camera parameters. SfM [18] is the common source, but it is computationally expensive and brittle in textureless or repetitive scenes. SLAM systems [3], [42] produce poses efficiently in these environments, yet their residual error still exceeds the pixellevel accuracy a radiance field requires. Joint photometric optimization against the field corrects such errors, and existing methods differ mainly in what they start from. Some recover the cameras with little or no prior. Starting from rough or identity poses, they optimize the camera parameters jointly with the field through the photometric loss: the poses in BARF [20], [43], and also the focal length in NeRF-- [21]. SCNeRF [44] additionally calibrates the intrinsics and the lens distortion, and InstantSplat [45] optimizes the poses jointly with 3D Gaussians from sparse, unposed views. Others estimate the poses incrementally as the frames arrive. COLMAPfree 3DGS [46] holds the camera fixed and instead estimates the rigid transform of the Gaussians between consecutive frames, which is the relative pose. 3DGS SLAM systems [47], [48], mostly with RGB-D input, track each incoming camera by minimizing the photometric and depth residuals against the map rendered at its pose, and then extend the map with that frame. A third group refines cameras that are already approximately known, which is the setting closest to ours. CamP [49] refines the poses and intrinsics of SfM cameras jointly with the field, and reparameterizes the cameras to make this optimization more stable. With photometric supervision alone, however, the joint optimization can still converge to misaligned geometry. With LiDAR, MOISST [22] and Uni-Cal [23] optimize a neural scene representation jointly with the extrinsics between the sensors, while holding the sensor trajectory from SLAM or localization fixed. In contrast, our method refines every camera of the input trajectory, such as one from LiDAR-inertial-visual odometry [3], whose poses are metrically plausible but carry residual errors in every frame. It optimizes the cameras against a map whose geometry is anchored by LiDAR. The cameras and the map are updated in alternation. Each update corrects the pose and focal length of a camera by an error-state step around its initialization.

![](images/d3206428b0791dea60e8b2b227954df6ef2216a2c6117824b88c86e669440776.jpg)  
Fig. 3. Overview of SpaRV. From RGB images, LiDAR scans, and initial camera parameters (1), SpaRV builds a sparse radiance voxel map for geometricall consistent rendering. The top panel shows one frame as its allocated voxels (left) and the view rendered from them (right). (2) A granularity-driven cut selects voxels from the octree, and a differentiable ray marcher renders them into color C and depth �. (3) Training runs over the submaps coarse-to-fine and alternates map training with camera optimization. The trained map is applied to novel view synthesis, surface reconstruction, and LiDAR simulation.

## D. Large-Scale Radiance Field Reconstruction

Scaling radiance fields to large scenes raises two distinct issues: representing unbounded outdoor extents, and keeping active GPU memory bounded during training and rendering. NeRF++ [50] and Mip-NeRF 360 [51] address unbounded extents by wrapping the whole space with a single non-linear parameterization into a bounded domain, so the field can be queried anywhere. 3DGS does not require this contraction: Gaussian positions live in Euclidean world space, so distant content is represented by placing primitives farther away rather than by warping the domain. Active GPU memory is kept bounded by divide-and-conquer training and levelof-detail (LoD) rendering. Divide-and-conquer methods such as Block-NeRF [24] and SiLVR [25] decompose a largescale capture into individually trained NeRFs, which decouples rendering time from scene size and lets one block be updated independently. LoD rendering methods such as Octree-GS [26] organize the Gaussians into multi-resolution anchors and select a level per view, so the number of primitives inside the viewing frustum stays stable as the viewpoint zooms out. Hierarchical 3DGS (H3DGS) [27] combines the two. It partitions the capture into parts trained independently, merges the Gaussians of each part into coarser intermediate nodes, and consolidates the parts into one hierarchy that is optimized further. Rendering selects Gaussian nodes through this hierarchy by the projected size of its Gaussians and interpolates between levels for smooth transitions. In contrast, our method addresses both issues. For unbounded extents, it follows the NeRF line and queries its bounded voxel domain in contracted space. For GPU memory, a granularity-driven cut of one octree holds the submap in focus at the finest available level and the rest of the scene at coarser levels, in training as well as in rendering.

![](images/823b090599971e97f98653f42da70e54b7778cc434c8335dd1e3e41909407307.jpg)  
Fig. 4. Sparse voxels on a two-object scene. (a) Voxels are allocated only around the surfaces. (b) Finer voxels (darker) concentrate where the detail is required. (c) Trilinear interpolation of the eight vertex densities.

## III. SPARSE RADIANCE VOXEL REPRESENTATION

Given RGB images and LiDAR points, together with initial camera parameters, SpaRV reconstructs a sparse radiance voxel map for geometrically consistent rendering (Fig. 3). A granularity-driven cut selects the resident voxels from the octree by their projected size, so GPU memory stays bounded as the scene grows (Sec. VII). The selected voxels form an adaptive sparse-voxel map, with finer voxels where additional detail is required (Sec. III-A). A differentiable ray marcher traverses these resident voxels along each ray independently, and therefore renders camera images and irregular LiDAR scans in their native sampling patterns (Sec. IV). Both then supervise the map through the same renderer. SpaRV alternates between geometry-grounded map training (Sec. V) and camera optimization against the map (Sec. VI). The trained map supports novel view synthesis, surface extraction, and LiDAR simulation.

## A. Sparse Voxels

SVRaster allocates voxels following an octree spacepartition layout. A maximum level of detail � defines the finest voxel resolution at $( 2 ^ { L } ) ^ { 3 }$ . Let $w _ { \mathrm { s } } \in \mathbb { R }$ be the map size and ${ \mathbf w } _ { \mathrm { c } } \in \mathbb { R } ^ { 3 }$ be its center in world space. The voxel index $\mathbf { v _ { i d x } } = \{ i , j , k \} \in [ 0 , \ldots , 2 ^ { L } - 1 ] ^ { 3 }$ together with an octree level $l \in [ 1 , L ]$ define the voxel size $\nu _ { \mathrm { s } }$ and voxel center ${ \bf v } _ { \mathrm { c } }$ as:

$$
\nu _ { \mathrm { s } } = 2 ^ { - l } w _ { \mathrm { s } } , \quad \mathbf { v } _ { \mathrm { c } } = \mathbf { w } _ { \mathrm { c } } - 0 . 5 w _ { \mathrm { s } } + \left( \mathbf { v } _ { \mathrm { i d x } } + 0 . 5 \right) \nu _ { \mathrm { s } } \ .\tag{1}
$$

Internally, this index is mapped to a Morton code for efficient spatial indexing. The voxels are kept in a compact flat 1D array indexed by (octpath, octlevel) pairs. Fig. 4 (a)–(b) illustrate the resulting sparsity and the detail-adaptive resolution obtained by the voxel adaptation of Sec. V-A.

The density field is piecewise trilinear: the value at a query point is determined by the voxel that contains it, with closed-form spatial derivatives inside each voxel (Fig. 4 (c)). Learnable densities reside at the vertices, which adjacent voxels share. For a query point p inside a voxel, the density is the trilinear interpolant of the eight vertex values:

$$
\sigma ( { \bf p } ) = \mathrm { i n t e r p } ( { \bf v } _ { \mathrm { g e o } } , { \bf p } ) = \sum _ { i , j , k \in \{ 0 , 1 \} } w _ { i j k } ( { \bf p } ) \sigma _ { i j k } ~ ,\tag{2}
$$

where $\mathbf { v } _ { \mathrm { g e o } } \in \mathbb { R } ^ { 2 \times 2 \times 2 }$ holds the densities $\sigma _ { i j k }$ at the eight voxel vertices [16] and $w _ { i j k } ( \mathbf { p } )$ are the trilinear weights of p.

For view-dependent appearance, SVRaster stores spherical harmonic (SH) coefficients per voxel. A per-voxel coefficient is constant inside the voxel, giving no gradient with respect to the sample position. Camera optimization backpropagates the photometric residual through that position and requires an appearance gradient with respect to it. SpaRV stores the degree-0 (DC) component $\mathbf { v } _ { \mathrm { s h } _ { 0 } }$ at the voxel vertices and interpolates it like the density, while the higher-order terms remain per voxel:

![](images/786250a71d7da15241ca8271fbebf8566243f9c932183525cd0b05de0537cc88.jpg)  
Fig. 5. Contraction map Contr in 2D. (a) World space: rays are straight, the identity box of half-extent $r _ { i }$ is undistorted, and the warped regions map to axis-aligned voxels in contracted space. (b) Contracted space: the same rays curve, and the voxels fall into the inner, mixed, outer, and outside-scene classes. Dots mark corresponding voxel-boundary intersections.

$$
\mathrm { S H } _ { 0 } ( { \bf p } ) = \mathrm { i n t e r p } ( { \bf v } _ { \mathrm { s h } _ { 0 } } , { \bf p } ) \ .\tag{3}
$$

The interpolated DC field supplies that gradient throughout the voxel, and directional shading stays compact.

## B. Contracted-Space Representation

The sparse-voxel map occupies a bounded domain of size $w _ { \mathrm { s } } .$ Outdoor scenes, however, contain distant content that lies outside this domain. SVRaster [16] covers such content by enlarging the domain in world space by a fixed number of octree levels. At a fixed maximum level �, voxel size scales with $w _ { \mathrm { s } } \left( \mathrm { E q . \left( 1 \right) } \right)$ , so each added level doubles the size of the finest voxels in the near field, and content beyond the enlarged bound can only be fitted onto the outermost voxels, without correct geometry. Following unbounded-scene radiance-field practice [50], [51], SpaRV therefore maps the unbounded world into a finite domain by parameterizing the voxels in a contracted space as shown in Fig. 5. Let $\mathbf { x } \in \mathbb { R } ^ { 3 }$ be a worldspace point, $r _ { i } > 0$ the half-extent of the inner identity box centered at ${ \bf w } _ { \mathrm { c } } ,$ and $r _ { c } = 0 . 5 w _ { \mathrm { s } }$ the outer contraction radius. We define a contraction map Contr : $\mathbf x \mapsto \tilde { \mathbf x }$ such that

$$
\begin{array} { r } { \tilde { \mathbf { x } } = \mathrm { C o n t r } ( \mathbf { x } ) = \left\{ \begin{array} { l l } { \mathbf { x } , \quad } & { \| \mathbf { x } - \mathbf { w } _ { \mathrm { c } } \| _ { \infty } \leq r _ { i } , } \\ { \mathbf { w } _ { \mathrm { c } } + s ( \mathbf { x } ) \left( \mathbf { x } - \mathbf { w } _ { \mathrm { c } } \right) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{4}
$$

Let $\delta = \mathbf { x } - \mathbf { w } _ { \mathrm { c } } , m = \| \pmb { \delta } \| _ { \infty }$ , and $\rho = \| \delta \| _ { 2 }$ . For $\| \delta \| _ { \infty } > r _ { i }$ , the scale factor is

$$
s ( \mathbf { x } ) = s ( m , \rho ) = \frac { m ^ { 2 } r _ { c } - r _ { i } ^ { 2 } \rho } { m \ ( m ( r _ { c } + \rho ) - 2 r _ { i } \rho ) } \ .\tag{5}
$$

The near-field geometry is represented without distortion, while distant scene content is compressed into a finite domain and can still be indexed by the same sparse voxels. Voxels fall into four classes relative to the identity box and the outer radius $r _ { c } .$ inner voxels lie entirely inside the identity box, outer voxels entirely outside it, mixed voxels straddle its boundary, and outside-scene voxels fall beyond $r _ { c } .$ . Fig. 5 (b) shows these four classes. The classification determines how a ray leaves a voxel during marching (Sec. IV-A).

Algorithm 1: Sparse Voxel Query   
Input: Query point q, sorted occupied octpaths   
$\bar { \{ p _ { i } \} } _ { i = 0 } ^ { \bar { M } - 1 }$ , and their levels $\dot { \{ l _ { i } \} } _ { i = 0 } ^ { M - 1 }$   
Output: Voxel level �, octpath �, and occupancy flag   
$\delta \in \{ 0 , 1 \}$   
1 Let $[ I ^ { - } ( p , l ) , I ^ { + } ( p , l ) )$ denote the finest-level Morton   
interval covered by voxel $( p , l ) ;$   
2 � ← MortonEncode(q, �);   
3 Binary search the predecessor $( p ^ { - } , l ^ { - } )$ and successor   
$( p ^ { + } , l ^ { + } )$ of $q ;$   
4 if $\boldsymbol { p } ^ { + } = \boldsymbol { q }$ then   
5 return $( l ^ { + } , p ^ { + } , 1 ) ;$ // exact leaf hit   
6 end   
7 if $I ^ { - } ( p ^ { - } , l ^ { - } ) \leq q < I ^ { + } ( p ^ { - } , l ^ { - } )$ then   
8 return $( l ^ { - } , p ^ { - } , 1 ) ;$ // the predecessor   
leaf covers the query   
9 end   
10 for $\ell = 0$ to � do   
11 � ← MortonEncode $( \mathbf { q } , \ell ) ;$   
12 if $I ^ { + } ( p ^ { - } , l ^ { - } ) \leq I ^ { - } ( p , \ell )$ and $I ^ { + } ( p , \ell ) \leq I ^ { - } ( p ^ { + } , l ^ { + } )$   
then   
13 return $( \ell , p , 0 ) ;$ $/ /$ largest empty   
voxel containing <sub>�</sub>   
14 end   
15 end

## C. Octree Query

LiDAR supervision and camera optimization query the field along independent rays in arbitrary directions. Most of a ray crosses free space where no voxel is materialized, so empty space must be skipped in large steps. SpaRV therefore queries the field by an octree lookup, and rendering marches over the resulting voxel segments (Sec. IV). Since only occupied leaves are stored in a flat Morton-indexed array, the renderer does not perform a pointer-based hierarchical octree walk. Instead, Alg. 1 maps each query point to its finest-level Morton code and locates it in the sorted leaf set by binary search (lines 2– 3). If the queried code matches a stored leaf, or falls inside the finest-level Morton interval covered by the predecessor leaf, the ray enters that occupied voxel directly (lines 5 and 8). Otherwise, the predecessor and successor occupied leaves bound the query in Morton order, and the largest empty voxel between them is skipped (line 13). The marcher can therefore skip free space without traversing unoccupied fine voxels individually. The lookup thus returns either an occupied voxel to be evaluated or an inferred empty skip voxel that advances the ray to the next relevant event. The event-driven ray marcher of Sec. IV runs entirely on this occupied-or-empty interface.

## IV. SPARSE RADIANCE VOXEL RENDERING

SpaRV renders the sparse voxels with a differentiable ray marcher that traverses them along each ray independently.

Camera pixels, LiDAR rays, and rays of non-pinhole cameras then supervise the map through the same renderer. In contrast, the rasterizer of SVRaster [16] batches a pinhole pixel lattice and does not directly apply to LiDAR rays or non-pinhole cameras (Fig. 6). Rendering converts the occupied-or-empty lookup of Sec. III-C into pixel values through event-driven ray marching and volumetric alpha blending.

![](images/e21069e27bd381c1c1a106f11d553d2c54571563b84bd7a76415abb36d85e231.jpg)  
Fig. 6. Rasterization versus ray marching. (a) Rasterization batches a pinhole pixel lattice. (b) Ray marching evaluates each ray independently.

## A. Ray Marching Rendering

Ray marching proceeds by advancing from one returned voxel event to the next, as summarized in Alg. 2. At each step, the current world-ray parameter � determines the query point $\mathbf { x } ( t ) = \mathrm { C o n t r } ( \mathbf { o } + t \mathbf { d } )$ , and the lookup returns either an occupied voxel to be rendered or an empty skip voxel to be traversed (line 5). In either case, the next step requires the exit parameter $t _ { \mathrm { e x i t } }$ at which the ray leaves the voxel. How it is found depends on the voxel’s class relative to the identity box and the contracted domain (line 6): inner, outer, mixed, or outside-scene, as drawn in Fig. 5 (b). For inner voxels, the exit is a ray–box intersection (line 11). In outer and mixed voxels, the straight world ray maps to a curve in contracted space, so the exit has no closed-form ray–box solution. For $m > r _ { i }$ , the Jacobian of Eq. (4) is

$$
\mathbf { J } ( \mathbf { x } ) = s ( m , \rho ) \mathbf { I } + ( \mathbf { x } - \mathbf { w } _ { \mathrm { c } } ) \otimes { \boldsymbol { \nabla } } s ( \mathbf { x } ) \ ,\tag{6}
$$

where ∇� is the gradient of the scale factor in Eq. (5). The Jacobian maps the world direction d to a local contracted direction. The curved-exit solve (line 13) uses a slab test along Jd to propose a candidate and evaluates Contr exactly to accept or reject it. Once an occupied segment is identified, SpaRV divides the secant between its mapped entry and exit points into equal steps and evaluates density and appearance at each step’s midpoint (line 16). These samples are then alphablended front to back (Sec. IV-B), which accumulates the color and depth of the ray and attenuates its transmittance (line 17). B. Sparse-Voxel Alpha Blending

For a point p inside a voxel, the density is obtained by trilinear interpolation interp( $\mathbf { \sigma } _ { \mathbf { y } \in \mathbf { o } } , \mathbf { p } )$ (Eq. (2)), followed by an exponential-linear activation [15], [52] to ensure nonnegativity. Each ray–voxel intersection segment of Sec. IV-A is discretized into � midpoint samples: in the �-th intersected voxel, the �-th sample is located at $\mathbf { x } _ { i , k }$ with spacing $\Delta t _ { i } ,$ , the segment length divided by �. The sample opacity is computed as

$$
\alpha _ { i , k } = 1 - \exp \left( - \Delta t _ { i } \exp \mathrm { l i n } \big ( \mathrm { i n t e r p } ( \mathbf { v } _ { \mathrm { g e o } _ { i } } , \mathbf { x } _ { i , k } ) \big ) \right) .\tag{7}
$$

Algorithm 2: Ray Marching Rendering   
Input: Ray origin o, direction d, sparse voxels.   
Output: Color C, depth �, transmittance �.   
1 Let $\mathbf { x } ( t ) = \mathbf { C o n t r } ( \mathbf { o } + t \mathbf { d } )$ and let $B ( l , p )$ be the   
contracted-space box of voxel $( l , p ) ;$   
2 $t \gets t _ { \mathrm { n e a r } } ,$ the entry of the world ray into the scene   
bounds;   
3 $\mathbf { C } \gets \mathbf { 0 } , D \gets 0 , T \gets 1 ;$   
4 while $T > T _ { \mathrm { m i n } }$ do   
5 (�, �, �) ← OctreeQuery(x(�)); // Alg. 1   
6 � ← VoxelClass(�, �);   
7 if � = outside then   
8 return $\mathbf { C } , D , T ;$   
9 end   
10 if $\kappa =$ inner then   
11 $t _ { \mathrm { e x i t } }$ ← RayBoxExit �, �(�, �);   
12 else   
13 �<sub>exit</sub> ← CurvedExit �, �(�, �), �;   
14 end   
15 if $\delta = 1$ then // occupied voxel   
16 $\begin{array} { r } { \mathbf { x } _ { k } \gets \mathbf { x } ( t ) + \frac { k - 1 / 2 } { K } \big ( \mathbf { x } ( t _ { \mathrm { e x i t } } ) - \mathbf { x } ( t ) \big ) , \ k = 1 , \ldots , K ; } \end{array}$   
17 (C, �, �) ← AlphaBlending {x� }, �, �;   
18 end   
19 $t \gets t _ { \mathrm { e x i t } } + \epsilon ( t _ { \mathrm { e x i t } } )$ ; // cross the boundary   
20 end   
21 return $\mathbf { C } , D , T ;$

The view-dependent color $\mathbf { c } _ { i , k }$ is evaluated at the same sample using the query-ray direction relative to the voxel. The DC component $\mathrm { { S H } _ { 0 } }$ is trilinearly interpolated from the vertex-stored coefficients at $\mathbf { x } _ { i , k }$ , while the higher-order SH coefficients remain shared by the voxel. Spherical-harmonic evaluation then combines the two into $\mathbf { c } _ { i , k }$ . The low-frequency color therefore varies continuously within each voxel, and view-dependent shading stays compact.

For a ray passing through � intersected voxels, the rendered pixel color C is accumulated over the voxel-sample sequence encountered along the ray:

$$
\mathbf { C } = \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } T _ { i , k } \alpha _ { i , k } \mathbf { c } _ { i , k } \ ,\tag{8}
$$

$$
T _ { i , k } = \prod _ { ( j , l ) \prec ( i , k ) } \left( 1 - \alpha _ { j , l } \right) ,\tag{9}
$$

where ≺ denotes the ray-marching sampling order. Unlike autograd implementations such as NeRF [5], which store every sample for back-propagation, SpaRV differentiates Eq. (8) in a custom CUDA kernel whose memory grows with the number of rays rather than samples.

## V. GEOMETRY-GROUNDED MAP TRAINING

This section trains the sparse radiance voxel map from RGB images, LiDAR points, and the initial camera parameters, and adapts its voxels to the scene during training (Sec. V-A). The goal is a map that renders the captured views photorealistically and keeps correct geometry in regions that the images alone under-constrain, so that it stays geometrically consistent beyond the captured trajectory. To this end, photometric supervision fits the images, LiDAR depth supervision and occupied voxel guidance anchor the density field at the measured surfaces, and normal and smoothness regularizers keep the surface smooth between LiDAR points (Sec. V-B).

## A. Voxel Initialization and Adaptation

SpaRV inherits the voxel initialization and adaptation scheme of SVRaster [16]. Initialization uses the same foreground-background layout, with voxel maximum sampling rate as the guiding quantity for image-space coverage. The implementation starts from uniform dense voxels and iteratively subdivides the voxels with the highest sampling rate until the initial target voxel budget is reached. Voxels with zero camera visibility are then removed. This initialization provides a coarse scene-covering layout that is subsequently refined by adaptive subdivision and pruning during training.

During training, we adopt the same adaptive structural update criteria. At each adaptation, voxels whose maximum blending weight $T _ { i , k } \alpha _ { i , k }$ in Eq. (8) across training views falls below a threshold are pruned. Voxels are thus kept only where the rendering needs them, which makes the map sparse, as shown in Fig. 4 (a). For subdivision, each leaf voxel is ranked by the accumulated alpha-weighted loss gradient

$$
\nu _ { \mathrm { p r i o r i t y } } = \sum _ { \mathbf { r } \in R } \left\| \alpha ( \mathbf { r } ) \frac { \partial \mathcal { L } ( \mathbf { r } ) } { \partial \alpha ( \mathbf { r } ) } \right\| \ ,\tag{10}
$$

where � denotes the training rays collected over one adaptation interval and $\mathcal { L } ( \mathbf { r } )$ is the per-ray training loss defined in Sec. V-B. The priority selects which voxels are subdivided, and the maximum sampling rate suppresses subdivision in poorly sampled regions. Finer voxels therefore concentrate where detail is required, as shown in Fig. 4 (b).

## B. Radiance Field Training

We train the sparse voxel radiance field using the standard rendering losses and render-normal consistency regularization [16], together with LiDAR depth supervision, occupied voxel guidance, and structural smoothness regularization. The map-training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { r g b } } + \lambda _ { \mathrm { n } } \mathcal { L } _ { \mathrm { n } } + \lambda _ { \mathrm { d } } \mathcal { L } _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { o c c } } \mathcal { L } _ { \mathrm { o c c } } + \lambda _ { \mathrm { s } } \mathcal { L } _ { \mathrm { s m o o t h } } \ , } \end{array}\tag{11}
$$

where $\mathcal { L } _ { \mathrm { r g b } } , \ \mathcal { L } _ { \mathrm { n } } , \ \mathcal { L } _ { \mathrm { d e p t h } } , \ \mathcal { L } _ { \mathrm { o c c } } ,$ and $\mathcal { L } _ { \mathrm { { s m o o t h } } }$ denote color reconstruction, normal consistency, LiDAR depth supervision, occupied voxel guidance, and density-field smoothness, respectively. They constrain the field in complementary regions, as summarized in Fig. 7. Color supervision fits every camera ray but under-constrains geometry, since different density profiles can render the same pixel color (Fig. 7 (a)). LiDAR depth supervision anchors the field metrically, but only along measured rays (Fig. 7 (b)). Between the anchors, where no range measurement is available, the normal and smoothness regularizers keep the surface locally smooth (Fig. 7 (c)). Occupied voxel guidance suppresses floaters that depth supervision leaves off the measured rays (Fig. 8).

1) Photometric Supervision: For each training camera, SpaRV synthesizes an RGB image C from the current sparsevoxel field using the camera pose, and compares it with the observed color C<sup>∗</sup>. The color loss combines a pixel-wise reconstruction term with SSIM:

![](images/167d3af069617b9f2e306eec0cd024af04348a694fdacd63e9a985c85b51d1e5.jpg)  
Fig. 7. Loss terms of Eq. (11) and the regions they constrain. (a) $\mathcal { L } _ { \mathrm { r g b } } \mathrm { : a }$ floater (dashed) and a surface-aligned density profile (solid) render the same pixel color. (b) $\mathcal { L } _ { \mathrm { d e p t h } } \mathrm { : }$ the target $q _ { r } ( d )$ (rose) pulls the rendered depth distribution �<sub>�</sub> (�) (blue) onto the measured range $d _ { r } ^ { * } . \ ( { \mathrm { c } } ) \ { \mathcal { L } } _ { \mathrm { n } }$ and $\mathcal { L } _ { \mathrm { { s m o o t h } } } \colon$ between the LiDAR anchors, a rippling iso-surface (blue, dashed) fits the images as well as a smooth one, and the regularizers smooth it (rose).

$$
\mathcal { L } _ { \mathrm { r g b } } = \left( 1 - \lambda _ { \mathrm { s s i m } } \right) \sum _ { \mathrm { p i x e l s } } \| \mathbf { C } - \mathbf { C } ^ { * } \| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { s s i m } } \mathcal { L } _ { \mathrm { S S I M } } \ ,\tag{12}
$$

where $\lambda _ { \mathrm { s s i m } }$ controls the SSIM contribution.

2) LiDAR Depth Supervision: Photometry constrains depth only through the parallax between views, and a short baseline gives little parallax. Where the captured trajectory observes a region from few and uneven viewpoints, the color loss alone leaves its geometry under-constrained. SpaRV therefore uses LiDAR range measurements as metric anchors along their rays. Because these rays are sparse and irregular, SpaRV evaluates them with the event-driven ray marcher of Sec. IV-A rather than a rasterization-only camera renderer, as illustrated in Fig. 6. The rendered depth along a ray is:

$$
D = \sum _ { i = 1 } ^ { N } T _ { i } \alpha _ { i } d _ { i } \ ,\tag{13}
$$

where $d _ { i }$ is the depth of the �-th sample along the ray.

LiDAR depth supervision treats the volume-rendering weights $\begin{array} { r c l } { w _ { i } } & { = } & { T _ { i } \alpha _ { i } } \end{array}$ as a discrete depth distribution and minimizes a KL-divergence loss [34] to a unimodal target distribution centered at the measured LiDAR range:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { d e p t h } } = \displaystyle \sum _ { r \in \mathcal { R } _ { \mathrm { L i D A R } } } D _ { \mathrm { K L } } ( q _ { r } ( d ) \| p _ { r } ( d ) ) } \\ & { \equiv - \displaystyle \sum _ { r \in \mathcal { R } _ { \mathrm { L i D A R } } } \sum _ { i } q _ { r } ( d _ { i } ) \log ( w _ { i } ) ~ , } \end{array}\tag{14}
$$

where $p _ { r } ( d _ { i } ) = w _ { i }$ is the rendered depth distribution along LiDAR ray $r ,$ and $q _ { r } ( d _ { i } ) \propto \Delta t _ { i } \exp \left[ - ( d _ { i } - d _ { r } ^ { * } ) ^ { 2 } / ( 2 \nu _ { \mathrm { s } _ { i } } ^ { ~ 2 } ) \right]$ is the target distribution centered at the observed range $d _ { r } ^ { * }$ , with $\Delta t _ { i }$ the sample spacing and $\nu _ { \mathrm { s } i }$ the size of the voxel containing sample �. The target therefore sharpens as the map refines.

3) Structural Regularization: Depth supervision constrains the field only along measured LiDAR rays. Between those anchors, the density can still ripple or drift while reproducing the training images, as shown in Fig. 7 (c). The normalconsistency loss ${ \mathcal { L } } _ { \mathrm { n } }$ [32] penalizes the inconsistency between the rendered density-gradient normal and the normals estimated from the rendered depth maps, including both the expected-depth normal $\mathbf { N } _ { \mathrm { m e a n } }$ and the median-depth normal $\mathbf { N } _ { \mathrm { m e d } } { : }$

$$
\mathcal { L } _ { \mathrm { n } } = \Vert \mathbf { N } _ { \sigma } - \mathbf { N } _ { \mathrm { m e a n } } \Vert _ { 2 } ^ { 2 } + \Vert \mathbf { N } _ { \sigma } - \mathbf { N } _ { \mathrm { m e d } } \Vert _ { 2 } ^ { 2 } \ ,\tag{15}
$$

where ${ \bf N } _ { \sigma }$ is the rendered density-gradient normal.

![](images/65d49be0823274a84ca2081a6016263c4f191f47a2804a722606af5e66d1a2ec.jpg)  
Fig. 8. Occupied voxel guidance on the sparse voxel map. Teal dots represent input LiDAR points. (a) LiDAR depth supervision leaves floaters off the measured rays unconstrained. (b) Guiding camera rays to terminate in the first occupied voxel suppresses floaters in front of it. (c) Voxel adaptation removes floater voxels whose blending weights fall below the pruning threshold.

![](images/ef12001f291d988806586a7ff7750afbbd40faa77fea969b37a9000ed6cca87e.jpg)

![](images/081542349a93893aedb69629cac1a96324487bba8f200fac992b6ded7490ae5b.jpg)  
(a) w/o L<sub>occ</sub>  
(b) w/ L<sub>occ</sub>  
Fig. 9. Effect of the occupied voxel guidance on an Oxford Spires scene: the same view from a map trained (a) without and (b) with $\mathcal { L } _ { \mathrm { o c c } }$ of Eq. (17), as color, median depth, and the normal map derived from that depth.

To further smooth the local geometry, we apply a geometryaware smoothness regularization $\mathcal { L } _ { \mathrm { { s m o o t h } } }$ across face-adjacent voxels. For a neighboring voxel pair $( \alpha , \beta )$ , it penalizes the difference between their local density-gradient normals:

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \sum _ { \left( \alpha , \beta \right) \in N } \omega _ { \alpha \beta } \left\| \nabla \sigma _ { \alpha } - \nabla \sigma _ { \beta } \right\| _ { 2 } ^ { 2 } ,\tag{16}
$$

where N denotes the set of face-neighbor voxel pairs, $\nabla \sigma _ { \alpha }$ and $\nabla \sigma _ { \beta }$ denote the normalized density gradients of the two neighboring voxels, and $\omega _ { \alpha \beta }$ is a geometry-aware edge weight determined by the shared face area and the distance between voxel centers.

4) Occupied Voxel Guidance: A LiDAR scan is sparse, and its field of view does not coincide with the camera’s. LiDAR depth supervision therefore acts only along measured rays, and free-space floaters off those rays cannot be supervised by Eq. (14) (Fig. 8 (a)). The point cloud still marks which voxels contain measured surface. Occupied voxel guidance converts this occupancy into a termination target for camera rays that intersect an occupied voxel.

Let $\boldsymbol { \mathcal { N } } _ { \mathrm { o c c } }$ denote the set of current sparse voxels that contain at least one input LiDAR point. For each camera ray that intersects this set, let $\mathbf { v } _ { r } ^ { \star }$ denote the first occupied voxel along the ray. The guidance encourages the ray to terminate inside $\mathbf { v } _ { r } ^ { \star }$ by maximizing the sum of its blending weights $w _ { i } = T _ { i } \alpha _ { i } \mathrm { : }$

![](images/16906386e601615e7216099d83803c160b599dcbbc99653e22e4158edd6e424b.jpg)  
Fig. 10. Rendering results with the odometry and the optimized cameras (Sec. VI) on training views of FAST-LIVO2 Drive (top) and Sculpture (bottom).

$$
\mathcal { L } _ { \mathrm { { o c c } } } = - \frac { 1 } { | \mathcal { R } _ { \mathrm { { o c c } } } | } \sum _ { r \in \mathcal { R } _ { \mathrm { o c c } } } \log \left( \sum _ { i \in \mathbf { v } _ { r } ^ { \star } } w _ { i } + \epsilon \right) ,\tag{17}
$$

where $\mathcal { R } _ { \mathrm { o c c } }$ is the set of camera rays that intersect an occupied voxel and � is a small constant for numerical stability.

As illustrated in Fig. 8 (b), the guidance increases the accumulated blending weight in the first occupied voxel. It encourages ray termination near the LiDAR-observed surface and suppresses the weights of floaters in front of that surface. As the rendered surface converges toward the observed surface, floater voxels whose maximum blending weight across training views falls below the pruning threshold are removed by voxel adaptation (Sec. V-A; Fig. 8 (c)). Selecting only the first occupied voxel focuses the guidance on the foreground surface along each ray. Occupied voxels behind it are not additional termination targets and can remain occluded when photometric supervision has already recovered the foreground surface. The guidance applies only to occupied voxels that a camera ray actually reaches. When photometric supervision recovers an opaque surface in front of an occupied voxel, the ray terminates before that voxel (Alg. 2), and the guidance on that ray lapses. LiDAR occupancy and the images thus compete for where each ray terminates, and a surface recovered from the images survives as long as it explains them. The guidance is also applied only in the early stage of training, while the geometry is still forming. Fig. 9 shows the resulting difference on an Oxford Spires scene: trained without the term, the map reproduces the view in color yet leaves floaters in regions observed only by cameras; with it, the same view renders a clean structure.

## VI. CAMERA OPTIMIZATION

This section optimizes the camera poses and focal lengths against the map. The initial camera parameters are metrically plausible but carry residual errors, which blur the textures of the map. The map under training predicts the image each camera should see, and the photometric residual against the captured image exposes these errors. The trilinear density and the vertex-interpolated DC appearance of Sec. III-A give this residual dense gradients with respect to the camera parameters. SpaRV alternates this camera optimization with map training to sharpen the textures of the map.

## A. Camera Parameterization

Camera poses are parameterized as rigid transformations $\mathbf { T } _ { c } ^ { w } \in S E ( 3 )$ from camera to world coordinates:

$$
\begin{array} { r } { \mathbf { T } _ { c } ^ { w } = \left[ \begin{array} { c c } { \mathbf { R } } & { \mathbf { t } } \\ { \mathbf { 0 } } & { 1 } \end{array} \right] \ , \ } \end{array}\tag{18}
$$

where $\textbf { R } \in \ S O ( 3 )$ is the rotation matrix and $\textbf { t } \in \mathbb { R } ^ { 3 }$ is the translation vector. Since the initial poses are already metrically plausible, SpaRV optimizes an error-state correction $\xi ~ = ~ [ \omega , \delta \mathbf { t } ] ~ \in ~ \mathbb { R } ^ { 6 }$ in the tangent space of ��(3), where $\begin{array} { r } { \begin{array} { r l } { \omega } & { { } \in \enspace s o \left( 3 \right) } \end{array} } \end{array}$ is the rotation perturbation and �t $\epsilon \mathbb { R } ^ { 3 }$ is the translation perturbation [48], [53]. The optimized pose is written as

$$
\mathbf { T } _ { c } ^ { w } = \exp \left( \pmb { \xi } ^ { \wedge } \right) \mathbf { T } _ { c } ^ { w , ( 0 ) } \ ,\tag{19}
$$

where $\mathbf { T } _ { c } ^ { w , ( 0 ) }$ denotes the odometry initialization, which stays fixed across rounds. Intrinsic optimization is illustrated with a shared pinhole model for each intrinsic group; other camera models replace only the pixel-to-ray map:

$$
\mathbf { K } = \left[ { \begin{array} { c c c } { f _ { x } } & { 0 } & { c _ { x } } \\ { 0 } & { f _ { y } } & { c _ { y } } \\ { 0 } & { 0 } & { 1 } \end{array} } \right] \ .\tag{20}
$$

Within each group, the principal point $( c _ { x } , c _ { y } )$ remains fixed at its initialized value, while the focal lengths are represented by the vector $\mathbf { f } = [ f _ { x } , f _ { y } ] ^ { \top }$ and optimized jointly with the camera poses. For numerical stability, SpaRV optimizes a log-space focal error state $\pmb { \phi } _ { f } \in \mathbb { R } ^ { 2 }$ relative to the initialization $\mathbf { f } ^ { ( 0 ) }$ such that

$$
\log { \bf f } = \log { \bf f } ^ { ( 0 ) } + \pmb { \phi } _ { f } \ .\tag{21}
$$

For a pinhole pixel $( u , \nu )$ , the camera-space ray direction is $\mathbf { r } _ { \mathrm { c a m } } \propto \mathbf { K } ^ { - 1 } [ u , \nu , 1 ] ^ { \top }$ . The loss is backpropagated through the rendered rays to both the pose and focal error states.

## B. Photometric Camera Optimization

Each camera-optimization step updates the camera poses and focal lengths by minimizing a direct photometric objective between the rendered image and the observed RGB image

through differentiable rendering. Given the rendered and observed images C and $\mathbf { C } ^ { * }$ , the camera-optimization objective is defined as

$$
\mathcal { L } _ { \mathrm { c a m } } = \mathcal { L } _ { \mathrm { r g b } } \left( \mathbf { C } , \mathbf { C } ^ { * } \right) + \lambda _ { \xi } \rho _ { \xi } ( \pmb { \xi } ) + \lambda _ { f } \rho _ { f } \left( \pmb { \phi } _ { f } \right) ,\tag{22}
$$

where $\mathcal { L } _ { \mathrm { r g b } } ( \cdot , \cdot )$ denotes the color loss defined in Eq. (12). Robust kernels regularize both the pose correction and the shared focal update around their initialized values [54]. The vertex-interpolated DC SH component of Sec. III-A varies continuously within each voxel and across faces shared by voxels, so the photometric gradient of this objective does not vanish inside the voxels.

## C. Alternating Optimization Schedule

SpaRV alternates between updating the map parameters and updating the camera parameters. During map updates, the camera parameters are held fixed and the voxel field is trained with Eq. (11). Camera optimization runs on the same fixed period as voxel adaptation. At each camera-optimization round, SpaRV freezes the map and collects the cameras that will be sampled within the next voxel-adaptation interval. SpaRV then optimizes this look-ahead set for a small number of steps. The updated camera parameters are rolled back if they do not reduce the color loss $\mathcal { L } _ { \mathrm { r g b } } ( \cdot , \cdot )$ . The optimized camera parameters are then reused in the subsequent map-training iterations. As shown in Fig. 10, alternating optimization improves rendering quality as the camera parameters are updated.

## VII. LARGE-SCALE RECONSTRUCTION

The GPU memory required by an explicit voxel map grows with both the extent of the scene and the requested level of detail. Consequently, the map of a large scene fits within a fixed GPU memory budget only at a coarser level of detail. A voxel’s projected size decreases with its distance from the camera, so a view needs fine voxels only near the camera. SpaRV therefore keeps the full octree in host memory and places only a selection of its voxels, called a granularity-driven cut, on the GPU. The cut selects voxels by their projected size: fine voxels inside a box of interest, and progressively coarser voxels farther away. Throughout this section, $N _ { \mathrm { m a x } }$ denotes the maximum number of voxels resident on the GPU at one time. During training, voxel subdivision stops when the number of resident voxels reaches this budget. During deployment, the cut is loaded subject to the same budget. GPU memory then stays bounded as the scene grows.

Fig. 11 summarizes the pipeline. Coarse training first builds the coarse map of the whole scene and initializes the octree. Submap training then visits the disjoint submaps one at a time and refines each within its own cut. The box of each cut bounds the submap in focus, so the cut holds that submap at the finest available level and the rest of the scene at coarser levels. Deployment applies the same cut rule to the refined octree, with the box placed around the moving camera. We first describe the octree and the cut, then detail their roles in training and rendering.

## A. Level-of-Detail Data Structure

1) Structure and Growth: The octree O stores both the current leaves and the coarser internal nodes created during subdivision (Fig. 11 (1)), so every level that training passes through stays available to the cut. It uses the flat (octpath, octlevel) indexing of Sec. III-A. When a voxel subdivides according to Sec. V-A, its trained parameters are frozen and retained as the parent node, while its eight children become trainable leaves. The parent is not trained further because a voxel is split only after it has been sufficiently trained at its current level. Its parameters therefore remain a valid approximation at that coarser level. Training continues on the children, which add finer detail while the parent stays a snapshot of the preceding level. Fig. 12 (a)–(d) illustrates the detail represented at different levels of the same octree.

2) Granularity-Driven Cut: A cut selects voxels from different levels of the octree to cover the whole scene: full detail inside a box $B ,$ and coarser voxels outside it. It is defined by the box � and by a granularity target � in pixels [27]. The cut should select as few voxels as possible, while no selected voxel projects to more than � pixels in any camera inside � unless the octree has no finer voxel there. To measure how large a voxel node � appears from �, we use the projected-size proxy

$$
r ( n , B ) = f \frac { \nu _ { \mathrm { s } } ( n ) } { d ( n , B ) } ,\tag{23}
$$

where $f = \operatorname* { m a x } ( f _ { x } , f _ { y } )$ is the focal length of a pinhole camera, $\nu _ { \mathrm { s } } ( n )$ is the voxel size, and $d ( n , B )$ is the minimum Euclidean distance between voxel � and box �. When � collapses to the camera’s focal point, $r ( n , B )$ is the number of pixels the voxel spans on the image plane (Fig. 13 (a)). For a larger box, $d ( n , B )$ never exceeds the distance from � to any camera inside $B ,$ so $r ( n , B )$ bounds the projected size of � in every camera inside �. Coarser nodes lie closer to the root, so along each root-to-leaf path the first node with $r ( n , B ) \leq \tau$ is the coarsest node that is fine enough. The cut takes that node, or the leaf when no node on the path is fine enough:

$$
\begin{array} { c } { { C ( \tau , B , O ) = \left\{ n \in O : \left[ r ( \pi ( n ) , B ) > \tau \right] \right. } } \\ { { \left. \qquad \wedge \left[ r ( n , B ) \leq \tau \vee n \mathrm { ~ i s ~ a ~ l e a f } \right] \right\} , } } \end{array}\tag{24}
$$

where $\pi ( n )$ denotes the parent of �, with $r ( \pi ( n ) , B ) ~ = ~ \infty$ for the root. The cut thus takes exactly one node along each root-to-leaf path. Since a parent covers the same space as its children together, the selected voxels cover the scene without overlaps or holes and form a sparse radiance voxel map that the renderer uses directly. Inside �, $d ( n , B ) = 0$ and $r ( n , B ) = \infty ,$ so the cut descends to the leaves and keeps the finest available voxels, while outside � the selected voxels grow coarser with distance (Fig. 13 (b)). Increasing � reduces the number of selected voxels at the cost of coarser detail (Fig. 12 (e)– (h)), whereas enlarging � keeps a larger region at the finest available level and increases the number of selected voxels.

## B. Coarse-to-Fine Training

1) Coarse Training: Coarse training applies the training of Sec. V to the whole scene under the budget $N _ { \mathrm { m a x } } { \mathrm { : } }$ voxels subdivide to capture detail until the leaf count reaches $N _ { \mathrm { m a x } } .$ This yields the coarse map covering the whole scene, with optimized cameras and the initial octree $O _ { 0 }$ (Fig. 11 (1)).

![](images/afb7fc5dcd8975865bbe94e8b5b3e20c45ff638e31b258e638c477e905e9db4d.jpg)  
Fig. 11. Large-scale training and deployment of SpaRV. (1) Coarse training (Sec. VII-B1) subdivides the octree over the whole scene up to the memory budget and keeps each coarser level as frozen nodes. (2) Submap training (Sec. VII-B2) trains one submap $B _ { k }$ at a time, with its relevant views, on the train cut $C ( \tau _ { t } , B _ { k } , O _ { 0 } )$ : the interior (blue) is subdivided (dashed leaves), the exterior (violet) stays frozen, and grey nodes are not in the cut. (3) Deployment (Sec. VII-C) renders from the deploy cut $C ( \tau _ { d } , B _ { d } , O _ { d } )$ of the refined octree around the deploy box $B _ { d }$ that contains the camera.

![](images/35a43d44f51ad44af4be6adcc32ccb9ced8cba7859a5ce9699b7abadb77f20e6.jpg)  
(e) � = 1  
(f) � = 5  
(g) � = 9  
(h) � = 17

Fig. 12. Levels of detail of one octree, seen from a single viewpoint. (a)– (d) Increasingly fine levels. (e)–(h) Cuts at increasing �.  
![](images/7d1aea2b4046206ae98dfa94815f783be5985795075919038ef3ac5b40fa1987.jpg)

![](images/7faf03f0b793ba36e70adf4ffab8f010985296650d08b9531d3af3a00cc7ffc5.jpg)  
Fig. 13. Granularity-driven cut selection. (a) With the box � collapsed to the camera’s focal point, � (�, �) of Eq. (23) is the number of pixels node � spans on the image plane. (b) The cut of Eq. (24) against the distance $d ( n , B )$ : on each root-to-leaf path, it takes the coarsest node that projects to at most � pixels. Here the parent � (�) projects larger than � and is rejected (dashed, crossed), and its child � is taken. Inside �, where $d ( n , B ) = 0 ,$ the cut keeps the leaves.

2) Submap Training: As shown in Fig. 11 (2), submap training first partitions the scene into non-overlapping axisaligned boxes. The target submap size $s _ { t }$ determines the octree level $l _ { t } = \left\lceil \log _ { 2 } ( w _ { \mathrm { s } } / s _ { t } ) \right\rceil$ . This level defines the box $B _ { k }$ of each submap, with edge length $2 ^ { - l _ { t } } w _ { \mathrm { s } }$ . The �-th submap owns the subtrees of $O _ { 0 }$ inside $B _ { k } ,$ , and submaps refine disjoint portions of the octree. Each submap’s training views are selected with the coarse map: a view is assigned to a submap if its rays reach voxels inside $B _ { k }$

Each submap is initialized from its cut $C _ { k } = C ( \tau _ { t } , B _ { k } , O _ { 0 } )$ with granularity target $\tau _ { t }$ . The cut is partitioned into an interior ${ \mathcal { I } } _ { k } ~ = ~ \{ n ~ \in ~ C _ { k } ~ : ~ n ~ \subseteq ~ B _ { k } \}$ , the leaves inside $B _ { k }$ , and an exterior ${ \mathcal { E } } _ { k } ~ = ~ { \cal C } _ { k } ~ \backslash ~ { \bar { \mathcal { I } } } _ { k }$ , the remaining voxels of the cut. Only the interior is trained, following Sec. V: subdivision adds finer leaves inside $B _ { k }$ and appends their trained parents to the octree (Sec. VII-A1). The exterior stays frozen but participates in rendering, so the training views are rendered with the surrounding scene in place and the interior does not have to explain content outside $B _ { k }$ . Both parts count against the budget $N _ { \mathrm { m a x } }$ . After all submaps are trained in turn, the octree already holds their refined subtrees and serves directly as the octree $O _ { d }$ for deployment.

## C. Deployment

Deployment applies the same cut rule around the camera, so rendering preserves local detail without making all fineresolution voxels of the entire scene resident on the GPU at once. It selects the cut from the refined octree $O _ { d }$ , using a granularity target $\tau _ { d }$ and a deploy box $B _ { d }$ of target size $s _ { d } \colon$ $C _ { d } = C ( \tau _ { d } , B _ { d } , O _ { d } )$ . The deploy box contains the camera and aligns with the octree partition at the level determined by $s _ { d } ,$ following the same sizing rule as in training (Fig. 11 (3)). The deploy box is smaller than a training box $( s _ { d } ~ < ~ s _ { t } )$ so the finest voxels cover a smaller region and more of the memory budget remains for the exterior. The granularity target $\tau _ { d }$ sets how coarse the exterior is and is increased when the cut exceeds $N _ { \mathrm { m a x } }$ . The cut is recomputed only when the camera moves beyond the current deploy box by a small margin, preventing repeated switching near the box boundary. The resident voxels are then updated by incremental host-device transfer. Training and rendering thus share one octree and one cut rule, and both keep the resident voxels within $N _ { \mathrm { m a x } }$ as the scene grows.

## VIII. EXPERIMENTS

This section tests the four contributions of the paper on five datasets. Radiance-field methods fit their training views closely, so differences in geometry appear mainly at viewpoints away from the captured trajectory. To evaluate geometric consistency, we compare the methods at interpolated viewpoints along the trajectory and at extrapolated viewpoints away from it, on the synthetic Replica dataset (Sec. VIII-B) and the real FAST-LIVO2 dataset (Sec. VIII-C). Replica provides ground-truth poses, so its extrapolated views are evaluated against ground truth, while those of FAST-LIVO2 are compared qualitatively. Since the cameras of FAST-LIVO2 come from odometry and carry residual errors, we also use this dataset to evaluate camera optimization (Sec. VIII-C1). We further compare the methods with accurate cameras on the

Oxford Spires benchmark (Sec. VIII-D), and ablate occupied voxel guidance on it. To evaluate large-scale reconstruction under bounded GPU memory, we use two kilometer-scale scenes that we captured, Garden and Campus (Secs. VIII-E–VIII-F). To evaluate the ray-marching renderer beyond pinhole cameras, we train SpaRV on the EgoNeRF dataset (Sec. VIII-G), where an omnidirectional camera supervises the map through the same renderer. Finally, Sec. VIII-H uses the trained maps for surface extraction and, through the same renderer, for the simulation of other camera and LiDAR models.

## A. Experimental Setup

1) Baselines and Metrics: We compare SpaRV against the following methods: InstantNGP [13]: hash-grid NeRF; 3DGS [6]: image-only 3D Gaussian splatting; SVRaster [16]: sparse-voxel radiance field whose representation SpaRV shares; LetsGo [9]: 3DGS initialized from a LiDAR Poisson mesh and regularized by LiDAR depth; GS-SDF [10]: Gaussians coupled to a LiDAR-built neural SDF; EgoNeRF [55]: NeRF on a balanced spherical grid for egocentric omnidirectional views (Sec. VIII-G); OmniGS [56]: omnidirectional 3DGS (Sec. VIII-G); H3DGS [27]: hierarchical 3DGS with divide-and-conquer training and LoD rendering (Sec. VIII-E). We report PSNR, SSIM, and LPIPS [57]. PSNR measures per-pixel fidelity and tolerates blur, whereas SSIM and LPIPS respond to lost structure and perceptual detail.

2) Implementation Details: SpaRV is implemented in Py-Torch with custom CUDA kernels. Every SpaRV result in this paper is produced on a single NVIDIA RTX 4090 with 24 GB of device memory, an Intel Core i7-13700K CPU, and 64 GB of host memory. On the benchmark scenes of Secs. VIII-B– VIII-G, training runs 20,000 iterations of sparse Adam [16]: voxels are subdivided every 1,000 iterations and pruned on each coverage epoch, and camera optimization alternates with map training on the same interval. The large-scale scenes of Sec. VIII-E use the coarse-to-fine training of Sec. VII-B. Coarse training stretches the benchmark schedule by a factor that adapts to the dataset size. Here the factor is 5.0, so coarse training runs 100,000 iterations and every milestone above scales with it. Each submap is trained as an ordinary map-training run on the 20,000-iteration schedule. The target submap size is $s _ { t } = 1 0 0 \mathrm { m }$ , and the granularity target is � = 1. Deployment uses a target deploy-box size of $s _ { d } = 1 0$ m at the same granularity.

## B. Geometric Consistency on Replica

Replica [58] is a synthetic indoor dataset with groundtruth camera poses, which remove camera error from the comparison. We evaluate on 8 scenes at both interpolated and extrapolated viewpoints, the latter generated by uniformly sampling positions and orientations in each scene [41]. In the extrapolation evaluation, pixels that are not seen in the training views are excluded.

Interpolation evaluates how well each method fits the observed views; extrapolation evaluates whether the map stays geometrically consistent at viewpoints away from the captured trajectory. Table I reports the extrapolated results per scene and the averages of both, and Fig. 14 shows one interpolated and one extrapolated view of each method on Replica Room-0. At interpolated views the methods are not distinguishable: all of them fit the training images closely, and the six renderings in the top two rows are nearly identical. The methods differ at the extrapolated view of the same maps. Geometric inconsistency appears first. The captured trajectory covers the ceiling from few views and over a short baseline, so the image-only methods fit it to those views instead of recovering it as a surface. The ceiling renders correctly from the training views, and degenerates into thin spikes and blurred blobs once the camera leaves the trajectory. LetsGo, although its Gaussians are regularized by a LiDAR depth term, still smears the tile seams and leaves floaters. GS-SDF and SpaRV keep the plane flat and the seams straight. The extrapolated columns of Table I show the same result: SpaRV gains 3.3 dB in extrapolated PSNR over SVRaster, whose sparse-voxel representation it shares, and leads every baseline on all three averaged metrics.

![](images/72509ea4fa1b5d009ab4b7386b93d63f3d5b5c18c57bdcce1f3e4c285feb370d.jpg)  
Fig. 14. Rendering results on Replica Room-0. Top two rows: an interpolated training view, where every method is close to the reference. Bottom two rows: an extrapolated view of the same maps.

The two boxed regions of Fig. 14 compare texture detail and shading, which can differ even where the geometry is correct. In the ceiling exhaust vent, GS-SDF and SpaRV recover the individual slats that InstantNGP blurs into a single patch and that 3DGS, SVRaster and LetsGo break into thin spikes. On the wall beside the ceiling light, GS-SDF places the surface correctly, yet the illumination falloff breaks into streaks. Its overlapping Gaussians have no intrinsic compositing order, so the per-view sort leaves discontinuities at primitive boundaries [11]. SpaRV renders the falloff smoothly, because its non-overlapping voxels keep the compositing order exact. The LPIPS rows measure these appearance differences, and SpaRV leads at both interpolated and extrapolated viewpoints, ahead of GS-SDF.

## C. Real-World Mapping on FAST-LIVO2 Dataset

The FAST-LIVO2 dataset [3] is a real-world LiDAR-visual dataset captured with a solid-state Livox AVIA LiDAR. Its non-repetitive scanning pattern makes the range measurements sparse and irregular within any single frame. We evaluate on six outdoor scenes with three trajectory types (Fig. 15): forward-facing (Landmark, CBD, Street), objectcentric (Sculpture), and free-view (Culture, Drive). Their initial poses and intrinsic parameters come from the FAST-LIVO2 odometry system [3].

TABLE I  
QUANTITATIVE RENDERING RESULTS ON THE REPLICA DATASET. PER-SCENE VALUES ARE AT EXTRAPOLATED VIEWPOINTS (E); THE LAST TWO COLUMNS AVERAGE THE EXTRAPOLATED AND THE INTERPOLATED (I) VIEWPOINTS OVER THE 8 SCENES.
<table><tr><td rowspan="2">Metrics</td><td rowspan="2">Methods</td><td colspan="9">Extrapolated (E)</td><td>Interp. (I)</td></tr><tr><td>Office-0</td><td>Office-1</td><td>Office-2</td><td>Office-3</td><td>Office-4</td><td>Room-0</td><td>Room-1</td><td>Room-2</td><td>Avg.</td><td>Avg.</td></tr><tr><td rowspan="6">PSNR↑</td><td>InstantNGP</td><td>39.863</td><td>40.473</td><td>35.799</td><td>35.928</td><td>34.217</td><td>32.019</td><td>31.761</td><td>35.080</td><td>35.643</td><td>33.713</td></tr><tr><td>3DGS</td><td>36.047</td><td>36.315</td><td>25.438</td><td>23.639</td><td>28.532</td><td>25.944</td><td>27.098</td><td>26.703</td><td>28.714</td><td>40.439</td></tr><tr><td>SVRaster</td><td>37.914</td><td>36.460</td><td>31.267</td><td>30.544</td><td>32.781</td><td>31.881</td><td>33.291</td><td>32.636</td><td>33.347</td><td>39.600</td></tr><tr><td>LetsGo</td><td>30.263</td><td>30.808</td><td>23.130</td><td>21.762</td><td>26.225</td><td>21.949</td><td>24.512</td><td>21.756</td><td>25.051</td><td>38.892</td></tr><tr><td>GS-SDF</td><td>40.523</td><td>39.488</td><td>33.214</td><td>33.719</td><td>37.017</td><td>33.833</td><td>35.015</td><td>35.615</td><td>36.053</td><td>40.548</td></tr><tr><td>SpaRV</td><td>40.568</td><td>39.496</td><td>35.326</td><td>35.089</td><td>37.572</td><td>34.382</td><td>35.716</td><td>35.374</td><td>36.690</td><td>41.001</td></tr><tr><td rowspan="6">SSIM↑</td><td>InstantNGP</td><td>0.967</td><td>0.965</td><td>0.951</td><td>0.949</td><td>0.951</td><td>0.903</td><td>0.919</td><td>0.945</td><td>0.944</td><td>0.933</td></tr><tr><td>3DGS</td><td>0.963</td><td>0.947</td><td>0.914</td><td>0.892</td><td>0.936</td><td>0.886</td><td>0.916</td><td>0.919</td><td>0.922</td><td>0.974</td></tr><tr><td>SVRaster</td><td>0.968</td><td>0.936</td><td>0.946</td><td>0.943</td><td>0.953</td><td>0.936</td><td>0.942</td><td>0.947</td><td>0.946</td><td>0.975</td></tr><tr><td>LetsGo</td><td>0.924</td><td>0.895</td><td>0.892</td><td>0.890</td><td>0.913</td><td>0.851</td><td>0.892</td><td>0.831</td><td>0.886</td><td>0.965</td></tr><tr><td>GS-SDF</td><td>0.976</td><td>0.960</td><td>0.961</td><td>0.958</td><td>0.969</td><td>0.948</td><td>0.957</td><td>0.963</td><td>0.962</td><td>0.973</td></tr><tr><td>SpaRV</td><td>0.977</td><td>0.970</td><td>0.963</td><td>0.961</td><td>0.969</td><td>0.948</td><td>0.956</td><td>0.959</td><td>0.963</td><td>0.982</td></tr><tr><td rowspan="6">LPIPS↓</td><td>InstantNGP</td><td>0.095</td><td>0.128</td><td>0.149</td><td>0.147</td><td>0.159</td><td>0.199</td><td>0.178</td><td>0.149</td><td>0.150</td><td>0.154</td></tr><tr><td>3DGS</td><td>0.097</td><td>0.154</td><td>0.181</td><td>0.187</td><td>0.142</td><td>0.196</td><td>0.168</td><td>0.160</td><td>0.161</td><td>0.095</td></tr><tr><td>SVRaster</td><td>0.091</td><td>0.160</td><td>0.157</td><td>0.145</td><td>0.121</td><td>0.141</td><td>0.133</td><td>0.134</td><td>0.135</td><td>0.090</td></tr><tr><td>LetsGo</td><td>0.134</td><td>0.186</td><td>0.224</td><td>0.219</td><td>0.174</td><td>0.247</td><td>0.192</td><td>0.210</td><td>0.198</td><td>0.122</td></tr><tr><td>GS-SDF</td><td>0.060</td><td>0.151</td><td>0.115</td><td>0.105</td><td>0.085</td><td>0.123</td><td>0.091</td><td>0.093</td><td>0.103</td><td>0.078</td></tr><tr><td>SpaRV</td><td>0.045</td><td>0.076</td><td>0.079</td><td>0.082</td><td>0.065</td><td>0.101</td><td>0.079</td><td>0.086</td><td>0.077</td><td>0.047</td></tr></table>

![](images/a30730549df2c1ccba424714209b22d6544638f579dc5abccdfd3712bcbd89de.jpg)  
Fig. 15. The six FAST-LIVO2 scenes: LiDAR map, trajectory, and a SpaRV render per scene.

1) Camera Optimization: Residual camera errors blur the map even at the training views, so we train on all images of each sequence and evaluate each method on those same images (Table II). The table has two blocks, one per camera source. In the odometry cameras block, each method is trained on the FAST-LIVO2 odometry poses, and SpaRV is comparable to SVRaster and ahead of 3DGS and InstantNGP. The last row of that block, SpaRV (cam. opt.), alternates map training with the camera optimization of Sec. VI and outperforms every baseline. The optimized cameras block repeats the comparison with every method trained on the cameras that SpaRV (cam. opt.) optimized.

Camera optimization raises the PSNR of SpaRV on every scene, by 3.3 dB on average, and retraining SpaRV on the optimized cameras reproduces the gain. The optimized cameras are not specific to SpaRV: every baseline trained under both settings in Table II improves when retrained on them. Fig. 10 shows the textures on one training view of Drive and one of Sculpture, each rendered with 3DGS, SVRaster and SpaRV under both camera settings. Both zooms fall on the paving. The two scenes show two different effects of the odometry cameras. On Drive the texture is incomplete: the brick pattern collapses to a flat blur under 3DGS and SVRaster. On Sculpture the pattern survives but is not sharp: the joint lines of the plaza slabs and the edge of the curved step are blurred for every method. The optimized cameras complete the pattern on Drive and sharpen it on Sculpture. Under the odometry cameras, SpaRV already yields the sharpest of the three renders; under the optimized cameras it is the closest to the captured frame.

InstantNGP  
3DGS  
SVRaster  
LetsGo  
GS-SDF  
SpaRV  
Ground truth  
![](images/25f93d8415d77cb2ea3c3bf2c5d4b5ee7f3a31aba7a4a0d13a008586748f44f8.jpg)  
Fig. 16. Rendering results on three Oxford Spires sequences: Keble-05, Obs.-01, and Bodleian-02.

TABLE II  
TRAINING-VIEW PSNR ON FAST-LIVO2.
<table><tr><td>Method</td><td>Landmark</td><td>CBD</td><td>Culture</td><td>Drive</td><td>Sculpture</td><td>Street</td><td>Avg.</td></tr><tr><td colspan="8">odometry cameras</td></tr><tr><td>InstantNGP</td><td>31.705</td><td>25.312</td><td>23.131</td><td>18.379</td><td>25.276</td><td>23.429</td><td>24.539</td></tr><tr><td>3DGS</td><td>32.260</td><td>22.033</td><td>21.571</td><td>25.570</td><td>24.294</td><td>26.836</td><td>25.427</td></tr><tr><td>SVRaster</td><td>31.616</td><td>26.419</td><td>24.749</td><td>26.035</td><td>24.636</td><td>26.025</td><td>26.580</td></tr><tr><td>SpaRV</td><td>31.223</td><td>25.923</td><td>24.681</td><td>26.871</td><td>24.483</td><td>26.130</td><td>26.552</td></tr><tr><td>SpaRV (cam. opt.)</td><td>36.190</td><td>28.114</td><td>26.512</td><td>29.817</td><td>28.548</td><td>30.007</td><td>29.865</td></tr><tr><td colspan="8">optimized cameras</td></tr><tr><td>InstantNGP</td><td>31.693</td><td>25.182</td><td>24.064</td><td>19.622</td><td>25.290</td><td>23.596</td><td>24.908</td></tr><tr><td>3DGS</td><td>36.256</td><td>23.159</td><td>21.231</td><td>27.006</td><td>27.937</td><td>30.657</td><td>27.708</td></tr><tr><td>SVRaster</td><td>36.011</td><td>28.680</td><td>24.755</td><td>27.587</td><td>28.411</td><td>29.443</td><td>29.148</td></tr><tr><td>LetsGo</td><td>32.066</td><td>27.107</td><td>24.218</td><td>17.712</td><td>23.177</td><td>28.408</td><td>25.448</td></tr><tr><td>GS-SDF</td><td>35.744</td><td>28.584</td><td>27.143</td><td>28.298</td><td>28.066</td><td>30.493</td><td>29.721</td></tr><tr><td>SpaRV</td><td>36.403</td><td>28.157</td><td>26.601</td><td>29.914</td><td>28.605</td><td>30.013</td><td>29.949</td></tr></table>

2) Geometrically Consistent Rendering: Table III evaluates rendering on a test split under the optimized cameras of Sec. VIII-C1. The split excludes one image out of every eight from training and uses it only for evaluation. SpaRV has the lowest LPIPS on all six scenes and the highest SSIM on five of them, ahead of GS-SDF, the closest baseline. Figs. 17 and 2 contrast interpolated with laterally extrapolated views on Street. The methods are again indistinguishable at the interpolated view and separate once the camera moves off the captured trajectory. At the extrapolated view, the image-only methods (InstantNGP, 3DGS, SVRaster) degrade, while the LiDAR-supervised methods (LetsGo, GS-SDF, SpaRV) keep the geometry and texture more stable. The Gaussian-based LetsGo and GS-SDF show popping on the billboard, where

![](images/b49445bedd45e502f8c40fd8f643ae1d1ea470b2b700d2a8fd46e5f6732063b5.jpg)  
(d) LetsGo  
(e) GS-SDF  
(f) SpaRV  
Fig. 17. Rendering results on the FAST-LIVO2 Street scene at an interpolated view on the captured trajectory.

TABLE III  
QUANTITATIVE RESULTS ON THE FAST-LIVO2 TEST SPLIT, UNDER THE OPTIMIZED CAMERAS.
<table><tr><td>Metrics</td><td>Methods</td><td>Landmark</td><td>CBD</td><td>Culture</td><td>Drive</td><td>Sculpture</td><td>Street</td><td>Avg.</td></tr><tr><td rowspan="6">PSNR↑</td><td>InstantNGP</td><td>30.983</td><td>25.080</td><td>23.339</td><td>24.111</td><td>25.080</td><td>23.506</td><td>25.350</td></tr><tr><td>3DGS</td><td>34.567</td><td>22.659</td><td>21.112</td><td>27.234</td><td>27.496</td><td>30.123</td><td>27.198</td></tr><tr><td>SVRaster</td><td>34.172</td><td>27.042</td><td>24.720</td><td>26.763</td><td>27.311</td><td>28.242</td><td>28.042</td></tr><tr><td>LetsGo</td><td>33.580</td><td>27.849</td><td>25.171</td><td>29.386</td><td>24.588</td><td>29.798</td><td>28.395</td></tr><tr><td>GS-SDF</td><td>34.057</td><td>27.157</td><td>25.158</td><td>28.555</td><td>27.794</td><td>30.093</td><td>28.802</td></tr><tr><td>SpaRV</td><td>33.971</td><td>26.605</td><td>24.720</td><td>29.227</td><td>27.887</td><td>29.643</td><td>28.676</td></tr><tr><td rowspan="6">SSIM↑</td><td>InstantNGP</td><td>0.853</td><td>0.802</td><td>0.684</td><td>0.675</td><td>0.777</td><td>0.772</td><td>0.761</td></tr><tr><td>3DGS</td><td>0.924</td><td>0.814</td><td>0.704</td><td>0.818</td><td>0.874</td><td>0.938</td><td>0.845</td></tr><tr><td>SVRaster</td><td>0.921</td><td>0.905</td><td>0.812</td><td>0.818</td><td>0.878</td><td>0.904</td><td>0.873</td></tr><tr><td>LetsGo</td><td>0.911</td><td>0.900</td><td>0.809</td><td>0.918</td><td>0.823</td><td>0.929</td><td>0.881</td></tr><tr><td>GS-SDF</td><td>0.915</td><td>0.905</td><td>0.820</td><td>0.897</td><td>0.879</td><td>0.936</td><td>0.892</td></tr><tr><td>SpaRV</td><td>0.923</td><td>0.914</td><td>0.844</td><td>0.939</td><td>0.902</td><td>0.947</td><td>0.911</td></tr><tr><td rowspan="6">LPIPS↓</td><td>InstantNGP</td><td>0.202</td><td>0.341</td><td>0.419</td><td>0.426</td><td>0.294</td><td>0.265</td><td>0.325</td></tr><tr><td>3DGS</td><td>0.119</td><td>0.298</td><td>0.371</td><td>0.275</td><td>0.195</td><td>0.084</td><td>0.224</td></tr><tr><td>SVRaster</td><td>0.107</td><td>0.143</td><td>0.210</td><td>0.255</td><td>0.169</td><td>0.127</td><td>0.168</td></tr><tr><td>LetsGo</td><td>0.137</td><td>0.166</td><td>0.229</td><td>0.105</td><td>0.250</td><td>0.097</td><td>0.164</td></tr><tr><td>GS-SDF</td><td>0.138</td><td>0.140</td><td>0.203</td><td>0.146</td><td>0.190</td><td>0.081</td><td>0.149</td></tr><tr><td>SpaRV</td><td>0.100</td><td>0.099</td><td>0.144</td><td>0.071</td><td>0.135</td><td>0.067</td><td>0.103</td></tr></table>

SpaRV stays consistent.

## D. Benchmarking with Accurate Trajectories on Oxford Spires

Oxford Spires [59] is an outdoor LiDAR-visual benchmark of 12 sequences around Oxford landmarks, captured with a

TABLE IV  
OXFORD SPIRES DATASET RECONSTRUCTION BENCHMARK RESULTS.
<table><tr><td>Metric</td><td>Method</td><td>Keble-02</td><td>Keble-03</td><td>Keble-04</td><td>Keble-05</td><td>Obs.-01</td><td>Obs.-02</td><td>Blenheim-01</td><td>Blenheim-02</td><td>Blenheim-05</td><td>Christ-02</td><td>Christ-03</td><td>Bodleian-02</td><td>Avg.</td></tr><tr><td rowspan="7">PSNR↑</td><td>InstantNGP</td><td>22.182 28.910</td><td>22.265 29.146</td><td>22.627 24.652</td><td>23.342 25.082</td><td>24.082 25.727</td><td>24.280 26.027</td><td>24.845</td><td>25.788 25.992</td><td>25.189 25.503</td><td>23.703 19.971</td><td>27.079</td><td>23.847 24.716</td><td>24.102</td></tr><tr><td>3DGS SVRaster</td><td></td><td></td><td></td><td></td><td></td><td></td><td>24.631</td><td></td><td></td><td></td><td>28.752</td><td>26.286</td><td>25.759</td></tr><tr><td></td><td>26.927 27.801</td><td>27.544</td><td>24.562</td><td>24.816</td><td>25.240</td><td>25.431</td><td>23.367</td><td>24.011</td><td>23.938</td><td>22.491</td><td>28.819</td><td></td><td>25.286</td></tr><tr><td>LetsGo</td><td></td><td>27.819</td><td>24.612</td><td>26.202</td><td>23.468</td><td>23.556</td><td>25.135</td><td>24.837</td><td>24.090</td><td>24.966</td><td>29.070</td><td>27.009</td><td>25.714</td></tr><tr><td>GS-SDF</td><td>28.063</td><td>28.805</td><td>25.270</td><td>25.728</td><td>25.667</td><td>25.765</td><td>24.624</td><td>25.541</td><td>25.054</td><td>23.834</td><td>26.676</td><td>25.536</td><td>25.880</td></tr><tr><td>SpaRV</td><td>28.452</td><td>29.159</td><td>26.171</td><td>26.705</td><td>26.937</td><td>27.217</td><td>25.033</td><td>25.850</td><td>25.936</td><td>24.357</td><td>28.485</td><td>27.740</td><td>26.837</td></tr><tr><td>SpaRV (occ off)</td><td>28.320</td><td>29.031</td><td>26.041</td><td>26.577</td><td>26.762</td><td>27.014</td><td>24.580</td><td>25.204</td><td>25.444</td><td>24.231</td><td>28.392</td><td>27.682</td><td>26.607</td></tr><tr><td rowspan="7">SSIM↑</td><td>InstantNGP</td><td>0.750</td><td>0.753</td><td>0.722</td><td>0.737</td><td>0.756 0.759</td><td></td><td>0.782</td><td>0.783</td><td>0.779</td><td>0.795</td><td>0.894</td><td>0.752</td><td>0.772</td></tr><tr><td>3DGS</td><td>0.865</td><td>0.865</td><td>0.767</td><td>0.776</td><td>0.795</td><td>0.801</td><td>0.754</td><td>0.773</td><td>0.768</td><td>0.720</td><td>0.919</td><td>0.746</td><td>0.796</td></tr><tr><td>SVRaster</td><td>0.863</td><td>0.866</td><td>0.787</td><td>0.800</td><td>0.818</td><td>0.823</td><td>0.765</td><td>0.781</td><td>0.772</td><td>0.779</td><td>0.916</td><td>0.800</td><td>0.814</td></tr><tr><td>LetsGo</td><td>0.893</td><td>0.892</td><td>0.807</td><td>0.838</td><td>0.805</td><td>0.811</td><td>0.798</td><td>0.793</td><td>0.784</td><td>0.843</td><td>0.928</td><td>0.823</td><td>0.835</td></tr><tr><td>GS-SDF</td><td>0.866</td><td>0.870</td><td>0.796</td><td>0.810</td><td>0.810</td><td>0.811</td><td>0.779</td><td>0.791</td><td>0.785</td><td>0.805</td><td>0.905</td><td>0.780</td><td>0.817</td></tr><tr><td>SpaRV</td><td>0.899</td><td>0.901</td><td>0.832</td><td>0.846</td><td>0.864</td><td>0.869</td><td>0.810</td><td>0.825</td><td>0.817</td><td>0.839</td><td>0.927</td><td>0.849</td><td>0.857</td></tr><tr><td>SpaRV (occ off)</td><td>0.899</td><td>0.901</td><td>0.834</td><td>0.847</td><td>0.865</td><td>0.870</td><td>0.808</td><td>0.825</td><td>0.813</td><td>0.838</td><td>0.926</td><td></td><td>0.857</td></tr><tr><td rowspan="7">LPIPS↓</td><td>InstantNGP</td><td>0.412</td><td>0.412</td><td>0.433</td><td>0.419</td><td>0.422</td><td>0.421</td><td>0.460</td><td>0.457</td><td>0.459</td><td>0.444</td><td>0.325</td><td>0.853 0.475</td><td>0.428</td></tr><tr><td>3DGS</td><td>0.267</td><td>0.273</td><td>0.368</td><td>0.365</td><td>0.337</td><td>0.331</td><td>0.448</td><td>0.422</td><td>0.429</td><td>0.490</td><td>0.227</td><td>0.432</td><td>0.366</td></tr><tr><td>SVRaster</td><td>0.228</td><td>0.226</td><td>0.300</td><td>0.294</td><td>0.268</td><td>0.264</td><td>0.385</td><td>0.362</td><td>0.370</td><td>0.377</td><td>0.216</td><td>0.328</td><td>0.302</td></tr><tr><td>LetsGo</td><td>0.203</td><td>0.205</td><td>0.308</td><td>0.278</td><td>0.313</td><td>0.305</td><td>0.389</td><td>0.386</td><td>0.411</td><td>0.324</td><td>0.196</td><td>0.328</td><td>0.304</td></tr><tr><td>GS-SDF</td><td>0.244</td><td>0.246</td><td>0.318</td><td>0.309</td><td>0.308</td><td>0.309</td><td>0.408</td><td>0.387</td><td>0.403</td><td>0.388</td><td>0.248</td><td>0.384</td><td>0.329</td></tr><tr><td>SpaRV</td><td>0.160</td><td>0.160</td><td>0.232</td><td>0.225</td><td>0.196</td><td>0.192</td><td>0.321</td><td>0.294</td><td>0.304</td><td>0.274</td><td>0.164</td><td>0.256</td><td>0.232</td></tr><tr><td>SpaRV (occ off)</td><td>0.159</td><td>0.159</td><td>0.228</td><td>0.221</td><td>0.193</td><td>0.189</td><td>0.317</td><td>0.288</td><td>0.305</td><td>0.274</td><td>0.167</td><td>0.245</td><td>0.229</td></tr></table>

64-channel spinning Hesai QT64 LiDAR and three synchronized global-shutter cameras. Its ground-truth trajectories are registered against a millimeter-accurate Leica RTC360 laser scan, so they are much more accurate than robotic odometry. Table IV summarizes the benchmark results. SpaRV leads on all three averaged metrics. Its LPIPS is the lowest on all 12 scenes and its SSIM on 10 of them, while PSNR separates the methods less. Fig. 16 shows Keble-05, Obs.-01, and Bodleian-02. On Keble-05, the baselines blur the bricks and the tree trunks are missing. On Obs.-01, they blur the roof and the building corner is missing. On Bodleian-02, the paving joints become a grey patch. SpaRV reconstructs these details.

The SpaRV (occ off) row of Table IV turns off the occupied voxel guidance $\mathcal { L } _ { \mathrm { o c c } }$ (Sec. V-B4) and keeps every other loss and camera setting fixed. PSNR then drops on all 12 scenes, SSIM remains nearly unchanged, and LPIPS is slightly better without the term. Fig. 9 compares the same view with and without the term. The color images are nearly indistinguishable, which is consistent with SSIM and LPIPS, but the depth and normal maps without the term reveal a band of floaters over the lawn.

## E. Large-Scale Reconstruction on Garden and Campus

We captured Garden and Campus (Fig. 1) with a handheld Manifold Pocket 2 scanner<sup>1</sup> carrying a solid-state Livox MID360 LiDAR and three fisheye cameras. Their poses are produced by the vendor’s commercial software with loop closure and bundle adjustment. Garden is an outdoor district densely covered with complex objects: 44,550 images and 146.7 M LiDAR points along a 2,239 m trajectory covering 45,199 m<sup>2</sup>. Campus spans an entire university: 63,582 images and 151.6 M LiDAR points along a 3,534 m trajectory covering 234,035 m<sup>2</sup>.

On each scene, we compare SpaRV at two stages with H3DGS. SpaRV (coarse) is the output of coarse training, which spreads the budget $N _ { \mathrm { m a x } }$ over the whole scene. SpaRV (deploy) adds submap training (Sec. VII-B2), which refines one submap at a time as far as the budget allows, and renders through a cut around the camera (Sec. VII-C). H3DGS [27] is the large-scale baseline. Table V reports the rendering quality of H3DGS, SpaRV (coarse), and SpaRV (deploy) on the training views of both scenes. Submap training improves SpaRV (coarse) on every metric, most on Campus (+2.5 dB PSNR), whose larger extent spreads the coarse budget more thinly. SpaRV (deploy) has the highest PSNR and SSIM on both scenes. Fig. 18 shows one matched viewpoint per scene. SpaRV (coarse) already recovers the layout of both scenes, but fine content is blocky: the lettering on the Garden hydrant cabinet and on the Campus poster is unreadable. SpaRV (deploy) makes both legible and restores smaller details, such as the two red valves on the cabinet and the wet-floor sign in front of the poster. H3DGS smears the same regions and leaves streaks on the Garden lawn and the Campus glass wall.

H3DGS  
SpaRV (deploy)  
SpaRV (coarse)  
![](images/a0a4bee012d4a92ea46c8321e6cf3be452d345d137216ee45c488cad0884a9de.jpg)  
Fig. 18. Rendering results on Garden (top) and Campus (bottom): H3DGS, SpaRV (coarse), and SpaRV (deploy).

## F. Runtime Analysis

Table VI reports the training and rendering cost of the two systems on Campus. Both systems are evaluated on the same

TABLE V  
RENDERING QUALITY ON THE TWO SELF-COLLECTED SCENES, EVALUATED ON THE TRAINING VIEWS.
<table><tr><td>Scene</td><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="3">Garden</td><td>H3DGS</td><td>18.02</td><td>0.620</td><td>0.282</td></tr><tr><td>SpaRV (coarse)</td><td>20.89</td><td>0.712</td><td>0.346</td></tr><tr><td>SpaRV (deploy)</td><td>21.12</td><td>0.713</td><td>0.295</td></tr><tr><td rowspan="3">Campus</td><td>H3DGS</td><td>20.01</td><td>0.660</td><td>0.295</td></tr><tr><td>SpaRV (coarse)</td><td>20.62</td><td>0.652</td><td>0.451</td></tr><tr><td>SpaRV (deploy)</td><td>23.16</td><td>0.760</td><td>0.283</td></tr></table>

3960-frame fly-through at 1280×960, each cutting at �=1, one pixel of projected size. SpaRV needs 8.7 h in total to produce the deployment map. The coarse map alone trains in 0.8 h and renders at 248.4 FPS in 4.3 GiB, so it already serves as a fast preview of the whole scene, although with less detail than the deployment map (Table V). To render the deployment map, SpaRV applies the cut around the camera under a 16 GiB GPU memory budget. It renders at 68.8 FPS in 15.6 GiB, against H3DGS’s 29.6 FPS in 22.5 GiB, and its rate includes the cost of the cut. The cut is recomputed only when the camera leaves the deploy box, which happens on 12% of frames and adds a bounded 63.6 ms to each of them; the remaining frames reuse the voxels already resident. Training peaks at 12.5 GiB and rendering at 15.6 GiB, so SpaRV reconstructs and renders the kilometer-scale Campus scene on a single 24 GB GPU.

TABLE VI  
TRAINING AND RENDERING COST ON CAMPUS, OVER ONE 3960-FRAME FLY-THROUGH AT 1280×960. PRIM. IS THE LARGEST NUMBER OF PRIMITIVES RENDERED IN A FRAME.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Train [h]↓ FPS↑ Prim. [M]</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="2">Peak VRAM [GiB]↓</td></tr><tr><td>Train</td><td>Render</td></tr><tr><td>H3DGS</td><td>7.3</td><td>29.6</td><td>15.5</td><td>8.3</td><td>22.5</td></tr><tr><td>SpaRV (coarse)</td><td>0.8</td><td>248.4</td><td>9.9</td><td>12.5</td><td>4.3</td></tr><tr><td>SpaRV (deploy)</td><td>8.7</td><td>68.8</td><td>14.3</td><td>11.7</td><td>15.6</td></tr></table>

## G. Omnidirectional Cameras on the EgoNeRF Dataset

The EgoNeRF dataset [55] is an egocentric view-synthesis benchmark of 22 equirectangular sequences, 11 synthetic scenes in OmniBlender and 11 real captures in Ricoh360, each recorded along a short orbit around a fixed center. The benchmark carries no range data, so SpaRV is trained from images alone in this setting. The only change from the benchmark experiments of Secs. VIII-B–VIII-D is the ray generator: each pixel of an equirectangular image is mapped to a ray direction by its longitude and latitude, instead of through the pinhole intrinsics. The map, the renderer, and the training schedule are unchanged. Following the OmniGS [56] protocol, we reconstruct from the OpenMVG poses shipped with the benchmark, train EgoNeRF [55] on the same poses and split, and evaluate all methods on the same test frames.

SpaRV outperforms OmniGS on all three metrics on both splits (Table VII). The PSNR margin is small, 0.34 dB on the synthetic OmniBlender scenes and 0.15 dB on the real Ricoh360 captures, while LPIPS drops by 26% and 11%. Fig. 19 compares the three methods on two Ricoh360 scenes.

TABLE VII  
NOVEL-VIEW SYNTHESIS RESULTS ON THE TEST FRAMES OF THE EGONERF DATASET.
<table><tr><td rowspan="2">Method</td><td colspan="3">OmniBlender (11)</td><td colspan="3">Ricoh360 (11)</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>EgoNeRF [55]</td><td>30.93</td><td>0.892</td><td>0.207</td><td>24.89</td><td>0.752</td><td>0.353</td></tr><tr><td>OmniGS [56]</td><td>33.09</td><td>0.918</td><td>0.174</td><td>25.99</td><td>0.825</td><td>0.262</td></tr><tr><td>SpaRV</td><td>33.43</td><td>0.926</td><td>0.129</td><td>26.14</td><td>0.838</td><td>0.233</td></tr></table>

OmniGS specializes its rasterizer for the equirectangular camera, whereas SpaRV uses the renderer and training schedule of the pinhole experiments unchanged.

Farm  
Bridge  
![](images/36a24ec3bec3eb3be8b6453db4501c5646ff4c94acd7e32d05d5e00f080f2e44.jpg)  
Fig. 19. Rendering results on two Ricoh360 scenes of the EgoNeRF dataset.

## H. Applications

The trained map also supports the simulation of other camera and LiDAR models and surface extraction. The field answers arbitrary ray queries, so it can simulate sensors it was never trained on (Sec. VIII-H1). Its field can be polygonized into an explicit surface (Sec. VIII-H2).

1) Sensor Simulation: After training, SpaRV renders a different sensor by changing only the ray generator; the geometry and appearance stay fixed. Fig. 20 shows three camera models (pinhole, fisheye, and omnidirectional) and three LiDAR scan patterns (AVIA, MID360, and OS128), all from one trained map. The Livox-style AVIA and MID360 patterns use timeordered scan tables from the official Livox scan-mode assets. OS128 is a wide-field ring-azimuth grid approximation [60]. For each preset, SpaRV ray-marches those rays and converts the rendered median depth into per-frame point clouds.

![](images/5f67e34681ca2d1118af3801feab5c70da9f43e559c2b8399ee9640baed98fad.jpg)  
(a) Pinhole

![](images/406751e7b445ca467868a81b1a99097cbf2879b078dd7be27a5b0e34d2551645.jpg)

![](images/ff72d1206e43c52942f2edcc9c46522b3a2992af4e5aaef5682e7c6fe97fcf24.jpg)

(b) Fisheye  
![](images/3ae948e6622d7c8db1dadb86cb0f37d5562fff1a5482e4f3e8b202676ae3a1f1.jpg)

(d) AVIA pattern  
![](images/8a4da5b4c8916a60e154e17dac42473adc89f9fd33a13e2d6cc02fa5060cf851.jpg)  
(e) MID360 pattern

(c) Omni  
![](images/e87948c74ca19c5cfd472f73382b6a320d12a967279125f759c7a11fbfa412b6.jpg)  
(f) OS128 pattern  
Fig. 20. Sensor simulation from one trained map: three camera models (top) and three LiDAR scan patterns (bottom).

2) Surface Extraction: Collision checking, physics, and simulator assets need an explicit mesh. Instead of fusing rendered depth maps into a TSDF volume [32], [61], which smooths away detail the field already encodes, SpaRV maps each trained grid-point density � to a per-vertex opacity through Eq. (7), with the interval set to the local voxel size $\nu _ { \mathrm { s } } .$ It then polygonizes that alpha field by marching cubes [62]. The mesh resolution thus follows the adaptive voxels. Fig. 21 shows the meshes extracted from the six FAST-LIVO2 maps of Sec. VIII-C.

![](images/efe312c34ee37a2a953bd85d8040f81e4ad4a64c4b92053aebdd393e497a56ca.jpg)  
(a) Street

![](images/b75c0c39f7145d28d2f74bf7e86bb1a38a28149307005754527efeae8c1032a7.jpg)  
(b) CBD

![](images/79bc47e4d859a3ca5afd7536417089e9a33db4e0f89a4c57587925be0da62379.jpg)  
(c) Landmark

![](images/e9f8eee2a6a2b9729fa6abcd606e9f8d850f2e5de589fde1b7c5ca07e964cfad.jpg)

![](images/50493d723dbc4fa45190422d3b7fa412f9ca686dc58f4c37d386cd22c74eb4ff.jpg)  
(e) Culture

(d) Sculpture  
![](images/a9fc3cfe639d109252098cb63487154e72f047a7dc1e44c7258a7b6fa41006d6.jpg)  
(f) Drive  
Fig. 21. Meshes extracted from the six FAST-LIVO2 maps of Sec. VIII-C by marching cubes.

## IX. CONCLUSION AND FUTURE WORK

This paper presented SpaRV, a sparse radiance voxel map that fuses LiDAR and cameras for geometrically consistent mapping and rendering. A differentiable ray marcher renders the sparse voxels along each ray, so LiDAR rays supervise the map alongside images. LiDAR depth supervision and occupied voxel guidance ground the geometry, so the map stays consistent at viewpoints away from the captured trajectory. Map training alternates with photometric optimization of the camera poses and focal lengths, which sharpens the textures of the map. A granularity-driven cut of the octree bounds GPU memory in training and rendering, so SpaRV reconstructs and renders the kilometer-scale scene on a single 24 GB GPU. The trained maps also support surface extraction and the simulation of other camera and LiDAR models.

SpaRV stores geometry and appearance in the same voxels, but texture is often high-frequency where the geometry is not: a flat poster needs small voxels for its texture, not for its shape. Keeping geometry in the sparse voxels and moving appearance to a more compact representation would let the map resolve finer detail within the same GPU memory. SpaRV also assumes a static scene and carries no semantic labels. We leave the decoupled representation, dynamic objects, and semantics to future work.

## REFERENCES

[1] A. Mazumder, M. Sahed, Z. Tasneem, P. Das, F. Badal, M. Ali, M. Ahamed, S. Abhi, S. Sarker, S. Das et al., “Towards next generation digital twin in robotics: Trends, scopes, challenges, and future,” Heliyon, vol. 9, no. 2, 2023.

[2] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa et al., “Isaac gym: High performance gpu-based physics simulation for robot learning,” arXiv preprint arXiv:2108.10470, 2021.

[3] C. Zheng, W. Xu, Z. Zou, T. Hua, C. Yuan, D. He, B. Zhou, Z. Liu, J. Lin, F. Zhu et al., “Fast-livo2: Fast, direct lidar-inertial-visual odom etry,” arXiv preprint arXiv:2408.14035, 2024.

[4] B. Zhou, C. Zheng, Z. Wang, F. Zhu, Y. Cai, and F. Zhang, “Fastlivo2 on resource-constrained platforms: lidar-inertial-visual odometry with efficient memory and computation,” IEEE Robotics and Automation Letters, vol. 10, no. 8, pp. 7931–7938, 2025.

[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.

[6] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, pp. 1–14, 2023.

[7] K. Rematas, A. Liu, P. P. Srinivasan, J. T. Barron, A. Tagliasacchi, T. Funkhouser, and V. Ferrari, “Urban radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12 932–12 942.

[8] C. Jiang, R. Gao, K. Shao, Y. Wang, R. Xiong, and Y. Zhang, “Ligs: Gaussian splatting with lidar incorporated for accurate large-scale reconstruction,” IEEE Robotics and Automation Letters, 2024.

[9] J. Cui, J. Cao, F. Zhao, Z. He, Y. Chen, Y. Zhong, L. Xu, Y. Shi, Y. Zhang, and J. Yu, “Letsgo: Large-scale garage modeling and rendering via lidar-assisted gaussian primitives,” ACM Transactions on Graphics, vol. 43, no. 6, 2024.

[10] J. Liu, Y. Wan, B. Wang, C. Zheng, J. Lin, and F. Zhang, “Gs-sdf: Lidaraugmented gaussian splatting and neural sdf for geometrically consistent rendering and reconstruction,” arXiv preprint arXiv:2503.10170, 2025.

[11] L. Radl, M. Steiner, M. Parger, A. Weinrauch, B. Kerbl, and M. Steinberger, “Stopthepop: Sorted gaussian splatting for view-consistent realtime rendering,” ACM Transactions on Graphics (TOG), vol. 43, no. 4, pp. 1–17, 2024.

[12] T. Takikawa, J. Litalien, K. Yin, K. Kreis, C. Loop, D. Nowrouzezahrai, A. Jacobson, M. McGuire, and S. Fidler, “Neural geometric level of detail: Real-time rendering with implicit 3d shapes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 11 358–11 367.

[13] T. Muller, A. Evans, C. Schied, and A. Keller, “Instant neural graph-¨ ics primitives with a multiresolution hash encoding,” arXiv preprint arXiv:2201.05989, 2022.

[14] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, “Plenoxels: Radiance fields without neural networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5501–5510.

[15] C. Sun, M. Sun, and H.-T. Chen, “Direct voxel grid optimization: Superfast convergence for radiance fields reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5459–5469.

[16] C. Sun, J. Choe, C. Loop, W. Ma, and Y. F. Wang, “Sparse voxels rasterization: Real-time high-fidelity radiance field rendering,” in CVPR, 2025.

[17] P. Furgale, J. Rehder, and R. Siegwart, “Unified temporal and spatial calibration for multi-sensor systems,” in 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2013, pp. 1280–1286.

[18] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”¨ in Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[19] R. Li, X. Liu, H. Li, Z. Liu, J. Lin, Y. Cai, and F. Zhang, “LVBA: LiDAR-visual bundle adjustment for RGB point cloud mapping,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 2041–2047.

[20] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey, “Barf: Bundle-adjusting neural radiance fields,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5741–5751.

[21] Z. Wang, S. Wu, W. Xie, M. Chen, and V. A. Prisacariu, “NeRF--: Neural radiance fields without known camera parameters,” arXiv preprint arXiv:2102.07064, 2021.

[22] Q. Herau, N. Piasco, M. Bennehar, L. Roldao, D. Tsishkou, C. Migniot,˜ P. Vasseur, and C. Demonceaux, “MOISST: Multimodal optimization of implicit scene for spatiotemporal calibration,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023, pp. 1810– 1817.

[23] Z. Yang, G. Chen, H. Zhang, K. Ta, I. A. Barsan, D. Murphy, S. Mani-ˆ vasagam, and R. Urtasun, “UniCal: Unified neural sensor calibration,” in European Conference on Computer Vision (ECCV), ser. Lecture Notes in Computer Science, vol. 15094, 2024, pp. 327–345.

[24] M. Tancik, V. Casser, X. Yan, S. Pradhan, B. Mildenhall, P. P. Srinivasan, J. T. Barron, and H. Kretzschmar, “Block-nerf: Scalable large scene neural view synthesis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 8248–8258.

[25] Y. Tao and M. Fallon, “Silvr: Scalable lidar-visual radiance field reconstruction with uncertainty quantification,” IEEE Transactions on Robotics, 2025.

[26] K. Ren, L. Jiang, T. Lu, M. Yu, L. Xu, Z. Ni, and B. Dai, “Octree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians,” arXiv preprint arXiv:2403.17898, 2024.

[27] B. Kerbl, A. Meuleman, G. Kopanas, M. Wimmer, A. Lanvin, and G. Drettakis, “A hierarchical 3d gaussian representation for real-time rendering of very large datasets,” ACM Transactions on Graphics (TOG), vol. 43, no. 4, pp. 1–15, 2024.

[28] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance fields,” in European Conference on Computer Vision. Springer, 2022, pp. 333–350.

[29] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, “imap: Implicit mapping and positioning in real-time,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 6229–6238.

[30] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, “Nice-slam: Neural implicit scalable encoding for slam,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12 786–12 796.

[31] L. Liu, J. Gu, K. Zaw Lin, T.-S. Chua, and C. Theobalt, “Neural sparse voxel fields,” Advances in Neural Information Processing Systems, vol. 33, pp. 15 651–15 663, 2020.

[32] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting for geometrically accurate radiance fields,” in ACM SIGGRAPH 2024 conference papers, 2024, pp. 1–11.

[33] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu, H. Bao, and G. Zhang, “Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction,” arXiv preprint arXiv:2406.06521, 2024.

[34] K. Deng, A. Liu, J.-Y. Zhu, and D. Ramanan, “Depth-supervised nerf: Fewer views and faster training for free,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12 882–12 891.

[35] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth anything: Unleashing the power of large-scale unlabeled data,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 10 371–10 381.

[36] H. Xu, S. Peng, F. Wang, H. Blum, D. Barath, A. Geiger, and M. Pollefeys, “Depthsplat: Connecting gaussian splatting and depth,” in 2025

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 16 453–16 463.

[37] Z. Xie, J. Zhang, W. Li, F. Zhang, and L. Zhang, “S-nerf: Neural radiance fields for street views,” in International Conference on Learning Representations (ICLR), 2023.

[38] C. Jiang, H. Zhang, P. Liu, Z. Yu, H. Cheng, B. Zhou, and S. Shen, “H {2}-mapping: Real-time dense mapping using hierarchical hybrid representation,” IEEE Robotics and Automation Letters, 2023.

[39] X. Lang, L. Li, C. Wu, C. Zhao, L. Liu, Y. Liu, J. Lv, and X. Zuo, “Gaussian-lic: Real-time photo-realistic slam with gaussian splatting and lidar-inertial-camera fusion,” in IEEE International Conference on Robotics and Automation (ICRA), 2025.

[40] S. Hong, J. He, X. Zheng, H. Wang, H. Fang, K. Liu, C. Zheng, and S. Shen, “Liv-gaussmap: Lidar-inertial-visual fusion for real-time 3d radiance field map rendering,” arXiv preprint arXiv:2401.14857, 2024.

[41] J. Liu, C. Zheng, Y. Wan, B. Wang, Y. Cai, and F. Zhang, “Neural surface reconstruction and rendering for lidar-visual systems,” arXiv preprint arXiv:2409.05310, 2024.

[42] W. Xu, Y. Cai, D. He, J. Lin, and F. Zhang, “Fast-lio2: Fast direct lidarinertial odometry,” IEEE Transactions on Robotics, vol. 38, no. 4, pp. 2053–2073, 2022.

[43] Y. Chen, X. Chen, X. Wang, Q. Zhang, Y. Guo, Y. Shan, and F. Wang, “Local-to-global registration for bundle-adjusting neural radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8264–8273.

[44] Y. Jeong, S. Ahn, C. Choy, A. Anandkumar, M. Cho, and J. Park, “Selfcalibrating neural radiance fields,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 5846– 5854.

[45] Z. Fan, W. Cong, K. Wen, K. Wang, J. Zhang, X. Ding, D. Xu, B. Ivanovic, M. Pavone, G. Pavlakos, Z. Wang, and Y. Wang, “InstantSplat: Sparse-view Gaussian splatting in seconds,” arXiv preprint arXiv:2403.20309, 2024.

[46] Y. Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, and X. Wang, “Colmapfree 3d gaussian splatting,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20 796–20 805.

[47] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, “Gsslam: Dense visual slam with 3d gaussian splatting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 595–19 604.

[48] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, “Gaussian Splatting SLAM,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[49] K. Park, P. Henzler, B. Mildenhall, J. T. Barron, and R. Martin-Brualla, “CamP: Camera preconditioning for neural radiance fields,” ACM Transactions on Graphics, vol. 42, no. 6, pp. 1–11, 2023.

[50] K. Zhang, G. Riegler, N. Snavely, and V. Koltun, “Nerf++: Analyzing and improving neural radiance fields,” arXiv preprint arXiv:2010.07492, 2020.

[51] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5470–5479.

[52] A. Karnewar, T. Ritschel, O. Wang, and N. J. Mitra, “Relu fields: The little non-linearity that could,” in ACM Transactions on Graphics (SIGGRAPH), 2022.

[53] E. Eade, “Lie groups for 2d and 3d transformations,” URL http://ethaneade. com/lie. pdf, revised Dec, vol. 117, p. 118, 2013.

[54] P. J. Huber, “Robust estimation of a location parameter,” in Breakthroughs in statistics: Methodology and distribution. Springer, 1992, pp. 492–518.

[55] C. Choi, S. M. Kim, and Y. M. Kim, “Balanced spherical grid for egocentric view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 16 590– 16 599.

[56] L. Li, H. Huang, S.-K. Yeung, and H. Cheng, “OmniGS: Fast radiance field reconstruction using omnidirectional Gaussian splatting,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025.

[57] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 586–595.

[58] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma et al., “The replica dataset: A digital replica of indoor spaces,” arXiv preprint arXiv:1906.05797, 2019.

[59] Y. Tao, M. A. Mu <sup>´</sup> noz-Ba ˜ n˜ on, L. Zhang, J. Wang, L. F. T. Fu, and´ M. Fallon, “The oxford spires dataset: Benchmarking large-scale lidarvisual localisation, reconstruction and radiance field methods,” arXiv preprint arXiv:2411.10546, 2024.

[60] F. Kong, X. Liu, B. Tang, J. Lin, Y. Ren, Y. Cai, F. Zhu, N. Chen, and F. Zhang, “Marsim: A light-weight point-realistic simulator for lidarbased uavs,” IEEE Robotics and Automation Letters, vol. 8, no. 5, pp. 2954–2961, 2023.

[61] J. Liu and H. Chen, “Towards large-scale incremental dense mapping using robot-centric implicit neural representation,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 4045–4051.

[62] W. E. Lorensen and H. E. Cline, “Marching cubes: A high resolution 3d surface construction algorithm,” ACM Siggraph Computer Graphics, vol. 21, no. 4, pp. 163–169, 1987.