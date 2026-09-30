# PowerSim: Diferentiable Physics Simulation and Rendering with Power Diagrams

Trong-Tung Nguyen Anand Bhattad

Johns Hopkins University

Project Website: https://power-sim.github.io

![](images/675534a73617862612fb47e7eecd0f8a87dcf158dc16956cc64e51a0097c2dfa.jpg)  
Figure 1: PowerSim unifies MPM simulation and rendering with bounded power diagrams in an end-toend differentiable pipeline. (a) Objects from independent captures can be selected and composed into a scene. (b) Primitive geometry and appearance evolve under applied forces. (c) Spatially varying material properties can be estimated from video or assigned to control deformation. (d) Ray-traced reflections follow the object’s motion and deformation.

## Abstract

We introduce PowerSim, a method to bring physically grounded, differentiable dynamics to PowerFoam’s power diagram based 3D representation. PowerSim directly couples a pre-trained PowerFoam scene to the Material Point Method (MPM) by exploiting a natural alignment between the two: the geometric and appearance properties of each primitive correspond closely to the quantities MPM already tracks as an object deforms. Consequently, simulated motion can drive the scene’s geometry and appearance directly, without an auxiliary representation in between. Built on this framework, we enable a range of applications on real and synthetic scenes: (1) simulating a static scene under user interaction, (2) recovering spatially varying material fields, (3) compositing primitives from independently captured scenes into a single simulation-ready scene and (4) ray-tracing reflections that update consistently as the object deforms. Our results suggest that PowerSim excels over previous frameworks for physically grounded dynamics, while unlocking unique advantages—such as secondary ray lighting effects on dynamic scenes. Results are best viewed on our project website: https://power-sim.github.io/

## 1 Introduction

Radiance field methods have made 3D scene capture routine (Mildenhall et al., 2021; Kerbl et al., 2023). A handful of images are enough to render it from novel viewpoints. But without temporal data, they only give us one instant in time, and many interesting questions about a scene require us to infer what might happen next. For example, what would happen if we replaced the vase on the table with a plant and pushed it? How would its reflection in a mirror change (Fig. 1)? What would happen if we pulled a loaf of bread apart (Fig. 2)? To answer these questions, we need a simulation that preserves the visual fidelity of the captured scene. To infer physical parameters from observed motion, we also need to make this simulation-and-rendering pipeline differentiable.

However, simulation and rendering place different demands on a scene representation. Continuum simulation requires material volumes and a treatment of contact and separation, while rendering requires geometry and appearance sufficient to generate images. Mesh-based pipelines start from explicit geometry for simulation and attach an appearance model to it (Mai et al., 2026; Held et al., 2026). Gaussian-based pipelines start from a representation designed for rendering and adapt it for simulation using the Material Point Method (MPM) (Xie et al., 2023).

But Gaussian mixtures do not explicitly specify non-intersecting volumes of materials, an explicit boundary surface or adjacency of primitives. PhysGaussian provides optional regularization of Gaussian anisotropy and filling of interior regions of objects with additional particles, which, however, do not lead to establishing explicit boundaries between primitives. Thus, specifying material volumes and masses, contacts, and refractions implies making some extra assumptions or using additional constructions beyond the Gaussian representation itself (Moenne-Loccoz et al., 2024).

PowerFoam (Govindarajan et al., 2026) reconstructs a scene as bounded power-diagram cells with explicit boundaries and neighbors, providing geometry for both simulation and rendering. But deforming a partition is harder than moving a set of points. Moving one site changes neighboring cell boundaries, and large deformations can change cell adjacency. The challenge is to update the primitives and their neighborhoods together as the object deforms.

To address this, we introduce PowerSim, which treats each PowerFoam primitive as an MPM material point. MPM updates its position, while the deformation gradient supplies a rotation for its dipole plane and appearance directions, and an isotropic radius scale that matches the local volume change. We rebuild adjacency as the primitives move and change size, keeping cell boundaries consistent with the updated representation. The result is a differentiable pipeline from forces to pixels: we can simulate a captured scene under new forces and backpropagate through simulation and rendering to estimate spatially varying material properties from monocular deformation videos. PowerSim improves average dynamic reconstruction quality and Young’s modulus estimation over the compared methods, and preserves more surface detail under large stretching and twisting deformations (Fig. 2). We also demonstrate elastic motion, compression, granular collapse, and multi-object interactions with reflections that follow the simulated motion.

In summary, our contributions are:

1. PowerSim, a differentiable framework coupling PowerFoam with MPM through updates to primitive geometry, appearance, and adjacency, supporting large deformation and separation.

2. Estimation of spatially varying material fields from monocular deformation videos through differentiable simulation and rendering.

3. Optimization-free selection of object primitives from multi-view masks, enabling scene composition and multi-object simulation with ray-traced reflections.

## 2 Related Work

Physics-based scene simulation. A growing line of work couples reconstructed scenes to physics solvers. PhysGaussian (Xie et al., 2023) treats 3D Gaussians as MPM particles, using the same primitives for simulation and rendering. PhysDreamer (Zhang et al., 2024), DreamPhysics (Huang et al., 2024), and Physics3D (Liu et al., 2024) estimate material properties using generated videos or video diffusion priors. Spring-Gaus (Zhong et al., 2024) instead fits a spring-mass model to observed videos. PhysTwin (Jiang et al., 2025) reconstructs deformable objects and estimates dense physical properties from sparse interaction videos using spring-mass physics and Gaussian rendering. Meanwhile, GIC (Cai et al., 2024) estimates physical properties using a Gaussian-informed continuum, with surface and silhouette supervision, and NeuMA (Cao et al., 2024) learns corrections to material models and uses Particle-GS to propagate image gradients into the simulator. Other representations include PAC-NeRF’s voxel–particle coupling (Li et al., 2023) and PhysConvex’s deformable convex primitives with reduced-order simulation (Wang et al., 2026). Pixie (Le et al., 2025) predicts material fields from visual features, while Vid2Sim (Chen et al., 2025) combines feed-forward reconstruction with optimization. PowerSim couples bounded power-diagram cells to MPM, updating their geometry, appearance, and adjacency during deformation.

![](images/f2f1c6d1d3567a5ffb54b70cb16d750cf7a27b72f8c9b3ab56b4a5ac1e9f8070.jpg)  
Figure 2: PowerSim partitions the reconstructed object volume into explicit cells, while PhysGaussian (Xie et al., 2023) uses Gaussian kernels that tend to concentrate near surfaces and optionally adds interior particles. Under extreme stretching with the same applied force, PAC-NeRF loses texture and develops abrupt cut-like boundaries, while PhysGaussian exhibits pronounced blurring in stretched regions. PowerSim retains more surface detail and sharper boundaries during separation.

3D primitives selection. Selecting the subset of representation primitives that belong to an object is a prerequisite task for editing or simulating reconstructed scenes. A common strategy attaches a learnable semantic attribute to each primitive and optimizes it through differentiable rendering against 2D supervision from foundation models such as SAM (Kirillov et al., 2023). Gaussian Grouping (Ye et al., 2024) augments each Gaussian with an identity encoding that is rendered and classified by a linear layer supervised with view consistent SAM masks. SAGA (Cen et al., 2025) distills SAM features into per-Gaussian affinity features for prompt-based selection, and LabelGS (Zhang et al., 2025) lifts multi-class pixel labels to individual Gaussians. SemanticFoam (Sharafeldin et al., 2026) addresses this problem on a different representation RadiantFoam (Govindarajan et al., 2025) by learning per-cell identity encodings and regularizing them with a total-variation loss over the Voronoi adjacency graph (Govindarajan et al., 2025). Beyond methods that learn optimized semantic fields, FlashSplat (Shen et al., 2024) derives Gaussian labels using a closed-form solution, while masked-gradient voting (Joseph et al., 2024) and gradient-weighted back-projection (Joseph et al., 2025)

transfer 2D masks and features without training per scene. In contrast, we propose render-weighted voting for PowerFoam cells, label propagation for unseen primitives by adjacency between cells in a training-free manner rather than optimization-based like SemanticFoam (Sharafeldin et al., 2026), and perform simulation using the selected primitives for scene composition.

Rendering and scene representations. 3DGRT and 3DGUT support ray tracing and secondary-ray effects with Gaussian representations (Moenne-Loccoz et al., 2024; Wu et al., 2025). PowerFoam (Govindarajan et al., 2026) supports both rasterization and ray tracing using bounded power-diagram cells. We build on this representation to simulate captured objects, estimate their material properties, and render reflections that follow their deformation.

## 3 Preliminaries

PowerFoam (Govindarajan et al., 2026) represents a scene as bounded power-diagram cells. Each primitive has a center $\mathbf { p } _ { i } \in \mathbb { R } ^ { 3 }$ and a scalar radius $\mathbf { r } _ { i } ,$ which define its power cell:

$$
\mathbf { P } _ { i } = \left\{ \mathbf { x } \in \mathbb { R } ^ { 3 } \mid \left\| \mathbf { x } - \mathbf { p } _ { i } \right\| ^ { 2 } - \mathbf { r } _ { i } ^ { 2 } \leq \left\| \mathbf { x } - \mathbf { p } _ { j } \right\| ^ { 2 } - \mathbf { r } _ { j } ^ { 2 } , \forall j \right\} .\tag{1}
$$

The cell is bounded by intersecting it with the ball $\mathbf { B } _ { i } = \mathbf { B } ( \mathbf { p } _ { i } , \mathbf { r } _ { i } )$ . These cells provide explicit boundaries and neighbor relations.

Geometry. Each primitive contains a dipole plane through p<sub>i</sub>, with a material side of density $\sigma _ { i }$ and an empty side. A quaternion $\mathbf { q } _ { i }$ defines the plane’s orthonormal frame: normal $\mathbf { n } _ { i }$ and tangents $\mathbf { t } _ { i } , \mathbf { b } _ { i }$ . To represent surface detail, each plane carries K detail sites with local in-plane coordinates $\mathbf { s } _ { i , k } \in \mathbb { R } ^ { 2 }$ and normal displacements $d _ { i , k } \in \mathbb { R }$ . Their world positions are

$$
\mathbf { x } _ { i , k } = \mathbf { p } _ { i } + \mathbf { r } _ { i } \big ( \mathbf { s } _ { i , k } ^ { 1 } \mathbf { t } _ { i } + \mathbf { s } _ { i , k } ^ { 2 } \mathbf { b } _ { i } \big ) + d _ { i , k } \mathbf { n } _ { i } .\tag{2}
$$

Interpolating the displacements gives a height field over the plane.

Appearance. Each detail site stores L world-space appearance directions $\alpha _ { i , k } ^ { ( l ) }$ and corresponding RGB colors $\mathbf { c } _ { i , k } ^ { ( l ) }$ . The viewing direction determines how these colors are blended; a second interpolation blends colors across detail sites. We give both interpolation rules in Appendix A.2.

We write the reconstructed rest scene as $S _ { 0 }$ , with image $I = \mathcal { R } ( S _ { 0 } )$ . For each primitive, ${ \bf s } _ { i } , d _ { i } ,$ , and $\alpha _ { i }$ collect its detail-site coordinates, displacements, and appearance directions. During simulation, we update $\{ \mathbf { p } _ { i } ^ { t } , \mathbf { r } _ { i } ^ { t } , \mathbf { q } _ { i } ^ { t } , \alpha _ { i } ^ { t } \}$ and keep the remaining attributes fixed. The local detail sites follow the primitive through Eq. 2, while the world-space appearance directions must be rotated explicitly.

Material Point Method (MPM) (Jiang et al., 2016) simulates a continuum body using particles and a background grid. Each particle carries position $\mathbf { x } _ { p } ,$ velocity $\mathbf { v } _ { p } ,$ , mass $m _ { p } ,$ , deformation gradient $\mathbf { F } _ { p } ,$ and an affine velocity matrix $\mathbf { C } _ { p } .$ . Its material parameters are $\theta _ { p } = ( E _ { p } , \nu _ { p } )$ , where $E _ { p }$ is Young’s modulus and $\nu _ { p }$ is Poisson’s ratio. The deformation gradient tracks local rotation, stretch, and shear and determines stress through the material’s constitutive model. Each step transfers particle mass and momentum to the grid, updates grid velocities under internal and external forces, and transfers the result back to the particles. Writing the simulator as $\mathcal { M } _ { \theta } ,$ , one substep gives

$$
\begin{array} { r } { \left( { \bf x } ^ { t + 1 } , { \bf v } ^ { t + 1 } , { \bf F } ^ { t + 1 } , { \bf C } ^ { t + 1 } \right) = \mathcal { M } _ { \theta } \left( { \bf x } ^ { t } , { \bf v } ^ { t } , { \bf F } ^ { t } , { \bf C } ^ { t } \right) . } \end{array}\tag{3}
$$

The full simulation loop is given in Appendix A.

## 4 Method

Given a pretrained PowerFoam scene, we update its primitives using MPM simulation output (§4.1; Fig. 3). We differentiate through simulation, primitive updates, and rendering to estimate spatially varying material properties from monocular deformation videos (§4.2). We also describe an optimization-free method to select simulation-ready object primitives from multi-view masks (§4.3).

![](images/7bd800807e0e9f48f819567a4d6805b06a3ba42141a0892276c23103aefed392.jpg)  
Figure 3: PowerSim overview. After N MPM substeps, we update primitive positions, isotropically scale their radii to match local volume change, and rotate their dipole frames and appearance directions. We then rebuild adjacency for the updated primitives. Insets show two selected primitives before and after deformation.

## 4.1 PowerSim: Physics-grounded PowerFoam Simulation with MPM

A pretrained PowerFoam scene is purely static: every primitive carries geometry and appearance but no physical quantities, since nothing in photometric reconstruction needs them. In this work, we turn these primitives into continuum dynamic bodies by treating each primitive as an MPM particle and augmenting each with exactly the physical state MPM requires. Each augmented primitive thus carries geometry, appearance, and physical properties at once, and as it moves, all three must remain consistent with the physics-grounded simulation. The physical state ${ \bf X } _ { t } = \{ { \bf x } _ { i } ^ { t } , { \bf v } _ { i } ^ { t } , { \bf F } _ { i } ^ { t } , { \bf C } _ { i } ^ { t } \}$ is advanced by MPM itself as discussed in $\operatorname { E q } . 3 ;$ what remains is to propagate these physical state update to the primitive channels. As shown in Fig.3, we derive an update operator $\mathcal { U }$ that, given MPM’s simulation output at time $t ,$ maps a primitive’s rest-state channels to their deformed versions:

$$
\mathcal { U } : \ \big ( \mathbf { p } _ { i } ^ { 0 } , \ \mathbf { r } _ { i } ^ { 0 } , \ \mathbf { q } _ { i } ^ { 0 } , \ \alpha _ { i } ^ { 0 } \big ) \ \longmapsto \ \big ( \mathbf { p } _ { i } ^ { t } , \ \mathbf { r } _ { i } ^ { t } , \ \mathbf { q } _ { i } ^ { t } , \ \alpha _ { i } ^ { t } \big ) ,\tag{4}
$$

Volume and Mass. We partition the MPM background grid into voxels of side length $\Delta x$ and divide each occupied voxel’s volume $\Delta x ^ { 3 }$ equally among the particles it contains to obtain the rest volume $V _ { i } ^ { 0 }$ . Particle’s mass is then $m _ { i } = \rho V _ { i } ^ { 0 }$ for a uniform density $\rho$ per scene. $V _ { i } ^ { 0 }$ is computed once at rest, and subsequent volume change is carried by $J = \operatorname* { d e t } ( \mathbf { F } )$ .

Primal Sites Update. We let primitives’ centers evolve by transforming their rest position $\mathbf { p } _ { i } ^ { 0 }$ to MPM’s own coordinate frame, $\mathbf { x } _ { i } ^ { 0 } = \Phi ( \mathbf { p } _ { i } ^ { 0 } )$ ; from there, MPM’s loop runs through several $N$ substeps to evolve $\mathbf { x } _ { i } ^ { t }$ directly, and we recover the primitive’s world-space position at any time via the inverse map $\mathbf { p } _ { i } ^ { t } = \Phi ^ { - 1 } ( \mathbf { x } _ { i } ^ { t } )$ Every other primitive property updates differently, by leveraging the local deformation information MPM tracks throughout the simulation: the deformation gradient $\mathbf { F } _ { i : } ^ { t }$ , representing the primitive’s accumulated deformation relative to its rest state. Physically, $\mathbf { F } _ { i } ^ { t }$ encodes a mix of rotation, stretch, and shear accumulated in the material around primitive i. We first apply polar decomposition to $\mathbf { F } _ { i } ^ { t }$ to obtain its rotation and stretch components:

$$
\mathbf { F } _ { i } ^ { t } = \mathbf { R } _ { i } ^ { t } \mathbf { S } _ { i } ^ { t } , \qquad \mathbf { F } _ { i } ^ { t } = \mathbf { U } _ { i } \boldsymbol { \Sigma } _ { i } \mathbf { V } _ { i } ^ { \top } , \ \mathbf { R } _ { i } ^ { t } = \mathbf { U } _ { i } \mathbf { V } _ { i } ^ { \top } ,\tag{5}
$$

Here $\mathbf { R } _ { i } ^ { t }$ is a rigid rotation and $\mathbf { S } _ { i } ^ { t }$ a symmetric stretch, with singular values $\Sigma _ { i } = \mathrm { d i a g } ( \sigma _ { i } ^ { ( 1 ) } , \sigma _ { i } ^ { ( 2 ) } , \sigma _ { i } ^ { ( 3 ) } )$ ). Next, we map $\mathbf { R } _ { i } ^ { t }$ and $\mathbf { S } _ { i } ^ { t }$ onto PowerFoam’s remaining primitive channels.

Power Radii Update. PowerFoam’s scalar radius has no anisotropic degree of freedom, so we project $\mathbf { S } _ { i } ^ { t }$ onto the isotropic scale that reproduces the same local volume change:

$$
\bar { \sigma } _ { i } ^ { t } = \big ( \sigma _ { i } ^ { ( 1 ) } \sigma _ { i } ^ { ( 2 ) } \sigma _ { i } ^ { ( 3 ) } \big ) ^ { 1 / 3 } , \qquad { \bf r } _ { i } ^ { t } = { \bf r } _ { i } ^ { ( 0 ) } \cdot \bar { \sigma } _ { i } ^ { t } .\tag{6}
$$

In this way, we deliberately leave each individual primitive to stay isotropic and compact, anisotropy arises from the developing geometry of the power diagram. On the other hand, the use of anisotropic 3DGS may lead to absorbing deformation through stretching primitives, thus resulting in very elongated primitives and artifacts in the case of significant deformation as shown in Fig.2. PowerSim, however, offloads the problem of anisotropic shape changes to the geometry of neighboring cells, thus making the representation possible to be significantly deformed without elongating the primitives.

Young Modulus (logE)  
Branch is too flexible, swaying aggressively  
![](images/7935820dec0ed977f7572f46a3d66aa7a38c152b9baec825345ff82516dfffc5.jpg)  
Figure 4: Material estimation changes the simulated response. Under the same applied force, random material assignment produces large stem bending, while PhysDreamer produces little bending. PowerSim produces back-and-forth motion with visible stem deformation. The left column shows Young’s modulus; the remaining columns show matching simulation times.

Dipole Planes Update. Dipole plane’s normal and local reference frame $\left( \mathbf { n } _ { i } , \mathbf { t } _ { i } , \mathbf { b } _ { i } \right)$ are parameterized by a single quaternion ${ \bf q } _ { i } ^ { 0 }$ . We update ${ \bf q } _ { i } ^ { 0 }$ correspondingly with the rotation $\mathbf { R } _ { i } ^ { t }$ obtained from the polar decomposition, since this naturally rotates the primitive’s geometry to the corresponding angle without disturbing the relative orthogonality of $\mathbf { n } _ { i } , \mathbf { t } _ { i } , \mathbf { b } _ { i }$ . The quaternion update is achieved via

$$
\mathbf { q } _ { i } ^ { t } = \mathbf { q } _ { i } ^ { 0 } \otimes \Delta \mathbf { q } _ { i } ^ { t } , \qquad \Delta \mathbf { q } _ { i } ^ { t } = \mathrm { q u a t } \big ( \mathbf { R } _ { i } ^ { t ^ { \top } } \big ) ,\tag{7}
$$

composed with the rest orientation ${ \bf q } _ { i } ^ { 0 }$ as the left operand. Note that the normal and tangents are encoded as row vectors of the rotation matrix, which is what leads to the order of multiplication and to composing with the inverse (transpose) of $\mathbf { R } _ { i } ^ { t }$

Detail Sites Update. From §3, each dipole plane carries K detail sites with local coordinates $\mathbf { s } _ { i } ,$ displacement $d _ { i } ,$ and radiance axis $\pmb { \alpha } _ { i }$ . The first two are expressed relative to the primal site’s position, radius, and frame, so they need no update: re-evaluating Eq. 2 against the deformed primal site places them correctly for free. The radiance axis is a world-space direction and inherits nothing from the frame, so we update it explicitly, rotating the axis while holding its color coefficients fixed. We require that a co-rotating viewer sees no change: for every world direction d, the shading response must match what the rest-pose primitive produces for the back-rotated direction $\mathbf { R } _ { i } ^ { t \top } \mathbf { d }$

$$
\left\| \mathbf { d } - \boldsymbol { \alpha } _ { i } ^ { t } \right\| _ { 2 } = \left\| \mathbf { R } _ { i } ^ { t ^ { \top } } \mathbf { d } - \boldsymbol { \alpha } _ { i } ^ { 0 } \right\| _ { 2 } ,\tag{8}
$$

which, by orthogonality of $\mathbf { R } _ { i } ^ { t }$ , gives

$$
\begin{array} { r } { { \bf \alpha } \alpha _ { i } ^ { t } = { \bf R } _ { i } ^ { t } \alpha _ { i } ^ { 0 } . } \end{array}\tag{9}
$$

Topology Update. Since primitives change every step, neighbor relations are created and broken as the object deforms, and a stale topology would clip cells against the wrong neighbors. We therefore rebuild the adjacency after every update as the Cech complex of the primitives’ bounding spheres, which is the same<sup>ˇ</sup> structure PowerFoam already uses for rendering.

## 4.2 Material Field Optimization

Coupled with MPM, PowerFoam primitives evolve under physically grounded rules; what remains is to determine the material that drives them. Their effect is not subtle: Fig. 4 contrasts the unrealistic motion produced by a wrong assignment (top row) with the plausible deformation obtained from correct parameters (third row). Manually assigning them, however, demands expert knowledge, does not scale beyond a handful of hand-tuned demo scenes, and is simply unavailable for objects reconstructed from video alone. We instead recover them directly via optimization from an observed video of the object deforming, after which new forces can be applied for new simulation.

Problem setup. Given a reference video $\{ I _ { t } \} _ { t = 1 } ^ { T }$ of an object in motion, which is captured directly or synthesized by a video generation model, our goal is to estimate a spatial material field, $\pmb { \theta } _ { i } = \{ \theta _ { i } \} _ { i = 1 } ^ { N }$ , where $\theta _ { i } = ( E _ { i } , \nu _ { i } )$ indicates the Young’s modulus and Poisson’s ratio, such that PowerSim’s simulated results reproduce the observed motion. Because the impulse setting the object in motion is generally unobserved, we optimize each primitive’s initial velocity $\mathbf { v } _ { i } ^ { 0 }$ alongside the material.

Optimization. At each training iteration, we run the full forward pipeline – MPM simulation, our primitivechannel update, and rendering – to obtain a sequence of simulated frames. Here we abuse t to index video frames rather than substeps: advancing from frame t to t+1 takes $N _ { \mathrm { s u b } } = \Delta t _ { \mathrm { f r a m e } } / \Delta t _ { \mathrm { s u b } }$ MPM substeps, so one simulated frame of the pipeline is

$$
\hat { I } _ { t + 1 } = \big ( \mathcal { R } \circ \mathcal { U } \circ \mathcal { M } _ { \theta } ^ { N _ { \mathrm { s u b } } } \big ) ( \mathbf { X } _ { t } ) ,\tag{10}
$$

where $\mathcal { M } _ { \pmb { \theta } } ^ { N _ { \mathrm { s u b } } }$ advances the physical state to $\mathbf { X } _ { t + 1 }$ , U maps it – with the rest state $S _ { 0 }$ held fixed – to the primitive state $S _ { t + 1 } = \mathcal { U } ( S _ { 0 } , { \bf X } _ { t + 1 } )$ , and R renders it. The rollout starts from rest, ${ \bf X } _ { 0 } = { \big ( } \Phi ( \mathbf { p } ^ { 0 } ) , \mathbf { v } ^ { 0 } , \mathbf { I } , \mathbf { 0 } { \big ) }$ Because $\mathcal { R } , \mathcal { U } ,$ and $\mathcal { M }$ are each differentiable, so is Eq. 10. To update neighbors, we follow similarly to PowerFoam’s training and rebuild adjacency at every frame from detached copies of the deformed sites. Then the rendering uses the original gradient-carrying sites with the updated adjacency list, neighborhood relations determine cell clipping without contributing gradients. This allows photometric losses on rendered frames to backpropagate to the unknowns. We compare each simulated frame against its reference video under:

$$
\mathcal { L } ( \hat { I } _ { t } , I _ { t } ) = ( 1 - \lambda ) \Vert \hat { I } _ { t } - I _ { t } \Vert _ { 2 } ^ { 2 } + \lambda \left( 1 - \mathrm { S S I M } ( \hat { I } _ { t } , I _ { t } ) \right) ,\tag{11}
$$

so that $\partial \mathcal { L } / \partial \pmb { \theta }$ and $\partial \mathcal { L } / \partial \mathbf { v } ^ { 0 }$ pass from the rendered pixels through R and U into the MPM rollout, and the unknowns are updated by the gradient descent. Following PhysDreamer (Zhang et al., 2024), we apply truncated backpropagation-through-time (BPTT) to avoid gradient explosion/vanishing. Rather than fitting $\pmb \theta$ and $\mathbf { v } ^ { 0 }$ jointly, we optimize in two stages, recovering the initial velocity first and the material afterwards (more details in $\ S \mathrm { A } )$

## 4.3 Dynamic Primitives Selection via Render-weighted voting

A reconstructed real-world scene rarely contains a single object: a single kitchen scene contains thousands of primitives covering different objects, yet a simulation concerns one of them. This requires identifying the simulated primitives and assigning them material properties $\pmb \theta _ { i ; }$ , leaving the rest as static geometry that is rendered but not simulated. However, curating these primitives by hand in 3D is tedious, and hence we tackle this issue with a simple method that leverages multi-view 2D segmentation masks at test-time, making it natural for picking primitives ready for simulation.

Render-weighted voting. The user specifies the object through 2D segmentation masks $\{ M _ { j } \} _ { j \in \mathcal { I } }$ over a subset J of the training views, where $M _ { j } ( u ) \in \{ 0 , 1 \}$ marks whether pixel u belongs to the object; off-the-shelf models can easily produce these from a text prompt or a click (Kirillov et al., 2023; Liu et al., 2023; Cheng et al., 2023). We seek a per-primitive label $\ell _ { i } \in \{ 0 , 1 \}$ } whose selected primitives, rendered alone, reproduce $M _ { j }$ on the annotated views and stay consistent on the rest. SemanticFoam (Sharafeldin et al., 2026) solves this problem by optimization and supervises the rendered feature against the masks; we instead read it off a trained checkpoint, since PowerFoam’s rasterizer composites any per-primitive quantity the same way it composites color. Replacing colors with scalar features $f _ { i ; }$ , the rendered feature at pixel u of view $j$ is

$$
\hat { F } _ { j } ( u ) = \sum _ { i = 1 } ^ { N } w _ { i , j } ( u ) f _ { i } ,\tag{12}
$$

(a) Original Scene  
(b) Object Removal  
(c) Selected Object  
![](images/4e64c4daeb00689423e9770e767310bfd023ff1e134a8d2220afc8d9b0ea82c8.jpg)  
(d) Simulation-Ready Scene  
Figure 5: Scene composition. Our optimization-free primitive selection enables removing the vase from a captured scene (a–b) and selecting a plant from another capture (c, green). We align and insert the selected primitives to create a scene ready for simulation (d).

where $w _ { i , j } ( u )$ is the compositing weight of primitive i along the ray through $u ;$ which is the weight with which its color reaches the pixel. A primitive’s membership then follows from how much of its rendered contribution falls inside the masks versus outside, accumulated over views:

$$
s _ { i } ^ { + } = \sum _ { j \in \mathcal { I } } \sum _ { u } w _ { i , j } ( u ) M _ { j } ( u ) , \qquad s _ { i } ^ { - } = \sum _ { j \in \mathcal { I } } \sum _ { u } w _ { i , j } ( u ) \big ( 1 - M _ { j } ( u ) \big ) ,\tag{13}
$$

We select primitive $i ,$ setting $\ell _ { i } = 1$ , when $s _ { i } ^ { + } > \beta s _ { i } ^ { - }$ , with $\beta < 1$ discounting the outside vote since masks tend to miss thin structures (We provide ablation of $\beta$ in §A.5).

Computing $\{ s _ { i } ^ { + } , s _ { i } ^ { - } \}$ never requires the weights themselves. Since Eq. 12 is linear in $f ,$ the vote of primitive i in view j is a derivative of the rendered feature image:

$$
\sum _ { u } w _ { i , j } ( u ) M _ { j } ( u ) = \frac { \partial } { \partial f _ { i } } \sum _ { u } \hat { F } _ { j } ( u ) M _ { j } ( u ) .\tag{14}
$$

We therefore render the feature image, sum it over the masked pixels, and differentiate that scalar with respect to $f \colon$ the gradient on $f _ { i }$ is exactly primitive i’s vote, so one backward pass per view yields all N votes at once, and $s _ { i } ^ { - }$ follows with $1 - M _ { j }$ . Primitives no view observes $( w _ { i , j } ( u ) = 0$ everywhere, hence $s _ { i } ^ { + } = s _ { i } ^ { - } = 0 )$ inherit the majority label of their power-diagram neighbors, using the adjacency the renderer already maintains. The procedure needs no training and no checkpoint change, and handles several objects by voting one score per class and taking the arg-max.

From selection to simulation and editing. The selected primitives become material points, receiving $\theta _ { i } - \mathsf { a }$ user-specified material or a field recovered by §4.2 — while the rest stay fixed and are rendered alongside them. The selection also supports editing. Removal deletes the selected primitives and rebuilds the power-diagram adjacency as shown in Fig.5. Insertion brings cropped primitives into a target scene of unrelated frame and scale. The two are aligned by a simple transform: rotation, translation, and scaling, which is followed by adjacency rebuilt over the merged set. Replacement is removal followed by insertion; Fig.5 shows the vase replaced.

## 5 Experimental Results

We evaluate simulation quality and material estimation against prior physics-coupled pipelines. We also demonstrate varied object dynamics (Fig. 6) and secondary-ray effects in dynamic scenes (Fig. 9). Primitive selection is evaluated in §A.5. We provide more simulation results in §A.

Benchmarks and Metrics. To assess both physics simulation and material estimation, we use Vid2Sim benchmark (Chen et al., 2025) which includes 12 objects from GSO (Google Scanned Object) (Downs et al., 2022), each dropped under gravity and simulated with FEM with known ground-truth material parameters. We evaluate dynamic reconstruction quality against the reference videos via PSNR and SSIM (Wang et al., 2004), and material estimation by the absolute error of the recovered $\log _ { 1 0 }$ E and ν.

![](images/10b660696692cc0821450870624f7c5c0e05c031c336aca85db36b63bdd71e72.jpg)  
Figure 6: Diverse Material Behaviors. Simulation results on different material settings.

Baselines. For physics simulation and material estimation, we compare against representative physicscoupled pipelines on other representations: PhysDreamer (Zhang et al., 2024) on 3DGS, and PAC-NeRF (Li et al., 2023) on NeRF. PhysDreamer extends PhysGaussian (Xie et al., 2023) with additional material optimization. For 3D segmentation, we compare against Semantic Foam (Sharafeldin et al., 2026), which requires optimization to obtain a per-primitive semantic field on the same foam family, and against the Gaussian-based Gaussian Grouping (Ye et al., 2024), and LabelGS (Zhang et al., 2025), with numbers taken from (Sharafeldin et al., 2026).

Dynamic Reconstruction Evaluation. PowerSim achieves the highest mean PSNR and SSIM across the 12 objects (Tab. 1). It leads in PSNR on 10 objects, reaching 25.73 dB on average compared with 22.06 dB for PAC-NeRF and 19.00 dB for PhysDreamer. SSIM gains are smaller, with a mean of 0.930 versus 0.924 and 0.908, respectively. Fig. 7 highlights PowerSim’s preservation of texture and object shape during motion, across both the deformable backpack and the nearly rigid Mario figure. Baselines exhibit texture blurring, shape distortion, or appearance artifacts.

Material Estimation Evaluation. PowerSim achieves the lowest mean MAE in log(E), reducing the error from 0.60 for PhysDreamer to 0.44, a 27% reduction (Tab. 2). It achieves the lowest error on seven of the 12 objects, with particularly large gains on blocks, lion, and turtle. For Poisson’s ratio, PowerSim matches PhysDreamer’s mean MAE of 0.16 at the reported precision.

![](images/c0cdb72e860e5bd47f579563f2178368f763b60410b6034fd96f5191a4021f48.jpg)  
Time Frame

Figure 7: Dynamic reconstruction under deformation and contact. PowerSim preserves the backpack’s texture and Mario’s shape as they fall and contact the ground. PAC-NeRF has blurred textures and distorted silhouettes, while PhysDreamer shows discrepancies in deformation and appearance. Reference frames are shown in the bottom row; time progresses from left to right.  
Table 1: Dynamic reconstruction on 12 objects. PowerSim achieves the highest PSNR on 10 objects and SSIM on seven. Bold and underline indicate the best and second-best results, respectively.
<table><tr><td>Metrics</td><td>Method</td><td>backpack</td><td>bell</td><td>blocks</td><td>bus</td><td>cream</td><td>elephant</td><td>grandpa</td><td>leather</td><td>lion</td><td>mario</td><td>sofa</td><td>turtle</td><td>Mean</td></tr><tr><td rowspan="3">PSNR ↑</td><td>PAC-NeRF</td><td>19.37</td><td>25.00</td><td>23.36</td><td>20.72</td><td>23.24</td><td>22.27</td><td>21.63</td><td>20.85</td><td>22.66</td><td>21.01</td><td>22.49</td><td>22.19</td><td>22.06</td></tr><tr><td>PhysDreamer</td><td>18.93</td><td>19.54</td><td>19.79</td><td>18.82</td><td>19.83</td><td>17.14</td><td>16.91</td><td>18.27</td><td>18.28</td><td>17.56</td><td>19.58</td><td>23.41</td><td>19.00</td></tr><tr><td>Ours</td><td>23.69</td><td>24.39</td><td>29.56</td><td>26.84</td><td>25.82</td><td>22.80</td><td>22.01</td><td>35.33</td><td>25.17</td><td>20.31</td><td>24.72</td><td>28.11</td><td>25.73</td></tr><tr><td rowspan="3">SSIM ↑</td><td>PAC-NeRF</td><td>0.887</td><td>0.956</td><td>0.940</td><td>0.908</td><td>0.893</td><td>0.922</td><td>0.939</td><td>0.932</td><td>0.936</td><td>0.921</td><td>0.926</td><td>0.923</td><td>0.924</td></tr><tr><td>PhysDreamer</td><td>0.856</td><td>0.940</td><td>0.917</td><td>0.909</td><td>0.876</td><td>0.892</td><td>0.922</td><td>0.945</td><td>0.910</td><td>0.901</td><td>0.894</td><td>0.939</td><td>0.908</td></tr><tr><td>Ours</td><td>0.889</td><td>0.944</td><td>0.953</td><td>0.938</td><td>0.923</td><td>0.908</td><td>0.889</td><td>0.978</td><td>0.930</td><td>0.941</td><td>0.918</td><td>0.953</td><td>0.930</td></tr></table>

![](images/0a82f68eb89f4a3c8865e49f049b97ea3c4c104aaa754ca45108a4d76a54f595.jpg)  
Figure 8: Geometry and appearance updates under large deformation. Fixing dipole orientations produces ragged boundaries, while fixing appearance directions introduces color inconsistencies. The full PowerSim update preserves cleaner surface detail; PhysGaussian shows blurring and fragmentation in stretched regions. Insets highlight these differences. Best viewed on supplementary webpage.

Ablation Studies. We test role of geometry and appearance updates under large twisting and upward pulling (Fig. 8). Keeping dipole orientations fixed produces ragged boundaries and protruding surface fragments. Keeping appearance directions fixed introduces color inconsistencies as the surface rotates. Full update preserves cleaner boundaries and more consistent surface appearance throughout the motion. Under the same loading conditions, PhysGaussian shows blurred textures and fragmented stretched regions.

Table 2: Material estimation on 12 objects. MAE in log Young’s modulus and Poisson’s ratio (lower is better). Bold and underline indicate the best and second-best results, respectively.
<table><tr><td>Metrics</td><td>Method</td><td>backpack</td><td>bell</td><td>blocks</td><td>bus</td><td>cream</td><td>elephant</td><td>grandpa</td><td>leather</td><td>lion</td><td>mario</td><td>sofa</td><td>turtle</td><td>Mean</td></tr><tr><td rowspan="3">log(E)</td><td>PAC-NeRF</td><td>3.28</td><td>1.08</td><td>4.02</td><td>3.30</td><td>3.22</td><td>3.05</td><td>2.99</td><td>1.20</td><td>2.34</td><td>3.37</td><td>0.20</td><td>1.94</td><td>2.50</td></tr><tr><td>PhysDreamer</td><td>0.10</td><td>1.07</td><td>0.74</td><td>0.40</td><td>0.28</td><td>0.54</td><td>0.43</td><td>0.33</td><td>1.01</td><td>1.18</td><td>0.68</td><td>0.44</td><td>0.60</td></tr><tr><td>Ours</td><td>0.06</td><td>0.81</td><td>0.16</td><td>0.60</td><td>0.37</td><td>0.36</td><td>0.47</td><td>1.21</td><td>0.20</td><td>0.60</td><td>0.33</td><td>0.09</td><td>0.44</td></tr><tr><td rowspan="3">ν</td><td>PAC-NeRF</td><td>0.21</td><td>0.23</td><td>0.33</td><td>0.16</td><td>0.12</td><td>0.06</td><td>0.36</td><td>0.26</td><td>0.14</td><td>0.33</td><td>0.30</td><td>0.01</td><td>0.21</td></tr><tr><td>PhysDreamer</td><td>0.26</td><td>0.01</td><td>0.10</td><td>0.14</td><td>0.14</td><td>0.04</td><td>0.33</td><td>0.11</td><td>0.06</td><td>0.44</td><td>0.05</td><td>0.29</td><td>0.16</td></tr><tr><td>Ours</td><td>0.17</td><td>0.18</td><td>0.16</td><td>0.12</td><td>0.17</td><td>0.27</td><td>0.01</td><td>0.11</td><td>0.16</td><td>0.25</td><td>0.09</td><td>0.21</td><td>0.16</td></tr></table>

![](images/4e79f80cc83a18257b4e8657847b12be4a6557ea2f19ecd43f84a579e8ad1261.jpg)  
Figure 9: Ray tracing on dynamic scenes. Independently captured objects are composited and simulated together, with mirror reflections that follow their motion and deformation.

![](images/656c4fe3e56f26f83a50281453b540fa688609a31e117ed26f741125d7d03b15.jpg)  
Figure 10: Disocclusion artifacts. PowerSim can select and move the telephone handset, but its motion exposes poorly reconstructed regions, revealing gaps and visual artifacts.

Additional Applications. With PowerSim, users can combine multiple objects into a single background scene and simulate all at once. Additionally, PowerSim supports both ray tracing and rasterization, opening up interesting application for physics simulation under complex secondary ray tracing effects. As demonstrated in Fig. 9, our framework successfully handles multi-object interactions, capturing dynamic physics simulations and their reflections in a mirror environment.

## 6 Discussion

PowerSim shows how a structured scene representation can support both physics simulation and rendering as objects deform. Updating primitive geometry, appearance and adjacency allows captured scenes to respond to new forces while retaining surface detail and supporting dynamic reflections. Differentiation through simulation and rendering also enables material estimation from deformation videos. Our coupling remains an approximation. Scalar radii capture isotropic scale changes, while dipole frames and appearance directions follow local rotation; individual primitives do not reproduce the full stretch and shear of the continuum. Although neighboring cells change shape as their sites move, this does not guarantee exact agreement between cell volumes and simulated material volumes. A richer deformation model could improve this correspondence. The results also depend on the captured geometry. Our selection method can isolate parts such as a telephone handset without optimization, but moving them may expose regions poorly observed during capture, revealing gaps and artifacts (Fig. 10). Generative priors could help complete these regions, which we leave to future work. Material estimation has a related ambiguity: different combinations of stiffness, initial velocity, and physical assumptions can explain similar image motion. The recovered parameters should therefore be interpreted under the assumed density, scale, and boundary conditions.

## AI use statement

In this work, we used generative AI tools for polishing and translating the manuscript text to English, producing initial drafts of some passages, condensing related literature, and configuring the software environments. The core research, including the research questions, the coupling between PowerFoam and MPM, the experimental design, and the implementation of PowerSim, was carried out by the authors without AI assistance; theoretical and proof-related uses do not apply to this paper. Every AI-assisted contribution was checked by the authors: generated text was revised by hand, environment setup scripts were validated by executing them and examining their outputs, and literature summaries were compared against the cited papers. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## References

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5460–5469. IEEE, 2022.

Junhao Cai, Yuji Yang, Weihao Yuan, Yisheng HE, Zilong Dong, Liefeng Bo, Hui Cheng, and Qifeng Chen. GIC: Gaussian-informed continuum for physical property identification and simulation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum? id=SSCtCq2MH2.

Junyi Cao, Shanyan Guan, Yanhao Ge, Wei Li, Xiaokang Yang, and Chao Ma. NeuMA: Neural material adaptor for visual grounding of intrinsic dynamics. In The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Segment any 3d gaussians. In Proceedings of the AAAI conference on artificial intelligence, volume 39, pp. 1971–1979, 2025.

Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16123–16133, 2022.

Chuhao Chen, Zhiyang Dou, Chen Wang, Yiming Huang, Anjun Chen, Qiao Feng, Jiatao Gu, and Lingjie Liu. Vid2sim: Generalizable, video-based reconstruction of appearance, geometry and physics for mesh-free simulation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26545–26555, 2025.

Yangming Cheng, Liulei Li, Yuanyou Xu, Xiaodi Li, Zongxin Yang, Wenguan Wang, and Yi Yang. Segment and track anything. arXiv preprint arXiv:2305.06558, 2023.

Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3d scanned household items. In 2022 international conference on robotics and automation (ICRA), pp. 2553–2560. Ieee, 2022.

Shrisudhan Govindarajan, Daniel Rebain, Kwang Moo Yi, and Andrea Tagliasacchi. Radiant foam: Real-time differentiable ray tracing. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4135–4145. IEEE, 2025.

Shrisudhan Govindarajan, Daniel Rebain, Dor Verbin, Kwang Moo Yi, Anish Prabhu, and Andrea Tagliasacchi. Power foam: Unifying real-time differentiable ray tracing and rasterization. arXiv, 2026.

Jan Held, Sanghyun Son, Renaud Vandeghen, Daniel Rebain, Matheus Gadelha, Yi Zhou, Anthony Cioppa, Ming C. Lin, Marc Van Droogenbroeck, and Andrea Tagliasacchi. MeshSplatting: Differentiable Rendering with Opaque Meshes. In Computer Vision and Pattern Recognition (CVPR), 2026. URL https://arxiv.org/ abs/2512.06818. oral.

Tianyu Huang, Yihan Zeng, Hui Li, Wangmeng Zuo, and Rynson WH Lau. Dreamphysics: Learning physical properties of dynamic 3d gaussians with video diffusion priors. arXiv preprint arXiv:2406.01476, 2024.

Chenfanfu Jiang, Craig Schroeder, Andrew Selle, Joseph Teran, and Alexey Stomakhin. The affine particle-incell method. ACM Transactions on Graphics (TOG), 34(4):1–10, 2015.

Chenfanfu Jiang, Craig Schroeder, Joseph Teran, Alexey Stomakhin, and Andrew Selle. The material point method for simulating continuum materials. In Acm siggraph 2016 courses, pp. 1–52. 2016.

Hanxiao Jiang, Hao-Yu Hsu, Kaifeng Zhang, Hsin-Ni Yu, Shenlong Wang, and Yunzhu Li. Phystwin: Physicsinformed reconstruction and simulation of deformable objects from videos. ICCV, 2025.

Joji Joseph, Bharadwaj Amrutur, and Shalabh Bhatnagar. Gradient-driven 3d segmentation and affordance transfer in gaussian splatting from 2d masks. arXiv preprint arXiv:2409.11681, 2024. URL https://arxiv. org/abs/2409.11681.

Joji Joseph, Bharadwaj Amrutur, and Shalabh Bhatnagar. Gradient-weighted feature back-projection: A fast alternative to feature distillation in 3d gaussian splatting. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, SA Conference Papers ’25, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400721373. doi: 10.1145/3757377.3763926. URL https://doi.org/10.1145/ 3757377.3763926.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023.

Long Le, Ryan Lucas, Chen Wang, Chuhao Chen, Dinesh Jayaraman, Eric Eaton, and Lingjie Liu. Pixie: Fast and generalizable supervised learning of 3d physics from pixels. arXiv preprint arXiv:2508.17437, 2025.

Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu Jiang, and Chuang Gan. Pac-nerf: Physics augmented continuum neural radiance fields for geometry-agnostic system identification. arXiv preprint arXiv:2303.05512, 2023.

Fangfu Liu, Hanyang Wang, Shunyu Yao, Shengjun Zhang, Jie Zhou, and Yueqi Duan. Physics3d: Learning physical properties of 3d gaussians via video diffusion. arXiv preprint arXiv:2406.04338, 2024.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.

Alexander Mai, Trevor Hedstrom, George Kopanas, Janne Kontkanen, Falko Kuester, and Jonathan T Barron. Radiance meshes for volumetric reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8267–8277, 2026.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99–106, 2021.

Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo De Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, and Zan Gojcic. 3d gaussian ray tracing: Fast tracing of particle scenes. ACM Transactions on Graphics (TOG), 43(6):1–19, 2024.

Amr Sharafeldin, Shrisudhan Govindarajan, Thomas Walker, Aryan Mikaeili, Daniel Rebain, Kwang Moo Yi, and Andrea Tagliasacchi. Semantic foam: Unifying spatial and semantic scene decomposition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

Qiuhong Shen, Xingyi Yang, and Xinchao Wang. Flashsplat: 2d to 3d gaussian splatting segmentation solved optimally. European Conference of Computer Vision, 2024.

Dan Wang, Xinrui Cui, Serge Belongie, and Ravi Ramamoorthi. Physconvex: Physics-informed 3d dynamic convex radiance fields for reconstruction and simulation, 2026. URL https://arxiv.org/abs/2602. 18886.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, and Zan Gojcic. 3dgut: Enabling distorted cameras and secondary rays in gaussian splatting. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26036–26046. IEEE, 2025.

Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physics-integrated 3d gaussians for generative dynamics. arXiv preprint arXiv:2311.12198, 2023.

Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. In European conference on computer vision, pp. 162–179. Springer, 2024.

Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y. Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T. Freeman. PhysDreamer: Physics-based interaction with 3d objects via video generation. In European Conference on Computer Vision. Springer, 2024.

Yupeng Zhang, Dezhi Zheng, Ping Lu, Han Zhang, Lei Wang, Liping Xiang, Cheng Luo, Kaijun Deng, Xiaowen Fu, Linlin Shen, et al. Labelgs: Label-aware 3d gaussian splatting for 3d scene segmentation. In Chinese Conference on Pattern Recognition and Computer Vision (PRCV), pp. 47–61. Springer, 2025.

Licheng Zhong, Hong-Xing Yu, Jiajun Wu, and Yunzhu Li. Reconstruction and simulation of elastic objects with spring-mass 3d gaussians. In European Conference on Computer Vision (ECCV), 2024.

## A Appendix

## A.1 Project Website

We provide link to our project website at https://power-sim.github.io/.

## A.2 PowerFoam Geometry and Appearance Details

We provide the interpolation rules used by the PowerFoam representation introduced in Section 3. These rules are inherited from PowerFoam (Govindarajan et al., 2026).

Surface interpolation. The dipole plane separates the cell into a material side opposite its normal $\mathbf { n } _ { i } ,$ with density $\pmb { \sigma } _ { i } \in \mathbb { R } _ { + }$ , and an empty side with zero density. The detail-site displacements bend this plane into a height field. For an in-plane query $\bar { \mathbf { x } } \in \mathbb { R } ^ { 2 }$ , expressed in the same radius-normalized coordinates as ${ \bf s } _ { i , k } ,$ the weight of detail site k is

$$
w _ { i , k } ( \bar { \bf x } ) = \exp ( - \tau \| \bar { \bf x } - { \bf s } _ { i , k } \| _ { 2 } ) ,\tag{15}
$$

where $\tau$ controls the sharpness of the interpolation. The interpolated normal displacement is

$$
h _ { i } ( \bar { \bf x } ) = \frac { \sum _ { k = 1 } ^ { K } w _ { i , k } ( \bar { \bf x } ) d _ { i , k } } { \sum _ { k = 1 } ^ { K } w _ { i , k } ( \bar { \bf x } ) } .\tag{16}
$$

The corresponding surface point is ${ \bf p } _ { i } + { \bf r } _ { i } ( \bar { x } ^ { 1 } { \bf t } _ { i } + \bar { x } ^ { 2 } { \bf b } _ { i } ) + h _ { i } ( \bar { x } ) { \bf n } _ { i }$ , restricted to the bounded cell.

Directional appearance. Each detail site stores L appearance directions ${ \pmb { \alpha } } _ { i , k } ^ { ( l ) } \in \mathbb { R } ^ { 3 }$ and corresponding colors $\mathbf { c } _ { i , k } ^ { ( l ) } \in \mathbb { R } ^ { 3 }$ . For a unit viewing direction d, its color is

$$
\begin{array} { r l } & { \mathbf { c } _ { i , k } ( \mathbf { d } ) = \frac { \sum _ { l = 1 } ^ { L } w _ { i , k } ^ { ( l ) } ( \mathbf { d } ) \mathbf { c } _ { i , k } ^ { ( l ) } } { \sum _ { l = 1 } ^ { L } w _ { i , k } ^ { ( l ) } ( \mathbf { d } ) } , } \\ & { w _ { i , k } ^ { ( l ) } ( \mathbf { d } ) = \exp \left( - \| \mathbf { d } - \hat { \alpha } _ { i , k } ^ { ( l ) } \| _ { 2 } \right) , } \end{array}\tag{17}
$$

where $\hat { \pmb { \alpha } } _ { i , k } ^ { ( l ) } = { \pmb { \alpha } } _ { i , k } ^ { ( l ) } / \| { \pmb { \alpha } } _ { i , k } ^ { ( l ) } \| _ { 2 }$ . We then blend the detail-site colors using the spatial weights from Eq. 15:

$$
\mathbf { c } ( \bar { \mathbf { x } } , \mathbf { d } ) = \frac { \sum _ { k = 1 } ^ { K } w _ { i , k } ( \bar { \mathbf { x } } ) \mathbf { c } _ { i , k } ( \mathbf { d } ) } { \sum _ { k = 1 } ^ { K } w _ { i , k } ( \bar { \mathbf { x } } ) } .\tag{18}
$$

Rest state and fixed attributes. For completeness, the reconstructed rest scene contains

$$
\begin{array} { r } { S _ { 0 } = \left\{ \mathbf { p } _ { i } ^ { 0 } , \mathbf { r } _ { i } ^ { 0 } , \mathbf { q } _ { i } ^ { 0 } , \mathbf { s } _ { i } , d _ { i } , \pmb { \sigma } _ { i } , \alpha _ { i } ^ { 0 } , \mathbf { c } _ { i } \right\} _ { i = 1 } ^ { N } , } \end{array}\tag{19}
$$

where $\mathbf { c } _ { i }$ collects the colors $\{ \mathbf { c } _ { i , k } ^ { ( l ) } \} _ { k , l }$ . Simulation changes the positions, radii, frames, and appearance directions. Detail-site coordinates, normal displacements, densities, and color coefficients remain fixed. In particular, Eq. 2 scales the in-plane offsets through $\mathbf { r } _ { i } ,$ while the normal displacements $d _ { i , k }$ retain their original magnitudes.

## A.3 Material Point Method Simulation Loop

In this section, we provide additional background details on MPM simulation loop. We first define Lamé coefficients as:

$$
\mu = \frac E { 2 ( 1 + \nu ) } , \qquad \lambda = \frac { E \nu } { ( 1 + \nu ) ( 1 - 2 \nu ) } ,\tag{20}
$$

Particle-to-grid (P2G) transfer. Mass and momentum are accumulated onto the grid with the affine particlein-cell (APIC) scheme (Jiang et al., 2015), augmenting each particle’s velocity with a local affine velocity term $\mathbf { C } _ { p } ^ { t }$ •

$$
m _ { i } ^ { t } = \sum _ { p } w _ { i p } ^ { t } m _ { p } ,\tag{21}
$$

$$
( m \mathbf { v } ) _ { i } ^ { t } = \sum _ { p } w _ { i p } ^ { t } m _ { p } \left[ \mathbf { v } _ { p } ^ { t } + \mathbf { C } _ { p } ^ { t } \left( \mathbf { x } _ { i } ^ { t } - \mathbf { x } _ { p } ^ { t } \right) \right] ,\tag{22}
$$

where $\boldsymbol { w _ { i p } ^ { t } }$ is the B-spline weight coupling particle $p$ to grid node i.

Grid update. Each grid node velocity is integrated forward under the net nodal force $\mathbf { f } _ { i } ,$ combining internal and external forces:

$$
\mathbf { v } _ { i } ^ { t + 1 } = \mathbf { v } _ { i } ^ { t } + \frac { \Delta t } { m _ { i } ^ { t } } \mathbf { f } _ { i } \left( \mathbf { x } _ { i } ^ { t } ; \pmb { \theta } _ { p } \right) .\tag{23}
$$

The force follows from a hyperelastic energy $\Psi ( \mathbf { F } )$ , and $\theta _ { p }$ gathers the governing material properties, Young’s modulus $E ,$ and Poisson’s ratio ν.

Grid-to-particle (G2P) transfer. The updated nodal velocities are interpolated back to the particles, whose positions are then advanced, the local affine velocity term is also updated correspondingly:

$$
\mathbf { v } _ { p } ^ { t + 1 } = \sum _ { i } w _ { i p } ^ { t } \mathbf { v } _ { i } ^ { t + 1 } , \qquad \mathbf { x } _ { p } ^ { t + 1 } = \mathbf { x } _ { p } ^ { t } + \Delta t \mathbf { v } _ { p } ^ { t + 1 } , \qquad \mathbf { C } _ { p } ^ { t + 1 } = \frac { 4 } { ( \Delta x ) ^ { 2 } } \sum _ { i } w _ { i p } ^ { t } \mathbf { v } _ { i } ^ { t + 1 } ( \mathbf { x } _ { i } - \mathbf { x } _ { p } ^ { t } ) ^ { T }\tag{24}
$$

Deformation gradient update. Each particle’s deformation gradient is updated from the velocity gradient sampled off the grid:

$$
\mathbf { F } _ { p } ^ { t + 1 } = \left[ \mathbf { I } + \Delta t \sum _ { i } \mathbf { v } _ { i } ^ { t + 1 } \left( \nabla w _ { i p } ^ { t } \right) ^ { \top } \right] \mathbf { F } _ { p } ^ { t } .\tag{25}
$$

## A.4 Additional Implementation Details

Material Field Optimization. Following PhysDreamer (Zhang et al., 2024), we represent both velocity field and material field using smooth spatial fields: a triplane (Chan et al., 2022), queried at each primitive’s rest position $ { \mathbf { p } } _ { i } ^ { 0 } .$ . For triplane representation, each field uses three $2 4 \times 2 4$ feature planes with 32 channels and a two-layer MLP decoder of width 64. The velocity field is zero-initialized so the object starts at rest. The material field, with weights $\phi ,$ outputs both components of $\pmb { \theta } _ { p } = ( E _ { p } , \nu _ { p } )$ . We optimize Young’s in log-scale space around an initial guess $E$ and clamped to a plausible range [1e2, 1e10], while Poisson’s ratio is bounded to its physical interval [0, 0.5]. We optimize with AdamW (weight decay $1 \dot { 0 } ^ { - 4 } )$ under a linear warmup over the first 10% of iterations followed by linear decay to zero, clipping the gradient norm to 1.0, with the photometric loss of Eq. 11 at $\lambda = 0 . 2$ . Stage 1 fits the velocity field for 30 iterations on the first three frames at learning rate $1 0 ^ { - 2 }$ . Stage 2 freezes the velocity and fits the material field on the full video for 20 iterations at $5 \times 1 0 ^ { - 3 }$ , initialized at $\bar { E } = 1 0 ^ { 7 } \ \mathrm { P a }$

Simulation Details. We provide a list of constitutive models that we used for simulation in each scene in Tab.3

## A.5 Additional Results

Additional Physics Simulation Results As shown in Fig.6, we provide more simulation results spanning a range of objects, materials, and interactions. Our results are best viewed on our project website: https: //powersim.github.io/.

Table 3: List of constitutive models used for simulation.
<table><tr><td>Scene</td><td>Figure</td><td>Constitutive Model</td></tr><tr><td>Bonsai</td><td>Fig. 1</td><td>Fixed corotated</td></tr><tr><td>Bread roll</td><td>Fig. 2</td><td>Fixed corotated</td></tr><tr><td>Carnation</td><td>Fig. 4</td><td>Fixed corotated</td></tr><tr><td>GSO benchmark</td><td>Fig. 7</td><td>Fixed corotated</td></tr><tr><td>Bread twist</td><td>Fig. 8</td><td>Fixed corotated</td></tr><tr><td>Car</td><td>Fig. 9</td><td>Fixed corotated</td></tr><tr><td>Can</td><td>Fig. 6</td><td>von Mises</td></tr><tr><td>Telephone</td><td>Fig. 6</td><td>Fixed corotated</td></tr><tr><td>Microphone</td><td>Fig. 6</td><td>Fixed corotated</td></tr><tr><td>Ficus</td><td>Fig. 6</td><td>Fixed corotated</td></tr><tr><td>Pillow</td><td>Fig. 6</td><td>Fixed corotated</td></tr><tr><td>Wolf</td><td>Fig. 6</td><td>Drucker-Prager</td></tr><tr><td>Paper Plane</td><td>Fig. 9</td><td>Fixed corotated</td></tr></table>

Table 4: Ablation studies on discounting factor $\beta$ with mIoU / mAcc per scene.
<table><tr><td> $\beta$ </td><td>garden</td><td>bonsai</td><td>room</td><td>counter</td><td>kitchen</td><td>average</td></tr><tr><td>0.1</td><td>0.912 / 0.948</td><td>0.837 / 0.881</td><td>0.604 / 0.953</td><td>0.758 / 0.912</td><td>0.855 / 0.947</td><td>0.793 / 0.928</td></tr><tr><td>0.2</td><td>0.916 / 0.948</td><td>0.833 / 0.877</td><td>0.608 / 0.955</td><td>0.758 / 0.912</td><td>0.855 / 0.947</td><td>0.794 / 0.928</td></tr><tr><td>0.4</td><td>0.922 / 0.948</td><td>0.826 / 0.868</td><td>0.615 / 0.953</td><td>0.759 / 0.912</td><td>0.858 / 0.947</td><td>0.796 / 0.926</td></tr><tr><td>0.5</td><td>0.920 / 0.945</td><td>0.822 / 0.863</td><td>0.619 / 0.953</td><td>0.759 / 0.912</td><td>0.859 / 0.947</td><td>0.796 / 0.924</td></tr><tr><td>0.6</td><td>0.920 / 0.944</td><td>0.819 / 0.860</td><td>0.622 / 0.952</td><td>0.759 / 0.912</td><td>0.859 / 0.947</td><td>0.796 / 0.923</td></tr><tr><td>0.8</td><td>0.917 / 0.940</td><td>0.814 / 0.853</td><td>0.625 / 0.948</td><td>0.759 / 0.912</td><td>0.859 / 0.947</td><td>0.795 / 0.920</td></tr><tr><td>1.0</td><td>0.916 / 0.938</td><td>0.810 / 0.848</td><td>0.630 / 0.939</td><td>0.759 / 0.912</td><td>0.861 / 0.947</td><td>0.795 / 0.917</td></tr></table>

3D Semantic Segmentation To evaluate the performance of our proposed optimization-free dynamic primitive selection, we follow Semantic Foam(Sharafeldin et al., 2026) and evaluate on the 3D segmentation task using five Mip-NeRF 360 (Barron et al., 2022) for fair comparison with its released object masks, reporting mean Intersection over Union (mIoU) and mean Accuracy (mAcc). Tab. 5 reports quantitative results. Without any optimization, our primitive selection is competitive with methods that train a dedicated semantic field. We additionally show objects extracted by our method alongside those of Semantic Foam’s in Fig.11.

Ablation Studies on discounting factor. Tab.4 varies the discounting factor $\beta$ from 0.1 to 1.0. We use β = 0.5 for all scenes.

Table 5: Per-scene segmentation result on Mip-NeRF 360. Baseline numbers are taken from the SemanticFoam paper. All baselines optimize a per-scene semantic field; ours requires no optimization.
<table><tr><td>Method</td><td>Opt.-free</td><td>Garden mIoU↑/mAcc↑</td><td>Bonsai mIoU↑/mAcc↑</td><td>Room mIoU↑/mAcc↑</td><td>Counter mIoU↑/mAcc↑</td><td>Kitchen mIoU↑/mAcc↑</td><td>Average mIoU↑/mAcc↑</td></tr><tr><td>LabelGS</td><td>X</td><td>0.79 / 0.95</td><td>0.70 0.92</td><td>0.64 / 0.93</td><td>0.55 / 0.93</td><td>0.82 / 0.96</td><td>0.70 / 0.94</td></tr><tr><td>Gaussian Grouping</td><td>X</td><td>0.88 / 0.92</td><td>0.75 / 0.82</td><td>0.55 / 0.78</td><td>0.74 / 0.92</td><td>0.83 / 0.92</td><td>0.75 / 0.87</td></tr><tr><td>SemanticFoam</td><td>x</td><td>0.94 / 0.96</td><td>0.90 / 0.94</td><td>0.63 / 0.95</td><td>0.75 / 0.91</td><td>0.90 / 0.94</td><td>0.82 / 0.94</td></tr><tr><td>Ours</td><td>√</td><td>0.92 / 0.95</td><td>0.82 / 0.86</td><td>0.62 / 0.95</td><td>0.76 / 0.91</td><td>0.86 0.95</td><td>0.80 / 0.92</td></tr></table>

## A.6 Assets Info

We provide more information about the assets that we use in the multi-object simulation. Other simulation scenes or objects are taken from PhysGaussian Xie et al. (2023) or PhysDreamer Zhang et al. (2024).

Paper Plane: https://sketchfab.com/3d-models/paper-plane-6e9201cd4b614d879741ea79524c57c6   
Can: https://sketchfab.com/3d-models/french-coke-can-606cf0ae5cb44fa384bf2586982f7163

![](images/f6620a49774a1881b59cb2846851fd5e7542e68a42cce0a0859c5f86c7d16bd4.jpg)  
Figure 11: We provide qualitative results compared with SemanticFoam on Mip-NeRF 360 scenes  
Car: https://superspl.at/scene/bdb92f05