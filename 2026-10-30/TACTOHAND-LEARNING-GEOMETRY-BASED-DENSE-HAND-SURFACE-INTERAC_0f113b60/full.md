# TACTOHAND: LEARNING GEOMETRY-BASED DENSE HAND-SURFACE INTERACTION FIELDS FROM VIDEO

Ziheng Xi<sup>1</sup>, Weibo Li<sup>2</sup>, Jiayi Yu<sup>1</sup>, Sipeng Zheng<sup>3</sup>, Wanpeng Zhang<sup>3</sup>,

Haiweng Xu<sup>2</sup>, Hao Luo<sup>2</sup>, Chuan Mao<sup>2</sup>, Zongqing Lu<sup>2</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>Peking University <sup>3</sup>BeingBeyond

## ABSTRACT

Understanding how humans interact with the world requires a representation of physical interactions at the hand surface. Tactile sensing captures these interactions, but collecting tactile data at scale requires specialized hardware. We introduce TactoHand, a model that predicts three dense hand-surface interaction fields—contact, proximity, and 3D force—from monocular video. We derive geometry-based contact and proximity labels and motion-derived force labels from eligible training data within a curated collection spanning approximately 50 million source frames. TactoHand uses a temporal transformer to aggregate visual evidence across frames, while three learned queries decode the corresponding hand-surface fields. A differentiable physics layer further constrains force predictions using observed object motion. Across in-domain and out-of-domain benchmarks, TactoHand improves contact and proximity estimation over prior methods. In addition, physics supervision reduces position and rotation errors in motion rollouts, and the predicted contact and proximity fields improve downstream hand–object reconstruction. Together, these results show a scalable, lowcost path to learning hand-surface interaction fields without collecting tactile measurements. Code will be released at https://github.com/zhenqis123/ Tactohand.

## 1 INTRODUCTION

Hands mediate human interactions with objects, the surrounding environment, and other hands. Although the interacting entity and its geometry vary, the hand surface provides a common reference for describing where and how these interactions occur. Contact, proximity, and force provide complementary geometric and physical cues that could support interaction reconstruction, action understanding, and embodied learning. We call these vertex-wise quantities hand-surface interaction fields: contact describes where interaction occurs, proximity captures separation from the counterpart surface, and 3D force characterizes the hand’s mechanical effect.

Prior work has modeled subsets of these hand-surface quantities. HACO predicts dense contact across diverse interaction types (Jung & Lee, 2025); InterField estimates hand–object distance (Fan et al., 2023), HOPE jointly predicts contact and normal pressure from video (Jeon et al., 2026), and EgoPHI estimates contact and normal forces from egocentric images given object geometry (Ilic et al., 2026). These works motivate a shared hand-surface representation that brings contact, proximity, and 3D vertex forces together. We therefore represent all three as dense, aligned fields on the canonical MANO hand surface.

Learning these fields from monocular video, however, requires supervision that covers diverse interaction types and physical quantities. We harmonize 24 datasets comprising around 50 million frames, spanning hand–object, hand–hand, and hand–scene interactions. We derive contact and proximity labels from the available geometry. For eligible hand–object sequences, we additionally derive vertex-force labels from observed object trajectories. Object motion constrains the aggregate force and torque, but does not uniquely determine how forces are distributed over the hand surface.

![](images/7beb552980f58b4066b54d8f4ccec671b08620c4b45c65e70e9b1632ead0f252.jpg)  
Figure 1: Overview of TactoHand. A shared video encoder extracts frame-level features, aggregated by a temporal transformer. Three learned queries decode the contact, proximity, and 3D force interaction fields at corresponding MANO vertices. Predicted vertex forces are aggregated into an object-level wrench for differentiable rigid-body supervision. Contact, proximity, force, and pose losses are applied according to the supervision available for each sample.

We therefore use an SDF-aware formulation with geometric, frictional, and temporal constraints to obtain regularized vertex-force estimates.

Inferring these fields remains challenging because contact regions are often occluded and individual frames provide limited evidence about interaction dynamics. We introduce TactoHand, which uses a temporal transformer to aggregate visual evidence across frames and three learned queries to predict contact, proximity, and 3D force fields on corresponding MANO vertices, as shown in Figure 1. To connect vertex-level force predictions with object motion, we transform the predicted vertex forces into a common coordinate frame and analytically aggregate them into object-level wrenches. For eligible rigid sequences, a differentiable dynamics layer uses these wrenches to propagate object motion and constrains force estimates through observed trajectories. This motion-level supervision complements the vertex-level targets by encouraging predicted forces to explain the resulting object motion.

We evaluate TactoHand on both interaction field prediction and downstream hand–object reconstruction. TactoHand improves contact and proximity estimation across multiple benchmarks. On HOCap, which is excluded from training, TactoHand achieves a contact Macro-F1 of 50.5, compared with 18.0 for HACO. Physics supervision further reduces position and rotation errors in referenceassisted rigid-body rollouts. In downstream hand–object reconstruction, the predicted contact and proximity fields provide geometric constraints that improve hand–object alignment and reconstruction accuracy.

Our contributions are:

• A unified representation of hand-surface interaction that jointly models dense contact, proximity, and 3D vertex forces at corresponding vertices of the MANO mesh.

• TactoHand, a model that estimates these fields from monocular video, with differentiable rigid-body supervision to constrain vertex forces.

• A heterogeneous collection of 24 hand interaction datasets, with geometry-derived contact and proximity labels and motion-derived force estimates where applicable, supporting direct field evaluation and downstream hand–object reconstruction.

## 2 RELATED WORK

Hand-surface Contact and Proximity. Grasp datasets and image-based reconstruction pipelines represent hand–object contact (Brahmbhatt et al., 2020; Tse et al., 2022; Hu et al., 2024). HACO predicts dense MANO contact across interaction types, and ContactPrompt predicts contact without training (Jung & Lee, 2025; 2026). InterField instead predicts continuous hand–object distance, while contact potential fields encode vertex relationships for pose refinement (Fan et al., 2023; Yang et al., 2021). Related full-body work models human–scene contact or implicit human–object geometry (Hassan et al., 2021; Huang et al., 2022; Tripathi et al., 2023; Xie et al., 2022). TactoHand aligns contact and proximity on MANO vertices across hand–object, hand–hand, and hand–scene data.

Visual Prediction of Pressure and Force. Visual motion has been used to estimate interaction forces (Li et al., 2022). PressureVision and EgoPressure learn from instrumented planar surfaces, whereas EgoTactile and TouchSight use glove measurements (Grady et al., 2022; Zhao et al., 2025; Zeng et al., 2026; Zhou et al., 2026). HOPE and EgoTac combine measured pressure or force with geometric contact on MANO vertices (Jeon et al., 2026; Zhang et al., 2026). EgoPHI predicts contact and normal-direction force given an egocentric image and object geometry (Ilic et al., 2026). TactoHand derives three-component vertex-force targets from object motion for sequences with suitable object trajectories.

Motion and Physical Consistency. Prior work matches simulated contact forces to observed motion or couples them with hand–object reconstruction (Ehsani et al., 2020; Hu et al., 2022); ViTaM-D adds distributed tactile sensing (Yu et al., 2025). Differentiable rigid-body models allow motion errors to supervise predicted forces (Qiao et al., 2020; Werling et al., 2021). Since motion constrains only the aggregate wrench, TactoHand regularizes its vertex-force targets and applies a referenceassisted motion loss to eligible rigid sequences.

Interaction Cues for Downstream Tasks. Contact and distance constraints aid hand–object reconstruction (Hasson et al., 2019; Grady et al., 2021; Hasson et al., 2021; Qi et al., 2024). Hand affordances and measured glove forces have also supported robot learning and physical action understanding (Srirama et al., 2024; Dessalene et al., 2026). We test downstream utility by adding predicted contact and proximity as fixed constraints to existing reconstruction pipelines.

## 3 PROBLEM FORMULATION

Let $I _ { 1 : T }$ denote a monocular RGB clip of T frames sampled at 10 Hz. Each hand is represented by the fixed MANO topology with $V = 7 7 8$ vertices (Romero et al., 2017). We index each vertex by $i = ( h , v )$ , where $\bar { h ( \in \{ L , R \} }$ denotes the hand side and $v \in \{ 1 , \ldots , V \}$ denotes the vertex index. TactoHand estimates three aligned hand-surface interaction fields on these vertices:

$$
\begin{array} { r } { \Phi _ { \theta } ( I _ { 1 : T } ) = \big \{ ( \widehat c _ { t , i } , \widehat r _ { t , i } , \widehat \ell _ { t , i } ) \big \} _ { t , i } , \quad \widehat c _ { t , i } , \widehat r _ { t , i } \in [ 0 , 1 ] , \quad \widehat \ell _ { t , i } \in { \mathbb R } ^ { 3 } . } \end{array}\tag{1}
$$

Here, $\widehat { c } _ { t , i }$ denotes the contact probability, while $\widehat { r } _ { t , i }$ measures proximity to the counterpart surface, with larger values indicating greater closeness. $\widehat { \ell } _ { t , i }$ contains raw vertex-local force components predicted from RGB. Converting the gated components into camera-frame forces $\widehat { \mathbf { f } } _ { t , i }$ additionally requires ground-truth posed MANO geometry (Equation 7). Force supervision is applied only to eligible hand–object samples. For hand–hand and hand–scene samples, force losses are masked out, while contact and proximity supervision is retained wherever valid annotations are available.

## 4 DATASET CONSTRUCTION

## 4.1 DATA SOURCES

We combine 24 sources spanning diverse hand interaction scenarios, comprising 282,874 curated sequences and 49.02 million captured frames across camera views, approximately 5.2× HACO’s frame count. For all datasets, we derive geometry-based contact and proximity labels on a shared set of MANO vertices. We additionally derive motion-based vertex-force labels for 15 hand–object datasets, including 13 with rigid objects and two with articulated objects. These are collection counts; training excludes GigaHands force targets and all HOCap data. Appendix Table 4 reports the sequence and frame counts, together with the available supervision, for each source.

## 4.2 CONTACT AND PROXIMITY LABELING

For a hand vertex p, we compute its unsigned distance u to the original object surface in the object’s coordinate frame. We then obtain the signed distance s using a multiray inside–outside test, described in Appendix A.2, retaining the distance magnitude and assigning a negative sign to vertices inside the object and a nonnegative sign otherwise. For hand–hand and hand–scene data, we query the other posed hand or the available scene surface, respectively. When the counterpart surface is open and a reliable inside–outside classification is unavailable, we use the unsigned nearest-surface distance and disable inside-mesh inclusion. The proximity target is defined as:

$$
r = \exp ( - \tau \operatorname* { m i n } \{ \operatorname* { m a x } ( s , 0 ) , d _ { \operatorname* { m a x } } \} ) .\tag{2}
$$

This target equals one on and inside the surface, decreases with separation outside the surface, and remains constant beyond $d _ { \mathrm { m a x } }$ . We use $\tau = 2 0 \mathrm { { m } ^ { - 1 } }$ and $d _ { \mathrm { m a x } } = 0 . 2 5 \mathbf { \mathrm { I } }$ m. We assign contact labels using the following rule.

Contact Decision. Let $\delta _ { D }$ be the datasetspecific distance threshold, q the closest point on the counterpart surface, and n the outward handsurface normal. For an outside vertex with $u > 0 ,$ let $\kappa = \mathbf { n } ^ { \top } ( \mathbf { q } - \mathbf { p } ) / u$ measure normal alignment. For a closed object mesh, we assign contact by

$$
c ( \mathbf { p } ) = \left\{ \begin{array} { l l } { 1 , } & { \mathbf { p } \mathrm { ~ i s ~ i n s i d e ~ t h e ~ o b j e c t , } } \\ { 1 , } & { \mathrm { o u t s i d e , ~ } u \leq \delta _ { D } , \ u < 3 \mathrm { m m } , } \\ { 1 , } & { \mathrm { o u t s i d e , ~ } 3 \mathrm { m m } \leq u \leq \delta _ { D } , \ \kappa \geq 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

![](images/7e900bebece1b90479c70af2587e322b6e1b67a7d746670b93a23fcff31fb1c5.jpg)  
Figure 2: Contact-label construction: normal filtering (left) and inside-mesh inclusion (right).

(3)

Here, “outside” includes on-surface vertices (u =

0), which are labeled as contact without evaluating κ; the 3 mm band likewise bypasses the normal test where the closest-point direction is unstable. Inside-mesh inclusion retains potential contacts displaced into the object by pose error. For hand–hand and hand–scene data, we apply the same distance and normal tests, using inside-mesh inclusion only when a reliable sign is available. Appendix A.2 provides the distance thresholds and a visual comparison of the labeling rules. HACO assigns contacts by thresholding unsigned nearest-surface distance (Jung & Lee, 2025). Our geometric corrections provide more faithful contact supervision, as illustrated in Appendix Figure 6.

## 4.3 MOTION-DERIVED FORCE LABELS

The observed object motion determines the net wrench acting on the object, but does not uniquely determine the distribution of forces over the hand surface. We therefore construct force supervision in two stages: we first recover an object-level target wrench from motion and then distribute it over plausible hand vertices under geometric, frictional, and temporal constraints.

For rigid objects, we smooth the observed pose trajectories at 30 Hz and estimate linear and angular accelerations. Newton–Euler inverse dynamics then gives the non-gravitational external wrench that must be explained by the hands and any support from the environment:

$$
\mathbf { w } _ { t } ^ { \star } = \left[ \underset { \mathbf { R } _ { t } ( \mathbf { I } \alpha _ { t } + \omega _ { t } \times \mathbf { I } \omega _ { t } ) } { m ( \mathbf { a } _ { t } - \mathbf { g } ) } \right] , \qquad \mathbf { w } _ { t } ^ { \mathrm { h a n d } } + \mathbf { w } _ { t } ^ { \mathrm { e n v } } = \mathbf { w } _ { t } ^ { \star } .\tag{4}
$$

Here, $\mathbf { R } _ { t }$ is the object orientation, $m = 1$ kg is the reference object mass and I is the body-frame inertia scaled to this mass. The linear acceleration a and gravity g are expressed in the solver world frame with $\| \mathbf { g } \| = 9 . 8 1 \mathrm { m / s ^ { 2 } }$ , while the angular velocity $\omega _ { t }$ and acceleration $\pmb { \alpha } _ { t }$ are expressed in the body frame. We use dataset-provided gravity directions when available; otherwise, we estimate them from UprightNet RGB predictions (Xian et al., 2019) or scene geometry.

The target wrench alone does not uniquely determine the force at each hand vertex. We resolve this ambiguity using the local hand–object geometry illustrated in Figure 3. Vertices with signed distance $s _ { t , i } < 2 0$ mm are treated as force candidates, and each candidate force is constrained by a local friction cone defined by the hand-surface normal and tangent directions. We further bias the allocation toward vertices that are closer to, or penetrating, the object through

$$
q _ { t , i } = \frac { 2 } { 1 + \exp ( s _ { t , i } / \ell ) } , \qquad \ell = 4 \ : \mathrm { m m } .\tag{5}
$$

Because the force penalty is weighted by $( q _ { t , i } ^ { 2 } + \epsilon ) ^ { - 1 }$ , vertices closer to or penetrating the object incur a lower cost.

(a) Local polyhedral friction cone

![](images/7fcabeace5a76864d7298fe5181a28e23159d9d4b2a510676e5723401710cff1.jpg)  
(b) Signed distance to object

![](images/009ec2f2d7eaa06e6a3ca6413c442774fa9405f8d474c2c7dae6ef0baa0361e8.jpg)

(c) Per-vertex hand-local frame  
![](images/efb62e2fd199ba4c2c6a68437786df31d9db527a89cb623489e3cf4810cbd60c.jpg)

(d) Force allocation by signed distance  
![](images/50425157080bf2d55aa0e8f748219a309acc49065286922261e26456b8c2e367.jpg)  
Figure 3: Force allocation geometry and weighting. Upper left: a local polyhedral friction cone constrains each active vertex force. Upper right: signed distance s distinguishes penetrated $( s < 0 )$ contacting $( s \ : = \ : 0 )$ , and separated $( s > 0 )$ vertices. Lower left: vertex-local force directions on the hand surface. Lower right: the penalty weight $( q ( s ) ^ { 2 } + \epsilon ) ^ { - 1 }$ rises toward the relaxed 20 mm candidate cut-off.

We then solve for the hand forces jointly across the sequence. To account for interactions with supporting surfaces, we additionally introduce an upward environment force with magnitude $e _ { t } \geq 0$ wrench map $\mathbf { G } _ { t } ^ { \mathrm { e n v } }$ , and regularizer $\mathcal { R } _ { \mathrm { e n v } } ( e _ { 1 : T } )$ . The resulting sequence-level convex program is

$$
\begin{array} { r l } { \underset { \{ \mathbf { f } _ { t , i } \} , \{ e _ { t } \} } { \operatorname* { m i n } } } & { \lambda _ { f } \displaystyle \sum _ { t , i } \frac { \| \mathbf { f } _ { t , i } \| _ { 2 } ^ { 2 } } { q _ { t , i } ^ { 2 } + \epsilon } + \lambda _ { \Delta f } \displaystyle \sum _ { t , i } \| \mathbf { f } _ { t + 1 , i } - \mathbf { f } _ { t , i } \| _ { 2 } ^ { 2 } + \mathcal { R } _ { \mathrm { e n v } } ( e _ { 1 : T } ) } \\ { \mathrm { s . t . } } & { \displaystyle \sum _ { i } \mathbf { G } _ { t , i } \mathbf { f } _ { t , i } + \mathbf { G } _ { t } ^ { \mathrm { e n v } } e _ { t } = \mathbf { w } _ { t } ^ { \star } , } \\ & { \mathbf { f } _ { t , i } \in \mathcal { K } ( \mathbf { n } _ { t , i } , \mu ) , \quad \mathbf { f } _ { t , i } = \mathbf { 0 } \mathrm { f o r i n a c t i v e } i , \quad e _ { t } \geq 0 , } \end{array}\tag{6}
$$

where $\mathbf { G } _ { t , i }$ maps a vertex force to its force-and-torque contribution about the object center of mass, and K is a six-facet polyhedral approximation of the unilateral Coulomb friction cone. The equality constraint ensures that the allocated hand and environment forces reproduce the motion-derived target wrench, while the magnitude and temporal terms select a spatially and temporally regularized solution among the feasible allocations.

For articulated objects, we also account for joint forces and torques, mapping each hand contact to the object part it acts on (Appendix A.3.3). Articulated sequences provide vertex-force supervision but are excluded from the rigid rollout loss. Appendix A.3 provides further details on trajectory processing, rigid-force allocation and validation, and the articulated extension.

## 5 TACTOHAND

## 5.1 VIDEO ENCODING AND INTERACTION-FIELD DECODING

Each frame is encoded by a pretrained ViT-L DINOv3 backbone (Simeoni et al., 2025). A tem- ´ poral transformer aggregates clip-level visual information, and three task-specific queries decode the hand-surface interaction fields for both hands at each 10 Hz image timestamp. Architecture and input settings are provided in Appendix C.1.

## 5.2 DENSE CONTACT AND PROXIMITY

Let $\widehat { a } _ { t , h }$ denote the probability that hand h makes any contact in frame t (engagement), and $\widehat { b } _ { t , h , \iota }$ the vertex contact probability conditioned on that event. We factor the final contact probability as $\widehat { c } _ { t , h , v } = \widehat { a } _ { t , h } \widehat { b } _ { t , h , v } .$ , with engagement target $\begin{array} { r } { a _ { t , h } ^ { \star } = \mathcal { H } [ \sum _ { v } c _ { t , h , v } ^ { \star } > 0 ] } \end{array}$ on valid hand frames. Binary cross-entropy supervises engagement, while HACO’s vertex-level class-balanced (VCB) loss (Jung & Lee, 2025) supervises conditional contact on engaged frames. A separate proximity head is supervised with masked mean squared error against the targets defined in Equation 2.

The same engagement scalar $\widehat { a } _ { t , h }$ also gates all vertex forces of hand h. This soft gate allows force and motion losses to propagate into both the engagement and force branches while encouraging zero force on no-contact hand frames.

## 5.3 VERTEX FORCES AND ANALYTIC WRENCH AGGREGATION

Omitting the time index, let ${ \widehat { \ell } } _ { i , 1 } , { \widehat { \ell } } _ { i , 2 } , { \widehat { \ell } } _ { i , n }$ denote raw local force components along two tangents and the outward normal. Ground-truth posed MANO geometry defines this basis in camera coordinates. With the engagement gate for hand $h ( i )$ , the hand-on-object force is

$$
\widehat { \mathbf { f } } _ { i } = \widehat { a } _ { h ( i ) } \left( \widehat { \ell } _ { i , 1 } \mathbf { t } _ { i , 1 } + \widehat { \ell } _ { i , 2 } \mathbf { t } _ { i , 2 } + \widehat { \ell } _ { i , n } \mathbf { n } _ { i } \right) .\tag{7}
$$

The basis vectors and $\widehat { \mathbf { f } } _ { i }$ are therefore expressed in the camera frame. Predicting force components relative to the hand surface reduces sensitivity to camera orientation (Figure 3, lower left; Appendix C.2).

At each image timestamp, the point-force loss ${ \mathcal { L } } _ { \mathrm { p o i n t } }$ supervises the reconstructed camera-frame forces while balancing active and near-zero targets (Appendix C.2). Given the object center of mass x and vertex positions $\mathbf { p } _ { i }$ in the same frame, the corresponding hand wrench is obtained analytically as

$$
\mathbf { w } ^ { \mathrm { h a n d } } = \left[ \sum _ { i } \widehat { \mathbf { f } } _ { i } \right] .\tag{8}
$$

## 5.4 MOTION-CONSISTENT ROLLOUT

We further constrain the predicted forces through short-horizon rigid-body motion. Each 10 Hz visual interval contains three 30 Hz force intervals indexed by $s \in \{ 0 , 1 , 2 \}$ $\mathrm { \bf A t } \ s = 0$ , we use the image-aligned network prediction $\widehat { \mathbf { f } } _ { t , i } ;$ at $s \in \{ 1 , 2 \}$ , we use the detached motion-derived reference stopgrad ${ \bf l } ( { \bf f } _ { t , s , i } ^ { \star } )$ . This keeps the image-aligned prediction differentiable while avoiding unconstrained force extrapolation between visual observations.

Starting from the observed object state, we aggregate the vertex forces into a hand wrench and transform it from camera coordinates into the solver world frame. We then combine it with gravity and the estimated environment-support terms and integrate the resulting rigid-body dynamics. Because this integration is differentiable, motion-level errors propagate back through wrench aggregation to the predicted vertex forces.

Let k index object states at the boundaries of successive 30 Hz force intervals, with $k = 0$ denoting the initial state. Let H denote the indices included in the short-horizon objective. We compare the propagated and observed object position and orientation:

$$
\mathcal { L } _ { \mathrm { r o l l } } = \frac { 1 } { \lvert \mathcal { H } \rvert } \sum _ { k \in \mathcal { H } } \left( \lambda _ { x } \lVert \widehat { \mathbf { x } } _ { k } - \mathbf { x } _ { k } ^ { \star } \rVert _ { 2 } ^ { 2 } + \frac { \lambda _ { R } } { 2 } \lVert \widehat { \mathbf { R } } _ { k } - \mathbf { R } _ { k } ^ { \star } \rVert _ { F } ^ { 2 } \right) .\tag{9}
$$

![](images/baecdc8e44f7ac6fff9f57347d7ac24afdac078bc860ab3b583c2e90cc3e09dc.jpg)  
Figure 4: Differentiable rigid-body supervision. Indices i, i + 1, i + 2 are 10 Hz image timestamps; each transition contains three 30 Hz force intervals. Predicted vertex forces determine the object wrench at the image-aligned interval, while the two intermediate intervals use detached motionderived reference forces. Pose errors supervise the propagated state.

The differentiable state update is detailed in Appendix C.2 (Equation 17). The rollout loss is applied only to eligible rigid sequences for which the aggregated hand force can be attributed to the target object; articulated and otherwise ineligible sequences retain their available field-level supervision.

## 5.5 TRAINING OBJECTIVE

Each training sample activates only losses supported by its annotations. We combine the weighted engagement and conditional vertex losses into $\mathcal { L } _ { \mathrm { { c o n t a c t } } }$ and define the overall training objective as

$$
\mathcal { L } = \lambda _ { c } \mathcal { L } _ { \mathrm { c o n t a c t } } + \lambda _ { d } \mathcal { L } _ { \mathrm { p r o x } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p o i n t } } + \lambda _ { r } \mathcal { L } _ { \mathrm { r o l l } } .\tag{10}
$$

Geometry-only sequences use contact and proximity supervision. Sequences with valid force labels additionally use ${ \mathcal { L } } _ { \mathrm { { p o i n t } } }$ , while eligible rigid sequences also activate ${ \mathcal { L } } _ { \mathrm { r o l l } } .$ . Missing force supervision is masked. Repeated frames used to pad clips are excluded from force and rollout losses (Appendix C.2).

## 6 EXPERIMENTS

## 6.1 DENSE PROXIMITY AND CONTACT COMPARISON

Table 1 compares TactoHand with prior proximity and contact predictors. We re-score released baseline checkpoints against the same annotations. For proximity, we compare against InterField-SF and InterField-LSTM (Fan et al., 2023). For contact, we compare against HACO (Jung & Lee, 2025), EgoPHI (Ilic et al., 2026), and ContactPrompt (Jung & Lee, 2026). EgoPHI uses single RGB frames together with solver-derived MANO vertices and target-object masks and meshes. ContactPrompt is evaluated quantitatively only on MOW because full evaluation requires a separate VLM prompt for each frame; elsewhere, we include it qualitatively in Figure 5. Input settings, dataset splits, and the frames and vertices included in each metric are detailed in Appendix D.

We evaluate cross-dataset transfer on HOCap, which is excluded from training.

For proximity, InterField-SF predicts distances per frame, whereas InterField-LSTM aggregates temporal features; both are trained on ARCTIC (Fan et al., 2023). InterField covers 100 mm and TactoHand is supervised up to 250 mm, so we report full-range ADE and ADE-100, which clips both predictions and targets at 100 mm.

TactoHand lowers full-range ADE across all four benchmarks: the temporal variant performs best on ARCTIC, DexYCB, and HOCap, and the single-frame variant on MOW. The latter also leads on HOCap ADE-100. ADE-100 separates accuracy within the shared distance range from the methods different full ranges.

Turning to contact, the temporal model consistently improves Macro-F1 over HACO on DexYCB, HOCap, and ARCTIC, with the largest gain on HOCap, from 18.0 to 50.5. On MOW, TactoHand matches HACO in Macro-F1 (34.5) while improving Micro-F1 from 41.5 to 44.3. Figure 5 illustrates representative missed and spurious contact regions across methods.

Table 1: Proximity (left) and contact (right). ADE is in mm, Accel. in m/s<sup>2</sup>, and P/R/F1 in percent. ADE-100 clips predictions and targets at 100 mm; Macro-F1 averages F1 over frame–hand pairs with ground-truth or predicted contact. Bold marks the best per dataset. ‡ HOCap is excluded from training. See Appendix D for metrics.
<table><tr><td></td><td colspan="4">PROXIMITY</td><td colspan="5">CONTACT</td></tr><tr><td>Dataset</td><td>Method</td><td>ADE↓</td><td>ADE-100↓</td><td>Accel. ↓</td><td>Method</td><td>P↑</td><td></td><td>R ↑ Macro-F1 ↑ Micro-F1 ↑</td><td></td></tr><tr><td>HOCap‡</td><td>InterField-SF</td><td>132.51</td><td>38.15</td><td>1.06</td><td>HACO</td><td></td><td>11.1 77.7</td><td>18.0</td><td>19.5</td></tr><tr><td></td><td>InterField-LSTM</td><td>134.69</td><td>40.10</td><td>0.47</td><td>EgoPHI</td><td>10.337.3</td><td></td><td>11.4</td><td>16.2</td></tr><tr><td></td><td>Ours (SF)</td><td>46.09</td><td>6.35</td><td>1.35</td><td>Ours (SF)</td><td>68.9 57.7</td><td></td><td>50.1</td><td>62.8</td></tr><tr><td></td><td>Ours (T)</td><td>42.51</td><td>6.40</td><td>1.07</td><td>Ours (T)</td><td>68.8 60.6</td><td></td><td>50.5</td><td>64.5</td></tr><tr><td>ARCTIC (s05, ego) InterField-SF</td><td></td><td>29.65</td><td>9.81</td><td>0.82</td><td>HACO</td><td>42.2 82.2</td><td></td><td>42.2</td><td>55.8</td></tr><tr><td></td><td>InterField-LSTM</td><td>29.68</td><td>9.77</td><td>0.76</td><td>EgoPHI</td><td>33.5 45.2</td><td></td><td>31.8</td><td>38.5</td></tr><tr><td></td><td>Ours (SF)</td><td>20.02</td><td>10.58</td><td>1.09</td><td>Ours (SF)</td><td>59.2 67.3</td><td></td><td>50.9</td><td>63.0</td></tr><tr><td></td><td>Ours (T)</td><td>16.23</td><td>9.32</td><td>1.10</td><td>Ours (T)</td><td>64.063.7</td><td></td><td>51.0</td><td>63.8</td></tr><tr><td>DexYCB</td><td>InterField-SF</td><td>128.99</td><td>32.83</td><td>1.76</td><td>HACO</td><td>27.4 85.5</td><td></td><td>34.3</td><td>41.5</td></tr><tr><td></td><td>InterField-LSTM</td><td>127.43</td><td>31.05</td><td>1.18</td><td>EgoPHI</td><td></td><td>4.4 4.2</td><td>2.7</td><td>4.3</td></tr><tr><td></td><td>Ours (SF)</td><td>26.14</td><td>3.92</td><td>1.74</td><td>Ours (SF)</td><td>63.6 71.4</td><td></td><td>58.5</td><td>67.3</td></tr><tr><td></td><td>Ours (T)</td><td>23.29</td><td>3.42</td><td>1.46</td><td>Ours (T)</td><td>65.7 69.7</td><td></td><td>59.5</td><td>67.6</td></tr><tr><td>MOW</td><td>InterField-SF</td><td>108.13</td><td>33.15</td><td></td><td>HACO</td><td>29.2 71.5</td><td></td><td>34.5</td><td>41.5</td></tr><tr><td></td><td>InterField-LSTM 110.08</td><td></td><td>35.00</td><td></td><td>ContactPrompt</td><td>25.2 66.2</td><td></td><td>29.7</td><td>36.5</td></tr><tr><td></td><td>Ours (SF)</td><td>46.86</td><td>12.21</td><td></td><td>EgoPHI</td><td>22.5 39.4</td><td></td><td>12.4</td><td>28.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Ours (T)</td><td>44.2 44.3</td><td></td><td>34.5</td><td>44.3</td></tr></table>

![](images/226cb98ebcffc75a0e538420dafe6eac2554fe4c9ab2b8046a81598e8c7c70b0.jpg)  
Figure 5: Contact comparison showing the input scene, hand crop, predicted contact maps, and geometry-derived reference.

## 6.2 FORCE AND ROLLOUT ABLATION

We next evaluate the effect of motion-level physics supervision. We train single-frame and temporal variants with and without the rollout loss while holding all other training settings fixed. Table 2 reports both vertex-force errors and a reference-assisted rigid-body rollout diagnostic. Appendix D.3 details the evaluation protocol and window-exclusion criteria.

Table 2: Force and rollout ablation on DexYCB (DINOv3 ViT-L). SF/T: single-frame/temporal; MAE: mN; rollout ADE: cm; angles: degrees. Active targets exceed 0.05 N. The 0.4 s rollout uses reference-assisted intervals and excludes divergent windows (Appendix D.3); bold indicates column minima.
<table><tr><td>Model</td><td>Physics</td><td></td><td>Vertex MAE ↓ Active MAE↓</td><td></td><td>Force angle ↓ Rollout ADE↓</td><td>Rollout angle ↓</td></tr><tr><td>SF</td><td>Off</td><td>12.6</td><td>124.9</td><td>28.2</td><td>5.3</td><td>45.5</td></tr><tr><td>SF</td><td>On</td><td>12.9</td><td>126.7</td><td>29.8</td><td>3.8</td><td>27.4</td></tr><tr><td>T</td><td>Off</td><td>13.7</td><td>126.4</td><td>31.0</td><td>5.7</td><td>45.2</td></tr><tr><td>T</td><td>On</td><td>12.8</td><td>122.7</td><td>27.5</td><td>3.3</td><td>29.4</td></tr></table>

Physics supervision primarily improves motion consistency: for the temporal model, rollout ADE falls from 5.7 to 3.3 cm and active-force MAE from 126.4 to 122.7 mN. The single-frame model also lowers rollout error, despite slightly worse active-force MAE and force-angle error.

## 6.3 DOWNSTREAM HAND–OBJECT RECONSTRUCTION REFINEMENT

We replace the original contact objective in EasyHOI (Liu et al., 2025) and ArtHOI (Wang et al., 2026) with terms derived from fixed TactoHand contact and proximity predictions, keeping all other settings unchanged. This controlled setting uses metric object geometry; EasyHOI also uses groundtruth object orientation (Appendix D).

On DexYCB, HOCap, and MOW, TactoHand improves PA-VPE in all six method–dataset pairs and gives the lowest hand–object Chamfer distance in each (Table 3). Full metrics and visuals appear in Appendix Table 5 and Figures 7–8.

Table 3: Controlled reconstruction refinement. PA-VPE: mm; $\mathrm { C D } _ { \mathrm { h o } } \mathrm { : }$ bidirectional mean squared hand–object Chamfer in $\mathrm { 1 0 ^ { - 3 } m ^ { 2 } }$ . Bold marks each method–dataset best.
<table><tr><td colspan="2"></td><td colspan="3">PA-VPE↓</td><td colspan="3"> $\mathrm { C D } _ { \mathrm { h o } } \downarrow$ </td></tr><tr><td>Dataset</td><td>Method</td><td>Base</td><td>+ HACO</td><td>+ Ours</td><td>Base</td><td>+HACO</td><td>+ Ours</td></tr><tr><td>DexYCB</td><td>EasyHOI</td><td>9.937</td><td>9.516</td><td>9.077</td><td>24.140</td><td>12.552</td><td>2.004</td></tr><tr><td rowspan="2">HOCap</td><td>ArtHOI</td><td>7.317</td><td>6.027</td><td>4.626</td><td>11.587</td><td>13.627</td><td>1.431</td></tr><tr><td>EasyHOI</td><td>9.179</td><td>9.051</td><td>7.592</td><td>59.416</td><td>68.943</td><td>11.461</td></tr><tr><td rowspan="2">MOW</td><td>ArtHOI</td><td>6.984</td><td>6.684</td><td>6.073</td><td>5.234</td><td>5.199</td><td>1.104</td></tr><tr><td>EasyHOI</td><td>14.832</td><td>14.276</td><td>13.939</td><td>69.080</td><td>72.638</td><td>15.266</td></tr><tr><td rowspan="2"></td><td>ArtHOI</td><td>13.572</td><td>13.261</td><td>13.100</td><td>73.993</td><td>73.147</td><td>32.413</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 7 LIMITATIONS

Geometry-derived contact is a proxy sensitive to source geometry and alignment. Motion-derived vertex forces are non-unique estimates, not measurements, and depend on inertia, friction, and environmental support assumptions. Motion-level evaluation is limited to short-horizon, referenceassisted rigid-body rollouts. Future work should incorporate measured forces and articulated rollout supervision while reducing reliance on auxiliary physical inputs.

## 8 CONCLUSION

TactoHand predicts contact, proximity, and 3D force as dense interaction fields on the hand surface. Geometry-derived labels across 24 sources, supplemented with regularized motion-derived force targets where available, support heterogeneous training. In our evaluations, TactoHand improves contact and proximity estimation, and its contact and proximity fields improve controlled hand– object reconstruction. Physics supervision reduces trajectory error in the reference-assisted rigidbody evaluation. Together, these results show how a shared hand-surface representation connects visual prediction, object motion, and hand–object reconstruction.

## REPRODUCIBILITY STATEMENT

The main text defines the hand-surface fields, label construction, model, and evaluation. Appendices A–D document data sources and labels, additional results, model and training settings, and evaluation protocols. We will release the code and derived annotations to support reproduction of the labels and trained model.

## AI USE STATEMENT

Generative AI tools assisted with retrieval and discovery of related work, drafting sections of the paper, and polishing its writing. The authors take responsibility for the final content, including its claims, citations, figures, and results.

## ETHICS STATEMENT

This work uses only previously released datasets and collects no new human-subject data. The sources were assembled by their original authors under their own consent and licensing arrangements, and we neither re-identify participants nor add identifying information. We record each dataset’s source and preprocessing steps so that derived annotations can be traced to the original data. Users must comply with the terms of each source dataset.

The labels we derive are estimates, not measurements. Vertex forces are obtained from smoothed object motion through inverse dynamics and a constrained allocation, and the construction admits many local force fields for a single aggregate wrench; the dataset records which frames satisfy the force-allocation constraints and never substitutes a zero-force target for an infeasible one. Monocular force estimation is also dual-use, in that an accurate estimate of applied force could inform physical manipulation or contact-based control. We therefore report force labels with their validity flags and residuals, and we caution against treating them, or any force prediction trained on them, as ground truth in safety-critical settings.

## REFERENCES

Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon, Shreyas Hampali, Shangchen Han, Fan Zhang, Linguang Zhang, Jade Fountain, Edward Miller, Selen Basol, Richard Newcombe, Robert Wang, Jakob Julian Engel, and Tomas Hodan. HOT3D: Hand and object tracking in 3D from egocentric multi-view videos. In CVPR, pp. 7061–7071, 2025. URL https://openaccess. thecvf.com/content/CVPR2025/html/Banerjee\_HOT3D\_Hand\_and\_Object\_ Tracking\_in\_3D\_from\_Egocentric\_Multi-View\_CVPR\_2025\_paper.html.

Samarth Brahmbhatt, Chengcheng Tang, Christopher D. Twigg, Charles C. Kemp, and James Hays. ContactPose: A dataset of grasps with object contact and hand pose. In ECCV, pp. 361–378. Springer, 2020. doi: 10.1007/978-3-030-58601-0 22.

Zhe Cao, Ilija Radosavovic, Angjoo Kanazawa, and Jitendra Malik. Reconstructing hand-object interactions in the wild. In ICCV, pp. 12417–12426, 2021. URL https://openaccess. thecvf.com/content/ICCV2021/html/Cao\_Reconstructing\_Hand-Object\_ Interactions\_in\_the\_Wild\_ICCV\_2021\_paper.html.

Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S. Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, Jan Kautz, and Dieter Fox. DexYCB: A benchmark for capturing hand grasping of objects. In CVPR, 2021. URL https://openaccess.thecvf.com/content/CVPR2021/html/Chao\_DexYCB\_ A\_Benchmark\_for\_Capturing\_Hand\_Grasping\_of\_Objects\_CVPR\_2021\_ paper.html.

Woojin Cho, Jihyun Lee, Minjae Yi, Minje Kim, Taeyun Woo, Donghwan Kim, Taewook Ha, Hyokeun Lee, Je-Hwan Ryu, Woontack Woo, and Tae-Kyun Kim. Dense hand-object (HO) graspnet with full grasping taxonomy and dynamics. In ECCV, volume 15140 of Lecture

Notes in Computer Science, pp. 284–303. Springer Nature Switzerland, 2024. doi: 10.1007/ 978-3-031-73007-8 17. URL https://hograspnet2024.github.io/.

Eadom Dessalene, Botao He, Michael Maynord, Yonatan Tussa, Pavan Mantripragada, Yianni Karabatis, Nirupam Roy, and Yiannis Aloimonos. FEEL (force-enhanced egocentric learning): A dataset for physical action understanding. In ECCV, 2026. URL https://arxiv.org/ abs/2603.15847. Preprint: arXiv:2603.15847.

Kiana Ehsani, Shubham Tulsiani, Saurabh Gupta, Ali Farhadi, and Abhinav Gupta. Use the force, Luke! learning to predict physical forces by simulating effects. In CVPR, 2020. doi: 10.1109/ CVPR42600.2020.00030. URL https://arxiv.org/abs/2003.12045.

Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J. Black, and Otmar Hilliges. ARCTIC: A dataset for dexterous bimanual hand-object manipulation. In CVPR, pp. 12943–12954, 2023. URL https://openaccess.thecvf. com/content/CVPR2023/html/Fan\_ARCTIC\_A\_Dataset\_for\_Dexterous\_ Bimanual\_Hand-Object\_Manipulation\_CVPR\_2023\_paper.html.

Rao Fu, Dingxi Zhang, Alex Jiang, Wanjia Fu, Austin Funk, Daniel Ritchie, and Srinath Sridhar. GigaHands: A massive annotated dataset of bimanual hand activities. In CVPR, pp. 17461– 17474, 2025. URL https://openaccess.thecvf.com/content/CVPR2025/ html/Fu\_GigaHands\_A\_Massive\_Annotated\_Dataset\_of\_Bimanual\_Hand\_ Activities\_CVPR\_2025\_paper.html.

Patrick Grady, Chengcheng Tang, Christopher D. Twigg, Minh Vo, Samarth Brahmbhatt, and Charles C. Kemp. ContactOpt: Optimizing contact to improve grasps. In CVPR, 2021. doi: 10.1109/CVPR46437.2021.00152. URL https://arxiv.org/abs/2104.07267.

Patrick Grady, Chengcheng Tang, Samarth Brahmbhatt, Christopher D. Twigg, Chengde Wan, James Hays, and Charles C. Kemp. PressureVision: Estimating hand pressure from a single RGB image. In ECCV, volume 13666 of Lecture Notes in Computer Science, pp. 328–345. Springer, Cham, 2022. doi: 10.1007/978-3-031-20068-7 19. URL https://arxiv.org/ abs/2203.10385.

Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vincent Lepetit. HOnnotate: A method for 3D annotation of hand and object poses. In CVPR, 2020. URL https://openaccess. thecvf.com/content\_CVPR\_2020/html/Hampali\_HOnnotate\_A\_Method\_ for\_3D\_Annotation\_of\_Hand\_and\_Object\_CVPR\_2020\_paper.html.

Shreyas Hampali, Sayan Deb Sarkar, Mahdi Rad, and Vincent Lepetit. Keypoint transformer: Solving joint identification in challenging hands and object interactions for accurate 3D pose estimation. In CVPR, 2022. URL https://openaccess.thecvf.com/ content/CVPR2022/papers/Hampali\_Keypoint\_Transformer\_Solving\_ Joint\_Identification\_in\_Challenging\_Hands\_and\_Object\_CVPR\_2022\_ paper.pdf.

Mohamed Hassan, Vasileios Choutas, Dimitrios Tzionas, and Michael J. Black. Resolving 3D human pose ambiguities with 3D scene constraints. In ICCV, pp. 2282–2292, 2019. doi: 10. 1109/ICCV.2019.00237. URL https://prox.is.tue.mpg.de/.

Mohamed Hassan, Partha Ghosh, Joachim Tesch, Dimitrios Tzionas, and Michael J. Black. Populating 3D scenes by learning human-scene interaction. In CVPR, 2021. doi: 10.1109/CVPR46437. 2021.01447. URL https://arxiv.org/abs/2012.11581.

Yana Hasson, Gul Varol, Dimitrios Tzionas, Igor Kalevatykh, Michael J. Black, Ivan Laptev, and¨ Cordelia Schmid. Learning joint reconstruction of hands and manipulated objects. In CVPR, 2019. URL https://hassony2.github.io/obman.

Yana Hasson, Gul Varol, Cordelia Schmid, and Ivan Laptev. Towards unconstrained joint hand-¨ object reconstruction from RGB videos. In 3DV, pp. 659–668, 2021. doi: 10.1109/3DV53792. 2021.00075. URL https://arxiv.org/abs/2108.07044.

Haoyu Hu, Xinyu Yi, Hao Zhang, Jun-Hai Yong, and Feng Xu. Physical interaction: Reconstructing hand-object interactions with physics. In SIGGRAPH Asia, 2022. doi: 10.1145/3550469. 3555421. URL https://arxiv.org/abs/2209.10833.

Junxing Hu, Hongwen Zhang, Zerui Chen, Mengcheng Li, Yunlong Wang, Yebin Liu, and Zhenan Sun. Learning explicit contact for implicit reconstruction of hand-held objects from monocular images. Proceedings of the AAAI Conference on Artificial Intelligence, 38(3):2220–2228, 2024. doi: 10.1609/aaai.v38i3.27995. URL https://ojs.aaai.org/index.php/ AAAI/article/view/27995.

Chun-Hao P. Huang, Hongwei Yi, Markus Hoschle, Matvey Safroshkin, Tsvetelina Alexiadis, Senya¨ Polikovsky, Daniel Scharstein, and Michael J. Black. Capturing and inferring dense full-body human-scene contact. In CVPR, 2022. doi: 10.1109/CVPR52688.2022.01292. URL https: //arxiv.org/abs/2206.09553.

Andela Ilic, Rachel Schuchert, Yijing Jiang, and Christian Holz. EgoPHI: Estimating 3D handobject contact and force from egocentric vision. arXiv preprint, 2026. doi: 10.48550/arxiv.2608. 13014. URL https://arxiv.org/abs/2608.13014.

Subin Jeon, Byungjun Kim, and Hanbyul Joo. HOPE: Hand-object pressure estimation from monocular videos. arXiv preprint, 2026. URL https://arxiv.org/abs/2608.06192.

Daniel Sungho Jung and Kyoung Mu Lee. Learning dense hand contact estimation from imbalanced data. In NeurIPS, volume 38, 2025. doi: 10.52202/085713-4017. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ ae484111f7a8f02aac14f25d5e7c498a-Abstract-Conference.html.

Daniel Sungho Jung and Kyoung Mu Lee. Training-free dense hand contact estimation with multimodal large language models. arXiv preprint arXiv:2605.05886, 2026.

Taein Kwon, Bugra Tekin, Jan Stuhmer, Federica Bogo, and Marc Pollefeys. H2O: Two hands¨ manipulating objects for first person interaction recognition. In ICCV, 2021. doi: 10.1109/ ICCV48922.2021.00998. URL https://taeinkwon.com/projects/h2o/.

Zongmian Li, Jiri Sedlar, Justin Carpentier, Ivan Laptev, Nicolas Mansard, and Josef Sivic. Estimating 3D motion and forces of human-object interactions from internet videos. International Journal of Computer Vision, 130(2):363–383, 2022. doi: 10.1007/s11263-021-01540-1. URL https://arxiv.org/abs/2111.01591.

Yumeng Liu, Xiaoxiao Long, Zemin Yang, Yuan Liu, Marc Habermann, Christian Theobalt, Yuexin Ma, and Wenping Wang. EasyHOI: Unleashing the power of large models for reconstructing hand-object interactions in the wild. In CVPR, pp. 7037–7047, 2025. URL https://openaccess.thecvf.com/content/CVPR2025/html/Liu\_ EasyHOI\_Unleashing\_the\_Power\_of\_Large\_Models\_for\_Reconstructing\_ Hand-Object\_CVPR\_2025\_paper.html.

Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, and Li Yi. TACO: Benchmarking generalizable bimanual tool-action-object understanding. In CVPR, 2024. URL https://taco2024.github.io/.

Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi. HOI4D: A 4D egocentric dataset for category-level human-object interaction. In CVPR, 2022. URL https://openaccess.thecvf. com/content/CVPR2022/html/Liu\_HOI4D\_A\_4D\_Egocentric\_Dataset\_for\_ Category-Level\_Human-Object\_Interaction\_CVPR\_2022\_paper.html.

Gyeongsik Moon, Shoou-I Yu, He Wen, Takaaki Shiratori, and Kyoung Mu Lee. InterHand2.6M: A dataset and baseline for 3D interacting hand pose estimation from a single RGB image. In ECCV, 2020. URL https://www.ecva.net/papers/eccv\_2020/papers\_ECCV/ html/3540\_ECCV\_2020\_paper.php.

Gyeongsik Moon, Shunsuke Saito, Weipeng Xu, Rohan Joshi, Julia Buffalini, Harley Bellan, Nicholas Rosen, Jesse Richardson, Mallorie Mize, Philippe de Bree, Tomas Simon, Bo Peng, Shubham Garg, Kevyn McPhail, and Takaaki Shiratori. A dataset of relighted 3D interacting hands. In NeurIPS, volume 36, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/396beafa6feba781a7114780e6837253-Abstract-Datasets\_and\_ Benchmarks.html.

Haozhe Qi, Chen Zhao, Mathieu Salzmann, and Alexander Mathis. HOISDF: Constraining 3D hand-object pose estimation with global signed distance fields. In CVPR, pp. 10392–10402, 2024. doi: 10.1109/CVPR52733.2024.00989. URL https://arxiv.org/abs/2402.17062.

Yi-Ling Qiao, Junbang Liang, Vladlen Koltun, and Ming C. Lin. Scalable differentiable physics for learning and control. In ICML, volume 119 of Proceedings of Machine Learning Research, pp. 7847–7856. PMLR, 2020. URL https://arxiv.org/abs/2007.02168.

Patrick Rim, Kevin Harris, Braden Copple, et al. SHOW3D: Capturing scenes of 3D hands and objects in the wild. In CVPR, 2026. URL https://show3d-dataset.github.io/.

Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics, 36(6), 2017. doi: 10.1145/3130800. 3130883. URL https://doi.org/10.1145/3130800.3130883.

Soshi Shimada, Vladislav Golyanik, Patrick Perez, and Christian Theobalt. Decaf: Monocular´ deformation capture for face and hand interactions. ACM Transactions on Graphics, 42(11), 2023. doi: 10.1145/3618329. URL https://vcai.mpi-inf.mpg.de/projects/Decaf/.

Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose,´ Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, Francisco Massa, Daniel¨ Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, et al. DINOv3.´ arXiv preprint arXiv:2508.10104, 2025.

Mohan Kumar Srirama, Sudeep Dasari, Shikhar Bahl, and Abhinav Gupta. HRP: Human affordances for robotic pre-training. In RSS, 2024. doi: 10.15607/RSS.2024.XX.068. URL https://www.roboticsproceedings.org/rss20/p068.html.

Bartolomeo Stellato, Goran Banjac, Paul Goulart, Alberto Bemporad, and Stephen Boyd. OSQP: An operator splitting solver for quadratic programs. Mathematical Programming Computation, 12(4):637–672, 2020. doi: 10.1007/s12532-020-00179-2. URL https://stanford.edu/ <sub>˜</sub>boyd/papers/osqp.html.

The Newton Contributors. Newton: GPU-accelerated physics simulation for robotics and simulation research. Software, 2025. URL https://github.com/newton-physics/newton.

Shashank Tripathi, Agniv Chatterjee, Jean-Claude Passy, Hongwei Yi, Dimitrios Tzionas, and Michael J. Black. DECO: Dense estimation of 3D human-scene contact in the wild. In ICCV, 2023. doi: 10.1109/ICCV51070.2023.00735. URL https://arxiv.org/abs/2309. 15273.

Tze Ho Elden Tse, Zhongqun Zhang, Kwang In Kim, Ales Leonardis, Feng Zheng, and Hyung Jin Chang. S<sup>2</sup>Contact: Graph-based network for 3D hand-object contact estimation with semisupervised learning. In ECCV, volume 13661 of Lecture Notes in Computer Science, pp. 568– 584. Springer, Cham, 2022. doi: 10.1007/978-3-031-19769-7 33. URL https://arxiv. org/abs/2208.00874.

Dimitrios Tzionas, Luca Ballan, Abhilash Srikantha, Pablo Aponte, Marc Pollefeys, and Juergen Gall. Capturing hands in action using discriminative salient points and physics simulation. International Journal of Computer Vision, 2016. URL https://hic.is.tuebingen.mpg. de/.

Jikai Wang, Qifan Zhang, Yu-Wei Chao, Bowen Wen, Xiaohu Guo, and Yu Xiang. HO-Cap: A capture system and dataset for 3D reconstruction and pose tracking of hand-object interaction. In NeurIPS, 2025. URL https://arxiv.org/abs/2406.06843.

Zikai Wang, Zhilu Zhang, Yiqing Wang, Hui Li, and Wangmeng Zuo. ArtHOI: Taming foundation models for monocular 4D reconstruction of hand-articulated-object interactions. In CVPR, pp. 15998–16009, 2026. URL https://openaccess.thecvf. com/content/CVPR2026/html/Wang\_ArtHOI\_Taming\_Foundation\_Models\_ for\_Monocular\_4D\_Reconstruction\_of\_Hand-Articulated-Object\_CVPR\_ 2026\_paper.html.

Keenon Werling, Dalton Omens, Jeongseok Lee, Ioannis Exarchos, and C. Karen Liu. Fast and feature-complete differentiable physics engine for articulated rigid bodies with contact constraints. In RSS, 2021. doi: 10.15607/RSS.2021.XVII.034. URL https://arxiv.org/ abs/2103.16021.

Wenqi Xian, Zhengqi Li, Matthew Fisher, Jonathan Eisenmann, Eli Shechtman, and Noah Snavely. UprightNet: Geometry-aware camera orientation estimation from single images. In ICCV, 2019. doi: 10.1109/ICCV.2019.01007. URL https://openaccess.thecvf.com/ content\_ICCV\_2019/papers/Xian\_UprightNet\_Geometry-Aware\_Camera\_ Orientation\_Estimation\_From\_Single\_Images\_ICCV\_2019\_paper.pdf.

Xianghui Xie, Bharat Lal Bhatnagar, and Gerard Pons-Moll. CHORE: Contact, human and object reconstruction from a single RGB image. In ECCV, volume 13662 of Lecture Notes in Computer Science, pp. 125–145. Springer, Cham, 2022. doi: 10.1007/978-3-031-20086-1 8. URL https: //arxiv.org/abs/2204.02445.

Lixin Yang, Xinyu Zhan, Kailin Li, Wenqiang Xu, Jiefeng Li, and Cewu Lu. CPF: Learning a contact potential field to model the hand-object interaction. arXiv preprint, 2021. URL https: //arxiv.org/abs/2012.00924.

Lixin Yang, Kailin Li, Xinyu Zhan, Fei Wu, Anran Xu, Liu Liu, and Cewu Lu. OakInk: A large-scale knowledge repository for understanding hand-object interaction. In CVPR, pp. 20953–20962, 2022. URL https://openaccess.thecvf.com/content/ CVPR2022/html/Yang\_OakInk\_A\_Large-Scale\_Knowledge\_Repository\_ for\_Understanding\_Hand-Object\_Interaction\_CVPR\_2022\_paper.html.

Yifei Yin, Chen Guo, Manuel Kaufmann, Juan Jose Zarate, Jie Song, and Otmar Hilliges. Hi4D: 4D instance segmentation of close human interaction. In CVPR, 2023. URL https://openaccess.thecvf.com/content/CVPR2023/html/Yin\_Hi4D\_ 4D\_Instance\_Segmentation\_of\_Close\_Human\_Interaction\_CVPR\_2023\_ paper.html.

Zhenjun Yu, Wenqiang Xu, Pengfei Xie, Yutong Li, Brian W. Anthony, Zhuorui Zhang, and Cewu Lu. Dynamic reconstruction of hand-object interaction with distributed force-aware contact representation. In ICCV, pp. 8590–8599, 2025. doi: 10.1109/ICCV51701.2025.00804. URL https://arxiv.org/abs/2411.09572.

Yuan Zeng, Yujia Shi, Tiao Tan, Xingting Li, Yaqi Qin, Zongqing Lu, Wenming Yang, Jing-Hao Xue, and Qingmin Liao. EgoTactile: Learning grasp pressure for everyday objects from egocentric video. In ICML, 2026. URL https://arxiv.org/abs/2606.09243.

Xinyu Zhan, Lixin Yang, Yifei Zhao, Kangrui Mao, Hanlin Xu, Zenan Lin, Kailin Li, and Cewu Lu. OAKINK2: A dataset of bimanual hands-object manipulation in complex task completion. In CVPR, 2024. URL https://oakink.net/v2/.

Wenkang Zhang, Chengbo Yuan, Zicheng Zhang, Zhengxue Cheng, and Yang Gao. EgoTac: In-thewild tactile prediction from egocentric vision. arXiv preprint, 2026. URL https://arxiv. org/abs/2608.15060.

Yiming Zhao, Taein Kwon, Paul Streli, Marc Pollefeys, and Christian Holz. EgoPressure: A dataset for hand pressure and pose estimation in egocentric vision. In CVPR, pp. 27727–27738, 2025. doi: 10.1109/CVPR52734.2025.02582. URL https://arxiv.org/abs/2409.02224.

Danyan Zhou, Jinxuan Lu, Jiawei Lin, Tianxing Chen, Chuqiao Lyu, and Wenbo Ding. TouchSight: Bare-handed tactile prediction from egocentric video via generative visual augmentation. arXiv preprint, 2026. doi: 10.48550/arXiv.2609.20414. URL https://arxiv.org/abs/2609. 20414.

## APPENDIX ROADMAP

The appendix presents dataset composition and label construction (Appendix A), additional results (Appendix B), model and training details (Appendix C), and evaluation protocols and coverage (Appendix D).

## A DATASET AND LABEL CONSTRUCTION

## A.1 DATASET INVENTORY

Table 4 lists the 24 curated sources and their label types. We derive contact and proximity on the canonical MANO surface from source geometry. Sources with motion-derived force labels are grouped by whether the object motion used to construct them is rigid or articulated.

Counts describe our converted collection, not the full dataset releases or the number of training samples. Frames are counted across all captured camera views. Except for ObMan and MOW (∗), sources contain ordered sequences; for those two datasets, “sequences” are groups of independent images.

For GigaHands (†), we select 15 objects because the remaining objects have less reliable tracking. The table counts all camera views of sequences containing at least one selected object, including frames in those sequences that are excluded from training. HOCap (‡) contributes to the collection total but is excluded from training. PROX contributes 178 distinct source frames captured at 5 Hz; the table reports this original frame count. HO3Dv2/v3 refer to HO-3D release versions, and OakInk-v1/v2 to OakInk and OakInk2.

Table 4: Data sources and labels in the curated collection.
<table><tr><td>Dataset</td><td>Sequences</td><td>Frames</td></tr><tr><td>Force labels: articulated objects</td><td></td><td></td></tr><tr><td>ARCTIC (Fan et al., 2023)</td><td>300</td><td>1 903 869</td></tr><tr><td>HOI4D (Liu et al., 2022)</td><td>2973</td><td>891 900</td></tr><tr><td>Force labels: rigid objects</td><td></td><td></td></tr><tr><td>GigaHands† (Fu et al., 2025)</td><td>1424</td><td>9 449 724</td></tr><tr><td>Show3D (Rim et al., 2026)</td><td>1682</td><td>6931 884</td></tr><tr><td>TACO (Liu et al., 2024)</td><td>2120</td><td>4287490</td></tr><tr><td>HOGraspNet (Cho et al., 2024)</td><td>13575</td><td>4270 708</td></tr><tr><td>OakInk-v2 (Zhan et al., 2024)</td><td>611</td><td>3 898 190</td></tr><tr><td>HOT3D (Banerjee et al., 2025)</td><td>198</td><td>701 519</td></tr><tr><td>HOCap‡ (Wang et al., 2025)</td><td>64</td><td>653 175</td></tr><tr><td>DexYCB (Chao et al., 2021)</td><td>1000</td><td>581 968</td></tr><tr><td>H2O (Kwon et al., 2021)</td><td>184</td><td>453 855</td></tr><tr><td>OakInk-v1 (Yang et al., 2022)</td><td>792</td><td>235 604</td></tr><tr><td>HO3Dv3 (Hampali et al., 2020)</td><td>17</td><td>103 722</td></tr><tr><td>HO3Dv2 (Hampali et al., 2020)</td><td>17</td><td>77 953</td></tr><tr><td>H2O3D (Hampali et al., 2022)</td><td>15</td><td>67 525</td></tr><tr><td>Contact and proximity labels only</td><td></td><td></td></tr><tr><td>InterHand2.6M (Moon et al., 2020)</td><td>255 378</td><td>12469 860</td></tr><tr><td>ReInterHand (Moon et al., 2023)</td><td>250</td><td>886470</td></tr><tr><td>DECAF (Shimada et al., 2023)</td><td>16</td><td>778 817</td></tr><tr><td>Hi4D (Yin et al., 2023)</td><td>1600</td><td>191 120</td></tr><tr><td>ObMan* (Hasson et al., 2019)</td><td>23</td><td>118133</td></tr><tr><td>RICH (Huang et al., 2022)</td><td>111</td><td>62 969</td></tr><tr><td>HIC (Tzionas et al., 2016)</td><td>11</td><td>1824</td></tr><tr><td>MOW* (Cao et al., 2021)</td><td>512</td><td>512</td></tr><tr><td>PROX (Hassan et al., 2019)</td><td>1</td><td>178</td></tr><tr><td>Total</td><td>282874</td><td>49 018 969</td></tr></table>

Sources for which force labels can be constructed account for approximately 70% of curated frames; this does not mean that every frame contributes force supervision. GigaHands contributes contact and proximity supervision, while its motion-derived force targets are omitted because of noise in object poses. HOCap is reserved for evaluation. The rollout loss uses eligible rigid-object episodes only; articulated episodes provide vertex-force supervision.

## A.2 GEOMETRIC CONTACT-LABEL COMPARISON

Following HACO (Jung & Lee, 2025), we use distance thresholds of 10 mm for hand–object sources, 35 mm for MOW, and 5 mm for InterHand2.6M, HIC, and ReInterHand; all remaining sources use 10 mm. Equation 3 further applies normal filtering and inside-mesh inclusion, as illustrated in Figure 6.

Multiray inside–outside test. To determine whether a hand vertex lies inside a closed object mesh, we transform it into object coordinates and cast several oblique rays against the original, unrepaired triangles. For each ray, an odd number of intersections indicates that the vertex is inside; a majority vote across rays sets the sign of its nearest-surface distance. A small offset in the ray test avoids ambiguous intersections at edges and vertices. For open or unreliable surfaces, we retain the unsigned distance and skip inside-mesh inclusion.

![](images/7c47a0cba464a6b7ce208cb61a229f944a977c7e15b1b56bed122d6ebe29c1bb.jpg)  
Figure 6: Comparison of contact labels generated by HACO’s distance-only rule and our geometric rule. Normal filtering removes back-facing false positives, while inside-mesh inclusion recovers penetrating vertices missed by distance thresholding.

## A.3 MOTION-DERIVED FORCE LABELS

## A.3.1 TRAJECTORY PROCESSING

Temporal sampling and smoothing. We interpolate observed object trajectories onto a fixed 30 Hz timeline for both rigid- and articulated-object force labeling. Translation is smoothed with a symmetric, truncated Gaussian kernel. For rotation, we project each input matrix onto SO(3) by SVD, then compute a weighted intrinsic mean. We initialize the rotation estimate at the center rotation and update it iteratively as follows:

$$
\begin{array} { r } { \boldsymbol { \delta } _ { t } ^ { ( k ) } = \displaystyle \sum _ { j = - r _ { R } } ^ { r _ { R } } g _ { j } ^ { R } \mathrm { L o g } \Big ( ( \bar { \mathbf { R } } _ { t } ^ { ( k ) } ) ^ { \top } \mathbf { R } _ { t + j } \Big ) , \qquad \bar { \mathbf { R } } _ { t } ^ { ( k + 1 ) } = \bar { \mathbf { R } } _ { t } ^ { ( k ) } \mathrm { E x p } ( \boldsymbol { \delta } _ { t } ^ { ( k ) } ) , } \end{array}\tag{11}
$$

where $g _ { j } ^ { R }$ are normalized Gaussian weights. We use eight iterations and bandwidths of $\sigma _ { p } = 0 . 0 7 5$ s for translation and $\sigma _ { R } = 0 . 1 5 0 \mathrm { s }$ for rotation. At solver rate $f ,$ either kernel has radius $r ( \sigma , f ) =$ max $\{ \mathrm { r o u n d } ( 0 . 3 0 f ) , \lceil 3 \sigma f \rceil \}$ frames. We clamp indices outside the trajectory to the nearest endpoint when smoothing, but supervise a transition only when the full smoothing windows around both states fall within the observed trajectory.

## A.3.2 RIGID-OBJECT FORCE ALLOCATION AND VALIDATION

Trajectory-derived target wrench. For a rigid object, let $( \bar { \mathbf { x } } _ { t } , \bar { \mathbf { R } } _ { t } )$ denote its smoothed center-ofmass pose and let $\Delta t = \bar { 1 } / f$ . The displacement over transition $t \to t + 1$ gives the outgoing linear velocity:

$$
\mathbf { v } _ { t + 1 } = \frac { \bar { \mathbf { x } } _ { t + 1 } - \bar { \mathbf { x } } _ { t } } { \Delta t } .\tag{12}
$$

We obtain the outgoing world-frame angular velocity from $\bar { \mathbf { R } } _ { t + 1 } \bar { \mathbf { R } } _ { t } ^ { \top }$ using the Cayley map. With $\phi _ { t } = \mathrm { L o g } ( \bar { \mathbf { R } } _ { t + 1 } \bar { \mathbf { R } } _ { t } ^ { \top } )$ and $\theta _ { t } = \| \phi _ { t } \| _ { 2 }$ ,

$$
\omega _ { t + 1 } ^ { w } = \frac { 2 \tan ( \theta _ { t } / 2 ) } { \Delta t \theta _ { t } } \phi _ { t } ,\tag{13}
$$

with the continuous limit $\phi _ { t } / \Delta t$ as $\theta _ { t } \ \to \ 0 .$ . The incoming velocity is the preceding transition’s outgoing velocity; for the first transition, we copy its outgoing velocity. Dividing the velocity change by $\bar { \Delta } t$ yields the acceleration used in target wrench $\mathbf { w } _ { t } ^ { \star }$ . We express angular velocity and acceleration in the body frame when computing the torque in Equation 4.

We obtain gravity direction from dataset metadata where available, UprightNet RGB estimates (Xian et al., 2019), or a manually inferred scene or world-down axis. Camera-frame estimates are transformed into the solver world frame. The inverse-dynamics target uses a 1 kg reference mass and the corresponding scaled inertia, without rollout optimization.

Candidate vertices and grasp map. We consider force candidates for a hand–object pair only when at least one hand vertex is labeled as contact under the 10 mm rule in Section 4. Within that pair, vertices with $s _ { t , i } < 2 0$ mm are candidates, weighted by Equation 5.

Let $\mathbf { p } _ { i }$ and $\mathbf { f } _ { i }$ denote the world position and hand-on-object force of an active vertex. The grasp map

$$
\mathbf { G } _ { i } \mathbf { f } _ { i } = \left[ \underset { \left( \mathbf { p } _ { i } - \mathbf { x } \right) \times \mathbf { f } _ { i } } { \mathbf { f } _ { i } } \right]\tag{14}
$$

maps its force to a wrench about the object’s center of mass x. Candidate normals $\mathbf { n } _ { t , i }$ are normalized, and each force is constrained to a six-facet unilateral Coulomb cone with $\mu = 1 . 2$

Environment support. In Equation $6 , e _ { t } \geq 0$ denotes upward support at an estimated tabletop contact point, and $\mathbf { G } _ { t } ^ { \mathrm { e n v } }$ maps it to a wrench about the object center of mass. For tabletop sequences, we estimate this point from mesh vertices within 2 mm of the object’s lowest point. We infer whether the object rests on the table or is held above it by comparing its height with nearby hand-free frames. For each hand-contact interval, median heights from up to 15 such frames on either side provide reference heights before and after contact. Before identifying these intervals, we bridge gaps of at most five solver frames in the hand-contact signal. A rise of more than 7 mm for three frames marks pickup; a sustained return to the post-contact reference height marks placement. If the object is already held at the start of an interval, we use the post-contact reference when available; without reference frames, we treat it as off-table.

These support states are estimates rather than observed labels. We accept a frame without nearby active hand force only if table support alone can explain its motion-derived target wrench. When hand and environment contact coexist, the QP jointly allocates hand forces and nonnegative support at the estimated point.

QP implementation. We solve Equation 6 as a sparse sequence QP. The force-magnitude, forcetemporal, and environment-temporal coefficients default to one; the environment-magnitude coefficient and priority denominator ϵ are $1 0 ^ { - 4 }$ . Torque equations are scaled by 0.05 N m. Temporal differences use the union of adjacent active-vertex sets, treating absent vertices as zero. OSQP (Stellato et al., 2020) uses absolute and relative tolerances of $1 0 ^ { - 6 }$ . Partitioning, cached factorizations, and warm starts leave the objective unchanged.

Validity and replay. We reconstruct the fitted wrench from the solved vertex and environment forces and record its force and torque residuals. Independently, we replay that wrench from the stored smoothed state in Newton Physics (The Newton Contributors, 2025) and compare the onestep pose with the next smoothed state.

## A.3.3 ARTICULATED-OBJECT FORCE LABELS

For an articulated object with J joints, we retain its kinematic tree on the same 30 Hz timeline. We distribute the 1 kg reference mass among links in proportion to mesh volume and compute each link’s center of mass and inertia at its assigned mass. Multibody inverse dynamics on the smoothed trajectory gives the required generalized effort. Subtracting the generalized gravity contribution yields the contact target

$$
\pmb { \eta } _ { t } ^ { \star } = \pmb { \eta } _ { t } ^ { \mathrm { I D } } - \pmb { \eta } _ { t } ^ { g } \in \mathbb { R } ^ { 6 + J } ,\tag{15}
$$

whose first six components are the floating-base force and torque, expressed in the solver world frame with torque about the system center of mass; the remaining components are joint efforts. The target can be computed from smoothed base and joint trajectories or, equivalently, by projecting per-link Newton–Euler wrenches onto these generalized coordinates.

We assign each candidate hand vertex to its nearest object part. A force $\mathbf { f } _ { t , i }$ applied at $\mathbf { p } _ { t , i }$ contributes $\mathbf { f } _ { t , i }$ to the base force and $\left( \mathbf { p } _ { t , i } - \mathbf { c } _ { t } \right) \times \mathbf { f } _ { t , i }$ to torque about the system center of mass $\mathbf { c } _ { t } .$ . For a joint j ancestral to the contacted part, its joint-effort contribution is

$$
\gamma _ { t , i , j } = { \left\{ \begin{array} { l l } { \mathbf { a } _ { t , j } ^ { \top } ( ( \mathbf { p } _ { t , i } - \mathbf { o } _ { t , j } ) \times \mathbf { f } _ { t , i } ) , } & { { \mathrm { r e v o l u t e } } , } \\ { \mathbf { a } _ { t , j } ^ { \top } \mathbf { f } _ { t , i } , } & { { \mathrm { p r i s m a t i c } } , } \end{array} \right. }\tag{16}
$$

where $\mathbf { a } _ { t , j }$ and $\mathbf { o } _ { t , j }$ are the joint axis and pivot in world coordinates. Joints outside the part’s ancestor chain receive zero contribution. This part-aware map replaces the rigid grasp map in a sequencelevel allocation with the same signed-distance priority, unilateral friction constraints, and temporal force penalty.

Permitted environment support contributes to base balance. Joint effort unexplained by contact forces remains an internal, non-hand contribution instead of being assigned to hand vertices. Handfree frames have zero hand force. If a hand-contact allocation fails, we record that the force estimate is invalid or that environment support was used as a fallback, rather than treating the failed estimate as a valid hand force. We check fitted generalized effort against its target and use valid articulated vertex forces for field-level supervision, without the rigid rollout loss.

## B ADDITIONAL RESULTS AND VISUALIZATIONS

Table 5 extends the main reconstruction comparison with joint errors and hand–object F-scores. Across all six method–dataset pairs, TactoHand also improves these metrics over the base and HACO variants. Figure 7 illustrates the resulting hand–object alignment on HOCap, while Figure 8 separates the effects of the contact and proximity terms on individual examples.

![](images/8ed6a911ff1798384c9de6422c0cf729716868ed633c36d3026228dba24f823a.jpg)  
Figure 7: HOCap (Wang et al., 2025) reconstructions. Red circles highlight hand–object contact artifacts in the HACO variants.

Table 5: Full reconstruction results. PA-VPE/JPE: mm; $\mathrm { C D } _ { \mathrm { h o } } \mathrm { : }$ $1 0 ^ { - 3 } \mathrm { m } ^ { 2 } ;$ F-5/F-10: 5/10 mm; ACC: mm $/ \mathrm { s } ^ { 2 }$ . Bold marks the best per method–dataset block; shading highlights PA-VPE and $\mathrm { C D } _ { \mathrm { h o } } .$
<table><tr><td>Methods</td><td>PA-VPE↓</td><td>PA-JPE↓</td><td> $\mathrm { C D } _ { \mathrm { h o } } \downarrow$ </td><td> $\mathrm { F } { - } 5 _ { \mathrm { h o } }$  ←</td><td> $\mathrm { F } { - } 1 0 _ { \mathrm { h o } } \uparrow$ </td><td>ACC↓</td></tr><tr><td colspan="7">DexYCB</td></tr><tr><td>EasyHOI</td><td>9.937</td><td>10.483</td><td>24.140</td><td>0.3609</td><td>0.5359</td><td></td></tr><tr><td> ${ \mathrm { E a s y H O I } } + { \mathrm { H A C O } }$ </td><td>9.516</td><td>10.032</td><td>12.552</td><td>0.3609</td><td>0.5392</td><td></td></tr><tr><td> $\mathrm { E a s \dot { y } H O I + O u r s }$ </td><td>9.077</td><td>9.559</td><td>2.004</td><td>0.3817</td><td>0.5606</td><td></td></tr><tr><td>ArtHOI</td><td>7.317</td><td>7.520</td><td>11.587</td><td>0.3930</td><td>0.5240</td><td>4311.6</td></tr><tr><td> $\mathrm { A r t H O I } + \mathrm { H A C O }$ </td><td>6.027</td><td>6.190</td><td>13.627</td><td>0.3970</td><td>0.5300</td><td>3978.9</td></tr><tr><td> $\mathrm { A r t H O I + O u r s }$ </td><td>4.626</td><td>4.747</td><td>1.431</td><td>0.4540</td><td>0.6200</td><td>3892.7</td></tr><tr><td colspan="7">HOCap</td></tr><tr><td>EasyHOI</td><td>9.179</td><td>9.790</td><td>59.416</td><td>0.2306</td><td>0.3991</td><td></td></tr><tr><td> $\mathrm { E a s y H O I + H A C O }$ </td><td>9.051</td><td>9.623</td><td>68.943</td><td>0.2216</td><td>0.3824</td><td></td></tr><tr><td> $\mathrm { E a s \dot { y } H O I + O u r s }$ </td><td>7.592</td><td>8.075</td><td>11.461</td><td>0.2456</td><td>0.4319</td><td></td></tr><tr><td>ArtHOI</td><td>6.984</td><td>7.243</td><td>5.234</td><td>0.3215</td><td>0.5085</td><td>5176.9</td></tr><tr><td> $\mathrm { A r t H O I } + \mathrm { H A C O }$ </td><td>6.684</td><td>6.892</td><td>5.199</td><td>0.3249</td><td>0.5118</td><td>4666.3</td></tr><tr><td> $\mathrm { A r t H O I + O u r s }$ </td><td>6.073</td><td>6.264</td><td>1.104</td><td>0.3672</td><td>0.5869</td><td>4599.7</td></tr><tr><td colspan="7">MOW</td></tr><tr><td>EasyHOI</td><td>14.832</td><td>15.860</td><td>69.080</td><td>0.2262</td><td>0.3728</td><td></td></tr><tr><td> $\mathrm { E a s y H O I + H A C O }$ </td><td>14.276</td><td>15.215</td><td>72.638</td><td>0.2195</td><td>0.3617</td><td></td></tr><tr><td> $\mathrm { E a s y H O I + O u r s }$ </td><td>13.939</td><td>14.715</td><td>15.266</td><td>0.2370</td><td>0.3888</td><td></td></tr><tr><td>ArtHOI</td><td>13.572</td><td>14.054</td><td>73.993</td><td>0.2330</td><td>0.3640</td><td></td></tr><tr><td> $\mathrm { A r t H O I } + \mathrm { H A C O }$ </td><td>13.261</td><td>13.727</td><td>73.147</td><td>0.2340</td><td>0.3670</td><td></td></tr><tr><td> $\mathrm { A r t H O I + O u r s }$ </td><td>13.100</td><td>13.545</td><td>32.413</td><td>0.2610</td><td>0.4060</td><td></td></tr></table>

![](images/dac598bd2cf937ed0ad60112986a51291df0d03ad32eeae318906b5f95756667.jpg)  
Figure 8: EasyHOI (Liu et al., 2025) reconstruction ablation. + Contact: contact-normal term; + Proximity: distance term; + Both: both terms. Values are per-example PA-VPE (mm).

## C MODEL AND TRAINING DETAILS

## C.1 NETWORK ARCHITECTURE

A trainable DINOv3 ViT-L encodes each 256×256 RGB frame independently. We average its final patch tokens, excluding class and register tokens, to obtain one 1024-dimensional feature vector per frame $( D = 1 0 2 4$ in Figure 1). A six-layer temporal transformer with eight attention heads processes these features with learned positional embeddings. A shared two-layer, eight-head query decoder processes three learned queries, one each for force, contact, and proximity. At each image timestamp, each decoded query is fused with the corresponding frame feature to predict its field for both hands. Transformer feed-forward layers have width 4D, and dropout is zero.

The four variants combine two temporal modes (SF and T) with rollout supervision (Off and On). T processes the full clip through the temporal transformer and shared query decoder. SF applies the same modules to each frame as a length-one sequence, preserving the architecture and output shapes while removing cross-frame attention. All variants share the force, contact, and proximity head designs; On enables the rollout loss, whereas Off sets its weight to zero.

## C.2 FORCE COORDINATES, TEMPORAL SAMPLING, AND SUPERVISION

Vertex-local force coordinates. Figure 3 (lower left) shows the local basis used by the force head. Its normal is the outward normal of the posed MANO mesh; one tangent points approximately toward the wrist, and the other completes an orthonormal basis. We expand the predicted force com ponents along these axes in camera coordinates before force supervision and wrench aggregation. Contact and proximity retain canonical vertex indexing.

Vertex-force loss. Before computing the camera-frame force loss, we cap predicted and target magnitudes at 20 N by rescaling each vector without changing its direction. We then separate targets by capped magnitude: active targets above 0.05 N receive a vertex-weighted Huber penalty with a 0.05 N transition, while the remaining targets receive mean squared error weighted by 0.1.

Differentiable rigid-body rollout. The model predicts fields at 10 Hz. Each 100 ms visual interval spans three 30 Hz force intervals: the first uses predicted vertex forces, and the next two use detached reference forces. Each force interval contains four 120 Hz integration steps of size $h = 1 / 1 2 0 \mathrm { s } ,$ following Equation 17. The object state propagates continuously, without resets to ground-truth poses.

For one step, let $( \mathbf { x } , \mathbf { v } , \mathbf { q } , \omega ^ { w } )$ denote the center-of-mass position, linear velocity, orientation quaternion, and world-frame angular velocity, and let $( \mathbf { F } , \tau )$ be the net world-frame wrench about the center of mass. Define $\begin{array} { r } { \mathbf { R } ^ { } = { \mathbf { R } } ( \mathbf { q } ) , \omega ^ { \bar { b } } = { \mathbf { R } } ^ { \top } \omega ^ { \bar { w } } , \tau ^ { b ^ { } } = { \mathbf { R } } ^ { \top } \tau } \end{array}$ , and $\bar { \boldsymbol { \omega } } ^ { b } = ( \boldsymbol { \omega } ^ { b } + \boldsymbol { \omega } ^ { b , + } ) / 2$ . With body-frame inertia I, the update is

$$
\begin{array} { r l } & { \mathbf { v } ^ { + } = \mathbf { v } + \frac { h } { m } \mathbf { F } , } \\ & { \mathbf { x } ^ { + } = \mathbf { x } + h \mathbf { v } ^ { + } , } \\ & { \omega ^ { b , + } = \omega ^ { b } + h \mathbf { I } ^ { - 1 } \big ( \pmb { \tau } ^ { b } - \bar { \omega } ^ { b } \times \mathbf { I } \bar { \omega } ^ { b } \big ) , } \\ & { \omega ^ { w , + } = \mathbf { R } \omega ^ { b , + } , } \\ & { \mathbf { q } ^ { + } = \mathrm { n o r m a l i z e } \big ( \mathcal { Q } ( h \omega ^ { w , + } ) \otimes \mathbf { q } \big ) . } \end{array}\tag{17}
$$

We solve the body-frame angular-velocity update with six unrolled Newton iterations. The preupdate rotation R then maps the result to the world frame for the orientation update; Q maps an axis-angle vector to a unit quaternion. All state updates are differentiable, allowing the rollout loss to backpropagate through wrench aggregation to predicted vertex forces. Each visual interval contains one force interval driven by predictions; the other two use fixed reference forces while preserving gradients through the evolving object state.

Supervision eligibility. The available annotations determine which losses a sequence contributes to. Geometry-only sequences contribute contact and proximity losses. Rigid and articulated sequences with force labels also contribute force loss at image timestamps where those labels are valid. A field-level loss is evaluated only when the corresponding hand and label are available, regardless of whether that hand is occluded in the image. Invalid force labels do not contribute to the force loss.

Rigid rollout further requires a trajectory whose hand force can be attributed to the target object. We exclude a trajectory if, at any force-valid time step, the hand applies nonzero force to another object while the target receives none. Repeated boundary frames used to complete a visual clip do not contribute to force or rollout losses; both use only time intervals within the recorded sequence.

## C.3 TRAINING CONFIGURATION

The four variants share the final-stage settings in Table 6. Dataset sampling weights are proportional to the number of available 32-frame windows, multiplied by the listed dataset-specific factors.

Table 6: Training settings and dataset sampling multipliers. Unlisted training sources use a multiplier of 1.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Hardware and batch</td><td>One node with eight A800 GPUs; 7 samples per GPU; two accumulation steps (112 samples per update)</td></tr><tr><td>Optimizer</td><td>AdamW; learning rate  $\bar { 2 } \times 1 0 ^ { - 5 }$  for backbone and heads; weight decay  $1 0 ^ { - 4 } ;$  cosine decay with two-epoch warmup</td></tr><tr><td>Window sampling</td><td>32 frames, stride 32; each epoch samples 20% of the windows allocated to each dataset</td></tr><tr><td>Rollout-loss weight Dataset sampling multi-</td><td> $1 0 ^ { - 4 } \ \mathrm { w i t h }$  physics; 0 without physics 4: MOW; 2: DexYCB; 0.5: HOI4D, OakInk-v2; 0.1: DECAF, ReInter-</td></tr><tr><td>pliers</td><td>Hand; 0.05: HOGraspNet, ObMan, GigaHands; 0.02: InterHand2.6M</td></tr></table>

Each 150-epoch final-stage run initializes model weights from an earlier checkpoint but resets the optimizer, scheduler, and epoch state. In Equation 10, $\lambda _ { c } = 1 , \lambda _ { d } = 2 ,$ , and $\lambda _ { p } ~ = ~ 1 0 .$ , with $\bar { \mathcal { L } } _ { \mathrm { c o n t a c t } } = \mathcal { L } _ { \mathrm { c o n d i t i o n a l } } + \bar { 0 . 2 5 } \mathcal { L } _ { \mathrm { e n g a g e m e n t } } .$ No separate direct wrench loss is used. For variants with physics supervision, $\lambda _ { r }$ increases from zero in the second epoch to $1 0 ^ { - 4 }$ in the sixth epoch; it remains zero for variants without physics supervision. In Equation 9, $\mathcal { H } = \{ 1 , \ldots , 1 2 \}$ spans four 10 Hz intervals, $\lambda _ { x } = [ 2 ( 0 . 0 1 \mathrm { { m } ) ^ { 2 } ] ^ { - 1 } }$ , and $\dot { \lambda _ { R } } = [ 2 ( 0 . 1 \mathrm { r a d } ) ^ { 2 } ] ^ { - 1 }$ . Rollout losses are also averaged over eligible windows. At test time, contact is predicted when $\hat { c } > 0 . 5$

We report the checkpoints with the best validation performance. Our validation sets are the nominal DexYCB test split and the ARCTIC P2 validation split, and the reported results are measured on these same sets. HOCap is excluded from training. Its results characterize cross-dataset transfer.

## D EVALUATION PROTOCOL AND COVERAGE

## D.1 CONTACT AND PROXIMITY COMPARISONS

Table 1 re-scores released checkpoints on our geometry-derived annotations and nominal dataset splits. Input pipelines can differ. SF/T denote single-frame/temporal fusion; a dash denotes an unavailable result or undefined acceleration for static MOW. MOW uses SF for proximity and T for contact. For MOW, we form the temporal input by repeating each image across the 32-frame clip; no additional temporal observations are available.

Metric definitions. We decode predicted proximity $\widehat { r }$ as $\widehat { d } = - \log ( \widehat { r } ) / \tau$ in metres and report mean absolute distance error $( \mathrm { A D E } )$ in mm over valid vertices. ADE-100 clips both prediction and target at 100 mm before computing the error; all valid vertices remain in the average. This matches InterField’s 100 mm output range, whereas TactoHand’s target extends to 250 mm. For distance error $e _ { t , i } = \widehat { d } _ { t , i } - d _ { t , i }$ , Accel. averages $| e _ { t + 1 , i } - 2 e _ { t , i } + e _ { t - 1 , i } |$ multiplied by the squared framesampling rate, in $\mathrm { { m } / \mathrm { { s } ^ { 2 } } }$ . It measures temporal variation in distance error. We exclude any three-frame group containing a repeated frame inserted to pad a clip.

For Macro-F1, we compute F1 over the vertices of each hand in each frame, then average over frame–hand pairs with contact in either the ground truth or the prediction. Failing to predict any ground-truth contact on a hand, or predicting contact on a hand with none, gives a score of zero. Pairs with no contact in either are excluded from this average. Micro-F1 pools vertex-level true positives (TP), false positives (FP), and false negatives (FN) over all valid frame–hand pairs and computes $2 T P / ( 2 T \dot { P } + F P + F N )$ . Correctly predicted non-contact vertices do not increase either score.

Input protocols. ARCTIC uses its official P2 validation split (subject s05, egocentric view, all labeled frames); DexYCB, HOCap, and MOW use their nominal test splits. On ARCTIC, InterField uses its released P2 egocentric checkpoints with the official preprocessing of raw fisheye images and crops around the interaction bounding box. Its reported ADE against its original reference labels is 8.83 mm for SF and 8.40 mm for LSTM. On DexYCB, HOCap, and MOW, InterField instead uses P1 allocentric checkpoints on hand crops, making these out-of-domain evaluations. EgoPHI is trained only on ARCTIC and uses its released weights and deployed contact threshold of 0.3. Following its training protocol, it uses native full frames except for MOW, where it uses hand crops. TactoHand uses full-frame images on DexYCB, HOCap, and ARCTIC, and hand crops on MOW, where the hand occupies a small fraction of the frame.

## D.2 DOWNSTREAM RECONSTRUCTION PROTOCOL

We evaluate TactoHand’s contact and proximity fields in EasyHOI (Liu et al., 2025) and ArtHOI (Wang et al., 2026) on DexYCB (Chao et al., 2021), HOCap (Wang et al., 2025), and MOW (Cao et al., 2021). DexYCB and MOW are included in training; HOCap is excluded. MOW contains independent images, while the other two provide video. Within each reconstruction method, variants share the model, initialization, images, and optimization budget. Table 5 compares base, HACO (Jung & Lee, 2025), and TactoHand variants; Figure 8 isolates the EasyHOI contact and proximity terms.

Shared interaction terms. We hold the predicted contact and proximity values at the 778 MANO vertices fixed throughout reconstruction refinement. Proximity is decoded as $\widehat { d } = - \log ( \widehat { r } ) / \tau$ with $\tau = 2 0 \mathrm { { m } ^ { - 1 } }$ ; values saturated at 0.25 m specify a lower bound on distance. The proximity term applies a Huber penalty $( \delta = 1 )$ to signed distance residuals normalized by 5 mm. In the contactnormal term, predicted contact probability weights the factor max $( 0 , \mathbf { n } _ { h } ^ { \top } \mathbf { n } _ { o } ^ { \top } + 0 . 5 ) ^ { 2 }$ , where ${ \mathbf { n } } _ { h }$ and $\mathbf { n } _ { o }$ denote the hand and object surface normals. The proximity and contact-normal terms both have coefficient 1. They replace the original contact objective, while reconstruction, penetration, and regularization terms remain.

Table 7: Reconstruction stages and optimization settings shared across the compared variants.
<table><tr><td>Stage</td><td>Updates</td><td>Optimized variables</td><td>Learning rate</td></tr><tr><td>ArtHOI alignment</td><td>250</td><td>Hand translation</td><td> $1 0 ^ { - 2 } \to 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>ArtHOI hand refine- 399 ment</td><td></td><td>Finger pose, hand translation</td><td> $1 0 ^ { - 3 }  1 0 ^ { - 4 }$ </td></tr><tr><td>ArtHOI penetration 300</td><td></td><td>Hand translation, global orientation</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>correction ArtHOI smoothing</td><td>300</td><td>Hand translation</td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>EasyHOI global align- ment</td><td> $2 0 \times 1 0 0$ </td><td>Hand-global parameters, object trans- lation</td><td> $5 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>EasyHOI hand refine- 1,000 ment</td><td></td><td>Pose residual, object translation; MANO shape</td><td> $1 0 ^ { - 2 } ; 1 0 ^ { - 4 }$  for shape</td></tr></table>

EasyHOI refinement. We fix metric CAD geometry and ground-truth object orientation, optimizing translation from its image-based initialization. Table 7 lists $3 { , } 0 0 0$ updates, excluding imagebased initialization and auxiliary PCA fitting, with constant learning rates within each stage. Hand refinement starts from the aligned hand parameters, without backpropagating through the preceding alignment stage; the hand’s root rotation remains fixed during refinement. During global alignment, interaction losses are enabled under the same conditions as in the original method. The proximity term ramps up over the first 50 updates in which those losses are evaluated, while the contact-normal term is off. During refinement, proximity is fully enabled and the contact-normal term ramps up over 10 such updates.

ArtHOI refinement. We fix metric object geometry, skip scale optimization, and use Adam with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ . Table 7 lists the 1,249 refinement updates performed after initialization. For the first two stages, learning rates are taken from step 151 onward of the original 800-step schedule. An L1 penalty with coefficient 0.4 discourages deviation from the initial pose. The proximity term ramps up over 50 alignment iterations, and the contact-normal term over 10 handrefinement iterations. Both remain active during penetration correction; only proximity remains during smoothing.

PA-VPE and PA-JPE are Procrustes-aligned mean vertex and joint errors, respectively, in mm. $\mathrm { { C D } _ { \mathrm { { h o } } } }$ sums bidirectional mean squared nearest-neighbor distances in $1 0 ^ { - 3 } \mathrm { m ^ { 2 } ; \vec { F } \mathrm { - 5 _ { h o } } }$ and $\mathrm { F { - } 1 0 _ { h o } }$ are hand–object F-scores at 5 and 10 mm. ACC is acceleration error in $\mathrm { m m / s ^ { 2 } }$ , available only for ArtHOI on DexYCB and HOCap; dashes in Table 5 mean unavailable. These are reconstruction outcomes, not direct contact or proximity scores.

## D.3 FORCE AND ROLLOUT EVALUATION

Table 2 evaluates four single-frame/temporal × physics-off/on variants on the DexYCB test split. They share data, backbone, and optimization settings (Appendix C.1); physics-on adds rigid rollout loss. Force metrics compare predicted and motion-derived reference vectors in the camera frame at image timestamps with valid labels, excluding repeated frames used for padding. Vertex MAE averages component-wise absolute error over all supervised vertices, including zero-force targets. Active MAE and force angle use only targets above 0.05 N; angle is the mean predicted-to-reference vector angle. MAEs are in mN and angles in degrees.

Rollout uses a 0.4 s horizon: four 10 Hz visual intervals or twelve 30 Hz force intervals. Within each visual interval, predicted forces drive the first force interval and detached reference forces drive the next two; four 120 Hz updates integrate each force interval. Sliding windows start from observed object states and include specified gravity and estimated environment-support forces. We score only intervals within the recorded sequence, excluding those added to pad a clip. Position ADE averages object-center displacement error over the horizon, and rollout angle averages rotation geodesic error. A window is excluded if any state is non-finite or its linear or angular speed exceeds 100 m/s or 1000 rad/s, respectively. Reported errors therefore describe the non-divergent windows of this reference-assisted rollout.