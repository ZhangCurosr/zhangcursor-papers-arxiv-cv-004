# Where Do the Gaussians Go? Foreground Attraction for Camera-Based Weakly Supervised 3D Occupancy

Jeagon Song<sup>1</sup> and Jhonghyun An<sup>1,∗</sup>

Abstract— Camera-based 3D semantic occupancy models can be trained without 3D labels by rendering their predictions into the images and supervising them with 2D pseudo labels. In sparse Gaussian representations trained this way, Gaussian positions are learned only through pixel-wise rendering losses, which we hypothesize give small foreground objects little weight. In trained GaussianFlowOcc models, the Gaussian density on foreground voxels is on average only 1.62 times that on background voxels. We propose a training-only foreground attraction loss that back-projects foreground pseudo-label pixels with pseudo depth and pulls nearby Gaussians toward these 3D points, favoring Gaussians that already predict the target class; the inference network is unchanged. On Occ3D-nuScenes, the loss raises this concentration to 2.40 over three seeds. Over three seeds, the mean foreground IoU increases from 6.29 to 6.96 and is higher for each seed, whereas the change in mean mIoU (13.53 to 13.82) is within the seed-to-seed variation of the baseline. Single-run ablations further indicate that pseudo-label targets outperform targets placed at ground-truth voxels and that acting on the Gaussian positions alone is not sufficient, and a single-seed test shows that the model degrades less when fewer Gaussians are kept at inference.

## I. INTRODUCTION

Camera-based 3D semantic occupancy prediction describes the surroundings of a vehicle as a voxel grid in which every cell is labeled as free or as one of several semantic classes, providing a dense representation for planning in autonomous driving [1]. Dense 3D occupancy labels, however, are costly to produce, as they rely on additional data such as LiDAR [2]. This has motivated methods that learn occupancy from 2D signals alone, by rendering the predicted 3D scene into the cameras and supervising the renderings with 2D labels, pseudo labels, photometric consistency, or features of 2D foundation models [2]–[7].

Gaussian-based occupancy models replace the dense voxel grid with a sparse set of 3D Gaussians [2], [8]. Gaussian-FlowOcc [2] is a representative weakly supervised example: it predicts a fixed number of Gaussians with temporal flow from multi-view images and trains them with semantic pseudo labels from Grounded SAM [9] and pseudo depth from Metric3D [10]. Because the number of Gaussians is fixed, where the network places them determines how much of the representation is spent on each part of the scene.

We argue that rendering supervision provides only a weak signal for placing Gaussians on foreground objects such as vehicles, pedestrians, and barriers. Rendering losses are computed per pixel, so an object that covers few pixels contributes little to the loss, while Gaussians on large surfaces such as roads and buildings affect many pixels. We find that averaged over three trained GaussianFlowOcc models, foreground voxels receive 132.6 Gaussians per validation frame, compared with 1193.7 for background voxels, and that the Gaussian density per volume on the foreground is only 1.62 times that on the background.

![](images/18ad06080861f9efee39ff1d75671daaaeb9b74d0970db68f5d1ee4991229eaf.jpg)  
Fig. 1. Overview. The baseline learns Gaussian positions only through 2D rendering losses on pseudo semantic labels and pseudo depth. Our foreground attraction loss (dashed, training only) back-projects foreground pseudo-label pixels to 3D and pulls nearby Gaussians toward them, favoring Gaussians that already predict the target class. The inference network is unchanged. Image thumbnails are schematic illustrations.

Existing ways to control the distribution of Gaussians do not directly address this setting. 3D Gaussian splatting interleaves optimization and density control for each captured scene [11], and F4Splat predicts where to densify in a feedforward manner with an explicit control of the Gaussian budget [12]; F4Splat targets novel-view synthesis rather than occupancy prediction. GaussianFormer-2 observes that many Gaussians in occupancy models describe empty space and initializes them in non-empty regions [13]. Simply adding Gaussians also appears to be of limited use in our setting: doubling their number changed mIoU by less than 0.2 points in two single-run tests with attraction-trained models, although we did not repeat this test for the baseline (Sec. IV-D). We instead keep the architecture and the number of Gaussians unchanged and add a training-only 3D objective: foreground pixels of the existing pseudo labels are back projected with the pseudo depth, and nearby Gaussians that already predict the corresponding class are pulled toward them (Fig. 1).

Our contributions are as follows:

• We quantify how Gaussians are distributed between foreground and background under rendering supervision by assigning each Gaussian to the ground-truth voxel containing its center, and show that the resulting foreground concentration is weak (Sec. IV-C).

• We propose a foreground attraction loss that derives its targets only from the pseudo labels of the baseline, combines distance- and class-aware soft assignment with a robust distance, and leaves the inference network and its cost unchanged (Sec. III).

• On Occ3D-nuScenes, we show over three seeds that the loss increases the foreground concentration from 1.62 to 2.40 and raises the mean foreground IoU while the change in mIoU stays within seed variation; we further analyze the target source, the gradient path, and robustness to a reduced inference budget (Sec. IV).

## II. RELATED WORK

## A. Camera-Based 3D Semantic Occupancy

Occ3D [1] builds dense, visibility-aware occupancy labels for nuScenes [14] and Waymo, establishing the Occ3DnuScenes and Occ3D-Waymo benchmarks. Camera-based methods infer occupancy from a single image, as in semantic scene completion with MonoScene [15], or from multiple cameras with tri-perspective-view features in TPV-Former [16] and dual-path transformers in OccFormer [17]. SurroundOcc [18] supervises multi-camera occupancy prediction at multiple scales with dense occupancy labels generated by fusing multi-frame LiDAR scans. Subsequent work has focused on efficient scene representations. FlashOcc [19] keeps features in bird’s-eye view and lifts them to 3D with a channel-to-height transformation, and SparseOcc [20] replaces dense latent volumes with a sparse latent representation to reduce computation. GaussianFormer [8] describes the scene with sparse, object-centric semantic Gaussians that are refined from multi-view images and splatted into voxels, and GaussianFormer-2 [13] improves efficiency with a probabilistic Gaussian superposition and an initialization in nonempty regions. EmbodiedOcc [21] keeps a global memory of 3D semantic Gaussians and updates the Gaussians in observed regions online. Our work builds on such Gaussian representations but addresses where the Gaussians are placed when no 3D labels are available for supervision.

## B. Occupancy Learning from 2D Supervision

A growing line of work avoids dense 3D labels by rendering the 3D prediction into the cameras. RenderOcc [3] volume-renders a NeRF-style volume and trains it with 2D semantic and depth labels. SelfOcc [4] learns occupancy from video by rendering previous and future frames, and OccNeRF [5] renders depth supervised by multi-frame photometric consistency and obtains semantics from an openvocabulary 2D segmentation model. GaussianOcc [7] uses Gaussian splatting for fully self-supervised occupancy estimation without ground-truth poses, and GaussTR [6] aligns splatted Gaussian features with foundation models. Gaussian-FlowOcc [2] combines sparse Gaussians with temporal flow and supervises them with pseudo semantic labels and pseudo depth. Rendering has also been used as an auxiliary training signal with 3D labels: GaussRender [22] projects predicted and ground-truth occupancy into camera views with Gaussian splatting and enforces their consistency, leaving inference unchanged. These methods differ in how the 2D supervision is obtained, but in the weakly and self-supervised ones the 3D prediction is supervised only through rendered 2D views. We complement this supervision with a 3D objective derived from the same pseudo labels.

## C. Controlling the Distribution of Gaussians

In 3D Gaussian splatting [11], the set of Gaussians is adapted by density control during the optimization of each scene, and Rota Bulo et al. [23] revise this densification\` with a pixel-error criterion and a limit on the total number of primitives. Feed-forward methods predict Gaussians directly from images: pixelSplat [24] samples Gaussian centers from predicted probability distributions, and MVSplat [25] localizes them with a plane-sweep cost volume. F4Splat [12] predicts densification scores in a feed-forward manner, allocating Gaussians according to spatial complexity and view overlap while allowing explicit control of the final budget. For occupancy, MR-Occ [26] refines features only for critical voxels in a camera-LiDAR setting, and GaussianFormer-2 [13] initializes Gaussians in non-empty regions. A simpler alternative is to reweight the rendering losses by class frequency, which the public GaussianFlowOcc implementation offers as an option; such reweighting changes the contribution of pixels in image space but provides no 3D target for the Gaussian positions. In contrast to methods that adapt the number, initialization, or resolution of the representation, we keep the Gaussian budget and the network fixed and change only the training objective so that the learned positions concentrate on foreground objects.

## III. METHOD

Fig. 1 summarizes our approach. We keep the Gaussian-FlowOcc architecture and its training losses unchanged [2] and add a single loss term that is active only during training. The term pulls Gaussian centers toward 3D foreground points obtained from the same 2D pseudo labels that supervise the rendering losses.

## A. Preliminaries: Gaussian Occupancy from 2D Supervision

GaussianFlowOcc represents the scene around the ego vehicle with N 3D Gaussians and predicts semantic occupancy from multi-view camera images [2]. Each Gaussian i carries a center $\pmb { \mu } _ { i }$ and semantic logits $\mathbf { z } _ { i } \in \mathbb { R } ^ { C }$ with $C = 1 6$ , among other attributes. The decoder consists of $L = 3$ blocks, each of which refines a query feature $\mathbf { f } _ { i }$ through temporal, self-, and cross-attention; each decoder block ends with a rectifier that moves the Gaussian by predicting a residual from its current center:

$$
\pmb { \mu } _ { i } ^ { ( l ) } = \pmb { \mu } _ { i } ^ { ( l - 1 ) } + r _ { \theta } ^ { ( l ) } \bigl ( \mathbf { f } _ { i } ^ { ( l ) } \bigr ) , \qquad l = 1 , \ldots , L ,\tag{1}
$$

where $\mu _ { i } ^ { ( 0 ) }$ is the initial center and $\mu _ { i } = \mu _ { i } ^ { ( L ) }$ . The model is not supervised with 3D occupancy labels. Instead, the

Gaussians are splatted into each camera, and the rendered semantics and depth are supervised by a semantic loss $\mathcal { L } _ { \mathrm { s e m } }$ against Grounded SAM pseudo labels [9] and a depth loss ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ against Metric3D pseudo depth [10]. We denote the complete original training objective by $\mathcal { L } _ { \mathrm { b a s e } }$ . Consequently, the rectifiers $\mathbf { \widetilde { \Gamma } } _ { r _ { \theta } } ^ { ( l ) }$ , which decide where the Gaussians go, are trained only through these image-space losses.

## B. Foreground Concentration under Rendering Supervision

We hypothesize that rendering supervision gives the rectifiers only a weak incentive to place Gaussians on foreground objects. Rendering losses are computed per pixel, so the influence of a scene element grows with its projected area: a pedestrian or a traffic cone covers few pixels, whereas a Gaussian on the road or on a building facade affects many. To examine the resulting placement, we assign each Gaussian to the ground-truth voxel that contains its center; ground truth is used here only for analysis. Averaged over three trained baseline models and all validation frames, 132.6 Gaussians per frame lie in foreground voxels and 1193.7 in background voxels. Normalized by the occupied volume of each group, the Gaussian density on the foreground is only 1.62 times that on the background (Sec. IV-C). Rather than adding Gaussians, we aim to raise this concentration by giving the rectifiers a 3D objective whose strength does not depend on projected area.

## C. Foreground Targets from 2D Pseudo Labels

We construct attraction targets from the pseudo labels that already supervise the baseline, so no new supervision source is introduced. Let $\mathcal { C } _ { \mathrm { f g } }$ denote the set of foreground classes (barrier, bicycle, bus, car, construction vehicle, motorcycle, pedestrian, traffic cone, trailer, and truck). For camera m with intrinsic matrix ${ \bf K } _ { m }$ and extrinsic transform $\mathbf { T } _ { m } .$ , a pixel $( u , v )$ is back-projected with its pseudo depth $D _ { m } ( u , v )$ only if its pseudo label $S _ { m } ( u , v )$ belongs to ${ \mathcal { C } } _ { \mathrm { f g } } ,$ all pixels in its $7 \times 7$ neighborhood carry the same label, its pseudo depth is at most 40 m, and the resulting point lies inside the occupancy grid:

$$
{ \binom { \mathbf { x } _ { p } } { 1 } } = \mathbf { T } _ { m } ^ { - 1 } \left[ { \cal D } _ { m } ( u , v ) \mathbf { K } _ { m } ^ { - 1 } [ u , v , 1 ] ^ { \top } \right] , \qquad c _ { p } = S _ { m } ( u , v ) .\tag{2}
$$

The neighborhood condition removes pixels near label boundaries, where monocular depth tends to be blurred across object edges; without it, a foreground label could be paired with the depth of an adjacent surface and back projected to a wrong location. For every training sample, we select $P$ targets from these candidates. If there are at most $P$ candidates, all of them are used; otherwise, $P$ candidates are drawn without replacement, each weighted by the inverse of the number of candidates of its class, so that every class present contributes a similar number of targets. Because the targets combine two pseudo-label sources, they contain class and depth errors; the loss below is designed to tolerate them.

## D. Foreground Attraction Loss

For each target $p ,$ we consider only the set $\textstyle { \mathcal { N } } _ { K } ( p )$ of the $K$ Gaussians whose centers are closest to $\mathbf { x } _ { p } ,$ with distances $d _ { p i } = \lVert \mathbf { x } _ { p } - \pmb { \mu } _ { i } \rVert _ { 2 }$ . A Gaussian should be responsible for the target when it is close and when it already predicts the target class. We express the second condition with the class probability $s _ { p i } = \mathrm { s o f t m a x } ( \mathbf { z } _ { i } ) _ { c _ { p } }$ and define the soft assignment

$$
\tilde { a } _ { p i } = \frac { \exp ( - d _ { p i } ^ { 2 } / \tau ) } { \sum _ { j \in \mathcal { N } _ { K } ( p ) } \exp ( - d _ { p j } ^ { 2 } / \tau ) } \left( \epsilon + s _ { p i } \right) ,\tag{3}
$$

$$
\begin{array} { r } { a _ { p i } = \tilde { a } _ { p i } \bigg / \sum _ { j \in \mathcal { N } _ { K } ( p ) } \tilde { a } _ { p j } , } \end{array}\tag{4}
$$

where $\tau$ is a distance temperature and ϵ is a floor on the class term. The attraction loss is the class-weighted average of robust distances over all targets:

$$
\mathcal { L } _ { \mathrm { a t t } } = \gamma ( t ) \frac { \sum _ { p } w _ { c _ { p } } \sum _ { i \in \mathcal { N } _ { K } ( p ) } a _ { p i } \rho _ { \delta } ( d _ { p i } ) } { \sum _ { p } w _ { c _ { p } } } ,\tag{5}
$$

where $\rho _ { \delta }$ is the Huber function with threshold $\delta ,$

$$
\rho _ { \delta } ( d ) = \left\{ { \begin{array} { l l } { \frac { 1 } { 2 } { d } ^ { 2 } , } & { d \leq \delta , } \\ { \delta \left( d - \frac { 1 } { 2 } \delta \right) , } & { d > \delta . } \end{array} } \right.\tag{6}
$$

Without the class term and the class weights, Eq. (5) reduces to a soft, one-sided Chamfer-type distance from the target points to the Gaussian centers [27]; the class term restricts which Gaussians are attracted. The class weight is a normalized log inverse frequency,

$$
w _ { c } = \frac { \bar { C } \log ( F / F _ { c } ) } { \sum _ { c ^ { \prime } } \log ( F / F _ { c ^ { \prime } } ) } , \qquad F = \sum _ { c } F _ { c } ,\tag{7}
$$

where $F _ { c }$ is the voxel frequency of class c and the sums run over the $\bar { C } = 1 7$ classes of the frequency table, which, unlike the semantic logits, include the others class. We use the per-class voxel frequencies that are provided with the public implementation of GaussianFlowOcc [2]; these dataset-level statistics are the only quantity derived from 3D labels that enters training. The schedule $\gamma ( t )$ is zero for the first $t _ { \mathrm { w } }$ iterations, increases linearly to one over the next $t _ { \mathrm { r } }$ iterations, and stays at one afterwards.

## E. Design Rationale

Each component of Eqs. (3)–(7) is motivated by a failure mode that we expect without it; we did not ablate the components individually. Local candidates. Restricting the assignment to $\mathcal { N } _ { K } ( \boldsymbol { p } )$ keeps the pull local; attracting all Gaussians would drag the whole scene toward the foreground. Distance term. The softmax over $- d _ { p i } ^ { 2 } / \tau$ acts as a soft nearest-neighbor selection, so the one or two closest Gaussians receive most of the assignment weight. Class term. The factor $\epsilon + s _ { p i }$ favors Gaussians that already describe the target class, so the loss is less likely to pull Gaussians away from neighboring background surfaces and leave holes there. The floor ϵ keeps the assignment defined when no nearby Gaussian predicts the class yet, which is exactly where attraction is most needed. Normalization. Because $\textstyle \sum _ { i } a _ { p i } = 1$ , every target demands the same total attraction, and well-covered targets are not penalized more than poorly covered ones. Robust distance. The targets inherit monocular depth errors, and the Huber function bounds the gradient from far-away outliers that a squared loss would amplify. Warm-up. The semantic logits are uninformative at the start of training, so $\gamma ( t )$ delays attraction until the class term becomes meaningful.

Gradients of $\mathcal { L } _ { \mathrm { a t t } }$ reach both the centers $\mu _ { i } ,$ , and thus the rectifiers in $\operatorname { E q . } \ ( 1 ) .$ , and the semantic logits $\mathbf { z } _ { i }$ through $s _ { p i }$ We keep both paths. In a single-seed ablation with pseudolabel targets, stopping the gradient through $s _ { p i }$ reduced mIoU from 13.80 to 12.14, whereas the same change raised mIoU when the targets were taken from 3D ground truth (Sec. IV-D).

## F. Training and Inference

The model is trained end to end with

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { b a s e } } + \lambda \mathcal { L } _ { \mathrm { a t t } } ,\tag{8}
$$

where λ balances the attraction term against the rendering losses. The targets and $\mathcal { L } _ { \mathrm { a t t } }$ are computed only during training. The inference network is therefore identical to the baseline by construction, with the same number of Gaussians, parameters, and operations (Sec. IV-E).

## IV. EXPERIMENTS

## A. Setup

Dataset and metrics. We evaluate on Occ3DnuScenes [1], [14], whose 3D occupancy labels are used only for evaluation and analysis. Owing to our compute budget, all models are trained on the same 100 scenes, sampled at random from the first 385 of the 700 training scenes (in dataset order), for which pseudo labels had been generated, and evaluated on all 150 validation scenes. We evaluate only camera-visible voxels at a resolution of 0.4 m. We report the mean IoU (mIoU) over 15 semantic classes (excluding others and other flat), the mean IoU over the 10 foreground classes (FG mIoU), and the occupancy IoU, i.e., the IoU of the occupied class in the binary occupied/free classification. Because the original GaussianFlowOcc results are obtained with all 700 training scenes [2], our numbers are not comparable to those reported in that work; we therefore compare against GaussianFlowOcc trained and evaluated in the same setting.

Pseudo labels. Semantic pseudo labels are generated with Grounded SAM [9], which combines the open-set detector Grounding DINO [28] with the promptable segmentation model SAM [29], in one of two prompting modes: separate, which queries each class name individually, and combine, which concatenates all class names into one prompt. In the combine mode, the class name “traffic cone” is split into several tokens and cannot be recovered from the prompt. The baseline trained with combine-mode labels obtains zero IoU on construction vehicle, traffic cone, and trailer. Our main comparison uses separate-mode labels with three training seeds for both the baseline and our method. The ablation in

TABLE I  
MAIN COMPARISON ON OCC3D-NUSCENES (SEPARATE-MODE PSEUDO  
LABELS, TRAINED ON 100 SCENES, EVALUATED ON ALL 150 VALIDATION SCENES, CAMERA-VISIBLE VOXELS). $\small \mathbf { M E A N \pm S A M P L E }$ STANDARD DEVIATION OVER THREE SEEDS. MIOU: MEAN IOU OVER 15 SEMANTIC CLASSES; FG MIOU: MEAN IOU OVER THE 10 FOREGROUND CLASSES; OCC. IOU: OCCUPANCY IOU.
<table><tr><td>Method</td><td>mIoU</td><td>FG mIoU</td><td>Occ. IoU</td></tr><tr><td>GaussianFlowOcc</td><td> $1 3 . 5 3 \pm 0 . 6 2$ </td><td> $6 . 2 9 \pm 0 . 5 0$ </td><td> $3 7 . 7 2 \pm 1 . 2 5$ </td></tr><tr><td>Ours</td><td> $1 3 . 8 2 \pm 0 . 1 0$ </td><td> $6 . 9 6 \pm 0 . 2 7$ </td><td> $3 7 . 5 2 \pm 0 . 7 4$ </td></tr></table>

Sec. IV-D was run earlier with combine-mode labels, a single run per variant, and a shared seed; we report it separately and do not compare its numbers with the main results.

Implementation details. Both methods share the GaussianFlowOcc configuration with $N = 1 0 { , } 0 0 0$ Gaussians, 18 training epochs, a batch size of 2, and gradient clipping at a norm of 5, trained on a single RTX 5080 GPU with 16 GB of memory. For the attraction loss, we use $K = 8 , \tau = 1 . 0 \mathrm { m ^ { 2 } } .$ ϵ = 0.1, δ = 1.0 m, λ = 0.3, P = 512, and $t _ { \mathrm { w } } = t _ { \mathrm { r } } = 2 0 0 0$ iterations. These values were set once at design time and were not tuned on the validation set. Training the baseline takes approximately 12 h.

## B. Main Results

Table I compares our method with the baseline over three seeds. Runs with the same seed index share the random seed, so we also compare them pairwise. Foreground attraction raises the mean FG mIoU from 6.29 to 6.96, and the FG mIoU is higher for each of the three seeds (+0.51, +0.21, +1.30). The mean mIoU changes from 13.53 to 13.82, but per seed index the difference is −0.12, −0.03, and +1.03, and the mean difference of 0.29 is smaller than the standard deviation of the baseline (0.62). We therefore do not claim an improvement in mIoU; the FG difference of 0.67 is comparable to the standard deviation of the baseline (0.50), so the results are consistent with, but do not establish, a gain on the foreground classes. The seed-to-seed standard deviation of mIoU is lower for our method (0.10 vs. 0.62), but three seeds are too few to establish a difference in variance. The occupancy IoU is essentially unchanged (37.72 for the baseline and 37.52 for ours; per seed −0.65, −0.68, and +0.72), so the loss does not improve the overall geometry.

Table II shows where the change comes from. The IoU increases for seven of the ten foreground classes, with the largest gain on bus (+3.94) and gains of +0.73 to +1.12 on pedestrian, bicycle, barrier, and traffic cone. Construction vehicle, trailer, and truck decrease slightly (−0.58, −0.11, and −0.74). On the five background classes, the mean IoU decreases by 0.46, mainly on sidewalk (−1.89) and vegetation (−0.43), while the mean over the foreground classes increases by 0.67. On average, the gains on objects are accompanied by a small decrease on background surfaces.

TABLE II  
PER-CLASS IOU ON OCC3D-NUSCENES VALIDATION (CAMERA-VISIBLE VOXELS), AVERAGED OVER THREE SEEDS WITH SEPARATE-MODE PSEUDO LABELS; SAME SETTING AS TABLE I. MANY PER-CLASS DIFFERENCES ARE SMALLER THAN THE SEED-TO-SEED VARIATION. THE FIRST TEN COLUMNS ARE FOREGROUND CLASSES.
<table><tr><td>Method</td><td>barr.↑</td><td>bicy.↑</td><td>bus↑</td><td>car↑</td><td>c.veh.↑</td><td>moto.↑</td><td>ped.↑</td><td>cone↑</td><td>trail.↑</td><td>truck↑</td><td>drive.↑</td><td>sidew.↑</td><td>terr.↑</td><td>manm.↑</td><td>veg.↑</td><td>mIoU↑</td></tr><tr><td>GaussianFlowOcc</td><td>5.44</td><td>2.55</td><td>9.05</td><td>14.11</td><td>1.42</td><td>6.37</td><td>7.04</td><td>6.68</td><td>0.42</td><td>9.79</td><td>57.64</td><td>28.81</td><td>31.83</td><td>11.94</td><td>9.88</td><td>13.53</td></tr><tr><td>Ours</td><td>6.24</td><td>3.67</td><td>12.99</td><td>14.25</td><td>0.84</td><td>6.88</td><td>7.77</td><td>7.59</td><td>0.31</td><td>9.05</td><td>58.26</td><td>26.92</td><td>31.25</td><td>11.92</td><td>9.45</td><td>13.82</td></tr></table>

TABLE III

GAUSSIAN PLACEMENT ON ALL 150 VALIDATION SCENES(SEPARATE-MODE PSEUDO LABELS; MEAN ± SAMPLE STANDARDDEVIATION OVER THREE SEEDS). A GAUSSIAN IS ASSIGNED TO THEGROUND-TRUTH VOXEL CONTAINING ITS CENTER; GAUSSIANS IN FREEVOXELS ARE IGNORED. FG/FRAME: GAUSSIANS IN FOREGROUNDVOXELS PER FRAME. FG/BG DENS.: GAUSSIANS PER $\mathbf { M } ^ { 3 }$ OFGROUND-TRUTH FOREGROUND/BACKGROUND VOXELS. RATIO: FG/BGDENSITY (CONCENTRATION).
<table><tr><td>Method</td><td>FG/frame</td><td>FG dens.</td><td>BG dens.</td><td>Ratio</td></tr><tr><td>Baseline</td><td> $1 3 2 . 6 \pm 1 1 . 9$ </td><td> $1 . 9 1 \pm 0 . 1 7$ </td><td> $1 . 1 8 \pm 0 . 0 5$ </td><td> $1 . 6 2 \pm 0 . 2 1$ </td></tr><tr><td>Ours</td><td> $2 2 8 . 8 \pm 1 9 . 9$ </td><td> $3 . 3 0 \pm 0 . 2 9$ </td><td> $1 . 3 8 \pm 0 . 1 2$ </td><td> $2 . 4 0 \pm 0 . 1 1$ </td></tr></table>

## C. Where Do the Gaussians Go?

We next check whether the loss changes the placement of Gaussians as intended; since the loss directly pulls Gaussians toward foreground points, this analysis is a sanity check of its mechanism rather than evidence for the accuracy gains. For each Gaussian, we look up the ground-truth voxel that contains its center and label the Gaussian as foreground or background accordingly, ignoring Gaussians in free voxels, over all validation frames and for the three seeds of each method. Table III reports the counts and the number of Gaussians per cubic meter of ground-truth voxels in each group.

With the attraction loss, the number of foreground Gaussians per frame increases from 132.6 to 228.8 on average. The foreground density rises from 1.91 to $3 . 3 0 \mathrm { m ^ { - 3 } }$ , whereas the background density changes only from 1.18 to $1 . 3 8 \mathrm { m ^ { - 3 } }$ The ratio between the two, which we call concentration, grows from 1.62 to 2.40, and every seed of our method has a higher concentration than every baseline seed. The number of Gaussians in background voxels also grows (from 1193.7 to 1389.3 per frame), so the additional foreground Gaussians are not taken away from background surfaces. On the first 400 validation frames, most Gaussians lie outside the camera-visible voxels that the benchmark evaluates (80.0 % for the baseline and 75.9 % for ours, averaged over seeds), and a substantial share lies outside the voxel grid (31.6 % and 26.1 %, the latter varying by 11.0 points across seeds). Fig. 2 shows an example validation frame rendered with the separate-mode models of Sec. IV-B, in which more of our Gaussians appear near vehicles and pedestrians.

TABLE IV  
ABLATION OF THE ATTRACTION TARGET AND GRADIENT PATH (SINGLE RUN PER ROW, SAME SEED, COMBINE-MODE PSEUDO LABELS; 150 VALIDATION SCENES). POS./CLS.: SOURCE OF THE TARGET POSITIONS AND OF THE TARGET CLASSES; DETACH: STOP-GRADIENT THROUGH THE CLASS TERM OF THE ASSIGNMENT. IOU: OCCUPANCY IOU. BEST IN BOLD.
<table><tr><td>Variant</td><td>Pos.</td><td>Cls.</td><td>Detach mIoU↑</td><td></td><td>IoU↑</td></tr><tr><td>No attraction</td><td></td><td>一</td><td>一</td><td>13.40</td><td>38.38</td></tr><tr><td>Pseudo (ours)</td><td>pseudo pseudo</td><td></td><td>no</td><td>13.80</td><td>39.54</td></tr><tr><td>Pseudo + detach</td><td>pseudo</td><td>pseudo</td><td>yes</td><td>12.14</td><td>36.77</td></tr><tr><td>GT</td><td>GT</td><td>GT</td><td>no</td><td>12.91</td><td>37.80</td></tr><tr><td>GT + detach</td><td>GT</td><td>GT</td><td>yes</td><td>13.61</td><td>39.46</td></tr><tr><td>Mix</td><td>GT</td><td>pseudo</td><td>no</td><td>12.21</td><td>36.38</td></tr><tr><td>Mix + detach</td><td>GT</td><td>pseudo</td><td>yes</td><td>13.39</td><td>38.33</td></tr></table>

## D. Ablation: Target Source and Gradient Path

Table IV varies where the targets come from and whether the gradient through the class term $s _ { p i }$ is stopped; all rows use combine-mode labels and a single run. GT replaces the pseudo-label targets with the centers and classes of cameravisible Occ3D voxels of the foreground classes, and Mix keeps the ground-truth positions but takes the class from the pseudo label of the pixel onto which each voxel projects; if the voxel is occluded in that pixel (depth mismatch above 2 m), the class term is dropped and the assignment uses distance only. Because each variant was trained once, we read these differences as indications rather than conclusive rankings.

Ground-truth target positions do not help by themselves. Pseudo-label targets reach 13.80 mIoU, whereas GT and Mix targets reach only 12.91 and 12.21, both below the baseline without attraction (13.40). A natural explanation is that ground-truth classes disagree with the pseudo labels that drive the rendering losses: on 300 training frames, 77.7 % of the labeled pixels agree with the projected front surface of the Occ3D labels, and the agreement is 0 % for construction vehicle, traffic cone, and trailer. However, Mix uses the pseudo-label classes and performs worst, so class disagreement alone does not explain the drop. Both variants with ground-truth positions also show larger gradients: the median gradient norm over training is 4.72 with pseudo-label targets, 7.32 with GT, and 8.71 with Mix targets, and at the last epoch gradient clipping is active in all iterations of the

![](images/850104b3173f84adb9a263377782d47304e725b56492692f503c76a270bc6284.jpg)  
Fig. 2. Gaussian placement on a validation frame (scene-0016, frame 8). Left: ground-truth occupancy (camera-visible voxels). Middle: GaussianFlowOcc. Right: ours. Each Gaussian is drawn at its center and scale and colored by its predicted class; background classes are shown in gray. Both models are trained with separate-mode pseudo labels and seed 0.

GT and Mix runs but in none of ours. Raising the clipping threshold did not recover the accuracy: resuming the Mix run from epoch 8 with a threshold of 20 yields 12.15 mIoU, compared with 12.21 without this change.

Stopping the class gradient. When the gradient through $s _ { p i }$ is stopped, the GT and Mix variants recover to 13.61 and 13.39 mIoU, whereas our pseudo-label variant drops from 13.80 to 12.14. Stopping the gradient thus helps both variants with ground-truth positions but hurts ours; with single runs, we cannot determine the cause. Notably, with the stopped gradient our loss acts only on the positions and its mIoU falls below the baseline (12.14 vs. 13.40), suggesting that moving Gaussians toward foreground points is not sufficient by itself and that the effect of the loss on the semantic logits contributes to its benefit. Consistent with this reading, placement statistics of the combine-mode models show that this variant still concentrates Gaussians on the foreground (concentration 3.45, compared with 1.49 for the baseline and 4.48 for our model). Finally, doubling the number of Gaussians from 10,000 to 20,000 changed mIoU by only −0.19 for our model and +0.11 for the GT variant with stopped gradients, suggesting that adding Gaussians is not an effective alternative in this setting, although the baseline was not tested.

## E. Inference Budget and Cost

Because the attraction loss is used only for training, our model and the baseline share the same inference network, which has 39.245 M parameters, 391.48 G FLOPs, and 13.7 frames per second on an RTX 5080 GPU with a batch size of 1, a 256 × 704 input, and 32-bit floating-point precision. We further examine how the two models behave when only the B Gaussians with the highest opacity are kept at inference, without retraining. This experiment uses the separate-mode models with seed 0 and all 150 validation scenes.

As shown in Fig. 3, the two models differ little at $B \geq$ 6,000 (mIoU −0.12 to +0.26, occupancy IoU −0.91 to −0.55), but our model degrades much less when the budget is reduced. At $B = 4 { , } 0 0 0$ , it is higher by 1.23 mIoU and 2.19 occupancy IoU, and at $B = 2 { , } 0 0 0$ by 3.39 and 14.11. This single-seed result suggests that concentrating Gaussians on objects makes the representation more robust to a reduced inference budget.

![](images/f5604fc0c1361e3cb8ed899f0adee043b74c08e5710051a7ab7b70f226548a98.jpg)  
Fig. 3. Accuracy when only the B Gaussians with the highest opacity are kept at inference (no retraining). All 150 validation scenes; separate-mode models with seed 0.

## F. Limitations

Our study has several limitations. All models are trained on a single GPU, using 100 scenes drawn from the first 385 of the 700 training scenes. So the training data may not represent the full training set, and whether the gains persist with all training scenes remains open. With three seeds, the mIoU and occupancy IoU differences are smaller than the variation of the baseline, the ablation relies on single runs with combine-mode pseudo labels, and the budget analysis uses a single seed. The evaluation covers only camera-visible voxels, so occluded regions are not assessed. The targets inherit pseudo-label errors, and classes that the pseudo labels do not recognize correctly cannot benefit from the loss. We did not study the sensitivity to K, τ , ϵ, and $\lambda ,$ did not ablate the individual loss components, and did not compare against reweighting the rendering losses by class frequency.

## V. CONCLUSION

We studied where a camera-based, weakly supervised Gaussian occupancy model places its Gaussians and found that rendering supervision alone concentrates them only weakly on foreground objects. To address this, we introduced a foreground attraction loss that back-projects foreground pseudo labels into 3D targets and pulls nearby Gaussians toward them, favoring those that already predict the target class, without changing the inference network. On Occ3DnuScenes, the loss increased the foreground concentration of Gaussians from 1.62 to 2.40 over three seeds. Over three seeds, the mean foreground IoU was higher than that of the baseline, whereas the differences in mIoU and occupancy IoU lie within the seed-to-seed variation of the baseline. In a single-seed test, the model also degraded less when fewer Gaussians were kept at inference. Single-run ablations showed that targets built from pseudo labels outperformed targets placed at ground-truth voxels and that acting on the positions alone appeared insufficient.

The improvements are modest relative to seed-to-seed variation, and our experiments use a reduced training set. Future work includes training on the full dataset, improving the pseudo labels for classes that are currently not recognized correctly, and adapting the attraction targets to where the rendering losses indicate errors.

## ACKNOWLEDGMENT

The authors used Claude (Anthropic), accessed through Claude Code, to assist in drafting the text of all sections, formatting the tables, generating the plotting code for Fig. 3, and checking bibliographic entries. The illustration in Fig. 1 was created by the authors with Gemini (Google), based on outputs of the authors’ own pipeline. All experiments and results were produced by the authors. The authors reviewed and revised all AI-assisted content and take full responsibility for the paper.

## REFERENCES

[1] X. Tian, T. Jiang, L. Yun, Y. Mao, H. Yang, Y. Wang, Y. Wang, and H. Zhao, “Occ3D: A large-scale 3D occupancy prediction benchmark for autonomous driving,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, 2023, pp. 64 318–64 330.

[2] S. Boeder, F. Gigengack, and B. Risse, “GaussianFlowOcc: Sparse and weakly supervised occupancy estimation using Gaussian splatting and temporal flow,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 24 943–24 954.

[3] M. Pan, J. Liu, R. Zhang, P. Huang, X. Li, H. Xie, B. Wang, L. Liu, and S. Zhang, “RenderOcc: Vision-centric 3D occupancy prediction with 2D rendering supervision,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024, pp. 12 404–12 411.

[4] Y. Huang, W. Zheng, B. Zhang, J. Zhou, and J. Lu, “SelfOcc: Selfsupervised vision-based 3D occupancy prediction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 19 946– 19 956.

[5] C. Zhang, J. Yan, Y. Wei, J. Li, L. Liu, Y. Tang, Y. Duan, and J. Lu, “OccNeRF: Advancing 3D occupancy prediction in LiDAR-free environments,” IEEE Trans. Image Process., vol. 34, pp. 3096–3107, 2025.

[6] H. Jiang, L. Liu, T. Cheng, X. Wang, T. Lin, Z. Su, W. Liu, and X. Wang, “GaussTR: Foundation model-aligned Gaussian transformer for self-supervised 3D spatial understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 11 960– 11 970.

[7] W. Gan, F. Liu, H. Xu, N. Mo, and N. Yokoya, “GaussianOcc: Fully self-supervised and efficient 3D occupancy estimation with Gaussian splatting,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 28 980–28 990.

[8] Y. Huang, W. Zheng, Y. Zhang, J. Zhou, and J. Lu, “GaussianFormer: Scene as Gaussians for vision-based 3D semantic occupancy prediction,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 376–393.

[9] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, Z. Zeng, H. Zhang, F. Li, J. Yang, H. Li, Q. Jiang, and L. Zhang, “Grounded SAM: Assembling open-world models for diverse visual tasks,” arXiv preprint arXiv:2401.14159, 2024.

[10] W. Yin, C. Zhang, H. Chen, Z. Cai, G. Yu, K. Wang, X. Chen, and C. Shen, “Metric3D: Towards zero-shot metric 3D prediction from a single image,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 9009–9019.

[11] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3D Gaussian¨ splatting for real-time radiance field rendering,” ACM Trans. Graph., vol. 42, no. 4, pp. 1–14, 2023.

[12] I. Kim, C. Kim, M. Bae, M. Joo, and H. J. Kim, “F4Splat: Feedforward predictive densification for feed-forward 3D Gaussian splatting,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2026, pp. 1–19.

[13] Y. Huang, A. Thammatadatrakoon, W. Zheng, Y. Zhang, D. Du, and J. Lu, “GaussianFormer-2: Probabilistic Gaussian superposition for efficient 3D occupancy prediction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 27 477–27 486.

[14] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuScenes: A multimodal dataset for autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 11 618–11 628.

[15] A.-Q. Cao and R. de Charette, “MonoScene: Monocular 3D semantic scene completion,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 3981–3991.

[16] Y. Huang, W. Zheng, Y. Zhang, J. Zhou, and J. Lu, “Tri-perspective view for vision-based 3D semantic occupancy prediction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 9223–9232.

[17] Y. Zhang, Z. Zhu, and D. Du, “OccFormer: Dual-path transformer for vision-based 3D semantic occupancy prediction,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 9399–9409.

[18] Y. Wei, L. Zhao, W. Zheng, Z. Zhu, J. Zhou, and J. Lu, “SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 21 672– 21 683.

[19] Z. Yu, C. Shu, J. Deng, K. Lu, Z. Liu, J. Yu, D. Yang, H. Li, and Y. Chen, “FlashOcc: Fast and memory-efficient occupancy prediction via channel-to-height plugin,” arXiv preprint arXiv:2311.12058, 2023.

[20] P. Tang, Z. Wang, G. Wang, J. Zheng, X. Ren, B. Feng, and C. Ma, “SparseOcc: Rethinking sparse latent representation for vision-based semantic occupancy prediction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 15 035–15 044.

[21] Y. Wu, W. Zheng, S. Zuo, Y. Huang, J. Zhou, and J. Lu, “EmbodiedOcc: Embodied 3D occupancy prediction for vision-based online scene understanding,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 26 360–26 370.

[22] L. Chambon, E. Zablocki, A. Boulch, M. Chen, and M. Cord, “GaussRender: Learning 3D occupancy with Gaussian rendering,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 27 010– 27 020.

[23] S. Rota Bulo, L. Porzi, and P. Kontschieder, “Revising densification in \` Gaussian splatting,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 347–362.

[24] D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann, “pixelSplat: 3D Gaussian splats from image pairs for scalable generalizable 3D reconstruction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 19 457–19 467.

[25] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.-J. Cham, and J. Cai, “MVSplat: Efficient 3D Gaussian splatting from sparse multi-view images,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 370–386.

[26] M. Seong, J. Kim, G. Bang, H. Jeong, and J. W. Choi, “MR-Occ: Efficient camera-LiDAR 3D semantic occupancy prediction using hierarchical multi-resolution voxel representation,” arXiv preprint arXiv:2412.20480, 2024.

[27] H. Fan, H. Su, and L. Guibas, “A point set generation network for 3D object reconstruction from a single image,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 2463–2471.

[28] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, J. Zhu, and L. Zhang, “Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 38–55.

[29] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Gir-´ shick, “Segment anything,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 3992–4003.