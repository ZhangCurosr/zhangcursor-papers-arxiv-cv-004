# TRACE: Topology-Aware Reconnection via Prediction-Adaptive Connectivity Enforcement for Thin-Structure Segmentation

Junho Moon   
Hanyang University   
Seoul, Republic of Korea   
jhmoon6807@hanyang.ac.kr

Ikbeom Jang<sup>\*</sup> Hankuk University of Foreign Studies Yongin, Republic of Korea ijang@hufs.ac.kr

## Abstract

Thin, elongated structures are particularly sensitive to connectivity errors, as even afew local segmentation mistakes can disrupt otherwise continuous paths. Existing skeletonbased objectives typically supervise fixed centerline locations and do not account for alternative local routes that may restore connectivity. We propose TRACE, a topologyaware, prediction-adaptive objective that uses a target skeleton to define required local connections. For each connection, TRACE identifies the easiest valid reconnection under the current prediction andjointly supervises its bottleneck with the corresponding skeleton point. The supervision thus adapts to the evolving prediction rather than remainingfixed to the centerline. We further introduce a topology-aware skeleton carrier that preserves both foreground and classspecific topology, enabling reliable definition ofthese connections. Extensive experiments across six 2D and 3D benchmarks show that TRACE improves connectivity and topology while maintaining strong segmentation accuracy with low training overhead. Code will be released upon acceptance.

## 1. Introduction

Segmentation of thin, elongated structures is particularly vulnerable to connectivity errors, as even a few misclassified pixels or voxels can break an otherwise continuous structure. This issue arises in retinal vessels [21], cerebral vessels [26, 27], hepatic vessels [2], pavement cracks [15], and road networks [7], where continuity is essential to structural integrity. Such local errors can fragment a single structure into multiple components while producing only minor changes in conventional region-overlap metrics [20]. Reliable thinstructure segmentation therefore requires preserving connectivity in addition to accurate region and boundary prediction.

Prior work preserves connectivity through centerline or skeleton supervision, global topological constraints, and local relational objectives. Centerline- and skeleton-based losses [1, 19, 20] provide direct structural supervision and may additionally incorporate boundary or width information. Target-only methods [13] avoid skeletonizing the prediction, reducing training-time computation and memory overhead. Their supervision, however, remains confined to fixed support derived from the ground truth. More broadly, skeletonbased support depends on the chosen skeletonization procedure [17], and even topology-preserving support may not identify the prediction error responsible for a connectivity break. Global objectives based on persistent homology or component graphs [3, 11, 16, 22] constrain connected components, loops, and cavities, yet provide limited guidance for repairing a specific local break. Neighbor-, affinity-, and path-based objectives [10, 24, 25, 28] provide localized supervision through individual neighbors, selected point pairs, or graph paths. These formulations organize connectivity around individual relations or pairwise paths, leaving unresolved how to jointly reconnect all same-class branches affected by a local break under the current prediction.

![](images/494b0ba665b3c70e9ff0b542029b60d099298234d558365218b6acc7f60b5424.jpg)  
Figure 1. Conventional target-skeleton supervision [13] relies on fixed skeleton locations and can alter the ground-truth topology (a). TRACE preserves it (b), selects the easiest valid reconnection among candidates (c, d), and supervises its bottleneck (d).

We propose TRACE, a topology-aware objective that extends skeleton supervision from a fixed target to guidance for adaptive reconnection. TRACE uses a topology-preserving skeleton to identify neighboring same-class branches that should remain connected. Based on the current prediction, it identifies the easiest valid local reconnection and jointly supervises the skeleton point with the bottleneck limiting this reconnection. To support this formulation, we further introduce a Topology-Aware Skeleton Carrier (TASC), which preserves both foreground and class-specific topology while defining local connectivity requirements. We identify the reconnection bottleneck through an exact local search using centered logit-margin scores derived directly from standard semantic logits. The proposed objective avoids costly prediction-side skeletonization and requires no architectural modifications, enabling efficient plug-and-play integration into existing segmentation pipelines.

Our contributions are as follows:

• We introduce TRACE, a prediction-adaptive connectivity objective that uses a skeleton to define required connections and supervises the easiest reconnection bottleneck.

• We present TASC, a class-labeled skeleton that preserves both foreground and class-specific topology while identifying same-class branches that should remain connected.

• Across six 2D and 3D thin-structure benchmarks, TRACE improves connectivity and topology while maintaining strong region-overlap and boundary accuracy.

• We further demonstrate that TRACE is computationally efficient, generalizes across segmentation backbones, and complements existing topology-aware methods.

## 2. Related Work

Centerline and skeleton supervision. Centerline- and skeleton-based objectives promote connectivity by supervising thin structural representations derived from the prediction, target, or both. clDice [20] measures reciprocal overlap between the prediction and target using differentiable soft skeletons. Menten et al. [17] developed topology-preserving differentiable skeletonization for gradient-based losses. clCE [1] applies cross-entropy at centerline locations to improve topology while preserving region accuracy. cbDice [19] incorporates boundary and radius cues to preserve geometry across varying widths. Skeleton Recall Loss (SkelRecall) [13] uses a dilated, class-labeled target skeleton and avoids prediction-side skeletonization. These methods, however, tie structural supervision to centerline or skeleton support and do not explicitly identify the prediction error that blocks a valid reconnection. TRACE addresses this gap with a topology-preserving carrier that defines the local same-class branches to be connected and extends pointwise supervision to the bottleneck of their easiest valid reconnection.

Global topology supervision. Global topology objectives constrain connected components, loops, and cavities across the segmentation. TopoLoss [11] uses persistent homology to identify predictions associated with topological errors. Betti Matching [22] matches topological features between the prediction and target while accounting for their spatial correspondence. Subsequent extensions support multiclass segmentation [3]. Topograph [16] represents prediction– target component relations as a graph to provide efficient binary and multiclass constraints. Whereas these methods constrain global or component-level topology, TRACE targets a required local reconnection and directly supervises its prediction-dependent bottleneck.

Pairwise and path supervision. Pairwise- and path-based objectives supervise connectivity through neighboring elements, point pairs, or graph paths. MALIS [24] assigns pairwise connectivity errors to the weakest edge along a maximin path in an affinity graph. CAPE [10] penalizes shortest-path costs between selected target-graph nodes on a predicted distance map. SCNP [25] adjusts semantic logits according to the weakest same-class neighbor. WPRF [28] optimizes widest-path reachability between sampled point pairs on a target-derived support graph using a separate affinity prediction. In contrast, TRACE defines each local reconnection on a class-labeled carrier that preserves the topology of both the foreground union and each semantic class. Removing a carrier point exposes all neighboring same-class branches, which we jointly reconnect via an exact search over at most 9 candidate points in 2D or 27 in 3D, using a centered logit margin computed directly from standard multiclass semantic logits.

## 3. Method

TRACE turns target-skeleton supervision from fixed-point enforcement into prediction-adaptive local reconnection by identifying and supervising the bottleneck of the easiest valid reconnection under the current prediction. We first formulate local reconnection as a connectivity bottleneck problem that requires a connected set of target-class points to reach all neighboring branches (Sec. 3.1). We then construct TASC, a thin class-labeled carrier that encodes these connectivity requirements while preserving both foreground and classspecific topology (Sec. 3.2). Next, we define a centered logit-margin score that ranks candidate points by their rivalto-target margin relative to the carrier average (Sec. 3.3). Finally, we combine the pointwise and connectivity penalties into a structural objective (Sec. 3.4).

## 3.1. Local Connectivity Bottleneck Supervision

We first formalize the local reconnection event supervised by TRACE, as illustrated in Figure 2. Throughout, a point denotes a pixel or voxel, and all quantities are defined on a single output grid. Let $d \in \{ 2 , 3 \}$ , let $\Omega \subset \mathbb { Z } ^ { d }$ be the grid, and let $C \geq 2$ be the number of semantic classes, including background. The target labeling is $Y : \Omega  \{ 0 , \ldots , C - 1 \}$

Step 1) Remove carrier point  
![](images/e742242b8de8ff7184887410868dba70ce1f055afba37666eeb238e082eb62cb.jpg)

Step 2) Define valid reconnection candidates  
![](images/4adf3d8de8a99c926a5d1e940f088b850a838b20c069105def681ebfa9ee43e0.jpg)

Step 3) Bottleneck Selection and Supervision  
![](images/b03aef75178372db286331aea4e65ad9b03b027f7ebd235a6465460aa8b4c908.jpg)  
Figure 2. Prediction-adaptive local reconnection in TRACE. Step 1) Removing a carrier point exposes neighboring same-class branches that should remain connected. Step 2) Valid local reconnections reaching all exposed branches are considered. Step 3) Candidates are activated in ascending order of centered logit-margin score until a valid reconnection forms; the point activated at this threshold is the supervised bottleneck. The carrier is TASC (Sec. 3.2), numbers indicate $g _ { c } ( x ) \ ( \mathrm { S e c . } \ 3 . 3 )$ , and τ denotes the activation threshold.

We represent the target connectivity to be preserved using a class-labeled carrier $S : \Omega \to \{ 0 , \ldots , C - 1 \}$ derived from $Y ;$ its construction is described in Sec. 3.2. The carrier is a thin representation that preserves both foreground and class-specific topology, and its local branches define the connectivity requirements that TRACE supervises. For each foreground class c, define

$$
K _ { c } = \{ x \in \Omega \mid S ( x ) = c \} , \quad K = \bigcup _ { c = 1 } ^ { C - 1 } K _ { c } ,\tag{1}
$$

where $K _ { c }$ denotes the class-c carrier and K is the union of all foreground carriers. Removing a carrier point $v \in K _ { c }$ reveals the neighboring same-class branches that should remain connected. For each $x \in \Omega$ with $Y ( x ) = c , \operatorname { l e t } g _ { c } ( x )$ denote the centered logit-margin score defined in Sec. 3.3, which ranks each target point by its rival-to-target margin relative to the carrier average.

Let $\sim _ { d }$ denote 8-connectivity for $\ d \ = \ 2$ and 26- connectivity for $d = 3$ . For a carrier point $v \in K _ { c }$ , define its local $3 ^ { d }$ neighborhood as

$$
\mathcal { N } ( v ) = \{ x \in \Omega \mid \| x - v \| _ { \infty } \leq 1 \} .\tag{2}
$$

Let $P _ { v 1 } , \ldots , P _ { v m _ { i } }$ be the connected components of $( K _ { c } \cap$ $\mathcal { N } ( v ) ) \backslash \{ v \}$ under $\sim _ { d } ;$ these are the neighboring same-class branches. When $m _ { v } \ \geq \ 2 .$ these branches define a local reconnection event: the target requires a connected set of class-c points within $\mathcal { N } ( v )$ that reaches every branch. Carrier points with $m _ { v } < 2$ do not define a reconnection event.

Define the local class-c target support $T _ { v } = \{ x \in \mathcal { N } ( v ) \mid $ $Y ( x ) = c \}$ and the family of valid local reconnections

$$
\begin{array} { r l } & { { \mathcal R } _ { v } = \{ R \subseteq T _ { v } ~ \vert ~ R \mathrm { ~ i s ~ c o n n e c t e d ~ u n d e r ~ } \sim _ { d } , } \\ & { ~ R \cap { \cal P } _ { v k } \not = \emptyset , ~ \forall k \in \{ 1 , \dots , m _ { v } \} \} . } \end{array}\tag{3}
$$

The local connectivity bottleneck is then

$$
r _ { v } = \operatorname* { m i n } _ { R \in \mathcal { R } _ { v } } \operatorname* { m a x } _ { x \in R } g _ { c } ( x ) .\tag{4}
$$

For $m _ { v } \ = \ 2 ,$ , each $R \in \mathcal { R } _ { v }$ contains a same-class path connecting the two branches; at a junction, one connected reconnection reaches all branches simultaneously. Thus, $r _ { v }$ is the lowest achievable bottleneck score among valid local reconnections. A large $r _ { v }$ means that every valid reconnection contains a point with a large rival-to-target margin relative to the carrier average.

Eq. (4) has an equivalent threshold-connectivity form that enables exact bottleneck selection. For $\tau \geq 0$ , let $T _ { v } ( \tau ) =$ $\{ x \in T _ { v } \mid g _ { c } ( x ) \leq \tau \}$ and let $\mathcal { C } _ { v } ( \tau )$ denote the connected components of $T _ { v } ( \tau )$ under $\sim d \cdot$ Then

$$
\begin{array} { r l } & { r _ { v } = \operatorname* { m i n } \big \{ \tau \geq 0 : \exists Q \in { \mathcal C } _ { v } ( \tau ) , } \\ & { \qquad Q \cap P _ { v k } \neq \emptyset , \quad \forall k \in \{ 1 , \dots , m _ { v } \} \big \} . } \end{array}\tag{5}
$$

This characterization in Eq. (5) yields an exact finite solver. We activate at most $3 ^ { d }$ points in $T _ { v }$ in nondecreasing order of $g _ { c }$ , with ties resolved by a fixed local order, and merge adjacent active components until a component reaches every branch. $\operatorname { I f } x _ { v } ^ { \star }$ is the last activated point, then $r _ { v } = g _ { c } ( x _ { v } ^ { \star } )$ We perform the combinatorial search using detached scores to select $\boldsymbol { x } _ { v } ^ { \star }$ , while propagating gradients through the corresponding non-detached score $g _ { c } ( x _ { v } ^ { \star } )$ . Because the carrier means in Eq. (7) remain differentiable, this update includes a direct contribution at $x _ { v } ^ { \star }$ and the associated mean-centering correction over K.

## 3.2. Topology-Aware Skeleton Carrier

TASC constructs a thin, class-labeled carrier whose local branches encode same-class connectivity requirements while preserving the digital topology of both the foreground union and each semantic class. Starting from $S = Y$ , consider a foreground candidate point v in the current carrier, with class $c = S ( v )$ , and define

Algorithm 1 Topology-aware skeleton carrier construction   
Require: Target Y, thinning order ≺, predicates Simple and Ist   
1: $\mathbf { \bar { \boldsymbol { S } } } \gets \mathbf { \boldsymbol { Y } } ; \mathbf { \bar { \mathcal { A } } } \gets \boldsymbol { \emptyset }$   
2: repeat   
3: deleted ← false   
4: for all foreground points v $\not \in A$ in order ≺ do   
5: c ← S(v)   
6: $U  \{ x : S ( x ) > 0 \} ; U _ { c }  \{ x : S ( x ) = c \}$   
7: if Ist<sub>1</sub>(v; U) ∨ Ist<sub>1</sub>(v; U<sub>c</sub>) then   
8: A ← A ∪ {v}   
9: else if Simple(v; U) ∧ Simple(v; U<sub>c</sub>) then   
10: $S ( v )  0$   
11: deleted ← true   
12: end if   
13: end for   
14: until deleted = false   
15: return S

$$
\begin{array} { c } { { U ( S ) = \{ x \in \Omega \mid S ( x ) > 0 \} , } } \\ { { U _ { c } ( S ) = \{ x \in \Omega \mid S ( x ) = c \} . } } \end{array}\tag{6}
$$

We use two standard digital-topology predicates: simplepoint deletion for topology-preserving thinning [5] and 1- isthmus detection for retaining curve-like structures exposed during thinning [4, 6].

TASC preserves topology by protecting 1-isthmuses and deleting a candidate only when it is simple in both the foreground union and its class-specific support. Let Simple(v; A) indicate that deleting v is simple in the binary set A, and let Ist $. ( v ; A )$ indicate that v is a 1-isthmus of A. A candidate is permanently protected if it is a 1-isthmus in either $U ( S )$ or $U _ { c } ( S )$ ; otherwise, delete it only if it is simple in both. Each accepted deletion therefore preserves the topology of the foreground union and class c, while leaving all other class supports unchanged. Because the predicates are evaluated on the current carrier, deletions may expose additional 1-isthmuses, which we protect when we encounter them later.

We apply these predicates using the $8 / 4$ foreground/background adjacency pair in 2D and 26/6 in 3D. In 2D, the 1-isthmus predicate holds when the proper neighborhood contains exactly two 8-connected foreground components; in 3D, we use the audited DGtal 26/6 predicate [8]. Algorithm 1 summarizes the construction. The returned carrier preserves the labels of all retained points and satisfies $S ( x ) \in \{ 0 , Y ( x ) \}$ for every $x \in \Omega$ . For architectures with deep supervision [12], we construct TASC directly from the target at each output resolution to avoid resampling-induced topological changes.

## 3.3. Centered Logit Margin for Local Connectivity

The local reconnection search ranks candidate target points using a centered logit-margin score $g _ { c } ( x )$ . Let $z _ { j } ( x )$ denote the logit for class $j \in \{ 0 , \ldots , C - 1 \}$ at x. We center each logit channel by its mean over the union carrier K, so that the score expresses local rival-to-target margins relative to the carrier. For each class channel $j$ and each foreground target point x with $Y ( x ) = c , $ define

$$
\begin{array} { l } { { \displaystyle \bar { z } _ { j } = \frac { 1 } { | K | } \sum _ { u \in K } z _ { j } ( u ) , \qquad \widetilde { z } _ { j } ( x ) = z _ { j } ( x ) - \bar { z } _ { j } , } } \\ { { \displaystyle g _ { c } ( x ) = \left[ \operatorname* { m a x } _ { \ell \neq c } \widetilde { z } _ { \ell } ( x ) - \widetilde { z } _ { c } ( x ) \right] _ { + } . } } \end{array}\tag{7}
$$

Here, $[ a ] _ { + } = \operatorname* { m a x } \{ a , 0 \}$ . Equivalently, $g _ { c } ( x )$ is the largest positive difference between a local rival-to-target margin and the corresponding carrier-average margin. This score ranks reconnection candidates, while $\mathcal { L } _ { \mathrm { s e g } }$ supervises the overall class prediction. In binary segmentation, the only rival is background; in multiclass segmentation, the maximization also considers other foreground classes.

## 3.4. Pointwise-Connectivity Penalty Integration

We combine the pointwise centered logit-margin score and local connectivity bottleneck into a bounded structural penalty at each carrier point. For $a > 0 ,$ define

$$
b ( a ) = 2 \arcsin \left( \operatorname { t a n h } { \frac { a } { 2 } } \right) , \quad \phi ( a ) = { \frac { { \sqrt { 2 } } a b ( a ) } { \sqrt { a ^ { 2 } + b ( a ) ^ { 2 } } } } .\tag{8}
$$

We define $\phi ( 0 ) = 0$ by continuity. The mapping is increasing, has unit slope at the origin, and approaches the finite limit ${ \sqrt { 2 } } \pi$ . It therefore preserves sensitivity to small scores while limiting the influence of large ones.

For each carrier point $v \in K _ { c } ,$ let $p _ { v } = \phi ( g _ { c } ( v ) )$ . When $m _ { v } \ge 2$ , additionally let $t _ { v } = \phi ( r _ { v } )$ . We define the structural penalty at v as

$$
q _ { v } = \left\{ \sqrt { \frac { p _ { v } ^ { 2 } + t _ { v } ^ { 2 } } { 2 } } , \quad m _ { v } \geq 2 , \right.\tag{9}
$$

For $m _ { v } \geq 2 , q _ { v }$ is the root mean square of the pointwise and connectivity penalties, so $q _ { v } = p _ { v } = t _ { v }$ whenever $p _ { v } = t _ { v } .$ In particular, if $x _ { v } ^ { \star } = v$ , then $r _ { v } = g _ { c } ( v )$ , and the same local score is not double-counted.

To prevent classes with larger carriers from receiving greater weight, we average the penalties within each nonempty class carrier and normalize over the $C - 1$ foreground classes:

$$
\mathcal { L } _ { \mathrm { T R A C E } } = \frac { 1 } { C - 1 } \sum _ { \stackrel { c = 1 } { K _ { c } \neq \emptyset } } ^ { C - 1 } \frac { 1 } { | K _ { c } | } \sum _ { v \in K _ { c } } q _ { v } .\tag{10}
$$

The overall training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { s e g } } + \lambda \mathcal { L } _ { \mathrm { T R A C E } } , } \end{array}\tag{11}
$$

where $\mathcal { L } _ { \mathrm { s e g } }$ denotes a standard segmentation loss, such as Dice loss, cross-entropy, or their combination, and λ controls the relative weight of $\mathcal { L } _ { \mathrm { T R A C E } }$

## 4. Experiments

## 4.1. Datasets and Evaluation Metrics

Datasets. We evaluate TRACE on six thin-structure segmentation benchmarks spanning 2D and 3D data, including binary and multiclass tasks. DRIVE [21] is used for retinal vessel segmentation and contains 40 RGB fundus images, split into 20/10/10 for training/validation/testing. DeepCrack [15] contains 537 RGB images for surface-crack segmentation (270/30/237), while DeepGlobe Roads [7] contains 6,226 RGB satellite images for road segmentation (4,980/622/624). For DSCA [27], we convert 224 cerebral digital subtraction angiography sequences, annotated for the main arterial trunk and bifurcated vessels, into 2D minimum-intensity projections and use a 144/36/44 split. MSD Hepatic Vessel [2] contains 303 portal-venous-phase 3D CT scans with hepatic vessel and tumor annotations; we treat tumor voxels as background for binary vessel segmentation and use a 243/30/30 split. TopCoW [26] contains 125 3D CT angiography scans with multiclass annotations for 13 Circle of Willis vessel segments, split into 100/12/13 cases.

Metrics. We evaluate segmentation quality using complementary metrics for connectivity, topology, boundary accuracy, and region overlap. We first report clDice [20] to measure centerline consistency and connectivity preservation. We also report the absolute Betti number error, $\beta = | \Delta \beta _ { 0 } | + | \Delta \beta _ { 1 } |$ [22], to quantify topological errors associated with structural connectivity. The 95th-percentile Hausdorff distance (HD95) [23] assesses boundary accuracy by measuring spatial disagreement while limiting outlier sensitivity. Finally, Dice [9] measures region overlap and assesses whether improved structural consistency is achieved without sacrificing overall segmentation accuracy. We compute each metric independently for each foreground class in every test sample. For multiclass datasets, we average the class-wise values to obtain one score per metric per sample, then average these per-sample scores over the entire test set.

## 4.2. Main Comparison Setup

Baselines. Our primary focus is connectivity preservation; therefore, we conduct extensive comparisons with representative skeleton-based losses, including clDice [20], clDice with topology-preserving skeletonization (Topo-clDice) [17], clCE [1], cbDice [19], and SkelRecall [13]. We also include the recent multiclass extension of Betti Matching [3] as a consistent global-topology reference across binary and multiclass benchmarks, using the projection-based protocol of [14] for 3D volumes. To isolate the effects of the training objective, the main benchmark evaluates representative methods using standard semantic outputs while keeping the prediction head and inference pipeline fixed. We exclude methods that require alternative output formulations, auxiliary or correction modules, or post-processing from this controlled comparison.

Training protocol. We use a common nnU-Net [12] training setup for all methods to ensure a controlled comparison. Following Eq. (11), we use Dice loss combined with crossentropy loss (Dice + CE), the default nnU-Net objective, as $\mathcal { L } _ { \mathrm { s e g } }$ , and replace $\mathcal { L } _ { \mathrm { T R A C E } }$ with each compared structural loss while keeping all other training settings fixed. For TRACE and the skeleton-based losses, we select λ from {0.10, 0.30, 0.60, 1.00} based on the validation Dice score. Betti Matching is tuned separately because excessive topological weighting can degrade segmentation performance [22]; its weight is selected on the validation set from {0.00001, 0.0001, 0.001, 0.01}.

## 4.3. Topology Preservation and Training Efficiency

Topology preservation. We assess whether each method’s target-side skeleton representation preserves topology across deep-supervision scales. We use clDice [20] as a representative differentiable soft-skeletonization method, Topo-clDice [17] as its topology-preserving counterpart, and SkelRecall [13] as a target skeleton-based method. At each active scale, we construct the skeleton from the corresponding training target using each method and record a topology violation if the Betti numbers change for the foreground union or any present semantic class. We report the percentage of target–scale pairs exhibiting at least one violation.

Training efficiency. We evaluate the training overhead of TRACE against the same representative methods under an identical nnU-Net setup, with Dice + CE as the reference. We report training time per epoch and peak process GPU memory on an NVIDIA H200 (140 GB); runs exceeding the available GPU memory are reported as out of memory (OOM). Topology preservation can still be evaluated in OOM cases, whereas training efficiency cannot be assessed without a completed epoch.

## 4.4. Backbone Transfer and Method Compatibility

We evaluate TRACE’s plug-and-play capability in two settings: transfer to a different segmentation backbone and integration with another topology-aware method.

Transfer to $\mathbf { C S ^ { 2 } - N e t }$ [18]. We integrate TRACE with $\mathrm { C S ^ { 2 } } .$ Net to assess its generalization performance across segmentation architectures. We compare TRACE against the standard training objective and representative structural losses on DeepCrack and DRIVE.

Integration with SCNP [25]. We further investigate whether TRACE provides complementary connectivity supervision when combined with an existing topology-aware approach. Specifically, we compare Dice + CE applied to the SCNP-transformed logits with the same objective augmented by TRACE. We conduct experiments on DRIVE and DSCA using nnU-Net.

Table 1. Comparison of training objectives on four 2D thin-structure datasets spanning binary and multiclass tasks under the same nnU-Net setup. Results are mean ± standard deviation over test samples. Bold and underlined values indicate the best and second-best results.
<table><tr><td rowspan="2">Method</td><td colspan="4">DRIVE [21]</td><td colspan="4">DSCA [27]</td></tr><tr><td>clDice↑</td><td> $\beta \downarrow$ </td><td>HD95↓</td><td>Dice↑</td><td>clDice↑</td><td> $\beta \downarrow$ </td><td>HD95↓</td><td>Dice↑</td></tr><tr><td>Dice + CE [12]</td><td> $8 1 . 4 3 { \pm } 3 . 5 6 $ </td><td> $1 1 5 . 9 0 { \scriptstyle \pm 1 8 . 5 1 }$ </td><td> $4 . 0 0 \pm 2 . 0 5$ </td><td> $8 2 . 0 4 \pm 1 . 4 0 $ </td><td> $8 7 . 0 3 \pm 1 4 . 9 5$ </td><td> $6 2 . 0 7 { \scriptstyle \pm 5 2 . 2 4 }$ </td><td> $4 9 . 5 0 { \scriptstyle \pm 1 6 1 . 2 4 }$ </td><td> $8 6 . 6 2 \pm 1 5 . 7 4$ </td></tr><tr><td>+ clDice [20]</td><td> $8 1 . 9 7 { \scriptstyle \pm 3 . 6 9 }$ </td><td> $9 7 . 8 0 { \pm } 2 7 . 5 2 $ </td><td> $4 . 0 0 \pm 2 . 1 9$ </td><td> $8 2 . 2 2 { \pm } 1 . 3 7$ </td><td> $8 7 . 7 1 \pm 1 2 . 6 9$ </td><td> ${ \pm 3 . 8 1 \pm 4 7 . 2 7 }$ </td><td> $2 9 . 0 1 \pm 1 1 1 . 5 2$ </td><td> $8 7 . 2 5 { \pm } 1 3 . 5 6 $ </td></tr><tr><td> $+ \mathrm { T o p o - c l D i c e } \ [ 1 7 ]$ </td><td> $8 1 . 9 9 \pm 3 . 6 0$ </td><td> $8 2 . 5 0 { \pm } 1 8 . 2 6 $ </td><td> $3 . 8 2 \pm 2 . 2 2$ </td><td> $\overline { { 8 2 . 1 0 \pm 1 . 4 1 } }$ </td><td></td><td> $- \cot$ </td><td>memory</td><td></td></tr><tr><td>+ clCE [1]</td><td> $\overline { { 8 1 . 8 8 \pm 3 . 4 5 } }$ </td><td> $1 0 6 . 9 0 { \pm } 1 8 . 8 1$ </td><td> $\overline { { 4 . 0 9 \pm 2 . 3 3 } }$ </td><td> $8 2 . 1 8 { \pm } 1 . 3 1 $ </td><td> $8 7 . 5 3 \pm 1 2 . 5 1$ </td><td> $6 5 . 9 1 \pm 5 5 . 2 5$ </td><td> $4 1 . 7 7 { \scriptstyle \pm 1 5 3 . 7 0 }$ </td><td> $8 7 . 3 3 \pm 1 3 . 3 6$ </td></tr><tr><td>+ cbDice [19]</td><td> $8 1 . 6 5 { \scriptstyle \pm 3 . 6 2 }$ </td><td> $1 0 9 . 7 0 { \scriptstyle \pm 2 4 . 7 0 }$ </td><td> $4 . 0 6 \pm 2 . 2 2$ </td><td> $8 2 . 1 0 { \pm } 1 . 3 7$ </td><td> $8 7 . 5 2 \pm 1 3 . 2 8$ </td><td> $6 1 . 8 6 { \pm } 5 4 . 6 8$ </td><td> $4 1 . 6 2 \pm 1 5 3 . 5 9$ </td><td> $8 7 . 0 9 \pm 1 4 . 0 0$ </td></tr><tr><td>+ SkelRecall [13]</td><td> $8 1 . 8 9 { \pm } 3 . 5 0 $ </td><td> $9 7 . 0 0 { \scriptstyle \pm 2 2 . 7 8 }$ </td><td> $4 . 0 0 \pm 2 . 3 5$ </td><td> $8 2 . 1 8 { \pm } 1 . 2 7$ </td><td> $8 7 . 8 8 \pm 1 2 . 8 1$ </td><td> $5 6 . 4 2 \pm 4 7 . 6 1$ </td><td> $4 1 . 7 5 { \scriptstyle \pm 1 5 3 . 6 9 }$ </td><td> $8 7 . 4 7 \pm 1 3 . 5 8$ </td></tr><tr><td>+ Betti Matching [3]</td><td> $8 1 . 6 0 { \pm } 3 . 5 9$ </td><td> $1 0 3 . 7 0 { \scriptstyle \pm 1 9 . 3 7 }$ </td><td> $4 . 3 0 { \pm } 2 . 3 0 $ </td><td> $8 2 . 1 1 \pm 1 . 3 0$ </td><td> $\overline { { 8 7 . 4 2 \pm 1 3 . 8 7 } }$ </td><td> $5 6 . 6 3 { \scriptstyle \pm 5 0 . 9 2 }$ </td><td> $4 4 . 8 0 \pm 1 4 9 . 3 6$ </td><td> $\overline { { 8 7 . 1 2 \pm 1 4 . 2 1 } }$ </td></tr><tr><td>+ TRACE (ours)</td><td> $\mathbf { 8 2 . 6 5 } \pm 3 . 5 \mathbf { 0 }$ </td><td> $\mathbf { 7 9 . 3 0 { \pm } 2 0 . 5 1 }$ </td><td> $3 . 4 5 { \scriptstyle \pm 1 . 7 7 }$ </td><td> $\mathbf { 8 2 . 3 8 { \pm 1 . 3 2 } }$ </td><td> $\mathbf { 8 9 . 1 8 \pm 1 0 . 5 6 }$ </td><td> $5 5 . 4 9 { \scriptstyle \pm 5 0 . 4 5 }$ </td><td> $\pm 5 . 2 9 \pm 1 1 1 . 2 0$ </td><td> $\mathbf { 8 8 . 7 8 { \pm } 1 1 . 0 6 }$ </td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="4">DeepCrack [15]</td><td colspan="4">DeepGlobe Roads [7]</td></tr><tr><td> $\mathrm { c l D i c e \uparrow }$ </td><td> $\beta \downarrow$ </td><td>HD95↓</td><td>Dice↑</td><td>clDice↑</td><td> $\beta \downarrow$ </td><td>HD95↓</td><td>Dice↑</td></tr><tr><td>Dice + CE [12]</td><td> $8 9 . 4 8 { \pm } 9 . 5 8 $ </td><td> $5 . 2 1 \pm 5 . 6 4$ </td><td> $1 1 . 4 8 \pm 1 6 . 8 0$ </td><td> $8 2 . 4 9 { \pm } 1 1 . 2 0 $ </td><td> $8 1 . 0 9 { \pm } 1 4 . 4 0 $ </td><td> $1 1 . 7 6 \pm 1 2 . 9 2$ </td><td> $1 3 9 . 6 3 { \pm } 1 8 7 . 5 0 $ </td><td> $7 4 . 2 4 \pm 1 2 . 9 4$ </td></tr><tr><td>+ clDice [20]</td><td> $8 9 . 7 2 { \scriptstyle \pm 8 . 9 2 }$ </td><td> $5 . 5 7 \pm 5 . 9 6$ </td><td> $1 1 . 0 2 { \scriptstyle \pm 1 6 . 7 1 }$ </td><td> $8 2 . 8 5 { \scriptstyle \pm 1 0 . 0 9 }$ </td><td> $\mathbf { 8 2 . 3 2 \pm 1 4 . 2 6 }$ </td><td> $1 0 . 2 7 { \scriptstyle \pm 1 2 . 7 0 }$ </td><td> $1 2 7 . 5 5 { \scriptstyle \pm 1 8 7 . 1 2 }$ </td><td> $7 4 . 6 9 { \pm } 1 3 . 1 9$ </td></tr><tr><td> $+ \mathrm { T o p o - c l D i c e } \ [ 1 7 ]$ </td><td></td><td> $- \cot$ </td><td> $\mathrm { { \ m e m o r y } ~ - }$ </td><td></td><td> $8 1 . 9 6 \pm 1 4 . 8 3$ </td><td> $9 . 6 8 \pm 1 7 . 3 6$ </td><td> $1 2 3 . 4 0 { \scriptstyle \pm 1 9 7 . 7 2 }$ </td><td> $7 4 . 7 8 \pm 1 4 . 4 4$ </td></tr><tr><td>+ clCE [1]</td><td> $8 9 . 9 6 { \pm } 8 . 5 9$ </td><td> $5 . 6 5 { \pm } 6 . 3 5$ </td><td> $1 1 . 3 8 { \pm } 2 3 . 9 2$ </td><td> $8 3 . 4 8 { \pm } 9 . 7 8$ </td><td> $8 1 . 8 8 \pm 1 4 . 3 9$ </td><td> $1 1 . 7 4 \pm 1 4 . 4 1$ </td><td> $1 2 9 . 7 5 { \scriptstyle \pm 1 8 4 . 4 6 }$ </td><td> $\overline { { 7 4 . 6 7 \pm 1 3 . 0 0 } }$ </td></tr><tr><td>+ cbDice [19]</td><td> $8 9 . 6 7 { \scriptstyle \pm 9 . 1 7 }$ </td><td> $6 . 1 8 \pm 7 . 4 0$ </td><td> $1 3 . 0 6 \pm 2 4 . 9 4$ </td><td> $8 2 . 8 0 { \scriptstyle \pm 1 0 . 7 3 }$ </td><td> $8 1 . 2 3 { \pm } 1 4 . 9 5$ </td><td> $1 1 . 3 3 \pm 1 4 . 2 6$ </td><td> $1 2 8 . 2 7 \pm 1 8 1 . 1 9$ </td><td> $7 3 . 8 0 \pm 1 3 . 9 6$ </td></tr><tr><td>+ SkelRecall [13]</td><td> $9 0 . 4 2 { \pm } 8 . 5 9$ </td><td> $5 . 1 0 { \pm } 5 . 5 3 $ </td><td> $1 2 . 0 4 \pm 2 4 . 9 5$ </td><td> $8 3 . 8 9 \pm 1 0 . 2 1$ </td><td> $8 1 . 4 0 \pm 1 4 . 8 0$ </td><td> $1 1 . 5 1 { \pm } 1 3 . 1 9$ </td><td> $1 3 7 . 4 8 { \pm } 1 9 5 . 1 3$ </td><td> $7 4 . 3 7 { \pm } 1 2 . 7 3$ </td></tr><tr><td>+ Betti Matching [3]</td><td> $9 0 . 4 4 { \scriptstyle \pm 8 . 5 6 }$ </td><td> $6 . 1 1 \pm 7 . 2 4$ </td><td> $1 1 . 6 2 { \pm } 2 2 . 9 4 $ </td><td> $8 4 . 2 8 { \pm } 1 0 . 2 2$ </td><td> $8 1 . 6 9 { \pm } 1 4 . 1 0$ </td><td> ${ \bf 9 . 1 0 \pm 1 1 . 4 8 }$ </td><td> $1 3 7 . 6 6 { \scriptstyle \pm 1 8 9 . 8 1 }$ </td><td> $7 4 . 3 0 { \scriptstyle \pm 1 2 . 6 2 }$ </td></tr><tr><td>+ TRACE (ours)</td><td> $\overline { { 9 0 . 6 0 \pm 8 . 4 6 } }$ </td><td> $\mathbf { 4 . 9 5 \pm 6 . 1 9 }$ </td><td> $\mathbf { 1 1 . 0 2 \pm 1 7 . 6 0 }$ </td><td> $\overline { { 8 4 . 5 3 \pm 9 . 7 3 } }$ </td><td> $8 2 . 0 3 \pm 1 4 . 9 4$ </td><td> $9 . 1 3 \pm 1 2 . 2 2$ </td><td> $\mathbf { 1 1 7 . 0 6 { \scriptstyle \pm 1 6 3 . 5 5 } }$ </td><td> ${ \bf 7 4 . 8 8 \pm 1 3 . 2 2 }$ </td></tr></table>

Table 2. Comparison of training objectives on two 3D vessel datasets spanning binary and multiclass tasks under the same nnU-Net setup. Results are mean ± standard deviation over test samples. Bold and underlined values indicate the best and second-best results.
<table><tr><td rowspan="2">Method</td><td colspan="4">MSD Hepatic Vessel [2]</td><td colspan="4">TopCoW [26]</td></tr><tr><td> $\mathrm { c l D i c e \uparrow }$ </td><td> $\beta \downarrow$ </td><td>HD95↓</td><td>Dice↑</td><td> $\mathrm { c l D i c e \uparrow }$ </td><td> $\beta \downarrow$ </td><td>HD95↓</td><td>Dice↑</td></tr><tr><td> $\mathrm { D i c e } + \mathbf { C E } \left[ 1 2 \right]$ </td><td> $6 7 . 0 0 \pm 1 6 . 6 1 $ </td><td> $4 0 . 9 0 { \scriptstyle \pm 2 4 . 4 4 }$ </td><td> $3 9 . 4 1 \pm 3 4 . 7 9$ </td><td> $5 8 . 9 0 \pm 1 5 . 6 6$ </td><td> $8 6 . 3 2 { \scriptstyle \pm 9 . 3 9 }$ </td><td> $0 . 3 2 { \pm } 0 . 2 5$ </td><td> $8 . 3 1 \pm 6 . 0 0$ </td><td> $8 0 . 5 7 { \pm } 7 . 1 2 $ </td></tr><tr><td>+ clDice [20]</td><td> $6 8 . 7 3 \pm 1 6 . 8 7$ </td><td> $3 8 . 5 7 \pm 2 2 . 2 6 $ </td><td> $2 8 . 0 9 { \scriptstyle \pm 2 5 . 1 5 }$ </td><td> $5 9 . 7 0 { \scriptstyle \pm 1 6 . 1 1 }$ </td><td> $8 5 . 4 9 { \scriptstyle \pm 6 . 9 6 }$ </td><td> $0 . 3 6 { \pm } 0 . 2 6$ </td><td> $7 . 9 7 { \pm } 5 . 6 1 $ </td><td> $8 1 . 2 0 { \pm } 5 . 4 5$ </td></tr><tr><td>+ Topo-clDice [17]</td><td> $6 8 . 9 2 \pm 1 6 . 8 2 $ </td><td> $3 5 . 6 7 \pm 2 0 . 5 6$ </td><td> $2 3 . 4 0 { \scriptstyle \pm 2 7 . 9 8 }$ </td><td> $6 2 . 8 3 \pm 1 6 . 2 8$ </td><td></td><td>Out of memory —</td><td></td><td></td></tr><tr><td>+ clCE [1]</td><td> $6 8 . 3 8 \pm 1 6 . 5 7$ </td><td> $3 2 . 1 3 { \pm } 2 0 . 3 2$ </td><td> $2 8 . 8 4 \pm 2 8 . 2 0$ </td><td> $6 0 . 2 5 { \scriptstyle \pm 1 5 . 5 1 }$ </td><td> $8 6 . 2 7 { \scriptstyle \pm 7 . 6 4 }$ </td><td> $0 . 2 9 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $7 . 7 1 \pm 6 . 7 5$ </td><td> $8 0 . 7 9 2 6 . 0 8$ </td></tr><tr><td>+ cbDice [19]</td><td> $6 7 . 6 1 \pm 1 5 . 8 7$ </td><td> $\overline { { 3 3 . 7 7 \pm 2 2 . 4 3 } }$ </td><td> $2 3 . 3 3 { \pm } 2 2 . 5 4$ </td><td> $6 1 . 7 0 { \scriptstyle \pm 1 5 . 9 6 }$ </td><td> $8 6 . 1 9 { \scriptstyle \pm 7 . 5 6 }$ </td><td> $0 . 3 0 { \pm } 0 . 2 8$ </td><td> $7 . 3 0 { \pm } 5 . 3 6$ </td><td> $8 1 . 2 0 { \scriptstyle \pm 6 . 7 6 }$ </td></tr><tr><td> $+ \mathrm { S k e l R e c a l l } \left[ 1 3 \right]$ </td><td> $6 8 . 8 3 \pm 1 7 . 1 0$ </td><td> $3 3 . 9 0 { \scriptstyle \pm 2 0 . 5 6 }$ </td><td> $2 3 . 6 4 \pm 3 2 . 2 2$ </td><td> $6 2 . 3 0 { \scriptstyle \pm 1 6 . 2 9 }$ </td><td> $8 5 . 7 0 { \scriptstyle \pm 9 . 0 0 }$ </td><td> $0 . 3 0 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $8 . 0 5 { \pm } 7 . 6 6$ </td><td> $8 1 . 0 0 { \pm } 7 . 8 9$ </td></tr><tr><td>+ Betti Matching [3]</td><td> $6 6 . 9 5 { \scriptstyle \pm 1 5 . 2 8 }$ </td><td> $3 3 . 4 7 \pm 2 1 . 1 2$ </td><td> $2 4 . 7 7 \pm 2 4 . 4 0$ </td><td> $6 1 . 3 6 \pm 1 4 . 9 4$ </td><td> $8 7 . 0 7 { \pm } 8 . 2 8$ </td><td> $0 . 3 1 { \pm } 0 . 2 0$ </td><td> $5 . 7 3 { \pm } 7 . 0 3$ </td><td> $8 2 . 3 1 { \pm } 6 . 9 0$ </td></tr><tr><td>+ TRACE (ours)</td><td> ${ \bf 6 9 . 0 8 { \scriptstyle \pm 1 6 . 7 5 } }$ </td><td> $\mathbf { 3 0 . 7 7 \pm 2 0 . 4 7 }$ </td><td> $2 1 . 3 6 { \scriptstyle \pm 3 4 . 9 1 }$ </td><td> $\mathbf { 6 3 . 8 3 \pm 1 6 . 8 0 }$ </td><td> $\mathbf { 8 7 . 3 5 \pm 7 . 7 5 }$ </td><td> $\mathbf { 0 . 2 6 { \pm 0 . 2 3 } }$ </td><td> ${ \pm . 4 2 \pm 5 . 2 1 }$ </td><td> $\overline { { 8 2 . 4 2 \pm 5 . 6 9 } }$ </td></tr></table>

## 4.5. Ablation Study

jective incorporating exact local reconnection and bounded penalty integration (TRACE). We evaluate all configurations on DRIVE.

## 5. Result and Discussion

We perform a cumulative ablation to assess the contribution of TRACE’s main components. We compare six configurations: the Dice + CE baseline (DCE) used in nnU-Net [12]; conventional target-skeleton recall (Skel) [13]; its scale-native TASC replacement (TASC); TASC with the centered logit-margin score (TASC-R); an unbounded reconnection variant with $\begin{array} { r } { \phi ( a ) = a \left( \mathrm { T R A C E \mathrm { - } U } \right) } \end{array}$ ; and the full ob-

Main comparison. TRACE consistently outperforms competing methods across all six benchmark datasets. As shown in Tabs. 1 and 2, it achieves the best or near-best results in both 2D and 3D settings, covering binary and multiclass segmentation tasks. The qualitative results in Fig. 3 further support these findings: compared with competing methods, TRACE produces more continuous local structures with fewer visible discontinuities while preserving the overall geometry and semantic extent of the predictions. This agreement between quantitative and qualitative results indicates that prediction-adaptive bottleneck supervision effectively improves structural connectivity. In particular, the proposed approach places structural supervision at predictiondependent reconnection bottlenecks rather than repeatedly enforcing it at fixed skeleton locations.

![](images/6bdcdf81b05b232280cdfddcc6e78b222217d4ca60db74cb52387e01d9e2642a.jpg)  
Figure 3. Qualitative comparison across six 2D and 3D thin-structure datasets. Each row shows the ground truth, an enlarged region, and predictions from the compared methods. Yellow arrows highlight representative breaks, missing branches, and incorrect connections.

Table 3. Backbone-transfer evaluation of TRACE using $\mathrm { C S ^ { 2 } }$ -Net on DeepCrack and DRIVE. Results are averaged over test samples; bold and underlined values indicate the best and second-best results.
<table><tr><td>Dataset</td><td>Method</td><td>clDice↑</td><td>β↓</td><td>HD95↓</td><td>Dice↑</td></tr><tr><td rowspan="3">[15]</td><td>Dice + CE</td><td>89.03</td><td>6.16</td><td>16.46</td><td>82.78</td></tr><tr><td>DeepCrack + clDice [20]</td><td>90.42</td><td>5.06 4.74</td><td>15.75</td><td>84.58</td></tr><tr><td>+ SkelRecall [13] + TRACE</td><td>90.07 91.10</td><td>3.69</td><td>15.68 15.04</td><td>84.26 85.20</td></tr><tr><td rowspan="3">DRIVE [21]</td><td>Dice + CE</td><td>82.19</td><td>88.80</td><td>4.56</td><td>82.27</td></tr><tr><td>+ clDice [20]</td><td>82.58</td><td>73.90</td><td>4.04</td><td>82.64</td></tr><tr><td>+ SkelRecall [13] + TRACE</td><td>82.93 84.07</td><td>60.00 39.10</td><td>3.71 3.63</td><td>82.62 82.39</td></tr></table>

Topology preservation and training efficiency. TRACE preserves target topology across output scales while maintaining practical training efficiency. As shown in Fig. 4,

Table 4. Effect of adding TRACE to SCNP on DRIVE and DSCA under the same nnU-Net setup. Results are averaged over test samples; bold values indicate the better result within each dataset.
<table><tr><td>Dataset</td><td>Method</td><td>clDice↑</td><td>β↓</td><td>HD95↓</td><td>Dice↑</td></tr><tr><td rowspan="2">DRIVE [21]</td><td>Dice + CE w/ SCNP [25]</td><td>80.10</td><td>98.20</td><td>6.90</td><td>81.54</td></tr><tr><td>+ TRACE</td><td>81.48</td><td>90.20</td><td>4.60</td><td>81.99</td></tr><tr><td rowspan="3">DSCA [27]</td><td>Dice + CE</td><td>87.25</td><td>46.17</td><td>29.39</td><td>87.40</td></tr><tr><td>w/ SCNP [25] + TRACE</td><td>88.44</td><td>38.47</td><td>27.01</td><td>88.07</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 5. Cumulative ablation of TRACE components on DRIVE. Results are averaged over test samples; bold and underlined values indicate the best and second-best results.
<table><tr><td>Dataset</td><td>Variant</td><td>clDice↑</td><td>β↓</td><td>HD95↓</td><td>Dice↑</td></tr><tr><td rowspan="6">DRIVE [21]</td><td>DCE[12]</td><td>81.43</td><td>115.90</td><td>4.00</td><td>82.04</td></tr><tr><td>Skel [13]</td><td>81.89</td><td>97.00</td><td>4.00</td><td>82.18</td></tr><tr><td>TASC</td><td>82.01</td><td>93.40</td><td>3.82</td><td>82.24</td></tr><tr><td>TASC-R</td><td>82.23</td><td>86.70</td><td>3.58</td><td>82.31</td></tr><tr><td>TRACE-U</td><td>82.48</td><td>82.60</td><td>3.51</td><td>82.35</td></tr><tr><td>TRACE</td><td>82.65</td><td>79.30</td><td>3.45</td><td>82.38</td></tr></table>

TASC exhibits no observed topology violations on any benchmark, whereas the compared skeleton constructions can alter the original target topology. Topo-clDice preserves the topology of individual classes; however, independently skeletonizing each class can alter the topology of the foreground union. The efficiency results further show that TRACE requires substantially less training time and GPU memory than differentiable soft-skeletonization alternatives, while remaining close to lightweight target-only supervision. This efficiency stems from constructing TASC solely from the target and restricting reconnection search to a small local neighborhood, making TRACE practical for routine use.

![](images/c796cbe10c09614e2bdc08fd4a07520420f5305ca5913f9f877d8dbfa46d1bd9.jpg)  
Figure 4. Topology preservation and training efficiency across six datasets. Target topology violation rate across deep-supervision scales (top); per-epoch training time and peak GPU memory on an NVIDIA H200 (middle/bottom). OOM denotes out of memory.

Backbone transfer and method compatibility. TRACE is a plug-and-play objective that transfers across segmentation backbones and complements existing topology-aware supervision. The $\mathrm { C S ^ { 2 } }$ -Net results in Tab. 3 show improved connectivity and topology over the standard objective and compared structural losses. In Tab. 4, adding TRACE to SCNP improves all reported metrics, demonstrating complementary benefits beyond SCNP’s weakest-neighbor supervision. Because TRACE operates directly on standard semantic logits and target labels, it integrates into existing training pipelines without modifying the prediction head or inference procedure.

Ablation. The ablation study shows that the main components of TRACE provide complementary improvements. As shown in Tab. 5, replacing the conventional target skeleton with TASC improves structural consistency, while the centered logit-margin score further improves topology and boundary accuracy. Prediction-adaptive reconnection provides additional gains, and the bounded penalty integration in the full TRACE yields the best performance. These results suggest that topology-aware supervision, class-aware scoring, and adaptive reconnection each improve structural accuracy.

![](images/7cc4da85f60387b5e2fc1c2ce481cd2d9f024699a9b0a19b4f6d5f6c645b36e0.jpg)  
Figure 5. Annotation-induced artifacts in TASC. Small isolated regions and local irregularities in the ground-truth masks (arrows) are retained as isolated points or short loops because TASC preserves the annotated topology.

Limitation and future work. TRACE shows consistent performance across diverse thin-structure segmentation tasks. Nevertheless, its structural supervision depends on the topology encoded in the ground-truth annotations. Small annotation errors may introduce isolated fragments, spurious branches, or local loops, which TASC preserves because removing them would alter the annotated topology, as illustrated in Fig. 5. A promising direction is to combine TASC with uncertainty-aware filtering or consistency-based refinement to suppress unreliable topological annotations.

## 6. Conclusion

We propose TRACE, an efficient topology-aware objective that reframes skeleton supervision as prediction-adaptive local reconnection. TASC provides a class-labeled topological carrier that preserves foreground and class-specific topology while identifying same-class branches that should remain connected. Based on this representation, TRACE identifies the easiest valid reconnection under the current prediction and supervises its critical bottleneck. Across six 2D and 3D benchmarks covering binary and multiclass segmentation, TRACE improves connectivity and topology while maintaining region-overlap and boundary accuracy. It also preserves target topology across deep-supervision scales, introduces only modest training overhead, and requires no architectural or inference-time modifications. Its transferability across backbones and compatibility with existing topology-aware methods further demonstrate that TRACE is a practical plugand-play approach for thin-structure segmentation.

## References

[1] Cesar Acebes, Abdel Hakim Moustafa, Oscar Camara, and Adrian Galdran. The centerline-cross entropy loss for vessellike structure segmentation: Better topology consistency without sacrificing accuracy. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, pages 710– 720, 2024. 1, 2, 5, 6

[2] Michela Antonelli, Annika Reinke, Spyridon Bakas, Keyvan Farahani, Annette Kopp-Schneider, Bennett A. Landman, Geert Litjens, Bjoern Menze, Olaf Ronneberger, Ronald M. Summers, et al. The medical segmentation decathlon. Nature Communications, 13:4128, 2022. 1, 5, 6

[3] Alexander H. Berger, Laurin Lux, Nico Stucki, Vincent Burgin, Suprosanna Shit, Anna Banaszak, Daniel Rueckert,¨ Ulrich Bauer, and Johannes C. Paetzold. Topologically faithful multi-class segmentation in medical images. In Medical Image Computing and Computer Assisted Intervention – MIC-CAI 2024, pages 721–731, 2024. 1, 2, 5, 6

[4] Gilles Bertrand and Michel Couprie. Two-dimensional parallel thinning algorithms based on critical kernels. Journal of Mathematical Imaging and Vision, 31(1):35–56, 2008. 4

[5] Michel Couprie and Gilles Bertrand. New characterizations of simple points in 2d, 3d, and 4d discrete spaces. IEEE Transactions on Pattern Analysis and Machine Intelligence, 31(4):637–648, 2008. 4

[6] Michel Couprie and Gilles Bertrand. Asymmetric parallel 3d thinning scheme and algorithms based on isthmuses. Pattern Recognition Letters, 76:22–31, 2016. 4

[7] Ilke Demir, Krzysztof Koperski, David Lindenbaum, Guan Pang, Jing Huang, Saikat Basu, Forest Hughes, Devis Tuia, and Ramesh Raskar. DeepGlobe 2018: A challenge to parse the earth through satellite images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 172–181, 2018. 1, 5, 6

[8] DGtal Team. DGtal: Digital geometry tools and algorithms library. Software library and documentation, 2024. 4

[9] Lee R Dice. Measures of the amount of ecologic association between species. Ecology, 26(3):297–302, 1945. 5

[10] Elyar Esmaeilzadeh, Ehsan Garaaghaji, Farzad Hallaji Azad, and Doruk Oner. CAPE: Connectivity-aware path enforcement loss for curvilinear structure delineation. In Medical Image Computing and Computer Assisted Intervention – MIC-CAI 2025, pages 183–193, 2025. 1, 2

[11] Xiaoling Hu, Fuxin Li, Dimitris Samaras, and Chao Chen. Topology-preserving deep image segmentation. In Advances in Neural Information Processing Systems, 2019. 1, 2

[12] Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen, and Klaus H. Maier-Hein. nnU-Net: A selfconfiguring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2):203–211, 2021. 4, 5, 6, 7

[13] Yannick Kirchhoff, Maximilian R. Rokuss, Saikat Roy, Balint Kovacs, Constantin Ulrich, Tassilo Wald, Maximilian Zenk, Philipp Vollmuth, Jens Kleesiek, Fabian Isensee, and Klaus H. Maier-Hein. Skeleton recall loss for connectivity conserving and resource efficient segmentation of thin tubular structures.

In European Conference on Computer Vision, pages 218–234, 2024. 1, 2, 5, 6, 7

[14] Jialu Liu, Yue Cui, and Shan Yu. Ag-tal: Anatomicallyguided topology-aware loss for multiclass segmentation of the circle of willis using large-scale multi-center datasets. arXiv preprint arXiv:2604.27357, 2026. 5

[15] Yahui Liu, Jian Yao, Xiaohu Lu, Renping Xie, and Li Li. Deepcrack: A deep hierarchical feature learning architecture for crack segmentation. Neurocomputing, 338:139–153, 2019. 1, 5, 6, 7

[16] Laurin Lux, Alexander H. Berger, Alexander Weers, Nico Stucki, Daniel Rueckert, Ulrich Bauer, and Johannes C. Paetzold. Topograph: An efficient graph-based framework for strictly topology preserving image segmentation. In International Conference on Learning Representations, 2025. 1, 2

[17] Martin J. Menten, Johannes C. Paetzold, Veronika A. Zimmer, Suprosanna Shit, Ivan Ezhov, Robbie Holland, Monika Probst, Julia A. Schnabel, and Daniel Rueckert. A skeletonization algorithm for gradient-based optimization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21394–21403, 2023. 1, 2, 5, 6

[18] Lei Mou, Yitian Zhao, Huazhu Fu, Yonghuai Liu, Jun Cheng, Yalin Zheng, Pan Su, Jianlong Yang, Li Chen, Alejandro F. Frangi, Masahiro Akiba, and Jiang Liu. CS<sup>2</sup>-Net: Deep learning segmentation of curvilinear structures in medical imaging. Medical Image Analysis, 67:101874, 2021. 5

[19] Pengcheng Shi, Jiesi Hu, Yanwu Yang, Zilve Gao, Wei Liu, and Ting Ma. Centerline boundary dice loss for vascular segmentation. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, pages 46–56, 2024. 1, 2, 5, 6

[20] Suprosanna Shit, Johannes C. Paetzold, Anjany Sekuboyina, Ivan Ezhov, Alexander Unger, Andrey Zhylka, Josien P. W. Pluim, Ulrich Bauer, and Bjoern H. Menze. clDice: A novel topology-preserving loss function for tubular structure segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16560– 16569, 2021. 1, 2, 5, 6, 7

[21] Joes Staal, Michael D. Abramoff, Meindert Niemeijer, Max A.\` Viergever, and Bram van Ginneken. Ridge-based vessel segmentation in color images of the retina. IEEE Transactions on Medical Imaging, 23(4):501–509, 2004. 1, 5, 6, 7

[22] Nico Stucki, Johannes C. Paetzold, Suprosanna Shit, Bjoern H. Menze, and Ulrich Bauer. Topologically faithful image segmentation via induced matching of persistence barcodes. In Proceedings of the 40th International Conference on Machine Learning, pages 32698–32727, 2023. 1, 2, 5

[23] Abdel Aziz Taha and Allan Hanbury. Metrics for evaluating 3d medical image segmentation: Analysis, selection, and tool. BMC Medical Imaging, 15:29, 2015. 5

[24] Srinivas C. Turaga, Kevin L. Briggman, Moritz Helmstaedter, Winfried Denk, and H. Sebastian Seung. Maximin affinity learning of image segmentation. In Advances in Neural Information Processing Systems, pages 1865–1873, 2009. 1, 2

[25] Juan Miguel Valverde, Dim P. Papadopoulos, Rasmus Larsen, and Anders Bjorholm Dahl. Towards high-quality image

segmentation: Improving topology accuracy by penalizing neighbor pixels. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13123– 13133, 2026. 1, 2, 5, 7

[26] Kaiyuan Yang, Fabio Musio, Yihui Ma, Norman Juchler, Johannes C Paetzold, Rami Al-Maskari, Luciano Hoher, Hong-¨ wei Bran Li, Ibrahim Ethem Hamamci, Anjany Sekuboyina, et al. The topcow challenge—topology-aware circle of willis segmentation for ct and mr angiography. NEJM AI, 3(8): AIdbp2500994, 2026. 1, 5, 6

[27] Jiong Zhang, Qihang Xie, Lei Mou, Dan Zhang, Da Chen, Caifeng Shan, Yitian Zhao, Ruisheng Su, and Mengguo Guo. DSCA: A digital subtraction angiography sequence dataset and spatio-temporal model for cerebral artery segmentation. IEEE Transactions on Medical Imaging, 44(6):2515–2527, 2025. 1, 5, 6, 7

[28] Youcheng Zong, Runda Jia, Minxuan Hu, Weilan Su, and Dakuo He. Widest-path reachability fields for connectivitypreserving slender structure segmentation. arXiv preprint arXiv:2607.07123, 2026. 1, 2