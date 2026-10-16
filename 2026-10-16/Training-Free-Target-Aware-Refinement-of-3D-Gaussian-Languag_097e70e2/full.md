# Training-Free Target-Aware Refinement of 3D Gaussian Language Map

Sibaek Lee<sup>1</sup>, Kyeongsu Kang<sup>1</sup>, Seongbo Ha<sup>1</sup>, Yewon Jeon<sup>1</sup>, Joonyeol Choi<sup>1</sup>, Daegeol Ko<sup>1</sup>, Jaehyuk Hur<sup>1</sup>, Taemin Choi<sup>2</sup>, Minsu Kim<sup>2</sup>, Hyungjoo Ahn<sup>2</sup>, Wonsok Yoo<sup>2</sup>, and Hyeonwoo Yu<sup>1</sup>

![](images/e1af96dd75e6fe30d4d1ddca230deb4f2a01eacb32041d6cfb5e40ab2cb59d50.jpg)  
Fig. 1: Training-free target-aware refinement of a 3D Gaussian language map. An LLM turns a free-form request into a target word, and the refined map localizes it (green stars), whereas the baseline misplaces the “outlet” (red cross). The localized goal then guides a quadruped robot.

Abstract— Distilling vision-language features into 3D Gaussian Splatting yields a language map that can be queried with arbitrary text. In deployment, however, a small set of user-specified targets must be reliably located, which current distilled maps often miss, while open-vocabulary queryability for other words must remain intact. We propose a training-free, target-aware refinement of an already-distilled language map. Guided by a text-promptable segmenter, the visual embedding inside each detected target region is blended with the target text embedding, a convex combination whose coefficient sets the amount of injected prior, while non-target regions are left untouched. Injecting too much overwrites appearance cues and general queryability, so we seek, for each target, the smallest coefficient that still lets it win, and we obtain every coefficient in closed form, leaving no hyperparameters to tune. We further identify a failure mode of injection in languagegoal localization. Text embeddings share a common component, and injecting them makes the standard cosine read-out queryagnostic, which a centering step in the read-out removes. Across three benchmarks, the refinement consistently improves openvocabulary segmentation, roughly doubling the base mIoU on Replica, and localizes language goals more reliably than the base map. It also transfers as a plug-in to off-the-shelf fieldbased and object-level language maps and supports a real-world demonstration in which a legged robot navigates to naturallanguage goals. Project page and code: https://lab-of-ai-androbotics.github.io/target-aware-langmap/

## I. INTRODUCTION

A reliable spatial representation is central to robotic autonomy, since a robot must understand its environment before it can act within it across tasks such as localization, navigation, and manipulation. The recent emergence of Neural Radiance Fields (NeRF) [21] and 3D Gaussian Splatting (3DGS) [12] has made photorealistic spatial representation possible. Moreover, by combining these representations with vision language models [25], a scene is no longer restricted to a fixed set of semantic labels, and open-vocabulary NeRF and 3DGS representations [13], [37], [24] have become possible, allowing the space to be queried with arbitrary natural language.

However, existing methods for building such language maps tend to fall toward one of two ends of a spectrum, and we instead pursue the practical middle ground, precise localization of user-specified targets without giving up openvocabulary generality. Closed-set methods are trained per scene with semantic supervision [18], [38] and are accurate on their predefined labels, but cannot be queried with any word outside that set. Open-vocabulary methods instead distill vision-language features and respond to arbitrary text [13], [24], [37]. Yet because the distillation blends surrounding context into each feature, the resulting relevancy can be spatially diffuse, so a query often yields a plausible heatmap whose peak is not always precise enough to serve directly as a navigation goal. We first tried to sharpen such maps with a training-free post-processing, but found that the limitation is already present in the 2D features themselves, whose topranked retrieval is often wrong before any lifting to 3D. Purely post-hoc processing is therefore insufficient, which motivates injecting the target text directly into the features.

To this end, we propose a training-free, target-aware refinement of an already-distilled language map. Given a small set of user-specified target words, we localize each target in every training view with a text-promptable segmenter, retain the open-vocabulary base from a class-agnostic segmenter, and encode both with a vision-language model. Within each target region the visual embedding is then blended with the target text embedding, a convex combination whose coefficient decides how much prior is injected, while non-target regions are left untouched. Such coefficients are conventionally treated as hyperparameters, swept over a grid or set by heuristics. Injecting too much overwrites the region’s visual embedding and erodes the appearance cues and general queryability that make the map open-vocabulary. We therefore formulate the problem as finding the smallest coefficient that still lets each target win, and we solve it in closed form. Each detection yields this minimum as a flip point, and the per-target strength is read off the knee of the capture curve that the flip points form, leaving no hyperparameters to tune. This minimal injection, together with the untouched non-target regions, is what keeps the refined map open-vocabulary.

Because the refinement acts only on the 2D features before distillation, it is pipeline-agnostic and drops into different language maps as a plug-in module, from dense feature fields to object-level maps. It also carries over to language-goal localization, where the map must turn a query word into a 3D goal position. We observe a failure mode rooted in the geometry of contrastive vision-language embeddings [19], [22]. Text embeddings cluster around a shared direction, so injecting them plants this common component in the map, and under the standard cosine read-out every query then retrieves the same dominant regions. A centering step in the readout removes this common mode. Notably, centering barely changes the base map but is decisive on the refined one, after which the refined map localizes goals more reliably than the base across Replica [30], ScanNet [3], and HM3D-Semantics [35]. We summarize our contributions as follows:

• A training-free, target-aware refinement that makes user-specified targets reliably retrievable while keeping the map open-vocabulary, with each target’s injection strength obtained in closed form instead of tuned.

• A centered read-out that identifies and removes the query-agnostic bias of injection, turning the refined map into a reliable source of language-goal locations.

• Consistent gains in segmentation and localization on three benchmarks, plug-in transfer to field-based and object-level maps, and a real-world demonstration on a legged robot.

## II. RELATED WORK

## A. Open-vocabulary 3D language maps

Semantic mapping has long attached labels to reconstructed geometry, but classical systems remain closed set, recognizing only the categories their segmenters were trained on [20]. Open-vocabulary maps remove this limit by distilling visionlanguage embeddings into 3D, first over point clouds and implicit fields [23], [9], [13] and more recently into 3D Gaussian Splatting [24], [37], [33], with closed-form solvers reducing the distillation cost [34], [10] and online variants embedding language during SLAM [15], [6]. Dedicated benchmarks now track the open-vocabulary quality of such maps [11]. All of these works concern how to build the map, and to our knowledge none refines the language field of an alreadybuilt one, so our training-free, target-aware refinement is complementary to all of them.

## B. Language-goal localization and navigation

A language map supports many downstream tasks, and this work concerns the one closest to deployment, turning a query word into a navigation goal. Spatial memories built from vision-language features have enabled language-driven navigation over grids and implicit fields [7], [28], [16], zeroshot object search [4], scene graphs queried through language model reasoning [32], [5], and Gaussian memories for goal retrieval [36], evaluated under object-goal navigation protocols [1], [14]. Across these systems the goal is typically read out by raw cosine similarity between the query embedding and the stored features. We show that this standard readout breaks once text embeddings are injected into the map, and that a centering step in the read-out restores and then surpasses the base retrieval.

## III. METHOD

Our method refines an existing 3D Gaussian language map so that user-specified targets become reliably retrievable while the map stays open-vocabulary. For every training view we localize each target, blend its visual embedding toward the target text by a coefficient obtained in closed form, and lift the refined per-view features into 3D. At query time a centered read-out turns the refined map into 3D goal locations for navigation. Fig. 2 summarizes the pipeline.

## A. Problem Setup

We are given a posed 3DGS scene whose language field is built from multi-view vision-language features. Each Gaussian i carries an L2-normalized feature $z _ { i } \in \mathbb { R } ^ { D }$ in the joint embedding space, and a text query t scores Gaussians by the cosine similarity $z _ { i } ^ { \top } f ( t )$ , where $f ( \cdot )$ denotes the text encoder. The application additionally provides a small set of target queries $\{ t _ { k } \} _ { k = 1 } ^ { n }$ with text anchors $f ( t _ { k } )$ . A target region is retrieved only when its features score higher for $t _ { k }$ than for every competing anchor. Densely distilled features frequently fail this test, since distillation mixes surrounding context into each feature and the argmax drifts toward larger context categories.

In this setting, our goal is to refine the features so that two conditions hold. First, each target $t _ { k }$ must win this comparison reliably, so that querying $t _ { k }$ retrieves its own instances rather than the surrounding context that currently absorbs them. Second, the map must remain open-vocabulary, so features that belong to no target must stay unchanged, and even the target features should move only as much as needed. To this end the refinement introduces, for each target, one injection coefficient $\alpha _ { k } \in [ 0 , 1 ]$ that blends the target features toward the text anchor as a convex combination of the two. The problem we solve is choosing each $\alpha _ { k }$ as the smallest value for which $t _ { k }$ still wins, with no tuned hyperparameter.

![](images/cbb300d961ae455e8521c85913b06da2ff2391a1c29970dbc5692e5c93aa589b.jpg)  
Fig. 2: Overview. 1) SAM 3 localizes each user-specified target in every training view, and the region embedding u is blended toward the text anchor $f ( t _ { k } )$ with the per-target strength $\alpha _ { k } ^ { * } . \ 2 )$ The flip point of each detection, the smallest α that wins, has a closed form, and the knee of the capture curve over all views sets $\alpha _ { k } ^ { * }$ without hyperparameters. 3) The refined features are lifted onto the frozen geometry, and a centered read-out turns the map into 3D goal locations.

## B. Target-Aware Feature Injection

We construct a target-aware feature map for every training view and then lift the per-view maps into 3D, in our implementation with a closed-form feature splatting solver [34]. For each view, a class-agnostic segmenter, instantiated with SAM 2 [26] in automatic mask generation mode, produces masks that cover the view, and a vision-language model, for which we use SigLIP 2 [31], encodes each masked region into a D-dimensional embedding. These region embeddings form the open-vocabulary base of the map. For each target word $t _ { k }$ , a text-promptable segmenter, SAM 3 [2], localizes the target with a per-target mask in the same view. The shared encoder embeds that region into $u ^ { v , k }$ , and the text anchor $f ( t _ { k } )$ comes from the paired text encoder. Regions outside the target masks keep their visual embeddings, so the base is preserved by construction. Targets that overlap each other are resolved smaller mask first, which is deterministic and favors the more specific detection.

Within each target region we form the convex combination

$$
z ^ { v , k } ( \alpha ) = \mathrm { N o r m } \big ( ( 1 - \alpha ) u ^ { v , k } + \alpha f ( t _ { k } ) \big ) ,\tag{1}
$$

where $\alpha = 0$ keeps the pure visual embedding and $\alpha = 1$ is a hard text paste. What remains is the choice of this coefficient, which we settle next in closed form.

## C. Closed-Form Minimal Injection

For each view, the smallest coefficient at which the detected target region starts to win has a closed form. Write $u = u ^ { v , k } , \kappa = u ^ { \top } f ( t _ { k } ) , \beta _ { j } = u ^ { \top } f ( t _ { j } )$ , and $\gamma _ { j } = f ( t _ { k } ) ^ { \top } f ( t _ { j } )$ for unit vectors. The winning condition co $\mathsf { s } ( z ^ { v , k } ( \alpha ) , f ( t _ { k } ) ) \ge$ cos $( z ^ { v , k } ( \alpha ) , f ( t _ { j } ) )$ shares the denominator $\| ( 1 - \alpha ) u \perp$ $\alpha f ( t _ { k } ) \parallel$ on both sides, so it reduces to the linear condition $( 1 - \alpha ) \kappa + \alpha \geq ( 1 - \alpha ) \beta _ { j } + \alpha \gamma _ { j }$ , whose boundary ${ a _ { j } } =$ $\frac { \beta _ { j } - \kappa } { \beta _ { j } - \kappa + 1 - \gamma _ { j } }$ , clipped to $[ 0 , 1 ]$ , is the point where the blend overtakes competitor $j .$ The view’s flip point is the coefficient that beats all competitors,

$$
a ^ { v , k } \ = \ \operatorname* { m a x } _ { j } a _ { j } \ = \ \operatorname* { m i n } \{ \alpha \in [ 0 , 1 ] : z ^ { v , k } ( \alpha ) \ { \mathrm { r e t r i e v e s ~ } } t _ { k } \} ,\tag{2}
$$

which is 0 when the region already wins and 1 when it never does. Computing it costs a handful of dot products per view and introduces no extra constants to tune.

A single per-target coefficient $\alpha _ { k }$ cannot be minimal for every view at once, since each view flips at its own point, and injecting each view exactly at its own flip point would win by only a zero margin while varying the strength across the views that are later averaged. We therefore aggregate the flip points into an area-weighted capture curve,

$$
W _ { k } ( \alpha ) = \frac { \sum _ { v } A ^ { v , k } \mathcal { H } [ a ^ { v , k } \leq \alpha ] } { \sum _ { v } A ^ { v , k } } ,\tag{3}
$$

the non-decreasing fraction of the target’s detected area that wins at strength α, where $A ^ { v , k }$ denotes mask area. The pertarget strength is read off this curve at its knee, the point of maximum height above the chord joining the endpoints,

$$
\alpha _ { k } ^ { * } = \operatorname * { a r g m a x } _ { \alpha \in [ 0 , 1 ] } \big ( W _ { k } ( \alpha ) - \mathrm { c h o r d } _ { k } ( \alpha ) \big ) .\tag{4}
$$

The knee is where additional injection stops buying capture, a scale-invariant operating point that assumes no exchange rate between gain and cost. It may leave the most stubborn views unrecovered, trading complete capture for minimal overwrite, and after multi-view averaging the captured majority typically suffices for map-level retrieval. Fig. 3 shows the resulting curves and knees for one scene. The knee keeps $\alpha _ { k } ^ { * }$ small and adaptive, with mean strengths around 0.1 to 0.2 in our experiments and exactly 0 for targets the base already retrieves, so already-correct features are left untouched.

![](images/7a04c30810d56aa9b83620603cab7c8e11af187dbcd6e1399a59bcf80aa8b9bb.jpg)  
Fig. 3: Capture curves and knees on Replica room0. Each curve is a target’s capture curve $W _ { k } ( \alpha )$ and each dot marks its knee $\alpha _ { k } ^ { * }$ , with the remaining targets in grey. Targets whose views flip late, such as switch, receive a larger coefficient than targets the base almost retrieves, such as plate.

The per-view maps, SAM 2 regions carrying their visual embeddings and SAM 3 regions carrying $z ^ { v , k } ( \alpha _ { k } ^ { * } )$ , are then lifted by the solver, which multi-view averages the injections. Because every view is nudged only slightly and the views are averaged, the target wins class-level retrieval while its instance-level appearance survives, whereas a hard paste collapses it (Sec. IV-C). We decide the coefficient per view before aggregation, since fixing a single coefficient against the averaged map-level feature would lose the per-view adaptivity that recovers the hardest targets.

## D. Centered Localization Read-out

The refined map is N Gaussians with positions $x _ { i }$ and unitnorm features $z _ { i } .$ What remains is the read-out, the procedure that turns a text query t with anchor $q = f ( t )$ into a few 3D goal locations. The natural choice, ranking Gaussians by the raw cosine $z _ { i } ^ { \top } q .$ , works on the base map but fails once the map is refined. Injection pulls the target features toward the text anchors, which all lie in a narrow cone of the embedding space [19], [22], so every injected feature picks up a large shared component, a common mode, pointing into that cone. Because this common mode is the same for any query, it dominates the score, and different query words end up ranking the Gaussians in nearly the same order.

We remove it by centering the query with the mean anchor $\begin{array} { r } { \bar { q } = \frac { 1 } { | \mathcal { V } | } \sum _ { w \in \mathcal { V } } \bar { f ( w ) } } \end{array}$ over a query vocabulary V,

$$
s _ { i } = z _ { i } ^ { \top } ( q - \bar { q } ) = \underbrace { z _ { i } ^ { \top } q } _ { \mathrm { q u e r y ~ r e s p o n s e } } - \underbrace { z _ { i } ^ { \top } \bar { q } } _ { \mathrm { c o m m o n ~ m o d e } } ,\tag{5}
$$

which equivalently subtracts from each Gaussian its mean similarity over V, leaving only its query-specific response.

The segmentation argmax is unaffected, since the subtracted term is shared by every class anchor at a given Gaussian, so centering matters only for the per-query ranking over Gaussians. Subtracting this shared component barely changes the base map but becomes decisive here, since injection is what plants the common mode in the first place. The step is feed-forward, leaves the map itself unchanged, and needs only a generic vocabulary, as a scene-independent word list works equally well.

To extract goals we keep the top-p% Gaussians by $s _ { i } ,$ cluster their positions with DBSCAN, and rank the clusters by their similarity mass $\textstyle \sum _ { i \in c } \operatorname* { m a x } ( s _ { i } , 0 ) ^ { 2 }$ . The top-K clusters return their similarity-weighted centroids

$$
\begin{array} { r } { g _ { c } = \Big ( \sum _ { i \in c } \operatorname* { m a x } ( s _ { i } , 0 ) ^ { 2 } x _ { i } \Big ) \Big / \sum _ { i \in c } \operatorname* { m a x } ( s _ { i } , 0 ) ^ { 2 } } \end{array}\tag{6}
$$

as the goal candidates. The extraction constants, the kept fraction $p ,$ the cluster radius, and $K ,$ are fixed once across all scenes and datasets, so the pipeline stays free of perscene tuning. This centered read-out turns the segmentationoriented refinement into a reliable localizer for navigation (Sec. IV-D).

## IV. EXPERIMENTS

## A. Experimental Setup

1) Datasets and metrics: We evaluate on 8 Replica [30], 10 ScanNet [3], and the 4 multi-room HM3D-Semantics [35] scenes that LagMemo [36] curates from GOAT-Bench [14]. On each per-scene map we run two evaluations, renderedview open-vocabulary segmentation and language-goal localization for navigation. For segmentation the target queries are each scene’s annotated categories, which serve as both the deployment vocabulary and the evaluation set. For each test view we label every pixel by the argmax cosine similarity to the GT class names and score mIoU and mAcc against the rendered GT, ignoring unlabeled pixels. On HM3D-Sem, whose scenes carry over a thousand free-form categories, we follow the common practice of restricting to frequent categories [27] and keep per scene those covering at least 0.5% of the labeled pixels. The navigation-oriented localization study of Sec. IV-D instead queries the shared object-level HOV-SG vocabulary [32]. Beyond the main comparison we run two further studies, each with a single protocol across all scenes. Table II ablates injection strength, comparing our adaptive knee against the base and a range of fixed strengths. To check that injection preserves open-vocabulary behavior, the heldout study in Table III splits each scene’s classes 50/50 into an injected target set and a never-injected held-out set, scores both with the full class set, and reports the held-out change from the base.

2) Implementation details: Our pipeline builds a base from SAM 2 [26] masks encoded by SigLIP 2 [31] and applies the proposed hyperparameter-free refinement to it, lifting the per-view features with the feature-splatting solver [34]. On Replica and ScanNet the geometry is reconstructed from RGB alone, matching the RGB-only baselines, while the housescale HM3D-Sem scenes, where RGB-only reconstruction is unreliable, are initialized with depth following the RGB-D setting of LagMemo [36]. The language distillation and refinement are identical in both the RGB and RGB-D settings. On HM3D-Sem the geometry is trained on every rendered frame while the language stages of feature extraction, injection, and evaluation run on every fifth frame. The readout constants are fixed across all scenes and datasets, a kept fraction of 4% for the centered read-out and 0.5% for the raw read-out, so each is scored at its own operating point, with a cluster radius of 0.3 m and K=5.

TABLE I: Open-vocabulary segmentation, zero-shot rendered-view mIoU and mAcc, dataset means. Bold marks the best per column.
<table><tr><td></td><td colspan="2">Replica</td><td colspan="2">ScanNet</td><td colspan="2">HM3D-Sem</td></tr><tr><td>Method</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td></tr><tr><td>LangSplat [24]</td><td>0.176</td><td>0.280</td><td>0.164</td><td>0.297</td><td>0.049</td><td>0.079</td></tr><tr><td>LangSplat + Ours</td><td>0.315</td><td>0.416</td><td>0.366</td><td>0.518</td><td>0.104</td><td>0.185</td></tr><tr><td>Feature-3DGS [37]</td><td>0.263</td><td>0.345</td><td>0.571</td><td>0.708</td><td>0.127</td><td>0.179</td></tr><tr><td> $\mathrm { F e a t u r e } { - } 3 \mathrm { D G S } + \mathrm { O u r s }$ </td><td>0.324</td><td>0.406</td><td>0.611</td><td>0.750</td><td>0.170</td><td>0.238</td></tr><tr><td>Base (SAM2+SigLIP2)</td><td>0.207</td><td>0.311</td><td>0.355</td><td>0.512</td><td>0.144</td><td>0.199</td></tr><tr><td>Ours</td><td>0.436</td><td>0.546</td><td>0.542</td><td>0.694</td><td>0.231</td><td>0.313</td></tr></table>

3) Baselines: We compare against two field-based 3DGS language maps listed in Table I, each run in its published configuration and scored with our protocol. LangSplat [24] distills three-granularity SAM regions encoded by Open-CLIP [8] into a per-scene autoencoder latent, and Feature-3DGS [37] distills LSeg [17] features into the Gaussians. For each baseline, +Ours applies the same refinement to its per-view features, changing only the text anchors to that baseline’s paired text encoder. The Base row is our SAM 2 + SigLIP 2 map without refinement, the reference practice of existing open-vocabulary maps. For localization in Table IV the baseline is this same base map queried with the standard cosine read-out, against which we isolate the effect of the refinement and of the centering step. Table V additionally runs two object-level systems, HOV-SG [32] and Concept-Graphs [5], end to end in their published configurations, with + Ours applying the same refinement to their per-frame region features via their paired CLIP encoder.

## B. Open-vocabulary segmentation

Our initial attempt, a training-free entropy-based postprocessing, degraded mIoU on two language maps. The failure originates in the 2D features themselves. Even where a target region is segmented almost perfectly, the text-cosine ranks the correct word far down the list, so the region does not win its own query. A global post-hoc reweighting cannot fix this, which is why we inject the target text directly, small and selective, to preserve the original discriminability.

Our target-aware refinement instead raises the SAM 2+SigLIP 2 base from 0.207 to 0.436 mIoU on Replica. Most of the gain comes from targets the base misses entirely, such as picture, vase, and rug, which the refinement recovers from near-zero IoU to well above 0.4, while already-retrieved classes are preserved. The injection stays small and adaptive, with a mean strength of $\bar { \alpha } ^ { * } = 0 . 1 7$ Classes never detected by SAM 3 remain at zero, so the residual bottleneck there is the text-promptable segmenter’s recall rather than the injection mechanism.

TABLE II: Injection-strength ablation on all three benchmarks. The last row gives $\bar { \alpha } ^ { * }$ as Replica/ScanNet/HM3D.
<table><tr><td></td><td>Base</td><td></td><td>α=0.1 α=0.2</td><td>Ours</td><td>α=0.5</td><td>α=1</td></tr><tr><td>Replica</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>mIoU</td><td>0.207</td><td>0.360</td><td>0.416</td><td>0.436</td><td>0.461</td><td>0.471</td></tr><tr><td>mAcc</td><td>0.311</td><td>0.462</td><td>0.525</td><td>0.546</td><td>0.576</td><td>0.588</td></tr><tr><td>ScanNet</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>mIoU</td><td>0.355</td><td>0.501</td><td>0.541</td><td>0.542</td><td>0.557</td><td>0.560</td></tr><tr><td>mAcc</td><td>0.512</td><td>0.659</td><td>0.697</td><td>0.694</td><td>0.711</td><td>0.713</td></tr><tr><td>HM3D</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>mIoU</td><td>0.144</td><td>0.190</td><td>0.215</td><td>0.231</td><td>0.258</td><td>0.262</td></tr><tr><td>mAcc</td><td>0.199</td><td>0.266</td><td>0.294</td><td>0.313</td><td>0.350</td><td>0.365</td></tr><tr><td>α*</td><td>0.00</td><td>0.10</td><td>0.20</td><td>0.17/0.12/0.14</td><td>0.50</td><td>1.00</td></tr></table>

![](images/864f5543683cfbf9891278232d5e7ae39c7b815c2098109db7b74185b6a5e465.jpg)

![](images/8a301713f8786b13bfd73a37a4d4c61925c8cd62a4984e3867cf38483056d03b.jpg)  
(a) nine cushions  
(b) Ours (α \* )

![](images/73c5be7165c2f70ca5393fff8b4f9d7591cb18200194492415440da7e6938caf.jpg)  
(c) α = 1 paste  
Fig. 4: Instance-level discrimination on room0 cushions, colored by the differential relevancy for “blue cushion”. The GT blue pair is boxed in cyan and magnified, and the remaining cushions are outlined in white. Ours preserves the distinction that the α=1 paste collapses.

The refinement also works as a plug-in on off-the-shelf maps, where the only change is that the text anchors come from each map’s own text encoder. On LangSplat it raises mIoU by 79% to 123% across the three datasets. On Feature-3DGS it lifts mIoU by 23% on Replica and 34% on HM3D-Sem, improving all twelve of their scenes, and by a more modest 4.0 points on ScanNet. The injected strength adapts to the base, shrinking where the base is already strong, with a correlation of −0.80 between base mIoU and mean strength across all scenes of the three datasets, so the gains concentrate where the base is weak.

## C. Why minimal injection

A natural question is why we do not paste the target text directly at α=1 or pick one fixed α for every target. On the closed-set metric, mIoU rises monotonically with injection strength and the α=1 hard paste is highest (Table II). Our adaptive refinement reaches 0.436 on Replica, higher than any fixed α of comparable mean strength and within 8% of the hard paste. But this metric can be gamed, since it rewards pasting the very word it scores against, so a larger α is favored regardless of whether the region still carries any appearance.

The cost of hard paste appears as soon as a query needs appearance. Replica room0 contains nine cushions of different colors. Under α=1 their per-instance features collapse to essentially one vector, with mean pairwise cosine 0.999, so “blue cushion” cannot prefer the blue ones in principle. Under the differential relevancy co ${ \mathfrak { z } } ( z , f ( q ) ) - \cos ( z , f ( \mathrm { c l a s s } ) )$ which cancels the class component, the hard paste ranks neither blue cushion first, whereas our adaptive map preserves base-level appearance diversity, pairwise cosine 0.883 vs base 0.873, and ranks the two blue cushions top-2, matching the uninjected base (Fig. 4). In short, adaptive $\alpha ^ { * }$ retains most of the closed-set gain while keeping the visual information that hard paste burns.

TABLE III: Held-out generalization in mAcc, with each scene’s classes split evenly at random into an injected target half and a held-out half. $\Delta$ columns give the relative change from the base in %, and Average is the three-dataset mean.
<table><tr><td></td><td colspan="4">Replica</td><td colspan="4">ScanNet</td></tr><tr><td>α</td><td>tgt</td><td>held</td><td>∆tgt</td><td>∆held</td><td>tgt</td><td>held</td><td>∆tgt</td><td>∆held</td></tr><tr><td>Base (α=0)</td><td>0.271</td><td>0.349</td><td></td><td></td><td>0.448</td><td>0.578</td><td></td><td></td></tr><tr><td>Fixed 0.1</td><td>0.409</td><td>0.334</td><td> $+ 5 0 . 9$ </td><td>-4.3</td><td>0.662</td><td>0.530</td><td> $+ 4 7 . 8$ </td><td>-8.3</td></tr><tr><td>Fixed 0.2</td><td>0.492</td><td>0.315</td><td> $+ 8 1 . 5$ </td><td>-9.7</td><td>0.736</td><td>0.511</td><td> $+ 6 4 . 3$ </td><td>-11.6</td></tr><tr><td>Fixed 0.5</td><td>0.558</td><td>0.268</td><td> $+ 1 0 5 . 9$ </td><td>-23.2</td><td>0.775</td><td>0.491</td><td> $+ 7 3 . 0$ </td><td>-15.1</td></tr><tr><td>Fixed 1.0</td><td>0.582</td><td>0.257</td><td> $+ 1 1 4 . 8$ </td><td>-26.4</td><td>0.782</td><td>0.484</td><td> $+ 7 4 . 6$ </td><td>-16.3</td></tr><tr><td>Ours (adaptive)</td><td>0.498</td><td>0.320</td><td> $+ 8 3 . 8$ </td><td>-8.3</td><td>0.676</td><td>0.530</td><td> $+ 5 0 . 9$ </td><td>-8.3</td></tr><tr><td colspan="7">HM3D-Sem</td><td colspan="2">Average</td></tr><tr><td>Base  $( \alpha { = } 0 )$ </td><td>0.168</td><td>0.226</td><td></td><td></td><td>0.296</td><td>0.384</td><td></td><td></td></tr><tr><td>Fixed 0.1</td><td>0.268</td><td>0.206</td><td> $+ 5 9 . 5$ </td><td> $- 8 . 8$ </td><td>0.446</td><td>0.357</td><td> $+ 5 1 . 0 \mathrm { ~ \ } - 7 . 2$ </td><td></td></tr><tr><td>Fixed 0.2</td><td>0.310</td><td>0.200</td><td> $+ 8 4 . 5$ </td><td>-11.5</td><td>0.513</td><td>0.342</td><td> $+ 7 3 . 4$ </td><td>-11.0</td></tr><tr><td>Fixed 0.5</td><td>0.392</td><td>0.176</td><td> $+ 1 3 3 . 3$ </td><td>-22.1</td><td>0.575</td><td>0.312</td><td> $+ 9 4 . 5$ </td><td>-18.9</td></tr><tr><td>Fixed 1.0</td><td>0.418</td><td>0.163</td><td> $+ 1 4 8 . 8$ </td><td>-27.9</td><td>0.594</td><td>0.301</td><td> $+ 1 0 0 . 9$ </td><td>-21.6</td></tr><tr><td>Ours (adaptive)</td><td>0.335</td><td>0.195</td><td> $+ 9 9 . 4$ </td><td>-13.7</td><td>0.503</td><td>0.348</td><td> $+ 7 0 . 1$ </td><td> $^ { - 9 . 4 }$ </td></tr></table>

A complementary held-out analysis makes the same point at the open-vocabulary level (Table III), reported in mAcc since held-out mIoU is dominated by a few large background classes. Across all three datasets a larger fixed α trades held-out accuracy for target accuracy, degrading the held-out classes monotonically, by up to 26% on Replica, 16% on ScanNet, and 28% on HM3D-Sem at $\alpha { = } 1 .$ . Our adaptive map instead keeps the held-out loss small, from −8% on Replica and ScanNet to −14% on the fine-grained HM3D-Sem set, while still capturing most of the target gain, a better trade-off than any fixed α of comparable target gain. Minimal, adaptive injection is what preserves open-vocabulary performance beyond the target set.

## D. Language-goal localization

We evaluate language-goal localization directly with an instance-aware success rate, reported at top-K for $K { = } 1 \ldots 5 .$ A predicted candidate counts as a hit if it lands within 1 m of any ground-truth instance of the queried class, following the official success radius of the Habitat ObjectNav challenge [1], also used by GOAT-Bench [14]. For the query vocabulary we adopt the 20 object categories of the HOV-SG objectlevel localization benchmark [32]. Following HOV-SG, each scene is queried only with the categories present in its groundtruth annotation, with dataset-native synonyms mapped to the shared vocabulary<sup>1</sup>, which yields 56 queries on Replica, 70 on ScanNet, and 49 on HM3D-Sem, 175 in total. Candidates are extracted with the clustering read-out of Sec. III-D. The Base rows apply the same clustering to the raw cosine scores, retrieving goals by raw similarity as existing language maps do [7], [28], [36], [32], so the base stands in for the practice of these systems. Running such systems end to end would change the underlying map together with the readout, whereas holding our map fixed isolates the effect of the refinement and of the centering from the representation itself. Since the goal coordinate comes directly from the map, the agent follows the shortest path to it, path efficiency is nearoptimal by construction, and SPL reduces to the success rate, which we therefore report alone.

TABLE IV: Instance-aware localization success rate (%), top-K, on our field-based map. Base follows the raw-similarity read-out of existing maps [36], [32]. All rows share the same clustering read-out and differ only in the feature and the centering step.
<table><tr><td></td><td colspan="5">Replica</td><td colspan="5">ScanNet</td></tr><tr><td>Method</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>Base</td><td>44.6</td><td>51.8</td><td>60.7</td><td>73.2</td><td>73.2</td><td>72.9</td><td>90.0</td><td>91.4</td><td>92.9</td><td>94.3</td></tr><tr><td>Base + centering</td><td>57.1</td><td>62.5</td><td>66.1</td><td>75.0</td><td>76.8</td><td>71.4</td><td>87.1</td><td></td><td></td><td>87.1 88.6 90.0</td></tr><tr><td>Ours w/o centering</td><td>26.8</td><td>42.9</td><td>44.6</td><td></td><td></td><td></td><td>48.2 50.0 30.042.9</td><td>50.0 50.0 50.0</td><td></td><td></td></tr><tr><td>Ours</td><td>76.8 80.4 83.9</td><td></td><td></td><td>85.7</td><td></td><td></td><td>89.3 75.7 90.0 92.9</td><td></td><td>92.9 92.9</td><td></td></tr><tr><td></td><td colspan="5">HM3D-Sem</td><td colspan="5">Aggregate</td></tr><tr><td>Base</td><td>24.5</td><td></td><td></td><td></td><td></td><td></td><td>42.9 44.9 46.9 53.1 50.3 64.6 68.6 73.7 76.0</td><td></td><td></td><td></td></tr><tr><td>Base + centering</td><td>24.5</td><td>34.7</td><td>51.0 53.1 57.1 53.7</td><td></td><td></td><td></td><td>64.6</td><td></td><td></td><td>70.3 74.3 76.6</td></tr><tr><td>Ours w/o centering</td><td>10.2</td><td>14.3</td><td>16.3 22.4 22.4 23.4 34.9 38.9 41.7 42.3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td></td><td></td><td>32.7 40.8 46.9 55.1 63.3 64.0 73.1 77.1 80.0 83.4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/6b188461b6bb20287e9fb31501f51d400d22c6269cf395da355b1d6f6c29b672.jpg)  
Fig. 5: Language-goal navigation on HM3D-Sem. The green star marks the localized goal, the cyan circle the GT object, and the red trace shows the agent reaching it along the planned path. Left to right, first-person view, our 3DGS render, and the top-down map.

Table IV exposes a failure mode and its fix. Applying the refined map with the standard cosine read-out, Ours w/o centering, collapses far below the base on all three datasets, on ScanNet from 72.9 to 30.0 at top-1, because injection makes the map query-agnostic. As Fig. 6 shows, the per-Gaussian similarities of different query words become almost perfectly correlated, with the scene-averaged crossquery correlation growing from 0.28 to 0.98. Injection inflates a shared common mode that dwarfs the query-specific variation, raising the average similarity of every Gaussian to all queries from 0.056 to 0.233, so every query retrieves the same dominant injected object. The centering step of Sec. III-D removes this common mode. It helps the base only modestly, by 3.4 points at top-1 over the 175 queries, because the base has little common mode to remove, but on the refined map it is decisive.

TABLE V: Refinement as a plug-in on object-level maps, instance-aware localization success rate (%), top-K, same protocol as Table IV. HOV-SG [32] and ConceptGraphs [5] are queried through their own object-retrieval interfaces, and + Ours applies our refinement to their region features.
<table><tr><td></td><td colspan="5">Replica</td><td colspan="5">ScanNet</td></tr><tr><td>Method</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>HOV-SG [32] + Ours</td><td>14.3 12.5</td><td>16.1 19.6</td><td>26.8 21.4</td><td>26.8 21.4</td><td>30.4 25.0</td><td>78.6 90.0</td><td>82.9 90.0</td><td>87.1 92.9</td><td>95.7 98.6</td><td>98.6 100.0</td></tr><tr><td>ConceptGraphs [5] + Ours</td><td>10.7 8.9</td><td>17.9 8.9</td><td>21.4 8.9</td><td>25.0 12.5</td><td>25.0 16.1</td><td>71.4 87.1</td><td>82.9 94.3</td><td>90.0 95.7</td><td>91.4 97.1</td><td>92.9 98.6</td></tr><tr><td>HOV-SG [32]</td><td></td><td></td><td>HM3D-Sem</td><td></td><td></td><td></td><td></td><td>Aggregate</td><td></td><td></td></tr><tr><td>+ Ours</td><td>36.7 42.9</td><td>44.9 49.0</td><td>51.0 55.1</td><td>57.1 57.1</td><td>57.1 57.1</td><td>46.3 52.0</td><td>50.9 56.0</td><td>57.7 59.4</td><td>62.9 62.3</td><td>65.1 64.0</td></tr><tr><td>ConceptGraphs [5] + Ours</td><td>20.4</td><td>26.5</td><td>30.6</td><td>36.7</td><td>40.8</td><td>37.7</td><td>46.3</td><td>51.4</td><td>54.9</td><td>56.6</td></tr></table>

Ours, the refined map with the centered read-out, beats the base at every K on Replica and on the aggregate, 64.0 vs 50.3 at top-1, and beats the centering-matched base at every aggregate K as well, by 10.3 points at top-1, isolating a genuine contribution of the refinement itself rather than of centering alone. The gain is largest where the base is weakest, on Replica office1 from 17 to 83 at top-1, while on ScanNet, whose large furniture-level queries leave the base close to its ceiling, ours trails it by at most one query at every K. The ordering is unchanged at a looser 2 m radius, 76.0 vs 62.9 at top-1. The gain is also leak-free, staying identical when each query is excluded from its own centering mean and when a generic vocabulary unrelated to the scene is used instead.

Beyond this controlled comparison, Table V runs the object-level HOV-SG and ConceptGraphs end to end. Both are strong on the furniture-scale queries of ScanNet, where one clean feature per detected object avoids the context mixing of dense distillation, but collapse on the small, cluttered targets of Replica, where their mapping never instantiates the queried objects, so ours leads both by a wide margin on the aggregate, 64.0 vs 46.3 and 37.7 at top-1. Injection transfers the refinement to this map family as well, raising the aggregate top-1 of HOV-SG to 52.0 and of ConceptGraphs to 46.3. The one block that does not improve is Replica, since the injection edits regions that already exist in the map and cannot create the missing small-object instances, which is precisely what the dense field representation provides.

An injection-strength sweep on HM3D-Sem confirms the mechanism. Without centering, top-1 success collapses from 24.5% on the base to 10.2% on the refined map, whereas with centering it rises from 24.5% to 32.7%. The centered readout even keeps improving past the knee, 38.8 at a fixed α=1 vs 32.7, but that regime collapses the held-out performance of Table III and the instance appearance of Fig. 4, so the knee remains the right operating point. Finally, Fig. 5 shows the read-out driving actual navigation, with the agent reaching each queried object along the planned path.

![](images/b29b975a619f2653a7b8d8bdbd3512c59a8858359c1b0d13ba0527376fb23f26.jpg)  
(a) Base

![](images/a8c61f6b7943c211aa1ff668d433f787dfcdec251fde1d1b2d49b0838b173ab3.jpg)  
(b) Ours w/o cent.

![](images/0b2c4a5e3e23eaabf2fbf60c6eb4eeebb1d806ffaa89b7f3f151f2469d2215d3.jpg)  
(c) Ours + cent.  
Fig. 6: Cross-query similarity correlation on a representative HM3D-Sem scene. Injection makes the refined map queryagnostic, and centering restores discriminability.

## E. Real-world demonstration

We deploy the full pipeline on a real building floor with a Unitree Go2 quadruped, shown in Fig. 1. A handheld rig sweeps the floor once, a LiDAR-inertial SLAM system [29] estimates the sensor poses, and the posed RGB frames are lifted into a 3DGS language map by the same featuresplatting solver [34] and refined by our method, so the map construction is identical to the simulated setting. At run time the user issues a free-form natural-language request, a language-model front end selects the matching target word from the map’s vocabulary, and the centered read-out turns that word into a 3D goal on the refined map. The robot localizes itself against the LiDAR map and plans toward the goal with $\mathbf { A } ^ { * }$ on a 2D occupancy grid, so the language map supplies the goal and the LiDAR map supplies the pose and the grid in one shared frame. The refined map guides the robot to the correct object with a success rate of 78%, whereas the base map with the standard raw-similarity read-out reaches 22%, and the margin comes mostly from the more specific queries on which the base map fails.

## V. CONCLUSIONS

We presented a training-free, target-aware refinement of 3D Gaussian language maps that makes user-specified targets reliably retrievable while keeping the map open-vocabulary. Each target’s injection strength is obtained in closed form, so the refinement introduces no hyperparameters, and a centered read-out removes the query-agnostic common mode that injection plants, turning the refined map into a dependable source of language-goal locations. Across three benchmarks the same refinement improves both open-vocabulary segmentation and localization, and the gain carries over to a legged robot that navigates to natural-language goals on a real building floor. Because the refinement acts on per-view features before lifting, it drops into existing language-map pipelines, and extending it to online mapping and to targets observed only sparsely is a natural next step.

## ACKNOWLEDGMENT

This work was supported by Samsung Electronics.

[1] D. Batra, A. Gokaslan, A. Kembhavi, O. Maksymets, R. Mottaghi, M. Savva, A. Toshev, and E. Wijmans, “Objectnav revisited: On evaluation of embodied agents navigating to objects,” arXiv preprint arXiv:2006.13171, 2020.

[2] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang, et al., “Sam 3: Segment anything with concepts,” arXiv preprint arXiv:2511.16719, 2025.

[3] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor scenes,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828–5839.

[4] S. Y. Gadre, M. Wortsman, G. Ilharco, L. Schmidt, and S. Song, “Cows on pasture: Baselines and benchmarks for language-driven zero-shot object navigation,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 23 171–23 181.

[5] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, et al., “Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 5021–5028.

[6] S. Ha, S. Lee, K. Kang, J. Choi, S. Tak, and H. Yu, “Langgs-slam: Real-time language-feature gaussian splatting slam,” arXiv preprint arXiv:2602.06991, 2026.

[7] C. Huang, O. Mees, A. Zeng, and W. Burgard, “Visual language maps for robot navigation,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 10 608–10 615.

[8] G. Ilharco, M. Wortsman, N. Carlini, R. Taori, A. Dave, V. Shankar, H. Namkoong, J. Miller, H. Hajishirzi, A. Farhadi, et al., “Openclip,” Zenodo, 2021.

[9] K. M. Jatavallabhula, A. Kuwajerwala, Q. Gu, M. Omama, T. Chen, A. Maalouf, S. Li, G. Iyer, S. Saryazdi, N. Keetha, et al., “Conceptfusion: Open-set multimodal 3d mapping,” arXiv preprint arXiv:2302.07241, 2023.

[10] K. Jun-Seong, G. Kim, K. Yu-Ji, Y.-C. F. Wang, J. Choe, and T.-H. Oh, “Dr. splat: Directly referring 3d gaussian splatting via direct language embedding registration,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 14 137– 14 146.

[11] C. Kassab, S. Morin, M. Buchner, M. Mattamala, K. Gupta, A. Valada,¨ L. Paull, and M. Fallon, “Openlex3d: A tiered benchmark for openvocabulary 3d scene representations,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[12] B. Kerbl, G. Kopanas, T. Leimkuhler, G. Drettakis,¨ et al., “3d gaussian splatting for real-time radiance field rendering.” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.

[13] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, and M. Tancik, “Lerf: Language embedded radiance fields,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 19 729–19 739.

[14] M. Khanna, R. Ramrakhya, G. Chhablani, S. Yenamandra, T. Gervet, M. Chang, Z. Kira, D. S. Chaplot, D. Batra, and R. Mottaghi, “Goat-bench: A benchmark for multi-modal lifelong navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16 373–16 383.

[15] S. Lee, S. Ha, K. Kang, J. Choi, S. Tak, and H. Yu, “Legoslam: Language-embedded gaussian optimization slam,” arXiv preprint arXiv:2511.16144, 2025.

[16] S. Lee, H. Yu, G. Kim, and S. Choi, “Lamp: Implicit language map for robot navigation,” IEEE Robotics and Automation Letters, 2025.

[17] B. Li, K. Q. Weinberger, S. Belongie, V. Koltun, and R. Ranftl, “Language-driven semantic segmentation,” arXiv preprint arXiv:2201.03546, 2022.

[18] M. Li, S. Liu, H. Zhou, G. Zhu, N. Cheng, T. Deng, and H. Wang, “Sgsslam: Semantic gaussian splatting for neural dense slam,” in European Conference on Computer Vision. Springer, 2024, pp. 163–179.

[19] V. W. Liang, Y. Zhang, Y. Kwon, S. Yeung, and J. Y. Zou, “Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning,” Advances in Neural Information Processing Systems, vol. 35, pp. 17 612–17 625, 2022.

[20] J. McCormac, A. Handa, A. Davison, and S. Leutenegger, “Semanticfusion: Dense 3d semantic mapping with convolutional neural networks,” in 2017 IEEE International Conference on Robotics and automation (ICRA). IEEE, 2017, pp. 4628–4635.

[21] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.

[22] J. Mu, S. Bhat, and P. Viswanath, “All-but-the-top: Simple and effective postprocessing for word representations,” arXiv preprint arXiv:1702.01417, 2017.

[23] S. Peng, K. Genova, C. Jiang, A. Tagliasacchi, M. Pollefeys, and T. Funkhouser, “Openscene: 3d scene understanding with open vocabularies,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 815–824.

[24] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister, “Langsplat: 3d language gaussian splatting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 051–20 060.

[25] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[26] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland, L. Gustafson, ¨ et al., “Sam 2: Segment anything in images and videos,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 28 085–28 128.

[27] D. Rozenberszki, O. Litany, and A. Dai, “Language-grounded indoor 3d semantic segmentation in the wild,” in European conference on computer vision. Springer, 2022, pp. 125–141.

[28] N. M. M. Shafiullah, C. Paxton, L. Pinto, S. Chintala, and A. Szlam, “Clip-fields: Weakly supervised semantic fields for robotic memory,” arXiv preprint arXiv:2210.05663, 2022.

[29] T. Shan, B. Englot, D. Meyers, W. Wang, C. Ratti, and D. Rus, “Lio-sam: Tightly-coupled lidar inertial odometry via smoothing and mapping,” in 2020 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2020, pp. 5135–5142.

[30] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, et al., “The replica dataset: A digital replica of indoor spaces,” arXiv preprint arXiv:1906.05797, 2019.

[31] M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y. Xia, B. Mustafa, et al., “Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features,” arXiv preprint arXiv:2502.14786, 2025.

[32] A. Werby, C. Huang, M. Buchner, A. Valada, and W. Burgard,¨ “Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation,” in First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024.

[33] Y. Wu, J. Meng, H. Li, C. Wu, Y. Shi, X. Cheng, C. Zhao, H. Feng, E. Ding, J. Wang, et al., “Opengaussian: Towards point-level 3d gaussian-based open vocabulary understanding,” Advances in Neural Information Processing Systems, vol. 37, pp. 19 114–19 138, 2024.

[34] B. Xiong, R. Liu, K. Xu, M. Chen, and A. Feng, “Splat feature solver,” arXiv preprint arXiv:2508.12216, 2025.

[35] K. Yadav, R. Ramrakhya, S. K. Ramakrishnan, T. Gervet, J. Turner, A. Gokaslan, N. Maestre, A. X. Chang, D. Batra, M. Savva, et al., “Habitat-matterport 3d semantics dataset,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4927–4936.

[36] H. Zhou, X. Wang, H. Li, Z. Qi, J. Yin, H. Kong, J. Xu, and H. Zhao, “Lagmemo: Language 3d gaussian splatting memory for multi-modal open-vocabulary multi-goal visual navigation,” arXiv preprint arXiv:2510.24118, 2025.

[37] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You, Z. Wang, and A. Kadambi, “Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 676–21 685.

[38] S. Zhu, G. Wang, H. Blum, J. Liu, L. Song, M. Pollefeys, and H. Wang, “Sni-slam: Semantic neural implicit slam,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 167–21 177.