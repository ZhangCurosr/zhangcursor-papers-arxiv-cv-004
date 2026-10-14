# SPARCL: Spatially Adaptive Regional Coverage for Active Learning in Chest X-ray Analysis

Juho Jung, Seongjae Kang, Mijoo Kim, Sungduk Cho, Yunkyoung Jun, Sunghoon Joo and Yoona Hwang<sup>†</sup>

Abstract— Pixel-level annotation for chest X-ray (CXR) lesion segmentation is expensive, making active learning (AL) essential. However, existing AL strategies fail to account for the spatial distribution of selected samples, resulting in inconsistent performance across focal (e.g., aortic calcification) and diffuse (e.g., nodules) pathologies. In this paper, we propose SPARCL (Spatially Adaptive Regional Coverage for Active Learning) to dynamically balance predictive uncertainty with cumulative spatial coverage. By employing a concave coverage function, SPARCL imposes diminishing returns on previously annotated regions, naturally directing the annotation budget toward under-labeled areas. Furthermore, an entropy gate reverts the algorithm to standard uncertainty sampling when spatial variation is absent. This coverage objective is monotone submodular, guaranteeing near-optimal greedy batch selection. To address how aggregate metrics like Dice obscure regional failures, we further introduce two metrics—Spatial Inconsistency (SI) and Performance Coverage (PC)—to quantify anatomical fairness across AL rounds. Evaluated on an in-house calcification cohort and public CXR benchmarks, SPARCL reaches 95% of fully supervised performance using only 24% of the data, compared to 29% for the best baseline. Ultimately, SPARCL ensures robust spatial reliability, surfacing regional gains that aggregate metrics miss.

Index Terms— Active learning, Chest X-ray, Segmentation, Spatial Fairness, Uncertainty

## I. INTRODUCTION

Pixel-level annotation for chest X-ray (CXR) lesion segmentation is expensive and time-consuming, requiring radiologists to carefully delineate subtle anatomical structures across diverse pathological contexts [1], [2]. Crucially, model performance is not uniform; it often degrades in challenging areas such as the retrocardiac space or lung apices [3], [4]. Active Learning (AL) addresses this by iteratively selecting the most informative samples for annotation, aiming to maximize model performance under limited labeling budgets [10], [11]. However, standard classification AL strategies [5], [9], [15], [16] often fail in medical segmentation due to their inability to adapt to distinct spatial patterns [12], [17], [22]. The core challenge is not merely selecting “hard” samples, but selecting samples that provides the robust improvements across spatial regions [6], [7].

Specifically, we identify two opposing regional patterns arising from the spatially heterogeneous nature of CXR pathologies: a diffuse pattern, where lesions scatter across distant anatomical regions (e.g., diffuse pulmonary nodules), and a focal pattern, where pathology is spatially compact (e.g., focal aortic calcification, pneumothorax). However, existing AL strategies fail to account for this spatial structure: uncertainty-based methods [12], [24]–[26], [37], [39] often oversample redundant local neighborhoods while missing informative signals in distant regions [13], [14], [17], [19]– [21]. Diversity-based methods [8], [16], [19], [20], [31], [32] promote image-level variety but remain agnostic to region-level spatial structure, offering no guarantee that selected batches cover underrepresented anatomical areas [17], [18], [22]. Even naive region-balancing heuristics [22], [23], [33], [34] assume uniform information distribution across regions, wasting budget on uninformative normal regions.

To bridge this gap, we propose SPARCL (Spatially Adaptive Regional Coverage for Active Learning). Unlike static policies, SPARCL balances predictive uncertainty with spatial coverage that prioritizes regions underrepresented in the labeled set. Coverage is accumulated across rounds, so regions in which lesions have already been annotated yield diminishing returns while under-labeled regions keep attracting the budget; a gate driven by the positional diversity of uncertainty across the most uncertain candidates switches coverage off only when there is no spatial variation to exploit, in which case SPARCL reduces to uncertainty sampling. This allows a unified framework to handle focal and diffuse pathologies without manual heuristic tuning.

Furthermore, we argue that aggregate metrics like Dice or IoU are insufficient for AL evaluation for CXR segmentation, as they obscure regional biases [27]–[29], [36]. We introduce two region-level metrics that quantify spatial reliability: Spatial Inconsistency (SI), which measures performance disparity across anatomical regions, and Performance Coverage (PC), which captures consistent improvements across anatomical regions between AL rounds. We summarize the key contributions as follows.

1) We demonstrate that CXR pathologies exhibit two distinct spatial regimes (focal vs. diffuse), and reveal how existing AL methods fail to maintain consistent performance across these diverse anatomical patterns.

![](images/0679e42905457d6a2e9adb594eac6673dc97098be4fb04cbaf204c0591376d07.jpg)  
Fig. 1. Overall framework of SPARCL. Step 1: Each unlabeled image $\mathbf { \Delta } x \in \mathcal { U } _ { t - 1 }$ yields an uncertainty score U(x) and a spatial distribution q(x) of its pixel entropy. Over the top-rB candidates, the entropy H(¯q) of their pooled distribution sets the positional-diversity gate $\lambda ^ { ( t ) }$ , lowering the spatial weight when the candidates’ uncertainty concentrates in the same few bins. Step 2: Rather than starting from zero, the coverage state H(S) is initialized to $H _ { 0 } ,$ the ground-truth lesion distribution g(x) of the labeled set. Greedy marginal-gain maximization then adds one image at a time, updating $H ( s )$ after each pick, so that it rewards uncertainty and bins underrepresented by both the labeled set and the current batch. Step 3: The selected samples are annotated, merged into $\pmb { \mathcal { L } } _ { t } ,$ , and used to retrain the model.

2) We propose SPARCL, an entropy-gated acquisition strategy that dynamically balances predictive uncertainty with cumulative spatial coverage. Its coverage objective is rigorously shown to be monotone submodular, guaranteeing near-optimal greedy batch selection while seamlessly adapting to both focal and diffuse regimes.

3) We introduce two novel region-level evaluation metrics— Spatial Inconsistency (SI) and Performance Coverage (PC)—to rigorously quantify anatomical “blind spots.” These provide a new standard to evaluate spatial fairness across any AL acquisition strategy, exposing regional biases obscured by standard aggregate metrics.

## II. RELATED WORK

## A. Active learning for medical image segmentation

Uncertainty sampling selects images on which the model is least confident, using predictive entropy or MC-dropout/BALD estimates [15], [17], [24], learned loss prediction [38], or edge/region-aware cues [21], [25]. Representativeness-based methods instead promote batch diversity through core-set [16], gradient embeddings [20], or hybrid criteria [5], [8], [11], [30]. None of these criteria tracks where in the image evidence has already been labeled.

## B. Batch and submodular acquisition

Submodularity prevents redundant batch selections by guaranteeing a greedy $( 1 - 1 / e )$ approximation [44]. Prior works apply submodularity to batch mutual information [19], [37] or Gaussian random fields [45] in feature space. Coverage has also been cast in embedding space [31], [32]; SPARCL instead applies the concave-over-modular reward of Lin and Bilmes [35] to a spatial histogram accumulated over the labeled lesions, inheriting the $( 1 - 1 / e )$ guarantee.

## C. Evaluating spatial reliability

Aggregate metrics like Dice and IoU summarize an entire image but mask large regional disparities [27]–[29]. Metrics

Reloaded [36] formalizes these pitfalls and recommends complementary metrics that expose localized failures hidden by aggregate overlap; in that spirit, and alongside fairness-oriented evaluation [26], SI and PC are region-level complements to Dice—not replacements—that expose anatomical blind spots and quantify spatial learning uniformity across AL rounds.

## III. METHODS

## A. Problem Setup and Notation

Let $\mathcal { U } _ { 0 } ~ = ~ \{ x _ { j } \} _ { j = 1 } ^ { N }$ be an initial unlabeled pool of N CXR images, and $\mathbf { \bar { \mathcal { L } } } _ { 0 } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { M _ { 0 } }$ be a seed set of $M _ { 0 }$ labeled images with binary masks $y _ { i } \in \{ 0 , 1 \} ^ { H \times W }$ . We train a segmentation network $f _ { \theta }$ to minimize the supervised loss $L _ { \mathrm { s e g } }$ In each active learning (AL) round t, a batch of unlabeled images $B _ { t } \subset \mathcal U _ { t - 1 }$ of size B is selected by maximizing a set-level acquisition function A(S) (Eq. (4)). These selected samples are then annotated by an oracle $( \mathrm { e . g . }$ , radiologist) to obtain their corresponding masks y, and both sets are updated: $\mathcal { L } _ { t }  \mathcal { L } _ { t - 1 } \cup \{ ( x , y ) \mid x \in \mathcal { B } _ { t } \}$ and ${ \mathcal { U } } _ { t } \gets { \mathcal { U } } _ { t - 1 } \setminus { \mathcal { B } } _ { t }$ . To evaluate spatial reliability, we partition the image domain Ω into M disjoint spatial bins $\lbrace \Omega _ { m } \rbrace _ { m = 1 } ^ { M }$

## B. SPARCL: Spatially Adaptive Regional Coverage

As outlined in Fig. 1, SPARCL balances predictive uncertainty with cumulative spatial coverage. Each round scores every unlabeled image by how uncertain it is and where that uncertainty lies, restricts selection to the most uncertain candidates, initializes a coverage state from the bins that already-labeled lesions occupy, and greedily selects a batch that is both uncertain and complementary to what has been labeled; a gate $\lambda ^ { ( t ) }$ switches coverage off only when the candidates’ uncertainty has no spatial variation to exploit.

1) Predictive Uncertainty Estimation.: Let $p ( i ) \in [ 0 , 1 ]$ be the foreground probability that $f _ { \theta }$ assigns to pixel i. Its predictive entropy is $H ( i ) = - p ( i ) \log p ( i ) - ( 1 - p ( i ) ) \log ( 1 - p ( i ) )$ and the image-level uncertainty is the mean entropy over pixels, $\begin{array} { r } { U ( x ) = \frac { 1 } { | \Omega | } \mathrm { \bar { \sum } } _ { i \in \Omega } H ( i ) } \end{array}$ . Any stochastic estimate of $p ( i ) \ ( { \mathrm { e . g . } }$

![](images/c12ba00f1e60d9e092c5c455650b7c806b5215d33a3ed0e7f93c349dedcd6c68.jpg)  
Fig. 2. Active learning performance across annotation budgets (10–100%). SPARCL reaches the highest Dice(↑ ) and lowest SI(↓) at every budget up to 50% and the highest PC(↑ ) at ≤ 20%. PC is a per-round gain and decays for all methods as learning saturates.

MC Dropout or test-time augmentation) can be substituted without changing what follows.

2) Uncertainty Spatial Histogram.: For each image $x ,$ we construct a spatial histogram $\bar { \mathbf { h } } ( x ) \in \mathbb { R } ^ { M }$ with $M = K L$ for a $K \times L$ grid. The m-th entry is $\begin{array} { r } { h _ { m } ( x ) \ = \ \sum _ { i \in \Omega _ { m } } H ( i ) } \end{array}$ aggregating pixel entropy within bin m. We normalize it per image, $\begin{array} { c l r } { { q _ { m } ( x ) } } & { { = } } & { { h _ { m } ( x ) / \sum _ { j } { h _ { j } ( x ) } } } \end{array}$ , so that $\mathbf { q } ( x )$ is a distribution over bins that describes where the image’s uncertainty lies irrespective of its total amount, which $U ( x )$ already captures. Both the gate and the coverage term below operate on q(x).

3) Candidate Set (Uncertainty Floor).: Coverage should redirect the budget among informative images, not admit uninformative ones. Each round is therefore restricted to the candidate set $\mathcal { U } _ { t - 1 } ^ { \prime } \subseteq \mathcal { U } _ { t }$ of the rB images with the largest $U ( x )$ $( r = 5$ throughout), over which $U ( x )$ is min–max normalized to [0, 1]; whatever its coverage gain, an image outside the toprB by uncertainty is never selected. All quantities below are computed on $\mathcal { U } _ { t } ^ { \prime } .$

4) Cumulative Coverage State.: For every lesion-bearing labeled image $x \in \mathcal { L } _ { t - 1 } ^ { + }$ we form the analogous distribution of its ground-truth lesion over bins, $g _ { m } ( x ) = | \{ i \in \Omega _ { m } : y ( i ) =$ $1 \} | / | \{ i \in \Omega : y ( i ) = 1 \} |$ , and initialize the coverage state of round t as

$$
H _ { 0 , m } = \frac { B } { | \mathcal { L } _ { t - 1 } ^ { + } | } \sum _ { x \in \mathcal { L } _ { t - 1 } ^ { + } } g _ { m } ( x ) , \qquad m = 1 , \ldots , M ,\tag{1}
$$

the labeled set’s lesion distribution rescaled to the mass of one batch. $H _ { 0 }$ records where lesions have already been annotated and q(x) where the model is now uncertain; both are perimage distributions over the same bins, so they accumulate in one histogram, and bins already dense in the labeled set yield small concave gains while under-labeled bins keep a large gain from the first selection of the round. Within a round the state advances by the predicted $\mathbf { q } ( x )$ of each selected image, but it is rebuilt from ground truth at the next round, so cross-round memory is carried by annotations and estimation noise in q does not accumulate.

5) Positional-Diversity Gate.: Cumulative coverage requires spatial variation across images, so we gate the coverage weight by the entropy of the pooled uncertainty distribution over bins,

Algorithm 1: SPARCL Sample Acquisition (round t)   
Input: Segmenter f ; pool $\mathcal { U } _ { t - 1 } ;$ labeled set $\overline { { \mathcal { L } _ { t - 1 } } }$ with masks   
$\{ y _ { x } \}$ ; budget B; M bins $\{ \Omega _ { m } \} ;$ candidate ratio r; gate   
ceiling $\lambda _ { \operatorname* { m a x } } ;$ concave $\psi ( z ) = { \sqrt { z } }$   
Output: Batch B   
foreach $x \in \mathcal { U } _ { t - 1 }$ do   
H(i) ← pixel entropy of $\begin{array} { r } { f _ { \theta } ( x ) ; \ U ( x ) \gets \frac { 1 } { | \Omega | } \sum _ { i } H ( i ) ; } \end{array}$   
$\begin{array} { r } { q _ { m } ( x ) \gets \sum _ { i \in \Omega _ { m } } H ( i ) / \sum _ { i \in \Omega } H ( i ) } \end{array}$   
$\mathcal { U } _ { t } ^ { \prime }  \mathrm { t o p } – r B \ \mathrm { o f } \ \mathcal { U } _ { t - 1 } \ \mathrm { b y } \ U ( x ) ;$   
$U ( x ) \stackrel { * } {  } \big ( U ( x ) - \mathrm { m i n } _ { \mathcal { U } _ { t } ^ { \prime } } U \big ) \big / \big ( \mathrm { m a x } _ { \mathcal { U } _ { t } ^ { \prime } } U - \mathrm { m i n } _ { \mathcal { U } _ { t } ^ { \prime } } U \big )$   
$\begin{array} { r } { \bar { q } _ { m }  \frac { 1 } { \vert \mathcal { U } _ { \mathrm { * } } ^ { \prime } \vert } \sum _ { x \in \mathcal { U } _ { \mathrm { * } } ^ { \prime } } q _ { m } ( x ) ; \ \lambda ^ { ( t ) }  \lambda _ { \operatorname* { m a x } } \mathcal { H } ( \bar { \bf q } ) / \log M } \end{array}$   
g<sub>m</sub> $\quad : \quad \ : \ : \vert \{ i \in \Omega _ { m } : y _ { x } ( i ) = 1 \} \vert / \vert \{ i : y _ { x } ( i ) = 1 \} \vert$ for   
$\begin{array} { r } { x \in \mathcal { L } _ { t - 1 } ^ { + } ; H _ { S , m }  \frac { B } { \vert \mathcal { L } _ { t - 1 } ^ { + } \vert } \sum _ { x \in \mathcal { L } _ { t - 1 } ^ { + } } g _ { m } ( x ) ; \mathcal { S }  \emptyset } \end{array}$   
while $| S | < B$ do   
$\begin{array} { r } { | \begin{array} { l } { x ^ { \star }  \arg \operatorname* { m a x } _ { x \in \mathcal { U } _ { t } ^ { \prime } \backslash S } U ( x ) + \lambda ^ { ( t ) } \sum _ { m } \big [ \psi ( H _ { S , m } + q _ { m } ( x ) ) - \psi ( H _ { S , m } + q _ { m } ( x ) ) \big ] } \end{array}  } \end{array}$   
$\psi ( H _ { S , m } ) ]$   
$S \stackrel { . } {  } \bar { S } \cup \{ x ^ { \star } \} ; H _ { S , m } \mathrel { + } = q _ { m } ( x ^ { \star } )$   
return $\begin{array} { r l } { B _ { t } = S } & { { } / / } \end{array}$ greedy on monotone submodular A:   
(1−1/e)-optimal $\mathrm { ~  ~ { ~ \vert ~ \mathrm { ~ C ~ o ~ r ~ . ~ } ~ 1 ~ ) ~ } ~ }$   
// Detection: $\begin{array} { r } { q _ { m } ( x ) \propto \sum _ { o \in \mathcal { B } _ { x } } s _ { o } \mathcal { H } ( s _ { o } ) \mathrm { I o U } ( o , \Omega _ { m } ) , } \end{array}$   
$\begin{array} { r } { U ( x ) \gets \sum _ { o } s _ { o } \mathcal { \hat { H } } ( \dot { s } _ { o } ) } \end{array}$ (box score s<sub>o</sub>), g<sub>m</sub>(x)∝   
ground-truth box area in $\Omega _ { m } ;$ steps 2-5   
unchanged

$$
\bar { q } _ { m } = \frac { 1 } { \vert \mathcal { U } _ { t } ^ { \prime } \vert } \sum _ { x \in \mathcal { U } _ { t } ^ { \prime } } q _ { m } ( x ) , \qquad \lambda ^ { ( t ) } = \lambda _ { \operatorname* { m a x } } \frac { \mathcal { H } ( \bar { \mathbf { q } } ) } { \log M } \in [ 0 , \lambda _ { \operatorname* { m a x } } ] ,\tag{2}
$$

with $\begin{array} { r } { \mathcal { H } ( \bar { \bf q } ) = - \sum _ { m } \bar { q } _ { m } \log \bar { q } _ { m } } \end{array}$ . This entropy is high when uncertainty spans many bins (e.g., diffuse lesions, or focal lesions varying across patients). It is low only when uncertainty consistently concentrates in the same few bins (e.g., a single-site pathology). In this regime, enforcing coverage would merely select spurious background noise, so the gate closes, reverting SPARCL to pure uncertainty sampling. We use the pooled rather than mean per-image entropy because, by Jensen’s inequality, their difference $\begin{array} { r } { \mathcal { H } ( \bar { \mathbf { q } } ) - \frac { 1 } { | \mathcal { U } _ { * } ^ { \prime } | } \sum _ { x } \mathcal { H } ( \mathbf { q } ( x ) ) } \end{array}$ captures the betweenpatient spatial variation—precisely the diversity our coverage exploits.

6) Acquisition Objective.: We employ a greedy strategy with a concave gain ψ, by default $\psi ( z ) ~ = ~ { \sqrt { z } } ~ [ 3 5 ]$ (Remark 1 covers the regularized family). The marginal gain of adding image x to the selected set S is:

TABLE I  
LABEL EFFICIENCY ON SEGMENTATION BENCHMARKS, MEAN ± STD OVER FIVE SHARED RANDOM SEEDS. DICE@20% IS THE PER-IMAGE DICE (×100) AT A 20% ANNOTATION BUDGET, RANDOM HELD-OUT SPLIT; AULC IS THE NORMALIZED AREA UNDER THE DICE-VS-BUDGET CURVE. BUDGET@95% IS THE MINIMUM BUDGET TO REACH 95% OF FULL-DATA DICE (IN-HOUSE 46.8 ± 1.7, SIIM-ACR 61.8 ± 0.4 AT 100% BUDGET). BOLD = BEST IN COLUMN, UNDERLINE = SECOND BEST IN COLUMN.
<table><tr><td></td><td colspan="3">In-house Calcification</td><td colspan="3">SIIM-ACR</td></tr><tr><td>Method</td><td>Dice@20%↑</td><td>AULC↑</td><td>Budget@95%↓</td><td>Dice@20%↑</td><td>AULC↑</td><td>Budget@95%↓</td></tr><tr><td>Random</td><td> $3 7 . 3 \pm 1 . 2$ </td><td> $4 2 . 7 \pm 1 . 3$ </td><td> $2 9 . 7 \pm 3 . 3 \%$ </td><td> $5 2 . 9 \pm 2 . 2$ </td><td> $5 6 . 9 \pm 1 . 5$ </td><td> $2 5 . 5 \pm 3 . 2 \%$ </td></tr><tr><td>Area-Random</td><td> $3 8 . 3 \pm 0 . 4$ </td><td> $4 3 . 0 \pm 1 . 2$ </td><td> $2 9 . 9 \pm 1 . 6 \%$ </td><td> $5 4 . 1 \pm 2 . 1$ </td><td> ${ \underline { { 5 9 . 3 \pm 2 . 3 } } }$ </td><td> $2 4 . 7 \pm 2 . 1 \%$ </td></tr><tr><td>Uncertainty</td><td> $4 1 . 2 \pm 1 . 1$ </td><td> $\underline { { 4 3 . 7 \pm 0 . 5 } }$ </td><td> $2 9 . 3 \pm 0 . 9 \%$ </td><td> $5 3 . 6 \pm 1 . 8$ </td><td> $\overline { { 5 8 . 9 \pm 1 . 6 } }$ </td><td> $\underline { { 2 2 . 4 } } \pm 1 . 1 \%$ </td></tr><tr><td>Area-Uncertainty</td><td> $4 1 . 5 \pm 0 . 2$ </td><td> $\overline { { 4 3 . 4 \pm 0 . 8 } }$ </td><td> $\overline { { 2 9 . 5 \pm 0 . 7 \% } }$ </td><td> $5 3 . 8 \pm 0 . 5$ </td><td> $5 8 . 1 \pm 2 . 2$ </td><td> $\overline { { 2 4 . 2 \pm 1 . 6 \% } }$ </td></tr><tr><td>Coreset</td><td> $4 0 . 9 \pm 2 . 2$ </td><td> $4 3 . 4 \pm 0 . 9$ </td><td> $2 9 . 9 \pm 2 . 2 \%$ </td><td> $5 2 . 7 \pm 1 . 0$ </td><td> $5 7 . 5 \pm 0 . 7$ </td><td> $3 1 . 3 \pm 2 . 8 \%$ </td></tr><tr><td>Diversity</td><td> $3 8 . 4 \pm 2 . 1$ </td><td> $4 2 . 3 \pm 1 . 1$ </td><td> $5 7 . 2 \pm 2 . 9 \%$ </td><td> $5 1 . 1 \pm 1 . 4$ </td><td> $5 6 . 8 \pm 2 . 1$ </td><td> $2 9 . 5 \pm 3 . 1 \%$ </td></tr><tr><td>USIMC</td><td> $4 1 . 0 \pm 1 . 8$ </td><td> $4 2 . 6 \pm 0 . 7$ </td><td> $6 0 . 1 \pm 1 . 8 \%$ </td><td> $5 2 . 6 \pm 1 . 7$ </td><td> $5 6 . 9 \pm 1 . 5$ </td><td> $2 5 . 2 \pm 1 . 2 \%$ </td></tr><tr><td>TAUDIS</td><td> $3 7 . 6 \pm 1 . 1$ </td><td> $4 2 . 7 \pm 0 . 6$ </td><td> $4 1 . 4 \pm 1 . 6 \%$ </td><td> $5 1 . 7 \pm 2 . 1$ </td><td> $5 5 . 8 \pm 1 . 4$ </td><td> $3 4 . 6 \pm 1 . 9 \%$ </td></tr><tr><td>Loss-Prediction</td><td> $\underline { { 4 2 . 5 \pm 1 . 2 } }$ </td><td> $4 2 . 9 \pm 0 . 4$ </td><td> $3 6 . 4 \pm 2 . 1 \%$ </td><td> $5 1 . 3 \pm 2 . 0$ </td><td> $5 5 . 9 \pm 1 . 8$ </td><td> $2 8 . 1 \pm 1 . 8 \%$ </td></tr><tr><td>PAAL</td><td> $3 5 . 4 \pm 1 . 9$ </td><td> $4 1 . 9 \pm 0 . 3$ </td><td> $6 7 . 2 \pm 2 . 4 \%$ </td><td> $4 9 . 4 \pm 2 . 2$ </td><td> $5 5 . 0 \pm 2 . 2$ </td><td> $2 6 . 8 \pm 2 . 2 \%$ </td></tr><tr><td>SPARCL (Ours)</td><td> ${ \bf 4 3 . 4 \pm 1 . 3 }$ </td><td> ${ \bf 4 5 . 1 \pm 0 . 8 }$ </td><td> $2 4 . 4 \pm 1 . 4 \%$ </td><td> ${ \bf 5 5 . 1 \pm 1 . 4 }$ </td><td> ${ \bf 6 0 . 6 \pm 1 . 7 }$ </td><td> $2 1 . 2 \pm 2 . 1 \%$ </td></tr></table>

TABLE II

SPATIAL RELIABILITY AT A FIXED 20% ANNOTATION BUDGET ON THE IN-HOUSE CAL CIFICATION DATASET, GLOBAL DICE IS THE POOLED MICRO-DICE (×100) OVER THE TEST SET; MEAN BIN DICE IS THE MEAN POOLED DICE OVER THE 7×7 GRID BINS, RANDOM HELD-OUT SPLIT; ANAT-ZONE WORST IS THE LOWEST POOLED DICE OVER THE SEVEN LUNG ZONES (FIG. 7); DEAD-BIN% IS THE SHARE OF GT-BEARING 7×7 BINS WITH ZERO POOLED OVERLAP. SI IS THE SPATIAL INCONSISTENCY OF EQ. (6) (↓) AND PC THE PERFORMANCE COVERAGE (↑ ). MEAN ± STD OVER FIVE SHARED SEEDS; TIES ARE BROKEN BY THE LOWER STD. BOLD = BEST IN COLUMN, UNDERLINE = SECOND BEST.
<table><tr><td>Method</td><td>Global Dice ↑</td><td>Mean Bin Dice ↑</td><td>Dead-Bin %↓</td><td>Anat-Zone Worst ↑</td><td>SI↓</td><td>PC↑</td></tr><tr><td>Random</td><td> $2 8 . 8 \pm 3 . 1$ </td><td> $2 2 . 0 \pm 2 . 1$ </td><td> $2 3 . 0 \pm 3 . 3$ </td><td> $1 1 . 3 \pm 3 . 3$ </td><td> $0 . 9 3 1 \pm 0 . 1 8 1$ </td><td> $0 . 1 3 8 \pm 0 . 0 3 9$ </td></tr><tr><td>Area-Random</td><td> $2 9 . 2 \pm 2 . 8$ </td><td> $2 7 . 3 \pm 3 . 1$ </td><td> $1 7 . 6 \pm 2 . 0$ </td><td> $1 0 . 6 \pm 2 . 7$ </td><td> $0 . 8 4 7 \pm 0 . 1 0 9$ </td><td> $0 . 1 3 7 \pm 0 . 0 4 2$ </td></tr><tr><td>Uncertainty</td><td> $3 9 . 8 \pm 1 . 1$ </td><td> $3 1 . 0 \pm 2 . 0$ </td><td> $1 6 . 0 \pm 1 . 2$ </td><td> $1 6 . 6 \pm 2 . 9$ </td><td> $0 . 7 5 0 \pm 0 . 0 4 5$ </td><td> $0 . 1 1 1 \pm 0 . 0 3 5$ </td></tr><tr><td>Area-Uncertainty</td><td> $\overline { { 3 8 . 1 \pm 1 . 2 } }$ </td><td> $3 3 . 7 \pm 2 . 4$ </td><td> $1 2 . 0 \pm 1 . 5$ </td><td> $\overline { { 1 6 . 4 \pm 3 . 3 } }$ </td><td> $0 . 8 7 9 \pm 0 . 1 1 9$ </td><td> $0 . 1 3 7 \pm 0 . 0 2 2$ </td></tr><tr><td>Coreset</td><td> $3 6 . 9 \pm 1 . 3$ </td><td> $2 9 . 4 \pm 2 . 3$ </td><td> $1 6 . 1 \pm 2 . 5$ </td><td> $1 4 . 5 \pm 2 . 9$ </td><td> $0 . 7 5 1 \pm 0 . 0 2 3$ </td><td> $0 . 0 9 8 \pm 0 . 0 0 9$ </td></tr><tr><td>Diversity</td><td> $3 1 . 2 \pm 3 . 1$ </td><td> $2 8 . 7 \pm 3 . 4$ </td><td> $1 8 . 8 \pm 3 . 2$ </td><td> $1 5 . 7 \pm 4 . 6$ </td><td> $\underline { { 0 . 6 5 5 } } \pm 0 . 0 7 1$ </td><td> $0 . 1 3 3 \pm 0 . 0 4 2$ </td></tr><tr><td>USIMC</td><td> $3 8 . 9 \pm 2 . 4$ </td><td> $2 9 . 0 \pm 3 . 3$ </td><td> $1 9 . 4 \pm 1 . 9$ </td><td> $1 5 . 0 \pm 3 . 8$ </td><td> $0 . 7 9 6 \pm 0 . 1 0 8$ </td><td> $0 . 1 2 8 \pm 0 . 0 5 0$ </td></tr><tr><td>TAUDIS</td><td> $2 7 . 5 \pm 3 . 6$ </td><td> $2 5 . 0 \pm 3 . 0$ </td><td> $1 0 . 8 \pm 2 . 9$ </td><td> $1 2 . 5 \pm 2 . 5$ </td><td> $0 . 7 3 5 \pm 0 . 0 8 2$ </td><td> $0 . 1 2 8 \pm 0 . 0 3 4$ </td></tr><tr><td>Loss-Prediction</td><td> $3 1 . 2 \pm 2 . 9$ </td><td> $2 4 . 5 \pm 4 . 0$ </td><td> $2 3 . 0 \pm 2 . 7$ </td><td> $1 4 . 5 \pm 4 . 2$ </td><td> $0 . 9 4 2 \pm 0 . 1 3 7$ </td><td> $\underline { { 0 . 1 4 6 \pm 0 . 0 1 3 } }$ </td></tr><tr><td>PAAL</td><td> $3 0 . 7 \pm 1 . 2$ </td><td> $2 4 . 3 \pm 2 . 5$ </td><td> $2 0 . 0 \pm 1 . 1$ </td><td> $6 . 6 \pm 3 . 7$ </td><td> $0 . 9 5 0 \pm 0 . 1 7 2$ </td><td> $0 . 1 2 0 \pm 0 . 0 4 7$ </td></tr><tr><td>SPARCL (Ours)</td><td> ${ \bf 4 0 . 5 \pm 1 . 1 }$ </td><td> ${ \bf 3 6 . 1 } \pm { \bf 2 . 3 }$ </td><td> ${ \bf 9 . 3 \pm 1 . 2 }$ </td><td> ${ \bf 1 9 . 2 \pm 3 . 0 }$ </td><td> ${ \bf 0 . 5 7 1 \pm 0 . 0 3 8 }$ </td><td> ${ \bf 0 . 1 4 8 \pm 0 . 0 0 9 }$ </td></tr></table>

$$
\Delta ( x | S ) = U ( x ) + \lambda ^ { ( t ) } \sum _ { m = 1 } ^ { M } \left[ \psi ( H _ { m } ( S ) + q _ { m } ( x ) ) - \psi ( H _ { m } ( S ) ) \right]\tag{3}
$$

where $\begin{array} { r } { H _ { m } ( S ) = H _ { 0 , m } + \sum _ { x ^ { \prime } \in S } q _ { m } ( x ^ { \prime } ) } \end{array}$ is the coverage mass in bin m contributed by the labeled set (Eq. (1)) and by the images already selected in this round. Both terms are bounded: $U ( x ) \in [ 0 , 1 ]$ after normalization and, since each q(x) sums to one and ψ is concave with $\begin{array} { r } { \psi ( 0 ) = 0 , \sum _ { m } [ \psi ( H _ { m } + q _ { m } ( x ) ) - } \end{array}$ $\begin{array} { r } { \psi ( H _ { m } ) ] \leq \sum _ { m } \psi ( q _ { m } ( x ) ) \leq \sqrt { M } } \end{array}$ , with equality only for an empty coverage state; the gain shrinks as bins fill, so $\lambda _ { \mathrm { m a x } }$ sets the relative weight of the two terms on a fixed scale. Equation (3) is the marginal gain of the set function

$$
A ( \boldsymbol { S } ) = \underbrace { \sum _ { \boldsymbol { x } \in \boldsymbol { S } } \boldsymbol { U } ( \boldsymbol { x } ) } _ { \mathrm { u n c e r t a i n t y ~ ( m o d u l a r ) } } + \lambda ^ { ( t ) } \underbrace { \sum _ { m = 1 } ^ { M } \Big [ \psi \big ( H _ { m } ( \boldsymbol { S } ) \big ) - \psi \big ( H _ { 0 , m } \big ) \Big ] } _ { \mathrm { c o v e r a g e ~ } F ( \boldsymbol { S } ) } ,
$$

## C. Theoretical Guarantees

Casting acquisition as maximizing the set function A of Eq. (4) yields a standard approximation guarantee for the greedy schedule; proofs are in Appendix I.

(4)

so that $\Delta ( x \mid S ) = A ( S \cup \{ x \} ) - A ( S )$ , and each round selects a batch from the candidate set $\mathcal { U } _ { t } ^ { \prime }$ by greedily maximizing A subject to $| S | \le B$ (Algorithm 1). Sec. III-C shows this greedy rule inherits a constant-factor optimality guarantee.

Proposition 1 (Monotone submodularity). Let $\psi : \mathbb { R } _ { \geq 0 } \to \mathbb { R } _ { \geq 0 }$ be concave and non-decreasing with $\psi ( 0 ) = 0 ,$ let $q _ { m } ( x ) \geq 0 ,$ and let $H _ { 0 , m } \geq 0$ be fixed. Then the coverage term $F ( S ) =$ $\begin{array} { r } { \sum _ { m = 1 } ^ { M } \left[ \psi \left( H _ { 0 , m } + \sum _ { x \in S } q _ { m } ( x ) \right) - \psi \left( H _ { 0 , m } \right) \right] } \end{array}$ is monotone and submodular, and for any gate $\lambda ^ { ( t ) } \geq 0$ the acquisition value A(S) of $E q .$ (4) is monotone submodular with $A ( \emptyset ) = 0 .$

Corollary 1 (Greedy near-optimality). As A is monotone submodular, normalized, and non-negative, the greedy rule of Algorithm 1 selecting a batch of size B returns $ { s } _ { \mathrm { g r e e d y } }$ with $A ( S _ { \mathrm { g r e e d y } } ) \geq ( 1 - 1 / e ) A ( S ^ { \star } ) $ , where $S ^ { \star }$ is the optimal size-B subset of the candidate set $\mathcal { U } _ { t } ^ { \prime }$ from which the batch is drawn [44]. Since $H _ { 0 }$ and $\lambda ^ { ( t ) }$ are fixed within a round, the bound applies round by round.

The guarantee concerns the acquisition objective A, not downstream Dice: it certifies that the linear-time greedy rule nearoptimally solves an otherwise combinatorial batch-selection problem.

![](images/c371c36fb0e9bbd0ed25f2f398c0f42e72127c3bc91e9a127bf6556d1494a09f.jpg)  
Fig. 3. Acquisition behavior and its spatial consequences (in-house; rows: Uncertainty vs. SPARCL; columns: rounds 5/20/30). (a) Queries over the pool lesion density (•: earlier rounds; ⋆: this round) beside the 7×7 per-bin test Dice grid (red→green as Dice 0 → 1). The upper-lung-zone bins (2nd–3rd rows), where pool lesions are densest, remain “blind spots” under Uncertainty (Dice 0.76–0.86 at round 30), whereas SPARCL raises them to 0.89–0.93. (b) The red dashed circle marks a lesion-free region in the right upper lung zone of a test case: at round 30 Uncertainty places a false positive there (Dice 0.739, SI 0.338) while SPARCL leaves it empty (Dice 0.956, SI 0.066).

Remark 1 (Regularized coverage removes the empty-bin singularity). The default gain $\psi ( z ) ~ = ~ \sqrt { z }$ has $\psi ^ { \prime } ( 0 ^ { + } ) =$ +∞. Every gain actually evaluated is nevertheless bounded, $\psi ( H + q ) - \psi ( H ) \leq \psi ( q ) \leq 1$ since $q _ { m } ( x ) \leq 1$ , and the ε-power gain $\psi _ { \varepsilon } ( z ) = ( z + \varepsilon ) ^ { p } - \varepsilon ^ { p } , ~ p \in ( 0 , 1 )$ , removes the singularity itself $( \psi _ { \varepsilon } ^ { \prime } ( 0 ) = p \varepsilon ^ { p - 1 } < \infty )$ while remaining concave, non-decreasing, and $\psi _ { \varepsilon } ( 0 ) = 0 ;$ it therefore leaves Proposition 1 and Corollary 1 intact and recovers $\sqrt { z }$ as $\varepsilon \to 0$ (Appendix I).

## D. Spatial Reliability Metrics

We evaluate performance across the same M bins $\lbrace \Omega _ { m } \rbrace _ { m = 1 } ^ { M }$ with a per-image, active-bin, unweighted formulation (naive per-bin Dice is undefined in empty regions). SI measures crossregion disparity at a round and PC the breadth of improvement between rounds; both are reported alongside Dice, which they complement rather than replace.

1) Per-bin Dice and active bins.: For image x and bin $m ,$ let $P _ { m } ( x )$ and $G _ { m } ( x )$ denote the predicted and ground-truth foreground pixel sets, and define the per-bin Dice

$$
D _ { m } ( x ) = \frac { 2 | P _ { m } ( x ) \cap G _ { m } ( x ) | } { | P _ { m } ( x ) | + | G _ { m } ( x ) | } .\tag{5}
$$

The active set is $A ( x ) = \{ m : | P _ { m } ( x ) | + | G _ { m } ( x ) | > 0 \}$ This excludes only the truly empty bins $( | G _ { m } | = | P _ { m } | = 0$ where Dice is undefined) while including false-positive bins $( | G _ { m } | = 0 , | P _ { m } | > 0 \Rightarrow D _ { m } = 0 )$ and missed-lesion bins $( | G _ { m } | > 0 , | P _ { m } | = 0 \Rightarrow D _ { m } = 0 )$ . Hence scattered false positives in lesion-free regions correctly penalise the metric rather than being discarded.

2) Spatial Inconsistency (SI).: Let D(x) = ${ \mathrm { m e a n } } _ { m \in A ( x ) } D _ { m } ( x )$ . SI is the active-bin coefficient of variation:

$$
\operatorname { S I } ( x ) = \left\{ \begin{array} { l l } { \ln / { \mathsf { a } } , } & { | { \mathcal { A } } ( x ) | < 2 , } \\ { 0 , } & { | { \mathcal { A } } ( x ) | \geq 2 , ~ \overline { { D } } ( x ) = 0 , } \\ { \frac { \operatorname { s t d } _ { m \in { \mathcal { A } } ( x ) } D _ { m } ( x ) } { \overline { { D } } ( x ) } , } & { | { \mathcal { A } } ( x ) | \geq 2 , ~ \overline { { D } } ( x ) > 0 . } \end{array} \right.\tag{6}
$$

SI is deliberately unweighted to prevent high-mass regions from dominating and obscuring rare-region blind spots. Because SI = 0 for both uniformly strong and weak predictions, we always report the triple  overall Dice, D, SI.

3) Performance Coverage $( P C ) .$ .: To quantify broad improvement between rounds, we average a concave gain $( \psi ( z ) = { \sqrt { z } } )$ over ground-truth active bins $( | G _ { m } | > 0 )$

$$
\begin{array} { r l r } {  { \operatorname { P C } ( t ) = \frac { 1 } { | \boldsymbol { A } | } \sum _ { m \in \boldsymbol { A } } \psi \big ( \Delta D _ { m } ( t ) \big ) , } } \\ & { } & { \Delta D _ { m } ( t ) = \operatorname* { m a x } \big ( D _ { m } ^ { \mathrm { p o s t } } ( t ) - D _ { m } ^ { \mathrm { p r e } } ( t ) , 0 \big ) , } \end{array}\tag{7}
$$

The concave gain structurally rewards uniform spatial improvements over isolated local gains. Although ψ resembles the acquisition gain, PC evaluates held-out per-bin Dice improvement, entirely independent of the pool uncertainty mass optimized by SPARCL.

![](images/424a5e4e71711228b4b40d3578214ed04d715642366258e5b9273faaa5525016.jpg)  
Fig. 4. Qualitative detection comparison on VinDr-CXR. Ground-truth boxes (green) and each method’s predictions (red; confidence > 0.5, top-5 shown). On the same cases, SPARCL localises the dispersed abnormalities more tightly and produces fewer spurious boxes than other baselines.  
TABLE III

LABEL EFFICIENCY AND NATIVE OBJECT-DETECTION PERFORMANCE ON PUBLIC DETECTION BENCHMARKS, MEAN ± STD OVER FIVE SHARED RANDOM SEEDS. $\mathsf { M A P ^ { \dagger } } , \mathsf { A P _ { 5 0 } } ^ { \dagger }$ , AND FROC<sup>†</sup> ARE AT A FIXED 20% BUDGET; AULC IS THE NORMALIZED AREA UNDER THE MAP-VS-BUDGET CURVE; BUDGET@95% IS THE MINIMUM BUDGET TO REACH 95% OF FULL-DATA MAP (VINDR-CXR 9.5 ± 0.1 MAP / 25.8 ± 0.2 AP ; CHESTX-DET10 13.2 ± 0.4 MAP / 36.5 ± 0.3 AP AT 100% BUDGET). BOLD = BEST IN COLUMN, UNDERLINE = SECOND BEST IN COLUMN.
<table><tr><td></td><td colspan="5">VinDr-CXR</td><td colspan="5">ChestX-Det10</td></tr><tr><td>Method</td><td> $\mathbf { m } \mathbf { A } \mathbf { P } ^ { \dagger } \uparrow$ </td><td> ${ \bf A P 5 0 } ^ { \dagger } \uparrow$ </td><td> $\mathbf { F R O C } ^ { \dagger } \uparrow$ </td><td> $\mathbf { A U L C \uparrow }$ </td><td>Budget@95%↓</td><td> $\mathbf { m } \mathbf { A } \mathbf { P } ^ { \dagger } \uparrow$ </td><td> ${ \bf A P 5 0 } ^ { \dagger } \uparrow$ </td><td> $\mathbf { F R O C } ^ { \dagger } \uparrow$ </td><td>AULC↑</td><td>Budget@95%↓</td></tr><tr><td>Random</td><td> $7 . 1 \pm 0 . 3$ </td><td> $1 9 . 9 \pm 0 . 6$ </td><td> $2 5 . 0 \pm 0 . 5$ </td><td> $9 . 3 \pm 0 . 3$ </td><td> $4 9 \pm 8 \%$ </td><td> $8 . 5 \pm 0 . 4$ </td><td> $2 5 . 8 \pm 0 . 5$ </td><td> $\underline { { 2 6 . 5 \pm 0 . 3 } }$ </td><td> $1 0 . 5 \pm 0 . 2$ </td><td> $7 5 \pm 4 \%$ </td></tr><tr><td>Area-Random</td><td> $\underline { { 7 . 3 \pm 0 . 2 } }$ </td><td> $2 0 . 7 \pm 0 . 2$ </td><td> $2 5 . 7 \pm 0 . 7$ </td><td> $9 . 6 \pm 0 . 1$ </td><td> $4 2 \pm 4 \%$ </td><td> $8 . 2 \pm 0 . 2$ </td><td> $2 6 . 1 \pm 0 . 2$ </td><td> $2 6 . 4 \pm 0 . 2$ </td><td> $9 . 5 \pm 0 . 1$ </td><td> $8 2 \pm 1 \%$ </td></tr><tr><td>Uncertainty</td><td> $6 . 7 \pm 0 . 4$ </td><td> $1 7 . 5 \pm 1 . 1$ </td><td> $2 2 . 9 \pm 0 . 5$ </td><td> $8 . 6 \pm 0 . 1$ </td><td> $4 8 \pm 2 \%$ </td><td> $7 . 6 \pm 0 . 3$ </td><td> $2 4 . 4 \pm 0 . 6$ </td><td> $2 5 . 4 \pm 0 . 5$ </td><td> $9 . 9 \pm 0 . 1$ </td><td> $89 \pm 4 \%$ </td></tr><tr><td>Area-Uncertainty</td><td> $7 . 1 \pm 0 . 1$ </td><td> $2 0 . 4 \pm 1 . 0$ </td><td> $2 4 . 8 \pm 0 . 7$ </td><td> $9 . 3 \pm 0 . 3$ </td><td> $4 7 \pm 6 \%$ </td><td> $7 . 3 \pm 0 . 3$ </td><td> $2 3 . 3 \pm 0 . 6$ </td><td> $2 4 . 2 \pm 0 . 5$ </td><td> $1 0 . 0 \pm 0 . 2$ </td><td> $8 0 \pm 2 \%$ </td></tr><tr><td>Coreset</td><td> $6 . 9 \pm 0 . 3$ </td><td> $1 9 . 7 \pm 0 . 7$ </td><td> $2 5 . 5 \pm 0 . 9$ </td><td> $9 . 5 \pm 0 . 3$ </td><td> $4 5 \pm 4 \%$ </td><td> $\underline { { 8 . 6 \pm 0 . 2 } }$ </td><td> $2 5 . 7 \pm 0 . 3$ </td><td> $2 6 . 2 \pm 0 . 2$ </td><td> $\underline { { 1 0 . 6 \pm 0 . 1 } }$ </td><td> $7 3 \pm 3 \%$ </td></tr><tr><td>Diversity</td><td> $6 . 9 \pm 0 . 2$ </td><td> $1 9 . 1 \pm 0 . 9$ </td><td> $2 5 . 1 \pm 1 . 0$ </td><td> $9 . 1 \pm 0 . 1$ </td><td> $4 3 \pm 2 \%$ </td><td> $8 . 5 \pm 0 . 1$ </td><td> $2 6 . 0 \pm 0 . 9$ </td><td> $2 5 . 7 \pm 0 . 7$ </td><td> $9 . 6 \pm 0 . 2$ </td><td> $8 1 \pm 1 \%$ </td></tr><tr><td>TAUDIS</td><td> $6 . 4 \pm 0 . 2$ </td><td> $1 8 . 4 \pm 0 . 7$ </td><td> $2 3 . 7 \pm 0 . 9$ </td><td> $9 . 3 \pm 0 . 3$ </td><td> $4 6 \pm 2 \%$ </td><td> $8 . 5 \pm 0 . 3$ </td><td> $2 5 . 2 \pm 0 . 2$ </td><td> $2 5 . 9 \pm 0 . 3$ </td><td> $1 0 . 4 \pm 0 . 1$ </td><td> $7 9 \pm 2 \%$ </td></tr><tr><td>USIMC</td><td> $6 . 4 \pm 0 . 2$ </td><td> $1 8 . 4 \pm 1 . 3$ </td><td> $2 4 . 3 \pm 1 . 5$ </td><td> $8 . 8 \pm 0 . 2$ </td><td> $5 8 \pm 3 \%$ </td><td> $7 . 7 \pm 0 . 3$ </td><td> $2 4 . 3 \pm 0 . 6$ </td><td> $2 5 . 1 \pm 0 . 3$ </td><td> $1 0 . 0 \pm 0 . 2$ </td><td> $8 1 \pm 3 \%$ </td></tr><tr><td>SPARCL (Ours)</td><td> ${ \bf 7 . 7 \pm 0 . 1 }$  </td><td> ${ \bf 2 1 . 7 \pm 0 . 8 }$  </td><td> ${ \bf 2 7 . 1 \pm 0 . 8 }$  </td><td> ${ \bf 9 . 8 \pm 0 . 2 }$  </td><td> $4 0 \pm 3 \%$ </td><td> ${ \bf 8 . 9 \pm 0 . 1 }$  </td><td> ${ \bf 2 6 . 2 \pm 0 . 5 }$  </td><td> ${ \bf 2 6 . 6 \pm 0 . 4 }$  </td><td> ${ \bf 1 0 . 7 \pm 0 . 1 }$  </td><td> $7 1 \pm 2 \%$  </td></tr></table>

## IV. EXPERIMENTS

## A. Datasets and Preprocessing

To validate adaptability across distinct spatial regimes, we evaluate SPARCL on four datasets: (1) In-house Calcification (Seg., Mixed): 2.8k private CXRs with two expert radiologists annotations for Calcified Nodules (900, diffuse) and Aortic Calcification (900, focal). (2) VinDr-CXR [40] (Det., Diffuse): 15k CXRs with bounding boxes for Pulmonary Nodules and other scattered abnormalities. (3) SIIM-ACR [41] (Seg., Focal): 10.7k CXRs for Pneumothorax targeting the pleural boundary. (4) ChestX-Det10 [42] (Det., Mixed): A 3.5k instance-level subset of NIH ChestX-ray14 covering 10 disease categories.

To probe generalization beyond chest radiography, we additionally use (5) DDR [46] (Seg., Diffuse): color fundus photographs for microaneurysm segmentation, a strongly diffuse regime with tiny, scattered retinal lesions. DDR is used only for the cross-modality generalization study (Table IV), not for the main CXR comparisons.

1) Training and AL Protocol: All experiments are implemented in PyTorch on a single NVIDIA V100 (32 GB)

## B. Implementation Details

GPU. To isolate the effect of the acquisition rule, all AL strategies share identical architectures (U-Net [2] with an EfficientNet-B4 encoder and scSE decoder trained with a Dice+BCE loss for segmentation; Faster R-CNN with a ResNet-50 backbone trained with focal and smooth-L<sub>1</sub> losses for detection), optimizers (AdamW, LR=3e-4), augmentations, 150 epochs (batch size 16), and early-stopping (patience 10, $\delta = 0 . 0 0 1 )$ criteria. In each AL round, we expand the labeled pool by 1% of the total dataset. Default SPARCL parameters are a $7 \times 7$ grid $( K = L = 7 ) , \lambda _ { \operatorname* { m a x } } = 1 . 0$ , and candidate ratio $r = 5 ;$ all three were selected on the validation split of the in-house cohort. Every method starts from the same initial labeled set and acquires the exact same number of samples per round. Results are reported as mean ± standard deviation over five shared random seeds using patient-level splits to avoid data leakage.

2) Evaluation Split Protocols.: The held-out set is drawn in one of three modes, using ground-truth masks only to assign images to bins: Random (patient-level, ignores lesion location, follows natural spatial prevalence); Spatial-Equal (each of the $K \times L$ bins contributes $N _ { \mathrm { t e s t } } / ( K \times L )$ lesion-bearing images, so every region is weighted equally regardless of prevalence); and

Random  
Uncertainty  
Coreset  
SPARCL (ours)  
![](images/0259885b850f861df30d004b4d3513cd1344e02a5d44d9e295e1988ffee3aea7.jpg)  
Fig. 5. Modality generalization to color fundus (DDR, microaneurysms). Rows: five test cases; columns: acquisition methods. Ground truth in green, predictions in blue, each true-positive overlap boxed in red. SPARCL recovers more of the tiny scattered lesions (more red boxes, higher per-image Dice) than Random, Uncertainty, and Coreset.

Spatial-Dynamic (bin quotas $N _ { \mathrm { t e s t } } { \cdot } \vert \Omega _ { m } \vert / \sum _ { j } \vert \Omega _ { j } \vert$ proportional to lesion prevalence).

3) Generalization to Object Detection.: To adapt SPARCL for detection, we reformulate the spatial histogram using predicted boxes $\begin{array} { r } { \mathcal { B } _ { x } \colon h _ { m } ( x ) \ : = \ : \sum _ { b \in \mathcal { B } _ { x } } s _ { b } \mathcal { H } ( s _ { b } ) \cdot \mathrm { I o U } ( b , \mathrm { b i n } _ { m } ) } \end{array}$ , where $s _ { b }$ is the confidence score and H is binary entropy. The image-level uncertainty is correspondingly the aggregate box uncertainty $\begin{array} { r } { U ( x ) = \sum _ { b \in { \mathcal { B } } _ { x } } s _ { b } { \mathcal { H } } ( s _ { b } ) } \end{array}$ , which replaces the mean pixel entropy used for segmentation. This $s _ { b } \mathcal { H } ( s _ { b } )$ weighting filters background noise by prioritizing object-like (high s<sub>b</sub>) yet uncertain regions. The histogram is normalized to q(x) exactly as for segmentation; for the cumulative coverage state, $g _ { m } ( x )$ is the fraction of ground-truth box area of a labeled image that falls in bin m. The candidate set, gate, coverage state, and greedy step are otherwise unchanged.

## C. Baselines

We compare SPARCL against 10 strategies across four categories. Random: (1) Random (uniform pool selection) and (2) Area-Random (spatially-stratified). Uncertainty: (3) Uncertainty [17] (standard pixel/box entropy) and (4) Area-Uncertainty (entropy with spatial stratification). Diversity: (5) Coreset [16] (global k-center feature coverage over the entire labeled set) and (6) Diversity (intra-batch farthest-point feature sampling). Hybrid / learned: (7) TAUDIS [39] (uncertainty filtering + diversity), (8) USIMC [37] (uncertainty-aware submodular mutual information), (9) Loss-Prediction [38] (a mod ule predicts each sample’s target loss), and (10) PAAL [12] (an auxiliary network predicts each sample’s segmentation accuracy and the lowest-predicted-accuracy samples are queried). Note that PAAL and Loss-Prediction are designed specifically for segmentation and are therefore applied only to the segmentation benchmarks.

TABLE IV  
MODALITY GENERALIZATION TO COLOR FUNDUS PHOTOGRAPHY (DDR, MICROANEURYSM SEGMENTATION, THREE SHARED SEEDS), SHOWING THE CXR PIPELINE TRANSFERS TO A DIFFERENT IMAGING MODALITY. DICE@20% AND AULC USE THE ANNOTATION-BUDGET FRACTION; FINAL DICE IS AT THE LAST AL ROUND.
<table><tr><td>Method</td><td>Dice@20%↑</td><td> $\mathbf { A U L C \uparrow }$ </td><td>Final Dice ↑</td></tr><tr><td>Random</td><td> $2 1 . 0 \pm 0 . 2 9$ </td><td> $2 5 . 9 \pm 0 . 1 3$ </td><td> $3 1 . 3 \pm 0 . 2 1$ </td></tr><tr><td>Area-Random</td><td> $2 3 . 4 \pm 0 . 1 3$ </td><td> $2 6 . 3 \pm 0 . 0 7$ </td><td> $3 1 . 5 \pm 0 . 1 4$ </td></tr><tr><td>Uncertainty</td><td> $2 1 . 4 \pm 0 . 1 2$ </td><td> $2 6 . 4 \pm 0 . 0 5$ </td><td> $3 1 . 6 \pm 0 . 1 0$ </td></tr><tr><td>Coreset</td><td> $\underline { { 2 2 . 9 \pm 0 . 2 1 } }$ </td><td> $2 6 . 9 \pm 0 . 1 3$ </td><td> ${ \underline { { 3 2 . 2 \pm 0 . 1 1 } } }$ </td></tr><tr><td>Diversity</td><td> $\overline { { 1 5 . 4 \pm 0 . 2 6 } }$ </td><td> $2 4 . 2 \pm 0 . 2 1$ </td><td> $\overline { { 2 9 . 4 \pm 0 . 1 2 } }$ </td></tr><tr><td>TAUDIS</td><td> $1 6 . 5 \pm 0 . 1 7$ </td><td> $2 2 . 0 \pm 0 . 1 5$ </td><td> $3 0 . 5 \pm 0 . 1 3$ </td></tr><tr><td>Loss-Prediction</td><td> $1 4 . 9 \pm 0 . 1 7$ </td><td> $2 2 . 9 \pm 0 . 0 9$ </td><td> $2 9 . 1 \pm 0 . 1 0$ </td></tr><tr><td>SPARCL (Ours)</td><td> ${ \bf 2 6 . 2 \pm 0 . 2 3 }$ </td><td> ${ \bf 2 9 . 5 \pm 0 . 1 1 }$ </td><td> ${ \bf 3 2 . 6 \pm 0 . 1 5 }$ </td></tr></table>

## D. Quantitative Analysis

1) Label Efficiency and Spatial Reliability: At a restricted 20% annotation budget, SPARCL consistently outperforms all baselines. On the in-house dataset (Table I), it achieves a Dice score of 43.4%, surpassing Random (37.3%) and Uncertainty sampling (41.2%). Moreover, Uncertainty sampling generates false positives in local blind spots, exhibiting limited spatial stability (Fig. 3b). Regarding spatial safety (Table II), SPARCL maintains the lowest Spatial Inconsistency $( S I = 0 . 5 7 1 )$ and the highest Performance Coverage $( P C = 0 . 1 4 8 )$ at a 20% budget. By consistently yielding the highest Anat-Zone Worst Dice (19.2) and the lowest Dead-Bin rate (9.3%), SPARCL prevents local overfitting and substantially reduces anatomical blind spots (Dead-Bin 9.3% vs. 16.0% for Uncertainty).

2) Generalization Across Diverse Patterns: Table I and Table III summarize efficiency across diverse public datasets. SPARCL requires less data to reach 95% of fully supervised performance (Budget@95%). It reaches this milestone at only a 24% budget on the in-house cohort and 21% on the focal SIIM-ACR dataset. This extends to the highly dispersed VinDr-CXR (40% budget) and instance-level ChestX-Det10 (71% budget) detection tasks, i.e., across lesion distributions and task types.

3) Object Detection.: With the box-based histogram of Sec. IV, a Faster R-CNN on VinDr-CXR and ChestX-Det10 (Table III) attains the best mAP (7.7% and 8.9%), $\mathsf { A P } _ { 5 0 }$ (21.7% and 26.2%), and FROC (27.1% and 26.6%) at a 20% budget under SPARCL. Fig. 4 shows it localizes dispersed findings more tightly and with fewer false positives than baselines.

4) Modality Generalization to Fundus.: Beyond CXR, the same pipeline transfers to color fundus photography (DDR) [46] for microaneurysm segmentation, a strongly diffuse regime with tiny retinal lesions. Table IV and Fig. 5 show that SPARCL recovers more scattered lesions than baselines, and reaches highest dice score, indicating cross-modality adaptability.

![](images/b962bc6887bafd5b651f389546602dd88cd675d37bb316a8e10dc925928da89a.jpg)  
Fig. 6. Qualitative segmentation on the in-house calcified-nodule cohort. Ground-truth contours in green, predictions in blue, and true-positive overlaps boxed in red, shown with per-image Dice. Baselines often miss these scattered calcifications (Dice ≈ 0) while SPARCL recovers them.

## E. Qualitative Analysis

Fig. 3 visualizes acquisition: Uncertainty sampling clusters queries around high-contrast edges and leaves large anatomical areas undersampled, producing “anatomical blind spots” where the model later emits false positives (red circles), whereas SPARCL’s cumulative coverage spreads queries over the lung field and the corresponding false positives disappear. Fig. 6 shows the same effect on the in-house calcified-nodule cohort, where baselines frequently miss the small, scattered lesions (Dice ≈ 0) that SPARCL recovers.

## F. Ablation Study

1) Component Ablation: Table V isolates the three components of SPARCL on the in-house cohort. (A) Resetting $H _ { 0 }$ every round, i.e. within-batch coverage only, costs about two points of Budget@95% (24.4%→26.5%) and 2.2 points of Dice@20%, and dropping U(x) costs over four (28.7%, Dice 37.1): coverage helps only when it redirects the budget among uncertain images and remembers where lesions have already been annotated. (B) Because uncertainty on this mixed cohort spans many bins, $\lambda ^ { ( t ) }$ stays near $\lambda _ { \mathrm { m a x } }$ and removing the gate changes all three metrics by less than one seed standard deviation, as intended (Sec. III-B.5); the gate is a safeguard rather than a source of gain here. Halving or doubling $\lambda _ { \mathrm { m a x } }$ worsens Budget@95% to 25.2% and 25.5%. (C) Without the floor, $r { = } 1$ reduces to uncertainty sampling (29.3%, SI 0.750) and “full” is worse still (30.2%, SI 0.771), since coverage then admits images whose uncertainty in empty bins is spurious.

2) Robustness to Grid Resolution and Evaluation Protocols: Table VI evaluates sensitivity to grid size (M) and evaluation split. Finer bins $( \mathbf { e } . \mathbf { g } . , 8 \times 8 )$ degrade performance through overfragmentation, whereas the $7 \times 7$ grid gives the best overall Dice (0.463) and Mean Bin Dice (0.465) at the lowest budget (24%). Under the Spatial-Equal and Spatial-Dynamic splits SPARCL retains the lowest SI and highest $P C ,$ , and under Spatial-Equal it reaches 95% performance at a 20% budget versus 29% for USIMC and 31% for Uncertainty.

3) Robustness to the Bin Definition: Grid vs. Anatomy.: A fixed $7 \times 7$ cell need not correspond to the same anatomy across patients. To confirm that SPARCL does not hinge on this registration-free choice, we replace the grid with seven

## TABLE V

ABLATION OF SPARCL COMPONENTS AND HYPERPARAMETERS (IN-HOUSE CALCIFICATION, $7 \times 7 \mathsf { G R I D } ,$ MEAN ± STD OVER FIVE   
SHARED SEEDS). (A) COVERAGE STATE: CUMULATIVE $\scriptstyle { H _ { 0 } }$ (EQ. (1)) VS. RESETTING THE STATE EVERY ROUND, AND COVERAGE WITHOUT THE UNCERTAINTY TERM. (B) GATE: REMOVING IT $( \lambda \equiv \lambda _ { \operatorname* { m a x } } )$ AND VARYING THE CEILING $\lambda _ { \mathbf { m a x } }$ . (C) UNCERTAINTY FLOOR: r=1 IS   
UNCERTAINTY SAMPLING (THE BATCH IS THE TOP-B BY U(x)); “FULL” RUNS THE GREEDY STEP OVER THE WHOLE POOL.

<table><tr><td>Configuration</td><td></td><td>|Dice@20% ↑ | Budget@95%↓ |</td><td>SI ↓</td></tr><tr><td colspan="4">(A) Coverage state</td></tr><tr><td>SPARCL (cumulative  $H _ { 0 } , \mathrm { g a t e } , r { = } 5 )$ </td><td> $4 3 . 4 \pm 1 . 3$ </td><td> $2 4 . 4 \pm 1 . 4 \%$ </td><td> $0 . 5 7 1 \pm 0 . 0 3 8$ </td></tr><tr><td>reset  $( H _ { 0 } \equiv 0 ;$  within-batch coverage only)</td><td> $4 1 . 2 \pm 1 . 4$ </td><td> $2 6 . 5 \pm 1 . 6 \%$ </td><td> $0 . 5 9 2 \pm 0 . 0 1 2$ </td></tr><tr><td>coverage only (U(x) removed)</td><td> $3 7 . 1 \pm 1 . 6$ </td><td> $2 8 . 7 \pm 1 . 5 \%$ </td><td> $0 . 6 2 4 \pm 0 . 0 2 6$ </td></tr><tr><td colspan="4">(B) Positional-diversity gate</td></tr><tr><td> $\mathrm { w / o ~ g a t e } ~ ( \lambda \equiv \lambda _ { \operatorname* { m a x } } )$ </td><td> $4 3 . 1 \pm 1 . 5$ </td><td> $2 5 . 1 \pm 1 . 9 \%$ </td><td> $0 . 5 9 9 \pm 0 . 0 3 4$ </td></tr><tr><td> $\lambda _ { \mathrm { m a x } } = 0 . 5$ </td><td> $4 3 . 1 \pm 0 . 9$ </td><td> $2 5 . 2 \pm 1 . 2 \%$ </td><td> $0 . 5 8 3 \pm 0 . 0 1 8$ </td></tr><tr><td> $\lambda _ { \mathrm { m a x } } = 2 . 0$ </td><td> $4 2 . 9 \pm 1 . 1$ </td><td> $2 5 . 5 \pm 1 . 1 \%$ </td><td> $0 . 5 9 2 \pm 0 . 0 2 1$ </td></tr><tr><td colspan="4">(C) Uncertainty floor</td></tr><tr><td>r = 1 (= uncertainty sampling)</td><td> $4 1 . 2 \pm 1 . 1$ </td><td> $2 9 . 3 \pm 0 . 9 \%$ </td><td> $0 . 7 5 0 \pm 0 . 0 4 5$ </td></tr><tr><td>full (no floor)</td><td> $3 9 . 1 \pm 0 . 8$ </td><td> $3 0 . 2 \pm 1 . 3 \%$ </td><td> $0 . 7 7 1 \pm 0 . 0 2 1$ </td></tr></table>

per-image lung-anatomy zones obtained from an off-the-shelf lung segmentation (Fig. 7) and compare both partitions under an identical protocol (Table VII). The three are statistically indistinguishable in label efficiency and spatial reliability: the grid matches the anatomy-aware zones without an additional segmentation dependency, so the reported spatial-reliability gains are not grid artifacts and the method remains applicable where anatomical parcellation is unavailable.

4) Robustness to the Concave Gain ψ.: The coverage gain rewards under-represented bins under any concave, nondecreasing ψ with $\psi ( 0 ) = 0 ;$ Table VIII compares three members of this family under an identical protocol. Replacing $\sqrt { z }$ with log(1 + z) or $z ^ { 1 / 3 }$ shifts Budget@95% by less than two points and leaves Dice@20%, $S I ,$ and $P C$ within one seed standard deviation. SPARCL is therefore not sensitive to the specific concave form; we adopt $\sqrt { z }$ because it gives the best budget and Dice among the three and its steeper slope near the origin most strongly rewards covering empty bins in early rounds.

5) Hyperparameter Sensitivity.: SPARCL exposes two hyperparameters beyond the grid, the gate ceiling $\lambda _ { \mathrm { m a x } }$ and the candidate ratio r; both are swept in Table V (B)/(C). r=1

Uniform grid (7×7=49 bins)

Anatomy zones (7 bins)

Chest X-ray  
Anatomy grid, finer (13 bins)  
![](images/5bc1495e9564827bf52d448286f61d8e8c125f328694ad638c9782727ba11267.jpg)  
Fig. 7. Uniform grid vs. registration-free anatomy bins. Rows: two patients; columns: chest X-ray, fixed 7×7 grid, seven lung-anatomy zones (apex/mid/base per lung + central), and a finer 13-bin within-lung subdivision. Anatomy bins come per image from lung segmentation, so a given bin denotes the same anatomical region across patients despite body habitus, rotation, or view.

TABLE VI  
ABLATION STUDY ON EVALUATION SPLIT STRATEGIES AND GRID SIZE M USING THE In-house Calcification DATASET. DICE, SI, PC, AND MEAN BIN DICE ARE MEASURED AT THE FINAL AL ROUND AND AVERAGED OVER 5 SEEDS. SPARCL CONSISTENTLY MAINTAINS SUPERIOR SPATIAL CONSISTENCY AND LABEL EFFICIENCY ACROSS ALL SETTINGS.
<table><tr><td colspan="2">Ablation Category</td><td rowspan="2">Method</td><td rowspan="2">Budget @95%(↓)</td><td rowspan="2">Dice(↑)</td><td rowspan="2">SI(↓)</td></tr><tr><td>Split Strategy</td><td> $\overrightarrow { \mathbf { G r i d ~ S i z e } ( K \times L ) }$ </td></tr><tr><td rowspan="4">Random</td><td> $\overline { { { M = 6 } \mathrm { ~ ( 3 ~ \times ~ 2 ) ~ } } }$ </td><td>SPARCL</td><td>37%</td><td>0.433</td><td>0.421</td></tr><tr><td> $M = 2 5 \ ( 5 \times 5 )$ </td><td>SPARCL</td><td>28%</td><td>0.447</td><td>0.493</td></tr><tr><td> $M = 4 9 \mathrm { ~ ( 7 ~ times ~ 7 ~ ) ~ }$ </td><td>SPARCL</td><td>24%</td><td>0.463</td><td>0.414</td></tr><tr><td> $M = 6 4 ~ ( 8 \times 8 )$ </td><td>SPARCL</td><td>24%</td><td>0.443</td><td>0.455</td></tr><tr><td rowspan="4">Spatial Equal</td><td rowspan="4"> $M = 4 9 \mathrm { ~ ( 7 ~ times ~ 7 ~ ) ~ }$ </td><td>Random</td><td>34%</td><td>0.445</td><td>0.511</td></tr><tr><td>Uncertainty</td><td>31%</td><td>0.455</td><td>0.486</td></tr><tr><td>USIMC [37]</td><td>29%</td><td>0.444</td><td>0.520</td></tr><tr><td>SPARCL</td><td>20%</td><td>0.466</td><td>0.409</td></tr><tr><td rowspan="4">Spatial Dynamic</td><td rowspan="4"> $M = 4 9 \mathrm { ~ ( 7 ~ times ~ 7 ~ ) ~ }$ </td><td>Random</td><td>39%</td><td>0.429</td><td>0.516</td></tr><tr><td>Uncertainty</td><td>34%</td><td>0.432</td><td>0.505</td></tr><tr><td>USIMC [37]</td><td>28%</td><td>0.431</td><td>0.514</td></tr><tr><td>SPARCL</td><td>24%</td><td>0.464</td><td>0.481</td></tr></table>

and “full” bracket the design—uncertainty sampling on one side, unrestricted cumulative coverage on the other. We fix $\lambda _ { \operatorname* { m a x } } = 1 . 0$ and $r = 5$ throughout.

## V. CONCLUSION

In this paper, we proposed SPARCL, an active learning framework that combines predictive uncertainty with spatial coverage accumulated over the lesions already labeled, gated off only when the pool offers no positional diversity to exploit. SPARCL reaches 95% of fully supervised performance using only a 24% annotation budget on our in-house calcification cohort and 21% on the SIIM-ACR benchmark. Evaluated through our proposed Spatial Inconsistency and Performance Coverage metrics, SPARCL effectively reduces anatomical blind spots that aggregate metrics obscure. Future work will extend this spatially adaptive mechanism to 3D volumetric modalities (e.g., 3D CT) to resolve spatial information loss.

## VI. DATA AVAILABILITY

The in-house calcification dataset is private and cannot be released publicly; it may be made available from the authors upon reasonable request. The public datasets are openly available: VinDr-CXR [40], SIIM-ACR [41], ChestX-Det10 [42], and DDR [46].

## TABLE VII

BIN DEFINITION: UNIFORM GRID VS. REGISTRATION-FREE ANATOMY ZONES (IN-HOUSE, 20% BUDGET). COMPARES THE STANDARD 7×7 GRID AGAINST ANATOMICAL REGIONS (FIG. 7). THE SIMPLE GRID MAINTAINS COMPETITIVE SPATIAL RELIABILITY WITHOUT REQUIRING AN EXTERNAL LUNG SEGMENTER. ACQUISITION USES THE PARTITION IN EACH ROW; SI AND PC ARE EVALUATED ON THE FIXED 7×7 GRID FOR ALL ROWS.
<table><tr><td>Bin scheme</td><td>Dice@20%↑</td><td>SI↓</td><td>PC↑</td></tr><tr><td>Uniform grid (7×7)</td><td> $4 3 . 4 \pm 1 . 3$ </td><td> $0 . 5 7 1 \pm 0 . 0 3 8$ </td><td> $0 . 1 4 8 \pm 0 . 0 0 9$ </td></tr><tr><td>Anatomy zones (7 lung zones)</td><td> $4 1 . 8 \pm 3 . 1$ </td><td> $0 . 6 3 0 \pm 0 . 0 4 1$ </td><td> $0 . 1 4 7 \pm 0 . 0 0 8$ </td></tr><tr><td>Anatomy zones (13 within-lung bins)</td><td> $4 2 . 4 \pm 2 . 2$ </td><td> $0 . 5 9 7 \pm 0 . 0 4 0$ </td><td> $0 . 1 3 9 \pm 0 . 0 1 3$ </td></tr></table>

TABLE VIII

CONCAVE-GAIN (ψ) ABLATION (IN-HOUSE CALCIFICATION, RANDOM HELD-OUT SPLIT, $7 \times 7 \mathsf { G R I D } ,$ MEAN ± STD OVER 5 SEEDS). RESULTS ARE STABLE ACROSS CONCAVE FUNCTIONS, CONFIRMING THE ROBUSTNESS OF THE DIMINISHING-RETURNS COVERAGE REWARD.
<table><tr><td></td><td>Gain ψ(z) | Budget@95% ↓ | Dice@20% ↑ | SI ↓</td><td></td><td></td><td> PC ↑</td></tr><tr><td> $\log ( 1 + z )$ </td><td> $2 6 . 1 \pm 2 . 1 \%$ </td><td> $4 2 . 8 \pm 1 . 9$ </td><td> $0 . 6 1 3 \pm 0 . 0 8 0$ </td><td> $0 . 1 3 1 \pm 0 . 0 4 3$ </td></tr><tr><td> $\sqrt { z } \ \mathrm { ( o u r s ) }$ </td><td> $2 4 . 4 \pm 1 . 4 \%$ </td><td> $4 3 . 4 \pm 1 . 3$ </td><td> $0 . 5 7 1 \pm 0 . 0 3 8$ </td><td> $0 . 1 4 8 \pm 0 . 0 0 9$ </td></tr><tr><td> $\dot { z } ^ { 1 / 3 }$ </td><td> $2 5 . 8 \pm 1 . 5 \%$ </td><td> $4 2 . 3 \pm { 1 . 8 }$ </td><td> $0 . 6 0 8 \pm 0 . 0 5 1$ </td><td> $0 . 1 4 0 \pm 0 . 0 1 5$ </td></tr></table>

## APPENDIX I PROOFS OF SECTION III-C

Lemma 1 (concave over modular). Let $g : 2 ^ { V } \to { \mathbb { R } } _ { \geq 0 }$ be non-negative modular, $\begin{array} { r } { g ( S ) = \sum _ { x \in \mathcal { S } } w _ { x } } \end{array}$ with $w _ { x } \geq 0 ,$ and $\phi : \mathbb { R } _ { \ge 0 }  \mathbb { R }$ concave and non-decreasing. Then ${ \cal S } \mapsto \phi ( g ( { \cal S } ) )$ is monotone and submodular.

Proof. Monotonicity: for $\mathcal { A } \subseteq B , g ( \mathcal { A } ) \leq g ( \mathcal { B } )$ (non-negative weights) and $\phi$ non-decreasing give $\phi ( g ( \mathcal { A } ) ) ~ \le ~ \phi ( g ( \mathcal { B } ) )$ Submodularity: for ${ \mathcal { A } } \subseteq B , x \notin B .$ , write $a = g ( \mathcal { A } ) \leq b = g ( \mathcal { B } )$ and $w = w _ { x } \ge 0$ . Since ϕ is concave, $z \mapsto \phi ( z + w ) - \phi ( z )$ is non-increasing, so $\phi ( a + w ) - \phi ( a ) \geq \phi ( b + w ) - \phi ( b )$ . □ Proposition 1. Fix m and set $\phi _ { m } ( z ) = \psi ( H _ { 0 , m } + z ) - \psi ( H _ { 0 , m } )$ for $z \geq 0 ;$ since $\psi$ is concave and non-decreasing and $H _ { 0 , m } ~ \geq ~ 0$ is a constant, $\phi _ { m }$ is concave, non-decreasing, and $\phi _ { m } ( 0 ) ~ = ~ 0 .$ . The map $\begin{array} { r } { S \mapsto \sum _ { x \in S } q _ { m } ( x ) } \end{array}$ is nonnegative modular $( q _ { m } ( x ) \ge 0 )$ , so by Lemma 1 each $s \mapsto$ $\begin{array} { r } { \phi _ { m } \big ( \sum _ { x \in \mathcal { S } } q _ { m } ( x ) \big ) \ = \ \psi ( H _ { m } ( \mathcal { S } ) ) \ - \ \psi ( H _ { 0 , m } ) } \end{array}$ is monotone submodular, and a finite sum of monotone submodular functions is monotone submodular, so $F$ is monotone submodular. The uncertainty term $\textstyle \sum _ { x \in S } U ( x )$ is modular with non-negative coefficients $U ( x ) \geq 0$ (min–max normalized entropy), hence monotone submodular. As $\begin{array} { r } { \lambda ^ { ( t ) } \ge 0 , A = \sum _ { x } U ( x ) + \lambda ^ { ( t ) } F } \end{array}$ is a non-negative combination of monotone submodular functions and is therefore monotone submodular, with $A ( \emptyset ) =$ $\begin{array} { r } { \lambda ^ { ( t ) } \sum _ { m } \phi _ { m } ( 0 ) = 0 } \end{array}$ □ Corollary 1. A is monotone submodular, normalized $( A ( \emptyset ) =$ 0). For maximization under the cardinality constraint $| S | \leq b ,$ the greedy algorithm achieves a $( 1 - 1 / e )$ approximation of the optimum [44]. □ Remark 1. For $\psi _ { \varepsilon } ( z ) = ( z + \varepsilon ) ^ { p } - \varepsilon ^ { p }$ with $p \in ( 0 , 1 ) , \varepsilon > 0 \colon$ $\psi _ { \varepsilon } ^ { \prime } ( z ) = p ( z + \varepsilon ) ^ { p - 1 } > 0$ (non-decreasing) and $\psi _ { \varepsilon } ^ { \prime \prime } ( z ) =$ $p ( p - 1 ) ( z + \varepsilon ) ^ { p - 2 } < 0$ (concave); $\psi _ { \varepsilon } ( 0 ) = \varepsilon ^ { p } - \varepsilon ^ { p } = 0$ and $\psi _ { \varepsilon } ^ { \prime } ( 0 ) = p \varepsilon ^ { p - 1 } < \infty$ . Hence $\psi _ { \varepsilon }$ satisfies the hypotheses of Proposition 1, so F and A built from $\psi _ { \varepsilon }$ remain monotone submodular and Corollary 1 holds; and $\psi _ { \varepsilon } ( z )  z ^ { p }$ pointwise as $\varepsilon  0$ , recovering $\sqrt { z }$ at $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$ □

## REFERENCES

[1] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2015, pp. 3431–3440.

[2] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2015, pp. 234–241.

[3] X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, “ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 2097–2106.

[4] J. Irvin et al., “CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison,” in Proc. AAAI Conf. Artif. Intell., vol. 33, no. 1, 2019, pp. 590–597.

[5] S. Budd, E. C. Robinson, and B. Kainz, “A survey on active learning and human-in-the-loop deep learning for medical image analysis,” Med. Image Anal., vol. 71, 2021, Art. no. 102062.

[6] V. Nath, D. Yang, B. A. Landman, D. Xu, and H. R. Roth, “Diminishing uncertainty within the training pool: Active learning for medical image segmentation,” IEEE Trans. Med. Imag., vol. 40, no. 10, pp. 2534–2547, 2020.

[7] A. Smailagic et al., “MedAL: Accurate and robust deep active learning for medical image analysis,” in Proc. IEEE Int. Conf. Mach. Learn. Appl. (ICMLA), 2018, pp. 481–488.

[8] L. Cai, X. Xu, L. Zhang, and C.-S. Foo, “Exploring spatial diversity for region-based active learning,” IEEE Trans. Image Process., vol. 30, pp. 8702–8712, 2021.

[9] C. Nguyen et al., “GOAL: Gist-set online active learning for efficient chest X-ray image annotation,” in Proc. Med. Imag. Deep Learn. (MIDL), 2021, pp. 545–553.

[10] B. Settles, “Active learning literature survey,” Dept. Comput. Sci., Univ. Wisconsin–Madison, Madison, WI, USA, Comput. Sci. Tech. Rep. 1648, 2009.

[11] P. Ren et al., “A survey of deep active learning,” ACM Comput. Surv., vol. 54, no. 9, pp. 1–40, 2021.

[12] J. Shi et al., “Predictive accuracy-based active learning for medical image segmentation,” 2024, arXiv:2405.00452.

[13] C. Gao et al., “Deep learning in pulmonary nodule detection and segmentation: A systematic review,” Eur. Radiol., vol. 35, no. 1, pp. 255–266, 2025.

[14] I. Marinakis, K. Karampidis, and G. Papadourakis, “Pulmonary nodule detection, segmentation and classification using deep learning: A comprehensive literature review,” BioMedInformatics, vol. 4, no. 3, pp. 2043–2106, 2024.

[15] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,” in Proc. Int. Conf. Mach. Learn. (ICML), 2016, pp. 1050–1059.

[16] O. Sener and S. Savarese, “Active learning for convolutional neural networks: A core-set approach,” 2017, arXiv:1708.00489.

[17] L. Yang, Y. Zhang, J. Chen, S. Zhang, and D. Z. Chen, “Suggestive annotation: A deep active learning framework for biomedical image segmentation,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2017, pp. 399–407.

[18] M. Gorriz, A. Carlier, E. Faure, and X. Giro-i-Nieto, “Cost-effective active learning for melanoma segmentation,” 2017, arXiv:1711.09168.

[19] A. Kirsch, J. Van Amersfoort, and Y. Gal, “BatchBALD: Efficient and diverse batch acquisition for deep Bayesian active learning,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 32, 2019.

[20] J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford, and A. Agarwal, “Deep batch active learning by diverse, uncertain gradient lower bounds,” 2019, arXiv:1906.03671.

[21] Y. Siddiqui, J. Valentin, and M. Nießner, “ViewAL: Active learning with viewpoint entropy for semantic segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 9430–9440

[22] R. Mackowiak, P. Lenz, O. Ghori, F. Diego, O. Lange, and C. Rother, “CEREALS – Cost-effective region-based active learning for semantic segmentation,” 2018, arXiv:1810.09726.

[23] K. Wang, D. Zhang, Y. Li, R. Zhang, and L. Lin, “Cost-effective active learning for deep image classification,” IEEE Trans. Circuits Syst. Video Technol., vol. 27, no. 12, pp. 2591–2600, 2016.

[24] Y. Gal, R. Islam, and Z. Ghahramani, “Deep Bayesian active learning with image data,” in Proc. Int. Conf. Mach. Learn. (ICML), 2017, pp. 1183–1192.

[25] M. A. Kadir, H. M. T. Alam, and D. Sonntag, “EdgeAL: An edge estimation based active learning approach for OCT segmentation,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2023, pp. 79–89.

[26] Y. Tian, M. Shi, Y. Luo, A. Kouhana, T. Elze, and M. Wang, “FairSeg: A large-scale medical image segmentation dataset for fairness learning using segment anything model with fair error-bound scaling,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2024, pp. 44056–44070.

[27] A. A. Taha and A. Hanbury, “Metrics for evaluating 3D medical image segmentation: Analysis, selection, and tool,” BMC Med. Imag., vol. 15, no. 1, 2015, Art. no. 29.

[28] C. H. Sudre, W. Li, T. Vercauteren, S. Ourselin, and M. Jorge Cardoso, “Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations,” in Proc. Int. Workshop Deep Learn. Med. Image Anal. (DLMIA), 2017, pp. 240–248.

[29] A. Reinke et al., “Common limitations of image processing metrics: A picture story,” 2021, arXiv:2104.05642.

[30] M. Gaillochet, C. Desrosiers, and H. Lombaert, “Active learning for medical image segmentation with stochastic batches,” Med. Image Anal., vol. 90, 2023, Art. no. 102958.

[31] G. Hacohen, A. Dekel, and D. Weinshall, “Active learning on a budget: Opposite strategies suit high and low budgets,” in Proc. Int. Conf. Mach. Learn. (ICML), 2022, pp. 8175–8195.

[32] O. Yehuda, A. Dekel, G. Hacohen, and D. Weinshall, “Active learning through a covering lens,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 35, 2022, pp. 22354–22367.

[33] B. Xie, L. Yuan, S. Li, C. H. Liu, and X. Cheng, “Towards fewer annotations: Active learning via region impurity and prediction uncertainty for domain adaptive semantic segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 8058–8068.

[34] G. Shin, W. Xie, and S. Albanie, “All you need are a few pixels: Semantic segmentation with PixelPick,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW), 2021, pp. 1687–1697.

[35] H. Lin and J. Bilmes, “A class of submodular functions for document summarization,” in Proc. 49th Annu. Meeting Assoc. Comput. Linguistics: Human Lang. Technol. (ACL-HLT), 2011, pp. 510–520.

[36] L. Maier-Hein et al., “Metrics reloaded: Recommendations for image analysis validation,” Nat. Methods, vol. 21, no. 2, pp. 195–212, 2024.

[37] B. Föllmer, K. Schulze, C. Wald, S. Stober, W. Samek, and M. Dewey, “Active learning with the nnU-Net and sample selection with uncertainty aware submodular mutual information measure,” in Proc. Med. Imag. Deep Learn. (MIDL), 2024.

[38] D. Yoo and I. S. Kweon, “Learning loss for active learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 93–102.

[39] K. Yu et al., “Two-step active learning for instance segmentation with uncertainty and diversity sampling,” 2023, arXiv:2309.16139.

[40] H. Q. Nguyen et al., “VinDr-CXR: An open dataset of chest X-rays with radiologist’s annotations,” Sci. Data, vol. 9, no. 1, 2022, Art. no. 429.

[41] R. W. Filice et al., “Crowdsourcing pneumothorax annotations using machine learning annotations on the NIH chest X-ray dataset,” J. Digit. Imag., vol. 33, no. 2, pp. 490–496, 2020.

[42] J. Liu, J. Lian, and Y. Yu, “ChestX-Det10: Chest X-ray dataset on detection of thoracic abnormalities,” 2020, arXiv:2006.10550.

[43] J. Lian et al., “A structure-aware relation network for thoracic diseases detection and segmentation,” IEEE Trans. Med. Imag., vol. 40, no. 8, pp. 2042–2052, 2021.

[44] G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher, “An analysis of approximations for maximizing submodular set functions—I,” Math. Program., vol. 14, no. 1, pp. 265–294, 1978.

[45] Y. Ma, R. Garnett, and J. Schneider, “Submodularity in batch active learning and survey problems on Gaussian random fields,” 2012, arXiv:1209.3694.

[46] T. Li, Y. Gao, K. Wang, S. Guo, H. Liu, and H. Kang, “Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening,” Inf. Sci., vol. 501, pp. 511–522, 2019.