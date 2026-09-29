# REDUCE, THEN ENCODE: MULTISCALE VOLUMETRICREDUCTION FOR 2D FOUNDATION MODELS IN BRAINMRI

Dexuan Ding Macquarie University

Yuankai Qi <sup>∗</sup> Macquarie University {yuankai.qi}@mq.edu.au

Bogong Wang Australian National University

Luping Zhou University of Sydney

Amin Beheshti Macquarie University

## ABSTRACT

Pretrained 2D foundation models offer a practical alternative to dedicated 3D pretraining for brain structural magnetic resonance imaging (sMRI), but their use on volumetric data requires bridging the mismatch between a 2D encoder and a 3D volume input. Existing methods typically encode slices independently and integrate their features afterwards. We introduce Multiscale Volumetric Reduction (MVR), a reduce-then-encode approach that compresses each anatomical view from D slices into M ≪ D complementary 2D components before foundationmodel encoding. MVR combines an uncentered-PCA base component derived from the original through-plane intensities with residual detail components constructed from multiscale spatial descriptors. The reduction is estimated from the training volumes without diagnostic labels or gradient-based optimization and remains fixed thereafter. The resulting components are independently processed by a shared frozen 2D foundation model and concatenated for linear probing. Under this frozen-encoder setting, MVR achieves strong overall performance across ADNI, OASIS, and ABIDE relative to the evaluated 2D-to-3D adaptation methods and simple input-reduction baselines, while also generalizing strongly from ADNI to AIBL.

## 1 INTRODUCTION

Brain structural magnetic resonance imaging (sMRI) provides a non-invasive view of brain anatomy and is widely used to study structural alterations associated with neurological and neurodevelopmental disorders (Tak et al., 2026; van Rooij et al., 2018). Unlike ordinary 2D images, an sMRI scan is represented as a three-dimensional volume, where anatomical and disease-related variations may extend across multiple slices and spatial scales. Effective analysis therefore requires preserving informative volumetric structure while extracting features useful for downstream prediction.

A direct solution is to learn representations from 3D medical volumes themselves. Recent medical foundation models such as BrainIAC (Tak et al., 2026), BrainMVP (Rui et al., 2025), and 3DINO (Xu et al., 2025) demonstrate the effectiveness of large-scale volumetric pretraining, but require dedicated 3D architectures and substantial medical imaging data (Fig. 1(a)). An attractive alternative is to reuse general-purpose 2D vision foundation models pretrained on large and diverse image collections. Although their pretraining domain differs substantially from medical imaging, recent studies have shown that their representations can still transfer effectively to volumetric medical imaging (Rafsani et al., 2025). RAPTOR (An et al., 2025), for example, constructs volumetric embeddings using a frozen 2D foundation model (DINOv2 (Oquab et al., 2024)), while AnyMC3D (Liu et al., 2026) further improves transfer through lightweight adaptation and learned aggregation. These results suggest that general-purpose 2D foundation models can provide useful representations for medical volumes without requiring dedicated volumetric pretraining.

![](images/14793663ce1041eaa6b56b7bb2a1c3c13127f6bc51dffc2484e45ce9d0ff07de.jpg)  
Figure 1: Comparison of volumetric representation strategies for brain MRI. (a) Medical foundation models learn volumetric representations through dedicated 3D pretraining. (b) Approaches based on pretrained 2D foundation models encode D selected 2D slices independently and integrate the resulting features after encoding, with method-specific adaptation where applicable. (c) Our MVR instead performs reduction in the input space, transforming each volume into $M \ll D$ dense 2D components before encoding with a shared frozen 2D foundation model. The resulting component features are concatenated to form the volume representation. $\delta$ and $\frac { 3 4 } { x }$ denote trainable and frozen modules.

However, a remaining challenge is the mismatch between a 2D encoder and a 3D volume. A 2D foundation model processes one two-dimensional input at a time (Fig. 1(b)), whereas an MRI scan contains a spatially ordered sequence of slices. Existing approaches commonly address this mismatch through an encode-then-integrate strategy: slices are first encoded independently, after which their features are pooled, projected, attended to, or otherwise aggregated into a volume-level repre sentation. RAPTOR (An et al., 2025) reduces independently extracted slice features through pooling and random projection, while AnyMC3D (Liu et al., 2026) combines slice-wise encoding with learned feature aggregation. In this paradigm, cross-slice information is incorporated only afte slice-level feature extraction. When the 2D encoder is frozen, however, its slice-level representations cannot adapt to facilitate subsequent volumetric aggregation. This raises the possibility that preserving cross-slice structure in the encoder inputs themselves may provide a more effective inductive bias than relying exclusively on post-encoding integration. This motivates us to instead integrate volumetric information before encoding.

To this end, we introduce Multiscale Volumetric Reduction (MVR), a reduce-then-encode approach that transforms each anatomical view from D slices into $M \ll D$ dense 2D components before foundation-model encoding (Fig. 1(c)). The key challenge is to achieve substantial through-plane reduction without collapsing useful volumetric variation. Simple averaging can suppress localized structure, while a single linear projection provides only one summary of the through-plane intensity pattern. MVR therefore adopts a base-and-detail construction. A base component projects the original through-plane intensities onto a shared direction estimated by uncentered $\mathrm { P C A } ,$ providing a compact reference derived from the original volume. To retain complementary information beyond this single projection, MVR constructs multiscale descriptors from differences between Gaussian-smoothed slices, incorporating in-plane context at multiple spatial scales while preserving the complete through-plane sequence. After removing variation linearly associated with the base component, PCA of the residual descriptors produces $\bar { M } - 1$ detail components capturing dominant complementary variation. The reduction operators are data-adaptively estimated from the training volumes without diagnostic labels or gradient-based optimization and remain fixed thereafter. Each component is independently processed by the same frozen 2D foundation model, and the resulting features are concatenated across components and anatomical views to form the final volumetric representation.

![](images/6a6487229c88701de1fd766535e3f39a30796160d96a4f0e430d39cb99e54fe6.jpg)  
Figure 2: Main architecture of our Multiscale Volumetric Reduction (MVR). (a) Base component extraction summarizes the original through-plane intensities into a dense base component and sampled base scores. (b) Multiscale residual expansion incorporates in-plane variation at multiple scales and produces M − 1 complementary detail components after accounting for the base reference. (c) Encoding and classification applies MVR to the axial, coronal, and sagittal views, independently encodes the resulting components with a shared frozen 2D foundation model, and concatenates their features for classification.

We evaluate MVR on Alzheimer’s disease classification using ADNI and OASIS-3, autism classification using ABIDE, and external ADNI-to-AIBL evaluation. With a frozen DINOv3 encoder and a linear classifier, MVR achieves strong overall performance relative to the evaluated 2D-to-3D adaptation methods and simple input-reduction baselines across ADNI, OASIS, and ABIDE. Ablation studies further show that the multiscale construction and the combination of base and detail components contribute consistently to classification performance.

Our main contributions are threefold:

• We introduce a reduce-then-encode strategy for adapting frozen 2D foundation models to volumetric brain MRI, incorporating information across slices before rather than only after 2D feature extraction.

• We develop MVR, a data-adaptive and gradient-free base-and-detail reduction that converts D slices into M ≪ D complementary 2D components by combining through-plane reduction with multiscale in-plane context.

• We demonstrate the effectiveness of MVR across multiple brain-MRI cohorts and diagnostic tasks, including external ADNI-to-AIBL evaluation, and show through ablation that its multiscale construction and base-detail decomposition consistently contribute to classification performance.

## 2 RELATED WORK

Medical foundation models. Medical pretraining has been used to learn transferable representations for volumetric imaging. BrainIAC learns generalizable representations from unlabeled brain MRI (Tak et al., 2026), while BrainMVP uses multi-parametric MRI with reconstruction, contrastive learning, and distillation (Rui et al., 2025). 3DINO extends self-supervised pretraining to large-scale multi-organ 3D medical data (Xu et al., 2025). These approaches obtain volumetric representations through pretraining directly on 3D medical images. In contrast, we consider the complementary setting in which a general-purpose pretrained 2D foundation model is kept frozen, and ask how volumetric information should be represented for effective 2D encoding.

Adapting 2D models to medical volumes. Pretrained 2D foundation models can be extended to volumetric data by aggregating features extracted from individual cross-sections. RAPTOR reduces frozen 2D features through pooling and random projection (An et al., 2025), while AnyMC3D combines lightweight encoder adaptation with learned aggregation across slices and views (Liu et al., 2026). Both perform volumetric integration after 2D feature extraction. MVR instead integrates cross-slice information in the input space before encoding. Other related methods, such as Eigenslices (Jonemo & Eklund, 2023), reduce 3D brain MRI to a small set of 2D projections, but ¨ rely on a trainable 2D CNN to adapt to the projected representation.

## 3 METHOD

## 3.1 OVERVIEW

Let I denote a normalized brain sMRI volume. We denote the axial, coronal, and sagittal views by $\mathcal { V } = \{ a , c , s \}$ . For each anatomical view $v \in \mathcal V$ , we use $I _ { v } \in \mathbb { R } ^ { H \times \dot { W } \times D }$ for the reoriented volume, where H and W denote the in-plane dimensions and D denotes the number of slices.

Given $I _ { v } , \mathbf { M V R }$ constructs $M \ll D$ dense 2D components before feature extraction by the pretrained 2D foundation model. A base component summarizes the original through-plane intensities using uncentered PCA (Sec. 3.2), while $M - 1$ detail components capture complementary multiscale spatial variation after removing variation associated with the base reference (Sec. 3.3). The resulting components are encoded independently by a shared frozen 2D foundation model, and their features are concatenated across components and anatomical views to form the final volume representation for downstream classification (Sec. 3.4).

## 3.2 BASE COMPONENT EXTRACTION

As illustrated in Fig. 2(a), MVR first constructs a base component that summarizes the original intensities across the slice sequence. At each in-plane location $( h , w )$ , the through-plane vector $\mathbf { x } _ { v } ( h , w ) \in \mathbb { R } ^ { D }$ contains the intensities $I _ { v } ( h , w , d )$ in slice order, for $d = 1 , \dotsc , \bar { D }$ . A shared projection (detailed below) combines these D entries into one base-component value while retaining its corresponding in-plane location.

We fit this projection using through-plane vectors sampled from the training volumes. From each of the S training subjects, we uniformly sample P vectors containing at least one intensity value different from the volume minimum. Let $\bar { N } \stackrel { - } { = } S P$ denote the total number of sampled vectors and $\mathbf { x } _ { i } ~ \in ~ \mathbb { R } ^ { D }$ denote the i-th sampled vector. Stacking them as rows gives the training matrix $X _ { v } = [ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times D }$

The projection vector ${ \mathbf { u } } _ { b } \in \mathbb { R } ^ { D }$ is then obtained by applying uncentered PCA to $X _ { v }$ . Specifically, $\mathbf { u } _ { b }$ is the unit eigenvector corresponding to the largest eigenvalue of the second-moment matrix $\bar { M _ { b } } = X _ { v } ^ { \top } X _ { v } / \bar { N } \in \mathbb R ^ { D \times D }$ . Unlike centered $\mathrm { P C A }$ , uncentered PCA retains the average intensity structure across slice positions as part of the projection. The selected direction therefore captures the dominant through-plane intensity pattern in the sampled vectors.

Once obtained, $\mathbf { u } _ { b }$ remains fixed and serves two purposes. Applying it to the sampled vectors gives the base scores used for residual fitting, $( b _ { 1 } , \ldots , \bar { b _ { N } } ) ^ { \top } \ = \ \bar { X } _ { v } \mathbf { u } _ { b } \in \ \mathbb { R } ^ { N \times 1 }$ , where $b _ { i } = \mathbf { x } _ { i } ^ { \top } \mathbf { u } _ { b }$ Applying the same projection at every in-plane location of a volume gives

$$
b _ { v } ( h , w ) = \mathbf { u } _ { b } ^ { \top } \mathbf { x } _ { v } ( h , w ) = \sum _ { d = 1 } ^ { D } u _ { b , d } I _ { v } ( h , w , d ) ,\tag{1}
$$

where $u _ { b , d }$ is the weight assigned to slice d. The projected values retain their original in-plane locations, forming the base component $b _ { v } \in \mathbb { R } ^ { H \times W }$ . Additional derivation details for the base projection are provided in Appendix A.1.1.

## 3.3 MULTISCALE RESIDUAL EXPANSION

The base component provides a compact reference to the original through-plane intensities, but a single projected value cannot preserve all D-dimensional intensity patterns. Simply retaining additional projections of the original through-plane vectors would still describe only the values observed at the same in-plane location across slices. Brain MRI analysis, however, often benefits from modeling local spatial context and structural boundaries across neighboring locations (Chen et al., 2023). We therefore construct the detail components from multiscale spatial descriptors before reducing across slices. To keep these components complementary to the base reference, we remove multiscale variation that is linearly associated with the sampled base scores before retaining the dominant residual variation. Below we provide the details and Fig. 2(b) summarizes this construction.

Multiscale decomposition. To incorporate surrounding in-plane context, we construct components that describe intensity variation over neighborhoods of different sizes. Let $\mathcal { G } _ { \sigma }$ denote Gaussian smoothing with standard deviation $\sigma > 0$ , applied independently to each 2D slice. We choose K ordered smoothing scales $0 < \sigma _ { 1 } < \cdots < \sigma _ { K }$ . Let $L _ { v , k }$ denote the smoothed volume at scale $\sigma _ { k }$ with $L _ { v , 0 } = I _ { v }$ , and let $B _ { v , k }$ denote the difference between successive smoothing levels. These volumes retain the dimensions $H \times W \times D$ and are constructed as

![](images/1e912abfb98c6bb4870619bab600e86fabdec52d688673dbfae7e7dd360cfbd1.jpg)  
Figure 3: Visualization of the multiscale decomposition and resulting MVR components. (a) Gaussian smoothing at increasing scales produces multiscale difference components $B _ { 0 } , \ldots , B _ { 4 }$ , capturing in-plane variation at different spatial scales. (b) The base component $b _ { v }$ summarizes the original through-plane intensities, while the detail components capture complementary residual multiscale variation. The first four detail components are shown.

$$
\begin{array} { l l } { { L _ { v , k } = \mathcal { G } _ { \sigma _ { k } } ( I _ { v } ) , } } & { { k = 1 , \ldots , K , } } \\ { { B _ { v , k } = L _ { v , k } - L _ { v , k + 1 } , } } & { { k = 0 , \ldots , K - 1 . } } \end{array}\tag{2}
$$

These differences describe intensity variation suppressed between successive smoothing levels, providing components sensitive to different in-plane spatial scales while leaving the through-plane coordinates unchanged. We retain all K difference components for detail construction. Fig. 3(a) visualizes the Gaussian smoothing levels and corresponding multiscale difference components for a representative slice.

To prevent difference components with larger responses from dominating the subsequent projection, we normalize each component as $\widetilde { B } _ { v , k } = B _ { v , k } / s _ { k }$ , where $s _ { k }$ is its root mean square (RMS) computed over training voxels whose original intensity differs from the corresponding volume minimum. The RMS factors are estimated separately for each anatomical view and fixed thereafter.

We concatenate the normalized through-plane vectors at each location to form a multiscale descriptor ${ \bf q } _ { v } ( h , w ) \in \mathbb R ^ { K D }$

$$
\begin{array} { r } { \mathbf { q } _ { v } ( h , w ) = \operatorname { C o n c a t } \left( \widetilde { B } _ { v , 0 } ( h , w , \cdot ) , \dots , \widetilde { B } _ { v , K - 1 } ( h , w , \cdot ) \right) , } \end{array}\tag{3}
$$

where each argument contains all D entries in slice order. Each descriptor therefore combines in-plane context at multiple scales with the complete through-plane sequence. For each sampled through-plane vector $\mathbf { x } _ { i }$ in $X _ { v } ,$ , let $\mathbf { q } _ { i } \in \mathbb { R } ^ { K D }$ denote the corresponding multiscale descriptor. The sampled descriptors $\{ \mathbf { q } _ { i } \} _ { i = 1 } ^ { N }$ and base scores $\{ b _ { i } \} _ { i = 1 } ^ { N }$ are used together for residual fitting.

Residual detail extraction. Some multiscale variation may already be linearly associated with the sampled base scores. To remove this contribution, we first center the sampled descriptors and base scores using their training means, $\begin{array} { r } { \pmb { \mu _ { q } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf q _ { i } \in \mathbb { R } ^ { K D } } \end{array}$ and $\begin{array} { r } { \mu _ { b } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } b _ { i } \in \mathbb { R } } \end{array}$ . We then fit a least-squares regression with coefficient vector $\beta \in \mathbb { R } ^ { K D }$ to predict the centered descriptors from the centered base scores:

$$
\begin{array} { r l } & { \beta = \frac { \sum _ { i = 1 } ^ { N } ( b _ { i } - \mu _ { b } ) ( \mathbf { q } _ { i } - \mu _ { q } ) } { \sum _ { i = 1 } ^ { N } ( b _ { i } - \mu _ { b } ) ^ { 2 } } , } \\ & { \mathbf { r } _ { i } = ( \mathbf { q } _ { i } - \pmb { \mu } _ { q } ) - \beta ( b _ { i } - \mu _ { b } ) . } \end{array}\tag{4}
$$

The resulting residual descriptor $\mathbf { r } _ { i } \in \mathbb { R } ^ { K D }$ therefore retains multiscale variation that is not linearly predicted by the base score.

We next apply PCA to the sampled residual descriptors. Stacking them as rows gives $\begin{array} { r l } { R _ { v } } & { { } = } \end{array}$ $\left[ \mathbf { r } _ { 1 } , \ldots , \mathbf { r } _ { N } \right] ^ { \mathsf { T } ^ { \prime } } \in \mathbb { R } ^ { N \times K D }$ , with empirical covariance matrix $C _ { r } = \breve { R _ { v } ^ { \top } } R _ { v } / N \in \mathbb { R } ^ { K D \breve { \times } _ { K D } }$ . We retain the mutually orthogonal unit eigenvectors $\mathbf { w } _ { 1 } , \hdots , \mathbf { w } _ { M - 1 } \in \mathbb { R } ^ { K D }$ corresponding to the $M - 1$ largest eigenvalues of $\bar { C } _ { r }$ , ordered by decreasing eigenvalue. These projection vectors capture the dominant residual multiscale variation in the sampled descriptors.

Once fitted, we construct the residual descriptor $\mathbf { r } _ { v } ( h , w ) \in \mathbb { R } ^ { K D }$ at each in-plane location using the training means and regression coefficient. The detail-component values are then

$$
d _ { v , j } ( h , w ) = \mathbf { w } _ { j } ^ { \top } \mathbf { r } _ { v } ( h , w ) , \qquad j = 1 , \dotsc , M - 1 .\tag{5}
$$

The projected values retain their original in-plane locations, forming the detail components $d _ { v , j } \in$ $\mathbb { R } ^ { H \times W }$ . Fig. 3(b) shows the resulting base component and the first four detail components for the same volume. The mutually orthogonal detail projection vectors therefore retain complementary directions of residual multiscale variation beyond the base reference. A detailed formulation of the residual PCA and detail projections is provided in Appendix A.1.2.

## 3.4 ENCODING AND CLASSIFICATION

Foundation-model encoding. Once fitted on the training volumes, all MVR projection vectors, regression coefficients, and associated statistics are fixed and reused for unseen volumes. The resulting base and detail components are then independently encoded by the shared frozen DINOv3 encoder. Let ϕ denote feature extraction using the shared frozen DINOv3 encoder. Each component is encoded independently, and the resulting features are concatenated within each view and then across the axial, coronal, and sagittal views:

$$
\begin{array} { r l } & { \quad { \bf h } _ { v } = \mathrm { C o n c a t } \left( \phi ( b _ { v } ) , \phi ( d _ { v , 1 } ) , \dots , \phi ( d _ { v , M - 1 } ) \right) , } \\ & { { \bf h } ( I ) = \mathrm { C o n c a t } \left( { \bf h } _ { a } , { \bf h } _ { c } , { \bf h } _ { s } \right) . } \end{array}\tag{6}
$$

The pretrained 2D foundation model weights remain fixed and the resulting ${ \bf h } ( I )$ forms the volume representation used for downstream classification.

Downstream classification. We use a linear classifier to assess the discriminative quality of ${ \bf h } ( I )$ while limiting classifier expressiveness. Each feature coordinate is standardized using its mean and standard deviation estimated from the training subjects, followed by class-balanced, L2-regularized logistic regression.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Datasets. We evaluate MVR on two diagnostic classification tasks using three brain MRI cohorts: ADNI (Petersen et al., 2010) (1,027 subjects), OASIS-3 (LaMontagne et al., 2019) (550 subjects), and ABIDE (Heinsfeld et al., 2018) (1,099 subjects). ADNI and OASIS-3 are used for Alzheimer’s disease (AD) versus cognitively normal (CN) classification, while ABIDE is used for autism spectrum disorder versus control classification. We further evaluate the ADNI-trained models on an external AIBL (Ellis et al., 2009) cohort containing 224 subjects, without AIBL-specific fitting or model selection. All methods receive the same T1-weighted volumes following N4 bias correction (Tustison et al., 2010), skull stripping, affine registration to MNI152 (Fonov et al., 2011), resampling to 1 mm isotropic resolution, and intensity normalization.

All methods use identical subject-level five-fold splits, with each fold held out for testing in turn. All data-dependent reduction and feature-standardization statistics are estimated from the corresponding training partition.

Metrics. We report the area under the receiver operating characteristic curve (AUC) and balanced accuracy (BAcc). AUC measures discrimination across decision thresholds, while BAcc averages sensitivity and specificity, giving equal weight to both classes. Classification thresholds are selected using inner cross-validation on the training data and fixed before testing. Results are reported as mean ± sample standard deviation across the five held-out folds. For AIBL, we report variation across the five ADNI-fold-trained models evaluated on the same external cohort, using their ADNI derived thresholds.

Table 1: Classification performance under the frozen-encoder setting on ADNI, OASIS, and ABIDE. Results are mean ± sample standard deviation across five held-out folds; best and second-best means are highlighted in bold and underlined. <sup>†</sup> denotes the use of frozen DINOv3 encoder.
<table><tr><td rowspan="2">Method</td><td colspan="2">ADNI</td><td colspan="2">OASIS</td><td colspan="2">ABIDE</td></tr><tr><td>AUC↑</td><td>BAcc ↑</td><td>AUC ↑</td><td>BAcc ↑</td><td>AUC↑</td><td>BAcc ↑</td></tr><tr><td>BrainIAC (Tak et al., 2026)</td><td> $7 6 . 9 6 \pm 3 . 9 4$ </td><td> $7 0 . 3 4 \pm 4 . 6 6$ </td><td> $6 3 . 3 1 \pm 9 . 1 5$ </td><td> $5 8 . 2 2 \pm 6 . 1 8$ </td><td> $5 1 . 4 0 \pm 3 . 1 4$ </td><td> $5 1 . 4 8 \pm 1 . 3 5$ </td></tr><tr><td>BrainMVP (Rui et al., 2025)</td><td> $7 4 . 3 8 \pm 2 . 7 0$ </td><td> $6 8 . 6 1 \pm 3 . 6 9$ </td><td> $6 7 . 7 6 \pm 5 . 2 2$ </td><td> $6 4 . 3 1 \pm 2 . 9 5$ </td><td> $5 0 . 4 1 \pm 4 . 1 1$ </td><td> $5 0 . 7 8 \pm 4 . 1 1$ </td></tr><tr><td>3DINO (Xu et al., 2025)</td><td> $8 5 . 9 6 \pm 2 . 1 7$ </td><td> $7 8 . 1 4 \pm 2 . 2 6$ </td><td> $6 7 . 0 5 \pm 2 . 5 8$ </td><td> $6 3 . 0 1 \pm 3 . 3 0$ </td><td> $5 9 . 3 0 \pm 2 . 4 3 $ </td><td> $5 5 . 3 0 \pm 1 . 8 2$ </td></tr><tr><td>MedicalNet (Chen et al., 2019)</td><td> $8 6 . 6 0 \pm 1 . 4 8$ </td><td> $7 8 . 1 0 \pm 2 . 0 0$ </td><td> $6 8 . 3 5 \pm 4 . 9 9$ </td><td> ${ \bf 6 4 . 9 1 \pm 3 . 8 8 }$ </td><td> $6 2 . 3 7 \pm 2 . 6 5$ </td><td> $5 8 . 8 5 \pm 2 . 2 5$ </td></tr><tr><td>RAPTOR (An et al., 2025)</td><td> $8 0 . 2 6 \pm 0 . 9 4$ </td><td> $7 2 . 9 1 \pm 2 . 1 8$ </td><td> $6 4 . 2 6 \pm 5 . 6 7$ </td><td> $6 1 . 4 7 \pm 5 . 2 6$ </td><td> $6 2 . 9 5 \pm 1 . 7 9$ </td><td> $5 7 . 2 7 \pm 2 . 2 2$ </td></tr><tr><td>AnyMC3D† (Liu et al., 2026)</td><td> $8 4 . 4 5 \pm 2 . 9 3$ </td><td> $7 6 . 9 0 \pm 3 . 8 1$ </td><td> $6 6 . 7 6 \pm 6 . 9 0$ </td><td> $6 2 . 2 9 \pm 5 . 4 5$ </td><td> $5 9 . 6 6 \pm 4 . 6 5$ </td><td> $5 5 . 1 6 \pm 2 . 0 5$ </td></tr><tr><td>Eigenslices (Jönemo &amp; Eklund, 2023)</td><td> $8 4 . 9 7 \pm 2 . 2 7$ </td><td> $7 6 . 8 9 \pm 3 . 4 3$ </td><td> $6 8 . 9 3 \pm 2 . 3 5$ </td><td> $6 4 . 0 8 \pm 1 . 6 8$ </td><td> $6 1 . 2 0 \pm 3 . 0 0$ </td><td> $5 8 . 4 6 \pm 3 . 0 9$ </td></tr><tr><td>Mean pooling</td><td> $8 1 . 7 7 \pm 3 . 9 7$ </td><td> $7 1 . 4 8 \pm 4 . 7 6$ </td><td> $6 3 . 4 8 \pm 5 . 6 0$ </td><td> $5 9 . 0 6 \pm 5 . 9 1$ </td><td> $5 5 . 7 1 \pm 3 . 8 4$ </td><td> $5 4 . 0 3 \pm 3 . 3 7$ </td></tr><tr><td>Uniform slice sampling (M = 16)</td><td> $8 0 . 8 7 \pm 2 . 1 4$ </td><td> $7 3 . 0 6 \pm 2 . 1 3$ </td><td> $6 2 . 1 5 \pm 2 . 7 1$ </td><td> $5 7 . 8 9 \pm 2 . 7 7$ </td><td> $6 5 . 9 9 \pm 2 . 8 8$ </td><td> $5 9 . 5 6 \pm 3 . 3 4$ </td></tr><tr><td>Direct projection  $( M = 1 )$ </td><td> $8 0 . 0 8 \pm 1 . 1 5$ </td><td> $7 1 . 7 3 \pm 1 . 3 5$ </td><td> $6 1 . 1 2 \pm 4 . 8 5$ </td><td> $5 7 . 2 1 \pm 3 . 1 2$ </td><td> $5 3 . 9 9 \pm 5 . 0 9$ </td><td> $5 2 . 0 9 \pm 1 . 7 1 $ </td></tr><tr><td>Direct projection  $( M = 1 6 )$ </td><td> $8 2 . 4 9 \pm 1 . 5 6$ </td><td> $7 5 . 7 6 \pm 3 . 3 0$ </td><td> $5 9 . 2 5 \pm 2 . 9 5$ </td><td> $5 7 . 2 5 \pm 3 . 1 4$ </td><td> $5 9 . 9 6 \pm 1 . 0 6$ </td><td> $5 5 . 8 9 \pm 1 . 2 3$ </td></tr><tr><td>MVR (Ours)</td><td> ${ \bf 8 8 . 8 5 \pm 1 . 2 2 }$ </td><td> $\mathbf { 8 0 . 9 2 \pm 1 . 0 2 }$ </td><td> ${ \bf 7 0 . 3 0 \pm 2 . 1 0 }$ </td><td> $\underline { { 6 4 . 4 9 \pm 2 . 5 3 } }$ </td><td> ${ \bf 6 6 . 0 2 \pm 1 . 4 1 }$ </td><td> ${ \bf 6 1 . 0 3 \pm 3 . 2 1 }$ </td></tr></table>

Comparison methods. We compare MVR with medically pretrained 3D foundation models, including BrainIAC (Tak et al., 2026), BrainMVP (Rui et al., 2025), and 3DINO (Xu et al., 2025). We also include MedicalNet (Chen et al., 2019) as a conventional medically pretrained 3D ResNet, together with RAPTOR (An et al., 2025) and AnyMC3D (Liu et al., 2026), which adapt generalpurpose pretrained 2D foundation model encoders to volumetric inputs. We also evaluate several input-reduction baselines with frozen DINOv3: mean pooling, uniform slice sampling with $M = 1 6$ fixed-interval slices per view, supervised direct projection to M = 1 or M = 16 components, and an adapted Eigenslices (Jonemo & Eklund, 2023) using 16 uncentered principal components per view.¨ The supervised direct-projection baseline provides a controlled comparison in which the dimensionality and pretrained encoder are matched to MVR, but the input reduction is learned from diagnostic supervision. All pretrained encoder weights remain fixed, while method-specific trainable modules, including AnyMC3D’s aggregation and the supervised input projection, are retained.

Implementation details. We use $M ~ = ~ 1 6$ components per view, selected on preliminary validation and fixed for all five-fold experiments. Gaussian smoothing uses dyadic scales {1, 2, 4, 8, 16} mm, and projection fitting uses P = 256 sampled through-plane vectors per training subject and view. We use the frozen DINOv3 ViT-B/16 encoder (Simeoni et al., 2026) and concate-´ nate the class-token representations from Transformer blocks 6 and 12. Sec. 4.3 and Appendix A.2 examine sensitivity to M, P, Gaussian scales, and feature readout, while Appendix A.3 describes encoder input preparation. The MVR projections and preprocessing statistics are estimated from the training subjects within each fold. For classification, we standardize features using training statistics and fit class-balanced, L2-regularized logistic regression, with regularization strength and decision threshold selected from the training data.

## 4.2 RESULTS

Classification performance. Table 1 presents classification performance across all evaluated methods. MVR achieves the highest AUC on ADNI and OASIS at 88.85 and 70.30, and the highest BAcc on ADNI and ABIDE at 80.92 and 61.03. Compared with the medically pretrained 3D foundation models, MVR achieves higher AUC and BAcc across all three datasets. Compared with the conventional medically pretrained 3D ResNet, MedicalNet, MVR improves AUC from 86.60 to 88.85 on ADNI and from 68.35 to 70.30 on OASIS, while remaining close in OASIS BAcc (64.49 versus 64.91). MVR also exceeds RAPTOR and AnyMC3D, which adapt general-purpose pretrained 2D foundation models to volumetric inputs, in both metrics across all three datasets. For the input-reduction methods, MVR achieves higher mean AUC and BAcc than Eigenslices across all three datasets. On ABIDE, uniform slice sampling achieves a mean AUC of 65.99, close to 66.02 for MVR, while MVR achieves higher mean BAcc (61.03 versus 59.56). At the matched $M = 1 6$ setting, direct projection reaches AUC/BAcc of 82.49/75.76, 59.25/57.25, and 59.96/55.89 on ADNI, OASIS, and ABIDE, remaining far below MVR across all six metrics. Because this baseline learns the input projection from diagnostic supervision while using the same frozen encoder and component count, the result shows that task-supervised optimization of the reduction alone does not automatically yield a stronger representation. Instead, the structured base-and-detail construction of MVR provides an effective inductive bias for presenting volumetric information to a frozen 2D encoder.

Table 2: ADNI-to-AIBL external evaluation without AIBL-specific fitting or model selection. Results are mean ± sample standard deviation across five ADNI-fold-trained models. Best and secondbest means are highlighted in bold and underlined. <sup>†</sup> denotes the use of Frozen DINOv3 encoder.
<table><tr><td>Method</td><td>AUC↑</td><td>BAcc ↑</td></tr><tr><td>BrainIAC (Tak et al., 2026)</td><td> $6 9 . 2 6 \pm 2 . 1 6$ </td><td> $6 0 . 1 4 \pm 1 . 5 6$ </td></tr><tr><td>BrainMVP (Rui et al., 2025)</td><td> $7 8 . 4 3 \pm 0 . 7 7$ </td><td> $6 6 . 5 1 \pm 2 . 7 7$ </td></tr><tr><td>3DINO (Xu et al., 2025)</td><td> $8 3 . 7 3 \pm 0 . 9 3$ </td><td> ${ \underline { { 7 2 . 1 9 \pm 2 . 7 1 } } }$ </td></tr><tr><td>MedicalNet (Chen et al., 2019)</td><td> $8 6 . 2 1 \pm 0 . 2 5$ </td><td> $7 1 . 4 5 \pm 2 . 6 3$ </td></tr><tr><td>RAPTOR (An et al., 2025)</td><td> $7 7 . 6 6 \pm 1 . 4 2$ </td><td> $6 2 . 9 5 \pm 2 . 6 0$ </td></tr><tr><td>AnyMC3D† (Liu et al., 2026)</td><td> $8 0 . 4 6 \pm 0 . 9 5$ </td><td> $6 5 . 6 1 \pm 3 . 8 8$ </td></tr><tr><td>Eigenslices (Jönemo &amp; Eklund, 2023)</td><td> $8 5 . 0 2 \pm 0 . 4 9$ </td><td> $7 0 . 2 0 \pm 2 . 4 7$ </td></tr><tr><td>Mean pooling</td><td> $7 8 . 5 7 \pm 0 . 7 8$ </td><td> $6 6 . 2 7 \pm 1 . 7 8$ </td></tr><tr><td>Uniform slice sampling  $( M = 1 6 )$ </td><td> $7 7 . 9 5 \pm 1 . 7 6$ </td><td> $6 5 . 3 8 \pm 1 . 8 4$ </td></tr><tr><td>Direct projection  $( M = 1 )$ </td><td> $7 4 . 4 1 \pm 4 . 0 3$ </td><td> $6 2 . 8 7 \pm 2 . 1 3$ </td></tr><tr><td>Direct projection (M = 16)</td><td> $7 9 . 2 5 \pm 3 . 2 3$ </td><td> $6 1 . 9 3 \pm 7 . 2 9$ </td></tr><tr><td>MVR (Ours)</td><td> ${ \bf 8 7 . 2 7 \pm 1 . 1 7 }$ </td><td> ${ \bf 7 2 . 4 7 \pm 1 . 5 2 }$ </td></tr></table>

External generalization. Table 2 evaluates the five ADNI-trained models on AIBL without AIBLspecific fitting or model selection. MVR achieves the highest AUC and BAcc at 87.27 and 72.47. Compared with the medically pretrained 3D foundation models, MVR exceeds BrainIAC, Brain-MVP, and 3DINO in AUC, while its BAcc is close to 3DINO (72.47 versus 72.19). Compared with the conventional medically pretrained 3D ResNet, MedicalNet, MVR improves AUC from 86.21 to 87.27 and BAcc from 71.45 to 72.47. It also exceeds RAPTOR and AnyMC3D, which adapt general-purpose pretrained 2D foundation models to volumetric inputs, in both metrics. For the input-reduction methods, Eigenslices gives the strongest comparison at 85.02/70.20, while uniform slice sampling reaches 77.95/65.38. Overall, the ADNI advantage of MVR is largely maintained on the external AIBL cohort.

Effect of encoder adaptation. Our primary experiments intentionally keep the 2D foundation model frozen to isolate how volumetric information is presented to a pretrained 2D representation. We additionally evaluate parameter-efficient encoder adaptation using LoRA (Hu et al., 2022) on ADNI (Appendix Table 7). MVR improves slightly from 88.85/80.92 to 89.10/81.15 AUC/BAcc, whereas AnyMC3D reaches 92.93/84.56 when its encoder adaptation and slice aggregation are jointly optimized. These results indicate that the advantage of MVR should be interpreted specifically in the frozen-encoder regime rather than as a general advantage of gradient-free reduction over learned adaptation. They also suggest that learned adaptation can become advantageous when task-specific encoder optimization is permitted.

## 4.3 ABLATION STUDY

Table 3 examines the contributions of multiscale construction, component composition, and feature fusion. Below we provide detailed analysis.

Multiscale construction. Removing the Gaussian scale space while retaining the remaining MVR construction reduces AUC by 1.84 points on ADNI, 0.91 on OASIS, and 2.64 on ABIDE. The corresponding BAcc drops are 2.58, 3.38, and 1.86 points. The consistent degradation across all three datasets supports incorporating multiscale in-plane variation before through-plane reduction.

Table 3: Component ablations of MVR. AUC and BAcc are reported on a 0–100 scale as mean ± sample standard deviation across five held-out folds. The variant w/o base component in encoding omits the base component from encoder inputs while retaining its use in residual construction. The variant w/o detail components retains only the base component.
<table><tr><td rowspan="2">Variant</td><td colspan="2">ADNI</td><td colspan="2">OASIS</td><td colspan="2">ABIDE</td></tr><tr><td>AUC ↑</td><td>BAcc ↑</td><td>AUC↑</td><td>BAcc ↑</td><td>AUC↑</td><td>BAcc ↑</td></tr><tr><td>Multiscale construction</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o Gaussian scale space</td><td> $8 7 . 0 1 \pm 2 . 2 9$ </td><td> $7 8 . 3 4 \pm 3 . 9 7$ </td><td> $6 9 . 3 9 \pm 2 . 2 1$ </td><td> $6 1 . 1 1 \pm 4 . 6 1$ </td><td> $6 3 . 3 8 \pm 1 . 3 7$ </td><td> $5 9 . 1 7 \pm 3 . 8 2$ </td></tr><tr><td>Component composition</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o base component in encoding</td><td> $8 6 . 4 7 \pm 0 . 9 9$ </td><td> $8 0 . 6 9 \pm 2 . 3 5$ </td><td> $6 8 . 3 7 \pm 2 . 3 5$ </td><td> $6 2 . 4 2 \pm 2 . 8 2$ </td><td> $6 6 . 0 2 \pm 1 . 3 3$ </td><td> $5 9 . 2 1 \pm 2 . 4 0$ </td></tr><tr><td>w/o detail components</td><td> $8 5 . 4 4 \pm 2 . 3 7$ </td><td> $7 5 . 6 8 \pm 3 . 1 8$ </td><td> $6 7 . 2 9 \pm 2 . 6 2$ </td><td> $6 0 . 3 2 \pm 3 . 8 1$ </td><td> $6 1 . 5 3 \pm 3 . 9 4$ </td><td> $5 7 . 9 4 \pm 3 . 1 8$ </td></tr><tr><td>Feature fusion</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mean</td><td> $8 7 . 1 9 \pm 2 . 2 2$ </td><td> $8 0 . 1 5 \pm 3 . 9 1$ </td><td> $7 0 . 5 1 \pm 1 . 9 8$ </td><td> $6 5 . 7 1 \pm 4 . 4 6$ </td><td> $6 2 . 6 3 \pm 0 . 7 2$ </td><td> $5 8 . 3 7 \pm 1 . 8 2$ </td></tr><tr><td>Concatenation (MVR)</td><td> $8 8 . 8 5 \pm 1 . 2 2 $ </td><td> $8 0 . 9 2 \pm 1 . 0 2$ </td><td> $7 0 . 3 0 \pm 2 . 1 0$ </td><td> $6 4 . 4 9 \pm 2 . 5 3$ </td><td> $6 6 . 0 2 \pm 1 . 4 1$ </td><td> $6 1 . 0 3 \pm 3 . 2 1$ </td></tr></table>

Table 4: Effect of component count using separate encoding and concatenation across all three views. FM encodings are counted per volume. AUC and BAcc are reported on a 0–100 scale as mean ± sample standard deviation across five held-out folds.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">FM encodings</td><td colspan="2">ADNI</td><td colspan="2">OASIS</td><td colspan="2">ABIDE</td></tr><tr><td>AUC ↑</td><td>BAcc ↑</td><td>AUC↑</td><td>BAcc ↑</td><td>AUC↑</td><td>BAcc ↑</td></tr><tr><td> $M = 3$ </td><td>9</td><td> $8 6 . 9 8 \pm 2 . 1 8$ </td><td> $7 7 . 7 6 \pm 1 . 6 6$ </td><td> $7 0 . 7 2 \pm 2 . 4 6$ </td><td> $6 5 . 2 2 \pm 2 . 6 7$ </td><td> $6 2 . 5 7 \pm 2 . 3 6$ </td><td> $5 9 . 1 8 \pm 3 . 6 8$ </td></tr><tr><td> $M = 5$ </td><td>15</td><td> $8 7 . 6 7 \pm 1 . 9 6$ </td><td> $8 0 . 6 4 \pm 0 . 4 4$ </td><td> $7 0 . 6 5 \pm 2 . 9 2$ </td><td> $6 3 . 1 6 \pm 1 . 7 9$ </td><td> $6 4 . 8 4 \pm 1 . 9 0$ </td><td> $5 7 . 6 6 \pm 3 . 4 9$ </td></tr><tr><td> $M = 8$ </td><td>24</td><td> $8 7 . 8 9 \pm 1 . 2 7$ </td><td> $8 0 . 1 9 \pm 1 . 5 3$ </td><td> $7 0 . 5 3 \pm 2 . 5 0$ </td><td> $6 3 . 5 1 \pm 5 . 3 2$ </td><td> $6 4 . 8 0 \pm 0 . 8 6$ </td><td> $5 9 . 3 3 \pm 2 . 2 2$ </td></tr><tr><td> $M = 1 6 ( \mathrm { d e f a u l t } )$ </td><td>48</td><td> $8 8 . 8 5 \pm 1 . 2 2 $ </td><td> $8 0 . 9 2 \pm 1 . 0 2$ </td><td> $7 0 . 3 0 \pm 2 . 1 0$ </td><td> $6 4 . 4 9 \pm 2 . 5 3$ </td><td> $6 6 . 0 2 \pm 1 . 4 1$ </td><td> $6 1 . 0 3 \pm 3 . 2 1$ </td></tr><tr><td> $M = 3 2$ </td><td>96</td><td> $8 8 . 2 8 \pm 1 . 3 7$ </td><td> $8 0 . 0 1 \pm 0 . 9 9$ </td><td> $7 0 . 4 1 \pm 2 . 0 2$ </td><td> $6 3 . 5 6 \pm 2 . 8 8$ </td><td> $6 5 . 4 1 \pm 1 . 5 1$ </td><td> $5 8 . 0 4 \pm 3 . 5 6$ </td></tr></table>

Component composition. Retaining only the base component reduces AUC by 3.41 points on ADNI, 3.01 on OASIS, and 4.49 on ABIDE, while BAcc decreases by 5.24, 4.17, and 3.09 points. This shows that the base component alone is insufficient to form the complete representation. Removing the base component from the encoder inputs while retaining it for residual construction also lowers both metrics on ADNI and OASIS. On ABIDE, mean AUC remains unchanged at the reported precision, while BAcc decreases from 61.03 to 59.21. The encoded base component therefore provides a complementary overall contribution, while the detail components deliver larger and more consistent gains across all three datasets.

Feature fusion. Full concatenation performs best on ADNI and ABIDE, improving AUC over mean fusion by 1.66 and 3.39 points and BAcc by 0.77 and 2.66 points. On OASIS, mean fusion performs slightly better, reaching 70.51 AUC and 65.71 BAcc compared with 70.30 and 64.49 for concatenation. These results show that preserving separate component features is beneficial on ADNI and ABIDE, while averaging remains competitive on OASIS.

Component count. Table 4 examines the number of MVR components retained per anatomical view. On ADNI, performance generally improves with M and peaks at $M = 1 6 ,$ , reaching 88.85 AUC and 80.92 BAcc. ABIDE shows a similar pattern, with M = 16 achieving the highest AUC and BAcc of 66.02 and 61.03. OASIS is less sensitive to component count, with AUC remaining between 70.30 and 70.72 across all tested values and the highest AUC/BAcc obtained at $M = 3 .$ Increasing M from 16 to 32 provides no consistent improvement across datasets and metrics.

## 5 CONCLUSION

We introduced Multiscale Volumetric Reduction (MVR), a data-adaptive framework that departs from the conventional encode-then-integrate paradigm by reducing volumetric information before foundation-model encoding. MVR constructs complementary base and multiscale detail components that preserve through-plane structure while incorporating spatial context, enabling a frozen 2D foundation model to represent a 3D brain MRI volume using a compact set of dense 2D inputs.

Across multiple brain-MRI cohorts and diagnostic tasks, MVR achieves strong and consistent performance and transfers well to an external cohort. These results support reduction before encoding as a viable alternative to post-encoding volumetric integration for adapting frozen 2D foundation models to brain MRI.

## REFERENCES

Ulzee An, Moonseong Jeong, Simon A. Lee, Aditya Gorla, Yuzhe Yang, and Sriram Sankararaman. Raptor: Scalable train-free embeddings for 3d medical volumes leveraging pretrained 2d foundation models. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, 2025.

Liangjun Chen, Zhengwang Wu, Fenqiang Zhao, Ya Wang, Weili Lin, Li Wang, and Gang Li. An attention-based context-informed deep framework for infant brain subcortical segmentation. Neuroimage, 269:119931, Apr 2023.

Sihong Chen, Kai Ma, and Yefeng Zheng. Med3d: Transfer learning for 3d medical image analysis. arXiv preprint arXiv:1904.00625, 2019.

Kathryn A Ellis, Ashley I Bush, David Darby, Daniela De Fazio, Jonathan Foster, Peter Hudson, Nicola T Lautenschlager, Nat Lenzo, Ralph N Martins, Paul Maruff, Colin Masters, Andrew Milner, Kerryn Pike, Christopher Rowe, Greg Savage, Cassandra Szoeke, Kevin Taddei, Victor Villemagne, Michael Woodward, and David Ames. The australian imaging, biomarkers and lifestyle (aibl) study of aging: methodology and baseline characteristics of 1112 individuals recruited for a longitudinal study of alzheimer’s disease. Int Psychogeriatr, 21(4):672–687, Aug 2009.

Vladimir Fonov, Alan C. Evans, Kelly Botteron, C. Robert Almli, Robert C. McKinstry, and D. Louis Collins. Unbiased average age-appropriate atlases for pediatric studies. NeuroImage, 54 (1):313–327, 2011.

Anibal Solon Heinsfeld, Alexandre Rosa Franco, R. Cameron Craddock, Augusto Buchweitz, and´ Felipe Meneguzzi. Identification of autism spectrum disorder using deep learning and the abide dataset. NeuroImage: Clinical, 17:16–23, 2018.

Edward J Hu, Yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Johan Jonemo and Anders Eklund. Brain age prediction using 2d projections based on higher-order¨ statistical moments and eigenslices from 3d magnetic resonance imaging volumes. J Imaging, 9 (12), Dec 2023.

Pamela J. LaMontagne, Tammie LS. Benzinger, John C. Morris, Sarah Keefe, Russ Hornbeck, Chengjie Xiong, Elizabeth Grant, Jason Hassenstab, Krista Moulder, Andrei G. Vlassenko, Marcus E. Raichle, Carlos Cruchaga, and Daniel Marcus. Oasis-3: Longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and alzheimer disease. medRxiv, 2019.

Han Liu, Bogdan Georgescu, Yanbo Zhang, Youngjin Yoo, Michael Baumgartner, Riqiang Gao, Jianing Wang, Gengyan Zhao, Eli Gibson, Dorin Comaniciu, and Sasa Grbic. Revisiting 2d foundation models for scalable 3d medical image classification. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 30021–30031, June 2026.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khali-´ dov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=a68SUt6zFt. Featured Certification.

R C Petersen, P S Aisen, L A Beckett, M C Donohue, A C Gamst, D J Harvey, C R Jr Jack, W J Jagust, L M Shaw, A W Toga, J Q Trojanowski, and M W Weiner. Alzheimer’s disease neuroimaging initiative (adni): clinical characterization. Neurology, 74(3):201–209, Jan 2010.

Fazle Rafsani, Jay Shah, Catherine D. Chong, Todd J. Schwedt, and Teresa Wu. Dinoatten3d: Slicelevel attention aggregation of dinov2 for 3d brain mri anomaly classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pp. 3594–3603, October 2025.

Shaohao Rui, Lingzhi Chen, Zhenyu Tang, Lilong Wang, Mianxin Liu, Shaoting Zhang, and Xiaosong Wang. Multi-modal vision pre-training for medical image analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5164–5174, June 2025.

Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose,´ Vasil Khalidov, Marc Szafraniec, Seung Eun Yi, Michael Ramamonjisoa, Francisco Massa, Daniel HAZIZA, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel´ Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve Jegou, Patrick Labatut, and Piotr Bojanowski. DINOv3. Transactions on Machine Learning Research, 2026. ISSN 2835-8856.

Divyanshu Tak, Biniam A. Garomsa, Anna Zapaishchykova, Tafadzwa L. Chaunzwa, Juan Carlos Climent Pardo, Zezhong Ye, John Zielke, Yashwanth Ravipati, Suraj Pai, Sri Vajapeyam, Maryam Mahootiha, Mitchell Parker, Luke R. G. Pike, Ceilidh Smith, Ariana M. Familiar, Kevin X. Liu, Sanjay Prabhu, Omar Arnaout, Pratiti Bandopadhayay, Ali Nabavizadeh, Sabine Mueller, Hugo JWL Aerts, Raymond Y. Huang, Tina Y. Poussaint, and Benjamin H. Kann. A general izable foundation model for analysis of human brain mri. Nature Neuroscience, 29(4):945–956, 2026.

Nicholas J Tustison, Brian B Avants, Philip A Cook, Yuanjie Zheng, Alexander Egan, Paul A Yushkevich, and James C Gee. N4itk: improved n3 bias correction. IEEE Trans Med Imaging, 29(6):1310–1320, Jun 2010.

Daan van Rooij, Evdokia Anagnostou, Celso Arango, Guillaume Auzias, Marlene Behrmann, Geraldo F Busatto, Sara Calderoni, Eileen Daly, Christine Deruelle, Adriana Di Martino, Ilan Dinstein, Fabio Luis Souza Duran, Sarah Durston, Christine Ecker, Damien Fair, Jennifer Fedor, Jackie Fitzgerald, Christine M Freitag, Louise Gallagher, Ilaria Gori, Shlomi Haar, Liesbeth Hoekstra, Neda Jahanshad, Maria Jalbrzikowski, Joost Janssen, Jason Lerch, Beatriz Luna, Mauricio Moller Martinho, Jane McGrath, Filippo Muratori, Clodagh M Murphy, Declan G M Murphy, Kirsten O’Hearn, Bob Oranje, Mara Parellada, Alessandra Retico, Pedro Rosa, Katya Rubia, Devon Shook, Margot Taylor, Paul M Thompson, Michela Tosetti, Gregory L Wallace, Fengfeng Zhou, and Jan K Buitelaar. Cortical and subcortical brain morphometry differences between patients with autism spectrum disorder and healthy individuals across the lifespan: Result from the enigma asd working group. Am J Psychiatry, 175(4):359–369, Apr 2018.

Tony Xu, Sepehr Hosseini, Chris Anderson, Anthony Rinaldi, Rahul G Krishnan, Anne L Martel, and Maged Goubran. A generalizable 3d framework and model for self-supervised learning in medical imaging. NPJ Digit Med, 8(1):639, Nov 2025.

## A APPENDIX

## A.1 DETAILED FORMULATION OF THE MVR PROJECTIONS

We provide the detailed formulation of the base and detail projections used in MVR, including their PCA objectives and eigendecomposition solutions.

## A.1.1 BASE PROJECTION

Let $X _ { v } = [ { \bf x } _ { 1 } , \ldots , { \bf x } _ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times D }$ contain the sampled through-plane vectors for anatomical view v, following the notation in the main paper Sec.3.2. We want to find a unit projection vector u $\in \mathbb { R } ^ { D }$

that maximizes the average squared projection of these vectors:

$$
\mathbf { u } _ { b } = \arg \operatorname* { m a x } _ { \| \mathbf { u } \| _ { 2 } = 1 } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \big ( \mathbf { x } _ { i } ^ { \top } \mathbf { u } \big ) ^ { 2 } .\tag{7}
$$

The objective can be rewritten as

$$
\frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \mathbf { x } _ { i } ^ { \top } \mathbf { u } \right) ^ { 2 } = \mathbf { u } ^ { \top } \left( \frac { 1 } { N } X _ { v } ^ { \top } X _ { v } \right) \mathbf { u } = \mathbf { u } ^ { \top } M _ { b } \mathbf { u } ,\tag{8}
$$

where

$$
M _ { b } = \frac { 1 } { N } X _ { v } ^ { \top } X _ { v }\tag{9}
$$

is the uncentered second-moment matrix.

Since $M _ { b }$ is symmetric and positive semidefinite, we compute its eigendecomposition

$$
M _ { b } = U _ { b } \Lambda _ { b } U _ { b } ^ { \top } ,\tag{10}
$$

where the eigenvalues in $\Lambda _ { b }$ are ordered from largest to smallest. The objective in Eq. 7 is maximized by the unit eigenvector corresponding to the largest eigenvalue. We therefore select this eigenvector as the base projection vector $\mathbf { u } _ { b }$

The use of uncentered PCA is motivated by the role of the base component as an original-intensity reference. Let $\begin{array} { r } { \pmb { \mu _ { x } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { x } _ { i } } } \end{array}$ denote the mean sampled through-plane vector and let

$$
C _ { x } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \mathbf { x } _ { i } - \pmb { \mu } _ { x } ) ( \mathbf { x } _ { i } - \pmb { \mu } _ { x } ) ^ { \top }\tag{11}
$$

denote the centered covariance matrix. Expanding the covariance gives

$$
C _ { x } = M _ { b } - \mu _ { x } \mu _ { x } ^ { \top } ,\tag{12}
$$

or equivalently

$$
M _ { b } = C _ { x } + \mu _ { x } \mu _ { x } ^ { \top } .\tag{13}
$$

Thus, centered PCA removes the average through-plane intensity structure before estimating its principal directions, whereas the uncentered second-moment matrix retains this structure. This is consistent with using the base component as an original-intensity reference.

Applying the fitted projection vector to the sampled through-plane vectors gives the sampled base scores

$$
\mathbf { b } = X _ { v } \mathbf { u } _ { b } = [ b _ { 1 } , \ldots , b _ { N } ] ^ { \top } , \qquad b _ { i } = \mathbf { x } _ { i } ^ { \top } \mathbf { u } _ { b } .\tag{14}
$$

The same projection is applied at each in-plane location of a volume to construct the base component:

$$
b _ { v } ( h , w ) = \mathbf { u } _ { b } ^ { \top } \mathbf { x } _ { v } ( h , w ) .\tag{15}
$$

## A.1.2 DETAIL PROJECTION

Following the residualization defined in the main paper Sec. 3.3, let $\mathbf { r } _ { i } \in \mathbb { R } ^ { K D }$ denote the residual descriptor associated with the i-th sampled multiscale descriptor. Stacking the residual descriptors as rows gives $R _ { v } = [ \mathbf { r } _ { 1 } , \ldots , \mathbf { r } _ { N } ] ^ { \top } \in \bar { \mathbb { R } } ^ { \bar { N } \times K D }$ , with covariance matrix

$$
C _ { r } = \frac { 1 } { N } R _ { v } ^ { \top } R _ { v } .\tag{16}
$$

The detail projections retain the directions with the largest residual variance. For the first detail projection, we solve

$$
\mathbf { w } _ { 1 } = \arg \operatorname* { m a x } _ { \| \mathbf { w } \| _ { 2 } = 1 } \mathbf { w } ^ { \top } C _ { r } \mathbf { w } .\tag{17}
$$

Subsequent projection vectors are obtained under orthogonality constraints:

$$
\mathbf { w } _ { j } = \arg \operatorname* { m a x } _ { \| \mathbf { w } \| _ { 2 } = 1 } \mathbf { w } ^ { \top } C _ { r } \mathbf { w } , \qquad \mathbf { w } ^ { \top } \mathbf { w } _ { \ell } = 0 \quad \mathrm { f o r } \ell < j .\tag{18}
$$

Since $C _ { r }$ is symmetric and positive semidefinite, we compute its eigendecomposition

$$
\begin{array} { r } { C _ { r } = U _ { r } \Lambda _ { r } U _ { r } ^ { \top } , } \end{array}\tag{19}
$$

where the eigenvalues in Λ<sub>r</sub> are ordered from largest to smallest. The M −1 detail projection vectors are therefore the unit eigenvectors corresponding to the $M - 1$ largest eigenvalues:

$$
[ \mathbf { w } _ { 1 } , \dots , \mathbf { w } _ { M - 1 } ] = U _ { r } [ : , 1 { : } M - 1 ] .\tag{20}
$$

The selected projection vectors capture the dominant residual multiscale variation and are mutually orthogonal in the residual PCA space. Applying ${ \bf w } _ { j }$ to the residual descriptor $\mathbf { r } _ { v } ( h , w )$ gives the corresponding detail-component value, as defined in the main text.

## A.2 ABLATION

Table 5: Effect of the number of sampled through-plane vectors per training subject on ADNI. AUC and BAcc are reported on a 0–100 scale as mean ± sample standard deviation across five held-out folds. Best means are in bold and second-best means are underlined.

<table><tr><td>P</td><td>AUC↑</td><td>BAcc ↑</td></tr><tr><td>64</td><td> $8 8 . 3 3 \pm 1 . 6 3$ </td><td> $8 0 . 8 4 \pm 1 . 5 1$ </td></tr><tr><td>128</td><td> $8 8 . 4 6 \pm 1 . 4 5$ </td><td> $7 9 . 9 9 \pm 1 . 5 1$ </td></tr><tr><td>256</td><td> ${ \bf 8 8 . 8 5 \pm 1 . 2 2 }$ </td><td> $8 0 . 9 2 \pm 1 . 0 2$ </td></tr><tr><td>512</td><td> $\underline { { 8 8 . 8 0 \pm 1 . 3 7 } }$ </td><td> $\mathbf { 8 1 . 6 8 \pm 0 . 8 7 }$ </td></tr><tr><td>All</td><td> $8 8 . 6 2 \pm 1 . 4 6$ </td><td> $8 0 . 5 9 \pm 1 . 5 2$ </td></tr></table>

Sampling sensitivity. Table 5 examines the number of through-plane vectors sampled per training subject for fitting the MVR projections. Performance is relatively stable across the tested values, with AUC ranging from 88.33 to 88.85 and BAcc from 79.99 to 81.68. The default setting $P = 2 5 6$ achieves the highest AUC (88.85), while P = 512 gives the highest BAcc (81.68). Using all available vectors does not improve over the sampled settings, indicating that a moderate number of sampled vectors is sufficient for estimating the shared projections.

Table 6: Effect of anatomical-view composition on classification performance. AUC and BAcc are reported on a 0–100 scale. Best results are highlighted in bold and second-best results are underlined.
<table><tr><td rowspan="2">Retained views</td><td colspan="2">ADNI</td><td colspan="2">OASIS</td><td colspan="2">ABIDE</td></tr><tr><td>AUC↑</td><td>BAcc ↑</td><td>AUC↑</td><td>BAcc ↑</td><td>AUC↑</td><td>BAcc ↑</td></tr><tr><td>Axial</td><td>86.80</td><td>79.22</td><td>68.80</td><td>62.79</td><td>65.86</td><td>60.77</td></tr><tr><td>Coronal</td><td>85.36</td><td>77.52</td><td>69.94</td><td>64.32</td><td>61.76</td><td>58.58</td></tr><tr><td>Sagittal</td><td>85.77</td><td>76.60</td><td>67.88</td><td>62.43</td><td>62.71</td><td>58.52</td></tr><tr><td>Axial + coronal</td><td>87.90</td><td>80.05</td><td>69.77</td><td>62.17</td><td>65.25</td><td>61.47</td></tr><tr><td>Axial + sagittal</td><td>88.48</td><td>80.80</td><td>69.47</td><td>63.15</td><td>65.95</td><td>59.92</td></tr><tr><td>Coronal + sagittal</td><td>87.20</td><td>79.89</td><td>70.16</td><td>64.34</td><td>63.27</td><td>59.71</td></tr><tr><td>All three</td><td>88.85</td><td>80.92</td><td>70.30</td><td>64.49</td><td>66.02</td><td>61.03</td></tr></table>

Anatomical-view composition. Table 6 evaluates the contribution of the axial, coronal, and sagittal views individually and in combination. Using all three views gives the highest AUC and BAcc on ADNI (88.85/80.92) and OASIS (70.30/64.49), supporting the use of complementary information across anatomical orientations. On ABIDE, using all three views gives the highest AUC (66.02), while axial+coronal gives the highest BAcc (61.47). Nevertheless, the three-view representation remains competitive on both metrics, suggesting that the relative contribution of each anatomical view is dataset dependent.

Gaussian-scale sensitivity. Table 8 compares the default five-scale configuration with a denser nine-scale configuration spanning the same 1–16 mm range. Increasing the sampling density of the Gaussian scale space changes the mean AUC only marginally, from 88.85 to 88.93, while the mean BAcc remains unchanged at 80.92. This suggests that MVR is not sensitive to the precise density of Gaussian scales within the evaluated range.

Table 7: Classification performance with LoRA adaptation on ADNI. AUC and BAcc are reported on a 0–100 scale as mean ± sample standard deviation across five held-out folds. Best means are in bold and second-best means are underlined.
<table><tr><td>Method</td><td>AUC ↑</td><td>BAcc ↑</td></tr><tr><td>BrainIAC</td><td> $7 9 . 6 5 \pm 2 . 2 9$ </td><td> $7 2 . 0 1 \pm 1 . 2 0$ </td></tr><tr><td>BrainMVP</td><td> $8 5 . 2 4 \pm 3 . 3 8$ </td><td> $7 5 . 2 5 \pm 4 . 8 0$ </td></tr><tr><td>3DINO</td><td> $8 7 . 0 1 \pm 1 . 6 0$ </td><td> $7 9 . 1 0 \pm 0 . 9 4$ </td></tr><tr><td>AnyMC3D</td><td> ${ \bf 9 2 . 9 3 \pm 3 . 1 4 }$ </td><td> ${ \bf 8 4 . 5 6 \pm 4 . 2 8 }$ </td></tr><tr><td>MVR (Ours)</td><td> $8 9 . 1 0 \pm 1 . 1 6$ </td><td> $8 1 . 1 5 \pm 2 . 5 9$ </td></tr></table>

Table 8: Sensitivity to the Gaussian scale configuration on ADNI. AUC and BAcc are reported on a 0–100 scale as mean ± sample standard deviation across five held-out folds. Best means are in bold and second-best means are underlined.
<table><tr><td>Gaussian scales (mm)</td><td>AUC ↑</td><td>BAcc ↑</td></tr><tr><td> $\{ 1 , 2 , 4 , 8 , 1 6 \}$ </td><td> $8 8 . 8 5 \pm 1 . 2 2 $ </td><td> $\mathbf { 8 0 . 9 2 \pm 1 . 0 2 }$ </td></tr><tr><td> $\{ 1 , \sqrt { 2 } , 2 , 2 \sqrt { 2 } , 4 , 4 \sqrt { 2 } , 8 , 8 \sqrt { 2 } , 1 6 \}$ </td><td> $\mathbf { 8 8 . 9 3 \pm 1 . 5 8 }$ </td><td> $\mathbf { 8 0 . 9 2 \pm 2 . 3 4 }$ </td></tr></table>

LoRA adaptation. Table 7 evaluates parameter-efficient encoder adaptation using LoRA with a common rank of $r = 8$ across all methods. MVR reaches 89.10 AUC and 81.15 BAcc, outperforming the medically pretrained 3D encoders. AnyMC3D achieves the highest performance, with 92.93 AUC and 84.56 BAcc. This setting is outside the primary design of MVR, which constructs the volumetric representation for a frozen DINOv3 encoder without gradient-based encoder adaptation. By contrast, AnyMC3D is designed to jointly optimize trainable encoder adaptation and slice aggregation for volumetric classification. The LoRA results therefore show that MVR remains competitive when encoder adaptation is introduced, while its primary setting is the use of a frozen pretrained 2D foundation model.

Feature-layer ablation. Table 9 compares representations extracted from intermediate and final DINOv3 blocks on ADNI. Block 12 provides a stronger single-layer representation than Block $^ { 6 , }$ improving AUC by 1.48 points and BAcc by 2.01 points. Concatenating features from Blocks 6 and 12 further improves performance to 88.85 AUC and 80.92 BAcc, corresponding to gains of 0.54 AUC points and 0.30 BAcc points over Block 12 alone. These results suggest that the final block contains the strongest task-relevant representation, while the intermediate block contributes complementary information. The additional gain is modest relative to the doubled classifier input dimension, although concatenating the two blocks requires no additional DINOv3 forward passes.

## A.3 IMPLEMENTATION

Component scaling and encoder input preparation. For each anatomical view v and component m, the projected values are reshaped to form a 2D component $Z _ { v , m } \ \in \ \mathbb R ^ { H _ { v } \times W _ { v } }$ Foreground locations are identified from the constant minimum-valued background of each normalized volume, without requiring an external brain mask or segmentation. All scaling statistics are estimated from the corresponding training fold. Let $q _ { v , m } ^ { 1 }$ and $q _ { v , m } ^ { \overleftarrow { 9 } 9 }$ denote the 1st and 99th percentiles of the training values for component m. We define

$$
c _ { v , m } = \frac { q _ { v , m } ^ { 1 } + q _ { v , m } ^ { 9 9 } } { 2 } , \qquad s _ { v , m } = \operatorname* { m a x } \left( \frac { q _ { v , m } ^ { 9 9 } - q _ { v , m } ^ { 1 } } { 2 } , 1 0 ^ { - 8 } \right) .\tag{21}
$$

Each foreground value is then mapped to [0, 1] as

$$
\widetilde { Z } _ { v , m } ( p ) = \frac { 1 } { 2 } \left[ \mathrm { c l i p } \left( \frac { Z _ { v , m } ( p ) - c _ { v , m } } { s _ { v , m } L _ { v } } , - 1 , 1 \right) + 1 \right] ,\tag{22}
$$

where $L _ { v }$ is the 99th percentile of the absolute standardized responses estimated from the training fold for view v. Background locations are set to zero. Each resulting component is replicated across three channels, resized to $2 5 6 \times 2 5 6$ using antialiased bilinear interpolation, and normalized using the ImageNet channel statistics used by DINOv3. All scaling and clipping parameters are fixed before application to validation and test subjects.

Table 9: Feature-layer ablation on ADNI. AUC and BAcc are reported on a 0–100 scale as mean ± sample standard deviation across five held-out folds. Best results are highlighted in bold.
<table><tr><td>Representation</td><td>AUC↑</td><td>BAcc ↑</td></tr><tr><td>Block 6 only</td><td> $8 6 . 8 3 \pm 1 . 6 1$ </td><td> $7 8 . 6 1 \pm 0 . 6 2$ </td></tr><tr><td>Block 12 only</td><td> $8 8 . 3 1 \pm 1 . 3 2 $ </td><td> $8 0 . 6 2 \pm 1 . 9 9$ </td></tr><tr><td>Block 6 + Block 12</td><td> ${ \bf 8 8 . 8 5 \pm 1 . 2 2 }$ </td><td> $\mathbf { 8 0 . 9 2 \pm 1 . 0 2 }$ </td></tr></table>

Uniform Slice Sampling. For the uniform-sampling comparison, we select $M = 1 6$ fixed slice positions independently for each anatomical view and outer fold. For each training subject s, foreground is identified directly from the constant minimum-valued background of the normalized volume. A slice at depth d is considered occupied if it contains at least one foreground voxel. Let $a _ { s , x }$ and $b _ { s , v }$ denote the first and one-past-last occupied slice for subject s in view v. We obtain a fold-level foreground interval from the median boundaries of the training subjects,

$$
a _ { v } = \mathrm { r o u n d } ( \mathrm { m e d i a n } _ { s \in \mathcal { T } } a _ { s , v } ) , \qquad b _ { v } = \mathrm { r o u n d } ( \mathrm { m e d i a n } _ { s \in \mathcal { T } } b _ { s , v } ) .\tag{23}
$$

The interval $[ a _ { v } , b _ { v } )$ is divided into M equal-width bins, and the slice nearest the center of each bin is selected:

$$
i _ { v , m } = \left\lfloor a _ { v } + \left( m + \frac { 1 } { 2 } \right) \frac { b _ { v } - a _ { v } } { M } \right\rfloor , \qquad m = 0 , \ldots , M - 1 .\tag{24}
$$

The resulting indices are fixed and applied to every training and held-out subject in the corresponding fold, so slice position m represents approximately the same anatomical location across subjects. Each selected slice is prepared using the same training-fold intensity scaling and DINOv3 input preprocessing as the MVR components, encoded independently by the frozen encoder, and concatenated across slices and views.