# Prototype-Rule Neurosymbolic Regularization for Rank-Constrained Tensor Neural Networks under Label Scarcity

Eftychios Protopapadakis<sup>a,∗</sup>, Konstantinos Makantasis<sup>b</sup>, Konstantinos M. Giannoutakis<sup>a</sup>

<sup>a</sup>Department of Applied Informatics, University of Macedonia, 156 Egnatia Street, 54636 Thessaloniki, Greece

<sup>b</sup>Department of Artificial Intelligence, Faculty of Information & Communication Technology, University of Malta, Msida, Malta

## Abstract

Rank - constrained tensor neural networks reduce the parameterization of high - order inputs, but they do not explicitly constrain class geometry in the learned representation. This study investigates whether a diferentiable prototype - rule can provide a complementary inductive bias for Rank-R tensor learning under limited supervision. The proposed framework augments the Rank-R objective with prototype-based regularization and optionally fuses prototype evidence with neural logits at inference. Four hyperspectral benchmarks are evaluated with four Rank-R configurations under both seven-fold stratification and spatially separated folds that mitigate leakage; a separate spatial study varies the class support budget from 2 to 20 samples. Under spatial evaluation, full neurosymbolic inference changes Macro-F1 score by +8.82 percentage points on Botswana, +5.49 on Indian Pines, +1.59 on Pavia

University, and -0.62 on Salinas. Most of the benefit arises from training time regularization, whereas inference fusion is small and dataset dependent. Keywords: neurosymbolic learning, Rank-R neural networks, tensor learning, prototype regularization, label scarcity, hyperspectral classification, spatial leakage

## 1. Introduction

Tensor-valued observations arise naturally in remote sensing and in other high-dimensional problems, in which the ordering of input modes carries information. A conventional dense neural layer can ignore that structure and may introduce many parameters relative to the number of labeled observations. Rank-constrained feedforward neural networks [1, 2] address this issue by representing the input-to-hidden weight tensors through a Canonical/Polyadic (CP) decomposition rather than learning unrestricted dense tensors [3, 4]. The resulting Rank-R formulation retains multilinear structure while substantially reducing the number of free parameters, an attractive property for small-sample high-order learning [3]. Rank-R representations have also been reused as embeddings for graph-based semi-supervised hyperspectral classification [5].

Nevertheless, parameter eficiency constrains how the mapping is represented, rather than how classes should be arranged in the learned latent space. A complementary line of research incorporates prior relations or logical constraints into diferentiable learning objectives [6–9]. Prototype-based methods provide a particularly simple class-level structure: examples of a class are related to a representative point in embedding space, and membership can be expressed through relative compatibility with class prototypes [10, 11]. This suggests a natural question for Rank-R learning: can an explicit prototype relation provide information that is complementary to the low-rank tensor constraint?

The present work answers that question as a controlled characterization, rather than as a state-of-the-art hyperspectral-classification benchmark. We introduce a diferentiable prototype-rule regularizer into a Rank-R neural model and deliberately separate two possible mechanisms: representation shaping during training and the use of prototype evidence at inference. We further distinguish conventional stratified evaluation from spatially separated evaluation because patch-based hyperspectral experiments can be optimistic when neighboring samples from the same scene are allowed to cross split boundaries [12, 13].

Finally, we evaluate the mechanism under explicit per-class label budgets to test the common intuition that stronger inductive biases should become more useful as supervision decreases. This hypothesis is consistent with prior work in constraint-based prior-knowledge regularization and prototype-based few-shot learning, where additional structural information is introduced specifically to improve learning when direct supervision is limited [6, 10].

To the best of our knowledge, prior Rank-R FNN studies have not investigated explicit diferentiable class rules as an additional training constraint. The novelty claim is therefore intentionally narrow: we study prototype-rule neurosymbolic regularization of CP-constrained Rank-R representations and characterize when its contribution is positive, neutral, or adverse.

## 1.1. Contributions and research questions

The paper makes four contributions. First, it integrates a diferentiable prototype-rule mechanism into a CP-constrained Rank-R neural representation. Second, it introduces a matched ablation that isolates training-time prototype regularization from inference - time prototype fusion without changing the Rank-R backbone. Third, it evaluates the same mechanism under both spatially leakage-controlled and conventional stratified protocols and keeps their evidence separate. Fourth, it performs a dedicated spatial label-scarcity study over seven per-class budgets and five support realizations, allowing the dependence of the neurosymbolic efect on supervision, dataset, and Rank-R configuration to be examined directly.

These contributions are organized around five research questions. The first concerns whether the mechanism works at all: does diferentiable prototype - rule regularization change classification performance relative to the corresponding purely neural Rank-R baseline (RQ1)? Where an efect occurs, we ask how much of it is attributable to training - time regularization and how much to prototype-based inference fusion (RQ2), and how the observed efect difers between conventional stratified evaluation and spatially leakage - controlled evaluation (RQ3). We further ask whether the contribution of the prototype-rule mechanism systematically increases as the number of labeled samples per class decreases (RQ4), and to what extent the efect is conditioned by dataset and Rank-R architecture (RQ5).

## 2. Related Work

The proposed framework draws on two lines of work that have largely been developed separately. Rank-constrained tensor networks address the parameterization of high-order inputs but leave the geometry of the learned representation unconstrained, while neurosymbolic and prototype-based methods impose explicit class-level structure without regard on how the underlying representation is parameterized. Section 2.1 reviews the former, Section 2.2 the latter, and Section 2.3 positions the present work as a deliberately minimal combination of the two.

## 2.1. Rank-R tensor neural networks

The Rank-R FNN constrains each input-to-hidden weight tensor through a CP decomposition, preserving the high-order organization of the input while replacing a full tensor by a sum of rank-one factors [3, 4]. This parameterization is especially relevant when the input dimensionality is high but labeled data are limited. Subsequent work has reused Rank-R-derived tensor embeddings in graph-based semi-supervised hyperspectral classification, showing that the representation can support additional relational structure beyond the original classifier [5]. The present work difers from these studies by changing the learning objective itself through an explicit class-level constraint.

## 2.2. Neurosymbolic and prototype-based learning

Neurosymbolic learning broadly combines neural representation learning with symbolic, logical, or knowledge-based structure [8, 14]. Diferentiable approaches such as Logic Tensor Networks and constraint-based regularization map relations into continuous objectives that can participate in gradientbased optimization [6, 7, 9]. The symbolic component need not be a discrete theorem prover; it can be a declarative relation whose degree of satisfaction is made diferentiable [15]. In the present framework, this means that the symbolic knowledge is the explicit class-level rule that an embedding should be closer to the prototype of its own class than to competing class prototypes; the degree to which this rule is satisfied is converted into a diferentiable loss and optimized jointly with the neural model.

Prototype-based learning ofers a related geometric mechanism. Prototypical networks, for example, classify a query according to its relation to class prototypes computed in an embedding space [10, 16]. Prototype learning has also been explored in hyperspectral image classification under limited labeled supervision [11], including prototype-rectification strategies designed specifically for cross-domain few-shot classification [17]. More recent work has further developed global–local prototype representations to construct discriminative spatial–spectral metric spaces for few-shot HSI classification [18].

In the present study, prototypes are not used as a meta-learning episode construction. Instead, they ground the explicit relation that an embedding belonging to class c should be more compatible with prototype c than with competing class prototypes. The resulting term is used as a training constraint, while prototype-based inference is treated as a separable optional component.

Table 1: Conceptual positioning of the proposed mechanism. The table compares mechanisms rather than predictive performance.
<table><tr><td>Approach</td><td></td><td></td><td>Tensor structure Explicit relation Training constraint Prototype inference</td><td></td></tr><tr><td>Rank-R FNN</td><td>√</td><td></td><td></td><td></td></tr><tr><td>Prototype-based learning</td><td>varies</td><td></td><td>varies</td><td>varies</td></tr><tr><td>Differentiable constraint-based NS</td><td>varies</td><td>√</td><td>√</td><td>varies</td></tr><tr><td>Present framework</td><td>√</td><td>√</td><td>√</td><td>optional</td></tr></table>

## 2.3. Positioning of the present work

The proposed model is deliberately a minimal neurosymbolic construction. Unlike approaches employing knowledge-graph reasoning [19], ontologybased knowledge representation [20], or symbolic state-space search [21], the proposed framework relies on a diferentiable class-level relation. This relation explicitly constrains the geometry of the learned representation. The resulting formulation enables a controlled evaluation of the additional inductive bias while preserving the underlying Rank-R architecture.

## 3. Proposed Neurosymbolic Rank-R Framework

This section defines the proposed framework in three parts. Section 3.1 recalls the CP-constrained Rank-R backbone that produces the latent embedding used throughout. Section 3.2 introduces the diferentiable prototyperule regularizer, its class-prototype distance, and its combination with the cross-entropy objective. Section 3.3 then defines the three matched training and inference variants — RankR, RankR - NS - RegOnly, and RankR-NS — used to separate training-time regularization from inference-time prototype fusion. Figure 1 provides a high-level overview of the proposed neurosymbolic Rank-R framework.

![](images/407fb0b25bf69ca1bb3f0b4469d1a9d8a711bc085f3829dbec9b1ea783023d21.jpg)  
Figure 1: High-level overview of the proposed neurosymbolic Rank-R framework. Hyperspectral input patches are mapped to a latent representation through the CP-constrained Rank-R backbone. The learned embedding can be further shaped by the prototype-rule constraint, which introduces class-prototype attraction and inter-class separation during training. The resulting representation supports three matched variants: the purely neural RankR baseline, RankR-NS-RegOnly using prototype-rule regularization during training, and RankR-NS additionally incorporating prototype evidence during inference.

## 3.1. Rank-R backbone and latent representation

Let an input sample be a D-order tensor $\mathcal { X } \in \mathbb { R } ^ { I _ { 1 } \times \cdots \times I _ { D } }$ . For hidden unit $q ,$ the dense weight tensor is replaced by a rank-R CP representation,

$$
{ \mathscr W } ^ { ( q ) } = \sum _ { r = 1 } ^ { R } { \bf w } _ { 1 } ^ { ( q , r ) } \circ { \bf w } _ { 2 } ^ { ( q , r ) } \circ \cdots \circ { \bf w } _ { D } ^ { ( q , r ) } ,\tag{1}
$$

where $\bigcirc$ denotes the outer product. he output $z ^ { ( q ) }$ of hidden unit $q$ is the computed as

$$
\begin{array} { r } { z ^ { ( q ) } = f ( \mathrm { v e c } ( \mathcal { X } ) ^ { T } \cdot \mathrm { v e c } ( \mathcal { W } ^ { ( q ) } ) ) , } \end{array}\tag{2}
$$

where vec() is the vectorization operator which transforms tensor into a col umn vector, and $f ( )$ is a non-linear activation function. With H neurons in

the first hidden layer, the input tensor is transformed into a vector embedding

$$
\mathbf { z } = [ z ^ { ( 1 ) } z ^ { ( 2 ) } \cdot \cdot \cdot z ^ { ( H ) } ] \in \mathbb { R } ^ { H } .\tag{3}
$$

The ordinary Rank-R classifier maps z to neural class logits and is optimized by cross-entropy.

## 3.2. Prototype-rule neurosymbolic regularization

Each class c is associated with a prototype vector $\mathbf { p } _ { c } \in \mathbb { R } ^ { H }$ . The prototypes are inactive during an initial cross-entropy warm-up. At rule activation they are initialized from the mean embedding of the labeled training samples of the corresponding class,

$$
\mathbf { p } _ { c }  \frac { 1 } {  S _ { c }  } \sum _ { i \in S _ { c } } \mathbf { z } _ { i } ,\tag{4}
$$

where $S _ { c }$ contains only training samples with label $c .$ This mean is an initialization rather than a permanent recomputation: after activation, $\mathbf { p } _ { c }$ remains a trainable model parameter and is updated jointly with the Rank-R factors and neural classifier by backpropagation.

The explicit class-level relation represented by the rule can be stated as follows: IF $y _ { i } = c \mathrm { T H E N } \ \mathbf { z } _ { i }$ should be close to $\mathbf { p } _ { c }$ and farther from competing class prototypes. The implementation grounds this relation with cosine distance. Defining normalized embeddings and prototypes as $\bar { \bf z } _ { i } = { \bf z } _ { i } / \| { \bf z } _ { i } \| .$ 2 and $\bar { \mathbf p } _ { c } = \mathbf p _ { c } / \| \mathbf p _ { c } \| _ { 2 }$ , the class-prototype distance is

$$
d _ { i c } = 1 - \bar { \bf z } _ { i } ^ { \top } \bar { \bf p } _ { c } .\tag{5}
$$

For sample $i ,$ let $d _ { i } ^ { + } = d _ { i y _ { i } }$ and let

$$
d _ { i } ^ { - } = \operatorname* { m i n } _ { c \neq y _ { i } } d _ { i c }\tag{6}
$$

denote the distance to the closest competing prototype.The implemented truth value of the rule for the labeled class is

$$
t _ { i } = \mathrm { m a x } \big ( \mathrm { e x p } \big [ { - d _ { i } ^ { + } } / { \tau } \big ] , 1 0 ^ { - 8 } \big ) ,\tag{7}
$$

and the minibatch rule loss is

$$
\mathcal { L } _ { \mathrm { r u l e } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log t _ { i } + \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \operatorname* { m a x } \bigl ( 0 , m + d _ { i } ^ { + } - d _ { i } ^ { - } \bigr ) .\tag{8}
$$

Thus the first term attracts an embedding toward its labeled-class prototype, while the second requires the nearest competing prototype to be at least margin m farther away. After warm-up, this constraint is combined with the ordinary neural cross-entropy,

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { C E } } + \lambda _ { \mathrm { r u l e } } \mathcal { L } _ { \mathrm { r u l e } } .\tag{9}
$$

Across all neurosymbolic experiments, $\lambda _ { \mathrm { r u l e } } = 0 . 1 , m = 0 . 1$ , and $\tau = 0 . 2 5$ The rule activates after 20 warm-up epochs in the standard experiments and after 60 optimization updates in the fixed-update label-scarcity study. Because the prototypes are registered model parameters before the optimizer is created, gradient updates after activation jointly modify the prototypes and the neural representation. The reported experiments use a single fixed setting for these rule hyperparameters across every dataset, split protocol, and Rank-R architecture. No dataset-specific retuning is applied, so the reported comparisons reflect the same rule strength, margin, and temperature throughout the study rather than per-dataset optimization.

## 3.3. Training and inference variants

Three matched variants isolate the two proposed mechanisms. RankR is the purely neural baseline trained with cross-entropy. The regularization - only variant (RankR-NS-RegOnly) uses the neurosymbolically trained checkpoint but evaluates it with the ordinary neural classifier; its contrast with RankR measures training - time representation shaping. The full NS variant (RankR-NS) uses the same checkpoint and adds prototype evidence at inference. Specifically, the prototype score for class c is

Table 2: Experimental variants and mechanistic interpretation.
<table><tr><td>Method</td><td></td><td>CE Rule regularization Prototype fusion</td><td></td><td>Scientific role</td></tr><tr><td>RankR</td><td>√</td><td></td><td></td><td>Purely neural Rank-R baseline</td></tr><tr><td>RankR-NS-RegOnly√</td><td></td><td>√</td><td></td><td>Training-time rule effect</td></tr><tr><td>RankR-NS</td><td>√</td><td>√</td><td>√</td><td>Regularization plus inference fusion</td></tr></table>

$$
r _ { i c } = - \frac { d _ { i c } } { \tau } ,\tag{10}
$$

and, if $\ell _ { i c }$ is the neural class logit, the fused logit is

$$
\tilde { \ell } _ { i c } = \ell _ { i c } + \beta r _ { i c } = \ell _ { i c } - \beta \frac { d _ { i c } } { \tau } .\tag{11}
$$

The experiments use $\beta = 0$ for RankR-NS-RegOnly and $\beta = 0 . 5$ for RankR-NS. Consequently, both variants use the same neurosymbolically trained parameters and prototypes, and difer only in whether the prototype-derived logits alter the final decision. The shared checkpoint prevents the fusion comparison from being confounded by a second training run.

## 4. Experimental Methodology

This section describes the experimental setup used to answer the research questions in Section 1.1. Section 4.1 introduces the four hyperspectral datasets and the input construction; Section 4.2 specifies the Rank-R architecture grid and training controls. Section 4.3 defines the spatial leakagecontrolled and stratified evaluation protocols, and Section 4.4 details the label-scarcity protocol built on the spatial split. Section 4.5 closes with the Macro-F1-based evaluation and the statistical procedures used to compare methods.

## 4.1. Datasets and input construction

Experiments use four established hyperspectral scenes: Pavia University, Indian Pines, Salinas, and Botswana<sup>1</sup>. Their standard corrected forms difer substantially in spatial extent, number of classes, and labeled sample count (Table 3), providing both urban and vegetation-dominated cases. Each labeled center pixel is represented by a 5×5 spatial patch retaining the available spectral channels.Each extracted patch is independently normalized to the [0, 1] range using min–max scaling computed over all spatial and spectral entries within that patch.

The study uses 103 bands for Pavia University, 200 for Indian Pines, 204 for Salinas, and 145 for Botswana, consistent with the commonly used corrected versions in which unusable/noisy bands have already been removed where applicable. The 5 × 5 neighborhood provides local spatial context while remaining compact enough to support the dead-zone-constrained spatial splitting strategy used below; larger neighborhoods would increase the area that must be excluded around split boundaries and would further reduce the number of feasible spatial partitions.

Table 3: Hyperspectral datasets used in the experiments. “Bands” denotes the usable channels in the corrected data used by the experiments.
<table><tr><td>Dataset</td><td>Spatial size</td><td>Bands</td><td>Classes</td><td>Labeled pixels</td><td>Patch</td></tr><tr><td>Pavia University</td><td> $6 1 0 \times 3 4 0$ </td><td>103</td><td>9</td><td>42,776</td><td> $5 \times 5$ </td></tr><tr><td>Indian Pines</td><td> $1 4 5 \times 1 4 5$ </td><td>200</td><td>16</td><td>10,249</td><td> $5 \times 5$ </td></tr><tr><td>Salinas</td><td> $5 1 2 \times 2 1 7$ </td><td>204</td><td>16</td><td>54,129</td><td> $5 \times 5$ </td></tr><tr><td>Botswana</td><td> $1 4 7 6 \times 2 5 6$ </td><td>145</td><td>14</td><td>3,248</td><td> $5 \times 5$ </td></tr></table>

## 4.2. Architecture grid and training controls

The Rank-R grid is the Cartesian product $R \in \{ 3 , 5 \}$ and hidden dimension $H \in \{ 5 0 , 7 5 \}$ , yielding R3-H50, R3-H75, R5-H50, and R5-H75. These configurations are fixed experimental conditions rather than statistical replicates. All standard runs use model seed 1, AdamW with learning rate $2 \times 1 0 ^ { - 3 }$ and weight decay $1 0 ^ { - 4 }$ , batch size 64, gradient-norm clipping with a maximum norm of 5, and at most 50 training epochs, subject to early stopping.

Model selection uses validation Macro-F1 from the ordinary neural logits, including for the NS-trained model; prototype fusion therefore does not participate in checkpoint selection. Neurosymbolic training activates the rule after 20 warm-up epochs, initializes the prototypes from one non-shufled pass over the training set, and then jointly optimizes all model parameters. Early stopping uses patience 10 after selection becomes active. The same rule parameters are used throughout: $\lambda _ { \mathrm { r u l e } } = 0 . 1$ , margin $m = 0 . 1$ , temperature $\tau = 0 . 2 5$ , and inference-fusion coeficient $\beta = 0 . 5$ for RankR-NS (zero

for RankR-NS-RegOnly).

The label-scarcity experiment uses a fixed budget of 150 optimization updates with 60 warm-up updates, batch size 64, evaluation batch size 512, learning rate $2 \times 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , and gradient-norm clipping at 5. The same four architectures and the same neurosymbolic hyperparameters are retained. Model seed 1 is fixed so that the intended source of repetition in this experiment is the support-set realization rather than repeated neural initializations.

## 4.3. Spatial and stratified evaluation protocols

The primary leakage-aware protocol constructs spatially separated train, validation, and test regions and applies a four-pixel dead-zone/bufer between split roles [22]. Because the input is a $5 \times 5$ patch, this bufer prevents patch footprints from overlapping and, thus, reduces the spatial dependence across splits. The number of feasible spatial folds is constrained by scene geometry and class coverage: the executed standard evaluation contains three folds for Pavia University and two folds each for Botswana, Indian Pines, and Salinas. Figure 2 demonstrates the proposed area separation.

Botswana required a more permissive validation-block search because of its sparse labeled geometry: the target validation fraction was reduced from 0.10 to 0.06, the minimum validation class-coverage fraction from 0.45 to 0.25, and the minimum validation-patch count from 30 to 10; smaller candidate block scales (0.55 and 0.65) were also admitted. These changes afect split feasibility only and do not alter the Rank-R or neurosymbolic objectives. With so few independent spatial partitions, these runs are summarized descriptively rather than converted into underpowered fold-level significance

![](images/2caeebb9eba9a4d052cec4b16c7573036033f9a8adff4aac7410a9ffc39489d6.jpg)

![](images/84627cc1bdf56754012e3e4ad1fe2e84353728533db96e9cadc9529ffe192480.jpg)  
(b) Indian Pines

(a) Botswana (central crop)  
![](images/aa2678cf969c911c4374dfe065551cd8f9278546a0934c17e20fb50fc031edb3.jpg)  
(c) Pavia University

![](images/0e8a805847b4fa4f12d73be1a2ff0169e5ff1e4c7a8d926b9509a71895b2eb2d.jpg)  
(d) Salinas  
Figure 2: Spatial split configurations used for leakage-controlled evaluation: (a) Botswana, (b) Indian Pines, (c) Pavia University, and (d) Salinas. Blue and green indicate the selected training and validation samples, respectively, drawn from spatially separated eligible regions. Red denotes all valid labeled samples within the spatially held-out test block. Yellow indicates the spatial dead zone introduced to prevent patch overlap between split roles. Light-gray pixels correspond to labeled samples not selected for training, validation, or testing in the illustrated fold, whereas white denotes unlabeled background. Owing to the highly elongated geometry of the Botswana scene, panel (a) shows an enlarged central crop for visual clarity; the split itself is constructed on the complete scene.

claims.

A secondary seven-fold stratified protocol is retained for comparability with conventional Rank-R evaluation. Labeled centers are disjoint between train, validation, and test roles, but nearby patches may remain spatially correlated. The standard stratified setup uses 20 training and 10 validation samples per class where feasible; Botswana retains 20 training samples but uses six validation samples per class because of its smaller class supports. Consequently, stratified and spatial results are reported separately and are not pooled as exchangeable replications. The distinction is central to RQ3: the stratified experiment asks how the method behaves under conventional sampling, whereas the spatial experiment provides the more conservative estimate of geographic generalization within a scene [12, 13].

## 4.4. Label-scarcity protocol

Because the support-set size increases with K while the optimization budget remains fixed at 150 updates, the efective number of passes through the support set is larger at small K and smaller at large K. The scarcity experiment therefore controls update budget rather than epoch budget.

The scarcity study is conducted only with spatial separation. Per-class support budgets are

$$
K \in \{ 2 , 3 , 5 , 7 , 1 0 , 1 5 , 2 0 \} .\tag{12}
$$

For each spatial fold, five support seeds (101, 202, 303, 404, and 505) generate nested support sets, so that a larger K extends rather than replaces the support ordering associated with that seed. This pairing is essential because changes with K are then less contaminated by unrelated resampling. The experiment contains 3,360 method-level evaluations (2,240 trained models, since RankR-NS and RankR-NS-RegOnly share a checkpoint) across datasets, spatial folds, support seeds, budgets, architectures, and the three method variants.

The usable scarcity-fold counts are three for Pavia University, two for Botswana, two for Salinas, and one for Indian Pines. K = 1 is intentionally excluded from the main study because a one-sample class prototype is identical to its sole support embedding and therefore ceases to represent an aggregate class-level relation.

## 4.5. Evaluation and statistical analysis

Macro-F1 is the primary response because the datasets are class imbalanced and the neurosymbolic rule is defined at class level. For each evaluation fold, Macro-F1 is calculated as the unweighted mean of the class-specific F1 scores, considering only classes with non-zero ground-truth support in that fold. Overall accuracy is retained as a secondary descriptive metric in the archived results but is not used to drive the paper’s inferential claims. All method comparisons are paired within the same split and architecture.

For the seven-fold stratified protocol, the dataset-level analysis first averages architecture-specific scores within each matched fold. A Friedman test compares the three methods; when the omnibus test is significant, paired Wilcoxon signed-rank post-hoc comparisons are performed with Holm correction [23–26]. Paired direction is summarized with mean and median ∆Macro-F1, win/tie/loss counts, and rank-biserial efect size. Architecture-specific analyses use the same gatekeeping logic. For the spatial protocol, the two or three available folds per dataset are insuficient for informative rank-based tests, so only paired magnitudes and directions are interpreted. These descriptive quantities are retained in the archived paired-comparison outputs and accompany the Wilcoxon/Holm results used for the stratified analysis.

For label scarcity, the primary descriptive quantity is the paired diference

$$
\Delta \mathrm { F } 1 ( K ) = \mathrm { M a c r o F 1 } _ { \mathrm { N S } } ( K ) - \mathrm { M a c r o F } 1 _ { \mathrm { R a n k R } } ( K ) .\tag{13}
$$

Cross-dataset summaries first average within each dataset and architecture and then weight the four datasets equally, preventing the much larger Salinas and Pavia scenes from dominating the summary. To test whether the gain changes systematically with label availability without treating support seeds as independent spatial replications, a sensitivity analysis first averages support-seed repetitions within each independent spatial fold and architecture and then estimates the slope of paired gain against $\log _ { 2 } K$ . Thus, a slope is interpretable as the change in gain for each doubling of labeled samples per class. Architecture-wise fold slopes are tested against zero with Wilcoxon signed-rank tests and Holm correction.

A mixed-efects formulation was also explored in the archived analysis. For the two primary contrasts against RankR, however, the fitted randomefect covariance collapsed to a singular boundary and the Hessian was not positive definite, producing unstable standard errors and confidence intervals. Those model-based intervals are therefore not used as evidence in this paper. The reported scarcity conclusions rely on the observed paired gains and the fold-level slope sensitivity analysis rather than on a numerically invalid fit.

## 5. Results

The results are organized to test whether the prototype-rule efect is stable across the conditions introduced in Section 4, or whether it depends on how and where it is measured. Section 5.1 reports dataset-level changes under the spatial and stratified protocols, which already show considerable heterogeneity; Section 5.2 shows that this heterogeneity persists, and in places reverses, at the level of individual architectures. Section 5.3 turns to the label-scarcity study, asking whether the efect strengthens as supervision decreases.

## 5.1. Standard evaluation: dataset-level efects

Table 5 reports architecture-averaged paired changes in Macro-F1 for the three mechanistic contrasts. Under spatial evaluation, full neurosymbolic inference improves on RankR in three of the four datasets: +8.82 percentage points (pp) on Botswana, +5.49 pp on Indian Pines, and +1.59 pp on Pavia University. Salinas is the exception, with a -0.62 pp change. The regularization-only contrast is positive in all four spatial datasets, ranging from +0.13 pp on Salinas to +7.94 pp on Botswana. The diference between RankR-NS and RankR-NS-RegOnly is comparatively small: +0.88 pp on Botswana, +0.21 pp on Indian Pines, -0.05 pp on Pavia University, and -0.76 pp on Salinas. These values answer RQ2 more directly than the end-to-end contrast alone: most of the spatial improvement is already present before prototype fusion is applied. The direction and magnitude of these spatial contrasts are summarized visually in Fig. 3.

Table 4 provides the corresponding absolute performance levels and foldto-fold dispersion, complementing the paired changes reported in Table 5.

The conventional stratified protocol exhibits the same broad heterogeneity but smaller end-to-end efects. RankR-NS is positive on Pavia University (+1.18 pp) and Botswana (+1.52 pp), essentially neutral on Indian Pines $\left( + 0 . 0 5 ~ \mathrm { p p } \right)$ , and negative on Salinas (-1.42 pp). The Pavia University Friedman test is significant $( p = 0 . 0 1 1 9 )$ , but the baseline-to-NS post-hoc contrasts do not survive Holm correction $( p _ { \mathrm { H o l m } } = 0 . 0 6 2 5 $ for both RankR-NS-RegOnly and RankR-NS). The fusion contrast is negative in all seven Pavia folds and does survive correction $( p _ { \mathrm { H o l m } } = 0 . 0 4 6 9 )$ , showing that the prototype fusion step slightly erodes the gain produced during training. Salinas also has a significant omnibus test $( p = 0 . 0 0 5 8 4 )$ ; RankR-NS is below RankR in all seven folds and the post-hoc contrast survives Holm correction $( p _ { \mathrm { H o l m } } = 0 . 0 4 6 9 )$ Botswana and Indian Pines do not pass the dataset-level Friedman gate, so no post-hoc significance claim is made for them. Figure 4 provides the corresponding dataset-level stratified contrasts.

Table 4: Absolute dataset-level Macro-F1 $( \mathrm { m e a n } \pm \mathrm { S D }$ , percentage points). For each method, architecture-specific scores are first averaged within each matched fold and the resulting fold-level values are then summarized across folds.
<table><tr><td>Protocol</td><td>Dataset</td><td>RankR</td><td> $\mathrm { R a n k R - N S - R e g O n l y }$ </td><td>RankR-NS</td></tr><tr><td rowspan="5">Spatial</td><td>Pavia University</td><td> $6 2 . 5 7 \pm 1 5 . 2 8$ </td><td> $6 4 . 2 0 \pm 1 8 . 1 5$ </td><td> $6 4 . 1 6 \pm 1 7 . 9 6$ </td></tr><tr><td>Indian Pines</td><td> $2 4 . 9 1 \pm 2 . 2 6$ </td><td> $3 0 . 1 9 \pm 0 . 7 2$ </td><td> $3 0 . 4 0 \pm 0 . 0 2$ </td></tr><tr><td>Salinas</td><td> $5 8 . 0 4 \pm 9 . 2 4$ </td><td> $5 8 . 1 8 \pm 9 . 1 5$ </td><td> $5 7 . 4 2 \pm 8 . 3 6$ </td></tr><tr><td>Botswana</td><td> $4 8 . 6 4 \pm 5 . 2 1$ </td><td> $5 6 . 5 8 \pm 1 . 0 2$ </td><td> $5 7 . 4 6 \pm 2 . 8 8$ </td></tr><tr><td>Pavia University</td><td> $6 8 . 0 4 \pm 1 . 3 6$ </td><td> $6 9 . 4 1 \pm 1 . 5 0$ </td><td> $6 9 . 2 3 \pm 1 . 6 1$ </td></tr><tr><td rowspan="4">Stratified</td><td></td><td></td><td></td><td></td></tr><tr><td>Indian Pines</td><td> $5 1 . 7 7 \pm 1 . 6 8$ </td><td> $5 2 . 3 1 \pm 1 . 0 8$ </td><td> $5 1 . 8 2 \pm 0 . 9 0$ </td></tr><tr><td>Salinas</td><td> $8 1 . 7 7 \pm 1 . 1 5$ </td><td> $8 0 . 6 4 \pm 2 . 0 0$ </td><td> $8 0 . 3 5 \pm 1 . 9 7$ </td></tr><tr><td>Botswana</td><td> $8 4 . 3 6 \pm 1 . 8 3 $ </td><td> $8 5 . 8 3 \pm 0 . 5 7$ </td><td> $8 5 . 8 9 \pm 0 . 7 2$ </td></tr></table>

Table 5: Dataset-level paired change in Macro-F1 (percentage points). Architecture scores are averaged within matched folds before contrasts are formed. Positive values favor the method named first. An asterisk (\*) denotes a Holm-corrected post-hoc $p \ < \ 0 . 0 5$ in the seven-fold stratified analysis. No inferential tests are reported for the two/three-fold spatial protocol.
<table><tr><td>Protocol</td><td>Dataset</td><td>RankR-NS-RegOnly-RankR</td><td>RankR-NS-RankR</td><td>RankR-NS-RankR-NS-RegOnly</td></tr><tr><td rowspan="4">Spatial</td><td>Pavia University</td><td>+1.63</td><td>+1.59</td><td>-0.05</td></tr><tr><td>Indian Pines</td><td>+5.28</td><td>+5.49</td><td>+0.21</td></tr><tr><td>Salinas</td><td>+0.13</td><td>-0.62</td><td>-0.76</td></tr><tr><td>Botswana</td><td>+7.94</td><td>+8.82</td><td>+0.88</td></tr><tr><td rowspan="5">Stratified</td><td>Pavia University</td><td>+1.36</td><td>+1.18</td><td>-0.18*</td></tr><tr><td>Indian Pines</td><td>+0.54</td><td>+0.05</td><td>-0.49</td></tr><tr><td>Salinas</td><td>-1.13</td><td>-1.42*</td><td>-0.29</td></tr><tr><td>Botswana</td><td>+1.46</td><td>+1.52</td><td>+0.06</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

## 5.2. Architecture dependence

The dataset averages conceal substantial architecture dependence (Fig. 5). Pavia University is the clearest example under spatial evaluation: the regularization efect is negative for R = 3 (mean changes of -3.56 pp at H = 50 and -1.71 pp at $H = 7 5 )$ but positive for R = 5 (+3.60 and +8.20 pp, respectively). This reversal explains why the dataset - level mean is only moderately positive despite large gains in some configurations. Botswana is also heterogeneous in magnitude, with particularly large spatial gains in R3-H50, but its dataset - level direction remains positive.

The stratified architecture analysis likewise contains efects that are diluted by architecture averaging. For Botswana with $R = 5 , H = 7 5$ , both RankR-NS-RegOnly and RankR-NS improve over RankR in all seven folds; the mean gains are +2.73 and +3.02 pp, and both paired comparisons survive Holm correction $( p _ { \mathrm { H o l m } } ~ = ~ 0 . 0 4 6 9 )$ . Conversely, Pavia University at

![](images/81f3de5ba6aa604045ee76000359300f5c107b22b8e5b3f6346d724ff32b598a.jpg)  
Figure 3: Paired dataset-level Macro-F1 changes under the spatially separated protocol. The small number of spatial folds is treated descriptively; the figure emphasizes efect direction and magnitude rather than fold-level significance.

R = 5, H = 75 shows a significant negative fusion contribution with the same corrected p-value. These results answer RQ5: the prototype-rule efect is not solely dataset-specific; it interacts with the capacity/structure of the Rank-R representation.

## 5.3. Scarcity impact

The scarcity experiment does not support a simple monotonic statement that neurosymbolic regularization becomes progressively more useful as labels disappear. After equal weighting of the four datasets and averaging across the four architectures, the RankR-NS gain over RankR is positive at every tested budget but varies non-monotonically: +0.53 pp at K = 2, +0.56 at $K = 3$ 2 $+ 0 . 9 7$ at K = 5, +0.21 at K = 7, +0.61 at K = 10, +0.46 at $K = 1 5$ , and $+ 0 . 4 9$ at K = 20 (Table 6). RankR-NS-RegOnly is also positive on average at all budgets, with smaller gains of $+ 0 . 2 4 - + 0 . 6 2 \ \mathrm { p p }$ . Taking the diference between the two rows of Table 6, the average incremental contribution of fusion is modest, ranging from −0.03 pp at K = 7 to +0.35 pp at $K = 5$ The equal-dataset-weighted full-neurosymbolic trajectory is shown in Fig. 6.

![](images/87a7b9b4dc300e87e7ffd649f8d3fde6f16ebf15334dfd5d9079e150aa93d672.jpg)  
Figure 4: Paired dataset-level Macro-F1 changes under seven-fold stratified evaluation. Post-hoc testing is performed only when the dataset-level Friedman omnibus test is significant.

Dataset-level behavior explains the irregular aggregate curve. Botswana consistently benefits from RankR-NS, with architecture-averaged gains between +1.10 and +2.92 pp across all seven budgets. Pavia University is positive from K = 2 through $K = 1 0 \ ( + 0 . 2 1 \ \mathrm { t o } \ + 0 . 7 8 \ \mathrm { p p } )$ , is approximately neutral at $K = 2 0 ~ ( + 0 . 0 3 ~ \mathrm { p p } )$ , and is slightly negative at $K = 1 5$ (-0.20 pp). Indian Pines is mixed, ranging from -0.27 to +1.02 pp depending on K. Salinas is negative at six of seven budgets; its only positive architectureaveraged value is at $K = 5 \ ( + 0 . 5 1 \ \mathrm { p p } )$ . The same dataset dependence was already visible in the standard evaluation, indicating that label budget alone

![](images/eda45f5606f64ceb893f4b693658dcfe4018ba35a953337865ad74ab63680ff7.jpg)  
Figure 5: Architecture-level directional Macro-F1 efects for spatial (left) and stratified (right) evaluation. The heatmaps show that architecture averaging can conceal changes in both magnitude and direction.

Table 6: Equal-dataset-weighted paired Macro-F1 gain over RankR across the label budgets. Values are percentage points and are averaged over the four Rank-R architectures after dataset-level aggregation.
<table><tr><td>K</td><td>2</td><td>3</td><td>5</td><td>7</td><td>10</td><td>15</td><td>20</td></tr><tr><td>RankR-NS-RankR</td><td>0.53</td><td>0.56</td><td>0.97</td><td>0.21</td><td>0.61</td><td>0.46</td><td>0.49</td></tr><tr><td>RankR-NS-RegOnly-RankR</td><td>0.31</td><td>0.43</td><td>0.62</td><td>0.24</td><td>0.38</td><td>0.39</td><td>0.42</td></tr></table>

is not the dominant moderator.

The fold-level slope sensitivity analysis provides a more direct test of RQ4. For RankR-NS versus RankR, 22 of 32 architecture-by-spatial-fold slopes are negative and 10 are positive when gain is regressed on $\log _ { 2 } K$ . A negative slope means the NS gain shrinks as labels increase, i.e., the gain is larger when labels are scarcer.

The mean slopes are -0.137, -0.039, -0.009, and -0.156 pp per doubling of K for R3-H50, R3-H75, R5-H50, and R5-H75, respectively. R5-H75 has seven negative slopes out of eight, but its Holm-corrected Wilcoxon p-value is $p _ { \mathrm { H o l m } } = 0 . 2 1 8 8$ ; the other three architecture-wise tests are also non-significant after correction. Thus the directional imbalance is suggestive, but the experiments do not establish a general label-budget trend at the conventional 0.05 level.

Figure 7 makes the heterogeneity explicit: the same architecture can change direction across datasets and budgets, and no single low-K regime dominates uniformly.

![](images/c2c5f7dcaab4bd2a0769a9213e9a14af07730c4be9b68c69f582555c2d4aa7cd.jpg)  
Figure 6: Equal-dataset-weighted full-neurosymbolic Macro-F1 gain over RankR across label budgets and Rank-R architectures. The overall gain is usually positive but not monotonic in K.

## 6. Discussion

In this section we discuss the results in relation to the five research questions. Section 6.1 addresses RQ1 and RQ3, arguing that the neurosymbolic efect is conditional rather than universal and depends on the evaluation protocol. Section 6.2 addresses RQ2, attributing most of the efect to trainingtime regularization rather than inference-time fusion, while Section 6.3 addresses RQ4 and RQ5, showing that the efect of label scarcity is not monotonic and interacts with dataset and architecture. Section 6.4 clarifies the intended, restricted sense in which the mechanism is called neurosymbolic, and Section 6.5 states the study’s limitations.

![](images/c43643c6f9538d500098d70004f7a85d646da84da3e380ebe33d773d95605332.jpg)  
Figure 7: Dataset-specific RankR-NS gain over RankR across label budgets. The principal result is heterogeneity rather than a universal scarcity curve.

## 6.1. A conditional rather than universal neurosymbolic efect

The experiments answer RQ1 with a qualified result. Prototype-rule regularization can materially improve a Rank-R model, but the sign and magnitude of the efect depend on the dataset and architecture. This is most visible in the spatial evaluation, where the end-to-end RankR-NS efect ranges from +8.82 pp on Botswana to -0.62 pp on Salinas. The result is therefore stronger than a claim of negligible average change but weaker than a claim of universal benefit. Such heterogeneity is also visible in the stratified protocol, where Salinas is the only dataset for which full NS is consistently below RankR in all seven folds and the post-hoc comparison survives Holm correction.

The fact that protocol changes can alter the magnitude and, for some contrasts, the apparent direction is also important for RQ3. Stratified folds provide more replications and support non-parametric testing, but they do not reproduce the geographic independence of the spatial folds. The spatial experiment should therefore be treated as the primary evidence about leakagecontrolled within-scene generalization, while the stratified experiment serves as a conventional comparison. A method whose advantage is visible only under one sampling scheme would warrant caution; here, the broad dataset pattern (strong Botswana, positive Indian/Pavia in several settings, weak or negative Salinas) persists suficiently to indicate genuine dataset dependence rather than a single protocol artifact.

## 6.2. Training-time regularization is the main mechanism

RQ2 is addressed by the shared-checkpoint ablation. Across the spatial datasets, RankR-NS-RegOnly already accounts for most of the diference from RankR. Full inference fusion adds less than one percentage point in absolute Macro-F1 on every dataset and is negative on Pavia University and Salinas. Under stratified evaluation, fusion is negative in all seven Pavia folds and significantly reduces the regularization-only result after Holm correction. In the label-scarcity study, the equal-weighted diference between RankR-NS and RankR-NS-RegOnly is likewise small relative to the full contrast with RankR.

This pattern suggests that the primary value of the prototype rule is not a second classifier layered on top of the network. Instead, it acts mainly as a training constraint on the geometry of the latent representation. This interpretation is consistent with the intended complementarity of the two inductive biases: the Rank-R decomposition restricts the tensor mapping, while the prototype rule constrains class organization in the embedding space. The inference fusion can still help in particular regimes, but the results do not justify treating it as uniformly beneficial.

## 6.3. Label scarcity does not produce a simple monotonic response

RQ4 yields a more nuanced result than the initial hypothesis that symbolic structure should become increasingly valuable as supervision decreases. The cross-dataset gain is positive at every tested K, and 22 of 32 fold-level slopes point toward a larger advantage at lower budgets. Nevertheless, the aggregate curve is non-monotonic and none of the architecture-specific slope tests survives Holm correction. The strongest directional pattern occurs for R5-H75 (seven of eight negative slopes), but even there the corrected evidence is insuficient for a general claim.

The more defensible interpretation is therefore that scarcity can expose useful prototype regularization in some configurations, but it does not act as a single control variable that determines the efect. Dataset geometry, class compactness, architecture, and the quality of prototypes estimated from very small supports all plausibly interact with K. Salinas is instructive: reducing the label budget does not reliably turn a negative standard-evaluation efect into a positive one. Conversely, Botswana remains favorable across essentially the entire scarcity range. These observations make RQ5 at least as important as RQ4.

## 6.4. Why call the mechanism neurosymbolic?

The prototype itself is a learned numerical parameter and is not symbolic merely because it is initialized from an average embedding. The neurosymbolic element is the explicit relation imposed on those grounded class symbols: an embedding labeled as class c should satisfy a compatibility relation with the prototype representing c and should be separated from prototypes representing alternative classes. That relation is named, inspectable, and added to the neural objective as a diferentiable constraint. The relation is only as meaningful as the grounding of the symbols it refers to [27]. Here, that grounding is minimal, since each class symbol is anchored solely to the mean of the labelled embeddings of that class and is thereafter free to drift with the representation. Because the anchoring comes from class labels rather than from unsupervised concept discovery, the rule can be satisfied only through class-consistent geometry. This is a deliberately modest use of the term neurosymbolic: the framework does not claim discrete reasoning or external domain knowledge, but it does separate a declarative class relation from the unconstrained neural classification objective.

## 6.5. Limitations and implications

Several limitations bound the conclusions. First, the standard experiments primarily use one model seed, so the independent spatial partitions provide stronger evidence about sampling variation than about variation due to neural initialization. Second, only two or three spatial folds are feasible for most datasets, and the scarcity study has only one independent Indian Pines spatial fold; support seeds improve characterization of support selection but are not substitutes for independent spatial replications. Third, a single prototype per class favors compact class geometry and may be inappropriate for multimodal classes. Recent few-shot HSI work has similarly questioned prototypes derived solely from a small number of support samples and explored trainable semantic anchors to enrich prototype representations [28]. Fourth, the neurosymbolic hyperparameters are held fixed across datasets rather than tuned separately, which is useful for controlled comparison but may understate achievable performance in some scenes. Fifth, the knowledge grounding is induced from labeled embeddings rather than supplied by an external ontology or expert knowledge base. Finally, the study covers four hyperspectral datasets and four Rank-R architectures; it characterizes this mechanism rather than neurosymbolic learning in general.

The failed mixed-efects diagnostics also motivate a methodological point. Repeated support-set realizations create a rich empirical surface, but treating them as if they supplied many independent spatial units can produce unstable hierarchical fits. The fold-level sensitivity analysis used here is more conservative: it sacrifices nominal sample size in exchange for an inferential unit that better matches the spatial generalization question.

## 7. Conclusion

This work evaluates a simple diferentiable prototype rule as a second inductive bias for Rank-R tensor neural networks. The evidence does not support a universal neurosymbolic advantage. Under spatial leakage control, full NS improves architecture-averaged Macro-F1 on Botswana (+8.82 pp), Indian Pines (+5.49 pp), and Pavia University (+1.59 pp), but decreases it on Salinas (-0.62 pp). The regularization-only ablation preserves most of the positive efect, whereas prototype fusion contributes only small, datasetdependent changes and can be harmful. Stratified evaluation confirms the heterogeneity and yields corrected evidence of a full-NS degradation on Salinas and a fusion penalty on Pavia University.

The dedicated label-scarcity study further shows that the NS gain is positive on average across all tested budgets but is not monotonic in the number of labels. Although 22 of 32 fold-level slopes are directionally consistent with a larger advantage at lower K, none of the architecture-wise slope tests remains significant after Holm correction. The principal conclusion is therefore mechanistic rather than competitive: prototype-rule regularization can complement the structural bias of Rank-R learning, primarily by shaping the learned representation, but its utility is conditioned by dataset, architecture, and evaluation regime. Future work should test multi-prototype or structured expert-grounded rules and evaluate them with more independent spatial repetitions and neural seeds.

## Acknowledgment

Open access publication of this article was funded by HEAL-Link under an agreement between Elsevier and HEAL-Link.

## References

[1] K. Makantasis, A. D. Doulamis, N. D. Doulamis, A. Nikitakis, Tensorbased classification models for hyperspectral data analysis., IEEE Trans. Geosci. Remote. Sens. 56 (12) (2018) 6884–6898.

[2] K. Makantasis, A. Voulodimos, A. Doulamis, N. Doulamis, I. Georgoulas, Hyperspectral image classification with tensor-based rank-r learning models, in: 2019 IEEE International Conference on Image Processing (ICIP), IEEE, 2019, pp. 3148–3152.

[3] K. Makantasis, A. Georgogiannis, A. Voulodimos, I. Georgoulas, A. Doulamis, N. Doulamis, Rank-r fnn: A tensor-based learning model for high-order data classification, IEEE Access 9 (2021) 58609–58620. doi:10.1109/ACCESS.2021.3072973.

[4] T. G. Kolda, B. W. Bader, Tensor decompositions and applications, SIAM Review 51 (3) (2009) 455–500. doi:10.1137/07070111X.

[5] I. Georgoulas, E. Protopapadakis, K. Makantasis, D. Seychell, A. Doulamis, N. Doulamis, Graph-based semi-supervised learning with tensor embeddings for hyperspectral data classification, IEEE Access 11 (2023) 124819–124832. doi:10.1109/ACCESS.2023.3328388.

[6] S. Roychowdhury, M. Diligenti, M. Gori, Regularizing deep networks with prior knowledge: A constraint-based approach, Knowledge-Based Systems 222 (2021) 106989. doi:10.1016/j.knosys.2021.106989.

[7] S. Badreddine, A. d’Avila Garcez, L. Serafini, M. Spranger, Logic tensor networks, Artificial Intelligence 303 (2022) 103649. doi:10.1016/j. artint.2021.103649.

[8] P. Hitzler, A. Eberhart, M. Ebrahimi, M. K. Sarker, L. Zhou, Neurosymbolic approaches in artificial intelligence, National Science Review 9 (6) (2022) nwac035. doi:10.1093/nsr/nwac035.

[9] Z. Li, Y. Huang, Z. Li, Y. Yao, J. Xu, T. Chen, X. Ma, J. Lu, Neurosymbolic learning yielding logical constraints, Advances in Neural Information Processing Systems 36 (2023) 21635–21657.

[10] J. Snell, K. Swersky, R. S. Zemel, Prototypical networks for fewshot learning, in: Advances in Neural Information Processing Systems, Vol. 30, 2017.

[11] C. Ding, M. Zheng, S. Zheng, Y. Xu, L. Zhang, W. Wei, Y. Zhang, Integrating prototype learning with graph convolution network for effective active hyperspectral image classification, IEEE Transactions on Geoscience and Remote Sensing 62 (2024) 1–16.

[12] H. Feng, Y. Wang, Z. Li, N. Zhang, Y. Zhang, Y. Gao, Information leakage in deep learning-based hyperspectral image classification: A survey, Remote Sensing 15 (15) (2023) 3793. doi:10.3390/rs15153793.

[13] X. Cao, C. Li, J. Feng, L. Jiao, Semi-supervised feature learning for disjoint hyperspectral imagery classification, Neurocomputing 526 (2023) 9–18. doi:10.1016/j.neucom.2023.01.054.

[14] W. Wang, Y. Yang, F. Wu, Towards data-and knowledge-driven ai: A survey on neuro-symbolic computing, IEEE Transactions on Pattern Analysis and Machine Intelligence 47 (2) (2025) 878–899. doi:10.1109/ TPAMI.2024.3483273.

[15] G. Marra, S. Dumančić, R. Manhaeve, L. De Raedt, From statistical relational to neurosymbolic artificial intelligence: A survey, Artificial Intelligence 328 (2024) 104062. doi:https:

//doi.org/10.1016/j.artint.2023.104062.

URL https://www.sciencedirect.com/science/article/pii/ S0004370223002084

[16] D. Zhang, Y. Ren, C. Liu, Z. Han, J. Wang, Weighted contrastive prototype network for few-shot hyperspectral image classification with noisy labels, Remote sensing 16 (18) (2024) 3527.

[17] A. Qin, C. Yuan, Q. Li, X. Luo, F. Yang, T. Song, C. Gao, Few-shot learning with prototype rectification for cross-domain hyperspectral im age classification, IEEE Transactions on Geoscience and Remote Sensing 62 (2024) 1–15. doi:10.1109/TGRS.2024.3414392.

[18] H. Tang, Y. Wu, H. Li, D. Tang, X. Yang, W. Xie, Global–local prototype-based few-shot learning for cross-domain hyperspectral image classification, Knowledge-Based Systems 314 (2025) 113199. doi:https://doi.org/10.1016/j.knosys.2025.113199. URL https://www.sciencedirect.com/science/article/pii/ S0950705125002461

[19] L. N. DeLong, R. F. Mir, J. D. Fleuriot, Neurosymbolic ai for reasoning over knowledge graphs: A survey, IEEE Transactions on Neural Networks and Learning Systems 36 (5) (2024) 7822–7842.

[20] P. Armary, C. B. El-Vaigh, O. L. Narsis, C. Nicolle, Ontology learning towards expressiveness: A survey, Computer Science Review 56 (2025) 100693.

[21] D. Fišer, Á. Torralba, J. Hofmann, Boosting optimal symbolic planning: Operator-potential heuristics, Artificial Intelligence 334 (2024) 104174.

[22] G. D. Gkologkinas, E. Protopapadakis, A. Stamou, I. Tavantzis, A. Dosiou, I. Skalidi, E. Stylianidis, Spatially robust land cover classification with multi-seasonal sentinel-2 imagery: A comparison of cnn, unet++, convnext and vit, Remote Sensing 18 (15) (2026) 2463.

[23] M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (200) (1937) 675–701. doi:10.1080/01621459.1937. 10503522.

[24] F. Wilcoxon, Individual comparisons by ranking methods, Biometrics Bulletin 1 (6) (1945) 80–83. doi:10.2307/3001968.

[25] S. Holm, A simple sequentially rejective multiple test procedure, Scandinavian Journal of Statistics 6 (2) (1979) 65–70.

[26] J. Demšar, Statistical comparisons of classifiers over multiple data sets, Journal of Machine Learning Research 7 (2006) 1–30.

[27] E. Marconato, G. Bontempo, E. Ficarra, S. Calderara, A. Passerini, S. Teso, Neuro-symbolic continual learning: Knowledge, reasoning shortcuts and concept rehearsal, in: Proceedings of the 40th International Conference on Machine Learning, 2023, pp. 23915–23936.

[28] H. Yan, J. He, X. Zhang, Z. Li, Semantic anchor enhanced prototype learning for few-shot hyperspectral image classification, Knowledge-Based Systems 341 (2026) 115822. doi:https:

//doi.org/10.1016/j.knosys.2026.115822.

URL https://www.sciencedirect.com/science/article/pii/ S0950705126005484