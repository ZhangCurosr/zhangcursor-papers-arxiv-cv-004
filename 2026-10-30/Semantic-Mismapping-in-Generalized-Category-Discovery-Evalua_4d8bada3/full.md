# Semantic Mismapping in Generalized Category Discovery: Evaluation and Mitigation

Zitai Wang, Qianqian Xu\*, Senior Member, IEEE, Cong Hua, Xi Chen, Zhiyong Yang, Xiaochun Cao, Senior Member, IEEE, and Qingming Huang\*, Fellow, IEEE

Abstract—Generalized Category Discovery (GCD) aims to perform old-category recognition and novel-category discovery in the same unlabeled set. Early methods address both objectives through clustering and evaluate the exchangeable predicted indices with global Hungarian (GH) matching. Recent methods adopt trainable parametric heads, using supervision for old-category recognition and discovery-oriented objectives for new categories. In this way, the output space becomes partially exchangeable: old indices are semantically anchored, while new indices remain exchangeable. However, these methods still apply global matching, which can reassign anchored old indices to other categories and hide some prediction errors. Empirically, we are surprised to observe that this reassignment can induce an overestimation of new-category accuracy by up to 25.63 points. We term this phenomenon semantic mismapping and further derive a permutation-cycle condition that characterizes when semantic mismapping can increase the global matching score. To fix this issue, we propose Semantics-Preserving Evaluation (SPE), which preserves the supervised semantics of old indices and restricts Hungarian matching to the new region. Furthermore, we propose Mismapping-Mitigating Optimization (MMO) to alleviate the prediction errors that can give rise to semantic mismapping, combining a region-balance loss (RBL) for old/new routing with a novel-diversity loss (NDL) that encourages orthogonality among new prototypes. Theoretically, we show that minimizing the two losses can tighten an upper bound on spurious accuracy gains from semantic mismapping, with controllable residual terms. Empirical results under both GH and SPE demonstrate the effectiveness of our proposed method.

Index Terms—Generalized category discovery, evaluation protocol, semantic mismapping.

## 1 INTRODUCTION

time are covered by the labeled training set. However, in real-world scenarios, a deployed recognition system may receive samples from known categories together with samples from categories that have never been labeled before [1, 2]. In view of this, Generalized Category Discovery (GCD) considers a more challenging open-world setting, where a model is trained with labeled samples from old categories and unlabeled samples drawn from both old and new categories [3]. This setting gives rise to two objectives that typically belong to different learning paradigms: (O1) recognizing old categories in the unlabeled samples (i.e., classification) and (O2) discovering underlying clusters for new categories (i.e., clustering). Hence, it becomes a core question in GCD: how to unify the two distinct objectives into a single framework?

Early GCD methods adopt a unified clustering-based framework, learning visual features from both labeled and unlabeled data and then applying non-parametric clustering [3, 4, 5]. The induced cluster indices are exchangeable as their semantics have been lost. Hence, the standard GCD evaluation protocol aligns these indices with ground-truth categories through Hungarian matching [6] and then calculates accuracy.

Recent advances, represented by SimGCD [7] and its follow-ups [8, 9, 10], turn to a divide-and-conquer parametric framework, achieving superior performance. These methods replace the non-parametric clustering procedure with trainable classifier or prototype heads. For old categories (O1), the parametric head is effectively trained with the supervision from labeled samples. For new categories (O2), the corresponding indices are not tied to fixed categories and are trained with discovery-oriented objectives, such as self-distillation [7], pseudo-labeling [8], and contrastive learning [11]. Overall, the output space becomes partially exchangeable: old indices are semantically anchored, while new indices remain exchangeable.

Although effective, this divide-and-conquer framework can exhibit notorious prediction bias. Since old categories receive more direct supervision, the parametric head tends to produce an imbalanced distribution across old and new categories [7]. A typical failure is false-old, where many new samples are predicted as old categories. To address this bias, prior work has designed a series of training objectives and regularizers [7, 9, 12, 13].

![](images/c84c105a21d97b2735bcbb60bc7d0f73e51ff0534ff3c252aa7102dde2ff5f36.jpg)  
Fig. 1. (a) When evaluating parametric GCD methods, global matching could reassign an old output to a new category (e.g., o → n ), making a false-old prediction appear correct. (b) Our SPE protocol preserves the semantics of old outputs and performs local matching within the new region. (c) Our mismapping-mitigating optimization (MMO) combines region-balance loss (RBL), which matches old/new prediction mass to estimated sample proportions, with novel-diversity loss (NDL), which encourages orthogonality among new prototypes.

Despite their progress, parametric methods still apply Hungarian matching to the entire output space. However, we find that such global matching can reassign semantically anchored old indices to new categories, making some falseold predictions appear correct. As illustrated in Fig.1(a), a new bird with an incorrect old-category prediction is reassigned to its ground-truth new category. In view of this, we call any assignment that overrides an anchored index semantic mismapping, which may occur both across the old/new boundary and within the old region. Theoretically, we further derive a permutation-cycle condition that characterizes when semantic mismapping can increase the global matching score. For fine-grained datasets, such as Herbarium 19, this mismapping can induce an overestimation of new accuracy by 25.63 points, whereas for coarsegrained datasets, such as CIFAR-10 and ImageNet-100, its effect becomes negligible. This observation raises a natural question:

## How should parametric GCD methods be evaluated to avoid semantic mismapping?

To answer this question, we propose a divide-andconquer protocol named Semantics-Preserving Evaluation (SPE). As illustrated in Fig.1(b), SPE preserves the semantics of old outputs and performs local matching within the new region, while predictions crossing the old/new boundary remain errors. This design directly measures old-category recognition for (O1) while preserving the permutation invariance required for novel-category discovery under (O2). We further extend SPE to non-parametric GCD methods and settings with an estimated number of new categories.

Motivated by our analysis, we propose Mismapping-

Mitigating Optimization (MMO), which combines two plugand-play regularizers following the divide-and-conquer principle, as summarized in Fig.1(c). On the one hand, a region-balance loss (RBL) explicitly balances prediction mass across the old and new regions. On the other hand, a novel-diversity loss (NDL) promotes new-category discrimination by encouraging orthogonality among new prototypes. We further derive two bounds showing how the two losses can suppress spurious accuracy gains when residual errors are controlled. These bounds provide sufficient conditions for GH and SPE to yield the same overall accuracy. Finally, extensive experiments on several challenging finegrained benchmarks validate the necessity of SPE and the effectiveness of the proposed regularizers. Overall, our contributions are summarized as follows:

• New insight. We identify semantic mismapping in parametric GCD and theoretically characterize when it produces spurious accuracy gains under GH.

• New protocol. Building on this insight, we propose SPE to prevent semantic mismapping from hiding prediction errors, enabling faithful evaluation of old-category recognition and novel-category discovery.

• New method. To mitigate the underlying prediction errors, we propose MMO with two plug-and-play losses, supported by theoretical bounds and extensive experiments.

## 2 RELATED WORK

## 2.1 Category-Discovery Settings

Category discovery aims to identify previously unseen categories in unlabeled data by transferring knowledge from labeled old categories [14, 15, 16].

As one of the most representative settings, Novel Category Discovery (NCD) provides labeled old-category data alongside an unlabeled discovery set containing only new categories [17, 18, 19, 20, 21]. Mainstream NCD methods combine self-supervised learning with old-category supervision to improve the clustering of unlabeled samples [18, 22, 23, 24, 25]. Class-incremental NCD extends this setting by first learning on labeled old-category data and then receiving unlabeled data for novel-category discovery [26, 27]. Generalized Category Discovery (GCD) is a more challenging setting, where old and new categories coexist in the unlabeled set [3, 28]. Subsequent studies investigate fine-grained discovery [29, 30, 31], domain shifts [32], and continuous discovery [33]. OpenGCD integrates generalized category discovery into open-world recognition [34, 35] to assist in grouping unlabeled samples for subsequent incremental learning [36].

This paper focuses on the standard GCD setting, studying how to faithfully evaluate and jointly improve oldcategory recognition and novel-category discovery.

## 2.2 Evaluation Protocols for Category Discovery

Evaluation protocols for the aforementioned settings mainly differ in how they emphasize old-category recognition and novel-category discovery. Early NCD studies focus on clustering accuracy for new categories [17, 18]. Later NCD studies evaluate both old and new categories, distinguishing task-aware and task-agnostic protocols according to whether the model is told if each sample belongs to an old or a new category [37, 38, 39]. Class-incremental NCD also measures how well old-category recognition is maintained as the model learns new categories [26, 27]. Notably, FRoST also identifies old–new errors hidden by Hungarian matching and preserves old-category semantics during NCD evaluation [27]. However, its new-category mapping is established on new-only training data, leaving evaluation on the mixed unlabeled set in GCD unaddressed.

To handle the coexistence of old and new categories, GCD extends NCD evaluation by applying a global Hungarian matching to the whole unlabeled set [3]. This clusteringbased protocol remains widely used by parametric GCD methods, providing a common basis for comparison with non-parametric approaches [7, 12]. µGCD also observes that Hungarian matching can report nonzero new-category accuracy even when no samples are predicted to new outputs [8]. However, its evaluation still follows global matching, without preserving anchored old-output semantics. We systematically study this mismatch in standard GCD, characterize matching gains through permutation cycles, quantify their effects on empirical performance and method rankings, and propose SPE for faithful evaluation.

Recent work also highlights the distinction between clustering accuracy and semantic classification. SECOS uses category names and vision-language knowledge to classify both old and new categories, without post-hoc matching [40]. In standard GCD, however, new categories are expected to be discovered without their names being given. Hence, our work focuses on evaluation in the standard GCD setting.

## 2.3 GCD Methods and Prediction Bias

Early GCD approaches learn transferable features from both labeled and unlabeled data, followed by non-parametric clustering [3, 4, 5]. Further studies improve feature adaptation through spatial prompts or textual knowledge [10, 41, 42]. Parametric methods replace non-parametric clustering with trainable classifiers or prototypes [7, 8, 12]. This divide-and-conquer paradigm achieves state-of-the-art performance, as the parametric classifier leverages old-category supervision more effectively.

However, this parametric paradigm can also lead to prediction bias, as old and new categories rely on different learning signals. Direct supervision can bias predictions toward old categories [7], while overconfidence in new categories can also hinder old-category recognition accuracy [43]. To mitigate this bias, existing methods apply constraints at three levels. At the prediction level, existing methods use entropy and class-wise regularization $[ 7 ,$ 13], consistency constraints [9], or auxiliary debiased classifiers [43, 44] to balance old- and new-category predictions. At the feature level, existing methods use prototype separation [12], feature-geometry regularization [45], or relational distillation [46] to improve category discrimination. At the gradient level, EAGC coordinates supervised and unsupervised gradients to reduce conflicts between old-category recognition and novel-category discovery [47].

Our work instead examines prediction bias at the evaluation level, revealing how global matching can hide the resulting errors. Guided by this insight, we design regularizers at the prediction and feature levels.

## 3 PRELIMINARIES

## 3.1 Generalized Category Discovery

Let $\mathcal { D } _ { l } ~ = ~ \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { l } }$ and $\mathcal { D } _ { u } ~ = ~ \{ { \pmb x } _ { j } \} _ { j = 1 } ^ { N _ { u } }$ denote the labeled and unlabeled training sets, respectively. The labels in $\mathcal { D } _ { l }$ belong to a set of old categories $\mathcal { D } _ { \mathrm { o l d } } ,$ , whereas $\mathcal { D } _ { u }$ contains samples from both $\mathcal { V } _ { \mathrm { o l d } }$ and a set of new categories $\mathcal { V } _ { \mathrm { n e w } } .$ . The two category sets are disjoint, and together form the target space $\mathcal { Y } \overset { \sim } { = } \dot { \mathcal { y } } _ { \mathrm { o l d } } \cup \mathcal { Y } _ { \mathrm { n e w } }$ . Given $\mathcal { D } _ { l }$ and $\mathcal { D } _ { u } ,$ GCD aims to learn a model $f : \mathcal { X }  \widehat { \mathcal { Y } }$ , where $\widehat { \mathcal { V } }$ denotes the predicted output indices. Unless otherwise specified, we assume that the total number of target categories is given and set $| \widehat { \mathcal { V } } | = | \mathcal { V } |$ , following the common practice [3, 7]. When this number is unknown, $| \widehat { \mathcal { V } } |$ can be estimated via offthe-shelf methods [3, 48]. During evaluation, f is applied to each sample in $\mathcal { D } _ { u } ,$ recognizing old-category samples and assigning new-category samples to proper category indices.

Early GCD methods follow a unified clustering-based framework [3, 4, 5]. Specifically, the model can be formulated as $f = \psi \circ \phi ,$ where $\phi : \mathcal { X } \overset { \cdot } {  } \mathbb { R } ^ { d }$ is a feature extractor and $\psi : \mathbb { R } ^ { d }  \widehat { \mathcal { V } }$ is a non-parametric clustering procedure. In this framework, $\widehat { \mathcal { V } }$ is fully exchangeable and will be aligned with the ground-truth categories during evaluation.

Recent methods instead adopt a divide-and-conquer parametric framework [7, 8, 9, 10]. These methods replace ψ with a trainable classifier or prototype head $h : \mathbb { R } ^ { d } \overset { \cdot } {  } \widehat { \mathcal { V } } _ { \varepsilon }$ , so that the model can be formulated as $f = h \circ \phi .$ In this case, the output space is partitioned as $\widehat { \mathcal { Y } } \ = \ \widehat { \mathcal { Y } } _ { \mathrm { o l d } } \cup \widehat { \mathcal { Y } } _ { \mathrm { n e w } }$ and is only partially exchangeable: the old-category dimensions in $\widehat { \mathcal { V } } _ { \mathrm { o l d } }$ are semantically anchored by the supervision from $\mathcal { D } _ { l } ,$ while the new-category dimensions in $\widehat { \mathcal { V } } _ { \mathrm { n e w } }$ remain exchangeable and are learned through discovery-oriented objectives defined on $\mathcal { D } _ { u }$

## 3.2 Standard GCD Evaluation Protocol

The standard GCD protocol treats every predicted index as exchangeable and finds one assignment over $\mathcal { D } _ { u }$ with the

![](images/a883e3d140d707564f34eb641afe8cf188287e578e7bdeb2e70b5d8a27ffe656.jpg)  
Fig. 2. Comparison of GH and SPE using the same predictions. (a) The raw confusion matrix has predicted outputs as rows and groundtruth categories as columns, with old and new categories denoted by o and n. (b) GH reassigns the anchored old output $\hat { o } _ { 1 }$ to $n _ { 1 }$ and the new output nˆ to $o _ { 1 }$ , counting 46 false-old predictions and 44 old-to-new errors as correct. (c) SPE preserves the semantics of old outputs and applies local matching within the new region. In this way, predictions in either cross-region block remain errors, including the 46 and 44 samples hidden by GH.

Hungarian algorithm $[ 3 , 6 ] .$ . Let Π be the set of one-to-one assignments from $\widehat { \mathcal { V } }$ to Y. The evaluator has access to the ground-truth category $y ( \pmb { x } )$ and obtains

$$
\mathcal { A } _ { \mathrm { G H } } \in \arg \operatorname* { m a x } _ { \mathcal { A } \in \Pi } \sum _ { \boldsymbol { x } \in \mathcal { D } _ { \boldsymbol { u } } } \mathbb { I } \{ \mathcal { A } ( \boldsymbol { f } ( \boldsymbol { x } ) ) = \boldsymbol { y } ( \boldsymbol { x } ) \} .\tag{1}
$$

This single assignment is then reused for every reported subset. For $r \in  { \left\{ \mathrm { o l d } \right. } $ , new, all}, define $\mathcal { D } _ { u , r } : = \{ \pmb { x } \in \mathcal { D } _ { u }$ $y ( \pmb { x } ) \in \mathcal { D } _ { r } \}$ , where $\mathcal { V } _ { \mathrm { a l l } } = \mathcal { V }$ . The corresponding accuracy is

$$
\mathtt { A c c } _ { \mathrm { G H } } ^ { r } ( f ) = \frac { 1 } { | \mathscr { D } _ { u , r } | } \sum _ { \mathbf { x } \in \mathscr { D } _ { u , r } } \mathbb { I } \{ \mathcal { A } _ { \mathrm { G H } } ( f ( \pmb { x } ) ) = y ( \pmb { x } ) \} .\tag{2}
$$

For conciseness, we write these three scores as Old<sub>GH</sub>, New<sub>GH</sub>, and $\mathbb { A } \bot \bot _ { \mathrm { G H } }$ , respectively.

This protocol is appropriate when all output indices are exchangeable. However, a parametric GCD head is only partially exchangeable. This mismatch can induce a problem we call semantic mismapping, which we will analyze in the next section.

## 4 SEMANTIC MISMAPPING AND SEMANTICS-PRESERVING EVALUATION

## 4.1 Semantic Mismapping under Global Matching

To illustrate semantic mismapping, we first partition the confusion matrix into four blocks, with predicted output groups as rows and ground-truth category groups as columns:

$$
\frac { \mathcal { V } _ { \mathrm { o l d } } } { \widehat { \mathcal { V } } _ { \mathrm { o l d } } } \quad \frac { \mathcal { V } _ { \mathrm { n e w } } } { \mathcal { V } _ { \mathrm { o l d } } \times \widehat { \mathcal { V } } _ { \mathrm { o l d } } } \quad \mathcal { V } _ { \mathrm { n e w } } \times \widehat { \mathcal { V } } _ { \mathrm { o l d } }  \\  \widehat { \mathcal { V } } _ { \mathrm { n e w } } \ : \left| \begin{array} { l l } { \mathcal { V } _ { \mathrm { o l d } } \times \widehat { \mathcal { V } } _ { \mathrm { n e w } } } & { \mathcal { V } _ { \mathrm { n e w } } \times \widehat { \mathcal { V } } _ { \mathrm { n e w } } } \end{array} \right.
$$

The diagonal blocks, $i . e . , \ \mathcal { V } _ { \mathrm { o l d } } \times \widehat { \mathcal { V } } _ { \mathrm { o l d } }$ and $\mathcal { V } _ { \mathrm { n e w } } \times \widehat { \mathcal { V } } _ { \mathrm { n e w . } }$ , represent old-to-old and new-to-new predictions, respectively.

In contrast, the off-diagonal blocks, i.e., $\mathcal { V } _ { \mathrm { o l d } } \times \widehat { \mathcal { V } } _ { \mathrm { n e w } }$ and $\mathcal { V } _ { \mathrm { n e w } } \times \hat { \mathcal { V } } _ { \mathrm { o l d . } }$ , represent old-to-new and new-to-old errors, respectively.

How does global matching induce semantic mismapping? Next, we take the predictions in $\mathrm { F i g } . 2$ as an example. In this case, keeping $\hat { o } _ { 1 } \mapsto o _ { 1 }$ and $\hat { n } _ { 1 }  n _ { 1 }$ only yields $7 + 5 = 1 2$ correct predictions, whereas swapping the two mappings yields $4 6 + 4 4 = 9 0$ seemingly correct ones. Thus, global matching will choose $\mathcal { A } ( \hat { o } _ { 1 } ) \stackrel { \cdot } { = } \hat { n } _ { 1 }$ and $\mathcal { A } ( \hat { n } _ { 1 } ) = o _ { 1 } ,$ counting both groups of cross-boundary errors as correct, as shown in Fig.2(b). Inspired by this example, we have the following definition of semantic mismapping.

Definition 1 (Semantic mismapping). For each old category $y _ { o } \in \mathcal { D } _ { \mathrm { o l d } } ,$ let $\hat { y } _ { o } \in \hat { \mathcal { V } } _ { \mathrm { o l d } }$ denote its output index in the parametric head h. Given an assignment $\mathcal { A } \in \Pi ,$ , semantic mismapping occurs at yˆ<sub>o</sub> $i f \mathcal { A } ( \hat { y } _ { o } ) \neq y _ { o } .$

Next, we ask when semantic mismapping can increase the global matching score. To answer this question, we generalize the two-category swap in Fig.2(b) to arbitrary permutations of the output indices. Suppose the outputs are initially assigned as follows: $\hat { y } _ { 1 } \mapsto y _ { 1 } , \hat { y } _ { 2 } \mapsto y _ { 2 } ,$ , and $\hat { y } _ { 3 } \mapsto y _ { 3 }$ . Global matching may replace these mappings with the cycle $\hat { y } _ { 1 } \mapsto y _ { 2 } , \hat { y } _ { 2 } \mapsto y _ { 3 }$ , and $\hat { y } _ { 3 } \mapsto y _ { 1 }$ , while leaving all other outputs unchanged. Since any reassignment can be decomposed into such cycles, its change in matching score can be measured by the correct predictions lost and gained along each cycle. The following theorem, proved in $\operatorname { A p p . A } .$ formalizes this comparison.

Theorem 1 (Semantic mismapping margin). Let $C _ { y  \hat { y } }$ denote the number of samples from true category y predicted as output index $\hat { y } \in \widehat { \mathcal { V } }$ . We define the highest-scoring assignment that preserves the supervised old-output meanings by

$$
\begin{array} { r } { \mathcal { A } _ { 0 } \in \arg \underset { A \in \Pi } { \operatorname* { m a x } } \displaystyle \sum _ { \hat { y } \in \widehat { \mathcal { V } } } C _ { A ( \hat { y } ) \to \hat { y } } } \\ { \mathrm { s . t . } \quad \mathcal { A } ( \hat { y } _ { o } ) = y _ { o } , \quad \forall y _ { o } \in \mathcal { V } _ { \mathrm { o l d } } , } \\ { \mathcal { A } ( \widehat { \mathcal { V } } _ { \mathrm { n e w } } ) = \mathcal { V } _ { \mathrm { n e w } } . } \end{array}
$$

For distinct categories $y _ { 1 } , \ldots , y _ { k }$ with $k \ \geq \ 2 ,$ set ${ \hat { y } } _ { i } ~ =$ $\mathcal { A } _ { 0 } ^ { - 1 } ( y _ { i } ) , y _ { k + 1 } \stackrel { \smile } { = } y _ { 1 }$ , and $y _ { \gamma } : = \{ y _ { 1 } , \ldots , y _ { k } \}$ . The permutation cycle $\gamma = ( y _ { 1 } , \dots , y _ { k } )$ defines an assignment $\mathcal { A } _ { \gamma }$ that maps yˆ<sub>i</sub> to $y _ { i + 1 }$ and agrees with $\mathcal { A } _ { \mathrm { 0 } }$ elsewhere. Define the cycle margin as

$$
m _ { \gamma } : = \sum _ { i = 1 } ^ { k } ( C _ { y _ { i }  \hat { y } _ { i } } - C _ { y _ { i + 1 }  \hat { y } _ { i } } ) .\tag{3}
$$

The assignment $\mathcal { A } _ { \gamma }$ improves the global matching score if and only $i f m _ { \gamma } < 0 ,$ , and ties when $m _ { \gamma } = 0$ . In other words, define the semantic-mismapping margin by

$$
m _ { \mathrm { s e m } } : = \operatorname* { m i n } _ { \substack { \gamma : \mathcal { V } _ { \gamma } \cap \mathcal { V } _ { \mathrm { o l d } } \neq \mathcal { O } } } m _ { \gamma } .\tag{4}
$$

Then, $\mathcal { A } _ { \mathrm { 0 } }$ is globally optimal over the entire category space $i f$ and only if $m _ { \mathrm { s e m } } \geq 0 .$

TABLE 1  
Comparison of existing GCD evaluation protocols. The first property indicates whether a predicted output can be repeatedly assigned to different categories. The latter two properties correspond to the prevention of Type (I) and Type (II) semantic mismapping, respectively.
<table><tr><td>Protocol</td><td>Matching rule</td><td>Each output assigned only once?</td><td>Within-old semantics preserved?</td><td>Cross-region errors detected?</td></tr><tr><td>Legacy NCD</td><td>Repeated local matching</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Global Hungarian (GH)</td><td>Single global matching</td><td>√</td><td>x</td><td>x</td></tr><tr><td>SPE</td><td>Single local matching</td><td> $\checkmark$ </td><td>√</td><td>√</td></tr></table>

```latex
Algorithm 1 Semantics-Preserving Evaluation for paramet
ric GCD.
Input: Samples $\mathbf { \boldsymbol { x } } \in \mathcal { D } _ { u } ,$ labels $y ,$ predictions ${ \hat { y } } = f ( { \pmb x } )$
old categories $\mathcal { D } _ { \mathrm { o l d } }$ , new categories $\mathrm { \bar { \mathcal { P } } _ { n e w } }$
Output: Old<sub>SPE</sub>, New<sub>SPE</sub>, All<sub>SPE</sub>, and H<sub>SPE</sub>
1: Split $\mathcal { D } _ { u }$ into $\mathcal { D } _ { u , \mathrm { o l d } }$ and $\mathcal { D } _ { u , \mathrm { n e w } } .$
2: Compute Old<sub>SPE</sub> using the old outputs.
3: Build the confusion matrix between $\mathcal { V } _ { \mathrm { n e w } }$ and $\widehat { \mathcal { V } } _ { \mathrm { n e w } }$
using samples $\{ \pmb { x } : y \in \mathcal { V } _ { \mathrm { n e w } } , \hat { y } \in \widehat { \mathcal { V } } _ { \mathrm { n e w } } \}$
4: Obtain the local Hungarian matching $\mathcal { A } _ { \mathrm { n e w } } ^ { \ast }$ using
$\operatorname { E q . } ( 5 ) .$
5: Compute New<sub>SPE</sub> and All<sub>SPE</sub> using Eq.(7), and H<sub>SPE</sub>
using Eq.(8).
```

Remark 1 (Mismatch types). We distinguish two types: Type (I), within-old semantic mismapping, occurs when $\ddot { \mathcal { A } } ( \hat { y } _ { o } ) \in \mathcal { V } _ { \mathrm { o l d } } \ \backslash \ \{ y _ { o } \}$ . Type (II), cross-boundary semantic mismapping, occurs when $\mathcal { A } ( \hat { y } _ { o } ) \in \mathcal { V } _ { \mathrm { n e w } }$ . Accordingly, an old-only cycle with $\mathcal { V } _ { \gamma } \subseteq \mathcal { V } _ { \mathrm { o l d } }$ produces only type (I), whereas a mixed cycle satisfying $\mathcal { V } _ { \gamma } \cap \mathcal { V } _ { \mathrm { o l d } } \neq \emptyset$ and $\bar { \mathcal { V } } _ { \gamma } \cap \mathcal { V } _ { \mathrm { n e w } } \neq \emptyset$ may additionally produce type (II).

## 4.2 Semantics-Preserving Evaluation

So far, we have shown that global matching can overestimate the performance of a parametric GCD model. To mitigate this issue, we propose Semantics-Preserving Evaluation (SPE), an evaluation protocol aligned with the divide-andconquer framework of parametric GCD. Specifically, SPE evaluates the different output regions as follows:

• Old output region. Since each old output represents a known category, SPE directly checks whether the prediction is correct.

• New output region. Since new outputs do not have predefined categories, SPE applies local Hungarian matching between new outputs and new categories.

• Cross-region predictions. An old sample predicted in the new output region or a new sample predicted in the old output region is directly counted as an error, as SPE no longer allows the matching algorithm to correct such cross-region predictions.

Formally, SPE keeps the original category meanings of the old outputs in the classifier head h. For new outputs, let $\Pi _ { \mathrm { n e w } }$ be the one-to-one mappings from $\widehat { \mathcal { V } } _ { \mathrm { n e w } }$ to ${ \mathcal { V } } _ { \mathrm { n e w } }$ . Then, the local matching can be denoted as

$$
\mathcal { A } _ { \mathrm { n e w } } ^ { * } \in \arg \operatorname* { m a x } _ { \mathcal { A } _ { \mathrm { n e w } } \in \Pi _ { \mathrm { n e w } } } \sum _ { \hat { y } \in \widehat { \mathcal { D } } _ { \mathrm { n e w } } } C _ { \mathcal { A } _ { \mathrm { n e w } } ( \hat { y } )  \hat { y } } .\tag{5}
$$

Overall, the SPE mapping is defined as:

$$
\begin{array} { r } { A _ { \mathrm { S P E } } ( \hat { y } ) = \left\{ \begin{array} { l l } { y _ { o } , } & { \hat { y } = \hat { y } _ { o } , y _ { o } \in \mathcal { V } _ { \mathrm { o l d } } , } \\ { A _ { \mathrm { n e w } } ^ { * } ( \hat { y } ) , } & { \hat { y } \in \widehat { \mathcal { V } } _ { \mathrm { n e w } } . } \end{array} \right. } \end{array}\tag{6}
$$

Using the same subsets $\mathcal { D } _ { u , r }$ as in Sec.3.2, the SPE accuracies are defined as

$$
\mathtt { A c c } _ { \mathrm { S P E } } ^ { r } ( f ) = \frac { 1 } { | \mathscr { D } _ { u , r } | } \sum _ { \mathbf { x } \in \mathscr { D } _ { u , r } } \mathbb { I } \{ \mathcal { A } _ { \mathrm { S P E } } ( f ( \pmb { x } ) ) = y ( \pmb { x } ) \} ,\tag{7}
$$

where $r \in \{ \mathrm { o l d } , \mathrm { n e w } , \mathrm { a l l } \}$ . For conciseness, we denote the three accuracies by Old<sub>SPE</sub>, New<sub>SPE</sub>, and All<sub>SPE</sub>, respectively. Overall, the SPE procedure is summarized in Alg.1. For a single summary of the balance between old-category recognition and novel-category discovery, we additionally report their harmonic mean:

$$
\mathrm { \Delta \ H _ { S P E } = \frac { 2 0 1 d _ { S P E } \cdot N e w _ { S P E } } { 0 1 d _ { S P E } + N e w _ { S P E } } . }\tag{8}
$$

SPE vs. Existing GCD Evaluation Protocols. In Tab.1, we compare existing GCD evaluation protocols through three properties. Legacy NCD uses ground-truth categories to pre-partition samples into old and new subsets and performs two separate local matchings. Consequently, the same predicted output can be reused across the two subsets and assigned to different categories, as discussed in [3]. Although GH does not pre-partition samples, its global matching neither preserves anchored old semantics nor reliably detects cross-region errors, making it possible to overestimate model performance. In contrast, SPE satisfies all three properties. First, unlike Legacy NCD, SPE only performs a single local matching for the new outputs. Second, SPE directly calculates the classification accuracy of the old outputs, excluding matching procedures and thus preventing Type (I) semantic mismapping. Third, SPE directly counts cross-region predictions as errors, preventing their correction through reassignment and thus preventing Type (II) semantic mismapping. Besides, the local matching among new outputs preserves the permutation invariance required for category discovery.

## 4.3 Extensions of SPE Protocol

In this part, we extend SPE beyond parametric GCD methods and the default equal-cardinality setting, $i . e . , | \widehat { \mathcal { V } } | = | \mathcal { V } | .$

Extension to Non-Parametric GCD. SPE requires a partially exchangeable output space and thus cannot be directly applied to non-parametric GCD methods. Fortunately, we can extend SPE to non-parametric methods by identifying the old-category indices based on the labeled set D . To this end, let $\Pi _ { l }$ denote the set of one-to-one maps from $\mathcal { V } _ { \mathrm { o l d } }$ to ${ \widehat { \bf y } } .$ We define the old-output mapping by

$$
b _ { l } ^ { * } \in \arg \operatorname* { m a x } _ { b \in \Pi _ { l } } \sum _ { ( \pmb { x } , y ) \in \mathcal { D } _ { l } } \mathbb { I } \{ f ( \pmb { x } ) = b ( y ) \} .\tag{9}
$$

Then, the output space of $f = \psi \circ \phi$ is partitioned as

$$
\widehat { \mathcal { Y } } _ { \mathrm { o l d } } : = b _ { l } ^ { * } ( \mathcal { Y } _ { \mathrm { o l d } } ) , \qquad \widehat { \mathcal { Y } } _ { \mathrm { n e w } } : = \widehat { \mathcal { V } } \setminus \widehat { \mathcal { Y } } _ { \mathrm { o l d } } .\tag{10}
$$

On top of this partition, we can apply SPE to evaluate the non-parametric GCD model. Notably, this extension is applicable to most non-parametric GCD methods, which use semi-supervised k-means as the non-parametric clustering procedure [3, 4, 5].

Extension to Estimated New Category Number. When $| \widehat { \mathcal { V } } |$ is estimated and thus not equal to |Y|, standard GCD evaluation performs global matching over $\widehat { \mathcal { V } }$ and $\mathcal { V }$ after padding the smaller set with null elements [3]. Following this practice, we extend SPE by applying the padding only to the new region. Formally, if $| \widehat { \mathcal { V } } _ { \mathrm { n e w } } | ~ < ~ | \widehat { \mathcal { V } } _ { \mathrm { n e w } } | ,$ , we add $| \mathcal { V } _ { \mathrm { n e w } } | - | \widehat { \mathcal { V } } _ { \mathrm { n e w } } |$ dummy outputs; if $| \hat { \mathcal { V } } _ { \mathrm { n e w } } | > | \mathcal { V } _ { \mathrm { n e w } } |$ , we add $| \widehat { \mathcal { V } } _ { \mathrm { n e w } } | - | \mathcal { V } _ { \mathrm { n e w } } |$ dummy categories. The resulting new sets are

$$
\widehat { \mathcal { Y } } _ { \mathrm { n e w } } ^ { D } : = \widehat { \mathcal { Y } } _ { \mathrm { n e w } } \cup \widehat { \mathcal { Y } } ^ { D } , \qquad \mathcal { Y } _ { \mathrm { n e w } } ^ { D } : = \mathcal { Y } _ { \mathrm { n e w } } \cup \mathcal { Y } ^ { D } .\tag{11}
$$

Matching a real output to a dummy category leaves the output unmatched, while matching a dummy output to a real category leaves the category unmatched. Hence, every confusion-matrix entry involving a dummy is set to zero, and the problem is converted into an equal-cardinality assignment. Accordingly, Thm.1 yields the following corollary.

Corollary 1 (Cycle margin with estimated cardinalities). Define the zero-padded confusion matrix by

$$
C _ { y  \hat { y } } ^ { D } : = \{ \begin{array} { l l } { C _ { y  \hat { y } } , } & { y \notin \mathcal { y } ^ { D } , \hat { y } \notin \widehat { \mathcal { y } } ^ { D } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array}\tag{12}
$$

Let $\mathcal { A } _ { 0 } ^ { D }$ be the highest-scoring one-to-one assignment from $\widehat { \mathcal { V } } _ { \mathrm { o l d } } \cup \widehat { \mathcal { V } } _ { \mathrm { n e w } } ^ { D }$ to $\mathcal { \bar { P } } _ { \mathrm { o l d } } \cup \mathcal { V } _ { \mathrm { n e w } } ^ { D }$ that preserves the old-output meanings. For a cycle $\gamma = ( y _ { 1 } , \dots , y _ { k } )$ with $k \geq 2 ,$ , let $\hat { y } _ { i }$ denote the output assigned to $y _ { i }$ by $A _ { 0 } ^ { D } ,$ , and set $y _ { k + 1 } = y _ { 1 }$ and $\mathcal { V } _ { \gamma } : = \{ y _ { 1 } , \dotsc , y _ { k } \}$ . The margin is

$$
m _ { \gamma } ^ { D } : = \sum _ { i = 1 } ^ { k } ( C _ { y _ { i }  \hat { y } _ { i } } ^ { D } - C _ { y _ { i + 1 }  \hat { y } _ { i } } ^ { D } ) .\tag{13}
$$

Define

$$
m _ { \mathrm { s e m } } ^ { D } : = \operatorname* { m i n } _ { \gamma : \mathcal { V } _ { \gamma } \cap \mathcal { V } _ { \mathrm { o l d } } \neq \emptyset } m _ { \gamma } ^ { D } .\tag{14}
$$

Global matching achieves a higher score than $\mathcal { A } _ { 0 } ^ { D }$ if and only if $m _ { \mathrm { s e m } } ^ { D } < 0$ . Equivalently, $\mathbf { \mathcal { A } } _ { 0 } ^ { D }$ is globally optimal on the padded sets if and only if $\dot { m } _ { \mathrm { s e m } } ^ { D } \geq 0$

## 5 MMO

SPE prevents semantic mismapping during evaluation by preserving anchored old-output semantics. During training, we propose MMO to boost existing parametric GCD methods with two plug-and-play losses.

## 5.1 Objectives of MMO

The two losses follow the divide-and-conquer principle. On the one hand, Region-Balance Loss (RBL) regulates prediction mass across the old and new regions to improve old/new routing. On the other hand, Novel-Diversity Loss (NDL) strengthens discrimination within the new region by encouraging new outputs to be diverse and well-separated.

Region-Balance Loss. To alleviate prediction bias, we constrain the aggregate probability assigned to each region on unlabeled data. To this end, we encourage the batch-average probability mass ${ \bar { q } } ^ { r }$ assigned to each region $r \in \{ \mathrm { o l d } , \mathrm { n e w } \}$ to match its estimated sample proportion $p _ { r }$ in the unlabeled training set.

Formally, for sample i, let $s _ { i , j }$ denote the logit of output j. We define the probability mass assigned to region r as

$$
q _ { i } ^ { r } = \frac { \sum _ { j \in \widehat { \mathcal { D } } _ { r } } \exp ( s _ { i , j } / \tau ) } { \sum _ { j \in \widehat { \mathcal { D } } } \exp ( s _ { i , j } / \tau ) } , \qquad r \in \{ \mathrm { o l d } , \mathrm { n e w } \} ,\tag{15}
$$

where $\tau$ is the temperature hyperparameter and $q _ { i } ^ { \mathrm { o l d } } \ +$ $q _ { i } ^ { \mathrm { n e w } } ~ = ~ 1$ . For an unlabeled batch $\boldsymbol { B } _ { u } ,$ we compute the average regional probabilities as

$$
\bar { q } ^ { r } = \frac { 1 } { | B _ { u } | } \sum _ { i \in B _ { u } } q _ { i } ^ { r } .\tag{16}
$$

Then, let $p _ { r }$ denote the estimated proportion of region r in the unlabeled training set, with $p _ { \mathrm { o l d } } + p _ { \mathrm { n e w } } = 1$ . We constrain the regional probability allocation through the following cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { R B } } = \sum _ { r \in \{ \mathrm { o l d } , \mathrm { n e w } \} } - p _ { r } \log \bar { q } ^ { r } .\tag{17}
$$

In practice, the old/new sample proportions in each minibatch may differ from the overall proportions $p _ { r }$ . To address this issue, we propose a tolerance-based variant that accommodates such differences, as detailed in App.D. We use this variant in our experiments.

Remark 2 (Estimating the regional prior). We consider two cases, with $p _ { \mathrm { n e w } } ~ = ~ 1 - p _ { \mathrm { o l d } }$ in both. (1) Balanced category sizes. If samples are approximately uniformly distributed across categories before splitting, let $\dot { C } _ { r } = | \mathcal { V } _ { r } |$ . Then

$$
p _ { \mathrm { o l d } } = \frac { C _ { \mathrm { o l d } } ( N _ { l } + N _ { u } ) / ( C _ { \mathrm { o l d } } + C _ { \mathrm { n e w } } ) - N _ { l } } { N _ { u } } .\tag{18}
$$

(2) Imbalanced category sizes. If the fraction of old-category samples selected for labeling, denoted by $\rho ,$ is known, we use

$$
p _ { \mathrm { o l d } } = \frac { ( 1 - \rho ) N _ { l } } { \rho N _ { u } } .\tag{19}
$$

Otherwise, we estimate $\rho$ with TIcE [49], assuming that old-category samples are selected uniformly at random for labeling. App.D.2 describes the procedure.

Novel-Diversity Loss. SPE evaluates novel-category discovery through local matching within the new output region. To improve discrimination among these categories, we encourage orthogonality among new prototypes.

Formally, let ${ \bf w } _ { u } ~ \in ~ \mathbb { R } ^ { d }$ denote the prototype of new output u, which is typically instantiated by a learnable classifier weight vector [7, 12]. We normalize each nonzero vector as

TABLE 2  
Basic information about the datasets used in the benchmark.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Labeled  $\mathcal { D } _ { l }$ </td><td colspan="2">Unlabeled  $\mathcal { D } _ { u }$ </td><td rowspan="2">Dataset</td><td colspan="2">Labeled  $\mathcal { D } _ { l }$ </td><td colspan="2">Unlabeled  $\mathcal { D } _ { u }$ </td></tr><tr><td>Images</td><td>Classes</td><td>Images</td><td>Classes</td><td>Images</td><td>Classes</td><td>Images</td><td>Classes</td></tr><tr><td>CIFAR-10</td><td>12.5k</td><td>5</td><td>37.5k</td><td>10</td><td>CUB</td><td>1.5k</td><td>100</td><td>4.5k</td><td>200</td></tr><tr><td>CIFAR-100</td><td>20.0k</td><td>80</td><td>30.0k</td><td>100</td><td>Stanford Cars</td><td>2.0k</td><td>98</td><td>6.1k</td><td>196</td></tr><tr><td>ImageNet-100</td><td>31.9k</td><td>50</td><td>95.3k</td><td>100</td><td>FGVC-Aircraft</td><td>1.7k</td><td>50</td><td>5.0k</td><td>100</td></tr><tr><td>ImageNet-1K</td><td>321k</td><td>500</td><td>960k</td><td>1000</td><td>Herbarium 19</td><td>8.9k</td><td>341</td><td>25.4k</td><td>683</td></tr></table>

$$
\widetilde { \mathbf { w } } _ { u } = \frac { \mathbf { w } _ { u } } { \vert \vert \mathbf { w } _ { u } \vert \vert _ { 2 } } , \qquad u = 1 , \dots , C _ { \mathrm { n e w } } .\tag{20}
$$

To encourage mutual orthogonality among new-category prototypes, we define the novel-diversity loss as

$$
\mathcal { L } _ { \mathrm { N D } } = \frac { 1 } { C _ { \mathrm { n e w } } ( C _ { \mathrm { n e w } } - 1 ) } \sum _ { u \neq v } \left( \widetilde { \mathbf { w } } _ { u } ^ { \top } \widetilde { \mathbf { w } } _ { v } \right) ^ { 2 } .\tag{21}
$$

Let $\widetilde { \bf W } _ { \mathrm { n e w } } = [ \widetilde { \bf w } _ { 1 } , \dots , \widetilde { \bf w } _ { C _ { \mathrm { n e w } } } ] ^ { \top } \in \mathbb { R } ^ { C _ { \mathrm { n e w } } \times d }$ . Then, the loss can equivalently be written as

$$
\mathcal { L } _ { \mathrm { N D } } = \frac { \left\| \widetilde { \mathbf { W } } _ { \mathrm { n e w } } \widetilde { \mathbf { W } } _ { \mathrm { n e w } } ^ { \top } - \mathbf { I } _ { C _ { \mathrm { n e w } } } \right\| _ { F } ^ { 2 } } { C _ { \mathrm { n e w } } ( C _ { \mathrm { n e w } } - 1 ) } .\tag{22}
$$

Notably, normalization fixes the diagonal entries at one, so the loss penalizes only off-diagonal inner products. Exact orthogonality is attainable when $C _ { \mathrm { n e w } } \leq d ;$ otherwise, NDL still reduces correlations between new-category prototypes without requiring zero loss.

Overall Objective. Overall, the training objective can be summarized as

$$
\mathcal { L } _ { \mathrm { t r a i n } } = \mathcal { L } _ { \mathrm { b a s e } } + \lambda _ { \mathrm { R B } } \mathcal { L } _ { \mathrm { R B } } + \lambda _ { \mathrm { N D } } \mathcal { L } _ { \mathrm { N D } } ,\tag{23}
$$

where $\mathcal { L } _ { \mathrm { b a s e } }$ is the original parametric GCD objective, and $\lambda _ { \mathrm { R B } } , \lambda _ { \mathrm { N D } } \geq 0$ control the two regularizers. Together, these objectives address cross-region probability allocation and within-new prototype diversity, while preserving the baseline method’s supervised and discovery-oriented learning objectives.

## 5.2 Theoretical Analysis

To understand how MMO mitigates semantic mismapping, we examine the spurious performance gains obtained by GH. To this end, we bound $\Delta _ { \mathrm { a l l } } : = \mathrm { \ a l l { \mathrm { \Sigma } } \mathrm { { z } } \mathrm { { H } } - \mathrm { \ a l { \mathrm { \Sigma } } 1 \mathrm { { s p } } } }$ E through $\mathcal { L } _ { \mathrm { R B } }$ and $\mathcal { L } _ { \mathrm { N D } }$

Intuitively, with p fixed, RBL minimizes $\mathbb { K L } _ { \mathrm { R B } } ( p \Vert \bar { q } )$ and thus encourages $\bar { q } ^ { \mathrm { o l \bar { d } } }$ to match $p _ { \mathrm { o l d } }$ , where $p : = ( p _ { \mathrm { o l d } } , p _ { \mathrm { n e w } } )$ and $\bar { q } : = ( \bar { q } ^ { \mathrm { o l d } } , \bar { q } ^ { \mathrm { n e w } } )$ . In this way, RBL penalizes deviations from the estimated old/new proportions, helping mitigate cross-boundary mismapping. On the other hand, NDL encourages orthogonal new prototypes to reduce confusion between new categories. Reassigning outputs can recover incorrect predictions but also sacrifices correct predictions of the affected categories. The following bound accounts for both effects.

The following theorem formalizes these contributions, with the residual $\epsilon _ { \mathrm { r e s } }$ accounting for additional errors, such as old-category misclassification and inaccurate prior estimates. With $\epsilon _ { \mathrm { r e s } }$ controlled, reducing the two losses can tighten the upper bound on the matching gain and raise the lower bound on semantic cycle margins. The full definition and proof are provided in $\mathrm { \ A p p . A . } 2$

Theorem 2 (Upper bound on semantic mismapping). Under the conditions in App.A.2, define

$$
B : = \sqrt { \frac { \mathbb { K } \mathbb { L } _ { \mathrm { R B } } ( p \| \bar { q } ) } { 2 } } + \alpha \mathcal { L } _ { \mathrm { N D } } + \epsilon _ { \mathrm { r e s } } ,\tag{24}
$$

where $\begin{array} { r } { \alpha : = \pi _ { \mathrm { n e w } } ^ { * } C _ { \mathrm { n e w } } ( C _ { \mathrm { n e w } } - 1 ) , \pi _ { \mathrm { n e w } } ^ { * } : = \operatorname* { m a x } _ { y \in \mathcal { Y } _ { \mathrm { n e w } } } \pi _ { y } , } \end{array}$ and $\pi _ { y }$ denotes the sample proportion of category y. Then, the matching gain satisfies

$$
\Delta _ { \mathrm { a l l } } \leq \operatorname* { m i n } \{ B , [ 2 B - \pi _ { * } ] _ { + } \} ,\tag{25}
$$

where [x]<sub>+</sub> := max{x, 0} and

$$
\pi _ { * } : = \operatorname* { m i n } _ { c \in \mathcal { V } _ { \mathrm { o l d } } , y \in \mathcal { V } \setminus \{ c \} } ( \pi _ { c } + \pi _ { y } ) .
$$

Furthermore, for every permutation cycle $\gamma ,$ we have

$$
\frac { m _ { \gamma } } { N _ { u } } \ge \sum _ { y \in \mathcal { V } _ { \gamma } } \pi _ { y } - 2 B .\tag{26}
$$

Remark 3. Eq.(25) provides a global guarantee: GH and SPE yield the same overall accuracy when $2 B \le \pi _ { * } . \ E q . ( 2 6 )$ provides a local guarantee for a given cycle γ involving old categories: the reassignment along this cycle cannot improve accuracy when $\begin{array} { r } { 2 B \le \sum _ { y \in \mathcal { Y } _ { \mathbb { Y } } } \pi _ { y } . } \end{array}$ . Since minimizing the two losses can reduce B, both results show how MMO helps suppress the potential gains from semantic mismapping.

## 6 EXPERIMENTS

In this section, we conduct experiments to answer the following research questions:

• RQ1: Can we empirically observe performance overestimation caused by GH in parametric GCD, and when is this overestimation more significant?

• RQ2: How effective is MMO under GH and SPE, and how do its loss components and coefficients affect performance?

• RQ3: Does SPE remain applicable when the newcategory number is estimated and when the evaluated method is non-parametric?

## 6.1 Experimental Setup

Datasets. We use eight standard GCD benchmarks, CIFAR-10 and CIFAR-100 [50], ImageNet-100 and ImageNet-1K [51, 52], CUB-200-2011 [53], Stanford Cars [54], FGVC-Aircraft [55], and Herbarium 19 [56]. They cover coarsegrained, fine-grained, and large-scale recognition. All experiments follow the established GCD splits [3, 7] as shown in Tab.2. Additional dataset details are provided in App.B.1.

TABLE 3  
Comparison between GH and SPE across benchmark families for seven method–backbone configurations. Gray shading highlights configurations whose ranks differ between the two protocols. The score gaps $\Delta _ { r } , r \in \{ \mathrm { n e w } , \mathrm { a l l } \}$ and rank changes reveal GH-induced performance overestimation, as the answer to RQ1.
<table><tr><td>Dataset</td><td>Method</td><td>Backbone</td><td>NewGH</td><td>NewSPE</td><td> $\Delta _ { \mathrm { n e w } }$ </td><td>OldSPE</td><td> $\mathbb { A } \bot \bot _ { \mathrm { S P E } }$ </td><td> $\Delta _ { \mathrm { a l l } }$ </td></tr><tr><td colspan="9">(a) Coarse-grained benchmarks</td></tr><tr><td></td><td>LegoGCD</td><td>DINO</td><td>97.93</td><td>97.93</td><td>0.00</td><td>95.76</td><td>97.21 93.91</td><td>0.00</td></tr><tr><td></td><td>SimGCD-BIA ProtoGCD</td><td>DINO</td><td>98.74 98.94</td><td>98.74 98.94</td><td>0.00 0.00</td><td>84.25 81.94</td><td>93.27</td><td>0.00</td></tr><tr><td>CIFAR-10</td><td>SimGCD</td><td>DINO DINO</td><td></td><td></td><td>0.00</td><td>93.71</td><td></td><td>0.00</td></tr><tr><td></td><td></td><td></td><td>98.48</td><td>98.48</td><td></td><td></td><td>96.89</td><td>0.00</td></tr><tr><td></td><td>SimGCD</td><td>CLIP</td><td>98.11 98.42</td><td>98.11</td><td>0.00</td><td>94.75</td><td>96.99</td><td>0.00</td></tr><tr><td></td><td>GET-TAR GET</td><td>CLIP CLIP</td><td>98.36</td><td>98.42 98.36</td><td>0.00 0.00</td><td>94.70 94.87</td><td>97.18 97.19</td><td>0.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.00</td></tr><tr><td></td><td>LegoGCD</td><td>DINO DINO</td><td>85.99 78.60</td><td>85.99 78.60</td><td>0.00</td><td>81.71 75.50</td><td>83.13 76.53</td><td>0.00</td></tr><tr><td></td><td>SimGCD-BIA</td><td>DINO</td><td>82.49</td><td></td><td>0.00</td><td></td><td></td><td>0.00</td></tr><tr><td>CIFAR-100</td><td>ProtoGCD</td><td></td><td>74.02</td><td>82.49</td><td>0.00</td><td>77.82</td><td>79.38 79.13</td><td>0.00</td></tr><tr><td></td><td>SimGCD</td><td>DINO</td><td>73.75</td><td>74.02</td><td>0.00 2.07</td><td>81.69</td><td></td><td>0.00</td></tr><tr><td></td><td>SimGCD GET-TAR</td><td>CLIP CLIP</td><td>71.69</td><td>71.68 71.69</td><td></td><td>84.90 85.68</td><td>80.50</td><td>0.12</td></tr><tr><td></td><td>GET</td><td>CLIP</td><td>71.12</td><td>71.12</td><td>0.00 0.00</td><td>85.69</td><td>81.01 80.83</td><td>0.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.00</td></tr><tr><td></td><td>LegoGCD</td><td>DINO</td><td>77.76 78.70</td><td>77.76</td><td>0.00</td><td>94.88</td><td>83.49</td><td>0.00</td></tr><tr><td></td><td>SimGCD-BIA</td><td>DINO</td><td></td><td>78.70</td><td>0.00</td><td>86.05</td><td>81.16</td><td>0.00</td></tr><tr><td>ImageNet-100</td><td>ProtoGCD</td><td>DINO</td><td>78.36</td><td>78.36</td><td>0.00</td><td>91.00</td><td>82.59</td><td>0.00</td></tr><tr><td></td><td>SimGCD</td><td>DINO</td><td>81.26</td><td>81.26</td><td>0.00</td><td>92.58</td><td>85.04</td><td>0.00</td></tr><tr><td></td><td>SimGCD</td><td>CLIP</td><td>87.80</td><td>87.80</td><td>0.00</td><td>94.76</td><td>90.13</td><td>0.00</td></tr><tr><td></td><td>GET-TAR</td><td>CLIP</td><td>88.28</td><td>88.28</td><td>0.00</td><td>95.43</td><td>90.67</td><td>0.00</td></tr><tr><td>(b) Fine-grained benchmarks</td><td>GET</td><td>CLIP</td><td>92.02</td><td>92.02</td><td>0.00</td><td>95.69</td><td>93.25</td><td>0.00</td></tr><tr><td colspan="9"></td></tr><tr><td></td><td>LegoGCD</td><td>DINO</td><td>(rank) 37.48 (4)</td><td>(rank) 17.98 (2)</td><td>19.50</td><td>69.02</td><td>35.83</td><td>8.99</td></tr><tr><td></td><td>SimGCD-BIA</td><td>DINO</td><td>35.65 (7)</td><td>16.34 (6)</td><td>19.31</td><td>65.47</td><td>33.53</td><td>9.00</td></tr><tr><td></td><td>ProtoGCD</td><td>DINO</td><td>35.92 (6)</td><td>18.52 (1)</td><td>17.40</td><td>68.04</td><td>35.84</td><td>7.90</td></tr><tr><td>Herbarium 19</td><td>SimGCD</td><td>DINO</td><td>36.72 (5)</td><td>17.66 (3)</td><td>19.06</td><td>67.09</td><td>34.95</td><td>9.03</td></tr><tr><td></td><td>SimGCD</td><td>CLIP</td><td>38.26 (2)</td><td>12.63 (7)</td><td>25.63</td><td>77.34</td><td>35.27</td><td>12.13</td></tr><tr><td></td><td>GET-TAR</td><td>CLIP</td><td>37.77 (3)</td><td>16.46 (5)</td><td>21.31</td><td>78.59</td><td>38.20</td><td>9.13</td></tr><tr><td></td><td>GET</td><td>CLIP</td><td>41.11 (1)</td><td>17.10 (4)</td><td>24.01</td><td>78.52</td><td>38.59</td><td>10.44</td></tr><tr><td></td><td>LegoGCD</td><td>DINO</td><td>43.83 (7)</td><td>40.72 (7)</td><td>3.11</td><td>79.41</td><td>53.32</td><td>0.70</td></tr><tr><td></td><td>SimGCD-BIA</td><td>DINO</td><td>45.16 (5)</td><td>42.19 (6)</td><td>2.97</td><td>74.91</td><td>52.85</td><td>0.80</td></tr><tr><td></td><td>ProtoGCD</td><td>DINO</td><td>45.21 (4)</td><td>43.59 (4)</td><td>1.62</td><td>69.07</td><td>51.89</td><td>0.40</td></tr><tr><td>Stanford Cars</td><td>SimGCD</td><td>DINO</td><td>44.61 (6)</td><td>42.70 (5)</td><td>1.91</td><td>74.11</td><td>52.93</td><td>0.60</td></tr><tr><td></td><td>SimGCD</td><td>CLIP</td><td>64.06 (3)</td><td>59.38 (3)</td><td>4.68</td><td>89.01</td><td>69.03</td><td>1.43</td></tr><tr><td></td><td>GET-TAR</td><td>CLIP</td><td>72.80 (2)</td><td>60.51 (2)</td><td>12.29</td><td>95.75</td><td>71.99</td><td>3.97</td></tr><tr><td></td><td>GET</td><td>CLIP</td><td>74.58 (1)</td><td>70.75 (1)</td><td>3.83</td><td>92.30</td><td>77.77</td><td>1.19</td></tr><tr><td></td><td>LegoGCD</td><td>DINO</td><td>57.69 (7)</td><td>55.72 (7)</td><td>1.97</td><td>73.05</td><td>61.50</td><td>0.69</td></tr><tr><td></td><td>SimGCD-BIA</td><td>DINO</td><td>62.00 (4)</td><td>60.89 (4)</td><td>1.11</td><td>60.91</td><td>60.90</td><td>0.71</td></tr><tr><td>CUB</td><td>ProtoGCD</td><td>DINO</td><td>59.13 (5)</td><td>58.09 (6)</td><td>1.04</td><td>64.78</td><td>60.32</td><td>0.65</td></tr><tr><td></td><td>SimGCD</td><td>DINO</td><td>58.09 (6)</td><td>58.39 (5)</td><td>-0.30</td><td>65.04</td><td>60.61</td><td>0.33</td></tr><tr><td></td><td>SimGCD</td><td>CLIP</td><td>65.10 (3)</td><td>64.40 (3)</td><td>0.70</td><td>76.12</td><td>68.31</td><td>0.22</td></tr><tr><td></td><td>GET-TAR</td><td>CLIP</td><td>73.01 (2)</td><td>72.07 (2)</td><td>0.94</td><td>78.79</td><td>74.31</td><td>0.29</td></tr><tr><td></td><td>GET</td><td>CLIP</td><td>74.97 (1)</td><td>74.74 (1)</td><td>0.23</td><td>76.85</td><td>75.44</td><td>0.03</td></tr><tr><td></td><td>LegoGCD</td><td>DINO</td><td>50.52 (6)</td><td>47.53 (7)</td><td>2.99</td><td>64.71</td><td>53.25</td><td>1.14</td></tr><tr><td></td></table>

Methods. We evaluate and compare seven configurations under both GH and SPE. With DINO [57] as the backbone, we evaluate four baselines: LegoGCD [9], SimGCD-BIA [7, 45], ProtoGCD [12], and SimGCD [7]. With CLIP [58] as the backbone, we evaluate SimGCD [7], GET [10], and GET-TAR [59]. We apply the proposed MMO to all seven baselines. Descriptions of these methods are provided in App.B.2.

TABLE 4  
Empirical validation of the cycle-margin analysis in Thm.1 using SimGCD. The summed cycle margins $\begin{array} { r } { \sum _ { \gamma \in \Gamma ( \mathcal { A } _ { \mathrm { G H } } , \mathcal { A } _ { 0 } ) } m _ { \gamma } } \end{array}$ exactly recover the observed accuracy gaps $\Delta _ { \mathrm { a l l } }$ through Eq.(27). The last two columns report the numbers of old outputs involved in Type (I) within-old and Type (II) cross-boundary semantic mismapping, respectively, as defined in Rem.1.
<table><tr><td>Dataset (Backbone)</td><td> $\Delta _ { \mathrm { a l l } }$ </td><td> $N _ { u }$ </td><td>Total cycle margin  $\begin{array} { r } { \sum _ { \gamma \in \Gamma ( A _ { \mathrm { G H } } , A _ { 0 } ) } m _ { \gamma } } \end{array}$ </td><td>Old→wrong-old mismapping (Type I)</td><td>Old→new mismapping (Type II)</td></tr><tr><td>CIFAR-10 (DINO)</td><td>0.00 (Zero)</td><td>37,500</td><td>0</td><td></td><td>0</td></tr><tr><td>CUB (CLIP)</td><td>0.22 (Low)</td><td>4,496</td><td>-10</td><td></td><td>3</td></tr><tr><td>Stanford Cars (CLIP)</td><td>1.43 (Medium)</td><td>6,144</td><td>-88</td><td></td><td>6</td></tr><tr><td>Herbarium 19 (CLIP)</td><td>12.13 (High)</td><td>25,356</td><td>-3,076</td><td>4</td><td>119</td></tr></table>

Metrics. We report Old<sub>GH</sub>, New<sub>GH</sub>, and All<sub>GH</sub> under Global Hungarian (GH), together with Old<sub>SPE</sub>, New<sub>SPE</sub>, All<sub>SPE</sub>, and H<sub>SPE</sub> under our proposed SPE. For $r \_ \in$ {new, all}, we use $\Delta _ { r } : = \mathsf { A c c } _ { \mathrm { G H } } ^ { r } - \mathsf { A c c } _ { \mathrm { S P E } } ^ { r }$ to measure how much GH overestimates the model performance compared to SPE. Since GH optimizes over a superset of the assignments available to SPE, $\Delta _ { \mathrm { a l l } } \geq 0$

Implementation details. We implement our experiments in PyTorch [60]. We follow the training pipeline of each baseline method and introduce MMO only through the auxiliary terms in Eq.(23). Specifically, the SimGCD-based runs follow the original SimGCD optimization [7]: batch size 128, 200 epochs, SGD with an initial learning rate of 0.1 and cosine decay. Its DINO instantiation uses a DINO-pretrained ViT-B/16 [61] and fine-tunes only the final transformer block. SimGCD–CLIP uses the same SimGCD objective and optimization schedule, with the DINO backbone replaced by the CLIP visual encoder. The GET-based runs follow the original two-stage CLIP ViT-B/16 pipeline [10]: each stage is trained for 200 epochs using SGD, a training batch size of 128, an initial learning rate of 0.1 with cosine annealing, and the original augmentation; evaluation uses a batch size of 256. Other baseline-specific hyperparameters follow the original implementations and are detailed in App.B.3. For MMO, the sensitivity of λ<sub>RB</sub> and $\lambda _ { \mathrm { N D } }$ is analyzed in Sec.6.3. We use $\rho = 0 . 5$ given by the GCD benchmark protocols and only the automatic-estimation experiment in Sec.6.3 uses the estimated value ${ \hat { \rho } } .$

## 6.2 Empirical Evidence of GH Overestimation (RQ1)

We first examine whether GH-induced performance overestimation is empirically observable, going beyond the illustrative example in Fig.2. To this end, we evaluate the same predictions under GH and SPE, guaranteeing that the score differences arise only from semantic mismapping. As shown in Tab.3, GH and SPE yield identical new and overall accuracies for all configurations on coarse-grained datasets, such as CIFAR-10 and ImageNet-100. On CIFAR-100, only SimGCD–CLIP shows a narrow gap, with $\Delta _ { \mathrm { n e w } } ~ = ~ 2 . 0 7$ and $\Delta _ { \mathrm { a l l } } ~ = ~ 0 . 1 2$ . However, larger differences appear on fine-grained benchmarks, most clearly on Herbarium 19, where $\Delta _ { \mathrm { n e w } }$ ranges from 17.40 to 25.63. What’s worse, the ranking of these methods changes when evaluated under SPE. On Herbarium 19, switching from GH to SPE moves ProtoGCD–DINO from sixth to first, SimGCD–CLIP from second to seventh, and GET–CLIP from first to fourth. Similar rank reversals occur within the DINO configurations on CUB, Stanford Cars, and FGVC-Aircraft. All these results show that GH can significantly overestimate novelcategory discovery, especially on fine-grained datasets, and that ignoring semantic mismapping can mislead the ranking of methods, as the answer to RQ1. Complete results under both protocols are provided in App. Tab.9.

Prediction-level cycle validation. To connect the observed overestimation to the analysis in Thm.1, we examine the final SimGCD predictions in four configurations with different GH–SPE gaps. Let $\Gamma ( \mathcal { A } _ { \mathrm { G H } } , \mathcal { A } _ { 0 } )$ denote the set of nontrivial permutation cycles relating the GH assignment $\mathcal { A } _ { \mathrm { G H } }$ to the highest-scoring semantics-preserving assignment $\mathcal { A } _ { \mathrm { 0 } }$ . Tab.4 reports the summed cycle margins $\sum _ { \lambda \gamma \in \Gamma ( \mathcal { A } _ { \mathrm { G H } } , \mathcal { A } _ { 0 } ) } m _ { \gamma }$ and the observed accuracy gaps $\Delta _ { \mathrm { a l l } } .$ In all four configurations, the negative margin $- m _ { \gamma }$ summed over all the cycles $\gamma ~ \in ~ \Gamma$ exactly equals the number of additional samples counted as correct under GH compared with SPE, yielding

$$
\Delta _ { \mathrm { a l l } } = - { \frac { 1 0 0 } { N _ { u } } } \sum _ { \gamma \in \Gamma ( \mathcal { A } _ { \mathrm { G H } } , \mathcal { A } _ { 0 } ) } m _ { \gamma } .\tag{27}
$$

This exact agreement provides direct empirical support for our cycle-margin analysis provided in Thm.1. The complete calculation procedure and individual cycle margins are provided in App.C.1.

Following the two types defined in Rem.1, Tab.4 further reports how many anchored old outputs GH reassigns to wrong old categories, i.e., Type (I), and to new categories, i.e., Type (II). Neither type occurs in the zero-gap CIFAR-10 configuration. CUB exhibits both types, with 1 and 3 affected outputs, respectively, while Stanford Cars has 6 Type (II) outputs and no Type (I) outputs. On Herbarium 19, Type (II) predominates, with 119 affected outputs compared with 4 for Type (I), showing that most mismapped old outputs are reassigned across the old–new boundary.

## 6.3 Effectiveness and Analysis of MMO (RQ2)

In this part, we evaluate whether MMO mitigates the prediction errors that can give rise to semantic mismapping. To this end, we assess its combined effect under both GH and SPE, then examine hyperparameter sensitivity and visual representations.

Overall effectiveness under both protocols. Tab.5 compares all seven baselines with their MMO variants on four fine-grained benchmarks. MMO improves new-category accuracy under both GH and SPE in 23 of the 28 configurations. For example, on CUB, MMO improves the newcategory accuracy of SimGCD–CLIP by 5.57 and 6.10 points under GH and SPE, respectively. Overall, the average newaccuracy gain is 2.55 points under GH and 6.48 points under SPE. The larger gains under SPE support the alignment of MMO with our proposed evaluation protocol. Additional results on the coarse-grained ImageNet-100 benchmark are provided in App.C.2.

TABLE 5  
Comprehensive comparison on four fine-grained benchmarks under GH and SPE. Adding MMO (i.e., +Ours) helps improve new-category recognition and old–new balance under both GH and SPE, with particularly large gains on Herbarium 19. Moreover, MMO yields larger average gains under SPE than under GH, indicating better alignment with our semantics-preserving evaluation, as the answer to RQ2.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="4">GH</td><td colspan="4">SPE</td></tr><tr><td>AllGH</td><td>OldGH</td><td>NewGH</td><td>HGH</td><td>AllsPE</td><td>OldsPE</td><td>NewSPE</td><td>HSPE</td></tr><tr><td rowspan="14">CUB</td><td>LegoGCD</td><td>DINO</td><td>62.19</td><td>71.18</td><td>57.69</td><td>63.73</td><td>61.50</td><td>73.05</td><td>55.72</td><td>63.22</td></tr><tr><td>+ Ours</td><td>DINO</td><td>63.88</td><td>74.85</td><td>58.39 (+0.70)</td><td>65.60 (+1.87)</td><td>63.63</td><td>74.72</td><td>58.09 (+2.37)</td><td>65.36 (+2.14)</td></tr><tr><td>SimGCD-BIA DINO</td><td></td><td>61.61</td><td>60.84</td><td>62.00</td><td>61.41</td><td>60.90</td><td>60.91</td><td>60.89</td><td>60.90</td></tr><tr><td>+ Ours</td><td>DINO</td><td>61.81</td><td>62.17</td><td>61.63 (-0.37)</td><td>61.90 (+0.49)</td><td>61.17</td><td>62.91</td><td>60.29 (-0.60)</td><td>61.57 (+0.67)</td></tr><tr><td>ProtoGCD</td><td>DINO</td><td>60.97</td><td>64.64</td><td>59.13</td><td>61.76</td><td>60.32</td><td>64.78</td><td>58.09</td><td>61.25</td></tr><tr><td>+ Ours</td><td>DINO</td><td>61.08</td><td>64.84</td><td>59.19 (+0.06)</td><td>61.89 (+0.13)</td><td>60.50</td><td>64.98</td><td>58.26 (+0.17)</td><td>61.44(+0.19)</td></tr><tr><td>SimGCD</td><td>DINO</td><td>60.94</td><td>66.64</td><td>58.09</td><td>62.07</td><td>60.61</td><td>65.04</td><td>58.39</td><td>61.54</td></tr><tr><td>+ Ours</td><td>DINO</td><td>61.43</td><td>63.91</td><td>60.19(+2.10)</td><td>61.99 (-0.08)</td><td>61.14</td><td>65.64</td><td>58.89 (+0.50)</td><td>62.08 (+0.54)</td></tr><tr><td>SimGCD</td><td>CLIP</td><td>68.53</td><td>75.38</td><td>65.10</td><td>69.86</td><td>68.31</td><td>76.12</td><td>64.40</td><td>69.77</td></tr><tr><td>+ Ours</td><td>CLIP</td><td>72.69</td><td>76.72</td><td>70.67 (+5.57)</td><td>73.57 (+3.71)</td><td>72.26</td><td>75.78</td><td>70.50 (+6.10)</td><td>73.04 (+3.27)</td></tr><tr><td>GET-TAR</td><td>CLIP</td><td>74.60</td><td>77.79</td><td>73.01</td><td>75.32</td><td>74.31</td><td>78.79</td><td>72.07</td><td>75.28</td></tr><tr><td>+ Ours</td><td>CLIP</td><td>74.89</td><td>74.32</td><td>75.18 (+2.17)</td><td>74.75 (-0.57)</td><td>74.62</td><td>75.78</td><td>74.04(+1.97)</td><td>74.90 (-0.38)</td></tr><tr><td>GET + Ours</td><td>CLIP</td><td>75.47</td><td>76.45</td><td>74.97</td><td>75.70</td><td>75.44</td><td>76.85</td><td>74.74</td><td>75.78</td></tr><tr><td></td><td>CLIP</td><td>76.89</td><td>75.72</td><td>77.48 (+2.51)</td><td>76.59 (+0.89)</td><td>76.18</td><td>76.78</td><td>75.88 (+1.14)</td><td>76.33 (+0.55)</td></tr><tr><td rowspan="14">FGVC-Aircraft</td><td>LegoGCD</td><td>DINO</td><td>54.39</td><td>62.12</td><td>50.52</td><td>55.72</td><td>53.25</td><td>64.71</td><td>47.53</td><td>54.81</td></tr><tr><td>+ Ours</td><td>DINO</td><td>52.51</td><td>58.88</td><td>49.33 (-1.19)</td><td>53.68 (-2.04)</td><td>51.79</td><td>61.40</td><td>46.99 (-0.54)</td><td>53.24 (-1.57)</td></tr><tr><td>SimGCD-BIA DINO</td><td></td><td>52.09</td><td>55.46</td><td>50.40</td><td>52.81</td><td>51.99</td><td>56.30</td><td>49.84</td><td>52.87</td></tr><tr><td>+ Ours</td><td>DINO</td><td>54.15</td><td>58.70</td><td>51.87 (+1.47)</td><td>55.07 (+2.26)</td><td>53.83</td><td>60.50</td><td>50.49 (+0.65)</td><td>55.04(+2.17)</td></tr><tr><td>ProtoGCD</td><td>DINO</td><td>51.87</td><td>46.40</td><td>54.60</td><td>50.17</td><td>51.63</td><td>46.04</td><td>54.42</td><td>49.88</td></tr><tr><td>+ Ours</td><td>DINO</td><td>53.43</td><td>59.72</td><td>50.28 (-4.32)</td><td>54.59 (+4.42)</td><td>53.35</td><td>61.76</td><td>49.15 (-5.27)</td><td>54.74 (+4.86)</td></tr><tr><td>SimGCD</td><td>DINO</td><td>53.31</td><td>58.58</td><td>50.67</td><td>54.34</td><td>53.15</td><td>60.14</td><td>49.66</td><td>54.40</td></tr><tr><td>+ Ours</td><td>DINO</td><td>54.55</td><td>59.84</td><td>51.90 (+1.23)</td><td>55.59 (+1.25)</td><td>54.39</td><td>59.96</td><td>51.60 (+1.94)</td><td>55.47 (+1.07)</td></tr><tr><td>SimGCD</td><td>CLIP</td><td>55.59</td><td>59.54</td><td>53.61</td><td>56.42</td><td>55.47</td><td>59.54</td><td>53.43</td><td>56.32</td></tr><tr><td>+ Ours</td><td>CLIP</td><td>60.41</td><td>60.92</td><td>60.15(+6.54)</td><td>60.53 (+4.11)</td><td>59.61</td><td>62.42</td><td>58.20 (+4.77)</td><td>60.24(+3.92)</td></tr><tr><td>GET-TAR</td><td>CLIP</td><td>60.37</td><td>64.59</td><td>58.26</td><td>61.26</td><td>60.19</td><td>66.39</td><td>57.09</td><td>61.39</td></tr><tr><td>+ Ours</td><td>CLIP</td><td>60.47</td><td>62.00</td><td>59.70 (+1.44)</td><td>60.83 (-0.43)</td><td>60.31</td><td>62.85</td><td>59.04 (+1.95)</td><td>60.89 (-0.50)</td></tr><tr><td>GET + Ours</td><td>CLIP CLIP</td><td>60.31</td><td>64.59 65.85</td><td>58.17</td><td>61.21</td><td>59.69</td><td>67.65</td><td>55.71</td><td>61.10</td></tr><tr><td></td><td></td><td>60.89</td><td></td><td>58.41 (+0.24)</td><td>61.91 (+0.70)</td><td>60.71</td><td>65.31</td><td>58.41 (+2.70)</td><td>61.67 (+0.57)</td></tr><tr><td rowspan="14">Herbarium 19</td><td>LegoGCD</td><td>DINO</td><td>44.82</td><td>58.47</td><td>37.48</td><td>45.68</td><td>35.83</td><td>69.02</td><td>17.98</td><td>28.53</td></tr><tr><td>+ Ours</td><td>DINO</td><td>44.51</td><td>58.87</td><td>36.79 (-0.69)</td><td>45.28 (-0.40)</td><td>36.53</td><td>69.49</td><td>18.80 (+0.82)</td><td>29.59 (+1.06)</td></tr><tr><td>SimGCD-BIA DINO</td><td></td><td>42.53</td><td>55.30</td><td>35.65</td><td>43.35</td><td>33.53</td><td>65.47</td><td>16.34</td><td>26.15</td></tr><tr><td>+ Ours</td><td>DINO</td><td>42.77</td><td>56.27</td><td>35.51 (-0.14)</td><td>43.54(+0.19)</td><td>34.10</td><td>65.37</td><td>17.28 (+0.94)</td><td>27.33 (+1.18)</td></tr><tr><td>ProtoGCD</td><td>DINO</td><td>43.74</td><td>58.28</td><td>35.92</td><td>44.45</td><td>35.84</td><td>68.04</td><td>18.52</td><td>29.12</td></tr><tr><td>+ Ours</td><td>DINO</td><td>44.31</td><td>55.60</td><td>38.24 (+2.32)</td><td>45.31 (+0.86)</td><td>41.56</td><td>60.97</td><td>31.12(+12.60)</td><td>41.21 (+12.09)</td></tr><tr><td>SimGCD</td><td>DINO</td><td>43.98</td><td>57.47</td><td>36.72</td><td>44.81</td><td>34.95</td><td>67.09</td><td>17.66</td><td>27.96</td></tr><tr><td>+ Ours</td><td>DINO</td><td>43.74</td><td>47.34</td><td>41.81 (+5.09)</td><td>44.40 (-0.41)</td><td>42.89</td><td>48.57</td><td>39.83 (+22.17)</td><td>43.77 (+15.81)</td></tr><tr><td>SimGCD</td><td>CLIP</td><td>47.40</td><td>64.39</td><td>38.26</td><td>48.00</td><td>35.27</td><td>77.34</td><td>12.63</td><td>21.71</td></tr><tr><td>+ Ours GET-TAR</td><td>CLIP CLIP</td><td>49.14 47.33</td><td>56.91 65.08</td><td>44.96 (+6.70) 37.77</td><td>50.23 (+2.23) 47.80</td><td>48.58 38.20</td><td>57.90 78.59</td><td>43.56(+30.93) 16.46</td><td>49.72(+28.01) 27.22</td></tr><tr></table>

Mitigating errors hidden by semantic mismapping. The largest SPE gains occur on Herbarium 19, where the overestimation is most significant. MMO raises New<sub>SPE</sub> by 22.17, 30.93, and 26.76 points for SimGCD–DINO, SimGCD–CLIP, and GET–CLIP, respectively. Their full-set gaps simultaneously decrease from 9.03, 12.13, and 10.44 points to 0.85, 0.56, and 0.75 points. Recall that the full-set GH–SPE gap is determined by the summed cycle margins through Eq.(27), as verified in Sec.6.2. This result is consistent with the theoretical bounds in Sec.5.2 and validates the effectiveness of MMO in reducing the semantic-mismapping errors.

![](images/7073ec1a375a7595029ff16b2d50004ba214341a77a96be7b96ddb6122ff00b3.jpg)  
(a) CUB (GH)

![](images/6c2482a967092e873e447b82922651f901055d7c04f0a569cb5928fe32423d35.jpg)  
(b) Aircraft (GH)

![](images/3971c076c3e93d6ee42b4b6f093faa3c94a0314451b8bc0b1a50667cb7176eb3.jpg)  
(c) H19 (GH)

![](images/7cb7fe15deee138069773f7a73d2f97039eb8bef4128e05fdb08aa289e2fd765.jpg)  
(d) Cars (GH)

![](images/e489a1fc58e2ea5a6a4cdc5941c0a0bddb29476f80f95def1e987a89724bbf64.jpg)  
(e) CUB (SPE)

![](images/460a72ff5959f601e1408dd6e74dd83db1c4910c21a34abfb981bd7a94bf6041.jpg)  
(f) Aircraft (SPE)

![](images/4849641bbbebea0c7b1ce902ffc4354a96b008fbc6973ed0645714f3e5b74704.jpg)  
(g) H19 (SPE)

![](images/870127be8519f10a8341be2a9e06f0fed50dc50837bc051d0627680d9b349580.jpg)  
(h) Cars (SPE)

Fig. 3. Old–new accuracy trade-offs for the CLIP-based methods in Tab.5. The top and bottom rows report GH and SPE, respectively; columns show CUB, FGVC-Aircraft, Herbarium 19, and Stanford Cars. TAR denotes GET-TAR, and +Ours denotes the corresponding method with MMO.  
![](images/3f087780b8d99bec9b1b3fb57155ec2d1ff2327906a53ad86e7b769eb1f023de.jpg)

![](images/f6c79ae90f10ba530299f2ab451d223a01f133d65d63de2048a2cc5b1a6635e1.jpg)

![](images/0c2cf6664a3cb2c3d4c7e58a4667ab9a0c8c258a1cd651d475d7813b17feb99c.jpg)

![](images/0f92045fc77ba05b46c6de5e1b15476d008269936ce5c3a61e8ccce4fec06c9c.jpg)  
Fig. 4. Sensitivity to the labeling fraction on the imbalanced Herbarium 19 benchmark with SimGCD–CLIP. From left to right, the panels show All, Old, New, and harmonic-mean (H) accuracy under GH and SPE. Baseline denotes the unmodified result, the middle points use manually specified ρb, and Auto uses the TIcE estimate. Complete numerical results are provided in App. Tab.13.

Better old–new trade-off of MMO. As shown in Fig.3, the 12 CLIP-based configurations all improve new accuracy under both protocols, while old-category accuracy can decrease. For example, on CUB, SimGCD–CLIP gains 6.10 points in $\mathrm { \Delta N e w { s p } { e } }$ and 3.27 points in H<sub>SPE</sub>, with a 0.34- point decrease in ${ \mathrm { ~ O 1 d } } _ { \mathrm { S P E } }$ . This improvement shows that MMO helps achieve a better balance between old-category recognition and novel-category discovery. App.C.5 provides more CLIP-based trade-off plots.

Large-scale evaluation. We further extend the comparison to ImageNet-1K in Tab.6. Again, MMO improves both overall and new accuracy under GH and SPE with both backbones. For DINO, the overall/new gains are 1.65/2.40 points under GH and 2.07/3.02 points under SPE; for CLIP, they are 1.04/1.27 and 1.37/2.13 points, respectively. Complete results are provided in App.C.3.

TABLE 6  
Large-scale comparison on ImageNet-1K. Adding MMO improves both new-category and overall accuracy under GH and SPE with DINO and CLIP backbones, demonstrating its effectiveness on the large-scale benchmark, as the answer to RQ2.
<table><tr><td>Method</td><td>Backbone AllGH</td><td></td><td>OldGH</td><td>NewGH</td><td>AllSPE</td><td>OldSPE</td><td>NewSPE</td></tr><tr><td>SimGCD</td><td>DINO</td><td>57.04</td><td>77.14</td><td>46.94</td><td>54.81</td><td>85.15</td><td>39.58</td></tr><tr><td>+ Ours</td><td>DINO</td><td>58.69</td><td>77.29</td><td>49.34</td><td>56.88</td><td>85.28</td><td>42.60</td></tr><tr><td>SimGCD</td><td>CLIP</td><td>57.99</td><td>77.06</td><td>48.41</td><td>55.98</td><td>85.26</td><td>41.27</td></tr><tr><td>+ Ours CLIP</td><td></td><td>59.03</td><td>77.63</td><td>49.68</td><td>57.35</td><td>85.10</td><td>43.40</td></tr></table>

Sensitivity to the labeling fraction. On the imbalanced benchmark such as Herbarium 19, nearly equal numbers of old and new categories do not imply equal sample proportions [62, 63, 64, 65]. RBL therefore uses the labeling fraction $\rho ,$ or its estimate ${ \widehat { \rho } } ,$ to infer the old-region sample prior through Eq.(19). Fig.4 compares manually specified estimates with the data-driven TIcE estimate for SimGCD– CLIP, while the true labeling fraction remains fixed at 0.50.

![](images/c7f544cf08d5dca07590684de3e52986dfa525d562a43100ed5676e7b267e03c.jpg)  
(a) All<sub>GH</sub>

![](images/a523ae4f8be6480fc3848299bbc6d5124a09aca7847f04e41a38cc3f7193dca0.jpg)

![](images/df844d6d49c28edb45735fec65667b0532ec29770ac3182dd282d3a4695ba632.jpg)

(b) Old<sub>GH</sub>  
![](images/ec11e3a53405491714bf1b8bd45e449b914dbbafa270439db8b5253bfbd686c9.jpg)  
(c) New<sub>GH</sub>

(d) All<sub>SPE</sub>  
![](images/e09c343a8e514e0cd14fc2fe860a22575470603491e7b09929a7f8c4e51c8d94.jpg)  
(e) Old<sub>SPE</sub>

![](images/dd9d4d01d17128f18fafff77856d1959a12eb54e9f8c5db6c30f88998584067b.jpg)  
(f) New<sub>SPE</sub>  
Fig. 5. Joint sensitivity of SimGCD–DINO on CUB to the balancing coefficient $\lambda _ { \mathrm { R B } }$ and the NDL coefficient $\lambda _ { \mathrm { N D } } .$ The top and bottom rows report GH and SPE, respectively; the columns from left to right show All, Old, and New accuracy.

The results show that increasing $\widehat { \rho }$ from 0.40 to 0.70 lowers the inferred old-region prior and shifts the accuracy trade-off toward new categories: Old decreases from 67.50% to 37.89%, while New<sub>SPE</sub> increases from 41.04% to 45.37%. This trend agrees with RBL’s role in controlling probability allocation across output regions. Among the manually specified settings, $\widehat { \rho } ~ = ~ 0 . 4 0$ yields the highest All<sub>SPE</sub> and H<sub>SPE</sub>, at 50.30% and 51.04%. This suggests that slight perturbations to the regional prior may improve robustness and classification performance.

TIcE estimates $\widehat { \rho } = 0 . 3 5 7 \widehat { 5 }$ and $\widehat { p } _ { \mathrm { o l d } } = 0 . 6 2 8 7 ,$ yielding $\mathtt { A l l } _ { \mathrm { S P E } } = 4 6 . 0 2 \% , \mathtt { N e w } _ { \mathrm { S P E } } = 3 2 . 7 2 \%$ and $\mathrm { H } _ { \mathrm { S P E } } = 4 4 . 7 4 \%$ These scores exceed the baseline values of 35.27%, 12.63%, and 21.71%, respectively, although they fall below the best manually specified setting. In other words, MMO improves performance even when the regional prior is estimated automatically.

Hyperparameter sensitivity. Fig.5 examines $\lambda _ { \mathrm { R B } }$ and λ jointly for SimGCD–DINO on CUB. Moderate RBL weights, $\lambda _ { \mathrm { R B } } ^ { \prime } \in [ 0 . 2 5 , 0 . 7 5 ]$ , preserve old-category accuracy, while moderate nonzero NDL weights, typically λ<sub>ND</sub> ∈ [0.01, 0.05], improve new-category recognition. These trends are consistent with the complementary roles of regional balance and new-prototype diversity. Jointly using the largest tested weights, $\bar { \lambda _ { \mathrm { R B } } } ~ = ~ 1 . 0$ and $\lambda _ { \mathrm { N D } } ~ = ~ 0 . { \overset { \cdot } { 1 } } ,$ , reduces $\mathtt { A l l 1 _ { G H } / A l l _ { S P E } }$ to $5 9 . 1 0 / 5 8 . 6 1$ and New<sub>GH</sub>/New<sub>SPE</sub> to $5 6 . 0 9 / 5 5 . 4 6 ,$ showing that stronger regularization does not necessarily improve performance. Overall, the moderate coefficient ranges provide a favorable balance between oldcategory recognition and novel-category discovery. Component ablation results for RBL and NDL are provided in App.C.6.

![](images/79afee49f68d5afb0922c9796e9f9df3a4e9f976a684fac0109193b169f6447c.jpg)

![](images/bb1df8cd0eda240e3156b8ebd137ff1d9fe011ff9dc14a07f3402f221f08fb93.jpg)

Visual analysis. Fig.6 complements the aforementioned quantitative evaluation with t-SNE [66] views of selected new categories on Herbarium 19. Although GET and GET-TAR achieve only 17.10% and 16.46% new accuracy under SPE, samples from several new categories already form well-defined clusters. In other words, visually distinct clusters can coexist with low SPE accuracy. The large GH– SPE gaps further indicate that global reassignment can hide such prediction errors, validating the necessity of our SPE protocol.

With MMO, several displayed clusters become more compact and less overlapping, qualitatively supporting improved new-category discrimination. Meanwhile, New<sub>SPE</sub> increases to 43.86% for GET and 44.80% for GET-TAR. The visualization illustrates changes in local feature structure, while the SPE results measure whether predictions respect the fixed old-output meanings and the exchangeability of new outputs.

## 6.4 Applicability of SPE Extensions (RQ3)

In this part, we examine the two SPE extensions in Sec.4.3: estimating the number of new categories and evaluating non-parametric GCD methods. Both extensions preserve the divide-and-conquer principle of SPE by fixing the oldoutput correspondence and restricting category matching to the new region.

Estimated number of new categories. Following Vaze et al. [3], we estimate the total category number $\stackrel { \triangledown } { \hat { C } }$ and obtain $\widehat { C } _ { \mathrm { n e w } } = \widehat { C } - C _ { \mathrm { o l d } }$ . SPE accommodates the resulting mismatch by padding only the new region, while preserving the anchored old-output semantics. In Tab.7, DINO and CLIP estimate 131 and 112 new categories on CUB, whose true number is 100, and 132 and 136 on Stanford Cars, whose true number is 98. SPE remains applicable in all these configurations and exposes score gains from global reassignment. For example, the Stanford Cars SimGCD–CLIP baseline has $\Delta _ { \mathrm { n e w } } ~ { = } ~ 3 . 9 1$ and $\Delta _ { \mathrm { a l l } } ~ = ~ 1 . 0 6$ . With MMO, these gaps decrease to 0.48 and 0.05 points, respectively, and All<sub>SPE</sub> increases from 72.51% to 76.22%. These results show that the evaluation extension supports both baseline and MMO when the new-category number is estimated. Complete results are provided in App. Tab.15.

![](images/b1d29a90b1e80f0bb220abb56e4de2fa3885920d3691167586e78d1abe3bc5c8.jpg)

![](images/a0f02c4e30307d964db499c605d48cc670b12fc1cc73a90d86ff56543acbf8e5.jpg)  
Fig. 6. t-SNE visualization of ten selected new categories from Herbarium 19 for (a) GET, (b) GET+Ours, (c) GET-TAR, and (d) GET-TAR+Ours. Colored circles denote samples, and stars denote the corresponding class prototypes. Red dashed ellipses highlight representative local structures, while arrows connect the baseline and +Ours views for visual comparison. The complete visualizations and the definitions of the high-dimensiona intra- and inter-class statistics are provided in App.C.7.

TABLE 7  
Evaluation of SPE with an estimated new-category number following Vaze et al. [3]. SPE remains applicable when the estimated category number differs from the ground truth and reveals GH–SPE score differences on CUB and Stanford Cars, as the answer to RQ3.
<table><tr><td>Dataset</td><td>Method</td><td>Estimator</td><td> $C _ { \mathrm { n e w } }$ </td><td> $\widehat { C } _ { \mathrm { n e w } }$ </td><td>NewGH</td><td>NewSPE</td><td>OldSPE</td><td>AllsPE</td><td> $\Delta _ { \mathrm { n e w } }$ </td><td> $\Delta _ { \mathrm { a l l } }$ </td></tr><tr><td rowspan="4">CUB</td><td>SimGCD</td><td>DINO</td><td>100</td><td>131</td><td>60.19</td><td>60.56</td><td>63.11</td><td>61.41</td><td>-0.37</td><td>0.27</td></tr><tr><td>+ Ours</td><td>DINO</td><td>100</td><td>131</td><td>60.46 (+0.27)</td><td>60.99 (+0.43)</td><td>66.64</td><td>62.88</td><td>-0.53</td><td>0.15</td></tr><tr><td>SimGCD</td><td>CLIP</td><td>100</td><td>112</td><td>67.60</td><td>67.23</td><td>76.32</td><td>70.26</td><td>0.37</td><td>0.13</td></tr><tr><td>+ Ours</td><td>CLIP</td><td>100</td><td>112</td><td>68.57 (+0.97)</td><td>67.27 (+0.04)</td><td>74.92</td><td>69.82</td><td>1.30</td><td>0.71</td></tr><tr><td rowspan="4">Stanford Cars</td><td>SimGCD</td><td>DINO</td><td>98</td><td>132</td><td>44.53</td><td>42.60</td><td>72.91</td><td>52.47</td><td>1.93</td><td>0.41</td></tr><tr><td>+ Ours</td><td>DINO</td><td>98</td><td>132</td><td>46.68 (+2.15)</td><td>45.84 (+3.24)</td><td>68.97</td><td>53.37</td><td>0.84</td><td>0.28</td></tr><tr><td>SimGCD</td><td>CLIP</td><td>98</td><td>136</td><td>69.30</td><td>65.39</td><td>87.26</td><td>72.51</td><td>3.91</td><td>1.06</td></tr><tr><td>+ Ours</td><td>CLIP</td><td>98</td><td>136</td><td>70.87 (+1.57)</td><td>70.38 (+4.99)</td><td>88.31</td><td>76.22</td><td>0.48</td><td>0.05</td></tr></table>

TABLE 8

Evaluation of the non-parametric SPE extension using SelEx [30] equipped with Bures-Isotropy Alignment (BIA) [45]. Labeled-set calibration enables SPE to evaluate non-parametric predictions and reveals GH–SPE score differences across the five benchmarks, as the answer to RQ3.
<table><tr><td>Dataset</td><td>Cali.  $_ \mathrm { A c c . }$ </td><td colspan="3">Old</td><td colspan="3">New</td><td colspan="3">All</td></tr><tr><td></td><td></td><td>OldGH</td><td>OldsPE</td><td> $\Delta _ { \mathrm { o l d } }$ </td><td>NewGH</td><td>NewSPE</td><td> $\Delta _ { \mathrm { n e w } }$ </td><td>AllGH</td><td>AllsPE</td><td> $\Delta _ { \mathrm { a l l } }$ </td></tr><tr><td>CIFAR-100</td><td>94.77</td><td>81.04</td><td>82.15</td><td>-1.11</td><td>68.20</td><td>65.45</td><td>2.75</td><td>76.76</td><td>76.58</td><td>0.18</td></tr><tr><td>CUB</td><td>99.93</td><td>67.71</td><td>67.58</td><td>0.13</td><td>74.04</td><td>73.74</td><td>0.30</td><td>71.93</td><td>71.69</td><td>0.24</td></tr><tr><td>Stanford Cars</td><td>99.75</td><td>75.21</td><td>77.26</td><td>-2.05</td><td>50.88</td><td>48.73</td><td>2.15</td><td>58.81</td><td>58.02</td><td>0.78</td></tr><tr><td>FGVC-Aircraft</td><td>99.70</td><td>65.01</td><td>65.97</td><td>-0.96</td><td>57.00</td><td>56.34</td><td>0.66</td><td>59.67</td><td>59.55</td><td>0.12</td></tr><tr><td>Herbarium 19</td><td>82.93</td><td>45.08</td><td>45.22</td><td>-0.14</td><td>30.10</td><td>28.17</td><td>1.93</td><td>35.34</td><td>34.13</td><td>1.21</td></tr></table>

Non-parametric GCD. Non-parametric cluster indices are initially exchangeable. We therefore establish the oldoutput correspondence using only the labeled set $\mathcal { D } _ { l }$ and freeze this correspondence before evaluating $\mathcal { D } _ { u } ,$ as prescribed by Eq.(9). Tab.8 evaluates this extension with $\mathrm { S e ^ { - } }$ lEx [30] equipped with BIA [45]. We report calibration accuracy to assess how well the fixed old-output correspondence matches the labeled samples, since SPE relies on this correspondence to anchor old-category semantics. With $N _ { l } = | \mathcal { D } _ { l } |$ and the mapping $b _ { l } ^ { * }$ defined in Eq.(9),

$$
\mathrm { C a l i . A c c . } = \frac { 1 } { N _ { l } } \sum _ { ( { \pmb x } , { \pmb y } ) \in \mathcal { D } _ { l } } \mathbb { I } \{ f ( { \pmb x } ) = b _ { l } ^ { * } ( { \pmb y } ) \} .\tag{28}
$$

This measures the fraction of labeled samples correctly assigned under the calibrated mapping. No unlabeled groundtruth categories are used to establish this mapping.

GH–SPE differences remain observable after calibration. On CUB, calibration accuracy is 99.93%, yet GH exceeds SPE by 0.30 points in new accuracy and 0.24 points overall.

The new-accuracy gaps reach 2.75 points on CIFAR-100 and 2.15 points on Stanford Cars. On Herbarium 19, calibration accuracy is lower at 82.93%, and the overall gap reaches 1.21 points. These results show that semantic mismapping also occurs in non-parametric GCD after labeled-set calibration. Across the shared benchmarks, the non-parametric method exhibits smaller GH–SPE gaps on average than the parametric methods.

Together, these experiments answer RQ3: SPE extends to estimated category numbers through new-region padding and to non-parametric predictions through labeled-set calibration, which supplies the fixed old-output semantics required by the protocol.

## 7 CONCLUSION

In this paper, we revisit the evaluation of parametric generalized category discovery and show that global matching can overestimate performance by overriding the semantic meaning of supervised old outputs. Our theoretical analysis and empirical findings pinpoint semantic mismapping as a source of misleading accuracy gains that can also affect method rankings. To address this issue, we introduce SPE to preserve anchored output semantics during evaluation, and develop MMO from this evaluation perspective to mitigate prediction errors that global matching can hide. Experiments demonstrate the applicability of SPE across different GCD settings and show that MMO generally improves newcategory recognition and the balance between old and new categories. These findings support using SPE alongside GH to provide a more informative assessment of GCD models while maintaining comparability with previous work. More broadly, our study highlights the importance of respecting learned output semantics in evaluation and exploiting evaluation insights to guide model optimization.

## REFERENCES

[1] D. Hendrycks and K. Gimpel, “A baseline for detecting misclassified and out-of-distribution examples in neural networks,” in International Conference on Learning Representations, 2017, pp. 1–12.

[2] Z. Xu, Q. Xu, Z. Wang, C. Hua, S. Li, Z. Yang, and Q. Huang, “Mind the way you select negative texts: Pursuing the distance consistency in OOD detection with VLMs,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 34 650–34 661.

[3] S. Vaze, K. Han, A. Vedaldi, and A. Zisserman, “Generalized category discovery,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 7482– 7491.

[4] N. Pu, Z. Zhong, and N. Sebe, “Dynamic conceptional contrastive learning for generalized category discovery,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 7579–7588.

[5] S. Zhang, S. Khan, Z. Shen, M. Naseer, G. Chen, and F. S. Khan, “PromptCAL: Contrastive affinity learning via auxiliary prompts for generalized novel category discovery,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 3479–3488.

[6] H. W. Kuhn, “The Hungarian method for the assignment problem,” Naval Research Logistics Quarterly, vol. 2, no. 1–2, pp. 83–97, 1955.

[7] X. Wen, B. Zhao, and X. Qi, “Parametric classification for generalized category discovery: A baseline study,” in IEEE/CVF International Conference on Computer Vision, 2023, pp. 16 544–16 554.

[8] S. Vaze, A. Vedaldi, and A. Zisserman, “No representation rules them all in category discovery,” in Advances in Neural Information Processing Systems, 2023, pp. 19 962–19 989.

[9] X. Cao, X. Zheng, G. Wang, W. Yu, Y. Shen, K. Li, Y. Lu, and Y. Tian, “Solving the catastrophic forgetting problem in generalized category discovery,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16 880–16 889.

[10] E. Wang, Z. Peng, Z. Xie, F. Yang, X. Liu, and M. Cheng, “GET: unlocking the multi-modal potential of CLIP for generalized category discovery,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 20 296–20 306.

[11] S. Choi, D. Kang, and M. Cho, “Contrastive meanshift learning for generalized category discovery,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 23 094–23 104.

[12] S. Ma, F. Zhu, X.-Y. Zhang, and C.-L. Liu, “ProtoGCD: Unified and unbiased prototype learning for generalized category discovery,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 7, pp. 6022–6038, 2025.

[13] D. Liu, Z. Tan, L. Zhao, Z. Zhang, X. Fang, and W. Huang, “Generalized category discovery via reciprocal learning and class-wise distribution regularization,” in International Conference on Machine Learning, 2025, pp. 39 651–39 668.

[14] Y. Hsu, Z. Lv, and Z. Kira, “Learning to cluster in order to transfer across domains and tasks,” in International Conference on Learning Representations, 2018, pp. 1–20.

[15] Y. Hsu, Z. Lv, J. Schlosser, P. Odom, and Z. Kira, “Multi-class classification without multi-class labels,” in International Conference on Learning Representations, 2019, pp. 1–16.

[16] Z. Zhong, L. Zhu, Z. Luo, S. Li, Y. Yang, and N. Sebe, “OpenMix: Reviving known knowledge for discovering novel visual categories in an open world,” in IEEE Conference on Computer Vision and Pattern Recognition, 2021, pp. 9462–9470.

[17] K. Han, A. Vedaldi, and A. Zisserman, “Learning to discover novel visual categories via deep transfer clustering,” in IEEE/CVF International Conference on Computer Vision, 2019, pp. 8400–8408.

[18] K. Han, S. Rebuffi, S. Ehrhardt, A. Vedaldi, and A. Zisserman, “Automatically discovering and learning new visual categories with ranking statistics,” in International Conference on Learning Representations, 2020, pp. 1–13.

[19] H. Chi, F. Liu, W. Yang, L. Lan, T. Liu, B. Han, G. Niu, M. Zhou, and M. Sugiyama, “Meta discovery: Learning to discover novel classes given very limited data,” in International Conference on Learning Representations, 2022, pp. 1–20.

[20] M. Yang, L. Wang, C. Deng, and H. Zhang, “Bootstrap your own prior: Towards distribution-agnostic novel class discovery,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 3459–3468.

[21] Y. Liu, Y. Cai, Q. Jia, B. Qiu, W. Wang, and N. Pu, “Novel class discovery for ultra-fine-grained visual categorization,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 17 679–17 688.

[22] Z. Zhong, E. Fini, S. Roy, Z. Luo, E. Ricci, and N. Sebe, “Neighborhood contrastive learning for novel class discovery,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 10 867–10 875.

[23] X. Jia, K. Han, Y. Zhu, and B. Green, “Joint representation learning and novel category discovery on singleand multi-modal data,” in IEEE/CVF International Conference on Computer Vision, 2021, pp. 590–599.

[24] K. Han, S. Rebuffi, S. Ehrhardt, A. Vedaldi, and A. Zisserman, “AutoNovel: Automatically discovering and learning novel visual categories,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 10, pp. 6767–6781, 2022.

[25] B. Zhao and K. Han, “Novel visual category discovery with dual ranking statistics and mutual knowledge distillation,” in Advances in Neural Information Processing Systems, 2021, pp. 22 982–22 994.

[26] K. J. Joseph, S. Paul, G. Aggarwal, S. Biswas, P. Rai, K. Han, and V. N. Balasubramanian, “Novel class discovery without forgetting,” in European Conference on Computer Vision, 2022, pp. 570–586.

[27] S. Roy, M. Liu, Z. Zhong, N. Sebe, and E. Ricci, “Classincremental novel class discovery,” in European Conference on Computer Vision, 2022, pp. 317–333.

[28] R. Ouldnoughi, C. Kuo, and Z. Kira, “CLIP-GCD: simple language guided generalized category discovery,” CoRR, vol. abs/2305.10420, 2023.

[29] Y. Fei, Z. Zhao, S. Yang, and B. Zhao, “XCon: Learning with experts for fine-grained category discovery,” in British Machine Vision Conference, 2022, pp. 1–12.

[30] S. Rastegar, M. Salehi, Y. M. Asano, H. Doughty, and

C. G. M. Snoek, “SelEx: Self-expertise in fine-grained generalized category discovery,” in European Conference on Computer Vision, 2024, pp. 440–458.

[31] H. Zheng, N. Pu, W. Li, N. Sebe, and Z. Zhong, “Generalized fine-grained category discovery with multigranularity conceptual experts,” Int. J. Comput. Vis., vol. 134, no. 8, p. 379, 2026.

[32] H. Wang, S. Vaze, and K. Han, “HiLo: A learning framework for generalized category discovery robust to domain shifts,” in International Conference on Learning Representations, 2025, pp. 1–42.

[33] H. Kim, S. Suh, D. Kim, D. Jeong, H. Cho, and J. Kim, “Proxy anchor-based unsupervised learning for continuous generalized category discovery,” in IEEE/CVF International Conference on Computer Vision, 2023, pp. 16 642–16 651.

[34] Z. Wang, Q. Xu, Z. Yang, Y. He, X. Cao, and Q. Huang, “OpenAUC: Towards AUC-oriented open-set recognition,” in Advances in Neural Information Processing Systems, 2022, pp. 25 033–25 045.

[35] C. Hua, Q. Xu, Z. Yang, Z. Wang, S. Bao, and Q. Huang, “OpenworldAUC: Towards unified evaluation and optimization for open-world prompt tuning,” in International Conference on Machine Learning, 2025, pp. 24 975– 25 020.

[36] F. Gao, W. Zhong, Z. Cao, X. Peng, and Z. Li, “OpenGCD: Assisting open world recognition with generalized category discovery,” CoRR, vol. abs/2308.06926, 2023.

[37] E. Fini, E. Sangineto, S. Lathuiliere, Z. Zhong, M. Nabi,\` and E. Ricci, “A unified objective for novel class discovery,” in IEEE/CVF International Conference on Computer Vision, 2021, pp. 9264–9272.

[38] M. Yang, Y. Zhu, J. Yu, A. Wu, and C. Deng, “Divide and conquer: Compositional experts for generalized novel class discovery,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 14 248– 14 257.

[39] W. Li, Z. Fan, J. Huo, and Y. Gao, “Modeling inter-class and intra-class constraints in novel class discovery,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 3449–3458.

[40] H. Liu, J. Yang, J. Gao, M. Li, Y. Zhang, S. N. Gowda, and Y. Lu, “SECOS: Semantic capture for rigorous classification in open-world semi-supervised learning,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 39 627–39 636.

[41] H. Wang, S. Vaze, and K. Han, “SPTNet: An efficient alternative framework for generalized category discovery with spatial prompt tuning,” in International Conference on Learning Representations, 2024, pp. 1–36.

[42] H. Zheng, N. Pu, W. Li, N. Sebe, and Z. Zhong, “Textual knowledge matters: Cross-modality co-teaching for generalized visual class discovery,” in European Conference on Computer Vision, 2024, pp. 41–58.

[43] P. Guo, Y. Song, and B. Wang, “Towards debiased generalized category discovery,” in International Joint Conference on Artificial Intelligence, 2025, pp. 5271–5279.

[44] Y. Liu and K. Han, “DebGCD: Debiased learning with distribution guidance for generalized category discovery,” in International Conference on Learning Representa-

tions, 2025, pp. 1–14.

[45] L. Tang, K. Huang, C. Chen, and C. Chen, “Buresisotropy alignment: Manifold learning of generalized category discovery,” in International Conference on Learning Representations, 2026, pp. 1–25.

[46] Y. Liu, Z. He, J. Tang, and K. Han, “CloSeR: Unified relational distillation from closed-set teachers for category discovery,” in European Conference on Computer Vision, 2026, pp. 1–18.

[47] H. Zheng, N. Pu, Y. Cai, T. Long, W. Li, N. Sebe, and Z. Zhong, “The devil is in gradient entanglement: Energy-aware gradient coordinator for robust generalized category discovery,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 3563– 3573.

[48] B. Zhao, X. Wen, and K. Han, “Learning semisupervised gaussian mixture models for generalized category discovery,” in IEEE/CVF International Conference on Computer Vision, 2023, pp. 16 577–16 587.

[49] J. Bekker and J. Davis, “Estimating the class prior in positive and unlabeled data through decision tree induction,” in AAAI Conference on Artificial Intelligence, 2018, pp. 2712–2719.

[50] A. Krizhevsky and G. Hinton, “Learning multiple layers of features from tiny images,” University of Toronto, Technical Report, 2009.

[51] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255.

[52] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. S. Bernstein, A. C. Berg, and L. Fei-Fei, “ImageNet large scale visual recognition challenge,” Int. J. Comput. Vis., vol. 115, no. 3, pp. 211–252, 2015.

[53] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, “The Caltech-UCSD Birds-200-2011 dataset,” California Institute of Technology, Technical Report CNS-TR-2011-001, 2011.

[54] J. Krause, M. Stark, J. Deng, and L. Fei-Fei, “3D object representations for fine-grained categorization,” in IEEE International Conference on Computer Vision Workshops, 2013, pp. 554–561.

[55] S. Maji, E. Rahtu, J. Kannala, M. B. Blaschko, and A. Vedaldi, “Fine-grained visual classification of aircraft,” CoRR, vol. abs/1306.5151, 2013.

[56] K. C. Tan, Y. Liu, B. Ambrose, M. Tulig, and S. J. Belongie, “The herbarium challenge 2019 dataset,” CoRR, vol. abs/1906.05372, 2019.

[57] M. Caron, H. Touvron, I. Misra, H. Jegou, J. Mairal,´ P. Bojanowski, and A. Joulin, “Emerging properties in self-supervised vision transformers,” in IEEE/CVF International Conference on Computer Vision, 2021, pp. 9630–9640.

[58] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning, 2021, pp. 8748–8763.

[59] X. Yang, Y. Zhang, S. Mi, and X.-S. Wei, “TAR: Token-

aware refinement for fine-grained generalized category discovery,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 31 995–32 004.

[60] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Z. Yang, Z. DeVito, M. Rai-¨ son, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, “PyTorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems, 2019, pp. 8024– 8035.

[61] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021, pp. 1–22.

[62] Z. Wang, Q. Xu, Z. Yang, Y. He, X. Cao, and Q. Huang, “A unified generalization analysis of re-weighting and logit-adjustment for imbalanced learning,” in Advances in Neural Information Processing Systems, 2023, pp. 48 417–48 430.

[63] Z. Wang, Q. Xu, Z. Yang, Z. Xu, L. Zhang, X. Cao, and Q. Huang, “A unified perspective for loss-oriented imbalanced learning via localization,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 1, pp. 639–656, 2026.

[64] Z. Yang, Q. Xu, S. Li, Z. Wang, X. Cao, and Q. Huang, “DirMixE: Harnessing test agnostic long-tail recognition with hierarchical label variations,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 4, pp. 4605–4622, 2026.

[65] S. Li, Q. Xu, Z. Yang, Z. Wang, L. Zhang, X. Cao, and Q. Huang, “Focal-SAM: Focal sharpness-aware minimization for long-tailed classification,” in International Conference on Machine Learning, 2025, pp. 36 624–36 651.

[66] L. van der Maaten and G. Hinton, “Visualizing data using t-SNE,” J. Mach. Learn. Res., vol. 9, pp. 2579–2605, 2008.

![](images/9015791c383347c23e05790337e33eb46882489db52ef2bab52654187e6c1185.jpg)  
Zitai Wang received the B.E. degree in computer science and technology from Beijing Jiaotong University in 2019 and the Ph.D. degree from the University of Chinese Academy of Sciences in 2024. He is currently a post-doctoral research fellow with the Institute of Computing Technology, Chinese Academy of Sciences. His research interests include machine learning and data mining. He has authored or coauthored 20+ academic papers in top-tier international conferences and journals including IEEE T-PAMI, IJCV,

![](images/e6bfb9ca5c3b01c5eb1afff5e02a8a79c369d8f4c5beb6391227c65e41fda898.jpg)  
ICML, NeurIPS, CVPR, AAAI, and ACM Multimedia.

Qianqian Xu received the B.S. degree in computer science from China University of Mining and Technology in 2007 and the Ph.D. degree in computer science from the University of Chinese Academy of Sciences in 2013. She is currently a Professor with the Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China. Her research interests include statistical machine learning, with applications in multimedia and computer vision. She has authored or coauthored 100+ academic papers in prestigious international journals and conferences (including T-PAMI, IJCV, T-IP, NeurIPS, ICML, CVPR, AAAI, etc.). Moreover, she serves as an associate editor of IEEE Transactions on Circuits and Systems for Video Technology, IEEE Transactions on Multimedia, and ACM Transactions on Multimedia Computing, Communications, and Applications.

![](images/a6272fef3ddb93cf5702d93cfb345e75aee1debdee7e57de1b79f34f6a89d7d2.jpg)

![](images/15ce0c08483ca3906b008c7e7fbd7356eaa6e0d0b3409543b57c3534d30e2c67.jpg)

Xi Chen received the B.Eng. degree from Tsinghua University in 2026. She is currently pursuing her Ph.D. degree with the Institute of Computing Technology, Chinese Academy of Sciences. Her research interest is machine learning.

Cong Hua received the B.S. degree from Jilin University in 2023. He is currently working toward the Ph.D. degree with the Institute of Computing Technology, Chinese Academy of Sciences. His research interests are machine learning and multi-modal learning. He has authored or coauthored several academic papers in top-tier international conferences and journals including ICML, AAAI, and TNNLS.

![](images/828c6cd92687c619c5029d50d9878170fa6af5a0f1f7eece4510ecd1174c315d.jpg)

Zhiyong Yang received his M.S. degree in computer science and technology from the University of Science and Technology Beijing (USTB) in 2017 and his Ph.D. degree from the University of Chinese Academy of Sciences (UCAS) in 2021. He is currently an Associate Professor at the UCAS. His research interests include trustworthy machine learning, long-tail learning, and optimization frameworks for complex metrics. His work has been recognized with various awards, including Top 100 Baidu AI Chinese Rising Stars

Around the World, Top-20 Nomination for the Baidu Fellowship, Asian Trustworthy Machine Learning (ATML) Fellowship, and the China Computer Federation (CCF) Doctoral Dissertation Award. He has authored or co-authored over 60 papers in top-tier international conferences and journals, including 30+ papers in T-PAMI, ICML, and NeurIPS. He has also served as an AC for NeurIPS/ICLR/ICML, an SPC member for IJCAI.

![](images/148b850e78343b9c74aff7d23dddb2d60ecf5ae21b0eba6e25631177a02613de.jpg)

Xiaochun Cao is a Professor in the School of Cyber Science and Technology, Shenzhen Campus of Sun Yat-sen University. He received the B.E. and M.E. degrees both in computer science from Beihang University (BUAA), China, and the Ph.D. degree in computer science from the University of Central Florida, USA, with his dissertation nominated for the university-level Outstanding Dissertation Award. After graduation, he spent about three years at ObjectVideo Inc. as a Research Scientist. From 2008 to 2012, he

was a professor at Tianjin University. Before joining SYSU, he was a professor at the Institute of Information Engineering, Chinese Academy of Sciences. He has authored and coauthored over 200 journal and conference papers. In 2004 and 2010, he was the recipient of the Piero Zamperoni best student paper award at the International Conference on Pattern Recognition. He is on the editorial boards of IEEE Transactions on Image Processing and IEEE Transactions on Multimedia, and was on the editorial board of IEEE Transactions on Circuits and Systems for Video Technology.

![](images/34340d082a29d3da3ae36a818680b838f8b52a9feac65c8fa2ecf78c759f4b2d.jpg)

Qingming Huang is a chair professor in the University of Chinese Academy of Sciences and an adjunct research professor in the Institute of Computing Technology, Chinese Academy of Sciences. He graduated with a bachelor’s degree in Computer Science in 1988 and a Ph.D. degree in Computer Engineering in 1994, both from Harbin Institute of Technology, China. His research areas include multimedia computing, image processing, computer vision and pattern recognition. He has authored or coauthored more than 400 academic papers in prestigious international journals and top-level international conferences. He was the associate editor of IEEE Trans. on CSVT and Acta Automatica Sinica, and the reviewer of various international journals including IEEE Trans. on PAMI, IEEE Trans. on Image Processing, IEEE Trans. on Multimedia, etc. He is a Fellow of IEEE and has served as general chair, program chair, area chair and TPC member for various conferences, including ACM Multimedia, CVPR, ICCV, ICME, ICMR, PCM, BigMM, PSIVT, etc.

## Appendix Contents

Appendix A: Proofs of the Theoretical Results 19   
A.1 Semantic Mismapping Margin 19   
A.2 Theoretical Analysis of MMO 20   
Appendix B: Additional Experimental Setup 24   
B.1 Datasets 24   
B.2 Evaluated Methods 25   
B.3 Implementation Details 25   
Appendix C: Additional Experimental Results 26   
C.1 Additional Comparison between GH and SPE (RQ1) 26   
C.2 Additional Results on the Coarse-Grained Benchmark (RQ2) 28   
C.3 Complete Results on the Large-Scale Benchmark (RQ1 and RQ2) 28   
C.4 Complete ρ-Sensitivity Results on an Imbalanced Benchmark (RQ2) 28   
C.5 Old–New Trade-offs and Overall Accuracy (RQ2) 29   
C.6 Ablation of MMO Components (RQ2) 29   
C.7 Additional Results of the t-SNE Embedding Visualization (RQ2) 30   
C.8 Detailed Results with Estimated Category Numbers (RQ3) 31   
Appendix D: Implementation Details of Region-Balance Loss 33   
D.1 Minibatch Tolerance and Numerical Stability 33   
D.2 Estimating an Unknown Labeling Fraction 33

## APPENDIX A PROOFS OF THE THEORETICAL RESULTS A.1 Semantic Mismapping Margin

Thm.1 (Semantic mismapping margin). Let $C _ { y  \hat { y } }$ be the number of samples from true category y predicted as output index $\hat { y } \in \widehat { \mathcal { V } }$ . We define the highest-scoring assignment that preserves the supervised old-output meanings by

$$
\begin{array} { r } { \mathcal { A } _ { 0 } \in \arg \underset { A \in \Pi } { \operatorname* { m a x } } \displaystyle \sum _ { \hat { y } \in \widehat { \mathcal { V } } } C _ { A ( \hat { y } ) \to \hat { y } } } \\ { \mathrm { s . t . } \quad \mathcal { A } ( \hat { y } _ { o } ) = y _ { o } , \quad \forall y _ { o } \in \mathcal { V } _ { \mathrm { o l d } } , } \\ { \quad \mathcal { A } ( \widehat { \mathcal { V } } _ { \mathrm { n e w } } ) = \mathcal { V } _ { \mathrm { n e w } } . } \end{array}
$$

For different categories $y _ { 1 } , \ldots , y _ { k }$ with $k \geq 2 ,$ , set $\hat { y } _ { i } = A _ { 0 } ^ { - 1 } ( y _ { i } ) , y _ { k + 1 } = y _ { 1 }$ , and $\mathcal { V } _ { \gamma } : = \{ y _ { 1 } , \dotsc , y _ { k } \}$ . The permutation cycle $\gamma = \left( y _ { 1 } , \dots , y _ { k } \right)$ defines an assignment $\mathcal { A } _ { \gamma }$ that maps $\hat { y } _ { i }$ to $y _ { i + 1 }$ <sub>1</sub> and agrees with $\mathcal { A } _ { \mathrm { 0 } }$ elsewhere. Define

$$
m _ { \gamma } : = \sum _ { i = 1 } ^ { k } ( C _ { y _ { i }  \hat { y } _ { i } } - C _ { y _ { i + 1 }  \hat { y } _ { i } } ) .\tag{29}
$$

The assignment $\mathcal { A } _ { \gamma }$ improves the global matching score $i f$ and only if $m _ { \gamma } < 0 ,$ , and ties when $m _ { \gamma } = 0$ . Define

$$
m _ { \mathrm { s e m } } : = \operatorname* { m i n } _ { \substack { \gamma : \mathcal { y } _ { \gamma } \cap \mathcal { y } _ { \mathrm { o l d } } \neq \mathcal { O } } } m _ { \gamma } .\tag{30}
$$

Then $\mathcal { A } _ { \mathrm { 0 } }$ is globally optimal over the entire category space $i f$ and only i $^ c m _ { \mathrm { s e m } } \geq 0 .$

Proof. We first examine how a single cycle changes the matching score, then extend the argument to an arbitrary assignment. Let

$$
S ( \mathcal { A } ) : = \sum _ { \hat { y } \in \hat { \mathcal { V } } } C _ { \mathcal { A } ( \hat { y } )  \hat { y } }\tag{31}
$$

be the number of correct predictions under assignment A. Along a cycle $\gamma = ( y _ { 1 } , \dots , y _ { k } ) , A _ { \cdot }$ assigns output $\hat { y } _ { i }$ to $y _ { i + 1 }$ instead of $y _ { i } .$ . All other outputs keep their assignments, so their contributions cancel when we subtract the two scores:

$$
\mathcal { S } ( \mathcal { A } _ { \gamma } ) - \mathcal { S } ( \mathcal { A } _ { 0 } ) = \sum _ { i = 1 } ^ { k } ( C _ { y _ { i + 1 }  \hat { y } _ { i } } - C _ { y _ { i }  \hat { y } _ { i } } ) = - m _ { \gamma } .\tag{32}
$$

Thus, a negative margin means that the cycle gains more correct predictions than it loses. The score increases exactly when $m _ { \gamma } < 0$ and remains unchanged when $m _ { \gamma } = 0$

To compare $A _ { 0 }$ with any global assignment, let $\mathcal { A } \in \Pi$ , where Π is the set of all bijections from output indices $\widehat { \mathcal { V } }$ to categories $\mathcal { \hat { 3 } } .$ . Their relative reassignment is described by the category permutation

$$
\pi _ { \boldsymbol { A } } : = \boldsymbol { A } \circ \boldsymbol { A } _ { 0 } ^ { - 1 } .\tag{33}
$$

For each category $y ,$ this permutation gives the category that $\mathcal { A }$ assigns to the output previously assigned to $y$ by $A _ { 0 }$ Decompose it into disjoint nontrivial cycles $\gamma _ { 1 } , \dots , \gamma _ { r }$ . Since these cycles affect disjoint outputs and fixed points leave assignments unchanged, the total score change is

$$
S ( \mathcal { A } ) - S ( \mathcal { A } _ { 0 } ) = - \sum _ { j = 1 } ^ { r } m _ { \gamma _ { j } } .\tag{34}
$$

A cycle involving only new categories leaves all old-output assignments unchanged. Since $\mathcal { A } _ { \mathrm { 0 } }$ already maximizes the score with these assignments fixed, such a cycle cannot increase the score and thus has a nonnegative margin. Hence, any improvement must come from a cycle involving an old category, and $m _ { \mathrm { s e m } }$ is exactly the minimum margin over these cycles. If $m _ { \mathrm { s e m } } \geq 0 ,$ , all cycle margins are nonnegative, so no assignment improves on $\mathcal { A } _ { 0 } ;$ otherwise, a negative-margin cycle yields a higher score. Hence, $\mathcal { A } _ { \mathrm { 0 } }$ is globally optimal if and only if $m _ { \mathrm { s e m } } \geq 0 .$ □

## A.2 Theoretical Analysis of MMO

Assump. 1. We have the following assumptions.

• Evaluation setting. We consider a fixed model on an unlabeled set $\mathcal { D } _ { u }$ of size $N _ { u }$ containing at least one sample from every category. The output and category sets have equal cardinality, with at least one old category and $C _ { \mathrm { n e w } } \geq 2$ new categories.

• Prediction–similarity consistency. Whenever the prediction $\hat { y } _ { i } \in \arg \operatorname* { m a x } _ { j \in \widehat { \mathcal { Y } } } s _ { i , j }$ lies in the new region, its prototype also maximizes cosine similarity among new outputs:

$$
\begin{array} { r } { \hat { y } _ { i } \in \widehat { \mathcal { V } } _ { \mathrm { n e w } } \quad \Longrightarrow \quad z _ { i } ^ { \top } \widetilde { \mathbf { w } } _ { \hat { y } _ { i } } \geq z _ { i } ^ { \top } \widetilde { \mathbf { w } } _ { u } , \quad \forall u \in \widehat { \mathcal { V } } _ { \mathrm { n e w } } , } \end{array}\tag{35}
$$

where $z _ { i }$ and $\widetilde { \mathbf { w } } _ { u }$ denote unit-norm features and new-output prototypes, respectively.

Remark 4. These assumptions are mild in the common GCD setting.

• For the evaluation setting, the conditions hold as long as the full evaluation set covers every category and the head uses the given category count, with at least two new categories.

• For prediction–similarity consistency, the condition is mildfor normalized cosine heads because it only requires consistency with similarity ordering, not correct predictions or well-separated prototypes. The cosine classifiers used in SimGCD [7], LegoGCD [9], and ProtoGCD [12] satisfy this condition: their logits are inner products between normalized features and prototypes, so the predicted output naturally maximizes cosine similarity. More generally, the condition holds for any shared strictly increasing transformation of cosine similarity. For example, logits $s _ { i , u } = a _ { i } z _ { i } ^ { \top } \widetilde { \mathbf { w } } _ { u } + b _ { i }$ , with $a _ { i } > 0$ and $a _ { i } , b _ { i }$ shared across new outputs, satisfy Eq.(35), and a shared positive temperature τ corresponds to $a _ { i } = 1 / \tau$ and $b _ { i } = 0 .$

Lem. 1 (Prediction and prototype distance). Under Assump.1, suppose sample i is predicted as a new output v. For any other new output u $\neq v ,$ let $c _ { u v } : = \widetilde { \mathbf { w } } _ { u } ^ { \top } \widetilde { \mathbf { w } } _ { v }$ . Then,

$$
\begin{array} { r } { 1 - c _ { u v } \leq 2 \| z _ { i } - \widetilde { \mathbf { w } } _ { u } \| _ { 2 } ^ { 2 } . } \end{array}\tag{36}
$$

Proof. Prediction–similarity consistency gives

$$
\begin{array} { r l } & { 0 \geq z _ { i } ^ { \top } \big ( \widetilde { \mathbf { w } } _ { u } - \widetilde { \mathbf { w } } _ { v } \big ) } \\ & { \quad = 1 - c _ { u v } + \big ( z _ { i } - \widetilde { \mathbf { w } } _ { u } \big ) ^ { \top } \big ( \widetilde { \mathbf { w } } _ { u } - \widetilde { \mathbf { w } } _ { v } \big ) } \\ & { \quad \geq 1 - c _ { u v } - \| z _ { i } - \widetilde { \mathbf { w } } _ { u } \| _ { 2 } \sqrt { 2 ( 1 - c _ { u v } ) } . } \end{array}\tag{37}
$$

The first inequality follows from Assump.1: predicting v implies $z _ { i } ^ { \top } \widetilde { \mathbf { w } } _ { v } \geq z _ { i } ^ { \top } \widetilde { \mathbf { w } } _ { u }$ . The equality adds and subtracts $\widetilde { \mathbf { w } } _ { u }$ and uses its unit norm. The final inequality applies Cauchy–Schwarz and $\lVert \widetilde { \mathbf { w } } _ { u } - \widetilde { \mathbf { w } } _ { v } \rVert _ { 2 } ^ { 2 } = 2 ( \widetilde { 1 } - \dot { c _ { u v } } )$ . For $c _ { u v } < 1$ , rearranging and squaring gives

$$
\| z _ { i } - \widetilde { \mathbf { w } } _ { u } \| _ { 2 } \geq \frac { 1 - c _ { u v } } { \sqrt { 2 ( 1 - c _ { u v } ) } } \quad \Longrightarrow \quad \| z _ { i } - \widetilde { \mathbf { w } } _ { u } \| _ { 2 } ^ { 2 } \geq \frac { 1 - c _ { u v } } { 2 } .\tag{38}
$$

When $c _ { u v } = 1$ , the resulting inequality $1 - c _ { u v } \leq 2 \| z _ { i } - \widetilde { \mathbf { w } } _ { u } \| _ { 2 } ^ { 2 }$ also holds since its left-hand side is zero.

Thm.2 (Upper bound on semantic mismapping). Under Assump.1, define

$$
B : = \sqrt { \frac { \mathbb { K } \mathbb { L } _ { \mathrm { R B } } ( p \| \bar { q } ) } { 2 } } + \alpha \mathcal { L } _ { \mathrm { N D } } + \epsilon _ { \mathrm { r e s } } ,
$$

where $\begin{array} { r } { \alpha : = \pi _ { \mathrm { n e w } } ^ { * } C _ { \mathrm { n e w } } ( C _ { \mathrm { n e w } } - 1 ) , \pi _ { \mathrm { n e w } } ^ { * } : = \operatorname* { m a x } _ { y \in \mathcal { V } _ { \mathrm { n e w } } } \pi _ { y } , } \end{array}$ , and $\pi _ { y }$ denotes the sample proportion of category y. Then, the matching gain satisfies

$$
\Delta _ { \mathrm { a l l } } \leq \operatorname* { m i n } \{ B , [ 2 B - \pi _ { * } ] _ { + } \} ,
$$

where $[ x ] _ { + } : = \operatorname* { m a x } \{ x , 0 \}$ and

$$
\pi _ { * } : = \operatorname* { m i n } _ { c \in \mathcal { V } _ { \mathrm { o l d } } , y \in \mathcal { V } \setminus \{ c \} } ( \pi _ { c } + \pi _ { y } ) .
$$

Furthermore, for every permutation cycle $\gamma ,$ we have

$$
\frac { m _ { \gamma } } { N _ { u } } \ge \sum _ { y \in \mathcal { Y } _ { \gamma } } \pi _ { y } - 2 B .
$$

Proof. Setup and residual errors. Let $\mathcal { A } _ { 0 }$ be the highest-scoring semantics-preserving assignment from Thm.1, and set

$t ( y ) : = \mathcal { A } _ { 0 } ^ { - 1 } ( y )$ . Define the empirical category proportions by

$$
\pi _ { \boldsymbol { y } } : = \frac { 1 } { N _ { u } } \sum _ { i = 1 } ^ { N _ { u } } \mathbb { I } \{ \boldsymbol { y } _ { i } = \boldsymbol { y } \} , \qquad \pi _ { \mathrm { o l d } } : = \sum _ { \boldsymbol { y } \in \mathcal { V } _ { \mathrm { o l d } } } \pi _ { \boldsymbol { y } } .\tag{39}
$$

Minimizing RBL and NDL alone does not guarantee correct category predictions, so the error bound also includes four residual terms:

$$
\begin{array} { r l r } { \displaystyle \epsilon _ { \mathrm { o l d } } : = \frac { 1 } { N _ { u } } \displaystyle \sum _ { 1 \leq i \leq N _ { u } } \mathbb { I } \{ \hat { y } _ { i } \neq t ( y _ { i } ) \} , } & { \displaystyle \epsilon _ { \mathrm { p r i o r } } : = | p _ { \mathrm { o l d } } - \pi _ { \mathrm { o l d } } | , } & \\ { \displaystyle \epsilon _ { \mathrm { r o u t e } } : = \frac { 1 } { N _ { u } } \displaystyle \sum _ { i = 1 } ^ { N _ { u } } \left| q _ { i } ^ { \mathrm { o l d } } - \mathbb { I } \{ \hat { y } _ { i } \in \widehat { \mathcal { Y } } _ { \mathrm { o l d } } \} \right| , } & { \displaystyle \epsilon _ { \mathrm { a l i g n } } : = \frac { 1 } { N _ { u } } \displaystyle \sum _ { 1 \leq i \leq N _ { u } } \| z _ { i } - \widetilde { \mathbf { w } } _ { t ( y _ { i } ) } \| _ { 2 } ^ { 2 } . } & \end{array}\tag{40}
$$

We explain below what each residual measures and why minimizing the two regularizers alone does not force it to vanish.

$\epsilon _ { \mathrm { o l d } }$ measures errors relative to old outputs. RBL controls the averaged regional probability and NDL controls new-output prototypes. Hence, neither directly enforces correct old-category predictions. Instead, supervision from the labeled set can help reduce this term.

$\epsilon _ { \mathrm { p r i o r } }$ measures the discrepancy between the estimated old-category proportion $p _ { \mathrm { o l d } }$ and its actual value $\pi _ { \mathrm { o l d } }$ . RBL uses $p _ { \mathrm { o l d } }$ as a fixed target, and neither regularizer corrects its estimation error. More accurate estimation of $p _ { \mathrm { o l d } } ,$ , for example using TIcE [49] as in Rem.2, can help reduce this term.

• ϵ measures the average discrepancy between the soft old-region probability $q _ { i } ^ { \mathrm { o l d } }$ and the hard prediction indicator $\mathbb { I } \{ \hat { y } _ { i } \in \widehat { \mathcal { V } } _ { \mathrm { o l d } } \}$ . RBL matches average probability mass, and neither regularizer directly enforces agreement with individual argmax decisions. Prediction sharpening is a potential way to reduce this discrepancy, which we leave for future work.

$\epsilon _ { \mathrm { a l i g n } }$ measures feature–prototype misalignment of new categories under $A _ { 0 }$ . NDL separates prototypes and RBL controls regional probability totals; neither directly aligns features with their corresponding prototypes. Self-distillation with reliable pseudo-labels, as used in SimGCD [7], can help reduce this term.

The following derivation shows how each term enters the prediction-error bound.

Step 0: prediction-error decomposition. Under ${ \mathcal { A } } _ { 0 } ,$ every incorrect prediction belongs to one of three disjoint groups: errors on old-category samples, new-category samples predicted in the old region, or new-category samples assigned to an incorrect new output. Their proportions among all $N _ { u }$ samples are $\epsilon _ { \mathrm { o l d } } , e _ { \mathrm { n e w } \to \mathrm { o l d } }$ , and $e _ { \mathrm { n n } } ,$ respectively. Thus, we have

$$
\begin{array} { r } { e _ { \mathrm { S P E } } : = 1 - \mathbb { A } \mathrm { 1 1 } _ { \mathrm { S P E } } = \underbrace { \epsilon _ { \mathrm { o l d } } } _ { \mathrm { E r r o r s ~ o n ~ o n d ~ s a m p l e s } } + \underbrace { e _ { \mathrm { n e w } \to \mathrm { o l d } } + e _ { \mathrm { n n } } } _ { \mathrm { E r r o r s ~ o n ~ n o v e l ~ s a m p l e s } } . } \end{array}\tag{41}
$$

Next, we bound $e _ { \mathrm { n e w } \to \mathrm { o l d } }$ in Step 1 and $e _ { \mathrm { n n } }$ in Step 2, then use these bounds to prove Eqs.(25) and (26) in Steps 3.1 and 3.2, respectively.

Step 1: regional balance and cross-region errors. Let $e _ { \mathrm { o l d } \to \mathrm { n e w } }$ denote the fraction of all samples that belong to old categories but are predicted in the new region. Samples predicted in the old region come from two groups: samples from old categories predicted within this region and samples from new categories incorrectly predicted in this region. Thus,

$$
\frac { 1 } { N _ { u } } \sum _ { i = 1 } ^ { N _ { u } } \mathbb { I } \{ \hat { y } _ { i } \in \widehat { \mathcal { V } } _ { \mathrm { o l d } } \} = \pi _ { \mathrm { o l d } } - e _ { \mathrm { o l d \to n e w } } + e _ { \mathrm { n e w \to o l d } } .
$$

Subtracting $\pi _ { \mathrm { o l d } }$ from both sides gives

$$
e _ { \mathrm { n e w } \to \mathrm { o l d } } - e _ { \mathrm { o l d } \to \mathrm { n e w } } = \frac { 1 } { N _ { u } } \sum _ { i = 1 } ^ { N _ { u } } \mathbb { I } \{ \hat { y } _ { i } \in \widehat { \mathcal { V } } _ { \mathrm { o l d } } \} - \pi _ { \mathrm { o l d } } .\tag{42}
$$

Furthermore, we have

$$
\begin{array} { r l } & { { \displaystyle  e _ { \mathrm { n e w - o l d } } - e _ { \mathrm { o l d - } } \mathrm { n e w } | } } \\ & { ~ = | \frac { 1 } { N _ { u } } \sum _ { i = 1 } ^ { N _ { u } } ( \mathbb { I } \{ \hat { y } _ { i } \in \widehat { \mathcal { V } } _ { \mathrm { o l d } } \} - q _ { i } ^ { \mathrm { o l d } } ) + ( \bar { q } ^ { \mathrm { o l d } } - p _ { \mathrm { o l d } } ) + ( p _ { \mathrm { o l d } } - \pi _ { \mathrm { o l d } } )  } \\ & { ~ \leq  \frac { 1 } { N _ { u } } \sum _ { i = 1 } ^ { N _ { u } } | \mathbb { I } \{ \hat { y } _ { i } \in \widehat { \mathcal { V } } _ { \mathrm { o l d } } \} - q _ { i } ^ { \mathrm { o l d } } | + | \bar { q } ^ { \mathrm { o l d } } - p _ { \mathrm { o l d } } | + | p _ { \mathrm { o l d } } - \pi _ { \mathrm { o l d } } | } \\ & { ~ = \epsilon _ { \mathrm { r o u t e } } + | \bar { q } ^ { \mathrm { o l d } } - p _ { \mathrm { o l d } } | + \epsilon _ { \mathrm { p r i o r } } } \\ & { ~ \leq \epsilon _ { \mathrm { r o u t e } } + \sqrt { \frac { \mathbb { K } \mathbb { I } _ { \mathrm { R B } } ( p | \bar { q } ) } { 2 } } + \epsilon _ { \mathrm { p r i o r } } . } \end{array}\tag{43}
$$

The first equality adds and subtracts $\bar { q } ^ { \mathrm { o l d } }$ and $p _ { \mathrm { o l d } } ,$ using $\begin{array} { r } { \bar { q } ^ { \mathrm { o l d } } = N _ { u } ^ { - 1 } \sum _ { i = 1 } ^ { N _ { u } } q _ { i } ^ { \mathrm { o l d } } } \end{array}$ . The first inequality applies the triangle inequality, and the second equality substitutes the definitions of $\epsilon _ { \mathrm { r o u t e } }$ and $\epsilon _ { \mathrm { p r i o r } } .$ . The final inequality follows from Pinsker’s inequality for Bernoulli distributions, $\mathbb { K L } _ { \mathrm { R B } } ( p \Vert \bar { q } ) \geq 2 ( p _ { \mathrm { o l d } } - \bar { q } ^ { \mathrm { o l d } } ) ^ { 2 }$ , with natural logarithms. More specifically, the fixedprior RBL in Eq.(17), evaluated over $\mathcal { D } _ { u } ,$ gives $\mathbb { K L } _ { \mathrm { R B } } ( p \Vert \bar { q } ) = \mathcal { L } _ { \mathrm { R B } } - H ( p )$ , where $\begin{array} { r } { H ( p ) : = - \sum _ { r } p _ { r } \log \bar { p } _ { r } } \end{array}$ is the entropy of $p .$

Every sample from an old category predicted in the new region is also incorrectly assigned relative to its anchored old output. Thus, these errors have been included in $\epsilon _ { \mathrm { o l d } } , \mathrm { g i v i n g } e _ { \mathrm { o l d } \to \mathrm { n e w } } \leq \epsilon _ { \mathrm { o l d } }$ . Combining this observation with $\operatorname { E q . } ( 4 3 )$ yields

$$
\begin{array} { r } { e _ { \mathrm { n e w } \to \mathrm { o l d } } \le e _ { \mathrm { o l d } \to \mathrm { n e w } } + | e _ { \mathrm { n e w } \to \mathrm { o l d } } - e _ { \mathrm { o l d } \to \mathrm { n e w } } | } \\ { \le \epsilon _ { \mathrm { o l d } } + \epsilon _ { \mathrm { p r i o r } } + \epsilon _ { \mathrm { r o u t e } } + \sqrt { \frac { \mathbb { K } \mathbb { L } _ { \mathrm { R B } } ( p \| \bar { q } ) } { 2 } } . } \end{array}\tag{44}
$$

Step 2: prototype correlations and within-new-category errors. For distinct new outputs $u , v \in \widehat { \mathcal { V } } _ { \mathrm { n e w } } ,$ define the error proportion

$$
e _ { u v } : = \frac { 1 } { N _ { u } } \sum _ { i = 1 } ^ { N _ { u } } \mathbb { I } \{ y _ { i } = \mathcal { A } _ { 0 } ( u ) , \ \hat { y } _ { i } = v \} .\tag{45}
$$

The corresponding error groups are disjoint and together account for all errors within the new region, so $\begin{array} { r } { e _ { \mathrm { n n } } = \sum _ { u \ne v } e _ { u v } . } \end{array}$ Unless otherwise specified, all sums over u ̸= v below run over distinct outputs in $\widehat { \mathcal { V } } _ { \mathrm { n e w } }$ . Then, we have

$$
\begin{array} { l } { { \displaystyle e _ { \mathrm { n n } } = \sum _ { u \neq v } e _ { u v } = \sum _ { u \neq v } e _ { u v } \big [ ( 1 - c _ { u v } ) + c _ { u v } \big ] } } \\ { { \displaystyle ~ = \sum _ { u \neq v } e _ { u v } ( 1 - c _ { u v } ) + \sum _ { u \neq v } e _ { u v } c _ { u v } } . } \\ { { \displaystyle ( * ) } } \end{array}\tag{46}
$$

The first equality uses the definition of $e _ { \mathrm { n n } }$ . We then write $1 = ( 1 - c _ { u v } ) + c _ { u v }$ and distribute the sum, where $c _ { u v }$ is the prototype cosine similarity defined in Lem.1.

To bound (∗), we expand the sum over individual misclassified samples:

$$
\begin{array} { r l } {  { ( * ) = \frac { 1 } { N _ { u } } \sum _ { 1 \leq i \leq N _ { u } , \ y _ { i } \in \mathcal { N } _ { \mathrm { n e v } } } ( 1 - c _ { t ( y _ { i } ) , \hat { y } _ { i } } ) } } \\ & { \qquad \frac { \hat { y } _ { i } \in \widetilde { \mathcal { N } } _ { \mathrm { n e v } } , \ \hat { y } _ { i } \in \mathcal { N } _ { \mathrm { n e v } } } { \hat { y } _ { i } \in \widetilde { \mathcal { N } } _ { \mathrm { n e v } } , \ \hat { y } _ { i } \in \mathcal { N } ( y _ { i } ) } } \\ & { \leq \frac { 2 } { N _ { u } } \sum _ { 1 \leq i \leq N _ { u } , \ y _ { i } \in \mathcal { N } _ { \mathrm { n e v } } } \| z _ { i } - \widetilde { \mathbf { w } } _ { t ( y _ { i } ) } \| _ { 2 } ^ { 2 } } \\ & { \qquad \quad \leq \frac { 2 } { N _ { u } } \sum _ { 1 \leq i \leq N _ { u } } \| z _ { i } - \widetilde { \mathbf { w } } _ { t ( y _ { i } ) } \| _ { 2 } ^ { 2 } = 2 \epsilon _ { \mathrm { a l i g n } } . } \end{array}\tag{47}
$$

The first inequality applies Lem.1 with $u = t ( y _ { i } )$ and $v = \hat { y } _ { i }$ . The second includes all new-category samples in the sum. The final equality uses the definition of $\epsilon _ { \mathrm { a l i g n } } .$

To bound (∗∗), we use weighted Cauchy–Schwarz and NDL:

$$
\begin{array} { r l r } {  { ( * * ) \leq \sqrt { e _ { \mathrm { n n } } \sum _ { u \ne v } e _ { u v } c _ { u v } ^ { 2 } } } } \\ & { } & { \leq \sqrt { e _ { \mathrm { n n } } \pi _ { \mathrm { n e w } } ^ { * } \sum _ { u \ne v } c _ { u v } ^ { 2 } } } \\ & { } & { = \sqrt { e _ { \mathrm { n n } } \alpha \mathcal { L } _ { \mathrm { N D } } } . } \end{array}\tag{48}
$$

For the first inequality, write $e _ { u v } c _ { u v } = \sqrt { e _ { u v } } ( \sqrt { e _ { u v } } c _ { u v } )$ and apply Cauchy–Schwarz to these two factors:

$$
\left| \sum _ { u \neq v } \sqrt { e _ { u v } } \big ( \sqrt { e _ { u v } } c _ { u v } \big ) \right| \leq \sqrt { \sum _ { u \neq v } e _ { u v } } \sqrt { \sum _ { u \neq v } e _ { u v } c _ { u v } ^ { 2 } } = \sqrt { e _ { \operatorname* { m i n } } \sum _ { u \neq v } e _ { u v } c _ { u v } ^ { 2 } } .
$$

Since $( * * ) \ \leq \ | ( * * ) |$ , this also bounds the original correlation sum. For the second inequality, all misclassified samples contributing to $e _ { u v }$ belong to category $\boldsymbol { \mathcal { A } } _ { 0 } ( \boldsymbol { u } )$ . Their proportion cannot exceed that of the entire category, so $e _ { u v } \le \pi _ { \mathcal { A } _ { 0 } ( u ) }$ In turn, $\pi _ { \boldsymbol { A } _ { 0 } ( u ) } \le \pi _ { \mathrm { n e w } } ^ { * } : = \operatorname* { m a x } _ { y \in \mathcal { Y } _ { \mathrm { n e w } } } \pi _ { y } ,$ the proportion of samples belonging to the largest new category. Replacing each weight $e _ { u v } \mathbf { b y } \pi _ { \mathrm { n e w } } ^ { * }$ gives the inequality. For the final equality, NDL in $\operatorname { E q . } ( 2 1 )$ is the average squared cosine similarity over the $\bar { C } _ { \mathrm { n e w } } ( C _ { \mathrm { n e w } } - 1 )$ ) ordered choices of distinct new outputs. Thus, $\begin{array} { r } { \sum _ { u \ne v } \hat { c } _ { u v } ^ { 2 } = C _ { \mathrm { n e w } } ( C _ { \mathrm { n e w } } - \mathrm { \hat { 1 } } ) \mathcal { L } _ { \mathrm { N D } } } \end{array}$ . Multiplying by $\pi _ { \mathrm { n e w } } ^ { * }$

and using $\alpha = \pi _ { \mathrm { n e w } } ^ { * } C _ { \mathrm { n e w } } ( C _ { \mathrm { n e w } } - 1 )$ gives the equality.

Substituting the bounds on (∗) and (∗∗) in Eqs.(47) and (48) into Eq.(46) gives

$$
\begin{array} { r l r } {  { e _ { \mathrm { n n } } \leq 2 \epsilon _ { \mathrm { a l i g n } } + \sqrt { e _ { \mathrm { n n } } \alpha \mathcal { L } _ { \mathrm { N D } } } } } \\ & { } & { \leq 2 \epsilon _ { \mathrm { a l i g n } } + \frac { e _ { \mathrm { n n } } + \alpha \mathcal { L } _ { \mathrm { N D } } } { 2 } . } \end{array}\tag{49}
$$

The second inequality uses ${ \sqrt { x y } } \leq ( x + y ) / 2$ for nonnegative x, y. Subtracting $e _ { \mathrm { n n } } / 2$ from both sides and multiplying by two yields

$$
e _ { \mathrm { n n } } \leq 4 \epsilon _ { \mathrm { a l i g n } } + \alpha \mathcal { L } _ { \mathrm { N D } } .\tag{50}
$$

Step 3.1: matching-gain bound $ { \bf \delta E q . } ( 2 5 )$ . Substituting the new-to-old error bound in Eq.(44) and the within-newcategory error bound in Eq.(50) into the decomposition in Eq.(41) gives

$$
e _ { \mathrm { S P E } } \leq \sqrt { \frac { \mathbb { K } \mathbb { L } _ { \mathrm { R B } } ( p \| \bar { q } ) } { 2 } } + \alpha \mathcal { L } _ { \mathrm { N D } } + \underbrace { 2 \epsilon _ { \mathrm { o l d } } + \epsilon _ { \mathrm { p r i o r } } + \epsilon _ { \mathrm { r o u t e } } + 4 \epsilon _ { \mathrm { a l i g n } } } _ { \epsilon _ { \mathrm { r e s } } } .\tag{51}
$$

Thus, $e _ { \mathrm { S P E } } \leq B$ . Since GH accuracy cannot exceed one, its improvement over SPE satisfies

$$
\Delta _ { \mathrm { a l l } } = \mathrm { a 1 1 } _ { \mathrm { G H } } - \mathrm { a 1 1 } _ { \mathrm { S P E } } \leq 1 - \mathrm { a 1 1 } _ { \mathrm { S P E } } = e _ { \mathrm { S P E } } \leq B .
$$

To sharpen this matching-gain bound, we examine which reassignments can increase the score. Following Thm.1, we express the changes from $\dot { \boldsymbol { A } } _ { 0 }$ to a GH-optimal assignment as disjoint cycles. With the old-output assignments fixed, $\mathcal { A } _ { \mathrm { 0 } }$ already gives the highest score by matching new outputs to new categories. A cycle involving only new categories merely changes this new-output matching and therefore cannot increase the score. Any cycle that increases the score must consequently change at least one old-output assignment and thus contain an old category. To quantify the gain from these cycles, define the number of errors on category y under $\mathcal { A } _ { \mathrm { 0 } }$ as

$$
E _ { y } : = \sum _ { i = 1 } ^ { N _ { u } } \mathbb { I } \{ y _ { i } = y , \ \hat { y } _ { i } \neq t ( y ) \} .\tag{52}
$$

Category y contributes $N _ { u } \pi _ { y } - E _ { y }$ correct predictions under $\scriptstyle A _ { 0 } ,$ but at most $E _ { y }$ when matched to any output $v \neq t ( y )$ When $\Delta _ { \mathrm { a l l } } > 0$ , let S collect the categories in all score-increasing cycles. Since $S$ contains an old category and at least one other category, the definition of π<sub>∗</sub> gives $\sum { } _ { y \in S } \pi _ { y } \geq \pi ,$ <sub>∗</sub>. Summing the gains over these disjoint cycles then yields

$$
\begin{array} { r l } {  { \Delta \mathrm { a l l } \le \frac { 1 } { N _ { u } } [ \sum _ { y \in \cal S } E _ { y } - \sum _ { y \in \cal S } ( N _ { u } \pi _ { y } - E _ { y } ) ] } } \\ & { \le \frac { 2 } { N _ { u } } \sum _ { y \in \cal S } E _ { y } - \sum _ { y \in \cal S } \pi _ { y } } \\ & { \le 2 e _ { \mathrm { S P E } } - \sum _ { y \in \cal S } \pi _ { y } } \\ & { \le 2 e _ { \mathrm { S P E } } - \pi _ { * } \le 2 B - \pi _ { * } . } \end{array}\tag{53}
$$

The first inequality omits cycles that do not increase the score and bounds the gain on the remaining categories. Before reassignment, these categories contribute $\begin{array} { r } { \sum _ { y \in S } ( N _ { u } \pi _ { y } - E _ { y } ) } \end{array}$ correct predictions. After reassignment, each category y is matched to an output other than $t ( y )$ and contributes at most $E _ { y }$ correct predictions. Their difference, divided by $N _ { u } ,$ gives the inequality. Since the cycles are disjoint, each category in S is counted once; collecting the terms gives the next equality. The subsequent inequalities use $\begin{array} { r } { \sum _ { y \in S } E _ { y } \le N _ { u } e _ { \mathrm { S P E } } , \sum _ { y \in S } \pi _ { y } \ge \pi _ { * } , } \end{array}$ and $e _ { \mathrm { S P E } } \leq B ,$ respectively. If $\Delta _ { \mathrm { a l l } } = 0 ,$ , the bound $\Delta _ { \mathrm { a l l } } \leq [ 2 \dot { B ^ { ' } } - \pi _ { * } ] .$ <sub>+</sub> holds directly.

Combining both cases with $\Delta _ { \mathrm { a l l } } \ \leq \ B$ proves $\operatorname { E q . } ( 2 5 )$ , which bounds the matching gain characterized by the cycle decomposition in Thm.1. $\mathrm { I f 2 } B < \pi _ { * } , \mathrm { G F }$ and SPE have the same accuracy because $\Delta _ { \mathrm { a l l } } = 0$ . When $2 B = \pi _ { * } , \mathsf { a }$ reassignment may instead tie with $A _ { 0 } ,$ allowing semantic mismapping without changing accuracy.

Step 3.2: cycle-margin bound → Eq.(26). We next bound the margin of an arbitrary cycle. For a cycle ${ \boldsymbol { \gamma } } = \left( y _ { 1 } , \dots , y _ { k } \right)$

with $y _ { k + 1 } = y _ { 1 } .$

$$
\begin{array} { r l } {  { m _ { \tau } = \sum _ { j = 1 } ^ { K } ( C _ { y _ { 3 } \to \tau \{ y _ { 4 } \} } - C _ { y _ { 3 } + 1 \to j \leq \tau } ) } } \\ & { \ge \sum _ { y \in \mathcal { Y } _ { y } } ( N _ { u } \pi _ { y } - E _ { u } ) - \sum _ { y \in \mathcal { Y } _ { y } } E _ { y } } \\ & { = N _ { u } \sum _ { y \in \mathcal { Y } _ { y } } \pi _ { y } - 2 \sum _ { y \in \mathcal { Y } _ { y } } E _ { y } } \\ & { \ge N _ { u } ( \sum _ { y \in \mathcal { Y } _ { y } } \pi _ { y } - 2 \epsilon _ { \mathrm { s r e g } } ) } \\ & { \ge N _ { u } ( \sum _ { y \in \mathcal { Y } _ { y } } \pi _ { y } - 2 \epsilon _ { \mathrm { s r e g } } ) } \\ & { \ge N _ { u } ( \sum _ { y \in \mathcal { Y } _ { y } } \pi _ { y } - 2 B ) , } \end{array}\tag{54}
$$

The first equality recalls that the cycle margin compares the numbers of correct predictions before and after reassignment. Under $\boldsymbol { A } _ { 0 } ,$ , output $t ( y _ { j } )$ correctly classifies $\overset { \cdot } { C } _ { y _ { j }  t ( y _ { j } ) } = N _ { u } \pi _ { y _ { j } } - E _ { y _ { j } }$ samples. The cycle reassigns this output to category $y _ { j + 1 } ,$ so its correct predictions become $C _ { y _ { j + 1 }  t ( y _ { j } ) }$ . Because $t ( y _ { j } ) \ne t ( y _ { j + 1 } )$ , these samples were errors under $\scriptstyle A _ { 0 } ,$ , and their number is at most $E _ { y _ { j + 1 } }$ . Substituting these counts gives the first inequality. As the cycle visits each category once, $\begin{array} { r } { \sum _ { j = 1 } ^ { k } E _ { y _ { j + 1 } } = \sum _ { y \in \mathcal { y } _ { \gamma } } E _ { y } , } \end{array}$ so the two error sums combine into $2 \textstyle \sum _ { y \in { \mathcal { y } } _ { \gamma } } E _ { y }$ . This sum can then be bounded using the total number of errors, since $\begin{array} { r } { \sum _ { y \in \mathcal { V } _ { \mathbb { Y } } } E _ { y } \le \sum _ { y \in \mathcal { V } } E _ { y } = N _ { u } } \end{array}$ e<sub>SPE</sub>. Substituting this upper bound into the subtracted term, followed by $e _ { \mathrm { S P E } } \leq B ,$ gives the final two inequalities.

Finally, dividing by $N _ { u }$ proves Eq.(26), which bounds the cycle margin in Thm.1.

## APPENDIX B

## ADDITIONAL EXPERIMENTAL SETUP

## B.1 Datasets

We evaluate our method on eight standard GCD datasets: CIFAR-10, CIFAR-100, ImageNet-100, ImageNet-1K, CUB-200- 2011, Stanford Cars, FGVC-Aircraft, and Herbarium 19. They cover coarse-grained recognition, fine-grained recognition, and large-scale category discovery; their detailed properties and the GCD splits used in our experiments are introduced below.

CIFAR-10. CIFAR-10 [50] contains $3 2 \times 3 2$ natural images from 10 coarse-grained object categories. Following the standard GCD split, 12.5K labeled images come from 5 old categories, while 37.5K unlabeled images cover all 10 categories and therefore include both old and new categories.

CIFAR-100. CIFAR-100 [50] extends CIFAR-10 to 100 object categories and provides a more challenging coarse-grained benchmark. We use 20.0K labeled images from 80 old categories and 30.0K unlabeled images from all 100 categories, leaving 20 categories to be discovered as new categories.

ImageNet-100. ImageNet-100 is a 100-category subset of ImageNet [51, 52] with greater visual diversity and higherresolution images than CIFAR. Its GCD split contains 31.9K labeled images from 50 old categories and 95.3K unlabeled images covering all 100 categories.

ImageNet-1K. ImageNet-1K [51, 52] contains 1,000 object categories and serves as our large-scale benchmark. We use 321K labeled images from 500 old categories and 960K unlabeled images spanning all 1,000 categories, which tests the scalability of GCD methods when many new categories must be identified.

CUB-200-2011. CUB-200-2011 (CUB) [53] is a fine-grained dataset containing 200 bird species whose distinctions often depend on subtle shape, color, and texture cues. The adopted split uses 1.5K labeled images from 100 old species and 4.5K unlabeled images from all 200 species.

Stanford Cars. Stanford Cars [54] contains 196 fine-grained car categories defined by make, model, and year. We use 2.0K labeled images from 98 old categories and 6.1K unlabeled images covering all 196 categories, making the dataset useful for evaluating discovery under small inter-class differences.

FGVC-Aircraft. FGVC-Aircraft [55] comprises 100 aircraft variants with highly similar global appearances and discriminative local details. Its GCD split includes 1.7K labeled images from 50 old variants and 5.0K unlabeled images from all 100 variants.

Herbarium 19. Herbarium 19 [56] contains 683 plant species and exhibits both fine-grained visual differences and substantial class imbalance. We use 8.9K labeled images from 341 old species and 25.4K unlabeled images from all 683 species, providing a challenging test of GCD under a naturally long-tailed distribution.

## B.2 Evaluated Methods

We evaluate and compare seven method–backbone configurations. The four DINO-based baselines are LegoGCD–DINO, SimGCD-BIA–DINO, ProtoGCD–DINO, and SimGCD–DINO; the three CLIP-based baselines are SimGCD–CLIP, GET– CLIP, and GET-TAR–CLIP. These configurations cover representative parametric classifiers, prototype-based learning, regularized GCD methods, and multimodal approaches. We apply MMO to every baseline while preserving its original objective, backbone, and training pipeline. The individual baselines are introduced below.

DINO-based baselines.

LegoGCD–DINO. LegoGCD [9] builds on SimGCD and focuses on reducing catastrophic forgetting of old categories. With a DINO backbone, it introduces local entropy regularization and dual-view prediction consistency to preserve oldcategory knowledge while learning new categories, without changing the basic classifier architecture.

SimGCD-BIA–DINO. This configuration augments the DINO-based SimGCD baseline with Bures-Isotropy Alignment (BIA) [45]. BIA regularizes the feature geometry toward an isotropic structure on unlabeled data, reducing dimensional collapse and improving the separation of old and new categories while preserving the original SimGCD training framework.

ProtoGCD–DINO. ProtoGCD [12] uses a unified prototype-learning framework to represent both old and new categories. Its debiased prototype learning reduces the imbalance between supervised old-category signals and unsupervised new-category signals, with DINO features providing the visual initialization in this configuration.

SimGCD–DINO. SimGCD [7] is a parametric GCD baseline that jointly uses labeled old-category supervision and self-distillation on unlabeled data. In the DINO configuration, a learnable classifier is trained on DINO-pretrained visual features to predict a shared output space containing both old and new categories.

CLIP-based baselines.

SimGCD–CLIP. This configuration uses the SimGCD classifier, objective, and optimization strategy but replaces the DINO backbone with the CLIP visual encoder [58]. It allows us to isolate how the representation learned through vision– language pretraining affects parametric GCD performance.

GET–CLIP. GET [10] is designed to exploit the multimodal knowledge of CLIP for generalized category discovery. It introduces textual semantics into the discovery process so that visual samples can be organized with richer semantic guidance than that available from image-only supervision.

GET-TAR–CLIP. GET-TAR applies token-aware refinement [59] to the CLIP-based GET training pipeline [10]. The refinement suppresses distracting attention patterns and strengthens informative local cues, which is particularly helpful for recognizing and discovering visually similar fine-grained categories.

## B.3 Implementation Details

General implementation details.

Our implementation follows standard practices in recent GCD methods [7, 10]. Image preprocessing, including random cropping, random horizontal flipping, and normalization, follows the settings of the corresponding baselines. We train for 200 epochs (per stage for GET-based methods) and use the final checkpoint for evaluation. For MMO, we select λ and λ via grid search over $\lambda _ { \mathrm { R B } } \in \{ 0 . 2 5 , 0 . 5 , 0 . 7 5 \}$ and $\lambda _ { \mathrm { N D } } \in \{ 0 . 0 1 , 0 . \dot { 0 } 5 , 0 . 1 \}$ . We use the same random seeds as the original baseline implementations. All experiments are conducted on NVIDIA GeForce RTX 3090 GPUs.

Baseline-Specific Hyperparameters.

SimGCD. Following SimGCD [7], we set the supervised–unsupervised balancing coefficient λ to 0.35; the supervised and unsupervised terms receive weights λ and 1 − λ, respectively, in both representation learning and classification. The self-supervised and supervised contrastive temperatures are $\tau _ { u } = 0 . 0 7$ and $\tau _ { c } = 1 . 0 ,$ respectively. In the classification objective, $\tau _ { s } = 0 . 1$ controls the student prediction distribution, while $\tau _ { t }$ controls the teacher pseudo-target distribution. The teacher temperature follows a linear warm-up schedule from 0.07 to 0.04 during the first 30 epochs. SimGCD–CLIP uses the same settings with the CLIP visual encoder.

LegoGCD. LegoGCD [9] preserves the SimGCD losses and adds local entropy regularization (LER) to protect potential old-class samples, together with a dual-view Kullback–Leibler (DKL) constraint in the classification loss. Its objective weights the SimGCD representation and classification losses by $\alpha = 1$ and LER by $\beta .$ The original paper selects $\beta = 2$ on CUB and β = 1 on CIFAR-100; it also reports a confidence threshold $\delta = 0 . 8 5$ for selecting potential old-class samples on Stanford Cars and CIFAR-100. These are LegoGCD-specific parameters, distinct from the MMO regularization weights.

SimGCD-BIA. Bures-Isotropy Alignment (BIA) [45] preserves the SimGCD objective and adds a geometry regularizer on the projected class tokens of the unlabeled minibatch. Under the trace constraint described in the BIA paper, this term maximizes the nuclear norm of the stacked tokens; its implementation subtracts $\lambda _ { \mathrm { B I A } } \| Z _ { u } \|$ <sub>∗</sub> from the baseline loss. The paper uses $\lambda _ { \mathrm { B I A } } = 0 . 0 0 4$ and otherwise follows the underlying GCD method’s optimization settings.

ProtoGCD. Following ProtoGCD [12], the supervised component is weighted by $\lambda _ { \mathrm { s u p } } = 0 . 3 5$ . Its entropy and prototypeseparation regularizers use $\lambda _ { \mathrm { e n t r o p y } } = 2$ and $\lambda _ { \mathrm { s e p } } ^ { \bar { } } = 0 . 1$ , respectively. The base and separation temperatures are $\tau _ { \mathrm { b a s e } } =$ $\tau _ { \mathrm { s e p } } = 0 . 1$ , the pseudo-label sharpening temperature is $\tau _ { \mathrm { s h a r p } } = 0 . 0 5$ , and the dual-level pseudo-label schedule has a linear 100-epoch ramp-up.

GET. Following GET [10], the Text Embedding Synthesizer (TES) maps visual embeddings to seven pseudo-text tokens, which are passed to the CLIP text encoder. The coefficient $\lambda = 0 . 3 5$ balances the supervised and unsupervised objectives in each modality branch. The coefficient $\lambda _ { c } ~ = ~ 1$ weights the cross-modal instance consistency loss, and $\tau _ { a } = 0 . 0 1$ is the temperature in the visual–textual alignment loss used to train TES. The remaining contrastive and student/teacher temperatures follow the corresponding SimGCD settings.

GET-TAR. GET-TAR [59] builds on GET and applies token reweighting, consistency-aware token sampling, and gated residual refinement before classification. The official GET-TAR implementation preserves the GET optimization settings, selects the top half of visual tokens $( \mathrm { t o p \_ m \_ r a t i o = 0 . 5 } )$ , and weights the additional sampling-distribution KL loss by 0.001.

## APPENDIX C

## ADDITIONAL EXPERIMENTAL RESULTS

## C.1 Additional Comparison between GH and SPE (RQ1)

Tab.9 provides the complete old-category, new-category, and overall results under GH and SPE. We define every difference as $\Delta _ { r } = \mathsf { A c c } _ { \mathrm { G H } } ^ { r } - \mathsf { A c c } _ { \mathrm { S P E } } ^ { \bar { r } }$ . It is important to distinguish $\Delta _ { \mathrm { n e w } }$ from $\Delta _ { \mathrm { a l l } }$ . The new-category gap $\Delta _ { \mathrm { n e w } } = \mathrm { N e w } _ { \mathrm { G H } } - \mathrm { N e w } _ { \mathrm { S P E } }$ can be positive, zero, or negative because GH maximizes accuracy over all categories rather than over the new subset alone: a global assignment may improve old-category accuracy while sacrificing some new-category predictions. By contrast, $\Delta _ { \mathrm { a l l } } = \tt A l l _ { G H } - A l l _ { S P E } \geq 0 ,$ , since the assignment space optimized by GH contains the semantics-preserving assignments allowed by SPE.

Observations. On the coarse-grained benchmarks, GH and SPE give the same scores for almost all available runs. The only visible difference is SimGCD–CLIP on CIFAR-100, where the changes remain small. This indicates that the two protocols behave similarly when old and new categories are well separated. The fine-grained benchmarks are more sensitive to the evaluation protocol. On Herbarium 19, GH increases the reported new-category and overall scores for every method, while SPE gives higher old-category accuracy in most cases. The largest differences occur for SimGCD–CLIP and GET–CLIP.

Calculation of the cycle diagnostics. Tab.4 compares GH and SPE on the same predictions; only the output-to-category matching changes. We use the confusion counts $\dot { C } _ { y  \hat { y } }$ and assignments $\mathcal { A } _ { \mathrm { 0 } }$ and $\mathcal { A } _ { \mathrm { G H } }$ introduced in the main text to compute the cycle diagnostics as follows.

From changed assignments to cycles. Starting from each category y, we follow $\boldsymbol { y } \mapsto \mathcal { A } _ { \mathrm { G H } } ( \mathcal { A } _ { 0 } ^ { - 1 } ( \boldsymbol { y } ) )$ until the sequence returns to its starting category. Repeating this procedure for unvisited categories constructs $\Gamma ( \mathcal { A } _ { \mathrm { G H } } , \mathcal { A } _ { 0 } )$ . We omit unchanged categories but include nontrivial cycles whose margins are zero. For each cycle, we compute $m _ { \gamma }$ using its definition in Thm.1 and sum them to obtain the total reported in Tab.4. For brevity in the numerical examples below, write this sum as

$$
M : = \sum _ { \gamma \in \Gamma ( \mathcal { A } _ { \mathrm { G H } } , \mathcal { A } _ { 0 } ) } m _ { \gamma } .
$$

We then obtain $\Delta _ { \mathrm { a l l } }$ from Eq.(27).

Examples using the counts in Tab.4. On CIFAR-10–DINO, no assignment changes, so M = 0 and $\Delta _ { \mathrm { a l l } } = 0$ . On CUB–CLIP, the four cycle margins are $0 , - 8 , - 2 , 0$ . Therefore,

$$
M = 0 - 8 - 2 + 0 = - 1 0 , \qquad \Delta _ { \mathrm { a l l } } = { \frac { 1 0 0 \times 1 0 } { 4 . 4 9 6 } } = 0 . 2 2 .
$$

On Stanford Cars–CLIP, one cycle gives $M = - 8 8$ and $\Delta _ { \mathrm { a l l } } = ( 1 0 0 \times 8 8 ) / 6 , 1 4 4 = 1 . 4 3$ . On Herbarium 19–CLIP, three cycles give

$$
M = - 1 , 3 9 1 - 1 , 6 4 4 - 4 1 = - 3 , 0 7 6 , \qquad \Delta _ { \mathrm { a l l } } = { \frac { 1 0 0 \times 3 , 0 7 6 } { 2 5 , 3 5 6 } } = 1 2 . 1 3 .
$$

What the Type (I) and Type (II) columns count. For each anchored old output, we check its GH-assigned category. Type (I) counts outputs reassigned to a different old category; Type (II) counts outputs reassigned to a new category. These are counts of output indices, not samples or cycles. For example, Herbarium 19 has 4 Type (I) and 119 Type (II) outputs, for $4 + 1 1 9 = 1 2 { \bar { 3 } }$ reassigned old outputs.

TABLE 9  
Detailed comparison between GH and SPE on coarse- and fine-grained benchmarks. GH–SPE differences are orange (positive) or blue (negative/zero).
<table><tr><td rowspan=1 colspan=12>Old categories        New categoriesMethodBackbone        All categoriesDatasetOldGH OldSPE  $\Delta _ { \mathrm { o l d } }$ NewGH NewSPE∆new AllGH AllSPE  $\Delta _ { \mathrm { a l l } }$ </td></tr><tr><td rowspan=2 colspan=12>(a) Coarse-grained benchmarksLegoGCD   DINO     95.76  95.76 0.00 97.93  97.93 0.00 97.21  97.210.00</td></tr><tr><td rowspan=1 colspan=3>97.9397.930.00</td><td rowspan=1 colspan=2>97.2197.21</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=6>SimGCD-BIA DINO     84.25  84.25 0.00</td><td rowspan=1 colspan=3>98.74  98.74 0.00</td><td rowspan=1 colspan=2>93.91  93.91</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=6>ProtoGCD   DINO     81.94  81.94 0.00</td><td rowspan=1 colspan=3>98.94  98.94 0.00</td><td rowspan=1 colspan=2>93.27  93.27</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=6>CIFAR-10    SimGCD    DINO     93.71  93.71 0.00</td><td rowspan=1 colspan=3>98.48  98.48 0.00</td><td rowspan=1 colspan=2>96.89  96.89</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=6>SimGCD    CLIP      94.75  94.75 0.00</td><td rowspan=1 colspan=3>98.11  98.11 0.00</td><td rowspan=1 colspan=2>96.99  96.99</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=6>GET-TAR   CLIP      94.70  94.70 0.00</td><td rowspan=1 colspan=3>98.42  98.420.00</td><td rowspan=1 colspan=2>97.18  97.18</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=6>GET        CLIP     94.87  94.87 0.00</td><td rowspan=1 colspan=3>98.36  98.36 0.00</td><td rowspan=1 colspan=2>97.19  97.19</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=6>LegoGCD   DINO     81.71  81.71 0.00</td><td rowspan=1 colspan=3>85.99  85.99 0.00</td><td rowspan=1 colspan=2>83.13  83.13</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SimGCD-BIA</td><td rowspan=1 colspan=3>DINO     75.50  75.50</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>78.60</td><td rowspan=1 colspan=2>78.60 0.00</td><td rowspan=1 colspan=2>76.53  76.53</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ProtoGCD</td><td rowspan=1 colspan=2>DINO     77.82</td><td rowspan=1 colspan=1>77.82</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>82.49</td><td rowspan=1 colspan=1>82.49</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>79.38  79.38</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>CIFAR-100</td><td rowspan=1 colspan=1>SimGCD</td><td rowspan=1 colspan=2>DINO     81.69</td><td rowspan=1 colspan=1>81.69</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>74.02</td><td rowspan=1 colspan=2>74.020.00</td><td rowspan=1 colspan=2>79.13  79.13</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SimGCD</td><td rowspan=1 colspan=2>CLIP      84.05</td><td rowspan=1 colspan=1>84.90</td><td rowspan=1 colspan=1>-0.85</td><td rowspan=1 colspan=1>73.75</td><td rowspan=1 colspan=2>71.682.07</td><td rowspan=1 colspan=2>80.62  80.50</td><td rowspan=1 colspan=1>0.12</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>GET-TAR   CLIP      85.68  85.68</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>71.69</td><td rowspan=1 colspan=2>71.69 0.00</td><td rowspan=1 colspan=2>81.01  81.01</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=4>GET        CLIP      85.69  85.69</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>71.12</td><td rowspan=1 colspan=2>71.12 0.00</td><td rowspan=1 colspan=2>80.83  80.83</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=3>LegoGCD   DINO     94.88</td><td rowspan=1 colspan=1>94.88</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>77.76</td><td rowspan=1 colspan=2>77.760.00</td><td rowspan=1 colspan=2>83.49  83.49</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SimGCD-BIA</td><td rowspan=1 colspan=2>DINO     86.05</td><td rowspan=1 colspan=1>86.05</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>78.70</td><td rowspan=1 colspan=1>78.70</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>81.16</td><td rowspan=1 colspan=1>81.16</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ProtoGCD</td><td rowspan=1 colspan=2>DINO     91.00</td><td rowspan=1 colspan=1>91.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>78.36</td><td rowspan=1 colspan=1>78.36</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>82.59</td><td rowspan=1 colspan=1>82.59</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>ImageNet-100</td><td rowspan=1 colspan=1>SimGCD</td><td rowspan=1 colspan=2>DINO     92.58</td><td rowspan=1 colspan=1>92.58</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>81.26</td><td rowspan=1 colspan=1>81.26</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>85.04</td><td rowspan=1 colspan=1>85.04</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SimGCD</td><td rowspan=1 colspan=2>CLIP      94.76</td><td rowspan=1 colspan=1>94.76</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>87.80</td><td rowspan=1 colspan=1>87.80</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>90.13</td><td rowspan=1 colspan=1>90.13</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>GET-TAR</td><td rowspan=1 colspan=2>CLIP     95.43</td><td rowspan=1 colspan=1>95.43</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>88.28</td><td rowspan=1 colspan=1>88.28</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>90.67</td><td rowspan=1 colspan=1>90.67</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>GET       CLIP      95.69</td><td rowspan=1 colspan=1>95.69</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>92.02</td><td rowspan=1 colspan=1>92.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>93.25  93.25</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>(b) Fine-grained b</td><td rowspan=1 colspan=4>enchmarks</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>LegoGCD</td><td rowspan=1 colspan=1>DINO</td><td rowspan=1 colspan=1>58.47</td><td rowspan=1 colspan=1>69.02-</td><td rowspan=1 colspan=1>10.55</td><td rowspan=1 colspan=1>37.48</td><td rowspan=1 colspan=1>17.98</td><td rowspan=1 colspan=1>19.50</td><td rowspan=1 colspan=1>44.82</td><td rowspan=1 colspan=1>35.83</td><td rowspan=1 colspan=1>8.99</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SimGCD-BIA</td><td rowspan=1 colspan=1>DINO</td><td rowspan=1 colspan=1>55.30</td><td rowspan=1 colspan=1>65.47-</td><td rowspan=1 colspan=1>10.17</td><td rowspan=1 colspan=1>35.65</td><td rowspan=1 colspan=1>16.34</td><td rowspan=1 colspan=1>19.31</td><td rowspan=1 colspan=1>42.53</td><td rowspan=1 colspan=1>33.53</td><td rowspan=1 colspan=1>9.00</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ProtoGCD</td><td rowspan=1 colspan=2>DINO     58.28</td><td rowspan=1 colspan=1>68.04</td><td rowspan=1 colspan=1>-9.76</td><td rowspan=1 colspan=1>35.92</td><td rowspan=1 colspan=1>18.52</td><td rowspan=1 colspan=1>17.40</td><td rowspan=1 colspan=1>43.74</td><td rowspan=1 colspan=1>35.84</td><td rowspan=1 colspan=1>7.90</td></tr><tr><td rowspan=1 colspan=1>Herbarium 19</td><td rowspan=1 colspan=1>SimGCD</td><td rowspan=1 colspan=2>DINO     57.47</td><td rowspan=1 colspan=1>67.09</td><td rowspan=1 colspan=1>-9.62</td><td rowspan=1 colspan=1>36.72</td><td rowspan=1 colspan=1>17.66</td><td rowspan=1 colspan=1>19.06</td><td rowspan=1 colspan=1>43.98</td><td rowspan=1 colspan=1>34.95</td><td rowspan=1 colspan=1>9.03</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SimGCD</td><td rowspan=1 colspan=2>CLIP      64.39</td><td rowspan=1 colspan=1>77.34</td><td rowspan=1 colspan=1>-12.95</td><td rowspan=1 colspan=1>38.26</td><td rowspan=1 colspan=1>12.63</td><td rowspan=1 colspan=1>25.63</td><td rowspan=1 colspan=1>47.40</td><td rowspan=1 colspan=1>35.27</td><td rowspan=1 colspan=1>12.13</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>GET-TAR</td><td rowspan=1 colspan=2>CLIP      65.08</td><td rowspan=1 colspan=1>78.59</td><td rowspan=1 colspan=1>-13.51</td><td rowspan=1 colspan=1>37.77</td><td rowspan=1 colspan=1>16.46</td><td rowspan=1 colspan=1>21.31</td><td rowspan=1 colspan=1>47.33</td><td rowspan=1 colspan=1>38.20</td><td rowspan=1 colspan=1>9.13</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>GET        CLIP      63.75</td><td rowspan=1 colspan=1>78.52</td><td rowspan=1 colspan=1>-14.77</td><td rowspan=1 colspan=1>41.11</td><td rowspan=1 colspan=1>17.10</td><td rowspan=1 colspan=1>24.01</td><td rowspan=1 colspan=1>49.03</td><td rowspan=1 colspan=1>38.591</td><td rowspan=1 colspan=1>0.44</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>LegoGCD</td><td rowspan=1 colspan=2>DINO     75.11</td><td rowspan=1 colspan=1>79.41</td><td rowspan=1 colspan=1>-4.30</td><td rowspan=1 colspan=1>43.83</td><td rowspan=1 colspan=1>40.72</td><td rowspan=1 colspan=1>3.11</td><td rowspan=1 colspan=1>54.02</td><td rowspan=1 colspan=1>53.32</td><td rowspan=1 colspan=1>0.70</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SimGCD-BIA</td><td rowspan=1 colspan=2>DINO     71.21</td><td rowspan=1 colspan=1>74.91</td><td rowspan=1 colspan=1>-3.70</td><td rowspan=1 colspan=1>45.16</td><td rowspan=1 colspan=1>42.19</td><td rowspan=1 colspan=1>2.97</td><td rowspan=1 colspan=1>53.65</td><td rowspan=1 colspan=1>52.85</td><td rowspan=1 colspan=1>0.80</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ProtoGCD</td><td rowspan=1 colspan=2>DINO     66.97</td><td rowspan=1 colspan=1>69.07</td><td rowspan=1 colspan=1>-2.10</td><td rowspan=1 colspan=1>45.21</td><td rowspan=1 colspan=1>43.59</td><td rowspan=1 colspan=1>1.62</td><td rowspan=1 colspan=1>52.29</td><td rowspan=1 colspan=1>51.89</td><td rowspan=1 colspan=1>0.40</td></tr><tr><td rowspan=1 colspan=1>Stanford Cars</td><td rowspan=1 colspan=1>SimGCD</td><td rowspan=1 colspan=2>DINO    72.01</td><td rowspan=1 colspan=1>74.11</td><td rowspan=1 colspan=1>-2.10</td><td rowspan=1 colspan=1>44.61</td><td rowspan=1 colspan=1>42.70</td><td rowspan=1 colspan=1>1.91</td><td rowspan=1 colspan=1>53.53</td><td rowspan=1 colspan=1>52.93</td><td rowspan=1 colspan=1>0.60</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SimGCD</td><td rowspan=1 colspan=2>CLIP      83.71</td><td rowspan=1 colspan=1>89.01</td><td rowspan=1 colspan=1>-5.30</td><td rowspan=1 colspan=1>64.06</td><td rowspan=1 colspan=1>59.38</td><td rowspan=1 colspan=1>4.68</td><td rowspan=1 colspan=1>70.46</td><td rowspan=1 colspan=1>69.03</td><td rowspan=1 colspan=1>1.43</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>GET-TAR   CLIP      82.51</td><td rowspan=1 colspan=1>95.75</td><td rowspan=1 colspan=1>-13.24</td><td rowspan=1 colspan=1>72.80</td><td rowspan=1 colspan=1>60.51</td><td rowspan=1 colspan=1>12.29</td><td rowspan=1 colspan=1>75.96</td><td rowspan=1 colspan=1>71.99</td><td rowspan=1 colspan=1>3.97</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>GET        CLIP      88.01</td><td rowspan=1 colspan=1>92.30</td><td rowspan=1 colspan=1>-4.29</td><td rowspan=1 colspan=1>74.58</td><td rowspan=1 colspan=1>70.75</td><td rowspan=1 colspan=1>3.83</td><td rowspan=1 colspan=1>78.96</td><td rowspan=1 colspan=1>77.77</td><td rowspan=1 colspan=1>1.19</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>LegoGCD</td><td rowspan=1 colspan=2>DINO     71.18</td><td rowspan=1 colspan=1>73.05</td><td rowspan=1 colspan=1>-1.87</td><td rowspan=1 colspan=1>57.69</td><td rowspan=1 colspan=1>55.72</td><td rowspan=1 colspan=1>1.97</td><td rowspan=1 colspan=1>62.19</td><td rowspan=1 colspan=1>61.50</td><td rowspan=1 colspan=1>0.69</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SimGCD-BIA</td><td rowspan=1 colspan=2>DINO     60.84</td><td rowspan=1 colspan=1>60.91</td><td rowspan=1 colspan=1>-0.07</td><td rowspan=1 colspan=1>62.00</td><td rowspan=1 colspan=1>60.89</td><td rowspan=1 colspan=1>1.11</td><td rowspan=1 colspan=1>61.61</td><td rowspan=1 colspan=1>60.90</td><td rowspan=1 colspan=1>0.71</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ProtoGCD</td><td rowspan=1 colspan=2>DINO     64.64</td><td rowspan=1 colspan=1>64.78</td><td rowspan=1 colspan=1>-0.14</td><td rowspan=1 colspan=1>59.13</td><td rowspan=1 colspan=1>58.09</td><td rowspan=1 colspan=1>1.04</td><td rowspan=1 colspan=1>60.97</td><td rowspan=1 colspan=1>60.32</td><td rowspan=1 colspan=1>0.65</td></tr><tr><td rowspan=1 colspan=1>CUB</td><td rowspan=1 colspan=1>SimGCD</td><td rowspan=1 colspan=2>DINO     66.64</td><td rowspan=1 colspan=1>65.04</td><td rowspan=1 colspan=1>1.60</td><td rowspan=1 colspan=1>58.09</td><td rowspan=1 colspan=1>58.39</td><td rowspan=1 colspan=1>-0.30</td><td rowspan=1 colspan=1>60.94</td><td rowspan=1 colspan=1>60.61</td><td rowspan=1 colspan=1>0.33</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SimGCD</td><td rowspan=1 colspan=2>CLIP      75.38</td><td rowspan=1 colspan=1>76.12</td><td rowspan=1 colspan=1>-0.74</td><td rowspan=1 colspan=1>65.10</td><td rowspan=1 colspan=1>64.40</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>68.53</td><td rowspan=1 colspan=1>68.31</td><td rowspan=1 colspan=1>0.22</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>GET-TAR</td><td rowspan=1 colspan=2>CLIP      77.79</td><td rowspan=1 colspan=1>78.79</td><td rowspan=1 colspan=1>-1.00</td><td rowspan=1 colspan=1>73.01</td><td rowspan=1 colspan=1>72.07</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>74.60</td><td rowspan=1 colspan=1>74.31</td><td rowspan=1 colspan=1>0.29</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>GET        CLIP      76.45  76.85</td><td rowspan=1 colspan=1>-0.40</td><td rowspan=1 colspan=1>74.97</td><td rowspan=1 colspan=1>74.74</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>75.47</td><td rowspan=1 colspan=1>75.44</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>LegoGCD   DINO     62.12</td><td rowspan=1 colspan=1>64.71</td><td rowspan=1 colspan=1>-2.59</td><td rowspan=1 colspan=1>50.52</td><td rowspan=1 colspan=1>47.53</td><td rowspan=1 colspan=1>2.99</td><td rowspan=1 colspan=1>54.39</td><td rowspan=1 colspan=1>53.25</td><td rowspan=1 colspan=1>1.14</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SimGCD-BIA</td><td rowspan=1 colspan=1>DINO</td><td rowspan=1 colspan=1>55.46</td><td rowspan=1 colspan=1>56.30</td><td rowspan=1 colspan=1>-0.84</td><td rowspan=1 colspan=1>50.40</td><td rowspan=1 colspan=1>49.84</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>52.09</td><td rowspan=1 colspan=1>51.99</td><td rowspan=1 colspan=1>0.10</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ProtoGCD</td><td rowspan=1 colspan=1>DINO</td><td rowspan=1 colspan=1>46.40</td><td rowspan=1 colspan=1>46.04</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>54.60</td><td rowspan=1 colspan=1>54.42</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>51.87</td><td rowspan=1 colspan=1>51.63</td><td rowspan=1 colspan=1>0.24</td></tr><tr><td rowspan=1 colspan=1>FGVC-Aircraft</td><td rowspan=1 colspan=1>SimGCD</td><td rowspan=1 colspan=2>DINO     58.58</td><td rowspan=1 colspan=1>60.14</td><td rowspan=1 colspan=1>-1.56</td><td rowspan=1 colspan=1>50.67</td><td rowspan=1 colspan=1>49.66</td><td rowspan=1 colspan=1>1.01</td><td rowspan=1 colspan=1>53.31</td><td rowspan=1 colspan=1>53.15</td><td rowspan=1 colspan=1>0.16</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SimGCD</td><td rowspan=1 colspan=3>CLIP      59.54  59.54</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>53.61</td><td rowspan=1 colspan=1>53.43</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>55.59</td><td rowspan=1 colspan=1>55.47</td><td rowspan=1 colspan=1>0.12</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>GET-TAR</td><td rowspan=1 colspan=3>CLIP      64.59  66.39</td><td rowspan=1 colspan=1>-1.80</td><td rowspan=1 colspan=1>58.26</td><td rowspan=1 colspan=1>57.09</td><td rowspan=1 colspan=1>1.17</td><td rowspan=1 colspan=1>60.37</td><td rowspan=1 colspan=1>60.19</td><td rowspan=1 colspan=1>0.18</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>GET        CLIP      64.59  67.65</td><td rowspan=1 colspan=3>-3.06 58.17  55.71</td><td rowspan=1 colspan=1>2.46</td><td rowspan=1 colspan=2>60.31  59.69</td><td rowspan=1 colspan=1>0.62</td></tr></table>

## C.2 Additional Results on the Coarse-Grained Benchmark (RQ2)

We evaluate the effectiveness of our method on the coarse-grained ImageNet-100 benchmark. The GH and SPE scores are identical for every reported baseline and +Ours run, indicating no observable GH-induced overestimation in these experiments. Nevertheless, MMO improves performance across all five configurations, with gains of 0.15–3.05 points in overall accuracy and 0.02–3.30 points in new accuracy. Thus, the method also generalizes to a coarse-grained setting where GH-induced overestimation is not observed.

TABLE 10  
Baseline and +Ours on ImageNet-100 under GH and SPE. Orange New/H cells indicate gains from the baseline.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="4">GH</td><td colspan="4">SPE</td></tr><tr><td>AllGH</td><td>OldGH</td><td>NewGH</td><td>HGH</td><td>AllSPE</td><td>OldSPE</td><td>NewSPE</td><td>HSPE</td></tr><tr><td rowspan="10">ImageNet-100</td><td>LegoGCD</td><td>DINO</td><td>83.49</td><td>94.88</td><td>77.76 85.47</td><td></td><td>83.49</td><td>94.88</td><td>77.76</td><td>85.47</td></tr><tr><td>+ Ours</td><td>DINO</td><td>84.89</td><td>94.87</td><td>79.88 86.73</td><td></td><td>84.89</td><td>94.87</td><td></td><td>79.88 86.73</td></tr><tr><td>SimGCD-BIA</td><td>DINO</td><td>81.16</td><td>86.05</td><td>78.7082.21</td><td></td><td>81.16</td><td>86.05</td><td></td><td>78.7082.21</td></tr><tr><td>+ Ours</td><td>DINO</td><td>84.21</td><td>92.11</td><td>80.24 85.77</td><td></td><td>84.21</td><td>92.11</td><td>80.24</td><td>85.77</td></tr><tr><td>ProtoGCD</td><td>DINO</td><td>82.59</td><td>91.00</td><td>78.36 84.21</td><td></td><td>82.59</td><td>91.00</td><td></td><td>78.36 84.21</td></tr><tr><td>+ Ours</td><td>DINO</td><td>82.83</td><td>91.68</td><td>78.38 84.51</td><td></td><td>82.83</td><td>91.68</td><td></td><td>78.38 84.51</td></tr><tr><td>SimGCD</td><td>DINO</td><td>85.04</td><td>92.58</td><td>81.26 86.55</td><td></td><td>85.04</td><td>92.58</td><td></td><td>81.26 86.55</td></tr><tr><td>+ Ours</td><td>DINO</td><td>85.19</td><td>92.89</td><td>81.32 86.72</td><td></td><td>85.19</td><td>92.89</td><td>81.32</td><td>86.72</td></tr><tr><td>SimGCD</td><td>CLIP</td><td>90.13</td><td>94.76</td><td>87.8091.15</td><td></td><td>90.13</td><td>94.76</td><td>87.80</td><td>91.15</td></tr><tr><td>+ Ours</td><td>CLIP</td><td>92.51</td><td>95.31</td><td>91.10 93.16</td><td></td><td>92.51</td><td>95.31</td><td>91.10</td><td>93.16</td></tr></table>

## C.3 Complete Results on the Large-Scale Benchmark (RQ1 and RQ2)

Tab.11 compares the three reproduced ImageNet-1K baselines. Following Tab.9, every difference is defined as $\begin{array} { r l } { \Delta _ { r } } & { { } = } \end{array}$ $\mathsf { A c c } _ { \mathrm { G H } } ^ { r } - \mathsf { A c c } _ { \mathrm { S P E } } ^ { r } .$ All three baselines exhibit the same protocol pattern. Compared with SPE, GH increases new-category accuracy by 5.19–7.36 points and overall accuracy by 1.41–2.23 points, while decreasing the reported old-category accuracy by 6.12–8.20 points. Thus, the GH–SPE discrepancy persists on the large-scale benchmark and is not tied to a single backbone.

TABLE 11  
Detailed GH–SPE comparison of reproduced baselines on ImageNet-1K. GH–SPE differences are orange (positive) or blue (negative/zero).
<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="3">Old categories</td><td colspan="3">New categories</td><td colspan="3">All categories</td></tr><tr><td>OldGH</td><td>OldSPE</td><td> $\Delta _ { \mathrm { o l d } }$ </td><td>NewGH</td><td>NewSPE</td><td> $\Delta _ { \mathrm { n e w } }$ </td><td>AllGH AllSPE</td><td></td><td> $\Delta _ { \mathrm { a l l } }$ </td></tr><tr><td>SimGCD</td><td>DINO</td><td>77.14</td><td>85.15</td><td>-8.01</td><td>46.94</td><td>39.58</td><td>7.36</td><td>57.04</td><td>54.81</td><td>2.23</td></tr><tr><td>LegoGCD</td><td>DINO</td><td>79.18</td><td>85.30</td><td>-6.12</td><td>54.88</td><td>49.69</td><td>5.19</td><td>63.01</td><td>61.60</td><td>1.41</td></tr><tr><td>SimGCD</td><td>CLIP</td><td>77.06</td><td>85.26</td><td>-8.20</td><td>48.41</td><td>41.27</td><td>7.14</td><td>57.99</td><td>55.98</td><td>2.01</td></tr></table>

TABLE 12  
ImageNet-1K comparison of SimGCD and +Ours. Orange New cells indicate gains from the baseline; higher scores are in bold.
<table><tr><td>Method</td><td>Backbone</td><td> $\mathrm { \tt A l l { G H } }$ </td><td> ${ \mathrm { O 1 d } } _ { \mathrm { G H } }$ </td><td> $\scriptstyle \mathrm { N e w } _ { \mathrm { G H } }$ </td><td>AllSPE</td><td> ${ \tt O 1 d _ { S P E } }$ </td><td> ${ \mathrm { N e w } } _ { \mathrm { S P E } }$ </td></tr><tr><td>SimGCD</td><td>DINO</td><td>57.04</td><td>77.14</td><td>46.94</td><td>54.81</td><td>85.15</td><td>39.58</td></tr><tr><td>+ Ours DINO</td><td></td><td>58.69</td><td>77.29</td><td>49.34</td><td>56.88</td><td>85.28</td><td>42.60</td></tr><tr><td>SimGCD</td><td>CLIP</td><td>57.99</td><td>77.06</td><td>48.41</td><td>55.98</td><td>85.26</td><td>41.27</td></tr><tr><td>+ Ours CLIP</td><td></td><td>59.03</td><td>77.63</td><td>49.68</td><td>57.35</td><td>85.10</td><td>43.40</td></tr></table>

Under both GH and SPE, +Ours improves the overall and new-category scores for both DINO and CLIP. The gains in New are 2.40/3.02 points for DINO and 1.27/2.13 points for CLIP under GH/SPE, respectively. The corresponding All gains are 1.65/2.07 and 1.04/1.37 points. These consistent improvements demonstrate that MMO remains effective when extended to the large-scale benchmark.

## C.4 Complete ρ-Sensitivity Results on an Imbalanced Benchmark (RQ2)

Herbarium 19 is a naturally long-tailed, class-imbalanced benchmark. Its nearly equal numbers of old and new categories do not imply equal sample frequencies, so the old-region sample prior in the unlabeled set cannot be recovered from category counts alone. We therefore require ρ (or ρb) to infer $p _ { \mathrm { o l d } }$ through Eq.(19). Tab.13 reports the complete GH and SPE metrics for the runs with manually specified $\widehat { \rho }$ and the TIcE estimate, which is described in Sec.D.2.

![](images/a74f7558002b4a29a31d7b0ff90573d971cb769258623745ab17050f1cc9e8e0.jpg)  
(a) CUB

![](images/1b520d333de5378ed381eea63564e886636c72e41bd43312db4e7a222133cb14.jpg)  
(b) Aircraft

![](images/10e485d44225ffdbde7fbe54a59d98be4cb9561e9c038394a035bc920dd1a771.jpg)  
(c) H19

![](images/639a7114d2ecb00472ef9b905e0ec69cf96b10c2decefa312a9832a1d6da06de.jpg)  
(d) Cars  
Fig. 7. Old–new trade-offs under GH for CLIP-based methods. The axes show old-category (horizontal) and new-category (vertical) accuracy (%). TAR denotes GET-TAR.

TABLE 13  
Complete sensitivity results on the imbalanced Herbarium 19 benchmark with SimGCD–CLIP. The true labeling fraction is $\rho = 0 . 5 0$ . TIcE estimates $\widehat { \rho }$ from the training data to infer the required sample-level prior $p _ { \mathrm { o l d } }$ . The best result in each metric column is in bold.
<table><tr><td rowspan="2">Variant</td><td rowspan="2"> $\rho$ </td><td rowspan="2"> $\widehat { \rho }$ </td><td rowspan="2"> $\widehat { p } _ { \mathrm { o l d } }$ </td><td colspan="4">GH</td><td colspan="4">SPE</td></tr><tr><td>AllGH</td><td>OldGH</td><td>NewGH</td><td>HGH</td><td>AllsPE</td><td>OldSPE</td><td>NewSPE</td><td>HSPE</td></tr><tr><td>Baseline</td><td>0.50</td><td>一</td><td></td><td>47.40</td><td>64.39</td><td>38.26</td><td>48.00</td><td>35.27</td><td>77.34</td><td>12.63</td><td>21.71</td></tr><tr><td>Manually specified</td><td>0.50</td><td>0.40</td><td>0.5247</td><td>51.83</td><td>64.31</td><td>45.12</td><td>53.03</td><td>50.30</td><td>67.50</td><td>41.04</td><td>51.04</td></tr><tr><td>Manually specified</td><td>0.50</td><td>0.50</td><td>0.3498</td><td>49.14</td><td>56.91</td><td>44.96</td><td>650.23</td><td>48.58</td><td>57.90</td><td>43.56</td><td>49.72</td></tr><tr><td>Manually specified 0.50</td><td></td><td>0.60</td><td>0.2332</td><td>46.28</td><td>48.47</td><td></td><td>45.1146.73</td><td>46.00</td><td>47.50</td><td>45.20</td><td>46.32</td></tr><tr><td>Manually specified 0.50</td><td></td><td>0.70</td><td>0.1499</td><td>43.26</td><td>40.06</td><td>44.99</td><td>42.38</td><td>42.76</td><td>37.89</td><td>45.37</td><td>41.29</td></tr><tr><td>TIcE estimate</td><td>0.50</td><td>0.3575</td><td>0.6287</td><td>50.34</td><td>64.64</td><td></td><td>42.6451.38</td><td>46.02</td><td>70.74</td><td>32.72</td><td>44.74</td></tr></table>

On Herbarium 19, increasing $\widehat { \rho }$ from 0.40 to 0.70 raises New from 41.04 to 45.37 but lowers All from 50.30 to 42.76. A modest deviation from the true $\rho = 0 . 5 0$ can even improve overall and harmonic-mean accuracy. At ${ \widehat { \rho } } = 0 . 4 0 ,$ the harmonic means reach $\mathrm { H } _ { \mathrm { G H } } = 5 3 . 0 3$ and $\mathrm { H } _ { \mathrm { S P E } } = 5 1 . 0 4 $ . The baseline’s large GH–SPE gaps indicate severe semantic mismapping. With the TIcE estimate, MMO improves All<sub>SPE</sub> and New<sub>SPE</sub> by 10.75 and 20.09 points over the baseline and narrows the GH–SPE gaps. Thus, MMO mitigates prediction errors hidden by semantic mismapping and remains competitive when $\rho$ is estimated automatically in a more practical setting.

## C.5 Old–New Trade-offs and Overall Accuracy (RQ2)

We visualize the CLIP-based SimGCD, GET-TAR (TAR), and GET results and their +Ours variants from Tab.5 on CUB, FGVC-Aircraft, Herbarium 19, and Stanford Cars. Figs.7 and 8 show old versus new-category accuracy, while Figs.9 and 10 show new-category versus overall accuracy.

Old–new trade-off. The SPE plots show that MMO often trades a small decrease in old-category accuracy for a larger gain in new-category accuracy. For SimGCD on CUB, new-category accuracy rises by 6.10 points while old-category accuracy falls by only 0.34 points. The new-category gains reach 13.45 points for GET-TAR on Stanford Cars and 30.93 points for SimGCD on Herbarium 19. Together with the harmonic means in Tab.5, these results show that MMO generally improves the old–new balance.

From new-category gains to overall gains. The New–All plots show that MMO consistently improves new-category accuracy and improves overall accuracy in most CLIP-based configurations. Under SPE, SimGCD on Stanford Cars gains 12.38 points in New and 6.80 points in All . TAR on Herbarium 19 gains 28.34 and 14.52 points in the same metrics. These results show that better recognition of new categories can also improve overall performance.

## C.6 Ablation of MMO Components (RQ2)

Tab.14 examines the effect of RBL and the additional contribution of NDL when combined with RBL. Compared with the baseline, RBL improves overall accuracy and the old–new harmonic mean on all three datasets under both GH and SPE. Under SPE, the gains in H<sub>SPE</sub> are 1.76, 2.59, and 5.62 points on CUB, FGVC-Aircraft, and Stanford Cars, respectively. Adding NDL to RBL further increases New by 2.83, 4.38, and 0.46 points, respectively, with corresponding gains of 1.51, 1.33, and 0.65 points in $\operatorname { H } _ { \mathrm { S P E } }$ . The same improvement in new accuracy and harmonic mean is observed under GH. These results show that RBL improves the old–new balance, while NDL provides an additional benefit in terms of new-category recognition and overall performance.

![](images/16f90ac67e0fd2f21a5b77cf1408ff42b55ee062c2e720482fbd9929915ef6af.jpg)  
(a) CUB

![](images/acf6143d21897509744022a2331ed6eccc6cfa7e1cb817a68cee3af3a59397e6.jpg)  
(b) Aircraft

![](images/a10a3cd8019381c64587a897ad5c3263bd444960e1857fa92296264d7dd1cd05.jpg)  
(c) H19

![](images/32eb1ca33f142762928d9422de7fcc2d304381ce252228020f6e264574d4f775.jpg)  
(d) Cars

Fig. 8. Old–new trade-offs under SPE for CLIP-based methods. The axes show old-category (horizontal) and new-category (vertical) accuracy (%). TAR denotes GET-TAR.  
![](images/4db26533bdcbe2e22ba6cbbdfaa1b5dfcb3d42c642f0a3750ef72c08b8b9eed8.jpg)  
(a) CUB

![](images/a5d4630c25fa4706d9eac58682212f025706b6abd521e2f347d710c248bcdfcd.jpg)  
(b) Aircraft

![](images/bbc04659d56c436e03d9ba189a3da18a7428cb6ec8130bb5beb49ff917466eb3.jpg)  
(c) H19

![](images/db13482bbfbeb07d0f64b5ab9e53d9ea4997a162b742c2825a577d2e84aef5dd.jpg)  
(d) Cars

Fig. 9. New-category versus overall accuracy under GH for CLIP-based methods. The axes show new-category (horizontal) and overall (vertical) accuracy (%). TAR denotes GET-TAR.  
![](images/789b5a4135dce4ee750823a243aa8d400ef19250ca716dc97c9f0e2899abfaca.jpg)  
(a) CUB

![](images/81a97af92731c8e0427e09089770baad6c3fa06341697cdb143f0bd19796eead.jpg)  
(b) Aircraft

![](images/d234bef49602f29c836b10a44d8ed78eef0c024589170965abff1edeb2d8afca.jpg)  
(c) H19

![](images/9c0805e9dbbe22aeba304260f5a0f910f0b37adf18323e8d807d9af12bdb5975.jpg)  
(d) Cars  
Fig. 10. New-category versus overall accuracy under SPE for CLIP-based methods. The axes show new-category (horizontal) and overall (vertical) accuracy (%). TAR denotes GET-TAR.

## C.7 Additional Results of the t-SNE Embedding Visualization (RQ2)

Figs.11–13 provide the complete baseline and +Ours visualizations for SimGCD, GET, and GET-TAR on Herbarium 19. Each comparison uses the same ten selected categories and the same color coding, with the baseline shown on the left and its +Ours counterpart on the right. The circles represent samples and the stars represent class prototypes. Because t-SNE is a nonlinear two-dimensional projection, these plots are used only for qualitative inspection; the distances defined below are computed in the original high-dimensional embedding space.

Across the three methods, some baseline categories already form compact clusters despite low New<sub>SPE</sub> scores, showing that visual compactness alone does not rule out semantic mismapping. With MMO, clusters generally become tighter and less mixed, while Intra decreases and Inter increases. The accuracy gains are much larger than the visible changes, suggesting that MMO improves feature separation and, more importantly, output–category alignment.

Intra-class distance. Let K denote the number of selected categories, let category c contain $n _ { c }$ samples, and let $\pmb { x } _ { c , i } \in \mathbb { R } ^ { D }$ be the high-dimensional embedding of its i-th sample. We first compute the centroid of category c as

$$
\pmb { \mu } _ { c } = \frac { 1 } { n _ { c } } \sum _ { i = 1 } ^ { n _ { c } } \pmb { x } _ { c , i } .\tag{55}
$$

TABLE 14  
Component ablation of MMO with SimGCD–CLIP on three fine-grained benchmarks. Orange New/H cells indicate gains from the baseline. Best/second-best scores are bold/underlined.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Setting</td><td rowspan="2">RBL NDL</td><td rowspan="2"></td><td colspan="4">GH</td><td colspan="4">SPE</td></tr><tr><td> $\mathbb { A } \mathbb { 1 } \mathbb { 1 } _ { \mathrm { G H } }$ </td><td>OldGH</td><td>NewGH</td><td>HGH</td><td>AllsPE</td><td>OldSPE</td><td>NewSPE</td><td>HSPE</td></tr><tr><td rowspan="3">CUB</td><td>Baseline</td><td>一</td><td>1</td><td>68.53</td><td>75.38</td><td>65.10</td><td>69.86</td><td>68.31</td><td>76.12</td><td>64.40</td><td>69.77</td></tr><tr><td>+RBL</td><td>√</td><td>一</td><td>70.55</td><td>76.38</td><td>67.63</td><td>71.74</td><td>70.40</td><td>75.85</td><td>67.67</td><td>71.53</td></tr><tr><td>+RBL+NDL</td><td>√</td><td>√</td><td>72.69</td><td>76.72</td><td>70.67</td><td>73.57</td><td>72.26</td><td>75.78</td><td>70.50</td><td>73.04</td></tr><tr><td rowspan="3">FGVC-Aircraft</td><td>Baseline</td><td>一</td><td>1</td><td>55.59</td><td>59.54</td><td>53.61</td><td>56.42</td><td>55.47</td><td>59.54</td><td>53.43</td><td>56.32</td></tr><tr><td>+RBL</td><td>√</td><td>一</td><td>57.79</td><td>63.99</td><td>54.69</td><td>58.98</td><td>57.57</td><td>65.07</td><td>53.82</td><td>58.91</td></tr><tr><td>+RBL+NDL</td><td>√</td><td>√</td><td>60.41</td><td>60.92</td><td>60.15</td><td>60.53</td><td>59.61</td><td>62.42</td><td>58.20</td><td>60.24</td></tr><tr><td rowspan="3">Stanford Cars</td><td>Baseline</td><td>一</td><td>一</td><td>70.46</td><td>83.71</td><td>64.06</td><td>72.58</td><td>69.03</td><td>89.01</td><td>59.38</td><td>71.24</td></tr><tr><td>+RBL</td><td>√</td><td>一</td><td>75.50</td><td>82.56</td><td>72.10</td><td>76.98</td><td>75.23</td><td>83.36</td><td>71.30</td><td>76.86</td></tr><tr><td>+RBL+NDL</td><td>√</td><td>√</td><td>76.11</td><td>83.46</td><td>72.56</td><td>77.63</td><td>75.83</td><td>84.26</td><td>71.76</td><td>77.51</td></tr></table>

The mean distance from the samples of this category to their centroid is

$$
d _ { \mathrm { i n t r a } } ^ { ( c ) } = \frac { 1 } { n _ { c } } \sum _ { i = 1 } ^ { n _ { c } } \| \pmb { x } _ { c , i } - \pmb { \mu } _ { c } \| _ { 2 } ,\tag{56}
$$

and the reported class-balanced statistic is

$$
\mathrm { I n t r a } = \frac { 1 } { K } \sum _ { c = 1 } ^ { K } d _ { \mathrm { i n t r a } } ^ { ( c ) } .\tag{57}
$$

A smaller Intra value indicates that samples from the same category form a more compact cluster. We average within each category before averaging across categories, so every selected category has the same weight. This is particularly important for the long-tailed Herbarium 19 benchmark, where otherwise a few frequent categories could dominate the statistic.

Inter-class distance. Using the same high-dimensional centroids, we compute the mean Euclidean distance between distinct categories, counting each combination once:

$$
\mathrm { I n t e r } = \frac { 2 } { K ( K - 1 ) } \sum _ { 1 \le c < d \le K } \| \pmb { \mu } _ { c } - \pmb { \mu } _ { d } \| _ { 2 } .\tag{58}
$$

For the ten visualized categories, this is the average of ${ \binom { 1 0 } { 2 } } = 4 5$ distances between distinct centroids. A larger Inter value indicates better separation between categories. Accordingly, the desired directions are Intra↓ and Inter↑. The two measures are complementary: compact clusters can still have nearby centroids, while well-separated centroids do not by themselves guarantee small within-class dispersion.

## C.8 Detailed Results with Estimated Category Numbers (RQ3)

Following Vaze et al. [3], we estimate the total category number $\widehat { C }$ and derive $\widehat { C } _ { \mathrm { n e w } } = \widehat { C } - C _ { \mathrm { o l d } }$ . Tab.15 reports the complete GH and SPE metrics under these estimates. The notation $C  { \widehat { C } }$ gives the ground-truth and estimated total category numbers. Semantic mismapping persists when the category number is estimated. For SimGCD–CLIP on Stanford Cars, the baseline GH–SPE gap in new-category accuracy is 3.91 points. MMO reduces this gap to 0.48 points and improves New , All , and H by 4.99, 3.71, and 3.57 points, respectively. MMO also improves new-category accuracy and the harmonic mean for both DINO configurations. Overall, the gains under estimated category numbers support the robustness and generalizability of MMO beyond the known-cardinality setting.

![](images/9fe214b5caf20cd515f8e075b117a855705e841c1dc01f444309bc11f8b28527.jpg)

Fig. 11. Comparative t-SNE visualization on Herbarium 19 for SimGCD (left) and SimGCD+Ours (right). Colors identify the same ten selected categories, circles denote samples, and stars denote class prototypes.  
![](images/3429a9383ea120971a5fb10705d9bcadaaa953be41feeec5f78702adfdef2607.jpg)

Fig. 12. Comparative t-SNE visualization on Herbarium 19 for GET (left) and GET+Ours (right). Several baseline categories are already locally compact despite the low New<sub>SPE</sub> score, while +Ours further tightens representative clusters.  
![](images/4a3f74baec47f2cfa41b0eeaf5695a6b4abad5a12fd3f9030d4ff69fb657dc3f.jpg)  
Fig. 13. Comparative t-SNE visualization on Herbarium 19 for GET-TAR (left) and GET-TAR+Ours (right). The relatively modest visual change compared with the large New<sub>SPE</sub> improvement is consistent with MMO mitigating prediction errors hidden by semantic mismapping.

TABLE 15  
Complete results with category numbers estimated following Vaze et al. [3]. Colored New/H cells show changes from the baseline (orange: gains; blue: decreases).
<table><tr><td rowspan="2">Dataset  $( C \to { \widehat { C } } )$ </td><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="3">GH</td><td colspan="4">SPE</td><td colspan="2">Difference</td></tr><tr><td>AllGH</td><td>OldGH</td><td> $\scriptstyle \mathrm { N e w } _ { \mathrm { G H } }$ </td><td>AllsPE</td><td>OldSPE</td><td>NewSPE</td><td>HSPE</td><td> $\Delta _ { \mathrm { n e w } } ~ \Delta _ { \mathrm { a l l } }$ </td><td></td></tr><tr><td rowspan="2">CUB (200 →231)</td><td>SimGCD</td><td>DINO</td><td>61.68</td><td>64.64</td><td>60.19</td><td>61.41</td><td>63.11</td><td>60.5661.81</td><td></td><td>-0.370.27</td><td></td></tr><tr><td>+ Ours</td><td>DINO</td><td>63.03</td><td>68.18</td><td>60.46</td><td>62.88</td><td>66.64</td><td>60.99</td><td>63.69</td><td>-0.530.15</td><td></td></tr><tr><td rowspan="2">CUB (200 →212)</td><td>SimGCD</td><td>CLIP</td><td>70.40</td><td>75.98</td><td>67.60</td><td>70.26</td><td>76.32</td><td>67.23 71.49</td><td></td><td>0.37 0.13</td><td></td></tr><tr><td>+ Ours CLIP</td><td></td><td>70.53</td><td>74.45</td><td>68.57</td><td>69.82</td><td>74.92</td><td>67.27</td><td>70.89</td><td>1.300.71</td><td></td></tr><tr><td rowspan="2">Stanford Cars (196 → 230)</td><td>SimGCD</td><td>DINO</td><td>52.88</td><td>70.16</td><td>44.53</td><td>52.47</td><td>72.91</td><td>42.60 53.78</td><td></td><td>1.93 0.41</td><td></td></tr><tr><td>+ Ours</td><td>DINO</td><td>53.65</td><td>68.07</td><td>46.68</td><td>53.37</td><td>68.97</td><td>45.84</td><td>55.07</td><td>0.84 0.28</td><td></td></tr><tr><td rowspan="2">Stanford Cars (196 → 234)</td><td>SimGCD</td><td>CLIP</td><td>73.57</td><td>82.41</td><td>69.30</td><td>72.51</td><td>87.26</td><td>65.39 74.76</td><td></td><td>3.91 1.06</td><td></td></tr><tr><td>+ Ours</td><td>CLIP</td><td>76.27</td><td>87.46</td><td>70.87</td><td>76.22</td><td>88.31</td><td>70.38 78.33</td><td></td><td>0.48 0.05</td><td></td></tr></table>

## APPENDIX D

## D.1 Minibatch Tolerance and Numerical Stability

The main text defines RBL using the fixed regional prior $p _ { r }$ . The tolerance-based variant used in our experiments accommodates differences between minibatches by replacing $p _ { r }$ with an effective target $p _ { r } ^ { ( \delta ) }$ :

$$
\begin{array} { r l } & { ~ \mathcal { T } = [ p _ { \mathrm { o l d } } - \delta , p _ { \mathrm { o l d } } + \delta ] \cap [ \epsilon , 1 - \epsilon ] , } \\ & { p _ { \mathrm { o l d } } ^ { ( \delta ) } = \mathrm { c l i p } ( \bar { q } ^ { \mathrm { o l d } } , \mathcal { T } ) , ~ p _ { \mathrm { n e w } } ^ { ( \delta ) } = 1 - p _ { \mathrm { o l d } } ^ { ( \delta ) } , } \\ & { \mathcal { L } _ { \mathrm { R B } } ^ { ( \delta ) } = - \displaystyle \sum _ { r \in \{ \mathrm { o l d } , \mathrm { n e w } \} } p _ { r } ^ { ( \delta ) } \log \bar { q } ^ { r } . } \end{array}\tag{59}
$$

The effective target is treated as constant during backpropagation. This variant applies no corrective gradient when the batch-average regional probabilities lie inside the allowed interval and pushes them toward the nearest boundary otherwise. The underlying prior $p _ { r }$ remains fixed; the effective target does not estimate or correct that prior. Setting $\delta = 0$ recovers the fixed-prior loss, subject to numerical clipping.

For numerical stability, $p _ { \mathrm { o l d } }$ is clipped to $[ \epsilon , 1 - \epsilon ]$ before constructing the interval, and $p _ { \mathrm { n e w } } = 1 - p _ { \mathrm { o l d } }$ . Probabilities inside logarithms are also clipped to this range, with $\epsilon = 1 0 ^ { - 6 }$ . We set the tolerance to $\delta = 0 . 0 5$ by default.

## D.2 Estimating an Unknown Labeling Fraction

On imbalanced benchmarks, the old- and new-category sample proportions cannot be inferred from category counts alone. When the labeling fraction $\rho$ is unknown, we estimate it with TIcE [49] and use the estimate to obtain the old-region sample prior required by RBL. TIcE uses frozen features from $\mathcal { D } _ { l } \cup \mathcal { D } _ { u }$ and the observed labeled/unlabeled status, without using hidden category labels.

Let $S = 1$ indicate a labeled sample, $Z = 1$ an old-category sample, and X its frozen feature representation. Assuming that old-category samples are selected for labeling independently of X with probability $\rho ,$ any feature-defined subset A satisfies

$$
\operatorname* { P r } ( S = 1 \mid X \in A ) = \rho \operatorname* { P r } ( Z = 1 \mid X \in A ) \leq \rho .\tag{60}
$$

The labeled fraction therefore approaches $\rho$ in subsets dominated by old-category samples. TIcE searches for such subsets with decision trees. It uses separate folds to select subsets and estimate their labeled fractions, avoiding evaluation on the same samples used to build the tree. For a subset A with $L _ { A }$ labeled samples among $T _ { A }$ samples, it computes the uncertainty-adjusted score

$$
b _ { A } = \frac { L _ { A } } { T _ { A } } - \sqrt { \frac { \rho _ { 0 } ( 1 - \rho _ { 0 } ) ( 1 - \eta ) } { \eta T _ { A } } } , \qquad \eta = \operatorname * { m a x } \left\{ 0 . 0 2 5 , \frac { 1 } { 1 + 0 . 0 0 4 T _ { \mathrm { e s t } } } \right\} ,\tag{61}
$$

where $T _ { \mathrm { e s t } }$ is the estimation-fold sample count. Starting with $\rho _ { 0 } = 0 . 5$ , TIcE takes the maximum of the subset scores and the overall labeled fraction in each estimation fold, then averages these estimates across folds. It repeats this step once with the updated $\rho _ { 0 }$ . We substitute the resulting $\widehat { \rho } _ { \mathrm { T I c E } }$ into Eq.(19) to obtain $\widehat { p } _ { \mathrm { o l d } }$