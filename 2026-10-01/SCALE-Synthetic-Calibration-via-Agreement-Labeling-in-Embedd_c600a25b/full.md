# SCALE: Synthetic Calibration via Agreement Labeling in Embedding Space

Wenjun Liu   
Department of Computer Science Dartmouth College Hanover, NH, USA   
wenjun.liu.gr@dartmouth.edu

Saeed Hassanpour Departments of Biomedical Data Science, Computer Science, and Epidemiology Dartmouth College, Hanover, NH, USA saeed.hassanpour@dartmouth.edu

## Abstract

Foundation models for computational pathology are usually evaluated using AUC and accuracy, while calibration is often left untested. This matters because a model can be accurate on average but still assign overly confident probabilities to cases that are difficult even for pathologists.

We study calibration across eight pathology foundation models. Using pathologist agreement as a measure of diagnostic difficulty, we find that calibration error is consistently higher on low-agreement cases than on high-agreement cases. This pattern is not apparent from aggregate expected calibration error (ECE) alone.

We then propose synthetic agreement calibration, a method for improving calibration without collecting multi-annotator labels. Given a trained linear probe, we select high-confidence embeddings as class anchors and interpolate between anchors from opposite classes. The interpolation weights encode a continuous notion of diagnostic ambiguity, which we use as a synthetic agreement signal to retrain the probe with agreement-aware label smoothing.

On MHIST, which includes annotations from seven pathologists, synthetic agreement calibration recovers most of the calibration improvement obtained by label smoothing based on real pathologist agreement, while substantially reducing lowagreement ECE relative to the uncalibrated baseline. Discrimination metrics are preserved. On PatchCamelyon and BreakHis, public histopathology datasets without multi-annotator labels, the method improves calibration across the evaluated foundation models, whereas annotator-dependent approaches cannot be used without additional expert annotation.

## 1 Introduction

Foundation models pretrained on large histopathology image collections have become standard feature extractors for computational pathology tasks [Chen et al., 2024, Xu et al., 2024, Lu et al., 2024, Wang et al., 2024]. The standard evaluation protocol measures discrimination—AUC, accuracy, F1—and reports a single aggregate number per dataset. Calibration, which measures whether a model’s predicted probability matches the empirical frequency of correctness, is almost never reported.

This omission matters clinically. If a model is overconfident on an ambiguous case—one where pathologists themselves disagree—this can misdirect triage or reduce the perceived need for additional workup. Discrimination tells us how well a model separates classes; calibration tells us whether the model knows what it does not know.

We make three contributions:

(1) A difficulty-stratified calibration audit. Rather than reporting a single ECE per model, we stratify test cases by inter-pathologist agreement and measure calibration separately on high-, medium , and low-agreement subsets. Across all eight models we evaluate, calibration error on low-agreement cases is markedly higher than on high-agreement cases (Figure 2). This pattern is entirely hidden by aggregate ECE.

(2) Synthetic agreement calibration (SCALE). We propose a lightweight probe-level calibration method that improves calibration without multi-annotator labels (Figure 1). The key idea is to generate a synthetic difficulty spectrum by interpolating between high-confidence class anchors in embedding space, then use the interpolation weights as synthetic agreement labels for agreement-aware label smoothing. Only the linear probe is retrained; the foundation model stays frozen.

(3) Validation across multiple datasets. On MHIST, where seven-pathologist annotations are available, SCALE recovers most of the calibration improvement of real-agreement label smoothing. On PatchCamelyon and BreakHis, where no multi-annotator labels exist, SCALE consistently reduces calibration error—a setting where annotator-dependent methods cannot be applied at all.

## 2 Related Work

Calibration of neural networks. Guo et al. [2017] showed that modern neural networks are poorly calibrated and that temperature scaling—learning a single scalar to rescale logits—is a strong post-hoc baseline. Other approaches include histogram binning [Zadrozny and Elkan, 2001], Platt scaling [Platt, 1999], and focal loss [Mukhoti et al., 2020]. Beyond in-distribution calibration, Ovadia et al. [2019] showed that calibration and predictive uncertainty can degrade under dataset shift, highlighting the importance of evaluating reliability beyond aggregate accuracy. These methods treat all test examples uniformly and do not account for variation in diagnostic difficulty.

Label smoothing and annotator agreement. Label smoothing [Szegedy et al., 2016, Müller et al., 2019] replaces hard one-hot labels with soft targets, improving both generalization and calibration. Wei et al. [2022] extended this to histopathology by conditioning the degree of smoothing on interannotator agreement: images with low pathologist agreement receive softer labels. They showed substantial ECE reductions on MHIST using linear, piecewise, and nonlinear smoothing functions. Their method requires per-image annotations from multiple pathologists, which is expensive and rarely available.

Pathology foundation models. Recent foundation models for pathology—UNI [Chen et al., 2024], GigaPath [Xu et al., 2024], CONCH [Lu et al., 2024], CHIEF [Wang et al., 2024], CTransPath [Wang et al., 2022], Phikon [Filiot et al., 2023], PLIP [Huang et al., 2023]—are evaluated on linear probing accuracy across downstream tasks. Calibration is not part of any standard evaluation protocol for these models. We fill this gap.

Interpolation in embedding space. Mixup [Zhang et al., 2018] and manifold mixup [Verma et al., 2019] interpolate representations for data augmentation to improve generalization. Our use of interpolation differs in purpose: we generate a synthetic difficulty spectrum to improve calibration, not to augment training data for discrimination.

## 3 Method

## 3.1 Setup and Notation

We consider the standard linear probing protocol for pathology foundation models. A frozen encoder $f _ { \theta }$ maps an input image x to an embedding $z = f _ { \boldsymbol { \theta } } ( x )$ . A linear probe $g _ { \phi }$ is trained on these embeddings to output a class probability $\hat { p } = \sigma ( g _ { \phi } ( z ) )$ for binary classification.

A model is well-calibrated if $\scriptstyle P ( Y = 1 \mid { \hat { p } } = p ) = p$ for all $p \in [ 0 , 1 ]$ . For binary classification, we bin samples by their predicted positive-class probability $\hat { p } _ { i }$ into $B$ equal-width bins and measure expected calibration error:

$$
\mathrm { E C E } = \sum _ { b = 1 } ^ { B } \frac { \lvert S _ { b } \rvert } { N } \left| \frac { 1 } { \lvert S _ { b } \rvert } \sum _ { i \in S _ { b } } y _ { i } - \frac { 1 } { \lvert S _ { b } \rvert } \sum _ { i \in S _ { b } } \hat { p } _ { i } \right|\tag{1}
$$

where $S _ { b } = \{ i : \hat { p } _ { i } \in I _ { b } \}$ is the set of samples whose predicted positive-class probability falls in bin $I _ { b }$ . We use $B { = } 1 5$ following Wei et al. [2022].

## 3.2 Difficulty-Stratified Evaluation

Rather than reporting a single ECE value, we partition the test set into strata by pathologist agreement:

• High agreement: 6/7 or 7/7 pathologists agree

• Medium agreement: 5/7 pathologists agree

• Low agreement: 3/7 or 4/7 pathologists agree

We compute ECE within each stratum. As shown in Section 5.2, this reveals that calibration error is concentrated in the low-agreement stratum across all models.

## 3.3 Synthetic Agreement Calibration (SCALE)

SCALE has three steps: anchor selection, embedding interpolation, and label-smoothed retraining.

Step 1: Anchor selection. From the training set, we select the K embeddings with the highest probe confidence for each class. Let $\mathcal { A } ^ { + } = \{ z _ { 1 } ^ { \mp } , \ldots , z _ { K } ^ { + } \}$ be the positive-class anchors (highest pˆ) and $\mathcal { A } ^ { - } = \{ z _ { 1 } ^ { - } , \ldots , z _ { K } ^ { - } \}$ be the negative-class anchors (lowest $\hat { p } )$

Step 2: Embedding interpolation. For each pair $( z _ { j } ^ { + } , z _ { k } ^ { - } )$ , we generate interpolated embeddings:

$$
z _ { j k } ^ { ( \alpha ) } = ( 1 - \alpha ) \cdot z _ { j } ^ { + } + \alpha \cdot z _ { k } ^ { - }\tag{2}
$$

where $\alpha \in \{ \frac { 1 } { N _ { a } } , \frac { 2 } { N _ { a } } , . . . , \frac { N _ { a } - 1 } { N _ { a } } \}$ . Here $N _ { a }$ is the number of annotator bins (for MHIST, $N _ { a } { = } 7 .$ giving 6 interpolation levels). The endpoints $\alpha { = } 0$ and $\alpha { = } 1$ are the anchors themselves and are not interpolated.

Embeddings near $\alpha { = } 0 . 5$ lie between class clusters and represent synthetically difficult cases. Those near $\alpha { = } 0$ or $\alpha { = } 1$ represent easy cases. The interpolation weight thus encodes a continuous notion of diagnostic ambiguity.

Step 3: Agreement-aware label smoothing. We convert the interpolation weight to a synthetic annotator count $n _ { k } = \mathrm { r o u n d } ( ( 1 - \alpha ) \cdot N _ { a } )$ ) and pass it through a smoothing function to obtain a soft label. Following Wei et al. [2022], we evaluate three smoothing functions (visualized in Figure 5, Appendix A):

Piecewise:

$$
h _ { \mathrm { p w } } ( n _ { k } ) = \left\{ \begin{array} { l l } { ( 1 - \Omega ) + \Omega \cdot \frac { n _ { k } - n _ { m } } { n _ { m } - 1 } } & { \mathrm { i f ~ } n _ { k } > n _ { m } } \\ { 0 . 5 } & { \mathrm { i f ~ } n _ { k } = n _ { m } } \\ { \Omega \cdot \frac { n _ { k } } { n _ { m } - 1 } } & { \mathrm { i f ~ } n _ { k } < n _ { m } } \end{array} \right.\tag{3}
$$

Linear:

$$
h _ { \mathrm { l i n } } ( n _ { k } ) = ( 1 - \alpha _ { s } ) \cdot \frac { n _ { k } } { N _ { a } } + \frac { \alpha _ { s } } { C }\tag{4}
$$

Nonlinear:

$$
h _ { \mathrm { n l } } ( n _ { k } ) = \sigma \bigg ( \Phi \cdot \bigg ( \frac { n _ { k } } { N _ { a } } - 0 . 5 \bigg ) \bigg )\tag{5}
$$

where $n _ { m } = \lceil N _ { a } / C \rceil$ , C is the number of classes, and $\Omega , \alpha _ { s } ,$ , Φ are hyperparameters.

The probe is retrained on the union of original training embeddings (hard labels) and interpolated embeddings (soft labels) using cross-entropy loss. The full procedure is illustrated in Figure 1 and detailed in Algorithm 1 (Appendix A).

![](images/b7f68a777ed3cf2bae2ff6c924404ab960363a749a9f41508a98c76689315c9e.jpg)  
Figure 1: Overview of SCALE. (a) The K highest-confidence embeddings per class are selected as anchors. (b) Linear interpolation between opposing-class anchors generates embeddings at varying synthetic difficulty levels, parameterized by α. Points near α=0.5 lie close to the decision boundary and represent synthetically hard cases. (c) The interpolation weight is mapped to a soft label through a smoothing function (piecewise shown). (d) The linear probe is retrained on original embeddings with hard labels plus interpolated embeddings with soft labels.

## 4 Experimental Setup

## 4.1 Foundation Models

We evaluate eight models: GigaPath [Xu et al., 2024], UNI [Chen et al., 2024], CONCH [Lu et al., 2024], CHIEF [Wang et al., 2024], CTransPath [Wang et al., 2022], Phikon [Filiot et al., 2023], PLIP [Huang et al., 2023], and a ResNet-50 pretrained on ImageNet [He et al., 2016]. These span self-supervised (DINOv2, iBOT), contrastive (CLIP-based), and supervised pretraining strategies. For each model, we freeze the encoder and train a linear probe.

## 4.2 Datasets

MHIST [Wei et al., 2021]. 3,152 colorectal polyp images (HP vs. SSA) with annotations from 7 board-certified GI pathologists per image. Train/test split: 2,175/977. This is our primary dataset because multi-annotator labels allow head-to-head comparison with real-agreement methods.

PatchCamelyon [Veeling et al., 2018]. 327,680 lymph node tissue patches (tumor vs. normal). No multi-annotator labels. This tests generalizability to a different organ and a much larger dataset.

BreakHis [Spanhol et al., 2016]. 7,909 breast tumor histopathology images (benign vs. malignant), acquired at multiple magnification factors (40X, 100X, 200X, 400X). We resize images to 224×224 for model input. No multi-annotator labels. This tests generalizability to a third organ type and a dataset with magnification variation.

## 4.3 Baselines

We compare SCALE against four baselines: (1) No calibration: linear probe with hard labels, no post-hoc adjustment; (2) Temperature scaling [Guo et al., 2017]: single learned temperature on a validation set; (3) Vanilla label smoothing [Szegedy et al., 2016]: uniform smoothing with a fixed ϵ=0.1 applied to all training examples, without any difficulty or agreement information; (4) Real-agreement label smoothing [Wei et al., 2022]: smoothing conditioned on pathologist agreement (MHIST only), using all three variants. Vanilla label smoothing is a particularly important baseline: if uniform smoothing already achieves most of the calibration benefit, there would be little value in generating synthetic agreement labels.

## 4.4 Metrics and Statistical Testing

We report ECE (15 bins) stratified by agreement level and overall, along with AUC and accuracy. All experiments use 10 random seeds. We conduct paired t-tests across seeds to assess whether SCALE significantly reduces ECE relative to the uncalibrated baseline $( p < 0 . 0 5 )$ . To compare SCALE with real-agreement smoothing, we report the absolute ECE gap and the percentage of real-agreement improvement recovered by SCALE. The reported p-values assess consistency of the SCALE-vs-baseline comparison across seeds; all conclusions are also supported by effect sizes in the tables.

Table 1: Replication of Wei et al. [2022] on MHIST. End-to-end ResNet, 10 seeds. Published values in parentheses.
<table><tr><td>Method</td><td>AUC (%)</td><td>ECE (%)</td></tr><tr><td>Baseline</td><td> $8 4 . 5 \pm 0 . 9 ( 8 4 . 7 \pm 0 . 8 )$ </td><td> $9 . 1 \pm 1 . 5 ( 8 . 9 \pm 1 . 4 ) $ </td></tr><tr><td>Vanilla LS</td><td> $8 5 . 4 \pm 0 . 7 ( 8 5 . 6 \pm 0 . 6 )$ </td><td> $3 . 4 \pm 0 . 7 ( 3 . 2 \pm 0 . 6 )$ </td></tr><tr><td>Agree-Linear</td><td> $8 5 . 9 \pm 0 . 8 ( 8 6 . 1 \pm 0 . 9 )$ </td><td> $5 . 5 \pm 0 . 8 ( 5 . 3 \pm 0 . 7 )$ </td></tr><tr><td>Agree-Piecewise</td><td> $8 6 . 2 \pm 0 . 5 ( 8 6 . 4 \pm 0 . 4 )$ </td><td> $3 . 1 \pm 0 . 5 ( 2 . 9 \pm 0 . 4 )$ </td></tr><tr><td>Agree-Nonlinear</td><td> $8 6 . 1 \pm 0 . 8 ( 8 6 . 3 \pm 0 . 7 )$ </td><td> $3 . 0 \pm 0 . 7 ( 2 . 8 \pm 0 . 6 )$ </td></tr></table>

Table 2: Baseline ECE stratified by agreement level (no calibration applied).
<table><tr><td>Model</td><td>Overall</td><td>High-Agree</td><td>Med-Agree</td><td>Low-Agree</td></tr><tr><td>GigaPath</td><td>0.045</td><td>0.035</td><td>0.065</td><td>0.111</td></tr><tr><td>UNI</td><td>0.022</td><td>0.018</td><td>0.045</td><td>0.150</td></tr><tr><td>CONCH</td><td>0.045</td><td>0.032</td><td>0.058</td><td>0.170</td></tr><tr><td>CHIEF</td><td>0.038</td><td>0.028</td><td>0.055</td><td>0.198</td></tr><tr><td>CTransPath</td><td>0.023</td><td>0.019</td><td>0.048</td><td>0.180</td></tr><tr><td>Phikon</td><td>0.039</td><td>0.030</td><td>0.060</td><td>0.210</td></tr><tr><td>PLIP</td><td>0.033</td><td>0.025</td><td>0.075</td><td>0.310</td></tr><tr><td>ResNet-50</td><td>0.031</td><td>0.024</td><td>0.055</td><td>0.195</td></tr></table>

## 5 Results

## 5.1 Pipeline Validation

We first replicate Wei et al. [2022] using their original setup—end-to-end ResNet on MHIST, 10 seeds—to verify our implementation. Table 1 shows our numbers match published values within error bars.

## 5.2 Calibration Audit

Table 2 and Figure 2 present baseline ECE stratified by agreement. Low-agreement ECE is consistently elevated across all models—ranging from 0.111 (GigaPath) to 0.310 (PLIP)—while highagreement ECE remains at or below 0.035 for all models.

## 5.3 SCALE vs. Baselines on MHIST

Table 3 and Figure 3 compare SCALE (piecewise, K=10) against baselines. SCALE reduces lowagreement ECE by 29.2% on average relative to baseline, achieving 83% of the improvement obtained by real-agreement piecewise smoothing (35.2% reduction). Notably, vanilla label smoothing—which applies uniform smoothing without any difficulty information—reduces low-agreement ECE by only 20.8%, compared to SCALE’s 29.2%. This additional 8.4 percentage-point reduction relative to the uncalibrated baseline suggests that the synthetic agreement signal provides calibration benefit beyond uniform smoothing alone. Calibration improvements do not come at an observed cost to AUC or accuracy; discrimination metrics remain within seed-level variation across all methods.

## 5.4 Pairwise: Real vs. Synthetic Agreement

To isolate the effect of real vs. synthetic agreement, we compare each smoothing function paired with either source. Table 4 reports low-agreement ECE averaged across all eight models. The absolute ECE gap is small and consistent across variants $( \Delta = + 0 . 0 1 1$ to +0.014). SCALE recovers 77–83% of the calibration improvement provided by real annotator agreement, with piecewise smoothing yielding the best ratio.

![](images/57c9b64fca42dfdc3315c9e991c3cb7a2d3cd1d1aa3b59eb691d2b57f44ad770.jpg)  
Figure 2: Baseline calibration error stratified by diagnostic difficulty. Low-agreement cases show markedly higher ECE across all eight models.

Table 3: Calibration and discrimination on MHIST, averaged across 8 foundation models (mean over 10 seeds). <sup>†</sup>Requires multi-annotator labels. $p \mathrm { : }$ paired t-test on model-averaged low-agreement ECE across 10 matched seeds. Per-model results with standard deviations are reported in Appendix B.
<table><tr><td>Method</td><td>Overall ECE</td><td>Low-Agree ECE</td><td>Low-Agree ∆%</td><td>AUC</td><td>p</td></tr><tr><td>No calibration</td><td>0.035</td><td>0.191</td><td></td><td>90.4</td><td></td></tr><tr><td>Temp scaling</td><td>0.031</td><td>0.175</td><td>-8.1%</td><td>90.4</td><td></td></tr><tr><td>Vanilla LS</td><td>0.030</td><td>0.151</td><td>-20.8%</td><td>90.4</td><td></td></tr><tr><td>Real-agree  $\mathrm { P W } ^ { \dagger }$ </td><td>0.028</td><td>0.123</td><td>-35.2%</td><td>90.3</td><td></td></tr><tr><td>SCALE (ours)</td><td>0.030</td><td>0.135</td><td>-29.2%</td><td>90.4</td><td>&lt;0.001</td></tr></table>

## 5.5 Ablation Studies

Number of anchors K. Figure 4 shows that SCALE is robust to K for $K \geq 1 0$ . K=5 performs slightly worse due to limited anchor diversity, but the method is not sensitive to this hyperparameter.

Anchor selection strategy. We compare SCALE (top-K by confidence) against a variant using randomly selected anchors (Table 8, Appendix B). Random anchors still outperform vanilla label smoothing (25.8% vs. 20.8% low-agreement ECE reduction), confirming that the interpolation structure provides value. High-confidence anchors yield a further improvement (29.2%), indicating that both the interpolation mechanism and anchor selection contribute to SCALE’s performance.

## 5.6 Generalization to Datasets without Multi-Annotator Labels

On PatchCamelyon and BreakHis (Table 5), multi-annotator labels are unavailable, so real-agreement methods cannot be applied. SCALE outperforms both the uncalibrated baseline and vanilla label smoothing across all eight models on both datasets, reducing overall ECE by 20.4% on average on PCam and 18.0% on BreakHis relative to baseline. Vanilla LS achieves intermediate results, confirming that the advantage of synthetic agreement over uniform smoothing generalizes beyond MHIST. AUC is preserved across all configurations (see Appendix B).

Summary. Across all experiments, a consistent picture emerges. On MHIST, SCALE reduces low-agreement ECE by 29.2% relative to baseline, recovering 83% of the improvement from realagreement label smoothing that requires seven pathologist annotations per image. Vanilla label smoothing achieves only 20.8%, confirming the value of synthetic agreement. On PatchCamelyon and BreakHis—where annotator-dependent methods cannot be applied—SCALE reduces overall ECE by 20.4% and 18.0% respectively, outperforming vanilla LS on both datasets. Discrimination metrics are preserved throughout.

![](images/193191ee5e0a8708b70fc3355ca018968f91773545c10f6724555a447b90c170.jpg)  
Figure 3: Low-agreement ECE across eight models. SCALE (blue) substantially improves over the uncalibrated baseline (red), temperature scaling (orange), and vanilla label smoothing (dark orange), tracking close to real-agreement smoothing (green). The gap between vanilla LS and SCALE demonstrates the value of difficulty-aware synthetic agreement beyond uniform smoothing.

Table 4: Real vs. synthetic agreement, same smoothing function (low-agreement ECE averaged across 8 models). SCALE recovers the majority of the real-agreement benefit across all three variants.
<table><tr><td>Smoothing</td><td>Real-Agree ECE</td><td>SCALE ECE</td><td>Δ ECE</td><td>% of Real Gain</td></tr><tr><td>Linear</td><td>0.143</td><td>0.154</td><td>+0.011</td><td>77%</td></tr><tr><td>Piecewise</td><td>0.123</td><td>0.135</td><td>+0.011</td><td>83%</td></tr><tr><td>Nonlinear</td><td>0.127</td><td>0.141</td><td>+0.014</td><td>78%</td></tr></table>

## 6 Discussion

Why does SCALE work? SCALE assumes that linear interpolation in a well-structured embedding space produces representations intermediate in diagnostic difficulty. The path from a high-confidence positive anchor to a high-confidence negative anchor passes through the decision boundary, traversing regions of increasing ambiguity. The smoothing function translates this geometric property into a training signal that teaches the probe to be less confident near the boundary.

Limitations. SCALE currently handles binary classification; multi-class extensions require interpolation paths between multiple class anchors. The method depends on embedding space quality: if the foundation model’s representations do not vary smoothly with diagnostic difficulty, the synthetic signal may not be meaningful. On datasets without multi-annotator labels, we can only evaluate overall ECE, not difficulty-stratified ECE. The low-agreement subset on MHIST contains 164 images, so ECE estimates on this subset have higher variance; we mitigate this by reporting results across 10 seeds and 8 models.

Clinical relevance. Improving calibration on ambiguous cases may support more reliable clinical decision support after external validation. SCALE lowers the barrier to developing and evaluating better-calibrated pathology models by removing the need for multi-annotator panels. However, improved calibration alone does not establish clinical safety, and SCALE is not intended to replace pathologist review.

![](images/50bfbbddcc663e3fe8a87796f0646340bfacf68e23497f43760f9f190b17095b.jpg)  
Figure 4: Sensitivity to K (averaged across 8 models). All variants plateau around K=10. Dashed lines: baseline (red) and real-agreement piecewise (green).

Table 5: Generalization to datasets without multi-annotator labels (overall ECE, mean over 10 seeds). SCALE outperforms both baseline and vanilla LS across all models. AUC and standard deviations are in Appendix B.
<table><tr><td></td><td colspan="3">PatchCamelyon</td><td colspan="3">BreakHis</td></tr><tr><td>Model</td><td>Base</td><td>Van. LS</td><td>SCALE</td><td>Base</td><td>Van. LS</td><td>SCALE</td></tr><tr><td>GigaPath</td><td>0.052</td><td>0.046</td><td>0.041</td><td>0.048</td><td>0.043</td><td>0.039</td></tr><tr><td>UNI</td><td>0.038</td><td>0.034</td><td>0.030</td><td>0.042</td><td>0.038</td><td>0.035</td></tr><tr><td>CONCH</td><td>0.048</td><td>0.043</td><td>0.038</td><td>0.050</td><td>0.045</td><td>0.041</td></tr><tr><td>CHIEF</td><td>0.045</td><td>0.040</td><td>0.036</td><td>0.055</td><td>0.049</td><td>0.045</td></tr><tr><td>CTransPath</td><td>0.035</td><td>0.031</td><td>0.028</td><td>0.046</td><td>0.041</td><td>0.038</td></tr><tr><td>Phikon</td><td>0.042</td><td>0.038</td><td>0.034</td><td>0.051</td><td>0.046</td><td>0.042</td></tr><tr><td>PLIP</td><td>0.058</td><td>0.051</td><td>0.045</td><td>0.062</td><td>0.055</td><td>0.050</td></tr><tr><td>ResNet-50</td><td>0.040</td><td>0.036</td><td>0.033</td><td>0.058</td><td>0.052</td><td>0.048</td></tr><tr><td>Average</td><td>0.045</td><td>0.040</td><td>0.036</td><td>0.052</td><td>0.046</td><td>0.042</td></tr></table>

## 7 Conclusion

We presented a difficulty-stratified calibration audit of eight pathology foundation models, showing that calibration error concentrates in diagnostically difficult cases. We proposed SCALE, which generates synthetic agreement labels through embedding-space interpolation and uses them for agreement-aware label smoothing. On MHIST, SCALE recovers most of the calibration benefit of methods that require real multi-annotator labels. On PatchCamelyon and BreakHis, where no such labels exist, SCALE consistently improves calibration in a setting where annotator-dependent methods cannot be applied. The method is model-agnostic, preserves discrimination, and requires no additional data collection.

## References

Richard J. Chen, Tong Ding, Ming Y. Lu, Drew F. K. Williamson, Guillaume Jaume, et al. Towards a general-purpose foundation model for computational pathology. Nature Medicine, 30(3):850–862, 2024. doi: 10.1038/s41591-024-02857-3.

Alexis Filiot, Ridouane Ghermi, Antoine Olivier, Paul Jacob, Lucas Fidon, Anne-Marie Kain, Christel Saillard, and Jean-Baptiste Schiratti. Scaling self-supervised learning for histopathology with masked image modeling. medRxiv, 2023. doi: 10.1101/2023.07.21.23292757.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning, pages 1321–1330, 2017.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

Zhi Huang, Federico Bianchi, Mert Yuksekgonul, Thomas J. Montine, and James Zou. A visuallanguage foundation model for pathology image analysis using medical twitter. Nature Medicine, 29(9):2307–2316, 2023. doi: 10.1038/s41591-023-02504-3.

Ming Y. Lu, Bowen Chen, Drew F. K. Williamson, Richard J. Chen, et al. A visual-language foundation model for computational pathology. Nature Medicine, 30(3):863–874, 2024. doi: 10.1038/s41591-024-02856-4.

Jishnu Mukhoti, Viveka Kulharia, Amartya Sanyal, Stuart Golodetz, Philip H. S. Torr, and Puneet K. Dokania. Calibrating deep neural networks using focal loss. In Advances in Neural Information Processing Systems, 2020.

Rafael Müller, Simon Kornblith, and Geoffrey E. Hinton. When does label smoothing help? In Advances in Neural Information Processing Systems, 2019.

Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift. In Advances in Neural Information Processing Systems, 2019.

John Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In Advances in Large Margin Classifiers, pages 61–74. MIT Press, 1999.

Fabio A. Spanhol, Luiz S. Oliveira, Caroline Petitjean, and Laurent Heutte. A dataset for breast cancer histopathological image classification. IEEE Transactions on Biomedical Engineering, 63 (7):1455–1462, 2016. doi: 10.1109/TBME.2015.2496264.

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In IEEE Conference on Computer Vision and Pattern Recognition, pages 2818–2826, 2016.

Bastiaan S. Veeling, Jasper Linmans, Jim Winkens, Taco Cohen, and Max Welling. Rotation equivariant cnns for digital pathology. In Medical Image Computing and Computer-Assisted Intervention, pages 210–218, 2018.

Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas, Aaron Courville, David Lopez-Paz, and Yoshua Bengio. Manifold mixup: Better representations by interpolating hidden states. In International Conference on Machine Learning, pages 6438–6447, 2019.

Xiyue Wang, Sen Yang, Jun Zhang, Minghui Wang, Jing Zhang, Wei Yang, Junzhou Huang, and Xiao Han. Transformer-based unsupervised contrastive learning for histopathological image classification. Medical Image Analysis, 81:102559, 2022. doi: 10.1016/j.media.2022.102559.

Xiyue Wang, Junhan Zhao, Eliana Marostica, Wei Yuan, Jiawen Jin, Jun Zhang, et al. A pathology foundation model for cancer diagnosis and prognosis prediction. Nature, 634(8035):970–978, 2024. doi: 10.1038/s41586-024-07894-z.

Jerry Wei, Arief Suriawinata, Bing Ren, Xiaoying Liu, Mikhail Lisovsky, Louis Vaickus, Charles Brown, Michael Baker, Naofumi Tomita, Lorenzo Torresani, Jason Wei, and Saeed Hassanpour. A petri dish for histopathology image analysis. In Artificial Intelligence in Medicine, pages 11–24. Springer, 2021.

Jerry Wei, Lorenzo Torresani, Jason Wei, and Saeed Hassanpour. Calibrating histopathology image classifiers using label smoothing. In Artificial Intelligence in Medicine, 2022.

Hanwen Xu, Naoto Usuyama, Jaspreet Bagga, Sheng Zhang, Ryan Rao, Tristan Naumann, Cliff Wong, et al. A whole-slide foundation model for digital pathology from real-world data. Nature, 630(8015):181–188, 2024. doi: 10.1038/s41586-024-07441-w.

Bianca Zadrozny and Charles Elkan. Obtaining calibrated probability estimates from decision trees   
and naive bayesian classifiers. In International Conference on Machine Learning, pages 609–616,   
2001.   
Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical   
risk minimization. In International Conference on Learning Representations, 2018.

## A Additional Experimental Details

SCALE algorithm. Algorithm 1 provides pseudocode for the full SCALE procedure. Figure 5 visualizes the three smoothing functions.

Algorithm 1 SCALE: Synthetic Agreement Calibration   
Require: Frozen encoder $f _ { \theta } ,$ , training set $\left\{ \left( x _ { i } , y _ { i } \right) \right\}$ , anchors $K$ , smoothing function h   
1: Train linear probe $g _ { \phi }$ on $z _ { i } = f _ { \theta } ( x _ { i } )$ with hard labels   
2: Select top-K positive anchors A<sup>+</sup> ${ \mathcal { A } } ^ { + }$ and top-K negative anchors $A ^ { - }$ by probe confidence   
3: for each pair $( z _ { j } ^ { + } , z _ { k } ^ { - } )$ do   
4: for $\alpha \in \{ \frac { 1 } { N _ { a } } , \frac { 2 } { N _ { a } } , \dots , \frac { N _ { a } - 1 } { N _ { a } } \}$ do   
5: $z _ { j k } ^ { ( \alpha ) } \gets ( 1 { - } \alpha ) \cdot z _ { j } ^ { + } + \alpha \cdot z _ { k } ^ { - }$   
6: $\tilde { y } _ { j k } ^ { ( \alpha ) } \gets h ( \mathrm { r o u n d } ( ( 1 { - } \alpha ) \cdot N _ { a } ) )$   
7: end for   
8: end for   
9: Retrain $g _ { \phi }$ on $\{ ( z _ { i } , y _ { i } ) \} \cup \{ ( z _ { j k } ^ { ( \alpha ) } , \tilde { y } _ { j k } ^ { ( \alpha ) } ) \}$   
10: return Calibrated probe $g _ { \phi }$

Linear   
1.0   
0.8   
Softlabely 0.6   
0.4   
0.2   
0.0   
0.00 0.25 0.50 0.75 1.00   
${ { \mathfrak { n } } _ { k } } / { N _ { a } }$

Piecewise   
1.0   
0.8   
Softlabely 0.6   
0.4   
0.2   
0.0   
0.00 0.25 0.50 0.75 1.00   
$\mathfrak { n } _ { k } / N _ { a }$

Nonlinear   
1.0   
0.8   
Softlabely 0.6   
0.4   
0.2   
0.0   
0.00 0.25 0.50 0.75 1.00   
${ \mathfrak { n } } _ { k } / N _ { a }$  
Figure 5: The three agreement-aware label smoothing functions from Wei et al. [2022]. The x-axis is annotator agreement $( n _ { k } / N _ { a } )$ ; the y-axis is the soft label.

Hyperparameters. For SCALE, we use $K { = } 1 0$ anchors per class and piecewise smoothing with Ω=0.4 as defaults. Temperature scaling fits a single parameter via L-BFGS on the validation set. Linear probes use Adam (lr=1e-3) for 50 epochs with early stopping on validation loss.

Foundation model details. All foundation models use their publicly released checkpoints with 224×224 input resolution. Input images are resized and normalized according to each model’s published preprocessing pipeline. ResNet-50 uses ImageNet-pretrained weights from torchvision.

Dataset splits. For MHIST, we use the original train/test split (2,175/977). A 10% random subset of training data is held out for validation (temperature scaling and early stopping). For PatchCamelyon, we use the official train/validation/test split. For BreakHis, we use a 70/10/20 train/validation/test split stratified by patient to avoid data leakage across magnifications.

ECE computation. We use 15 equal-width bins following Wei et al. [2022]. For binary classification, confidence is defined as the predicted probability of the positive class (not max probability), consistent with the original MHIST evaluation protocol.

Agreement strata sample sizes (MHIST test set). High agreement (6/7 or 7/7): 652 images (66.7%). Medium agreement (5/7): 161 images (16.5%). Low agreement (3/7 or 4/7): 164 images (16.8%). Total: 977.

Compute. Feature extraction for all eight models takes approximately 2 hours on MHIST, 12 hours on PatchCamelyon, and 1 hour on BreakHis, each on a single NVIDIA L40S GPU (46GB). Probe training and SCALE retraining add under 5 minutes per model per dataset.

Licenses. Datasets: MHIST (CC BY 4.0), PatchCamelyon (CC0 1.0), BreakHis (CC BY 4.0). We use all foundation model checkpoints under their stated research-use licenses or access terms, including GigaPath (research-use-only), UNI (Apache 2.0), CONCH (research-only), CHIEF (Apache 2.0), CTransPath (GPL-3.0), Phikon (Apache 2.0), PLIP (MIT), and ResNet-50 (torchvision, BSD-3- Clause). All use is for non-commercial research purposes.

## B Full Results with Standard Deviations

Table 6 reports the full mean ± standard deviation (over 10 random seeds) for the main MHIST comparison. Table 7 reports the same for PatchCamelyon and BreakHis.

Table 6: Full MHIST results with standard deviations (10 seeds). <sup>†</sup>Requires multi-annotator labels.
<table><tr><td>Model</td><td>Method</td><td>Overall ECE</td><td>Low-Agree ECE</td><td>AUC (%)</td><td>Acc (%)</td></tr><tr><td rowspan="5">GigaPath</td><td>No calibration</td><td>0.045±0.004</td><td>0.111±0.013</td><td>93.5±0.5</td><td>89.9±0.5</td></tr><tr><td>Temp scaling</td><td>0.040±0.004</td><td>0.095±0.011</td><td>93.5±0.5</td><td>89.9±0.5</td></tr><tr><td>Vanilla LS</td><td>0.039±0.004</td><td>0.085±0.010</td><td>93.4±0.4</td><td>89.8±0.4</td></tr><tr><td>Real-agree PW†</td><td>0.037±0.003</td><td>0.070±0.008</td><td>93.6±0.5</td><td>90.0±0.6</td></tr><tr><td>SCALE (ours)</td><td>0.039±0.003</td><td>0.078±0.008</td><td>93.5±0.5</td><td>89.9±0.6</td></tr><tr><td rowspan="5">UNI</td><td>No calibration</td><td>0.022±0.002</td><td>0.150±0.022</td><td>91.2±0.6</td><td>87.2±0.6</td></tr><tr><td>Temp scaling</td><td>0.020±0.002</td><td>0.140±0.020</td><td>91.2±0.6</td><td>87.2±0.6</td></tr><tr><td>Vanilla LS</td><td>0.019±0.002</td><td>0.116±0.013</td><td>91.3±0.5</td><td>87.3±0.6</td></tr><tr><td>Real-agree PW†</td><td>0.018±0.002</td><td>0.092±0.012</td><td>91.1±0.3</td><td>87.1±0.6</td></tr><tr><td>SCALE (ours)</td><td>0.019±0.002</td><td>0.100±0.013</td><td>91.2±0.3</td><td>87.2±0.6</td></tr><tr><td rowspan="5">CONCH</td><td>No calibration</td><td>0.045±0.004</td><td>0.170±0.024</td><td>92.1±0.6</td><td>88.1±0.4</td></tr><tr><td>Temp scaling</td><td>0.041±0.005</td><td>0.160±0.025</td><td>92.1±0.6</td><td>88.1±0.4</td></tr><tr><td>Vanilla LS</td><td>0.040±0.004</td><td>0.133±0.019</td><td>92.0±0.4</td><td>88.0±0.7</td></tr><tr><td>Real-agree PW†</td><td>0.038±0.003</td><td>0.105±0.012</td><td>92.2±0.4</td><td>88.2±0.5</td></tr><tr><td>SCALE (ours)</td><td>0.040±0.003</td><td>0.115±0.012</td><td>92.0±0.6</td><td>88.0±0.4</td></tr><tr><td rowspan="5">CHIEF</td><td>No calibration</td><td>0.038±0.004</td><td>0.198±0.023</td><td>90.5±0.4</td><td>86.5±0.5</td></tr><tr><td>Temp scaling</td><td>0.035±0.003</td><td>0.185±0.026</td><td>90.5±0.4</td><td>86.5±0.4</td></tr><tr><td>Vanilla LS</td><td>0.034±0.004</td><td>0.159±0.018</td><td>90.6±0.3</td><td>86.6±0.5</td></tr><tr><td>Real-agree PW†</td><td>0.032±0.003</td><td>0.130±0.014</td><td>90.4±0.5</td><td>86.4±0.6</td></tr><tr><td>SCALE (ours)</td><td>0.034±0.003</td><td>0.142±0.022</td><td>90.5±0.6</td><td>86.5±0.5</td></tr><tr><td rowspan="5">CTransPath</td><td>No calibration</td><td>0.023±0.002</td><td>0.180±0.019</td><td>89.8±0.6</td><td>85.8±0.4</td></tr><tr><td>Temp scaling</td><td>0.021±0.002</td><td>0.168±0.018</td><td>89.8±0.6</td><td>85.8±0.7</td></tr><tr><td>Vanilla LS</td><td>0.020±0.002</td><td>0.145±0.017</td><td>89.9±0.5</td><td>85.9±0.6</td></tr><tr><td>Real-agree PW†</td><td>0.019±0.002</td><td>0.118±0.013</td><td>89.7±0.6</td><td>85.7±0.5</td></tr><tr><td>SCALE (ours)</td><td>0.020±0.002</td><td>0.130±0.015</td><td>89.9±0.4</td><td>85.9±0.7</td></tr><tr><td rowspan="5">Phikon</td><td>No calibration</td><td>0.039±0.004</td><td>0.210±0.032</td><td>90.1±0.5</td><td>86.1±0.6</td></tr><tr><td>Temp scaling</td><td>0.036±0.005</td><td>0.195±0.031</td><td>90.1±0.5</td><td>86.1±0.5</td></tr><tr><td>Vanilla LS</td><td>0.035±0.003</td><td>0.169±0.024</td><td>90.0±0.5</td><td>86.0±0.4</td></tr><tr><td>Real-agree PW†</td><td>0.033±0.003</td><td>0.140±0.017</td><td>90.2±0.5</td><td>86.2±0.5</td></tr><tr><td>SCALE (ours)</td><td>0.035±0.004</td><td>0.152±0.023</td><td>90.1±0.5</td><td>86.1±0.6</td></tr><tr><td rowspan="5">PLIP</td><td>No calibration</td><td>0.033±0.003</td><td>0.310±0.043</td><td>88.3±0.4</td><td>84.3±0.4</td></tr><tr><td>Temp scaling</td><td>0.028±0.002</td><td>0.280±0.040</td><td>88.3±0.4</td><td>84.3±0.5</td></tr><tr><td>Vanilla LS</td><td>0.027±0.002</td><td>0.251±0.029</td><td>88.4±0.3</td><td>84.4±0.6</td></tr><tr><td>Real-agree  $\mathrm { P W } ^ { \dagger }$ </td><td>0.025±0.002</td><td>0.215±0.030</td><td>88.2±0.5</td><td>84.2±0.6</td></tr><tr><td>SCALE (ours)</td><td>0.027±0.003</td><td>0.232±0.035</td><td>88.3±0.6</td><td>84.3±0.5</td></tr><tr><td rowspan="5">ResNet-50</td><td>No calibration</td><td>0.031±0.004</td><td>0.195±0.024</td><td>87.5±0.3</td><td>83.5±0.4</td></tr><tr><td>Temp scaling</td><td>0.027±0.002</td><td> $0 . 1 7 8 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td>87.5±0.3</td><td>83.5±0.5</td></tr><tr><td>Vanilla LS</td><td>0.026±0.002</td><td>0.149±0.023</td><td>87.4±0.4</td><td>83.4±0.6</td></tr><tr><td>Real-agree PW†</td><td>0.024±0.002</td><td>0.118±0.016</td><td>87.6±0.3</td><td>83.6±0.5</td></tr><tr><td>SCALE (ours)</td><td>0.026±0.003</td><td>0.130±0.013</td><td>87.5±0.3</td><td>83.5±0.6</td></tr></table>

Table 7: Full generalization results with standard deviations (10 seeds). Vanilla LS included for comparison. AUC is reported for SCALE; baseline and Vanilla LS AUC remain within seed-level variation.
<table><tr><td></td><td colspan="4">PatchCamelyon</td><td colspan="4">BreakHis</td></tr><tr><td>Model</td><td>Base ECE</td><td>Van. LS ECE</td><td>SCALE ECE</td><td>AUC</td><td>Base ECE</td><td>Van. LS ECE</td><td>SCALE ECE</td><td>AUC</td></tr><tr><td>GigaPath</td><td> $0 . 0 5 2 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td>0.046±0.005</td><td>0.041±0.004</td><td>96.2±0.3</td><td>0.048±0.005</td><td>0.043±0.004</td><td> $0 . 0 3 9 { \pm } 0 . 0 0 4$ </td><td>94.5±0.4</td></tr><tr><td>UNI</td><td> $0 . 0 3 8 { \pm } 0 . 0 0 4$ </td><td>0.034±0.003</td><td>0.030±0.003</td><td>95.8±0.3</td><td>0.042±0.004</td><td>0.038±0.004</td><td> $0 . 0 3 5 { \pm } 0 . 0 0 4$ </td><td>93.8±0.4</td></tr><tr><td>CONCH</td><td> $0 . 0 4 8 { \pm } 0 . 0 0 5$ </td><td> $0 . 0 4 3 { \pm } 0 . 0 0 4$ </td><td>0.038±0.004</td><td>95.5±0.3</td><td>0.050±0.005</td><td>0.045±0.005</td><td> $0 . 0 4 1 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td>93.2±0.4</td></tr><tr><td>CHIEF</td><td> $0 . 0 4 5 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td>0.040±0.004</td><td>0.036±0.004</td><td>94.8±0.3</td><td>0.055±0.006</td><td>0.049±0.005</td><td> $0 . 0 4 5 { \pm } 0 . 0 0 5$ </td><td>92.5±0.4</td></tr><tr><td>CTransP.</td><td> $0 . 0 3 5 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 0 3 1 { \pm } 0 . 0 0 3$ </td><td>0.028±0.003</td><td>93.5±0.3</td><td>0.046±0.005</td><td>0.041±0.004</td><td> $0 . 0 3 8 { \pm } 0 . 0 0 4$ </td><td>91.8±0.5</td></tr><tr><td>Phikon</td><td> $0 . 0 4 2 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 0 3 8 { \pm } 0 . 0 0 4$ </td><td>0.034±0.003</td><td>94.2±0.3</td><td>0.051±0.005</td><td>0.046±0.005</td><td> $0 . 0 4 2 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td>92.0±0.4</td></tr><tr><td>PLIP</td><td> $0 . 0 5 8 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 0 5 1 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td>0.045±0.004</td><td>93.0±0.3</td><td>0.062±0.006</td><td>0.055±0.006</td><td> $0 . 0 5 0 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td>90.5±0.5</td></tr><tr><td>ResNet-50</td><td> $0 . 0 4 0 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 0 3 6 { \pm } 0 . 0 0 4$ </td><td>0.033±0.003</td><td>92.5±0.3</td><td>0.058±0.006</td><td>0.052±0.005</td><td>0.048±0.005</td><td>89.8±0.5</td></tr></table>

Table 8: Anchor selection ablation (low-agreement ECE on MHIST, piecewise smoothing, K=10).
<table><tr><td>Model</td><td>Vanilla LS</td><td>Random Anchors</td><td>SCALE (Top-K)</td></tr><tr><td>GigaPath</td><td>0.085</td><td>0.081</td><td>0.078</td></tr><tr><td>UNI</td><td>0.116</td><td>0.106</td><td>0.100</td></tr><tr><td>CONCH</td><td>0.133</td><td>0.122</td><td>0.115</td></tr><tr><td>CHIEF</td><td>0.159</td><td>0.149</td><td>0.142</td></tr><tr><td>CTransPath</td><td>0.145</td><td>0.136</td><td>0.130</td></tr><tr><td>Phikon</td><td>0.169</td><td>0.159</td><td>0.152</td></tr><tr><td>PLIP</td><td>0.251</td><td>0.240</td><td>0.232</td></tr><tr><td>ResNet-50</td><td>0.149</td><td>0.138</td><td>0.130</td></tr><tr><td>Average</td><td>0.151</td><td>0.141</td><td>0.135</td></tr></table>