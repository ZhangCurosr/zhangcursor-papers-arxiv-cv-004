# Revisiting Handcrafted Minutiae Detection: A Simple and Efective Open Source Baseline for Modern Fingerprint Workflows

Rafaele Cappelli

Department of Computer Science and Engineering, University of Bologna, Cesena, Italy raffaele.cappelli@unibo.it

## Abstract

Handcrafted minutiae detection algorithms remain fundamental to biometric science and forensic practice due to their full auditability, adherence to international standards, and operational independence from training datasets or GPU hardware. However, current open-source traditional baselines are severely outdated, relying almost exclusively on legacy C/C++ codebases that lack seamless integration with modern scientific software ecosystems. To bridge this gap, the present work introduces SBMEX (Skeleton-Based Minutiae EXtraction), a fast and deterministic minutiae detection method integrated into the open source pyfing package. SBMEX achieves high computational throughput by employing a dual Look-Up Table architecture that replaces runtime neighborhood scanning during Crossing Number computation and skeleton tracking. Additionally, it incorporates a continuous quality scoring framework driven by tracking path length, dual ridge-valley skeleton fusion, and spatial density decay. Rigorous evaluation on NIST SD302 datasets demonstrates that SBMEX delivers feature extraction accuracy comparable to or outperforming traditional open-source baselines without fine-tuning, while achieving a drastic reduction in minutiae detection latency relative to classical Crossing Number Python implementations.

Keywords: Fingerprint, Minutiae, Skeleton, Crossing Number, Open source

## 1 Introduction

Fingerprint recognition remains one of the most reliable and widely deployed biometric technologies for personal identification in civilian, law enforcement, and forensic applications [1]. Automated Fingerprint Identification Systems (AFIS) rely heavily on minutiae—local ridge discontinuities—as the primary distinctive features for representation and matching. Among various minutia types, ridge endings (points where a ridge abruptly terminates) and bifurcations (points where a ridge forks into two branches) constitute the vast majority of extracted features (see Fig. 1.d). Despite decades of advancement, accurate minutiae detection remains a challenging task, particularly when processing low-quality impressions such as latent fingermarks characterized by severe noise, incomplete ridge structures, and background interference [1].

In recent years, state-of-the-art performance in minutiae detection has been increasingly driven by deep neural networks [2]. Nevertheless, the traditional handcrafted minutiae detection pipeline, historically developed over more than half a century, retains paramount importance in biometric science and forensic practice. A typical classical pipeline (Fig. 1) operates sequentially through image enhancement, binarization, skeletonization, and minutiae detection via connectivity analysis. This deterministic paradigm ofers some critical advantages:

• Explainability and legal admissibility: Handcrafted feature extraction directly mirrors the explicit, rule-based procedures employed by human latent examiners, providing full algorithmic transparency and auditability required in judicial settings where black-box deep learning models may face strict legal evidentiary hurdles.

• Standardization compliance: International biometric standards, such as ISO/IEC 19794-2 [3], define minutiae attributes (e.g., location, direction, and type) based on explicit geometric rules along skeletonized ridge lines, ensuring seamless interoperability across heterogeneous system architectures.

![](images/713cfbc6dea11c9aff276eff9682454665655a9a27dfcda437ea2f7bba953295.jpg)  
Figure 1: Classical minutiae extraction pipeline. (a) Portion of a fingerprint image displaying dark gray ridges separated by lighter valleys. (b) Enhanced and binarized image, where ridges are mapped to white pixels and valleys to black pixels. (c) Topological skeletonization of the ridges (1-pixel wide white paths). (d) Extracted minutiae features, highlighting a ridge ending near the top (blue circle) and a bifurcation located lower towards the center (red circle). The line segment originating from the center of each circle indicates the direction of the corresponding minutia, defined by an angle in [0, 2π).

• Computational eficiency and zero-shot deployment: Handcrafted algorithms execute deterministically without requiring extensive training datasets, high-end GPU hardware, or hyperparameter fine-tuning across domain shifts, enabling immediate zero-shot deployment on edge devices or resource-constrained environments.

Despite the ongoing necessity for traditional minutiae detection, the current landscape of open-source baselines for classical pipelines is severely outdated. The research community relies almost exclusively on legacy software tools, primarily NIST MINDTCT [4] and FingerJet [5]. While foundational, these packages present major operational drawbacks: they are written in legacy C/C++ codebases, are rigid to modify, and lack native integration with modern scientific software ecosystems, which are now predominantly Python-based. Consequently, researchers aiming to prototype or benchmark traditional workflows are forced to resort to ineficient subprocess wrappers around compiled binaries or to implement ad-hoc scripts from scratch; a practical constraint illustrated by Oblak et al. [6], who had to integrate MINDTCT and FingerJet via Python wrappers alongside a custom Crossing Number (CN) implementation. The persistence of this limitation is evidenced by numerous very recent works [7, 8, 9, 10], which continue to depend on legacy tools like MINDTCT as their primary handcrafted baseline due to the absence of a modern alternative.

To bridge this gap, the present work introduces SBMEX (Skeleton-Based Minutiae EXtraction), a fast, deterministic, and fully auditable minutiae detection algorithm designed to provide a modern reference baseline for traditional fingerprint pipelines. The main contributions of this work are summarized as follows:

• A novel dual LookUp-Table (LUT) architecture that accelerates both CN computation and ridge tracking, drastically speeding up skeleton traversal without complex conditional branching.

• A continuous quality scoring paradigm that replaces traditional binary hard-filtering. Initial minutia quality is scaled continuously with tracking path length and attenuated via exponential cluster decay and dual-skeleton (ridge-valley) agreement.

• A flexible detection framework allowing lightweight ridgeonly analysis or full dual-skeleton merging to adapt to processing constraints.

• An extensive zero-shot evaluation on the challenging NIST SD302 benchmark, demonstrating that SBMEX consistently outperforms or matches traditional baselines without requiring domain-specific fine-tuning.

• A fully open-source Python implementation integrated into the pyfing ecosystem, ofering the research community an easily accessible, modular, and reproducible tool.

The remainder of this paper is organized as follows. Section 2 reviews related work on traditional and learningbased minutiae detection algorithms. Section 3 describes the mathematical formulation and structural mechanics of the proposed algorithm. Section 4 presents the experimental setup, comparative evaluations on benchmark datasets, and performance analysis. Finally, Section 5 concludes the paper and outlines future directions.

## 2 Related work

This section reviews the main paradigms in minutiae extraction, covering traditional frameworks, deep learning models, and the existing open-source traditional baselines against which the proposed SBMEX is evaluated.

## 2.1 Traditional methods

Traditional algorithms are broadly categorized into binarization-based and direct gray-scale approaches.

Binarization-based frameworks enhance the input image, segment ridge-valley patterns, and tipycally compute a onepixel-wide skeleton. Minutiae are subsequently detected by applying local connectivity rules, such as the CN metric, defined as the total number of 0 → 1 pixel transitions along the ordered 3×3 neighborhood sequence [1]. Because binarization is sensitive to noise, these pipelines heavily rely on heuristic post-processing rules to eliminate spurious detections caused by scars, dry skin, or low contrast.

To avoid information loss and artifacts introduced by binarization and skeletonization, direct gray-scale methods extract minutiae directly from raw (or enhanced) intensity values. These algorithms typically rely on ridge-following strategies along local orientation vectors [11] or model minutiae as local discontinuities in the ridge flow [12].

## 2.2 Deep learning-based methods

Deep learning has significantly advanced minutiae extraction by replacing handcrafted heuristics with end-to-end learned feature representations.

Patch-based and Multi-stage Networks. Early deep learning frameworks utilized sliding-window CNNs to classify local patches as minutia or non-minutia [13, 14]. To improve context awareness, multi-stage architectures introduced region proposal networks followed by refinement subnetworks to regress coordinates and directions [15, 16, 17]. While accurate, these sequential pipelines sufer from high computational overhead.

Single-pass Fully Convolutional Pipelines. State-of-theart approaches deploy single-pass fully convolutional networks that process the entire image in a single forward pass [18, 19, 20, 2]. These architectures output dense feature maps encoding minutia location, direction, and type simultaneously, achieving high accuracy and computational eficiency even on low-quality fingerprints.

## 2.3 Existing open-source traditional baselines

To evaluate SBMEX within a rigorous and reproducible framework, its performance is compared against three representative open-source traditional algorithms, each embodying a distinct extraction paradigm.

Developed by NIST within the NBIS suite [4], MINDTCT relies on directional filtering for binarization, local connectivity analysis for candidate extraction, and heuristic rules to eliminate false minutiae.

CNBaseline is the traditional algorithm in [6], which applies standard morphological skeletonization to a prebinarized image and applies the CN to identifies candidate minutiae. A final distance-based filtering pass removes closely adjacent candidate pairs.

FingerJet [5] is an open-source lightweight engine representative of direct gray-scale extraction methods, since code analysis reveals that it avoids explicit binarization and skeletonization.

Table 1: Summary of SBMEX parameters and default values at 500 DPI.
<table><tr><td>Symbol</td><td>Description</td><td>Default</td><td>Unit</td></tr><tr><td> $t _ { b }$ </td><td>Global binarization threshold</td><td>0</td><td></td></tr><tr><td> $d _ { \mathrm { b g } }$ </td><td>Minimum distance to mask border</td><td>10</td><td>px</td></tr><tr><td> $d _ { \mathrm { m i n } }$ </td><td>Minimum tracking distance</td><td>5</td><td>px</td></tr><tr><td> $d _ { \mathrm { m a x } }$ </td><td>Maximum tracking distance</td><td>20</td><td>px</td></tr><tr><td> $d _ { \mathrm { p a i r } }$ </td><td>Max pairing distance</td><td>6</td><td>px</td></tr><tr><td> $\theta _ { \mathrm { p a i r } }$ </td><td>Max pairing angle difference</td><td> $\pi / 6$ </td><td>rad</td></tr><tr><td> $w _ { \mathrm { u n p a i r } }$ </td><td>Penalty for unpaired minutiae</td><td>0.5</td><td></td></tr><tr><td> $\delta _ { \mathrm { v a l } }$ </td><td>Spatial shift for valley minutiae</td><td>3</td><td>px</td></tr><tr><td> $\sigma _ { \mathrm { c l } }$ </td><td>Interaction radius for clusters</td><td>11</td><td>px</td></tr><tr><td> $\gamma _ { \mathrm { c l } }$ </td><td>Quality decay base factor</td><td>0.5</td><td></td></tr></table>

## 3 Proposed method: SBMEX

This section details SBMEX, a fast and deterministic minutiae detection method designed for standard fingerprint processing pipelines.

## 3.1 High-level overview

Given an input grayscale image $I ( x , y ) \in [ 0 , 2 5 5 ]$ and a segmentation mask $M ( x , y ) \in \{ 0 , 1 \}$ , SBMEX executes in five sequential stages:

1. Binarization and thinning: The image I is binarized within M using a global threshold $t _ { b }$ to yield a binary ridge map $B _ { R } ( x , y ) = \mathbb { I } ( I ( x , y ) \le t _ { b } \land M ( x , y ) = 1 )$ where $\mathbb { I } ( \cdot )$ is the indicator function. The parallel thinning algorithm of Guo and Hall [21] reduces $B _ { R }$ to a 1-pixel-wide skeleton $S _ { R } ( x , y ) \in \dot { \{ 0 , 1 \} }$

2. Candidate detection and perimeter filtering: Candidate minutiae are extracted on $S _ { R }$ using a branchless CN lookup, discarding spurious candidates located near the background boundary of M (Section 3.2).

3. Skeleton tracking and direction estimation: Validated candidates undergo LUT-accelerated ridge-following to compute their direction $\theta \in [ 0 , 2 \pi )$ and initial quality score $q _ { 0 } \in ( 0 , 1 ]$ based on the tracked path length (Section 3.3).

4. Dual ridge-valley fusion (optional): To improve extraction in low-contrast areas, steps 1–3 can be mirrored on the inverted binary valley map $B _ { V } ( x , y ) = \mathbb { I } ( I ( x , y ) >$ $t _ { b } \land M ( x , y ) = 1 )$ . Inverted valley minutiae are geometrically shifted and paired with ridge minutiae, penalizing candidates lacking dual-domain agreement (Section 3.4).

5. Cluster-based quality decay: Final quality scores are continuously attenuated via spatial density decay to down-weight spurious minutiae clusters (Section 3.5).

The global parameters governing SBMEX are summarized in Table 1 and detailed in the following subsections.

## 3.2 Candidate detection and perimeter filtering

To avoid runtime conditional loops, SBMEX encodes the 8-neighborhood of each skeleton pixel into an 8-bit integer state $E ( x , y ) \in [ 0 ,$ , 255] via 2D spatial cross-correlation:

$$
E ( x , y ) = \sum _ { k = 0 } ^ { 7 } 2 ^ { k } \cdot S _ { R } ( x + \Delta x _ { k } , y + \Delta y _ { k } )\tag{1}
$$

Table 2: Representative $3 \times 3$ neighborhood patterns, their encoded state $E ( x , y )$ , CN values from $\mathrm { L U T } _ { C N }$ , and valid successor directions $\mathcal { D } _ { \mathrm { n e x t } }$ from $\mathrm { L U T } _ { N D }$ for reachable prior directions $d _ { \mathrm { p r e v } } \in \{ 0 , \ldots , 7 \}$ and start state $( d _ { \mathrm { p r e v } } = 8 )$
<table><tr><td>Pattern</td><td> $E ( x , y )$ </td><td> $\mathrm { L U T } _ { C N }$ </td><td></td><td> $\mathrm { L U T } _ { N D } ( d _ { \mathrm { p r e v } } \to [ \mathcal { D } _ { \mathrm { n e x t } } ^ { 1 } , \mathcal { D } _ { \mathrm { n e x t } } ^ { 2 } , . . . ] )$ </td></tr><tr><td><img src="images/dc04f797bcebffd3954f08d4f246932845e07952c33d6c485bb60e88b115cf3d.jpg"/></td><td></td><td>0</td><td></td><td></td></tr><tr><td><img src="images/f0c3be4c14ba994539d221b693dd73da021874ddf311178d52128f8eb99be1cd.jpg"/></td><td>0 1</td><td>1</td><td> $4  [ ]$ </td><td></td></tr><tr><td></td><td>…</td><td>.</td><td> $8  [ 0 ]$ </td><td></td></tr><tr><td><img src="images/27d1281a419b5f6b9db7bed9750d413ce997f88928451e8969bc91d3fe33ac42.jpg"/></td><td>32</td><td>1</td><td> $1  [ ]$  8 → [5]</td><td></td></tr><tr><td><img src="images/af4beb8ffcf8ec8c743f36ab4bbc9c926186552cbe11519c4f9b4d9aed315034.jpg"/></td><td>33</td><td>2</td><td> $1  [ 0 ] ; 4  [ 5 ]$  8 → [0, 5]</td><td></td></tr><tr><td>0</td><td>…</td><td>.</td><td></td><td></td></tr><tr><td>2 3</td><td>37</td><td>3</td><td> $1 \to [ 0 , 2 ] ; 4 \to [ 5 , 2 ] ; 6 \to [ 5 , 0 ]$ </td><td></td></tr><tr><td>54</td><td></td><td></td><td> $8  [ 0 , 2 , 5 ]$ </td><td></td></tr></table>

where $( \Delta x _ { k } , \Delta y _ { k } )$ denote the spatial ofset vectors for the 8-neighborhood, ordered clockwise starting from $( - 1 , - 1 )$ Formulating this step as a standard 2D correlation allows the algorithm to leverage highly optimized routines from mainstream scientific computing libraries.

The CN value for each pixel $C N ( x , y )$ is subsequently retrieved in $O ( 1 )$ time using a pre-computed lookup table $\mathrm { L U T } _ { C N }$

$$
C N ( x , y ) = \left\{ { \begin{array} { l l } { \mathrm { L U T } _ { C N } [ E ( x , y ) ] } & { { \mathrm { i f } } \ S _ { R } ( x , y ) = 1 } \\ { 0 } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.\tag{2}
$$

Pixels with $C N ( x , y ) = 1$ are candidate ridge endings, while $C N ( x , y ) = 3$ identifies candidate bifurcations. All other configurations are ignored. Representative neighborhood patterns and their corresponding encoded byte $E ( x , y )$ and $\mathrm { L U T } _ { C N }$ values are illustrated in Table 2.

Boundary artifacts are eliminated using the Chebyshev distance transform $D _ { M } ( x , y )$ of the segmentation mask M. Candidates satisfying $D _ { M } ( x , y ) \leq d _ { \mathrm { b g } }$ are pruned prior to ridge tracking.

## 3.3 Skeleton tracking and direction estimation

Validated candidates undergo fast skeleton traversal to estimate their direction $\theta \in [ 0 , 2 \pi )$ and initial quality score $q _ { 0 }$ Path traversal is modeled as a deterministic finitestate transition governed by a pre-computed state table $\mathrm { L U T } _ { N D } [ b ] [ d _ { \mathrm { p r e v } } ]$ , where $b = E ( x , y )$ is the current neighborhood state and $d _ { \mathrm { p r e v } } \in \{ 0 , \ldots , 7 , 8 \}$ encodes the entry direction $( d _ { \mathrm { p r e v } } = 8$ denotes initial candidate departure). As shown in Table $2 , \mathrm { L U T } _ { N D }$ directly returns an ordered list of valid successor directions $\mathcal { D } _ { \mathrm { n e x t } }$ . This state transition mechanism intrinsically prevents 180-degree backtracking and orders candidate branches by minimal angular deflection relative to $d _ { \mathrm { p r e v } }$ , bypassing conditional branching during line tracking.

Tracking traverses the skeleton until an intersection/endpoint $\left( C N \neq 2 \right)$ is encountered or the cumulative path distance reaches $d _ { \mathrm { m a x } }$ . The minutia direction θ and efective path length L are determined as follows:

• Ridge endings: L is the length of the single tracked path, and θ is the orientation of the line segment connecting the candidate location to the final reached location.

• Bifurcations: Tracking proceeds independently along all three departing branches. L is computed as the arithmetic mean of the three branch lengths, whereas θ is defined as the bisecting direction of the two branches forming the acute enclosed angle.

Candidates with path length $L < d _ { \mathrm { m i n } }$ are discarded as short noise artifacts. Validated minutiae receive an initial quality score proportional to their tracking extent, defined as $q _ { 0 } = L / d _ { \mathrm { m a x } }$

## 3.4 Dual ridge-valley fusion

When dual-skeleton extraction is enabled, candidate detection and tracking are performed independently on both the ridge binary map $B _ { R }$ and the inverted valley binary map $B _ { V }$ . To map valley minutiae $\mathcal { M } _ { V }$ into the ridge domain, their structural types are inverted (endings become bifurcations and vice versa), and their spatial coordinates are shifted by $\delta _ { \mathrm { v a l } }$ pixels along (for endings) or opposite to (for bifurcations) their direction θ.

Bipartite matching then pairs candidates between ridge set $\mathcal { M } _ { R }$ and transformed valley set $\mathcal { M } _ { V }$ within a maximum distance tolerance $d _ { \mathrm { p a i r } }$ and angular tolerance $\theta _ { \mathrm { p a i r } } .$ Unpaired minutiae from both domain sets are preserved to ensure high recall, but their quality scores are penalized by factor $w _ { \mathrm { u n p a i r } }$ , yielding $q = q _ { 0 } \cdot w _ { \mathrm { u n p a i r } }$

## 3.5 Cluster-based quality decay

High minutiae density typically indicates localized background noise, scars, or skeletonization artifacts. Rather than enforcing hard distance pruning—which risks removing legitimate closely-spaced minutiae—SBMEX applies a continuous quality attenuation based on local proximity.

For a set of extracted candidate minutiae $\begin{array} { r l } { \mathcal { M } } & { { } = } \end{array}$ $\{ m _ { 1 } , m _ { 2 } , . . . \}$ , the spatial proximity $\Psi _ { i j }$ between any pair of distinct minutiae $m _ { i } , m _ { j } \in { \mathcal { M } }$ separated by Euclidean distance $d _ { i j } = \lVert ( x _ { i } , y _ { i } ) - ( \bar { x } _ { j } , y _ { j } ) \rVert _ { 2 }$ is modeled via a Gaussian kernel:

$$
\Psi _ { i j } = \exp \left( - \frac { d _ { i j } ^ { 2 } } { 2 \sigma _ { \mathrm { c l } } ^ { 2 } } \right)\tag{3}
$$

where $\sigma _ { \mathrm { c l } }$ represents the cluster interaction radius. The cumulative density index for minutia $m _ { i }$ is given by $D _ { i } =$ $\textstyle \sum _ { j \neq i } \Psi _ { i j }$ . The final adjusted quality score $q _ { i } ^ { * }$ is calculated via exponential decay: $q _ { i } ^ { * } = q _ { i } \cdot \gamma _ { \mathrm { c l } } ^ { D _ { i } }$ , where $\gamma _ { \mathrm { c l } } \in ( 0 , 1 ]$ is the decay base factor. Highly dense clusters experience exponential quality attenuation, providing soft-scoring that downstream matchers can easily threshold.

## 4 Experimental evaluation

This section evaluates the accuracy and computational eficiency of SBMEX in comparison with the three traditional open-source baselines described in Section 2.3.

## 4.1 Datasets and enhancement pipeline

Experiments are conducted on two distinct subsets of the NIST SD302 benchmark [22, 23]:

![](images/9bcdae43c0ed03af056c473b732bf9f38f5aecc345dcfce2f9964578967ff653.jpg)  
Figure 2: Representative samples from the NIST SD302 datasets: (a) high-quality optical exemplar impression from SD302g, and (b) challenging latent impression from SD302h showing partial ridge patterns and severe background noise.

• Exemplars (NIST SD302g): A collection of 2,380 highquality exemplar fingerprints acquired at 1000 or 500 DPI using optical sensors.

• Latents (NIST SD302h): A challenging set of 3,815 latent impressions left naturally by subjects during simulated daily activities (e.g., handling paper currency or physical objects) and developed via standard forensic chemical and physical techniques. All images in this subset are categorized as “value for comparison” by latent examiners.

Both subsets feature ground-truth minutiae annotations meticulously validated by certified forensic examiners [23]. While the Latent subset is accompanied by manually annotated segmentation masks, those for Exemplar impressions are automatically generated via standard thresholding owing to their uniform white background. To standardize the evaluation across all experiments, all images are downsampled to a spatial resolution of 500 DPI. Representative samples from both subsets are shown in Fig. 2.

For algorithms requiring an almost-binary enhanced input (namely SBMEX and CNBaseline), a traditional deterministic enhancement pipeline is deployed using the open-source pyfing ecosystem [24]. Specifically, raw input images undergo four sequential processing stages: i) Ridge pattern segmentation using GMFS [25]; ii) Orientation field estimation via GBFOE [26]; iii) Spatial ridge frequency estimation via XSFFE [27]; iv) Image enhancement using GBFEN [28].

## 4.2 Evaluation protocol

To ensure fair and rigorous evaluation, the default parameters of SBMEX (summarized in Table 1) were tuned ofline on the FVC2000 DB2-A benchmark [29]. This dataset consists of fingerprint impressions acquired from untrained subjects without quality control, providing a representative baseline for parameter optimization. Crucially, FVC2000 DB2-A is completely disjoint from the NIST SD302 evaluation sets in terms of both subjects and acquisition sensor technologies. All experimental evaluations across both NIST SD302 subsets are conducted using these exact fixed parameters, without any dataset-specific fine-tuning.

The systematic evaluation protocol established in [2] is adopted to assess intrinsic minutiae detection accuracy independently of downstream matching heuristics, which may mask detection errors (e.g., in position, direction, or type) via error-tolerant algorithms. The protocol comprises two main evaluation steps:

1. Minutiae pairing and filtering: Extracted and groundtruth minutiae are paired by bipartite matching [30]. A match is classified as a True Positive (TP) if the spatial distance and angular deviation are within specified thresholds $( \rho _ { t } , \theta _ { t } )$ Three increasingly stringent threshold levels are evaluated: $\left( \rho _ { t } = 1 6 \mathrm { p x } , \theta _ { t } = \pi / 6 \mathrm { r a d } \right)$ (12 px, π/8 rad), and (8 px, π/10 rad). The evaluation region is restricted to the ground-truth segmentation bounding box, excluding a 14 px boundary margin to eliminate upstream segmentation biases (see [2]).

2. Aggregate Performance Metrics: Precision-Recall (PR) curves are generated by sweeping the confidence threshold for extractors that produce quality scores, reporting the maximum achievable F<sub>1</sub>-score. For CNBaseline, which yields unranked detections without confidence scores, performance is evaluated at its single operational point. Both type-agnostic and type-aware matching regimes are evaluated, where the latter requires matching minutia types for a TP (see [2]).

## 4.3 Results and discussion

Table 3 presents the quantitative comparison (F<sub>1</sub>-scores across the three threshold levels) on the NIST SD302 benchmark datasets under both type-agnostic and type-aware matching regimes. To isolate feature extraction performance from pre-processing efects, all extractors capable of processing grayscale images—encompassing both traditional baselines (MINDTCT, FingerJet) and state-of-theart methods (VeriFinger, LEADER)—are evaluated on both original (Raw) inputs and pre-processed (Enh) inputs enhanced via the pyfing pipeline (Section 4.1)<sup>1</sup>. Although passing enhanced images to of-the-shelf tools may induce redundant internal filtering steps not originally intended by their design, it provides the most rigorous way to evaluate these tools “as-is” under identical pre-processing conditions. Commercial (VeriFinger) and deep learning (LEADER) methods are included on both inputs for completeness, serving as external benchmarks for context, and are excluded from the traditional baseline ranking. All reported F -score diferences are statistically robust; 95% confidence intervals estimated via 1,000 non-parametric image-leve bootstrap resamples were omitted from Table 3 for space constraints, as they all fall within ≤ 0.01.

On Exemplar impressions, SBMEX achieves top-ranking performance among traditional open-source tools. In the type-agnostic regime, SBMEX consistently outperforms all traditional competitors across all threshold levels, achieving an F<sub>1</sub>-score of 0.68 at (16 px, π/6 rad). Under the type-aware regime, SBMEX remains superior or competitive, matched only by MINDTCT with enhanced input. Notably, at the most stringent thresholds (8 px, π/10 rad), SBMEX (0.63 type-agnostic) substantially narrows the performance gap with commercial COTS systems such as VeriFinger (0.64 on Raw input). On the challenging

Latent subset, performance drops markedly across all traditional extractors due to severe noise. Nevertheless, SBMEX and its ridge-only variant SBMEX-R demonstrate superior robustness, outperforming all traditional open-source baselines by a significant margin in both type-agnostic and type-aware regimes. Across both datasets, SBMEX-R exhibits only a slight performance decay compared to the full dual-skeleton SBMEX, establishing SBMEX-R as a highly efective option when processing resources are constrained.

Figure 3 depicts the Precision-Recall (PR) curves across two representative threshold levels. The PR trajectories confirm the findings from Table 3: the curve for SBMEX (and SBMEX-R running closely parallel just below it) consistently dominates those of traditional baselines across almost the entire operational range. Minor exceptions occur on Exemplars in the type-aware regime at the (16 px, π/6 rad) threshold level, where MINDTCT (Enh) and FingerJet (Raw) gain a localized advantage at low recall operating points.

Figure 4 illustrates the trade-of between extraction accuracy (F<sub>1</sub>-score) and execution latency measured at the (16 px, π/6 rad) threshold level under the type-agnostic regime. Benchmark timings were recorded on an Intel® Xeon® Silver 4112 CPU @ 2.60 GHz, with the exception of LEADER (executed on GPU). For SBMEX, SBMEX-R, and CN Baseline, reported latencies explicitly encompass the full end-to-end pipeline including pyfing preprocessing, whereas other baselines are measured on original raw inputs. SBMEX achieves an average total execution time of 757 ms on Exemplars and 403 ms on Latents. Although legacy C/C++ compiled packages such as MINDTCT (183 ms / 109 ms) and FingerJet (37 ms / 21 ms) exhibit lower processing latency on raw inputs, this speed comes at the cost of substantially lower extraction accuracy, particularly on noisy Latents where SBMEX provides a marked F<sub>1</sub>-score improvement (0.39 vs. 0.30 and 0.22). Conversely, SBMEX is considerably faster than the commercial COTS engine VeriFinger (1282 ms / 1310 ms) and drastically outperforms CNBaseline (2358 ms / 1805 ms), the only other Python-native baseline. Notably, when isolating pure minutiae detection by subtracting the pyfing pre-processing latency (645 ms for Exemplars and 322 ms for Latents), the core detection time of SBMEX drops to just 112 ms (81 ms on Latents), while the lightweight SBMEX-R reaches 42 ms (30 ms on Latents). In comparison, CNBaseline requires 1713 ms and 1483 ms for core detection alone, underscoring the superior eficiency-accuracy balance achieved by the LUT-based architecture of SB-MEX.

## 5 Conclusion

This paper presented SBMEX, a deterministic, skeletonbased minutiae detection method designed to provide a modern, transparent baseline for standard fingerprint pipelines. By combining LUT-accelerated skeleton processing, dual-skeleton fusion, and continuous quality decay, SBMEX eliminates runtime neighborhood scans while maintaining high accuracy and algorithmic explainability. Experimental evaluations on the NIST SD302 benchmark demonstrate that SBMEX achieves minutiae detection performance fully comparable to legacy traditional baselines, while delivering a drastic reduction in core processing time relative to a classical CN Python implementation.

(8 px, �∕10 rad), Type-agnostic  
Table 3: Comparative performance $\left( F _ { \mathrm { 1 } } { \mathrm { - s c o r e } } \right)$ on the two datasets across the three threshold levels under type-agnostic and type-aware regimes. Bold indicates the best performance among competing methods, while underline denotes the second best.
<table><tr><td rowspan="3"></td><td rowspan="3"></td><td colspan="6">NIST SD302 Exemplar</td><td colspan="6">NIST SD302 Latent</td></tr><tr><td colspan="3">Type-agnostic</td><td colspan="3">Type-aware</td><td colspan="3">Type-agnostic</td><td colspan="3">Type-aware</td></tr><tr><td>16 px π/6rad</td><td>12px π/8rad</td><td>8px π/10 rad</td><td>16 px π/6rad</td><td>12 px π/8rad</td><td>8px π/10 rad</td><td>16 px π/6 rad</td><td>12px π/8rad</td><td>8px π/10 rad</td><td>16 px π/6 rad</td><td>12 px π/8 rad</td><td>8px π/10 rad</td></tr><tr><td>SBMEX</td><td>Enh</td><td>0.68</td><td>0.66</td><td>0.63</td><td>0.50</td><td>0.49</td><td>0.47</td><td>0.39</td><td>0.36</td><td>0.32</td><td>0.25</td><td>0.24</td><td>0.21</td></tr><tr><td>SBMEX-R</td><td>Enh</td><td>0.67</td><td>0.65</td><td>0.61</td><td>0.49</td><td>0.48</td><td>0.46</td><td>0.36</td><td>0.34</td><td>0.30</td><td>0.24</td><td>0.22</td><td>0.20</td></tr><tr><td>CNBaseline</td><td>Enh</td><td>0.61</td><td>0.58</td><td>0.52</td><td>0.44</td><td>0.42</td><td>0.39</td><td>0.26</td><td>0.22</td><td>0.18</td><td>0.16</td><td>0.14</td><td>0.11</td></tr><tr><td>MINDTCT</td><td>Raw</td><td>0.62</td><td>0.59</td><td>0.53</td><td>0.48</td><td>0.46</td><td>0.43</td><td>0.30</td><td>0.28</td><td>0.25</td><td>0.22</td><td>0.20</td><td>0.18</td></tr><tr><td>MINDTCT</td><td>Enh</td><td>0.65</td><td>0.62</td><td>0.57</td><td>0.51</td><td>0.49</td><td>0.45</td><td>0.31</td><td>0.28</td><td>0.22</td><td>0.22</td><td>0.19</td><td>0.15</td></tr><tr><td>FingerJet</td><td>Raw</td><td>0.61</td><td>0.59</td><td>0.54</td><td>0.47</td><td>0.45</td><td>0.42</td><td>0.22</td><td>0.20</td><td>0.18</td><td>0.16</td><td>0.15</td><td>0.13</td></tr><tr><td>FingerJet</td><td>Enh</td><td>0.60</td><td>0.56</td><td>0.50</td><td>0.43</td><td>0.40</td><td>0.36</td><td>0.37</td><td>0.33</td><td>0.26</td><td>0.23</td><td>0.21</td><td>0.16</td></tr><tr><td>Commercial and Deep</td><td></td><td>Learning</td><td></td><td>SOTA Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VeriFinger</td><td>Raw</td><td>0.72</td><td>0.69</td><td>0.64</td><td>0.54</td><td>0.52</td><td>0.48</td><td>0.60</td><td>0.57</td><td>0.52</td><td>0.41</td><td>0.39</td><td>0.36</td></tr><tr><td>VeriFinger</td><td>Enh</td><td>0.64</td><td>0.61</td><td>0.55</td><td>0.47</td><td>0.45</td><td>0.41</td><td>0.39</td><td>0.35</td><td>0.30</td><td>0.25</td><td>0.23</td><td>0.19</td></tr><tr><td>LEADER</td><td>Raw</td><td>0.81</td><td>0.80</td><td>0.78</td><td>0.71</td><td>0.70</td><td>0.69</td><td>0.69</td><td>0.67</td><td>0.64</td><td>0.53</td><td>0.52</td><td>0.50</td></tr><tr><td>LEADER</td><td>Enh</td><td>0.74</td><td>0.73</td><td>0.70</td><td>0.59</td><td>0.58</td><td>0.56</td><td>0.48</td><td>0.46</td><td>0.42</td><td>0.34</td><td>0.33</td><td>0.30</td></tr></table>

(16 px, �∕6 rad), Type-aware  
(8 px, �∕10 rad), Type-aware  
![](images/a29b57dd244fc295e76e10472db2e6379ce1f04e399b445b50aec61a132441c8.jpg)  
Figure 3: PR curves on the two datasets across two threshold levels under type-agnostic and type-aware regimes.

![](images/b6f5c4ece09a310aec1575d772ea6ce29d049dd038eddc8c045c7e569c503aec.jpg)  
Figure 4: Accuracy vs. Latency trade-of, at (16 px, π/6 rad), type-agnostic, on Exemplars (blue) and Latents (red).

Rather than attempting to supersede end-to-end deep learning architectures or commercial matchers in unconstrained scenarios, SBMEX addresses the ongoing need for handcrafted pipelines that ofer full auditability, standard compliance, and operational independence from GPUs or extensive training data. To bridge the gap between legacy C/C++ tools and modern scientific workflows, SBMEX is fully integrated into the open-source pyfing ecosystem [24]. Future research will focus on a comprehensive empirical study comparing fully traditional, deep learning-based, and hybrid pipelines across diverse operational conditions.

## References

[1] Davide Maltoni, Dario Maio, Anil K. Jain, and Jianjiang Feng. Handbook of Fingerprint Recognition. Springer Nature Switzerland AG, 2022. doi: 10.1007/ 978-3-030-83624-5.

[2] Rafaele Cappelli and Matteo Ferrara. Leader: Lightweight end-to-end attention-gated dual autoencoder for robust minutiae extraction, 2026. URL https://arxiv.org/abs/ 2602.15493.

[3] ISO/IEC. ISO/IEC 19794-2:2011 Information technology – Biometric data interchange formats – Part 2: Finger minutiae data, 2011.

[4] Craig I Watson, Michael D Garris, Elham Tabassi, Charles L Wilson, R Michael McCabe, Stanley Janet, and Kenneth Ko. User’s guide to nist biometric image software (nbis). techreport 7392, National Institute of Standards and Technology, 2007.

[5] DigitalPersona. Fingerprint feature extractor, open source edition. URL https://github.com/FingerJetFXOSE/ FingerJetFXOSE.

[6] Tim Oblak, Rudolf Haraksim, Peter Peer, and Laurent Beslay. Fingermark quality assessment framework with classic and deep learning ensemble models. Knowledge-Based Systems, 250:109148, 2022. ISSN 0950-7051. doi: https://doi.org/10.1016/j.knosys.2022. 109148. URL https://www.sciencedirect.com/science/ article/pii/S0950705122005718.

[7] Barbara de O. Koop, Jo˜ao H. P. Machado, Luiz F. P. Southier, Jeferson T. Oliva, Marcelo Teixeira, Marcelo Filipak, Luiz A. Zanlorensi, Myriam R. B. S. Delgado, and Dalcimar Casanova. Newborn fingerprint recognition: Toward reliable super-resolution equivalences. IEEE Transactions on Technology and Society, pages 1–15, 2026. doi: 10.1109/TTS.2026.3721786.

[8] Poornima E. Gundgurti and Padmavati E. Gundgurti. Shi-tomasi corner detection and random sample consensus for latent fingerprint analysis. In 2026 3rd International Conference on Integrated Intelligence and Communication Systems (ICIICS), pages 1–6, 2026. doi: 10.1109/ICIICS67880.2026.11483566.

[9] Ziyi Wang. Quality-aware minutiae matching for forensic fingerprints with explicit no-score decisions. IEEE Access, 14:70530–70550, 2026. doi: 10.1109/ACCESS.2026. 3691260.

[10] Kamel Houari and Salim Chikhi. Msf-fingerprint: Multilevel stability framework for partial fingerprint recognition. In 2026 12th International Conference on Control, Decision and Information Technologies (CoDIT), pages 855–860, 2026. doi: 10.1109/CoDIT70676.2026.11630986.

[11] D. Maio and D. Maltoni. Direct gray-scale minutiae detection in fingerprints. IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(1):27–40, 1997. doi: 10.1109/34.566808.

[12] Hartwig Fronthaler, Klaus Kollreider, and Josef Bigun. Local features for enhancement and minutiae extraction in fingerprints. IEEE Transactions on Image Processing, 17(3):354–363, 2008. doi: 10.1109/TIP.2007.916155.

[13] Anush Sankaran, Prateekshit Pandey, Mayank Vatsa, and Richa Singh. On latent fingerprint minutiae extraction using stacked denoising sparse autoencoders. In IEEE International Joint Conference on Biometrics, pages 1–7, 2014. doi: 10.1109/BTAS.2014.6996300.

[14] Luke Nicholas Darlow and Benjamin Rosman. Fingerprint minutiae extraction using deep learning. In 2017 IEEE International Joint Conference on Biometrics (IJCB), pages 22–30, 2017. doi: 10.1109/BTAS.2017.8272678.

[15] Yao Tang, Fei Gao, and Jufu Feng. Latent fingerprint minutia extraction using fully convolutional network. In 2017 IEEE International Joint Conference on Biometrics (IJCB), pages 117–123, 2017. doi: 10.1109/BTAS.2017. 8272689.

[16] Dinh-Luan Nguyen, Kai Cao, and Anil K. Jain. Robust minutiae extractor: Integrating deep networks and fin-

gerprint domain knowledge. In 2018 International Conference on Biometrics (ICB), pages 9–16, 2018. doi: 10.1109/ICB2018.2018.00013.

[17] Baicun Zhou, Congying Han, Yonghong Liu, Tiande Guo, and Jin Qin. Fast minutiae extractor using neural network. Pattern Recognition, 103:107273, 2020. ISSN 0031-3203. doi: https://doi.org/10.1016/j.patcog.2020. 107273. URL https://www.sciencedirect.com/science/ article/pii/S0031320320300789.

[18] Yao Tang, Fei Gao, Jufu Feng, and Yuhang Liu. Fingernet: An unified deep network for fingerprint minutiae extraction. In 2017 IEEE International Joint Conference on Biometrics (IJCB), page 108–116, Denver, CO, USA, 2017. IEEE Press. doi: 10.1109/BTAS.2017.8272688. URL https://doi.org/10.1109/BTAS.2017.8272688.

[19] Van Huan Nguyen, Jinsong Liu, Thi Hai Binh Nguyen, and Hakil Kim. Universal fingerprint minutiae extractor using convolutional neural networks. IET Biometrics, 9(2):47–57, 2020. doi: https://doi.org/10.1049/iet-bmt.2019.0017.

[20] Yulin Feng and Ajay Kumar. Detecting locally, patching globally: An end-to-end framework for high speed and accurate detection of fingerprint minutiae. IEEE Transactions on Information Forensics and Security, 18:1720–1733, 2023. doi: 10.1109/TIFS.2023.3251862.

[21] Zicheng Guo and Richard W. Hall. Parallel thinning with two-subiteration algorithms. Communications of the ACM, 32(3):359–373, 1989. doi: 10.1145/62065.62074.

[22] Gregory Fiumara, Patricia Flanagan, John Grantham, Kenneth Ko, Karen Marshall, Matthew Schwarz, Elham Tabassi, Bryan Woodgate, and Christopher Boehnen. Nist special database 302: Nail to nail fingerprint challenge, 2019.

[23] Gregory Fiumara, Matthew Schwarz, Jessica Heising, Jennifer Peterson, Kenneth Ko, Patricia Flanagan, and Karen Marshall. Nist special database 302: Annotated latent distal phalanxes, 2026.

[24] Rafaele Cappelli. pyfing: A python library for fingerprint processing, 2024. URL https://github.com/ raffaele-cappelli/pyfing.

[25] Rafaele Cappelli. Unveiling the power of simplicity: Two remarkably efective methods for fingerprint segmentation. IEEE Access, 11:144530–144544, 2023. ISSN 2169-3536. doi: 10.1109/access.2023.3345644.

[26] Rafaele Cappelli. Exploring the power of simplicity: A new state-of-the-art in fingerprint orientation field estimation. IEEE Access, 12:55998–56018, 2024. doi: 10.1109/ACCESS.2024.3389701.

[27] Rafaele Cappelli. No feature left behind: Filling the gap in fingerprint frequency estimation. IEEE Access, 12:153605– 153617, 2024. doi: 10.1109/ACCESS.2024.3481507. URL https://doi.org/10.1109/ACCESS.2024.3481507.

[28] Rafaele Cappelli. Unleashing the power of simplicity: A minimalist strategy for state-of-the-art fingerprint enhancement, 2026. URL https://arxiv.org/abs/2603.19004.

[29] D. Maio, D. Maltoni, R. Cappelli, J.L. Wayman, and A.K. Jain. Fvc2000: fingerprint verification competition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(3):402–412, 2002. doi: 10.1109/34.990140.

[30] David F. Crouse. On implementing 2d rectangular assignment algorithms. IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679–1696, 2016. doi: 10.1109/TAES.2016.140952.