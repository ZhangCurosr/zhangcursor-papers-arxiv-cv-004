# WIDE: Wildcard Inference with Dynamic Expansion for Cross-Modal Generative Retrieval

Teng Guo   
College of Computer Science and   
Technology   
Jilin University   
Changchun, China   
guoteng24@mails.jlu.edu.cn   
Keying Zhou   
College of Computer Science and   
Technology   
Jilin University   
Changchun, China   
zhouky25@mails.jlu.edu.cn   
Xin Wang<sup>✉∗†</sup>   
College of Computer Science and   
Technology   
Jilin University   
Changchun, China   
w\_x@jlu.edu.cn   
Jifeng Shen   
School of Electrical and Information   
Engineering   
Jiangsu University   
Zhenjiang, China   
shenjifeng@ujs.edu.cn   
Jiayou Xu   
College of Computer Science and   
Technology   
Jilin University   
Changchun, China   
xjy25@mails.jlu.edu.cn   
Haoxin Ruan   
College of Computer Science and   
Technology   
Jilin University   
Changchun, China   
ruanhx24@mails.jlu.edu.cn

## Abstract

Generative retrieval has demonstrated significant success by uni fying representation learning and search into a single sequenceto-sequence generation task. However, extending this paradigm to cross-modal retrieval reveals a critical challenge arising from the inherent information asymmetry across diferent modalities, such as the gap between concise text queries and dense visual candidates. This structural mismatch causes the autoregressive decoder to sufer from forced hallucination when generating identifiers via standard trie-constrained beam search, where the model is severely penalized for failing to guess fine-grained details absent from the query, allowing irrelevant candidates to hijack top rankings. To address this issue, we propose Wildcard Inference with Dynamic Expansion (WIDE). WIDE employs Adaptive Entropy Thresholding (AET) to calibrate layer-specific uncertainty boundaries ofline. Dur ing the decoding generation phase, Asymmetry-aware Wildcard Decoding (AWD) detects semantic blind spots and emits wildcards instead of forced deterministic identifiers, dynamically expanding the search space without incurring log-probability penalties. Finally, Blind-Spot Re-ranking (BSR) evaluates the expanded candidate pool using a hybrid scoring mechanism that combines discrete generation confidence with continuous semantic similarity. Extensive experiments on the M-BEIR benchmark demonstrate that WIDE outperforms state-of-the-art generative retrieval methods, efectively suppressing forced hallucination while maintaining compact index structures.

<sup>∗</sup>Also with Key Laboratory of Symbolic Computation and Knowledge Engineering of Ministry of Education, Jilin University.   
<sup>†</sup>The corresponding author.

![](images/2e9cf2deaa754246a4d99703513d03a01b8b68835cf406dfe09a93d224cdc685.jpg)  
Figure 1: Information asymmetry and methods comparison. Standard beam search forces identifier generation when encountering semantic blind spots. The proposed method detects predictive uncertainty and emits a wildcard identifier.

## CCS Concepts

• Information systems → Multimedia and multimodal retrieval; Image search; Retrieval models and ranking.

Generative Retrieval, Cross-modal Information Asymmetry, Wildcard Decoding, Candidate Re-ranking

## Keywords

## ACM Reference Format:

Teng Guo, Xin Wang,Jiayou Xu, Keying Zhou,Jifeng Shen, and Haoxin Ruan. 2026. WIDE: Wildcard Inference with Dynamic Expansion for Cross-Modal Generative Retrieval. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3767308.3835042

## 1 Introduction

Generative retrieval introduces a paradigm shift in the architecture of information retrieval systems [33]. Instead of maintaining a fixed dense embedding and performing nearest-neighbor search during inference, this approach formulates retrieval as a sequence-tosequence generation task [2]. By training an autoregressive model to map a query directly to a target identifier, it unifies representation learning and the retrieval process into a single step. This framework has demonstrated significant empirical success in text retrieval settings, where the query and the target share the same modality [4, 33].

However, extending this generative paradigm to cross-modal retrieval reveals a critical limitation rooted in the inherent information asymmetry between queries and candidates [21]. Current methods process continuous visual features by recursively compressing image embeddings into hierarchical discrete codebook sequences through residual quantization [15, 40]. As illustrated in Fig. 1(a), a text query typically outlines only a sparse subset of semantic attributes. In contrast, the candidate image contains an exhaustive record of visual details encoded into these discrete identifiers. This discrepancy reflects the fundamental diference between linguistic abstraction and visual density. Consequently, this structural mismatch disrupts the standard autoregressive decoding process. As illustrated in Fig. 1(b), standard trie-constrained beam search [2] forces the decoder to emit a deterministic identifier code at every step. When certain visual details encoded by codebook layers are not specified in the text query, the model encounters se mantic blind spots and defaults to statistical co-occurrence patterns learned during training. We define this phenomenon as forced hallucination. It constitutes a systematic error driven by cross-modal asymmetry rather than a failure of the underlying representation.

Forced hallucination creates an unfair scoring bias that unnecessarily penalizes correct retrieval paths. When encountering a semantic blind spot, the predictive distribution of the decoder becomes less concentrated, spreading probability mass across multiple identifier candidates [14]. Forcing a deterministic identifier selection under this high uncertainty incurs a severe penalty in the accumulated decoding log-probability. Consequently, a candidate that perfectly matches the query in early layers sufers continuous log-probability drops in uncertain layers simply for failing to guess unmentioned visual details. This flaw allows irrelevant candidates to hijack the top rankings, not through semantic matching, but because their unconstrained details coincidentally align with frequent training priors, while trie constraints only guarantee valid identifier generation rather than relevance.

We argue that this failure mode requires a fundamentally different solution. As illustrated in Fig. 1(b), the objective is not to force the decoder to guess uninformed layers more accurately, as no amount of training can recover information absent from the user query. Instead, the goal is to prevent the decoder from being penalized for inherent information asymmetry. To this end, we propose Wildcard Inference with Dynamic Expansion for Cross-Modal Generative Retrieval (WIDE). This framework addresses forced hallucination through a cohesive pipeline comprising ofline calibration, online intervention, and adaptive candidate evaluation.

The framework operates by first deploying Adaptive Entropy Thresholding (AET) as an ofline calibrator to quantify the baseline uncertainty of the model. By profiling the training data, AET establishes layer-specific entropy thresholds that explicitly map the boundaries where semantic blind spots naturally occur. These data-driven boundaries provide a clear reference for the subsequent generation phase. Guided by this reference, Asymmetry-aware Wildcard Decoding (AWD) monitors the real-time autoregressive process to intercept forced hallucinations. Rather than emitting a deterministic token that risks a severe log-probability penalty, AWD generates a wildcard. This specific intervention temporarily suspends the rigid pruning of the prefix trie, dynamically expanding the search space into a semantically coherent candidate set that preserves the correct retrieval path. Finally, to resolve these expanded clusters and prevent irrelevant candidates from hijacking the top results, Blind-Spot Re-ranking (BSR) evaluates the retrieved candidates. BSR ignores unreliable discrete signals from wildcarded positions and combines reliable non-wildcard generation confidence with continuous embedding similarity for final ranking.

Extensive experiments on the M-BEIR benchmark demonstrate that WIDE achieves strong performance compared with existing generative retrieval methods. Further analyses show that wildcard inference with dynamic candidate expansion efectively alleviates forced hallucination while keeping the expanded search space highly constrained.

The main contributions of this paper are summarized as follows:

• We identify and formulate forced hallucination as a structural failure mode in trie-constrained generative retrieval caused by cross-modal information asymmetry.

• We introduce WIDE, a novel framework that addresses this limitation through adaptive uncertainty handling, wildcard based decoding, and hybrid candidate re-ranking.

• We conduct extensive experiments across diverse cross-modal retrieval benchmarks, showing that WIDE achieves stateof-the-art generative retrieval performance and efectively addresses cross-modal information asymmetry.

## 2 Related Work

## 2.1 Generative Retrieval

Generative Retrieval (GR) [33, 34] has emerged as an alternative retrieval paradigm that replaces the conventional match-and-rank pipeline with direct identifier generation.

Building upon success in text retrieval, GR has been extended to cross-modal scenarios [6, 16, 19, 20, 28]. Unlike single-modal retrieval [17], cross-modal retrieval must address the representational gap between heterogeneous modalities, which complicates the design of identifiers. GENIUS [13] proposes a generalized multimodal generative retrieval framework. The Modality-Decoupled Semantic Quantization of this framework learns modality-invariant semantic anchors to improve identifier generation across diferent modalities. While this line of work emphasizes the semantic quality and modality alignment of identifiers, existing methods largely overlook the inherent information asymmetry between queries and visual targets during the decoding process.

Furthermore, hybrid designs incorporating late fusion [31] have recently been introduced to generative retrieval. Representative approaches [38, 39] generate initial candidate identifiers and subsequently refine them using additional similarity signals. These designs are highly practical, as they preserve the eficiency of discrete generative retrieval while leveraging complementary continuous signals to correct the outputs of discrete generation. However, current reranking strategies generally apply flat embedding similarity metrics, failing to account for the specific semantic blind spots caused by decoding uncertainty.

## 2.2 Constrained Decoding and Uncertainty Estimation

Trie-constrained beam search [1, 2, 10] serves as the standard decoding strategy in GR. This mechanism organizes all valid target sequences into a prefix tree, where each path from the root to a leaf corresponds to a registered identifier. While this hard constraint prevents the generation of out-of-vocabulary identifiers, it forces the decoder to make a specific identifier decision even when the query provides insuficient evidence.

Entropy-based uncertainty estimation has been extensively explored in generative modeling [37]. Prior research [27] demonstrates that predictive entropy serves as an efective signal for reliability estimation at both the token and sequence levels. Recent studies [14, 41] propose semantic entropy to capture uncertainty beyond variations in surface forms, while token-level uncertainty enables fine-grained confidence assessment and factuality evaluation.

These findings indicate that uncertainty signals derived from the output distribution of the model provide critical guidance for identifying unreliable generation. Our work leverages this metric to dynamically calibrate decoding interventions, transforming unguided forced guesses into structured candidate expansion.

## 3 Preliminary

## 3.1 Problem Formulation

We consider cross-modal generative retrieval over a heterogeneous candidate database $C = \mathsf { \bar { \{ } c ^ { ( 1 ) } , . . . , c ^ { ( N ) } \} }$ , where each candidate � represents a multimodal entity. A composite query q comprises content signals, including text, images, and interleaved modalities, alongside task-specific instructions that guide the desired target retrieval [35].

Generative retrieval formulates this task as a constrained sequence generation process [33]. Each candidate $c \in C$ is uniquely mapped to a discrete identifier sequence $T _ { c } = ( c _ { 1 } , \ldots , c _ { M } ) $ . The retrieval objective is to maximize the conditional generation probability of the identifier sequence given the query context according to the following formulation.

$$
\hat { c } = \underset { c \in C } { \arg \operatorname* { m a x } } \prod _ { k = 1 } ^ { M } p _ { \theta } ( c _ { k } \mid \mathbf { q } , c _ { < k } ) ,\tag{1}
$$

where $c _ { < k }$ denotes the generated sequence prefix and $\theta$ represents the model parameters. To ensure validity, the generated sequence must traverse a pre-constructed prefix trie containing all registered database identifiers [2]. This structural constraint enables the system to bypass exhaustive similarity scoring across the entire database.

The efectiveness of this paradigm relies on two interconnected stages. It first requires learning compact and discriminative identifiers during quantization and then accurately generating these identifiers during decoding, especially when queries provide incomplete semantic coverage. Our work focuses on addressing the decoding-stage challenge.

## 3.2 Residual Quantization

To map multimodal representations into the discrete space, Residual Quantization (RQ) compresses a dense candidate embedding ${ \bf v } _ { c } \in \mathbb { R } ^ { D }$ through an iterative residual approximation [15, 40]. Given a sequence of � codebooks $\{ \mathcal { B } _ { 1 } , . . . , \mathcal { B } _ { M } \}$ , where each $\mathcal { B } _ { k } = \{ \mathbf { b } _ { k , v }$ ∈ $\mathbb { R } ^ { D } \} _ { v = 1 } ^ { \hat { V } _ { k } }$ contains $V _ { k }$ code vectors, the quantization process initializes the residual vector as $\boldsymbol { \Delta } _ { 1 } = \mathbf { v } _ { c }$ . In our implementation, we use $M = 9$ quantization levels, where the first level encodes modality information. At each quantization level $k \in \{ 1 , \ldots , M \}$ , RQ identifies the nearest code vector to the current residual state:

$$
c _ { k } = \underset { v \in [ V _ { k } ] } { \arg \operatorname* { m i n } } \left\| \Delta _ { k } - \mathbf { b } _ { k , v } \right\| _ { 2 } ^ { 2 } , \qquad \Delta _ { k + 1 } = \Delta _ { k } - \mathrm { s g } \big ( \mathbf { b } _ { k , c _ { k } } \big ) ,\tag{2}
$$

where $[ V _ { k } ] = \{ 1 , \dots , V _ { k } \}$ and $s \mathrm { g } ( \cdot )$ denotes the stop-gradient operator. The resulting quantized approximation is $\begin{array} { r } { \hat { \mathbf { v } } _ { c } = \sum _ { k = 1 } ^ { M } \mathbf { b } _ { k , c _ { k } } } \end{array}$

The codebooks are initialized using �-means and updated through exponential moving averages. The RQ objective encourages the residual representation at each level to remain close to its selected code vector:

$$
\mathcal { L } _ { \mathrm { R Q } } = \frac { 1 } { M D } \sum _ { k = 1 } ^ { M } \left\| \Delta _ { k } - \mathrm { s g } \big ( \mathbf { b } _ { k , c _ { k } } \big ) \right\| _ { 2 } ^ { 2 } .\tag{3}
$$

By sequentially minimizing the residual error, Eq. (2) provides an ordered refinement of the quantized representation. The shallow levels capture dominant components of the candidate representation, while deeper levels encode the remaining residual information, which often corresponds to finer-grained visual details.

This hierarchical residual structure introduces a challenge during cross-modal retrieval. Textual queries are inherently selective and sparse. They usually specify particular objects or attributes while omitting visual details that are not explicitly mentioned. Consequently, a query may provide strong semantic constraints for some identifier levels while leaving other levels insuficiently grounded. This varying degree of query coverage across quantization levels can disrupt the subsequent identifier generation process.

## 3.3 Trie-Constrained Decoding and Forced Hallucination

To perform retrieval, an autoregressive decoder generates the target sequence $T _ { c }$ conditioned on the query. The model is optimized via token-level cross-entropy using the ground-truth sequence $\mathbf { c } ^ { * }$

$$
\mathcal { L } _ { \mathrm { G R } } = - \sum _ { k = 1 } ^ { M } \log p _ { \theta } \bigl ( c _ { k } ^ { * } \mid \mathbf { q } , c _ { < k } ^ { * } \bigr ) .\tag{4}
$$

During inference, beam search is employed to explore the prefix trie. To guarantee that the output corresponds to an existing candidate, the token space at step $k$ is restricted to $\mathcal { T } _ { k } ,$ which represents the admissible set of valid child nodes extending from the current prefix $c _ { < k } \ [ 2 ]$ . The beam search algorithm evaluates hypotheses based on their cumulative log-probability:

$$
\operatorname { S c o r e } ( T _ { c } ) = \sum _ { k = 1 } ^ { M } \log p _ { \theta } ( c _ { k } \mid \mathbf { q } , c _ { < k } ) .\tag{5}
$$

![](images/5c1d086e92b1b24cbb11aa2aeaa92c5d8aa265c6ebcb5ebb310da3a94f38ccf5.jpg)  
Figure 2: Overview of the proposed WIDE framework. Within the generative inference pipeline (dashed box), AET establishes layer-wise entropy thresholds to guide AWD in dynamically emitting wildcards at semantic blind spots. BSR subsequently evaluates the wildcard-expanded candidate sets to yield the retrieval results.

While the constraint space $\mathcal { T } _ { k }$ prevents the generation of outof-vocabulary identifiers, it requires the decoder to commit to a specific identifier choice at every step. Under cross-modal information asymmetry, when the decoder reaches a quantization level where the query provides insuficient semantic evidence, the predictive distribution $p _ { \theta } ( t \mid { \bf q } , c _ { < k } )$ may become less concentrated and dominated by prefix-dependent statistical priors rather than query-specific evidence [14].

Despite this uncertainty, standard trie-constrained decoding forces the model to select a specific token. This forced selection can incur a log-probability penalty [32], approaching − log |T<sub>�</sub> | when the distribution becomes nearly uniform. We define this behavior as forced hallucination. A hypothesis beam that correctly captures the query-informed attributes in early layers will accumulate massive numerical penalties in the unconstrained layers. Consequently, the ground-truth candidate is frequently displaced by irrelevant distractors whose deeper identifiers happen to statistically align with the training priors of the decoder [11]. Addressing this structural failure necessitates a dynamic intervention mechanism that quantifies query coverage boundaries, permits explicit expression of uncertainty, and adaptively evaluates the resulting candidate sets. We detail our proposed framework to achieve these objectives in Section 4.

## 4 Methodology

## 4.1 Overview

Generative retrieval recasts the traditional dense matching paradigm into an autoregressive sequence generation task. As illustrated in Fig. 2, our pipeline processes multimodal inputs through specific encoders. On the query side, the query text, task instructions and the query image are encoded and subsequently integrated via a fusion module (FUSE & MLP) to form a unified fusion embedding. On the candidate side, database images and texts are mapped into a shared semantic space, where RQ compresses them into hierarchical discrete identifiers. During training, an autoregressive decoder learns to generate the target discrete identifier conditioned on the unified fusion embeddings.

To resolve forced hallucination highlighted above, we propose the WIDE framework. As depicted in the dashed box ofFig. 2, WIDE integrates into the generative retrieval inference pipeline through three sequential modules. First, AET (Sec. 4.2) computes layer-wise entropy thresholds $\left( L _ { 1 } \ldots L _ { m } \right)$ from the training data to serve as an ofline reference for predictive uncertainty. Next, AWD (Sec. 4.3) utilizes these entropy thresholds to monitor the autoregressive decoding process. When predictive uncertainty surpasses the entropy threshold, it bypasses forced generation by inserting a wildcard into the search path, seamlessly producing discrete identifier sequences without terminating the generation. Finally, BSR (Sec. 4.4) evaluates the candidate pool expanded by AWD using a hybrid scoring mechanism to yield the final retrieval results.

## 4.2 Adaptive Entropy Thresholding

The central premise of AWD is that predictive uncertainty of the output distribution at level � provides a useful signal for identifying potential query semantic blind spots caused by cross-modal information asymmetry. However, raw entropy values are not directly comparable across levels because the trie-constrained admissible set size |T<sub>�</sub> | varies across RQ levels and prefixes, which afects the entropy scale. Therefore, evaluating uncertainty requires a levelspecific, data-driven reference. AET establishes this reference ofline to calibrate the online decoding process.

Given a generative retrieval model $\hbar \theta$ and the training data $\{ ( \mathbf { q } , \mathbf { c } ^ { * } ) \}$ , as shown in the left panel of Fig. 3, we define the dynamic entropy threshold $\tau _ { k }$ as the expected predictive entropy at each level � under teacher forcing. By conditioning the decoder on the true prefix $c _ { < k } ^ { * }$ , we estimate the predictive uncertainty at each level under a consistent generation context:

$$
\tau _ { k } = \bar { H } _ { k } = \mathbb { E } _ { ( \mathbf { q } , \mathbf { c } ^ { * } ) } \left[ - \sum _ { t \in \mathcal { T } _ { k } } \mathfrak { p } _ { \theta } \big ( t \mid \mathbf { q } , c _ { < k } ^ { * } \big ) \log \mathfrak { p } _ { \theta } \big ( t \mid \mathbf { q } , c _ { < k } ^ { * } \big ) \right] .\tag{6}
$$

This expected entropy $\bar { H } _ { k }$ captures the typical uncertainty associated with each RQ level. Shallow levels often exhibit lower uncertainty because they encode dominant residual components, whereas deeper levels may exhibit higher uncertainty when the remaining visual details are insuficiently specified by the query.

![](images/59c5d2e5fb4d134a5336cd17aa0b1863e50296b8437101a78d1b53d0976e87ea.jpg)  
Figure 3: The architecture of AET, AWD and BSR. (1) AET quantifies predictive uncertainty by calculating the entropy of per-level identifier generation to establish thresholds. (2) AWD circumvents forced generation by inserting a wildcard (∗) with no penalty, maintaining the continuous expansion of the sequence. (3) BSR re-ranks the wildcard-inclusive can didate sequences by fusing discrete structural scores with embedding similarities.

Using ofline expected entropy $\bar { H } _ { k }$ as a level-specific reference for real-time entropy $H _ { k }$ allows AWD to identify unusually uncertain decoding states [9]. During inference, if a novel query yields a real-time entropy $H _ { k } > \tau _ { k }$ , it indicates that the decoder is more uncertain than the historical baseline for that specific semantic level. This parameter-free comparison enables adaptive wildcard triggering while accounting for the intrinsic uncertainty diferences across RQ levels.

## 4.3 Asymmetry-aware Wildcard Decoding

Standard trie-constrained beam search maintains � identifier prefixes and selects the top-� valid extensions at each level according to cumulative log-probability. When the decoder encounters a semantic blind spot, it is still forced to select a specific identifier code from the admissible set $\mathcal { T } _ { k } .$ . Under high uncertainty, this forced decision introduces a large negative log-probability contribution that can suppress the entire retrieval path. AWD avoids this unnec essary penalty by replacing deterministic identifier generation at uncertain levels with wildcards.

At each decoding level $k ,$ AWD computes the entropy of the decoder output distribution over $\mathcal { T } _ { k }$

$$
H _ { k } = - \sum _ { t \in \mathcal { T } _ { k } } \dot { p } _ { \theta } ( t \mid \mathbf { q } , c _ { < k } ) \log \dot { p } _ { \theta } ( t \mid \mathbf { q } , c _ { < k } ) ,\tag{7}
$$

As shown in the middle panel of Fig. 3, AWD applies a binary decision for each active beam �. When $H _ { k } \leq \tau _ { k } ,$ , the query provides suficient constraint, and the log-probability of the selected token is added to the beam score. When $H _ { k } > \tau _ { k } , \mathrm { A W D }$ registers level � as a blind spot, records it in the specific blind-spot set ${ \mathcal { A } } _ { b }$ of the beam, and emits a wildcard [∗]. Crucially, no log-probability penalty is added to the beam score for this level.

$$
\log P ^ { ( A W D ) } ( T _ { b } ) = \sum _ { k \notin \mathcal { A } _ { b } } \log p _ { \theta } ( c _ { k } \mid \mathbf { q } , c _ { < k } ) ,\tag{8}
$$

This scoring rule ensures that each beam is evaluated only on the levels where the identifier generation is supported by the query.

Instead of forcing an unreliable identifier code generation, the prefix trie temporarily activates all valid child branches when a wildcard is triggered. However, AWD does not terminate the autoregressive decoding at this first semantic blind spot. If the query provides explicit semantic constraints at a subsequent deeper level $( k \notin \mathcal { R } _ { b } )$ , the decoder seamlessly resumes standard identifier code generation to prune the expanded branches. Let $\hat { c } _ { k }$ denote the confident identifier code generated at non-blind-spot levels. The final candidate set retrieved by beam � is defined by dynamic set containment.

$$
S _ { b } = \left\{ { \bf c } \in C \mid c _ { k } = \hat { c } _ { k } \forall k \notin \mathcal { A } _ { b } \right\} ,\tag{9}
$$

The set $S _ { b }$ contains every database candidate that perfectly matches the confident discrete generations of the beam, while leaving the wildcard levels unconstrained. Thus, intermediate blind spots are explicitly bypassed rather than forcibly generated, allowing reliable identifier constraints from diferent levels to be jointly preserved without prematurely abandoning the search.

The dynamic expansion mechanism may introduce additional candidates. Under a uniform identifier distribution assumption, activating a wildcard after fixing the first $m - 1$ identifier levels reduces the expected candidate space by a factor of $V ^ { m - 1 }$ , yielding an expected candidate size of $| S _ { b } | \approx | C | / V ^ { m - 1 }$ . This suggests that wildcard expansion remains limited when reliable identifier prefixes are preserved. Since real identifier distributions may be non-uniform, we further analyze the actual expansion behavior empirically.

At the end of the AWD process, all candidates satisfying the wildcard constraints of the retained beams form a global candidate pool $S _ { g l o b a l } .$ , which is then forwarded to BSR.

## 4.4 Blind-Spot Re-ranking

Following the AWD process, the top-� beam paths yield a global candidate pool $S _ { g l o b a l }$ . Candidates associated with the same beam share the same reliable identifier constraints at non-wildcard levels, while candidates from diferent beams correspond to diferent decoding paths. Relying solely on continuous similarity to rank $S _ { g l o b a l }$ is suboptimal because it ignores the discrete identifier constraints preserved during generation and treats wildcard-induced variations equally. To address this issue, BSR introduces a hybrid scoring function that combines reliable discrete generation confidence from constrained levels with continuous embedding similarity.

As shown in the right panel of Fig. 3, for each candidate $c ^ { ( i ) }$ ∈ $S _ { g l o b a l }$ originating from a beam with a blind-spot set $\mathcal { A } _ { i }$ , the final retrieval score �(�) is defined as a convex combination ofthe discrete and continuous scores.

$$
s ( i ) = ( 1 - \alpha _ { i } ) \cdot \frac { 1 } { M - \left| \mathcal { R } _ { i } \right| } \sum _ { k \notin \mathcal { R } _ { i } } \ p _ { \theta } \left( c _ { k } ^ { ( i ) } \mid \mathbf { q } , c _ { < k } ^ { ( i ) } \right) + \alpha _ { i } \cdot \frac { 1 + \cos \left( \mathbf { f } _ { q } , \mathbf { f } _ { i } \right) } { 2 } ,\tag{10}
$$

where $\mathbf { f } _ { q }$ and $\mathbf { f } _ { i }$ denote the query and candidate embeddings used to compute continuous semantic similarity.

The first term computes the average autoregressive probability over all levels that are not identified as blind spots $( k \notin \mathcal { R } _ { i } )$ . Because AWD skips unreliable levels while continuing generation at the remaining levels, confident identifier generations are distributed across the sequence. Therefore, averaging over the valid sequence length $\left( M - \left| \mathcal { R } _ { i } \right| \right)$ normalizes the discrete score to the [0, 1] range. This normalization neutralizes the length bias caused by varying blind-spot proportions across diferent beams. This mechanism explicitly utilizes the identifier generation confidence of the model to rank candidates based on query. The second term computes the cosine similarity between the global embeddings, shifted to the [0, 1] range to align with the scale of the discrete probabilities.

To regulate the contribution of each term without relying on empirical hyperparameters, the interpolation weight $\alpha _ { i }$ is determined by the proportion of blind spots in the source beam of the candidate.

$$
\alpha _ { i } = \frac { | \mathcal { R } _ { i } | } { M } ,\tag{11}
$$

where � is the maximum sequence length. When a query provides detailed constraints $( | \mathcal { R } _ { i } |$ is small), $\alpha _ { i }$ approaches 0, and the ranking primarily depends on the discrete generative confidence. When the query lacks specific details and triggers multiple wildcards $( | \mathcal { R } _ { i } |$ is large), �<sub>�</sub> increases. This shift assigns greater weight to the continuous semantic similarity, utilizing the global cross-modal embeddings to resolve visual ambiguities within the expanded clusters. Finally, BSR sorts $S _ { g l o b a l }$ by �(�) to output the top-� retrieval results.

## 5 Experiments

To validate the efectiveness of the proposed WIDE framework, we conduct extensive experiments focusing on its ability to resolve semantic blind spots under cross-modal asymmetry. We benchmark our method against state-of-the-art embedding-based and generative retrieval paradigms.

## 5.1 Experimental Setup

Datasets and Evaluation Metrics. We evaluate our model on the comprehensive M-BEIR benchmark [35], which aggregates 10 datasets and a pool of 5.6 million candidate items across eight distinct multimodal retrieval tasks. Specifically, the benchmark encompasses widely adopted datasets including MS-COCO [22], VisualNews [23], Fashion200K [8], NIGHTS [7], FashionIQ [36], CIRR [25], WebQA [3], OVEN [12], InfoSeek [5], and EDIS [24]. Rather than a monolithic collection, M-BEIR encompasses various complex scenarios: standard cross-modal retrieval, fine-grained attribute manipulation, image-to-image similarity reasoning. Adhering strictly to the established M-BEIR evaluation protocol, we report Recall@5 (R@5) as the primary evaluation metric across most tasks, and adopt Recall@10 (R@10) specifically for Fashion200K and FashionIQ to ensure fair comparisons.

Baseline. We acknowledge the foundational contributions of the GENIUS framework [13], as its multimodal general generative retrieval architecture serves as the direct starting point for our approach. Consequently, we establish a primary baseline that adopts a highly analogous two-stage architecture relying exclu sively on residual quantization and autoregressive decoding. Our baseline utilizes the fine-tuned variants of CLIP with score-level fusion (SF) [29], as the foundational dual-encoder to extract visual and textual representations. For the generative component, we instantiate the autoregressive decoder using a randomly initialized T5-small model [30]. Due to computational limitations, we train this baseline model using a constrained hardware configuration consisting of two NVIDIA L20 GPUs.

Training Strategy. The network is optimized using the AdamW optimizer [26] with a batch size of 256. We apply a peak learning rate of $1 \times 1 0 ^ { - 4 }$ coupled with a cosine decay schedule. The RQ module is trained for 20 epochs, and the autoregressive decoder is subsequently trained for 30 epochs.

## 5.2 Overall Retrieval Performance

Table 1 presents the comprehensive evaluation of our WIDE framework against both embedding-based and generative baselines across the M-BEIR benchmark. The results yield several key observations regarding the efectiveness of uncertainty-guided wildcard decoding and hybrid re-ranking mechanisms.

WIDE achieves state-of-the-art performance across almost all tasks within the generative retrieval track, demonstrating significant margins over the strongest model, GENIUS<sup>R</sup>. Notably, in complex reasoning scenarios that exhibit severe cross-modal asymmetry, such as fine-grained visual manipulation (CIRR) and knowledgebased visual question retrieval (WebQA $q _ { t } \to c _ { t } )$ , WIDE outperforms GENIUS by 2.0 and 1.6 percentage points, respectively. Similar substantial gains are observed in image retrieval tasks like VisualNews (+2.8 percentage points). By deploying the AWD module to dynamically emit wildcards and the BSR module to re-rank candidates afected by identifier hallucinations, WIDE successfully avoids the premature pruning of relevant retrieval paths in standard trie-constrained beam search.

Generative retrieval often underperforms embedding-based retrieval because continuous representations must be compressed into discrete identifiers, inevitably introducing quantization loss and reducing representational capacity. As shown in Table 1, WIDE reduces this performance gap. Without relying on exhaustive similarity computation over the entire database, our framework achieves comparable performance to BLIP-FF and CLIP-SF [35] on specific benchmarks, including NIGHTS and EDIS. This demonstrates that adaptive uncertainty handling can improve the efectiveness of generative retrieval while preserving the structural advantages of a unified prefix trie.

We also include U-MARVEL (built upon the Qwen2-VL-7B backbone) [18] as a strong reference for multimodal retrieval. While U-MARVEL achieves superior performance on the benchmark, it relies on a large multimodal foundation model with billions of parameters, introducing a much larger computational footprint during inference. In contrast, WIDE is built upon a lightweight T5- small decoder. The competitive performance of WIDE against such a heavyweight model highlights the efectiveness of uncertaintyaware decoding as an eficient alternative to simply scaling model size.

## 5.3 Ablation Study

To validate the individual contributions of the proposed components within the WIDE framework, we conduct a systematic ablation study. Table 2 reports the performance variations on four representative datasets covering diverse cross-modal retrieval settings, including MS-COCO, VisualNews, WebQA, and FashionIQ. The baseline follows the standard generative retrieval pipeline with trie-constrained beam search and does not apply any uncertainty intervention.

Table 1: Comprehensive Performance on the M-BEIR Benchmark. Unlike previous works, we group the diverse task formulations by their target modality to demonstrate the robust generalization of our method. Metrics include Recall@5 and Recall@10.
<table><tr><td rowspan="3">Method</td><td colspan="6">Target: Image Retrieval (ci)</td><td colspan="6">Target: Text Retrieval (ct)</td><td colspan="4">Target: Multimodal  $( c _ { i } , c _ { t } )$ </td></tr><tr><td colspan="3">Query: qt</td><td colspan="3">qi</td><td>Query: qt</td><td colspan="3">qi</td><td colspan="2"> $( q _ { i } , q _ { t } )$ </td><td colspan="2">Query: qt</td><td colspan="2"> $( q _ { i } , q _ { t } )$ </td></tr><tr><td>COCO</td><td>VN</td><td>F200K</td><td>NIGHTS</td><td>FIQ</td><td>CIRR</td><td>WebQA</td><td>COCO</td><td>VN</td><td>F200K</td><td>OVEN</td><td>InfoS</td><td>EDIS</td><td>WebQA</td><td>OVEN</td><td>InfoS</td></tr><tr><td colspan="10">R@5 R@10 R@5 R@10 R@5</td><td></td><td>R@5</td><td>R@5</td><td>R@5</td><td>R@5</td><td>R@5</td><td>R@5</td></tr><tr><td colspan="10">Embedding-based Retrieval</td><td></td><td></td><td></td><td>59.4</td><td>78.7</td><td>67.6</td><td></td></tr><tr><td>CLIP-SF BLIP-FF</td><td>81.1 79.7</td><td>42.6</td><td>18.0</td><td>32.0 33.0</td><td>24.4</td><td>44.6</td><td>84.7</td><td>92.3 43.1</td><td>18.3</td><td>45.5</td><td>27.9 22.4</td><td></td><td></td><td></td><td></td><td>48.9 33.0</td></tr><tr><td>U-MARVEL</td><td>84.4</td><td>23.4 47.3</td><td>26.1 33.6</td><td>34.2</td><td>29.2 36.4</td><td>52.2 60.7</td><td>80.0 97.1</td><td>89.9 93.5</td><td>22.8 47.3</td><td>28.9 35.1</td><td>41.0 62.5</td><td>58.3</td><td>50.9</td><td>79.8 88.5</td><td>55.8 79.4</td><td>74.7</td></tr><tr><td colspan="10"></td><td colspan="3"></td><td colspan="3">78.8</td><td colspan="3"></td></tr><tr><td>IRGen</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Generative Retrieval</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRACE</td><td>50.7</td><td>一</td><td></td><td>一</td><td></td><td></td><td>一</td><td></td><td></td><td></td><td></td><td>一</td><td></td><td></td><td></td><td></td></tr><tr><td>Baseline</td><td>39.5 75.5</td><td>一 22.2</td><td>13.3</td><td>一 29.0</td><td>16.3</td><td>37.2</td><td>一</td><td>89.3</td><td>22.2</td><td>14.9</td><td>41.5</td><td>一 19.7</td><td>40.0</td><td>53.3</td><td>52.1</td><td>30.8</td></tr><tr><td>GENIUSR</td><td>78.0</td><td>27.4</td><td>16.2</td><td>30.2</td><td>19.3</td><td>39.5</td><td>38.4 44.6</td><td>91.1</td><td>28.4</td><td>16.3</td><td>41.9</td><td>20.7</td><td>44.3</td><td>60.6</td><td>52.5</td><td>30.1</td></tr><tr><td>WIDE (Ours)</td><td>79.8</td><td>30.2</td><td>17.6</td><td>31.5</td><td>21.7</td><td>41.5</td><td>46.2</td><td>89.8</td><td>30.2</td><td>18.1</td><td>42.7</td><td>21.4</td><td>44.8</td><td>63.1</td><td>53.2</td><td>32.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 2: Ablation Study of the WIDE Framework. Performance is evaluated by incrementally integrating wildcard decoding, adaptive thresholding, and re-ranking modules. The “Fixed” variant uses an empirically tuned static entropy threshold.
<table><tr><td rowspan="2">Model Variants</td><td colspan="3">Components</td><td rowspan="2">COCO qt → ci</td><td rowspan="2">VN</td><td rowspan="2">WebQA qt → ct</td><td rowspan="2">FIQ  $( q _ { i } , q _ { t } ) \to c _ { i }$ </td></tr><tr><td>AWD</td><td>AET</td><td>BSR</td></tr><tr><td>Baseline</td><td></td><td></td><td></td><td>75.5</td><td>qt → ci 22.2</td><td>38.4</td><td>16.3</td></tr><tr><td>Row 1 (Fixed τ)</td><td>√</td><td></td><td></td><td>76.9</td><td>24.8</td><td>41.0</td><td>18.1</td></tr><tr><td>Row 2 (Dynamic τ)</td><td>√</td><td>√</td><td></td><td>77.2</td><td>26.7</td><td>43.0</td><td>18.9</td></tr><tr><td>WIDE (Full)</td><td>√</td><td>√</td><td>√</td><td>79.8</td><td>30.2</td><td>46.2</td><td>21.7</td></tr></table>

5.3.1 Efectiveness of Wildcard Decoding. Row 1 of Table 2 evaluates the efect of wildcard decoding using a fixed entropy threshold. Compared with the baseline, introducing wildcard decoding consistently improves retrieval performance across all four datasets, with gains ranging from 1.4 to 2.6 percentage points. These results show that replacing uncertain intermediate identifier generations with wildcards can retain relevant retrieval paths that would otherwise be prematurely pruned during trie-constrained decoding.

5.3.2 Adaptive versus Fixed Thresholding. Row 2 replaces the fixed entropy threshold with the proposed adaptive entropy thresholding module. Compared with the fixed-threshold variant, adaptive thresholding further improves performance on all four datasets, with gains of 0.3, 1.9, 2.0, and 0.8 percentage points on MS-COCO, VisualNews, WebQA, and FashionIQ, respectively. This consistent improvement indicates that a level-specific entropy criterion is more efective than a single fixed threshold for determining when wildcard decoding should be activated. By assigning diferent entropy thresholds to diferent RQ levels, AET more accurately identifies intermediate query semantic blind spots.

5.3.3 Efectiveness of BSR. The full WIDE framework further introduces BSR to re-rank the candidate pool produced by AWD and

![](images/db2b12b751bfbd29327e2147b9674e21c7443d90089e7396aface39d1bd33f19.jpg)  
Figure 4: Layer-wise dynamics of the adaptive entropy threshold and wildcard trigger ratio.

AET. Adding BSR improves performance by 2.6, 3.5, 3.2, and 2.8 percentage points on MS-COCO, VisualNews, WebQA, and FashionIQ, respectively. After wildcard decoding, candidates associated with the same beam satisfy the same reliable identifier constraints. BSR therefore combines generation confidence from the constrained levels with continuous embedding similarity to distinguish these candidates more efectively. The consistent gains over Row 2 demonstrate the importance of re-ranking the expanded candidate pool after wildcard decoding.

## 5.4 Analysis

5.4.1 Precision ofAdaptive Uncertainty Intervention. Figure 4 illustrates the distribution of the dynamic entropy threshold alongside the corresponding wildcard trigger ratio across residual levels. A significant discrepancy in entropy exists between the shallow and deep quantization layers. Because the adaptive threshold derives from historical entropy statistics, this distinct upward trajectory demonstrates that the model inherently experiences greater uncertainty when generating identifiers at deeper levels compared to shallow ones. Shallow levels capture semantics that align well with text queries, resulting in low historical uncertainty. Deeper levels encode details typically absent from the text, which elevates the expected entropy boundary.

Table 3: Wildcard Expansion Statistics of the WIDE Framework. The table reports the database size |C|, the average number of wildcarded levels, and the candidate-set size induced by wildcard decoding before cross-beam aggregation and BSR re-ranking.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Database Size |C|</td><td>Wildcarded Levels</td><td colspan="3">Wildcard Expansion Size</td></tr><tr><td>Avg. |Ai|</td><td>Average</td><td>P95</td><td>Reduction (%)</td></tr><tr><td>MS-COCO</td><td>24,809</td><td>1.77</td><td>1.1</td><td>1.0</td><td>99.99%</td></tr><tr><td>Fashion200K</td><td>201,824</td><td>1.96</td><td>21.9</td><td>34.0</td><td>99.99%</td></tr><tr><td>WebQA</td><td>544,457</td><td>1.90</td><td>276.8</td><td>2.0</td><td>99.95%</td></tr><tr><td>CIRR</td><td>21,551</td><td>1.92</td><td>55.0</td><td>3.0</td><td>99.74%</td></tr><tr><td>VisualNews</td><td>542,246</td><td>1.93</td><td>1,654.2</td><td>57.0</td><td>99.69%</td></tr><tr><td>EDIS</td><td>1,047,067</td><td>2.06</td><td>2,357.7</td><td>224.0</td><td>99.77%</td></tr><tr><td>Query-weighted Avg.</td><td>=</td><td>1.97</td><td>1,142.7</td><td>=</td><td>99.76%</td></tr></table>

Despite the substantial increase in the entropy threshold at deeper levels, the wildcard trigger ratio exhibits a smooth and steady rise. This observation aligns with our fundamental design objective that wildcard activation should not concentrate disproportionately within specific architectural layers. Instead, the intervention relies entirely on the dynamic gap between the real-time predictive entropy and AET threshold. The steady trajectory of the trigger rate confirms that the AET module efectively regulates the decoding process. It prevents the system from emitting wildcards prematurely while successfully ensuring that no individual layer undergoes excessive candidate expansion.

5.4.2 Candidate Expansion under Wildcard Decoding. Table 3 analyzes the candidate expansion induced by wildcard decoding relative to the full database size |C|. Rather than measuring the final candidate pool after aggregating all source beams, the reported statistics characterize the candidates matched by wildcard-based expansion while preserving the remaining identifier constraints. Across the evaluated datasets, this local expansion retains only a small fraction of the database, with a query-weighted reduction of 99.76%.

On average, wildcard decoding is activated at 1.97 identifier levels. This relatively small number indicates that AWD modifies only a limited portion of the identifier sequence, while the remaining levels continue to constrain candidate matching. Consequently, uncertain intermediate generations can be relaxed without removing the discrete constraints provided by the rest of the identifier.

The expansion size also exhibits a strongly skewed distribution on several datasets. For example, WebQA has an average expansion size of 276.8 but a $P _ { 9 5 }$ of only 2.0, indicating that most wildcard activations correspond to very small candidate sets, while a small number of wildcard patterns match substantially more candidates and increase the average. Similar behavior can be observed on CIRR, VisualNews, and EDIS.

5.4.3 Qualitative Analysis ofRetrieval Results. To comprehensively illustrate the advantages of the WIDE framework over the baseline, we present qualitative retrieval results in Figure 5. Specifically, Figure 5(a) demonstrates that the framework efectively resolves cross-modal information asymmetry by utilizing the BSR module to compute hybrid scores, which elevates the rank of true targets containing rich visual details absent from concise text queries. Furthermore, Figure 5(b) illustrates robust candidate pool recovery. By substituting deterministic identifier generation at highly uncertain levels with wildcards, WIDE successfully retrieves ground truths that the baseline entirely misses or ranks exceptionally low. As evidenced in Figure 5(c), even when the baseline already positions the target near the top of the retrieved list, the framework further refines the ranking to achieve the highest possible placement. Finally, Figure 5(d) highlights a strong capacity for distractor mitigation in complex scenes, where WIDE consistently prioritizes the correct target over visually similar but inaccurate candidates.

![](images/2693177103540d1da4eb126e1492451d01bc3090958d6e6c4a4a666245be7e05.jpg)  
Figure 5: Qualitative comparison of retrieval results. For each query, the first and second rows show the top-5 results retrieved by the baseline and WIDE, respectively. The numbers below each candidate denote its discrete identifier, and the wildcarded identifier shown for WIDE marks the levels replaced by “\*” during wildcard decoding.

## 6 Limitations

WIDE improves retrieval accuracy by expanding candidate paths at uncertain identifier levels, which inevitably introduces additional decoding and re-ranking overhead compared with standard trieconstrained beam search despite the restricted wildcard activation. The expanded candidate pool is further re-ranked using continuous embedding similarity, meaning that the final retrieval stage still partially relies on the semantic discrimination provided by embedding-based representations. Moreover, WIDE relies on entropy deviations to identify query semantic blind spots, while high uncertainty may also arise from noisy or ambiguous queries or insuficient decoder confidence, potentially causing unnecessary wildcard activation.

## 7 Conclusion

In this paper, we revisit cross-modal generative retrieval from the perspective of information asymmetry and identify forced hallucination as an important failure mode in discrete identifier generation. To address this issue, we propose WIDE, which introduces wildcard inference with dynamic expansion to relax unreliable decoding decisions and preserve alternative retrieval paths. WIDE further incorporates a hybrid re-ranking strategy that combines reliable generative evidence with continuous semantic similarity to distinguish the expanded candidates. Experiments on the M-BEIR benchmark show that WIDE provides clear improvements over existing generative retrieval methods across diverse cross-modal retrieval tasks. Overall, our results demonstrate the efectiveness of explicitly handling uncertainty in cross-modal generative retrieval while retaining its discrete retrieval formulation.

## Acknowledgments

This work was supported by the Department of Science and Technology of Jilin Province (Grant Number: 20260301004YY).

## References

[1] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. 2017. Guided Open Vocabulary Image Captioning with Constrained Beam Search. In Proceedings ofthe 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP 2017, Copenhagen, Denmark, September 9-11, 2017. Association for Computational Linguistics, 936–945.

[2] Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2021. Au toregressive Entity Retrieval. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

[3] Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. 2022. Webqa: Multihop and multimodal qa. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 16495–16504.

[4] Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Yiqun Liu, Yixing Fan, and Xueqi Cheng. 2022. CorpusBrain: Pre-train a Generative Retrieval Model for Knowledge Intensive Language Tasks. In Proceedings ofthe 31st ACM International Conference on Information & Knowledge Management, Atlanta, GA, USA, October 17-21, 2022. ACM, 191–200.

[5] Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei Chang. 2023. Can pre-trained vision and language models answer visual information-seeking questions?. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing. 14948–14968

[6] Minghui Fang, Shengpeng Ji, Jialong Zuo, Hai Huang, Yan Xia, Jieming Zhu, Xize Cheng, Xiaoda Yang, Wenrui Liu, Gang Wang, Zhenhua Dong, and Zhou Zhao. 2025. CART: A Generative Cross-Modal Retrieval Framework With Coarse-To-Fine Semantic Modeling. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025. Association for Computational Linguistics, 15120–15133.

[7] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. 2023. DreamSim: Learning new dimensions of human visual similarity using synthetic data. In Advances in Neural Information Processing Systems, Vol. 36.

[8] Xintong Han, Zuxuan Wu, Phoenix X Huang, Xiao Zhang, Menglong Zhu, Yuan Li, Yang Zhao, and Larry S Davis. 2017. Automatic spatially-aware fashion concept discovery. In Proceedings of the IEEE international conference on computer vision. 1202–1211.

[9] Dan Hendrycks and Kevin Gimpel. 2017. A Baseline for Detecting Misclas sified and Out-of-Distribution Examples in Neural Networks. In International Conference on Learning Representations.

[10] Chris Hokamp and Qun Liu. 2017. Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search. In Proceedings ofthe 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers. Association for Computational Linguistics, 1535–1546.

[11] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The Curious Case ofNeural Text Degeneration. In International Conference on Learning Representations.

[12] Hexiang Hu, Yi Luan, Yang Chen, Urvashi Khandelwal, Mandar Joshi, Kenton Lee, Kristina Toutanova, and Ming-Wei Chang. 2023. Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 11206–11216.

[13] Sungyeon Kim, Xinliang Zhu, Xiaofan Lin, Muhammet Bastan, Douglas Gray, and Suha Kwak. 2025. GENIUS: A generative framework for universal multimodal search. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 19659–19669.

[14] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. In International Conference on Learning Representations.

[15] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. 2022. Autoregressive Image Generation using Residual Quantization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11523– 11532.

[16] Haoxuan Li, Yi Bin, Yunshan Ma, Guoqing Wang, Yang Yang, See-Kiong Ng, and Tat-Seng Chua. 2025. SemCORE: A Semantic-Enhanced Generative Cross-Modal Retrieval Framework with MLLMs. CoRR abs/2504.13172 (2025).

[17] Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yuyao Zhang, Peitian Zhang, Yutao Zhu, and Zhicheng Dou. 2025. From matching to generation: A survey on generative information retrieval. ACM Transactions on Information Systems 43, 3 (2025), 1–62.

[18] Xiaojie Li, Chu Li, Shi-Zhe Chen, and Xi Chen. 2026. U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning with

MLLMs. In International Conference on Learning Representations.

[19] Yongqi Li, Hongru Cai, Wenjie Wang, Leigang Qu, Yinwei Wei, Wenjie Li, Liqiang Nie, and Tat-Seng Chua. 2025. Revolutionizing Text-to-Image Retrieval as Autoregressive Token-to-Voken Generation. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2025, Padua, Italy, July 13-18, 2025. ACM, 813–822.

[20] Yongqi Li, Wenjie Wang, Leigang Qu, Liqiang Nie, Wenjie Li, and Tat-Seng Chua. 2024. Generative Cross-Modal Retrieval: Memorizing Images in Multimodal Language Models for Retrieval and Beyond. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, 11851–11861.

[21] Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Zou. 2022. Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

[22] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft COCO: Common objects in context. In European conference on computer vision. Springer, 740–755.

[23] Fuxiao Liu, Yinghan Wang, Tianjun Wang, and Vicente Ordonez. 2021. Visu alNews: Benchmark and Challenges in News Image Captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 6761–6771.

[24] Siqi Liu, Weixi Feng, Tsu-Jui Fu, Wenhu Chen, and William Wang. 2023. EDIS: Entity-Driven Image Search over Multimodal Web Content. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 4877–4894.

[25] Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, and Stephen Gould. 2021. Image Retrieval on Real-life Images with Pre-trained Vision-and-Language Models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 2125–2134.

[26] Ilya Loshchilov and Frank Hutter. 2019. Decoupled Weight Decay Regularization. In International Conference on Learning Representations.

[27] Andrey Malinin and Mark J. F. Gales. 2021. Uncertainty Estimation in Autoregressive Structured Prediction. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

[28] Leigang Qu, Haochuan Li, Tan Wang, Wenjie Wang, Yongqi Li, Liqiang Nie, and Tat-Seng Chua. 2025. TIGeR: Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

[29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning. PMLR, 8748–8763.

[30] Colin Rafel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research 21, 140 (2020), 1–67.

[31] EuiYul Song, Sangryul Kim, Haeju Lee, Joonkee Kim, and James Thorne. 2024. Re3val: Reinforced and Reranked Generative Retrieval. In Findings ofthe Association for Computational Linguistics: EACL 2024, St. Julian’s, Malta, March 17-22, 2024 (Findings ofACL). Association for Computational Linguistics, 393–409.

[32] Felix Stahlberg and Bill Byrne. 2019. On NMT Search Errors and Model Errors: Cat Got Your Tongue?. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 3356–3362.

[33] Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Prakash Gupta, Tal Schuster, William W. Cohen, and Donald Metzler. 2022. Transformer Memory as a Diferentiable Search Index. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.).

[34] Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Sun, Weiwei Deng, Qi Zhang, and Mao Yang. 2022. A Neural Corpus Indexer for Document Retrieval. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022

[35] Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. 2024. UniIR: Training and Benchmarking Universal Multimodal Information Retrievers. In European Conference on Computer Vision. Springer.

[36] Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogerio Feris. 2021. Fashion IQ: A new dataset towards retrieving images by natural language feedback. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11307–11317.

[37] Zhiqiu Xia, Jinxuan Xu, Yuqian Zhang, and Hang Liu. 2025. A Survey of Uncertainty Estimation Methods on Large Language Models. In Findings of the Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025 (Findings of ACL). Association for Computational Linguistics, 21381–21396.

[38] Liu Yang, Fabian Paischer, Kaveh Hassani, Jiacheng Li, Shuai Shao, Zhang Gabriel Li, Yun He, Xue Feng, Nima Noorshams, Sem Park, Bo Long, Robert D. Nowak, Xiaoli Gao, and Hamid Eghbalzadeh. 2025. Unifying Generative and Dense Retrieval for Sequential Recommendation. Trans. Mach. Learn. Res. 2025 (2025).

[39] Yuhao Yang, Zhi Ji, Zhaopeng Li, Yi Li, Zhonglin Mo, Yue Ding, Kai Chen, Zijian Zhang, Jie Li, Shuanglong Li, and Liu Lin. 2025. Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations. In

Advances in Neural Information Processing Systems, Danielle Belgrave, Cheng Zhang, Huan Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, and Niao Chen (Eds.), Vol. 38. Curran Associates, Inc., 93746–93770.

[40] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. 2022. Soundstream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing 30 (2022), 495–507.

[41] Tunyu Zhang, Haizhou Shi, Yibin Wang, Hengyi Wang, Xiaoxiao He, Zhuowei Li, Haoxian Chen, Ligong Han, Kai Xu, Huan Zhang, Dimitris N. Metaxas, and Hao Wang. 2026. TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning. In The Fourteenth International Conference on Learning Representations.