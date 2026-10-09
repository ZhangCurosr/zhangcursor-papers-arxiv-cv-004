# Perceptually Grounded and Semantics-Aware Evaluation for Holistic Co-Speech Gesture Generation

Nick Milkin<sup>1,2†</sup>, Lanmiao Liu<sup>1,2,3†</sup>, Esam Ghaleb<sup>1,3</sup>, Asli Ozyurek<sup>1,3</sup>, Zerrin Yumak<sup>2</sup>

<sup>1</sup> Max Planck Institute for Psycholinguistics, Nijmegen, The Netherlands <sup>2</sup> Utrecht University, Utrecht, The Netherlands

<sup>3</sup> Donders Institute for Brain, Cognition and Behaviour, Nijmegen, The Netherlands nick.milkin, lanmiao.liu, esam.ghaleb, asli.ozyurek@mpi.nl, z.yumak@uu.nl

## Abstract

Holistic and semantics-aware co-speech gesture generation has advanced rapidly, yet evaluation remains behind: objective metrics do not consistently reflect human perception, and semantic appropriateness remains difficult to quantify. We present a perceptually grounded and semantics-aware benchmark that combines standardized model comparison, human-centered metric validation, and fine-grained semantic evaluation. We first curate a list of 13 objective metrics covering different as pects, including distributional similarity, geometric fidelity, kinematic quality, cross-modal synchrony, and semantic appropriateness. For the semantic-appropriateness category, we propose a new metric, Semantic Gesture Preservation (SGP), which measures how far semantic gestures in the ground truth are preserved in the generated gestures. For this, we aug ment the BEAT2 dataset’s annotations using a multi-modal LLM. We then conduct a perceptual study where 101 participants score generated gestures among five dimensions, including human-likeness, motion diversity, absence of animation errors speech timing and content match. We systematically analyze objective metric–subjective score correlations. Unlike Seman tic Score (SC), which shows no significant association with the evaluated perceptual dimensions, SGP is selectively aligned with speech-aware human judgments. We construct five target-specific composite metrics aligned with the subjective dimensions. These composites improve perceptual alignment across all five dimensions, with the largest gains for absence of ani mation errors and content match, indicating that complementary objective signals can better approximate human judgments than individual metrics alone. Overall, our results show that objective metrics require validation against subjective evaluations. Gesture quality assessment is complex and can benefit from semantics-aware and composite objective metrics. We hope that these findings shed light to and form the basis for further fine-grained semantics-aware and composite metric assessments across a larger number of methods. Demos and qualitative illustrations are available on the anonymized project page: https://holisticgesturebenchmark.github.io/HolisticGestureBenchmark/. Code for individual and composite metrics and anno tations for the dataset will be provided upon acceptance with a Github page.

• Computing methodologies → Computer vision; Animation; Machine learning;

## 1. Introduction

Human communication is fundamentally multimodal [HL19], where speech is linked to non-verbal behaviors that encode intent, emphasis, and discourse structure. As a cornerstone for developing virtual humans, co-speech gestures must exhibit deep behavioral coherence and social appropriateness rather than just visual fidelity [CGM 20,HM12,GWK26,GKOF25]. While early learningbased approaches targeted rhythmic, acoustics-driven upper-body motion [HXM<sup>∗</sup>21], the field has recently undergone a paradigm shift toward a holistic synthesis that integrates the body, hands, and face while leveraging richer linguistic and semantic conditioning [LZB<sup>∗</sup>24]. However, as these generative architectures grow in complexity, the methodology for their evaluation has lagged. Cospeech gesture quality is inherently multifaceted, demanding a synergy of naturalness, diversity, temporal synchrony, and semantic relevance. Capturing these attributes remains an open challenge, leaving current practices reliant on fragmented and narrow evaluation protocols.

A fundamental question therefore persists regarding whether prevalent objective metrics designed to quantify motion fidelity, diversity, or temporal alignment [LZI 22, AZL23] truly align with human perception. Recent evidence suggests a significant divergence between numerical gains and subjective preferences [NVY<sup>∗</sup>24], indicating that current metrics often fail to capture the nuanced perceptual dimensions of high-quality gestures. This disconnect is most pronounced in semantic gesture evaluation: a gesture may be kinematically natural and temporally synchronized yet remain semantically vacuous or mismatched. Unlike rhythmic motion, semantic gestures are sparse, context-dependent, and exhibit a complex many-to-many mapping between meaning and form, rendering conventional statistical measures ill-equipped to capture these high-level relationships.

Reliable perceptual validation is further complicated by a systemic lack of standardization across the literature, where disparate datasets, preprocessing pipelines, rendering conditions, and subjective study designs have fostered a “Wild West” of evaluation in non-verbal behavior generation [NVHM<sup>∗</sup>26, HPY25]. In this environment, performance gains reported in individual studies are often not directly comparable, as apparent improvements may stem from idiosyncratic experimental settings rather than fundamental algorithmic advances, necessitating a transition toward evaluating representative models under a common protocol. Establishing such a standardized setting is therefore essential for determining whether objective metrics genuinely reflect human perception and for enabling controlled analysis of semantic gesture quality.

In this work, we propose a unified evaluation framework for semantics-aware holistic co-speech gesture generation bridging the gap between perceptual validation and objective metrics. We first conduct a systematic study of objective-subjective correlations drawing inspiration from multi-dimensional evaluation in other domains (i.e facial animation [DSR<sup>∗</sup>25] and general body motion [RWH<sup>∗</sup>26]). Rather than treating objective metrics as isolated scores, we analyze their perceptual relevance across five distinct dimensions and construct five target-specific composite metrics that combine complementary objective signals to predict human judgments of human-likeness, motion diversity, absence of animation errors, speech timing, and content match. To address the semantic gesture evaluation gap, we further introduce an LLM-assisted framework for fine-grained semantic annotation and derive a new semantics-aware metric. We empirically validate this metric against human judgments of content appropriateness, assessing its efficacy as an automated proxy for semantic performance. To support these analyses under controlled and directly comparable conditions, we establish a benchmark involving four state-of-the-art holistic gesture generation models: EMAGE [LZB<sup>∗</sup>24], SemTalk [ZLZ<sup>∗</sup>25], GestureLSM [LSH<sup>∗</sup>25], and SemConFlow [LGÖY26], all evaluated under a shared data and rendering pipeline. By integrating these components, we provide a rigorous foundation for measuring progress in semantics-aware holistic co-speech gesture generation, offering insights into the strengths and limitations of the current evaluative toolkit. Our primary contributions are:

• Perceptual Validation and Target-Specific Composite Metrics: We systematically evaluate objective metrics against human judgments across five perceptual dimensions and construct five target-specific composite predictors that combine complementary objective signals to better approximate dimensionspecific human ratings.

• Semantics-Aware Gesture Evaluation Framework: We establish a scalable paradigm for fine-grained semantic gesture annotation leveraging the cross-modal reasoning of LLMs, contributing an annotated corpus of semantic gesture descriptions to the community. Building upon this, we introduce the Semantic Gesture Preservation (SGP) metric, a new objective measure for semantic evaluation that is assessed against human perception.

• A Semantics-aware Holistic Co-Speech Gesture Generation Benchmark: We provide standardized objective metrics and subjective scores comparison of four representative generative models under a unified dataset, rendering pipeline, and evaluation protocol, enabling direct and fair performance analysis.

## 2. Related Work

## 2.1. Co-speech gesture generation

Co-speech gesture generation aims to synthesize human motion conditioned on multimodal signals such as speech and text, with the goal of producing behavior that is both perceptually plausible and communicatively appropriate [NKA<sup>∗</sup>23a]. Early data-driven approaches mainly focused on generating rhythmic motion that followed the prosodic structure of speech [HXM<sup>∗</sup>21], whereas later methods increasingly incorporated linguistic and semantic information to improve the correspondence between speech content and generated gestures. Representative directions include combining low-level acoustic features with higher-level speech representations [CLW<sup>∗</sup>24], explicitly modeling semantically rich gestures separately from rhythmic motion [ZCC<sup>∗</sup>23], retrieving semantically relevant motion from external databases [ZGP<sup>∗</sup>23,MDS<sup>∗</sup>25], and optimizing semantic consistency between speech and motion during training [LGOY25] using contrastive learning [LGÖY26]. These developments reflect a broader shift from generating motion that is primarily synchronized with speech toward gestures that also capture communicative meaning.

In parallel, advances in motion representation have enabled increasingly expressive and holistic generation. Discrete latent representations based on VQ-VAEs [PLXL21] and RQ-VAEs [LKK<sup>∗</sup>22] have been widely adopted to model complex motion distributions and provide stronger priors for full-body synthesis. EMAGE $\scriptstyle [ \mathrm { L Z B } ^ { * } 2 4 ]$ models different body components through compositional VQ-VAEs, while SemTalk [ZLZ<sup>∗</sup>25] employs separate residual-quantized representations for different body regions. Together with other recent systems such as GestureLSM [LSH<sup>∗</sup>25] and TalkSHOW [YLL<sup>∗</sup>23], these methods extend co-speech gesture generation toward coordinated face, hand, and body motion under richer multi-modal conditioning [LGÖY26]. Despite rapid progress, evaluation of co-speech gesture synthesis methods remains an open topic as these methods are aspiring to be more semantically-aware and holistic. Current methods do not specifically evaluate the semantic aspect in depth and rely on a set of common objective metrics and limited subjective evaluations.

## 2.2. Human-Centered Validation of Objective Metrics

Human evaluation remains the most direct way to assess the perceptual quality of generated gestures and is widely regarded as the gold standard in co-speech gesture generation $\mathrm { [ N V Y ^ { * } 2 4 ] }$ . However, subjective results are sensitive to study design, including question formulation, presented modalities, and the perceptual dimensions participants are asked to judge [NVHM 26, WRB22]. The GENEA evaluations sought to reduce such confounds by separating motion-quality judgments from speech-dependent appropriateness, for example through muted motion evaluation and mismatched speech–motion pairs $[ \mathrm { N V Y ^ { * } } 2 4 , \mathrm { K W Y ^ { * } } 2 4 ]$ . Nevertheless, subjective protocols remain heterogeneous across studies and are costly to reproduce at scale, motivating the widespread use of automatic objective metrics.

Objective metrics provide efficient and reproducible estimates of properties such as motion fidelity, diversity, and speech–motion alignment, yet their correspondence with human perception remains uncertain. Large-scale evaluations have shown that correlations between commonly used objective metrics and subjective judgments are not always strong $[ \mathrm { K W Y ^ { * } } 2 4 , \mathrm { N V Y ^ { * } } 2 4 , \mathrm { N K A ^ { * } } 2 3 \mathrm { b } ]$ In other fields such as general body motion Rekik et al. [RWH<sup>∗</sup>26] and facial animation Delbosc et al. $\mathrm { [ D S R } ^ { * } 2 5 ]$ researchers suggested to use combined metrics as an alternative. These studies suggest that complementary metrics can provide more informative perceptual signals than individual scores alone. However, for the field of co-speech gesture generation the idea of combined metrics particularly for holistic and semantics-aware methods have not been explored. In this work, we first systematically analyze correlations between automatic metrics and human judgments for recent semantics-aware and holistic methods. We explicitly extend the current repertoire of objective metrics to better capture these aspects and correlate them with a carefully selected list of subjective scores. Based on these findings, we propose combined objective metrics that better correlates with subjective evaluations.

## 2.3. Assessing Semantic Grounding in Co-Speech Gestures

Multi-modal large language models (MLLMs) have recently emerged as scalable tools for semantic annotation and automatic evaluation. Their ability to jointly reason over visual and linguistic information has enabled applications beyond conventional recognition, including fine-grained multi-modal assessment and data annotation. Gemini -2.5 [CBS<sup>∗</sup>25] has been investigated as a generalpurpose evaluator for vision-language tasks, showing promising agreement with human judgments across several evaluation settings [TKZ 26]. Hu et al. $\mathrm { [ H A H H ^ { * } 2 6 ] }$ studies the use of MLLMs for scoring, pairwise comparison, and ranking across multiple tasks. Recent dataset construction pipelines have also exploited multimodal models for scalable annotation; $\scriptstyle [ Z \mathbf { J } \mathbf { Y } ^ { * } 2 6 ]$ uses Gemini-2.5 to generate fine-grained video reasonin MLLMs can provide rich semantic supervision without requiring every annotation to be manually specified. However, their potential for systematically annotating and evaluating the semantic content of co-speech gestures remains largely unexplored.

Existing semantics-aware evaluation approaches instead rely on predefined semantic annotations [LZI<sup>∗</sup>22] and learned cross-modal representations [AZL23, LGOY25, PLÖ<sup>∗</sup>26, LYS<sup>∗</sup>23]. Semantic Relevant Gesture Recall (SRGR) [LZI<sup>∗</sup>22] weights pose accuracy according to semantic relevance annotations, emphasizing motion correctness at semantically important moments. Semantic Score (SC) [AZL23] measures similarity between speech and generated motion in a learned gesture-text embedding space, while Emotional Accuracy (EA) [QLL<sup>∗</sup>24] evaluates whether generated motion conveys a target affective category. Canonical Correlation Analysis (CCA) [KWY<sup>∗</sup>24, Tho84] provides a more indirect measure by quantifying shared structure between generated and reference motion. Although these metrics extend evaluation beyond low-level motion statistics, they provide only partial access to whether a generated gesture is semantically present, what meaning it conveys, and whether that meaning is consistent with the accompanying speech. Overall, fine-grained semantic gesture evaluation remains constrained by limited scalable annotation and by metrics that only indirectly capture gesture meaning. We address this gap by leveraging large multimodal models for detailed semantic gesture annotation and, building on these annotations, introduce a semanticsaware metric that takes into account micro-level annotations.

## 3. Methodology

We curate a diverse set of objective metrics covering multiple aspects of motion and introduce a new semantics metric based on MLLM-derived fine-grained motion annotations. We evaluate recent holistic and semantics-aware gesture models, conduct a perceptual study extending existing subjective criteria with motion diversity, absence of animation errors, and content match, and analyze objective–subjective correlations to derive potential composite metrics.

## 3.1. Benchmark Objective Metrics

To quantify model performance, we organize objective metrics into a multidimensional taxonomy spanning distributional similarity, geometric fidelity, kinematic quality, cross-modal synchrony, and semantic appropriateness. The selected metrics operate across latent, distributional, and raw-signal levels, providing complementary measures of physical plausibility and communicative quality in co-speech gesture generation.

## 3.1.1. Latent-space Distribution-Based Metrics

3.1.1.1. Fréchet Gesture Distance (FGD) Fréchet Gesture Distance (FGD) [YCL<sup>∗</sup>20] measures the discrepancy between real and generated motion distributions in a learned latent feature space. Le $: \mu ^ { A } , \Sigma ^ { A }$ and $\mu ^ { G } , \Sigma ^ { G }$ denote the corresponding means and covariances. FGD is defined as

$$
\mathrm { F G D } = \left\| \boldsymbol { \mu } ^ { A } - \boldsymbol { \mu } ^ { G } \right\| _ { 2 } ^ { 2 } + \mathrm { T r } \left( \boldsymbol { \Sigma } ^ { A } + \boldsymbol { \Sigma } ^ { G } - 2 \left( \boldsymbol { \Sigma } ^ { A } \boldsymbol { \Sigma } ^ { G } \right) ^ { 1 / 2 } \right) .\tag{1}
$$

Lower values indicate closer agreement between the two motion distributions, while the metric depends on the learned feature encoder and does not directly capture speech–gesture semantics.

3.1.1.2. Density Density [NOU<sup>∗</sup>20] measures how well generated samples align with local regions of the real-data manifold. Unlike binary precision, it counts the number of reference neighborhoods containing each generated feature. Let $\{ X _ { i } \} _ { i = 1 } ^ { N }$ and $\{ Y _ { j } \} _ { j = 1 } ^ { M }$ denote reference and generated features, respectively, and let ${ \mathrm { N N D } } _ { k } ( X _ { i } )$ be the distance from $X _ { i }$ to its k-th nearest reference neighbor. Density is defined as

$$
{ \mathrm { D e n s i t y } } = { \frac { 1 } { k M } } \sum _ { j = 1 } ^ { M } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \left[ Y _ { j } \in B \left( X _ { i } , { \mathrm { N N D } } _ { k } ( X _ { i } ) \right) \right] ,\tag{2}
$$

where $B ( \cdot )$ denotes the local reference neighborhood. Higher values indicate that generated motions more frequently fall within well-supported regions of the reference distribution.

3.1.1.3. Coverage Coverage $[ \mathrm { N O U } ^ { * } 2 0 ]$ complements Density by measuring how extensively the generated samples cover the support of the reference motion distribution. Using the same k-nearestneighbor neighborhoods defined for Density, Coverage computes the proportion of reference samples whose local neighborhood contains at least one generated sample:

$$
\mathrm { C o v e r a g e } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \left[ \exists j \mathrm { s u c h } \mathrm { t h a t } Y _ { j } \in B \left( X _ { i } , \mathrm { N N D } _ { k } ( X _ { i } ) \right) \right] .\tag{3}
$$

Higher Coverage indicates that the generated samples represent a larger fraction of the real-motion manifold. Density and Coverage therefore provide complementary estimates of generation fidelity and support coverage in latent feature space.

## 3.1.2. Geometric Fidelity Metrics

3.1.2.1. Chamfer Distance Chamfer Distance $[ \mathrm { R W H } ^ { * } 2 6 ]$ measures geometric similarity between generated and reference motions using their root-aligned vertex point clouds. Let $S _ { i } ^ { A }$ and $S _ { i } ^ { G }$ denote the reference and generated vertex sets at frame i. The symmetric distance over n frames is

$$
\begin{array} { r l r } & { } & { C ( \{ S _ { i } ^ { A } \} _ { i = 1 } ^ { n } , \{ S _ { i } ^ { G } \} _ { i = 1 } ^ { n } ) = \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \frac { 1 } { | S _ { i } ^ { A } | } \sum _ { x \in S _ { i } ^ { A } } \operatorname* { m i n } _ { y \in S _ { i } ^ { G } } | | x - y | | _ { 2 } ^ { 2 } \right. } \\ & { } & { \left. + \frac { 1 } { | S _ { i } ^ { G } | } \sum _ { y \in S _ { i } ^ { G } } \operatorname* { m i n } _ { x \in S _ { i } ^ { A } } | | x - y | | _ { 2 } ^ { 2 } \right) . } \end{array}\tag{4}
$$

Lower values indicate closer average surface geometry, while primarily reflecting geometric fidelity rather than perceptual or semantic appropriateness.

3.1.2.2. Hausdorff Distance Hausdorff Distance $[ \mathrm { v M O } ^ { \ast } 2 2 ]$ measures the largest geometric discrepancy between generated and reference motion surfaces, making it more sensitive to local deviations than average-based distances. For reference and generated point sets $\boldsymbol { S } _ { i } ^ { A }$ and $\bar { S } _ { i } ^ { G }$ at frame i, it is defined as

$$
\begin{array} { r } { H ( \{ S _ { i } ^ { A } \} _ { i = 1 } ^ { n } , \{ S _ { i } ^ { G } \} _ { i = 1 } ^ { n } ) = \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \operatorname* { m a x } \bigg \{ \underset { x \in S _ { i } ^ { A } \ y \in S _ { i } ^ { G } } { \operatorname* { m a x } } \| x - y \| _ { 2 } , } \\ { \underset { y \in S _ { i } ^ { G } } { \operatorname* { m a x } } \underset { x \in S _ { i } ^ { A } } { \operatorname* { m i n } } \| x - y \| _ { 2 } \bigg \} . } \end{array}\tag{5}
$$

Lower values indicate smaller worst-case geometric deviations from the reference motion.

3.1.2.3. Mean Joint Distance Mean Joint Distance (MJD) [RWH<sup>∗</sup>26] measures pose-space error between generated and reference motion using corresponding root-aligned body and hand joints. Let $p _ { c t } ^ { j }$ and $\hat { p } _ { c m t } ^ { j }$ denote the reference and generated positions of joint j at frame t. MJD is defined as

$$
\mathrm { M J D } _ { c m } = \frac { 1 } { T J } \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { J } \left. p _ { c t } ^ { j } - \hat { p } _ { c m t } ^ { j } \right. _ { 2 } ,\tag{6}
$$

where T and J denote the numbers of aligned frames and evaluated joints, respectively. Lower values indicate closer agreement with the reference pose sequence.

3.1.2.4. Dice Coefficient The Dice Coefficient [TCMU24] measures spatial overlap between generated and reference gesture trajectories after projection onto a binary occupancy map in the frontal xy plane. Let A and B denote the occupied regions of the reference and generated motions, respectively. It is defined as

$$
\operatorname { D i c e } ( A , B ) = { \frac { 2 | A \cap B | } { | A | + | B | } } .\tag{7}
$$

The score lies in [0, 1], with higher values indicating greater spatial overlap. Dice captures the overall motion footprint but discards depth information and temporal ordering.

## 3.1.3. Motion Quality and Physical Plausibility Metrics

3.1.3.1. Foot Contact Foot Contact measures lower-body plausibility by comparing stable ground-contact rates between generated and reference motion. A foot is considered in contact when at least two of the heel, big-toe, and small-toe vertices are within 3 cm of the estimated floor and have horizontal velocity below 0.10 m/s. Let $r _ { \mathrm { G T } }$ and $r _ { \mathrm { G E N } }$ denote the proportions of frames with at least one foot in contact. The Foot Contact error is

$$
\mathrm { F C } = \left| { r _ { \mathrm { G T } } - r _ { \mathrm { G E N } } } \right| .\tag{8}
$$

Lower values indicate closer agreement with reference contact behavior. The metric primarily reflects lower-body stability and artifacts such as implausible contact or foot sliding, rather than speechrelated gesture quality.

3.1.3.2. Log Dimensionless Jerk (LDLJ) Log Dimensionless Jerk (LDLJ) [BMCRBB15] measures motion smoothness from the third derivative of position while normalizing for movement duration and amplitude. For a trajectory $p ( t )$ with duration T and path length $L ,$ it is defined as

$$
\mathrm { L D L J } ( p ) = - \ln \left( \frac { T ^ { 5 } } { L ^ { 2 } } \int _ { 0 } ^ { T } \left( \frac { d ^ { 3 } p ( t ) } { d t ^ { 3 } } \right) ^ { 2 } d t \right) .\tag{9}
$$

Higher LDLJ indicates smoother motion. To compare generated and reference sequences, we use the relative score $\scriptstyle \mathrm { L D L J _ { R E L } } =$ $\mathrm { L D L J _ { G T } - L D L J _ { G E N } }$ , where values closer to zero indicate more similar smoothness and positive values indicate smoother reference motion.

## 3.1.4. Speech-Motion Alignment Metrics

3.1.4.1. Beat Consistency Beat Consistency (BC) [LYRK21] measures temporal synchronization between kinematic motion

beats and acoustic beats in the accompanying speech. Let $B ^ { x } = \{ t _ { i } ^ { x } \}$ denote detected gesture-beat timestamps and $\boldsymbol { B ^ { y } } = \{ t _ { i } ^ { y } \}$ detected audio-beat timestamps. BC is computed as

$$
\mathbf { B C } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \exp \left( - \frac { \operatorname* { m i n } _ { t _ { j } ^ { y } \in B ^ { y } } \| t _ { i } ^ { x } - t _ { j } ^ { y } \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,\tag{10}
$$

where σ controls the temporal tolerance around each acoustic beat. Higher values indicate stronger temporal correspondence between gesture and speech beats. BC is therefore prosody-aware but semantics-agnostic: it evaluates when a gesture occurs, but not whether the gesture expresses information consistent with the spoken content.

## 3.1.5. Semantics-Aware Metrics

3.1.5.1. SRGR Semantic Relevant Gesture Recall (SRGR) [LZI<sup>∗</sup>22] extends PCK by weighting pose correctness according to semantic relevance. Let $p _ { t } ^ { j }$ and $\hat { p } _ { t } ^ { j }$ denote the reference and generated positions of joint j at frame t, respectively. SRGR is defined as

$$
D _ { \mathrm { S R G R } } = \lambda \frac { 1 } { T J } \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { J } \mathbf { 1 } \left[ \Vert p _ { t } ^ { j } - \hat { p } _ { t } ^ { j } \Vert _ { 2 } < \delta \right] ,\tag{11}
$$

where $\lambda$ is the semantic relevance weight and δ the distance threshold. Higher values indicate greater pose agreement at semantically important intervals. As a reference-pose-based metric, SRGR may penalize alternative but semantically valid gesture realizations.

3.1.5.2. Semantic Score Semantic Score (SC) [AZL23] measures speech–gesture semantic alignment in a learned cross-modal embedding space by comparing transcript and generated-motion embeddings with cosine similarity. Let e<sub>text</sub> and $\mathbf { e } _ { \mathrm { m o t i o n } }$ denote the corresponding embeddings. SC is defined as

$$
\mathbf { S C } = \frac { \mathbf { e } _ { \mathrm { t e x t } } ^ { \top } \mathbf { e } _ { \mathrm { m o t i o n } } } { \| \mathbf { e } _ { \mathrm { t e x t } } \| _ { 2 } \| \mathbf { e } _ { \mathrm { m o t i o n } } \| _ { 2 } } .\tag{12}
$$

Higher values indicate stronger semantic alignment. Unlike SRGR, SC does not require frame-level semantic annotations, but depends on the quality of the learned gesture–text embedding space.

3.1.5.3. Semantic Gesture Preservation (SGP) We introduce Semantic Gesture Preservation (SGP), a new semantics-aware metric designed to evaluate whether generated motion preserves semantic gesturing during speech intervals annotated as semantically meaningful; see details in 3.2. In contrast to reference-pose metrics such as SRGR, SGP does not require the generated gesture to reproduce the exact physical realization observed in the ground truth. This is important for co-speech gestures, where the same communicative meaning can be expressed through multiple valid motion forms.

SGP operates on semantic gesture events annotated in BEAT2 and uses Gemini 2.5 Pro [CBS<sup>∗</sup>25] as a multimodal gesture classifier. For each annotated semantic event, the corresponding generated video is extracted with temporal context and presented to Gemini together with the speech and annotation metadata. The model is instructed to determine the visibly observed gesture rather than assume that a semantic gesture is present, and classifies its type as iconic, metaphoric, deictic, beat, none, or uncertain. This yields a scalable event-level assessment of whether intervals expected to contain semantic gestures instead collapse into generic beat-like motion.

Let $E _ { c }$ denote the set of annotated semantic gesture events in clip $c ,$ and let $\hat { \tau } _ { c m e }$ denote the gesture type identified by Gemini for event e from model m. An event is considered semantically degraded when $\hat { \tau } _ { c m e } =$ beat. The clip-level Semantic Gesture Preservation score is then defined as

$$
\mathrm { S G P } _ { c m } = 1 - \frac { n \left( \hat { \tau } _ { c m e } = \mathrm { b e a t } \right) } { n ( E _ { c } ) } ,\tag{13}
$$

where $n ( E _ { c } )$ is the number of valid semantic events in the reference clip and $n ( \hat { \tau } _ { c m e } = \mathrm { \mathbf { b } e a t } )$ counts how many of these intervals are classified as beat gestures in the generated motion. Higher values indicate that semantic-gesture intervals are less frequently replaced by generic beat gestures, with SGP = 1 indicating that none of the evaluated semantic events are classified as beats.

The model-level SGP is obtained by averaging the clip-level scores:

$$
\mathrm { S G P } _ { m } = \frac { 1 } { C _ { m } } \sum _ { c = 1 } ^ { C _ { m } } \mathrm { S G P } _ { c m } ,\tag{14}
$$

where $C _ { m }$ denotes the number of evaluated clips for model $m .$ Each clip therefore contributes equally regardless of its number of annotated semantic events.

SGP should be interpreted specifically as a measure of semantic gesture preservation against beat-gesture collapse. It does not aim to verify that the generated gesture reproduces the exact annotated meaning or specific semantic gesture type. Instead, it provides a complementary semantic metric that shows how far semantic gestures in the reference motion are preserved in the generated motion.

## 3.2. LLM-Based Semantic Gesture Annotation

The existing BEAT2 annotations identify semantically relevant gesture intervals and provide coarse gesture categories, but they contain limited information about how a gesture is physically realized or what communicative meaning it expresses in context. Such information is important for semantic evaluation because semantically equivalent gestures may differ substantially in their exact pose trajectories. We therefore enrich the existing intervals with structured multimodal annotations using Gemini 2.5 Pro [CBS 25]. For each predefined semantic interval $e ,$ we extract the corresponding rendered motion segment and pair it with the annotated target word or expression. Gemini is instructed to analyze only the target interval and to ground its output in the visible motion. The annotation decomposes semantic gesture information into three complementary components: (i) gesture form, describing handedness, hand shape, orientation, spatial location, and movement trajectory; (ii) contextual meaning, describing the communicative interpretation of the observed gesture in relation to the accompanying expression; and (iii) gesture type, selected from {iconic, metaphoric, deictic, beat}. The model additionally reports a confidence value. We represent the resulting annotation as

$$
\begin{array} { r } { \mathcal { A } _ { e } = \left( w _ { e } , d _ { e } , m _ { e } , \tau _ { e } , c _ { e } \right) , } \end{array}\tag{15}
$$

where $w _ { e }$ denotes the target word or expression, $d _ { e }$ the fine-grained description of observable gesture form, $m _ { e }$ the inferred contextual meaning, $\tau _ { e }$ the gesture type, and $c _ { e }$ the annotation confidence. For example, for an interval associated with the word had, the annotator describes a small downward right-hand motion in front of the torso, interprets the motion as conveying possession, and assigns the gesture to the iconic category. This structured representation separates what is visibly performed from how that motion is interpreted, which is useful when multiple physical realizations can convey similar communicative content. The complete annotation prompt and additional examples are provided in Appendix 8.1. To assess annotation reliability, a subset of the generated annotations is manually evaluated along the same information dimensions used downstream: gesture-type agreement, contextualmeaning correctness, and the quality of the gesture-form description (Appendix 8.3). This validation does not treat Gemini outputs as semantic ground truth; rather, it assesses whether the generated annotations are sufficiently consistent and interpretable to support subsequent semantics-aware analysis. The enriched annotations are subsequently used in the Semantic Gesture Preservation metric described in Section 3.1.5.

## 3.3. Benchmark Models

This subsection summarizes the representative holistic co-speech gesture generation models selected for our benchmark, covering diverse methodological paradigms within a unified full-body motion space. Architectural details are provided in the respective original papers.

## 3.3.1. EMAGE

EMAGE [LZB<sup>∗</sup>24] proposes a unified framework for holistic cospeech gesture generation, jointly synthesizing facial expressions, upper- and lower-body motion, hand gestures, and global translation. The model uses a Masked Audio Gesture Transformer to jointly optimize masked gesture reconstruction and audioconditioned generation, allowing partially observed motion to provide structured body hints during synthesis. To capture the heterogeneous dynamics of different body regions, EMAGE employs four compositional VQ-VAEs that separately model the face, upper body, hands, and lower body in discrete latent spaces. It further introduces Content Rhythm Attention (CRA), which adaptively combines rhythmic speech cues with transcript semantics to balance beat-aligned and content-related gesture generation. The reconstructed latent features are decoded into local facial and body motion, while a separate global motion predictor estimates body translation. This design enables coherent, audio-synchronized holistic gesture generation while supporting partially predefined spatialtemporal gesture constraints.

## 3.3.2. GestureLSM

GestureLSM [LSH<sup>∗</sup>25] introduces a flow-matching-based framework for real-time holistic co-speech gesture generation, with emphasis on modeling interactions across body regions. Rather than treating the face, hands, upper body, and lower body as independent streams, it represents them with separate residual vector-quantized latent tokens and captures their dependencies through sequential spatial and temporal attention. Spatial attention models inter-region coordination within each frame, while temporal attention captures motion evolution over time. Speech conditioning combines lowlevel acoustic cues with high-level semantic transcript features and is fused with gesture representations through cross-attention. To improve sampling efficiency, GestureLSM further introduces a latent shortcut model for learning direct transitions along the flow trajectory, together with beta-distributed timestep sampling. This design substantially reduces inference steps while maintaining highquality, speech-coordinated holistic motion generation.

## 3.3.3. SemTalk

SemTalk [ZLZ<sup>∗</sup>25] generates holistic co-speech motion by separating rhythm-related base motion from sparse semantic motion and adaptively integrating them at the frame level. A hierarchical coarse-to-fine cross-attention module propagates speech-driven motion cues from the face to the hands, upper body, and lower body, while local and global rhythm objectives encourage temporal synchronization and coherence. To model infrequent but semantically salient gestures, SemTalk introduces semantic emphasis learning that combines local text features, sentence-level CLIP representations, emotional cues, and high-level speech features to estimate frame-level semantic scores. These scores selectively activate semantic motion and control its fusion with the rhythm-aligned base motion. Separate RVQ-VAE representations are used for different body regions to preserve region-specific dynamics and support expressive holistic synthesis.

## 3.3.4. SemConFlow

SemConFlow [LGÖY26] introduces a semantically grounded framework for holistic co-speech gesture generation by combining multimodal semantic alignment with contrastive flow matching. It first learns region-specific motion priors for the face, hands, upper body, and lower body using hierarchical RVQ-VAEs and aggregates them into a composite full-body latent space. A Semantics-Aware Composite Module aligns this representation with textual and acoustic features through cosine and contrastive objectives, encouraging consistent semantics across body regions. SemCon-Flow then applies contrastive flow matching, using mismatched audio–text conditions as negative contexts so that the learned velocity field favors semantically congruent speech–motion trajectories while diverging from incongruent alternatives. Temporal crossattention and self-attention further integrate multimodal conditioning and maintain motion coherence. Experiments on BEAT2 and SHOW demonstrate strong performance in motion realism, diversity, synchronization, and semantic consistency.

![](images/ec208d0009c420d5e5db9b7e91ca10108c77c9725129027576f322069d53b91a.jpg)  
Figure 1: Perceptual study interfaces for the audio (left) and muted (right) conditions, evaluating speech-aware and visual-quality dimensions, respectively. Participants complete one condition using the same seven-point Likert scale and video interface.

## 3.4. Perception Study Design

We conduct a perceptual study to obtain human judgments of holistic co-speech gesture quality and use these ratings as targets for validating the objective metrics. Following the general separation adopted in GENEA-style evaluation [NVHM<sup>∗</sup>26], we distinguish visual motion quality from speech-dependent appropriateness through two independent conditions. Unlike GENEA, however, our perceptual criteria are selected to align with the multidimensional evaluation objectives of this benchmark. The muted condition assesses human-likeness, motion diversity, and absence of animation errors, capturing complementary aspects of visual motion quality. The audio condition evaluates speech timing and content match, explicitly separating temporal coordination from semantic correspondence. This distinction enables a more finegrained analysis of how different objective metrics relate to specific perceptual properties of generated gestures. All criteria are rated on a seven-point Likert scale, with higher scores indicating better perceived quality. The two evaluation interfaces are shown in Figure 1.

The study compares EMAGE, SemConFlow, SemTalk, and GestureLSM together with BEAT2 motion-capture ground truth. All conditions use the same avatar, camera, crop, and rendering pipeline. Each stimulus is a 10-second clip. As we are specifically interested in evaluating semantically rich gesture clips, BEAT2 annotations are used to select windows with the highest occurrence of iconic, metaphoric, and deictic gestures. The same temporal interval is used for all models and the corresponding ground-truth motion. The selected stimuli contain 80 annotated semantic events: 44 iconic, 18 metaphoric, and 18 deictic gestures. A total of 103 participants were recruited online with compensation: 51 for the muted condition and 52 for the audio condition. After quality control, two participants were excluded from the muted study, resulting in 101 valid participants in total (49 muted and 52 audio).

Each study evaluates 15 unique sequences from five motion sources (four models and ground truth), yielding 75 unique source sequence pairs for analysis.Videos must be watched in full before ratings can be submitted, and participants failing two or more attention checks are excluded. For each model sequence pair, ratings are summarized by their mean and standard deviation, with the mean used as the subjective target in the objective–subjective correlation analysis. This yields 75 observations across all five motion sources and 60 generated-motion observations after excluding BEAT2 ground truth. We further compute pairwise Spearman correlations among the five subjective dimensions to assess the extent to which the selected perceptual criteria provide complementary judgments (Section 4.4.2). Further details of the user-study procedure and interface are provided in Appendix 7.1.

## 4. Experiments

## 4.1. Datasets

BEAT2 [LZB 24] contains approximately 60 hours of synchronized speech and holistic motion from 25 speakers (12 female and 13 male). The dataset is divided into BEAT2-Standard (27 hours) and BEAT2-Additional (33 hours), with the latter capturing more spontaneous and natural gestural behavior. Full-body motion is represented using SMPL-X [PCG<sup>∗</sup>19] and further refined with MoSh++ [MGT<sup>∗</sup>19], providing detailed body, hand, facial, and global motion aligned with speech. Compared with the original BEAT dataset, BEAT2 provides a more accurate mesh-based representation with improved body proportions, hand articulation, and facial motion, making it well suited for holistic co-speech gesture generation and evaluation. In our experiments, we use the BEAT2- Standard subset and adopt the official 85%/7.5%/7.5% train/validation/test split over all 25 speakers.

## 4.2. Implementation Detail

All benchmark experiments are conducted on a single NVIDIA A100 GPU under a unified evaluation setup. To ensure a consistent comparison, all evaluated methods are retrained using their official implementations and original optimization settings, while sharing the full BEAT2 training set as the common training-data regime. This extends the original single-speaker training setup used by EMAGE [LZB<sup>∗</sup>24], SemTalk [ZLZ<sup>∗</sup>25], and GestureLSM [LSH<sup>∗</sup>25] to 60 hours of multi-speaker training data. During inference, all models are evaluated on the same BEAT2 test sequences with identical speech inputs and sequence boundaries. Their outputs are converted to a common SMPL-X representation and processed through the same post-processing, rendering, and metriccomputation pipeline, reducing variation introduced by differences in implementation or evaluation protocol.

## 4.3. Objective Evaluation

Table 1 reveals pronounced differences across evaluation dimensions. SemConFlow achieves the strongest performance on most distributional, geometric, and speech-alignment metrics, including FGD, BC, Diversity, Dice, SRGR, PCK, MJD, Chamfer, Hausdorff, and SC. Its substantially lower FGD together with higher Diversity suggests a closer match to the reference motion distribution without sacrificing motion variability, while the improvements in MJD, Chamfer, and Hausdorff indicate stronger geometric agreement with the reference motion. SemConFlow also obtains the highest BC and SC, suggesting stronger temporal coordination with speech and higher cross-modal semantic similarity in the learned gesture–text embedding space.

No single method, however, dominates across all metric families.

EMAGE achieves the best Density and Coverage and the relative LDLJ value closest to zero, indicating stronger support within the reference feature distribution and smoothness closer to the captured motion. Notably, EMAGE also obtains the highest SGP despite not achieving the highest SC. This discrepancy highlights an important distinction between the two semantics-aware measures: SC evaluates global speech–motion similarity in a learned embedding space, whereas SGP targets whether annotated semantic-gesture intervals are preserved rather than collapsing into generic beat-like motion. SemTalk performs best on Foot Contact, indicating more consistent lower-body contact behavior, while GestureLSM does not lead on an individual metric.

Overall, the varying model rankings across distributional, geometric, kinematic, temporal, and semantic metrics show that these measures capture complementary aspects of gesture quality. While SC and SGP scores overall indicates EMAGE and SemConFlow as best, overall rankings are different suggesting more analysis in the semantics-aware metrics direction. These results therefore motivate a multidimensional evaluation protocol and, more importantly, the subsequent analysis of which objective signals are actually aligned with human perception.

## 4.4. Subjective Evaluation

## 4.4.1. Main Analysis.

The final subjective dataset contains ratings from 49 participants in the muted condition and 52 participants in the audio condition. Participants reported English fluency in both studies. The muted group had a mean age of 37.5 years (SD = 10.9; 24 male, 25 female), while the audio group had a mean age of 38.9 years (SD = 11.9; 29 male, 23 female). For each motion source, we first average ratings at the sequence level and then aggregate over the 15 unique sequences. Figure 2 summarizes the resulting model means together with between-sequence variation on the common seven-point scale.

BEAT2 reference motion receives the highest mean rating across all five perceptual dimensions, with particularly large margins for human-likeness, absence of animation errors, speech timing, and content match. The gap is notably smaller for motion diversity, where the reference score is closer to the strongest generated result. This suggests that, under the predefined perceptual criterion of motion diversity used in our study, current systems can approach the reference more closely in perceived variation of motion, while a larger gap remains in overall human-likeness, animation quality, and speech-dependent appropriateness.

Among the generated systems, SemConFlow achieves the highest mean score on all five dimensions. In the muted condition, SemTalk generally ranks second, followed by EMAGE and GestureLSM, with SemTalk and EMAGE remaining relatively close on motion diversity. In the audio condition, EMAGE ranks second for both speech timing and content match, followed by SemTalk, while GestureLSM consistently receives the lowest mean ratings. The change in ordering between muted and audio evaluations indicates that visual motion quality and speech-conditioned appropriateness are not interchangeable perceptual properties.

Overall, the subjective results indicate that motion-centric and speech-aware criteria capture related but non-identical aspects of perceived gesture quality. Models do not preserve the same relative behavior across visual and speech-conditioned judgments, motivating an explicit analysis of how strongly the five perceptual dimensions covary before using them as separate targets for objectivemetric validation.

![](images/db0e84428c7c7b137b88108630e4898d04ac30cf00e20ecb2154cfd3a2237b0f.jpg)  
Figure 2: Subjective ratings across five perceptual dimensions. Points show mean ratings with standard-deviation error bars; BEAT2 serves as the reference, and the remaining methods are generated results.

## 4.4.2. Relationship among perceptual dimensions.

To examine whether the five perceptual criteria capture distinct or shared aspects of gesture quality, we compute pairwise Spearman correlations between sequence-level subjective ratings. Table 2 reports the associations for the 60 generated model–sequence observations, with the 15 BEAT2 reference sequences included as a complementary comparison. Among generated motions, all dimensions are positively correlated $( \rho = 0 . 5 9 5  – 0 . 9 7 4 )$ . Human-likeness is strongly associated with absence of animation errors $( \rho = 0 . 9 2 8 )$ while motion diversity shows moderate-to-strong correlations with the other dimensions $( \rho = 0 . 6 2 9  – 0 . 7 7 0 )$ . Cross-condition correlations are also substantial, including human-likeness with speech timing $( \rho = 0 . 6 5 0 )$ and content match $( \rho = 0 . 6 2 4 )$ . The strongest association is between speech timing and content match $( \rho =$ 0.974), indicating that temporal and semantic appropriateness are closely coupled in the evaluated stimuli. Overall, the results reveal shared perceptual aspects without making the five criteria interchangeable, supporting a multidimensional evaluation while highlighting the particularly close relationship between speech timing and content match.

## 4.5. Objective–Subjective Correlation Analysis

Figure 3 shows a strongly dimension-dependent relationship between objective metrics and human judgments. No single metric consistently explains all five perceptual targets, indicating that objective measures capture distinct aspects of holistic gesture quality. SC is included in the analysis but does not exhibit a statistically meaningful correlation with any of the five subjective dimensions. For the muted dimensions, $\mathrm { L D L J } _ { R E L }$ correlates most strongly with human-likeness $( \rho = - 0 . 6 1 )$ and absence of animation errors $( \rho = - 0 . 6 2 )$ , while Coverage and Density are most informative for motion diversity. MJD and Hausdorff show little correspondence with subjective ratings, suggesting limited value of direct geometric distances for perceptual assessment.

<table><tr><td>Method</td><td>FGD↓</td><td>BC↑</td><td>Div.↑</td><td>Dens.↑</td><td>Cov.↑</td><td>Dice↑</td><td> $\mathrm { L D L J _ { R E L } }$  →0 SRGR↑</td><td>PCK↑</td><td>MJD↓</td><td>Chamfer↓ Hausdorff↓ Foot Contact↓</td><td></td><td></td><td>SC↑ SGP↑</td></tr><tr><td>EMAGE</td><td>5.523</td><td>0.692</td><td>88</td><td>0.8972</td><td>0.07275</td><td>0.6407</td><td>0.0530</td><td>0.05494 0.3309</td><td>0.2410</td><td>0.01375</td><td>0.2908</td><td>0.5048</td><td>0.248 0.4372</td></tr><tr><td>SemConFlow</td><td>2.245</td><td>0.780</td><td>120</td><td>0.01559</td><td>0.06647</td><td>0.6793</td><td>-0.4294</td><td>0.05868 0.3552 0.2007</td><td></td><td>0.01112</td><td>0.2530</td><td>0.5932</td><td>0.314 0.4205</td></tr><tr><td>SemTalk</td><td>4.266</td><td>0.727</td><td>116</td><td>0.1123</td><td>0.06673</td><td>0.6241</td><td>-0.6020</td><td>0.05059 0.3037 0.2303</td><td></td><td>0.13976</td><td>0.4663</td><td>0.4040</td><td>0.268 0.4014</td></tr><tr><td>GestureLSM</td><td>4.268</td><td>0.525</td><td>112</td><td>0.009239</td><td>0.009817 0.6573</td><td></td><td>1.5386</td><td>0.045360.27150.2391</td><td></td><td>0.01797</td><td>0.2909</td><td>0.8883</td><td>0.248 0.2622</td></tr></table>

Table 1: Objective comparison of the benchmarked holistic co-speech gesture generation methods on BEAT2. All metrics are computed over the 265 shared test sequences, except SGP, which is evaluated on the 15 sequences used in the perceptual study with Gemini-based semantic annotations. Arrows indicate the preferred direction, and bold denotes the best result for each metric.

Table 2: Pairwise Spearman correlations among subjective perceptual dimensions.
<table><tr><td>Perceptual dimension pair</td><td>Generated (n = 60)</td><td>Including GT (n = 75)</td></tr><tr><td>Human-likeness – Motion diversity</td><td>0.728</td><td>0.626</td></tr><tr><td>Human-likeness – Absence of animation errors</td><td>0.928</td><td>0.960</td></tr><tr><td>Human-likeness – Speech timing</td><td>0.650</td><td>0.813</td></tr><tr><td>Human-likeness – Content match</td><td>0.624</td><td>0.801</td></tr><tr><td>Motion diversity – Absence of animation errors</td><td>0.770</td><td>0.647</td></tr><tr><td>Motion diversity – Speech timing</td><td>0.665</td><td>0.564</td></tr><tr><td>Motion diversity – Content match</td><td>0.629</td><td>0.538</td></tr><tr><td>Absence of animation errors – Speech timing</td><td>0.620</td><td>0.795</td></tr><tr><td>Absence of animation errors – Content match</td><td>0.595</td><td>0.783</td></tr><tr><td>Speech timing – Content match</td><td>0.974</td><td>0.985</td></tr></table>

For the speech-dependent dimensions, Coverage reaches $\rho =$ 0.73 for speech timing and $\rho = 0 . 7 0$ for content match, with Density and FGD showing similarly strong associations. Although these metrics do not explicitly model semantics, their correlations indicate that motion-distribution quality can co-vary with speech appropriateness. BC, by contrast, is negatively correlated with all five perceptual dimensions, showing that stronger beat alignment alone does not imply better perceived quality.

The semantics-aware metrics provide an especially informative comparison. SC explicitly measures speech–motion similarity in a learned semantic embedding space, yet shows no statistically meaningful correlation with any of the five subjective dimensions, including speech timing and content match. In contrast, SGP exhibits the intended selectivity toward speech-aware judgments, correlating with both speech timing and content match at $\rho = 0 . 3 9$ while remaining weak or non-significant for the muted dimensions. SGP also shows stronger alignment with human judgments on speech-aware dimensions than SRGR, while remaining comparable to SRGR on the other perceptual dimensions. Although its absolute correlations are lower than those of Coverage, Density, and FGD, SGP provides a more targeted and interpretable semantic signal by explicitly measuring whether semantic gesture intervals are preserved rather than reduced to generic beat-like motion. Together, the results for SC, SRGR, and SGP demonstrate that semanticsaware metrics differ substantially in their perceptual alignment and should therefore be validated directly against human judgments of communicative content. Overall, these findings motivate the subsequent composite analysis, where complementary objective signals are combined to better approximate human perception.

![](images/8599ab0aa0ff7a7c417b863c64e6df41e27fbb8e1a5465ba39a258df6cccc295.jpg)  
Figure 3: Spearman correlations between thirteen objective metrics and five subjective evaluation dimensions $( n = 6 0 )$ , with significance indicated by $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1$ , and $^ { * * * } p < 0 . 0 0 1$

## 4.6. Target-Specific Composite Metrics and Analysis

## 4.6.1. Learning Target-Specific Perceptual Composites

Despite the positive associations among the subjective dimensions, the preceding analyses show that they are not interchangeable: their pairwise correlations vary substantially, model behavior differs across perceptual targets, and individual objective metrics exhibit strongly dimension-dependent correspondence with human judgments. We therefore construct a separate composite predictor for each perceptual dimension rather than collapsing gesture quality into a single global score.

Let $\mathbf { x } _ { i } = [ x _ { i 1 } , \ldots , x _ { i K } ] ^ { \top }$ , with $K = 1 3$ , denote the objective-metric vector for model–sequence observation i, comprising FGD, BC, Density, Coverage, Dice, $\mathrm { L D L J _ { R E L } }$ , SRGR, PCK, MJD, Hausdorff, Foot Contact, SC, and SGP. Chamfer Distance is excluded from the joint composite model because of its strong redundancy with Hausdorff Distance, while SC is retained as an independent semanticsaware predictor. Because these metrics operate on different numerical scales, each predictor is standardized using statistics estimated exclusively from the training observations,

$$
z _ { i k } = \frac { x _ { i k } - \mu _ { k } } { \sigma _ { k } } ,\tag{16}
$$

where $\mu _ { k }$ and $\sigma _ { k }$ denote the training-set mean and standard deviation of metric k.

For each perceptual target $q \in \{ \mathrm { H L , M D , A E , S T , C M } \} ,$ , corresponding to human-likeness, motion diversity, absence of animation errors, speech timing, and content match, we fit an independent ordinary least-squares predictor with an intercept,

$$
C _ { q } ( \mathbf { x } _ { i } ) = \mathsf { \beta _ { 0 q } } + \mathsf { \beta _ { q } ^ { \top } } \mathbf { z } _ { i } = \mathsf { \beta _ { 0 q } } + \sum _ { k = 1 } ^ { K } \mathsf { \beta _ { k q } } z _ { i k } ,\tag{17}
$$

where $\beta _ { 0 q }$ is the intercept and $\beta _ { q }$ contains the target-specific coefficients. The regression target is the mean subjective rating for dimension $q .$ Standardization is applied only to the predictors and does not normalize the subjective targets, so $C _ { q } ( \mathbf { x } _ { i } )$ remains on the original seven-point rating scale. For ordinary least squares with an intercept, fitting standardized predictors is algebraically equivalent to fitting the same predictors in their original units after reparameterizing the coefficients. Standardization therefore changes coefficient scaling but does not dampen or otherwise alter the fitted predictions.

To avoid information leakage, all standardization statistics and regression coefficients are estimated independently within each training fold. We evaluate the composite predictors using leaveone-out cross-validation (LOOCV) over the 60 generated model– sequence observations. In each fold, the held-out observation is standardized using statistics from the remaining 59 observations and predicted by the corresponding regression model, yielding one out-of-fold prediction per observation and perceptual target.

## 4.6.2. Cross-Validated Perceptual Alignment

Table 3 evaluates how well each target-specific composite approximates its corresponding subjective dimension. Mean Squared Error (MSE) and Mean Absolute Error (MAE) quantify out-of-fold prediction error on the original seven-point scale, while Composite $\rho$ denotes the Spearman correlation between the 60 out-of-fold predictions and subjective ratings. “Best individual” identifies the single objective metric with the strongest absolute Spearman correlation for that target, and

$$
\Delta | \rho | = | \rho _ { \mathrm { c o m p o s i t e } } | - | \rho _ { \mathrm { b e s t i n d i v i d u a l } } |\tag{18}
$$

measures the gain over that metric.

Composite correlations range from 0.5375 for motion diversity to 0.7824 for content match. All five composites improve over their strongest individual metric, although the gains vary substantially across perceptual targets. The largest improvements are observed for absence of animation errors (+0.0819) and content match (+0.0805), followed by human-likeness (+0.0468). Motion diversity and speech timing show smaller gains of +0.0117 and +0.0246, respectively. These results indicate that the benefit of combining objective signals is target-dependent, with some perceptual dimensions drawing more strongly on complementary information distributed across multiple metrics.

The composite predictors should be interpreted as dimensionspecific automated proxies for human judgments rather than substitutes for direct perceptual evaluation. We therefore assess them at the sequence level through out-of-fold correlation and prediction error rather than deriving an additional model-level ranking from averaged predictions. The results show that the usefulness of metric combination is strongly target-dependent. The largest improvements occur for absence of animation errors and content match, suggesting that these perceptual judgments benefit from complementary signals across multiple objective measures. Human-likeness also shows a moderate improvement, whereas the gains for motion diversity and speech timing are comparatively small. Thus, no single pattern of metric combination applies uniformly across perceptual dimensions.

Table 3: LOOCV performance of target-specific composite predictors over 60 generated model–sequence observations using 13 predictors. MSE and MAE report out-of-fold error on the seven-point subjective scale, while composite ρ measures Spearman correlation with subjective ratings. Best individual is the strongest single objective metric by absolute Spearman correlation, and $\Delta | \rho |$ denotes the composite gain.
<table><tr><td>Target Composite Metric</td><td>MSE↓</td><td>MAE↓</td><td>Composite ρ↑</td><td>Best individual</td><td>∆|ρ| ↑</td></tr><tr><td>Human-likeness</td><td>0.2846</td><td>0.4140</td><td>0.6568</td><td>LDLJREL</td><td>+0.0468</td></tr><tr><td>Motion diversity</td><td>0.2035</td><td>0.3610</td><td>0.5375</td><td>Coverage</td><td>+0.0117</td></tr><tr><td>Absence of animation errors</td><td>0.1560</td><td>0.2984</td><td>0.7065</td><td>LDLJREL</td><td>+0.0819</td></tr><tr><td>Speech timing</td><td>0.2671</td><td>0.4349</td><td>0.7439</td><td>Coverage</td><td>+0.0246</td></tr><tr><td>Content match</td><td>0.3665</td><td>0.4844</td><td>0.7824</td><td>Coverage</td><td>+0.0805</td></tr></table>

## 5. Conclusions

We present a perceptually grounded and semantics-aware benchmark for holistic co-speech gesture generation, combining standardized model comparison, human-centered metric validation, and fine-grained semantic evaluation. By retraining recent methods under a shared BEAT2 protocol and using a unified rendering and evaluation pipeline, we enable more controlled and directly comparable assessment across systems.

Our results show that gesture quality is strongly multidimensional. Model rankings vary across distributional, geometric, kinematic, temporal, and semantic measures, while no single objective metric consistently reflects all perceptual dimensions. Targetspecific composite metrics improve alignment with human judgments across all five dimensions, with the largest gains observed for absence of animation errors and content match. These findings show that complementary objective signals can provide additional perceptual information beyond the strongest individual metric, although the benefit of combination remains target-dependent.

For semantic evaluation, we enrich BEAT2 annotations with fine-grained gesture descriptions using Gemini 2.5 Pro and derive Semantic Gesture Preservation (SGP). Unlike SC, which shows no significant association with the evaluated perceptual dimensions, SGP is selectively aligned with speech-aware judgments and captures whether semantic gesture intervals are preserved rather than reduced to generic beat-like motion. Together, these findings provide a more systematic basis for evaluating progress in holistic cospeech gesture generation.

## 6. Limitations and Discussion

Although our benchmark reduces several sources of experimental variation, the evaluation remains constrained by the structure of the available data. The perceptual analysis is based on 15 unique speech sequences evaluated across four generation systems, yielding 60 generated model sequence observations rather than 60 fully independent samples. In addition, the selected study windows were deliberately enriched for semantic gestures, which improves the sensitivity of semantic evaluation but may overrepresent semantically dense speech relative to the broader BEAT2 distribution.

The semantic analysis introduces a further limitation. Gemini 2.5 Pro provides a scalable way to enrich coarse BEAT annotations with gesture descriptions, contextual meaning, and gesture type, but its outputs should not be treated as semantic ground truth. Likewise, SGP specifically measures whether annotated semantic intervals avoid collapsing into generic beat gestures; it does not directly verify that the generated motion preserves the exact intended meaning or gesture category. More broadly, the observed objective– subjective relationships are specific to the evaluated models, sequences, and metric implementations. Future work should therefore extend the benchmark to larger and more diverse datasets, models, and perceptual studies, and develop semantic metrics that can more directly assess meaning preservation while retaining the scalability of multimodal-model-based evaluation.

## References

[AZL23] AO T., ZHANG Z., LIU L.: Gesturediffuclip: Gesture diffusion model with clip latents. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1–18. 2, 3, 5

[BMCRBB15] BALASUBRAMANIAN S., MELENDEZ-CALDERON A., ROBY-BRAMI A., BURDET E.: On the analysis of movement smoothness. Journal of NeuroEngineering and Rehabilitation 12 (12 2015). doi:10.1186/s12984-015-0090-9. 4

[CBS<sup>∗</sup>25] COMANICI G., BIEBER E., SCHAEKERMANN M., PASUPAT I., SACHDEVA N., DHILLON I., BLISTEIN M., RAM O., ZHANG D., ROSEN E., ET AL.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025). 3, 5

[CGM<sup>∗</sup>20] CARROZZINO M., GALDIERI R., MACHIDON O., BERGA-MASCO M., POTEL M.: Do virtual humans dream of digital sheep? IEEE computer graphics and applications 40 (07 2020), 71–83. doi: 10.1109/MCG.2020.2993345. 1

[CLW<sup>∗</sup>24] CHEN J., LIU Y., WANG J., ZENG A., LI Y., CHEN Q.: Diffsheg: A diffusion-based approach for real-time speech-driven holistic 3d expression and gesture generation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024), IEEE, pp. 7352–7361. 2

[DSR<sup>∗</sup>25] DELBOSC A., SABOURET N., RAVENET B., AYACHE S., OCHS M.: Automatic objective metric for the optimization of nonverbal behavior generative models. In Proceedings of the 25th ACM International Conference on Intelligent Virtual Agents (New York, NY, USA, 2025), IVA ’25, Association for Computing Machinery. URL: https://doi.org/10.1145/3717511.3749302, doi: 10.1145/3717511.3749302. 2, 3

[GKOF25] GHALEB E., KHAERTDINOV B., OZYUREK A., FERNÁN-DEZ R.: I see what you mean: Co-speech gestures for reference resolution in multimodal dialogue. In Findings of the Association for Computational Linguistics: ACL 2025 (2025), pp. 13191–13206. 1

[GWK26] GHALEB E., WONG H. M., KOBROCK K.: Investigating

multimodal informativity under different partner visibility conditions in video-mediated dialogue. arXiv preprint arXiv:2608.08915 (2026). 1

[HAHH<sup>∗</sup>26] HU Y., ASKARI-HEMMAT R., HALL M., DINAN E., ZETTLEMOYER L., GHAZVININEJAD M.: Multimodal rewardbench 2: Evaluating omni reward models for interleaved text and image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2026), pp. 36904–36915. 3

[HL19] HOLLER J., LEVINSON S. C.: Multimodal language processing in human communication. Trends in Cognitive Sciences 23, 8 (2019), 639–652. URL: https://www.sciencedirect.com/ science/article/pii/S1364661319301299, doi:https: //doi.org/10.1016/j.tics.2019.05.006. 1

[HM12] HUANG C.-M., MUTLU B.: Robot behavior toolkit: Generating effective social behaviors for robots. In 2012 7th ACM/IEEE International Conference on Human-Robot Interaction (HRI) (2012), pp. 25–32. 1

[HPY25] HAQUE K. I., PAVLOU A., YUMAK Z.: “wild west” of evaluating speech-driven 3d facial animation synthesis: A benchmark study. Computer Graphics Forum 44, 2 (2025), e70073. URL: https://onlinelibrary.wiley.com/doi/abs/ 10.1111/cgf.70073, arXiv:https://onlinelibrary. wiley.com/doi/pdf/10.1111/cgf.70073, doi:https: //doi.org/10.1111/cgf.70073. 2

[HXM<sup>∗</sup>21] HABIBIE I., XU W., MEHTA D., LIU L., SEIDEL H.-P., PONS-MOLL G., ELGHARIB M., THEOBALT C.: Learning speechdriven 3d conversational gestures from video. In Proceedings of the 21st ACM international conference on intelligent virtual agents (2021), pp. 101–108. 1, 2

[KWY<sup>∗</sup>24] KUCHERENKO T., WOLFERT P., YOON Y., VIEGAS C., NIKOLOV T., TSAKOV M., HENTER G. E.: Evaluating gesture generation in a large-scale open challenge: The genea challenge 2022. ACM Transactions on Graphics 43, 3 (2024). URL: http://dx.doi. org/10.1145/3656374, doi:10.1145/3656374. 3

[LGOY25] LIU L., GHALEB E., OZYUREK A., YUMAK Z.: Semges: Semantics-aware co-speech gesture generation using semantic coherence and relevance learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (2025), pp. 13963–13973. 2, 3

[LGÖY26] LIU L., GHALEB E., ÖZYÜREK A., YUMAK Z.: Semconflow: Semantic grounding of holistic co-speech gesture generation with contrastive flow-matching. In European Conference on Computer Vision (2026), Springer, pp. 528–545. 2, 6

[LKK<sup>∗</sup>22] LEE D., KIM C., KIM S., CHO M., HAN W.-S.: Autoregressive image generation using residual quantization. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR) (2022), IEEE, pp. 11513–11522. 2

[LSH<sup>∗</sup>25] LIU P., SONG L., HUANG J., LIU H., XU C.: Gesturelsm: Latent shortcut based co-speech gesture generation with spatial-temporal modeling. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV) (2025), IEEE, pp. 10929–10939. 2, 6, 7

[LYRK21] LI R., YANG S., ROSS D. A., KANAZAWA A.: Ai choreographer: Music conditioned 3d dance generation with aist++. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV) (2021), Ieee, pp. 13381–13392. 4

[LYS<sup>∗</sup>23] LIU L., YU C., SONG S., SU Z., TAPUS A.: Human gesture recognition with a flow-based model for human robot interaction. In Companion ofthe 2023 ACM/IEEE International Conference on Human-Robot Interaction (2023), pp. 548–551. 3

[LZB<sup>∗</sup>24] LIU H., ZHU Z., BECHERINI G., PENG Y., SU M., ZHOU Y., ZHE X., IWAMOTO N., ZHENG B., BLACK M. J.: Emage: Towards unified holistic co-speech gesture generation via expressive masked audio gesture modeling. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024), IEEE, pp. 1144–1154. 1, 2, 6, 7

[LZI<sup>∗</sup>22] LIU H., ZHU Z., IWAMOTO N., PENG Y., LI Z., ZHOU Y.,

BOZKURT E., ZHENG B.: Beat: A large-scale semantic and emotional multi-modal dataset for conversational gestures synthesis. In European conference on computer vision (2022), Springer, pp. 612–630. 2, 3, 5

[MDS<sup>∗</sup>25] MUGHAL M. H., DABRAL R., SCHOLMAN M. C., DEM-BERG V., THEOBALT C.: Retrieving semantics from the deep: an rag solution for gesture synthesis. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2025), IEEE, pp. 16578– 16588. 2

[MGT<sup>∗</sup>19] MAHMOOD N., GHORBANI N., TROJE N. F., PONS-MOLL G., BLACK M. J.: Amass: Archive of motion capture as surface shapes. In Proceedings of the IEEE/CVF international conference on computer vision (2019), pp. 5442–5451. 7

[NKA<sup>∗</sup>23a] NYATSANGA S., KUCHERENKO T., AHUJA C., HENTER G. E., NEFF M.: A comprehensive review of data-driven co-speech gesture generation. In Computer Graphics Forum (2023), vol. 42, Wiley Online Library, pp. 569–596. 2

[NKA<sup>∗</sup>23b] NYATSANGA S., KUCHERENKO T., AHUJA C., HENTER G. E., NEFF M.: A comprehensive review of data-driven co-speech gesture generation. Computer Graphics Forum 42, 2 (2023), 569–596. URL: https://onlinelibrary.wiley.com/doi/abs/ 10.1111/cgf.14776, arXiv:https://onlinelibrary. wiley.com/doi/pdf/10.1111/cgf.14776, doi:https: //doi.org/10.1111/cgf.14776. 3

[NOU<sup>∗</sup>20] NAEEM M. F., OH S. J., UH Y., CHOI Y., YOO J.: Reliable fidelity and diversity metrics for generative models. In International conference on machine learning (2020), PMLR, pp. 7176–7185. 3, 4

[NVHM<sup>∗</sup>26] NAGY R., VOSS H., HOANG-MINH T., TSAKOV M., NIKOLOV T., ZHANG Z., AO T., YANG S., HUANG S., CHENG Y., ET AL.: Towards reliable human evaluations in gesture generation: Insights from a community-driven state-of-the-art benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2026), pp. 2152–2164. 2, 3, 7

[NVY<sup>∗</sup>24] NAGY R., VOSS H., YOON Y., KUCHERENKO T., NIKOLOV T., HOANG-MINH T., MCDONNELL R., KOPP S., NEFF M., HEN-TER G. E.: Towards a genea leaderboard – an extended, living benchmark for evaluating and advancing conversational motion synthesis, 2024. URL: https://arxiv.org/abs/2410.06327, arXiv: 2410.06327. 2, 3

[PCG<sup>∗</sup>19] PAVLAKOS G., CHOUTAS V., GHORBANI N., BOLKART T., OSMAN A. A., TZIONAS D., BLACK M. J.: Expressive body capture: 3d hands, face, and body from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2019), pp. 10975–10985. 7

[PLÖ<sup>∗</sup>26] PAAR F., LIU L., ÖZYÜREK A., THILL S., GHALEB E.: Duogesture: Neuro-inspired and biomechanically informed dual-stream cospeech gesture generation. arXiv preprint arXiv:2605.26236 (2026). 3

[PLXL21] PENG J., LIU D., XU S., LI H.: Generating diverse structure for image inpainting with hierarchical vq-vae. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021), IEEE, pp. 10770–10779. 2

[QLL<sup>∗</sup>24] QI X., LIU C., LI L., HOU J., XIN H., YU X.: Emotiongesture: Audio-driven diverse emotional co-speech 3d gesture generation. IEEE Transactions on Multimedia 26 (2024), 10420–10430. doi:10.1109/TMM.2024.3407692. 3

[RWH<sup>∗</sup>26] REKIK R., WUHRER S., HOYET L., ZIBREK K., OLIVIER A.-H.: Quality assessment of 3d human animation: Subjective and objective evaluation. IEEE Transactions on Visualization and Computer Graphics 32, 2 (Feb. 2026), 1780–1792. URL: http://dx.doi.org/10.1109/TVCG.2025.3631385, doi: 10.1109/tvcg.2025.3631385. 2, 3, 4

[TCMU24] TONOLI R. L., COSTA P. D. P., MARQUES L. B. D. M. M., UEDA L. H.: Gesture area coverage to assess gesture expressiveness and human-likeness. In Companion Proceedings of the 26th International Conference on Multimodal Interaction (New York, NY,

USA, 2024), ICMI ’24 Companion, Association for Computing Machinery, p. 165–169. URL: https://doi.org/10.1145/3686215. 3688822, doi:10.1145/3686215.3688822. 4

[Tho84] THOMPSON B.: Canonical correlation analysis. Sage 47 (1984). URL: https://doi.org/10.4135/9781412983570, doi:10.4135/9781412983570. 3

[TKZ<sup>∗</sup>26] TANG L., KIM G., ZHAO X., LAKE T., DING W., YIN F., SINGHAL P., WADHWA M., LIU Z., SPRAGUE Z., ET AL.: Chartmuseum: Testing visual reasoning capabilities of large vision-language models. Advances in Neural Information Processing Systems 38 (2026). 3

[vMO<sup>∗</sup>22] VAN KREVELD M., MILTZOW T., OPHELDERS T., SONKE W., VERMEULEN J. L.: Between shapes, using the hausdorff distance. Computational Geometry 100 (2022), 101817. URL: https://www.sciencedirect.com/ science/article/pii/S0925772121000730, doi:https: //doi.org/10.1016/j.comgeo.2021.101817. 4

[WRB22] WOLFERT P., ROBINSON N., BELPAEME T.: A review of evaluation practices of gesture generation in embodied conversational agents. IEEE Transactions on Human-Machine Systems 52, 3 (2022), 379–389. URL: http://dx.doi.org/10.1109/THMS.2022. 3149173, doi:10.1109/thms.2022.3149173. 3

[YCL<sup>∗</sup>20] YOON Y., CHA B., LEE J.-H., JANG M., LEE J., KIM J., LEE G.: Speech gesture generation from the trimodal context of text, audio, and speaker identity. ACM Transactions on Graphics 39, 6 (2020). URL: http://dx.doi.org/10.1145/3414685. 3417838, doi:10.1145/3414685.3417838. 3

[YLL<sup>∗</sup>23] YI H., LIANG H., LIU Y., CAO Q., WEN Y., BOLKART T., TAO D., BLACK M. J.: Generating holistic 3d human motion from speech. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023), IEEE, pp. 469–480. 2

[ZCC<sup>∗</sup>23] ZHI Y., CUN X., CHEN X., SHEN X., GUO W., HUANG S., GAO S.: Livelyspeaker: Towards semantic-aware co-speech gesture generation. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV) (2023), IEEE, pp. 20750–20760. 2

[ZGP<sup>∗</sup>23] ZHANG M., GUO X., PAN L., CAI Z., HONG F., LI H., YANG L., LIU Z.: Remodiffuse: Retrieval-augmented motion diffusion model. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV) (2023), IEEE, pp. 364–373. 2

[ZJY<sup>∗</sup>26] ZHU K., JIN Z., YUAN H., LI J., TU S., CAO P., CHEN Y., LIU K., ZHAO J.: Mmr-v: What’s left unsaid? a benchmark for multimodal deep reasoning in videos. In International Conference on Learning Representations (2026), vol. 2026, pp. 67949–67991. 3

[ZLZ<sup>∗</sup>25] ZHANG X., LI J., ZHANG J., DANG Z., REN J., BO L., TU Z.: Semtalk: Holistic co-speech motion generation with frame-level semantic emphasis. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV) (2025), IEEE, pp. 13761–13771. 2, 6, 7

## 7. Appendix

## 7.1. User Study Interface and Instructions

This section summarizes the participant-facing interfaces and instructions used in the two perceptual studies. The muted and audio conditions follow the same interaction flow but differ in the perceptual dimensions being assessed. In both studies, participants were required to watch each clip in full before submitting their ratings.

The muted condition focuses exclusively on visual motion quality. Participants first viewed the animation without audio and, once playback was completed, rated human-likeness, motion diversity, and absence of animation errors. Figure 4 illustrates the interface before and after video playback.

![](images/cd997a5893da35707527d4f885afc82d75014c50355244fc5e8e408be0eee163.jpg)  
(a) Before video playback.

![](images/68f819c91196e74451e52e93f5b4fb0059b4543fce142a090b9260b6065931a7.jpg)  
(b) After video playback.  
Figure 4: Participant interface for the muted condition. Ratings become available only after the complete stimulus has been viewed.

Before beginning the muted study, participants were given explicit definitions of the three evaluation criteria and instructed to base their judgments solely on the observed motion. The exact participant-facing instructions were as follows:

## Participant Instructions: Muted Condition

In the following pages, you will watch animations without audio and evaluate the body motion according to three criteria: Human-likeness: Does the movement look like something a real person could perform?

Motion diversity: Does the animation contain a variety of different movements, rather than repeating the same motions throughout?

Absence of animation errors: Is the movement free from noticeable issues such as body parts suddenly jumping, getting stuck, moving through each other, or behaving strangely?

For each criterion, provide a rating on a 7-star scale, where 1 = Very poor and 7 = Excellent.

Please base your ratings only on the visual motion.

You will begin with 2 practice questions and 1 practice attention question.

The audio condition follows the same presentation structure but evaluates speech-dependent properties. Participants viewed the animation together with the accompanying speech and subsequently rated speech timing and content match. The corresponding interface is shown in Figure 5.

The instructions explicitly distinguish temporal coordination from semantic correspondence, allowing the two speech-dependent dimensions to be judged independently:

![](images/de53921ebb43d7378a8c1dff5cfe7e68e5c25a0ce1a68c493be1589193b68010.jpg)  
(a) Before video playback.

![](images/3df4c203a27eff2c077986549cd1f29082b539f24d5e66fa034177da9211b35c.jpg)  
(b) After video playback.  
Figure 5: Participant interface for the audio condition. After viewing the audiovisual stimulus, participants rate speech timing and content match.

## Participant Instructions: Audio Condition

In the following pages, you will watch animations with audio and evaluate the gestures according to two criteria: Speech timing: Do the gestures occur at appropriate moments relative to the rhythm, pauses, and emphasis of the speech? Content match: Do the gestures meaningfully correspond to the spoken content?

For each criterion, provide a rating on a 7-star scale, where 1 = Very poor and 7 = Excellent.

Some attention-check questions regarding the spoken content will appear throughout the study. Please listen carefully before responding.

You will begin with 2 practice questions and 1 practice attention question.

To monitor response quality, attention checks were distributed throughout both studies. The muted condition used instructedresponse checks, whereas the audio condition used questions about the content of the preceding clip. Representative examples are shown in Figure 6.

![](images/2e990172f7f04ecdf0b093ae0d9464f602cc2f09858b93e55d84800812e6e744.jpg)  
(a) Muted condition.  
(b) Audio condition.  
Figure 6: Attention-check interfaces used in the two perceptual studies. The muted condition uses an instructed-response check, while the audio condition uses a content-based question about the preceding stimulus.

## 8. LLM-Based Semantic Gesture Annotation

This section provides additional details of the LLM-based semantic gesture annotation procedure introduced in Sec. 3.5. We describe the annotation pipeline, the prompt used for multimodal annotation, an example output, and human validation of the resulting annotations.

You are an expert in multimodal gesture understanding.   
TASK: Given a short video clip of a person speaking and the semantic-word interval below,   
annotate only the gesture corresponding to the specified semantic word.   
1. Gesture description. Describe the visible gesture involving the hands, arms, head, or   
upper body. Specify: (i) which hand(s) move; (ii) hand shape; (iii) palm or hand orientation;   
(iv) spatial location relative to the body; and (v) movement trajectory, direction, amplitude,   
and speed. If multiple micro-gestures occur within the interval, describe them in temporal   
order.   
2. Contextual meaning. Infer the communicative function of the gesture in context. Assign   
one primary class: iconic, metaphoric, deictic, or beat, and briefly describe the conveyed   
meaning.   
ANNOTATION SCOPE: Annotate only the specified semantic-word interval and ignore all   
other time spans.   
RULES: Be objective and concise. Base the description strictly on what is visible within the   
target interval. If the gesture is not visible or is ambiguous, use unknown. If the hands are   
occluded, explicitly state this. Do not borrow cues from outside the specified interval. Prefer   
precise descriptions over speculative interpretations.   
OUTPUT FORMAT:   
{   
"word": "..."   
"gesture\_description": "...",   
"contextual\_meaning": "...",   
"class": "...",   
"confidence": ..   
}

## 8.1. Annotation Procedure

Our annotation pipeline converts predefined semantic gesture intervals into fine-grained natural-language descriptions of gesture form and communicative meaning. The procedure consists of four steps.

First, each source sequence is manually segmented into short video clips according to the existing semantic-word intervals. Each clip therefore corresponds to a predefined semantic expression and contains the motion associated with that expression. Second, the semantic word associated with each interval is retrieved from the existing annotation. Third, the segmented video and its target semantic word are provided to Gemini using the prompt shown in Fig. 7. The model is instructed to describe only the gesture occurring within the specified interval and to characterize its visible form in terms of handedness, hand shape, orientation, spatial location, and movement trajectory. It additionally infers the communicative meaning of the gesture and assigns one primary gesture class from {iconic, metaphoric, deictic, beat}. Finally, the generated annotations are checked and inserted into the corresponding annotation files.

The resulting annotation for each semantic interval is represented as

$$
a _ { i } = ( w _ { i } , d _ { i } , m _ { i } , c _ { i } ) ,
$$

where $w _ { i }$ denotes the target semantic word or expression, $d _ { i }$ the fine-grained gesture description, $m _ { i }$ the inferred contextual meaning, and $c _ { i }$ the gesture class. This representation explicitly separates observable motion form from its inferred semantic function, enabling subsequent semantics-aware evaluation.

## 8.2. Annotation Example

In addition to the annotated intervals used throughout the dataset, we provide a representative event-level example in Figure 8. For this event, Gemini 2.5 Pro observes a rendered motion clip of approximately 2-3 seconds centered on the target semantic interval, together with the corresponding target word and the structured annotation prompt shown in 7. The model is instructed to ground its response in the visible motion and to identify the observed gesture form, infer its contextual meaning, and assign a gesture type. Figure 9 shows the resulting annotation output for the target word had, illustrating how the proposed pipeline converts a short motion event into a structured semantic description.

## 8.3. Human Validation

To assess the reliability of the automatically generated annotations, we conducted a manual validation study on a randomly sampled subset of the annotated semantic gestures. Human validation considered both the visual description of the gesture and its inferred semantic interpretation.

For each sampled interval, the annotator viewed the corresponding video and compared it with the LLM-generated annotation. Gesture descriptions were checked with respect to five observable components: handedness, hand shape, orientation, location, and movement. The semantic interpretation was additionally checked for whether the predicted gesture class and contextual meaning

## LLM Prompt: Semantic Gesture Annotation

Figure 7: Prompt used for LLM-based semantic gesture annotation. The model receives a short video segment together with its predefined semantic word and returns a structured description of gesture form and contextual meaning.  
![](images/e044be12fad4f169e370b787011fecd19d1ea4b7076173eb39bdfb51f60cc0cf.jpg)  
.. We had ...  
Figure 8: Representative event-level motion example used for semantic gesture annotation. The figure visualizes a short rendered motion clip of approximately 2–3 s centered on the annotated semantic interval. This event clip, together with the corresponding target word and structured annotation prompt, is provided to Gemini 2.5 Pro as input for fine-grained semantic annotation.

were consistent with the observed gesture and associated semantic expression.

An annotation was considered correct when its gesture description captured the principal visible motion and its semantic interpretation was compatible with the manually assessed communicative function. Under this criterion, 75% of the evaluated LLMgenerated annotations were judged correct. This validation indicates that the LLM annotations provide a useful scalable source of semantic supervision, while also motivating the use of human verification for ambiguous gesture instances.

Example annotation output for an additional semantic-gesture   
event   
Target semantic word: had   
LLM annotation:   
{   
"word": "had",   
"gesture\_description":   
"Right hand moves at shoulder level in front   
of the torso. Motion is a small, straight   
downward flick with very small amplitude   
and speed.",   
"contextual\_meaning":   
"Holding or bringing an implied object into   
possession.",   
"class": "iconic",   
"confidence": 0.58   
}  
Figure 9: Example of an LLM-generated semantic gesture annotation. The target semantic expression is paired with a finegrained description of the visible motion and an interpretation of its communicative meaning.