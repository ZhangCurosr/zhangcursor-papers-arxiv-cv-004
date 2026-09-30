# OMNIROUTE: MAPPING TEMPORAL SEMANTIC EVIDENCE TO AUDIO-VISUAL TOKEN BUDGETS FOR EFFICIENT OMNIMODAL LARGE LANGUAGE MODELS

Yuchen Deng<sup>1,2,∗</sup>, Zidang Cai<sup>1,∗</sup>, Feidiao Yang<sup>2</sup>, Yufei Wang<sup>1,2</sup>, Jie Wang<sup>1,2</sup>, Hai-Tao Zheng<sup>1,2</sup>, Yuxing Han<sup>1,†</sup>

<sup>1</sup>Shenzhen International Graduate School, Tsinghua University, China <sup>2</sup>Pengcheng Laboratory, China

## ABSTRACT

Omnimodal large language models (Omni-LLMs) encode audio and visual streams into temporally interleaved token sequences for multimodal reasoning. However, processing long audio-visual token sequences incurs substantial prefill costs. Existing compression methods have made progress, but often overlook temporal changes in audio-visual semantic relevance. Motivated by temporal variation and local continuity, we propose OmniRoute, a training-free, twostage compression framework. First, Temporal Evidence-Guided Budgeting (TEGB) derives chunk-wise modality preferences and initial leading-modality budgets from semantic relevance and local content variation. Second, Budget-Constrained Semantic Compression (BCSC) compresses the leading modality and then calibrates the follower’s retention target using the actual retained fraction. For video, it combines spatiotemporal grouping with query-guided selection; for audio, it selects tokens based on encoder attention and query relevance, then merges residual tokens into context anchors under visual guidance. Experiments on four representative benchmarks demonstrate a better trade-off between inference efficiency and performance than competitive baselines. The code and interface will be released to facilitate further research.

## 1. INTRODUCTION

Omnimodal large language models (Omni-LLMs) [1] extend multimodal LLMs [2, 3] toward unified understanding of text, vision, and audio, with applications in multimedia content analysis and embodied intelligence. Representative Omni-LLMs, such as Qwen2.5- Omni, encode audio and visual streams into a temporally aligned token sequence for reasoning over complementary evidence [4, 5].

However, long audio-visual token sequences incur substantial prefill costs due to quadratic attention complexity. Therefore, token compression has emerged as an important direction [6, 7, 8] for efficient Omni-LLMs inference, aiming to reduce computational overhead while maintaining model performance.

Existing methods compress multimodal tokens before prefill through cross-modal guidance and structure-aware selection, preserving cross-modal complementarity [9, 10, 11]. Other methods reduce model-internal redundancy by adapting token and KV-cache retention using layer-wise signals and cross-modal interactions [12, 13, 14]. Recent methods further incorporate query relevance into token selection to prioritize task-relevant evidence [15, 16, 17]. However, these methods often overlook temporal changes in audiovisual semantic relevance. Across 500 WorldSense videos analyzed using Qwen2.5-Omni-7B, audio-visual semantic relevance exhibits temporal variation and local continuity. This motivates us to account for temporal characteristics in compression design.

![](images/911d9446d71b6978bb885fa8195f25135d3e7f30f40410bdee6fc7593e8e2fdc.jpg)  
Fig. 1: WorldSense accuracy using Qwen2.5-Omni-3B/7B.

To this end, we propose OmniRoute, a training-free, two-stage framework for semantics-aware token compression in Omni-LLMs. First, Temporal Evidence-Guided Budgeting (TEGB) estimates chunk-wise modality preferences by comparing the query relevance of audio and video. It then sets initial retention budgets for the leading modality based on preference strength and visual content variation within and across chunks. Second, Budget-Constrained Semantic Compression (BCSC) performs chunk-wise compression guided by the leading modality. It compresses this modality according to its initial budget and then uses its actual retention rate to calibrate the follower’s target retention rate. For video, it constructs a candidate mask through hierarchical spatial grouping and temporal grouping of similar regions, then applies query-guided selection to meet the retention target. For audio, it combines encoder-attention importance with query relevance to select tokens, then merges selected residual tokens into context anchors under visual guidance.

Extensive experiments on WorldSense, AVUTBench, VideoMME, and DailyOmni show that OmniRoute consistently achieves a better efficiency-performance trade-off than strong baselines. As illustrated in Fig. 1, OmniRoute reaches 46.9% accuracy on Qwen2.5- Omni-7B using 45% of the tokens, exceeding the full-token baseline.

We summarize the contributions of this paper as follows.

• Motivated by temporal variation and local continuity in semantic relevance, we propose OmniRoute, a training-free, two-stage token compression framework for Omni-LLMs.

• We introduce TEGB for chunk-wise modality preferences and initial budgets reflecting semantic relevance and local content variation, and BCSC for modality-led semantic compression with cross-modal budget calibration using actual retention.

• Experiments demonstrate that OmniRoute better balances inference efficiency and model performance than baselines.

## 2. METHOD

## 2.1. Preliminaries

Existing methods may overlook temporal variation and local continuity in query relevance. For the i-th chunk, let $E _ { i } ^ { a }$ and $E _ { i } ^ { v }$ denote the mean cosine similarities between the mean query embedding q and the audio and visual token embeddings. Their contrast is

$$
D _ { i } = E _ { i } ^ { a } - E _ { i } ^ { v } .\tag{1}
$$

Larger $D _ { i }$ indicates higher query relevance of audio relative to vision, while the sequence $\{ \bar { D _ { i } } \} _ { i = 1 } ^ { n }$ tracks how this balance evolves across successive chunks. Using Qwen2.5-Omni-7B, we analyze 500 randomly sampled WorldSense videos. For the s-th video with $n _ { s }$ native chunks, we quantify dispersion using the scale-normalized interdecile range

$$
V _ { s } = \frac { Q _ { 0 . 9 } \Big ( \{ D _ { i } ^ { ( s ) } \} _ { i = 1 } ^ { n _ { s } } \Big ) - Q _ { 0 . 1 } \Big ( \{ D _ { i } ^ { ( s ) } \} _ { i = 1 } ^ { n _ { s } } \Big ) } { \mathrm { m e d i a n } _ { 1 \leq i \leq n _ { s } } \Big ( \lvert E _ { i } ^ { a , ( s ) } \rvert + \lvert E _ { i } ^ { v , ( s ) } \rvert \Big ) + \varepsilon } .\tag{2}
$$

We measure local continuity using

$$
\begin{array} { r } { C _ { s } = \frac { \frac { 1 } { n _ { s } - 1 } \sum _ { i = 1 } ^ { n _ { s } - 1 } | D _ { i + 1 } ^ { ( s ) } - D _ { i } ^ { ( s ) } | } { \binom { n _ { s } } { 2 } ^ { - 1 } \sum _ { 1 \le i < j \le n _ { s } } | D _ { i } ^ { ( s ) } - D _ { j } ^ { ( s ) } | } . } \end{array}\tag{3}
$$

Here, $Q _ { p }$ denotes the p-quantile, and $\varepsilon = 1 0 ^ { - 1 2 }$ is a numerical stabilizer. A larger $V _ { s }$ indicates greater within-video dispersion of $D _ { i } ^ { ( s ) }$ , while $C _ { s } ~ < ~ 1$ indicates that adjacent chunks differ less in $D _ { i } ^ { ( s ) }$ than arbitrary chunk pairs.

As illustrated in Fig. 2(a), the $D _ { i }$ trajectory of an example video exhibits recurring peaks and valleys while evolving coherently across neighboring chunks. Panels (b)-(c) summarize the results across 500 videos: $V _ { s }$ ranges from 0.062 to 0.600 (median 0.225), while $C _ { s }$ is below 1 for 391 videos (78.2%) and averages 0.860 (video-level bootstrap 95% CI: 0.844–0.875). Together, these results indicate that the query-conditioned audio-visual relevance contrast varies across chunks yet often remains locally continuous.

## 2.2. Overview of OmniRoute

OmniRoute is a training-free framework for semantics-aware audiovisual token compression in Omni-LLMs. As illustrated in Fig. 3, it operates before LLM prefill through two stages: Temporal Evidence-Guided Budgeting (TEGB) and Budget-Constrained Semantic Compression (BCSC).

## 2.3. Temporal Evidence-Guided Budgeting

TEGB estimates the modality preference for each chunk from audiovisual semantic evidence. It uses content changes within and between chunks, together with preference strength, to construct the leading modality’s retention budget.

Modality preference. Using the query-conditioned evidence contrast $D _ { i }$ defined in Eq. (1), we compute

$$
\lambda _ { i } = \sigma \left( { \frac { D _ { i } - \tau } { T } } \right) ,\tag{4}
$$

where $\sigma$ is the sigmoid, τ is the direction threshold, and the temperature $T > 0$ controls preference sharpness. For each chunk, the leading modality is $\ell _ { i } = a \mathrm { i f } \lambda _ { i } \geq 1 / 2$ , and $\ell _ { i } = v$ otherwise.

![](images/6e8ca77477b527faf4ca410db7343097bca56183455353a06fc97a244937d045.jpg)  
(c) Local Continuity

(b) Semantic Volatility  
![](images/311b529dcc3b9afd76416b5acb2048752ad98af4eb2184c15105e47ea4bc6897.jpg)

![](images/92926cccb63aba994cd3d07b0994a2223c0015c5ac10ff5cf44c28c97f0585b4.jpg)  
Fig. 2: Temporal audio-visual evidence. (a) The $D _ { i }$ trajectory and its centered five-chunk moving average for a WorldSense video. (b)- (c) Distributions of $V _ { s }$ and $C _ { s }$ over 500 videos.

Local content variation. We characterize visual changes at two temporal scales. Within chunk $i , \ m _ { i }$ measures the mean cosine distance between visual features at corresponding spatial positions across consecutive time steps. Between adjacent chunks, we compare mean visual embeddings: $n _ { i } = 1 - \cos ( \mu _ { i } , \mu _ { i - 1 } )$ for $i > 1$ with $n _ { 1 } = 0$ , where $\pmb { \mu } _ { i }$ averages visual token embeddings in chunk i. They adjust the initial visual retention rate:

$$
g _ { i } = 0 . 7 m _ { i } + 0 . 3 n _ { i } ,
$$

$$
r _ { i , 0 } ^ { v } = \Pi _ { v } ( \overline { { r } } _ { v } + \alpha ( g _ { i } - \overline { { g } } ) ) ,\tag{5}
$$

(6)

where ${ \overline { { g } } } \ = \ n ^ { - 1 } \sum _ { i = 1 } ^ { n } g _ { j }$ is the mean variation score over the n chunks of the video, $\scriptstyle { \overline { { r } } } _ { v }$ is the base visual retention rate, α $\geq 0$ controls sensitivity, and Π<sub>v</sub> clips visual retention to its allowed range. Leading-modality budget. We increase retention with preference strength by adding $\Delta _ { i } = 2 K | \lambda _ { i } - 1 / 2 | \left( K \geq 0 \right) !$

$$
r _ { i } ^ { \ell _ { i } } = \left\{ \prod _ { a } ( \overline { { r } } _ { a } + \Delta _ { i } ) , \ell _ { i } = a , \right.\tag{7}
$$

where $\overline { { r } } _ { a }$ is the base audio retention rate and $\Pi _ { a }$ clips the audio token retention rate to its allowed range. These rates define each chunk’s initial leading-modality retention demand.

## 2.4. Budget-Constrained Semantic Compression

Given the modality preference and initial leading-modality retention demand from TEGB, BCSC compresses the selected leading modality before its follower in each paired chunk.

Video-led compression. BCSC recursively partitions each spatial token grid into regions represented by mean features, then merges similar, overlapping regions across adjacent time units to construct a candidate token mask. The semantic weight $\gamma _ { v } \geq 0$ raises spatial and temporal similarity thresholds for query-relevant regions, encouraging finer partitions and more conservative merging. Queryguided selection adjusts this mask to TEGB’s target retention rate $r _ { i } ^ { v }$ ; the retained visual fraction $ { \widehat { r } } _ { i } ^ { v }$ calibrates the audio target:

$$
\begin{array} { r } { \check { r } _ { i } ^ { a } = \Pi _ { a } ( \overline { { r } } _ { a } + \beta ( \widehat { r } _ { i } ^ { v } - \overline { { r } } _ { v } ) ) , } \end{array}\tag{8}
$$

where $\beta \in [ 0 , 1 ]$ controls cross-modal coupling in both directions.   
Let $\mathbf { a } _ { i , j }$ denote the embedding of the j-th audio token in chunk i.

![](images/267b5af4248d324afc404083d21179c549e299b65d3e2663ca0ab20ec0406f03.jpg)  
Fig. 3: OmniRoute overview. TEGB derives chunk-wise modality preferences and leading-modality budgets from semantic relevance and local variation. BCSC calibrates follower targets using actual leading-modality retention. Video compression combines spatial and temporal grouping with query-guided selection; audio compression combines attention- and query-based selection with visually guided merging.

BCSC combines encoder-attention importance $h _ { i , j }$ and query relevance $e _ { i , j } ^ { a } = \cos ( \mathbf { q } , \mathbf { a } _ { i , j } )$ into a selection score:

$$
o _ { i , j } ^ { a } = \widehat { h } _ { i , j } + \gamma _ { a } \widehat { e } _ { i , j } ^ { a } ,\tag{9}
$$

where $\widehat { h } _ { i , j }$ and $\widehat { e } _ { i , j } ^ { a }$ are the within-chunk min-max normalized attention and query relevance scores, respectively; $\gamma _ { a } \geq 0$ controls semantic guidance. BCSC selects audio tokens in descending $o _ { i , j } ^ { a }$ order at rate $\check { r } _ { i } ^ { a }$ and adds regularly spaced context anchors from the remainder. Remaining tokens are assigned to their nearest anchors by cosine similarity. Within groups, tokens are ranked by maximum cosine similarity to retained visual tokens. Each anchor is averaged with a softmax-weighted aggregate of its top-ranked tokens.

Audio-led compression. BCSC first retains audio tokens with the largest scores from Eq. (9), using TEGB’s initial audio target retention rate $r _ { i } ^ { a }$ to determine their number. It adds regularly spaced context anchors from the remainder, assigns residual tokens to their most similar anchors, and merges top-ranked tokens within each group into these anchors using softmax-normalized visual similarities. Both residual token ranking and merging weights use all original visual tokens in the paired chunk, since video compression follows the audio step. The actual retained audio fraction ${ \hat { \widehat { r } } } _ { i } ^ { a }$ , including context anchors, calibrates the target visual retention rate:

$$
\begin{array} { r } { \check { r } _ { i } ^ { v } = \Pi _ { v } \big ( \overline { { r } } _ { v } + \beta ( \widehat { r } _ { i } ^ { a } - \overline { { r } } _ { a } ) \big ) . } \end{array}\tag{10}
$$

BCSC then applies spatial and temporal masking, followed by query-guided visual token selection at this mapped rate.

## 3. EXPERIMENTS

## 3.1. Experiment Setup

Datasets. We evaluate OmniRoute on four representative benchmarks: WorldSense [18], covering audio–visual understanding across diverse real-world domains; VideoMME [19], spanning short, medium, and long videos; AVUTBench [20], emphasizing audio-centric video understanding; and DailyOmni [21], focusing on audio–visual reasoning in everyday scenarios.

Implementation Details. We evaluate our approach on Qwen2.5- Omni 3B/7B [1] and OmniVinci [4] using a single NVIDIA L20 GPU (48 GB). For Qwen2.5-Omni, we uniformly sample videos at a target rate of 2 fps with a maximum of 768 frames per video. For OmniVinci, we uniformly sample 128 frames per video. Across all experiments, we set the spatial and temporal similarity thresholds to $\eta _ { s } = 0 . 8 2$ and $\eta _ { t } = 0 . 5 8 ,$ , respectively, the routing temperature to $T \ = \ 0 . 0 3$ , and the context-anchor ratio to 0.05. The default semantic weights are $\gamma _ { v } ~ = ~ 0 . 2 1$ and $\gamma _ { a } ~ = ~ 0 . 0 9$ , with a shared cross-modal coupling strength of $\beta = 0 . 5 0$ . Repeated OmniRoute evaluations under identical settings yielded identical accuracy.

Baselines. We use full-token inference as the uncompressed reference and random pruning (Random) as a control baseline to evaluate token selection effectiveness. We also compare with DyCoke (V&A) [22], with its temporal compression module applied to both video and audio tokens; FlashVID [23], a strong visual token compression method for video LLMs; and OmniZip [9], a strong compression method specifically designed for Omni-LLMs.

## 3.2. Main Results

Comparison with State-of-the-Art Methods. As shown in Table 1, OmniRoute achieves the highest reported average normalized accuracy across Qwen2.5-Omni-7B/3B and OmniVinci on audiovisual understanding tasks. At 45% token retention, it preserves 97.9–98.7% of full-token accuracy on average across four benchmarks while reducing FLOPs by 61–64%, outperforming Random and DyCoke despite their higher token retention. This favorable accuracy–efficiency trade-off is further supported by gains of 1.1, 0.5, and 1.3 percentage points over OmniZip at matched retention on Qwen2.5-Omni-7B/3B and OmniVinci, respectively. Notably, on WorldSense, OmniRoute achieves 46.9% accuracy with Qwen2.5- Omni-7B at 45% token retention, surpassing the full-token baseline. Efficiency Analyses. On WorldSense with Qwen2.5-Omni-7B (Table 2), OmniRoute achieves the lowest peak memory and highest accuracy among the compared methods at 45% token retention. It reduces peak GPU memory by 38.6% (44 to 27 GB), with 2.53× prefill and 1.35× end-to-end speedups over full-token inference, outperforming DyCoke and FlashVID in both memory usage and latency. At matched retention, OmniRoute achieves prefill and endto-end latencies comparable to those of OmniZip while reducing peak GPU memory by 5 GB and improving accuracy by 1.0 percentage point. These results demonstrate improved inference efficiency while preserving predictive accuracy.

Table 1: Comparison with token compression methods across different omni-models.
<table><tr><td>Model</td><td>Method</td><td>Retained Ratio</td><td>FLOPs Ratio</td><td>WorldSense</td><td>AVUTBench</td><td>VideoMME</td><td>DailyOmni</td><td>Avg.</td></tr><tr><td rowspan="6">Qwen2.5- Omni-7B</td><td>Full Tokens</td><td>100%</td><td>100%</td><td>46.8</td><td>64.5</td><td>66.0</td><td>62.9</td><td>100%</td></tr><tr><td>Random</td><td>55%</td><td>48%</td><td>43.6</td><td>61.0</td><td>65.4</td><td>58.6</td><td>95.0%</td></tr><tr><td>DyCoke (V&amp;A)</td><td>50%</td><td>44%</td><td>44.6</td><td>62.0</td><td>65.5</td><td>55.8</td><td>94.8%</td></tr><tr><td>FlashVID</td><td>45%</td><td>39%</td><td>46.0</td><td>63.0</td><td>65.9</td><td>57.0</td><td>96.6%</td></tr><tr><td>OmniZip</td><td>45%</td><td>39%</td><td>45.9</td><td>63.0</td><td>66.1</td><td>59.5</td><td>97.6%</td></tr><tr><td>OmniRoute (Ours)</td><td>45%</td><td>39%</td><td>46.9</td><td>63.4</td><td>66.0</td><td>60.6</td><td>98.7%</td></tr><tr><td rowspan="6">Qwen2.5- Omni-3B</td><td>Full Tokens</td><td>100%</td><td>100%</td><td>46.4</td><td>62.2</td><td>62.6</td><td>62.0</td><td>100%</td></tr><tr><td>Random</td><td>55%</td><td>45%</td><td>42.8</td><td>58.7</td><td>61.1</td><td>54.8</td><td>93.2%</td></tr><tr><td>DyCoke (V&amp;A)</td><td>50%</td><td>40%</td><td>44.0</td><td>60.7</td><td>61.6</td><td>54.4</td><td>94.6%</td></tr><tr><td>FlashVID</td><td>45%</td><td>37%</td><td>45.4</td><td>59.6</td><td>61.8</td><td>57.0</td><td>96.1%</td></tr><tr><td>OmniZip</td><td>45%</td><td>36%</td><td>45.2</td><td>61.3</td><td>62.6</td><td>58.6</td><td>97.6%</td></tr><tr><td>OmniRoute (Ours)</td><td>45%</td><td>36%</td><td>45.8</td><td>61.7</td><td>62.6</td><td>58.7</td><td>98.1%</td></tr><tr><td rowspan="6">OmniVinci</td><td>Full Tokens</td><td>100%</td><td>100%</td><td>49.2</td><td>67.4</td><td>68.6</td><td>64.9</td><td>100%</td></tr><tr><td>Random</td><td>55%</td><td>49%</td><td>47.0</td><td>63.1</td><td>66.3</td><td>56.4</td><td>93.2%</td></tr><tr><td>DyCoke (V&amp;A)</td><td>50%</td><td>44%</td><td>47.6</td><td>63.4</td><td>68.0</td><td>58.1</td><td>94.9%</td></tr><tr><td>FlashVID</td><td>50%</td><td>44%</td><td>44.7</td><td>64.0</td><td>67.6</td><td>60.2</td><td>94.3%</td></tr><tr><td>OmniZip</td><td>45%</td><td>39%</td><td>47.1</td><td>63.7</td><td>68.4</td><td>62.7</td><td>96.6%</td></tr><tr><td>OmniRoute (Ours)</td><td>45%</td><td>38%</td><td>47.7</td><td>64.9</td><td>68.2</td><td>64.1</td><td>97.9%</td></tr></table>

Table 2: Efficiency on WorldSense with Qwen2.5-Omni-7B. Mem. denotes peak GPU memory.
<table><tr><td>Method</td><td>Mem. ↓</td><td>Prefill (ms) ↓</td><td>Acc. ↑</td><td>Latency (s) ↓</td></tr><tr><td>Full Tokens (100%)</td><td>44 G</td><td> $2 3 7 1 \left( 1 . 0 0 \times \right)$ </td><td>46.8</td><td> $1 0 . 9 9 \left( 1 . 0 0 \times \right)$ </td></tr><tr><td>DyCoke (50%)</td><td>36 G</td><td> $1 3 8 6 \left( 1 . 7 1 \times \right)$ </td><td>44.6</td><td> $8 . 5 9 \left( 1 . 2 8 \times \right)$ </td></tr><tr><td>FlashVID (45%)</td><td>35 G</td><td> $1 0 7 3 \ : ( 2 . 2 1 \times )$ </td><td>46.0</td><td> $9 . 6 3 \ : ( 1 . 1 4 \times )$ </td></tr><tr><td>OmniZip (45%)</td><td>32 G</td><td>894 (2.65×)</td><td>45.9</td><td> $\mathbf { 7 . 9 9 } \left( \mathbf { 1 . 3 8 } \times \right)$ </td></tr><tr><td>Ours (45%)</td><td>27 G</td><td>936 (2.53×)</td><td>46.9</td><td>8.12 (1.35×)</td></tr></table>

![](images/a9c6394a6876382bb16e277d79788f4812924c243ebd4505e36a46278b49936c.jpg)  
Fig. 4: Hyperparameter analysis on WorldSense: modality-specific weights $\gamma _ { v }$ and $\gamma _ { a } ,$ , and cross-modal coupling $\beta .$

## 3.3. Ablation Study

Ablation of Core Components. Table 3 examines budget adjustment and modality routing on WorldSense with Qwen2.5-Omni-7B at 45% token retention. Disabling TEGB’s within- and across-chunk variation signals $( m _ { i }$ and n<sub>i</sub>), while retaining dynamic routing, lowers accuracy from 46.9% to 46.4%. With these signals retained, fixed audio-led and video-led compression reduce accuracy by 1.0 and 1.7 percentage points, respectively. These results highlight the benefits of temporally adaptive budgeting and modality routing for preserving audio–visual understanding under token compression.

Analysis of Compression Hyperparameters. Fig. 4 examines sensitivity to semantic guidance and cross-modal coupling on WorldSense. We first vary $\gamma _ { v } ,$ which adjusts query-dependent spatial and temporal similarity thresholds, with accuracy peaking at $\gamma _ { v } ~ = ~ 0 . 2 1 ~ ( 4 6 . 6 \% )$ . Fixing $\gamma _ { v } ~ = ~ 0 . 2 1$ , we then vary $\gamma _ { a }$ , which controls the contribution of query relevance relative to encoder attention; accuracy peaks at $\gamma _ { a } = 0 . 0 9 ( 4 6 . 9 \% )$ . Both sweeps exhibit non-monotonic trends, indicating that stronger semantic guidance does not necessarily improve accuracy. For cross-modal coupling, $\beta = 0 . 5 0$ yields the highest tested accuracy (46.9%), with lower scores at both smaller and larger coupling strengths. Within the tested ranges, these results favor moderate semantic guidance and cross-modal coupling for preserving audio–visual understanding under compression. Accordingly, we adopt these values as the default hyperparameter settings.

Table 3: Ablation of OmniRoute’s core components on World-Sense with Qwen2.5-Omni-7B.
<table><tr><td>Setting</td><td> $m _ { i } , n _ { i }$ </td><td>Routing</td><td>Retained Ratio</td><td>Acc.</td></tr><tr><td>Full OmniRoute</td><td>√</td><td>Dynamic</td><td>45%</td><td>46.9</td></tr><tr><td>w/o mi/ni</td><td>×√</td><td>Dynamic</td><td>45%</td><td>46.4</td></tr><tr><td>Audio-led only</td><td></td><td>Audio-led</td><td>45%</td><td>45.9</td></tr><tr><td>Video-led only</td><td>√</td><td>Video-led</td><td>45%</td><td>45.2</td></tr></table>

Table 4: Performance of OmniRoute under different retained ratios on WorldSense with Qwen2.5-Omni-7B.
<table><tr><td>Retained Ratio</td><td>FLOPs Ratio</td><td>Mem. ↓</td><td>Prefill (ms) ↓</td><td> $\operatorname { A c c . \uparrow }$ </td><td>Latency (s) ↓</td></tr><tr><td>45%</td><td>39%</td><td>27 G</td><td>936</td><td>46.9</td><td>8.12</td></tr><tr><td>40%</td><td>31%</td><td>27 G</td><td>788</td><td>46.3</td><td>8.00</td></tr><tr><td>35%</td><td>26%</td><td>27 G</td><td>640</td><td>46.1</td><td>7.73</td></tr><tr><td>30%</td><td>18%</td><td>27 G</td><td>558</td><td>45.6</td><td>7.59</td></tr></table>

Effect of Token Retention Ratio. Table 4 evaluates OmniRoute under different token budgets on WorldSense with Qwen2.5-Omni-7B. As token retention decreases from 45% to 30%, accuracy declines from 46.9% to 45.6%, while FLOPs and prefill time decrease by 54% and 40%, respectively. End-to-end latency decreases from 8.12 to 7.59 s. These results show that OmniRoute offers a flexible accuracy-efficiency trade-off, achieving faster inference under tighter token budgets with limited accuracy degradation.

## 4. CONCLUSION

We introduce OmniRoute, a training-free, two-stage token compression framework for Omni-LLMs. It combines temporal evidenceguided budgeting with modality-led semantic compression, calibrating follower budgets using actual leading-modality retention. Experiments on four benchmarks across three models demonstrate that OmniRoute achieves a better accuracy-efficiency trade-off than competitive baselines. Despite these gains, OmniRoute still relies on manually configured modality-specific semantic weights and crossmodal coupling strength. Future work will explore adapting these parameters to target token budgets to reduce manual tuning. We will also optimize the implementation to lower inference latency.

## 5. REFERENCES

[1] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin, “Qwen2.5- omni technical report,” 2025.

[2] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al., “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 24185–24198.

[3] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao, “Videochat: Chat-centric video understanding,” Science China Information Sciences, vol. 68, no. 10, pp. 200102, 2025.

[4] Hanrong Ye, Chao-Han Huck Yang, Arushi Goel, Wei Huang, Ligeng Zhu, Yuanhang Su, Sean Lin, An-Chieh Cheng, Zhen Wan, Jinchuan Tian, et al., “Omnivinci: Enhancing architecture and data for omni-modal understanding llm,” arXiv preprint arXiv:2510.15870, 2025.

[5] Wenwen Tong, Hewei Guo, Dongchuan Ran, Jiangnan Chen, Jiefan Lu, Kaibin Wang, Keqiang Li, Xiaoxu Zhu, Jiakui Li, Kehan Li, et al., “Interactiveomni: A unified omni-modal model for audio-visual multi-turn dialogue,” arXiv preprint arXiv:2510.13747, 2025.

[6] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al., “Qwen3-omni technical report,” arXiv preprint arXiv:2509.17765, 2025.

[7] Qwen Team, “Qwen3. 5-omni technical report,” arXiv preprint arXiv:2604.15804, 2026.

[8] Guangzhi Sun, Yixuan Li, Yudong Yang, and Chao Zhang, “Omnimem: Perturbation-aware memory compression for streaming audio-visual llms,” arXiv preprint arXiv:2606.07577, 2026.

[9] Keda Tao, Kele Shao, Bohan Yu, Weiqiang Wang, Huan Wang, et al., “Omnizip: Audio-guided dynamic token compression for fast omnimodal large language models,” arXiv preprint arXiv:2511.14582, 2025.

[10] Chao Gong, Depeng Wang, Zhipeng Wei, Ya Guo, Huijia Zhu, and Jingjing Chen, “Echoingpixels: Cross-modal adaptive token reduction for efficient audio-visual llms,” arXiv preprint arXiv:2512.10324, 2025.

[11] Yuchen Deng, Zidang Cai, Hai-Tao Zheng, Jie Wang, Feidiao Yang, and Yuxing Han, “Omnirefine: Alignment-aware cooperative compression for efficient omnimodal large language models,” arXiv preprint arXiv:2605.12056, 2026.

[12] Zining Wang, Zhihang Yuan, Yingjie Zhai, Wenshuo Li, Han Shu, Ruihao Gong, Jinyang Guo, and Xianglong Liu, “Omnifit: Bridging modalities via layer-adaptive token compression for omnimodal large language models,” in Forty-third International Conference on Machine Learning.

[13] Chaeyoung Jung, Youngjoon Jang, Seungwoo Lee, and Joon Son Chung, “Fastav: Efficient token pruning for audio-visual large language model inference,” arXiv preprint arXiv:2601.13143, 2026.

[14] Zhonghua Jiang, Kui Chen, Kunxi Li, Keting Yin, Yiyun Zhou, Zhaode Wang, Chengfei Lv, and Shengyu Zhang, “Acckv: Towards efficient audio-video llms inference via adaptivefocusing and cross-calibration kv cache optimization,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2026, vol. 40, pp. 5494–5502.

[15] Morunliu Yang, Ruotao Xu, Le Li, Yue Wang, Jianxin Zhang, Juntao Li, Yihang Lou, Siwei Feng, and Peifeng Li, “Omniselect: Dynamic modality-aware token compression for efficient omni-modal large language models,” arXiv preprint arXiv:2605.18041, 2026.

[16] Yeo Jeong Park, Hyemi Jang, Minseo Choi, Jongsun Lee, Jooyoung Choi, and Yongkweon Jeon, “Omnidrop: Layer-wise token pruning for omni-modal llms via query-guidance,” arXiv preprint arXiv:2605.14458, 2026.

[17] Yijing Chen, Wenhui Tan, Xiaoyi Yu, Yuyue Wang, Xin Cheng, Kaisi Guan, Hao Jiang, Xiangyang Li, Guojie Zhu, and Ruihua Song, “Avoc: Enhancing hour-level audio-video understanding in omni-modal llms via retrieval-inspired token compression,” arXiv preprint arXiv:2606.24286, 2026.

[18] Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie, “Worldsense: Evaluating real-world omnimodal understanding for multimodal llms,” arXiv preprint arXiv:2502.04326, 2025.

[19] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al., “Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2025, pp. 24108–24118.

[20] Yudong Yang, Jimin Zhuang, Guangzhi Sun, Changli Tang, Yixuan Li, Peihan Li, Yifan Jiang, Wei Li, Zejun Ma, and Chao Zhang, “Audio-centric video understanding benchmark without text shortcut,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 6580–6598.

[21] Ziwei Zhou, Rui Wang, Zuxuan Wu, and Yu-Gang Jiang, “Daily-omni: Towards audio-visual reasoning with temporal alignment across modalities,” arXiv preprint arXiv:2505.17862, 2025.

[22] Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang, “Dycoke: Dynamic compression of tokens for fast video large language models,” in Proceedings ofthe Computer Vision and Pattern Recognition Conference, 2025, pp. 18992–19001.

[23] Ziyang Fan, Keyu Chen, Ruilong Xing, Yulin Li, Li Jiang, and Zhuotao Tian, “FlashVID: Efficient video large language models via training-free tree-based spatiotemporal token merging,” in The Fourteenth International Conference on Learning Representations, 2026.