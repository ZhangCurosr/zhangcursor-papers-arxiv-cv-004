# VideoJev: System One Model for Video Understanding and Interaction

Yuhan Zhu<sup>∗,1</sup>, Xiangyu Zeng<sup>∗,1</sup>, Qingyi Si<sup>∗,‡,2</sup>, Chenxu Yang<sup>∗,3</sup>, Yifei Li<sup>∗,4</sup>, Luozheng Qin<sup>5</sup> Junhao Zhou<sup>3</sup>, Dingyu Yao<sup>3</sup>, Jun Zhang<sup>2</sup>, Chuanyu Qin<sup>3</sup>, Zhiqiu Zhang<sup>6</sup>, Yuandong Yang<sup>1</sup>, Yuqi Xu<sup>2</sup>, Tianyu Zhang<sup>7</sup>, Hongcheng Guo<sup>8</sup>, Jiaqi Wang<sup>2</sup>, Nan Duan<sup>2</sup>, Limin Wang<sup>‡,†,1</sup>

<sup>1</sup>NJU, <sup>2</sup>JD.com, <sup>3</sup>UCAS, <sup>4</sup>THU, <sup>5</sup>XMU, <sup>6</sup>SJTU, <sup>7</sup>USTC, <sup>8</sup>FDU

<sup>∗</sup>Equal contribution, <sup>‡</sup>Project lead, <sup>†</sup>Corresponding author

## Abstract

Visual agents must turn video observations into timely decisions to track progress and choose what to do next. Building on the emerging System One paradigm, we introduce VideoJev, a family of models that cast video understanding and interaction as structured decision tasks. We adapt pretrained multimodal large language models (MLLMs) by replacing autoregressive answer decoding with a shared head that scores temporal intervals and textual options against video evidence. These scores yield event spans or answer choices, unifying temporal grounding and multiple-choice video question answering. This combines the semantic flexibility of MLLMs with eficient decisions that agents can use directly. As an interaction unfolds, new goals and intermedi ate results prompt further queries about the same visual history. VideoJev therefore reuses visual computation through Cached inference for queries as they arrive and Joint inference for multi ple queries in a single forward pass. Across three temporal grounding benchmarks, VideoJev-4B and VideoJev-35B-A3B improve mean mIoU over their backbones by 6.7 and 10.8 points, re spectively, with measured inference speedups of 2.8–3.1×. On four video QA benchmarks, both achieve performance comparable to their backbones in Thinking mode, with measured inference speedups exceeding 100×. These results ofer a unified, reusable decision interface for composing structured visual judgments into complex, open-ended workflows.

PROJECT 7 CODE MODELS

Direct decisions. When? Shared computation. Which? Structured judgments for open-ended goals

![](images/067f630d3cd6c4a3028b3149e6a77a9c2df2efc86f062d2f228124fdf97117e4.jpg)

![](images/57df199b4827082342ace2382a638b6d01e6cf1b2ab2ea53e56cfc237f3e5fb0.jpg)

![](images/99b92007dec92dc1d06d57e0687f28c223497703d0bc0c1908de1564b3747ad0.jpg)  
Figure 1 VideoJev: structured visual decisions with shared computation. Top: temporal intervals and textual options share a candidate-scoring interface; Cached inference reuses the video prefix for arriving queries, while Joint inference scores each query group in one forward pass. Bottom: prediction quality versus model forward latency at both model scales, using the results in Tables 1 and 2. Left: mean grounding mIoU across three benchmarks and per-query latency on full Charades. Right: mean Video QA score across four benchmarks and per-question latency on full LVBench.

## 1 Introduction

Visual intelligence enables systems to perceive, reason about, and interact with the world. For such systems, video understanding must support timely decisions that guide reasoning and action. An agent may need to locate evidence, check progress, or assess possible actions, often using the same visual history. For example, an assistant following a cooking procedure may check whether an ingredient has been added before choosing the next step. High-level goals can be decomposed into conditions to verify and options to evaluate, through predefined procedures or generated plans [1, 47]. This suggests a central premise: structured visual judgments can support open-ended behavior without requiring open-ended generation at every step.

Existing visual models ofer complementary strengths. Discriminative vision models make eficient predictions but often depend on task-specific designs or fixed label spaces. Video multimodal large language models (MLLMs) provide broad semantic understanding and flexible language interfaces [24, 51], yet typically express decisions through autoregressive generation. They generate timestamps for temporal grounding [16, 35, 61] or answer text for video question answering [23, 25], even when the output is a compact, structured decision. Our goal is to combine eficient direct prediction with the semantic flexibility of pretrained MLLMs through a unified, language-conditioned interface.

The System One paradigm ofers a foundation for this goal. Jev produces typed probabilistic judgments that software can use directly, without generating answers token by token [2, 38]. Recent work extends this idea to images [48, 58]. Video introduces an additional requirement: decisions must reflect temporal structure while sharing expensive visual context across queries. When successive questions share the same observations, the cost of each additional judgment becomes central to responsive interaction.

We introduce VideoJev, a family of video System One models built around two principles: direct candidate scoring and reusable visual computation (Figure 1). Temporal grounding scores intervals for membership in a queried event and assembles them into a contiguous span. Multiple-choice video question answering (Video QA) scores textual options and selects the answer best supported by the video. Both tasks share a scoring architecture, while task-specific supervision and output rules preserve their diferent structures. This design draws on pretrained multimodal representations without autoregressive answer decoding.

To support repeated judgments, VideoJev organizes training and inference around a shared video prefix. In Cached inference, the video is encoded and prefilled once, and queries reuse the cached context as they arrive. In Joint inference, multiple queries follow the same prefix and receive separate scores in one causal forward pass. Training uses the same layout to share computation across questions.

We evaluate VideoJev-4B, a dense model, and VideoJev-35B-A3B, a mixture-of-experts (MoE) model, on three temporal grounding and four Video QA benchmarks. Relative to their Qwen3.5-4B and Qwen3.6-35B-A3B backbones in Instruct mode, the two models improve mean grounding mIoU by 6.7 and 10.8 points and mean QA scores by 8.0 and 5.7 points, respectively (Tables 1 and 2). Measured inference speedups reach 2.8–3.1× for grounding; on QA, both remain competitive with their Thinking counterparts while achieving over 100× speedups. The timed generative baselines also reuse the video prefix.

With identical training data and settings, VideoJev-4B also exceeds autoregressive fine-tuning by 1.3 points in mean grounding mIoU and 1.9 points in mean QA score (Table 3), supporting the benefit of direct decision learning beyond data alone. These findings support learning structured judgments for visual agents.

Our contributions are threefold:

• We introduce VideoJev, a family of video System One models that unifies temporal grounding and multiplechoice Video QA through direct candidate scoring.

• We reuse visual computation across training queries and through two inference modes: Cached inference for arriving queries and Joint inference for queries processed together.

• We evaluate dense 4B and MoE 35B-A3B models across seven benchmarks, demonstrating strong prediction quality and inference eficiency, with a controlled comparison against autoregressive fine-tuning.

![](images/dcae277ec2cef273776d5ba391a3ee8a3c564f2581d98c8c82b6487be14ad0ba.jpg)  
Figure 2 VideoJev: direct candidate scoring with shared visual computation. (a) Temporal intervals, represented by timestamps, and textual options share an input format. (b) Cached inference prefills the video once and processes each query sufix separately; Joint inference processes each query group in one causal language-model forward pass. Both modes share the backbone and scorer. (c) Temporal scores yield an event span; option scores select an answer. Scores and the reference interval are illustrative.

## 2 VideoJev

## 2.1 A Unified Video Decision Framework

VideoJev scores temporal intervals and textual options with a shared head over pretrained multimodal representations. Figure 2 summarizes the architecture, video-prefix reuse, and task-specific output rules.

Architecture. We adapt a pretrained MLLM with a causal language backbone and native video inputs. Following the observation that pre-MLP features support semantic similarity [62], we remove the final MLP and replace the language-model output head with a candidate scorer. Two new tokens, <|option\_end|> and <|decide|>, provide candidate readouts and a query-level decision readout. Both are initialized from the pretrained <|im\_end|> embedding. Separate linear projections map their hidden states to 512 dimensions and normalize them to unit length:

$$
u = \frac { W _ { q } h _ { d } + b _ { q } } { \| W _ { q } h _ { d } + b _ { q } \| _ { 2 } } , \qquad v _ { i } = \frac { W _ { k } h _ { i } + b _ { k } } { \| W _ { k } h _ { i } + b _ { k } \| _ { 2 } } , \qquad s _ { i } = 1 0 u ^ { \top } v _ { i } + b .\tag{1}
$$

Here, $h _ { d }$ and $h _ { i }$ are the decision and candidate states; $W _ { q } , W _ { k } , b _ { q } , b _ { k }$ are learned projection parameters, and $\| \cdot \| _ { 2 }$ is the Euclidean norm. Their normalized similarity gives logit $s _ { i }$ , with fixed scale 10 and learned scalar bias b.

Unifying temporal and option decisions. Given a video V, a language query $q ,$ and K candidates $C = \{ c _ { 1 } , \ldots , c _ { K } \}$ , we place video tokens first, followed by the task instruction, query, and candidate list. The instruction specifies temporal grounding or multiple-choice Video QA. Each candidate ends with <|option\_- end|>, and the list ends with <|decide|>. Causal modeling lets candidate readouts use the video, query, and preceding candidates; the decision readout sees the full list. Temporal candidates are timestamp texts copied from the video prefix into this list, each identifying a frame-pair interval. Their sigmoid scores indicate event membership, from which we decode a contiguous span. Option candidates are answer texts; softmax normalizes their logits into probabilities for selecting the highest-scoring option. Thus, candidates and output rules specify the task.

Reusable video computation. Causal modeling makes the video prefix independent of subsequent queries, allowing its computation to be shared. In Cached inference, we encode and prefill the video once. Each arriving query branches from the cached prefix and requires one language-model forward over its query, candidates, and readout tokens. In Joint inference, multiple queries and their candidate lists follow a single video prefix, with a separate <|decide|> readout for each query. One causal language-model forward pass yields all query scores. Later queries can access preceding query text and candidates, but no reference or predicted answers are appended. Training uses the same concatenated query layout to share video computation across questions for eficiency.

## 2.2 Temporal Grounding through Interval Decisions

An event may span many candidate intervals. Grounding therefore requires both identifying relevant intervals and covering the event as a whole. We train these complementary judgments with local membership and coverage supervision, then decode a single continuous span.

Learning temporal decisions. The backbone’s twofold temporal compression groups frames into pairs. We partition the timeline into K candidate intervals, one per pair. For interval $\boldsymbol { B } _ { i } = [ \ell _ { i } , r _ { i } ] , t _ { i }$ is the mean of its two frame timestamps, and $\ell _ { i } , r _ { i }$ are its boundaries. Each internal boundary lies halfway between the previous pair’s last frame and the next pair’s first frame; $\ell _ { 1 } = 0$ and $r _ { K }$ equals the video duration. Timestamp strings identify intervals; labels and boundaries use unrounded times. Let G be the union of annotated event intervals. Candidate i has label $y _ { i } = \mathbb { 1 } \{ t _ { i } \in G \}$ and membership probability $p _ { i } = \sigma ( s _ { i } )$ , where 1 is the indicator function and σ is the sigmoid function. Several candidates may belong to the same event, so we use multi-label binary cross-entropy to supervise each one independently:

$$
\mathcal { L } _ { \mathrm { B C E } } = - \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \left[ y _ { i } \log p _ { i } + ( 1 - y _ { i } ) \log ( 1 - p _ { i } ) \right] .\tag{2}
$$

BCE supervises local membership but does not directly reward event coverage. We therefore add a soft intersection-over-union (IoU) loss over the positive intervals. With interval duration $d _ { i } = r _ { i } - \ell _ { i } .$ , we define

$$
\mathcal { L } _ { \mathrm { I o U } } = 1 - \frac { \sum _ { i } d _ { i } p _ { i } y _ { i } } { \sum _ { i } d _ { i } \left[ y _ { i } + p _ { i } ( 1 - y _ { i } ) \right] } .\tag{3}
$$

The numerator rewards coverage of the event, while the denominator penalizes responses outside it. Weighting by duration accounts for unequal interval lengths.

Soft IoU is omitted when no midpoint is positive; BCE uses only valid candidates.

Decoding a temporal interval. To combine interval scores into an event span, we adjust logits by a probability threshold τ and select the nonempty contiguous range with the largest sum. We fix $\tau = 0 . 2 5$ across both model sizes and all inference modes in every experiment, with smoothing window 1:

$$
( a ^ { \star } , b ^ { \star } ) = \arg \operatorname* { m a x } _ { 1 \leq a \leq b \leq K } \sum _ { i = a } ^ { b } \left[ s _ { i } - \log \frac { \tau } { 1 - \tau } \right] .\tag{4}
$$

Here, a and b index the first and last candidates in a range, and $a ^ { \star }$ and $b ^ { \star }$ identify the selected range. Kadane’s algorithm finds this range in $O ( K )$ time. The selected candidates’ interval boundaries give the continuous prediction $[ \ell _ { a ^ { \star } } , r _ { b ^ { \star } } ]$ . If all adjusted logits are negative, the decoder returns the interval of the strongest candidate. We then apply lightweight boundary postprocessing to obtain a valid, nonempty interval.

## 2.3 Video QA through Option Selection

Multiple-choice Video QA requires selecting one textual option rather than identifying multiple positive intervals.

Learning candidate preferences. For a question with K answer options, the shared head assigns logit $s _ { i }$ to option i. Let $y$ be the index of the correct option. Since exactly one answer is correct, we normalize the logits with softmax and train with cross-entropy:

$$
p _ { i } = \frac { \exp ( s _ { i } ) } { \sum _ { j = 1 } ^ { K } \exp ( s _ { j } ) } , \qquad \mathcal { L } _ { \mathrm { C E } } = - \log p _ { y } ,\tag{5}
$$

where $p _ { i }$ is the probability of option i.

Reducing position bias. To discourage shortcuts based on option position, we apply a reproducible, answer-independent permutation to each eligible training question and update its correct-answer index. Questions referring to option positions or earlier turns retain their original order to preserve meaning. At inference, we keep the supplied order and select the option with the highest $p _ { i }$

## 2.4 Training Recipe

We instantiate VideoJev with Qwen3.5-4B and Qwen3.6-35B-A3B. Scoring-head projection weights are initialized from a Gaussian with standard deviation 0.005; biases start at zero.

Training data. We train temporal membership on TimeLens-100K [55] and TimeLens2-93K [61], converting annotated intervals into candidate-level labels. Option selection uses an internally curated corpus of approximately 240K multiple-choice questions over 19K videos. The content includes gaming, sports, cooking, music and dance, outdoor activities, and crafts. Questions cover entity and action recognition, spatiotemporal reasoning, information retrieval, event understanding, counting, OCR, and summarization. Videos range from 4 minutes to nearly 2 hours, and each question has four or five answer options. We group queries from the same video around a shared prefix.

Two-stage training. Inspired by projector alignment in MLLMs [28], we first train the new scorer against fixed pretrained features, then adapt the language backbone. This gives the scorer a stable representation space during its initial updates. Stage one trains only the head on 30% of the videos, with the backbone and new marker embeddings frozen. The grounding and QA objectives are

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { T G } } = \mathcal { L } _ { \mathrm { B C E } } , \qquad \mathcal { L } _ { \mathrm { Q A } } = \mathcal { L } _ { \mathrm { C E } } . } \end{array}\tag{6}
$$

Stage two trains the language backbone, marker embeddings, and aligned head on the remaining 70% of the videos. Soft IoU adds event-level coverage supervision to grounding, while $\mathrm { Q A }$ retains cross-entropy:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { T G } } = \mathcal { L } _ { \mathrm { B C E } } + 2 \mathcal { L } _ { \mathrm { I o U } } , } \end{array}\tag{7}
$$

$$
\mathcal { L } _ { \mathrm { Q A } } = \mathcal { L } _ { \mathrm { C E } } .\tag{8}
$$

Task losses are averaged equally over questions. The visual encoder and merger remain frozen throughout.   
Each stage runs for one epoch with a separate optimizer and learning-rate schedule.

## 3 Experiments

We evaluate prediction quality and inference eficiency on grounding and multiple-choice Video $\mathrm { Q A }$ , then examine model design and confidence. Unless otherwise stated, VideoJev uses Joint inference.

Table 1 Video temporal grounding. Generative models use Instruct mode by default; VideoJev uses Joint inference. The superscript <sup>TL</sup> marks TimeLens-Bench refined annotations. Bold marks the best value and underline the second best. Appendix A details the evaluation settings.
<table><tr><td rowspan="2">Model</td><td>Latency (↓)</td><td colspan="4">CharadesTL</td><td colspan="4">ActivityNetTL</td><td colspan="4">QVHighlightsTL</td><td rowspan="2">Avg.</td></tr><tr><td>seconds</td><td></td><td>R1@.3 R1@.5 R1@.7</td><td></td><td></td><td>mloU R1@.3</td><td>R1@.5 R1@.7</td><td></td><td>mloU R1@.3</td><td></td><td>R1@.5</td><td>R1@.7</td><td>mloU</td></tr><tr><td colspan="10">Proprietary models</td><td colspan="7"></td></tr><tr><td>GPT-5 [31]</td><td>--</td><td>59.3</td><td>42.0</td><td>22.0</td><td>40.5</td><td>57.4</td><td>44.9</td><td>30.4</td><td>42.9</td><td>72.4</td><td>60.4</td><td>46.4</td><td>56.8</td><td>46.7</td></tr><tr><td>Gemini-2.5-Pro [13]</td><td>--</td><td>74.1</td><td>61.1</td><td>34.0</td><td>52.8</td><td>72.3</td><td>64.2</td><td>47.1</td><td>58.1</td><td>84.1</td><td>75.9</td><td>61.1</td><td>70.4</td><td>60.4</td></tr><tr><td colspan="10">Open-source dense models (4B language backbone)</td><td colspan="7"></td></tr><tr><td>Molmo2-4B [8]</td><td>0.636</td><td>44.4</td><td>31.1</td><td>18.5</td><td>34.7</td><td>50.8</td><td>40.3</td><td>29.9</td><td>40.9</td><td>73.7</td><td>62.6</td><td>51.8</td><td>60.8</td><td>45.5</td></tr><tr><td>Qwen3-VL-4B [3]</td><td>0.703</td><td>71.8</td><td>53.8</td><td>28.7</td><td>49.4</td><td>67.4</td><td>54.8</td><td>37.6</td><td>50.7</td><td>78.4</td><td>66.4</td><td>51.1</td><td>62.3</td><td>54.1</td></tr><tr><td>Mage-VL [45]</td><td>1.437</td><td>75.5</td><td>63.1</td><td>36.1</td><td>55.0</td><td>65.1</td><td>57.0</td><td>39.7</td><td>51.6</td><td>76.4</td><td>68.1</td><td>53.6</td><td>63.4</td><td>56.7</td></tr><tr><td>Qwen3.5-4B [33]</td><td>0.763</td><td>75.2</td><td>56.6</td><td>31.5</td><td>52.3</td><td>72.6</td><td>61.6</td><td>42.6</td><td>55.6</td><td>82.9</td><td>73.1</td><td>58.1</td><td>67.9</td><td>58.6</td></tr><tr><td>VideoChat3-4B [25]</td><td>1.067</td><td>78.4</td><td>64.9</td><td>35.9</td><td>56.1</td><td>70.1</td><td>60.5</td><td>42.2</td><td>54.6</td><td>81.0</td><td>72.5</td><td>56.8</td><td>67.0</td><td>59.2</td></tr><tr><td>Video-ORA-4B [26]</td><td>2.347</td><td>77.1</td><td>64.9</td><td>37.9</td><td>56.8</td><td>69.3</td><td>60.1</td><td>43.8</td><td>57.3</td><td>81.4</td><td>72.6</td><td>58.6</td><td>67.5</td><td>60.5</td></tr><tr><td>TimeLens2-4B [61]</td><td>0.646</td><td>80.3</td><td>66.6</td><td>37.5</td><td>57.7</td><td>74.6</td><td>65.5</td><td>47.2</td><td>59.0</td><td>83.3</td><td>75.2</td><td>60.0</td><td>69.3</td><td>62.0</td></tr><tr><td>VideoJev-4B</td><td>0.276</td><td>82.8</td><td>69.4</td><td>40.3</td><td>60.1</td><td>78.8</td><td>70.5</td><td>51.4</td><td>62.8</td><td>87.3</td><td>79.4</td><td>64.8</td><td>73.2</td><td>65.3</td></tr><tr><td colspan="10">Open-source MoE models (30-35B total, 3B active)</td><td colspan="7"></td></tr><tr><td>Nemotron-3-Nano-Omni-30B-A3B [30]</td><td>3.882</td><td>65.2</td><td>49.9</td><td>26.6</td><td>45.3</td><td>54.7</td><td>43.5</td><td>27.9</td><td>40.6</td><td>66.7</td><td>55.7</td><td>41.2</td><td>52.6</td><td>46.2</td></tr><tr><td>Qwen3-VL-30B-A3B [3]</td><td>1.055</td><td>70.3</td><td>46.5</td><td>25.1</td><td>48.1</td><td>65.7</td><td>51.1</td><td>36.5</td><td>49.4</td><td>79.3</td><td>67.6</td><td>52.6</td><td>63.2</td><td>53.6</td></tr><tr><td>Qwen3.5-35B-A3B [33]</td><td>1.021</td><td>71.5</td><td>44.2</td><td>21.1</td><td>46.0</td><td>72.1</td><td>57.3</td><td>40.4</td><td>54.4</td><td>80.3</td><td>67.3</td><td>53.2</td><td>64.3</td><td>54.9</td></tr><tr><td>Qwen3.6-35B-A3B [34]</td><td>1.054</td><td>72.6</td><td>47.0</td><td>23.2</td><td>47.3</td><td>73.9</td><td>59.8</td><td>42.4</td><td>56.0</td><td>83.3</td><td>72.5</td><td>57.6</td><td>67.5</td><td>56.9</td></tr><tr><td>VideoJev-35B-A3B</td><td>0.336</td><td>86.1</td><td>74.8</td><td>44.5</td><td>63.7</td><td>81.0</td><td>72.5</td><td>54.4</td><td>65.5</td><td>88.0</td><td>79.9</td><td>65.5</td><td>74.0</td><td>67.7</td></tr></table>

## 3.1 Experimental Setup

Training configuration. We train both model sizes on 64 NVIDIA H200 GPUs with BF16 computation and a global batch size of 64 packed sequences. Training videos are sampled at 2 fps, with a maximum sequence length of 144K tokens and an 80% visual-token budget. We use AdamW with weight decay 0.01 and gradient clipping at 1.0. Each stage uses 5% warm-up followed by cosine decay to 10% of its peak learning rate. The head learning rate is $3 \times 1 0 ^ { - 4 }$ during head alignment and $1 0 ^ { - 4 }$ during joint adaptation. In the latter stage, the language-backbone learning rate is $5 \times 1 0 ^ { - 6 }$ for 4B and $2 \times 1 0 ^ { - 6 }$ for 35B-A3B. The two stages run for 160 and 375 optimizer updates, respectively. We use DeepSpeed ZeRO-2 for 4B and ZeRO-3 for 35B-A3B.

Temporal grounding benchmarks and metrics. We evaluate on the TimeLens-Bench refined annotations [55] of Charades-STA [12], ActivityNet Captions [19], and QVHighlights [21]. For each query, we report rank-one recall at temporal IoU thresholds 0.3, 0.5, and 0.7 (R1@0.3/0.5/0.7), together with mean temporal IoU (mIoU). VideoJev uses a fixed decoding threshold τ = 0.25 and smoothing window 1 across both model sizes and all inference modes.

Video QA benchmarks and metrics. We evaluate multiple-choice Video QA on the LVBench [41] test set (1,549 questions), Video-MME [11] test set (2,700 questions), LongVideoBench [42] validation set (1,337 questions), and the seven multiple-choice tasks of MLVU [57] development (2,174 questions). Inputs contain video frames only, without subtitles or audio. We report accuracy on LVBench, Video-MME, and LongVideoBench, and M-Avg on MLVU.

Inference latency. We measure grounding latency on the complete Charades test set (1,313 videos; 3,363 queries) and QA latency on the complete LVBench test set (103 videos; 1,549 questions), using H200 GPUs at batch size 1. Both tasks measure model forward time only, excluding preprocessing and data transfers. VideoJev uses Joint inference, encoding each video once and scoring its queries together. Generative baselines share visual encoding and prefix prefill, then generate answers from independent query branches. We divide accumulated forward time, including visual encoding, prefill, and all autoregressive decoding steps, by the number of queries. Generations reaching the output limit are included. Grounding uses 4 fps inputs and at most $6 4 0 \times 6 4 0$ pixels per frame; QA models retain their native visual settings. Appendix A.1 gives the timing details.

## 3.2 Video Temporal Grounding

Table 1 shows consistent gains across the three grounding benchmarks. VideoJev-4B reaches 65.3 mean mIoU, improving on Qwen3.5-4B by 6.7 points and TimeLens2-4B by 3.3 points. Against TimeLens2-4B, it also raises R1@0.7 by 2.8–4.8 points across the three datasets. Gains under this stricter overlap criterion indicate more precise temporal localization, supporting interval membership scoring as an efective way to recover event spans.

VideoJev-35B-A3B raises mean mIoU to 67.7, 10.8 points above its Qwen3.6-35B-A3B backbone. Scaling from VideoJev-4B adds 2.4 points with only a 22% increase in measured latency. Both sizes lead all reported grounding metrics and achieve the lowest measured latency within their respective open-source groups, with 2.8–3.1× speedups over their backbones. Since the generative baselines also share video prefill, these speedups highlight the value of avoiding autoregressive timestamp generation when repeated queries draw on the same visual context.

## 3.3 Option Selection on Video QA Benchmarks

All VideoJev long-video QA evaluations in this paper use a fixed 256K (262,144-token) context budget across benchmarks, model sizes, and inference modes. We sample the full video at a requested 2 fps, uniformly subsampling when needed to cap the input at 2,048 frames. Frames contain at most 640 × 640 pixels; resolution is reduced as needed to keep visual tokens within 80% of the context budget. Table 2 groups baselines by language-backbone scale and distinguishes Instruct and Thinking modes where reported.

In Table 2, VideoJev-4B and VideoJev-35B-A3B improve four-benchmark mean scores over their Instruct backbones by 8.0 and 5.7 points, respectively. Both lead their open-source groups on LVBench, and the MoE model achieves the highest group mean. The gains extend to every QA benchmark.

Direct scoring also remains competitive with explicit reasoning. VideoJev-4B is within 0.6 points of Qwen3.5- 4B Thinking in mean score, while VideoJev-35B-A3B exceeds Qwen3.6-35B-A3B Thinking by 0.7 points. Both outperform their Thinking counterparts on LVBench and match or improve on LongVideoBench; Thinking retains an edge on Video-MME and MLVU. Measured speedups over these Thinking baselines are 129× and 100×, respectively. Section 3.5 further compares direct-option and reasoning baselines.

## 3.4 Ablation Study

We control for training data, then examine inference modes, task composition, supervision, and design choices. Architectural, objective, and schedule ablations use VideoJev-4B with the same complete test sets, Joint inference, and grounding decoder. QA ablations and the autoregressive control retain the fixed 256K-token context; tables report changes from the marked baseline.

Isolating VideoJev’s architectural gains. To test whether the gains arise solely from our training data, we fine-tune an ordinary Qwen3.5-4B autoregressive model on exactly the same post-training data as VideoJev. It retains the native final MLP and language-model head and learns to generate temporal intervals or answer letters. Both models use single-stage training on 100% of the data; VideoJev results are copied from the single-stage variant in Table 8. Table 3 shows that VideoJev improves mean grounding and QA scores by 1.3 and 1.9 points. The gains under matched data and schedules support an architectural benefit: directly supervising temporal membership and option preferences aligns the readout with the required decisions, enabling more efective use of the same supervision. Appendix A.3 gives the key settings.

Cached inference and Joint inference. Cached inference evaluates each query independently against the same video prefix, providing a reference without interference between queries. Joint inference eficiently scores all queries in a group in one forward pass. To retain the benefits of optimized kernels such as FlashAttention, it uses standard causal attention without custom masks to isolate queries. Later queries can therefore attend to preceding questions and candidates. We verified that earlier questions do not leak answers to later ones; no reference or predicted answers are inserted. Table 4 tests whether these interactions degrade prediction quality using the same trained checkpoint within each size. Surprisingly, Joint inference improves mean grounding/QA scores by 0.8/0.5 points at 4B and 0.3/0.7 points at 35B-A3B. Scores improve in 13 of the 14 model–benchmark comparisons, with only a 0.1-point decrease on QVHighlights at 35B-A3B. These small gains may reflect alignment with training, which uses the same joint layout for eficiency: the model learns to accommodate query interactions while sharing visual computation.

Table 2 Multiple-choice Video QA. Molmo2 uses a 32K context; Nemotron uses 128K. Bold marks the best value and underline the second best. Appendix A details the evaluation settings.
<table><tr><td>Model</td><td>Latency (↓) seconds</td><td>LVBench test; w/o sub.</td><td>test; w/o sub.</td><td>Video-MME LongVideoBench val; w/o sub.</td><td>MLVU dev MCQ; M-Avg</td><td>Avg. 4-bench mean</td></tr><tr><td colspan="7">Proprietary models</td></tr><tr><td>GPT-5.2 [33]</td><td></td><td>73.7</td><td>85.8</td><td>--</td><td>85.6</td><td></td></tr><tr><td>Gemini-3-Pro [33]</td><td></td><td>76.2</td><td>87.7</td><td>--</td><td>83.0</td><td></td></tr><tr><td>Claude-Opus-4.5 [33]</td><td></td><td>57.3</td><td>81.4</td><td>--</td><td>81.7</td><td></td></tr><tr><td>Seed2.0-Pro [4]</td><td>--</td><td>76.4</td><td>--</td><td>80.3</td><td>--</td><td>--</td></tr><tr><td colspan="7">Open-source dense models (4B language backbone)</td></tr><tr><td>Molmo2-4B [8]</td><td>0.748</td><td>53.9</td><td>69.6</td><td>68.0</td><td>74.3</td><td>66.5</td></tr><tr><td>VideoChat3-4B [25]</td><td>1.210</td><td>56.7</td><td>70.1</td><td>64.1</td><td>73.8</td><td>66.2</td></tr><tr><td>Qwen3-VL-4BInstruct [3]</td><td>3.012</td><td>56.2</td><td>69.3</td><td>66.5</td><td>75.3</td><td>66.8</td></tr><tr><td>Qwen3-VL-4BThinking [3]</td><td>19.713</td><td>53.5</td><td>68.9</td><td>63.4</td><td>75.7</td><td>65.4</td></tr><tr><td>Qwen3.5-4BInstruct [33]</td><td>1.355</td><td>60.0</td><td>70.6</td><td>61.2</td><td>73.3</td><td>66.3</td></tr><tr><td>Qwen3.5-4BThinking [33]</td><td>144.929</td><td>66.4</td><td>76.9</td><td>73.6</td><td>82.8</td><td>74.9</td></tr><tr><td>VideoJev-4B</td><td>1.123</td><td>67.9</td><td>75.0</td><td>73.6</td><td>80.6</td><td>74.3</td></tr><tr><td colspan="7">Open-source MoE models (30-35B total, 3B active)</td></tr><tr><td>Nemotron-3-Nano-Omni-30B-A3BInstruct [30]</td><td>1.106</td><td>54.9</td><td>70.8</td><td>67.5</td><td>77.2</td><td>67.6</td></tr><tr><td>Nemotron-3-Nano-Omni-30B-A3BThinking [30]</td><td>229.095</td><td>52.2</td><td>72.2</td><td>69.6</td><td>79.6</td><td>68.4</td></tr><tr><td>Qwen3-VL-30B-A3BInstruct [3]</td><td>4.817</td><td>62.5</td><td>74.5</td><td>70.9</td><td>81.3</td><td>72.3</td></tr><tr><td>Qwen3-VL-30B-A3BThinking [3]</td><td>9.637</td><td>59.2</td><td>73.3</td><td>69.1</td><td>78.9</td><td>70.1</td></tr><tr><td>Qwen3.5-35B-A3BInstruct [33]</td><td>1.683</td><td>58.8</td><td>74.4</td><td>65.7</td><td>77.4</td><td>69.1</td></tr><tr><td>Qwen3.5-35B-A3BThinking [33]</td><td>164.730</td><td>71.4</td><td>82.5</td><td>74.5</td><td>85.6</td><td>78.5</td></tr><tr><td>Qwen3.6-35B-A3BInstruct [34]</td><td>2.379</td><td>66.0</td><td>77.1</td><td>72.4</td><td>79.5</td><td>73.8</td></tr><tr><td>Qwen3.6-35B-A3BThinking [34]</td><td>149.777</td><td>71.4</td><td>82.5</td><td>74.9</td><td>86.2</td><td>78.8</td></tr><tr><td>VideoJev-35B-A3B</td><td>1.493</td><td>73.4</td><td>81.8</td><td>77.1</td><td>85.9</td><td>79.5</td></tr></table>

Table 3 Isolating architectural gains with an autoregressive control. VideoJev-4B and an autoregressive Qwen3.5-4B are both fine-tuned in a single stage on exactly the same post-training data. † denotes the VideoJev-4B Single-stage (100%) variant from Table 8. The two models share the same training and evaluation settings.
<table><tr><td rowspan="2">Model</td><td colspan="4">Temporal grounding</td><td colspan="4">Video QA</td></tr><tr><td>CharadesTL</td><td>ActivityNetTL QVHighlightsTL Avg.</td><td></td><td></td><td></td><td></td><td>LVBench Video-MME LongVideoBench MLVU</td><td>Avg.</td></tr><tr><td>VideoJev-4B† (baseline)</td><td>60.6</td><td>62.8</td><td>72.9</td><td>65.4</td><td>67.7</td><td>74.9</td><td>73.5 82.6</td><td>74.7</td></tr><tr><td rowspan="2">Qwen3.5-4B AR SFT</td><td>61.0</td><td>61.3</td><td>70.0</td><td>64.1</td><td>66.9</td><td>74.5</td><td>71.4 78.6</td><td>72.8</td></tr><tr><td>(+0.4)</td><td>(-1.5)</td><td>(-2.9)</td><td>(-1.3)</td><td>(-0.8)</td><td>(-0.4) (-2.1)</td><td>(-4.0)</td><td>(-1.9)</td></tr></table>

Table 4 Cached inference versus Joint inference.
<table><tr><td rowspan="2">Setting</td><td colspan="4">Temporal grounding</td><td colspan="4">Video QA</td></tr><tr><td>CharadesTL</td><td>ActivityNetTL QVHighlightsTL</td><td></td><td>Avg.</td><td></td><td>LVBench Video-MME LongVideoBench MLVU Avg.</td><td></td><td></td></tr><tr><td>VideoJev-4B</td><td></td><td></td><td></td><td></td><td></td><td>75.0</td><td></td><td></td></tr><tr><td>Joint inference (baseline)</td><td>60.1</td><td>62.8</td><td>73.2</td><td>65.3</td><td>67.9</td><td>74.19</td><td>73.6 72.99</td><td>80.6 74.3 80.1</td></tr><tr><td>Cached inference</td><td>5-18.</td><td>(1.1.8</td><td>73.111</td><td>64.5 (-0.8)</td><td>67.11)</td><td></td><td></td><td>73.8 (-0.5)</td></tr><tr><td>VideoJev-35B-A3B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Joint inference (baseline)</td><td>63.7</td><td>65.5</td><td>74.0</td><td>67.7</td><td>73.4</td><td>81.8</td><td>77.1</td><td>85.9 79.5</td></tr><tr><td></td><td>6 3.1</td><td>65 5)</td><td></td><td></td><td></td><td></td><td></td><td>78.8</td></tr><tr><td>Cached inference</td><td></td><td></td><td>74..1</td><td>67.4</td><td>72.22</td><td>81.33</td><td>76.7</td><td>(-0.7)</td></tr></table>

Joint training for grounding and QA. Temporal grounding and video QA require diferent forms of prediction: localizing an event in time and selecting a textual answer. We therefore examine whether the two tasks interfere, remain independent, or reinforce each other within VideoJev. Table 5 compares VideoJev-4B trained jointly on both tasks with models trained on either task alone. Joint training improves performance on all seven benchmarks, raising mean grounding mIoU by 0.7 points and mean QA score by 1.3 points, with the largest QA gain on LongVideoBench (+1.9 points). These gains indicate mutual benefit: temporal supervision may sharpen evidence localization, while QA strengthens semantic discrimination. The shared scorer provides a simple interface for learning these complementary capabilities.

Table 5 Joint versus single-task training. VideoJev-4B is trained on temporal grounding (TG), video question answering (QA), or both.
<table><tr><td rowspan="2">Setting CharadesTL</td><td rowspan="2">Temporal grounding</td><td colspan="3"></td><td colspan="5">Video QA</td></tr><tr><td></td><td>ActivityNetTL QVHighlightsTL Avg.</td><td></td><td></td><td>LVBench Video-MME LongVideoBench MLVU</td><td></td><td></td><td>Avg.</td></tr><tr><td>TG+QA (baseline)</td><td>60.1</td><td>62.8</td><td>73.2</td><td>65.3</td><td>67.9</td><td>75.0</td><td>73.6</td><td>80.6</td><td>74.3</td></tr><tr><td rowspan="2">TG-only</td><td>59.6</td><td>61.8</td><td>72.5</td><td>64.6</td><td></td><td></td><td></td><td>--</td><td></td></tr><tr><td>(-0.5)</td><td>(-1.0)</td><td>(-0.7)</td><td>(-0.7)</td><td>--</td><td>--</td><td>--</td><td></td><td>--</td></tr><tr><td>QA-only</td><td>--</td><td></td><td>--</td><td></td><td>66.3 (-1.6)</td><td>74.9 (-0.1)</td><td>71.7 (-1.9)</td><td>79.2 (-1.4)</td><td>73.0 (-1.3)</td></tr></table>

Auxiliary supervision. Correct local judgments must also cover the event as a whole. Table 6 tests the value of adding Soft IoU to the grounding objective in Eq. (7), while QA retains cross-entropy. Soft IoU raises mean grounding mIoU from 60.7 to 65.3 (+4.6 points) and mean QA score from 73.9 to 74.3 (+0.4 points), with improvements on all seven benchmarks. These results highlight the value of event-level coverage supervision and suggest that better temporal learning also benefits the shared representations used for QA.

Table 6 Effect of Soft IoU supervision in VideoJev-4B. Soft IoU augments the grounding BCE objective; QA retains cross-entropy. Deltas are relative to BCE/CE only (baseline). Both variants use two-stage training.
<table><tr><td rowspan="2">Setting</td><td colspan="4">Temporal grounding</td><td colspan="5">Video QA</td></tr><tr><td>CharadesTL</td><td>ActivityNetTL QVHighlightsTL Avg.</td><td></td><td></td><td></td><td></td><td>LVBench Video-MME LongVideoBench MLVU</td><td></td><td>Avg.</td></tr><tr><td>BCE/CE only (baseline)</td><td>53.3</td><td>58.8</td><td>70.0</td><td>60.7</td><td>67.4</td><td>74.6</td><td>72.9</td><td>80.5</td><td>73.9</td></tr><tr><td rowspan="2">+ Soft loU</td><td>60.1</td><td>62.8</td><td>73.2</td><td>65.3</td><td>67.9</td><td>75.0</td><td>73.6</td><td>80.6</td><td>74.3</td></tr><tr><td>(+6.8)</td><td>(+4.0)</td><td>(+3.2)</td><td>(+4.6)</td><td>(+0.5)</td><td>(+0.4)</td><td>(+0.7)</td><td>(+0.1)</td><td>(+0.4)</td></tr></table>

Final-layer MLP removal. The final MLP adds a transformation before candidate scoring, but Table 7 shows that it is not needed to maintain prediction quality. Removing it raises mean grounding and QA scores by 0.2 points each, leaving overall performance essentially unchanged. Following Occam’s razor, we adopt the simpler readout: pre-MLP features preserve both capabilities while avoiding an additional layer of computation.

Table 7 Effect of the final-layer MLP in VideoJev-4B.
<table><tr><td rowspan="2">Setting</td><td colspan="4">Temporal grounding</td><td colspan="4">Video QA</td></tr><tr><td>CharadesTL</td><td>ActivityNetTL QVHighlightsTL Avg.</td><td></td><td></td><td>LVBench Video-MME LongVideoBench MLVU</td><td></td><td></td><td>Avg.</td></tr><tr><td>Remove final MLP (baseline)</td><td>60.1</td><td>62.8</td><td>73.2</td><td>65.3</td><td>67.9</td><td>75.0 73.6</td><td>80.6</td><td>74.3</td></tr><tr><td>Retain final MLP</td><td>60.5</td><td>62.1</td><td>72.)</td><td>65.1</td><td>67.9</td><td>74.3 73.0</td><td>80.8</td><td>74.1</td></tr><tr><td></td><td>(+0.4)</td><td></td><td></td><td>(-0.2)</td><td>(0.0)</td><td>(-0.6)</td><td>(+0.2)</td><td>(-0.2)</td></tr></table>

Two-stage adaptation. The two-stage schedule first aligns the scorer with frozen features, then adapts the language backbone and scorer together. Table 8 shows broadly comparable performance to single-stage training: two-stage adaptation scores slightly higher on four of seven benchmarks, while single-stage training leads by only 0.1 point in mean grounding mIoU and 0.4 points in mean QA, mainly due to MLVU. Twostage adaptation achieves this quality while updating the language backbone on only 70% of the videos; the remaining 30% use cheaper head-only training. We adopt this schedule for its lower cost and comparable performance; aligning the scorer first may also stabilize backbone adaptation.

Table 8 Two-stage versus single-stage adaptation of VideoJev-4B. The baseline warms up the head on 30% of videos and jointly adapts on the remaining 70%; single-stage training directly adapts on all videos with the full objectives.
<table><tr><td rowspan="2">Setting</td><td colspan="4">Temporal grounding</td><td colspan="5">Video QA</td></tr><tr><td>Charades™L</td><td>ActivityNetTL</td><td>QVHighlightsTL Avg.</td><td></td><td></td><td></td><td>LVBench Video-MME LongVideoBench MLVU Avg.</td><td></td><td></td></tr><tr><td rowspan="2">Two-stage (baseline)</td><td>60.1</td><td>62.8</td><td>73.2</td><td>65.3</td><td>67.9</td><td>75.0</td><td>73.6</td><td>80.6</td><td>74.3</td></tr><tr><td>60.6</td><td>62.8</td><td>72.9</td><td>65.4</td><td>67.7</td><td>74.9</td><td>73.5</td><td>82.6</td><td>74.7</td></tr><tr><td>Single-stage (100%)</td><td>(+0.5)</td><td>(0.0)</td><td>(-0.3)</td><td>(+0.1)</td><td></td><td>(-0.1)</td><td></td><td>(+2.0)</td><td>(+0.4)</td></tr></table>

## 3.5 Analysis

Query scaling and inference eficiency. Figure 3 compares the computational eficiency of Cached inference and Joint inference as the number of queries about the same video increases. VideoJev supports both modes. Autoregressive QA supports joint processing in Direct mode, which predicts exactly one option token without reasoning. Multi-token outputs, including temporal intervals and reasoning traces, require sequential token feedback and use Cached inference. Appendix A.4 details these modes.

![](images/e27464c225940772291620f1c107dfd8f2deea7ddb47681ef038b619a9c7b2ff.jpg)  
(b) Question answering

![](images/e52b1b71c319f702f3245ee34cdbd0e6f4e03ebf86a4d3564d8d62f979e87310.jpg)  
Figure 3 Multi-query efficiency at 4B. Mean forward time for 1–10 queries per video on Charades (a) and Video-Holmes [6] (b); shading shows 95% confidence intervals. Solid/dotted lines use Cached inference (dotted: multitoken generation); dashed lines use Joint inference. End labels pair ten-query latency with separately evaluated full-benchmark Charades mIoU or LVBench accuracy; quality settings follow the source evaluations in Appendix A.2.

We measure 1–10 queries about the same video at batch size 1, counting visual encoding and prefix prefill once. For each task, we randomly sample 40 videos within duration strata. End labels report full-benchmark Charades mIoU and LVBench accuracy. Quality is evaluated separately from timing; Appendices A.1 and A.2 specify the respective protocols and remaining score sources.

VideoJev’s Joint inference adds only 55 ms for grounding and 31 ms for QA as query count grows from one to ten. At ten queries, forward times are 0.39 s and 1.64 s, versus 0.85 s and 2.17 s under Cached inference, while independently measured Charades mIoU and LVBench accuracy improve by 1.5 and 0.8 points, respectively. The autoregressive Direct QA baseline benefits less from joint processing: its accuracy drops from 60.0% under Cached inference to 58.7% under Joint inference, below VideoJev’s 67.9% at comparable latency (1.67 s versus 1.64 s). Multi-token generation remains costly: Qwen3.5-4B Instruct takes 5.98 s for ten grounding queries at 52.3 mIoU, versus VideoJev’s 0.39 s and 60.1 mIoU; Qwen Thinking takes 1,393 s for ten QA queries at 66.4% accuracy. The small marginal cost comes without a loss in prediction quality.

![](images/942fd87f2bdd8e0f4fc2da4246b867cf3143531d81537dc6cb1be0c95b9ddbea.jpg)  
Figure 4 Confidence calibration on Video QA. Points show mean confidence and observed accuracy in equalwidth bins containing at least 20 questions; all bins contribute to ECE. The dashed diagonal denotes perfect calibration. MLVU uses seven-task-balanced weights; the other benchmarks weight questions equally. Curves use the reported per-benchmark predictions.

Confidence and calibration. Candidate scores also provide a signal for deciding when to trust a prediction. We examine the highest option probability, max<sub>i</sub> p<sub>i</sub>, using the predictions from Table 2. Higher confidence generally accompanies higher accuracy (Figure 4). On LVBench, the 35B-A3B model reaches 95.8% accuracy on its 591 predictions with confidence at least 99%, compared with 73.4% overall. This separation suggests potential for selecting reliable answers or escalating uncertain cases.

Calibration requires confidence to match the frequency of correct predictions [14, 15]. Jev explicitly optimizes this property through reinforcement learning [2]; VideoJev receives no calibration-specific training. Its probabilities tend to overestimate correctness: the four-benchmark mean expected calibration error (ECE; ten equal-width bins) is 8.65 and 8.13 percentage points for 4B and 35B-A3B. Temperature scaling on an independent calibration set [15] or calibration-aware training could address this gap.

## 4 Related Work

MLLM video understanding. VideoChat [22] and Video-LLaMA [53] connect visual representations to language models for instruction-driven video understanding. VideoChat2 [23] and VideoChat3 [25] expand these capabilities through multimodal training and diverse supervision. For long videos, VideoChat-Flash [24] compresses visual tokens, while Video-o3 [51] interleaves reasoning with visual clue seeking. VideoLLMonline [5] supports streaming interaction with continuous cache reuse, and StreamingVLM [44] maintains a bounded cache for long-running video streams. JoyAI-VL-Interaction [46] learns when to remain silent, respond, or delegate during real-time interaction, while OneStreamer [50] jointly models streaming perception, memory formation, and proactive response. Complementary work studies vision-language transfer in AWT [59] and eficient test-time prompt adaptation [63]. Benchmarks probe compositional understanding in TVQA [20], temporal and causal reasoning in NExT-QA [43], long-horizon understanding in EgoSchema [29], and broader video skills in MVBench and Video-MME [11, 23]. VideoJev focuses on direct decisions over time and options, sharing an encoded video context across multiple queries.

Temporal grounding. Temporal grounding localizes language-described events through candidate matching [12, 56], span prediction [54], moment-set prediction [21], or clip-level foreground, boundary, and saliency outputs [27]. In temporal action detection, DualDETR [60] jointly refines instance and boundary queries.

TimeChat and VTimeLLM bring temporal grounding into video MLLMs [16, 35]; LITA and Grounded VideoLLM introduce explicit temporal tokens [17, 39]. TimeSuite uses grounded tuning to improve long-video understanding [49], and TimeLens2 advances generalist temporal modeling [61]. DisTime predicts boundary distributions [52], while controlled comparisons examine temporal output designs [18]. Recent analysis also extracts temporal evidence from prefill attention to guide subsequent answer generation [10]. Grounding and QA have already been combined: GroundVQA uses a temporal localizer and an answer decoder [9], while SEAL learns compact representations for both tasks [40]. VideoJev uses a shared scoring head for temporal membership and textual options, decoding a contiguous span from temporal scores.

System One models. Jev [2, 38] introduces typed probabilistic judgments, while this-that-model-1.0 and JevLite explore one-pass decision readouts [7, 36]. Visual Jev [48] evaluates multiple forced-choice questions over a shared image, and PixelJev [58] supports image recognition and multiple-choice VQA. FreeRet [62] extracts multimodal representations and reranks candidates using an MLLM’s multiple-choice probabilities; its reranker can be viewed as a training-free route to Jev-style relevance judgments. Related video studies assess quality from metadata and derived features [37] or anomalies from captions and summaries [32]. VideoJev extends this paradigm to temporal decisions and repeated video queries, aligning multi-query training with inference.

## 5 Future Work

Toward general-purpose visual decisions. Beyond temporal grounding and multiple-choice Video QA, tasks such as action recognition, event detection, scene-state classification, and visual condition verification can also be formulated as decisions over candidate hypotheses. Extending the candidate interface to these tasks and output structures could reduce reliance on task-specific models and prediction heads.

From closed-set decisions to open-ended agent workflows. A higher-level agent could decompose open-ended goals into visual checks using standard operating procedures or generated plans. It could propose candidate events, actions, or states for VideoJev to evaluate, then use these judgments to guide planning and execution. Ambiguous or unexpected observations could trigger candidate revision or further reasoning. Here, System Two organizes the decision space, while System One evaluates it against visual evidence. Applications include procedural monitoring, interactive assistance, embodied agents, and computer-use systems.

Toward streaming and reliable visual interaction. Streaming inputs will require incremental visual encoding, eficient memory management, and timely updates to the decision context. Reliability also needs improvement: higher confidence generally accompanies higher Video QA accuracy, but predicted probabilities tend to overestimate correctness. Calibration-aware training, uncertainty estimation, and selective prediction could help determine when to accept System One decisions or defer to higher-level reasoning.

## 6 Conclusion

We introduced VideoJev, a family of video System One models that unifies temporal grounding and multiplechoice video QA through direct candidate scoring. Cached inference and Joint inference make repeated judgments eficient by reusing visual computation. Across seven benchmarks, dense 4B and MoE 35B-A3B models achieve strong prediction quality and inference eficiency. Ablations show that these gains extend beyond training data alone and that the two tasks benefit from joint learning. VideoJev thus ofers a foundation for visual agents that compose structured judgments into complex, open-ended workflows.

## References

[1] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jefrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina

Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. Do As I Can, Not As I Say: Grounding Language in Robotic Afordances, 2022. https://arxiv.org/abs/2204.01691.

[2] Diogo Almeida. Introducing System One Models & Jev. TypeSafe AI Blog, n.d. https://typesafe.ai/blog/ introducing-system-one-models-and-jev. Accessed 2026-10-12.

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-VL Technical Report, 2025. https://arxiv.org/abs/2511.21631.

[4] Bytedance Seed. Seed2.0 Model Card: Towards Intelligence Frontier for Real-World Complexity, 2026. https://lf3-static.bytednsdoc.com/obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/seed2/0214/Seed2. 0%20Model%20Card.pdf.

[5] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. VideoLLM-online: Online Video Large Language Model for Streaming Video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18407–18418, June 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Chen\_ VideoLLM-online\_Online\_Video\_Large\_Language\_Model\_for\_Streaming\_Video\_CVPR\_2024\_paper.html.

[6] Junhao Cheng, Yuying Ge, Teng Wang, Yixiao Ge, Jing Liao, and Ying Shan. Video-Holmes: Can MLLM Think Like Holmes for Complex Video Reasoning?, 2025. https://arxiv.org/abs/2505.21374.

[7] Zehua Cheng, Wei Dai, and Jiahao Sun. this-that-model-1.0: A typed decision model that decides in 30 ms, for a millionth of a cent, 2026. https://arxiv.org/abs/2609.23886.

[8] Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Mohammadreza Salehi, Rohun Tripathi, Sangho Lee, Zhongzheng Ren, Chris Dongjoo Kim, Yinuo Yang, Vincent Shao, Yue Yang, Weikai Huang, Ziqi Gao, Taira Anderson, Jianrui Zhang, Jitesh Jain, George Stoica, Winson Han, Ali Farhadi, and Ranjay Krishna. Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding, 2026. https://arxiv.org/abs/2601.10611.

[9] Shangzhe Di and Weidi Xie. Grounded Question-Answering in Long Egocentric Videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12934– 12943, June 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Di\_Grounded\_Question-Answering\_ in\_Long\_Egocentric\_Videos\_CVPR\_2024\_paper.html.

[10] Dazhao Du, Liao Duan, Jian Liu, Tao Han, Yujia Zhang, Eric Liu, Xi Chen, and Song Guo. MLLMs Know When Before Speaking: Revealing and Recovering Temporal Grounding via Attention Cues, 2026. https://arxiv.org/ abs/2605.21954.

[11] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, Ran He, and Xing Sun. Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24108–24118, June 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Fu\_Video-MME\_The\_First-Ever\_Comprehensive\_ Evaluation\_Benchmark\_of\_Multi-modal\_LLMs\_in\_CVPR\_2025\_paper.html.

[12] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. TALL: Temporal Activity Localization via Language Query. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017. https: //openaccess.thecvf.com/content\_iccv\_2017/html/Gao\_TALL\_Temporal\_Activity\_ICCV\_2017\_paper.html.

[13] Gemini 2.5 Team. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities, June 2025. https://storage.googleapis.com/deepmind-media/gemini/ gemini\_v2\_5\_report.pdf.

[14] Tilmann Gneiting, Fadoua Balabdaoui, and Adrian E. Raftery. Probabilistic Forecasts, Calibration and Sharpness. Journal of the Royal Statistical Society Series B: Statistical Methodology, 69(2):243–268, March 2007. ISSN 1467- 9868. doi: 10.1111/j.1467-9868.2007.00587.x. http://dx.doi.org/10.1111/j.1467-9868.2007.00587.x.

[15] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On Calibration of Modern Neural Networks. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR, 06–11 Aug 2017. https://proceedings.mlr.press/v70/guo17a.html.

[16] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. VTimeLLM: Empower LLM to Grasp Video Moments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recogni tion (CVPR), pages 14271–14280, June 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Huang\_ VTimeLLM\_Empower\_LLM\_to\_Grasp\_Video\_Moments\_CVPR\_2024\_paper.html.

[17] De-An Huang, Shijia Liao, Subhashree Radhakrishnan, Hongxu Yin, Pavlo Molchanov, Zhiding Yu, and Jan Kautz. LITA: Language Instructed Temporal-Localization Assistant, 2024. https://arxiv.org/abs/2403.19046.

[18] Shengji Jin, Yuanhao Zou, Victor Zhu, Zhengping Ji, and Chen Chen. How Should Video LLMs Output Time? An Analysis of Eficient Temporal Grounding Paradigms, 2026. https://arxiv.org/abs/2604.08966.

[19] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-Captioning Events in Videos. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017. https://openaccess.thecvf.com/content\_iccv\_2017/html/Krishna\_Dense-Captioning\_Events\_in\_ICCV\_ 2017\_paper.html.

[20] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara Berg. TVQA: Localized, Compositional Video Question Answering. In Ellen Rilof, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1369–1379, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1167. https: //aclanthology.org/D18-1167/.

[21] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting Moments and Highlights in Videos via Natural Language Queries. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 11846–11858. Curran Associates, Inc., 2021. https: //proceedings.neurips.cc/paper\_files/paper/2021/file/62e0973455fd26eb03e91d5741a4a3bb-Paper.pdf.

[22] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. VideoChat: Chat-Centric Video Understanding. arXiv preprint arXiv:2305.06355, 2023. https://arxiv.org/ abs/2305.06355.

[23] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. MVBench: A Comprehensive Multi-modal Video Understanding Benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22195–22206, June 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Li\_MVBench\_ A\_Comprehensive\_Multi-modal\_Video\_Understanding\_Benchmark\_CVPR\_2024\_paper.html.

[24] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, Yu Qiao, Yali Wang, and Limin Wang. VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust, editors, International Conference on Learning Representations, volume 2026, pages 109089–109117, 2026. https://proceedings.iclr.cc/paper\_files/paper/2026/file/ b14d7175755b180dc2163e15e3110cb6-Paper-Conference.pdf.

[25] Xinhao Li, Yuhan Zhu, Xiangyu Zeng, Yuhao Dong, Haoning Wu, Zhiqiu Zhang, Yuandong Yang, Changlian Ma, Qingyu Zhang, Yansong Shi, Xinyu Chen, Haoran Chen, Zizheng Huang, Jun Zhang, Kun Ouyang, Lin Sui, Ziang Yan, Yicheng Xu, Chenting Wang, Yinan He, Hongjie Zhang, Yi Wang, Yu Qiao, Yali Wang, Ziwei Liu, Kai Chen, and Limin Wang. VideoChat3: Fully Open Video MLLM for Eficient and Generalist Video Understanding. arXiv preprint arXiv:2607.14935, 2026. https://arxiv.org/abs/2607.14935.

[26] Yunheng Li, Guohong Mu, Hao Li, Shengsheng Qian, Dingwen Zhang, Qibin Hou, and Ming-Ming Cheng. Annotations as Rollouts: Eficient and Scalable Reinforcement Learning for Video MLLMs. arXiv preprint arXiv:2608.20492, 2026. https://arxiv.org/abs/2608.20492.

[27] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. UniVTG: Towards Unified Video-Language Temporal Grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2794–2804, October 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Lin\_UniVTG\_Towards\_Unified\_Video-Language\_ Temporal\_Grounding\_ICCV\_2023\_paper.html.

[28] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 34892–34916. Curran Associates, Inc., 2023. doi: 10.52202/075280-1516. https://proceedings. neurips.cc/paper\_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf.

[29] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 46212–46244. Curran Associates, Inc., 2023. doi: 10.52202/075280-2004. https://proceedings.neurips.cc/paper\_files/paper/2023/ file/90ce332aff156b910b002ce4e6880dec-Paper-Datasets\_and\_Benchmarks.pdf.

[30] NVIDIA. Nemotron 3 Nano Omni: Eficient and Open Multimodal Intelligence, 2026. https://arxiv.org/abs/ 2604.24954.

[31] OpenAI. GPT-5 System Card. Technical report, OpenAI, 2025. https://cdn.openai.com/gpt-5-system-card. pdf.

[32] Xukui Qin, Youting Wang, Xinjie He, Ziyang Luo, Runxiong Wu, Yan-Syuan Chen, and Zhongyao Chu. Decision Readouts for Text-Mediated Video Anomaly Detection: An Exploratory Evaluation of Jev and Qwen, 2026. https://arxiv.org/abs/2609.34180.

[33] Qwen Team. Qwen3.5: Accelerating Productivity with Native Multimodal Agents, February 2026. https://qwen. ai/blog?id=qwen3.5.

[34] Qwen Team. Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All, April 2026. https://qwen.ai/blog? id=qwen3.6-35b-a3b.

[35] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14313–14323, June 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Ren\_TimeChat\_A\_Time-sensitive\_Multimodal\_Large\_ Language\_Model\_for\_Long\_Video\_CVPR\_2024\_paper.html.

[36] Simiao Ren, Kidus Zewde, Xingyu Shen, Yuchen Zhou, Dennis Ng, Ankit Raj, Tommy Duong, Yuxin Zhang, and Neo Tiangratanakul. Open-Jev Judgments on CallScreenBench: Calibrated One-Pass Scam Screening with a Small Language Model, 2026. https://arxiv.org/abs/2609.23959.

[37] Werner Robitza. JEVQA - Video Quality from Metadata, Bitstream, and Pixel Features with a General-Purpose Decision Model, 2026. https://arxiv.org/abs/2609.24395.

[38] TypeSafe AI. System One. TypeSafe AI, n.d. https://docs.typesafe.ai/concepts/system-one. Accessed 2026- 10-12.

[39] Haibo Wang, Zhiyang Xu, Yu Cheng, Shizhe Diao, Yufan Zhou, Yixin Cao, Qifan Wang, Weifeng Ge, and Lifu Huang. Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings of the Association for Computational Linguistics: EMNLP 2025, pages 959–975, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.50. https://aclanthology.org/2025.findings-emnlp.50/.

[40] Lan Wang, Yujia Chen, Du Tran, Vishnu Naresh Boddeti, and Wen-Sheng Chu. SEAL: Semantic Attention Learning for Long Video Representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26192–26201, June 2025. https://openaccess.thecvf.com/content/CVPR2025/html/ Wang\_SEAL\_Semantic\_Attention\_Learning\_for\_Long\_Video\_Representation\_CVPR\_2025\_paper.html.

[41] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, and Jie Tang. LVBench: An Extreme Long Video Understanding Benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22958–22967,

October 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Wang\_LVBench\_An\_Extreme\_Long\_Video\_ Understanding\_Benchmark\_ICCV\_2025\_paper.html.

[42] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 28828–28857. Curran Associates, Inc., 2024. doi: 10.52202/079017-0907. https://proceedings.neurips.cc/paper\_files/paper/2024/ file/329ad516cf7a6ac306f29882e9c77558-Paper-Datasets\_and\_Benchmarks\_Track.pdf.

[43] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9777–9786, June 2021. https://openaccess.thecvf.com/content/CVPR2021/html/ Xiao\_NExT-QA\_Next\_Phase\_of\_Question-Answering\_to\_Explaining\_Temporal\_Actions\_CVPR\_2021\_paper.html.

[44] Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. StreamingVLM: Real-Time Understanding for Infinite Video Streams. In International Conference on Learning Representations, 2026. https://openreview.net/forum?id=gVbPWbA97s.

[45] Senqiao Yang, Kaichen Zhang, Zhaoyang Jia, Jinghao Guo, Yifei Shen, Xinjie Zhang, Xiaoyi Zhang, Haoqing Wang, Xiao Li, Peng Zhang, Xiang An, Yin Xie, Zhening Liu, Xun Guo, Jiahao Li, Shicheng Zheng, Jinglu Wang, Zongyu Guo, Wenxuan Xie, Zihan Zheng, Yuxuan Luo, Bin Li, and Yan Lu. Mage-VL: An Eficient Codec-Native Streaming Multimodal Foundation Model, 2026. https://arxiv.org/abs/2607.24904.

[46] Dingyu Yao, Junhao Zhou, Chenxu Yang, Chuanyu Qin, Xiangyu Zeng, Yifei Li, Haowen Hou, Zheming Liang, Congcong Wang, Kaiwen Tuo, Jun Zhang, Yuhan Zhu, Yuhang Cao, Shenglong Ye, Shuai Xie, Shuhuan Gu, Haoyang Huang, Qingyi Si, Nan Duan, and Jiaqi Wang. JoyAI-VL-Interaction: Real-Time Vision-Language Interaction Intelligence, 2026. https://arxiv.org/abs/2606.14777.

[47] Jiashu Yao, Heyan Huang, Daiqing Wu, Wangke Chen, Huaxi Ai, Haoyu Wen, Zeming Liu, and Yuhang Guo. Benchmarking Living-Screen-Native GUI Agents on Short-Video Platforms, 2026. https://arxiv.org/abs/2606. 04701.

[48] Guanxu Yu and Yuhang Yao. Visual Jev: Accurate and Eficient Decisions from Shared Visual Context, 2026. https://arxiv.org/abs/2609.25845.

[49] Xiangyu Zeng, Kunchang Li, Chenting Wang, Xinhao Li, Tianxiang Jiang, Ziang Yan, Songze Li, Yansong Shi, Zhengrong Yue, Yi Wang, Yali Wang, Yu Qiao, and Limin Wang. TimeSuite: Improving MLLMs for Long Video Understanding via Grounded Tuning. In International Conference on Learning Representations, 2025. https://openreview.net/forum?id=nAVejJURqZ.

[50] Xiangyu Zeng, Yuandong Yang, Zhiqiu Zhang, Yuhan Zhu, Xinhao Li, Qingyi Si, Dingyu Yao, Changlian Ma, Haoran Chen, Xinyu Chen, Yansong Shi, Junhao Zhou, Yifei Li, Jun Zhang, Chuanyu Qin, Chenxu Yang, Xinlei Yu, Kun Ouyang, Yuchen Shao, Qianshan Wei, Changhai Zhou, Jun Gao, Jiaqi Wang, and Limin Wang. OneStreamer: Unifying Perception, Memory, and Proactive Response in Streaming Video Interaction. arXiv preprint arXiv:2610.01762, 2026. https://arxiv.org/abs/2610.01762.

[51] Xiangyu Zeng, Zhiqiu Zhang, Yuhan Zhu, Xinhao Li, Zikang Wang, Changlian Ma, Qingyu Zhang, Zizheng Huang, Kun Ouyang, Tianxiang Jiang, Ziang Yan, Yi Wang, Hongjie Zhang, Yali Wang, and Limin Wang. Video-o3: Native Interleaved Clue Seeking for Long Video Multi-Hop Reasoning. arXiv preprint arXiv:2601.23224, 2026. https://arxiv.org/abs/2601.23224.

[52] Yingsen Zeng, Zepeng Huang, Yujie Zhong, Chengjian Feng, Jie Hu, Lin Ma, and Yang Liu. Dis-Time: Distribution-based Time Representation for Video Large Language Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 21961–21971, October 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Zeng\_DisTime\_Distribution-based\_Time\_ Representation\_for\_Video\_Large\_Language\_Models\_ICCV\_2025\_paper.html.

[53] Hang Zhang, Xin Li, and Lidong Bing. Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding. In Yansong Feng and Els Lefever, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 543–553, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-demo.49. https://aclanthology.org/ 2023.emnlp-demo.49/.

[54] Hao Zhang, Aixin Sun, Wei Jing, and Joey Tianyi Zhou. Span-based Localizing Network for Natural Language Video Localization. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6543–6554, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.585. https://aclanthology.org/ 2020.acl-main.585/.

[55] Jun Zhang, Teng Wang, Yuying Ge, Yixiao Ge, Xinhao Li, and Limin Wang. TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10419–10429, June 2026. https://openaccess.thecvf.com/content/CVPR2026/ html/Zhang\_TimeLens\_Rethinking\_Video\_Temporal\_Grounding\_with\_Multimodal\_LLMs\_CVPR\_2026\_paper.html.

[56] Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language. In the AAAI Conference on Artificial Intelligence, 2020.

[57] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. MLVU: Benchmarking Multi-task Long Video Understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13691–13701, June 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Zhou\_MLVU\_Benchmarking\_ Multi-task\_Long\_Video\_Understanding\_CVPR\_2025\_paper.html.

[58] Xunlan Zhou, Xianliang Yang, and Li Zhao. From Text Decisions to Pixels: An Study of Jev-Style Visual Choice Model, 2026. https://arxiv.org/abs/2609.29283.

[59] Yuhan Zhu, Yuyang Ji, Zhiyu Zhao, Gangshan Wu, and Limin Wang. AWT: Transferring Vision-Language Models via Augmentation, Weighting, and Transportation. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 25561–25591. Curran Associates, Inc., 2024. doi: 10.52202/079017-0804. https://proceedings.neurips. cc/paper\_files/paper/2024/file/2d4eaf042567f1c03c086103cc154c1f-Paper-Conference.pdf.

[60] Yuhan Zhu, Guozhen Zhang, Jing Tan, Gangshan Wu, and Limin Wang. Dual DETRs for Multi-Label Temporal Action Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18559–18569, June 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Zhu\_Dual\_ DETRs\_for\_Multi-Label\_Temporal\_Action\_Detection\_CVPR\_2024\_paper.html.

[61] Yuhan Zhu, Changlian Ma, Xiangyu Zeng, Xinhao Li, Zhiqiu Zhang, Songze Li, Jun Zhang, Tianxiang Jiang, Yuandong Yang, Ziang Yan, Zikang Wang, Xinyu Chen, Haoran Chen, Shaowei Zhang, and Limin Wang. Time-Lens2: Generalist Video Temporal Grounding with Multimodal LLMs. arXiv preprint arXiv:2607.17423, 2026. https://arxiv.org/abs/2607.17423.

[62] Yuhan Zhu, Xiangyu Zeng, Chenting Wang, Xinhao Li, Chunxu Liu, Yicheng Xu, Ziang Yan, Yi Wang, and Limin Wang. FreeRet: MLLMs as Training-Free Retrievers. In Tong Zhang, Miroslav Dudik, Martin Jaggi, Alekh Agarwal, Sharon Li, Dale Schuurmans, Jerry Zhu, Felix Berkenkamp, Hanze Dong, and Alberto Bietti, editors, Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 167936–167952. PMLR, 06–11 Jul 2026. https://proceedings.mlr.press/ v306/zhu26ba.html.

[63] Yuhan Zhu, Guozhen Zhang, Chen Xu, Haocheng Shen, Xiaoxin Chen, Gangshan Wu, and Limin Wang. Eficient Test-Time Prompt Tuning for Vision-Language Models. International Journal of Computer Vision, 134(10):468, Oct 2026. ISSN 1573-1405. doi: 10.1007/s11263-026-03019-3. https://doi.org/10.1007/s11263-026-03019-3.

## Supplementary Material

## Contents

A Detailed Evaluation Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18   
A.1 Latency . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18   
A.2 Performance . . . . . . . .   
A.3 Autoregressive Control with the Same Training Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19   
A.4 Inference Modes and Direct Baseline . . . . . . . . .   
B Qualitative Benchmark Cases . . . . . . 20

## A Detailed Evaluation Protocol

Generative baselines use Instruct unless labeled Thinking; VideoJev defaults to Joint inference. Both tasks use video frames without subtitles or audio. Main benchmarks use full-video sampling at 4 fps for grounding and up to 2 fps (at most 2,048 frames) for QA, with frame area capped at 640 × 640 pixels.

## A.1 Latency

Both tasks use initially idle H200 GPUs at batch size 1. We sum synchronized forward time for visual encoding, prefill, and all decoding steps through EOS or the model-specific output limit, including Thinking tokens, then divide by the query count. Video loading and decoding, preprocessing, transfers, and postprocessing (including parsing and metric computation) are excluded; truncated and invalid outputs remain included.

VideoJev encodes each video once and scores its queries in one forward. Generative baselines reuse visual encoding and prefix prefill, then decode from independent query branches without answer history.

Temporal grounding latency. Table 1 covers all 1,313 Charades videos (3,363 queries) on a single H200, with one untimed warm-up per video. VideoJev removes the final-layer MLP, retains native FP32 scoring heads and MoE routers, and matches the visual inputs of Qwen3.5-4B and Video-ORA-4B. Greedy decoding allows 4,096 output tokens (Nemotron: 1,024). Context is 262,144 tokens, except for Molmo2 (110,592; YaRN factor 3). Mage lowers frame resolution only to avoid context overflow; native visual merging and pruning are retained.

Video QA latency. Table 2 covers all 103 LVBench test videos (1,549 questions) on a single H200, measuring each video once after untimed warm-up. VideoJev-4B and Qwen use fixed RGB geometry with the QA sampling defaults, a 256K context, and an 80% visual-token budget; other architectures retain native visual settings and backends. Instruct requests an option letter and Thinking an answer field; Qwen3.5-4B Thinking retains its earlier prompt. VideoJev-35B-A3B retains FP32 routers and a FP32 decision head. Qwen3.5/3.6 MoE and VideoJev-35B-A3B use Transformers’ native grouped\_mm experts.

Multi-query scaling. Figure 3 uniformly samples 40 distinct videos per task with a fixed seed: ten per Charades duration quartile and five per Video-Holmes [6] duration stratum (eight strata). All methods use the same ordered 1–10 queries per video, cycling native questions when needed, at 2 fps, at most 640 × 640 pixels per frame, and a 262,144-token context. Inference modes follow Appendix A.4; Qwen sampling follows Appendix A.2, with at most 32,768 output tokens. Shared visual encoding and prefill are charged once, followed by query and decoding forwards. On exclusively idle H200 GPUs, we run an untimed warm-up and three fixed-seed traces per video, retaining the whole trace with median ten-query time. We report equalvideo means and pointwise 95% confidence intervals from 10,000 paired bootstrap resamples within duration strata. Generation monitoring overhead is measured separately and excluded.

## A.2 Performance

Temporal grounding performance. Qwen3.5-4B/35B-A3B, Qwen3.6-35B-A3B, Mage-VL, and Nemotron are evaluated on all Charades (3,363), ActivityNet (4,500), and QVHighlights (1,541) test queries. Full videos use 4 fps, reducing frame area to fit a 144K (147,456-token) context and 114,688-token input budget. Decoding follows Appendix A.1; invalid intervals count as failures. Other baselines retain published scores. VideoJev uses the same budget with Joint inference and fixes $\tau = 0 . 2 5$ and smoothing window 1 across both sizes and all experiments.

Video QA performance. All VideoJev QA evaluations—both sizes, inference modes, and ablations—use the QA sampling defaults and a fixed 256K (262,144-token) context. Frame area is reduced to keep visual tokens within 80% of this budget. MLVU averages seven development tasks.

Generative baselines answer questions independently without answer history, unlike the shared-prefix latency protocol. Qwen also uses the QA sampling defaults and a 256K token budget. Qwen3.5/3.6 Instruct requests one option letter; Thinking retains the earlier answer-field prompt. Nemotron retains Conv3D temporal merging with factor 2 and disables EVS, with Instruct and Thinking evaluated separately. Published scores retain their source settings. Before scoring, we verify complete prediction and answer-extraction coverage. Deterministic option parsing is followed, when needed, by a text-only extractor without ground-truth access; invalid answers count as incorrect.

Multi-query prediction quality. Figure 3 reports full-Charades mIoU and full-LVBench accuracy, evaluated separately from timing. All grounding scores, including the fresh Qwen Thinking evaluation, follow the grounding input protocol above. Direct predicts exactly one option token and is evaluated separately from the Instruct baseline in Table 2. Qwen uses temperature/top-p 0.7/0.8 (Instruct) or 1.0/0.95 (Thinking), top-k 20, presence penalty 1.5, and at most 32,768 output tokens. Direct masks invalid option tokens before sampling filters; CoT reasons before emitting Answer: X. Scoring accepts unambiguous final options with optional answer text, ignores preceding reasoning, and assigns zero to invalid or truncated answers.

## A.3 Autoregressive Control with the Same Training Data

Table 3 compares single-stage VideoJev-4B (Table 8) with native Qwen3.5-4B retaining its final MLP and tied language-model head. Both train on the same 404,037 questions from 58,424 videos across VideoJev’s training partitions. The control trains for one epoch (546 updates) with XTuner FSDP2, global batch size 64, 147,456-token packs, a $5 \times 1 0 ^ { - 6 }$ learning rate, 5% warm-up, and cosine decay to $5 \times 1 0 ^ { - 7 }$ . Inputs and optimizer settings follow VideoJev’s joint adaptation; the video encoder and merger stay frozen. Answertoken cross-entropy includes EOS and is averaged within questions, then across questions. Targets are all annotated intervals as JSON pairs in seconds, or a QA option letter; Thinking is disabled.

Evaluation uses VideoJev’s visual features, full benchmark splits, and context budgets: 147,456 tokens for grounding and 262,144 for QA. Queries branch independently from an immutable video prefix without answer history. Greedy decoding allows 256 tokens for interval lists and 16 for QA. Grounding scores the first interval, counting invalid or truncated outputs as failures; QA uses benchmark-specific prompts and deterministic option parsing.

## A.4 Inference Modes and Direct Baseline

Shared inference modes. Cached inference encodes and prefills the video once, then processes queries sequentially from independent copies of the prefix cache. Each branch contains only its query and any generated tokens, without query or answer history. Joint inference concatenates queries after one video prefix and reads all predictions in one causal forward. Later queries can attend to earlier queries, but no reference or predicted answers are inserted. Both modes use batch size 1.

VideoJev supports both modes for grounding and QA through direct candidate scores. Autoregressive Joint inference requires single-token answers read from each query’s next-token distribution. Multi-token outputs,

![](images/74c8f50113abc88d83d98502fb6e072620108baf7a780affd60f682e2236f2ba.jpg)  
Joint turn: USER Q + options <|im\_end|> ASSISTANT <think></think> Answer: [readout] <|im\_end|>

Figure 5 Qwen3.5-4B Direct inference. (a) Queries branch independently from a shared video cache. (b) All option predictions are read in one causal forward. Both modes use batch size 1, restrict logits to valid option tokens, and leave answer slots empty; [readout] marks a logit readout, not inserted text.

including intervals, CoT, and Thinking, require sequential token feedback and thus use Cached inference.   
Their dotted curves in Figure 3 use this mode even when all queries are available together.

Direct option prediction. Direct predicts exactly one option token without reasoning. Figure 5 shows Qwen3.5-4B’s native non-thinking template. Cached reads option logits after one query sufix forward; Joint reads every Answer: position in one forward over complete turns with empty answer slots. Both mask logits to valid plain or leading-space option tokens before applying oficial sampling filters.

Empty answer histories. With per-turn non-thinking prefixes, unconstrained LVBench readouts produced valid options for all 103 first questions but <|im\_end|> for 1,418 of 1,446 later questions. Inserting earlier answers, even incorrect ones, restored option outputs, suggesting sensitivity to empty answer histories. Reported Direct results use complete turns, uniform oficial non-thinking prefixes, no inserted answers, and the option-token masking above; unconstrained outputs are archived separately.

## B Qualitative Benchmark Cases

Eight cases across seven benchmarks compare Joint inference outputs. Frames retain native aspect ratios; details are enlarged for display only. Green denotes ground truth; brown and red denote the models. QA bars show uncalibrated scores with selected options outlined. Grounding times are video-relative.

## 01 Six questions in one forward

![](images/79987320f1e89d4bb7cc175b64a0cad9295d168cf6c1c548fd8ee7e2a5b41938.jpg)  
Figure 6 Joint inference grounds six Charades queries in one forward; three readouts are shown in order. Both models match the one-second transition, while VideoJev-35B-A3B locates adjacent actions more precisely.

<table><tr><td colspan="3">Q1 What year appears in the opening caption of the video?</td></tr><tr><td>A.1636 B.1366 C.1363 D.1633 Ground truth: D VideoJev-4B: D (&gt;99.9%)</td><td></td><td>VideoJev-35B-A3B: D (&gt;99.9%)</td></tr><tr><td>Q2 How is the weather like in the opening?</td><td></td><td></td></tr><tr><td>A. Cloudy B. Snowy C. Sunny D. Rainy</td><td></td><td></td></tr><tr><td>Ground truth: B</td><td>VideoJev-4B: B (&gt;99.9%)</td><td>VideoJev-35B-A3B: B (&gt;99.9%)</td></tr></table>

## 02 Locating a compound sports event ActivityNet Captions / 147.4s

v\_FrV8r4l5ZUM

Two children performed a set of jump rope moves, then two girls continued jumping.

![](images/0f5cda08d91f5438149de40be9e1678586d839e4572531c373b42a1454264f04.jpg)  
Figure 7 Localizing a compound jump-rope event on ActivityNet Captions. VideoJev-35B-A3B more closely matches the annotated interval.

## 03 Retrieving an event amid distractors

QVHighlights / 150.0s

NUsG9BgSes0\_510.0\_660.0

Someone opens a cup of instant noodles and adds the seasoning packet and hot water.

![](images/698a12bbd5482c5dfb8d9289a1bbe293e959da6b726419a496cfcfb7d39061af.jpg)  
Figure 8 Retrieving a preparation sequence amid vlog distractors on QVHighlights. Both models match the annotated interval.

04 Eighteen questions over an hour-long video  
![](images/015517680ce1c21373b5d832555134c4c0cbea87e1e8985ee219e2084ac8b4c4.jpg)  
Figure 9 Joint inference answers 18 questions over a 3,665.5-second LVBench video in one forward. Two readouts test caption retrieval and weather recognition; the caption detail is enlarged for display.

28s

VideoJev-35B-A3B

## 05 Relating an event to what happens next

LongVideoBench / 538.1s

h4jIoMxZopU

In front of a blue background, a gentleman wearing a shirt with pink floral patterns is speaking. What did the gentleman do after becoming friends with the unicorn?

![](images/e4ebd6a0aa3a7da4e8462ce72d776743653b52714c7f841e7c39cef13dcbc650.jpg)

![](images/c18a754dfcac6fdac27c40f462c875c12d58bd52f9aa816ee178b62befaa8ba9.jpg)

![](images/2ec6e38d3b2ebec45372394a897ad680e936179634563fc1b2257196033c109e.jpg)

![](images/c767717f2de37172a6f2f58702df04529bf277256539fac94ea50b948bee11a7.jpg)

![](images/26ece4d352b61bf9a807aae2fcb9ff793aab9ce43a3c490f1c7c9163d40ef9ab.jpg)  
VideoJev-4B

![](images/afda1ca7cf39b5211c9889d0111f57f4f7fbfb899a02dd5480e251a45fa8392c.jpg)  
VideoJev-35B-A3B

A. Put on a watch   
B. Changed into a different shirt   
C. Put on a mask   
D. Put on a pair of sunglasses   
E. Put on a unicorn headpiece

![](images/e49da6366a44cebdf3241c5c747882ec9d60eee7540945ef07aef68ad0dab288.jpg)

![](images/9bd9a0fe614e28b8c4594a10c81c4f782d4f3b9ddea37064280f28458b15e991.jpg)  
Figure 10 Recognizing an event relation on LongVideoBench. Both models select the annotated answer from six options.

## 06 Counting recurring scenes across a video

MLVU / 500.1s

count\_108

In this video, how many instances are there of the 'carving pumpkin' action scene in total?

![](images/c08f6a6407bef738b4eb69a49d38ee88fdb035aca2bea99be8079ad6e33dbf32.jpg)  
Figure 11 Counting five recurring scenes amid unrelated footage on MLVU. Each frame pair shows one occurrence. Both models select the annotated answer, with diferent option scores.

## 07 Inferring intent at a specified moment

LVBench / 6856.5s

5dZ\_lvDgevk

![](images/86e5a289e633bb0bf25f298fa2d98c71560bbee421c61871d71b6420770787f6.jpg)

What goals or objectives are the man holding a woman's arm trying to achieve at 101:21?

![](images/92b516bc1e986ce647839ac02292419f1a85214dd2e1c88f2e3362d8e41cc179.jpg)

![](images/1f24acb6fd05da79578bbe8d3fbe8cf86a90035e74a0c5e106fe1f1381c43f59.jpg)

![](images/93a01bc337a6d400660c69d732c80ffe18529a9789850c0be400e23cee2666cc.jpg)

![](images/f0fbc7bbae9532f41c4e54313ac2ad2e85f36494576bf7d2d90eb38eb35b6098.jpg)  
VideoJev-4B

![](images/e6a7333938b7b67072a14f7d9eb575aa280ccae7fc1560730b205f10ad57ca56.jpg)

![](images/8204ea5b60f87f09cbc990db07ee50cbe741e91b571ce81abc679ac463c74bab.jpg)  
Figure 12 An intent-inference disagreement on LVBench: VideoJev-4B interprets the interaction as courtesy, while VideoJev-35B-A3B selects the annotated arrest intent.

![](images/1354d55f5cd0dde6498f7bf2998b368ad1d32b0f5feb97abc5865690cf911db3.jpg)  
aqTIB\_q40bo  
Figure 13 A shared failure on small-object counting in Video-MME. Both models miss the annotated answer of four books; the book detail is enlarged for display.

## 08 A shared failure on small-object counting

Video-MME / 38.0s

At the end of the video, how many books can be seen resting on the living room table?

![](images/c36ec356b3e28827d9fb3f8e05ae2cda37dd16134276cadbb548ff8e8b8b4259.jpg)

![](images/2ae10bccc8ddb6137b9eb862a34db21c08e6ac96c8b0ef8ef4a21ad8e5f5eedd.jpg)  
25s / book detail Display enlargement; models receive the full video.