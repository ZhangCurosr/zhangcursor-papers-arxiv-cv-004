# SIGNRAG: UNIFIED RETRIEVAL-AUGMENTED GLOSS-FREE SIGN LANGUAGE TRANSLATION

Zhi Rao<sup>1</sup>, Yucheng Zhou<sup>2</sup>, Qianran Sun<sup>3</sup>, Yiqing Huang<sup>1</sup>, Longcan Yuan<sup>1</sup>, Jiayi Hou<sup>4</sup>, Chengwen Yao<sup>5</sup>, Lin Cheng<sup>5</sup>, Donghui Sun<sup>5</sup>, Xiaoxin Chen<sup>5</sup>, Jun Wan<sup>1,3∗</sup>

<sup>1</sup>Faculty of Innovation Engineering, Macau University of Science and Technology, Macau, China <sup>2</sup>SKL-IOTSC, CIS, University of Macau, Macau, China

<sup>3</sup>MAIS, Institute of Automation, Chinese Academy of Sciences, Beijing, China

<sup>4</sup>Yale University, New Haven, CT, USA

<sup>5</sup>VIVO AI Lab, China

zhir@student.must.edu.mo jun.wan@ia.ac.cn

## ABSTRACT

Contemporary decoder-only large language models (LLMs) have demonstrated strong capabilities across a wide range of domains. However, existing pretraining paradigms for gloss-free sign language translation (SLT) are largely designed around conventional encoder-decoder pretrained language models, which limits their direct applicability to decoder-only LLMs. To address this limitation, we propose SignRAG, a unified framework combining hierarchical pretraining, target-domain retrieval augmentation, and retrieval-aware reinforcement fine-tuning. Hierarchical pretraining first learns linguistically grounded sign representations and then jointly aligns the sign encoder with an LLM, mitigating cross-modal optimization imbalance. For downstream adaptation, SignRAG complements parameter-based fine-tuning with a target-domain retrieval gallery that provides instance-specific translation cues. To ensure that retrieved contexts are used appropriately, we further introduce Retrieval Utility-Guided Reinforcement Fine-Tuning (RUG-RFT), which combines translation-quality and retrieval-utility rewards to encourage beneficial retrieval use while suppressing harmful reliance. Experiments on multiple SLT benchmarks establish new state-of-the-art performance. In particular, to the best of our knowledge, SignRAG is the first gloss-free approach to outperform gloss-supervised methods across all reported metrics on CSL-Daily. Our code has been released at GitHub, together with models of different sizes to support future academic research.

## 1 INTRODUCTION

Sign language translation (SLT) aims to translate visual sign sequences into natural language. Despite recent progress, existing methods still struggle to generalize across datasets with different signers, signing styles, and vocabulary distributions. A natural solution is to pretrain on large-scale sign language corpora and subsequently adapt the model to downstream SLT benchmarks.

Existing pretraining paradigms for gloss-free SLT can be broadly divided into two categories. Encoder-only pretraining independently pretrains the sign encoder before transferring it to downstream translation models (Zhou et al., 2023; Chen et al., 2024; Wong et al., 2024; Chen et al., 2025; Jiao et al., 2024), whereas end-to-end pretraining jointly optimizes visual and language components on large-scale sign-language corpora (Uthus et al., 2023; Zhang et al., 2024; Tanzer & Zhang, 2025; Li et al., 2025b; Fish & Bowden, 2026). However, most existing methods still rely on T5-based encoder–decoder models, whose language modeling capacity remains limited compared with contemporary LLMs.

Replacing these decoders with a substantially stronger LLM is not a drop-in upgrade. Because the language branch is already strongly pretrained, it can dominate optimization and leave the comparatively weaker visual pathway under-trained. We refer to this phenomenon as Cross-Modal Optimization Imbalance. Such imbalance has been widely observed in multimodal learning (Wang et al., 2020; Huang et al., 2022; Peng et al., 2022) and can further induce shortcut learning driven by strong language priors (Goyal et al., 2017). It raises a key question: how should large-scale sign pretraining be organized so that visual pathway still learns semantically grounded representations?

![](images/13b4b3590e8f8a2cbff09437ca171cabb1e034f49bed00cbccf7ef732cbcf0bc.jpg)

![](images/86b5be20987ea2ac51c8ddf1b3eaa4416225019c9b7f259fa72ad96c9b188924.jpg)

![](images/2a7e544f10e1a718740c8b4d16a07e924ac499e30d5368631b2c7b66870d8536.jpg)  
Figure 1: Motivation. Hierarchical pretraining and retrieval augmentation improve sign-to-language learning. (a), (b) Comparison of different pretraining strategies. The solid and dashed lines denote the previous state-of-the-art performance under the gloss-free and gloss-based settings, respectively, corresponding to Geo-Sign (Fish & Bowden, 2026) and CV-SLT (Zhao et al., 2024). (c) Comparison of the semantic distance between generated and ground-truth (GT) texts with and without RAG. Semantic distance is measured using the independent sentence encoder paraphrase-multilingual-MiniLM-L12-v2. Bars report the mean cosine distance over all samples, with error bars denoting ±SEM. Samples are drawn from the CSL-Daily dev set.

We therefore propose a Hierarchical Pretraining strategy for LLM-based SLT. Inspired by the staged-alignment paradigm of general-purpose multimodal large language models (MLLMs) (Liu et al., 2023; Ye et al., 2024; Bai et al., 2025; McKinzie et al., 2024), we organize large-scale pretraining into two stages. We first use fine-grained sign–text supervision to learn linguistically grounded sign representations, and then jointly pretrain the sign encoder, multimodal projector, and LLM on the same corpus. In this way, the model first learns what the sign means and then learns how to express its semantics in natural language. As shown in Fig. 1(a) and (b), hierarchical pretraining clearly outperforms alternative pretraining schemes, while the Pretrain→SFT variant alone already surpasses previous state-of-the-art methods under both gloss-free and gloss-based settings.

After large-scale pretraining, downstream adaptation introduces a second challenge, which we term the Target-Domain Adaptation Dilemma. Sign-language datasets exhibit substantial distribution shifts across languages, recording conditions, and signers, while downstream training sets are typically much smaller than the pretraining corpus. Aggressive fine-tuning may therefore erode transferable knowledge acquired during pretraining, whereas conservative fine-tuning may leave the target domain insufficiently adapted. To address this dilemma, we complement parametric SFT with a non-parametric target-domain semantic memory. Specifically, we construct a retrieval gallery exclusively from the target-domain training split and retrieve semantically related sign instances to gether with their translations as instance-specific linguistic cues. As shown in Fig. 1(c), retrieval augmentation reduces the semantic distance between generated and GT texts from 0.234 to 0.206.

Retrieval, however, is not always beneficial, leading to a third challenge that we term the Retrieval Reliance Dilemma. Standard SFT optimizes token-level likelihood but provides no explicit supervision on whether retrieved contexts actually benefit generation, making both underutilization and harmful reliance difficult to regulate. We first introduce retrieval dropout during retrieval-augmented SFT to improve robustness when retrieved information is absent. We then propose Retrieval Utility-Guided Reinforcement Fine-Tuning (RUG-RFT) based on Group Relative Policy Optimization (GRPO) (Shao et al., 2024). RUG-RFT combines translation-quality and retrieval-utility rewards, encouraging the model to exploit informative retrieval while suppressing harmful reliance on mis leading contexts.

We conduct extensive experiments on four downstream SLT benchmarks: CSL-Daily, PHOENIX-2014T, How2Sign, and OpenASL. SignRAG consistently outperforms existing gloss-free state-ofthe-art methods. On CSL-Daily, it further surpasses gloss-supervised methods across all metrics; to the best of our knowledge, this is the first gloss-free SLT approach to achieve such performance. Our main contributions are summarized as follows:

• We propose a hierarchical pretraining paradigm that first learns linguistically grounded sign representations through fine-grained sign–text alignment and then performs large-scale joint multimodal pretraining with a contemporary LLM.

• We introduce a target-domain retrieval-augmented adaptation mechanism that constructs a nonparametric semantic memory to complement parameter-based SFT with instance-specific targetdomain knowledge.

• We propose Retrieval Utility-Guided Reinforcement Fine-Tuning (RUG-RFT), which jointly optimizes translation quality and retrieval contribution to enable more effective and robust use of retrieved contexts.

• Extensive experiments across multiple SLT benchmarks validate the effectiveness of the proposed framework and establish new state-of-the-art performance.

## 2 RELATED WORK

Sign language translation. Early SLT relies on gloss supervision via gloss recognition or cascaded translation (Camgoz et al., 2020; Zhou et al., 2021; Chen et al., 2022c;b), whereas gloss-free SLT maps sign videos directly to text using vision–language learning, LLM decoding, or cross-modal alignment (Camgoz et al., 2018; Li et al., 2020; Zhao et al., 2021; Yin et al., 2023; Zhou et al., 2023; Wong et al., 2024; Chen et al., 2024; Gong et al., 2024; Jiao et al., 2024; Chen et al., 2025), and external large-scale pretraining further improves transferability (Uthus et al., 2023; Rust et al., 2024; Li et al., 2025b). Retrieval-augmented generation and reinforcement learning. RAG conditions generation on retrieved non-parametric contexts (Lewis et al., 2020; Chen et al., 2022a), retrievalbased SLT (Yao et al., 2026) mainly targets gloss-based settings, and GRPO (Shao et al., 2024; Guo et al., 2025) has been applied to retrieval-enhanced reasoning and generation (Huang et al., 2025; Li et al., 2025a; Wang et al., 2025). SignRAG studies how large-scale sign pretraining should be organized under a modern LLM decoder, and complements parametric adaptation with a sign-to-text retrieval gallery and retrieval Utility-Guided reinforcement fine-tuning, without gloss annotations. A more detailed discussion is provided in Appendix A.

## 3 METHOD

As illustrated in Fig. 2, SignRAG consists of three stages for gloss-free sign language translation (SLT). Stage 1 performs hierarchical pretraining, where Stage I learns linguistically grounded sign representations and Stage II jointly pretrains the sign encoder, multimodal projector, and LLM. Stage 2 performs retrieval-augmented SFT using a target-domain Sign Retrieval Gallery, which retrieves semantically related sign instances and provides their paired translations as auxiliary cues; retrieval dropout is further applied during training. Stage 3 applies Retrieval Utility-Guided Reinforcement Fine-Tuning (RUG-RFT) with GRPO, combining a Quality Reward and a Retrieval-Utility Reward to improve translation quality and encourage appropriate use of retrieved contexts. We next describe these three components in detail.

## 3.1 HIERARCHICAL PRETRAINING

Stage I: Sign–text alignment. We first align skeletal sign representations with natural-language semantics. Given a sign video V, MMPose (MMPose Contributors, 2020) first extracts 2D keypoints, which are subsequently encoded by a CoSign-based skeleton encoder (Jiao et al., 2023) into frame-level skeletal representations. The paired translation T is encoded by an mBART-based text encoder (Liu et al., 2020). The two encoders are jointly optimized with a bidirectional InfoNCE contrastive objective ${ \mathcal { L } } _ { \mathrm { c o n } }$ (Cheng et al., 2023) and a CoCa-style skeleton-to-text translation objective $\mathcal { L } _ { \mathrm { s l t } }$ (Yu et al., 2022):

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p t } } = \mathcal { L } _ { \mathrm { c o n } } + \mathcal { L } _ { \mathrm { s l t } } . } \end{array}\tag{1}
$$

Detailed formulations are provided in Appendix B.1. This stage endows the sign encoder with linguistically grounded representations before it is coupled with the LLM.

Stage II: Large-scale sign MLLM pretraining. We then couple the aligned sign encoder with an LLM. A GELU-activated MLP projector maps the skeletal features into the embedding space of Qwen3 (Yang et al., 2025), and LoRA adapters are attached to the query and value projections of the LLM self-attention layers. The sign encoder, the projector, and the LoRA-adapted LLM are jointly optimized on large-scale sign–text corpora, CSL-News (Li et al., 2025b) for Chinese sign language and YouTube-ASL (Uthus et al., 2023) for American sign language, with the autoregressive objective

![](images/702dccac571d63b7f32ee29360ae25ee2b299d01da4f036247528ffe66961354.jpg)  
Figure 2: Overview of SignRAG. (1) Hierarchical pretraining on large-scale sign corpora: Stage I learns language-aware sign representations with a contrastive objective ${ \mathcal { L } } _ { \mathrm { c o n } }$ and a skeleton-to-text objective $\mathcal { L } _ { \mathrm { { s l t } } } ;$ Stage II couples the pretrained sign encoder with the LLM (Qwen3) via a multimodal projector and performs large-scale joint pretraining. (2) Target-domain supervised fine-tuning: a sign retrieval gallery provides semantically related reference translations as auxiliary cues, and the model is fine-tuned with sign features and retrieved texts, with random retrieval dropout. (3) Retrieval utility-guided reinforcement fine-tuning: a quality reward and a retrieval-utility reward are combined and optimized with GRPO.

$$
\mathcal { L } _ { \mathrm { l m } } = - \sum _ { t = 1 } ^ { T } \log p _ { \boldsymbol { \theta } } ( y _ { t } \mid y _ { < t } , \mathbf { I } , \mathbf { Z } ) ,\tag{2}
$$

where Z denotes the projected sign features and I the instruction. Because the sign encoder enters joint training already grounded in language semantics, it no longer acts as the weak branch when coupled with the LLM, which mitigates the Cross-Modal Optimization Imbalance discussed in the Introduction.

## 3.2 RETRIEVAL-AUGMENTED SFT

Sign retrieval gallery. For each downstream dataset, we fine-tune the sign encoder on its training set and build a sign-to-text retrieval gallery from that split only; validation and test samples serve exclusively as queries, so that retrieved knowledge always comes from the target domain. For a gallery video $\bar { V ^ { g } }$ , its frame-level representation is

$$
Z ^ { g } = { \mathrm { S k e l e t o n E n c } } ( V ^ { g } ) \in \mathbb { R } ^ { L _ { g } \times C } ,\tag{3}
$$

where $L _ { g }$ denotes the temporal length and $C$ is the feature dimension. Each gallery entry stores the encoded sign feature sequence and its paired ground-truth translation,

$$
\mathcal { G } _ { n } = \left( Z _ { n } ^ { g } , T _ { n } ^ { g } \right) .\tag{4}
$$

Query videos are encoded by the same skeleton encoder for sign-level retrieval.

Attention-weighted sign–sign similarity. Given a query sign sequence $Z ^ { q } ~ \in ~ \mathbb { R } ^ { L _ { q } \times C }$ and a gallery sequence $Z ^ { g } ~ \in ~ \mathbb { R } ^ { \breve { L } _ { g } \times C }$ , we first project their frame-level features into a shared Ddimensional embedding space:

$$
Q = \mathrm { N o r m } ( \phi _ { q } ( Z ^ { q } ) ) , \qquad G = \mathrm { N o r m } ( \phi _ { g } ( Z ^ { g } ) ) ,\tag{5}
$$

where $Q \in \mathbb { R } ^ { L _ { q } \times D }$ and $G \in \mathbb { R } ^ { L _ { g } \times D }$ . Here, $\phi _ { q } ( \cdot )$ and $\phi _ { g } ( \cdot )$ denote the projection heads, and Norm(·) performs row-wise $\ell _ { 2 }$ normalization.

We then compute the frame-level cosine similarity between the query and gallery sequences:

$$
E _ { i , j } = Q _ { i } ^ { \top } G _ { j } ,\tag{6}
$$

where $\boldsymbol { E } \in \mathbb { R } ^ { L _ { q } \times L _ { g } }$ . For each query frame, we normalize its similarities over the gallery sequence to obtain attention weights:

$$
P _ { i , j } = \frac { \exp ( E _ { i , j } ) } { \sum _ { j ^ { \prime } = 1 } ^ { L _ { g } } \exp ( E _ { i , j ^ { \prime } } ) } .\tag{7}
$$

The global sign–sign similarity is obtained by aggregating the attention-weighted frame-level similarities:

$$
S ( Q , G ) = \frac { 1 } { L _ { q } } \sum _ { i = 1 } ^ { L _ { q } } \sum _ { j = 1 } ^ { L _ { g } } P _ { i , j } E _ { i , j } .\tag{8}
$$

Compared with pooling an entire sign video into a single global representation, our formulation preserves fine-grained frame-level correspondences between sign sequences, which are crucial for retrieving semantically relevant samples. For clarity, the padding masks used to handle variablelength sequences are omitted from the above formulations. To improve memory efficiency, we shard the gallery across distributed workers and compute sign–sign similarities in query–gallery chunks. The exact global top-K results are then obtained by merging the local top-K candidates returned by all workers. Further implementation details are provided in Appendix B.4.

Retrieval-augmented fine-tuning. We use skeleton embeddings $Z _ { s }$ as the visual representation of sign language and project them into the LLM latent space:

$$
Z = { \mathrm { M L P } } ( Z _ { s } ) ,\tag{9}
$$

where the projector is a GELU-activated MLP. We use Qwen3 (Yang et al., 2025) as the base LLM and form its input by concatenating the instruction I, sign features $\mathbf { Z } ,$ and retrieved contexts R.

To support the subsequent Retrieval-Utility Reward, we apply retrieval dropout during SFT:

$$
\tilde { \mathbf { R } } = \left\{ \begin{array} { l l } { \mathbf { R } , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 1 - p , } \\ { \boldsymbol { \emptyset } , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } p , } \end{array} \right.\tag{10}
$$

where ∅ denotes an empty token sequence without placeholder tokens. The model is fine-tuned with LoRA by minimizing

$$
\mathcal { L } _ { \mathrm { s f t } } = - \sum _ { t = 1 } ^ { T } \log p _ { \theta } \Big ( y _ { t } \mid y _ { < t } , \mathbf { I } , \mathbf { Z } , \tilde { \mathbf { R } } \Big ) ,\tag{11}
$$

where $T$ is the target sequence length and θ denotes the trainable LoRA parameters. This encourages coherent generation under both retrieval-available and retrieval-absent conditions.

## 3.3 RETRIEVAL UTILITY-GUIDED REINFORCEMENT FINE-TUNING

SFT optimizes token-level likelihood and provides no explicit signal for whether retrieved contexts are useful. We therefore introduce RUG-RFT to optimize sentence-level quality while regulating retrieval reliance. RUG-RFT combines a Quality Reward with a Retrieval-Utility Reward under Group Relative Policy Optimization (GRPO) (Shao et al., 2024), which estimates advantages from intra-group reward statistics without a separate value model.

GRPO with group sampling. Given an input (I, Z, R), the current policy $\pi _ { \theta }$ samples N candidate outputs $\bar { \{ O _ { 1 } , \ldots , O _ { N } \} }$ , each scored by $\mathcal { R } ( O _ { i } )$ to obtain $r _ { i } .$ . GRPO standardizes rewards within the group to compute

$$
A _ { i } = { \frac { r _ { i } - \operatorname* { m e a n } \{ r _ { 1 } , \dots , r _ { N } \} } { \operatorname { s t d } \{ r _ { 1 } , \dots , r _ { N } \} } } .\tag{12}
$$

Let $c _ { 1 } = \pi _ { \theta } ( O _ { i } \mid \mathbf { I } , \mathbf { Z } , \mathbf { R } ) / \pi _ { \mathrm { o l d } } ( O _ { i } \mid \mathbf { I } , \mathbf { Z } , \mathbf { R } )$ and $c _ { 2 } = \mathrm { c l i p } ( c _ { 1 } , 1 - \varepsilon , 1 + \varepsilon )$ . The policy is updated by maximizing

$$
\mathcal { I } _ { \mathrm { G R P O } } ( \boldsymbol { \theta } ) = \mathbb { E } \left[ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { m i n } ( c _ { 1 } A _ { i } , c _ { 2 } A _ { i } ) - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } \lVert \boldsymbol { \pi } _ { \mathrm { r e f } } ) \right] ,\tag{13}
$$

where the KL term regularizes deviation from the reference policy $\pi _ { \mathrm { r e f } }$

## 3.3.1 RETRIEVAL-UTILITY REWARD

We quantify the contribution of retrieval for a fixed output using teacher-forcing conditional likelihoods. Given (I, Z, R), the current policy π<sub>θ</sub> generates $Y \overset { \vartriangle } { \boldsymbol { = } } \left( y _ { 1 } , \ldots , y _ { T } \right)$ under the retrievalaugmented condition. We then score the same Y with retrieval using π<sub>θ</sub> and without retrieval using a frozen reference policy $\pi _ { \mathrm { r e f } } .$

$$
\ell ^ { \mathrm { w i t h } } ( Y ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \log p _ { \theta } ( y _ { t } \mid y _ { < t } , \mathbf { I } , \mathbf { Z } , \mathbf { R } ) ,\tag{14}
$$

$$
\ell ^ { \mathrm { w o } } ( Y ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \log p _ { \mathrm { r e f } } ( y _ { t } \mid y _ { < t } , \mathbf { I } , \mathbf { Z } ) .\tag{15}
$$

Here, $\pi _ { \mathrm { r e f } }$ is the frozen SFT checkpoint and is identical to the KL anchor in Eq. 13. Using $\pi _ { \mathrm { r e f } }$ for $\ell ^ { \mathrm { w o } } ( Y )$ prevents the policy from increasing the reward by artificially lowering its retrieval-free likelihood. We define the retrieval-induced likelihood gain as

$$
\Delta \ell _ { \mathrm { { r e t } } } ( Y ) = \ell ^ { \mathrm { { w i t h } } } ( Y ) - \ell ^ { \mathrm { { w o } } } ( Y ) ,\tag{16}
$$

where a positive value indicates that retrieval increases the model’s confidence in $Y ,$

## 3.3.2 QUALITY REWARD AND REWARD COMPOSITION

To align with task metrics while regulating retrieval reliance, we combine a sentence-level Quality Reward with the Retrieval-Utility Reward. Let $\mathcal { R } _ { \mathrm { q u a l } } ( Y )$ denote a translation-quality score, such as BLEU or ROUGE, computed against the ground truth. We retain only positive retrieval utility:

$$
\mathcal { R } _ { \mathrm { u t i l } } ^ { + } ( Y ) = \operatorname* { m a x } ( 0 , \Delta \ell _ { \mathrm { r e t } } ( Y ) ) .\tag{17}
$$

To balance the magnitudes of the two rewards, we introduce

$$
\gamma = \frac { \mathbb { \widehat E } \big [ | \mathcal { R } _ { \mathrm { q u a l } } | \big ] } { \operatorname* { m a x } \Big ( \mathbb { \widehat E } \big [ \mathcal { R } _ { \mathrm { u t i l } } ^ { + } \big ] , \tau \Big ) } ,\tag{18}
$$

where ${ \widehat { \mathbb { E } } } [ \cdot ]$ is an exponential moving average over recent batches, τ prevents early instability, and γ is treated as stop-gradient. The total reward used in Eq. 12 is

$$
\mathcal { R } _ { \mathrm { t o t a l } } ( Y ) = \mathcal { R } _ { \mathrm { q u a l } } ( Y ) + \alpha \cdot \gamma \cdot \mathcal { R } _ { \mathrm { u t i l } } ^ { + } ( Y ) ,\tag{19}
$$

where α controls the strength of the Retrieval-Utility Reward.

## 4 EXPERIMENT

## 4.1 EXPERIMENTAL SETUP

Datasets and metrics. We use CSL-News (Li et al., 2025b) and YouTube-ASL (Uthus et al., 2023) as large-scale pretraining datasets for Chinese Sign Language and American Sign Language, respectively. We then evaluate SignRAG on four downstream SLT benchmarks: CSL-Daily (Zhou et al., 2021), PHOENIX-2014T (Camgoz et al., 2018), How2Sign (Duarte et al., 2021), and OpenASL (Shi et al., 2022). Their train/validation/test splits are 18,401/1,077/1,176, 7,096/519/642, 31,085/1,739/2,348, and 96,476/966/975, respectively. We report BLEU-1/2/3/4 (Papineni et al., 2002) and ROUGE-L (Lin, 2004) following standard SLT evaluation protocols.

Table 1: Main results on four SLT benchmarks. “w/ gloss” and “w/o gloss” denote methods with and without gloss supervision, respectively. “Ext.” indicates whether external sign-language datasets are used for pre-training. For the “w/o gloss” setting, methods with and without external pre-training data are separated and compared independently. Bold scores indicate the best results within each dataset, gloss setting, and external-data setting. “Venue” denotes the publication venue and year.
<table><tr><td>Dataset</td><td>Gloss</td><td>Method</td><td>Venue</td><td>Ext.</td><td>R-L↑ 49.31</td><td>B@1↑</td><td>B@2↑</td><td>B@3↑</td><td>B@4↑</td></tr><tr><td rowspan="3">CS--Daiy</td><td>w/ gloss</td><td>SignBT (Zhou et al., 2021) SLTUNET (Zhang et al., 2023) TS-SLT (Chen et al., 2022c) CV-SLT (Zhao et al., 2024) GFSLT-VLP (Zhou et al., 2023) Sign2GPT (Wong et al., 2024)</td><td>CVPR&#x27;21 ICLR&#x27;23 NeurIPS&#x27;22 AAAI&#x27;24 ICCV’23 ICLR&#x27;24</td><td>x x x x</td><td>54.08 55.72 57.06 36.44</td><td>51.42 54.98 55.44 58.29 39.37</td><td>37.26 41.44 42.59 45.15 24.93</td><td>27.76 31.84 32.87 35.77 16.26</td><td>21.34 25.01 25.79 28.94 11.00 15.40</td></tr><tr><td>VAP (Jiao et al., 2024) C2RL (Chen et al., 2025) RVLF (Rao et al., 2026) w/o gloss SignRAG-SFT SignRAG-RFT</td><td></td><td>ECCV&#x27;24 TCSVT’25 Findings CVPR’26</td><td>××× x x x x</td><td>42.36 48.56 48.21 55.92 55.43 56.36</td><td>41.75 49.99 49.32 54.96 54.22 56.60</td><td>28.73 36.28 42.45 41.07 43.87</td><td>20.60 一 27.54 33.33 32.21 34.43</td><td>20.85 21.61 26.71 26.06 27.73</td></tr><tr><td>Uni-Sign (Li et al., 2025b) SignRAG-SFT SignRAG-RFT SignBT (Zhou et al., 2021)</td><td>Geo-Sign (Fish &amp; Bowden, 2026)</td><td>ICLR&#x27;25 NeurIPS&#x27;25 CVPR&#x27;21</td><td></td><td>56.51 57.97 60.20 62.34 49.54</td><td>55.08 55.89 59.39 61.12</td><td>42.14 46.94 48.74</td><td>32.98 37.95 39.73</td><td>26.36 27.42 31.22 32.91</td></tr><tr><td rowspan="2">PHO11-14T w/o gloss</td><td>w/ gloss TS-SLT (Chen et al., 2022c) CV-SLT (Zhao et al., 2024) GFSLT-VLP (Zhou et al., 2023) Sign2GPT (Wong et al., 2024)</td><td></td><td>NeurIPS&#x27;22 AAAI&#x27;24 ICCV’23 ICLR&#x27;24</td><td>×× x x x</td><td>53.48 54.33 42.49 48.90</td><td>50.80 54.90 54.88 43.71</td><td>37.75 42.43 42.68 33.18</td><td>29.72 34.46 34.79 26.11</td><td>24.32 28.95 29.27 21.44</td></tr><tr><td>FLa-LLM (Chen et al., 2024) VAP (Jiao et al., 2024) C²RL (Chen et al., 2025) RVLF (Rao et al., 2026) SignRÁG-SFT SignRAG-RFT GloFE-VN (Lin et al., 2023)</td><td>ECCV&#x27;24 TCSVT’25 ACL&#x27;23</td><td>LREC-COLING’24 Findings CVPR’26 x</td><td>x x x x ××</td><td>45.27 51.28 50.96 52.56 52.14 53.07 12.6 14.9</td><td>49.54 46.29 53.07 52.81 53.60 53.05 54.09</td><td>35.96 35.33 40.20 41.20 40.88 42.62 7.3</td><td>28.83 28.03 32.20 33.24 32.68 34.31 3.9</td><td>22.52 23.09 26.16 26.75 27.86 27.30 28.63</td></tr><tr><td rowspan="2">Howign w/o gloss</td><td>OpenSLT (Tarrés et al., 2023) C2RL (Chen et al., 2025) FLa-LLM (Chen et al., 2024) VAP (Jiao et al., 2024) RVLF (Rao et al., 2026) SignRAG-SFT SignRAG-RFT</td><td>TCSVT’25 ECCV&#x27;24 NeurIPS D&amp;B&#x27;23</td><td>CVPRW/WiCV’23 x x LREC-COLING&#x27;24 x x Findings CVPR&#x27;26 x x x</td><td>27.0 27.8 27.8 34.7 34.7 36.2</td><td></td><td>34.0 29.1 29.8 39.2 39.4 40.2 42.9</td><td>19.3 18.6 19.0 25.3 25.9 28.0</td><td>12.2 12.9 13.3 18.6 18.4 20.3</td><td>2.2 8.0 9.4 9.7 12.9 14.3 13.7 15.2</td></tr><tr><td>YouTube-ASL (Uthus et al., 2023) SignMusketeers (Gueuwou et al., 2025) Findings ACL&#x27;25 Uni-Sign (Li et al., 2025b) SSVP-ŠLT (Rust et al., 2024) SignRAG-SFT SignRAG-RFT</td><td>ICLR&#x27;25 ACL&#x27;24</td><td>√ √ √ V VV</td><td>36.0 38.4 36.7 38.5</td><td>37.8 41.5 40.2 43.2 43.3 44.7</td><td></td><td>24.1 27.2 27.1 28.8 28.6 30.3</td><td>16.9 19.3 19.7 20.8 20.7 22.1</td><td>12.4 14.3 14.9 15.5 15.6 16.7</td></tr><tr><td rowspan="2">OPASL</td><td>GÎoFE-VN (Lin et al., 2023) C2RL (Chen et al., 2025) VAP (Jiao et al., 2024) w/o gloss RVLF (Rao et al., 2026) SignRÁG-SFT</td><td>OpenASL (Shi et al., 2022)</td><td>EMNLP&#x27;22 ACL&#x27;23 TCSVT&#x27;25 ECCV&#x27;24 Findings CVPR’26</td><td>×× x x x x</td><td>21.02 21.75 31.36 41.38 43.92 42.96</td><td>20.92 21.56 31.46 45.92 48.73 46.24</td><td>12.08 12.74 21.85 35.40 34.25</td><td>8.59 9.05 16.58 27.99 27.12</td><td>6.72 7.06 13.21 21.23 22.84</td></tr><tr><td>SignRAG-RFT Uni-Sign (Li et al., 2025b) SignRÄG-SFT SignRAG-RFT</td><td>ICLR&#x27;25</td><td>x √</td><td>43.87 43.22 45.33 46.91</td><td></td><td>49.14 49.35 50.31 51.56</td><td>36.58 36.22 38.45 39.52</td><td>29.01 28.55 30.28 32.06</td><td>22.24 23.61 23.14 25.12 26.62</td></tr></table>

Retrieval gallery protocol. For each dataset, we build a separate train-only retrieval gallery from its training split, where each entry stores the encoded sign representation and paired translation. Validation and test samples are used only as queries and are never included in the gallery; during training, the current query instance is excluded to prevent self-retrieval. The details are shown in Appendix C.7. A detailed discussion of the efficiency, computational cost, and retrieval scalability of SignRAG is provided in Appendix E.

Implementation details. We adopt CoSign (Jiao et al., 2023) with a three-layer Transformerbased skeleton encoder and use Qwen3-8B as the decoder-only LLM across all datasets. All experiments are conducted on four NVIDIA RTX 4090D GPUs, each equipped with 48 GB of memory, using bfloat16 precision. The first and second pre-training stages are run for 200 and 50 epochs, respectively, followed by 10 epochs of SFT and 2 epochs of RFT. We use SGD with a learning rate of 0.01 in the first pre-training stage and AdamW in all subsequent stages. The learning rates for SFT and RFT are set to $1 \times 1 0 ^ { - 4 }$ and $1 \times 1 0 ^ { - 5 }$ , respectively, with a retrieval dropout probability of $p = 0 . 5$ . Checkpoints are selected based on validation BLEU-4 scores. For the number of retrieved items, we set $\bar { K = 3 }$ . Full implementation details are provided in Appendix C.

Table 2: Ablation studies on CSL-Daily. Components in (a) are added cumulatively. Reward variants in (b) start from the same retrieval-augmented SFT model.
<table><tr><td>Configuration</td><td>R-L↑</td><td>B@1↑</td><td>B@2↑</td><td>B@3↑</td><td>B@4↑</td></tr><tr><td>(a) Framework components</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SFT baseline</td><td>55.03</td><td>51.16</td><td>37.99</td><td>29.40</td><td>23.46</td></tr><tr><td>+ Pretraining</td><td>58.78</td><td>56.81</td><td>44.85</td><td>36.18</td><td>29.72</td></tr><tr><td>+ Retrieval</td><td>60.20</td><td>59.39</td><td>46.97</td><td>37.95</td><td>31.22</td></tr><tr><td>+ RUG-RFT</td><td>62.34</td><td>61.12</td><td>48.74</td><td>39.73</td><td>32.91</td></tr><tr><td>(b) Reward design</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Retrieval-augmented SFT</td><td>60.20</td><td>59.39</td><td>46.97</td><td>37.95</td><td>31.22</td></tr><tr><td>Quality Reward only</td><td>61.67</td><td>60.76</td><td>48.42</td><td>38.97</td><td>31.97</td></tr><tr><td>Retrieval-Utility Reward only</td><td>61.37</td><td>60.55</td><td>47.93</td><td>38.62</td><td>31.66</td></tr><tr><td>Both rewards</td><td>62.34</td><td>61.12</td><td>48.74</td><td>39.73</td><td>32.91</td></tr></table>

## 4.2 MAIN RESULTS

Table 1 summarizes the main results. SignRAG outperforms existing state-of-the-art methods across all evaluated datasets. In particular, on CSL-Daily, SignRAG further surpasses gloss-based approaches, suggesting that large-scale pretraining on external sign language data, together with the capabilities of modern LLMs, can partially compensate for the absence of costly gloss annotations. Even without pretraining on additional data, SignRAG achieves state-of-the-art performance under the same setting. The training protocol follows RVLF (Rao et al., 2026). It is worth noting that, due to the lack of a large-scale German sign language pretraining dataset, all results on PHOENIX-2014T are obtained without pretraining on additional data. Compared with the Chinese datasets, the gains of SignRAG over existing state-of-the-art methods are more moderate on the English datasets. This discrepancy may be attributed to differences in dataset difficulty and evaluation protocols between Chinese and English benchmarks. In addition, the YouTube-ASL data used for pretraining are incomplete, which may limit the effectiveness of large-scale pretraining on English sign language translation. Further details are provided in Appendix D.4. For a fair comparison, we do not include methods that use substantially larger pretraining datasets or introduce modules specifically designed for English SLT datasets (Zhang et al., 2024; Jang et al., 2025).

## 4.3 ABLATION STUDY

Table 2(a) evaluates the contributions of pretraining, retrieval augmentation, and RUG-RFT on CSL-Daily. Pretraining improves ROUGE-L from 55.03 to 58.78 and BLEU-4 from 23.46 to 29.72. Adding retrieval further raises them to 60.20 and 31.22, respectively. With RUG-RFT, performance reaches 62.34 ROUGE-L and 32.91 BLEU-4, with consistent gains across all metrics. These results demonstrate the effectiveness of progressively introducing pretraining, retrieval, and retrieval utilityguided reinforcement fine-tuning. Table 2(b) analyzes the reward design. Starting from the same retrieval-augmented SFT model, both the Quality Reward and Retrieval-Utility Reward improve performance independently. Their combination achieves the best results across all metrics, reaching 62.34 ROUGE-L and 32.91 BLEU-4. This confirms that translation quality and retrieval utility provide complementary optimization signals for RUG-RFT.

## 4.4 ANALYSIS

Retrieval quality analysis. We investigate the relationship between the number of retrieved contexts, retrieval quality, and final translation performance. For each query, we vary the number of retrieved contexts K and measure the token-level overlap between the retrieved texts and the groundtruth translation. The resulting scores are then averaged over all queries to reflect the overall retrieval quality. Chinese sentences are segmented using Jieba, with punctuation, empty tokens, and symbolonly tokens removed. Let $\tau _ { \mathrm { G T } }$ denote the token set of the ground-truth translation, and let $\mathcal { T } _ { \mathrm { r e t } } ^ { ( K ) }$ denote the token set aggregated from the top-K retrieved translations. We define Retrieval Recall, Retrieval Precision, and Retrieval F1 as

![](images/c3b9642772b610c8780ad2cfaabb3ca4aeb77807109e932b7711fa4af46ce8b8.jpg)  
(a) Retrieval quality vs. retrieval number

![](images/11645e656023632a538162f3dd26464e1777856a411bdefe00c771a8b25a5447.jpg)  
(b) BLEU-4 vs. LLM size  
Figure 3: Main analysis. (a) Relationship between retrieval quality and final translation performance under different numbers of retrieved contexts. (b) BLEU-4 scaling with LLM size. We evaluate Qwen3 backbones with different parameter scales (0.6B, 4B, and 8B) under the same training and evaluation protocol.

$$
\begin{array} { r } { \mathrm { R e c } _ { \mathrm { r e t } } ( K ) = \frac { | { \mathcal T } _ { \mathrm { G T } } \cap { \mathcal T } _ { \mathrm { r e t } } ^ { ( K ) } | } { | { \mathcal T } _ { \mathrm { G T } } | } , ~ \mathrm { P r e c } _ { \mathrm { r e t } } ( K ) = \frac { | { \mathcal T } _ { \mathrm { G T } } \cap { \mathcal T } _ { \mathrm { r e t } } ^ { ( K ) } | } { | { \mathcal T } _ { \mathrm { r e t } } ^ { ( K ) } | } , ~ \mathrm { F } 1 _ { \mathrm { r e t } } ( K ) = \frac { 2 \mathrm { R e c } _ { \mathrm { r e t } } ( K ) \mathrm { P r e c } _ { \mathrm { r e t } } ( K ) } { \mathrm { R e c } _ { \mathrm { r e t } } ( K ) + \mathrm { P r e c } _ { \mathrm { r e t } } ( K ) } . } \end{array}\tag{20}
$$

Here, $\mathrm { R e c } _ { \mathrm { r e t } }$ measures how much ground-truth lexical information is covered by the retrieved con texts, while $\mathrm { P r e c } _ { \mathrm { r e t } }$ measures how much of the retrieved lexical content is relevant to the ground truth. Their harmonic mean, $\mathrm { F 1 } _ { \mathrm { r e t } } ,$ captures the trade-off between retrieval coverage and retrieval noise. As shown in Fig. 3a, as K increases, $\mathrm { F 1 } _ { \mathrm { r e t } }$ first improves and then continuously decreases, and the final translation performance exhibits the same trend. This is because introducing additional less-relevant contexts can instead interfere with SLT performance. All BLEU-4 scores in Fig. 3a are obtained from SFT models.

Model scaling analysis. We further investigate how the scale of the language model affects translation performance. As shown in Fig. 3b, under the same training and evaluation protocol, BLEU-4 increases from 22.72 to 28.58 and 32.91 when scaling the Qwen3 backbone from 0.6B to 4B and 8B, respectively. The consistent improvement shows that contemporary LLMs also exhibit stronger capabilities in sign language translation. These results further indicate that SignRAG can effectively benefit from scaling the underlying LLM.

Case study. We provide a qualitative comparison with the current SOTA model, Geo-Sign (Fish & Bowden, 2026), to analyze the effect of retrieved contexts. As shown in Table 3, the reference sentence contains fine-grained semantic cues, including “Traditionally” and “lucky money.” Although Geo-Sign produces a partially related translation, it fails to preserve these cues. In contrast, SignRAG retrieves semantically aligned contexts that provide relevant topical and lexical cues. This further supports our finding that retrieval can improve translation performance. Additional qualitative examples are provided in Table 15 in the Appendix. Meanwhile, the bad case presented in Appendix D.7 shows that noisy retrieval can mislead the model, further motivating RUG-RFT.

<table><tr><td>Source</td><td>Sentence</td></tr><tr><td>Reference Retrieved</td><td>红包在传统意义上也叫压岁钱。(Traditionally, a red envelope is also called “lucky money.&quot;) [1] 你打算怎么花这笔压岁钱呢? (How do you plan to spend this lucky money?) [2] 带着祝福的红包比攀比 中的计较更有意义。(A red envelope carrying good wishes is more meaningful than one given for comparison or competition.) [3] 你打算怎么花这笔压岁钱呢? (How do you plan to spend this lucky money?)</td></tr><tr><td>Geo-Sign</td><td>红包在历史上的原意是成年礼。(Historically, the original meaning of a red envelope was a coming-of-age gift.)</td></tr><tr><td>SignRAG</td><td>红包在传统意义上是压岁钱。(Traditionally, a red envelope means “lucky money.&quot;)</td></tr></table>

Table 3: Qualitative comparison on CSL-Daily.

## 5 CONCLUSION

We propose SignRAG, a unified framework for gloss-free sign language translation. SignRAG strengthens sign representation learning through staged pretraining and leverages target-domain semantic memory to provide instance-specific linguistic cues for downstream translation. Furthermore, RUG-RFT jointly considers translation quality and retrieval utility to encourage effective use of beneficial retrieved contexts. Experiments across multiple SLT benchmarks show that SignRAG achieves new state-of-the-art performance and surpasses existing gloss-supervised methods across all reported metrics on CSL-Daily. Future work will explore more efficient and robust retrieval mechanisms, as well as knowledge transfer across datasets and languages.

## AI USE STATEMENT

We used generative AI tools primarily to assist with writing and language polishing, including improvements to grammar, wording, clarity, and readability. We did not use generative AI tools for research design, experimental design, data analysis, or result interpretation. All AI-assisted text was carefully reviewed and revised by the authors, who take full responsibility for the final content of the manuscript.

## REPRODUCIBILITY STATEMENT

Although we have provided the experimental settings, training protocols, and implementation details as comprehensively as possible in the main paper and Appendix, we recognize that reproducibility is essential for impactful scientific research. To facilitate further verification and future research, upon acceptance, we will release the source code, pretrained checkpoints from different training stages, and models of different sizes. We hope these resources will support reproducible evaluation and further academic research.

## REFERENCES

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

Necati Cihan Camgoz, Simon Hadfield, Oscar Koller, Hermann Ney, and Richard Bowden. Neural sign language translation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7784–7793, 2018.

Necati Cihan Camgoz, Oscar Koller, Simon Hadfield, and Richard Bowden. Sign language transformers: Joint end-to-end sign language recognition and translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10023–10033, 2020.

Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W. Cohen. MuRAG: Multimodal retrieval-augmented generator for open question answering over images and text. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 5558–5570, 2022a.

Yutong Chen, Fangyun Wei, Xiao Sun, Zhirong Wu, and Stephen Lin. A simple multi-modality transfer learning baseline for sign language translation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5120–5130, 2022b.

Yutong Chen, Ronglai Zuo, Fangyun Wei, Yu Wu, Shujie Liu, and Brian Mak. Two-stream network for sign language recognition and translation. Advances in Neural Information Processing Systems, 35:17043–17056, 2022c.

Zhigang Chen, Benjia Zhou, Jun Li, Jun Wan, Zhen Lei, Ning Jiang, Quan Lu, and Guoqing Zhao. Factorized learning assisted with large language model for gloss-free sign language translation. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, pp. 7071–7081, 2024.

Zhigang Chen, Benjia Zhou, Yiqing Huang, Jun Wan, Yibo Hu, Hailin Shi, Yanyan Liang, Zhen Lei, and Du Zhang. C<sup>2</sup>rl: Content and context representation learning for gloss-free sign language translation and retrieval. IEEE Transactions on Circuits and Systemsfor Video Technology, 2025. doi: 10.1109/TCSVT.2025.3553052.

Yiting Cheng, Fangyun Wei, Jianmin Bao, Dong Chen, and Wenqiang Zhang. Cico: Domain-aware sign language retrieval via cross-lingual contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19016–19026, 2023.

Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and Lidong Bing. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-LLMs. arXiv preprint arXiv:2406.07476, 2024.

Amanda Duarte, Shruti Palaskar, Lucas Ventura, Deepti Ghadiyaram, Kenneth DeHaan, Florian Metze, Jordi Torres, and Xavier Giró-i Nieto. How2sign: A large-scale multimodal dataset for continuous american sign language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2735–2744, 2021.

Edward Fish and Richard Bowden. Geo-sign: Hyperbolic contrastive regularisation for geometrically aware sign language translation. Advances in Neural Information Processing Systems, 38: 99293–99330, 2026.

Jia Gong, Lin Geng Foo, Yixuan He, Hossein Rahmani, and Jun Liu. LLMs are good sign language translators. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18362–18372, 2024.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

Shester Gueuwou, Xiaodan Du, Greg Shakhnarovich, and Karen Livescu. Signmusketeers: An efficient multi-stream approach for sign language translation at scale. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pp. 22506–22521, 2025. doi: 10.18653/v1/2025. findings-acl.1157.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Jerry Huang, Siddarth Madala, Risham Sidhu, Cheng Niu, Hao Peng, Julia Hockenmaier, and Tong Zhang. RAG-RL: Advancing retrieval-augmented generation via reinforcement learning and curriculum learning. arXiv preprint arXiv:2503.12759, 2025.

Yu Huang, Junyang Lin, Chang Zhou, Hongxia Yang, and Longbo Huang. Modality competition: What makes joint training of multi-modal network fail in deep learning? (provably). In Proceed ings ofthe 39th International Conference on Machine Learning (ICML), 2022.

Youngjoon Jang, Liliane Momeni, Zifan Jiang, Joon Son Chung, Gül Varol, and Andrew Zisserman. Lost in translation, found in embeddings: Sign language translation and alignment. arXiv preprint arXiv:2512.08040, 2025.

Peiqi Jiao, Yuecong Min, Yanan Li, Xiaotao Wang, Lei Lei, and Xilin Chen. Cosign: Exploring co-occurrence signals in skeleton-based continuous sign language recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20676–20686, 2023.

Peiqi Jiao, Yuecong Min, and Xilin Chen. Visual alignment pre-training for sign language translation. In Computer Vision – ECCV 2024, pp. 349–367, 2024. doi: 10.1007/978-3-031-72946-1\_ 20.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pp. 9459–9474, 2020.

Dongxu Li, Chenchen Xu, Xin Yu, Kaihao Zhang, Benjamin Swift, Hanna Suominen, and Hongdong Li. Tspnet: Hierarchical feature learning via temporal semantic pyramid for sign language translation. Advances in Neural Information Processing Systems, 33:12034–12045, 2020.

Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan Cheng, Bo Wang, Yining Zheng, Yuxin Wang, Zhangyue Yin, and Xipeng Qiu. R3-RAG: Learning step-by-step reasoning and retrieval for LLMs via reinforcement learning. arXiv preprint arXiv:2505.23794, 2025a.

Zecheng Li, Wengang Zhou, Weichao Zhao, Kepeng Wu, Hezhen Hu, and Houqiang Li. Uni-sign: Toward unified sign language understanding at scale. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview.net/forum?id= 0Xt7uT04cQ.

Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pp. 74–81, 2004.

Kezhou Lin, Xiaohan Wang, Linchao Zhu, Ke Sun, Bang Zhang, and Yi Yang. Gloss-free end-toend sign language translation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12904–12916, 2023. doi: 10.18653/v1/ 2023.acl-long.722.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems, 36:34892–34916, 2023.

Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. Multilingual denoising pre-training for neural machine translation. Transactions ofthe Associationfor Computational Linguistics, 8:726–742, 2020.

Brandon McKinzie, Zhe Gan, Jean-Philippe Fatahalian, Alexander Toshev, Philipp Dufter, Alon Shalit, and Yinfei Yang. Mm1: Methods, analysis & insights from multimodal LLM pre-training. In Proceedings of the European Conference on Computer Vision (ECCV), 2024.

MMPose Contributors. OpenMMLab pose estimation toolbox and benchmark. https:// github.com/open-mmlab/mmpose, 2020.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: A method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pp. 311–318, 2002.

Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, and Di Hu. Balanced multimodal learning via on-the-fly gradient modulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8238–8247, June 2022.

Maja Popovic. chrF: Character n-gram f-score for automatic mt evaluation. In´ Proceedings of the Tenth Workshop on Statistical Machine Translation, pp. 392–395, 2015.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pp. 8748–8763, 2021.

Zhi Rao, Yucheng Zhou, Benjia Zhou, Yiqing Huang, Sergio Escalera, and Jun Wan. Rvlf: A reinforcing vision-language framework for gloss-free sign language translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pp. 9237–9247, June 2026.

Phillip Rust, Bowen Shi, Skyler Wang, Necati Cihan Camgöz, and Jean Maillard. Towards privacyaware sign language translation at scale. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8624–8641, 2024. doi: 10.18653/v1/2024.acl-long.467.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Bowen Shi, Diane Brentari, Gregory Shakhnarovich, and Karen Livescu. Open-domain sign language translation learned from online video. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pp. 6365–6379, 2022.

Garrett Tanzer and Biao Zhang. Youtube-sl-25: A large-scale, open-domain multilingual sign language parallel corpus. In International Conference on Learning Representations, volume 2025, pp. 81921–81934, 2025.

Laia Tarrés, Gerard I. Gállego, Amanda Duarte, Jordi Torres, and Xavier Giró-i Nieto. Sign language translation from instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 5625–5635, 2023.

Dave Uthus, Garrett Tanzer, and Manfred Georg. Youtube-ASL: A large-scale, open-domain american sign language-english parallel corpus. Advances in Neural Information Processing Systems, 36:29029–29047, 2023.

Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang, and Feng Zhao. VRAG-RL: Empower vision-perception-based RAG for visually rich information understanding via iterative reasoning with reinforcement learning. arXiv preprint arXiv:2505.22019, 2025.

Weiyao Wang, Du Tran, and Matt Feiszli. What makes training multi-modal classification networks hard? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Ryan Wong, Necati Cihan Camgoz, and Richard Bowden. Sign2GPT: Leveraging large language models for gloss-free sign language translation. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= LqaEEs3UxU.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Huijie Yao, Wengang Zhou, Hao Zhou, Hezhen Hu, and Houqiang Li. Retrieval-augmented sign language translation. ACM Transactions on Multimedia Computing, Communications and Applications, 22(2), 2026. doi: 10.1145/3771277.

Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. Mplug-owi2: Revolutionizing multi-modal large language model with modality collaboration. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13040–13051. IEEE, 2024.

Aoxiong Yin, Tianyun Zhong, Li Tang, Weike Jin, Tao Jin, and Zhou Zhao. Gloss attention for gloss-free sign language translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2551–2562, 2023.

Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022.

Biao Zhang, Mathias Müller, and Rico Sennrich. SLTUNET: A simple unified model for sign language translation. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=EBS4C77p\_5S.

Biao Zhang, Garrett Tanzer, and Orhan Firat. Scaling sign language translation. Advances in neural information processing systems, 37:114018–114047, 2024.

Jian Zhao, Weizhen Qi, Wengang Zhou, Nan Duan, Ming Zhou, and Houqiang Li. Conditional sentence generation and cross-modal reranking for sign language translation. IEEE Transactions on Multimedia, 24:2662–2672, 2021.

Rui Zhao, Liang Zhang, Biao Fu, Cong Hu, Jinsong Su, and Yidong Chen. Conditional variational autoencoder for sign language translation with cross-modal alignment. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19643–19651, 2024.

Benjia Zhou, Zhigang Chen, Albert Clapés, Jun Wan, Yanyan Liang, Sergio Escalera, Zhen Lei, and Du Zhang. Gloss-free sign language translation: Improving from visual-language pretraining. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20871–20881, 2023.

Hao Zhou, Wengang Zhou, Weizhen Qi, Junfu Pu, and Houqiang Li. Improving sign language translation with monolingual data by sign back-translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1316–1325, 2021.

Qidan Zhu, Jing Li, Fei Yuan, Jiaojiao Fan, and Quan Gan. A chinese continuous sign language dataset based on complex environments. arXiv preprint arXiv:2409.11960, 2024.

## APPENDIX

A Extended Related Work 16   
A.1 Sign Language Translation 16   
A.2 RL for Retrieval-Enhanced Reasoning 16   
B Detailed Formulations and Derivations 17   
B.1 Vision-Language Pre-training . 17   
B.2 Attention-weighted Skeleton–Text Similarity 17   
B.3 Bidirectional InfoNCE Loss 17   
B.4 Memory-bounded distributed retrieval. 17   
C More Implementation Details 18   
C.1 Computing Infrastructure 19   
C.2 Keypoint Extraction. 19   
C.3 Pretraining(Stage I): Sign-text pre-training. 20   
C.4 Pretraining(Stage II): Large-scale sign MLLM pretraining. 20   
C.5 SFT: Retrieval-Augmented Supervised Fine-Tuning. 20   
C.6 RFT: Retrieval Utility-Guided Reinforcement Fine-Tuning. 21   
C.7 Identical-text filtering in training retrieval. 21   
D Further Analyses 22   
D.1 Analysis of different pretraining strategies. 22   
D.2 Directly Fine-Tuning a Video LLM. 22   
D.3 Why Does Hierarchical Pretraining Benefit Sign MLLMs? 23   
D.4 Dataset statistics. 24   
D.5 Metric overfitting analysis. 24   
D.6 Transferability across Chinese SLT benchmarks. . 25   
D.7 Failure Case (Harmful Reliance on Retrieved Contexts). 25   
E Efficiency, Computational Cost, and Retrieval Scalability 25   
E.1 Online SLT Inference Efficiency 25   
E.2 Offline Retrieval Efficiency on CSL-Daily 26   
E.3 Retrieval Scalability on OpenASL 27   
E.4 Training Cost 27   
E.5 Discussion on Million-Scale Retrieval 27   
F More Cases 28

## A EXTENDED RELATED WORK

This section provides the full version of the related-work discussion that is condensed into a single paragraph in the main text.

## A.1 SIGN LANGUAGE TRANSLATION

Sign language translation (SLT) aims to recover the semantic content conveyed by a signer from a sequence of performed signs. Unlike standard video understanding tasks, SLT must account for the unique linguistic structure of sign language. Specifically, sign sequences are more naturally aligned with gloss units, whereas spoken language sentences often follow substantially different grammatical patterns. This mismatch makes direct translation challenging. As a result, early SLT methods (Camgoz et al., 2020; Zhou et al., 2021; Chen et al., 2022c;b) commonly relied on gloss supervision, typically by introducing auxiliary Sign Language Recognition modules or explicitly modeling gloss generation before sentence translation. Despite their strong performance, such methods depend heavily on costly gloss annotations, which limits scalability. To address this issue, recent work has increasingly focused on gloss-free SLT, which learns a direct mapping from sign videos to spoken language sentences. Early studies (Camgoz et al., 2018; 2020; Li et al., 2020; Zhao et al., 2021; Yin et al., 2023) demonstrated the feasibility of this setting, although a clear performance gap from gloss-based methods remains. More recent efforts attempt to close this gap by improving visual-text representation learning. For example, GFSLT (Zhou et al., 2023) adapts CLIP (Radford et al., 2021) to sign language, inspiring a series of vision-language SLT models (Wong et al., 2024; Chen et al., 2024; Gong et al., 2024; Jiao et al., 2024; Chen et al., 2025). Another line of work improves SLT from the data perspective. Recent studies (Uthus et al., 2023; Rust et al., 2024; Li et al., 2025b; Zhang et al., 2024; Gueuwou et al., 2025; Jang et al., 2025; Fish & Bowden, 2026) show that pre-training on large-scale sign corpora can significantly benefit downstream translation. However, most existing efforts focus on pre-training T5-based encoder–decoder architectures, whose language priors, general knowledge, and generative capabilities remain limited compared with contemporary LLMs. Therefore, how to design effective pre-training strategies for adapting sign representations to contemporary LLM-based SLT models remains largely unexplored.

## A.2 RL FOR RETRIEVAL-ENHANCED REASONING

Retrieval-augmented generation (RAG) enhances parametric language models by incorporating nonparametric external contexts, where retrieved evidence is used as additional information during generation (Lewis et al., 2020; Chen et al., 2022a). By providing explicit external references, RAG improves knowledge utilization and reduces the dependence on purely parametric memory. Beyond general text generation, retrieval-based approaches have also been explored in sign language translation. However, existing retrieval-based SLT methods (Yao et al., 2026) mainly focus on gloss-based settings, where retrieval is performed under the guidance of gloss representations.

In parallel, reinforcement learning (RL) has become an effective post-training strategy for improving the reasoning and generation abilities of large language models. Group Relative Policy Optimization (GRPO) (Shao et al., 2024; Guo et al., 2025) provides an efficient RL framework by estimating relative advantages among grouped samples without requiring an additional value model. Recent studies have explored combining RL with RAG to optimize retrieval-enhanced reasoning and generation. RAG-RL (Huang et al., 2025) improves answer generation in retrieval contexts through reinforcement learning and curriculum strategies. R3-RAG (Li et al., 2025a) further enables joint reasoning and retrieval optimization with outcome and process rewards. VRAG-RL (Wang et al., 2025) extends this paradigm to visual-rich scenarios by jointly optimizing retrieval, reasoning, and generation.

These studies demonstrate the potential of RL for dynamically optimizing the interaction between retrieval and generation. However, existing RL-based RAG methods mainly target knowledgeintensive reasoning tasks, rather than multimodal translation with sign language inputs. In contrast, SignRAG investigates how large-scale sign pretraining should be organized under a modern LLM decoder, and complements parametric adaptation with a sign-to-text retrieval gallery and retrieval utility-guided reinforcement fine-tuning. Unlike previous retrieval-based SLT approaches,

SignRAG operates without gloss annotations and explicitly optimizes the usefulness of retrieved translation contexts during generation.

## B DETAILED FORMULATIONS AND DERIVATIONS

## B.1 VISION-LANGUAGE PRE-TRAINING

For completeness, we provide additional details of the vision-language pre-training procedure.

## B.2 ATTENTION-WEIGHTED SKELETON–TEXT SIMILARITY

Skeleton and text features are projected into a shared D-dimensional space and $\ell _ { 2 } \cdot$ -normalized. Given skeleton features $\{ f _ { i } \} _ { i = \cdot } ^ { M }$ and text features $\{ t _ { j } \} _ { j = 1 } ^ { L }$ , we compute token-level similarities

$$
E _ { i , j } = f _ { i } ^ { \top } t _ { j } .\tag{21}
$$

To calculate the global skeleton-to-text similarity $z _ { S , T }$ , a row-wise Softmax (Cheng et al., 2023) is applied to E to obtain the attention weights $\mathbf { \nabla } _ { P } :$

$$
P _ { i , j } = \frac { \exp ( E _ { i , j } / \tau ) } { \sum _ { k = 1 } ^ { L } \exp ( E _ { i , k } / \tau ) } ,\tag{22}
$$

where τ denotes a trainable temperature parameter. The global skeleton–text similarity is defined as

$$
z _ { S , T } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { L } P _ { i , j } \ : E _ { i , j } .\tag{23}
$$

The text-to-skeleton similarity is computed analogously.

## B.3 BIDIRECTIONAL INFONCE LOSS

Given a mini-batch of B paired samples, we construct batch-level similarity matrices and apply a bidirectional InfoNCE objective with temperature $\tau ^ { \prime }$

## B.4 MEMORY-BOUNDED DISTRIBUTED RETRIEVAL.

This section describes the implementation of memory-efficient retrieval in SignRAG. Let G denote a training gallery containing N sign sequences, each represented by frame-level features with channel dimension C. Loading the entire gallery into GPU memory requires storage proportional to $\mathcal { O } ( N L C )$ , where L denotes the padded temporal length. Comparing a query batch against the full gallery further introduces memory costs for pairwise similarities and intermediate computations. To support large galleries, we combine CPU-resident gallery storage, distributed sharding, and blockwise similarity computation, followed by exact distributed top-K aggregation.

CPU-resident gallery sharding. With R distributed workers, we partition the gallery into disjoint shards:

$$
\mathcal { G } = \bigcup _ { r = 1 } ^ { R } \mathcal { G } ^ { ( r ) } , \qquad \mathcal { G } ^ { ( r ) } \cap \mathcal { G } ^ { ( r ^ { \prime } ) } = \emptyset , \quad r \ne r ^ { \prime } .\tag{24}
$$

Each worker is responsible for approximately $N / R$ gallery samples. The encoded features of its local shard are retained in CPU memory and transferred to GPU only for the current similarity computation. Thus, no worker needs to keep its entire gallery shard, or the complete gallery, in GPU memory.

Block-wise similarity computation. We divide the queries into chunks $\mathcal { Q } _ { b }$ of at most $B _ { q }$ sequences and each local gallery shard into chunks $\mathcal { G } _ { c } ^ { ( r ) }$ of at most $B _ { g }$ sequences:

$$
| \mathcal { Q } _ { b } | \leq B _ { q } , \qquad | \mathcal { G } _ { c } ^ { ( r ) } | \leq B _ { g } .\tag{25}
$$

Each query chunk is broadcast to all workers. Worker r then compares it against the chunks of its local gallery shard in turn. For the current query–gallery block, the aggregated similarities are

$$
A _ { b , c } ^ { ( r ) } [ u , v ] = S \left( Q _ { u } ^ { b } , G _ { v } ^ { ( r , c ) } \right) , \qquad A _ { b , c } ^ { ( r ) } \in \mathbb { R } ^ { | \mathcal { Q } _ { b } | \times | \mathcal { G } _ { c } ^ { ( r ) } | } ,\tag{26}
$$

where $S ( \cdot , \cdot )$ is the attention-weighted sign–sign similarity defined in Eq. 8. After evaluating a block, its similarity scores are transferred to CPU memory, and the GPU storage occupied by the current gallery chunk can be released before loading the next one. The query chunk remains available while the worker processes its local gallery. Scores from successive gallery chunks are collected on CPU for local candidate selection.

GPU memory usage. Let L bound the padded temporal lengths of the active query and gallery chunks. The resident retrieval tensors on each GPU require

$$
\mathcal { M } _ { \mathrm { r e s i d e n t } } = \mathcal { O } \left( ( B _ { q } + B _ { g } ) L C + B _ { q } B _ { g } \right) .\tag{27}
$$

The first term accounts for the active query and gallery features, and the second accounts for their aggregated similarity scores. This bound excludes model parameters, temporary intermediate activations, and kernel workspace; the actual peak GPU memory also depends on the implementation of $S ( \cdot , \cdot )$ . For fixed chunk sizes and sequence lengths, the resident GPU storage is independent of the total gallery size N. The chunk sizes can therefore be adjusted to the available device memory. CPU storage still grows with the local gallery size and the similarity scores retained for candidate selection, and retrieval still evaluates every eligible query–gallery pair.

Candidate filtering. During training-set retrieval, we exclude the query itself, identified by its sample identity, and gallery samples whose reference translations are identical to that of the query. These exclusions prevent trivial retrieval of the query or entries with identical supervision. Let $y _ { x }$ denote the reference translation of sample x. The eligible local gallery for a training query q is

$$
{ \mathcal { E } } ^ { ( r ) } ( q ) = \left\{ g \in { \mathcal { G } } ^ { ( r ) } \ \middle | \ g \neq q , \ y _ { g } \neq y _ { q } \right\} .\tag{28}
$$

For development and test queries, retrieval is performed against the training gallery without using the query reference translation, so $\mathcal { E } ^ { ( r ) } ( q ) = \mathcal { G } ^ { ( r ) }$ . All exclusions are applied before local top- $. K$ selection to ensure that eligible candidates are not displaced by entries that would subsequently be removed.

Distributed top-K aggregation. After evaluating its local gallery, worker r selects

$$
\mathcal { C } _ { K } ^ { \left( r \right) } ( { q } ) = \mathrm { T o p K } _ { { g } \in { \mathcal { E } } ^ { \left( r \right) } ( { q } ) } S ( { q } , { g } ) ,\tag{29}
$$

returning all eligible entries if fewer than $K$ are available. Only the selected sample identifiers and their similarity scores are communicated for final aggregation. The global retrieved set is

$$
\begin{array} { r } { \mathcal { R } _ { K } ( q ) = \mathrm { T o p K } _ { g \in \bigcup _ { r = 1 } ^ { R } \mathcal C _ { K } ^ { ( r ) } ( q ) } S ( q , g ) . } \end{array}\tag{30}
$$

This aggregation preserves exact exhaustive retrieval. Under a consistent ordering for tied scores, any candidate outside its shard’s local top- $. K$ is preceded by at least $K$ eligible candidates from that shard and therefore cannot belong to the global top-K. Consequently, selecting from the union of local candidates yields the same result as selecting directly from the full eligible gallery. Final aggregation requires at most $R K$ candidates per query, without gathering the gallery features or the full similarity matrix across workers.

Communication and CPU fallback. In our implementation, variable-length query payloads and retrieved candidate metadata are communicated through a CPU-side Gloo process group. This avoids the additional GPU-memory allocation associated with object collectives over NCCL. If a query–gallery block still exceeds the available GPU memory, the implementation supports a CPU fallback that evaluates the same similarity function $S ( \cdot , \cdot )$ . The fallback changes the execution device while preserving the scoring definition, eligibility rules, and exhaustive retrieval procedure.

## C MORE IMPLEMENTATION DETAILS

To enable a thorough understanding of SignRAG, this section presents more implementation details.

Table 4: Implementation details of the four training stages. LR denotes learning rate, FFN denotes feed-forward network, and RFT denotes reinforcement fine-tuning.
<table><tr><td>Stage</td><td>Optimization</td><td>Model Configuration</td><td>Data / Training Details</td></tr><tr><td>Pretraining I Sign-Text Pretraining</td><td>Epochs: 200 Optimizer: SGD LR: 0.01 Weight decay: 0.001 Scheduler: cosine Warmup: 20 epochs Batch size: 8</td><td>Text encoder/decoder: first three layers of mBART Skeleton encoder: three-layer Transformer Attention heads: 8 Hidden size: 512 FFN dimension: 2048 Contrastive dimension: 1024</td><td>Training corpus: CSL-News Vocabulary size: 11,540</td></tr><tr><td>Pretraining II Large-Scale Sign MLLM Pretraining</td><td>Epochs: 50 Optimizer: AdamW LR: 1 × 10 -4 Batch size: 8 Objective: cross-entropy</td><td>Base LLM: Qwen3-8B Projector: MLP with GELU Projection: 512 → 4096 LoRA targets: q_proj, v_proj LoRA rank: 16 Scaling factor: 32 Dropout: 0.3</td><td>Training corpora: CSL-News / YouTube-ASL The sign encoder, multimodal projector, and LLM are jointly pretrained.</td></tr><tr><td>SFT Retrieval- Augmented SFT</td><td>Epochs: 10 Optimizer: AdamW LR: 1 × 10−4 Batch size: 4 Objective: cross-entropy</td><td>Stage-II LoRA weights are merged into the base LLM, followed by a new set of LoRA modules. LoRA targets: q_proj, v_proj LoRA rank: 16 Scaling factor: 32 Dropout: 0.3</td><td>Training data: downstream SLT datasets Retrieved texts and sign features are jointly used as multimodal input. Prompt: Fig. 4</td></tr><tr><td>RFT Retrieval Utility-Guided RFT</td><td>Epochs: 2 Optimizer: AdamW LR: 1 × 10 5 Batch size: 8 Responses per input: 8</td><td>SFT LoRA weights are merged with the pretrained LLM weights, followed by a new set of LoRA modules. LoRA targets: qproj, v_proj LoRA rank: 64 Scaling factor: 128 Dropout: 0.05</td><td>Optimization: GRPO Rewards: sentence-level quality reward and retrieval-utility reward The same prompt template as SFT is used.</td></tr></table>

## C.1 COMPUTING INFRASTRUCTURE

The entire experimental process was conducted on a computational infrastructure comprising an AMD EPYC 7502 32-Core Processor, 4 NVIDIA RTX 4090D GPUs with 48GB of VRAM, running the Ubuntu 20.04 operating system, with Python 3.10, CUDA 11.8, transformers 4.51.3 and PyTorch 2.1.2.

## C.2 KEYPOINT EXTRACTION.

Following the preprocessing protocol adopted in CoSign (Jiao et al., 2023), we extract frame-level human keypoints from sign-language videos using the off-the-shelf MMPose estimator. To focus on motion and appearance cues that are most relevant to sign understanding, we retain a compact set of keypoints covering the upper body, hands, mouth, and face. Specifically, the selected keypoints include 9 body joints, 21 joints for each hand, 8 mouth landmarks, and 18 facial landmarks, resulting in 77 keypoints per frame. Additional implementation details follow CoSign (Jiao et al., 2023).

## C.3 PRETRAINING(STAGE I): SIGN-TEXT PRE-TRAINING.

In the first stage, we train a compact sign-text model to align skeleton-based sign representations with textual semantics. Following GFSLT-VLP (Zhou et al., 2023), the text encoder and decoder are initialized by reusing the first three layers of mBART, while the skeleton encoder adopts the same three-layer Transformer structure. Each Transformer block uses 8 attention heads, a hidden size of 512, and a feed-forward dimension of 2048. The model is optimized for 200 epochs using SGD with a learning rate of 0.01 and a weight decay of 0.001. A cosine learning rate schedule is applied with 20 warmup epochs. The batch size is set to 8. For CSL-News, we follow prior work and reduce the vocabulary to a task-specific subset of 11,540 tokens. The contrastive feature dimension for both visual and textual representations is set to 1024.

## C.4 PRETRAINING(STAGE II): LARGE-SCALE SIGN MLLM PRETRAINING.

We use Qwen3-8B as the base language model. Following the design of LLaVA (Liu et al., 2023), an MLP projector with GELU activation maps the 512-dimensional visual features into the 4096- dimensional embedding space of the LLM. To enable parameter-efficient adaptation, we apply LoRA to the query and value projection matrices in the self-attention layers of the LLM. The LoRA rank, scaling parameter, and dropout rate are set to 16, 32, and 0.3, respectively. The model is trained for 50 epochs using the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 8. Cross-entropy loss is used as the training objective.

## C.5 SFT: RETRIEVAL-AUGMENTED SUPERVISED FINE-TUNING.

After large-scale sign MLLM pretraining, we merge the learned LoRA parameters into the base LLM to obtain the final pretrained LLM checkpoint. We then attach a new set of LoRA modules using the same LoRA configuration, including the rank, scaling factor, and dropout rate. The main difference in this stage is that the training data are switched from large-scale pretraining corpora, such as CSL-News and YouTube-ASL, to downstream SLT datasets. Retrieved texts are additionally incorporated and combined with the sign features to construct the multimodal input. The system prompt and instruction template used in this stage are shown in Fig. 4. During SFT, the model is trained for 10 epochs using AdamW with a learning rate of $1 \times 1 0 ^ { - \overline { { 4 } } }$ . The batch size is set to 4, and cross-entropy loss is used as the training objective.

![](images/3fe90894704dea895adc7fea83180f47a6b527c8c41827a21fa36349ecfb65c5.jpg)  
Figure 4: System prompt and instruction template. Here, with RAG denotes the setting where retrieved contexts are provided to the LLM, while without RAG denotes the setting where no retrieved contexts are provided.

Algorithm 1 Retrieval Context Construction   
Input: Query $q$ with $i d _ { q }$ and $t _ { q } ;$ gallery $\mathcal { G } = \{ ( g _ { i } , t _ { i } , i d _ { i } ) \} _ { i = 1 } ^ { N } ;$ similarity function $S ( \cdot , \cdot ) ;$ ; retrieval number K   
Output: Retrieved contexts $\mathcal { R } _ { K } ( q )$   
1: Compute visual similarities:   
$\hat { s } _ { i } = S ( q , g _ { i } ) , \qquad i = 1 , \ldots , N$   
2: Rank gallery samples in descending order:   
$\pi = \mathrm { A }$ rgsort (s<sub>i</sub>)   
3: Initialize $\mathcal { R } _ { K } ( \boldsymbol { q } ) \dot {  } \emptyset$   
4: for $i \in \pi$ do   
5: if $i d _ { i } = i d _ { q }$ then   
6: continue   
7: end if   
8: if TRAINRETRIEVAL $\wedge t _ { i } = t _ { q }$ then   
9: continue   
10: end if   
11: $\mathcal { R } _ { K } ( q )  \mathcal { R } _ { K } ( q ) \cup \{ t _ { i } \}$   
12: if $| \mathcal { R } _ { K } ( q ) | = K$ then   
13: break   
14: end if   
15: end for   
16: return $\mathcal { R } _ { K } ( q )$

## C.6 RFT: RETRIEVAL UTILITY-GUIDED REINFORCEMENT FINE-TUNING.

After supervised fine-tuning, we merge the learned LoRA parameters with our pretrained LLM weights to obtain the SignRAG-SFT checkpoint. We then attach a new set of LoRA modules for reinforcement fine-tuning. In this stage, the LoRA rank is increased to 64, the scaling factor is set to 128, and the dropout rate is reduced to 0.05. The model is trained for 2 epochs using AdamW with a learning rate of $1 \times 1 0 ^ { - 5 }$ and a batch size of 8. Following GRPO, we sample 8 candidate responses for each input and compute group-relative advantages based on both the sentence-level quality reward and the retrieval-utility reward. The same prompt template used in the SFT stage is adopted for RFT. For datasets containing very short sentences, we append sentence-final punctuation before reward computation to reduce the instability of BLEU-4-based training signals.

## C.7 IDENTICAL-TEXT FILTERING IN TRAINING RETRIEVAL.

We use CSL-Daily as an example to illustrate the retrieval procedure. The retrieval gallery G consists of all training samples, where each gallery entry is represented as $\mathcal { G } _ { i } = ( g _ { i } , t _ { i } , i d _ { i } )$ , containing the sign sample, its translation, and sample ID. Given a query $q ,$ all gallery samples are first ranked according to their visual similarity, followed by sample-level and text-level filtering. The complete procedure is summarized in Algorithm 1.

During training-set self-retrieval, identical-text filtering is introduced to prevent the target sentence of the query from being directly provided as retrieval context. Without this filtering, samples sharing the same target annotation may dominate the retrieved candidates, allowing the translation model to exploit a trivial shortcut by copying the retrieved target sentence. More importantly, this would introduce a mismatch between training and dev/test inference, since the exact target translation of a query is generally unavailable in the retrieval gallery at inference time.The effectiveness of identical-text filtering on CSL-Daily is evaluated in Table 5. Only 2 test samples share identica translations with the training set, suggesting that the improvements of SignRAG are not driven by a trivial retrieval shortcut that directly matches test queries to training samples with the same target translation.

Table 5: Effect of identical-text filtering during training-set self-retrieval on CSL-Daily.
<table><tr><td>Setting</td><td>BLEU-4</td></tr><tr><td>SignRAG-SFT w/o identical-text filtering</td><td>11.21</td></tr><tr><td>SignRAG-SFT w/ identical-text filtering</td><td>31.22</td></tr></table>

## D FURTHER ANALYSES

## D.1 ANALYSIS OF DIFFERENT PRETRAINING STRATEGIES.

We compare three pretraining strategies. Encoder-only pretraining initializes only the sign encoder from Stage I before downstream SFT. End-to-end pretraining directly pretrains a randomly initialized sign encoder jointly with the LLM. Hierarchical pretraining first learns a language-aware sign encoder and then jointly pretrains the sign encoder, multimodal projector, and LLM. As shown in Table 6, hierarchical pretraining substantially outperforms direct end-to-end pretraining on CSL-Daily, achieving BLEU-4 and ROUGE-L scores of 29.72 and 58.78, compared with 9.53 and 38.70, respectively. These results highlight the importance of learning semantically grounded sign representations before integrating them with an LLM.

Table 6: Comparison of pretraining strategies. Results on CSL-News and CSL-Daily correspond to pretraining and downstream SFT evaluation, respectively.
<table><tr><td>Pretraining Strategy</td><td>R-L↑</td><td>B@1↑</td><td>B@2↑</td><td>B@3↑</td><td>B@4↑</td></tr><tr><td>Pretraining evaluation on CSL-News Sign Encoder + Shallow Transformer</td><td>36.31</td><td>41.58</td><td>30.24</td><td>23.51</td><td>19.02</td></tr><tr><td colspan="6">Downstream SFT evaluation on CSL-Daily</td></tr><tr><td>Encoder-only Pretraining</td><td>22.51</td><td>18.22</td><td>8.76</td><td>5.02</td><td>3.20</td></tr><tr><td>End-to-End Pretraining</td><td>38.70</td><td>32.74</td><td>19.87</td><td>13.30</td><td>9.53</td></tr><tr><td></td><td>58.78</td><td>56.81</td><td>44.85</td><td></td><td></td></tr><tr><td>Hierarchical Pretraining (Ours)</td><td></td><td></td><td></td><td>36.18</td><td>29.72</td></tr></table>

## D.2 DIRECTLY FINE-TUNING A VIDEO LLM.

Constructing large-scale sign language datasets remains challenging due to the high cost of data collection and annotation, as well as the limited availability of annotators with professional sign language expertise. Consequently, a substantial cross-modal semantic gap persists between sign videos and natural language text. Video LLMs have demonstrated strong capabilities in temporal modeling and natural language generation. Adapting such models with a limited number of sign–text pairs therefore provides a potentially effective way to leverage their existing video understanding and language modeling capabilities for cross-modal alignment, while reducing the dependence on largescale sign language annotations and costly training. From this perspective, adapting general-purpose Video LLMs to sign language translation represents a promising research direction.

To ensure a fair comparison and isolate the effect of the model architecture as much as possible, we strictly follow the official training configuration of VideoLLaMA2, keeping all training settings and hyperparameters unchanged while replacing only the original video data with sign language data. As shown in Fig. 5, our preliminary experiments reveal a substantial performance gap between directly fine-tuning VideoLLaMA2 (7B) (Cheng et al., 2024) on sign–text pairs and our SFT model. We attribute this gap in part to the distinctive nature of sign language, whose semantics are substantially more structured and abstract than those of general video content. The visual encoder of a general-purpose Video LLM is not specifically designed to model the fine-grained visual patterns underlying sign language and therefore struggles to learn sufficiently discriminative and linguistically relevant sign representations. Moreover, compared with generic video understanding, SLT requires visual semantics at a much finer granularity, demanding accurate modeling of handshape, motion, spatial location, non-manual cues, and their temporal compositions. Capturing such subtle linguistic information remains challenging for general-purpose video encoders.

These observations also motivate a central goal of our work: to move beyond the direct adaptation of existing general-purpose Video LLMs and toward a multimodal foundation model with stronger sign representation learning and language modeling capabilities for SLT. Although the current analysis is based on a relatively limited experimental setting, further exploring Video LLMs with stronger visual backbones, larger model scales, and more extensive sign-language-oriented pre-training remains an important direction for future work.

![](images/75a95ebadbfc501abb5d641ee6148f895a46c99306909ec7d75e129232b1b53e.jpg)  
Figure 5: Performance comparison between directly fine-tuning VideoLLaMA2 and the SignRAG-SFT model.

## D.3 WHY DOES HIERARCHICAL PRETRAINING BENEFIT SIGN MLLMS?

We further analyze why hierarchical pretraining is particularly important when replacing conventional pretrained language models with stronger LLMs. Empirically, we observe an interesting phe nomenon: directly coupling a sign encoder with a powerful LLM can perform substantially worse than using a T5-based architecture, whereas pretraining the sign encoder before multimodal joint training significantly improves performance.

We attribute this phenomenon to Cross-Modal Optimization Imbalance (Wang et al., 2020; Huang et al., 2022; Peng et al., 2022). At the beginning of end-to-end multimodal training, the sign encoder is either randomly initialized or only weakly aligned with linguistic semantics, while the LLM already possesses strong language modeling capability and mature linguistic priors. Therefore, the sign representation

$$
Z _ { s } = E _ { \mathrm { s i g n } } ( X )\tag{31}
$$

may still mainly encode low-level motion patterns, whereas the LLM can already partially fit the target sentence through its pretrained language knowledge. This imbalance can lead to

$$
\underbrace { \mathrm { \backslash { W e a k ~ S i g n ~ F e a t u r e s } } } _ { \mathrm { \ i s u a l \ p a t h w a y } } + \underbrace { \mathrm { \backslash { S t r o n g ~ L a n g u a g e ~ P r i o r } } } _ { \mathrm { \normalfont { L L M } } }  \mathrm { \normalfont { L a n g u a g e - D o m i n a t e d ~ O p t i m i z a t i o n } } .\tag{32}
$$

As a result, the model may reduce the generation loss without sufficiently improving the semantic quality of the sign representation.

There are two main reasons for this behavior. First, the sign encoder only receives indirect supervision from the final generation objective:

$$
{ \mathcal { L } } _ { \mathrm { g e n } } \to { \mathrm { L L M } } \to { \mathrm { P r o j e c t o r } } \to { \mathrm { S i g n ~ E n c o d e r } } .\tag{33}
$$

Hence, the gradient signal reaching the sign encoder is less direct than explicit sign–text semantic supervision. Second, the multimodal projector may partially compensate for immature sign representations by mapping them into the LLM latent space, rather than forcing the sign encoder itself to develop linguistically meaningful features. Consequently, end-to-end optimization may successfully reduce the generation loss while still producing weak visual grounding. In other words,

$$
\mathrm { L o w e r G e n e r a t i o n L o s s } \not = \mathrm { B e t t e r S i g n R e p r e s e n t a t i o n } .\tag{34}
$$

This issue is less pronounced for T5-based SLT models. Compared with modern LLMs, T5-style language models generally provide weaker linguistic priors and therefore cannot rely as heavily on language-side shortcuts. To minimize the translation objective, they must depend more strongly on visual evidence from the sign encoder, resulting in tighter visual-to-language coupling. This explains why a T5-based model can remain relatively stable even when the sign encoder and language model are trained jointly.

In contrast, stronger LLMs require more semantically mature modality representations before multimodal optimization. Our hierarchical pretraining explicitly addresses this requirement. In the first stage, we perform sign–text alignment:

$$
E _ { \mathrm { s i g n } } \xrightarrow [ ] { \mathrm { S i g n - T e x t } \mathrm { A l i g n m e n t } } Z _ { s } ^ { \mathrm { s e m } } ,\tag{35}
$$

where $Z _ { s } ^ { \mathrm { s e m } }$ is encouraged to encode linguistically structured sign semantics rather than only motion patterns. In the second stage, the aligned sign encoder is integrated with the multimodal projector and LLM:

$$
Z _ { s } ^ { \mathrm { s e m } }  \mathrm { P r o j e c t o r }  \mathrm { L L M } .\tag{36}
$$

This decomposition transforms a difficult joint optimization problem into two progressively learned objectives:

$$
| \ { \mathrm { S i g n ~ R e p r e s e n t a t i o n ~ L e a r n i n g } } \to { \mathrm { C r o s s } } { \mathrm { - M o d a l ~ G e n e r a t i o n } } \ |\tag{37}
$$

The first stage focuses on what the sign means, while the second stage focuses on how to express the acquired sign semantics in natural language.

Overall, these results suggest that the advantage of a stronger LLM can only be fully realized when it is paired with a sufficiently strong modality interface. More generally, effective Sign MLLM pretraining depends not only on language-model capacity, but also on the quality of modality representations, cross-modal alignment, and optimization order:

MLLM Performance ∝ Representation Quality + Alignment Quality + Optimization Strategy.

(38)

This observation motivates our hierarchical design and explains why directly scaling the language backbone does not necessarily lead to better sign language translation.

## D.4 DATASET STATISTICS.

Table 7: Statistics of the pretraining and fine-tuning datasets. For YouTube-ASL, we report both the original dataset size and the number of samples available to us.
<table><tr><td>Stage</td><td>Dataset</td><td>Sign language</td><td># Samples</td></tr><tr><td rowspan="3">Pretraining</td><td>CSL-News (Li et al., 2025b)</td><td>Chinese</td><td>722,715</td></tr><tr><td>YouTube-ASL (Uthus et al., 2023) (original)</td><td>American</td><td>530,161</td></tr><tr><td>YouTube-ASL (available)</td><td>American</td><td>434,953</td></tr><tr><td rowspan="4">Fine-tuning</td><td>How2Sign (Duarte et al., 2021)</td><td>American</td><td>35,172</td></tr><tr><td>OpenASL (Shi et al., 2022)</td><td>American</td><td>98,417</td></tr><tr><td>CSL-Daily (Zhou et al., 2021)</td><td>Chinese</td><td>20,654</td></tr><tr><td>PHOENIX-2014T(Camgoz et al., 2018)</td><td>German</td><td>8,257</td></tr></table>

Table 7 summarizes the datasets used for pretraining and fine-tuning, all of which consist of sentence-level sign–text pairs. For YouTube-ASL, we successfully obtained 434,953 of the 530,161 reported samples, corresponding to 82.04% of the original dataset. Some of the provided YouTube links have become invalid over time, preventing access to the corresponding source videos. We therefore use the available subset for pretraining.

## D.5 METRIC OVERFITTING ANALYSIS.

Since RUG-RFT optimizes sentence-level BLEU and ROUGE rewards, we further evaluate SignRAG with chrF (Popovic, 2015) and BERTSim, a BERT-based´ semantic similarity score computed with bert-basechinese.<sup>1</sup> As shown in Table 8, SignRAG-RFT improves over SignRAG-SFT on ROUGE-L, BLEU-4, chrF, and BERTSim. These complementary metrics suggest that the gains are not limited to direct BLEU/ROUGEoriented optimization.

Table 8: Additional automatic metrics on CSL-Daily for checking metric overfitting. SFT and RFT denote SignRAG-SFT and SignRAG-RFT; R-L, B@4, and BSim denote ROUGE-L, BLEU-4, and BERTSim.
<table><tr><td colspan="4">R-L↑ B@4↑ chrF↑ BSim↑</td></tr><tr><td>SFT</td><td>60.20</td><td>31.22</td><td>26.93 82.10</td></tr><tr><td>RFT</td><td>62.34</td><td>32.91 28.42</td><td>82.96</td></tr></table>

## D.6 TRANSFERABILITY ACROSS CHINESE SLT BENCHMARKS.

To further evaluate whether our pretraining strategy generalizes beyond a specific Chinese SLT dataset, we transfer the pretrained model to CE-CSL (Zhu et al., 2024), a continuous sign language benchmark collected under complex environments. As shown in Table 9, our Pretrain→SFT consistently outperforms Uni-Sign and Geo-Sign across all metrics. In particular, it achieves 24.44 BLEU-4 and 54.55 ROUGE-L, surpassing the strongest baselines by 2.37 and 2.54 points, respectively. These results demonstrate that the learned representations can effectively transfer across different Chinese SLT benchmarks, supporting the generalizability of our pretraining strategy.

Table 9: Comparison on the CE-CSL benchmark. We further evaluate the adaptability of different pretrained SLT models on CE-CSL. Uni-Sign and Geo-Sign are included as representative baselines with publicly available pretrained weights. The results demonstrate the capability of our Pretrain→SFT to transfer and adapt to different downstream SLT benchmarks.
<table><tr><td>Method</td><td>B@1↑</td><td>B@2↑</td><td>B@3↑</td><td>B@4↑</td><td>R-L↑</td></tr><tr><td>Uni-Sign</td><td>51.23</td><td>37.14</td><td>27.89</td><td>21.66</td><td>52.01</td></tr><tr><td>Geo-Sign</td><td>51.81</td><td>37.22</td><td>28.31</td><td>22.07</td><td>50.24</td></tr><tr><td>Pretrain→SFT (Ours)</td><td>53.18</td><td>39.14</td><td>30.27</td><td>24.44</td><td>54.55</td></tr></table>

## D.7 FAILURE CASE (HARMFUL RELIANCE ON RETRIEVED CONTEXTS).

Table 10 presents a representative failure case illustrating harmful reliance on retrieved contexts. Although the Pretrain→SFT variant, which is fine-tuned using only visual input without retrieval, correctly translates the input as The weather is nice today, SignRAG-SFT is misled by the incorrect retrieved context and generates The supermarket is holding an event. After RFT, SignRAG-RFT recovers the correct translation, indicating that retrieval utility-guided reinforcement fine-tuning can reduce harmful reliance on misleading retrieved information.

Table 10: Qualitative example illustrating harmful reliance on retrieved contexts and the effect of RFT. Red text highlights misleading or incorrectly generated content.
<table><tr><td>Source</td><td>Text</td></tr><tr><td>Ground-Truth</td><td>The weather is nice today.</td></tr><tr><td>Retrieved Text</td><td>The supermarket is closed today.</td></tr><tr><td>Pretrain→SFT</td><td>The weather is nice today.</td></tr><tr><td>SignRAG-SFT</td><td>The supermarket is holding an event.</td></tr><tr><td>SignRAG-RFT</td><td>The weather is nice today.</td></tr></table>

## E EFFICIENCY, COMPUTATIONAL COST, AND RETRIEVAL SCALABILITY

In this section, we provide a systematic analysis of the computational overhead, retrieval scalability, memory consumption, and training cost of SignRAG. Unless otherwise specified, all efficiency experiments are conducted using Qwen3-8B on four NVIDIA RTX 4090D GPUs. For a fair comparison, the Non-RAG baseline (Pretrain→SFT) and SignRAG-SFT are evaluated using the same hardware, batch size, numerical precision, and decoding strategy. Since inference latency reported in prior work is often measured under different hardware and implementation settings, we focus on the incremental cost of SignRAG relative to this hardware-controlled baseline.

## E.1 ONLINE SLT INFERENCE EFFICIENCY

A key property of SignRAG is that retrieval is performed offline. Specifically, gallery search is conducted once before training or evaluation, and the retrieved results are stored in the data manifests. Therefore, no full-gallery search is performed during the deployed model.generate path.

The additional online inference cost mainly arises from the increased input length introduced by the retrieved textual contexts.

Table 11 reports the online generation efficiency on CSL-Daily. Compared with the Non-RAG baseline (Pretrain→SFT), SignRAG-SFT increases the average latency from 42.69 ms/sample to 61.98 ms/sample, corresponding to an additional 19.29 ms/sample. The throughput decreases from 23.42 to 16.13 samples/s, while the peak GPU memory increases from 23.28 to 29.03 GiB/GPU. This corresponds to a throughput reduction of approximately 31.13% and an additional memory consumption of 5.75 GiB/GPU. Despite this moderate overhead, SignRAG-SFT improves BLEU-4 on CSL-Daily from 29.72 to 31.22.

Table 11: Online inference efficiency on CSL-Daily. Both methods use Qwen3-8B and four NVIDIA RTX 4090D GPUs under the same inference configuration. Retrieval itself is performed offline and is therefore excluded from the deployed generation path.
<table><tr><td>Method</td><td>Latency (ms/sample)</td><td>Throughput (samples/s)</td><td>Peak GPU Mem. (GiB/GPU)</td></tr><tr><td>Non-RAG(Pretrain→SFT)</td><td>42.69</td><td>23.42</td><td>23.28</td></tr><tr><td>SignRAG-SFT</td><td>61.98</td><td>16.13</td><td>29.03</td></tr></table>

## E.2 OFFLINE RETRIEVAL EFFICIENCY ON CSL-DAILY

We next evaluate the efficiency of the offline retrieval stage. The retrieval gallery consists of the target-domain training samples, while each evaluation sample serves as a query. Table 12 reports the retrieval latency and throughput when progressively increasing the gallery size while keeping the same set of 256 queries.

As the gallery grows from 1K to 18.4K samples, the average retrieval latency increases from 3.49 to 23.88 ms/query. At the same time, retrieval throughput decreases from 286.67 to 42.02 queries/s. These results show that the retrieval cost increases with gallery size, but remains practical at the scale of CSL-Daily.

Table 12: Offline retrieval efficiency on CSL-Daily with varying gallery sizes. All settings use the same 256 queries.
<table><tr><td>Gallery Size</td><td>Latency (ms/query)</td><td>Throughput (queries/s)</td></tr><tr><td>1K</td><td>3.49</td><td>286.67</td></tr><tr><td>5K</td><td>7.10</td><td>140.80</td></tr><tr><td>10K</td><td>14.65</td><td>69.32</td></tr><tr><td>18.4K</td><td>23.88</td><td>42.02</td></tr></table>

For the complete CSL-Daily test set, retrieving 1,176 queries from the full gallery of 18,401 training samples requires 64.70 s in total, corresponding to 55.01 ms/query. The difference between this endto-end retrieval latency and the 256-query micro-benchmark in Table 12 is caused by the complete retrieval pipeline, which additionally includes data loading, batching, result aggregation, and manifest construction. The peak active retrieval memory is 1.17 GiB per GPU, while the corresponding CPU memory usage is 10.42 GiB per rank.

Combining the one-time offline retrieval cost with SignRAG generation, the complete retrieval-and generation evaluation on CSL-Daily requires 137.59 s for 1,176 test samples. This corresponds to approximately 117.00 ms/sample and 0.153 GPU-hours when accounting for all four GPUs. Since retrieval results can be pre-computed and reused, the deployed online inference latency remains the 61.98 ms/sample reported in Table 11.

## E.3 RETRIEVAL SCALABILITY ON OPENASL

To further evaluate scalability beyond CSL-Daily, we conduct the same retrieval analysis on OpenASL, whose training gallery contains approximately 96K samples. Table 13 reports the retrieval efficiency under progressively increasing gallery sizes.

With a gallery of 95,888 samples, the retrieval latency reaches 705.85 ms/query for the 256-query benchmark, corresponding to a throughput of 1.42 queries/s. For the complete OpenASL test set, retrieving 971 queries from the full 95,888-sample gallery requires 715.90 s in total, corresponding to 737.29 ms/query. These results demonstrate that token-level retrieval remains feasible at the approximately 100K-gallery scale, although its cost grows substantially with gallery size.

Table 13: Offline retrieval efficiency at different gallery sizes on OpenASL. The same 256 queries are used for all gallery-size experiments.
<table><tr><td>Gallery Size</td><td>Latency (ms/query)</td><td>Throughput (queries/s)</td></tr><tr><td>1,000</td><td>11.20</td><td>89.38</td></tr><tr><td>5,000</td><td>39.98</td><td>25.02</td></tr><tr><td>10,000</td><td>77.28</td><td>12.94</td></tr><tr><td>95,888</td><td>705.85</td><td>1.42</td></tr></table>

## E.4 TRAINING COST

Table 14 summarizes the training cost of the complete SignRAG pipeline, with CSL-News serving as the pre-training dataset and CSL-Daily as the downstream fine-tuning dataset. The two-stage pre-training procedure requires 106.67 GPU-hours for Stage I and 277.50 GPU-hours for Stage II. Downstream SignRAG-SFT requires 24.04 GPU-hours, while SignRAG-RFT requires an additional 5.78 GPU-hours. Overall, the complete training pipeline requires 413.99 GPU-hours.

Table 14: Training cost of SignRAG.
<table><tr><td>Component</td><td>Cost</td></tr><tr><td>Pre-training Stage I Pre-training Stage II</td><td>106.67 GPU-hours</td></tr><tr><td>SignRAG-SFT</td><td>277.50 GPU-hours 24.04 GPU-hours</td></tr><tr><td>SignRAG-RFT</td><td>5.78 GPU-hours</td></tr><tr><td>Total training compute</td><td>413.99 GPU-hours</td></tr></table>

## E.5 DISCUSSION ON MILLION-SCALE RETRIEVAL

The above experiments demonstrate the practical feasibility of SignRAG on CSL-Daily and OpenASL, with gallery sizes of approximately 18K and 96K, respectively. However, token-level matching becomes increasingly expensive as the gallery grows. For substantially larger corpora such as BOBSL, which contains approximately 1.2 million sentences, we do not currently report direct retrieval results. At present, BOBSL is not available to us, and a reliable evaluation would additionally require data preparation, model pretraining, feature extraction, index construction, and retrieval-quality validation. Therefore, without direct experimental evidence, we avoid extrapolating the current measurements to the million-scale setting.

A potential direction for improving scalability is coarse-to-fine retrieval. For example, global video representations or domain information could first be used to shortlist a candidate pool through approximate nearest-neighbor search, after which the proposed fine-grained token-level matching would be applied only to the shortlisted candidates. Gallery sharding and batched retrieval could further reduce wall-clock latency.

However, coarse-to-fine retrieval introduces a non-trivial accuracy–efficiency trade-off. A coarse retrieval stage based on a single global representation may discard samples that would otherwise receive high scores under full token-level matching. Therefore, we do not claim that coarse-to-fine retrieval can fully preserve the accuracy of exhaustive SignRAG retrieval. A more comprehensive evaluation at the million-scale should jointly consider (i) retrieval latency, (ii) Top-K candidate preservation, (iii) retrieval quality, and (iv) downstream SLT performance under different candidatepool sizes.

Finally, we emphasize that the primary contribution of SignRAG is not to claim unrestricted scalability to arbitrary million-scale open-domain corpora, but rather to extend retrieval-augmented SLT from settings that rely on gloss annotations to a gloss-free formulation and to demonstrate its effectiveness on multiple downstream benchmarks, including relatively large-scale datasets such as How2Sign and OpenASL. The OpenASL results in Table 13 directly characterize the computational behavior of the current implementation at the largest scale evaluated in this work.

For very large datasets, the limited vocabulary of sign language may also reduce the need for global Top-K retrieval. A fixed subset of the gallery may already provide useful retrieval candidates. However, this hypothesis still requires empirical validation. Million-scale retrieval therefore remains a limitation of the current work and an important direction for future study.

## F MORE CASES

Due to space limitations in the main paper, we provide additional qualitative examples in Table 15. These cases offer a more comprehensive view of the translation performance of our method across diverse examples.

<table><tr><td>Source</td><td>Sentence</td></tr><tr><td>Reference</td><td>我想要找借口。 (I want to find an excuse.)</td></tr><tr><td>Retrieved</td><td>[1] 他随口编了一个迟到的借口。(He casually made up an excuse for being late.) [2]为自已找借口的人，永远不会进步。(People who make excuses for themselves will never improve.)</td></tr><tr><td>Geo-Sign</td><td>[3] 我不会用出差作为任何事情的借口。(I would not use a business trip as an excuse for anything.) 你找我有什么事吗？</td></tr><tr><td>SignRAG</td><td>(Why are you looking for me?) 找借口。 (Find an excuse.)</td></tr><tr><td>Reference</td><td>封面设计是一种艺术设计。</td></tr><tr><td>Retrieved</td><td>(Cover design is a form of artistic design.) [1] 一个好的封面设计会让你脱颖而出。(A good cover design will make you stand out.)</td></tr><tr><td></td><td>[2] 我姐姐是从事封面设计工作。(My sister works in cover design.) [3] 我姐姐是从事封面设计工作。(My sister works in cover design.) 平面设计是一种艺术与技术的结合。</td></tr><tr><td>Geo-Sign SignRAG</td><td>(Graphic design is a combination of art and technology.) 封面设计是一种艺术设计。</td></tr><tr><td></td><td>(Cover design is a form of artistic design.) 这个视频拍下了你作弊的行为。</td></tr><tr><td>Reference Retrieved</td><td>(This video recorded you cheating.) [1] 他对自已作弊的行为感到非常羞愧。(He felt very ashamed of his cheating behavior.)</td></tr><tr><td></td><td>[2] 他对自已作弊的行为感到非常羞愧。(He felt very ashamed of his cheating behavior.) [3] 你的行动还不够迅速。(Your actions are not quick enough.) 他以借书为名，行偷盗之实。</td></tr><tr><td>Geo-Sign</td><td>(He used borrowing books as a pretext for stealing.)</td></tr><tr><td></td><td></td></tr><tr><td>Reference</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>今天水果很新鲜，买一个西瓜回家。</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>SignRAG</td><td>你作弊的行为是可耻的。</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>(Your cheating behavior is shameful.)</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Retrieved</td><td>(The fruit is very fresh today; buy a watermelon to take home.)</td></tr><tr><td></td><td>[1] 天气热，买点水果回家给爸爸吃。(The weather is hot; buy some fruit to take home for Dad.)</td></tr><tr><td></td><td>[2] 这些水果真新鲜，我们买西瓜还是苹果? (These fruits are really fresh; should we buy a watermelon or apples?)</td></tr><tr><td></td><td>[3] 这些水果真新鲜，我们买西瓜还是苹果? (These fruits are really fresh; should we buy a watermelon or apples?)</td></tr><tr><td>Geo-Sign</td><td>水果今天很新鲜，买一串香蕉回家。</td></tr><tr><td></td><td>(The fruit is very fresh today; buy a bunch of bananas to take home.)</td></tr><tr><td>SignRAG</td><td></td></tr><tr><td></td><td>水果今天很新鲜，买一个西瓜回家。</td></tr><tr><td></td><td></td></tr><tr><td></td><td>(The fruit is very fresh today; buy a watermelon to take home.)</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

Table 15: Additional qualitative comparisons on CSL-Daily. Each example presents the reference translation, top-3 retrieved texts, and outputs of Geo-Sign and SignRAG. English translations are shown in parentheses below the original Chinese sentences.