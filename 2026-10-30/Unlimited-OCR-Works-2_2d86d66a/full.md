# Unlimited OCR Works 2

Pioneering the OCR-3.0 Era: From Long- to Unlimited-Horizon Parsing

Baidu ERNIE Team

## Abstract

Unlimited OCR ushers the era of long-horizon parsing, substantially mitigating the growth of working memory as document lengthens. However, as document length scales to hundreds of pages, the visual context grows substantially, sharply increasing the computational cost of each decoding step and limiting the scalability of ultra-long document parsing. Inspired by human book transcription via localized focus without re-scanning preceding pages, we introduce Unlimited OCR 2, a document parsing model that looks only at what it transcribes. Building on Unlimited OCR, we upgrade all the decoder attention modules to Sparse Reference Sliding Window Attention (SR-SWA), bounding active reference tokens to a fixed capacity and thus sustaining constant computational costs across arbitrarily long documents. By combining SR-SWA with Unlimited-Horizon Decoding Paradigm, Unlimited OCR 2 enables document parsing over virtually unlimited lengths without prohibitive memory overhead. At the hundredpage scale, Unlimited OCR 2 achieves 7.0× and 5.6× per-step speedups over Full Attention and R-SWA, respectively. Furthermore, it delivers state-of-the-art end-to-end performance on both MPDocBench and MarathonDocBench, our newly proposed benchmark targeting hundredpage document parsing. Ultimately, the capacity for uninterrupted, one-shot document parsing signifies a fundamental paradigm shift into the OCR 3.0 era.

![](images/620db77daffb822aae75b49aa02f887db21a70d543991148554d6cb6c35e50f1.jpg)  
Figure 1 | R-SWA vs SR-SWA. R-SWA computes attention over all visual tokens across pages, inducing extreme visual redundancy; SR-SWA sparsely attends to relevant visual regions, aligning machine decoding with human reading intuition.

## Contents

1 Introduction 3   
2 Related Works 4   
2.1 VLM-based Document Parsing 4   
2.2 Efficient Long-Context Modeling 4   
3 Methodology 4   
3.1 Problem Setup 4   
3.1.1 Unlimited-horizon Parsing 4   
3.1.2 MarathonDocBench 5   
3.2 Architecture 5   
3.3 Sparse Reference Sliding Window Attention . 6   
3.3.1 Page Gate 6   
3.3.2 Token Sieve 6   
3.4 Unlimited-Horizon Decoding Paradigm 7   
4 Experimental Settings 8   
4.1 Data Engine 8   
4.2 Implementation Details . 8   
5 Evaluation 9   
5.1 Benchmarks and Metrics . 9   
5.2 Main Results 10   
5.3 Scaling with Document Length 11   
5.3.1 Feasibility of Unlimited-horizon Parsing 11   
5.3.2 Length-Invariant Accuracy 11   
5.3.3 Constant Per-Step Decoding Latency 12   
6 Limitation and Future Work 12   
7 Conclusion 12   
8 Author List 13

## 1. Introduction

"Pixels are my body, and tokens are my blood. I have parsed over a thousand pages. Unknown to KV cache. Nor known to context limits. So, as I pray: Unlimited OCR Works."

Snarcy on huggingface (Adapted from UBW)

Document parsing has undergone a major paradigm shift, transitioning from traditional multi-stage pipelines (OCR-1.0) [5, 6, 8, 20, 22, 26, 40] with handcrafted detection and recognition modules to end-to-end Vision-Language Models (OCR-2.0) [7, 18, 19, 21, 31]. By formulating document understanding as a unified sequence-to-sequence transcription task, OCR-2.0 frameworks effectively eliminate cumulative pipeline errors. Our previous work, Unlimited OCR [37], pushed OCR-2.0 boundaries to tens of pages by using Reference Sliding Window Attention (R-SWA) to maintain a constant KV cache. Nevertheless, it still computes attention over all visual tokens across pages during decoding. As document length scales, this persistent attention to non-informative visual tokens severely escalates visual redundancy, compounding the difficulty of end-to-end inference. This visual attention bottleneck naturally prompts a fundamental question: how can we transition from length-limited parsing to an unlimited one-shot paradigm and genuinely inaugurate the OCR-3.0 era?

To answer this question, human transcription of lengthy books provides a clear physical intuition for parsing scaling: as we copy text page by page, we maintain localized visual focus, casting only momentary glances back when cross-page references are needed, rather than keeping the entire volume active in mind. This cognitive mechanism reveals that visual dependencies in multi-page documents are inherently page-sparse yet intra-page concentrated. However, standard dense global attention completely overlooks this sparse dynamic, incurring redundant attention interactions across all pages. This not only incurs prohibitive computational overhead, but also leads to severe memory bottlenecks as document length scales. To break through these efficiency and memory bottlenecks, we align machine perception with human reading patterns and present Unlimited OCR 2, pioneering an unlimited-horizon parsing framework for the OCR-3.0 era. Our main contributions are as follows:

• We introduce Sparse Reference Sliding Window Attention (SR-SWA), a hierarchical sparse attention mechanism designed for unlimited visual context. Specifically, Page Gate first narrows each query's visual reference scope from the entire document to relevant pages while robustly modeling multi-page positional relationships. Token Sieve selectively incorporates top-k remote token buckets to capture long-range visual dependencies, thereby drastically reducing attention overhead while preserving comprehensive context.

• We introduce MarathonDocBench, a benchmark for hundred-page document parsing. It substantially extends existing evaluations in document length and cross-page structural complexity, enabling a comprehensive assessment of long-document parsing performance in the hundred-page regime.

• Comprehensive evaluations in long-horizon settings demonstrate that our approach maintains constant per-step decoding latency regardless of document length while achieving state-of-the-art end-to-end performance on multi-page benchmarks. Building on these advances, we establish an efficient unlimited-horizon parsing paradigm by integrating SR-SWA with the Unlimited-Horizon Decoding Paradigm, enabling document parsing over virtually unlimited lengths.

In summary, SR-SWA fundamentally breaks the computational and memory bottlenecks of ultra-long visual context. Operating at constant per-step decoding latency with length-invariant accuracy across hundreds of pages, Unlimited OCR 2 unlocks scalable long-document parsing and paves the way for the OCR-3.0 era of unlimited-horizon understanding.

## 2. Related Works

## 2.1. VLM-based Document Parsing

Document parsing has increasingly been reshaped by advances in vision-language models, evolving from pipeline-based OCR-1.0 systems [5, 6, 8, 20, 22, 26, 40], which combine specialized page analysis modules with VLM-based decoding, to unified VLM-based OCR-2.0 systems that directly generate structured representations from document images. OCR-2.0 encompasses both single-page parsers [7, 18, 19, 21, 30, 31] and Unlimited OCR [37], which extends end-to-end parsing to multi-page documents by jointly modeling their visual content within a shared context. However, although Unlimited OCR improves long-document efficiency with R-SWA, it still densely attends to all remote visual tokens, causing the visual context accessible to each decoding query to expand with document length and creating a fundamental scalability bottleneck at the hundred-page scale. Unlimited OCR 2 overcomes this limitation by explicitly bounding remote visual access, enabling attention-efficient end-to-end parsing of hundred-page documents and advancing document parsing into the OCR-3.0 era.

## 2.2. Efficient Long-Context Modeling

Efficient long-context modeling has been explored through both attention replacement and sparse attention. Linear or recurrent alternatives [11, 15, 17, 35, 36] reduce long-context cost by summarizing historical information into compact states. Document parsing, however, requires long-range memory to preserve fine-grained visual evidence for accurate recognition and crosspage retrieval, which may not be fully retained within such compact representations.

Sparse attention retains direct access to selected historical tokens or KV blocks while avoiding dense all-to-all computation. Early methods use predefined patterns such as local windows and global tokens [2, 38], whereas recent approaches adopt query-dependent retrieval. Hard-selection methods, such as Compressed Sparse Attention(CSA) [33] and MiniMax Sparse Attention(MSA) [16], rely on discrete top-k routing and typically require auxiliary supervision or warmup. In contrast, DashAttention [13] and HiLS [12] incorporate differentiable or forward-integrated routing scores, enabling more direct end-to-end optimization.

## 3. Methodology

## 3.1. Problem Setup

## 3.1.1. Unlimited-horizon Parsing

Unlimited-horizon parsing aims to enable end-to-end processing of documents of arbitrary lengths, from single-page forms to massive multi-page documents, with bounded computational and memory complexities. Formally, multi-page document parsing generates a transcription $y ~ = ~ ( y _ { 1 } , \ldots , y _ { L } )$ from an ordered document $\mathcal { D } \ = \ ( P _ { 1 } , \ldots , P _ { N } )$ with visual tokens $\mathcal { N } _ { 1 : N }$ . At decoding step t, the model operates on:

$$
X _ { t } = \operatorname { C o n c a t } \left( \mathscr { V } _ { 1 } , \dots , \mathscr { V } _ { N } , y _ { < t } \right) .\tag{1}
$$

While Unlimited OCR constrained text history to a fixed sliding window via R-SWA, it still requires attending over all preceding visual tokens $\mathcal { N } _ { 1 : N }$ . In the hundred-page regime, this pagedependent visual expansion becomes the primary bottleneck. Decoupling visual context from page count N to maintain a bounded computational and memory footprint directly motivates our proposed Sparse Reference Sliding Window Attention (SR-SWA).

![](images/29c3ebafc7c8c3bebbed2d41e292fb5fe34b36d18fb412ff84898ef215171ca2.jpg)  
Figure 2 | MarathonDocBench versus MPDocBench. MarathonDocBench extends document length from tens to hundreds of pages (a), with substantially richer cross-page structures as shown in (b) and (c).

## 3.1.2. MarathonDocBench

Existing multi-page document parsing benchmarks neither extend to the hundred-page scale nor sufficiently capture the rich cross-page structures that emerge in long documents. To address this gap, we introduce MarathonDocBench, the first benchmark specifically designed for hundred-page document parsing. It contains 100 documents spanning 103–300 pages with 205k annotated layout elements, extending the document-length range by roughly an order of magnitude beyond MPDocBench [41].

Beyond document length, MarathonDocBench contains substantially richer cross-page structures than MPDocBench, with cross-page text and tables appearing much more frequently and densely across documents, as summarized in Figure 2. This makes MarathonDocBench a substantially more challenging testbed for evaluating whether end-to-end parsers can preserve document-level structure while scaling from tens to hundreds of pages. To ensure reliable evaluation, structural annotations, including bounding boxes, block type, hierarchy, reading order, and cross-page continuations of text and tables, are initialized from a strong page-by-page parser and carefully refined by human annotators. Content transcriptions are initialized from the same parser outputs and verified through an iterative VLM-based refinement pipeline, with flagged disagreements adjudicated by human annotators.

## 3.2. Architecture

Unlimited OCR 2 inherits the fundamental backbone from Unlimited OCR, leveraging DeepSeek OCR [28] as its foundation. Specifically, it employs the DeepEncoder paired with a Mixture-of-Experts (MoE) decoder architecture featuring 3B total parameters and 500M activated parameters per token. Building upon this base, we replace the baseline R-SWA with our proposed SR-SWA as shown in Figure 3. While R-SWA bounds local interactions, its visual context scope still expands with growing document length, leading to noticeable latency accumulation over ultralong horizons. By introducing a selective remote visual routing mechanism, SR-SWA strictly bounds the active context scope, enabling Unlimited OCR 2 to maintain a constant KV cache footprint across both text and visual modalities during unlimited-horizon decoding. We detail the inner mechanics of SR-SWA in the following subsections.

![](images/f272814eae8fac6d0fadf752f122708f8175631c5c3f5970ffede1a774a14143.jpg)  
Figure 3 | Inspired by the process of humans copying books, we propose Unlimited OCR 2. SR-SWA combines Page Gate and Token Sieve to construct a fixed-length set of working reference visual tokens, substantially reducing computational and memory overhead while keeping both independent of document length throughout generation.

## 3.3. Sparse Reference Sliding Window Attention

Following standard block-sparse attention [13, 16, 32, 33], we partition Xt into fixed-size blocks $\{ \mathcal { B } _ { b } \} _ { b = 1 } ^ { B }$ and preserve a fixed local window L(q) for each query q. SR-SWA constructs S(q) hierarchically, using Page Gate for page-level organization and Token Sieve for fixed-budget selection.

## 3.3.1. Page Gate

To eliminate irrelevant visual tokens and concentrate focus on the current page, Page Gate restricts the visual attention scope for a query q associated with page g(q) by retaining only blocks from designated target pages:

$$
C _ { \mathrm { p a g e } } ( q ) = \left\{ b \mid \mathcal { B } _ { b } \cap \mathcal { V } _ { g ( q ) } \neq \emptyset \right\} .\tag{2}
$$

Since block boundaries are independent of page boundaries, a candidate block may straddle adjacent pages. Meanwhile, to prevent positional explosion as page count scales, Page Gate factorizes each token position into a bounded page-level coordinate (following an alternating 0–1 pattern) and a monotonic intra-page coordinate:

$$
\begin{array} { r } { \mathbf { p } _ { i , j } = \left[ \mathbf { p } _ { i } ^ { \mathrm { p a g e } } , \mathbf { p } _ { i , j } ^ { \mathrm { i n t r a } } \right] , } \end{array}\tag{3}
$$

ensuring both stable length extrapolation and efficient cross-page visual routing.

## 3.3.2. Token Sieve

While Page Gate restricts the routing space across pages, efficient fixed-budget routing requires a principled block selection strategy. Conventional sparse routers typically rely on dense-attention distillation, which lacks direct alignment with the language modeling objective. Inspired by recent end-to-end routing designs [12, 13], Token Sieve directly optimizes fixed-budget block selection under the language modeling objective.

Given the candidate set $C _ { \mathrm { p a g e } } ( q )$ from Page Gate, Token Sieve summarizes each candidate block b into a compact key representation $\bar { \mathbf { k } } _ { b }$ and computes its relevance to query q. The router then forms a sparse routing distribution and retains only the top-k blocks:

$$
\begin{array} { l } { { s _ { b } ( q ) = \displaystyle \frac { \tilde { \bf q } ^ { \top } \bar { \bf k } _ { b } } { \sqrt { d } } , \qquad \pi ( q ) = \alpha { \mathrm { - } } \mathrm { e n t m a x } \left( \{ s _ { b } ( q ) \} _ { b \in C _ { \mathrm { p a g e } } ( q ) } \right) , } } \\ { { S ( q ) = \mathrm { T o p K } \left( \pi ( q ) , k \right) . } } \end{array}\tag{4}
$$

Here, d denotes the hidden dimension. By selectively attending to a fixed number of top-k remote buckets, Token Sieve captures long-range visual dependencies with minimal computational overhead. To preserve relative preferences among selected blocks, the routing probabilities are converted into a mean-centered log-prior:

$$
\beta _ { b } ( q ) = \log \pi _ { b } ( q ) - \frac { 1 } { \vert S ( q ) \vert } \sum _ { b ^ { \prime } \in S ( q ) } \log \pi _ { b ^ { \prime } } ( q ) , \qquad b \in S ( q ) .\tag{5}
$$

With the selected remote blocks $S ( q )$ and the fixed local window $\mathcal { L } ( q )$ , each query attends to a length-agnostic retained token set defined as:

$$
\mathcal { R } ( q ) = \mathcal { L } ( q ) \cup \bigcup _ { b \in S ( q ) } \mathcal { B } _ { b } .\tag{6}
$$

The attention weights over $\mathcal { R } ( q )$ are computed as:

$$
\alpha _ { j } ( q ) = \mathrm { S o f t m a x } \left( \frac { q ^ { \top } k _ { j } } { \sqrt { d } } + \mathbb { I } [ b ( j ) \in S ( q ) ] \beta _ { b ( j ) } ( q ) \right) ,\tag{7}
$$

where the softmax is normalized over all tokens in $\mathcal { R } ( q ) , b ( j )$ denotes the block containing token $j ,$ and $k _ { j }$ denotes its key projection. Tokens within the local window use standard attention logits, while tokens from selected remote blocks additionally incorporate the corresponding routing bias. Since the routing bias directly participates in these logits, gradients from the parsing objective can propagate to the routing module, enabling end-to-end optimization without an auxiliary KL-divergence distillation loss.

Algorithm 1 Cross-Chunk KV Reuse for Unlimited-Horizon Decoding   
Require: Pages $P _ { 1 } , \dots , P _ { N } ;$ instruction q; chunk size $W ;$ tail length T   
Ensure: Transcription Y   
Notation: C is a KV cache; ⊕ concatenates KV in order;   
ENC(X | C) returns the KV of input X computed with C as context.   
1: Partition N pages into K overlapping chunks $C _ { 1 } , \ldots , C _ { K }$ with a 1-page boundary overlap   
K=1 indicates unsegmented   
2: C ← ENC(C1 ⊕ q) only the first chunk carries q   
3: y1 ← DECODE(C); Y ← y1   
4: for k = 2 to K do   
5: C ← Cverlap  ENC(Ck | Cielap img ) ⊕ C tx t   
tail   
6: yk ← DECODE(C); Y ← Y θ yk   
7: end for   
8: return Y

## 3.4. Unlimited-Horizon Decoding Paradigm

While SR-SWA bounds the number of reference tokens accessed by each query, processing extremely long multi-page documents remains constrained by the GPU memory required to prefill and retain their visual KV caches. As illustrated in Figure 4, conventional all-at-once prefill keeps all page-level visual KV caches in memory, causing the memory footprint to grow linearly with document length. To overcome this limitation, we introduce the Unlimited-Horizon Decoding Paradigm, which partitions the document into overlapping page-level chunks and processes them sequentially. Adjacent chunks share their boundary page, whose KV cache is directly reused to preserve cross-chunk continuity, while visual KV caches outside the active chunk are released. Consequently, only a fixed-size chunk remains resident in GPU memory, bounding the memory footprint independently of the total document length. The complete procedure is formally described in Algorithm 1.

![](images/402784a7a2ecd6247c0b038c2d0a681d8d9d8ca295605b87cdec4c6123fe644c.jpg)  
Figure 4 | Comparison of (a) standard decoding and (b) our Unlimited-Horizon Decoding paradigm. Standard decoding retains visual KV caches for all pages, causing GPU memory usage to grow linearly with document length and eventually overflow. Our paradigm divides long documents into manageable chunks and processes them sequentially, reusing the KV cache of shared boundary pages and releasing inactive caches to keep GPU memory usage bounded while preserving cross-chunk continuity.

## 4. Experimental Settings

## 4.1. Data Engine

To train Unlimited OCR 2 for versatile document parsing, we construct a large-scale multimodal corpus comprising 2M multi-page and 10M single-page documents. Pseudo-labels across both single- and multi-page subsets are extracted from MinerU2.5-Pro [26] and PaddleOCR-VLv1.6 [40], tailored to their respective domain strengths. To enrich visual and layout diversity, we additionally incorporate versatile synthetic document generation pipelines. Across both real and synthetic data, we leverage VLM-based multi-dimensional feature analysis to conduct balance filtering, ensuring uniform coverage over complex layout distributions. Finally, to explicitly fortify structural hierarchy and cross-page continuity in multi-page scenarios, we apply MinerU-Popo [34] refinement alongside dedicated heuristic post-processing rules to reconcile layout inconsistencies across adjacent page boundaries.

## 4.2. Implementation Details

We initialize Unlimited OCR 2 from the Unlimited OCR checkpoint and perform post-training for 1 epoch on the curated dataset. Training is conducted on a cluster of 25 × 8 GPU NVIDIA H800 nodes with a global batch size of 1,600 and a maximum sequence length of 64K tokens.

During training, we freeze both the DeepEncoder and the aligner, optimizing only the LLM parameters. Optimization is carried out using the AdamW optimizer with a cosine learning rate decay schedule, starting at an initial learning rate of $2 . 5 \times 1 0 ^ { - 4 }$ . The entire training workflow is implemented using the MS-Swift framework. For deployment, the Unlimited-Horizon Decoding Paradigm is integrated into SGLang to facilitate efficient long-document inference.

## 5. Evaluation

## 5.1. Benchmarks and Metrics

To comprehensively evaluate Unlimited OCR 2 on document parsing across varying lengths, we conduct experiments on two challenging multi-page benchmarks: MPDocBench [41] (up to 30 pages) and our proposed MarathonDocBench (ranging from 100 to 300 pages). Following the official evaluation protocol of MPDocBench, we report Edit Distance for textual content and reading order recognition, TEDS for table structure recognition, CDM for mathematical formula recognition, and F1 score for figure region detection. To further evaluate cross-page consistency, we include T-TextEdit and T-TEDS to assess text and table parsing across page boundaries, respectively. An overall score aggregated across these fine-grained structural categories is also computed to summarize overall parsing performance.

![](images/a1fb4f466cb4cb70bb0b1b554c215539b056a2a1df5bfeac808bc03a7fd741d8.jpg)  
Figure 5 | Performance comparison on multi-page benchmarks. Unlimited OCR 2 (Ours) consistently achieves state-of-the-art performance across all evaluation categories.

Table 1 | Performance comparison on MPDocBench. All models in the table are end-to-end VLM-based architectures.
<table><tr><td>Method</td><td>Overall↑</td><td>TextEdit↓</td><td>T-TextEdit ↓</td><td>FormulaCDM ↑ TableTEDS</td><td></td><td>T-TableTEDS</td><td></td><td>FigureF1 ↑ ReadOrderEdit↓</td><td>HeadingTEDS ↑</td></tr><tr><td colspan="12">General VLMs</td></tr><tr><td>Gemini-3.1-Pro Preview [9]</td><td>71.79</td><td>0.070</td><td>0.232</td><td>88.37</td><td>81.68</td><td>61.26</td><td>59.08</td><td>0.127</td><td>26.94</td></tr><tr><td>Qwen3.8-Max [25]</td><td>78.30</td><td>0.057</td><td>0.196</td><td>89.24</td><td>85.89</td><td>67.18</td><td>70.19</td><td>0.097</td><td>48.80</td></tr><tr><td>Kimi-K3 [24]</td><td>60.77</td><td>0.162</td><td>0.551</td><td>41.16</td><td>75.45</td><td>81.57</td><td>40.47</td><td>0.206</td><td>39.30</td></tr><tr><td>Qwen3.6-plus [25]</td><td>72.14</td><td>0.096</td><td>0.245</td><td>88.77</td><td>82.99</td><td>60.50</td><td>64.80</td><td>0.179</td><td>32.06</td></tr><tr><td>Qwen3-VL-235B [1]</td><td>73.98</td><td>0.087</td><td>0.187</td><td>84.71</td><td>81.35</td><td>61.25</td><td>63.25</td><td>0.136</td><td>42.34</td></tr><tr><td>InternVL-3.5-38B [27]</td><td>57.01</td><td>0.133</td><td>0.515</td><td>84.30</td><td>69.86</td><td>51.69</td><td>7.97</td><td>0.199</td><td>27.01</td></tr><tr><td>Minimax-M3 [16]</td><td>62.65</td><td>0.167</td><td>0.422</td><td>81.30</td><td>77.91</td><td>76.51</td><td>7.18</td><td>0.231</td><td>40.39</td></tr><tr><td>Gemini-3.8-Flash [10]</td><td>76.25</td><td>0.081</td><td>0.122</td><td>83.27</td><td>83.15</td><td>87.41</td><td>62.86</td><td>0.294</td><td>43.05</td></tr><tr><td>Qwen-3.8-Flash-Next [25]</td><td>77.36</td><td>0.078</td><td>0.163</td><td>86.70</td><td>86.24</td><td>88.36</td><td>64.31</td><td>0.277</td><td>45.09</td></tr><tr><td>DeepSeek-4.1-Flash [32]</td><td>66.25</td><td>0.114</td><td>0.451</td><td>82.12</td><td>80.14</td><td>87.72</td><td>35.55</td><td>0.312</td><td>32.10</td></tr><tr><td>GLM-5.3-Flash [39]</td><td>68.02</td><td>0.111</td><td>0.344</td><td>81.86</td><td>83.70</td><td>70.71</td><td>43.09</td><td>0.288</td><td>39.09</td></tr><tr><td colspan="2"></td><td></td><td></td><td>Specialized VLMs (OCR-2.0)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unlimited-OCR [37]</td><td>61.89</td><td>0.167</td><td>0.605</td><td>73.71</td><td>73.51</td><td>60.76</td><td>53.19</td><td>0.204</td><td>31.58</td></tr><tr><td colspan="10">Specialized VLMs (OCR-3.0)</td></tr><tr><td>Unlimited-OCR 2 (Ours)</td><td>84.59</td><td>0.052</td><td>0.100</td><td>89.39</td><td>87.78</td><td>89.54</td><td>72.91</td><td>0.077</td><td>60.10</td></tr></table>

Table 2 | Performance comparison on MarathonDocBench. Unlimited OCR is not reported on MarathonDocBench due to its 40-page input limitation.
<table><tr><td>Method</td><td>Overall↑</td><td>TextEdit↓</td><td>T-TextEdit ↓</td><td>FormulaCDM ↑ TableTEDS</td><td></td><td>T-TableTEDS</td><td></td><td>FigureF1 ↑ ReadOrderEdit↓</td><td>HeadingTEDS ↑</td></tr><tr><td colspan="10">General VLMs</td></tr><tr><td>Gemini-3.1-Pro Preview [9]</td><td>26.97</td><td>0.516</td><td>0.764</td><td>30.44</td><td>28.48</td><td>21.71</td><td>18.94</td><td>0.654</td><td>9.68</td></tr><tr><td>Qwen3.8-Max [25]</td><td>22.93</td><td>0.600</td><td>0.804</td><td>18.82</td><td>15.20</td><td>13.71</td><td>33.39</td><td>0.733</td><td>16.09</td></tr><tr><td>Kimi-K3 [24]</td><td>33.74</td><td>0.475</td><td>0.664</td><td>40.50</td><td>27.80</td><td>26.30</td><td>21.38</td><td>0.611</td><td>28.90</td></tr><tr><td>Minimax-M3 [16]</td><td>17.97</td><td>0.659</td><td>0.818</td><td>14.20</td><td>17.79</td><td>14.84</td><td>3.67</td><td>0.763</td><td>17.16</td></tr><tr><td>Gemini-3.8-Flash [10]</td><td>28.61</td><td>0.519</td><td>0.658</td><td>26.35</td><td>23.40</td><td>19.56</td><td>18.62</td><td>0.687</td><td>27.34</td></tr><tr><td>Qwen-3.8-Flash-Next [25]</td><td>27.83</td><td>0.544</td><td>0.753</td><td>28.31</td><td>25.38</td><td>21.04</td><td>24.07</td><td>0.679</td><td>21.50</td></tr><tr><td>DeepSeek-4.1-Flash [32]</td><td>12.34</td><td>0.740</td><td>0.933</td><td>10.45</td><td>12.74</td><td>12.59</td><td>6.95</td><td>0.843</td><td>7.62</td></tr><tr><td>GLM-5.3-Flash [39]</td><td>7.51</td><td>0.860</td><td>0.936</td><td>4.50</td><td>7.12</td><td>6.65</td><td>7.10</td><td>0.923</td><td>6.55</td></tr><tr><td colspan="10">Specialized VLMs (OCR-3.0)</td></tr><tr><td>Unlimited-OCR 2 (Ours)</td><td>77.06</td><td>0.104</td><td>0.123</td><td>87.93</td><td>74.81</td><td>71.11</td><td>71.79</td><td>0.147</td><td>48.20</td></tr></table>

## 5.2. Main Results

As visually summarized in Figure 6 and further detailed in Table 1 and Table 2, Unlimited OCR 2 takes a significant step toward the OCR 3.0 era, advancing document parsing along three key dimensions. First, it establishes the practical feasibility of hundred-page end-to-end parsing, achieving a score of 77.06 on MarathonDocBench and demonstrating its ability to handle documents at a scale beyond conventional multi-page settings. Second, it maintains robust parsing performance as document length increases. Through end-to-end modeling with SR-SWA, Unlimited OCR 2 filters redundant visual tokens while preserving relevant cross-page information, achieving the highest Overall scores and state-of-the-art performance on both cross-page text and table metrics, T-Text and T-Table, respectively, from dozens of pages on MPDocBench to hundreds of pages on MarathonDocBench. Third, it establishes a new stateof-the-art in long-document parsing, outperforming leading closed-source models and top-tier open-source MLLMs across these benchmarks. Furthermore, Unlimited OCR 2 retains strong single-page recognition capabilities, achieving an Overall score of 94.02% on OmniDocBench in Table 3, confirming that its sparse attention design preserves fine-grained local recognition.

Table 3 | Comparison on OmniDocBench v1.6. All models in the table are end-to-end VLM-based architectures.
<table><tr><td>Model</td><td>Size |Overall ↑</td><td colspan="6"> $\mathbf { \left| T e x t ^ { E d i t } \right. } ,$  FormulaCDM ↑ TableTEDs ↑  $\mathbf { T a b l e } ^ { \mathrm { T E D S - S } }$  ←  $\mathbf { R e a d O r d e r } ^ { \mathrm { E d i t } } \downarrow$ </td></tr><tr><td>HunyuanOCR [23]</td><td>1B</td><td>89.95</td><td>0.088</td><td>87.68</td><td>91.01</td><td>92.23</td><td>0.171</td></tr><tr><td>DeepSeek-OCR 2 [29]</td><td>3B-A0.5B</td><td>90.25</td><td>0.050</td><td>91.84</td><td>83.89</td><td>87.75</td><td>0.144</td></tr><tr><td>dots.ocr [19]</td><td>3B</td><td>90.77</td><td>0.048</td><td>89.95</td><td>87.18</td><td>90.58</td><td>0.138</td></tr><tr><td>FireRed-OCR [31]</td><td>2B</td><td>93.26</td><td>0.037</td><td>95.44</td><td>88.04</td><td>91.06</td><td>0.131</td></tr><tr><td>ABot-OCR [14]</td><td>2B</td><td>93.30</td><td>0.037</td><td>94.86</td><td>88.69</td><td>91.87</td><td>0.137</td></tr><tr><td>Logics-Parsing-v2 [4]</td><td>4B</td><td>93.33</td><td>0.041</td><td>95.65</td><td>88.42</td><td>91.98</td><td>0.137</td></tr><tr><td>Qianfan-OCR [7]</td><td>4B</td><td>93.90</td><td>0.040</td><td>95.08</td><td>90.53</td><td>93.31</td><td>0.130</td></tr><tr><td>OvisOCR2 [21]</td><td>0.8B</td><td>96.47</td><td>0.027</td><td>97.49</td><td>94.58</td><td>96.98</td><td>0.112</td></tr><tr><td>TeleOCR [3]</td><td>1.2B</td><td>96.91</td><td>0.027</td><td>96.60</td><td>96.82</td><td>98.18</td><td>0.118</td></tr><tr><td>Unlimited-OCR [37]</td><td>|3B-A0.5B</td><td>93.92</td><td>0.042</td><td>95.79</td><td>90.16</td><td>93.32</td><td>0.129</td></tr><tr><td>Unlimited-OCR 2</td><td>3B-A0.5B</td><td>94.02</td><td>0.038</td><td>96.29</td><td>89.60</td><td>92.70</td><td>0.128</td></tr></table>

Table 4 | Feasibility evaluation of the Unlimited-Horizon Decoding Paradigm. Sequential chunk processing reduces peak KV cache memory at B = 1 and improves end-to-end throughput at $B = 1 6 ,$ enabling hundred-page parsing under constrained GPU memory.
<table><tr><td>Method</td><td>Peak KV Memory (B = 1)</td><td>TPS (B = 16)</td></tr><tr><td>Standard Decoding</td><td>47.00 GB</td><td>360</td></tr><tr><td>Unlimited-Horizon Decoding</td><td>1.77 GB</td><td>1315</td></tr></table>

## 5.3. Scaling with Document Length

## 5.3.1. Feasibility of Unlimited-horizon Parsing

Benefiting from SR-SWA and the Unlimited-Horizon Decoding Paradigm, Unlimited OCR 2 enables efficient parsing of extremely long documents under constrained GPU memory. We evaluate its feasibility on the hundred-page MarathonDocBench benchmark by measuring peak KV cache memory and end-to-end throughput, as reported in Table 4. Compared with standard decoding, page-level chunking with boundary-page KV cache reuse reduces peak KV cache memory from 47.00 GB to 1.77 GB under single-sequence inference (B = 1). Meanwhile end-to-end throughput increases from 360 to 1,315 TPS under concurrent inference with B = 16. These results demonstrate that sequential chunk processing with KV cache reuse substantially reduces memory requirements while improving throughput, making hundred-page document parsing more practical under limited GPU memory.

## 5.3.2. Length-Invariant Accuracy

To verify whether extending the document horizon compromises parsing fidelity, we evaluate the performance trajectories of Unlimited OCR 2 across continuously increasing document lengths. As illustrated in Figure 6 (a), frontier Vision-Language Models (VLMs) typically experience a noticeable performance degradation as the sequence length grows, primarily caused by attention diffusion and context dilution over ultra-long horizons. In contrast, Unlimited OCR 2 exhibits remarkable length-invariant accuracy. By leveraging SR-SWA to dynamically isolate and retrieve only the most relevant visual tokens, our model maintains consistently high parsing precision from short sequences up to extreme horizons of 300 pages.

![](images/b837264eecebadcc6d244e6f9bccc8b45c673af013ec932286ae8a62ed5e2117.jpg)  
(a) Parsing accuracy

![](images/5288a88fd0822275c5bc3e03b488295e8b2737e43db658ea74127c399086e21f.jpg)  
(b) Decoding latency  
Figure 6 | Performance and efficiency comparison across varying document lengths. Unlimited-OCR 2 demonstrates (a) robust parsing accuracy and (b) near-constant decoding latency (ms/step) as document length scaling.

## 5.3.3. Constant Per-Step Decoding Latency

Figure 6 (b) examines how decoding efficiency scales with document length. All measurements are conducted in Gundam mode with a batch size of 3 and a fixed generation length of 1,024 tokens per page. As the document length grows, Full Attention incurs rapidly growing per-step decoding latency because each query must attend to a linearly expanding visual context. While R-SWA mitigates this overhead by bounding local window, its remote visual scope still scales with sequence length, leading to noticeable latency accumulation over ultra-long horizons. In contrast, SR-SWA strictly bounds the effective per-query attention scope, achieving near-constant per-step decoding latency from short contexts up to hundred-page documents. Consequently, its latency and throughput advantage over Full Attention and naive windowed baselines becomes increasingly pronounced as document length scales, confirming the computational viability of our paradigm for real-world long-document parsing.

## 6. Limitation and Future Work

Although Unlimited OCR 2 enables efficient parsing of hundred-page documents, its training data remains limited in diversity, and further data refinement will improve its robustness and generalization. More importantly, the current sparse-attention mechanism does not preserve global document information, limiting its understanding of document-level structures such as heading hierarchies. Future work may introduce architectures such as linear attention to retain global information with constant per-token computational cost, further improving documentlevel structure understanding.

## 7. Conclusion

In this technical report, we presented Unlimited OCR 2, a next-generation architecture tailored for unlimited-horizon document parsing. Specifically, Unlimited OCR 2 introduces Sparse Reference Sliding Window Attention (SR-SWA) to break the long-context visual computation bottleneck, guaranteeing constant per-token computational complexity throughout decoding. Additionally, integrated with an optimized page-split decoding paradigm, the model significantly mitigates GPU memory consumption, enabling robust unlimited-horizon inference on resource-constrained hardware. Extensive evaluations and analysis confirm that Unlimited OCR 2 maintains superior parsing accuracy alongside exceptional efficiency across all scales, thereby pioneering the OCR-3.0 era of unlimited-horizon document understanding.

## 8. Author List

Core Contributors: Shu Wei, Jingjing Wu, Qunyi Xie

Contributors: Lingshu Zhang, Xiangcheng Du, Yannian Fu, Mengjie Wang, Zuyi Wang, Zhichao Liu, Yangliu Xu, Youyang Yin, Huanhuan Liu, YY, Yi Yang

## References

[1] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[2] I. Beltagy, M. E. Peters, and A. Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

[3] P. Cai, Z. Zou, S. Liu, Y. Wang, J. Tang, K. Yang, M. Tong, M. Jiang, Z. He, and H. Sun. Teleocr: Navigating document parsing across digital and camera-captured documents. arXiv preprint arXiv:2608.12898, 2026. URL https://arxiv.org/abs/2608.12898.

[4] X. Chen, S. Li, X. Zhu, Y. Chen, F. Yang, C. Fang, L. Qu, X. Xu, H. Wei, and M. Wu. Logics-parsing technical report. arXiv preprint arXiv:2509.19760, 2025.

[5] C. Cui, T. Sun, S. Liang, T. Gao, Z. Zhang, J. Liu, X. Wang, C. Zhou, H. Liu, M. Lin, et al. Paddleocr-vl: Boosting multilingual document parsing via a 0.9 b ultra-compact vision-language model. arXiv preprint arXiv:2510.14528, 2025.

[6] C. Cui, T. Sun, S. Liang, T. Gao, Z. Zhang, J. Liu, X. Wang, C. Zhou, H. Liu, M. Lin, et al. Paddleocr-vl-1.5: Towards a multi-task 0.9 b vlm for robust in-the-wild document parsing. arXiv preprint arXiv:2601.21957, 2026.

[7] D. Dong, M. Zheng, D. Xu, C. Luo, B. Zhuang, Y. Li, R. He, H. Wang, W. Zhang, W. Wang, et al. Qianfan-ocr: A unified end-to-end model for document intelligence. arXiv preprint arXiv:2603.13398, 2026.

[8] S. Duan, Y. Xue, W. Wang, Z. Su, H. Liu, S. Yang, G. Gan, G. Wang, Z. Wang, S. Yan, et al. Glm-ocr technical report. arXiv preprint arXiv:2603.10910, 2026.

[9] Google DeepMind. Gemini 3.1. https://deepmind.google/models/gemini/pro/, 2026.

[10] Google DeepMind. Gemini3.8. https://deepmind.google/models/gemini/flash/, 2026.

[11] A. Gu and T. Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

[12] X. Hu, X. Wei, H. Gu, M. Zhang, T. Liang, H. Li, L. Zhu, Y. Wang, S. Han, Y. Bai, et al. Hierarchical sparse attention done right: Toward infinite context modeling. arXiv preprint arXiv:2607.02980, 2026.

[13] Y. Huang, N. M. Gonçalves, F. Alvetreti, L. Li, X. Han, E. M. Ponti, A. F. Martins, and M. V. Treviso. Dashattention: Differentiable and adaptive sparse hierarchical attention. arXiv preprint arXiv:2605.18753, 2026.

[14] K. Jiang, R. Gong, X. Cheng, K. Niu, T. Li, and M. Xu. Abot-ocr technical report. arXiv preprint arXiv:2605.27978, 2026.

[15] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR, 2020.

[16] X. Lai, W. Xu, Y. Yang, Q. Chen, Y. Xu, L. Zeng, X. Li, H. Sun, H. Zhu, V. Zhang, et al. Minimax sparse attention. arXiv preprint arXiv:2606.13392, 2026.

[17] A. Li, B. Gong, B. Yang, B. Shan, C. Liu, C. Zhu, C. Zhang, C. Guo, D. Chen, D. Li, et al. Minimax-01: Scaling foundation models with lightning attention. arXiv preprint arXiv:2501.08313, 2025.

[18] G. Li, X. Wan, S. Peng, W. Wang, H. Feng, Y. Du, B. Wu, Z. Ruan, Z. Lu, L. Wu, et al. Hunyuanocr-1.5: Making lightweight ocr vlms faster and better. arXiv preprint arXiv:2607.04884, 2026.

[19] Y. Li, G. Yang, H. Liu, B. Wang, and C. Zhang. dots. ocr: Multilingual document layout parsing in a single vision-language model. arXiv preprint arXiv:2512.02498, 2025.

[20] Z. Li, Y. Liu, Q. Liu, Z. Ma, Z. Zhang, S. Zhang, Z. Guo, J. Zhang, X. Wang, and X. Bai. MonkeyOCR: Document parsing with a structure-recognition-relation triplet paradigm. arXiv preprint arXiv:2506.05218, 2025.

[21] S. Lu, Y. Li, Y. Xia, Y. Chen, A.-Y. Ji, J.-P. Jiang, Q.-G. Chen, J. Zhao, E. Lin, H. Li, et al. Ovisocr2 technical report. arXiv preprint arXiv:2607.13639, 2026.

[22] J. Niu, Z. Liu, Z. Gu, B. Wang, L. Ouyang, Z. Zhao, T. Chu, T. He, F. Wu, Q. Zhang, et al. MinerU2.5: A decoupled vision-language model for efficient high-resolution document parsing. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics-Industry Track, 2025.

[23] H. V. Team, P. Lyu, X. Wan, G. Li, S. Peng, W. Wang, L. Wu, H. Shen, Y. Zhou, C. Tang, et al. Hunyuanocr technical report. arXiv preprint arXiv:2511.19575, 2025.

[24] K. Team, T. Bai, Y. Bai, Y. Bao, S. Cai, Y. Cao, Y. Charles, H. Che, C. Chen, G. Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

[25] TongYi. TongYi. https://www.aliyun.com/product/tongyi,2025.

[26] B. Wang, T. He, L. Ouyang, F. Wu, Z. Zhao, T. Chu, Y. Qu, Z. Jin, W. Zeng, Z. Miao, et al. Mineru2. 5-pro: Pushing the limits of data-centric document parsing at scale. arXiv preprint arXiv:2604.04771, 2026.

[27] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

[28] H. Wei, Y. Sun, and Y. Li. Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234, 2025.

[29] H. Wei, Y. Sun, and Y. Li. Deepseek-ocr 2: Visual causal flow. arXiv preprint arXiv:2601.20552, 2026.

[30] S. Wei, J. Wu, L. Zhang, Q. Xie, H. Zou, L. Xiang, X. Fan, Y. Xu, M. Lin, X. Ma, et al. Hpd-parsing: Hierarchical parallel document parsing. arXiv preprint arXiv:2607.18839, 2026.

[31] H. Wu, H. Lou, X. Li, Z. Zhong, Z. Sun, P. Chen, X. Zhou, K. Zuo, Y. Chen, X. Tang, et al. Firered-ocr technical report. arXiv preprint arXiv:2603.01840, 2026.

[32] A. Xu, B. Li, B. Lin, B. Xue, B. Xian, B. Xu, B. Wu, B. Zhang, B. Deng, C. Yu, et al. Deepseekv4. 1-flash: Pushing the limits of kv cache compression. arXiv preprint arXiv:2609.19969, 2026.

[33] A. Xu, B. Lin, B. Xue, B. Wang, B. Xu, B. Wu, B. Zhang, C. Lin, C. Dong, C. Ling, et al. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

[34] B. Xu, Z. Miao, X. Zhou, Y. Lin, Z. Tang, X. Zhao, F. Wu, C. Tan, B. Wang, and C. He. Mineru-popo: Universal post-processing model for structured document parsing. arXiv preprint arXiv:2605.24973, 2026.

[35] S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim. Gated linear attention transformers with hardware-efficient training. arXiv preprint arXiv:2312.06635, 2023.

[36] S. Yang, B. Wang, Y. Zhang, Y. Shen, and Y. Kim. Parallelizing linear transformers with the delta rule over sequence length. Advances in neural information processing systems, 37: 115491–115522, 2024.

[37] Y. Yin, H. Liu, Q. Xie, C. Liu, S. Yang, S. Wang, Z. Liu, H. Zou, J. Chen, S. Wei, et al. Unlimited ocr works. arXiv preprint arXiv:2606.23050, 2026.

[38] M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti, S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang, et al. Big bird: Transformers for longer sequences. Advances in neural information processing systems, 33:17283–17297, 2020.

[39] Z.AI. GLM-5.3-Flash. https://huggingface.co/zai-org/GLM-5.3-Flash. Model card, accessed 2026-09-25.

[40] Z. Zhang, H. Liu, S. Liang, Y. Zhang, Y. Xiang, J. Liu, T. Sun, M. Lin, Y. Zhang, C. Zhou, et al. Paddleocr-vl-1.6: Expanding the frontier of document parsing with under-optimized region refinement and progressive post-training. arXiv preprint arXiv:2606.03264, 2026.

[41] B. Zhou, H. Xing, Y. Chen, J. Xu, Q. Zheng, F. Gao, Z. Yang, S. Bai, M. Yan, J. Ye, et al. Mpdocbench-parse: Benchmarking practical multi-page document parsing. arXiv preprint arXiv:2605.22100, 2026.