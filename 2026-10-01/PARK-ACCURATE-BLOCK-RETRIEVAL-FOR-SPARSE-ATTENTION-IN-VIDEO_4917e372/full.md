# PARK: ACCURATE BLOCK RETRIEVAL FOR SPARSE ATTENTION IN VIDEO DIFFUSION TRANSFORMERS

Yun Dai<sup>1</sup> Jiarui Wen<sup>1</sup> Huiping Zhuang<sup>1</sup> Cen Chen<sup>1</sup> Ziqian Zeng<sup>1†</sup>   
<sup>1</sup>South China University of Technology   
zqzeng@scut.edu.cn

## ABSTRACT

Diffusion Transformers (DiTs) have become a dominant architecture for video generation, but their efficiency is limited by the quadratic complexity of full attention. Sparse attention reduces this cost by retrieving important blocks and computing attention only within them, but inaccurate retrieval can either degrade generation quality or yield unnecessary computation. We identify two retrieval mismatches in methods that retrieve blocks using the averaged representations of query and key blocks: (i) query-side aggregation mismatch, where averaging queries before Softmax fails to preserve their individual attention preferences, and (ii) key-side clustering metric mismatch, where standard Euclidean clustering in the original key space can group keys with dissimilar QK scores under the current query, so their average representation may not accurately represent how the current query scores individual keys. These mismatches can lead to inaccurate block retrieval. To address these mismatches, we propose PARK, a training-free sparse attention method for accurate block retrieval. PARK retains every original query, independently normalizes its attention over key blocks, and then averages these distributions within each query block. It also uses information from the current queries to transform keys before clustering, so that keys receiving similar QK scores are grouped together. A fused GPU kernel further reduces the overhead of block retrieval. Experiments on HunyuanVideo and Wan demonstrate that PARK improves block retrieval accuracy and preserves generation quality while accelerating inference, achieving the best quality-efficiency trade-off among the compared sparse attention methods.

## 1 INTRODUCTION

Diffusion Transformers (DiTs) (Peebles & Xie (2023)) have emerged as a dominant paradigm for modern video generation, demonstrating strong scalability and enabling high-fidelity, temporally coherent synthesis (Yang et al. (2025b); Kong et al. (2024); Wan et al. (2025)). However, generating high-resolution and long videos involves a large number of spatio-temporal tokens. Since 3D full attention scales quadratically with token count, it becomes a major computational bottleneck in video generation (Xi et al. (2025); Zhang et al. (2025b); Vaswani et al. (2017)).

Fortunately, attention maps in video DiTs are inherently sparse, with only a small fraction of token interactions contributing significantly to the output (Sun et al. (2026); Zhang et al. (2026)). Sparse attention exploits this property by computing attention only over these critical token interactions, substantially reducing computation with little degradation in generation quality. In practice, query and key tokens are partitioned into blocks by grouping consecutive tokens, and attention is computed only for a selected subset of blocks. These blocks are retrieved by estimating their importance and selecting those with the highest scores (Guo et al. (2024); Li et al. (2025c)). This process, referred to as block retrieval, produces a sparse mask that specifies which blocks are computed.

Existing training-free sparse attention methods broadly fall into static and dynamic methods. Static methods (Li et al. (2025a); Xi et al. (2025); Li et al. (2025b); Zhang et al. (2025c)) derive sparse masks from offline attention profiling or predefined rules and apply them to individual layers or

![](images/827a3595c16cb06e38663eb1e0e80507dc038731eed175fe81321491c0a00855.jpg)  
(a) Query-side Aggregation Mismatch

![](images/4e8b42ee4429b2c605dbf5fba31f92a9d0bc0bab4443eedefd9cdf3e9a514fe0.jpg)  
(b) Key-side Clustering Metric Mismatch

![](images/6907e3d2c64e9d0235fe46a1c266bb7f2ddfecdaef6535e25d5807f5883afc3b.jpg)  
(c) Two-sided Approximation Error

Figure 1: Two-sided retrieval mismatches and approximation errors in centroid-based block retrieval. (a) Averaging queries into a single centroid before Softmax normalization ignores their individual key preferences, leading to inaccurate block retrieval. (b) Standard Euclidean clustering considers only distances between keys in feature space and does not ensure that keys within the same cluster receive similar scores from the current query, leading to inaccurate key-centroid estimates. (c) Retaining the original queries instead of using query centroids (top) and clustering keys by QK-score similarity (bottom) both reduce approximation errors. More experimental details are provided in Appendix A.

attention heads during inference. Dynamic methods (Xia et al. (2025); Zhang et al. (2025a); Xu et al. (2025)) perform block retrieval at runtime using lightweight block importance estimation, adapting the sparse masks to the current attention pattern. Additionally, recent token-reordering methods (Yang et al. (2025a); Zhao et al. (2025); Tan et al. (2026a); Luo et al. (2026)) cluster semantically similar tokens and permute them into contiguous blocks. This helps retain more critical interactions with fewer blocks, improving the efficiency-quality trade-off.

Even with token reordering, the effectiveness of sparse attention depends on how accurately it retrieves important blocks. Missing important blocks can degrade generation quality, while selecting unimportant blocks wastes computation. To compensate for inaccurate retrieval, more blocks may need to be retained to preserve generation quality, limiting the achievable speedup. Accurate block retrieval is therefore essential to improving the efficiency-quality trade-off of sparse attention.

To keep block retrieval efficient, many existing methods estimate block importance from the average-pooled representations (or centroids) of query and key blocks, Our theoretical analysis of the resulting estimation errors reveals two retrieval mismatches in this common centroid-based design:

• M1: Query-side aggregation mismatch. Each query independently yields an attention distribution over all key blocks, and these query-specific preferences jointly determine which blocks should be selected for the sparse mask. As analyzed in Sec. 4.1, averaging the original queries into a single query centroid before Softmax normalization ignores these individual preferences and can misestimate the attention assigned to each key block, leading to an inaccurate sparse mask. Figure 1(c) shows that retaining the original queries instead of using query centroids reduces attention-weight approximation error by 47.7–49.0%.

• M2: Key-side clustering metric mismatch. Existing reordering methods cluster key tokens using standard Euclidean clustering in the original key space and use the key block centroids for importance estimation. As analyzed in Sec. 4.2, centroid-based estimation is more accurate when keys in the same block receive similar dot-product scores from the current query. However, this clustering metric compares only the keys themselves and does not account for how the current query scores them. Consequently, standard Euclidean clustering may group keys that are scored differently by the current query, making key centroids less accurate for block importance estimation. Figure 1(c) shows that clustering keys by QK-score similarity reduces centered log-mass error by 16.9–17.2% compared with standard Euclidean clustering.

To address these mismatches, we propose PARK, a training-free sparse attention method for accurate block retrieval via Per-query Attention-Mass Aggregation (PAMA) and Query-Response Key Clustering (QRKC). First, PAMA retains original queries, computes a separate Softmax distribution over the key blocks for each query, and aggregates these distributions by averaging them within each query block to estimate block importance for sparse mask selection. This preserves query-specific attention preferences and eliminates the aggregation mismatch introduced by query centroids. We fuse per-query normalization and averaging into a custom kernel for efficient execution. Second, QRKC uses a linear transformation derived from the current queries to map the keys into a queryconditioned space for clustering, where Euclidean distance reflects how similarly those queries score them. By grouping keys that are scored similarly by the current queries, QRKC reduces the approximation error introduced by key centroids. Together, PAMA and QRKC improve block retrieval with negligible overhead. Our contributions are summarized as follows:

• We identify two retrieval mismatches in existing centroid-based block retrieval methods: query-side aggregation mismatch and key-side clustering metric mismatch, and analyze how they introduce errors in block-importance estimation.

• We propose PARK, a novel training-free sparse attention method for fast video generation. It addresses the two identified mismatches through Per-query Attention-Mass Aggregation and Query-Response Key Clustering.

• Extensive experiments on HunyuanVideo and Wan demonstrate that PARK consistently improves block retrieval accuracy and preserves generation quality with high end-to-end speedup, achieving the best quality-efficiency trade-off among the compared training-free sparse attention methods.

## 2 RELATED WORK

Training-based Sparse Attention. Training-based sparse attention methods fine-tune video diffusion models with task-specific sparse attention patterns. VSA (Zhang et al. (2026)) employs a coarse-to-fine routing mechanism to select important attention blocks, while VMoBA (Wu et al. (2026)) cycles through 1D, 2D, and 3D block partitions across layers and retrieves blocks using a global similarity threshold. Other methods, such as DSV (Tan et al. (2026b)), BSA (Zhan et al. (2025)), and MoGA (Jia et al. (2026)), train lightweight predictors, routers, or cluster assignments to estimate sparse attention patterns before computation. However, they generally require additional training or fine-tuning and may introduce auxiliary prediction modules.

Training-free Sparse Attention. Training-free methods directly exploit attention sparsity in pretrained video DiTs without additional optimization. Static methods (Xi et al. (2025); Zhang et al. (2025c); Li et al. (2025a;b)) derive sparse masks from offline profiling or predefined spatio-temporal structures. Dynamic methods (Jiang et al. (2024); Zhang et al. (2025a); Xu et al. (2025); Xia et al. (2025)) instead construct sparse masks at runtime by estimating the importance of attention blocks. Recent methods further improve sparse attention through token reordering. PAROAttention (Zhao et al. (2025)) performs pattern-aware token reordering for sparse and quantized attention. SVG2 (Yang et al. (2025a)) and AdaCluster (Tan et al. (2026a)) apply K-means clustering to query and key tokens and reorder semantically similar tokens into contiguous blocks, while SVOO (Luo et al. (2026)) incorporates query–key interactions through bidirectional co-clustering.

## 3 PRELIMINARY

## 3.1 ATTENTION AND SPARSE ATTENTION

For a single attention head, the input tokens are projected into query, key, and value matrices Q, K, $\mathbf { V } \ \in \ \mathbb { R } ^ { L \times d }$ , where L is the sequence length and d is the head dimension. Full attention (Vaswani et al. (2017); Dao et al. (2022)) is computed as

$$
\mathbf { Z } = \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } , \qquad \mathbf { A } = \mathrm { S o f t m a x } ( \mathbf { Z } ) , \qquad \mathbf { O } = \mathbf { A } \mathbf { V } ,\tag{1}
$$

where the Softmax is applied along the key dimension independently for each query. Computing the full attention matrix incurs quadratic complexity with respect to the sequence length.

Given a sparse mask $\mathbf { M } \in \{ 0 , 1 \} ^ { L \times L }$ , sparse attention is computed as

$$
\mathbf { O } ^ { \mathrm { s p } } = ( \mathbf { A } \odot \mathbf { M } ) \mathbf { V } ,\tag{2}
$$

where ⊙ denotes element-wise masking, and $M _ { i j } = 0$ indicates that the interaction from query i to key j is skipped.

In practice, query and key tokens are partitioned into query blocks $\{ \mathcal { Q } _ { u } \} _ { u = 1 } ^ { N _ { Q } }$ and key blocks $\{ \mathcal { K } _ { v } \} _ { v = 1 } ^ { N _ { K } }$ , respectively. Each query–key block consists of all attention interactions between the queries in $\mathcal { Q } _ { u }$ and the keys in $\bar { \kappa } _ { v } ,$ , corresponding to a submatrix of $\mathbf { A }$ . The sparse mask selects or skips each query–key block as a whole. The block sizes are denoted by $n _ { u } ^ { Q } = \bar { | } \mathcal { Q } _ { u } |$ and $n _ { v } ^ { K } = | \mathcal { K } _ { v } |$ with $\begin{array} { r } { \sum _ { u = 1 } ^ { N _ { Q } } n _ { u } ^ { Q } = \sum _ { v = 1 } ^ { N _ { K } } n _ { v } ^ { K } = L } \end{array}$

## 3.2 BLOCK IMPORTANCE AND ATTENTION RECALL

For a query block $\mathcal { Q } _ { u }$ and a key block $\kappa _ { v } ,$ we define their exact block importance $W _ { u v }$ as the average attention weight that the queries in $\mathcal { Q } _ { u }$ assign to $\textstyle { \mathcal { K } } _ { v }$ :

$$
W _ { u v } = \frac { 1 } { n _ { u } ^ { Q } } \sum _ { i \in \mathcal { Q } _ { u } } \sum _ { j \in \mathcal { K } _ { v } } A _ { i j } .\tag{3}
$$

Here, $A _ { i j }$ denotes the dense post-Softmax attention weight from query token i to key token $j .$ For every query block, the block importance is normalized over all key blocks, i.e., $\begin{array} { r } { \sum _ { v = 1 } ^ { N _ { K } } W _ { u v } = 1 } \end{array}$ Under Top-p retrieval, key blocks are retrieved for each query block until their cumulative estimated importance reaches the threshold $p .$ . In practice, key blocks are retrieved in descending order of $\widehat { W } _ { u v }$ , which estimates the exact block importance $W _ { u v }$ . The retained query–key block pairs form the block-level sparse mask M.

The block retrieval accuracy of sparse attention is measured using attention recall, defined as

$$
\operatorname { R e c a l l } ( \mathbf { A } , \mathbf { M } ) = \frac { \sum _ { i = 1 } ^ { L } \sum _ { j = 1 } ^ { L } A _ { i j } M _ { i j } } { \sum _ { i = 1 } ^ { L } \sum _ { j = 1 } ^ { L } A _ { i j } } .\tag{4}
$$

Attention recall (Treviso et al. (2022)) measures the proportion of full-attention weights covered by the block-level sparse mask. A higher recall indicates more accurate block retrieval.

## 4 MOTIVATION

Centroid-based block retrieval compresses both query and key blocks before estimating their importance. Although this design is efficient, the two compressions introduce fundamentally different approximations. We analyze them relative to the exact block importance in Eq. (3) and show how they lead to the two retrieval mismatches.

## 4.1 QUERY-SIDE AGGREGATION MISMATCH

Averaging queries into a centroid before Softmax can lead to inaccurate block-importance estimates. To analyze this error, we retain all original keys. The exact block importance $W _ { u v }$ in Eq. (3) is computed by first normalizing attention for each query and then averaging the attention weights of queries in the same block. Query-centroid estimation instead replaces these queries with their centroid before Softmax normalization. A Taylor expansion expresses the approximation error as

$$
\Delta _ { u v } ^ { Q } = W _ { u v } - \widehat { W } _ { u v } ^ { \mathrm { c e n t r o i d } } = \frac { 1 } { 2 } \left. \mathbf { H } _ { \phi _ { v } } ( \overline { { \mathbf { q } } } _ { u } ) , \pmb { \Sigma } _ { Q , u } \right. _ { F } + \mathcal { R } _ { u v } ^ { Q } .\tag{5}
$$

Here, $\widehat { W } _ { u v } ^ { \mathrm { c e n t r o i d } } = \phi _ { v } ( \overline { { \mathbf { q } } } _ { u } )$ is the query-centroid estimate, where $\phi _ { v } ( \mathbf { q } )$ is the exact attention weight that query q assigns to key block $\mathcal { \kappa } _ { v } .$ , and $\overline { { \mathbf { q } } } _ { u }$ is the query-block centroid. $\Sigma _ { Q , u }$ is the within-block query covariance, and $\mathbf { H } _ { \phi _ { v } } ( \overline { { \mathbf { q } } } _ { u } )$ is the Hessian of $\phi _ { v } ,$ capturing how $\phi _ { v }$ curves as the query varies

Table 1: Top-p block-retrieval recall $( p = 0 . 9 )$ under different query/key representations. Mean, P05, and Worst denote the mean, fifth percentile, and minimum attention recall across all attention heads and layers, respectively. All values are percentages. Higher is better.
<table><tr><td rowspan="2">Ordering</td><td rowspan="2">Query</td><td rowspan="2">Key</td><td colspan="3">HunyuanVideo (Kong et al. (2024))</td><td colspan="3">Wan 2.1-14B (Wan et al. (2025))</td></tr><tr><td>Mean</td><td>P05</td><td>Worst</td><td>Mean</td><td>P05</td><td>Worst</td></tr><tr><td rowspan="4">Clustered</td><td>Full</td><td>Full</td><td>91.29</td><td>90.02</td><td>90.01</td><td>91.71</td><td>90.02</td><td>90.01</td></tr><tr><td>Centroid Centroid</td><td></td><td>88.06</td><td>81.53</td><td>64.21</td><td>90.43</td><td>87.11</td><td>73.89</td></tr><tr><td>Centroid</td><td>Full</td><td>89.10</td><td>84.44</td><td>68.40</td><td>91.00</td><td>88.64</td><td>79.06</td></tr><tr><td>Full</td><td>Centroid</td><td>90.70</td><td>87.62</td><td>81.64</td><td>91.56</td><td>89.36</td><td>83.61</td></tr><tr><td rowspan="4">Original</td><td>Full</td><td>Full</td><td>90.64</td><td>90.02</td><td>90.01</td><td>90.89</td><td>90.03</td><td>90.02</td></tr><tr><td>Centroid Centroid</td><td></td><td>85.59</td><td>72.40</td><td>22.94</td><td>90.01</td><td>83.96</td><td>69.11</td></tr><tr><td>Centroid</td><td>Full</td><td>86.18</td><td>56.17</td><td>15.02</td><td>89.24</td><td>74.92</td><td>39.94</td></tr><tr><td>Full</td><td>Centroid</td><td>87.80</td><td>76.35</td><td>24.30</td><td>91.32</td><td>87.02</td><td>73.55</td></tr></table>

near its centroid. $\langle \cdot , \cdot \rangle _ { F }$ denotes the Frobenius inner product, and $\mathcal { R } _ { u v } ^ { Q }$ is the higher-order remainder.   
The full derivation is provided in Appendix D.1.

Accurate query-centroid estimation requires a small $| \Delta _ { u v } ^ { Q } | .$ , whose leading error is determined by $\langle \mathbf { H } _ { \phi _ { v } } ( \overline { { \mathbf { q } } } _ { u } ) , \overline { { \mathbf { \Lambda } } } _ { \mathbf { \Lambda } ^ { \perp } \mathbf { \Lambda } _ { u } } \rangle _ { F }$ . A sufficient condition for a small leading error is that the product of the norm of ${ \bf H } _ { \phi _ { v } } ( \overline { { \bf q } } _ { u } )$ and the total within-block query variance is small. Standard Euclidean query clustering reduces the total within-block query variance, captured by $\Sigma _ { Q , u } ,$ but does not generally eliminate it. Since query clustering does not account for the Hessian ${ \bf H } _ { \phi _ { v } } ( \overline { { \bf q } } _ { u } )$ , it does not by itself ensure that this product is sufficiently small. Empirically, Table 4 shows that query-centroid estimation with full keys incurs even higher approximation error than retaining original queries with key centroids.

We further evaluate whether retaining the original queries remains beneficial when keys are represented by centroids. Figure 1(c) shows that retaining original queries reduces approximation error when both estimates use the same key centroids. Table 1 further shows that retaining full queries with key centroids achieves the highest recall among the compressed representations across both ordering schemes and models. These empirical results support preserving query-specific attention preferences in practical block retrieval and motivate Per-Query Attention-Mass Aggregation (§5.1).

## 4.2 KEY-SIDE CLUSTERING METRIC MISMATCH

With the original queries retained, the exact attention weight $\phi _ { v } ( \mathbf { q } _ { i } )$ is obtained by applying Softmax to the block log-masses: $\begin{array} { r } { \phi _ { v } ( \mathbf { q } _ { i } ) = \exp ( S _ { i v } ^ { \star } ) / \sum _ { r = 1 } ^ { N _ { K } } \exp ( S _ { i r } ^ { \star } ) } \end{array}$ , where $S _ { i v } ^ { \star }$ is the exact log-mass of key block $\mathcal { K } _ { v }$ for query $\mathbf { q } _ { i }$ . The exact block log-mass and its key-centroid estimate are

$$
S _ { i v } ^ { \star } = \log \sum _ { j \in \mathcal { K } _ { v } } \exp \left( \frac { { \bf q } _ { i } ^ { \top } { \bf k } _ { j } } { \sqrt { d } } \right) , \qquad \widehat { S } _ { i v } = \log n _ { v } ^ { K } + \frac { { \bf q } _ { i } ^ { \top } { \bf c } _ { v } } { \sqrt { d } } .\tag{6}
$$

Here, $\mathbf { c } _ { v } = ( 1 / n _ { v } ^ { K } ) \sum _ { j \in { \mathcal { K } } _ { v } } \mathbf { k } _ { j }$ is the key-block centroid, and $n _ { v } ^ { K }$ is the number of keys in $\textstyle { \mathcal { K } } _ { v }$ . The term log $n _ { v } ^ { K }$ accounts for key-block size. An expansion expresses the key-centroid approximation error as

$$
\Delta _ { i v } ^ { K } = S _ { i v } ^ { \star } - \widehat { S } _ { i v } = \frac 1 { 2 d } \mathbf { q } _ { i } ^ { \top } \boldsymbol { \Sigma } _ { K , v } \mathbf { q } _ { i } + \mathcal { R } _ { i v } ^ { K } .\tag{7}
$$

Here, $\Sigma _ { K , v }$ is the within-block key covariance, $\begin{array} { r } { \epsilon _ { i v } ^ { K } = \frac { 1 } { 2 d } \mathbf q _ { i } ^ { \top } \pmb { \Sigma } _ { K , v } \mathbf q _ { i } } \end{array}$ is the leading term of $\Delta _ { i v } ^ { K }$ , and $\mathcal { R } _ { i v } ^ { K }$ is the higher-order remainder. The full derivation is provided in Appendix D.2.

Accurate key-centroid log-mass estimation requires a small $| \Delta _ { i v } ^ { K } |$ . The leading term $\epsilon _ { i v } ^ { K }$ is proportional to the within-block key variance after projection onto the current query, which is the variance of the unscaled QK scores within that block. Standard euclidean clustering aims to reduce withinblock key variation, captured by $\Sigma _ { K , v }$ , but does not account for the current query $\mathbf { q } _ { i }$ and therefore does not directly minimize $\epsilon _ { i v } ^ { K } .$ . Additional empirical evidence for this metric mismatch is provided in Appendix B. Therefore, key clustering should account for QK-score similarity under the current query, motivating Query-Response Key Clustering (§5.2).

![](images/0b18561ec2bb2a5ebe22fc7b9e225b68cba51f42303defdcc58b6cca40ea1e8c.jpg)  
Figure 2: Overview of PARK. QRKC transforms keys using a metric derived from the current queries, so that keys receiving similar QK scores are grouped together, while queries are clustered using standard Euclidean K-means. PAMA retains every original query, computes its Softmaxnormalized weights over the key centroids, and averages them within each query block to produce the block-importance map.

## 5 METHODOLOGY

In this section, we introduce PARK, a training-free sparse attention framework designed for accurate block retrieval via Per-Query Attention-Mass Aggregation (PAMA) and Query-Response Key Clustering (QRKC). PAMA preserves all original queries, computes their normalized attention over key blocks separately, and then averages these attention weights within each query block for blockimportance estimation (§5.1). QRKC uses information from the current queries to transform the keys before clustering, so that keys receiving similar QK scores are grouped together (§5.2). These two modules improve block retrieval on both the query and key sides. An overview of our proposed PARK framework is shown in Figure 2.

## 5.1 PAMA: PER-QUERY ATTENTION-MASS AGGREGATION

As shown in Eq. (3), exact block importance is computed by normalizing attention independently for every query and then averaging the attention weights within each query block. Following this order, PAMA retains all original queries during block-importance estimation. For each key block $\mathcal { \kappa } _ { v }$ , we compute its centroid $\mathbf { c } _ { v }$ in the original key space and approximate its log-mass $\widehat { S } _ { i v }$ for query q<sub>i</sub> as

$$
\mathbf { c } _ { v } = \frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in \mathcal { K } _ { v } } \mathbf { k } _ { j } , \qquad \widehat { S } _ { i v } = \frac { \mathbf { q } _ { i } ^ { \top } \mathbf { c } _ { v } } { \sqrt { d } } + \log n _ { v } ^ { K } .\tag{8}
$$

Here, $\kappa _ { v }$ denotes a key block containing $n _ { v } ^ { K }$ keys. The block-size term log $n _ { v } ^ { K }$ accounts for the different numbers of keys represented by the block centroids.

PAMA then normalizes the estimated scores over all key blocks for each query and averages the block weights within query block $\mathcal { Q } _ { u }$

$$
\widehat { P } _ { i v } = \frac { \exp ( \widehat { S } _ { i v } ) } { \sum _ { r = 1 } ^ { N _ { K } } \exp ( \widehat { S } _ { i r } ) } , \qquad \widehat { W } _ { u v } ^ { \mathrm { P A M A } } = \frac { 1 } { n _ { u } ^ { Q } } \sum _ { i \in \mathcal { Q } _ { u } } \widehat { P } _ { i v } .\tag{9}
$$

Here, $N _ { K }$ is the number of key blocks, and $\mathcal { Q } _ { u }$ is a query block containing $n _ { u } ^ { Q }$ queries. $\widehat { P } _ { i v }$ is the estimated attention weight that query $\mathbf { q } _ { i }$ assigns to key block $\textstyle { \mathcal { K } } _ { v }$ , and $\widehat { W } _ { u v } ^ { \mathrm { P A M A } }$ is the importance of the query–key block.

Efficient Kernel Implementation. For each attention head, PAMA requires $\mathcal { O } ( L N _ { K } d )$ operations for QK-centroid scores and $\mathcal { O } ( L N _ { K } )$ for row-wise Softmax and averaging within query blocks, far below full attention’s $\mathcal { O } ( L ^ { 2 } d )$ cost since $N _ { K } \ll L$ . A straightforward implementation stores $L \times N _ { K }$ logits and probabilities in HBM, incurring $\mathcal { O } ( L N _ { K } )$ ) intermediate storage and additional memory access. To reduce this memory overhead, we fuse the QK-centroid matrix multiplication, row-wise Softmax, and averaging within query blocks into a single GPU kernel. The intermediate logits and probabilities remain in on-chip registers, and only the final $N _ { Q } \times N _ { K }$ block-importance matrix is written to HBM. The pseudocode and overhead analysis are provided in Appendix E.

## 5.2 QRKC: QUERY-RESPONSE KEY CLUSTERING

QRKC aims to reduce the leading key-centroid approximation error $\epsilon _ { i v } ^ { K }$ identified in Sec. 4.2. To construct a key-clustering objective that accounts for all current queries, we average the squared QK-score error caused by replacing keys with their centroids over these queries:

$$
\mathcal { I } _ { \mathrm { Q R K C } } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \sum _ { v = 1 } ^ { N _ { K } } \sum _ { j \in K _ { v } } \left[ \mathbf { q } _ { i } ^ { \top } ( \mathbf { k } _ { j } - \mathbf { c } _ { v } ) \right] ^ { 2 } = \sum _ { v = 1 } ^ { N _ { K } } \sum _ { j \in K _ { v } } \| \mathbf { k } _ { j } - \mathbf { c } _ { v } \| _ { \mathbf { G } _ { Q } } ^ { 2 } .\tag{10}
$$

Here, $\mathbf { G } _ { Q } = ( 1 / L ) \mathbf { Q } ^ { \top } \mathbf { Q } \in \mathbb { R } ^ { d \times d }$ is the uncentered second-moment matrix of the current queries, and $\| \mathbf { x } \| _ { \mathbf { G } _ { Q } } ^ { 2 } = \mathbf { x } ^ { \top } \mathbf { G } _ { Q } \mathbf { x }$ . The relation $\begin{array} { r } { \mathcal { I } _ { \mathrm { Q R K C } } = 2 d \sum _ { v } { n } _ { v } ^ { K } ( 1 / L ) \sum _ { i } \epsilon _ { i v } ^ { K } } \end{array}$ connects this clustering objective $\mathcal { I } _ { \mathrm { Q R K C } }$ directly to the leading key-centroid approximation error $\epsilon _ { i v } ^ { K }$

To minimize $\mathcal { I } _ { \mathrm { Q R K C } }$ efficiently, we reuse a standard Euclidean K-means implementation. However, standard K-means uses the squared Euclidean distance $\| \mathbf { k } _ { j } - \mathbf { c } _ { v } \| _ { 2 } ^ { 2 }$ , whereas our objective in Eq. (10) uses $\| \mathbf { k } _ { j } - \mathbf { c } _ { v } \| _ { \mathbf { G } _ { \mathcal { O } } } ^ { 2 } = ( \mathbf { k } _ { j } - \mathbf { c } _ { v } ) ^ { \top } \mathbf { G } _ { \boldsymbol { Q } } ( \mathbf { k } _ { j } - \mathbf { c } _ { v } )$ . We therefore transform the keys into a space where the squared Euclidean distance between each key and its centroid satisfies the following equality:

$$
\| \widetilde { \mathbf { k } } _ { j } - \widetilde { \mathbf { c } } _ { v } \| _ { 2 } ^ { 2 } = ( \mathbf { k } _ { j } - \mathbf { c } _ { v } ) ^ { \top } \mathbf { G } _ { Q } ( \mathbf { k } _ { j } - \mathbf { c } _ { v } ) .\tag{11}
$$

Here, $\widetilde { \mathbf { k } } _ { j }$ and $\widetilde { \mathbf { c } } _ { v }$ denote a transformed key and its cluster centroid, respectively. The sum of these squared distances over all keys equals $\mathcal { I } _ { \mathrm { Q R K C } }$ in Eq. (10). Thus, standard Euclidean K-means in the transformed space directly minimizes this objective $\mathcal { I } _ { \mathrm { Q R K C } }$

Directly mapping each key to $\mathbf { Q } \mathbf { k } _ { j } / \sqrt { L }$ also realizes this distance in Eq. (11), but increases the feature dimension from d to $L .$ . To construct a transformation that avoids this increase in feature dimension while satisfying Eq. (11), we apply Cholesky decomposition and transform each key as

$$
\mathbf { G } _ { Q } = \mathbf { R } \mathbf { R } ^ { \top } , \qquad \widetilde { \mathbf { k } } _ { j } = \mathbf { R } ^ { \top } \mathbf { k } _ { j } \in \mathbb { R } ^ { d } .\tag{12}
$$

Here, R is a Cholesky factor when $\mathbf { G } _ { Q }$ is positive definite. Since the transformation is linear, $\widetilde { \mathbf { c } } _ { v } = \mathbf { R } ^ { \top } \mathbf { c } _ { v }$ is the centroid of the transformed keys in block v. This transformation satisfies the distance equality in Eq. (11). For numerical stability, we use a regularized metric, which yields the corresponding regularized clustering objective. The derivation and stabilization details are provided in Appendix D.2, and the QRKC pseudocode is provided in Appendix F.

Clustering and Reuse Strategy. We apply standard Euclidean K-means to cluster and reorder the queries, while using QRKC to cluster and reorder the keys. Prior studies (Hu et al. (2026); Luo et al. (2026); Xia et al. (2025)) have shown that sparse masks and clustering results change little across adjacent diffusion steps. To reduce online clustering and block-retrieval overhead, we reuse the sparse masks and clustering results across diffusion steps and recompute them every R steps.

Difference from SVG2 and SVOO. SVG2 (Yang et al. (2025a)) independently clusters queries and keys using Euclidean K-means, while SVOO (Luo et al. (2026)) couples the two partitions through iterative bidirectional co-clustering. Both methods then estimate block importance using query and key centroids. In contrast, PAMA retains every original query, normalizes its attention over key blocks separately, and then averages these weights within each query block to estimate block importance. QRKC derives a query-dependent metric from all current queries and transforms the keys before clustering, so that Euclidean K-means minimizes the average squared QK-score error introduced by key centroids. PARK therefore adopts an asymmetric retrieval design that preserves query-specific preferences while clustering keys using a theoretically derived metric.

## 6 EXPERIMENTS

## 6.1 EXPERIMENTAL SETUP

Models. We evaluate PARK on four open-source DiT models, including Wan2.1-T2V-1.3B, Wan2.1-T2V-14B, Wan2.2-T2V-14B (Wan et al. (2025)), and HunyuanVideo-T2V-13B (Kong et al.

Table 2: Quality and efficiency benchmark results for PARK and the baselines.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="7">Quality</td><td colspan="2">Efficiency</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>SubCons.↑</td><td>BackCons.↑</td><td>ImageQual.↑</td><td>AesQual.↑</td><td>Latency↓</td><td>Speedup↑</td></tr><tr><td rowspan="7">Wan2.2-14B</td><td>Dense</td><td></td><td></td><td></td><td>97.13%</td><td>97.22%</td><td>73.78%</td><td>65.56%</td><td>3792s</td><td>1.00×</td></tr><tr><td>SpargeAttn</td><td>18.32</td><td>0.714</td><td>0.295</td><td>97.10%</td><td>97.33%</td><td>73.73%</td><td>65.55%</td><td>2614s</td><td>1.45×</td></tr><tr><td>XAttention</td><td>17.31</td><td>0.640</td><td>0.353</td><td>96.89%</td><td>97.10%</td><td>73.87%</td><td>66.07%</td><td>2819s</td><td>1.34×</td></tr><tr><td>SVG</td><td>17.33</td><td>0.663</td><td>0.342</td><td>96.99%</td><td>96.96%</td><td>73.73%</td><td>65.85%</td><td>2668s</td><td>1.42×</td></tr><tr><td>SVG2</td><td>18.77</td><td>0.730</td><td>0.288</td><td>96.88%</td><td>97.11%</td><td>73.71%</td><td>65.20%</td><td>2672s</td><td>1.42×</td></tr><tr><td>SVOO</td><td>19.08</td><td>0.753</td><td>0.267</td><td>97.15%</td><td>97.34%</td><td>73.76%</td><td>65.31%</td><td>2722s</td><td>1.39×</td></tr><tr><td>PARK</td><td>19.19</td><td>0.754</td><td>0.265</td><td>97.11%</td><td>97.23%</td><td>73.73%</td><td>65.29%</td><td>2452s</td><td>1.55×</td></tr><tr><td rowspan="7">Wan2.1-1.3B</td><td>Dense</td><td></td><td></td><td></td><td>97.36%</td><td>96.99%</td><td>70.59%</td><td>61.23%</td><td>745s</td><td>1.00×</td></tr><tr><td>SpargeAttn</td><td>25.15</td><td>0.857</td><td>0.231</td><td>97.28%</td><td>96.95%</td><td>71.44%</td><td>61.58%</td><td>478s</td><td>1.57×</td></tr><tr><td>XAttention</td><td>23.43</td><td>0.786</td><td>0.296</td><td>96.68%</td><td>96.36%</td><td>68.71%</td><td>60.82%</td><td>495s</td><td>1.51×</td></tr><tr><td>SVG</td><td>22.39</td><td>0.754</td><td>0.336</td><td>96.73%</td><td>96.47%</td><td>72.16%</td><td>60.99%</td><td>506s</td><td>1.47×</td></tr><tr><td>SVG2</td><td>25.98</td><td>0.867</td><td>0.230</td><td>96.45%</td><td>96.03%</td><td>67.78%</td><td>58.90%</td><td>435s</td><td>1.71×</td></tr><tr><td>SVOO</td><td>27.36</td><td>0.901</td><td>0.187</td><td>97.32%</td><td>97.07%</td><td>70.94%</td><td>61.25%</td><td>472s</td><td>1.58×</td></tr><tr><td>PARK</td><td>27.52</td><td>0.908</td><td>0.177</td><td>97.22%</td><td>96.87%</td><td>70.48%</td><td>61.02%</td><td>423s</td><td>1.76×</td></tr><tr><td rowspan="7">Wan2.1-14B</td><td>Dense</td><td></td><td></td><td></td><td>97.95%</td><td>97.83%</td><td>72.46%</td><td>61.83%</td><td>3595s</td><td>1.00×</td></tr><tr><td>SpargeAttn</td><td>20.55</td><td>0.757</td><td>0.295</td><td>94.88%</td><td>95.49%</td><td>72.80%</td><td>61.73%</td><td>2732s</td><td>1.32×</td></tr><tr><td>XAttention</td><td>21.67</td><td>0.794</td><td>0.266</td><td>97.82%</td><td>97.73%</td><td>71.94%</td><td>62.13%</td><td>2710s</td><td>1.33×</td></tr><tr><td>SVG</td><td>20.87</td><td>0.785</td><td>0.277</td><td>97.74%</td><td>97.90%</td><td>71.99%</td><td>63.18%</td><td>2747s</td><td>1.31×</td></tr><tr><td>SVG2</td><td>22.98</td><td>0.848</td><td>0.219</td><td>97.82%</td><td>97.51%</td><td>72.24%</td><td>61.61%</td><td>2490s</td><td>1.44×</td></tr><tr><td>SVOO</td><td>23.66</td><td>0.871</td><td>0.188</td><td>97.89%</td><td>97.76%</td><td>72.42%</td><td>61.63%</td><td>2754s</td><td>1.31×</td></tr><tr><td>PARK</td><td>24.01</td><td>0.875</td><td>0.186</td><td>97.91%</td><td>97.74%</td><td>72.28%</td><td>62.03%</td><td>2354s</td><td>1.53×</td></tr><tr><td rowspan="7">HunyuanVideo</td><td>Dense</td><td></td><td></td><td></td><td>96.95%</td><td>96.98%</td><td>69.69%</td><td>55.58%</td><td>3058s</td><td>1.00×</td></tr><tr><td>SpargeAttn</td><td>29.32</td><td>0.923</td><td>0.187</td><td>96.83%</td><td>96.44%</td><td>69.90%</td><td>55.44%</td><td>2064s</td><td>1.48×</td></tr><tr><td>XAttention</td><td>29.08</td><td>0.909</td><td>0.198</td><td>96.80%</td><td>96.28%</td><td>69.44%</td><td>55.27%</td><td>2243s</td><td>1.36×</td></tr><tr><td>SVG</td><td>27.88</td><td>0.898</td><td>0.221</td><td>96.71%</td><td>96.38%</td><td>68.96%</td><td>55.32%</td><td>2066s</td><td>1.48×</td></tr><tr><td>SVG2</td><td>29.27</td><td>0.914</td><td>0.213</td><td>96.60%</td><td>96.05%</td><td>66.50%</td><td>54.05%</td><td>1978s</td><td>1.55×</td></tr><tr><td>SVOO</td><td>30.31</td><td>0.931</td><td>0.172</td><td>96.88%</td><td>96.39%</td><td>69.19%</td><td>55.05%</td><td>1975s</td><td>1.55×</td></tr><tr><td>PARK</td><td>30.69</td><td>0.936</td><td>0.166</td><td>96.87%</td><td>96.50%</td><td>69.41%</td><td>55.07%</td><td>1837s</td><td>1.66×</td></tr></table>

(2024)). All models generate videos at 720p resolution. In latent space, the Wan2.1 models process 21 frames, whereas HunyuanVideo processes 32 frames, with 3,600 tokens per frame in all cases.

Metrics. We use PSNR (OpenCV Contributors (2025)), SSIM (Wang et al. (2004)), and LPIPS (Zhang et al. (2018)) to measure the similarity between videos generated with sparse and dense attention. For video quality evaluation, we report Subject Consistency, Background Consistency, Image Quality, and Aesthetic Quality from VBench (Huang et al. (2024)). To assess efficiency, we report the end-to-end latency and the overall speedup.

Datasets. For all the text-to-video generation experiments, we use the original prompts from VBench (Huang et al. (2024)) for both HunyuanVideo and Wan.

Baselines. We compare PARK with state-of-the-art training-free sparse attention methods, including SpargeAttention (Zhang et al. (2025a)), XAttention (Xu et al. (2025)), SVG (Xi et al. (2025)), SVG2 (Yang et al. (2025a)), and SVOO (Luo et al. (2026)). All baselines are evaluated under the same hardware configuration and input settings as PARK.

Implementations. We set the number of query blocks to $N _ { Q } = 1 2 8$ for all models, and the number of key blocks to $N _ { K } = 1 , 0 2 4$ for HunyuanVideo and $N _ { K } = 5 1 2$ for Wan. All models generate videos with 50 denoising steps. We implement the fused PAMA kernel with Triton (Tillet et al. (2019)) and use a FlashInfer (Ye et al. (2025)) kernel with dynamic block sizes for sparse attention computation. All models use dense attention during the first 20% of the diffusion steps and a fixed Top-p threshold of $p = 0 . 9$ . We recompute sparse masks and clustering results every $R = 1 0$ diffusion steps. All experiments are conducted on a single NVIDIA A800 GPU.

## 6.2 QUALITY AND EFFICIENCY EVALUATION

As shown in Table 2, PARK consistently achieves the best similarity to dense attention across all four models, attaining the highest PSNR and SSIM and the lowest LPIPS among the compared sparse attention methods. It also remains close to dense attention on the VBench metrics, showing that

HunyuanVideo 720p, Text-to-Video

![](images/5d312fb8861feabaced10c57f5294a18e0fd58a2de87da15a243662d4b46cea2.jpg)  
Prompt：A cute happy Corgi playing in park, sunset, pan right

Wan2.1-14B 720p, Text-to-Video  
![](images/4e0b3554de9343a9b7b3569a4f61363e774ff09e7b065fd8a775051357d43ded.jpg)  
Prompt：A panda drinking coffee in a cafe in Paris, pixel art

Figure 3: Examples of videos generated by PARK and SVG2 on HunyuanVideo and Wan2.1-14B.

PARK preserves high generation quality. Compared with SVG2, PARK improves Image Quality by 2.91 and 2.70 percentage points on HunyuanVideo and Wan2.1-1.3B, respectively, with smaller improvements on Wan2.1-14B and Wan2.2-14B. As shown in Figure 3, PARK produces videos that more closely match dense attention than SVG2, with better-preserved fine-grained visual details.

While preserving this level of generation quality, PARK delivers 1.55×, 1.76×, 1.53×, and 1.66× end-to-end speedups over dense attention on Wan2.2-14B, Wan2.1-1.3B, Wan2.1-14B, and HunyuanVideo, respectively, achieving the highest speedup among the compared methods on all four models. Overall, PARK achieves the best quality-efficiency trade-off among the compared sparseattention methods.

Table 3: Ablation study of PAMA and QRKC on HunyuanVideo and Wan2.1-14B.
<table><tr><td>Model</td><td>Variant</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Latency ↓</td></tr><tr><td rowspan="3">HunyuanVideo</td><td>PARK</td><td>29.85</td><td>0.931</td><td>0.168</td><td>1827s</td></tr><tr><td>w/o PAMA</td><td>28.40</td><td>0.911</td><td>0.205</td><td>1824s</td></tr><tr><td>w/o QRKC</td><td>29.54</td><td>0.928</td><td>0.173</td><td>1826s</td></tr><tr><td rowspan="3">Wan2.1-14B</td><td>PARK</td><td>23.90</td><td>0.871</td><td>0.189</td><td>2359s</td></tr><tr><td>w/o PAMA</td><td>23.48</td><td>0.864</td><td>0.199</td><td>2353s</td></tr><tr><td>w/o QRKC</td><td>23.80</td><td>0.869</td><td>0.190</td><td>2354s</td></tr></table>

## 6.3 ABLATION STUDY

We evaluate the contributions of PAMA and QRKC on a fixed subset of VBench prompts by removing each module individually, while keeping the model configurations and inference settings identical to those used in the main evaluation. Removing PAMA replaces per-query estimation with query-centroid estimation, while removing QRKC replaces query-response key clustering with standard Euclidean K-means.

As shown in Table 3, removing PAMA consistently degrades PSNR, SSIM, and LPIPS, especially on HunyuanVideo, confirming the importance of preserving query-specific attention preferences during block retrieval. Removing QRKC also reduces similarity to dense attention on both models, demonstrating the benefit of grouping keys according to their QK-score similarity under the current queries. Meanwhile, the latency differences among the three variants remain within 0.3%, indicating that PAMA and QRKC improve generation quality from the query and key sides, respectively, with negligible overhead.

## 7 CONCLUSION

In this paper, we proposed PARK, a training-free sparse attention method designed for fast diffusionbased video generation. We identified query-side aggregation mismatch and key-side clustering metric mismatch in centroid-based block retrieval. To address them, PAMA preserves query-specific attention preferences, while QRKC clusters keys using a query-dependent metric. Experiments across four models show that PARK preserves high generation quality while achieving 1.53×–1.76× end-to-end speedups over dense attention.

## REFERENCES

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory- ´ efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

Jialin Guo, Haotian Tang, Shang Yang, Zhao Zhang, Zirui Liu, and Song Han. Block sparse attention. https://github.com/mit-han-lab/Block-Sparse-Attention, 2024.

Jie Hu, Zixiang Gao, Yutong He, and Kun Yuan. Dfsattn: Dynamic fine-grained sparse attention for efficient video generation. arXiv preprint arXiv:2605.23445, 2026.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Weinan Jia, Yuning Lu, Mengqi Huang, Hualiang Wang, Binyuan Huang, Nan Chen, Jidong Jiang, Zhendong Mao, et al. Moga: Mixture-of-groups attention for end-to-end long video generation. In International Conference on Learning Representations, volume 2026, pp. 115944–115964, 2026.

Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. Advances in Neural Information Processing Systems, 37:52481–52515, 2024.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Qirui Li, Guangcong Zheng, Qi Zhao, Jie Li, Bin Dong, Yiwu Yao, and Xi Li. Compact attention: Exploiting structured spatio-temporal sparsity for fast video generation. arXiv preprint arXiv:2508.12969, 2025a.

Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, et al. Radial attention: O(n log n) sparse attention with energy decay for long video generation. arXiv preprint arXiv:2506.19852, 2025b.

Yucheng Li, Huiqiang Jiang, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Amir H Abdi, Dongsheng Li, Jianfeng Gao, Yuqing Yang, et al. Mminference: Accelerating pre-filling for long-context visual language models via modality-aware permutation sparse attention. In International Conference on Machine Learning, pp. 34998–35020. PMLR, 2025c.

Jiayi Luo, Jiayu Chen, Jiankun Wang, Cong Wang, Hanxin Zhu, Qingyun Sun, Chen Gao, Zhibo Chen, and Jianxin Li. Attention sparsity is input-stable: Training-free sparse attention for video generation via offline sparsity profiling and online qk co-clustering. arXiv preprint arXiv:2603.18636, 2026.

OpenCV Contributors. Opencv: Open source computer vision library. https://github.com/ opencv/opencv, 2025. Accessed: 2025-02-26.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4172–4182. IEEE, 2023.

Wenhao Sun, Rong-Cheng Tu, Yifu Ding, Jingyi Liao, Zhao Jin, Shunyu Liu, and Dacheng Tao. Vorta: Efficient video diffusion via routing sparse attention. Advances in Neural Information Processing Systems, 38:7837–7863, 2026.

Haoyue Tan, Shengnan Wang, Yulin Qiao, Juncheng Zhang, Youhui Bai, Ping Gong, Zewen Jin, and Cheng Li. Adacluster: Adaptive query-key clustering for sparse attention in video generation. arXiv preprint arXiv:2604.18348, 2026a.

Xin Tan, Yuetao Chen, Yimin Jiang, Xing Chen, Kun Yan, Nan Duan, Yibo Zhu, Daxin Jiang, and Hong Xu. Dynamic sparsity in large-scale video dit training. In Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1, pp. 101–116, 2026b.

TileLang. Tilelang github repository. https://github.com/tile-ai/tilelang, 2024.

Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pp. 10–19, 2019.

Marcos Treviso, Antonio G ´ ois, Patrick Fernandes, Erick Fonseca, and Andr ´ e FT Martins. Predicting´ attention sparsity in transformers. In Proceedings of the sixth workshop on structured prediction for nlp, pp. 67–81, 2022.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600– 612, 2004.

Jianzong Wu, Liang Hou, Haotian Yang, Ye Tian, Pengfei Wan, Di ZHANG, and Yunhai Tong. Vmoba: Mixture-of-block attention for video diffusion models. In International Conference on Learning Representations, volume 2026, pp. 132534–132548, 2026.

Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.

Yifei Xia, Suhan Ling, Fangcheng Fu, Yujie Wang, Huixia Li, Xuefeng Xiao, and Bin Cui. Trainingfree and adaptive sparse attention for efficient long video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15982–15993, 2025.

Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, and Song Han. Xattention: Block sparse attention with antidiagonal scoring. In International Conference on Machine Learning, pp. 69819–69831. PMLR, 2025.

Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pp. 83048–83077, 2025b.

Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, et al. Flashinfer: Efficient and customizable attention engine for llm inference serving. Proceedings of Machine Learning and Systems, 7, 2025.

Chenlu Zhan, Wen Li, Chuyu Shen, Jun Zhang, Suhui Wu, and Hao Zhang. Bidirectional sparse attention for faster video diffusion training. arXiv preprint arXiv:2509.01085, 2025.

Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. In International Conference on Machine Learning, pp. 76397–76413. PMLR, 2025a.

Jintao Zhang, Kaiwen Zheng, Kai Jiang, Haoxu Wang, Ion Stoica, Joseph E Gonzalez, Jianfei Chen, and Jun Zhu. Turbodiffusion: Accelerating video diffusion models by 100-200 times. arXiv preprint arXiv:2512.16093, 2025b.

Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention. In International Conference on Machine Learning, pp. 74714–74731. PMLR, 2025c.

Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric Xing, and Hao Zhang. Faster video diffusion with trainable sparse attention. Advances in Neural Information Processing Systems, 38:152509–152534, 2026.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pp. 586–595. IEEE, 2018.

Tianchen Zhao, Ke Hong, Xinhao Yang, Xuefeng Xiao, Huixia Li, Feng Ling, Ruiqi Xie, SiQi Chen, Hongyu Zhu, Zhang Yichong, et al. Paroattention: Pattern-aware reordering for efficient sparse and quantized attention in visual generation models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

## A DIRECT EVALUATION OF RETRIEVAL MISMATCHES

We evaluate both mismatches on Wan2.1-14B and HunyuanVideo. We measure approximation errors against exact full-query, full-key references, with the results summarized in Figure 4.

![](images/fa60a0c057c470602db31eb05bad2607e71b7212d6bada2898d8d652f87c8811.jpg)

![](images/f3f590158c30bfebaa4a890171b2c3a0c3ea96d3ea13c2212f45eeba6a60e6c7.jpg)  
Figure 4: Direct evaluation of the two retrieval mismatches. Left: retaining original queries reduces attention-weight approximation error compared with query-centroid estimation. Right: QRKC reduces centered log-mass error compared with standard Euclidean clustering. Results correspond to Figure 1(c). Lower is better.

M1: Attention-weight approximation error. We fix the Euclidean query and key partitions and compare per-query and query-centroid estimation using the same key centroids. Unlike the fullkey analysis in Section 4.1, this experiment includes key-centroid approximation in both estimates. The per-query estimation normalizes attention separately for each query before averaging the block weights, whereas the query-centroid estimation averages queries before normalization. Both estimates are evaluated against the exact block importance $\hat { W } _ { u v }$ in Eq. (3), computed from the full queries and keys. For each layer–head pair, we measure

$$
E _ { \mathrm { T V } } ^ { m } = \sum _ { u = 1 } ^ { N _ { Q } } \frac { n _ { u } ^ { Q } } { L } \left( \frac { 1 } { 2 } \sum _ { v = 1 } ^ { N _ { K } } \left| \widehat { W } _ { u v } ^ { m } - W _ { u v } \right| \right) , \qquad m \in \{ \mathrm { p e r } , \mathrm { c e n t r o i d } \} .\tag{13}
$$

Here, $E _ { \mathrm { T V } } ^ { m }$ is the mean attention-weight TV error for one layer–head pair, with each query block weighted by its number of queries. The superscripts m = per and $m =$ centroid denote perquery and query-centroid estimation, respectively, and $\widehat { W } _ { u v } ^ { m }$ is the corresponding block-importance estimate. For each query block $\begin{array} { r } { u , \frac { 1 } { 2 } \sum _ { v = 1 } ^ { N _ { K } } | \widehat { W } _ { u v } ^ { m } - W _ { u v } | } \end{array}$ is the total variation (TV) distance between the estimated and exact block-weight distributions. We average the resulting errors equally across layer–head pairs. This metric compares each estimate with the exact reference, rather than measuring the difference between the two estimates.

As shown in Figure 4, retaining the original queries reduces mean TV from 0.1774 to 0.0927 on Wan2.1-14B and from 0.2092 to 0.1066 on HunyuanVideo. These correspond to relative reductions of 47.7% and 49.0%, respectively. These results show that using query centroids introduces substantial additional approximation error even when the key representation is unchanged.

Query-centroid error with full keys. To directly evaluate the full-key analysis in Section 4.1, we additionally retain all original keys and replace only the queries with their block centroids. For this configuration, the estimated block importance is $\widehat { W } _ { u v } = \phi _ { v } ( \overline { { \mathbf { q } } } _ { u } )$ , where $\phi _ { v }$ uses all original keys as defined in Eq. (17). We use the same Euclidean query and key partitions, exact reference $W _ { u v } .$ , and TV weighting and averaging as above. Table 4 includes the two key-centroid configurations from Figure 4 for comparison.

As shown in Table 4, query-centroid estimation incurs mean TV errors of 0.1335 and 0.1667 even without key compression, directly measuring the approximation error analyzed in Section 4.1. Retaining the original queries while compressing keys yields lower mean errors than compressing queries while retaining full keys, with relative reductions of 30.6% and 36.0% on Wan2.1-14B and HunyuanVideo, respectively.

Table 4: Mean attention-weight TV under different query and key representations, using the same Euclidean partitions. Full-query, full-key estimation is the exact reference and has zero error by definition. The two key-centroid rows reproduce the M1 results in Figure 4. Lower is better. Bold indicates the lowest error among the compressed representations.
<table><tr><td>Query</td><td>Key</td><td>Wan2.1-14B</td><td>HunyuanVideo</td></tr><tr><td>Full</td><td>Full</td><td>0.0000</td><td>0.0000</td></tr><tr><td>Centroid</td><td>Centroid</td><td>0.1774</td><td>0.2092</td></tr><tr><td>Centroid</td><td>Full</td><td>0.1335</td><td>0.1667</td></tr><tr><td>Full</td><td>Centroid</td><td>0.0927</td><td>0.1066</td></tr></table>

M2: Centered log-mass approximation error. We retain all original queries and compare Euclidean key clustering with QRKC. For each partition, we compute the exact log-mass $S _ { i v } ^ { \star }$ using the log-sum-exp expression in Eq. (7) and its key-centroid estimate $\widehat { S } _ { i v }$ using Eq. (8). Their difference gives the approximation error:

$$
\Delta _ { i v } ^ { K } = { S } _ { i v } ^ { \star } - \widehat { S } _ { i v } .\tag{14}
$$

For each query, we subtract the mean error across key blocks, since a common additive offset does not affect Softmax. We then measure the centered error using

$$
\delta _ { i v } = \Delta _ { i v } ^ { K } - \frac { 1 } { | \mathcal { V } | } \sum _ { r \in \mathcal { V } } \left( S _ { i r } ^ { \star } - \widehat { S } _ { i r } \right) , \qquad E _ { \mathrm { L M } } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \sqrt { \frac { 1 } { | \mathcal { V } | } \sum _ { v \in \mathcal { V } } \delta _ { i v } ^ { 2 } } .\tag{15}
$$

Here, $E _ { \mathrm { L M } }$ is the mean centered log-mass RMSE over all queries for one layer–head pair. V is the set of nonempty key blocks, and r is the index of these blocks in the sum. Thus, $\delta _ { i v }$ subtracts the mean log-mass error over all nonempty key blocks for query i from its error on block v. We compute $E _ { \mathrm { L M } }$ across key blocks separately for each query, average over the L queries, and then average equally across layer–head pairs. Each method uses the exact reference under its own key partition.

As shown in Figure 4, QRKC reduces mean centered log-mass RMSE from 0.4960 to 0.4109 on Wan2.1-14B and from 0.4949 to 0.4113 on HunyuanVideo, corresponding to relative reductions of 17.2% and 16.9%, respectively. These results show that clustering keys by QK-score similarity improves log-mass approximation compared with Euclidean clustering.

## B KEY-SPACE AND QK-SCORE APPROXIMATION ERRORS

Using the same configuration as Appendix A, we compare the errors caused by replacing each key with its cluster centroid in the original key space and in the scaled QK scores:

$$
D _ { \mathrm { k e y } } = \frac { 1 } { L _ { K } } \sum _ { j = 1 } ^ { L _ { K } } \lVert \mathbf { k } _ { j } - \mathbf { c } _ { v ( j ) } \rVert _ { 2 } ^ { 2 } , \qquad D _ { \operatorname { Q K } } = \frac { 1 } { L _ { Q } L _ { K } } \sum _ { i = 1 } ^ { L _ { Q } } \sum _ { j = 1 } ^ { L _ { K } } \left[ \frac { \mathbf { q } _ { i } ^ { \top } ( \mathbf { k } _ { j } - \mathbf { c } _ { v ( j ) } ) } { \sqrt { d } } \right] ^ { 2 } .\tag{16}
$$

Here, $v ( j )$ is the cluster containing key $j ,$ and $L _ { Q }$ and $L _ { K }$ are the numbers of queries and keys. Raw-key MSE $D _ { \mathrm { k e y } }$ averages squared Euclidean distances over keys, without dividing by the feature dimension. $D _ { \mathrm { Q K } }$ averages squared scaled-score errors over all query–key pairs. Equivalently, it averages the within-block score variance over queries, weighting key blocks by their sizes. It therefore corresponds to twice the similarly weighted leading error term $\cdot \epsilon _ { i v } ^ { K }$ , not the full log-mass error. Both metrics are averaged equally across layer–head pairs.

Table 5: Key-space and QK-score errors. Changes are relative to Euclidean clustering.
<table><tr><td>Model</td><td>Metric</td><td>Euclidean</td><td>QRKC</td><td>Change</td></tr><tr><td rowspan="2">Wan2.1-14B</td><td>Raw-key MSE</td><td>66.88</td><td>72.59</td><td>+8.5%</td></tr><tr><td>QK-score squared error</td><td>11.49</td><td>7.28</td><td>-36.6%</td></tr><tr><td rowspan="2">HunyuanVideo</td><td>Raw-key MSE</td><td>79.51</td><td>91.31</td><td>+14.8%</td></tr><tr><td>QK-score squared error</td><td>2.92</td><td>2.26</td><td>-22.8%</td></tr></table>

![](images/7847b6353aebb06c1cc6e3ff6ee19b9eaed43e16b98a48f549dd1ebc4324c133.jpg)  
Figure 5: Effect of QRKC on mean and P05 attention recall.

Table 5 shows that QRKC increases raw-key MSE while reducing QK-score error on both models. Thus, smaller Euclidean key-centroid distances do not necessarily yield more accurate QK scores, supporting the key-side clustering metric mismatch analyzed in Sec. 4.2.

## C EFFECT OF QUERY-RESPONSE KEY CLUSTERING

We evaluate QRKC by replacing Euclidean key clustering while keeping PAMA and the other settings unchanged. As shown in Figure 5, QRKC improves both mean and P05 attention recall on Wan2.1-14B and HunyuanVideo. These results support clustering keys according to their QK-score similarity under the current queries for more accurate block retrieval.

## D DETAILED DERIVATIONS FOR CENTROID-BASED BLOCK RETRIEVAL

This appendix provides the complete expansions underlying the analyses in Sections 4.1 and 4.2. We retain all original keys when analyzing query-centroid approximation, and retain all original queries when analyzing key-centroid approximation.

The query and key partitions are held unchanged within each comparison. All sums over blocks below concern nonempty blocks. The quantities $\Delta _ { u v } ^ { Q } , \epsilon _ { i v } ^ { K }$ , and $\Delta _ { i v } ^ { K }$ have the same meanings as in the main text: the exact block importance minus its query-centroid estimate, the leading key-side error term, and the full key-side log-mass error, respectively.

## D.1 DERIVATION OF THE QUERY-SIDE AGGREGATION MISMATCH

Replacing queries with their centroid can introduce error even when all keys are retained exactly. Following Section 4.1, we analyze this error without any key-side compression. The exact attention weight assigned to a key block is

$$
\phi _ { v } ( \mathbf { q } ) = \sum _ { j \in { \cal K } _ { v } } p _ { j } ( \mathbf { q } ) , \qquad p _ { j } ( \mathbf { q } ) = \frac { \exp ( \mathbf { q } ^ { \top } \mathbf { k } _ { j } / \sqrt { d } ) } { \sum _ { \ell = 1 } ^ { L } \exp ( \mathbf { q } ^ { \top } \mathbf { k } _ { \ell } / \sqrt { d } ) } .\tag{17}
$$

Here, $p _ { j } ( \mathbf { q } )$ is the exact token-level attention weight for key $\mathbf { k } _ { j } .$ , and $\phi _ { v } ( \mathbf { q } )$ sums these weights over key block $\textstyle { \mathcal { K } } _ { v }$ . The denominator includes all $\mathit { \check { L } }$ original keys, and ℓ indexes individual keys. The query and key vectors belong to $\mathbb { R } ^ { d }$ . Thus, averaging the exact block weights over the original queries gives the ground truth, whereas evaluating them at the query centroid gives an approximation:

$$
W _ { u v } = \frac { 1 } { n _ { u } ^ { Q } } \sum _ { i \in { \mathcal { Q } } _ { u } } \phi _ { v } ( \mathbf { q } _ { i } ) , \qquad \widehat { W } _ { u v } ^ { \mathrm { c e n t r o i d } } = \phi _ { v } ( \overline { { \mathbf { q } } } _ { u } ) .\tag{18}
$$

Here, $W _ { u v }$ is the exact block importance in Eq. (3), and $\widehat { W } _ { u v } ^ { \mathrm { c e n t r o i d } }$ is its query-centroid estimate. The set $\mathcal { Q } _ { u }$ contains $n _ { u } ^ { Q }$ queries, and $\overline { { { \bf { q } } } } _ { u }$ is their centroid. Both expressions use every original key. The only approximation in this comparison is replacing the queries with their centroid before Softmax normalization. Per-query computation itself is exact.

The query-block statistics used in the expansion are

$$
\overline { { \mathbf { q } } } _ { u } = \frac { 1 } { n _ { u } ^ { Q } } \sum _ { i \in \mathcal { Q } _ { u } } \mathbf { q } _ { i } , \qquad \epsilon _ { i } = \mathbf { q } _ { i } - \overline { { \mathbf { q } } } _ { u } .\tag{19}
$$

Here, $\overline { { \mathbf { q } } } _ { u }$ is the arithmetic mean of the queries and $\epsilon _ { i }$ is the vector deviation of query i from that mean. This vector is distinct from the scalar key-side error term $\epsilon _ { i v } ^ { K }$ . The deviations have zero mean. Indeed,

$$
\frac { 1 } { n _ { u } ^ { Q } } \sum _ { i \in \mathcal { Q } _ { u } } \mathbf { \epsilon } _ { i } = \frac { 1 } { n _ { u } ^ { Q } } \sum _ { i \in \mathcal { Q } _ { u } } \mathbf { q } _ { i } - \overline { { \mathbf { q } } } _ { u } = \overline { { \mathbf { q } } } _ { u } - \overline { { \mathbf { q } } } _ { u } = \mathbf { 0 } .\tag{20}
$$

Their second moment is

$$
\pmb { \Sigma } _ { Q , u } = \frac { 1 } { n _ { u } ^ { Q } } \sum _ { i \in \mathcal { Q } _ { u } } \pmb { \epsilon } _ { i } \pmb { \epsilon } _ { i } ^ { \top } \in \mathbb { R } ^ { d \times d } .\tag{21}
$$

Here, $\Sigma _ { Q , u }$ is the within-block query covariance, normalized by $n _ { u } ^ { Q }$ rather than the samplecovariance denominator $n _ { u } ^ { Q } - 1$

Taylor expansion of the exact block importance. For finite key vectors, $\phi _ { \tau }$ in Eq. (17) is smooth because its denominator is strictly positive. We expand it along the segments connecting the query centroid to each query in $\mathcal { Q } _ { u }$ . The derivatives at the centroid are

$$
\begin{array} { r } { \mathbf { g } _ { v } = \nabla _ { \mathbf { q } } \phi _ { v } ( \overline { { \mathbf { q } } } _ { u } ) , \qquad \mathbf { H } _ { v } = \nabla _ { \mathbf { q } } ^ { 2 } \phi _ { v } ( \overline { { \mathbf { q } } } _ { u } ) . } \end{array}\tag{22}
$$

Here, $\mathbf { g } _ { v }$ is the gradient and $\mathbf { H } _ { v } = \mathbf { H } _ { \phi _ { v } } ( \overline { { \mathbf { q } } } _ { u } )$ is exactly the Hessian appearing in Eq. (5). Their dependence on the query block u is omitted from the notation for simplicity. A second-order Taylor expansion of each $\phi _ { v } ( \mathbf { q } _ { i } )$ about the common point $\overline { { { \bf { q } } } } _ { u }$ gives

$$
\phi _ { v } ( \mathbf { q } _ { i } ) = \phi _ { v } ( \overline { { \mathbf { q } } } _ { u } ) + \mathbf { g } _ { v } ^ { \top } \boldsymbol { \epsilon } _ { i } + \frac { 1 } { 2 } \boldsymbol { \epsilon } _ { i } ^ { \top } \mathbf { H } _ { v } \boldsymbol { \epsilon } _ { i } + \mathcal { R } _ { i v } ^ { Q , ( 3 ) } .\tag{23}
$$

Here, $\mathcal { R } _ { i v } ^ { Q , ( 3 ) }$ is the higher-order remainder. Taylor’s theorem gives

$$
\left| \mathcal { R } _ { i v } ^ { Q , ( 3 ) } \right| \leq \frac { M _ { v } } { 6 } \| \epsilon _ { i } \| _ { 2 } ^ { 3 } .\tag{24}
$$

Here, $M _ { v }$ is a uniform upper bound on the operator norm of the third derivative $\nabla ^ { 3 } \phi _ { v }$ along all segments connecting $\overline { { \mathbf { q } } } _ { u }$ to the queries in $\mathcal { Q } _ { u }$ . The same bound $M _ { v }$ applies to all queries in $\mathcal { Q } _ { u } ^ { \phantom { \dagger } } ,$ , so it also bounds the remainder after averaging within the block.

Substituting Eq. (23) into the definition of the exact block importance $W _ { u v }$ yields

$$
\begin{array} { r l } & { W _ { u v } = \phi _ { v } ( \overline { { \mathbf { q } } } _ { u } ) + \mathbf { g } _ { v } ^ { \top } \left( \frac { 1 } { n _ { u } ^ { Q } } \displaystyle \sum _ { i \in \mathcal { Q } _ { u } } \epsilon _ { i } \right) } \\ & { \quad \quad \quad + \frac { 1 } { 2 n _ { u } ^ { Q } } \displaystyle \sum _ { i \in \mathcal { Q } _ { u } } \epsilon _ { i } ^ { \top } \mathbf { H } _ { v } \epsilon _ { i } + \frac { 1 } { n _ { u } ^ { Q } } \displaystyle \sum _ { i \in \mathcal { Q } _ { u } } \mathcal { R } _ { i v } ^ { Q , ( 3 ) } . } \end{array}\tag{25}
$$

Equation (20) proves explicitly that the gradient term vanishes:

$$
\mathbf { g } _ { v } ^ { \top } \left( \frac { 1 } { n _ { u } ^ { Q } } \sum _ { i \in \mathcal { Q } _ { u } } \epsilon _ { i } \right) = \mathbf { g } _ { v } ^ { \top } \mathbf { 0 } = 0 .\tag{26}
$$

For the quadratic term, the identity $\epsilon _ { i } ^ { \top } { \bf H } _ { v } \epsilon _ { i } = \mathrm { t r } ( { \bf H } _ { v } \epsilon _ { i } \epsilon _ { i } ^ { \top } )$ gives

$$
\frac { 1 } { n _ { u } ^ { Q } } \sum _ { i \in \mathcal { Q } _ { u } } \boldsymbol { \epsilon } _ { i } ^ { \top } \mathbf { H } _ { v } \boldsymbol { \epsilon } _ { i } = \mathrm { t r } ( \mathbf { H } _ { v } \boldsymbol { \Sigma } _ { Q , u } ) = \langle \mathbf { H } _ { v } , \boldsymbol { \Sigma } _ { Q , u } \rangle _ { F } .\tag{27}
$$

Here, $\operatorname { t r } ( \cdot )$ is the matrix trace and $\langle \mathbf { A } , \mathbf { B } \rangle _ { F } = \mathrm { t r } ( \mathbf { A } ^ { \top } \mathbf { B } )$ is the Frobenius inner product. Because a Hessian is symmetric, $\mathrm { t r } ( \mathbf { H } _ { v } \pmb { \Sigma } _ { Q , u } ) = \langle \dot { \mathbf { H } } _ { v } , \pmb { \Sigma } _ { Q , u } \rangle _ { F } ,$

Averaging the pointwise remainders gives

$$
\mathcal { R } _ { u v } ^ { Q } = \frac { 1 } { n _ { u } ^ { Q } } \sum _ { i \in \mathcal { Q } _ { u } } \mathcal { R } _ { i v } ^ { Q , ( 3 ) } .\tag{28}
$$

Here, $\mathcal { R } _ { u v } ^ { Q }$ is the higher-order remainder used in the main text. Combining Eqs. (25)–(27), the exact block importance expands as

$$
W _ { u v } = \phi _ { v } ( \overline { { \mathbf { q } } } _ { u } ) + \frac { 1 } { 2 } \langle \mathbf { H } _ { v } , \boldsymbol { \Sigma } _ { Q , u } \rangle _ { F } + \mathcal { R } _ { u v } ^ { Q } .\tag{29}
$$

In contrast, the query-centroid estimator evaluates $\phi _ { v }$ exactly at the expansion point and is simply

$$
\widehat { W } _ { u v } ^ { \mathrm { c e n t r o i d } } = \phi _ { v } ( \overline { { \mathbf { q } } } _ { u } ) .\tag{30}
$$

Subtracting Eq. (30) from Eq. (29) now gives

$$
\Delta _ { u v } ^ { Q } = W _ { u v } - \widehat { W } _ { u v } ^ { \mathrm { c e n t r o i d } } = \frac { 1 } { 2 } \langle \mathbf { H } _ { v } , \boldsymbol { \Sigma } _ { Q , u } \rangle _ { F } + \mathcal { R } _ { u v } ^ { Q } .\tag{31}
$$

Using the uniform bound in Eq. (24) and the triangle inequality, the averaged remainder satisfies

$$
| \mathcal { R } _ { u v } ^ { Q } | \leq \frac { M _ { v } } { 6 n _ { u } ^ { Q } } \sum _ { i \in \mathcal { Q } _ { u } } \| \epsilon _ { i } \| _ { 2 } ^ { 3 } .\tag{32}
$$

Thus, the first-order effect cancels exactly, giving the expression for $\Delta _ { u v } ^ { Q }$ in Eq. (5). Its leading term couples query covariance with the curvature of the exact normalized block weight. The signed error can be positive or negative: a query centroid can either underestimate or overestimate the importance of a particular key block. In this analysis, $\Delta _ { u v } ^ { Q }$ is the error relative to ground truth, not the difference between two approximate estimators, because no key-side approximation is used.

Dependence of the Hessian on all key blocks. To make the curvature term explicit without introducing key centroids, we write the exact token scores as

$$
z _ { j } ( \mathbf { q } ) = \mathbf { a } _ { j } ^ { \top } \mathbf { q } , \qquad \mathbf { a } _ { j } = \frac { \mathbf { k } _ { j } } { \sqrt { d } } .\tag{33}
$$

Here, $\mathbf { a } _ { j }$ is the scaled original key, and $z _ { j } ( \mathbf { q } )$ is its scaled QK score for query q. The corresponding attention-weighted statistics over all original keys are

$$
\overline { { \mathbf { a } } } ( \mathbf { q } ) = \sum _ { j = 1 } ^ { L } p _ { j } ( \mathbf { q } ) \mathbf { a } _ { j } , \qquad \mathbf { C } _ { a } ( \mathbf { q } ) = \sum _ { j = 1 } ^ { L } p _ { j } ( \mathbf { q } ) ( \mathbf { a } _ { j } - \overline { { \mathbf { a } } } ( \mathbf { q } ) ) ( \mathbf { a } _ { j } - \overline { { \mathbf { a } } } ( \mathbf { q } ) ) ^ { \top } .\tag{34}
$$

Here, $\overline { { \mathbf { a } } } ( \mathbf { q } )$ and ${ \mathbf { C } } _ { a } ( \mathbf { q } )$ are the attention-weighted mean and covariance of the scaled keys, respectively. Both are computed using the exact token attention weights $p _ { j } ( \mathbf { q } )$ defined in Eq. (17). Differentiating token-level Softmax gives

$$
\nabla _ { \mathbf { q } } p _ { j } ( \mathbf { q } ) = p _ { j } ( \mathbf { q } ) ( \mathbf { a } _ { j } - \overline { { \mathbf { a } } } ( \mathbf { q } ) ) , \qquad \nabla _ { \mathbf { q } } \overline { { \mathbf { a } } } ( \mathbf { q } ) = \mathbf { C } _ { a } ( \mathbf { q } ) .\tag{35}
$$

The second identity follows by substituting the first into the derivative of $\begin{array} { r } { \sum _ { j } p _ { j } ( \mathbf { q } ) \mathbf { a } _ { j } } \end{array}$ . Here, $\nabla _ { \mathbf { q } } \mathbf { \overline { { a } } }$ denotes its Jacobian. Differentiating the token weight once more yields

$$
\nabla _ { \mathbf { q } } ^ { 2 } p _ { j } ( \mathbf { q } ) = p _ { j } ( \mathbf { q } ) \left[ ( \mathbf { a } _ { j } - \overline { { \mathbf { a } } } ( \mathbf { q } ) ) ( \mathbf { a } _ { j } - \overline { { \mathbf { a } } } ( \mathbf { q } ) ) ^ { \top } - \mathbf { C } _ { a } ( \mathbf { q } ) \right] .\tag{36}
$$

Since $\phi _ { v }$ sums the token weights in key block $\textstyle { \mathcal { K } } _ { v }$ , its gradient and Hessian at the query centroid are

$$
\mathbf { g } _ { v } = \sum _ { j \in \mathcal { K } _ { v } } p _ { u j } ( \mathbf { a } _ { j } - \overline { { \mathbf { a } } } _ { u } ) ,
$$

$$
\mathbf { H } _ { v } = \sum _ { j \in { \mathcal { K } } _ { v } } p _ { u j } \left[ ( \mathbf { a } _ { j } - \overline { { \mathbf { a } } } _ { u } ) ( \mathbf { a } _ { j } - \overline { { \mathbf { a } } } _ { u } ) ^ { \top } - \mathbf { C } _ { a , u } \right] .\tag{37}
$$

Here, $p _ { u j } = p _ { j } ( \overline { { \mathbf { q } } } _ { u } )$ is the attention weight assigned to key j by the query centroid. The quantities $\overline { { \mathbf { a } } } _ { u } = \overline { { \mathbf { a } } } ( \overline { { \mathbf { q } } } _ { u } )$ and ${ \bf C } _ { a , u } = { \bf C } _ { a } ( \overline { { \bf q } } _ { u } )$ are the mean and covariance of the scaled keys, respectively, weighted by these attention weights. These expressions use individual keys, not key centroids. Both $\overline { { \mathbf { a } } } _ { u }$ and $\mathbf { C } _ { a , u }$ depend on all original keys and their Softmax weights, including keys outside block v. Consequently, the Hessian captures how the exact attention weight of block v changes as the query changes, accounting for competition with all other keys under Softmax normalization.

Implications for query-centroid estimation. Accurate query-centroid estimation requires a small $| \Delta _ { u v } ^ { \bar { Q } } |$ . Since the covariance matrix $\Sigma _ { Q , u }$ is positive semidefinite, the leading error term satisfies

$$
\left| \frac 1 2 \langle \mathbf { H } _ { v } , \pmb { \Sigma } _ { Q , u } \rangle _ { F } \right| \leq \frac { 1 } { 2 } \| \mathbf { H } _ { v } \| _ { \mathrm { o p } } \operatorname { t r } ( \pmb { \Sigma } _ { Q , u } ) , \qquad \mathrm { t r } ( \pmb { \Sigma } _ { Q , u } ) = \frac { 1 } { n _ { u } ^ { Q } } \sum _ { i \in \mathcal { Q } _ { u } } \| \epsilon _ { i } \| _ { 2 } ^ { 2 } .\tag{38}
$$

Here, $\mathbf { H } _ { v } = \mathbf { H } _ { \phi _ { v } } ( \overline { { \mathbf { q } } } _ { u } )$ is the Hessian at the query centroid, and $\| \mathbf { H } _ { v } \| _ { \mathrm { o p } }$ is its operator norm, equal to its largest absolute eigenvalue. The trace $\mathrm { ~ \ i r } ( \dot { \Sigma } _ { Q , u } )$ is the total within-block query variance, or the average squared Euclidean distance from the queries to their centroid. The inequality follows by applying $| \bar { \epsilon } _ { i } ^ { \top } \mathbf { H } _ { v } \epsilon _ { i } | \leq \| \mathbf { H } _ { v } \| _ { \mathrm { o p } } \| \epsilon _ { i } \| _ { 2 } ^ { 2 }$ to each query and averaging. Thus, a small product of the Hessian norm and the total query variance is sufficient for a small leading error. Combining this inequality with Eq. (32) also bounds the full approximation error:

$$
| \Delta _ { u v } ^ { Q } | \leq \frac { 1 } { 2 } \| \mathbf { H } _ { v } \| _ { \mathrm { o p } } \operatorname { t r } ( \Sigma _ { Q , u } ) + \frac { M _ { v } } { 6 n _ { u } ^ { Q } } \sum _ { i \in \mathcal { Q } _ { u } } \| \epsilon _ { i } \| _ { 2 } ^ { 3 } .\tag{39}
$$

Controlling both terms on the right-hand side is sufficient to make the full error small.

If all queries in a block are identical, every deviation $\epsilon _ { i }$ is zero and query-centroid estimation is exact. More generally, with bounded derivatives, both the second-order term and the remainder vanish as the query deviations approach zero. In practice, clustered queries may still have different attention preferences. Standard euclidean query clustering reduces but does not generally eliminate within-block query variance and does not account for the Hessian, so it does not by itself ensure that the product in Eq. (38) is sufficiently small. We next evaluate the approximation error empirically under practical clustering settings. Table 4 shows that query-centroid estimation with full keys incurs mean attention-weight TV errors of 0.1335 and 0.1667 on Wan2.1-14B and HunyuanVideo, respectively, exceeding the errors obtained by retaining original queries with key centroids.

Relation to estimation with key centroids. The analysis above retains all original keys to examine the error introduced by query centroids. PAMA additionally uses key centroids, which introduce a separate approximation error. Figure 1(c) shows that retaining original queries reduces attention-weight approximation error when both estimates use the same key centroids. Details of this experiment are provided in Appendix A. Table 1 compares different query and key compression settings and shows that retaining full queries while using key centroids achieves the highest recall among the compressed representations. Together, these empirical results support preserving original queries for practical block retrieval and motivate PAMA.

## D.2 DERIVATION OF THE QUERY-RESPONSE KEY METRIC

For each key block, we decompose its keys around their mean:

$$
{ \bf c } _ { v } = \frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in { \mathcal K } _ { v } } { \bf k } _ { j } , \qquad \delta _ { v j } = { \bf k } _ { j } - { \bf c } _ { v } .\tag{40}
$$

Here, $\mathbf { c } _ { v }$ is the centroid of the $n _ { v } ^ { K }$ keys in $\kappa _ { v } .$ , and $\delta _ { v j }$ is the deviation of key $j$ from this centroid. As on the query side, the deviations have zero mean:

$$
\frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in { \mathcal { K } } _ { v } } \delta _ { v j } = \frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in { \mathcal { K } } _ { v } } \mathbf { k } _ { j } - \mathbf { c } _ { v } = \mathbf { 0 } .\tag{41}
$$

The corresponding scaled score deviation is

$$
x _ { i v j } = \frac { { \bf q } _ { i } ^ { \top } \delta _ { v j } } { \sqrt { d } } .\tag{42}
$$

Here, $x _ { i v j }$ is the difference between the scaled QK score of query $\mathbf { q } _ { i }$ and key $\mathbf { k } _ { j }$ and the score obtained by replacing $\mathbf { k } _ { j }$ with its block centroid $\mathbf { c } _ { v }$ . Equation (41) immediately implies

$$
\frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in { \mathcal { K } } _ { v } } x _ { i v j } = \frac { \mathbf { q } _ { i } ^ { \top } } { \sqrt { d } } \left( \frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in { \mathcal { K } } _ { v } } \delta _ { v j } \right) = 0 .\tag{43}
$$

Exact decomposition. Using $\mathbf { k } _ { j } = \mathbf { c } _ { v } + \delta _ { v j }$ , the exact block log-mass decomposes as

$$
\begin{array} { r l r } {  { S _ { i v } ^ { \star } = \log \sum _ { j \in \mathcal { K } _ { v } } \exp ( \frac { \mathbf { q } _ { i } ^ { \top } \mathbf { k } _ { j } } { \sqrt { d } } ) } } \\ & { } & { = \log n _ { v } ^ { K } + \frac { \mathbf { q } _ { i } ^ { \top } \mathbf { c } _ { v } } { \sqrt { d } } + \underbrace { \log [ \frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in \mathcal { K } _ { v } } \exp ( x _ { i v j } ) ] } _ { \Delta _ { i v } ^ { K } } . } \end{array}\tag{44}
$$

Here, $S _ { i v } ^ { \star }$ is the exact log-mass in Eq. (7). The first two terms form $\begin{array} { r } { \widehat { S } _ { i v } = \log n _ { v } ^ { K } + \mathbf { q } _ { i } ^ { \top } \mathbf { c } _ { v } / \sqrt { d } , } \end{array}$ , the estimate in Eq. (8). Hence, $\Delta _ { i v } ^ { K } = S _ { i v } ^ { \star } - \widehat { S } _ { i \imath }$ is the full log-mass approximation error. Their relation to attention is

$$
P _ { i v } ^ { \star } = \frac { e ^ { S _ { i v } ^ { \star } } } { \sum _ { r } e ^ { S _ { i r } ^ { \star } } } , \qquad \widehat { P } _ { i v } = \frac { e ^ { \widehat { S } _ { i v } } } { \sum _ { r } e ^ { \widehat { S } _ { i r } } } .\tag{45}
$$

Here, $P _ { i v } ^ { \star }$ and $\widehat { P } _ { i v }$ are the exact and estimated attention weights for query $i ,$ respectively, and the sums run over all nonempty key blocks. These expressions show how key-centroid approximation affects attention weights through the block log-masses. The term log $n _ { v } ^ { K }$ accounts for the number of keys represented by each centroid and is necessary when key blocks have different sizes.

Second-order expansion. To derive the second-order term and bound the remainder, we compute

$$
m _ { 2 , i v } = \frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in \mathcal { K } _ { v } } x _ { i v j } ^ { 2 } , \qquad m _ { 3 , i v } = \frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in \mathcal { K } _ { v } } | x _ { i v j } | ^ { 3 } .\tag{46}
$$

Here, $m _ { 2 , i v }$ is the second moment and $m _ { 3 , i v }$ is the absolute third moment of the centered scaled scores. For each deviation, Taylor’s theorem gives $\exp ( x ) = 1 + x + x ^ { 2 } / 2 + r _ { 3 } ( x )$ . Here, $r _ { 3 } ( x )$ is the remainder after retaining the terms $1 , x ,$ and $\scriptstyle { \bar { x } } ^ { 2 } / 2$ . If $| x _ { i v j } | \le \eta$ within the block, then $| r _ { 3 } ( x _ { i v j } ) | \leq e ^ { \eta } | x _ { i v j } | ^ { 3 } / 6$ . Here, η is an upper bound on the absolute scaled-score deviations for query i within key block $\textstyle { \mathcal { K } } _ { v }$ , satisfying $\eta \geq \operatorname* { m a x } _ { j \in \mathcal { K } _ { v } } | x _ { i v j }$ . Averaging this expansion and using Eq. (43) therefore gives

$$
\frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in K _ { v } } \exp ( x _ { i v j } ) = 1 + \frac { 1 } { 2 } m _ { 2 , i v } + \rho _ { i v } ^ { \mathrm { e x p } } , \qquad | \rho _ { i v } ^ { \mathrm { e x p } } | \leq \frac { e ^ { \eta } } { 6 } m _ { 3 , i v } .\tag{47}
$$

Here, $\begin{array} { r } { \rho _ { i v } ^ { \mathrm { e x p } } = ( 1 / n _ { v } ^ { K } ) \sum _ { j \in \mathcal { K } _ { v } } r _ { 3 } ( x _ { i v j } ) } \end{array}$ is the average of the Taylor remainders within key block $\kappa _ { v }$ for query i. The key deviations also give

$$
\boldsymbol { \Sigma } _ { K , v } = \frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in \mathcal { K } _ { v } } \delta _ { v j } \delta _ { v j } ^ { \top } \in \mathbb { R } ^ { d \times d } .\tag{48}
$$

Here, $\Sigma _ { K , v }$ is the within-block key covariance, normalized by the number of keys. The second moment can now be written as a quadratic form:

$$
m _ { 2 , i v } = \frac { 1 } { d n _ { v } ^ { K } } \sum _ { j \in { \mathscr K } _ { v } } ( { \bf q } _ { i } ^ { \top } \pmb \delta _ { v j } ) ^ { 2 } = \frac { 1 } { d } { \bf q } _ { i } ^ { \top } \pmb \Sigma _ { K , v } { \bf q } _ { i } .\tag{49}
$$

Substituting Eq. (47) into the exact error expression in Eq. (44) gives

$$
\Delta _ { i v } ^ { K } = \log \left( 1 + \frac { 1 } { 2 } m _ { 2 , i v } + \rho _ { i v } ^ { \exp } \right) .\tag{50}
$$

For sufficiently small score deviations, $y = m _ { 2 , i v } / 2 + \rho _ { i v } ^ { \mathrm { e x p } }$ is close to zero. Using the Taylor expansion log $( 1 + y ) = y + \mathcal { O } ( y ^ { 2 } )$ and collecting the higher-order terms into $\mathcal { R } _ { i v } ^ { K }$ , we obtain

$$
\Delta _ { i v } ^ { K } = \frac { 1 } { 2 } m _ { 2 , i v } + \mathcal { R } _ { i v } ^ { K } = \frac { 1 } { 2 d } \mathbf { q } _ { i } ^ { \top } \boldsymbol { \Sigma } _ { K , v } \mathbf { q } _ { i } + \mathcal { R } _ { i v } ^ { K } .\tag{51}
$$

To match the main-text notation, the leading term and full error are

$$
\epsilon _ { i v } ^ { K } = \frac { m _ { 2 , i v } } { 2 } = \frac { 1 } { 2 d } \mathbf { q } _ { i } ^ { \top } \boldsymbol { \Sigma } _ { K , v } \mathbf { q } _ { i } , \qquad \Delta _ { i v } ^ { K } = \epsilon _ { i v } ^ { K } + \mathcal { R } _ { i v } ^ { K } .\tag{52}
$$

Here, $\epsilon _ { i v } ^ { K }$ is the leading error term, while $\mathcal { R } _ { i v } ^ { K }$ collects higher-order contributions from the exponential and logarithmic expansions. In particular, $\epsilon _ { i v } ^ { K }$ is one half of the within-block variance of scaled QK scores, or $1 / ( 2 d )$ times the variance of unscaled scores. It is not the full approximation error. For bounded, sufficiently small score deviations, there is a constant $C _ { \eta } > 0$ such that

$$
\vert \mathcal { R } _ { i v } ^ { K } \vert \leq C _ { \eta } \left( m _ { 3 , i v } + m _ { 2 , i v } ^ { 2 } \right) .\tag{53}
$$

The first term in the bound comes from the exponential remainder, while the second arises from the quadratic remainder of $\log ( 1 + y )$ . Independently of the small-deviation approximation, Jensen’s inequality gives the exact result

$$
\Delta _ { i v } ^ { K } = \log \left( \frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in \mathcal { K } _ { v } } e ^ { x _ { i v j } } \right) \geq \frac { 1 } { n _ { v } ^ { K } } \sum _ { j \in \mathcal { K } _ { v } } x _ { i v j } = 0 .\tag{54}
$$

Thus, the centroid estimate is always a lower bound on the exact block log-mass, and the leading error term $\epsilon _ { i v } ^ { K }$ is proportional to the variance of the keys after projection onto the current query.

Query-induced clustering metric. To construct a key-clustering objective that accounts for all current queries, we use

$$
\mathbf { G } _ { Q } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \mathbf { q } _ { i } \mathbf { q } _ { i } ^ { \top } = \frac { 1 } { L } \mathbf { Q } ^ { \top } \mathbf { Q } \in \mathbb { R } ^ { d \times d } .\tag{55}
$$

Here, $\mathbf { G } _ { Q }$ is the uncentered second-moment matrix of the L current queries, stored as rows of $\mathbf { Q }$ Unlike a centered covariance, it also retains their mean direction. Using $\mathbf { x } ^ { \top } \mathbf { A } \mathbf { x } = \mathrm { t r } ( \mathbf { A } \mathbf { x } \mathbf { x } ^ { \top } )$ and the cyclic property of the trace, we have

$$
\begin{array} { l } { { \displaystyle { \overline { { \epsilon } } } _ { v } ^ { K } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \frac { 1 } { 2 d } \mathbf { q } _ { i } ^ { \top } \boldsymbol { \Sigma } _ { K , v } \mathbf { q } _ { i } } } \\ { { \displaystyle ~ = \frac { 1 } { 2 d } \mathrm { t r } ( \mathbf { G } _ { Q } \boldsymbol { \Sigma } _ { K , v } ) } } \\ { { \displaystyle ~ = \frac { 1 } { 2 d n _ { v } ^ { K } } \sum _ { j \in \mathcal { K } _ { v } } \delta _ { v j } ^ { \top } \mathbf { G } _ { Q } \delta _ { v j } } . } \end{array}\tag{56}
$$

Here, $\bar { \epsilon } _ { v } ^ { K } = ( 1 / L ) \sum _ { i } \epsilon _ { i v } ^ { K }$ is the leading error term averaged over all current queries. Equation (56) connects $\bar { \epsilon } _ { v } ^ { K }$ to a query-dependent clustering objective. Multiplying by the block size and summing over key blocks gives the total token-weighted distortion

$$
\begin{array} { r l } & { \mathcal { I } _ { \mathrm { Q R K C } } = \displaystyle \sum _ { v = 1 } ^ { N _ { K } } \sum _ { j \in \mathcal { K } _ { v } } \delta _ { v j } ^ { \top } \mathbf { G } _ { Q } \delta _ { v j } = 2 d \sum _ { v = 1 } ^ { N _ { K } } n _ { v } ^ { K } \overline { { \epsilon } } _ { v } ^ { K } } \\ & { = \displaystyle \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \sum _ { v = 1 } ^ { N _ { K } } \sum _ { j \in \mathcal { K } _ { v } } \left( \mathbf { q } _ { i } ^ { \top } \delta _ { v j } \right) ^ { 2 } . } \end{array}\tag{57}
$$

The last equality follows by substituting the definition of $\mathbf { G } _ { Q }$ into each quadratic form. $\mathcal { I } _ { \mathrm { Q R K C } } / d$ measures the squared error introduced by replacing each key with its block centroid in the scaled attention logits. This error is summed over all keys and averaged over all queries. This objective weights each key block’s query-averaged leading error term $\bar { \epsilon } _ { v } ^ { K }$ by its number of keys $n _ { v } ^ { K }$

Ordinary Euclidean clustering instead minimizes

$$
\mathcal { I } _ { \mathrm { E u c } } = \sum _ { v = 1 } ^ { N _ { K } } \sum _ { j \in \mathcal { K } _ { v } } \delta _ { v j } ^ { \top } \mathbf { I } \delta _ { v j } ,\tag{58}
$$

where I is the identity matrix. Comparing Eqs. (57) and (58) shows that the mismatch is precisely the use of I instead of the query-induced metric $\mathbf { G } _ { Q }$

Euclidean realization. To optimize the query-dependent objective with a standard Euclidean Kmeans implementation, we need a key transformation whose squared Euclidean distances equal the quadratic forms defined by $\mathbf { G } _ { Q }$ . Directly mapping each key to $\mathbf { Q } \mathbf { k } _ { j } / \sqrt { L }$ satisfies this requirement, but produces an L-dimensional feature vector instead of a d-dimensional one. For the long video sequences considered here, $L \gg d ,$ so clustering these vectors would increase both storage and distance-computation costs. We instead factorize the d × d matrix $\mathbf { G } _ { Q }$ to construct a transformation that retains the feature dimension d while realizing the same distances. We first establish this property using a symmetric square root and then show how Cholesky decomposition provides the factor used in our implementation. The matrix ${ \bf G } _ { Q }$ is positive semidefinite because, for any $\mathbf { x } \in \mathbb { R } ^ { d }$

$$
\mathbf { x } ^ { \top } \mathbf { G } _ { Q } \mathbf { x } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } ( \mathbf { q } _ { i } ^ { \top } \mathbf { x } ) ^ { 2 } = \frac { 1 } { L } \| \mathbf { Q } \mathbf { x } \| _ { 2 } ^ { 2 } \geq 0 .\tag{59}
$$

It therefore admits the eigendecomposition and principal square root

$$
\mathbf { G } _ { Q } = \mathbf { U } \mathbf { A } \mathbf { U } ^ { \top } , \qquad \mathbf { G } _ { Q } ^ { 1 / 2 } = \mathbf { U } \mathbf { A } ^ { 1 / 2 } \mathbf { U } ^ { \top } ,\tag{60}
$$

where $\mathbf { U } \in \mathbb { R } ^ { d \times d }$ contains orthonormal eigenvectors and $\pmb { \Lambda } \in \mathbb { R } ^ { d \times d }$ is diagonal with nonnegative eigenvalues. The symmetric-square-root transformation is

$$
\widetilde { \mathbf { k } } _ { j } ^ { \mathrm { s q r t } } = \mathbf { G } _ { Q } ^ { 1 / 2 } \mathbf { k } _ { j } , \qquad \widetilde { \mathbf { c } } _ { v } ^ { \mathrm { s q r t } } = \mathbf { G } _ { Q } ^ { 1 / 2 } \mathbf { c } _ { v } .\tag{61}
$$

Here, the superscript sqrt identifies the square-root realization of the transformed key and centroid. For the residual $\delta _ { v j } = \mathbf { k } _ { j } - \mathbf { c } _ { v }$ , we have

$$
\lVert \widetilde { \mathbf { k } } _ { j } ^ { \mathrm { s q r t } } - \widetilde { \mathbf { c } } _ { v } ^ { \mathrm { s q r t } } \rVert _ { 2 } ^ { 2 } = \lVert \mathbf { G } _ { Q } ^ { 1 / 2 } \delta _ { v j } \rVert _ { 2 } ^ { 2 } = ( \mathbf { k } _ { j } - \mathbf { c } _ { v } ) ^ { \top } \mathbf { G } _ { Q } ( \mathbf { k } _ { j } - \mathbf { c } _ { v } ) .\tag{62}
$$

When $\mathbf { G } _ { Q }$ is positive definite, it also admits the Cholesky decomposition

$$
\mathbf { G } _ { Q } = \mathbf { R } \mathbf { R } ^ { \top } .\tag{63}
$$

The corresponding transformation is

$$
\begin{array} { r } { \widetilde { \mathbf { k } } _ { j } ^ { \mathrm { c h o l } } = \mathbf { R } ^ { \top } \mathbf { k } _ { j } , \qquad \widetilde { \mathbf { c } } _ { v } ^ { \mathrm { c h o l } } = \mathbf { R } ^ { \top } \mathbf { c } _ { v } . } \end{array}\tag{64}
$$

Here, the superscript chol identifies the Cholesky realization, whose transformed key is $\widetilde { \mathbf { k } } _ { j }$ in Eq. (12). Their squared Euclidean distance is

$$
\begin{array} { r l } & { \| \widetilde { \mathbf { k } } _ { j } ^ { \mathrm { c h o l } } - \widetilde { \mathbf { c } } _ { v } ^ { \mathrm { c h o l } } \| _ { 2 } ^ { 2 } = \| \mathbf { R } ^ { \top } \delta _ { v j } \| _ { 2 } ^ { 2 } } \\ & { \qquad = \delta _ { v j } ^ { \top } \mathbf { R } \mathbf { R } ^ { \top } \delta _ { v j } } \\ & { \qquad = \delta _ { v j } ^ { \top } \mathbf { G } _ { Q } \delta _ { v j } . } \end{array}\tag{65}
$$

In particular, the direct query-based transformation and the Cholesky transformation satisfy

$$
\left\| \frac { \mathbf { Q } \delta _ { v j } } { \sqrt { L } } \right\| _ { 2 } ^ { 2 } = \delta _ { v j } ^ { \top } \mathbf { G } _ { Q } \delta _ { v j } = \| \mathbf { R } ^ { \top } \delta _ { v j } \| _ { 2 } ^ { 2 } .\tag{66}
$$

The vector on the left has L components, whereas the transformed residual on the right has only d components. Thus, Cholesky decomposition lets us realize the query-dependent distance in Eq. (11) without constructing an L-dimensional representation for every key or modifying the distance computation in K-means. For each attention head, we compute R once before K-means and use it to transform all keys. For the same numbers of keys, clusters, and iterations, the K-means stage retains the computational complexity of clustering the original d-dimensional keys. The additional work is computing $\mathbf { G } _ { Q }$ , factorizing it, and transforming the keys. Figure 6 shows the time breakdown of these operations, whose total measured overhead remains below 2 ms across the evaluated settings. Equations (62) and (65) show that the symmetric-square-root and Cholesky transformations produce the same squared distance under $\mathbf { G } _ { Q }$ . Because a linear transformation also maps each original centroid to the centroid of the transformed keys, summing either distance over all clusters gives

$$
\sum _ { v = 1 } ^ { N _ { K } } \sum _ { j \in { \cal K } _ { v } } \| \widetilde { { \bf k } } _ { j } - \widetilde { { \bf c } } _ { v } \| _ { 2 } ^ { 2 } = \sum _ { v = 1 } ^ { N _ { K } } \sum _ { j \in { \cal K } _ { v } } \delta _ { v j } ^ { \top } { \bf G } _ { Q } \delta _ { v j } = \mathcal { I } _ { \mathrm { Q R K C } } .\tag{67}
$$

Therefore, Euclidean K-means after either transformation optimizes exactly the same QRKC objective in Eq. (57). The symmetric square root remains valid when $\mathbf { G } _ { Q }$ is rank deficient, whereas standard Cholesky decomposition requires a positive-definite matrix.

Numerically stable Cholesky factorization. To ensure a stable Cholesky factorization, we apply scale-aware shrinkage to the query-induced metric. The scale used for stabilization is

$$
s _ { Q } = \operatorname * { m a x } \left( \frac { 1 } { d } \operatorname { t r } ( \mathbf { G } _ { Q } ) , s _ { \operatorname { m i n } } \right) , s _ { \operatorname { m i n } } > 0 .\tag{68}
$$

Here, $s _ { Q }$ is the average diagonal value floored at $s _ { \operatorname* { m i n } } > 0$ , which is set to $1 0 ^ { - 1 2 }$ in the implementation. We construct the stabilized metric

$$
\overline { { \mathbf { G } } } _ { Q } = ( 1 - \alpha ) \mathbf { G } _ { Q } + \alpha s _ { Q } \mathbf { I } + \epsilon s _ { Q } \mathbf { I } ,\tag{69}
$$

where $\alpha \in [ 0 , 1 ]$ is the shrinkage coefficient and $\epsilon > 0$ is a small stability constant. The isotropic terms make $\overline { { \mathbf { G } } } _ { Q }$ positive definite while keeping their scale consistent with $\mathbf { G } _ { Q }$ . We then compute

$$
\overline { { { \bf G } } } _ { Q } = { \bf R } { \bf R } ^ { \top }\tag{70}
$$

and transform each key by $\widetilde { \mathbf { k } } _ { j } = \mathbf { R } ^ { \top } \mathbf { k } _ { j }$ . Following the same argument as Eq. (65), Euclidean K-means on these transformed keys optimizes the stabilized objective

$$
\overline { { \mathcal { T } } } _ { \mathrm { Q R K C } } = \sum _ { v = 1 } ^ { N _ { K } } \sum _ { j \in \mathcal { K } _ { v } } \delta _ { v j } ^ { \top } \overline { { \mathbf { G } } } _ { Q } \delta _ { v j } .\tag{71}
$$

The stabilized objective satisfies

$$
\begin{array} { r } { \overline { { \mathcal { I } } } _ { \mathrm { Q R K C } } = ( 1 - \alpha ) \mathcal { I } _ { \mathrm { Q R K C } } + ( \alpha + \epsilon ) s _ { Q } \mathcal { I } _ { \mathrm { E u c } } . } \end{array}\tag{72}
$$

Thus, stabilization adds an isotropic penalty, so the resulting objective differs from the unregularized objective. Even small stabilization terms can change cluster assignments when the original metric assigns very little weight to certain directions. The scalar ϵ here is a numerical stability parameter, distinct from the leading approximation error $\epsilon _ { i v } ^ { K }$

K-means optimizes the specified objective iteratively. The shrinkage term also provides a controlled interpolation between the query-induced metric and ordinary Euclidean distance: setting $\alpha = 0$ recovers the original metric up to the small stability term, whereas $\alpha = 1$ yields an isotropic metric up to a global scaling factor. With $\alpha = 0 . 0 5$ and the numerical stabilization described above, we observed no Cholesky decomposition failures in our experiments.

## E FUSED PAMA KERNEL

Algorithms 1 and 2 describe two fused implementations for one attention head. Programs for different attention heads and query blocks are executed in parallel. The queries are stored contiguously according to their query-block assignments, and $o _ { u }$ and $o _ { u + 1 }$ denote the beginning and end of query block $\mathcal { Q } _ { u } ,$ , respectively. The Full-K kernel retains the logits over all key centroids in registers and is used when $N _ { K } \leq 1 0 \dot { 2 } 4$ . For larger $N _ { K }$ , retaining the complete logit row creates excessive register pressure, so we use a two-pass kernel that processes the key centroids in tiles.

In both algorithms, i indexes a query in the current tile I, and v indexes a key block in the current key-centroid tile J or in $\{ 1 , \ldots , N _ { K } \}$ for Full-K. Tile entries $S _ { i \tau }$ and $P _ { i v }$ use these query and keyblock indices. All indexed assignments apply in parallel over the relevant indices in these ranges, not as serial loops.

Algorithm 1 Full-K fused PAMA kernel   
Require: Reordered queries $\textbf { Q } \in \ \mathbb { R } ^ { L \times d } ;$ ; query-block offsets $\left\{ o _ { u } \right\} _ { u = 1 } ^ { N _ { Q } + 1 }$ ; key centroids ${ \textbf { C } } \in$   
$\mathbb { R } ^ { N _ { K } \times d } ;$ key-block sizes $\{ n _ { v } ^ { K } \} _ { v = 1 } ^ { N _ { K } }$   
Ensure: Block importance $\widehat { \mathbf { W } } ^ { \mathrm { P A M A } } \in \mathbb { R } ^ { N _ { Q } \times N _ { K } }$   
1: for all query blocks $u = 1 , \ldots , N _ { Q }$ in parallel do   
2: $\mathbf { a }  \mathbf { 0 } \in \mathbb { R } ^ { N _ { K } }$ ▷ register-resident accumulator   
3: for each query tile $I \subseteq \{ o _ { u } , \dotsc , o _ { u + 1 } - 1 \}$ do   
4: $\mathbf { S } \gets \mathbf { 0 } \in \mathbb { R } ^ { | I | \times N _ { K } }$ ▷ register-resident logits   
5: for each feature tile $D _ { t } \subseteq \{ 1 , \dots , d \}$ do   
6: Load $\mathbf { Q } _ { I , D _ { t } }$ and $\mathbf { C } _ { : , D _ { t } }$   
7: $\mathbf { S } \gets \mathbf { S } + \mathbf { Q } _ { I , D _ { t } } \mathbf { C } _ { : , D _ { t } } ^ { \top }$   
8: end for   
9: $S _ { i v }  S _ { i v } / \sqrt { d } + \log n _ { v } ^ { K }$   
10: $m _ { i } \gets \operatorname* { m a x } _ { v } S _ { i v }$ ▷ row maximum   
11: $E _ { i v } \gets \exp ( S _ { i v } - m _ { i } )$ ▷ register resident   
12: $z _ { i } \gets \sum _ { v } E _ { i v }$ ▷ row normalization factor   
13: $P _ { i v }  \breve { E } _ { i v } / z _ { i }$ ▷ row-wise Softmax   
14: $\begin{array} { r } { \underset { - } { a } _ { v }  { a } _ { v } + \sum _ { i \in I } P _ { i v } } \end{array}$ ▷ sum over queries   
15: end for   
16: $\widehat { \mathbf { W } } _ { u , : } ^ { \mathrm { P A M A } } \gets \mathbf { a } / ( o _ { u + 1 } - o _ { u } )$ ▷ the only HBM write   
17: end for   
18: return $\widehat { \mathbf { W } } ^ { \mathrm { P A M A } }$

The intermediate tensors and attention-weight accumulator in Algorithm 1 remain in on-chip registers, without materializing the full $L \times N _ { K } ^ { - }$ logit or probability matrix in HBM. It computes every QK-centroid product once, but its register footprint grows with $N _ { K }$

Algorithm 2 Two-pass fused PAMA kernel   
Require: Reordered queries $\textbf { Q } \in \ \mathbb { R } ^ { L \times d } ;$ query-block offsets $\left\{ o _ { u } \right\} _ { u = 1 } ^ { N _ { Q } + 1 }$ ; key centroids ${ \textbf { C } } \in$   
$\mathbb { R } ^ { N _ { K } \times d } ;$ key-block sizes $\{ n _ { v } ^ { K } \} _ { v = 1 } ^ { N _ { K } }$   
Ensure: Block importance $\widehat { \mathbf { W } } ^ { \mathrm { P A M A } } \in \mathbb { R } ^ { N _ { Q } \times N _ { K } }$   
1: Initialize $\widehat { \mathbf { W } } ^ { \mathrm { P A M A } }  \mathbf { 0 }$ in HBM   
2: for all query blocks $u = 1 , \ldots , N _ { Q }$ in parallel do   
3: for each query tile $I \subseteq \{ o _ { u } , \dotsc , o _ { u + 1 } - 1 \}$ do   
4: Load $\mathbf { Q } _ { I , : }$   
5: $m _ { i } \gets - \infty , z _ { i } \gets 0$ ▷ Softmax states in registers   
6: for each key-centroid tile $J \subseteq \{ 1 , \dots , N _ { K } \}$ do ▷ Pass 1   
7: $\mathbf { S } \gets \mathbf { Q } _ { I , : } \mathbf { C } _ { J , : } ^ { \top } / \sqrt { d }$   
8: $S _ { i v }  S _ { i v } + \log n _ { v } ^ { K }$   
9: $m _ { i } ^ { \prime } \gets \operatorname* { m a x } ( m _ { i } ,$ max $\cdot _ { v \in J } S _ { i v } )$   
10: $z _ { i } \gets z _ { i } \exp ( m _ { i } - m _ { i } ^ { \prime } ) + \sum _ { v \in J } \exp ( S _ { i v } - m _ { i } ^ { \prime } )$   
11: $m _ { i } \gets m _ { i } ^ { \prime }$   
12: end for   
13: $\ell _ { i } \gets m _ { i } + \log z _ { i }$ ▷ row-wise LogSumExp in registers   
14: for each key-centroid tile $J \subseteq \{ 1 , \dots , N _ { K } \}$ do ▷ Pass 2   
15: Recompute $\mathbf { S } \gets \mathbf { Q } _ { I , : } \mathbf { C } _ { J , : } ^ { \top } / \sqrt { d }$   
16: $S _ { i v }  S _ { i v } + \log n _ { v } ^ { K }$   
17: $P _ { i v }  \exp ( S _ { i v } - \bar { \ell } _ { i } )$ ▷ in registers   
18: $\begin{array} { r } { \widehat { W } _ { u v } ^ { \mathrm { P A M A } } + = \sum _ { i \in I } P _ { i v } / ( o _ { u + 1 } - o _ { u } ) } \end{array}$ ▷ HBM accumulation   
19: end for   
20: end for   
21: end for   
22: return $\widehat { \mathbf { W } } ^ { \mathrm { P A M A } }$

The first pass computes row-wise LogSumExp through online updates over key-centroid tiles. The second recomputes the logits, normalizes each tile, and accumulates block importance. This approximately doubles QK-centroid GEMM work but keeps register usage independent of $N _ { K }$ for fixed tile sizes. Logits, probabilities, and LogSumExp values remain on chip rather than in HBM, allowing larger ${ \cal N } _ { K } ^ { - }$ than Full-K.

Table 6 compares PyTorch with the Two-pass and Full-K kernels in Triton (Tillet et al. (2019)) and TileLang (TileLang). Both benchmarks use batch size 1, $N _ { Q } = 2 5 6 , N _ { K } = 1 , 0 2 4$ , and $d = 1 2 8 .$ with $( \bar { H , } L ) = ( 2 \bar { 4 , } 1 1 5 , 2 0 0 )$ for HunyuanVideo and (40, 75,600) for Wan2.1-14B.

Additional full-model timing shows that Triton Full-K completes PAMA block-importance estimation across all layers in less than 1 s on both models. Four such block-importance recomputations during video generation account for less than 0.21% of the end-to-end latencies in Table 2.

Table 6: PAMA runtime and additional GPU memory. Speedup, memory reduction (multiplicative factor), and memory saving are relative to PyTorch.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Implementation</td><td colspan="2">Runtime</td><td colspan="3">Additional GPU Memory</td></tr><tr><td>Latency ↓</td><td>Speedup ↑</td><td>Peak↓</td><td>Reduction ↑</td><td>Saving ↑</td></tr><tr><td rowspan="5">HunyuanVideo</td><td>PyTorch</td><td>87.95ms</td><td>1.00×</td><td>21.09 GB</td><td>1.00×</td><td>0.00%</td></tr><tr><td>Triton (Two-pass)</td><td>28.12ms</td><td>3.13×</td><td>24.12 MB</td><td>895.60×</td><td>99.89%</td></tr><tr><td>Triton (Full-K)</td><td>15.82ms</td><td>5.56×</td><td>24.17 MB</td><td>893.86×</td><td>99.89%</td></tr><tr><td>TileLang (Two-pass)</td><td>21.35ms</td><td>4.12×</td><td>24.12 MB</td><td>895.60×</td><td>99.89%</td></tr><tr><td>TileLang (Full-K)</td><td>21.08ms</td><td>4.17×</td><td>24.17 MB</td><td>893.86×</td><td>99.89%</td></tr><tr><td rowspan="5">Wan2.1-14B</td><td>PyTorch</td><td>92.81ms</td><td>1.00×</td><td>23.07 GB</td><td>1.00×</td><td>0.00%</td></tr><tr><td>Triton (Two-pass)</td><td>31.35ms</td><td>2.96×</td><td>40.20 MB</td><td>587.71×</td><td>99.83%</td></tr><tr><td>Triton (Full-K)</td><td>17.72ms</td><td>5.24×</td><td>40.27 MB</td><td>586.57×</td><td>99.83%</td></tr><tr><td>TileLang (Two-pass)</td><td>23.82ms</td><td>3.90×</td><td>40.20 MB</td><td>587.71×</td><td>99.83%</td></tr><tr><td>TileLang (Full-K)</td><td>23.22ms</td><td>4.00×</td><td>40.27 MB</td><td>586.57×</td><td>99.83%</td></tr></table>

## F QRKC PSEUDOCODE

Algorithm 3 summarizes QRKC for one attention head. The transformed keys are used only to determine the cluster assignments. The labels are then applied to the original keys and values, and PAMA uses centroids computed in the original key space.

Algorithm 3 Query-Response Key Clustering (QRKC)   
Require: Queries $\overline { { \mathbf { Q } \in \mathbb { R } ^ { L \times d } } } ;$ keys $\overline { { \mathbf { K } \in \mathbb { R } ^ { L \times d } } } ;$ values $\mathbf { V } \in \mathbb { R } ^ { L \times d } ;$ number of key clusters $N _ { K } ;$   
K-means iterations $T ;$ shrinkage $\alpha ;$ stability constant ϵ   
Ensure: Key labels ${ \bf z ; }$ permutation $\tau _ { K } ;$ reordered keys $\mathbf { K } ^ { \prime }$ and values $\mathbf { V } ^ { \prime } ;$ original-space centroids   
$\{ \mathbf { c } _ { v } \} _ { v = 1 } ^ { N _ { K } } \dot { , }$ cluster sizes $\{ n _ { v } ^ { K } \} _ { v = 1 } ^ { N _ { K } }$   
1: $\mathbf { G } _ { Q } \gets \mathbf { \bar { Q } } ^ { \top } \mathbf { Q } / L$   
2: $\dot { \mathbf { G } _ { Q } } \gets ( \mathbf { G } _ { Q } + \mathbf { G } _ { Q } ^ { \top } ) / 2$ $\triangleright$ enforce numerical symmetry   
3: $s _ { Q } \gets \operatorname* { m a x } \bigl ( \mathrm { t r } ( \mathbf { G } _ { Q } ) / d , 1 0 ^ { - 1 2 } \bigr )$   
4: $\overline { { \mathbf { G } } } _ { Q }  ( 1 - \alpha ) \mathbf { G } _ { Q } + \alpha s _ { Q } \mathbf { I } + \epsilon s _ { Q } \mathbf { I }$   
5: $\mathbf { R }  \mathrm { C h o l e s k y } ( \overline { { \mathbf { G } } } _ { Q } )$ ▷ $\overline { { { \bf G } } } _ { Q } = { \bf R } { \bf R } ^ { \top }$   
6: Ke ← KR ▷ query-conditioned key features   
7: Initialize transformed centroids $\{ \widetilde { \mu } _ { v } \} _ { v = 1 } ^ { N _ { K } }$ from $\widetilde { \bf K }$   
8: for $t = 1 , \dots , T$ do   
9: for all keys $j = 1 , \dots , L$ in parallel do   
10: $z _ { j }  \arg \operatorname* { m i n } _ { v \in \{ 1 , . . . , N _ { K } \} } \| \widetilde { \mathbf { k } } _ { j } - \widetilde { \pmb { \mu } } _ { v } \| _ { 2 } ^ { 2 }$   
11: end for   
12: for all clusters $v = 1 , \ldots , N _ { K }$ in parallel do   
13: $\widetilde { \pmb { \mu } } _ { v }  \mathrm { m e a n } \{ \widetilde { \mathbf { k } } _ { j } : z _ { j } = v \}$   
14: end for   
15: end for   
16: $\pi _ { K } $ argsort(z)   
17: $\dot { \mathbf { K } ^ { \prime } }  \mathbf { K } [ \pi _ { K } , : ] , \stackrel { } { \mathbf { V } ^ { \prime } }  \mathbf { V } [ \pi _ { K } , : ]$   
18: for all clusters $v = 1 , \ldots ,  { { N _ { K } } }$ in parallel do   
19: $\mathcal { K } _ { v }  \{ j : z _ { j } = v \} , n _ { v } ^ { K }  \{ \dot { \mathcal { K } } _ { v } \vert $   
20: $\begin{array} { r } { \mathbf { c } _ { v } \gets ( n _ { v } ^ { K } ) ^ { - 1 } \sum _ { j \in \mathcal { K } _ { v } } \mathbf { k } _ { j } } \end{array}$ ▷ centroid in the original key space   
21: end for   
22: return $\mathbf { z } , \pi _ { K } , \mathbf { K } ^ { \prime } , \mathbf { V } ^ { \prime } , \{ \mathbf { c } _ { v } , n _ { v } ^ { K } \} _ { v = 1 } ^ { N _ { K } }$

Because $\widetilde { \mathbf { K } } = \mathbf { K } \mathbf { R }$ , Euclidean K-means in the transformed space minimizes the stabilized QRKC objective in Eq. (71). The transformation does not change the feature dimension, and no transformed values are required. Thus, K-means retains the same computational complexity for the same token count, cluster count, and number of iterations.

![](images/9aac02951a79a116c69e647d3140b2d39f178e9e8174e2b5bec0c73bb1f752e3.jpg)  
Figure 6: Time breakdown of QRKC overhead excluding K-means clustering. The total measured overhead remains below 2 ms. We use the same experimental settings as in Table 6.

## G SENSITIVITY TO QUERY AND KEY CENTROID COUNTS

We analyze the sensitivity of PARK to the numbers of query and key centroids, denoted by $N _ { Q }$ and $N _ { K }$ , respectively.

Table 7: Sensitivity to the numbers of query and key centroids on HunyuanVideo.
<table><tr><td> $N _ { Q }$ </td><td> $N _ { K }$ </td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Latency ↓</td><td>Speedup ↑</td></tr><tr><td>64</td><td>1,024</td><td>29.38</td><td>0.926</td><td>0.175</td><td>1817s</td><td>1.68×</td></tr><tr><td>128</td><td>1,024</td><td>29.85</td><td>0.931</td><td>0.168</td><td>1827s</td><td>1.67×</td></tr><tr><td>192</td><td>1,024</td><td>29.94</td><td>0.933</td><td>0.167</td><td>1841s</td><td>1.66×</td></tr><tr><td>256</td><td>1,024</td><td>30.13</td><td>0.934</td><td>0.164</td><td>1855s</td><td>1.64×</td></tr><tr><td>320</td><td>1,024</td><td>30.13</td><td>0.935</td><td>0.162</td><td>1875s</td><td>1.63×</td></tr><tr><td>128</td><td>512</td><td>29.56</td><td>0.928</td><td>0.174</td><td>1830s</td><td>1.67×</td></tr><tr><td>128</td><td>768</td><td>29.74</td><td>0.931</td><td>0.170</td><td>1834s</td><td>1.66×</td></tr></table>

Table 8: Sensitivity to the numbers of query and key centroids on Wan2.1-1.3B.
<table><tr><td> $N _ { Q }$ </td><td> $N _ { K }$ </td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Latency ↓</td><td>Speedup ↑</td></tr><tr><td>64</td><td>512</td><td>27.30</td><td>0.903</td><td>0.183</td><td>410s</td><td>1.82×</td></tr><tr><td>128</td><td>512</td><td>27.52</td><td>0.908</td><td>0.177</td><td>423s</td><td>1.76×</td></tr><tr><td>192</td><td>512</td><td>27.66</td><td>0.911</td><td>0.174</td><td>427s</td><td>1.74×</td></tr><tr><td>256</td><td>512</td><td>27.71</td><td>0.911</td><td>0.173</td><td>430s</td><td>1.73×</td></tr><tr><td>320</td><td>512</td><td>27.79</td><td>0.913</td><td>0.172</td><td>440s</td><td>1.69×</td></tr><tr><td>128</td><td>768</td><td>27.67</td><td>0.910</td><td>0.175</td><td>416s</td><td>1.79×</td></tr><tr><td>128</td><td>1024</td><td>27.72</td><td>0.912</td><td>0.173</td><td>420s</td><td>1.77×</td></tr></table>

## H CLUSTERING AND SPARSE-MASK REUSE

We evaluate the effect of the recomputation schedule on HunyuanVideo with 50 denoising steps. Let T denote the set of denoising step indices at which clustering results and sparse masks are computed or recomputed. Both are reused during subsequent sparse-attention steps until the next recomputation. Table 9 compares four schedules in terms of generation quality and inference efficiency.

Table 9: Effect of clustering and sparse-mask recomputation schedules on HunyuanVideo with 50 denoising steps. Speedup is measured relative to dense attention.
<table><tr><td>T</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>Latency ↓</td><td>Speedup ↑</td></tr><tr><td>{10}</td><td>29.55</td><td>0.928</td><td>0.174</td><td>1829s</td><td>1.67×</td></tr><tr><td>{10,30}</td><td>29.67</td><td>0.930</td><td>0.171</td><td>1819s</td><td>1.68×</td></tr><tr><td>{10, 20, 30}</td><td>29.84</td><td>0.932</td><td>0.168</td><td>1819s</td><td>1.68×</td></tr><tr><td>{10, 20, 30, 40}</td><td>29.85</td><td>0.931</td><td>0.168</td><td>1827s</td><td>1.67×</td></tr></table>

As shown in Table 9, increasing the number of recomputations generally improves similarity to dense attention, while three and four recomputations yield nearly identical results. All four schedules achieve similar end-to-end latency. These results support reusing clustering results and sparse masks across diffusion steps, with occasional recomputation to preserve generation quality.

## I EFFICIENCY-QUALITY TRADE-OFF STUDY

We study the efficiency-quality trade-off of PARK under different sparsity settings on a subset of VBench prompts using HunyuanVideo, Wan2.1-1.3B, and Wan2.1-14B. As shown in Figure 7, although similarity to dense attention decreases as latency is reduced, PARK maintains high generation quality across the tested sparsity settings, as reflected by its relatively stable VBench-Q scores.

![](images/265167b7d998d069d9a16896278f7cd8dfb839969ee23251c8be557e8e0e466d.jpg)  
Figure 7: Efficiency-quality trade-off of PARK.

## J QUERY AND KEY COMPRESSION UNDER TOP-K RETRIEVAL

We extend the query/key compression comparison in Table 1 to Top-k retrieval with a fixed retrieval ratio $\rho .$ For each query block, we retrieve the top ρ fraction of key blocks in descending order of estimated importance. We set $\rho = 0 . 1$ , corresponding to approximately 10% of the key blocks. Table 10 reports attention recall under clustered and original token orderings, with full-query and full-key estimation serving as the reference.

Table 10: Top-k block-retrieval attention recall at a fixed retrieval ratio $( \rho = 0 . 1 )$ under different query/key representations. Mean, P05, and Worst denote the mean, fifth percentile, and minimum attention recall across all attention heads and layers, respectively. All values are percentages. Higher is better.
<table><tr><td rowspan="2">Ordering</td><td rowspan="2">Query</td><td rowspan="2">Key</td><td colspan="3">HunyuanVideo</td><td colspan="3">Wan 2.1-14B</td></tr><tr><td>Mean</td><td>P05</td><td>Worst</td><td>Mean</td><td>P05</td><td>Worst</td></tr><tr><td rowspan="4">Clustered</td><td>Full</td><td>Full</td><td>77.05</td><td>27.41</td><td>18.97</td><td>73.48</td><td>32.81</td><td>19.66</td></tr><tr><td>Centroid Centroid</td><td></td><td>75.37</td><td>27.27</td><td>18.96</td><td>72.35</td><td>32.78</td><td>19.66</td></tr><tr><td>Centroid Full</td><td></td><td>76.14</td><td>27.38</td><td>18.96</td><td>72.88</td><td>32.80</td><td>19.66</td></tr><tr><td>Full</td><td>Centroid</td><td>76.50</td><td>27.29</td><td>18.96</td><td>73.12</td><td>32.79</td><td>19.66</td></tr><tr><td rowspan="4">Original</td><td>Full</td><td>Full</td><td>61.90</td><td>18.93</td><td>12.73</td><td>56.49</td><td>21.17</td><td>10.51</td></tr><tr><td>Centroid</td><td>Centroid</td><td>56.78</td><td>17.56</td><td>08.46</td><td>54.78</td><td>20.52</td><td>10.51</td></tr><tr><td>Centroid</td><td>Full</td><td>56.91</td><td>18.91</td><td>12.72</td><td>53.17</td><td>21.13</td><td>10.51</td></tr><tr><td>Full</td><td>Centroid</td><td>56.94</td><td>17.49</td><td>08.59</td><td>54.84</td><td>20.66</td><td>10.51</td></tr></table>

## K LIMITATIONS

This work focuses on accelerating video generation with diffusion models. Our evaluation is limited to this setting and does not establish the applicability of PARK to text-based large language models or other multimodal tasks. These settings may exhibit different attention patterns and inference workflows, which may require adaptations to PARK. Further research is needed to determine whether PARK can provide similar efficiency benefits in these settings while preserving output quality.

## L VISUALIZATION OF THE GENERATED VIDEOS

We provide visual comparisons between PARK and Dense Attention on HunyuanVideo, Wan2.1- 14B, Wan2.1-1.3B, and Wan2.2-14B. The results in Figures 8, 9, 10, and 11 show that PARK preserves high pixel-level fidelity and achieves generation quality similar to that of dense attention. More video samples are provided in the supplementary materials.

![](images/836504c7ae7680a716aa3ea296be5a63924eadb6f052cb407cc0b6b5c3b2cc46.jpg)  
Figure 8: Comparison of Dense Attention and PARK for text-to-video generation with Hunyuan-Video.

![](images/a2edf00c7007f94e9d819c27eb845c37e06b316a6537af1c2606b5670b5ea888.jpg)  
Figure 9: Comparison of Dense Attention and PARK for text-to-video generation with Wan2.1- 14B.

![](images/a56b9d3eceb6a61ce943716c1fda1f65b4f17815de97647dd76db17ecdcd5d0b.jpg)  
Figure 10: Comparison of Dense Attention and PARK for text-to-video generation with Wan2.1- 1.3B.

![](images/16020fc8717022b4203c2d2c4630e1b3682900f11d62687a9001cd6af2e65d28.jpg)  
Figure 11: Comparison of Dense Attention and PARK for text-to-video generation with Wan2.2- 14B.