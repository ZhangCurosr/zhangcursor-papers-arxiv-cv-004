# SV-TAD: Native Sparse Convolutions for Eficient Temporal Action Detection

Ricardo Pizarro , <sup>1</sup>, Roberto Valle <sup>2</sup>, Jos´e M. Buenaposada <sup>3</sup>, Luis M. Bergasa <sup>1</sup>, and Luis Baumela <sup>2</sup>

<sup>1</sup>Universidad de Alcal´a, Alcal´a de Henares, Spain <sup>2</sup>Universidad Polit´ecnica de Madrid, Madrid, Spain <sup>3</sup>Universidad Rey Juan Carlos, M´ostoles, Spain

ricardo.pizarroc@edu.uah.es

## Abstract

To adapt billion-parameter Vision Transformers for long-video understanding, recent methods freeze the backbone and train lightweight convolutional modules. While efective for parameter-eficient training, existing adapters do not reduce inference-time computation, leaving scalability with respect to video length largely unaddressed. Token selection can reduce attention cost by pruning redundant tokens, but it breaks the spatial grid structure required by convolutional adapters. This forces an expensive dense reconstruction, nullifying much of the potential speedup. We address this by introducing native sparse 2D convolutions, a primitive that allows these adapters, for the first time, to operate directly and eficiently on dynamically pruned token sets. We integrate this primitive into SV-TAD, an adapter framework for temporal action detection, reducing VideoMAEv2-L computation by up to 64% and achieving 2.2× faster inference, while maintaining state-of-the-art accuracy on THUMOS-14 and ActivityNet-1.3. When scaled to InternVideoNext-L, our approach surpasses the previous state of the art at roughly half its computational cost. Moreover, the sparse formulation naturally supports auxiliary task tokens, which improves fine-grained assembly detection on ATTACH. Code and trained models are available at https: //github.com/pcr-upm/eccv26\_tad.

keywords: Sparse Convolutions, Token Selection, Vision Transformers, Temporal Action Detection, Eficient Inference, Eficient Training

## 1 Introduction

Recent advances in large-scale video foundation models have significantly improved video understanding [31]. To adapt these billion-parameter Vision Transformers to temporal action detection (TAD) tasks without prohibitive finetuning cost, adapter-based approaches insert lightweight convolutional modules into a frozen backbone, allowing parameter-eficient specialization [21, 24]. However, despite their parameter eficiency, these adapters still operate on dense token grids and their computational cost scales proportionally with the full sequence length of the video. As video resolution and temporal duration increase, this dense processing becomes a major computational bottleneck. Moreover, existing adapter-based approaches primarily improve training eficiency and do not reduce the inference-time computational cost of the backbone, leaving scalability with respect to video length largely unaddressed.

In parallel, token selection [18, 4] has emerged as an efective mechanism to reduce the quadratic cost of self-attention by dynamically pruning uninformative tokens. Unlike architectural modifications, such as eficient attention variants [28] or distillation [3], token selection preserves the original model’s representations. This makes it uniquely attractive for adapter-based fine-tuning, where any reduction in token count could directly translate into wall-clock savings.

However, while sparsification substantially reduces attention complexity, it disrupts the regular spatial structure required by convolutional adapters, producing a fundamental representational gap. This afects convolutions, feature pyramid networks, and any operation that assumes fixed spatial neighborhoods. Existing convolutional adapters reshape tokens into spatial feature maps and apply grid-based convolutions, requiring dense reconstruction when tokens are removed. This reconstruction step substantially diminishes the computational gains of sparsification, so current adapter frameworks cannot fully benefit from token-level sparsity. Additionally, existing 3D sparse convolutional engines [10, 7] cannot be repurposed here, as their format requirements and host-side hash-table operations are architecturally incompatible with the dynamic sparsity of token selection (Sec. 2).

To address this, we introduce SparseConv2D, a native sparse 2D convolution that bridges token selection and convolutions without dense reconstruction. Our key insight is that the sparse token set already contains everything a convolution needs: a convolution does not require the full spatial grid, only the adjacency between each output position and its $K ^ { 2 }$ neighbors; encoding this adjacency explicitly over the sparse set makes the grid unnecessary. We precompute a neighbor index table that maps each retained token to its neighbors’ positions in the sparse sequence, then execute the convolution through custom CUDA kernels that gather features directly via this table. The regular 2D patch grid further allows the table to be built in $\mathcal { O } ( N _ { k e p t } )$ ), with $N _ { k e p t }$ being the number of tokens selected, translating token reduction into proportional speedups in both memory and computation.

We instantiate this primitive in SparseViT-TAD (SV-TAD), a temporal action detection framework where sparse convolutional adapters are inserted into each Transformer block of a frozen ViT backbone [31, 29], while token selection modules progressively prune sequences in strategic layers. As shown in Fig. 1, SV-TAD achieves better eficiency-accuracy trade-ofs than existing adapter methods [21, 11]. With ViT-L, it achieves 64% fewer TFLOPs and 2.2x faster throughput. Our contributions are:

![](images/bb74b20a794714fd88b0d409b295b7a7c9653ce3da91207febd653c73cbaa53f.jpg)  
Figure 1: TFLOPs vs. accuracy on THUMOS-14. SV-TAD (red squares) matches or exceeds AdaTAD [21] (blue circles) at substantially lower compute across three backbone scales: 43% fewer TFLOPs with VMAEv2-B, 64% with VMAEv2-L, and 51% when comparing our InternVideoNext-L against AdaTAD’s VMAEv2-G.

• We introduce native sparse 2D convolutions for Vision Transformers, a computational primitive that processes dynamically sparse token sequences through precomputed neighbor index tables and custom CUDA kernels, without dense grid reconstruction.

• We design sparse adapter architectures that integrate this primitive into parameter-eficient fine-tuning, enabling convolutional adapters to operate on token-selected sequences for the first time.

• We demonstrate 64% reduction of TFLOPs, 2.2x inference speedup, and 1.7x training acceleration over the state-of-the-art on THUMOS-14, while maintaining mAP. We validate on ActivityNet-1.3 with consistent gains.

• We show that our sparse framework is flexible enough to support multitask learning: in ATTACH, auxiliary landmark supervision improves over the baseline by +2.41%, demonstrating that secondary tasks can be incorporated with minimal overhead.

• We release our CUDA kernels as a standalone library for sparse token processing in PyTorch.

## 2 Related Work

Parameter-Eficient Fine-Tuning (PEFT). Large-scale ViTs, with hundreds of millions to billions of parameters, cannot be fully fine-tuned for TAD, where each input spans hundreds of frames. Adapter-based methods [14] address this by keeping the backbone weights fixed and learning only small modules inserted between layers. AIM [35] proposes spatial and temporal adapters for video recognition, while ST-Adapter [24] employs 3D convolutions to capture spatio-temporal patterns. LoSA [11] introduces long-short-range adapters and scales to VMAEv2-G, yet underperforms relative to the capacity of its billionparameter backbone. Other methods such as ViT-TAD [34] adapt a short-term ViT-B for clip-level processing. While this keeps per-snippet FLOPs low, ViT-TAD relies on full fine-tuning of all backbone parameters, fundamentally limiting scaling to larger backbones. AdaTAD [21] applies temporal-informative 1D convolutional adapters, surpassing both LoSA and ViT-TAD while training only ∼2% of parameters. A common limitation across these methods is that their convolutional modules always operate on the complete spatial grid, regardless of token informativeness. Recent image-domain work [16, 36] combines PEFT with token selection but requires explicit attention-matrix computation, incompatible with FlashAttention [9] and infeasible for TAD sequences exceeding 70k tokens.

Token Selection in Vision Transformers. The quadratic complexity of self-atten-tion in the number of tokens N has motivated numerous strategies to alleviate this cost by reducing the number of tokens that traverse the network. EViT [18] ranks patches according to their class-token attention and discards the lowest-scoring ones, while ToMe [4] progressively merges nearby tokens using bipartite matching, eliminating the need for additional training. DynamicViT [26] instead trains a lightweight decision network that predicts per-token keep/drop masks. In video, TokenLearner [27] and STTS [30] extend pruning and merging to the spatio-temporal setting. A shared limitation is that all of these approaches target only the self-attention bottleneck. None specifies how the irregular token sets produced should interact with local operators, such as convolutions. Merging methods carry an additional constraint, as computing the N×N pairwise similarity matrix is at odds with sub-quadratic attention kernels like FlashAttention [9], limiting applicability to long sequences.

Sparse Convolutions. Processing spatially sparse data with convolutions has a long history in 3D vision. Submanifold sparse convolutions [10] and Minkowski-Engine [7] represent occupied voxels in hash tables, dispatching computation only where data exists, an approach widely adopted for LiDAR point clouds [32]. However, these engines are poorly suited to the dynamic token selection in ViTs: their COO-style sparse-tensor format requires a costly round-trip to the packed dense tensors used by a ViT at every block, and pruning positions invalidates cached kernel maps, forcing the coordinate manager to rebuild all $K ^ { d }$ maps via hash-table probes per active site with no reuse across layers whose sparsity patterns difer. This hash-based construction is heavier than the problem requires, since ViT coordinates lie on a known regular 2D patch grid, rendering existing 3D sparse engines impractical for adapter-based video models. Separately, sparse-transformer research [6] sparsifies attention patterns rather than convolutional adapters. To our knowledge, native sparse 2D convolutions for the dynamic, layer-varying sparsity of token selection have not been explored before.

Our Proposal. Each research line reviewed above addresses one dimension of the eficiency challenge, but leaves the others unresolved. PEFT adapters [14, 35, 24, 21] always operate on the full spatial grid, so cost scales with total tokens regardless of informativeness. Token selection [18, 4, 26] reduces the count but does not specify how irregular token sets should interact with local operators, limiting savings to attention-only models. Existing sparse engines [10, 7, 32] are incompatible with ViT tensor formats and unsuitable for dynamic, layer-varying sparsity. SV-TAD bridges all three: sparse convolutional adapters provide PEFT-style eficiency, token selection reduces the sequence, and our native sparse 2D convolution enables both to work together, translating token reduction into proportional savings without dense reconstruction or external sparse-tensor engines.

## 3 Method

We present SV-TAD, a framework that bridges token selection and convolutional adapters for TAD. Central to our approach is SparseConv2D, a native sparse 2D convolution (Sec. 3.2), this computational primitive processes dynamically sparse token sequences without dense grid reconstruction. We integrate this primitive into a sparse adapter architecture (Sec. 3.4) and show that the resulting framework is flexible enough to support auxiliary task supervision (Sec. 3.3). Fig. 2 provides an overview.

## 3.1 Token Selection

We employ attention-based token selection [18, 25] to progressively prune uninformative tokens during the forward pass. Crucially, unlike prior work that discards spatial information after pruning, we preserve the original grid indices of kept tokens, this positional information is essential for our sparse convolution.

At designated Transformer blocks, we compute importance scores from aggregated attention weights between auxiliary tokens and visual tokens. In the standard setting, only the class token serves as the auxiliary token. When more than one auxiliary task token is present (Sec. 3.3), importance scores incorporate attention from all K tokens with learnable weights $\mathbf { w } \in \mathbb { R } ^ { K }$

![](images/0b94a797b67112ffb724651779bbd72cd9542e4403321e629aa321d873f86912.jpg)  
Figure 2: SV-TAD architecture. Input frames and auxiliary tokens (X) pass through a frozen pretrained ViT backbone with interleaved token selection (TS-ON) blocks. The first three blocks use standard 2D convolutions, with the rest using SparseConv2D. The expanded view shows our modified ViT block: frozen attention computes importance scores, token selection prunes uninformative patches, and the trainable sparse adapter operates directly on the sparse sequence. Timing annotations show per-block latency decreasing from 60ms to 7.6ms as the sequence progressively shrinks.

$$
\mathbf { s } = \frac { 1 } { n _ { h } } \sum _ { h = 1 } ^ { n _ { h } } \sum _ { k = 1 } ^ { K } w _ { k } \cdot \mathbf { A } _ { k } ^ { ( h ) } , \quad \mathcal { T } _ { k e p t } = \mathrm { T S } \left( \mathbf { s } , \ \lfloor k _ { r } \cdot N \rfloor \right)\tag{1}
$$

where TS is the token selection algorithm, N is the number of visual tokens, $\mathbf { A } _ { k } ^ { ( h ) } \in \mathbb { R } ^ { N }$ denotes the attention weights from the k-th auxiliary token to all visual tokens at head $h , n _ { h }$ is the number of attention heads, and $k _ { r }$ is the keep rate. This formulation allows task-specific tokens $\mathrm { ( e . g . , }$ , auxiliary landmarks) to influence which $N _ { k e p t }$ patches are retained, biasing selection toward regions relevant for the auxiliary task. When K=1 (class token only), this reduces to the standard EViT formulation [18]. The tokens kept $\mathbf { X } _ { v i s } = \mathbf { X } _ { v i s } [ \mathcal { T } _ { k e p t } ]$ are passed to subsequent layers, while $\mathcal { T } _ { k e p t }$ encodes their original positions $( y , x )$ in the spatial grid $H \times W$

Compatibility with FlashAttention. A critical advantage of our token selection is that it requires only the K attention rows from auxiliary tokens to visual tokens, not the full $N \times N$ attention matrix. We extract the required K rows (where $K \ll N )$ as a byproduct of the standard attention computation. This allows our entire pipeline, backbone attention, token selection, and sparse adapters to operate with FlashAttention enabled, which is essential for scaling to video sequences.

Following prior work on token selection [18, 26, 25], the selection is applied at evenly-spaced layers (blocks 4, 8, 12, 16 for ViT-L), progressively reducing the sequence length. Early layers process all tokens to build robust representations before pruning begins.

![](images/27d8934cf00a032e359f40705a275665dfb2ed3d24bad3eb4b6a1fa6eb76bfe5.jpg)  
Figure 3: Kernel Timing comparison on THUMOS-14. DenseConv2D (blue) keeps latency nearly constant across diferent keep rates, while our SparseConv2D (orange) scales linearly with $N _ { k e p t }$

## 3.2 Native Sparse 2D Convolution

The straightforward way to combine token selection with convolutional adapters is dense reconstruction. It consists of scattering the $N _ { k e p t }$ sparse tokens back to their original grid positions $\mathbf { X } _ { d e n s e } \in \mathbb { R } ^ { T \times H \times W \times D }$ , filling pruned locations with zeros, applying standard convolution, and then gathering outputs at kept positions. While functionally correct, this approach has fundamental eficiency limitations: (1) memory scales with O(T HW) regardless of sparsity, (2) convolution computes over all THW positions including empty padding, and (3) empirically, speedup plateaus as reconstruction overhead dominates, pruning more tokens yields no additional benefits. The general incompatibility and reconstruction algorithm are explained in detail in Sec. G of the supplementary material.

We propose a native sparse 2D convolution that operates directly on sparse token sequences, $\mathbf { X _ { v i s } } \in \mathbf { \bar { \mathbb { R } } } ^ { N _ { k e p t } \times D / 4 }$ . The key insight is that convolution is inherently local, e.g., a 3×3 kernel only needs 9 neighbors per output, not the entire grid. We exploit this locality through an indirection table that maps sparse tokens to their sparse neighbors, enabling computation without dense reconstruction. For simplicity in the following sections, we use a $3 \times 3$ kernel. Fig. 3 shows that our sparse kernel achieves parity with dense convolution at ∼55% keep rate; below this threshold, sparse is strictly faster, reaching 8× speedup at 10% keep rate.

The data structure enabling our sparse convolution is a neighbor index table $\mathbf { N } \in \mathbb { Z } ^ { N _ { k e p t } \times 9 }$ that maps each kept token to its spatial neighbors’ positions in the sparse token list. For kept token k at grid position $( y _ { k } , x _ { k } )$ and kernel ofsets

$\delta _ { x } \in \{ - 1 , 0 , 1 \}$ and $\delta _ { y } \in \{ - 1 , 0 , 1 \}$ :

$$
\mathbf { N } [ k , j ] = { \left\{ \begin{array} { l l } { \operatorname { s p a r s e - i d x } ( y _ { k } + \delta _ { y , j } , x _ { k } + \delta _ { x , j } ) } & { { \mathrm { i f ~ n e i g h b o r ~ i s ~ k e p t } } } \\ { - 1 } & { { \mathrm { i f ~ p r u n e d ~ o r ~ o u t - o f - b o u n d s } } } \end{array} \right. }\tag{2}
$$

where $j \in \{ 1 , \ldots , 9 \}$ indexes kernel positions and −1 indicates zero-padding.

We construct this table eficiently in two steps, exploiting the regular 2D grid structure:

1. Build a lookup map. Create a dense array $\mathbf { M } \in \mathbb { Z } ^ { H \times W }$ initialized to −1. For each kept token k at grid position $( y _ { k } , x _ { k } )$ , set $\mathbf { M } [ y _ { k } , x _ { k } ] = k$ This maps grid coordinates to sparse indices.

2. Fill the neighbor table. For each kept token $k ,$ look up its 8 neighbors via $\mathbf { N } [ k , j ] = \mathbf { M } [ y _ { k } + \delta _ { y , j } , x _ { k } + \delta _ { x , j } ]$ . If the neighbor was pruned, M returns −1, correctly indicating zero-padding.

Both steps are $\mathcal { O } ( N _ { k e p t } )$ , computed on CPU with negligible overhead (less than ∼3ms) relative to the GPU-bound adapter forward pass, and M is reused across all adapter layers until the next token selection.

## 3.2.1 Custom CUDA Kernels.

We implement forward and backward passes as custom CUDA kernels. The forward pass computes:

$$
\mathbf { y } _ { k } = \sum _ { j = 1 } ^ { 9 } \mathbf { W } _ { j } \cdot \mathbf { x } [ \mathbf { N } [ k , j ] ] + \mathbf { b }\tag{3}
$$

where $\mathbf { W } _ { j } \in \mathbb { R } ^ { C _ { o u t } \times C _ { i n } }$ is the weight slice for kernel position $j ,$ and $\mathbf { x } [ \mathbf { N } [ k , j ] ]$ gathers neighbor features (zeros if $\mathbf { N } [ k , j ] = - 1 )$ . We provide two strategies, automatically selected based on $C _ { o u t } \colon$ an implicit GEMM path via cuBLAS for large channels $( C _ { o u t } \ge 2 5 6 )$ that fully utilizes Tensor Cores, and custom tiled kernels for smaller channels that fuse gather and GEMM into a single launch. Full kernel design details are in Sec. A of the supplementary.

## 3.2.2 Complexity Analysis.

Our sparse convolution requires $\mathcal { O } ( N _ { k e p t } \cdot K ^ { 2 } \cdot C _ { i n } \cdot C _ { o u t } )$ operations versus $\mathcal { O } ( T H W \cdot K ^ { 2 } \cdot C _ { i n } \cdot C _ { o u t } )$ for dense convolution. With $N _ { k e p t } = k _ { r } \cdot T H W$ compute scales linearly with keep rate $k _ { r }$ . Memory is decoupled from total sequence length through chunked processing, O(chunk size · $\left( { { C _ { i n } } + { C _ { o u t } } } \right) )$ . The full model applies token selection at S stages $( \mathrm { e . g . , } S = 4$ at blocks 4, 8, 12, 16 for ViT-L), with cumulative savings compounding across layers. At $k _ { r } { = } 0 . 6$ per stage, the efective rate after two stages drops to 0.36 and after four to 0.13, placing the majority of layers well below the per-kernel crossover point where sparse convolution outperforms dense (Fig. 3).

The default keep rate used for every dataset and backbone is listed in Sec. C of the supplementary.

Table 1: Efect of keep rate $k _ { r }$ on THUMOS-14 (InternVideoNext-L backbone).
<table><tr><td>Keep Rate</td><td>TFLOPs</td><td>vid/s</td><td>mAP@0.5</td><td>Avg. mAP</td></tr><tr><td>0.7</td><td>87.05</td><td>0.42</td><td>78.22</td><td>74.13</td></tr><tr><td>0.6</td><td>74.09</td><td>0.50</td><td>77.22</td><td>73.64</td></tr><tr><td>0.5</td><td>60.13</td><td>0.60</td><td>76.40</td><td>73.00</td></tr></table>

Keep-Rate Schedule. Table 1 sweeps the per-stage keep rate $k _ { r }$ on THUMOS-14 with the InternVideoNext-L backbone. For this model $k _ { r } { = } 0 . 7$ gives the best trade-of; lowering it trades accuracy for throughput gracefully, $\mathrm { e . g . }$ $k _ { r } { = } 0 . 5$ reaches 0.60 videos $\mathrm { \Omega } _ { \mathrm { { / } \mathrm { { s } } } }$ (up from 0.42) at only −1.13% avg. mAP. SV-TAD thus degrades smoothly rather than collapsing as more tokens are pruned, so $k _ { r }$ can be tuned per deployment budget.

## 3.3 Auxiliary Task Supervision

We introduce auxiliary tokens, learnable embeddings that aggregate task-specific information from visual features via cross-attention. Within each adapter, after the convolution processes the bottleneck features $\mathbf { X } _ { \mathbf { v i s } } ^ { \prime } \in \mathbb { R } ^ { B \times N _ { k e p t } \times D / 4 }$ , a lightweight cross-attention module lets auxiliary tokens query visual tokens:

$$
\mathbf { X } _ { a u x } ^ { \prime } = \operatorname { C r o s s } \mathrm { A t t n } ( \mathbf { X } _ { a u x } , \mathbf { X } _ { \mathbf { v i s } } ^ { \prime } )\tag{4}
$$

where only the auxiliary tokens $\mathbf { X _ { a u x } }$ are updated. In this module, the tokens are further down projected $( D / 8 )$ . The queries $\mathbf { Q } _ { a u x }$ are projected from the auxiliary tokens $\mathbf { X _ { a u x } }$ , while keys ${ \bf K } _ { v i s }$ and values $\mathbf { V } _ { v i s }$ are projected from the visual features ${ \mathbf { X } } _ { \mathbf { v i s } } ^ { \prime } .$ This asymmetric design enables information flow from visual patches to auxiliary tokens without overhead on the visual token set. The cross-attention module also serves as the adapter’s mechanism for modeling temporal dynamics.

This makes our architecture flexible enough to support auxiliary task supervision without requiring per-task dense reconstructions. We demonstrate this capability with landmark-guided token selection and pose heatmap prediction, which we deploy on fine-grained assembly tasks (Sec. 4.5).

## 3.4 Sparse Adapter Architecture

We integrate our sparse convolution and the auxiliary cross-attention mechanism into a bottleneck adapter, named SV-TAD, inserted after each Transformer block’s attention layer. As illustrated in Fig. 4, input features pass through a down-projection, GELU activation, our sparse 2D convolution, and the crossattention module, followed by an up-projection that restores the original dimen-

![](images/c015b1cb8a8339086ac6a75d0b4064201e72e2fe953c3f3f1db2cf50d29f9bd7.jpg)  
Figure 4: Sparse adapter architecture. Our bottleneck adapter (orange, trainable; the surrounding ViT block is frozen) is inserted after each Transformer block’s attention layer. Visual $\left( \mathbf { X } _ { v i s } \right)$ and auxiliary $\left( \mathbf { X } _ { a u x } \right)$ tokens are downprojected $( \mathrm { D o w n P r o j } _ { 1 } )$ before $\mathrm { S p a r s e C o n v 2 D }$ gathers each kept token’s spatial neighbors via the precomputed neighbor index table N (−1 marks pruned/outof-bounds neighbors, top-left inset). CrossAttn then lets auxiliary tokens query the sparse visual features, updating only $\mathbf { X } _ { a u x }$ at no extra cost to the visual path, before $\mathrm { U p P r o j } _ { \mathrm { l } }$ restores the channel dimension and a learnable scalar $\gamma$ modulates the adapter’s residual contribution.

sion:

$$
\mathbf { X } _ { \mathbf { v i s } } ^ { \prime } = \operatorname { S p a r s e } \operatorname { C o n v 2 D } ( \sigma ( \operatorname { D o w n P r o j } _ { 1 } ( \mathbf { X } _ { \mathbf { v i s } } ) ) , ~ \mathbf { N } )\tag{5}
$$

$$
{ \bf X } ^ { \prime } = \mathrm { C o n c a t } ( \mathrm { C r o s s A t t n } ( { \bf X } _ { a u x } , { \bf X } _ { \bf v i s } ^ { \prime } ) , { \bf X } _ { \bf v i s } ^ { \prime } )\tag{6}
$$

$$
{ \bf { X } } _ { o u t } = \boldsymbol { \gamma } \cdot \mathrm { U p P r o j } _ { 1 } ( { \bf { X } } ^ { \prime } ) + { \bf { X } } ,\tag{7}
$$

where DownP roj<sub>1</sub>: $\mathbb { R } ^ { D } \to \mathbb { R } ^ { D / 4 }$ , σ is GELU, SparseConv2D uses neighbor table N, CrossAttn lets auxiliary tokens query the sparse bottleneck features $\mathbf { X _ { v i s } } , U p P r o j _ { 1 } \colon \mathbb { R } ^ { D / 4 }  \mathbb { R } ^ { D }$ , and γ is a learnable scalar (initialized to 1) that controls adapter influence. For early layers (before token selection), we use a standard 2D convolution, as our sparse convolution ofers no gains in this dense regime.

Temporal Context Flow. Temporal information reaches visual tokens through two complementary pathways. First, after the adapter enriches the auxiliary tokens (e.g., CLS) at layer ℓ, the frozen ViT self-attention at layer ℓ+1 naturally propagates this enriched representation to every visual token across all frames. Second, the backbone output concatenates the final CLS token along with frame features that the detection head processes directly. The CLS thus serves both as an intermediate temporal relay between adapter layers and as a global temporal summary available to the detection head.

## 4 Experiments

We demonstrate that native sparse convolutions deliver substantial eficiency gains while maintaining accuracy. We report average mAP over IoU thresholds {0.3, 0.4, 0.5, 0.6, 0.7} for THUMOS-14, {0.5, 0.75, 0.95} for ActivityNet-1.3, and {0.1, 0.2, 0.3, 0.4, 0.5} for ATTACH, following the standard protocol for each benchmark.

## 4.1 Datasets and Setup

We evaluate SV-TAD on 3 benchmarks for TAD: THUMOS-14 [15] (200/211 train/test videos, 20 sports classes), ActivityNet-1.3 [13] (10k/4,728 train/test videos, 200 activity classes), and ATTACH [1] (1,200+ assembly videos, 51 fine-grained classes with 68% temporal overlap, person-split: $2 8 / 4 / 1 0$ participants for train/val/test). For THUMOS-14 and ATTACH, we use $T = 7 6 8$ frames at $2 2 4 \times 2 2 4$ resolution as input. For ActivityNet-1.3, $T \ = \ 1 9 2$ at $1 6 0 \times 1 6 0$ . Full dataset statistics and preprocessing details are in Sec. B of the supplementary material.

We use frozen video transformer backbones (i.e., VideoMAEv2 [31], Intern-VideoNext [29]), with the OpenTAD framework [22]. Sparse adapters are inserted after each transformer block, with token selection in blocks 4, 8, 12, 16 (ViT-L) using $k _ { r } { = } 0 . 6$ . We employ ActionFormer [37] as the detection head. Full hyperparameters are provided in Sec. C of the supplementary material.

## 4.2 Eficiency Analysis

We evaluate inference of SV-TAD (ViT-L) against AdaTAD [21] (ViT-L) with identical input sizes on THUMOS.

End-to-End Throughput. Figure 1 shows an overview of TFLOPs at different model sizes. SV-TAD (ViT-L) processes 1.57 videos/second vs. the baseline’s 0.73, a 2.2x improvement: our native sparse convolution translates the 64% TFLOPs reduction at $k _ { r } { = } 0 . 6$ directly into real-world throughput gains, validating the approach for both faster training and practical deployment.

Training Speed. Our eficiency gains also extend to training: measuring average GPU active time, SV-TAD requires 3.01s per step (forward= 0.70s, backward= 2.31s) vs. the baseline’s 5.35s (forward= 1.49s, backward= 3.86s), a 1.7x speedup. With ViT-L, where token selection prunes four times over a 24- layer backbone, SV-TAD requires 9.23GB peak allocated memory per training step (single RTX 3090, batch size 1) vs. AdaTAD’s 13.30GB, $\mathrm { { a } \sim 3 1 \% }$ reduction. Despite SV-TAD having more parameters than AdaTAD at this scale (361M vs. 346M), progressively shrinking the token count that reaches selfattention at deeper layers reduces activation memory enough to outweigh the parameter count, so the memory advantage of token pruning, like the TFLOPs advantage in Fig. 1, grows with backbone depth.

![](images/16addd74d69308839b96fe61378e4ba0c8ff01465d4b9b7b1055af98b7128f19.jpg)  
(a) Speed scaling

![](images/63d8e7f078b8b0e57395bdc47b02668aff98565d0b51cb9e3e84a9ae02661415.jpg)  
(b) Memory scaling  
Figure 5: Sparse convolution eficiency scaling at $k _ { r } { = } 0 . 3 .$ . (a) Our sparse convolution (orange) achieves 1.75× speedup over dense (blue) at 6,144 frames (40ms vs. 70ms). (b) Dense convolution memory grows to ∼6GB at 6,144 frames regardless of sparsity. Our sparse convolution maintains constant memory (∼800MB) via chunked processing, achieving 7.5× reduction.

Kernel-Level Benchmarks. To verify these end-to-end gains stem from the kernel itself, we isolate the convolution module and benchmark it against dense convolution (with scatter-gather reconstruction) across temporal resolutions at $k _ { r } { = } 0 . 3$

Kernel-Level Speed. Fig. 5(a) confirms both kernels scale linearly with sequence length, but ours operates on only $N _ { k e p t } = k _ { r }$ · N tokens, yielding consistent speedups: 2× at 768 frames (4ms vs. 8ms) and $\mathbf { 1 . 7 5 } \times$ at 6,144 frames (40ms vs. 70ms). Since sparse adapters sit at every Transformer block, these per-layer gains compound across the network.

Kernel Memory Scaling. Dense convolution materializes the full $H \times W \times C$ grid regardless of sparsity, therefore memory grows linearly with sequence length (Fig. 5(b)). Ours instead allocates memory proportional to $N _ { k e p t }$ within a fixed chunk size, holding peak memory near-constant and yielding a 87% reduction at 6,144 frames (800MB vs. 6GB), enabling sequence lengths that would otherwise run out of memory.

## 4.3 Convolution Comparison: Dense vs. Sparse

To isolate the efect of our sparse convolution primitive from other architectural choices $\mathrm { ( e . g . , } $ 1D vs. 2D convolution), we compare against dense convolution baselines using the same adapter architecture and token selection, with scattergather reconstruction. Only SparseConv2D bypasses reconstruction entirely. For comparison, DenseConv1D is the temporal-convolution adapter identical to AdaTAD [21]. Table 2 presents this comparison on THUMOS-14 with the VideoMAE-B backbone.

Table 2: Dense vs. sparse convolution on THUMOS-14.  
Table 3: Backbone-matched comparison on THUMOS-14 with VideoMAEv2-B. All methods use identical settings. \* indicates our reproduction.
<table><tr><td>Conv Type</td><td>Avg.</td></tr><tr><td>DenseConv1D</td><td>69.60</td></tr><tr><td>DenseConv2D</td><td>68.83</td></tr><tr><td>SparseConv2D</td><td>69.35</td></tr></table>

<table><tr><td colspan="2">Method</td><td>TFLOPs Avg.</td></tr><tr><td>AdaTAD*</td><td>17.9</td><td>72.15</td></tr><tr><td>SV-TAD (No CrossAttn)</td><td>10.18</td><td>71.80</td></tr><tr><td>SV-TAD</td><td>10.29</td><td>72.44</td></tr></table>

DenseConv2D vs. SparseConv2D. Both apply a 3×3 spatial convolution over the same zero-padded grid, yet SparseConv2D outperforms the dense baseline by +0.52% average mAP. We attribute the gap to numerical divergence, where implementation diferences change the gradient accumulation trajectory over training. We verify that in FP32 without token selection (k<sub>r</sub>=1), the gap shrinks to 0.10%. This shows that SparseConv2D matches dense accuracy while being substantially more eficient. A detailed analysis is provided in Sec. E of the supplementary.

DenseConv1D vs. SparseConv2D. DenseConv1D achieves a slightly higher average mAP (69.60 vs. 69.35, +0.25%), as its temporal kernel directly models inter-frame dynamics, consistent with AdaTAD [21]; however, this comparison is conducted under scatter-gather reconstruction, which drastically reduces the gains obtained by token selection. Implementing an eficient native sparse 1D temporal kernel, in contrast, is dificult in practice: temporal neighbors reside in diferent frame blocks, causing ∼8x larger memory access distances than spatial neighbors, which breaks cache locality and coalesced warp access (see Sec. G of the supplementary for a detailed analysis).

## 4.4 Backbone-Matched Comparison

To isolate the contribution of our sparse convolution framework from backbone diferences, we conducted controlled experiments using the same VideoMAEv2- B backbone. Table 3 presents this fair comparison on THUMOS-14.

Analysis. The key result is the TFLOPs column: SV-TAD requires only 10.29 TFLOPs (43% fewer than AdaTAD’s 17.9) while maintaining accuracy (72.44% vs. 72.15%), confirming that native sparse convolutions eliminate reconstruction overhead without sacrificing quality and translate directly into 2.2x faster inference and 1.7x faster training (Sec. 4.2).

Efect of Cross-Attention for Temporal Recombination. The variant SV-TAD (No CrossAttn) already achieves the bulk of the computation reduction (10.18 TFLOPs) but drops slightly below AdaTAD in accuracy (71.80% vs. 72.15%). As shown in Table 2, replacing a temporal 1D convolution with a spatial 2D convolution sacrifices inter-frame modeling: DenseConv1D outperforms DenseConv2D by +0.77%. Our cross-attention module compensates for this: a single CLS query aggregates information across retained patches from diferent frames, recovering global temporal context, and closes the gap at negligible compute cost (+0.11 TFLOPs). SparseConv2D is also robust to the choice of token-selection criterion (e.g.,, EViT-style attention vs. norm-based), at matched compute (Sec. E of the supplementary).

Table 4: Ablation of auxiliary landmark supervision on ATTACH (Person Split) using VideoMAEv2-B.
<table><tr><td>Method</td><td> $\mathbf { T F L O P s } \downarrow$ </td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>Avg.</td></tr><tr><td>SV-TAD</td><td>10.29</td><td>20.17</td><td>18.69</td><td>16.90</td><td>14.39</td><td>11.25</td><td>16.28</td></tr><tr><td> $\mathrm { S V - T A D + K p }$ </td><td>11.17</td><td>22.58</td><td>21.10</td><td>19.21</td><td>16.80</td><td>13.76</td><td>18.69</td></tr></table>

## 4.5 Auxiliary Task Tokens Experiments

Sparse representations create a structural advantage beyond eficiency: since our framework explicitly tracks which tokens survive selection and their original grid positions, it naturally enables auxiliary task supervision. Existing convolutional adapters operate exclusively on the visual feature grid, with no mechanism to incorporate task-specific tokens. We demonstrate this on ATTACH [1], a finegrained assembly actions benchmark. Prior work on ATTACH evaluates only frame-level classification, we are the first to formulate it as a full TAD problem.

Table 4 shows that adding auxiliary landmark supervision (SV-TAD + Kp, Sec. 3.3) improves avg. mAP by +2.41% (18.69 vs. 16.28), demonstrating that the sparse adapter’s cross-attention injection point is an efective channel for multitask supervision. We provide a detailed ablation of these results in the supplementary material (Sec. F). Sec. E of the supplementary also validates that SparseConv2D generalizes beyond TAD, to image-to-video adaptation on a frozen CLIP ViT-B/16.

## 4.6 Comparison with State-of-the-Art

Table 5 compares SV-TAD with recent TAD methods on ActivityNet-1.3 and THUMOS-14. The TFLOPs for SV-TAD and AdaTAD [21] are for input with the same number of frames for each data set. For fair comparison, we report AdaTAD last-epoch test mAP from the authors’ publicly released training logs.

In THUMOS-14, SV-TAD with VideoMAEv2-B requires 43% fewer TFLOPs than AdaTAD [21] (10.29 vs. 17.9) while achieving 72.44% average mAP, surpassing AdaTAD’s 70.67% by +1.77%. We note that ViT-TAD’s 69.5% with ViT-B is obtained under a substantially reduced input regime (256 frames at 160×160 vs. our 768 frames at 224×224), and its full fine-tuning design prevents scaling beyond ViT-B. Scaling to ViT-L widens the eficiency gap: SV-TAD uses 64% fewer TFLOPs (21.25 vs. 59.4) while matching AdaTAD’s accuracy (73.47% vs. 73.50%), demonstrating that our eficiency gains grow with model size.

Table 5: Results comparison on ActivityNet-1.3 and THUMOS14 (\* indicates mAP obtained from oficial logs). Size-matched comparisons are highlighted.
<table><tr><td rowspan="2"></td><td rowspan="2">Backbone</td><td colspan="5">ActivityNet-1.3</td><td colspan="7">THUMOS14</td></tr><tr><td>TFLOPs↓</td><td>0.5</td><td>0.75</td><td>0.95</td><td>Avg.</td><td>TFLOPs↓</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>Avg.</td></tr><tr><td>AFSD [19]</td><td>I3D</td><td></td><td>52.40</td><td>35.30</td><td>6.50</td><td>34.40</td><td></td><td>67.3</td><td>62.4</td><td>55.5</td><td>43.7</td><td>31.1</td><td>52.0</td></tr><tr><td>E2E-TAD [23]</td><td>SlowFast-R50</td><td>82.48</td><td>50.47</td><td>35.99</td><td>10.33</td><td>34.10</td><td>82.48</td><td>69.4</td><td>64.3</td><td>56.0</td><td>46.4</td><td>34.9</td><td>54.2</td></tr><tr><td>BasicTAD [33]</td><td>SlowOnly-R50</td><td></td><td>51.20</td><td>33.41</td><td>7.57</td><td>33.12</td><td></td><td>75.5</td><td>70.8</td><td>63.5</td><td>50.9</td><td>37.4</td><td>59.2</td></tr><tr><td>TALLFormer [5]</td><td>VideoSwin-B</td><td></td><td>54.10</td><td>36.20</td><td>7.90</td><td>35.60</td><td></td><td>76.0</td><td>70.0</td><td>63.2</td><td></td><td>34.5</td><td>59.4</td></tr><tr><td>Re2TAD [38]</td><td>Re2VSwin-T</td><td></td><td>54.75</td><td>37.81</td><td>9.03</td><td>36.80</td><td></td><td>77.0</td><td>71.5</td><td>62.4</td><td>49.7</td><td>36.3</td><td>59.4</td></tr><tr><td>LoSA [11]</td><td>VMAEv2-G</td><td>132</td><td>58.5</td><td>39.8</td><td>7.80</td><td>38.6</td><td>132</td><td>85.0</td><td>81.1</td><td>74.5</td><td>65.1</td><td>49.3</td><td>71.0</td></tr><tr><td>ViT-TAD [34]</td><td>ViT-B</td><td></td><td>55.87</td><td>38.47</td><td>8.80</td><td>37.40</td><td></td><td>85.1</td><td>80.9</td><td>74.2</td><td>61.8</td><td>45.4</td><td>69.5</td></tr><tr><td>AdaTAD [21]*</td><td>VMAE-B</td><td>8.05</td><td>56.66</td><td>39.49</td><td>9.38</td><td>38.31</td><td>17.90</td><td>85.58</td><td>81.40</td><td>74.31</td><td>63.04</td><td>49.00</td><td>70.67</td></tr><tr><td>AdaTAD [21]*</td><td>VMAE-L</td><td>27.42</td><td>57.68</td><td>40.52</td><td>9.96</td><td>39.18</td><td>59.40</td><td>87.11</td><td>83.65</td><td>76.84</td><td>66.82</td><td>53.07</td><td>73.50</td></tr><tr><td>AdaTAD [21]*</td><td>VMAEv2-G</td><td>83.59</td><td>57.40</td><td>40.44</td><td>9.78</td><td>39.20</td><td>176.87</td><td>87.50</td><td>84.02</td><td>77.61</td><td>67.05</td><td>53.19</td><td>73.87</td></tr><tr><td>SV-TAD</td><td>VMAEv2-B</td><td>4.75</td><td>57.09</td><td>39.97</td><td>9.77</td><td>38.80</td><td>10.29</td><td>87.54</td><td>83.24</td><td>75.28</td><td>65.54</td><td>50.61</td><td>72.44</td></tr><tr><td>SV-TAD</td><td>VMAEv2-L</td><td>10.10</td><td>58.40</td><td>40.81</td><td>9.76</td><td>39.58</td><td>21.25</td><td>87.26</td><td>83.80</td><td>77.22</td><td>66.21</td><td>52.85</td><td>73.47</td></tr><tr><td>SV-TAD</td><td>InternVideoNext-L</td><td>34.00</td><td>58.79</td><td>41.30</td><td>10.41</td><td>40.06</td><td>87.05</td><td>88.38</td><td>84.40</td><td>78.22</td><td>66.82</td><td>52.85</td><td>74.13</td></tr></table>

When scaling to InternVideoNext-L, the eficiency advantage becomes most pronounced. On THUMOS-14, SV-TAD requires only 87.05 TFLOPs vs.176.87 for AdaTAD’s billion-parameter VMAEv2-G, while surpassing its 73.87% Avg. mAP by +0.26%: a Large-sized backbone with native sparse convolutions surpasses Giant-level performance at half the computational cost. On Activity-Net-1.3, SV-TAD with VideoMAEv2-B uses 41% fewer TFLOPs (4.75 vs. 8.05) while achieving 38.80% average mAP, surpassing AdaTAD (38.39%) by +0.41%. The most compelling result is with InternVideoNext-L: SV-TAD requires 59% fewer TFLOPs (34 vs. 83.59) while achieving 40.06% average mAP, surpassing AdaTAD with VMAEv2-G (39.20%) by +0.86% at every IoU threshold (Table 5). To isolate this gain from the backbone switch itself, we also run AdaTAD with the same InternVideoNext-L backbone: it reaches 39.72% average mAP at 95.3 TFLOPs, confirming that, at matched backbone, SV-TAD still cuts compute substantially (34 vs. 95.3 TFLOPs) while improving accuracy (+0.34%), so the gains stem from SV-TAD itself rather than from the choice of backbone.

## 5 Conclusion

We introduced a native sparse 2D convolution primitive that enables convolutional adapters to operate on dynamically pruned token sets without reconstructing the dense spatial grid. This makes sparsity efective across the entire model rather than only within attention layers. Integrated into our SV-TAD framework, this design yields substantial reductions in compute and inference time while maintaining or improving detection accuracy across multiple benchmarks.

Crucially, the gains grow with model capacity: with ViT-L, we match top performing accuracy at 64% fewer TFLOPs, while with InternVideoNext-L we set a new state of the art on both THUMOS-14 and ActivityNet-1.3, achieving the best accuracy while requiring roughly half the compute (51% and 59% fewer TFLOPs, respectively). In ATTACH, auxiliary landmark supervision enabled by the sparse adapter’s cross-attention injection point, sets a new state of the art and improves our baseline model by +2.41% mAP, demonstrating that the framework extends naturally to multitask learning on fine-grained assembly.

Beyond temporal action detection, our primitive is applicable wherever convolutional modules follow token selection in Vision Transformers, including spatial adapters, feature pyramid networks, and task-specific heads.

Limitations and Future Work. Our implementation targets NVIDIA GPUs; porting to other accelerators requires replacing the backend, although the core algorithm is architecture-agnostic. Additionally, the sparse kernel becomes faster than its dense counterpart only below the ∼55% keep rate (Fig. 3), so the primitive is most beneficial when progressive pruning through the layers is acceptable. Exploring learned, content-adaptive keep-rate schedules and extending the primitive to 3D spatio-temporal neighborhoods are promising directions for future work.

## Acknowledgements

This work was supported by projects PID2022-137581OB-I00, PLEC2023-010343 (INARTRANS 4.0) and PID2024-161576OB-I00, funded by the Spanish MI-CIU/AEI/10.13039/501100011033 and FEDER, UE, co-funded by the European Regional Development Fund (ERDF, “A way of making Europe”), and from iRoboCity2030-CM project (grant TEC-2024/TEC-62), awarded by the Community of Madrid. RP, JMB, LMB and LB are members of the Madrid ELLIS Unit, funded by the Autonomous Community of Madrid, Spain.

## References

[1] Dustin Aganian, Benedict Stephan, Markus Eisenbach, Corinna Stretz, and Horst-Michael Gross. Attach dataset: Annotated two-handed assembly actions for human action understanding. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 11367–11373, 2023.

[2] Tanay Agrawal, Abid Ali, Antitza Dantcheva, and Francois Bremond. Scaling action detection: Adatad++ with transformer-enhanced temporalspatial adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12222–12231, 2025.

[3] Lucas Beyer, Xiaohua Zhai, Am´elie Royer, Larisa Markeeva, Rohan Anil, and Alexander Kolesnikov. Knowledge distillation: A good teacher is patient and consistent. In IEEE/CVF Conference on Computer Vision and

Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 10915–10924. IEEE, 2022.

[4] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hofman. Token merging: Your vit but faster. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[5] Feng Cheng and Gedas Bertasius. Tallformer: Temporal action localization with a long-memory transformer. In Shai Avidan, Gabriel J. Brostow, Moustapha Ciss´e, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXXIV, volume 13694 of Lecture Notes in Computer Science, pages 503–521. Springer, 2022.

[6] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

[7] Christopher B. Choy, JunYoung Gwak, and Silvio Savarese. 4d spatiotemporal convnets: Minkowski convolutional neural networks. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 3075–3084. Computer Vision Foundation / IEEE, 2019.

[8] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 702–703, 2020.

[9] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-eficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

[10] Benjamin Graham and Laurens Van der Maaten. Submanifold sparse convolutional networks. arXiv preprint arXiv:1706.01307, 2017.

[11] Akshita Gupta, Gaurav Mittal, Ahmed Magooda, Ye Yu, Graham W. Taylor, and Mei Chen. Losa: Long-short-range adapter for scaling end-to-end temporal action localization. In IEEE/CVF Winter Conference on Applications of Computer Vision, WACV 2025, Tucson, AZ, USA, February 26 - March 6, 2025, pages 2092–2102. IEEE, 2025.

[12] Joakim Bruslund Haurum, Sergio Escalera, Graham W. Taylor, and Thomas B. Moeslund. Which tokens to use? investigating token reduction in vision transformers. In IEEE/CVF International Conference on Computer Vision, ICCV 2023 - Workshops, Paris, France, October 2-6, 2023, pages 773–783. IEEE, 2023.

[13] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pages 961–970. IEEE Computer Society, 2015.

[14] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-eficient transfer learning for nlp. In International conference on machine learning, pages 2790–2799. PMLR, 2019.

[15] Haroon Idrees, Amir R. Zamir, Yu-Gang Jiang, Alex Gorban, Ivan Laptev, Rahul Sukthankar, and Mubarak Shah. The THUMOS challenge on action recognition for videos ”in the wild”. Comput. Vis. Image Underst., 155:1– 23, 2017.

[16] Kwonyoung Kim, Jungin Park, Jin Kim, Hyeongjun Kwon, and Kwanghoon Sohn. Faster parameter-eficient tuning with token redundancy reduction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 30189–30198. Computer Vision Foundation / IEEE, 2025.

[17] Yingwei Li, Yi Li, and Nuno Vasconcelos. Resound: Towards action recognition without representation bias. In Proceedings of the European Conference on Computer Vision (ECCV), pages 513–528, 2018.

[18] Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. Evit: Expediting vision transformers via token reorganizations. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.

[19] Chuming Lin, Chengming Xu, Donghao Luo, Yabiao Wang, Ying Tai, Chengjie Wang, Jilin Li, Feiyue Huang, and Yanwei Fu. Learning salient boundary feature for anchor-free temporal action localization. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 3320–3329. Computer Vision Foundation / IEEE, 2021.

[20] Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, and Shilei Wen. BMN: boundary-matching network for temporal action proposal generation. In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019, pages 3888– 3897. IEEE, 2019.

[21] Shuming Liu, Chen-Lin Zhang, Chen Zhao, and Bernard Ghanem. Endto-end temporal action detection with 1b parameters across 1000 frames. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 18591–18601. IEEE, 2024.

[22] Shuming Liu, Chen Zhao, Fatimah Zohra, Mattia Soldan, Alejandro Pardo, Mengmeng Xu, Lama Alssum, Merey Ramazanova, Juan Le´on Alc´azar, Anthony Cioppa, Silvio Giancola, Carlos Hinojosa, and Bernard Ghanem. Opentad: A unified framework and comprehensive study of temporal action detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2025, Nashville, TN, USA, June 11-15, 2025, pages 2625–2635. Computer Vision Foundation / IEEE, 2025.

[23] Xiaolong Liu, Song Bai, and Xiang Bai. An empirical study of end-to-end temporal action detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 19978–19987. IEEE, 2022.

[24] Junting Pan, Ziyi Lin, Xiatian Zhu, Jing Shao, and Hongsheng Li. Stadapter: Parameter-eficient image-to-video transfer learning. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.

[25] Ricardo Pizarro, Roberto Valle, Jos´e M. Buenaposada, Luis M. Bergasa, and Luis Baumela. Pose-guided token selection for the recognition of activities of daily living. Image and Vision Computing, 162:105686, 2025.

[26] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Eficient vision transformers with dynamic token sparsification. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 13937–13949, 2021.

[27] Michael S. Ryoo, A. J. Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. Tokenlearner: Adaptive space-time tokenization for videos. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 12786–12797, 2021.

[28] Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Eficient transformers: A survey. ACM Comput. Surv., 55(6):109:1–109:28, 2023.

[29] Chenting Wang, Yuhan Zhu, Yicheng Xu, Jiange Yang, Lang Lin, Ziang Yan, Yali Wang, Yi Wang, and Limin Wang. Internvideo-next: Towards general video foundation models without video-text supervision. arXiv preprint arXiv:2512.01342, 2025.

[30] Junke Wang, Xitong Yang, Hengduo Li, Li Liu, Zuxuan Wu, and Yu-Gang Jiang. Eficient video transformers with spatial-temporal token selection. In Shai Avidan, Gabriel J. Brostow, Moustapha Ciss´e, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXXV, volume 13695 of Lecture Notes in Computer Science, pages 69–86. Springer, 2022.

[31] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae V2: scaling video masked autoencoders with dual masking. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 14549–14560. IEEE, 2023.

[32] Yan Yan, Yuxing Mao, and Bo Li. SECOND: sparsely embedded convolutional detection. Sensors, 18(10):3337, 2018.

[33] Min Yang, Guo Chen, Yin-Dong Zheng, Tong Lu, and Limin Wang. Basictad: An astounding rgb-only baseline for temporal action detection. Comput. Vis. Image Underst., 232:103692, 2023.

[34] Min Yang, Huan Gao, Ping Guo, and Limin Wang. Adapting short-term transformers for action detection in untrimmed videos. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 18570–18579. IEEE, 2024.

[35] Taojiannan Yang, Yi Zhu, Yusheng Xie, Aston Zhang, Chen Chen, and Mu Li. AIM: adapting image models for eficient video action recognition. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[36] Xin Yuan, Hongliang Fei, and Jinoo Baek. Eficient transformer adaptation with soft token merging. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024 - Workshops, Seattle, WA, USA, June 17-18, 2024, pages 3658–3668. IEEE, 2024.

[37] Chen-Lin Zhang, Jianxin Wu, and Yin Li. Actionformer: Localizing moments of actions with transformers. In Shai Avidan, Gabriel J. Brostow, Moustapha Ciss´e, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part IV, volume 13664 of Lecture Notes in Computer Science, pages 492–510. Springer, 2022.

[38] Chen Zhao, Shuming Liu, Karttikeya Mangalam, and Bernard Ghanem. Re<sup>{}^\mbox</sup> <sup>{2}</sup> tal: Rewiring pretrained video backbones for reversible temporal action localization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 10637–10647. IEEE, 2023.

## Supplemental Materials

This supplementary provides: CUDA kernel implementation details (Sec. S-I), dataset descriptions (Sec. S-II), full training hyperparameters (Sec. S-III), AdaTAD++ reproduction analysis (Sec. S-IV), additional ablations including keep rate sensitivity, dense vs. sparse numerical divergence, robustness to the selection criterion, and generality beyond TAD (Sec. S-V), detailed ATTACH analysis (Sec. S-VI), and a discussion covering the incompatibility between token selection and standard convolutions, dense vs. sparse convolution comparisons, and why a native sparse 1D temporal kernel is infeasible..

## S-I CUDA Kernel Implementation Details

We implement two complementary kernel strategies, automatically selected based on the channel dimension $C _ { o u t }$ . Both share the same interface: they consume the neighbor index table N (Sec. 3.2 of the main paper) and produce outputs only at kept positions, with no dense reconstruction.

## S-I.1 Implicit GEMM via cuBLAS (C ≥ 256)

A direct implementation would iterate over all nine 3×3 kernel positions, issuing a separate GEMM, gather, and scatter per position, 36 kernel launches with substantial host-side overhead. We instead reorganise the data layout so that a single strided batched GEMM replaces all nine multiplications. The forward pass reduces to three kernel launches:

1. Fused gather with implicit zero-padding: A single CUDA kernel populates $\mathbf { X } _ { g } \in \mathbb { R } ^ { 9 \times N \times C _ { i n } }$ by gathering all 9 neighbors for all N tokens simultaneously. Invalid neighbors $( \mathbf { N } [ k , j ] = - 1 )$ are written as zeros directly.

2. Single batched GEMM: The layout $[ 9 , N , C _ { i n } ]$ is specifically designed for cublasGemmStridedBatchedEx, which computes all 9 matrix multiplications $\mathbf Y _ { j } = \mathbf X _ { g } [ j ] \cdot \mathbf W _ { j }$ in one call, fully utilizing Tensor Cores.

3. Fused reduction: A parallel reduction sums over the nine kernel positions, adds bias, and casts to the output dtype. This replaces scatterbased accumulation (index add ) with a simple strided read, eliminating all CPU synchronisation points.

Chunked processing extends naturally: tokens are partitioned into fixed-size chunks (e.g.,, 32K), bounding peak memory independently of sequence length.

## S-I.2 Custom Tiled Kernel $( C < 2 5 6 )$

When $C _ { o u t }$ is small, cuBLAS launch overhead dominates compute. We therefore provide hand-tuned CUDA kernels that fuse gather and multiply–accumulate into a single launch with no intermediate bufers, following a tiled GEMM structure adapted for sparse convolution:

1. Tile configuration: Each thread block processes a tile of 32 output tokens $\times \ 3 2$ output channels. Threads are mapped so each computes 2 tokens × 4 channels (a float4), achieving 2× thread coarsening that amortizes neighbor index lookups and register pressure.

2. Cooperative loading: In the load phase, threads collaboratively fetch a $3 2 \times 3 2$ tile of input features and the corresponding weight slice into shared memory. The sparse gather happens here: each thread reads its assigned row’s neighbor index from the indirection table and fetches from the neighbor’s location (or writes zero if −1). This fuses gather with the tile load at no extra cost.

3. Compute phase: A fully-unrolled inner loop multiplies the shared memory tiles, accumulating into float4 registers. The outer loops iterate over the 9 kernel positions and $C _ { i n } / 3 2$ input channel tiles.

Sensitivity to spatial distribution. While the sparse kernels outperform dense baselines even under uniform random pruning, they reach peak throughput when retained tokens are spatially clustered. In practice, our token selection modules learn to concentrate tokens around salient actors and objects, so threads within a warp access neighbouring memory addresses, maximising cache-line reuse and memory coalescing.

## S-I.3 Shared Kernel Infrastructure

Vectorised memory access. All kernels use 128-bit loads (float4 for FP32, int4 for FP16/BF16) with specialised unpack routines per dtype to maximise memory bandwidth.

Mixed-precision accumulation. Inputs in FP16/BF16 are accumulated in FP32 registers. A fused operation adds bias, casts to the output dtype, and writes results to global memory in a single pass.

## S-I.4 Complexity Analysis

Our sparse convolution requires $\mathcal { O } ( N _ { k e p t } \cdot K ^ { 2 } \cdot C _ { i n } \cdot C _ { o u t } )$ operations versus $\mathcal { O } ( T H W \cdot K ^ { 2 } \cdot C _ { i n } \cdot C _ { o u t } )$ for the dense baseline, yielding a direct $k _ { r } \times$ compute reduction. Chunked processing further decouples peak memory from sequence length: with a fixed chunk of 32K tokens, memory is bounded to O(chunk size · $( C _ { i n } + C _ { o u t } ) )$ , enabling arbitrarily long video sequences. Since the full model applies token selection at $S$ stages $( \mathrm { e . g . , ~ } S \mathrm { = 3 }$ at blocks 4, 7, 10 for ViT-B), layers after the final selection stage operate at the most aggressive sparsity, where per-layer gains are largest.

## S-II Dataset Details

## S-II.1 THUMOS-14

THUMOS-14 [15] contains 200 validation and 211 test videos from 20 sports action classes, totaling approximately 26 hours of video at 30 fps (∼2.8M frames). Videos are untrimmed with an average of 15 action instances per video, exhibiting significant temporal overlap and diverse action durations ranging from fractions of a second to minutes. Following AdaTAD [21], we sample 768 frames per video with temporal stride 1, chunking each into 48 non-overlapping 16- frame clips processed by the ViT backbone. We follow standard protocol: train on validation set, evaluate on test set.

## S-II.2 ActivityNet-1.3

ActivityNet-1.3 [13] contains 10,024 training, 4,926 validation, and 5,044 test videos spanning 200 activity classes, comprising approximately 849 hours (∼91M frames at 30 fps). As test labels are not publicly released, we train on the training split and evaluate on the validation split, excluding 198 videos blocked on YouTube following the convention of BMN [20], yielding 4,728 usable videos for evaluation. Videos average 2 minutes with 1.5 action instances each. Due to longer durations, we sample 192 frames uniformly across each video with adaptive stride, yielding 12 clips regardless of original length.

## S-II.3 ATTACH

ATTACH [1] features fine-grained bimanual assembly tasks at 30 fps, totaling ∼40 hours (∼4.3M frames). The dataset contains 51 action classes across 1,200+ videos with dense temporal annotations. We sample 768 frames per video with stride 4, covering ∼100 seconds per video. Over 68% of annotations temporally overlap due to simultaneous two-handed activities.

## S-III Full Training Configuration

Table S1 summarizes the complete hyperparameters used across all three datasets.

Data Augmentation. We apply random temporal cropping (crop ratio 0.9– 1.0), random horizontal flipping (probability 0.5), and RandAugment [8].

Detection Head. ActionFormer with 6 pyramid levels, regression range ratios {1, 2, 4, 8, 16, 32}, focal loss (γ=2, α=0.25) for classification, and DIoU loss for regression.

Table S1: Complete training hyperparameters across all datasets.
<table><tr><td>Hyperparameter</td><td>THUMOS-14 ActivityNet ATTACH</td><td></td></tr><tr><td>Epochs</td><td>100</td><td>10 50</td></tr><tr><td>Batch size</td><td>2</td><td>4</td></tr><tr><td>Number of GPUs</td><td>2</td><td>2</td></tr><tr><td>Optimizer</td><td>AdamW</td><td></td></tr><tr><td>Base learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td></td></tr><tr><td>Weight decay</td><td> $5 \times 1 0 ^ { - 2 }$ </td><td></td></tr><tr><td>LR schedule</td><td>Cosine</td><td></td></tr><tr><td>Warmup epochs</td><td>5</td><td></td></tr><tr><td>Precision</td><td>BFloat16</td><td></td></tr><tr><td>Frames sampled</td><td>768</td><td>192</td></tr><tr><td>Temporal stride</td><td>1</td><td>768 adaptive</td></tr><tr><td>Spatial resolution</td><td>224 × 224</td><td>4 160 × 160 224 × 224</td></tr><tr><td>Adapter bottleneck ratio</td><td>4</td><td></td></tr><tr><td>Sparse conv kernel</td><td>3 × 3</td><td></td></tr><tr><td>Token sel. blocks (ViT-B)</td><td>4, 7, 10</td><td></td></tr><tr><td>Token sel. blocks (ViT-L)</td><td>4, 8, 12, 16</td><td></td></tr><tr><td>Default keep rate kr</td><td>0.6 (VMAE), 0.7 (InternVideo)</td><td></td></tr><tr><td>Auxiliary loss weight λ</td><td></td><td>2.0</td></tr><tr><td>Number of keypoints</td><td></td><td>17</td></tr><tr><td>Heatmap resolution</td><td></td><td>56 × 56</td></tr><tr><td></td><td></td><td></td></tr></table>

Table S2: AdaTAD++ reproduction comparison. Reported values from original paper vs. our reproduction using OpenTAD.
<table><tr><td></td><td></td><td colspan="6">THUMOS-14 (mAP@IoU)</td></tr><tr><td>Method</td><td>Source</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>Avg.</td></tr><tr><td>AdaTAD++ (VMAE-B)</td><td>Reported</td><td>88.3</td><td>83.7</td><td>76.6</td><td>65.0</td><td>51.8</td><td>73.3</td></tr><tr><td>AdaTAD++ (VMAE-B)</td><td>Ours</td><td>84.64</td><td>80.33</td><td>72.68</td><td>62.44</td><td>49.67</td><td>69.95</td></tr></table>

## S-III.1 End-to-End Training and Inference

Our complete pipeline integrates a frozen pretrained[31, 29] backbone with trainable sparse adapters and an ActionFormer [37] detection head. During training, we optimize only the sparse adapters and detection head, keeping the backbone frozen for parameter eficiency. When auxiliary supervision is used $\mathrm { ( e . g . , } $ on ATTACH), the auxiliary head is jointly optimized with loss $\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { d e t } + \lambda \mathcal { L } _ { a u x } , } \end{array}$ where $\mathcal { L } _ { a u x }$ is MSE for heatmap regression. Otherwise, training uses $\mathcal { L } = \mathcal { L } _ { d e t }$ alone.

## S-IV AdaTAD++ Reproduction Analysis

We attempted to reproduce AdaTAD++ [2] results using the OpenTAD framework [22]. Table S2 summarizes our reproduction attempt.

Reproduction Gap Analysis. Our AdaTAD++ reproduction achieves 69.95% vs. 73.3% reported (−3.35%). We attribute this discrepancy to:

• Missing hyperparameters: The original AdaTAD++ paper does not release code or full hyperparameter configurations.

• Data augmentation: Specific augmentation strategies are not fully documented.

• Training dynamics: Training techniques are not fully specified.

Despite extensive hyperparameter search, we were unable to close this gap. Our reproduction used the publicly available AdaTAD [21] codebase as the base configuration, applying the additional AdaTAD++ modifications as described in the paper. We contacted the authors to request the missing configurations but received no reply.

## S-V Additional Ablations

## S-V.1 Dense vs. Sparse 2D Convolution: Numerical Divergence

Although a dense 2D convolution over a zero-padded grid and our sparse 2D convolution are mathematically equivalent, SparseConv2D outperforms the dense baseline by +0.52% average mAP on THUMOS-14 (Table 2 of the main paper). We identify two sources of numerical divergence that explain this gap.

Diferent forward computational backends. Our sparse convolution dispatches a cuBLAS strided batched GEMM with explicit FP32 accumulation (cuda compute type = CUDA R 32F), whereas the dense path uses cuDNN’s conv2d, which may employ diferent tiling strategies, Winograd transforms, or accumulation orders internally. Because floating-point addition is non-associative, these difering reduction orders produce slightly diferent rounding trajectories even when the mathematical operation is identical. Under BFloat16 inputs, perelement quantization amplifies these trajectory diferences with every forward pass.

Sparse backward kernel skips pruned positions. The backward kernels of our sparse convolution (both sparse bwd in and sparse bwd w) explicitly skip pruned positions via the −1 sentinel in the neighbor table (if (n idx == -1) continue). The dense backward, dispatched through PyTorch autograd on F.conv2d, accumulates weight-gradient contributions from all H×W grid positions including the zero-padded ones. Although these zero-input contributions are analytically zero, excluding them changes the number of terms in the accumulation and their ordering, yielding marginally diferent weight updates that compound over thousands of training steps.

Verification. We validate this explanation with two controlled experiments. First, we compare kernel-level outputs at diferent precision levels: at FP32, dense and sparse outputs match to within machine epsilon, whereas under B-Float16 discrepancies emerge and increase as the keep rate decreases, consistent with greater backend divergence when more positions difer between the two paths. Second, we eliminate both sources simultaneously by running both methods at $k _ { r } { = } 1 . 0$ (no pruning) in FP32, making the computational graphs equivalent. Under these conditions, SparseConv2D obtains 69.79% and Dense 2DConv obtains 69.89%, a 0.10% gap well within run-to-run variance. This confirms that the +0.52% gap observed at $k _ { r } { = } 0 . 6$ under BFloat16 is a reproducible consequence of the backend and gradient-path diferences described above, not a methodological artifact.

## S-V.2 Robustness to the Token Selection Criterion

SparseConv2D consumes only the kept-token grid coordinates, so it is agnostic to the criterion used to choose which tokens are kept. To test this, we swap only the selector on THUMOS-14 with VideoMAEv2-B, keeping every other component fixed: an EViT-style attention criterion [18] reaches 72.43%/10.301 TFLOPs and a norm-based criterion (ranking tokens by post-LayerNorm activation magnitude, following the survey of selection criteria in [12]) reaches 68.06%/10.296 TFLOPs, against our default attention-based selector at 72.44%/10.297 TFLOPs. The near-identical compute confirms that SparseConv2D adds no overhead regardless of selector; the residual accuracy gap is determined by the selection criterion itself, not by the sparse convolution.

## S-V.3 Generality Beyond Temporal Action Detection

SparseConv2D is designed as a general token-selection primitive for PEFT adapters in ViTs, not one specific to TAD or to the SV-TAD architecture. To test cross-task and cross-backbone applicability, we integrate it into image-tovideo adaptation: a frozen CLIP ViT-B/16 fine-tuned with AIM [35] for action recognition on Diving-48 [17]. We replace AIM’s S Adapter, a dense spatial adapter, with SparseConv2D combined with token selection, leaving the rest of the AIM recipe unchanged. Our variant reaches 88.7% top-1 accuracy at 1.01 TFLOPs, against AIM’s 88.9% at 1.62 TFLOPs, a 37.5% compute reduction at comparable accuracy (−0.2%). This shows that the eficiency benefit of SparseConv2D is not specific to temporal action detection: it transfers directly to a diferent task (action classification) and a diferent frozen backbone (CLIP ViT-B/16).

## S-VI Detailed ATTACH Analysis

ATTACH poses a distinct challenge for sparse methods: over 68% of its annotations temporally overlap due to simultaneous bimanual actions, requiring the model to represent multiple concurrent activities per frame. Table S3 ablates our design choices on this benchmark.

Table S3: Ablation on ATTACH (Person Split) with VideoMAEv2-B. SV-TAD + Kp adds auxiliary landmark supervision. \* denotes our reproduction.
<table><tr><td>Method</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>Avg.</td></tr><tr><td>AdaTAD* [21]</td><td>21.68</td><td>20.20</td><td>18.32</td><td>16.00</td><td>13.03</td><td>17.84</td></tr><tr><td>SV-TAD (No CrossAttn)</td><td>21.11</td><td>19.53</td><td>17.52</td><td>15.09</td><td>12.20</td><td>17.09</td></tr><tr><td>SV-TAD</td><td>20.17</td><td>18.69</td><td>16.90</td><td>14.39</td><td>11.25</td><td>16.28</td></tr><tr><td> $\mathrm { S V - T A D + K p }$ </td><td>22.58</td><td>21.10</td><td>19.21</td><td>16.80</td><td>13.76</td><td>18.69</td></tr></table>

Efect of cross-attention without auxiliary tokens. Comparing rows 2 and 3 isolates the cross-attention module: adding it reduces performance from 17.09% to 16.28% (−0.81%). Under sparsity, a single CLS query must aggregate information for multiple concurrent actions, creating a representational bottleneck that is particularly harmful on multilabel data where frames routinely contain two simultaneous hand actions.

Landmark supervision resolves this bottleneck. Replacing the single CLS query with K=17 pose-landmark queries provides spatially-grounded attention targets, each tied to a specific body region. SV-TAD + Kp achieves 18.69% avg. mAP, surpassing AdaTAD by +0.85% while requiring 38% fewer FLOPs (11.17 vs. 17.90 TFLOPs). Assembly actions are defined by hand and arm movements, so pose landmarks directly inform the disambiguation of dense, overlapping action boundaries. As a secondary benefit, the auxiliary loss guides token selection toward task-relevant regions (i.e., hands and manipulated objects), improving both feature quality and computational focus.

Summary. These results show that the cross-attention module is beneficial only when paired with informative auxiliary tokens. On datasets lacking such supervision, the simpler No CrossAttn variant is preferable. More broadly, they demonstrate the modularity of our framework: domain-specific auxiliary tasks can be incorporated with minimal architectural overhead when suitable supervision is available, and cleanly removed otherwise.

## S-VII Discussion

We address three design questions that arise from the main paper: whether memory overhead increases (Sec. S-VII.1), why standard convolutions cannot directly operate on sparse tokens (Sec. S-VII.2), and why we chose a spatial rather than temporal sparse kernel (Sec. S-VII.3)..

## S-VII.1 Memory Consumption

Peak GPU memory remains comparable between SV-TAD and AdaTAD. The frozen backbone activations dominate the overall memory footprint regardless of adapter design. At 768 frames with VideoMAEv2-Base, the backbone accounts for the vast majority of allocated memory, while the adapter modules (whether dense or sparse) contribute a negligible fraction. Because our sparse kernels replace rather than augment the convolution pathway, they do not introduce additional bufers beyond the neighbor index table $( N _ { k e p t } \times 9$ int32 entries, ∼0.2 MB at $k _ { r } { = } 0 . 6 )$ . In practice, SV-TAD and AdaTAD occupy nearly the same peak memory, meaning the throughput and FLOP reductions come at no memory cost.

## S-VII.2 Incompatibility Between Token Selection and Standard Convolutions

Standard convolutions assume a dense, regularly structured input tensor. Token selection breaks this assumption by producing a variable-length list of kept tokens rather than a dense grid. After global TopK pruning [18] across the flattened spatio-temporal sequence $( B , T \times H \times W , C )$ , the output is a variablelength list of kept tokens, not a dense grid. This creates two structural problems: (i) non-uniform token counts per frame (frames with salient content retain more tokens, violating the fixed $H \times W$ layout that Conv1d/Conv2d expect), and (ii) misaligned spatial positions across frames (the which positions survive difers per frame, so grid indices in the sparse list have no consistent spatial correspondence).

The scatter-gather workaround. A solution is to reconstruct the dense tensor before convolution: scatter the $N _ { k e p t }$ sparse tokens back to their original grid positions in a zero-initialized tensor of full size, apply the convolution, then gather outputs at the kept positions. Concretely, for a 1D temporal adapter, this involves:

1. Allocate a dense tensor ${ \bf X } _ { d e n s e } \in \mathbb { R } ^ { B \times H \times W \times T \times C }$ , initialized to zero.

2. Scatter: for each kept token k at grid position $( b , t , h , w )$ , write $\mathbf { X } _ { d e n s e } [ b , h , w , t , : ] = \mathbf { x } _ { k }$

3. Reshape to $( B { \times } H { \times } W , C , T )$ and apply nn.Conv1d.

4. Gather: extract outputs at the $N _ { k e p t }$ positions via gather, discarding the zero-padded results.

For a 2D spatial adapter, the process is analogous:

scatter into $( B \times T , C , H , W )$ , apply nn.Conv2d, and gather back. The key point is that every standard convolution variant requires this same reconstructconvolve-extract pipeline.

Eficiency limitations. This workaround has three compounding costs. (1) Memory: the dense tensor allocates $\mathcal { O } ( H \times W { \times } T { \times } C )$ regardless of how many tokens were pruned, growing linearly with frame count. (2) Wasted computation: the convolution operates over the full tensor including zero-padded positions; at $k _ { r } { = } 0 . 6 , \sim 4 0 \%$ of FLOPs produce no useful output. (3) Scatter-gather overhead: both operations involve irregular memory access patterns that create a hard latency floor, pruning more aggressively reduces convolution FLOPs but cannot reduce this fixed cost.

Native sparse convolution. SparseConv2D replaces the entire allocate-scatterconvolve-gather pipeline with a single fused operation. The precomputed neighbor index table $\mathbf { \hat { N } } \in \mathbb { Z } ^ { N _ { k e p t } \times 9 }$ maps each kept token directly to its neighbors in the sparse list. The CUDA kernel gathers features through this indirection, computes the convolution, and writes outputs, all without materializing a dense grid. Memory and computation scale with $N _ { k e p t } .$ not $H \times W \times T$ , converting token reduction ratios into proportional wall-clock savings.

## S-VII.3 Why Spatial Rather Than Temporal Sparse Convolution

Dense 1DConv achieves marginally higher mAP than Dense 2DConv in our ablations (Table 2 of the main paper), raising the question of why we implement a native sparse spatial rather than temporal kernel. We find that three GPUlevel constraints make the temporal variant impractical.

Memory locality. Tokens within a single frame are stored contiguously: spatial neighbors are at most ∼14 positions apart (one grid row), placing the gather within 5–11 KB, well within L1 cache. A temporal neighbor resides in an adjacent frame, $N _ { k e p t / f r a m e } \times C \times 2$ bytes away (∼22–60 KB), an ${ \sim } 8 \times$ increase in access stride. This causes L1/L2 cache thrashing, poor memory coalescing (adjacent warp threads need data from diferent frame blocks), and higher DRAM trafic as temporal loads cannot be served from cache.

Tiled kernel constraints. Our custom tiled kernel gains eficiency from cooperative loading within a single frame block, where spatially nearby tokens share locality. A temporal kernel would require 32 threads to fetch from up to 32 distinct frame blocks simultaneously, causing severe warp divergence and destroying the coalesced access pattern.

Reduced kernel utility under global TopK. A sparse temporal kernel would be less useful than its spatial counterpart. With global TopK at $k _ { r } { = } 0 . 6 .$ the probability that a given spatial position (h, w) survives selection in both frames t and t+1 is approximately $\bar { k } _ { r } ^ { 2 } \approx 0 . 3 6$ . In contrast, spatial neighbors within the same frame share the same selection mask, so a kept token’s 8 spatial neighbors have a much higher survival probability, making the spatial kernel’s computation substantively meaningful.

In summary, while a native sparse 1D temporal convolution is algorithmically possible, every aspect of its GPU implementation, gather patterns, memory locality, tiling strategy, and efective kernel utilization, is fundamentally worse than the spatial 2D case. The marginal mAP advantage of temporal convolution (+0.25%) does not justify the performance degradation of a native sparse temporal kernel. Our 2D spatial design provides a direct path from token reduction ratios to proportional wall-clock savings.