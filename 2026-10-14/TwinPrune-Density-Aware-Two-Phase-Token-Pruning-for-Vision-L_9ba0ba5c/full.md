# TwinPrune: Density-Aware Two-Phase Token Pruning for Vision-Language Models

Kai Liu<sup>1∗</sup>, Anqi Li<sup>1∗</sup>, Junxian Li<sup>1</sup>, Zhixin Wang<sup>2</sup>, Zhikai Chen<sup>2</sup>, Renjing Pei<sup>2</sup>, Yulun Zhang<sup>1†</sup> <sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>Huawei Technologies Ltd

![](images/ea8e0eec48c704041d069bdb4300c93b0c92982def0d8491e1138279c0d7471a.jpg)

![](images/8b971182b6ca042f25da7496ee131b0e45db7c803bea39c88d506b69c207fe1c.jpg)  
Figure 1: Two-Stage Pruning in TwinPrune. TwinPrune improves VLM inference through a two-stage pruning framework using a single unified pruning signal. The left panel evaluates visual token pruning in the prefilling stage, showing better accuracy–speedup trade-offs, while the right panel evaluates KV cache pruning on long-output benchmarks in the decode stage. Both panels report results on InternVL-3.5-8B under the same evaluation setting.

## Abstract

We propose a unified spatial-density signal that scores token importance for both phases of inference, instantiated as a bipartite merge during prefill and a KV eviction during decode, with measurable gains in each phase. To verify the two phases independently, we partition benchmarks into two diagnostic roles by their scoring rule. Prefill probes such as MMBench, POPE, and ScienceQA score only the first decoded token, so their accuracy reflects only the argmax of prefill logits, computed before any decode-phase KV eviction takes effect. Full-pipeline probes such as TextVQA, DocVQA, and MM-Vet score the full decoded sequence and therefore reveal both phases. We expose a fundamental evaluation flaw: published claims of decode-phase token pruning validated only on prefill probes have measured prefill quality, not decode quality. Empirically, removing all visual KV at decode leaves prefill-probe accuracy essentially unchanged while collapsing TextVQA and MM-Vet by tens of points, and the insensitivity is a property of the scoring rule, not of the compression strength. Across seven model variants from the InternVL-3.5, Qwen3-

VL, and LLaVA-1.5 families and the three full-pipeline-probe benchmarks, we deliver an honest three-axis Pareto over compute, memory, and accuracy. Densitybased prefill compression is competitive on the accuracy frontier, scales linearly in the visual-token count, and is roughly two orders of magnitude faster in token selection at high resolution. On the decode side, density top-k KV eviction wins every TextVQA and MM-Vet cell against the strongest published baselines H2O, SnapKV, and StreamingLLM across three keep ratios. The advantage widens to 9.6 percentage points over H2O at 12.5% keep on TextVQA and holds 7 to 10 percentage points across the InternVL-3.5 series from 2B to 38B. Compression also scales gracefully with model size: every matched 8B-to-38B transition we measure reduces, rather than amplifies, the accuracy loss. We recommend MM-Vet or DocVQA as the minimum full-pipeline-probe test for any decode-phase claim. Our code is available at https://github.com/Kai-Liu001/TwinPrune.

## 1 Introduction

Vision-Language Models (VLMs) now drive a fast-growing set of applications such as document understanding, embodied agents, multi-step visual reasoning, and GUI control. Each of these demands processing hundreds to thousands of visual tokens through tens of transformer layers. As image resolution and tile count rise, two efficiency bottlenecks tighten in tandem. The first is prefill compute, dominated by encoding all visual tokens through the language model. The second is decode memory, dominated by the per-layer key/value cache that must persist for autoregressive generation. Whether a model can be deployed at all, whether on a memory-constrained device, in a low-latency interactive setting, or under a high-throughput serving regime, is increasingly determined less by how the model is trained and more by how aggressively redundant visual tokens can be removed at inference time.

The token-pruning literature has expanded rapidly in response. Prefill-time methods such as FastV [6], ToMe [4], DivPrune [1], and DART [28] target the compute axis, while decode-time methods such as H2O [33], StreamingLLM [29], and SnapKV [13] target the memory axis. These works converge on a striking shared finding: aggressive compression preserves accuracy almost perfectly, often within a fraction of a point of the uncompressed baseline. The implication, repeated across dozens of papers, is that visual tokens are deeply redundant and can be cut without harm. If that picture were correct, token pruning would be a solved problem, a routine deployment knob rather than an open question.

We argue this consensus rests on a measurement artifact. The benchmarks most commonly used to validate VLM pruning are dominated by single-token outputs such as a multiple-choice letter or a yes/no answer. Their score depends only on the first decoded logit, which is fixed by the prefill pass before any decode-phase KV eviction is executed. By construction, decode-phase eviction therefore cannot move those scores, so what is measured under a “decode-phase” label is in fact prefill-phase quality. Whether a method preserves behavior under longer generation, in which information has to be retrieved from the cache during decode, is a separate question, and it is not the one those benchmarks answer. The same critique applies in a milder form to prefill compression: prefill-probe scores see only the argmax shift, not the distributional damage that compression does to longer continuations.

We make the gap concrete by running a VLM with all visual KV removed at decode time. On the prefill-probe benchmarks (MMBench, POPE, ScienceQA) the accuracy is nearly unchanged. On full-pipeline-probe benchmarks such as TextVQA and MM-Vet, accuracy drops sharply, and prediction strings flip on a large fraction of samples (Tab. 1, Fig. 2, Sec. 3). The model is changing its behavior under compression, but the prefill-probe benchmarks simply cannot register that change.

This re-scopes what an honest unified compression framework should look like and what it should be evaluated on. We propose a spatial-density signal ρ that scores each visual token by the local non-redundancy of its w×w neighborhood on the regular spatial grid, computed once at an early decoder layer from token-to-token cosine distances. The same ρ drives both compression phases. During prefill, we deploy it inside a density-tilted farthest-point recurrence that combines k-center coverage with density priority through a single tilt scalar β, bipartitions the visual tokens, and merges the discarded-token features into the kept set before passing the reduced sequence forward. During decode, the same ρ values are reused as a top-k score for visual-KV eviction, with no new attention pattern computed. The two phases can be applied independently or jointly, and we supply a three-axis (compute, memory, accuracy) Pareto study across seven model variants and three benchmarks.

Empirically, the unified density signal is competitive on the prefill accuracy frontier and is roughly two orders of magnitude faster in token selection at high resolution, and the implementation, with a Triton-fused farthest-point inner loop and a cached CUDA graph, delivers a 3.24× end-to-end prefill speedup at the most aggressive operating point on InternVL-3.5-8B. On the decode side, density top-k KV eviction wins every TextVQA and MM-Vet cell at three keep ratios against H2O, SnapKV, and StreamingLLM, with the gap widening to 9.6 percentage points over H2O at 12.5% keep on TextVQA and holding 7 to 10pp across the InternVL-3.5 series from 2B to 38B. Compression cost does not grow with model size in the 8B-to-38B regime. Our contributions are as follows:

1. Methodology finding (Sec. 3): prefill-probe benchmarks (those scoring only the first decoded token) structurally cannot measure decode-phase compression. We provide the proof, an extended cross-architecture mismatch table, and a recommendation for how the field should evaluate decode-phase claims going forward.

2. Unified density framework (Sec. 4): a single spatial-density signal that scores token importance for both compression phases, with one kernel and two deployment points, guiding bipartite merge during prefill and top-k eviction during decode. We also derive a coreset-theoretic justification (Sec. 4.1) showing why the same ranking serves both phases.

3. Three-axis Pareto study (Sec. 5 and Sec. 6): across seven model variants and three full-pipeline-probe benchmarks, density-based prefill compression matches the strongest accuracy frontier while scaling linearly in the visual-token count and selecting tokens an order of magnitude faster at high resolution.

4. Cross-architecture and cross-scale robustness (Appendix P): a structural risk taxonomy explaining why pure-selection methods catastrophically fail on Qwen3-VL-4B’s multiaxis RoPE while merge-based density does not, together with the empirical finding that compression cost does not grow with model size across the InternVL-3.5 from 2B to 38B.

## 2 Related Work

Prefill-stage compression. FastV [6] drops visual tokens by attention score after a chosen layer. ToMe [4] performs bipartite token merging for ViTs and was adapted to VLMs. DivPrune [1] uses farthest-point sampling on visual features for diversity-driven selection. DART [28] combines L1-norm pivots with cosine-distance expansion. VisionZip [30] selects dominant tokens by CLSattention and merges the rest by key-state similarity. TokenPacker [11] introduces a coarse-to-fine visual projector that compresses tokens by 75 to 89 percent before the LLM backbone. AvgPool, FitPrune [31], and PruMerge [23] round out a crowded landscape, and concurrent preprints continue to appear, including SAINT [9]. All evaluate primarily on the four-benchmark set we critique below.

Decode-stage KV eviction. For text-only LLMs, H2O [33] and StreamingLLM [29] maintain “heavy hitter” KV entries. SnapKV [13] aggregates attention. FastGen [8] profiles attention-head patterns to build a query-adaptive eviction policy. Quest [25] selects KV pages by query similarity for efficient long-context inference. KV quantization, exemplified by the 2-bit asymmetric KIVI [16], provides a complementary compression axis orthogonal to eviction. On the visual side, LOOK-M [26] merges multimodal KV entries guided by text-prior attention. ElasticCache [17] uses importancedriven KV merging for vision instruction-following models. R-KV [5] combines attention aggregation with a redundancy penalty for reasoning-LLM KV compression. Most works report on the same prefill-probe benchmarks that, as we show, cannot measure their decode-phase effect.

Two-phase view. To our knowledge, no prior work explicitly separates prefill (compute) and decode (memory) as orthogonal axes with distinct algorithms and evaluation requirements. Our framework makes this separation a first-class design principle.

Token pruning as coreset selection. Keeping k of N visual tokens is a coreset query on X ∈ R<sup>N×d</sup>, with two dual classical strategies: k-center diversity (DivPrune [1], Sener and Savarese [22]) and density- or sensitivity-weighted sampling (Feldman and Langberg [7], Bachem, Lucic, and Krause [2], submodular maximization [19]). Our density signal ρ is a training-free, deterministicthreshold proxy of the latter family (Appendix H). We are not aware of prior work framing VLM token pruning as a two-phase coreset whose subset must simultaneously approximate prefill attention and decode KV recall, and this paper is the first to exploit the resulting signal reuse.

![](images/b8c25d948b4fa3b1eb454f9b41a1ae50cfd5243fe3a94a3ee4b67a240d8931d5.jpg)  
(a) Acc vs decode KV keep

![](images/3d88ba8b85c37aeb9f28ca63d5030551aca1699b346dc857c5db88b8b6cd54f4.jpg)  
(b) Cross-arch mismatch  
Figure 2: Prefill-probe benchmarks are structurally insensitive to decode-phase KV eviction across three VLM families. Left: accuracy vs. visual KV kept on InternVL-3.5-8B. Right: prediction mismatch at DALL grouped by (benchmark, model). Details in Sec. 3.

Table 1: Prediction-mismatch $\%$ vs. no-compression baseline on InternVL-3.5-8B. Decode KV eviction flips up to 46% of TextVQA predictions but $\leq 0 . 5 4 \%$ on prefill-probe benches, since $y _ { 1 }$ depends only on prefill logits (Eq. 1). Prefill rows fix L=15 (varying $L$ shifts levels but not the scoring-rule gap). Appendix N extends to all 3 archs.
<table><tr><td>Phase</td><td>Signal ·Keep</td><td>MMBench</td><td>POPE</td><td>SQA</td><td>TextVQA</td></tr><tr><td>Prefill</td><td>Density ·75/50/25%</td><td>0</td><td>0-0.03</td><td>一</td><td>3.1/7.0/14.9</td></tr><tr><td>Prefill</td><td>Random·25%</td><td>0</td><td>0</td><td></td><td>32.8</td></tr><tr><td>Prefill</td><td>Attention·25%</td><td>0</td><td>0.03</td><td></td><td>25.1</td></tr><tr><td>Decode</td><td>Density · 25/12.5/6.25%</td><td>0.54</td><td>0.32-0.35</td><td>0.20</td><td>15.9/23.0/31.1</td></tr><tr><td>Decode</td><td>Density · 0% (drop all)</td><td>0.54</td><td>0.38</td><td>0.20</td><td>45.5</td></tr></table>

## 3 Two-Phase Compression, Two-Phase Diagnostic

## 3.1 The argmax mechanism

Consider a VLM prefill that yields hidden states $H \in \mathbb { R } ^ { T \times d }$ over T tokens (visual + text). The first decoded token is

$$
y _ { 1 } = \arg \operatorname* { m a x } _ { v } \left[ W _ { o } \ : h _ { T } \right] _ { v } ,\tag{1}
$$

where $h _ { T }$ is the last hidden state and $W _ { o }$ is the language-model head. Decode-phase compression, for example KV eviction E acting on the cached $\{ ( K _ { \ell } , V _ { \ell } ) \}$ , takes effect only when computing y<sub>2</sub>, y<sub>3</sub>, . . . from the now-reduced cache. For benchmarks whose ground-truth answer is a single token, such as a multiple-choice letter or a yes/no token, the score depends only on $y _ { 1 }$ . Thus decode-phase E has no effect on these scores by construction. Density-based compression operates on both prefill and decode, with measurable gains in each phase (Sec. 5 and Sec. 6). To verify the two phases independently, we partition benchmarks into two diagnostic roles based on their scoring rule. A benchmark is a prefill probe if its score depends only on $y _ { 1 }$ , as in a multiple-choice letter, a yes/no token, or any single-letter argmax over a fixed set such as MMBench, POPE, and ScienceQA. Such a probe isolates prefill quality and is structurally blind to any decode-phase change. A benchmark is a full-pipeline probe if its score depends on the decoded sequence $y _ { 1 } , y _ { 2 } , \dotsc , $ including free-form answers scored by exact-match, ANLS edit distance, or an LLM judge such as TextVQA, DocVQA, and MM-Vet. Such a probe reveals both phases and their interaction effects. The split is a property of the scoring rule, not the literal answer length during generation.

## 3.2 Empirical proof

We instantiate $\mathcal { E }$ ranging from D75 (keep 25% of visual KV) to DALL (drop all visual KV). Table 1 reports prediction mismatch % relative to the no-compression baseline.

![](images/7bd4ebe9e9e85a1b8cce3d4ce252a3b5b97ad823146cb35d86f537b8f9ab3b33.jpg)  
Figure 3: TwinPrune pipeline. A single spatial density signal $\rho ,$ defined as each visual token’s average cosine dissimilarity to its w×w neighborhood (b), drives both phases. Prefill bipartite-merges low $- \rho$ source tokens into high- $- \rho$ destination anchors, cutting FLOPs (c). Decode ranks visual KV by $\rho$ and keeps the top-k with original RoPE positions, cutting KV memory (d). One signal, two levers, and a three-axis Pareto trade-off over compute, KV, and accuracy (a).

Implication. The pattern is universal across architectures: TextVQA’s 45.52% mismatch under DALL (Tab. 1) refutes the “tied logits” objection, and Appendix N extends the cross-model grid to confirm full-pipeline probes register both phases while prefill probes register neither in practice. Any claim of decode-phase token pruning that preserves accuracy validated only on MMBench, POPE, or ScienceQA must be re-examined; future VLM token-pruning evaluation should include at least one full-pipeline-probe benchmark such as MM-Vet, DocVQA, or LLaVA-Bench-Wilder.

## 4 Density-Guided Two-Phase Compression

## 4.1 Spatial density signal

For visual tokens arranged on a grid of size $H \times W$ , define density at position $( i , j )$ via local cosine dissimilarity:

$$
\rho _ { i j } = 1 - \frac { 1 } { | \mathcal { N } _ { w } ( i , j ) | } \sum _ { ( i ^ { \prime } , j ^ { \prime } ) \in \mathcal { N } _ { w } ( i , j ) } \frac { \langle x _ { i j } , x _ { i ^ { \prime } j ^ { \prime } } \rangle } { \| x _ { i j } \| \cdot \| x _ { i ^ { \prime } j ^ { \prime } } \| } ,\tag{2}
$$

where $\mathcal { N } _ { w } ( i , j )$ is the spatial $w \times w$ neighborhood (excluding self). We compute $\rho$ at multiple scales $w \in \{ 3 , 5 , 7 \}$ and take the max, capturing both fine and coarse structure. High $\rho _ { i j }$ indicates that the token differs from its neighbors and is therefore informative, whereas low $\rho _ { i j }$ indicates redundancy.

Why density? Density is an intrinsic, query-invariant feature-tensor statistic. It is monotone in a finite-sample kernel- $\cdot H _ { 2 }$ estimator [21], contrasting the luminance-biased $\ell _ { 2 } \cdot$ -norm and the sink-biased, query-dependent attention scores. Section 5.3 confirms density wins or ties at prefill across all three probe benchmarks and wins on the two deterministic full-pipeline probes at decode (information-geometric framing and the failed QC-density extension are in Appendix F).

Coreset reading. A coreset that approximates the visual-token feature matrix approximates every downstream computation that accesses it through attention aggregates, so prefill merge and decode eviction inherit the same approximation guarantee from a single density top-k subset. Theorem 1 in Appendix H bounds the approximation error by $\varepsilon \leq 2 e Q r _ { k } / \sqrt { d }$ where $r _ { k }$ is the k-center cover radius of the chosen subset. Density top-k minimises $r _ { k }$ in expectation while farthest-point sampling minimises it by construction, which explains why the two greedy heuristics agree at mild compression and diverge at aggressive compression on skewed distributions (Sec. 5). A Feldman–Langberg-style $1 / \sqrt { k }$ rate (Theorem 2) and the full proofs are in Appendix H.

## 4.2 Prefill: density-tilted farthest-point selection

At a chosen insertion layer L during prefill, we keep $k = \lceil N \cdot ( 1 { - } r ) \rceil$ of the N visual tokens by a density-tilted greedy selection. Compute density $\rho$ on the layer-L hidden states via Eq. 2, normalize to $\tilde { \rho } \in [ 0 , 1 ]$ , and pre-compute the pairwise cosine distance matrix $D _ { i j } \ = \ 1 - \left. \hat { x } _ { i } , \hat { x } _ { j } \right.$ where $\hat { x } = x / \| \hat { x } \|$ . Seed with $s _ { 0 } = \arg \operatorname* { m a x } _ { i } \rho _ { i } ,$ the most informative token, and maintain min $\bar { \mathrm { i } _ { j } } = D _ { s _ { 0 } j }$ as the minimum distance from $j$ to the selected set. For each subsequent step, pick

$$
s _ { t } = \arg \operatorname* { m a x } _ { j \notin S } \big ( \mathrm { m i n d } _ { j } + \beta \widetilde { \rho } _ { j } \big ) , \quad \mathrm { t h e n } \quad \mathrm { m i n d } _ { j } \gets \mathrm { m i n } ( \mathrm { m i n d } _ { j } , D _ { s _ { t } j } ) ,\tag{3}
$$

with $\beta = 0 . 3$ throughout. Selected indices are sorted to preserve the visual-token-grid ordering, and KV-cache rows for the unselected tokens are dropped before the next layer. Eq. 3 unifies two greedy heuristics for the same coreset, as discussed in Sec. 4.1. Pure farthest-point sampling at $\beta { = } 0$ recovers DivPrune [1], and density top-k is the limit $\beta \to \infty$ . The small additive $\beta = 0 . 3$ tilt lets coverage dominate at high prune ratios while density breaks ties when mind<sub>j</sub> saturates, which is the regime where pure FPS picks isolated outliers with low local relevance.

Implementation. The k-step inner loop is the latency-critical path of density-tilted FPS. A naive cuBLAS-plus-graph realisation issues five kernel launches per step, and at our operating point $( N { = } 3 0 7 2 , D { = } \mathrm { \bar { 4 } 0 \bar { 9 } 6 } , K { = } 6 1 4 )$ the per-launch overhead dominates the actual compute. We fuse the body into two Triton kernels, an argmax reduction and an inner-iter mind update with a $3 2 \times 6 4$ rowcolumn tile, and back the recurrence with an fp16 normalized-token buffer. fp16’s 10-bit mantissa resolves $1 0 ^ { - 3 }$ differences in cosine similarity, comfortably finer than the random-vector cosine standard deviation $1 / \sqrt { D } { = } 0 . 0 1 6 \mathrm { a t } D { = } 4 0 9 6$ . The full recurrence is captured into a CUDA graph keyed on $( N , D , k , \beta )$ , replayed once per layer rather than re-launched per step, and cached at module scope across the sweep. Without this cache, the per-call capture overhead of roughly 80 ms inverts the legacy-versus-graph latency ranking at our K. The combination delivers a 3.24× end-to-end prefill speedup at the B3 cell on InternVL-3.5-8B, with cached-Triton fp16 TwinPrune at 150.8 ms per sample versus the 487.7 ms uncompressed baseline.

## 4.3 Decode: density-guided KV eviction

After prefill completes, with or without prefill-stage compression, evict visual KV entries by density rank in a single shot. We keep $k _ { \mathrm { d e c } } = \bar { \lceil } \bar { N } _ { \mathrm { p o s t } } \cdot r _ { \mathrm { d e c o d e } } \rceil$ entries with the highest $\rho ,$ where $N _ { \mathrm { p o s t } }$ is the post-prefill visual-token count. Surviving KV rows retain their original RoPE-rotated keys without rotation re-application, and the first decoded token is positioned at original\_prefill\_length+1 to maintain query and key alignment. Density is reused from prefill when both phases are active, so the decode stage adds no extra similarity computation. Eviction is one-shot at the prefill-decode boundary, in contrast to streaming approaches that maintain a sliding window during generation. The choice is justified by the coreset reading in Sec. 4.1: a density-coreset of the post-prefill visual tokens approximates every subsequent attention aggregate that queries those tokens.

## 4.4 Joint and the three-axis Pareto

Joint compression composes prefill merge with decode eviction. The resulting design space is naturally three-dimensional: (i) prefill FLOPs (reduced by merge), (ii) KV memory (reduced by eviction), (iii) accuracy. Single-phase methods occupy the axes, and joint configurations fill the interior. We characterize this Pareto surface in Sections 5 and 6, and Algorithm 1 in Appendix A gives the full pseudocode.

## 5 Prefill compression Pareto

## 5.1 Setup

We evaluate seven model variants from three VLM families: InternVL-3.5 at 2B, 4B, 8B, and 38B [27], Qwen3-VL at 2B and 4B [3], and LLaVA-1.5-7B [14]. Prefill-probe benchmarks (MMBench [15],

Table 2: Per-benchmark prefill compression on InternVL-3.5-8B; same $( L , r )$ within each block. Bold marks per-column best; daggers (†) flag structurally invalid output (acc ≈ 0, App. S). Block 4 in App. V. Speedup: measured prefill speedup vs. uncompressed baseline at batch=8, HD mode (12 tiles, $N _ { \mathrm { v i s } } { \stackrel { - } { = } } 3 0 7 2 { \bar { ) } }$ on RTX A6000.
<table><tr><td>Method</td><td>Venue</td><td>TPL %</td><td>MMBench</td><td>POPE</td><td>SQA</td><td>TextVQA</td><td>Avg-4</td><td>Speedup</td></tr><tr><td colspan="2">Baseline (no compression)</td><td>100.0</td><td>0.8707</td><td>0.8903</td><td>0.9539</td><td>0.6493</td><td>0.8411</td><td>1.00×</td></tr><tr><td colspan="9">Block 1: low-TPL — L = 15, prune 50%, TPL :</td></tr><tr><td>Random</td><td></td><td>65.2</td><td>65.2% 0.8862</td><td>0.9155</td><td>0.9782</td><td>0.6202</td><td>0.8500</td><td>1.28×</td></tr><tr><td>ToMe</td><td>ICLR&#x27;23 [4]</td><td>65.2</td><td>0.8584</td><td>0.8913</td><td>0.9346</td><td>0.5325</td><td>0.8042</td><td>1.16×</td></tr><tr><td>FastV</td><td>ECCV’24 [6]</td><td>65.2</td><td>0.8584</td><td>0.8868</td><td>0.9326</td><td>0.5150</td><td>0.7982</td><td>1.28×</td></tr><tr><td>PruMerge</td><td>ICCV’25 [23]</td><td>65.2</td><td>0.8584</td><td>0.8943</td><td>0.9410</td><td>0.5937</td><td>0.8219</td><td>1.21×</td></tr><tr><td>DivPrune</td><td>CVPR’25 [1]</td><td>65.2</td><td>0.8630</td><td>0.8923</td><td>0.9430</td><td>0.6017</td><td>0.8250</td><td>1.17×</td></tr><tr><td>DART</td><td>EMNLP’25 [28]</td><td>65.2</td><td>0.8467</td><td>0.8877</td><td>0.9316</td><td>0.6026</td><td>0.8171</td><td>1.28×</td></tr><tr><td>CAC (ours)</td><td></td><td>65.2</td><td>0.8909</td><td>0.9142</td><td>0.9792</td><td>0.7195</td><td>0.8759</td><td>1.16×</td></tr><tr><td colspan="9">Block 2: extreme stress — L =3, prune 85%, TPL = 24.1%</td></tr><tr><td>Random ToMe</td><td>ICLR&#x27;23</td><td>24.1 24.1</td><td>0.7980 0.8320</td><td>0.8398 0.8798</td><td>0.8622 0.8860</td><td>0.3285 0.4524</td><td>0.7071 0.7626</td><td>3.12× 2.07×</td></tr><tr><td>FastV</td><td>ECCV’24</td><td>24.1</td><td>0.7423</td><td>0.7665</td><td>0.7833</td><td>0.1939</td><td>0.6215</td><td>3.11×</td></tr><tr><td>PruMerge</td><td>ICCV’25</td><td>24.1</td><td>0.8313</td><td>0.8578</td><td>0.8825</td><td>0.3866</td><td>0.7396</td><td>2.47×</td></tr><tr><td>DivPrune†</td><td>CVPR&#x27;25</td><td>24.1</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0041</td><td>0.0010</td><td>2.98×</td></tr><tr><td>DART†</td><td>EMNLP&#x27;25</td><td>24.1</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0041</td><td>0.0010</td><td>3.09×</td></tr><tr><td>CAC (ours)</td><td></td><td>24.1</td><td>0.8282</td><td>0.8670</td><td>0.8929</td><td>0.5696</td><td>0.7894</td><td>2.67×</td></tr><tr><td colspan="9">Block 3: visual-only pure prune — L = 0, prune 80%, TPL = 20.0%</td></tr><tr><td>Random</td><td></td><td>20.0</td><td>0.8297</td><td>0.8565</td><td>0.8751</td><td>0.3665</td><td>0.7320</td><td>3.57×</td></tr><tr><td>ToMe</td><td>ICLR&#x27;23</td><td>20.0</td><td>0.8351</td><td>0.8893</td><td>0.8875</td><td>0.4599</td><td>0.7679</td><td>2.31×</td></tr><tr><td>FastV</td><td>ECCV’24</td><td>20.0</td><td>0.8320</td><td>0.8057</td><td>0.9043</td><td>0.6149</td><td>0.7892</td><td>3.58×</td></tr><tr><td>PruMerge</td><td>ICCV’25</td><td>20.0</td><td>0.8297</td><td>0.8715</td><td>0.8875</td><td>0.3399</td><td>0.7321</td><td>2.79×</td></tr><tr><td>DivPrune†</td><td>CVPR’25</td><td>20.0</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0041</td><td>0.0010</td><td>3.11×</td></tr><tr><td>DART†</td><td>EMNLP&#x27;25</td><td>20.0</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0041</td><td>0.0010</td><td>3.30×</td></tr><tr><td>CAC (ours)</td><td></td><td>20.0</td><td>0.8467</td><td>0.8835</td><td>0.8914</td><td>0.5793</td><td>0.8003</td><td>3.24×</td></tr><tr><td colspan="9">Block 3b: + LoRA recovery, fair control — L = 0, prune 70%, TPL = </td></tr><tr><td>Random + LoRA</td><td></td><td>30.0</td><td>0.8615</td><td>0.8983</td><td>30.0% 0.9142</td><td>(matched) 0.4512</td><td>0.7813</td><td>1.32×</td></tr><tr><td>DivPrune + LoRA</td><td>CVPR&#x27;25</td><td>30.0</td><td>0.8607</td><td>0.9060</td><td>0.9355</td><td>0.5981</td><td>0.8251</td><td>0.92×</td></tr><tr><td>DART + LoRA</td><td>EMNLP&#x27;25</td><td>30.0</td><td>0.8475</td><td>0.8837</td><td>0.9346</td><td>0.6149</td><td>0.8202</td><td>1.83×</td></tr><tr><td>CAC + LoRA (ours)</td><td></td><td>30.0</td><td>0.8514</td><td>0.9058</td><td>0.9321</td><td>0.6385</td><td>0.8319</td><td>2.14×</td></tr></table>

POPE [12], ScienceQA [18]) score the first decoded token, isolating prefill quality. Full-pipelineprobe benchmarks reveal both phases. The full-pipeline-probe set comprises TextVQA [24] (n=5000, exact-match), MM-Vet [32] (n=218, free-form, official MM-Vet evaluator), and DocVQA (n=5349, deterministic ANLS scoring). On the prefill side we compare density-based merging (TwinPrune and a variant TwinPrune-2a, ours) against DivPrune [1], DART [28], FastV [6], ToMe [4], and PruMerge [23]. On the decode side, we compare density (ours) against random and attention-valuenorm baselines. We sweep prefill prune ∈ {25, 50, 75}% at insertion layers {0, 1, 2, 3, 5, 10, 15, 20} and decode keep ∈ {75, 50, 25, 12.5, 6.25, 0}%.

Configuration shorthand. We refer to specific compression configurations by short codes throughout. TwinPrune \_LL\_r denotes prefill-only TwinPrune compression inserted at LLM layer L with visual-token prune ratio r%, so TwinPrune \_L3\_50 is prefill-only compression at layer 3 pruning 50% of visual tokens. decode\_only\_Dd denotes decode-only KV eviction that drops d% of the visual KV span after prefill, so D75 keeps 25% of visual KV, D875 keeps 12.5%, and Dall drops every visual KV. joint\_Pp\_Dd denotes joint compression that combines prefill-merge prune p% at layer 3 with decode-eviction keep (100−d)%, so joint\_P50\_D75 is layer-3 prefill prune 50% paired with decode keep 25%.

![](images/9904b05dc021d7fdc0b45339d6faab4b2baeb34fd79d36f248affff3a9c75a5b.jpg)

![](images/cca75d8f75ca56d0136bb6dd89f291c9988eff4e79e189fc5aa22b8f80275e9e.jpg)  
Figure 4: Decode KV Eviction on TextVQA. On InternVL-3.5-8B, density-guided eviction Paretodominates random and attention-value-norm in the memory–accuracy trade-off. Right inset: Qwen3- VL-4B, LLaVA-1.5-7B, and InternVL-3.5-38B show the same monotonic density trend.

## 5.2 Prefill compression Pareto

Table 2 reports per-benchmark accuracies on InternVL-3.5-8B under a fair-comparison protocol where each block fixes a single $( L , r )$ pair shared by all methods, isolating the selection-rule difference. Three blocks in the main paper span the TPL spectrum. Block B1 uses $L = 1 5$ with prune 50% and TPL≈ 65%. Block B2 uses $L = 3$ with prune 85% and $\mathrm { T P L } \approx 2 4 \%$ , an extreme stress setting. Block B3 uses $L = 0$ with prune 80% and TPL≈ 20%, a visual-only pure-prune setting. The saturated high-TPL block B4 at $L = 2 0$ , prune 50%, TPL≈ 86% is reported in Appendix V. The matched cross-architecture sweep on Qwen3-VL-4B uses the same four-block schedule with M-RoPE 3-axis position encoding and is reported in Appendix Q as Tab. 15.

Headline. TwinPrune leads Avg-4 in every block on InternVL-8B. The lead is +2.6pp at the moderate B1 (L = 15, prune 50%) and grows to +1.1pp over the next healthy method at the visual-only B3 (L = 0, prune 80%), where pure-selection methods DivPrune and DART collapse to acc ≈0. Under the fair LoRA control at matched TPL= 30% (B3b), TwinPrune+LoRA reaches Avg-4 = 0.8319 vs DivPrune+LoRA at 0.8251, with a +4.0pp gap on TextVQA. Block-by-block reading and the matched Qwen3-VL-4B sweep are in Appendix D. Selection-side complexity is in Appendix B.

## 5.3 Signal ablation

Is the advantage of our method really density, or could a simpler score match it? Holding every pipeline component fixed (L=15, prune 50%, InternVL-3.5-8B, prune-only top-k) and swapping only the ranking score across four benchmarks (TextVQA with n=5000, MMBench, V<sup>∗</sup>Bench, MM-Vet), density wins or ties on all four: it outperforms L2 norm on TextVQA by 2.2 percentage points (0.616 vs. 0.594), attention V-norm on TextVQA by 3.2 percentage points, and V-norm on V<sup>∗</sup>Bench by 4.7 percentage points. MMBench scores cluster near 0.86 because the prefill-probe scoring rule is insensitive to which visual tokens are kept (Sec. 3). Multi-scale density is nearly indistinguishable from single-scale, differing by 0.03 percentage points on TextVQA, and the product ρ·∥v∥ is neutral-to-hurt: the two signals are not independent, so combining them injects noise. Full four-benchmark table in Appendix I.

## 6 Decode compression on full-pipeline-probe benchmarks

## 6.1 Decode KV eviction Pareto (Figure 4)

Density-guided eviction outperforms random selection by 5 to 11 percentage points at aggressive (75%) eviction and outperforms the value-norm proxy by 4 to 8 percentage points, consistently across all six tested models. The signal is robust under cross-architecture variation and, as Section 5.3 shows, outperforms L2-norm by 2.2 percentage points on TextVQA at matched configuration.

Table 3: Decode-phase KV eviction comparison on InternVL-3.5-8B at three keep ratios. Density top-k wins all six TextVQA and MM-Vet cells. On DocVQA, SnapKV narrowly leads at keep ≥ 25% but density reclaims the lead at 12.5% keep. Bold marks per-column best. TextVQA uses n=5000 exact-match. MM-Vet uses n=218. DocVQA uses n=5349 with ANLS scoring.
<table><tr><td rowspan="2">Method (keep k)</td><td colspan="3">TextVQA</td><td colspan="3">MM-Vet</td><td colspan="3">DocVQA</td></tr><tr><td>50%</td><td>25%</td><td>12.5%</td><td>50%</td><td>25%</td><td>12.5%</td><td>50%</td><td>25%</td><td>12.5%</td></tr><tr><td>Baseline (no eviction)</td><td></td><td>0.733</td><td></td><td></td><td>0.756</td><td></td><td></td><td>0.642</td><td></td></tr><tr><td>StreamingLLM [29]</td><td>0.637</td><td>0.573</td><td>0.507</td><td>0.667</td><td>0.561</td><td>0.514</td><td>0.468</td><td>0.384</td><td>0.324</td></tr><tr><td>H2O [33]</td><td>0.697</td><td>0.629</td><td>0.552</td><td>0.662</td><td>0.571</td><td>0.497</td><td>0.555</td><td>0.438</td><td>0.347</td></tr><tr><td>SnapKV [13]</td><td>0.707</td><td>0.643</td><td>0.573</td><td>0.654</td><td>0.585</td><td>0.513</td><td>0.577</td><td>0.469</td><td>0.377</td></tr><tr><td>Density (ours)</td><td>0.726</td><td>0.697</td><td>0.648</td><td>0.708</td><td>0.634</td><td>0.551</td><td>0.575</td><td>0.466</td><td>0.382</td></tr></table>

Comparison with published decode-eviction baselines. Table 3 extends the comparison to H2O [33], SnapKV [13], and StreamingLLM [29] on the visual-KV span at three keep ratios. Density top-k wins all six TextVQA and MM-Vet cells outright, and the advantage widens at extreme compression: at keep = 12.5% density beats H2O by 9.6, 5.4, and 3.5pp on TextVQA, MM-Vet, and DocVQA respectively. SnapKV narrowly leads density on DocVQA at the looser keeps but density reclaims the lead at 12.5%. V-norm takes MM-Vet at decode by 4.8pp at keep 25% (Tab. 6); density still wins TextVQA by 4.5pp and DocVQA by 2.4pp at the same keep, and V-norm requires decode-time attention maps that break the two-phase signal reuse (Appendix I).

## 6.2 Joint compression analysis

Per-model MM-Vet scores under TwinPrune \_L3\_50, decode\_only\_D75, and joint\_P50\_D50 are in Tab. 8 (Appendix J); both single phases cost 10 to 18pp, with no free lunch. The 3×3 dense grid (Tab. 10, Appendix L) confirms a smooth bilinear surface in the interior: accuracy falls monotonically along both axes, and joint configurations are legitimate budget trades but never Pareto-dominate single-phase methods at matched KV budgets. A complementary 3-prefill×3-decode-signal sweep (Appendix K) shows density’s lead over H2O is concentrated in the decode-only regime; once aggressive prefill at L=3, r=85% has removed the redundant visual signal, the three decode signals collapse within 1pp on TextVQA.

## 7 Discussion and Conclusion

Summary and deployment. VLM token pruning has been evaluated on a benchmark mix that limits visibility into decode-phase effects. We provide the proof, propose a unified density-based framework that scores token importance for both phases, and supply a three-axis Pareto across seven model variants and three full-pipeline-probe benchmarks. We further identify a structural risk class explaining when pure-selection methods fail on multi-axis RoPE backbones, and we show compression cost does not grow with model size in the 8B-to-38B regime. In deployment, latencybound use cases use prefill compression (3.24× speedup at the B3 cell), memory-bound use cases use decode eviction, and joint configurations apply when both axes are constrained.<sup>3</sup>

Limitations. We identify two principled limitations. (i) Single-stage prefill compression, with multi-stage scheduling unexplored. (ii) Spatial-grid density assumes regular tokenization, so non-grid tokenizers such as Q-Former require alternative formulations.

Broader impact and future work. Efficient VLM serving lowers energy use and enables edge deployment. Compression can subtly alter behavior on long-tail or safety-critical inputs without changing prefill-probe scores, so high-stakes deployments must validate on full-pipeline-probe benchmarks. Future work: learning the optimal density-aggregation kernel, and task-conditioned signals beyond pre-attention geometry (Appendix F).

## References

[1] Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. DivPrune: Diversity-based visual token pruning for large multimodal models. In CVPR, 2025. arXiv:2503.02175.

[2] Olivier Bachem, Mario Lucic, and Andreas Krause. Practical coreset constructions for machine learning. arXiv preprint arXiv:1703.06476, 2017.

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[4] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your ViT but faster. In ICLR, 2023.

[5] Zefan Cai, Wen Xiao, Hanshi Sun, Cheng Luo, Yikai Zhang, Ke Wan, Yucheng Li, Yeyang Zhou, Li-Wen Chang, Jiuxiang Gu, et al. R-kv: Redundancy-aware kv cache compression for reasoning models. In NeurIPS, 2025.

[6] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In ECCV, 2024.

[7] Dan Feldman and Michael Langberg. A unified framework for approximating and clustering data. Proceedings of the 43rd ACM Symposium on Theory of Computing (STOC), pages 569–578, 2011. arXiv:1106.1379.

[8] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive KV cache compression for LLMs. In ICLR, 2024.

[9] Ahmadreza Jeddi, Negin Baghbanzadeh, Elham Dolatabadi, and Babak Taati. Similarity-aware token pruning: Your VLM but faster. arXiv preprint arXiv:2503.11549, 2025.

[10] Michael Langberg and Leonard J. Schulman. Universal ε-approximators for integrals. In Proceedings ofthe 21st Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 598–607, 2010.

[11] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. TokenPacker: Efficient visual projector for multimodal LLM. International Journal of Computer Vision, 133:6794–6812, 2025.

[12] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, 2023.

[13] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. NeurIPS, 2024.

[14] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024.

[15] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In ECCV, 2024.

[16] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 32332–32344. PMLR, 2024.

[17] Zuyan Liu, Benlin Liu, Jiahui Wang, Yuhao Dong, Guangyi Chen, Yongming Rao, Ranjay Krishna, and Jiwen Lu. Efficient inference of vision instruction-following models with elastic cache. In ECCV, 2024. arXiv:2407.18121.

[18] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. NeurIPS, 2022.

[19] Baharan Mirzasoleiman, Ashwinkumar Badanidiyuru, Amin Karbasi, Jan Vondrák, and Andreas Krause. Lazier than lazy greedy. In Proceedings of the AAAI Conference on Artificial Intelligence, 2015. arXiv:1409.7938.

[20] Miloš Radovanovic, Alexandros Nanopoulos, and Mirjana Ivanovi´ c. Hubs in space: Popular´ nearest neighbors in high-dimensional data. Journal of Machine Learning Research, 11:2487– 2531, 2010.

[21] Alfréd Rényi. On measures of entropy and information. Proc. 4th Berkeley Symp. on Mathematical Statistics and Probability, 1961.

[22] Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations (ICLR), 2018. arXiv:1708.00489.

[23] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. LLaVA-PruMerge: Adaptive token reduction for efficient large multimodal models. In ICCV, 2025. arXiv:2403.15388.

[24] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019.

[25] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. QUEST: Query-aware sparsity for efficient long-context LLM inference. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 47901–47911. PMLR, 2024.

[26] Zhongwei Wan, Ziang Wu, Che Liu, Jinfa Huang, Zhihong Zhu, Peng Jin, Longyue Wang, and Li Yuan. LOOK-M: Look-once optimization in KV cache for efficient multimodal long-context inference. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 4065–4078, Miami, Florida, USA, November 2024. Association for Computational Linguistics.

[27] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

[28] Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. Stop looking for “important tokens” in multimodal language models: Duplication matters more. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9961–9980, Suzhou, China, November 2025. Association for Computational Linguistics.

[29] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In ICLR, 2024.

[30] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. VisionZip: Longer is better but not necessary in vision language models. In CVPR, 2025. arXiv:2412.04467.

[31] Weihao Ye, Qiong Wu, Wenhao Lin, and Yiyi Zhou. Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. In AAAI, 2025. arXiv:2409.10197.

[32] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. MM-Vet: Evaluating large multimodal models for integrated capabilities. In ICML, 2024.

[33] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. NeurIPS, 2023.