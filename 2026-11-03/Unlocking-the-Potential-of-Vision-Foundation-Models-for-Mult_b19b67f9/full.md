# Unlocking the Potential of Vision Foundation Models for Multi-Task Dense Prediction via Mixture of Task Adapters

Yunzhi Zhuge, Xinzhuo Yu, Sitong Gong, Lu Zhang, Pingping Zhang, Huchuan Lu, Fellow, IEEE

Abstract—Multi-task dense prediction seeks a single, coherent representation that supports simultaneously discriminative semantic recognition and regularized geometric/structural inference. For vision foundation models, pretrained semantic invariances conflict with the frequency and smoothness biases of dense objectives; joint fine-tuning and single-route adapters therefore compress competing gradients into one update channel, causing negative transfer and unstable cross-task agreement. We frame this as two-level adaptation: task-generic alignment that preserves priors, and task-aware specialization that resolves late-stage conflicts. Based on this principle, we propose D<sup>2</sup>MoE , a Mixture of Task Adapters that explicitly factorizes these roles within a frozen DINOv3 backbone. D<sup>2</sup>MoE introduces Task-Generic Low-Rank Adaptation (TG-LoRA) in attention projections throughout the network to provide a stable, shared adaptation channel, and a Task-Adaptive Mixture of Experts (TA-MoE) injected only at the final attention block of each stage to enact sparse, task-conditioned perturbations of token mixing in a single forward pass. A Semantic Prior Injector further couples multi-level task features with frozen foundation priors to reinforce semantic coherence while preserving boundary and structural fidelity. Experiments across diverse benchmarks substantiate that this structured factorization yields robust multi-task improvements with a small trainable footprint, establishing a principled and scalable route to prior-preserving, foundation-centric multi-task dense perception.

Index Terms—Multi-task Dense Prediction, Vision Foundation Model, Mixture of Task Adapters.

## 1 INTRODUCTION

Multi-task dense prediction (MTDP) [1], [2], [3] unifies multiple pixel-level tasks (e.g., semantic segmentation [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], depth estimation [15], [16], [17], [18], [19], [20], [20], [21], and saliency detection [22], [23], [24], [25], [26], [27]) in a single model to share computation, couple semantic/geometric/structural cues, and reduce redundancy relative to task-specific systems. This joint setting further promotes cross-task spatial coherence, especially near boundaries and geometric discontinuities. Recent MTDP increasingly builds on vision foundation models (VFMs) [28], [29], [30], using pretrained transformers (e.g., ViT [2], [31], [32], [33], [34], [35], [36] and Swin Transformer [37], [38], [39]). As a result, the central problem shifts from learning shared representations to coordinating adaptation of semantically organized foundation features across heterogeneous dense objectives.

Although vision foundation models provide rich, transferable features, MTDP under VFMs faces a core incompatibility: one representation must satisfy objectives with distinct spatial and frequency preferences. This yields three challenges: (i) Conflicting Task Priors. Semantic objectives depend on localized discriminative detail, whereas geometric objectives prefer smooth, globally consistent structure. (ii) Prior Erosion. Naive end-to-end optimization overwrites transferable invariances, degrading generalization and cross-task consistency. (iii) Single-route Adaptation Interference, i.e.,forcing all tasks through shared prompts/adapters/decoders projects heterogeneous gradients into one subspace, amplifying negative transfer. These observations motivate conditional, task-aware adaptation that selectively updates the representation to enable taskspecific refinement while preserving the shared prior. Selective routing via mixture of experts provides a principled mechanism for allocating distinct update paths to competing objectives and mitigating negative transfer [40], [41].

These observations indicate that once a single shared adaptation path bottlenecks, MTDP needs task-conditioned computation at the adaptation interface while preserving a shared foundation prior rather than hard-partitioning the backbone. (i) Unscalable Factorization. Separate encoders/decoders for each task destroys sharing and is prohibitive for foundation-scale backbones and dense outputs [31], [42]. (ii) Delayed Conflict Resolution. Latebranching heads postpone specialization, leaving upstream features exposed to the same conflicting gradients. (iii) Sparse, Task-conditioned Capacity. Effective MTDP should keep a single backbone but introduce task-dependent specialization via sparsely activated auxiliary parameters. Mixture of experts (MoE) implements this through sparse gating/routing, allocating partially disjoint subspaces per objective without sacrificing the shared representation [40], [41], [43], [44], [45]. In contrast, non-MoE MTDP either applies largely uniform encoder adaptation [46], [47] or shifts capacity into decoders whose task interaction is weakly coupled to the backbone features being adapted [37], [38].

![](images/218de89f8a8295fa1b76f5bb5e3a700187a124502d84a13fe90a41ef9b86f3ff.jpg)  
Fig. 1. Motivation of D<sup>2</sup>MoE. (a) SAK distills from multiple VFMs into a shared student, coupling heterogeneous task gradients and incurring multi-teacher overhead. (b) D<sup>2</sup>MoE instead freezes a single VFM and enforces early task decoupling through a Mixture of Task Adapter (MoTA), allocating sparse, task-conditioned low-rank update routes for specialization without eroding pretrained invariances. (c) Correlation probes (Pearson maps around sampled points) confirm that model with MoTA induces more task-consistent feature organization for both semantic segmentation and depth, improving boundary adherence without sacrificing long-range structural coherence.

![](images/acdbcba543659ebcbc4c4e1c0cd4126d07041edbe5f5505b9eefc92c963c0977.jpg)  
Fig. 2. Performance vs. Model Complexity. Comparison of D<sup>2</sup>MoE against state-of-the-art methods on NYUD-v2 [48]. D<sup>2</sup>MoE achieves superior performance (higher mIoU and lower RMSE) despite a substantial reduction in the total number of trainable parameters. The trainableparameter count varies with the task configuration, with 45M on NYUD-v2 (shown here) and 54M on Pascal-Context (as shown in Tab. 1.)

Several recent parameter-efficient and conditional multitask adaptation methods are closely related to this direction. Polyhistor [49] and VMT-Adapter [50] improve multi-task transfer by sharing lightweight adaptation modules while preserving task-specific knowledge, and TADFormer [51] further introduces task-adaptive dynamic filtering for inputconditioned task adaptation. DiTask [52] studies multi-task fine-tuning from a weight-geometry perspective by preserving pretrained singular-vector structures and modulating singular values. These methods validate the importance of parameter-efficient multi-task adaptation, yet most of them focus on designing general adaptation modules or taskconditioned filters, rather than explicitly locating where task conflicts should be resolved inside a frozen vision foundation model for dense prediction. In parallel, MoE-based MTL methods such as M<sup>3</sup>ViT [53], Mod-Squad [54], AdaMV-MoE [55], TaskExpert [33] and MLoRE [34] introduce conditional expert selection to improve cooperation-specialization trade-offs. However, existing MoE designs typically insert trainable experts throughout the backbone, assemble task features in the decoder, or target compact task-specific execution, which may either perturb pretrained representations or leave task conflicts insufficiently addressed.

Motivated by the above analysis, we view the MTDP bottleneck under frozen VFMs as requiring two complementary capabilities: task-generic adaptation and task-aware specialization. Task-generic adaptation aligns the foundation representation to the downstream data distribution while maintaining transferable semantic prior, whereas task-aware specialization explicitly resolves conflicts induced by heterogeneous dense objectives. A closely related foundationmodel-driven approach is SAK [36], which uses a shared teacher-agnostic stem with teacher-specific adapter paths to distill complementary biases from multiple VFMs and fuses them via task routers for multi-task prediction. However, the student is optimized to satisfy heterogeneous teacher feature spaces, yielding a compromise representation that weakens domain adaptation, while specialization remains teacher-conditioned rather than explicitly resolving task-level gradient conflicts. Practically, requiring ≥3 VFM teachers incurs O(K) compute/memory (multiple frozen backbones, matching losses) during training, making it bulky for deployment (as shown in Fig. 1(a)).

To address the above limitations, we propose D<sup>2</sup>MoE, a foundation model driven framework for multi-task dense prediction. Unlike prior MoE-based MTDP methods that mainly emphasize task routing and expert specialization, D<sup>2</sup>MoE uses MoE as part of a lightweight adapter design to introduce task-adaptive updates without tuning the VFM backbone. D<sup>2</sup>MoE builds on a single frozen VFM backbone and performs multi-task decoupling through a unified Mixture of Task Adapters (MoTA) design (as shown in Fig. 1(b)), which combines Task-Generic Low-rank Adaptation (TG-LoRA) with Task-Adaptive Mixture of Experts (TA-MoE). TG-LoRA is applied across backbone layers to achieve stable and parameter-efficient adaptation, while TA-MoE is selectively placed at the final layer of each stage to enable sparse, task-aware specialization within a single forward pass. In addition, D<sup>2</sup>MoE incorporates semantic priors from DINOv3 to guide cross-layer feature integration, facilitating both global context modeling and fine-grained detail preservation. As a result, D<sup>2</sup>MoE achieves state-of-the-art performance on standard multi-task dense prediction benchmarks with minimal additional parameters (as shown in Fig. 2).

Our contributions are summarized as follows.

1) We formalize multi-task dense prediction with frozen vision foundation models as a two-level adaptation problem, disentangling task-generic alignment from taskaware specialization. This perspective explains negative transfer under single-route adaptation and motivates sparse, task-conditioned computation.

2) We propose D<sup>2</sup>MoE , a Mixture of Task Adapters architecture that combines Task-Generic Low-Rank Adaptation with Task-Adaptive Mixture of Experts. The design introduces sparse, late-stage task specialization while preserving semantic priors within a single backbone.

3) Extensive experiments on PASCAL-Context, NYUD-v2, and Cityscapes show that D<sup>2</sup>MoE achieves state-ofthe-art multi-task performance, delivering the highest aggregated gains across benchmarks while requiring substantially fewer trainable parameters.

The remainder of this paper is structured as follows. Sec. 2 surveys related works on multi-task dense prediction, vision foundation models, and mixture of experts. Sec. 3 introduces D<sup>2</sup>MoE , including the Mixture of Task Adapters and the Semantic Prior Injector for prior-guided decoding. Sec. 4 presents the experimental protocol and results on PASCAL-Context, NYUD-v2, and Cityscape. Sec. 5 concludes with key findings and potential implications.

## 2 RELATED WORKS

## 2.1 Multi-task Dense Prediction

Multi-task dense prediction, which jointly addresses semantic segmentation, depth estimation, and auxiliary tasks for scene understanding, has substantively progressed via relational modeling, efficient architectures, and generative formulations [1], [2], [47], [56], [57], [58], [59], [60], [61], [62], [63]. ATRC [64] explicitly formalizes inter-task dependencies to improve transferable generalization. MTPSL [65] tackles incomplete annotations in a partially-supervised setting, enabling resilient learning under partial labels. Transformer-based designs such as the InvPT enable hierarchical feature fusion and granular scene parsing [2], [31]. Efficiency has been advanced by cross-task distillation, epitomized by KDAM [66]. MLoRA employs a mixture of low-rank experts for parameter-efficient, per-task routing that scales to heterogeneous dense outputs [34]. Synergyand compactness-oriented approaches include SEM, which promotes salutary task interactions beyond naive sharing [67], and Bi-MTPD, which binarizes predictors for compact models with negligible performance loss [68]. State-space innovations such as MTMamba [37] capture long-range dependencies for enhanced multi-task understanding, whereas MTMamba++ [38] streamlines decoding to improve efficiency. BridgeNet [42] introduces bridge embeddings to facilitate comprehensive cross-task feature interactions. Collectively, these lines of work converge on on holistic, adaptive and resource-conscious frameworks. Collectively, these lines of work converge holistic and resource-conscious frameworks.

## 2.2 Parameter Efficient Multi-task Adaption

Parameter-efficient multi-task adaptation studies how to transfer pretrained models to multiple downstream tasks with limited trainable parameters, making it complementary to MTDP-specific architectures. Polyhistor [49] uses decomposed hypernetworks and layer-wise scaling kernels to share knowledge without full task-wise fine-tuning. VMT-Adapter [50] introduces a once-for-all adapter that combines shared cross-task knowledge with lightweight task-specific extraction, yielding nearly task-number-independent cost. TADFormer [51] performs task-aware adaptation through parameter-efficient prompting and dynamic task filters conditioned on task contexts. DiTask [52] preserves pretrained weight geometry via diffeomorphic transformations of singular values for shared and task-specific modulation. These methods show that separating shared adaptation from task-specific modulation is effective for multi-task transfer. Different from them, D<sup>2</sup>MoE targets frozen VFMs for MTDP by coupling global TG-LoRA alignment with sparse TA-MoE specialization at late-stage attention projections, where heterogeneous dense objectives most strongly diverge.

## 2.3 Vision Foundation Models

Vision foundation models (VFMs) have revolutionized computer vision by learning generalizable representations from vast unlabeled datasets, enabling transfer to diverse downstream tasks [69]. Pioneering architectures like Vision Transformers (ViT) [70] scaled transformers for image processing, achieving state-of-the-art performance on benchmarks such as ImageNet. Self-supervised approaches, including Masked Autoencoders (MAE) [71], reconstruct masked image patches to capture semantic features without annotations. DINO [28] employs self-distillation to emerge invariant representations, fostering emergent properties like object segmentation. Multimodal models such as CLIP align vision and language embeddings for zero-shot classification and retrieval. Subsequent iterations like DINOv2 [29] refine training with curated data and improved objectives, yielding robust features across scales. The Segment Anything Model (SAM) [72] introduces promptable segmentation for arbitrary objects, advancing interactive vision applications. Most recently, DINOv3 [30] extends these capabilities with enhanced multitask potential through advanced self-supervision techniques. Accordingly, this work targets prior-preserving, parameterefficient task-conditioned adaptation of VFMs for multi-task dense prediction.

## 2.4 Mixture of Experts

Mixture of experts (MoE) has emerged as a de facto paradigm for conditional computation: a router sparsely activates a small subset of experts per token/sample, increasing capacity without proportional FLOPs. Foundational designs such as Switch Transformer [40] and GLaM [41] establish scalable sparse routing, while Expert-Choice routing [45] and DeepSpeed-MoE [73] show that routing policies and systems support are central to utilization and efficiency.

In vision and multimodal foundation models, MoE is used to reduce domain/task interference while retaining scalable capacity. V-MoE [44] adapts sparse experts to

![](images/f9f7092ae7c6626a1270473cb84631d22f852934d46ecb2170e755d8215869af.jpg)  
Fig. 3. Overall architecture of D<sup>2</sup>MoE.(§3.1) Built upon a frozen DINOv3 backbone, D<sup>2</sup>MoE organizes MoTA across four stages to preserve shared priors through task-generic adaptation and introduce localized task-specific adaption at stage endings. SPI fuses prior and task features, followed by lightweight heads for multi-task dense predictions.

ViTs, LIMoE [74] shows modality-aware partitioning under contrastive learning, and Uni-Perceiver-MoE [75] integrates conditional experts into unified generalist architectures. More recent multimodal variants route over heterogeneous expert pools for generalist MLLMs or vision experts (MoME [76], MoVA [77]), with Uni-MoE [78] further systematizing scaling in unified multimodal LLMs.

MoE is also effective for multi-task learning and parameter-efficient specialization, where conditional routing allocates capacity to tasks with higher complexity or conflict. Vision MTL approaches route along task-conditioned expert pathways with varied sharing/gating (M<sup>3</sup>ViT [53], Mod-Squad [54], AdaMV-MoE [55], TaskExpert [33]), and MLoRE [34] scales experts for dense prediction via low-rank parameterization. Recent work reframes MoE as adapterstyle tuning: MoE-Adapters [79] sparsely selects from a fixed adapter pool (with OOD fallback), and MoE-Adapters++ [80] introduces dynamic adapters and learned expert selection. Inspired by adapter-style MoE, we route among lightweight task adapters atop frozen DINOv3 features, concentrating specialization in small modules while preserving backbone transferability.

## 3 METHODOLOGY

This section presents D<sup>2</sup>MoE , a parameter-efficient framework for adapting frozen vision foundation models to multitask dense prediction. Sec. 3.1 first provides an overview of the pipeline, including the frozen DINOv3 backbone, the Mixture of Task Adapters (MoTA), and the Semantic Prior Injector (SPI). Sec. 3.2 then details MoTA, which decouples shared adaptation from task-conditioned specialization. Finally, Sec. 3.3 introduces SPI for prior-guided multi-scale decoding.

## 3.1 Framework Overview

Fig. 3 illustrates the overall architecture of D<sup>2</sup>MoE . Given an input image $I \in \mathbb { R } ^ { H \times W \times 3 }$ , we employ a frozen DINOv3 backbone as the shared feature extractor. The backbone provides two types of representations: task-agnostic semantic priors $\{ S _ { i } \} _ { i = 1 } ^ { 4 }$ , which retain the pretrained foundation knowledge, and task-adapted features $\{ F _ { i } ^ { t } \} _ { i = 1 } ^ { 4 }$ for each downstream task $t \in \{ 1 , \ldots , T \}$

To adapt the frozen backbone without full fine-tuning, we insert the proposed Mixture of Task Adapters (MoTA) into the attention blocks. MoTA serves as the only backboneside trainable adaptation path and produces task-aware multi-level features while preserving the shared VFM prior. The adapted features are then passed to the Semantic Prior Injector (SPI), which fuses multi-level task features with frozen semantic priors for prior-guided decoding. Finally, lightweight task-specific heads map the refined representations to the corresponding prediction spaces, yielding taskwise outputs $\{ O _ { t } \} _ { t = 1 } ^ { T }$

## 3.2 Mixture of Task Adapters

MoTA is designed to address the single-route adaptation bottleneck in frozen VFM-based multi-task dense prediction. Instead of forcing all tasks to share an identical adaptation subspace, MoTA factorizes backbone adaptation into two complementary components: a shared low-rank pathway for stable task-generic alignment, and a sparse task-conditioned pathway for resolving task-specific conflicts. This design allows D<sup>2</sup>MoE to retain the semantic priors of the frozen DINOv3 while introducing lightweight conditional capacity only where task divergence needs to be modeled. The following sections describe the two components in detail.

## 3.2.1 Task-Generic Low-Rank Adaptation

To obtain parameter-efficient transfer while preserving the semantic geometry encoded by the DINOv3 backbone, we introduce Task-Generic Low-Rank Adaptation (TG-LoRA) inside the self-attention modules. In contrast to full finetuning, TG-LoRA keeps all pretrained backbone weights frozen and learns only structured low-rank increments on a small set of high-impact linear maps. This constrained reparameterization provides a stable, task-agnostic adaptation channel that aligns the foundation representation to downstream data distributions with reduced risk of representation drift and cross-task interference.

TG-LoRA is inserted exclusively into the query, key, value projections and the attention output projection of each Transformer block. Let $X \in \mathbb { R } ^ { N \times C }$ denote the input token matrix (N tokens, embedding dimension C). Under a prenormalization design, we first compute:

$$
\tilde { X } = \operatorname { L N } ( X ) .\tag{1}
$$

For $P \in \{ Q , K , V \}$ , we express the projection as a frozen base weight plus a rank-r residual update:

$$
\Delta W _ { P } = B _ { P } A _ { P } ,\tag{2}
$$

![](images/af19720ab7eb3821544cb183020ca0899358e6dbb50af86219f690678a08dc06.jpg)  
Fig. 4. Illustration of the key components in D²MoE. (§3.2 and 3.3) (a) Task-Generic Adaptation inserts TG-LoRA into attention projections to provide shared, low-rank adaptation while keeping the DINOv3 backbone frozen. (b) Task-Specific Adaptation employs TA-MoE, where task routers select Top-k low-rank experts to produce sparse, task-conditioned updates for attention projections. (c) The Semantic Prior Injector fuses multi-leve task features with frozen semantic priors to enhance semantic consistency and structural coherence for dense prediction tasks.

$$
P = \tilde { X } \big ( W _ { P } + s _ { P } \Delta W _ { P } \big ) ,\tag{3}
$$

where $W _ { P } \in \mathbb { R } ^ { C \times C }$ is frozen, while $A _ { P } \in \mathbb { R } ^ { C \times r }$ and $B _ { P } \in$ $\mathbb { R } ^ { r \times C }$ are learnable with $r \ll C$ . The scalar $s _ { P }$ controls the effective update magnitude; a common choice is $s _ { P } = \alpha _ { P } / r$ to keep the perturbation scale approximately stable as r varies. This can be interpreted as learning a constrained perturbation of the form $W _ { P } \mapsto W _ { P } + s _ { P } \bar { B } _ { P } A _ { P }$ without modifying the pretrained weights.

The projected features are then processed by multi-head self-attention. Let h denote the number of heads and $d _ { h } =$ $C / h$ the per-head dimension. After reshaping into head-wise tensors $\{ Q ^ { ( m ) } , K ^ { ( m ) } , V ^ { ( m ) } \} _ { m = 1 } ^ { h }$ with $\mathring { Q } ^ { { ( m ) } } , K ^ { ( m ) } , V ^ { ( m ) } \in$ $\mathbb { R } ^ { N \times d _ { h } }$ , each head computes:

$$
H ^ { ( m ) } = \mathrm { S o f t m a x } \left( \frac { Q ^ { ( m ) } ( K ^ { ( m ) } ) ^ { \top } } { \sqrt { d _ { h } } } \right) V ^ { ( m ) } , \qquad m = 1 , \dots , h ,\tag{4}
$$

and the heads are concatenated to form $Z \quad =$ Concat $( H ^ { ( 1 ) } , \dots , H ^ { ( h ) } ) \in \mathbb { R } ^ { N \times d }$

TG-LoRA is further applied to the attention output projection that maps contextualized tokens back into the shared embedding space:

$$
Z ^ { \prime } = Z \bigl ( W _ { O } + s _ { O } B _ { O } A _ { O } \bigr ) ,\tag{5}
$$

where $W _ { O } \in \mathbb { R } ^ { d \times d }$ is frozen and $( A _ { O } , B _ { O } )$ are learnable rank r adapters. The attention sub-layer output is then obtained through the residual pathway:

$$
Y = X + Z ^ { \prime } .\tag{6}
$$

The FFN and LayerScale follow the original DINOv3 design and are kept frozenThe TG-LoRA augmented output is then propagated to the next Transformer block. To preserve the pretrained VFM’s general-purpose representation, semantic priors $\{ S _ { i } \} _ { i = 1 } ^ { 4 }$ are extracted from the corresponding unmodified DINOv3 blocks applied to the same input

X, ensuring that the priors reflect task-agnostic backbone features rather than adaptation-induced artifacts.

By restricting TG-LoRA to $\{ W _ { Q } , W _ { K } , W _ { V } , W _ { O } \}$ , the trainable degrees of freedom primarily modulate (i) the tokento-token interaction geometry via (Q, K) and (ii) the injection of contextual evidence back into the embedding space via $( V , W _ { O } )$ , while leaving the typically more representationally disruptive FFN untouched. As a result, TG-LoRA acts as a globally shared, task-generic adaptation mechanism that improves robustness to domain shifts while preserving the semantic organization of the foundation model, providing a stable substrate for subsequent task-adaptive specialization.

## 3.2.2 Task-Adaptive Mixture of Experts

TG-LoRA provides a shared, task-agnostic adaptation route that preserves the backbone’s semantic prior. Nevertheless, heterogeneous dense prediction objectives often demand incompatible token-interaction patterns at late representation levels, where high-level semantics are consolidated and task divergence becomes most pronounced. To explicitly factorize these late-stage conflicts without fragmenting the backbone, we introduce Task-Adaptive Mixture of Experts (TA-MoE) as a conditional specialization mechanism.

In contrast to MoE designs that distribute experts broadly across backbone layers or postpone task differentiation to decoder heads [53], [54], TA-MoE is injected only into the final self-attention block of each DINOv3 stage and acts directly on the attention linear maps (Q/K/V projections and the output projection). This placement targets the token-mixing operator where task-dependent contextual aggregation is most sensitive, while keeping earlier layers fully shared for stable representation learning.

Let $H ^ { s } \in \mathbb { R } ^ { N \times C }$ denote the token features entering the last attention block of stage s, where N is the number of tokens and C is the channel dimension. For a T-task setting, we introduce task-indexed inputs $\{ H ^ { s , t } \} _ { t = 1 } ^ { T }$ as notational replicas initialized by broadcasting $H ^ { s }$ , so that task-conditioned operators can be written uniformly, while the shared backbone features $H ^ { s }$ are computed only once.

A task-conditioned router predicts a sparse gating distribution over E candidate experts from a global stage summary. We first compute a compact descriptor via tokenwise pooling:

$$
\bar { h } ^ { t } = \mathrm { P o o l } \bigl ( H ^ { s , t } \bigr ) \in \mathbb R ^ { C } ,\tag{7}
$$

and obtain expert logits using a lightweight task-specific linear map:

$$
a ^ { t } = R ^ { t } \bar { h } ^ { t } \in \mathbb R ^ { E } ,\tag{8}
$$

where $R ^ { t } \in \mathbb { R } ^ { E \times C }$ . Sparse activation is enforced by selecting the Top-k experts $( k \ll E )$ and normalizing on the active set:

$$
\pi ^ { t } = \operatorname { S o f t m a x } ( \operatorname { T o p } - k ( a ^ { t } ) ) \in \Delta ^ { E } ,\tag{9}
$$

where $\Delta ^ { E }$ denotes the probability simplex with zeros on inactive experts.

Each expert is implemented as a low-rank residual update. For each attention projection $P \in \{ Q , K , V , O \}$ , the e-th expert is parameterized by factors $A _ { P } ^ { ( e ) } \in \mathbb { R } ^ { C \times r }$ and $B _ { P } ^ { ( e ) } \in$ $\mathbb { R } ^ { r \times C }$ with rank $r \ll C$ . The task-specific residual to the frozen projection $W _ { P } \in \mathbb { R } ^ { C \times C }$ is:

$$
\Delta W _ { P } ^ { t } = \frac { \alpha } { r } \sum _ { e = 1 } ^ { E } \pi _ { e } ^ { t } B _ { P } ^ { ( e ) } A _ { P } ^ { ( e ) } ,\tag{10}
$$

with scaling factor $\alpha .$ Accordingly, task-specialized projections are computed as:

$$
P ^ { t } = H ^ { s , t } \big ( W _ { P } + \Delta W _ { P } ^ { t } \big ) , \qquad P \in \{ Q , K , V \} .\tag{11}
$$

The task-adaptive $Q ^ { t } , K ^ { t }$ , and $V ^ { t }$ are then processed by standard multi-head self-attention:

$$
Z ^ { t } = \mathrm { M S A } ( Q ^ { t } , K ^ { t } , V ^ { t } ) = \mathrm { S o f t m a x } \left( \frac { Q ^ { t } K ^ { t ^ { \top } } } { \sqrt { d _ { h } } } \right) V ^ { t } ,\tag{12}
$$

where $d _ { h }$ denotes the per-head dimension (head splitting/concatenation is omitted for brevity). TA-MoE is further applied to the attention output projection to complete a coherent task-conditioned attention pathway:

$$
\widetilde { H } ^ { s , t } = Z ^ { t } \big ( W _ { O } + \Delta W _ { O } ^ { t } \big ) ,\tag{13}
$$

where $W _ { O }$ is the frozen output projection and $\Delta W _ { O } ^ { t }$ is the routed low-rank residual defined above.

The resulting tokens are processed by the standard residual connections, feed-forward transformation, and LayerScale of DINOv3, yielding the task-specific stage-wise outputs $\{ F _ { i } ^ { t } \} _ { i = 1 } ^ { 4 }$ . Overall, TA-MoE interprets task divergence as a conditional perturbation of the token-mixing operator: the backbone projections remain frozen to preserve a shared semantic prior, while a sparse mixture of low-rank experts injects task-selective capacity precisely at the late layers where conflicts are most acute. $\mathtt { B y }$ restricting conditional computation to the final attention block of each stage, we obtain a clean separation between globally shared representation learning and task-specific refinement, achieving effective disentanglement with minimal additional overhead.

## 3.3 Semantic-Prior Guided Decoding

TG-LoRA and TA-MoE adapt the shared backbone for multitask dense prediction, yet the final outputs still depend on a decoder that maps adapted features to task-specific label spaces. Most VFM-style decoders remain shallow and task-agnostic, typically performing FPN-like fusion without explicitly exploiting the high-level semantic structure already encoded in foundation models. We thus propose a semanticprior guided decoder that integrates task-adapted features with frozen DINOv3 representations, enabling prior-aware multi-scale aggregation and task-conditioned refinement.

## 3.3.1 Semantic Prior Injection

For each task t, let $\{ X _ { i } ^ { t } \} _ { i = 1 } ^ { 4 }$ be the task-adapted features from four backbone stages, where $X _ { i } ^ { t } \in \mathbb { R } ^ { C \times H \times \mathbf { \dot { W } } }$ , and let $\{ S _ { i } \} _ { i = 1 } ^ { 4 }$ be the aligned semantic priors extracted from DINOv3, with $S _ { i } \in \mathbb { R } ^ { C \times H \times W }$ . The Semantic Prior Injector (SPI) performs channel alignment, task-conditioned multi-scale fusion, and semantic prior injection.

Since features from all stages share the same spatial resolution and channel dimensions, no explicit spatial rescaling is required. We project both task features and semantic priors C using independent $1 \times 1$ convolutions:

$$
\tilde { X } _ { i } ^ { t } = \phi _ { x } ( X _ { i } ^ { t } ) ,\tag{14}
$$

$$
{ \tilde { S } } _ { i } = \phi _ { s } ( S _ { i } ) ,\tag{15}
$$

where $\tilde { X } _ { i } ^ { t } , \tilde { S } _ { i } \in \mathbb { R } ^ { C \times H \times W }$ . To aggregate task features across scales, we concatenate the aligned multi-scale task features along the channel dimension:

$$
\begin{array} { r } { \boldsymbol { X } _ { \mathrm { c a t } } ^ { t } = \operatorname { C o n c a t } \left( \tilde { X } _ { 1 } ^ { t } , \tilde { X } _ { 2 } ^ { t } , \tilde { X } _ { 3 } ^ { t } , \tilde { X } _ { 4 } ^ { t } \right) \in \mathbb { R } ^ { 4 C \times H \times W } . } \end{array}\tag{16}
$$

We then employ a lightweight fusion head composed of a $1 \times 1$ convolution for channel mixing and compression, followed by batch normalization and a non-linear activation function, and a subsequent $3 \times 3$ convolution to capture local spatial interactions, corresponding to pixel-wise fusion logits for each stage:

$$
L ^ { t } = \psi ( { X } _ { \mathrm { c a t } } ^ { t } ) , \qquad L ^ { t } \in \mathbb { R } ^ { 4 \times H \times W } ,\tag{17}
$$

where $\psi ( \cdot )$ denotes the convolutional fusion module. A softmax operation is applied along the stage dimension to obtain normalized pixel-wise fusion weights:

$$
A _ { i } ^ { t } ( h , w ) = \frac { \exp ( L _ { i } ^ { t } ( h , w ) ) } { \sum _ { j = 1 } ^ { 4 } \exp \Bigl ( L _ { j } ^ { t } ( h , w ) \Bigr ) } , \qquad \sum _ { i = 1 } ^ { 4 } A _ { i } ^ { t } ( h , w ) = 1 .\tag{18}
$$

Finally, the aggregated task feature is computed as a pixelwise weighted sum over scales:

$$
X _ { \mathrm { a g g } } ^ { t } ( h , w ) = \sum _ { i = 1 } ^ { 4 } A _ { i } ^ { t } ( h , w ) \tilde { X } _ { i } ^ { t } ( h , w ) .\tag{19}
$$

In parallel, we merge semantic priors across stages into a single map. Concretely, we concatenate $\{ \tilde { S } _ { i } \} _ { i = 1 } ^ { 4 }$ along the channel dimension and apply a lightweight fusion operator:

$$
S _ { \mathrm { a g g } } = \psi _ { s } \left( \mathrm { C o n c a t } ( \tilde { S } _ { 1 } , \tilde { S } _ { 2 } , \tilde { S } _ { 3 } , \tilde { S } _ { 4 } ) \right) ,\tag{20}
$$

where $\psi _ { s } ( \cdot )$ is implemented as $1 \times 1$ channel mixing followed by a $. 3 \times 3$ convolution for local context integration.

TABLE 1  
Performance comparison on the PASCAL-Context datasets. ↑/↓ indicates that the higher/lower results represent better performance.
<table><tr><td colspan="2">Method</td><td>Backbone</td><td>Sem.  $\mathbf { s e g . }$  mIoU↑</td><td>H. Pars. mIoU↑</td><td>Sal. maxF↑</td><td>Norm. mErr↓</td><td>Edge odsF↑</td><td> $\Delta _ { m }$   $( \% ) \uparrow$ </td><td>Para. (M)↓</td><td>Train. Para. (M)↓)</td></tr><tr><td>MTL Baseline</td><td></td><td>ViT-L</td><td>82.61</td><td>76.38</td><td>77.09</td><td>17.63</td><td>74.00</td><td>0.00</td><td>328</td><td>25</td></tr><tr><td>PAD-Net [58]</td><td>[CVPR2018]</td><td>HRNet-18</td><td>53.60</td><td>59.60</td><td>65.80</td><td>15.30</td><td>72.50</td><td>-12.11</td><td>81</td><td>81</td></tr><tr><td>MTI-Net [60]</td><td>[ECCV2020]</td><td>HRNet-18</td><td>61.70</td><td>60.18</td><td>84.78</td><td>14.23</td><td>70.80</td><td>-4.32</td><td>128</td><td>128</td></tr><tr><td>ATRC [64]</td><td>[ICCV2021]</td><td>HRNet-18</td><td>67.67</td><td>62.93</td><td>82.29</td><td>14.24</td><td>72.42</td><td>-2.37</td><td>96</td><td>96</td></tr><tr><td>ASTMT [81]</td><td>[ECCV2024]</td><td>ResNet-50</td><td>68.00</td><td>61.12</td><td>65.71</td><td>14.68</td><td>72.40</td><td>-7.57</td><td>49</td><td>49</td></tr><tr><td>MTMamba [37]</td><td>[ECCV2024]</td><td>Swin-L</td><td>81.11</td><td>72.62</td><td>84.14</td><td>14.14</td><td>78.80</td><td>5.74</td><td>336</td><td>336</td></tr><tr><td>MTMamba++ [38]</td><td>[TPAMI2025]</td><td>Swin-L</td><td>81.94</td><td>72.87</td><td>85.56</td><td>14.29</td><td>78.60</td><td>6.15</td><td>343</td><td>343</td></tr><tr><td>BIM [39]</td><td>[ICCV2025]</td><td>Swin-L</td><td>81.25</td><td>73.15</td><td>84.14</td><td>14.13</td><td>79.05</td><td>5.99</td><td>388</td><td>388</td></tr><tr><td>InvPT [31]</td><td>[ECCV2022]</td><td>ViT-L</td><td>79.03</td><td>67.61</td><td>84.81</td><td>14.15</td><td>73.00</td><td>2.52</td><td>423</td><td>423</td></tr><tr><td>InvPT++ [32]</td><td>[TPAMI2024]</td><td>ViT-L</td><td>80.44</td><td>69.19</td><td>84.85</td><td>13.70</td><td>74.20</td><td>4.12</td><td>421</td><td>421</td></tr><tr><td>TaskPrompter [2]</td><td>[ICLR2023]</td><td>ViT-L</td><td>80.89</td><td>68.89</td><td>84.83</td><td>13.72</td><td>73.50</td><td>3.93</td><td>401</td><td>401</td></tr><tr><td>TaskExpert [33]</td><td>[ICCV2023]</td><td>ViT-L</td><td>80.64</td><td>69.42</td><td>84.87</td><td>13.56</td><td>74.20</td><td>4.39</td><td>420</td><td>420</td></tr><tr><td>MLoRE [34]</td><td>[CVPR2024]</td><td>ViT-L</td><td>81.41</td><td>70.52</td><td>84.90</td><td>13.51</td><td>75.42</td><td>5.26</td><td>407</td><td>407</td></tr><tr><td>TaskDiffusion [35]</td><td>[ICLR2025]</td><td>ViT-L</td><td>81.21</td><td>69.62</td><td>84.94</td><td>13.55</td><td>74.89</td><td>4.80</td><td>416</td><td>416</td></tr><tr><td>BridgeNet [42]</td><td>[TPAMI2025]</td><td>ViT-L</td><td>80.64</td><td>70.06</td><td>84.64</td><td>13.82</td><td>72.96</td><td>3.87</td><td>477</td><td>477</td></tr><tr><td>SAK [82]</td><td>[ICLR2025]</td><td>ViT-L</td><td>84.01</td><td>76.99</td><td>84.65</td><td>13.82</td><td>76.27</td><td>7.40</td><td>407</td><td>407</td></tr><tr><td>D2MoE</td><td>[Ours]</td><td>ViT-L</td><td>84.48</td><td>78.05</td><td>85.55</td><td>13.57</td><td>78.40</td><td>8.88</td><td>358</td><td>54</td></tr></table>

Given the aggregated task feature $X _ { \mathrm { a g g } } ^ { t } ,$ SPI injects semantic priors by directly concatenating the aligned semantic representation with the task feature, followed by joint feature refinement. Specifically, we fuse the task feature and semantic prior as:

$$
U ^ { t } = \mathrm { C o n c a t } \big ( X _ { \mathrm { a g g } } ^ { t } , S _ { \mathrm { a g g } } \big ) ,\tag{21}
$$

where the concatenation is performed along the channel dimension. The fused representation is then processed by a lightweight refinement block to produce the final priorenhanced feature:

$$
\boldsymbol { Z } ^ { t } = \psi _ { z } \big ( \boldsymbol { U } ^ { t } \big ) ,\tag{22}
$$

where $\psi _ { z } ( \cdot )$ is implemented as a Conv–Norm–GeLU–Conv stack.

## 3.3.2 Task-Specific Decoding Head

Given $Z ^ { t } .$ , we employ a lightweight head that maps the shared latent to the prediction space of task t:

$$
\hat { Y } ^ { t } = \mathcal { U } \big ( \mathcal { P } ^ { t } ( Z ^ { t } ) \big ) ,\tag{23}
$$

where $\mathcal { P } ^ { t } ( \cdot )$ is a task-dependent $1 \times 1$ projection and $\mathcal { U } ( \cdot )$ denotes bilinear upsampling to the input resolution.

## 3.3.3 Joint Optimization Losses

We train all tasks jointly with ${ \mathcal { L } } _ { \mathrm { t o t a l } } ,$ , which is defined as:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \sum _ { t = 1 } ^ { T } \lambda _ { t } { \mathcal { L } } _ { t } ,\tag{24}
$$

where $\mathcal { L } _ { t }$ is task-specific (e.g., cross-entropy for segmentation, $\ell _ { 1 }$ /scale-invariant losses for depth, and cosine/angular losses for normals) and $\lambda _ { t }$ balances tasks. We use uniform weighting $\lambda _ { t } = 1$ in all experiments.

## 4 EXPERIMENTS

In this section, we evaluate D<sup>2</sup>MoE for multi-task dense prediction. §4.1 introduces the benchmarks (Pascal-Context, NYUD-v2, and Cityscapes) and the evaluation protocol, including per-task metrics and the overall multi-task gain $\Delta _ { m }$ . §4.2 details our training and implementation. §4.3 reports quantitative and qualitative comparisons with recent state-of-the-art methods. §4.4 presents ablations studies and analyze the contribution of each component. §4.5 gives feature visualization and analysis. §4.6 evaluates robustness.

## 4.1 Experimental Setups.

## 4.1.1 Datasets

Following recent MTL works [32], [37], [42], we evaluate D<sup>2</sup>MoE on three widely used benchmarks under standardized protocols, with official splits and their canonical task suites. Pascal-Context [83] contains 4998 training and 5105 test images with rich pixel-level annotations, on which we report semantic segmentation (Sem. Seg.), human parsing (H. Pars.), object edge detection (Edge), surface normal estimation (Norm.), and saliency detection (Sal.). NYUDv2 [48] focuses on indoor scenes with 795 training and 654 test samples; we evaluate Sem. Seg., monocular depth estimation (Dep.), Norm., and Edge Cityscapes provides urban streetview imagery with 2975 training images and 500 validation images, including 19 semantic classes and stereo-derived disparity; we follow common practice and report Sem. Seg. together with geometry-related predictions (Dep.) derived from the fine annotations. Together, these datasets cover outdoor scenes, indoor environments, and driving scenarios, enabling a broad and consistent evaluation of multi-task dense prediction under diverse conditions.

## 4.1.2 Evaluation Metrics

To enable a rigorous, fair and fully comprehensive comparison for multi-task dense prediction, we adopt standard task-specific measures and a normalized cross-task summary metric for overall assessment. Specifically, we report:

TABLE 2  
Performance comparison on the NYUDv2 dataset. ↑/↓ indicates that the higher/lower results represent better performance.
<table><tr><td colspan="2">Methods</td><td>Backbone</td><td>Sem. Seg. mIoU↑</td><td>Dep. RMŠE↓</td><td>Norm. mErr↓</td><td>Edge odsF↑</td><td> $\Delta _ { m }$  (%)↑</td></tr><tr><td colspan="2">MTL Baseline</td><td>ViT-L</td><td>63.15</td><td>0.5119</td><td>18.81</td><td>75.10</td><td>0.00</td></tr><tr><td colspan="2">MTI-Net [60]</td><td>[ECCV2024] HRNet-18</td><td>45.97</td><td>0.5365</td><td>20.27</td><td>77.84</td><td>-9.03</td></tr><tr><td colspan="2">PAD-Net [58]</td><td>[CVPR2018]</td><td>HRNet-18</td><td>36.61</td><td>0.6270</td><td>20.85 76.38</td><td>-18.41</td></tr><tr><td colspan="2">ATRC [64]</td><td>[ICCV2021]</td><td>HRNet-18</td><td>46.33</td><td>0.5363 20.18</td><td>77.94</td><td>-8.73</td></tr><tr><td colspan="2">ASTMT [81]</td><td>[ECCV2024]</td><td>ResNet-50 32.16</td><td>0.5700</td><td>23.18</td><td>74.50</td><td>-21.11</td></tr><tr><td colspan="2">MTMamba [37]</td><td>[ECCV2024]</td><td>Swin-L</td><td>55.82</td><td>0.5066 18.63</td><td>78.70</td><td>-1.21</td></tr><tr><td colspan="2">MTMamba++ [38]</td><td>[TPAMI2025]</td><td>Swin-L</td><td>57.01</td><td>0.4818</td><td>18.27 79.40</td><td>1.19</td></tr><tr><td colspan="2">BIM [39]</td><td>[ICCV2025]</td><td>Swin-L</td><td>57.40</td><td>0.4733 18.55</td><td>78.72</td><td>1.16</td></tr><tr><td colspan="2">InvPT [31]</td><td>[ECCV2024]</td><td>ViT-L</td><td>53.56</td><td>0.5183 19.04</td><td>78.10</td><td>-3.42</td></tr><tr><td colspan="2">InvPT++ [32]</td><td>[ECCV2024]</td><td>ViT-L</td><td>53.89</td><td>0.5091 18.65</td><td>78.10</td><td>-2.32</td></tr><tr><td colspan="2">TaskPrompter [2]</td><td>[ECCV2024]</td><td>ViT-L</td><td>55.30</td><td>0.5152 18.47</td><td>78.20</td><td>-1.78</td></tr><tr><td colspan="2">TaskExperts [33]</td><td>[ECCV2024]</td><td>ViT-L</td><td>55.35</td><td>0.5157 18.54</td><td>78.40</td><td>-1.82</td></tr><tr><td colspan="2">MLoRE [34]</td><td>[ECCV2024]</td><td>ViT-L</td><td>55.96</td><td>0.5076 18.33</td><td>78.43</td><td>-0.89</td></tr><tr><td colspan="2">TaskDiffusion [35]</td><td>[ICLR2025]</td><td>ViT-L</td><td>55.65</td><td>0.5020 18.43</td><td>78.64</td><td>-0.80</td></tr><tr><td colspan="2">BridgeNet [42]</td><td>[TPAMI2025]</td><td>ViT-L</td><td>55.51 0.4930</td><td>18.47</td><td>78.22</td><td>-0.61</td></tr><tr><td colspan="2">SAK [82]</td><td>[ICLR2025]</td><td>ViT-L</td><td>63.18 0.4313</td><td>16.25</td><td>79.43</td><td>8.79</td></tr><tr><td colspan="2">D2MoE</td><td>[Ours]</td><td>ViT-L</td><td>65.91 0.4209</td><td>15.82</td><td>80.60</td><td>11.34</td></tr></table>

TABLE 3  
Performance comparison on the Cityscapes dataset.
<table><tr><td>Methods</td><td>Backbone</td><td>Sem. Seg. mIoU↑</td><td>Dep. RMSE↓</td><td> $\Delta _ { m }$  (%) ↑</td></tr><tr><td>MTL Baseline</td><td>ViT-L</td><td>76.45</td><td>4.36</td><td>0</td></tr><tr><td>MTI-Net [60]</td><td>HRNet-18</td><td>53.19</td><td>5.05</td><td>-23.13</td></tr><tr><td>PAD-Net [58]</td><td>HRNet-18</td><td>59.85</td><td>5.06</td><td>-18.88</td></tr><tr><td>MTMamba [37]</td><td>Swin-L</td><td>78.00</td><td>4.66</td><td>-2.43</td></tr><tr><td>MTMamba++[38]</td><td>Swin-L</td><td>79.13</td><td>4.63</td><td>-1.34</td></tr><tr><td>InvPT [31]</td><td>ViT-L</td><td>71.78</td><td>4.67</td><td>-6.61</td></tr><tr><td>TaskPrompter [2]</td><td>ViT-L</td><td>72.41</td><td>5.49</td><td>-15.60</td></tr><tr><td>D²MoE</td><td>ViT-L</td><td>82.10</td><td>4.22</td><td>5.30</td></tr></table>

• Sem. Seg. / H. Pars.: mean Intersection-over-Union (mIoU).

• Dep.: root mean square error (RMSE).

• Norm.: mean angular error (mErr).

• Sal.: maximal F-measure (maxF).

• Edge: Optimal Dataset Scale F-measure (odsF)

To summarize performance across heterogeneous task objectives, the overall performance gain of multi-task learning $( \Delta _ { m } )$ is computed relative to our multi-task baseline:

$$
\Delta _ { m } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } ( - 1 ) ^ { \ell _ { t } } \frac { M _ { t } - M _ { t } ^ { m } } { M _ { t } ^ { m } } ,\tag{25}
$$

where $T$ is the number of tasks, $M _ { t }$ and ${ M } _ { t } ^ { m }$ denote the comparing result and our multi-task baseline results for task $t ,$ and $\ell _ { t } = 1$ for lower-is-better metrics (RMSE, mErr) and $\ell _ { t } = 0$ otherwise (mIoU, maxF, ODS-F). A positive $\Delta _ { m }$ therefore indicates an overall improvement. We report all per-task metrics together with $\Delta _ { m }$ on the official dataset splits.

## 4.2 Implementation Details

To enable parameter-efficient multi-task dense prediction, we use DINOv3 [30] as frozen backbone and adapt it with our designed modules. Models are trained with AdamW (weight decay $1 \times 1 0 ^ { - 6 } )$ and a polynomial learning rate schedule. Base learning rates are $8 \times \mathrm { i 0 ^ { - 5 } } , 2 \times 1 0 ^ { - 5 }$ , and $1 \times 1 0 ^ { - 4 }$ for Pascal-Context, NYUDv2, and Cityscapes, respectively. We train on two NVIDIA A100 GPUs with a batch size of 4 (2 per GPU). Loss functions are detailed in Sec. 3.3.3. TG-LoRA is applied to the Q, K, V , and output layers of all attention blocks with rank 16. TA-MoE is applied in the final attention layer, with each expert as a rank-4 LoRA module. The SPI module has a fused feature embedding dimension of 384. All LoRA adapters use a scaling factor of 4 and a dropout rate of 0.05.

To isolate the effect of our modules from DINOv3 itself, we build the MTL Baseline with the same frozen DINOv3 backbone and identical training protocol as D<sup>2</sup>MoE . This baseline trains only task-specific prediction heads, without TG-LoRA, TA-MoE, or SPI. Thus, its gap to D<sup>2</sup>MoE directly reflects the contribution of our proposed modules.

## 4.3 Benchmark Results

## 4.3.1 Quantitative Comparison Results

We conduct comprehensive quantitative comparisons on three standard benchmarks, namely Pascal-Context, NYUDv2, and Cityscapes, using the official data splits and widely adopted multi-task settings for fair evaluation. We report task-specific metrics for each benchmark and additionally summarize overall performance using the aggregated multitask gain $\Delta _ { m }$ , which facilitates a holistic comparison across heterogeneous dense prediction tasks.

Pascal-Context. Tab. 1 summarizes results on PASCAL-Context across five dense prediction tasks including semantic segmentation (Sem. Seg.), human parsing (H. Pars.), saliency detection (Sal.), surface normal estimation (Norm.) and edge detection (Edge) under the standard multi-task setting. D<sup>2</sup>MoE achieves the highest aggregated multitask gain $( \Delta _ { m } = 8 . 8 8 )$ . Relative to the MTL Baseline (ViT-L), it improves Sem. Seg. (82.61→84.48 mIoU), H. Pars. (76.38→78.05 mIoU) and Sal. ( 77.09→85.55 maxF), while reducing Norm. error (17.63→13.57 mErr) and increasing boundary accuracy (74.00→78.40 odsF). The concurrent gains across recognition-oriented and geometry-sensitive tasks suggest that $\mathrm { D ^ { 2 } M o E }$ mitigates negative transfer while preserving beneficial sharing. Compared with decodercentric methods such as MTMamba++ and BIM (Swin-L), $\mathrm { D ^ { 2 } M o E }$ attains higher performance on segmentation and parsing and a larger overall gain (8.88 versus 6.15 and 5.99). Among ViT-L-based approaches, D<sup>2</sup>MoE also improves upon MLoRE, TaskDiffusion and BridgeNet, consistent with the view that task-adaptive routing, coupled with the semantic priors provided by DINOv3, supports stronger task disentanglement and more effective cross-task coordination.

![](images/caaa8aad821cf037af176adb2d10abdbc38657f85b951ea0fa1285eb4faf5247.jpg)  
Fig. 5. Qualitative Comparison on Pascal-Context. Compared with TaskDiffusion [35] and MTMamba [37], D<sup>2</sup>MoE produces crisper boundaries and fewer cross-task inconsistencies (e.g., aligned silhouettes and object extents across parsing/segmentation/saliency and normals), indicating improved shared structure modeling and reduced task interference in multi-objective dense prediction.

NYUD-v2. Tab. 2 reports quantitative results on NYUDv2 for Sem. Seg, Dep., Norm. and Edge D<sup>2</sup>MoE achieves the highest aggregated multi-task gain $( \Delta _ { m } = 1 1 . 3 4 )$ , with improvements observed across all tasks. Relative to the MTL baseline (ViT-L), performance increases from 63.15 to 65.91 mIoU for Sem. Seg and from 75.10 to 80.60 odsF for Edge, while Dep. RMSE is reduced from 0.5119 to 0.4209 and Norm. from 18.81 to 15.82 mErr. Compared with recent strong baselines (e.g., BIM, TaskDiffusion), $\mathrm { D ^ { 2 } M o E }$ shows a larger overall gain, with particularly pronounced benefits on geometry-related predictions, consistent with the importance of modelling indoor structure and fine-grained details.

Cityscapes. We further evaluate D<sup>2</sup>MoE on Cityscapes for joint Sem. Seg. and Dep. (Tab. 3). D<sup>2</sup>MoE attains 82.10 mIoU and 4.22 RMSE, corresponding to a $\Delta _ { m } ~ = ~ 5 . 3 0$ overall gain with the ViT-L baseline (76.45 mIoU and 4.36 RMSE). Notably, several existing approaches yield limited or negative overall gains under the same protocol, reflecting the optimisation trade-offs between semantic recognition and geometric estimation in urban driving scenes. We further observe that the relative advantage of $\mathrm { D ^ { 2 } M o E }$ is more evident on Cityscapes than on NYUD-v2 and Pascal-Context. This may stems from its regular driving layouts and tight semantic–geometric coupling, where roads, vehicles, and buildings follow stable spatial/depth patterns, making DINOv3 priors and task-aware routing more effective.

Across Pascal-Context, NYUD-v2 and Cityscapes, $\mathrm { D ^ { 2 } M o E }$ consistently yields the largest aggregated multi-task gain $\Delta _ { m } ,$ despite substantial differences in visual domains and task compositions. This pattern strongly supports our motivation that effective MTDP with a frozen foundation backbone requires (i) task-generic adaptation to align pretrained representations with dense prediction, and (ii) sparse, task-adaptive specialization to resolve conflicts.

## 4.3.2 Qualitative Comparison

We present qualitative results on Pascal-Context and NYUDv2 to visually assess D<sup>2</sup>MoE . Fig. 5 and Fig. 6 compare D<sup>2</sup>MoE with strong recent baselines (TaskDiffusion and MTMamba) under the same multi-task setting.

Pascal-Context. As shown in Fig. 5, D<sup>2</sup>MoE yields more accurate and spatially coherent predictions across tasks. For Sem. Seg., D<sup>2</sup>MoE better preserves object extent and thin structures (e.g., limbs), reducing the label fragmentation observed in competing methods. For Norm., $\mathrm { \Delta \vec { D } ^ { 2 } M o E }$ produces smoother orientation fields with sharper transitions at geometric discontinuities, indicating stronger local shape consistency. For Edge, D<sup>2</sup>MoE outputs sharper and more continuous contours aligned with object silhouettes. For Sal., D<sup>2</sup>MoE improves foreground–background separation with fewer spurious activations.

NYUD-v2. Fig. 6 demonstrates gains in indoor scene parsing. D<sup>2</sup>MoE yields Depth maps with refined geometry and reliable occlusion ordering, reducing planar noise. Norm. maps exhibit spatial consistency with crisp boundary transitions. For Sem. Seg., D<sup>2</sup>MoE successfully disambiguates adjacent objects via DINOv3 semantic priors. Finally, Edge results show improved completeness and connectivity even in cluttered environments.

![](images/5d7e0df23191ccfc48f3f4eb5afd68b0080dd252747ffa9e08e13a8c29716de7.jpg)  
Fig. 6. Qualitative Comparison on NYUD-v2. Compared with TaskDiffusion [35] and MTMamba [37], D<sup>2</sup>MoE better preserves depth/normal discontinuities at object boundaries and yields structurally coherent geometry that agrees with segmentation, suggesting stronger cross-task geometric consistency across heterogeneous dense objectives.

Edge  
Norm.  
H. Pars.  
Sal.  
![](images/b728d04401fba69e623035f24829f94af1360ccb6eb190cb77c1637c6f699660.jpg)  
Sem. Seg.

Edge  
Dep.  
Norm.  
Sem. Seg.  
![](images/87ef1f3c44194492e972dddf8f4a67229ac81c9e777ff5c5e91f20a8e0dd629c.jpg)  
(b) PCA visualization on NYUD-v2  
Fig. 7. PCA Visualizations on Pascal-Context (Left) and NYUD-v2 (Right). PCA visualizations are provided to analyze feature representations learned with TA-MoE, TG-LoRA, and the Semantic Prior Injector (SPI). As shown, features extracted from different network stages exhibit progressively clearer task-specific structures, demonstrating the effectiveness of the proposed modules in enhancing task-aware representation learning across both datasets.

## 4.4 Ablation Study

We conduct comprehensive ablation studies on the Pascal-Context dataset. In particular, §4.4.1 isolates the contributions of our proposed modules and their placement, whereas §4.4.2 examines the key hyperparameters that operationalize the intended balance between conditional specialization and parameter efficiency.

## 4.4.1 Component and Design Ablations

Overall Components Analysis. Tab. 4 disentangles the contributions of the Mixture of Task Adapters (MoTA) and Semantic Prior Injector (SPI) to our prior-preserving multitask framework. Relative to the MTL baseline, MoTA alone increases the aggregated gain from 0.00 to 6.10%, with marked improvements in Sal. (82.61 → 84.12 mIoU) and Norm. (15.60 → 14.01 mErr), supporting the need for taskconditioned adaptation to separate dense objectives without perturbing the frozen backbone. SPI alone provides a smaller gain $( \Delta _ { m } ~ = ~ 5 . 6 6 \% )$ , implying that semantic priors are insufficient to resolve cross-task interference under a single shared adaptation path. Enabling both yields the best performance, i.e.84.48 mIoU (segmentation), 78.05 mIoU (parsing), 85.55 maxF (saliency), and $\Delta _ { m } = 8 . 8 8 \% ,$ consistent with complementarity: MoTA supplies controlled specialization, whereas SPI re-injects frozen semantic structure to improve global coherence and boundary fidelity.

Ablation Results of MoTA. Tab. 5 probes MoTA’s internal factorization. Removing TG-LoRA lowers the gain to 7.74%, indicating degradation of the task-generic alignment channel that stabilizes downstream adaptation; removing TA-MoE produces a comparable reduction $( \Delta _ { m } = 7 . 9 7 \% )$ , reflecting weaker task-specific discrimination without conditional expert routing. The full model outperforms both variants across tasks, showing that frozen-backbone multi-task learning benefits from jointly coupling a globally shared low-rank substrate (TG-LoRA) with sparse task-adaptive specialization (TA-MoE) rather than relying on either uniform adaptation or specialization alone.

![](images/92c7512e94cd1a545e3784c177fdf294bbae4ff1ad01dcaf75210f437ac4e206.jpg)  
Fig. 8. Hyperparameter analysis of TG-LoRA and TA-MoE. The results demonstrate clear performance trends across different LoRA ranks and expert configurations, highlighting an effective trade-off between model capacity and multi-task performance.

TABLE 4  
Overall ablation of the proposed components (MoTA and SPI) under the unified multi-task setting.
<table><tr><td>MoTA</td><td>SPI</td><td>Sem. Seg. mIoU↑</td><td>H. Pars. mIoU↑</td><td>Sal. maxF↑</td><td>Norm. mErr↓</td><td>Edge odsF↑</td><td> $\begin{array} { c } { { \Delta _ { m } } } \\ { { ( \% ) \uparrow } } \end{array}$ </td></tr><tr><td>x</td><td>x</td><td>82.61</td><td>76.38</td><td>77.09</td><td>17.63</td><td>74.00</td><td>0.00</td></tr><tr><td>x</td><td>√</td><td>83.27</td><td>77.21</td><td>82.51</td><td>15.24</td><td>78.30</td><td>5.66</td></tr><tr><td>√</td><td>x</td><td>82.44</td><td>74.90</td><td>84.12</td><td>14.01</td><td>76.20</td><td>6.10</td></tr><tr><td>√</td><td>√</td><td>84.48</td><td>78.05</td><td>85.55</td><td>13.57</td><td>78.40</td><td>8.88</td></tr></table>

TABLE 5

Component ablation of Mixture of Task Adapters (MoTA), isolating the contributions of TG-LoRA and TA-MoE under the unified multi-task setting.
<table><tr><td>Modules</td><td>Sem. Seg. mIoU↑</td><td>H. Pars. mIoU↑</td><td>Sal. maxF↑</td><td>Norm. mErr↓</td><td>Edge odsF↑</td><td> $\Delta _ { m }$  (%)↑</td></tr><tr><td>MTL Baseline</td><td>82.61</td><td>76.38</td><td>77.09</td><td>17.63</td><td>74.00</td><td>0.00</td></tr><tr><td>w/o TG-LoRA</td><td>84.07</td><td>77.92</td><td>84.32</td><td>14.15</td><td>78.30</td><td>7.74</td></tr><tr><td>w/o TA-MoE</td><td>83.31</td><td>77.60</td><td>84.35</td><td>13.65</td><td>78.00</td><td>7.97</td></tr><tr><td>Ours</td><td>84.48</td><td>78.05</td><td>85.55</td><td>13.57</td><td>78.40</td><td>8.88</td></tr></table>

TABLE 6

Effect of inserting the MoE module at different locations within the backbone transformer block.
<table><tr><td>QKV</td><td>Proj</td><td>FFN</td><td>Sem. Seg mIoU↑</td><td>H. Pars. mIoU↑</td><td>Sal. maxF↑</td><td>Norm. mErr↓</td><td>Edge odsF↑</td><td> $\Delta _ { m }$  (%) ↑</td></tr><tr><td>x</td><td>x</td><td>x</td><td>83.31</td><td>77.60</td><td>84.35</td><td>13.65</td><td>78.00</td><td>7.97</td></tr><tr><td>√</td><td>x</td><td>x</td><td>83.74</td><td>77.80</td><td>85.18</td><td>13.57</td><td>78.70</td><td>8.62</td></tr><tr><td>x</td><td>√</td><td>x</td><td>83.31</td><td>77.67</td><td>84.86</td><td>13.73</td><td>78.80</td><td>8.24</td></tr><tr><td>x</td><td>x</td><td>√</td><td>83.34</td><td>77.24</td><td>84.63</td><td>13.58</td><td>78.00</td><td>8.03</td></tr><tr><td>√</td><td>√</td><td>x</td><td>84.48</td><td>78.05</td><td>85.55</td><td>13.57</td><td>78.40</td><td>8.88</td></tr><tr><td>√</td><td>√</td><td>√</td><td>83.75</td><td>77.51</td><td>85.02</td><td>13.51</td><td>79.10</td><td>8.68</td></tr></table>

$\Delta _ { m }$ s computed relative to the MTL baseline, which is omitted.

Ablation Results of TA-MoE. Tab. 6 evaluates where specialization most effectively alleviates the single-route bottleneck. Injecting TA-MoE into attention QKV projections outperforms insertion solely in the FFN or projection layers, indicating that conditioning token mixing (task-dependent contextual aggregation) is more effective than modifying downstream feed-forward transformations. Combining QKV and projection layers achieves the maximum gain (8.88%), suggesting that co-conditioning attention and output projection best supports task disentanglement.

Backbone Variants. Tab. 7 quantifies task-dependent scaling across ViT backbones. ViT-S to ViT-B gives the largest segmentation jump (80.62 → 85.25 mIoU) with modest parameter growth. Scaling to ViT-L shifts gains to structured prediction: H. Pars. 75.66 → 78.05 mIoU, Sal. and Norm. peak at 85.55 maxF and 13.57 mErr, while Sem. Seg. saturates at 84.48 mIoU. ViT-H adds slightly for H. Pars./Edge (78.87 mIoU, 79.60 odsF) but degrades Sal. (83.17 maxF) and Norm. (13.62 mErr), implying diminishing or negative returns from additional backbone capacity. This degradation likely arises because the larger ViT-H encodes stronger high-level semantic invariances, while our frozenbackbone and low-rank adaptation cannot fully recalibrate its spatial-frequency bias for saliency and normals, making these geometry-sensitive tasks prone to over-smoothing. Overall, ViT-L provides the optimal accuracy-efficiency trade-off.

Comparison with MLoRE under a Strong Backbone. Tab. 8 reimplements MLoRE on DINOv3. MLoRE places experts in the decoder, improving task outputs over the MTL baseline and substantiating decoder-side specialization. D<sup>2</sup>MoE instead instantiates experts in the backbone, decoupling tasks during representation learning while retaining DINOv3 priors through conditional adaptation. This shift delivers larger mean gains $( \Delta _ { m } = 8 . 8 8 \%$ vs. 6.30%) with substantially fewer trainable parameters (54M vs. 407M). Although D<sup>2</sup>MoE incurs higher FLOPs/latency than MLoRE (1029.9G/224.2ms vs. 829.9G/203.2ms), the additional computation is moderate and leads to stronger multi-task performance with a 7.5× smaller trainable footprint, demonstrating a favorable trade-off among accuracy, trainable footprint, and computational cost.

![](images/304d01c9ca1d208df0ce19866758b0194515c9d62e9dbf714797ca3d4e7e26e4.jpg)  
(a) t-SNE Visualization of Sem. Seg.

![](images/9a6854e749b13e658678cd39b7fd4274b480914dde7e2a1605c0b0d537401e20.jpg)  
(b) t-SNE Visualization of Dep.  
Fig. 9. 3D t-SNE Visualizations of Feature Embeddings. We compare the MTL baseline (top) with our D<sup>2</sup>MoE framework (bottom) across three benchmarks. (a) Semantic Segmentation: Points are colored by class label. D<sup>2</sup>MoE achieves significantly higher intra-class compactness and inter-class separability than the baseline. (b) Depth Estimation: Points are colored by discretized depth intervals (see legend). Our model yields more structured, monotonic groupings of scene geometry. Axes denote the three primary t-SNE components.

Comparison with Other Variants under Matched Budgets. Tab. 9 compares three parameter- and protocol-matched controls under the same frozen DINOv3 setting. Shared LoRA uses a single task-shared LoRA (r = 171) at the same QKV/projection positions, achieving $\Delta _ { m } ~ = ~ 8 . 4 8 \%$ . Task LoRA replaces routing with independent task-wise LoRA modules (r = 186), yielding $\Delta _ { m } = 8 . 3 1 \%$ . Decoder MoE moves the low-rank MoE to the decoder, using the same expert pool and Top-8 routing with 25.2M MoE parameters, but reaches only $\Delta _ { m } = 5 . 9 0 \%$ . In comparison, D<sup>2</sup>MoE achieves the best $\Delta _ { m } = 8 . 8 8 \%$ . Overall, these matched controls show that the gains of D<sup>2</sup>MoE are not explained by parameter count alone, but arise from combining shared low-rank alignment with sparse, learned task-aware specialization at appropriate backbone locations.

## 4.4.2 Hyperparameter Analysis

We quantify how hyperparameters shape the capacity– efficiency trade-off in Fig. 8, varying TG-LoRA rank, TA-MoE expert count, TA-MoE LoRA rank, and Top-k routing.

LoRA Rank of TG-LoRA. Increasing TG-LoRA rank from 2 to 16 (Fig. 8(a)) improves all tasks (Sem. Seg. 83.40 → 84.48 mIoU, H. Pars. 77.66 → 78.05 mIoU, Sal. 84.73 → 85.55 maxF). Consistently, $\Delta _ { m }$ increases from 8.07% to 8.88%, indicating stronger overall multi-task performance. All parameters rise modestly (355.5M → 357.6M). At rank 32, gains saturate or reverse (e.g., Sem. Seg. 84.35) despite higher parameters/FLOPs, indicating over-parameterization without improved alignment.

Number of Experts in TA-MoE. Varying experts (Fig. 8(b)) shows a similar saturation: 2→16 experts improves segmentation (83.47 → 84.48 mIoU) and boundary detection (78.30 → 78.50 odsF) with limited parameter growth (355.9M → 357.6M). The aggregated gain peaks at 16 experts with $\Delta _ { m } = 8 . 8 8 \%$ , suggesting the best balance between specialization capacity and efficiency. At 32 experts, performance saturates or slightly decreases (e.g., parsing 78.05 → 77.64), suggesting redundancy rather than stronger specialization.

TABLE 7  
Comparison of backbone variants under the same multi-task setting.
<table><tr><td>Backbone</td><td>Para. (M)</td><td>Sem. Seg. mIoU↑</td><td>Pars. mIoU↑</td><td>Sal. maxF↑</td><td>Norm. mErr↓</td><td>Edge odsF↑</td></tr><tr><td>ViT-S</td><td>62.0</td><td>80.62</td><td>73.36</td><td>85.28</td><td>13.79</td><td>76.40</td></tr><tr><td>ViT-B</td><td>126.6</td><td>85.25</td><td>75.66</td><td>84.95</td><td>13.66</td><td>77.50</td></tr><tr><td>ViT-L</td><td>357.6</td><td>84.48</td><td>78.05</td><td>85.55</td><td>13.57</td><td>78.40</td></tr><tr><td>ViT-H</td><td>911.3</td><td>84.80</td><td>78.87</td><td>83.17</td><td>13.62</td><td>79.60</td></tr></table>

LoRA Rank of TA-MoE. Fig. 8(c) indicates that moderate per-expert rank is sufficient: rank 4 achieves strong overall accuracy (segmentation 84.48 mIoU) and the lowest normal error (13.57 mErr) at 357.6M parameters. It also obtains the highest aggregated gain $( \Delta _ { m } \bar { = } 8 . 8 8 \% )$ , confirming that moderate rank is sufficient for overall multi-task capacity. Lower ranks underfit task-specific refinement, whereas higher ranks increase parameters/FLOPs with marginal, unstable gains.

Top-k Value in TA-MoE. Fig. 8(d) shows that increasing k from 2 to 8 improves performance, peaking at 84.48 mIoU (segmentation) and 85.55 maxF (saliency) with unchanged parameters. $\Delta _ { m }$ also reaches the best value of 8.88% when k = 8. Larger k (e.g., 12 or 16) yields marginal gains or mild degradation, consistent with reduced sparsity-driven specialization under broader expert activation.

Collectively, these ablations identify a coherent regime where improvements arise from structured task-adaptive design, including TG-LoRA alignment, sparse TA-MoE specialization, and SPI prior injection, rather than from indiscriminate increases in trainable capacity.

## 4.5 Feature Visualization and Analysis

Feature Embedding Visualization (t-SNE). Fig. 9 shows 3D t-SNE projections of the learned feature embeddings, directly comparing the MTL baseline (top row) with D<sup>2</sup>MoE (bottom row). D<sup>2</sup>MoE reshapes the embedding manifold into taskaligned submanifolds rather than an entangled cloud. On

![](images/dee3cdabcdd8129d7b9077439422a7631dd5480d8e6b36507e7068aed4baa418.jpg)

![](images/43a91c56af46554d736967a201df8aacb270b5b98bb7cc94e856ccc10292b92f.jpg)

![](images/6a0e4416059e0d2b959a0558d3fee1270035470b26c45ad82796444e46a6d344.jpg)  
Fig. 10. Router visualization of TA-MoE across backbone stages. We visualize the task-wise expert selection patterns of TA-MoE inserted at different backbone stages and positions. For each task, we aggregate expert activation values over all samples in the Pascal-Context dataset and compute the cosine similarity between task-specific activation vectors. The resulting similarity matrices reveal how task relationships evolve across stages and insertion locations.

<table><tr><td></td><td></td><td>Noise</td><td colspan="3"></td><td colspan="3">Blur</td><td colspan="3">Weather</td><td colspan="3">Digital</td><td colspan="3"></td><td>w/0</td></tr><tr><td>Level 1</td><td>82.34</td><td>82.58</td><td>81.61</td><td></td><td>83.27</td><td>82.81</td><td>83.18</td><td>71.25</td><td>82.41</td><td>82.06</td><td>83.45</td><td>84.33</td><td>84.42</td><td>83.03</td><td>84.12</td><td>83.86</td><td>84.48</td><td>80</td></tr><tr><td>Se Seg. Level 3</td><td>76.02</td><td>77.20</td><td></td><td>76.65</td><td>79.45</td><td>75.95</td><td>78.35</td><td>59.70</td><td>78.65</td><td>74.82</td><td>82.44</td><td>83.71</td><td>84.09</td><td>79.39</td><td>83.45</td><td>83.43</td><td>84.48</td><td>60</td></tr><tr><td>Level 5</td><td>47.50</td><td></td><td>52.58</td><td>50.47</td><td>70.96</td><td>62.13</td><td>68.07</td><td>47.60</td><td>74.97</td><td>70.67</td><td>77.90</td><td>82.84</td><td>81.95</td><td>70.04</td><td>82.04</td><td>81.52</td><td>84.48</td><td></td></tr><tr><td></td><td>Gauss.</td><td>Shot.</td><td></td><td>Imp.</td><td>Defoc.</td><td>Glass.</td><td>Mot.</td><td>Zoom.</td><td>Snow</td><td>Frost</td><td>Fog</td><td>Bright.</td><td>Contr.</td><td>Elas.</td><td>Pixel.</td><td>JPEG.</td><td>Clean</td><td></td></tr><tr><td>Level 1</td><td>75.98</td><td></td><td>76.47</td><td>75.78</td><td>77.52</td><td>76.32</td><td>77.41</td><td>65.62</td><td>76.00</td><td>75.39</td><td>77.45</td><td>77.91</td><td>78.06</td><td>77.21</td><td>77.90</td><td>77.37</td><td>78.05</td><td>70</td></tr><tr><td>H.rs Level 3</td><td>71.25</td><td>72.23</td><td></td><td>71.89</td><td>73.84</td><td>69.63</td><td>71.87</td><td>57.10</td><td>72.92</td><td>69.50</td><td>76.61</td><td>77.47</td><td>77.84</td><td>75.02</td><td>77.40</td><td>76.43</td><td>78.05</td><td>60</td></tr><tr><td>Level 5</td><td>58.75</td><td>60.55</td><td></td><td>61.03</td><td>65.88</td><td>62.97</td><td>62.85</td><td>49.67</td><td>70.49</td><td>66.40</td><td>73.60</td><td>76.64</td><td>76.58</td><td>71.55</td><td>75.32</td><td>73.10</td><td>78.05</td><td>50</td></tr><tr><td></td><td>Gauss.</td><td>Shot.</td><td></td><td>Imp.</td><td>Defoc.</td><td>Glass.</td><td>Mot.</td><td>Zoom.</td><td>Snow</td><td>Frost</td><td>Fog</td><td>Bright.</td><td>Contr.</td><td>Elas.</td><td>Pixel.</td><td>JPEG.</td><td>Clean</td><td></td></tr><tr><td>Level 1</td><td>83.43</td><td>83.34</td><td></td><td>83.02</td><td>73.70</td><td>75.77</td><td>81.64</td><td>77.05</td><td>84.33</td><td>84.84</td><td>84.30</td><td>85.45</td><td>85.35</td><td>85.04</td><td>84.62</td><td>81.33</td><td>85.55</td><td>80</td></tr><tr><td>Sal. Level 3</td><td>78.61</td><td>78.40</td><td></td><td>78.81</td><td>68.03</td><td>67.04</td><td>77.46</td><td>73.02</td><td>81.87</td><td>81.52</td><td>82.24</td><td>84.80</td><td>84.78</td><td>84.39</td><td>82.20</td><td>79.99</td><td>85.55</td><td>70</td></tr><tr><td>Level 5</td><td>61.52</td><td>62.60</td><td></td><td>62.29</td><td>65.55</td><td>60.95</td><td>73.55</td><td>67.99</td><td>80.39</td><td>79.53</td><td>73.85</td><td>83.60</td><td>79.53</td><td>82.18</td><td>79.42</td><td>78.56</td><td>85.55</td><td></td></tr><tr><td></td><td>Gauss.</td><td>Shot.</td><td></td><td>Imp.</td><td>Defoc.</td><td>Glass.</td><td>Mot.</td><td>Zoom.</td><td>Snow</td><td>Frost</td><td>Fog</td><td>Bright.</td><td>Contr.</td><td>Elas.</td><td>Pixel.</td><td>JPEG.</td><td>Clean</td><td></td></tr><tr><td>N0orm&#x27; Level 1</td><td>16.52</td><td>16.52</td><td></td><td>18.79</td><td>14.84</td><td>14.29</td><td>14.19</td><td>18.62</td><td>16.26</td><td>15.87</td><td>15.19</td><td>13.69</td><td>13.65</td><td>13.87</td><td>13.54</td><td>13.79</td><td>13.57</td><td>30</td></tr><tr><td>Level 3</td><td>23.40</td><td>22.96</td><td></td><td>25.17</td><td>17.39</td><td>17.11</td><td>16.94</td><td>21.01</td><td>19.84</td><td>19.75</td><td>16.68</td><td>14.51</td><td>14.00</td><td>14.90</td><td>13.83</td><td>14.16</td><td>13.57</td><td>20</td></tr><tr><td>Level 5</td><td>33.42</td><td>32.20</td><td></td><td>33.59</td><td>19.67</td><td>20.55</td><td>19.92</td><td>22.61</td><td>21.65</td><td>21.02</td><td>19.58</td><td>15.82</td><td>15.61</td><td>16.25</td><td>14.65</td><td>15.58</td><td>13.57</td><td></td></tr><tr><td></td><td>Gauss.</td><td>Shot.</td><td></td><td>Imp.</td><td>Defoc.</td><td>Glass.</td><td>Mot.</td><td>Zoom.</td><td>Snow</td><td>Frost</td><td>Fog</td><td>Bright.</td><td>Contr.</td><td>Elas.</td><td>Pixel.</td><td>JPEG.</td><td>Clean</td><td></td></tr><tr><td>Level 1</td><td>76.80</td><td>77.00</td><td></td><td>76.50</td><td>77.80</td><td>74.20</td><td>77.90</td><td>53.20</td><td>76.90</td><td>76.70</td><td>78.10</td><td>78.80</td><td>78.80</td><td>77.50</td><td>78.40</td><td>77.50</td><td>78.40</td><td>70</td></tr><tr><td>Ede Level 3</td><td>72.00</td><td>72.80</td><td></td><td>72.80</td><td>72.00</td><td>59.50</td><td>68.50</td><td>44.70</td><td>74.10</td><td>71.70</td><td>77.10</td><td>78.10</td><td>78.40</td><td>74.60</td><td>77.70</td><td>76.30</td><td>78.40</td><td>60</td></tr><tr><td>Level 5</td><td>60.20</td><td>62.50</td><td></td><td>61.90</td><td>57.60</td><td>53.30</td><td>54.50</td><td>40.10</td><td>71.30</td><td>69.40</td><td>74.30</td><td>76.90</td><td>75.20</td><td>69.90</td><td>74.90</td><td>72.40</td><td>78.40</td><td>50</td></tr><tr><td></td><td>Gauss.</td><td>Shot.</td><td></td><td>Imp.</td><td>Defoc.</td><td>Glass.</td><td>Mot.</td><td>Zoom.</td><td>Snow</td><td>Frost</td><td>Fog</td><td>Bright.</td><td>Contr.</td><td>Elas.</td><td>Pixel.</td><td>JPEG.</td><td>Clean</td><td></td></tr></table>

Fig. 11. Corruption Robustness Evaluation on Pascal-Context. Across five tasks, performance decreases with corruption severity, with zoom blur and gaussian noise inducing the largest degradations (e.g., Sem. Seg. drops from 84.48 on clean to 47.50 at Level 5). Weather/digital corruptions are comparatively mild at Levels 1–3, while normal error rises primarily under noise

![](images/5c4b637c85d55a7e44fb71063c33acfe4746918207be604e9206f7fa35d1e5e4.jpg)  
Fig. 12. Corruption Robustness Comparison with TaskDiffusion [35]. All examples are drawn from the Pascal-Context dataset under level-5 corruptions. Our method demonstrates stronger robustness to severe degradations, yielding cleaner object boundaries and more stable predictions.

TABLE 8  
Comparison with MLoRE using a DINOv3 backbone.
<table><tr><td>Modules</td><td>Sem. Seg.  $\mathrm { m I o U \uparrow }$ </td><td>H. Pars. mIoU↑</td><td>Sal.  $\mathrm { m a x F \uparrow }$ </td><td>Norm. mErr↓</td><td>Edge  $\mathrm { o d s } \mathrm { \bar { F } } \mathrm { \uparrow }$ </td><td> $\Delta _ { m }$   $( \% ) \uparrow$ </td><td>Para.  $( M ) \downarrow$ </td><td>Train. Para. (M) ↓</td><td>Flops (G)↓)</td><td>Latency (ms)↓</td></tr><tr><td>MTL Baseline</td><td>82.61</td><td>76.38</td><td>77.09</td><td>17.63</td><td>74.00</td><td>0.00</td><td>328</td><td>25</td><td>759.7</td><td>198.9</td></tr><tr><td>MLoRE [34]</td><td>83.85</td><td>78.05</td><td>84.98</td><td>15.51</td><td>78.10</td><td>6.30</td><td>407</td><td>407</td><td>829.9</td><td>203.2</td></tr><tr><td>D2MoE</td><td>84.48</td><td>78.05</td><td>85.55</td><td>13.57</td><td>78.40</td><td>8.88</td><td>358</td><td>54</td><td>1029.9</td><td>224.2</td></tr></table>

Sem. Seg. (Pascal-Context, NYUD-v2, Cityscapes), D<sup>2</sup>MoE forms tighter, well-separated clusters with higher intra-class compactness and reduced cross-class overlap than the MTL baseline, reflecting increased inter-class variance and less neg ative transfer. This behavior matches selective routing within a shared backbone: early layers preserve broadly reusable cues (edges, textures), while routed experts enhance classconditional directions, widening margins between semantic modes while maintaining the global manifold structure. Regarding depth estimation, D<sup>2</sup>MoE yields clearer and more structured groupings, suggesting routing induces a more geometry-aware embedding where features are organized topologically according to their respective depth bins, a marked improvement over the disordered feature clouds observed in the baseline.

TABLE 9  
Comparison with shared LoRA, task-specific LoRA and decoder-side MoE under matched trainable-parameter budgets.
<table><tr><td>Method</td><td>Sem. Seg. mIoU↑</td><td>H. Pars. mIoU↑</td><td> $\mathbf { s a l . }$  maxF↑</td><td>Norm. mErr↓</td><td> $\mathbf { E d g e }$   $\mathrm { o d s } \mathrm { \bar { F } } \uparrow$ </td><td> $\Delta _ { m }$   $( \% ) \uparrow$ </td><td>Backbone Train. (M)↓</td><td>Decoder Train.  $( \mathbf { M } ) \downarrow$ </td><td>Train. Para. (M)↓</td><td>Para. (M)↓</td></tr><tr><td>MTL Baseline</td><td>82.61</td><td>76.38</td><td>77.09</td><td>17.63</td><td>74.00</td><td>0.00</td><td>25.2</td><td>0</td><td>25.2</td><td>328.4</td></tr><tr><td>Shared LoRA</td><td>83.60</td><td>77.52</td><td>85.19</td><td>13.51</td><td>78.30</td><td>8.48</td><td>25.2</td><td>29.2</td><td>54.4</td><td>357.6</td></tr><tr><td>Task LoRA</td><td>84.25</td><td>77.95</td><td>84.65</td><td>13.77</td><td>78.30</td><td>8.31</td><td>25.2</td><td>29.2</td><td>54.4</td><td>357.6</td></tr><tr><td>Decoder MoE</td><td>83.22</td><td>78.00</td><td>82.46</td><td>15.16</td><td>78.20</td><td>5.90</td><td>0</td><td>54.4</td><td>54.4</td><td>357.6</td></tr><tr><td>D2MoE</td><td>84.88</td><td>78.05</td><td>85.55</td><td>13.57</td><td>78.40</td><td>8.88</td><td>25.2</td><td>29.2</td><td>54.4</td><td>357.6</td></tr></table>

TABLE 10

Averaged robustness comparison on PASCAL-Context across all corruptions.
<table><tr><td>Method</td><td>Sem. Seg. mIoU↑</td><td>H. Pars. mIoU↑</td><td>Sal. maxF↑</td><td>Norm. mErr↓</td><td>Edge odsF↑</td></tr><tr><td>MTL Baseline</td><td>65.36</td><td>64.54</td><td>70.08</td><td>22.45</td><td>63.07</td></tr><tr><td>TaskDiffusion</td><td>57.01</td><td>49.48</td><td>67.42</td><td>22.81</td><td>59.04</td></tr><tr><td>D2MoE</td><td>68.08</td><td>67.03</td><td>72.77</td><td>21.47</td><td>64.96</td></tr></table>

<table><tr><td rowspan=1 colspan=1>97.35</td><td rowspan=1 colspan=1>91.34</td><td rowspan=1 colspan=1>79.12</td><td rowspan=1 colspan=1>97.44</td><td rowspan=1 colspan=1>92.59</td><td rowspan=1 colspan=1>80.59</td><td rowspan=1 colspan=1>96.10</td><td rowspan=1 colspan=1>88.24</td><td rowspan=1 colspan=1>70.25</td></tr><tr><td rowspan=1 colspan=1>97.71</td><td rowspan=1 colspan=1>89.76</td><td rowspan=1 colspan=1>84.50</td><td rowspan=1 colspan=1>97.58</td><td rowspan=1 colspan=1>93.18</td><td rowspan=1 colspan=1>85.88</td><td rowspan=1 colspan=1>95.17</td><td rowspan=1 colspan=1>87.09</td><td rowspan=1 colspan=1>71.14</td></tr><tr><td rowspan=1 colspan=1>98.88</td><td rowspan=1 colspan=1>94.25</td><td rowspan=1 colspan=1>90.91</td><td rowspan=1 colspan=1>96.10</td><td rowspan=1 colspan=1>92.20</td><td rowspan=1 colspan=1>85.45</td><td rowspan=1 colspan=1>97.88</td><td rowspan=1 colspan=1>95.61</td><td rowspan=1 colspan=1>90.75</td></tr><tr><td rowspan=1 colspan=1>99.04</td><td rowspan=1 colspan=1>87.28</td><td rowspan=1 colspan=1>78.53</td><td rowspan=1 colspan=1>88.63</td><td rowspan=1 colspan=1>74.93</td><td rowspan=1 colspan=1>63.20</td><td rowspan=1 colspan=1>90.15</td><td rowspan=1 colspan=1>78.82</td><td rowspan=1 colspan=1>67.08</td></tr><tr><td rowspan=1 colspan=1>97.57Edge</td><td rowspan=1 colspan=1>92.26</td><td rowspan=1 colspan=1>85.23</td><td rowspan=1 colspan=1>96.61</td><td rowspan=1 colspan=1>91.01</td><td rowspan=1 colspan=1>82.86</td><td rowspan=1 colspan=1>94.98</td><td rowspan=1 colspan=1>87.33</td><td rowspan=1 colspan=1>78.84</td></tr><tr><td rowspan=1 colspan=1>-1.89</td><td rowspan=1 colspan=1>-9.39</td><td rowspan=1 colspan=1>-17.52</td><td rowspan=1 colspan=1>-5.02</td><td rowspan=1 colspan=1>-12.89</td><td rowspan=1 colspan=1>-24.69</td><td rowspan=1 colspan=1>-5.36</td><td rowspan=1 colspan=1>-13.72</td><td rowspan=1 colspan=1>-27.62</td></tr></table>

Fig. 13. Clean-normalized Performance Retention under Corruption. Task-wise retention for the MTL baseline, D<sup>2</sup>MoE, and TaskDiffusion. Avg. ∆ denotes mean relative degradation from clean performance.

Router Correlation Analysis. Fig. 10 reveals stage- and operator-specific specialization. In QKV projections, early stages display strong cross-task coupling (off-diagonal correlations), whereas deeper stages become near diagonaldominant, implying D<sup>2</sup>MoE progressively privatizes attention access patterns (queries/keys/values) to mitigate negative transfer in information selection. Conversely, attention output projections exhibit structured, non-monotonic sharing: an early block structure (notably among Hum./Nor./Sem.) persists, and late-stage correlations remain strong for specific pairs (e.g., Nor.–Sem., Hum.–Sal.), indicating D<sup>2</sup>MoE avoids forced full separation and reuses experts when supervision is complementary. Overall, TA-MoE specializes where tasks attend (QKV) while selectively sharing what is read out (output projection), disentangling conflicts while enabling transfer when tasks align.

## 4.6 Robustness to Distribution Shift

We assess domain generalization on Pascal-Context under the corruption benchmark of ImageNet [84]. Fig. 11 reports corruption-wise performance across severities (1/3/5), and Fig. 12 visualizes level-5 out-of-distribution (OOD) predictions against TaskDiffusion [35] and MTMamba [37]. To separate corruption sensitivity from gains due to higher clean accuracy, Fig. 13 additionally reports clean-normalized performance retention, while Tab. 10 reports severity-5 results averaged over all 15 corruptions.

Across all tasks, the dominant trend in Fig. 11 is graded performance change with increasing corruption severity, indicating stable behavior under covariate shift. Noise corruptions are consistently the most damaging: Sem. Seg. drops from 82.34 (Gaussian, level-1) to 47.50 (level-5), and H. Pars. from 75.98 to 58.75 under the same shift, while surface-normal error increases smoothly from 16.52 (clean) to 33.42 (Gaussian, level-5). In the blur family, zoom blur is the primary failure case across tasks (e.g., Sem. Seg. 71.25 → 47.60, H. Pars. 65.62 → 49.67, Edge 53.20 → 40.10 from level-1 to level-5), whereas defocus/motion blur produce substantially smaller degradations. Weather and digital corruptions are comparatively benign for most tasks: for Sem. Seg., level-5 contrast remains close to clean (81.95 vs. 84.48), and saliency stays high across many level-5 weather/digital shifts, demonstrating broad OOD robustness beyond a single corruption type.

Tab. 10 shows that D<sup>2</sup>MoE achieves the best average corrupted performance across all five tasks, outperforming both the MTL baseline and TaskDiffusion. Fig. 13 provides a complementary normalized comparison: D<sup>2</sup>MoE shows smaller average degradation than TaskDiffusion at all severities (−5.02%/ − 12.89%/ − 24.69% vs. −5.36%/ − 13.72%/ − 27.62%), with stronger retention particularly on Sem. Seg. and H. Pars., whereas the MTL baseline exhibits smaller overall degradation $( - 1 . 8 9 \% / - 9 . 3 9 \% / - 1 7 . 5 2 \% )$ Thus, D<sup>2</sup>MoE’s superior corrupted performance stems from both stronger clean performance and improved retention over TaskDiffusion, rather than uniformly lower corruption sensitivity.

Fig. 12 qualitatively confirms these trends under severe shift. Under contrast change and defocus blur, D<sup>2</sup>MoE preserves coherent region topology in Sem. Seg. and H. Pars. and maintains continuous edge structure. Even under

Gaussian noise which corresponds to the largest metric drops, D<sup>2</sup>MoE retains structured masks, plausible part layouts, and recognizable boundaries, whereas TaskDiffusion [35] exhibits region fragmentation, unstable part layouts, and de graded boundary continuity. Compared with MTMamba [37], D<sup>2</sup>MoE better preserves foreground completeness and taskconsistent structures, avoiding over-smoothing and missing responses, especially in saliency and edge predictions.

Overall, the absolute and clean-normalized results provide complementary evidence: D<sup>2</sup>MoE maintains stronger absolute OOD performance and degrades less than TaskDiffusion on average, while the MTL baseline exhibits smaller overall normalized degradation despite D<sup>2</sup>MoE showing better retention on selected tasks.

## 5 CONCLUSION

In this work, we addressed the challenge of multi-task dense prediction in the vision foundation model paradigm, where heterogeneous tasks must be adapted without compromising the robustness of pretrained representations. We proposed D<sup>2</sup>MoE , which introduces a structured Mixture of Task Adapters to decouple task-generic alignment from taskspecific specialization atop a frozen DINOv3. By combining Task-Generic Low-Rank Adaptation across all attention layers with a Task-Adaptive Mixture of Experts selectively applied in later stages, D<sup>2</sup>MoE resolves task conflicts while preserving a shared semantic prior within a single forward pass. In addition, semantic prior injection further improves structural coherence and cross-task consistency. Extensive experiments across diverse benchmarks demonstrate that this design effectively mitigates negative transfer and achieves strong multi-task performance with high parameter efficiency. Overall, our results highlight the importance of layer-aware, conditional adaptation for unlocking the full potential of vision foundation models in multi-task dense prediction.

## REFERENCES

[1] S. Vandenhende, S. Georgoulis, W. Van Gansbeke, M. Proesmans, D. Dai, and L. Van Gool, “Multi-task learning for dense prediction tasks: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 7, pp. 3614–3633, 2021.

[2] H. Ye and D. Xu, “Taskprompter: Spatial-channel multi-task prompting for dense scene understanding,” in Int. Conf. Learn. Represent., 2023.

[3] F. Jiang, S. Wang, and X. Gong, “Task-conditional adapter for multitask dense prediction,” in ACM Int. Conf. Multimedia, 2024, pp. 2059–2068.

[4] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in IEEE Conf. Comput. Vis. Pattern Recog., 2015, pp. 3431–3440.

[5] E. Shelhamer, J. Long, and T. Darrell, “Fully convolutional networks for semantic segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 4, pp. 640–651, 2017.

[6] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in IEEE Conf. Comput. Vis. Pattern Recog., 2017, pp. 2881– 2890.

[7] V. Badrinarayanan, A. Kendall, and R. Cipolla, “Segnet: A deep convolutional encoder-decoder architecture for image segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 12, pp. 2481–2495, 2017.

[8] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 40, no. 4, pp. 834–848, 2018.

[9] S. Zheng, J. Lu, H. Zhao, X. Zhu, Z. Luo, Y. Wang, Y. Fu, J. Feng, T. Xiang, P. H. Torr et al., “Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers,” in IEEE Conf. Comput. Vis. Pattern Recog., 2021, pp. 6881–6890.

[10] R. Strudel, R. Garcia, I. Laptev, and C. Schmid, “Segmenter: Transformer for semantic segmentation,” in Int. Conf. Comput. Vis., 2021, pp. 7262–7272.

[11] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” Adv. Neural Inform. Process. Syst., vol. 34, pp. 12 077–12 090, 2021.

[12] T. Zhou, W. Wang, E. Konukoglu, and L. Van Gool, “Rethinking semantic segmentation: A prototype view,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 2582–2593.

[13] Z. Huang, X. Wang, Y. Wei, L. Huang, H. Shi, W. Liu, and T. S. Huang, “Ccnet: Criss-cross attention for semantic segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 6, pp. 6896–6908, 2023.

[14] Y. Zhu, Z. Zhang, C. Wu, Z. Zhang, T. He, H. Zhang, R. Manmatha, M. Li, and A. Smola, “Improving semantic segmentation via efficient self-training,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 3, pp. 1589–1602, 2024.

[15] F. Liu, C. Shen, and G. Lin, “Deep convolutional neural fields for depth estimation from a single image,” in IEEE Conf. Comput. Vis. Pattern Recog., 2015, pp. 5162–5170.

[16] H. Fu, M. Gong, C. Wang, K. Batmanghelich, and D. Tao, “Deep ordinal regression network for monocular depth estimation,” in IEEE Conf. Comput. Vis. Pattern Recog., 2018, pp. 2002–2011.

[17] A. Gordon, H. Li, R. Jonschkowski, and A. Angelova, “Depth from videos in the wild: Unsupervised monocular depth learning from unknown cameras,” in Int. Conf. Comput. Vis., 2019, pp. 8977–8986.

[18] S. F. Bhat, I. Alhashim, and P. Wonka, “Adabins: Depth estimation using adaptive bins,” in IEEE Conf. Comput. Vis. Pattern Recog., 2021, pp. 4009–4018.

[19] J.-W. Bian, H. Zhan, N. Wang, Z. Li, L. Zhang, C. Shen, M.-M. Cheng, and I. Reid, “Unsupervised scale-consistent depth learning from video,” Int. J. Comput. Vis., vol. 129, no. 9, pp. 2548–2564, 2021.

[20] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth anything: Unleashing the power of large-scale unlabeled data,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 10 371–10 381.

[21] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” Adv. Neural Inform. Process. Syst., vol. 37, pp. 21 875–21 911, 2024.

[22] W. Wang, Q. Lai, H. Fu, J. Shen, H. Ling, and R. Yang, “Salient object detection in the deep learning era: An in-depth survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 6, pp. 3239–3259, 2022.

[23] N. Liu and J. Han, “Dhsnet: Deep hierarchical saliency network for salient object detection,” in IEEE Conf. Comput. Vis. Pattern Recog., June 2016.

[24] L. Wang, H. Lu, Y. Wang, M. Feng, D. Wang, B. Yin, and X. Ruan, “Learning to detect salient objects with image-level supervision,” in IEEE Conf. Comput. Vis. Pattern Recog., 2017, pp. 136–145.

[25] Q. Hou, M.-M. Cheng, X. Hu, A. Borji, Z. Tu, and P. H. S. Torr, “Deeply supervised salient object detection with short connections,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 4, pp. 815–828, 2019.

[26] Y. Zhuge, Y. Zeng, and H. Lu, “Deep embedding features for salient object detection,” in Proc. AAAI Conf. Artif. Intell., vol. 33, no. 01, 2019, pp. 9340–9347.

[27] H. Chen, Y. Li, Y. Deng, and G. Lin, “Cnn-based rgb-d salient object detection: Learn, select, and fuse,” Int. J. Comput. Vis., vol. 129, no. 7, pp. 2076–2096, 2021.

[28] M. Caron, H. Touvron, I. Misra, H. Jegou, J. Mairal, P. Bojanowski,´ and A. Joulin, “Emerging properties in self-supervised vision transformers,” in Int. Conf. Comput. Vis., 2021, pp. 9650–9660.

[29] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.

[30] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose,´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.

[31] H. Ye and D. Xu, “Inverted pyramid multi-task transformer for dense scene understanding,” in Eur. Conf. Comput. Vis. Springer, 2022, pp. 514–530.

[32] H. Ye and D. Xu, “Invpt++: Inverted pyramid multi-task transformer for visual scene understanding,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 7493–7508, 2024.

[33] H. Ye and D. Xu, “Taskexpert: Dynamically assembling multi-task representations with memorial mixture-of-experts,” in Int. Conf. Comput. Vis., 2023, pp. 21 828–21 837.

[34] Y. Yang, P.-T. Jiang, Q. Hou, H. Zhang, J. Chen, and B. Li, “Multitask dense prediction via mixture of low-rank experts,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 27 927–27 937.

[35] Y. Yang, P.-T. Jiang, Q. Hou, H. Zhang, J. Chen, and B. Li, “Multitask dense predictions via unleashing the power of diffusion,” in Int. Conf. Learn. Represent., 2025.

[36] Y. Lu, S. Cao, and Y.-X. Wang, “Swiss army knife: Synergizing biases in knowledge from vision foundation models for multi-task learning,” in Int. Conf. Learn. Represent., 2025. [Online]. Available: https://openreview.net/forum?id=eePww5u7J3

[37] B. Lin, W. Jiang, P. Chen, Y. Zhang, S. Liu, and Y.-C. Chen, “Mtmamba: Enhancing multi-task dense scene understanding by mamba-based decoders,” in Eur. Conf. Comput. Vis. Springer, 2024, pp. 314–330.

[38] B. Lin, W. Jiang, P. Chen, S. Liu, and Y.-C. Chen, “Mtmamba++: Enhancing multi-task dense scene understanding via mamba-based decoders,” IEEE Trans. Pattern Anal. Mach. Intell., 2025.

[39] M. Cao, S. Zhou, Y. Li, Y. Deng, W. Huang, and L. Wang, “Enhancing mamba decoder with bidirectional interaction in multitask dense prediction,” in Int. Conf. Comput. Vis., 2025, pp. 18 815– 18 824.

[40] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” J. Mach. Learn. Res., vol. 23, no. 120, pp. 1–39, 2022.

[41] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat et al., “Glam: Efficient scaling of language models with mixture-of-experts,” in Int. Conf. Mach. Learn. PMLR, 2022, pp. 5547–5569.

[42] J. Zhang, J. Fan, P. Ye, B. Zhang, H. Ye, B. Li, Y. Cai, and T. Chen, “Bridgenet: Comprehensive and effective feature interactions via bridge feature for multi-task dense predictions,” IEEE Trans. Pattern Anal. Mach. Intell., 2025.

[43] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, “Gshard: Scaling giant models with conditional computation and automatic sharding,” in Int. Conf. Learn. Represent., 2021.

[44] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. Susano Pinto, D. Keysers, and N. Houlsby, “Scaling vision with sparse mixture of experts,” Adv. Neural Inform. Process. Syst., vol. 34, pp. 8583–8595, 2021.

[45] Y. Zhou, T. Lei, H. Liu, N. Du, Y. Huang, V. Zhao, A. M. Dai, Q. V. Le, J. Laudon et al., “Mixture-of-experts with expert choice routing,” Adv. Neural Inform. Process. Syst., vol. 35, pp. 7103–7114, 2022.

[46] A. Agiza, M. Neseem, and S. Reda, “Mtlora: Low-rank adaptation approach for efficient multi-task learning,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 16 196–16 205.

[47] Y. Lu, S. Sirejiding, Y. Ding, C. Wang, and H. Lu, “Prompt guided transformer for multi-task dense prediction,” IEEE Trans. Multimedia, vol. 26, pp. 6375–6385, 2024.

[48] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus, “Indoor segmentation and support inference from rgbd images,” in Eur. Conf. Comput. Vis. Springer, 2012, pp. 746–760.

[49] Y.-C. Liu, C.-Y. Ma, J. Tian, Z. He, and Z. Kira, “Polyhistor: Parameter-efficient multi-task adaptation for dense vision tasks,” Adv. Neural Inform. Process. Syst., vol. 35, pp. 36 889–36 901, 2022.

[50] Y. Xin, J. Du, Q. Wang, Z. Lin, and K. Yan, “Vmt-adapter: Parameter-efficient transfer learning for multi-task dense scene understanding,” in Proc. AAAI Conf. Artif. Intell., vol. 38, no. 14, 2024, pp. 16 085–16 093.

[51] S. Baek, S. Lee, H. Jo, H. Choi, and D. Min, “Tadformer: Taskadaptive dynamic transformer for efficient multi-task learning,” in IEEE Conf. Comput. Vis. Pattern Recog., 2025, pp. 14 858–14 868.

[52] K. S. I. Mantri, C.-B. Schonlieb, B. Ribeiro, C. Baskin, and M. Eliasof,¨ “Ditask: Multi-task fine-tuning with diffeomorphic transformations,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 25 218–25 229.

[53] Z. Fan, R. Sarkar, Z. Jiang, T. Chen, K. Zou, Y. Cheng, C. Hao, Z. Wang et al., “M<sup>3</sup>vit: Mixture-of-experts vision transformer for efficient multi-task learning with model-accelerator co-design,” Adv. Neural Inform. Process. Syst., vol. 35, pp. 28 441–28 457, 2022.

[54] Z. Chen, Y. Shen, M. Ding, Z. Chen, H. Zhao, E. G. Learned-Miller, and C. Gan, “Mod-squad: Designing mixtures of experts as modular multi-task learners,” in IEEE Conf. Comput. Vis. Pattern Recog., 2023, pp. 11 828–11 837.

[55] T. Chen, X. Chen, X. Du, A. Rashwan, F. Yang, H. Chen, Z. Wang, and Y. Li, “Adamv-moe: Adaptive multi-task vision mixture-ofexperts,” in Int. Conf. Comput. Vis., 2023, pp. 17 346–17 357.

[56] I. Misra, A. Shrivastava, A. Gupta, and M. Hebert, “Cross-stitch networks for multi-task learning,” in IEEE Conf. Comput. Vis. Pattern Recog., 2016, pp. 3994–4003.

[57] I. Kokkinos, “Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory,” in IEEE Conf. Comput. Vis. Pattern Recog., 2017, pp. 6129–6138.

[58] D. Xu, W. Ouyang, X. Wang, and N. Sebe, “Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing,” in IEEE Conf. Comput. Vis. Pattern Recog., 2018, pp. 675–684.

[59] S. Liu, E. Johns, and A. J. Davison, “End-to-end multi-task learning with attention,” in IEEE Conf. Comput. Vis. Pattern Recog., 2019, pp. 1871–1880.

[60] S. Vandenhende, S. Georgoulis, and L. Van Gool, “Mti-net: Multiscale task interaction networks for multi-task learning,” in Eur. Conf. Comput. Vis. Springer, 2020, pp. 527–543.

[61] A. R. Zamir, A. Sax, W. Shen, L. J. Guibas, J. Malik, and S. Savarese, “Taskonomy: Disentangling task transfer learning,” in IEEE Conf. Comput. Vis. Pattern Recog., June 2018, pp. 3712–3722.

[62] X. Xu et al., “Mtformer: Multi-task learning via transformer and cross-task reasoning,” in Eur. Conf. Comput. Vis., 2022.

[63] S. Yang, H. Ye, and D. Xu, “Contrastive multi-task dense prediction,” in Proc. AAAI Conf. Artif. Intell., 2023.

[64] D. Bruggemann, M. Kanakis, A. Obukhov, S. Georgoulis, and¨ L. Van Gool, “Exploring relational context for multi-task dense prediction,” in Int. Conf. Comput. Vis., 2021, pp. 15 869–15 878.

[65] W.-H. Li, X. Liu, and H. Bilen, “Learning multiple dense prediction tasks from partially annotated data,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 18 879–18 889.

[66] Y. Xu, Y. Yang, and L. Zhang, “Multi-task learning with knowledge distillation for dense prediction,” in Int. Conf. Comput. Vis., 2023, pp. 21 550–21 559.

[67] H. Huang, Y. Huang, L. Lin, R. Tong, Y.-W. Chen, H. Zheng, Y. Li, and Y. Zheng, “Going beyond multi-task dense prediction with synergy embedding models,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 28 181–28 190.

[68] Y. Shang, D. Xu, G. Liu, R. R. Kompella, and Y. Yan, “Efficient multitask dense predictor via binarization,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 15 899–15 908.

[69] M. Awais, M. Naseer, S. Khan, R. M. Anwer, H. Cholakkal, M. Shah, M.-H. Yang, and F. S. Khan, “Foundation models defining a new era in vision: A survey and outlook,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 4, pp. 2245–2264, 2025.

[70] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[71] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked ´ autoencoders are scalable vision learners,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 16 000–16 009.

[72] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Int. Conf. Comput. Vis., 2023, pp. 4015–4026.

[73] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, “Deepspeed-moe: Advancing mixtureof-experts inference and training to power next-generation ai scale,” in Int. Conf. Mach. Learn. PMLR, 2022, pp. 18 332–18 346.

[74] B. Mustafa, C. Riquelme, J. Puigcerver, R. Jenatton, and N. Houlsby, “Multimodal contrastive learning with limoe: the language-image mixture of experts,” Adv. Neural Inform. Process. Syst., vol. 35, pp. 9564–9576, 2022.

[75] J. Zhu, X. Zhu, W. Wang, X. Wang, H. Li, X. Wang, and J. Dai, “Uniperceiver-moe: Learning sparse generalist models with conditional moes,” Adv. Neural Inform. Process. Syst., vol. 35, pp. 2664–2678, 2022.

[76] L. Shen, G. Chen, R. Shao, W. Guan, and L. Nie, “Mome: Mixture of multimodal experts for generalist multimodal large language models,” Adv. Neural Inform. Process. Syst., vol. 37, pp. 42 048–42 070, 2024.

[77] Z. Zong, B. Ma, D. Shen, G. Song, H. Shao, D. Jiang, H. Li, and Y. Liu, “Mova: Adapting mixture of vision experts to multimodal context,” Adv. Neural Inform. Process. Syst., vol. 37, pp. 103 305– 103 333, 2024.

[78] Y. Li, S. Jiang, B. Hu, L. Wang, W. Zhong, W. Luo, L. Ma, and M. Zhang, “Uni-moe: Scaling unified multimodal llms with mixture of experts,” IEEE Trans. Pattern Anal. Mach. Intell., 2025.

[79] J. Yu, Y. Zhuge, L. Zhang, P. Hu, D. Wang, H. Lu, and Y. He, “Boosting continual learning of vision-language models via mixtureof-experts adapters,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 23 219–23 230.

[80] J. Yu, Z. Huang, Y. Zhuge, L. Zhang, P. Hu, D. Wang, H. Lu, and Y. He, “Moe-adapters++: Toward more efficient continual learning of vision-language models via dynamic mixture-of-experts adapters,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 12, pp. 11 912–11 928, 2025.

[81] K.-K. Maninis, I. Radosavovic, and I. Kokkinos, “Attentive singletasking of multiple tasks,” in IEEE Conf. Comput. Vis. Pattern Recog., 2019, pp. 1851–1860.

[82] Y. Lu, S. Cao, and Y.-X. Wang, “Swiss army knife: Synergizing biases in knowledge from vision foundation models for multi-task learning,” in Int. Conf. Learn. Represent., 2025.

[83] X. Chen, R. Mottaghi, X. Liu, S. Fidler, R. Urtasun, and A. Yuille, “Detect what you can: Detecting and representing objects using holistic models and body parts,” in IEEE Conf. Comput. Vis. Pattern Recog., 2014, pp. 1971–1978.

[84] D. Hendrycks and T. Dietterich, “Benchmarking neural network robustness to common corruptions and perturbations,” in Int. Conf. Learn. Represent., 2019.