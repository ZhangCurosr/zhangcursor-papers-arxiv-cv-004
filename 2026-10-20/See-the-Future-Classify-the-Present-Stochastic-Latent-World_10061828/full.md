# See the Future, Classify the Present: Stochastic Latent World Modeling for Hyperspectral Images

Muhammad Ahmad, Gemine Vivone, Ronny Hansch, Ahmed Sohaib, Manuel Mazzara, Danfeng Hong ¨

Abstract—Hyperspectral image classification (HSIC) relies on static spatial-spectral representations learned from labeled samples, limiting its ability to model structured dependencies in the latent representation space. To address this limitation, this work proposes Stochastic Latent World Modeling (LWM), a trajectorybased representation-learning framework that enriches classification through stochastic latent-state prediction. Given a hyperspectral patch, LWM tokenizes and encodes spectral and spatial information separately using dedicated transformer branches, then fuses their outputs into a unified representation. This representation is further organized into an environment state that captures center, neighborhood, spectral, and global contextual information. A stochastic latent transition module then predicts multiple future latent states through Gaussian parameterization and reparameterization, with recurrent refinement used to model dependencies across successive states. The resulting latent trajectory is adaptively aggregated by a rollout planner to construct a planned representation, which is integrated with the base representation for final classification. By extending static spatialspectral encoding with stochastic multi-step latent-state modeling, LWM enables the classifier to exploit structured dependencies in the representation space and provides richer features for HSIC. Across five benchmark datasets, LWM achieves overall accuracies ranging from 89.21% to 99.52%, demonstrating consistent classification performance under limited-label settings.

Index Terms—Hyperspectral Image Classification; Spatial-Spectral Transformer; World Model; Latent Rollout; Stochastic Representation Learning; Trajectory-Aware Classification.

## I. INTRODUCTION

YPERSPECTRAL image (HSI) classification (HSIC) aims to assign semantic labels to pixels by leveraging high-dimensional spectral signatures together with spatial context [1]. The rich spectral resolution of hyperspectral imagery enables fine-grained material discrimination [2]; however, it also introduces significant challenges, including high dimensionality, strong inter-band correlation, and severe scarcity of labeled samples [3]. Despite substantial advances, most existing HSIC approaches remain fundamentally based on static spatial-spectral representations [4], where classification is performed from a single-stage encoding of local patches

[5]. This paradigm limits the ability of models to capture higher-order dependencies and structured relationships beyond immediate context, particularly in low-label regimes [6].

Limitations of Existing Approaches. Traditional HSIC methods, including hand-crafted feature pipelines and early deep models, represent each patch as a fixed embedding that is directly mapped to a class label [7]. Convolutional neural networks (CNNs) and spatial-spectral hybrid models improve representation capacity through hierarchical feature extraction and local context aggregation [8], [9]. However, these models remain inherently feed-forward and static: the learned representation is a single point estimate in latent space [10]. Such formulations do not explicitly encode relationships between latent states, nor do they model how representations could evolve under structured transformations [11]. Consequently, they fail to exploit the continuity and smoothness inherent in hyperspectral data [12], [13].

Attention-Based Context Modeling Without Structure. Transformer-based models have been introduced to capture long-range dependencies across spectral bands and spatial locations [14]–[16]. While multi-head self-attention (MHSA) improves global context aggregation [17], it operates as a weighted averaging mechanism over tokens within a single forward pass [18]. The resulting representation is still a onestep aggregation without explicit constraints on latent organization [19]–[21]. In particular, attention does not impose transition structure or predictive consistency across representations, which limits its ability to enforce physically meaningful relationships such as spectral continuity and spatial coherence [22]–[24].

Self-Supervised Learning Without Latent Transitions. Self-supervised approaches, including masked reconstruction [7], [25], contrastive learning [26], [27], and generative modeling [28], [29], aim to reduce dependence on labeled data by introducing auxiliary objectives [30]. These methods encourage representations to preserve information or maintain invariances, but they largely operate at the level of reconstruction and similarity [31]. They do not explicitly model dependencies between latent states, nor do they enforce how representations should transform under structured variations. As a result, they provide regularization but not structured predictive modeling [32].

World Models and Their Limitations. Recent advances in world models and latent dynamics learning [33], particularly in reinforcement learning and sequential modeling, aim to learn compact representations that evolve through predictive transitions [34]–[36]. These methods typically rely on explicit temporal observations and learn state transition dynamics conditioned on actions or sequential inputs [37], [38]. While effective in domains with well-defined temporal structure, such formulations are not directly applicable to hyperspectral imagery, which is inherently static and lacks explicit temporal supervision [39].

Moreover, existing world models assume that transitions correspond to physically meaningful state changes observed in time [40]–[44], whereas hyperspectral data primarily exhibit structural correlations rather than temporal evolution. Applying standard world modeling frameworks to HSIC would therefore require artificial sequence construction or surrogate temporal assumptions, which may introduce bias and instability. In addition, world models focus on generative prediction in observation space, rather than enforcing structured organization within latent representations for discriminative tasks [45]–[47]. Consequently, they do not directly address the need for latent predictive regularization in static spatial-spectral data [48], [49].

Absence of Latent Predictive Structure. A key limitation across these categories is the lack of latent predictive modeling tailored to HSIC. Hyperspectral data exhibit strong intrinsic structure, including smooth spectral variation across bands and spatial consistency across neighboring pixels. However, current models do not capture these properties as transitions in latent space. Without modeling such transitions, representations remain locally optimized but globally unstructured. This restricts the ability to generalize under limited supervision, as the model cannot reason beyond the immediate encoded observation.

1) Proposed Perspective: To overcome these limitations, this work introduces stochastic latent world modeling (LWM), a trajectory-aware representation learning framework for HSIC. The central idea is to move beyond static encoding and instead learn representations that are structured through latent evolution. Rather than treating the representation of a patch as a fixed embedding, LWM models how this representation can transform under a learned stochastic process in latent space.

Unlike classical world models that rely on explicit temporal observations, the proposed formulation operates entirely in representation space. A latent environment state is first constructed from fused spatial-spectral features. A stochastic transition model then generates a sequence of latent states via multi-step rollouts, parameterized as Gaussian transitions with recurrent refinement. These rollouts represent plausible variations of the underlying representation, acting as a form of structured regularization rather than modeling physical time.

To extract decision-relevant information, a rollout planning mechanism adaptively aggregates latent states using learned importance weights. This produces a trajectory-aware representation that complements the base transformer encoding. The final prediction is obtained by combining the static and trajectory-based representations, enabling the model to leverage both immediate evidence and structured latent evolution.

The framework is trained with complementary objectives that jointly enforce reconstruction fidelity, latent dynamics consistency, stochastic regularization, and physically meaningful constraints. In particular, spectral and spatial dynamics losses explicitly model structured relationships within latent space, while stochastic rollout regularization enforces probabilistic coherence across predicted transitions. This unified formulation enables the model to learn representations that are not only discriminative but also structured and robust. The main contributions of this work are summarized as follows:

• Trajectory-Aware Representation Learning for HSIC: We reformulate HSIC as a trajectory-aware latent representation problem, moving beyond static spatial-spectral encoding toward structured latent reasoning.

• Stochastic Latent World Modeling: We propose a stochastic latent transition framework that generates multi-step rollouts in representation space using Gaussian parameterization and recurrent refinement, enabling predictive structure without requiring temporal supervision.

• Latent Rollout Planning Mechanism: We introduce an adaptive rollout aggregation strategy that learns to weight latent states, producing a trajectory-aware representation for classification.

• Structured Multi-Objective Learning: We design a unified training objective combining masked reconstruction, spectral and spatial latent dynamics, stochastic rollout regularization, and physically motivated constraints, ensuring both discriminative power and structural consistency.

The remainder of this paper is organized as follows. Section II introduces the proposed Stochastic Latent World Modeling (LWM) framework and its learning objectives. Section III details the experimental setup. Sections IV-V analyze latent rollout reasoning and robustness, while Section VI examines model interpretability. Section VII presents ablation studies, and Section VIII compares LWM with state-of-the-art methods. Section IX discusses limitations and failure cases. Finally, Section X concludes the paper and outlines future directions.

## II. MATHEMATICAL MODELING

The proposed LWM model formulates HSIC as a trajectoryaware spatial-spectral representation learning problem. Each input patch is decomposed into spectral and spatial token sequences and encoded through dedicated Transformer branches, whose outputs are fused into a joint representation. From this representation, a compact environment state is constructed and propagated through a stochastic LWM module to generate multi-step rollouts in representation space. Importantly, the term “world modeling” does not imply temporal simulation: the latent state represents the observed spatial-spectral context, while the learned stochastic transition operator maps this state to plausible future latent configurations under the structural dynamics captured by the model. Accordingly, a “state transition” denotes a learned transformation in representation space, and a “rollout” denotes repeated application of this transition to generate a sequence of latent hypotheses. These latent trajectories do not correspond to explicit temporal observations or imply that the hyperspectral scene evolves over time.

For static HSIC, the purpose of such latent transitions is to provide structurally grounded alternatives to unavailable temporal supervision. Specifically, spectral adjacency, spatial neighborhood relations, and their fused context define the relational structure of the latent transition space. The resulting rollout therefore explores plausible representations consistent with the observed patch structure, rather than simulating physical or temporal evolution. A rollout planner aggregates the latent trajectory into a planned state, which is combined with the current state to produce trajectory-aware class predictions. In this way, the trajectory serves both as a predictive regularizer and as a decision mechanism, allowing the classifier to aggregate multiple structurally plausible latent alternatives rather than relying exclusively on a single static embedding. The overall procedure is summarized in Algorithm 1 and Fig. 1.

![](images/0fcb506b7542bdb3523bd4d3c4041f1635f9353b825a41d4a31621840db5df24.jpg)  
Fig. 1: Overview of LWM framework. (A) Input hyperspectral patches are preprocessed with spectral band and spatial pixel masking. (B) Each patch is decomposed into spectral tokens (per-band) and spatial tokens (per-pixel), which are encoded by dedicated transformer branches and fused through a joint transformer to produce a unified representation and a global scene embedding (CLS token) for base classification. (C) An environment state is constructed by aggregating center, neighborhood, spectral, and global features, and is propagated through a stochastic latent transition model to generate multi-step latent rollouts via Gaussian parameterization and recurrent refinement. A rollout planner adaptively aggregates latent states into a planned representation for trajectory-based classification, which is combined with the base prediction. (D) The model is trained using masked reconstruction, spectral and spatial latent dynamics, stochastic rollout regularization with KL constraint, trajectory supervision, and physically motivated regularization. (E) During inference, masking is disabled and prediction is obtained through a forward pass with latent rollout and trajectory aggregation.

1) Problem Formulation and Notation: HSIC assigns a semantic label to each spatial location using spectral signatures and local spatial context. Let the HSI be $\dot { \mathbf { X } } \in \mathbb { R } ^ { H _ { 0 } \times W _ { 0 } \times C }$ where $H _ { 0 } , W _ { 0 } ,$ , and $C$ denote spatial dimensions and spectral bands. For a labeled pixel at spatial coordinates $\mathbf { q } _ { i } = ( r _ { i } , c _ { i } )$

where $r _ { i }$ and $c _ { i }$ denote its row and column indices, respectively, a local odd patch of size $P \times P$ is extracted using mirror padding, yielding $\mathbf { x } _ { i } ~ \in ~ \mathbb { R } ^ { C \times P \times P }$ . For a mini-batch of B samples, the input is $\mathbf { x } ~ \in ~ \mathbb { R } ^ { B \times C \times P \times P }$ with labels $\mathbf { y } \in \{ 0 , \ldots , K - 1 \} ^ { B }$ , where K is the number of classes.

2) Spectral and Spatial Tokenization: The first step transforms each hyperspectral patch into complementary spectral and spatial token representations [14].

Spectral Tokenization. The spectral tokenizer constructs tokens along the wavelength axis, where each spectral band is treated as an individual token retaining its local spatial context. For the i-th sample, the c-th spectral band $\mathbf { x } _ { i , c , : , : } \in \mathbb { R } ^ { P \times P }$ is flattened into a vector $\mathbf { u } _ { i , c } \in \mathbb { R } ^ { P ^ { 2 } }$ and projected into a Ddimensional embedding space. A learnable spectral positional encoding is added to preserve the ordering of wavelength indices. For a mini-batch, this process is expressed as:

Algorithm 1 Stochastic Latent World Modeling   
Require: HSI mini-batch $\mathbf { x } \in \mathbb { R } ^ { B \times C \times P \times P }$ , labels y, coordi  
nates ${ \bf q } ,$ rollout steps $T$   
Ensure: Final logits ℓ and total loss $\mathcal { L } _ { \mathrm { t o t a l } }$   
1: Construct spectral tokens $\mathbf { S } ^ { 0 } = \phi _ { \mathrm { s p e c } } ( \mathbf { x } ) \in \mathbb { R } ^ { B \times C \times D }$   
2: Construct spatial tokens $\mathbf { A } ^ { 0 } = \phi _ { \mathrm { s p a t } } ( \mathbf { \bar { x } } ) ^ { \prime } \in \mathbb { R } ^ { B \times N \times D }$   
3: Encode sequences $\begin{array} { r l r } { { \bf S } } & { { } = } & { \mathrm { S p e c E n c } _ { \Theta \lambda } ( { \bf S } ^ { 0 } ) } \end{array}$ $\begin{array} { r l } { \mathbf { A } } & { { } = } \end{array}$   
$\mathrm { S p a t E n c } _ { \ominus s } ( \mathbf { \bar { A } } ^ { 0 } )$   
4: Form fused tokens: $\mathbf { F } ^ { 0 }$ (see $\operatorname { E q . 7 } )$   
5: Fuse tokens ${ \bf F } = \mathrm { F u s e } _ { \Theta } ( { \bf F } ^ { 0 } )$   
6: Extract components: $\mathbf { h } _ { \mathrm { c l s } } , \mathbf { Z } _ { \lambda } , \mathbf { Z } _ { s }$   
7: Compute base logits: $\ell _ { \mathrm { b a s e } } = g _ { \mathrm { b a s e } } ( \mathbf { h } _ { \mathrm { c l s } } )$   
8: Construct environment state $\mathbf { z } _ { 0 }$ (see Eq. 11)   
9: Initialize rollout $\mathbf { Z } _ { \mathrm { r o l l } }  [ \mathbf { z } _ { 0 } ]$   
10: for $t = 0$ to $T - 1$ do   
11: Predict transition: $[ \pmb { \mu } _ { t + 1 } , \log \pmb { \sigma } _ { t + 1 } ^ { 2 } ] = G _ { \theta } ( \mathbf { z } _ { t } )$   
12: Clip log-variance: log $\pmb { \sigma } _ { t + 1 } ^ { 2 }  \mathrm { c l i p } ( \log \pmb { \sigma } _ { t + 1 } ^ { 2 } , a , b )$   
13: Latent state: $\widetilde { \mathbf { z } } _ { t + 1 } = \pmb { \mu } _ { t + 1 } + \epsilon _ { t + 1 } \odot \exp \left( \frac { 1 } { 2 } \log \pmb { \sigma } _ { t + 1 } ^ { 2 } \right)$   
14: Refine state: $\mathbf { z } _ { t + 1 } = R _ { \theta } ( \widetilde { \mathbf { z } } _ { t + 1 } , \mathbf { z } _ { t } )$   
15: Define projected target: $\widehat { \mathbf { z } } _ { t + 1 } = \operatorname { s g } ( Q _ { \theta } ( \mathbf { z } _ { t } ) )$   
16: Append $\mathbf { z } _ { t + 1 }$ to rollout   
17: end for   
18: Compute rollout weights: $w _ { t } = \operatorname { s o f t m a x } ( u _ { \theta } ( \mathbf { z } _ { t } ) )$   
19: Aggregate planned state: $\mathbf { z } _ { \mathrm { p l a n } } = \sum _ { t = 0 } ^ { T }$ w<sub>t</sub>z<sub>t</sub>   
20: Compute trajectory logits: $\ell _ { \mathrm { t r a j } }$ (see Eq. 21)   
21: Final prediction: $\ell = { \bf G } \odot \ell _ { \mathrm { b a s e } } + ( { \bf 1 } _ { B \times K } - { \bf G } ) \odot \ell _ { \mathrm { t r a j } }$   
22: Compute $\mathcal { L } _ { \mathrm { t o t a l } }$ (see Eq. 31)   
23: return $\ell , \mathcal { L } _ { \mathrm { t o t a l } }$

$$
\begin{array} { r } { \mathbf { S } ^ { 0 } = \phi _ { \mathrm { s p e c } } ( \mathbf { x } ) = \mathbf { x } _ { \mathrm { s p e c } } \times _ { 3 } \mathbf { W } _ { \mathrm { s p e c } } + \boldsymbol { \widetilde { \mathbf { P } } } _ { \lambda } \in \mathbb { R } ^ { B \times C \times D } , } \end{array}\tag{1}
$$

where $\mathbf { x } _ { \mathrm { s p e c } } \in \mathbb { R } ^ { B \times C \times P ^ { 2 } }$ is the flattened spectral representation, $\mathbf { W } _ { \mathrm { s p e c } } ~ \in ~ \mathbb { R } ^ { P ^ { 2 } \times D }$ is a learnable projection matrix, and $\mathbf { P } _ { \lambda } \in \mathbb { R } ^ { 1 \times C \times D }$ denotes the learnable spectral positional embedding. The batch-expanded positional embedding is defined as $\widetilde { \mathbf { P } } _ { \lambda } = \mathbf { 1 } _ { B } \otimes \mathbf { P } _ { \lambda } \in \overset { \bullet } { \mathbb { R } } ^ { B \times C \times \bar { D } }$ , where $\mathbf { 1 } _ { B }$ is a B-dimensional vector of ones and ⊗ denotes replication along the batch dimension.

Spatial Tokenization. The spatial tokenizer constructs tokens along the spatial domain, where each token corresponds to a spatial location and carries its full spectral signature. The input tensor $\mathbf { x } \in \mathbb { R } ^ { B \times C \times P \times P }$ is rearranged by flattening the two spatial dimensions into a single spatial-token dimension, such that $\mathbf { x } _ { \mathrm { s p a t } } \in \mathbb { R } ^ { B \times N \times C }$ with ${ \bar { N } } = P ^ { 2 }$ . Thus, for the i-th sample and the p-th spatial location, the corresponding spectral vector is $\mathbf { x } _ { \mathrm { s p a t } , i , p , : } \in \mathbb { R } ^ { C }$ , which is projected into the same Ddimensional embedding space. A learnable spatial positional encoding is added to retain spatial layout information. The resulting spatial token sequence is given by

$$
\mathbf { A } ^ { 0 } = \phi _ { \mathrm { s p a t } } ( \mathbf { x } ) = \mathbf { x } _ { \mathrm { s p a t } } \times _ { 3 } \mathbf { W } _ { \mathrm { s p a t } } + \widetilde { \mathbf { P } } x y \in \mathbb { R } ^ { B \times N \times D } ,\tag{2}
$$

where $\mathbf { x } _ { \mathrm { s p a t } } \ \in \ \mathbb { R } ^ { B \times N \times C }$ is the spatially rearranged tensor, $\mathbf { W } _ { \mathrm { s p a t } } \in \mathbf { \dot { \mathbb { R } } } ^ { C \times D }$ is a learnable projection matrix, and $\mathbf { P } _ { x y } \in$ $\mathbb { R } ^ { 1 \times N \times D }$ represents the spatial positional embeddings. The batch-expanded positional embedding is defined as $\widetilde { \mathbf { P } } _ { x y } =$ $\mathbf { 1 } _ { B } \otimes \mathbf { \bar { P } } _ { x y } \in \mathbb { R } ^ { B \times N \times D }$

3) Transformer Encoding: Let $\mathrm { S p e c E n c } _ { \Theta \lambda } ( \cdot )$ denote the spectral transformer encoder composed of $L _ { \lambda }$ transformer layers, and let $\mathrm { S p a t E n c } _ { \Theta s } ( \cdot )$ denote the spatial transformer encoder composed of $L _ { s }$ transformer layers. The corresponding encoded representations are defined as

$$
\mathbf { S } = \mathrm { S p e c E n c } _ { \Theta \lambda } ( \mathbf { S } ^ { 0 } ) \in \mathbb { R } ^ { B \times C \times D } ,\tag{3}
$$

$$
\mathbf { A } = \mathrm { S p a t E n c } _ { \ominus s } ( \mathbf { A } ^ { 0 } ) \in \mathbb { R } ^ { B \times N \times D } ,\tag{4}
$$

where C and $N = P ^ { 2 }$ denote the sequence lengths of the spectral and spatial branches, respectively. The parameters $\Theta _ { \lambda }$ and $\Theta _ { s }$ are learned independently. For a token sequence $\mathbf { T } ^ { \ell - 1 } \in \mathbb { R } ^ { B \times M \times D }$ at layer ℓ − 1, a transformer block updates the representation through pre-normalized self-attention and feed-forward sublayers:

$$
\widetilde { \mathbf { T } } ^ { \ell } = \mathbf { T } ^ { \ell - 1 } + \mathrm { D r o p P a t h } \left( \mathrm { M H S A } \left( \mathrm { L N } ( \mathbf { T } ^ { \ell - 1 } ) \right) \right) ,\tag{5}
$$

$$
\begin{array} { r } { \mathbf { T } ^ { \ell } = \widetilde { \mathbf { T } } ^ { \ell } + \mathrm { D r o p P a t h } \left( \mathrm { M L P } \left( \mathrm { L N } ( \widetilde { \mathbf { T } } ^ { \ell } ) \right) \right) , } \end{array}\tag{6}
$$

where MHSA(·) denotes multi-head self-attention [17], MLP(·) denotes a feed-forward network, LN(·) denotes layer normalization, and DropPath(·) introduces stochastic depth. The sequence length M is set to C for the spectral branch and N for the spatial branch. Applying this transformer block for $L _ { \lambda }$ and $L _ { s }$ layers yields the encoded representations S and A, respectively.

Fusion. To enable joint modeling, the spectral and spatial representations are fused using a transformer encoder (using $L _ { f }$ transformer layers). A learnable classification token $\mathbf { c } _ { 0 } \in \breve { \mathbb { R } } ^ { 1 \times 1 \times D }$ and type embeddings $\mathbf { e } _ { \mathrm { c l s } } , \mathbf { e } _ { \lambda } , \mathbf { e } _ { s } \in \mathbb { R } ^ { 1 \times 1 \times D }$ are introduced to distinguish token types. Since these embeddings are shared across the mini-batch, their batch-expanded forms are defined as $\widetilde { \mathbf { c } } _ { 0 } = \mathbf { 1 } _ { B } \otimes \mathbf { c } _ { 0 } \in \mathbb { R } ^ { B \times 1 \times D } , \widetilde { \mathbf { e } } _ { \mathrm { c l s } } = \mathbf { 1 } _ { B } \otimes \mathbf { e } _ { \mathrm { c l s } } \in$ $\mathbb { R } ^ { B \times 1 \times D } , \ \widetilde { \mathbf { e } } _ { \lambda } \ = \ \mathbf { 1 } _ { B } \ \otimes \mathbf { e } _ { \lambda } \ \in \ \mathbb { R } ^ { B \times 1 \times D }$ , and $\widetilde { \mathbf { e } } _ { s } \ = \ \mathbf { 1 } _ { B } \otimes$ $\mathbf { e } _ { s } \in \mathbb { R } ^ { B \times 1 \times D }$ . Let $\mathrm { B r o a d c a s t } _ { C } ( \cdot )$ and Broadcast $_ N \left( \cdot \right)$ denote replication along the spectral and spatial sequence dimensions, respectively, and let Concat seq(·) denote concatenation along the sequence dimension. The fused input sequence is defined as

$$
\begin{array} { r l } & { \mathbf { F } ^ { 0 } = \operatorname { C o n c a t } _ { \mathrm { s e q } } \left( \widetilde { \mathbf { c } } _ { 0 } + \widetilde { \mathbf { e } } _ { \mathrm { c l s } } , \mathbf { S } + \operatorname { B r o a d c a s t } _ { C } \left( \widetilde { \mathbf { e } } _ { \lambda } \right) , \right. } \\ & { \quad \left. \mathbf { A } + \operatorname { B r o a d c a s t } _ { N } \left( \widetilde { \mathbf { e } } _ { s } \right) \right) \in \mathbb { R } ^ { B \times \left( 1 + C + N \right) \times D } . } \end{array}\tag{7}
$$

The fused sequence is processed by the fusion transformer encoder as $\mathbf { F } = \operatorname { F u s e } _ { \Theta } ( \dot { \mathbf { F } } ^ { 0 } ) \in \mathbb { R } ^ { B \times ( 1 + C + N ) \times D }$ . The output is decomposed as $\mathbf { h } _ { \mathrm { c l s } } = \mathbf { F } _ { : , 0 , : } \in \mathbb { R } ^ { B \times D } , \mathbf { Z } _ { \lambda } = \mathbf { F } _ { : , 1 : 1 + C , : } \in$ $\mathbb { R } ^ { B \times C \times D ^ { \bullet } }$ , and $\mathbf { Z } _ { s } = \mathbf { F } _ { : , 1 + C : , : } \in \mathbb { R } ^ { B \times N \times D }$ . The classification token $\mathbf { h } _ { \mathrm { c l s } }$ provides a global representation for the base prediction. The base classification head $g _ { \mathrm { b a s e } } : \mathbb { R } ^ { D }  \mathbb { R } ^ { K }$ consists of layer normalization, a linear layer, GELU activation, dropout, and a final linear layer mapping to K classes. The base logits are given by

$$
\ell _ { \mathrm { b a s e } } = g _ { \mathrm { b a s e } } ( \mathbf { h } _ { \mathrm { c l s } } ) \in \mathbb { R } ^ { B \times K } ,\tag{8}
$$

and serve as an initial prediction that is refined through latent rollout modeling in subsequent stages.

## A. Environment State Construction

Following spatial-spectral encoding and fusion, a compact latent representation is constructed to capture the local environment of the target pixel. While the classification token $\mathbf { h } _ { \mathrm { c l s } }$ provides a global summary, it does not explicitly separate local spatial structure and spectral composition. To address this, the environment state is formed by aggregating complementary components derived from the fused tokens. Let $\begin{array} { r } { { { n } _ { 0 } } = \frac { { { P } ^ { 2 } } - { \bar { 1 } } } { 2 } } \end{array}$ denote the flattened index of the center location in the spatial token sequence, assuming P is odd, and let $\mathcal { N } _ { 8 } ( n _ { 0 } )$ denote the set of indices corresponding to its eight first-ring neighbors. The center state is defined as $\mathbf { z } _ { c } \ = \ \mathbf { Z } _ { s , : , n _ { 0 } } \ \in \ \mathbb { R } ^ { \Breve { \times } N \times D }$ which encodes the latent representation of the target pixel. The neighbor state is computed as:

$$
\mathbf { z } _ { n } = \frac { 1 } { | \mathcal { N } _ { 8 } ( n _ { 0 } ) | } \sum _ { j \in \mathcal { N } _ { 8 } ( n _ { 0 } ) } \mathbf { Z } _ { s , : , j , : } \in \mathbb { R } ^ { B \times D } ,\tag{9}
$$

capturing local spatial context. The global spectral state is given by:

$$
\mathbf { z } _ { \lambda } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \mathbf { Z } _ { \lambda , : , c , : } \in \mathbb { R } ^ { B \times D } ,\tag{10}
$$

and the scene-level representation is $\mathbf { z } _ { g } ~ = ~ \mathbf { h } _ { \mathrm { c l s } } ~ \in ~ \mathbb { R } ^ { B \times D }$ These components are concatenated along the column dimension using the operator $\mathrm { { C o n c a t } _ { \mathrm { { c o l } } } ( \cdot ) }$ , yielding $\begin{array} { r l } { \mathbf { z } _ { \mathrm { c a t } } } & { { } = } \end{array}$ $\mathrm { C o n c a t } _ { \mathrm { c o l } } \left( { \bf z } _ { c } , { \bf z } _ { n } , { \bf z } _ { \lambda } , { \bf \bar { z } } _ { g } \right) \in \mathbb { R } ^ { B \times 4 D }$ . The concatenated representation is then mapped into a unified latent representation:

$$
\mathbf { z } _ { \mathrm { 0 } } = E _ { \mathrm { e n v } } ( \mathbf { z } _ { \mathrm { c a t } } ) \in \mathbb { R } ^ { B \times D } ,\tag{11}
$$

where $E _ { \mathrm { e n v } } \colon \mathbb { R } ^ { B \times 4 D } \  \ \mathbb { R } ^ { B \times D }$ is implemented using layer normalization, linear projection, Gaussian error linear unit (GELU) activation, dropout, and a final normalization layer. The resulting $\mathbf { z } _ { 0 }$ serves as the initial latent environment state, integrating local spatial, spectral, and global contextual information for subsequent latent rollout modeling.

## B. Stochastic Latent World Modeling

Given the environment state $\mathbf { z } _ { 0 } .$ , the model simulates its evolution in latent space through a stochastic rollout mechanism [42], [43]. This process captures variations of the local spatial-spectral representation by modeling uncertainty in latent transitions [50]. The stochastic LWM module defines a multi-step rollout starting from ${ \bf z } _ { 0 } \ [ 4 4 ]$ . For $t = \{ 0 , \ldots , T { - } 1 \}$ , a transition network $G _ { \theta }$ predicts the parameters of a Gaussian latent transition:

$$
[ \pmb { \mu } _ { t + 1 } , \log \pmb { \sigma } _ { t + 1 } ^ { 2 } ] = G _ { \theta } ( \mathbf { z } _ { t } ) \in \mathbb { R } ^ { B \times 2 D } ,\tag{12}
$$

where $G _ { \theta } : \mathbb { R } ^ { B \times D }  \mathbb { R } ^ { B \times 2 D }$ is implemented as a feedforward network consisting of layer normalization, a linear layer mapping D to 2D, GELU activation, dropout, and a final linear layer mapping 2D to 2D. The output is split along the feature dimension into the predicted mean $\boldsymbol { \mu } _ { t + 1 } \in \mathbb { R } ^ { B \times D }$ and log-variance log $\pmb { \sigma } _ { t + 1 } ^ { 2 } \in \mathring { \mathbb { R } } ^ { B \times D }$ . For numerical stability, the predicted log-variance is clipped element-wise to a fixed interval [a, b], where $a \textless b$ are predefined lower and upper bounds, respectively:

$$
\log \pmb { \sigma } _ { t + 1 } ^ { 2 }  \mathrm { c l i p } ( \log \pmb { \sigma } _ { t + 1 } ^ { 2 } , a , b ) ,\tag{13}
$$

where, for each element v of log $\sigma _ { t + 1 } ^ { 2 }$ is $\mathrm { c l i p } ( v , a , b ) \ =$ min $\{ \operatorname* { m a x } \{ v , a \} , b \}$ . Thus, all predicted log-variance values are constrained to the interval $[ a , b ]$ . During training, the next latent state is sampled using the reparameterization trick [39], [51]:

$$
\widetilde { \mathbf { z } } _ { t + 1 } = \pmb { \mu _ { t + 1 } } + \epsilon _ { t + 1 } \odot \exp \left( \frac { 1 } { 2 } \log \pmb { \sigma } _ { t + 1 } ^ { 2 } \right) ,\tag{14}
$$

where $\begin{array} { r l r } { \mathbf { \epsilon } _ { \epsilon + 1 } } & { { } \sim } & { \mathcal { N } _ { B \times D } \left( \mathbf { 0 } _ { B \times D } , \mathbf { I } _ { B } \otimes \mathbf { I } _ { D } \right) } \end{array}$ , so that $\epsilon _ { t + 1 } \in$ $\mathbb { R } ^ { B \times D }$ and ⊙ denotes element-wise Hadamard multiplication. During inference, stochastic sampling is disabled and $\widetilde { \mathbf { z } } _ { t + 1 } =$ $\pmb { \mu } _ { t + 1 }$ . The state is then refined via the recurrent update

$$
\mathbf { z } _ { t + 1 } = R _ { \theta } \left( \widetilde { \mathbf { z } } _ { t + 1 } , \mathbf { z } _ { t } \right) \in \mathbb { R } ^ { B \times D } ,\tag{15}
$$

where $R _ { \theta }$ is implemented as a gated recurrent unit (GRU) cell [45]. This yields a latent trajectory. Since each latent state, predicted mean, and log-variance is represented as a matrix in $\mathbb { R } ^ { B \times D }$ , we define $\mathrm { { C o n c a t } _ { 3 } ( \cdot ) }$ as concatenation along a new third dimension. For $T$ matrices $\{ \mathbf { X } _ { 1 } , \ldots , \mathbf { X } _ { T } \} \ \in \ \mathbb { R } ^ { B \times D }$ as Concat $( \mathbf { X } _ { 1 } , \dots , \mathbf { X } _ { T } ) \ \in$ $\mathbb { R } ^ { B \times \grave { D } \times T }$ , and reshape $B \times T \times D ^ { ( \cdot ) }$ denotes the corresponding reshaping of this tensor into the batch-trajectoryfeature ordering. The latent trajectory is therefore defined as $\begin{array} { r l } & { \mathbf { Z } _ { \mathrm { r o l l ~ } } = \mathrm { ~ \ r e s h a p e } _ { B \times ( T + 1 ) \times D } \left( \mathrm { C o n c a t } _ { 3 } \left( \mathbf { z } _ { 0 } , \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { T } \right) \right) } \\ & { \mathrm { e r e ~ } \mathbf { Z } _ { \mathrm { r o l l } } \in \mathbb { R } ^ { B \times ( T + 1 ) \times D } . } \end{array}$ wh

Similarly, the predicted means and log-variances are collected as µ reshap ${ \bf \psi } _ { B \times T \times D } ^ { 3 } \left( \mathrm { C o n c a t } _ { 3 } \left( \pmb { \mu } _ { 1 } , \ldots , \pmb { \mu } _ { T } \right) \right) \qquad \in \qquad \mathbb { R } ^ { B \times T \times D } ,$ and ${ \bf V } =$ reshape<sub>B×T×D</sub>  Concat<sub>3</sub> $( \log \pmb { \sigma } _ { 1 } ^ { 2 } , \dots , \log \pmb { \sigma } _ { T } ^ { 2 } ) ) \in$ $\mathbb { R } ^ { B \times T \times D }$

To construct an auxiliary latent target, let $Q _ { \theta } : \mathbb { R } ^ { B \times D } $ $\cdot ^ { B \times D }$ denote a learnable latent projection network that maps the current latent state to a target representation. The stopgradient operator $\operatorname { s g } ( \cdot )$ returns its input while preventing gradients from propagating through that input during backpropagation, i.e., $\begin{array} { r } { \mathrm { , ~ } \mathrm { s g } ( \mathbf { u } ) = \mathbf { \bar { u } } , \frac { \partial \mathrm { s g } ( \bar { \mathbf { u } } ) } { \partial \mathbf { u } } = \mathbf { 0 } } \end{array}$ . The auxiliary latent target is then defined as:

$$
\widehat { \mathbf { z } } _ { t + 1 } = \operatorname { s g } \left( Q _ { \theta } ( \mathbf { z } _ { t } ) \right) \in \mathbb { R } ^ { B \times D } ,\tag{16}
$$

and the corresponding sequence is constructed using the same third-dimension concatenation and reshaping operation $\widehat { \textbf { Z } } = \mathrm { \ r e s h a p e { } } _ { B \times T \times D } \left( \mathrm { C o n c a t { } } _ { 3 } \left( \widehat { \mathbf { z } } _ { 1 } , \ldots , \widehat { \mathbf { z } } _ { T } \right) \right) \overset { = } \ \mathbb { R } ^ { \widehat { B } \times T \times D }$ This target acts as an internal projection used to regularize the stochastic transitions. The rollout operates entirely in representation space and does not correspond to explicit temporal observations. The resulting latent trajectory provides a set of states that are subsequently used for trajectory-aware aggregation and classification.

## C. Trajectory Planning and Classification

Given the latent trajectory $\mathbf { z } _ { t } ^ { \textit { T } } _ { \{ t = 0 \} }$ , the model performs trajectory-aware aggregation to obtain a decision-relevant representation. For each rollout state $\mathbf { z } _ { t } ~ \in ~ \mathbb { R } ^ { B \times D }$ , a learnable scoring network $u _ { \theta } : \mathbb { R } ^ { D }  \mathbb { R }$ assigns a scalar score independently to each sample. Specifically, u<sub>θ</sub> is implemented as a two-layer MLP with layer normalization, a linear projection from D to $D / 2$ , GELU activation, and a linear projection to a scalar:

$$
u _ { \boldsymbol \theta } ( { \mathbf { z } } ) = \mathbf { w } _ { u } ^ { \top } \mathrm { G E L U } \left( \mathbf { W } _ { u } \mathrm { L N } ( { \mathbf { z } } ) + { \mathbf { b } } _ { u } \right) + b _ { u } ,\tag{17}
$$

where $\mathbf { W } _ { u } \in \mathbb { R } ^ { \frac { D } { 2 } \times D } , \ : \mathbf { b } _ { u } \in \mathbb { R } ^ { D / 2 } , \ : \mathbf { w } _ { u } \in \mathbb { R } ^ { D / 2 }$ , and $b _ { u } \in$ R are learnable parameters. Thus, for sample i and rollout step $t ,$ the corresponding score is $\alpha _ { i , t } = u _ { \theta } ( \mathbf { z } _ { i , t } ) \in \mathbb { R } , \ i =$ $\{ 1 , . . . , B \} , \ t = \{ 0 , . . . , T \}$ , where $\mathbf { z } _ { i , t } \in \mathbb { R } ^ { D }$ denotes the i-th row of $\mathbf { z } _ { t }$ . The scores are normalized independently for each sample along the trajectory dimension using the softmax operator:

$$
w _ { i , t } = \mathrm { s o f t m a x } ( \pmb { \alpha } _ { i } ) _ { t } = \frac { \exp ( \alpha _ { i , t } ) } { \sum _ { \tau = 0 } ^ { T } \exp ( \alpha _ { i , \tau } ) } ,\tag{18}
$$

where $\textstyle \sum _ { t = 0 } ^ { T } w _ { i , t } = 1 , \ w _ { i , t } \geq 0$ and $\pmb { \alpha } _ { i } = [ \alpha _ { i , 0 } , \ldots , \alpha _ { i , T } ] \in$ $\mathbb { R } ^ { T + 1 }$ . Equivalently, softmax(α<sub>i</sub>) converts the scalar scores associated with the $T + 1$ rollout states of sample i into normalized trajectory weights. The planned representation is then obtained by a weighted aggregation of the latent states $\begin{array} { r } { \mathbf { z } _ { \mathrm { p l a n } , i } = \sum _ { t = 0 } ^ { T } \dot { w } _ { i , t } \mathbf { z } _ { i , t } \in \mathbb { R } ^ { D } } \end{array}$ , for the complete mini-batch,

$$
\mathbf { z } _ { \mathrm { p l a n } } = \sum _ { t = 0 } ^ { T } \mathbf { w } _ { t } \odot \mathbf { z } _ { t } \in \mathbb { R } ^ { B \times D } ,\tag{19}
$$

where $\mathbf { w } _ { t } \ = \ [ w _ { 1 , t } , \ldots , w _ { B , t } ] ^ { \top } \ \in \ \mathbb { R } ^ { B }$ and the weights are broadcast across the feature dimension. The trajectory representation is constructed by concatenating the initial latent state and the planned state along the feature dimension:

$$
\mathbf { r } _ { \mathrm { t r a j } } = { \mathrm { C o n c a t } } _ { \mathrm { c o l } } \left( \mathbf { z } _ { 0 } , \mathbf { z } _ { \mathrm { p l a n } } \right) \in \mathbb { R } ^ { B \times 2 D } .\tag{20}
$$

The trajectory-based logits are produced by a trajectory classifier $\overline { { g _ { \mathrm { t r a j } } } } \ \mathrm { ~ : ~ } \mathbb { R } ^ { 2 D } \  \ \overline { { \mathbb { R } } } ^ { K }$ which consists of layer normalization, a linear projection from 2D to $D ,$ GELU activation, dropout, and a final linear projection from D to K as $g _ { \mathrm { t r a j } } ( \mathbf { r } ) = \mathbf { W } _ { \mathrm { t r a j , 2 } } { \mathrm { G E L U } } ( \mathbf { W } _ { \mathrm { t r a j , 1 } } \mathrm { L N } ( \mathbf { r } ) + \mathbf { b } _ { \mathrm { t r a j , 1 } } ) + \mathbf { b } _ { \mathrm { t r a j , 2 } } $ where $\mathbf { W } _ { \mathrm { t r a j } , 1 } \ \in \mathbb { R } ^ { D \times 2 D } , \ \mathbf { b } _ { \mathrm { t r a j } , 1 } \ \in \ \mathbb { R } ^ { D } , \ \mathbf { W } _ { \mathrm { t r a j } , 2 } \ \in \ \mathbb { R } ^ { K \times D }$ and $\mathbf { b } _ { \mathrm { t r a j } , 2 } \ \in \ \mathbb { R } ^ { K }$ are learnable parameters, and dropout is applied between the GELU activation and the linear projection. Thus,

$$
\ell _ { \mathrm { t r a j } } = g _ { \mathrm { t r a j } } \left( \mathbf { r } _ { \mathrm { t r a j } } \right) \in \mathbb { R } ^ { B \times K } .\tag{21}
$$

The final prediction combines the base and trajectory logits using a sample-dependent branch gate. The gate is computed from the concatenation of the fused CLS representation and the planned latent state as:

$$
\begin{array} { r } { g _ { i } = \sigma \Big ( \mathbf { w } _ { g } ^ { \top } \mathrm { G E L U } \left( \mathbf { W } _ { g } \mathrm { L N } \left( \mathrm { C o n c a t } _ { \mathrm { c o l } } \left( \mathbf { h } _ { \mathrm { c l s } , i } , \mathrm { s g } ( \mathbf { z } _ { \mathrm { p l a n } , i } ) \right) \right) \right) } \\ { + \mathbf { b } _ { g } \Big ) + b _ { g } \Big ) , } \end{array}
$$

where $\mathbf { h } _ { \mathrm { c l s } , i } \in \mathbb { R } ^ { D }$ and $\mathbf { z } _ { \mathrm { p l a n } , i } \in \mathbb { R } ^ { D }$ denote the i-th samples of the CLS and planned representations, respectively, ${ \bf W } _ { g } \in \mathbf { \Sigma }$ $\mathbb { R } ^ { D \times 2 D } , \ \mathbf { b } _ { g } \ \in \ \dot { \mathbb { R } } ^ { D } , \ \mathbf { w } _ { g } \ \in \ \mathbb { R } ^ { D }$ , and $b _ { g } \in \mathbb { R }$ are learnable parameters, and $\sigma ( \cdot )$ denotes the sigmoid function. The stopgradient operation sg(·) prevents gradients from the branchgating network from propagating through the planned state. Collecting the sample-wise gates gives $\mathbf { g } = [ g _ { 1 } , \ldots , g _ { B } ] ^ { \top } \in$ $( 0 , 1 ) ^ { B \times \mathrm { i } }$ . To apply the scalar gate to all K class logits of each sample, define $\mathbf { \bar { G } } = \mathbf { g } \mathbf { \partial } \mathbf { 1 } _ { K } ^ { \top } \in ( \bar { 0 } , 1 ) ^ { B \times K }$ , where $\mathbf { 1 } _ { K } \in \mathbb { R } ^ { K }$ is a vector of ones. The final logits are then obtained by elementwise interpolation:

$$
\ell = \mathbf G \odot \ell _ { \mathrm { b a s e } } + \left( \mathbf { 1 } _ { B \times K } - \mathbf { G } \right) \odot \ell _ { \mathrm { t r a j } } \in \mathbb { R } ^ { B \times K } ,\tag{22}
$$

where ⊙ denotes element-wise multiplication and ${ \bf 1 } _ { B \times K }$ is a $B \times K$ matrix of ones. Thus, each sample uses a single learned gate to control the contribution of the base and trajectory branches across all classes. For the predicted label, let $\boldsymbol { \ell } _ { i } ^ { \check { \mathbf { \ell } } } \in \mathbb { R } ^ { \check { \cal K } }$ denote the i-th row of the final logit matrix $\boldsymbol { \ell } \in \mathbb { R } ^ { B \times K }$ . The predicted class is given by

$$
\widehat { y } _ { i } = \underset { k \in \{ 0 , \ldots , K - 1 \} } { \arg \operatorname* { m a x } } \ \mathrm { s o f t m a x } \left( \ell _ { i } \right) _ { k } .\tag{23}
$$

This trajectory planning mechanism enables the model to aggregate information across the latent rollout while retaining the initial environment state for trajectory-based classification. By learning sample-dependent importance weights over the rollout states, the model incorporates structured information from the latent trajectory into the final prediction.

## D. Reconstruction and Dynamics Constraints

The model incorporates reconstruction and latent dynamics constraints to regularize the spectral and spatial token representations. These constraints operate on the fused spectral and spatial representations $\mathbf { Z } _ { \lambda }$ and $\mathbf { Z } _ { s }$ obtained from the fusion transformer. In particular, the reconstruction heads recover the corresponding input-domain representations, while the dynamics modules impose predictive structure along the spectral sequence and across the local spatial neighborhood. Spectral Reconstruction. The spectral reconstruction head $R _ { \lambda }$ maps each D-dimensional spectral token back to the flat tened spatial patch associated with the corresponding spectral band. Given $\dot { \mathbf { Z } } _ { \lambda } \in \mathbb { R } ^ { B \times C \times D }$ , the same reconstruction mapping is applied independently to all $C$ spectral tokens. Specifically, $R _ { \lambda } : \mathbb { R } ^ { D }  \mathbb { R } ^ { \mathbf { \bar { N } } }$ is implemented using layer normalization, a linear projection from D to D, GELU activation, and a final linear projection from $D$ to $N \colon$

$$
R _ { \lambda } ( \mathbf { z } ) = \mathbf { W } _ { \lambda , 2 } ^ { \mathrm { r e c } } \operatorname { G E L U } \left( \mathbf { W } _ { \lambda , 1 } ^ { \mathrm { r e c } } \operatorname { L N } ( \mathbf { z } ) + \mathbf { b } _ { \lambda , 1 } ^ { \mathrm { r e c } } \right) + \mathbf { b } _ { \lambda , 2 } ^ { \mathrm { r e c } } ,\tag{24}
$$

where $\mathbf { W } _ { \lambda , 1 } ^ { \mathrm { r e c } } ~ \in ~ \mathbb { R } ^ { D \times D } , ~ \mathbf { b } _ { \lambda , 1 } ^ { \mathrm { r e c } } ~ \in ~ \mathbb { R } ^ { D }$ ${ \bf W } _ { \lambda , 2 } ^ { \mathrm { r e c } } ~ \in ~ \mathbb { R } ^ { N \times D }$ and $\mathbf { b } _ { \lambda , 2 } ^ { \mathrm { r e c } } \in \mathbb { R } ^ { N }$ are learnable parameters. Applying $R _ { \lambda }$ to every spectral token gives $\widehat { \mathbf { x } } _ { \lambda } = R _ { \lambda } ( \mathbf { Z } _ { \lambda } ) \in \mathbb { R } ^ { B \times C \times N }$ . The corresponding target is obtained by flattening the two spatial dimensions of the input patch $\mathbf { x } _ { \lambda } = \mathrm { r e s h a p e } ( \mathbf { x } ) \in \mathbb { R } ^ { B \times \bar { C } \times N }$ Thus, the c-th output token of $R _ { \lambda }$ reconstructs the $P \times P$ spatial values belonging to the c-th spectral band.

Spatial Reconstruction. The spatial reconstruction head $R _ { s }$ maps each spatial token back to its corresponding spectral vector. Given $\dot { \mathbf { Z } } _ { s } \in \mathbb { R } ^ { B \times N \times D }$ , the same reconstruction mapping is applied independently to all N spatial tokens. The mapping $R _ { s } : \mathbb { R } ^ { D }  \bar { \mathbb { R } } ^ { C }$ is implemented using layer normalization, a linear projection, GELU activation, and a linear projection from D to C:

$$
R _ { s } ( \mathbf { z } ) = \mathbf { W } _ { s , 2 } ^ { \mathrm { r e c } } \operatorname { G E L U } \left( \mathbf { W } _ { s , 1 } ^ { \mathrm { r e c } } \operatorname { L N } ( \mathbf { z } ) + \mathbf { b } _ { s , 1 } ^ { \mathrm { r e c } } \right) + \mathbf { b } _ { s , 2 } ^ { \mathrm { r e c } } ,\tag{25}
$$

where ${ \bf W } _ { s , 1 } ^ { \mathrm { r e c } } ~ \in ~ \mathbb { R } ^ { D \times D }$ $\mathbf { b } _ { s , 1 } ^ { \mathrm { r e c } } ~ \in ~ \mathbb { R } ^ { D }$ ${ \bf W } _ { s , 2 } ^ { \mathrm { r e c } } ~ \in ~ \mathbb { R } ^ { C \times D }$ and $\mathbf { b } _ { s , 2 } ^ { \mathrm { r e c } } ~ \in ~ \mathbb { R } ^ { C }$ are learnable parameters. Applying $R _ { s }$ to all spatial tokens yields $\widehat { \mathbf { x } } _ { s } ~ = ~ \bar { \mathbf { \xi } } R _ { s } ( \mathbf { Z } _ { s } ) ~ \in ~ \bar { \mathbb { R } } ^ { \bar { B } \times N \times C }$ . The corresponding target is obtained by first permuting the input tensor so that the spatial dimensions precede the spectral dimension and then flattening the spatial dimensions $\mathbf { x } _ { s } ~ =$ reshape $( \mathrm { p e r m u t e } ( \mathbf { x } ) ) \in \mathbb { R } ^ { B \times N \times C }$ . Hence, each spatial token is trained to reconstruct the complete C-dimensional spectral signature of its corresponding spatial location. When masking is applied, the reconstruction losses are evaluated only on the masked spectral bands or spatial locations; otherwise, all entries are used. Let $\rho _ { \beta } ( \cdot , \cdot )$ denote the element-wise Smooth-$\ell _ { 1 }$ penalty with transition parameter $\beta > 0 !$

$$
\rho _ { \beta } ( a , b ) = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { 2 } ( a - b ) ^ { 2 } / \beta , } & { \displaystyle | a - b | < \beta , } \\ { \displaystyle | a - b | - \frac { 1 } { 2 } \beta , } & { \displaystyle | a - b | \geq \beta . } \end{array} \right.\tag{26}
$$

The spectral and spatial reconstruction losses are then defined as L<sub>spec-rec</sub> = mean $\left[ \rho _ { \beta _ { \mathrm { r e c } } } \left( \widehat { \mathbf { x } } _ { \lambda } , \mathbf { x } _ { \lambda } \right) \right]$ and $\mathcal { L } _ { \mathrm { s p a t - r e c } } =$ mean $\left[ \rho _ { \beta _ { \mathrm { r e c } } } \left( \widehat { \mathbf { x } } _ { s } , \mathbf { x } _ { s } \right) \right]$ , where the mean is taken over the selected masked entries when masking is enabled and over all entries otherwise.

Spectral Dynamics. The spectral dynamics module models the transition from one spectral token to the next. Given ${ \bf Z } _ { \lambda } = [ { \bf z } _ { \lambda , 1 } , \ldots , { \bf z } _ { \lambda , C } ]$ with $\mathbf { \bar { z } } _ { \lambda , c } \in \mathbb { R } ^ { B \times D }$ , the initial recurrent state is the first spectral token $\mathbf { h } _ { \lambda , 1 } \ = \ \mathbf { Z } _ { \lambda , : , 1 , : } \ \in \ \mathbb { R } ^ { B \times D }$ For each subsequent spectral position $c = \{ 2 , \ldots , C \}$ , the recurrent state is updated using a GRU cell:

$$
\mathbf { h } _ { \lambda , c } = \operatorname { G R U C e l l } _ { \lambda } \left( \mathbf { Z } _ { \lambda , : , c - 1 , : } , \mathbf { h } _ { \lambda , c - 1 } \right) \in \mathbb { R } ^ { B \times D } .\tag{27}
$$

The updated state is mapped to the predicted representation of the next spectral token using a prediction head consisting of layer normalization, a linear projection, GELU activation, dropout, and a linear projection from D to D:

$$
\widehat { \mathbf { z } } _ { \lambda , c } = H _ { \lambda } \left( \mathbf { h } _ { \lambda , c } \right) \in \mathbb { R } ^ { B \times D } ,\tag{28}
$$

where $\begin{array} { r c l } { H _ { \lambda } ( \mathbf { h } ) } & { = } & { \mathbf { W } _ { \lambda , 2 } ^ { \mathrm { d y n } } } \end{array}$ Dropout $\mathrm { [ G E L U ( W _ { \lambda , 1 } ^ { d y n } L N ( \mathbf { h } ) \Lambda + }$ $\mathbf { b } _ { \lambda , 1 } ^ { \mathrm { d y n } } ) ] + \mathbf { b } _ { \lambda , 2 } ^ { \mathrm { d y n } }$ . The resulting predictions are collected as $\widehat { \mathbf { Z } } _ { \lambda } ^ { + } \mathbf { \Lambda } = \mathbf { \Lambda }$ stack $( \widehat { \mathbf { z } } _ { \lambda , 2 } , \ldots , \widehat { \mathbf { z } } _ { \lambda , C } ) \ \in \ \mathbb { R } ^ { B \times ( C - 1 ) \times D }$ . The corresponding target is the sequence of subsequent spectral tokens with gradients detached:

$$
\mathbf { Z } _ { \lambda } ^ { + } = \mathrm { s g } \left( \mathbf { Z } _ { \lambda , : , 2 , : } \right) \in \mathbb { R } ^ { B \times ( C - 1 ) \times D } ,\tag{29}
$$

Thus, $\widehat { \mathbf { Z } } _ { \lambda } ^ { + }$ is not a slice of the latent rollout tensor $\hat { \mathbf { z } } ;$ it is the sequence of next-band predictions generated directly from the spectral tokens by the spectral dynamics module. The spectral dynamics loss is defined as the mean-squared error between the predicted next-band representations and their detached targets $\mathcal { L } _ { \mathrm { s p e c - d y n } } = \mathrm { m e a n } [ ( \widehat { \mathbf { Z } } _ { \lambda } ^ { + } - \mathbf { Z } _ { \lambda } ^ { + } ) ^ { 2 } ]$

Spatial Dynamics. The spatial dynamics module models the relationship between the center spatial token and its eight firstring neighbors. Let $\mathcal { N } _ { 8 } ( n _ { 0 } )$ denote the set of the eight spatial indices obtained from the offsets $( \Delta _ { r } , \Delta _ { c } ) \in \{ - \bar { 1 } , 0 , \bar { 1 } \} ^ { 2 } \ \backslash$ $( 0 , 0 )$ , around the center location. The corresponding neighboring spatial tokens are ${ \bf Z } _ { s , N } = { \bf Z } _ { s , : , \sqrt { } _ { 8 } ( n _ { 0 } ) , : } \in \bar { \mathbb { R } ^ { B \times 8 \times D } }$ while the center token is $\mathbf { z } _ { s , c } = \mathbf { Z } _ { s , : , n _ { 0 } , : } \in \mathbb { R } ^ { \tilde { B } \times D }$ . The center token is passed through a prediction head consisting of layer normalization, a linear projection, GELU activation, dropout, and a linear projection from D to $D \colon$

$$
\widehat { \mathbf { z } } _ { s , n } = H _ { s } \left( \mathbf { z } _ { s , c } \right) \in \mathbb { R } ^ { B \times D } ,\tag{30}
$$

where $\begin{array} { c c l } { H _ { s } ( \mathbf { z } ) } & { = } & { \mathbf { W } _ { s , 2 } ^ { \mathrm { d y n } } } \end{array}$ Dropout $\mathrm { [ G E L U ( W } _ { s , 1 } ^ { \mathrm { d y n } } \mathrm { L N } ( \mathbf { z } ) \ +$ $\mathbf { b } _ { s , 1 } ^ { \mathrm { d y n } } ) ] + \mathbf { b } _ { s , 2 } ^ { \mathrm { d y n } }$ . The same predicted representation is compared with each of the eight first-ring neighbors $\widehat { \mathbf { Z } } _ { s , N } =$ Broadcast<sub>8</sub> $( \widehat { \mathbf { z } } _ { s , n } ) \ \in \ \mathbb { R } ^ { B \times \breve { 8 } \times D }$ , where Broadcast<sub>8</sub>(·) replicates the predicted center representation across the eight neighbor positions. The corresponding target is detached from the encoder $\mathbf { Z } _ { s , \mathcal { N } } ^ { \mathrm { s t o p } } = \mathrm { s g } \left( \mathbf { Z } _ { s , : , \mathcal { N } _ { 8 } \left( n _ { 0 } \right) , : } ^ { \mathbf { \tilde { \alpha } } } \right) \ \in \ \mathbb { R } ^ { B \times 8 \times D }$ . The spatial dynamics loss is therefore $\mathcal { L } _ { \mathrm { s p a t - d y n } } = \mathrm { m e a n } [ ( \widehat { \mathbf { Z } } _ { s , N } - \mathbf { Z } _ { s , N } ^ { \mathrm { s t o p } } ) ^ { 2 } ]$ Composite Training Objective. The complete training objective combines the final classification loss, base classification loss, reconstruction constraints, spectral and spatial dynamics losses, stochastic latent rollout regularization, trajectory supervision, trajectory consistency, and the additional spectral and spatial structural constraints. To avoid fixed weighting constants, all relative contributions are controlled by nonnegative learnable training hyperparameters:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { f i n a l } } \mathcal { L } _ { \mathrm { f i n a l } } + \lambda _ { \mathrm { b a s e } } \mathcal { L } _ { \mathrm { b a s e } } + \lambda _ { \mathrm { s r } } \mathcal { L } _ { \mathrm { s p e c - r e c } } + } \\ & { \qquad \lambda _ { \mathrm { p r } } \mathcal { L } _ { \mathrm { s p a t - r e c } } + \lambda _ { \mathrm { s d } } \mathcal { L } _ { \mathrm { s p e c - d y n } } + \lambda _ { \mathrm { p d } } \mathcal { L } _ { \mathrm { s p a t - d y n } } + } \\ & { \qquad \lambda _ { \mathrm { e n v } } \mathcal { L } _ { \mathrm { e n v } } + \lambda _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } + \lambda _ { \mathrm { t r a j } } \mathcal { L } _ { \mathrm { t r a j - c l s } } + \lambda _ { \mathrm { t c } } \mathcal { L } _ { \mathrm { t r a j - c o n s } } + } \\ & { \qquad \lambda _ { \mathrm { S A M } } \mathcal { L } _ { \mathrm { S A M } } + \lambda _ { \mathrm { s m } } \mathcal { L } _ { \mathrm { s m o o t h } } + \lambda _ { \mathrm { s c } } \mathcal { L } _ { \mathrm { s p a t - c o n s } } , } \end{array}\tag{31}
$$

where all coefficients $\lambda _ { \mathrm { f i n a l } } , \lambda _ { \mathrm { b a s e } } , \lambda _ { \mathrm { s r } } , \lambda _ { \mathrm { p r } } , \lambda _ { \mathrm { s d } } , \lambda _ { \mathrm { p d } } , \lambda _ { \mathrm { e n v } } ,$ $\lambda _ { \mathrm { K L } } , \lambda _ { \mathrm { t r a j } } , \lambda _ { \mathrm { t c } } , \lambda _ { \mathrm { S A M } } , \lambda _ { \mathrm { s m } } , \lambda _ { \mathrm { s c } } \geq 0$ are hyperparameters. The final and base classification losses use label-smoothed crossentropy. Let $\ b { \ell } _ { i } ~ \in ~ \mathbb { R } ^ { K }$ denote the final logits for sample $i , \ \ell _ { \mathrm { b a s e } , i } \ \in \ \mathbb { R } ^ { K }$ the corresponding base logits, and $y _ { i } \in$ $\{ 0 , \ldots , K - 1 \}$ the ground-truth (GT) class. With labelsmoothing parameter $\varepsilon _ { \mathrm { l s } } \in [ 0 , 1 )$ , define $q _ { i , k } = ( 1 - \varepsilon _ { \mathrm { l s } } ) \mathbb { I } [ k =$ $\begin{array} { r } { y _ { i } ] + \frac { \varepsilon _ { \mathrm { l s } } } { K } } \end{array}$ and $\begin{array} { r } { \mathcal { L } _ { \mathrm { f i n a l } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { k = 0 } ^ { K - 1 } q _ { i , k } } \end{array}$ log softmax $\mathbf { \Omega } _ { : ( } \ell _ { i } ) k$ and $\begin{array} { r } { \mathcal { L } _ { \mathrm { b a s e } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { k = 0 } ^ { K - 1 } q _ { i , k } } \end{array}$ log softmax $( \ell _ { \mathrm { b a s e } , i } ) k$

The trajectory-classification loss $\mathcal { L } _ { \mathrm { t r a j - c l s } }$ is defined analogously using the trajectory logits $\ell _ { \mathrm { t r a j } , i }$ as $\begin{array} { r l } { \mathcal { L } _ { \mathrm { t r a j - c l s } } } & { { } = } \end{array}$ $- \frac { 1 } { B } et { } { ' } \sum _ { i = 1 } ^ { B } \sum _ { k = 0 } ^ { \overline { { K } } - 1 } q _ { i , k }$ log softmax $( \ell _ { \mathrm { t r a j } , i } ) _ { k }$

The trajectory-consistency loss encourages the trajectory branch to remain consistent with the detached base prediction. Let $\begin{array} { r l r l } { \mathbf { p } _ { \mathrm { b a s e } } } & { { } } & { = } \end{array}$ softmax $( \mathrm { s g } ( \ell _ { \mathrm { b a s e } } ) )$ and log $\mathbf { p } _ { \bot } \mathbf { \sigma } = \mathbf { \sigma } _ { . }$ log softmax $( \ell _ { \mathrm { t r a j } } )$ . Then $\begin{array} { r l } { \mathcal { L } _ { \mathrm { t r a j - c o n s } } } & { { } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { k = 0 } ^ { K - 1 } p _ { \mathrm { b a s e } , i , k } \log \frac { p _ { \mathrm { b a s e } , i , k } } { p _ { \mathrm { t r a j } , i , k } } , } \end{array}$

For the stochastic latent rollout, let $\widehat { \mathbf { Z } } \in \mathbb { R } ^ { B \times T \times D }$ denote the detached rollout targets, let $\pmb { \mu } ~ \in ~ \mathbb { R } ^ { B \times T \times D }$ denote the predicted Gaussian means, and let $\mathbf { V } \in \mathbb { R } ^ { B \times T \times D }$ denote the predicted log-variances. The Gaussian negative log-likelihood is

$$
\mathcal { L } _ { \mathrm { N L L } } = \frac { 1 } { 2 } \operatorname { m e a n } \left[ \mathbf { V } + \frac { \left( \widehat { \mathbf { Z } } - \mu \right) ^ { 2 } } { \exp ( \mathbf { V } ) } \right] .\tag{32}
$$

The rollout smoothness term penalizes abrupt changes between consecutive latent states L<sub>smooth-roll</sub> = mean $[ | \mathbf { Z } _ { \mathrm { r o l l } } [ : , 1 : , : ] - \mathbf { Z } _ { \mathrm { r o l l } } [ : , : - 1 , : ] | ]$ where $\begin{array} { r l r } { { \bf Z } _ { \mathrm { r o l l } } } & { { } \in } & { \mathbb { R } ^ { B \times ( T + 1 ) \ddot { \times } D } } \end{array}$ denotes the latent rollout trajectory. The complete environment rollout loss is $\begin{array} { r l r } { \mathcal { L } _ { \mathrm { e n v } } } & { { } ~ = ~ } & { \mathcal { L } _ { \mathrm { N L L } } ~ + ~ \lambda _ { \mathrm { s m o o t h - r o l l } } \mathcal { L } _ { \mathrm { s m o o t h - r o l l } } , } \end{array}$ where λ<sub>smooth-roll</sub> ≥ 0 is a hyperparameter rather than a fixed constant. The latent Gaussian regularization term is the KL divergence between the predicted Gaussian transition distribution and a standard normal prior $\begin{array} { r } { \mathcal { L } _ { \mathrm { K L } } = - \frac { 1 } { 2 } } \end{array}$ mean $\big [ 1 + \mathbf { V } - { \pmb { \mu } } ^ { 2 } - \mathrm { e x p } ( \mathbf { V } ) \big ]$

Finally, the spectral-angle, spectral-smoothness, and spatialconsistency terms are defined from the corresponding model outputs. Let $\widehat { \mathbf { x } } _ { \lambda }$ denote the spectral reconstruction. The spectral-angle loss is computed on the center spatial position of each reconstructed patch:

$$
\mathcal { L } _ { \mathrm { S A M } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \operatorname { a r c c o s } \left( \frac { \widehat { \mathbf { v } } _ { i } ^ { \top } \mathbf { v } _ { i } } { \lvert \widehat { \mathbf { v } } _ { i } \rvert ^ { 2 } \lvert \mathbf { v } _ { i } \rvert ^ { 2 } + \epsilon _ { \mathrm { S A M } } } \right) ,\tag{33}
$$

where $\widehat { \mathbf { v } } _ { i } , \mathbf { v } _ { i } \ \in \ \mathbb { R } ^ { C }$ are respectively the reconstructed and true center-pixel spectral vectors and $\epsilon _ { \mathrm { S A M } } > 0$ is a numerical-stability parameter. The spectral smoothness loss penalizes changes between adjacent reconstructed spectral bands $\mathcal { L } _ { \mathrm { s m o o t h } } = \mathrm { m e a n } \left[ | \widehat { \mathbf { x } } _ { \lambda } [ : , 2 , : ] - \widehat { \mathbf { x } } _ { \lambda } [ : , 1 , : ] | \right]$

For a batch containing spatially neighboring labeled pixels, let $\mathbf { p } _ { i } ~ = ~ \mathrm { s o f t m a x } ( \ell _ { i } )$ and let $\mathcal { P }$ denote the set of sample pairs whose spatial coordinates are within the prescribed local neighborhood. The spatial-consistency loss is $\mathcal { L } _ { \mathrm { s p a t - c o n s } } ~ =$ $\begin{array} { r } { \frac { 1 } { \left| \mathcal { P } \right| } \overset { \mathbf { \bar { \rho } } } { \sum } _ { ( i , j ) \in \mathcal { P } } \left| \mathbf { p } _ { i } - \mathbf { p } _ { j } \right| ^ { 2 } } \end{array}$ . When no valid neighboring pair is present in the mini-batch, $\mathcal { L } _ { \mathrm { s p a t - c o n s } }$ is set to zero.

## III. EXPERIMENTAL SETUP

The proposed LWM framework evaluates trajectory-aware spatial-spectral representation learning for HSIC in a withinscene, label-scarce setting, where limited labeled samples from a fixed hyperspectral scene are used to learn structured latent representations for classifying held-out samples from the same scene.

Datasets. We evaluate the proposed LWM framework on five publicly available HSI benchmarks: WHU-HongHu (HH) [52], Tangdaowan (TD) [53], DFC18 (training set only) [54], a subset of HyRANK e.g., Dioni, and Loukia [55]. These datasets encompass a broad range of scene types, spatial resolutions, spectral characteristics, and classification complexities, providing a comprehensive evaluation of the proposed framework across agricultural, urban, coastal, and mixed landcover environments.

Implementation Details. The proposed LWM model is optimized using AdamW with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ and weight decay of $7 \times 1 0 ^ { - 4 }$ . A cosine annealing learningrate scheduler is employed over 100 training epochs, while early stopping with a patience of 15 epochs is used to prevent overfitting. Gradient clipping with a maximum norm of 1.0 is applied for training stability, and mixed-precision training is utilized to improve computational efficiency. For all datasets, hyperspectral cubes are normalized using per-band standardization. Local spatial neighborhoods are extracted using mirror padding with a fixed patch size of $2 5 \times 2 5$ Unless otherwise specified, the labeled samples are divided into training, validation, and testing subsets using stratified sampling, with ratios of 2%, 2%, and 96% for the Dioni, HH, and TD datasets, and 10%, 10%, and 80% for the DFC18 and Loukia datasets, respectively. Each selected pixel serves as the center of its corresponding spatial-spectral patch; thus, the sampling ratios refer to labeled center pixels. Unless otherwise stated, a batch size of 64 is used, and label smoothing is set to 0.1.

The transformer backbone employs an embedding dimension of $D = 1 2 8 ,$ , with spectral, spatial, and fusion transformer depths set to 2. Each transformer layer uses 8 attention heads, an MLP expansion ratio of 4.0, dropout of 0.1, and stochastic depth with a maximum drop-path rate of 0.05. During training, random spectral-band masking and spatial-pixel masking are applied with masking ratios of 25% and 15%, respectively. For LWM module, the rollout horizon is fixed to $T \ = \ 3$ The stochastic transition model predicts Gaussian latent distributions with log-variance constrained to the interval $[ a , b ] .$ where the lower and upper bounds are fixed as $\textit { a } = \textit { - } 6$ and $b = 2$ . Rollout states are refined through a GRU-based latent transition module and aggregated using the proposed rollout planner. The latent state dimension is set equal to the transformer embedding dimension $( D = 1 2 8 )$ . The overall objective combines classification, reconstruction, latent dynamics, rollout regularization, trajectory supervision, and physically motivated constraints. Unless otherwise specified, the corresponding weighting coefficients are set to $\lambda _ { \mathrm { s p e c - r e c } } =$ $0 . 2 5 , \lambda _ { \mathrm { s p a t - r e c } } = 0 . 2 5 , \lambda _ { \mathrm { s p e c - d y n } } = 0 . 2 0 , \lambda _ { \mathrm { s p a t - d y n } } = 0 . 2 0 ,$ $\lambda _ { \mathrm { { r o l l } } } = 0 . 2 5 , \lambda _ { \mathrm { { K L } } } = 0 . 0 0 1 , \lambda _ { \mathrm { { t r a j - c o n s } } } = 0 . 1 0 , \lambda _ { \mathrm { { t r a j - c l s } } } = 0 . 2 5 ,$ $\lambda _ { \mathrm { S A M } } = 0 . 0 8 , \lambda _ { \mathrm { s m o o t h } } = 0 . 0 5$ , and $\lambda _ { \mathrm { s p a t - c o n s } } = 0 . 0 5$ . These hyperparameters are optimized using Bayesian optimization and kept fixed for each dataset unless otherwise specified.

Evaluation Metrics. The classification performance is evaluated using Overall Accuracy (OA), Average Accuracy (AA), and Cohen’s Kappa coefficient (κ), which are the standard metrics for HSIC. To provide a more comprehensive assessment, we additionally report the Macro-F1 score and Matthews Correlation Coefficient (MCC). Macro-F1 evaluates the harmonic mean of precision and recall averaged across all classes, whereas MCC provides a balanced measure of classification quality even in highly imbalanced scenarios.

Model calibration is assessed using the Expected Calibration Error (ECE) and multiclass Brier Score [56], [57]. ECE measures the discrepancy between prediction confidence and empirical accuracy, while the Brier Score quantifies the mean squared error between predicted probabilities and GT labels.

Lower values indicate better-calibrated probabilistic predictions. To evaluate the contribution of latent rollout reasoning, branch-wise analysis is performed by separately measuring the performance of the base classifier, trajectory classifier, and their fused prediction. In addition to OA, AA, and Kappa, we report the correction rate (CR), harm rate (HR), and net gain (NG). The CR measures the proportion of samples incorrectly classified by the base branch but correctly classified by the trajectory branch, whereas the HR measures the opposite case. Their difference, referred to as NG, quantifies the overall benefit introduced by latent rollout reasoning. Log Loss measures the quality of the predicted class probabilities by computing the average negative log-probability assigned to the GT class. Top3 accuracy measures the proportion of test samples for which the GT class appears among the three classes with the highest predicted probabilities.

## IV. LATENT ROLLOUT CONTRIBUTION

To investigate the contribution of latent rollout reasoning, we analyze the performance of the individual prediction branches and their fusion. Specifically, the proposed framework produces three predictions: (i) the Base branch, which relies solely on the fused transformer representation; (ii) the Trajectory branch, which performs classification using the planned latent state obtained from stochastic rollout reasoning; and (iii) the Final prediction, which combines both branches through logit fusion. This analysis provides insight into whether latent rollout modeling contributes complementary information beyond static spatial-spectral encoding. Table I reports the branch-wise classification results. The performance difference between an individual branch and the final prediction is quantified as $\Delta \ : = \ : \mathrm { O A } _ { \mathrm { F i n a l } } \ : - \ : \mathrm { O A } _ { \mathrm { B r a n c h } }$ , where $\mathrm { O A } _ { \mathrm { B r a n c h } }$ denotes the OA of either the Base or Trajectory branch. A positive $\Delta$ indicates that fusion improves upon the corresponding individual predictor.

TABLE I: Branch-wise performance of the Base, Trajectory, and Final prediction branches. ∆ denotes the OA improvement of the final fused prediction over the corresponding branch. Best results are in boldface.
<table><tr><td colspan="2">Branch</td><td>OA</td><td>AA</td><td>κ</td><td>∆</td></tr><tr><td rowspan="3">DC18</td><td>Base</td><td>0.9100</td><td>0.9198</td><td>0.8858</td><td>+0.0149</td></tr><tr><td>Trajectory</td><td>0.9113</td><td>0.9273</td><td>0.8976</td><td>+0.0136</td></tr><tr><td>Final</td><td>0.9249</td><td>0.9352</td><td>0.9021</td><td>一</td></tr><tr><td rowspan="3">Dion</td><td>Base</td><td>0.8648</td><td>0.8835</td><td>0.8343</td><td>+0.0523</td></tr><tr><td>Trajectory</td><td>0.9037</td><td>0.9208</td><td>0.8837</td><td>+0.0134</td></tr><tr><td>Final</td><td>0.9171</td><td>0.9346</td><td>0.8979</td><td></td></tr><tr><td rowspan="3">Loukia</td><td>Base</td><td>0.8470</td><td>0.8596</td><td>0.8207</td><td>+0.0437</td></tr><tr><td>Trajectory</td><td>0.8811</td><td>0.8678</td><td>0.8613</td><td>+0.0096</td></tr><tr><td>Final</td><td>0.8907</td><td>0.8775</td><td>0.8709</td><td></td></tr><tr><td rowspan="3">HH</td><td>Base</td><td>0.9703</td><td>0.9643</td><td>0.9677</td><td>+0.0211</td></tr><tr><td>Trajectory</td><td>0.9810</td><td>0.9763</td><td>0.9787</td><td>+0.0104</td></tr><tr><td>Final</td><td>0.9914</td><td>0.9864</td><td>0.9891</td><td></td></tr><tr><td rowspan="3">TD</td><td>Base</td><td>0.9852</td><td>0.9859</td><td>0.9846</td><td>+0.0100</td></tr><tr><td>Trajectory</td><td>0.9946</td><td>0.9953</td><td>0.9939</td><td>+0.0006</td></tr><tr><td>Final</td><td>0.9952</td><td>0.9958</td><td>0.9945</td><td></td></tr></table>

Table I shows that the Trajectory branch consistently outperforms the Base branch across all evaluated datasets. This

observation indicates that latent rollout reasoning captures predictive structure that is not fully represented by static transformer features alone. The improvement is particularly pronounced on Dioni and Loukia, where the trajectory branch increases OA from 86.48% to 90.37% and from 84.70% to 88.11%, respectively. These gains suggest that stochastic latent evolution is especially beneficial for scenes exhibiting large intra-class variability and complex spatial organization. The fused prediction consistently achieves the highest performance. The largest fusion gain is observed on Dioni, where the final model improves OA by 5.23 percentage points relative to the Base branch, highlighting the importance of latent rollout reasoning in challenging remote sensing scenarios. Although the gains are smaller on HH due to its already saturated performance, the fused prediction still improves upon both individual branches.

Beyond branch-level accuracy, we further evaluate the complementarity between the Base and Trajectory branches. Let $y _ { i }$ denote the GT label, $\hat { y } _ { i } ^ { ( b ) }$ the prediction of the Base branch, and $\hat { y } _ { i } ^ { ( t ) }$ the prediction of the Trajectory branch. The CR measures the proportion of samples that are misclassified by the Base branch but correctly classified by the Trajectory branch $\begin{array} { r } { \mathrm { { C R } } = \frac { 1 } { N _ { \mathrm { { t e s t } } } } \sum _ { i = 1 } ^ { N _ { \mathrm { { t e s t } } } } \mathbb { I } \left( \mathcal { \hat { y } } _ { i } ^ { ( b ) } \neq y _ { i } \wedge \mathcal { \hat { y } } _ { i } ^ { ( t ) } = y _ { i } \right) } \end{array}$ , where I(·) is the indicator function and $N _ { \mathrm { t e s t } }$ is the number of test samples. Similarly, the HR measures the proportion of samples correctly classified by the Base branch but misclassified by the Trajectory branch $\begin{array} { r } { \mathrm { H R } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \mathbb { I } \left( \hat { y } _ { i } ^ { ( b ) } = y _ { i } \wedge \hat { y _ { i } ^ { ( t ) } } \neq y _ { i } \right) } \end{array}$

The overall benefit of the trajectory branch is quantified through the NG as $\begin{array} { r c l } { \mathrm { N G } } & { = } & { \mathrm { C R \mathrm { ~ - ~ } H R } } \end{array}$ . To estimate the maximum achievable benefit from combining both branches, we define the Oracle Accuracy $\mathrm { O A } _ { \mathrm { o r a c l e } } =$ $\begin{array} { r } { \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \mathbb { I } \left( \hat { y } _ { i } ^ { ( b ) } = y _ { i } \vee \hat { y } _ { i } ^ { ( t ) } = y _ { i } \right) } \end{array}$

The Agreement Rate (AGR) measures the fraction of samples on which both branches produce identical predictions $\begin{array} { r } { \mathrm { \ddot { A } G R } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \mathbb { I } \left( \hat { y } _ { i } ^ { ( b ) } = \hat { \hat { y } } _ { i } ^ { ( t ) } \right) } \end{array}$

Finally, the diversity between the predictive distributions of the two branches is measured using the Jensen-Shannon divergence (JSD) as $\begin{array} { r } { \mathrm { J S D } = \frac { 1 } { 2 } D _ { \mathrm { K L } } \big ( \mathbf { p } ^ { ( \widetilde { b } ) } \left\| \mathbf { m } \right) + \frac { 1 } { 2 } D _ { \mathrm { K L } } \big ( \mathbf { p } ^ { ( t ) } \left\| \mathbf { m } \right) } \end{array}$ ， $\begin{array} { r } { \mathbf { \dot { m } } = \frac { 1 } { 2 } \left( \mathbf { p } ^ { ( b ) } + \mathbf { p } ^ { ( t ) } \right) } \end{array}$ and the Kullback-Leibler divergence is defined as $\begin{array} { r } { D _ { \mathrm { K L } } ( \mathbf { p } \| \mathbf { q } ) = \sum _ { k = 0 } ^ { K - 1 } p _ { k } \log \left( \frac { p _ { k } } { q _ { k } } \right) } \end{array}$ , where, $\mathbf { p } ^ { ( b ) }$ and $\mathbf { p } ^ { ( t ) }$ denote the softmax probability vectors from the Base and Trajectory branches, respectively. A larger JSD indicates greater diversity between the Base and Trajectory predictions.

The correction analysis in Table II provides a direct assessment of the complementary behavior between the Base and Trajectory branches. In most datasets, the CR exceeds the HR, resulting in positive NG values. The largest gains are observed on Dioni and Loukia, where the trajectory branch corrects 6.41% and 6.37% of the errors made by the Base branch while introducing only 1.51% and 1.96% additional errors, respectively. This yields NGs of 4.89% and 4.41%, demonstrating substantial complementary information from the rollout branch on these datasets. In contrast, for TD, the CR is 0.11% while the HR is 0.18%, resulting in a negative NG of −0.06%. This indicates that the trajectory branch introduces slightly more additional errors than the number of Base-branch

errors it corrects on this dataset.  
TABLE II: Trajectory branch correction analysis. Higher CR, NG, and $\mathrm { O A } _ { \mathrm { o r a c l e } }$ indicate stronger complementary information provided by the latent rollout branch.
<table><tr><td>Dataset</td><td>CR (%)</td><td>HR (%)</td><td>NG (%)</td><td> $\mathrm { O A } _ { \mathrm { o r a c l e } }$  (%)</td><td>AGR (%)</td><td>JSD</td></tr><tr><td>DFC18</td><td>1.76</td><td>1.25</td><td>0.51</td><td>93.67</td><td>96.31</td><td>0.0196</td></tr><tr><td>Dioni</td><td>6.41</td><td>1.51</td><td>4.89</td><td>92.89</td><td>90.86</td><td>0.0817</td></tr><tr><td>Loukia</td><td>6.37</td><td>1.96</td><td>4.41</td><td>91.08</td><td>90.74</td><td>0.0433</td></tr><tr><td>HH</td><td>0.25</td><td>0.17</td><td>0.08</td><td>99.28</td><td>99.54</td><td>0.0033</td></tr><tr><td>TD</td><td>0.11</td><td>0.18</td><td>-0.06</td><td>99.64</td><td>99.70</td><td>0.0017</td></tr></table>

The $\mathrm { O A } _ { \mathrm { o r a c l e } }$ further reveals the potential upper bound obtainable from combining both branches, reaching 92.89% on Dioni and 93.67% on DFC18. Moreover, the JSD correlates strongly with the observed gains. Dioni exhibits the largest JSD (0.0817) and the largest NG, indicating that greater predictive diversity between branches translates into larger performance improvements after fusion. In contrast, HH exhibits very high AGR (99.54%) and very small JSD (0.0033), indicating that both branches converge to nearly identical predictions on this near-saturated dataset.

## V. ROBUSTNESS ANALYSIS

A practical HSIC system should remain reliable under incomplete spectral observations and sensor noise. To evaluate the robustness of the proposed LWM framework, we consider two challenging perturbation scenarios: (i) random spectral band removal and (ii) additive Gaussian noise. For spectral corruption, a fraction ρ of spectral bands is randomly removed and set to zero. Let m ${ \bf \bar { \Psi } } \in \mathrm { \bf \Psi } \{ 0 , 1 \} ^ { C }$ be a random binary spectral mask satisfying $\mathrm { P r } ( m _ { c } ~ = ~ 0 ) ~ = ~ \rho .$ The mask is expanded across the batch and spatial dimensions as $\tilde { \mathbf { m } } = \mathbf { 1 } _ { B } \tilde { \otimes } \mathbf { m } \otimes \mathbf { 1 } _ { P } \otimes \mathbf { 1 } _ { P } \in \{ 0 , 1 \} ^ { B \times C \times \overset { \cdot } { P } \times P }$ , where $\mathbf { 1 } _ { B }$ and ${ \bf 1 } _ { P }$ denote vectors of ones, and ⊗ denotes replication along the corresponding dimensions. The corrupted input is then defined as $\widetilde { \mathbf { x } } = \mathbf { x } \odot \widetilde { \mathbf { m } }$

For noise robustness, zero-mean Gaussian noise is added to the input $\widetilde { \mathbf { x } } = \mathbf { x } + \pmb { \eta } , \pmb { \eta } \sim \mathcal { N } \left( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } \right)$ , where $\pmb { \eta } \in \mathbb { R } ^ { B \times C \times P \times P }$ has the same dimensionality as x. The experiments are performed without retraining the model. Performance is evaluated using OA, AA, and κ under increasing corruption levels. The results are reported in Table III.

The robustness results reveal several important characteristics of the proposed LWM framework. First, LWM exhibits strong resilience to additive Gaussian noise across all datasets. Even at $\sigma = 0 . 0 5$ , the degradation in OA remains negligible. For example, on DFC18, the OA decreases from 92.49% to 92.42%, while on HH, the performance remains virtually unchanged. This behavior indicates that the learned latent representations are stable under moderate perturbations and are not overly dependent on individual spectral measurements.

Second, performance degrades more gradually under spectral band removal than would typically be expected for models relying heavily on individual wavelengths. Although accuracy decreases as the fraction of removed bands increases, the model retains strong predictive capability even when 30% of the spectral information is discarded. On Dioni and Loukia, the

TABLE III: Robustness evaluation under spectral band removal and additive Gaussian noise.
<table><tr><td>Dataset</td><td>Perturbation</td><td>Level</td><td>OA</td><td>AA</td><td>κ</td></tr><tr><td rowspan="6">DFC18</td><td>Missing Bands</td><td>10%</td><td>0.9132</td><td>0.9341</td><td>0.8872</td></tr><tr><td>Missing Bands</td><td>20%</td><td>0.8752</td><td>0.9130</td><td>0.8389</td></tr><tr><td>Missing Bands</td><td>30%</td><td>0.7925</td><td>0.8514</td><td>0.7355</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 1$ </td><td>0.9249</td><td>0.9352</td><td>0.9020</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 3$ </td><td>0.9247</td><td>0.9351</td><td>0.9018</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 5$ </td><td>0.9242</td><td>0.9344</td><td>0.9012</td></tr><tr><td rowspan="6">Dioni</td><td>Missing Bands</td><td>10%</td><td>0.9167</td><td>0.9092</td><td>0.8967</td></tr><tr><td>Missing Bands</td><td>20%</td><td>0.8928</td><td>0.8662</td><td>0.8661</td></tr><tr><td>Missing Bands</td><td>30%</td><td>0.8372</td><td>0.7869</td><td>0.7937</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 1$ </td><td>0.9170</td><td>0.9346</td><td>0.8978</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 3$ </td><td>0.9168</td><td>0.9345</td><td>0.8975</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 5$ </td><td>0.9165</td><td>0.9343</td><td>0.8972</td></tr><tr><td rowspan="6">Loukia</td><td>Missing Bands</td><td>10%</td><td>0.8876</td><td>0.8666</td><td>0.8669</td></tr><tr><td>Missing Bands</td><td>20%</td><td>0.8717</td><td>0.8435</td><td>0.8476</td></tr><tr><td>Missing Bands</td><td>30%</td><td>0.8442</td><td>0.8012</td><td>0.8143</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 1$ </td><td>0.8903</td><td>0.8770</td><td>0.8704</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 3$ </td><td>0.8911</td><td>0.8782</td><td>0.8714</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 5$ </td><td>0.8905</td><td>0.8779</td><td>0.8707</td></tr><tr><td rowspan="6">HH</td><td>Missing Bands</td><td>10%</td><td>0.9917</td><td>0.9860</td><td>0.9895</td></tr><tr><td>Missing Bands</td><td>20%</td><td>0.9887</td><td>0.9792</td><td>0.9857</td></tr><tr><td>Missing Bands</td><td>30%</td><td>0.9686</td><td>0.9476</td><td>0.9603</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 1$ </td><td>0.9914</td><td>0.9865</td><td>0.9892</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 3$ </td><td>0.9914</td><td>0.9865</td><td>0.9892</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 5$ </td><td>0.9914</td><td>0.9864</td><td>0.9892</td></tr><tr><td rowspan="6">TD</td><td>Missing Bands</td><td>10%</td><td>0.9925</td><td>0.9940</td><td>0.9915</td></tr><tr><td>Missing Bands</td><td>20%</td><td>0.9802</td><td>0.9838</td><td>0.9774</td></tr><tr><td>Missing Bands</td><td>30%</td><td>0.9441</td><td>0.9499</td><td>0.9364</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 1$ </td><td>0.9952</td><td>0.9958</td><td>0.9946</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 3$ </td><td>0.9952</td><td>0.9958</td><td>0.9945</td></tr><tr><td>Gaussian Noise</td><td> $\sigma = 0 . 0 5$ </td><td>0.9952</td><td>0.9958</td><td>0.9945</td></tr></table>

OA remains above 83%, while HH retains an OA of 96.86%. These results suggest that the latent rollout mechanism learns distributed spatial-spectral representations that remain informative even when substantial portions of the spectrum are unavailable.

Third, the robustness trends are consistent with the design of LWM. The masked reconstruction objectives encourage recovery of missing information from surrounding spectral and spatial context, while the spectral and spatial dynamics constraints enforce structured relationships between latent representations. In addition, stochastic latent rollout reasoning promotes representation consistency across multiple plausible latent states rather than relying on a single deterministic embedding. Consequently, the model becomes less sensitive to localized corruption and measurement noise.

## VI. ATTENTION-BASED INTERPRETABILITY ANALYSIS

To better understand the decision-making behavior, we analyze the spatial distribution of the learned attention responses. While quantitative metrics evaluate classification performance, they provide limited insight into how the model utilizes spatialspectral information during inference. To address this limitation, the learned attention responses are projected back to the image plane and visualized as attention heatmaps and attention overlays on the corresponding GT scenes. These visualizations (Figs 2-6) are further compared with classification error maps to examine the relationship between attention localization and prediction quality. Such analysis provides qualitative evidence regarding whether the latent rollout mechanism guides the model toward discriminative and spatially coherent scene representations.

![](images/0ce49c56a3913b599f579d5e557c182c428f468ce1d4c1c494bb31ee6bb913e6.jpg)

![](images/6612347d22ace6c84882cd268a0bb1527d07143df4a899d9aab0ef3a6ab02eb1.jpg)  
(b) Attention Heatmap

(a) GT  
![](images/17a7b817669f95aa2d621b0de840cc4a7243b395450b6825186e27d559c3ab72.jpg)

![](images/05433801688b7ea833e10edfb3007306100203a3dfbea0051f8e4987e7849177.jpg)  
(c) Attention Overlay  
(d) Error Map  
Fig. 2: Attention-based interpretability analysis on the DFC18 dataset. (a) GT. (b) Spatial distribution of the learned attention responses. (c) Attention responses overlaid on GT, illustrating the regions that contribute most strongly to the classification decision. (d) Error map, where green pixels indicate correctly classified samples and red pixels indicate misclassified samples.

![](images/5f0651ac03e02a6bc1cda6464f3a6af11ee38cd15b4e94d7a161d551e4514ab1.jpg)  
(a) GT

![](images/5653d5e66097da68a2fdc19b8fa835ff6c32a10bb0d578769dca775f786e2647.jpg)

![](images/2a03642a000166ca90f27c305c9270b3c32f2c6047df71ddd3b27c85000ef084.jpg)

![](images/3c206999430dc153b6906dc9afe0f7df9f312be0ddce557c399b9e1d85911569.jpg)  
(c) Attention Overlay  
(d) Error Map  
Fig. 3: Attention-based analysis on the Dioni dataset.

![](images/440af49430fe4aaa360e8405a024c28c65c4484a4143f95abb2dabe50d7b3d88.jpg)  
(a) GT

![](images/38d22058998c28436f1e84034393374fcf67eecb5b1991ad5da2dd1821c46f63.jpg)

![](images/e423bdcb860d764426c9eb1fef533f95fed3d0df6612144850a1e26c62a082af.jpg)

(b) Attention Heatmap  
![](images/0a7cfa1224abef7b7d5241a9b47ce2a5717af415a24214e0e023b527cec30660.jpg)  
(c) Attention Overlay  
(d) Error Map  
Fig. 4: Attention-based analysis on the Loukia dataset.

The attention visualizations reveal several consistent patterns. First, high-attention responses are concentrated within semantically homogeneous regions rather than being uniformly distributed across the scene. In DFC18 (Fig. 2), the learned attention focuses on dominant urban structures, including transportation corridors, vegetation regions, and built-up areas. For the Dioni and Loukia datasets (Figs. 3 and 4), attention responses remain spatially coherent over large land-cover regions, indicating that the model successfully captures longrange contextual relationships despite substantial intra-class variability. Similarly, for the agricultural HH dataset (Fig. 5), the attention maps exhibit highly compact and class-consistent patterns that align closely with crop distributions.

![](images/35738c9ccd418bf4fb516eb0771f24e61cb7831a1c1ef6ec2a9e30d6b66cfc41.jpg)  
(a) GT

![](images/eb6e322dc6908e3b54c8981e016b3ae333d303bc3c8482269784fda581f61050.jpg)  
(b) Heatmap

![](images/d2db4130761d2256d57af0699393b5471d99e38ebb435f15f29c31f91a136159.jpg)  
(c) Overlay

![](images/0d662b8e0dfc362e915ffbfe7fce5426bae20fb15077c311cf6249444aca7b5b.jpg)  
(d) Error Map

Fig. 5: Attention-based analysis on the HH dataset.  
![](images/ed671e5a84983ff5ec68e07bf6c7eb60893338645f9930bf6b0f2049710840a1.jpg)  
(a) GT

![](images/ffae3328b4afd9f242ec4e5c20ef67cd6bf04f4b2542663b9ce129b07a15f19f.jpg)

![](images/44140bb75bef3b02b29c831227084c2da5817d0bd23d872799cae645a97de4ca.jpg)  
(b) Heatmap  
(c) Overlay

![](images/9c153e40d413b2fd15863f1eea890a18def540ec9b27b90d9a132ee292817a1f.jpg)  
(d) Error Map  
Fig. 6: Attention-based analysis on the TD dataset.

Second, the attention overlays demonstrate strong correspondence between the learned responses and semantically meaningful structures in the GT scenes. Regions receiving close attention generally coincide with dominant land-cover patterns and object boundaries that are critical for discrimination. This behavior suggests that the latent rollout mechanism encourages the model to focus on informative spatial-spectral structures while suppressing irrelevant background responses. Third, the error maps show that the majority of misclassifications occur near class boundaries and transition regions where neighboring categories exhibit similar spectral characteristics. In contrast, regions associated with strong attention responses are predominantly classified correctly.

## VII. ABLATION STUDY

To quantify the contribution of each component, we progressively construct the proposed LWM framework from a baseline Spatial-Spectral Transformer (SST). Specifically, five configurations are evaluated: (i) SST only, which contains only the spatial-spectral transformer backbone; (ii) SST+Mask Recon, which additionally introduces masked spectral and spatial reconstruction objectives; (iii) SST+World, which incorporates the stochastic LWM module and rollout reasoning; (iv) LWM w/o Physical Priors, which further includes latent rollout planning and trajectory supervision but excludes SAM, spectral smoothness, and spatial consistency constraints; and (v) the full LWM model. The relative improvement of the full LWM model over the SST baseline is computed as Relative $\begin{array} { r } { \mathrm { G a i n } ( \% ) = \frac { \mathcal { M } _ { \mathrm { L W M } } - \mathcal { M } _ { \mathrm { S S T } } } { \mathcal { M } _ { \mathrm { S S T } } } \times 1 0 0 } \end{array}$ , where M denotes the evaluation metric of interest. For metrics where lower values are better, such as Log Loss, negative values indicate performance improvement.

TABLE IV: Ablation study of the proposed LWM framework. Relative Gain (%) is computed with respect to the SST-only baseline. For OA, AA, κ, MCC, Top3, and Macro-F1, higher values are better, whereas for Loss, Log Loss, Brier Score, and ECE, lower values are preferred. Best values are highlighted in bold.
<table><tr><td>Dataset</td><td>Setting</td><td>Loss↓</td><td>0A↑</td><td>AA↑</td><td>κ↑</td><td>MCC↑</td><td>Log Loss↓</td><td>Top3↑</td><td>Macro-F1↑</td><td>Brier↓</td><td>ECE↓</td></tr><tr><td rowspan="6">DFC18</td><td>SST only</td><td>0.5794</td><td>0.8992</td><td>0.9083</td><td>0.8683</td><td>0.8689</td><td>0.3925</td><td>0.9830</td><td>0.8919</td><td>0.1447</td><td>0.0385</td></tr><tr><td>SST+Mask Recon</td><td>0.6267</td><td>0.9076</td><td>0.9190</td><td>0.8796</td><td>0.8802</td><td>0.3390</td><td>0.9841</td><td>0.8994</td><td>0.1630</td><td>0.0493</td></tr><tr><td>SST+World</td><td>0.5196</td><td>0.9175</td><td>0.9324</td><td>0.8925</td><td>0.8931</td><td>0.2832</td><td>0.9878</td><td>0.9030</td><td>0.1260</td><td>0.0280</td></tr><tr><td>LWM w/o Physical</td><td>0.4601</td><td>0.9225</td><td>0.9325</td><td>0.8991</td><td>0.8996</td><td>0.2942</td><td>0.9877</td><td>0.9082</td><td>0.1234</td><td>0.0296</td></tr><tr><td>Full LWM</td><td>0.4100</td><td>0.9249</td><td>0.9352</td><td>0.9021</td><td>0.9024</td><td>0.2783</td><td>0.9899</td><td>0.9226</td><td>0.1192</td><td>0.0283</td></tr><tr><td>Relative Gain (%)</td><td>-29.23</td><td>+2.86</td><td>+2.96</td><td>+3.89</td><td>+3.86</td><td>-29.10</td><td>+0.70</td><td>+3.44</td><td>-17.62</td><td>-26.49</td></tr><tr><td rowspan="6">Dioni</td><td>SST only</td><td>0.7405</td><td>0.8388</td><td>0.8696</td><td>0.8035</td><td>0.8069</td><td>0.4981</td><td>0.9829</td><td>0.8061</td><td>0.2341</td><td>0.0436</td></tr><tr><td>SST+Mask Recon</td><td>0.7712</td><td>0.8688</td><td>0.8888</td><td>0.8395</td><td>0.8408</td><td>0.4318</td><td>0.9847</td><td>0.8436</td><td>0.2017</td><td>0.0577</td></tr><tr><td>SST+World</td><td>0.6116</td><td>0.8967</td><td>0.9166</td><td>0.8731</td><td>0.8748</td><td>0.3362</td><td>0.9922</td><td>0.8625</td><td>0.1691</td><td>0.0356</td></tr><tr><td>LWM w/o Physical</td><td>0.5774</td><td>0.8920</td><td>0.8915</td><td>0.8677</td><td>0.8689</td><td>0.3702</td><td>0.9850</td><td>0.9014</td><td>0.1626</td><td>0.0183</td></tr><tr><td>Full LWM</td><td>0.5519</td><td>0.9171</td><td>0.9346</td><td>0.8979</td><td>0.8986</td><td>0.2894</td><td>0.9943</td><td>0.9146</td><td>0.1332</td><td>0.0080</td></tr><tr><td>Relative Gain (%)</td><td>-25.47</td><td>+9.34</td><td>+7.48</td><td>+11.75</td><td>+11.36</td><td>-41.90</td><td>+1.16</td><td>+13.46</td><td>-43.10</td><td>-81.65</td></tr><tr><td rowspan="6">Loukia</td><td>SST only</td><td>0.7106</td><td>0.8541</td><td>0.8858</td><td>0.8293</td><td>0.8308</td><td>0.4360</td><td>0.9899</td><td>0.8426</td><td>0.2229</td><td>0.0415</td></tr><tr><td>SST+Mask Recon</td><td>0.7201</td><td>0.8557</td><td>0.8725</td><td>0.8307</td><td>0.8315</td><td>0.4579</td><td>0.9856</td><td>0.8400</td><td>0.2038</td><td>0.0411</td></tr><tr><td>SST+World</td><td>0.7296</td><td>0.8695</td><td>0.8626</td><td>0.8465</td><td>0.8470</td><td>0.4438</td><td>0.9820</td><td>0.8430</td><td>0.2021</td><td>0.0516</td></tr><tr><td>LWM w/o Physical</td><td>0.6806</td><td>0.8697</td><td>0.8718</td><td>0.8471</td><td>0.8479</td><td>0.4201</td><td>0.9891</td><td>0.8446</td><td>0.2208</td><td>0.0374</td></tr><tr><td>Full LWM</td><td>0.6112</td><td>0.8907</td><td>0.8775</td><td>0.8709</td><td>0.8713</td><td>0.3699</td><td>0.9902</td><td>0.8737</td><td>0.1723</td><td>0.0492</td></tr><tr><td>Relative Gain (%)</td><td>-14.00</td><td>+4.29</td><td>-0.94</td><td>+5.02</td><td>+4.88</td><td>-15.16</td><td>+0.03</td><td>+3.69</td><td>-22.70</td><td>+18.55</td></tr><tr><td rowspan="6">HH</td><td>SST only</td><td>0.2452</td><td>0.9874</td><td>0.9805</td><td>0.9841</td><td>0.9841</td><td>0.0553</td><td>0.9992</td><td>0.9649</td><td>0.0252</td><td>0.0114</td></tr><tr><td>SST+Mask Recon</td><td>0.2694</td><td>0.9906</td><td>0.9841</td><td>0.9882</td><td>0.9882</td><td>0.0484</td><td>0.9992</td><td>0.9725</td><td>0.0197</td><td>0.0119</td></tr><tr><td>SST+World</td><td>0.1608</td><td>0.9840</td><td>0.9836</td><td>0.9799</td><td>0.9799</td><td>0.0650</td><td>0.9989</td><td>0.9739</td><td>0.0198</td><td>0.0118</td></tr><tr><td>LWM w/o Physical</td><td>0.1759</td><td>0.9878</td><td>0.9812</td><td>0.9846</td><td>0.9846</td><td>0.0556</td><td>0.9992</td><td>0.9774</td><td>0.0152</td><td>0.0130</td></tr><tr><td>Full LWM</td><td>0.1496</td><td>0.9914</td><td>0.9864</td><td>0.9891</td><td>0.9891</td><td>0.0486</td><td>0.9991</td><td>0.9826</td><td>0.0144</td><td>0.0131</td></tr><tr><td>Relative Gain (%)</td><td>-38.99</td><td>+0.41</td><td>+0.60</td><td>+0.51</td><td>+0.51</td><td>-12.12</td><td>-0.01</td><td>+1.83</td><td>-42.86</td><td>+14.71</td></tr><tr><td rowspan="6">TD</td><td>SST only</td><td>0.1263</td><td>0.9893</td><td>0.9930</td><td>0.9878</td><td>0.9879</td><td>0.0652</td><td>0.9995</td><td>0.9848</td><td>0.0096</td><td>0.0148</td></tr><tr><td>SST+Mask Recon</td><td>0.1564</td><td>0.9904</td><td>0.9926</td><td>0.9891</td><td>0.9891</td><td>0.0533</td><td>0.9996</td><td>0.9927</td><td>0.0128</td><td>0.0147</td></tr><tr><td>SST+World</td><td>0.2661</td><td>0.9918</td><td>0.9946</td><td>0.9907</td><td>0.9907</td><td>0.0410</td><td>0.9999</td><td>0.9911</td><td>0.0191</td><td>0.0112</td></tr><tr><td>LWM w/o Physical</td><td>0.2091</td><td>0.9942</td><td>0.9959</td><td>0.9934</td><td>0.9934</td><td>0.0364</td><td>0.9995</td><td>0.9913</td><td>0.0162</td><td>0.0141</td></tr><tr><td>Full LWM</td><td>0.1166</td><td>0.9952</td><td>0.9958</td><td>0.9945</td><td>0.9945</td><td>0.0324</td><td>0.9999</td><td>0.9947</td><td>0.0079</td><td>0.0153</td></tr><tr><td>Relative Gain (%)</td><td>-7.68</td><td>+0.60</td><td>+0.28</td><td>+0.68</td><td>+0.67</td><td>-50.31</td><td>+0.04</td><td>+1.01</td><td>-17.71</td><td>+3.38</td></tr></table>

From Table IV, several important observations can be made. First, masked spatial-spectral reconstruction consistently improves performance over the SST baseline. Although the gains in OA are relatively modest, reconstruction supervision produces noticeable improvements in AA, κ, MCC, and calibration metrics. This indicates that recovering masked spectral bands and spatial pixels encourages the encoder to preserve richer spatial-spectral information, resulting in more robust latent representations under limited supervision.

Second, introducing the stochastic LWM yields substantially larger improvements than reconstruction alone. For example, on Dioni, OA increases from 83.88% to 89.67%, while κ improves from 0.8035 to 0.8731. Similar trends are observed on DFC18 and Loukia. These results demonstrate that latent rollout reasoning provides information beyond conventional transformer-based feature aggregation.

Third, trajectory planning and trajectory-aware supervision further improve performance. Comparing SST+World with LWM w/o physical priors shows consistent gains in OA, κ, MCC, and Macro-F1 on most datasets. This observation validates the effectiveness of aggregating multiple rollout states through learned importance weighting rather than relying solely on the initial latent representation. The results suggest that trajectory-aware aggregation enables the classifier to exploit complementary information distributed across the latent rollout trajectory.

Finally, the full LWM model achieves the best overall performance across almost all datasets. The inclusion of physically motivated priors, including spectral angle alignment, spectral smoothness, and spatial consistency regularization, consistently improves classification accuracy while simultaneously reducing probabilistic uncertainty. In particular, Dioni exhibits the largest gains, where the full model achieves relative improvements of 9.34%, 7.48%, 11.75%, and 11.36% in OA, AA, κ, and MCC, respectively, compared with the SST baseline. Furthermore, the Brier score is reduced by 43.10% and ECE by 81.65%, indicating substantially improved prediction calibration. Similar trends are observed on DFC18, where calibration improves by 17.62% and 26.49% in terms of Brier score and ECE, respectively. The HH dataset exhibits comparatively smaller improvements because the baseline transformer already achieves near-saturated performance.

TABLE V: Comparison with recent state-of-the-art HSIC methods. Results are reported as ${ \mathrm { m e a n ~ } } \pm$ standard deviation over five runs. The best and second-best results are highlighted in bold and underlined italics, respectively. The training sample ratio used by each method is reported in the row “Training Samples (%)”.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Metric</td><td colspan="8">Methods</td></tr><tr><td>LWM</td><td>WF</td><td>DF</td><td>MCMTN</td><td>SMFAT</td><td>MSST</td><td>S2CIFT</td><td>S2CAT</td></tr><tr><td>Training Samples (%)</td><td></td><td>10</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td></tr><tr><td></td><td>OA</td><td> $\mathbf { 9 2 . 4 9 9 7 { \scriptstyle \pm 0 . 1 0 9 6 } }$ </td><td> $9 1 . 4 1 0 1 { \pm } 0 . 5 5 5 9$ </td><td> $8 9 . 6 1 0 9 { \scriptstyle \pm 0 . 0 7 0 7 }$ </td><td> $9 0 . 4 0 7 0 { \scriptstyle \pm 0 . 1 6 7 3 }$ </td><td>91.9506±0.1864</td><td> $8 7 . 0 5 4 1 { \scriptstyle \pm 0 . 3 5 0 9 }$ </td><td> $9 1 . 6 8 6 6 { \pm } 0 . 0 1 5 9$ </td><td> $9 1 . 5 2 9 3 { \scriptstyle \pm 0 . 2 2 0 6 }$ </td></tr><tr><td>DCI18</td><td>AA</td><td> $\mathbf { 9 3 . 5 2 7 5 \pm 0 . 0 1 2 6 }$ </td><td> $8 8 . 9 7 8 9 { \scriptstyle \pm 0 . 1 0 7 1 }$ </td><td> $8 1 . 3 0 5 7 { \scriptstyle \pm 0 . 3 9 9 3 }$ </td><td> $8 8 . 0 6 3 9 { \pm } 0 . 1 9 8 3$ </td><td> $\overline { { 8 5 . 6 9 4 8 { \pm 0 . 1 3 8 6 } } }$ </td><td> $7 8 . 4 1 4 3 { \scriptstyle \pm 0 . 7 3 8 9 }$ </td><td> $9 0 . 9 7 6 3 { \scriptstyle \pm 0 . 3 4 9 4 }$ </td><td> $8 8 . 9 7 0 7 { \scriptstyle \pm 0 . 9 1 8 0 }$ </td></tr><tr><td></td><td>κ</td><td> $\mathbf { 9 0 . 2 1 9 1 { \scriptstyle \pm 0 . 0 2 3 1 } }$ </td><td> $\underline { { 8 9 . 9 8 0 7 \pm 0 . 7 3 7 8 } }$ </td><td> $8 6 . 2 5 1 6 { \scriptstyle \pm 0 . 1 7 3 0 }$ </td><td> $8 7 . 2 4 0 9 { \scriptstyle \pm 0 . 1 4 5 6 }$ </td><td> $8 9 . 0 0 4 6 { \scriptstyle \pm 0 . 1 0 6 4 }$ </td><td> $8 2 . 7 7 2 5 { \scriptstyle \pm 0 . 4 6 9 9 }$ </td><td> $8 9 . 6 3 8 8 { \pm } 0 . 0 1 9 3$ </td><td> $8 9 . 1 5 9 9 { \scriptstyle \pm 0 . 3 3 7 6 }$ </td></tr><tr><td></td><td>OA</td><td> $\mathbf { 8 9 . 2 1 0 7 \pm 0 . 1 0 9 4 }$ </td><td> $8 8 . 3 2 0 5 { \scriptstyle \pm 0 . 2 0 9 5 }$ </td><td> $8 8 . 0 2 5 8 { \scriptstyle \pm 0 . 9 5 3 0 }$ </td><td> $8 0 . 4 6 8 4 { \scriptstyle \pm 0 . 2 9 8 3 }$ </td><td> $8 2 . 7 8 2 6 { \scriptstyle \pm 0 . 1 9 5 6 }$ </td><td> $7 6 . 2 8 8 5 { \pm } 1 . 2 3 5 8$ </td><td> $8 7 . 3 6 5 5 { \scriptstyle \pm 0 . 1 8 8 5 }$ </td><td>87.2983±0.3037</td></tr><tr><td>Ioukia</td><td>AA</td><td> $\mathbf { 8 7 . 9 8 7 5 \pm 0 . 1 4 5 3 }$ </td><td> $\overline { { 8 7 . 2 2 5 3 \pm 0 . 3 4 7 7 } }$ </td><td> $8 3 . 2 5 7 1 { \scriptstyle \pm 2 . 2 7 0 7 }$ </td><td> $8 0 . 0 3 7 5 { \scriptstyle \pm 0 . 2 0 9 1 }$ </td><td> $8 2 . 5 0 4 6 { \scriptstyle \pm 0 . 2 1 9 3 }$ </td><td> $6 7 . 3 7 5 6 { \scriptstyle \pm 0 . 2 0 7 2 }$ </td><td> $8 6 . 2 4 4 3 { \scriptstyle \pm 0 . 1 0 0 1 }$ </td><td> $8 6 . 1 6 3 3 { \scriptstyle \pm 0 . 5 2 9 5 }$ </td></tr><tr><td></td><td>κ</td><td> $\mathbf { 8 7 . 9 9 1 6 { \pm 0 . 1 3 1 2 } }$ </td><td> $8 7 . 0 3 6 3 { \scriptstyle \pm 0 . 2 5 5 8 }$ </td><td> $8 5 . 7 3 8 1 { \pm } 1 . 1 0 6 9$ </td><td> $7 6 . 5 4 5 9 { \scriptstyle \pm 0 . 1 6 5 9 }$ </td><td> $7 9 . 4 1 1 3 { \pm } 0 . 2 5 1 0$ </td><td> $7 1 . 8 0 7 0 { \scriptstyle \pm 1 . 4 1 1 0 }$ </td><td> $8 6 . 7 1 2 1 { \scriptstyle \pm 0 . 2 2 5 4 }$ </td><td> $8 7 . 0 1 2 5 { \scriptstyle \pm 0 . 3 6 4 5 }$ </td></tr><tr><td>Training Samples (%)</td><td></td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td></td><td>OA</td><td> $\mathbf { 9 1 . 7 1 3 2 \pm 0 . 1 4 5 3 }$ </td><td> $9 0 . 8 9 7 9 { \scriptstyle \pm 0 . 4 6 6 9 }$ </td><td> $9 0 . 7 2 2 8 { \scriptstyle \pm 0 . 0 1 9 6 }$ </td><td> $8 8 . 9 9 2 9 { \scriptstyle \pm 0 . 2 3 1 9 }$ </td><td> $8 3 . 2 1 9 5 { \scriptstyle \pm 0 . 1 6 3 9 }$ </td><td> $8 9 . 0 9 1 1 { \scriptstyle \pm 1 . 4 5 9 6 }$ </td><td> $g 0 . 9 9 0 7 { \scriptstyle \pm 0 . 8 2 4 0 }$ </td><td> $9 0 . 9 8 9 5 { \scriptstyle \pm 0 . 1 5 6 9 }$ </td></tr><tr><td>Dionı</td><td>AA</td><td> $9 3 . 4 6 9 8 { \scriptstyle \pm 0 . 1 6 5 2 }$ </td><td> $\underline { { 8 9 . 7 3 3 0 { \pm } 2 . 0 5 8 0 } }$ </td><td> $8 5 . 1 5 9 4 { \scriptstyle \pm 0 . 7 7 6 5 }$ </td><td> $7 6 . 1 7 7 4 { \scriptstyle \pm 0 . 1 9 8 2 }$ </td><td> $7 0 . 9 2 0 6 { \scriptstyle \pm 0 . 2 1 9 7 }$ </td><td> $7 5 . 2 1 7 3 { \scriptstyle \pm 6 . 3 9 1 7 }$ </td><td> $8 6 . 8 3 5 9 { \scriptstyle \pm 1 . 9 0 7 8 }$ </td><td> $8 9 . 5 4 0 6 { \scriptstyle \pm 3 . 6 4 7 9 }$ </td></tr><tr><td></td><td>κ</td><td> $\mathbf { 8 9 . 7 9 8 9 \pm 0 . 1 4 5 2 }$ </td><td> $8 8 . 6 6 1 0 { \scriptstyle \pm 0 . 5 8 9 8 }$ </td><td> $8 8 . 9 7 8 4 { \scriptstyle \pm 0 . 0 2 3 6 }$ </td><td> $8 6 . 2 6 3 5 { \scriptstyle \pm 0 . 1 5 9 6 }$ </td><td> $7 8 . 9 0 4 0 { \scriptstyle \pm 0 . 1 9 3 2 }$ </td><td> $8 6 . 3 1 8 4 \pm 1 . 9 4 8 5$ </td><td> $8 9 . 5 2 6 2 { \scriptstyle \pm } I . 0 2 7 9$ </td><td> $8 8 . 7 8 0 6 { \scriptstyle \pm 0 . 2 0 2 6 }$ </td></tr><tr><td></td><td>OA</td><td> $\mathbf { 9 9 . 1 4 9 7 \pm 0 . 0 9 1 4 }$ </td><td> $9 6 . 9 6 7 7 { \scriptstyle \pm 0 . 1 9 7 9 }$ </td><td> $9 6 . 8 4 8 8 { \scriptstyle \pm 0 . 0 6 2 5 }$ </td><td> $9 8 . 1 8 7 1 { \scriptstyle \pm 0 . 1 9 3 4 }$ </td><td> $9 8 . 8 7 4 0 { \scriptstyle \pm 0 . 1 8 6 4 }$ </td><td> $9 7 . 1 1 0 5 { \pm } 0 . 2 3 9 1$ </td><td> $9 7 . 8 3 7 0 { \scriptstyle \pm 0 . 3 5 6 0 }$ </td><td> $9 7 . 0 1 7 8 { \scriptstyle \pm 0 . 0 1 0 1 }$ </td></tr><tr><td>HH</td><td>AA</td><td> $\mathbf { 9 8 . 6 4 8 9 { \pm 0 . 0 9 1 2 } }$ </td><td> $9 2 . 5 0 3 7 { \pm } 1 . 0 1 1 1$ </td><td> $9 1 . 8 8 3 0 { \scriptstyle \pm 0 . 0 3 2 3 }$ </td><td> $9 5 . 3 1 5 3 { \scriptstyle \pm 0 . 1 6 8 4 }$ </td><td> $g 7 . 5 9 2 I \pm 0 . I 2 7 4$ </td><td> $9 0 . 6 2 4 4 { \pm } 2 . 2 4 8 9$ </td><td> $9 4 . 3 5 1 7 { \scriptstyle \pm 1 . 8 3 0 7 }$ </td><td> $9 3 . 0 1 4 4 { \scriptstyle \pm 0 . 4 2 3 4 }$ </td></tr><tr><td></td><td>κ</td><td> $\mathbf { 9 8 . 9 1 9 8 { \scriptstyle \pm 0 . 0 7 3 1 } }$ </td><td> $9 6 . 1 6 8 1 { \scriptstyle \pm 0 . 2 4 9 5 }$ </td><td> $9 6 . 0 2 0 4 { \scriptstyle \pm 0 . 0 7 9 3 }$ </td><td> $9 7 . 7 0 8 4 { \scriptstyle \pm 0 . 1 3 3 7 }$ </td><td> $g 8 . 5 7 6 7 { \scriptstyle \pm 0 . I 4 2 5 }$ </td><td> $9 6 . 3 4 5 9 { \scriptstyle \pm 0 . 3 0 4 8 }$ </td><td> $9 7 . 2 6 7 4 { \scriptstyle \pm 0 . 4 5 1 0 }$ </td><td> $9 6 . 2 3 1 1 { \scriptstyle \pm 0 . 0 1 2 9 }$ </td></tr><tr><td></td><td>OA</td><td> $\mathbf { 9 9 . 5 2 4 3 { \scriptstyle \pm 0 . 0 1 7 6 } }$ </td><td> $9 6 . 8 8 4 7 { \scriptstyle \pm 0 . 1 5 5 4 }$ </td><td> $9 6 . 9 7 2 7 { \pm } 1 . 2 5 1 5$ </td><td> $9 8 . 1 6 3 2 { \pm } 0 . 2 9 8 4$ </td><td> $9 8 . 7 5 6 2 { \scriptstyle \pm 0 . 1 9 7 3 }$ </td><td> $9 7 . 3 4 6 9 { \pm } 0 . 0 5 7 6$ </td><td> $9 7 . 8 7 7 9 { \scriptstyle \pm 0 . 6 4 9 8 }$ </td><td> $9 6 . 7 2 1 9 { \scriptstyle \pm 0 . 0 5 8 7 }$ </td></tr><tr><td>TD</td><td>AA</td><td> $\mathbf { 9 9 . 5 8 5 3 \pm 0 . 0 9 1 2 }$ </td><td> $9 3 . 8 9 5 8 { \scriptstyle \pm 0 . 8 3 0 0 }$ </td><td> $9 3 . 5 0 8 1 { \scriptstyle \pm 4 . 6 3 2 0 }$ </td><td> $9 8 . 0 9 4 5 { \scriptstyle \pm 0 . 2 5 3 2 }$ </td><td> $g 8 . 6 9 6 5 { \scriptstyle \pm 0 . 1 6 7 3 }$ </td><td> $9 3 . 8 6 8 5 { \scriptstyle \pm 0 . 7 3 7 8 }$ </td><td> $9 5 . 2 8 1 7 { \pm } 1 . 0 2 5 8$ </td><td> $9 3 . 5 6 5 9 { \pm } 1 . 1 7 7 3 $ </td></tr><tr><td></td><td>κ</td><td> $\mathbf { 9 9 . 4 5 4 3 { \scriptstyle \pm 0 . 0 1 5 4 } }$ </td><td> $9 6 . 4 4 9 4 { \scriptstyle \pm 0 . 1 7 8 0 }$ </td><td> $9 6 . 5 5 0 5 { \scriptstyle \pm 1 . 4 3 1 2 }$ </td><td> $9 7 . 9 0 9 6 { \scriptstyle \pm 0 . 1 9 0 3 }$ </td><td> $9 8 . 5 8 4 3 { \scriptstyle \pm 0 . 1 4 3 7 }$ </td><td> $9 6 . 9 7 5 7 { \scriptstyle \pm 0 . 0 6 6 4 }$ </td><td> $9 7 . 5 8 3 9 { \pm } 0 . 7 4 4 1$ </td><td> $9 6 . 2 6 4 1 { \scriptstyle \pm 0 . 0 6 3 4 }$ </td></tr></table>

It is important to note that the compared methods were originally designed and evaluated under different training protocols and dataset complexities. Preliminary experiments showed that several baseline methods experienced severe performance degradation and unstable convergence when trained using the same extremel limited number of labeled samples employed by LWM on the more challenging benchmark datasets. Despite using substantially fewer labeled samples, LWM consistently achieves competitive performance across all datasets.

![](images/2e5b4bab311a399d8d9b38764ef056485e9f56539227099389ae87ed228958c8.jpg)

(a) GT  
![](images/916e75b1987665f670de56fc9a3a267dbaa2f0fb32ad8a47e847dfdf96c537d8.jpg)

![](images/29e289f1f984d584b8a3021f445a83316cb1b4db54b741441657c45e009f7e77.jpg)  
(b) LWM

![](images/9de1e1a6d8b13c9314897264db95c44d8ca026962537a1965cdef741472be4ca.jpg)

(d) DF  
![](images/e9105077f1caa411947c2be9b75a445762d1d87f1e084eafb85095c14d58bf4f.jpg)  
(e) MCMTN

(c) WF  
![](images/6d3ccd5eede91482af08331ad1c4182b4ee0015de8321ca6cd2541b2527c68b9.jpg)

![](images/5a6fe837374ea5c134fe4c1d9782cdaf1ea22629f6c40799a366016702b8c6ec.jpg)  
(g) MSST

![](images/a620b77c3a70b33008950473f6401706ecd074f4534541378cd0b4ee8ac097ec.jpg)  
(h) S2CIFT

(f) SMFAT  
![](images/f140ad7b4adc0e1cd8b1f6c2fbd8e1fbdd17326989b5b794f266e0c2409bd240.jpg)  
(i) S2CAT  
Fig. 7: Qualitative comparison on the DFC18 dataset. The proposed LWM framework exhibits improved spatial consistency and reduced boundary confusion compared with existing transformer-based methods. Error maps reveal fewer fragmented predictions and more coherent class structures, indicating that latent trajectory modeling enhances representation robustnes under spectrally similar categories.

## VIII. COMPARISON WITH STATE-OF-THE-ART

To evaluate the effectiveness of the proposed LWM framework, we compare it against several recent state-of-theart transformer-based HSIC methods, including Spectral-Guided Multiscale Feature-Aware Transformer (SMFAT) [58], WaveFormer (WF) [59], Differential Spatial-Spectral Transformer (DF) [60], Mid-Range Convolutional Modulated Transformer Network (MCMTN) [61], Multiscale spatial-spectral Supertoken Aggregation Transformer (MSST) [62], Spatial-Spectral Coupling and Interactive Fusion Transformer Network (S2CIFT) [63], and Spatial-Spectral CNN-Transformer Network (S2CAT) [64]. All methods are evaluated under identical experimental settings. Quantitative comparisons are reported in Table V, while qualitative error-map comparisons are presented in Figs. 7-11.

Quantitative Results Discussion. The quantitative results in Table V demonstrate that the proposed LWM framework consistently achieves the strongest overall performance. For the Dioni dataset, which contains substantial intra-class variability and complex spatial structures, LWM achieves an OA of 91.71%, outperforming the strongest competing method by approximately 0.7-0.8%. More importantly, the improvement in AA is considerably larger, demonstrating that the proposed framework improves classification performance not only for dominant categories but also for minority classes. A similar trend is observed on the Loukia dataset. Although several competing methods achieve competitive OA values, LWM consistently provides superior OA, AA, and κ scores.

![](images/0516ab18565ebba42750f956a61d3b75b0650982265234e25f9e30767470414e.jpg)

![](images/dce8033bb422ee1d07f1f12bc9b6f0c2499d44f5aa9536cd9a5aeba03aa221e0.jpg)

![](images/5e9dde6eac10eacf89b8c7add85e0a83ca73a03d73d1a3dd45fab5baae45f6e8.jpg)

![](images/63f0b47d06ce73dc7d13b5ceb406ee97f90c67893bd3f8ce76568425e9895835.jpg)

(a) GT  
![](images/a29a07a0c8a9ca72fc0542d5a394b69fab71aeb81c45fda663cb1b4b027ef964.jpg)

(b) LWM  
![](images/1a62e7154130bdb6218c3cc234ba79887af6c9e82c59259c21d78f21252db3f4.jpg)  
(d) DF

(c) WF  
![](images/dbc7c7b5414b7865afa24a8ab69df79437ff48148a667b3133bd6c597f7a6a2a.jpg)

![](images/d014f79f8f6d9a756e913afb59b2215b05e211078f6d730011507c63a81511a4.jpg)  
(g) MSST

(e) MCMTN  
(f) SMFAT  
![](images/7104876966c1fbd1f9995d59d3a00e8f7963917ad0f8af8289008b3410d73017.jpg)  
(h) S2CIFT

![](images/62f8dec6425bb8b13ccd02f31c4edef673121c517205a4a88a335069c8e08527.jpg)  
(i) S2CAT

Fig. 8: Qualitative comparison on the Dioni dataset.  
![](images/9194e778ca85373cea4fd63bf7c8d5838bc09a2d1be6a02004242df6c2ab781f.jpg)  
(a) GT

![](images/b604813a37527de08ac878e3b940f20a0cc660e551f3e04c68633c16d55c5268.jpg)

![](images/6d24910a20dcbb4171f6d7ca5eb92999d4a8dc88222c8d94ae47bbc8f3b9ac79.jpg)

![](images/5858304e464e402c36895daad190623d74f0451bfe69b31335cdd0cd200cb126.jpg)  
(d) DF

(b) LWM  
![](images/7a341d3b26620cc15fb479ae92c07b4b77932ff5bf632721ed300c686c6cb2d8.jpg)

(c) WF  
![](images/1f9afdba0097a7a4fd64ecaa6744f2c82fcdb4e320f87db8490ffa1ec2ee9334.jpg)

![](images/038cd3cc22b3ed2a637f0e9afa9eaded3cc27fe9de34a7593346fcfb9b8b82a1.jpg)

(e) MCMTN  
![](images/f9acd2f20faf587a71b2544e06c8a40023db31b3965ff7ad2ecfe98b4ffb89c7.jpg)  
(g) MSST

(f) SMFAT  
![](images/3feb2cff675612055e2cc21b9f771e45ed04ff69b1b9f3943fb144a7b90e31dc.jpg)  
(h) S2CIFT  
(i) S2CAT  
Fig. 9: Qualitative comparison on the Loukia dataset.

![](images/b6119a2b5d5e781825e770d857331492e9c6e87cc2a289d98a64f9047ba34b27.jpg)  
(a) GT

![](images/e17e05932eb1c05588165f69011d3954baf2ffe4a37c286ef97d376badcda276.jpg)  
(b) LWM

![](images/6e8795f16b1f5f2f99cbb2ba750f591ece6f9a52c4fa1b66d12f5cd9467e2ed0.jpg)  
(c) WF

![](images/fa4fce3419693656292125e1a286dc22113be0d9aec05119fceec3f90bb4b12f.jpg)  
(d) DF

![](images/10a208ab848cf9572f605fd6430fa61d2b1ca6d18e5201fd87b387f51861112e.jpg)  
(e) MCMTN

(f) SMFAT  
![](images/9f8f9c55adcc0ad09d3f4b6bd80d4df277d5233565f726a83c9f3bdd880d972d.jpg)  
(g) MSST

![](images/d4066d1eaf75e7b9258a92b306f23e80b9dbc1d76db580c5ef2a326aa46c5d82.jpg)  
(h) S2CIFT

![](images/33da3c46faf74e8e53f83099e50f1c8b17726cfbaaa82d3f932cf1c02635b084.jpg)  
(i) S2CAT

Fig. 10: Qualitative comparison on the HH dataset.  
![](images/1429b9f1b4f3b2ac03533dc658d97273911d1d5ccf7dee334aac09db9d4e045f.jpg)  
(a) GT

![](images/0815f58d98047142b7f6afa82f9afe9711b4056e1fd37a9fade232460e76b68f.jpg)  
(b) LWM

![](images/def4d8625dba1dcdfef63f4ed2bfc6230571b8d2054e6685eaf39cc5cb61ac4b.jpg)  
(c) WF

![](images/78438e7524c64e1dddbec53837a102ae4d0d719fe540900595e5f118dc57c213.jpg)  
(d) DF

![](images/1a5c7e7bf261c8b9fc6533664e3c052b645c7da6eddf98cb8745d51e4bf1610d.jpg)  
(e) MCMTN

![](images/ef11953b9ff26194d4c1794705343f728cda1e32fa6f4869e967dc0e34f121a5.jpg)  
Fig. 11: Qualitative comparison on the TD dataset.  
(f) SMFAT

![](images/effb43e6bc3bcd2cb515784dc5329f25abf1b67b46aee0a3520c1b347bcf6a56.jpg)  
(g) MSST

![](images/50869c2e9d3b2f551b2b03e1fd395e3e6b0f046f6d6b071dcfad065d5ff591c7.jpg)  
(h) S2CIFT

![](images/bb2e535d7ccfd21305a7e2394c21ca9ca2adbf03a3753851bdf7954150aa4579.jpg)  
(i) S2CAT

For the HH and TD datasets, which already exhibit extremely high baseline accuracies, LWM still achieves the best overall performance. On HH, the proposed framework exceeds all competing methods, demonstrating that latent trajectory reasoning remains beneficial even in near-saturated classification scenarios. On TD, LWM achieves an OA of 99.52%, indicating exceptional discrimination capability under highly structured coastal and urban environments. The relatively small standard deviations further demonstrate the stability and robustness of the proposed learning framework.

Qualitative Results Discussion. The qualitative comparisons further support the quantitative findings. Across all evaluated datasets (Figs. 7-11), the error maps generated by LWM contain noticeably fewer misclassified pixels than competing methods. The improvement is particularly evident near class boundaries, heterogeneous regions, and spectrally ambiguous areas, where traditional attention-based models frequently produce fragmented predictions. In contrast, LWM generates smoother and more spatially coherent classification maps while preserving fine structural details. Unlike conventional transformer architectures that perform classification using a single static representation, LWM augments representation learning with stochastic latent rollout reasoning. The latent trajectory provides multiple plausible future representations that are aggregated through the rollout planner, enabling the classifier to exploit structured predictive information before making a decision.

## IX. LIMITATIONS AND FAILURE ANALYSIS

Although the proposed LWM framework demonstrates consistent improvements across multiple hyperspectral benchmarks, it is important to analyze its limitations and identify scenarios where latent rollout reasoning may provide limited benefit. Understanding such failure modes is essential for assessing the practical applicability of the framework and for guiding future research on latent predictive modeling for HSI analysis. In this section, we discuss computational considerations, modeling limitations, and representative failure cases observed during experimentation.

1) Modeling Limitations: The proposed framework introduces additional computational complexity compared with conventional transformer-based classifiers. In addition to spectral, spatial, and fusion transformers, LWM performs multistep latent rollouts and trajectory aggregation. Although the rollout horizon used in this work is relatively short $( T = 3 )$ , the computational cost grows approximately linearly with the rollout length. Consequently, very long rollout horizons may provide diminishing returns while increasing memory consumption and inference latency.

Another limitation stems from the absence of explicit physical dynamics. Unlike world models used in temporal environments, the latent transitions learned by LWM do not correspond to true temporal evolution. Instead, they represent structured transformations in representation space. While this formulation is effective for classification, the learned latent trajectories should not be interpreted as physically meaningful spectral or environmental changes. Consequently, the rollout process primarily acts as a predictive regularization mechanism rather than a physical simulation model.

Furthermore, the current implementation assumes a fixed patch size and local spatial context. Extremely large-scale structures or long-range spatial dependencies extending beyond the extracted patch may not be fully represented. Although transformer attention partially alleviates this limitation, global scene-level reasoning remains constrained by the local patch formulation adopted in most HSIC benchmarks.

Finally, the effectiveness of latent rollout reasoning appears to depend on dataset complexity. As observed in the branchwise analysis, highly saturated datasets such as HH and TD exhibit extremely high agreement between the Base and Trajectory branches. In such cases, the predictive diversity between branches becomes very small, reducing the potential gains obtainable through latent rollout reasoning.

2) Failure Analysis: Detailed inspection of classification maps, attention visualizations, and error maps reveals several recurring failure patterns. First, most misclassifications occur near class boundaries where neighboring categories exhibit similar spectral signatures. Such boundary regions often contain mixed pixels whose spectral responses represent combinations of multiple materials. Although latent rollout reasoning improves contextual understanding, the ambiguity inherent in mixed pixels remains challenging.

Second, errors are frequently observed among semantically related classes that possess strong spectral overlap. In agricultural scenes, crops at similar growth stages may produce nearly indistinguishable spectral responses, while in urban environments different materials can exhibit highly correlated reflectance patterns. These cases provide limited discriminative evidence even after latent trajectory aggregation.

Third, the latent rollout mechanism occasionally propagates uncertainty from an incorrect initial representation. Because rollout generation begins from the environment state $\mathbf { z } _ { 0 } .$ , severe errors in the initial representation may lead to trajectories that remain consistently biased. This behavior is most likely to occur in highly corrupted regions, rare classes with limited training samples, or samples located near complex class transitions.

The branch correction analysis provides additional insight into these failure modes. Datasets such as Dioni and Loukia exhibit relatively large JSD and positive NG values, indicating that the trajectory branch successfully corrects a substantial number of Base-branch errors. In contrast, HH and TD show extremely high agreement rates and very small divergence values, suggesting that both branches converge to nearly identical predictions. In such scenarios, the latent rollout branch contributes limited corrective information because the underlying classification problem is already nearly solved by the static representation. Despite these limitations, the observed failure cases are largely concentrated in ambiguous boundary regions, spectrally overlapping categories, and highly challenging samples.

## X. CONCLUSIONS AND FUTURE RESEARCH DIRECTIONS

This paper introduced Stochastic Latent World Modeling (LWM), a trajectory-aware representation learning framework for hyperspectral image classification (HSIC). Unlike conventional approaches that rely on static spatial-spectral representations, the proposed framework models structured latent evolution through stochastic rollout reasoning in representation space. By constructing an environment state from fused spatial-spectral transformer features and propagating it through a stochastic latent transition process, LWM enables the classifier to exploit predictive latent structure beyond the information available in a single forward-pass representation.

Extensive experiments conducted on diverse hyperspectral benchmarks demonstrated the effectiveness of the proposed approach. The results showed that LWM consistently achieves superior classification performance compared with recent state-of-the-art methods while also improving calibration quality, robustness to spectral corruption and noise. Branch-wise analysis further confirmed that latent rollout reasoning contributes complementary predictive information beyond static transformer features. In addition, ablation studies verified the importance of each component of the framework.

Several research directions remain open for further investigation, as already discussed in Section IX. First, the current rollout mechanism operates at a single spatial scale; extending LWM to hierarchical and multi-scale representations may further improve the modeling of complex scene structures. Second, incorporating adaptive rollout horizons and uncertainty-aware planning strategies could allow the model to dynamically adjust its latent reasoning process according to scene complexity. Third, integrating LWM with semisupervised, self-supervised, and foundation-model paradigms may improve scalability under extremely limited supervision. Finally, extending the proposed framework to hyperspectral change detection, multimodal remote sensing fusion, domain adaptation, and spatiotemporal Earth observation analysis represents an exciting direction for future research.

## REFERENCES

[1] D. Hong, C. Li, N. Yokoya, B. Zhang, X. Jia, A. Plaza, P. Gamba, J. A. Benediktsson, and J. Chanussot, “Hyperspectral imaging,” Nature Reviews Methods Primers, vol. 6, no. 1, p. 19, 2026.

[2] W. Wang, L.-J. Deng, Q. Cao, and G. Vivone, “A general image fusion approach exploiting gradient transfer learning and fusion rule unfolding,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 6, pp. 6347–6364, 2026.

[3] J. Hou, Z. Zhu, J. Hou, H. Liu, H. Zeng, and D. Meng, “Deep diversityenhanced feature representation of hyperspectral images,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 8123–8138, 2024.

[4] D. Hong, N. Yokoya, J. Chanussot, J. Xu, and X. X. Zhu, “Joint and progressive subspace analysis (jpsa) with spatial–spectral manifold alignment for semisupervised hyperspectral dimensionality reduction,” IEEE Transactions on Cybernetics, vol. 51, no. 7, pp. 3602–3615, 2020.

[5] M. Ahmad, M. Mazzara, S. Distefano, A. Mehmood Khan, M. Hassaan Farooq Butt, and D. Hong, “PolicyMamba: Localized policy attention with state space model for land cover classification,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 10, pp. 17 814– 17 825, 2025.

[6] M. Ahmad, S. Distefano, A. M. Khan, M. Mazzara, C. Li, H. Li, J. Aryal, Y. Ding, G. Vivone, and D. Hong, “A comprehensive survey for hyperspectral image classification: The evolution from conventional to transformers and Mamba models,” Neurocomputing, vol. 644, p. 130428, 2025. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0925231225011002

[7] L. Scheibenreif, M. Mommert, and D. Borth, “Masked vision transformers for hyperspectral image classification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2023, pp. 2166–2176.

[8] X. Li, F. Xu, J. Li, Y. Su, L. Li, X. Lyu, Z. Xu, and A. Kaup, “Frequency domain-enhanced spectral-spatial fusion transformer for semantic segmentation of remote sensing images,” Information Fusion, vol. 132, p. 104248, 2026. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S1566253526001272

[9] M. Ahmad, U. Ghous, D. Hong, A. M. Khan, J. Yao, S. Wang, and J. Chanussot, “A disjoint samples-based 3D-CNN with active transfer learning for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–16, 2022.

[10] Q. Liu, Z. Zhan, J. Wang, C. Wang, and S. J. Qin, “Toward lightweight dynamic convolutional neural network modeling for soft sensors,” IEEE Transactions on Cybernetics, vol. 56, no. 4, pp. 1958–1969, 2026.

[11] G. Tong, S. Liu, Y. Lv, H. Pei, and F.-L. Fan, “A survey on medical image compression: From traditional to learning-based approaches,” IEEE Transactions on Radiation and Plasma Medical Sciences, pp. 1–1, 2026.

[12] M. Lokumarambage, T. Sivalingam, F. Dong, N. Rajatheva, and A. Fernando, “Latent prediction-based generative semantic communication for video transmission in wireless networks,” IEEE Open Journal of the Communications Society, vol. 7, pp. 3974–3986, 2026.

[13] J. Tang, W. Cao, B. Zhang, C. Luo, Y. Liu, and M. Nießner, “Motion2VecSets: Non-rigid shape reconstruction and tracking with 4d latent set diffusion,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–18, 2026.

[14] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=YicbFdNTTy

[15] C. Li, B. Zhang, D. Hong, J. Zhou, G. Vivone, S. Li, and J. Chanussot, “Casformer: Cascaded transformers for fusion-aware computational hyperspectral imaging,” Information Fusion, vol. 108, p. 102408, 2024.

[16] M. Ahmad, H. Luqman, and M. Mazzara, “Relational graph–guided selective state-space model for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, pp. 1–1, 2026.

[17] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017, pp. 5998–6008.

[18] J. Zhang, Y. Song, Z. Liu, W. Yan, Z. Wang, and X. Wang, “Wavelet spectral–spatial Mamba network for hyperspectral image classification,” IEEE Transactions on Circuits and Systems for Video Technology, pp. 1–1, 2026.

[19] A. Waseem, P. Ruiu, S. Nixon, A. Lagorio, and M. Tistarelli, “Self-attention as the backbone: A survey on vision transformers,” Computer Vision and Image Understanding, vol. 269, p. 104791, 2026. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S107731422600158X

[20] M. Ahmad and M. Mazzara, “Iformer: Irregular spatial–spectral transformer for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 5 521 012–5 521 012, 2026.

[21] C. Li, D. Hong, B. Zhang, and J. Chanussot, “Im2state: Image-to-state Mamba for hyperspectral and multispectral image fusion,” Information Fusion, p. 104385, 2026.

[22] R. Wang and R. Yu, “Physics-guided deep learning for dynamical systems: A survey,” ACM Comput. Surv., vol. 58, no. 5, 2025. [Online]. Available: https://doi.org/10.1145/3766887

[23] C. Li, D. Hong, B. Zhang, Z. Pan, N. Yokoya, and J. Chanussot, “Mambax: Image super-resolution with state predictive control,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[24] V. Ntouskos, F. Pirri, M. Pizzoli, A. Sinha, and B. Cafaro, “Saliency prediction in the coherence theory of attention,” Biologically Inspired Cognitive Architectures, vol. 5, pp. 10–28, 2013, extended versions of selected papers from the Third Annual Meeting

of the BICA Society (BICA 2012). [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S2212683X13000479

[25] D. Hong, B. Zhang, X. Li, Y. Li, C. Li, J. Yao, N. Yokoya, H. Li, P. Ghamisi, X. Jia et al., “Spectralgpt: Spectral remote sensing foundation model,” IEEE transactions on pattern analysis and machine intelligence, vol. 46, no. 8, pp. 5227–5244, 2024.

[26] L. Zhao, W. Luo, Q. Liao, S. Chen, and J. Wu, “Hyperspectral image classification with contrastive self-supervised learning under limited labeled samples,” IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1–5, 2022.

[27] H. Xie, X. Zhao, J. Yin, J. Peng, Q. Wang, and G. Vivone, “Crossarchitecture contrastive learning for few-shot hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 1–15, 2026.

[28] H. Qin, S. Wang, Y. Li, W. Xie, K. Jiang, and K. Cao, “Hyperspectral target detection based on generative self-supervised learning with wavelet transform,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–15, 2025.

[29] M. Ahmad, M. Mazzara, S. Distefano, and A. M. Khan, “Byte latent Mamba with state space and knowledge distillation for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–15, 2025.

[30] H. Qin, S. Wang, Y. Li, W. Xie, K. Jiang, and K. Cao, “A signatureconstrained two-stage framework for hyperspectral target detection based on generative self-supervised learning,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–17, 2025.

[31] M. Ahmad, M. Mazzara, S. Distefano, A. M. Khan, M. H. F. Butt, M. Usama, and D. Hong, “GraphMamba: Graph tokenization Mamba for hyperspectral image classification,” IEEE Transactions on Emerging Topics in Computing, vol. 13, no. 4, pp. 1510–1521, 2025.

[32] J. Wang, W. Song, H. Chen, J. Ren, and H. Zhao, “FusDreamer: Labelefficient remote sensing world model for multimodal data classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1– 14, 2025.

[33] D. Ha and J. Schmidhuber, “World models,” 2018. [Online]. Available: https://zenodo.org/record/1207631

[34] Y. Matsuo, Y. LeCun, M. Sahani, D. Precup, D. Silver, M. Sugiyama, E. Uchibe, and J. Morimoto, “Deep learning, reinforcement learning, and world models,” Neural Networks, vol. 152, pp. 267–275, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0893608022001150

[35] M. Ahmad, A. Maslovskaya, and M. Mazzara, “WorldMamba: Worldstate representation learning for hyperspectral image classification,” IEEE Geoscience and Remote Sensing Letters, vol. 23, pp. 5 506 505– 5 506 505, 2026.

[36] D. Hong, C. Li, X. Li, G. Camps-Valls, and J. Chanussot, “Foundation models in remote sensing: Evolving from unimodality to multimodality,” IEEE Geoscience and Remote Sensing Magazine, vol. 14, no. 2, pp. 10– 35, 2026.

[37] N. Chung, T. Hanyu, T. Nguyen, H. Le, F. Bumgarner, D. M. H. Nguyen, K. Vo, K. Yamazaki, C. Rainwater, T. Kieu et al., “Rethinking progression of memory state in robotic manipulation: An objectcentric perspective,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 5, pp. 3407–3415, 2026. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/37337

[38] F. Feng, P. Lippe, and S. Magliacane, “Learning interactive world model for object-centric reinforcement learning,” in Advances in Neural Information Processing Systems, D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, Eds., vol. 38. Curran Associates, Inc., 2025, pp. 89 827–89 862. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2025/file/8187faaf6759ef6d4e93293339bc656e-Paper-Conference.pdf

[39] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” in 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014. [Online]. Available: http://arxiv.org/abs/1312.6114

[40] J. Wu, S. Yin, N. Feng, and M. Long, “RLVR-World: Training world models with reinforcement learning,” in Advances in Neural Information Processing Systems, D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, Eds., vol. 38. Curran Associates, Inc., 2025, pp. 125 312–125 350. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2025/file/b63a24a1832bd14fa945c71f535c0095-Paper-Conference.pdf

[41] H. He, Y. Zhang, L. Lin, Z. Xu, and L. Pan, “Pre-trained video generative models as world simulators,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 6, pp. 4645–4653, 2026. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/42465

[42] D. Ha and J. Schmidhuber, “World models,” 2018. [Online]. Available: https://zenodo.org/record/1207631

[43] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Learning latent dynamics for planning from pixels,” in International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 2555–2565. [Online]. Available: https://proceedings.mlr.press/v97/hafner19a.html

[44] J. Chung, K. Kastner, L. Dinh, K. Goel, A. C. Courville, and Y. Bengio, “A recurrent latent variable model for sequential data,” in Advances in Neural Information Processing Systems, vol. 28, 2015.

[45] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, “Empirical evaluation of gated recurrent neural networks on sequence modeling,” CoRR, vol. abs/1412.3555, 2014. [Online]. Available: http://arxiv.org/abs/1412.3555

[46] J. Ding, Y. Zhang, Y. Shang, Y. Zhang, Z. Zong, J. Feng, Y. Yuan, H. Su, N. Li, N. Sukiennik, F. Xu, and Y. Li, “Understanding world or predicting future? a comprehensive survey of world models,” ACM Comput. Surv., vol. 58, no. 3, 2025. [Online]. Available: https://doi.org/10.1145/3746449

[47] M. Abukmeil, S. Ferrari, A. Genovese, V. Piuri, and F. Scotti, “A survey of unsupervised generative models for exploratory data analysis and representation learning,” ACM Comput. Surv., vol. 54, no. 5, 2021. [Online]. Available: https://doi.org/10.1145/3450963

[48] T. Saanum, P. Dayan, and E. Schulz, “Simplifying latent dynamics with softly state-invariant world models,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 38 355–38 382. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2024/file/43ba0466af2b1ac76aa85d8fbec714e3-Paper-Conference.pdf

[49] H. Fan, L. Bai, B. Fei, Y. Xiao, K. Chen, Y. Liu, Y. Qu, F. Ling, and P. Gentine, “Physically consistent global atmospheric data assimilation with machine learning in latent space,” Science Advances, vol. 12, no. 1, p. eaea4248, 2026. [Online]. Available: https://www.science.org/doi/abs/10.1126/sciadv.aea4248

[50] J. Chung, K. Kastner, L. Dinh, K. Goel, A. Courville, and Y. Bengio, “A recurrent latent variable model for sequential data,” in Proceedings of the 29th International Conference on Neural Information Processing Systems - Volume 2, ser. NIPS’15. Cambridge, MA, USA: MIT Press, 2015, p. 2980–2988.

[51] J.-B. Grill, F. Strub, F. Altche, C. Tallec, P. H. Richemond,´ E. Buchatskaya, C. Doersch, B. A. Pires, Z. D. Guo, M. G. Azar, B. Piot, K. Kavukcuoglu, R. Munos, and M. Valko, “Bootstrap your own latent a new approach to self-supervised learning,” in Proceedings of the 34th International Conference on Neural Information Processing Systems, ser. NIPS ’20. Red Hook, NY, USA: Curran Associates Inc., 2020.

[52] Y. Zhong, X. Hu, C. Luo, X. Wang, J. Zhao, and L. Zhang, “WHU-Hi: UAV-borne hyperspectral with high spatial resolution (H2) benchmark datasets and classifier for precise crop identification based on deep convolutional neural network with CRF,” Remote Sensing of Environment, vol. 250, p. 112012, 2020.

[53] H. Fu, G. Sun, L. Zhang, A. Zhang, J. Ren, X. Jia, and F. Li, “Three-dimensional singular spectrum analysis for precise land cover classification from uav-borne hyperspectral benchmark datasets,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 203, pp. 115– 134, 2023. [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S0924271623001946

[54] Y. Xu, B. Du, L. Zhang, D. Cerra, M. Pato, E. Carmona, S. Prasad, N. Yokoya, R. Hansch, and B. Le Saux, “Advanced multi-sensor optical¨ remote sensing for urban land use and land cover classification: Outcome of the 2018 ieee grss data fusion contest,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 12, no. 6, pp. 1709–1724, 2019.

[55] K. Karantzalos, C. Karakizi, Z. Kandylakis, and G. Antoniou, “Hyrank hyperspectral satellite dataset i,” Apr. 2018. [Online]. Available: https://doi.org/10.5281/zenodo.1222202

[56] S. Gruber and F. Buettner, “Better uncertainty calibration via proper scores for classification and beyond,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 8618–8632. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2022/file/3915a87ddac8e8c2f23dbabbcee6eec9-Paper-Conference.pdf

[57] R. Hebbalaguppe, J. Prakash, N. Madan, and C. Arora, “A stitch in time saves nine: A train-time regularizing loss for improved neural network calibration,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 16 060–16 069.

[58] Z. Shu, K. Zeng, Y. Wang, S. Tang, Z. Yu, and L. Xiao, “Spectralguided multiscale feature-aware transformer for hyperspectral image classification,” IEEE Transactions on Neural Networks and Learning Systems, pp. 1–15, 2025.

[59] M. Ahmad, U. Ghous, M. Usama, and M. Mazzara, “WaveFormer: Spectral–spatial wavelet transformer for hyperspectral image classification,” IEEE Geoscience and Remote Sensing Letters, vol. 21, pp. 1–5, 2024.

[60] M. Ahmad, M. Mazzara, S. Distefano, A. M. Khan, and S. L. Ullo, “DiffFormer: A differential spatial-spectral transformer for hyperspectral image classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 10 419–10 428, 2025.

[61] Y. Wang, M. Tai, Z. Liu, Z. Shu, F. Xiong, and Z. Yu, “Mid-range convolutional modulated transformer network for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–15, 2025.

[62] H. Yao, P. Zhang, C. Zheng, L. Zhao, Y. Jin, M. Tian, Y. Hou, J. Li, and Q. Qiu, “MSSTFormer: Multiscale spectral–spatial supertoken aggregation transformer for hyperspectral image classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 19, pp. 5248–5266, 2026.

[63] H. Yang, S. Cheng, A. Du, and J. Zhang, “S2CIFTNet: Spatial–spectral coupling and interactive fusion transformer network for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 1–18, 2026.

[64] G. Wan, D. Liu, Y. Liu, T. Yang, and Y. Guo, “S2Cat: Spatial–spectralbased cnn and transformer network for hyperspectral image classification,” IEEE Access, vol. 14, pp. 17 434–17 453, 2026.