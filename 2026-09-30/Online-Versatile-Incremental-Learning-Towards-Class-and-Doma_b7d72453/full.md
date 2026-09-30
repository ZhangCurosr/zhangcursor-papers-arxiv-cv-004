# Online Versatile Incremental Learning: Towards Class and Domain-Agnostic Adaptation at Any Time

Jae-Ho Lee<sup>1</sup> , Min-Yeong Park<sup>2\*</sup> , Jun-Yeong Moon<sup>1</sup> , Jung Uk Kim<sup>2</sup>† , and Gyeong-Moon Park<sup>1</sup>†

<sup>1</sup> Korea University, Seoul, Republic of Korea   
{jaeho-lee, moonjunyyy, gm-park}@korea.ac.kr   
2 Kyung Hee University, Yongin, Republic of Korea {pmy0792, ju.kim}@khu.ac.kr

Abstract. Continual learning enables vision systems to adapt to everchanging data distributions. Despite significant advances, existing approaches fail to capture continuous and concurrent shifts in classes and domains, a critical capability for real-world deployment. This work introduces Online VIL (Online Versatile Incremental Learning), a novel scenario where class concepts and visual domains evolve simultaneously online without explicit boundaries. To better adapt to the challenges of such dynamic environments that more closely resemble real-world conditions, we propose a novel framework TopFlow, Topology preservation with Flow matching representation that contains two complementary mechanisms: Domain-agnostic Flow Matching (DFM) and Global Topology Preservation (GTP). DFM guides the model to have domain-agnostic representations by integrating the geodesic flow kernel into contrastive learning. In contrast, GTP maintains the global structure of the feature space without explicitly storing past examples. Our extensive experiments demonstrate that TopFlow efectively addresses the limitations of existing methods within the Online VIL scenario, achieving state-of-the-art performance in challenging Online VIL. The proposed methods suggest potential directions for building continual learning systems in realistic dynamic environments. Our implementation code is available at https://github.com/KU-VGI/Online-VIL.

Keywords: Online learning · Incremental learning · Real-world scenario

## 1 Introduction

Continual Learning (CL) [11,15,24,28,35–37,40] has gained increasing attention as deep learning moves closer to the real world, where data distributions evolve. A key challenge is catastrophic forgetting, where adapting to new knowledge disrupts previous knowledge. To mitigate this, prior research has introduced distinct paradigms such as Class Incremental Learning (CIL) and Domain Incremental Learning (DIL), targeting a specific form of non-stationarity. More recently, Online Continual Learning (OCL) [10, 33, 38] has been proposed to handle streaming data under memory and single-pass constraints, including scenarios with blurry task boundaries [1, 12, 23]. However, most works still rely on CIL or DIL, limiting their ability to capture the full complexity of real-world dynamics.

![](images/51fa1b0a0f6111ee5f58afce85c8e7d05844120e3cd684e8f1516e9bff5a7195.jpg)  
<sup>T</sup>1(a) CIL.

![](images/87c9b83605a46055405eb54f6e716465f0a04b3174c76580780b5d6b53872a59.jpg)  
(b) DIL.

![](images/e850573d397f0c8619a19e0e9d3ce353ccb6c61331a295b99240d6345780abb1.jpg)  
<sub>T</sub><sup>T</sup>56(c) VIL.

Class  
![](images/e8c96f7f642c324a6d583d8a43106c845dc41556f022eaa2fc96c0fa70bbb889.jpg)  
(d) Online VIL (Ours).  
<sub>T</sub> T7<sup>6</sup>Fig. 1: Conceptual comparison of (a) Class Incremental Learning (CIL), (b) Domain <sup>7</sup>Incremental Learning (DIL), (c) Versatile Incremental Learning (VIL), and (d) Online Versatile Incremental Learning (Online VIL, Ours). Red lines indicate the scopes observable at once, while black dashed lines depict transitions across time.

A recently introduced Versatile Incremental Learning (VIL) [25] suggests a more realistic scenario where new tasks have a broader chance to evolve in both directions of the classes and domains, without prior knowledge. While VIL marks a step toward realism, it assumes discrete increments that provide implicit structural information about distribution shifts. In contrast, real-world environments exhibit continuous transitions without boundaries, ofering no such organizational cues. Moreover, environmental change in reality is continuous and unpredictable, due to locally constrained information and multi-factor interactions. For instance, an autonomous driving system may suddenly face both a new object and a shift in weather or city conditions, requiring immediate (online) adaptation without knowing whether it concerns classes, domains, or both.

To this end, we introduce Online Versatile Incremental Learning (Online VIL), a new scenario that captures both the unpredictable heterogeneous shifts in an online manner. Online VIL is distinguished by reflecting the evolution of the natural information stream, characterized by unpredictable, gradual, and heterogeneous shifts that occur along multiple evolutionary trajectories. As shown in Figure 1, Online VIL allows flexible transitions across class and domain dimensions; each sequence presents distinct challenges without heuristic patterns. Consequently, Online VIL enables faithful evaluation of CL models and provides a foundation for systems that operate under real-world dynamics.

In the Online VIL scenario, adaptation to chaotic shifts in classes and domains without explicit access to prior inputs is crucial for distinguishing classdiscriminative knowledge from domain-specific knowledge. Otherwise, models rely on the spurious features of current distributions and tend to exhibit rapid forgetting and a lack of generalization. Through systematic layer-wise feature analysis of pre-trained Vision Transformers, we observe that early layers predominantly capture domain-specific patterns (e.g., texture, lighting), while deeper layers encode class-discriminative features and more abstract semantic representations. This motivates a novel Domain-agnostic Flow Matching (DFM) technique, which aligns features with the intrinsic geometry of pre-trained knowledge through reconceptualized geodesic flow kernel [6] while mitigating domain-specific shifts.

In addition to class and domain-agnostic alignment, preserving the global topology of learned features is essential for maintaining semantic continuity across evolving tasks. Conventional objectives often fail to maintain the structural relationships of feature space because they shrink the occupation of missing classes in the feature space. To address this, we propose Global Topology Preservation (GTP), which preserves invariant geometric configurations in feature space, enabling robust knowledge retention without requiring complete class coverage.

Integrating these components, we introduce Topology preservation with Flow matching representation (TopFlow), a novel framework designed for Online VIL. With recognition of DFM and GTP regularization for geometry, TopFlow ensures robust adaptation to unpredictable shifts. To evaluate its efectiveness, we conduct extensive experiments in Online VIL and observe that TopFlow consistently outperforms existing state-of-the-art methods across multiple benchmarks. Our contributions are summarized as follows:

– We introduce Online VIL, a realistic scenario for evaluating continual learning where class and domain distributions evolve continuously and jointly with ambiguous task boundaries.

– We reveal a novel role of the pre-trained ViT layer that encodes class and domain knowledge. Leveraging this insight, we propose Domain-agnostic Flow Matching (DFM) to learn domain-agnostic representations by integrating the geodesic flow kernel into contrastive learning.

We propose Global Topology Preservation (GTP), a mechanism for maintaining knowledge representations using feature topologies without explicit memory of previous inputs.

– We demonstrate that the proposed TopFlow significantly outperforms existing state-of-the-art methods through comprehensive experiments in our challenging Online VIL scenario.

## 2 Related Work

Online Continual Learning. Online Continual Learning (OCL) [3, 8] has emerged as a pragmatic paradigm that reflects the challenges of real-world settings, where data arrive as continuous streams and models must operate under minimal batch sizes, single-pass constraints, and strict computational and memory limitations. Traditionally, OCL methods rely on replay bufers [22, 26] to store a small subset of past data, thereby mitigating catastrophic forgetting while learning new tasks. Recent advances explore leveraging prototypes [38], replay-free strategies [39], and pre-trained models with prompt [23]. While efective in class or domain increments, this dependence on memory limits scalability and realism. Methods addressing more realistic scenarios with blurry or ambiguous task boundaries have emerged [1, 12]. However, the replay methods require growing memory in proportion to task diversity. Also, blurry boundary methods still assume either class-only or domain-only shifts, failing to capture the heterogeneous evolution of a realistic data stream. Therefore, we propose Online Versatile Incremental Learning (Online VIL). This scenario exposes models to unpredictable shifts in both classes and domains while enforcing online constraints that limit memory and multi-pass access.

Geodesic Flow Kernel. The Geodesic Flow Kernel (GFK) [6, 7] has been widely used in unsupervised domain adaptation to align feature distributions between a predefined source and target domain. Conventional applications approximate the geodesic on the Grassmannian manifold between two static domains, which requires that the data of the source and target domains are fully available and static. In contrast, Online VIL presents unique challenges: domain shifts occur continuously and unpredictably, and data arrive sequentially, making it impossible to estimate the flow ofline. To address these challenges, we reformulate the approach from GFK and design a novel Domain-Agnostic Flow Matching (DFM). Unlike traditional geometry estimation in feature space, DFM is designed for sequentially arriving data and evolving domains without relying on holistic data access. Through both empirical and theoretical analysis of feature geometry, DFM enables online alignment on feature geometry in dynamic conditions, avoiding computationally expensive higher-order manifold computations. This design fundamentally extends the applicability and purpose of GFK, enabling robust domain-agnostic feature matching in the OCL.

Feature Topology. Several works have leveraged the topology of feature space to mitigate catastrophic forgetting in sequential learning. [30] employs elastic Hebbian graphs to preserve neighborhood relationships during incremental updates, while [31] uses self-organizing maps to identify representative feature points and restrict their displacement. More recent approaches, such as [19] and [34], maintain pair-wise instance similarity or local topological relations by decomposing the global structure, further reducing forgetting across tasks. However, these methods are limited to ofline CL, requiring full access to past samples to construct the feature topology. Furthermore, existing topology preservation methods rely on complete semantic information across all classes. In online scenarios where only partial class information is available at each time step, these methods sufer from incomplete topology construction, resulting in suboptimal feature space organization. In contrast, our work proposes a Global Topology Preservation (GTP) strategy that maintains structural knowledge without requiring the storage of previous data.

## 3 Method

## 3.1 Problem Setup: Online Versatile Incremental Learning

In real-world scenarios, data distributions gradually evolve, often exhibiting significant variability across multiple dimensions such as spatial domains, semantic classes, and temporal shifts. We propose a novel scenario termed Online Versatile Incremental Learning (Online VIL), which simulates these properties by constructing a continuous data stream with stochastically varying distributions. Given a dataset with n domains and $n _ { \mathcal { C } }$ classes, we define the product category space as follows:

$$
\mathcal { X } = \bigcup _ { i = 1 } ^ { n _ { \mathcal { D } } } \mathcal { D } _ { i } , \mathit { \Pi } _ { i } ^ { n _ { \mathcal { C } } } \bigcup _ { j = 1 } ^ { n _ { \mathcal { D } } } \mathcal { C } _ { j } , \mathit { \Pi } _ { i = 1 } ^ { n _ { \mathcal { D } } } \bigcup _ { j = 1 } ^ { n _ { \mathcal { C } } } K _ { i , j } , \quad \mathrm { w h e r e } \quad \mathcal { K } _ { i , j } = \mathcal { D } _ { i } \cap \mathcal { C } _ { j } ,\tag{1}
$$

where $\mathcal { D } _ { i }$ is a set of samples in a domain i and $\mathcal { C } _ { j }$ is in a class $j ,$ and $\boldsymbol { \mathcal { K } } _ { i , j }$ is samples that belong to domain i and class $j .$ . Machine learning typically assumes the data $\mathcal { X }$ as a lower-dimensional manifold embedded in a high-dimensional space [2]. In this context, we can conceptualize the data stream as a trajectory through the joint domain-class space, where each timestep t provides an observation window $\nu _ { t } \subset \mathcal { X }$ which is a disjoint open neighborhood of data . This geometric interpretation naturally leads to our manifold-based approach in the subsequent technical development.

Inspired by Si-Blurry [23], we divide categories into disjoint sets $( K _ { \mathrm { d i s j o i n t } } )$ with clear distributional boundaries and blurry sets $( \kappa _ { \mathrm { b l u r r y } } )$ with deliberately abstracted distributions.

Algorithm A.1 in Appendix details our task construction process, which proceeds in six main steps: (i) category partitioning to establish variable distributional clarity, (ii) sample extraction to create diverse distributional patterns, (iii) sample redistribution to simulate realistic category overlap, (iv) randomized task assignment to eliminate artificial task boundaries, (v) task construction with varying characteristics, and (vi) batch generation with constrained visibility windows. The Online VIL scenario distinguishes itself from traditional CL scenarios through three key characteristics:

1. Locally Limited Visibility. At any time step, the model observes only a small fraction in both number of samples $\left( \left| \mathcal { V } _ { t } \right| \ \ll \ \left| \mathcal { D } \right| \right)$ and categories $( \exists K _ { i , j } : x \in K _ { i , j } \land x \notin \mathcal { V } _ { t } )$ , reflecting real-world constraints on data accessibility, combines both spatial and temporal restrictions.

2. Continuous Smooth Variation. The variation is smooth and continuous over time, making it hard to distinguish the distribution shift without a global context.

3. Dynamic Distribution. Each category appears and disappears gradually. The variance across spatial (domain appearances), semantic (class properties), and temporal (distribution evolution) dimensions requires simultaneous adaptation to new patterns and retention of previously acquired knowledge.

This formulation demands adaptation mechanisms that extract meaningful patterns despite high variance and constrained observability. Traditional CL methods, which assume either domain stability or class stability, fail to resemble these fundamental challenges in real-world perception systems.

![](images/44b986dfa1c552d97c04f717fcbd3fe91e0f8e76c2c13976993cd3c1f7e76685.jpg)

![](images/2a322d2ff7bd917e3745beff4b42fac7f2e1d34c5ed53eb58b7972ea7c319fe6.jpg)

(a) t-SNE visualization of features from pre-trained ViT.  
![](images/a3a40554e471b338dccc7b71c428fd7770a036892867013f4fe03726fde47ff0.jpg)  
(c) Linear Probing results.  
Fig. 2: Analysis of layer-wise knowledge on pre-trained ViT for data with changing distributions (CORe50). (a) t-SNE visualization of features from certain layers of the pre-trained ViT. The color indicates the domain. (b) The t-SNE visualization of the last layer with class-wise color. (c) Accuracy of linear probing for class and domain classification from diferent layers.

## 3.2 Domain-agnostic Flow Matching

The Online VIL scenario, with limited visibility, multi-dimensional variance, and dynamic distribution, poses challenges beyond traditional CL. While the original VIL work measured feature similarity, it did not explicitly examine how domain and class signals are encoded across the backbone under noisy and concurrent class/domain shifts. This matters because the one-pass constraint and the lack of rehearsal make representations highly sensitive to the local batch composition, and thus prone to overfitting transient domain cues. To better understand this, we analyze the representations of frozen pre-trained Vision Transformers, which have become the standard backbone in recent CL research [5, 28, 35–37]. Although not trained under Online VIL, a frozen backbone provides a stable proxy, reflecting common practice in CL of freezing the ViT and updating only a small number of parameters (e.g., prompts). Its hierarchical representations largely dictate how domain-specific variations and class-level semantics interact. Motivated by this, we conduct a layer-wise analysis, expecting deeper layers to align more closely with class semantics, as their outputs directly drive the classification head.

We perform a t-SNE study and linear probing to investigate this. Our t-SNE visualizations (Figures 2a, 2b) show that intermediate features group by domain, while final features group by class. Linear probing (Figure 2c) further confirms stronger domain discrimination in intermediate layers and stronger class discrimination in the final layer. These findings reveal a structural representation gap in Online VIL, motivating our proposed Domain-agnostic Flow Matching (DFM): a geometry-informed contrastive loss that pulls the adaptable intermedi-<sup>1 1</sup>ate representation toward the semantic geometry of the final layer while pushing it away from domain-biased directions.

![](images/c5c47da404125c9574dca7a9383c12212ba94517682ad48d5825cd7cc1d6a122.jpg)  
Fig. 3: Architecture overview. TopFlow comprises two components: Domain-agnostic Flow Matching (DFM) and Global Topology Preservation (GTP). DFM promotes domain-agnostic representation accumulation, while GTP preserves the global feature topology.

From Geodesic-Flow Intuition to Layer-Wise Comparison Geometry. The Geodesic Flow Kernel (GFK) [6] suggests that features from a distribution can be represented by a subspace on the Grassmann manifold, and similarity <sup>1 1</sup>can be computed through a comparison geometry defined over such subspaces. In Online VIL, however, explicitly estimating inter-domain subspaces is often ill-posed due to continuous, mixed, and unlabeled domain evolution. Instead, we leverage the empirical layer-wise gap in Fig. 2 and treat layers as the unit of geometry: intermediate features are typically more domain-sensitive, while final features are more class-semantic. Accordingly, we extract a batch-wise common comparison geometry between intermediate and final representations, and use it to normalize similarities in a contrastive objective. Residual-layer dynamics and local approximation. For convenience, we consider architectures with matched feature dimensionality across layers, such as ViTs [4]. With a cascade of functions with residual connections:

$$
\begin{array} { r } { h _ { 0 } = x , \quad h _ { n } = f _ { n } ( h _ { n - 1 } ) + h _ { n - 1 } , \quad n = 1 , 2 , \ldots , l , } \end{array}\tag{2}
$$

where $f _ { i } : \mathbb { R } ^ { b \times d }  \mathbb { R } ^ { b \times d }$ is a function of the i-th layer, and $\boldsymbol { h } _ { l } \in \mathbb { R } ^ { b \times d }$ is the last layer feature with batch size b and feature dimension d. With the local neighborhood $\nu _ { t }$ on a Riemannian manifold as mentioned in Section 3.1, we can take a first-order approximation with the infinitesimal variation of feature $h _ { l } ,$ as detailed in Equation A.3 and A.5. Then, the inner product between $h _ { n }$ and $h _ { l }$

can be written as

$$
\langle \pmb { h } _ { n } , \pmb { h } _ { l } \rangle = \int _ { X } \pmb { h } _ { n } ^ { T } \pmb { h } _ { l } d \pmb { x } = \mathbb { E } _ { X } \left[ ( \bar { \pmb { h } } _ { n } + \delta \pmb { h } _ { n } ) ^ { T } ( \bar { \pmb { h } } _ { l } + \delta \pmb { h } _ { l } ) \right] ,\tag{3}
$$

where $\bar { \pmb { h } } _ { n } = \mathbb { E } _ { { \pmb { x } } \in \mathcal { V } _ { t } } [ { \pmb { h } } _ { n } ]$ denotes the local mean feature over the neighborhood $\nu _ { t }$ and $\delta { \pmb h } _ { n } = { \pmb h } _ { n } - { \bar { \pmb h } } _ { n }$ is the corresponding local deviation.

Common Subspace Extraction and the Role of $U .$ . To enable a geometryaware comparison, we extract a common subspace from the combined space of $h _ { n }$ and $h _ { l }$ by stacking $H = \left[ \pmb { h } _ { n } ^ { T } \pmb { h } _ { l } ^ { T } \right] ^ { T } = U \Sigma V ^ { T }$ . As detailed in Equation A.3, projecting onto the common subspace $U$ provides an orthogonal basis that summarizes correlated directions shared across layers within the local neighborhood. Based on observation in Figure 2, we interpret the projected components as:

$- \ \delta h _ { n } ^ { T } \delta h _ { n }$ represents the residual component $\left( \pmb { u } _ { \mathrm { r e s i d u a l } } \right)$ , dominated by layerlocal (often domain-sensitive) variation;

$- \delta h _ { n } ^ { T } \delta h _ { l }$ and $\delta h _ { l } ^ { T } \delta h _ { n }$ represent push-forward induced by transportation $( \pmb { u } _ { \mathrm { p u s h } } )$ , capturing cross-layer transport toward semantic features;

$- \ \delta \pmb { h } _ { n } ^ { \top } \left( \varPhi _ { n } ^ { l } \right) ^ { \top } \varPhi _ { n } ^ { l } \delta \pmb { h } _ { n }$ provides metric of the push-forward transportation $\left( \pmb { u } _ { \mathrm { m e t r i c } } \right)$ reflecting local metric/curvature information.

While component-wise basis isolation is non-trivial in an online mixed stream, the shared subspace $U$ provides a batch-wise geometry that emphasizes correlated cross-layer directions and suppresses batch-specific noise. This geometry is used for similarity normalization in DFM.

Domain-agnostic Flow Matching Loss. With this insight, we guide intermediate representations away from domain-specific geometry and toward classdiscriminative geometry. Let $\boldsymbol { h } _ { n } ^ { * }$ be a perturbed function of $h _ { n }$ with continuous deformation $( \mathrm { e . g . }$ , induced by prompts); locally, we apply the same $U$ extracted above. For a sample $\mathbf { \pmb { x } } _ { ( i ) }$ in batch $\boldsymbol { X }$ , let the intermediate feature $\begin{array} { r } { \pmb { h } _ { n , ( i ) } \in \pmb { H } _ { n } . } \end{array}$ last-layer feature $\pmb { h } _ { l , ( i ) } \in \pmb { H } _ { l }$ from the frozen backbone model, and intermediate features from the perturbed function ${ h } _ { n , ( i ) } ^ { \ast } \in H _ { n } ^ { \ast }$

Positive/Negative Design: Semantic Pull and Domain-Biased Push. We use the final-layer feature as a semantic anchor and set ${ \cal H } _ { ( i ) } ^ { + } = \{ h _ { l , ( i ) } \}$ as the positive set. To push $h _ { n , ( i ) } ^ { * }$ away from domain-sensitive intermediate regions $\left( \mathrm { F i g . \ 2 a } \right)$ , we use frozen intermediate features as domain-biased negatives. Other samples’ final-layer features are also included to prevent collapse and preserve instance-level discrimination. The negative set is

$$
\begin{array} { r } { H _ { ( i ) } ^ { - } = \left\{ h _ { n , ( j ) } ~ | ~ h _ { n , ( j ) } \in H _ { n } \right\} \cup \left\{ h _ { l , ( j ) } ~ | ~ i \neq j \land h _ { l , ( j ) } \in H _ { l } \right\} } \end{array}\tag{4}
$$

We propose a contrastive loss formulated as follows:

$$
\mathcal { L } _ { \mathrm { { D F M } } } = - \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \log \frac { \sum _ { \boldsymbol { h } _ { n , ( m ) } ^ { + } \in \boldsymbol { H } _ { ( m ) } ^ { + } } \exp \left( \cos ( \boldsymbol { h } _ { ( m ) } ^ { * } \boldsymbol { U } , \boldsymbol { h } _ { ( m ) } ^ { + } \boldsymbol { U } ) / \tau \right) } { \sum _ { \boldsymbol { h } _ { n , ( m ) } ^ { + } \in \boldsymbol { H } _ { ( m ) } ^ { + } } \exp \left( \cos ( \boldsymbol { h } _ { ( m ) } ^ { * } \boldsymbol { U } , \boldsymbol { h } _ { ( m ) } ^ { - } \boldsymbol { U } / \tau \right) } ,\tag{5}
$$

where cos is cosine similarity and τ is a temperature, while dividing by norm reduces the efect of the amplitude of the feature. The further details of the derivation and procedure are provided in Appendix A.2 and Algorithm A.2.

## 3.3 Global Topology Preservation

One significant challenge in Online VIL is that the model must train from batches that only partially represent the overall distribution. This issue is particularly acute when input batches exhibit non-stationary class-domain compositions. This challenge is particularly acute in the Online VIL scenario, where input batches exhibit non-stationary class-domain compositions. To address this representational instability and to ensure topological coherence of the feature space, we introduce Global Topology Preservation (GTP), a novel approach designed to maintain semantic structural integrity across temporally evolving data streams. The efective information content of batch $B _ { t }$ at layer ${ h } _ { l } ( \pmb { x } )$ can be measured by the rank of its empirical covariance matrix:

$$
C _ { t } = \mathbb { E } _ { \pmb { x } \in B _ { t } } \left[ ( h _ { l } ( \pmb { x } ) - \mu _ { t } ) ( h _ { l } ( \pmb { x } ) - \mu _ { t } ) ^ { T } \right] , \quad \mu _ { t } = \mathbb { E } _ { \pmb { x } \in B _ { t } } \left[ h _ { l } ( \pmb { x } ) \right] ,\tag{6}
$$

when certain classes are absent, $C _ { t }$ captures only partial information about the global feature distribution, leading to rank deficiency. When batch $B _ { t }$ contains samples from only $k < C$ classes, the rank of the gradient is limited to $k ;$ the gradient provides not only the construction of decision boundaries for the present classes but also distorts the feature space by contracting around them. This rank deficiency yields incomplete feature representations and biases gradients toward observed classes, causing the feature space to contract around present classes while neglecting absent ones. This issue persists even with a prototype-based classifier [29], which assumes each class is independent, thereby removing the absent rank into a zero-eigenvalue space.

To circumvent these limitations while preserving global topological properties, GTP constructs a surrogate representation of the feature manifold through two key components: a set of $k$ global prototypes $\{ \bar { p } _ { g } ^ { j } \} _ { j = 1 } ^ { k }$ , and a set of global relationship vectors $\{ \bar { r } _ { g } ^ { j , l } \} _ { j , l = 1 } ^ { k }$ . For each incoming batch $B _ { t }$ , we derive batch-specific prototypes $\{ p _ { b } ^ { i } \} _ { i = 1 } ^ { k }$ from the final layer features using FINCH clustering [27]. We employ an exponential moving average (EMA) update strategy to ensure smooth temporal evolution of global prototypes while mitigating batch-to-batch fluctuations. Once the correspondence is established, each global prototype $\bar { p } _ { g }$ is updated with the batch prototype p<sub>b</sub>:

$$
\bar { p } _ { g } ^ { \pi ^ { * } ( i ) , \mathrm { n e w } } = ( 1 - \alpha ) \bar { p } _ { g } ^ { \pi ^ { * } ( i ) , \mathrm { o l d } } + \alpha p _ { b } ^ { i } \quad \mathrm { f o r } \ i = 1 , \dots , k ,\tag{7}
$$

where α controls EMA update rate, balancing stability and adaptability. We find optimal assignment $\pi ^ { * }$ by solving: $\begin{array} { r } { \pi ^ { * } = \arg \operatorname* { m i n } _ { \pi \in S _ { k } } \sum _ { i = 1 } ^ { k } \| p _ { b } ^ { i } - \bar { p } _ { g } ^ { \pi ( i ) } \| } \end{array}$ using Hungarian algorithm [14].

The relationships between its constituent elements fundamentally characterize the topological structure. To capture these structural properties, We introduce a learnable mapping function $\phi : \mathbb { R } ^ { 2 d }  \mathbb { R } ^ { m }$ , where concatenation preserves both individual prototype and relative positioning. We define batch-specific relationship vectors $r _ { b } ^ { i , \bar { j } } = \phi ( [ p _ { b } ^ { i } ; p _ { b } ^ { j } ] )$ and global relationship vectors $\bar { r } _ { g } ^ { i ^ { \prime } , j ^ { \prime } } \dot { = } \phi ( [ \bar { p } _ { g } ^ { i ^ { \prime } } ; \bar { p } _ { g } ^ { j ^ { \prime } } ] )$ . The GTP loss then aligns these relationship structures as follows:

$$
\mathcal { L } _ { \mathrm { G T P } } = \sum _ { i = 1 } ^ { k } \sum _ { j = 1 \atop j \neq i } ^ { k } D ( r _ { b } ^ { i , j } , \bar { r } _ { g } ^ { \pi ^ { * } ( i ) , \pi ^ { * } ( j ) } ) ,\tag{8}
$$

where D represents the cosine distance metric, this formulation encourages consistent pairwise relationships between semantic prototypes, efectively preserving the global topological structure without explicitly computing spectral properties. Since the number of global prototypes $\bar { p } _ { g }$ is smaller than the number of samples in each batch, these prototypes deliberately capture a coarse-grained representation, reflecting the intended vagueness under limited observations. Each prototype is updated considering its relationships with all other prototypes. These prototypes cover the feature space of several semantic classes, which reduces the distortion of the feature space from the concept without samples. Additionally, the EMA update strategy stabilizes the prototypes over time and removes outdated information from previous batches. As shown in Figure A.3 and A.4 in Appendix, this efectively filters out the unseen class information in recent batches, performing a role similar to a short-term memory. This mechanism provides a computationally eficient solution to maintaining representational stability in non-stationary environments, complementing the domain-invariance properties induced by DFM.

## 4 Experiments

This section evaluates and compares proposed TopFlow against state-of-theart methods. Section 4.1 describes the experimental setup, including datasets, baselines, and evaluation metrics. Section 4.2 presents extensive quantitative results, demonstrating the efectiveness of TopFlow across multiple benchmarks. To further analyze TopFlow, Section 4.3 provides in-depth analyses, including ablation studies, to isolate and assess the contribution of each component in TopFlow. Further details about the experiments are provided in the Appendix.

## 4.1 Experimental Setup

Datasets. We conducted experiments on three benchmarks, including iDigits [32], CORe50 [20], and CLEAR100 [17], for which it is possible to construct Online VIL scenarios that can cause a significant shift in distribution by clearly distinguishing both classes and domains. We split the CORe50 and CLEAR100 datasets into 10 tasks for the task construction, and 5 for the iDigits dataset.

Baselines. We compared our proposed method with traditional naive baselines and the latest state-of-the-art methods. First, we set the lower bound as the usual supervised sequential fine-tuning result (FT) and the upper bound as the usual supervised joint fine-tuning result. Then, we compared our proposed method with replay-based methods such as ER [26], Rainbow Memory (RM) [1], CLIB [12], and CBA [33], regularization-based method EWC [11], LwF [16], SLCA [40], DYSON [10], OCM [9], OnPro [38], PEC [39], S6MOD [18], DUCT [41] and prompt-based CODA-P [28], ICON [25] and MVP [23].

Implementation Details. We used the MVP [23] as our baseline for model and experimental setups. We used Adam optimizer with a learning rate of 5e-3, and implemented with a batch size of 64. As a mapping function $\psi$ which maps prototypes into a relation vector, we used a simple 2-layer MLP function. The hidden dimension of $\psi$ is 64 in CORe50, and 32 in CLEAR100. The size of the dimension of the relation vector m is 10. For the EMA update of global feature topology, a decay factor of 0.99 was used. Experiments were conducted under assumption of the memory-free setting, we adopted naïve reservoir memory for the experiments with memory bufer. We conducted experiments with 3 random seeds, and note that using more seeds (e.g., 10 runs) also yields consistent results. Details are described in the Appendix.

Table 1: Experimental results with proposed Online VIL scenarios. We used bold and underlined as brief indications of the best and the second best, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">iDigits</td><td colspan="2">CORe50</td><td colspan="2">CLEAR100</td></tr><tr><td> $\overline { { A _ { \mathrm { A U C } } } }$ </td><td> $A _ { \mathrm { L a s t } }$ </td><td> $\overline { { A _ { \mathrm { A U C } } } }$ </td><td> $\overline { { A _ { \mathrm { L a s t } } } }$ </td><td> $\overline { { A _ { \mathrm { A U C } } } }$ </td><td> $A _ { \mathrm { L a s t } }$ </td></tr><tr><td>Upper-bound</td><td></td><td>87.18±0.13</td><td></td><td>91.66±0.25</td><td></td><td>94.36±0.28</td></tr><tr><td>Lower-bound</td><td></td><td>13.45±0.62 12.71±3.34</td><td></td><td>3.39±0.103.24±1.09</td><td></td><td>2.38±0.32 2.66±0.55</td></tr><tr><td>EWČ [11]</td><td>20.07±2.8414.67±3.61</td><td></td><td></td><td></td><td>18.60±4.64 16.07±1.5623.61±3.11 19.93±1.33</td><td></td></tr><tr><td>LwF [16]</td><td>19.61±3.50 15.38±1.04</td><td></td><td>25.27±3.77 21.97±4.22</td><td></td><td>23.80±2.24 21.70±4.19</td><td></td></tr><tr><td>CODA-P [28]</td><td>23.96±4.74 20.62±3.47</td><td></td><td></td><td></td><td>54.06±5.32 48.88±2.9228.82±5.77 25.61±3.52</td><td></td></tr><tr><td>SLCA [40]</td><td>35.81±3.98 24.88±2.82</td><td></td><td></td><td></td><td>33.49±4.47 27.36±2.0632.16±2.28 31.70±1.13</td><td></td></tr><tr><td>PEC [39]</td><td>34.77±3.02 28.01±2.69</td><td></td><td></td><td></td><td>51.35±4.3946.95±2.2853.93±3.20 51.66±2.09</td><td></td></tr><tr><td>ICON [25]</td><td>33.60±2.16 30.63±2.77</td><td></td><td></td><td></td><td>49.42±3.29 45.15±2.9459.38±2.4658.60±2.57</td><td></td></tr><tr><td>S6MOD [18]</td><td>31.18±1.62 30.42±2.55</td><td></td><td></td><td>52.33±2.74 47.20±3.15</td><td>59.60±2.79 57.47±2.48</td><td></td></tr><tr><td>DUCT [41]</td><td>36.46±3.4830.94±2.41</td><td></td><td>53.66±4.37 47.24±1.28</td><td></td><td>66.79±2.9465.46±3.40</td><td></td></tr><tr><td>MVP [23]</td><td>38.29±5.74 31.05±3.15</td><td></td><td></td><td>58.30±4.48 52.84±1.17</td><td></td><td>79.73±3.59 77.11±2.33</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>TopFlow (Ours) 48.52±1.25 32.18±1.01 64.51±2.50 66.20±4.18 87.12±0.01 80.64±2.67</td></tr></table>

Evaluation Metrics. To evaluate online learning performance, we employed two metrics: $A _ { \mathrm { A U C } }$ and $A _ { \mathrm { L a s t } }$ [12]. The $A _ { \mathrm { A U C } }$ metric quantifies performance under anytime inference, where inference queries may occur at arbitrary points during training as new classes are encountered. Conversely, $A _ { \mathrm { L a s t } }$ assesses inference accuracy after training. In real-world applications, models must deliver reliable predictions on demand, regardless of training stage. Thus, $A _ { \mathrm { A U C } }$ and $A _ { \mathrm { L a s t } }$ provide a robust framework for benchmarking online learning performance.

## 4.2 Experimental Results

We conducted extensive experiments in the proposed Online VIL scenario, and the results are summarized in Table 1 and Table 2. As shown in Table 1, under the non-replay setting, TopFlow consistently outperforms existing methods in both $A _ { \mathrm { A U C } }$ and $A _ { \mathrm { L a s t } }$ . These results demonstrate that our approach mitigates catastrophic forgetting while enabling continual adaptation to the input stream. Moreover, in Table 2, the introduction of replay memory generally improves performance, but the proposed TopFlow consistently outperforms all baselines. Surprisingly, in replay-bufer settings, we observe that naïve methods such as Experience Replay (ER) and earlier methods tend to achieve the best performance, except for our proposed method. This suggests that most existing online-incremental learning methods struggle in realistic Online VIL scenarios where distribution shifts are frequent and task boundaries are unclear. In contrast, our proposed TopFlow maintains strong performance across all settings, confirming its robustness. Furthermore, TopFlow achieves a more stable accuracy trajectory over time, with fewer drastic performance drops between tasks. This indicates that DFM and GTP help smooth the learning process by leveraging more structured representations and efective feature alignment.

Table 2: Results of OnlineVIL scenarios using replay bufer sizes 500 and 2000.
<table><tr><td rowspan=2 colspan=8>Buffer                         iDigits               CORe50             CLEAR100MethodSize                      $\overline { { A _ { \mathrm { A U C } } } }$       $A _ { \mathrm { L a s t } }$ </td></tr><tr><td rowspan=1 colspan=2>AAUCALast</td><td rowspan=1 colspan=2>AAUCALast</td></tr><tr><td rowspan=1 colspan=4>ER [26]     59.43±6.2448.70±2.517</td><td rowspan=1 colspan=2>4.77±4.85 72.25±2.277</td><td rowspan=1 colspan=2>3.92±3.93 $\overline { { 7 1 . 4 9 { \pm } 3 . 2 0 } }$ </td></tr><tr><td rowspan=1 colspan=2>RM [1]</td><td rowspan=1 colspan=2>55.02±5.37 51.73±2.05</td><td rowspan=1 colspan=2>81.06±3.90 70.41±3.17</td><td rowspan=1 colspan=2>72.42±4.64 $7 2 . 9 3 { \pm } 2 . 0 5$ </td></tr><tr><td rowspan=1 colspan=2>CLIB [12]</td><td rowspan=1 colspan=2>57.38±4.16 52.63±3.38</td><td rowspan=1 colspan=2>75.06±5.81 71.93±1.06</td><td rowspan=1 colspan=2>68.39±5.25 66.92±1.52</td></tr><tr><td rowspan=1 colspan=2>OCM [9]</td><td rowspan=1 colspan=2>57.40±3.60 52.88±2.52</td><td rowspan=1 colspan=2>75.29±3.10 72.66±1.93</td><td rowspan=1 colspan=2>77.80±3.25 $7 5 . 1 0 { \pm } 2 . 4 2$ </td></tr><tr><td rowspan=1 colspan=2>500    CBA [33]</td><td rowspan=1 colspan=2>58.05±4.39 54.28±3.07</td><td rowspan=1 colspan=2>81.92±4.0481.02±1.39</td><td rowspan=1 colspan=2>75.26±4.08 74.47±1.82</td></tr><tr><td rowspan=1 colspan=2>OnPro [38]</td><td rowspan=1 colspan=1>46.92±4.82</td><td rowspan=1 colspan=1>44.86±1.63</td><td rowspan=1 colspan=2>71.39±4.0370.92±2.24</td><td rowspan=1 colspan=2>81.36±4.99 77.46±1.53</td></tr><tr><td rowspan=1 colspan=2>DYSON [10]</td><td rowspan=1 colspan=1>42.31±3.11</td><td rowspan=1 colspan=1>38.18±3.56</td><td rowspan=1 colspan=2>62.92±5.61 60.72±2.16</td><td rowspan=1 colspan=2>66.56±4.65 65.62±2.74</td></tr><tr><td rowspan=1 colspan=2>MVP-R [23]</td><td rowspan=1 colspan=1>48.29±3.734</td><td rowspan=1 colspan=1>0.97±2.15</td><td rowspan=1 colspan=2>83.26±5.16 80.16±1.03</td><td rowspan=1 colspan=2>87.82±3.1785.65±2.18</td></tr><tr><td rowspan=1 colspan=4>TopFlow (Ours)62.48±4.17 56.48±1.52</td><td rowspan=1 colspan=2>85.16±0.84 91.14±0.05</td><td rowspan=1 colspan=2>91.67±0.03 90.12±1.00</td></tr><tr><td rowspan=1 colspan=4>ER [26]     61.72±5.1257.81±1.47</td><td rowspan=1 colspan=2>79.44±5.17 76.61±3.91</td><td rowspan=1 colspan=2>83.51±4.61 81.03±3.59</td></tr><tr><td rowspan=1 colspan=3>RM [1]     43.96±6.18</td><td rowspan=1 colspan=1>41.23±3.62</td><td rowspan=1 colspan=2>82.42±3.03 79.84±2.19</td><td rowspan=1 colspan=2>84.73±4.63 81.49±2.26</td></tr><tr><td rowspan=1 colspan=3>CLIB [12]    51.96±3.81</td><td rowspan=1 colspan=1>46.06±2.34</td><td rowspan=1 colspan=2>84.58±4.26 81.63±2.51</td><td rowspan=1 colspan=2>85.62±5.05 83 $\hphantom { - } 3 . 2 5 { \pm } 1 . 6 2$ </td></tr><tr><td rowspan=1 colspan=3>OCM [9]    57.52±5.08</td><td rowspan=1 colspan=3>50.60±3.3484.92±4.03 83.24±2.72</td><td rowspan=1 colspan=2>84.26±4.82 82.91±3.56</td></tr><tr><td rowspan=1 colspan=4>2000    CBA [33]     $6 0 . 0 2 { \pm } 4 . 5 2 5 5 . 0 4 { \pm } 2 . 8 2$ </td><td rowspan=1 colspan=2>85.16±3.28 83.49±3.20</td><td rowspan=1 colspan=2>88.02±3.80 85.94±2.36</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>OnPro [38]    $5 4 . 8 2 { \pm } 4 . 9 5 ~</td><td rowspan=1 colspan=1>5 1 . 0 3 { \pm } 3 . 7 7$ </td><td rowspan=1 colspan=1>81.35±5.51 7</td><td rowspan=1 colspan=1>8.09±3.91</td><td rowspan=1 colspan=1>88.83±5.16 8</td><td rowspan=1 colspan=1>6.11±3.57</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>DYSON [10]   $4 6 . 7 4 \pm 4 . 4 1 \</td><td rowspan=1 colspan=1>4 3 . 3 8 \pm 4 . 2 8$ </td><td rowspan=1 colspan=1>51.21±3.71</td><td rowspan=1 colspan=1>49.29±1.74</td><td rowspan=1 colspan=1>57.05±4.18 5</td><td rowspan=1 colspan=1>5.48±3.43</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>MVP-R [23]   $5 2 . 1 4 { \pm } 2 . 9 9 ~</td><td rowspan=1 colspan=1>4 7 . 7 4 { \pm } 2 . 3 6$ </td><td rowspan=1 colspan=1>87.33±3.37 8</td><td rowspan=1 colspan=1>2.39±1.10</td><td rowspan=1 colspan=1>89.48±1.71 8</td><td rowspan=1 colspan=1>8.93±1.18</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>TopFlow (Ours)65.45±3.84 60.39±4.28 8</td><td rowspan=1 colspan=1>7.56±0.82</td><td rowspan=1 colspan=1>92.24±0.15</td><td rowspan=1 colspan=2>93.55±0.02 92.97±0.65</td></tr></table>

Table 4: The Efectiveness of TopFlow in Standard CL Scenarios.  
Table 5: Ablation of layer selection in the DFM loss (w/o GTP).

Table 3: Ablation study for DFM and GTP on CORe50.
<table><tr><td colspan="2">DFM GTP  $A _ { \mathrm { A U C } }$ </td><td> $A _ { \mathrm { L a s t } }$ </td></tr><tr><td>Baseline  $\bar { \surd }$ </td><td>64.13 64.57</td><td>58.3052.84</td></tr><tr><td>√</td><td>63.9565.28</td><td></td></tr><tr><td></td><td>64.5166.20</td><td></td></tr></table>

<table><tr><td>Method</td><td colspan="2">Accuracy CIL, CODA-P DIL, S-Prompt</td></tr><tr><td>Baseline</td><td>84.17</td><td>82.96</td></tr><tr><td>+DFM</td><td>84.19</td><td>85.36</td></tr><tr><td>+ GTP</td><td></td><td>86.88</td></tr><tr><td>+ DFM, GTP</td><td></td><td>85.52 86.04 87.50</td></tr></table>

<table><tr><td>Layer</td><td> $A _ { \mathrm { L a s t } }$ </td></tr><tr><td>Baseline</td><td>52.84</td></tr><tr><td>(0, 5)</td><td> ${ \overline { { 4 9 } } } . { \overline { { 9 2 } } } { \overline { { 2 } } }$ </td></tr><tr><td>(5, 10) (5, 10), (6, 11)</td><td>52.98</td></tr><tr><td>(6, 11) (Ours) 54.82</td><td>53.49</td></tr></table>

## 4.3 Ablation Studies and Analysis

Ablation Study. Table 3 demonstrates the efectiveness of each component in our TopFlow framework. Implementing DFM or GTP individually significantly enhanced both $A _ { \mathrm { A U C } }$ and $A _ { \mathrm { L a s t } } .$ , validating their respective contributions. DFM stabilizes learning by distilling domain-invariant self-knowledge and enhancing class discrimination in later layers. GTP preserves performance under varying batch compositions by consolidating transient batch cues into a persistent feature topology. Together, they provide complementary gains: DFM improves representations, while GTP maintains their structure for Online VIL.

Table 7: The Efectiveness of TopFlow in Ofline VIL Scenarios.
<table><tr><td rowspan="3">Method</td><td colspan="2">iDigits</td><td colspan="2">CORe50</td><td colspan="2">DomainNet</td></tr><tr><td></td><td></td><td>Avg. Acc ↑ Forgetting ↓ Avg. Acc ↑ Forgetting ↓ Avg. Acc ↑ Forgetting ↓</td><td></td><td></td><td></td></tr><tr><td>ICON</td><td>70.67±2.13 13.49±2.09</td><td></td><td> $\overline { { 7 9 . 6 1 \pm 1 . 7 3 } }$ </td><td> $8 . 1 9 { \pm } 1 . 4 7$ </td><td>49.08±2.11 17.88±2.40</td><td></td></tr><tr><td>TopFlow (Ours) 73.33±1.72</td><td></td><td> $\mathbf { 1 2 . 1 4 } 2 4 . 2 0$ </td><td> ${ \bf 7 9 . 7 7 { \pm } 1 . 4 7 }$ </td><td> $\mathbf { 8 . 0 3 \pm 1 . 1 9 }$ </td><td>49.62±1.82 17.30±3.09</td><td></td></tr></table>

Efectiveness of TopFlow in Standard CL Scenarios. As shown in Table 4, DFM and GTP improve performance in both standard CIL and DIL, individually and combined. We adopt CIFAR-100 [13] 10-split CIL and CORe50 [20] 8-split DIL as standard setups, with CODA-P [28] and S-Prompts [35] as baselines, respectively. Despite the limited domain variation in CIL, both components provide consistent gains, and the improvements are more pronounced in DIL, supporting their motivation under domain shifts.

Layer Selection for DFM. To analyze the efect of layer selection for Eq. 5, we conduct an ablation study using various $( n , l )$ pairs in the ViT encoder (Table 5). Applying DFM to early layers such as (0,5) degrades performance, suggesting that low-level features are not well-aligned with the high-level semantics captured in the final layer. In contrast, using deeper layers such as (5,10) or (6,11) improves performance; combining (5,10) and (6,11) reaches 53.49, and our final configuration (6,11) achieves the best $A _ { \mathrm { L a s t } }$ of 54.82. This indicates that later layers better preserve semantic information suitable for matching with final representations, yielding a favorable trade-of between abstraction and compatibility.

Table 6: Performance comparison of diferent methods and variants.
<table><tr><td>Method</td><td> $A _ { \mathbf { L a s t } }$ </td></tr><tr><td>CODA-P</td><td>49.13</td></tr><tr><td>+ DFM</td><td>52.88</td></tr><tr><td>+ GTP</td><td>53.17</td></tr><tr><td>+ DFM, GTP 54.62</td><td></td></tr><tr><td>PEC</td><td>45.99</td></tr><tr><td>+ DFM</td><td>46.64</td></tr><tr><td>+ GTP</td><td>48.19</td></tr><tr><td>+ DFM, GTP 48.86</td><td></td></tr></table>

DFM and GTP with Other Models. Since DFM and GTP are designed as general components, we further examine whether they remain efective when integrated with other CL algorithms. We conducted ablation experiments to verify whether the proposed DFM and GTP could also yield performance improvements on baseline CL algorithms other than the MVP. As shown in Table 6, DFM and GTP lead to performance improvements over the baseline, even independently. DFM and GTP robustly improve performance over the baseline on the CORe50 dataset and other existing continual learning algorithms used in the main experiments.

Efectiveness of TopFlow in Ofline VIL Scenarios. As shown in Table 7, the proposed TopFlow is efective not only in standard Ofline CL scenarios such as CIL and DIL, but also in the more challenging Ofline VIL setting, where both class and domain distributions change simultaneously. TopFlow achieves this by explicitly maintaining structural consistency by enforcing topology-preserving and Domain-agnostic matching across tasks, allowing newly learned representations to align with the previously learned feature structure. As a result, TopFlow achieves stable representation learning and consistently improves performance across Ofline VIL benchmarks. From a scenario perspective, TopFlow was originally designed for the more challenging Online VIL setting, where the model must adapt to non-stationary data streams without access to the full dataset distribution. Nevertheless, as shown in Table 7, TopFlow also achieves strong performance in the Ofline VIL scenario. In contrast, the Ofline VIL method ICON fails to maintain competitive performance in the Online VIL setting, as reported in Table 2. This result suggests that while methods tailored to Ofline VIL may not generalize well to streaming environments, TopFlow provides a more robust solution that transfers across diferent VIL scenarios.

![](images/6a8a650db90f7190252e4f7d0e3b1547522f3427c76f349cc482b9a372664d31.jpg)  
Fig. 4: t-SNE visualization of output features over tasks with GTP and without GTP.

Visualization of GTP. To qualitatively analyze the efect of GTP, we visualize the output features using t-SNE [21] over tasks with and without GTP in Figure 4. We tracked the evolution of features from the first 10 classes, which are available for all tasks in the CORe50 dataset. And low-dimensional approximation is performed across all tasks to maintain consistency and relative positions of features. While it is challenging to project high-dimensional topological structures into 2D space, we observe that GTP helps to preserve the relative arrangement and internal structure of clusters across tasks. Especially in red circles, the meaningful topologies (e.g., relative connections or penetrations between clusters) are better maintained with GTP, whereas they drift and dismorph without it. This qualitative analysis supports our findings, demonstrating that GTP preserves global topological relationships in the feature space during online learning.

## 5 Conclusion

We proposed a new online continual learning scenario named Online VIL, which simulates a complex real world where states are ever-changing and there are no concepts of tasks and clear boundaries between them. Through analysis, we determined the direction for problem-solving in Online VIL and defined novel TopFlow framework. We demonstrated that the proposed TopFlow showed SOTA performance in the challenging Online VIL scenario, and its efectiveness through various experiments. Online VIL involves stochastic task construction, where the composition of data to each task is influenced by random seed. While this design captures more realistic dynamics, it can introduce variability in results. Despite this, we hope that our Online VIL scenario will serve as a new benchmark for advancing real-world incremental learning research, providing a more realistic and challenging setting for future studies.

## Acknowledgements

This work was partly supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS-2026-25507543, Development of AI Co-Scientist based on Scientific Causal World Model, 80%), and in part by Korea Planning & Evaluation Institute of Industrial Technology (KEIT) grant funded by the Korea government (MOTIE) (RS-2024-00444344), and by the “Advanced GPU Utilization Support Program” funded by the Government of the Republic of Korea (Ministry of Science and ICT).

## References

1. Bang, J., Kim, H., Yoo, Y., Ha, J.W., Choi, J.: Rainbow memory: Continual learning with a memory of diverse samples. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8218–8227 (2021)

2. Cayton, L., et al.: Algorithms for manifold learning. Univ. of California at San Diego Tech. Rep 12(1-17), 1 (2005)

3. De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., Slabaugh, G., Tuytelaars, T.: A continual learning survey: Defying forgetting in classification tasks. IEEE transactions on pattern analysis and machine intelligence 44(7), 3366– 3385 (2021)

4. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. In: Proceedings of the International Conference on Learning Representations (2020)

5. Gao, Q., Zhao, C., Sun, Y., Xi, T., Zhang, G., Ghanem, B., Zhang, J.: A unified continual learning framework with general parameter-eficient tuning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 11483–11493 (2023)

6. Gong, B., Shi, Y., Sha, F., Grauman, K.: Geodesic flow kernel for unsupervised domain adaptation. In: 2012 IEEE conference on computer vision and pattern recognition. pp. 2066–2073. IEEE (2012)

7. Gopalan, R., Li, R., Chellappa, R.: Domain adaptation for object recognition: An unsupervised approach. In: 2011 international conference on computer vision. pp. 999–1006. IEEE (2011)

8. Gunasekara, N., Pfahringer, B., Gomes, H.M., Bifet, A.: Survey on online streaming continual learning. In: Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence. pp. 6628–6637 (2023)

9. Guo, Y., Liu, B., Zhao, D.: Online continual learning through mutual information maximization. In: International conference on machine learning. pp. 8109–8126. PMLR (2022)

10. He, Y., Chen, Y., Jin, Y., Dong, S., Wei, X., Gong, Y.: Dyson: Dynamic feature space self-organization for online task-free class incremental learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 23741–23751 (2024)

11. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al.: Overcoming

catastrophic forgetting in neural networks. Proceedings of the national academy of sciences 114(13), 3521–3526 (2017)

12. Koh, H., Kim, D., Ha, J.W., Choi, J.: Online continual learning on class incremental blurry task configuration with anytime inference. arXiv preprint arXiv:2110.10031 (2021)

13. Krizhevsky, A., Hinton, G., et al.: Learning multiple layers of features from tiny images.(2009) (2009)

14. Kuhn, H.W.: The hungarian method for the assignment problem. Naval research logistics quarterly 2(1-2), 83–97 (1955)

15. Li, Z., Hoiem, D.: Learning without forgetting. IEEE Trans. Pattern Anal. Mach. Intell. 40 (2017). https://doi.org/10.1109/TPAMI.2017.2773081, https://doi. org/10.1109/TPAMI.2017.2773081

16. Li, Z., Hoiem, D.: Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence 40(12), 2935–2947 (2017)

17. Lin, Z., Shi, J., Pathak, D., Ramanan, D.: The clear benchmark: Continual learning on real-world imagery. In: Thirty-fifth conference on neural information processing systems datasets and benchmarks track (round 2) (2021)

18. Liu, S., Yang, Y., Li, X., Clifton, D.A., Ghanem, B.: Enhancing online continual learning with plug-and-play state space model and class-conditional mixture of discretization. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 20502–20511 (2025)

19. Liu, Y., Hong, X., Tao, X., Dong, S., Shi, J., Gong, Y.: Model behavior preserving for class-incremental learning. IEEE Transactions on Neural Networks and Learning Systems 34(10), 7529–7540 (2022)

20. Lomonaco, V., Maltoni, D.: Core50: a new dataset and benchmark for continuous object recognition. In: Conference on Robot Learning. pp. 17–26. PMLR (2017)

21. van der Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of Machine Learning Research 9(86), 2579–2605 (2008), http://jmlr.org/papers/v9/ vandermaaten08a.html

22. Mai, Z., Li, R., Kim, H., Sanner, S.: Supervised contrastive replay: Revisiting the nearest class mean classifier in online class-incremental continual learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3589–3599 (2021)

23. Moon, J.Y., Park, K.H., Kim, J.U., Park, G.M.: Online class incremental learning on stochastic blurry task boundary via mask and visual prompt tuning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 11731–11741 (2023)

24. Park, K.H., Song, K., Park, G.M.: Pre-trained vision and language transformers are few-shot incremental learners. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 23881–23890 (2024)

25. Park, M.Y., Lee, J.H., Park, G.M.: Versatile incremental learning: Towards class and domain-agnostic incremental learning. In: European Conference on Computer Vision. pp. 271–288. Springer (2024)

26. Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., Wayne, G.: Experience replay for continual learning. Advances in neural information processing systems 32 (2019)

27. Sarfraz, S., Sharma, V., Stiefelhagen, R.: Eficient parameter-free clustering using first neighbor relations. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8934–8943 (2019)

28. Smith, J.S., Karlinsky, L., Gutta, V., Cascante-Bonilla, P., Kim, D., Arbelle, A., Panda, R., Feris, R., Kira, Z.: Coda-prompt: Continual decomposed attention-based

prompting for rehearsal-free continual learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11909–11919 (2023)

29. Snell, J., Swersky, K., Zemel, R.: Prototypical networks for few-shot learning. Advances in neural information processing systems 30 (2017)

30. Tao, X., Hong, X., Chang, X., Dong, S., Wei, X., Gong, Y.: Few-shot classincremental learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12183–12192 (2020)

31. Tao, X., Hong, X., Chang, X., Gong, Y.: Bi-objective continual learning: Learning ‘new’while consolidating ‘known’. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 34, pp. 5989–5996 (2020)

32. Volpi, R., Larlus, D., Rogez, G.: Continual adaptation of visual representations via domain randomization and meta-learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4443–4453 (2021)

33. Wang, Q., Wang, R., Wu, Y., Jia, X., Meng, D.: Cba: Improving online continual learning via continual bias adaptor. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19082–19092 (2023)

34. Wang, S., Shi, W., Dong, S., Gao, X., Song, X., Gong, Y.: Semantic knowledge guided class-incremental learning. IEEE Transactions on Circuits and Systems for Video Technology 33(10), 5921–5931 (2023)

35. Wang, Y., Huang, Z., Hong, X.: S-prompts learning with pre-trained transformers: An occam’s razor for domain incremental learning. In: Oh, A.H., Agarwal, A., Belgrave, D., Cho, K. (eds.) Proceedings of the Advances in Neural Information Processing Systems (2022)

36. Wang, Z., Zhang, Z., Ebrahimi, S., Sun, R., Zhang, H., Lee, C.Y., Ren, X., Su, G., Perot, V., Dy, J., et al.: Dualprompt: Complementary prompting for rehearsal-free continual learning. In: European Conference on Computer Vision. pp. 631–648. Springer (2022)

37. Wang, Z., Zhang, Z., Lee, C.Y., Zhang, H., Sun, R., Ren, X., Su, G., Perot, V., Dy, J., Pfister, T.: Learning to prompt for continual learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 139–149 (2022)

38. Wei, Y., Ye, J., Huang, Z., Zhang, J., Shan, H.: Online prototype learning for online continual learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 18764–18774 (2023)

39. Zając, M., Tuytelaars, T., van de Ven, G.M.: Prediction error-based classification for class-incremental learning. In: International Conference on Learning Representations (2024)

40. Zhang, G., Wang, L., Kang, G., Chen, L., Wei, Y.: Slca: Slow learner with classifier alignment for continual learning on a pre-trained model. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19148–19158 (2023)

41. Zhou, D.W., Cai, Z.W., Ye, H.J., Zhang, L., Zhan, D.C.: Dual consolidation for pretrained model-based domain-incremental learning. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 20547–20557 (2025)