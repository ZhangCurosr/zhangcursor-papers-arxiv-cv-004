# Visual Memory and Textual Alignment for Cross-Modal Few-Shot Segmentation

Tianqi Wei<sup>1</sup>, Zhi Chen<sup>2</sup>, Xin Yu<sup>3</sup>, Scott Chapman<sup>1</sup>, and Zi Huang<sup>1</sup>

<sup>1</sup> The University of Queensland, Brisbane, Australia

{tianqi.wei,scott.chapman,helen.huang}@uq.edu.au

<sup>2</sup> University of Southern Queensland, Toowoomba, Australia

Zhi.Chen@unisq.edu.au

3 Adelaide University, Adelaide, Australia xin.yu@adelaide.edu.au

Abstract. Few-shot segmentation (FSS) aims to segment target objects with limited annotated examples. Recent studies have begun to explore large foundation models to improve generalization for this task. In particular, SAM 3 provides strong visual representations and supports textguided segmentation. However, its performance remains limited in specialized domains, where semantic prompts and visual representations are often poorly aligned. To address this challenge, we propose ViTA-SAM, a framework built upon SAM 3 that integrates visual memory and textual features through cross-modal alignment. Specifically, we construct a visual memory module that encodes support features together with their corresponding masks, providing support-aware visual guidance for query prediction. In parallel, we leverage the text encoder of SAM 3 to extract semantic features as complementary priors. To bridge the gap between semantic and visual representations, we introduce a cross-modal alignment module that conditions textual features on memory-enhanced visual representations, yielding more consistent and visually grounded predictions. In addition, we propose a pseudo-label refinement strategy that updates the memory with predicted query masks, enabling query-adaptive representation learning. Extensive experiments on multiple benchmarks show that ViTA-SAM achieves the best overall performance among the compared methods, outperforming the strongest competing method by over 5% in average mIoU.

Keywords: Few-shot segmentation · SAM 3.

## 1 Introduction

Semantic segmentation is a core task in computer vision and multimedia applications, with broad applications in medical imaging [27], autonomous system [1, 37], and visual content understanding [31, 35]. Recent advances in deep learning, together with the availability of large-scale annotated datasets, have led to substantial progress in segmentation performance [6, 42]. Despite these successes, pixel-wise annotation is still highly labor-intensive and costly, making large-scale dataset construction dificult in practice. Few-shot segmentation (FSS) [28, 29, 34, 38, 39] has therefore attracted increasing attention as a promising alternative. Its goal is to segment target objects from only a few annotated examples.

![](images/1b691e14e1bfe3ed0bbf70322e3bcc7f0be28371288ebbffcb62cc4d585a8994.jpg)  
Fig. 1: (a) Under text prompts, predictions show large deviations from the ground truth, revealing severe semantic–visual misalignment under domain shifts. (b) Traditional FSS methods rely on prototype matching between support and query features. (c) Our method introduces textual representations and performs crossmodal alignment with visual features to provide complementary semantic cues.

Conventional FSS methods mainly focus on constructing representative prototypes (Figure 1b). The pioneering work PANet [34] learns class-specific prototypes from support images and segments the query image via pixel-wise matching in the feature space. CANet [39] performs multi-level feature comparison between query features and global support representations. PFENet [32] leverages support-guided prior masks to encode coarse object locations and uses them to alleviate spatial inconsistency between support and query targets. Despite their efectiveness, the visual matching functions learned by these methods are often biased toward the training distribution, limiting their generalization to diverse scenarios. Subsequent methods improve robustness to such distribution variations. For example, PATNet [14] introduces a prototype alignment module to reduce feature discrepancies. DMTNet [4] addresses distribution variations via a dynamic multi-scale mechanism that adaptively weights features based on input characteristics. ABCDFSS [11] performs task-specific feature adaptation at test time, thereby better aligning representations with the current input.

Meanwhile, recent advances in large-scale foundation models, particularly the Segment Anything Model (SAM) family, have opened new opportunities for generalizable segmentation. The original SAM [12] introduces promptable zeroshot segmentation using visual prompts such as points and bounding boxes. SAM 2 [26] extends this capability to video segmentation by incorporating a temporal memory mechanism for consistent mask propagation. More recently, SAM 3 [3] integrates the memory mechanism from SAM 2 and improves vision– language alignment, enabling the use of both visual concepts and textual prompts. However, SAM models are primarily pretrained on natural images, which limits their generalization to specialized domains. As illustrated in Figure 1a, SAM 3 struggles with specialized imagery (e.g., medical or satellite images) and domainspecific concepts (e.g., “keratosis”), suggesting a misalignment between semantic and visual representations that degrades segmentation performance.

Based on these observations, we propose ViTA-SAM, a novel framework built on SAM 3 that aligns visual memory and textual prompts through cross-modal interaction and enables robust knowledge transfer (Figure 1c). Specifically, we leverage the visual memory module to encode support features and their corresponding masks into a memory bank, producing object-aware representations. During inference, the memory is fused with query features to propagate structural and boundary information from support to query. In parallel, we incorporate textual information using the SAM 3 text encoder to extract task-specific semantic features as complementary priors. To bridge the misalignment between semantic and visual representations under domain shifts, we align textual features with memory-enhanced visual features via cross-attention, yielding visually grounded semantic features. Furthermore, we introduce a pseudo-label refinement strategy that uses predicted query masks to update the memory with query features, constructing a query-adaptive memory for progressively refined predictions. During inference, the memory-enhanced visual features and aligned textual features are fused using prediction-confidence-based weights. Overall, the main contributions of this work are as follows:

– We propose ViTA-SAM, a novel framework for few-shot segmentation. It integrates visual memory and textual features through cross-modal alignment, enabling efective knowledge transfer under domain shifts.

We introduce a cross-modal alignment mechanism that aligns textual and visual representations via cross-attention, producing visually grounded semantic features and improving support-query alignment. A pseudo-label refinement strategy is also used to obtain query-adaptive representations.

– Extensive experiments on four widely used benchmarks demonstrate the effectiveness of our method, which generally outperforms existing approaches.

## 2 Related Work

Few-shot Semantic Segmentation. Few-shot semantic segmentation (FSS) aims to segment query images using only a few annotated support samples. Existing methods can be broadly categorized into prototype-based and afinity-based methods. Prototype-based approaches [8, 15, 13, 19, 29, 32, 34] construct category prototypes from support images and segment queries by measuring feature similarity. However, prototype construction typically relies on masked average pooling, which may discard fine-grained spatial structures and limit representation capacity. Afinity-based methods [20, 21, 24, 40] model dense correspondences between query and support images, better preserving contextual and structural information. Despite their strong performance, traditional FSS approaches generally assume similar training and testing distributions, and often degrade under significant domain shifts.

Cross-domain FSS. Cross-domain few-shot semantic segmentation (CD-FSS) considers a more challenging setting with significant domain shifts between source and target domains. Existing methods mainly address this problem by learning domain-invariant representations or adapting models to target-domain data. PATNet [14], DMTNet [4], and IFA [22] model support-query relations to reduce domain-specific variations, while ABCDFSS [11] and DR-Adapter [30] adapt pre-trained features or rectify domain styles at test time. APSeg [10] further constructs query prototypes through cycle consistency and maps features into a stable domain-agnostic space. More recently, TVGTANet [18] incorporates vision-language modeling by leveraging semantic information from prompts to guide CD-FSS.

Foundation Models for Segmentation. Large-scale vision and vision-language foundation models, such as CLIP [25], DINO [23], and LLaVA [17], have demonstrated strong zero-shot and few-shot generalization across downstream tasks. The Segment Anything Model (SAM) family has emerged as a representative foundation model for promptable segmentation. SAM [12] enables prompt-based zero-shot segmentation using visual cues, while SAM 2 [26] extends this capability to video through a temporal memory mechanism. More recently, SAM 3 [3] incorporates vision-language alignment to support open-vocabulary segmentation. Despite their strong zero-shot capabilities, these models remain constrained by significant domain shifts in specialized scenarios. To bridge this gap, we propose ViTA-SAM, which combines the visual representations and textual guidance of SAM 3 with annotated support images for few-shot segmentation.

## 3 Methodology

## 3.1 Task Formulation

Following the meta-learning paradigm [14], the FSS task is formulated as learning from a series of episodes. Each episode consists of a query set $\mathbb { Q } = \{ ( I ^ { q } , M ^ { q } ) \}$ and a support set $\mathbb { S } = \{ ( I _ { k } ^ { s } , M _ { k } ^ { s } ) \} _ { k = 1 } ^ { K } .$ , where both sets belong to the same category. We denote $\pmb { I } \in \bar { \mathbb { R } ^ { H \times W \times 3 } }$ as the input image and use $\bar { M } \in \{ 0 , 1 \} ^ { H \times W }$ to represent the corresponding binary ground-truth mask. K represents the shot number, indicating the number of annotated support samples available to guide the segmentation of the query image $I ^ { q } .$ . The objective is to acquire a segmentation model $f _ { \theta }$ that predicts the query mask $\hat { M } ^ { q } = f _ { \theta } ( { \cal I } ^ { q } , \mathbb { S } )$ . During the inference process, the model is evaluated on episodes sampled from the test set $\mathcal { D } _ { t e s t }$

## 3.2 ViTA-SAM

The multi-modal alignment capability of SAM 3 forms the representational basis of our ViTA-SAM by projecting diferent modalities into a shared latent space. As illustrated in Figure 2, given a support image, a query image $\pmb { I } ^ { q }$ , and a category-level text prompt $T ,$ , the image encoder IE first extracts the support feature $\pmb { F } ^ { s }$ and the query feature ${ \pmb F } ^ { q }$ , while the text encoder T E produces the textual feature ${ \pmb F } ^ { t }$ . The support feature is then encoded into a memory bank $B ,$ from which the query feature retrieves relevant information to obtain an enhanced representation for segmentation. In parallel, the textual feature is conditioned on the memory-enhanced visual representation through a lightweight cross-attention module. To facilitate cross-modal alignment, this module is trained using pseudo-labels derived from the output of the visual memory branch. Furthermore, to improve prediction consistency, we introduce a pseudo-label refinement step that updates the memory with query-specific information derived from the current prediction.

![](images/bd8f4052baed6d77acb1b7e22fbc3c1f2323a7baafad2c13a7ea88cf4d65e1be.jpg)  
Fig. 2: Overview of ViTA-SAM. The method integrates visual memory and textual alignment to capture both appearance and semantic cues.

## 3.3 Support Memory for Query Understanding

The visual memory module aims to extract and preserve appearance-level cues from annotated support images and transfer these cues to the query representation. Unlike traditional prototypical approaches that compress support information into a single vector, our design maintains spatially-aware feature representations to provide fine-grained guidance during query segmentation.

Memory Construction. Given K-shot support images $\{ I _ { k } ^ { s } \} _ { k = 1 } ^ { K }$ , we first use the SAM 3 image encoder IE to obtain the visual features $F _ { k } ^ { s } = I E ( I _ { k } ^ { s } ) \in$ $\mathbb { R } ^ { h \times w \times C }$ . Then, the memory encoder ME encodes the image features with their corresponding masks as memories:

$$
M e m _ { k } = M E ( F _ { k } ^ { s } , M _ { k } ^ { s } ) , \quad k \in \{ 1 , \ldots , K \} .\tag{1}
$$

The visual memory bank B is then defined as the collection of these maskconditioned representations:

$$
\pmb { \mathcal { B } } = C o n c a t ( M e m _ { 1 } , M e m _ { 2 } , \ldots , M e m _ { K } ) .\tag{2}
$$

Query Interaction. The query interaction enables the query to retrieve spatially relevant support cues from the memory and incorporate appearance priors into its representation. For a query image $I ^ { q } ,$ , the image encoder extracts the features $\bar { \mathbf { F } ^ { q } } \in \mathbb { R } ^ { h w \times C }$ . The core of our guidance mechanism lies in the fusion of ${ \pmb F } ^ { q }$ with the structured memory $\boldsymbol { B }$ in the memory attention $M A ( \cdot )$ mechanism which is implemented as the cross-attention as:

$$
Q ^ { m a } = F ^ { q } W _ { q } ^ { \top } , K ^ { m a } = { \cal B } W _ { k } ^ { \top } , V ^ { m a } = { \cal B } W _ { v } ^ { \top } ,\tag{3}
$$

$$
A ^ { m a } = \mathrm { s o f t m a x } \Big ( Q ^ { m a } K ^ { m a \top } / \sqrt { d } \Big ) , \hat { F } ^ { q } = M A ( F ^ { q } , \mathcal { B } ) = A ^ { m a } V ^ { m a } ,
$$

where d is the attention dimension. In this way, the query tokens serve as queries, while the concatenated support memories in $\boldsymbol { B }$ act as the keys and values for feature aggregation. The resulting enhanced query representation ${ \hat { F } } ^ { q }$ is then reshaped back to the spatial layout and fed into the mask decoder of SAM 3 to produce the segmentation mask logits $\hat { M } ^ { q }$

## 3.4 Cross-Modal Alignment

Textual Feature Extraction. While the support memory captures appearancelevel cues from annotated support images, semantic priors provide complementary category-level information. To incorporate such semantic guidance, we extract textual representations using the text encoder TE of SAM 3. Given a textual prompt $T _ { \mathrm { : } }$ the text encoder produces a semantic embedding:

$$
\pmb { F } ^ { t } = \pmb { T } \pmb { E } ( T ) \in \mathbb { R } ^ { L \times C } ,\tag{4}
$$

where L denotes the number of text tokens. We keep the text encoder frozen to preserve its generalization ability. The resulting embedding encodes categorylevel semantics and serves as a complementary source of guidance for segmentation. Compared with the support memory, which mainly conveys instance-level visual cues, the textual feature provides more abstract semantic information and is generally more robust to appearance variations across domains.

Grounding Text with Visual Memory. In cross-modal few-shot segmentation, semantic priors often fail to align with visual patterns under significant domain shifts, leading to inconsistent predictions. To address this issue, we introduce a lightweight cross-attention module that aligns the textual features with the memory-enhanced visual representations. Let the textual feature be denoted by $F ^ { t } \in \mathbb { R } ^ { L \times C }$ , and let the memory-enhanced query feature be denoted by $\hat { \pmb { F } } ^ { q } \in \mathbb { R } ^ { h w \times C }$ . We construct the cross-modal alignment module $C M ( \cdot )$ in the cross-attention form:

$$
Q ^ { t a } = F ^ { t } \tilde { W } _ { q } ^ { \top } , K ^ { t a } = \hat { F } ^ { q } \tilde { W } k ^ { \top } , V ^ { t a } = \hat { F } ^ { q } \tilde { W } _ { v } ^ { \top } ,\tag{5}
$$

$$
A ^ { t a } = \mathrm { s o f t m a x } \Big ( Q ^ { t a } K ^ { t a \top } / \sqrt { d } \Big ) , \hat { F } ^ { t } = C M ( F ^ { t } , \hat { F } ^ { q } ) = A ^ { t a } V ^ { t a } ,\tag{6}
$$

where d denotes the attention dimension. In this way, the textual feature serves as the query, while the memory-enhanced visual representation provides the keys and values. This operation conditions the semantic representation on support-aware visual cues, producing a visually grounded textual representation. The updated feature is then processed by a normalization layer and a feedforward network, and finally fed into the frozen detector of SAM 3 to generate the mask logits for the textual branch, denoted by $\hat { M } ^ { t }$

To encourage consistent cross-modal alignment, we supervise the textual branch using the output of the memory-based visual prediction. Specifically, the predicted mask logits $\hat { M } ^ { q }$ are binarized to obtain a pseudo mask $M ^ { p } .$ , which serves as supervision for the textual branch. The training objective is defined as

$$
\mathcal { L } _ { \mathrm { C E } } = \mathrm { B C E } ( \hat { M } ^ { t } , M ^ { p } ) .\tag{7}
$$

This supervision encourages the textual representation to converge toward the memory-guided visual prediction, thereby reducing cross-modal discrepancies and improving robustness under severe domain shifts.

## 3.5 Inference

Dual-Branch Prediction. During inference, we combine predictions from the visual memory and textual branches using a confidence-aware fusion strategy. Let $\hat { M } ^ { q }$ and $\hat { M } ^ { t }$ denote their mask logits. The confidence of each branch is estimated by the average logit magnitude:

$$
\pmb { s } _ { q } = \mathrm { G A P } ( | \hat { \pmb { M } } ^ { q } | ) , \qquad \pmb { s } _ { t } = \mathrm { G A P } ( | \hat { \pmb { M } } ^ { t } | ) ,\tag{8}
$$

where $\mathrm { G A P } ( \cdot )$ denotes global average pooling. The normalized fusion weights are

$$
{ \pmb w } _ { q } = \frac { { \pmb s } _ { q } } { { \pmb s } _ { q } + { \pmb s } _ { t } } , \qquad { \pmb w } _ { t } = \frac { { \pmb s } _ { t } } { { \pmb s } _ { q } + { \pmb s } _ { t } } .\tag{9}
$$

The fused prediction is computed as

$$
\hat { M } = { \pmb w } _ { q } \cdot \sigma ( \hat { M } ^ { q } ) + { \pmb w } _ { t } \cdot \sigma ( \hat { M } ^ { t } ) ,\tag{10}
$$

where $\sigma ( \cdot )$ denotes the Sigmoid function. The initial binary mask is obtained as $\tilde { M } = \mathbb { I } ( \hat { M } > 0 . 5 )$

Pseudo-Label Memory Refinement. We further use the initial prediction M<sup>˜</sup> as a pseudo label to construct query-adaptive memory:

$$
M e m _ { p } = M E ( F ^ { q } , \tilde { M } ) ,\tag{11}
$$

where $\mathrm { M E } ( \cdot )$ denotes the memory encoder of SAM 3. The support memory is then augmented with the pseudo-labeled query memory:

$$
\mathcal { B } ^ { \prime } = \mathrm { C o n c a t } ( M e m _ { 1 } , M e m _ { 2 } , \dots , M e m _ { K } , M e m _ { p } ) .\tag{12}
$$

The updated memory bank $B ^ { \prime }$ is fed back into the visual memory branch to refine the query representation and prediction.

## 4 Experiment

## 4.1 Datasets

We evaluate our method on four widely used benchmarks covering diverse target domains: ISIC2018 [5], Chest X-ray [2], DeepGlobe [7], and FSS-1000 [16]. ISIC2018 contains 2,596 images from three skin lesion categories and presents challenges in fine-grained texture discrimination. Chest X-ray consists of 566 annotated lung X-ray images characterized by low contrast and structural noise. DeepGlobe provides high-resolution satellite imagery for land-cover segmentation. Following [14], we exclude the ‘unknown’ class and use the remaining six categories (5,666 images). FSS-1000 contains 1,000 natural object classes, and we evaluate on its oficial test set of 2,400 images from 240 classes.

## 4.2 Implementation Details.

To preserve the foundational segmenting capabilities, we keep all SAM 3 components frozen and adopt a parameter-eficient adaptation strategy by optimizing only the proposed cross-modal alignment module. We adopt the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and supervise the model using binary cross-entropy loss. To align with the original image input setting of SAM 3, images are resized to $1 0 2 4 \times 1 0 2 4$ , with a batch size of 1. For text input, we use template-based prompts constructed from category names Following the standard evaluation protocol in prior work [34], we use mean Intersection over Union (mIoU) as the evaluation metric.

## 4.3 Main Results

Comparison with Existing Methods. The proposed method is comprehensively compared with state-of-the-art approaches for both traditional few-shot semantic segmentation and cross-domain few-shot semantic segmentation. The results are presented in Table 1. Traditional FSS methods perform competitively on FSS-1000, which mainly contains natural images, but generally sufer substantial performance degradation on specialized target domains. In contrast, ViTA-SAM achieves the best average performance, surpassing the previous best results by 6.13% and 4.69% in average mIoU under the 1-shot and 5-shot settings, respectively. It achieves the best performance on Chest X-ray, ISIC, and FSS-1000, demonstrating strong generalization across diverse target domains. Particularly, notable gains are observed on the medical datasets Chest X-ray and ISIC, demonstrating the efectiveness of our method under substantial domain shifts. On the challenging Deepglobe benchmark, ViTA-SAM still achieves the best 1-shot result and remains highly competitive in the 5-shot setting. Moreover, the larger overall improvement under the 1-shot setting suggests that ViTA-SAM is particularly efective when only limited support supervision is available. Overall, these results demonstrate the robustness and generalization capability of our method across diverse domains and few-shot settings.

Table 1: Quantitative comparison of the proposed ViTA-SAM with existing few-shot and cross-domain few-shot segmentation methods in mIoU (%).
<table><tr><td rowspan=2 colspan=5>Methods</td><td rowspan=1 colspan=4>Chest X-ray</td><td rowspan=1 colspan=1>ISIC</td><td rowspan=1 colspan=1>FSS-1000</td><td rowspan=1 colspan=1>Deepglobe</td><td rowspan=1 colspan=1>Average</td></tr><tr><td rowspan=1 colspan=4>1-shot 5-shot</td><td rowspan=1 colspan=1>1-shot 5-shot</td><td rowspan=1 colspan=1>1-shot 5-shot</td><td rowspan=1 colspan=1>1-shot 5-shot</td><td rowspan=1 colspan=1>1-shot 5-shot</td></tr><tr><td rowspan=1 colspan=13>Traditional Few-shot Semantic: Segmentation Methods</td></tr><tr><td rowspan=1 colspan=5>AMP [28]</td><td rowspan=1 colspan=4>51.2353.04</td><td rowspan=1 colspan=1>28.4230.41</td><td rowspan=1 colspan=1>57.1859.24</td><td rowspan=1 colspan=1>37.6140.61</td><td rowspan=1 colspan=1>43.6145.83</td></tr><tr><td rowspan=1 colspan=5>PANet [34]</td><td rowspan=1 colspan=4>57.7569.31</td><td rowspan=1 colspan=1>25.2933.99</td><td rowspan=1 colspan=1>69.1571.68</td><td rowspan=1 colspan=1>36.5545.43</td><td rowspan=1 colspan=1>47.1955.10</td></tr><tr><td rowspan=1 colspan=5>CANet [39]</td><td rowspan=1 colspan=4>28.3528.62</td><td rowspan=1 colspan=1>26.1229.22</td><td rowspan=1 colspan=1>70.6772.03</td><td rowspan=1 colspan=1>20.3723.03</td><td rowspan=1 colspan=1>36.3838.23</td></tr><tr><td rowspan=1 colspan=5>RPMMs [36]</td><td rowspan=1 colspan=4>30.1130.15</td><td rowspan=1 colspan=1>18.0220.04</td><td rowspan=1 colspan=1>65.1267.06</td><td rowspan=2 colspan=1>12.9913.4710.5211.86</td><td rowspan=2 colspan=1>31.5632.6833.5134.74</td></tr><tr><td rowspan=1 colspan=3>PFENet [32]</td><td rowspan=1 colspan=3>Net [32]</td><td rowspan=1 colspan=2>27.</td><td rowspan=1 colspan=1>7.22</td><td rowspan=1 colspan=1>27.57</td><td rowspan=1 colspan=1>25.7727.35</td><td rowspan=1 colspan=1>70.5272.16</td></tr><tr><td rowspan=2 colspan=5>HSNet [21]PerSAM [41]</td><td rowspan=1 colspan=3>SNet [21]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>51.8</td><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=1>854.36</td><td rowspan=1 colspan=1>31.9635.33</td></tr><tr><td rowspan=1 colspan=4>29.9530.05</td><td rowspan=1 colspan=1>23.2725.33</td><td rowspan=1 colspan=1>60.9266.53</td><td rowspan=1 colspan=1>36.0840.65</td><td rowspan=1 colspan=1>37.5640.64</td></tr><tr><td rowspan=1 colspan=13>Cross-domain Few-shot Semantic Segmentation Methods</td></tr><tr><td rowspan=1 colspan=5>PATNet [14]</td><td rowspan=1 colspan=4>66.6170.20</td><td rowspan=1 colspan=1>41.1653.58</td><td rowspan=1 colspan=1>78.5981.23</td><td rowspan=1 colspan=1>37.8942.97</td><td rowspan=1 colspan=1>56.0661.99</td></tr><tr><td rowspan=5 colspan=5>DMTNet [4]APSeg [10]DR-Adapter [30]ABCDFSS [11]ISA [9]TVGTANet [18]DFN [33]</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>73.7477.30</td><td rowspan=1 colspan=1>43.5552.30</td><td rowspan=1 colspan=1>81.5283.28</td><td rowspan=1 colspan=1>40.1451.17</td></tr><tr><td rowspan=4 colspan=4>84.1084.5082.3582.3579.8081.4083.4086.3084.5887.2783.1887.14</td><td rowspan=3 colspan=1>45.4353.9840.7748.8745.7053.3037.2056.1047.2158.75</td><td rowspan=2 colspan=1>79.7181.9079.0580.4074.6076.2078.8086.00</td><td rowspan=1 colspan=1>35.9439.98</td><td rowspan=1 colspan=1>61.3065.09</td></tr><tr><td rowspan=2 colspan=1>41.2950.1242.6049.0044.3052.1042.0450.67</td><td rowspan=1 colspan=1>60.8665.4260.6864.9860.9370.13</td></tr><tr><td rowspan=1 colspan=1>78.3281.44</td><td rowspan=1 colspan=1>63.0469.53</td></tr><tr><td rowspan=1 colspan=1>50.3658.53</td><td rowspan=1 colspan=1>82.9785.72</td><td rowspan=1 colspan=1>39.4547.67</td><td rowspan=1 colspan=1>63.9969.77</td></tr><tr><td rowspan=1 colspan=5>ViTA-SAM (Ours)</td><td rowspan=1 colspan=4>|90.53 91.97</td><td rowspan=1 colspan=1>|57.80 66.42</td><td rowspan=1 colspan=1>|87.71 88.83</td><td rowspan=1 colspan=1>44.4252.04</td><td rowspan=1 colspan=1>|70.12 74.82</td></tr></table>

Qualitative Analysis of Diferent Modules. Figure 3 presents qualitative comparisons of output logit maps from diferent modules. When target regions are relatively regular and semantically clear, such as in Chest X-ray and FSS, both branches respond efectively, while the textual branch produces smoother and more concentrated activations. On ISIC, both branches can localize the lesion, but the visual branch yields a more compact and cleaner response. On Deepglobe, the textual branch produces scattered activations over irrelevant regions, whereas the visual branch provides more stable localization despite residual background noise. These observations highlight the complementary roles of the two branches: the textual branch ofers stronger semantic emphasis, while the visual branch provides more reliable spatial grounding in complex scenarios.

## 4.4 Ablation Study

Efectiveness of Specific Modules. Table 2 analyzes the individual and combined contributions of the core components in ViTA-SAM. Results are reported under the 1-shot setting. The visual memory module (Visual) alone achieves strong performance by efectively transferring appearance-level cues from support to query via the memory attention mechanism of SAM 3. The textual branch, which produces memory-aligned semantic features through cross-modal alignment, also achieves competitive results, indicating that semantic guidance provides complementary information beyond pure visual matching. Distinct behaviors are observed across datasets. The textual branch performs better on the Chest X-ray dataset, while the visual memory module shows stronger performance on other benchmarks, highlighting its advantage in modeling complex visual patterns and fine-grained structures. Combining both components consistently improves performance across all benchmarks, confirming their complementary roles in aligning appearance and semantic representations. Further incorporating pseudo-label refinement (Refinement) leads to additional gains, demonstrating the efectiveness of leveraging semantically confident predictions to enhance the visual memory.

![](images/191e1e83113be7b5ca1ce8b5299cf771bfd728ea2547bfbd993eaf84d2651204.jpg)  
Fig. 3: Visualization of output logit maps of the textual and visual modules. From left to right: input images, logits from the aligned text module (T-branch), logits from the visual memory module (V-branch), final prediction, and GT masks.

Table 2: Ablation study of the proposed modules. Visual: visual memory branch; Textual: aligned textual branch; Refine: pseudo-label refinement.
<table><tr><td>Visual Textual Refine|</td><td></td><td></td><td>|CXR ISIC</td><td>FSS DGL</td></tr><tr><td>√</td><td>x</td><td>x</td><td>87.28 53.36</td><td>85.37 43.10</td></tr><tr><td>x</td><td>√</td><td>x</td><td>88.64 46.95</td><td>83.90 36.49</td></tr><tr><td>√</td><td>√</td><td>x</td><td>90.29 56.98</td><td>87.6043.90</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>90.53 57.80 87.71 44.42</td></tr></table>

Multi-step Pseudo-label Refinement. To examine the efectiveness of pseudolabel refinement, we conduct a multi-step study. Starting from the initial prediction M<sup>˜</sup> , the predicted mask is used as a pseudo-label and re-encoded into the memory to construct a query-adaptive memory. The updated memory is then combined with query features to produce a refined prediction. This process can be repeated iteratively, resulting in $\tilde { M } _ { n }$ after n steps and progressively refining the query representations. We evaluate the impact of diferent refinement steps on performance, as reported in Table 3. A single refinement step (n = 1)

Table 3: Efect of iterative pseudo-label refinement.
<table><tr><td>Iteration|</td><td>|CXR ISIC</td><td>FSS</td><td>DGL</td></tr><tr><td>0</td><td>90.29 56.98</td><td>87.60</td><td>43.90</td></tr><tr><td>1</td><td>90.53 57.80 87.71 44.42</td><td></td><td></td></tr><tr><td>2</td><td>90.59 57.79</td><td>87.52</td><td>44.11</td></tr><tr><td>3</td><td>90.6157.20</td><td>087.1344.09</td><td></td></tr></table>

consistently improves performance across all benchmarks. This gain comes from introducing query-derived pseudo-labels, which augment the support-only memory with query-specific cues and improve alignment with the target distribution. Increasing the number of refinement steps does not yield consistent gains. While slight improvements are observed on Chest X-ray, performance saturates or degrades on other datasets. This behavior is mainly due to error accumulation in pseudo-labels: repeatedly encoding self-predicted masks introduces noisy signals into the memory, degrading feature alignment. This efect is less pronounced on Chest X-ray, where pseudo-labels are more reliable. In contrast, datasets with lower pseudo-label quality are more susceptible to noise accumulation, resulting in performance degradation with multiple refinement steps. Therefore, we adopt a single refinement step (n = 1) as the default setting.

## 4.5 Discussion

Efectiveness of Cross-modal Alignment. SAM 3 supports text-driven segmentation by taking textual prompts as input and predicting corresponding regions. However, under significant domain shifts, such as medical and satellite imagery, textual semantics can be poorly aligned with target-domain visual patterns. We therefore compare the raw outputs of the SAM 3 text branch with those obtained after introducing the proposed cross-modal alignment mechanism, which incorporates memory-enhanced visual representations. As shown in Table 4, the original SAM 3 achieves moderate performance on FSS, which contains relatively common object categories, but performs poorly on the other datasets. In particular, the results on ISIC and Chest X-ray are close to zero, indicating that textual prompts alone provide insuficient guidance under substantial domain shifts. After introducing the proposed alignment strategy, performance improves consistently across all benchmarks, demonstrating the importance of grounding textual representations with memory-enhanced visual features for reliable cross-domain segmentation.

Visualization of Text–Visual Alignment. To analyze cross-modal alignment, we visualize response maps across benchmarks in Figure 4. The first row displays input images, while the subsequent rows compare the text-visual response maps of raw SAM 3 and our aligned model. Without cross-modal alignment, SAM 3 produces weak and spatially difuse activations that fail to localize target regions, particularly under large domain shifts. In contrast, our method yields more concentrated and consistent activations with clearer foreground–background separation. These results indicate that aligning textual semantics with memory-enhanced visual representations produces more reliable responses and improves spatial localization.

Table 4: Comparison of the SAM 3 textual branch performance before and after the proposed cross-modal alignment.
<table><tr><td>Method</td><td>|CXR ISIC FSS DGL</td></tr><tr><td>SAM 3 (Text)</td><td>0.66 0.65 51.04 6.38</td></tr><tr><td>SAM 3 (Text + Alignment)</td><td>88.64 46.95 83.90 36.49</td></tr></table>

![](images/44d1cbda729e887413d3c4bb8fd9e966bbc103e3dcdf28b30cd4192e5d798c43.jpg)  
Fig. 4: Comparison of response maps from raw SAM 3 and ViTA-SAM.

## 5 Conclusion

In this work, we present ViTA-SAM, a cross-modal few-shot semantic segmentation framework built upon the SAM 3 foundation model. Our approach integrates visual memory with textual features, where the visual memory provides support-aware structural guidance and the textual features introduce complementary semantic priors. We introduce a lightweight cross-modal interaction mechanism that conditions textual features on memory-enhanced visual representations, producing more consistent and visually grounded predictions. In addition, we introduce a pseudo-label refinement strategy that updates the memory using predicted query masks, enabling query-adaptive representation learning. Extensive experiments on multiple benchmarks show that ViTA-SAM consistently improves over existing methods.

Acknowledgments. This work was partially supported by the Grains Research and Development Corporation (UOQ2301-010OPX, USQ2605-002BGX) and by The University of Queensland (DVCR2201A).

## References

1. Cai, Y., Shang, Y., Yin, J.: Multidan: Unsupervised, multistage, multisource and multitarget domain adaptation for semantic segmentation of remote sensing images. In: Cai, J., Kankanhalli, M.S., Prabhakaran, B., Boll, S., Subramanian, R., Zheng, L., Singh, V.K., César, P., Xie, L., Xu, D. (eds.) Proceedings of the 32nd ACM International Conference on Multimedia, MM 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November 2024. pp. 1168–1177. ACM (2024). https://doi.org/10.1145/3664647.3681494, https://doi.org/10.1145/3664647.3681494

2. Candemir, S., Jaeger, S., Palaniappan, K., Musco, J.P., Singh, R.K., Xue, Z., Karargyris, A., Antani, S., Thoma, G., McDonald, C.J.: Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. IEEE transactions on medical imaging 33(2), 577–590 (2013)

3. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., et al.: Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 (2025)

4. Chen, J., Quan, R., Qin, J.: Cross-domain few-shot semantic segmentation via doubly matching transformation. In: Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24. pp. 641–649 (2024)

5. Codella, N., Rotemberg, V., Tschandl, P., Celebi, M.E., Dusza, S., Gutman, D., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., et al.: Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic). arXiv preprint arXiv:1902.03368 (2019)

6. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene understanding. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3213–3223 (2016)

7. Demir, I., Koperski, K., Lindenbaum, D., Pang, G., Huang, J., Basu, S., Hughes, F., Tuia, D., Raskar, R.: Deepglobe 2018: A challenge to parse the earth through satellite images. In: Proceedings of the IEEE conference on computer vision and pattern recognition workshops. pp. 172–181 (2018)

8. Dong, N., Xing, E.P.: Few-shot semantic segmentation with prototype learning. In: BMVC. vol. 3, p. 4 (2018)

9. Fan, Q., Liu, K., Liu, N., Cholakkal, H., Anwer, R.M., Li, W., Gao, Y.: Adapting indomain few-shot segmentation to new domains without source domain retraining. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 21429–21439 (October 2025)

10. He, W., Zhang, Y., Zhuo, W., Shen, L., Yang, J., Deng, S., Sun, L.: Apseg: Autoprompt network for cross-domain few-shot semantic segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 23762–23772 (2024)

11. Herzog, J.: Adapt before comparison: A new perspective on cross-domain few-shot segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 23605–23615 (2024)

12. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4015–4026 (2023)

13. Lang, C., Cheng, G., Tu, B., Han, J.: Learning what not to segment: A new perspective on few-shot segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8057–8067 (2022)

14. Lei, S., Zhang, X., He, J., Chen, F., Du, B., Lu, C.T.: Cross-domain few-shot semantic segmentation. In: European Conference on Computer Vision. pp. 73–90. Springer (2022)

15. Li, G., Jampani, V., Sevilla-Lara, L., Sun, D., Kim, J., Kim, J.: Adaptive prototype learning and allocation for few-shot segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8334–8343 (2021)

16. Li, X., Wei, T., Chen, Y.P., Tai, Y.W., Tang, C.K.: Fss-1000: A 1000-class dataset for few-shot segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2869–2878 (2020)

17. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36 (2024)

18. Liu, J., Qiu, W., Wei, H.: Textual and visual guided task adaptation for source-free cross-domain few-shot segmentation. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 5150–5159 (2025)

19. Liu, Y., Liu, N., Yao, X., Han, J.: Intermediate prototype mining transformer for few-shot semantic segmentation. Advances in Neural Information Processing Systems 35, 38020–38031 (2022)

20. Lu, Z., He, S., Zhu, X., Zhang, L., Song, Y.Z., Xiang, T.: Simpler is better: Few-shot semantic segmentation with classifier weight transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8741–8750 (2021)

21. Min, J., Kang, D., Cho, M.: Hypercorrelation squeeze for few-shot segmentation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 6941–6952 (2021)

22. Nie, J., Xing, Y., Zhang, G., Yan, P., Xiao, A., Tan, Y.P., Kot, A.C., Lu, S.: Crossdomain few-shot segmentation via iterative support-query correspondence mining. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3380–3390 (2024)

23. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

24. Peng, B., Tian, Z., Wu, X., Wang, C., Liu, S., Su, J., Jia, J.: Hierarchical dense correlation distillation for few-shot segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 23641–23651 (2023)

25. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021)

26. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024)

27. Shi, J., Shang, C., Sun, Z., Yu, L., Yang, X., Yan, Z.: PASSION: towards efective incomplete multi-modal medical image segmentation with imbalanced missing rates. In: Cai, J., Kankanhalli, M.S., Prabhakaran, B., Boll, S., Subramanian, R., Zheng, L., Singh, V.K., César, P., Xie, L., Xu, D. (eds.) Proceedings of the 32nd ACM International Conference on Multimedia, MM 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 Novem-

ber 2024. pp. 456–465. ACM (2024). https://doi.org/10.1145/3664647.3681543, https://doi.org/10.1145/3664647.3681543

28. Siam, M., Oreshkin, B.N., Jagersand, M.: Amp: Adaptive masked proxies for fewshot segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5249–5258 (2019)

29. Snell, J., Swersky, K., Zemel, R.: Prototypical networks for few-shot learning. Advances in neural information processing systems 30 (2017)

30. Su, J., Fan, Q., Pei, W., Lu, G., Chen, F.: Domain-rectifying adapter for crossdomain few-shot segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 24036–24045 (2024)

31. Sun, P., Zhang, H., Hu, D.: Unveiling and mitigating bias in audio visual segmentation. In: Cai, J., Kankanhalli, M.S., Prabhakaran, B., Boll, S., Subramanian, R., Zheng, L., Singh, V.K., César, P., Xie, L., Xu, D. (eds.) Proceedings of the 32nd ACM International Conference on Multimedia, MM 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November 2024. pp. 7259–7268. ACM (2024). https://doi.org/10.1145/3664647.3680802, https://doi.org/10.1145/3664647.3680802

32. Tian, Z., Zhao, H., Shu, M., Yang, Z., Li, R., Jia, J.: Prior guided feature enrichment network for few-shot segmentation. IEEE transactions on pattern analysis and machine intelligence 44(2), 1050–1065 (2020)

33. Tong, J., Ma, R., Zou, Y., Chen, G., Li, Y., Li, R.: Adapter naturally serves as decoupler for cross-domain few-shot semantic segmentation. In: Forty-second International Conference on Machine Learning (2025)

34. Wang, K., Liew, J.H., Zou, Y., Zhou, D., Feng, J.: Panet: Few-shot image semantic segmentation with prototype alignment. In: proceedings of the IEEE/CVF international conference on computer vision. pp. 9197–9206 (2019)

35. Wu, Y., Li, X., Li, J., Yang, K., Zhu, P., Zhang, S.: DINO is also a semantic guider: Exploiting class-aware afinity for weakly supervised semantic segmentation. In: Cai, J., Kankanhalli, M.S., Prabhakaran, B., Boll, S., Subramanian, R., Zheng, L., Singh, V.K., César, P., Xie, L., Xu, D. (eds.) Proceedings of the 32nd ACM International Conference on Multimedia, MM 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November 2024. pp. 1389–1397. ACM (2024). https://doi.org/10.1145/3664647.3681710, https://doi.org/10.1145/3664647.3681710

36. Yang, B., Liu, C., Li, B., Jiao, J., Ye, Q.: Prototype mixture models for few-shot semantic segmentation. In: European conference on computer vision. pp. 763–778. Springer (2020)

37. Yue, P., Lin, J., Zhang, S., Hu, J., Lu, Y., Niu, H., Ding, H., Zhang, Y., Jiang, G., Cao, L., Ji, R.: Adaptive selection based referring image segmentation. In: Cai, J., Kankanhalli, M.S., Prabhakaran, B., Boll, S., Subramanian, R., Zheng, L., Singh, V.K., César, P., Xie, L., Xu, D. (eds.) Proceedings of the 32nd ACM International Conference on Multimedia, MM 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November 2024. pp. 1101–1110. ACM (2024). https://doi.org/10.1145/3664647.3680850, https://doi.org/10.1145/3664647.3680850

38. Zhang, B., Xiao, J., Qin, T.: Self-guided and cross-guided learning for few-shot segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8312–8321 (2021)

39. Zhang, C., Lin, G., Liu, F., Yao, R., Shen, C.: Canet: Class-agnostic segmentation networks with iterative refinement and attentive few-shot learning. In: Proceedings

of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5217–5226 (2019)

40. Zhang, G., Kang, G., Yang, Y., Wei, Y.: Few-shot segmentation via cycle-consistent transformer. Advances in Neural Information Processing Systems 34, 21984–21996 (2021)

41. Zhang, R., Jiang, Z., Guo, Z., Yan, S., Pan, J., Dong, H., Gao, P., Li, H.: Personalize segment anything model with one shot. arXiv preprint arXiv:2305.03048 (2023)

42. Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., Torralba, A.: Scene parsing through ade20k dataset. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 633–641 (2017)