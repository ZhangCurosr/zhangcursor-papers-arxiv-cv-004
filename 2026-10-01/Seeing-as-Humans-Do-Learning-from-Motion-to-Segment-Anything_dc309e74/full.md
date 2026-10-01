# Seeing as Humans Do: Learning from Motion to Segment Anything Without Supervision

Weijian Jian<sup>1,∗</sup> , Xiaoyue Zhang<sup>2,∗</sup> , Bin Xiao<sup>3</sup> , Chunyu Xie<sup>1</sup> , Yixiao He<sup>4</sup> , Yutao Liu<sup>1</sup> , Dawei Leng<sup>1,⊠</sup> , and Yuhui Yin<sup>1</sup>

<sup>1</sup> 360 AI Research

<sup>2</sup> Independent Researcher

<sup>3</sup> University of Ottawa

4 Beijing University of Posts and Telecommunications

Abstract. The Segment Anything Model (SAM) relies heavily on massive manual annotations, creating a fundamental bottleneck for model scaling. While unsupervised methods attempt to learn object concepts from motion, they typically overfit to moving entities, lacking both multigranularity understanding and the ability to generalize to static objects. To overcome this, we introduce Motion-Grounded Segment Anything (MoSA), a highly scalable unsupervised framework that learns a transferable objectness prior from unlabeled videos. MoSA operates in three progressive stages: (1) automatically generating multi-granularity motion pseudo-labels from large-scale video data; (2) training a Perceptual Grouping Model (PGM) via contrastive learning to internalize a generalized, appearance-driven concept of objects; and (3) transferring this learned prior into a prompt-guided architecture for segment-anythingstyle inference on images. Extensive zero-shot evaluations across seven challenging benchmarks (e.g., COCO and ADE20K) demonstrate that MoSA significantly outperforms existing unsupervised methods. Notably, despite using zero manual annotations, MoSA achieves segmentation performance comparable to the fully supervised SAM. Our findings reveal that harnessing large-scale unlabeled motion is a feasible and highly scalable alternative to annotation-driven segment-anything pipelines. Code is available at https://github.com/360CVGroup/MoSA.

Keywords: Unsupervised Learning · Segment Anything

## 1 Introduction

The Segment Anything Model (SAM) [15] has established a new standard for zero-shot generalization in computer vision. However, its success heavily relies on the massive, manually annotated SA-1B dataset. This dependence on costly supervision creates a fundamental bottleneck for further model scaling. Consequently, annotation-free unsupervised learning has emerged as a crucial direction for scalable, general-purpose segmentation.

![](images/0d8188a918b646bf1c1bfd0d8dddb24ab1cf30fe2ed4cd7a365f42d419100a05.jpg)  
Fig. 1: Overview of our proposed MoSA. The top-left and top-right panels respectively show qualitative comparisons of MoSA against the fully supervised SAM [15] and the unsupervised UnSAM [38] on point-based promptable segmentation and wholeimage segmentation. The middle-right panel shows that MoSA achieves SOTA performance compared to other unsupervised segmentation methods. The bottom panel illustrates the pipeline of MoSA’s three-stage unsupervised framework. Note that Un-SAM is not strictly unsupervised, as it utilizes the supervised CascadePSP [7] refiner.

Existing unsupervised segmentation methods generally fall into two categories: image-based and video-based approaches. Image-based methods, such as CutLER [36] and UnSAM [38], leverage self-supervised representations (e.g., DINO [4]) to perform feature clustering or pseudo-label generation, achieving competitive unsupervised segmentation performance. Nevertheless, these approaches infer objectness primarily from static appearance cues, such as texture, color, and shape, overlooking the cognitive priors humans develop through continuous observation of the dynamic world.

Research in developmental psychology, particularly Gestalt theory [16, 17], suggests that the human ability to group objects is largely driven by motion cues. Video-based methods naturally exploit these motion cues. Early approaches typically focus on detecting moving foreground regions, often producing binary masks without explicitly modeling multiple object instances or detailed structures [20, 41, 43]. Recent advances extend unsupervised video object segmentation (U-VOS) toward instance- and scene-level understanding. For example, OCLR [40] proposes an object-centric model trained on synthetic data to discover and track multiple moving objects. Other methods rely on videos during training but can perform inference on static images without requiring motion information. Methods such as RCF [22] and DyStaB [42] exploit motion-induced grouping to generate object masks, often combining motion and appearance refinement within bootstrapping frameworks. Furthermore, approaches like DIOD [14] integrate motion-guided discovery with self-distillation to improve pseudo-label quality. Although these methods prove that motion is a powerful supervisory signal for object concepts, they face two major limitations. First, they easily overfit to moving objects, struggling to generalize to static or weakly moving entities (e.g., the roof tiles in Fig. 1). Second, they are largely confined to instance-level segmentation, lacking multi-granularity understanding. To truly emulate human perception, a scalable model must generalize from moving entities to anything in static scenes, expanding its object vocabulary from growing unlabeled videos.

Motivated by this insight, we introduce Motion-Grounded Segment Anything (MoSA), a three-stage unsupervised framework designed to learn a transferable, general-purpose segmentation capability from unlabeled videos (see Fig. 1). In the first stage, we develop a Multi-Granularity Motion Segmentation (MGMS) pipeline. Pre-trained on synthetic data and subsequently applied to real-world unlabeled videos, MGMS generates high-quality motion-based pseudo-labels at scale without manual annotation. In the second stage, we train a Perceptual Grouping Model (PGM) using these motion pseudo-labels. This stage encourages the model to internalize a generalized, appearance-driven notion of objectness that no longer depends on motion cues. Finally, an eficient adaptation stage transfers the learned objectness prior to a prompt-guided, high-resolution segmentation architecture, enabling strong zero-shot segment-anything performance. We conduct extensive evaluations on seven challenging segmentation benchmarks, including COCO [23] and ADE20K [46]. Experimental results demon strate that MoSA consistently outperforms existing unsupervised segmentation methods across all benchmarks. Notably, despite being trained entirely without manual annotations, MoSA achieves zero-shot segmentation performance comparable to SAM, which relies on large-scale supervised data. These findings suggest that learning general object perception from large-scale unlabeled motion is both feasible and scalable, ofering a promising path toward foundational general-purpose segmentation. Our main contributions are as follows:

– We propose MoSA, a motion-grounded segmentation framework that learns from unlabeled videos to acquire a general-purpose, multi-granularity segmentation capability without manual annotation.

– We introduce a Perceptual Grouping Model (PGM) with a contrastive learning objective that allows the model to generalize beyond motion-specific cues, facilitating the discovery and segmentation of unseen static objects.

– MoSA achieves new state-of-the-art performance in unsupervised segmentation across extensive experiments, proving that learning general objectness from motion is a scalable and practical route to developing foundational vision models.

## 2 Related Work

Unsupervised Image Segmentation Unsupervised image segmentation aims to automatically identify and segment objects from static images without manual annotations [13, 34]. Methods such as LOST [31] and TokenCut [39] leverage patch-level features from pretrained vision transformers (e.g., DINO family [4,27,32]) to localize salient objects via feature afinity and graph-based partitioning. CutLER [36] introduces a cut-and-learn pipeline that discovers pseudomasks and trains segmentation models in a self-supervised manner, demonstrating strong multi-instance detection and segmentation capabilities. UnSAM [38] adopts a divide-and-conquer strategy that combines hierarchical grouping and mask refinement to generate multi-granularity pseudo-labels, enabling both automatic and promptable segmentation. A common thread in these image-based approaches is their reliance on DINO features and the subsequent use of hierarchical clustering to generate pseudo-labels. However, they may lack the robust, world-grounded prior knowledge that humans develop through extensive, longterm observation.

Motion-to-Objectness Bootstrapping A growing body of research leverages motion as a natural supervisory signal for self-supervised object discovery. The standard paradigm extracts motion segments from videos to train static image models, allowing the concept of “objects” to generalize beyond initially moving regions. For instance, DyStaB [42] partitions the motion field by minimizing the mutual information between segments, then employs a dynamic-static bootstrapping strategy that iteratively alternates between motion-based segmentation and static model learning. RCF [22] combines relaxed common-fate grouping with appearance refinement to generate object masks. Furthermore, Bao et al. [2] utilize a slot-attention framework to discover independently moving objects, while DIOD [14] integrates motion guidance with self-distillation to enhance pseudolabel quality. However, existing methods remain confined to narrow, specialized scenarios. Their training objectives focus heavily on extracting precise instance masks rather than learning a universal, transferable “object prior” necessary for foundational “segment-anything” capabilities. Furthermore, they are strictly restricted to instance-level predictions and cannot comprehend objects across multiple granularities (e.g., distinguishing a part from a whole). Consequently, their learned representations sufer from a severe motion bias—they overfit to motionsalient regions and fail on static or rarely moving objects. Ultimately, building a scalable, general-purpose segmentation model solely from motion cues remains an open challenge.

## 3 Motion Pseudo-Label Generation

To learn object concepts from motion without manual supervision, MoSA begins with a Multi-Granularity Motion Segmentation (MGMS) pipeline that generates high-quality pseudo-labels from unlabeled videos. Within the MGMS pipeline, motion segmentation modules are first trained on synthetic video data. These modules are subsequently applied to large-scale real-world videos to produce a large set of candidate masks (see Fig. 2). Finally, a dedicated quality assessment metric is introduced to filter the generated candidates, resulting in a high-quality, multi-granularity motion pseudo-label dataset for subsequent-stage training.

![](images/5471e6709a6c54cde0398b97debdd2df4000e52d4559ca6b50609903c97f6fdf.jpg)  
Fig. 2: An overview of our Multi-Granularity Motion Segmentation (MGMS) pipeline for generating high-quality pseudo-labels from unlabeled videos. The pipeline first extracts bidirectional optical flow from videos and processes it through two parallel modules: the Rigid Object Segmentation Module and the Motion Foreground Segmentation Module. These generate masks for rigid parts and the moving foreground of the central frame (highlighted in red). Finally, masks are filtered and processed with NMS to yield the final multi-granularity pseudo-labels.

## 3.1 Synthetic Video Generation

Segmenting moving objects from real-world videos is inherently challenging due to confounding factors such as complex camera motion, occlusions, and diverse object dynamics (e.g., rigid versus non-rigid motion). To mitigate these challenges and provide clean supervisory signals, we introduce a video synthesis pipeline to train our motion segmentation modules. Inspired by [19], we generate synthetic videos with accurately constructed ground-truth masks. We extend the original framework to support multiple objects and complex occlusions. Specifically, for each synthetic video, a background image is sampled as the background texture, and random afine transformations are applied to simulate camera motion. We then generate multiple foreground objects, represented as randomly shaped and textured patches. Each object moves independently along a smooth B-spline trajectory and undergoes random perspective transformations, with inter-frame smoothness constraints to mimic natural motion and deformation. Although synthetic videos lack semantic realism, they provide unambiguous motion cues and pixel-accurate masks, making them well-suited for training general-purpose motion segmentation models.

## 3.2 Multi-Granularity Motion Segmentation

To capture motion at varying levels of detail, we design a pipeline with two complementary modules, both trained on our synthetic data. The first module focuses on instance-level segmentation of rigidly moving objects, while the second identifies the entire moving foreground. This multi-granularity strategy ensures a comprehensive capture of diverse motion patterns. Further training details for each module are provided in Appendix C.1.

Rigid Object Segmentation Module (ROSM) This module is designed to segment individual objects that undergo coherent motion. In our context, “rigid motion” refers to any motion pattern that can be well-approximated by a single perspective transformation. This assumption holds not only for truly rigid objects (e.g., a car), but also for instance parts (e.g., a person’s arm) or even static elements exhibiting parallax due to camera motion. Our synthetic data, where each object follows a distinct transformation, provides an ideal training ground for this task. We formulate this as an instance segmentation problem, adapting the SOLOv2 framework [35] to operate on bidirectional optical flow. For a given flow field, the module outputs a set of instance masks, each corresponding to a distinct moving entity, along with a confidence score.

Motion Foreground Segmentation Module (MFSM) While the rigid object module excels at segmenting coherent motion, it struggles with complex, non-rigid deformations, such as a walking person whose limbs move semiindependently. To address this, we introduce a module that performs a simpler, yet highly efective task: separating the entire moving foreground from the background. This yields high-quality masks for complex non-rigid objects, complementing the instance-level output of the first module. To this end, we employ a U-Net architecture [30] that takes bidirectional optical flow as input and outputs a single binary mask delineating all moving regions within the central frame.

## 3.3 Mask Quality Assessment and Filtering

ROSM and MFSM produce a large volume of candidate masks, many of which are low-quality and can degrade the performance of downstream models. To prune these, we introduce a Mask Quality Score $( S _ { \mathrm { q u a l i t y } } )$ , which combines two components to retain high-fidelity predictions:

Maskness $( S _ { m a s k n e s s } )$ This score measures the model’s confidence in the pixels it predicts as belonging to the mask. Given a soft mask prediction $p \in [ 0 , \bar { 1 } ] ^ { H \times W }$ 2 we define the set of positive pixels as $\mathcal { P } _ { \mathrm { p o s } } = \{ ( x , y ) \ | \ p ( x , y ) > \tau _ { c } \}$ , where $\tau _ { c }$ is a confidence threshold. The score is then the mean prediction value for this set of pixels:

$$
S _ { \mathrm { m a s k n e s s } } = \frac { 1 } { | \mathcal { P } _ { \mathrm { p o s } } | } \sum _ { ( x , y ) \in \mathcal { P } _ { \mathrm { p o s } } } p ( x , y ) .\tag{1}
$$

Boundary Sharpness $( S _ { s h a r p n e s s } )$ This score rewards masks with well-defined, sharp edges. Let $M _ { b }$ be the mask binarized from p at threshold $\tau _ { c } .$ . Let B be a $d -$ pixel wide band around the contour of $M _ { b } .$ , and $\nabla p$ be the gradient magnitude of

$p .$ The sharpness is the proportion of boundary pixels with a gradient magnitude above a threshold $\gamma _ { g } .$

$$
S _ { \mathrm { s h a r p n e s s } } = \frac { 1 } { | B | } \sum _ { ( x , y ) \in B } \mathbb { I } ( \| \nabla p ( x , y ) \| > \gamma _ { g } ) ,\tag{2}
$$

where I(·) is the indicator function.

The final quality score is a weighted sum of these two components: $S _ { \mathrm { q u a l i t y } } =$ $\beta \cdot S _ { \mathrm { m a s k n e s s } } + ( 1 - \beta ) \cdot S _ { \mathrm { s h a r p n e s s } } .$ , where $\beta$ is a weighting factor. We discard masks with a score below a predefined threshold. Finally, we apply Non-Maximum Suppression (NMS) [26] to the filtered masks to eliminate duplicates, yielding the final set of pseudo-labels.

## 3.4 Pseudo-Label Dataset Construction

We applied our MGMS pipeline to a diverse collection of large-scale public video datasets<sup>5</sup>. The videos span a wide range of scenes, including human activities, animal behaviors, and driving footage. The complementary nature of our two segmentation modules allows us to generate a rich, multi-granularity dataset. The rigid instance module identifies distinct objects and parts, while the motion foreground module captures complete non-rigid entities. From approximately 10,000 hours of raw video, we generated around 21 million high-quality motion pseudo-labels. This large-scale, automatically curated dataset forms the foundation for training our Perceptual Grouping Model, as detailed in the next section.

## 4 Learning from Motion to Segment Anything

While the pseudo-labels from Section 3 are derived from motion, our ultimate goal is to train a universal segmentation model that operates on static images, without any reliance on motion cues at inference time. However, a fundamental challenge arises from the nature of our pseudo-supervision: the masks generated from motion are high-quality but inherently sparse, covering only a fraction of the objects present in an image, e.g., one of the tiger’s paws in Fig. 2. To bridge this gap, we introduce the Perceptual Grouping Model (PGM), a framework designed to learn a generalizable concept of “objectness” from this limited supervision. Central to our approach is a novel contrastive training strategy, which enables the model to discover unlabeled objects beyond the provided masks.

## 4.1 Perceptual Grouping Model Architecture

PGM is built upon a standard Vision Transformer (ViT) [8] backbone. To capture objects at various scales, PGM incorporates K parallel linear projection heads on top of the ViT features. During training, each pseudo-mask is assigned to a specific head based on its bounding box size. The loss for that mask is computed exclusively using the features from its assigned head. This strategy encourages diferent heads to specialize in segmenting objects of diferent scales. Furthermore, to enhance robustness against occlusions and complex scenes, we employ the Copy-Paste data augmentation technique [9].

## 4.2 Perceptual Grouping Contrastive Learning (PGCL)

Standard segmentation objectives, such as Binary Cross-Entropy (BCE) [30] or Dice loss [33], are suboptimal for our training data. Since motion-based pseudolabels are inherently sparse, these losses erroneously penalize unannotated object instances as background. While Slot Attention [14] addresses sparsity, it struggles in open-world settings: its fixed slot capacity clashes with the varying number of objects in open-world scenes, and its rigid pixel-to-slot assignment hampers multi-granularity segmentation. Crucially, slot-centric representations tend to overfit to the specific object categories seen during training, limiting zero-shot generalization to unseen domains.

To overcome these significant limitations and cultivate a more robust and generalizable notion of “objectness” independent of explicit object count or categories, we introduce Perceptual Grouping Contrastive Learning (PGCL), as illustrated in Fig. 3. The central idea is to impose intra-object feature compactness while promoting inter-object feature separability in the patch-level feature space. Specifically, for an input image tokenized into N patches, each of the K heads produces a set of patch embeddings $F ^ { k } \in \mathbb { R } ^ { N \times D }$ . Consider a pseudomask $\mathcal { M } _ { i }$ assigned to head $H _ { k }$ . We sample an anchor patch embedding $\mathbf { f } _ { t } ^ { k }$ from within the mask region. We then compute its cosine similarity $s _ { t , j } ^ { k }$ with all patch embeddings $\{ \mathbf { f } _ { j } ^ { k } \} _ { j = 1 } ^ { N }$ from the same head. The probability $p _ { t , j } ^ { k }$ that patch $j$ belongs to the same object as anchor t is modeled via a sigmoid function with a learnable temperature $\tau _ { k } \colon$

$$
p _ { t , j } ^ { k } = \sigma ( \tau _ { k } \cdot s _ { t , j } ^ { k } ) ,\tag{3}
$$

where $\sigma ( \cdot )$ is the sigmoid function. We define a binary label $m _ { t , j }$ , which is 1 if both the anchor patch t and patch j are within the mask $\mathcal { M } _ { i } .$ , and 0 otherwise. The PGCL loss is a BCE loss, averaged across all heads and sampled anchors:

$$
\mathcal { L } _ { \mathrm { P G C L } } = - \frac { 1 } { K N _ { a } N } \sum _ { k = 1 } ^ { K } \sum _ { t = 1 } ^ { N _ { a } } \sum _ { j = 1 } ^ { N } \left[ m _ { t , j } \log p _ { t , j } ^ { k } + ( 1 - m _ { t , j } ) \log ( 1 - p _ { t , j } ^ { k } ) \right] ,\tag{4}
$$

where $N _ { a }$ is the number of sampled anchors. By explicitly contrasting positive pairs (patches within the same mask) against negative ones, PGCL enables the model to capture object coherence based on feature similarity. This allows the network to generalize beyond the sparse pseudo-labels, efectively handling the “segment anything” task for unseen objects.

![](images/a9199f704f17937f96300b2110989e39a41048fa51b909fef9d5799fc2ae6934.jpg)  
Fig. 3: Illustration of Perceptual Grouping Contrastive Learning (PGCL). Our PGM features multiple heads, each trained on pseudo-masks of a specific scale range. For each assigned mask, we compute a similarity map by comparing an in-mask anchor patch (+) with all patches in the corresponding head’s feature space. The PGCL loss supervises this map using the mask as the ground truth, training the model to group coherent patches into objects.

## 4.3 Adaptation to Segment Anything

While PGM learns a general notion of objectness, it is trained on relatively low-resolution video frames and lacks prompt-based interaction capabilities. To bridge this gap and enable high-performance, high-resolution segmentation with prompt-based interaction, we leverage PGM to generate high-quality masks on high-resolution static images. We then utilize these perceptual grouping masks to train two specialized models for distinct segment anything tasks.

PGM Inference Our perceptual grouping masks generation begins by feeding a static image into the pre-trained PGM. PGM processes the image to generate an initial pool of candidate soft masks, derived from patch embeddings through pairwise cosine similarity. These candidates are then filtered based on a predicted mask quality score (Sec. 3.3) and refined using a dense Conditional Random Field (CRF) [18] to sharpen object boundaries. For high-resolution imagery, we employ a multi-scale, sliding-window approach: PGM inference is applied independently to overlapping tiles at various resolutions. Masks from all tiles are subsequently projected back, merged, and filtered via Non-Maximum Suppression (NMS) to remove duplicates and produce the final comprehensive and fine-grained results. Further implementation details can be found in Appendix C.2.

Training the Segment Anything Models We apply the aforementioned inference pipeline to generate perceptual grouping masks on a 1% subset of the high-resolution SA-1B dataset [15]. We then train two Segment Anything models, following the approach of UnSAM [38]:

– Whole-Image Segmentation: For universal object segmentation, we train a Mask2Former model [6]. This model accepts a full high-resolution image as input and automatically outputs a set of masks, where each mask corresponds to a distinct instance discovered in the scene.

– Promptable Segmentation: To build a prompt-driven model, we adopt an architecture based on Semantic-SAM [21], which excels at predicting masks at multiple granularity levels from a single click. To enable interactivity, we simulate user clicks during training by randomly sampling a point within the foreground of each mask to serve as a positive point prompt.

## 5 Experiments

## 5.1 Training Setups

MGMS Pipeline Training Settings Our MGMS Pipeline comprises two key modules: ROSM employs a SOLO-v2 architecture [35] with a ResNet-18 backbone [12], while MFSM utilizes a Res-UNet architecture [45], also with a ResNet-18 backbone. Both modules take a sequence of 7 frames as input and were trained on a synthetic dataset for 50,000 iterations with an input resolution of 512 × 512 and a batch size of 32.

PGM Training Settings The training data for PGM comprises 21 million pseudo-labels distributed across 10 million static frames. This dataset was generated from a 10,000-hour video corpus constructed from three major sources: Kinetics-700 [5], which contains approximately 650,000 10-second clips covering a wide range of human actions; BDD100K [44], a large-scale driving video dataset designed for diverse road scenarios; and a 5% random sample from YouTube-8M [1], a broad-coverage video collection sourced from YouTube. PGM utilizes a ViT-base architecture [8] with an input size of 512 × 512 pixels. The architecture is topped by four parallel heads, each implemented as a linear layer that produces 128-dimensional feature vectors. We employ the AdamW [25] optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 32.

Segment Anything Model Training Settings For whole-image segmentation, we construct a model by integrating a Mask2former [6] decoder with a ResNet-50 [12] backbone. Training is conducted for 8 epochs, configured with a $5 \times 1 0 ^ { - 5 }$ learning rate, a batch size of 16, and 0.05 weight decay. For promptable segmentation, we leverage the Semantic-SAM [21] architecture with a Swin-Transformer [24] Tiny model as the backbone to improve performance. Both models are trained with only a 1% subset of unlabeled images from SA-1B [15].

## 5.2 Evaluation Datasets and Metrics

We evaluate MoSA under two segmentation settings: Whole-Image Segmentation and Point-Based Promptable Segmentation. For Whole-Image Segmentation, we evaluate MoSA on seven benchmark datasets, including COCO [23], LVIS [10], ADE20K [46], EntitySeg [28], SA-1B [15], PartImageNet [11], and PACO [29]. Importantly, each dataset only labels a subset of objects, whereas our model generates masks across heterogeneous levels and open-world categories. As a result, the standard COCO-style Average Precision (AP) metric may not fully capture the model’s ability to segment diverse, open-world entities. Therefore, following prior unsupervised segmentation works [3, 36, 38], we adopt Average Recall (AR) as the primary metric for whole-image segmentation comparisons. For Point-Based Promptable Segmentation, we evaluate on COCO Val2017 [23]. Following prior promptable segmentation methods [15,21], we report two metrics: MaxIoU and OracleIoU. MaxIoU measures the Intersection over Union (IoU) between the ground truth mask and the highest-confidence predicted mask, while OracleIoU reports the maximum IoU achieved among all predicted masks.

Table 1: Evaluation results of PGM on seven datasets. <sup>†</sup>: The pseudo-labels used by UnSAM [38] are not strictly unsupervised as they are refined by CascadePSP [7].
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=2>|PtIn LVIS Entity PACO ADE COCO SA-1B|Average</td></tr><tr><td rowspan=1 colspan=1>UnSAM [38](CRF [18])UnSAM [38] (CascadePSP†[7])</td><td rowspan=1 colspan=1>23.216.5 22.1  10.8 15.1 21.9  15.327.319.9 24.3  12.8 17.8 24.8  24.7</td><td rowspan=1 colspan=1>17.821.7</td></tr><tr><td rowspan=1 colspan=1>PGM (ours)(w/o refinement)PGM (ours) (CRF [18])</td><td rowspan=1 colspan=1>34.727.9 28.2 16.1 25.5 32.6  36.636.1 29.330.1 16.926.733.8 37.8</td><td rowspan=1 colspan=1>28.830.1</td></tr></table>

![](images/239bd4726fb323b339e5d388918a1e9035726e880317d43f50be2bd755f9212c.jpg)  
Fig. 4: Qualitative comparison of PGM (ours) with DINOv3 family [32], where the red + denotes anchor patches.

## 5.3 Evaluation Results

Perceptual Grouping Masks We quantitatively evaluate PGM on seven benchmarks. As shown in Table 1, PGM generates high-quality masks. Notably, even without refinement, they significantly outperform the pseudo-labels used by UnSAM [38], which are post-processed with CascadePSP [7] (a method relying on manual annotations). Qualitatively, Fig. 4 compares feature similarity maps of PGM and the strong self-supervised vision model DINOv3 [32]. The results highlight PGM’s superior ability to learn a holistic concept of “objectness”. For instance, in the top-right image, DINOv3 only activates on part of the snail, whereas PGM correctly segments the entire snail. Moreover, PGM shows remarkable zero-shot generalization. When applied to an out-of-distribution handdrawn image (bottom-left), it successfully identifies the child figure, whereas the baseline fails to grasp this abstract concept. This suggests that our PGCL efectively teaches the model a generalizable notion of object coherence from sparse motion-based supervision.

Table 2: Results on the whole-image segmentation task. <sup>†</sup>: UnSAM [38] utilizes CascadePSP [7], a method that relies on manual annotations, and is therefore not strictly unsupervised. MoSA achieves SOTA performance among unsupervised methods only using a ResNet-50 backbone and delivers results comparable to SAM [15].
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Backbone #params</td><td rowspan="2"># images</td><td rowspan="2">Avg.</td><td colspan="5">with Whole Entities</td><td colspan="2">with Parts</td></tr><tr><td></td><td>COCO LVIS ADE Entity SA-1B</td><td></td><td></td><td></td><td></td><td>PtIn PACO</td></tr><tr><td>SAM [15]</td><td>VIT-B(85M)</td><td>11M</td><td></td><td>42.1</td><td>49.6 46.1</td><td>45.8</td><td>45.9</td><td>60.8</td><td>28.3</td><td>18.1</td></tr><tr><td>FreeSOLO [34]</td><td>RN-101 (45M)</td><td>1.3M</td><td>7.3</td><td>11.6</td><td>5.9</td><td>7.3</td><td>8.0</td><td>2.2</td><td>13.8</td><td>2.4</td></tr><tr><td>CutLER [36]</td><td>RN-50 (23M)</td><td>1.3M</td><td>21.8</td><td>28.1</td><td>20.2</td><td>26.3</td><td>23.1</td><td>17.0</td><td>28.7</td><td>8.9</td></tr><tr><td>SOHES [3]</td><td>VIT-B(85M)</td><td>0.2M</td><td>30.1</td><td>30.5</td><td>29.1</td><td>31.1</td><td>33.5</td><td>33.3</td><td>36.0</td><td>17.1</td></tr><tr><td>UnSAM† [38]</td><td>RN-50(23M)</td><td>0.1M</td><td>39.2</td><td>40.5</td><td>37.7</td><td>35.7</td><td>39.6</td><td>41.9</td><td>51.6</td><td>27.5</td></tr><tr><td>MoSA (ours)</td><td>RN-50(23M)</td><td>0.1M</td><td>42.1</td><td>43.5</td><td></td><td>42.2 38.4</td><td>41.1</td><td>48.2</td><td>52.7</td><td>28.4</td></tr></table>

Whole-Image Segmentation Table 2 presents the whole-image segmentation performance, measured by AR, of our unsupervised MoSA against baselines. Notably, MoSA, trained with only 1% of the SA-1B training data and utilizing a ResNet-50 backbone (only 23M parameters), achieves an average AR of 42.1%. This performance significantly outperforms previous unsupervised methods and the improvement is consistent across all seven datasets. Furthermore, on PartImageNet [11] and PACO [29], MoSA surpasses the supervised SAM (ViT-B, 11M images) by 24.4% and 10.3% in AR, respectively. This suggests that pre-training on video motion cues enables the discovery of fine-grained objects and details often overlooked by human annotators. Qualitative results in Fig. 1 further illustrate this; notably, MoSA efectively segments even completely static objects, such as tiles on a roof. This capability is a direct consequence of PGCL, which extrapolates from sparse, motion-derived pseudo-labels to learn a universal concept of “objectness”. MoSA’s results align more closely with human perception of object instances and exhibit less segmentation noise.

Point-Based Promptable Segmentation As shown in Table 3, MoSA achieves 41.6% MaxIoU and 63.4% OracleIoU on COCO [23]. This notably outperforms the previous SOTA, UnSAM [38], by +1.3% and +3.9% respectively. Crucially, UnSAM is not strictly unsupervised, as it relies on CascadePSP [7], a refiner trained with manual annotations. While fully supervised SAM [15] sets a higher benchmark (52.1% MaxIoU, 68.2% OracleIoU), MoSA’s performance is compelling given its unsupervised nature and high eficiency, requiring 3.4× fewer parameters and 100× less SA-1B data. Qualitatively (Fig. 1), MoSA provides stable, perceptually-aligned, multi-granularity labels. For instance, even in challenging scenarios such as segmenting a person riding a horse (where their motion is often congruent), MoSA leverages video-learned semantics to accurately distinguish them. Additional qualitative results are in Appendix E.

Table 3: Quantitative comparison on the point-based promptable image segmentation task. MoSA achieves superior performance over the previous SOTA, UnSAM [38], which is not strictly unsupervised as it employs CascadePSP [7].
<table><tr><td>Methods</td><td>Backbone (# params)</td><td>% of SA-1B</td><td>Point Point (Max) (Oracle)</td><td></td></tr><tr><td>SAM [15] (supervised) ViT-B/8 (85M)</td><td></td><td>100%</td><td>52.1</td><td>68.2</td></tr><tr><td>UnSAM [38]</td><td>Swin-Tiny (25M)</td><td>1%</td><td>40.3</td><td>59.5</td></tr><tr><td>MoSA (ours)</td><td>Swin-Tiny (25M)</td><td>1%</td><td>41.6</td><td>63.4</td></tr></table>

Table 4: Ablation study of PGM training.
<table><tr><td>Method</td><td>AR1000 ARs  $\mathrm { A R } _ { \mathrm { m } }$  ARl</td></tr><tr><td>BCE+Dice 3.8</td><td>1.2 2.8 6.6</td></tr><tr><td>Slot Attention [14] 8.2</td><td>5.2 8.6 8.7</td></tr><tr><td>PGCL (ours) 28.3</td><td>14.7 26.6 36.1</td></tr></table>

## 5.4 Ablation Study

PGM Training To verify the efectiveness of our proposed PGCL, we conduct a comparative analysis against strong baselines using identical backbone architectures and training data (1,000 hours of video). Specifically, we compare PGCL with: (1) standard BCE+Dice loss, and (2) Slot Attention [14] with 256 slots. We evaluate the zero-shot transfer performance on 1,000 images from the SA-1B dataset, reporting Average Recall (AR) overall $\left( \mathrm { A R } _ { 1 0 0 0 } \right)$ and across varying scales $\left( \mathrm { A R _ { s } , \ A R _ { m } , \ A R _ { l } } \right)$ . As presented in Table 4, the conventional BCE+Dice baseline fails to segment most objects because sparse motion pseudo-labels wrongly suppress unlabeled static regions as background. Similarly, Slot Attention exhibits suboptimal performance (8.2 AR). We attribute this to the limitation of fixed slot capacity, which struggles to accommodate the uncertain number of objects and the multi-granular hierarchies inherent in open-world scenes. In contrast, our method achieves a substantial improvement, reaching 28.3 AR. This result confirms the superiority of PGCL in handling the complexity of the “segment anything” task.

Table 5: Ablation study of modules in the MGMS pipeline.
<table><tr><td colspan="3">ROSM MFSM  $\mathrm { A R } _ { 1 0 0 0 }$   $\mathrm { { A R _ { s } } }$   $\mathrm { { A R } _ { \mathrm { { m } } } }$   $\mathrm { A R _ { l } }$ </td></tr><tr><td>√</td><td>28.8</td><td>20.6 31.1 28.7</td></tr><tr><td>√</td><td>34.7</td><td>13.7 34.9 43.1</td></tr><tr><td>√ √</td><td>36.6</td><td>20.6 35.4 45.1</td></tr></table>

MGMS Pipeline We ablate the MGMS Pipeline’s key components: the Rigid Object (ROSM) and Motion Foreground (MFSM) Segmentation Modules. Their eficacy is evaluated based on PGM’s Average Recall (AR) on 1,000 SA-1B images when trained on their generated pseudo-labels. As shown in Table 5, ROSM alone excels on small objects (20.6% $\mathrm { A R _ { s } ) }$ but struggles with larger ones. Conversely, MFSM is efective for large objects (43.1% AR<sub>l</sub>) but performs poorly on small ones (13.7% AR ). The full pipeline integrates both modules to achieve the best overall score $( 3 6 . 6 \% \mathrm { \ A R _ { 1 0 0 0 } } )$ . This confirms their complementary nature: combining ROSM’s precision for small objects with MFSM’s broader motion capture is essential for generating robust, multi-scale motion cues.

Number of Prediction Heads We ablate the number of prediction heads in PGM for multi-granularity segmentation. As shown in Fig. 5 (left), performance improves substantially from 27.9% AR with a single head to 36.6% AR with four heads. The most significant gain occurs when adding the second head (+4.2% AR), confirming the fundamental benefit of specialized heads. However, performance saturates beyond four heads, dropping marginally to 36.5% AR with five. This indicates that four heads provide the optimal trade-of, efectively capturing the full range of object scales from motion cues without introducing redundancy.

Impact of Pre-training Video Data Scale We investigate how the amount of unlabeled video data impacts PGM’s ability to learn motion priors. As illustrated on the right side of Fig. 5, we observe a strong positive correlation between data scale and performance. Specifically, increasing the training data from 100h to 1,000h, and further to 10,000h, substantially boosts AR from 16.8% to 28.3% and finally to 36.6%. This clear, non-saturating trend underscores that our PGM efectively leverages large-scale video corpora and strongly suggests that its performance can be further enhanced by scaling to even larger datasets.

## 6 Conclusion

Perceptual grouping is fundamental to human visual intelligence, enabling us to transform raw visual data into semantic understanding through simple, yet powerful rules. Inspired by this biological process, we introduce MoSA, a novel three-stage unsupervised framework for learning generalized segmentation from motion. Extensive experiments demonstrate that by observing visual motion alone, MoSA learns robust segmentation abilities applicable to diverse objects and scenes. This approach not only outperforms existing unsupervised methods, but also ofers a more interpretable and scalable path toward realizing “segmentanything.” Our work highlights the immense potential of learning object perception from motion as a fundamental building block for artificial visual intelligence, bridging the gap towards human-like perceptual understanding.

![](images/0a8510a1f7ad59584ed2e0788b1bc13b965a6699c58203b725088a91d8f9ccee.jpg)

![](images/b8e6d1547a6bffc1b3471e3b68ba087c92053e4f5a1fcbdbd0b8f5ceb5b6ef99.jpg)  
Fig. 5: Impact of the number of prediction heads (left) and pre-training video data scale (right) in PGM.

## References

1. Abu-El-Haija, S., Kothari, N., Lee, J., Natsev, P., Toderici, G., Varadarajan, B., Vijayanarasimhan, S.: Youtube-8m: A large-scale video classification benchmark. CoRR abs/1609.08675 (2016)

2. Bao, Z., Tokmakov, P., Jabri, A., Wang, Y.X., Gaidon, A., Hebert, M.: Discovering objects that can move. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11789–11798 (2022)

3. Cao, S., Gu, J., Kuen, J., Tan, H., Zhang, R., Zhao, H., Nenkova, A., Gui, L., Sun, T., Wang, Y.: SOHES: self-supervised open-world hierarchical entity segmentation. In: ICLR. OpenReview.net (2024)

4. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: ICCV. pp. 9630– 9640. IEEE (2021)

5. Carreira, J., Noland, E., Hillier, C., Zisserman, A.: A short note on the kinetics-700 human action dataset. CoRR abs/1907.06987 (2019)

6. Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R.: Masked-attention mask transformer for universal image segmentation. In: CVPR. pp. 1280–1289. IEEE (2022)

7. Cheng, H.K., Chung, J., Tai, Y., Tang, C.: Cascadepsp: Toward class-agnostic and very high-resolution segmentation via global and local refinement. In: CVPR. pp. 8887–8896. Computer Vision Foundation / IEEE (2020)

8. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR. OpenReview.net (2021)

9. Ghiasi, G., Cui, Y., Srinivas, A., Qian, R., Lin, T., Cubuk, E.D., Le, Q.V., Zoph, B.: Simple copy-paste is a strong data augmentation method for instance segmentation. In: CVPR. pp. 2918–2928. Computer Vision Foundation / IEEE (2021)

10. Gupta, A., Dollár, P., Girshick, R.B.: LVIS: A dataset for large vocabulary instance segmentation. In: CVPR. pp. 5356–5364. Computer Vision Foundation / IEEE (2019)

11. He, J., Yang, S., Yang, S., Kortylewski, A., Yuan, X., Chen, J., Liu, S., Yang, C., Yu, Q., Yuille, A.L.: Partimagenet: A large, high-quality dataset of parts. In: ECCV (8). Lecture Notes in Computer Science, vol. 13668, pp. 128–145. Springer (2022)

12. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR. pp. 770–778. IEEE Computer Society (2016)

13. Hu, Y., Huang, J., Schwing, A.G.: Unsupervised video object segmentation using motion saliency-guided spatio-temporal propagation. In: ECCV (1). Lecture Notes in Computer Science, vol. 11205, pp. 813–830. Springer (2018)

14. Kara, S., Ammar, H., Denize, J., Chabot, F., Pham, Q.C.: Diod: Self-distillation meets object discovery. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3975–3985 (2024)

15. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W., Dollár, P., Girshick, R.B.: Segment anything. In: ICCV. pp. 3992–4003. IEEE (2023)

16. Kofka, K.: Principles of Gestalt psychology. routledge (2013)

17. Köhler, W.: Gestalt psychology. Psychologische forschung 31(1), XVIII–XXX (1967)

18. Laferty, J.D., McCallum, A., Pereira, F.C.N.: Conditional random fields: Probabilistic models for segmenting and labeling sequence data. In: ICML. pp. 282–289. Morgan Kaufmann (2001)

19. Lamdouar, H., Xie, W., Zisserman, A.: Segmenting invisible moving objects. In: BMVC. p. 231. BMVA Press (2021)

20. Lee, M., Cho, S., Lee, S., Park, C., Lee, S.: Unsupervised video object segmentation via prototype memory network. In: WACV. pp. 5913–5923. IEEE (2023)

21. Li, F., Zhang, H., Sun, P., Zou, X., Liu, S., Li, C., Yang, J., Zhang, L., Gao, J.: Segment and recognize anything at any granularity. In: ECCV (48). Lecture Notes in Computer Science, vol. 15106, pp. 467–484. Springer (2024)

22. Lian, L., Wu, Z., Yu, S.X.: Bootstrapping objectness from videos by relaxed common fate and visual grouping. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14582–14591 (2023)

23. Lin, T., Maire, M., Belongie, S.J., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: common objects in context. In: ECCV (5). Lecture Notes in Computer Science, vol. 8693, pp. 740–755. Springer (2014)

24. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: ICCV. pp. 9992–10002. IEEE (2021)

25. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (Poster). OpenReview.net (2019)

26. Neubeck, A., Gool, L.V.: Eficient non-maximum suppression. In: ICPR (3). pp. 850–855. IEEE Computer Society (2006)

27. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

28. Qi, L., Kuen, J., Wang, Y., Gu, J., Zhao, H., Torr, P.H.S., Lin, Z., Jia, J.: Open world entity segmentation. IEEE Trans. Pattern Anal. Mach. Intell. 45(7), 8743– 8756 (2023)

29. Ramanathan, V., Kalia, A., Petrovic, V., Wen, Y., Zheng, B., Guo, B., Wang, R., Marquez, A., Kovvuri, R., Kadian, A., Mousavi, A., Song, Y., Dubey, A., Mahajan, D.: PACO: parts and attributes of common objects. In: CVPR. pp. 7141–7151. IEEE (2023)

30. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: MICCAI (3). Lecture Notes in Computer Science, vol. 9351, pp. 234–241. Springer (2015)

31. Siméoni, O., Puy, G., Vo, H.V., Roburin, S., Gidaris, S., Bursuc, A., Pérez, P., Marlet, R., Ponce, J.: Localizing objects with self-supervised transformers and no labels. In: BMVC. p. 310. BMVA Press (2021)

32. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S.E., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jégou, H., Labatut, P., Bojanowski, P.: Dinov3. CoRR abs/2508.10104 (2025)

33. Sudre, C.H., Li, W., Vercauteren, T., Ourselin, S., Cardoso, M.J.: Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations. In: DLMIA/ML-CDS@MICCAI. Lecture Notes in Computer Science, vol. 10553, pp. 240–248. Springer (2017)

34. Wang, X., Yu, Z., Mello, S.D., Kautz, J., Anandkumar, A., Shen, C., Álvarez, J.M.: Freesolo: Learning to segment objects without annotations. In: CVPR. pp. 14156–14166. IEEE (2022)

35. Wang, X., Zhang, R., Kong, T., Li, L., Shen, C.: Solov2: Dynamic and fast instance segmentation. In: NeurIPS (2020)

36. Wang, X., Girdhar, R., Yu, S.X., Misra, I.: Cut and learn for unsupervised object detection and instance segmentation. In: CVPR. pp. 3124–3134. IEEE (2023)

37. Wang, X., Misra, I., Zeng, Z., Girdhar, R., Darrell, T.: Videocutler: Surprisingly simple unsupervised video instance segmentation. In: CVPR. pp. 22755–22764. IEEE (2024)

38. Wang, X., Yang, J., Darrell, T.: Segment anything without supervision. In: NeurIPS (2024)

39. Wang, Y., Shen, X., Yuan, Y., Du, Y., Li, M., Hu, S.X., Crowley, J.L., Vaufreydaz, D.: Tokencut: Segmenting objects in images and videos with self-supervised transformer and normalized cut. IEEE Trans. Pattern Anal. Mach. Intell. 45(12), 15790–15801 (2023)

40. Xie, J., Xie, W., Zisserman, A.: Segmenting moving objects via an object-centric layered representation. In: NeurIPS (2022)

41. Yang, C., Lamdouar, H., Lu, E., Zisserman, A., Xie, W.: Self-supervised video object segmentation by motion grouping. In: ICCV. pp. 7157–7168. IEEE (2021)

42. Yang, Y., Lai, B., Soatto, S.: Dystab: Unsupervised object segmentation via dynamic-static bootstrapping. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2826–2836 (2021)

43. Yang, Y., Loquercio, A., Scaramuzza, D., Soatto, S.: Unsupervised moving object detection via contextual information separation. In: CVPR. pp. 879–888. Computer Vision Foundation / IEEE (2019)

44. Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., Darrell, T.: BDD100K: A diverse driving dataset for heterogeneous multitask learning. In: CVPR. pp. 2633–2642. Computer Vision Foundation / IEEE (2020)

45. Zhang, Z., Liu, Q., Wang, Y.: Road extraction by deep residual u-net. IEEE Geosci. Remote. Sens. Lett. 15(5), 749–753 (2018)

46. Zhou, B., Zhao, H., Puig, X., Xiao, T., Fidler, S., Barriuso, A., Torralba, A.: Semantic understanding of scenes through the ADE20K dataset. Int. J. Comput. Vis. 127(3), 302–321 (2019)

## Appendix

## A Synthetic Video Generation Details

Each synthetic video is generated by first rendering a dynamic background and then superimposing multiple moving foreground objects. The key components of this process are detailed below. Notably, the entire synthesis process is unsupervised and does not rely on manually annotated data; all ground-truth masks are generated programmatically.

## A.1 Background

The dynamic background is designed to simulate various camera movements and global scene changes.

Source For each synthetic video, we randomly select a single image from a data source that is distinct from all evaluation benchmarks to serve as the base background texture.

Motion Generation The static background image is animated by moving it along a smooth, randomly generated trajectory. This trajectory is defined by a sequence of 2D control points interpolated with B-splines. Concurrently, the background undergoes a random perspective transformation between frames. The parameters for these transformations are sampled randomly but are constrained to ensure smooth, temporally coherent changes, mimicking realistic camera motion.

Motion Profile To introduce variability in relative motion speeds, we simulate two primary scenarios:

Fast Camera Motion: The background undergoes rapid transformations, such as large inter-frame displacements and perspective shifts. In this scenario, foreground objects are set to move more slowly.

– Slow Camera Motion: The background exhibits slower, more subtle movements. Correspondingly, foreground objects exhibit faster and more pronounced motion.

## A.2 Foreground Objects

Foreground objects are synthesized to introduce diverse shapes, appearances, and independent motion patterns, which are critical for learning motion-based segmentation.

Source Object shapes are randomly generated in accordance with the procedure in OCLR [40], with textures sourced from a data collection kept separate from all evaluation benchmarks.

Appearance Generation The visual appearance of each foreground object is diversified through a two-step process. First, the generated object mask undergoes random resizing and in-plane rotation, with angles sampled uniformly from [0, 360<sup>◦</sup>). Second, the transformed mask is treated as a sprite and re-textured. This involves randomly selecting an image, cropping a random patch from it, and applying this patch as the new texture for the sprite.

Motion Generation To simulate dynamic scenes with varying complexity, the number of distinct moving foreground objects per video is randomly sampled from 1 to 15. Each object is initialized at a random 2D position in the first frame and follows an independently generated, smooth B-spline trajectory. Concurrent with its translation, each object also undergoes its own sequence of random perspective transformations across frames. These transformations are constrained to ensure inter-frame smoothness, simulating plausible object movement and deformation.

Motion Disambiguation A critical aspect of our synthesis is ensuring that motion cues are suficient to distinguish individual objects and separate them from the background, especially for optical flow-based models. To this end, we enforce two key constraints. First, the motion trajectories (including translation, velocity, and perspective changes over time) of any two foreground objects within the same video are made unique by sampling diferent control points and parameters for their respective B-spline paths and transformations. Second, the motion trajectory of each foreground object is also guaranteed to be distinct from the global background motion.

## A.3 Synthetic Data Visualizations

Fig. A1 provides visual examples from our synthetic dataset. These examples are specifically designed to be challenging, featuring heavy inter-object occlusions and complex motion trajectories that emulate dificulties found in real-world scenes. The visualizations include the input frames, the resulting optical flow, and the ground-truth segmentation masks, illustrating the direct link between the generated motion and the desired output. It is important to emphasize that the visual fidelity of textures and the precise shapes of the foreground objects are not the primary goal of our synthesis. Since our downstream model relies solely on optical flow as input, it is invariant to such high-level semantic cues. The key objective of our data generation is to produce diverse and complex motion patterns, which are efectively captured by the optical flow representation.

![](images/a8d36a18ac897f8dbee6265e94aa54df4ecdc7d2ba31f9ef46d68af718fabc6f.jpg)  
Fig. A1: Examples from our synthetic video dataset, showcasing challenging scenarios with significant occlusions and complex object interactions. Each row presents one example. Columns 1-5: Five sequential video frames. Column 6: Optical flow for the central frame. Column 7: Individual instance masks for each moving object in the central frame. Column 8: The complete motion foreground segmentation mask for the central frame.

## B Motion Pseudo-Labels Generation Details

Serving as the critical data foundation of MoSA, the Multi-Granularity Motion Segmentation (MGMS) pipeline is designed to automatically generate highquality motion pseudo-labels from large-scale unannotated videos. This section details how this framework processes unlabeled video data to extract robust motion cues. We first introduce the diverse video datasets that serve as input, followed by a comprehensive description of our preprocessing steps and the outputs derived from the specialized modules of the MGMS pipeline.

## B.1 Dataset Description

The training data for our model are derived from three large-scale public video datasets:

– Kinetics-700 [5]: This dataset consists of approximately 650,000 video clips, encompassing 700 distinct human action classes. The videos feature a wide array of human-object interactions (e.g., playing instruments) and human-human interactions (e.g., shaking hands, hugging). Each action class is represented by at least 700 video clips, with each clip having an approximate duration of 10 seconds.

– BDD100K [44]: A substantial driving video dataset containing 100,000 videos. It is designed for evaluating image recognition algorithms in the context of autonomous driving across 10 diferent tasks. The dataset is characterized by its diversity in geographic locations, environmental conditions, and weather patterns.

– YouTube-8M [1]: A very large-scale video dataset comprising over 7 million videos, annotated with 4,716 classes by an automated system. Videos are categorized into 24 broad topics based on their visual content, including sports, gaming, arts & entertainment, among others. For our experiments, we utilized a random sample of 5% of the videos from this dataset.

It is important to reiterate that we did not utilize any human-provided annotations (such as labels, object bounding boxes, or segmentation masks) from these datasets. Our methodology relies solely on the visual content of the videos. Collectively, these datasets provide a rich and diverse collection of scenarios, encompassing human motion, object interactions, animal behavior, autonomous driving footage, and first-person perspectives. This diversity is crucial for training robust models capable of understanding complex dynamic scenes.

## B.2 Data Preprocessing

Our data processing pipeline begins with an initial filtering step to ensure that we only process videos with significant motion. To achieve this, the raw videos from the aforementioned datasets were processed by sampling clips at fixed 5-second intervals. We compute optical flow for sampled video clips using GMFlow. We then calculate the mean magnitude of the optical flow vectors across all pixels and frames. Clips where this mean magnitude falls below a predefined threshold are discarded, as they likely contain only static scenes or negligible camera jitter. This step focuses our computational resources on dynamically rich content.

For each remaining clip, we extract a central 1.5-second segment and uniformly sample 7 frames. We compute bidirectional optical flow between adjacent frames with GMFlow. The optical flow fields from the central 7 frames yield 24 input channels (6 forward and 6 backward flows, each with 2 channels). These are fed into two modules: the Rigid Object Segmentation Module (ROSM) and the Motion Foreground Segmentation Module (MFSM).

## B.3 Module Outputs

The outputs of the ROSM and MFSM are illustrated in Fig. A3 and Fig. A4, respectively. Specifically, ROSM (Fig. A3) not only identifies rigid moving objects (e.g., spoons, cars) but also leverages parallax to segment static foreground objects (e.g., a table next to the dog) and discerns object parts (e.g., an elephant’s trunk, a human arm). In contrast, MFSM (Fig. A4) extracts a broader range of moving foreground elements, including pedestrians, animals, and human-object interactions.

## B.4 Data Postprocessing

The raw masks generated by these two modules undergo a rigorous post-processing pipeline to ensure their quality and uniqueness. First, to prune low-fidelity predictions, we introduce a Mask Quality Score $( S _ { \mathrm { q u a l i t y } } )$ for each candidate mask. This score synergizes two critical metrics. The first, maskness $\left( S _ { \mathrm { m a s k n e s s } } \right)$ quantifies model confidence by averaging the prediction values for all pixels exceeding a threshold of $\tau _ { c } = 0 . 5$ . The second, boundary sharpness $( S _ { \mathrm { s h a r p n e s s } } )$ evaluates contour clarity by calculating the proportion of high-gradient pixels (magnitude $> \gamma _ { g } = 0 . 3 )$ within a 4-pixel-wide band around the mask’s binarized edge. The final score is a weighted sum: $S _ { \mathrm { q u a l i t y } } = \beta \cdot S _ { \mathrm { m a s k n e s s } } + ( 1 - \beta ) \cdot S _ { \mathrm { s l } }$ <sub>harpness</sub>, where we set $\beta = 0 . 4$ to place a slightly higher emphasis on boundary sharpness. Masks with a score below a stringent quality threshold of $\tau _ { \mathrm { q u a l i t y } } = 0 . 8 5$ are discarded. Subsequently, to eliminate redundant masks for the same object, we employ Non-Maximum Suppression (NMS) [26]. Finally, since frames with significant motion can sufer from blur, we filter the generated image-label pairs based on image sharpness. We assess this by calculating the variance of the Laplacian of the image region defined by the mask, and discard pairs where this variance falls below a threshold $\tau _ { \mathrm { b l u r } }$ . This multi-stage filtering protocol is crucial for populating our final dataset with sharp, distinct, and high-confidence pseudo-labels.

## C Model Implementation Details

To provide a comprehensive understanding of MoSA, this section outlines the architectural design and implementation specifics of its key components. We will first describe the Multi-Granularity Motion Segmentation (MGMS) pipeline, which extracts candidate object masks from unlabeled videos by identifying coherently moving regions. Subsequently, we will detail the Perceptual Grouping Model (PGM), focusing on its Vision Transformer architecture [8] and the training and inference procedures. Lastly, we will describe the Whole-Image Segmentation and Promptable Segmentation models. These models are designed to distill object concepts learned by the PGM, enabling highresolution segmentation capabilities across entire images, akin to a “segment anything” functionality.

## C.1 MGMS Pipeline

Our MGMS Pipeline is composed of two interconnected modules: Rigid Object Segmentation Module (ROSM) and Motion Foreground Segmentation Module (MFSM). We now detail the model architecture and training procedures for each of these modules. Table A1 shows their specific training hyperparameters.

Rigid Object Segmentation Module This module is dedicated to segmenting individual rigidly moving objects. We adapt the SOLOv2 [35] architecture, modifying its input to utilize bidirectional optical flow. Given that the input is solely optical flow, which has a relatively low information density compared to RGB images, we employ a lightweight ResNet-18 [12] as the backbone to achieve strong performance eficiently.

Table A1: Training Hyperparameters for ROSM and MFSM.
<table><tr><td>Config</td><td>|ROSM</td><td>MFSM</td></tr><tr><td>Optimizer</td><td> $\mathrm { \ A d a m W \ [ 2 5 ] }$ </td><td> $\mathrm { A d a m W \ [ 2 5 ] }$ </td></tr><tr><td>Base learning rate</td><td> $\mathrm { 5 \times 1 0 ^ { - 4 } }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td> $ { \lvert { 1 \times 1 0 ^ { - 4 } } }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Optimizer momentum</td><td> $\left| \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 \right.$ </td><td> $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ </td></tr><tr><td>Batch size</td><td>32</td><td>32</td></tr><tr><td>Learning rate schedule</td><td>Linear Decay</td><td>Linear Decay</td></tr><tr><td>Warmup iterations</td><td>1000</td><td>0</td></tr><tr><td>Number of iterations</td><td>60000</td><td>30000</td></tr><tr><td>Augmentation</td><td>Random flip</td><td>Random flip</td></tr><tr><td>Input frames</td><td>7</td><td>7</td></tr><tr><td>Input resolution</td><td> $5 1 2 \times 5 1 2$ </td><td> $5 1 2 \times 5 1 2$ </td></tr><tr><td>Input channels</td><td>24</td><td>24</td></tr></table>

Table A2: Training Hyperparameters for the Perceptual Grouping Model (PGM).
<table><tr><td>Hyperparameter</td><td>|Value</td></tr><tr><td>Optimizer</td><td>[AdamW [25]</td></tr><tr><td>Base Learning Rate</td><td>1e-4</td></tr><tr><td>Weight Decay</td><td>1e-2</td></tr><tr><td>Optimizer Momentum</td><td> $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ </td></tr><tr><td>Batch Size</td><td>32</td></tr><tr><td>Learning Rate Schedule</td><td>Linear Decay</td></tr><tr><td>Warmup Epochs</td><td>0.1</td></tr><tr><td>Total Epochs</td><td>10</td></tr><tr><td>Data Augmentation</td><td>Random Flip, ColorJitter, Copy-Paste, Random Crop 4</td></tr><tr><td>Number of Projection Heads</td><td></td></tr><tr><td>Head Output Feature Dimension</td><td>128</td></tr><tr><td>Pretrained Backbone</td><td>MAE ViT-B/8 [4]</td></tr><tr><td>Input Resolution</td><td> $\mathrm { 5 1 2 \times 5 1 2 }$  6</td></tr><tr><td>Number of Anchor Patches</td><td></td></tr></table>

– Input: The input to this module is bidirectional optical flow. For an input video clip of length $t = 7$ frames, we extract $t - 1 = 6$ forward and $t - 1 = 6$ backward optical flow fields. Each optical flow field has 2 channels (x and y components). Thus, the total number of input channels is $( t - 1 ) \times 2 \times 2 = 2 4$ The input spatial resolution is $5 1 2 \times 5 1 2$ pixels.

– Output: The module outputs instance segmentation masks for rigidly moving objects within the central frame of the input clip.

Motion Foreground Segmentation Module This module is responsible for segmenting the entire motion foreground, which may include multiple rigid or non-rigid objects. It employs a U-Net [30] architecture for this binary segmentation task, also using a ResNet-18 [12] backbone, as the task of segmenting the entire motion foreground from optical flow is relatively straightforward.

– Input: Similar to the rigid object segmentation module, the input is bidirectional optical flow. For ${ \mathrm { ~ a ~ } } t = 7$ frame clip, this results in $( t - 1 ) \times 2 \times 2 = 2 4$ input channels. The input spatial resolution is $5 1 2 \times 5 1 2$ pixels.

– Output: The module outputs a binary segmentation mask delineating the motion foreground for the central frame of the input clip.

## C.2 Perceptual Grouping Model (PGM)

The Perceptual Grouping Model (PGM) is trained to internalize perceptual grouping principles from motion-derived pseudo-masks, enabling it to segment unseen objects in static images. This section details its training hyperparameters and inference strategy.

PGM Training Details Motion cues were extracted from $^ \mathrm { ~ a ~ 1 0 , 0 0 0 }$ -hour video dataset using our MGMS pipeline. These cues, in the form of pseudo-masks, were subsequently filtered based on a mask quality score. This process yielded approximately 10,000,000 high-quality video frames, with an average of 2.1 pseudomasks per frame. These frames and their corresponding motion-derived masks served as the training data for the PGM. The model was trained using a perceptual grouping contrastive learning strategy to learn the grouping logic inherent in these dynamic exemplars.

As described in the main paper, our PGM employs $K = 4$ parallel heads to specialize in segmenting objects of varying scales. During training, each pseudomask is deterministically assigned to a specific head based on the size of its normalized bounding box. First, for each pseudo-mask, we compute its tightest bounding box and normalize its coordinates to the range of [0, 1], resulting in coordinates $( x _ { \mathrm { m i n } } , y _ { \mathrm { m i n } } , x _ { \mathrm { m a x } } , y _ { \mathrm { m a x } } )$ . We then calculate a scale metric, s, defined as the geometric mean of the bounding box’s side lengths:

$$
s = \sqrt { \left( x _ { \mathrm { m a x } } - x _ { \mathrm { m i n } } \right) \cdot \left( y _ { \mathrm { m a x } } - y _ { \mathrm { m i n } } \right) }\tag{A1}
$$

A pseudo-mask is then assigned to one of the four heads based on the value of s according to the following criteria:

– Head 1: Assigned masks with $\begin{array} { r } { s < \frac { 1 } { 8 } } \end{array}$ (small objects).

Head 2: Assigned masks with ${ \frac { 1 } { 8 } } \leq { \tilde { s } } < { \frac { 1 } { 4 } } { \mathrm { ~ ( m e d i u m - s m a l l ~ o b j e c t s ) } }$

Head 3: Assigned masks with $\begin{array} { r } { \frac 1 4 \leq s < \frac { 1 } { 2 } \ ( \mathrm { m e d i u m \mathrm { - } l a r g e \ o b j e c t s } ) } \end{array}$

Head 4: Assigned masks with ${ \frac { 1 } { 2 } } \leq s \leq { \bar { 1 } } { \mathrm { ~ ( l a r g e ~ o b j e c t s ) } }$

This scale-based routing ensures that the loss for any given mask is computed exclusively by the head designated for its scale, compelling each head to develop expertise in segmenting objects within its specific size range. The PGM was trained on a server with 8 NVIDIA A100 GPUs. Specific training hyperparameters are detailed in Table A2.

PGM Inference Strategy The PGM is trained primarily on video frames, which often present inherent challenges such as lower resolutions (typically 360p to 720p) and significant motion blur. These characteristics can limit the efective resolution perceived by the model. Furthermore, the PGM’s ViT-B/8 architecture, given a $5 1 2 \times 5 1 2$ pixel training input, generates segmentation maps at a coarse $6 4 \times 6 4$ resolution. To bridge this gap and enable high-resolution segmentation, especially for smaller objects and finer details, we implement a multi-scale tiling inference strategy. The procedure is as follows:

1. Global Context Pass: The entire input image is initially resized to $5 1 2 \times$ 512 pixels and processed by the PGM. This step yields a coarse, global segmentation map.

2. Multi-Scale Sliding Window Pass: The original, high-resolution image is subsequently processed using sliding tiles at two distinct scales:

– Scale 1: Tiles of size $s _ { 1 } \times s _ { 1 } .$ , where $\begin{array} { r } { s _ { 1 } = \frac { 1 } { 2 } } \end{array}$ min(width, height).

– Scale 2: Tiles of size $s _ { 2 } \times s _ { 2 }$ , where $s _ { 2 } = \textstyle { \frac { 1 } { 4 } } \operatorname* { m i n } ( { \mathrm { w i d t h } } , { \mathrm { h e i g h t } } )$

These tiles are applied with a 50% overlap to ensure comprehensive coverage and smooth transitions. The PGM performs an independent inference on each tile.

3. Aggregation and Refinement: The segmentation masks from the global pass and all tiles are aggregated. Non-Maximum Suppression [26] is then applied to the combined predictions to resolve overlaps and redundant detections. As a final refinement step, we apply a fully-connected Conditional Random Field (CRF) [18] to the resulting masks. CRF leverages the color information from the original high-resolution image to sharpen mask bound aries, yielding a pixel-accurate final segmentation.

## C.3 Adaptation to Segment Anything

To operationalize the learned perceptual grouping principles, we distill the knowledge from the PGM into two Segment Anything models: a whole-image segmentation model for automatic segmentation and a promptable segmentation model for interactive use.

Whole-Image Segmentation Model This model is designed for automatic, high-resolution segmentation. It employs a DINO pre-trained ResNet-50 backbone with a Mask2former decoder. To efectively manage the high density of perceptual grouping masks generated by the PGM, the decoder is configured with 1000 learnable object queries, though we sample a maximum of 400 masks per image during loss computation to ensure training eficiency. The model was trained for 8 epochs using the AdamW optimizer with a learning rate of $5 \times 1 0 ^ { - 5 }$ and a batch size of 16, using random $1 0 2 4 \times 1 0 2 4$ crops for data augmentation.

Table A3: Ablation of mask-quality filtering. We report AR on seven evaluation benchmarks and their average.
<table><tr><td></td><td>PtIn LVIS</td><td>Entity PACO</td><td></td><td>ADE</td><td></td><td></td><td>COCO SA-1B Average</td></tr><tr><td>Only maskness</td><td>28.6 24.3</td><td>24.5</td><td>13.9</td><td>21.2</td><td>27.6</td><td>28.9</td><td>24.1</td></tr><tr><td>Only sharpness</td><td>26.3 22.7</td><td>23.5</td><td>12.8</td><td>20.0</td><td>27.1</td><td>27.2</td><td>22.8</td></tr><tr><td>Maskness + sharpness 34.7 27.9</td><td></td><td>28.2</td><td>16.1</td><td>25.5</td><td>32.6</td><td>36.6</td><td>28.8</td></tr></table>

Promptable Segmentation model This model is built for interactive use, generating hierarchical masks from user prompts. It is based on the Semantic-SAM framework and uses a self-supervised pretrained Swin-Tiny backbone. A key feature is its ability to output 6 masks for each prompt, capturing diferent levels of semantic granularity (e.g., a part, the whole object, and a group of objects). This design is inspired by the nested structure of real-world annotations. The model was trained for 4 epochs with a batch size of 8, using AdamW with a base learning rate of $1 \times 1 0 ^ { - 4 }$ and a multi-step decay schedule.

## D More Ablation Experiments

We provide additional analyses of the mask filtering strategy and video-based alternatives to further validate the design choices in MoSA.

## D.1 Mask Quality Filtering

The mask quality score in our MGMS post-processing stage combines maskness and boundary sharpness. Table A3 shows that the two criteria are complementary. Using maskness alone yields 24.1 average AR, while using boundary sharpness alone yields 22.8 average AR. Combining both scores improves the average AR to 28.8, confirming that reliable mask confidence and crisp boundaries are both important for constructing high-quality motion pseudo-labels.

## D.2 Video-Based Alternatives

We also compare MGMS with representative video-based unsupervised segmentation methods. OCLR [40] and VideoCutLER [37] mainly target instance-level video segmentation, whereas MGMS is designed to extract multi-granularity motion masks for learning a transferable object prior. As shown in Table A4, MGMS achieves competitive zero-shot video segmentation performance on DAVIS2017, reaching 55.9 J &F. This is close to VideoCutLER (57.3) and OCLR (55.1), even though MGMS and OCLR use only optical flow while VideoCutLER uses RGB appearance cues.

Furthermore, we study how diferent pseudo-label sources scale with the amount of video pre-training data. When training PGM with VideoCutLER pseudo-labels, the reliance on DINO-based appearance features limits the scaling trend, likely because the pseudo-labels inherit static appearance biases. In contrast, our motion-driven pseudo-labels yield consistent improvements as the video pre-training data grows, as shown in Fig. A2.

![](images/9d052a71d5187454d92d1fedee33fd10d30e92f29250c6dfed11bc95a6b7b08c.jpg)

Table A4: Zero-shot video segmentation on DAVIS2017.
<table><tr><td>J&amp;F J (Mean) F (Mean)</td></tr><tr><td>OCLR [40] 55.1 54.5</td></tr><tr><td>55.7 57.4 57.2</td></tr><tr><td>VideoCutLER [37] 57.3 MGMS (ours) 55.9 55.8 56.0</td></tr></table>

Fig. A2: Scaling trend with video pre-training data. Compared with PGM trained using VideoCutLER pseudo-labels, PGM trained with our motion-driven pseudo-labels scales more robustly as video data increases.

## E Additional Qualitative Results

We provide additional qualitative results demonstrating MoSA’s segment anything capabilities in Fig. A5 and Fig. A6 for whole-image segmentation and promptable image segmentation, respectively.

## F Limitations

Our current work focuses on extracting general segmentation knowledge from motion cues, with the goal of enabling zero-shot segmentation on static images without relying on explicit motion information during inference. Consequently, all evaluation experiments have been conducted on image datasets. Future research will extend MoSA to video data, exploring its applicability and adaptation to dynamic scenes and temporal segmentation tasks.

One practical failure mode arises from degraded motion estimation in lowlight environments, such as night scenes. Since our pseudo-label generation stage depends on optical flow, unreliable flow can fail to provide clear motion cues and may weaken the learned supervision for fine structures. For example, as shown in the penultimate row of Fig. A5, MoSA can struggle to segment subtle body parts such as legs in dark regions, whereas a fully supervised model such as SAM can still succeed with dense manual supervision. Improving robustness to low-light videos and other flow-degraded conditions is an important direction for future work.

![](images/768f964bd94d6adbf58138d1a6d7e47ab1ac757a24688d7727a58bf4d0938a1b.jpg)  
Fig. A3: Rigid Object Segmentation Module (ROSM) output examples. For each case, the first three columns show key input frames, and the fourth column presents ROSM’s segmentation of the dominant rigidly moving entity. ROSM efectively segments whole rigid objects (cars, spoon), isolates articulated parts (elephant’s ear/trunk, human head), and distinguishes smaller moving components (leaves). Furthermore, ROSM can leverage parallax cues from camera motion to segment static objects(e.g., table next to the dog).

![](images/422935510bd6fe7483a8cdbfd023f2f78fedd61dc8b1e314d4052e2944a9743f.jpg)

![](images/9c6599e9ca7d865557be12fb4ee6741bad5d5a86d7f26a677c0f4f37ee808189.jpg)  
Fig. A4: Motion Foreground Segmentation Module (MFSM) efectively extracts all moving foreground elements. For each case, the first three columns display representative input frames, while the fourth column shows MFSM’s segmentation results.

![](images/bf89f61ae5bcd53cd48b40ce8ae613670f250d977a4458a7798faee8b7a2f439.jpg)  
Fig. A5: Additional qualitative results for whole-image segmentation, comparing our MoSA (rightmost column) with the supervised SAM [15] (second column) and the previous state-of-the-art unsupervised method UnSAM [38] (third column). Note that UnSAM is not strictly unsupervised, as its refinement stage utilizes the supervised CascadePSP [7] refiner.

![](images/fd95afac8d37c309ef8371a78986eb08e70a498928a456be7a99c8841de0b9ec.jpg)  
Fig. A6: Comparative results for promptable image segmentation, evaluating our unsupervised MoSA against supervised SAM [15] and unsupervised UnSAM [38] across diverse scenario. The examples illustrate that when prompted by points (indicated as star marks), MoSA consistently generates multi-granular segmentation masks of higher visual quality. Note that UnSAM is not strictly unsupervised, as its refinement stage utilizes the supervised CascadePSP [7] refiner.