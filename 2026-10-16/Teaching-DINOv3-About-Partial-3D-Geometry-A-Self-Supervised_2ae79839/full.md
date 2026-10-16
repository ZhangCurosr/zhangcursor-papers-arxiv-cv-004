# Teaching DINOv3 About Partial 3D Geometry: A Self-Supervised Geometry-Aware Approach

Viktoria Ehm<sup>1,2</sup> Dongliang Cao<sup>3</sup> Riccardo Marin<sup>1,2</sup> Daniel Scholz<sup>1,2</sup>

Weikang Wang<sup>3</sup> Florian Bernard<sup>3</sup> Daniel Cremers<sup>1,2</sup>

<sup>1</sup> Technical University of Munich <sup>2</sup> Munich Center for Machine Learning <sup>3</sup> University of Bonn

![](images/fc2add8d84b279c56c0d5b372f6b6fab6232713492303c8ebf19b4c78454e94e.jpg)

Figure 1. GeoLoRA (Geometry-Aware Low-Rank Adaptation) injects geometry awareness into 2D foundation features for partia shape matching. Our method is self-supervised and produces robust features for different kinds and amounts of partiality, even for out-of distribution real-world shapes (in the image, from left: two shapes cut by a plane; two partial views; a full template for reference; three shapes with holes; a statue from [1]).

## Abstract

Partial shape matching is a crucial yet underexplored problem in 3D vision, with significant relevance to real-world scenarios where shapes are often only partially observed. Existing feature descriptors face difficulties in this setting, as traditional representations either struggle with the boundaries of partial shapes or heavily depend on the shape’s spatial position. While existing approaches have employed DINO features for partial shape matching, these features are not inherently suited for handling partial observations. In this work, we propose a method to refine DINO features using LoRA-based self-supervised learning, enabling the generation offeature descriptors that are robust to partiality. Our features substantially improve performance on partial shape matching compared to traditional or visionfoundationfeatures. Additionally, when integrated into existing partial shape matching pipelines, we achieve state-of-the-art results on partial shape matching and left-right prediction benchmarks. Our code is available at https://github.com/vikiehm/geo-lora.

## 1. Introduction

Finding dense correspondences between 3D deformable shapes is a fundamental step in understanding geometric data, enabling several downstream tasks, including information transfer and learning statistical models. Specifically, when the geometries do not share the same identity (e.g., different humans) and undergo non-rigid deformations (e.g., different poses), establishing such correspondences remains an open research problem. The challenge is further exacerbated when one (or both) of the shapes comes from partial observations, a common real-world case due to occlusions, acquisition noise, or local errors in the 3D reconstruction.

Scholars have relied on handcrafted descriptors [5, 55, 57] for decades, both standalone and as input to deep learning pipelines. Such descriptors demonstrate powerful capabilities for full shapes, but they often rely on theoretical premises (e.g., watertight meshes with the same topology) that make them impractical for partial use cases. Recently, the emergence of foundation models like DINO [14, 44, 53] or Stable Diffusion [50] has introduced a new class of features. Trained on billions of images in a self-supervised manner, these models demonstrate an impressive ability to characterise points semantically, while being robust across deformations and identities. Such features have been applied to various computer vision tasks, including detection [40, 63], segmentation [36, 60], and, more recently, 3D shape matching [19]. However, these models have a limited understanding of 3D geometry, often exhibiting geometrical confusion, such as left-right mismatches and occlusions/disocclusions. These problems persist even in the most advanced foundation models (DINOv3 [53]), as we will demonstrate in our experiments. While recent works have explored 2D-based fine-tuning strategies [30], models trained solely on single-view images lack true 3D ground ing. Achieving this physical understanding requires incorporating 3D information directly into the training process.

While training a new foundation model for 3D shapes from scratch would require a huge amount of data and resources, we propose a more natural and elegant solution that injects 3D awareness into 2D foundation features. We propose GeoLoRA (Geometry-Aware Low-Rank Adaptation), a new self-supervised approach that, using only synthetic data, informs the backbone 2D foundation features of the 3D structure. Our idea is to start from full shapes and generate synthetic partial versions of them, using procedures to simulate real occlusions or artefacts. Since the partial shapes originate from the full ones, they also enjoy a natural 3D correspondence. We use this information to supervise a lightweight LoRA [33] module on top of the 2D foundation model, rendering multiple images for both the full and partial shapes, and optimising the features to be aligned between the two. Finally, unlike fully 2D approaches, since we have access to the 3D geometry of the shapes that generated the image, we incorporate a geodesic penalty into our contrastive loss, providing a strong geometric cue. In practice, we learn a feature space where dissimilar points are pushed away proportionally to their distance on the object surface.

As a result, we achieve the best features for partial-tofull shape matching available to date. We surpass handcrafted and 2D foundation feature alternatives across different partiality classes and datasets, both in partial-to-full and partial-to-partial settings (some results are shown in Fig. 1). The refined features integrate seamlessly with state-of-theart matching methods, boosting their performance. They also show better chirality (left-right disambiguation) than the original backbone features, suggesting that partial-tofull matching can be a useful auxiliary task for enhancing the geometrical understanding of foundation models. In summary, our contributions are:

• We propose GeoLoRA, a novel framework to inject 3D information into 2D foundation features for 3D partial observations.

• We demonstrate that GeoLoRA produces the best features for partial 3D shapes across multiple benchmark datasets.

• We show that these features improve SOTA performance in downstream tasks as partial-to-full and partial-to-partial shape matching, and strengthen the model’s ability to handle geometrical challenges (e.g., left-right disambiguation).

## 2. Related Work

In this section, we give an overview of the most relevant works, including partial shape matching, features for shape matching and self-supervised learning approaches.

## 2.1. Partial Shape Matching

While many full-to-full shape matching works use spatial features [13, 25, 28, 56] or functional maps [6, 12, 29, 38, 45], the task of matching partial observations remains a challenging open problem [23]. We focus on the most important partial-shape matching works in this section and refer to these surveys [17, 51] for an in-depth overview of general shape matching methods.

Partial-to-Full In partial-to-full shape matching, we aim to find correspondences between a partial observation and a full template shape. PFM [48] analyses partial shape matching with functional maps and uses the slanted diagonal structure for optimisation. DPFM [3] incorporates this idea into a deep learning framework and adds a cross-attention refinement module for feature communication between the partial and full shape. ULRSSM [12] ensures that the obtained functional map is coupled with valid point-wise correspondences. Other partial shape matching methods focus on distance preservation [8, 9] for isometric matching. Axiomatic methods [22, 26, 49] ensure smoothness through geometric consistency with an (Integer) Linear Programs.

Partial-to-Partial Partial-to-partial shape matching is even more challenging, as both shapes have missing parts that may also differ between the two. In this setting, we not only want to find correspondences between shapes, but also determine the regions that overlap. DPFM [3] presents a supervised method, which predicts both overlap regions and correspondences. Echo-Match [62] improves DPFM’s overlap predictor and shows the usefulness of 2D image foundation features for partial-to-partial shape matching. Besides learning-based frameworks, GC-PPSM [21] and Partial-GeCo [24] tackle the problem using a (non-)linear integer program, while enforcing geometric consistency.

## 2.2. Features for Shape Matching

Hand-Crafted Features Most full-to-full shape matching methods start from hand-crafted geometrical features, such as the Heat Kernel Signature [55] and the Wave Kernel Signature [5]. While showing success in full-to-full shape matching, partial shape matching features are fragile due to missing parts and boundaries. Hence, SHOT [57] features, or the vertex coordinates themselves, are generally preferred as input. On the other hand, their extrinsic nature leads to biases toward shape alignment, as shown in [4, 23, 62].

Vision Foundation Models Vision foundation models (VFM), such as the DINO family [14, 44, 53], masked autoencoders [31], and Segment Anything [35] are trained on a large corpus of data providing general-purpose features for various tasks. They also have proven useful in numerous domains outside of natural image analysis. Examples include adaptations in medicine for multi-modal images [52], registration [54], and volume classification [43], as well as in robotics [18, 40, 59, 66]. The field of shape matching has also leveraged these general-purpose features in seminal works on full [19, 39, 58] and partial [62] shape matching. Our work distinguishes itself from prior shape analy sis papers by being the first to not only utilise the features generated by DINOv3 as input, but also incorporates 3D information about partial shapes into the foundation model. However, it is worth mentioning that 2D foundation models suffer from geometrical ambiguity [27, 64]. Several works rely on 2D tasks, incorporating geometrical augmentations such as image flipping [65], or optimal transport objectives to account for (dis-)occlusions in the images [30]. Involving 3D data has been significantly less explored, and generally limited to textured rigid objects [46]. To the best of our knowledge, we are the first to inject geometry awareness into foundation models for partial, non-rigid, untextured 3D shape matching.

## 2.3. Self-Supervised Learning for 3D Shapes

PointContrast [61] introduces the contrastive loss called PointInfoNCE for pre-training on 3D point clouds, aiming to facilitate transfer learning to high-level scene understanding. This approach was later extended by follow-up works [15, 32], all of which tackle static scenes, whereas we focus on non-rigid, deformed shapes. In [4], the authors show that pre-training on rigid scenes can improve non-rigid shape matching performance by optimising the receptive field size of local features. In [11], they use a self-supervised learning approach together with the functional map framework to improve matching performance for shapes represented as point clouds. While these methods show the effectiveness of self-supervised learning for 3D shapes, they do not yet tackle partial non-rigid shapes at a large scale using vision foundation models.

## 3. Method

Overview GeoLoRA is the first self-supervised approach that develops geometry-aware foundation features for partial shape matching. We start by generating partial shapes from full ones, mimicking synthetic occlusions and missing surfaces. Then, we use these pairs of full and partia shapes, and train a LoRA module to align the features between them, via a novel geodesic-based contrastive loss. Figure 2 illustrates our approach, with individual components detailed below. For better understandability, we refer to the supplementary material for a comprehensive outline of our notation.

## 3.1. Self-Supervised Training

Partial Shape Generation Given a full 3D shape $\mathcal { X } =$ $\left( \mathbf { V } _ { \mathcal { X } } , \mathbf { T } _ { \mathcal { X } } \right)$ we generate a partial shape $\mathcal { y } = ( \mathbf { V } _ { \mathcal { y } } , \mathbf { T } _ { \mathcal { y } } )$ by randomly selecting one of the following approaches: for CUTS we follow [16, 21] and remove part of the shape using a random plane through its center; for HOLES we follow [8], selecting a random number of areas with a random radius; for the RAY technique, we simulate the partiality induced by a partial observation from a random camera view [23]. These approaches generate partial shapes that are a subset of the full one $( \mathbf { V } _ { \mathcal { Y } } \subset \mathbf { V } _ { \mathcal { X } }$ and $\mathbf { T } _ { \mathcal { Y } } \subset \mathbf { T } _ { \mathcal { X } } )$ which allows us to know the ground-truth correspondences $\Pi _ { \mathcal { V X } } \in \mathbb { R } ^ { | \mathbf { V } _ { \mathcal { V } } | \times | \mathbf { V } _ { \mathcal { X } } | }$ between them.

Foundation Features After the partial shape generation, we collect the foundation features for the full shape X and its partial counterpart Y. We randomly rotate the shapes around the Y-axis, and following Diff3f [19] sample N different cameras around the full shape X, to obtain a set of cameras $\boldsymbol { \mathcal { C } } ~ = ~ ( c ^ { 1 } , . . . , c ^ { N } )$ . For each camera $c ^ { i } .$ , we render an image $\mathbf { I } _ { \mathcal { X } } ^ { i } \in \mathbb { R } ^ { 5 1 2 \times 5 1 2 \times 3 }$ from its viewpoint. Each image is then processed by a DINOv3 [53] module. Our goal is to improve the features obtained from the foundation model. However, fine-tuning the entire foundation model would cause an additional computational burden. Hence, we keep the DINOv3 backbone frozen during training and add a low-rank adaptation (LoRA) module with rank r on top [33]. We use LoRA in a siamese fashion, meaning that the weights are shared between the full and partial observations. For every image $\mathbf { I } _ { \mathcal { X } } ^ { i }$ we obtain pixel-wise features $\mathbf { Q } _ { \mathcal { X } } ^ { i } \in \mathbb { R } ^ { 5 1 2 \times 5 1 2 \times 7 6 8 }$ , which we backproject to the mesh by averaging all pixels of all images observing a given vertex. This produces vertex-wise features for the full shape, $\mathbf { F } _ { \mathcal { X } } \in \mathbb { R } ^ { | \bar { \mathbf { V } } _ { \mathcal { X } } | \times 7 6 8 }$ . We apply the same procedure to the partial shape Y by sampling N cameras around it, resulting in a set of cameras $\mathcal { C } .$ For each camera $c _ { i } \in \mathcal { C }$ , we extract image features $\mathbf { Q } _ { \mathcal { Y } } ^ { i }$ , which we then backproject to produce vertex-wise features $\mathbf { F } _ { \mathcal { Y } } \in \mathbb { R } ^ { | \mathbf { V } _ { \mathcal { Y } } | \times 7 6 8 }$ . At this point, it is worth highlighting that the full and partial shapes represent the same object in the same pose. At first glance, this may appear restrictive, as we actually aim to solve for nonrigid shape matching, and at inference time, we can expect to match objects that have undergone various deformations. However, we believe our training setting exactly represents the task our backbone should solve: recover the semantics of an observed partial shape, as if it belonged to a full geometry. In practice, we set a small number of N = 9 cameras for the full shape and N = 4 for its partial observation ensuring efficient model training.

![](images/b73923713bef5c7c72081dbaebbc777044952c39d17877b1ca8cab81e8220cb1.jpg)  
Figure 2. Our Self-Supervised Training Pipeline: Given a full mesh $x ,$ we first generate a partial observation Y. After randomly rotating both shapes with different rotation matrices around the Y-axis, we render the partial and full shape from different perspectives with cameras C. For every camera $c ^ { i } \in { \mathcal { C } } .$ , we obtain an image $\mathbf { I } _ { \mathcal { X } } ^ { i }$ of the full shape, which is then passed through a frozen DINOv3 architecture with a LoRA module with rank r to extract pixel-wise features $\mathbf { Q } _ { \mathcal { X } } ^ { i }$ . These are then backprojected on the shape and averaged to get vertex-wise features $\mathbf { F } _ { \mathcal { X } }$ . Similarly, we proceed for the partial shape Y. Finally, we use the Geodesic-Aware PointInfoNCE contrastive loss to align corresponding vertices of both shapes and separate all others based on geodesic distance. In our experiments we use 4 views for the partia and 9 views for the full shape.

Geodesic-Aware PointInfoNCE At this point, for every pair of full and partial shapes, we collected several rendered views and corresponding foundation features from the DI-NOv3 backbone and the appended LoRA module. Now, we aim to utilise this information to optimise the parameters of the LoRA module and enhance the alignment between fullshape and partial-shape features. For the loss, a common choice is the PointInfoNCE contrastive loss [3, 61], already shown promising results in previous shape matching works:

$$
{ \cal L } _ { N C E } = - \sum _ { ( v _ { i } , v _ { j } ) \in G T } \log \frac { \exp ( { \bf F } _ { \mathcal { V } } ^ { i } \cdot { \bf F } _ { \mathcal { X } } ^ { j } / \tau ) } { \displaystyle \sum _ { v _ { k } \in { \bf V } _ { \mathcal { V } } } \exp ( { \bf F } _ { \mathcal { V } } ^ { k } \cdot { \bf F } _ { \mathcal { X } } ^ { j } / \tau ) } ,\tag{1}
$$

where we define $G T$ as the set of corresponding pairs between the full and partial shapes, τ is a temperature hyperparameter, and $\mathbf { F } _ { \mathcal { X } }$ and $\mathbf { F } _ { \mathcal { Y } }$ are the features of the ful and partial shapes, respectively. While the numerator pulls the feature representation of corresponding vertices of the partial and full shape together, the denominator pushes the features apart, making sure that the feature representation does not degenerate to one value. However, in this formulation, there is a clear distinction between correct and incorrect predictions. Namely, it penalises all mismatches equally, without considering the severity of the error in the prediction. Hence, we add a weighting to the loss, such that points further away are penalised more heavily in the loss function, taking into account the actual distances between the vertices. We use the geodesic error matrix on the full shape, $\mathbf { D } _ { \mathcal { X } } \in \mathbb { R } ^ { | \mathbf { V } _ { \mathcal { X } } | \times | \mathbf { V } _ { \mathcal { X } } | }$ , where $\mathbf { D } _ { \mathcal { X } } ^ { j k }$ represents the geodesic distance (distance along the surface) between two vertices $v _ { j } , v _ { k } \in \mathbf { V } _ { \mathcal { X } }$ When referring to a partial vertex $v _ { k } \in \mathbf { V } _ { \mathcal { Y } }$ , we assume its known correspondence on the full shape, allowing us to index into $\mathbf { D } _ { \mathcal { X } }$ appropriately. We adjust the PointInfoNCE Loss, such that:

$$
L = - \sum _ { ( v _ { i } , v _ { j } ) \in G T } \log \frac { \exp ( \mathbf { F } _ { \mathcal { Y } } ^ { i } \cdot \mathbf { F } _ { \mathcal { X } } ^ { j } / \tau ) } { \sum _ { v _ { k } \in \mathbf { V } _ { \mathcal { Y } } } \exp ( ( \mathbf { D } _ { \mathcal { X } } ^ { j k } + 0 . 5 ) \cdot \mathbf { F } _ { \mathcal { Y } } ^ { k } \cdot \mathbf { F } _ { \mathcal { X } } ^ { j } / \tau ) } .\tag{2}
$$

We shift the geodesic distance matrix by 0.5 to obtain a similar range as in the original PointInfoNCE loss, where the denominator is multiplied by 1. Providing the network with this informative geometric cue produces strong supervision signals, which can provide fundamental characterisation of ambiguous geometry, such as symmetric points that are far apart (e.g., right vs. left limbs). Our intuition will be later confirmed in the experiments, where our method shows improvements over the vanilla PointInfoNCE for left-right disambiguation, and in terms of geodesic error.

## 4. Experiments

We demonstrate the effectiveness of GeoLoRA with an extensive set of experiments. First, we compare the matching induced by the features from GeoLoRA with those obtained from handcrafted and learning-based alternatives. Further, we incorporate GeoLoRA features into partial-to-full and partial-to-partial shape matching SOTA methods, demonstrating their natural blend with them. To show the improved geometrical representation, we also test their leftright characterisation on partial shapes. Finally, we justify our design choices through an ablation study.

## 4.1. Implementation Details

We use the DINOv3 ViT-B/16 backbone and integrate LoRA modules into all attention layers. The LoRA configuration uses a rank of r = 16 and a scaling factor of $\alpha = 3 2$ . To maintain training feasibility, we use 9 views for each full shape and 4 views for each partial shape. During inference, we increase the number of views to 49. In each training batch, we generate two partial observations per full shape. All methods are trained for 50.000 iterations.

## 4.2. Datasets

For partial-to-full matching, we use the SHREC16 dataset [16], which is based on the synthetic shapes of TOSCA [10]. It contains 76 nearly-isometric shapes from eight different classes of humans and animals. Each class has one template mesh in a neutral pose. The partial shapes are divided into CUTS and HOLES. The CUTS dataset contains partial shapes cut by a plane, whereas the HOLES dataset contains shapes with irregular holes that are produced by an erosion process. We rely on the CUTS24 split [21] to ensure that no shapes from the test set have been used for training. We also test on the PFAUST dataset [8], a modification of [7], where 10 different humans (8 are used for training and 2 for testing) are seen in 10 different poses. PFAUST processes the dataset adding artificial holes of varying difficulty, including PFAUST-M (medium difficulty) and PFAUST-H (hard difficulty). A relevant dataset is also CP2P24 [3, 21], a partial-to-partial matching dataset based on SHREC16, including cuts for near-isometric shapes. We also stress-test our method on non-isometric partial correspondence tasks using the PSMAL dataset [21], derived from SMAL [67] by adding artificial cuts. For PSMAL, the train and test sets are split by different animal species. Finally, we also include the recent BECOS dataset [23], which integrates both humanoid and animal shapes from FAUST [7], SCAPE [2], SMAL [67], DT4D [37], TOSCA [10], KIDS [47], and SHREC20 [20], totaling over 2,500 unique shapes. BECOS provides the most extensive benchmark, with full-to-full, partial-to-full, and partial-to-partial shape instances.

## 4.3. Metrics

We report the widely used Geodesic Error metric following the Princeton Protocol [34]. The error is scaled by the shape diameter of the corresponding full shape, also for partial-to-partial shape matching [21]. We report the mean geodesic error over all vertices in the dataset. For the partial-to-partial datasets, we report the geodesic error on the overlapping region. As proposed in [21], we use the Intersection over Union (IoU) to evaluate the overlap region prediction. For a shape Y we compare the overlap prediction $P y \in \{ 0 , 1 \} ^ { | \mathbf { V } _ { y } | }$ and the ground truth overlap $G _ { y } \in \{ 0 , 1 \} ^ { | \mathbf { V } _ { y } | }$ , which includes for every vertex in $\mathbf { V } _ { \mathcal { Y } }$ if it is in the overlapping region. We evaluate the quality of the matching with the $\begin{array} { r } { \mathrm { I o } \bar { \mathrm { U } } = \frac { | P _ { y } \cap G _ { y } | } { | P _ { y } \cup G _ { y } | } } \end{array}$ . Finally, we follow [58] for the evaluation of Left-Right (chirality) Accuracy. For every partial shape Y with vertices $\mathbf { V } _ { \mathcal { Y } }$ , we get the predicted chirality vector $\mathsf { \bar { P } } _ { \mathrm { c h i } } \in \mathsf { \{ 0 , 1 \} } ^ { | \mathbf { V } _ { y } | }$ , where 0 indicates one side and 1 the other side. We compute the accuracy with the ground truth chirality vector $G _ { \mathrm { c h i } } \mathbf { \bar { \Psi } } \in \{ 0 , 1 \} ^ { | \mathbf { v } _ { y } | }$ as:

$$
\operatorname { a c c } _ { \mathcal { V } } = \frac { 1 } { \left| \mathbf { V } _ { \mathcal { V } } \right| } \sum _ { v \in \mathbf { V } _ { \mathcal { V } } } \mathbb { 1 } ( \operatorname { s i g n } ( P _ { \mathrm { c h i } } ) = G _ { \mathrm { c h i } } ) ,\tag{3}
$$

where 1 is the indicator function. We refer to the supplementary for the computation for a full dataset.

## 4.4. Matching with Raw Features

Baselines and Data. We compare matchings induced by GeoLoRA with established handcrafted descriptors—WKS [5] and HKS [55]—as well as features from foundation models DINOv2 [44], DINOv3 [44], and Diff3f [19]. For DINOv2, DINOv3, and GeoLoRA, we utilise 49 views, providing the optimal balance between performance and runtime (see our ablation study). For training our selfsupervised network, we always use the full shapes from the training set, ensuring that no shapes of the test/validation set have been seen during training. Since SHREC16’s training set includes full shapes of partials used in its test set, we create BECOS\*, the BECOS training dataset excluding SHREC16 data. We evaluate the geodesic error of these feature extractors by performing a nearest-neighbour search using the L2 distance.

Results We report the quantiative analysis in Table 1. We observe a significant improvement in terms of geodesic error over all the datasets, particularly for the very challenging BECOS dataset. We report qualitative examples in Figure 3. Our improvement is noticeable across all datasets, with better left-right disambiguation compared to using DI-NOv3 features directly.

<table><tr><td>Type</td><td>Dataset</td><td>WKS</td><td>HKS</td><td>Diff3f</td><td>DINOv2</td><td>DINOv3</td><td>Trained on</td><td>GeoLoRA</td></tr><tr><td rowspan="5">P2F</td><td>BECOS [23]</td><td>58.35</td><td>69.50</td><td>25.61</td><td>21.54</td><td>19.95</td><td>BECOS</td><td>5.57</td></tr><tr><td>SHREC16 CUTS [16]</td><td>62.08</td><td>53.32</td><td>28.01</td><td>26.58</td><td>21.25</td><td>BECOS*</td><td>10.86</td></tr><tr><td>SHREC16 HOLES [16]</td><td>50.33</td><td>65.56</td><td>26.67</td><td>20.22</td><td>20.22</td><td>BECOS*</td><td>9.70</td></tr><tr><td>PFAUST-M [8]</td><td>50.28</td><td>52.19</td><td>12.60</td><td>11.38</td><td>11.36</td><td>FAUST</td><td>2.88</td></tr><tr><td>PFAUST-H [8]</td><td>54.75</td><td>60.83</td><td>19.32</td><td>17.47</td><td>13.82</td><td>FAUST</td><td>3.33</td></tr><tr><td rowspan="3">P2P</td><td>BECOS [23]</td><td>41.35</td><td>40.48</td><td>19.45</td><td>17.43</td><td>16.84</td><td>BECOS</td><td>6.72</td></tr><tr><td>CP2P24 [3, 21]</td><td>38.40</td><td>37.04</td><td>22.60</td><td>18.43</td><td>18.55</td><td>BECOS*</td><td>12.46</td></tr><tr><td>PSMAL [21]</td><td>39.50</td><td>39.53</td><td>30.86</td><td>19.59</td><td>18.64</td><td>SMAL</td><td>6.85</td></tr></table>

Table 1. Improvement of Feature Quality Through Our Self-Supervised Pipeline: We show that with GeoLoRA we can improve the feature quality on all partial datasets in terms of the mean geodesic error (×100). BECOS\* indicates BECOS without SHREC16 data.

![](images/289e2e500717c9a6b25b93969817b51c8897ff61ea49dd72ea17c1c74346a3fd.jpg)  
Figure 3. Qualitative Results of Feature Quality: We show colour-coded correspondences based on the features computed with different feature extractors. GeoLoRA exhibits improved performance, especially in left-right predictions.

## 4.5. Integration with Matching Methods

State-of-the-art shape matching methods do not use features as they are, but further refine them for specific datasets and settings. Hence, we also explore the applicability of our features as replacements for alternatives in matching pipelines, both for the partial-to-full and partial-to-partial matching methods. Notice that this latter is a case for which we have not explicitly trained our features, and hence will test their generality.

Partial-to-Full Shape Matching For partial-to-full shape matching, we use SOTA unsupervised methods, DPFM [3] and ULRSSM [12]. While originally they used spatial coordinates as input, this has been shown to heavily depend on the alignment of shapes in the datasets [4, 23, 62]. Therefore, we use vision foundation features as input. We compare GeoLoRA with DINOv3 [53] features, as this latter is the second-best-performing (see Sec. 4.4). We report quantitative results for the partial-to-full matching in Table 2, where the numbers in brackets show the features after test-time adaptation from ULRSSM. We show a qualitative example in Figure 4. For both methods, we observe the improvement provided by GeoLoRA over the DINOv3 alternative. Note that for the BECOS dataset, the geodesic error is higher than when using GeoLoRA directly. We see two reasons for this: Firstly, BECOS is the only dataset providing unnormalised shapes, meaning that the size ratio of the shapes is realistic (e.g., an elephant is larger than a mouse). This may lead to incorrect predictions of the slanted diagonal ratio in the partial functional map prediction of ULRSSM and DPFM. Secondly, the raycasting technique used to generate the partiality produces numerous boundaries, degrading the quality of the functional map.

<table><tr><td>Geo Err (↓)</td><td>Features</td><td>ULRSSM [12]</td><td>DPFM [3]</td></tr><tr><td rowspan="2">BECOS</td><td>DINOv3</td><td>26.97 (26.19)</td><td>23.16</td></tr><tr><td>GeoLoRA</td><td>11.62 (11.54)</td><td>13.38</td></tr><tr><td rowspan="2">SHREC16CUTS</td><td>DINOv3</td><td>5.94 (4.30)</td><td>10.78</td></tr><tr><td>GeoLoRA</td><td>3.01 (1.97)</td><td>6.57</td></tr><tr><td rowspan="2">SHREC16HOLES</td><td>DINOv3</td><td>12.41 (9.66)</td><td>10.41</td></tr><tr><td>GeoLoRA</td><td>8.07 (6.88)</td><td>7.38</td></tr><tr><td rowspan="2">PFAUST-M</td><td>DINOv3</td><td>1.71 (1.69)</td><td>3.78</td></tr><tr><td>GeoLoRA</td><td>1.61 (1.61)</td><td>3.62</td></tr><tr><td rowspan="2">PFAUST-H</td><td>DINOv3</td><td>5.19 (5.22)</td><td>6.46</td></tr><tr><td>GeoLoRA</td><td>2.29 (2.19)</td><td>5.66</td></tr></table>

Table 2. Partial-to-Full Shape Matching with GeoLoRA: We show that we can improve the downstream task performance of partial-to-full shape matching with GeoLoRA in comparison to DINOv3 features in terms of mean geodesic error (×100). The numbers in brackets of ULRSSM show the geodesic error after test-time adaptation.

<table><tr><td></td><td>Features</td><td colspan="2">EchoMatch [62]</td><td colspan="2">DPFM [3]</td></tr><tr><td></td><td></td><td>Err</td><td>mIoU</td><td>Err</td><td>mIoU</td></tr><tr><td>BECOS</td><td>DINOv3 GeoLoRA</td><td>9.74 5.55</td><td>67.07 71.27</td><td>9.83 8.20</td><td>49.80 56.04</td></tr><tr><td>PSMAL</td><td>DINOv3 GeoLoRA</td><td>5.56 4.33</td><td>82.75 85.41</td><td>11.25</td><td>73.25</td></tr><tr><td>CP2P24</td><td>DINOv3</td><td>2.97</td><td>87.02</td><td>8.91 3.89</td><td>78.95 71.55</td></tr><tr><td></td><td>GeoLoRA</td><td>2.97</td><td>87.92</td><td>3.48</td><td>70.41</td></tr></table>

Table 3. Partial-to-Partial Shape Matching with GeoLoRA: We demonstrate that our GeoLoRA features achieve better downstream partial-to-partial shape matching performance than DI-NOv3, as measured by geodesic error (×100) and Intersection over Union (IoU).

Partial-to-Partial Shape Matching Since no unsupervised methods are available for partial-to-partial shape matching, we rely on two SOTA supervised methods DPFM [3] and EchoMatch [62]. In this setting, in conjunction with the correspondence, we also aim to identify the points that have a correspondence between the shapes, and thus determine their overlapping regions. Following [21], we report both the geodesic error and the IoU in Table 3. For BECOS and PSMAL, we see improvement in terms of correspondence quality and overlap prediction. For the CP2P24 dataset, we observe similar results with GeoLoRA compared to DINOv3, likely because the supervision prevents it from benefiting from improved feature descriptors.

![](images/1f3647c38f2bdf810058e8ebcd0c8d797950268d4ff69ce6ef4a796bd8898c94.jpg)

Figure 4. Qualitative Results of Partial-to-Full Shape Matching Methods: We show one qualitative result of partial-to-full shape matching on the SHREC16 CUTS dataset. We can see improvement in the quality of the matching, especially at the arm with GeoLoRA features.
<table><tr><td></td><td>16</td><td>25</td><td>36</td><td>49</td><td>64</td><td>100</td></tr><tr><td>Geo Err</td><td>25.47</td><td>20.80</td><td>21.48</td><td>20.09</td><td>20.29</td><td>20.00</td></tr><tr><td>Time(s)</td><td>1.52</td><td>2.35</td><td>3.36</td><td>4.59</td><td>6.01</td><td>9.47</td></tr></table>

Table 4. Number of Views N with Frozen DINOv3: With an increasing number of views, the time increases linearly, while the geodesic error (×100) already saturates earlier. Thus, we choose 49 views for our experiments.

## 4.6. Ablation Studies

To analyse our design choices, we train on the BECOS fullto-full train set and validate on the BECOS partial-to-full validation set. Experiments were conducted on an NVIDIA A40 GPU and an Intel Xeon Gold 6248R CPU. We show further ablation studies in the supplementary material.

Number of Views We present in Table 4 an ablation study for the number of cameras/views N, reporting the mean geodesic error and runtime in seconds. Using 49 views is the sweet spot between the two metrics, as a larger number of views does not improve the mean geodesic error. For fairness, we choose the same number of views for GeoLoRA and the baselines DINOv2 and DINOv3.

Loss Formulation We compare our loss formulation with the widely used PointInfoNCE loss on the BECOS validation set. We observe a strong improvement with GeoLoRA in terms of geodesic error in Table 5.

<table><tr><td></td><td>PointInfoNCE</td><td>GeoLoRA</td></tr><tr><td>Err (↓)</td><td>11.29</td><td>5.89</td></tr></table>

Table 5. PointInfoNCE vs GeoLoRA: We show that our loss formulation decreases the geodesic error (×100) significantly compared to the widely used PointInfoNCE loss.

<table><tr><td></td><td>DINOv3</td><td>PointInfoNCE</td><td>GeoLoRA</td></tr><tr><td>Acc (↑)</td><td>70.45</td><td>84.09</td><td>91.42</td></tr></table>

Table 6. Chirality Score for BECOS Partial Shapes: We show that we can improve the left-right prediction of [58] with GeoLoRA in terms of accuracy.

![](images/6e6d1470f5b5ea7154726f007172457f9db5f114d8e5636ae473e4f38ec37dba.jpg)  
Figure 5. Qualitative Results of Left-Right Prediction: GeoLoRA shows better chirality prediction than DINOv3 (shapes from BECOS).

## 4.7. Applications

Chirality Classification Distinguishing the left and right of a mesh is an essential task for 3D shape analysis. We train the network from [58] with different features: DI-NOv3, GeoLoRA, and features obtained by our pipeline, but with the PointInfoNCE loss. For training, we use the BECOS full-to-full train set, and we evaluate on the partial shapes from the BECOS partial-to-full test set. From the results in Table 6, we observe that GeoLoRA provides significant improvement in terms of accuracy. The qualitative improvement can be seen in Figure 5, even for highly partial shapes. In comparison to the PointInfoNCE loss, our formulation further improves the left-right prediction.

Real-World Partiality Mapping a partial observation to a full template is particularly relevant for real-world applications, such as interpreting data from acquisition pipelines. In Figure 6, we report an illustrative result, obtained from a depth-sensor-based body scanner. During the acquisition, the user was partially misaligned with respect to the capturing volume. While DINOv3 fails to recover the semantics of such ruined geometry, our features show robustness. We found this result really exciting, as it can open up new techniques for non-rigid reconstruction of moving objects.

![](images/ade6c079e28bc5b3e60cf18f67b1a6c3f36dd4ca0cc45e9fb32bb6da8d5fdaff.jpg)  
Figure 6. Real-World Scan: We match a real-world partial scan on a template human. DINOv3 struggles to characterise incomplete geometry, while GeoLoRA shows robust performance.

## 5. Discussion and Conclusion

We introduced GeoLoRA, a novel self-supervised approach for learning robust features on partial 3D shapes. By relying solely on synthetic data and self-supervised tasks, we demonstrated that our features not only possess a better geometrical understanding than 2D foundation backbones but are also the best available features to date for partial-to-partial and partial-to-full shape matching. The main limitation is likely inherited from DINOv3, which involves pretraining on natural images and thus incorporates a bias toward shapes in an upright orientation. Although 3D objects are often assumed to be upright, this assumption might worsen the results in upside-down poses. While our method still outperforms the DINOv3 baseline on upside-down poses, as shown in the supplementary material, both approaches show reduced performance compared to their results on upright-oriented shapes. Overall, our work demonstrates that adapting vision foundation models through low-rank adaptation can substantially advance partial shape understanding. In the future, our work could be extended to also consider external clutter and man-made objects. Ultimately, we view our work as a promising step toward bridging the gap between large-scale image representations and 3D geometry, providing valuable insights to the broader communities of shape analysis and 3D vision.

## 6. Acknowledgments

We thank Paul Roetzer for valuable feedback on our manuscript. This work was supported by the ERC Advanced Grant “SIMULACRON” (agreement #884679), the ERC starting grant no. 101160648 (Harmony) and the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 458610525.

## References

[1] Scan the world on myminifactory.com. Last accessed: 13th November 2025. 1, 2

[2] Dragomir Anguelov, Praveen Srinivasan, Daphne Koller, Sebastian Thrun, Jim Rodgers, and James Davis. Scape: shape completion and animation of people. ACM SIGGRAPH 2005 Papers, pages 408–416, 2005. 5

[3] Souhaib Attaiki, Gautam Pai, and Maks Ovsjanikov. Dpfm: Deep partial functional maps. In 3DV, 2021. 2, 4, 5, 6, 7, 1

[4] Souhaib Attaiki, Lei Li, and Maks Ovsjanikov. Generalizable local feature pre-training for deformable shape analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13650–13661, 2023. 3, 6

[5] Mathieu Aubry, Ulrich Schlickewei, and Daniel Cremers. The wave kernel signature: A quantum mechanical approach to shape analysis. In ICCV Workshops, pages 1626–1633, 2011. 1, 3, 5

[6] Lennart Bastian, Yizheng Xie, Nassir Navab, and Zorah Lahner. Hybrid functional maps for crease-aware non-¨ isometric shape matching. CVPR, 2024. 2

[7] Federica Bogo, Javier Romero, Matthew Loper, and Michael J Black. Faust: Dataset and evaluation for 3d mesh registration. In CVPR, pages 3794–3801, 2014. 5

[8] Amit Bracha, Thomas Dages, and Ron Kimmel. On unsu-\` pervised partial shape correspondence. ACCV, 2024. 2, 3, 5, 6

[9] Amit Bracha, Thomas Dages, and Ron Kimmel. Wormhole\` loss for partial shape matching. In NeurIPS, 2024. 2

[10] Alexander Bronstein, Michael Bronstein, and Ron Kimmel. Numerical Geometry of Non-Rigid Shapes. Springer Publishing Company, Incorporated, 1 edition, 2008. 5

[11] Dongliang Cao and Florian Bernard. Self-supervised learning for multimodal non-rigid 3d shape matching. In CVPR, 2023. 3

[12] Dongliang Cao, Paul Roetzer, and Florian Bernard. Unsupervised Learning of Robust Spectral Shape Match ing. ACM Transactions on Graphics, 42(4):1–15, 2023. arXiv:2304.14419 [cs]. 2, 6, 7

[13] Dongliang Cao, Marvin Eisenberger, Nafie El Amrani, Daniel Cremers, and Florian Bernard. Spectral meets spatial: Harmonising 3d shape matching and interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3658–3668, 2024. 2

[14] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou,´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Pro-

ceedings of the IEEE/CVF international conference on com puter vision, pages 9650–9660, 2021. 1, 3

[15] Yujin Chen, Matthias Nießner, and Angela Dai. 4dcontrast: Contrastive learning with dynamic correspondences for 3d scene understanding. In European Conference on Computer Vision, pages 543–560. Springer, 2022. 3

[16] Luca Cosmo, Emanuele Rodola, Michael M Bronstein, An drea Torsello, Daniel Cremers, Y Sahillioglu, et al. Shrec’16: Partial matching of deformable shapes. Proc. 3DOR, 2(9): 12, 2016. 3, 5, 6

[17] Bailin Deng, Yuxin Yao, Roberto M. Dyke, and Juyong Zhang. A Survey of Non-Rigid 3D Registration. Computer Graphics Forum, 41(2):559–589, 2022. eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.14502. 2

[18] Norman Di Palo and Edward Johns. Dinobot: Robot manipulation via retrieval and alignment with vision foundation models. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 2798–2805. IEEE, 2024. 3

[19] Niladri Shekhar Dutt, Sanjeev Muralikrishnan, and Niloy J. Mitra. Diffusion 3D Features (Diff3F) Decorating Untextured Shapes with Distilled Semantic Features. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4494–4504, Seattle, WA, USA, 2024. IEEE. 2, 3, 5

[20] Roberto M Dyke, Yu-Kun Lai, Paul L Rosin, Stefano Zap pala, Seana Dykes, Daoliang Guo, Kun Li, Riccardo Marin, Simone Melzi, and Jingyu Yang. Shrec’20: Shape correspondence with non-isometric deformations. Computers & Graphics, 92:28–43, 2020. 5

[21] Viktoria Ehm, Maolin Gao, Paul Roetzer, Marvin Eisenberger, Daniel Cremers, and Florian Bernard. Partial-topartial shape matching with geometric consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27488–27497, 2024. 2, 3, 5, 6, 7

[22] Viktoria Ehm, Paul Roetzer, Marvin Eisenberger, Maolin Gao, Florian Bernard, and Daniel Cremers. Geometrically Consistent Partial Shape Matching. In 2024 Inter national Conference on 3D Vision (3DV), pages 914–922, 2024. ISSN: 2475-7888. 2

[23] V Ehm, N El Amrani, Y Xie, L Bastian, M Gao, W Wang, L Sang, D Cao, T Weißberg, Z Lahner, et al. Beyond complete¨ shapes: A benchmark for quantitative evaluation of 3d shape surface matching algorithms. In Computer Graphics Forum, page e70186. Wiley Online Library, 2025. 2, 3, 5, 6

[24] Viktoria Ehm, Paul Roetzer, Florian Bernard, and Daniel Cremers. An integer linear programming approach to geometrically consistent partial-partial shape matching. In 3DV, 2026. 2

[25] Marvin Eisenberger, Zorah Lahner, and Daniel Cremers.¨ Smooth shells: Multi-scale shape registration with functional maps. In CVPR, 2020. 2

[26] N. El Amrani, P. Roetzer, and F. Bernard. High-resolution 3d shape matching with global optimality and geometric consis tency. Computer Graphics Forum, 44(5):e70208, 2025. 2

[27] Mohamed El Banani, Amit Raj, Kevis-Kokitsi Maninis, Abhishek Kar, Yuanzhen Li, Michael Rubinstein, Deqing Sun,

Leonidas Guibas, Justin Johnson, and Varun Jampani. Probing the 3d awareness of visual foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21795–21806, 2024. 3

[28] Danielle Ezuz, Behrend Heeren, Omri Azencot, Martin Rumpf, and Mirela Ben-Chen. Elastic correspondence between triangle meshes. In Computer Graphics Forum, pages 121–134. Wiley Online Library, 2019. 2

[29] Oshri Halimi, Or Litany, Emanuele Rodola, Alex M Bronstein, and Ron Kimmel. Unsupervised learning of dense shape correspondence. In CVPR, pages 4370–4379, 2019. 2

[30] Regine Hartwig, Dominik Muhle, Riccardo Marin, and Daniel Cremers. Geco: Geometrically consistent embedding with lightspeed inference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9309– 9319, 2025. 2, 3

[31] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable´ vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 3

[32] Ji Hou, Benjamin Graham, Matthias Nießner, and Saining Xie. Exploring data-efficient 3d scene understanding with contrastive scene contexts. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 15587–15597, 2021. 3

[33] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 2, 3

[34] Vladimir G Kim, Yaron Lipman, and Thomas Funkhouser. Blended intrinsic maps. ACM transactions on graphics (TOG), 30(4):1–12, 2011. 5

[35] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 3

[36] Feng Li, Hao Zhang, Huaizhe Xu, Shilong Liu, Lei Zhang, Lionel M Ni, and Heung-Yeung Shum. Mask dino: Towards a unified transformer-based framework for object detection and segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3041–3050, 2023. 2

[37] Yang Li, Hikari Takehara, Takafumi Taketomi, Bo Zheng, and Matthias Nießner. 4dcomplete: Non-rigid motion estimation beyond the observable surface. In ICCV, pages 12706–12716, 2021. 5

[38] Or Litany, Tal Remez, Emanuele Rodola, Alex Bronstein, and Michael Bronstein. Deep functional maps: Structured prediction for dense shape correspondence. In ICCV, pages 5659–5667, 2017. 2

[39] Haolin Liu, Xiaohang Zhan, Zizheng Yan, Zhongjin Luo, Yuxin Wen, and Xiaoguang Han. Stable-score: A stable registration-based framework for 3d shape correspondence.

In Proceedings of the Computer Vision and Pattern Recogni tion Conference, pages 917–928, 2025. 3

[40] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In Euro pean conference on computer vision, pages 38–55. Springer, 2024. 2, 3

[41] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. ACM Transactions on Graphics, 34(6), 2015. 3

[42] Juhong Min, Jongmin Lee, Jean Ponce, and Minsu Cho. Spair-71k: A large-scale benchmark for semantic correspon dence. arXiv preprint arXiv:1908.10543, 2019. 3

[43] Gustav Muller-Franzes, Firas Khader, Robert Siepmann,¨ Tianyu Han, Jakob Nikolas Kather, Sven Nebelung, and Daniel Truhn. Medical slice transformer for improved diag nosis and explainability on 3d medical images with dinov2. Scientific Reports, 15(1):23979, 2025. 3

[44] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 1, 3, 5

[45] Maks Ovsjanikov, Mirela Ben-Chen, Justin Solomon, Adrian Butscher, and Leonidas Guibas. Functional maps: a flexible representation of maps between shapes. ACM Transactions on Graphics (TOG), 31(4):30:1–30:11, 2012. 2

[46] Qiyang Qian, Hansheng Chen, Masayoshi Tomizuka, Kurt Keutzer, Qianqian Wang, and Chenfeng Xu. Bridging viewpoint gaps: Geometric reasoning boosts semantic correspon dence. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11579–11589, 2025. 3

[47] Emanuele Rodola, Samuel Rota Bulo, Thomas Windheuser, Matthias Vestner, and Daniel Cremers. Dense non-rigid shape correspondence using random forests. In CVPR, pages 4177–4184, 2014. 5

[48] Emanuele Rodola, Luca Cosmo, Michael M Bronstein, An-\` drea Torsello, and Daniel Cremers. Partial functional correspondence. In Computer graphics forum, pages 222–236. Wiley Online Library, 2017. 2

[49] Paul Roetzer and Florian Bernard. Fast globally optimal and geometrically consistent 3d shape matching. In ICCV, 2025. 2

[50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1

[51] Yusuf Sahillioglu. Recent advances in shape correspon-˘ dence. The Visual Computer, 36(8):1705–1721, 2020. 2

[52] Daniel Scholz, Ayhan Can Erdur, Viktoria Ehm, Anke Meyer-Baese, Jan C Peeken, Daniel Rueckert, and Benedikt Wiestler. Mm-dinov2: Adapting foundation models for multi-modal medical image analysis. In International Conference on Medical Image Computing and Computer Assisted Intervention, pages 320–330. Springer, 2025. 3

[53] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa,¨ Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve J´ egou, Patrick La-´ batut, and Piotr Bojanowski. DINOv3, 2025. 1, 2, 3, 6

[54] Xinrui Song, Xuanang Xu, and Pingkun Yan. Dinoreg: General purpose image encoder for training-free multimodal deformable medical image registration. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, pages 608–617, Cham, 2024. Springer Nature Switzerland. 3

[55] Jian Sun, Maks Ovsjanikov, and Leonidas Guibas. A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion. Computer Graphics Forum, 28(5):1383– 1392, 2009. 1, 3, 5

[56] Ramana Sundararaman, Nicolas Donati, Simone Melzi, Etienne Corman, and Maks Ovsjanikov. Deformation recovery: Localized learning for detail-preserving deformations. SIG-GRAPH Asia, 2024. 2

[57] Federico Tombari, Samuele Salti, and Luigi Di Stefano. Unique signatures of histograms for local surface description. In ECCV, pages 356–369. Springer, 2010. 1, 3

[58] Weikang Wang, Tobias Weißberg, Nafie El Amrani, and Florian Bernard. kh: Symmetry understanding of 3d shapes via chirality disentanglement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 28292– 28302, 2025. 3, 5, 8, 2

[59] Xin Wen, Bingchen Zhao, Yilun Chen, Jiangmiao Pang, and Xiaojuan Qi. A data-centric revisit of pre-trained vision models for robot learning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12143– 12154, 2025. 3

[60] Yuanchen Wu, Xiaoqiang Li, Jide Li, Kequan Yang, Pinpin Zhu, and Shaohua Zhang. Dino is also a semantic guider: Exploiting class-aware affinity for weakly supervised semantic segmentation. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 1389–1397, 2024. 2

[61] Saining Xie, Jiatao Gu, Demi Guo, Charles R Qi, Leonidas Guibas, and Or Litany. Pointcontrast: Unsupervised pretraining for 3d point cloud understanding. In European conference on computer vision, pages 574–591. Springer, 2020. 3, 4

[62] Yizheng Xie, Viktoria Ehm, Paul Roetzer, Nafie El Amrani, Maolin Gao, Florian Bernard, and Daniel Cremers. EchoMatch: Partial-to-Partial Shape Matching via Correspondence Reflection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11665– 11675, 2025. 2, 3, 6, 7

[63] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022. 2

[64] Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, and Ming-Hsuan

Yang. A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence. Advances in Neural Information Processing Systems, 36:45533–45547, 2023. 3

[65] Junyi Zhang, Charles Herrmann, Junhwa Hur, Eric Chen, Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. Telling left from right: Identifying geometry-aware semantic correspondence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3076–3085, 2024. 3

[66] Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. arXiv preprint arXiv:2411.04983, 2024. 3

[67] Silvia Zuffi, Angjoo Kanazawa, David Jacobs, and Michael J. Black. 3D menagerie: Modeling the 3D shape and pose of animals. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2017. 5

# Teaching DINOv3 About Partial 3D Geometry: A Self-Supervised Geometry-Aware Approach

Supplementary Material

## Summary

In this supplementary material, we provide additional details supporting the main paper. We begin with an overview of the notation (Sec. 7) used throughout the method section, followed by extended ablation studies (Sec. 8), like runtime comparisons, and a formal definition of the dataset-wide chirality accuracy (Sec. 9). We further describe implementation details (Sec. 10), compare our feature representation to those used in prior work (Sec. 11 and 12), present failure cases (Sec. 13) and analyse the upright bias (Sec. 14). Additionally, we show more real-world scans (Sec. 15) and provide additional qualitative results (Sec. 16), including feature visualisations. Finally, we present further 2D/3D application examples (Sec. 17) and we give some more insights on rendering with and without texture (Sec. 18).

## 7. Notation Table

For better understandability, we show an overview of the notation used in the method section in Table 7.

<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $\mathcal { X } = ( \mathbf { V } _ { \mathcal { X } } , \mathbf { T } _ { \mathcal { X } } )$ </td><td>3D Full Shape</td></tr><tr><td> $\mathbf { V } _ { \mathcal { X } }$ </td><td>Vertices of shape  $\mathcal { X }$ </td></tr><tr><td> $\mathbf { T } _ { \mathcal { X } }$ </td><td>Triangles of shape  $\mathcal { X }$ </td></tr><tr><td> $\mathcal { y } = ( \mathbf { V } \boldsymbol { y } , \mathbf { T } \boldsymbol { y } )$   $\Pi _ { \mathcal { V } \mathcal { X } }$ </td><td>3D Partial Shape,  $\mathbf { V } y \subset \mathbf { V } x , \mathbf { T } y \subset \mathbf { T } x$  Correspondence from partial to full shape</td></tr><tr><td> $\mathcal { C } = ( c ^ { 1 } , . . . , c ^ { N } )$ </td><td>Set of all cameras</td></tr><tr><td> $\mathbf { I } _ { \mathcal { X } } ^ { i }$ </td><td>Image of full shape from  $c ^ { i } \in { \mathcal { C } }$ </td></tr><tr><td> $\mathbf { Q } _ { \mathcal { X } } ^ { i }$ </td><td>Features computed on  $\mathbf { I } _ { \mathcal { X } } ^ { i }$ </td></tr><tr><td> $\mathbf { F } _ { \mathcal { X } }$ </td><td>Vertex-wise features of shape X</td></tr><tr><td> $\mathbf { D } _ { \mathcal { X } }$ </td><td>Geodesic Distance Matrix of X</td></tr></table>

Table 7. Overview of Symbols

## 8. Further Ablation Studies

LoRA Size We evaluate different values for hyperparameters, α and the rank r in LoRA on the BECOS validation set in terms of geodesic error (Table 8). The values r = 16 and $\alpha = 3 2$ yield the best results.

Inference Time Table 9 compares the inference time of different methods. The additional use of stable diffusion in Diff3f results in a longer processing time compared to using DINO features alone. The other feature descriptors, including our method GeoLoRA, have similar runtimes.

<table><tr><td>r/α</td><td>4/8</td><td>8/16</td><td>16/32</td></tr><tr><td>Geo Error (↓)</td><td>8.14</td><td>6.56</td><td>5.89</td></tr></table>

Table 8. LoRA Hyperparameter Ablation: We observe the best results with rank r = 16 and α = 32.
<table><tr><td></td><td>DINOv2</td><td>DINOv3</td><td>Diff3f</td><td>GeoLoRA</td></tr><tr><td>Time [s] (↓)</td><td>4.72</td><td>4.45</td><td>360.45</td><td>4.59</td></tr></table>

Table 9. Inference Time Comparison: While DINOv2, DINOv3 and GeoLoRA have similar runtime during inference, Diff3f’s runtime is significantly increased due to the need to additionally run inference with Stable Diffusion.

Backbone Scale Replacing the ViT-B backbone with a ViT-L backbone does not lead to significant performance gains in terms of geodesic error (see Table 10). Nevertheless, ViT-L requires 50% more training time and twice the amount of VRAM on the GPU than ViT-B.

<table><tr><td>PFAUST</td><td>DINOv3-B</td><td>GeoLoRA-B</td><td>DINOv3-L</td><td>GeoLoRA-L</td></tr><tr><td>-M</td><td>11.36</td><td>2.88</td><td>11.40</td><td>2.43</td></tr><tr><td>-H</td><td>13.82</td><td>3.33</td><td>12.50</td><td>2.65</td></tr></table>

Table 10. Performance of Different Backbone Scales: We show that with a larger backbone we do not gain significant improvement in terms of geodesic error.

## 9. Dataset-wide Chirality Accuracy Definition

Given a full dataset U with M partial shapes $\begin{array} { r l } { U } & { { } = } \end{array}$ $\{ \mathcal { V } _ { 1 } , . . . , \mathcal { V } _ { M } \}$ , we define the chirality accuracy as:

$$
a c c _ { U } = \frac { 1 } { M } \sum _ { y _ { i } \in U } \operatorname { a c c } _ { y _ { i } }\tag{4}
$$

We can define the final chirality accuracy as:

$$
a c c _ { \mathrm { c h i } } = \operatorname* { m a x } \{ a c c _ { U } , 1 - a c c _ { U } \}\tag{5}
$$

## 10. Additional Implementation Details

For the temperature parameter, we follow [3] and use $\tau =$ 0.07. We run the training on one A40 GPU and five cores of a Xeon Gold 6248R CPU. The training takes approximately 5 days for 50, 000 iterations for the longest-lasting BECOS dataset. The training time can vary between 2-5 days between the different datasets according to different number of vertices. As a prompt to Stable Diffusion inside Diff3f, we use ”animal” for animal shapes, ”human” for human shapes and ”centaur” for centaur shapes.

## 11. ULRSSM: Comparison to Original Features

Instead of using foundation features or handcrafted descriptors, an alternative approach is to start with the raw XYZ coordinates as input. Since previous methods have shown promising results, for fairness, we also compare our approach with ULRSSM [12] using XYZ coordinates as input. In Table 11, we show that the performance falls far behind the use of foundation features. The use of XYZ is sensitive to shape orientations, which are particularly ambiguous in partial shapes.

<table><tr><td>Geo Error (↓)</td><td>XYZ</td><td>Ours</td></tr><tr><td>BECOS</td><td>40.46 (40.18)</td><td>11.62 (11.54)</td></tr><tr><td>SHREC16CUTS</td><td>7.04 (3.2)</td><td>3.01 (1.97)</td></tr><tr><td>SHREC16 HOLES</td><td>13.75 (8.2)</td><td>8.07 (6.88)</td></tr><tr><td>PFAUST-M</td><td>7.65 (7.49)</td><td>1.61 (1.61)</td></tr><tr><td>PFAUST-H</td><td>9.39 (9.24)</td><td>2.29 (2.19)</td></tr></table>

Table 11. Comparison of XYZ features vs GeoLoRA features: We compare the geodesic error (×100) on different partial-to-full datasets with ULRSSM. On all datasets, our features outperform the originally used XYZ features. The numbers in brackets “()” indicate geodesic error results after test-time adaptation [12].

## 12. Left-Right Prediction: Comparison to Original Features

In the main paper, we compare the GeoLoRA features with those of DINOv3. For completeness, here we report a comparison with the features originally used in [58], which combine DINOv2 and Stable Diffusion features (Table 12).

<table><tr><td></td><td>Original</td><td>Ours</td></tr><tr><td>Chirality Accuracy (↑)</td><td>57.40</td><td>91.42</td></tr></table>

Table 12. Comparison to DINOv2 + Stable Diffusion Features: We compare the chirality (left-right) accuracy of the features used in [58] to our GeoLoRA features. Our method’s features show higher accuracy.

## 13. Failure Cases

The datasets used to train our models primarily contain upright shapes, resulting in most failure cases involving shapes in different poses, such as a rotated person in a pike position or a person lying on their side with their hands on the ground. We show two examples in Figure 7.

![](images/6af220b38986395b62417107c04f5aaad4840a8f620b4d5bdef988a2148deedf.jpg)  
Figure 7. Failure Cases: We show two failure cases of our method. The method primarily struggles with shapes in challenging poses, where the shapes are not positioned upright.

![](images/231075271d0b6721e2ab8b60bcce35de801c584feb784ef679972f9ff00a4040.jpg)  
Figure 8. Additional Results on Real-World Statues from [1]: DINOv3 fails to find good correspondences between the partial statues and the full template shape, where GeoLoRA finds reasonably dense correspondences.

## 14. Upright Bias

For consistency with the literature, we assume that shapes are oriented upright. However, we train GeoLoRA on FAUST (for 70K iterations), sample random rotation along all axes, and test upside-down partial shapes. GeoLoRA achieves 6.69 geodesic error, while DINOv3 performs 10× worse (63.84).

## 15. Additional Real-World Experiments

Additional Real World Scans of Statues We demonstrate further real-world scans of statues from [1] and match them to a template shape from FAUST. Here, GeoLoRA shows qualitatively better matchings than DINOv3.

Real-World Validation Quantitative evaluation on realworld partial scans is limited by the lack of ground-truth data availability. Therefore we evaluate the annotated real scan of SHREC20. Our method significantly reduces geodesic error compared to DINOv3 (5.92 vs. 16.26), indicating improved robustness on real-world data.

## 16. Additional Qualitative Results

Feature Quality We show more qualitative comparisons to other features in Figure 11. Additionally, we show further examples of our feature predictions in Figure 12.

Partial Shape Matching We show qualitative results for partial-to-partial shape matching in Figure 13 and additional results for partial-to-full shape matching in Figure 14.

## 17. Further 2D/3D Applications

While we focus on learning features for 3D partial shape matching, we find their use in other contexts exciting. Hence, we use our GeoLoRA model (trained on BeCoS) to perform image matching on SPair-71k [42] dataset images. In several cases, our approach provides better matching than DINOv3, especially in images where foreground elements differ in their view (see Table 9). Additionally, we show that our method can also be used for 3D keypoint matching for shapes of the BeCoS test set (see Table 10).

![](images/3adf964df9de2218e3c44eb404e3fdb831fa5a2c9c729d0f8944e12624f89c75.jpg)  
Figure 9. 2D Keypoint Matching: We show examples for keypoint matching on the SPair-71k [42] dataset. Our method shows superior matching results in these examples.

![](images/cb0d8dd60245dc09db77e25e4e0b766a6615bb23c6b51a9700e03ed13d88ad40.jpg)  
Figure 10. 3D Keypoint Matching: Our method can be used for 3D keypoint matching. We show examples of the BeCoS test set.

## 18. Texture & Rendering

All partial shape matching benchmarks are textureless, so meshes are rendered without texture (single colour with shadow on the surface, see Fig. 2 Image Column). Using SMPL [41], we empirically test texture sensitivity and find that adding texture degrades performance (geodesic error 3.65 vs. 1.32). We hypothesise that texture helps identify correspondences within the same (e.g., person), but not across different identities (e.g., due to different clothing).

Source

HKS

WKS

Diff3f

DINOv2

DINOv3

GeoLoRA

![](images/6959798b7e1c062ffa813f9b775eadfbf8b27731242265aff4d758cddb8dadd8.jpg)

Figure 11. Qualitative Results of Feature Quality: We show colour-coded correspondences based on the features computed with differen feature extractors. GeoLoRA exhibits improved performance, especially in left-right and front-back predictions.

# AYXOEC

Figure 12. More Qualitative Results: We show more qualitative results of our partial-to-full features.

Source

DPFM

DINOv3

EchoMatch

GeoLoRA

DINOv3

GeoLoRA

BECOS

![](images/5c785dd9b2e3a4c25f1afb0bd393ddb9948b71a59e4741f83d5777e0d60b7828.jpg)

# PSMAL

CP2P24

Figure 13. Qualitative Results of Partial-to-Partial Matching: We show colour-coded correspondences for partial-to-partial shape matching methods. We compare DINOv3 features with GeoLoRA features. The red area indicates points that are predicted not to be in the overlapping region. On the BECOS dataset and the PSMAL dataset, we demonstrate improved performance with GeoLoRA features compared to DINOv3 features. On CP2P24 we observe similar matching quality.

![](images/d97820a0f58a3e4f7edec38997fe05e28df40fa30873c4312a6df84376f6a137.jpg)  
Figure 14. Qualitative Results of Partial-to-Full Shape Matching Methods: We show two more examples of partial-to-full shape matching on the PFAUST-H and SHREC16 HOLES dataset. We can see improved matching quality with our GeoLoRA features, especiall in the extremities of the shapes, such as the left arm of the human and the legs of the horse.