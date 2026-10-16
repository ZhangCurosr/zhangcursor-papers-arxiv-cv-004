# Unsupervised Clustering of Public Transport Vehicles Using Deep Morphological Features and Color Histograms

Juan Irving Vasquez

Centro de Innovacion y Desarrollo Tecnol´ ogico en C´ omputo´

Instituto Politecnico Nacional´

Mexico City, Mexico

jvasquezg@ipn.mx

Abstract—Surveillance cameras along urban roads produce large volumes of unlabeled video that could support the monitoring of public transport, but grouping the observed vehicles into meaningful categories without manual annotation remains challenging, since buses from different routes often share the same body structure and differ mainly in their color schemes. This study presents an unsupervised clustering methodology for public transport vehicles that combines morphological features extracted with a frozen DINOv2 Vision Transformer and handcrafted HSV color histograms through weighted early fusion, followed by UMAP projection and K-Means clustering. The method was evaluated on 7,972 bus images extracted from about 1,460 hours of video of an arterial road in Mexico City, with the goal of isolating the buses of the city’s public bus network (RTP) without supervision. With a fusion of 65% morphological and 35% color features, the pipeline produced 14 clusters that separate distinct routes and vehicle types, and it increased the precision of the target cluster with respect to morphological features alone. These results show that color complements self-supervised features for route-level clustering, provided that it does not dominate the representation. The dataset is available under MIT license at: https://www.kaggle. com/datasets/irvingvasquez/public-bus-images

Keywords—Unsupervised clustering, vehicle identification, DI-NOv2, color histograms, early fusion, public transport.

## I. INTRODUCTION

Public transport is the backbone of mobility in large cities, and its efficient management requires continuous information about which vehicles operate on each corridor, how often they pass, and whether they follow their assigned routes. Intelligent Transportation Systems (ITS) increasingly obtain this information from the surveillance cameras already installed along urban roads. These cameras generate thousands of hours of video, far more than can be inspected or annotated manually, so methods that extract useful information from such footage without human supervision are of great practical value.

In this work, we address the problem of grouping public transport vehicles captured by a fixed surveillance camera into semantically meaningful categories, such as transit routes and operators, without using labeled data. The problem is challenging for two complementary reasons. First, buses operating on different routes often share the same body structure and differ mainly in their paint schemes. Second, vehicles of different types, such as full-size buses, microbuses (small buses), and vans, may share similar colors while differing substantially in shape. A useful representation must therefore capture both morphology and color.

Most vision-based vehicle classification systems are supervised and rely on large, site-specific annotated datasets with fixed taxonomies in which “bus” is a single class [1]–[3]. Selfsupervised learning has reduced the need for labels during representation learning [4], but in the vehicle domain it has mainly been used as pre-training for a supervised classifier [5]. Unsupervised clustering, on the other hand, has been applied mainly to vehicle re-identification, which groups images of the same physical vehicle rather than different vehicles that share a category [6]. Self-supervised Vision Transformers such as DINOv2 [7] now provide general-purpose descriptors that capture image structure well without fine-tuning. However, their features may under-represent pure chromatic information, which is crucial for distinguishing transit routes.

To address this problem, we propose a hybrid unsupervised clustering pipeline for public transport vehicles. Buses are first detected and tracked in the video, and a single crop is stored per vehicle. Each crop is then described by two complementary descriptors: a morphological descriptor given by the class token of a frozen DINOv2 model, and a color descriptor given by hue and saturation histograms computed in the HSV color space. Both descriptors are normalized and combined through weighted early fusion, and the fused vectors are projected with UMAP and grouped with K-Means, as illustrated in Fig. 1. We apply the pipeline to images of buses operating on one of the main arterial roads of Mexico City, and evaluate whether it can isolate the buses of the Red de Transporte de Pasajeros (RTP), the city’s public bus network, as a coherent cluster.

The main contributions of this paper are as follows:

• An unsupervised pipeline that fuses frozen selfsupervised morphological features with handcrafted color histograms to cluster public transport vehicles captured by a surveillance camera, without requiring labels at any stage of feature extraction or clustering.

![](images/a076210fb7839628dd5f7087479d11dc40c92292eb6d2ac6bc7a32fa11756ce7.jpg)  
Fig. 1. Overview of the proposed unsupervised clustering pipeline. Buses are detected and tracked in surveillance video, each crop is described by frozen DINOv2 morphological features and HSV color histograms, both descriptors are combined through weighted early fusion, and the fused vectors are projected with UMAP and grouped with K-Means. Bus illustrations were generated with AI.

• A dataset of 7,972 bus images extracted from about 1,460 hours of video of a Mexico City arterial road, manually annotated for evaluation and publicly available at https://www.kaggle.com/datasets/irvingvasquez/ public-bus-images.

• An ablation study on the fusion weight showing that a moderate color contribution increases the precision of the target cluster with respect to morphological features alone, whereas an excessive color weight leads to oversegmentation or to the collapse of the clustering space.

The remainder of this paper is organized as follows. Section II reviews related work on vehicle classification, selfsupervised learning, and clustering. Section III describes the dataset and the proposed clustering pipeline. Section IV presents the quantitative and qualitative results. Finally, Section V concludes the paper and outlines future work.

## II. RELATED WORK

Vision-based vehicle classification has evolved from handcrafted descriptors, such as Haar-like features and histograms of oriented gradients (HOG), combined with shallow classifiers such as support vector machines and AdaBoost, to convolutional neural networks trained end-to-end [1]. Supervised deep models now reach high accuracy. Butt et al. [2] finetuned several CNNs on 10,000 images of six vehicle types captured under adverse illumination and reported 99.68% accuracy, and Neupane et al. [3] fine-tuned YOLO detectors on nearly 30,000 manually annotated samples of seven classes, including buses. The latter study also shows the cost of domain shift: accuracy roughly doubled (from at most 30% to 71%) only after fine-tuning on data from the target cameras. These systems therefore depend on large, site-specific annotated datasets and on fixed taxonomies in which “bus” is a single class, so they cannot distinguish buses of different routes or operators.

Self-supervised learning (SSL) reduces this dependence by learning representations from unlabeled images [4]. In the vehicle domain, Ma and Yang [5] showed that Vision Transformer features learned with DINO and data2vec, combined with wheel-position cues, reach 97.2% top-1 accuracy on the 13 axle-based classes of the U.S. Federal Highway Administration. However, SSL is used there only as pre-training for a supervised classifier, and buses again form a single class.

Deep clustering removes labels altogether by coupling a representation-learning module with a clustering module [8]. Zhou et al. [8] group these methods into multistage, generative, iterative, and simultaneous approaches; our pipeline belongs to the multistage family, in which features are extracted first and clustered afterwards. Hou et al. [9] note that, before deep learning, image clustering relied on hand-crafted descriptors such as color histograms, and identify robustness to noise and real-world applicability as open challenges for deep image clustering. In the vehicle domain, clustering has mainly been used for unsupervised re-identification (Re-ID). Qiu et al. [6], for instance, generate pseudo-identity labels with a camera-aware DBSCAN scheme and refine them with a focal contrastive loss, outperforming previous unsupervised methods on VeRi-776 and VERI-Wild. Re-ID, however, groups images of the same physical vehicle, whereas our goal is to group different vehicles that share a semantic category.

In summary, existing work either classifies vehicles into coarse, predefined types [2], [3], [5] or matches individual vehicles across cameras [6]. To our knowledge, the unsupervised, category-level partitioning of the bus class, in which vehicles of different routes share a body structure and differ mainly in livery, has not been addressed. Moreover, none of the reviewed works combines self-supervised deep features with explicit color descriptors, even though color is a decisive cue for route identity. This work addresses both gaps.

![](images/14adddb10c3216c812f2254c15c9b0bf8ffd6861164a4fe2933be6345b40b498.jpg)  
Fig. 2. Example images from the dataset, cropped from the surveillance video after bus detection and tracking. Because YOLO26 does not distinguish between bus subtypes, the dataset includes full-size buses, microbuses, and vans.

## III. MATERIALS AND METHODS

## A. Dataset

The dataset was constructed from approximately 1,460 hours of continuous video recorded during June and July 2026. Recordings were acquired with a single fixed camera mounted at a height of 5 m and positioned 30 m from Circuito Interior, one of the main arterial roads of Mexico City. The camera was oriented toward the roadway, providing a consistent view of passing traffic, including public transport buses operating on multiple routes.

1) Bus Detection: Given the large volume of recorded video, manual annotation of the full videos was impractical. The videos were therefore preprocessed using the YOLO26 object detector [10] combined with a multi-object tracker. Only detections belonging to the bus class were retained, and a single image was stored for each tracked vehicle to avoid redundant samples of the same bus. To prevent truncation of the vehicle body, each bounding box was expanded by 10% before cropping. This process yielded a total of 7,972 cropped images of public transport vehicles. Because the detector does not distinguish between subtypes, this set includes buses, microbuses, and vans, all of which were classified as bus by YOLO26. Fig. 2 shows examples of the resulting images.

2) Ground-Truth Annotation: Although the proposed clustering approach is fully unsupervised, ground-truth labels were required to evaluate its performance. To this end, all 7,972 images were manually annotated. These labels were used exclusively for evaluation and were not involved at any stage of feature extraction or clustering. Two hierarchical binary labels were assigned to each image: is bus and is RTP bus, where the latter is a subset of the former. The is bus label indicates whether the crop contains a full-size bus and was introduced to assess the precision of the YOLO26- based preprocessing stage, given that smaller vehicles such as microbuses and vans were also detected under the bus class. The is RTP bus label identifies vehicles belonging to the Red de Transporte de Pasajeros (RTP), Mexico City’s public bus network, which constitute the target class of this study.

## B. Bus Clustering

The objective of this stage is to group the detected vehicles into clusters that correspond to semantically meaningful categories, such as bus routes and vehicle types, without relying on labeled data. The proposed pipeline consists of three steps: (i) extraction of complementary morphological and color descriptors from each image, (ii) early fusion of both descriptors into a single weighted feature vector, and (iii) unsupervised clustering of the fused representations. The following subsections describe each step in detail.

1) Feature Extraction: Buses operating on different routes often share similar body structures but differ in their paint schemes, whereas vehicles of different types (e.g., full-size buses, microbuses, and vans) may exhibit similar colors while differing substantially in shape. To capture both sources of variation, two complementary types of features were extracted from each image: deep morphological features and handcrafted color features.

Morphological features: These features were extracted using DINOv2 [7], a family of Vision Transformer (ViT) [11] models trained through self-supervised learning. Unlike textguided approaches such as CLIP [12], which depend on paired image and caption data, DINOv2 learns visual representations from images alone. Its training objective combines an imagelevel self-distillation loss, computed on the class token between a student and a teacher network observing different crops of the same image, with a patch-level masked image modeling loss inherited from iBOT [13].

A key property of DINOv2 is that its features perform well as frozen, general-purpose descriptors, without requiring task-specific fine-tuning. Its authors report strong results with simple linear classifiers and nearest-neighbor retrieval across a wide range of benchmarks, including fine-grained vehicle recognition on Stanford Cars [7]. These properties make DINOv2 well suited to an unsupervised setting such as ours, in which no labeled data are available for fine-tuning and the cluster structure depends directly on the geometry of the embedding space.

In this work, we employed the publicly available facebook/dinov2-small checkpoint, which corresponds to the ViT-S/14 variant of DINOv2 distilled from the ViT-g/14 teacher. Its main architectural characteristics are summarized in Table I. The model was used exclusively as a frozen feature extractor: it was loaded in evaluation mode, gradient computation was disabled, and no fine-tuning or retraining was performed. For each image $I _ { i } ,$ the [CLS] token of the final hidden layer was taken as a global descriptor of the vehicle’s structure, yielding a morphological feature vector

TABLE I ARCHITECTURE OF THE DINOV2 VIT-S/14 CHECKPOINT (F A C E B O O K/D I N O V2-S M A L L) USED AS THE MORPHOLOGICAL FEATURE EXTRACTOR.
<table><tr><td>Property</td><td>Value</td></tr><tr><td>Architecture</td><td>Vision Transformer (ViT-S/14)</td></tr><tr><td>Transformer blocks</td><td>12</td></tr><tr><td>Attention heads</td><td>6</td></tr><tr><td>Embedding dimension</td><td>384</td></tr><tr><td>Feed-forward network</td><td>MLP with GELU, ratio 4 (1536)</td></tr><tr><td>Patch size</td><td>14 × 14</td></tr><tr><td>Register tokens</td><td>None</td></tr><tr><td>Parameters</td><td>≈22 M</td></tr><tr><td>Pretraining data</td><td>LVD-142M (self-supervised)</td></tr><tr><td>Training regime</td><td>Distilled from ViT-g/14</td></tr><tr><td>Weight precision</td><td>float32</td></tr></table>

$$
\mathbf { f } _ { i } ^ { \mathrm { m o r p h } } \in \mathbb { R } ^ { 3 8 4 } .\tag{1}
$$

Color features: To capture the route-specific paint schemes that distinguish otherwise structurally similar buses, color histograms were computed in the HSV color space. HSV was preferred over RGB because it separates chromatic information from intensity, making the descriptor less sensitive to the illumination changes that occur throughout the day in outdoor recordings. Accordingly, only the Hue and Saturation channels were used, and the Value channel was discarded.

Because the expanded bounding boxes include surrounding background such as asphalt, sky, and other vehicles, a central crop covering 50% of the image area was applied before computing the histograms, so that the descriptor predominantly reflects the vehicle body. One-dimensional histograms with 32 bins were then computed for the Hue and Saturation channels and concatenated, yielding a color feature vector

$$
\mathbf { f } _ { i } ^ { \mathrm { c o l o r } } = \left[ \mathbf { h } _ { i } ^ { H } \parallel \mathbf { h } _ { i } ^ { S } \right] \in \mathbb { R } ^ { 6 4 } ,\tag{2}
$$

where $\mathbf { h } _ { i } ^ { H }$ and $\mathbf { h } _ { i } ^ { S }$ denote the Hue and Saturation histograms of image $I _ { i } ,$ and ∥ denotes concatenation.

2) Early Feature Fusion: The morphological and color descriptors differ in both dimensionality (384 versus 64) and numerical scale. To make them comparable, each vector was first L2-normalized independently:

$$
\hat { \mathbf { f } } _ { i } ^ { \mathrm { m o r p h } } = \frac { \mathbf { f } _ { i } ^ { \mathrm { m o r p h } } } { \Vert \mathbf { f } _ { i } ^ { \mathrm { m o r p h } } \Vert _ { 2 } } , \qquad \hat { \mathbf { f } } _ { i } ^ { \mathrm { c o l o r } } = \frac { \mathbf { f } _ { i } ^ { \mathrm { c o l o r } } } { \Vert \mathbf { f } _ { i } ^ { \mathrm { c o l o r } } \Vert _ { 2 } } .\tag{3}
$$

This normalization ensures that both descriptors lie on the unit hypersphere, so that their relative influence is governed by the fusion weights rather than by their original dimensionality.

An early fusion strategy was then adopted, in which the normalized descriptors were weighted and concatenated into a single feature vector prior to clustering:

$$
\mathbf { v } _ { i } = \left[ \alpha \hat { \mathbf { f } } _ { i } ^ { \mathrm { m o r p h } } \parallel \left( 1 - \alpha \right) \hat { \mathbf { f } } _ { i } ^ { \mathrm { c o l o r } } \right] \in \mathbb { R } ^ { 4 4 8 } ,\tag{4}
$$

where $\alpha ~ \in ~ [ 0 , 1 ]$ controls the relative contribution of the morphological features and $1 - \alpha$ that of the color features. Because both components are unit-normalized, the squared Euclidean distance between two fused vectors decomposes as

$$
\begin{array} { r l } & { \| \mathbf { v } _ { i } - \mathbf { v } _ { j } \| _ { 2 } ^ { 2 } = \alpha ^ { 2 } \| \hat { \mathbf { f } } _ { i } ^ { \mathrm { m o r p h } } - \hat { \mathbf { f } } _ { j } ^ { \mathrm { m o r p h } } \| _ { 2 } ^ { 2 } } \\ & { \qquad + \left( 1 - \alpha \right) ^ { 2 } \| \hat { \mathbf { f } } _ { i } ^ { \mathrm { c o l o r } } - \hat { \mathbf { f } } _ { j } ^ { \mathrm { c o l o r } } \| _ { 2 } ^ { 2 } , } \end{array}\tag{5}
$$

so that α directly regulates the trade-off between structural and chromatic similarity. Three weighting schemes were evaluated, with $\alpha \in \{ 0 . 3 5 , 0 . 5 0 , 0 . 6 5 \}$ , in order to analyze the effect of this balance on the resulting cluster structure.

3) Clustering and Evaluation: Distance-based clustering algorithms such as K-Means tend to degrade in highdimensional spaces, where pairwise distances become less discriminative. Therefore, the fused vectors were first projected onto a lower-dimensional manifold using Uniform Manifold Approximation and Projection (UMAP) [14], a nonlinear dimensionality reduction technique that preserves the local neighborhood structure of the data while retaining much of its global organization.

K-Means clustering [15] was subsequently applied to the reduced representations for each number of clusters $K \in$ $\{ 2 , \ldots , 3 0 \}$ . The optimal value of K was selected by jointly considering two internal validation criteria. The Elbow Method examines the within-cluster sum of squares as a function of K and identifies the point beyond which additional clusters yield diminishing reductions. The Silhouette Score [16] measures, for each sample, how similar it is to its own cluster relative to the nearest neighboring cluster, ranging from −1 to 1, where higher values indicate compact and well-separated clusters. This procedure was repeated independently for each value of $\alpha .$

## IV. RESULTS

The objective of the proposed method is to cluster the bus images and to determine whether the buses operated by RTP can be isolated into a coherent group without supervision. Accordingly, this section evaluates the proposed unsupervised early fusion approach against a baseline that clusters the images using morphological features alone. In addition, an ablation study was conducted to analyze the effect of the fusion weight α on clustering performance, considering three configurations: $\alpha ~ = ~ 0 . 6 5 , ~ \alpha ~ = ~ 0 . 5 0$ , and $\alpha ~ = ~ 0 . 3 5$ The resulting clusters were assessed both quantitatively, by comparing them against the manually annotated target class, and qualitatively, through visual inspection of their contents.

## A. Implementation Details

All components were implemented in Python and executed on a workstation equipped with an Intel Core i7 processor and an NVIDIA GTX 1080 GPU. To facilitate experimentation, feature extraction and clustering were implemented as independent stages. Morphological features were extracted with PyTorch [17] and Hugging Face Transformers [18] using the pretrained $\mathtt { f a c e b o o k } / \mathtt { d i n o v } 2 \mathtt { - s m a l 1 }$ checkpoint. Images were loaded with PIL, processed by the library’s default image processor in batches of 64, and the resulting class tokens were stored as NumPy arrays. Color histograms were computed with OpenCV [19] on the central crop of each image. Both descriptors were L2-normalized with scikit-learn [20], the fused matrix was embedded with the umap-learn library [14], and K-Means clustering and Silhouette analysis were performed with scikit-learn.

TABLE II  
CLUSTERING RESULTS WITH RESPECT TO THE TARGET CLASS (is RTP bus) FOR DIFFERENT METHODS AND VALUES OF α. M.C.: MORPHOLOGICAL CLUSTERING (DINOV2 FEATURES ONLY). E.F.: EARLY FUSION. ENTROPY (H) IS REPORTED IN BITS. BEST VALUES IN BOLD.
<table><tr><td>Method</td><td>K</td><td>H</td><td>Acc.</td><td>Prec.</td><td> $\overline { { { \bf R e c . } } }$ </td><td>F1</td></tr><tr><td>M.C.</td><td>11</td><td>1.175</td><td>0.9326</td><td>0.7703</td><td>0.6022</td><td>0.6759</td></tr><tr><td>E.F.  $\overline { { ( \alpha = 0 . 3 5 ) } }$ </td><td>3</td><td>0.012</td><td>0.6692</td><td>0.2606</td><td>0.9989</td><td>0.4133</td></tr><tr><td>E.F.  $( \alpha = 0 . 5 0 )$ </td><td>20</td><td>1.680</td><td>0.9328</td><td>0.8988</td><td>0.4774</td><td>0.6236</td></tr><tr><td>E.F.  $( \alpha = 0 . 6 5 )$ </td><td>14</td><td>1.059</td><td>0.9521</td><td>0.9507</td><td>0.6215</td><td>0.7516</td></tr></table>

## B. Evaluation Protocol

For each configuration, Table II reports the number of clusters K selected by the Silhouette and Elbow criteria, together with the Shannon entropy of the distribution of RTP buses across clusters, defined as

$$
H = - \sum _ { k = 1 } ^ { K } p _ { k } \log _ { 2 } p _ { k } ,\tag{6}
$$

where $p _ { k }$ denotes the fraction of all ground-truth RTP buses assigned to cluster k. The entropy equals zero when all RTP buses are grouped into a single cluster and increases as they become dispersed across multiple clusters. In addition, accuracy, precision, recall, and F1-score were computed by treating the cluster containing the largest number of RTP buses as the predicted RTP class and comparing its members against the manually annotated labels.

## C. Quantitative Analysis

The early fusion configuration with $\alpha = 0 . 6 5$ achieved the best overall performance, with the highest accuracy (0.9521), precision (0.9507), and F1-score (0.7516) among all methods. Compared with the morphological baseline, incorporating color information increased precision by 18 percentage points (from 0.7703 to 0.9507) and the F1-score by 7.6 points (from 0.6759 to 0.7516), while also reducing the entropy from 1.175 to 1.059 bits. In contrast, recall improved only marginally (from 0.6022 to 0.6215). This indicates that the main contribution of the color descriptor is to increase cluster purity: the dominant RTP cluster obtained through fusion contains far fewer non-RTP vehicles, most likely buses from other routes that share a similar body structure but differ in livery, which morphological features alone cannot reliably distinguish.

Nevertheless, the recall of approximately 0.62 obtained by both the baseline and the best fusion configuration shows that a substantial fraction of RTP buses (about 38%) is assigned to clusters other than the dominant one. This is consistent with the nonzero entropy values and suggests that the RTP class is not homogeneous in the feature space, possibly due to variations in viewpoint, illumination, partial occlusions, or differences among RTP vehicle models and paint schemes.

Performance did not vary monotonically with the color weight. With $\alpha \ : = \ : 0 . 5 0 .$ , precision remained high (0.8988), but recall dropped to 0.4774 and the entropy reached its highest value (1.680 bits), indicating that the RTP buses were fragmented across a larger number of clusters $( K = 2 0 )$ . As a result, the F1-score of this configuration (0.6236) fell below that of the morphological baseline. It should be noted that, because both descriptors are unit-normalized and the weights act quadratically on the squared Euclidean distance, $\alpha = 0 . 5 0$ assigns equal influence to both descriptors, whereas $\alpha = 0 . 3 5$ gives color roughly 3.4 times the influence of morphology.

The configuration with $\alpha = 0 . 3 5$ illustrates the failure mode of an excessive color weight. Although it yielded the lowest entropy (0.012 bits) and a near-perfect recall (0.9989), these values do not reflect a successful separation. With only three clusters, virtually all RTP buses fall into a single cluster, but this cluster also contains a large number of non-RTP vehicles, as reflected by a precision of only 0.2606 and the lowest accuracy and F1-score of all configurations (0.6692 and 0.4133, respectively). In other words, the low entropy is a consequence of the collapse of the clustering space rather than of an accurate grouping of the target class. This case highlights that entropy and recall must be interpreted jointly with precision, since each of them can be trivially optimized by a degenerate solution.

Accuracy should also be interpreted with caution. Because RTP buses represent a minority of the dataset, a trivial classifier that labels every image as non-RTP would already achieve a high accuracy. The F1-score and precision are therefore more informative indicators of the quality of the RTP cluster.

Overall, these results show that, with an appropriate fusion weight, the proposed methodology groups the majority of the target class into a single, highly pure cluster. This supports our hypothesis and suggests that the approach can facilitate the detection of public transport vehicles without the need for large annotated datasets.

## D. Qualitative Analysis

Visual inspection of the resulting clusters supports the quantitative findings. With $\alpha = 0 . 6 5 ,$ , the Silhouette analysis indicated an optimal value of $K \ = \ 1 4$ , and the clusters were highly homogeneous: vehicles with distinct shapes were correctly separated, while buses belonging to the same transit route remained chromatically consistent within their clusters. Fig. 3 shows twelve randomly selected members of the dominant RTP cluster.

Increasing the color weight to $\alpha = 0 . 5 0$ shifted the optimal number of clusters to $K = 2 0$ . In this configuration, cohesive groups were split into fragmented sub-clusters that appeared to be driven primarily by minor variations in lighting or saturation rather than by differences in vehicle structure.

![](images/2a82c8862779ac87892aeaa05e9038549af92589e2d4e8d00fb9ccf06fbbdb5f.jpg)  
Fig. 3. Randomly selected images from the dominant RTP cluster obtained with early fusion (α = 0.65).

When color dominated the fusion $( \alpha = 0 . 3 5 )$ , the clustering collapsed to $K = 3$ . The model became largely insensitive to structural differences, grouping morphologically distinct vehicles, such as vans and full-size buses, into the same cluster simply because they shared dominant colors.

Overall, these results confirm that color is a highly discriminative cue for identifying public transport routes, but that it must complement, rather than dominate, the morphological representation. A moderate color weight improves the purity of the target cluster, whereas over-prioritizing color degrades the structural understanding of the vehicles and leads either to over-segmentation or to the collapse of the clustering space.

## V. CONCLUSIONS

This study presented an unsupervised pipeline for clustering public transport vehicles captured by a surveillance camera, in which frozen DINOv2 morphological features and HSV color histograms are combined through weighted early fusion and grouped with UMAP and K-Means. Without using any labels during feature extraction or clustering, the best configuration isolated RTP buses in a cluster with higher precision and F1- score than morphological features alone. These results confirm that color provides information that self-supervised features alone do not capture well, since it separates buses that share a body structure but belong to different routes.

The ablation study also showed that the fusion weight is critical. A balanced weight fragmented the target class into many small clusters, whereas a color-dominant weight collapsed the space into three clusters that mixed vans, microbuses, and fullsize buses. Color should therefore complement, not replace, the morphological representation.

The approach has limitations. Some RTP buses were still assigned to secondary clusters and the number of clusters was selected with internal criteria that do not guarantee semantic correspondence. Future work will explore late fusion based on descriptor-specific distances combined with density-based algorithms such as HDBSCAN, which do not require the number of clusters to be fixed in advance. Evaluation on multiple cameras and routes will also be necessary to assess the generality of the method.

## REFERENCES

[1] M. A. Berwo, A. Khan, Y. Fang, H. Fahim, S. Javaid, J. Mahmood, Z. U. Abideen, and S. M.S., “Deep learning techniques for vehicle detection and classification from images/videos: A survey,” Sensors, vol. 23, no. 10, p. 4832, 2023.

[2] M. A. Butt, A. M. Khattak, S. Shafique, B. Hayat, S. Abid, K.-I. Kim, M. W. Ayub, A. Sajid, and A. Adnan, “Convolutional neural network based vehicle classification in adverse illuminous conditions for intelligent transportation systems,” Complexity, vol. 2021, no. 1, p. 6644861, 2021.

[3] B. Neupane, T. Horanont, and J. Aryal, “Real-time vehicle classification and tracking using a transfer learning-improved deep learning network,” Sensors, vol. 22, no. 10, p. 3813, 2022.

[4] T. Uelwer, J. Robine, S. S. Wagner, M. Hoftmann, E. Upschulte, S. Koni-¨ etzny, M. Behrendt, and S. Harmeling, “A survey on self-supervised methods for visual representation learning,” Machine Learning, vol. 114, no. 4, p. 111, 2025.

[5] S. Ma and J. J. Yang, “Image-based vehicle classification by synergizing features from supervised and self-supervised learning paradigms,” Eng, vol. 4, no. 1, pp. 444–456, 2023.

[6] M. Qiu, Y. Lu, X. Li, and Q. Lu, “Camera-aware differentiated clustering with focal contrastive learning for unsupervised vehicle reidentification,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 10, pp. 10 121–10 134, 2024.

[7] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, pp. 1–31, 2024.

[8] S. Zhou, H. Xu, Z. Zheng, J. Chen, Z. Li, J. Bu, J. Wu, X. Wang, W. Zhu, and M. Ester, “A comprehensive survey on deep clustering: Taxonomy, challenges, and future directions,” ACM Computing Surveys, vol. 57, no. 3, pp. 69:1–69:38, 2024.

[9] H. Hou, S. Ding, C. G. Cao, X. Xu, L. Guo, and X. Li, “A comprehensive survey of image clustering based on deep learning,” Pattern Recognition, vol. 172, p. 112590, 2026.

[10] Ultralytics, “Ultralytics YOLO26,” https://github.com/ultralytics/ ultralytics, 2025.

[11] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proceedings of the International Conference on Learning Representations (ICLR), 2021.

[12] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in Proceedings of the International Conference on Machine Learning (ICML). PMLR, 2021, pp. 8748–8763.

[13] J. Zhou, C. Wei, H. Wang, W. Shen, C. Xie, A. Yuille, and T. Kong, “iBOT: Image BERT pre-training with online tokenizer,” in Proceedings of the International Conference on Learning Representations (ICLR), 2022.

[14] L. McInnes, J. Healy, N. Saul, and L. Großberger, “UMAP: Uniform manifold approximation and projection,” Journal of Open Source Software, vol. 3, no. 29, p. 861, 2018.

[15] S. Lloyd, “Least squares quantization in PCM,” IEEE Transactions on Information Theory, vol. 28, no. 2, pp. 129–137, 1982.

[16] P. J. Rousseeuw, “Silhouettes: A graphical aid to the interpretation and validation of cluster analysis,” Journal of Computational and Applied Mathematics, vol. 20, pp. 53–65, 1987.

[17] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan et al., “PyTorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019.

[18] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi et al., “Transformers: State-of-the-art natural language processing,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2020, pp. 38–45.

[19] G. Bradski, “The OpenCV library,” Dr. Dobb’s Journal of Software Tools, vol. 25, no. 11, pp. 120–123, 2000.

[20] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.