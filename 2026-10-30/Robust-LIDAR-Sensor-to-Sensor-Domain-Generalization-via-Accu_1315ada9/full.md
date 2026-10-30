# Robust LIDAR Sensor-to-Sensor Domain Generalization via Accumulated-Point Scene Completion

Christoph B. Rist<sup>1</sup> , Larissa T. Triess , Markus Enzweiler<sup>2</sup> , Dariu M. Gavrila<sup>1</sup>

Abstract— We study domain generalization for LIDAR semantic segmentation. Deep neural networks achieve strong performance on 3D point clouds, yet real-world LIDAR exhibits sensorspecific sampling patterns that hinder transfer across datasets. In many applications, performance on completely unseen sensors is crucial. A model should preferably work on a new setup without prior access to its data or labels. To this end, we introduce a source-only training scheme that adds an occupancy prediction auxiliary objective. We aggregate source-domain scans into a simple world model and use it as the supervision signal, replacing single-scan standard segmentation training.

A key property of our approach is that it does not alter the input scan. During training and testing the network consumes the same single-frame LIDAR input, only the supervision signal is enriched via the occupancy objective. At deployment, there is no runtime overhead as the auxiliary head can be dropped for pure segmentation, or kept to obtain scene completion alongside semantic segmentation. In an experimental evaluation across distinct LIDAR datasets, we demonstrate that a model’s generalization to new domains can be significantly affected by how it is supervised. In certain setups, performance on unseen domains improves without any explicit domain adaptation (no target scans, no target labels, no sensor priors) and without prediction refinement. At the same time, our method maintains competitive source-domain segmentation performance and incurs no additional inference cost.

## I. INTRODUCTION

Highly automated vehicles and mobile robots rely on deep learning–based perception models that typically require large volumes of annotated data. Yet these systems must operate across diverse domains induced by weather, geography, traffic culture, and sensor generations. Since data collection and annotation is costly and time-consuming, repeating them for every new domain is impractical. Unsupervised domain adaptation (UDA) targets this challenge with automated methods that adapt models to new domains without requiring manual annotations in the target domain. Going further, domain generalization (DG) seeks to build models using only sourcedomain data without any access to potential target domains (neither labels nor raw sensor data) and to maximize performance on these unseen targets at test time. Models that generalize well to unseen domains reduce retraining efforts when experimenting with new platforms and sensor setups. They accelerate deployment and iteration, and they improve robustness in safety critical systems.

Generic, feature-based domain adaptation (DA) techniques struggle with LIDAR domain shifts dominated by samplingpattern changes, motion distortion, and mounting geometry [1]. Recent LIDAR-tailored methods that leverage sensor geometry, re-sample beam layouts [2], [3], [4], or use 3D surface/occupancy cues [5], [6], [7], [8], prove substantially more effective in practice. These include sensor-aware simulation, surface and occupancy completion from multi-scan aggregation, and geometry-guided label transfer across registered frames [9]. In parallel, representation learning for LIDAR has gained traction. Occupancy, surface completion, depth auxiliaries, and contrastive objectives help distill transferable 3D features from point clouds. The resulting encoders can be fine-tuned with fewer labels [6], [7]. However, many of these pipelines still rely on some access to the target domain during training, either by means of unlabeled scans for alignment or pseudo-labels for refinement.

![](images/53d504229823b71d568726d8c77367799ed1691e64af9cb26e4e250cf73f9283.jpg)  
Fig. 1: Sensor-to-sensor Domain Generalization with richer supervision: Comparing two training signals on the source domain (nuScenes) and their effect on the target domain (Waymo) segmentation task (blue: car, green: vegetation, magenta: ground). The left column uses standard per-scan semantic segmentation, the right column uses accumulated multi-scan points as supervision signal. When evaluated on an unseen target domain sensor (bottom row), the accumulated points training is robust against sensor ground patterns and yields correct predictions with consistent boundaries.

While these strategies improve robustness, explicit domain mappings require sensor-specific priors, depend on assumptions about future sensor distributions and may overfit to particular heuristics. Moreover, they require considerable engineering at training time and their gains can saturate when sensor discrepancies exceed what feature alignment or entropy minimization can bridge.

We address LIDAR DG by exploiting geometric structure through an occupancy prediction pretext task during training. Crucially, unlike prior DG work that modifies the input scans (e.g., sensor sparsification, pattern simulation, or domain stylization), our study keeps the input LIDAR data unchanged and alters only the supervision signal.

Fig. 1 compares single-scan supervision versus accumulated-points supervision and their effect on an unseen target example side-by-side. Following [6], [10], we attach an occupancy decoder and loss to the shared encoder features during training. At test time, the occupancy head is discarded and we use the encoder–segmentation head to perform semantic segmentation on target scans. Our training requires only source-domain LIDAR and labels.

We aggregate training targets and evaluate on three large scale LIDAR semantic segmentation datasets SemanticKITTI [11], nuScenes [12] and Waymo [13], which differ in LIDAR models and sampling patterns. Our approach assumes that source sequences can be registered reliably to generate aggregated supervision.

Summarized, our contributions are:

• We propose a geometry-aware auxiliary task that uses accumulated points as supervision during source-domain training.

• Our formulation is source-only (no target labels, target sensor models, or target scans), sensor-agnostic, and alters only the supervision signal via an auxiliary head that is discarded at test time.

• We evaluate the semantic segmentation task on three large-scale LIDAR datasets covering distinct sensor models and sampling patterns, and provide ablations against standard segmentation training.

## II. PREVIOUS WORK

Domain shifts in LIDAR arise both from sensor-to-sensor (e.g., beam layout, rotation rate, range noise) and scene-toscene changes (e.g., geography, traffic mix). Prior work addresses these gaps through generic feature-learning approaches that do not rely on LIDAR specifics, and LIDAR-aware approaches that exploit 3D geometry or measurement physics. We briefly mention works that use entropy minimization as feature-learning technique used in unsupervised DA. While general feature learning is modality-agnostic and broadly applicable, its effectiveness on LIDAR can be limited by sparsity, severe class imbalance, and sensor-induced sampling artifacts. This motivates the LIDAR-specific strategies reviewed next, which make use of geometric and surface completion, realistic re- and subsampling, and accumulation-based training. For multi-modal DA and DG we refer to the recent survey [14], and for a LIDAR-focused overview to [15].

## A. Entropy Minimization

Entropy minimization is a baseline strategy for UDA, encouraging confident predictions by minimizing the model’s output entropy on unlabeled target data without exploiting sensor- or task-specific structure [16], [17], [18]. In dense prediction settings, entropy yields a straightforward objective on the per-point categorical distribution. Beyond pointwise entropy, aligning the structure of predictions proves beneficial. Adversarial training in the output space reduces distribution shift by matching target and source prediction layouts rather than only local entropy [16]. To prevent learning from being dominated by confident target predictions, a maximumsquares loss reshapes the gradient landscape by suppressing gradients for well-classified samples while emphasizing misclassified ones [17]. From a statistical-alignment view, Minimal-Entropy-Correlation-Alignment shows that entropy minimization is induced by optimal alignment of secondorder statistics between source and target. Using geodesic instead of Euclidean distances better respects the non-Euclidean curvature of these statistics on symmetric positive-definite manifolds [18].

## B. Domain Adaptation

A common pattern in UDA for images is to attach multimodal auxiliary tasks that are easy to supervise on both domains, most prominently monocular depth or geometry signals used to guide features and alignment [19], [20], [21], [22]. The premise is that learning on complementary, weakly supervised targets produces representations that transfer better to the target domain. For LIDAR, a pre-training strategy alternating on heterogeneous datasets shows similar robustness benefits [23].

Several works exploit 3D geometry to obtain domaininvariant features by predicting surfaces, occupancy, or both [5], [6], [7], [8]. Complete & Label [5] first completes sparse voxels to a canonical surface representation and then segments it. Because the intermediate 3D proxy is independent of a sensor’s sampling pattern, the approach bridges sensor-to-sensor gaps and outperforms generic entropy-based DA. Follow-ups predict occupancy (and sometimes semantics) in the neighborhood of a query point and use such heads as pre-training to improve segmentation and detection with limited target labels [6], [7]. SALUDA [8] replaces explicit completion with an implicit surface head jointly trained with segmentation so that source and target share a common surface representation. This yields gains on real-to-real and sim-toreal benchmarks. More broadly, occupancy prediction has also been shown to be an effective proxy task for learning transferable 3D representations. SPOT [24] pre-trains LiDAR backbones via semantic occupancy prediction and beam resampling augmentation, yielding robust cross-dataset transfer across downstream detection and segmentation tasks under varying sensor configurations.

Another line performs explicit sensor-to-sensor mapping by resampling the training data. This can operate on single scans or on temporally accumulated point clouds [2], [3], [4]. Instead of predicting a canonical surface as in [5], these methods produce semi-synthetic scans with target-like properties. To assess the fidelity of such artificial scans, Triess et al. propose a learned realism metric for LIDAR point clouds, showing that local point cloud realism correlates with downstream segmentation performance [25]. Surface-aware re-simulation can be computationally heavy when explicit recovery is needed [3], while z-buffer ray casting on accumulations is faster [4]. A second stage can align statistics between domains, for example by geodesic correlation alignment to generalize beyond a single sensor-to-sensor pair toward dataset-to-dataset shifts [18], [26].

![](images/a9baa6edd6a0b0c95a07a925104f177b93b2a1ce7a0ea272fb34b4b96df7ebd9.jpg)  
Fig. 2: Proposed method: We add an occupancy prediction task based on accumulated LiDAR points while training on the source domain to improve model generalization (orange blocks). When this task is removed at test time, a vanilla semantic segmentation model remains. Even during domain generalization training, the input to the model is unchanged. Data for the occupancy prediction is generated automatically on the source domain

Finally, mixing samples from different sensors and training a student with pseudo-labels from a teacher improves target performance without requiring target annotations [27].

## C. Domain Generalization

Kim et al. [2] propose DGLSS, which trains on a single source and simulates unseen sensors by random subsampling with the main objective to generalize from denser to sparser sensor domains. Two density-aware constraints align features between dense and sparsified views and stabilize class relations. This label-free strategy increases robustness but requires a sufficiently dense source sensor. The inverse direction from sparse to dense sensor domains is harder to achieve with DGLSS: The augment baseline and the SCC-only variant outperform the full method in this setting. Sanchez et al. [9] ag-´ gregate multiple sweeps and propagate labels to accumulated points, improving robustness across sensors. Their method includes an explicit geometric association step that assigns labels to accumulated points using hand-designed rules. The resulting labeled points are then used both as supervision and as (subsampled) model input. In contrast, our approach introduces a free-space class directly into the accumulation process and trains jointly on occupied and free-space points, without modifying the input representation.

Kim et al. [28] target generalization by explicitly learning across diverse densities within the source domain. A densityaware feature module and targeted density augmentation strengthen robustness of LIDAR segmentation. Ryu et al. [29] synthesize target-like frames from accumulated source clouds using detailed sensor models that account for ego-motion, spin distortions, and sampling patterns. This removes the need for target data but requires accurate sensor specifications to apply realistic ego-motion and spin distortions as well as sampling patterns. Zhao et al. [30] define a DG setting for synthetic-toreal indoor segmentation and improve robustness with pointlevel pattern augmentations and instance mixing.

## D. Enriched Supervision Signal

All of the above DG methods operate exclusively on source data but alter the source-domain method inputs via subsampling, accumulation, mixing, or sensor-aware rendering to expose the model to domain diversity without any target samples. There are notable works in image classification/segmentation showing that enriching the supervision signal alone can improve robustness. Incorporating self-supervised tasks (jigsaw puzzle [31], rotation prediction [32]), predicting depth along with labels in multi-task learning [33], or privileged simulation depth as an alternative label modality [34] have led to more domain-invariant features. These works demonstrate that a model’s generalization can be significantly affected by richer supervision, all while keeping the input data unchanged.

## III. PROPOSED METHOD

## A. Overview

We aim to train a LIDAR semantic segmentation network that generalizes to unseen sensor domains without altering the input scan at test time. To this end, we introduce an additional occupancy prediction pretext task during training, see Fig. 2. The point-cloud semantic segmentation problem is defined over a dataset $\mathcal { D } = \{ ( P _ { i } , y _ { i } , \mathbf { \bar { \mu } } p _ { i } ) \} _ { i = 1 } ^ { N }$ of LIDAR sweeps $P _ { i }$ point-wise labels $y _ { i } .$ , and ego poses $p _ { i }$ . The model predicts y<sub>i</sub> from $P _ { i }$

Using the ego-motion, we aggregate successive sweeps into a denser, static world frame and samplefree-space points along each registered LIDAR ray between the sensor origin and the first return. This yields an enriched supervision signal with more labeled surface points via temporal accumulation, and additional free-space points that supervise scene completion. At test time on both source and unseen target domains, the occupancy branch is simply ignored and the network is used as a standard point-cloud semantic segmenter.

## B. Training Data

We construct the occupancy supervision by accumulating LIDAR scans over time in a common world frame using the provided ego-motion. For nuScenes, we refine the provided poses with an ICP alignment over local scan windows before fusion, improving registration accuracy. We retain an individual LIDAR point in the accumulation only if its distance to its own recording sensor origin is below 100 m. This cap filters far-range, typically noisier returns and reduces the resulting point density. To keep training efficient, we limit the number of retained points per voxel, which is particularly important near the ego vehicle where raw point density is highest. After aggregation, we place a 3D voxel grid with an edge length of 16 cm over the dense point cloud to thin out high-density regions. When downsampling within a voxel, we preferentially retain labeled points to maximize the density of semantic supervision. This higher density of labeled surface points and reduced overlap among points describing the same geometry improves training efficiency as each retained point carries more supervision signal. This procedure defines our canonical, training domain shared across datasets. Visually, the right column of Fig. 3 shows that the accumulated point clouds exhibit similar spatial structure across KITTI, nuScenes, and Waymo.

To avoid painting motion trails into the fused world model, we explicitly remove scan points belonging to dynamic objects from all non-current frames. We identify dynamic objects via either a dedicated dynamic flag (KITTI) or via 3D annotated bounding boxes with temporal interpolation to the LIDAR timestamp (nuScenes, Waymo). Concretely, each non-current sweep is time-aligned to the reference frame, its boxes are interpolated to the sweep time, and all points falling inside any dynamic box are excluded from accumulation. Only the dynamic-object points observed at the current ego frame are retained in the resulting training example, preserving instantaneous geometry without leaving history traces in the static world model.

During training, we generate free-space supervision by sampling points along registered LIDAR rays between the sensor and the first return, labeling them as non-occupied. The model thus learns semantic segmentation on a larger set of surface points from temporal fusion, and a scene-completion signal via one extra “free-space” class on sampled ray points.

We use only raw 3D coordinates (no reflectivity), following [9], [28]. We do not sub-sample or augment the source-domain sweeps based on domain assumptions. Only raw 3D points are consumed by the pre-processing and the model. Table I lists the training split sizes (scenes and annotated points, in millions) for KITTI, nuScenes, and Waymo, including pairwise and joint combinations. As expected, temporal accumulation increases the number of semantically labeled points compared to single sweeps.

## C. Model

Our network (Fig. 2) takes a raw 3D point list, voxelizes it into point pillars, and processes them with a multi-resolution backbone [35]. Predictions are produced by a Coordinate MLP head that can be queried at arbitrary 3D locations [10]. This design makes it straightforward to train either only on the input LIDAR scan in a default semantic segmentation setup or on the accumulated surface points for semantic segmentation and sampled free-space points for occupancy, without changing the output head. In practice, we query the head at the accumulated LIDAR point coordinates and at free-space sample locations. At test time we simply drop the free-space output class, normalize probabilities for only the semantic classes and query only the LIDAR points. Alternatively, the accumulated-point training can be implemented on a voxel grid by pooling point labels into voxels and supervising a scene-completion grid. Conversely, standard backprojection (“unpooling”) used in (sparse)-voxel and BEV-conv architectures can map grid outputs back onto the original surface and free-space samples to provide an equivalent supervision signal.

## IV. EXPERIMENTS

## A. Existing Approaches and Label Sets

We compare against three representative methods: C&L (Yi et al. [5]) is an unsupervised domain adaptation approach that leverages access to unlabeled target-domain data. DGLSS (Kim et al. [2]) is a domain generalization method that simulates unseen sensors by subsampling the input scan. 3DLabelProp (Sanchez et al. [9]) also accumulates points, but uses´ them both as supervision and (subsampled) model input. Both [5] and [2] employ MinkowskiNet [36]-based architectures and deliver source-domain segmentation performance comparable to ours.

We consider the widely used KITTI, nuScenes, and Waymo LIDAR segmentation datasets. All three are automotive datasets with large numbers of outdoor LIDAR scans captured in sequences and annotated with point-wise semantics. Because each dataset defines its own taxonomy, we map classes to two variants of a 10-class intersection label set. Variant 10- A, introduced by Yi et al.[5], contains: car, bicycle, motorcycle, truck, other vehicle, person, drivable surface, sidewalk, terrain, vegetation, infrastructure. Variant 10-B from Kim et al.[2] merges terrain and vegetation and distinguishes sidewalk from walkable, yielding a slightly different 10-class mapping used in semantic DG evaluation.

![](images/65e72cc305a86f65bf4839900944c0d0e3fff58ca30607b543c782a451ec326c.jpg)  
(a) LiDAR scan bird’s-eye view

![](images/8b46925f4e05e52b8cbdd9bb2de4b93efe9719b9cdb3627d40e3ddefa9fb5071.jpg)  
(b) LiDAR scan perspective view

![](images/05ac1a7f326d534199aafad59b64e696d08019ba237e2db8277536fb4f7170de.jpg)  
(c) World-frame accumulated points  
Fig. 3: Dataset characteristics. Accumulated world-frame ground truth is semantically color-coded by the 10-A label set. Rows illustrate differing LIDAR configurations across datasets. KITTI uses a single 64-beam LIDAR, yielding up to 115k points within 100 ms. nuScenes fuses the five most recent 32-beam scans over 250 ms. Waymo employs multiple LIDARs to reduce blind spots.

## B. Dataset Characteristics

The LIDAR configurations differ substantially. KITTI is recorded with a 64-beam Velodyne HDL-64 at 10 Hz; nuScenes uses a 32-beam Velodyne VLP-32 at 20 Hz; the Waymo LIDAR segmentation dataset employs multiple proprietary roof-mounted LIDARs in a coordinated setup. Beyond beam count and spin rate, vertical beam elevation patterns and multi-sensor layouts yield distinct sampling geometries and point densities, visible in our scan comparison (Fig. 3). On nuScenes, we follow common practice for its sparser frames and use the last five LIDAR scans as input. Table I lists the numbers of training samples, semantic labels, and accumulated point labels per dataset and dataset combination. We use the default training/validation splits of each dataset. Accumulation increases the number of labeled LIDAR points by roughly 50 % on KITTI and Waymo, and by about ∼3.5× on nuScenes.

## C. Single-Source Domain Generalization

The single-source DG results are reported in the top sections of Table II and Table III. All values are mean Intersection-over-

Union (mIoU, %) for semantic segmentation on the respective validation splits.

a) Results on 10-B.: Table II compares our sourceonly segmentation baseline, our proposed accumulated-points (scene-completion) training, and DGLSS [2]. Source-domain performance is nearly unchanged between segmentation and accumulated-points training; only on KITTI we observe a slight drop with accumulated points. DGLSS attains slightly higher source-domain scores. On the target domain, trends depend on the dataset pair: Training on nuScenes or Waymo yields substantial improvements toward the other dataset with our accumulated-points supervision (8 % and 7 %, highlighted in green). In contrast, generalization to or from KITTI slightly degrades relative to the segmentation baseline. Accordingly, our method modestly outperforms DGLSS for Waymo,→,nuScenes transfers, but not for transfers involving KITTI. Fig. 4 illustrates four qualitative examples, showing improved robustness to scan-pattern artifacts on ground surfaces and crisper object boundaries.

TABLE I: Number of LIDAR semantic segmentation labels in the dataset training splits, including all pairwise combinations (table rows). Training samples (”scenes”) and number of annotated LIDAR points are distinguished (table columns).
<table><tr><td colspan="2"></td><td rowspan="2">Samples</td><td colspan="2">Semantic Accumulated</td></tr><tr><td>Dataset ↓</td><td></td><td>Labels</td><td>Labels</td></tr><tr><td>KITTI V</td><td>NuScenes Waymo</td><td></td><td>in Million Points</td><td></td></tr><tr><td rowspan="3">√</td><td></td><td>19,310</td><td>1,776</td><td>2,684</td></tr><tr><td>√</td><td>27,430 23,691</td><td>519 2,480</td><td>1,832</td></tr><tr><td></td><td>46,740</td><td></td><td>3,671</td></tr><tr><td>√ V</td><td>√ V</td><td>43,001</td><td>2,295 4,256</td><td>4,516 6,355</td></tr><tr><td></td><td>V V</td><td>51,121</td><td>2,998</td><td>5,503</td></tr><tr><td>√ √</td><td>√</td><td>70,431</td><td>4,775</td><td>8,188</td></tr></table>

b) Results on 10-A.: The 10-A results are similar to the above. Transfers involving KITTI benefit somewhat more than under 10-B, with accumulated-points supervision improving robustness over the segmentation baseline (marked in green). From KITTI to nuScenes, accumulated-points training can surpass C&L’s domain adaptation relative to the segmentation baseline. 3DLabelProp achieves higher domain generalization performance overall, indicating that altering the input point cloud remains a strong approach for cross-domain robustness.

## D. Ablations on Single-Source and Multi-Source Domain Generalization

We observe that source-domain performance is very similar with and without the occupancy (scene-completion) auxiliary task. The bottom block of Table II reports multi-source training. All three pairwise combinations (KITTI+nuScenes, KITTI+Waymo, nuScenes+Waymo, rows 4-6) and the allsources setting (KITTI+nuScenes+Waymo, row 7). Training on all three datasets simultaneously slightly reduces perdataset source accuracy but still exceeds the single-source domain generalization results reported above for both the segmentation baseline and the accumulated-points variant. Adding KITTI to the training mix benefits the segmentation baseline the most, nearly closing the gap to its best single-source domain generalization performance. However, accumulated-points training on KITTI+nuScenes and KITTI+Waymo continues to outperform the corresponding segmentation baselines. Relative to single-source training, gains are modest for KITTI+Waymo (row 5) and largely neutral for KITTI+nuScenes (row 4). For transfer to KITTI, combining nuScenes and Waymo (row 6) provides a clear boost over single-source training—interestingly, in this twosource combination the segmentation baseline attains the better score.

## E. Discussion

Our results indicate that enriching the supervision signal with accumulated-points (scene-completion) targets yields a domain-robustness benefit at almost no cost to source-domain accuracy. The observed source trade-off is negligible, indicating that the added objective does not overfit or dilute features critical for in-domain segmentation. This points less to a model-capacity limitation and more to supervision shaping, meaning that completion targets encourage geometry- and boundary-aware features that generalize well across sampling patterns. Importantly, our method does not alter the input point cloud at any stage. It isolates the effect of richer supervision from the effect of input transformations seen in prior DG/DA work, clarifying that more informative outputs from the same inputs can improve feature robustness.

The effectiveness nonetheless depends on the source–target pairing. Gains are pronounced for Waymo ↔ nuScenes, while transfers involving KITTI remain more difficult. We do not attribute this specifically to scan pattern or density as these factors likely differ sufficiently across all datasets. Instead, broader dataset-to-dataset domain gaps might be more influential. nuScenes and Waymo are captured primarily in dense urban environments, whereas KITTI spans a mix of less densely populated areas, highways, and urban scenes. In particular, the KITTI validation split offers limited diversity, which may skew cross-domain results. Future work should verify this with targeted experiments on additional datasets. Differences in scene composition and geographic context, object frequency and scale, and the balance between static and dynamic classes can all reduce the utility of features learned in one domain when applied to another.

Multi-source training further indicates that a simple data pooling is no solution to the domain generalization problem. Adding KITTI can narrow gaps for source-only baselines, and accumulated-points supervision generally improves performance but with modest gains. It might be necessary to address the domain factors in a more targeted way. Overall, requesting more outputs from the same network on unchanged scans is a simple, stable lever for domain generalization. It is effective when inter-domain semantics overlap sufficiently. Beyond that, richer geometric supervision on unchanged scans forces the network to learn mid-level, geometry-aware cues, which improves robustness to domain shift. This is aligned with prior image-domain results where adding geometric supervision forces models to learn more comprehensive representations and transferable high-level semantics. We observe the same effect for LIDAR.

## V. CONCLUSION

We approached LiDAR domain generalization by keeping the input scan unchanged while enriching supervision with a scene-completion task from accumulated points, and we evaluated this setup on combinations of KITTI, nuScenes, and Waymo. Our experiments show that switching to a richer supervision signal, while keeping the input scan unchanged, can considerably shift domain-generalization outcomes for LIDAR segmentation. To our knowledge, prior work has not isolated this effect without modifying the input itself. We observe strong dataset-pair dependence. Transfers involving

TABLE II: Domain generalization performance (10-B) of our proposed approach (left, center) and DGLSS [2] (right). Values are mIoU scores of the semantic segmentation task on the 10-B label set (higher is better). A gray background highlights sourcedomain evaluation, green background highlights improvements on the unseen target domain with accumulated points training over the segmentation baseline.
<table><tr><td></td><td></td><td></td><td colspan="3">Acc. Points Training</td><td colspan="3">Baseline Segmentation Evaluated on ↓</td><td colspan="3">DGLSS [2]</td></tr><tr><td colspan="3">Trained on↓</td><td colspan="6"></td><td colspan="3"></td></tr><tr><td>KITTI</td><td></td><td>nuScenes Waymo</td><td>KITTI</td><td>nuScenes</td><td>Waymo</td><td>KITTI</td><td>nuScenes</td><td>Waymo</td><td>KITTI</td><td>nuScenes</td><td>Waymo</td></tr><tr><td>√</td><td></td><td></td><td>58.3</td><td>34.5</td><td>36.0</td><td>61.1</td><td>36.4</td><td>36.3</td><td>59.6</td><td>44.8</td><td>40.7</td></tr><tr><td></td><td>√</td><td></td><td>47.2</td><td>70.9</td><td>52.5</td><td>48.7</td><td>69.7</td><td>45.0</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>V</td><td>47.5</td><td>50.2</td><td>73.7</td><td>49.2</td><td>43.4</td><td>74.7</td><td>51.2</td><td>49.6</td><td>75.3</td></tr><tr><td>√</td><td>√</td><td></td><td>58.2</td><td>68.2</td><td>52.0</td><td>62.9</td><td>68.8</td><td>50.6</td><td></td><td>1</td><td></td></tr><tr><td>V</td><td></td><td>V</td><td>60.7</td><td>54.9</td><td>73.0</td><td>61.6</td><td>52.7</td><td>72.2</td><td></td><td></td><td></td></tr><tr><td></td><td>V</td><td>√</td><td>51.7</td><td>66.4</td><td>71.1</td><td>52.0</td><td>68.4</td><td>72.5</td><td></td><td></td><td></td></tr><tr><td>√</td><td>√</td><td>√</td><td>57.9</td><td>61.7</td><td>64.4</td><td>63.6</td><td>65.4</td><td>68.9</td><td>一</td><td>1</td><td></td></tr></table>

TABLE III: Domain generalization performance (10-A) of our scene completion approach (left), our baseline semantic segmentation (middle) and previous work (right). Semantic segmentation on 10-A label set. Values are mIoU scores. Highe is better. A gray background highlights source-domain evaluation.
<table><tr><td colspan="2"></td><td colspan="2">Acc. Points Training</td><td colspan="2">Baseline Segmentation</td><td colspan="2">C&amp;L [5]</td><td colspan="2">3DLabelProp [9]</td></tr><tr><td colspan="2">Trained on ↓</td><td colspan="7">Evaluated on↓</td></tr><tr><td>KITTI</td><td>nuScenes</td><td>KITTI</td><td>nuScenes</td><td>KITTI</td><td>nuScenes</td><td>KITTI</td><td>nuScenes</td><td>KITTI</td><td>nuScenes</td></tr><tr><td rowspan="2">√</td><td></td><td>64.8</td><td>32.5</td><td>62.1</td><td>28.3</td><td>50.2</td><td>31.6</td><td>69.0</td><td>42.7</td></tr><tr><td>√</td><td>46.2</td><td>67.7</td><td>45.2</td><td>71.2</td><td>33.7</td><td>54.4</td><td>50.5</td><td>66.5</td></tr></table>

KITTI remain challenging, whereas Waymo-nuScenes generalize substantially better. Although our approach does not surpass the very latest domain generalization methods, it is competitive with a classic surface-based DA baseline and with a more recent DG approach. Our approach is complementary to previous work. Input-preserving supervision can be combined with input-altering strategies to aggregate their strengths.

Next, we propose to adapt our setup to UDA by deriving target-domain completion labels and using them only as auxiliary supervision during training. This adds supervision while keeping single-scan inference unchanged, testing whether target-side completion can close the remaining gap.

## REFERENCES

[1] L. Triess, D. Peter, C. Rist, M. Enzweiler, and J. M. Zollner, “CNN-¨ based synthesis of realistic high-resolution LiDAR data,” in IV, 2019.

[2] H. Kim, Y. Kang, C. Oh, and K.-J. Yoon, “Single Domain Generalization for LiDAR Semantic Segmentation,” in CVPR, 2023.

[3] F. Langer, A. Milioto, A. Haag, J. Behley, and C. Stachniss, “Domain Transfer for Semantic Segmentation of LiDAR Data using Deep Neural Networks,” in IROS, 2020.

[4] K. Ryu, S. Hwang, and J. Park, “Instant Domain Augmentation for LiDAR Semantic Segmentation,” in CVPR, 2023.

[5] L. Yi, B. Gong, and T. Funkhouser, “Complete & Label: A Domain Adaptation Approach to Semantic Segmentation of LiDAR Point Clouds,” in CVPR, 2021.

[6] A. Boulch, C. Sautier, B. Michele, G. Puy, and R. Marlet, “ALSO: Automotive Lidar Self-supervision by Occupancy estimation,” in CVPR, 2023.

[7] Z. Zhang, M. A. B. AWS, and L. A. E. L. AWS, “Implicit Surface Contrastive Clustering for LiDAR Point Clouds,” in CVPR, 2023.

[8] B. Michele, A. Boulch, G. Puy, T. Vu, R. Marlet, and N. Courty, “SALUDA: Surface-based Automotive Lidar Unsupervised Domain Adaptation,” in 3DV, 2024, spotlight paper; arXiv preprint arXiv:2304.03251.

[9] J. Sanchez, J.-E. Deschaud, and F. Goulette, “Domain Generalization of 3D Semantic Segmentation in Autonomous Driving,” in ICCV, October 2023.

[10] C. B. Rist, D. Emmerichs, M. Enzweiler, and D. M. Gavrila, “Semantic Scene Completion using Local Deep Implicit Functions on LiDAR Data,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 10, pp. 7205–7218, 2021.

[11] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss, and J. Gall, “SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences,” in ICCV, 2019.

[12] W. K. Fong, R. Mohan, J. V. Hurtado, L. Zhou, H. Caesar, O. Beijbom, and A. Valada, “Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking,” arXiv preprint arXiv:2109.03805, 2021.

[13] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, V. Vasudevan, W. Han, J. Ngiam, H. Zhao, A. Timofeev, S. Ettinger, M. Krivokon, A. Gao, A. Joshi, Y. Zhang, J. Shlens, Z. Chen, and D. Anguelov, “Scalability in Perception for Autonomous Driving: Waymo Open Dataset,” in CVPR, June 2020.

[14] H. Dong, M. Liu, K. Zhou, E. Chatzi, J. Kannala, C. Stachniss, and O. Fink, “Advances in Multimodal Adaptation and Generalization: From Traditional Approaches to Foundation Models,” arXiv preprint arXiv:2501.18592, 2025.

[15] L. T. Triess, M. Dreissig, C. B. Rist, and J. M. Zollner, “A Survey¨ on Deep Domain Adaptation for LiDAR Perception,” in IV Workshops, 2021.

[16] T.-H. Vu, H. Jain, M. Bucher, M. Cord, and P. Perez, “ADVENT:´ Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation,” in CVPR, 2019.

[17] M. Chen, H. Xue, and D. Cai, “Domain Adaptation for Semantic Segmentation With Maximum Squares Loss,” in ICCV, 2019.

![](images/88a493ab927c5130f2c1f98619e96141df72ef1b13d079340bf1053a8e7165a2.jpg)

![](images/528ad3d09427f82ee45fbcd18e57803dd0f114cf0396390542f2cf7993c52bc6.jpg)

![](images/baeab580aca42286314cf2ab2f4010edb4a2cda078d92607bf497bd83c5135fe.jpg)  
EVALUATION ON WAYMO DATASET

![](images/252df4e0d0501cfe88fdff5c794be4be264c14f02e96115342c72463123be7de.jpg)

![](images/53966d665b33c5eefafd5e64e22d68e045d9d1a2a645ae67844e071172361d80.jpg)

![](images/b61c5c8a016603af6fec2e2bb49bb767ba09f86f4377aeaa6cc3593af24a6a65.jpg)  
Ground Truth

![](images/e41193c22e84481a63f746749a525d7e147d1a96f671415642f4be4cd284f79a.jpg)  
(a) Acc. Points nuScenes

![](images/5c290f20ca5c541f70d4d7d8ae5eee522fea9ae55bc06869fb4e5effd798e878.jpg)  
(b) Sem Segm. nuScenes

![](images/d1e0601bc52db127aa8023a59b2531f1699dfd4a8b7cb26575dc9c8c95989a4c.jpg)  
(c) Acc. Points Waymo

![](images/7e51b730a83a30f701ea7cf4396b9a6995b2b8be5578e905aeb312e6cbb932e1.jpg)  
(d) Sem Segm. Waymo  
Fig. 4: Qualitative Results: Examples of cross-sensor domain generalization with two evaluation examples from the nuScenes and two from the Waymo dataset (one per row). Each row displays, from left to right: ground truth reference, segmentation result trained on (a) nuScenes accumulated points, (b) nuScenes segmentation, (c) Waymo accumulated points, (d) Waymo segmentation. For clarity, all unlabeled points (gray) are shown as unlabeled in the prediction columns. A gray border highlights source-domain evaluation; green background highlights improvements on the unseen target domain. Training on accumulated multi-scan points in the source domain yields a noticeably more robust network. Ground segmentation issues due to scan patterns are effectively reduced and object boundaries are more consistent. Importantly, no visible regression is observed on the source domain when using accumulated points compared to standard segmentation.

[18] P. Morerio, J. Cavazza, and V. Murino, “Minimal-Entropy Correlation Alignment for Unsupervised Deep Domain Adaptation,” in ICLR, 2018.

[19] Q. Wang, D. Dai, L. Hoyer, L. Van Gool, and O. Fink, “Domain Adaptive Semantic Segmentation With Self-Supervised Depth Estimation,” in ICCV, October 2021.

[20] V. Guizilini, J. Li, R. Ambrus, and A. Gaidon, “Geometric Unsupervised Domain Adaptation for Semantic Segmentation,” in ICCV, October 2021.

[21] S. Saha, A. Obukhov, D. P. Paudel, M. Kanakis, Y. Chen, S. Georgoulis, and L. Van Gool, “Learning To Relate Depth and Semantics for Unsupervised Domain Adaptation,” in CVPR, June 2021.

[22] Q. Wu and H. Liu, “Unsupervised Domain Adaptation for Semantic Segmentation using Depth Distribution,” in NeurIPS, 2022.

[23] C. Rist, M. Enzweiler, and D. Gavrila, “Cross-Sensor Deep Domain Adaptation for LiDAR Detection and Segmentation,” in IV, 2019.

[24] X. Yan and et al., “Spot: Scalable 3d pre-training via occupancy prediction for learning transferable representations,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 11, pp. 9609–9625, 2025.

[25] L. T. Triess, C. B. Rist, D. Peter, and J. M. Zollner, “A Realism Metric¨ for Generated LiDAR Point Clouds,” Int. J. Comput. Vision, vol. 130, no. 12, p. 2962–2979, Dec. 2022.

[26] B. Wu, X. Zhou, S. Zhao, X. Yue, and K. Keutzer, “SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud,” in ICRA, 2019.

[27] C. Saltori, F. Galasso, G. Fiameni, N. Sebe, E. Ricci, and F. Poiesi, “CoSMix: Compositional Semantic Mix for Domain Adaptation in 3D LiDAR Segmentation,” in ECCV, 2022.

[28] J. Kim, J. Woo, J. Kim, and S. Im, “Rethinking LiDAR Domain Generalization: Single Source as Multiple Density Domains,” in ECCV, 2024.

[29] K. Ryu, S. Hwang, and J. Park, “Instant Domain Augmentation for LiDAR Semantic Segmentation,” in CVPR, 2023.

[30] Y. Zhao, N. Zhao, and G. H. Lee, “Synthetic-to-Real Domain Generalized Semantic Segmentation for 3D Indoor Point Clouds,” in BMVC, 2024.

[31] F. M. Carlucci, A. D’Innocente, S. Bucci, B. Caputo, and T. Tommasi, “Domain Generalization by Solving Jigsaw Puzzles,” in CVPR, 2019.

[32] Zhenkai Qin and Qining Luo and Xunyi Nong and Xiaolong Chen and Hongfeng Zhang and Wong, Cora Un In, “Self-Supervised Learning for Domain Generalization With a Multi-Classifier Ensemble Approach,” IET Image Processing, vol. 19, no. 1, 2025.

[33] Z. Ren and Y. J. Lee, “Cross-Domain Self-Supervised Multi-Task Feature Learning Using Synthetic Imagery,” in CVPR, June 2018.

[34] T.-H. Vu, H. Jain, M. Bucher, M. Cord, and P. Perez, “DADA: Depth-Aware Domain Adaptation in Semantic Segmentation,” in ICCV, October 2019.

[35] A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang, and O. Beijbom, “PointPillars: Fast Encoders for Object Detection From Point Clouds,” in CVPR, 2019.

[36] C. Choy, J. Gwak, and S. Savarese, “4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks,” in CVPR, 2019.