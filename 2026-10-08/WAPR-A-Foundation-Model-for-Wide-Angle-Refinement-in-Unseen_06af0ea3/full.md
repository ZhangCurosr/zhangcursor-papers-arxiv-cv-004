# WAPR: A Foundation Model for Wide-Angle Refinement in Unseen Object Pose Estimation

Yulin Wang<sup>\*1</sup> , Mengting Hu<sup>\*1</sup> , Hongli Li<sup>2</sup> , Jianghao Zhou<sup>1</sup> , and Chen LUO<sup>†1</sup>

<sup>1</sup> Southeast University, Nanjing, China

yulinwang@seu.edu.cn, 220240361@seu.edu.cn,

zhou\_jianghao@seu.edu.cn, chenluo@seu.edu.cn

<sup>2</sup> Purdue University, West Lafayette, IN, USA

li5125@purdue.edu

<sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding author.

Abstract. Real-world applications require 6D pose estimation to be accurate, fast, and scalable to unseen objects. This paper introduces WAPR, a zero-shot wide-angle pose refinement model that refines candidate poses with rotational deviations up to 90^\circ . With as few as 12 candidate poses per detected object instance, WAPR supports fast inference within 1 s per frame and reaches a pose-estimation throughput of up to 25 detected object instances per second. To support wide-angle training for rotationally symmetric objects, WAPR uses rotational symmetry priors to canonicalize symmetry-equivalent pose targets before loss computation. We further construct SA6D, a large-scale 6D training dataset with such priors. SA6D obtains KASAL-assisted rotational symmetry priors for 944 GSO scans and expands them through geometry and texture augmentation into \sim 50K augmented object instances and \sim 2M rendered RGB-D images. In addition, an angle-balanced loss stabilizes learning across diferent angular ranges by reducing the influence of uninformative large-error cases. Experiments on seven BOP core datasets show that WAPR achieves state-of-the-art performance in unseen-object 6D pose localization and detection under both fast and unconstrained inference settings. Project page: https://github.com/WangYuLin-SEU/WAPR.

Keywords: Unseen Object Pose Estimation · 6D Pose Refinement · Wide-Angle Pose Refinement · Rotational Symmetry Priors · Synthetic RGB-D Dataset · BOP Challenge

## 1 Introduction

6D object pose estimation [18–20, 40] is a fundamental problem in computer vision, with wide-ranging applications in augmented reality [29, 43, 58], robotics [33,49,59,61], autonomous driving [22,55,62], and industrial automation [23,45]. Traditional pipelines typically combine 2D detection or segmentation [11,14,46, 50] with 6D pose estimation [48,51,52], followed by optional refinement [30,60].

![](images/177b4deb2e6e5f03ed79fe73c6075b2c041cec6a1fdf38bbb528e4aede405a8a.jpg)  
Fig. 1: Visualization of multi-iteration refinement for candidate poses with large angular deviations. The example is from TUD-L [18], where the handheld object moves continuously. With large deviations between the candidate pose and the ground-truth pose, conventional small-angle refiners often fail, whereas WAPR corrects the pose through multi-iteration refinement.

While efective for trained objects, their reliance on object-specific training limits scalability in real-world deployments.

To improve scalability, unseen-object pose estimation [3, 37, 54] estimates poses for novel objects from object models with minimal or no retraining. Recent methods [3,36,37,39,63] commonly follow a candidate-based pipeline, where coarse pose hypotheses are generated from rendered views or feature correspondences and then refined or scored. This design makes candidate coverage and refinement range closely coupled. Using many candidates improves pose coverage but increases runtime, while sparse candidates are eficient but may leave larger angular deviations to the refiner. As shown in Fig. 1, such deviations can arise from coarse hypotheses, occlusion, and large viewpoint changes, motivating a refinement model that remains reliable across a wider angular range.

This paper introduces WAPR, a zero-shot foundation model for wide-angle pose refinement. WAPR follows the render-and-compare paradigm by comparing a rendered RGB-D observation and mask under a candidate pose with the real RGB-D input, and directly regresses rotation and translation updates. During training, ground-truth poses are perturbed by up to 90^\circ , allowing WAPR to learn pose updates over a wide angular range. At inference time, this wide-angle refinement capability enables accurate pose estimation with a small number of candidates, including only 12 candidates in the fast setting.

Wide-angle training also makes pose ambiguity more prominent for rotationally symmetric objects. Diferent symmetry-equivalent poses may produce indistinguishable or nearly indistinguishable observations, and assigning arbitrary raw rotation targets can make visually similar candidate–target pairs receive inconsistent supervision. Following the common practice in 6D pose estimation of treating symmetry-equivalent poses consistently [26, 48, 52, 57], WAPR uses rotational symmetry priors to canonicalize symmetry-equivalent pose targets before loss computation. This makes wide-angle candidate–target pairs well-defined during training.

To support this training at scale, this paper constructs SA6D, a large-scale 6D dataset with rotational symmetry priors. Large-scale synthetic datasets for unseen-object pose estimation [27,54] provide diverse object models and rendered RGB-D supervision. SA6D complements this direction by adding dataset-level rotational symmetry priors. It uses KASAL [53] to obtain reliable symmetry priors for 944 GSO [8] scans, and expands these annotated assets through geometry and texture augmentation into \sim 50K augmented object instances and 2M synthetic RGB-D images. The training objective also includes an angle-balanced loss, which reduces the dominance of low-correspondence large-error samples during wide-angle optimization.

Experiments on seven core BOP datasets [1,7,9,17,18,24,57] evaluate WAPR for zero-shot unseen-object 6D pose localization and detection. In the fast setting (<1 s per frame) with MUSE detections, WAPR reaches 80.6% AR / 78.6% AP, improving over Co-op [37] by +4.7% AR / +9.3% AP. With Co-op’s F3DT2D detector under the same fast-inference protocol, WAPR obtains 78.1% AR / 76.5% AP, giving a matched-detector gain of +2.2% AR / +7.2% AP. In the unconstrained multi-detector setting, WAPR reaches 84.5% AR / 85.3% AP, improving over FreeZeV2 [3, 4] by +1.2% AR / +2.0% AP.

The main contributions are as follows.

– This paper presents WAPR, a zero-shot foundation model for wide-angle pose refinement. With render-and-compare RGB-D inputs and iterative refinement, WAPR handles rotational deviations up to 90^\circ and enables accurate pose estimation with sparse candidate poses.

This paper constructs SA6D, a large-scale 6D dataset with rotational symmetry priors. SA6D uses a cost-eficient KASAL-assisted strategy to expand 944 annotated GSO scans into \sim 50K augmented object instances and 2M synthetic RGB-D images.

– This paper develops wide-angle training and sparse candidate-based inference for unseen-object pose estimation. Rotational symmetry priors canonicalize ambiguous training pairs, angle-balanced loss stabilizes wide-angle learning, and score-based selection produces accurate final poses from a small candidate set.

## 2 Related Works

This section reviews 6D pose estimation for seen objects [12, 13, 21, 34, 48, 56], unseen-object 6D pose estimation [3, 27, 32, 36, 37, 54], and symmetry handling in 6D pose learning and evaluation.

## 2.1 6D Pose Estimation for Seen Objects

Seen-object 6D pose estimation assumes that training data are available for each target object. Early methods directly regress object rotation and translation from RGB or RGB-D observations [2, 57]. Later methods improve robustness by predicting keypoints or dense surface coordinates and recovering the pose with PnP [16,28,31,42,44,52]. This correspondence-based formulation provides explicit 2D–3D constraints and is efective under partial occlusion.

Pose refinement further improves accuracy by aligning rendered observations with real images. Representative methods such as DeepIM [30] and DPOD [60] refine an initial pose by comparing rendered and observed images. Recent methods such as GDR-Net [51] and ZebraPose [48] combine detection, dense correspondence prediction, PnP, and optional refinement to achieve strong performance on trained objects. These object-specific pipelines provide the basis for many recent unseen-object methods, while scalable deployment to newly introduced objects motivates methods that reduce object-specific training.

## 2.2 6D Pose Estimation for Unseen Objects

Unseen-object 6D pose estimation estimates poses for novel objects from object models with minimal or no object-specific retraining. MegaPose [27] trains largescale render-and-compare models on synthetic data and demonstrates strong generalization to unseen objects. FoundationPose [54] combines pose initialization, iterative refinement, and pose scoring in a unified framework. SAM-6D [32] and CNOS [38] use large visual models for zero-shot object localization and segmentation. Recent methods including FreeZe [3], GenFlow [36], FoundPose [63], GigaPose [39], and Co-op [37] improve diferent stages of the pipeline, including object detection, coarse pose retrieval, feature matching, refinement, and pose selection.

Most unseen-object pipelines generate candidate poses from rendered templates or correspondence matching, refine these candidates, and then select the final pose. This design makes candidate coverage and inference eficiency closely connected. Dense candidate sampling improves pose coverage but increases runtime, while sparse candidates are more eficient but place higher demand on the refinement stage. Many existing refinement modules are most efective when the initial hypothesis is already close to the target pose. WAPR targets the complementary wide-angle regime, where sparse candidates leave larger residual deviations for the refiner.

## 2.3 Symmetry Handling in 6D Pose Learning and Evaluation

Object symmetry is a long-standing issue in 6D pose estimation because multiple poses may correspond to the same or nearly the same visual observation. Benchmarks and evaluation protocols therefore use symmetry-aware pose metrics to avoid penalizing equivalent poses [19]. Learning-based methods also incorporate symmetry into supervision. PoseCNN [57] introduces the ShapeMatch loss to make pose learning invariant to indistinguishable symmetric poses, while CosyPose [26] uses symmetry-aware pose objectives for object-specific pose estimation. Coordinate-based methods such as ZebraPose [48] and HccePose [52] also rely on symmetry-consistent definitions for learning or evaluation.

![](images/813335ec4ff8ff3b905d2000fba6fd8289336d77cec963f04207540ae460648b.jpg)  
Fig. 2: Overview of the proposed framework. Top: SA6D constructs large-scale RGB-D training data with KASAL-assisted rotational symmetry priors. These priors are used to canonicalize symmetry-equivalent training pairs. Bottom: WAPR refines sparse candidate poses by comparing real and rendered RGB-D observations with masks. The refined candidates are selected by a score-based pose selection module, enabling accurate 6D pose estimation under large initial misalignments of up to 90^\circ .

These works establish symmetry-consistent supervision as a standard principle in 6D pose estimation. Beyond loss design and evaluation, rotational symmetry priors are also useful as dataset-level annotations when synthetic pose pairs are generated for training. For wide-angle refinement, candidate and target poses may span symmetry-equivalent pose regions, and such pairs should receive consistent pose-update targets. Large-scale synthetic datasets have mainly focused on object diversity, rendering realism, and scalable pose supervision [27, 54]. SA6D complements this direction by adding KASAL-assisted rotational symmetry priors at scale and using them to canonicalize synthetic candidate–target pose pairs before training.

## 3 Methodology

Figure 2 summarizes the proposed framework. It consists of two connected parts. First, SA6D provides large-scale synthetic RGB-D training data with rotational symmetry priors. These priors are used to canonicalize symmetry-equivalent candidate–target pose pairs before training, so that wide-angle pose updates are well-defined for rotationally symmetric objects. Second, WAPR learns a renderand-compare refinement model that predicts rotation and translation updates from paired real/rendered RGB-D observations and masks. An angle-balanced loss further stabilizes training over a wide range of pose deviations, and sparse candidate generation enables eficient inference with only 12 or 24 candidates.

![](images/15b4e9f1af0fe8b956a7dcfd6272ac3efc19113bf69ac37a21e062886899c8ac.jpg)  
Fig. 3: Large-scale construction of SA6D with rotational symmetry priors. KASALassisted annotation provides texture-aware and geometry-only priors for the original GSO scans. Geometry scaling and texture randomization reuse these priors to generate \sim 20K non-symmetric and \sim 30K rotationally symmetric augmented instances, and BlenderProc [6] renders \sim 2M synthetic RGB-D images for training.

## 3.1 Symmetry-Aware 6D Dataset (SA6D)

SA6D is designed to train WAPR with synthetic RGB-D pairs whose pose targets remain consistent under rotational symmetry. It starts from 944 GSO scans [8], obtains KASAL-assisted rotational symmetry priors for these scans, and expands them through geometry and texture augmentation into \sim 50K augmented object instances, including \sim 30K rotationally symmetric instances. BlenderProc [6] is then used to render \sim 2M synthetic RGB-D images.

## Object Collection and Symmetry Augmentation

The construction of SA6D starts from the original GSO scans. For each scan, human annotators specify the rotational symmetry type and order, and KASAL [53] localizes the corresponding symmetry-axis direction and center. The annotated scans are then expanded into augmented instances by geometry scaling and texture randomization, while reusing the corresponding symmetry priors. This avoids repeatedly annotating each augmented instance.

SA6D uses two types of texture augmentation. The first modifies the texture while preserving the original texture-aware rotational symmetry prior, as illustrated by the disc in Fig. 3. The second replaces the texture with a uniform color, which removes texture-induced asymmetries and makes the augmented instance follow the geometry-only rotational symmetry prior, as shown by the carton in Fig. 3. Thus, each original scan is associated with both a texture-aware prior and a geometry-only prior. These priors are used in the ofline canonicalization procedure described below to generate ambiguity-free candidate–target pose pairs for WAPR training.

![](images/2d4ba486e47aa29368a604b3990210dc99120fea1bacde32bd570a99546457c6.jpg)  
Fig. 4: Rotational symmetry ambiguity in wide-angle training. The candidate and target views can have a large raw rotation gap, while their visible diference may be limited to subtle local texture details, as shown by the zoomed regions. With rotational symmetry priors, such pairs are treated as symmetry-equivalent and canonicalized to the same supervision before loss computation.

## Symmetry Disambiguation

To train WAPR, candidate poses are simulated by applying random rotations and translations to ground-truth poses in SA6D. For rotationally symmetric objects, multiple symmetry-equivalent poses may be visually valid for the same training pair. Directly using the raw perturbed rotation can therefore assign inconsistent pose-update targets to visually similar candidate–target pairs, as illustrated in Fig. 4. SA6D resolves this ambiguity by applying an ofline minover-symmetry canonicalization before loss computation. The procedure selects a canonical representative from the symmetry-equivalent poses, making the pose update used for training unique and prediction-independent.

Given a ground-truth pose, a set of augmented poses $\mathcal { P } _ { \mathrm { a u g } }$ is generated by applying random perturbations to both rotation and translation:

$$
\mathcal { P } _ { \mathrm { a u g } } = \{ ( R _ { j } , t _ { j } ) \ | \ j = 1 , 2 , \ldots , M \}\tag{1}
$$

Each object is aligned so that its rotational symmetry center coincides with the origin, allowing symmetry transformations to be represented as pure rotations. For an object with n-fold discrete rotational symmetry, the symmetry group is represented by n rotation matrices. For continuous rotational symmetry, the symmetry orbit is discretized into 360 uniformly spaced rotations, which bounds the angular quantization error by 0.5^\circ . The symmetry-consistent rotation set is

$$
\mathcal { R } _ { \mathrm { s y m } } = \{ R _ { k } \in \mathrm { S O ( 3 ) } \mid k = 1 , 2 , \ldots , K \}\tag{2}
$$

where each $R _ { k }$ represents a valid rotational symmetry of the object.

For each augmented pose $( R _ { j } , t _ { j } )$ , symmetry-extended variants are constructed as:

$$
\mathcal { P } _ { \mathrm { s y m } } ^ { ( j ) } = \{ ( R _ { j } R _ { k } , t _ { j } ) \ | \ R _ { k } \in \mathcal { R } _ { \mathrm { s y m } } \}\tag{3}
$$

A canonical representative is selected by choosing the variant with the smallest rotation from the identity:

$$
( R _ { \mathrm { j } } ^ { * } , t _ { \mathrm { j } } ^ { * } ) = \arg \operatorname* { m i n } _ { ( R , t ) \in \mathcal { P } _ { \mathrm { s y m } } ^ { ( j ) } } \| R - I \| _ { 2 }\tag{4}
$$

The selected pose $( R _ { j } ^ { * } , t _ { j } ^ { * } )$ is used as the final augmented candidate pose for training. The target pose is canonicalized under the same symmetry prior, yielding a consistent candidate–target pair with a unique rotation update for training. This canonicalization is performed when constructing SA6D training pairs and does not depend on the network prediction.

## Segmentation Mask Generation

WAPR also needs mask inputs during training. Instead of using perfect synthetic masks, which are cleaner than inference-time masks, SA6D generates training masks with a large-scale segmentation model such as SAM [25]. For each training sample, SAM produces candidate masks $\mathbfcal { S } = \{ S _ { i } \} _ { i = 1 } ^ { N }$ . Masks whose intersectionover-union (IoU) with the ground-truth mask G exceeds 0.5 are kept:

$$
\mathcal { Q } = \{ i \mid \mathrm { I o U } ( S _ { i } , G ) > 0 . 5 \}\tag{5}
$$

If \protec \mathcal {Q} is non-empty, one mask is randomly sampled for training; otherwise, a blank mask is used. Together, symmetry-prior canonicalization and SAMgenerated masks provide WAPR with well-defined pose targets and realistic mask inputs for wide-angle refinement training.

## 3.2 Wide-angle Pose Refinement

## Neural Network

Given a candidate pose and its canonicalized target pose from SA6D, WAPR learns to predict the pose update by comparing two RGB-D observations. One branch takes the observed RGB-D crop and mask, which corresponds to a PBRrendered sample during training and a real sensor observation during inference. The other branch takes the rendered RGB-D image and mask generated under the candidate pose. Two shared-weight ResNet layers based on BasicBlock-style residual units extract paired feature maps from the two branches.

The paired feature maps are concatenated and augmented with positional encodings. A two-layer Transformer encoder models global interactions in the fused representation, as illustrated in Fig. 2. Two independent one-layer Transformer encoders then produce separate representations for rotation and translation. Each branch applies attention-based pooling and a linear layer to regress the corresponding pose update.

Rotation updates are parameterized as 3D rotation vectors and converted to rotation matrices in \protect \mathrm {SO}(3) before being applied to candidate poses. For a candidate pose $( R _ { i } , t _ { i } )$ , the network predicts a rotation vector $\varDelta \tilde { \mathbf { r } } _ { i } \in \mathbb { R } ^ { 3 }$ and a translation ofset $\varDelta \tilde { t } _ { i } \in \mathbb { R } ^ { 3 }$

WAPR is trained and applied iteratively. During training, candidate poses are generated by perturbing the ground-truth poses with rotational deviations up to 90^\circ . After each iteration, the predicted rotation and translation updates are applied to the current candidate pose, and the updated pose is used to rerender the input for the next iteration. The rotation and translation are updated by

$$
\tilde { R } _ { i } = \operatorname { R o t } ( \varDelta \tilde { \mathbf { r } } _ { i } ) R _ { i }\tag{6}
$$

$$
\tilde { t } _ { i } = \varDelta \tilde { t } _ { i } + t _ { i }\tag{7}
$$

where Rot(·) maps a 3D rotation vector to a rotation matrix in \protect \mathrm {SO}(3) .

The $9 0 ^ { \circ }$ perturbation range is used as a practical wide-angle training range for render-and-compare refinement. As analyzed in the supplementary material, the Visual Correspondence Ratio (VCR) between the candidate and target views decreases with angular deviation, and very low-VCR pairs provide limited shared visual evidence for learning reliable updates. Perturbations far beyond 90^\circ , especially near 180^\circ , also make training convergence less stable. Therefore, 90^\circ provides a wide yet learnable range for WAPR training.

## Angle-Balanced Loss

Wide-angle training contains both easy local refinements and hard large-error pairs. When the rendered and observed views have limited overlap, high-error samples can dominate the gradients while providing little useful correspondence. WAPR therefore uses an angle-balanced loss to emphasize learnable wide-angle cases and down-weight uninformative large-error samples. The prediction error of each sample is measured in rotation-vector space as the L1 residual magnitude between the predicted and target rotation-update vectors:

$$
\theta _ { i } = \| \varDelta \tilde { \mathbf { r } } _ { i } - \varDelta \mathbf { r } _ { i } \| _ { 1 }\tag{8}
$$

Based on $\theta _ { i }$ , a dynamic weight $w _ { i }$ is assigned to each training sample:

$$
w _ { i } = \exp \left( \sigma \cdot \operatorname* { m i n } \left\{ \theta _ { i } , \tau - \theta _ { i } \right\} \right)\tag{9}
$$

where \tau is empirically set to 0.25\pi . Samples with rotation-vector residuals below \tau are treated as learnable wide-angle cases and receive larger weights, while samples with little overlap often yield $\theta _ { i } > \tau$ and are down-weighted. The coeficient \sigma controls the suppression strength and is set to 3.

The final angle-balanced loss is then formulated as:

$$
L = \alpha \cdot \frac { \boldsymbol { B } \cdot \sum _ { i = 1 } ^ { B } \boldsymbol { w } _ { i } \cdot \left| \Delta \tilde { \mathbf { r } } _ { i } - \Delta \mathbf { r } _ { i } \right| } { \displaystyle \sum _ { i = 1 } ^ { B } w _ { i } } + \beta \cdot \sum _ { i = 1 } ^ { B } \left| \Delta \tilde { \mathbf { t } } _ { i } - \Delta \mathbf { t } _ { i } \right|\tag{10}
$$

where B denotes the number of samples in the mini-batch, \alpha and $\beta$ are the balancing coeficients for the rotational and translational losses, respectively, and \left | \cdot \right | denotes the L1 loss. We set \alpha =3 and $\beta = 1$

## Inference Pipeline

WAPR’s wide-angle refinement ability allows the inference pipeline to use a sparse set of candidate poses. Given a 2D bounding box and mask, candidate poses are generated by combining a small set of viewing directions with evenly spaced in-plane rotations. In the fast setting, 12 candidates are formed by pairing the 4 viewing directions defined by the vertices of a regular tetrahedron with 3 in-plane rotations. In the unconstrained setting, 24 candidates are formed by pairing the 8 viewing directions defined by the vertices of a regular cube with the same 3 in-plane rotations.

For each candidate, WAPR pairs the cropped RGB-D image and mask with rendered RGB-D and mask images generated under the candidate pose. In the unconstrained setting, all 24 candidates are refined for five rounds with rerendered inputs at each round. In the fast setting, WAPR uses the two-stage schedule described in the supplementary material: all 12 candidates are first refined for two rounds and scored, after which only the top-4 candidates are refined for three additional rounds. Among the refined candidates, the pose with the highest confidence score predicted by the ScoreNet module of Foundation-Pose [54] is selected as the final result.

## 4 Experiments

This section evaluates WAPR through ablation studies and comparisons with state-of-the-art methods [3, 4, 37]. We first describe the experimental setup in Sec. 4.1, then analyze the main design choices in Sec. 4.2, and finally report BOP localization and detection results in Sec. 4.3.

## 4.1 Experimental setup

## Network Settings

Both input branches of WAPR use a fixed resolution of 160\times 160 . Each branch extracts features with a shared-weight encoder composed of a 7\times 7 convolution, a 3\times 3 convolution, and two ResNet basic blocks with 128 channels. The paired features are concatenated and processed by a fusion stage with 256- and 512- channel ResNet blocks, yielding a 512-dimensional fused feature representation. The Transformer encoder and attention-based pooling module both use four attention heads and a feed-forward dimension of 512. WAPR is trained for 800K iterations using Adam with a learning rate of 0.0002 and a batch size of 256. Each iteration samples RGB-D pairs from a single object category. Candidate poses are generated with translation ofsets up to 50% of the object diameter and rotational deviations up to 90^\circ , followed by five-step iterative refinement.

## Datasets & Metrics

The experiments follow the BOP 2024 single-view model-based unseen-object protocol, where comparable baselines and default 2D detections are available. We evaluate on seven BOP-Classic-Core datasets: LM-O [1], T-LESS [17], ITODD [9], HB [24], YCB-V [57], IC-BIN [7], and TUD-L [18]. Following the BOP protocol [19, 20], we use the average of VSD, MSSD, and MSPD as the BOP score. We report average recall (AR) for 6D localization and average precision (AP) for 6D detection.

Table 1: Analysis of candidate-pose coverage on IC-BIN. IC-BIN contains multiple identical objects in cluttered bin-like scenes, making it sensitive to the coverage of initial pose hypotheses. It is therefore used as a representative stress test for evaluating the number of candidate poses. “Num” denotes the number of candidate poses. Results are reported in AP for MSSD and MSPD, together with their mean.
<table><tr><td>Num</td><td>4</td><td>8</td><td>12</td><td>24</td><td>40</td><td>60</td></tr><tr><td>MSSD MSPD</td><td>59.6 54.2</td><td>67.3 63.1</td><td>71.4 68.9</td><td>71.9 69.8</td><td>72.8 70.4</td><td>73.3 70.9</td></tr><tr><td>Mean</td><td>56.9</td><td>65.2</td><td>70.1</td><td>70.9</td><td>71.6</td><td>72.1</td></tr></table>

## Fast inference

For fast inference, WAPR uses the 12-candidate tetrahedron setting described in Sec. 3.2. For each MUSE detection, the associated SAM-based mask is used to estimate the initial translation by a depth-weighted average within the mask and the depth map. Detections are filtered using MUSE-predicted object labels and confidence scores with a threshold of 0.35. On an RTX 5090 system, this setting estimates poses for up to 25 detected object instances per second using the two-stage refinement and scoring schedule described in the supplementary material.

## Unconstrained inference

When inference time is not constrained, WAPR uses the 24-candidate cube setting described in Sec. 3.2. When multiple 2D detectors are used, their detections are merged and NMS is applied to remove redundant boxes. For each retained detection, all category–detection–candidate combinations are evaluated without applying the confidence threshold used in the fast setting.

## 4.2 Ablation Study on Pose Refinement

This section analyzes the main factors afecting wide-angle pose refinement, including candidate-pose coverage, training inputs, symmetry priors, SA6D, anglebalanced optimization, network architecture, and plug-in refinement. The ablations in Table 2 are conducted on three representative datasets for controlled component analysis, while the final all-seven-dataset evaluation is reported in Table 4.

## Number of Candidate Poses

Table 1 studies how candidate-pose coverage afects WAPR on IC-BIN. Increasing the number of candidates improves the chance that at least one hypothesis falls within the efective refinement range. The gain is large when increasing from 4 to 12 candidates, where the mean AP improves from 56.9% to 70.1%. Further increasing the number to 24, 40, or 60 provides smaller gains, indicating that 12 candidates already provide a strong accuracy–eficiency trade-of for the fast setting, while 24 candidates are used in the unconstrained setting for higher coverage.

Table 2: Ablations of WAPR components on BOP detection AP. Results are reported as AP for MSSD and MSPD on three representative datasets, together with their mean. IC-BIN, TUD-L, and LM-O are selected to cover crowded multi-instance scenes, large object motion and viewpoint variation, and heavy occlusion, respectively. “w/o” denotes “without.” “WAPR (\pm 90^\circ )” is trained with candidate poses obtained by randomly perturbing the ground-truth pose within \pm 90^\circ .
<table><tr><td>Row</td><td>Method</td><td>IC-BIN [7] MSSD MSPD MSSD MSPD MSSD MSPD</td><td></td><td>TUD-L [18]</td><td></td><td>LM-O [1]</td><td></td><td>Mean</td></tr><tr><td>A0</td><td>|WAPR(±90°)</td><td>71.9</td><td>69.8</td><td>96.0</td><td>95.4</td><td>77.0</td><td>80.7</td><td>81.8</td></tr><tr><td>B0</td><td>|WAPR(±20°)</td><td>61.5</td><td>58.0</td><td>77.4</td><td>76.5</td><td>68.0</td><td>71.0</td><td>68.7</td></tr><tr><td>B1</td><td>WAPR(±40°)</td><td>69.9</td><td>67.1</td><td>95.0</td><td>94.7</td><td>74.4</td><td>77.8</td><td>79.8</td></tr><tr><td>B2</td><td>WAPR(±60°)</td><td>71.7</td><td>68.9</td><td>95.6</td><td>95.1</td><td>74.9</td><td>78.3</td><td>80.8</td></tr><tr><td>B3</td><td>WAPR(±140°)</td><td>70.6</td><td>68.1</td><td>95.5</td><td>95.0</td><td>74.4</td><td>77.7</td><td>80.2</td></tr><tr><td>B4</td><td>WAPR(±180°)</td><td>63.3</td><td>60.2</td><td>89.6</td><td>89.2</td><td>72.9</td><td>76.2</td><td>75.2</td></tr><tr><td>C0</td><td>|A0 + w/o mask input</td><td>71.0</td><td>68.5</td><td>95.2</td><td>94.9</td><td>75.6</td><td>78.9</td><td>80.7</td></tr><tr><td>C1</td><td>A0 + train with GT mask</td><td>61.8</td><td>59.4</td><td>94.3</td><td>93.9</td><td>73.9</td><td>77.2</td><td>76.8</td></tr><tr><td>D0</td><td>[A0 + w/o sym prior</td><td>70.8</td><td>68.4</td><td>93.6</td><td>92.6</td><td>75.8</td><td>79.4</td><td>80.1</td></tr><tr><td>D1</td><td>A0 + w/o angle-balanced Loss</td><td>69.8</td><td>67.3</td><td>91.9</td><td>91.1</td><td>72.8</td><td>76.5</td><td>78.2</td></tr><tr><td>D2</td><td>D1 + w/o sym prior</td><td>61.8</td><td>58.0</td><td>91.0</td><td>91.0</td><td>69.3</td><td>73.0</td><td>74.0</td></tr><tr><td>D3</td><td>A0 on GSO PBR (w/o SA6D)</td><td>64.8</td><td>62.2</td><td>92.0</td><td>91.1</td><td>71.5</td><td>74.7</td><td>76.1</td></tr><tr><td>D4</td><td>D0 on GSO PBR (w/o SA6D)</td><td>63.9</td><td>61.1</td><td>90.6</td><td>89.7</td><td>69.3</td><td>72.6</td><td>74.5</td></tr><tr><td>E0</td><td>A0 + w avg pool</td><td>71.5</td><td>68.8</td><td>95.5</td><td>95.3</td><td>75.5</td><td>78.8</td><td>80.9</td></tr><tr><td>E1</td><td>E0 + w/o transformer</td><td>70.9</td><td>68.5</td><td>95.3</td><td>94.9</td><td>74.3</td><td>77.7</td><td>80.3</td></tr></table>

<table><tr><td>Row</td><td>Method</td><td>IC-BIN MSSD MSPD MSSD MSPD MSSD MSPD</td><td></td><td>TUD-L</td><td></td><td>LM-O</td><td>Mean</td><td>Time/s</td></tr><tr><td>F0 F1</td><td>|FreeZeV2 [3, 4] F0 + C0</td><td>73.4 74.9</td><td>70.6 72.4</td><td>98.6 98.7</td><td>99.1 98.9</td><td>78.5 81.9 78.6 82.1</td><td>83.7 84.3</td><td>27.8 27.8 (+0.54)</td></tr><tr><td>F2 F3</td><td>[GigaPose [39]+GenFlow [36] F2+C0</td><td>42.0 43.4</td><td>38.4 40.0</td><td>71.4 72.7</td><td>71.7 56.1 73.0 57.3</td><td>59.5| 60.2</td><td>56.5 57.8</td><td>4.5 4.5 (+0.37)</td></tr></table>

## Angle Range for Training

Rows B0–B4 in Table 2 evaluate the rotation range used to generate candidate poses during training. Small perturbation ranges such as \pm 20^\circ provide insufficient coverage for wide-angle errors, while overly large ranges such as \pm 180^\circ introduce many low-correspondence samples and make optimization less stable. The \pm 90^\circ setting achieves the best overall performance, consistent with the VCR and training-convergence analysis discussed in Sec. 3.2.

Mask Input Rows C0 and C1 evaluate the role of mask inputs. Removing the mask slightly degrades performance, showing that masks provide useful objectregion guidance during render-and-compare refinement. Training with perfect ground-truth masks leads to a larger drop, since such masks difer from the predicted masks used at inference time. SAM-generated masks therefore provide a more realistic training signal for WAPR.

Rotational Symmetry Priors and SA6D Dataset Rows D0–D4 evaluate symmetry priors and SA6D through controlled comparisons. Rows D1 and D2 isolate the efect of rotational symmetry priors without angle-balanced loss, where adding the priors improves AP by 4.2%. Rows A0 and D0 show the efect when angle-balanced loss is used, where the priors improve AP by 1.7%. Rows D3 and D4 train the corresponding configurations on original GSO PBR data without SA6D augmentation. Compared with these GSO-PBR counterparts, A0 and D0 improve AP by 5.7% and 5.6%, respectively, showing the benefit of SA6D training data under matched model configurations.

Angle-Balanced Loss Function Rows D0–D2 analyze angle-balanced optimization. Compared with D2, which uses neither angle-balanced loss nor symmetry priors, D0 improves AP by 6.1%, showing that angle-balanced loss stabilizes wide-angle training. Combining angle-balanced loss with symmetry priors further improves performance, with A0 outperforming D2 by 7.8%.

Network Architecture Rows E0 and E1 ablate the feature aggregation modules. Replacing attention-based pooling with average pooling reduces AP by 0.9%, and further removing the Transformer encoder causes an additional 0.6% drop, resulting in a total 1.5% drop from A0. These results indicate that global feature interaction and attention-based aggregation are both useful for wideangle render-and-compare refinement.

Plug-in Refinement Rows F0–F3 evaluate the mask-free WAPR variant C0 as a plug-in refiner for existing pose estimators. When applied to FreeZeV2 poses, C0 improves AP by 0.6% with a 0.54 s per-image overhead on top of FreeZeV2’s 27.8 s runtime. When applied to GigaPose+GenFlow poses, C0 improves AP by 1.3% with 0.37 s additional time. These results show that WAPR-style wideangle refinement can further improve strong existing estimators, although the remaining gain is naturally smaller when the initial estimates are already strong.

Comparison with Refinement Baselines Table 3 compares WAPR with representative RGB-D refinement baselines on LM-O and YCB-V. This comparison complements the full-pipeline BOP evaluation in Table 4 by focusing on refinement behavior under comparable settings. Compared with OSOP+ICP, WAPR is 4 \times faster and improves mean AR by 29.7%; compared with PPF/SIFT+Zephyr, it improves AR by 26.7%. Under the same hypothesis setting, replacing WAPR with ICP reduces AR by 54.2%, reflecting the sensitivity of ICP to large hypothesis errors. Compared with FoundationPose, which uses small-angle refinement with a maximum correction of 2.5^\circ , WAPR improves AR by 8.3%.

Table 3: Comparison with representative RGB-D refinement baselines on LM-O and YCB-V, where comparable prior-refinement results are available. The last three rows use MUSE detections and 12 hypotheses per instance.
<table><tr><td>Method</td><td>LM-O</td><td>YCB-V</td><td>Mean</td><td>Time (s/img)</td></tr><tr><td>OSOP [47] + ICP</td><td>48.2</td><td>57.2</td><td>52.7</td><td>4</td></tr><tr><td>(PPF [i0], SIFT) + Zephyr [41]</td><td>59.8</td><td>51.6</td><td>55.7</td><td></td></tr><tr><td>MegaPose-RGBD [27]</td><td>58.3</td><td>63.3</td><td>60.8</td><td></td></tr><tr><td>FoundationPose [54] (MUSE) + 12 hypotheses</td><td>67.4</td><td>80.8</td><td>74.1</td><td></td></tr><tr><td>ICP (MUSE) + i2 hypotheses</td><td>26.6</td><td>29.7</td><td>28.2</td><td>35</td></tr><tr><td>WAPR (MUSE) + 12 hypotheses</td><td>75.5</td><td>89.3</td><td>82.4</td><td>1</td></tr></table>

Table 4: Comparison with SOTA methods on seven BOP-Classic-Core datasets for 6D localization (AR, top) and detection (AP, bottom). “2D Det.” reports the detector or hypothesis source. WAPR is trained on SA6D, while other methods use original training or released settings. “Multi-Det.” combines NIDS [35], MUSE [5], SAM6D [32], and CNOS [38]. FP denotes FoundationPose [54]. “Time” reports per-image runtime, and “\*” marks proposed methods.
<table><tr><td colspan="10">Model-based 6D localization of unseen objects – BOP-Classic-Core</td></tr><tr><td></td><td colspan="2">|Method</td><td>|2D Det.</td><td>|LM-O T-LESS TUD-L IC-BIN ITODD</td><td></td><td></td><td></td><td>HB</td><td></td><td>YCB-V|Mean|</td><td>|Time (s)</td></tr><tr><td rowspan="3">fasst</td><td>|Co-op [37]</td><td>|F3DT2D</td><td>73.0</td><td>68.0</td><td>92.9</td><td>62.4</td><td>60.0</td><td>86.3</td><td>88.6</td><td>75.9</td><td>0.765</td></tr><tr><td>WAPR*</td><td>F3DT2D</td><td>75.4</td><td>72.0</td><td>92.5</td><td>66.3</td><td>65.0</td><td>86.7</td><td>89.1</td><td>78.1</td><td>0.904</td></tr><tr><td>WAPR*</td><td>MUSE</td><td>75.4</td><td>73.1</td><td>96.1</td><td>70.9</td><td>70.5</td><td>90.1</td><td>88.3</td><td>80.6</td><td>0.72</td></tr><tr><td rowspan="6">unciced</td><td>[MegaPose [27] Genflow [36]</td><td>|CNOS</td><td>62.6</td><td>48.7</td><td>85.1</td><td>46.7</td><td>46.8</td><td>73.0</td><td>76.4</td><td>62.8</td><td>141.965</td></tr><tr><td></td><td>CNOS</td><td>67.8</td><td>55.6</td><td>81.1</td><td>56.3</td><td>57.5</td><td>79.1</td><td>82.5</td><td>68.6</td><td>11.14</td></tr><tr><td>[SAM6D [32]</td><td>SAM6D</td><td>69.9</td><td>51.5</td><td>90.4</td><td>58.8</td><td>60.2</td><td>77.6</td><td>84.5</td><td>70.4</td><td>4.367</td></tr><tr><td>[FP [54]</td><td>SAM6D</td><td>75.6</td><td>64.6</td><td>92.3</td><td>50.8</td><td>58.0</td><td>83.5</td><td>88.9</td><td>73.4</td><td>29.317</td></tr><tr><td>[Co-op [37]</td><td>F3DT2D</td><td>73.8</td><td>69.5</td><td>92.9</td><td>63.5</td><td>62.9</td><td>87.8</td><td>89.8</td><td>77.1</td><td>6.924</td></tr><tr><td>WAPR*</td><td>CNOS</td><td>76.8</td><td>82.6</td><td>92.4</td><td>70.7</td><td>74.3</td><td>89.1</td><td>91.1</td><td>82.4</td><td>43.068</td></tr><tr><td rowspan="3">|FreeZeV2 WAPR*</td><td>[3,4]|</td><td>|Multi-Det.</td><td>77.7</td><td>79.9</td><td>97.5</td><td>71.1</td><td>75.4</td><td>89.5</td><td>91.8</td><td>83.3</td><td>21.709</td></tr><tr><td></td><td>Multi-Det.</td><td>78.1</td><td>81.7</td><td>96.1</td><td>73.5</td><td>79.8</td><td>91</td><td>91.5</td><td>84.5</td><td>50.816</td></tr><tr><td>Model-based 6D detection of unseen objects – BOP-Classic-Core</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">|Method |LM-O T-LESS TUD-L IC-BIN ITODD</td><td>|2D Det.</td><td></td><td></td><td></td><td></td><td></td><td>HB</td><td></td><td></td><td>YCB-V|Mean|Time (s)</td></tr><tr><td rowspan="3">fasft</td><td>|Co-op [37]</td><td>|F3DT2D</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WAPR*</td><td>F3DT2D</td><td>69.8 74.7</td><td>62.0</td><td>84.1</td><td>56.4</td><td>57.6</td><td>74.6</td><td>80.8</td><td>69.3</td><td>0.865</td></tr><tr><td>WAPR*</td><td>MUSE</td><td>75.6</td><td>64.5 63.6</td><td>96.3 98.6</td><td>67.2 71.8</td><td>68.1 71.5</td><td>77.6 81.6</td><td>87.1 87.6</td><td>76.5 78.6</td><td>0.987 0.828</td></tr><tr><td rowspan="4">uoed</td><td>[Genflow [36]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>[Co-op [37]</td><td>|CNOS CNOS</td><td>59.7</td><td>56.5</td><td>68.8</td><td>43.9</td><td>42.5</td><td>70.7</td><td>60.7</td><td>57.5</td><td>15.486</td></tr><tr><td>WAPR*</td><td>CNOS</td><td>68.3 78.8</td><td>59.6 82.9</td><td>80.8 95.7</td><td>46.9 70.9</td><td>56.0 77.6</td><td>74.3 86.3</td><td>78.2 91.1</td><td>66.3 83.3</td><td>2.202 43.068</td></tr><tr><td>|FreeZeV2 [3, 4]|</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>[Multi-Det.</td><td>80.2</td><td>77.7</td><td>98.8</td><td>72.0</td><td>77.5</td><td>85.4</td><td>91.1</td><td>83.3</td><td>27.776</td></tr><tr><td>WAPR*</td><td></td><td>Multi-Det.</td><td>80.7</td><td>82.5</td><td>98.7</td><td>75.0</td><td>82.2</td><td>88.7</td><td>89.6</td><td>85.3</td><td>50.816</td></tr></table>

## 4.3 BOP Results on 6D Localization and Detection

Table 4 compares WAPR with state-of-the-art methods on seven BOP-Classic-Core datasets for 6D localization and detection. In the unconstrained Multi-Det. setting, WAPR achieves 84.5% AR and 85.3% AP, improving over FreeZeV2 by 1.2% AR and 2.0% AP. This accuracy-oriented setting uses a larger candidate budget and obtains higher mean accuracy with a longer per-image runtime. The fast setting reports the eficiency-oriented operating point under the <1 s perimage budget.

In the fast setting, WAPR with MUSE detections achieves 80.6% AR and 78.6% AP, improving over Co-op by 4.7% AR and 9.3% AP. For a matcheddetector comparison, WAPR is also evaluated with the same F3DT2D detector used by Co-op under the same fast-inference protocol. This variant achieves 78.1% AR and 76.5% AP, improving over Co-op by 2.2% AR and 7.2% AP. These results show that the gain is preserved under the same detector, while the MUSE-based variant gives the highest mean performance among the evaluated fast-setting variants.

## 5 Conclusion

This paper presented WAPR, a zero-shot wide-angle pose refinement model for unseen-object 6D pose estimation. WAPR refines sparse candidate poses by comparing real and rendered RGB-D observations with masks, enabling accurate pose estimation under large candidate-pose deviations. To support wide-angle training, this paper further constructed SA6D, a large-scale synthetic RGB-D dataset with KASAL-assisted rotational symmetry priors. These priors are used to canonicalize symmetry-equivalent candidate–target pose pairs, while anglebalanced loss stabilizes learning across diferent angular ranges. Experiments on seven BOP core datasets show that WAPR achieves strong performance in both fast and unconstrained settings. These results demonstrate that wide-angle refinement, scalable training data with symmetry priors, and sparse candidate evaluation provide an efective solution for unseen-object 6D pose estimation.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant No. 52375487, the Special Fund of Jiangsu Province for Key Research and Development under Grant No. BE2023041, and the Doctoral Research Innovation Ability Improvement Program of Southeast University under Grant No. CXJH\_SEU 26032. The authors sincerely thank the area chairs and anonymous reviewers for their careful reading, constructive feedback, and detailed suggestions.

## References

1. Brachmann, E., Krull, A., Michel, F., Gumhold, S., Shotton, J., Rother, C.: Learning 6D object pose estimation using 3D object coordinates. In: ECCV. pp. 536–551 (2014)

2. Capellen, C., Schwarz, M., Behnke, S.: ConvPoseCNN: Dense convolutional 6D object pose estimation. In: VISIGRAPP. pp. 162–172 (2020)

3. Carafa, A., Boscaini, D., Hamza, A., Poiesi, F.: FreeZe: Training-free zero-shot 6D pose estimation with geometric and vision foundation models. In: ECCV. pp. 414–431 (2024)

4. Carafa, A., Boscaini, D., Poiesi, F.: Accurate and eficient zero-shot 6D pose estimation with frozen foundation models (2025), arXiv preprint arXiv:2506.09784

5. Cho, S., Park, S., Oh, I.: Muse: Model-based uncertainty-aware similarity estimation for zero-shot 2d object detection and segmentation (2025), arXiv preprint arXiv:2510.17866

6. Denninger, M., Sundermeyer, M., Winkelbauer, D., Zidan, Y., Olefir, D., Elbadrawy, M., Lodhi, A., Katam, H.: Blenderproc (2019), arXiv preprint arXiv:1911.01911

7. Doumanoglou, A., Kouskouridas, R., Malassiotis, S., Kim, T.K.: Recovering 6D object pose and predicting next-best-view in the crowd. In: CVPR. pp. 3583–3592 (2016)

8. Downs, L., Francis, A., Koenig, N., Kinman, B., Hickman, R., Reymann, K., McHugh, T.B., Vanhoucke, V.: Google scanned objects: A high-quality dataset of 3d scanned household items. In: ICRA. pp. 2553 – 2560 (2022)

9. Drost, B., Ulrich, M., Bergmann, P., Hartinger, P., Steger, C.: Introducing MVTec ITODD — a dataset for 3D object recognition in industry. In: ICCVW. pp. 2200– 2208 (2017)

10. Drost, B., Ulrich, M., Navab, N., Ilic, S.: Model globally, match locally: Eficient and robust 3d object recognition. In: CVPR. pp. 998–1005 (2010)

11. Ge, Z., Liu, S., Wang, F., Li, Z., Sun, J.: YOLOX: Exceeding YOLO series in 2021 (2021), arXiv preprint arXiv:2107.08430

12. Hai, Y., Song, R., Li, J., Salzmann, M., Hu, Y.: Rigidity-aware detection for 6D object pose estimation. In: CVPR. pp. 8927–8936 (2023)

13. Haugaard, R.L., Buch, A.G.: SurfEmb: Dense and continuous correspondence distributions for object pose estimation with learnt surface embeddings. In: CVPR. pp. 6739–6748 (2022)

14. He, K., Gkioxari, G., Dollar, P., Girshick, R.: Mask r-cnn. In: ICCV. pp. 2980–2988 (2017)

15. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR. pp. 770–778 (2016)

16. Hodan, T., Barath, D., Matas, J.: EPOS: Estimating 6D pose of objects with symmetries. In: CVPR. pp. 11700–11709 (2020)

17. Hodan, T., Haluza, P., Obdrzalek, S., Matas, J., Lourakis, M., Zabulis, X.: T-LESS: An RGB-D dataset for 6D pose estimation of texture-less objects. In: WACV. pp. 880–888 (2017)

18. Hodan, T., Michel, F., Brachmann, E., Kehl, W., Buch, A.G., Kraft, D., Drost, B., Vidal, J., Ihrke, S., Zabulis, X., Sahin, C., Manhardt, F., Tombari, F., Kim, T.K., Matas, J., Rother, C.: BOP: Benchmark for 6D object pose estimation. In: ECCV. pp. 19–35 (2018)

19. Hodan, T., Sundermeyer, M., Drost, B., Labbe, Y., Brachmann, E., Michel, F., Rother, C., Matas, J.: BOP challenge 2020 on 6D object localization. In: ECCV. pp. 577–594 (2020)

20. Hodan, T., Sundermeyer, M., Labbe, Y., Nguyen, V.N., Wang, G., Brachmann, E., Drost, B., Lepetit, V., Rother, C., Matas, J.: BOP challenge 2023 on detection, segmentation and pose estimation of seen and unseen rigid objects. In: CVPRW. pp. 5610–5619 (2024)

21. Hong, Z.W., Hung, Y.Y., Chen, C.S.: RDPN6D: Residual-based dense point-wise network for 6DoF object pose estimation based on RGB-D images. In: CVPRW. pp. 5251–5260 (2024)

22. Hoque, S., Xu, S., Maiti, A., Wei, Y., Arafat, M.Y.: Deep learning for 6D pose estimation of objects — a case study for autonomous driving. Expert Syst. Appl. 223(119838), 1–15 (2024)

23. Kalra, A., Stoppi, G., Marin, D., Taamazyan, V., Shandilya, A., Agarwal, R., Boykov, A., Chong, T.H., Stark, M.: Towards co-evaluation of cameras, HDR, and algorithms for industrial-grade 6DoF pose estimation. In: CVPR. pp. 22691–22701 (2024)

24. Kaskman, R., Zakharov, S., Shugurov, I., Ilic, S.: HomebrewedDB: RGB-D dataset for 6D pose estimation of 3D objects. In: ICCVW. pp. 2767–2776 (2019)

25. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything. In: ICCV. pp. 3992–4003 (2023)

26. Labbe, Y., Carpentier, J., Aubry, M., Sivic, J.: CosyPose: Consistent multi-view multi-object 6D pose estimation. In: ECCV (2020)

27. Labbé, Y., Manuelli, L., Mousavian, A., Tyree, S., Birchfield, S., Tremblay, J., Carpentier, J., Aubry, M., Fox, D., Sivic, J.: MegaPose: 6D pose estimation of novel objects via render & compare. In: CoRL. pp. 1–10 (2022)

28. Lepetit, V., Moreno-Noguer, F., Fua, P.: EPnP: An accurate O(n) solution to the PnP problem. IJCV 81, 155–166 (2009)

29. Li, S., Schieber, H., Corell, N., Egger, B., Kreimeier, J., Roth, D.: Gbot: Graphbased 3d object tracking for augmented reality-assisted assembly guidance. In: VR. pp. 513–523 (2024)

30. Li, Y., Wang, G., Ji, X., Xiang, Y., Fox, D.: DeepIM: Deep iterative matching for 6D pose estimation. In: ECCV. pp. 695–711 (2018)

31. Li, Z., Wang, G., Ji, X.: CDPN: Coordinates-based disentangled pose network for real-time RGB-based 6-DoF object pose estimation. In: ICCV. pp. 7678–7687 (2019)

32. Lin, J., Liu, L., Lu, D., Jia, K.: SAM-6D: Segment anything model meets zero-shot 6D object pose estimation. In: CVPR. pp. 27906–27916 (2024)

33. Liu, J., Sun, W., Liu, C., Zhang, X., Fu, Q.: Robotic continuous grasping system by shape transformer-guided multiobject category-level 6-d pose estimation. IEEE TII 19(11), 11171–11181 (2023)

34. Liu, X., Zhang, R., Zhang, C., Wang, G., Tang, J., Li, Z., Ji, X.: GDRNPP: A geometry-guided and fully learning-based object pose estimator. PAMI 47(7), 5742 – 5759 (2025)

35. Lu, Y., P, J.J., Guo, Y., Ruozzi, N., Xiang, Y.: Adapting pre-trained vision models for novel instance detection and segmentation (2024), arXiv preprint arXiv:2405.17859

36. Moon, S., Son, H., Hur, D., Kim, S.: GenFlow: Generalizable recurrent flow for 6D pose refinement of novel objects. In: CVPR. pp. 10039–10049 (2024)

37. Moon, S., Son, H., Hur, D., Kim, S.: Co-op: Correspondence-based novel object pose estimation. In: CVPR. pp. 11622 – 11632 (2025)

38. Nguyen, V.N., Groueix, T., Ponimatkin, G., Lepetit, V., Hodan, T.: CNOS: A strong baseline for CAD-based novel object segmentation. In: ICCVW. pp. 2126– 2132 (2023)

39. Nguyen, V.N., Groueix, T., Salzmann, M., Lepetit, V.: GigaPose: Fast and robust novel object pose estimation via one correspondence. In: CVPR (2024)

40. Nguyen, V.N., Tyree, S., Guo, A., Fourmy, M., Gouda, A., Lee, T., Moon, S., Son, H., Ranftl, L., Tremblay, J., Brachmann, E., Drost, B., Lepetit, V., Rother, C., Birchfield, S., Matas, J., Labbe, Y., Sundermeyer, M., Hodan, T.: BOP challenge 2024 on model-based and model-free 6D object pose estimation. In: CVPRW (2025)

41. Okorn, B., Gu, Q., Hebert, M., Held, D.: Zephyr: Zero-shot pose hypothesis rating. In: ICRA. pp. 14141–14148 (2021)

42. Park, K., Patten, T., Vincze, M.: Pix2Pose: Pixel-wise coordinate regression of objects for 6D pose estimation. In: ICCV. pp. 7668–7677 (2019)

43. Park, K.B., Choi, S.H., Lee, J.Y.: Self-training based augmented reality for robust 3d object registration and task assistance. Expert Syst. Appl. 238(122331), 1–16 (2024)

44. Peng, S., Zhou, X., Liu, Y., Lin, H., Huang, Q., Bao, H.: PVNet: Pixel-wise voting network for 6DoF object pose estimation. PAMI 44(6), 3212–3223 (2022)

45. Qian, K., Erden, M.S., Kong, X.: Scalable network and adaptive refinement module for 6D pose estimation of diverse industrial components. In: IROS. pp. 11646–11653 (2024)

46. Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. PAMI 39(6), 1137–1149 (2017)

47. Shugurov, I., Li, F., Busam, B., Ilic, S.: Osop: A multi-stage one shot object pose estimation framework. In: CVPR. pp. 6825–6834 (2022)

48. Su, Y., Saleh, M., Fetzer, T., Rambach, J., Navab, N., Busam, B., Stricker, D., Tombari, F.: ZebraPose: Coarse to fine surface encoding for 6DoF object pose estimation. In: CVPR. pp. 6728–6738 (2022)

49. Thalhammer, S., Bauer, D., Hönig, P., Weibel, J.B., García-Rodríguez, J., Vincze, M.: Challenges for monocular 6-d object pose estimation in robotics. IEEE TRO 40(1), 4065–4084 (2024)

50. Tian, Z., Shen, C., Chen, H., He, T.: Fcos: Fully convolutional one-stage object detection. In: ICCV. pp. 9626–9635 (2019)

51. Wang, G., Manhardt, F., Tombari, F., Ji, X.: GDR-Net: Geometry-guided direct regression network for monocular 6D object pose estimation. In: CVPR. pp. 16606– 16616 (2021)

52. Wang, Y., Hu, M., Li, H., Luo, C.: HccePose(BF): Predicting front & back surfaces to construct ultra-dense 2D-3D correspondences for pose estimation. In: ICCV. pp. 7166–7175 (2025)

53. Wang, Y., Luo, C.: Key-axis-based localization of symmetry axes in 3d objects utilizing geometry and texture. TIP 33, 6720–6733 (2024)

54. Wen, B., Yang, W., Kautz, J., Birchfield, S.: FoundationPose: Unified 6D pose estimation and tracking of novel objects. In: CVPR. pp. 17868–17879 (2024)

55. Wu, D., Zhuang, Z., Xiang, C., Zou, W., Li, X.: 6D-VNet: End-to-end 6DoF vehicle pose estimation from monocular RGB images. In: CVPRW. pp. 1238–1247 (2019)

56. Wu, Y., Javaheri, A., Zand, M., Greenspan, M.: Keypoint cascade voting for point cloud based 6DoF pose estimation. In: 3DV. pp. 176–186 (2022)

57. Xiang, Y., Schmidt, T., Narayanan, V., Fox, D.: PoseCNN: A convolutional neural network for 6D object pose estimation in cluttered scenes. In: RSS. pp. 1–10 (2018)

58. Yang, X., Cai, J., Li, K., Fan, X., Cao, H.: A monocular-based tracking framework for industrial augmented reality applications. The International Journal of Advanced Manufacturing Technology 128(5), 2571–2588 (2023)

59. Yu, S., Zhai, D.H., Guan, Y., Xia, Y.: Category-level 6-d object pose estimation with shape deformation for robotic grasp detection (2024), early access in IEEE TNNLS.

60. Zakharov, S., Shugurov, I., Ilic, S.: DPOD: 6D pose object detector and refiner. In: ICCV. pp. 1941–1950 (2019)

61. Zhang, H., Liang, Z., Li, C., Zhong, H., Liu, L., Zhao, C., Wang, Y., Wu, Q.M.J.: A practical robotic grasping method by using 6-d pose estimation with protective correction. IEEE TIE 69(14), 3876–3886 (2022)

62. Zou, W., Wu, D., Tian, S., Xiang, C., Li, X., Zhang, L.: End-to-end 6DoF pose estimation from monocular RGB images. IEEE TCE 67(1), 87–96 (2021)

63. Örnek, E.P., Labbé, Y., Tekin, B., Ma, L., Keskin, C., Forster, C., Hodan, T.: Foundpose: Unseen object pose estimation with foundation features. In: ECCV. pp. 163–182 (2024)