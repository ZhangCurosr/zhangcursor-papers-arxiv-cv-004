# Smartphone-Based, Field-Deployed AI Inspection System for Power

Distribution Infrastructure

Sampath Balaji Masabathula, Srija Vanapalli, Gangadhar Kanchi, V. A. N. Sreenivasa Rao, V. Vijaya Lalitha, and Sasank Chilamkurthy

Abstract—Routine inspection of power distribution infrastructure remains challenging for utilities operating geographically dispersed networks, where conventional manual inspection can be difficult to perform consistently and at scale. This work presents a smartphone-based AI inspection system for assessing targeted distribution-infrastructure conditions from images captured by field personnel. The proposed workflow integrates image-quality and viewing-perspective validation with AI-assisted assessment of pole lean, crossarm and top-cleat tilt, voltage level, vegetation-clearance risk, and distribution-transformer (DTR)– ground clearance. The system integrates deep-learning models with monocular depth estimation, geometric analysis, and rule-based processing within a common multi-stage inspection workflow designed for practical field operation. The system was pilot deployed across five selected distribution feeders within the Eastern Power Distribution Company of Andhra Pradesh Ltd. (APEPDCL), where 1,222 defect instances were identified across the assessed categories and subsequently tracked through a rectification and verification workflow, with 84 defect instances recorded as rectified during the study period. Feeder-level interruption records were also examined over the study period; however, the available observations do not provide sufficient evidence to establish a reliability improvement attributable to the proposed system. A curated multi-module field-captured dataset, PD-Defect, is also publicly released to support reproducible research in distribution-infrastructure inspection. The field deployment demonstrates the practical integration of smartphone-based AI-assisted inspection into field inspection and maintenance activities, providing a mechanism for systematic identification, documentation, rectification tracking, and verification of targeted distribution-infrastructure defects.

Index Terms—Power distribution infrastructure, smartphonebased inspection, computer vision, deep learning, defect detection, AI-assisted inspection, field deployment

## I. INTRODUCTION

OWER distribution infrastructure comprises assets such formers (DTRs) that require periodic inspection for reliable and safe operation. Undetected defects and unsafe conditions may lead to service interruptions, safety hazards, and unplanned maintenance. However, routine field inspection of large distribution networks remains challenging for distribution companies (DISCOMs), as visual assessment often depends on the experience and judgment of field personnel, making consistent and objective evaluation across large numbers of assets difficult.

Recent advances in computer vision have demonstrated the potential of AI-assisted inspection for power distribution infrastructure, with applications including component detection, defect identification, and condition assessment. Much of the existing literature has focused on UAV-based inspection, predominantly targeting transmission infrastructure or isolated inspection tasks. Although these approaches have demonstrated promising performance, their reliance on specialized equipment limits their practicality for routine distributionfeeder inspections.

Practical deployment within DISCOMs, however, requires considerably more than accurate defect detection. Inspection systems must reliably support multiple inspection tasks, remain robust to the variability of real-world field imagery, and integrate seamlessly into existing utility inspection workflows. These practical requirements motivate the need for an integrated AI inspection framework designed specifically for routine distribution-feeder inspection.

To address these challenges, we developed a smartphonebased AI inspection system for power distribution infrastructure. The proposed system integrates image validation and multiple AI-based inspection modules within a multi-stage inspection pipeline. It provides AI-assisted assessment of pole lean, crossarm and top-cleat tilt, vegetation-clearance risk, DTR–ground clearance, and voltage level during routine field inspections. The system has been pilot-deployed within the Eastern Power Distribution Company of Andhra Pradesh Ltd. (APEPDCL), where utility field personnel have used it to conduct pole-to-pole inspections across five live distribution feeders.

The key contributions of this work are as follows: (1) a smartphone-based AI inspection system pilot-deployed within an Indian DISCOM; (2) an integrated multi-task inspection pipeline supporting multiple inspection tasks for power distribution infrastructure; and (3) the public release of PD-Defect [1], a curated multi-module smartphone-captured dataset supporting reproducible research in AI-assisted power distribution infrastructure inspection.

The underlying premise of this study is that systematic AI-assisted inspection can facilitate the identification of distribution-infrastructure defects and their subsequent rectification, which may contribute to reducing defect-related feeder interruptions and improving distribution-system reliability. Accordingly, the field evaluation examines both the performance of individual inspection modules and the use of the complete workflow for field inspection, defect rectification, and observation of feeder interruption behavior.

## II. RELATED WORK

Automated inspection of power infrastructure has been studied using aerial imagery and computer-vision methods. Whitworth et al. [2] used helicopter-mounted video for visual inspection of overhead power lines. Sampedro et al. [3] developed a deep-learning system for automatic recognition and diagnosis of electrical insulator strings. More recently, Zhang et al. [4] proposed a YOLOv11-based method for small-defect detection on power distribution towers using UAV imagery, while Li et al. [5] used UAV-acquired infrared imagery for composite-insulator overheating detection. These studies address a range of inspection tasks but rely on aerial platforms or specialized sensing modalities.

Dedicated hardware has also been investigated for infrastructure monitoring. Barua et al. [6] developed an autonomous pole-climbing robot for power-line inspection and fault detection, while Masoud and Proano [7] proposed an autonomous system for continuous monitoring of wooden utility poles using multimodal sensing. At the ground level, Wang et al. [8] proposed a low-cost monocular-camera method for estimating utility-pole leaning posture from geometric image features without prior model training or camera calibration. These studies demonstrate alternatives to aerial inspection but remain focused on specific inspection or monitoring tasks.

Across these approaches, automated inspection has primarily been applied to individual asset, defect, or conditionassessment tasks using aerial, specialized, or monocular sensing. The combination of smartphone-based image acquisition, multiple distribution-infrastructure inspection tasks, image and perspective validation, and deployment within routine DISCOM field operations is less explored in the literature reviewed here. This work addresses this setting through a unified smartphone-based inspection system integrating image validation, perspective validation, defect assessment, voltagelevel classification, and geometric clearance-risk assessment.

## III. METHODOLOGY

The proposed system is a smartphone-based AI inspection pipeline for routine inspection of power distribution infrastructure. Field personnel capture images of distribution assets using smartphones, after which the system performs inputquality, perspective, and task-specific visibility validation before directing valid inputs to the corresponding inspection modules. The pipeline integrates AI-based, rule-based, and geometric analysis for pole lean, crossarm and top-cleat tilt, voltage-level classification, vegetation-clearance risk, and DTR–ground clearance assessment. This staged design prevents unsuitable inputs from being propagated to downstream inspection tasks while allowing each module to use a method appropriate to its inspection objective.

The overall workflow of the proposed system is shown in Fig. 1.

## A. Data Collection and Dataset Preparation

## 1) Field Data Collection

Field images were collected by utility field personnel across urban and rural locations using entry-level Android smartphones under natural lighting conditions. The collected imagery covered distribution poles, crossarms, overhead conductors with surrounding vegetation, and DTR installations.

Initial field captures revealed that the tilt of the handheld camera could introduce angular deviations, affecting the judgment. To address this variability, the initially collected images were manually tilt-corrected under expert supervision, and a standardized capture protocol was subsequently adopted for pole and crossarm inspection. Field personnel were instructed to capture the target pole head-on from an approximately centered viewpoint, with both the pole top and base fully visible and a camera-to-pole distance of approximately 1–1.5 times the pole height. The survey application also incorporated an automatic image tilt-correction functionality adapted from the open-source Open Camera project [9], using smartphone orientation-sensor data to standardize image orientation during acquisition.

A separate capture protocol was used for DTR–ground clearance assessment. Images were required to include the pole top and base and the complete plinth on the same ground plane, with the pole and plinth visible without overlap.

## 2) Dataset Organization and Filtering

Following field acquisition, the collected images were manually curated, filtered, and organized according to the requirements of the downstream inspection tasks. The resulting data were organized into datasets for pole, crossarm, and topcleat analysis, vegetation-clearance assessment, perspective validation, and DTR–ground clearance assessment.

For pole-lean assessment, the initial dataset comprised 2,448 smartphone-captured images. After manual quality screening, 1,801 images were retained, while 647 images were discarded because of incomplete pole visibility or perspective distortion. Following integration of automatic tilt correction into the survey application and adoption of the standardized capture protocol, an additional 2,557 images were collected. Images with incomplete pole visibility or perspective distortion that could compromise reliable assessment were filtered out, and pole-top regions of interest (ROIs) were generated from the retained imagery for crossarm, top-cleat, and voltage-level assessment. For voltage-level classification, additional poletop crops with slight perspective distortion were included because such distortion did not materially affect voltage-level identification from the visible insulator configuration.

For vegetation-clearance risk assessment, field images were screened for visibility of both the conductor span and surrounding vegetation. Images lacking sufficient vegetation near the distribution line or adequate conductor visibility were excluded, and additional field data were collected to increase coverage of suitable vegetation–conductor scenarios. Unlike pole and crossarm assessment, this task did not require strict frontal alignment, provided that the conductor span and surrounding vegetation were captured under a reasonable viewing perspective. The resulting vegetation-clearance dataset comprised 1,781 smartphone-captured images.

![](images/5a287267666bbf30bbe6570a7880160e7983d338f45bd25acd05db82a7db144e.jpg)  
Fig. 1: Overall workflow of the proposed smartphone-based AI-assisted inspection system. Smartphone-captured field images undergo input-quality assessment and are processed through the corresponding pole or DTR–ground clearance assessment pathways. Task-specific validation and assessment are then performed, and the resulting inspection outputs are recorded for rectification tracking and post-rectification verification.

A separate dataset was collected for DTR–ground clearance assessment. Of 1,148 images collected, 476 were accepted for model development and 672 were rejected because of perspective distortion or incomplete visibility of the pole or plinth bases.

Images rejected because of perspective distortion across the different inspection datasets were subsequently retained for training the DINOv3 [10] perspective-validation models. Rejected full images were used for full-image perspective validation, rejected pole-top crops were used for crop-level perspective validation, and rejected transformer images were used for perspective validation within the clearance-risk pipeline. This allowed otherwise unsuitable field images to contribute to input-quality assessment rather than being discarded entirely.

## 3) Annotation Process

A structured annotation protocol was established to ensure consistency across the different inspection tasks. Objectdetection and pose datasets were prepared in the YOLO format, while vegetation semantic segmentation annotations were prepared in the COCO format [11]. Domain experts defined the defect categories, visual criteria, and structural boundaries before annotation, and all annotated datasets were manually reviewed for annotation quality.

For pole-lean assessment, bounding boxes tightly enclosed the visible pole shaft from top to base, excluding crossarms, top-cleats, conductors, and background. Poles were classified as Leaned when they exhibited a consistent visual deviation from vertical relative to the ground line and other fixed reference structures visible in the image, and as Straight when no observable deviation from vertical was present. Only poles with clearly identifiable top and base were annotated. The corresponding pose dataset used two keypoints along the approximate pole centerline, at the top and bottom of the visible pole. When partial occlusion by vegetation or other objects occurred, the centerline was continued through the occluded region only when its geometry could be reliably inferred; otherwise, samples with indeterminate keypoints were excluded.

For crossarm and top-cleat assessment, the V-crossarm was treated as a separate class from the other crossarms because of its distinct visual structure. Bounding boxes tightly enclosed the main horizontal structural member of the crossarm, the complete V-crossarm assembly from the apex to the outer ends of both crossarm members, and the top-cleat from the upper surface to the lower clamping end. Insulators, conductors, and field-fabricated or welded extensions were excluded. Components were classified as Straight when visually aligned with the expected structural orientation and as Tilted when they exhibited a consistent and visually discernible deviation from the expected orientation, assessed relative to the pole axis. For partially occluded top-cleat instances, the component extent was estimated conservatively from the visible geometry.

The corresponding pose dataset used two keypoints for each pole, crossarm, V-crossarm, and top-cleat instance. For crossarms, the keypoints were placed at the centers of the two outer ends of the visible crossarm structure. For V-crossarms, the keypoints were placed at the centers of the two outer structural joints of the V-crossarm assembly. For top-cleats, the keypoints were placed along the vertical centerline at the top and bottom of the visible cleat. When the bottom clamp was clearly visible, the lower keypoint was placed at the bottom of the visible clamp. Keypoints were not extended into regions that were not visible. Partially visible or bent components were annotated only when the required keypoints could be identified reliably; otherwise, the instances were excluded from the pose dataset.

For voltage-level classification, voltage labels were assigned based on the visible insulator configuration. Crossarm and Vcrossarm bounding boxes included their associated insulators, as the insulator configuration provides the basis for voltagelevel classification.

Vegetation-clearance images were annotated using COCO semantic segmentation with two classes: vegetation and conductor. Vegetation was annotated only within the conductor corridor or its immediate vicinity relevant to clearance assessment; ground vegetation, distant trees, and foliage clearly separated from the conductor span were excluded. Conductor masks were annotated along the visible span from the current pole toward the adjacent pole until occlusion or the next pole. Disconnected visible conductor segments caused by occlusion were annotated as separate masks, while minor interruptions were retained within a single mask when the conductor path remained clearly identifiable.

For DTR–ground clearance assessment, poles and plinths were annotated using tight bounding boxes. Pole boxes extended from the pole top to the ground-contact point, while plinth boxes extended from the upper surface to the base. When the pole base was partially occluded, the bounding box was extended to the estimated ground-contact point.

The PD-Defect dataset used in this work is publicly available [1].

## B. Input Quality and Perspective Validation

## 1) Image Quality Assessment

Each input image is first screened using a rule-based quality check based on image resolution, sharpness, pixelation, lighting, and noise. Images that do not satisfy these predefined quality criteria are discarded before downstream analysis.

## 2) Pole Detection and Selection

A YOLO12m-based detector is used to localize distribution poles from the top to the base [12]. When multiple poles are detected, a rule-based selection procedure identifies the primary target pole based on its proximity to the image center and its apparent pole height. For each detected pole, a selection score is computed as

$$
S = \frac { w _ { 1 } } { d _ { c } } + w _ { 2 } h _ { p } ,
$$

where $d _ { c }$ is the Euclidean distance between the pole boundingbox center and the image center, $h _ { p }$ is the apparent pole height, and $w _ { 1 }$ and w<sub>2</sub> are empirically selected weights used to balance the two terms. The pole with the highest score is selected for subsequent analysis. The selected pole is then subjected to a visibility check to ensure that both its top and base are sufficiently visible; poles failing this check are discarded.

## 3) Full-Image Perspective Validation

A DINOv3 ViT-B/16 model is used to validate the perspective of the full input image and reject captures with perspective distortion. The model performs binary classification between valid and perspective-distorted images. Only images classified as valid are passed to downstream inspection modules.

## 4) Pole-Top Crop Validation

A second DINOv3 ViT-B/16 model is used to validate the perspective of the extracted pole-top regions before crossarm, top-cleat assessment, and voltage-level classification. Unlike full-image validation, which evaluates the overall perspective of the input image, crop-level validation specifically assesses the local viewing geometry of the pole-top region. This additional validation is particularly important for crossarm and top-cleat assessment, where local perspective distortion can alter their apparent orientation. The same validated crops are then used for voltage-level classification to maintain consistent input conditions. Only crops classified as valid are passed to the corresponding inspection modules.

## 5) DTR–Ground Clearance Image Validation

A third DINOv3 ViT-B/16 model is used to validate the input images for DTR–ground clearance assessment. Before perspective validation, a rule-based visibility check ensures that the pole top, pole base, and plinth base are sufficiently visible. The DINOv3 model then performs binary classification between valid and perspective-distorted images to reject inputs that do not satisfy the required viewing geometry for ratiobased clearance assessment. Only images that pass both checks are passed to the clearance-risk assessment module.

## C. Distribution-Asset Inspection

## 1) Pole Lean Assessment

The selected distribution pole is classified as either leaned or straight using a YOLO12m-based object-detection model. The predicted class is associated with the pre-selected target pole using intersection-over-union (IoU) matching. When multiple pole detections are present, the prediction with the highest IoU with the selected pole is retained, provided that the IoU exceeds 0.5.

To provide an interpretable measure of apparent pole inclination, a supplementary rule-based module estimates the lean angle from the pole’s image-space geometry. Let $( x _ { 1 } , y _ { 1 } )$ and $( x _ { 2 } , y _ { 2 } )$ denote the two keypoints defining the detected pole axis, corresponding to the top and bottom points of the pole, respectively. The apparent lean angle in image space, expressed in degrees, is computed as

$$
\theta _ { \mathrm { d e g } } = \frac { 1 8 0 } { \pi } \mathrm { a t a n 2 } \left( x _ { 1 } - x _ { 2 } , y _ { 2 } - y _ { 1 } \right) ,\tag{1}
$$

where the reversed y-difference accounts for the downwardincreasing y-coordinate in image space. This quantity represents an image-space measure of apparent inclination and does not correspond directly to the physical lean angle of the pole. In the deployed system, a pole is considered leaned when the estimated apparent inclination exceeds $1 0 ^ { \circ }$

## 2) Pole-Top Region Extraction

Following pole lean assessment, a standardized pole-top region is extracted from the selected pole for subsequent crossarm, top-cleat, and voltage-level assessment. The crop is centered on the selected pole and covers the upper 50% of its detected height, with an additional 20% extension upward to include the crossarm and top-cleat structures. Horizontally, the crop extends 0.57 times the detected pole height on either side of the pole center.

## 3) Crossarm and Top-Cleat Tilt Assessment

The crossarm, V-crossarm, and top-cleat detection and pose models operate on the corresponding pole-top region crop. A YOLO12m-based object-detection model predicts the orientation of the detected components as either straight or tilted. The model distinguishes three component types: Crossarm, Vcrossarm, and Top-cleat.

To provide an interpretable measure of component tilt, a supplementary rule-based module estimates the deviation of each component from its expected mounting orientation relative to the pole. The pole, crossarm, V-crossarm, and topcleat axes are obtained from the corresponding pose-model keypoints. For an axis defined by two keypoints $( x _ { 1 } , y _ { 1 } )$ and $( x _ { 2 } , y _ { 2 } )$ , its image-space orientation in degrees is computed as

$$
\alpha = \frac { 1 8 0 } { \pi } \mathrm { a t a n 2 } ( y _ { 1 } - y _ { 2 } , x _ { 2 } - x _ { 1 } ) ,\tag{2}
$$

where the reversed $y \cdot$ -difference accounts for the downwardincreasing y-coordinate in image space. Let $\alpha _ { p }$ and $\alpha _ { o }$ denote the orientations of the pole and the corresponding component, respectively. Their angular separation is first computed as

$$
\delta = \operatorname* { m i n } \left( | \alpha _ { o } - \alpha _ { p } | , 3 6 0 ^ { \circ } - | \alpha _ { o } - \alpha _ { p } | \right) .\tag{3}
$$

Because the pole and component are represented by undirected axes, the angular separation is then folded to the smaller angle between the two axes:

$$
\delta _ { \mathrm { a c u t e } } = \operatorname* { m i n } \left( \delta , 1 8 0 ^ { \circ } - \delta \right) ,\tag{4}
$$

yielding an angle in the range $[ 0 ^ { \circ } , 9 0 ^ { \circ } ]$ . The component tilt is subsequently defined as the deviation of this measured component–pole angle from its expected mounting angle:

$$
\theta _ { \mathrm { t i l t } } = \left| \theta _ { \mathrm { n o m i n a l } } - \delta _ { \mathrm { a c u t e } } \right| ,\tag{5}
$$

where $\theta _ { \mathrm { n o m i n a l } } = 9 0 ^ { \circ }$ for crossarms and V-crossarms, which are expected to be perpendicular to the pole, and $\theta _ { \mathrm { n o m i n a l } } = 0 ^ { \circ }$ for top cleats, which are expected to be aligned with the pole. Thus, a tilt of $0 ^ { \circ }$ indicates the expected mounting orientation, while larger values indicate increasing angular deviation from that orientation. In the deployed system, a component is considered tilted when the computed $\theta _ { \mathrm { t i l t } }$ exceeds $1 5 ^ { \circ }$

## 4) Voltage-Level Classification

A YOLO12m-based classifier is used to predict the voltage level of the distribution line from visual cues in the pole-top region, including the insulators associated with the crossarm. The model predicts two voltage classes: High-Tension (HT) and Low-Tension (LT). The predicted voltage labels are mapped to the corresponding crossarms using a twostage geometric matching procedure based on center-in-box and intersection-over-union (IoU) criteria. Top-cleat instances are assigned the voltage level of the nearest V-crossarm.

## 5) Vegetation-Clearance Risk Assessment

## a) Problem Formulation

Reliable assessment of vegetation-clearance risk from fieldacquired smartphone images is challenging because vegetation and conductors at different depths may appear adjacent or spatially separated due to perspective projection, making pixelspace proximity an unreliable indicator of their actual spatial relationship. Consequently, semantic segmentation alone does not provide sufficient depth information to assess the spatial relationship between vegetation and conductors. The proposed method therefore combines semantic segmentation, monocular relative depth estimation, and geometric reasoning to distinguish potential clearance risks from perspective-induced false positives.

## b) Vegetation and Conductor Segmentation

Vegetation and overhead conductors are identified using a semantic segmentation model based on Mask2Former [13] with a Swin-Base backbone. The resulting binary vegetation and conductor masks are denoted by $M _ { v }$ and $M _ { c } ,$ respectively. The vegetation mask is dilated using a $1 5 \times 5$ square structuring element, while the conductor mask is retained without dilation. The resulting masks are used for subsequent depth and geometric risk analysis.

## c) Relative Depth Representation

Depth Anything V2 Large [14] is used to generate a monocular depth map for each input image. The resulting values represent relative rather than metric depth and are not interpreted as physical distances. Let $D ( x , y )$ denote the raw depth value at pixel $( x , y )$ . For each image, the valid depth values are independently min–max normalized to the range 0–100, producing the normalized depth representation $Z ( x , y )$ . Under this depth convention, larger values of $Z ( x , y )$ indicate regions that are relatively nearer to the camera, whereas smaller values indicate regions that are relatively farther away. The normalized depth map is then combined with the segmentation masks for the geometric analysis.

## d) Geometric Risk Analysis

The spatial relationship between vegetation and conductors is analyzed separately for overlapping and non-overlapping masks. Both configurations are considered potentially indicative of clearance risk, but for different reasons. In the overlap case, mask overlap does not necessarily imply physical contact, since the vegetation and conductor may lie at different depths. In the non-overlap case, spatial separation between the masks does not necessarily imply adequate clearance, since vegetation may still be sufficiently close to the conductor to pose a potential clearance risk. Accordingly, separate geometric analyses are applied to the two cases. The operating thresholds used in the geometric analysis were empirically determined on the labeled validation dataset.

The reference values used for depth-based image-scale normalization and candidate screening were empirically derived from representative field images. For each sampled image, a reference object was selected, and a tight bounding box was defined around the visible object. The normalized relativedepth value within the bounding box and the corresponding image-plane pixel length of the object were recorded for each image and aggregated across multiple images to define the reference values. This procedure yielded $Z _ { \mathrm { r e f } } ~ = ~ 2 8 . 3 8$ on the normalized relative-depth scale and an image-plane screening threshold of 420 pixels. Because the depth estimates are relative rather than metric, these values provide a heuristic reference for image-scale normalization and candidate screening rather than a physical depth-to-distance calibration.

1. Overlap Analysis: For overlapping masks, the overlap region $M _ { o }$ is defined as the intersection of the vegetation and conductor masks. The median depth $Z _ { o }$ of the overlap region is computed. Representative vegetation and conductor depths, $Z _ { v }$ and $Z _ { c } ,$ are obtained from a local region surrounding the conductor, with global median over the corresponding mask used when sufficient local depth information is unavailable. The depth-separation measure is defined as

$$
\Delta _ { o } = \mathrm { m i n } \left( \left| Z _ { o } - Z _ { c } \right| , \left| Z _ { o } - Z _ { v } \right| \right) .\tag{6}
$$

A smaller $\Delta _ { o }$ indicates greater depth similarity between the overlap region and at least one of the two reference regions. Based on empirically determined thresholds, an overlap risk state of high, medium, or low is assigned when $\Delta _ { o } < 9 . 9$ $9 . 9 \leq \Delta _ { o } < 4 9 . 4 $ or $\Delta _ { o } \geq 4 9 . 4$ , respectively.

2. Non-Overlap Analysis: For non-overlapping masks, candidate pairs of conductor and vegetation boundary points are evaluated as potential interaction locations. For each candidate pair $( p _ { c } , p _ { v } )$ , the 2D Euclidean distance $x _ { i }$ is computed. To account heuristically for variation in apparent image scale associated with the relative-depth representation, the imageplane separation $x _ { i }$ for each candidate pair is scaled using the corresponding relative-depth value and an empirically selected reference value as

$$
x _ { \mathrm { r e f } } = x _ { i } \frac { Z _ { i } } { Z _ { \mathrm { r e f } } } ,\tag{7}
$$

where $Z _ { i }$ is the normalized relative-depth value at the vegetation boundary point and $Z _ { \mathrm { r e f } } ~ = ~ 2 8 . 3 8$ is an empirically selected reference value on the normalized relative-depth scale. The resulting $x _ { \mathrm { r e f } }$ is treated as a depth-scaled image-plane separation for candidate screening and does not represent a metric distance or a physical projection. Candidate pairs satisfying $x _ { \mathrm { r e f } } ~ \leq ~ 4 2 0$ are retained for subsequent depthconsistency analysis.

For each retained pair, the local normalized depths of the conductor and vegetation are estimated as the median of valid depth values within a $3 \times 3$ neighborhood of the corresponding candidate point, restricted to the respective conductor or vegetation mask. Let $Z _ { c }$ and $Z _ { v }$ denote the resulting median normalized depths for the conductor and vegetation, respectively. The normalized depth separation between the conductor and vegetation is then defined as

$$
\delta = \lvert Z _ { c } - Z _ { v } \rvert .\tag{8}
$$

A normalized depth-separation threshold of 6 is used to identify potential vegetation–conductor interactions. If $\delta < 6$ for any retained candidate pair, the non-overlap analysis indicates a potential interaction. When vegetation is present in the local near-conductor band and at least one retained pair satisfies $\delta \ < \ 6 .$ , an initial high-risk state is assigned. The minimum image-plane edge-to-edge distance between the conductor and vegetation is then evaluated. If this distance exceeds 15 pixels, the risk state is downgraded from high to low. The resulting geometric features and risk state are incorporated into a feature vector.

## e) Risk Score Calibration and Operating-Point Selection

The resulting feature vector is used by a logistic regression model to obtain a vegetation-clearance risk score. Five-fold stratified cross-validation is used to generate out-of-fold risk scores on the labeled validation dataset. An operating threshold $\tau _ { p 8 5 }$ is then selected from the precision–recall curve by maximizing recall among thresholds achieving at least 0.85 precision for the risky class. The final logistic regression model is subsequently fitted using the full labeled validation dataset, and the selected threshold is applied to its predicted risk score during deployment. An image is classified as RISKY when the score is greater than or equal to $\tau _ { p 8 5 }$ , and SAFE otherwise.

## 6) Field-Based Module Evaluation

For vegetation-clearance risk assessment, an independent test set was formed from available production field data collected during the pilot deployment. The reference risk labels were provided primarily by field personnel and subsequently reviewed by domain experts. The resulting test set comprised 3,298 images, including 671 Risky and 2,627 Safe images, and was used to evaluate the complete vegetation-clearance riskassessment module under field conditions. These production test images were held out from the publicly released PD-Defect dataset. Although the labels were reviewed by domain experts, they are based on field assessments and may involve some degree of subjective judgment.

The validation and production test sets yielded false-positive rates of 11.02% and 8.72%, respectively, while the corresponding false-negative rates were 21.84% and 6.56%.

## 7) DTR–Ground Clearance Assessment

## a) Pole–Plinth Detection and Pair Selection

A YOLO12m-based detector is used to localize distribution poles and DTR plinths in the input image. The detected pole and plinth instances are subsequently processed to select the corresponding pole–plinth pair. The bounding-box heights of the selected pole and plinth are then used for clearance-ratio computation.

## b) Clearance-Ratio Formulation

The clearance assessment is based on the relative height of the pole and transformer plinth. Let H and h denote the ideal heights of the pole and plinth, respectively, and let x denote an effective loss in ground clearance, such as an increase in ground level due to resurfacing. Under this change in the ground level, the effective heights of both the pole and plinth are reduced by x. The baseline pole-to-plinth height ratio is given by

$$
r = { \frac { H } { h } } .\tag{9}
$$

After a clearance loss of $x ,$ the corresponding ratio becomes

$$
r ^ { \prime } = \frac { H - x } { h - x } .\tag{10}
$$

For $h \ < \ H$ and $0 < x < h , r ^ { \prime } > r $ , indicating that a reduction in effective ground clearance results in an increased pole-to-plinth height ratio. This occurs because the same clearance loss x represents a larger fraction of the plinth height than of the pole height, causing the denominator to decrease proportionally more than the numerator.

Let $H _ { p }$ and $H _ { p l }$ denote the heights of the selected pole and plinth bounding boxes, respectively. The operational clearance ratio is then defined as

$$
R = \frac { H _ { p } } { H _ { p l } } .\tag{11}
$$

The operational ratio R therefore provides an image-based indicator of relative clearance condition and serves as an approximation to the theoretical ratio $r ^ { \prime }$ for inputs satisfying the required visibility and perspective conditions.

Although the current implementation is evaluated for plinthmounted distribution transformers, the same clearance-ratio formulation can be extended to other transformer mounting configurations, including pillar-mounted and pole-mounted transformers. For these configurations, the pole and distribution transformer can be annotated using bounding boxes. Assuming that both share a common ground plane, the mounting height can be estimated from the vertical pixel distance between the bottom edge of the transformer bounding box and a horizontal baseline aligned with the bottom edge of the pole bounding box. This estimated mounting height can then replace the plinth height in the clearance-ratio formulation, allowing the same ratio-based approach to be applied to other mounting configurations. The present study is limited to plinth-mounted transformers because sufficient field data for the other mounting configurations were not available for annotation and evaluation.

## c) Risk Assessment

For valid inputs satisfying the visibility and perspective requirements described in Section III-B5, the operational ratio R is evaluated using predefined decision thresholds based on the recorded pole height H, in metres, obtained from the utility asset database. Assuming that one-sixth of the pole height is embedded below ground, the corresponding above-ground pole-height factor is $5 / 6 \approx 0 . 8 3$ . The denominator values represent the baseline plinth heights used for risk classification, with 1.83 m representing the standard plinth height and 1.52 m representing the lower baseline height associated with the very-risky condition. Accordingly, the risky threshold is defined as

$$
T _ { \mathrm { r i s k y } } = \frac { H ( 0 . 8 3 ) } { 1 . 8 3 } ,\tag{12}
$$

while the very-risky threshold is defined as

$$
T _ { \mathrm { v e r y - r i s k y } } = { \frac { H ( 0 . 8 3 ) } { 1 . 5 2 } } .\tag{13}
$$

The installation is classified as very risky when $R \ >$ $T _ { \mathrm { v e r y - r i s k y } }$ , as risky when $T _ { \mathrm { r i s k y } } < R \leq T _ { \mathrm { v e r y - r i s k y } }$ , and as no risk otherwise.

## D. Model Training and Evaluation

## 1) YOLO12m Object-Detection Models

YOLO12m object-detection models were used for pole localization and for multiple inspection tasks, including polelean assessment, crossarm, V-crossarm, and top-cleat tilt assessment, voltage-level classification, and DTR plinth–pole detection. Each model was trained on a task-specific annotated dataset.

For pole detection, a YOLO12m model was trained on 2,252 images, with 1,665 for training (73.93%), 293 for validation (13.01%), and 294 for testing (13.06%). For polelean assessment, the object-detection dataset comprised 1,801 images, with 1,441 for training (80.01%), 181 for validation (10.05%), and 179 for testing (9.94%). For crossarm, Vcrossarm, and top-cleat tilt assessment, a YOLO12m model was trained on 1,686 images, with 1,175 for training (69.69%), 254 for validation (15.07%), and 257 for testing (15.24%). The dataset contained six classes corresponding to straight and tilted crossarm, V-crossarm, and top-cleat instances. For voltage-level assessment, a YOLO12m model was trained on 1,772 images, with 1,337 for training (75.45%), 218 for validation (12.30%), and 217 for testing (12.25%). For DTR plinth– pole detection, the released dataset contained 476 images, with 362 for training (76.05%), 57 for validation (11.97%), and 57 for testing (11.97%). The dataset contained 1,191 annotated instances, comprising 699 pole and 492 plinth instances.

All YOLO12m models were trained using 640 × 640 input resolution for up to 350 epochs with mixed-precision. The optimizer was set to auto. The best-performing validation checkpoint was retained for final evaluation. During inference, task-specific confidence thresholds were applied: 0.35 for polelean and voltage-level models, 0.50 for crossarm, V-crossarm, and top-cleat tilt models, and 0.25 for transformer plinth–pole detection.

On the test sets, the pole detector achieved 97.08% mAP@0.5, providing reliable localization for primary-pole selection and subsequent visibility validation. The DTR plinth– pole detector achieved 97.45% mAP@0.5, with class-wise mAP@0.5 values of 98.92% for plinths and 95.97% for poles, supporting the subsequent ratio-based clearance assessment.

## 2) YOLO12m Pose Models

YOLO12m pose models were used to estimate structural keypoints for pole-lean and crossarm, V-crossarm, and topcleat tilt assessment. The pole pose model operates on the full input image, whereas the component pose models operate on the corresponding pole-top-region crops. The pose models were trained on task-specific datasets with keypoint annotations. For pole-lean assessment, the keypoint dataset comprised 1,798 images, with 1,438 for training (79.98%), 181 for validation (10.07%), and 179 for testing (9.96%). The dataset contained 2,025 annotated pole instances, with two keypoints per instance corresponding to the bottom and top of the pole. For crossarm, V-crossarm, and top-cleat tilt assessment, the keypoint dataset comprised 1,686 images, with 1,175 for training (69.69%), 254 for validation (15.07%), and 257 for testing (15.24%). The dataset contained 4,951 annotated instances across pole, crossarm, V-crossarm, and top-cleat classes, with two keypoints per instance. Horizontalflip augmentation was disabled for this pose model due to differences in keypoint ordering across the annotated component types.

The training configuration was kept consistent with the corresponding YOLO12m object-detection models, including the 640 × 640 input resolution, training epochs, optimizer configuration, and mixed-precision.

## 3) DINOv3 Models

Three DINOv3 ViT-B/16 models were used for perspective validation of full input images, pole-top crops, and DTR plinth–pole images. Each model performed binary classification between valid and perspective-distorted inputs. The fullimage validator was trained on 2,169 images, with 1,605 for training (74.00%), 282 for validation (13.00%), and 282 for testing (13.00%). These images comprised 1,018 valid and 1,151 perspective-distorted samples. The pole-top crop validator was trained on 1,743 images, with 1,221 for training (70.05%), 260 for validation (14.92%), and 262 for testing (15.03%), comprising 945 valid and 798 perspective-distorted samples. The DTR plinth–pole validator was trained on 1,038 images, with 768 for training (73.99%), 135 for validation (13.01%), and 135 for testing (13.01%), comprising 438 valid and 600 perspective-distorted samples.

All three models were initialized with pretrained weights, with the final four transformer blocks and a dropoutregularized linear classification head unfrozen for fine-tuning. All models used 448 × 448 input images and were trained for 75 epochs with a batch size of 16 using the AdamW optimizer, a learning rate of $1 \times 1 0 ^ { - 4 }$ , and weight decay of $1 \times 1 0 ^ { - 4 }$ . Training used the letterbox blur–jitter transformation with color jitter during training. The best-performing checkpoint based on validation accuracy was retained for the final evaluation.

## 4) Mask2Former Segmentation Model

A Mask2Former model with a Swin-Base backbone was fine-tuned for semantic segmentation of vegetation and overhead conductors. The model was initialized with the pretrained facebook/mask2former-swin-base-ade-semantic checkpoint from Hugging Face and trained to predict three classes: background, vegetation, and conductor. The dataset comprised 1,781 images, with 1,324 for training (74.34%) and 457 for validation (25.66%). Input images were resized to $1 2 8 0 \times 1 2 8 0$ pixels, with random horizontal and vertical flips and brightness, contrast, and sharpness adjustments applied as data augmentation.

Training used the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 5 }$ , a cosine learning rate schedule, and a weight decay of $5 \times 1 0 ^ { - 2 }$ . The model was trained for up to 150 epochs with a per-device batch size of 1 and gradient accumulation over 8 steps and mixed-precision enabled.

The final Mask2Former checkpoint was selected based on downstream vegetation-clearance risk-assessment performance on the labeled validation set, using the operating-point criterion of maximizing recall subject to at least 0.85 precision for the RISKY class. For the selected checkpoint, the validation segmentation performance yielded an mIoU of 74.57%, with IoU values of 87.64% for vegetation and 40.77% for conductors. Conductor segmentation achieved an F1-score of 57.92%, with 69.51% precision and 49.64% recall. The resulting masks were used for relative-depth and geometric analysis; thus, pixel-level performance alone does not determine the final image-level risk classification.

For the asset inspection tasks producing image-level decisions, evaluation was performed at the image level using accuracy, precision, recall, and F1-score.

## IV. RESULTS AND DISCUSSION

The quantitative performance of the individual inspection and validation modules is presented in Tables I–III. The following results focus on the field deployment of the complete system, the defects identified under actual utility conditions, their subsequent rectification status, and the observed feeder interruption behavior.

TABLE I: Image-Level Classif. Metrics on Val. & Test Sets
<table><tr><td>Class</td><td>Split</td><td>Acc</td><td>Prec</td><td>Rec</td><td>F1</td></tr><tr><td>Pole Lean Assessment</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Leaned_Pole</td><td>Test</td><td>89.94</td><td>88.51</td><td>90.59</td><td>89.53</td></tr><tr><td></td><td>Val</td><td>93.92</td><td>93.70</td><td>97.54</td><td>95.58</td></tr><tr><td>Straight_Pole</td><td>Test</td><td>91.06</td><td>90.15</td><td>97.54</td><td>93.70</td></tr><tr><td></td><td>Val</td><td>95.58</td><td>93.42</td><td>95.95</td><td>94.67</td></tr><tr><td>Crossarm and Top-Cleat Tilt Assessment</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>crossarm_straight</td><td>Test</td><td>94.94</td><td>92.98</td><td>95.50</td><td>94.22</td></tr><tr><td></td><td>Val</td><td>94.49</td><td>91.74</td><td>95.24</td><td>93.46</td></tr><tr><td>crossarm_tilted</td><td>Test</td><td>93.00</td><td>78.67</td><td>96.72</td><td>86.76</td></tr><tr><td></td><td>Val</td><td>93.70</td><td>80.95</td><td>92.73</td><td>86.44</td></tr><tr><td>v-crossarm_straight</td><td>Test</td><td>94.55</td><td>91.43</td><td>95.05</td><td>93.20</td></tr><tr><td>v-crossarm_tilted</td><td>Val</td><td>94.88</td><td>88.37</td><td>96.20</td><td>92.12</td></tr><tr><td></td><td>Test</td><td>94.94</td><td>88.16</td><td>94.37</td><td>91.16</td></tr><tr><td>topcleat_straight</td><td>Val</td><td>94.49</td><td>90.67</td><td>90.67</td><td>90.67</td></tr><tr><td></td><td>Test</td><td>90.27</td><td>89.19</td><td>88.39</td><td>88.79</td></tr><tr><td>topcleat_tilted</td><td>Val</td><td>93.70</td><td>93.52</td><td>91.82</td><td>92.66</td></tr><tr><td></td><td>Test</td><td>93.39</td><td>83.87</td><td>88.14</td><td>85.95</td></tr><tr><td></td><td>Val</td><td>94.88</td><td>85.11</td><td>86.96</td><td>86.02</td></tr><tr><td>Voltage-Level Classification</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>high_tension</td><td>Test</td><td>97.70</td><td>98.82</td><td>98.24</td><td>98.53</td></tr><tr><td></td><td>Val</td><td>95.41</td><td>98.15</td><td>95.78</td><td>96.95</td></tr><tr><td>low_tension</td><td>Test</td><td>91.71</td><td>90.76</td><td>93.91</td><td>92.31</td></tr><tr><td></td><td>Val</td><td>92.66</td><td>91.53</td><td>94.74</td><td>93.10</td></tr><tr><td>Vegetation-Clearance Risk Assessment Risky</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Test</td><td>93.44</td><td>73.25</td><td>93.44</td><td>82.12</td></tr><tr><td></td><td>Val</td><td>78.16</td><td>85.64</td><td>78.16</td><td>81.73</td></tr><tr><td></td><td>Test</td><td>91.28</td><td>98.20</td><td>91.28</td><td>94.61</td></tr><tr><td></td><td>Val</td><td>88.98</td><td>82.89</td><td>88.98</td><td>85.83</td></tr></table>

![](images/c526ef5568b3a46fb9e888b8ad94ac488cd3874c8acbcd88253d05c395ea716e.jpg)  
Fig. 2: Representative outputs showing (a) detected crossarm, V-crossarm, and top-cleat components with tilt angles of $2 . 8 6 ^ { \circ }$ $7 . 2 6 ^ { \circ } .$ , and $1 0 . 5 2 ^ { \circ }$ , respectively, estimated from axes defined by their detected keypoints. Although tilt is detected for the V-crossarm and top-cleat, their angles remain below the 15<sup>◦</sup> threshold and are therefore classified as non-defective; and (b) detected high-tension (HT) and low-tension (LT) structures for voltage-level classification.

TABLE II: Perspective-Validation Metrics on Val. & Test Sets
<table><tr><td>Class</td><td>Split</td><td>Acc</td><td>Prec</td><td>Rec</td><td>F1</td></tr><tr><td>Full-Image</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Valid</td><td>Test</td><td>91.67</td><td>88.32</td><td>91.67</td><td>89.96</td></tr><tr><td></td><td>Val</td><td>89.47</td><td>88.15</td><td>89.47</td><td>88.81</td></tr><tr><td>perspective-distorted</td><td>Test</td><td>89.33</td><td>92.41</td><td>89.33</td><td>90.85</td></tr><tr><td></td><td>Val</td><td>89.26</td><td>90.48</td><td>89.26</td><td>89.86</td></tr><tr><td>Pole-Top Crop</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Valid</td><td>Test</td><td>97.18</td><td>96.50</td><td>97.18</td><td>96.84</td></tr><tr><td></td><td>Val</td><td>97.16</td><td>93.84</td><td>97.16</td><td>95.47</td></tr><tr><td>perspective-distorted</td><td>Test</td><td>95.83</td><td>96.64</td><td>95.83</td><td>96.23</td></tr><tr><td></td><td>Val</td><td>92.44</td><td>96.49</td><td>92.44</td><td>94.42</td></tr><tr><td>DTR plinth-pole</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Valid</td><td>Test</td><td>84.21</td><td>81.36</td><td>84.21</td><td>82.76</td></tr><tr><td></td><td>Val</td><td>89.47</td><td>89.47</td><td>89.47</td><td>89.47</td></tr><tr><td>perspective-distorted</td><td>Test</td><td>85.90</td><td>88.16</td><td>85.90</td><td>87.01</td></tr><tr><td></td><td>Val</td><td>92.31</td><td>92.31</td><td>92.31</td><td>92.31</td></tr></table>

TABLE III: Pose Estimation and Derived Angular Metrics
<table><tr><td>Class</td><td>Split</td><td>Prec</td><td>Rec</td><td>mAP@0.5</td><td>Angular MAE (°)</td></tr><tr><td>Pole (Lean)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pole</td><td>Test</td><td>96.86</td><td>93.40</td><td>97.75</td><td>0.29</td></tr><tr><td></td><td>Val</td><td>96.47</td><td>90.33</td><td>98.19</td><td>0.60</td></tr><tr><td>Pole-top Component</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pole</td><td>Test</td><td>99.24</td><td>98.10</td><td>99.39</td><td>0.37</td></tr><tr><td></td><td>Val</td><td>96.85</td><td>94.49</td><td>98.09</td><td>0.38</td></tr><tr><td>Crossarm</td><td>Test</td><td>94.29</td><td>92.30</td><td>97.23</td><td>0.51</td></tr><tr><td></td><td>Val</td><td>93.52</td><td>91.71</td><td>95.87</td><td>0.43</td></tr><tr><td>V-crossarm</td><td>Test</td><td>99.65</td><td>99.45</td><td>99.50</td><td>0.51</td></tr><tr><td></td><td>Val</td><td>98.74</td><td>97.01</td><td>99.39</td><td>0.52</td></tr><tr><td>Top-cleat</td><td>Test</td><td>93.24</td><td>93.66</td><td>97.78</td><td>2.01</td></tr><tr><td></td><td>Val</td><td>89.22</td><td>95.52</td><td>96.49</td><td>1.67</td></tr></table>

![](images/0dd1a771c6c90c4b6076fe4d264a4fc0f126dbd9aa8f734b7fa0ecbd87e34ef4.jpg)  
Fig. 3: Representative pole-lean assessment showing a detected pole with an apparent inclination of $1 7 . 8 2 ^ { \circ }$ , estimated from the pole axis defined by its detected keypoints, resulting in a leaned classification.

## A. Field Deployment and Inspection

To evaluate the operational utility of the proposed system beyond model-level performance, the system was pilot deployed across five selected distribution feeders under actual utility field conditions. Field personnel used the smartphonebased inspection application to capture images of distribution assets and assess the supported inspection categories during the feeder inspection process.

TABLE IV: Defect Identification Across the Five Selected Feeders. The Defective column denotes distinct poles associated with identified defects; Crossarm and Top-Cleat denote tilt-related defect instances.
<table><tr><td>Feeder</td><td>Surveyed</td><td>Defective</td><td>Pole Lean</td><td>Crossarm</td><td>Top-Cleat</td><td>DTR-Ground Clearance</td><td>Vegetation Clearance</td><td>Total</td></tr><tr><td>11KV BONDADA</td><td>557</td><td>157</td><td>16</td><td>12</td><td>11</td><td></td><td>128</td><td>167</td></tr><tr><td>11KV PERUMALI</td><td>807</td><td>214</td><td>62</td><td>25</td><td>12</td><td></td><td>125</td><td>225</td></tr><tr><td>11KV YERRAKONERU</td><td>853</td><td>333</td><td>43</td><td>19</td><td>21</td><td></td><td>272</td><td>359</td></tr><tr><td>11KV BADIVANIPETA</td><td>836</td><td>317</td><td>19</td><td>19</td><td>9</td><td></td><td>287</td><td>338</td></tr><tr><td>11KV A.RAVIVALASA</td><td>757</td><td>120</td><td>17</td><td>18</td><td>7</td><td>3</td><td>88</td><td>133</td></tr><tr><td>Total</td><td>3,810</td><td>1,141</td><td>157</td><td>93</td><td>60</td><td>12</td><td>900</td><td>1,222</td></tr></table>

Across the five feeders, the inspection identified 1,222 defect instances associated with 1,141 distinct poles. Vegetationclearance risk accounted for the largest share, with 900 instances, followed by pole-lean defects (157), top-cleat-related defects (60), crossarm-related defects (93), and DTR–ground clearance defects (12). The feeder-wise distribution of the identified defect instances is presented in Table IV.

## B. Defect Findings and Rectification Status

The identified defects were automatically recorded in the inspection application as pending rectifications. Following physical rectification, field personnel captured a post-rectification image of the corresponding asset through the application, and the system validated whether the identified defect had been resolved. When a defect was fully resolved, the corresponding rectification was accepted and cleared. If only part of the identified defect was resolved, the resolved portion was cleared while the remaining defect continued to be recorded as pending. Unsuccessful rectification attempts likewise remained pending for further action.

During the study period, 84 defect cases were recorded as rectified across the five selected feeders, comprising 34 vegetation-clearance instances, 32 crossarm-tilt instances, 15 top-cleat-tilt instances, and 3 pole-lean instances. Consequently, only a subset of the identified conditions could be physically rectified and subsequently verified within the study period.

## C. Feeder Interruption Analysis

To examine feeder-level interruption behavior in the context of the field deployment, all recorded interruption events for the five selected feeders were analyzed over the study period. The interruption-type field was not used for filtering, as the recorded type is operator-entered and does not necessarily represent the actual cause of the interruption. Accordingly, the analysis considers the complete set of recorded interruption events without cause-based filtering. April through June 2026 were used as the baseline reference period, preceding the commencement of the field survey in July 2026. The field survey was conducted during July 2026 over approximately three weeks, followed by the subsequent observation period beginning in August 2026.

Across the five feeders, the April–June 2026 baseline period comprised 893 recorded interruption events, corresponding to a monthly average of 297.67 events. During July 2026, 273 interruption events were recorded, followed by 180 events in August and 334 events in September. The feeder-wise monthly interruption records are presented in Table V.

TABLE V: Feeder-Wise Monthly Interruption Records Across the Study Period
<table><tr><td>Feeder</td><td>Apr.</td><td>May</td><td>Jun.</td><td>Jul.</td><td>Aug.</td><td>Sep.</td></tr><tr><td>11KV A.RAVIVALASA</td><td>59</td><td>79</td><td>91</td><td>65</td><td>42</td><td>86</td></tr><tr><td>11KV BADIVANIPETA</td><td>54</td><td>40</td><td>52</td><td>29</td><td>30</td><td>63</td></tr><tr><td>11KV BONDADA</td><td>32</td><td>68</td><td>36</td><td>53</td><td>17</td><td>42</td></tr><tr><td>11KV PERUMALI</td><td>34</td><td>62</td><td>79</td><td>50</td><td>51</td><td>78</td></tr><tr><td>11KV YERRAKONERU</td><td>48</td><td>78</td><td>81</td><td>76</td><td>40</td><td>65</td></tr><tr><td>Total</td><td>227</td><td>327</td><td>339</td><td>273</td><td>180</td><td>334</td></tr></table>

The monthly interruption records show variation in feeder interruption behavior over the study period. The number of recorded interruption events decreased during August relative to the April–June baseline average and increased again in September. Such month-to-month variation may be influenced by multiple network and operational factors, including weather conditions, equipment failures, supply-side issues, and other events outside the scope of the present inspection workflow. However, the available interruption records do not provide sufficient information to isolate the contribution of these factors or to attribute the observed variation to the proposed inspection system. Therefore, the interruption records are interpreted as field-level reliability observations rather than as a direct measure of the effect of the proposed system on feeder reliability.

## D. Operational Implications and Limitations

The field deployment demonstrates the practical integration of the proposed smartphone-based AI-assisted inspection workflow into field inspection and maintenance activities. The workflow provides a mechanism for systematic identification, documentation, rectification tracking, and verification of targeted distribution-infrastructure defects within a common inspection process.

Only a subset of the identified defects could be physically rectified and subsequently verified during the study period, primarily because of operational constraints in obtaining the line-clearance permissions required for rectification work on energized distribution infrastructure. Such clearances require temporary interruption of power supply and must therefore be coordinated with feeder operating conditions and other planned or emergent network activities.

This limitation also constrains the extent to which downstream maintenance outcomes could be evaluated. In addition, feeder interruption behavior can be influenced by multiple network and operational factors, including conditions beyond those addressed through the current inspection workflow. The available field observations therefore do not provide sufficient evidence to establish a reliability improvement attributable to the proposed system. The results instead demonstrate the operational utility of AI-assisted inspection as a mechanism for systematic defect identification, documentation, rectification tracking, and verification within a broader distribution-system maintenance approach, in which identified defects can be addressed alongside other network maintenance and operational activities.

## V. CONCLUSION AND FUTURE WORK

This study presented a smartphone-based, field-deployed AI inspection workflow for assessing targeted distributioninfrastructure conditions using images captured by field personnel. The workflow integrates image-quality and viewingperspective validation with AI-assisted assessment of pole lean, crossarm, V-crossarm, and top-cleat tilt, voltage level, vegetation-clearance risk, and DTR–ground clearance within a common inspection process.

The system was pilot deployed across five selected distribution feeders under actual utility field conditions. During the deployment, the system identified 1,222 defect instances across the assessed defect categories. These defects were subsequently tracked through a rectification and verification workflow, with 84 cases recorded as rectified during the study period. The field deployment demonstrates the practical integration of the proposed smartphone-based AI-assisted inspection workflow into field inspection and maintenance activities.

Feeder-level interruption records were also examined over the study period. The observed variation in interruption behavior did not provide sufficient evidence to establish a reliability improvement attributable to the proposed system. This limitation is primarily associated with the limited number of identified defects that were physically rectified during the study period and the potential variation in feeder interruptions due to multiple network and operational factors. Consequently, the observed interruption behavior should be interpreted as a field-level reliability observation rather than as a direct measure of the effect of the proposed inspection workflow on feeder reliability.

The proposed system provides a practical foundation for incorporating AI-assisted inspection into periodic distributioninfrastructure maintenance. Future work will focus on extending deployment across a larger number of feeders and longer observation periods, increasing the number of identified defects that are physically rectified and subsequently verified, expanding the range of distribution-infrastructure defects assessed by the system, and monitoring feeder interruption behavior over longer periods. Such extended field studies can provide a stronger basis for evaluating the relationship between AI-assisted defect identification, subsequent rectification, and distribution-system reliability.

## ACKNOWLEDGMENT

The authors express their sincere gratitude to Prudhvi Tej Immadi, IAS, Chairman and Managing Director, and Sri T. V. Surya Prakash, Director–Operations, of the Eastern Power Distribution Company of Andhra Pradesh Ltd. (APEPDCL), for their support for this initiative. The authors also acknowledge APEPDCL, Visakhapatnam, for its support and cooperation. Special thanks are extended to the field data collection team and to Marla Krishna Sumwith and Shreya Vanapalli for their contributions to the annotation of the dataset used in this study.

## REFERENCES

[1] S. B. Masabathula, S. Vanapalli, G. Kanchi, S. R. V.A.N, V. L. V, and S. Chilamkurthy, “PD-Defect: A Field-Captured Smartphone Image Dataset for AI-Based Inspection of Power Distribution Infrastructure,” 10 2026. [Online]. Available: https://figshare.com/articles/dataset/ / 34329402

[2] C. C. Whitworth and A. W. Duller, “Aerial video inspection of overhead power lines,” Power Engineering Journal, vol. 15, no. 1, pp. 25–32, 2001.

[3] C. Sampedro, J. Rodriguez-Vazquez, A. Rodriguez-Ramos, A. Carrio, and P. Campoy, “Deep learning-based system for automatic recognition and diagnosis of electrical insulator strings,” IEEE Access, vol. 7, pp. 101 283–101 308, 2019.

[4] J. Zhang, S. Chen, W. Wang, and Q. Wang, “Fully autonomous real-time defect detection for power distribution towers: A small target defect detection method based on YOLOv11n,” Sensors, vol. 25, no. 20, p. 6445, 2025.

[5] C. Li, Y. Shi, M. Lu, S. Zhou, C. Xie, and Y. Chen, “A composite insulator overheating defect detection system based on infrared image object detection,” IEEE Transactions on Power Delivery, vol. 40, no. 1, pp. 203–214, 2025.

[6] I. J. Barua, M. I. Salman, S. Dutta, and N. S. Plabon, “Autonomous pole climbing robot for real-time power line inspection and fault detection,” in 2025 IEEE 2nd International Conference on Computing, Applications and Systems (COMPAS), 2025.

[7] A. Masoud and J. C. Proano, “Pole integrity monitoring system (pims): An autonomous solution for utility pole maintenance,” in 2026 IEEE 8th International Conference and Workshop on Obuda on Electrical<sup>´</sup> and Power Engineering (CANDO-EPE), 2026, pp. 359–366.

[8] L. Wang, G. Liu, S. Wang, and H. Wei, “Understanding of leaning utility poles for visual monitoring of power distribution infrastructure,” Journal of Civil Structural Health Monitoring, vol. 14, no. 8, pp. 1857–1868, 2024.

[9] M. Harman, “Open camera: Open source Android camera app,” [Online]. Available: https://opencamera.org.uk, 2024.

[10] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab,´ C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. Jegou, P. Labatut, and P. Bojanowski, “DINOv3,” 2025.´ [Online]. Available: https://arxiv.org/abs/2508.10104

[11] T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick, and P. Dollar, “Microsoft´ coco: Common objects in context,” 2015. [Online]. Available: https://arxiv.org/abs/1405.0312

[12] Y. Tian, Q. Ye, and D. Doermann, “YOLOv12: Attentioncentric real-time object detectors,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 78 433–78 457. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2025/file/7103444259031cc58051f8c9a4868533-Paper-Conference.pdf

[13] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, “Masked-attention mask transformer for universal image segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 1290–1299.

[14] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” arXiv:2406.09414, 2024.