# PCB-MC: Missing Component Analysis in Printed Circuit Boards

1<sup>st</sup> Betsy Villa   
EEMCS-CS-DMB   
University of Twente   
Enschede, The Netherlands   
b.j.villabrochero@utwente.nl   
2<sup>nd</sup> Ian Gibson   
ET-DPM-AMSPES   
University of Twente   
Enschede, The Netherlands   
i.gibson@utwente.nl   
3<sup>rd</sup> Estefania Talavera   
EEMCS-CS-DMB   
University of Twente   
Enschede, The Netherlands   
e.talaveramartinez@utwente.nl

Abstract—Detecting missing components on printed circuit boards (PCBs) differs fundamentally from conventional object detection, as the model must localize components that are not present. We introduce PCB-MC, a curated dataset for missing component detection with footprint level annotations built on top of the RF100 dataset. The dataset contains 197 distinct board types, each corresponding to a unique PCB design, with multiple augmented samples per type. We also provide benchmark results on PCB-MC by evaluating a diverse set of supervised and unsupervised methods. To ensure fair evaluation, we propose board type aware cross validation splits that prevent layout leakage between training and test sets. Supervised models showcase high false negative rates on unseen board designs, and unsupervised anomaly detection methods fail entirely due to the lack of spatial alignment with a board specific reference. These results confirm that missing component detection on diverse PCB layouts remains an open challenge. We release PCB-MC and all training protocols to support reproducible research on structural absence detection in industrial inspection.

Index Terms—Missing Component Detection, Industrial Visual Inspection, Object Detection, and Dataset.

## I. INTRODUCTION

Automated visual inspection is critical in Printed Circuit Board (PCB) manufacturing, where undetected defects can lead to device failure and costly rework [1]. While substantial progress has been made in detecting visible components and surface defects, missing component detection remains comparatively underexplored.

Unlike conventional object detection, missing component detection in PCBs requires inferring the expected footprint of an absent component based on subtle structural cues such as exposed solder pads and silkscreen outlines (Figure 1). This distinction is not captured by existing PCB benchmarks, which focus on component recognition or visible defect detection and lack annotations for missing components at their expected locations, making them unsuitable for evaluating absence based detection.

In this work, we introduce PCB-MC, a dataset for missing component detection built on RF100 [2]. PCB-MC provides footprint level annotations for absent components across 615 images and 31 categories, comprising over 9,400 missing instances. The dataset spans 197 distinct types of boards, each corresponding to a unique PCB layout, with an average of 3.12±1.02 samples per type. Missing components are labeled at their expected locations, enabling direct supervision of structural absence.

![](images/4a8149dda46b7bedfc2c6fa214934282dbdfe69c8eb805f0ae500fe95883aa84.jpg)

![](images/6a58b9811fd7441624fb14eef61d7f58a494f8154b142a47060703cbb9915fd5.jpg)  
Fig. 1. Illustration of missing component annotations in PCB-MC. Color coded bounding boxes denote some missing components: capacitors (blue), resistors (green), ICs (yellow), and LEDs (pink). The zoomed region highlights subtle footprint cues such as exposed solder pads and silkscreen outlines.

We benchmark modern detection models [3]–[7] alongside unsupervised anomaly detection methods [8]–[11]. In addition, we establish a controlled evaluation framework with board type aware splits to prevent layout memorization and high resolution inputs (1024 × 1024) to preserve fine grained structural cues, ensuring fair comparisons under realistic industrial conditions.

Beyond benchmarking, we conduct a controlled analysis to better understand the challenges of missing component detection, particularly examining the roles of candidate region generation and classification under weak visual evidence.

The PCB-MC annotations and the evaluation protocol will be made publicly available<sup>1</sup> to support reproducible research on structural absence detection in industrial inspection.

## II. RELATED WORK

## A. Computer Vision in Industrial Inspection

Most studies in this field have focused on anomaly detection, where models learn normal patterns and detect deviations [12], [13]. Methods such as [8], [9], [10], and [11] perform well on industrial anomaly benchmarks, but are primarily designed for visible defects. However, these approaches are less effective for complex production lines with small components or limited training data, where defining ’normal’ behavior is significantly more challenging. An example is PCB assembly. As electronics become more common and advanced, PCBs are getting more complex, with smaller parts and denser layouts that are much harder for standard models to inspect.

## B. PCB Inspection and Defect Detection

In practical applications, production environments exhibit considerable variability. High volume production lines typically employ automated assembly and inspection processes. However, numerous low and medium volume productions such as prototyping, specialized industrial boards, repair, and small batch manufacturing continue to rely on human assembly or reworking. In these contexts, the implementation of an automated system for detecting defects, particularly those involving missing components, is crucial. Such systems are essential for preventing latent failures, minimizing the need for manual reinspection, and enhancing consistency among operators [14].

Early PCB inspection relied on classical techniques such as template matching [15] and image differencing [16], which are highly sensitive to misalignment and illumination changes.

Recent advances in deep learning have significantly improved robustness and scalability in PCB inspection. Convolutional neural network (CNN) detectors, particularly YOLO [3]–[5] style architectures, are widely adopted for real time component localization and defect detection [17], [18].Transformer base detectors such as RT-DETR [19] further enhance global context modeling, which can be beneficial for complex board layouts [6].

Two stage detectors, including Faster R-CNN [20] and Mask R-CNN [21], have also been applied to PCB defect detection, often achieving strong localization performance at the cost of higher computational overhead [22]. Prior PCB specific work has focused on enhancing small object detection [23] and improving detection flexibility via architectural modifications [24]. More recent approaches incorporate contextual reasoning and attention mechanisms to further boost performance [25].

Despite these advances, most existing work focuses on detecting visible components or surface level defects (e.g., scratches, solder bridges, missing holes). Multi class missing component detection remains significantly more challenging than single defect scenarios, highlighting the need for dedicated datasets and systematic evaluation protocols.

## C. PCB Datasets

Previous works on PCB inspection have been defined by the provided labels in the publicly available datasets. RF100 [2],

FPIC [26], FICS-PCB [27], PCB-Metal [28], and PCB analysis [29] are primarily designed for component recognition tasks. Defect oriented datasets such as DeepPCB [30], PCB Defect [31], and DsPCBSD [32] focus on visible surface anomalies or template comparison approaches [33]. While these datasets support defect detection, they do not explicitly model missing components at their expected locations. PCB-MC provides new labels for the task of missing components detection, which provides a new benchmark for the area.

## III. PCB-MC: A DATASET FOR MISSING COMPONENT DETECTION

PCB-MC is derived from the “Printed Circuit Board” dataset from Roboflow Universe (RF100) [2], which contains 615 images with diverse layouts and component types. PCB-MC extends RF100 with annotations for missing components at their expected footprint locations. Moreover, all annotations were manually reviewed. We corrected inaccurate bounding box localizations, standardized label names, merged visually equivalent subclasses, and removed ambiguous categories to improve annotation consistency.

After our refinement and annotation protocol, our newly introduced PCB-MC dataset contains 615 images annotated across 31 classes. These 31 classes include 23 of the present components, 8 of which are also identified as absent. Table I presents the instance distribution across present and missing categories. The dataset exhibits substantial class imbalance, with passive components such as resistors and capacitors dominating both present and missing annotations. This imbalance reflects realistic industrial assembly conditions, where small passive components are both frequent and prone to omission.

In addition to instance counts, missing components are predominantly small in spatial extent relative to the full board image resolution. This motivates high resolution evaluation settings to preserve fine grained footprint cues.

Annotation Protocol for Missing Components: Missing components are annotated at expected footprint locations using cues such as solder pads, silkscreen outlines, and layout regularities. All annotations are cross validated by multiple annotators. In total, PCB-MC contains 117,091 present component instances and 9,435 missing component instances. Missing components are annotated at their expected footprint locations, enabling supervised learning for absence localization.

## IV. BENCHMARK PROTOCOL ON PCB-MC

We described the tasks that we propose on top of our newly introduced PCB-MC dataset, the implemented methods for missing object detection, and the designed evaluation protocol.

## A. Benchmark Tasks

PCB-MC supports multiple evaluation settings derived from the unified annotation set:

• Task A (All Classes): Joint detection of present and missing components across all 31 categories (615 images).

• Task M (Missing Only): Detection restricted to the 8 missing component classes (293 images).

TABLE I  
COMPONENT CLASS DISTRIBUTION IN THE PCB-MC DATASET.
<table><tr><td>Class</td><td>#Classes</td><td>#Present</td><td>#Missing</td></tr><tr><td>Button</td><td>1</td><td>235</td><td>0</td></tr><tr><td>Capacitor</td><td>2</td><td>46 377</td><td>3177</td></tr><tr><td>Clock</td><td>1</td><td>121</td><td>0</td></tr><tr><td>Connector</td><td>1</td><td>4011</td><td>0</td></tr><tr><td>Diode</td><td>2</td><td>212</td><td>146</td></tr><tr><td>Display</td><td>1</td><td>17</td><td>0</td></tr><tr><td>Electrolytic Cap.</td><td>1</td><td>697</td><td>0</td></tr><tr><td>EM</td><td>1</td><td>138</td><td>0</td></tr><tr><td>Ferrite Bead</td><td>2</td><td>316</td><td>42</td></tr><tr><td>Fuse</td><td>1</td><td>23</td><td>0</td></tr><tr><td>Heatsink</td><td>1</td><td>14</td><td>0</td></tr><tr><td>IC</td><td>2</td><td>6845</td><td>169</td></tr><tr><td>Inductor</td><td>2</td><td>192</td><td>72</td></tr><tr><td>Jumper</td><td>1</td><td>291</td><td>0</td></tr><tr><td>LED</td><td>2</td><td>677</td><td>19</td></tr><tr><td>Pads</td><td>1</td><td>327</td><td>0</td></tr><tr><td>Pins</td><td>1</td><td>1028</td><td>0</td></tr><tr><td>Potentiometer</td><td>1</td><td>25</td><td>0</td></tr><tr><td>Resistor</td><td>2</td><td>50 355</td><td>4450</td></tr><tr><td>Switch</td><td>1</td><td>165</td><td>0</td></tr><tr><td>Test Point</td><td>1</td><td>1108</td><td>0</td></tr><tr><td>Transistor</td><td>1</td><td>3904</td><td>0</td></tr><tr><td>Unknown</td><td>1</td><td>0</td><td>1360</td></tr><tr><td>Zener Diode</td><td>1</td><td>13</td><td>0</td></tr><tr><td>Total</td><td>31</td><td>117091</td><td>9435</td></tr></table>

TABLE II  
PCB-MC DATASET SPLITS AND THEIR STATISTICS.
<table><tr><td>Subset</td><td>Class Type</td><td>#Cls</td><td>#Imgs</td></tr><tr><td>PCB-MC-A</td><td>Components + Missing</td><td>31</td><td>615</td></tr><tr><td>PCB-MC-M</td><td>Missing parts only</td><td>8</td><td>293</td></tr><tr><td>PCB-MC-C</td><td>Components only</td><td>23</td><td>615</td></tr></table>

• Task C (Components Only): Detection restricted to the 23 present component classes (615 images).

These tasks enable controlled comparison between standard component detection and structural absence detection under consistent annotation conditions. Table II summarizes the dataset composition under each benchmark configuration.

## B. Missing Component Detection Methods

a) Supervised Object Detection Models: We evaluate five object detection architectures covering complementary paradigms, including convolutional detectors YOLOv8/11/26 [3]–[5] and transformer based models RT-DETR [19] and D-FINE [7]. We additionally evaluate tiled inference using SAHI [34] to assess whether increased local detail improves detection. All models are trained under identical settings to ensure fair comparison.

Furthermore, in this work, we design a two stage missing component detection pipeline that leverages object proposals by an object detector and a specialized classifier. We rely on the YOLOv11 model trained on the PCB-MC-A training split for each fold, comprising all 31 classes across both present and missing component annotations for the first stage of object proposals.YOLOv11 is then run at a low confidence threshold (conf = 0.05) to generate candidate bounding boxes across the entire image without class filtering. At this threshold, the detector produces approximately 240 proposals per image, covering every region it considers potentially relevant.

In the second stage, we rely on the ResNet-18 backbone for classification [35]. ResNet-18 is trained on proposals of present and missing components. We use focal loss [36] (α = $0 . 7 5 , \ \gamma \ : = \ : 2 . 0 )$ to address the 2:1 present to missing class ratio. During training, each patch is expanded by a factor of 1.4× to capture surrounding context such as pad geometry and silkscreen outlines. High confidence missing predictions are retained and filtered with non maximum suppression.

b) Anomaly Detection Models: Given the broad use of anomaly detection methods in industrial inspection, we assess their performance on the PCB-MC dataset. We evaluate the major anomaly detection families [12] PatchCore [8], PaDiM [9], DRAEM [10], and Reverse Distillation [11]. All models are implemented using Anomalib [37] and trained on images with boards including all present components, then tested on full images with boards with missing components.

Pixel level anomaly maps are converted to bounding boxes via thresholding (0.995 quantile to control false positives) [13], connected component extraction with noise removal, and non maximum suppression (IoU = 0.3) to merge overlapping detections.

## C. Evaluation

Detection performance is evaluated using mAP@0.5 as the primary metric, following standard PCB detection benchmarks [24], [28], together with F1 score and False Negative Rate (FNR) to better assess missed detections. Given the industrial importance of avoiding missed defects, we emphasize the FNR for missing component classes.

PCB-MC contains 197 board types, where each board type corresponds to a unique PCB layout and component arrangement. The images are of high resolution and with small components. Therefore, the missing object detection methods are applied at the image patch level.

To mitigate data leakage, we adopt a board type aware 5 fold cross validation protocol in which all image patches belonging to the same board type are assigned to the same fold. For each fold, 70% of board types are used for training and 30% for validation. Results are reported as mean ± standard deviation across folds without task specific hyperparameter tuning.

## D. Implementation Details

All models are initialized with COCO pretrained weights [38], which provides diverse object level features that improve convergence and generalization. The pretrained weights are then fine tuned on PCB-MC for 300 epochs using AdamW [39] with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and cosine scheduling.

To mitigate class imbalance and improve robustness to orientation and illumination variations, we apply data augmentation, including flipping, rotation, brightness/contrast adjustment, and mosaic augmentation. Inference uses a confidence threshold of 0.25 and class aware non maximum suppression.

Missing component footprints are small relative to the full board resolution, therefore all experiments are conducted with NVIDIA GPU acceleration at $1 0 2 4 \times 1 0 2 4$ resolution to preserve fine grained structural cues such as pads and silkscreen outlines, and configuration files are provided for reproducibility.

TABLE III  
DETECTION PERFORMANCE ACROSS DATASET SUBSETS (MEAN ± STD). TASK M (MISSING COMPONENTS) IS THE PRIMARY TASK OF INTEREST.
<table><tr><td>Method</td><td>Task</td><td>mAP</td><td>F1</td><td>FNR</td></tr><tr><td>YOLOv8 [3]</td><td>A</td><td>0.29±0.05</td><td> $0 . 3 7 { \pm } 0 . 0 3$ </td><td>0.69±0.04</td></tr><tr><td></td><td>M</td><td> $0 . 0 9 { \pm } 0 . 1 3$ </td><td> $0 . 1 5 { \pm } 0 . 0 2$ </td><td>0.90±0.02</td></tr><tr><td></td><td>C</td><td>0.37±0.07</td><td>0.46±0.05</td><td>0.51±0.07</td></tr><tr><td>YOLOv11 [4]</td><td>A</td><td>0.30±0.05</td><td> $0 . 3 9 { \pm } 0 . 0 6$ </td><td>0.67±0.05</td></tr><tr><td></td><td>M</td><td> $0 . 0 9 { \pm } 0 . 0 3$ </td><td> $0 . 1 5 { \pm } 0 . 0 4$ </td><td>0.87±0.03</td></tr><tr><td></td><td>C</td><td> $\mathbf { 0 . 4 2 \pm 0 . 0 9 }$ </td><td> $\mathbf { 0 . 5 0 } \pm \mathbf { 0 . 0 8 }$ </td><td> $0 . 5 7 { \pm } 0 . 0 6 $ </td></tr><tr><td>YOLO26 [5]</td><td>A</td><td> $0 . 2 9 { \pm } 0 . 0 3$ </td><td> $0 . 3 6 \pm 0 . 0 4$ </td><td> $0 . 6 9 { \pm } 0 . 0 2$ </td></tr><tr><td></td><td>M</td><td> $0 . 0 8 { \pm } 0 . 0 4$ </td><td> $0 . 1 5 { \pm } 0 . 0 5$ </td><td>0.87±0.05</td></tr><tr><td></td><td>C</td><td> $0 . 3 7 { \pm } 0 . 0 7$ </td><td> $0 . 4 7 { \pm } 0 . 0 3$ </td><td>0.61±0.07</td></tr><tr><td>YOLOv8 + SAHI [34]</td><td>A</td><td>0.27±0.05</td><td>0.45±0.05</td><td>0.51±0.08</td></tr><tr><td></td><td>M</td><td> $0 . 0 6 { \pm } 0 . 0 3$ </td><td> $0 . 1 3 { \pm } 0 . 0 7$ </td><td> $0 . 9 2 { \pm } 0 . 0 5$ </td></tr><tr><td></td><td>C</td><td> $0 . 3 4 \pm 0 . 1 3$ </td><td> $0 . 4 6 \pm 0 . 0 5$ </td><td> $0 . 5 3 { \pm } 0 . 0 5$ </td></tr><tr><td>YOLOv11 + SAHI [34]</td><td>A</td><td> $0 . 2 9 { \pm } 0 . 0 5$ </td><td> ${ \bf 0 . 4 7 \pm 0 . 0 5 }$ </td><td> $\mathbf { 0 . 5 0 { \pm } 0 . 0 9 }$ </td></tr><tr><td></td><td>M</td><td> $0 . 0 8 { \pm } 0 . 0 2$ </td><td> $\mathbf { 0 . 1 6 \pm 0 . 0 5 }$ </td><td> $0 . 9 0 { \scriptstyle \pm 0 . 0 4 }$ </td></tr><tr><td></td><td>C</td><td> $0 . 3 6 { \pm } 0 . 1 0$ </td><td> $0 . 4 8 { \pm } 0 . 0 4$ </td><td> $\mathbf { 0 . 5 1 \pm 0 . 0 4 }$ </td></tr><tr><td>YOLO26 + SAHI [34]</td><td>A</td><td> $0 . 2 9 { \pm } 0 . 0 6$ </td><td> $0 . 4 6 \pm 0 . 0 4$ </td><td> $0 . 5 2 { \pm } 0 . 0 6$ </td></tr><tr><td></td><td>M</td><td> $0 . 0 8 \pm 0 . 0 3$ </td><td> $0 . 1 3 { \pm } 0 . 0 5$ </td><td> $0 . 9 2 { \pm } 0 . 0 3$ </td></tr><tr><td></td><td>C</td><td> $0 . 3 8 { \pm } 0 . 0 7$ </td><td> $0 . 4 6 \pm 0 . 0 8$ </td><td> $0 . 5 8 { \pm } 0 . 0 8$ </td></tr><tr><td>RT-DETR [6]</td><td>A</td><td> $0 . 2 8 { \pm } 0 . 0 6$ </td><td> $0 . 3 7 { \pm } 0 . 0 5$ </td><td> $0 . 6 9 { \pm } 0 . 0 5$ </td></tr><tr><td></td><td>M</td><td> $0 . 0 4 \pm 0 . 0 3$ </td><td> $0 . 1 1 \pm 0 . 0 5$ </td><td> $0 . 9 2 { \pm } 0 . 0 4$ </td></tr><tr><td></td><td>C</td><td> $0 . 3 8 { \pm } 0 . 0 9$ </td><td> $0 . 4 6 \pm 0 . 0 8$ </td><td>0.58±0.08</td></tr><tr><td>D-FINE [7]</td><td>A</td><td> $0 . 2 4 \pm 0 . 0 3$ </td><td> $0 . 2 5 { \pm } 0 . 0 3$ </td><td> $0 . 7 5 { \pm } 0 . 0 3$ </td></tr><tr><td></td><td>M</td><td> $0 . 0 3 { \pm } 0 . 0 2$ </td><td> $0 . 0 4 \pm 0 . 0 2$ </td><td> $0 . 9 3 { \pm } 0 . 0 2$ </td></tr><tr><td></td><td>C</td><td> $0 . 3 1 { \pm } 0 . 0 8$ </td><td> $0 . 3 2 { \pm } 0 . 0 7$ </td><td> $0 . 6 7 { \pm } 0 . 0 7$ </td></tr><tr><td>Two stage</td><td>M</td><td> $\mathbf { 0 . 1 5 \bot 0 . 0 5 }$ </td><td> $0 . 1 3 { \pm } 0 . 0 4$ </td><td> $\mathbf { 0 . 7 5 \bot 0 . 0 7 }$ </td></tr><tr><td>PatchCore [8]</td><td>M</td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>PaDiM [9]</td><td>M</td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>DRAEM [10]</td><td>M</td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>Rev. Distill. [11]</td><td>M</td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr></table>

## V. RESULTS AND DISCUSSION

a) Benchmark Task Performance: The missing component task (M) constitutes the primary challenge addressed in this work and differs fundamentally from conventional object detection. Instead of recognizing visible appearance patterns, models must infer the absence of a component at an expected location using weak structural cues such as exposed pads or silkscreen outlines. As shown in Table III, all one stage and transformer based detectors perform poorly on this task, with FNR values consistently above 0.87. YOLOv11 achieves the best one stage result (mAP 0.09, FNR 0.87), while RT-DETR and D-FINE perform substantially worse (FNR 0.92 and 0.93, respectively). Applying SAHI achieves the highest F1 on task (M), yet the FNR remains at 0.90, indicating that the limitation is not primarily related to scale or spatial resolution but to the inability of current detectors to represent structural absence.

![](images/dce976fc1afb8b024af86235bd444428cb3fa5ebd87eb82471857b6461a79081.jpg)  
Fig. 2. Confusion matrix for YOLOv11 on PCB-MC-M (5-fold, normalized). Rows correspond to ground truth classes and columns to predicted classes.

In contrast, the proposed two stage approach achieves the best overall performance on task (M), with the highest mAP (0.15) and lowest FNR (0.75). Although its F1 (0.13) remains comparable to one stage detectors, the reduction in missed detections is critical in industrial inspection, where false negatives correspond to undetected assembly failures. These results suggest that explicitly separating proposal generation from classification provides a more suitable inductive bias for absence based detection.

The remaining tasks, All Classes (A) and Components only (C), follow conventional detection assumptions and achieve substantially stronger performance. In task (A), YOLOv11 reaches the highest mAP (0.30), while YOLOv11+SAHI obtains the best F1 (0.47) and lowest FNR (0.50), demonstrating the benefit of tiled inference for proposal coverage. Task (C) produces the strongest results overall, with YOLOv11 reaching mAP 0.42 and F1 0.50. Despite these improvements, FNR values remain relatively high across methods (≈0.50–0.70), confirming that PCB inspection is challenging even under standard detection settings.

b) Anomaly Detection vs. Object Detection: Anomaly detection methods, including PatchCore, PaDiM, DRAEM, and Reverse Distillation, completely fail on the missing component task, producing zero detection performance and an FNR of 1.0. This behavior is expected, as anomaly detection methods assume consistent object structure across training and testing, whereas PCB-MC contains diverse board layouts without aligned references.

The failure can be attributed to a fundamental mismatch between the assumptions of anomaly detection and the structure of the PCB-MC task. These methods rely on learning the distribution of normal appearance and identifying deviations; however, missing components do not necessarily introduce strong local visual anomalies, especially in unaligned or layout variable settings. Instead, they require reasoning about expected object presence at specific spatial locations, which is not captured by appearance based anomaly modeling.

![](images/4061122b7ff576867d3e2084fdd09c4ad1e27691d94da2c41959fa8ac3f77d09.jpg)  
Fig. 3. Component level diagnosis of detection failure on Task M. Each group shows a cropped missing component footprint at the inference scale Columns: Ground truth (red dashed box), detector output (YOLOv11, conf=0.25), and stage one proposals (conf=0.05). Samples 1-3: missed due to no proposal coverage. Samples 4-6: Correctly detected due to proposal coverage. Color legend: red dashed = ground truth, gray = proposals, green = successful coverage.

In comparison, object detection approaches implicitly capture spatial and semantic priors by means of bounding box supervision. However, they still struggle with missing components because training data typically lacks explicit negative examples of “expected but absent” objects. The two stage method partially overcomes this limitation by decoupling localization and verification, enabling more robust reasoning about absence.

c) Category Level Analysis: Figure 2 presents the normalized confusion matrix for YOLOv11 on the task M. The dominant failure mode is not inter class confusion, but missed detections, where missing component instances are incorrectly assigned to the background class (i.e., no detection produced for the corresponding footprint region). Across most categories, over 85% of missing components are classified as background, with the error reaching 100% for small footprints such as LEDs.

In contrast, larger components with stronger structural cues, such as ICs, achieve comparatively higher detection rates, although the majority of instances are still missed (68.0% predicted as background). The confusion matrix also shows minimal confusion between missing component categories, indicating that once a candidate region is generated, classification is comparatively less problematic than localization.

These results suggest that the primary limitation lies in identifying candidate regions corresponding to missing component footprints rather than discriminating between component types. Overall, the findings reinforce that missing component detection is fundamentally an absence localization problem requiring stronger spatial reasoning and structural priors beyond conventional appearance based detection.

d) Quantitative Evaluation: The failure patterns in Figure 3 corroborate the results in Table III. In Samples 1-3, one stage detectors fail to generate candidate regions over missing component footprints, confirming that these areas are not recognized as relevant objects. When proposal coverage is present (Samples 4-6), detection succeeds, which explains the improved FNR of the two stage method. Failures occur despite clear structural cues, and SAHI provides limited improvement, reinforcing that the bottleneck is semantic rather than resolution related.

e) Discussion: Taken together, these results demonstrate that PCB inspection is not a single homogeneous problem, but rather a spectrum ranging from pure appearance detection (C) to structure aware reasoning under absence (M). Current object detectors perform well when the task aligns with their underlying assumptions (C), but degrade significantly when those assumptions are not met (M). The combined setting (A) further reveals that these limitations are not isolated, but interact in realistic scenarios.

## VI. CONCLUSION

We introduced PCB-MC, a dataset for missing component detection with footprint level annotations and unified evaluation settings.

Our results show that detecting missing components is significantly more challenging than standard object detection, with high false negative rates across modern detectors and complete failure of anomaly methods.

Further research will explore structure aware modeling, including spatial expectations, layout priors, or relational reasoning. We expect PCB-MC to serve as a benchmark to stimulate research in this direction in industrial inspection.

## REFERENCES

[1] Q. Ling and N. A. M. Isa, “Printed Circuit Board Defect Detection Methods Based on Image Processing, Machine Learning and Deep Learning: A Survey,” IEEE Access, vol. 11, pp. 15921–15944, 2023.

[2] F. Ciaglia, F. S. Zuppichini, P. Guerrie, M. McQuade, and J. Solawetz, “Roboflow 100: A rich, multi-domain object detection benchmark,” 2022.

[3] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics yolov8,” 2023.

[4] G. Jocher and J. Qiu, “Ultralytics yolo11,” 2024.

[5] G. Jocher and J. Qiu, “Ultralytics yolo26,” 2026.

[6] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, and J. Chen, “Detrs beat yolos on real-time object detection,” 2024.

[7] Y. Peng, H. Li, P. Wu, Y. Zhang, X. Sun, and F. Wu, “D-fine: Redefine regression task in detrs as fine-grained distribution refinement,” 2024.

[8] K. Roth, L. Pemula, J. Zepeda, B. Scholkopf, T. Brox, and P. Gehler, “Towards total recall in industrial anomaly detection,” in CVPR, 2022.

[9] T. Defard, A. Setkov, A. Loesch, and R. Audigier, “Padim: a patch distribution modeling framework for anomaly detection and localization,” in ICPR, 2021.

[10] V. Zavrtanik, M. Kristan, and D. Skocaj, “Draem – a discriminativelyˇ trained reconstruction embedding for surface anomaly detection,” 2021.

[11] H. Deng and X. Li, “Anomaly detection via reverse distillation from one-class embedding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 9727–9736, 2022.

[12] J. Liu, G. Xie, J. Wang, S. Li, C. Wang, F. Zheng, and Y. Jin, “Deep industrial image anomaly detection: A survey,” Mach. Intell. Res., vol. 21, no. 1, pp. 104–135, 2024.

[13] P. Bergmann, S. Lowe, M. Fauser, D. Sattlegger, and C. Steger, “Mvtec¨ ad – a comprehensive real-world dataset for unsupervised anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

[14] A. Suksukont, J. Onshaunjit, and J. Srinonchat, “A Deep Learning-Based System for Detecting Defects in Printed Circuit Boards,” in 13th International Electrical Engineering Congress (iEECON), pp. 1–6, IEEE, Mar. 2025.

[15] D. Demir, S. Birecik, F. Kurugollu, M. Sezgin, I. Bucak, B. Sankur, and E. Anarim, “Quality inspection in pcbs and smds using computer vision techniques,” in 20th Annual Conference of IEEE Industrial Electronics, vol. 2, pp. 857–861 vol.2, 1994.

[16] R. R., S. A., G. D., M. B., and A. S.Vaidya, “Quality Control of PCB using Image Processing,” International Journal of Computer Applications, vol. 141, pp. 28–32, May 2016.

[17] S. P. Chhetri, S. Bhat, P. Timalsina, and B. T. Magar, “Detection of Missing Component in PCB Using YOLO,” International Journal on Engineering Technology, vol. 1, pp. 62–71, Dec. 2023.

[18] K. Xia, Z. Lv, K. Liu, Z. Lu, C. Zhou, H. Zhu, and X. Chen, “Global contextual attention augmented YOLO with ConvMixer prediction heads for PCB surface defect detection,” Scientific Reports, vol. 13, p. 9805, June 2023.

[19] H. Tang, D. Bu, G. Mo, P. Lu, Z. Li, and L. Huang, “Lrt-detr: An enhanced transformer-based framework for pcb surface defect detection,” IEEE 6th International Seminar on Artificial Intelligence, Networking and Information Technology (AINIT), pp. 1–10, 2025.

[20] S. Ren, K. He, R. B. Girshick, and J. Sun, “Faster R-CNN: towards real-time object detection with region proposal networks,” CoRR, vol. abs/1506.01497, 2015.

[21] K. He, G. Gkioxari, P. Dollar, and R. B. Girshick, “Mask R-CNN,”´ CoRR, vol. abs/1703.06870, 2017.

[22] M. Calabrese, L. Agnusdei, G. Fontana, G. Papadia, and A. Del Prete, “Application of Mask R-CNN and YOLOv8 algorithms for defect detection in printed circuit board manufacturing,” Discover Applied Sciences, vol. 7, p. 257, Mar. 2025.

[23] B. Hu and J. Wang, “Detection of pcb surface defects with improved faster-rcnn and feature pyramid network,” IEEE Access, 2020.

[24] J. Luo, Z. Yang, S. Li, and Y. Wu, “Fpcb surface defect detection: A decoupled two-stage object detection framework,” IEEE Transactions on Instrumentation and Measurement, 2021.

[25] T. Kiobya, J. Zhou, B. Maiseli, and M. Khan, “Attentive context and semantic enhancement mechanism for printed circuit board defect detection with two-stage and multi-stage object detectors,” Scientific Reports, 2024.

[26] D. Makwana, S. C. T. R, and S. Mittal, “Pcbsegclassnet - a Light-Weight Network for Segmentation and Classification of Pcb Component,” SSRN Electronic Journal, 2022.

[27] H. Lu, D. Mehta, O. P. Paradis, N. Asadizanjani, M. M. Tehranipoor, and D. Woodard, “Fics-pcb: A multi-modal image dataset for automated printed circuit board visual inspection,” IACR Cryptol. ePrint Arch., vol. 2020, p. 366, 2020.

[28] G. Mahalingam, K. M. Gay, and K. Ricanek, “PCB-METAL: A PCB Image Dataset for Advanced Computer Vision Machine Learning Component Analysis,” in 16th International Conference on Machine Vision Applications (MVA), pp. 1–5, IEEE, May 2019.

[29] C. Pramerdorfer and M. Kampel, “A dataset for computer-vision-based PCB analysis,” in 14th IAPR International Conference on Machine Vision Applications (MVA), pp. 378–381, IEEE, May 2015.

[30] S. Tang, F. He, X. Huang, and J. Yang, “Online PCB Defect Detector On A New PCB Defect Dataset,” 2019. Version Number: 1.

[31] R. Ding, L. Dai, G. Li, and H. Liu, “TDD-net: a tiny defect detection network for printed circuit boards,” CAAI Transactions on Intelligence Technology, vol. 4, pp. 110–116, June 2019.

[32] S. Lv, B. Ouyang, Z. Deng, T. Liang, S. Jiang, K. Zhang, J. Chen, and Z. Li, “A dataset for deep learning based detection of printed circuit board surface defect,” Scientific Data, vol. 11, p. 811, July 2024.

[33] A.-D. Savu, N. Bizon, and S.-A. Dragusin, “Reference-Based Detection and Classification of Printed Circuit Boards Defects Using Deep Learning and Image Processing Techniques,” in 17th International Conference on Electronics, Computers and Artificial Intelligence (ECAI), pp. 1–10, IEEE, June 2025.

[34] F. C. Akyon, S. Onur Altinuc, and A. Temizel, “Slicing aided hyper inference and fine-tuning for small object detection,” in 2022 IEEE International Conference on Image Processing (ICIP), pp. 966–970, 2022.

[35] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” CoRR, vol. abs/1512.03385, 2015.

[36] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for´ dense object detection,” 2018.

[37] S. Akcay, D. Ameln, A. Vaidya, B. Lakshmanan, N. Ahber, and U. Genc, “Anomalib: A deep learning library for anomaly detection,” in IEEE International Conference on Image Processing (ICIP), 2022.

[38] T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick, and P. Dollar, “Microsoft coco:´ Common objects in context,” 2015.

[39] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” 2019.