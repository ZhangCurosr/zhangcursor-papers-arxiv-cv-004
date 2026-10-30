# Uncertainty-aware Coarse-to-Fine Main Pancreatic Duct Segmentation for Width Measurement in 3D MRCP

Zipei Zhang<sup>1,</sup> <sup>\*</sup>, Jaakko Sahlsten<sup>1</sup>, Mikko Saukkoriipi<sup>1</sup>, Johanna Ronkainen<sup>2,3</sup>, Mona Lehto<sup>2,3</sup>, Akseli Aittoniemi<sup>2,3</sup>, Irina Rinta-Kiikka<sup>2,3</sup>, and Kimmo Kaski<sup>1</sup>

<sup>1</sup>Department of Computer Science, Aalto University School of Science, Finland

<sup>2</sup>Department of Radiology, Tampere University Hospital, Wellbeing Services County of Pirkanmaa, Finland

<sup>3</sup>Faculty of Medicine and Health Technology, Tampere University, Finland

Corresponding author: zipei.zhang@aalto.fi

## ABSTRACT

Assessment of the main pancreatic duct (MPD) from volumetric magnetic resonance cholangiopancreatography (MRCP) scans is an essential task for ductal dilatation diagnosis and pancreatic cancer analysis. However, manual assessment is affected by reader variability in which automated analysis may assist. Nevertheless, fully automated and uncertainty-aware MPD analysis from volumetric MRCP remains underexplored area of research. In this study, we propose an automatic and uncertainty-aware deep learning-based pipeline for MPD segmentation and maximum-width analysis from MRCP scans. A coarse localisation model first identifies the ductal region-of-interest, followed by a fine model that segments the MPD within the region. The dataset consists of 256 MPD-segmented MRCP scans from 69 patients. A randomly selected holdout test subset of 14 patients with 55 MRCP volumes was used for evaluation. We employed a five-model cross-validation ensemble fo improved performance and uncertainty estimation. The best MPD segmentation performance was 0.634 DSC, 0.744 precision, and 0.593 recall. The mean absolute error for maximum MPD width prediction was 0.355 mm and dilated MPD (≥ 5 mm) was detected with 75.0% sensitivity and 74.4% specificity. These novel results suggest that automated MPD segmentation can be used to aid MRCP-based duct measurements.

## 1 Introduction

Pancreatic cancer, most commonly pancreatic ductal adenocarcinoma (PDAC), is one of the most aggressive cancers worldwide ranking seventh among cancer-related deaths, with incidence still increasing<sup>1</sup>. One reason for the poor prognosis is that more than 80 % of patients with PDAC are diagnosed in late stages which is why early detection is critical for improving prognosis, as tumors often become inoperable after infiltrating adjacent structures. Imaging plays a central role in identification of early disease, including small solid tumors and secondary morphological changes in the pancreatic parenchyma and main pancreatic duct (MPD). Computed tomography (CT) is the main imaging modality used in the diagnosis of PDAC, while magnetic resonance imaging (MRI) is used as a problem-solving imaging modality for cases that are not conclusive in CT.

The main reason for this is that MRI provides a higher sensitivity for detecting small (< 2 cm) isodense lesions compared to CT<sup>2</sup>. Another main advantage of MRI is the ability to obtain images of the ducts by magnetic resonance cholangiopancreatography (MRCP). MRCP imaging is based on heavily T2-weighted pulse sequences and thus highlights the natural differences in T2-weighted contrast between fluid-filled structures such as ducts and surrounding soft tissues<sup>3</sup>. The MRCP is one of the imaging sequences taken routinely during abdominal MRI<sup>4</sup>. In addition to an expansive tumour finding, imaging findings may be secondary<sup>5</sup>. The most common of these is dilatation of the main pancreatic duct<sup>6,</sup> <sup>7</sup>which can be associated with an obstructing tumour that causes focal stricture with dilatation upstream, and the stricture is usually abrupt in morphology and over 6 mm long in malignant cases<sup>8</sup>. Changes in MPD can be detected in MRCP images<sup>3</sup>.

In their study Toshima et al.<sup>7</sup> concluded that most patients with pancreatic ductal adenocarcinoma showed focal pancreatic abnormalities on CT performed at least 1 year before diagnosis. The focal change in MPD diameter exhibited the shortest duration from its development to subsequent diagnosis<sup>7</sup>. Hence, it is important to develop tools to maximise the acquisition of information about the morphology of the main duct. Moreover, the volume and diameter of MPD as features have been shown to improve the classification performance of pancreatic ductal adenocarcinoma compared to using only radiomics features<sup>9</sup>.

Deep learning has been extensively applied in the field of medical image analysis, including a variety of tasks such as semantic segmentation, image to image enhancement, and classification. In medical image segmentation, U-net<sup>10</sup>, i.e., the encoder-decoder architecture with skip connections, has become the standard model architecture. For volumetric segmentation,

3D variants such as 3D U-Net has been proposed<sup>11</sup>. More recently, complex variants of U-net have been proposed, such as nnU-Net<sup>12</sup>, which include fully automated preprocessing pipelines and automated architectural choice, such as number of layers and features, and Swin-UNETR<sup>13</sup> that are still based on the encoder-decoder approach but instead of convolutional layers, it utilises shifted window approach and transformer blocks.

The segmentation of the main pancreatic duct using deep learning approaches has previously been studied using CT-based imaging. In the pancreatic domain, so far most automatic analysis of the main pancreatic duct (MPD) has been developed on CT rather than MRCP. Abi Nader et al.<sup>14</sup> proposed a deep learning pipeline on portal venous CT for pancreatic lesion detection and MPD dilatation assessment. Their method first segmented the pancreas, lesions, and visible MPD using nnU-Net, and then derived imaging features including MPD diameters in the pancreatic head, body, and tail. In a large multi-institutional cohort, they reported a mean MPD Dice score of 0.58 ± 0.37, showing both the feasibility and difficulty of automatic duct segmentation. More recently, Tan et al.<sup>15</sup> studied MPD segmentation on photon-counting CT and reported an average Dice score of 0.621 using nnU-Net. In addition, Ambo et al.<sup>16</sup> proposed an automated CT-based pipeline for MPD dilatation detection that combined pancreas segmentation, MPD segmentation, centreline estimation, and cross-sectional shape analysis. Together, these studies show that automatic duct analysis is possible, but the literature remains centred on CT. MPD segmentation has been studied with high resolution, high contrast, and large field-of-view Photon-Counting CT scans using nnU-Net, which was trained using 53 scans and evaluated on 13 scans, resulting in average DSC of 0.621<sup>15</sup>.

In contrast to studies with CT, there are few studies on MPD analysis based on MRCP volumes, largely due to limited availability of 3D MRCP datasets with expert annotation. However, MRCP is clinically important for MPD evaluation due to its high contrast for fluid-filled structures without ionizing radiation. Existing MRCP-based studies in pancreatic imaging are more commonly focused on detection or characterisation of pancreatic cystic lesions, or they rely on manual or semi-automatic duct segmentation, rather than delivering a fully automated 3D MPD segmentation tool<sup>17</sup>. Beyer et al.<sup>18</sup> established age-dependent reference values for the diameter of the pancreatic duct in MRCP in a large population dataset, showing that the diameter of the duct increases with age and that conventional upper limits may overestimate abnormality in healthy subjects. Ikezawa et al.<sup>19</sup> studied the correlation between MPD diameter measurements on special pancreatic ultrasonography and MRCP, again treating MRCP as a measurement reference rather than as an input to automated image analysis. Similarly, Zhang et al.<sup>20</sup> analysed MPD diameter on MRI to identify malignant intraductal papillary mucinous neoplasm, which reinforces the clinical importance of MPD width as a biomarker. Still there is no research work on the fully automated MPD segmentation-width analysis pipeline, the lack of which motivates the need for data-driven MRCP analysis approaches that explicitly address automated MPD width categorized classification.

## 2 Methods

In this study, we evaluated whether automatic segmentation can support quantitative maximum-width analysis of the main pancreatic duct from 3D MRCP volumes. We adopted established 3D encoder-decoder segmentation models in a task-specific coarse-to-fine pipeline designed for a thin tubular structure. Moreover, uncertainty of the model is quantified, analysed, and compared against its performance to estimate the utility of model uncertainty. The overall workflow consisted of three steps: (1) full-volume coarse localisation of the MPD region, (2) region-of-interest (ROI) extraction and detailed segmentation using a fine model, and (3) uncertainty estimation using five cross-validation models.

## 2.1 Dataset, annotation and preprocessing

In this retrospective study, we used 256 MRCP volumes of 69 patients from Tampere University Hospital pancreatic cyst registry. The study was conducted in accordance with the Declaration of Helsinki, it was approved and the need for patients’ informed consent was waived by the Institutional Review Board of the Wellbeing Services County of Pirkanmaa, Finland (reference approval number: R24295). The patient records were retrospectively collected and pseudonymised by the data-providing authority prior to transfer to the research environment. All analyses were performed within a secure and access-controlled Aalto University’s SECDATA computing environment compliant with national data protection policies and the General Data Protection Regulation (GDPR)<sup>21</sup>.

Annotation procedure Expert voxel-wise MPD segmentations were created in 3D Slicer (version 5.6.2)<sup>22</sup> primarily by two clinical researchers and after that, two experienced abdominal radiologists with over 20 years of MRI experience reviewed all the segmentation annotations. The annotation protocol focused on the visible main pancreatic duct in the source MRCP volume. Cases with uncertain visibility were reviewed jointly by the two radiologists to improve annotation consistency.

Image preprocessing All MRCP volumes were resampled to a voxel spacing of $0 . 9 \times 0 . 6 2 5 \times 0 . 6 2 5 \mathrm { { m m } ^ { 3 } }$ and intensitynormalised on a per-volume basis before training.

Dataset partitioning The dataset was divided at the patient level in the development and test sets with a 80:20 ratio, resulting in 201 and 55 scans, respectively. The characteristics of the dataset are shown in Table 1. In model training, the development set was further divided into 5-folds, where four folds were used for the training subset and one fold as the validation data for model selection. The test dataset consisted of 55 scans from 14 patients having 1 to 7 scans each, and was used to evaluate the developed methods. In particular, for patients with multiple MRCP scans the test dataset, the mean interval between the earliest and latest scan was 2.89 years, with a standard deviation of 1.47 years. In the patients with multiple scans, each of the scans was taken at different time points.

Table 1. Characteristics of Development and Test Datasets. In our dataset, main pancreatic duct (MPD) dilation is caused by MPD Intraductal Papillary Mucinous Neoplasm (IPMN) or mix type IPMN.
<table><tr><td>Characteristics</td><td>Development Dataset</td><td>Test Dataset</td></tr><tr><td>Number of MRCP volumes</td><td>201</td><td>55</td></tr><tr><td>Patient demographics</td><td></td><td></td></tr><tr><td>Number of individuals</td><td>55</td><td>14</td></tr><tr><td>Age, mean (SD), years</td><td>72.4 (9.3)</td><td>75.6 (7.9)</td></tr><tr><td>Male : Female</td><td>44 (80.0%) : 11 (20.0%) 5 (35.7%) : 9 (64.3%)</td><td></td></tr><tr><td>Normal MPD</td><td>44 (80.0%)</td><td>10 (71.4%)</td></tr><tr><td>Dilated MPD due to mixed type IPMN</td><td>10 (18.1%)</td><td>3 (21.4%)</td></tr><tr><td>Dilated MPD due to MPD IPMN</td><td>1 (1.8%)</td><td>1 (7.1%)</td></tr></table>

Each input MRCP volume was denoted by $\boldsymbol { x } \in \mathbb { R } ^ { 1 \times D \times H \times W }$ , where D, H, and W denoted the depth, height, and width of the 3D MRCP volume. The corresponding expert annotation is a binary label map $y \in \{ 0 , 1 \} ^ { D \times H \times \hat { W } }$ , where $y ( \nu ) = 1$ indicates MPD and $y ( \nu ) = 0$ indicates background. The MPD is a tubular structure with varying width, and in terms of the imaged MRCP volume, only consists of a small fraction of the full MRCP volume (0.019%). Therefore, a direct segmentation approach is affected by a severe class imbalance. Instead, we adopt a coarse-to-fine strategy: a coarse model first localises a high-recall candidate MPD region, and a fine model then performs detailed segmentation inside the cropped full resolution ROI.

## 2.2 Segmentation models

We evaluated established 3D segmentation models rather than proposing a new backbone architecture. The purpose of the model comparison was to identify a reliable and feasible model for MPD segmentation and downstream maximum-width estimation from 3D MRCP. The final selected model used a 3D U-Net as the coarse localisation model and a 3D SegResNet as the fine segmentation model.

Model selection criteria include: (1) practical training and inference GPU memory demand, (2) stable convergence on the collected dataset, (3) segmentation performance on validation set.

3D U-Net A $3 \mathrm { { D } U \mathrm { { - N e t } ^ { 1 1 } } }$ was used as a convolutional encoder–decoder baseline from MONAI, and selected as the coarse localisation model with full 3D MRCP image volumes as input. The model was implemented using the MONAI UNet module<sup>23</sup> and the input and output channels were adapted to single-channel MRCP input and binary MPD segmentation.

Swin UNETR Swin $\mathrm { U N E T R } ^ { 1 3 }$ was used as a transformer-based volumetric segmentation baseline from MONAI with SwinUNETR backbone from MONAI<sup>23</sup> . We used the sliding window input shape of $6 4 \times 6 4 \times 6 4$ , embedding dimension 12, depths (1, 1, 2, 1), and attention heads (3, 6, 12, 24). The input and output channels were adapted to single-channel MRCP input and binary MPD segmentation. No task-specific architectural modifications were introduced.

DynUNet DynUNet<sup>23</sup> is a dynamic UNet model heavily based on nnU-Net<sup>12</sup>, which was selected as a self-configuring medical image segmentation pipeline. We used DynUNet from MONAI with the 3D full-resolution configuration. The framework automatically determined the patch size, batch size, network topology, and training schedule from the training data.

Proposed coarse-to-fine pipeline The proposed pipeline included two stages, where the first stage includes a coarse 3D U-Net segmentation model that receives a full MRCP volume at four-times downsampled spatial resolution with the goal of approximate localisation of the MPD duct with near-perfect recall. This additional stage is important because the MPD occupies a very small fraction of the full image volume, making direct full-resolution segmentation inefficient and highly imbalanced. The data sample is defined as: $x ^ { c } = \mathcal { D } ( x )$ , where $\mathbf { \Psi } _ { X } ^ { c } \in \mathbb { R } ^ { 1 \times D _ { c } \times H _ { c } \times W _ { c } }$ and $\boldsymbol { p } ^ { c } = f _ { \theta _ { c } } ( \boldsymbol { x } ^ { c } )$ , and the prediction is defined as: $p ^ { c } \in [ 0 , 1 ] ^ { 1 \times D _ { c } \times H _ { c } \times W _ { c } }$ . Given a downsampled MRCP volume, the coarse 3D U-Net predicts a binary MPD segmentation map over the whole 3D image volume. The segmentation prediction on the downsampled 3D image volume is then used to crop the region of interest, defined as: $x _ { \mathrm { r o i } } = \mathcal { C } ( x , b )$ , where $x _ { \mathrm { r o i } } \in \mathbb { R } ^ { 1 \times d \times h \times w }$ . To avoid missing potential duct voxels during the ROI cropping stage, the ROI is spatially expanded by a fixed safety margin in all directions. In practice, we set the safety margin to 10. The resulting bounding box is used to crop a full-resolution ROI from the original MRCP volume ROI expanded with a safety margin. In this way, the fine model receives an input that is much smaller than the full scan but still contains the full MPD region with sufficient surrounding context.

In the second stage, a fine-resolution 3D SegResNet was applied to the cropped ROI. Since the search space had already been reduced by the coarse localisation stage, the fine model could focus on detailed MPD boundary delineation instead of global duct detection. This improved the effective foreground-to-background ratio and allowed the fine model to use higher spatial resolution around the duct. Final prediction is defined as: $p ^ { f } = f _ { \theta _ { f } } ( x _ { \mathrm { r o i } } )$ , where $p ^ { f } \in [ 0 , 1 ] ^ { 1 \times d \times h \times w }$ . The fine model predicts the final binary MPD segmentation inside the cropped ROI. During inference, this ROI-level prediction is mapped back to the original image space using the crop coordinates obtained from the coarse stage, yielding the final full-volume segmentation result.

## 2.3 Implementation details

Both the coarse and fine segmentation models were trained using a combination of weighted cross-entropy and Dice loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { C E } } \mathcal { L } _ { \mathrm { C E } } ( p , y ) + \lambda _ { \mathrm { D i c e } } \mathcal { L } _ { \mathrm { D i c e } } ( p , y ) , } \end{array}\tag{1}
$$

in which the weighted cross-entropy loss is defined as

$$
\mathcal { L } _ { \mathrm { C E } } = - \sum _ { \nu } \sum _ { k \in \{ 0 , 1 \} } w _ { k } y _ { k } ( \nu ) \log p _ { k } ( \nu ) ,\tag{2}
$$

where $w _ { k }$ is the class weight for class k. The MPD class was assigned a larger weight because MPD voxels are much rarer than background voxels. The Dice loss for the MPD class is

$$
\mathcal { L } _ { \mathrm { D i c e } } = 1 - \frac { 2 \sum _ { \nu } p _ { 1 } ( \nu ) y _ { 1 } ( \nu ) + \varepsilon } { \sum _ { \nu } p _ { 1 } ( \nu ) + \sum _ { \nu } y _ { 1 } ( \nu ) + \varepsilon } ,\tag{3}
$$

where $p _ { 1 } ( \nu )$ and $y _ { 1 } ( \nu )$ denote the predicted probability and ground-truth label for MPD at voxel v, and ε is a small constant for numerical stability.

During training, we applied multiple 3D data augmentations, including random flips, rotations, intensity scaling, intensity shifting, and additive Gaussian noise. All experiments were performed on NVIDIA H200 GPUs with 141 GB memory.

Five-fold cross-validation ensemble To obtain the final segmentation prediction and enable uncertainty estimation, we trained a five-model ensemble using five-fold cross-validation at the patient-level on the training set. The test set was kept separate and was not used during model training. During inference on the test set, each of the five model pairs was applied independently to every test case. The five probability maps were then averaged to form the ensemble prediction as follows:

$$
\bar { p } ( \nu ) = \frac { 1 } { 5 } \sum _ { j = 1 } ^ { 5 } p ^ { ( j ) } ( \nu ) ,\tag{4}
$$

where $p ^ { ( j ) } ( \nu )$ is the MPD probability predicted by the $j \cdot$ -th model at voxel v. The final segmentation was calculated by thresholding $\bar { p } ( \nu )$ . In addition, the predictions of five models were further used in the uncertainty analysis described next. The final ensemble segmentation is obtained by thresholding the ensemble mean:

$$
\hat { y } _ { \mathrm { e n s } } ( \nu ) = { \bf 1 } \big ( \bar { p } ( \nu ) \geq 0 . 5 \big ) .\tag{5}
$$

Uncertainty estimation The uncertainty of deep learning models were used to assess their utility by analysing the relationship and correlation between the uncertainty and the performance of the model. To improve model calibration, we trained an ensemble of five coarse-to-fine model pairs using five-fold cross-validation on the development set. In each fold, four folds were used for model training and one fold was used for model validation. Model performance on unseen data was evaluated using a test that was not used during any of the model development. During inference, the five trained coarse-to-fine models were applied independently to each test case producing five independent prediction masks. The prediction masks were then aggregated to produce the final segmentation and voxel-wise uncertainty as shown next.

The uncertainty of the model at the voxel v can be estimated using the predictive entropy of the ensemble prediction:

$$
H ( \nu ) = - \bar { p } ( \nu ) \log \left( \bar { p } ( \nu ) + \varepsilon \right) - \left( 1 - \bar { p } ( \nu ) \right) \log \left( 1 - \bar { p } ( \nu ) + \varepsilon \right) .\tag{6}
$$

We evaluated the uncertainty using volume-level aggregation with three commonly applied measures derived only from model output. A straightforward way to produce a volume-level uncertainty, is to take average of all voxel-level predictive entropy values:

$$
H ( V ) = \sum _ { \nu \in V } { \frac { H ( \nu ) } { | V | } } ,\tag{7}
$$

where $| V |$ is the total count of voxels of volume V. Another aggregation measure only inspects the positively predicted voxels with so called structure-based predictive entropy:

$$
S H ( V ) = \frac { \sum _ { \nu \in V } \hat { y } _ { \mathrm { e n s } } ( \nu ) \cdot H ( \nu ) } { \sum _ { \nu \in V } \hat { y } _ { \mathrm { e n s } } ( \nu ) } .\tag{8}
$$

A third commonly used model uncertainty estimate is the coefficient of variation measure, which is defined by the dispersion and mean value of model predictions in the ensemble as follows:

$$
C V ( V ) = \frac { \sigma _ { g } } { \mu _ { g } } = \frac { \sqrt { \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left( \left| V _ { m } \right| - \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left| V _ { m } \right| \right) ^ { 2 } } } { \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left| V _ { m } \right| } ,\tag{9}
$$

where $\textstyle | V _ { m } | = \sum _ { \nu \in V } \mathbf { 1 } { \bigl ( } p _ { m } ( \nu ) \geq 0 . 5 { \bigr ) }$ for model $m \in M$

Post-processing To improve robustness, a connected-component analysis is applied to the final prediction mask to remove small and isolated false-positive regions that are not connected to the largest component that is expected to be the MPD. This post-processing step reduces orphan positive voxels that are more likely to be false positive, thus improving the performance of downstream MPD maximum-width analysis.

## 2.4 Evaluation metrics

The performance of the model was evaluated in terms of the segmentation performance using volumetric similarity and surface distance measures. In addition, two abdominal radiologists performed a qualitative evaluation of segmentation performance as well as image quality and noted clinically relevant findings. Moreover, in this section we defined how the uncertainty of the model was quantified.

Segmentation evaluation The Dice similarity coefficient, precision and recall were defined as:

$$
\mathrm { D S C } = { \frac { 2 \mathrm { T P } } { 2 \mathrm { T P } + \mathrm { F P } + \mathrm { F N } } } , \qquad \mathrm { P r e c i s i o n } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } } , \qquad \mathrm { R e c a l l } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } } ,\tag{10}
$$

where TP, FP, and FN denote true-positive, false-positive, and false-negative MPD voxels, respectively. The surface distance was evaluated using average symmetric surface distance (ASSD) and 95th percentile Hausdorff distance (HD95). Let $S _ { \mathrm { p r e d } }$ and $S _ { \mathrm { r e f } }$ denote the predicted mask and ground truth mask, and let $\begin{array} { r } { d ( a , S ) = \operatorname* { m i n } _ { b \in S } \| a - b \| _ { 2 } } \end{array}$ show the shortest distance between a point a and a surface S. ASSD was defined as:

$$
\mathrm { A S S D } = \frac { \sum _ { a \in S _ { \mathrm { p r e d } } } d ( a , S _ { \mathrm { r e f } } ) + \sum _ { b \in S _ { \mathrm { r e f } } } d ( b , S _ { \mathrm { p r e d } } ) } { | S _ { \mathrm { p r e d } } | + | S _ { \mathrm { r e f } } | } ,\tag{11}
$$

and HD95 was defined as the 95th percentile of the pooled bidirectional surface distances:

$$
\mathrm { H D 9 5 } = P _ { 9 5 } \left( \{ d ( a , S _ { \mathrm { r e f } } ) : a \in S _ { \mathrm { p r e d } } \} \cup \{ d ( b , S _ { \mathrm { p r e d } } ) : b \in S _ { \mathrm { r e f } } \} \right) .\tag{12}
$$

Qualitative evaluation The test was independently reviewed by two abdominal radiologists. For each case, the readers recorded the maximum MPD width from the reference segmentation and from the model prediction. The readers also assigned a visual quality score to the prediction of the model and to the original MRCP images using a five-point Likert scale and provided reasoning for the grading. The Likert for model prediction are as follows: 1: Poor (not clinically usable at all), 2: Low, some usable parts mostly not usable (large errors, fine details do not matter), 3: Medium, many usable parts but some parts problematic (fine details missing), 4: Good, almost fully usable, some minor issues, 5: Perfect, fully usable, no major or minor issues. The Likert for the MRCP image quality are defined as follows: 1. Very poor, 2. Poor, 3. Fair, 4. Good, 5. Excellent.

Model performance in terms of the max MPD width was assessed by averaging the two readers’ measurements into a consensus and evaluated with mean absolute error (MAE) between the ground truth and model prediction. We further evaluated whether the model-derived maximum MPD width could support binary categorisation of duct dilatation. The cases were divided into two groups using a maximum-width threshold of 5mm: non-dilated MPD (< 5mm) and dilated MPD (≥ 5mm). using MAE and accuracy metrics.

Uncertainty evaluation The utility of the model uncertainty by the model was evaluated using referral-based analysis and correlation analysis. In the referral-based analysis, each sample is ordered by an uncertainty measure and then removed from the dataset for expert referral. The mean qualitative and quantitative performances of the remainder is then calculated, producing a lineplot of expected fully-automated performance if a proportion of the data is referred. We evaluate the referral from 100% to 30%, as the noise in the results at the lower count of samples increases considerably. In correlation analysis, quantitative measures are directly compared to an uncertainty measure in a scatterplot, and a linear regression model is fitted to show the correlation coefficient and statistical significance. In addition, two radiologists’ mean values for each of the quantitative measures are evaluated against an uncertainty measure using boxplots for each of their ordinal categories, which can be visually evaluated for correlation. Moreover, we calculate the Pearson correlation coefficient (r) when comparing the uncertainty measure and the segmentation metrics, and the Spearman rank correlation coefficient (ρ), when comparing the uncertainty measure and qualitative metrics.

## 3 Results

## 3.1 Segmentation performance

The segmentation performance was evaluated and reported for both coarse and fine models with a more detailed analysis of the final segmentation performance of the fine models.

Coarse model 3D U-Net achieved near-perfect ROI recall as the coarse model, which means that nearly all of the reference MPD voxels were included within the cropped region of interest. The ROI cropping reduced the spatial size of the input for the fine segmentation stage by approximately four-fold compared with the full MRCP volume, while preserving basically full MPD coverage.

Pipeline Within the coarse-to-fine framework, three fine-stage segmentation models were evaluated: 3D U-Net, Swin UNETR, and 3D SegResNet, in addition to which the DynUNet was included as a one-stage full-volume reference pipeline. Table 2 summarised the MPD segmentation performance on the test set with all four evaluated pipelines. The 3D U-Net + 3D SegResNet approach turned out to have the highest Dice similarity coefficient, precision, recall among the compared methods. These results suggest that ROI-based coarse-to-fine segmentation pipeline is well suited for MPD segmentation due to the fact the duct is a small tubular structure that occupies only a small fraction of the full MRCP volume.

Table 2. Main pancreatic duct segmentation performance on the test set volumes (N=55). Best results are shown in bold.
<table><tr><td>Method</td><td>DSC ↑</td><td>Precision ↑</td><td>Recall ↑</td><td>ASSD (mm) ↓</td><td>HD95 (mm) ↓</td><td>Zero prediction mask ↓</td></tr><tr><td>DynUNet</td><td>0.579</td><td>0.613</td><td>0.583</td><td>3.50</td><td>23.7</td><td>0 (0.00%)</td></tr><tr><td>3D U-Net + 3D U-Net</td><td>0.629</td><td>0.717</td><td>0.585</td><td>4.65</td><td>32.0</td><td>0 (0.00%)</td></tr><tr><td>3D U-Net + Swin UNETR</td><td>0.516</td><td>0.634</td><td>0.479</td><td>11.7</td><td>45.5</td><td>1 (1.82%)</td></tr><tr><td>3D U-Net + 3D SegResNet</td><td>0.634</td><td>0.744</td><td>0.593</td><td>4.18</td><td>22.9</td><td>1 (1.82%)</td></tr></table>

## 3.2 Qualitative evaluation

In this subsection, we report all the qualitative evaluation measures.

Max MPD Width Measurement Performance The MPD Max width was qualitatively measured from the segmentation model prediction and ground truth by the two radiologists using 3D Slicer. The MPD was considered dilated if the maximum width is ≥ 5mm. The consensus of the two radiologist was defined by taking the mean of their evaluated width values. In terms of model performance compared to ground truth, the mean absolute error was 0.355 mm. This resulted in a dilated MPD accuracy of 74.5%, with a sensitivity of 75.0%, a specificity of 74.4%, a precision of 45.0% and an F1-score of 56.3%. For the ground truth annotations, the inter-rater agreement between the two radiologists was 0.618 mm MAE and the accuracy of the classification of DMPD was 83.6%.

Qualitative vs. Quantitative metrics In the Supplementary, we present a comparison of qualitative Likert measures for model segmentation and MRCP image quality against quantitative metrics (DSC, ASSD, HD95). In addition, the reasons for Likert grading based on radiologists’ assessment are summarised in the Supplementary.

## 3.3 Model Uncertainty Analysis

In the referral-based evaluation of the model uncertainty, we measured the performance using quantitative metrics of DSC, ASSD, and HD95. When examining the change in the DSC values at 100% and 30% data remaining using the coefficient of variation (CV) uncertainty measure to refer to the most uncertain cases, the results turn out to be as follows: DynUNet achieved 0.579 and 0.594; 3D U-Net + 3D U-Net achieved 0.629 and 0.690; 3D U-Net + Swin UNETR achieved 0.516 and 0.661; and 3D U-Net + 3D SegResNet achieved 0.634 and 0.730. Focusing only on the best-performing model (3D U-Net + 3D SegResNet) and evaluating referral performance with the three uncertainty measures (CV, SH, and H), the initial performance at 100% data was 0.634 DSC. The mean DSC at 60% data remaining was 0.718 (CV), 0.699 (SH), and 0.644 (H). At 30% data remaining, the DSC values were 0.730 (CV), 0.677 (SH), and 0.665 (H). The complete evaluation is shown in Figure 1.

   
![](images/f4b200259239b73abb48424d93fc023433f566272ce66924132c2db250c91ed0.jpg)

  
![](images/d1f8a842d02d8dc42c0a625b60a8e91e39c0d6cb14ce656b1c1245bb73267ed6.jpg)

-  
![](images/14584690fedaa6cd7adb04c1b30e23bee33297b2ba421c42649fb862e31ec9b5.jpg)

  
![](images/88e0518e83aa108d65302c9b4d6ec569b01066ae021f92f20d609b1fb890ced9.jpg)

  
![](images/2b9d82abd8b0bd3398b07f46e6bf2035353cf36678f1e1d48b2309f8e10b27ad.jpg)

  
![](images/d0f63cab007751c3f265fd506d552310e3b4d99dfb5bc2c175317a741cc505fb.jpg)  
Figure 1. Model performance across uncertainty threshold on referral based approach. The most uncertain cases are referred and average performance on remaining cases is plotted. In subfigures A. to C. the pipelines are compared using the best-performing uncertainty measure coefficient of variation (CV), and in subfigures D. to F. the three uncertainty measures predictive entropy (H), structure-based predictive entropy (SH), and coefficient of variation (CV), are compared using the best performing model i.e., 3D U-Net + 3D SegResNet. Note that evaluation for two of the pipelines start at 98% data remaining on the distance based measures due to zero prediction mask.

In terms of correlation analysis, fitting a linear model to the CV uncertainty measure against quantitative metrics of DSC, ASSD, and HD95 showed statistically significant correlation $( p < 0 . 0 0 1 )$ with strong coefficient values of $r = - 0 . 7 5 9$ $r = 0 . 6 9 3$ and $r = 0 . 6 9 5$ , respectively. When the CV was compared with qualitative measures of model performance (MPD Max Width MAE [mm] and Likert scales), the correlation coefficient $\rho = 0 . 1 8 7 ( p = 0 . 1 7 3 ) , \rho = - 0 . 0 8 3 ( p = 0 . 5 4 7 ) , \rho = - 0 . 7 2 8$ $( p < 0 . 0 0 1 )$ ) for MPD Max Width MAE, MRCP image quality Likert, and model segmentation quality Likert, respectively. Specifically, in MPD max width MAE, the mean CV values were 0.18, 0.19, 0.24, 1.07, and 0.57 for the MAE values of 0, 0.5, 1.0, 1.5, and 3.5 mm, respectively. The MRCP image quality Likert did not strongly correlate with CV, having mean values of 0.27, 0.27, 0.25, 0.23, 0.20, 0.29, 0.11, and 0.22 for Likert scores of 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, and 5.0 respectively. The model segmentation quality Likert scale showed a stronger relationship with CV values of 0.74, 0.24, 0.31, 0.16, 0.16, 0.09, and 0.06 for Likert scores of 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, and 4.0, respectively. Complete analysis is shown in Figure 2.

## 3.4 Within-patient analysis

The best performing 3D U-Net + 3D SegResNet model and the best performing CV uncertainty measure based on scan-level analysis were inspected in more detail in terms of within-patient analysis. In this analysis, the patient-level variability is analysed for segmentation, qualitative and uncertainty estimation. In terms of variability, the mean and standard deviation

   
![](images/dd694493f4c98a71d4caa53b4a43b9eea9698dec068dd1780f61918b43ff762d.jpg)

![](images/a54fe21236e653ad7a931178aa0a82dfb7fe33b6c129f6cd5f04a036880432dc.jpg)

-  
![](images/5ba1958b85af13f75b59a8d275755d351562339f88c4c7d319e7aae11239b0c0.jpg)

  
![](images/c795be377a1cfc0f3d77928d0a778ff1edc37ec9a5de08828a003ccd8a089bbe.jpg)

  
![](images/e5a76321f573a55d7331c1c6f4aa60313fd979615c403a8e0888c6178d8eb708.jpg)

  
![](images/4ab85ad0b48bf8e156829407c67585da14271aa824c447da96f493d880d12837.jpg)  
Figure 2. Correlation analysis between uncertainty measure and model performance. Subfigures A. to C. are scatterplots with fitted linear line between quantitative measures and coefficient of variation (CV) and subfigures D. to F. are boxplots with ordinal qualitative measures and CV.

distance from the patient mean was 0.05 (0.04) with DSC, 1.53 (1.55) with ASSD, 9.69 (8.16) with HD95, and 0.16 (0.32) with MPD Max Width MAE, 0.24 (0.21) with model prediction Likert quality, and 0.09 (0.08) with CV. The results are visualised in boxplots for each patient, and patients are orderedby median DSC performance. The complete results are shown in Figure 3.

![](images/5a6d3e77655c84acc078c884f81df39fb6fa4485ed370e3fb00eb0d76f61794a.jpg)

![](images/53d868bdfb9f59baed1c69199acf9621502809d5410983022808f13e27a5f9c1.jpg)

![](images/955440cb6efd00a03ee7c3ff9f3fd5efc6b7cc0348add9e6b2bc8be6b02a2df2.jpg)

![](images/6f4599b1eff61e5572c070b88af02bd9a959c14e1e2bb4e24549dfe5594de4fb.jpg)

![](images/4c3f60a9046f31d8a376f6f56dd87becc4afd513d0cc15a3052739340f146793.jpg)

![](images/eb48fa25b9ebf7570891dd7337f0939d20120762109e9619c5e0874515f8aa8b.jpg)  
Figure 3. Detailed look at 3D U-Net + 3D SegResNet performance within each test-set patient ordered ascending by median DSC performance. Segmentation performance is shown on subfigures A. to C. while subfigures D. and E. show qualitative measures by radiologists. Subfigure F. shows the coefficient of variation uncertainty measure.

## 4 Discussion

In terms of segmentation performance, the 3D U-Net + 3D SegResNet approach outperformed all other methods due to its residual convolutional feature extraction mechanism, which can provide stable optimization and better preservation of local ductal boundary information on a limited dataset. However, our approach did produce a single empty prediction mask and had the second lowest ASSD value of the compared methods. The only transformer based approach, i.e., Swin UNETR, had the lowest performance out of all compared methods. This can likely be attributed to the relatively small amount of data for the transformer-based approach.

In the downstream task of MPD max width estimation and dilated MPD classification from deep learning-based segmentation masks, the automatic approach achieved good performance, though still with lower value than the inter-rater agreement, which is consistent with the segmentation metrics thus having room for improvement. The sensitivity indicates that most of the dilated cases were detected, while the relatively low precision reflects false-positive dilatation predictions. These results suggest that the model-derived maximum width may be useful for screening or prioritisation, but false-positive cases would still require review by a radiologist.

In qualitative analysis, the mean visual evaluation of the clinical usability of the prediction of the model using the Likert grade was 2.67 by the first radiologist and 2.50 by the second radiologists. This indicates that the predicted segmentations were often imperfect as complete anatomical masks. However, the sub-millimeter maximum-width error suggests that the method may still be useful as a measurement-support tool, especially when the intended endpoint is maximum MPD width rather than complete duct delineation. An important finding was that the prediction errors did not exhibit a readily identifiable pattern. In a single case, the model failed to detect MPD completely, producing a clinically misleading measurement, and it occasionally produced subtle and visually conspicuous ductal abnormalities, and some predictions extended outside the ductal lumen. This suggests that the performance of the model was not solely determined by the apparent complexity of the finding. One possible explanation is A shift in data distribution between training and validation datasets, including variations in anatomy, imaging protocols, or image quality, which may have affected model generalisability. This heterogeneity of failure modes complicates the identification of a single technical explanation and highlights the need for further investigation into the factors that could affect model robustness. Moreover, when comparing both segmentation quality and MRCP image quality Likert values to model performance, a visual relationship is evident for segmentation measures, but less clear for MPD max width estimation.

In the uncertainty analysis, UNet 3D + SegResNet3D had the highest performance in terms of referral-based analysis. In terms of uncertainty measures, the coefficient of variation measure outperformed the two entropy-based uncertainty measures, which is consistent with previous work in medical image segmentation<sup>24</sup>. The correlation analysis demonstrated that uncertainty can estimate segmentation quality. The boxplots show visual ordinal relationships between CV and qualitative measures. In terms of DMPD classification, the model had 74.5% accuracy, which is lower but relatively close to the inter-observer variability, where the raters had 83.6% agreement, setting a realistic upper bound for the model performance.

There are limitations in our study related to the number of cases studied and the tools for qualitative measurement. The number of individual patients and the number of image volumes were relatively low; there are 14 patients with 55 scans included in the test set. However, the average duration between scans was nearly three years, providing more heterogeneous cases even from the same patients with generalizability. In addition, the data were from a single hospital, which may reduce the generalizability of the results. The reference measurements were obtained using the native measurement tool provided by 3D Slicer. As measurement markers partially obscure the duct margins during placement, exact identification of the measurement endpoints is inherently limited by image resolution and observer judgment. Therefore, a small degree of measurement uncertainty should be assumed when interpreting the reported distances. This partially explains the disagreement between the two raters.

Future work should focus on external validation across institutions and hospitals, as the current study is based on a single hospital. Furthermore, the collected dataset should scale up to improve generalizability. Methodologically, future models should improve MPD continuity, reduce zero-prediction cases, and better preserve the MPD morphology. In addition, more robust automatic width-estimation methods, such as orthogonal cross-sectional analysis, is expected to reduce dependence on manual expert measurement.

This study demonstrates the feasibility of automated segmentation of the main pancreatic duct from 3D MRCP using convolutional neural networks. Although segmentation of the duct remains challenging due to the MPD occupying only a small fraction of the image volume and its irregular morphology, the proposed pipeline supports a quantitative assessment of its maximum-width. Ensemble-based uncertainty analysis showed a clear correlation between higher model uncertainty and lower segmentation performance, indicating that uncertainty estimates may help identify cases that require radiologist review. Expert validation further suggested that MRCP image quality is related to maximum-width measurement reliability, with lower image-quality scores associated with higher MPD max width measurement errors. Overall, the method shows potential as a support tool for MPD width assessment, but further external validation and robustness improvements are needed before clinical deployment.

## Acknowledgments

This research was funded by Business Finland under project Medical AI and Immersion (decision number 10912/31/2022).   
The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

The authors thank Mervi Asumaniemi, ICT manager at Tampere University Hospital for her assistance in data anonymisation and data transfer.

## Author contributions

Z.Z. contributed to conceptualisation, study design, methodology development, implementation, formal analysis, and writing of the main manuscript. J.S. contributed to study design, data collection, results analysis, figure plotting, and writing of the manuscript. M.S. contributed to results analysis, and writing of the manuscript. J.R. contributed to clinical supervision, supervised data annotation, conducted the clinical validation of the model outputs, and contributed to manuscript editing. M.L. and A.A. contributed to the data annotation. I.R. contributed to clinical supervision, supervised data annotation, conducted the clinical validation of the model outputs, and contributed to manuscript editing. K.K. contributed to conceptualisation and manuscript editing. All authors reviewed the manuscript.

## Competing interest

The authors have declared that there are no competing interests.

## Data availability

The patient data cannot be shared publicly because of the data protection law of Finland, the General Data Protection Regulation (GDPR) of European Union, and our research permission granted by Tampere University Hospital that do not allow sharing of individual patients’ data. The patient data is available from Tampere University Hospital for researchers who meet the criteria for access to confidential data. More specifically the datasets are not publicly available and restrictions apply to their use. The dataset restrictions are imposed by the Wellbeing Services County of Pirkanmaa, Tampere University Hospital, Research services, P.O. Box 2000, 33521 Tampere, Finland (requests for the data access can be made to clinicaltrials@pirha.fi).

## Code Availability

The code used for data processing, deep learning algorithms, and model evaluation is not publicly available. The Methods section provides sufficient detail for replication using open-source libraries. Code is available from the corresponding author upon reasonable request.

## References

1. Sung, H. et al. Global cancer statistics 2020: Globocan estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA: a cancerjournalfor clinicians 71, 209–249 (2021).

2. Choi, T. W. et al. Comparison of multidetector ct and gadobutrol-enhanced mr imaging for evaluation of small, solid pancreatic lesions. Korean J. Radiol. 17, 509–521 (2016).

3. Griffin, N., Charles-Edwards, G. & Grant, L. A. Magnetic resonance cholangiopancreatography: the ABC of MRCP. Insights Imaging 3, 11–21, DOI: 10.1007/s13244-011-0129-9 (2012).

4. Huang, C. et al. Standardization of MRI screening and reporting in individuals with elevated risk of pancreatic ductal adenocarcinoma: Consensus statement of the PRECEDE consortium. Am. J. Roentgenol. 219, 903–914, DOI: 10.2214 AJR.22.27859 (2022). PMID: 35856454.

5. Cannella, R. et al. Conditions at risk of pancreatic cancer: The radiology perspective. Eur. J. Radiol. 112119 (2025).

6. Kang, J. D., Clarke, S. E. & Costa, A. F. Factors associated with missed and misinterpreted cases of pancreatic ductal adenocarcinoma. Eur. Radiol. 31, 2422–2432 (2021).

7. Toshima, F. et al. Ct abnormalities of the pancreas associated with the subsequent diagnosis of clinical stage i pancreatic ductal adenocarcinoma more than 1 year later: a case-control study. Am. J. Roentgenol. 217, 1353–1364 (2021).

8. Kim, S. W. et al. Isolated main pancreatic duct dilatation: CT differentiation between benign and malignant causes. Am. J. Roentgenol. 209, 1046–1055, DOI: 10.2214/AJR.17.17963 (2017). PMID: 28858540.

9. Wang, L. et al. Predicting pancreatic ductal adenocarcinoma occurrence up to 10 years in advance using features of the main pancreatic duct in pre-diagnostic ct scans. Cancers 17, 1886 (2025).

10. Ronneberger, O., Fischer, P. & Brox, T. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, 234–241 (Springer, 2015).

11. Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T. & Ronneberger, O. 3d u-net: learning dense volumetric segmentation from sparse annotation. In International conference on medical image computing and computer-assisted intervention, 424–432 (Springer, 2016).

12. Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J. & Maier-Hein, K. H. nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nat. methods 18, 203–211 (2021).

13. Hatamizadeh, A. et al. Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images. In International MICCAI brainlesion workshop, 272–284 (Springer, 2021).

14. Abi Nader, C. et al. Automatic detection of pancreatic lesions and main pancreatic duct dilatation on portal venous ct scans using deep learning. Investig. Radiol. 58, 791–798 (2023).

15. Tan, J. Y. et al. Pancreatic duct centerline extraction for image unfolding in photon-counting ct. Comput. & Graph. 104426 (2025).

16. Ambo, S., Hirano, R. & Hattori, C. Automatic detection of main pancreatic duct dilation and pancreatic parenchymal atrophy based on a shape feature in abdominal contrast-enhanced ct images. J. Med. Imaging 12, 014504–014504 (2025).

17. Shi, Y.-J. et al. Deep learning automatic segmentation and radiomics model for diagnosing pancreatic solid neoplasms in mri. BMC cancer 25, 1563 (2025).

18. Beyer, G. et al. Definition of age-dependent reference values for the diameter of the common bile duct and pancreatic duct on mrcp: a population-based, cross-sectional cohort study. Gut 72, 1738–1744 (2023).

19. Ikezawa, K. et al. Correlation between main pancreatic duct diameter measurements: Special pancreatic ultrasonography versus magnetic resonance cholangiopancreatography. Medicine 103, e37283 (2024).

20. Zhang, H. et al. Threshold of main pancreatic duct diameter in identifying malignant intraductal papillary mucinous neoplasm by magnetic resonance imaging. Technol. Cancer Res. & Treat. 22, 15330338231170942 (2023).

21. European Parliament & Council of the European Union. Regulation (eu) 2016/679 of the european parliament and of the council of 27 april 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing directive 95/46/ec (general data protection regulation) (2016). Official Journal of the European Union L 119/1.

22. Fedorov, A. et al. 3d slicer as an image computing platform for the quantitative imaging network. Magn. resonance imaging 30, 1323–1341 (2012).

23. Cardoso, M. J. et al. Monai: An open-source framework for deep learning in healthcare. arXiv preprint arXiv:2211.02701 (2022).

24. Sahlsten, J. et al. Application of simultaneous uncertainty quantification and segmentation for oropharyngeal cancer use-case with bayesian deep learning. Commun. Medicine 4, 110 (2024).