# Soil Moisture Estimation in Precision Agriculture: A Knowledge Distillation Approach

Mohammed Rakib Oklahoma State University mohammed.rakib@osktate.edu

Vrinda Nagula Aurora High School vnagula07@gmail.com

Arunkumar Bagavathi Oklahoma State University abagava@okstate.edu

## ABSTRACT (REQUIRED)

Accurate soil moisture estimation is critical for precision agriculture, influencing irrigation, fertilization, and harvesting strategies. Traditional methods primarily rely on meteorological and soil property data, while recent approaches incorporate visual modalities such as aerial and geospatial imagery. However, these high-resolution images are expensive and computationally demanding to analyze. In this work, we propose Soil Moisture Distillation (SMD), a Knowledge Distillation (KD)-based multimodal learning framework that leverages unimodal teachers to enhance multimodal soil moisture estimation. SMD surpasses not only traditional multimodal fusion techniques, including MIS-ME, but also outperforms unimodal models, demonstrating the benefits of guided unimodal knowledge transfer. Additionally, we conduct a comparative analysis of soil patches and crop images, showing that crop images provide superior visual cues, leading to improved prediction accuracy. SMD achieves a 2% improvement in MAPE for soil patches and 1.2% for crops over state-of-the-art baselines, highlighting the effectiveness of leveraging both meteorological and visual data through knowledge distillation for enhanced soil moisture forecasting.

Keywords (Required)

soil-moisture estimation, feature distillation, multi-modal regression

## INTRODUCTION

Soil moisture plays a vital role in precision agriculture, influencing irrigation strategies, crop health monitoring, and overall farm productivity. Accurate soil moisture estimation helps farmers optimize water usage, reducing waste while ensuring crops receive adequate hydration. Traditionally, soil moisture is measured using in-situ sensors, which provide precise readings but are expensive to deploy and maintain at scale (Phillips et al., 2014; Feng et al., 2023). As an alternative, machine learning based predictive models leverage meteorological and soil property data to estimate soil moisture, offering a scalable and cost effective solution (Orth and others, 2021; Filipović et al., 2022). However, relying solely on numerical meteorological data may not be sufficient, as soil moisture is also influenced by physical soil conditions and crop health, which are better captured through visual cues.

Recent research has explored image-based soil moisture estimation, utilizing satellite, aerial, and multispectral imagery to enhance predictive performance (Bertalan et al., 2022; Singh and Gaurav, 2023). While these high-resolution images capture large-scale environmental features, they are expensive to acquire, computationally intensive to process, and often lack finegrained details necessary for localized soil moisture estimation. To address this, researchers have investigated using groundlevel images of soil patches and crops, which offer micro-level insights into soil texture, vegetation status, and water retention (Rakib et al., 2024). However, the effectiveness of such visual information depends on how well it complements traditional meteorological data in a multimodal setting.

Despite the promise of multimodal approaches, integrating heterogeneous data sources presents significant challenges (Du et al., 2023; Zhang et al., 2024). One key issue is modality imbalance (Peng et al., 2022), where dominant modalities can overshadow weaker modalities, leading to suboptimal feature fusion. Traditional multimodal learning models struggle to effectively balance and utilize information from different sources, often prioritizing the most predictive modality while neglecting complementary signals.

![](images/b920a1eff1c1656f6979ead9a4ef3c1dad6f06d036911390725b981f4c7c5c87.jpg)  
Figure 1: Overview of our Methodology. Soil patches or crop images, along with meteorological tabular data, serve as inputs to both the pretrained unimodal teacher models and a multimodal student model. The student aligns its unimodal encoders with the teacher models through feature-based losses while also optimizing for prediction accuracy using MSE loss against ground truth labels.

To address this challenge in soil moisture estimation, we propose Soil Moisture Distillation (SMD), a multimodal Knowledge Distillation (KD) framework that ensures both meteorological and visual data contribute effectively to the prediction task. Originally introduced for model compression (Bucila, Caruana and Niculescu-Mizil, 2006; Hinton, Vinyals and Dean, 2015), KD has evolved into a broader technique for transferring knowledge from a stronger teacher model to a student model, improving its learning efficiency. Multimodal KD extends this concept by leveraging information from multiple modalities to enhance representation learning, which supports several real-world AI applications (Wang et al., 2023; Yuan and others, 2024). By leveraging unimodal teacher models, trained separately on meteorological and image data, we guide a multimodal student model to align its feature representations with those of the teachers. This structured supervision ensures that both modalities contribute meaningfully, mitigating modality imbalance and leading to improved soil moisture prediction.

To evaluate SMD, we conduct extensive experiments on the MIS-ME (Rakib et al., 2024) dataset, comparing our KD-based approach against standard multimodal fusion techniques and state-of-the-art baseline models. Our results demonstrate that SMD outperforms traditional fusion methods and baselines. Additionally, we analyze the relative contributions of soil patches and crop images, showing that crop images provide richer visual cues, leading to superior predictive performance.

Our key contributions are as follows:

We propose Soil Moisture Distillation (SMD), a KD-based multimodal framework that leverages unimodal teachers to enhance soil moisture prediction.

• We analyze the impact of soil patches versus crop images, demonstrating that crop images offer superior predictive performance.

SMD achieves a 2% improvement in MAPE for soil patches and 1.2% for crops, surpassing baseline multimodal methods.

## METHODOLOGY

## Motivation

Integrating data from multiple modalities has been shown to significantly enhance model performance by capturing complementary information that is otherwise unavailable in a single modality. In the context of soil moisture estimation, combining meteorological tabular data and image-based features (e.g., crop images or soil patches) allows models to leverage both numerical patterns and visual cues, leading to more accurate predictions. This fusion of heterogeneous data enables the model to better generalize across diverse environmental conditions and reduces reliance on any single source of information.

However, multimodal learning often suffers from modality imbalance, where certain dominant modalities contribute disproportionately to the learning process while weaker modalities are underutilized. This imbalance can degrade model performance, as the network may prioritize features from a single modality instead of effectively integrating all available information. Multimodal Knowledge Distillation (KD) addresses this issue by employing unimodal teacher models, which are pretrained on individual modalities. These teachers guide the multimodal student model to align its features with unimodal expert representations, ensuring that each modality is utilized effectively. This structured supervision mitigates modality imbalance and enhances robustness.

For soil moisture estimation, the combination of meteorological tabular data (e.g., temperature, humidity, precipitation) and image data (e.g., soil/crop textures) is crucial for accurate forecasting. By distilling knowledge from unimodal teachers, SMD ensures that both meteorological and visual information contribute meaningfully to the prediction task, ultimately improving estimation accuracy and model interpretability.

## Problem Formulation

We consider a supervised learning setup where the goal is to estimate soil moisture content given multimodal inputs consisting of meteorological tabular data and image data. Formally, let m denote the number of available modalities, where each modality has an associated unimodal teacher model $T _ { i }$ that is trained independently. The multimodal student model � is designed to integrate these modalities while benefiting from teacher guidance.

Each input sample consists of: tabular meteorological data $\left( X _ { t a b } \right)$ and image data $( X _ { i m g } )$ , which can be either soil patches or crop images. The unimodal teacher models process their respective inputs to produce predictions:

$$
T _ { \mathrm { t a b } } ( X _ { \mathrm { t a b } } ) = Y _ { \mathrm { t a b } } , \quad T _ { \mathrm { i m g } } ( X _ { \mathrm { i m g } } ) = Y _ { \mathrm { i m g } }\tag{1}
$$

Similarly, the multimodal student model extracts modality-specific features $F _ { \mathrm { t a b } }$ and $F _ { \mathrm { i m g } }$ before applying a fusion strategy:

$$
F _ { S } = f \left( F _ { \mathrm { t a b } } , F _ { \mathrm { i m g } } \right)\tag{2}
$$

where $f$ represents the chosen fusion technique, which can be any fusion method concatenation, late fusion, summation, etc.) In our case, we use concatenation. The final prediction is then computed as:

$$
Y _ { S } = g ( F _ { S } )\tag{3}
$$

where $_ g$ is the output prediction function.

Figure 1 illustrates this pipeline, where both unimodal teacher models extract features independently from their respective inputs. The multimodal student model then integrates these features through the fusion module, producing a final multimodal output. Additionally, to compare the student model with the unimodal teachers, the unimodal outputs of the student model are derived by passing pre-fusion modality-specific features through dedicated linear layers. This setup allows direct supervision from each teacher model, ensuring that knowledge from each modality is preserved.

## Multimodal Student Loss

Since soil moisture estimation is a regression problem, the multimodal student is trained using Mean Squared Error (MSE) loss between its predictions and the ground truth labels:

$$
L _ { \mathrm { M S E } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bigl ( Y _ { S } ^ { ( i ) } - Y _ { \mathrm { G T } } ^ { ( i ) } \bigr ) ^ { 2 }\tag{4}
$$

Where $Y _ { S }$ is the multimodal student’s prediction and $Y _ { G T }$ is the true soil moisture value. The rationale behind this loss function is to minimize the squared differences, penalizing large deviations and ensuring accurate regression modeling.

## Feature Distillation Loss

To enforce alignment between the multimodal student and unimodal teachers, we introduce a feature distillation loss for each modality:

$$
L _ { \mathrm { F D } } = \sum _ { i = 1 } ^ { m } | F _ { S } ^ { ( i ) } - F _ { T } ^ { ( i ) } | ^ { 2 }\tag{5}
$$

where $F _ { S } ^ { ( i ) }$ and $F _ { T } ^ { ( i ) }$ are the feature representations of the student and the corresponding teacher for modality �.

The final training objective is a weighted sum of the MSE loss and feature distillation loss:

$$
L _ { \mathrm { t o t a l } } = \propto L _ { \mathrm { M S E } } + \beta L _ { F D }\tag{6}
$$

where � and � are hyperparameters balancing the contribution of the two losses. For simplicity, in our experiments, we set � = 1.0 and � = 1.0, treating both losses as equally important.

By aligning student features with their unimodal teacher counterparts, this approach ensures that each modality contributes effectively to the final prediction, mitigating the modality imbalance problem and leading to superior generalization in soil moisture estimation.

## EXPERIMENTS & RESULTS

## Experimental Setup

We evaluate SMD using the dataset introduced in the MIS-ME paper, which includes meteorological tabular data along with two types of image data: soil patches and crop images. Following the same 65:15:20 split for training, validation, and testing, we predict volumetric water content (VWC), measured in cm<sup>3</sup>cm<sup>-3</sup>. To investigate the impact of different image modalities, we integrate meteorological data once with soil patches and once with crops, comparing their performance to determine which modality contributes more effectively to soil moisture estimation.

We compare SMD with two baseline approaches: (1) Normal Fusion, which employs standard concatenation of features, and (2) the Hybrid Loss method proposed in the MIS-ME paper, which achieved the best results in their study. For consistency, we adopt the hyperparameter settings from the MIS-ME paper. For the image feature extractor, we use ResNet18 as the backbone for both the teacher and the student models. Similarly, for the meteorological feature extractor, we employ the fully-connected neural network architecture described in the MIS-ME paper for both teacher and student networks. This ensures a fair comparison between our KD approach and the baselines. We evaluate our method using Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and R² score, ensuring a comprehensive assessment of model performance.

## Comparing Our KD Method with Baselines

Table 1 presents a comparison of SMD against baseline approaches for both soil patches and crops. Our method outperforms all baselines, demonstrating its effectiveness in soil moisture estimation. For soil patches, SMD improves the MAPE score by approximately 2% compared to both the normal fusion and MIS-ME methods, which exhibit similar performance. For crops, SMD achieves a 1.2% improvement in MAPE over MIS-ME and around 2.5% over normal fusion. These results highlight the efficacy of our KD framework in integrating multimodal data.

Additionally, when examining unimodal performances, we observe that the integration via our KD method yields a significant improvement in soil moisture estimation. Compared to unimodal models, the combination of modalities using our KD approach results in better generalization and more accurate predictions. This demonstrates that unimodal teachers effectively guide the multimodal student, leading to more robust feature representations.

Finally, we note that crop images consistently outperform soil patches across all methods. This may be attributed to the fact that crop images capture a broader horizon, encompassing both soil properties and crop characteristics, thereby providing a more comprehensive context for soil moisture estimation. In contrast, soil patch images focus solely on the soil surface, offering limited information that may not fully represent the dynamic interactions between soil and vegetation.

Table 1: Result Comparison
<table><tr><td rowspan=2 colspan=1>Baselines</td><td rowspan=1 colspan=3>Soil Patches</td><td rowspan=1 colspan=3>Crops</td></tr><tr><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>MAPE</td><td rowspan=1 colspan=1>R2</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>MAPE</td><td rowspan=1 colspan=1>R2</td></tr><tr><td rowspan=1 colspan=1>Meteorological Model</td><td rowspan=1 colspan=1>0.035</td><td rowspan=1 colspan=1>13.080</td><td rowspan=1 colspan=1>0.485</td><td rowspan=1 colspan=1>0.040</td><td rowspan=1 colspan=1>15.495</td><td rowspan=1 colspan=1>0.321</td></tr><tr><td rowspan=1 colspan=1>Image Model</td><td rowspan=1 colspan=1>0.037</td><td rowspan=1 colspan=1>14.227</td><td rowspan=1 colspan=1>0.360</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>6.742</td><td rowspan=1 colspan=1>0.819</td></tr><tr><td rowspan=1 colspan=1>Normal Fusion</td><td rowspan=1 colspan=1>0.039</td><td rowspan=1 colspan=1>15.044</td><td rowspan=1 colspan=1>0.383</td><td rowspan=1 colspan=1>0.019</td><td rowspan=1 colspan=1>7.188</td><td rowspan=1 colspan=1>0.811</td></tr><tr><td rowspan=1 colspan=1>MIS-ME</td><td rowspan=1 colspan=1>0.039</td><td rowspan=1 colspan=1>15.163</td><td rowspan=1 colspan=1>0.395</td><td rowspan=1 colspan=1>0.015</td><td rowspan=1 colspan=1>5.928</td><td rowspan=1 colspan=1>0.865</td></tr><tr><td rowspan=1 colspan=1>SMD (Ours)</td><td rowspan=1 colspan=1>0.034</td><td rowspan=1 colspan=1>13.022</td><td rowspan=1 colspan=1>0.491</td><td rowspan=1 colspan=1>0.012</td><td rowspan=1 colspan=1>4.751</td><td rowspan=1 colspan=1>0.901</td></tr></table>

## Residual Analysis

Residual analysis evaluates prediction accuracy by examining the deviation between predicted and actual soil moisture values Figure 2 compares residual distributions for normal fusion and SMD using soil patches and crops. A higher percentage of residuals within ±0.05 cm³/cm³ indicates better alignment between predictions and ground truth.

For soil patches, normal fusion captures 67.04% of residuals within ±0.05 cm³/cm³, while SMD improves this to 75.88%, demonstrating reduced prediction error and improved generalization. The wider spread in normal fusion residuals suggests higher uncertainty, whereas SMD produces a more centered and consistent distribution.

For crops, normal fusion achieves 93.11%, whereas SMD further improves accuracy to 95.89%. This confirms that crops provide better predictive performance than soil patches and that SMD enhances feature extraction by leveraging unimodal teachers, reducing residual variance.

Overall, our framework significantly improves prediction accuracy, particularly for soil patches where normal fusion struggles with higher residual errors. By aligning student features with unimodal teachers, SMD refines feature representations and reduces uncertainty, making it a more robust framework for soil moisture estimation.

Actual vs. Residual VWC Comparison  
![](images/2d63c73edbef940ebdbed803cc3679df859d05570d4bbbf83412edd6cc237520.jpg)  
Figure 2: Residual Comparison between Normal Fusion and SMD

## CONCLUSION

In this work, we proposed Soil Moisture Distillation (SMD), a Knowledge Distillation (KD)-based multimodal learning framework for soil moisture estimation by integrating image and meteorological data. SMD addresses the modality imbalance problem by leveraging unimodal teacher models, ensuring effective utilization of both modalities. By aligning the multimodal student model with unimodal expert representations, we enhance feature extraction and improve soil moisture regression. Additionally, our analysis of soil patches and crop images highlights the importance of incorporating visual cues alongside meteorological data for a more comprehensive estimation framework.

For future work, we aim to explore KD for model compression, making the SMD models more lightweight and efficient for deployment on edge devices. Optimizing the model for low-power agricultural systems will enable real-time soil moisture prediction in resource-constrained environments, facilitating broader adoption of AI-driven precision agriculture solutions.

## ACKNOWLEDGEMENTS

This work is funded by the U.S. Department of Agriculture, National Institute of Food and Agriculture, Data Science for Food and Agricultural Systems (DSFAS) program. Accession number 1030694.

## REFERENCES

Bertalan, L. et al. (2022) ‘UAV-based multispectral and thermal cameras to predict soil water content–A machine learning approach’, Computers and Electronics in Agriculture, 200, p. 107262.

Bucila, C., Caruana, R. and Niculescu-Mizil, A. (2006) ‘Model compression’, in Knowledge Discovery and Data Mining.

Du, C. et al. (2023) ‘On Uni-Modal Feature Learning in Supervised Multi-Modal Learning’, in Proceedings of the 40th International Conference on Machine Learning. PMLR (Proceedings of Machine Learning Research), pp. 8632–8656.

Feng, Y. et al. (2023) ‘LTE-Based Low-Cost and Low-Power Soil Moisture Sensing’, in Proceedings of the 20th ACM Conference on Embedded Networked Sensor Systems. New York, NY, USA: ACM (SenSys ’22), pp. 421–434.

Filipović, N. et al. (2022) ‘Regional soil moisture prediction system based on Long Short-Term Memory network’, Biosystems engineering, 213, pp. 30–38.

Hinton, G., Vinyals, O. and Dean, J. (2015) ‘Distilling the Knowledge in a Neural Network’, arXiv preprint arXiv:1503.02531.

Orth, R. and others (2021) ‘Global soil moisture data derived through machine learning trained with in-situ measurements’, Scientific Data, 8(1), pp. 1–14.

Peng, X. et al. (2022) ‘Balanced multimodal learning via on-the-fly gradient modulation’, in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8238–8247.

Phillips, A.J. et al. (2014) ‘Integrated sensing of soil moisture at the field-scale: Measuring, modeling and sharing for improved agricultural decision support’, Computers and Electronics in Agriculture, 107, pp. 73–88.

Rakib, M. et al. (2024) ‘MIS-ME: A Multi-modal Framework for Soil Moisture Estimation’, in 2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA). IEEE, pp. 1–10.

Singh, A. and Gaurav, K. (2023) ‘Deep learning and data fusion to estimate surface soil moisture from multi-sensor satellite images’, Scientific Reports, 13(1), p. 2251.

Wang, H. et al. (2023) ‘Learnable Cross-modal Knowledge Distillation for Multi-modal Learning with Missing Modality’, in Medical Image Computing and Computer Assisted Intervention – MICCAI 2023: 26th International Conference, Vancouver, BC, Canada, October 8–12, 2023, Proceedings, Part IV. Berlin, Heidelberg: Springer-Verlag, pp. 216–226.

Yuan, Y. and others (2024) ‘Multiscale knowledge distillation with attention based fusion for robust human activity recognition’, Scientific Reports, 14, p. 63195.

Zhang, X. et al. (2024) ‘Multimodal Representation Learning by Alternating Unimodal Adaptation’, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 27456–27466.