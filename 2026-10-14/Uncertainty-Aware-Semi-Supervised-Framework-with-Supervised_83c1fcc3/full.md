# Uncertainty-Aware Semi-Supervised Framework with Supervised Contrastive Learning for Sugarcane Leaf Disease Detection

Shalini Dangi<sup>a</sup> (phd2301201003@iiti.ac.in), Divya Teja<sup>a</sup> (cse210001040@iiti.ac.in),

Omkar Kabde<sup>a</sup> (ugs22058\_cic.omkar@cbit.org.in), Shahid Shafi Dar<sup>a</sup> (phd2201201004@iiti.ac.in),

Nagendra Kumar<sup>a</sup> (nagendra@iiti.ac.in)

## Highlights

• An uncertainty-aware semi-supervised network is proposed for sugarcane leaf disease detection.

• Discrete wavelet transform is used to enhance disease-related image features.

• KL-divergence-based refinement improves the reliability of pseudo-labels.

• Contrastive learning strengthens class-discriminative feature representations.

• Experimental results show efective learning from limited annotations and competitive performance.

This is the preprint version of the accepted paper.

This paper is accepted in ISPRS Journal of Photogrammetry and Remote Sensing, 2026.

DOI: https://doi.org/10.1016/j.isprsjprs.2026.06.005

# Uncertainty-Aware Semi-Supervised Framework with Supervised Contrastive Learning for Sugarcane Leaf Disease Detection

Shalini Dangi<sup>a</sup>, Divya Teja<sup>a</sup>, Omkar Kabde<sup>b</sup>, Shahid Shafi Dar<sup>a</sup>, Nagendra Kumar<sup>a,∗</sup>

<sup>a</sup>Indian Institute of Technology Indore, Madhya Pradesh, India <sup>b</sup>Chaitanya Bharathi Institute of Technology, Telangana, India

## Abstract

Sugarcane leaf diseases severely impact crop yield and quality, making accurate and timely detection critical for sustainable agriculture. Conventional deep learning models for plant disease detection depend on large annotated datasets, which are expensive to obtain and challenging to prepare due to challenges commonly observed across crop disease detection tasks, including (i) high intra-class variation caused by diverse symptom expressions across growth stages, (ii) subtle inter-class similarities between diseased and healthy tissues, and (iii) the risk of noisy labels degrading model reliability. To handle these challenges, we presented an Uncertainty-aware Mean-Teacher based Semi-Supervised Learning Disease Detection Network (U-SSDNet) that leverages unlabeled data while maintaining robustness. First, discrete wavelet transform preprocessing is applied to enhance lesion visibility and reduce intraclass variation. Subsequently, the KL divergence is employed to refine the pseudo-labels and reduce the impact of confirmation bias. At the same time, supervised contrastive learning is applied to make the features more distinguishable and improve the separation among diferent classes. Experimental evaluations on sugarcane leaf disease datasets, considered a representative case study, demonstrate that the proposed approach achieves performance close to that of fully supervised models while requiring far fewer annotated samples and consistently outperforms state-of-the-art supervised and semi-supervised approaches.

Keywords: Precision Agriculture, Plant disease detection, Semi-supervised learning, Consistency regularization, Contrastive learning

## 1. Introduction

Agriculture sustains global food supply, rural life, and economic strength. With continued global population growth, the need for food and renewable resources increases, putting significant pressure on farming to produce more without harming the environment. Sugarcane is also an important crop worldwide, accounting for about 80% of global sugar production. However, cultivating sugarcane is challenging because many diseases attack the plants, leading to significant reductions in crop quality and yield. Leaf diseases can afect plant health by reducing growth rates and lowering sugar content, posing major challenges for both farmers and the sugar production sector (Barbedo, 2016). These leaf diseases spread rapidly, making early detection dificult in favorable weather. Although laboratory methods like PCR and DNA sequencing are highly accurate and require substantial resources, making them less practical for large-scale use (Strachan et al., 2022).

Conventional disease classification methods depend on biochemical tests and expert-based visual inspection. Although accurate, biochemical methods are costly, require chemical reagents and specialized labor, and cannot be easily scaled to large fields. On the other hand, Manual inspection is labor-intensive and impractical for monitoring large-scale farms. Computer vision and deep learning approaches have been used to overcome these constraints in leafdisease detection. Diferent Convolutional Neural Network (CNN) architectures such as Inception-V3, ResNet50, VGG16, and VGG19 have shown notable performance in extracting features from plant leafimages (Dey et al., 2022). The performance of CNN-based models mostly depends on the availability of labeled datasets. In practice, plant disease detection tasks commonly sufer from high intra-class variation, where the same disease may appear diferently across growth stages and environmental conditions, as well as subtle inter-class similarities, where diferent diseases exhibit overlapping visual patterns with healthy tissues (Lu and Young, 2020; Barbedo, 2018). These challenges are widely observed across crop disease detection tasks and are not specific to a single crop. Variations in symptom appearance and real-field imaging conditions in sugarcane leaf disease detection intensify these challenges. As a result, limited feature diversity and a higher risk of overfitting can restrict the practical use of deep learning models.

Limited labeled agricultural data is a major challenge for building accurate plant disease detection models, as expert annotation is costly and time-consuming. Semi-supervised learning (SSL) ofers a promising solution by leveraging large amounts of unlabeled data alongside limited labeled samples to improve generalization and robustness. For example, SSL has enhanced performance with ConvNeXt models using consistency regularization for precise weed detection (Benchallal et al., 2024), and attention-based self-supervised contrastive learning has improved plant disease detection by extracting diverse leaf disease features from unlabeled images (Yilma et al., 2025; Dangi et al., 2025). Frameworks that incorporate both intra-class and inter-class information have also improved pest identification in real-field scenarios (Hu et al., 2025), and uncertainty-aware optimization in semi-supervised contrastive learning has increased robustness in agricultural monitoring (Ai et al., 2025). These findings demonstrate that SSL provides a scalable, cost-efective solution for plant disease detection with limited supervision. However, challenges related to robustness and uncertainty handling remain open research problems.

This study presents an SSL-based Mean-Teacher approach for detecting sugarcane leaf disease. The Mean-Teacher model improves performance by enforcing consistency between the student and teacher networks and leveraging unlabeled data. Sugarcane leaf images are processed using the Discrete Wavelet Transform (DWT) to improve feature representation and focus on patterns of leaf diseases. In addition, Kullback-Leibler divergence is used for pseudo-label correction to reduce confirmation bias and increase prediction accuracy. Supervised contrastive learning is also applied to improve class separation and disease detection performance. The proposed approach combines these techniques to address intra-class variation and inter-class similarity in plant disease detection, us ing sugarcane leaf disease as a representative case study.

The proposed research outlines the following main contributions:

• We propose a novel semi-supervised Mean-Teacher framework for plant disease detection, evaluated using sugarcane leaf disease datasets as a representative case study.

• We introduce a discrete wavelet transform to enhance the semantic context of visual features, thereby improving the model’s representation ability.

• We leverage supervised contrastive learning across diferent leaf disease classes to improve class separability and enhance classification performance.

• We propose a KL-divergence-based pseudo-label correction approach to reduce noise in pseudo-labels and improve model stability.

The rest of the manuscript is structured as follows: Section 2 details the literature works, Section 3 define the problem formulation, Section 4 presents the detailed analysis of our proposed methodology, Section 5 describes the results analysis and the details of the experimental setup, Section 6 presents the discussions, including limitations, and Section 7 presents the conclusion.

## 2. Related Work

## 2.1. Traditional Methods

Plant leaf disease detection is a key challenge in modern agriculture. Early approaches mainly employ handcrafted feature extraction and machine learning classification methods such as Support Vector Machines (SVMs), Naive Bayes, Decision Trees, and Random Forests (Sari et al., 2021; Mia et al., 2020; Panigrahi et al., 2020). The above methods use handcrafted features such as color, shape, and texture to identify healthy and diseased leaves. These features can be afected by leaf position, background, changes in lighting, and inter-class variability. Their reliance on manual feature engineering introduces subjectivity and limits scalability, particularly in real-world conditions where data may be diverse and noisy. Therefore, their ability to generalize to unseen conditions or large-scale datasets remains limited. These challenges are not specific to a single crop and are commonly observed in plant disease detection under varied real-world conditions.

## 2.2. Deep Learning for Plant Disease Detection

Deep learning has emerged as a prevalent methodology for plant leaf disease detection due to the strong featurelearning capabilities ofconvolutional neural networks (CNNs). Early research demonstrated that CNN-based models are efective for identifying plant diseases, particularly under controlled imaging conditions (Dyrmann et al., 2016; Mohanty et al., 2016). To further enhance classification performance, deeper network architectures such as ResNet and DenseNet were subsequently introduced, enabling richer feature representations and improved learning capacity (Huang et al., 2017; He et al., 2016). Comparative studies have consistently shown that CNN-based approaches outperform traditional machine learning techniques for crop disease recognition across multiple plant species (Dey et al., 2022; Shah et al., 2022). Despite their success, CNN-based models remain afected by high intra-class varia tion, subtle inter-class similarities, and environmental variability, which are common across plant disease detection tasks. To address the challenge of limited annotated agricultural data, transfer learning has been widely adopted. In this approach, models pre-trained on large-scale datasets such as ImageNet are fine-tuned to improve plant leaf disease classification performance (Lambor et al., 2022; Singh et al., 2022; Kavala and Pothuraju, 2022). Furthermore, lightweight and computationally eficient architectures, including MobileNet and EficientNet, have been explored to support deployment in resource-constrained agricultural environments (Ali et al., 2024; Liu et al., 2022; Tang et al., 2020). However, prior studies indicate that the efectiveness of deep learning models is highly dependent on dataset size and diversity, and their performance often degrades under real-world field conditions (Barbedo, 2018, 2016). Deep learning methods have also been applied to other agricultural tasks, such as query–response systems (Rehman et al., 2023) and yield prediction (Dangi et al., 2025), to support farmers in decision-making and improve crop production.

## 2.3. Semi-Supervised Learning in Agriculture

To mitigate the reliance on large labeled datasets, semi-supervised learning (SSL) has emerged as a promising paradigm in agricultural vision. SSL methods aim to exploit unlabeled data through techniques such as pseudolabeling, self-training, and consistency regularization (Raghaw et al., 2026). Recent studies have shown that incorporating unlabeled samples can significantly improve plant disease recognition performance when labeled data are scarce (Benchallal et al., 2024; Ilsever and Baz, 2024). These approaches are particularly suitable for agricultural applications, where collecting large-scale annotations is labor-intensive and costly (Thite et al., 2024; Lu and Young, 2020). However, existing SSL methods face several challenges in practical agricultural scenarios. Pseudo-labeling approaches are prone to error accumulation due to incorrect predictions, while consistency-based methods often assume reliable predictions on unlabeled samples without explicitly modeling uncertainty (Hu et al., 2025). This limitation is critical for plant disease detection, where visual similarities between disease classes and variations in environmental conditions can lead to ambiguous predictions. Consequently, robust SSL frameworks that can handle noisy labels and uncertain samples remain an open research problem in agricultural vision.

## 2.4. Mean-Teacher and Uncertainty-Aware Learning

The Mean-Teacher framework is a semi-supervised learning approach that employs a teacher-student architecture. The teacher network is updated as an exponential moving average (EMA) of the student network parameters, by enforcing consistency between student and teacher predictions on unlabeled data. Mean-Teacher models have shown enhanced generality and stability in diferent computer vision tasks (Tarvainen and Valpola, 2017). This consistency-based learning strategy efectively regularizes the training process and reduces overfitting. Despite its success, most Mean-Teacher-based approaches do not explicitly account for prediction uncertainty and treat all unlabeled samples equally during training. Recent research work in other application domains has shown that incorporating uncertainty-aware mechanisms, such as contrastive learning and uncertainty-weighted optimization, can significantly improve semi-supervised learning performance (Ai et al., 2025; Liu et al., 2024). However, uncertaintyaware Mean-Teacher frameworks have rarely been explored in agricultural disease detection. Given the inheren variability, inter-class similarity, and noisy visual patterns in crop disease datasets (Ye et al., 2024; Strachan et al., 2022) and the lack of uncertainty modeling, this represents a critical gap that motivates the proposed approach.

## 3. Problem Formulation

We address semi-supervised image classification using both labeled and unlabeled data. The labeled dataset is defined in Equation 1:

$$
\mathcal { D } _ { L } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { L } }\tag{1}
$$

where $x _ { i }$ is an input image and $y _ { i } \in \{ 1 , \ldots , C \}$ denotes its class label. The unlabeled dataset is given by Equation 2.

$$
\mathcal { D } _ { U } = \{ x _ { j } \} _ { j = 1 } ^ { N _ { U } }\tag{2}
$$

A teacher-student framework is employed, where the predictions of the teacher and student networks are represented by $\hat { y } ^ { T }$ and $\hat { y } ^ { S }$ , respectively. The student is optimized using a supervised loss $\mathcal { L } _ { s u p }$ on $\mathcal { D } _ { L }$ and an uncertaintyguided consistency loss $\mathcal { L } _ { c o n s }$ to align predictions on both $\mathcal { D } _ { L }$ and $\mathcal { D } _ { U }$ . The total loss combines these two terms, enabling the model to utilize unlabeled data and improve generalization under limited annotations.

## 4. Methodology

The methodology section provides a complete overview ofthe proposed U-SSDNet framework for sugarcane leaf disease detection, as described in Figure 1. The framework includes four distinct sub-modules: (A) Sugarcane Leaf Preprocessing and Augmentation in which includes the initial processing such as resizing and augmentation; (B) Supervised Contrastive Learning for Inter-Class Discriminative Embedding, which optimizes feature representations by learning to contrast between diferent disease classes using a contrastive loss function in a supervised manner; (C) Mean-Teacher based Semi-Supervised Learning for Global Feature Representation, which utilizes a semi-supervised learning approach to enhance model generalization by uses both labeled and unlabeled data.

## 4.1. Sugarcane LeafPreprocessing and Augmentation:

The preprocessing stage includes resizing images and applying augmentation to increase data diversity and improve model performance. Data augmentation plays an essential role in leaf disease detection because the labeled dataset is often small and images vary in orientation and environmental conditions (Naveed et al., 2024; Xu et al., 2023). It strengthens the model’s robustness by replicating real-world aspects such as partial occlusion, changes in lighting, and image blur. We first apply traditional augmentations, such as horizontal and vertical flips and rotations, to enable the model to generalize across diferent leafpositions and symmetries and introduce basic spatial variations. For labeled data, strong augmentations such as Color Jittering, Gaussian Blur, and Random Erasing are applied to create highly diverse training samples, which help improve the model’s performance by providing it with a wide range of possible diferences between diseased and healthy leaves.

## 4.1.1. Color Jittering

Color jittering introduces random perturbations to the image’s color attributes, brightness, contrast, saturation, and hue to simulate various lighting and environmental conditions (Xu et al., 2023). Initially, the RGB image I is modified using a linear transformation to adjust brightness and contrast, as shown in Equation 3:

$$
I ^ { \prime } = T _ { \mathrm { j i t t e r } } ( I ) = \alpha \cdot I + \beta\tag{3}
$$

where α is a random scalar controlling contrast, and $\beta$ adjusts brightness. To introduce additional variation, the image is converted to the HSV color space, allowing separate control over hue and saturation. The transformation in HSV space is given in Equation $^ { 4 , }$ in which $\Delta H$ represents the hue shift, and $\alpha _ { s }$ and $\alpha _ { v }$ are random scaling factors that modify saturation and brightness, respectively.

![](images/ebb31fb99cb83dbfcced16724e6de4efdd2710672d14e7fb491664293824e085.jpg)  
Figure 1: The proposed U-SSDNet framework contains four key elements: (i) Sugarcane Leaf Preprocessing and Augmentation; (ii) A Mean-Teacher model for Consistency Learning; (iii) Supervised Contrastive Learning for discriminative representation; and (iv) KL-divergence–based Uncertainty Estimation to guide the utilization of unlabeled data.

$$
I _ { \mathrm { H S V } } ^ { \prime } = [ H + \Delta H , S \cdot \alpha _ { s } , V \cdot \alpha _ { v } ]\tag{4}
$$

## 4.1.2. Random Erasing

Random Erasing is a data augmentation technique that enhances model robustness by simulating occlusions or missing regions in an image (Naveed et al., 2024). A random rectangular region $R \subset I$ is selected, and the pixel values within this region are replaced as defined in Equation 5:

$$
I ^ { \prime } ( x , y ) = { \left\{ \begin{array} { l l } { v , } & { { \mathrm { i f ~ } } ( x , y ) \in R , } \\ { I ( x , y ) , } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{5}
$$

In our work with sugarcane leaf disease images, the region R is randomly positioned within the image, and its pixel values v are set to a fixed constant of 128, representing a neutral gray. The size of R is determined by an area ratio s between 0.02 and 0.33 relative to the image, and its width-to-height aspect ratio r is sampled from 0.3 to 3.3. Random Erasing is applied with a 50% probability, ensuring meaningful occlusions that improve model robustness without obscuring critical lesion information.

## 4.2. Supervised Contrastive Learning for Inter-Class Discriminative Embedding

In sugarcane leaf disease detection, a key challenge is learning features that can diferentiate between leaf diseases with similar visual appearances. Conventional feature extraction techniques often struggle to capture subtle diferences among disease types, leading to overlapping features and reduced classification performance. This problem becomes more dificult under conditions such as complex backgrounds, uneven lighting, and limited labeled data, which reduce the model’s ability to separate disease classes.

To address these issues, a Supervised Contrastive Learning (SCL) module is added to improve the discriminative ability ofthe EficientNet-based feature extractor in the student model. By using labeled data, the SCL module makes features of the similar leaf disease class more compact while increasing the distance between features of diferent classes. In the proposed SCL module, contrastive pairs are constructed within each training mini-batch using only labeled samples. For a given anchor sample, all other samples in the batch that share the same class label are treated as positive pairs, while samples belonging to diferent classes are treated as negative pairs. The contrastive loss for each anchor is computed against all other labeled samples in the batch, encouraging compact feature representations for the same class and separation across diferent classes (Raghaw et al., 2024). Unlabeled samples are excluded from the SCL loss and are instead optimized through consistency regularization and pseudo-label correction within the Mean-Teacher framework.

Following Chen et al. (2020), we implement SCL using a Siamese network. In this setup, two identical networks with shared weights process sample pairs to learn discriminative embeddings. Unlike many existing contrastive methods that are self-supervised or use pseudo-labeled data, our approach applies contrastive loss only to reliable labeled samples and integrates it within a Mean-Teacher framework. Labeled data improve class separation, while unlabeled data are handled via consistency regularization and uncertainty-aware pseudo-label refinement. This produces more stable and discriminative features under real-field conditions. The objective of SCL is to bring feature vectors of the same class closer together while pushing apart vectors of diferent classes. For each anchor sample i, the loss is computed against all other labeled samples in the mini-batch of size $B _ { l } .$ , encouraging compact representations for the same class and greater separation across classes, as shown in Equation 6.

$$
\mathcal { L } _ { i } = - \frac { 1 } { \left| S ( i ) \right| } \sum _ { j \in S ( i ) } \log \left( \frac { \exp ( \sin ( \mathbf { z } _ { i } , \mathbf { z } _ { j } ) / \tau ) } { \sum _ { k = 1 } ^ { B _ { l } } \mathbf { 1 } _ { [ k \neq i ] } \exp ( \sin ( \mathbf { z } _ { i } , \mathbf { z } _ { k } ) / \tau ) } \right)\tag{6}
$$

Here, i is the anchor, $j \in \mathcal { S } ( i )$ are the positive samples, and $k = 1 , \dots , B _ { l }$ indexes all labeled samples in the minibatch. $\mathbf { z } _ { i }$ and $\mathbf { z } _ { j }$ are L2-normalized feature vectors from the student network, and sim $( \mathbf { z } _ { i } , \mathbf { z } _ { j } )$ denotes the cosine similarity. The temperature parameter $\tau$ controls the sharpness of feature separation: smaller values emphasize diferences, while larger values smooth the similarity distribution. The total SCL loss over the mini-batch is defined in Equation 7:

$$
\mathcal { L } _ { \mathrm { S C L } } = \frac { 1 } { B _ { l } } \sum _ { i = 1 } ^ { B _ { l } } \mathcal { L } _ { i }\tag{7}
$$

Computing pairwise similarities for all labeled samples in a mini-batch of size $B _ { l }$ results in a time complexity of $\mathcal { O } ( B _ { l } ^ { 2 } d )$ , where d is the embedding dimension. The Mean-Teacher consistency regularization involves pointwise comparisons between student and teacher predictions for all samples in the batch, scaling linearly as $\mathcal { O } ( B d )$ , where B is the total batch size (labeled + unlabeled). Overall, the per-iteration training complexity is dominated by the SCL component but remains eficient for practical mini-batch sizes. By integrating SCL with the Mean-Teacher framework, the network learns discriminative, semantically meaningful features, improving classification accuracy and generalization for sugarcane leaf disease detection.

## 4.3. Mean-Teacher based Semi-Supervised Learning for Global Feature Representation

This Mean-Teacher-based feature extraction framework comprises four key components: feature enhancement, the Mean-Teacher model, pseudo-label rectification, and a unified overall loss formulation. The enhancement module leverages domain knowledge and data transformations to improve feature quality and preserve complex patterns before they enter the model. The core Mean-Teacher architecture facilitates semi-supervised learning by encouraging consistency between a student and its Exponential Moving Average (EMA) teacher under input, as shown in Figure 1. A pseudo-label rectification mechanism is introduced to refine the learning cue from unlabeled data, correcting noisy or uncertain teacher predictions before they are used as supervision targets. Finally, an overall loss function integrates supervised and unsupervised objectives, balancing cross-entropy on labeled data with consis tency and rectification losses on unlabeled data.

## 4.3.1. Feature Enhancement

Wavelet feature extraction plays a crucial role in leaf disease detection by capturing both spatial and frequency information from leaf images. The Discrete Wavelet Transform (DWT) decomposes an input image I into four

distinct sub-band components: approximation (A) and details in the horizontal (H), vertical (V ), and diagonal (D) directions. This decomposition is represented by Equation 8.

$$
\operatorname { D W T } ( I ) = \{ A , H , V , D \}\tag{8}
$$

The approximation component A captures the low-frequency structure of the image. At the same time, H, V , and D represent high-frequency components that preserve fine-grained textures and edge details related to disease symptoms, as described in Equation 9a–Equation 9d. These components are computed using convolution operations with wavelet filters, followed by downsampling:

$$
A = ( I * g _ { L } ^ { x } * g _ { L } ^ { y } ) \downarrow 2 ,\tag{9a}
$$

$$
H = ( I * g _ { H } ^ { x } * g _ { L } ^ { y } ) \downarrow 2 ,\tag{9b}
$$

$$
V = ( I * g _ { L } ^ { x } * g _ { H } ^ { y } ) \downarrow 2 ,\tag{9c}
$$

$$
D = ( I * g _ { H } ^ { x } * g _ { H } ^ { y } ) \downarrow 2 .\tag{9d}
$$

here, $g _ { L }$ and $g _ { H }$ are the low-pass and high-pass filters, and ↓ 2 denotes downsampling by a factor of 2. The convolution operator is denoted by ∗. After extraction, the wavelet components are flattened and concatenated into a single feature vector, as shown in Equation 10.

![](images/72386c57f413ad98dad3ca46004938fdbd8543254ce4f77b6484446dd5007812.jpg)  
Figure 2: DWT decomposition of a sugarcane leaf. Low-frequency (LL) captures the global structure; high-frequency sub-bands (LH, HL, HH) highlight textures and edges associated with lesions.

$$
\mathbf { f } _ { \mathrm { w a v e l e t } } = \mathrm { c o n c a t } ( \operatorname { f l a t t e n } ( A ) , \operatorname { f l a t t e n } ( H ) , \operatorname { f l a t t e n } ( V ) , \operatorname { f l a t t e n } ( D ) )\tag{10}
$$

The approximation (A), horizontal (H), vertical (V), and diagonal (D) wavelet sub-bands are each transformed into one-dimensional vectors and subsequently combined to retain diverse information across multiple frequency scales, as shown in Figure 2. The approximation sub-band primarily captures the global shape and structural character istics of the leaf, while the detail sub-bands emphasize directional patterns, texture, and edge information that are commonly associated with disease symptoms. Analysis of individual DWT sub-bands reveals that lesions with fine textures and small streaks produce strong responses in the high-frequency sub-bands (H, V, D). At the same time, overall leaf morphology is captured in the low-frequency sub-band (A). Visualizing these sub-bands demonstrates that DWT can highlight subtle disease patterns that are often missed by spatial-only CNN features, confirming that the transform targets specific frequency ranges associated with early-stage lesions. The resulting concatenated wavelet feature vector provides a comprehensive frequency-domain representation, which is then processed and fused with features extracted using convolutional neural networks (CNNs). This fusion enables the model to automatically learn the relative importance of diferent frequency components during training. Consequently, the system achieves enhanced texture discrimination and more accurate identification of healthy and diseased leaf regions, particularly in the early stages of disease progression. Beyond class activation maps (CAMs), the DWT features provide complementary frequency-domain information that highlights subtle lesion patterns. High-frequency sub-bands (H, V, D) consistently emphasize small spots and streaks that are often missed by spatial-only representations, while the low-frequency sub-band (A) preserves global leaf structure. This additional frequency information improves the model’s discriminative power, particularly for early-stage diseases, and complements CNN-based feature learning for more robust disease detection.

![](images/bbe4e805df70a3a0e549161a6b6a83008df9c09d1c203279947b7d50667329b1.jpg)  
Figure 3: Basic Mean-Teacher model for sugarcane leaf disease detection.

## 4.3.2. Mean-Teacher Model

The training dataset comprises both labeled and unlabeled images within the Mean-Teacher (MT) framework (Tarvainen and Valpola, 2017). Let $x _ { i } ^ { l }$ denote the i-th labeled image with its label $y _ { i } ^ { l } ,$ and $x _ { j } ^ { u }$ denote the j-th unlabeled image. The MT method employs a unified setup between a student and a teacher model to improve learning from unlabeled data samples, as shown in Figure 3. Both models use the EficientNetB4 architecture, which employs compound scaling of depth, width, and resolution to achieve high precision and eficiency, as shown in Figure 4. The student model is trained on both labeled and unlabeled data, while the teacher model is updated using an Exponential Moving Average (EMA) of the student’s weights. The teacher network generates pseudo-labels for unlabeled samples, while the student network aligns its predictions with the teacher’s outputs to encourage stable and generalized training.

In this framework, both the student and teacher models follow the same architecture consisting of an encoder and a decoder. Their parameters are denoted by $\theta _ { s }$ for the student and $\theta _ { t }$ for the teacher. Importantly, only the student model is optimized through gradient descent, whereas the teacher model is updated using an Exponential Moving Average (EMA) and does not receive gradient updates. At iteration $k ,$ the teacher parameters are updated as:

$$
\theta _ { t } ^ { ( k ) } = \lambda \cdot \theta _ { t } ^ { ( k - 1 ) } + ( 1 - \lambda ) \cdot \theta _ { s } ^ { ( k ) }\tag{11}
$$

where, λ is a smoothing factor, typically set to 0.99, controlling the balance between historical and recent information, as shown in Equation 11. In this work, λ is fixed at 0.99 throughout training, and this EMA formulation is used consistently across the framework. Labeled data is used to train the student in a supervised manner, while the teacher provides pseudo-labels for the unlabeled data to guide training. From a computational perspective, the proposed method introduces minimal overhead compared to standard supervised training. With a fixed input resolution, batch size, and training configuration, each training sample requires a constant amount of computation, including forward and backward passes through the student network and a forward pass through the teacher network with EMA-based updates. Therefore, the overall time complexity scales linearly with the number of training samples. Similarly, memory usage increases linearly with the dataset size due to input storage and intermediate feature representations, while the model parameters and EMA-related variables remain constant throughout training.

![](images/ee6661e023c3faeb9ddcb43117dcc9577efdb69c09053da0de5876a8e689ba8a.jpg)  
Figure 4: Architecture of EficientNetB4 model.

## 4.3.3. Pseudo Label Rectification

Pseudo-label rectification addresses the issue of unreliable pseudo-labels by using consistency learning, which encourages the student model to produce predictions that are consistent with those of the teacher model (Lu et al., 2023). The student generates predictions for both labeled and unlabeled samples and learns to match the teacher’s stable outputs, computed using an Exponential Moving Average (EMA) of the student’s weights. This process al lows the student to learn from the teacher’s guidance and improves the robustness to data deviations. The mutual interaction between the student and teacher also supports feature representation and improves the classification performance. The consistency loss $\mathcal { L } _ { \mathrm { c o n s } }$ , defined in Equation 12, incorporates both pointwise and distribution-level alignment.

$$
\mathcal { L } _ { \mathrm { c o n s } } = \mathbb { E } _ { x _ { u } \sim \mathcal { D } _ { u } } \Big [ \ell \big ( f _ { \mathrm { s t u d e n t } } ( x _ { u } ) , f _ { \mathrm { t e a c h e r } } ( x _ { u } ) \big ) \Big ] + \lambda \sum _ { i } p _ { \mathrm { t e a c h e r } } ( x _ { i } ) \log \frac { p _ { \mathrm { t e a c h e r } } ( x _ { i } ) } { p _ { \mathrm { s t u d e n t } } ( x _ { i } ) } ,\tag{12}
$$

Here, $\mathbb { E } _ { x _ { u } \sim \mathcal { D } _ { u } } [ \cdot ]$ denotes the expectation over unlabeled samples $x _ { u } .$ . The functions $f _ { \mathrm { s t u d e n t } } ( x _ { u } )$ and $f _ { \mathrm { t e a c h e r } } ( x _ { u } )$ represent the softmax probability outputs of the student and teacher models, respectively. The first term measures the pointwise diference between predictions using the $L _ { 2 }$ loss, while the second term represents the Kullback– Leibler (KL) divergence between the corresponding probability distributions. Unlike conventional formulations, the KL-divergence is not treated as an independent loss term but is used to estimate prediction uncertainty and guide the consistency learning process. The weighting factor λ, typically set between 0.1 and $1 . 0 ,$ controls the contribution of the KL divergence term. Minimizing $\mathcal { L } _ { \mathrm { c o n s } }$ encourages the student model to align with the teacher model at both the prediction and distribution levels, thereby improving consistency and reducing uncertainty in pseudo-labels.

Selecting an efective backbone is important for achieving high performance in semi-supervised learning, particularly for tasks such as leaf disease detection. ResNet shows strong performance in general image classification. However, it often fails to capture the fine details needed to distinguish between diferent leaf diseases. EficientNet achieves a better balance between accuracy and computational eficiency by scaling network depth, width, and in put resolution together. It also extracts detailed features from sugarcane leaf images, improving classification with fewer parameters and less computation.

## 4.3.4. Total Loss

The overall training objective of the proposed framework combines supervised classification, consistency regularization, and supervised contrastive learning. The total loss used to optimize the student network is defined in Equation 13 as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { s u p } } + \alpha \mathcal { L } _ { \mathrm { c o n s } } + \beta \mathcal { L } _ { \mathrm { S C L } }\tag{13}
$$

where $\mathcal { L } _ { \mathrm { s u p } }$ denotes the supervised classification loss computed on labeled samples, $\mathcal { L } _ { \mathrm { c o n s } }$ represents the consistency loss imposed by the Mean-Teacher framework on unlabeled data, and ${ \mathcal { L } } _ { \mathrm { S C L } }$ is the supervised contrastive loss that promotes inter-class discrimination in the embedding space. The weighting factors α and $\beta$ control the relative contributions of the consistency and contrastive loss terms, respectively. This combined objective enables the model to learn robust, discriminative, and semantically meaningful feature representations.

## 5. Results

In this section, we describe the experimental setup, the datasets used, the evaluation metrics, the qualitative results, and the ablation studies.

## 5.1. Dataset description

The three publicly available datasets used in our study are shown in Figure 5.

![](images/ca34003e74b1e56cad8d7d4d70be0b67fb02d3e32674b115965dbd4e57533506.jpg)

![](images/1fac3d2c3242c039f09a43efe988ce8ed2da72f791d354ab89784595264ee387.jpg)

![](images/8c27e68e435c9e7370e0fabe60c981941b5a963f18324ca08f950070d0e4414a.jpg)  
Figure 5: Distribution of samples across various categories in SLD-1, SLD-2, and SLD-3.

![](images/6dd6293967876f3260cbafcbc6cd728f08982e80ad0e3c75246e0e3b65264736.jpg)  
Figure 6: Illustration of sugarcane leaf disease types on the SLD-1 dataset.

## 5.1.1. SLD-1

This SLD-1 sugarcane dataset consists of 2,521 images of sugarcane leaves collected from farms in Maharashtra, India (Awais et al., 2025; Anika et al., 2025; Singh et al., 2024; Paramanandham et al., 2024; Nomani et al., 2024). The images were captured using smartphones with diferent camera specifications, which introduced variation in original image resolution, lighting conditions, and viewing angles. The SLD-1 dataset contains images with varying native resolutions. This dataset describes five categories: Healthy, Mosaic, Redrot, Rust, and Yellow, as described in Figure 6. All images are in RGB format, and the dataset is balanced across the five classes and provides a good level of variation within and between classes. This makes it suitable for training classification models in real-world conditions.

![](images/aa182fc5e7e7428b1339dcf3eb4a17764e984c8355dcb1a3ca5d1bccc6b17052.jpg)

Figure 7: Overview of the SLD-2 dataset illustrating various sugarcane leaf disease types.  
![](images/aaa20862fde438bf2bd97927ba41f44345dc5892d0e51326ff52043ff3984e65.jpg)  
Figure 8: Overview of sugarcane leaf disease types on the SLD-3 dataset.

## 5.1.2. SLD-2

SLD-2 dataset (Thite et al., 2024) is the largest and comprises 6,748 high-resolution images of sugarcane leaves. The images are in JPEG format with a resolution of 764 × 1024 pixels. The dataset covers 12 categories: 10 disease types, 1 healthy leaf class, and 1 dried leaf class, as shown in Figure 7. The images were collected through extensive field surveys and captured from multiple angles and at diferent disease stages to reflect natural variations. High image quality and wide visual diversity make SLD-2 well-suited for fine-grained disease classification.

## 5.1.3. SLD-3

SLD-3 dataset (Lambor et al., 2022) is smaller and contains 224 RGB images organized into three classes: Healthy, Redrot, and Redrust, as defined in Figure 8. The images were collected from sugarcane fields using the Google Teachable Machine platform, enabling quick image acquisition and labeling. All images in SLD-3 dataset have a fixed original resolution of 224 × 224 pixels.

## 5.2. Experimental Setup

This section provides an overview of comparative methods. A summary of the hyperparameter configurations used in our proposed approach is presented in Table 1. For experimental evaluation, the dataset is divided using a fixed train–test split. In this setup, 10% of the dataset is reserved as an independent test set, while the remaining 90% is used for training the models. The test set is kept strictly unseen during training and used only for final performance evaluation to ensure an unbiased assessment of the model’s generalization ability.

Table 1: This table shows the final hyperparameter details of U-SSDNet.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Image Size</td><td>224×224</td></tr><tr><td>Batch Size</td><td>8</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>0.001</td></tr><tr><td>Number of Epochs</td><td>35</td></tr><tr><td>Weight Decay</td><td>0.0001</td></tr><tr><td>Activation Function</td><td>ReLU</td></tr></table>

## 5.2.1. Comparison Methods

The various state-of-the-art methods used in our study are discussed as: MAIAnet (Zhang et al., 2024), SoyaTrans (Sharma et al., 2025), ResTS (Shah et al., 2022), TSwin-F (Ye et al., 2024), ConvNext (Benchallal et al., 2024), and CorePlant (Ilsever and Baz, 2024).

MAIAnet (Zhang et al., 2024) combines attention mechanisms with normalization layers to improve the precision of cassava leaf disease detection.

SoyaTrans (Sharma et al., 2025) propose a transformer-based model for leaf disease classification, focusing on fine-grained feature extraction in challenging real-world environments.

ResTS (Shah et al., 2022) introduces a residual deep learning approach for detecting plant diseases and improves both model interpretability and performance by focusing on the informative leaf regions.

TSwin-F (Ye et al., 2024) adopts a transformer model with multi-scale feature fusion to identify tomato leaf infections and to capture local characteristics as well as broader spatial relationships.

ConvNext (Benchallal et al., 2024) applies semi-supervised consistency regularization to achieve reliable classification results, even when only a small set of labeled samples is available.

CorePlant (Ilsever and Baz, 2024) uses consistency regularization to combine labeled and unlabeled data, reducing labeling efort while maintaining performance.

## 5.2.2. Evaluation Metrics for Leaf Disease Classification

To assess the performance of our method, we use four standard evaluation metrics: Accuracy (Acc), Precision (Pre), Recall (Rec), and F1-score (F1).

## 5.3. Experimental Results

We present qualitative results comparing state-of-the-art methods with our proposed approach. We also per form ablation analysis to investigate the contribution and efectiveness of individual components in the proposed framework.

## 5.3.1. Comparative Analysis with State-of-the-art Methods

We compare the performance of our proposed method with state-of-the-art methods on three datasets.

(a) On SLD-1 Dataset. Table 2 presents the performance of our proposed U-SSDNet approach compared to stateof-the-art leaf disease classification methods across varying labeled data proportions (10–50%). At all data ratios, U-SSDNet consistently outperforms both supervised and semi-supervised baselines, achieving the highest accuracy and F1-scores, for example, 94.80% and 94.20% at 10% labeled data, and up to 97.83% and 97.42% at 50% labeled data. Among supervised methods, CorePlant (Ilsever and Baz, 2024) shows strong performance at higher labeled data ratios but remains below the proposed approach, while ResTS (Shah et al., 2022) and SoyaTrans (Sharma et al., 2025) improve moderately with more labeled data but lag at lower data proportions. MAIAnet (Zhang et al., 2024) and TSwin-F (Ye et al., 2024) capture leaf disease patterns but achieve only moderate performance, potentially struggling with subtle or complex disease features. The semi-supervised ConvNext approach (Benchallal et al., 2024) leverages unlabeled data, thereby improving its ability to recognize disease patterns compared to fully supervised methods. The superior performance of U-SSDNet stems from its hybrid design, which integrates global context modeling with local feature extraction to capture both long-range dependencies and complex leaf-disease patterns.

Table 2: Quantitative evaluation of supervised and semi-supervised approaches on the SLD-1 dataset using mean ± standard deviation.
<table><tr><td rowspan="2">Approach me</td><td rowspan="2">Model</td><td colspan="2">10%</td><td colspan="2">20%</td><td colspan="2">30%</td><td colspan="2">40%</td><td colspan="2">50%</td></tr><tr><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td></tr><tr><td rowspan="4">Supervised</td><td>MAIAnet (Zhang et al., 2024)</td><td>35.76 ± 1.98</td><td>23.12 ± 1.19</td><td>40.14 ± 2.12</td><td>26.62 ± 1.22</td><td>47.08 ± 1.63</td><td>42.53 ± 1.34</td><td>53.15 ± 1.57</td><td>49.96 ± 1.68</td><td>54.96 ± 1.89</td><td>50.96 ± 1.02</td></tr><tr><td>SoyaTrans (Sharma et al., 2025)</td><td>56.18 ± 1.66</td><td>54.69 ± 1.81</td><td>70.31 ± 1.57</td><td>69.38 ± 1.56</td><td>72.61 ± 1.90</td><td>69.45 ± 1.82</td><td>76.78 ± 2.01</td><td>75.91 ± 1.61</td><td>79.55 ± 2.72</td><td>78.14 ± 2.89</td></tr><tr><td>ResTS (Shah et al., 2022)</td><td>43.41 ± 1.85</td><td>38.23 ± 0.92</td><td>59.32 ± 0.64</td><td>56.39 ± 1.03</td><td>64.91 ± 1.57</td><td>63.01 ± 1.62</td><td>76.02 ± 1.85</td><td>74.94 ± 1.21</td><td>83.88 ± 1.38</td><td>82.03 ± 1.68</td></tr><tr><td>TSwin-F (Ye et al., 2024)</td><td>20.74 ± 1.86</td><td>12.42 ± 2.15</td><td>29.65 ± 1.43</td><td>22.82 ± 1.59</td><td>32.74 ± 1.71</td><td>23.61 ± 1.59</td><td>34.16 ± 1.27</td><td>27.37 ± 1.15</td><td>34.85 ± 1.72</td><td>28.93 ± 1.80</td></tr><tr><td rowspan="2">Semi-supervised</td><td>ConvNext (Benchallal et al., 2024)</td><td>49.37 ± 1.78</td><td>47.49 ± 1.05</td><td>55.20 ± 1.94</td><td>54.48 ± 1.98</td><td>61.28 ± 1.74</td><td>60.44 ± 0.83</td><td>64.89 ± 1.83</td><td>64.45 ± 0.90</td><td>69.14 ± 1.29</td><td>68.81 ± 0.80</td></tr><tr><td>CorePlant (Ilsever and Baz, 2024)</td><td>64.82 ± 2.46</td><td>58.02 ± 2.39</td><td>78.95 ± 1.89</td><td>77.80 ± 1.73</td><td>85.17 ± 2.41</td><td>84.90 ± 2.39</td><td>87.49 ± 1.82</td><td>86.37 ± 1.86</td><td>88.72 ± 1.96</td><td>87.32 ± 2.04</td></tr><tr><td>Proposed</td><td>U-SSDNet</td><td>94.80 ± 1.40</td><td>94.20 ± 1.60</td><td>95.62 ± 0.49</td><td>95.61 ± 0.57</td><td>96.82 ± 0.50</td><td>96.37 ± 0.47</td><td>97.17 ± 1.07</td><td>96.38 ± 1.38</td><td>97.83 ± 1.27</td><td>97.42 ± 1.36</td></tr></table>

(b) On SLD-2 Dataset. For diferent labeled data proportions ranging from 10% to 50%, Table 3 presents quantitative results for the U-SSDNet method on the SLD-2 dataset. The U-SSDNet method consistently performs better in terms of accuracy and F1-score across all labeled data ratios, ranging from 90.87% and 89.88% at 10% labeled data to 95.07% and 94.73% at 50% labeled data. CorePlant (Ilsever and Baz, 2024) performs strongly at higher data proportions but lags behind U-SSDNet at lower labeled data. ResTS (Shah et al., 2022) and SoyaTrans (Sharma et al., 2025) show slow improvement with more training samples and perform worse than the proposed method when there is less labeled data. MAIAnet (Zhang et al., 2024) and TSwin-F (Ye et al., 2024) gain average results and indicate challenges in learning fine disease patterns. Semi-supervised ConvNext (Benchallal et al., 2024) benefits from unlabeled data, but its performance remains below U-SSDNet across all splits. The performance of U-SSDNet is better at capturing complex disease patterns and subtle visual cues, and at robust classification across all leaf disease classes.

Table 3: Quantitative evaluation of supervised and semi-supervised approaches on the SLD-2 dataset using mean ± standard deviation.
<table><tr><td rowspan="2">Approach</td><td rowspan="2">Model</td><td colspan="2">10%</td><td colspan="2">20%</td><td colspan="2">30%</td><td colspan="2">40%</td><td colspan="2">50%</td></tr><tr><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td></tr><tr><td rowspan="4">Supervised</td><td>MAIAnet (Zhang et al., 2024)</td><td>29.56 ± 2.35</td><td>25.76 ± 1.81</td><td>30.34 ± 1.75</td><td>27.35 ± 1.68</td><td>35.69 ± 1.83</td><td>28.57 ± 2.15</td><td>38.54 ± 1.28</td><td>30.27 ± 1.83</td><td>39.73 ± 1.85</td><td>31.35 ± 1.74</td></tr><tr><td>SoyaTrans (Sharma et al., 2025)</td><td>47.25 ± 1.71</td><td>35.82 ± 1.93</td><td>58.03 ± 1.57</td><td>50.22 ± 1.40</td><td>63.25 ± 1.78</td><td>58.02 ± 1.61</td><td>66.94 ± 1.35</td><td>61.44 ± 1.42</td><td>72.64 ± 1.71</td><td>69.73 ± 1.80</td></tr><tr><td>ResTS (Shah et al., 2022)</td><td>48.19 ± 0.69</td><td>43.25 ± 0.75</td><td>60.81 ± 1.37</td><td>56.98 ± 1.58</td><td>72.02 ± 0.81</td><td>70.26 ± 0.79</td><td>80.74 ± 1.51</td><td>78.73 ± 1.40</td><td>86.78 ± 1.93</td><td>85.91 ± 1.78</td></tr><tr><td>TSwin-F (Ye et al., 2024)</td><td>26.82 ± 1.90</td><td>16.74 ± 1.96</td><td>34.16 ± 0.92</td><td>25.14 ± 0.87</td><td>36.87 ± 1.31</td><td>25.85 ± 1.60</td><td>37.37 ± 0.79</td><td>26.18 ± 1.36</td><td>38.19 ± 2.31</td><td>28.30 ± 1.81</td></tr><tr><td rowspan="2">Semi-supervised</td><td>ConvNext (Benchallal et al., 2024)</td><td>63.18 ± 1.94</td><td>59.62 ± 1.80</td><td>69.46 ± 1.59</td><td>62.79 ± 1.67</td><td>75.02 ± 1.47</td><td>72.90 ± 1.41</td><td>79.51 ± 0.93</td><td>73.93 ± 1.13</td><td>83.97 ± 1.81</td><td>81.08 ± 1.85</td></tr><tr><td>CorePlant (Ilsever and Baz, 2024)</td><td>74.68 ± 0.88</td><td>66.15 ± 0.82</td><td>83.72 ± 1.31</td><td>79.35 ± 1.28</td><td>90.22 ± 1.67</td><td>88.02 ± 1.58</td><td>90.74 ± 1.35</td><td>88.13 ± 1.50</td><td>90.87 ± 1.17</td><td>88.64 ± 1.26</td></tr><tr><td>Proposed</td><td>U-SSDNet</td><td>90.87 ± 1.87</td><td>89.88 ± 3.51</td><td>92.85 ± 0.55</td><td>92.95 ± 1.20</td><td>93.05 ± 1.38</td><td>92.74 ± 1.82</td><td>93.79 ± 0.65</td><td>93.46 ± 1.77</td><td>95.07 ± 0.32</td><td>94.73 ± 0.30</td></tr></table>

(c) On SLD-3 Dataset. Table 4 shows the relative performance of the proposed U-SSDNet framework on the SLD-3 dataset against existing supervised and semi-supervised methods. Supervised methods, such as SoyaTrans (Sharma et al., 2025) and ResTS (Shah et al., 2022), show continuous improvements with larger proportions of labeled data but remain below U-SSDNet. MAIAnet (Zhang et al., 2024) and TSwin-F (Ye et al., 2024) achieve lower scores and struggle to capture fine and complex disease patterns. Semi-supervised models such as ConvNext (Benchallal, 2024) and CorePlant (Ilsever, 2024) perform better with more labeled data, but they still do not match the accuracy and consistency ofU-SSDNet. The U-SSDNet shows superior and stable performance for all labeled proportions, reaching up to 98.55% accuracy and 98.91% F1-score, and confirming its robustness for reliable leaf disease classification on the SLD-3 dataset.

## 5.3.2. Comparative Analysis with the Baseline Models

Table 5 and Table 6 present the performance of the proposed U-SSDNet compared with state-of-the-art models on SLD-1 and SLD-2 datasets under diferent labeled data ratios. On the SLD-1 dataset with 50% labeled data, U-SSDNet achieves an F1-score of 97.42%, outperforming classical CNNs such as ResNet-50 (93.67%), DenseNet-121 (93.22%), and VGG-16 (70.14%), as well as transformer-based models including ViT (93.42%) and Swin Transformer (85.66%). These results indicate that U-SSDNet can efectively utilize the available labeled data to achieve high performance.

<table><tr><td rowspan="2">Approach</td><td rowspan="2">Model</td><td colspan="2">10%</td><td colspan="2">20%</td><td colspan="2">30%</td><td colspan="2">40%</td><td colspan="2">50%</td></tr><tr><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td></tr><tr><td rowspan="4">Supervised</td><td>MAIAnet (Zhang et al., 2024)</td><td>37.33 ± 1.76</td><td>34.88 ± 1.50</td><td> $4 2 . 4 7 \pm 2 . 8 5$ </td><td> $4 0 . 7 8 \pm 2 . 5 4$ </td><td> $5 1 . 5 2 \pm 1 . 2 1$ </td><td>48.46 ± 1.22</td><td>52.48 ± 1.53</td><td>50.63 ± 1.48</td><td>55.15 ± 1.64</td><td>53.20 ± 1.58</td></tr><tr><td>SoyaTrans (Sharma et al., 2025)</td><td>52.64 ± 1.49</td><td> $4 8 . 8 3 \pm 1 . 4 0$ </td><td> $5 9 . 6 1 \pm 1 . 3 7$ </td><td> $5 5 . 7 8 \pm 1 . 4 2$ </td><td>66.89 ± 1.87</td><td> $6 5 . 0 2 \pm 1 . 8 9$ </td><td> $7 3 . 7 1 \pm 1 . 0 5$ </td><td>69.92 ± 1.17</td><td>78.06 ± 1.45</td><td> $7 6 . 2 5 \pm 1 . 4 4$ </td></tr><tr><td>ResTS (Shah et al., 2022)</td><td>42.08 ± 1.87</td><td> $3 6 . 9 1 \pm 1 . 5 8$ </td><td>48.59 ± 0.81</td><td>42.36 ± 1.11</td><td>52.71 ± 1.63</td><td> $4 8 . 4 7 \pm 1 . 5 2$ </td><td> $5 7 . 2 1 \pm 1 . 4 7$ </td><td>54.91 ± 1.27</td><td>64.35 ± 1.38</td><td>59.48 ± 1.25</td></tr><tr><td>TSwin-F (Ye et al., 2024)</td><td>32.37 ± 1.81</td><td>20.13 ± 2.09</td><td>36.43 ± 1.88</td><td>24.32 ± 1.60</td><td>37.16 ± 1.73</td><td> $2 6 . 0 1 \pm 1 . 5 9$ </td><td> $4 3 . 1 7 \pm 1 . 8 3$ </td><td>30.14 ± 1.57</td><td> $5 8 . 3 1 \pm 1 . 7 9$ </td><td> $5 0 . 6 3 \pm 2 . 0 8$ </td></tr><tr><td rowspan="2">Semi-supervised</td><td>ConvNext (Benchallal et al., 2024)</td><td>61.78 ± 1.04</td><td>58.79 ± 1.18</td><td>65.94 ± 1.80</td><td> $6 3 . 1 4 \pm 1 . 7 7$ </td><td> $7 0 . 8 3 \pm 1 . 6 9$ </td><td> $6 8 . 8 7 \pm 1 . 6 3 $ </td><td> $\overline { { 8 9 . 2 1 \pm 1 . 3 5 } }$ </td><td> $\overline { { 8 8 . 8 1 \pm 1 . 3 4 } }$ </td><td> $\overline { { 9 2 . 0 7 \pm 1 . 2 5 } }$ </td><td> $9 1 . 9 6 \pm 1 . 4 8$ </td></tr><tr><td>CorePlant (Ilsever and Baz, 2024)</td><td>62.50 ± 1.91</td><td> $5 3 . 8 2 \pm 1 . 8 0$ </td><td>89.91 ± 1.38</td><td>67.46 ± 1.41</td><td> $9 0 . 3 8 \pm 0 . 7 7$ </td><td> $9 0 . 3 6 \pm 0 . 6 7$ </td><td> $9 2 . 7 3 \pm 1 . 2 2$ </td><td> $9 0 . 4 9 \pm 1 . 5 0$ </td><td> $9 2 . 8 8 \pm 1 . 0 3$ </td><td> $9 2 . 0 1 \pm 1 . 4 2 $ </td></tr><tr><td>Proposed</td><td>U-SSDNet</td><td>96.76 ± 1.78 97.10 ± 1.65</td><td></td><td>97.18 ± 1.82</td><td>97.40 ± 1.67</td><td>97.74 ± 1.26 98.80 ± 0.21</td><td></td><td>98.27 ± 1.69 98.87 ± 1.43</td><td></td><td></td><td>98.55 ± 1.47 98.91 ± 1.38</td></tr></table>

When the amount oflabeled data is reduced to 20% and 10%, U-SSDNet still maintains superior performance compared with both conventional networks and semi-supervised methods, including FixMatch, FlexMatch, SoftMatch, and GapMatch. As shown in Table 6 on the SLD-2 dataset, the proposed method achieves F1-scores of94.73%, 92.95%, and 89.88% for 50%, 20%, and 10% labeled data, respectively. This demonstrates that U-SSDNet is robust in scenarios with limited labeled data and consistently outperforms existing approaches.

Table 5: Performance comparison of SOTA models and the proposed U-SSDNet framework on varying labeled data proportions on the SLD-1 dataset.  
Table 4: Quantitative evaluation of supervised and semi-supervised approaches on the SLD-3 dataset using mean ± standard deviation.
<table><tr><td rowspan="2">Model</td><td colspan="4">50% Labeled</td><td colspan="4">20% Labeled</td><td colspan="4">10% Labeled</td></tr><tr><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td></tr><tr><td>ResNet-50 (He et al., 2016)</td><td>93.67</td><td>93.77</td><td>93.67</td><td>93.67</td><td>87.72</td><td>87.87</td><td>87.72</td><td>87.52</td><td>82.97</td><td>84.31</td><td>82.97</td><td>83.03</td></tr><tr><td>VGG-16 (Liu and Deng, 2015)</td><td>70.50</td><td>70.45</td><td>70.50</td><td>70.14</td><td>53.07</td><td>54.27</td><td>53.07</td><td>52.88</td><td>46.93</td><td>48.82</td><td>46.93</td><td>47.03</td></tr><tr><td>DenseNet-121 (Huang et al., 2017)</td><td>93.27</td><td>93.33</td><td>93.27</td><td>93.22</td><td>90.30</td><td>90.22</td><td>90.30</td><td>90.21</td><td>70.20</td><td>70.22</td><td>71.02</td><td>70.22</td></tr><tr><td>ViT (Dosovitskiy et al., 2020)</td><td>93.42</td><td>94.22</td><td>93.42</td><td>93.42</td><td>70.22</td><td>71.82</td><td>70.71</td><td>70.81</td><td>50.02</td><td>49.05</td><td>50.04</td><td>50.55</td></tr><tr><td>Swin Transformer (Liu et al., 2021)</td><td>85.66</td><td>85.45</td><td>85.66</td><td>85.66</td><td>75.52</td><td>77.45</td><td>75.52</td><td>75.66</td><td>45.45</td><td>45.67</td><td>45.45</td><td>45.49</td></tr><tr><td>Mean Teacher (Tarvainen and Valpola, 2017)</td><td>91.52</td><td>92.55</td><td>91.52</td><td>91.98</td><td>89.78</td><td>89.90</td><td>89.78</td><td>89.92</td><td>87.22</td><td>88.32</td><td>87.22</td><td>87.29</td></tr><tr><td>FixMatch (Sohn et al., 2020)</td><td>97.25</td><td>97.46</td><td>97.14</td><td>97.22</td><td>89.80</td><td>89.80</td><td>89.72</td><td>89.69</td><td>87.45</td><td>87.45</td><td>87.32</td><td>87.33</td></tr><tr><td>FlexMatch (Zhang et al., 2021)</td><td>89.02</td><td>90.43</td><td>89.16</td><td>88.89</td><td>88.63</td><td>90.10</td><td>88.81</td><td>88.60</td><td>89.80</td><td>90.47</td><td>89.57</td><td>89.72</td></tr><tr><td>SoftMatch (Chen et al., 2023)</td><td>95.60</td><td>95.78</td><td>95.60</td><td>95.61</td><td>90.00</td><td>90.09</td><td>90.00</td><td>90.01</td><td>79.60</td><td>79.91</td><td>79.60</td><td>79.61</td></tr><tr><td>GapMatch (Huang et al., 2025)</td><td>93.45</td><td>93.70</td><td>93.14</td><td>92.89</td><td>91.90</td><td>90.67</td><td>91.90</td><td>90.13</td><td>90.19</td><td>90.58</td><td>90.19</td><td>90.01</td></tr><tr><td>U-SSDNet (Proposed)</td><td>97.83</td><td>97.91</td><td>97.37</td><td>97.42</td><td>95.62</td><td>95.38</td><td>95.88</td><td>95.61</td><td>94.80</td><td>94.82</td><td>94.88</td><td>94.20</td></tr></table>

Table 6: Performance comparison of SOTA models and the proposed U-SSDNet framework under diferent labeled data proportions on the SLD-2 dataset.
<table><tr><td rowspan="2">Model</td><td colspan="4">50% Labeled</td><td colspan="5">20% Labeled</td><td colspan="3">10% Labeled</td></tr><tr><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td></tr><tr><td>ResNet-50 (He et al., 2016)</td><td>91.48</td><td>91.73</td><td>91.48</td><td>91.51</td><td>92.07</td><td>92.03</td><td>92.07</td><td>91.96</td><td>85.48</td><td>85.18</td><td>85.48</td><td>84.93</td></tr><tr><td>VGG-16 (Liu and Deng, 2015)</td><td>93.93</td><td>94.12</td><td>93.93</td><td>93.95</td><td>91.48</td><td>91.51</td><td>91.48</td><td>91.46</td><td>80.44</td><td>82.73</td><td>80.44</td><td>79.42</td></tr><tr><td>DenseNet-121 (Huang et al., 2017)</td><td>91.93</td><td>92.20</td><td>91.93</td><td>91.89</td><td>88.67</td><td>89.23</td><td>88.67</td><td>88.67</td><td>86.81</td><td>88.24</td><td>86.81</td><td>86.71</td></tr><tr><td>ViT (Dosovitskiy et al., 2020)</td><td>84.59</td><td>85.59</td><td>84.59</td><td>84.67</td><td>74.81</td><td>77.53</td><td>74.81</td><td>75.01</td><td>65.19</td><td>65.72</td><td>65.19</td><td>63.92</td></tr><tr><td>Swin Transformer (Liu et al., 2021)</td><td>92.67</td><td>92.69</td><td>92.67</td><td>92.63</td><td>91.41</td><td>91.42</td><td>91.41</td><td>91.36</td><td>85.56</td><td>85.80</td><td>85.56</td><td>85.27</td></tr><tr><td>Mean Teacher (Tarvainen and Valpola, 2017)</td><td>93.74</td><td>93.84</td><td>93.74</td><td>93.71</td><td>91.80</td><td>91.82</td><td>91.80</td><td>91.65</td><td>83.63</td><td>83.96</td><td>83.63</td><td>83.55</td></tr><tr><td>FixMatch (Sohn et al., 2020)</td><td>94.63</td><td>91.88</td><td>92.30</td><td>91.92</td><td>91.95</td><td>88.66</td><td>89.94</td><td>89.04</td><td>89.12</td><td>86.75</td><td>85.16</td><td>85.65</td></tr><tr><td>FlexMatch (Zhang et al., 2021)</td><td>91.78</td><td>91.04</td><td>90.64</td><td>91.00</td><td>89.87</td><td>86.98</td><td>86.73</td><td>86.05</td><td>86.29</td><td>82.60</td><td>81.42</td><td>80.94</td></tr><tr><td>SoftMatch (Chen et al., 2023)</td><td>94.04</td><td>94.20</td><td>94.04</td><td>93.99</td><td>91.51</td><td>91.99</td><td>91.51</td><td>91.55</td><td>89.12</td><td>89.31</td><td>89.12</td><td>88.98</td></tr><tr><td>GapMatch (Huang et al., 2025)</td><td>94.90</td><td>93.82</td><td>94.90</td><td>93.17</td><td>92.04</td><td>92.54</td><td>92.01</td><td>91.78</td><td>90.26</td><td>89.49</td><td>90.26</td><td>89.21</td></tr><tr><td>U-SSDNet (Proposed)</td><td>95.07</td><td>94.73</td><td>94.95</td><td>94.73</td><td>92.85</td><td>92.98</td><td>92.25</td><td>92.95</td><td>90.87</td><td>90.16</td><td>90.42</td><td>89.88</td></tr></table>

## 5.3.3. Abalation Study

We performed an ablation study in which we removed individual modules and analyzed their impact on the sugarcane leaf disease dataset to evaluate the contributions of each key module in our proposed framework, as shown in Figure 9. Our findings indicate that each component and strategy significantly improves the performance of our method. We precisely analyzed the functions of Leaf Image Preprocessing and Augmentation, the Semi-Supervised Mean-Teacher Model, and Uncertainty Estimation. Section 4 gives a detailed overview of the proposed model pipeline. The ablation study evaluates the impact of diferent feature extraction techniques on the performance of the proposed U-SSDNet model, as defined in Table 9. The analysis shows that EficientNet consistently achieves outstanding results across all evaluation metrics and selects it as the most efective architecture among those examined. DenseNet performs better than ResNet, but still lags behind EficientNet. So the backbone choice is very important for optimizing U-SSDNet, and EficientNet provides the best feature representation.

ImpactofFeature Enhancement. Table 7 shows that diferent feature enhancement techniques afect the performance of the U-SSDNet model. Among these techniques, the Wavelet-based enhancement achieves the best results with the metrics accuracy, precision, recall, and F1-score. The HOG and LPB methods show lower performance, with accuracy gaps of 3.52% and 2.28%, respectively. These results show that Wavelet-based feature enhancement enables the model to capture important structural and textural details of the leaf and to improve classification.

Table 7: Comparative analysis of U-SSDNet model performance across diferent feature enhancement methods. The “dif” columns represent the performance gap relative to the best-performing configuration (Wavelet).
<table><tr><td>Model</td><td>Acc</td><td>diff</td><td>Pre</td><td>diff</td><td>Rec</td><td>diff</td><td>F1</td><td>diff</td></tr><tr><td>U-SSDNet (w/o)</td><td>92.21</td><td>2.59</td><td>92.38</td><td>2.44</td><td>92.98</td><td>1.90</td><td>92.51</td><td>1.69</td></tr><tr><td>U-SSDNet (HOG)</td><td>91.28</td><td>3.52</td><td>91.53</td><td>3.29</td><td>92.11</td><td>2.77</td><td>91.45</td><td>2.75</td></tr><tr><td>U-SSDNet (LPB)</td><td>92.52</td><td>2.28</td><td>92.78</td><td>2.04</td><td>93.19</td><td>1.69</td><td>92.81</td><td>1.39</td></tr><tr><td>U-SSDNet (Wavelet)</td><td>94.80</td><td>0.00</td><td>94.82</td><td>0.00</td><td>94.88</td><td>0.00</td><td>94.20</td><td>0.00</td></tr></table>

Impact ofAugmentation Techniques. Table 8 shows that combining Deep and Cut&Delete augmentations gives the best U-SSDNet performance, achieving the highest accuracy (94.80%), precision (94.82%), recall (94.88%), and F1-score (94.20%). Augmentations perform slightly better, while the model without augmentation shows the weakest results, confirming that combining augmentation techniques enhances model generalization and classification accuracy.

Table 8: Comparative analysis of U-SSDNet model performance across diferent augmentation techniques. The “dif” columns indicate the performance gap relative to the best-performing configuration (Deep+Cut&Delete).
<table><tr><td>Model</td><td>Acc</td><td>diff</td><td>Pre</td><td>diff</td><td>Rec</td><td>diff</td><td>F1</td><td>diff</td></tr><tr><td>U-SSDNet (w/o Aug)</td><td>92.52</td><td>2.28</td><td>92.61</td><td>2.21</td><td>93.02</td><td>1.86</td><td>92.62</td><td>1.58</td></tr><tr><td>U-SSDNet (Deep Aug)</td><td>92.83</td><td>1.97</td><td>93.09</td><td>1.73</td><td>93.31</td><td>1.57</td><td>93.14</td><td>1.06</td></tr><tr><td>U-SSDNet (Cut&amp;Delete Aug)</td><td>93.77</td><td>1.03</td><td>93.84</td><td>0.98</td><td>94.15</td><td>0.73</td><td>93.96</td><td>0.24</td></tr><tr><td>U-SSDNet (Deep+Cut&amp;Delete)</td><td>94.80</td><td>0.00</td><td>94.82</td><td>0.00</td><td>94.88</td><td>0.00</td><td>94.20</td><td>0.00</td></tr></table>

Impact of Feature Extraction Approaches. Table 9 shows that the choice of feature extraction method greatly afects the performance of the U-SSDNet model. The EficientNet backbone achieves the best performance with an accuracy of 94.80%, precision of 94.82%, recall of 94.88%, and an F1-score of 94.20%. while ResNet and DenseNet show considerably lower results with accuracies of 71.34% % and 84.42%. The results show that EficientNet captures more relevant and discriminative features, leading to improved classification performance.

Table 9: Comparative analysis of U-SSDNet model performance across diferent feature extraction methods. The “dif” columns represent the performance gap relative to the best-performing configuration (EficientNet).
<table><tr><td>Model</td><td>Acc</td><td>diff</td><td>Pre</td><td>diff</td><td>Rec</td><td>diff</td><td>F1</td><td>diff</td></tr><tr><td>U-SSDNet (ResNet)</td><td>71.34</td><td>23.46</td><td>59.56</td><td>35.26</td><td>73.04</td><td>21.84</td><td>64.88</td><td>29.32</td></tr><tr><td>U-SSDNet (DenseNet)</td><td>84.42</td><td>10.38</td><td>86.21</td><td>8.61</td><td>85.64</td><td>9.24</td><td>84.12</td><td>10.08</td></tr><tr><td>U-SSDNet (EfficientNet)</td><td>94.80</td><td>0.00</td><td>94.82</td><td>0.00</td><td>94.88</td><td>0.00</td><td>94.20</td><td>0.00</td></tr></table>

Impact ofUncertainty-aware Methods. Table 10 shows that using an uncertainty-aware approach improves the performance of the U-SSDNet model. The models without uncertainty awareness (w/o UA) and the MC-based UA approach achieved lower accuracy than the KL-Divergence method. This result shows that the KL-Divergence method efectively handles prediction uncertainty, improving the model’s stability and precision.

Table 10: Comparative analysis of U-SSDNet performance across diferent uncertainty-aware methods. The “dif” columns represent the performance gap relative to the best-performing configuration (KL-Divergence)
<table><tr><td>Model</td><td>Acc</td><td>diff</td><td>Pre</td><td>diff</td><td>Rec</td><td>diff</td><td>F1</td><td>diff</td></tr><tr><td>U-SSDNet (w/o UA)</td><td>91.28</td><td>3.52</td><td>91.54</td><td>3.28</td><td>92.05</td><td>2.83</td><td>91.55</td><td>2.65</td></tr><tr><td>U-SSDNet (MC)</td><td>92.83</td><td>1.97</td><td>93.45</td><td>1.37</td><td>93.06</td><td>1.82</td><td>93.17</td><td>1.03</td></tr><tr><td>U-SSDNet (KL-Divergence)</td><td>94.80</td><td>0.00</td><td>94.82</td><td>0.00</td><td>94.88</td><td>0.00</td><td>94.20</td><td>0.00</td></tr></table>

![](images/dcc31269e65d2e2bfa82d636fce6508a024f6d260301b9a0497ff7912e246069.jpg)  
(a) Augmentation module

![](images/d125b5d7d6af457f55195d89ac015e88f93ce5189d76d14f4b2a6092b784015b.jpg)  
(b) Feature Encoding module

![](images/f8c90f8d32757cd1b9f71443999f81175b2ee94ced8160df1c6d5a8311c005a4.jpg)  
(c) Feature Extraction module

![](images/5b4e9c31ada615c1c9f68511cf3030143a3d76542a19483c94ad12c5f4050678.jpg)  
(d) Uncertainty Aware module  
Figure 9: Ablation study results on the Sugarcane Leaf Disease (SLD-1) dataset on various components such as: (a) Augmentation module; (b) Feature Encoding module; (c) Feature Extraction module; and (d) Uncertainty-Aware module.

## 5.3.4. Qualitative Analysis

The qualitative evaluation of the U-SSDNet method on the SLD-1 and SLD-2 datasets is shown in Figure 10 and Figure 11, which focus on its prediction behavior across diferent leaf disease classes. The model shows consistently high confidence for most samples and correctly classifies healthy, Redrot, Mosaic, and Rust leaves with good prediction scores. Most samples achieve prediction confidence above 0.9 and indicate the model’s reliability in distinguishing between diferent disease categories. Only a few misclassifications are observed, such as one Yellow leaf sample is incorrectly predicted as Redrot with a confidence of 0.82. But these cases are rare and generated be cause the same feature is contained in two diferent leaf disease classes. These results show that U-SSDNet not only achieves high performance but also produces convincing predictions and maintains its efectiveness for leaf disease classification in SLD-1.

U-SSDNet shows better performance in precisely classifying sugarcane leaf diseases over a wide range of samples. Most of the predictions have confidence scores above 0.95, which shows strong reliability. Leaf diseases such as Brown Spot, Sett Rot, Viral Disease, and Yellow Leaf are accurately predicted and match the actual results. Even for similar diseases such as Banded Chlorosis and Brown Rust, the model maintains confidence scores above 0.75, and the few errors observed have little efect on overall performance. The findings highlight the proposed model’s ability to capture diferent features across patterns in leaf texture, color, and appearance. Consistently confident predictions indicate the model’s reliability.

## 5.3.5. Qualitative Comparison with Existing Methods

Figure 12–Figure 14 present Grad-CAM visualizations for qualitative comparison of diferent state-of-the-art methods on the SLD-1, SLD-2, and SLD-3 datasets using 10% labeled training data. The visualizations emphasize the regions contributing most to model predictions and provide insights into the discriminatory features learned under limited supervision. Compared with U-SSDNet, baseline models exhibit distributed attention patterns that do not consistently align with disease symptoms.

![](images/f7dc0d2f1d47d174eb82d189689b3e348cc6c8fd98f4163c5d08ec9c6570a5e4.jpg)  
Figure 10: Prediction results of sugarcane leaf diseases on the SLD-1 dataset.

Several competing models struggle to distinguish visually similar disease categories, leading to inconsistent localization of disease symptoms in the SLD-1 dataset, as shown in Figure 12. On the SLD-2 dataset, attention maps from baseline approaches frequently extend beyond symptomatic regions, indicating reliance on less informative features, as shown in Figure 13. In contrast, U-SSDNet consistently focuses on disease-relevant regions, producing more localized, semantically meaningful activations that closely align with visible symptoms.

A comparable pattern is observed on the SLD-3 dataset, where subtle inter-class diferences make disease recognition challenging, as shown in Figure 14. While baseline models often generate fragmented attention patterns, U-SSDNet efectively highlights attention on characteristic disease regions, leading to more reliable predictions. The Grad-CAM visualizations show that U-SSDNet focuses more consistently on disease-specific regions than competing methods.

## 5.3.6. Class-wise Performance Assessment

We perform a class-wise evaluation of the model on the SLD-1 and SLD-2 datasets, as shown in Figure 15. This includes confusion matrices and Receiver Operating Characteristic (ROC) curves for each class, with the Area Under the Curve (AUC) used as the main metric. Figure 15(a) and (b) show the confusion matrices for SLD-1 and SLD-2, indicating how well the model correctly classifies each disease and where errors occur. Figure 15(c) and (d) present the corresponding ROC curves, showing the model’s ability to distinguish between classes. High AUC values indicate strong performance, while lower values highlight more challenging classes with similar leaf disease patterns. This class-wise analysis provides a clear view of the model’s performance across all disease categories.

## 6. Discussion

## 6.1. CAM Visualization

Figure 16 displays activation maps produced by four well-known visualization methods, Grad-CAM, Grad-CAM++, Score-CAM, and LIME, to expose the regions most influential in the mean-teacher model’s decisions. For each disease category, Mosaic, RedRot, Rust, and Yellow, the first column shows a representative test image, while the remaining columns display the corresponding activation maps. These techniques show which parts of the image most afect the model’s predictions. Grad-CAM provides a broad overview of discriminative regions, while Grad-CAM++ produces sharper, more precisely localized regions. The visualizations demonstrate that the proposed model consistently focuses on the true disease-afected regions of sugarcane leaves, even when symptoms are subtle or low in intensity. The highlighted regions coincide well with visible lesion patterns and show that the network can detect early signs of infection and recognize fine structural details in Mosaic and RedRot samples. Grad-CAM++ in particular shows clear boundaries around the symptomatic areas, confirming that the learned features are biologically meaningful and not merely correlated background artifacts. Some variability is observed in Rust leaves, where the model occasionally concentrates on only part of the infected area. This reflects the greater dificulty in detecting scattered, small pustules compared to more extensive lesions. However, the overall consistency across the three visualization methods provides strong evidence that the mean-teacher classifier reliably identifies discriminative patterns.

![](images/91e8009b848f11884a3477c23e68b47d68a66c4e0d5ddd47cfec8b28279d9b8c.jpg)  
Figure 11: Prediction results of sugarcane leaf diseases on the SLD-2 dataset.

![](images/9421b6eb06c1e1979c8280147d042ca1dee7c03a6631eb38a095a0b7dffd7cc9.jpg)  
Figure 12: Comparison of Grad-CAM visualizations for diferent methods on the SLD-1 dataset (10% labeled data).

![](images/777663719fe8f4d6f4ce3a1dbccfa418afa845108198d53be224735494172bb2.jpg)  
Figure 14: Comparison of Grad-CAM visualizations for diferent methods on the SLD-3 dataset (10% labeled data).

![](images/c52affd73f0957165cf3503180f5ab2ac784acf1abfc29898c0286fef31b5389.jpg)  
(a)

![](images/f6e6079b3c29793168e0dd87b1c95ec11bf6649b4b988ac95d359eb079ea4c1f.jpg)  
(b)

![](images/27d5782b1669e3b9c98ee936239769f40569e10037044f0be116815b6035ee6a.jpg)  
(c)

![](images/505ff2703a0742954d7d74a4ed2bcff50bb856cdf5b8cc6770501fd017c14144.jpg)  
(d)  
Figure 15: Performance across various classes in the SLD-1 and SLD-2 datasets. Figures (a) and (b) show the confusion matrices of U-SSDNet for SLD-1 and SLD-2, respectively, while Figures (c) and (d) present the corresponding ROC curves for both datasets.

## 6.2. t-SNE Visualization

Figure 17 presents the t-SNE visualization of data representations before and after model training with diferent proportions of labeled leaf samples. The first plot Figure 17(a) shows the data before training. When 10% of the data is labeled Figure 17(b), diferent clusters begin to form, indicating improved organization of the feature space and better classification performance with fewer labeled samples. With 20% labeled data Figure 17(c), the separation between classes becomes more apparent, demonstrating further enhancement in feature discrimination. The clusters become more compact, well separated, and show more powerful feature learning and clear class boundaries as the labeling ratio increases to 30% labeled data (Figure 17(d))–50% labeled data (Figure 17(f)).

![](images/0eae2313cc8f16ffd93c98536827fd9b5f450ea5b86ad29f7f395c7601722150.jpg)  
Figure 16: This figure presents a comparison of CAM and LIME visualizations on the SLD-1 dataset for diferent diseases and shows the efectiveness of the proposed method in identifying disease-afected regions.

![](images/fec0b7a231b5f3d3750c9668ae0e3ac8f1dc1f2a2c5011240887e37b9760473e.jpg)  
(a) Before training

t-SNE After Training  
![](images/35ea7badcd427a57e0297edeec7e763dcdf58f425932af78cf36d715de6b1d48.jpg)  
(b) 10% labeled

![](images/8ffb82d2a54b13f585f2b4f97eed46d1e54df14b4e40f55a63bcdcc0e7db4d33.jpg)  
(c) 20% labeled

![](images/8992c5990a760f7bdaeac2f100dc1ce52a04d6bf463bd4d80b5fce280465a118.jpg)  
(d) 30% labeled

![](images/fdcc5d874bf09a0bca1e6dd94b28f0bf0b5ffdf5dbb8e0fc448d49fed815b905.jpg)  
(e) 40% labeled

![](images/efcf6ab100547a72d0c43170f3903479f8af80be8fb0081d8a913a05be872556.jpg)  
(f) 50% labeled  
Figure 17: t-SNE representation of SLD-1 data with diferent labeling ratios.

## 6.3. Limitation

While the proposed framework shows strong performance in sugarcane leaf disease detection, some limitations remain. Its generalizability to other crops or varied environmental conditions may be limited. The use of unlabeled data can introduce noise, mainly in images with distortions or overlapping symptoms. Moreover, accurately detecting multiple disease levels and distinguishing between visually similar classes remains challenging, thereby

afecting overall classification performance.

## 7. Conclusion

This study presents a U-SSDNet approach for identifying sugarcane leaf diseases that combines the Mean-Teacher framework with wavelet-based preprocessing, KL-divergence-driven pseudo-label correction, and Supervised Contrastive Learning. The presented approach provides benefits for leaf disease identification. It reduces the need for a large annotated dataset by leveraging an unlabeled dataset, making it suitable for real agricultural con ditions. The wavelet-based preprocessing improves lesion visibility and reduces divergence within disease classes. KL-divergence helps correct pseudo-labels, reduces confirmation bias, and yields more accurate predictions. Supervised contrastive learning further improves feature distinction and supports clearer separation between disease classes. An experimental analysis of the sugarcane leaf disease dataset shows that the presented model achieves performance levels comparable to those of fully supervised approaches with fewer labeled samples and outperforms existing methods. The results indicate that the method can serve as a practical and reliable approach for early disease detection and precise monitoring in agriculture.

## References

Ai, M., Zhang, J., Li, P., Wu, J., Tang, Z., Xie, Y., 2025. Semi-supervised contrastive learning for flotation process monitoring with uncertainty-aware prototype optimization. Engineering Applications of Artificial Intelligence 145, 110222. doi:10.1016/j.engappai.2025.110222.

Ali, M.U., Khalid, M., Farrash, M., Lahza, H.F.M., Zafar, A., Kim, S.H., 2024. Appleleafnet: a lightweight and eficient deep learning framework for diagnosing apple leafdiseases. Frontiers in Plant Science 15. doi:10.3389/fpls. 2024.1502314.

Anika, S., Alon, A.S., Alejandro, L.L., Vergara, C.I., Paguigan, J., 2025. Classification and identification of saccharum oficinarum diseases using deep transfer learning computer vision, in: 2025 15th International Conference on Advanced Computer Information Technologies (ACIT), IEEE. p. 927–932. doi:10.1109/acit65614.2025. 11185643.

Awais, M., Salem Abdulla Alharthi, A.H., Kumar, A., Cholakkal, H., Anwer, R.M., 2025. Agrogpt: Eficient agricultural vision-language model with expert tuning, in: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE. p. 5687–5696. doi:10.1109/wacv61041.2025.00555.

Barbedo, J.G.A., 2016. A review on the main challenges in automatic plant disease identification based on visible range images. Biosystems Engineering 144, 52–60. doi:10.1016/j.biosystemseng.2016.01.017.

Barbedo, J.G.A., 2018. Impact ofdataset size and variety on the efectiveness ofdeep learning and transfer learning for plant disease classification. Computers and Electronics in Agriculture 153, 46–53. doi:10.1016/j.compag. 2018.08.013.

Benchallal, F., Hafiane, A., Ragot, N., Canals, R., 2024. Convnext based semi-supervised approach with consistency regularization for weeds classification. Expert Systems with Applications 239, 122222. doi:10.1016/j.eswa. 2023.122222.

Chen, H., Tao, R., Fan, Y., Wang, Y., Wang, J., Schiele, B., Xie, X., Raj, B., Savvides, M., 2023. Softmatch: Addressing the quantity-quality trade-of in semi-supervised learning, in: Proceedings of the International Conference on Learning Representations.

Chen, T., Kornblith, S., Norouzi, M., Hinton, G., 2020. A simple framework for contrastive learning of visual repre sentations, in: International Conference on Machine Learning, PMLR. pp. 1597–1607.

Dangi, S., Mullapudi, S.K., Raghaw, C.S., Dar, S.S., Rehman, M.Z.U., Kumar, N., 2025. A multi-temporal multi-spectral attention-augmented deep convolution neural network with contrastive learning for crop yield prediction. Com puters and Electronics in Agriculture 239, 110895. doi:10.1016/j.compag.2025.110895.

Dey, B., Masum Ul Haque, M., Khatun, R., Ahmed, R., 2022. Comparative performance of four cnn-based deep learning variants in detecting hispa pest, two fungal diseases, and npk deficiency symptoms of rice (oryza sativa). Computers and Electronics in Agriculture 202, 107340. doi:10.1016/j.compag.2022.107340.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N., 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 .

Dyrmann, M., Karstoft, H., Midtiby, H.S., 2016. Plant species classification using deep convolutional neural network. Biosystems Engineering 151, 72–80. doi:10.1016/j.biosystemseng.2016.08.024.

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, in: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), IEEE. p. 770–778. doi:10.1109/cvpr.2016.90.

Hu, G., Zhao, Y., Bao, W., Zhang, D., 2025. A semi-supervised detection method for multi-scale tea geometrid by integrating intra- and inter-class information in natural scene images. Computers and Electronics in Agriculture 237, 110645. doi:10.1016/j.compag.2025.110645.

Huang, G., Liu, Z., Van Der Maaten, L., Weinberger, K.Q., 2017. Densely connected convolutional networks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 4700–4708.

Huang, W., Zhang, L., Wang, Z., Wang, Y., 2025. Gapmatch: Bridging instance and model perturbations for enhanced semi-supervised medical image segmentation. Proceedings of the AAAI Conference on Artificial Intelligence 39, 17458–17466. doi:10.1609/aaai.v39i16.33919.

Ilsever, M., Baz, I., 2024. Consistency regularization based semi-supervised plant disease recognition. Smart Agricultural Technology 9, 100613. doi:10.1016/j.atech.2024.100613.

Kavala, S.A.P.N., Pothuraju, R., 2022. Detection of grape leaf disease using transfer learning methods: Vgg16 &amp; vgg19, in: 2022 6th International Conference on Computing Methodologies and Communication (ICCMC), IEEE. pp. 1205–1208. doi:10.1109/iccmc53470.2022.9753773.

Lambor, S., Pungliya, V., Bhonsle, R., Purohit, A., Raut, A., Patel, A., 2022. Sugarcane leaf disease classification using transfer learning, in: 2022 IEEE Conference on Interdisciplinary Approaches in Technology and Management for Social Innovation (IATMSI), IEEE. p. 1–4. doi:10.1109/iatmsi56455.2022.10119309.

Liu, G., Peng, J., El-Latif, A.A.A., 2022. Sk-mobilenet: A lightweight adaptive network based on complex deep transfer learning for plant disease recognition. Arabian Journal for Science and Engineering 48, 1661–1675. doi:10. 1007/s13369-022-06987-z.

Liu, H., Li, Z., Wu, J., Zeng, K., Hu, R., Zeng, W., 2024. Ascl: Accelerating semi-supervised learning via contrastive learning. Concurrency and Computation: Practice and Experience 36. doi:10.1002/cpe.8293.

Liu, S., Deng, W., 2015. Very deep convolutional neural network based image classification using small training sample size, in: 2015 3rd IAPR Asian Conference on Pattern Recognition (ACPR), IEEE. p. 730–734. doi:10. 1109/acpr.2015.7486599.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B., 2021. Swin transformer: Hierarchical vision transformer using shifted windows, in: 2021 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE. p. 9992–10002. doi:10.1109/iccv48922.2021.00986.

Lu, L., Yin, M., Fu, L., Yang, F., 2023. Uncertainty-aware pseudo-label and consistency for semi-supervised medical image segmentation. Biomedical Signal Processing and Control 79, 104203. doi:10.1016/j.bspc.2022. 104203.

Lu, Y., Young, S., 2020. A survey of public datasets for computer vision tasks in precision agriculture. Computers and Electronics in Agriculture 178, 105760. doi:10.1016/j.compag.2020.105760.

Mia, M.R., Roy, S., Das, S.K., Rahman, M.A., 2020. Mango leaf disease recognition using neural network and support vector machine. Iran Journal of Computer Science 3, 185–193. doi:10.1007/s42044-020-00057-z.

Mohanty, S.P., Hughes, D.P., Salathé, M., 2016. Using deep learning for image-based plant disease detection. Frontiers in Plant Science 7. doi:10.3389/fpls.2016.01419.

Naveed, H., Anwar, S., Hayat, M., Javed, K., Mian, A., 2024. Survey: Image mixing and deleting for data augmentation. Engineering Applications of Artificial Intelligence 131, 107791. doi:10.1016/j.engappai.2023. 107791.

Nomani, K.M.S., Nuruzzaman, M., Nadia, A., Billal, M.M., 2024. Sugarcane leaf disease detection: A comparative analysis using deep learning, in: Proceedings of the 3rd International Conference on Computing Advancements, ACM. p. 139–144. doi:10.1145/3723178.3723197.

Panigrahi, K.P., Das, H., Sahoo, A.K., Moharana, S.C., 2020. Maize leaf disease detection and classification using machine learning algorithms, in: Das, H., Pattnaik, P.K., Rautaray, S.S., Li, K.C. (Eds.), Progress in Computing, Analytics and Networking, Springer Singapore, Singapore. pp. 659–669. doi:10.1007/978-981-15-2414-1\_ 66.

Paramanandham, N., Sundhar, S., Priya, P., 2024. Enhancing disease detection with weight initialization and residual connections using leafnet for groundnut leaf diseases. IEEE Access 12, 91511–91526. doi:10.1109/access. 2024.3422311.

Raghaw, C.S., Bhore, P.S., Rehman, M.Z.U., Kumar, N., 2024. An explainable contrastive-based dilated convolutional network with transformer for pediatric pneumonia detection. Applied Soft Computing 167, 112258. doi:10. 1016/j.asoc.2024.112258.

Raghaw, C.S., Parwal, A., Dar, S.S., Darade, P., Kumar, N., 2026. H-semis: Hierarchical fusion of semi and selfsupervised learning for knee osteoarthritis severity grading. Expert Systems with Applications 322, 132279. doi:10.1016/j.eswa.2026.132279.

Rehman, M.Z.U., Raghuvanshi, D., Kumar, N., 2023. Kisanqrs: A deep learning-based automated query-response system for agricultural decision-making. Computers and Electronics in Agriculture 213, 108180. doi:10.1016/ j.compag.2023.108180.

Sari, Y., Baskara, A.R., Wahyuni, R., 2021. Classification ofchili leafdisease using the gray level co-occurrence matrix (glcm) and the support vector machine (svm) methods, in: 2021 Sixth International Conference on Informatics and Computing (ICIC), IEEE. p. 1–4. doi:10.1109/icic54025.2021.9632920.

Shah, D., Trivedi, V., Sheth, V., Shah, A., Chauhan, U., 2022. Rests: Residual deep interpretable architecture for plant disease detection. Information Processing in Agriculture 9, 212–223. doi:10.1016/j.inpa.2021.06.001.

Sharma, V., Tripathi, A.K., Mittal, H., Nkenyereye, L., 2025. Soyatrans: A novel transformer model for fine-grained visual classification of soybean leaf disease diagnosis. Expert Systems with Applications 260, 125385. doi:10. 1016/j.eswa.2024.125385.

Singh, G., Guleria, K., Sharma, S., 2024. Redrot and mosaic sugarcane leaf disease classification using an eficientnet deep learning model, in: 2024 IEEE 3rd World Conference on Applied Intelligence and Computing (AIC), IEEE. p. 1058–1063. doi:10.1109/aic61668.2024.10731071.

Singh, R.K., Tiwari, A., Gupta, R.K., 2022. Deep transfer modeling for classification of maize plant leaf disease. Multimedia Tools and Applications 81, 6051–6067. doi:10.1007/s11042-021-11763-6.

Sohn, K., Berthelot, D., Li, C.L., Zhang, Z., Carlini, N., Cubuk, E.D., Kurakin, A., Zhang, H., Rafel, C., 2020. Fixmatch: Simplifying semi-supervised learning with consistency and confidence, in: Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 596–608.

Strachan, S., Bhuiyan, S.A., Thompson, N., Nguyen, N.T., Ford, R., Shiddiky, M.J., 2022. Latent potential of current plant diagnostics for detection of sugarcane diseases. Current Research in Biotechnology 4, 475–492. doi:10. 1016/j.crbiot.2022.10.002.

Tang, Z., Yang, J., Li, Z., Qi, F., 2020. Grape disease image classification based on lightweight convolution neural networks and channelwise attention. Computers and Electronics in Agriculture 178, 105735. doi:10.1016/j. compag.2020.105735.

Tarvainen, A., Valpola, H., 2017. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Advances in Neural Information Processing Systems 30.

Thite, S., Suryawanshi, Y., Patil, K., Chumchu, P., 2024. Sugarcane leaf dataset: A dataset for disease detection and classification for machine learning applications. Data in Brief 53, 110268. doi:10.1016/j.dib.2024. 110268.

Xu, M., Yoon, S., Fuentes, A., Park, D.S., 2023. A comprehensive survey of image augmentation techniques for deep learning. Pattern Recognition 137, 109347. doi:10.1016/j.patcog.2023.109347.

Ye, Y., Zhou, H., Yu, H., Hu, H., Zhang, G., Hu, J., He, T., 2024. Application of tswin-f network based on multi-scale feature fusion in tomato leaf lesion recognition. Pattern Recognition 156, 110775. doi:10.1016/j.patcog. 2024.110775.

Yilma, G., Dagne, M., Ahmed, M.K., Bellam, R.B., 2025. Attentive self-supervised contrastive learning (ascl) for plant disease classification. Results in Engineering 25, 103922. doi:10.1016/j.rineng.2025.103922.

Zhang, B., Wang, Y., Hou, W., Wu, H., Wang, J., Okumura, M., Shinozaki, T., 2021. Flexmatch: Boosting semisupervised learning with curriculum pseudo labeling, in: Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 18408–18419.

Zhang, J., Zhang, B., Qi, C., Nyalala, I., Mecha, P., Chen, K., Gao, J., 2024. Maianet: Signal modulation in cassava leaf disease classification. Computers and Electronics in Agriculture 225, 109351. doi:10.1016/j.compag. 2024.109351.