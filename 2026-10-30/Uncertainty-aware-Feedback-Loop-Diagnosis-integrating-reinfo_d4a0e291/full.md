# Uncertainty-aware Feedback Loop Diagnosis: integrating reinforcement learning and conformal prediction for reliable wafer defect classification

Aleksandr Dekhovich<sup>∗</sup> TU Delft a.dekhovich@elsevier.com

Carlas Smith TU Delft

Oleg Soloviev TU Delft & Flexible Optical B.V.

## Abstract

Semiconductor manufacturing is a complex, multi-step process that requires both high precision and low operating costs. Therefore, detecting defects in a highvolume setting is an important component of wafer map analysis for reliable production. However, most state-of-the-art deep learning approaches require a full scan of the surface, which hinders the development of high-throughput inspection systems. To address this problem, we propose Uncertainty-aware Feedback Loop Diagnosis (UFLD), a novel algorithm that employs reinforcement learning for adaptive measurement acquisition and provides a prediction uncertainty level after each step, stopping the process once sufficient information is obtained. Unlike existing methods that rely on ensemble models or auxiliary confidence estimation deep learning models, UFLD utilizes a more parameter-efficient conformal prediction framework to determine the optimal stopping time. The proposed UFLD reduces acquisition time by about 40% and achieves 98.4% inspection accuracy, without requiring an additional learned confidence-estimation network. We validated our approach on the MixedWM38 dataset, demonstrating its advantages over existing adaptive acquisition procedures and standard convolutional neural networks.

## 1 Introduction

The semiconductor industry’s evolution is historically driven by Moore’s Law, the empirical observation that the number of transistors in an integrated circuit doubles approximately every two years [1]. Due to the increasing demand for chips, defect detection plays a crucial role in semiconductor manufacturing, ensuring the quality of chips produced [2]. However, this task is even more challenging in real-time high-volume manufacturing [3], which led to the development of automated defect inspection systems [4, 5]. In the past decade, artificial intelligence (AI), and specifically deep learning (DL), algorithms have achieved remarkable progress in a range of tasks, including industrial applications [3, 6, 7]. A significant portion of these works aims to develop vision-based defect detection systems to analyze the surface of interest by classifying [8, 9], segmenting [10, 11], or detecting [12, 13] possible defective regions.

An efficient inspection system should be accurate, reliable and fast at the same time. Therefore, despite the success of convolutional neural networks (CNNs) in classification problems, CNN-based solutions typically require a full scan of the wafer. In modern high-volume manufacturing, collecting high-resolution data for every die is expensive and time-consuming. Consequently, there is a need for alternative approaches employing adaptive data acquisition (ADA) strategies that can maintain high classification accuracy while minimizing the number of scanned regions.

![](images/69828b889433958dc67da7f728eb8d2960de23d8e45835572f5e435886d19320.jpg)  
Figure 1: An overview of UFLD: the Causal Transformer predicts the defect type from the available patches; conformal prediction evaluates whether the predicted label is in the prediction set. If yes, the scanning process stops; otherwise, the policy network selects the next most informative patch, continuing the feedback loop.

To address this, we propose Uncertainty-aware Feedback Loop Diagnosis (UFLD) that uses a reinforcement learning (RL) strategy for ADA and a conformal prediction-based framework [14] to evaluate the trustworthiness of the prediction after every iteration of the acquisition process to mitigate the limitations of the previous works [15, 16]. Figure 1 illustrates the algorithm: the Causal Transformer plays the role of the classifier to predict the defect pattern from the available observations, while the RL-based policy network adaptively selects the next region for scanning. After each step, a simple threshold-based rule determines whether the acquisition process can be stopped. The main contributions of our work can be summarized as follows:

• We developed a feedback loop algorithm to increase the throughput of inspection tools, while maintaining the performance level of CNN models that require full scanning. The code implementation of the work is publicly available on GitHub: https://github.com/ adekhovich/UFLD.

• In contrast to previous works on sequential wafer inspection where RNN-based classifiers were used, we utilize a causal transformer-based model that has better predictive capabilities. A single transformer model achieves performance comparable to an ensemble of RNNs, while requiring fewer training parameters.

• We introduce a novel stopping rule for the adaptive data acquisition process based on the conformal prediction framework. Our approach reduces algorithmic complexity compared to previous works by replacing an auxiliary confidence-estimator network with a simple threshold-based rule without sacrificing prediction quality.

The rest of the article is organized as follows: in Section 2, we provide the necessary background for the work, Section 3 describes the proposed feedback loop algorithm, while Section 4 shows the results and compares with other methods, and Section 5 concludes the article.

## 2 Related work

Wafer defect recognition using deep learning Deep learning (DL) models are widely used in wafer failure pattern detection [17, 18, 19]. Convolutional neural networks (CNNs) have been trained under different scenarios, namely zero- and few-shot learning [20, 21] to meet the real-world data scarcity constraints, as well as transfer learning [22, 23] and supervised contrastive learning [24] for better generalization. In addition, some works consider unsupervised learning tasks for wafer map analysis with CNNs [25]. For instance, the active cluster framework [26] has been proposed for annotating unlabeled wafer maps. Furthermore, Nakazawa & Kulkarni [17] developed a CNN-based encoder–decoder model for abnormal defect pattern detection and segmentation of previously unseen real wafer maps.

Transformer-based architectures have been successfully used in defect classification tasks [27, 28]. By leveraging self-attention layers, ViTs [29] often generalize better than CNNs; however, they may require more training data for convergence [30]. Therefore, fine-tuning of the pretrained ViT is a better strategy than training a model from scratch [31, 32]. Recently, Liao et al. proposed WaferDSR-RAG [33], which uses retrieval-augmented generation (RAG) [34]. In WaferDSR-RAG, this technique is used for an LLM with a specialized semiconductor knowledge base, linking visual defect patterns to historical engineering logs.

In the discussed works, both CNNs and Transformer-based architectures require a full wafer map and do not process partial observations from the surface, which makes them impractical for real-time high-volume manufacturing. Consequently, we further discuss approaches that predict a label by adaptively selecting the most informative regions, which significantly reduces overall scanning time.

Adaptive data acquisition Adaptive data acquisition is a widely used technique to reduce the cost of measurement collection in high-cost processes. In scanning microscopy, Ye et al. [35] introduced a closed-loop denoising system with pixel-wise uncertainty estimation based on conformal prediction, where only the most uncertain regions of an image are adaptively selected for rescanning. Holman et al. [36] developed a grid-less autonomous adaptive sampling method for scanning hyperspectral imaging. By utilizing hybrid sequential design strategies for surrogate model construction, their system identifies regions of steeper physico-chemical gradients and increases sampling density in those areas in real-time. This approach significantly reduces acquisition time compared to traditiona uniform grid sampling while maintaining high fidelity in complex biological samples.

Several works employ reinforcement learning (RL) [37] for adaptive data acquisition [38, 39, 15], modeling the problem as a Markov Decision Process (MDP) defined by the tuple $( S , \mathcal { A } , P , R , \mu , \gamma )$ In this framework, S represents the state space, A the action space, $\textstyle P ( s _ { t + 1 } | s _ { t } , { \dot { a } } _ { t } )$ the transition distribution, $R ( s _ { t } , a _ { t } )$ the reward function, $\mu ( s _ { 0 } )$ the initial state distribution, and $\gamma \in \mathsf { ( 0 , 1 ] }$ the discount factor. The agent’s objective is to learn an optimal policy $\pi : S \times A \to [ 0 , 1 ]$ that maximizes the expected cumulative return, given by $\begin{array} { r } { \mathbb { E } _ { \pi } \left[ \sum _ { t } \bar { \gamma } ^ { t } R ( s _ { t } , \bar { a } _ { t } ) \right] } \end{array}$ , where actions and next states are sampled according to $a _ { t } \sim \pi ( \cdot | s _ { t } )$ and $s _ { t + 1 } \sim P ( \cdot | s _ { t } , a _ { t } )$ , respectively.

For instance, Bakker et al. [40] utilize a policy gradient RL method for adaptive sampling in k-space for MRI reconstruction, while Yen et al. [39] developed Adaptive Sampling for MR (ASMR) for image-based disease detection. However, these methods predefine the number of measurements that are adaptively selected. The recently proposed SupConWI-RL [16] adaptively selects the next patch and stops the process when an auxiliary confidence estimator model signals that the collected information is sufficient for a reliable prediction. As a result, the number of scanned patches is not predefined and depends on how difficult the failure pattern is to classify.

Conformal prediction Conformal prediction (CP) is a data-based, distribution-free technique for constructing prediction sets, using a calibration set $\mathcal { D } _ { \mathrm { c a l i b } } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n _ { \mathrm { c a l i b } } }$ that is different from the train or test set [41]. The key assumption for the coverage guarantee is the exchangeability property. In addition, it is worth noting that the i.i.d. assumption is often used since exchangeability follows from i.i.d., while the opposite is not true. Vanilla CP consists of the following steps:

1. select a non-conformity scoring function s : $X \times Y  [ 0 , 1 ]$

2. calculate non-conformity scores $s _ { 1 } , s _ { 2 } , \ldots , s _ { n _ { \mathrm { c a l i b } } } .$ , where $s _ { i } = s ( x _ { i } , y _ { i } ) , ( x _ { i } , y _ { i } ) \in \mathcal { D } _ { \mathrm { c a l i b } } \mathrm { . }$

![](images/e3d72b88cae4e8e6eed051d6e8ed16cdd2c3db3353d66268ffe89496932d979d.jpg)  
Figure 2: An illustration of the transformer-based classifier: at every time step t it aggregates t embedded patches along with their positional and temporal embedding, followed by L Transformer blocks. Finally, the layer normalization and MLP are applied to obtain the output.

3. calculate the empirical quantile $\begin{array} { r } { \hat { q } _ { \alpha } = \frac { \lceil ( n _ { \mathrm { c a l i b } } + 1 ) ( 1 - \alpha ) \rceil } { n _ { \mathrm { c a l i b } } } } \end{array}$ of the scores $s _ { 1 } , s _ { 2 } , \ldots , s _ { n _ { \mathrm { c a l i b } } }$ , where $1 - \alpha$ is the user-defined coverage level;

4. construct the prediction set for a test point $x _ { \mathrm { t e s t } }$ with a miscoverage level α: $C _ { \alpha } ( x _ { \mathrm { t e s t } } ) = \{ y$ $s ( x _ { \mathrm { t e s t } } , y ) \leq \hat { q } _ { \alpha } \}$

Vovk et al. [14] provide theoretical guarantees for the prediction sets depending on the size of the calibration set $n _ { \mathrm { c a l i b } } ^ { - } \colon \mathbb { P } \big ( \boldsymbol { y } \in C _ { \alpha } ( x _ { \mathrm { t e s t } } \bar { ) } \big ) \geq 1 - \alpha$ , for the true label y of the test point $x _ { \mathrm { t e s t } }$ . Adaptive Prediction Sets (APS) [42, 43] and Conformal Risk Control (CRC) [44] have been proposed as extensions of the vanilla CP. Like vanilla CP, APS guarantees marginal coverage, while its adaptive conformity score allows the prediction-set size to vary with the difficulty of each input. Furthermore, CRC generalizes vanilla CP to the scenario where the output is high-dimensional by controlling an average loss, e.g., an image segmentation task with a binary mask.

## 3 Proposed method: Uncertainty-aware feedback loop diagnosis

Our Uncertainty-aware Feedback Loop Diagnosis (UFLD) framework consists of three core components: a Transformer-based classifier, an RL-based adaptive patch selector, and a conformal prediction-based stopping rule. The goal of UFLD is to optimize the input sequence to minimize the number of measurements. To simulate real-world conditions, we split an input image, a wafer map, into patches that can be treated as dies. Formally, we denote the input as $\mathbf { X } = ( \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { T } )$ where $\mathbf { x } _ { i }$ is a patch. In our algorithm, the RL-based selector iteratively selects the next patch based on prior observations, while the Transformer-based classifier predicts the defect type after each acquisition. The loop continues until the conformal prediction-based stopping rule determines that the classification reaches a reliability threshold.

Transformer-based classifier First, we pretrain our classifier on randomly sampled sequences of patches $\mathbf { x } _ { i _ { 1 } } , \mathbf { x } _ { i _ { 2 } } , \ldots , \mathbf { x } _ { i _ { T } }$ , so the Transformer model can provide accurate predictions across different acquisition trajectories that the RL agent explores during optimization. To achieve this sequence robustness, a standard causal Transformer is augmented with a learnable sequence-step embedding $\mathbf { E } _ { t }$ This embedding encodes the temporal position of a patch within the acquisition sequence, supplementing spatial positional embeddings $\mathbf { E } _ { i _ { t } }$ of the patch on the wafer map. Figure 2 represents the architecture of the classifier. Overall, the formal computations within the model to compute output $\hat { \mathbf { y } } _ { t }$ for $1 \leq t \leq T$ are defined as follows:

$$
\begin{array} { r l } & { \mathbf { z } _ { s } = \mathbb { M } \mathrm { L P } \big ( \mathrm { F l a t t e n } \big ( \mathbf { x } _ { i _ { s } } \big ) \big ) + \mathbf { E } _ { i _ { s } } + \mathbf { E } _ { s } , \quad 1 \leq s \leq t } \\ & { \mathbf { z } _ { 1 : t } ^ { ( 0 ) } : = \big [ \mathbf { z } _ { 1 } \mathbf { z } _ { 2 } \dots \mathbf { z } _ { t } \big ] } \\ & { \mathbf { z } _ { 1 : t } ^ { ( l ) } = \mathrm { T r a n s f o r m e r B l o c k } \Bigg ( \mathbf { z } _ { 1 : t } ^ { ( l - 1 ) } \Bigg ) , \quad l = 1 , 2 , \dots , L , } \\ & { \mathbf { f } _ { t } = \mathrm { L a y e r N o r m } \Bigg ( \mathbf { z } _ { 1 : t } ^ { ( L ) } \Bigg ) } \\ & { \quad \hat { \mathbf { o } } _ { t } = \mathbb { M } \mathrm { L P } \big ( \mathbf { f } _ { t } \big ) . } \end{array}\tag{1}
$$

Since each wafer map can contain up to four defects in the MixedWM38 dataset, we are solving a multi-label classification problem. To address this, we apply multi-hot encoding to the labels, resulting in a target vector $\mathbf { y } ^ { \mathrm { ^ { - } } } \in \{ 0 , 1 \} ^ { C }$ , where C denotes the total number of possible defect classes. Therefore, the output of the classifier is a probability tensor $\mathbf { p } _ { t }$ that is calculated by applying the Sigmoid function to each component of the logit tensor $\hat { \mathbf { o } } _ { t } \colon \mathbf { p } _ { t } = \mathbf { S i g m o i d } \left( \hat { \mathbf { o } } _ { t } \right)$

Then, we obtain the prediction $\hat { \mathbf { y } } _ { t }$ by thresholding the probability tensor $\mathbf { p } _ { t } = ( p _ { t , 1 } , p _ { t , 2 } , . . . , p _ { t , C } )$ $\hat { \mathbf { y } } _ { t } = \left( \mathbf { 1 } \{ p _ { t , 1 } \geq 0 . 5 \} , \mathbf { 1 } \{ p _ { t , 2 } \geq 0 . 5 \} , \ldots , \mathbf { 1 } \{ p _ { t , C } \geq 0 . 5 \} \right)$

To optimize the parameters of the Transformer, we use standard binary cross-entropy loss for each type of defect. Several works have shown the effectiveness of the supervised contrastive learning loss term for the classification task [45], including sequential wafer map defect recognition [24, 11]. Therefore, we include it in the total loss that we optimize:

$$
\mathcal { L } ( \boldsymbol { \theta } _ { \mathrm { c l a s s i f i e r } } ) = \frac { 1 } { | \mathcal { B } _ { \mathrm { a g } } | } \frac { 1 } { T } \sum _ { ( \mathbf { X } , \mathbf { y } ) \in \mathcal { B } _ { \mathrm { a g } } } \sum _ { t = 1 } ^ { T } \bigg ( \mathrm { B C E L o s s } ( \boldsymbol { \theta } _ { \mathrm { c l a s s i f i e r } } ; \mathbf { p } _ { t } , \mathbf { y } ) +  \\  \lambda \cdot \mathrm { S C L o s s } ( \boldsymbol { \theta } _ { \mathrm { c l a s s i f i e r } } ; \left\{ \mathbf { f } _ { t } \right\} _ { i = 1 } ^ { | \mathcal { B } | } , \left\{ \tilde { \mathbf { f } } _ { t } \right\} _ { i = 1 } ^ { | \mathcal { B } | } , \mathbf { y } ) \bigg ) ,
$$

where batch $B _ { \mathrm { a u g } } = T _ { 1 } ( B ) \cup T _ { 2 } ( B )$ is obtained by applying two data augmentations $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ to the original training batch $B , \{ \mathbf { f } _ { t } \} _ { i = 1 } ^ { | B | }$ and $\{ \widetilde { \mathbf { f } } _ { t } \} _ { i = 1 } ^ { | B | }$ are the last feature representations of sequences up to time t of ${ \mathcal { T } } _ { 1 } ( B )$ and $\mathcal { T } _ { 2 } ( B )$ respectively.

RL-based patch selection Once the classifier is trained, our goal is to optimize the acquisition sequence to minimize the number of measurements (scanned patches) required for each wafer map. Therefore, we train a policy network $\pi _ { \theta _ { \mathrm { R I } } }$ that selects the next patch from the set of available candidates. During training, the next action (patch) $a _ { t }$ is sampled from a categorical distribution with parameters $\pi _ { t + 1 } = \pi _ { \theta _ { \mathrm { R L } } } ( \mathbf { s } _ { t } )$ ⊙ m<sub>t</sub>: a<sub>t+1</sub> ∼ Categorica $\lfloor \left( \pi _ { t + 1 } \right)$ , where $\mathbf { \bar { m } } _ { t } \in \{ 0 , 1 \} ^ { T }$ is a binary mask representing the remaining unselected patches and $\mathbf { s } _ { t } = ( \mathbf { f } _ { t } , \mathbf { m } _ { t } )$ ). At initial step $t = 0$ ${ \bf m } _ { 0 } = ( 1 , 1 , \dots , 1 )$ and $\mathbf { \check { f } } _ { 0 } = ( 0 , 0 , \ldots , \mathbf { \check { 0 } } )$ . During inference, the agent follows a deterministic policy by selecting the available action with the maximum probability: $a _ { t } = \operatorname { a r g m a x } \pi _ { t }$ . The reward function is designed as follows: $r _ { t } = \left\{ \begin{array} { l l } { 1 - c _ { \mathrm { R L } } , \quad \mathrm { i f } \ \hat { \mathbf { y } } _ { t } = \mathbf { y } } \\ { - c _ { \mathrm { R L } } , \quad \mathrm { o t h e r w i s e } , } \end{array} \right.$ where $c _ { \mathrm { R L } }$ is a penalty term for every step.

We should note that our reward function is different from the one in [16] because ours only depends on the correctness of the prediction, and not the score of some auxiliary confidence estimator, which may be unreliable or poorly calibrated.

Conformal prediction as stopping rule The main challenge in the proposed acquisition process is determining the stopping time t where the classification model is sufficiently certain to use the prediction $\hat { \mathbf { y } } _ { t }$ . Stopping too early may result in inaccurate predictions, while delaying the stopping point will increase the data collection costs. The conformal prediction framework provides the opportunity to control the coverage rate $1 - \alpha$ . However, the application of conformal prediction to sequential decision-making is complicated by the potential violation of the exchangeability property, which is the fundamental assumption for the calibration and test datasets. In our case, the predictions $\hat { \mathbf { y } } _ { t }$ are computed from partial observations $\mathbf { x } _ { 1 : t }$ . We ensure that the calibration set $\mathcal { D } _ { \mathrm { c a l i b } }$ is processed in the same manner as the test set $\mathcal { D } _ { \mathrm { t e s t } }$ . Therefore, we should calculate the non-conformity scores and, as a result, the threshold $\hat { q } _ { \alpha }$ on the calibration set using trajectories generated by the same policy across the same time steps as the test set.

In our work, we focus on point accuracy, where each component of the predicted label is correct. Therefore, our objective is the per-class certainty in the classifier output $\hat { \mathbf { y } } _ { t , i } , i = 1 , 2 , . . . , C$ at time step t. For a given calibration point $( \mathbf { x } ^ { i } , \mathbf { y } ^ { i } ) \in \mathcal { D } _ { \mathrm { c a l i b } }$ , we define non-conformity scores at time step t as follows:

$$
\begin{array} { r } { S _ { t , j } ^ { i } ( \mathbf { x } ^ { i } , \mathbf { y } ^ { i } ) = \left\{ { 1 - p _ { t , j } ^ { i } , \quad \mathrm { i f } \ y _ { j } ^ { i } = 1 } \atop { p _ { t , j } ^ { i } , } \right. } \end{array}\tag{2}
$$

where $p _ { t , j } ^ { i }$ is the j-th component of the classifier’s output probability distribution $\mathbf { p } _ { t }$ , and $\mathbf { y } ^ { i } = \mathbf { \partial }$ $( y _ { 1 } ^ { i } , y _ { 2 } ^ { i } , \ldots , y _ { C } ^ { i } )$

Our approach relies on two sets of label-wise quantiles $\hat { \mathbf { q } } _ { y } = ( \hat { q } _ { y , 1 } , \hat { q } _ { y , 2 } , \dots , \hat { q } _ { y , C } ) , \ y \in \{ 0 , 1 \}$ for positive and negative classes. These quantiles play the role of thresholds, and we further call them "negative" and "positive" thresholds. These thresholds are computed through the calibration procedure. During inference, for a new test input $\mathbf { x } ,$ we calculate non-conformity scores

$$
\begin{array} { r l } & { S _ { 0 , t , j } ( \mathbf { x } _ { 1 : t } ) = p _ { t , j } , } \\ & { S _ { 1 , t , j } ( \mathbf { x } _ { 1 : t } ) = 1 - p _ { t , j } } \end{array}\tag{3}
$$

after every acquisition step t. We simultaneously track these values for each label $j = 1 , 2 , \dots , C ,$ and our goal is to stop when the prediction set for each label has cardinality 1. Let us denote $E _ { t , j } : = \bar { \{ S _ { 0 , t , j } \leq \hat { q } _ { 0 , j } \hat { \wedge } S _ { 1 , t , j } > \hat { q } _ { 1 , j } \} } \vee \{ S _ { 0 , t , j } > \hat { q } _ { 0 , j } \wedge S _ { 1 , t , j } \leq \hat { q } _ { 1 , j } \}$ for each label $j = 1 , 2 , \dots , C$ Therefore, our requirement of the size of the prediction set for label j is equivalent to $E _ { t , j } = 1$ . We need to stop the acquisition process when we are sure about all the predicted labels, which means that $E _ { t } : = E _ { t , 1 } \wedge E _ { t , 2 } \wedge . . . \wedge E _ { t , C } = 1$ , and, therefore, every event $E _ { t , j }$ is true. This means that the non-conformity score of every label is lower than the corresponding component of precisely one of the thresholds $\hat { \bf q } _ { 0 } , \hat { \bf q } _ { 1 }$ . As a result, we can formulate our RL acquisition process with the conformal prediction stopping rule in Algorithm 1.

Due to the sequential nature of our problem and differences in stopping times for each of the wafers, the quantiles $\hat { \bf q } _ { 0 } , \hat { \bf q } _ { 1 }$ have to be computed from the non-conformity scores obtained from different time steps to better align the calibration setting with inference. However, at first, we do not have a rule to stop the acquisition process and collect the non-conformity scores for calibration. Thus, our calibration process consists of two parts: first, we compute the preliminary quantiles $\hat { \mathbf { q } } _ { 0 } ^ { A } , \hat { \mathbf { q } } _ { 1 } ^ { A }$ using the dataset $\mathcal { D } _ { A } \subset \mathcal { D } _ { \mathrm { c a l i b } }$ and by collecting the non-conformity scores $\mathbf { S } _ { T } ^ { i }$ at the final step T. Second, we recalibrate these quantiles using them as a stopping criterion for the learned acquisition policy π on dataset $\mathcal { D } _ { B } = \bar { \mathcal { D } } _ { \mathrm { c a l i b } } \backslash \mathcal { D } _ { A }$ . For each wafer in ${ \bar { \mathcal { D } } } _ { B } ,$ non-conformity scores are then collected at the induced stopping time and used to compute the refined thresholds $\hat { \bf q } _ { 0 } , \hat { \bf q } _ { 1 }$ . Algorithm 4 combines these stages into the full calibration algorithm. In this way, the calibration process reproduces the inference setting, where predictions may be produced at input-dependent stopping times. Algorithm 2 illustrates the first stage of the calibration process, where we collect the non-conformity scores at time T. In Algorithm 3, we use $\hat { \mathbf { q } } _ { 0 } ^ { A }$ and $\hat { \mathbf { q } } _ { 1 } ^ { A }$ to construct a stopping rule and to collect the non-conformity scores at the end of the scanning process, which can be at an arbitrary time step.

The fact that different calibration samples stop at different acquisition times does not break exchangeability. Conditional on the preliminary thresholds and the acquisition policy, the stopping time is a deterministic function of the observed wafer trajectory. Hence, each calibration score is obtained by applying the same stopping time scoring rule to an exchangeable sample. The resulting stopping time scores are therefore exchangeable, even though their realized stopping times may differ. We note, however, that the resulting coverage guarantee applies directly to predictions evaluated under the stopping rule induced by the preliminary thresholds. When the refined thresholds are subsequently used to modify the stopping rule at inference, the standard split conformal exchangeability argument no longer provides an exact finite-sample coverage guarantee for the resulting adaptive procedure.

Algorithm 1 Inference with adaptive data acquisition and conformal prediction   
1: function PREDICT(transformer classifier ${ \mathcal { F } } _ { : }$ , acquisition policy π, test point x, quantiles $\hat { \bf q } _ { 0 } , \hat { \bf q } _ { 1 } )$   
2: $\hat { \mathbf { y } }  \emptyset$   
3: $\mathbf { \dot { m } } _ { 0 } \gets ( 1 , 1 , \dots , 1 )$   
4: $\mathbf { f } _ { 0 } \gets ( 0 , 0 , \dots , 0 )$   
5: for $t = 0 , 1 , \ldots , T - 1$ do   
6: Initialize current state: $\mathbf { s } _ { t } \gets [ \mathbf { f } _ { t } , \mathbf { m } _ { t } ]$   
7: Predict next patch location: $a _ { t + 1 }  \pi ( \mathbf { s } _ { t } )$   
8: $\mathbf { m } _ { t + 1 } \gets \mathbf { m } _ { t }$   
9: $\mathbf { m } _ { t + 1 } [ a _ { t + 1 } ] \gets 0$   
10: $\mathbf { x } _ { 1 : t + 1 } \gets [ x _ { a _ { 1 } } , x _ { a _ { 2 } } , \ldots , x _ { a _ { t + 1 } } ]$   
11: $\hat { \mathbf { y } } _ { t + 1 } , \mathbf { f } _ { t + 1 } \gets \mathcal { F } ( \mathbf { x } _ { 1 : t + 1 } )$   
12: Compute $\mathbf { S } _ { 0 , t + 1 } \big ( \mathbf { x } _ { 1 : t + 1 } \big )$ and $\mathbf { S } _ { 1 , t + 1 } \big ( \mathbf { x } _ { 1 : t + 1 } \big )$ using (3)   
13: if $E ( \mathbf { \tilde { S } } _ { 0 , t + 1 } , \mathbf { \tilde { S } } _ { 1 , t + 1 } , \hat { \mathbf { q } } _ { 0 } , \hat { \mathbf { q } } _ { 1 } ) = 1$ then   
14: $\hat { \mathbf { y } } \gets \hat { \mathbf { y } } _ { t + 1 }$   
15: break loop   
16: end if   
17: end for   
18: if ${ \hat { \mathbf { y } } } = \emptyset$ then   
19: $\hat { \mathbf { y } } \gets \mathcal { F } ( \mathbf { x } _ { 1 : T } )$   
20: end if   
21: return $\hat { \mathbf { y } }$   
22: end function

```latex
Algorithm 2 Calibrate at time step $T$
1: function CALIBRATE_A(transformer classifier F, calibration dataset $\mathcal { D } _ { A } = \{ ( \mathbf { x } ^ { i } , \mathbf { y } ^ { i } ) \} _ { i = 1 } ^ { n _ { A } }$
miscoverage rate α)
2: Prediction at final step $T \colon \hat { \mathbf { y } } ^ { i } = \mathcal { F } ( \mathbf { x } ^ { i } ) \quad i = 1 , 2 , \ldots , n _ { A }$
3: Compute non-conformity scores $\mathbf { S } _ { y } ^ { i } = ( S _ { y , 1 } ^ { i } , S _ { y , 2 } ^ { i } , \ldots , S _ { y , C } ^ { i } ) , y = 0 , 1$ , using Equation (2)
4: Compute per-class quantiles $\hat { \mathbf { q } } _ { y } ^ { A } = ( \hat { q } _ { y , 1 } ^ { A } , \hat { q } _ { y , 2 } ^ { A } , \dots , \hat { q } _ { y , C } ^ { A } ) ,$ where:
$\begin{array} { r } { \hat { q } _ { y , j } ^ { A } = \mathrm { Q u a n t i l e } \bigg ( \{ S _ { y , j } ^ { i } : y _ { j } ^ { i } = y \} _ { i = 1 } ^ { n _ { \mathrm { A } _ { y , j } } } ; \frac { \lceil ( n _ { \mathrm { A } _ { y , j } } + 1 ) ( 1 - \alpha ) \rceil } { n _ { \mathrm { A } _ { y , j } } } \bigg ) , n _ { \mathrm { A } _ { y , j } } = \left| \left\{ i : y _ { j } ^ { i } = y \right\} \right| } \end{array}$
5: return $\hat { \mathbf { q } } _ { 0 } ^ { A } , \hat { \mathbf { q } } _ { 1 } ^ { A }$
6: end function
```

Remark 3.1 In our setting, the conformal thresholds $\hat { \mathbf { q } } _ { 0 }$ and $\hat { \mathbf { q } } _ { 1 }$ are used as certification thresholds. Therefore,from the definition ofnon-conformity scores in Equation (2) an increase ofα leads to a decrease of $\hat { q } _ { y , j } , y \in \{ 0 , 1 \} , j = 1 , 2 , \ldots , C .$ . As a result, it is harder to certify labels, and more labels remain undecided. Hence, the model requires more patches for scanning. This means that smaller thresholds lead to fewer early certifications, and therefore stopping happens later.

Remark 3.2 In some cases, the condition provided by the conformal prediction framework does not hold for any $t \in \{ 1 , 2 , \ldots , T \}$ This results in the acquisition process continuing until all measurements (patches) have been obtained. Ifthis occursfor a given wafer map, we can utilize a CNN model to predict the defect type. Since all patches are already available, this does not increase the number of scanning steps but can significantly improve inspection performance. A system that works in this manner is referred to as a hybrid model (a similar approach has been used in [16]).

## 4 Experiments

## 4.1 Experimental setup

Dataset We validated our method on the MixedWM38 dataset, which contains around 38,000 images of wafer maps with one, two, three and four defects, as well as normal ones. Each input image in the MixedWM38 dataset has a size of $5 2 \times 5 2$ , which we divide into $T = 1 6 9$ patches of size $4 \times 4$ . Before that, we apply two sets of augmentation transformations, $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ , which are required for the contrastive loss computation. These augmentations include horizontal and vertical flips with probability 0.5, random rotations in the range [−90, 90] degrees, and normalization. Figure 3 shows how many times each defect type occurs in the dataset, illustrating class imbalance and highlighting the need for robust evaluation metrics such as the F1 score.

```latex
Algorithm 3 Recalibrate using policy π
1: function CALIBRATE_B(transformer classifier ${ \overline { { \mathcal { F } } } } ,$ acquisition policy $\pi ,$ calibration dataset
$\mathcal { D } _ { B } = \{ ( \mathbf { x } ^ { i } , \mathbf { y } ^ { i } ) \} _ { i = 1 } ^ { n _ { B } }$ , preliminary quantiles $\hat { \mathbf { q } } _ { 0 } ^ { A } , \hat { \mathbf { q } } _ { 1 } ^ { A }$ , miscoverage rate α)
2: /* Set of "negative" non-conformity scores $^ { * } /$
3: $S _ { 0 } [ j ]  \emptyset , \breve { j } = 1 , 2 , \ldots , C .$
4: /* Set of "positive" non-conformity scores $^ { * }$
5: $S _ { 1 } [ j ]  \emptyset , j = 1 , 2 , \dotsc , C .$
6: for $i = 1 , 2 , \dotsc , n _ { B }$ do
7: $\mathbf { m } _ { 0 } \gets ( 1 , 1 , \dots , 1 )$
8: $\mathbf { f } _ { 0 } \gets ( 0 , 0 , \dots , 0 )$
9: for $t = 0 , 1 , \ldots , T - 1$ do
10: Initialize current state: $\mathbf { s } _ { t } \gets [ \mathbf { f } _ { t } , \mathbf { m } _ { t } ]$
11: Predict next patch location: $a _ { t + 1 }  \pi ( \mathbf { s } _ { t } )$
12: $\mathbf { m } _ { t + 1 } \gets \mathbf { m } _ { t }$
13: $\mathbf { m } _ { t + 1 } [ a _ { t + 1 } ] \gets 0$
14: $\mathbf { x } _ { 1 : t + 1 } ^ { i } \xleftarrow { } [ \stackrel { . } { x } _ { a _ { 1 } } ^ { i } , \stackrel { . } { x } _ { a _ { 2 } } ^ { i } , \dotsc , \stackrel { . } { x } _ { a _ { t + 1 } } ^ { i } ]$
15: $\hat { \mathbf { y } } _ { t + 1 } ^ { i } , \mathbf { f } _ { t + 1 } \gets \mathcal { F } ( \mathbf { x } _ { 1 : t + 1 } ^ { i } )$
16: Compute $\mathbf { S } _ { 0 , t + 1 } ^ { i } ( \mathbf { x } _ { 1 : t + 1 } ^ { i } )$ and $\mathbf { S } _ { 1 , t + 1 } ^ { i } ( \mathbf { x } _ { 1 : t + 1 } ^ { i } )$ using (3)
17: if $E ( \mathbf { S } _ { 0 , t + 1 } ^ { i } , \mathbf { S } _ { 1 , t + 1 } ^ { i } , { \hat { \mathbf { q } } } _ { 0 } ^ { A } , { \hat { \mathbf { q } } } _ { 1 } ^ { A } ) = { \mathrm { 1 ~ } } \mathbf { o r } \ t = T - 1$ then
18: /* Collect non-conformity scores at time $t + 1 ^ { * } /$
19: for $j = 1 , 2 , \dots , C$ do
20: if $y _ { j } ^ { i } = 0$ then
21: $\mathsf { \bar { S } } _ { 0 } [ j ] \gets S _ { 0 } [ j ] \cup \{ S _ { 0 , t + 1 , j } ^ { i } \}$
22: else
23: $S _ { 1 } [ j ]  S _ { 1 } [ j ] \cup \{ S _ { 1 , t + 1 , j } ^ { i } \}$
24: end if
25: end for
26: break
27: end if
28: end for
29: end for
30: $/ { * }$ Compute refined per-class quantiles $\hat { \mathbf { q } } _ { 0 } ^ { B } , \hat { \mathbf { q } } _ { 1 } ^ { B } \ast /$
31: $\begin{array} { r } { \hat { q } _ { y , j } ^ { B } = \mathrm { Q u a n t i l e } \Bigg ( S _ { y } [ j ] ; \frac { \lceil ( n _ { \mathrm { B } _ { y , j } } + 1 ) ( 1 - \alpha ) \rceil } { n _ { \mathrm { B } _ { y , j } } } \Bigg ) , y = 0 , 1 , j = 1 , 2 , \ldots , C . } \end{array}$
32: return $\hat { \mathbf { q } } _ { 0 } ^ { B } , \hat { \mathbf { q } } _ { 1 } ^ { B }$
33: end function
```

Algorithm 4 Calibrate the decision-making thresholds   
1: function CALIBRATE(transformer classifier $\mathcal { F }$ , acquisition policy π, calibration dataset $\mathcal { D } _ { \mathrm { c a l i b } } =$   
$\{ ( \mathbf { x } ^ { i } , \mathbf { y } ^ { i } ) \} _ { i = 1 } ^ { n _ { \mathrm { c a l i b } } }$ , miscoverage rate α)   
2: Split $\bar { \mathcal { D } } _ { \mathrm { c a l i b } }$ into $\mathcal { D } _ { A }$ and $\mathcal { D } _ { B }$   
3: /\* Compute quantiles using final time step $\textrm { T } ^ { * } /$   
4: $\hat { \mathbf { q } } _ { 0 } ^ { A } , \hat { \mathbf { q } } _ { 1 } ^ { A ^ { \bullet } } \gets \dot { \operatorname { C a l i b r a t e } } _ { - } \mathbf { A } ( \check { \mathcal { F } } , \mathcal { D } _ { A } , \alpha )$   
5: /\* Recalibrate quantiles using poli $\mathrm { c y } \pi ^ { * } /$   
6: $\hat { \mathbf { q } } _ { 0 } , \hat { \mathbf { q } } _ { 1 } \gets \mathrm { C a l i b r a t e \_ B } ( \mathcal { F } , \pi , \mathbf { \tilde { \mathcal { D } } } _ { B } , \hat { \mathbf { q } } _ { 0 } ^ { A } , \hat { \mathbf { q } } _ { 1 } ^ { A } , \alpha )$   
7: return $\hat { \bf q } _ { 0 } , \hat { \bf q } _ { 1 }$   
8: end function

Models Overall, we utilize two deep learning models that are trained sequentially: first, we pretrain the classifier—a causal Transformer—on random patch sequences; second, we train an RL-based patch selector that optimizes the acquisition process given the pretrained classifier; finally, we learn the conformal prediction thresholds. Table 1 summarizes how the dataset was split to train each of the components.

![](images/7aa86a2650191e1f4b72723499588c60510c2773b80c344f492230991c2f4691.jpg)  
Figure 3: Number of defects of each type present in the MixedWM38 dataset.

Table 1: Number of wafer maps in each subset that are used for training and evaluation.
<table><tr><td>Dataset</td><td>Purpose</td><td># of images</td></tr><tr><td> $\mathcal { D } _ { 1 }$ </td><td>train transformer</td><td>20,528</td></tr><tr><td> $\mathcal { D } _ { 2 }$ </td><td>learn conformal prediction thresholds  $\hat { \mathbf { q } } _ { 0 } , \hat { \mathbf { q } } _ { 1 }$ </td><td>2,281</td></tr><tr><td> $\mathcal { D } _ { 3 }$ </td><td>train patch selector</td><td>7,603</td></tr><tr><td> $\mathcal { D } _ { 4 }$ </td><td>test both transformer and patch selector</td><td>7,603</td></tr></table>

For the Transformer architecture, we employ three different configurations, which we refer to as small, medium, and large based on the number of learnable parameters. Table 2 highlights the differences among these three configurations. Here, embedding\_dim denotes the dimension of $\mathbf { z } _ { t }$ and $L$ represents the number of Transformer blocks (see Figure 2). Each of the configurations is pretrained for 240 epochs using a learning rate of 0.001. Additionally, we employ KV-caching to speed up the inference stage. More detailed information is available in our code implementation.

To train the RL agent, we utilize the PPO algorithm [46], where both the actor and critic networks are implemented as two-layer MLPs. Additionally, we employ a memory buffer to store transitions and stabilize the learning process. The agent is optimized for 100 iterations using the Adam optimizer with a learning rate of 0.001, the penalty term $c _ { \mathrm { R L } } = 0 . 0 0 5$

As baseline methods, we use SupConWI-RL [16] and its variants. Additionally, we compare our results with the ResNet-34 model [47], a standard convolutional neural network (CNN) baseline. It is worth noting that ResNet-34 requires the full image to be acquired before processing, which makes it less efficient in terms of scanning time. However, it achieves state-of-the-art accuracy on this dataset and serves as an upper-bound benchmark.

Table 2: Difference among the three Transformer configurations: small, medium and large.
<table><tr><td>Capacity</td><td>embedding_dim</td><td> $L$ </td></tr><tr><td>small</td><td>128</td><td>8</td></tr><tr><td>medium</td><td>256</td><td>4</td></tr><tr><td>large</td><td>256</td><td>8</td></tr></table>

Evaluation metrics To validate our model and compare it with others, we focus on Accuracy = $\# \{ i { = } 1 , 2 , { \ldots } , n _ { \mathrm { t e s t } } \colon \mathbf { y } _ { i } { = } \hat { \mathbf { y } } _ { i } \}$ (therefore, an image is correctly classified only if all components of the n<sub>test</sub> output $\hat { \mathbf { y } } \in \{ 0 , 1 \} ^ { C }$ are correctly predicted), $\begin{array} { r } { \mathrm { F 1 } ~ = ~ \frac { \mathrm { 2 \cdot R e c a l l \cdot P r e c i s i o n } } { \mathrm { R e c a l l + P r e c i s i o n } } } \end{array}$ , where Recall $\begin{array} { r l } { = } & { { } \frac { \mathrm { T P } } { \mathrm { T P + F N } } } \end{array}$ and $\begin{array} { r } { \mathrm { { P r e c i s i o n } } = \frac { \mathrm { { T P } } } { \mathrm { { T P } + \mathrm { { F P } } } } } \end{array}$ to measure how well the model predicts despite class imbalance. Here, TP, FP, and FN are the number of true positives, false positives, and false negatives, respectively, and t<sup>∗</sup> is the number of scanning steps.

Our objective is to evaluate the trade-off between performance (accuracy and F1 score) and the scanning time (t<sup>∗</sup>) required by different models. CNNs typically achieve state-of-the-art results by utilizing the full wafer surface. Therefore, our goal is to identify a model that achieves comparable or better accuracy/F1 score while significantly reducing the required scanning time.

## 4.2 Results

First, we compare the LSTM, the proposed Transformer, and CNN models when all information is available, i.e., all patches have been acquired. Table 3 summarizes the test accuracy on the MixedWM38 dataset when a full sequence of measurements is given to the classifier. The Transformer-based architectures achieve higher classification accuracy than the LSTM. It is noteworthy that all three transformer configurations give similar results despite significant differences in the number of parameters. Meanwhile, the CNN outperforms the LSTM and Transformer-based models. Nevertheless, the latter allow for predictions based on partial observations, as we show below. Additionally, the CNN model has significantly more parameters to train and requires more memory to store.

Table 3: Comparison of models in terms of accuracy and number of trainable parameters if all measurements are available to the network. For the Transformer, three different configurations are considered. The results are averaged over five independent runs, providing mean and standard deviation across runs.
<table><tr><td>Model</td><td></td><td>Accuracy (%) # of learnable parameters</td></tr><tr><td>LSTM</td><td> $9 6 . 3 5 \pm 0 . 2 5$ </td><td>1,508,232</td></tr><tr><td rowspan="3">Transformer</td><td> $9 8 . 0 2 \pm 0 . 1 4$ </td><td>1,641,160</td></tr><tr><td> $9 8 . 1 6 \pm 0 . 0 6$ </td><td>3,285,384</td></tr><tr><td> $9 8 . 1 0 \pm 0 . 1 1$ </td><td>6,444,424</td></tr><tr><td>CNN</td><td> $9 8 . 7 9 \pm 0 . 0 9$ </td><td>21,348,168</td></tr></table>

Table 4 shows the results when applying our adaptive acquisition strategy compared to the SupConWI RL variants. Our UFLD achieves significantly higher accuracy while maintaining a comparable scanning rate. Moreover, UFLD requires two deep learning models to be trained—the classifier and the RL agent—which is a more parameter-efficient strategy compared to SupConWI-RL, where the confidence estimator is another black-box deep learning model. Instead, conformal prediction provides a simple, fast, and justified approach as a stopping rule. An additional analysis of the effect of the miscoverage rate α is presented in Section 4.3.

Table 4: Comparison of the proposed UFLD with other methods. For single-model cases, the result are averaged over five independent runs, providing mean and standard deviation across runs.
<table><tr><td colspan="2">Model</td><td>Accuracy (%)</td><td>Scanned patches (%)</td><td># of parameters</td></tr><tr><td>CNN</td><td>ResNet-34</td><td> $9 8 . 7 9 \pm 0 . 0 9$ </td><td>100</td><td>21,348,168</td></tr><tr><td rowspan="4">Non-hybrid</td><td>Single SupConWI-RL</td><td>95.58 ± 0.26</td><td>59.3 ± 2.6</td><td>3,194,554</td></tr><tr><td>Ensemble SupConWI-RL</td><td>96.34</td><td>60.4</td><td>15,972,770</td></tr><tr><td>Single UFLD-large (ours)</td><td>97.88 ± 0.07</td><td>61.4 ± 1.6</td><td>7,174,194</td></tr><tr><td>Ensemble UFLD-small (ours)</td><td>97.82</td><td>59.7</td><td>11,342,650</td></tr><tr><td rowspan="4">Hybrid</td><td>Single SupConWI-RL</td><td>97.74 ± 0.20</td><td>59.3 ± 2.6</td><td>24,542,722</td></tr><tr><td>Ensemble SupConWI-RL</td><td>98.45</td><td>60.4</td><td>37,320,938</td></tr><tr><td>Single UFLD-large (ours)</td><td> $\overline { { 9 8 . 3 8 \pm 0 . 1 0 } }$ </td><td>61.4 ± 1.6</td><td>28,522,362</td></tr><tr><td>Ensemble UFLD-small (ours)</td><td>98.37</td><td>59.7</td><td>32,690,818</td></tr></table>

First, we compare results in the non-hybrid setting (see Remark 3.2), i.e., the case where we do not use CNN even if the whole surface was scanned. UFLD outperforms SupConWI-RL in both single-model and ensemble cases. Moreover, even single UFLD achieves 1.5 p.p. higher accuracy than Ensemble SupConWI-RL, while having half as many trainable parameters.

Accuracy and F1 score by defect  
![](images/ac22926dae2c105d21e9a4bacba415bed21cbf9652337cbef36fc5a500ae25c4.jpg)  
Figure 4: Accuracy and F1 score of UFLD-large calculated by defect and averaged over five runs with 2-sigma error bars.

In the hybrid setting, we do not observe much difference in either accuracy or scanned patches between UFLD variants and Ensemble SupConWI-RL, while the latter requires more trainable parameters. Single SupConWI-RL loses to our UFLD by 0.64 p.p. without significant advantage in scanning time. However, UFLD provides a trade-off between scanning time and accuracy that is controlled by $\alpha ,$ , which we analyze in Section 4.3. This flexibility allows for an adjustment of the inspection policy based on the specific sensitivity requirements of wafer production.

To analyze our results further, we demonstrate that UFLD effectively handles the class imbalance present in the MixedWM38 dataset. Figure 4 shows the accuracy and F1 score for predicting each defect separately. We can see consistency in both metric values across all defect types, which illustrates the robustness of UFLD despite class imbalance. Notably, we observe a slightly lower F1 score for Near\_Full, which can be explained by the very small number of these examples in the dataset.

In addition, we compare the overall number of scanned patches between our UFLD (large) and the Ensemble SupConWI-RL in Figure 5. Despite the equal average percentages of scanned surface (i.e., the number of patches), the behavior is different—UFLD has significantly fewer cases where it cannot stop the scanning process early. Ensemble SupConWI-RL more frequently reaches the maximum budget of patches due to higher uncertainty than UFLD. This hints that the conformal prediction framework provides a more robust and better-calibrated confidence threshold than the auxiliary confidence estimator counterpart.

## 4.3 Sensitivity of conformal prediction to the choice of hyperparameters

We want to show that our approach is robust to the choice of $\alpha ,$ regardless of transformer architecture. Table 5 demonstrates how coverage, percentage of scanned regions, and two types of accuracy change with small changes in the miscoverage level α for three variants of UFLD – small, medium and large. The results show a monotonic increase in accuracy and a decrease in coverage with the increase of α; the number of scanned patches also increases with the increase of α. This is consistent with our two-rule conformal stopping criterion: increasing α decreases the conformal thresholds $\hat { \mathbf { q } } _ { 0 }$ and $\hat { \mathbf { q } } _ { 1 }$ making positive or negative label certification stricter (see Remark 3.1). As a result, fewer labels are certified at early acquisition steps, and the RL policy must inspect more regions before meeting the stopping condition. This longer acquisition process provides the classifier with more information, which leads to higher accuracy. We observe the same behavior across all three UFLD variants, meaning that the stopping mechanism is stable with respect to model architecture. Here we report the empirical micro-averaged coverage at stopping time:

$$
{ \mathrm { C o v e r a g e } } = \frac { 1 } { N C } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { C } { \bf 1 } \{ S _ { i , \tau _ { i } , j } \leq \hat { q } _ { y _ { j } ^ { i } , j } \} ,
$$

where $\tau _ { i }$ is the stopping time of wafer i.

![](images/ed8e4793662f7c8300369a5aa87d0146db10de405b84be0f5ca2c2540d63f4b2.jpg)  
Figure 5: Comparison between Ensemble SupConWI-RL and UFLD (large) in the number of scanned patches.

Table 5: Results of UFLD models of different sizes (1.6M, 3.3M, and 6.4M), varying α from 0.05 to 0.1 with a step size of 0.005.
<table><tr><td>UFLD</td><td></td><td>0.05</td><td>0.055</td><td>0.06</td><td>0.065</td><td>0.07</td><td>0.075</td><td>0.08</td><td>0.085</td><td>0.09</td><td>0.095</td><td>0.1</td></tr><tr><td rowspan="4">small</td><td>Non-hybrid Coverage (%)</td><td>97.44</td><td>97.13</td><td>96.74</td><td>96.36</td><td>95.94</td><td>95.50</td><td>95.03</td><td>94.54</td><td>94.04</td><td>93.60</td><td>93.04</td></tr><tr><td>Scanned (%)</td><td>54.62</td><td>56.57</td><td>58.54</td><td>60.21</td><td>61.75</td><td>63.23</td><td>64.51</td><td>66.00</td><td>67.43</td><td>68.55</td><td>69.69</td></tr><tr><td>Accuracy (%)</td><td>97.34</td><td>97.44</td><td>97.51</td><td>97.56</td><td>97.59</td><td>97.64</td><td>97.67</td><td>97.70</td><td>97.71</td><td>97.72</td><td>97.74</td></tr><tr><td>Hybrid Accuracy (%)</td><td>97.90</td><td>98.06</td><td>98.18</td><td>98.26</td><td>98.31</td><td>98.38</td><td>98.42</td><td>98.46</td><td>98.50</td><td>98.51</td><td>98.55</td></tr><tr><td rowspan="4">medium</td><td>Non-hybrid Coverage (%)</td><td>97.18</td><td>96.79</td><td>96.48</td><td>96.13</td><td>95.72</td><td>95.30</td><td>94.87</td><td>94.50</td><td>94.04</td><td>93.49</td><td>93.01</td></tr><tr><td>Scanned (%)</td><td>55.45</td><td>57.30</td><td>58.68</td><td>60.21</td><td>61.76</td><td>63.25</td><td>64.54</td><td>65.72</td><td>67.08</td><td>68.42</td><td>69.58</td></tr><tr><td>Accuracy (%)</td><td>97.58</td><td>97.68</td><td>97.72</td><td>97.73</td><td>97.78</td><td>97.82</td><td>97.83</td><td>97.85</td><td>97.87</td><td>97.87</td><td>97.88</td></tr><tr><td>Hybrid Accuracy (%)</td><td>98.01</td><td>98.13</td><td>98.20</td><td>98.22</td><td>98.28</td><td>98.33</td><td>98.37</td><td>98.40</td><td>98.44</td><td>98.46</td><td>98.47</td></tr><tr><td rowspan="4">large</td><td>Non-hybrid Coverage (%)</td><td>97.34</td><td>96.97</td><td>96.60</td><td>96.25</td><td>95.85</td><td>95.47</td><td>95.05</td><td>94.60</td><td>94.25</td><td>93.83</td><td>93.34</td></tr><tr><td>Scanned (%)</td><td>56.3</td><td>58.3</td><td>60.0</td><td>61.4</td><td>62.9</td><td>64.2</td><td>65.5</td><td>66.9</td><td>68.0</td><td>69.0</td><td>70.1</td></tr><tr><td>Accuracy (%)</td><td>97.72</td><td>97.80</td><td>97.85</td><td>97.88</td><td>97.91</td><td>97.94</td><td>97.94</td><td>97.96</td><td>97.99</td><td>98.01</td><td>98.03</td></tr><tr><td>Hybrid Accuracy (%)</td><td>98.18</td><td>98.29</td><td>98.34</td><td>98.38</td><td>98.44</td><td>98.48</td><td>98.50</td><td>98.52</td><td>98.55</td><td>98.58</td><td>98.60</td></tr></table>

## 4.4 Ablation study

In addition to the comparison with the baselines, we conduct an ablation study to validate the contribution of each of the components in UFLD. Table 6 summarizes the classification accuracy and F1 score of a Causal Transformer classifier under different configurations. First, we evaluate the classifier that receives 61.5% of the patches using random sequences and RL-based adaptive data acquisition (ADA). This percentage matches the amount of surface that UFLD needs on average for reliable predictions. We perform this comparison to validate the effect of ADA. Second, we consider the configuration where the classifier receives the random sequence of patches but uses the conformal prediction (CP) stopping rule for certainty in the predictions. The third configuration corresponds to the UFLD algorithm, where we utilize both of these techniques.

Overall, we observe a significant performance improvement when we add each of the components: Causal Transformer alone performs the worst with the given data budget, while adding RL-based ADA increases both accuracy (+4.19 p.p.) and F1 score (+0.008), maintaining the same data budget of 61.5%. The classifier with the CP-based stopping rule without input sequence optimization also improves the indicators (+4.49 p.p. for the accuracy and +0.007 for the F1 score) but leads to an increase of $9 \mathrm { p . p }$ . in required patches. Finally, the employment of both ADA and the CPbased stopping rule boosts both accuracy $\left( + 5 . 0 8 \mathrm { p . p . } \right)$ and the F1 score (+0.008) utilizing the same percentage of the wafer surface (61.4%).

Table 6: Ablation study results where the Transformer classifier (large) with or without adaptive data acquisition (ADA) and conformal prediction (CP) stopping rule are considered. The results are averaged over five independent runs.
<table><tr><td>Model</td><td>RL</td><td>CP</td><td>Acc (%)</td><td>F1</td><td>Scanned (%)</td></tr><tr><td>Transformer</td><td>no</td><td>no</td><td>92.80</td><td>0.985</td><td>61.5</td></tr><tr><td>+ ADA</td><td>yes</td><td>no</td><td>96.99</td><td>0.993</td><td>61.5</td></tr><tr><td>+ Stopping rule</td><td>no</td><td>yes</td><td>97.29</td><td>0.992</td><td>70.5</td></tr><tr><td>UFLD</td><td>yes</td><td>yes</td><td>97.88</td><td>0.993</td><td>61.4</td></tr></table>

## 5 Conclusion

In this work, we addressed the wafer map defect classification problem using an adaptive data acquisition approach. We propose UFLD, a reinforcement learning-based algorithm that adaptively selects the next location for the measurement. In order to make the acquisition process more efficient and reduce operational costs, we employ a conformal prediction-based stopping rule. In contrast to previous work in the semiconductor domain, which requires an additional black-box deep learning model for confidence estimation, the conformal prediction framework is more parameter-efficient. In addition, the proposed UFLD uses a causal transformer architecture as a classifier, instead of RNN-based architectures that were utilized in previous works. This results in a 1.5 p.p. increase in diagnostic accuracy, requiring around 60% of the surface, which is the same number of measurements as previous methods. Moreover, single UFLD uses 50% fewer parameters compared to the baselines due to the elimination of the auxiliary confidence estimator network. Therefore, it further closes the gap to CNN-based approaches that require full scanning. Additionally, variations in the miscoverage rate provide a controllable trade-off between inspection performance and scanning time.

In our opinion, the UFLD framework contributes to more reliable and cost-efficient semiconductor manufacturing. By reducing the need for full surface scanning, our approach increases the throughput of inspection tools in high-volume manufacturing settings. Furthermore, conformal prediction provides a principled mechanism for constructing prediction sets per label and defining the stopping criterion through an interpretable miscoverage parameter α, compared to black-box deep learning methods. Future work could improve the current framework by developing more rigorous theoretical guarantees and by exploring extensions of this adaptive acquisition strategy to other problems, such as defect segmentation or surface reconstruction.

## Acknowledgments and Disclosure of Funding

The project is supported by the Chips Joint Undertaking and its members, including the top-up funding by RVO (The Netherlands Enterprise Agency).

## References

[1] Gordon E Moore. Cramming more components onto integrated circuits. Electronics, 38, 1965.

[2] Victor Azamfirei, Foivos Psarommatis, and Yvonne Lagrosen. Application of automation for in-line quality inspection, a zero-defect manufacturing approach. Journal of Manufacturing Systems, 67:1–22, 2023.

[3] Jay Lee, Hossein Davari, Jaskaran Singh, and Vibhor Pandhare. Industrial artificial intelligence for industry 4.0-based manufacturing systems. Manufacturing letters, 18:20–23, 2018.

[4] Paul Sandland. Automated defect inspection: past, present, and future. In Metrology, Inspection, and Process Control for Microlithography XII, volume 3332, pages 296–308. SPIE, 1998.

[5] Mira Park, Jesse S Jin, Sherlock L Au, Suhuai Luo, and Yue Cui. Automated defect inspection systems by pattern recognition. International Journal of Signal Processing, Image Processing and Pattern Recognition, 2(2):31–42, 2009.

[6] Rahul Rai, Manoj Kumar Tiwari, Dmitry Ivanov, and Alexandre Dolgui. Machine learning in manufacturing and industry 4.0 applications. International Journal of Production Research, 59(16):4773–4778, 2021.

[7] Farhan A Alenizi, Shirin Abbasi, Adil Hussein Mohammed, and Amir Masoud Rahmani. The artificial intelligence technologies in industry 4.0: A taxonomy, approaches, and future directions. Computers & Industrial Engineering, 185:109662, 2023.

[8] Guizhong Fu, Peize Sun, Wenbin Zhu, Jiangxin Yang, Yanlong Cao, Michael Ying Yang, and Yanpeng Cao. A deep-learning-based approach for fast and robust steel surface defects classification. Optics and Lasers in Engineering, 121:397–405, 2019.

[9] Swarit Anand Singh and Kaushal A Desai. Automated surface defect detection framework using machine vision and convolutional neural networks. Journal of Intelligent Manufacturing, 34(4):1995–2011, 2023.

[10] Taiheng Liu, Zhaoshui He, Zhijie Lin, Guang-Zhong Cao, Wenqing Su, and Shengli Xie. An adaptive image segmentation network for surface defect detection. IEEE Transactions on Neural Networks and Learning Systems, 35(6):8510–8523, 2022.

[11] Aleksandr Dekhovich and Miguel A Bessa. Continual learning for surface defect segmentation by subnetwork creation and selection. Journal ofIntelligent Manufacturing, 36(5):3051–3065, 2025.

[12] Yuxin Ma, Jiaxing Yin, Feng Huang, and Qipeng Li. Surface defect inspection of industrial products with object detection deep networks: A systematic review. Artificial Intelligence Review, 57(12):333, 2024.

[13] Wei Zhu, Hui Zhang, Chao Zhang, Xiaoyang Zhu, Zhen Guan, and Jiale Jia. Surface defect detection and classification of steel using an efficient swin transformer. Advanced Engineering Informatics, 57:102061, 2023.

[14] Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic learning in a random world. Springer, 2005.

[15] Aleksandr Dekhovich, Oleg Soloviev, and Michel Verhaegen. Sequential wafer map inspection via feedback loop with reinforcement learning. Expert Systems with Applications, 275:126996, 2025.

[16] Aleksandr Dekhovich and Oleg Soloviev. Supconwi-rl: Wafer inspection with reinforcement learning enhanced by supervised contrastive learning. In 2025 IEEE/CVF International Confer ence on Computer Vision Workshops (ICCVW), pages 1396–1405. IEEE, 2025.

[17] Takeshi Nakazawa and Deepak V Kulkarni. Wafer map defect pattern classification and image retrieval using convolutional neural network. IEEE Transactions on Semiconductor Manufacturing, 31(2):309–314, 2018.

[18] Naigong Yu, Huaisheng Chen, Qiao Xu, Mohammad Mehedi Hasan, and Ouattara Sie. Wafer map defect patterns classification based on a lightweight network and data augmentation. CAAI Transactions on Intelligence Technology, 8(3):1029–1042, 2023.

[19] Wei Yuan, Jinda Yan, and Minghao Piao. Position-aware self-supervised learning for wafer map defect pattern recognition. IEEE Transactions on Instrumentation and Measurement, 2025.

[20] Han Kyul Kim and Jaewoong Shim. Generalized zero-shot learning for classifying unseen wafer map patterns. Engineering Applications ofArtificial Intelligence, 133:108476, 2024.

[21] Hao Geng, Fan Yang, Xuan Zeng, and Bei Yu. When wafer failure pattern classification meets few-shot learning and self-supervised learning. In 2021 IEEE/ACM International Conference On Computer Aided Design (ICCAD), pages 1–8. IEEE, 2021.

[22] Jianbo Yu, Zongli Shen, and Shijin Wang. Wafer map defect recognition based on deep transfer learning-based densely connected convolutional network and deep forest. Engineering Applications ofArtificial Intelligence, 105:104387, 2021.

[23] Jia-Hong Chou, Fu-Kwun Wang, and Hsuan-Kai Chen. Adaptive transfer learning with deformable convolution for wafer defect pattern recognition. IEEE Transactions on Semiconductor Manufacturing, 2025.

[24] Youngjae Bae and Seokho Kang. Supervised contrastive learning for wafer map pattern classification. Engineering Applications ofArtificial Intelligence, 126:107154, 2023.

[25] Tongwha Kim and Kamran Behdinan. Advances in machine learning and deep learning applications towards wafer map defect recognition and classification: a review. Journal of Intelligent Manufacturing, 34(8):3215–3247, 2023.

[26] Jaewoong Shim, Seokho Kang, and Sungzoon Cho. Active cluster annotation for wafer map pattern classification in semiconductor manufacturing. Expert Systems with Applications, 183:115429, 2021.

[27] Shu-Kai S Fan and Shang-Hao Chiu. A new vit-based augmentation framework for wafer map defect classification to enhance the resilience of semiconductor supply chains. International Journal ofProduction Economics, 273:109275, 2024.

[28] Shaurya Jain, A Charan Kumari, and K Srinivas. Sdcvit: a transformer-powered hierarchical framework for semiconductor defect detection and classification. Journal of Intelligent Manufacturing, pages 1–31, 2026.

[29] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[30] Samantha Lafrance and Shonda Bernadin. Evaluating visual transformers for wafer defect detection: A feasibility study. In SoutheastCon 2025, pages 1288–1293. IEEE, 2025.

[31] Faisal Mohammad and Duksan Ryu. Semiconductor wafer map defect classification with tiny vision transformers. arXiv preprint arXiv:2504.02494, 2025.

[32] Shouhong Chen, Zhentao Huang, Tao Wang, Xingna Hou, and Jun Ma. Wafer map defect recognition based on multi-scale feature fusion and attention spatial pyramid pooling. Journal ofIntelligent Manufacturing, 36(1):271–284, 2025.

[33] Xinting Liao, Jie Zhang, Youlong Lyu, and Junliang Wang. Wafer defect semantic reasoning via a three-stage retrieval-augmented system in semiconductor manufacturing. Journal of Manufacturing Systems, 85:269–286, 2026.

[34] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.

[35] Cassandra Tong Ye, Jiashu Han, Kunzan Liu, Anastasios Angelopoulos, Linda Griffith, Kristina Monakhova, and Sixian You. Learned, uncertainty-driven adaptive acquisition for photonefficient scanning microscopy. Optics Express, 33(6):12269–12287, 2025.

[36] Elizabeth A Holman, Yuan-Sheng Fang, Liang Chen, Michael DeWeese, Hoi-Ying N Holman, and Paul W Sternberg. Autonomous adaptive data acquisition for scanning hyperspectral imaging. Communications biology, 3(1):684, 2020.

[37] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT Press, Cambridge, 1998.

[38] Jean-Raymond Betterton, Daniel Ratner, Samuel Webb, and Mykel Kochenderfer. Reinforcement learning for adaptive illumination with x-rays. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 328–334. IEEE, 2020.

[39] Chen-Yu Yen, Raghav Singhal, Umang Sharma, Rajesh Ranganath, Sumit Chopra, and Lerrel Pinto. Adaptive sampling of k-space in magnetic resonance for rapid pathology prediction. arXiv preprint arXiv:2406.04318, 2024.

[40] Tim Bakker, Herke van Hoof, and Max Welling. Experimental design for mri by greedy policy search. Advances in Neural Information Processing Systems, 33:18954–18966, 2020.

[41] Glenn Shafer and Vladimir Vovk. A tutorial on conformal prediction. Journal of machine learning research, 9(3), 2008.

[42] Yaniv Romano, Matteo Sesia, and Emmanuel Candes. Classification with valid and adaptive coverage. Advances in neural information processing systems, 33:3581–3591, 2020.

[43] Anastasios Angelopoulos, Stephen Bates, Jitendra Malik, and Michael I Jordan. Uncertainty sets for image classifiers using conformal prediction. arXiv preprint arXiv:2009.14193, 2020.

[44] Anastasios N Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. Conformal risk control. arXiv preprint arXiv:2208.02814, 2022.

[45] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in neural information processing systems, 33:18661–18673, 2020.

[46] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[47] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.