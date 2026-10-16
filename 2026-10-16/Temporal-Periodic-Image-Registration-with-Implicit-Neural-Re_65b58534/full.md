# Temporal Periodic Image Registration with Implicit Neural Representations

Mathias Lowes<sup>1[0009−0003−0002−1645]</sup>, Kristine Aavild

Sørensen<sup>1,2[0000−0003−1434−697X]</sup>, Maxime Sermesant<sup>3[0000−0002−6256−8350]</sup>, and Rasmus R. Paulsen<sup>1[0000−0003−0647−3215]</sup>

<sup>1</sup> Department of Applied Mathematics and Computer Science, Technical University of Denmark, Kgs. Lyngby, Denmark

mmilo@dtu.dk

<sup>2</sup> Novo Nordisk A/S, Copenhagen, Denmark

3 Inria Université Côte d’Azur, Nice, France

Abstract. Implicit Neural Representations (INRs) have recently gained popularity for their ability to model functions using simple and lightweight networks. These methods have also proven efective in deformable image registration. In this work, we extend INRs for deformable image registration to the domain of temporal image registration, focusing on periodic temporal image sequences. Our approach, Temporal-IDIR, optimizes a single INR to model deformations across all frames in a temporal image sequence simultaneously, allowing for self-regularization through its own deformation predictions. To achieve this, we introduce a temporal consistency loss that penalizes discrepancies between direct source-to-target transformations and those traversing intermediate frames. We evaluate our framework on the DIR-LAB dataset, using the target registration error (TRE) between annotated and moved landmarks as the metric. Here, we achieve a TRE of 1.03 mm, outperforming other INR-based registration methods. Additionally, our framework supports smooth interpolation between time frames by estimating deformations between the given input frames.

Keywords: Deformable Image Registration · Implicit Neural Representations · Neural Fields · Deep Learning · 4D CT.

## 1 Introduction

Temporal image registration is a fundamental process in image analysis that involves aligning two or more images taken at diferent times. In medical imaging, temporal registration is useful for analyzing short-term sequences, such as a heart beat, a respiratory cycle, or bowel movements, as well as long-term sequences to investigate disease progression through months or years. Registering the images across the temporal sequence ofers the possibility to propagate landmarks and thereby follow the movement of anatomical structures over time.

![](images/3cdc209e8a3006cabd52befd2153716fddf652dd63ea45c1a3492261ecaf9c8e.jpg)  
Fig. 1: Registration of multiple landmarks from source frame (left) to target frame (right). For each source point x the deformation u(x) is calculated to map the point to the target domain $\Phi _ { S  T }$ (x)

Standard deformable image registration (DIR) is performed pairwise by registering a source image $I _ { S }$ to a target image $I _ { T }$ . This can be achieved by applying a deformable vector field $u ( { \bf x } )$ to map each position in the source image x to align it with the target image (see Figure 1). Registration methods can be divided into two categories; optimization-based and learning-based methods. Traditional methods using B-splines to represent the deformation vector field [11,13,21] are examples of optimization-based methods, where the parameters of the splines are optimized individually for each registration. With convolutional neural networks, a group of learning-based registration methods appeared [1,7,8,15,19,22]. This approach ofers the possibility of learning the registration on one data set and generalize to new unseen data, however using 3D CNNs on large 3D image volumes requires large amounts of memory.

Implicit Neural Representations (INRs) can be used as an optimization-based registration method by parameterizing the deformation field as a neural network [20]. In the simplest form, the source point coordinate x is passed to a simple multi layer perceptron (MLP) that estimates the deformation u(x). This approach significantly lowers the memory requirements by operating on continuous points, while exploiting the neural networks ability to approximate any signal or function. Several works have expanded on the use of INRs by enforcing an inverse consistency between source-to-target and target-to-source registration [2,6,16] or applying geometric constraints [5].

The described methods are developed for pair-wise image registration. When a temporal sequence consists of more than two images, one can utilize pair-wise registration to either register the first image to all subsequent time frames (direct registration) or sequentially register each time frame to the next (sequential registration). Assuming smooth changes over the temporal sequence, the sequential registration will only require small deformations between consecutive time steps, whereas the direct registration may require increasingly larger deformations when the time diference is large. The sequential approach, however, builds up error through the time steps, causing problems in long sequences [12].

In this paper, we propose a novel method for temporal registration using INRs that combines direct and sequential registration, by simultaneously optimizing registration between all possible image pairs in a temporal image sequence. Our main contributions include extending INR-based registration with the temporal dimension, enabling tracking of landmark movement during a whole cycle of data, contrary to only the extreme phases. Furthermore, our method tracks landmarks between the input frames of the data, giving continuous landmark movement throughout the data cycle. We demonstrate the capabilities of the method on the open source DIR-LAB lung CT dataset [3], and show that the proposed method outperforms other INR-based registration methods.

## 2 Methods

Image Registration. Pairwise deformable image registration aims to find a mapping from a source image $I _ { S } : \varOmega \subset \mathbb { R } ^ { 3 } \to \mathbb { R }$ to a target image $I _ { T } : \varOmega \subset$ $\mathbb { R } ^ { 3 } \to \mathbb { R }$ . The mapping $\Phi _ { S  T } : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 3 }$ is expressed as $\Phi _ { S \to T } ( \mathbf { x } ) = u ( \mathbf { x } ) + \mathbf { x }$ where $u ( { \bf x } )$ represents the displacement field at the point x. The goal of the pairwise registration is to satisfy

$$
( I _ { S } \circ \Phi _ { S  T } ) ( \mathbf { x } ) = I _ { T } ( \mathbf { x } ) \mathrm { ~ f o r ~ a l l ~ } \mathbf { x } \in \varOmega .\tag{1}
$$

In scenarios involving temporal image sequences, temporal registration is employed to map images over the given sequence of time points. This can be performed using two approaches:

– Direct registration, aligning a single source frame to all other frames in the sequence e.g. $f _ { 0 }  f _ { 1 } , f _ { 0 }  f _ { 2 } .$ , etc.

– Sequential registration, aligning consecutive frames to each other, such as $f _ { 0 }  f _ { 1 } , f _ { 1 }  f _ { 2 }$ , etc.

To take advantage of both strategies, we propose a unified optimization framework that simultaneously optimizes the registrations between all source-target pairs. In this way, the temporal registration, i.e. the mapping between all source and target frames, Φ, is found by optimizing

$$
\Phi ^ { * } = \underset { \Phi } { \arg \operatorname* { m i n } } \Bigg \{ \sum _ { S , T } \mathcal { L } _ { \sin } ( I _ { S } \circ \Phi _ { S  T } , I _ { T } ) \Bigg \} ,\tag{2}
$$

where $\mathcal { L } _ { \mathrm { s i m } }$ is a similarity loss, chosen as the normalized cross-correlation loss.

Temporal Consistency Loss. To ensure point correspondence over all temporal mappings, we introduce a temporal consistency loss, $\mathcal { L } _ { \mathrm { t e m p o r a l } } ( \Phi _ { S  T } )$ . This loss enforces that a point moved directly from S to $T$ aligns with the same point moved stepwise through intermediate frames from S to $T$ . The loss is defined as

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { t e m p o r a l } } ( \Phi _ { S  T } ) = \| \Phi _ { S  T } ( \mathbf { x } ) - \Phi _ { S  T } ^ { + } ( \mathbf { x } ) \| _ { 2 } } & { } \\ { + \| \Phi _ { S  T } ( \mathbf { x } ) - \Phi _ { S  T } ^ { - } ( \mathbf { x } ) \| _ { 2 } } & { } \\ { + \| \Phi _ { S  T } ^ { + } ( \mathbf { x } ) - \Phi _ { S  T } ^ { - } ( \mathbf { x } ) \| _ { 2 } , } \end{array}\tag{3}
$$

where $\Phi _ { S  T } ^ { + }$ and $\Phi _ { S  T } ^ { - }$ is the sequential movement of point x from $S$ to T in the forward and backwards directions, defined by

$$
\begin{array} { r l } & { \Phi _ { S \to T } ^ { + } ( \mathbf { x } ) = \Phi _ { T - 1 \to T } \circ . . . \circ \Phi _ { S + 1 \to S + 2 } \circ \Phi _ { S \to S + 1 } ( \mathbf { x } ) , } \\ & { \Phi _ { S \to T } ^ { - } ( \mathbf { x } ) = \Phi _ { T + 1 \to T } \circ . . . \circ \Phi _ { S - 1 \to S - 2 } \circ \Phi _ { S \to S - 1 } ( \mathbf { x } ) . } \end{array}\tag{4}
$$

Given the periodic nature of the data sequence, modular indexing is used over the sequence length N, to ensure that $\Phi _ { N - 1  N }$ is equivalent to $\Phi _ { N - 1  0 }$

For regularization, we adopt the symmetric Jacobian regularization loss from van Harten et al. [6] penalizing the Jacobian matrix’s deviation from 1.

$$
\mathcal { L } _ { \mathrm { r e g } } ( \Phi _ { S  T } ) = \mathrm { m i n } \{ \frac { ( \mathrm { d e t } \nabla \Phi _ { S  T } - 1 ) ^ { 2 } } { | \mathrm { d e t } \nabla \Phi _ { S  T } | } , \tau \} ,\tag{5}
$$

where $\nabla \Phi _ { S  T }$ is the Jacobian matrix of $\Phi _ { S  T }$ and τ is a hyperparameter which clips the regularization. As the authors suggest, we use $\tau = 1 0$

With the inclusion of the temporal loss and the regularization term, the registration over a periodic temporal image sequence is found as

$$
\begin{array} { r l } & { \Phi ^ { * } = \underset { \Phi } { \arg \operatorname* { m i n } } \Bigg \{ \underset { S , T } { \sum } \big [ \mathcal { L } _ { \mathrm { s i m } } ( I _ { S } \circ \Phi _ { S  T } , I _ { T } ) + \alpha \mathcal { L } _ { \mathrm { s i m } } ( S _ { S } \circ \Phi _ { S  T } , S _ { T } ) } \\ & { \qquad + \beta \mathcal { L } _ { \mathrm { t e m p o r a l } } ( \Phi _ { S  T } ) + \lambda \mathcal { L } _ { \mathrm { r e g } } ( \Phi _ { S  T } ) \big ] \Bigg \} . } \end{array}\tag{6}
$$

With $\alpha , \beta$ and λ as hyperparameters for the weight of the regularization terms. The signed distance fields (SDFs) of the anatomy of interest (for DIR-LAB the lungs), are included in the optimization and denoted $S _ { S }$ and $S _ { T }$ for the source and target SDFs, respectively.

Implicit Neural Representations. An INR consists of a simple MLP network, trained to approximate the deformation fields between all frames in the data sequence. The INR takes a continuous coordinate x and source and target times as input, while returning the mapping x from source to target, see Fig. 1.

To enable the neural network to capture both low and high frequency signals we use the SIREN architecture [14], an MLP with sinusoidal activation functions. Additionally, the physical input coordinates are encoded with a harmonic encoding, projecting the coordinates into a higher-dimensional space, while the input times are mapped to the unit circle. The encoded physical coordinates, $\gamma ( \mathbf { x } )$ and the encoded time coordinates, ψ(t), are calculated as

$$
\gamma ( \mathbf { x } ) = \left[ \mathbf { x } , \cos ( 2 ^ { 0 } \mathbf { x } ) , \sin ( 2 ^ { 0 } \mathbf { x } ) , \ldots , \cos ( 2 ^ { L - 1 } \mathbf { x } ) , \sin ( 2 ^ { L - 1 } \mathbf { x } ) \right] ,\tag{7}
$$

$$
\psi ( t ) = \left[ \cos \left( \frac { 2 \pi } { N } t \right) , \sin \left( \frac { 2 \pi } { N } t \right) \right] ,\tag{8}
$$

where L is the number of harmonic encodings and N is the cycle length.

Temporal Inference. The registrations are evaluated based on the error between landmark in expiratory phase and landmark moved from inspiratory phase. We employ four methods for mapping from the inspiratory phase $t _ { \mathrm { i n s p } }$ to the expiratory phase $t _ { \mathrm { e x p } }$ . We define $\mathbf { x } _ { \mathrm { i n s p } }$ as the landmark position in the inspiratory phase and $\tilde { \mathbf { x } } _ { \mathrm { e x p } }$ as the estimated moved landmark position after applying the transformation. However, the methods can be used to map from any source time to any target time.

1. The direct method calculates the transformation from $t _ { \mathrm { i n s p } }$ $t _ { \mathrm { e x p } }$ as

$$
\tilde { \mathbf { x } } _ { \mathrm { e x p } } = \Phi _ { t _ { \mathrm { i n s p } }  t _ { \mathrm { e x p } } } ( \mathbf { x } _ { \mathrm { i n s p } } ) .\tag{9}
$$

2. To move the inspiratory landmark sequentially we use the definition from Eq. (4), moving the point one step at a time

$$
\tilde { \mathbf { x } } _ { \mathrm { e x p } } = \Phi _ { t _ { \mathrm { i n s p } }  t _ { \mathrm { e x p } } } ^ { + } ( \mathbf { x } _ { \mathrm { i n s p } } ) .\tag{10}
$$

3. Combining the direct and sequential methods aims to utilize more of the information the network has learned during training. The given point is moved one step sequentially forward and averaged with the point moved directly from $t _ { \mathrm { i n s p } }$ to the given time. This method relies on the previous point, thus it has to be calculated for intermediate points $\tilde { \mathbf { x } } _ { i }$ as well

$$
\tilde { \mathbf { x } } _ { i } = \frac { 1 } { 2 } ( \Phi _ { t _ { \mathrm { i n s p } }  t _ { i } } ( \mathbf { x } _ { \mathrm { i n s p } } ) + \Phi _ { t _ { i - 1 }  t _ { i } } ( \tilde { \mathbf { x } } _ { i - 1 } ) ) .\tag{11}
$$

4. The Taylor method for inference leverages the inverse consistency between the direct mapping from source to target and vice versa [6]. The mapping from source to target is averaged with the inverse mapping of the target to source mapping, which is approximated with a Taylor expansion

$$
\tilde { \mathbf { x } } _ { \mathrm { e x p } } = \frac { 1 } { 2 } ( \Phi _ { t _ { \mathrm { i n s p } }  t _ { \mathrm { e x p } } } ( \mathbf { x } _ { \mathrm { i n s p } } ) + \Phi _ { t _ { \mathrm { e x p } }  t _ { \mathrm { i n s p } } } ^ { - 1 } ( \mathbf { x } _ { \mathrm { i n s p } } ) ) ,\tag{12}
$$

where $\Phi _ { t _ { \mathrm { e x p } }  t _ { \mathrm { i n s p } } } ^ { - 1 }$ is is approximated with a first-order Taylor expansion. For ease of notation we will use $\Phi _ { F } = \Phi _ { t _ { \mathrm { i n s p } }  t _ { \mathrm { e x p } } } , \Phi _ { B } = \Phi _ { t _ { \mathrm { e x p } }  t _ { \mathrm { i n s p } } }$ and $\mathbf { x } =$ $\mathbf { x } _ { \mathrm { i n s p } } ,$ representing the forward and backward mappings and the inspiratory landmark, respectively. The inverse backward mapping is calculated as

$$
\Phi _ { B } ^ { - 1 } ( { \bf x } ) = \Phi _ { F } ( { \bf x } ) + \nabla ^ { - 1 } \Phi _ { B } ( \Phi _ { F } ( { \bf x } ) ) ( { \bf x } - \Phi _ { B } ( \Phi _ { F } ( { \bf x } ) ) ,\tag{13}
$$

where $\nabla ^ { - 1 } \Phi _ { B }$ is the inverse Jacobian matrix calculated at the point $\Phi _ { F } ( { \bf x } )$

## 3 Experiments and Results

Dataset. The DIR-LAB data set [3] consists of 4D CT scans of the lungs of 10 participants. Each 4D CT scan is composed of 10 volumetric frames $( f _ { 0 } , \ldots , f _ { 9 } )$ where the extreme phases, the inspiratory and expiratory phases, are at $f _ { 0 }$ and $f _ { 5 }$ , respectively. Registration from inspiratory phase to expiratory phase in DIR-LAB is commonly used to validate image registration for medical imaging [1,4,6,7,16,17,18,20]. In the two extreme phases of each 4D CT, 300 expert annotated landmarks are available to measure registration performance. As common practice, we only calculate the deformation field within the lungs, with a mask obtained from Hofmanninger et al. [9]. The DIR-LAB data set also contains 75 landmarks per patient that have been propagated from frame 0 and frame 5 to the intermediate frames, which is used to evaluate the outside the extreme phases. However, as these are not expert annotated, the results from these are not as reliable.

Table 1: Comparison of mean target registration error (TRE) with other registration methods for landmark registration inspiration to expiration. The results for the other methods are adopted from the literature.
<table><tr><td rowspan=1 colspan=1>Method               TRE [mm] ↓</td></tr><tr><td rowspan=1 colspan=1>VoxelMorph+ [1]          2.89</td></tr><tr><td rowspan=1 colspan=1>GradICON [16]              1.31</td></tr><tr><td rowspan=1 colspan=1>GraphRegNet [4]          1.39</td></tr><tr><td rowspan=1 colspan=1>PTVReg [18]              0.96</td></tr><tr><td rowspan=1 colspan=1>RRN [7]                    0.83</td></tr><tr><td rowspan=1 colspan=1>IDIR [20]                   1.07Cycle-IDIR [6]             1.04NePhi [17]                  1.46</td></tr><tr><td rowspan=1 colspan=1>Temporal-IDIR (ours)     1.03</td></tr></table>

Experimental Details. The INRs used for the cyclic registration are parameterized with 5 hidden layers each with 512 neurons. They are trained for 10 000 epochs using the Adam optimizer[10] with a learning rate of $1 0 ^ { - 4 }$ , decaying every 1000 epochs by a factor of 0.5. An epoch consists of 15 000 randomly sampled points within the union of lung masks for all time frames. A source and a target time are randomly sampled between 0 and 9, to accommodate each point, here the extreme phases are sampled more frequently.

For our optimal model, we use the hyperparameters $\alpha = 0 . 0 5 , \beta = 0 . 0 1$ , and $\lambda = 0 . 0 0 5$ for Eq. (6). We test multiple values of β, while the other parameters are taken from other INR registration frameworks. The methods are evaluated using the target registration error (TRE) between annotated landmarks in the expiratory phase and landmarks moved from inspiratory to expiratory. Registration over a whole cycle takes 12.5 min on an NVIDIA Geforce RTX 4090.

Registration Results. In Table 1 we compare our method to other stateof-the-art registration methods on the DIR-LAB dataset including several INRbased methods. The results are taken from their respective papers except VoxelMorph+ [1] where the testing was conducted by Hansen et al. [4]. Our method beats the other INR-based methods, but not the PTVReg [18] and RRN [7] methods.

Table 2 explores both the sensitivity to changes in the β-hyperparameter, i.e. the temporal loss, and the precision of the four proposed inference methods. The framework is quite robust to changes in the β-hyperparameter, with lowest registration error with the Taylor method across all experiments, and lowest with $\beta = 0 . 0 1$ . The registration errors for this model are illustrated as a cumulative distribution for each method in in Fig. 2(a) and as error over time in Fig. 2(b).

Table 2: Evaluation of β-values across the four inference methods in mean registration errors in mm for the landmark registration.
<table><tr><td>Setting</td><td>Direct</td><td>Sequential</td><td>Combined</td><td> $\mathrm { T a y l o r }$ </td></tr><tr><td> $\beta = 0 . 0$ </td><td>1.09</td><td>1.52</td><td>1.09</td><td>1.05</td></tr><tr><td> $\beta = 0 . 0 1$ </td><td>1.08</td><td>1.38</td><td>1.08</td><td>1.03</td></tr><tr><td> $\beta = 0 . 0 5$ </td><td>1.09</td><td>1.21</td><td>1.08</td><td>1.04</td></tr><tr><td> $\beta = 0 . 1$ </td><td>1.08</td><td>1.16</td><td>1.06</td><td>1.04</td></tr><tr><td> $\beta = 0 . 5$ </td><td>1.09</td><td>1.11</td><td>1.08</td><td>1.07</td></tr><tr><td> $\beta = 1 . 0$ </td><td>1.10</td><td>1.12</td><td>1.12</td><td>1.10</td></tr><tr><td> $\beta = 2 . 0$ </td><td>1.23</td><td>1.23</td><td>1.23</td><td>1.22</td></tr></table>

![](images/b4e19c6578a230d3484e0f4a19435087a0f663dd15b8a5060ee8b909a8d8e736.jpg)

![](images/767d4a532f65b49aea72e6bbe602c9a18c36e31fa81f755e05425a089fbdaa61.jpg)  
Fig. 2: Comparison of interference methods for model with $\beta = 0 . 0 1$ . (a) cumulative distribution of TRE for all in the dataset. (b) Mean TRE over time.

The estimated landmark movement throughout the respiratory cycle is illustrated for several landmarks in Fig. 3(c). Figure 3(a,b) illustrates how the network captures the movement between the training frames of the 4D CT.

## 4 Discussion and Conclusion

Our proposed framework, Temporal-IDIR, extends INR-based registration to periodic temporal data, by including two timesteps in the input to the INR and training for registration between all frames simultaneously. This enables the framework to model the movement between any two given time points in the 4D CT scans. With the inclusion of the temporal consistency loss, Eq. (3), in combination with the Taylor inference method, the network has a slight edge over the other INR-based registration methods on the DIR-LAB dataset.

Although our framework does not outperform the two methods RRN [7] and pTVreg [18] in the TRE metric, it has other advantages. INR based methods are not restricted to a grid as both these methods are, but operates on continuous points. This entails that the network size and memory load during optimization are more consistent over inputs with diferent sizes and resolutions.

![](images/6b8544506cea453b0abe0ae4623d436d3a63392d2f77c101984707c706afb4a2.jpg)  
Fig. 3: Landmark movement in the sagittal plane (yz). (a) and (b) illustrates the movement of a single point using 10 time steps and 100 time steps, respectively. (c) illustrates the movement of several points over a respiratory cycle.

The temporal loss $\mathcal { L } _ { \mathrm { t e m p o r a l } }$ is controlled with the β-hyperparameter. Having a small value of $\beta$ generally gives the best results as seen in Table 2, where especially the Taylor method obtain a low TRE. The sequential method is not on par with the other methods for small $\beta \mathrm { . }$ -values. Small errors propagate through multiple deformations, resulting in larger deviations when registering between extreme phases. The temporal loss penalizes this propagation of errors. Thus, we see an increase in the performance of the sequential method when increasing $\beta ,$ until it is on par with the direct method, at $\beta = 1 . 0$ . At this point, there is not more to gain from the temporal loss. The increased performance of the sequential method can, in some cases, make this method preferred to the other methods. Each deformation is smaller with the sequential method; this can give better regularity over the deformations, potentially increasing the quality of a moved object compared to moving it with one big deformation.

As the temporal inputs to the model are sampled from the unit circle, our framework is capable of modeling the movement of landmarks continuously, as opposed to only modeling the movement between extreme phases. The network is trained with 10 diferent temporal inputs, corresponding to the 10 frames in the 4D CT scans. However, as $\mathrm { F i g . \ 3 ( a , b ) }$ shows, we are able to input any encoding from the unit circle as the temporal input. This enables our method to interpolate between the 10 time frames on which it was trained and to do a continuous smooth movement of landmarks. In general, the framework can be trained with input with any number of temporal dimensions, but inference can also be done with higher resolution than the training process. The ability to model movement over the whole sequence of the data makes our method better suited for practical use, as clinical cases involving organ movement, e.g. the heart movement, the whole temporal movement is of interest.

In conclusion, our INR-based temporal registration method is able to capture the movement of the lungs over a respiratory cycle. The registration quality is better than other INR-based registration frameworks. Additionally, our method models the movement across the entire period of the data, and enables interpolation between the frames of the input modality, making the registration continuous in time.

## References

1. Balakrishnan, G., Zhao, A., Sabuncu, M., Guttag, J., Dalca, A.V.: Voxelmorph: A learning framework for deformable medical image registration. IEEE TMI: Transactions on Medical Imaging 38, 1788–1800 (2019)

2. Byra, M., Poon, C., Rachmadi, M.F., Schlachter, M., Skibbe, H.: Exploring the performance of implicit neural representations for brain image registration. Scientific Reports 13(1) (Oct 2023). https://doi.org/10.1038/s41598-023-44517-5, http://dx.doi.org/10.1038/s41598-023-44517-5

3. Castillo, R., Castillo, E., Guerra, R., Johnson, V.E., McPhail, T., Garg, A.K., Guerrero, T.: A framework for evaluation of deformable image registration spatial accuracy using large landmark point sets. Physics in Medicine & Biology 54(7), 1849–1870 (Apr 7 2009). https://doi.org/10.1088/0031-9155/54/7/001, epub 2009 Mar 5

4. Hansen, L., Heinrich, M.P.: Graphregnet: Deep graph regularisation networks on sparse keypoints for dense registration of 3d lung cts. IEEE Transactions on Medical Imaging 40(9), 2246–2257 (2021). https://doi.org/10.1109/TMI.2021. 3073986

5. van Harten, L., Herten, R.L.M.V., Stoker, J., Isgum, I.: Deformable image registration with geometry-informed implicit neural representations. In: Medical Imaging with Deep Learning (2023), https://openreview.net/forum?id=Pj9vtDIzSCE

6. van Harten, L.D., Stoker, J., Išgum, I.: Robust deformable image registration using cycle-consistent implicit representations. IEEE Transactions on Medical Imaging 43(2), 784–793 (2024). https://doi.org/10.1109/TMI.2023.3321425

7. He, X., Guo, J., Zhang, X., Bi, H., Gerard, S., Kaczka, D., Motahari, A., Hofman, E., Reinhardt, J., Barr, R.G., et al.: Recursive refinement network for deformable lung registration between exhale and inhale ct scans. arXiv preprint arXiv:2106.07608 (2021)

8. Heinrich, M.P., Hansen, L.: Voxelmorph++ going beyond the cranial vault with keypoint supervision and multi-channel instance optimisation. In: Biomedical Image Registration: 10th International Workshop, WBIR 2022, Munich, Germany, July 10–12, 2022, Proceedings. pp. 85–95. Springer International Publishing, Cham (2022)

9. Hofmanninger, J., Prayer, F., Pan, J., Röhrich, S., Prosch, H., Langs, G.: Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. European Radiology Experimental 4, 1–13 (2020)

10. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. CoRR abs/1412.6980 (2014), https://api.semanticscholar.org/CorpusID:6628106

11. Klein, S., Staring, M., Murphy, K., Viergever, M.A., Pluim, J.P.: elastix: a toolbox for intensity-based medical image registration. IEEE Transactions on Medical Imaging 29(1), 196–205 (Jan 2010). https://doi.org/10.1109/TMI.2009. 2035616, epub 2009 Nov 17

12. Lowes, M.M., Pedersen, J.J., Hansen, B.S., Kofoed, K.F., Sermesant, M., Paulsen, R.R.: Implicit neural representations for registration of left ventricle myocardium during a cardiac cycle. In: Statistical Atlases and Computational Models of the Heart. Workshop, CMRxRecon and MBAS Challenge Papers. pp. 172–182. Springer Nature Switzerland, Cham (2025)

13. Rueckert, D., Sonoda, L.I., Hayes, C., Hill, D.L., Leach, M.O., Hawkes, D.J.: Nonrigid registration using free-form deformations: application to breast mr images. IEEE Transactions on Medical Imaging 18(8), 712–721 (Aug 1999). https: //doi.org/10.1109/42.796284

14. Sitzmann, V., Martel, J.N., Bergman, A.W., Lindell, D.B., Wetzstein, G.: Implicit neural representations with periodic activation functions. In: Proc. NeurIPS (2020)

15. Sokooti, H., de Vos, B., Berendsen, F., Lelieveldt, B.P., Išgum, I., Staring, M.: Nonrigid image registration using multi-scale 3d convolutional neural networks. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 232–239. Springer, Cham (September 2017)

16. Tian, L., Greer, H., Vialard, F.X., Kwitt, R., Estépar, R.S.J., Rushmore, R.J., Makris, N., Bouix, S., Niethammer, M.: GradICON: Approximate difeomorphisms via gradient inverse consistency (2023), https://arxiv.org/abs/2206.05897

17. Tian, L., Sengupta, S., Greer, H., Estépar, R.S.J., Niethammer, M.: Nephi: Neural deformation fields for approximately difeomorphic medical image registration. ArXiv abs/2309.07322 (2023), https://api.semanticscholar.org/CorpusID: 261823163

18. Vishnevskiy, V., Gass, T., Szekely, G., Tanner, C., Goksel, O.: Isotropic total variation regularization of displacements in parametric image registration. IEEE Transactions on Medical Imaging 36(2), 385–395 (2017). https://doi.org/10.1109/ TMI.2016.2610583

19. de Vos, B.D., Berendsen, F.F., Viergever, M.A., Sokooti, H., Staring, M., Išgum, I.: A deep learning framework for unsupervised afine and deformable image registration. Medical Image Analysis 52, 128–143 (Feb 2019). https://doi.org/10.1016/ j.media.2018.11.010, http://dx.doi.org/10.1016/j.media.2018.11.010

20. Wolterink, J.M., Zwienenberg, J.C., Brune, C.: Implicit neural representations for deformable image registration. In: Medical Imaging with Deep Learning 2022 (2022)

21. Wu, Z., Lan, T., Wang, J., Ding, Y., Qin, Z.: Medical image registration using b-spline transform. International Journal of Simulation: Systems, Science & Technology 17, 1.1–1.6 (01 2016). https://doi.org/10.5013/IJSSST.a.17.48.01

22. Yang, X., Kwitt, R., Styner, M., Niethammer, M.: Quicksilver: Fast predictive image registration – a deep learning approach. NeuroImage 158, 378–396 (Sep 2017). https://doi.org/10.1016/j.neuroimage.2017.07.008, http://dx.doi. org/10.1016/j.neuroimage.2017.07.008