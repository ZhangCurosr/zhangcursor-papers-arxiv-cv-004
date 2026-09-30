# PCaPaint: Prostate Cancer Inpainting by Mitigating Shortcut Learning

Levente Lippenszky, Hongxu Yang, Marcell Dömötör, Krisztian Koos, and László Ruskó

GE HealthCare levente.lippenszky@gehealthcare.com

Abstract. The development of AI systems for tumor-specific applications is limited by the scarcity of labeled data. Synthetic tumor inpainting ofers a promising approach but faces challenges for prostate cancer MRI which contains high-resolution multi-sequence data. Although methods leveraging latent difusion models (LDMs) enable large-volume synthesis, they are prone to shortcut learning, simply reproducing the condition image created by masking the lesion region. In this work, we introduce PCaPaint, a prostate cancer inpainting method based on LDMs that explicitly addresses this failure mode. To overcome shortcut learning that compromises synthetic tumor texture, we propose a simple yet eficient conditioning strategy in which the condition image is filled with Gaussian noise, and we provide theoretical justification. In addition, we propose a novel training objective for LDM that emphasizes the error within the lesion region. Furthermore, we introduce a multisequence latent design, in which T2w scans and DWI&ADC scans are compressed using two separate autoencoders to preserve their distinct frequency characteristics. Extensive experiments demonstrate that the generated synthetic data improves downstream performance in prostate lesion segmentation, patient-level classification and lesion-level detection. Furthermore, our method significantly outperforms a recent state-of-theart LDM-based tumor inpainting method both in downstream performance and in synthetic image quality.

Keywords: Prostate cancer · Inpainting · Shortcut learning.

## 1 Introduction

The development of AI systems in healthcare is often constrained by the scarcity of labeled data [7, 11]. This challenge is even more pronounced in tumor-specific applications, where pathological cases are far less common than normal patient data [21]. Furthermore, developing robust deep learning models for these applications requires substantial amount of data with suficient diversity [12, 19, 5]. However, collecting these datasets is expensive, and the dificulty of accurately annotating tumors intensifies this challenge [3, 20].

Synthetic tumor inpainting ofers a promising approach for generating diverse lesion data while eliminating the need for manual annotation. However, prostate biparametric MRI (bpMRI) synthesis poses significant challenges as it involves high-resolution 3D data and three MRI sequences with substantially diferent frequency characteristics. In recent years, tumor inpainting methods based on difusion models [8] have emerged covering a large variety of lesion types. LeFusion [21] incorporates forward-difused background context into the reverse difusion process, but its image-space design restricts its use for prostate bpMRI. Other state-of-the-art (SOTA) lesion inpainting methods, such as Dif-Tumor [3] and SynBT [20], leverage latent difusion models (LDMs) [14]. These approaches use a healthy image as condition where the lesion region is zeroed out, and the LDM is trained to reconstruct the lesion. While latent representations enable prostate bpMRI synthesis, this conditioning strategy can lead to shortcut learning, causing the model to reconstruct the healthy image rather than inpaint lesion texture. Fig. 1 illustrates shortcut learning in DifTumor [3]. Fig. 1 (a) shows a ground truth T2-weighted (T2w) training example with a real lesion, and (b) the corresponding healthy condition image with the lesion region filled with zeros (gray). After 200,000 training steps, the generated output in (c) exhibits the shortcut failure mode, reproducing the healthy image instead of synthesizing lesion texture. Fig. 1 (d) and (e) show the same behavior when the conditioning image is filled with 1 values (black).

In this work, we propose PCaPaint, a novel prostate cancer inpainting method based on LDMs that generates realistic synthetic lesions in healthy prostate bpMRI cases, while explicitly addressing shortcut learning. Our main contributions are as follows.

1. We introduce a simple yet eficient conditioning strategy where the condition image is filled with Gaussian noise and provide theoretical justification that it mitigates shortcut learning. In addition, we propose a training objective for LDM that emphasizes the error within the lesion region to further mitigate this failure mode.

2. We introduce a novel multi-sequence latent design, in which T2w scans and DWI&ADC scans are compressed using two separate autoencoders to preserve their distinct frequency characteristics.

3. Extensive experiments show that our synthetic data improves performance in three downstream tasks. Moreover, our method significantly outperforms a SOTA LDM-based tumor inpainting method both in downstream performance and in synthetic image quality.

## 2 Methods

## 2.1 Conditioning with Noise Fill

We provide theoretical justification that filling the lesion region in the condition image with noise mitigates shortcut learning. Proposition 1 assumes a difusion model [8], but the reasoning extends to LDM [14]. This is illustrated qualitatively in Fig. 1 (f) and (g) where DifTumor [3] generates improved lesion texture, and quantitatively in Section 3.2.

![](images/6c3e02eff9c90de9f4ffac70291b736def37a94b204b80d5b366868de85399fe.jpg)  
Fig. 1. DifTumor [3] samples under diferent conditioning strategies. Panels show the condition and sample for zero fill (gray), −1 fill (black) and the proposed noise fill.

Let $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ denote the lesion image and h the healthy condition image. The healthy image is defined as $\pmb { h } = ( 1 - \pmb { m } ) \odot \pmb { x } _ { 0 } + \pmb { m } \odot \pmb { \eta } _ { 1 } $ , where m is the lesion mask and $\eta$ is the tensor used to fill the lesion region. In LDM-based tumor inpainting methods, η is the all-zero tensor [20, 3]. We define the lesion diference as $\pmb { d } : = \pmb { x } _ { 0 } - \pmb { h } = \pmb { m } \odot ( \pmb { x } _ { 0 } - \pmb { \eta } )$ . The difusion forward process defines the noisy lesion image at timestep t as $\pmb { x } _ { t } = \sqrt { \bar { \alpha } _ { t } } \pmb { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$ , where $\begin{array} { r } { \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) , \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$ and $\alpha _ { t } = 1 - \beta _ { t }$ , with $\{ \beta _ { t } \} _ { t = 1 } ^ { T }$ denoting the noise schedule and $T$ the number of time steps. The difusion model $\epsilon _ { \theta }$ predicts the noise as $\hat { \pmb { \epsilon } } = \epsilon _ { \theta } ( \pmb { x } _ { t } , t , h , \pmb { m } )$ Fig. $1 \ ( \mathrm { b } ) \mathrm { - ( e ) }$ shows that DifTumor learns a shortcut and denoises $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ by simply reconstructing the healthy image h when the fill tensor is constant. We assume the difusion model exploits a shortcut by approximating the analytically computable residual $\mathbf { } _ { \mathbf { } \mathbf { } \mathbf { } \mathbf { } r _ { t } }$ as its noise prediction

$$
r _ { t } = \frac { \pmb { x } _ { t } - \sqrt { \bar { \alpha } _ { t } } \pmb { h } } { \sqrt { 1 - \bar { \alpha } _ { t } } } .\tag{1}
$$

This solution depends only on the inputs $( \pmb { x } _ { t } , \pmb { h } , \pmb { m } , t )$ available to the network and the fixed noise schedule.

Proposition 1. Assume the network exploits the shortcut $\hat { \epsilon } = \boldsymbol { r } _ { t }$ . Consider $( i )$ constant fill $\mathbf { \eta } _ { \eta } = \mathbf { \epsilon } c$ and (ii) Gaussian noise fill $\pmb { \eta } \sim \mathcal { N } ( \pmb { c } , \sigma ^ { 2 } \pmb { I } )$ , where c is a constant tensor and $\sigma > 0$ denotes the standard deviation. Then, the diference of the expected training losses satisfies

$$
\mathcal { L } _ { \mathrm { n o i s e } } - \mathcal { L } _ { \mathrm { c o n s t } } = \kappa \sigma ^ { 2 } \mathbb { E } _ { { \pmb x } _ { 0 } } \left[ n _ { m } ( { \pmb x } _ { 0 } ) \right] > 0 ,\tag{2}
$$

where $\begin{array} { r } { t \sim \mathrm { U n i f o r m } ( \{ 1 , . . . , T \} ) , \kappa _ { t } : = \frac { \sqrt { \bar { \alpha } _ { t } } } { \sqrt { 1 - \bar { \alpha } _ { t } } } } \end{array}$ and $\kappa = \mathbb { E } _ { t } \left[ \kappa _ { t } ^ { 2 } \right] > 0 , n _ { m } ( { \pmb x } _ { 0 } ) > 0$ is the number of masked voxels $f o r { \pmb x } _ { 0 }$ . Noise fill increases the loss for the shortcut solution, encouraging the network to learn the intended denoising.

Proof. Substituting $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ into $\mathbf { \nabla } _ { \mathbf { r } _ { t } }$ gives

$$
r _ { t } = \frac { \sqrt { \bar { \alpha _ { t } } } x _ { 0 } + \sqrt { 1 - \bar { \alpha _ { t } } } \epsilon - \sqrt { \bar { \alpha _ { t } } } h } { \sqrt { 1 - \bar { \alpha } _ { t } } } = \epsilon + \frac { \sqrt { \bar { \alpha _ { t } } } \left( x _ { 0 } - h \right) } { \sqrt { 1 - \bar { \alpha } _ { t } } } = \epsilon + \kappa _ { t } d .\tag{3}
$$

The residual is the sum of the true noise and the scaled lesion diference. The shortcut behavior dominates as the lesion mask m approaches the empty mask, since $\mathbf { } _ { \mathbf { } } ^ { \mathbf { } } \mathbf { \Delta } \mathbf { r } _ { t }$ then converges to $\epsilon .$ The training loss $\mathcal { L }$ takes the following form

$$
\begin{array} { r } { \mathbb { E } \left[ \Vert \hat { \epsilon } - \epsilon \Vert ^ { 2 } \right] = \mathbb { E } _ { t , x _ { 0 } , m , \epsilon , \eta } \left[ \Vert \kappa _ { t } d \Vert ^ { 2 } \right] = \mathbb { E } _ { t } \left[ \kappa _ { t } ^ { 2 } \right] \mathbb { E } _ { { \mathbf { x } _ { 0 } } , m , \eta } \left[ \Vert m \odot ( { \mathbf { x } _ { 0 } } - \pmb { \eta } ) \Vert ^ { 2 } \right] } \end{array}\tag{4}
$$

We define the diagonal matrix $M : = \deg ( \sec ( m ) )$ ) that performs the same masking operation as m when the images are treated as vectors. We assume the lesion mask is provided for each $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ via manual annotation, and we treat $M = M ( \pmb { x } _ { 0 } )$ as deterministic given $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ . Using the law of total expectation and $\eta \perp x _ { 0 }$

$$
\mathcal { L } = \kappa \mathbb { E } _ { { \pmb x } _ { 0 } } \left[ \mathbb { E } _ { { \pmb \eta } } \left[ \| { \pmb M } ( { \pmb x } _ { 0 } ) ( { \pmb x } _ { 0 } - { \pmb \eta } ) \| ^ { 2 } \ | \ { \pmb x } _ { 0 } \right] \right] .\tag{5}
$$

By the bias-variance decomposition [1], the expected squared $L _ { 2 }$ norm of a random vector X can be decomposed as $\mathbb { E } \left\lceil \| X \| ^ { 2 } \right\rceil = \operatorname { t r } ( \operatorname { T o v } ( X ) ) + \| \mathbb { E } [ X ] \| ^ { 2 }$ . To decompose the inner expectation in Eq. (5), we define $Z : = { \cal M } ( { \pmb x } _ { 0 } ) ( { \pmb x } _ { 0 } - { \pmb \eta } )$ Then $\mathbb { E } \left[ Z \mid \pmb { x } _ { 0 } \right] = M ( \pmb { x } _ { 0 } ) ( \pmb { x } _ { 0 } - \pmb { \mu } _ { \eta } )$ and $\operatorname { C o v } ( Z \mid x _ { 0 } ) = M ( x _ { 0 } ) \Sigma _ { \eta } M ( x _ { 0 } ) ^ { T }$ where $\pmb { \mu } _ { \eta } : = \mathbb { E } [ \pmb { \eta } ]$ and $\Sigma _ { \eta } : = \mathrm { C o v } ( \eta )$ . Applying the decomposition yields

$$
\begin{array} { r } { \mathbb { E } \left[ \| Z \| ^ { 2 } \mid \pmb { x } _ { 0 } \right] = \operatorname { t r } \left( \operatorname { C o v } \left( Z \mid \pmb { x } _ { 0 } \right) \right) + \left\| \mathbb { E } \left[ Z \mid \pmb { x } _ { 0 } \right] \right\| ^ { 2 } . } \end{array}\tag{6}
$$

Substituting into Eq. (5) gives

$$
\mathcal { L } = \kappa \mathbb { E } _ { { \pmb x } _ { 0 } } \left[ \mathrm { t r } \left( M ( { \pmb x } _ { 0 } ) \pmb { \Sigma } _ { \eta } M ( { \pmb x } _ { 0 } ) ^ { T } \right) + \left\| M ( { \pmb x } _ { 0 } ) ( { \pmb x } _ { 0 } - { \pmb \mu } _ { \eta } ) \right\| ^ { 2 } \right] .\tag{7}
$$

In (i) constant fill, $\Sigma _ { \eta } = \mathbf { 0 }$ and $\mu _ { \eta } = c _ { ; }$ which implies that

$$
\mathcal { L } _ { \mathrm { c o n s t } } = \kappa \mathbb { E } _ { { \pmb x } _ { 0 } } \left[ \| { \pmb M } ( { \pmb x } _ { 0 } ) ( { \pmb x } _ { 0 } - { \pmb c } ) \| ^ { 2 } \right] .\tag{8}
$$

In (ii) noise fill, $\Sigma _ { \eta } = \sigma ^ { 2 } I , \mu _ { \eta } = c$ and $\mathrm { t r } ( M ( { \pmb x } _ { 0 } ) \sigma ^ { 2 } I M ( { \pmb x } _ { 0 } ) ^ { T } ) = \sigma ^ { 2 } n _ { m } ( { \pmb x } _ { 0 } )$ which yields

$$
\mathcal { L } _ { \mathrm { n o i s e } } = \kappa \mathbb { E } _ { { \pmb x } _ { 0 } } \left[ \sigma ^ { 2 } n _ { m } ( { \pmb x } _ { 0 } ) + \| { \pmb M } ( { \pmb x } _ { 0 } ) ( { \pmb x } _ { 0 } - { \pmb c } ) \| ^ { 2 } \right] .\tag{9}
$$

Subtracting Eq. (8) from Eq. (9) completes the proof.

## 2.2 PCaPaint

Autoencoders In the first stage, T2w scans and DWI&ADC scans are compressed separately to reflect their distinct characteristics: T2w contains rich high-frequency anatomical detail, while ADC and DWI exhibit lower-frequency intensity patterns. We employ the vector-quantized variational autoencoder with adversarial training (VQGAN) [6] for both autoencoders, $i \in \{ 1 , 2 \}$ . We denote the T2w image as $\mathbf { \bar { x } } ^ { ( 1 ) } \in \mathbb { R } ^ { H \times \bar { W } \times D }$ , where H, W and D represent the height, width, and depth, respectively. We further define $\pmb { x } ^ { ( 2 ) } \in \dot { \mathbb { R } } ^ { H \times W \times D \times 2 }$ , representing the DWI and ADC volumes concatenated along the channel dimension. The input MRI tensor $\mathbf { \boldsymbol { x } } ^ { ( i ) }$ is mapped to a continuous latent representation $z _ { e } ^ { ( i ) } \in \mathbb { R } ^ { h \times w \times d \times c }$ via the encoder $f ^ { ( i ) }$ , where $h , w , d$ denote the latent space dimensions and c is the number of channels. In the vector quantization step, each c-dimensional latent vector in $ { \boldsymbol { z } } _ { e } ^ { ( i ) }$ is replaced by its nearest codebook entry from the learned codebook $\mathcal { C } = \{ c _ { k } \} _ { k = 1 } ^ { K } \{$ , resulting in the quantized latent $\mathfrak { z } _ { q } ^ { ( i ) } = q ( \mathfrak { z } _ { e } ^ { ( i ) } ) \in \mathbb { R } ^ { h \times w \times d \times c }$ , where K is the codebook size. Lastly, the decoder $g ^ { ( i ) }$ reconstructs the quantized latent such that $\hat { \pmb { x } } ^ { ( i ) } = g ^ { ( i ) } ( z _ { q } ^ { ( i ) } )$ . The loss is defined as $\mathcal { L } = \mathcal { L } _ { \mathrm { r e c o n } } + \mathcal { L } _ { \mathrm { c o m m i t } } + \lambda _ { \mathrm { p e r c e p } } \mathcal { L } _ { \mathrm { p e r c e p } } + \lambda _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { a d v } }$ , where $\dot { \mathcal { L } } _ { \mathrm { r e c o n } } = \| \pmb { x } ^ { ( i ) } - \hat { \pmb { x } } ^ { ( i ) } \| _ { 1 }$ and $\mathcal { L } _ { \mathrm { c o m m i t } } = \beta \| \mathrm { s g } [ z _ { q } ^ { ( i ) } - z _ { e } ^ { ( i ) } ] \| _ { 2 } ^ { 2 }$ , with $\mathrm { s g } [ \cdot ]$ denoting the stop-gradient operator and $\beta$ the commitment cost. The perceptual loss $\mathcal { L } _ { \mathrm { p e r c e p } }$ measures feature similarity between $\mathbf { \boldsymbol { x } } ^ { ( i ) }$ and $\hat { \pmb x } ^ { ( i ) }$ in a pretrained network, and $\mathcal { L } _ { \mathrm { a d v } }$ computes the least-squares patch adversarial loss using a patch discriminator [10].

![](images/7824c9e550c3679d463f7f5e16188859695258b1b2c452b66c188d8c46869e89.jpg)  
Fig. 2. Training the latent difusion model (LDM). The lesion bpMRI $\scriptstyle { \mathbf { 2 } } 0$ is encoded by the $\mathrm { V Q G A N }$ encoders to form the latent $\scriptstyle z _ { 0 }$ . LDM is conditioned on the latent healthy image $_ { z _ { h } }$ and the downsampled lesion mask m˜ . The healthy image h is constructed by replacing the lesion region with Gaussian noise. LDM is trained with an objective that amplifies the error in the lesion region.

Latent Difusion Model In the second stage, we train an LDM [14] conditioned on the healthy image and the lesion mask. Lesion bpMRI $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ is encoded to the latent space using the VQGAN encoders, $z _ { 0 } = \mathrm { c o n c a t } ( z _ { 0 } ^ { ( 1 ) } , z _ { 0 } ^ { ( 2 ) } )$ , where $z _ { 0 } ^ { ( i ) } = f ^ { ( i ) } ( \pmb { x } _ { 0 } ^ { ( i ) } )$ and $i \in \{ 1 , 2 \}$ . In the forward process, $z _ { \mathrm { 0 } }$ is gradually transformed into standard Gaussian noise $z _ { T } \sim \mathcal { N } ( \mathbf { 0 } , I )$ following a Markov process. The reverse process is parametrized by a denoising U-Net $\epsilon _ { \theta }$ that predicts the noise added at each time step. The healthy image is constructed by filling the lesion region with Gaussian noise. Formally, we generate standard Gaussian noise $\pmb { \xi } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { I } )$ , scale it by $\sigma > 0$ and map it to the $( - 1 , 1 )$ range using the tanh function, $\pmb { \eta } = \operatorname { t a n h } ( \sigma \pmb { \xi } )$ . Then, h is defined as $\pmb { h } = ( 1 - \pmb { m } ) \odot \pmb { x } _ { 0 } + \pmb { m } \odot \pmb { \eta }$ and encoded into latent $z _ { h }$ where $z _ { h } = \mathrm { c o n c a t } ( z _ { h } ^ { ( 1 ) } , z _ { h } ^ { ( 2 ) } ) , z _ { h } ^ { ( i ) } = f ^ { ( i ) } ( h ^ { ( i ) } )$ and $i \in \{ 1 , 2 \}$ . Furthermore, we downsample the lesion mask to match the latent space dimensions, $\tilde { m } = \mathrm { d o w n } ( m )$ ). The denoising U-Net receives $z _ { h }$ and $\tilde { m }$ as conditions concatenated to ${ \boldsymbol { z } } _ { t }$ along the channel dimension. To further mitigate shortcut learning, we employ a lesion-weighted training objective that amplifies the error inside the lesion region:

$$
\begin{array} { r } { \mathbb { E } _ { \boldsymbol { x } _ { 0 } , m , \epsilon , \boldsymbol { \xi } , t } \left[ \lVert \epsilon - \epsilon _ { \theta } ( z _ { t } , t , z _ { h } , \tilde { m } ) \rVert ^ { 2 } + \lambda \lVert \tilde { m } \odot \epsilon - \tilde { m } \odot \epsilon _ { \theta } ( z _ { t } , t , z _ { h } , \tilde { m } ) \rVert ^ { 2 } \right] , } \end{array}\tag{10}
$$

where λ controls the contribution of the lesion-focused term.

Sampling In the third stage, we generate prostate lesion masks incorporating medical knowledge and inpaint synthetic lesions onto non-lesion bpMRIs. Lesion synthesis is performed by sampling the LDM using DDIM [17], conditioned on the healthy image and the lesion mask. Given peripheral zone (PZ) and transition zone (TZ) segmentation masks, we sample lesion center with probabilities of 0.75 from PZ and 0.25 from TZ [13]. We then sample a target lesion volume in voxels constrained to lie between minimum and maximum fractions of the whole gland volume. An ellipsoidal lesion is generated by drawing random axis ratios and scaling the ellipsoid to match the target volume while accounting for voxel spacing. To create realistic, irregular boundaries, we perturb the ellipsoid surface using Gaussian-filtered noise field scaled by the mean radius. Finally, we clip the mask to the gland by removing voxels outside the gland segmentation.

## 3 Experimental Results

## 3.1 Experimental Setup

Data Training set consisted of 900 bpMRI cases from the publicly available PI-CAI [16] dataset, including 252 cases with clinically significant prostate cancer (csPCa). The test set comprised of 599 bpMRI cases from PI-CAI, of which 172 were csPCa cases. Preprocessing was performed using the $\mathsf { p i c a i \_ p r e p }$ repository [15]. Each scan was resampled to a resolution of $0 . 5 \times 0 . 5 \times 3 . 0 ~ \mathrm { m m ^ { 3 } }$ and cropped or padded to a spatial size of $2 5 6 \times 2 5 6 \times 3 2$ voxels. Image intensities were rescaled to the [ 1, 1] range.

Implementation Details For PCaPaint, we trained the two VQGANs using the VQ-VAE [18] implementation of MONAI 1.5.0 [2] for 100,000 steps using the Adam optimizer. The learning rate was set to $5 \times 1 0 ^ { - 5 }$ for the T2w VQGAN and to $1 0 ^ { - 4 }$ for the DWI&ADC VQGAN. We set the embedding dimensions to 64 and the codebook size to 8192 for both autoencoders. The denoising U-Net was trained using the DifTumor [3] implementation for 200,000 steps with Adam and a learning rate of $5 \times 1 0 ^ { - 5 }$ . The standard deviation of the Gaussian noise fill was set to $\sigma = 0 . 5$ such that most sampled values lie within the [ 1, 1] range. The weight of the second loss term in Eq. (10) was empirically set to $\lambda = 1 5 0$ For DifTumor, we trained a single VQGAN to compress the bpMRIs with a learning rate of $5 \times 1 0 ^ { - 5 }$ . To provide larger capacity for the autoencoder, we increased the embedding dimensions to 128. We applied the same training setup as for PCaPaint to ensure a fair comparison. Trainings were performed on an NVIDIA A100 Tensor Core GPU with 40 GB memory.

## 3.2 Image Quality Evaluation

Synthetic image quality was evaluated for DifTumor, DifTumor with noise fill (NF) and PCaPaint by inpainting the lesion region of the 172 real bpMRIs in the test set. We computed structural similarity index measure (SSIM) and peak signal-to-noise ratio (PSNR) between the ground truth and synthetic images. In addition, we calculated PSNR separately for the foreground and background regions. Table 1 demonstrates that PCaPaint generates statistically significantly higher quality images compared to DifTumor across MRI sequences and image regions. Foreground (fg) results show that DifTumor with NF outperforms Dif-Tumor, while PCaPaint achieves the best image quality, highlighting the benefits of our noise fill conditioning and lesion-focused loss. PCaPaint also achieves the best performance in background (bg) metrics, underscoring the advantage of compressing bpMRIs using two separate autoencoders.

Table 1. Synthetic image quality evaluation for DifTumor [3], DifTumor with noise fill (NF) and PCaPaint. SSIM and PSNR were computed between ground truth and synthetic images for test lesion bpMRIs for the entire image, foreground (fg) and background (bg). 95% bootstrap CI widths are in brackets. Best and second-best are in bold and underline, respectively. <sup>∗</sup> indicates statistically significant diferences between DifTumor and PCaPaint at 5% level using two-sided Wilcoxon signed-rank tests with Holm correction.
<table><tr><td>MRI</td><td>Metric ↑</td><td>DiffTumor</td><td>DiffTumor NF</td><td>PCaPaint (Ours)</td></tr><tr><td rowspan="4">T2w</td><td>SSIM</td><td>0.77 (0.01)</td><td>0.79 (0.01)</td><td>0.83* (0.02)</td></tr><tr><td>PSNR</td><td>20.75 (0.56)</td><td>21.58 (0.60)</td><td>23.11* (0.62)</td></tr><tr><td>PSNR fg</td><td>12.38 (0.64)</td><td>16.26 (0.80)</td><td>20.42* (0.97)</td></tr><tr><td>PSNR bg</td><td>20.80 (0.55)</td><td>21.60 (0.60)</td><td>23.11* (0.62)</td></tr><tr><td rowspan="4">DWI</td><td>SSIM</td><td>0.51 (0.06)</td><td>0.51 (0.06)</td><td>0.64* (0.05)</td></tr><tr><td>PSNR</td><td>21.39 2 (0.66)</td><td>21.16 (0.67)</td><td>25.54* (1.26)</td></tr><tr><td>PSNR fg</td><td>15.58 (0.98)</td><td>16.21 (0.89)</td><td>17.98* (1.12)</td></tr><tr><td>PSNR bg</td><td>21.43 (0.66)</td><td>21.18 (0.67)</td><td>25.59* (1.27)</td></tr><tr><td rowspan="4">ADC</td><td>SSIM</td><td>0.71 (0.04)</td><td>0.71 (0.04)</td><td>0.79* (0.03)</td></tr><tr><td>PSNR</td><td>22.37 (0.52)</td><td>22.47 (0.57)</td><td>25.42* (0.60)</td></tr><tr><td>PSNR fg</td><td>15.82 (0.71)</td><td>18.31 (0.82)</td><td>22.27* (0.88)</td></tr><tr><td>PSNR bg</td><td>22.42 (0.52)</td><td>22.49 (0.57)</td><td>25.44* (0.61)</td></tr></table>

## 3.3 Prostate Lesion Downstream Tasks

We evaluated the utility of synthetic prostate lesion bpMRIs for three downstream tasks on the test set: csPCa segmentation, patient-level csPCa classification and lesion-level csPCa detection. Evaluation metrics included Dice score for segmentation, area under the receiver operating characteristic curve (AUC) for classification and average precision (AP) for detection. The latter two tasks follow the PI-CAI challenge protocol and were evaluated using the picai\_eval repository [15]. First, we trained nnU-Net [9] with default configuration on the training set. We then trained separate nnU-Net models on training sets augmented with synthetic lesions generated by DifTumor and PCaPaint at a ratio of three synthetic lesions per real lesion [4]. Fig. 3 illustrates three non-lesion bpMRI cases, each inpainted with synthetic lesion by both methods, using lesion masks described in Section 2.2. Results demonstrate that our method correctly inpaints lesion textures with modality-specific intensity—darker in T2w, brighter in DWI, and darker in ADC—and outperforms DifTumor. Statistical significance for segmentation was assessed using two-sided Wilcoxon signed-rank test. For classification and detection, significance was determined by bootstrapping AUC and AP diferences, CIs excluding zero indicated significance. Table 2 shows that synthetic examples from PCaPaint yield the best performance across all downstream tasks, significantly outperforming DifTumor. Although the Dice improvement is small, paired diferences in the Wilcoxon test consistently favored PCaPaint over DifTumor: 87 positives, 60 negatives and 25 ties across the 172 test lesion cases.

![](images/8a83f409f726464cbb3a6a19f4d583bf9a40d413547ce5c7d0061410c49e056a.jpg)  
Fig. 3. Three non-lesion bpMRI examples inpainted with synthetic lesions by DifTu mor [3] and PCaPaint using generated lesion masks.

Table 2. nnU-Net performance on csPCa segmentation, patient-level csPCa classification and lesion-level csPCa detection using real and synthetic data. 95% bootstrap CI widths are in brackets. Best performance is in bold. <sup>∗</sup> indicates statistically significant diferences between Real+DifTumor and Real+PCaPaint at 5% level.
<table><tr><td rowspan=1 colspan=1>Downstream Task</td><td rowspan=1 colspan=1>Real</td><td rowspan=1 colspan=1>Real+DiffTumor</td><td rowspan=1 colspan=1>Real+PCaPaint (Ours)</td></tr><tr><td rowspan=1 colspan=1>Segmentation, Dice ↑</td><td rowspan=1 colspan=1>0.47 (0.08)</td><td rowspan=1 colspan=1>0.47 (0.09)</td><td rowspan=1 colspan=1>0.48* (0.08)</td></tr><tr><td rowspan=1 colspan=1>Classification, AUC ↑</td><td rowspan=1 colspan=1>0.71 (0.09)</td><td rowspan=1 colspan=1>0.71 (0.09)</td><td rowspan=1 colspan=1>0.76* (0.08)</td></tr><tr><td rowspan=1 colspan=1>Detection, AP ↑</td><td rowspan=1 colspan=1>0.28 (0.13)</td><td rowspan=1 colspan=1>0.29 (0.13)</td><td rowspan=1 colspan=1>0.34* (0.15)</td></tr></table>

## 4 Conclusion

In this work, we propose a prostate cancer inpainting method based on LDMs. We address shortcut learning in existing LDM-based tumor inpainting methods by introducing a conditioning strategy with theoretical guarantees. As Proposition 1 applies broadly to difusion-based tumor inpainting, we hypothesize that our conditioning would benefit other imaging modalities and tumor types. Future work will explore these directions. As our synthetic data improves performance on prostate lesion downstream tasks, it will support the development of more robust AI systems for prostate cancer care.

## References

1. Bishop, C.M., Nasrabadi, N.M.: Pattern recognition and machine learning, vol. 4. Springer (2006)

2. Cardoso, M.J., Li, W., Brown, R., Ma, N., Kerfoot, E., Wang, Y., Murrey, B., Myronenko, A., Zhao, C., Yang, D., et al.: Monai: An open-source framework for deep learning in healthcare. arXiv preprint arXiv:2211.02701 (2022)

3. Chen, Q., Chen, X., Song, H., Xiong, Z., Yuille, A., Wei, C., Zhou, Z.: Towards generalizable tumor synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11147–11158 (2024)

4. Chen, Q., Zhou, X., Liu, C., Chen, H., Li, W., Jiang, Z., Huang, Z., Zhao, Y., Yu, D., He, J., et al.: Scaling tumor segmentation: Best lessons from real and synthetic data. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 24001–24013 (2025)

5. Chou, Y.C., Li, B., Fan, D.P., Yuille, A., Zhou, Z.: Acquiring weak annotations for tumor localization in temporal and volumetric data. Machine Intelligence Research 21(2), 318–330 (2024)

6. Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12873–12883 (2021)

7. Grabke, E.P., Taati, B., Haider, M.A.: Mitigating multi-sequence 3d prostate mri data scarcity through domain adaptation using locally-trained latent difusion models for prostate cancer detection. arXiv preprint arXiv:2507.06384 (2025)

8. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

9. Isensee, F., Jaeger, P.F., Kohl, S.A., Petersen, J., Maier-Hein, K.H.: nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods 18(2), 203–211 (2021)

10. Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with condi tional adversarial networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1125–1134 (2017)

11. Jin, C., Guo, Z., Lin, Y., Luo, L., Chen, H.: Label-eficient deep learning in medical image analysis: Challenges and future directions. arXiv preprint arXiv:2303.12484 (2023)

12. Liu, J., Zhang, Y., Chen, J.N., Xiao, J., Lu, Y., A Landman, B., Yuan, Y., Yuille, A., Tang, Y., Zhou, Z.: Clip-driven universal model for organ segmentation and tumor detection. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 21152–21164 (2023)

13. McNeal, J.E., Redwine, E.A., Freiha, F.S., Stamey, T.A.: Zonal distribution of prostatic adenocarcinoma: correlation with histologic pattern and direction of spread. The American journal of surgical pathology 12(12), 897–906 (1988)

14. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)

15. Saha, A., Bosma, J., Twilt, J., van Ginneken, B., Yakar, D., Elschot, M., Veltman, J., Fütterer, J., de Rooij, M., et al.: Artificial intelligence and radiologists at prostate cancer detection in mri—the pi-cai challenge. In: Medical Imaging with Deep Learning, short paper track (2023)

16. Saha, A., Twilt, J.J., Bosma, J.S., van Ginneken, B., Yakar, D., Elschot, M., Veltman, J., Fütterer, J., de Rooij, M., Huisman, H.: The picai challenge: public training and development dataset. (No Title) (2022), https://doi.org/10.5281/zenodo.6624726

17. Song, J., Meng, C., Ermon, S.: Denoising difusion implicit models. arXiv preprint arXiv:2010.02502 (2020)

18. Van Den Oord, A., Vinyals, O., et al.: Neural discrete representation learning. Advances in neural information processing systems 30 (2017)

19. Wang, S., Li, C., Wang, R., Liu, Z., Wang, M., Tan, H., Wu, Y., Liu, X., Sun, H., Yang, R., et al.: Annotation-eficient deep learning for automatic medical image segmentation. Nature communications 12(1), 5915 (2021)

20. Yang, H., Timko, E., Lippenszky, L., Czipczer, V., Ferenczi, L.: Synbt: High-quality tumor synthesis for breast tumor segmentation by 3d difusion model. In: Deep Breast Workshop on AI and Imaging for Diagnostic and Treatment Challenges in Breast Care. pp. 1–10. Springer (2025)

21. Zhang, H., Liu, Y., Yang, J., Wan, S., Wang, X., Peng, W., Fua, P.: Lefusion: Controllable pathology synthesis via lesion-focused difusion models. arXiv preprint arXiv:2403.14066 (2024)