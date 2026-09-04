# Zero-Shot Novel Depth Synthesis Using 3D Foundation Models Scene Representations

Denis M. Akola<sup>1</sup> and David F. Fouhey<sup>1,2</sup>

<sup>1</sup> New York University, Tandon School of Engineering, 1 MetroTech Center, Brooklyn, NY 11201, USA

<sup>2</sup> Courant Institute of Mathematical Sciences, 251 Mercer St, New York, NY 10012

Abstract. 3D Foundation Models (3DFMs) such as VGGT have recently pushed the boundaries of 3D vision by predicting rich unified representations with feed-foward transformers. The scene representations learned by these models enable strong performance on multiple 3D vision tasks. In this paper, we investigate using their internal representations to infer 3D in the scene from new views. Our hypothesis is that in order to solve the task of 3D reconstruction, these models need to learn a representation that includes a large amount of general knowledge about 3D scenes. After showing that it is possible to decode hidden surfaces from internal 3DFM representations, we propose a method, Z3D, that estimates pointmaps in unseen views by doing latent difusion on 3DFM representation. We show that Z3D can predict realistic depth maps for new views across multiple datasets. Project page: https://akola-mbey-denis.github.io/Z3D-page

## 1 Introduction

Understanding the full 3D structure of a scene from limited observations remains a fundamental challenge in computer vision. In many real-world settings like robot navigation, AR/VR and autonomous systems, we only observe partial views of a scene, yet must reason about the geometry that lies outside the visible surface, including regions occluded by foreground objects. Novel depth synthesis addresses this challenge. Given one or a few input views, the goal is to predict plausible and geometrically consistent depth for unseen viewpoints, including content hidden behind objects. Beyond view rendering, this capability enables physical interaction, planning, and spatial reasoning in incomplete environments, where missing geometry must be inferred rather than directly observed.

Recent advances in neural scene representations such as NeRF [24],3D Gaussian Splatting [20] and other neural methods [10, 12, 13, 18] have significantly improved novel view synthesis by optimizing continuous 3D representations from multi-view imagery. However, these methods primarily interpolate between observed views and are not designed to hallucinate geometry behind occlusions, especially in sparse-view settings. Prior works [10, 11] on novel depth synthesis often jointly learn view and depth prediction from multi-view data, but progress toward generalizable depth synthesis from sparse inputs remains limited. More recently, generative models [12,19] have been explored for depth prediction, but these often operate in 2D image space and lack strong 3D consistency priors, limiting their ability to produce coherent geometry across viewpoints. The core challenges stem from geometric ambiguity under limited observations and the dificulty of learning representations that transfer across diverse scenes.

The key insight of this paper is that the internal representations of pretrained 3D foundation models (3DFMs) [23, 45, 49, 52] can be efectively combined with difusion-based models for generalizable novel-view depth synthesis. Rather than treating novel depth synthesis as pure reconstruction or view interpolation, we formulate it as a conditional generation task over scene geometry. Specifically, we use the intermediate representations of 3DFMs for conditioning as well as the latent space in which to do latent difusion [32]. After difusion, the predicted latents can be converted to geometry by the original decoder. By decoupling representation learning (which is taken care of by the 3DFM) from generative completion (handled by difusion), our approach enables plausible hallucination of occluded or unobserved structure, and encourages multi-view consistency. To further support our premise that 3DFMs encode geometry beyond directly visible surfaces, we show that simply attaching a lightweight decoder to their features enables inference of hidden surfaces, demonstrating that the learned scene representations are well-suited for reasoning about occluded geometry.

We propose Z3D– Zero-Shot 3D Depth, a simple and efective method that leverages the rich geometric scene representations learned by 3DFMs in combination with a difusion framework to generate novel-view depth maps. Z3D integrates these pretrained 3DFM representations with the relative pose of a novel view to predict consistent and accurate depth for novel view cameras. We evaluate our method on multiple datasets. We compare Z3D with state-of-art methods [18] for novel view depth synthesis.

## 2 Related Work

We aim to generate consistent depth maps for novel views of a scene, given a sparse set of images (2 or more views). We first infer a geometric scene representation from 3DFMs and, using a difusion model conditioned on this representation and the poses of the target views, predict novel-view depths.

Novel View and Depth Synthesis. NeRF [24] and its extensions [25, 26, 54] render novel views of observed surfaces but often require many input images and per-scene optimization. In sparse view settings, they tend to produce incomplete geometry and cannot infer surfaces outside the original field of view or behind occlusions. In contrast, our method infers novel-view geometry from sparse inputs without per-scene optimization. Methods such as PixelNeRF [54], SRT [36], RUST [35], LVSM [18], and RayZer [17] generalize across scenes but focus on photometric quality rather than accurate depth, limiting their ability to hallucinate unseen structure. Our approach explicitly models geometry, generating consistent depth maps beyond visible surfaces. CUT3R [48] is a stateful 3D reconstruction model capable of recovering scene geometry from RGB observations, predicting depth for novel views, and inferring unobserved scene regions by querying virtual camera viewpoints. Unlike CUT3R, our method incorporates pre-trained 3DFM representations within a difusion framework. This design leverages the structured geometric knowledge learned by 3DFM while benefiting from the expressive generative power of difusion models, resulting in more accurate and realistic novel-view depth synthesis.

A few works jointly predict novel views and depth [10, 11]. For example, [10] learns scene representations within the difusion model and predicts depth for a single target view under restrictive camera assumptions. In contrast, we use pretrained 3DFM features, decoupling representation from generation, and produce plausible depths for multiple novel viewpoints without assuming that the target camera(s) are always placed at the origin.

3D Foundation Models (3DFMs). Traditional 3D reconstruction methods such as Structure-from-Motion (SfM) [2, 14, 40] and Multi-View Stereo (MVS) [7–9] recover camera poses and geometry through local feature matching and joint optimization via bundle adjustment. Recent 3D foundation models instead predict camera poses, depth, and point maps in a single feed-forward pass from multiple input images. Representative works in this line of research include DUSt3R [49] and its extensions [16,21,48,52], as well as multiview models such as VGGT [45] and WorldMirror [23]. These models are not designed for novelview synthesis out of the box, but we repurpose their representations for this task by pairing them with difusion models. Specifically, we treat their internal features as strong scene-level geometry representation. We empirically show that a lightweight decoder can use these features to predict hidden surfaces (see § 3.1), demonstrating that 3DFMs encode information useful for reasoning about occluded geometry and unseen viewpoints.

Difusion Models. Denoising Difusion Probabilistic Models (DDPMs) [15] generate samples by iteratively reversing a difusion process, which involves denoising Gaussian noise into structured outputs. Conditional DDPMs incorporate information like text [34], reference images [33], or semantic maps [57] to guide generation. Latent Difusion Models (LDMs) [6, 28, 32] perform denoising in a compact latent space. Difusion has recently been applied to monocular depth [12, 19, 38, 39, 51, 59] and camera pose [47, 56] prediction. These models operate in a compact latent space similar to original LDMs, allowing standard noise schedulers to work out of the box. These approaches [12, 19, 38, 39, 51, 59] predict depth only for surfaces explicitly visible in the input images and cannot hallucinate occluded or unobserved geometry. In contrast, our method conditions difusion on high dimensional geometric features extracted from pretrained 3DFMs that requires careful tuning of noise schedulers.

## 3 Method

Our goal is to predict the depth of a novel view given one or more input images as well as the pose of set of target camera views. In the base case of two views where one view is the source and the other view is target, the model should be able to plausibly hallucinate about the depth of the target view. With more views, the model should be able to predict the depth of the target view(s) more accurately. We propose Z3D, a simple and efective framework for novel view depth synthesis that integrates difusion-based generative modeling with 3DFMs to leverage their complementary strengths. In what follows, we analyze how the scene representations learned by 3DFMs encode geometric structure beyond what is immediately apparent, using a lightweight probe to predict layered depth for each scene. Building upon our findings, we introduce Z3D method that leverages the encoding of a scene from 3DFMs within a difusion framework to learn to generate consistent depth of novel views.

![](images/8d4c0ff622c5131fdf9b01ffd41c20a9e8eaccd74258fb53b4b71b4bfe9762a6.jpg)  
Fig. 1: Linear Probe Layered Depth Image (LDI) prediction capability of 3DFMs. To test whether 3DFMs implicitly contain information about unseen views, we train a linear model to predict LDIs. Good performance on LDI prediction after the first layer suggests that the model understands the full 3D scene, including hidden surfaces. For each example: (left) input RGB; (top) ground-truth LDI layers; (bottom) linearly decoded LDI layers. The ability to recover the non-trivial structure of the scene with only a linear layer suggests that 3DFMs implicitly encode this information.

Background of 3DFMs. 3DFMs [23,45] encode input images into patch tokens using a shared transformer backbone.

Each image is divided into patches, producing tokens that are processed independently in frame attention and jointly across images in global attention. The pretrained patch tokens from the transformer backbone capture rich multiview geometric information and are fed into task-specific heads, such as depth, point map, or camera pose. In our method, we extract these pretrained patch tokens and use them as a scene representation to condition a difusion model which allows us to predict the depth for novel viewpoints. These viewpoints reveal occluded and unobserved surfaces.

## 3.1 Do 3DFMs Know about Unseen Structure?

We first test whether knowledge of this information is already embedded in the 3DFM representations. To accomplish this, we follow the work of Banani et al. [4] and probe the knowledge of these networks. As our target representation, we train a linear probe to predict layered depth images (LDI) [41], a simple representation of occluded parts of the scene. LDIs generalize depth maps and represent, at each pixel, the first k surfaces that the ray through the pixel passes through. Accurately estimating the 2nd surfaces and subsequent surfaces of an LDI requires knowledge of the occluded parts of the scene.

Table 1: LDI results.
<table><tr><td>Method</td><td>AbsRel ↓ δ &lt; 1.25 ↑</td></tr><tr><td>Avg LDI</td><td>0.319 0.564</td></tr><tr><td>VGGT-LDI</td><td>0.197 0.717</td></tr><tr><td>WM-LDI</td><td>0.167 0.789</td></tr></table>

We test this by evaluating how well the last layer of a 3DFM’s DPT head can predict LDIs. Specifically, we expand the DPT head of VGGT [45] to produce four depth layers and confidence maps and train the model while keeping all other layers fixed. Figure 1 shows that 3DFM produces plausible layered depth predictions that extend

beyond the first visible surface. These results suggest, that 3DFMs do contain implicit knowledge for reasoning about occluded and hidden scene structure. We also quantitatively confirm that linear probes on 3DFM features predict LDIs. We report results with VGGT and WM in Table 1. Despite being a weak linear model, they halve the error rate compared to using the dataset average, showing the hidden/unseen information information is embedded in 3DFMs.

## 3.2 Network Architecture

Our goal is to predict novel-view depth maps by performing conditional difusion on 3DFM patch tokens, conditioned on relative poses and source view images. Z3D achieves this by combining a pretrained 3DFM with a Pose Encoder, a conditional Difusion Model, and a DPT head. The pretrained 3DFM provides a high-dimensional latent representation of the scene, capturing rich multi-view geometry. The Pose Encoder encodes the relative camera poses of target views, while the difusion model predicts the target view scene representation from noise conditioned on both the source views and target poses. Finally, the DPT head decodes the sampled scene representation into depth maps. Conceptually, this setup can be seen as latent difusion operating in the space of 3DFM patch tokens, enabling hallucination of occluded surfaces and multi-view consistency. During training, both source and target images are available. We first compute scene representations for the source and target views using the frozen 3DFM backbone. Gaussian noise is added to the target view representation, and the difusion model learns to predict this noise conditioned on the source view representations and relative target view poses. Conditioning is implemented via a cross-attention block in each DiTBlock, while target pose embeddings are added directly to the timestep embeddings, allowing the model to reason about depth from relative camera positions.

![](images/efe22c498311fd1b453fc4fa57d459e1c85267332b42d0d62d091ef79ee2d60a.jpg)  
Fig. 2: Z3D Inference Pipeline. Z3D takes one or more input images (top left images in this figure) and multiple query poses $( p _ { 1 } , p _ { 2 } , p _ { 3 } \ \mathrm { e t c } )$ . The input images are passed into a 3DFM. Z3D infers the depthmaps at the query poses via latent difusion using 3DFM features. In each DiT block, Z3D: (i) conditions on the encoded pose (plus difusion timestep) via adaptive layer normalization following DiT practice; and (ii) attends to the 3DFM intermediate activations while doing multiheaded attention. The difusion-predicted 3DFM feature is passed directly into the frozen DPT head.

3DFM Backbone, ϕ. The 3DFM backbone tokenizes each source image into patches and produces patch tokens, which are processed through alternating frame-level and global attention blocks to capture both per-frame and multiframe geometry. The resulting patch tokens are concatenated to form a scene representation for the input images. Only tokens from layer 17 are used for conditioning the difusion model, leveraging the most informative geometric features while keeping computation eficient.

Pose Encoder, p<sub>θ</sub>. Relative camera poses of target views are encoded as 12-dimensional vectors per view. Each pose is Fourier-encoded and processed through an MLP to produce embeddings that match the feature dimension of the 3DFM patch tokens. All poses are represented relative to the first view in the source set, simplifying spatial reasoning for the difusion model.

Difusion Model, $f _ { \theta } .$ . Our difusion model follows the DiT architecture [28] and operates directly in the high-dimensional latent space of 3DFM patch tokens. Let $I _ { s }$ denote the source view image(s), $I _ { t }$ the target view image(s), and $P _ { t }$ the relative camera pose(s) of the target view(s). We use a shared image backbone ϕ to extract features from both target and source images: $f _ { t } = \phi ( I _ { t } )$ and $f _ { s } = \phi ( I _ { s } )$ We denote the target representation as $x _ { 0 } = f _ { t }$ and the source representations as $z _ { s } = f _ { s }$ . Also, we encode the relative pose using an MLP $p = p _ { \theta } ( P _ { t } )$ .

Recent 3DFM architectures such as VGGT [45] and WorldMirror [23] aggregate transformer tokens from multiple intermediate layers for depth prediction.

Each of these token representations consists of high-dimensional embeddings (e.g., 2048-dimensional in both VGGT and WorldMirror), making it computationally prohibitive to apply difusion directly to the concatenated multi-layer outputs used for depth estimation.

Following an analysis similar to [58], we observe that, for depth prediction, tokens from layer 17 contribute significantly more than those from the other three aggregation layers. Based on this observation, we restrict difusion to the scene tokens extracted from layer 17 for the target view(s), substantially reducing computational cost while preserving the most depth-relevant representation. To mitigate potential information loss due to this constraint, we use the outputs from all four aggregation layers of the source views as conditioning signals. This allows the model to access richer multi-scale geometric information during crossattention, while keeping the difusion process tractable. We provide additional details of this analysis in the supplementary material.

During training, only the target-view tokens $x _ { 0 }$ are noised, while the sourceview tokens $z _ { s }$ remain clean and serve solely as conditioning signals. The forward difusion process corrupts the target tokens over timesteps $t \in \{ 1 , \dots , T \} \colon x _ { t } =$ $\sqrt { \alpha _ { t } } x _ { 0 } + \sqrt { 1 - \alpha _ { t } } \epsilon$ , where $\epsilon \sim \mathcal { N } ( 0 , I )$ and $\begin{array} { r } { \alpha _ { t } = \prod _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) } \end{array}$ , with $\{ \beta _ { t } \} _ { t = 1 } ^ { T }$ defining the variance schedule.

The denoiser $f _ { \theta }$ is trained to predict the velocity of injected noise: $\hat { v } _ { t } \ =$ $f _ { \theta } ( x _ { t } , t , z _ { s } , p )$ , where $x _ { t }$ are the noised target tokens. Conditioning on $z _ { s }$ is implemented through cross-attention layers inserted in each DiTBlock, allowing the model to extract geometry most relevant to the target view. The pose embedding $p$ is added to the timestep embedding.

Because standard noise schedulers are typically designed for lower-dimensional latent spaces $( \mathrm { e . g . }$ , dimension $\lesssim 1 0 2 4 )$ , directly applying them to the higherdimensional 3DFM token space (typically ≥ 1024) can lead to instability. Following [60], we apply a timestep shift to the noise scheduler, inspired by [6], which stabilizes training and enables efective denoising in this richer latent space.

Figure 2 shows Z3D’s inference pipeline. We initialize $x _ { T } \sim \mathcal { N } ( 0 , I )$ and iteratively apply $f _ { \theta }$ to recover an estimate of the clean target tokens $x _ { 0 }$ . The reconstructed tokens are then passed through the pretrained decoder D to produce the final target-view depth map: $\hat { d } = D ( x _ { 0 } )$

DPT Head, D. The DPT head acts as a decoder, analogous to the decoder in a variational autoencoder (VAE) [27]. It is the depth prediction heads of the 3DFMs. The sampled target-view scene representation tokens, $x _ { 0 }$ produced by the difusion model are passed through the DPT head, which decodes them into depth maps for the target views.

## 3.3 Implementation Details

We implement Z3D using PyTorch and the modified DiT model outlined in § 3.2). Our difusion model is trained using a flow-matching noise scheduler with a flowvelocity prediction objective, following the formulation introduced in Stable Diffusion v3 [6]. During training , we apply FlowMatchEulerDiscreteScheduler [6] noise scheduler with 1000 timesteps. We adjust the timestep shift factor using a dimension-dependent scaling rule, $\alpha = \sqrt { m / n }$ , where the scaling compensates for changes in the latent dimensionality. The timestep shift factor in an Euler flow matching scheduler is a small ofset applied to time steps to avoid evaluating the model exactly at the boundary times (0 and 1), improving numerical stability during sampling. Following [6,60], we use $n = 4 0 9 6$ as the reference base dimension and set m to the efective data dimension of the 3DFM backbone representation. At inference time, we apply the FlowMatchEulerDiscreteScheduler [60] scheduler and only sample 50 steps. We train our model in two stages. In the first stage, the difusion model is trained under a one-source–one-target view setting using an efective batch size of 128 for 98k training steps. In the second stage, the model is initialized with the weights obtained from Stage 1 and further trained using a multi-view configuration consisting of two source views and four target views, with an efective batch size of 32 for 156k steps.

Our model is not sensitive to batch size; however, we adopt this incremental training strategy because training the Stage 2 configuration from scratch was observed to be unstable and led to slower convergence. Initializing from Stage 1 led to much stable optimization and faster convergence. We train the model using the AdamW optimizer with $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 5$ , no weight decay, and a learning rate of $2 \times 1 0 ^ { - 4 }$ for both training stages. We apply a linear warmup over the first 10% of the total training steps, followed by a learning rate decay schedule that begins at 30% of the training progress and gradually decreases the learning rate to zero by the end of training.

## 3.4 Training Datasets

Z3D is trained on sequences of overlapping camera views, typically containing 2–6 images per sequence. The supplement has more data preparation details, but briefly: Training samples are constructed by mining 3D datasets to identify groups of views with suficient geometric overlap (defined via pairwise camera frustum overlap).

To ensure robustness and broad applicability, Z3D is trained on a largescale aggregation of five diverse datasets that are both real and synthetic. This combined corpus provides extensive coverage across indoor and outdoor environments.

The specific datasets include MegaDepth [22], Hypersim [31], Taskonomy [55], Replica [43], and Habitat HM3D [37].

## 4 Experiments

In this section, we present the experimental framework for Z3D. Our system is designed to predict novel-view depths from sparse input views using 3DFMs. Given the novelty of this problem, there are no existing methods that directly address this setting. To evaluate Z3D, we construct datasets as described in § 3.4 and perform a comprehensive experimental analysis. We consider two recent 3DFMs: VGGT [45] and WorldMirror [23].

## 4.1 Baselines

Several works attempt to predict novel-view depths jointly with novel-view synthesis [10, 11]. However, to the best of our knowledge, there are no existing methods that repurpose 3DFMs specifically for this task. To address this, we construct strong baselines from methods that tackle parts of the problem. Our work is closely related to MVGD [10]. Unfortunately, since the code for MVGD [10] is not available, we were unable to perform a direct comparison.

LVSM + 3DFM. We use LVSM [18] as a baseline because it is a state-ofthe-art, scene-agnostic novel view synthesis model that generates novel RGB views from sparse observations and camera parameters (intrinsics and extrinsics). Since LVSM does not directly predict depth, we apply pre-trained 3DFMs (e.g., VGGT and WorldMirror) to the synthesized images to estimate depth for the posed cameras. This baseline evaluates the extent to which a generalpurpose image synthesis model, combined with a strong monocular 3D foundation model, can be used for novel-view depth prediction. We denote these variants as LVSM+VGGT and LVSM+WM, corresponding to LVSM followed by VGGT and WorldMirror depth estimation, respectively.

Depth Difusion (DD). As an additional baseline, we modify our approach by training the difusion model directly in pixel space on the predicted depth maps from 3DFMs, rather than modeling the distribution of the target scene’s latent representation. To separate out the efects of the architecture from the space in which inference is done, we use the Z3D architecture but change the output space. We consider two variants of this baseline: Depth Difusion (VGGT), hereafter referred to as VGGT-DD, and Depth Difusion (WorldMirror), hereafter referred to as WM-DD.

## 4.2 Evaluation

In this section, we describe our quantitative and qualitative evaluations.

Datasets. We follow a similar procedure as outlined in § 3.4 to preprocess datasets for evaluation. We evaluate our model on both indoor and outdoor scenes, covering a diverse set of environments. Specifically, we perform quantitative evaluation on DTU [1], NRGBD [3], 7-Scenes [42], and an in-domain evaluation set sampled from the test splits of training datasets.

Depth Estimation Metrics. Following the afine-invariant depth evaluation protocol [29], we first align the predicted depth maps <sup>ˆ</sup>d to the ground truth d using least squares fitting (Weiszfeld method [5]), producing the absolute aligned depth map $\boldsymbol { a } = \boldsymbol { \hat { d } } \times \boldsymbol { s } + \boldsymbol { t }$ in the same units as the ground truth. The depth quality is then assessed using two widely adopted metrics [29,30,53]. The first, Absolute Mean Relative Error (AbsRel), is computed as AbsRel $\begin{array} { r } { = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } | a _ { i } - d _ { i } | / d _ { i } . } \end{array}$ where M is the total number of pixels. The second metric, $\delta < 1 . 2 5$ , measures the fraction of pixels satisfying max $( a _ { i } / d _ { i } , d _ { i } / a _ { i } ) < 1 . 2 5$

Multiview 3D Reconstruction Metrics. Since we predict novel depths for multiple cameras in a scene, we evaluate multiview 3D reconstruction by projecting the predicted depthmaps to point clouds. Consistent with prior works [3, 44, 48, 49], we report Accuracy (Acc.), and Completion (Comp.), which together quantify both the fidelity and completeness of the reconstructed 3D geometry.

![](images/955acb255969ac5497260d21c2cd84a1f94e841f53163a74cad7da133805e206.jpg)  
Fig. 3: Qualitative comparison of Novel View Depth Predictions. Columns show: source RGB, target RGB, ground truth depth, 3DFM predicted depth, Depth Difusion prediction, and Z3D prediction. Z3D results are substantially more smooth as compared to results from a method that works directly in patch-space.

Qualitative Results. Figure 3 presents qualitative results of Z3D, and Depth Difusion (DD) baselines. Both Z3D-VGGT and Z3D-WM predict reasonable novel-view depth when conditioned on the target camera pose. However, DD methods yield noticeably less smooth depth maps, likely because they operate directly in depth space rather than a latent representation.

To further substantiate this observation, we visualize the depth gradients of predictions from Z3D and DD baselines in Figure 4. In Figure 4, we observed pronounced gradient spikes in VGGT-DD depths in comparison to Z3D-VGGT, a trend that is consistently observed when comparing WM-DD and Z3D-WM.

We also visualize point clouds obtained by projecting predicted depth maps for novel viewpoints in Figures 5 and 6. In Figure 5, DD-based methods produce noticeably noisier point clouds compared to Z3D predictions. In the multi-view setting (Figure 6), Z3D generates consistent depth across novel views given only a few source views (two in this case).

Quantitative Evaluation We evaluate our method on out-of-domain datasets under two settings: 1-source → 1-target and 2-sources → 4-targets, to assess its generalization ability across unseen domains. Our approach is flexible with respect to the number of input views and performs well in both few-view and sparse-view settings. Under 2-view setting, our models often outperform our baselines on depth map metrics and are comparable in point cloud reconstruction metrics. However, we qualitatively find that Z3D produces smoother results than the depth difusion methods (see Figure 4).

Target View Image  
![](images/c69f7523fec93c5e71dfb864c1b1070b6764973ae52201b525ec1075b5038c5f.jpg)

![](images/ea73ea4d4f202695810d6955b28286a5500748518f0a2c457adfbc5e07202d49.jpg)  
Predicted Depth

![](images/3bb9507fed5b9f17d6716aa0805c26a507834b66bb77ec590ba4951eb8bcf5e7.jpg)

![](images/47d3f8227c630be97a27e072b73ffee60bcfca6c8a544a5f3a84f7efb9d9379d.jpg)  
Point Cloud

![](images/56d1bec916d31a59d18f329516b825cda877d0ed3bb345346e6bba3f1658fc48.jpg)

![](images/47e2e9c9e4198b14457d0f2212d20b735d188a584099b98bf6173cfd8c2d1602.jpg)  
Depth Gradient

![](images/24a35223a96618cb9cf45583b8ce9203f7a631568b890d001f77c39923423b74.jpg)

![](images/a3368356be2c869bad5aef018775a8c422a8483df7fdad4fea509e8dcc18240f.jpg)  
Zoomed Section

Fig. 4: Depth gradient analysis of Z3D-VGGT and VGGT-DD model predictions  
![](images/491aabc1483963c8ceecfb11e4f09537a10d84799a38929b744b4b7a8184ddf3.jpg)  
Fig. 5: Qualitative comparison of novel-view point cloud reconstructions from a single source image and target pose. Although VGGT-DD and WM-DD achieves competitive quantitative performance, its reconstructed point clouds exhibit noticeably higher noise and reduced geometric sharpness compared to Z3D-VGGT and Z3D-WM, which produce cleaner and more coherent 3D structures. Top left: source view (left of the vertical line). Bottom left: target view (not observed by the model). First row: Predictions of VGGT , VGGT-DD and Z3D-VGGT respectively. Second row: Predictions of WM , WM-DD and Z3D-WM respectively.

In-domain Evaluation. In this setting, we evaluate Z3D and DD models on subsets of the test splits from the training datasets. As shown in Tables 2, 3, 4, and 5, all models achieve strong performance on their respective training distributions.

![](images/4ab189833dbf33b675be195416ae530d4372124c50973c8fbaea0730106638a6.jpg)  
Fig. 6: Novel-view depth prediction from 2 source views → 4 target views. Conditioned on the target camera poses, Z3D-VGGT and Z3D-WM generate geometrically consistent depth maps across all target views, resulting in a coherent reconstructed point cloud.

Out-of-domain Evaluation. Here, we evaluate our method and baselines on the test splits of datasets not seen during the training of our model and baselines. Our LVSM+3DFM baselines provide a strong comparison since LVSM is trained end-to-end for novel-view synthesis and is expected to encode scene geometry accurately. However, we observe that LVSM sometimes struggles to generate novel views that are fully consistent with the target pose, suggesting that optimizing for perceptual quality can come at the expense of faithfully preserving the underlying scene geometry.

From Tables 2 and 3, our Depth Difusion baselines (WM-DD and VGGT-DD) are competitive but still struggle to produce accurate novel-view depths. This stems from the fact that they learn a direct mapping from observed depth to novel-view depth, where depth alone is not a suficiently rich representation of full 3D scene geometry. In contrast, 3DFMs predict depth from a richer latent scene representation encoded in the backbone, with the depth head acting as a lightweight decoder. While this head captures some geometric cues, it is less expressive than the full latent features, which better support reasoning about scene structure from novel viewpoints. Z3D-VGGT and Z3D-WM consistently outperform our baselines in out-of-domain evaluations in point clouds accuracy and depth map metrics and perform on par with each other across all experiments. This indicates that the 3D representations learned by both 3DFMs capture similar geometric structures of the scene. We attribute the strong performance of Z3D-WM and Z3D-VGGT to the rich geometric scene representations learned by these models, as well as our implicit modeling of novel-depth synthesis through the difusion framework.

Table 2: Point Cloud Metrics (1 source → 1 target). Best results per dataset in bold, second best underlined. Z3D performs well, especially in accuracy, and often has comparable completeness. Blue color indicates out of domain evaluation. Note: M means mean and Md means median
<table><tr><td rowspan="3">Model</td><td colspan="4">In-Domain</td><td colspan="4">DTU</td><td colspan="4">Out of Domain 7-Scenes</td><td colspan="4">NRGBD</td></tr><tr><td colspan="4">Acc ↓</td><td colspan="4"></td><td colspan="4"></td><td colspan="4"></td></tr><tr><td>M</td><td>Md</td><td>Comp ↓ M</td><td>Md</td><td>M</td><td>Acc ↓ Md</td><td>M</td><td>Comp ↓ Md</td><td>M</td><td>Acc ↓ Md</td><td>M</td><td>Comp ↓ Md</td><td>M</td><td>Acc ↓ Md</td><td>M</td><td>Comp ↓ Md</td></tr><tr><td>LVSM+VGGT</td><td></td><td>一</td><td></td><td></td><td>7.681</td><td>6.503</td><td>23.396</td><td>11.958</td><td>0.065</td><td>0.042</td><td>0.184</td><td>0.091</td><td>0.168</td><td>0.097</td><td>0.369</td><td>0.204</td></tr><tr><td>LVSM+WM</td><td></td><td></td><td></td><td></td><td>7.590</td><td>6.304</td><td>23.242</td><td>11.512</td><td>0.063</td><td>0.038</td><td>0.178</td><td>0.083</td><td>0.168</td><td>0.095</td><td>0.385</td><td>0.217</td></tr><tr><td>VGGT-DD</td><td>0.154</td><td>0.044</td><td>0.149</td><td>0.013</td><td>6.337</td><td>4.386</td><td>1.644</td><td>1.155</td><td>0.046</td><td>0.029</td><td>0.029</td><td>0.010</td><td>0.255</td><td>0.092</td><td>0.192</td><td>0.743</td></tr><tr><td>WM-DD</td><td>0.130</td><td>0.053</td><td>0.068</td><td>0.013</td><td>11.282</td><td>7.379</td><td>1.737</td><td>1.335</td><td>0.042</td><td>0.029</td><td>0.024</td><td>0.009</td><td>0.159</td><td>0.084</td><td>0.127</td><td>0.837</td></tr><tr><td>Z3D-VGGT</td><td>0.121</td><td>0.020</td><td>0.181</td><td>0.021</td><td>2.725</td><td>1.782</td><td>2.938</td><td>2.225</td><td>0.033</td><td>0.018</td><td>0.046</td><td>0.022</td><td>0.340</td><td>0.056</td><td>0.260</td><td>0.676</td></tr><tr><td>Z3D-WM</td><td>0.083</td><td>0.022</td><td>0.100</td><td>0.021</td><td>5.021</td><td>2.408</td><td>3.827</td><td>2.610</td><td>0.029</td><td>0.017</td><td>0.038</td><td>0.019</td><td>0.154</td><td>0.050</td><td>0.133</td><td>0.826</td></tr></table>

Table 3: Depth Estimation Metrics (1 source → 1 target). Best results per dataset in bold, second best underlined. Blue color indicates out of domain evaluation.
<table><tr><td rowspan="3"></td><td colspan="2"></td><td colspan="6">Out-of-Domain</td></tr><tr><td colspan="2">In-Domain</td><td colspan="2">DTU</td><td colspan="2">7-Scenes</td><td colspan="2">NRGBD</td></tr><tr><td>AbsRel↓ δ &lt; 1.25 ↑</td><td></td><td></td><td></td><td></td><td></td><td>AbsRel↓ δ &lt; 1.25 ↑ AbsRel↓ δ &lt; 1.25 ↑ AbsRel↓ δ &lt; 1.25 ↑</td><td></td></tr><tr><td>LVSM+VGGT</td><td></td><td>1</td><td>0.468</td><td>0.024</td><td>0.342</td><td>0.745</td><td>0.214</td><td>0.666</td></tr><tr><td>LVSM+WM</td><td></td><td></td><td>0.468</td><td>0.025</td><td>0.328</td><td>0.749</td><td>0.210</td><td>0.672</td></tr><tr><td>VGGT-DD</td><td>0.098</td><td>0.914</td><td>0.022</td><td>0.998</td><td>0.274</td><td>0.963</td><td>0.192</td><td>0.743</td></tr><tr><td>WM-DD</td><td>0.108</td><td>0.902</td><td>0.036</td><td>0.997</td><td>0.272</td><td>0.967</td><td>0.127</td><td>0.837</td></tr><tr><td>Z3D-VGGT</td><td>0.076</td><td>0.935</td><td>0.012</td><td>0.999</td><td>0.260</td><td>0.963</td><td>0.260</td><td>0.676</td></tr><tr><td>Z3D-WM</td><td>0.075</td><td>0.939</td><td>0.021</td><td>0.999</td><td>0.266</td><td>0.967</td><td>0.133</td><td>0.826</td></tr></table>

Outdoor Evaluation. Our training/evaluation data includes outdoor data from MegaDepth. This was included in the ‘In Domain’ grouping; we explicitly pull out the MegaDepth results in Tab. 6. Z3D substantially outperforms the DD models in depth metrics, and has substantially higher point cloud accuracy with slightly lower completeness than the DD. This also shows that Z3D method is domain agnostic and works for both indoor and outdoor scenes.

Generalization to Newer 3DFMs. We further evaluate Z3D on a recently introduced 3D foundation model, VGGT-Ω [46], demonstrating that the proposed framework is largely model-agnostic. We train Z3D-VGGT-Ω under the one-source-view → one-target-view setting without any algorithmic modifications. Tables 7 and 8 report the corresponding point cloud and depth estimation results. Z3D-VGGT-Ω achieves performance comparable to Z3D-VGGT and Z3D-WM, showing that Z3D can be readily adapted to new 3D foundation models while maintaining competitive performance.

Limitations. Z3D advances novel-view depth synthesis by leveraging 3DFM scene representations, but also has certain limitations. We observe that performance degrades when there is limited overlap between source and target views, making it dificult to recover fine-grained geometry in novel viewpoints. In our setup, the first source image is treated as the reference camera. Prior work on 3DFMs [50] has shown that the choice of reference view can afect model performance. Z3D inherits this sensitivity and reliable 3D reconstruction becomes challenging when input views provide insuficient scene coverage. Thus, with low-overlap conditions between source and target views, both Z3D-VGGT and Z3D-WM struggle to produce accurate depth predictions for novel views. Figure 9 illustrates a typical failure mode of Z3D, where the method struggles to predict novel-view depth with fine-grained details. In such cases, the resulting point clouds become inconsistent.

Table 4: Point Cloud Metrics (2 source → 4 target). Bold: best (lowest) per column per dataset, underline: second best.
<table><tr><td rowspan="3">Model</td><td colspan="4">In-Domain</td><td colspan="4">DTU</td><td colspan="4">Out of Domain 7-Scenes</td><td colspan="4">NRGBD</td></tr><tr><td colspan="4">Acc ↓</td><td colspan="4">Acc ↓</td><td colspan="4">Acc ↓</td><td colspan="4">Acc ↓</td></tr><tr><td>M</td><td>Md</td><td>M</td><td>Cmp ↓ Md</td><td>M</td><td>Md</td><td>Cmp ↓ M</td><td>Md</td><td>M</td><td>Md</td><td>M</td><td>Cmp ↓ Md</td><td>M</td><td>Md</td><td>Cmp ↓ M</td><td>Md</td></tr><tr><td>LVSM+VGGT</td><td></td><td>1</td><td></td><td></td><td>9.32</td><td>7.90</td><td>16.45</td><td>7.31</td><td>0.025</td><td>0.014</td><td>0.055</td><td>0.017</td><td>0.062</td><td>0.028</td><td>0.140</td><td>0.033</td></tr><tr><td>LVSM+WM</td><td></td><td></td><td></td><td></td><td>9.20</td><td>7.570</td><td></td><td>16.510 6.810</td><td>0.022</td><td>0.013</td><td>0.053</td><td>0.017</td><td>0.055</td><td>0.028</td><td>0.138</td><td>0.039</td></tr><tr><td>VGGT-DD</td><td>0.266</td><td>0.148</td><td>0.107</td><td>0.018</td><td>17.03</td><td>12.22</td><td>1.60</td><td>1.05</td><td>0.046</td><td>0.029</td><td>0.023</td><td>0.006</td><td>0.215</td><td>0.080</td><td>0.091</td><td>0.018</td></tr><tr><td>WM-DD</td><td>0.237</td><td>0.123</td><td>0.079</td><td>0.015</td><td>19.20</td><td>12.34</td><td>1.690</td><td>1.05</td><td>0.054</td><td>0.037</td><td>0.023</td><td>0.007</td><td>0.113</td><td>0.069</td><td>0.074</td><td>0.014</td></tr><tr><td>Z3D-VGGT</td><td>0.131</td><td>0.037</td><td>0.112</td><td>0.018</td><td>6.14</td><td>3.71</td><td>2.79</td><td>1.96</td><td>0.027</td><td>0.014</td><td>0.032</td><td>0.012</td><td>0.211</td><td>0.045</td><td>0.109</td><td>0.037</td></tr><tr><td>Z3D-WM</td><td>0.120</td><td>0.041</td><td>0.088</td><td>0.019</td><td>10.32</td><td>5.67</td><td>3.23</td><td>2.18</td><td>0.026</td><td>0.014</td><td>0.029</td><td>0.010</td><td>0.091</td><td>0.035</td><td>0.108</td><td>0.031</td></tr></table>

Table 5: Depth Estimation Metrics (2 sources → 4 targets). Bold: best per column per dataset, underline: second best.
<table><tr><td></td><td colspan="2">In-Domain</td><td colspan="2">DTU</td><td colspan="2">Out-of-Domain 7Scenes</td><td colspan="2">NRGBD</td></tr><tr><td>Model</td><td>AbsRel↓</td><td>δ &lt; 1.25 ↑</td><td>AbsRel↓</td><td>δ &lt; 1.25 ↑</td><td>AbsRel↓</td><td>δ &lt; 1.25 ↑</td><td>AbsRel↓</td><td>δ &lt; 1.25 ↑</td></tr><tr><td>LVSM+VGGT</td><td></td><td></td><td>0.476</td><td>0.006</td><td>0.343</td><td>0.933</td><td>0.127</td><td>0.828</td></tr><tr><td>LVSM+WM</td><td></td><td></td><td>0.475</td><td>0.006</td><td>0.346</td><td>0.935</td><td>0.121</td><td>0.842</td></tr><tr><td>VGGT-DD</td><td>0.248</td><td>0.667</td><td>0.055</td><td>0.986</td><td>0.372</td><td>0.959</td><td>0.225</td><td>0.686</td></tr><tr><td>WM-DD</td><td>0.215</td><td>0.735</td><td>0.059</td><td>0.979</td><td>0.383</td><td>0.943</td><td>0.147</td><td>0.812</td></tr><tr><td>Z3D-VGGT</td><td>0.112</td><td>0.891</td><td>0.024</td><td>0.995</td><td>0.342</td><td>0.965</td><td>0.223</td><td>0.715</td></tr><tr><td>Z3D-WM</td><td>0.118</td><td>0.889</td><td>0.036</td><td>0.987</td><td>0.362</td><td>0.962</td><td>0.131</td><td>0.830</td></tr></table>

Table 6: MegaDepth depth, point cloud metrics. 2 source → 4 targets. Bold: best, underline: second best. M: Mean, Md: Median
<table><tr><td></td><td>↓</td><td>↑</td><td colspan="2">Acc ↓</td><td colspan="2">Comp ↓</td></tr><tr><td>Model</td><td>Abs Rel</td><td>δ&lt;1.25</td><td>Mean</td><td>Med</td><td>Mean</td><td>Med</td></tr><tr><td>VGGT-DD</td><td>0.102</td><td>0.905</td><td>0.203</td><td>0.058</td><td>0.176</td><td>0.014</td></tr><tr><td>WM-DD</td><td>0.111</td><td>0.896</td><td>0.241</td><td>0.060</td><td>0.163</td><td>0.013</td></tr><tr><td>Z3D-VGGT</td><td>0.039</td><td>0.976</td><td>0.091</td><td>0.020</td><td>0.213</td><td>0.023</td></tr><tr><td>Z3D-WM</td><td>0.041</td><td>0.977</td><td>0.110</td><td>0.025</td><td>0.209</td><td>0.026</td></tr></table>

Table 7: Point Cloud Metrics (1 source → 1 target). Bold: best (lowest) per column per dataset, underline: second best.
<table><tr><td rowspan="3">Model</td><td colspan="4">In-Domain</td><td colspan="8">Out of Domain</td><td colspan="4">NRGBD</td></tr><tr><td colspan="4"></td><td colspan="4">DTU</td><td colspan="4">7-Scenes Acc ↓</td><td colspan="4"></td></tr><tr><td>Acc ↓ Md</td><td></td><td>Comp ↓ M</td><td>Md</td><td>Acc ↓ M</td><td>Md</td><td>M</td><td>Comp ↓ Md</td><td>M</td><td>Md</td><td>M</td><td>Comp ↓ Md</td><td></td><td> $\operatorname { W } ^ { \mathrm { A c c \downarrow } } = \operatorname { W } _ { \mathrm { M } }$ </td><td>M</td><td>Comp ↓ Md</td></tr><tr><td>VGGT-DD</td><td>0.154</td><td>0.044</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WM-DD</td><td>0.130</td><td>0.053</td><td>0.149 0.068</td><td>0.013 0.013</td><td>6.337 11.282</td><td>4.386 7.379</td><td></td><td>1.644 1.155 1.335</td><td>0.046 0.042</td><td>0.029 0.029</td><td>0.029 0.024</td><td>0.010 0.009</td><td>0.255 0.159</td><td>0.092 0.084</td><td>0.097 0.084</td><td>0.025 0.019</td></tr><tr><td>Z3D-VGGT</td><td>0.121</td><td>0.020</td><td></td><td>0.021</td><td>2.725</td><td></td><td>1.737 2.938</td><td>2.225</td><td>0.033</td><td>0.018</td><td>0.046</td><td>0.022</td><td>0.340</td><td>0.056</td><td></td><td>0.053</td></tr><tr><td>Z3D-WM</td><td>0.083</td><td>0.022</td><td>0.181 0.100</td><td>0.021</td><td>5.021</td><td>1.782 2.408</td><td>3.827</td><td>2.610</td><td>0.029</td><td>0.017</td><td>0.038</td><td>0.019</td><td>0.154</td><td>0.050</td><td>0.156 0.139</td><td>0.050</td></tr><tr><td>Z3D-VGGT-Ω</td><td>0.108</td><td>0.029</td><td>0.114</td><td>0.029</td><td>4.834</td><td>3.117</td><td>4.279</td><td>3.386</td><td>0.030</td><td>0.017</td><td>0.032</td><td>0.019</td><td>0.119</td><td>0.039</td><td>0.127</td><td>0.034</td></tr></table>

Table 8: Depth Estimation Metrics (1 sources → 1 target). Bold: best per column per dataset, underline: second best.
<table><tr><td></td><td colspan="2">In-Domain</td><td colspan="2">DTU</td><td colspan="2">Out-of-Domain 7-Scenes</td><td colspan="2">NRGBD</td></tr><tr><td>Model</td><td>AbsRel↓ δ</td><td>1.25</td><td>AbsRel↓ δ &lt; 1.25 ↑ AbsRel↓ δ &lt; 1.25 ↑ AbsRel↓ δ &lt; 1.25 ↑</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VGGT-DD</td><td>0.098</td><td>0.914</td><td>0.022</td><td>0.998</td><td>0.274</td><td>0.963</td><td>0.192</td><td>0.743</td></tr><tr><td>WM-DD</td><td>0.108</td><td>0.902</td><td>0.036</td><td>0.997</td><td>0.272</td><td>0.967</td><td>0.127</td><td>0.837</td></tr><tr><td>Z3D-VGGT</td><td>0.076</td><td>0.935</td><td>0.012</td><td>0.999</td><td>0.260</td><td>0.963</td><td>0.260</td><td>0.676</td></tr><tr><td>Z3D-WM</td><td>0.075</td><td>0.939</td><td>0.021</td><td>0.999</td><td>0.266</td><td>0.967</td><td>0.133</td><td>0.826</td></tr><tr><td>Z3D-VGGT-Ω</td><td>0.096</td><td>0.925</td><td>0.019</td><td>0.999</td><td>0.305</td><td>0.978</td><td>0.126</td><td>0.837</td></tr></table>

![](images/ae48a7a4aa376a044b34fe3dfed3537490111e7f3e0fe2ed94a35dcd3003ddf1.jpg)  
Fig. 7: Point maps obtained from the depth maps predicted by Z3D-VGGT and Z3D-WM. Both Z3D-VGGT and Z3D-WM struggles to predict depths when source views have limited overlap with target views. The depth maps from Z3D-VGGT and Z3D-WM preserve the overall scene structure, but finer details are missing, and the resulting projected point clouds is not consistent.

## 5 Conclusion

We present Z3D, novel depth synthesis model that requires at atleast one view and a new view pose to predict the novel depth of that new view. Our method can take an arbitrary number of posed images and predict the depth of novel views conditioned on the camera pose and latent scene representation of the scene extracted from 3DFMs.

## Acknowledgements

This work was partially supported by NSF #2437330 and NYU IT High Performance Computing resources, services, and staf expertise. Thanks to Sarah Jabbour, Joseph Tung, Ruoyu Wang, and V. Samuel Pérez-Díaz for helpful feedback.

## References

1. Aanæs, H., Jensen, R.R., Vogiatzis, G., Tola, E., Dahl, A.B.: Large-scale data for multiple-view stereopsis. International Journal of Computer Vision pp. 1–16 (2016)

2. Agarwal, S., Furukawa, Y., Snavely, N., Simon, I., Curless, B., Seitz, S.M., Szeliski, R.: Building rome in a day. Commun. ACM 54(10), 105–112 (Oct 2011). https: //doi.org/10.1145/2001269.2001293, https://doi.org/10.1145/2001269. 2001293

3. Azinović, D., Martin-Brualla, R., Goldman, D.B., Nießner, M., Thies, J.: Neural rgb-d surface reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 6290–6301 (June 2022)

4. Banani, M.E., Raj, A., Maninis, K.K., Kar, A., Li, Y., Rubinstein, M., Sun, D., Guibas, L., Johnson, J., Jampani, V.: Probing the 3D Awareness of Visual Foundation Models . In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 21795–21806. IEEE Computer Society, Los Alamitos, CA, USA (Jun 2024). https://doi.org/10.1109/CVPR52733.2024.02059, https://doi.ieeecomputersociety.org/10.1109/CVPR52733.2024.02059

5. Beck, A., Sabach, S.: Weiszfeld’s method: Old and new results. J. Optim. Theory Appl. 164(1), 1–40 (Jan 2015). https://doi.org/10.1007/s10957-014-0586-7, https://doi.org/10.1007/s10957-014-0586-7

6. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Rombach, R.: Scaling rectified flow transformers for high-resolution image synthesis. In: Proceedings of the 41st International Conference on Machine Learning. ICML’24, JMLR.org (2024)

7. Furukawa, Y., Hernández, C.: Multi-view stereo: A tutorial. Found. Trends Comput. Graph. Vis. 9, 1–148 (2015), https://api.semanticscholar.org/CorpusID: 61831046

8. Furukawa, Y., Ponce, J.: Accurate, dense, and robust multi-view stereopsis. 2007 IEEE Conference on Computer Vision and Pattern Recognition pp. 1–8 (2007), https://api.semanticscholar.org/CorpusID:2845053

9. Galliani, S., Lasinger, K., Schindler, K.: Massively parallel multiview stereopsis by surface normal difusion. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV) (December 2015)

10. Guizilini, V., Irshad, M.Z., Chen, D., Shakhnarovich, G., Ambrus, R.: Zero-shot novel view and depth synthesis with multi-view geometric difusion. In: Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR). pp. 764–776 (June 2025)

11. Guizilini, V., Vasiljevic, I., Fang, J., Ambru, R., Shakhnarovich, G., Walter, M.R., Gaidon, A.: Depth field networks for generalizable multi-view scene representation. In: Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXII. p. 245–262. Springer-Verlag, Berlin, Heidelberg (2022). https://doi.org/10.1007/978-3-031-19824-3\_15, https://doi.org/10.1007/978-3-031-19824-3\_15

12. Guizilini, V.C., Tokmakov, P., Dave, A., Ambrus, R.: Grin: Zero-shot metric depth with pixel-level difusion. 2025 International Conference on 3D Vision (3DV) pp. 112–122 (2024), https://api.semanticscholar.org/CorpusID:272689130

13. Guizilini, V.C., Vasiljevic, I., Chen, D., Ambrus, R., Gaidon, A.: Towards zeroshot scale-aware monocular depth estimation. 2023 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 9199–9209 (2023), https://api. semanticscholar.org/CorpusID:259309440

14. Hartley, R., Zisserman, A.: Multiple View Geometry in Computer Vision. Cambridge University Press, USA, 2 edn. (2003)

15. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. In: Proceedings of the 34th International Conference on Neural Information Processing Systems. NIPS ’20, Curran Associates Inc., Red Hook, NY, USA (2020)

16. Jang, W., Weinzaepfel, P., Leroy, V., Agapito, L., Revaud, J.: Pow3r: Empowering unconstrained 3d reconstruction with camera and scene priors. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 1071–1081 (2025), https://api.semanticscholar.org/CorpusID:277244721

17. Jiang, H., Tan, H., Wang, P., Jin, H., Zhao, Y., Bi, S., Zhang, K., Luan, F., Sunkavalli, K., Huang, Q., Pavlakos, G.: Rayzer: A self-supervised large view synthesis model. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 4918–4929 (October 2025)

18. Jin, H., Jiang, H., Tan, H., Zhang, K., Bi, S., Zhang, T., Luan, F., Snavely, N., Xu, Z.: Lvsm: A large view synthesis model with minimal 3d inductive bias. In: The Thirteenth International Conference on Learning Representations (2025), https: //openreview.net/forum?id=QQBPWtvtcn

19. Ke, B., Obukhov, A., Huang, S., Metzger, N., Daudt, R.C., Schindler, K.: Repurposing difusion-based image generators for monocular depth estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)

20. Kerbl, B., Kopanas, G., Leimkuehler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4) (Jul 2023). https: //doi.org/10.1145/3592433, https://doi.org/10.1145/3592433

21. Leroy, V., Cabon, Y., Revaud, J.: Grounding image matching in 3d with mast3r (2024)

22. Li, Z., Snavely, N.: Megadepth: Learning single-view depth prediction from internet photos. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2018)

23. Liu, Y., Min, Z., Wang, Z., Wu, J., Wang, T., Yuan, Y., Luo, Y., Guo, C.: Worldmirror: Universal 3d world reconstruction with any-prior prompting. arXiv preprint arXiv:2510.10726 (2025)

24. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: representing scenes as neural radiance fields for view synthesis. Commun. ACM 65(1), 99–106 (Dec 2021). https://doi.org/10.1145/3503250, https:// doi.org/10.1145/3503250

25. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph. 41(4), 102:1–102:15 (Jul 2022). https://doi.org/10.1145/3528223.3530127, https://doi.org/10. 114 /3 28223 3 30127

26. Niemeyer, M., Barron, J.T., Mildenhall, B., Sajjadi, M.S.M., Geiger, A., Radwan, N.: Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 5470–5480 (2021), https://api.semanticscholar.org/CorpusID: 244773517

27. van den Oord, A., Vinyals, O., Kavukcuoglu, K.: Neural discrete representation learning. In: Proceedings of the 31st International Conference on Neural Information Processing Systems. p. 6309–6318. NIPS’17, Curran Associates Inc., Red Hook, NY, USA (2017)

28. Peebles, W., Xie, S.: Scalable difusion models with transformers. In: 2023 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 4172–4182 (2023). https://doi.org/10.1109/ICCV51070.2023.00387

29. Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., Koltun, V.: Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 1623–1637 (2019), https://api.semanticscholar.org/CorpusID:195776274

30. Ranftl, R., Bochkovskiy, A., Koltun, V.: Vision transformers for dense prediction. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 12159–12168 (2021). https://doi.org/10.1109/ICCV48922.2021.01196

31. Roberts, M., Ramapuram, J., Ranjan, A., Kumar, A., Bautista, M.A., Paczan, N., Webb, R., Susskind, J.M.: Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 10912–10922 (October 2021)

32. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models (2021)

33. Saharia, C., Chan, W., Chang, H., Lee, C., Ho, J., Salimans, T., Fleet, D., Norouzi, M.: Palette: Image-to-image difusion models. In: ACM SIGGRAPH 2022 Conference Proceedings. SIGGRAPH ’22, Association for Computing Machinery, New York, NY, USA (2022). https://doi.org/10.1145/3528233.3530757, https://doi.org/10.1145/3528233.3530757

34. Saharia, C., Chan, W., Saxena, S., Lit, L., Whang, J., Denton, E., Ghasemipour, S.K.S., Ayan, B.K., Mahdavi, S.S., Gontijo-Lopes, R., Salimans, T., Ho, J., Fleet, D.J., Norouzi, M.: Photorealistic text-to-image difusion models with deep language understanding. In: Proceedings of the 36th International Conference on Neural Information Processing Systems. NIPS ’22, Curran Associates Inc., Red Hook, NY, USA (2022)

35. Sajjadi, M.S.M., Mahendran, A., Kipf, T., Pot, E., Duckworth, D., Lučić, M., Gref, K.: Rust: Latent neural scene representations from unposed imagery. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 17297–17306 (2023). https://doi.org/10.1109/CVPR52729.2023.01659

36. Sajjadi, M.S.M., Meyer, H., Pot, E., Bergmann, U., Gref, K., Radwan, N., Vora, S., Lucic, M., Duckworth, D., Dosovitskiy, A., Uszkoreit, J., Funkhouser, T., Tagliasacchi, A.: Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations. CVPR (2022), https://srtpaper.github.io/

37. Savva, M., Kadian, A., Maksymets, O., Zhao, Y., Wijmans, E., Jain, B., Straub, J., Liu, J., Koltun, V., Malik, J., Parikh, D., Batra, D.: Habitat: A Platform for Embodied AI Research . In: 2019 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9338–9346. IEEE Computer Society, Los Alamitos, CA, USA (Nov 2019). https://doi.org/10.1109/ICCV.2019.00943, https://doi.ieeecomputersociety.org/10.1109/ICCV.2019.00943

38. Saxena, S., Herrmann, C., Hur, J., Kar, A., Norouzi, M., Sun, D., Fleet, D.J.: The surprising efectiveness of difusion models for optical flow and monocular depth estimation. In: Proceedings of the 37th International Conference on Neural Information Processing Systems. NIPS ’23, Curran Associates Inc., Red Hook, NY, USA (2023)

39. Saxena, S., Hur, J., Herrmann, C., Sun, D., Fleet, D.J.: Zero-shot metric depth with a field-of-view conditioned difusion model (2023)

40. Schönberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4104–4113 (2016). https://doi.org/10.1109/CVPR.2016.445

41. Shade, J., Gortler, S.J., wei He, L., Szeliski, R.: Layered depth images. Proceedings of the 25th annual conference on Computer graphics and interactive techniques (1998), https://api.semanticscholar.org/CorpusID:1240104

42. Shotton, J., Glocker, B., Zach, C., Izadi, S., Criminisi, A., Fitzgibbon, A.: Scene coordinate regression forests for camera relocalization in rgb-d images. In: Proceedings of the 2013 IEEE Conference on Computer Vision and Pattern Recognition. p. 2930–2937. CVPR ’13, IEEE Computer Society, USA (2013). https: //doi.org/10.1109/CVPR.2013.377, https://doi.org/10.1109/CVPR.2013.377

43. Straub, J., Whelan, T., Ma, L., Chen, Y., Wijmans, E., Green, S., Engel, J.J., Mur-Artal, R., Ren, C., Verma, S., Clarkson, A., Yan, M., Budge, B., Yan, Y., Pan, X., Yon, J., Zou, Y., Leon, K., Carter, N., Briales, J., Gillingham, T., Mueggler, E., Pesqueira, L., Savva, M., Batra, D., Strasdat, H.M., Nardi, R.D., Goesele, M., Lovegrove, S., Newcombe, R.: The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797 (2019)

44. Wang, H., Agapito, L.: 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061 (2024)

45. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2025)

46. Wang, J., Chen, M., Zhang, S., Karaev, N., Schönberger, J., Labatut, P., Bojanowski, P., Novotny, D., Vedaldi, A., Rupprecht, C.: VGGT-Ω. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2026)

47. Wang, J., Rupprecht, C., Novotny, D.: Posedifusion: Solving pose estimation via difusion-aided bundle adjustment. In: 2023 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9739–9749 (2023). https://doi.org/10.1109/ ICCV51070.2023.00896

48. Wang\*, Q., Zhang\*, Y., Holynski, A., Efros, A.A., Kanazawa, A.: Continuous 3d perception model with persistent state. In: CVPR (2025)

49. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: CVPR (2024)

50. Wang, Y., Zhou, J., Zhu, H., Chang, W., Zhou, Y., Li, Z., Chen, J., Pang, J., Shen, C., He, T.: \pi ^3 : Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347 (2025)

51. Xu, G., Lin, H., Luo, H., Wang, X., Yao, J., Zhu, L., Pu, Y., Chi, C., Sun, H., Wang, B., et al.: Pixel-perfect depth with semantics-prompted difusion transformers. arXiv preprint arXiv:2510.07316 (2025)

52. Yang, J., Sax, A., Liang, K.J., Henaf, M., Tang, H., Cao, A., Chai, J., Meier, F., Feiszli, M.: Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2025)

53. Yin, W., Zhang, J., Wang, O., Niklaus, S., Mai, L., Chen, S., Shen, C.: Learning to Recover 3D Scene Shape from a Single Image . In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 204–213. IEEE Computer Society, Los Alamitos, CA, USA (Jun 2021). https://doi.org/ 10.1109/CVPR46437.2021.00027, https://doi.ieeecomputersociety.org/10. 1109/CVPR46437.2021.00027

54. Yu, A., Ye, V., Tancik, M., Kanazawa, A.: pixelNeRF: Neural radiance fields from one or few images. In: CVPR (2021)

55. Zamir, A.R., Sax, A., Shen, W., Guibas, L.J., Malik, J., Savarese, S.: Taskonomy: Disentangling task transfer learning. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2018)

56. Zhang, J.Y., Lin, A., Kumar, M., Yang, T.H., Ramanan, D., Tulsiani, S.: Cameras as rays: Pose estimation via ray difusion. In: International Conference on Learning Representations (ICLR) (2024)

57. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image difusion models. 2023 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 3813–3824 (2023), https://api.semanticscholar.org/CorpusID: 256827727

58. Zhang, Y., Tung, J., Cai, R., Fouhey, D., Averbuch-Elor, H.: Emergent extremeview geometry in 3d foundation models (2025), https://arxiv.org/abs/2511. 22686

59. Zhao, W., Rao, Y., Liu, Z., Liu, B., Zhou, J., Lu, J.: Unleashing text-to-image difusion models for visual perception. ICCV (2023)

60. Zheng, B., Ma, N., Tong, S., Xie, S.: Difusion transformers with representation autoencoders (2025)

# Zero-Shot Novel Depth Synthesis Using 3D Foundation Models Scene Representations -Supplementary Material

Denis Mbey Akola<sup>1</sup> and David F. Fouhey<sup>1,2</sup>

<sup>1</sup> New York University, Tandon School of Engineering, 1 MetroTech Center, Brooklyn, NY 11201, USA

2 Courant Institute of Mathematical Sciences, 251 Mercer St, New York, NY 10012

## 1 Introduction

The supplementary material provides: (1) details on the curation of the datasets used for training and evaluation, (2) implementation details of our difusion models, (3) additional results and (4) the limitations of Z3D.

## 2 Training dataset curation

To efectively train and evaluate Z3D for novel depth synthesis, we construct sequences of cameras with strong geometric overlap. We select a diverse set of datasets that provide depth maps and camera annotations, enabling the generation of such camera sequences. We considered the following datasets with the statistics shown in Table 1.

For each scene in the datasets, we use all available camera poses if the scene contains fewer than 1,000 images; otherwise, we evenly subsample 1,000 cameras. Depth maps are projected across all camera pairs to compute an N×N symmetric camera-overlap matrix. Each entry is obtained by measuring frustum overlap in both directions and taking the minimum value, ensuring mutual visibility (co-visibility) between the two views. This matrix is then used to extract camera sequences with strong geometric overlap for every view in the scene. For

<table><tr><td colspan="3">Dataset Total Scenes Total Images</td></tr><tr><td>Taskonomy [55] (Tiny)</td><td>35</td><td>381,840</td></tr><tr><td>Replica [43]</td><td>18</td><td>104,397</td></tr><tr><td>Hypersim [31]</td><td>457</td><td>74,619</td></tr><tr><td>Habitat-Matterport [37]</td><td>900</td><td>900,000</td></tr><tr><td>MegaDepth [22]</td><td>196</td><td>130,000</td></tr></table>

Table 1: Total number of scenes and total number of images for each dataset used in our experiments.

Table 2: The difusion timestep is encoded by the TimestepEmbedder (Step 1) and camera poses by the PoseEncoder (Step 2). Sampled Noise is modulated with positional embeddings and RoPE for spatial structure (Step 3). The DiTBlocks (Step 4) perform self-attention, cross-attention to source views,and the output is fed into an MLP layer. Finally, the FinalLayer (Step 5) applies adaLN-Zero modulation [28], cross-attention, and an MLP to produce the output features. Note that $S _ { t } , S , P , C$ are the number of target views, number of source views, number of tokens, and token dimension respectively.
<table><tr><td>Step Inputs</td><td></td><td>Operation</td><td>Output Shape</td></tr><tr><td>1</td><td>Timestep t</td><td>TimestepEmbedder MLP</td><td>C</td></tr><tr><td>2</td><td>Camera poses (S, 12)</td><td>PoseEncoder: Fourier features + MLP</td><td> $S \times C$ </td></tr><tr><td>2</td><td>Pose embed  $( S , C )$ </td><td>Pose Embedding is added to timestep embedding</td><td> $S \times C$ </td></tr><tr><td>3</td><td>Target tokens/sampled noise x</td><td>Add sin-cos and RoPe positional embedding</td><td> $S _ { t } \times P \times C$ </td></tr><tr><td>4</td><td>Source tokens</td><td>(3), Time + pose embedding, DiTBlocks: self-attn. + cross attn. + MLP</td><td> $S _ { t } \times P \times C$ </td></tr><tr><td>5</td><td>(4)</td><td>FinalLayer: adaLN-Zero + cross-attn.  $+ \ \mathrm { M L P }$ </td><td> $S _ { t } \times P \times C$ </td></tr></table>

Habitat-Matterport, we used the Habitat 2.0 simulator to generate 1,000 RGBD observations per scene.

We follow a similar strategy for all evaluation datasets, including DTU [1], 7-Scenes [42], and NRGB-D [3]. For both training and evaluation, we use the oficial splits provided by each dataset. Tab. 1 summarizes the statistics of each training dataset.

## 3 Implementation details

Network Architecture. Tab. 2 summarizes the modified DiT model used in Z3D.

Training. Our method is implemented in PyTorch and trained in two stages. In Stage 1, the model is trained on pairs of target and source views with an efective batch size of 128 for 98k steps. In Stage 2, we train the model using multiple views (2 source views and 4 target views) with an efective batch size of 32 for 156k steps. We train the model using AdamW optimizer with $\beta _ { 1 } =$ $0 . 9 0 , \beta _ { 2 } = 0 . 9 5$ and learning rate, lr of 2e-4.

We observe that training the model on high-dimensional latents, such as those produced by 3DFMs, does not work out-of-the-box with the default noise schedulers used in popular difusion architectures. Following the empirical findings of [60], we modify the width of the DiT model to match the token dimension of the 3DFM latents.

Previous difusion studies [6, 60] have also shown that increasing the spatial resolution reduces information corruption at the same noise level, which otherwise degrades difusion training. Following [60], we therefore make the timestep shift in our noise scheduler data-dependent. Specifically, we adjust the timestep shift factor using a dimension-dependent scaling rule, $\alpha = \sqrt { m / n }$ where $m = n _ { \mathrm { t a r g e t ~ v i e w } } \times P \times C$ represents the dimensionality of the latent tokens and $n = 4 0 9 6$ is the base token dimension used for scaling. Here, $P$ denotes the number of spatial tokens per view $( P \ : = \ : 1 3 6 9$ for both VGGT [45] and WorldMirror [23]), and C is the channel dimension $\mathrm { ( C = 2 0 4 8 }$ for both VGGT and WorldMirror) and $n _ { \mathrm { t a r g e t } }$ is the number of target views.

![](images/cf016b55fe31cefd9e47745f4e844e8fdc66620babd07786712591e7e622a34c.jpg)  
Fig. 1: Linear Probe Layered Depth Image (LDI) prediction capability of 3DFMs. To test whether 3DFMs implicitly contain information about unseen views, we train a linear model to predict LDIs. Good performance on LDI prediction after the first layer suggests that the model understands the full 3D scene, including hidden surfaces. For each example: (left) input RGB; (top) ground-truth LDI layers; (bottom) linearly decoded LDI layers. The ability to recover the non-trivial structure of the scene with only a linear layer suggests that 3DFMs implicitly encode this information.

## 4 Additional LDI Results

As described in the main paper, we provide additional LDI prediction results in Figure 1 for both the pretrained VGGT and WorldMirror backbones.

## 5 Which layer features of 3DFM backbones should be used for difusion?

Recent 3DFM architectures such as VGGT [45] and WorldMirror [23] aggregate transformer tokens from multiple intermediate layers for depth prediction. Each token representation contains high-dimensional embeddings (e.g., 2048 dimensions in both VGGT and WorldMirror), making it computationally expensive to apply difusion directly to the concatenated multi-layer outputs used for depth estimation.

In this section, we analyze the contribution of tokens from diferent backbone layers to depth prediction. Our analysis is inspired by observations in [58], which indicate that certain intermediate layers carry stronger geometric signals. The analysis in [58] identifies layers 4, 11, 17, and 23 as the most informative for geometric reasoning. However, applying difusion to the tokens from all four layers simultaneously is computationally intractable due to the high dimensionality of the representations. Therefore, we analyze which of these layers is most relevant for depth prediction in order to select an appropriate layer for difusion.

To perform this analysis, we infer depth using tokens from all possible combinations of these layers. We then evaluate which individual layer or subset of layers produces depth predictions that best approximate the depth inferred using the full set of layers.

As shown in Figures 2 and 3, for both VGGT and WorldMirror, using only features from layer 17 produces depth predictions that are nearly as good as using features from all four selected layers. Moreover, combinations that include layer 17 consistently outperform those that do not. This trend holds across single-layer, two-layer, and three-layer configurations, with layer 17 producing noticeably sharper depth maps across all datasets. These results suggest that the layer 17 tokens provide the most informative features for depth difusion.

Based on this analysis, we restrict the difusion process to the scene tokens extracted from layer 17 for the target view(s). This design substantially reduces computational cost while retaining the most depth-informative representation.

To mitigate potential information loss from restricting the difusion tokens, we use the backbone layer outputs from all four aggregation layers of the source views as conditioning signals. Through cross-attention, this conditioning provides the model with richer multi-layer geometric information while keeping the difusion process computationally tractable.

Fig. 2: Depth predictions for a target view using VGGT. Rows correspond to individual layers and layer combinations predictions from VGGT backbone used to predict the depth. Columns correspond to diferent input images.
<table><tr><td>Layers</td><td colspan="4">Image 1 Image 2 Image 3 Image 4 Image 5</td><td></td></tr><tr><td>Target Image</td><td>U</td><td>立</td><td>7</td><td>Y 国</td><td>贝甲 </td></tr><tr><td>Layer 4</td><td>业</td><td>R</td><td>日</td><td>10</td><td></td></tr><tr><td>Layer 11</td><td>H</td><td>AET</td><td>2</td><td>新</td><td>J興</td></tr><tr><td>Layer 17</td><td>T</td><td>A</td><td></td><td>7</td><td></td></tr><tr><td>Layer 23</td><td>1</td><td>I</td><td></td><td>口</td><td></td></tr><tr><td>4_11</td><td>H</td><td>A面</td><td>日</td><td>Y</td><td></td></tr><tr><td>4_17</td><td>T</td><td>A</td><td>2</td><td>7</td><td>H</td></tr><tr><td>11_17</td><td>1</td><td>A</td><td></td><td>V</td><td>H</td></tr><tr><td>4_11_17</td><td>1</td><td>E</td><td>lir</td><td>7</td><td>C</td></tr><tr><td>4_11_23</td><td>HLI</td><td>面</td><td>片</td><td>E</td><td></td></tr><tr><td>4_17_23</td><td>11</td><td>A</td><td></td><td>7</td><td>C</td></tr><tr><td></td><td>1</td><td></td><td>E</td><td></td><td>J</td></tr><tr><td>All Layers</td><td></td><td>A</td><td></td><td>7</td><td></td></tr></table>

Fig. 3: Depth predictions for a target view using WorldMirror. Rows correspond to individual layers and layer combinations predictions from WorldMirror backbone used to predict the depth. Columns correspond to diferent input images.
<table><tr><td>Layers</td><td colspan="5">Image 1 Image 2 Image 3 Image 4 Image 5</td></tr><tr><td>Target Image</td><td></td><td></td><td>7</td><td></td><td>可</td></tr><tr><td>Layer 4</td><td>39</td><td></td><td>—1</td><td>面</td><td></td></tr><tr><td>Layer 11</td><td>HIAL</td><td>MEIT</td><td>2</td><td>E</td><td></td></tr><tr><td>Layer 17</td><td>T</td><td>M</td><td>-</td><td>7</td><td>L</td></tr><tr><td>Layer 23</td><td></td><td>M</td><td>M</td><td>M</td><td>F</td></tr><tr><td>4_11</td><td>HAI</td><td>TET</td><td>H</td><td>E</td><td></td></tr><tr><td>4_17</td><td></td><td></td><td>i</td><td>7</td><td>L</td></tr><tr><td>11_17</td><td>9</td><td></td><td></td><td>7</td><td></td></tr><tr><td>4_11_17</td><td>F</td><td></td><td>-</td><td>7</td><td>足</td></tr><tr><td>4_11_23</td><td>H</td><td>METT</td><td>H</td><td>Y</td><td></td></tr><tr><td>4_17_23</td><td>0</td><td></td><td>2</td><td>7</td><td>H</td></tr><tr><td></td><td>1T</td><td></td><td>E</td><td></td><td></td></tr><tr><td>All Layers</td><td></td><td>P</td><td></td><td>7</td><td></td></tr></table>

## 6 More Qualitative Results

We provide additional qualitative results, comparing our model with our depthdifusion (DD)-based baselines (WM-DD and VGGT-DD). Figure 4 shows additional examples of novel depth inference using Z3D and the DD baselines.Z3D produces substantially smoother results compared to methods operating directly in patch space. Figures 5 and 6 show 3D point cloud projections of the predicted depth maps under the 1 source → 1 target setting. Reconstructions from VGGT-DD and WM-DD exhibit higher noise and reduced geometric sharpness, whereas Z3D-VGGT and Z3D-WM produce cleaner and more coherent 3D structures. Figures 7 and 8 show the combined 3D point cloud projections of all predicted maps in the 2-source views → 4 target views setting. When there is suficient overlap across views, the models (Z3D-VGGT and Z3D-WM) excel at novel view depth prediction, producing high-fidelity point clouds.

![](images/c11a981885ad162da47aa553f22814150e900321421284fba0937cc9e7d96337.jpg)  
Fig. 4: Qualitative comparison of Novel View Depth Predictions. Columns show: source RGB, target RGB, ground truth depth (GT), 3DFM predicted depth, Depth Difusion (DD) prediction, and Z3D prediction. Z3D results are substantially more smooth as compared to results from a method that works directly in patch-space.

Source View

![](images/934279af0d71b236ac7cd6b964afe6a89ff50458ac74df7ee6c11124548c29e8.jpg)

![](images/c9405efe9958bd39caa60ab9f9cfd5ff381a763f26c3423bdcf4958babeca3de.jpg)  
GT

![](images/9b811133f6a56f050da10797ca12bb7536dcfdc402fec76fbff1ac6228d87cd6.jpg)  
VGGT-DD

![](images/e6f6f066e0dce7f4e5bad6e9cd7b7874ff5249b1bf62618019583a81767d94b1.jpg)  
Z3D-VGGT  
Target View

![](images/84f16dd83d843867cd826dd1eb0959d7ba8cbcd37b62f6dfeff1408496198cc9.jpg)

![](images/c5738063ecd8d24f48c0dde5f8849c2d7c5f210e8cd902369639f693c099733a.jpg)  
GT

![](images/598bce173bbf05d0be839b1d1c1953c54ad6460d910fa374e413686539d8c935.jpg)  
WM-DD

![](images/9ce1b9a653775092bc7bbf1389958b3401635630bf3d2630925a7fbbfebe08a8.jpg)  
Z3D-WM

Fig. 5: Qualitative comparison of novel-view point cloud reconstructions from a single source image and target pose. Reconstructions from VGGT-DD and WM-DD exhibit higher noise and reduced geometric sharpness, whereas Z3D-VGGT and Z3D-WM produce cleaner and more coherent 3D structures. Top left: source view (left of the vertical line). Bottom left: target view (not observed by the model). First row: Predictions of VGGT , VGGT-DD and Z3D-VGGT respectively. Second row: Predictions of WM , WM-DD and Z3D-WM respectively.

Target View

![](images/ec407757f4651c07c6666363508dfcfd2cc712d69e3c229ab42382826868e569.jpg)

![](images/bfa78a5dc34057be5cc5f3c63dce276d2f224b3c76dca6131b22ff10417641e7.jpg)  
GT

![](images/27d616f4dfda921f02ca90c8294590506bb16c77a9e2ec29cc98a3c2a11ef0bb.jpg)  
VGGT-DD

![](images/7c2eb9de667938565224a8773cf99aef7d1601244ad7af629e885d5c2fdba05f.jpg)  
Z3D-VGGT

![](images/2942a329ecb44684d182737a8706b158ac263ccf498fd10babe1a8169c5a9f4d.jpg)

![](images/5a408e39c7939aacd67ba0eb192c154132959592368f615c1632ae7ee1085370.jpg)

![](images/d47cd24eb084ac983e24f3257d540a32384bcf5814835f592983da2e6a84ff4c.jpg)  
WM-DD

![](images/88f9dfd5e9bb2aba16666ac001a5dda20ba7ca2caa9601ac62dfde25da0b3cd1.jpg)  
Z3D-WM

Fig. 6: Qualitative comparison of novel-view point cloud reconstructions from a single source image and target pose. Reconstructions from VGGT-DD and WM-DD exhibit higher noise and reduced geometric sharpness, whereas Z3D-VGGT and Z3D-WM produce cleaner and more coherent 3D structures. Top left: source view (left of the vertical line). Bottom left: target view (not observed by the model). First row: Predictions of VGGT , VGGT-DD and Z3D-VGGT respectively. Second row: Predictions of WM, WM-DD and Z3D-WM respectively.  
Source Views  
![](images/19752ab2f807ed49d4b49002d4153332099356e26aea956a9798e31e208909aa.jpg)

Target Views  
![](images/8d863a84e91148d9fa2dada09db2f618a5fee2e9a9591432cdd4dee94c0c7bd5.jpg)

Z3D-VGGT  
![](images/0b66d90d7d9341580d979bba4daaf125174007038135a452d9137c81184952f5.jpg)

Z3D-WM  
![](images/0e4a1b0469c1df77b12a9e75c4a5be41b401843c5a55db2ae281cfb4372d1f22.jpg)

![](images/26fc04aee0f6a10e4754227b951080ae61a4d8447aaeba5d5507d5be8efe08e0.jpg)  
GT Point Cloud

![](images/06bf62516c469d9796dc871ce4f8867313224e0803cdcfb3449035123d7d1e12.jpg)  
Z3D-VGGT Point Cloud

![](images/d49df20d6bc1a6779a0e95643c5295aa03d0aa108d9486fbad63e10cbd21823a.jpg)  
Z3D-WM Point Cloud

Fig. 7: Novel-view depth prediction from 2 source views → 4 target views. Conditioned on the target camera poses, Z3D-VGGT and Z3D-WM generate geometrically consistent depth maps across all target views, resulting in a coherent reconstructed point cloud.

![](images/af9d78da0f9c5c93769ce504e9edcfc77193f792d2072449b65b8aac50c7cfd3.jpg)  
Fig. 8: Novel-view depth prediction from 2 source views → 4 target views. Conditioned on the target camera poses, Z3D-VGGT and Z3D-WM generate geometrically consistent depth maps across all target views, resulting in a coherent reconstructed point cloud.

## 7 More Quantitative Results

We provide additional quantitative evaluation using the depths predicted by the 3DFMs as pseudo ground truth. Specifically, the target-view depth maps produced by the 3DFMs are treated as reference depths, and all methods are evaluated against them using the same depth and point cloud metrics. We report results on both in-domain data (test split of the training datasets) and outof-domain datasets (used only for evaluation and never seen during training). This experiment evaluates whether the Z3D architecture learns scene geometry consistent with the geometric priors encoded by the 3DFMs.

Tab. 3 and Tab. 4 report the point cloud and depth results, respectively, for the 1 source → 1 target setting. Tab. 5 and Tab. 6 present the corresponding results for the 2 source → 4 target setting. Across both settings, Z3D predictions closely match the target-view depths produced by the foundation models, indicating that Z3D learns geometric representations that are highly consistent with the priors encoded by the 3DFMs.

Table 3: 1 source → 1 target Point Cloud Metrics (Mean / Median). Best results per dataset are shown in bold, second best are underlined. M: Mean, Md: Median
<table><tr><td rowspan="3">Model</td><td colspan="4">In-Domain</td><td colspan="3"></td><td colspan="5">Out of Domain 7-Scenes</td><td colspan="4">NRGBD</td></tr><tr><td colspan="4"> $\operatorname { A c c } \downarrow$ </td><td colspan="4">DTU</td><td colspan="4"></td><td colspan="4"></td></tr><tr><td>M</td><td>Md</td><td>M</td><td> $\operatorname { C o m p } \downarrow$  Md</td><td>M</td><td>Acc ↓ Md</td><td>M</td><td> $\operatorname { C o m p } \downarrow$  Md</td><td>M</td><td>Acc ↓ Md</td><td>M</td><td> ${ \mathrm { C o m p ~ } } \downarrow$  Md</td><td>M</td><td>Acc ↓ Md</td><td>M</td><td>Comp ↓ Md</td></tr><tr><td>VGGT-DD</td><td>0.079</td><td>0.039</td><td>0.046</td><td>0.013</td><td>7.105</td><td>5.487</td><td>1.612</td><td>1.144</td><td>0.027</td><td>0.022</td><td>0.011</td><td>0.008</td><td>0.113</td><td>0.059</td><td>0.090</td><td>0.021</td></tr><tr><td>WM-DD</td><td>0.130</td><td>0.053</td><td>0.068</td><td>0.013</td><td>10.323</td><td>7.610</td><td>1.672</td><td>1.311</td><td>0.028</td><td>0.023</td><td>0.009</td><td>0.007</td><td>0.120</td><td>0.063</td><td>0.063</td><td>0.016</td></tr><tr><td>Z3D-VGGT</td><td>0.046</td><td>0.016</td><td>0.049</td><td>0.016</td><td>2.924</td><td>2.044</td><td>2.574</td><td>1.937</td><td>0.012</td><td>0.009</td><td>0.013</td><td>0.009</td><td>0.122</td><td>0.036</td><td>0.119</td><td>0.038</td></tr><tr><td>Z3D-WM</td><td>0.042</td><td>0.016</td><td>0.043</td><td>0.016</td><td>2.644</td><td>1.777</td><td>2.314</td><td>1.707</td><td>0.011</td><td>0.009</td><td>0.011</td><td>0.008</td><td>0.117</td><td>0.029</td><td>0.105</td><td>0.028</td></tr></table>

Table 4: Depth Estimation Metrics (AbsRel and $\delta < 1 . 2 5 )$ for 1 Source → 1 source using 3DFM predictions as GT. Best results per dataset in bold, second best underlined.
<table><tr><td></td><td colspan="2"></td><td colspan="6">Out-of-Domain</td></tr><tr><td></td><td colspan="2">In-Domain</td><td colspan="2">DTU</td><td colspan="2">7-Scenes</td><td colspan="2">NRGBD</td></tr><tr><td>Model</td><td>AbsRel↓ δ &lt; 1.25 ↑</td><td></td><td>AbsRel↓</td><td> $\delta < 1 . 2 5$ </td><td>↑AbsRel↓</td><td> $\delta < 1 . 2 5$ </td><td>↑AbsRel↓</td><td> $\delta < 1 . 2 5 \uparrow$ </td></tr><tr><td>VGGT-DD</td><td>0.083</td><td>0.937</td><td>0.022</td><td>0.998</td><td>0.033</td><td>0.993</td><td>0.737</td><td>0.727</td></tr><tr><td>WM-DD</td><td>0.108</td><td>0.902</td><td>0.034</td><td>0.997</td><td>0.031</td><td>0.992</td><td>0.112</td><td>0.871</td></tr><tr><td>Z3D-VGGT</td><td>0.050</td><td>0.960</td><td>0.012</td><td>0.999</td><td>0.016</td><td>0.995</td><td>0.451</td><td>0.730</td></tr><tr><td>Z3D-WM</td><td>0.046</td><td>0.966</td><td>0.012</td><td>0.999</td><td>0.014</td><td>0.996</td><td>0.105</td><td>0.860</td></tr></table>

Table 5: 2 source → 4 target Point Cloud Metrics (Mean / Median). Best results per dataset are shown in bold, second best are underlined.
<table><tr><td rowspan="3">Model</td><td rowspan="2" colspan="3">In-Domain</td><td rowspan="2">Out of Domain</td><td colspan="8"></td><td rowspan="2" colspan="4"></td></tr><tr><td colspan="3">DTU</td><td colspan="2"></td><td colspan="2">7-Scenes</td><td colspan="2">NRGBD</td></tr><tr><td>Acc ↓ Md</td><td></td><td>M</td><td>Comp ↓ Md</td><td>Acc ↓ M</td><td>Md</td><td>Comp ↓ M</td><td>Md</td><td>Acc ↓ M</td><td>Md</td><td>Comp ↓ M</td><td>Md</td><td>Acc ↓</td><td>Md</td><td>Comp ↓ M</td><td>Md</td></tr><tr><td>VGGT-DD</td><td>0.196</td><td>0.120</td><td>0.060</td><td>0.017</td><td>17.841</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.005</td><td>M 0.083</td><td>0.041</td><td></td><td></td></tr><tr><td>WM-DD</td><td>0.031</td><td>0.053</td><td>0.068</td><td>0.013</td><td>10.323</td><td>13.084 7.610</td><td>1.672</td><td>1.643 1.043 1.311</td><td>0.028 0.028</td><td>0.021 0.023</td><td>0.009 0.009</td><td>0.007</td><td>0.120</td><td>0.063</td><td>0.077 0.063</td><td>0.014 0.016</td></tr><tr><td>Z3D-VGGT</td><td>0.067</td><td>0.026</td><td>0.045</td><td>0.016</td><td>6.584</td><td>4.483</td><td>2.630</td><td>1.762</td><td>0.011</td><td>0.008</td><td>0.009</td><td>0.005</td><td>0.070</td><td>0.026</td><td>0.079</td><td>0.025</td></tr><tr><td>Z3D-WM</td><td>0.067</td><td>0.027</td><td>0.047</td><td>0.017</td><td>9.069</td><td>5.385</td><td>3.375</td><td>2.245</td><td>0.011</td><td>0.007</td><td>0.008</td><td>0.005</td><td>0.068</td><td>0.023</td><td>0.084</td><td>0.021</td></tr></table>

Table 6: Depth Estimation Metrics (AbsRel and $\delta < 1 . 2 5 )$ . Best results per dataset in bold, second best underlined.
<table><tr><td rowspan="3"></td><td colspan="2"></td><td colspan="6">Out-of-Domain</td></tr><tr><td colspan="2">In-Domain</td><td colspan="2">DTU</td><td colspan="2">7-Scenes</td><td colspan="2">NRGBD</td></tr><tr><td>AbsRel↓</td><td> $\delta < 1 . 2 5 \uparrow$ </td><td>AbsRel↓</td><td> $\delta < 1 . 2 5 \uparrow$ </td><td>AbsRel↓</td><td> $\delta < 1 . 2 5$ </td><td>↑AbsRel↓</td><td> $\delta < 1 . 2 5 \uparrow$ </td></tr><tr><td>VGGT-DD</td><td>0.286</td><td>0.604</td><td>0.144</td><td>0.780</td><td>0.043</td><td>0.986</td><td>0.542</td><td>0.686</td></tr><tr><td>WM-DD</td><td>0.108</td><td>0.902</td><td>0.034</td><td>0.997</td><td>0.031</td><td>0.992</td><td>0.112</td><td>0.871</td></tr><tr><td>Z3D-VGGT</td><td>0.110</td><td>0.891</td><td>0.055</td><td>0.992</td><td>0.024</td><td>0.991</td><td>0.460</td><td>0.731</td></tr><tr><td>Z3D-WM</td><td>0.118</td><td>0.879</td><td>0.070</td><td>0.966</td><td>0.022</td><td>0.992</td><td>0.115</td><td>0.841</td></tr></table>

## 8 Limitations of Z3D

While Z3D advances novel depth synthesis by leveraging 3DFM scene representations, our method has some limitations. First, we observed that when there is limited overlap between source and target views, Z3D struggles to predict finegrained depths for novel views. In our setup, the first image in the source views is treated as the reference camera. Previous work [50] on 3DFMs have shown that the choice of reference view significantly impacts 3DFM performance. Since Z3D builds on 3DFMs, it inherits their limitations: predicting 3D geometry is challenging when scene images do not have suficient overlap. As illustrated in Figures 9 the source views have limited overlap with the target views, resulting in both Z3D-VGGT and Z3D-WM struggling to produce reasonable depth maps for the novel views.

![](images/7c29f21f9b9ba96e616f0424fd9dd7f95ff3df4373cdd897e2d871d64aa810a1.jpg)  
Fig. 9: Point maps obtained from the depth maps predicted by Z3D-VGGT and Z3D-WM. Both Z3D-VGGT and Z3D-WM struggles to predict depths when source views have limited overlap with target views. The depth maps from Z3D-VGGT and Z3D-WM preserve the overall scene structure, but finer details are missing, and the resulting projected point clouds is not consistent.