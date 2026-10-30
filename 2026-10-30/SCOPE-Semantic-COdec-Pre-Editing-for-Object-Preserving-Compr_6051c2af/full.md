# SCOPE: Semantic COdec Pre-Editing for Object-Preserving Compression

Silvia Dani<sup>1</sup> , Leonardo Galteri<sup>2</sup> , and Marco Bertini<sup>1,2</sup>

<sup>1</sup> University of Florence, Florence, Italy

{name.surname}@unifi.it

<sup>2</sup> Small Pixels, Florence, Italy

{name.surname}@small-pixels.com

Abstract. Image compression is increasingly used before machine vision systems, yet standard codecs are mainly optimized for human perception rather than downstream task accuracy. We propose SCOPE (Semantic COdec Pre-Editing), a conditional image pre-editor that modifies images before compression to improve the bitrate and detection accuracy trade-off. The model is conditioned on the codec family and compression severity, allowing a single trained network to adapt to multiple compression settings without retraining or loading codec-specific weights. Instead of uniformly preserving the whole image, the pre-editor learns to protect task-relevant regions while simplifying less important background content to improve compressibility. The resulting rate-accuracy curves show consistent gains across different codec families and detector architectures, together with strong transferability to unseen learned and wavelet-based compression families.

Keywords: Image compression for machines · Object detection

## 1 Introduction

Image compression is a fundamental component of visual data transmission and storage. Most codecs, however, are designed according to human perceptual criteria, with the goal of preserving visual quality at low bitrate. This assumption is increasingly mismatched with machine-to-machine scenarios, where images are acquired, compressed, transmitted, and then consumed directly by computer vision systems rather than by human viewers. In these settings, the objective is not only visual fidelity, but the trade-off between bitrate and downstream task accuracy. Strong compression can remove fine object boundaries, alter textures, or introduce artifacts that degrade object detection while still spending bits on background content that is less relevant to the task. This reflects the known mismatch between perceptual distortion and machine-task performance [30].

This gap has motivated a growing body of work on image and video coding for machines. One line of research modifies or optimizes codecs with task-aware objectives, either by adapting standard codecs through task-driven bit allocation and rate-distortion optimization, or by training learned codecs with machine vision losses [7,14,27]. Another line of work transmits intermediate neural features rather than reconstructed images, with the aim of reducing bitrate while preserving the information needed by the downstream network [9,10,43]. Although these approaches can improve rate-accuracy performance, they often require changes to the codec, to the transmitted representation, or to the downstream task pipeline. This limits their compatibility with standard image-compression systems and existing deployment settings.

A practical alternative is to insert a learned pre-processor before compression. Deep pre-editing has shown that images can be modified before encoding to improve compression efficiency while remaining compatible with standard codecs [48]. For machine consumption, pre-processing methods edit the input image to suppress taskirrelevant information and preserve task-relevant content before compression [33,56,57]. This family of methods is attractive because the transmitted signal remains a standard compressed image and the approach can be combined with off-the-shelf codecs. However, existing pre-processing methods are often tied to a fixed codec, a fixed quality setting, or a task-specific configuration. Even task-switchable approaches focus on switching between downstream tasks, rather than adapting a single model across different codec families and compression severities. As a result, moving to a new codec or operating point may require retraining or relying on codec-specific pre-processors.

In this work, we address this limitation with SCOPE (Semantic COdec Pre-Editing), a conditional image pre-editor for compression-aware object detection. Instead of learning a separate pre-processor for each codec or bitrate regime, our model is conditioned on both the codec family and a normalized compression severity. This allows a single trained network to adapt its edits to multiple compression settings without retraining or loading codec-specific weights. The model predicts an attention map together with object and background residuals. Object regions are preserved through bounded residual edits, while background regions are simplified through a blurred-background branch to improve compressibility. This design reflects the asymmetry of object detection, where object-level cues should be protected more strongly than background texture.

The proposed formulation focuses on codec-aware pre-editing rather than task switching. The same model learns how to modify images for different compression families and severity levels, while maintaining a fixed downstream task. In this way, the conditioning mechanism captures compression-dependent editing behavior and enables transfer to unseen codec families.

The main contributions of this paper are as follows:

– We propose a conditional pre-editing framework for object detection under image compression, where a single model adapts to different codec families and compression severities through learned conditioning.

– We introduce an attention-based object and background editing design that preserves object-relevant regions while simplifying less important background content to improve compressibility.

– We train the pre-editor with differentiable codec proxies and frozen detector supervision, enabling optimization of the rate-accuracy trade-off without modifying the downstream detector.

– We show that the learned model improves rate-accuracy performance across multiple codecs and detector architectures, and transfers to unseen compression families, including learned and wavelet-based codecs.

Code, configurations, and model weights are publicly available.<sup>3</sup>

## 2 Related Work

The classification-distortion-perception analysis of [30] has shown that classification accuracy, signal distortion and perceptual quality form a three-way trade-off that cannot be jointly optimized, meaning that improving perceptual realism can actively hurt a downstream classifier at fixed rate. That result reframes machine-oriented compression as a distinct operating regime rather than a tuning of the human one, and it is the premise that motivates our work. A more codec-specific formalization arrived with the rate-distortion theory for coding for machines of [22,23], which recasts the distortion term as a task-relevant fidelity measure and derives how bits should be allocated when the receiver is an analysis network. Together, these works establish that the object of optimization is a rate-accuracy curve, not a rate-PSNR curve, which is precisely the axis on which pre-editing methods, including the present paper, report their gains.

Coding for machines The paradigm of video coding for machines was named and structured by Duan et al. [16], who framed compression and analytics as a collaborative problem and set the agenda for jointly optimizing a codec against a downstream task. This development was soon followed by a series of fully end-to-end learned codecs. Le et al. [27] introduced an image compression model trained directly under a downstream detection and recognition loss, aligning the compressed representation with vision-task performance. Chamain et al. [7] conducted a comprehensive investigation of end-to-end optimized image compression tailored for machine consumption, analyzing how compression affects various automated vision tasks. The contentadaptive extension of [28] dynamically adjusts the bit allocation according to the spatial and semantic characteristics of the image, explicitly optimizing this allocation with respect to a machine-centric objective function. A recurring theme is how much task supervision to bake into the codec: [18] proposed “omnipotent feature learning” so that a single learned representation serves many tasks rather than overfitting the codec to one; the compact-visual-representation framework of [58] proposes a unified representation compressed once and reused across analytics. A common limitation of these approaches is deployment cost: they all replace the standard codec with a trained network, so a new codec, bitrate regime, or task typically means retraining, in addition to the cost of redesigning existing video coding pipelines.

An alternative approach is to transmit neural features instead of pixels, trading codec compatibility for bitrate. The initial works in [9,10] propose to use intermediatefeature compression, while end-to-end compressible-features were proposed in [43]: the goal is to learn representations that are cheap to code while remaining informative for the downstream network. Multi-scale feature compression with QP-adaptive channel selection has been proposed in [59], scalable multilayer feature compression in [8] and multiscale-importance-based bit allocation in [32]. More recent research has begun to abstract the notion of the transmitted representation itself, extending it across heterogeneous tasks and sensor modalities. Examples include the transformable feature framework TransVFC [46], which seeks a unified task-agnostic feature space, and stereo feature coding for machines [25], which focuses on machine-centric representations of binocular data. However, this line of work implies that once the transmitted signal ceases to be a standard decodable image, it can no longer be rendered by standard image viewers or processed by existing image-based pipelines and tools. This incompatibility with the existing image ecosystems motivates the alternative design choice, that we have followed, of maintaining the output as a standards-based compressed image. Because purely machine-oriented bitstreams sacrifice human viewa-

bility, a substantial effort targets bitstreams that serve both. Scalable designs place a machine-analysis base layer under a human-viewing enhancement layer: [13] for video and the scalable image variant of [54]. This line has matured into a standardization activity, with the open evaluation dataset of [21] and the large-scale VCM benchmarking of [34] supporting the MPEG effort, and recent assessments of the emerging MPEG VCM standard on segmentation [39] and its bitstream/streaming structure [35]. A pragmatic sub-thread keeps the standard codec and adds neural pre/post modules around it like the learned intra-coding partitioning for VCM [24] and the lightweight post-processing “task enhancement tiles” [36]. These standard-compatible wrappers are the closest neighbors to pre-editing in spirit, but most operate inside or after the codec rather than as a pure, codec-agnostic front-end.

The idea that objects deserve more bits than background, central to the paper’s attention-and-residual design, has a long codec-side history. Saliency- and object driven bit allocation appears in the saliency-driven hierarchical learned codec of [19], the saliency-map-guided codec of [20,37], the efficiency-oriented saliency approach of [61], ROI-enabled learned coding [1], and object-analysis-driven use of a standard codec [42]. Semantically disentangled coding “telling the codec what is worth compressing” [31] and ROI-packing for machine vision [17] continue the theme, and recent CLIP-conditioned variants [50,60] use language-model semantics to decide what to preserve. The distinction from the present work is architectural: these methods realize the object/background asymmetry through codec-internal bit allocation, whereas the pre-editor realizes it in pixel space before an unmodified codec, via a predicted attention map that blends an object-preserving branch with a blurred background.

Preprocessing before standard codecs The closest family of approaches learns a preprocessing transform before a standard encoder, so the transmitted signal remains a conventional compressed image. Deep pre-editing [48] uses this insertion point to improve the human-oriented rate-distortion trade-off. Earlier recognition aware approaches also adapt the input for machine analysis: Choi et al. [12] study efficient compression for object detection, while Suzuki et al. [47] propose image pre-transformation for recognition-aware compression. SCOPE follows the same standard-compatible philosophy, but learns the transformation specifically for the downstream rate-accuracy trade-off.

More recent machine-oriented preprocessors suppress task-irrelevant content before encoding [33], exploit task semantics [56], or switch a shared preprocessor between downstream tasks [57]. Complementary work includes learned enhancement [2], video preprocessing under compression [62], GAN-based preprocessing [51,52], and diffusion-based recovery [15]. These methods motivate input adaptation for machine consumption, but do not explicitly condition one pre-editor on both codec family and compression severity. SCOPE instead targets codec switchability: a single set of weights is used across the evaluated compression families and operating points.

Learned compression backbones and conditioning The paper’s own novelty, a single network that adapts across codec families and severities, is a conditioning problem, and what makes end-to-end training of any pre-codec module possible is a differentiable stand-in for the non-differentiable encoder. This paper builds on the differentiable-JPEG formulation whose subtleties were analyzed by Reich et al. [40], and on learned-codec proxies descended from the variational hyperprior of [3] and the earlier nonlinear-transform-coding formulation of [4]. This is the mechanism that lets detector gradients flow back through the compression stage to the pre-editor.

## 3 Proposed Method

We propose SCOPE, a conditional pre-editing framework that modifies an input image before compression in order to preserve task-relevant content while making non-critical regions easier to compress. Given an RGB image $I \in [ 0 , 1 ] ^ { 3 \times H \times W }$ , a codec family identifier $f ,$ and a normalized compression severity $s \in [ 0 , 1 ]$ , the model produces an edited image $I _ { \mathrm { e d i t } }$ . The edited image is then compressed by the target codec, or by a differentiable codec proxy during training. In our training setting, the codec family condition is sampled from Diff-JPEG [40], BMSHJ learned image compression [3], and a BPG-like proxy. At inference time, the method requires only the input image and the desired codec condition $( f , s )$

## 3.1 Model Architecture

The architecture of the proposed conditional pre-editor is illustrated in Fig. 1. The model consists of a conditioning branch and an image editing branch. The conditioning branch encodes the target codec family and compression severity into a compact conditioning vector. The codec family f is represented by a learnable embedding $e _ { f } \in \mathbb { R } ^ { 8 }$ . The scalar severity s is expanded to a fixed nonlinear severity basis:

$$
\phi ( s ) = \left[ s , s ^ { 2 } , s ^ { 3 } , h _ { 0 . 5 0 } ( s ) , h _ { 0 . 7 5 } ( s ) , h _ { 0 . 9 0 } ( s ) \right] ,\tag{1}
$$

where each high-severity basis term is defined as

$$
h _ { \tau } ( s ) = \mathrm { c l i p } \bigg ( \frac { s - \tau } { 1 - \tau } , 0 , 1 \bigg ) .\tag{2}
$$

This expansion provides the condition MLP with both smooth polynomial severity cues and explicit high-severity indicators. This is useful because the effect of compression is not necessarily linear in the severity parameter: mild compression may require only small edits, while stronger compression can introduce rapidly increasing artifacts and may require a different pre-editing behavior. The high-severity terms allow the model to specialize its response when s exceeds predefined thresholds, instead of forcing the MLP to infer these regimes from a single scalar input.

The family embedding and the severity basis are concatenated and processed by a condition MLP:

$$
c { = } \mathrm { M L P } ( [ e _ { f } , \phi ( s ) ] ) ,\tag{3}
$$

producing a conditioning vector $c \in \mathbb { R } ^ { 6 4 }$

![](images/f293d5a2c0cb768800f6d18ebc869582c7407620ddb0a0eed556b5c3fc62e896.jpg)  
Fig. 1: Architecture of the proposed conditional pre-editor. The codec family f and severity s are encoded into a conditioning vector that modulates the U-Net-style image branch through FiLM layers. The network predicts an attention map A and two residual maps, $\varDelta _ { \mathrm { o b j } }$ and $\varDelta _ { \mathrm { b g } } .$ , which are used to construct the edited image $I _ { \mathrm { e d i t } }$ by blending an object-preserving branch with a blurred-background branch.

The image branch follows a U-Net-style encoder-decoder architecture with skip concatenations. The encoder progressively reduces the spatial resolution while increasing the number of channels, producing multi-scale feature maps at resolutions $H \times W$ $H / 2 \times W / 2$ , and $H / 4 \times W / 4$ . The decoder restores the original resolution through bilinear upsampling followed by convolution, and concatenates the corresponding encoder features through skip connections. Each convolutional block is modulated by the conditioning vector through feature-wise linear modulation (FiLM) [38]:

$$
\mathrm { F i L M } ( x , c ) = x \odot ( 1 + \gamma ( c ) ) + \beta ( c ) ,\tag{4}
$$

where $\gamma ( c )$ and $\beta ( c )$ are predicted from the conditioning vector. This conditioning mechanism allows a single pre-editor to adapt its behavior to different codec families and compression severities.

Given the input image $I ,$ the pre-editor predicts an attention map A and two residual maps, $\varDelta _ { \mathrm { o b j } }$ and $\varDelta _ { \mathrm { b g } } \mathrm { : }$

$$
A , \varDelta _ { \mathrm { o b j } } , \varDelta _ { \mathrm { b g } } = P _ { \theta } ( I , f , s ) ,\tag{5}
$$

where $A \in [ 0 , 1 ] ^ { 1 \times H \times W }$ and $\varDelta _ { \mathrm { o b j } } , \varDelta _ { \mathrm { b g } } \in \mathbb { R } ^ { 3 \times H \times W }$ . The object branch applies a bounded residual directly to the input image:

$$
I _ { \mathrm { o b j } } { = } \mathrm { c l i p } ( I { + } \varDelta _ { \mathrm { o b j } } { , } 0 , 1 ) .\tag{6}
$$

The background branch uses a blurred version of the input image as its base:

$$
I _ { \mathrm { b g } } { = } \mathrm { c l i p } ( \mathrm { B l u r } ( I ) { + } \varDelta _ { \mathrm { b g } } , 0 , 1 ) .\tag{7}
$$

![](images/ad69907a6140cba97e9003b7f74d125c09b90b72f1bb1a8ef2cbcbe69f44687c.jpg)  
Fig. 2: Training pipeline of the proposed method. The pre-editor is the only trainable module. The edited image is passed through a differentiable codec proxy, while ground-truth boxes are used only to construct training masks. A frozen detector provides a teacher-student consistency signal between the clean image and the compressed edited image. At inference time, masks and detector supervision are removed.

The final edited image is obtained by blending the two branches according to the predicted attention map:

$$
I _ { \mathrm { e d i t } } { = } A \odot I _ { \mathrm { o b j } } { + } ( 1 { - } A ) \odot I _ { \mathrm { b g } } .\tag{8}
$$

This design allows the model to preserve object-relevant regions through the object residual while simplifying background regions through the blurred-background branch.

## 3.2 Training Pipeline

The training pipeline is shown in Fig. 2. During training, the edited image is passed through a differentiable codec proxy $C ( \cdot ; f , s )$ , which approximates the effect of compression under codec family $f$ and severity s:

$$
\hat { I } { = } C ( I _ { \mathrm { e d i t } } ; f , s ) ,\tag{9}
$$

where $\hat { \boldsymbol { I } }$ denotes the compressed edited image. Since the codec proxy is differentiable, gradients from the training objectives can flow through the compression stage and update the pre-editor. The codec proxy itself is kept fixed during pre-editor training.

Training setup. We train the pre-editor from scratch on COCO 2017 [29]. For each training sample, the codec family $f$ is sampled uniformly from

$$
\mathcal { F } { = } \{ \mathrm { D i f f \_ J P E G } , \mathrm { B M S H J } , \mathrm { B P G } _ { - } \mathrm { 1 i k e } \} .\tag{10}
$$

To enable end-to-end optimization, each codec family is represented during training by a differentiable proxy. For JPEG, we use a differentiable JPEG implementation that approximates transform coding and quantization while preserving gradient flow [40,49]. For learned compression, we use the BMSHJ hyperprior model, which is already differentiable by construction and provides both reconstructed images and rate estimates [3]. Since no established differentiable proxy is commonly used for BPG, we implement a BPG-like surrogate that approximates its main compression effects through differentiable operations, allowing gradients to propagate through the compression stage during pre-editor training [5, 45].

The normalized severity s is sampled uniformly from a linear grid in $[ 0 , 1 ]$ . We use no severity-dependent loss reweighting, so the codec family and severity determine the conditioning and proxy behavior but not the relative weights of the training losses.

Input images are cropped to $5 1 2 \times 5 1 2$ , with object-centered crops sampled during training to expose the model to task-relevant regions. The pre-editor uses base channel width 48, family embedding dimension $8 ,$ conditioning dimension $6 4 ,$ , maximum object residual magnitude 0.09, maximum background residual magnitude 0.03, and a blurred-background replacement branch. We train for 40 epochs with batch size $^ { 8 , }$ learning rate $1 0 ^ { - 4 }$ , cosine scheduling, mixed-precision training, and gradient clipping. Training masks. Ground-truth bounding boxes are used to construct spatial masks that indicate which regions should be preserved and which regions can be simplified. We define three masks:

$$
M _ { \mathrm { a t t n } } , \quad M _ { \mathrm { d e t } } , \quad M _ { \mathrm { b g } } .\tag{11}
$$

The attention mask $M _ { \mathrm { a t t n } }$ is a tight object-region mask that is used to supervise the predicted attention map A. The detector mask $M _ { \mathrm { d e t } }$ covers the object and nearby context regions and is used to restrict the detector-consistency loss to task-relevant areas. The background mask is defined as

$$
M _ { \mathrm { b g } } { = } 1 { - } M _ { \mathrm { d e t } } ,\tag{12}
$$

and identifies the regions where the model is encouraged to reduce visual complexity and improve compressibility.

Detector supervision. To preserve task-relevant features after compression, we use a frozen detector D as a teacher-student supervision signal. In this training configuration, D is a Faster R-CNN detector [41] with a ResNeXt-101-32x8d-FPN backbone [55], implemented in Detectron2 [53]. The clean image I is passed through the detector to obtain teacher features, while the compressed edited image <sup>ˆ</sup>I is passed through the same frozen detector to obtain student features:

$$
F _ { T } ^ { \ell } { = } D ^ { \ell } ( I ) , \qquad F _ { S } ^ { \ell } { = } D ^ { \ell } ( \hat { I } ) ,\tag{13}
$$

where $D ^ { \ell } ( \cdot )$ denotes the feature map of the detector layer ℓ. We extract features from the FPN levels and keep the detector frozen, so detector-based supervision updates only the pre-editor.

## 3.3 Training Objective

The model is trained using a combination of background-simplification, rate-oriented, attention-supervision, detector-consistency, and residual-regularization objectives. For readability, we group the loss terms according to their role:

$$
\mathcal { L } { = } \mathcal { L } _ { \mathrm { b g } } { + } \mathcal { L } _ { \mathrm { a t t } } { + } \mathcal { L } _ { \mathrm { t a s k } } { + } \mathcal { L } _ { \mathrm { r e g } } ,\tag{14}
$$

where

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { b g } } = \lambda _ { \mathrm { b g } \_ \mathrm { f l a t } } \mathcal { L } _ { \mathrm { b g } \_ \mathrm { f l a t } } + \lambda _ { \mathrm { b g } \_ \mathrm { s m o o t h } } \mathcal { L } _ { \mathrm { b g } \_ \mathrm { s m o o t h } } + \lambda _ { \mathrm { r a t e } } \mathcal { L } _ { \mathrm { r a t e } } , } \\ & { \mathcal { L } _ { \mathrm { a t t } } = \lambda _ { \mathrm { a t t n } } \mathcal { L } _ { \mathrm { a t t n } } + \lambda _ { \mathrm { a t t n } \_ \mathrm { s p a r s e } } \mathcal { L } _ { \mathrm { a t t n } \_ \mathrm { s p a r s e } } + \lambda _ { \mathrm { a t t n } \_ \mathrm { a r e a } } \mathcal { L } _ { \mathrm { a t t n } \_ \mathrm { a r e a } } , } \\ & { \mathcal { L } _ { \mathrm { t a s k } } = \lambda _ { \mathrm { d e t } } \mathcal { L } _ { \mathrm { d e t } } , } \\ & { \mathcal { L } _ { \mathrm { r e g } } = \lambda _ { \Delta } \mathcal { L } _ { \varDelta } + \lambda _ { \mathrm { { A T V } } } \mathcal { L } _ { \mathrm { { \varDelta T V } } } . } \end{array}\tag{15}
$$

Background simplification and rate regularization. The background group ${ \mathcal { L } } _ { \mathrm { b g } }$ encourages non-object regions to be easier to compress while avoiding disruptive artifacts. We use a blur-based flatness term and a smoothness term:

$$
\mathcal { L } _ { \mathrm { b g \_ f l a t } } = \frac { \left. M _ { \mathrm { b g } } \odot \left( \hat { I } - \mathrm { B l u r } ( \hat { I } ) \right) \right. _ { 1 } } { \sum M _ { \mathrm { b g } } + \epsilon } , \qquad \mathcal { L } _ { \mathrm { b g \_ s m o o t h } } = \frac { \left. M _ { \mathrm { b g } } \odot \nabla \hat { I } \right. _ { 1 } } { \sum M _ { \mathrm { b g } } + \epsilon } .\tag{16}
$$

The rate loss $\mathcal { L } _ { \mathrm { r a t e } }$ further encourages compressibility by penalizing structures that increase the expected coding cost. For learned compression, this can be estimated from the codec proxy rate term, while for hand-crafted proxies, we use differentiable smoothness and rate surrogates.

Attention supervision. The attention group $\mathcal { L } _ { \mathrm { a t t } }$ supervises the predicted attention map and regularizes its spatial support. The main attention loss is defined by

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a t t n } } { = } \mathcal { L } _ { \mathrm { B C E } } ( A , M _ { \mathrm { a t t n } } ) { + } \mathcal { L } _ { \mathrm { D i c e } } ( A , M _ { \mathrm { a t t n } } ) . } \end{array}\tag{17}
$$

We additionally use sparsity and area regularization terms, $\mathcal { L } _ { \mathrm { a t t n \_ s p a r s e } }$ and ${ \mathcal { L } } _ { \mathrm { a t t n } }$ <sub>\_area</sub>, to discourage unnecessary high attention outside object regions.

Detector consistency. The task group $\mathcal { L } _ { \mathrm { t a s k } }$ preserves detector-relevant features after pre-editing and compression. We compare teacher and student detector features inside the detector mask using a Smooth L1 feature loss:

$$
\mathcal { L } _ { \mathrm { { d e t } } } = \sum _ { \ell } \frac { \left. M _ { \mathrm { { d e t } } } ^ { \ell } \odot \rho \left( F _ { S } ^ { \ell } - F _ { T } ^ { \ell } \right) \right. _ { 1 } } { \sum M _ { \mathrm { { d e t } } } ^ { \ell } + \epsilon } ,\tag{18}
$$

where $\rho ( \cdot )$ denotes the Smooth L1 penalty and $M _ { \mathrm { d e t } } ^ { \ell }$ is the detector mask resized to the spatial resolution of layer ℓ.

Residual regularization. The regularization group ${ \mathcal { L } } _ { \mathrm { r e g } }$ keeps the predicted edits bounded and spatially smooth. We penalize the magnitude of the object and back ground residuals:

$$
\begin{array} { r } { \mathcal { L } _ { \varDelta } = \| \varDelta _ { \mathrm { o b j } } \| _ { 1 } + \| \varDelta _ { \mathrm { b g } } \| _ { 1 } , \qquad \mathcal { L } _ { \varDelta \mathrm { T V } } = \mathrm { T V } ( \varDelta _ { \mathrm { o b j } } ) + \mathrm { T V } ( \varDelta _ { \mathrm { b g } } ) . } \end{array}\tag{19}
$$

where TV(·) denotes total variation regularization.

## 3.4 Inference

At inference time, the training-only supervision is removed. No bounding boxes, masks, or detector features are required. Given an image I, a codec family f, and a target severity s, the pre-editor produces $I _ { \mathrm { e d i t } }$ , which is then encoded by the target codec. Thus, the method remains image-only at deployment while having been trained to preserve detector-relevant content under compression.

Although our experiments focus on object detection, SCOPE is not architecturally tied to detection. The training formulation only requires a frozen downstream task model to provide task-aware supervision; in principle, the detector could be replaced by a segmentation or pose estimator without modifying the codec-facing pre-editor architecture. We therefore keep the empirical conclusions of this work specific to object detection while noting that the formulation itself is task-model agnostic.

## 4 Experiments

We evaluate the proposed pre-editor by measuring the detection mAP obtained after compression as a function of the resulting bitrate. For each codec, we compare the detector performance on compressed images without pre-editing against the performance obtained after applying the proposed pre-editor before compression. We report curves for three detector architectures: Faster R-CNN with a ResNeXt-101- 32x8d-FPN backbone [41,55], RT-DETR-l [63], and YOLO26x [26]. This allows us to assess both rate-accuracy behavior and cross-detector generalization.

Evaluation setup. All quantitative experiments use the 5,000-image COCO 2017 validation split. For the codec families used during training, we evaluate JPEG at quality values {10,15,20,30,40,50}, BPG at QPs {28,31,34,37,41,44}, and BMSHJ at quality levels {1,2,3,4,5}. For zero-shot transfer, JPEG2000 uses OpenJPEG rate parameters {32,48,64,96,128,192} and Cheng uses quality levels $\{ 1 , . . . , 6 \}$ . The severity condition linearly normalizes the operating parameter within each codec-specific range, with the direction chosen so that larger s always denotes stronger compression. Importantly, the bitrate reported in the rate-accuracy curves is measured from the actual encoded output rather than inferred from the nominal codec parameter. The supplementary reports 7.93 ms for the network forward pass and 11.81 ms total pre-editing time per image, corresponding to 84.7 FPS including tensor preparation and conditioning.

## 4.1 Results

Fig. 3 reports results for JPEG, BMSHJ, and BPG. Across the three codecs, SCOPE generally shifts the rate-accuracy curves toward higher mAP at comparable bitrate or comparable mAP at lower bitrate. For BMSHJ and BPG, the improvement is consistent across Faster R-CNN, RT-DETR-l, and YOLO26x, showing that the learned edits are not limited to the detector used for training.

JPEG is more detector-dependent. The gain is clear for Faster R-CNN and YOLO26x (−15.88% and −14.92% BDBR), whereas for RT-DETR-l it is concentrated in the low-bitrate regime and the curves converge as compression weakens (−5.71%). This behavior is consistent with the role of pre-editing: when the baseline codec already preserves most task-relevant information, there is less room to improve the detector at a fixed rate.

Although training uses Faster R-CNN feature supervision, gains also transfer to RT-DETR-l and YOLO26x across BMSHJ, BPG, the unseen Cheng codec (Sec. 4.3), and part of the JPEG range. This cross-detector behavior suggests that the pixel-space edits preserve object-relevant evidence shared across two-stage, transformer-based, and one-stage detectors rather than overfitting to a single architecture.

![](images/48f300374e93283932644cbdc18e635cf82b977f322e936e032caf8fd9bdc3aa.jpg)

![](images/6c77a9d6c06d2cb89c67c1e6d2fbc7082494d38b496ec34074e639149ea479ae.jpg)

![](images/3ffc20c2629c6f831ade17ca428f26bc67f769c52a3bd6cd1ee6f57817cc20dc.jpg)  
Fig. 3: mAP versus bitrate for the codec families used during training: JPEG, BMSHJ, and BPG. Results are shown for Faster R-CNN, RT-DETR-l, and YOLO26x.

We summarize complete rate-accuracy curves with BDBR [6], using AP as quality and bitrate as rate, over each pair’s common AP interval without extrapolation. Negative values denote bitrate savings at equal accuracy. SCOPE averages −15.98% over the nine JPEG/BPG/BMSHJ-detector pairs, all of which are negative (Table 3).

## 4.2 Qualitative Comparison

Fig. 4 shows qualitative examples of BMSHJ compression. Across the examples, SCOPE does not enforce a single visual outcome but adapts the pre-editing to the detection-rate trade-off. In many cases, it reduces the bitrate while preserving the detections obtained after baseline compression, or even suppressing false positives. In other cases, it may slightly increase the bitrate, but recovers objects that are missed by the compressed baseline. Overall, these examples illustrate the intended behavior of the method: background content is modified to improve compressibility when possible, while object regions are kept sufficiently informative or selectively enhanced to support object detection.

## 4.3 Zero-Shot Codec Transferability

We evaluate transfer to two codecs that are not explicit training targets: Cheng [11], an unseen learned codec, and JPEG2000 [44], whose wavelet-based family is absent from training (Fig. 5). On Cheng, SCOPE improves direct compression for all three detectors. For Faster R-CNN it obtains −20.66% BDBR zero-shot, while the Chengtrained task-switchable pre-processor of Yang et al. [56] (TSW-pre) obtains −31.32%. Thus, the codec-specific method remains stronger on the codec seen during its training, whereas SCOPE retains substantial savings without Cheng-specific optimization. On the matched BPG/Faster R-CNN setting, SCOPE obtains −21.74% compared with −19.50% for Lu et al. [33]. The TSW-pre result is reconstructed from its published curve, and each BDBR is evaluated only over the corresponding valid common AP interval.

JPEG2000 tests a different form of transfer because its wavelet-based family is unseen. We do not use target-codec validation to choose a family condition. Instead, the same pretrained model is evaluated without retraining under all three available conditions: Diff\_JPEG, BMSHJ, and BPG\_like. Their mean BDBRs are −1.35%, −9.97%, and −11.92%, respectively (Table 1), with BPG-like strongest for all three detectors. The common ordering across detector architectures indicates that the conditioning effect is not specific to a single detector.

![](images/ab1c9bd07407a36a1b43feb869ec76d8f37a6c630ff71037ac5263858292f8b8.jpg)  
Fig. 4: Qualitative comparison for BMSHJ compression at quality level q3. From left to right: baseline compressed image without detections and with Faster R-CNN detections, SCOPE pre-edited compressed image without detections, and with Faster R-CNN detections.

![](images/22071a4ba97200cc65347de2b155aadfd48eb2ee1f8f79c4c591ca9020f4bc97.jpg)

![](images/1657ed4407c5437a8f1ad71dc1156154f29e048bee159875cc4b4ecca801538d.jpg)  
Fig. 5: Zero-shot codec transfer. Left: mAP versus bitrate for the unseen Cheng learned codec. Right: JPEG2000, where the same pretrained SCOPE model is evaluated under each of the three available seen-family conditions.

Differentiable codecs are used only as training proxies; all evaluation curves use the actual codecs. We assume no bitstream or numerical parameter equivalence between a proxy and a real codec. Severity represents codec-normalized relative compression strength, while the family condition represents the degradation type.

Table 1: JPEG2000 zero-shot BDBR (%).
<table><tr><td>Condition</td><td>Faster R-CNN</td><td>RT- DETR-1</td><td>YOLO26x</td><td>Mean</td></tr><tr><td>Diff-JPEG</td><td></td><td></td><td></td><td></td></tr><tr><td>BMSHJ</td><td>-6.82 -11.60</td><td>+7.00 -7.42</td><td>-4.24 -10.89</td><td>-1.35 -9.97</td></tr><tr><td>BPG-like</td><td>-13.73</td><td>-9.74</td><td>-12.28</td><td>-11.92</td></tr></table>

Table 2: Matched Faster R-CNN BDBR (%).
<table><tr><td>Codec</td><td>Method</td><td>Train</td><td>BDBR</td></tr><tr><td>BPG</td><td>Lu et al. [33]</td><td>Seen</td><td>-19.50</td></tr><tr><td>BPG</td><td>SCOPE</td><td>Seen</td><td>-21.74</td></tr><tr><td>Cheng</td><td>TSW-pre [56]</td><td>Seen</td><td>-31.32</td></tr><tr><td>Cheng</td><td>SCOPE</td><td>Unseen</td><td>-20.66</td></tr></table>

![](images/7ea8496ddd1970fd06deade72294d67d8225fc652e9fdc89eec23ea4a59d8f0b.jpg)

![](images/dede064c2e220eee22c4ec638f539ecb96a829af790bfd9ab713ab7f7b8c683c.jpg)

![](images/bcd3e3f356a1530588760282cfab08d5b6f69cd234e77b16c1c2f7a48bcc5778.jpg)  
Fig. 6: Ablation study on the training objectives. Dashed curves correspond to direct compression without pre-editing, solid circle curves to full SCOPE, downward-triangle curves to w/o task, and pentagon curves to w/o bg/rate. The task-aware loss is essential for preserving detection accuracy, while the background and rate group improves the compression-aware rate-accuracy trade-off.

## 4.4 Ablation Study

Training objectives. Fig. 6 compares full SCOPE with variants removing the taskaware loss $\mathcal { L } _ { \mathrm { t a s k } } \ ( w / o \ t a s k )$ or the background simplification and rate group $\mathcal { L } _ { \mathrm { b g } } \left( w / o \right.$ bg/rate). Removing $\mathcal { L } _ { \mathrm { t a s k } }$ has the strongest negative effect. Although the resulting curves can move toward lower bitrate, this comes with a substantial loss of detection accuracy and the variant often falls below direct compression. Bitrate reduction alone is therefore insufficient: detector-guided supervision is needed to prevent the pre-editor from discarding information that matters to the task.

The w/o bg/rate variant remains substantially closer to full SCOPE, indicating that task supervision still protects object-relevant evidence. Its rate-accuracy behavior is nevertheless less stable across codecs and detectors. The two groups therefore play complementary roles: $\mathcal { L } _ { \mathrm { t a s k } }$ protects task evidence, while background simplification and rate regularization encourage non-object regions to consume fewer unnecessary bits.

Codec-family and severity conditioning. We compare full SCOPE with four alternatives (Table 3): Family removes severity conditioning, Severity removes the family condition, Per-codec trains a separate unconditioned specialist for each codec, and Percodec+Sev. adds severity conditioning to each codec-specific specialist. Base reports BDBR relative to direct compression, while SCOPE reports a direct pairwise comparison with the full model. These latter values are recomputed over the valid common AP interval of each pair and are not obtained by subtracting two baseline-relative BDBRs.

Table 3: Conditioning ablation, BDBR (%). Base: relative to direct compression; SCOPE: pairwise relative to the full model. Lower is better. Bold/underline indicate the best/second-best Base means. <sup>†</sup>: available comparisons only.
<table><tr><td></td><td></td><td>SCOPE</td><td colspan="2">Family</td><td colspan="2">Severity</td><td colspan="2">Per-codec</td><td colspan="2">Per-codec+Sev.</td></tr><tr><td>Codec</td><td>Detector</td><td>Base</td><td>BaseSCOPE</td><td></td><td>BaseSCOPE</td><td></td><td>BaseSCOPE</td><td></td><td>Base</td><td>SCOPE</td></tr><tr><td>JPEG</td><td>Faster R-CNN</td><td>-15.88</td><td>-14.90</td><td>+0.96</td><td>-2.94 +14.95</td><td></td><td>+4.61 +22.34</td><td></td><td>+4.90</td><td>+22.09</td></tr><tr><td>JPEG</td><td>RT-DETR-1</td><td>-5.71</td><td>-6.58</td><td>-0.70</td><td>-1.21 +3.79</td><td></td><td></td><td>+71.19</td><td></td><td>+73.72</td></tr><tr><td>JPEG</td><td>YOLO26x</td><td>-14.92</td><td>-13.84</td><td>+1.11</td><td>-2.51 +14.30</td><td></td><td>+5.64 +19.93</td><td></td><td>+5.05</td><td>+18.98</td></tr><tr><td>JPEG</td><td>Mean</td><td>-12.17</td><td>-11.77</td><td>+0.46</td><td>-2.22 +11.01</td><td></td><td>+5.13†</td><td>+37.82</td><td>+4.98†</td><td>+38.26</td></tr><tr><td>BPG</td><td>Faster R-CNN</td><td>-21.74</td><td>-20.27</td><td>+1.85</td><td>-21.07</td><td>+0.99</td><td>-21.54</td><td>+0.20</td><td>-20.42</td><td>+1.86</td></tr><tr><td>BPG</td><td>RT-DETR-1</td><td>-19.51</td><td>-17.87</td><td>+2.04</td><td>-17.69</td><td>+2.37</td><td>-19.98</td><td>-0.60</td><td>-18.35</td><td>+1.41</td></tr><tr><td>BPG</td><td>YOLO26x</td><td>-20.88</td><td>-20.40</td><td>+0.47</td><td>-20.01</td><td>+0.95</td><td>-21.22</td><td>-0.60</td><td>-18.91</td><td>+2.42</td></tr><tr><td>BPG</td><td>Mean</td><td>-20.71</td><td>-19.51</td><td>+1.45</td><td>-19.59</td><td>+1.44</td><td>-20.91</td><td>-0.33</td><td>-19.23</td><td>+1.90</td></tr><tr><td>BMSHJ</td><td>Faster R-CNN</td><td>-19.21</td><td>-17.28</td><td>+1.79</td><td>-21.25</td><td>-2.80</td><td>-20.40</td><td>-2.16</td><td>-24.65</td><td>-6.46</td></tr><tr><td>BMSHJ</td><td>RT-DETR-1</td><td>-13.08</td><td>-12.58</td><td>+0.88</td><td>-16.19</td><td>-3.59</td><td>-16.27</td><td>-3.99</td><td>-16.64</td><td>-3.97</td></tr><tr><td>BMSHJ</td><td>YOLO26x</td><td>-12.91</td><td>-15.09</td><td>-3.18</td><td>-16.62</td><td>-4.56</td><td>-17.50</td><td>-6.01</td><td>-18.36</td><td>-5.92</td></tr><tr><td>BMSHJ</td><td>Mean</td><td>-15.07</td><td>-14.98</td><td>-0.17</td><td>-18.02</td><td>-3.65</td><td>-18.06</td><td>-4.05</td><td>-19.88</td><td>-5.45</td></tr><tr><td>Overall</td><td></td><td>-15.98</td><td>-15.42</td><td>+0.58</td><td>-13.28</td><td>+2.93</td><td>-13.33†</td><td>+11.14</td><td>-13.42†</td><td>+11.57</td></tr></table>

Full SCOPE gives the best overall result at −15.98% BDBR. Family-only is closest at −15.42%, whereas severity-only reaches −13.28%, indicating that codec family is the stronger global conditioning signal. Adding severity nevertheless yields the best aggregate result, although its effect is not uniform across codecs. Codec-specific specialists can be stronger locally. Per-codec+Sev. reaches −19.88% on BMSHJbut are substantially less consistent and reach +4.98% on JPEG over the available comparisons. The shared family-and-severity model therefore provides the strongest overall trade-off while retaining one set of weights.

## 5 Conclusion

We introduced SCOPE, a conditional image pre-editor for object detection under image compression. The model combines codec-family and severity conditioning with attention-based object/background editing, allowing one set of weights to adapt its transformation across multiple compression regimes. Training uses differentiable codec proxies and a frozen detector, while inference requires only the image and codec condition and leaves both the actual codec and downstream detector unchanged.

Across JPEG, BPG, and BMSHJ and the three evaluated detector architectures, SCOPE achieves an average BDBR of −15.98% relative to direct compression. The conditioning ablation shows that family information provides the strongest individual conditioning signal, while combining family and severity gives the best aggregate result with a single model. SCOPE also transfers zero-shot to the unseen Cheng and JPEG2000 codecs, although the JPEG2000 results show that the chosen seen-family condition can affect performance on an unseen family.

These results support conditional pre-editing as a practical, standard-compatible strategy for machine-oriented compression: adaptation is moved in front of the codec, preserving conventional compressed images while improving the rate-accuracy trade-off for downstream detection.

## References

1. Ahonen, J.I., Le, N., Zhang, H., Cricri, F., Rahtu, E.: Region of interest enabled learned image coding for machines. In: Proc. of IEEE International Workshop on Multimedia Signal Processing (MMSP). pp. 1–6. IEEE (2023). https://doi.org/10.1109/mmsp59012.2023.10337731

2. Ahonen, J.I., Youvalari, R.G., Le, N., Zhang, H., Cricri, F., Tavakoli, H.R., Hannuksela, M.M., Rahtu, E.: Learned enhancement filters for image coding for machines. In: Proc. of International Symposium on Multimedia (ISM). pp. 235–239. IEEE (2021). https://doi.org/10.1109/ism52913.2021.00046

3. Ballé, J., Minnen, D., Singh, S., Hwang, S.J., Johnston, N.: Variational image compression with a scale hyperprior (2018). https://doi.org/10.48550/arXiv.1802.01436, arXiv:1802.01436

4. Ballé, J., Laparra, V., Simoncelli, E.P.: End-to-end optimization of nonlinear transform codes for perceptual quality. In: Proc. of Picture Coding Symposium (PCS). pp. 1–5. IEEE (2016). https://doi.org/10.1109/pcs.2016.7906310

5. Bellard, F.: BPG Image Format. https://bellard.org/bpg/ (2014), accessed: 2026-06-29

6. Bjontegaard, G.: Calculation of average psnr differences between rd-curves. ITU SG16 Doc. VCEG-M33 (2001)

7. Chamain, L.D., Racape, F., Begaint, J., Pushparaja, A., Feltman, S.: End-to-end optimized image compression for machines, a study. In: Proc. of Data Compression Conference (DCC). pp. 163–172. IEEE (2021). https://doi.org/10.1109/dcc50243.2021.00024

8. Chen, Q., Gao, C., Liu, D.: End-to-end learned scalable multilayer feature compression for machine vision tasks. In: Proc. of IEEE International Conference on Image Processing (ICIP). pp. 1781–1787. IEEE (2024). https://doi.org/10.1109/icip51287.2024.10647798

9. Chen, Z., Fan, K., Wang, S., Duan, L.Y., Lin, W., Kot, A.: Lossy intermediate deep learning feature compression and evaluation. In: Proc. of ACM International Conference on Multimedia (ACM MM). pp. 2414–2422. ACM (2019). https://doi.org/10.1145/3343031.3350849

10. Chen, Z., Fan, K., Wang, S., Duan, L., Lin, W., Kot, A.C.: Toward intelligent sensing: Intermediate deep feature compression. IEEE Transactions on Image Processing 29, 2230–2243 (2020). https://doi.org/10.1109/tip.2019.2941660

11. Cheng, Z., Sun, H., Takeuchi, M., Katto, J.: Learned image compression with discretized gaussian mixture likelihoods and attention modules. In: Proc. of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 7936–7945. IEEE (2020). https://doi.org/10.1109/cvpr42600.2020.00796

12. Choi, H., Bajic, I.V.: High efficiency compression for object detection. In: 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1792–1796. IEEE (2018)

13. Choi, H., Bajic, I.V.: Scalable video coding for humans and machines. In: Proc. of IEEE International Workshop on Multimedia Signal Processing (MMSP). pp. 1–6. IEEE (2022). https://doi.org/10.1109/mmsp55362.2022.9949576

14. Choi, J., Han, B.: Task-aware quantization network for jpeg image compression. In: Lecture Notes in Computer Science, pp. 309–324. Springer International Publishing (2020). https://doi.org/10.1007/978-3-030-58565-5\_19

15. Dao, L.T.H., Gia Vien, A., Lee, J., Jeong, S., Yang, N., Lee, C.: Gradient-guided diffusion-based restoration of extremely compressed backgrounds for video coding for

machines. IEEE Transactions on Circuits and Systems for Video Technology 36(1), 685–701 (2026). https://doi.org/10.1109/tcsvt.2025.3598710

16. Duan, L., Liu, J., Yang, W., Huang, T., Gao, W.: Video coding for machines: A paradigm of collaborative compression and intelligent analytics. IEEE Transactions on Image Processing 29, 8680–8695 (2020). https://doi.org/10.1109/tip.2020.3016485

17. Eimon, M.E.H., Krause, A., Perera, A., Merlos, J., Kalva, H., Adzic, V., Furht, B.: Roi-packing: Efficient region-based compression for machine vision. In: Proc. of IEEE International Conference on Multimedia Information Processing and Retrieval (MIPR). pp. 233–238. IEEE (2025). https://doi.org/10.1109/mipr67560.2025.00044

18. Feng, R., Jin, X., Guo, Z., Feng, R., Gao, Y., He, T., Zhang, Z., Sun, S., Chen, Z.: Image coding for machines with omnipotent feature learning. In: Lecture Notes in Computer Science, pp. 510–528. Springer Nature Switzerland (2022). https://doi.org/10.1007/978-3-031-19836-6\_29

19. Fischer, K., Brand, F., Blum, C., Kaup, A.: Saliency-driven hierarchical learned image coding for machines. In: Proc. of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1–5. IEEE (2023). https://doi.org/10.1109/icassp49357.2023.10096674

20. Galteri, L., Bertini, M., Seidenari, L., Uricchio, T., Del Bimbo, A.: Increasing video perceptual quality with GANs and semantic coding. In: Proc. of ACM International Conference on Multimedia (ACM MM). pp. 862–870. MM ’20, Association for Computing Machinery, New York, NY, USA (2020). https://doi.org/10.1145/3394171.3413508, https://doi.org/10.1145/3394171.3413508

21. Gao, W., Xu, X., Qin, M., Liu, S.: An open dataset for video coding for machines standardization. In: Proc. of IEEE International Conference on Image Processing (ICIP). pp. 4008–4012. IEEE (2022). https://doi.org/10.1109/icip46576.2022.9897525

22. Harell, A., De Andrade, A., Bajic, I.V.: Rate-distortion in image coding for machines. In: Proc. of Picture Coding Symposium (PCS). pp. 199–203. IEEE (2022). https://doi.org/10.1109/pcs56426.2022.10018035

23. Harell, A., Foroutan, Y., Ahuja, N., Datta, P., Kanzariya, B., Somayazulu, V.S., Tickoo, O., de Andrade, A., Bajić, I.V.: Rate-distortion theory in coding for machines and its applications. IEEE Transactions on Pattern Analysis and Machine Intelligence 47(7), 5501–5519 (2025). https://doi.org/10.1109/tpami.2025.3548516

24. Iino, K., Akamatsu, S., Watanabe, H., Enomoto, S., Sakamoto, A., Eda, T.: Improving image coding for machines through optimizing encoder via auxiliary loss. In: Proc. of IEEE International Conference on Image Processing (ICIP). pp. 3771–3777. IEEE (2024). https://doi.org/10.1109/icip51287.2024.10647577

25. Jin, D., Lei, J., Peng, B., Pan, Z., Zhao, B., Ling, N., Huang, Q.: Stereo image coding for machines with joint visual feature compression. IEEE Transactions on Artificial Intelligence pp. 1–13 (2026). https://doi.org/10.1109/tai.2026.3671761

26. Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F.C., Kalfaoglu, M.E.: Ultralytics YOLO26: Unified real-time end-to-end vision models (2026). https://doi.org/10.48550/arXiv.2606.03748, arXiv:2606.03748

27. Le, N., Zhang, H., Cricri, F., Ghaznavi-Youvalari, R., Rahtu, E.: Image coding for machines: an end-to-end learned approach. In: Proc. of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1590–1594. IEEE (2021). https://doi.org/10.1109/icassp39728.2021.9414465

28. Le, N., Zhang, H., Cricri, F., Ghaznavi-Youvalari, R., Tavakoli, H.R., Rahtu, E.: Learned image coding for machines: A content-adaptive approach. In: Proc. of IEEE International Conference on Multimedia and Expo (ICME). pp. 1–6. IEEE (2021). https://doi.org/10.1109/icme51207.2021.9428224

29. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common objects in context. In: Proc. of European Conference on Computer Vision (ECCV). pp. 740–755. Springer (2014)

30. Liu, D., Zhang, H., Xiong, Z.: On the classification-distortion-perception tradeoff. In: Proc. of Advances in Neural Information Processing Systems (NeurIPS). vol. 32 (2019)

31. Liu, J., Wei, Y., Lin, J., Zhao, S., Sun, H., Chen, Z., Zeng, W., Jin, X.: Tell codec what worth compressing: Semantically disentangled image coding for machine with lmms. In: Proc. of IEEE International Conference on Visual Communications and Image Processing (VCIP). pp. 1–5. IEEE (2024). https://doi.org/10.1109/vcip63160.2024.10849922

32. Liu, J., Zhang, Y., Guo, Z., Huang, X., Jiang, G.: Multiscale feature importance-based bit allocation for end-to-end feature coding for machines. ACM Transactions on Multimedia Computing, Communications, and Applications 21(9), 1–19 (2025). https://doi.org/10.1145/3748654

33. Lu, G., Ge, X., Zhong, T., Hu, Q., Geng, J.: Preprocessing enhanced image compression for machine vision. IEEE Transactions on Circuits and Systems for Video Technology 34(12), 13556–13568 (2024). https://doi.org/10.1109/tcsvt.2024.3441049

34. Marie, A., Desnos, K., Morin, L., Zhang, L.: Video coding for machines: Large-scale evaluation of deep neural networks robustness to compression artifacts for semantic segmentation. In: Proc. of IEEE International Workshop on Multimedia Signal Processing (MMSP). pp. 01–06. IEEE (2022). https://doi.org/10.1109/mmsp55362.2022.9949999

35. Park, K., Lee, J., Jeong, S., Kim, K.: Design of a bitstream structure for streaming of video coding for machines. Journal of Broadcast Engineering 30(3), 320–330 (2025). https://doi.org/10.5909/jbe.2025.30.3.320

36. Partanen, T., Marie, A., Kortelahti, R., Mercat, A., Vanne, J., Hannuksela, M.M., Zhang, H., Aminlou, A., Cricri, F.: Task enhancement tiles for ultra lightweight post-processing in visual coding for machines. In: Proc. of Picture Coding Symposium (PCS). pp. 1–5. IEEE (2025). https://doi.org/10.1109/pcs65673.2025.11417579

37. Peng, B., Lin, T., Jin, D., Pan, Z., Lei, J.: Saliency map-guided end-to-end image coding for machines. IEEE Signal Processing Letters 31, 1755–1759 (2024). https://doi.org/10.1109/lsp.2024.3420178

38. Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A.: FiLM: Visual reasoning with a general conditioning layer. In: Proc. of AAAI Conference on Artificial Intelligence. vol. 32 (2018)

39. Pham-Dinh, K., Pakdaman, F., Zhang, H., Tavakoli, H.R., Le, N., Ahonen, J., Gabbouj, M.: Evaluating the emerging mpeg video coding for machines in semantic segmentation. In: Proc. of International Symposium on Multimedia (ISM). pp. 141–148. IEEE (2025). https://doi.org/10.1109/ism66958.2025.00038

40. Reich, C., Debnath, B., Patel, D., Chakradhar, S.: Differentiable JPEG: The devil is in the details. In: Proc. of IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 4126–4135 (2024)

41. Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object detection with region proposal networks. In: Proc. of Advances in Neural Information Processing Systems. vol. 28 (2015)

42. Różek, S., Stankiewicz, O., Maćkowiak, S., Domański, M.: Video coding for machines using object analysis and standard video codecs. In: Proc. of IEEE International Conference on Visual Communications and Image Processing (VCIP). pp. 1–5. IEEE (2023). https://doi.org/10.1109/vcip59821.2023.10402642

43. Singh, S., Abu-El-Haija, S., Johnston, N., Ballé, J., Shrivastava, A., Toderici, G.: End-to-end learning of compressible features. In: Proc. of IEEE Interna-

tional Conference on Image Processing (ICIP). pp. 3349–3353. IEEE (2020). https://doi.org/10.1109/icip40778.2020.9190860

44. Skodras, A., Christopoulos, C., Ebrahimi, T.: The JPEG 2000 still image compression standard. IEEE Signal Processing Magazine 18(5), 36–58 (2002)

45. Sullivan, G.J., Ohm, J.R., Han, W.J., Wiegand, T.: Overview of the high efficiency video coding (HEVC) standard. IEEE Transactions on Circuits and Systems for Video Technology 22(12), 1649–1668 (2012)

46. Sun, Y., Zhao, Y., Liu, M., Yao, C., Bai, H., Lin, C., Lin, W.: Transvfc: A transformable video feature compression framework for machines. Pattern Recognition 171, 112091 (2026). https://doi.org/10.1016/j.patcog.2025.112091

47. Suzuki, S., Takagi, M., Hayase, K., Onishi, T., Shimizu, A.: Image pre-transformation for recognition-aware image compression. In: 2019 IEEE International Conference on Image Processing (ICIP). pp. 2686–2690. IEEE (2019)

48. Talebi, H., Kelly, D., Luo, X., Dorado, I.G., Yang, F., Milanfar, P., Elad, M.: Better compression with deep pre-editing. IEEE Transactions on Image Processing 30, 6673–6685 (2021). https://doi.org/10.1109/tip.2021.3096085

49. Wallace, G.K.: The JPEG still picture compression standard. Communications of the ACM 34(4), 30–44 (1991)

50. Wang, Y., Liang, F., Sun, H., Katto, J.: CGICM: CLIP-guided semantic frequency adaptation in image compression for machines. In: Proc. of International Conference on Visual Communications and Image Processing (VCIP). pp. 1–5. IEEE (2025). https://doi.org/10.1109/vcip67698.2025.11396854

51. Wang, Y., Yang, J., Wang, S., Yang, X.: Regularized deep convolutional encodingdecoding generative network for remote sensing cloud image compression. In: Proc. of Data Compression Conference (DCC). pp. 405–405. IEEE (2025). https://doi.org/10.1109/dcc62719.2025.00092

52. Wu, H.: Generative adversarial network based deep learning method for machine vision inspection. In: Applications of Generative AI, pp. 209–225. Springer International Publishing (2024). https://doi.org/10.1007/978-3-031-46238-2\_10

53. Wu, Y., Kirillov, A., Massa, F., Lo, W.Y., Girshick, R.: Detectron2. https://github.com/facebookresearch/detectron2 (2019)

54. Wu, Z., Wang, H., Wang, H., Zhang, Y.: End to end scalable image coding for machines. In: Proc. of International Conference on Intelligent Communications and Computing (ICC). pp. 217–221. IEEE (2023). https://doi.org/10.1109/icc59986.2023.10421674

55. Xie, S., Girshick, R., Dollár, P., Tu, Z., He, K.: Aggregated residual transformations for deep neural networks. In: Proc. of IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1492–1500 (2017)

56. Yang, M., Herranz, L., Yang, F., Murn, L., Blanch, M.G., Wan, S., Yang, F., Mrak, M.: Semantic preprocessor for image compression for machines. In: Proc. of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1–5. IEEE (2023). https://doi.org/10.1109/icassp49357.2023.10096472

57. Yang, M., Yang, F., Murn, L., Blanch, M.G., Sock, J., Wan, S., Yang, F., Herranz, L.: Task-switchable pre-processor for image compression for multiple machine vision tasks. IEEE Transactions on Circuits and Systems for Video Technology 34(7), 6416–6429 (2024). https://doi.org/10.1109/tcsvt.2023.3348995

58. Yang, W., Huang, H., Hu, Y., Duan, L.Y., Liu, J.: Video coding for machines: Compact visual representation compression for intelligent collaborative analytics. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(7), 5174–5191 (2024). https://doi.org/10.1109/tpami.2024.3367293

59. Yoon, Y.U., Kim, D., Lee, J., Oh, B.T., Kim, J.G.: An efficient multi-scale feature compression with qp-adaptive feature channel truncation for video coding for machines. IEEE Access 11, 92443–92458 (2023). https://doi.org/10.1109/access.2023.3307404

60. Zhang, Z., Liu, Y.: Blur clip context model-based image coding for machines framework with down and up sample pairs for multitasks. IEEE Access 13, 45951–45963 (2025). https://doi.org/10.1109/access.2025.3550335

61. Zhang, Z., Yu, L., Zhang, Q., Mei, J., Guan, T.: Towards efficient learned image coding for machines via saliency-driven rate allocation. In: Proc. of IEEE International Conference on Visual Communications and Image Processing (VCIP). pp. 1–5. IEEE (2023). https://doi.org/10.1109/vcip59821.2023.10402607

62. Zhao, F., Guo, M., Zhao, S., Li, J., Zhang, L., Xie, X.: A preprocessing framework for video machine vision under compression. In: Proc. of Data Compression Conference (DCC). pp. 312–321. IEEE (2024). https://doi.org/10.1109/dcc58796.2024.00039

63. Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., Chen, J.: DETRs beat YOLOs on real-time object detection. In: Proc. of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16965–16974. IEEE (2024)

## Supplementary Material

## A Runtime Analysis

We measure the runtime overhead introduced by SCOPE independently from codec execution and detector evaluation. The benchmark is performed on the COCO 2017 [29] Validation set using the same checkpoint used in the main experiments. For each codec-family condition, namely Diff\_JPEG, BPG\_like, and BMSHJ, we evaluate five severity values $s \in \{ 0 . 0 0 , 0 . 2 5 , 0 . 5 0 , 0 . 7 5 , 1 . 0 0 \}$ . Each family-severity pair is measured on 5,000 validation images, resulting in 75,000 total pre-editing measurements. We use 20 warm-up images before timing and one timed pass per image.

The reported runtime isolates the pre-editing module. Image reading is measured separately and excluded from the reported pre-editing time, while codec execution and detector evaluation are not performed. Output saving is disabled during the benchmark, so the total pre-editing time includes tensor preparation, transfer to the GPU, construction of the conditioning inputs, and the forward pass of the pre-editor. The pre-editor is evaluated on CUDA with batch size one.

Table 4 summarizes the runtime averaged on all severity levels for each codec-family condition. The forward pass takes about 7.93 ms per image on average, corresponding to approximately 126 images per second. Including tensor preparation and conditioning setup, the total pre-editing time is about 11.81 ms per image, corresponding to approximately 85 images per second. The differences across codec-family conditions are very small, which is expected because SCOPE uses the same network architecture for all conditions; the codec family and severity only modify the conditioning vector.

Table 4: Runtime of SCOPE grouped by codec-family condition. Values are averaged over five severity levels and validation images per level. Forward time includes only the neural pre-editor forward pass, while total time additionally includes tensor preparation and conditioning setup. Codec execution, detector evaluation, and image reading are excluded.
<table><tr><td colspan="6">Condition family Images Forward mean Forward p95 Total mean Total p95 Total FPS</td></tr><tr><td>BMSHJ</td><td>25,000</td><td>7.92 ms</td><td>10.69 ms</td><td>11.78 ms 15.13 ms</td><td>84.89</td></tr><tr><td>BPG_like</td><td>25,000</td><td>7.92 ms</td><td>10.67 ms</td><td>11.79 ms 15.11 ms</td><td>84.85</td></tr><tr><td>Diff_JPEG</td><td>25,000</td><td>7.96 ms</td><td>10.85 ms</td><td>11.85 ms 15.17 ms</td><td>84.37</td></tr><tr><td>Overall</td><td>75,000</td><td>7.93 ms</td><td>10.74 ms</td><td>11.81 ms 15.14 ms</td><td>84.70</td></tr></table>

We also analyze whether the conditioning severity affects runtime. As shown in Table 5, the total runtime remains nearly constant across severity values. The small variation is within normal measurement noise and does not indicate a systematic dependence on the severity parameter. This confirms that changing the compression severity only changes the scalar conditioning input and does not alter the computational graph of the pre-editor.

Overall, the runtime analysis shows that SCOPE introduces a small and stable preprocessing overhead. The cost is effectively independent of the selected codec-family condition and severity level because all operating points are handled by the same conditional network. This supports the practical use of SCOPE as a codec-conditioned pre-editing module: the same model can be applied across different compression settings by changing only the family and severity conditions, without modifying the architecture, loading codec-specific weights, or increasing the inference cost.

Table 5: Runtime of SCOPE grouped by severity. Values are averaged over the three codec-family conditions. Runtime is nearly constant across severity values because severity only affects the conditioning vector.
<table><tr><td colspan="4">Severity Images Forward mean Total mean Total FPS</td></tr><tr><td>0.00</td><td>15,000</td><td>8.00 ms</td><td>11.87 ms</td></tr><tr><td>0.25</td><td>15,000</td><td>7.92 ms</td><td>84.28 11.78 ms 84.89</td></tr><tr><td>0.50</td><td>15,000</td><td>7.91 ms 11.79 ms</td><td>84.78</td></tr><tr><td>0.75</td><td>15,000</td><td>7.92 ms 11.79 ms</td><td>84.82</td></tr><tr><td>1.00</td><td>15,000</td><td>7.92 ms 11.80 ms</td><td>84.73</td></tr></table>

## B Semantic Decomposition of mAP Gains

We further analyze which semantic groups contribute to the detection gains obtained by SCOPE. The goal of this analysis is to decompose the same COCO mAP difference used in the main rate and accuracy curves, rather than to report independent AP values for each supercategory.

For this reason, we do not compute the difference between AP values evaluated directly on each supercategory. Instead, we first compute AP for each individual COCO category and then aggregate the category-level differences according to the official COCO supercategories. Let C be the complete set of COCO categories and let $\mathcal { C } _ { s } \subset \mathcal { C }$ be the set of categories belonging to the supercategory s. For a fixed codec setting and detector, the overall mAP difference is

$$
\varDelta \mathrm { m A P } = \frac { 1 } { | \mathcal { C } | } \sum _ { c \in \mathcal { C } } \Bigl ( \mathrm { A P } _ { c } ^ { \mathrm { S C O P E } } - \mathrm { A P } _ { c } ^ { \mathrm { b a s e } } \Bigr ) .\tag{20}
$$

The contribution of the supercategory s is then defined as

$$
\varDelta \mathrm { m A P } _ { s } = \frac { 1 } { \left| \mathscr { C } \right| } \sum _ { c \in \mathscr { C } _ { s } } \left( \mathrm { A P } _ { c } ^ { \mathrm { S C O P E } } - \mathrm { A P } _ { c } ^ { \mathrm { b a s e } } \right) .\tag{21}
$$

The denominator is therefore the total number of COCO categories, not the number of categories inside the supercategory. Consequently, the contributions recover the overall mAP difference:

$$
\varDelta \mathrm { m A P }  { = } \sum _ { s } \varDelta \mathrm { m A P } _ { s } .\tag{22}
$$

This decomposition should therefore be interpreted as a breakdown of the global COCO mAP variation. A positive bar indicates that the corresponding supercategory contributes positively to the overall gain, while a negative bar indicates that SCOPE decreases the category-averaged AP for that semantic group at that operating point.

Individual negative bars are possible even when the total mAP improves, as long as the sum of all contributions remains positive.

Table 6 reports the category grouping used in the decomposition. We use the official COCO supercategories. For single-category groups, we report only the group name.

Table 6: COCO category grouping used for the semantic decomposition.
<table><tr><td>Group</td><td>COCO categories</td></tr><tr><td>person</td><td>person</td></tr><tr><td>vehicle</td><td>bicycle, car, motorcycle, airplane, bus, train, truck, boat</td></tr><tr><td>outdoor</td><td>traffic light, fire hydrant, stop sign, parking meter, bench</td></tr><tr><td>animal</td><td>bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe</td></tr><tr><td>accessory</td><td>backpack, umbrella, handbag, tie, suitcase</td></tr><tr><td>sports</td><td>frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket</td></tr><tr><td>kitchen</td><td>bottle, wine glass, cup, fork, knife, spoon, bowl</td></tr><tr><td>food</td><td>banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake</td></tr><tr><td>furniture</td><td>chair, couch, potted plant, bed, dining table, toilet</td></tr><tr><td>electronic</td><td>tv, laptop, mouse, remote, keyboard, cell phone</td></tr><tr><td>appliance</td><td>microwave, oven, toaster, sink, refrigerator</td></tr><tr><td>indoor</td><td>book, clock, vase, scissors, teddy bear, hair drier, toothbrush</td></tr></table>

We report this analysis only for BMSHJ and BPG. These codecs provide more reliable paired comparisons between the compressed baseline and the SCOPE-preedited version at the same nominal compression setting. For JPEG, the resulting bitrates of the baseline and SCOPE variants can differ substantially more at the same quality value. A same-quality semantic decomposition would therefore not align as cleanly with the rate and accuracy curves, since the comparison would mix the semantic effect of pre-editing with a larger bitrate shift. For this reason, JPEG is kept in the global rate and accuracy analysis but excluded from the semantic decomposition plots.

Figures 7 and 8 show the semantic decomposition of BMSHJ and BPG. Overall, the histograms indicate that the gains are not concentrated in a single semantic group. Under stronger compression, most detectors exhibit positive contributions across several groups, especially food, animal, electronics, furniture, kitchen, and indoor categories. This suggests that SCOPE helps preserve task-relevant structures that are useful for detection across heterogeneous object types.

The effect becomes smaller as compression becomes lighter. This is expected because the compressed baseline already preserves most of the information needed by the detector at low compression. In this regime, the histograms often show nearzero contributions and a mixture of positive and negative bars. For BMSHJ at low compression, the total effect is close to neutral, with a small positive or nearly unchanged result for most detectors and a slight decrease for RT-DETR. For BPG at low compression, the same-quality decomposition is slightly negative for all detectors. Therefore, these low-compression results should be interpreted as comparable to the baseline rather than as a failure mode of the method.

This interpretation is also consistent with the main paper results in Fig. 3. At low compression, the main advantage of SCOPE is not necessarily a large same-quality mAP increase, but the ability to obtain comparable detection accuracy at a lower bitrate. Therefore, a small negative or near-zero same-quality contribution in the decomposition does not contradict the rate and accuracy curves. It indicates that, when the baseline is already strong, SCOPE mainly shifts the operating point toward lower bits per pixel while keeping the detector performance close to the baseline.

For medium and high compression, the decompositions are more clearly positive. BMSHJ shows the largest and most consistent gains at high compression, where compression artifacts are strongest and the pre-editor has more room to improve detectorrelevant content. BPG shows positive gains at medium and high compression, while becoming nearly neutral or slightly negative at low compression. Across both codecs, the recurrent positive contributions from object-rich groups such as food, animal, electronics, furniture, kitchen, and indoor suggest that SCOPE is especially helpful when compression affects localized structures and textures that are important for object recognition. Conversely, groups such as sports, outdoor, and vehicle are more variable across detectors and compression levels, indicating that the effect of pre-editing can depend on the detector architecture and on the visual statistics of each semantic group.

![](images/aa11bd4324d9bf7b4011f83005566c935bd62ced45ea83e74a5a23d5dd13b1c6.jpg)

BMSHJ — Low compression (q=5) — Standard COCO ΔAP decomposed by supercategory  
![](images/ab3e5c0531fd09fe9ca511268da6d016c68d45e360304be1933acc67a88c44c0.jpg)  
(a) BMSHJ, low compression.

![](images/630facc6393b29e7b277d30e37eabae646562384fbe32ec19f11b725545f2df5.jpg)

BMSHJ — Middle compression (q=3) — Standard COCO ΔAP decomposed by supercategory  
![](images/41574e635425f481cefbadf844695a92713357a7c0f1b1f5e2a95439fad835d4.jpg)

![](images/d39ed2eb5320dc70737256fee726c50b17ce790aa89d56ef1b2e660b09f681f4.jpg)  
(b) BMSHJ, middle compression.

![](images/2586f470e4c2c2d3d9a2a80fd516b5a90228eca781f759e77a87a19da5db7d3c.jpg)

BMSHJ — High compression (q=1) — Standard COCO ΔAP decomposed by supercategory  
![](images/0d5e10ec5d06b74658a0ba060ddc97209563a22628e0c5d0a1b0d19fd17162fe.jpg)

![](images/ce1de41ac39dc2658cc903149666c7de02df738188492e4b3881daae0102d74f.jpg)  
(c) BMSHJ, high compression.

![](images/3145d28a860d08f47a278a02e8f557956b8fc9da72e85ce252d48d392f506e24.jpg)

Fig. 7: Semantic decomposition of the COCO mAP difference for BMSHJ. Each bar represents the contribution of a COCO supercategory to the overall mAP change between SCOPE and the corresponding compressed baseline. Contributions are obtained from per-category AP differences and normalized by the total number of COCO categories, so the sum of the bars equals the overall mAP difference for the corresponding detector and compression setting.

BPG — Low compression (QP=28) — Standard COCO ΔAP decomposed by supercategory  
![](images/df1915188ba8c6d4ec01b1ba905fda434168e613d1986a8ca91120a360754eac.jpg)

![](images/323c94c8166cd788f7848f1d055234affa14a6f60b6d697886ac6216c98ddfb0.jpg)  
(a) BPG, low compression.

![](images/6ad813ddc7a3acbe5d5379fbd72462c9fe3d3288b648f7bd77526bed31debe45.jpg)

BPG — Middle compression (QP=37) — Standard COCO ΔAP decomposed by supercategory  
![](images/01a206ca8570869a53a0f2442ee4bd43061e88bd423d441245fd732007f147d5.jpg)

![](images/0def6535ad8f63f3cbbb4e469585ebdee82084b1dc6099e2a84461c14fcf4e86.jpg)  
(b) BPG, middle compression.

![](images/3f9218b82162e8dddf0b6f9979986f9d9b592d42f504f87a22eb7f4c9c2e1a69.jpg)

BPG — High compression (QP=44) — Standard COCO ΔAP decomposed by supercategory  
![](images/2fa3e00db00265e0dd692b7d9a72dd2ad25844043a594696bd42b69461b4ce57.jpg)

![](images/0e7f25f1f36ee7dffca0b0ae70c8f57a245da648dd7620c559ff57efc03f9255.jpg)  
(c) BPG, high compression.

![](images/7b090ca8f11f1245ab3bf9d9429f5c9d518c4a9d46979199bdf8a8b4d765c6f8.jpg)

Fig. 8: Semantic decomposition of the COCO mAP difference for BPG. As in Fig. 7, the decomposition is computed from per-category AP differences and then grouped by COCO supercategory using the full COCO category set as denominator. This makes the sum of the displayed contributions directly comparable to the overall mAP difference shown in the rate and accuracy curves.