# VesperMUSE: Multi-Modal Fusion Vision Transformer for Image Semantic Segmentation Under Diverse Light Conditions

MANCHESTER 1824

# A dissertation submitted to The University of Manchester for the degree of Bachelor of Science in Computer Science in the Faculty of Science and Engineering

Year of submission 2026

Author Raul Balmez

Supervisor

Tingting Mu

Department of Computer Science

## Contents

Contents 2   
List of figures 4   
List of tables 6   
Key terms and abbreviations . 7   
Declaration of originality 8   
Copyright statement 9   
Acknowledgments 10   
Abstract 11   
1 Introduction 12   
2 Related Work 18   
2.1 Synthetic Low-Light Image Adaptation 19   
2.2 Low-Light Image Enhancement 19   
2.3 Modality Fusion for Image Semantic Segmentation 21   
3 Methods 23   
3.1 Framework Design and Motivation 24   
3.2 Modalities 25   
3.2.1 Synthetic low-light adaptation algorithm 25   
3.2.2 Auxiliary modalities . 30   
3.3 Low-Light Image Enhancement Vision Transformer . 31   
3.3.1 Thermal-aware Transformer Block 33   
3.3.2 RGB-Thermal Cross-Attention 33   
3.3.3 Feed Forward Network 34   
3.3.4 Low-Light Image Enhancement Loss function 34   
3.3.5 Low-Light Image Enhancement Training . 36   
3.4 Fusion Vision Transformer 37   
3.4.1 Fusion Encoder Block 40   
3.4.2 Fusion Self-Attention Module 40   
3.4.3 Feed Forward Network 43   
3.4.4 Decoder Block . 43   
3.4.5 Segmentation Head 44   
3.4.6 Fusion Vision Transformer Loss function . 45   
3.4.7 Fusion Vision Transformer Training 46   
4 Experiments and Results 48   
4.1 Synthetic Low-Light Image Adaptation 50   
4.1.1 Quantitative Analysis 52   
4.1.2 Qualitative Analysis 54   
4.2 Low-Light Image Enhancement 55   
4.2.1 Quantitative Analysis 55   
4.2.2 Qualitative Analysis 56   
4.3 Semantic Segmentation . 57   
4.3.1 Quantitative Analysis 58   
4.3.2 Qualitative Analysis 59   
4.4 Computational Performance 62   
4.5 Ablation Studies 63   
4.5.1 The effect of LLIE on Semantic Segmentation performance 63   
4.5.2 Modality choices and their rank . 65   
4.5.3 Architectural motivation 67   
5 Conclusions 69   
5.1 Known Limitations 70   
5.2 Future Work . 73   
5.3 Final Thoughts 73   
References 74

Word count: 13696

## List of figures

1.1 Mean Intersection over Union (mIoU) benchmarking results for semantic segmentation   
under three input configurations, well-lit, low-light, and enhanced low-light maps, on   
the PST900 [8] (left) and CART [25] (right) datasets, highlighting the impact of illumi  
nation conditions and enhancement strategies on semantic segmentation performance. 15   
3.1 An illustration of the real-world light distribution which is followed in the Real-world   
Distribution Matching stage 28   
3.2 Visual representation of the VesperMUSE LLIE module. 32   
3.3 Visual representation of the VesperMUSE Semantic Segmentation module. 37   
3.4 Visual representation of the VesperMUSE modality-agnostic attention mechanism. 41   
4.1 Illumination vs. signal integrity. Real low-light data retain high SNR, while prior syn  
thetic methods introduce excessive noise and over-brightening. VesperMUSE preserves   
an accurate signal quality similar to that observable in true low-light conditions. . 53   
4.2 PST900 [8] synthetic low-light adapted RGB and the original image. 54   
4.3 CART [25] synthetic low-light adapted RGB and the original image. 54   
4.4 Qualitative comparison of VesperMUSE (VM) LLIE with SOTA methods enhancing   
the synthetically low-light adapted PST900 LL and CART LL datasets. 57   
4.5 Illustration of the per-class surface distribution on the PST900 [8] (left) and CART [25]   
(right) datasets. The percentage for each class is computed as the mean percentage of   
pixels covered across the samples where it appears (sample count is encoded as n). . 60   
4.6 Qualitative comparison of VesperMUSE (VM) and the LLIE configuration (VM LLIE)   
with SOTA models on the PST900 dataset [8]. 61   
4.7 Qualitative comparison of VesperMUSE (VM) and the LLIE configuration (VM LLIE)   
with SOTA models on the CART dataset. 62   
4.8 Semantic segmentation performance between different input configuration using: RGB   
(VM), LLIE RGB (VM LLIE), and low-light RGB without any enhancement (VM LL)   
on the PST900 dataset [8]. 64   
4.9 Semantic segmentation performance deltas between VesperMUSE LLIE and Vesper-  
MUSE trained and tested on low-light images, highlighting the role of LLIE enhance  
ment on the PST900 dataset [8]. 64   
4.10 Semantic segmentation performance between different input configuration using: RGB   
(VM), LLIE RGB (VM LLIE), and low-light RGB without any enhancement (VM LL)   
on the CART dataset [25]. 65   
4.11 Semantic segmentation performance deltas between VesperMUSE LLIE and Vesper-  
MUSE trained and tested on low-light images, highlighting the role of LLIE enhance  
ment on the CART dataset [25]. 65   
5.1 Qualitative maps highlighting the noise and excessive degradation observed in few   
synthetically low-light adapted images. 70   
5.2 Histogram comparison between the two pairs of RGB and LLIE RGB maps showing   
the colour balance. 72   
5.3 Visual representation of the ratio between mIoU and the parameter count (M) of SOTA   
models. 72

## List of tables

4.1 Quantitative comparison between real and low-light datasets across multiple statistical   
and physical metrics. The PST900 [8]/CART [25] synthetic table entries are paired   
images comprised from original RGB images from the two datasets and synthetically   
adapted low-light images. 52   
4.2 Quantitative comparison of LLIE methods on the synthetically low-light adapted   
PST900 LL and CART LL datasets. Best results are highlighted in bold. 55   
4.3 Performance comparison with SOTA methods on the PST900 [8] dataset. Best results   
are highlighted in bold. 58   
4.4 Performance comparison with SOTA methods on the CART dataset [25]. Best results   
are highlighted in bold. 59   
4.5 Complexity comparison with an image resolution of 480 × 640 pixels. mIoU results are   
reported on the CART dataset [25]. 63   
4.6 Table analyzing the impact of different primary and auxiliary input modalities on the   
PST900 [8] and CART [25] datasets. Grouping by the primary backbone highlights the   
performance delta when injecting various auxiliary streams. 67   
4.7 Ablation study analyzing the impact of different fusion and decoder strategies on the   
PST900 [8] and CART [25] datasets, including parameter counts and computational   
complexity. 68

# Key terms and abbreviations

BN Batch Normalization

CDF Cumulative Distribution Function

CNN Convolutional Neural Network

CV Computer Vision

DL Deep Learning

FFN Feed-Forward Network

FLOPs Floating-Point Operations

GELU Gaussian Error Linear Unit

IoU Intersection over Union

LLIE Low-Light Image Enhancement

LN Layer Normalization

MAE Mean Absolute Error

mIoU Mean Intersection over Union

MS-SSIM Multi-Scale Structure Similarity Index Measure

NMI Normalized Mutual Information

ReLU Rectified Linear Unit

SNR Signal-to-Noise Ratio

PSNR Peak-Signal-to-Noise Ratio

SSIM Structural Similarity Index Measure

SOTA State of the Art

SSR Single-Scale Retinex

UAV Unmanned Aerial Vehicle

ViT Vision Transformer

## Declaration of originality

I hereby confirm that no portion of the work referred to in the thesis has been submitted in support of an application for another degree or qualification of this or any other university or other institute of learning.

## Copyright statement

i The author of this thesis (including any appendices and/or schedules to this thesis) owns certain copyright or related rights in it (the “Copyright”) and s/he has given The University of Manchester certain rights to use such Copyright, including for administrative purposes.

ii Copies of this thesis, either in full or in extracts and whether in hard or electronic copy, may be made only in accordance with the Copyright, Designs and Patents Act 1988 (as amended) and regulations issued under it or, where appropriate, in accordance with licensing agreements which the University has from time to time. This page must form part of any such copies made.

iii The ownership of certain Copyright, patents, designs, trademarks and other intellectual property (the “Intellectual Property”) and any reproductions of copyright works in the thesis, for example graphs and tables (“Reproductions”), which may be described in this thesis, may not be owned by the author and may be owned by third parties. Such Intellectual Property and Reproductions cannot and must not be made available for use without the prior written permission of the owner(s) of the relevant Intellectual Property and/or Reproductions.

iv Further information on the conditions under which disclosure, publication and commercialisation of this thesis, the Copyright and any Intellectual Property and/or Reproductions described in it may take place is available in the University IP Policy (see http://documents.manchester. ac.uk/DocuInfo.aspx?DocID=24420), in any relevant Thesis restriction declarations deposited in the University Library, The University Library’s regulations (see http://www. library.manchester.ac.uk/about/regulations/) and in The University’s policy on Presentation of Theses.

## Acknowledgments

I would like to thank my supervisor, Dr. Tingting Mu, for all the continued support.

## Abstract

Semantic segmentation is critical for the reliable navigation and scene understanding of autonomous systems, yet its performance degrades severely in low-light environments due to noise, artifacts, colour distortion, and the loss of structural details. To address this challenge, this work introduces VesperMUSE, a unified multi-modal framework designed to achieve robust image semantic segmentation across diverse illumination conditions. The architecture addresses the light deficiency problem on multiple levels: proposing a novel physics-informed synthetic low-light adaptation algorithm, approaching Low-Light Image Enhancement (LLIE) through a thermal-guided paradigm, and finally aiming to recover the intrinsic scene details for semantic segmentation through a multi-modal fusion approach. Following enhancement, the framework dynamically integrates the primary RGB features with auxiliary thermal representations and surface normal maps highlighting geometric discontinuities and material boundaries. At the core of this integration is a highly scalable, modality-agnostic self-attention mechanism that extracts both global context and fine local features from an arbitrary number of auxiliary inputs. By combining and projecting these hierarchical multi-modal representations into a unified component, the attention module allows the network to selectively attend to relevant cross-modal cues without requiring any architectural modifications when different sensor configurations are used. Extensive evaluations demonstrate that VesperMUSE successfully recovers missing visual cues, providing a flexible, state-of-the-art (SOTA) solution for machine perception in extreme and unstructured real-world environments.

##

Introduction

Semantic Segmentation. The performance and goal-achievement capability of autonomous or Unmanned Aerial Vehicle (UAV) systems, which have become a preferred choice for navigating dangerous environments in recent years, rely strongly on scene understanding. High-level Computer Vision (CV) tasks such as pixel-level semantic segmentation [1, 2] or instance-level object detection [3] have emerged as powerful tools for mapping sensor observations into information and cues that guide decision-making. The main challenge remains robust perception under varying conditions [4, 5]. Semantic segmentation achieves scene understanding by assigning each pixel in an image to a specific class, with classes defined during training. In this way, the entire scene is parsed, and objects can be more easily identified by their shape or distinct attributes. Due to its highly reliable approach, the applications of semantic segmentation extend beyond outdoor scene parsing to domains such as medicine [6], indoor environments [7], and even more challenging missions such as survivor localisation [8] in hazy or poorly lit conditions.

Usually, high-level CV tasks perform best under near-ideal conditions: consistent illumination that ensures accurate colour balance, without haze, smoke, noise, or other degradations. However, realworld environments rarely provide such conditions, therefore, robust systems must be capable of handling non-ideal sensor inputs [9, 10]. In this study, the focus is on improving low-light semantic segmentation and understanding the steps required to recover lost detail, as well as how to compensate for features that cannot be restored [11].

Low-Light Conditions. Light-deficient environments occur as frequently as daylight scenes, making them not merely an edge case of CV systems but a domain that requires rigorous treatment [12]. What makes low-light images particularly challenging is that limited illumination introduces additional issues beyond the obvious reduced exposure, such as global and local structure loss, noise, artifacts, and prominent colour distortions [13, 14]. There are multiple approaches to improving semantic segmentation performance under such conditions, with the main categories being: enhancement-based methods [15] (where a network is applied prior to segmentation to enhance the images), knowledge distillation [16] (where a student network trained on low-light images is constrained to match the internal representations of a teacher network trained on well-lit data), singlenetwork approaches (where the feature space is analyzed using frequency-domain techniques [17] with a greater emphasis being placed on edge information), and multi-modal fusion approaches [18, 19], in which the limited information in low-light images is compensated by incorporating additional representations.

Thermal Imagery. An expected observation is the reliability of thermal (infrared) sensing in lowlight environments, as such representations do not depend on illumination but rather on heat irradiation, making them largely light-agnostic [20] and capable of preserving details often lost in lowlight imagery, such as clear structural boundaries and reduced artifacts [21]. However, this modality presents several limitations, including poor separation of objects or surfaces with similar temperatures [22], background interference when ambient temperatures approximate those of objects [23], and the absence of colour information, which remains an essential cue for accurate scene parsing and object identification [24]. Despite these constraints, the practical acquisition of thermal data has become increasingly accessible. The commercial miniaturization of Long-Wave Infrared microbolometers has rendered these sensors both cost-effective and easy to integrate into real-world autonomous systems. In the scope of this study, the thermal modality is technically leveraged to enhance low-light RGB data by providing a high Signal-to-Noise Ratio (SNR) in environments where visible-spectrum sensors succumb to photon scarcity. Specifically for semantic segmentation, thermal data provides an invariant heat-signature prior very helpful for class-specific identification (e.g., distinguishing biological entities from inanimate structures).

Improving Segmentation Performance Through Enhancement. VesperMUSE adopts a presegmentation thermal-guided enhancement approach, where low-light images are first enhanced using a model trained separately from the semantic segmentation network. This design provides greater modularity, allowing the framework to be extended to other degradations, like haze or blur, by simply replacing the enhancement module. Another advantage is that the training data can be decoupled, removing the need for low-light, well-lit, and semantic segmentation annotations to coexist within the same dataset. Since the enhancement network is trained independently, its predictions achieve a high degree of visual quality, as the module is optimized to approximate normal lighting conditions. In contrast, jointly trained (one-stage) networks may produce intermediate-enhanced outputs represented in latent space which is not suitable for direct human evaluation. This aspect is particularly important for the present work, as the low-light enhanced RGB map is further used as input for additional representations.

Challenge Understanding Through Data. To assess the claims made regarding the issues semantic segmentation models have in low-light conditions an analysis will be conducted in Figure 1.1 focusing on three scenarios, namely conducting segmentation on low-light data directly, LLIE imaging, and the well-lit representation respectively. For the scope of this analysis, a simple U-shaped [26] ViT [27] using different hierarchical layers of a pretrained ResNet50 [28] in the encoding stage and attention mechanisms on the decoding side is employed. The loss function is a hybrid combination of Dice and Focal loss assisted with the class weights derived from the training data. All three training processes are conducted with 40k iterations employing a cosine annealing learning rate scheduler with an initial $3 \times 1 0 ^ { - 4 }$ learning rate which drops to $1 \times 1 0 ^ { - 7 }$ by the end of training.

![](images/159aad64b798a6bd497168a8e492fe61b2c68da9c2c998e6d946168022bdd0d7.jpg)

![](images/c053f0caa66b80dab49aecebe11ea8d14a3a8ab9d383b177517abe5827073875.jpg)  
Fig. 1.1. Mean Intersection over Union (mIoU) benchmarking results for semantic segmentation under three input configurations, well-lit, low-light, and enhanced low-light maps, on the PST900 [8] (left) and CART [25] (right) datasets, highlighting the impact of illumination conditions and enhancement strategies on semantic segmentation performance.

Observations point out that the degradation of scene illumination severely bottlenecks semantic parsing. Across both the PST900 [8] and CART [25] datasets, processing plain low-light RGB inputs results in substantial performance drops, yielding mIoU scores of only 0.6614 and 0.5246, respectively. This stark decline compared to the well-lit baselines underscores the necessity of the pre-segmentation enhancement strategy. By routing the degraded inputs through the independently trained enhancement module, the network successfully recovers critical scene context, elevating the segmentation performance to 0.7010 and 0.5711 on the two datasets. While this enhancement bridges a significant gap, it does not fully restore the optimal original well-lit RGB performance (0.7855 and 0.7275).

Further improvements. In the context of VesperMUSE, to aid the semantic parsing process, LLIE is conducted on the low-light images with thermal being brought as an extra representation. The infrared imagery is later employed [9] for the segmentation training as well, since it provides relevant cues for both tasks. To make the current approach more robust, several questions need answers in order to move forward: How to integrate more structural information if thermal maps do not provide all that is needed? How will the information be fused in such a way that if specific scenarios might require more modalities, such as Light Detection and Ranging (LiDAR) [29], or event cameras, the network will structurally support that without architecture modifications?

The answer to the first question lies in the use of surface normal maps derived from monocular depth estimation. Specifically, DepthAnything V2 [30] is employed to extract monocular depth maps from the RGB input (which may be the original RGB or synthetic low-light adapted and enhanced depending on the configuration and objective). These monocular depth maps provide additional structural information which might be missing from the RGB and thermal maps by leveraging the pre-trained representations. Surface normals are then obtained by normalising these maps through gradient extraction.

The main innovation introduced by the core component of the VesperMUSE, its semantic segmentation module, lies in the attention mechanism employed during the encoding stage. The underlying concept is that each modality is analyzed both globally and at the level of fine local features, thereby providing a comprehensive representation. To capture the contribution of each modality, hierarchical feature maps are fused and treated as inputs to the main attention components. Given that the attention mechanism is conceptually designed for one primary modality and k ∈ N auxiliary modalities, it can be readily configured to support an arbitrary number of modalities with varying channel configurations (e.g., RGB with three channels or single-channel thermal maps). This flexibility makes the model highly adaptable to diverse scenarios and imaging conditions where the availability of different modalities is different than the current analysed scenario, thereby addressing the second question proposed before.

Data availability and adaptation. One important aspect to take into account is the limited availability of data for conducting such a study. While multi-modal semantic segmentation datasets provide a high degree of reliability, and a wide range of datasets serve as effective benchmarks for LLIE, no available dataset integrates well-lit and low-light counterparts RGB, thermal imagery, structural information, and annotated semantic maps simultaneously. Given that semantic segmentation annotations are expensive to obtain, the most appropriate approach is to utilize datasets that include well-lit RGB images, thermal/structural data, and semantic maps.

Consequently, a novel synthetic low-light adaptation pipeline is proposed as part of the Vesper-MUSE framework. The algorithm leverages local image structure and global lighting information, generating degradations that capture natural exposure variations, noise characteristics, and colour distortions. A dedicated analysis provides a quantitative evaluation of the physical and perceptual quality of the synthesized images to show the generated images’ conformity with real-world data. This integration of both local and real-world illumination information ensures that the generated images faithfully reproduce the deficiencies of natural low-light environments.

To summarize, the contributions of VesperMUSE are:

• Unified framework: VesperMUSE enables improved low-light semantic segmentation while preserving strong performance in well-lit settings.

• Thermal-guided LLIE: A highly computational-efficient enhancement module tailored to optimize downstream semantic segmentation.

• Modality-agnostic fusion: A scalable Fusion Vision Transformer supporting an arbitrary number of input modalities.

• Comprehensive evaluation: Consistent competitive performance with SOTA results across multiple benchmarks.

![](images/240fc22292820e533f42ab961e08f814f504ec4549c916f1adbf5d6fcf124d28.jpg)

Related Work

## 2.1 Synthetic Low-Light Image Adaptation

The development of LLIE models has been historically hampered by the limited dataset availability and the quality of light-deficient representations. Early efforts such as the SID dataset [31] focused on capturing short-exposure raw images along with long-exposure counterparts. This was a good representation of how a LLIE dataset should look like, however it was limited in terms of the format it used: RAW. The consumer and surveillance hardware operate much more easily using conventional formats such as PNG or JPG, which also provide more efficient computational processing. Moreover, the conversion of RAW images into other encodings fails to replicate the natural noise and illumination patterns captured in the low-light conditions, the quality of the resulting data hindering the supervised learning process of LLIE models.

More recently, the problem with data limitation was fixed with the appearance of paired real-world datasets such as LOL [32], LOL-v2 real [33], SDSD-in/out [34], and LOL-Blur [35]. Another field of concern, however, lies in the reliance of synthetic low-light adaptation methods. Widely-used current approaches, such as the work of Yang et al. through their LOL-v2 synthetic dataset [33] have provided a very solid foundation for how synthetic low-light images should be constructed: following a real-world dataset illumination distribution extracted from multiple datasets and delivering the result into an easy-to-use encoding format.

## 2.2 Low-Light Image Enhancement

Classical Methods. LLIE has been firstly addressed using plain-enhancement [36] which focused mainly on histogram equalisation [37] and gamma correction [38, 39]. These methods amplify the brightness and contrast of underexposed images through pixel intensity transformations. Because of the complex and uneven way illumination interacts with different areas from the image, these models struggled to model the light projection process, resulting in perceptually inconsistent outputs containing over-enhanced regions and amplified noise.

The later introduced Retinex theory [40] aims to decompose the image in two components: reflectance, which represents the intrinsic scene information and illuminance, focusing on lighting variations. To better account for the illumination effects on the local and global structures, LLIEmethods have employed the theory as the underlying assumption behind the image architecture [41, 42, 43]. These models focused on estimating and refining the illumination map and reconstructing the image in a later stage with the more balanced and denoised reflectance map. This component-separation approach lead to more visually appealing and plausible results than the traditional enhancement techniques which rely heavily on hand-crafted priors and assumptions, making them hard to generalize across multiple datasets.

CNN-based Methods. Besides the improvements brought in language applications or high level CV tasks, the advancement of Deep Learning (DL) has also transformed the field of image enhancement becoming the dominant paradigm [14, 44, 45]. One of the earliest DL models on the scene of LLIE was LLNet [46] which proved that supervised feature learning can improve brightness and contrast while also recovering lost details. Generative models, like EnGAN [15], focused on creating a direct mapping from low-light images to their enhanced counterparts by employing an endto-end framework. These methods further improved upon traditional hand-crafted heuristic-based algorithms through hierarchical feature extraction.

The Retinex theory [40] is furthermore employed as an assumption, leveraging the ability of DL frameworks to learn feature decomposition. While such models, benefit from both physical priors and data-driven learning, they often require multi-stage training pipelines through subnetwork specialisation [42] on tasks such as: decomposition, illumination/reflectance enhancement, and reconstruction. As a result, the computational and training complexity of this kind of models are high. To solve this, one-stage Retinex-inspired Convolutional Neural Networks (CNN) have also been proposed in order to increase the computational efficiency, but they struggle with noise mitigation and artifacts, resulting in an unsatisfactory enhancement process.

Another issue that CNN-based models exhibit is the limited receptive field of convolutional layers, hence reducing their ability to capture long-range spatial dependencies constructing the global contextual relationship across distant image regions. As light’s interaction with the scene is not linear, hence why histogram-matching models fail in the LLIE domain, this limitation restricts the models capacity of capturing diverse scene structures and mapping the relationships between them.

ViT-based Methods. To overcome the inherent locality of convolutional operators, ViT-based models [27, 47] have dominated the SOTA models charts for LLIE in the past years. The success achieved in machine translation by the Transformer architecture [48], has also spread to CV and more specifically to image enhancement. The principal particularity that ViTs possess is the selfattention module which manages to capture long-range dependencies which guide the enhancement process towards a complex resolution of the LLIE challenge.

Several architectures have marked the evolution of using ViTs [27] in LLIE. For example, UFormer [49] adapts the U-shaped [26] encoder-decoder paradigm, replacing the standard convolutions found in U-Net [26] with Transformer blocks while preserving level-wise skip connections. Retinexformer [50] incorporates illumination representations into the self-attention mechanism to model non-local interactions across regions with varying lighting conditions. More recent work, such as ISALux [51] or ModalFormer [52], successfully manage to integrate semantic and structure priors into the self-attention module aiming to compensate the light-deficiency induced detail loss with visual or feature-based information represented through different modalities.

Despite their strong performance, Transformer-based methods face notable computational challenges. The self-attention mechanism incurs substantial computational complexity due to its reliance on the spatial dimensions of the feature maps. To mitigate this limitation, several methods, such as Restormer [53], DepthLux [54], employ different strategies: computing the attention map channel wise considering the depth of the feature map as the number of channels, contracting the feature space before the self-attention module and then expanding it back to its original dimensions, or replacing convolutions with their more efficient variation, depthwise separable convolutions.

RGB-Thermal Fusion for LLIE. RGB-Thermal fusion has recently been explored to enhance robustness in extreme low-light conditions by leveraging the illumination-invariant nature of thermal imagery. In particular, some studies [55] propose a thermal-aware ViT architecture, where thermal features are explicitly embedded into the transformer pipeline to guide global feature modeling under severe illumination degradation. Complementarily, Jha et al. focus on the interaction between illumination cues derived from the low-light RGB image and the corresponding thermal map, designing a mechanism that aligns and integrates these complementary representations for more stable enhancement [56]. These works demonstrate that modeling cross-modal dependencies, especially between illumination and thermal structure, can significantly improve detail recovery and perceptual consistency in challenging low-light scenarios.

## 2.3 Modality Fusion for Image Semantic Segmentation

Multi-modal semantic segmentation has emerged as an effective strategy for improving scene understanding and feature richness limitations under real-world conditions where RGB might be insufficient [57]. Complementary sensing modalities such as thermal (infrared) or depth maps have been included as inputs to improve the information representation of a specific scene [9]. Depth maps

provide geometric and distance cues that can help disambiguate low-contrast regions and cluttered backgrounds [58], while thermal images capture thermic distributions, offering a visual representation which is not subject to change under diverse light patterns or other deficiencies often noticed in nature like smoke or haze [20]. Semantic segmentation is not the only high-level CV task that has been benefiting from adding multi-modality in learning, with object detection, tracking, autonomous driving perception systems [59] also improving performance by leveraging multiple representations of the same scenes.

In the case of RGB-Depth semantic segmentation, early methods focused on fusing modalities at the input or feature level with some approaches directly concatenating RGB and depth maps or fusing them by summation [60, 61]. In time, models have evolved to a two-stream architecture [4], where independent networks extract features separately before integrating them in the decoding process. The recent focus has been placed on feature fusion approach which separates the fusion logic from the backbone and employing modules like Separation-and-Aggregation Gate [62] or Attention Complementary Modules [63] to provide parallel modality streams at each hierarchical level.

A significant focal point of recent work is discriminative treatment of modalities [64]. This represents a change from the initial models which treated all network layers equally. More recent frameworks like LASNet [65] argue for level-specific treatment. This involves designing distinct models for different hierarchies: enhancing shallow spatial details (thermal-focused) and deep semantic features (RGB-focused) separately. Methods frequently integrate channel weighted selection and spatial-attention mechanisms to mitigate the inherent discrepancies (originating from distinct underlying data distributions) between modalities [44].

The inherent limitations of CNNs, stemming from their constrained receptive fields, are also evident in modality fusion models. The Transformer architecture [27] has emerged as a compelling solution to this issue [66]. Driven by the capacity of the self-attention mechanism to model complex rela tionships across different contexts, current research is shifting from dedicated dual-modality models toward unified architectures. While earlier methods perform well on specific pairings like RGB-Depth or RGB-Thermal [67], unified frameworks capable of integrating multiple concurrent data streams offer a much richer understanding of scene context, semantics, and structure. Consequently, these architectures yield superior performance on advanced scene understanding tasks.

## 3

Methods

## 3.1 Framework Design and Motivation

The framework behind VesperMUSE is based on three main components each having a dedicated role in the focus of this study: finding solutions for improving semantic segmentation under both low-light and well illuminated scenes.

VesperMUSE can be run both in low-light or normal light configuration. If the latter is activated, then the framework starts from Stage 3.

Stage 1: Low-Light Synthetic Adaptation. The first step is obtaining the synthetic low-light imaging which will be processed using the algorithm presented in Section 3.2.1. To link this framework as closely as possible to real world scenarios, the reasoning was physics-informed.

Stage 2: Low-Light Image Enhancement. After the low-light maps are processed they serve as inputs to the thermal-guided LLIE module described in Section 3.3, along with the thermal imaging. The LLIE model is trained under a supervised approach trying to predict the original RGB map coming from the two focus datasets: PST900 [8] and CART [25]. The training and test splits are based on the ones used for the semantic segmentation stage. The result of the training process is the best checkpoint in terms of Peak-Signal-to-Noise-Ratio (PSNR), each dataset having its own selected checkpoint.

Stage 3: Obtaining the Surface Normal maps. The RGB image (LLIE or not) serves as the input to the DepthAnything V2 [30] model, the resulting map being a monocular depth estimation. This representation is then used to produce surface normals by taking the gradients of it and thereby aiming to normalise the structural imagery.

Stage 4: Fusion-based Semantic Segmentation. The three inputs corresponding to the same scene: RGB, thermal, and surface normal are then passed to the Fusion Vision Transformer which processes each of them using a dedicated pre-trained backbone and then fuses them in a modalityagnostic manner. After the features have been fused, a decoding process commences processing the output of the framework, the segmented map. Likewise LLIE, each dataset has its own trained checkpoint selected on the mIoU performance.

## 3.2 Modalities

## 3.2.1 Synthetic low-light adaptation algorithm

To show that semantic segmentation on low-light images can be improved using additional modalities, it required a robust real-environments informed synthesis pipeline. The current methods are limited in terms of: input data format (relying on the RAW format), noise modeling (linear noise used as implicit), global histogram matching (introducing unwanted artifacts, as colour and luminance information are not separated), and fixed parameter configuration (controlling environment variables is impossible).

As a result, VesperMUSE framework proposes a novel pipeline for obtaining synthetic low-light images from RGB data. The algorithm aims to achieve a physics-informed degradation, accounting for the effect low-light conditions have on the visual representations.

Real-World Illumination Distribution. First of all, based on a significantly broader range of realworld datasets compared to prior work, namely (LOL [32], LOL-v2 real [33], SDSD-in/out [34], LOL Blur [35]), a cumulative illumination distribution is computed using the Y channel obtained after conversion to the YCbCr colour space. This choice is motivated by the fact that YCbCr provides a natural representation of illumination through its luminance component.

Let Γ denote the transformation from RGB to YCbCr, and let $\Lambda _ { Y }$ represent the projection operator that extracts the luminance (Y) channel from the transformed representation. Considering the operator $\Gamma _ { Y }$ that converts an RGB image $\mathcal { T } \in \mathbb { R } ^ { H \times W \times 3 }$ , with H and W denoting the image height and width, respectively, and separates the Y channel, the luminance extraction can be formally defined:

$$
\Gamma _ { Y } : \mathbb { R } ^ { H \times W \times 3 } \to \mathbb { R } ^ { H \times W } , \quad \Gamma _ { Y } ( \mathbb { Z } ) = \Lambda _ { Y } \big ( \Gamma ( \mathbb { Z } ) \big ) .\tag{3.1}
$$

Using the Y dimensionality of samples, the empirical cumulative distribution function (CDF) of luminance over the entire multi-dataset collection can be defined as follows. Let K denote the total number of images in the collection, $\Omega _ { k }$ the set of pixel coordinates in image $\mathcal { T } _ { k }$ , and $\begin{array} { r } { N = \sum _ { k = 1 } ^ { K } \left| \Omega _ { k } \right| } \end{array}$ the total number of pixels across all images. Then, the CDF that represents the cumulative proportion of pixels in the dataset is defined as:

$$
\mathcal { F } _ { Y } = \frac { 1 } { N } \sum _ { k = 1 } ^ { K } \Big | \big \{ \boldsymbol { x } \in \Omega _ { k } \big \} \Big | .\tag{3.2}
$$

After the distribution over the low-light image has been computed, each original RGB image from the 2 datasets used: PST900 [8] and CART [25], is passed through the following pipeline.

Retinex Decomposition. To better approximate illumination variations under low-light conditions, a Single-Scale Retinex (SSR) [68] approximation is employed to decouple each image into illumination and reflectance components. Given the luminance channel $Y$ of an input image $\mathcal { T } \in \mathbb { R } ^ { H \times W \times 3 }$ the decomposition assumes that the observed intensity at pixel x can be expressed as the product of illumination $L ( x )$ and reflectance $R ( x )$ :

$$
Y ( x ) = L ( x ) \cdot R ( x ) .\tag{3.3}
$$

An important point about the luminance channel $Y$ is that even though it encodes brightness, it is composed of illumination (lighting) and reflectance elements (texture, albedo) and it is not equivalent with the illumination component assumed in the Retinex Theory [40].

For the purpose of this study, the illumination component is approximated using a Gaussian smoothing operator $( { \mathcal { G } } _ { \sigma } )$ with kernel standard deviation $\sigma = 1 5$ , reflecting the assumption that illumination varies slowly across the spatial domain. The reflectance component, which captures high-frequency structural details, is obtained via point-wise division, where ϵ is a small constant introduced for numerical stability:

$$
L = \mathcal { G } _ { \sigma } ( Y ) , \quad R = \frac { Y } { L + \epsilon } .\tag{3.4}
$$

Real-world Distribution Matching. After decomposing the image into illumination and reflectance components, the illumination channel can be manipulated independently while preserving structural details. To emulate real-world low-light conditions (Figure 3.1), the illumination map $L ( x )$ is adjusted to match the cumulative luminance distribution $\mathcal { F } _ { Y }$ . This is achieved using histogram specification:

$$
L _ { \mathrm { m a t c h e d } } ( x ) = \mathcal { F } _ { Y } ^ { - 1 } \Big ( \mathcal { F } _ { L } \big ( L ( x ) \big ) \Big ) .\tag{3.5}
$$

Synthetic Luminance Reconstruction. Once the illumination has been adjusted, the synthetic lowlight luminance channel is reconstructed by combining the matched illumination with the original reflectance $R ( x )$ , ensuring that the image retains its original structural details.

$$
Y _ { \mathrm { r e t i n e x } } ( x ) = L _ { \mathrm { m a t c h e d } } ( x ) \cdot R ( x ) .\tag{3.6}
$$

Because histogram matching can introduce nonlinear intensity remapping and alter the global energy of the illumination component, it may produce luminance values that deviate from physically plausible exposure changes and lead to inconsistencies in overall brightness. There is therefore a need to enforce a globally consistent exposure adjustment that compensates for shifts in illumination statistics while preserving structural fidelity. A mean-based scaling factor is derived from the ratio between the matched and original illumination components, ensuring that the synthetic luminance remains photometrically coherent and consistent with realistic camera exposure behavior.

$$
s = \frac { \mathbb { E } [ L _ { \mathrm { m a t c h e d } } ] } { \mathbb { E } [ L ] + \varepsilon } .\tag{3.7}
$$

$$
Y _ { \mathrm { l i n e a r } } ( x ) = s \cdot Y ( x ) .\tag{3.8}
$$

Structure-Aware Luminance Blending. The main advantage of the proposed method is taking into account the current luminance level of images. If the conditions are already light-deficient due to real-environment conditions, the synthetic low-light degradation can be tuned to be low-impact.

A spatially adaptive blending mask is introduced to reconcile the complementary properties of the two luminance reconstructions. While the Retinex-based luminance enforces realistic illumination statistics, its spatially varying remapping may compress bright structures and attenuate salient edges. In contrast, the globally scaled luminance preserves structural consistency but does not fully reflect real-world low-light illumination distributions. To prevent over-suppression of high-intensity regions and to maintain perceptual stability in structurally significant areas, a highlight-aware blending mechanism is employed.

![](images/cee2212824807fe001829dc92bc28e9b9135f4a825b74489a89e87baef7683c4.jpg)  
Fig. 3.1. An illustration of the real-world light distribution which is followed in the Real-world Distribution Matching stage

Specifically, let $\tau = 1 5 0$ denote the luminance threshold beyond which highlight preservation becomes increasingly important, and let $\kappa = 1 0 0$ control the smooth transition range controlling how rapidly the blending weight increases from 0 to 1. Incorporating a structure bias parameter $\beta \in [ 0 , 1 ]$ guarantees a minimum contribution from the globally scaled branch. The mask is used to balance between the two illumination components:

$$
M ( x ) = \mathrm { m a x } \left( \mathrm { c l i p } \left( { \frac { Y ( x ) - \tau } { \kappa } } , 0 , 1 \right) , \beta \right) .\tag{3.9}
$$

$$
Y _ { \mathrm { f i n a l } } ( x ) = M ( x ) Y _ { \mathrm { l i n e a r } } ( x ) + \left( 1 - M ( x ) \right) Y _ { \mathrm { r e t i n e x } } ( x ) .\tag{3.10}
$$

Colour Re-integration. Before applying global exposure adjustments and simulating sensor noise, the synthetic luminance channel must be recombined with the original chrominance information. The synthetic low-light RGB image, $\mathcal { T } _ { \mathrm { l o w \mathrm { - r g b } } } \in \mathbb { R } ^ { H \times W \times 3 }$ , is reconstructed by merging $Y _ { \mathrm { f i n a l } }$ with the original $C r$ and $C b$ channels and applying the inverse colour space transformation $\Gamma ^ { - 1 }$ :

$$
\mathcal { T } _ { \mathrm { l o w - r g b } } ( x ) = \Gamma ^ { - 1 } \Big ( Y _ { \mathrm { f i n a l } } ( x ) , C r ( x ) , C b ( x ) \Big ) .\tag{3.11}
$$

Exposure Scaling (optional). Given the capturing conditions of real-world low-light images, which might include constant settings or scenarios, the pipeline includes a random sampling step. A global dimming factor is applied directly to the 3-channel RGB image to add variability and di-

versity in exposure patterns. Given the exposure range $[ s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ]$ , the scaled image is computed as:

$$
\mathcal { T } _ { \mathrm { s c a l e d } } ( x ) = \mathrm { c l i p } \left( s _ { r } \cdot \mathcal { T } _ { \mathrm { l o w } , \mathrm { r g b } } ( x ) , 0 , 2 5 5 \right) , \quad s _ { r } \sim \mathcal { U } ( s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ) .\tag{3.12}
$$

Noise Simulation. In low-light conditions, image sensors exhibit two primary types of noise: photon shot noise, caused by the discrete nature of light, and read noise, introduced by the sensor electronics. To realistically model these effects, the gamma-compressed RGB image $\mathcal { T } _ { \mathrm { s c a l e d } } ( x )$ (which is encoded in the 0–255 range) is first converted into a physically meaningful linear intensity representation $\mathcal { T } _ { \mathrm { l i n e a r } } ( x ) \in \mathbb { R } ^ { H \times W \times 3 }$

This linearisation ensures that the added noise correctly reflects the underlying photon and electronic statistics across all three colour channels, naturally simulating real-world chroma noise. The conversion is performed by normalizing $\mathcal { T } _ { \mathrm { s c a l e d } }$ to [0, 1] and inverting the gamma with an exponent of 2.2, a common approximation for sRGB images:

$$
\mathcal { T } _ { \mathrm { l i n e a r } } ( x , c ) = \left( \frac { \mathcal { T } _ { \mathrm { s c a l e d } } ( x , c ) } { 2 5 5 } \right) ^ { 2 . 2 } , \quad c \in \{ R , G , B \} .\tag{3.13}
$$

Noise is then added to this linear signal independently for each channel. Photon shot noise is modeled as a Poisson process scaled by $\lambda = 3 0 0$ , and read noise is modeled as zero-mean Gaussian noise with standard deviation $\sigma = 0 . 0 1$ . A tunable gain factor g controls the overall noise intensity:

$$
\mathcal { T } _ { \mathrm { n o i s y } } ( x , c ) = \mathcal { T } _ { \mathrm { i n c a r } } ( x , c ) + g \cdot \left( \frac { \mathrm { P o i s s o n } \left( \mathcal { T } _ { \mathrm { i n e a r } } ( x , c ) \cdot \lambda \right) } { \lambda } - \mathcal { T } _ { \mathrm { i n e a r } } ( x , c ) + \mathcal { N } ( 0 , \sigma ^ { 2 } ) \right) .\tag{3.14}
$$

The noisy linear signal is then clipped to [0, 1] and transformed back to the sRGB domain using standard gamma correction:

$$
\mathcal { T } _ { \mathrm { s r g b } } ( x , c ) = 2 5 5 \cdot \Big ( \mathrm { c l i p } ( \mathcal { T } _ { \mathrm { n o i s y } } ( x , c ) , 0 , 1 ) \Big ) ^ { \frac { 1 } { 2 . 2 } } .\tag{3.15}
$$

Hot Pixel Removal. Because statistical noise simulation can occasionally produce isolated, extremeintensity pixels (hot pixels) atypical for modern sensors, a conditional median filter is applied as a

final post-processing step. Let $\mathcal { M } _ { 3 \times 3 }$ denote a $3 \times 3$ median filter. To preserve high-frequency structural details across the majority of the image, the filter is only applied to replace pixels exceeding a high-intensity threshold (> 250):

$$
\mathcal { T } _ { \mathrm { f i n a l } } ( x , c ) = \left\{ \begin{array} { l l } { \mathcal { M } _ { 3 \times 3 } \left( \mathcal { T } _ { \mathrm { s r g b } } \right) ( x , c ) , } & { \mathrm { i f } \mathcal { T } _ { \mathrm { s r g b } } ( x , c ) > 2 5 0 . } \\ { } & { } \\ { \mathcal { T } _ { \mathrm { s r g b } } ( x , c ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{3.16}
$$

## 3.2.2 Auxiliary modalities

VesperMUSE’s segmentation module works with three modalities: the main modality (well-lit RGB or LLIE map obtained with the algorithm presented in Section 3.2.1 and the thermal-guided enhancement model described in Section 3.3), and two auxiliary modalities: thermal and surface normal derived from the monocular depth map obtained from the main RGB modality.

Thermal. As previously discussed, the thermal modality provides enhanced robustness under varying illumination conditions. Furthermore, its effectiveness is largely attributed to the intrinsic physical properties it captures, namely the thermal inertia and emissivity of different materials and surfaces. These characteristics offer complementary information to the Fusion Vision Transformer, beyond what is available in the visible spectrum.

In the PST900 dataset [8], which depicts a subterranean environment with minimal heat sources, thermal imaging acts as a reliable structural cue, enabling the detection of humans even in the presence of obscurants. Conversely, in the CART dataset [25], the thermal modality facilitates the discrimination of diverse textures commonly encountered in natural outdoor environments.

Surface Normal. To better extract the context of the scene, a monocular depth map is extracted to highlight the structural and material differences between objects or surfaces. The noise and colour distortions, attenuated but still present in the enhanced RGB images and the thermal consistency of nearby objects in thermal maps might misguide the network in distinguishing edges and separation of entities. The first order gradients of the depth map are taken and turned into unit normal vectors to isolate and emphasize the geometric discontinuities where objects might protrude from their backgrounds. Furthermore, the scale-invariant representation of the geometric features provides the robustness needed in terms of the relative position of the objects to the camera, enabling a informative features extraction for close objects, as found in the PST900 dataset [8], and distant objects as in the CART dataset [25].

Hence, a pretrained version of DepthAnything $\mathrm { V } 2 [ 3 0 ] , { \mathscr { D } _ { \theta } ( \cdot ) }$ , which is kept frozen during training and inference is employed to extract a monocular depth map from the main RGB modality, $\mathcal { T } _ { R G B } \in$ $\mathbb { R } ^ { H \times W \times 3 }$

$$
\begin{array} { r } { \mathbf { D } = \mathcal { D } _ { \theta } ( \mathcal { T } _ { R G B } ) . } \end{array}\tag{3.17}
$$

Surface Normal Computation. Minimum-maximum normalisation is performed on the monocular depth map. Given the normalised depth map D, the surface normals are computed and normalised to [0, 1]:

$$
\hat { \mathbf { n } } ( x , y ) = \frac { \left( - \frac { \partial \mathbf { D } } { \partial x } , \mathbf { \Pi } - \frac { \partial \mathbf { D } } { \partial y } , \mathbf { \Pi } 1 \right) } { \left\| \left( - \frac { \partial \mathbf { D } } { \partial x } , \mathbf { \Pi } - \frac { \partial \mathbf { D } } { \partial y } , \mathbf { \Pi } 1 \right) \right\| _ { 2 } } .\tag{3.18}
$$

## 3.3 Low-Light Image Enhancement Vision Transformer

The architecture of the VesperMUSE LLIE module follows a U-shaped [26] encoder-decoder structure, a widely adopted design in ViT-based [27] image restoration due to its capability for hierarchical, multi-scale feature extraction and progressive feature refinement. The overall structure is organized into three principal components: an encoder, a bottleneck, and a decoder.

The inputs consist of a synthetic low-light RGB image, denoted as $\mathcal { T } _ { \mathrm { R G B } } \in \mathbb { R } ^ { H \times W \times 3 }$ , and a corresponding thermal representation, $\mathcal { T } _ { \mathrm { T h e r m a l } } \ \in \ \mathbb { R } ^ { H \times W \times 1 }$ . Let V esperM $U S E _ { \mathrm { L L I E } }$ denote the LLIE module of VesperMUSE. The enhancement process can be formally expressed as:

$$
\hat { \mathcal { I } } _ { \mathrm { R G B } } = V e s p e r M U S E _ { \mathrm { L L I E } } \left( \mathcal { T } _ { \mathrm { R G B } } , \mathcal { T } _ { \mathrm { T h e r m a l } } \right) , \quad \hat { \mathcal { L } } _ { \mathrm { R G B } } \in \mathbb { R } ^ { H \times W \times 3 } .\tag{3.19}
$$

As the enhancement process commences, the RGB component is first mapped into a high-dimensional latent feature space through a pointwise convolution projection, with output depth, $C ,$ denoting the base embedding dimension. The projection aligns the spatial features with the transformer embedding space while preserving spatial resolution.

$$
\begin{array} { r } { \boldsymbol { F } ^ { 0 } = \operatorname { c o n v } _ { 1 \times 1 } ( \mathcal { T } _ { \mathrm { R G B } } ) , \quad \boldsymbol { F } ^ { 0 } \in \mathbb { R } ^ { H \times W \times C } . } \end{array}\tag{3.20}
$$

Encoding Stage. Feature extraction is performed using a two-stage hierarchical encoder composed of Thermal-aware Transformer Block described in the following section. After each block, the features are spatially reduced in terms of the height and the width by a factor of 2 through $\mathrm { c o n v } _ { 3 \times 3 }$ layers with a stride of 2, but also doubled in depth, for maintaining representational power. Each block integrates RGB features with scale-matched thermal representation. For each resolution level $k \in \{ 0 , 1 \}$ , the encoder generates $\mathcal { F } _ { \mathrm { e n c } } ^ { k } \in \mathbb { R } ^ { \frac { H } { 2 ^ { k } } \times \frac { W } { 2 ^ { k } } \times C \cdot 2 ^ { k } }$

Bottleneck. At the coarsest resolution, a dedicated Thermal-aware Transformer Block processes the compressed representation to model long-range dependencies and cross-modal correlations by aggregating global context under thermal guidance before reconstruction begins. The resulting latent tensor is yielding $\mathcal { F } _ { \mathrm { l a t e n t } } \in \mathbb { R } ^ { \frac { H } { 4 } \times \frac { W } { 4 } \times 4 C }$

![](images/e408c5e1f7ce3226104fb3e4ec51728bd3b65599f03c5d6991ac7b0c13948815.jpg)  
Fig. 3.2. Visual representation of the VesperMUSE LLIE module.

Decoding stage. The reconstruction path mirrors the encoder through two upsampling stages. Each stage applies transposed $\mathrm { c o n v } _ { 3 \times 3 }$ for spatial enlargement, concatenates the upsampled features with their encoder counterparts, and refines the fused representation using a Transformer block. For each decoding level $k \in \{ 1 , 0 \}$ , the representations are $\mathcal { F } _ { \mathrm { d e c } } ^ { k } \in \mathbb { R } ^ { \frac { H } { 2 ^ { k } } \times \frac { W } { 2 ^ { k } } \times C \cdot 2 ^ { k } }$

Image Reconstruction. The output of the decoding phase, $\mathcal { F } _ { \mathrm { f i n a l } } \in \mathbb { R } ^ { H \times W \times C }$ , is mapped back to the RGB domain through a pointwise convolution, producing the enhanced output:

$$
\begin{array} { r } { \hat { \mathcal { T } } _ { \mathrm { R G B } } = \mathrm { c o n v } _ { 1 \times 1 } ( \mathcal { F } _ { \mathrm { f n a l } } ) , \quad \hat { \mathcal { T } } _ { \mathrm { R G B } } \in \mathbb { R } ^ { H \times W \times 3 } . } \end{array}\tag{3.21}
$$

## 3.3.1 Thermal-aware Transformer Block

The core component of the LLIE module is represented by the Thermal-aware Transformer Block which has 2 main constituents: the RGB-Thermal Cross-Attention (RGB-T-CA) and the Feed-Forward Network (FFN). At each step, the resulting feature maps from the two components are normalized using Layer Normalisation (LN) and added with the skip connections, originating from the previous step. Hence, with k as the level of the block, $\mathcal { F } _ { \mathrm { T h e r m a l } }$ as correspondent thermal feature and $\mathcal { F } _ { i n }$ as the input, the functionality of the entire block can be expressed as:

$$
\mathcal { F } _ { i n } ^ { \prime } = \mathcal { F } _ { i n } + \mathrm { L N } ( \mathrm { R G B - T - C A } \left( \mathcal { F } _ { i n } , \mathcal { F } _ { \mathrm { T h e r m a l } } \right) ) ,\tag{3.22}
$$

$$
\mathcal { F } _ { o u t } = \mathcal { F } _ { i n } ^ { \prime } + \mathrm { L N } ( \mathrm { F F N } ( \mathcal { F } _ { i n } ^ { \prime } ) ) ,\tag{3.23}
$$

$$
\mathcal { F } _ { i n } , \mathcal { F } _ { \mathrm { T h e r m a l } } \in \mathbb { R } ^ { \frac { H } { 2 ^ { k } } \times \frac { W } { 2 ^ { k } } \times C \cdot 2 ^ { k } } .\tag{3.24}
$$

## 3.3.2 RGB-Thermal Cross-Attention

The dedicated self-attention module is the main driver of the enhancement process because it manages to capture the underlying patterns of the data which guide the restoration of lost details due to low-light conditions. Since RGB is the map which is degraded because of the direct interaction with poor-lighting, thermal (infrared) maps are employed to provide information about the local and global structure of the image leaving the colour information to be extracted from the low-light RGB map.

Given the RGB feature map $\mathcal { F } _ { \mathrm { R G B } } \in \mathbb { R } ^ { H \times W \times C }$ and the thermal feature map $\mathcal { F } _ { \mathrm { T h e r m a l } } \in \mathbb { R } ^ { H \times W \times C }$ query, key, and value embeddings are obtained through pointwise convolution projection:

$$
\begin{array} { r } { Q , K , V = \mathrm { c o n v } _ { 1 \times 1 } \big ( \mathcal { F } _ { \mathrm { R G B } } , \mathcal { F } _ { \mathrm { R G B } } , \mathcal { F } _ { \mathrm { T h e r m a l } } \big ) , \quad Q , K , V \in \mathbb { R } ^ { H \times W \times C } . } \end{array}\tag{3.25}
$$

The tensors are divided into h attention heads, each with channel dimension $C / h$ . Given that $\mathcal { T } _ { i } \in$ $\mathbb { R } ^ { + }$ is a learnable temperature parameter, the attention scores for each head are computed as:

$$
\mathrm { A t t n } _ { i } = \mathrm { s o f t m a x } \left( \mathcal { T } _ { i } Q _ { i } K _ { i } ^ { \top } \right) .\tag{3.26}
$$

The attention scores are then multiplied with the thermal V projection concatenated to gather the head representation back and projected using a pointwise convolution as:

$$
\mathcal { F } _ { \mathrm { T h e r m a l . A t t n } } = \mathrm { c o n v } _ { 1 \times 1 } \Bigg ( \mathrm { C o n c a t } \Big ( \{ A t t n _ { i } V _ { i } \} _ { i = 1 } ^ { h } \Big ) \Bigg ) , \quad \mathcal { F } _ { \mathrm { T h e r m a l . A t t n } } \in \mathbb { R } ^ { H \times W \times C } .\tag{3.27}
$$

## 3.3.3 Feed Forward Network

To further refine the thermal-guided feature map, $\mathcal { F } _ { \mathrm { T h e r m a l - A t t n } } \in \mathbb { R } ^ { H \times W \times C }$ , the FFN section consists of three consecutive convolutional layers (with $1 \times 1$ and $3 \times 3$ kernels) applied sequentially. These layers are interleaved with Gaussian Error Linear Unit (GELU) activations, denoted by $\sigma .$ . The FFN employed in VesperMUSE can thus be formally expressed as:

$$
\begin{array} { r } { \mathcal { F } _ { \mathrm { o u t } } = \mathrm { F F N } ( \mathcal { F } _ { \mathrm { T h e r m a l - A t t n } } ) = \mathrm { c o n v } _ { 1 \times 1 } \Big ( \sigma \big ( \mathrm { c o n v } _ { 3 \times 3 } \big ( \sigma \big ( \mathrm { c o n v } _ { 1 \times 1 } ( \mathcal { F } _ { \mathrm { T h e r m a l - A t t n } } ) \big ) \big ) \big ) \Big ) . } \end{array}\tag{3.28}
$$

## 3.3.4 Low-Light Image Enhancement Loss function

Hybrid Loss. The LLIE objective is formulated as a weighted hybrid loss that jointly enforces pixel-level fidelity, perceptual realism, structural consistency, and semantic preservation. Specifically, the loss combines: (i) a Mean Absolute Error (MAE) reconstruction term to ensure and intensity correctness, (ii) a perceptual loss computed in a pretrained feature space to enhance visual quality, (iii) a structural similarity loss to preserve local structures, and (iv) a semantic consistency term to maintain high-level content alignment with the ground truth, helping for the semantic focus required by the downstream semantic segmentation module. The relative contribution of each component is controlled by weighting coefficients $\alpha _ { 1 } , \alpha _ { 2 } , \alpha _ { 3 }$ , and $\alpha _ { 4 }$

$$
\mathcal { L } _ { \mathrm { L L I E } } = \alpha _ { 1 } \mathcal { L } _ { \ell _ { 1 } } + \alpha _ { 2 } \mathcal { L } _ { \mathrm { p e r c } } + \alpha _ { 3 } \mathcal { L } _ { \mathrm { S S I M } } + \alpha _ { 4 } \mathcal { L } _ { \mathrm { s e m } } .\tag{3.29}
$$

MAE Reconstruction Loss. Let $\mathbf { Y } \in \mathbb { R } ^ { H \times W \times C }$ denote the ground-truth image, $\hat { \mathbf { Y } } \in \mathbb { R } ^ { H \times W \times C }$ the enhanced prediction, $N = H \times W \times C$ the total number of elements (pixels and channels), and i the linear index over spatial and channel dimensions. The mean absolute error (MAE) reconstruction loss is defined as:

$$
\mathcal { L } _ { \ell _ { 1 } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \hat { \mathbf { Y } } _ { i } - \mathbf { Y } _ { i } \right| .\tag{3.30}
$$

Perceptual Loss. Let $\phi ( \cdot )$ denote a fixed pretrained feature extractor (VGG-19 [69], $\phi ( \mathbf { Y } ) \in \mathbb { R } ^ { M }$ and $\phi ( \hat { \mathbf { Y } } ) \in \mathbb { R } ^ { M }$ the corresponding feature representations, and M the total number of extracted feature elements. The perceptual loss is defined as:

$$
\mathcal { L } _ { \mathrm { p e r c } } = \frac { 1 } { M } \left. \phi ( \hat { \mathbf { Y } } ) - \phi ( \mathbf { Y } ) \right. _ { 2 } ^ { 2 } .\tag{3.31}
$$

Multi-Scale SSIM Loss. Let $\hat { \mathbf Y }$ and Y denote the enhanced and ground-truth images, respectively, $J$ the total number of scales, $\mu _ { \hat { Y _ { j } } }$ and $\mu _ { Y _ { j } }$ the local means at scale j, $\sigma _ { \hat { Y _ { j } } }$ and $\sigma _ { Y _ { j } }$ the corresponding standard deviations, $\sigma _ { \hat { Y _ { j } } Y _ { j } }$ the cross-covariance, $C _ { 1 } , C _ { 2 } , C _ { 3 }$ small positive constants for numerical stability, and $\alpha _ { J } , \beta _ { j } , \gamma _ { j }$ scale-dependent weighting exponents. The luminance $l _ { j } .$ , contrast $c _ { j }$ , and structural $s _ { j }$ components at scale j are defined as:

$$
l _ { j } = \frac { 2 \mu _ { \hat { Y } _ { j } } \mu _ { Y _ { j } } + C _ { 1 } } { \mu _ { \hat { Y } _ { j } } ^ { 2 } + \mu _ { Y _ { j } } ^ { 2 } + C _ { 1 } } , \quad c _ { j } = \frac { 2 \sigma _ { \hat { Y } _ { j } } \sigma _ { Y _ { j } } + C _ { 2 } } { \sigma _ { \hat { Y } _ { j } } ^ { 2 } + \sigma _ { Y _ { j } } ^ { 2 } + C _ { 2 } } , \quad s _ { j } = \frac { \sigma _ { \hat { Y } _ { j } Y _ { j } } + C _ { 3 } } { \sigma _ { \hat { Y } _ { j } } \sigma _ { Y _ { j } } + C _ { 3 } } .\tag{3.32}
$$

The Multi-Scale Structure Similarity index metric (MS-SSIM) taking into account the luminance, contrast and structure components is defined as, with the loss represented as the distance from the perfect score:

$$
\mathrm { M S - S S I M } ( \hat { \mathbf { Y } } , \mathbf { Y } ) = ( l _ { J } ) ^ { \alpha _ { J } } \prod _ { j = 1 } ^ { J } ( c _ { j } ) ^ { \beta _ { j } } ( s _ { j } ) ^ { \gamma _ { j } } ,\tag{3.33}
$$

$$
\mathcal { L } _ { \mathrm { S S I M } } = 1 - \mathrm { M S - S S I M } ( \hat { \mathbf { Y } } , \mathbf { Y } ) .\tag{3.34}
$$

Semantic Consistency Loss. While the perceptual loss enforces similarity in a generic feature space to improve visual realism, it does not explicitly guarantee preservation of task-relevant semantic content. In LLIE, an approach which produces visually appealing images through contrast and colour balance amplification, does not imply that the model is optimized to enhance images for downstream tasks. Usually, it is quite the opposite as the type of visual composition which is appealing to the human eye is not informative enough for computer scene-understanding tasks such as semantic segmentation or object detection.

To explicitly constrain the enhancement process toward semantic fidelity, a frozen pretrained ResNet50 [28] encoder $\psi ( \cdot )$ is employed as a high-level feature regularizer. This encoder acts as a fixed semantic prior, ensuring that the enhanced image preserves discriminative object representations consistent with the ground truth. By aligning deep semantic embeddings, the enhancement network is guided toward solutions that not only improve visual quality but also facilitate improved optimization for subsequent segmentation models.

Let $\psi ( \cdot )$ denote the frozen semantic feature extractor, $\psi ( \mathbf { Y } ) \in \mathbb { R } ^ { K }$ and $\psi ( \hat { \mathbf { Y } } ) \in \mathbb { R } ^ { K }$ the corresponding high-level feature representations of the ground-truth and enhanced images, respectively, and K the total number of semantic feature elements. The semantic consistency loss is defined as:

$$
\mathcal { L } _ { \mathrm { s e m } } = \frac { 1 } { K } \left. \psi ( \hat { \mathbf { Y } } ) - \psi ( \mathbf { Y } ) \right. _ { 1 } .\tag{3.35}
$$

## 3.3.5 Low-Light Image Enhancement Training

The LLIE module is trained for 300K iterations using the Adam optimizer [70] with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ and momentum parameters $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ . Gradient clipping is applied to ensure stable optimization. No warm-up phase is used.

Learning Rate Schedule. A two-stage cosine annealing restart schedule is employed. Training is divided into two consecutive cycles with a starting rate of $2 \times 1 0 ^ { - 4 }$ increasing to $3 \times 1 0 ^ { - 4 }$ up to 92K iterations, and then decreasing to $1 \times 1 0 ^ { - 6 }$ across 208K iterations. During the cosine phase, the learning rate at iteration t within the second cycle is computed as:

$$
\eta _ { t } = \eta _ { \mathrm { m i n } } + \frac { 1 } { 2 } ( \eta _ { 0 } - \eta _ { \mathrm { m i n } } ) \left( 1 + \cos \left( \frac { \pi t } { T } \right) \right) .\tag{3.36}
$$

The initial constant learning rate phase enables stable large-step optimization and rapid convergence in the early stage. The subsequent cosine annealing phase progressively reduces the update magnitude, facilitating fine-grained parameter refinement and improved restoration fidelity. The training is conducted on $2 5 6 \times 2 5 6$ patches randomly cropped from paired low-light and normal-light images, with a batch size of 8.

## 3.4 Fusion Vision Transformer

Encoding Phase. The objective of the Fusion Vision Transformer is to effectively integrate an arbitrary number of input modalities, defined as a primary modality, $M _ { \mathrm { m a i n } }$ , and $k \in \mathbb N$ auxiliary modalities, $M _ { \overline { { 1 , k } } }$

The modality-agnostic nature of the architecture is enabled by the self-attention mechanism within the Fusion Encoder Block. This design effectively integrates global and local patterns from the auxiliary modalities while remaining inherently scalable. It accommodates an arbitrary number of inputs, without requiring structural modifications to the underlying fusion method.

Features are processed using a encoder-decoder path, where the encoding process is conducted in 4 consecutive steps each of them being represented by the Fusion Encoding Block. The decoding process does not perform fusion as it process a single feature space originating from the end of the encoding stage.

![](images/58f56dae3b62f8fb9cfd5574b77c7bb5627dd38ac6506c84c5db3fdf75c76077.jpg)  
Fig. 3.3. Visual representation of the VesperMUSE Semantic Segmentation module.

During the encoding stage, the feature representations for each modality are enriched using a ResNet50 [28] (≈ 25.6M parameters) for the primary modality and ResNet18 [28] (≈ 11.7M parameters) backbones for the auxiliary modalities. This asymmetric backbone selection is motivated by computational efficiency, while the lighter ResNet18 [28] architectures minimize overhead, the deeper ResNet50 [28] is utilized for the RGB stream to leverage its superior representational power, which is critical for capturing the complex visual features required within the scope of VesperMUSE.

Before reaching the first Fusion Encoder Block, the inputs are formally defined as $\mathcal { T } _ { \mathrm { R G B } } \in \mathbb { R } ^ { H \times W \times 3 }$ $\mathcal { T } _ { \mathrm { T h e r m a l } } \in \mathbb { R } ^ { H \times W \times 1 }$ , and $\mathcal { T } _ { \mathrm { S u r f . N o r m a l s } } \in \mathbb { R } ^ { H \times W \times 3 }$ . Each modality is passed through its corresponding Feature Stem Network derived from a ResNet [28] backbone. Since all backbones expect threechannel inputs, the first convolutional layer in the ResNet18 [28] Stem, used for the thermal modality, is adapted to support single-channel inputs following a per-input-channel kernel summation approach:

$$
\tilde { W } ^ { ( 1 ) } = \sum _ { c = 1 } ^ { 3 } W _ { c } ^ { ( 1 ) } , \quad \tilde { W } ^ { ( 1 ) } \in \mathbb { R } ^ { 7 \times 7 \times 1 \times C } .\tag{3.37}
$$

This modification collapses the three input channels of the first convolutional layer into a single one, enabling compatibility with uni-dimensional thermal inputs while preserving the pretrained weights.

The encoding pathway of the Fusion Vision Transformer is composed of four cascaded Fusion Encoder Blocks. Each block sequentially processes the latent features through a Fusion Self-Attention module and a FFN before propagating the refined representations to the subsequent stage.

To formalize the hierarchical feature extraction across all encoding stages $k \in \{ 1 , \ldots , 4 \}$ , let the outputs of the initial Stem Feature Networks be denoted as:

$$
\begin{array} { r l r } & { \mathcal { F } _ { \mathrm { R G B } } ^ { 0 } = \mathrm { S t e m } _ { \mathrm { R e s N e t 5 0 } } ( \mathcal { T } _ { \mathrm { R G B } } ) , } & { \mathcal { F } _ { \mathrm { R G B } } ^ { 0 } \in \mathbb { R } ^ { \frac { H } { 4 } \times \frac { W } { 4 } \times 6 4 } . } \\ & { \mathcal { F } _ { \mathrm { T h e r m a l } } ^ { 0 } = \mathrm { S t e m } _ { \mathrm { R e s N e t 1 8 } } ( \mathcal { T } _ { \mathrm { T h e r m a l } } ) , } & { \mathcal { F } _ { \mathrm { T h e r m a l } } ^ { 0 } \in \mathbb { R } ^ { \frac { H } { 4 } \times \frac { W } { 4 } \times 6 4 } . } \\ & { \mathcal { F } _ { \mathrm { S u r f . ~ N o r m a l s } } ^ { 0 } = \mathrm { S t e m } _ { \mathrm { R e s N e t 1 8 } } ( \mathcal { T } _ { \mathrm { S u r f . ~ N o r m a l s } } ) , } & { \mathcal { F } _ { \mathrm { S u r f . ~ N o r m a l s } } ^ { 0 } \in \mathbb { R } ^ { \frac { H } { 4 } \times \frac { W } { 4 } \times 6 4 } . } \end{array}\tag{3.38}
$$

The feature maps for each subsequent stage are computed by propagating the representations through the corresponding residual levels of the parallel ResNet backbones:

$$
\begin{array} { r l r l } & { \mathcal { F } _ { \mathrm { R G B } } ^ { k } = \mathrm { R e s N e t . L a y e r } _ { 5 0 } ^ { k } \left( \mathcal { F } _ { \mathrm { R G B } } ^ { k - 1 } \right) , } & & { \mathcal { F } _ { \mathrm { R G B } } ^ { k } \in \mathbb { R } ^ { \frac { H } { 2 ^ { k + 1 } } \times \frac { W } { 2 ^ { k + 1 } } \times ( 2 5 6 . 2 ^ { k - 1 } ) } . } \\ & { \mathcal { F } _ { \mathrm { T h e r m a l } } ^ { k } = \mathrm { R e s N e t . L a y e r } _ { 1 8 } ^ { k } \left( \mathcal { F } _ { \mathrm { T h e r m a l } } ^ { k - 1 } \right) , } & & { \mathcal { F } _ { \mathrm { T h e r m a l } } ^ { k } \in \mathbb { R } ^ { \frac { H } { 2 ^ { k + 1 } } \times \frac { W } { 2 ^ { k + 1 } } \times ( 6 4 . 2 ^ { k - 1 } ) } . } \\ & { \mathcal { F } _ { \mathrm { S u r f . N o r m a l s } } ^ { k } = \mathrm { R e s N e t . L a y e r } _ { 1 8 } ^ { k } \left( \mathcal { F } _ { \mathrm { S u r f . N o r m a l s } } ^ { k - 1 } \right) , } & & { \mathcal { F } _ { \mathrm { S u r f . N o r m a l s } } ^ { k } \in \mathbb { R } ^ { \frac { H } { 2 ^ { k + 1 } } \times \frac { W } { 2 ^ { k + 1 } } \times ( 6 4 . 2 ^ { k - 1 } ) } . } \end{array}\tag{3.39}
$$

Furthermore, after the block-correspondent features $\mathcal { F } _ { \mathrm { R G B } } ^ { k } , \mathcal { F } _ { \mathrm { T h e r m a l } } ^ { k } , \mathcal { F } _ { \mathrm { D e p t h } } ^ { k }$ have been obtained, the thermal and surface normals modalities are projected using $\mathrm { c o n v } _ { 1 \times 1 }$ in order to match the expected richer dimensionality of ResNet50 [28] (as the auxiliary modalities are passed through ResNet18 [28] modules which have different channel sizes at equivalent hierarchical levels).

The purpose of this projection is not solely addressing shape mismatches, but also allowing the model to learn the shared latent space between all the modalities, and adapting the modalities to the expected patterns expected by the backbones, as the nature of the data distribution is different in the context of working different modalities.

$$
\begin{array} { r l r } & { \hat { \mathcal { F } } _ { \mathrm { T h e r m a l } } ^ { k } = p r o j _ { T } ^ { k } \left( \mathcal { F } _ { \mathrm { T h e r m a l } } ^ { k } \right) , } & { \hat { \mathcal { F } } _ { \mathrm { T h e r m a l } } ^ { k } \in \mathbb { R } ^ { \frac { H } { 2 ^ { k + 1 } } \times \frac { W } { 2 ^ { k + 1 } } \times ( 2 5 6 \cdot 2 ^ { k - 1 } ) } \cdot } \\ & { \hat { \mathcal { F } } _ { \mathrm { S u r f . ~ N o r m a l s } } ^ { k } = p r o j _ { S N } ^ { k } \left( \mathcal { F } _ { \mathrm { S u r f . ~ N o r m a l s } } ^ { k } \right) , } & { \hat { \mathcal { F } } _ { \mathrm { S u r f . ~ N o r m a l s } } ^ { k } \in \mathbb { R } ^ { \frac { H } { 2 ^ { k + 1 } } \times \frac { W } { 2 ^ { k + 1 } } \times ( 2 5 6 \cdot 2 ^ { k - 1 } ) } . } \end{array}\tag{3.40}
$$

Next, the extracted features enter the Fusion Encoder Block. This component is treated as a black box for now, with its detailed architecture explained in subsequent sections.

$$
\mathcal { F } _ { \mathrm { F u s e d } } ^ { k } = \mathrm { F u s i o n E n c } _ { k } \left( \mathcal { F } _ { \mathrm { R G B } } ^ { k } , \hat { \mathcal { F } } _ { \mathrm { T h e r m a l } } ^ { k } , \hat { \mathcal { F } } _ { \mathrm { S u r f . \mathrm { N o r m a l s } } } ^ { k } \right) .\tag{3.41}
$$

After the fused map has been computed and before continuing to the subsequent Fusion Encoder Block, the RGB future map $\mathcal { F } _ { \mathrm { R G B } } ^ { k }$ is updated using $\mathcal { F } _ { \mathrm { F u s e d } } ^ { k }$ , transmitting the current RGB information along with the obtained fusion map:

$$
\mathcal { F } _ { \mathrm { R G B } } ^ { k + 1 } = \mathcal { F } _ { \mathrm { F u s e d } } ^ { k } + \mathcal { F } _ { \mathrm { R G B } } ^ { k } .\tag{3.42}
$$

Decoding Phase. The fusion mechanism of the proposed Fusion Vision Transformer is performed during the encoding stage, while the decoding stage is dedicated to feature refinement and coherence enhancement. The decoding process is structured around a series of blocks that progressively refine the representations across three hierarchical spatial levels. At each level, skip connections from the corresponding encoding stages are concatenated with the outputs of the decoding blocks. This design facilitates stable gradient propagation and mitigates the information bottleneck that typically arises when integrating heterogeneous visual modalities characterized by distinct underlying probability distributions.

Considering the encoding stage ended, $\mathcal { F } _ { \mathrm { b o t t l e n e c k } }$ , the output of the last Fusion Encoder Block, is upsampled, concatenated with the correspondent skip connection from the encoding stage, and passed through the Decoder Block (Section 3.4.4). This process repeats three times. Given that $D$ is the depth used to project the fused map for the Segmentation Head (Section 3.4.5), the decoding process output can be represented as:

$$
\mathcal { F } _ { \mathrm { { D e c o d e d } } } = \mathrm { D e c o d e r } ( \mathcal { F } _ { \mathrm { { B o t t l e n e c k } } } ) , \mathcal { F } _ { \mathrm { { D e c o d e d } } } \in \mathbb { R } ^ { \frac { H } { 4 } \times \frac { W } { 4 } \times D } .\tag{3.43}
$$

The obtained feature map, $\mathcal { F } _ { \mathrm { D e c o d e d } }$ , is then upsampled using interpolation with a factor of 4 to obtain the final fusion feature map.

$$
\begin{array} { r } { \mathcal { F } _ { \mathrm { o u t } } = \mathrm { I n t e r p o l a t i o n } _ { \times 4 } ( \mathcal { F } _ { \mathrm { D e c o d e d } } ) , \mathcal { F } _ { \mathrm { o u t } } \in \mathbb { R } ^ { H \times W \times D } . } \end{array}\tag{3.44}
$$

## 3.4.1 Fusion Encoder Block

The Fusion Encoder Block represents the unitary component of the encoding stage of the Fusion Vision Transformer. Its role is to coherently combine the features originating from multiple modalities and to assess the unique information each one of them contains and how this information interacts with the other sets of specific data from the other modalities.

To achieve this, the Fusion Encoder Block employs 2 modules: the Fusion Self-Attention Module which combines and selects the most relevant features based on the current context and a FFN which further refines the fused maps. The modules interact through skip connection which aid the gradient flow and normalisation layers that provide data distribution unity across the samples.

The principal advantage of the proposed Fusion Encoder Block lies in its structural scalability to an arbitrary number of modalities, $k \in \mathbb N .$ . The only requirement is the designation of one modality as the primary modality, while the remaining k modalities are treated as auxiliary. Consequently, the functionality of the Fusion Encoder Block can be formally generalized at an abstract level to accommodate any number of modalities, and can be expressed as follows:

$$
\mathcal { F } _ { \mathrm { F u s e d } } = \mathcal { F } _ { \mathrm { R G B } } + \mathrm { L N } \Big ( \mathrm { F u s i o n \mathrm { \_ S A } } \Big ( \{ \mathcal { F } _ { m } \} _ { m = 1 } ^ { k } \Big ) \Big ) ,\tag{3.45}
$$

$$
\mathcal { F } _ { \mathrm { o u t } } = \mathcal { F } _ { \mathrm { F u s e d } } + \mathrm { L N } ( \mathrm { F F N } ( \mathcal { F } _ { \mathrm { F u s e d } } ) ) .\tag{3.46}
$$

## 3.4.2 Fusion Self-Attention Module

The main constituent of Vision Transformers is the self-attention module which captures relevant features from the images, highlighting the prevalent information from the visual space. However, its effectiveness represents a compromise in terms of computational complexity with the standard vision self attention yielding a complexity of $\mathcal { O } ( H ^ { 2 } W ^ { 2 } )$ , where H and W represent the height and the width of the image. To partially mitigate this problem, the Fusion Self-Attention Module adheres to the solution proposed by Restormer [53], where the attention scores produce a feature map in $\mathbb { R } ^ { C \times C }$ hence the complexity of the the self-attention module becoming O(HW C), where C represents the projection size of the Query (Q), Key (K), and Value (V ) components.

Besides computational complexity, most self-attention modules fail to integrate both local and global features from the image, treating the projection of the images as a unitary feature. The accuracy of the down-stream task on which VesperMUSE focuses on, image semantic segmentation, heavily relies on local components such as edges, isolated illumination patterns, and noise. The proposed Fusion Self-Attention Module aims to achieve a both global and local feature extraction, which allows the model to adapt the latent representations considering a wider range of information.

![](images/d061b1cf59af71618999923fcb4f163e4008bc07e4559fdcb55b188f59dd5084.jpg)  
Fig. 3.4. Visual representation of the VesperMUSE modality-agnostic attention mechanism.

Hence, considering $\mathcal { F } _ { m a i n } \in \mathbb { R } ^ { H \times W \times C }$ as the primary modality and $\{ F _ { m } \mid F _ { m } \in \mathbb { R } ^ { H \times W \times C }$ 2 $m =$ $\overline { { 1 , k } } \}$ the set of auxiliary modalities, the general formulation of the Fusion Self-Attention Module is:

$$
{ \mathcal { F } } _ { \mathrm { F u s e d } } = { \mathrm { F u s i o n . S A } } \Big ( { \mathcal { F } } _ { \operatorname* { m a i n } } , ~ \{ { \mathcal { F } } _ { m } ~ | ~ { \mathcal { F } } _ { m } \in \mathbb { R } ^ { H \times W \times C } , ~ m = \overline { { 1 , k } } \} \Big ) .\tag{3.47}
$$

The main modality, $\mathcal { F } _ { \mathrm { m a i n } }$ , serves as the source for both the Q and K representations which are obtained by projecting the main feature using two pointwise convolutional layers.

$$
\begin{array} { r } { Q = \mathrm { c o n v } _ { 1 \times 1 } ( \mathcal { F } _ { \mathrm { m a i n } } ) , } \\ { K = \mathrm { c o n v } _ { 1 \times 1 } ( \mathcal { F } _ { \mathrm { m a i n } } ) . } \end{array}\tag{3.48}
$$

The V component is obtained using both local and global information from all the auxiliary modalities. This approach facilitates better feature capturing and interaction. The local feature extraction for the auxiliary modalities collection is done as follows:

$$
\left\{ \operatorname* { m a x } _ { c } F _ { m } \biggm | F _ { m } \in \mathbb { R } ^ { H \times W \times C } , m = \overline { { 1 , k } } \right\} \longrightarrow \left\{ \hat { \mathcal { F } } _ { m } \biggm | \hat { \mathcal { F } } _ { m } \in \mathbb { R } ^ { H \times W \times 1 } , m = \overline { { 1 , k } } \right\} .\tag{3.49}
$$

Since the individual local representations are not useful alone in a fusion context, they are concatenated to obtain a unitary map which is then projected into a higher feature space.

$$
\begin{array} { r l r } & { \mathcal { F } _ { \mathrm { l o c a l } } = \operatorname { C o n c a t } _ { m = 1 } ^ { k } \left( \hat { \mathcal { F } } _ { m } \right) , } & { \mathcal { F } _ { \mathrm { l o c a l } } \in \mathbb { R } ^ { H \times W \times M } , } \\ & { } & \\ & { Z _ { \mathrm { l o c a l } } = \operatorname { c o n v } _ { 1 \times 1 } \left( \mathcal { F } _ { \mathrm { l o c a l } } \right) , } & { Z _ { \mathrm { l o c a l } } \in \mathbb { R } ^ { H \times W \times k C } . } \end{array}\tag{3.50}
$$

Following a similar approach, but with a different extraction technique, the global features are extracted using global pooling on the modality features.

$$
\left\{ \frac { 1 } { H W } \sum _ { h = 1 } ^ { H } \sum _ { w = 1 } ^ { W } \mathcal { F } _ { m } \middle | \mathcal { F } _ { m } \in \mathbb { R } ^ { H \times W \times C } , \ m = \overline { { 1 , k } } \right\} \ \longrightarrow \ \left\{ \hat { \mathcal { F } } _ { m } \middle | \hat { \mathcal { F } } _ { m } \in \mathbb { R } ^ { C } , \ m = \overline { { 1 , k } } \right\} .\tag{3.51}
$$

The global collection of features are concatenated as well and projected to a higher dimensionality:

$$
\begin{array} { r l r } & { \mathcal { F } _ { \mathrm { g l o b a l } } = \mathrm { C o n c a t } _ { m = 1 } ^ { k } \left( \hat { \mathcal { F } } _ { m } \right) , } & { \mathcal { F } _ { \mathrm { g l o b a l } } \in \mathbb { R } ^ { k \times C } , } \\ & { \mathcal { Z } _ { \mathrm { g l o b a l } } = \mathrm { c o n v } _ { 1 \times 1 } \left( \mathcal { F } _ { \mathrm { g l o b a l } } \right) , } & { \mathcal { Z } _ { \mathrm { g l o b a l } } \in \mathbb { R } ^ { k C \times C } . } \end{array}\tag{3.52}
$$

In order to combine the local and global features, the two projections are multiplied through matrix multiplication to obtain a contiguous auxiliary modality map which will represent the V component of the Fusion Self-Attention module:

$$
V = Z _ { \mathrm { l o c a l } } \ \cdot \ Z _ { \mathrm { g l o b a l } } , \qquad V \in \mathbb { R } ^ { H \times W \times C } .\tag{3.53}
$$

To reduce the computational complexity and allow better contextual interaction, the three components Q, K, V are split into h distinct heads.

$$
\begin{array} { r } { Q , K , V \longrightarrow \{ Q _ { i } , K _ { i } , V _ { i } \} _ { i = 1 } ^ { h } , Q _ { i } , K _ { i } , V _ { i } \in \mathbb { R } ^ { H \times W \times \frac { C } { h } } . } \end{array}\tag{3.54}
$$

Given that the components are computed and split, and $\tau$ as the temperature which controls how the attention scores are balanced, the self-attention formulation is defined as:

$$
{ \mathrm { A t t n } } _ { \mathrm { F u s e d } } = { \mathrm { s o f t m a x } } \left( { \frac { Q \cdot K ^ { \top } } { \mathcal { T } } } \right) \cdot V , \quad Q , K , V \in \mathbb { R } ^ { H \times W \times { \frac { c } { h } } } , ~ \mathcal { T } \in \mathbb { R } ^ { + } .\tag{3.55}
$$

To finalise the Fusion Self-Attention Module the final map is projected back to the original depth dimensionality using a pointwise convolution.

$$
\mathcal { F } _ { F u s e d } = \mathrm { c o n v } _ { 1 \times 1 } ( A t t n _ { \mathrm { F u s e d } } ) ,
$$

$$
A t t n _ { \mathrm { F u s e d } } \in \mathbb { R } ^ { H \times W \times C } .\tag{3.56}
$$

## 3.4.3 Feed Forward Network

The FFN used in the Fusion Vision Transformer is the same used in the LLIE module, with the formal description found in Section 3.3.3.

$$
\begin{array} { r } { \mathcal { F } _ { \mathrm { o u t } } = \mathrm { F F N } ( \mathcal { F } _ { F u s e d } ) . } \end{array}\tag{3.57}
$$

## 3.4.4 Decoder Block

The Decoder Block manages to achieve a unitary processing of the fused features and provides the final representations after iteratively processing previous outputs concatenated with skip connections from the encoding path.

In terms of the structure, it inherits the overall functionality of the Fusion Encoder Block, the only difference lying in the source of the Q, K, and V representations in its self-attention module and how the residual connections are formed in its structure.

Hence considering, $\mathcal { F } _ { i n } ~ \in ~ \mathbb { R } ^ { H \times W \times C }$ , as the input feature and $\mathcal { F } _ { o u t } \in \mathbb { R } ^ { H \times W \times C }$ as the resulting feature, following a plain ViT [27] structure, the underlying logic behind the Decoder Block can be formulated as:

$$
\mathcal { F } _ { i n } ^ { \prime } = \mathcal { F } _ { i n } + \mathrm { L N } \big ( \mathrm { A t t n } ( \mathcal { F } _ { i n } ) \big ) ,\tag{3.58}
$$

$$
\mathcal { F } _ { o u t } = \mathcal { F } _ { i n } ^ { \prime } + \mathrm { L N } ( \mathrm { F F N } ( \mathcal { F } _ { i n } ^ { \prime } ) ) .\tag{3.59}
$$

The Q, K, and V representations are all derived from the same source, namely $\mathcal { F } _ { \mathrm { i n } }$ , in contrast to the separate main and auxiliary modality sources employed in the Fusion Self-Attention module. The input feature is first projected into a higher-dimensional space and then partitioned into three components corresponding to the attention inputs Q, K, and V.

$$
\begin{array} { r } { Q , K , V = \mathrm { S p l i t } \big ( \mathrm { P r o j } ( \mathcal { F } _ { \mathrm { i n } } ) \big ) , \quad Q , K , V \in \mathbb { R } ^ { H \times W \times C } . } \end{array}\tag{3.60}
$$

## 3.4.5 Segmentation Head

To map the high-dimensional fused representations extracted by the Fusion Vision Transformer into the distinct probability logits necessary for each class, a CNN is utilized.

Within the VesperMUSE architecture, the Segmentation Head utilizes a CNN to map the extracted feature representations into class logits, employing Batch-Normalisation (BN) to stabilize the activations. To act as a stochastic regularizer, a Dropout layer is strategically inserted between the two convolutional layers. By randomly zeroing hidden units during training, this layer prevents feature co-adaptation and forces the network to learn robust, distributed representations. Consequently, it mitigates overfitting to high-dimensional noise within the dense feature maps and ensures that the final pixel-wise classification generalizes effectively across diverse spatial contexts.

Let cls denote the number of target semantic classes, $W _ { 1 }$ and $W _ { 2 }$ represent the weight matrices of the first and second convolutional layers respectively, and $\phi$ denote the Rectified Linear Unit (ReLU) activation function. Given an input feature map $\mathcal { F } _ { \mathrm { i n } }$ , the functionality of this Segmentation Head can thus be formalized as:

$$
\operatorname { S e g H e a d } ( X ; c l s ) = W _ { 2 } * \operatorname { D r o p } ( \phi ( \operatorname { B N } ( W _ { 1 } * X ) ) ) , \quad \operatorname { S e g H e a d } ( X ; c l s ) \in \mathbb { R } ^ { H \times W \times d s } .\tag{3.61}
$$

## 3.4.6 Fusion Vision Transformer Loss function

In the datasets VesperMUSE tests semantic segmentation on, there is both a significant structural and semantic difference between the classes (especially on the CART [25] dataset) and also a very heavy class imbalance. To ensure multi-modal feature alignment and good class representation a weighted hybrid loss function is employed which also acts on multiple levels allowing deep supervision.

Considering $\mathcal { L } _ { \mathrm { m a i n } } ( \cdot , \cdot )$ as the hybrid loss function, $\mathbf { P } ^ { ( 0 ) }$ as the final prediction, and $\mathbf { P } ^ { ( s ) }$ as the intermediate predictions originating from the decoding path and weighted by $\lambda _ { s } ,$ with $\mathbf { Y } ^ { ( 0 ) }$ denoting the ground-truth labels and $\mathbf { Y } ^ { ( s ) }$ representing the ground-truth labels resized to match the spatial resolution of the corresponding deep supervision outputs, the total loss function $\mathcal { L } _ { \mathrm { t o t a l } }$ is formulated as follows:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { h y b r i d } } \big ( \mathbf { P } ^ { ( 0 ) } , \mathbf { Y } \big ) + \sum _ { s \in \mathcal { S } } \lambda _ { s } \mathcal { L } _ { \mathrm { h y b r i d } \big ( \mathbf { P } ^ { ( s ) } , \mathbf { Y } ^ { ( s ) } \big ) } .\tag{3.62}
$$

Hybrid Loss. $\mathcal { L } _ { \mathrm { h y b r i d } } ( \cdot , \cdot )$ is a weighted combination of the multi-class Focal loss which operates at pixel level and is derived from the cross-entropy objective and the weighted Dice loss, which is region-based and directly optimizes the overlap between the predicted and ground-truth masks. Since the Dice loss is computed from class-wise statistics it is less sensitive to pixel-level imbalance and better captures structural and object-level coherence. The relative contribution of the constituents of the hybrid loss, $\mathcal { L } _ { \mathrm { h y b r i d } }$ , is balanced by the coefficients α and $\beta .$

$$
\mathcal { L } _ { \mathrm { h y b r i d } } = \alpha \mathcal { L } _ { \mathrm { F o c a l } } + \beta \mathcal { L } _ { \mathrm { D i c e } } .\tag{3.63}
$$

Multi-class Focal Loss. $\mathcal { L } _ { \mathrm { F o c a l } } ( \cdot , \cdot )$ is a pixel-wise classification loss designed to mitigate severe class imbalance. The strong pixel-density of the background class in both datasets can overwhelm the gradient signal and hinder learning of minority or hard classes. The Focal loss addresses this issue by introducing a modulating factor that reduces the importance of well-classified samples and emphasizes misclassified or uncertain predictions. Additionally, class-dependent weights are incorporated to further compensate for dataset imbalance.

Let $z _ { i , c }$ denote the logit corresponding to class c at pixel i, C the total number of classes, and $p _ { i , c }$ the softmax probability defined as:

$$
p _ { i , c } = \mathrm { s o f t m a x } ( \mathbf { z } _ { i } ) _ { c } = \frac { \exp ( z _ { i , c } ) } { \sum _ { k = 0 } ^ { C - 1 } \exp ( z _ { i , k } ) } .\tag{3.64}
$$

With $y _ { i } \in \{ 0 , \ldots , C - 1 \}$ being the ground-truth class label for pixel i, $w _ { y _ { i } }$ the corresponding class weight, γ the focusing parameter controlling the strength of down-weighting, and N the total number of pixels, the multi-class Focal loss is formulated as:

$$
\mathcal { L } _ { \mathrm { F o c a l } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( 1 - p _ { i , y _ { i } } ) ^ { \gamma } \left( - w _ { y _ { i } } \log ( p _ { i , y _ { i } } ) \right) .\tag{3.65}
$$

Weighted Dice Loss. $\mathcal { L } _ { \mathrm { D i c e } } ( \cdot , \cdot )$ is a region-based loss designed to directly optimize the overlap between the predicted segmentation and the ground-truth masks. Unlike pixel-wise classification losses, the Dice loss operates on global class-level statistics, making it inherently robust to class imbalance and particularly effective for small, sparse or weakly-represented objects. By incorporating class-dependent weights, the formulation further compensates for skewed class distributions while preserving structural consistency across object regions.

Let $p _ { i , c }$ denote the predicted softmax probability for class c at pixel i, $y _ { i , c }$ the one-hot encoding of the ground-truth label, N the total number of pixels, and ϵ a smoothing constant introduced for numerical stability. The Dice coefficient for class c is defined as:

$$
{ \mathrm { D i c e } } _ { c } = { \frac { 2 \sum _ { i = 1 } ^ { N } p _ { i , c } y _ { i , c } + \epsilon } { \sum _ { i = 1 } ^ { N } p _ { i , c } + \sum _ { i = 1 } ^ { N } y _ { i , c } + \epsilon } } .\tag{3.66}
$$

Let $w _ { c }$ denote the class weight associated with class $^ { c , }$ and $\delta$ a small constant preventing division by zero. Restricting normalization to classes present in the batch, the weighted Dice loss is expressed as:

$$
\mathcal { L } _ { \mathrm { { D i c e } } } = \frac { \sum _ { c = 0 } ^ { C - 1 } w _ { c } \left( 1 - \mathrm { D i c e } _ { c } \right) } { \sum _ { c = 0 } ^ { C - 1 } \mathbf { 1 } \left( \sum _ { i = 1 } ^ { N } y _ { i , c } > 0 \right) + \delta }\tag{3.67}
$$

## 3.4.7 Fusion Vision Transformer Training

The model is trained for 40K iterations for the PST900 dataset [8] and 80K iterations for the CART dataset [25] using the AdamW optimizer [71] implemented in PyTorch [72], with a base learning

rate of $3 \times 1 0 ^ { - 4 }$ , weight decay of 0.05, and momentum coefficients $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ . Gradient clipping with a maximum norm of 0.1 is applied to stabilize optimization.

Parameter-wise Learning Rates. Differential learning rates are employed for backbone and taskspecific components. The ResNet backbones [28] are optimized with a learning rate scaled by 0.1 relative to the base rate, while newly introduced fusion and prediction layers are trained with the full base learning rate. This strategy enables stable fine-tuning of pretrained representations while allowing faster convergence of randomly initialized layers.

Learning Rate Scheduling. A cosine annealing schedule is employed with $T _ { \mathrm { m a x } } = 1 0 K$ for the PST900 dataset [8] and $T _ { \mathrm { m a x } } = 2 0 K$ for CART [25]. $T _ { \mathrm { m a x } }$ controls the duration (in iterations) over which the learning rate gradually decreases from the initial value $\eta _ { 0 }$ to the minimum value $\eta _ { \mathrm { m i n } }$ . The minimum learning rate is $\eta _ { \mathrm { m i n } } = 1 \times 1 0 ^ { - 7 }$ . A short warm-up phase is used. Given that $\eta _ { 0 }$ denotes the initial learning rate the learning rate at iteration t is defined as:

$$
\eta _ { t } = \eta _ { \mathrm { m i n } } + \frac { 1 } { 2 } ( \eta _ { 0 } - \eta _ { \mathrm { m i n } } ) \left( 1 + \cos \left( \frac { \pi t } { T _ { \mathrm { m a x } } } \right) \right) .\tag{3.68}
$$

Cosine annealing is preferred because it provides a smooth, non-monotonic decay of the learning rate, allowing large updates in early stages for rapid convergence, followed by progressively smaller updates that improve stability and fine-grained parameter refinement. Compared to step decay, cosine annealing avoids abrupt learning rate drops, which can destabilize training.

For both semantic segmentation datasets the training is conducted on randomly-cropped patches.   
For PST900 [8] the dimension used is 512x512, while on CART [25] it is 480x400.

![](images/6adde0a231b9c143a0625e7dfe68e70b0b27da4d41cdd71a42d3932cd32c6656.jpg)

Experiments and Results

The evaluations for VesperMUSE across all tasks is conducted on two main datasets: PST900 [8] and CART [25]. The low-light adapted images and later enhanced are conducted on the same splits as used for the semantic segmentation evaluation.

PST900. The PST900 benchmark [8] consists of 894 synchronized RGB-Thermal image pairs acquired within demanding subterranean settings. The dataset features pixel-level semantic annotations for five distinct object categories: background, hand drill, backpack, fire extinguisher, and survivor. All modalities maintain a native resolution of 1280 × 720 pixels. For experimental consistency, the data is partitioned into training and evaluation subsets following a 2 : 1 distribution ratio.

CART. The CART dataset [25] comprises 2,282 co-registered RGB-Thermal pairs captured across heterogeneous outdoor environments, including aquatic regions (rivers, lakes, coastlines), arid deserts, and dense forestation. Semantic labels are provided for a comprehensive 10-class taxonomy: bare ground, rocky terrain, developed structures, road, shrubs, trees, sky, water, vehicles, and person. The imagery is provided at a resolution of 960 × 640 pixels. The corpus is stochastically divided into training, validation, and testing sets using a 6 : 1 : 1 ratio, respectively.

Empirical contributions. The results achieved by VesperMUSE across all levels and tasks can be summarised as:

• Robust Synthetic Low-Light Adaptation: The synthetic adaptation of the PST900 [8] and CART [25] datasets achieves a superior SNR of approximately 10.7 dB and a physical Gamma distribution of 1.00. These metrics surpass established synthetic low-light adaptation pipelines such as the one used for creating LOL-v2 synthetic [33] in terms of modeling low-light environments, ensuring that the low-light data maintains realistic luminance statistics and physicsinformed consequences of light-deficiency.

• Efficient and High-Performance LLIE: The VesperMUSE LLIE component demonstrates significant performance improvements over existing methods, achieving a 29.5% PSNR increase over LYT-Net [47] and a 2.3% increase over Retinexformer [50] on the syntheticallyadapted CART dataset [25]. Notably, the model achieves performance comparable to the stateof-the-art ISALux [51] while utilizing only 0.454M parameters, representing a 75.4% reduction in model complexity.

• State-of-the-Art Segmentation on CART: On the CART dataset [25], VesperMUSE establishes a new performance ceiling with an mIoU of 0.7672, outperforming the top-performing

CM-SSM [73] (0.7460) by 2.8% and CMX [9] (0.7400) by 3.7%. Substantial class-wise gains are observed in challenging categories, including Vehicles (0.6314) and Person (0.3426), validating the effectiveness of thermal and structure-aware multi-modal fusion in outdoor environments.

• Superior Generalization on PST900: Evaluation on the PST900 dataset [8] yields a competitive mIoU of 0.8559, representing a 1.4% improvement over LASNet [65] and a 3.1% increase over MDNet [74]. The method sets new accuracy benchmarks for task-critical objects, outperforming CM-SSM [73] by 2.9% for Hand-Drill (0.8282) and by 2.4% for Fire-Ext. (0.8776), highlighting the framework’s robustness across indoor dark scenarios.

## 4.1 Synthetic Low-Light Image Adaptation

The synthetic low-light adaptation pipeline described in Section 3.2.1 must demonstrate robustness and reliability in accurately emulating real low-light conditions across all key factors that influence such environments, including illumination levels (both linear and non-linear/gamma-based), noise characteristics, inter-pixel relationships, and global statistical properties such as mean and standard deviation computed over the entire dataset.

To enable a rigorous quantitative comparison with existing datasets, encompassing both real and synthetic low-light imagery, a diverse set of evaluation metrics is employed. This comprehensive analysis ensures a thorough assessment of the effects induced by light deficiency and the fidelity of the proposed synthetic generation process. The proposed metrics are as follows:

Signal-to-Noise Ratio (SNR). The goal is to quantify the level of corruption introduced by noise during the low-light generation process, while remaining invariant to global intensity scaling between the reference and degraded images. This is important because realistic low-light images are fundamentally characterized by photon-limited noise, and matching the real-world SNR distribution is critical for modeling the effect of low-light condition on the overall composition of the image. In the following formulation, N denotes the reference normal-light image, L the low-light image, $\mu _ { N }$ and $\mu _ { L }$ their mean intensities, α a scaling factor used to align global intensity levels, $\sigma _ { N }$ the standard deviation of the noise residual, and Ω the image domain.

$$
\mathrm { S N R } = 2 0 \log _ { 1 0 } \left( \frac { \mu _ { L } } { \sigma _ { N } } \right) , \quad \sigma _ { N } = \sqrt { \frac { 1 } { \left| \Omega \right| } \sum _ { i \in \Omega } \left( L _ { i } - \alpha N _ { i } \right) ^ { 2 } } , \quad \alpha = \frac { \mu _ { L } } { \mu _ { N } } .\tag{4.1}
$$

Normalized Mutual Information (NMI). The objective is to measure how much structural information is preserved between the reference and the low-light image. Realistic degradation should alter illumination while maintaining scene structure, avoiding excessive distortion or artificial patterns. Given $N$ and L denote the reference and degraded images, $p ( x , y )$ the joint distribution of their pixel intensities, $p ( x )$ and $p ( y )$ the marginal distributions, and $H ( \cdot )$ the entropy function, NMI can be formally formulated as:

$$
\mathrm { N M I } = \frac { 2 I ( N , L ) } { H ( N ) + H ( L ) } , \quad I ( N , L ) = \sum _ { x , y } p ( x , y ) \log \left( \frac { p ( x , y ) } { p ( x ) p ( y ) } \right) .\tag{4.2}
$$

Exposure Estimation. The aim is to capture the global illumination drop between the normal-light image and its low-light counterpart through the ratio of their average intensities. As the real lowlight conditions are primarily governed by reduced exposure, and matching this factor ensures physically consistent brightness attenuation. This metric is represented as a ration between $\mu _ { N }$ and $\mu _ { L }$ representing the mean intensities of the reference and low-light images, respectively.

$$
E = { \frac { \mu _ { L } } { \mu _ { N } } } .\tag{4.3}
$$

Gamma Estimation. The purpose is to estimate the non-linear transformation applied to intensities during the low-light formation process by modeling the relationship between the reference and degraded image in logarithmic space. Given that $N _ { i }$ and $L _ { i }$ denote pixel intensities from the reference and low-light images, E is the estimated exposure factor, ϵ is a small numerical constant for stability, and γ represents the non-linearity coefficient, the non-linear (gamma) estimation can be expressed as.

$$
\gamma = \operatorname { m e d i a n } \left( { \frac { \log { L _ { i } } } { \log ( N _ { i } \cdot E + \epsilon ) } } \right) .\tag{4.4}
$$

Mean Luminance. To characterize the overall brightness level of the synthesized dataset pixel intensities across the image domain are averaged. This is important because it provides a direct measure of how dark the generated images are, and offers a good insight in the environment conditions. In the following formulation, $Y _ { i }$ denotes the luminance value of pixel i, and Ω represents the full image domain.

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Quality Metrics</td><td colspan="3">Image Statistics</td><td colspan="3">Luminance (Y)</td></tr><tr><td>SNR (dB) ↑</td><td>NMI↑</td><td>Exp. ↓</td><td>Gamma ≈ 1</td><td>CV (%) ↑</td><td>Min / Max</td><td>Mean</td><td>Std</td></tr><tr><td>LOL v1 [32]</td><td>9.59</td><td>0.46</td><td>0.13</td><td>1.03</td><td>69.54</td><td>2.33 /82.58</td><td>15.49</td><td>10.77</td></tr><tr><td>LOL v2 real [33]</td><td>8.51</td><td>0.34</td><td>0.15</td><td>1.02</td><td>69.34</td><td>0.34 / 82.58</td><td>16.08</td><td>11.16</td></tr><tr><td>LOL v2 synthetic [33]</td><td>4.73</td><td>0.60</td><td>0.46</td><td>1.38</td><td>56.97</td><td>8.26 / 186.97</td><td>61.90</td><td>35.27</td></tr><tr><td>LOL Blur [35]</td><td>8.70</td><td>0.47</td><td>0.28</td><td>1.07</td><td>48.98</td><td>4.48 / 70.34</td><td>27.23</td><td>13.34</td></tr><tr><td>PST900 LL</td><td>10.71</td><td>0.31</td><td>0.29</td><td>1.00</td><td>51.96</td><td>3.38 /29.59</td><td>12.86</td><td>6.68</td></tr><tr><td>CART LL</td><td>10.73</td><td>0.29</td><td>0.16</td><td>1.00</td><td>54.40</td><td>3.38 /36.73</td><td>12.71</td><td>6.91</td></tr></table>

Table 4.1. Quantitative comparison between real and low-light datasets across multiple statistical and physical metrics. The PST900 [8]/CART [25] synthetic table entries are paired images comprised from original RGB images from the two datasets and synthetically adapted low-light images.

$$
\mu _ { Y } = { \frac { 1 } { | \Omega | } } \sum _ { i \in \Omega } Y _ { i } .\tag{4.5}
$$

Luminance Standard Deviation. To quantify the spread of luminance values within an image, capturing intra-image contrast and variability is paramount. Realistic low-light scenes exhibit diverse illumination patterns rather than uniform darkness. To formally express this, $Y _ { i }$ represents the luminance value at pixel i, $\mu _ { Y }$ is the mean luminance, and Ω denotes the image domain.

$$
\sigma _ { Y } = \sqrt { \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } ( Y _ { i } - \mu _ { Y } ) ^ { 2 } } .\tag{4.6}
$$

Coefficient of Variation. The aim is to normalize luminance variability with respect to the mean brightness, providing a scale-invariant measure of illumination diversity across the dataset, and reflecting how well the synthetic data captures the variability present in real-world lighting conditions. In the following formulation, $\sigma _ { Y }$ denotes the luminance standard deviation and $\mu _ { Y }$ the mean luminance.

$$
\mathrm { C V } = { \frac { \sigma _ { Y } } { \mu _ { Y } } } \times 1 0 0 .\tag{4.7}
$$

## 4.1.1 Quantitative Analysis

The purpose of VesperMUSE’s proposed synthetic adaptation algorithm was to represent the synthetic low-light images as close to the real-world low-light imaging present in LOL-v1 [32], LOLv2-real [33] and LOL Blur [35] as well as improve on other methods which proposed synthetic adaptation methods such as LOL-v2-synthetic [33].

![](images/6888726e3465d6fa813953ffed91f96c3d481912c58bc966f163736c9413a774.jpg)  
Fig. 4.1. Illumination vs. signal integrity. Real low-light data retain high SNR, while prior synthetic methods introduce excessive noise and over-brightening. VesperMUSE preserves an accurate signal quality similar to that observable in true low-light conditions.

Illumination and Intensity. To accurately assess the perceptual quality and light profile from the synthetically adapted images, it is required to interpret the absolute intensity, the non-linearity in the captured illumination as well as uniformity of the light distribution. The Mean Luminance assesses the physical brightness in the scene which is observed to be low in the real world datasets LOLv1 [32] (15.49) and LOL-v2-real [33] (16.08), but almost 4 times higher in LOL-v2-synthetic [33] (61.90). These values are strongly linked with the exposure which acts as a linear scalar multiplier for the incoming irradiance. The datasets obtained with the proposed method manage to share more similarity with the real-captured datasets in both mean luminance and exposure.

Moreover, a critical flaw in earlier synthetic approaches, is the introduction of artificial non-linearity values, with Gamma being on average approximately 26% higher than the values observed in realworld scenarios. In contrast, the proposed pipeline demonstrates a perfectly linear mapping, much closer to other low-light imaging datasets conditions.

Noise. A well-known problem of light-deficient scenes is the presence of noise due to the reduced frequency of photons captured by the sensor. Therefore, an accurate noise modeling must always be linked with the level of illumination. The level of alignment with the real world datasets as well as the coherence of this correlation between the level of light and noise can be visualised in Figure 4.1.

![](images/145b1866abbf4714d6eea934d7ee71ec587e4ec952268c3cc099e338c34a60fa.jpg)

Fig. 4.2. PST900 [8] synthetic low-light adapted RGB and the original image.  
![](images/a51ba79320e6d6494a45074baf42552e9dff6ed7ee00a332b284b8724cca6db3.jpg)  
Fig. 4.3. CART [25] synthetic low-light adapted RGB and the original image.

Information preservation. Both local and global structural information is also lost when comparing a low-light image and its well-lit counterpart. The absence of light naturally introduces colour distortions and more weak edges which ultimately lead to overall information loss which is measured using the NMI. The two proposed datasets present the most important challenge in terms of the amount of information that needs to be recovered through the LLIE process, mainly when compared with the LOL-v2-synthetic [33] where the shared information is much higher between the low-light and normal light images (0.60 compared with 0.31 and 0.29 respectively).

## 4.1.2 Qualitative Analysis

In terms of perceptual image quality, VesperMUSE generates a low-light counterpart that degrades the input in accordance with characteristic low-light conditions. Notably, colour distortions and structured noise patterns are clearly observable in the synthetically generated samples (Figures 4.2, 4.3). Furthermore, the illumination has spatially and statistically a high variability, similar to the real-world environmental conditions.

## 4.2 Low-Light Image Enhancement

Evaluation Metrics. To quantitatively evaluate the fidelity of the reconstructed images, PSNR is the main utilized measure. This metric is defined by the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. In this formulation, M and N represent the image dimensions, $I ( i , j )$ and $K ( i , j )$ denote the pixel intensities of the ground truth and reconstructed images at coordinates $( i , j )$ , and $M A X _ { I }$ signifies the maximum possible pixel value. The PSNR is expressed as:

$$
\mathrm { P S N R } = 1 0 \cdot \log _ { 1 0 } \left( \frac { M A X _ { I } ^ { 2 } } { \frac { 1 } { M N } \sum _ { i = 0 } ^ { M - 1 } \sum _ { j = 0 } ^ { N - 1 } [ I ( i , j ) - K ( i , j ) ] ^ { 2 } } \right) .\tag{4.8}
$$

Complementary to the pixel-level error captured by PSNR, Structural Similarity Index Measure (SSIM) [75] is also employed to better account for perceived visual quality and structural integrity. While PSNR relies on absolute differences, SSIM evaluates the similarity between the ground truth and the output by comparing luminance, contrast, and structural information across two images x and y. In the following expression, $\mu _ { x }$ and $\mu _ { y }$ represent the local means, $\sigma _ { x } ^ { 2 }$ and $\sigma _ { y } ^ { 2 }$ denote the local variances, $\sigma _ { x y }$ signifies the cross-covariance, and $c _ { 1 }$ and $c _ { 2 }$ are constants incorporated to ensure numerical stability, the formal representation of SSIM can be expressed as:

$$
\mathrm { S S I M } ( x , y ) = \frac { ( 2 \mu _ { x } \mu _ { y } + c _ { 1 } ) ( 2 \sigma _ { x y } + c _ { 2 } ) } { ( \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + c _ { 1 } ) ( \sigma _ { x } ^ { 2 } + \sigma _ { y } ^ { 2 } + c _ { 2 } ) } .\tag{4.9}
$$

## 4.2.1 Quantitative Analysis

<table><tr><td rowspan="2">Method</td><td colspan="2">PST900 LL</td><td colspan="2">CART LL</td><td rowspan="2">Params (M) ↓</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR↑</td><td>SSIM ↑</td></tr><tr><td>LYT-Net [47]</td><td>22.81</td><td>0.860</td><td>21.42</td><td>0.832</td><td>0.045</td></tr><tr><td>Retinexformer [50]</td><td>26.60</td><td>0.850</td><td>27.11</td><td>0.847</td><td>1.61</td></tr><tr><td>ISALux [51]</td><td>27.58</td><td>0.874</td><td>27.81</td><td>0.873</td><td>1.85</td></tr><tr><td>VesperMUSE LLIE</td><td>27.47</td><td>0.870</td><td>27.74</td><td>0.858</td><td>0.454</td></tr></table>

Table 4.2. Quantitative comparison of LLIE methods on the synthetically low-light adapted PST900 LL and CART LL datasets. Best results are highlighted in bold.

The quantitative results in Table 4.2 highlight several important points about the behaviour of thermalassisted enhancement. An observation that needs to be made is that VesperMUSE LLIE is natively thermal-aware, while the other methods only aim to predict the well-lit images from the low-light maps only. ISALux [51] achieves the highest overall performance across both datasets, reaching 27.58 dB/0.874 (PSNR/SSIM) on PST900 LL and 27.81 dB/0.873 on CART LL, indicating strong reconstruction fidelity under purely RGB-based enhancement. However, VesperMUSE LLIE demonstrates highly competitive performance, attaining 27.47 dB/0.870 on PST900 LL and 27.74 dB/0.858 on CART LL, while maintaining a substantially lower parameter count of 0.454M compared to 1.85M. This corresponds to approximately a 4× reduction in model size with only marginal degradation in PSNR (≤ 0.11 dB) and a limited drop in SSIM, particularly on PST900 LL.

Compared to Retinexformer [50], VesperMUSE LLIE consistently improves PSNR on both datasets (+0.87 dB on PST900 LL and +0.63 dB on CART LL), while using approximately 3.5× fewer parameters, highlighting a significantly better performance-to-complexity trade-off. Although Retinexformer [50] exhibits slightly higher SSIM on CART LL, but the gap remains small, suggesting that structural preservation is largely maintained by the proposed method.

Notably, the strong performance of VesperMUSE LLIE can be attributed to its thermal-aware design, which introduces complementary modality information that is particularly beneficial under challenging low-light conditions where RGB signals alone are not always stable and informative. This is especially evident on PST900 LL, where the gap to the top-performing method is minimal despite architectural differences and reduced model capacity. In contrast, LYT-Net [47], while extremely lightweight (0.045M parameters), suffers from a significant drop in both PSNR and SSIM, indicating that aggressive parameter reduction without multi-modal guidance limits restoration quality.

## 4.2.2 Qualitative Analysis

The qualitative results further demonstrate the strong enhancement capability of VesperMUSE LLIE, particularly in approximating the RGB reference from severely degraded inputs. The proposed method produces outputs with improved local contrast and more accurate colour saturation compared to Retinexformer [50] and LYT-Net [47], which tend to generate flatter or slightly desaturated images. Notably, chromatic fidelity is well preserved, as observed in the PST900 LL samples where objects such as the yellow helmet and red fire extinguisher are reconstructed with more natural and vivid colours.

The benefits of the thermal-aware design are also evident in the preservation of structural details. In regions where the low-light RGB input lacks visible information, VesperMUSE LLIE effectively leverages thermal cues to enhance edge sharpness and suppress noise, leading to clearer object boundaries and more stable textures. While ISALux [51] achieves comparable visual quality, the proposed method delivers competitive results with a significantly lower computational footprint, avoiding common artifacts such as halos or over-smoothing.

PST900 LL  
CART LL  
![](images/e7f3791471bc9d19d4ca78dc9361de846f30fc723d85319a1c97c961a7ee3204.jpg)  
Fig. 4.4. Qualitative comparison of VesperMUSE (VM) LLIE with SOTA methods enhancing the synthetically low-light adapted PST900 LL and CART LL datasets.

## 4.3 Semantic Segmentation

Evaluation Metrics. To quantitatively evaluate the efficacy of the proposed semantic segmentation framework, we employ the mIoU as the primary performance metric. Given that N denotes the total number of semantic classes, and T P, F P, and F N represent the true positives, false positives, and false negatives for class i, respectively, the mIoU can be formalised as:

$$
\mathrm { m I o U } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { T P _ { i } } { T P _ { i } + F P _ { i } + F N _ { i } } .\tag{4.10}
$$

## 4.3.1 Quantitative Analysis

PST900 Evaluation. While VesperMUSE achieves a highly competitive mIoU of 0.8559 (trailing the leading CM-SSM [73] by a marginal 0.0031), a closer examination of class-wise distributions reveals the distinct advantage of its tri-modal input architecture. Despite employing a comparatively standard ResNet50/18 [28] backbone ensemble, the model establishes new state-of-the-art thresholds for geometrically rigid, contextually isolated objects. It notably outperforms robust contemporary baselines, including SA-Gate [76] and CM-SSM [73], on the Hand-Drill (0.8282) and Fire-Extinguisher (0.8776) classes by significant margins. This suggests that the inclusion of surface normal maps act as an informative structural guide, allowing the network to delineate sharp object boundaries even against methods utilizing substantially heavier feature extractors (e.g., LASNet’s ResNet-152 [28]). Furthermore, the VesperMUSE LLIE configuration demonstrates remarkable robustness, achieving a strong 0.7818 mIoU that remains highly competitive with established baselines such as EGFNet [77]. Therefore, it is shown that the pre-enhancement process aids semantic understanding through noise removal and colour shift correction (two essential components of low-light information recovery) while simultaneously producing visually appealing, illuminated feeds.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="5">IoU per Class (%) ↑</td><td rowspan="2">mIoU (%) ↑</td></tr><tr><td>Background</td><td>Hand-Drill</td><td>Backpack</td><td>Fire-Ext.</td><td>Survivor</td></tr><tr><td>EGFNet [77]</td><td></td><td>99.26</td><td>64.67</td><td>83.05</td><td>71.29</td><td>74.30</td><td>78.51</td></tr><tr><td>SA-Gate [76]</td><td></td><td>99.25</td><td>81.01</td><td>79.77</td><td>72.97</td><td>62.22</td><td>79.05</td></tr><tr><td>MMSMCNet23 [78]</td><td>Mit-B3</td><td>–</td><td>62.40</td><td>89.20</td><td>73.30</td><td>74.70</td><td>79.80</td></tr><tr><td>CLNet-T24 [79]</td><td>Mit-B4</td><td></td><td>68.50</td><td>87.90</td><td>72.70</td><td>75.70</td><td>80.80</td></tr><tr><td>DBCNet23 [5]</td><td>ResNet101</td><td></td><td>77.20</td><td>82.70</td><td>73.00</td><td>76.70</td><td>81.80</td></tr><tr><td>C4Net-Mit25 [80]</td><td>Mit-B2</td><td></td><td>69.40</td><td>88.40</td><td>79.60</td><td>79.60</td><td>82.60</td></tr><tr><td>SGFNet23 [81]</td><td>ResNet152</td><td></td><td>76.70</td><td>85.40</td><td>75.60</td><td>76.70</td><td>82.80</td></tr><tr><td> $\mathrm { M D N e t _ { 2 5 } }$  [74]</td><td>Mit-B2</td><td></td><td>69.50</td><td>89.40</td><td>76.80</td><td>80.20</td><td>83.00</td></tr><tr><td>LASNet [65]</td><td>ResNet-152</td><td>99.46</td><td>77.75</td><td>86.48</td><td>82.80</td><td>83.43</td><td>84.40</td></tr><tr><td>CM-SSM [73]</td><td>EfficientVit-B1</td><td></td><td>80.50</td><td>85.40</td><td>85.70</td><td>78.10</td><td>85.90</td></tr><tr><td>VesperMUSE LLIE</td><td>ResNet50/ResNet18</td><td>99.00</td><td>75.73</td><td>56.89</td><td>82.08</td><td>77.22</td><td>78.18</td></tr><tr><td>VesperMUSE</td><td>ResNet50/ResNet18</td><td>99.46</td><td>82.82</td><td>83.36</td><td>87.76</td><td>74.56</td><td>85.59</td></tr></table>

Table 4.3. Performance comparison with SOTA methods on the PST900 [8] dataset. Best results are highlighted in bold.

CART Evaluation. VesperMUSE establishes a new state-of-the-art on the operationally complex CART dataset [25], achieving an mIoU of 0.7672 and decisively outperforming previous leading methods such as CM-SSM [73] (0.7460) and CMX [9] (0.7400). In contrast to PST900 [8], CART [25] necessitates pronounced resilience to unstructured outdoor environments and highly variable terrain boundaries. The architecture’s efficacy on ground-level classifications (e.g., Bare Grd. at

0.8706 compared to CM-SSM’s [73] 0.8260, and Road at 0.6538 over CRM’s [82] 0.6160) underscores the robust spatial synergy between thermal features and surface normals. This cross-modal integration effectively compensates for the severe degradation of visual texture typical of low-light outdoor scenes. Critically, VesperMUSE exhibits superior semantic preservation of dynamic obstacles, surpassing recent baselines like MCNet-T [83] on the highly challenging Person class (0.3426 vs. 0.3150) and providing a new best mIoU on Vehicles class with 0.6314. Conversely, while the VesperMUSE LLIE variant encounters challenges with fine-grained dynamic obstacles (most notably its inability to delineate the Person class (0.0000)), this explicit LLIE configuration offers two distinct operational advantages. As opposed to the fine-features constructed classes, the enhancement process successfully reconstructs macro-level environmental features, maintaining robust segmentation accuracy on foundational terrain classes such as Water (0.8996) and Rocky Terr. (0.7891).

<table><tr><td rowspan="2">Model</td><td rowspan="2">Backbone</td><td colspan="10">IoU per Class (%) ↑</td><td rowspan="2">mIoU (%) ↑</td></tr><tr><td>Bare Grd.</td><td>Rocky Terr.</td><td>Dev. Struct.</td><td>Road</td><td>Shrubs</td><td>Trees</td><td>Sky</td><td>Water</td><td>Vehicles</td><td>Person</td></tr><tr><td>GMNet21 [84]</td><td>ResNet152</td><td>81.50</td><td>88.80</td><td>83.80</td><td>60.30</td><td>70.80</td><td>79.50</td><td>92.00</td><td>97.40</td><td>54.80</td><td>17.80</td><td>72.70</td></tr><tr><td>EGFNet22 [77]</td><td>ResNet101</td><td>80.30</td><td>85.20</td><td>56.30</td><td>60.50</td><td>69.90</td><td>78.40</td><td>92.80</td><td>97.40</td><td>57.20</td><td>17.10</td><td>69.50</td></tr><tr><td>EAEFNet_23 [11]</td><td>ResNet152</td><td>80.50</td><td>86.30</td><td>81.30</td><td>54.50</td><td>70.20</td><td>80.20</td><td>93.70</td><td>97.60</td><td>48.10</td><td>0.00</td><td>69.20</td></tr><tr><td>CMNext23 [85]</td><td>Mit-B2</td><td>80.10</td><td>87.80</td><td>84.10</td><td>58.60</td><td>69.90</td><td>78.50</td><td>92.50</td><td>97.30</td><td>54.30</td><td>17.80</td><td>72.10</td></tr><tr><td>CMX23 [9]</td><td>Mit-B2</td><td>82.40</td><td>90.30</td><td>85.00</td><td>60.40</td><td>71.00</td><td>79.90</td><td>93.60</td><td>97.60</td><td>57.90</td><td>22.20</td><td>74.00</td></tr><tr><td>CRM24 [82]</td><td>Swin-B</td><td>81.60</td><td>90.00</td><td>85.10</td><td>61.60</td><td>69.50</td><td>77.90</td><td>92.30</td><td>97.40</td><td>37.70</td><td>20.60</td><td>71.40</td></tr><tr><td>CLNet-T24 [79]</td><td>Mit-B4</td><td>81.80</td><td>89.30</td><td>84.80</td><td>61.50</td><td>71.30</td><td>79.50</td><td>93.40</td><td>97.50</td><td>52.30</td><td>25.10</td><td>73.70</td></tr><tr><td>CAINet24 [86]</td><td>MobileNetV2</td><td>80.20</td><td>88.40</td><td>78.40</td><td>57.10</td><td>68.80</td><td>77.00</td><td>92.10</td><td>96.90</td><td>43.90</td><td>17.20</td><td>70.00</td></tr><tr><td>MCNet-T25 [83]</td><td>ConvNeXtV2-B</td><td>80.50</td><td>86.90</td><td>83.70</td><td>58.00</td><td>69.70</td><td>78.00</td><td>92.90</td><td>97.70</td><td>58.10</td><td>31.50</td><td>73.70</td></tr><tr><td>CM-SSM [73]</td><td>EfficientVit-B1</td><td>82.60</td><td>89.70</td><td>84.90</td><td>61.00</td><td>71.30</td><td>79.10</td><td>92.90</td><td>97.60</td><td>62.00</td><td>24.50</td><td>74.60</td></tr><tr><td>VesperMUSE LLIE</td><td>ResNet50/ResNet18</td><td>75.44</td><td>78.91</td><td>65.90</td><td>48.25</td><td>57.87</td><td>65.43</td><td>76.69</td><td>89.96</td><td>37.20</td><td>0.00</td><td>59.57</td></tr><tr><td>VesperMUSE</td><td>ResNet50/ResNet18</td><td>87.06</td><td>90.77</td><td>86.72</td><td>65.38</td><td>72.57</td><td>77.98</td><td>91.50</td><td>97.78</td><td>63.14</td><td>34.26</td><td>76.72</td></tr></table>

Table 4.4. Performance comparison with SOTA methods on the CART dataset [25]. Best results are highlighted in bold.

## 4.3.2 Qualitative Analysis

PST900 Evaluation. VesperMUSE’s strong quantitative results in underground scenarios are reinforced by visual comparisons against HAFNet [87], CRM-RGBTSeg [82], CAINet [86], and EAEFNet [11]. First, the standard VesperMUSE configuration consistently maintains crisp, accurate predictions for primary targets, such as humans and equipment. By effectively fusing thermal and RGB data, VesperMUSE isolates heat-emitting objects even in near-total darkness, avoiding the boundary fragmentation seen in baselines like CAINet [86].

![](images/6a34ee0277c64051b3989c232141fa0b7e75c0170e38b2baa694401096e08582.jpg)

![](images/e943ee248e552d8bd58eb00191c92690dc2a19ee68bf532ca98b0173d8a6f846.jpg)  
Fig. 4.5. Illustration of the per-class surface distribution on the PST900 [8] (left) and CART [25] (right) datasets. The percentage for each class is computed as the mean percentage of pixels covered across the samples where it appears (sample count is encoded as n).

Furthermore, the VesperMUSE LLIE configuration demonstrates a distinct advantage in pitch-black environments. While the noise is still a very serious problem and some classes with objects covering a smaller surface get misclassified because of the low-level feature shift caused by low-light environments, the LLIE configuration manages to recover essential hidden structural details even when surrounded by heavy noise (the Survivor instance on the last column of Figure 4.6).

CART Evaluation. The strong quantitative performance on CART [25] is supported by Vesper-MUSE’s visually comparison against methods like SFAF-MA [88], CMNext [85], SGFNet [81], CM-SSM [73]. A first observation would be that the proposed method manages to continuously maintain a prediction for larger objects and not predict other classes inside of them (this behaviour can be noticed for SFAF-MA [88] and CMNext [85]).

Moreover, the last column of Figure 4.7 shows another advantage of VesperMUSE on three out of the 4 compared models. In environments where multiple classes intertwine, all methods besides VesperMUSE and CMNext [85] seem to get confused on which is the dominant class in the problematic regions, as the Trees class is often erroneously conflated with Rocky Terrain or Bare Ground. By taking a look at the qualitative results for the VesperMUSE LLIE configuration, it can be easily concluded that the performance comes from predominant classes with very large objects like: Bare Ground, Rocky Terrain, Water, and that correctly segmenting smaller classes for the majority of the samples still remains a challenge.

![](images/64828f83fe95980b030da98ff513f9a6fa49b6e13f407b36e00fa3e967d452f9.jpg)  
Fig. 4.6. Qualitative comparison of VesperMUSE (VM) and the LLIE configuration (VM LLIE) with SOTA models on the PST900 dataset [8].

![](images/d2a0b3a1ee9aec96de611329d25ad703d9a35714cf31f8436c09dee621174fd9.jpg)  
Fig. 4.7. Qualitative comparison of VesperMUSE (VM) and the LLIE configuration (VM LLIE) with SOTA models on the CART dataset.

## 4.4 Computational Performance

Although the multi-modal architecture of VesperMUSE inherently incurs a higher computational cost in terms of floating-point operations (FLOPs), it maintains a competitive parameter footprint relative to its performance capabilities. Specifically, VesperMUSE establishes a new SOTA on the CART dataset [25] with an mIoU of 0.767, outperforming the previous leading method (CM-SSM [73]) by a significant margin of 2.1%. Furthermore, when compared to other larger architectures such as MCNet-T [83], VesperMUSE delivers a 3.0% absolute improvement in mIoU while utilizing approximately 19.3 million fewer parameters (180.45M vs. 199.79M). The primary sources of the increased computational demand are: the tri-input framework, which necessitates an additional backbone and a supplementary pathway through the encoder, and the integration of the DepthAnything V2 [30] model for monocular depth estimation, which alone accounts for 43 GFLOPs. Ultimately, this computational compromise represents a deliberate architectural trade-off through the explicit modeling of depth which provides critical geometric priors that justify the higher FLOPs count.

<table><tr><td>Model</td><td>Backbone</td><td>FLOPs (G) ↓</td><td>Params (M) ↓ mIoU (%) ↑</td><td></td></tr><tr><td> $\mathrm { S G F N e t _ { 2 3 } \ [ 8 1 ] }$ </td><td>ResNet152</td><td>143.73</td><td>125.25</td><td>68.5</td></tr><tr><td> $\mathrm { C M N e X t _ { 2 3 } \ [ 8 5 ] }$ </td><td>Mit-B2</td><td>68.70</td><td>58.68</td><td>72.1</td></tr><tr><td> $\mathrm { C M X _ { 2 3 } \ [ 9 ] }$ </td><td>Mit-B2</td><td>67.20</td><td>66.57</td><td>74.0</td></tr><tr><td> $\mathrm { C L N e t  – T _ { 2 4 } } \left[ 7 9 \right]$ </td><td>Mit-B4</td><td>217.85</td><td>130.84</td><td>73.7</td></tr><tr><td> $\mathrm { C A I N e t _ { 2 4 } \ [ 8 6 ] }$ </td><td>MobileNetV2</td><td>123.62</td><td>12.16</td><td>70.0</td></tr><tr><td> $\mathbf { M C N e t – T _ { 2 5 } } \left[ 8 3 \right]$ </td><td>ConvNeXt-B</td><td>278.85</td><td>199.79</td><td>73.7</td></tr><tr><td> $\mathrm { C M - S S M } [ 7 3 ]$ </td><td>EfficientVit-B1</td><td>10.34</td><td>12.59</td><td>74.6</td></tr><tr><td>VesperMUSE</td><td>ResNet50/ ResNet18</td><td>321.28</td><td>180.45</td><td>76.7</td></tr></table>

Table 4.5. Complexity comparison with an image resolution of $4 8 0 \times 6 4 0$ pixels. mIoU results are reported on the CART dataset [25].

## 4.5 Ablation Studies

## 4.5.1 The effect of LLIE on Semantic Segmentation performance

The LLIE process conducted before the segmentation in the low-light configuration of VesperMUSE is essential for recovering lost detail and removing the noise. In this section the effect of the enhancement will be thoroughly studied to determine what improvement it brings in the semantic segmentation process and if some classes can benefit more from LLIE.

Figures 4.8 and 4.10 show that the RGB configuration of VesperMUSE dominates in terms of performance which is absolutely expected. The main focus point of this study is to see if under a tri-modal input setting with no prior enhancement for the RGB, the performance is decreasing.

For the PST900 dataset [8] the overall dataset mIoU improvement is strongly influenced by the large performance differences on the Fire-Ext. and Survivor classes, while the Hand-Drill and Backpack classes show a slight performance decline. The differences, however, have a considerable offset in the magnitude. While for the Hand-Drill and Backpack classes it can be considered just a model or training particularity, it is not the same case for Fire-Ext. and Survivor. A good candidate explanation for this would be the common features shared by the two groups of classes. Survivor and Fire-Ext are represented using brighter colours and a larger surface covered by their instances, while Backpack and Hand-Drill often have less contrastive colours and definitely a smaller instance surface.

Conversely, on the CART dataset [25], Figure 4.11 shows a different distribution. Here, the model configuration where the input setting with low-light images which are not enhanced as RGB inputs, provides worse performance for all the classes. The natural light present in CART [25] which is distributed more evenly, opposed to the artificial light and the cave environment of PST900 [8], manages to provide a better colour balance which is preserved across the synthetic low-light adaptation and enhancement process throughout. The only issue that persists even after the enhancement process has been applied is the absence of the Person class among the predictions which can be again attributed, as in the PST900 [8] case, to the small surface covered by noise across its instances.

![](images/3b5102a9acfd77309fc3b2c3c54824ba09901d79f388a78fd63e389150277948.jpg)  
Fig. 4.8. Semantic segmentation performance between different input configuration using: RGB (VM), LLIE RGB (VM LLIE), and low-light RGB without any enhancement (VM LL) on the PST900 dataset [8].

![](images/9a977e34a115212dfd1b16ee71d3d1e08f7d23045b04e92077a7f6a23062c27d.jpg)  
Fig. 4.9. Semantic segmentation performance deltas between VesperMUSE LLIE and VesperMUSE trained and tested on low-light images, highlighting the role of LLIE enhancement on the PST900 dataset [8].

![](images/817dbfdbe97109fd860fc76d5db9d3b5b318cf1a41bb2f01357c1f3471909003.jpg)  
Fig. 4.10. Semantic segmentation performance between different input configuration using: RGB (VM), LLIE RGB (VM LLIE), and low-light RGB without any enhancement (VM LL) on the CART dataset [25].

![](images/54cd1a4435d531ccc6ea0f95775570396f973132a89ea8154540aa30518a25f4.jpg)  
Fig. 4.11. Semantic segmentation performance deltas between VesperMUSE LLIE and VesperMUSE trained and tested on low-light images, highlighting the role of LLIE enhancement on the CART dataset [25].

## 4.5.2 Modality choices and their rank

The inputs of VesperMUSE can be configured in a large number of ways. Although the standard and recommended configuration is composed of three inputs: RGB, thermal, and surface normals, the model can be run using any number of modalities as inputs with very little changes as the Fusion Self-Attention was specifically designed to structurally support any number of modalities.

The only constraint, however, is that one modality has to be declared as the main modality while the others are auxiliaries. The main modality is used as the main source of information in the encoding stage and also has a larger backbone responsible for extracting the features from it. In all the configurations presented until now, the main modality was the RGB (normal or synthetically low-light adapted and then enhanced), but the purpose of this ablation study is to show how the model behaves under different number of inputs and what are the effects of modifying the rank of the modalities.

Table 4.6 presents different input configurations and provides evidence for why RGB was chosen as a primary modality even when running under a LLIE configuration and the enhanced images present noise and colour distortions, while the thermal is left unaffected. Furthermore, LLIE RGB is shown to also validate the primary modality advantage when interchangeably switched with the surface normal representations.

Thermal and Surface Normals as Auxiliaries. A first observation, as shown in Figure 1.1, is that both modalities bring an important contribution if compared to the segmentation network using

only the LLIE RGB as input. Their impact is very similar when individually employed, with an average of $\approx 3 \%$ performance deficit when the surface normal is not used and an average of $\approx 3 . 5 \%$ performance deficit considering both datasets. The combined benefit is observed mainly on the PST900 [8] dataset with an average improvement of $\approx 5 \%$ for both modalities, confirming again the fact that the CART dataset [25] provides more informativeness in its RGB maps than PST900 [8] because of the different light conditions.

The relevance of the RGB modality is again confirmed when comparing the two configurations where LLIE RGB and surface normal representations are switched so that each one is alternatively a main modality and an auxiliary. The results on both datasets show that there is an average of ≈ 7.0% performance deficit when the surface normal map is used as a main modality.

Thermal as Main Modality. Given that this ablation study is conducted under low-light configuration, it would be expected for the thermal modality provide more information about the overall composition of the image, as it is not affected by noise, exposure and loss of global and local structures. However, a configuration with thermal used as a main modality, with surface normals, reflectance and illumination as auxiliaries shows a large performance decline compared to the other configurations where LLIE RGB is used as the main modality.

The reflectance and illuminance maps represent a decomposition of the LLIE RGB image conducted under the assumption of the Retinex Theory [40] which suggests that visual information can be separated into two core components: reflectance responsible for the colour information and illuminance encoding the intensity of the pixels.

The decomposition can be done using various ways, from classical to neural networks which are trained specifically for accurate representation of both components. However, for the purpose of this experiment, a simpler approach was used, where the illumination is considered as the maximum value across the channels and reflectance as the original image divided by the illumination, both filtered using, a Gaussian Kernel.

$$
L = G _ { \sigma } * \left( \operatorname* { m a x } _ { c \in \{ R , G , B \} } \mathcal { Z } _ { c } + \epsilon \right) , \qquad R = \frac { I } { G _ { \sigma } * \left( \operatorname* { m a x } _ { c \in \{ R , G , B \} } \mathcal { Z } _ { c } + \epsilon \right) } .\tag{4.11}
$$

The quantitative results from Table 4.6 confirm that the thermal modality cannot be successfully employed as a main modality, implying that colour information, even if it is shifted by the low-light adaptation and enhancement, provides more important features for the performance of semantic segmentation than the clean thermal imaging.

<table><tr><td rowspan="2">Primary Modality</td><td rowspan="2">Auxiliary Modalities</td><td colspan="2">mIoU (%) ↑</td></tr><tr><td>PST900</td><td>CART</td></tr><tr><td rowspan="3">LLIE RGB</td><td>Thermal, Surface Norm.</td><td>78.18</td><td>59.57</td></tr><tr><td>Thermal</td><td>74.53</td><td>58.56</td></tr><tr><td>Surface Norm.</td><td>73.80</td><td>58.62</td></tr><tr><td>Thermal</td><td>Surface Norm., Ref., Illum.</td><td>60.60</td><td>50.76</td></tr><tr><td>Surface Norm.</td><td>LLIE RGB</td><td>67.70</td><td>55.90</td></tr></table>

Table 4.6. Table analyzing the impact of different primary and auxiliary input modalities on the PST900 [8] and CART [25] datasets. Grouping by the primary backbone highlights the performance delta when injecting various auxiliary streams.

## 4.5.3 Architectural motivation

Impact of the Fusion Strategy. To motivate the design of the proposed Fusion Encoder Block, it is compared against a baseline strategy that relies on naive concatenation. In this baseline, multi-modal features are simply concatenated along the channel dimension prior to passing through a standard Transformer block. As shown in Table 4.7, the proposed cross-attention fusion approach yields a substantial performance improvement, outperforming the concatenation baseline by ≈ 4% mIoU.

This performance gap highlights the limitations of early concatenation. While concatenation aggregates multi-modal data into a unified tensor, it forces the subsequent layers to implicitly learn the cross-modal alignments from scratch without any structured guidance. In contrast, the proposed fusion mechanism explicitly models the spatial and semantic correlations between the primary and auxiliary modalities. By utilizing cross-attention, the network dynamically queries the auxiliary streams to enhance the primary features, ensuring that the model selectively attends to relevant multi-modal cues rather than treating all concatenated channels with equal initial weight.

Replacing Decoders with Convolutions. As detailed in Section 3.4.4, the core feature refinement component within the VesperMUSE decoder is a lightweight, attention-based Transformer block. This module is specifically designed to refine the multi-scale fused features, utilizing self-attention to dynamically weigh and prioritize spatial information that is most relevant to the current semantic context. To validate this design choice, an ablation study was conducted where these Transformer blocks were replaced with standard pointwise convolutions, effectively removing the global contextaggregation capabilities of the attention mechanism while maintaining a similar channel dimensionality.

Table 4.7, shows that reverting to a purely convolutional decoder results in a measurable performance degradation, dropping the mIoU from 0.7818 to 0.7351 on PST900 [8], and from 0.5957 to 0.5320 on CART [25]. While the convolutional substitution slightly reduces the total parameter count (from 180M to 179M), it increases the computational overhead (from 321G to 326G FLOPs). More importantly, the drop in segmentation accuracy confirms that the spatially-aware feature refinement provided by the Transformer block is crucial. The self-attention mechanism allows the network to effectively resolve local ambiguities by leveraging long-range dependencies, a capability that simple pointwise convolutions lack.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Params (M) ↓</td><td rowspan="2">FLOPs (G) ↓</td><td colspan="2">mIoU (%) ↑</td></tr><tr><td>PST900</td><td>CART</td></tr><tr><td>VesperMUSE LLIE</td><td>180</td><td>321</td><td>78.18</td><td>59.57</td></tr><tr><td>Concatenation-based Fusion</td><td>192</td><td>321</td><td>75.42</td><td>54.44</td></tr><tr><td>Convolution Decoder</td><td>179</td><td>326</td><td>73.51</td><td>53.20</td></tr></table>

Table 4.7. Ablation study analyzing the impact of different fusion and decoder strategies on the PST900 [8] and CART [25] datasets, including parameter counts and computational complexity.

## 5

Conclusions

![](images/7f1535ca202ecc3d6198a794c7d49c4390f10b472b6b087dbaccd530c126658c.jpg)  
Fig. 5.1. Qualitative maps highlighting the noise and excessive degradation observed in few synthetically low-light adapted images.

## 5.1 Known Limitations

Synthetic Low-Light Adaptation Algorithm Strong Degradation. In the current configuration, the main issue that is persistent within the low-light synthetic pipeline is that a small sample of the low-light samples are excessively degraded so that the LLIE module cannot recover lost information. The source of this issue remains unknown, however it is suspected that the issue comes from increasing the variability of illumination in the low-light conditions. As a result, a very few number of samples from the final light distribution are affected by this issue. Figure 5.1 presents a part of the affected samples with some managing to capture the overall structure of the image keeping some of the objects observable while others represent a totally noise and artifacts-covered representation which cannot be correctly evaluated by the segmentation module.

A solution to this problem would be either removing the affected samples or further tuning the synthetic low-light adaptation algorithm to diagnose problematic scenarios and increase the illumination threshold while keeping the noise levels lower than for other samples.

LLIE module Output Colour Imbalance. A comparative analysis of the global intensity distributions (Figure 5.2) reveals that the LLIE module introduces luminance-dependent chromatic aberrations via non-linear channel uncoupling. Specifically, in the shadow and mid-tone regimes $( 0 ~ \leq ~ I ~ \leq ~ 1 5 0 )$ , the model disproportionately attenuates the blue channel relative to the source distribution, inducing an unnatural warm colour cast. Conversely, in the extreme highlight regime $( I  2 5 5 )$ , the output exhibits anomalous green-channel clipping, with saturation peaks significantly exceeding both the original image limits and concurrent RGB channels. This divergent colour distortion fundamentally disrupts the intrinsic RGB ratios across the dynamic range, highlighting a deficiency in colour constancy and necessitating the integration of cross-channel coupled tone mapping.

Since the current loss function configuration does not have a colour loss integrated, this would strongly benefit the reduction of channel-based colour discrepancies between the well-lit images and the predicted maps from the LLIE module. In this case, the loss function would receive a new term, namely a colour-focused loss.

Standard spatial losses penalize the necessary brightness enhancements in low-light processing. To decouple intensity from colour, RGB channels for each pixel are treated as a 3D vector. By computing the cosine similarity between the input vector $( \mathbf { v } _ { i } ^ { i n } )$ and output vector $( \mathbf { v } _ { i } ^ { o u t } )$ across all N pixels, the colour loss $( \mathcal { L } _ { c o l o u r } )$ selectively penalizes chromatic angular deviation while ignoring magnitude shifts, preserving intrinsic RGB ratios.

$$
\mathcal { L } _ { c o l o u r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( 1 - \frac { \mathbf { v } _ { i } ^ { i n } \cdot \mathbf { v } _ { i } ^ { o u t } } { \| \mathbf { v } _ { i } ^ { i n } \| _ { 2 } \| \mathbf { v } _ { i } ^ { o u t } \| _ { 2 } } \right) .\tag{5.1}
$$

Another solution would be the processing of the input in a different colour space. Operating within an orthogonal colour space like YCbCr inherently avoids cross-channel distortion. An initial transformation (Γ) separates the signal, allowing the enhancement function to target exclusively the input luminance. Because the original chrominance channels $( C _ { b , i n } , C _ { r , i n } )$ remain strictly untouched, projecting the composite signal back via the inverse mapping $( \Gamma ^ { - 1 } )$ generates enhanced RGB outputs $( R _ { o u t } , G _ { o u t } , B _ { o u t } )$ free from luminance-driven colour artifacts.

$$
\begin{array} { r } { \Bigg [ R _ { o u t } \Bigg ] } \\ { \Bigg [ G _ { o u t } \Bigg ] = \Gamma ^ { - 1 } \left( \begin{array} { c } { \Bigg [ \mathcal { F } ( Y _ { i n } ) \Bigg ] } \\ { \Bigg . } \\ { \Bigg . } \\ { \Bigg . } \\ { \Bigg . } \end{array} \right) . } \end{array}\tag{5.2}
$$

Fusion Vision Transformer Computational Requirements. Currently, as the focus of this work was not directly placed on computational complexity, the computational requirements needed to perform training and inference are high. An analysis of the comparative performance plot reveals distinct trade-offs between model accuracy and computational complexity. VesperMUSE demonstrates the highest performance among the evaluated models, achieving a peak mIoU of 0.7670 on the CART dataset [25]. However, this superior accuracy incurs a higher computational cost relative to the baselines. Conversely, CM-SSM [73] emerges as a highly efficient alternative, delivering competitive accuracy $\mathrm { ( m I o U \approx 0 . 7 4 6 0 ) }$ while maintaining a significantly lower computational footprint. Overall, while VesperMUSE establishes a new performance ceiling and strictly dominates heavier baselines like MCNet-T [83], models such as CM-SSM [73] provide an optimal balance for resource-constrained applications.

![](images/0c871c36173e8179d15712cabd2a258135d42eb87a79bb10988bff21e89101bb.jpg)

![](images/05f77f374de61b6a521739b06981f60ec6c99f00059ac3ecf385b4d73d276547.jpg)  
Fig. 5.2. Histogram comparison between the two pairs of RGB and LLIE RGB maps showing the colour balance.

![](images/c9b53c9da71cd28ec2e23d65fe450143bcdb4c53e0e52961c9448c7b84991e0e.jpg)  
Fig. 5.3. Visual representation of the ratio between mIoU and the parameter count (M) of SOTA models.

Besides architectural changes such as backbone change, lowering the projections sizes in the decoder path of the Fusion Vision Transformer or even modality-based pruning (only if two or more features have a high feature similarity), other inference methods like mixed precision (fp16), quantisation (by reducing the precision of the weights to a smaller container like int8/int16) can be taken into account for both the LLIE and semantic segmentation modules.

## 5.2 Future Work

The subsequent development of VesperMUSE should primarily target both performance enhancement and the reduction of computational complexity. At present, further improvements in performance can be facilitated by adopting more efficient backbone architectures, potentially surpassing those derived from the ResNet family. Additionally, the utilization of modality-specific backbones for thermal and structural inputs represents a promising direction, as these are expected to outperform the current convolutional adaptation strategy employed within the model.

Furthermore, it is worthwhile to investigate alternative representations of image information, such as operating in different colour spaces or explicitly decomposing the input into low- and high-frequency components within the LLIE module. Such approaches may enable a more principled treatment of light irradiation, which has been shown to be an effective strategy in LLIE tasks.

## 5.3 Final Thoughts

The proposed framework is intended to establish a foundation for future advancements in semantic segmentation under challenging and highly variable illumination conditions. Despite recent progress, the field continues to be constrained by the limited availability of diverse and representative datasets, which significantly hinders the development of robust and generalizable methods.

The scenarios explored in this study reflect realistic and often critical environments, including safetysensitive missions where reliability is paramount. In such contexts, the deployment of models that are not only performant but also transparent with respect to their limitations becomes essential. This is particularly relevant for applications such as survivor identification, where the consequences of failure are profound. Therefore, advancing DL-based approaches in this domain is not solely a matter of improving accuracy, but also of fostering trust, robustness, and accountability in systems designed to operate under uncertainty.

## Bibliography

[1] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 3431–3440.

[2] V. Badrinarayanan, A. Kendall, and R. Cipolla, “Segnet: A deep convolutional encoderdecoder architecture for image segmentation,” IEEE transactions on pattern analysis and machine intelligence, vol. 39, no. 12, pp. 2481–2495, 2017.

[3] Z.-Q. Zhao, P. Zheng, S.-t. Xu, and X. Wu, “Object detection with deep learning: A review,” IEEE transactions on neural networks and learning systems, vol. 30, no. 11, pp. 3212–3232, 2019.

[4] Q. Ha, K. Watanabe, T. Karasawa, Y. Ushiku, and T. Harada, “Mfnet: Towards real-time semantic segmentation for autonomous vehicles with multi-spectral scenes,” in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017, pp. 5108– 5115.

[5] W. Zhou, T. Gong, J. Lei, and L. Yu, “Dbcnet: Dynamic bilateral cross-fusion network for rgb-t urban scene understanding in intelligent vehicles,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 53, no. 12, pp. 7631–7641, 2023.

[6] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computerassisted intervention. Springer, 2015, pp. 234–241.

[7] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus, “Indoor segmentation and support inference from rgbd images,” in European conference on computer vision. Springer, 2012, pp. 746–760.

[8] S. S. Shivakumar, N. Rodrigues, A. Zhou, I. D. Miller, V. Kumar, and C. J. Taylor, “Pst900: Rgb-thermal calibration, dataset and segmentation network,” in 2020 IEEE international conference on robotics and automation (ICRA). IEEE, 2020, pp. 9441–9447.

[9] J. Zhang, H. Liu, K. Yang, X. Hu, R. Liu, and R. Stiefelhagen, “Cmx: Cross-modal fusion for rgb-x semantic segmentation with transformers,” IEEE Transactions on intelligent transportation systems, vol. 24, no. 12, pp. 14 679–14 694, 2023.

[10] Y. Sun, W. Zuo, and M. Liu, “Rtfnet: Rgb-thermal fusion network for semantic segmentation of urban scenes,” IEEE Robotics and Automation Letters, vol. 4, no. 3, pp. 2576–2583, 2019.

[11] M. Liang, J. Hu, C. Bao, H. Feng, F. Deng, and T. L. Lam, “Explicit attention-enhanced fusion for rgb-thermal perception tasks,” IEEE Robotics and Automation Letters, vol. 8, no. 7, pp. 4060–4067, 2023.

[12] J. Liu, D. Xu, W. Yang, M. Fan, and H. Huang, “Benchmarking low-light image enhancement and beyond,” International Journal of Computer Vision, vol. 129, no. 4, pp. 1153–1184, 2021.

[13] C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, and R. Cong, “Zero-reference deep curve estimation for low-light image enhancement,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 1780–1789.

[14] Y. Zhang, J. Zhang, and X. Guo, “Kindling the darkness: A practical low-light image enhancer,” in Proceedings of the 27th ACM International Conference on Multimedia. ACM, Oct. 2019, pp. 1632–1640. [Online]. Available: http://dx.doi.org/10.1145/3343031.3350926

[15] Y. Jiang, X. Gong, D. Liu, Y. Cheng, C. Fang, X. Shen, J. Yang, P. Zhou, and Z. Wang, “Enlightengan: Deep light enhancement without paired supervision,” IEEE Transactions on Image Processing, vol. 30, pp. 2340–2349, 2021.

[16] X. Wu, Z. Wu, H. Guo, L. Ju, and S. Wang, “Dannet: A one-stage domain adaptation network for unsupervised nighttime semantic segmentation,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Jun. 2021, pp. 15 764–15 773. [Online]. Available: http://dx.doi.org/10.1109/cvpr46437.2021.01551

[17] P. Wang, Y. Zhuang, X. Jia, C. Lyu, and Z. Zheng, “Low-light image enhancement via masked frequency-spatial network,” SSRN preprint, 2025. [Online]. Available: http://dx.doi.org/10.2139/ssrn.5216705

[18] G. Li, Y. Wang, Z. Liu, X. Zhang, and D. Zeng, “Rgb-t semantic segmentation with location, activation, and sharpening,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 3, pp. 1223–1235, 2022.

[19] C. Zhang, Z. Wan, S. Stepputtis, K. Sycara, and Y. Xie, “Spectral-aware global fusion for rgbthermal semantic segmentation,” in 2025 IEEE International Conference on Image Processing (ICIP). IEEE, 2025, pp. 43–48.

[20] M. Kristo, M. Ivasic-Kos, and M. Pobar, “Thermal object detection in difficult weather conditions using yolo,” IEEE Access, vol. 8, pp. 125 459–125 476, 2020. [Online]. Available: http://dx.doi.org/10.1109/access.2020.3007481

[21] S. Hwang, J. Park, N. Kim, Y. Choi, and I. S. Kweon, “Multispectral pedestrian detection: Benchmark dataset and baseline,” in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Jun. 2015, pp. 1037–1045. [Online]. Available: http://dx.doi.org/10.1109/cvpr.2015.7298706

[22] M. Vollmer and K.-P. Mollmann, ¨ Infrared Thermal Imaging: Fundamentals, Research and Applications. Wiley, Nov. 2017. [Online]. Available: http://dx.doi.org/10.1002/9783527693306

[23] A. Gonzalez, Z. Fang, Y. Socarras, J. Serrat, D. V ´ azquez, J. Xu, and A. L ´ opez, “Pedestrian ´ detection at day/night time with visible and fir cameras: A comparison,” Sensors, vol. 16, no. 6, p. 820, Jun. 2016. [Online]. Available: http://dx.doi.org/10.3390/s16060820

[24] Y. Sun, W. Zuo, P. Yun, H. Wang, and M. Liu, “Fuseseg: Semantic segmentation of urban scenes based on rgb and thermal data fusion,” IEEE Transactions on Automation Science and Engineering, vol. 18, no. 3, pp. 1000–1011, Jul. 2021. [Online]. Available: http://dx.doi.org/10.1109/tase.2020.2993143

[25] C. Lee, M. Anderson, N. Raganathan, X. Zuo, K. Do, G. Gkioxari, and S.-J. Chung, “Cart: Caltech aerial rgb-thermal dataset in the wild,” 2024.

[26] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Medical image computing and computer-assisted intervention, 2015.

[27] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[28] B. Koonce, “Resnet 50,” in Convolutional neural networks with swiftfor tensorflow: image recognition and dataset categorization. Springer, 2021, pp. 63–72.

[29] T. Raj, F. Hanim Hashim, A. Baseri Huddin, M. F. Ibrahim, and A. Hussain, “A survey on lidar scanning mechanisms,” Electronics, vol. 9, no. 5, p. 741, 2020.

[30] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” Advances in Neural Information Processing Systems, vol. 37, pp. 21 875–21 911, 2024.

[31] C. Chen, Q. Chen, M. N. Do, and V. Koltun, “Seeing motion in the dark,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[32] C. Wei, W. Wang, W. Yang, and J. Liu, “Deep retinex decomposition for low-light enhancement,” arXiv preprint arXiv:1808.04560, 2018.

[33] W. Yang, W. Wang, H. Huang, S. Wang, and J. Liu, “Sparse gradient regularized deep retinex network for robust low-light image enhancement,” IEEE Transactions on Image Processing, vol. 30, pp. 2072–2086, 2021.

[34] R. Wang, X. Xu, C.-W. Fu, J. Lu, B. Yu, and J. Jia, “Seeing dynamic scene in the dark: A high-quality video dataset with mechatronic alignment,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[35] S. Zhou, C. Li, and C. C. Loy, “Lednet: joint low-light enhancement and deblurring in the dark,” Lecture Notes in Computer Science, pp. 573–589, 2022.

[36] W. Wang, X. Wu, X. Yuan, and Z. Gao, “An experiment-based review of low-light image enhancement methods,” IEEE Access, vol. 8, pp. 87 884–87 917, 2020. [Online]. Available: http://dx.doi.org/10.1109/access.2020.2992749

[37] S. Kansal, S. Purwar, and R. K. Tripathi, “Image contrast enhancement using unsharp masking and histogram equalization,” Multimedia Tools and Applications, vol. 77, no. 20, pp. 26 919–26 938, Mar. 2018. [Online]. Available: http://dx.doi.org/10.1007/s11042-018-5894-8 [38] L. Xiao, C. Li, Z. Wu, and T. Wang, “An enhancement method for x-ray image via fuzzy noise removal and homomorphic filtering,” Neurocomputing, vol. 195, pp. 56–64, Jun. 2016. [Online]. Available: http://dx.doi.org/10.1016/j.neucom.2015.08.113

[39] S. E. Kim, J. J. Jeon, and I. K. Eom, “Image contrast enhancement using entropy scaling in wavelet domain,” Signal Processing, vol. 127, pp. 1–11, Oct. 2016. [Online]. Available: http://dx.doi.org/10.1016/j.sigpro.2016.02.016

[40] E. H. Land, “The retinex theory of color vision,” Scientific american, vol. 237, no. 6, pp. 108–129, 1977.

[41] C. Wei, W. Wang, W. Yang, and J. Liu, “Deep retinex decomposition for low-light enhancement,” in Proceedings of the British Machine Vision Conference (BMVC), 2018.

[42] W. Wu, J. Weng, P. Zhang, X. Wang, W. Yang, and J. Jiang, “Uretinex-net: Retinex-based deep unfolding network for low-light image enhancement,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[43] X. Yi, H. Xu, H. Zhang, L. Tang, and J. Ma, “Diff-retinex: Rethinking low-light image enhancement with a generative diffusion model,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[44] Y. Zhang, Y. Tian, Y. Kong, B. Zhong, and Y. Fu, “Residual dense network for image restoration,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 7, pp. 2480–2495, Jul. 2021. [Online]. Available: http://dx.doi.org/10.1109/tpami.2020.2968521

[45] R. Wang, Q. Zhang, C.-W. Fu, X. Shen, W.-S. Zheng, and J. Jia, “Underexposed photo enhancement using deep illumination estimation,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Jun. 2019, pp. 6842–6850. [Online]. Available: http://dx.doi.org/10.1109/cvpr.2019.00701

[46] K. G. Lore, A. Akintayo, and S. Sarkar, “Llnet: A deep autoencoder approach to natural low-light image enhancement,” Pattern Recognition, vol. 61, pp. 650–662, 2017.

[47] A. Brateanu, R. Balmez, A. Avram, and C. Orhei, “Lyt-net: Lightweight yuv transformer-based network for low-light image enhancement,” 2024. [Online]. Available: https://arxiv.org/abs/2401.15204

[48] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[49] Z. Wang, X. Cun, J. Bao, W. Zhou, J. Liu, and H. Li, “Uformer: A general u-shaped transformer for image restoration,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[50] Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, and Y. Zhang, “Retinexformer: One-stage retinexbased transformer for low-light image enhancement,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[51] R. Balmez, A. Brateanu, C. Orhei, C. O. Ancuti, and C. Ancuti, “Isalux: Illumination and semantics-aware transformer employing mixture of experts for low light image enhancement,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2026, pp. 7862–7872.

[52] A. Brateanu, R. Balmez, C. Orhei, C. Ancuti, and C. Ancuti, “Modalformer: Multimodal transformer for low-light image enhancement,” arXiv preprint arXiv:2507.20388, 2025.

[53] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, “Restormer: Efficient transformer for high-resolution image restoration,” in IEEE/CVF Conf. on Computer Vision and Pattern Recog. (CVPR), 2022.

[54] R. Balmez, A. Brateanu, C. Orhei, C. O. Ancuti, and C. Ancuti, “Depthlux: Employing depthwise separable convolutions for low-light image enhancement,” Sensors, vol. 25, no. 5, p. 1530, 2025.

[55] S. Kumar Panda and P. K. Sa, “A thermal-aware multisensor framework for low-light image enhancement,” IEEE Sensors Journal, vol. 25, no. 17, pp. 33 831–33 839, Sep. 2025. [Online]. Available: http://dx.doi.org/10.1109/jsen.2025.3590815

[56] R. Jha, A. Lenka, M. Ramanagopal, A. Sankaranarayanan, and K. Mitra, “Rt-x net: Rgb-thermal cross attention network for low-light image enhancement,” in 2025 IEEE International Conference on Image Processing (ICIP). IEEE, Sep. 2025, pp. 1492–1497. [Online]. Available: http://dx.doi.org/10.1109/icip55913.2025.11084466

[57] S. Minaee, Y. Y. Boykov, F. Porikli, A. J. Plaza, N. Kehtarnavaz, and D. Terzopoulos, “Image segmentation using deep learning: A survey,” IEEE Transactions on Pattern Analysis and

Machine Intelligence, pp. 1–1, 2021. [Online]. Available: http://dx.doi.org/10.1109/tpami.2021. 3059968

[58] J. Hu, G. Zhao, S. You, and C. C. J. Kuo, “Evaluation of multimodal semantic segmentation using rgb-d data,” in Artificial Intelligence and Machine Learningfor Multi-Domain Operations Applications III, T. Pham, L. Solomon, and M. E. Hohil, Eds. SPIE, Apr. 2021, p. 47. [Online]. Available: http://dx.doi.org/10.1117/12.2587991

[59] X. Zhang, J. Li, Z. Li, H. Liu, M. Zhou, L. Wang, and Z. Zou, Overview of Data Fusion in Autonomous Driving Perception. Springer Nature Singapore, 2023, pp. 19–34. [Online]. Available: http://dx.doi.org/10.1007/978-981-99-3280-1 2

[60] J. Cao, H. Leng, D. Lischinski, D. Cohen-Or, C. Tu, and Y. Li, “Shapeconv: Shape-aware convolutional layer for indoor rgb-d semantic segmentation,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Oct. 2021, pp. 7068–7077. [Online]. Available: http://dx.doi.org/10.1109/iccv48922.2021.00700

[61] C. Hazirbas, L. Ma, C. Domokos, and D. Cremers, “Fusenet: Incorporating depth into semantic segmentation via fusion-based cnn architecture,” in Asian conference on computer vision. Springer, 2016, pp. 213–228.

[62] X. Chen, K.-Y. Lin, J. Wang, W. Wu, C. Qian, H. Li, and G. Zeng, Bi-directional Cross-Modality Feature Propagation with Separation-and-Aggregation Gate for RGB-D Semantic Segmentation. Springer International Publishing, 2020, pp. 561–577. [Online]. Available: http://dx.doi.org/10.1007/978-3-030-58621-8 33

[63] X. Hu, K. Yang, L. Fei, and K. Wang, “Acnet: Attention based network to exploit complementary features for rgbd semantic segmentation,” in 2019 IEEE International Conference on Image Processing (ICIP). IEEE, Sep. 2019, pp. 1440–1444. [Online]. Available: http://dx.doi.org/10.1109/icip.2019.8803025

[64] S. Zhao, Y. Liu, Q. Jiao, Q. Zhang, and J. Han, “Mitigating modality discrepancies for rgb-t semantic segmentation,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 7, pp. 9380–9394, Jul. 2024. [Online]. Available: http: //dx.doi.org/10.1109/tnnls.2022.3233089

[65] Y. Chen, W. Zhan, Y. Jiang, D. Zhu, R. Guo, and X. Xu, “Lasnet: A light-weight asymmetric spatial feature network for real-time semantic segmentation,” Electronics, vol. 11, no. 19, p. 3238, 2022.

[66] X. Zheng, Y. Luo, P. Zhou, and L. Wang, “Distilling efficient vision transformers from cnns for semantic segmentation,” SSRN preprint, 2024. [Online]. Available: http://dx.doi.org/10.2139/ssrn.4782766

[67] X. Hu, K. Yang, L. Fei, and K. Wang, “Acnet: Attention based network to exploit complementary features for rgbd semantic segmentation,” in 2019 IEEE international conference on image processing (ICIP). IEEE, 2019, pp. 1440–1444.

[68] G. Hines, Z.-u. Rahman, D. Jobson, and G. Woodell, “Single-scale retinex using digital signal processors,” in Global signal processing conference, no. Paper 1324, 2005.

[69] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” 2015. [Online]. Available: https://arxiv.org/abs/1409.1556

[70] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[71] P. Zhou, X. Xie, Z. Lin, and S. Yan, “Towards understanding convergence and generalization of adamw,” IEEE transactions on pattern analysis and machine intelligence, vol. 46, no. 9, pp. 6486–6493, 2024.

[72] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[73] X. Guo, Z. Lin, L. Hu, Z. Deng, T. Liu, and W. Zhou, “Cross-modal state space modeling for real-time rgb-thermal wild scene semantic segmentation,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 5938–5945.

[74] Z. Zhang, Y. Xie, F. Xing, M. McGough, and L. Yang, “Mdnet: A semantically and visually interpretable medical image diagnosis network,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 6428–6436.

[75] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Transactions on Image Processing, 2004.

[76] X. Chen, K.-Y. Lin, J. Wang, W. Wu, C. Qian, H. Li, and G. Zeng, “Bi-directional cross-modality feature propagation with separation-and-aggregation gate for rgb-d semantic segmentation,” in European conference on computer vision. Springer, 2020, pp. 561–577.

[77] S. Dong, W. Zhou, C. Xu, and W. Yan, “Egfnet: Edge-aware guidance fusion network for rgb–thermal urban scene parsing,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 1, pp. 657–669, 2023.

[78] W. Zhou, H. Zhang, W. Yan, and W. Lin, “Mmsmcnet: Modal memory sharing and morphological complementary networks for rgb-t urban scene semantic segmentation,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 12, pp. 7096–7108, 2023.

[79] X. Guo, W. Zhou, and T. Liu, “Contrastive learning-based knowledge distillation for rgb-thermal urban scene semantic segmentation,” Knowledge-Based Systems, vol. 292, p. 111588, 2024.

[80] S. Zhao, J. Li, and Q. Zhang, “C<sup>4</sup>net: Excavating cross-modal context-and contentcomplementarity for rgb-t semantic segmentation,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 2, pp. 1347–1361, 2024.

[81] Y. Wang, G. Li, and Z. Liu, “Sgfnet: Semantic-guided fusion network for rgb-thermal semantic segmentation,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 12, pp. 7737–7748, 2023.

[82] U. Shin, K. Lee, I. S. Kweon, and J. Oh, “Complementary random masking for rgb-thermal semantic segmentation,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 11 110–11 117.

[83] X. Guo, T. Liu, Y. Mou, S. Chai, B. Ren, Y. Wang, W. Shi, S. Liu, and W. Zhou, “Transferring prior thermal knowledge for snowy urban scene semantic segmentation,” IEEE Transactions on Intelligent Transportation Systems, 2025.

[84] W. Zhou, J. Liu, J. Lei, L. Yu, and J.-N. Hwang, “Gmnet: Graded-feature multilabel-learning network for rgb-thermal urban scene semantic segmentation,” IEEE Transactions on Image Processing, vol. 30, pp. 7790–7802, 2021.

[85] J. Zhang, R. Liu, H. Shi, K. Yang, S. Reiß, K. Peng, H. Fu, K. Wang, and R. Stiefelhagen, “Delivering arbitrary-modal semantic segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1136–1147.

[86] W. Li, H. Zhou, J. Yu, Z. Song, and W. Yang, “Coupled mamba: Enhanced multimodal fusion with coupled state space model,” Advances in Neural Information Processing Systems, vol. 37, pp. 59 808–59 832, 2024.

[87] P. Peng, T. Xu, B. Huang, and J. Li, “Hafnet: Hierarchical attentive fusion network for multispectral pedestrian detection,” Remote Sensing, vol. 15, no. 8, p. 2041, 2023.

[88] X. He, M. Wang, T. Liu, L. Zhao, and Y. Yue, “Sfaf-ma: Spatial feature aggregation and fusion with modality adaptation for rgb-thermal semantic segmentation,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–10, 2023.