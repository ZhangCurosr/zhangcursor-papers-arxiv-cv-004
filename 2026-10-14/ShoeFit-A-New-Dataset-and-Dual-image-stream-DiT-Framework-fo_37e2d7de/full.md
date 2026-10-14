# ShoeFit: A New Dataset and Dual-image-stream DiT Framework for Virtual Footwear Try-On

Yuhan Li<sup>1∗†</sup>, Zhiyu Jin<sup>2∗</sup>, Yifan Tong<sup>2</sup>, Wenxiang Shang<sup>2</sup>, Benlei Cui<sup>2</sup>, Xuanhong Chen<sup>1</sup>, Hangcheng Zhu<sup>2</sup>, Bingbing Ni<sup>1‡</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Alibaba Group

{melodious, nibingbing}@sjtu.edu.cn https://colorful-liyu.github.io/ShoeFit/

(1) Data Scarcity with difficult matchMisalignment (2) Viewpoint (3) Background-induced Color Distortion

![](images/833a969df8961b06b3be74fb55ff1b0d64a38fc272fa2a934daa3227ef5e7e59.jpg)  
(a) We illustrate the challenges in VFTON. (1) Subtle variations make it difficult to match data pairs. (2) Information needed by the target view is often not provided in the given product image. (3) Shoes often exhibit complex reflections and shadows that interact with environment lighting, leading to a confusion of their natural color.

![](images/bb02de0b9e88cee1a7087df692d5a33f5aa99725366c7a8ce46d99e5cc601161.jpg)  
Human feet (Target view)  
Product shoes (Source view)  
Try-on results  
(b) ShoeFit has demonstrated excellent preservation of the fidelity of footwear across real-world scenes.

Figure 1: VFTON presents three challenges, while ShoeFit gives a successful solution.

## Abstract

Virtual footwear try-on (VFTON), a critical yet underexplored area in virtual try-on (VTON), aims to synthesize faithful try-on results given diverse footwear and model images while maintaining 3D consistency and texture authenticity. Unlike conventional garment-focused VTON methods, VFTON presents unique challenges due to (1) Data Scarcity, which arises from the difficulty of perfectly matching product shoes with models wearing the identical ones, (2) Viewpoint Misalignment, where the target foot pose and source shoe views are always misaligned, leading to incomplete texture information and detail distortion, and (3) Background-induced Color Distortion, where complex material of footwear interacts with environmental lighting, causing unintended color contamination. To address these challenges, we introduce MVShoes, a multi-view shoe try-on dataset consisting of 7305 wellannotated image triplets, covering diverse footwear categories and challenging try-on scenarios. Furthermore, we propose a dual-stream DiT architecture, ShoeFit, designed to mitigate viewpoint misalignment through Multi-View Conditioning with 3D Rotary Position Embedding, and alleviate background-induced distortion

using the LayeredRefAttention which leverages background features to modulate footwear latents. The proposed framework effectively decouples shoe appearance from environmental interferences while preserving high-quality texture detail through decoupled denoising and conditioning branches. Extensive quantitative and qualitative experiments demonstrate that our method substantially improves rendering fidelity and robustness under challenging real-world product shoes, establishing a new benchmark in high-fidelity footwear try-on synthesis.

## 1 Introduction

Virtual Footwear Try-On (VFTON) involves synthesizing product shoes onto human foot images, preserving their geometric structure and material properties, based on input images of both the source product and the target foot. VFTON offers significant commercial potential and academic value, with Nike’s annual footwear sales reaching 33,427 million dollars [48], while shoes, as complex 3D multi view objects [57], present unique challenges in research on controllable generation. Unfortunately, this field has not been sufficiently explored within the community.

There are three main challenges hindering the advancement of VFTON. (1) Data Scarcity: Due to the prevalence of footwear with similar appearances but subtle variations, as in Fig. 1 (a-1), such as slight differences in shoelaces, heels, and stripes, it is challenging to accurately match product shoes with models wearing the identical ones. Further exacerbating this issue, companies tend to keep data related to shoe try-ons confidential due to its clear commercial value [75, 19]. As a result, there is currently no available open-source dataset for footwear try-ons. (2) Viewpoint Misalignment: Unlike flat-lay 2D garment VTON, where both the target garment and human pose are captured from aligned front-facing perspectives, as in Fig. 1 (a-2), virtual footwear try-on inherently involves mismatched viewpoints between in-shop product images (source view) and target human poses (target view). Specifically, information needed by the target view is often not provided in the given product image due to 3D constraints, creating an information-deficient scenario that results in detail distortion and artifacts [42, 56]. (3) Background-induced Color Distortion: Unlike the diffuse reflection properties of knitted fabrics in clothing, which ensure visual color largely remains unaffected by the background, shoes often exhibit complex reflective properties that interact with ambient lighting [32], making them more susceptible to visual interference from background colors, as in Fig. 1 (a-3). This interaction between appearance and background leads to color contamination from surroundings and distorted material rendering in synthesized results. The above issues are first identified and explicitly proposed in this paper, which make it impossible for VFTON to share the same pipeline as VTON, thus presenting VFTON with unique challenges and research value.

To tackle these challenges, we introduce MVShoes, the first multi-view footwear try-on dataset. It comprises 7305 annotated high-resolution image triplets, each containing two product shoe images from different views and a corresponding try-on result, as shown in Fig. 2 (a). The dataset covers a diverse range of shoe categories and try-on scenarios, supported by a comprehensive and rigorous data cleaning pipeline to ensure accurate matching of identical product shoes with their human images.

Furthermore, we propose ShoeFit, a customized dual-image-stream DiT [51] architecture for VFTON, which generates high-fidelity try-on results through decoupled denoising and conditioning branches. ShoeFit is also equipped with two key improvements: (1) Multi-View Conditioning, incorporating reference concatenation [10] and corresponding 3D Rotary Position Embedding (3D RoPE). While a single source product view often fails to provide complete shoe texture features required for the target try-on view, the try-on results can significantly benefit from multiple source views, effectively reducing uncertainty in target surface rendering and minimizing texture distortion. (2)LayeredRefAttention Module. LayeredRefAttention leverages background features to modulate shoe features [28], decoupling the intrinsic shoe appearance from the surroundings. Additionally, the subsequent denoising attention suppresses background-related computations, effectively mitigating background color contamination. This layer-aware attention mechanism significantly enhances robustness in real-world footwear try-on scenarios. Our contributions are summarized as follows:

• We formally define the VFTON problem, reporting the unique issues of Data Scarcity, Viewpoint Misalignment and Background-induced color distortion across real-world scenarios.

• We curate the first shoe try-on dataset, containing 7305 pairs of high-quality multi-view garment-model samples, paving the way for subsequent footwear try-on studies.

![](images/1037d497a5c26ea54b9ee46a9075c3e7347d2f3b56a654424961d56ad638a8c9.jpg)  
Figure 2: We illustrate examples of the MVShoes dataset in (a) and data processing pipeline in (b).

• We introduce a dual-stream architecture with Multi-View Conditioning and LayeredRefAttention modules to address the challenges in VFTON.

• Both quantitative and qualitative experiments validate the superiority of our approach

## 2 Related Works

Image-based Virtual Try-on. The virtual try-on task is concerned with synthesizing images of a person donning the designated garment with appropriate fit [21], while retaining salient characteristics of the original garment and person, given a pair of images depicting a person and a target garment. To execute this task, numerous works [14, 26, 36, 4, 11, 20] have utilized Generative Adversarial Networks (GANs) [17] with two-stage strategy [37, 16, 66]: (1) warping the clothing to the desired shape [5, 40] and (2) fusing the deformed clothing via try-on generator based on GAN. As significant progress in Text-to-Image diffusion models [22, 47, 25, 8] is witnessed in recent years, some works [6, 18, 45, 29, 38] have been motivated to incorporate pre-trained diffusion models [55, 52] as priors into virtual try-on task. Most recently, OOTDiffusion [65], IDM [9], and MMTryon [75] achieve garment feature extraction with a parallel U-Net and feed them through self-attention for enhanced integration. CatVTON [10] concatenates the garment and person images along the spatial dimension and feed them into a single UNet. Unfortunately, these methods serve for fabric clothing, intrinsically lack support for faithful try-on to 3D multi-view shoes with various materials, exhibiting severe color contamination from the background.

Controllable Diffusion Models. To attain conditional control in diffusion models, T2I-Adapter [46] and IP-Adapter [69] incorporate additional trainable modules to fuse condition features. Additionally, some investigations utilize various prompt engineering techniques [39, 68, 73] and implement crossattention constraints [7, 64, 27, 74] to facilitate more controllable generation. Recently, the release of FLUX [33] and SD3 [15] has sparked renewed interest in DiT architectures [51] with subsequent conditional frameworks such as Flux.1 Fill [34] and Flux.1 Redux [35] achieving inpainting and object injection capabilities through enhanced encoder designs. However, these methods remain constrained by coarse-grained conditioning mechanisms, which fail to ensure generation fidelity—particularly in VFTON tasks with high geometric variability and stringent fidelity requirements.

## 3 MVShoes Dataset

We collect the first footwear try-on dataset which contains a total of 7305 annotated high-resolution image triplets, each consisting of two product shoe images from different views, and a corresponding try-on result. The dataset features diverse shoe categories and includes foot-specific pose landmarks and shoes-agnostic masks to facilitate precise alignment.

## 3.1 MVShoes Collection

All images are sourced from the Internet and include a diverse range of model scenarios such as full-body models, half-body models, foot close-ups, as well as a comprehensive variety of shoe categories, effectively meeting practical application needs. Initially, we collected approximately 25,000 raw shoe product image sets, each consisting of 2 to 5 images per shoe, including standalone product images and images of people wearing the shoes from different perspectives.

Data Processing. We first employ Qwen-VL [3] to differentiate between product footwear images (without humans) and try-on images featuring human models. Subsequently, we use the CLIP model [53] to extract image features and set a similarity threshold to eliminate duplicate images within the same raw image set. Next, we segment the shoe region in each image, match the DINO [43] features and assess their similarity to filter those data pairs with different footwear colors and styles. This process results in the formation of preliminary human-footwear try-on triplets, followed by manual filtering to eliminate any errors or oversights. We then remove visually blurred images and compile statistics on shoe categories and try-on scenarios. Finally, following the method proposed in DWPose [67] and Grounding DINO [43], we extract foot poses and shoe-agnostic masks. This process culminates in the creation of high-resolution, category-comprehensive try-on triplets. The data processing pipeline is illustrated in Fig. 2 (b).

## 3.2 Dataset Statistic

Rich Scenes. In conventional VTON tasks [37, 12], model images typically consist of full-body or upper-body views. However, given that footwear occupies a relatively small proportion of the body in VFTON tasks, full-body and halfbody models are less effective in showcasing shoe details. To address this, we simulate the distribution of footwear model scenes observed on e-commerce platforms and curate 4 model scenarios: top-down foot close-up, horizontal foot close-up, half-body model, and full-body model, as illustrated in Fig. 3, where horizontal foot close-ups constitute the largest proportion. In such cases, accurate preservation of side details is particularly challenging, especially when compounded by viewpoint misalignment.

Diverse Footwear Categories. We also prioritize the comprehensiveness of footwear categories. By annotating shoe attributes [2], we establish a two-level product category label set and ensure that its distribution aligns with everyday use, as depicted in Fig. 3. Notably, we

![](images/5943961272048eb17c68603fad5d598156c50ed7dc6b85a7296f62f20a24d0e1.jpg)  
Figure 3: The category distribution of MVShoes.

observe that environmental lighting often affects the visual appearance of footwear, where reflections introduce background color contamination, particularly in categories such as boots, leather shoes, and high heels.

![](images/da293126669ece897ddc3a01504426e9e586386f9d37e5222b53a844bc6801f5.jpg)  
Figure 4: We illustrate existing and our DiT framework comparison in (a)-(b), and heatmaps visualization for component validation of Multi-view Conditioning and LayeredRefAttention in (c)-(d).

## 4 Method

An overview of the ShoeFit is presented in Fig. 4 (b). Given a human image X and target shoe images $C _ { I } ^ { ( n ) }$ , ShoeFit is aimed to generate an authentic try-on image. The backbone of ShoeFit employs the FLUX.1-Lite [33], with a customized dual-image-stream architecture in Sec. 4.1: denoising image stream and a copy of it as as conditioning branch to replace text modality. The features of human images and multi-view shoes are then integrated through Reference Concatenation and LayeredRefAttention modules, which are described in Sec. 4.2 and 4.3.

## 4.1 Dual-image-stream DiT Framework

The original FLUX [33] is a text-to-image model composed of a series of stacked MM-DiT blocks. Earlier works [60, 61] directly concatenate Image/Text condition tokens with noisy image tokens such as $[ X ; C _ { I } ; C _ { T } ]$ , relying on a single denoising stream to finish condition extraction and denoising, while the other deal with text modality, as shown in Fig. 4(a). However, for VFTON, the generated try-ons are primarily determined by the given shoe images rather than limited semantic prompts. Thus, we remove the text stream from FLUX, achieving approximately 40% parameter savings, including those for T5 [54] and CLIP models [53]. Instead, we replicate a portion of the denoising stream’s architecture and weights to initialize the conditioning image stream, establishing a dualimage-stream framework. This design decouples feature preservation of shoe details from the high-quality generation process, allocating additional effective model capacity to enhance overall fidelity performance compared to direct token concatenation approaches [60, 61].

## 4.2 Multi-view Conditioning

In VFTON, the information about shoes from the try-on view remains incomplete due to view mismatch and 3D-to-2D projection occlusion [63]. The inherent misalignment between source product images and target human images leads to unfaithful and physically implausible try-on results. Observing that e-commerce platforms often provide multiple-view product images (e.g., front, side, top views), we introduce a multi-view shoe image dataset and propose a multi-view conditional injection mechanism via Reference Concatenation and 3D RoPE. The additional perspectives comprehensively encompass more aspects of the shoe, as illustrated in Fig. 4 (c), significantly reducing uncertainty during generation and delivering high-fidelity and commercially viable synthetic results.

Reference Concatenation. Inspired by CatVTON [10], we adopt a simple yet effective multi-view injection strategy: concatenate multi-view conditioning shoe images along the spatial dimension as: $C _ { I } ^ { c a t } = [ \bar { C } _ { I } ^ { \breve { 1 } } , \bar { C } _ { I } ^ { 2 } , . . . C _ { I } ^ { N } ] \in \mathbb { R } ^ { b \times N \cdot l \times c }$ , where b s the batch $\mathrm { s i z e } , \mathrm { \Delta } l$ denotes the number of image tokens per conditioning image, and c is the feature channel. The concatenated multi-view latent $C _ { I } ^ { c a t }$ is then fed into the conditioning image stream for feature extraction and interaction with the target human image. Concurrently, the denoising stream receives 4 concatenated components with $1 6 + 1 6 + 1 + 1 6$ channels: the noisy human latents $X _ { t } ,$ the latent shoe-agnostic masked image $E ( X _ { m a s k e d } )$ , the resized agnostic mask $X _ { m }$ , and the foot pose landmarks $E ( X _ { p o s e } )$ , where $E ( \cdot )$ represents VAE [30] encoding. The effectiveness of Reference Concatenation stems from the global attention interactions in attention layers. Intuitively, during the attention, each Query patch in the human image accesses and matches the most semantically relevant Key-Value pairs from the full set of concatenated multi-view product features, as illustrated in Fig. 4 (c). As a result, our method ensures consistent alignment between all regions in the target pose and their corresponding details in the product images, thereby achieving high-fidelity virtual try-on.

3D RoPE. However, naive Reference Concatenation introduces position misalignment in the relative positional relation of conditioning images. FLUX employs an essentially 2D RoPE scheme [58] to encode spatial relative relationships along the height and width dimensions, where the inner product in attention depends solely on the token embeddings $x _ { m } , x _ { n }$ and their relative distances $m - n \colon$

$$
\langle f _ { q } ( x _ { m } , m ) , f _ { k } ( x _ { n } , n ) \rangle = g ( x _ { m } , x _ { n } , m - n ) .\tag{1}
$$

When multi-view conditioning images are naively concatenated, discontinuous semantic jumps emerge at the concatenation boundaries, where tokens with small relative distances may correspond to semantically distant regions (i.e., different views of the footwear).

To address this issue, we extend the 2D RoPE to three dimensions. Specifically, $E _ { p o s I D } =$ concat $[ E _ { p o s I D } ^ { n } ; E _ { p o s I D } ^ { h } ; E _ { p o s I D } ^ { w } ] \in \mathbb { R } ^ { H \times W \times 3 }$ , where the first channel $E _ { p o s I D } ^ { n }$ serves as a index mask to separate different conditioning views, while the second and third channels $E _ { p o s I D } ^ { \{ h , w \} }$ encode spatial positions within each image along the height and width dimensions, respectively. Within each channel of $E _ { p o s I D } ^ { \{ n , h , w \} }$ , frequency encoding is performed given a base frequency of θ = 10000.

$$
\omega _ { d } = \frac { 1 } { \theta ^ { 2 d / D } } , \mathrm { f o r } d = 0 , 1 , . . . , \frac { D } { 2 } - 1 , \quad v _ { \{ n , h , w \} } = E _ { p o s I D } ^ { \{ n , h , w \} } \cdot \omega _ { d } \in \mathbb { R } ^ { H \times W \times \frac { \{ D _ { n } , D _ { h } , D _ { w } \} } { 2 } } ,\tag{2}
$$

where hype-parameters $D _ { n } , D _ { h } , D _ { w }$ is predefined as 16, 56, 56. Subsequently, we obtain positional rotation matrices $R _ { 1 D }$ for each of the $n , h .$ , w channels internally. By concatenating these matrices along D dimension, we construct 3D RoPE matrix $\begin{array} { r } { R _ { 3 D } \in \mathbb { R } ^ { H \times W \times \frac { ( D _ { n } + D _ { h } + D _ { w } ) } { 2 } \times 2 \times 2 } } \end{array}$ as follow:

$$
R _ { 1 D } ^ { \{ n , h , w \} } = \left[ { \ L } _ { s i n } ^ { c o s \left( v _ { \{ n , h , w \} } \right) } \quad \ L ^ { - s i n \left( v _ { \{ n , h , w \} } \right) } \right] , \quad R _ { 3 D } = \mathrm { c o n c a t } [ R _ { 1 D } ^ { n } ; R _ { 1 D } ^ { h } ; R _ { 1 D } ^ { w } ]\tag{3}
$$

## 4.3 LayeredRefAttention Module

We observe that the appearance of the synthesized shoes is highly susceptible to contamination from the background colors of the conditioning images. This phenomenon arises from the complex reflective material of footwear interacting strongly with environmental lighting, termed Backgroundinduced Color Distortion. Specifically, the issue arises when Query patches from the human image inadvertently match key-value pairs originating from the background regions of the conditioning image, as illustrated in Fig. 4 (d). A naive solution involves masking out background regions during attention, but this would neglect critical illumination cues embedded in the background, which are essential for accurately inferring the shoe’s intrinsic appearance and texture [32].

To address this challenge, we propose a layer-aware attention module called LayeredRefAttention to explicitly differentiate between background and foreground content in the conditioning images, as illustrated in Fig. 4 (b). By leveraging background features to modulate foreground shoe features, our method decouples the intrinsic shoe appearance from its background context [28]. Subsequent denoising stream attention computations are restricted to interactions between the foreground shoe content and the target human image, effectively eliminating background illumination effects while avoiding background color contamination. Specifically, for $n \in \{ 1 , . . . , N \}$ conditioning view $C _ { I } ^ { ( n ) }$ we employ a Squeeze-and-Excitation (SE) block [24] followed by global average pooling over spatial dimensions to compute channel-specific weights:

Table 1: Quantitative comparison on MVshoes. ShoeFit significantly surpasses all baselines.
<table><tr><td>View Num.</td><td>Method</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID↓</td><td>KID↓</td><td>DISTS ↓</td></tr><tr><td rowspan="5">1-View</td><td>FLUX.1-Fill [34]</td><td>0.725</td><td>0.201</td><td>24.44</td><td>2.221</td><td>0.1293</td></tr><tr><td>FLUX.1-Fill-Redux [35]</td><td>0.724</td><td>0.199</td><td>24.67</td><td>2.086</td><td>0.1274</td></tr><tr><td>OOTDiffusion-SDXL [65]</td><td>0.724</td><td>0.194</td><td>22.60</td><td>2.345</td><td>0.1114</td></tr><tr><td>CatVTON-Flux-Lite [10]</td><td>0.745</td><td>0.185</td><td>22.71</td><td>3.327</td><td>0.1179</td></tr><tr><td>ShoeFit-1V</td><td>0.772</td><td>0.159</td><td>21.07</td><td>0.944</td><td>0.1059</td></tr><tr><td rowspan="2">2-View</td><td>CatVTON-Flux-Lite-2V [10]</td><td>0.756</td><td>0.171</td><td>22.64</td><td>3.318</td><td>0.1040</td></tr><tr><td>ShoeFit-2V</td><td>0.780</td><td>0.149</td><td>20.18</td><td>0.485</td><td>0.0941</td></tr></table>

$$
P ^ { ( n ) } = A v g P o o l ( S E ( C _ { I } ^ { ( n ) } ) \in \mathbb { R } ^ { b \times c } .\tag{4}
$$

Subsequently, we employ two linear layers $F ( \cdot )$ to extract background modulation parameters and modulate foreground shoe features, filtering irrelevant environmental lighting and background reflection to ensure faithful material preservation of shoes as:

$$
\begin{array} { r l } & { \beta _ { s c a l e } ^ { ( n ) } , \beta _ { s h i f t } ^ { ( n ) } = F ( P ^ { ( n ) } ) , \quad C _ { f g } ^ { ( n ) } = ( 1 + \beta _ { s c a l e } ^ { ( n ) } ) \cdot ( L N ( C _ { I } ^ { ( n ) } \odot M _ { f g } ^ { ( n ) } ) ) + \beta _ { s h i f t } ^ { ( n ) } , } \end{array}\tag{5}
$$

where $L N ( \cdot )$ means layer normalization and ${ M } _ { f g } ^ { ( n ) }$ represents the binary shoe masks for conditioning product image. The attention variables $\{ K , V \} _ { c , f g } ^ { ( n ) }$ extracted from N views of modulated foreground conditioning feature are concatenated with the corresponding $\{ Q , K , V \} _ { i m g }$ from human image for subsequent denosing-stream attention:

$$
Q _ { C , f g } ^ { ( n ) } , K _ { C , f g } ^ { ( n ) } , V _ { C , f g } ^ { ( n ) } = Q K V ( C _ { f g } ^ { ( n ) } ) ,\tag{6}
$$

$$
A t t n _ { i m a g e } = \mathrm { s o f t m a x } ( \frac { Q _ { i m g } \cdot \mathrm { c o n c a t } ( K _ { i m g } ; K _ { C , f g } ^ { 1 , \dots , N , \top } ) } { \sqrt d } ) \cdot \mathrm { c o n c a t } ( V _ { i m g } ; V _ { C , f g } ^ { 1 , \dots , N } ) .\tag{7}
$$

Meanwhile, the conditioning-stream branch performs attention computation using unmasked conditioning latents as:

$$
Q _ { C } ^ { ( n ) } , K _ { C } ^ { ( n ) } , V _ { C } ^ { ( n ) } = Q K V ( L N ( C _ { I } ^ { ( n ) } ) ) , \quad A t t n _ { C } = \mathrm { s o f t m a x } ( \frac { Q _ { C } ^ { 1 , . . . , N } \cdot K _ { C } ^ { 1 , . . . , N , \top } } { \sqrt { d } } ) \cdot V _ { C } ^ { 1 , . . . , N } .\tag{8}
$$

To clarify our contributions, we omit the distillation strength normalization and final residual connections in the attention layer of FLUX.1-Lite [33] that remain unchanged from the original implementation. In summary, LayeredRefAttention effectively enables material-background decoupling by modulating foreground shoe features to recover intrinsic color properties, resulting in texturepreserving footwear try-on synthesis, as illustrated in Fig. 4 (d).

## 5 Experiments

## 5.1 Experimental Setup

Implementation Details. We employ the FLUX.1-Lite [33] as the backbone and customize a dualimage-stream architecture as in Sec. 4.1. Our experiments are carried out on MVShoes at a resolution of 768 × 768, with 6305 pairs for training and 1000 pairs for testing. The model is trained for 7 days on 8 80GB-A100 GPUs with DeepSpeed [1] ZeRO-2 to reduce memory usage, at a batch size of 4. All parameters of DiT are trainable, using an AdamW optimizer [44] with a constant learning rate of 3e − 5. At inference time, we run ShoeFit on a single A10 GPU for 25 steps. The data augmentation follows the same protocol as in StableVTON [29]. Please refer to Appendix for more details.

ShoeFit (Ours)  
CatVTON  
FLUX.1-Fill-Redux  
FLUX.1-Fill  
![](images/3547bcd1261e53010451f36d74a143cd84dcc5ceb2809c667ccec7355b36ebae.jpg)

Figure 5: Comparisons on the hard scenes, where ShoeFit delivers high-fidelity and rubost results.  
Ablation  
Full  
Conditions  
Ablation  
Full  
Conditions  
Ablation  
Full  
![](images/adfade5b804947b534213a26b19da45d6ec7c8325fb51a89ac452c942777f5f8.jpg)  
Figure 6: Ablation study. We highlight the improvements by all three components of method in red.

Evaluation Protocols. Following evaluation in prior VTON works [45], we measure reconstruction accuracy by LPIPS [72], SSIM [62] and DISTS [13] in a paired setting given ground truth images, and authenticity of unpaired synthesized images by FID [50] and KID [59] without ground truth.

Baselines. As a subtask of inpainting, we first select two widely used inpainting FLUX variants as baselines: FLUX.1-Fill [34] and FLUX.1-Fill-Redux [35]. The former involves concatenating product images with human images on the spatial dimension, similar to CatVTON [10]. The latter enhances the conditioning stream with SigLip [70] to incorporate product shoe images. Additionally, we compare ShoeFit to two state-of-the-art methods in the VTON field: OOTDiffusion [65] based on the SDXL [52] and CatVTON [10] built upon the FLUX.1-Lite. All models are implemented as originally deployed for single-view product shoe conditional generation. Considering CatVTON’s capability to easily extend to multi-image injection, we further train a version of CatVTON to simultaneously inject two product views, the same as ShoeFit, denoted as CatVTON-FLUX-Lite-2V. Please refer to Appendix for more implementation details.

## 5.2 Qualitative Results

Fig. 5 provides a qualitative comparison between ShoeFit and the baselines on MVShoes, addressing challenges such as complex structures, background interference, and difficult target poses. For a fair comparison, we report the results of the single-view conditioning version of all methods. Notably, we observe that OOTDiffusion [65] tends to exhibit extreme color variations, while FLUX.1-Fill-Redux [35] and FLUX.1-Fill [34], possibly affected by the limitation of only LoRA [23] being trainable (r=64), often produce visually plausible results that do not strictly adhere to the input conditions. In contrast, ShoeFit accurately generates structural details of the shoes (rows 1 and 2), decouples environmental lighting (row 3), avoids background confusion (row 4), and demonstrates robustness to uncommon poses (row 5). Please refer to the Appendix for more results.

## 5.3 Quantitative Results

For a fair comparison, we report the results of one/two product view conditioning generation separately in Tab. 1. FLUX.1-Fill-Redux [35] and FLUX.1-Fill [34] achieve the worst FID and DISTS scores, while OOTDiffusion [65] and CatVTON [10] show complementary performance in unpaired and paired evaluations. ShoesFit, benefiting from the additional specialized parameters provided by the Dual-image-stream DiT framework and background decoupling enabled by the LayeredRefAttention, significantly surpasses all baselines under both single-view and multi-view conditional generation.

## 5.4 Ablation Study

To validate our technical contributions, we define a Vanilla Model that uses FLUX.1-Lite as the backbone, deploying both a text-stream and an image-stream as in Fig. 4 (a). Multi-view Conditioning and LayeredRefAttention Module are removed, using only one footwear image as conditioning.

Dual-image-stream DiT Framework. Since the text-stream only provides high-level semantics, its utility is limited in VFTON, which demands high image fidelity. Therefore, we first remove the text-stream and replicate a portion of the denoising stream’s architecture and weights to initialize the conditioning image stream, effectively decoupling the conditional feature extraction from the denoising process, while keeping the Vanilla Model’s other settings unchanged(denoted as “+ D-i-s”). As shown in Tab. 2 and Fig. 6, the vanilla model often generates artifacts and incorrect structures facing high-frequency textures due to the limitation in effective network capacity and the weakness in coupling of conditional images and noise. This demonstrates that the Dual-image-stream DiT framework plays a fundamental role in enhancing the model’s expressive capabilities.

LayeredRefAttention Module. Observing the visual contamination from background and environmental lighting, we further integrate the LayeredRefAttention into the “+ D-i-s” version, denoted as “+ D-i-s + LRA”, which effectively decouples the intrinsic appearance of the shoe from its surroundings. As illustrated in Tab. 2 and Fig. 6, the model without LayeredRefAttention often fails to accurately reproduce the natural colors of the shoe where environmental lighting or shadows of product images are prominent, resulting in outputs that are mixed with environmental light or shadows. Additionally, mismatches with background patches in Attention can introduce colors confusion from the background content, such as slippers being contaminated by floor colors in Column 3 of Fig. 6. In contrast, the one with "LRA" effectively avoids such color contamination, delivering superior try-on results.

To investigate the efficacy of simple background masking, we conducted an ablation study utilizing standard attention while removing the background from the product shoe images. Visually, this naive approach exacerbates color artifacts, particularly for highly saturated hues. We hypothesize that this occurs because removing the background discards crucial illumination cues embedded in the product image, which are essential for the model to infer the shoe’s intrinsic material properties and texture. Given that the training set contains inherent color variations between in-studio product shots and on-foot images due to disparate lighting, forcing the model to operate without background context introduces lighting ambiguity. Consequently, the model struggles to maintain color constancy, leading to degraded try-on performance and reduced visual fidelity.

Table 2: Quantitative ablation study of each component in Section of Method.
<table><tr><td>Method</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID↓</td><td>KID↓</td><td>DISTS↓</td></tr><tr><td>Vanilla Model</td><td>0.745</td><td>0.185</td><td>22.71</td><td>3.327</td><td>0.1179</td></tr><tr><td>+ D-i-s</td><td>0.761</td><td>0.174</td><td>21.57</td><td>1.493</td><td>0.1038</td></tr><tr><td>+ D-i-s + Standard Attn (w/ mask)</td><td>0.756</td><td>0.181</td><td>21.70</td><td>1.698</td><td>0.1056</td></tr><tr><td> $+ \mathrm { D - i - s + L R A }$  (w/ mask)</td><td>0.772</td><td>0.159</td><td>21.07</td><td>0.944</td><td>0.1059</td></tr><tr><td>十  $- \mathrm { D } \mathrm { - i } \mathrm { - s } + \mathrm { L R A } +$  MC (w/o 3D RoPE)</td><td>0.778</td><td>0.152</td><td>20.18</td><td>0.695</td><td>0.0971</td></tr><tr><td> $+ \mathrm { D - i - s + L R A + M C \left( F u l l \right) }$ </td><td>0.780</td><td>0.149</td><td>20.18</td><td>0.485</td><td>0.0941</td></tr></table>

Multi-view Conditioning System. While a single source product view often fails to capture the complete texture of footwear, try-on results can benefit from multiple source views, thereby effectively reducing uncertainty in generation. Thus, we introduce the Multi-view Conditioning pipeline using Reference Concatenation and 3D RoPE in the “+ D-i-s + LRA” version, denoted as “+ D-i-s + LRA + MC”. As demonstrated in Tab. 2 and Fig. 6, the model without “MC”, which uses only one product shoe image as input (yellow box), often generates unrealistic speculations and artifacts in unseen areas due to insufficient conditional information. In contrast, ShoeFit benefits from the Multi-view Conditioning system, producing details that are more faithful to the input across different views. Furthermore, we also conduct an ablation study upon 3D RoPE’s impact in Tab. 2, demonstrating the performance gain we achieve using the proposed RoPE without incurring any additional computational overhead. Please refer to the Appendix for more ablation results.

## 5.5 Inference Time and Memory Usage

We evaluated the efficiency of our proposed enhancements by measuring inference time and memory footprint on an NVIDIA H20 GPU (with batch size of 1, 25 steps, and torch.bfloat16 precision). The results, presented in Table 3, reveal that the ShoeFit model with our Dual-image-stream (D-i-s) architecture substantially outperforms CAT-VTON in both speed and memory efficiency. This stems from removing the text stream and its computationally expensive T5 and CLIP encoders.

Table 3: Comparison of inference time and GPU memory usage across different architectures.
<table><tr><td>Method</td><td>Inference Time per Image</td><td>GPU Memory Usage</td></tr><tr><td>Cat-VTON (image + text streams)</td><td>22.27 seconds</td><td>35.5 GB</td></tr><tr><td>ShoeFit (Dual-image-stream, D-i-s)</td><td>17.44 seconds</td><td>22.0 GB</td></tr><tr><td>ShoeFit + D-i-s + LayeredRefAttention</td><td>21.15 seconds</td><td>28.5 GB</td></tr></table>

## 6 Conclusion

In conclusion, we formally define the VFTON problem and curate the first shoe try-on dataset, MVShoes, containing 7305 pairs of high-quality multi-view garment-model samples. Our proposed framework addresses the critical challenges of viewpoint misalignment and background-induced color distortion by leveraging a dual-stream architecture with Multi-View Conditioning and LayeredRefAttention modules. Through a comprehensive evaluation on the proposed MVShoes dataset, we demonstrate significant improvements in detail fidelity and texture consistency across diverse scenarios. This indicates that our method effectively mitigates background interference while preserving fine-grained shoe details across views, establishing a strong baseline for future VFTON research.

Broader impacts. With the ability to synthesize images, arises the risk that ShoeFit might be used for inappropriate purposes such as producing media that breaches intellectual property rights or privacy norms. Because of these risks, we strongly advocate for the conscientious use of this technology.

Limitation and future work. ShoeFit demonstrates substantial improvements in generating highfidelity footwear try-on images across diverse scenes. However, certain limitations persist. Firstly, the model occasionally struggles with accurately rendering small logos and intricate text, particularly critical for e-commerce sellers. Secondly, we hope to introduce explicit 3D geometric and material priors in the future to achieve more robust multi-view representation and refined visual fidelity.

## Acknowledgements

This work is supported by the Science and Technology Commission of Shanghai Municipality under research grant No. 25ZR1401187.

## References

[1] Deepspeed. https://github.com/microsoft/DeepSpeed

[2] Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)

[3] Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., Zhou, J.: Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond (2023), https://arxiv.org/abs/2308.12966

[4] Bai, S., Zhou, H., Li, Z., Zhou, C., Yang, H.: Single stage virtual try-on via deformable attention flows. In: European Conference on Computer Vision (2022)

[5] Bookstein, F.L.: Principal warps: Thin-plate splines and the decomposition of deformations. IEEE Transactions on pattern analysis and machine intelligence (1989)

[6] Chen, M., Chen, X., Zhai, Z., Ju, C., Hong, X., Lan, J., Xiao, S.: Wear-any-way: Manipulable virtual try-on via sparse correspondence alignment. arXiv preprint (2024)

[7] Chen, M., Laina, I., Vedaldi, A.: Training-free layout control with cross-attention guidance. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 5343–5353 (2024)

[8] Chen, X., Huang, L., Liu, Y., Shen, Y., Zhao, D., Zhao, H.: Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481 (2023)

[9] Choi, Y., Kwak, S., Lee, K., Choi, H., Shin, J.: Improving diffusion models for virtual try-on. arXiv preprint arXiv:2403.05139 (2024)

[10] Chong, Z., Dong, X., Li, H., Zhang, S., Zhang, W., Zhang, X., Zhao, H., Jiang, D., Liang, X.: Catvton: Concatenation is all you need for virtual try-on with diffusion models. arXiv preprint arXiv:2407.15886 (2024)

[11] Chopra, A., Jain, R., Hemani, M., Krishnamurthy, B.: Zflow: Gated appearance flow-based virtual try-on with 3d priors. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2021)

[12] Davide, M., Matteo, F., Marcella, C., Federico, L., Fabio, C., Rita, C.: Dress code: Highresolution multi-category virtual try-on. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022)

[13] Ding, K., Ma, K., Wang, S., Simoncelli, E.P.: Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence 44(5), 2567–2581 (2020)

[14] Dong, H., Liang, X., Zhang, Y., Zhang, X., Shen, X., Xie, Z., Wu, B., Yin, J.: Fashion editing with adversarial parsing learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8120–8128 (2020)

[15] Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning (2024)

[16] Ge, Y., Song, Y., Zhang, R., Ge, C., Liu, W., Luo, P.: Parser-free virtual try-on via distilling appearance flows. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2021)

[17] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in neural information processing systems (2014)

[18] Gou, J., Sun, S., Zhang, J., Si, J., Qian, C., Zhang, L.: Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In: Proceedings of the 31st ACM International Conference on Multimedia (2023)

[19] Guo, H., Zeng, B., Song, Y., Zhang, W., Zhang, C., Liu, J.: Any2anytryon: Leveraging adaptive position embeddings for versatile virtual clothing tasks. arXiv preprint arXiv:2501.15891 (2025)

[20] Han, X., Hu, X., Huang, W., Scott, M.R.: Clothflow: A flow-based model for clothed person generation. In: Proceedings of the IEEE/CVF international conference on computer vision (2019)

[21] Han, X., Wu, Z., Wu, Z., Yu, R., Davis, L.S.: Viton: An image-based virtual try-on network. In: Proceedings of the IEEE conference on computer vision and pattern recognition (2018)

[22] Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems (2020)

[23] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. ICLR 1(2), 3 (2022)

[24] Hu, J., Shen, L., Sun, G.: Squeeze-and-excitation networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 7132–7141 (2018)

[25] Huang, L., Chen, D., Liu, Y., Shen, Y., Zhao, D., Zhou, J.: Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778 (2023)

[26] Jo, Y., Park, J.: Sc-fegan: Face editing generative adversarial network with user’s sketch and color. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1745–1753 (2019)

[27] Ju, X., Zeng, A., Zhao, C., Wang, J., Zhang, L., Xu, Q.: Humansd: A native skeleton-guided diffusion model for human image generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15988–15998 (2023)

[28] Karras, T., Laine, S., Aila, T.: A style-based generator architecture for generative adversarial networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4401–4410 (2019)

[29] Kim, J., Gu, G., Park, M., Park, S., Choo, J.: Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

[30] Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)

[31] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. arXiv preprint arXiv:2304.02643 (2023)

[32] Kocsis, P., Philip, J., Sunkavalli, K., Nießner, M., Hold-Geoffroy, Y.: Lightit: Illumination modeling and control for diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9359–9369 (2024)

[33] forest labs, B.: Flux.1-dev. https://github.com/black-forest-labs/flux (2024), https://github.com/black-forest-labs/flux

[34] forest labs, B.: Flux.1-fill-dev. https://huggingface.co/black-forest-labs/ FLUX.1-Fill-dev (2024), https://huggingface.co/black-forest-labs/FLUX. 1-Fill-dev

[35] forest labs, B.: Flux.1-redux-dev. https://huggingface.co/black-forest-labs/ FLUX.1-Redux-dev (2024), https://huggingface.co/black-forest-labs/FLUX. 1-Redux-dev

[36] Lee, C.H., Liu, Z., Wu, L., Luo, P.: Maskgan: Towards diverse and interactive facial image manipulation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5549–5558 (2020)

[37] Lee, S., Gu, G., Park, S., Choi, S., Choo, J.: High-resolution virtual try-on with misalignment and occlusion-handled conditions. In: European Conference on Computer Vision (2022)

[38] Li, Y., Zhou, H., Shang, W., Lin, R., Chen, X., Ni, B.: Anyfit: Controllable virtual try-on for any combination of attire across any scenario. arXiv preprint arXiv:2405.18172 (2024)

[39] Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., Lee, Y.J.: Gligen: Open-set grounded text-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22511–22521 (2023)

[40] Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature pyramid networks for object detection. In: Proceedings of the IEEE conference on computer vision and pattern recognition (2017)

[41] Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022)

[42] Liu, M., Xu, C., Jin, H., Chen, L., Varma T, M., Xu, Z., Su, H.: One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems 36, 22226–22246 (2023)

[43] Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., Zhu, J., Zhang, L.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection (2024), https://arxiv.org/abs/2303.05499

[44] Loshchilov, I., Hutter, F.: Fixing weight decay regularization in adam (2018)

[45] Morelli, D., Baldrati, A., Cartella, G., Cornia, M., Bertini, M., Cucchiara, R.: LaDI-VTON: Latent Diffusion Textual-Inversion Enhanced Virtual Try-On. In: Proceedings of the ACM International Conference on Multimedia (2023)

[46] Mou, C., Wang, X., Xie, L., Wu, Y., Zhang, J., Qi, Z., Shan, Y.: T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 4296–4304 (2024)

[47] Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741 (2021)

[48] NIKE, I.: Nike, inc. annual report. https://www.sec.gov/Archives/edgar/data/ 320187/000032018724000044/nke-20240531.htm (2024), https://www.sec.gov/ Archives/edgar/data/320187/000032018724000044/nke-20240531.htm

[49] Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Howes, R., Huang, P.Y., Xu, H., Sharma, V., Li, S.W., Galuba, W., Rabbat, M., Assran, M., Ballas, N., Synnaeve, G., Misra, I., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: Dinov2: Learning robust visual features without supervision (2023)

[50] Parmar, G., Zhang, R., Zhu, J.Y.: On aliased resizing and surprising subtleties in gan evaluation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022)

[51] Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023)

[52] Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)

[53] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning (2021)

[54] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer (2023), https://arxiv.org/abs/1910.10683

[55] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2022)

[56] Shi, R., Chen, H., Zhang, Z., Liu, M., Xu, C., Wei, X., Chen, L., Zeng, C., Su, H.: Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110 (2023)

[57] Shi, Y., Wang, P., Ye, J., Long, M., Li, K., Yang, X.: Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512 (2023)

[58] Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063 (2024)

[59] Sutherland, J., Arbel, M., Gretton, A.: Demystifying mmd gans. In: International Conference for Learning Representations (2018)

[60] Tan, Z., Liu, S., Yang, X., Xue, Q., Wang, X.: Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098 (2024)

[61] Wang, H., Peng, J., He, Q., Yang, H., Jin, Y., Wu, J., Hu, X., Pan, Y., Gan, Z., Chi, M., et al.: Unicombine: Unified multi-conditional combination with diffusion transformer. arXiv preprint arXiv:2503.09277 (2025)

[62] Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing (2004)

[63] Wikipedia: 3d projection. https://en.wikipedia.org/wiki/3D\_projection (2025), https://en.wikipedia.org/wiki/3D\_projection

[64] Xie, J., Li, Y., Huang, Y., Liu, H., Zhang, W., Zheng, Y., Shou, M.Z.: Boxdiff: Text-toimage synthesis with training-free box-constrained diffusion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7452–7461 (2023)

[65] Xu, Y., Gu, T., Chen, W., Chen, C.: Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. arXiv preprint arXiv:2403.01779 (2024)

[66] Yang, H., Zhang, R., Guo, X., Liu, W., Zuo, W., Luo, P.: Towards photo-realistic virtual try-on by adaptively generating-preserving image content. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2020)

[67] Yang, Z., Zeng, A., Yuan, C., Li, Y.: Effective whole-body pose estimation with two-stages distillation (2023), https://arxiv.org/abs/2307.15880

[68] Yang, Z., Wang, J., Gan, Z., Li, L., Lin, K., Wu, C., Duan, N., Liu, Z., Liu, C., Zeng, M., et al.: Reco: Region-controlled text-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14246–14255 (2023)

[69] Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023)

[70] Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training (2023), https://arxiv.org/abs/2303.15343

[71] Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3836–3847 (2023)

[72] Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition (2018)

[73] Zhang, T., Zhang, Y., Vineet, V., Joshi, N., Wang, X.: Controllable text-to-image generation with gpt-4. arXiv preprint arXiv:2305.18583 (2023)

[74] Zhang, X., Song, D., Zhan, P., Chen, Q., Xu, Z., Luo, W., Zhang, K., Liu, A.: Boowvton: Boosting in-the-wild virtual try-on via mask-free pseudo data training. arXiv preprint arXiv:2408.06047 (2024)

[75] Zhang, X., Lin, E., Li, X., Luo, Y., Kampffmeyer, M., Dong, X., Liang, X.: Mmtryon: Multi-modal multi-reference control for high-quality fashion generation. arXiv preprint arXiv:2405.00448 (2024)

• In Section A, we provide a more detailed discussion about the limitations of ShoeFit.

• In Section B, we provide additional details about the MVShoes datasets, including data processing and data statistics.

• In Section C, we provide preliminary knowledge about the FLUX [33] model, as well as how LayeredRefAttention modules are applied in the Single Stream DiT Block.

• In Section D, we provide additional details about the experimental setup, including baseline training, data augmentation, and hyperparameters.

• In Section E, we present a multitude of ShoeFit generated images, including more ablation results, more comparisons with the baselines, with additional results displayed in challenging scenarios.

## A Limitations and Future Work

We propose ShoeFit, a dual-stream DiT framework that addresses the critical challenges of viewpoint misalignment and backgroundinduced color distortion in VFTON by Multi-View Conditioning and LayeredRefAttention modules. However, certain limitations persist. Firstly, similar to the common constraints faced by existing generative models, the model occasionally struggles to accurately render small logos and intricate text due to their small portion in images and high-frequency variations. We illustrate this limitation in Fig. 7. These small logos are particularly critical for e-commerce sellers as they often convey brand information that consumers care about. Therefore, we plan to develop a detailed supplementary conditioning method in future research to pre-detect and enhance the injection of such patterns, fundamentally addressing this issue. Secondly, we aim to introduce explicit 3D geometric and material priors in the future to achieve more robust multi-view representation and refined visual fidelity.

![](images/4ce2b0f4a3650519b47f3cbc2f46d4a2f6176d3602ea1bb47f34e83799de6ce6.jpg)  
Figure 7: Similar to the common constraints faced by existing generative models, ShoeFit occasionally struggles to accurately render small logos and intricate text due to their small portion in images and high-frequency variations.

## B MVShoes Dataset Supplement

In the main body of the paper, we provide a brief description of the data processing pipeline and a rough visual presentation of the dataset statistics in terms of Rich Scenes and Diverse Footwear Categories due to space constraints. In this section, we offer more detailed information on the data processing pipeline to ensure reader comprehension. Additionally, we present the specific distribution statistics of MVShoes in a quantitative table format.

Data Processing Details. Given a dataset comprising raw images of various shoes and human models, we employ Qwen2.5-VL [3], which operates with 7 billion parameters, to distinguish between footwear images and human model images. The prompting framework utilized is as follows: “Analyze the provided image and determine whether it depicts a model image or a shoe image. A model image is defined as one that portrays a human model’s lower body or legs adorned with shoes, while a shoe image solely comprises images of shoes without any human presence. Assign a value of ‘1’ if it qualifies as a model image and ‘0’ if it does not. ”

Following this classification, we extract image features from the shoe images using the CLIP [53] model, implementing a similarity threshold of 0.9 to effectively eliminate duplicate images within the dataset. Subsequently, we perform segmentation by SAM [31] to isolate the shoe regions in each image, aligning the DINO [49] features to filter out shoes-model data pairs exhibiting inner similarities greater than 0.8.

This procedure facilitates the initial construction of human-footwear try-on triplets, which are then subjected to manual filtering to address any potential errors or oversights. Additionally, we exclude visually blurred images and compile statistical analyses pertaining to shoe categories and try-on scenarios. In accordance with methodologies established in DWPose [67] and GroundingDINO [43], we extract foot poses and shoe masks, ultimately resulting in the generation of high-resolution, category-comprehensive try-on triplets. All models referenced above are open-source, and their respective URLs are provided as follows:

• Qwen2.5-VL-7B [3]: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

• SAM [31]: https://github.com/facebookresearch/segment-anything

• CLIP [53]: https://github.com/openai/CLIP

• DINO [49]: https://github.com/facebookresearch/dinov2

• DWPose [67]: https://github.com/IDEA-Research/DWPose

• GroundingDINO [43]: https://github.com/IDEA-Research/GroundingDINO

Dataset Statistics. The dataset features diverse shoe categories and rich human scenes. We present the quantitative distribution statistics of MVShoes in Tab. 4 and Tab. 5. We also provide more visual samples from MVShoes datasets in Fig. 8.

Pose Estimation for Shoe Try-On. Traditional open-source pose estimation methods (e.g., Open-Pose, DWPose) in VITON pipeline, struggle to capture lower-body/foot poses in VFTON, which often only include partial leg shots or full lower-body views. To address this, we fine-tune the dwpose model on specially collected datasets containing diverse lower-body and close-up leg poses. This enables our model to generate accurate pose descriptions for various leg postures, significantly enhancing its generalization capability for shoe-related applications.

Generalizable Foot Mask Design. Given the diversity of shoe types (e.g., sneakers, boots, sandals) and their varying leg coverage, we propose a leg-pose-aware mask generation strategy. Unlike VTON’s body-centric masks, our method combines leg pose information with bounding box expansion from shoe detection to create adaptive masks. This allows seamless cross-shoe category try-on capabilities, for example, a model wearing long boots or flip-flops can be transformed into sneakers while maintaining anatomical alignment and realistic occlusion. As illustrated in the first row in Fig. 11, this approach lays the foundation for robust, universal shoe try-on systems.

However, our mask generation strategy is not without its limitations. A failure case is presented in the second row of Fig. 5(b). Here, the region of the rear shoe is both heavily occluded and minimally visible, which leads to the failure of our keypoint and shoe detection modules. Consequently, this rear shoe area is erroneously classified as background, and only the front foot is designated as the inpainting region. This incomplete mask ultimately results in an unsuccessful virtual try-on. We contend that this is an inherent challenge for all methods reliant on an explicit masking strategy, and developing more robust solutions to handle such severe occlusion cases remains an important direction for our future work.

Shoe Expansion Phenomenon. The volumetric expansion of footwear observed during the try-on process is a naturally occurring phenomenon. In real-world datasets, it is common to find that shoes with stiffer tongues tend to bulge outwards when worn, resulting in a slight expansion of the shoe’s overall volume. This effect is illustrated in Fig. 5, where the third column clearly shows a prominent bulge in the tongue area, leading to body expansion. This expansion occurs stochastically in real-world data pairs, as exemplified in the third row of Fig. 1(a), where the shoe upper in the second column exhibits a slightly thicker contour. Consequently, by effectively fitting to the training data distribution, our model learns to replicate this real-world phenomenon, accurately rendering the corresponding deformations inherent to the try-on process.

Table 4: Shoe Category Statistics. We report the sample count for each subcategory, with its proportion indicated in parentheses.
<table><tr><td rowspan=1 colspan=1>Primary Categories</td><td rowspan=1 colspan=1>Subcategory</td></tr><tr><td rowspan=1 colspan=1>Casual Shoes: 3158 (43.23%)</td><td rowspan=1 colspan=1>Lifestyle Casual Shoes: 803 (10.99%)Sneakers: 1428 (19.55%)Canvas Shoes: 141 (1.93%)Children Casual Shoes: 99 (1.36%)Men Casual Shoes: 687 (9.40%)</td></tr><tr><td rowspan=1 colspan=1>Athletic Shoes: 558 (7.64%)</td><td rowspan=1 colspan=1>Running Shoes: 322 (4.41%)Basketball Shoes: 121 (1.66%)Training Shoes: 45 (0.62%)Football Shoes: 70 (0.96%)</td></tr><tr><td rowspan=1 colspan=1>High-Heel Shoes: 895 (12.25%)</td><td rowspan=1 colspan=1>Classical High-heel Shoes: 340 (4.65%)Ladies Casual Shoes: 401 (5.49%)Mary Jane Shoes: 154 (2.11%)</td></tr><tr><td rowspan=1 colspan=1>Boots: 1045 (14.31%)</td><td rowspan=1 colspan=1>Ankle Boots: 416 (5.69%)Chelsea Boots: 56 (0.77%)High Boots: 153 (2.09%)Snow Boots: 183 (2.51%)Martin Boots: 237 (3.24%)</td></tr><tr><td rowspan=1 colspan=1>Sandals: 279 (3.82%)</td><td rowspan=1 colspan=1>Flip Flops: 183 (2.51%)Beach Sandals: 36 (0.49%)Strap Sandals: 60 (0.82%)</td></tr><tr><td rowspan=1 colspan=1>Dress Shoes: 389 (5.33%)</td><td rowspan=1 colspan=1>Loafers: 209 (2.86%)Formal Leather Shoes: 180 (2.46%)</td></tr><tr><td rowspan=1 colspan=1>Slippers: 981 (13.43%)</td><td rowspan=1 colspan=1>Clogs: 82 (1.12%)Thong Slippers: 18 (0.25%)House Slippers: 881 (12.06%)</td></tr></table>

Table 5: Human scene statistics.
<table><tr><td>Scene Types</td><td>Number of samples</td><td>Percentage</td></tr><tr><td>Top-down Foot</td><td>1719</td><td>23.53%</td></tr><tr><td>Horizontal Foot</td><td>4330</td><td>59.27%</td></tr><tr><td>Half-body Model</td><td>833</td><td>11.40%</td></tr><tr><td>Full-body Model</td><td>423</td><td>5.79%</td></tr></table>

## C Method Supplement

## C.1 Preliminary

FLUX.1 Our ShoeFit is an extension of Stable Diffusion 3 [55] and FLUX.1 [33], which are the most commonly used text-to-image diffusion models based on Flow Matching[41] and DiT[51]. FLUX.1 [33] employs a variational autoencoder [30] (VAE) that consists of an encoder E and a decoder D to enable image representations in the latent space. It is also equipped with rotary positional embeddings (RoPE)[58] and denoise-text-stream attention layers to improvement performance. FLUX.1 implements an actual two-dimensional RoPE scheme for encoding spatial positions in the latent space:

$$
\omega _ { d } = \frac { 1 } { \theta ^ { 2 d / D } } , \mathrm { f o r } D = 0 , 1 , . . . , D / 2 - 1 ,\tag{9}
$$

where $\theta$ is typically set to 10000. The position encoding applies a rotation matrix:

$$
\begin{array} { r l } { \left[ \cos ( \omega _ { d } \cdot \mathbf { p o s } ) \right. } & { { } - \sin ( \omega _ { d } \cdot \mathbf { p o s } ) } \\ { \left[ \sin ( \omega _ { d } \cdot \mathbf { p o s } ) \right. } & { { } \cos ( \omega _ { d } \cdot \mathbf { p o s } ) } \end{array}\tag{10}
$$

This rotation is applied to query and key vectors in the attention mechanism, enabling the model to capture relative positional relationships in the latent space.

Flow Matching Flow matching [41] aligns the flow of information between noise ϵ and data distributions by optimizing a velocity field $u _ { t }$ , which progressively converts noise into data over time. This technique ensures that the generative model maps the noise distribution to the actual data distribution in a structured manner. The text encoders [53] τ<sub>θ</sub> are employed to deal with the given text prompt y. The flow matching loss is defined as follows:

$$
\mathcal { L } = E _ { t , p _ { t } ( z | \epsilon ) , p ( \epsilon ) , y } \left[ \left| | v _ { \Theta } ( z , t , \tau _ { \theta } ( y ) ) - u _ { t } ( z | \epsilon ) | \right| ^ { 2 } \right] .\tag{11}
$$

In this context, $v _ { \Theta } ( z , t , \tau _ { \theta } ( y ) )$ ) signifies the conditional velocity field determined by the weights of the neural network, while $u _ { t } ( z | \epsilon )$ represents the vector field created by the model to delineate the probabilistic trajectory between the noise and actual data distributions. The symbol E stands for the expectation, which involves either integration or summation over time t, latent variables z, conditions $y ,$ and noise $\epsilon .$ This expectation computes the mean of the squared differences for all conditions, ensuring that the model’s performance is evaluated over numerous instances to yield a dependable estimate of its generative capability.

## C.2 LayeredRefAttention in Single-stream DiT Block

The original FLUX is a text-to-image model composed of a series of stacked MM-DiT (doublestream) blocks, followed by a series of stacked single-stream DiT blocks. Due to space constraints, the framework of the LayeredRefAttention module shown in the main text is its structure within the double-stream DiT blocks. Here, we provide how it is used within a single-stream DiT block in Fig. 10. Similar to the way in double-stream blocks, we employ a Squeeze-and-Excitation (SE) block [24] followed by global average pooling over spatial dimensions to compute channel-specific weights:

$$
P ^ { ( n ) } = A v g P o o l ( S E ( C _ { I } ^ { ( n ) } ) \in \mathbb { R } ^ { b \times c } .\tag{12}
$$

Subsequently, we employ two linear layers $F ( \cdot )$ to extract background modulation parameters and modulate foreground shoe features, filtering irrelevant environmental lighting and background reflection to ensure faithful material preservation of shoes as:

$$
\beta _ { s c a l e } ^ { ( n ) } , \beta _ { s h i f t } ^ { ( n ) } = F ( P ^ { ( n ) } ) , \quad C _ { f g } ^ { ( n ) } = ( 1 + \beta _ { s c a l e } ^ { ( n ) } ) \cdot ( L N ( C _ { I } ^ { ( n ) } \odot M _ { f g } ^ { ( n ) } ) ) + \beta _ { s h i f t } ^ { ( n ) } ,\tag{13}
$$

where $L N ( \cdot )$ means layer normalization and ${ M } _ { f g } ^ { ( n ) }$ represents the binary shoe masks for conditioning product image. The subsequent operations are then performed as in regular single-stream DiT blocks.

## D Implementation Supplement

## D.1 Baseline Training Details

We here provide detailed descriptions of the training processes for the baseline models, focusing specifically on the training configurations. All models are trained on MVShoes at a resolution of 768 × 768, utilizing 6305 pairs for training and 1000 pairs for testing.

For Flux.1 Fill [34], we implement the training and inference pipeline by concatenating footwear images with human model images. In the trainable components, we apply LoRA [23] with a rank of 64 to all attention modules in the model and a Flux ControlNet [71] comprising 6 single layers and 6 double layers, injecting the poses at every denoising step. The model is trained for 6 days on 8 80GB-A100 GPUs using DeepSpeed ZeRO-2, with a batch size of 4. We utilize the AdamW optimizer, setting a constant learning rate of 3e-5 for training, and operate the model on a single A100 GPU for 25 steps during inference.

![](images/68fd5b0a00dd5a1f3cbee3464f9546a8f6930e0d9c1f8e32902362f1e577a49a.jpg)  
Figure 10: We provide how LayeredRefAttention is used within a single-stream DiT block.<sup>4</sup>

For Flux.1 Redux [35], we adopt the same training settings as those used for FLUX.1-Fill. Additionally, we incorporate two linear layers for

projecting SigLip [70] features of the product shoe images, training these two layers concurrently with the attention LoRAs and ControlNet. The parameters of SigLip remain frozen throughout the training process. We retain the same inference settings as employed in FLUX.1-Fill.

For OOTDiffusion [65], we execute the training and inference pipelines based on the official code. We also implement ControlNet to facilitate pose injection. The model is trained for 6 days on 4 80GB-A100 GPUs with DeepSpeed ZeRO-2, at a batch size of 8. During inference, the model is run on a single A10 GPU for 25 steps.

For CatVTON-Flux-Lite-2V, we maintain the text stream for FLUX-lite and concatenate footwear images with human model images. The model undergoes training for 7 days on 12 80GB-A100 GPUs utilizing DeepSpeed ZeRO-2, with a batch size of 2. All other training and inference configurations align with those of ShoeFit-2V.

## D.2 Data Augmentation

We have implemented data augmentation techniques that could potentially enhance the model’s generalization ability as well as its fidelity performance. Specifically, the data augmentation operations include (a) horizontal flipping of images, (b) resizing footwear and human figures through padding (up to 10% of the image size), (c) randomly adjusting the image’s hue within a range of -5 to +5, and (d) randomly adjusting the image’s contrast within a specified range (between 0.8 and 1.2 times the original contrast). Each of these operations occurs independently with a 50% probability. Moreover, these operations are simultaneously applied to both the footwear and model images.

## D.3 LayeredRefAttention Hyperparameters

In the LayeredRefAttention layer, we primarily introduced a new Linear layer and an SE block. The Linear layer for the foreground follows the same dimensions as other linear layers within the module, which is the hidden size. The SE block takes input features with the same dimensions of hidden size, and internally, we use a compression rate of reduction = 4 for SE Block, as Linear(channels, channels/4), ReLU(), Linear(channels/4, channels).

## E More visual results

Fig. 9 provides more results about the ablation study. We highlight the improvements by all three components of the method in red.

Fig. 11 provides more results on MVShoes for comparisons between the baselines and ShoeFit. For a fair comparison, we report the results of the single-view conditioning version of all methods. Our method substantially improves rendering fidelity and robustness under challenging real-world product shoes, establishing a new benchmark in high-fidelity footwear try-on synthesis.

Fig. 12 provides more results on MVShoes for inspection to demonstrate that ShoeFit synthesizes high-fidelity and detail-faithful try-on results.

Fig. 13 details two key limitations of our model. First, in the early training phase, the model has not yet mastered spatial reasoning, resulting in structurally distorted outputs (row (a)). Second, generating precise masks for tall footwear remains a challenge. As shown in the row (b), inaccurate masks for items like boots can cause significant visual artifacts or fitting errors.

Fig. 14 illustrate more results on out-of-distribution (OOD) samples not present in the MVShoes dataset. The high-quality try-on results demonstrate the model’s notable stability and strong generalization capabilities.

![](images/55de8dd5b2d2b47211a9b674c43d282c00aa167a521307f2688ddd1e34054e0e.jpg)  
Figure 8: We provide more visual samples from MVShoes datasets.

Conditions  
Ablation  
Full  
Conditions  
Ablation  
Full  
![](images/5339f03015284ec418abee5f5e7890500c1be2379835be5a41aca4959da137b3.jpg)  
Figure 9: More visual results of the ablation study. We highlight the improvements by all three components of the method in red. Best viewed when zoomed in.

Input product shoes

FLUX.1-Fill-Redux

Input human feet

ShoeFit (Ours)  
CatVTON  
OOTDiffusion  
FLUX.1-Fill  
![](images/779825b1d15b05581030520d1a92a6baddd29a11adbdaf67f50fdcdfd27ed4dc.jpg)  
Figure 11: More visual comparisons on the MVShoes by ShoeFit. Best viewed when zoomed in.

Conditional shoes

Input

Generated

Conditional shoes

Input

Generated results

![](images/2acde09117584c4c6a09eeac363959fe7c8217e59d259c907756d3c4fa76fb81.jpg)  
Figure 12: More visual results on the MVShoes by ShoeFit. Best viewed when zoomed in.

![](images/13d8aab79314d9a4d0145af93f0aa8ae2e2f1b5ba08f04c4d23abaa44b8cb0a8.jpg)  
Figure 13: More visual results on two primary failure modes of our method: (a) structural failure observed during early training stages, and (b) artifacts arising from inaccurate masks in long footwear scenarios.

![](images/996f84efe1f1ab8078a4b79fcb167e016bea0809dbe0f431b44baa953e73b768.jpg)  
Figure 14: More visual results on out-of-distribution (OOD) samples not present in the MVShoes dataset. The high-quality try-on results demonstrate the model’s notable stability and strong general ization capabilities.