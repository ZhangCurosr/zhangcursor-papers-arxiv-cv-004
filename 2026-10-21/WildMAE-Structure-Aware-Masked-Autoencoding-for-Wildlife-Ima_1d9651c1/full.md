# WildMAE: Structure-Aware Masked Autoencoding for Wildlife Image Reconstruction

Yicheng Zhang Yujie Jin Ke Zhang Jiaxin Wu

The Chinese University of Hong Kong, Shenzhen

## Abstract

Camera-trap imagery presents substantial variation in animal appearance, illumination, back ground clutter, and visual degradation. Masked autoencoders (MAEs) ofer a label-free framework for modeling such data through reconstruction of randomly hidden image patches. We present WildMAE, a decoder-side extension of MAE that combines local structure-aware refinement with masked-to-visible cross-attention. A zero-gated convolutional residual path introduces a two-dimensional neighborhood prior, while gated cross-attention allows masked positions to retrieve contextual information from visible-position features without replacing the original Transformer decoder. The associated training recipe progressively introduces normalized-pixel, Sobel-edge, and perceptual objectives. In the original Snapshot Serengeti study, covering 4,744 test images at 75% masking, the reported reconstruction scores were 27.90 versus 24.33 dB PSNR, 0.840 versus 0.780 SSIM, and 0.160 versus 0.320 LPIPS for WildMAE and vanilla MAE, respectively. Additional historical records document favorable average diferences under four synthetic degradation conditions and describe component variants. We present the method, experimental observations, and their scope together: the retained metrics are historical protocol-dependent results, rather than a newly reproduced, harmonized comparison or a validation of restoration from genuinely unavailable pixels. WildMAE provides a practical architectural case study for combining global token mixing with spatial and targeted contextual refinement in wildlife masked-image modeling.

Keywords: masked autoencoders; wildlife camera traps; image reconstruction; cross-attention;   
structural refinement.

## 1 Introduction

Camera traps make it possible to observe wildlife at scale without continuous human presence, but the resulting images vary substantially in illumination, background, object scale, occlusion, and motion. In resources such as Snapshot Serengeti [1], even adjacent frames can depict challenging combinations of animals and cluttered natural scenes. This variability motivates image modeling approaches that can exploit visual structure without relying exclusively on species or behavior annotations. Masked image modeling is particularly appealing: a network receives a sparse set of visible patches and learns to predict those withheld during training.

Masked autoencoders (MAEs) [2] use an asymmetric Vision Transformer (ViT) [3] encoder– decoder, processing only visible patches in the encoder and reconstructing masked targets with a lightweight decoder. The full-sequence decoder already integrates positional embeddings and global self-attention; however, its patch sequence need not explicitly prioritize local two-dimensional adjacency after token mixing. For wildlife imagery, spatial continuity and targeted access to the remaining scene may be useful when predicting silhouettes, contours, and textured regions from sparse observations. This motivates an additional decoder-side inductive bias rather than a replacement of the MAE pathway.

We study WildMAE, which retains the global MAE decoder and appends two complementary refinements (Figure 1). First, a zero-gated local convolutional block processes decoder patch tokens on their restored spatial grid. Second, masked-position queries attend directly to features at visible positions, followed by a separately gated feed-forward update. In the accompanying training design, pixel reconstruction is established before edge and perceptual supervision are introduced. The intent is to combine the global context of self-attention, the locality of spatial convolution, and information retrieval targeted at masked regions.

The original project investigated this design on a 47,349-image checked subset of Snapshot Serengeti, reporting a 75%-masking comparison, component variants, and evaluations with synthetic degradations. The present manuscript turns that study into a self-contained method and results account while retaining the original measurements and specifying the evaluation setting more precisely. In particular, it distinguishes normalized-patch prediction from RGB reconstruction that uses full-input patch statistics, an important distinction when interpreting the image-quality scores.

The contributions of this work are:

• A decoder refinement architecture combining a local spatial residual branch and masked-to-visible cross-attention while preserving the original MAE decoder pathway.

• An explicit description of the zero-gated refinement updates and a progressive pixel–edge– perceptual training formulation.

• An organized presentation of the original wildlife-image reconstruction, component, and syntheticdegradation observations, together with clear boundaries on their interpretation and reproducibility.

## 2 Related Work

Wildlife camera-trap vision. Snapshot Serengeti provides a large, annotated collection of camera-trap photographs [1]. Related work has addressed animal identification, counting, and description [4], as well as filtering empty frames [5]. Such recognition pipelines and masked reconstruction address diferent questions: reconstructing withheld image regions is an imagemodeling task, not direct evidence of improved ecological recognition or a substitute for observations of real animals.

Masked image modeling. ViTs model image patches as tokens [3], and MAE reconstructs randomly masked patches from visible context [2]. CrossMAE studies patch dependence using masked queries and visible features for eficient decoding [6]. WildMAE difers in retaining the MAE full-token self-attention decoder and adding a separate masked-query refinement afterward; consequently, its design should not be interpreted as inheriting CrossMAE’s computational-eficiency claims. MCMAE [7] combines convolution and masked autoencoding, underscoring that local processing is not itself a novel ingredient. The distinction here is the placement of a local spatial branch after global decoder processing, followed by visible-to-masked feature retrieval. HAP [8] introduces human-part-based structure priors; WildMAE instead uses a generic patch-grid prior without part labels.

Training objectives and perceptual quality. Pixel reconstruction measures target fidelity, whereas Sobel gradients emphasize local image boundaries. SSIM [9] and learned perceptual image patch similarity (LPIPS) [10] characterize complementary aspects of visual agreement. Progressive supervision in WildMAE changes the terms of the reconstruction objective; it is distinct from learning a masking curriculum as in CL-MAE [11]. Because LPIPS is also a training component in the final stage, LPIPS evaluation is best interpreted alongside, rather than instead of, pixel and structural measurements.

## 3 Proposed Method

![](images/aba52c849ce002b70f318cf42969e0cab5c107d269f41e8898491ba8de739db5.jpg)  
Figure 1: High-level WildMAE pipeline: the original MAE encoder and Transformer decoder are retained, followed by a local spatial refinement, masked-to-visible contextual refinement, and the pixel-prediction head. Exact internal normalizations and residual gates are specified in Sections 3.2–3.3. The diagram is conceptual.

## 3.1 Overview and MAE backbone

Let $I \in \mathbb { R } ^ { 2 2 4 \times 2 2 4 \times 3 }$ be an RGB image. Patch size $1 6 \times 1 6$ yields $N = 1 4 \times 1 4 = 1 9 6$ patches of dimension $D = 1 6 ^ { 2 } \cdot 3 = 7 6 8$ . For a masking ratio $r = 0 . 7 5$ , the index sets of visible patches V and masked patches M have sizes 49 and 147. The retained ViT-B encoder contains 12 blocks with width 768 and 12 attention heads. It processes visible tokens, after which learned mask tokens and positional embeddings restore a full-sequence decoder input. The decoder provides a global context representation that the proposed refinements further process.

As in normalized-pixel MAE training, let $\mathbf { p } _ { i }$ denote a channel-normalized input patch and write its per-patch mean and variance as $\mu _ { i }$ and $s _ { i } ^ { 2 }$ . The prediction target and masked-region pixel objective are

$$
\widetilde { \mathbf { p } } _ { i } = \frac { \mathbf { p } _ { i } - \mu _ { i } } { \sqrt { s _ { i } ^ { 2 } + \epsilon } } , \qquad \mathcal { L } _ { \mathrm { p i x } } = \frac { 1 } { | \mathcal { M } | D } \sum _ { i \in \mathcal { M } } \left\| \widehat { \mathbf { p } } _ { i } - \widetilde { \mathbf { p } } _ { i } \right\| _ { 2 } ^ { 2 } , \quad \epsilon = 1 0 ^ { - 6 } .\tag{1}
$$

The normalization is used to construct the supervision target. Interpretation of the separate RGB reconstruction step is discussed in Section 4.2.

## 3.2 Structure-aware local refinement

Let $X \in \mathbb { R } ^ { N \times C }$ be the restored patch-token sequence after the MAE decoder blocks, excluding the class token. A reshape operator R maps patch tokens to their $C \times 1 4 \times 1 4$ grid. The local branch applies pre-normalization, a depthwise $3 \times 3$ convolution, and two pointwise projections with a GELU nonlinearity:

$$
\begin{array} { r } { F ( X ) = \mathcal { R } ^ { - 1 } [ P _ { 2 } ( \mathrm { G E L U } ( P _ { 1 } ( \mathrm { D W C o n v } _ { 3 \times 3 } ( \mathcal { R } ( \mathrm { L N } ( X ) ) ) ) ) ] , } \\ { S = X + \gamma _ { s } F ( X ) . \qquad } \end{array}\tag{2}
$$

(3)

The depthwise convolution uses padding one; the pointwise projections expand from $C$ to $2 C$ and return to $C .$ . The learnable scalar gate $\gamma _ { s }$ is initialized to zero, so the additional local branch initially preserves its input. This property refers to the added residual branch, not to equivalence between diferently sized complete decoders. Explicitly restoring the two-dimensional token layout encourages local neighborhood interactions while preserving information already combined by global self-attention.

![](images/774a5b78d602cfc5938750e8b8b945501e0087487f922c448fe32200e982f738.jpg)  
Figure 2: Conceptual illustration of the two appended refinement operations, adapted from the original project report. The actual implementation uses depthwise convolution and pre-normalization in the local branch, plus separately zero-gated cross-attention and MLP updates in each cross-refinement block (Equations 2–6).

## 3.3 Masked-to-visible cross-attention

After local refinement, use the restored mask to gather masked-token features $S _ { \mathcal { M } }$ and visibleposition features $S _ { \nu }$ . Define $U ^ { ( 0 ) } = S _ { \mathcal { M } }$ and $Z = S _ { \nu }$ . At refinement block $\ell ,$ masked tokens form queries and visible-position tokens form keys and values:

$$
\begin{array} { r } { A ^ { ( \ell ) } = \mathrm { M H A } \left( \mathrm { L N } _ { q } ( U ^ { ( \ell ) } ) , \mathrm { L N } _ { k v } ( Z ) , \mathrm { L N } _ { k v } ( Z ) \right) , } \end{array}\tag{4}
$$

$$
T ^ { ( \ell ) } = U ^ { ( \ell ) } + \gamma _ { a } ^ { ( \ell ) } A ^ { ( \ell ) } ,\tag{5}
$$

$$
\begin{array} { r } { U ^ { ( \ell + 1 ) } = T ^ { ( \ell ) } + \gamma _ { f } ^ { ( \ell ) } \mathrm { M L P } \big ( \mathrm { L N } _ { f } ( T ^ { ( \ell ) } ) \big ) . } \end{array}\tag{6}
$$

The attention operation includes learned query/key/value/output projections, and the feed-forward network has a fourfold hidden-width expansion with GELU. The attention and MLP residual gates, $\gamma _ { a } ^ { ( \ell ) }$ and $\gamma _ { f } ^ { ( \ell ) }$ , are independently initialized to zero. The retained constructors define two refinement blocks with eight heads, though the surviving files do not identify every historical checkpoint with equal precision. Visible-position features stay fixed within this stack, while the updated masked features are scattered back to their original token indices. After reattaching the class token, final decoder normalization and a prediction head produce normalized patch values.

Notably, Z comes from the globally mixed and locally refined decoder representation: it is not a set of untouched raw image patches. The local and targeted branches therefore complement, rather than substitute for, full-token decoder processing. Figure 2 summarizes the conceptual information flow.

## 3.4 Progressive structural–perceptual supervision

The original training plan has three phases. Stage 1 optimizes $\mathcal { L } _ { \mathrm { p i x } }$ for 160 epochs. Stage 2 adds edge supervision for 20 epochs. Stage 3 includes both the edge term and an LPIPS term for another 20 epochs:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { p i x } } + \lambda _ { e } \mathcal { L } _ { \mathrm { e d g e } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p e r c } } , } \end{array}\tag{7}
$$

with planned coeficients $( \lambda _ { e } , \lambda _ { p } ) = ( 0 , 0 ) , ( 0 . 0 2 , 0 )$ , and (0.02, 0.002) in the respective phases. These are the weights specified by the staged pipeline; other generic scripts have diferent defaults and are not evidence that those defaults produced the main reported checkpoint.

For edge and perceptual losses, retained fine-tuning code converts predicted normalized patches into RGB and creates a composite bI from visible input pixels and predicted masked pixels. With pixel mask m equal to one in masked regions, the Sobel magnitude of the channel-averaged image is $G ( I )$ and the masked edge term is

$$
\mathcal { L } _ { \mathrm { e d g e } } = \frac { \sum m \left| G ( \widehat { I } ) - G ( I ) \right| } { \sum m } .\tag{8}
$$

The perceptual term replaces visible pixels with gray in both inputs before measuring AlexNet-based LPIPS:

$$
C _ { m } ( J ) = m \odot J + ( 1 - m ) \odot 0 . 5 , \qquad \mathscr { L } _ { \mathrm { p e r c } } = \mathrm { L P I P S } _ { \mathrm { A l e x N e t } } \big ( 2 C _ { m } ( \widehat { I } ) - 1 , 2 C _ { m } ( I ) - 1 \big ) .\tag{9}
$$

The perceptual network is frozen during fine-tuning; gradients with respect to its reconstructedimage input remain available. This staged design balances pixel agreement with progressively introduced structural and perceptual constraints, without requiring species-class labels as training targets.

## 4 Experimental Setup

## 4.1 Dataset, preprocessing, and training

The study planned a 50,000-image subset of Snapshot Serengeti Season 1 and retained 47,349 checked image IDs, comprising 37,903 training, 4,702 validation, and 4,744 test records. Input images are center-cropped to a square and resized to $2 2 4 \times 2 2 4$ for evaluation; pretraining additionally uses random resized crops and horizontal flips. The checked subset includes animal and empty-frame images. Training and the primary reconstruction study use a fixed 75% random patch-masking ratio.

The reported optimizer is AdamW with weight decay 0.05. The first stage specifies base learning rate $1 0 ^ { - 3 } ;$ , 40 warm-up epochs, and efective batch size 64 (batch 8 with eight-step accumulation); the saved implementation scales this base learning rate by efective batch size over 256. Later stages specify nominal learning rates $2 \times 1 0 ^ { - 5 }$ and $1 0 ^ { - 5 }$ with two warm-up epochs, batch size four, and 16-step accumulation. This account describes the original recorded training recipe, not a fresh rerun. The original report identifies an NVIDIA RTX 5090 as the training hardware.

## 4.2 Model variants and evaluation protocol

The retained implementation contains a standard WildMAE constructor with decoder width 512 and depth 8, as well as a V4Strong constructor with width 768 and depth 16. The corresponding retained vanilla MAE constructor has width 512 and depth 8. All use the same ViT-B encoder family. Their existence documents implemented variants; the surviving materials do not establish a complete checkpoint-to-row mapping for the headline comparison. Consequently, Table 1 is retained as the original report’s result, not presented as a newly generated capacity-matched ablation.

The original evaluation reports PSNR (higher is better), SSIM (higher is better), and LPIPS (lower is better). To translate normalized predictions back into image-space values, the retained reconstruction code uses

$$
\widehat { \mathbf { p } } _ { i } ^ { \mathrm { R G B } } = \widehat { \mathbf { p } } _ { i } \sqrt { s _ { i } ^ { 2 } + \epsilon } + \mu _ { i } ,\tag{10}
$$

where $\mu _ { i }$ and $s _ { i } ^ { 2 }$ are measured on the full corresponding input patch. Thus the image-space scores characterize input-statistic-assisted masked reconstruction; they are not a test of RGB recovery when hidden-patch statistics are genuinely unavailable. The preserved baseline and WildMAE evaluator implementations also difer in aggregation and composite-image construction. We therefore distinguish historical reported numbers from results that would require a newly harmonized common evaluator.

## 4.3 Scope of available records

The present manuscript uses the original course report, surviving source code, manifests, and retained evaluation summaries. Existing saved robustness CSVs permit arithmetic re-aggregation, but image files and model checkpoints were not present in the research package used for manuscript preparation. No new training or checkpoint-based inference was conducted for this manuscript. In particular, there are no newly calculated multi-seed confidence intervals or downstream wildliferecognition results.

## 5 Results

## 5.1 Overall reconstruction observations

Table 1 reproduces the primary comparison from the original report on the 4,744-image test split at nominal 75% masking. Its recorded arithmetic diferences are +3.57 dB PSNR, +0.060 SSIM, and −0.160 LPIPS for WildMAE relative to vanilla MAE. The reported values are consistent in direction across pixel, structural, and perceptual metrics, and provide the principal historical observation motivating the proposed refinement design. Because the original checkpoint identities and common evaluator configuration cannot now be established, these diferences are descriptive rather than an independently validated controlled estimate of the added modules’ efect.

Table 1: Original reported reconstruction results at nominal 75% masking on 4,744 test images. Historical course-report values; not newly reproduced under a common evaluator.
<table><tr><td>Model</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>Vanilla MAE</td><td>24.33</td><td>0.780</td><td>0.320</td></tr><tr><td>WildMAE</td><td>27.90</td><td>0.840</td><td>0.160</td></tr></table>

## 5.2 Component variants

Table 2 preserves the original project’s component comparison. The single local-refinement and cross-attention variants each show favorable reported scores relative to the original baseline row. Under the basic pixel loss, the dual-refinement row remains above baseline but does not consistently improve over both individual variants. The strongest reported row combines the dual-refinement model with the staged objective. This pattern motivates examining complementary structure, context, and supervision, but should not be taken as a capacity-matched causal decomposition: complete initialization, model-size, and evaluator provenance for these historical rows is not retained.

Table 2: Component-variant results as reported in the original project at 75% masking. The rows are descriptive historical records, not a newly verified matched-control series.
<table><tr><td>Variant</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>Vanilla MAE + basic pixel loss</td><td>24.33</td><td>0.780</td><td>0.320</td></tr><tr><td>Cross-attention refinement only + basic loss</td><td>25.90</td><td>0.813</td><td>0.217</td></tr><tr><td>Local structural refinement only + basic loss</td><td>25.73</td><td>0.812</td><td>0.224</td></tr><tr><td>WildMAE dual refinement + basic loss</td><td>25.75</td><td>0.810</td><td>0.221</td></tr><tr><td>WildMAE dual refinement + staged loss</td><td>27.90</td><td>0.840</td><td>0.160</td></tr></table>

## 5.3 Synthetic-degradation observations

The original project also evaluated clean inputs and four synthetic perturbations—Gaussian noise, low light, motion blur, and random occlusion—under 75% masking. Table 3 reproduces rounded mean diferences from the retained robustness records; their arithmetic aggregation was independently checked during manuscript preparation. The recorded means favor WildMAE in each condition, with the smallest PSNR separation under low light. These are diferences of saved evaluation means, not fully controlled paired-corruption estimates. Only 2,372 of 4,744 image records per condition share matching mask hashes across the two model outputs, and identical corrupted inputs cannot be established from the retained files.

Table 3: Descriptive diferences in retained degradation records at nominal 75% masking. ∆PSNR and ∆SSIM are WildMAE minus baseline; LPIPS reduction is baseline minus WildMAE. The original report rounded these quantities to three decimal places.
<table><tr><td>Input condition</td><td>∆PSNR↑</td><td>∆SSIM↑</td><td>LPIPS reduction ↑</td></tr><tr><td>Clean</td><td>+2.591</td><td>+0.075</td><td>0.095</td></tr><tr><td>Gaussian noise</td><td>+1.128</td><td>+0.031</td><td>0.052</td></tr><tr><td>Low light</td><td>+0.078</td><td>+0.033</td><td>0.018</td></tr><tr><td>Motion blur</td><td>+1.537</td><td>+0.026</td><td>0.033</td></tr><tr><td>Random occlusion</td><td>+0.228</td><td>+0.030</td><td>0.035</td></tr></table>

The clean row of Table 3 is not interchangeable with Table 1: the robustness pipeline computes its masked-region structural and perceptual scores diferently from the primary reconstruction scripts. Separate historical experiments with other model versions, masking ratios, or loss schedules are therefore not substituted into the primary comparison.

## 6 Discussion and Limitations

WildMAE’s methodological contribution is the combination of a preserved global Transformer decoding pathway with an explicitly local residual branch and a directed masked-query refinement. Zero-initialized gates allow the added branches to enter optimization gradually. The original results provide encouraging, complementary pixel- and perceptual-metric observations under the studied reconstruction setting, while the component table illustrates the relevance of studying the architecture and supervision jointly. These observations do not establish improved downstream encoder representations or a new state of the art.

Evaluation considerations. The retained RGB reconstruction converts normalized predictions using full-input patch statistics, including those of masked patches. Accordingly, the reported imagequality scores describe statistic-assisted reconstruction, not recovery of fully unavailable image content. Moreover, the available baseline and WildMAE evaluators difer in metric aggregation and masking/composite conventions, while some model variants have diferent decoder capacities. The archived degradation means are not based on fully matched masks and corruption realizations. Finally, 2,590 of the 4,744 test records share capture-sequence identifiers with training records and all test records share location identifiers; the original image-level split is not a held-out-site assessment. These limitations qualify the strength and generality of the historical comparisons without changing the measurements originally reported.

A focused next step would use a shared evaluator and deterministic image-indexed masks, compare capacity-matched decoders, and test recovery without hidden-patch statistics. Sequencedisjoint evaluation would further clarify generalization to independent camera-trap capture events. Such studies, as well as repeated-seed stability tests, computational cost measurements, and downstream classification or detection, remain future work rather than implied results of this manuscript.

## 7 Conclusion

We presented WildMAE, a structure-aware masked-autoencoding design that preserves global MAE decoding while adding zero-gated local refinement, masked-to-visible cross-attention, and progressive pixel–edge–perceptual supervision. The original Snapshot Serengeti report and retained experiment records describe favorable reconstruction-metric observations under heavy masking and several synthetic degradations. By specifying the implementation and evaluation scope, this manuscript provides a clear account of the proposed approach and a basis for further controlled validation. Future work should assess the method using a harmonized, leakage-free RGB recovery protocol and stronger generalization splits before extending its claims to real-world wildlife-image restoration.

## Research provenance and availability

This manuscript develops a four-author final project for CIE6032, Deep Learning Foundations and its Applications, at The Chinese University of Hong Kong, Shenzhen. It retains original reported measurements and adds manuscript-level clarification and static review of surviving materials; no new training or checkpoint-based inference was performed. A public repository of original project materials is available at https://github.com/Keke946/WildMAE-Wildlife-Image-Reconstruction; that repository does not supply the original image files or model checkpoints. Any code reuse must respect its third-party notices and applicable source licenses.

## AI assistance disclosure

Generative AI assisted with drafting, editorial restructuring, literature lookup, static code review, and organization of evidence. It is not an author. All listed human authors must review and take responsibility for the factual claims, permissions, and final manuscript before any public submission.

## References

[1] A. Swanson, M. Kosmala, C. Lintott, R. Simpson, A. Smith, and C. Packer. Snapshot Serengeti, high-frequency annotated camera trap images of 40 mammalian species in an African savanna. Scientific Data, 2:150026, 2015.

[2] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16000–16009, 2022.

[3] A. Dosovitskiy et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[4] M. S. Norouzzadeh, A. Nguyen, M. Kosmala, A. Swanson, M. S. Palmer, C. Packer, and J. Clune. Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning. Proceedings of the National Academy of Sciences, 115(25):E5716–E5725, 2018.

[5] F. Cunha, E. M. dos Santos, R. Barreto, and J. G. Colonna. Filtering empty camera trap images in embedded systems. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 2438–2446, 2021.

[6] L. Fu, L. Lian, R. Wang, B. Shi, X. Wang, A. Yala, T. Darrell, A. A. Efros, and K. Goldberg. Rethinking patch dependence for masked autoencoders. Transactions on Machine Learning Research, 2025. Also available as arXiv:2401.14391.

[7] P. Gao, T. Ma, H. Li, Z. Lin, J. Dai, and Y. Qiao. MCMAE: Masked convolution meets masked autoencoders. In Advances in Neural Information Processing Systems, vol. 35, 2022.

[8] J. Yuan et al. HAP: Structure-aware masked image modeling for human-centric perception. In Advances in Neural Information Processing Systems, vol. 36, 2023.

[9] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

[10] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 586–595, 2018.

[11] N. Madan, N.-C. Ristea, K. Nasrollahi, T. B. Moeslund, and R. T. Ionescu. CL-MAE: Curriculum-learned masked autoencoders. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2492–2502, 2024.