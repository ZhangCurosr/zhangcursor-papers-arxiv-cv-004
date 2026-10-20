# VHA: VQA-BASED HALLUCINATION ASSESSMENT FOR GENERATIVE IMAGE SUPER-RESOLUTION

Donghun Ryou<sup>2</sup> Inju Ha<sup>1</sup> Jaeyoo Park<sup>1</sup> Bohyung Han<sup>1,2</sup> Computer Vision Laboratory, <sup>1</sup>ECE & <sup>2</sup>IPAI, Seoul National University {dhryou, hij1112, bellos1203, bhhan}@snu.ac.kr

## ABSTRACT

Generative image super-resolution produces visually realistic high-resolution images, but may also introduce hallucinations by adding unsupported content or altering existing details. Because existing super-resolution metrics fail to measure whether generated details remain grounded in the reference image, we present a hallucination assessment technique based on bidirectional visual question answering (VQA), which we term VQA-based Hallucination Assessment (VHA). The proposed approach uses a vision-language model (VLM) to determine whether the semantic content and low-level visual evidence in a super-resolved image are supported by its ground-truth counterpart. To this end, it constructs question–answer pairs from each image and evaluates them on both images. Discrepancies between the resulting scores in both directions reveal missing or altered ground-truth information and unsupported additions. However, obtaining these scores requires computationally expensive VLM inference. We therefore develop VHA+, a learned, lightweight model that directly predicts VHA scores from ground-truth and super-resolved image pairs. We construct a hallucination benchmark, VHA-bench, through a human study and show that both VHA and VHA+ correlate highly with human judgments of hallucination. The code, benchmark, and model weights are available at https://github.com/dhryougit/Hallucination-Assessment.

## 1 INTRODUCTION

Generative image super-resolution (SR) can synthesize visually realistic details from low-resolution inputs by drawing on learned image priors (Wang et al., 2018; 2024; Wei et al., 2025; Wu et al., 2026). Since a low-resolution input can correspond to multiple high-resolution images, a plausible reconstruction may differ from the ground truth in its semantic content or visual attributes. An SR output may alter an object’s identity, remove meaningful details, or introduce a texture absent from the original image (Ren et al., 2025; Li et al., 2025b). Such discrepancies illustrate hallucinations in generative SR, which we define as the loss or alteration of ground-truth evidence in the super-resolved image, or the addition of content unsupported by the ground truth.

Existing SR evaluation metrics do not adequately reflect whether reconstructed content is faithful to the ground truth. Figure 1 illustrates these limitations with visual examples in which conventional metrics favor less faithful reconstructions. Distortion metrics such as PSNR and perceptual similarity metrics such as LPIPS (Zhang et al., 2018) quantify discrepancies in pixel values or learned visual features, but these differences do not necessarily reflect the significance of a content change. Perceptual quality metrics (Ke et al., 2021; Wang et al., 2023; You et al., 2025) can assign favorable scores to visually realistic outputs even when their content is unsupported by the ground truth.

We introduce VQA-based Hallucination Assessment (VHA), which compares how strongly the ground truth and the SR output support individual visual evidence. Prior work (Ren et al., 2025) produces a hallucination rating and its explanation, but does not explicitly quantify how individual content or attribute changes contribute to the rating. We represent these evidences as question–answer pairs about specific image content and attributes. If the image content is preserved, the same answer should remain supported under both images. Inspired by VQAScore (Lin et al., 2024b), we measure the changes by comparing the VLM’s answer log-probabilities across the two images on same question-answer pairs. We aggregate these probability differences into the VHA score.

17.11 75.18 0.4954 0.3488  
GT  
![](images/736bea7880c7e6d3d38b986e3d7acd8b21b78a5705b43fdc4dcfae8a66463b7f.jpg)

StableSR  
![](images/de2274117d7e2fd2901deb1d098f476a64bccbed415bec711e31f94fac3a614c.jpg)  
17.63 74.14 0.6382 0.8641

DiffBIR  
![](images/be30a43b4f0f9ced10b5084e11f0b01d538f58fd72bc918a28e826e8c4c3da7e.jpg)

GT  
![](images/e40c2275608effee789f6d5ef9b1c4671452f6e2a5adbc0338ad3910a6920e9e.jpg)  
DiffBIR  
PSNR ↑ MUSIQ ↑ VHA (Ours) ↓ VHA+ (Ours) ↓ Human eval.

![](images/31f679d863a93348634df6fc8fb1e803735722b8fea69b48553853f50c6418c1.jpg)  
PSNR ↑ MUSIQ ↑ VHA (Ours) ↓ VHA+ (Ours) ↓ Human eval.  
23.24 70.82 0.5872 1.5106

PURE  
![](images/0524d32351b0c57e1f0e8789f158c07811a412e35d5edecc8d20447a5c1099fe.jpg)  
23.14 67.51 0.5126 0.4791

Figure 1: Limitation of conventional evaluation metrics. In the left example, PSNR favors a reconstruction that incorrectly depicts the building as a rock formation. In the right example, even the perceptual quality metric, MUSIQ (Ke et al., 2021) favors a reconstruction that alters the knitted texture present in the ground truth. In contrast, VHA and VHA+ assess the hallucinations, which conventional metrics fail to capture.

VHA assesses hallucinations using question–answer pairs generated from both the ground-truth and SR output. Questions from the ground truth assess whether original content is preserved in the SR output, while questions from the SR output assess whether its content is supported by the ground truth. In both directions, taxonomy-guided questions cover high-level semantics, such as objects and text, and low-level visual attributes, such as color and texture. To reduce the cost of repeated VLM inference, we introduce VHA+, a lightweight model trained on image pairs labeled with VHA scores. VHA+ predicts these scores directly from ground-truth and SR image pairs without VLM inference.

We evaluate the alignment of VHA and VHA+ with human hallucination judgments. We construct VHA-bench, a human-annotated benchmark covering outputs from multiple generative SR methods. On this benchmark, VHA and VHA+ achieve higher correlations with human judgments than conventional SR metrics and direct VLM ratings using the same backbones. We also show that fine-tuning an SR model on VHA+-based supervision with an image-quality objective both reduces the VHA score and improves image quality metrics. Our main contributions are as follows:

• We introduce a bidirectional VQA-based hallucination assessment method (VHA) that compares answer log-probabilities for same question–answer pairs across ground-truth and SR outputs. Taxonomy-guided question–answer generation enables VHA to assess hallucinations across diverse semantic and visual aspects.

• We propose VHA+, a lightweight model trained to predict VHA scores without VLM inference, and demonstrate that its learned representations can guide the fine-tuning of a super-resolution model.

• We construct a human-annotated VHA-bench and demonstrate that VHA and VHA+ correlate stronger with human judgments than conventional SR metrics and direct VLM ratings.

## 2 RELATED WORKS

Evaluation Metrics for Super-Resolution. SR evaluation commonly considers distortion, perceptual similarity, and perceptual quality. Distortion metrics such as PSNR and SSIM (Wang et al., 2004) summarize pixel-wise reconstruction errors and local structural differences against a reference, respectively. LPIPS (Zhang et al., 2018) and DISTS (Ding et al., 2020) compare deep image representations, with DISTS explicitly combining structure and texture similarity. No-reference image quality assessment methods, including MUSIQ (Ke et al., 2021), CLIP-IQA (Wang et al., 2023), and DeQA-Score (You et al., 2025), estimate perceptual quality from the test image alone. A-FINE (Chen et al., 2025) combines reference-based comparisons with image naturalness estimates, extending quality prediction to settings where the reference itself may be imperfect.

Hallucination in Generative Image Super-Resolution. Generative SR models can produce visually realistic details while introducing semantic errors and spurious structures (Wu et al., 2024b;

2026). PURE (Wei et al., 2025) addresses this issue by integrating image understanding with autoregressive restoration to preserve semantic content and local structures. However, its evaluation does not separately quantify hallucinations. To directly assess hallucinations, Ren et al. (2025) introduce Hallucination Score, which prompts GPT-4o with low-resolution, super-resolved, and ground-truth images to produce an overall hallucination rating and an accompanying explanation. Although the explanation can identify specific discrepancies, their individual contributions to the overall rating are not explicitly quantified. VHA constructs its assessment signal by aggregating measured changes in support for individual visual evidences.

VQA-based Evaluation of Generated Images. Visual question answering has been used to assess the semantic alignment and factual correctness of generated images. TIFA (Hu et al., 2023) and $\mathrm { \Delta V Q ^ { 2 } }$ (Yarom et al., 2023) generate questions from textual descriptions and use VQA models to evaluate whether the specified content appears in an image. Davidsonian Scene Graph (DSG) (Cho et al., 2024) organizes atomic questions into dependency graphs to improve semantic coverage and answer consistency. VQAScore (Lin et al., 2024b) measures image–text alignment through the probability of a $\mathbf { \Delta } ^ { 6 6 } \mathbf { Y } \mathrm { e } \mathbf { s } ^ { \prime }$ answer to a question asking whether the image depicts the supplied text. I-HallA (Lim et al., 2025) evaluates factual correctness in text-to-image generation using VQA accuracy on curated multiple-choice questions informed by factual knowledge and visual evidence. Building on this question-based evaluation paradigm, VHA compares answer log-probabilities across GT and SR outputs using same question-answer pairs derived independently from each image.

## 3 VQA-BASED HALLUCINATION ASSESSMENT

We introduce VQA-based Hallucination Assessment (VHA) to quantify hallucinations in a superresolved image $( I _ { \mathrm { S R } } )$ using its ground-truth counterpart $( I _ { \mathrm { G T } } )$ as the reference. To examine diverse semantic content and visual attributes, we generate taxonomy-guided question–answer (Q/A) pairs from both images (Section 3.1) using a vision-language model (VLM). For each Q/A pair, we compare the VLM’s answer log-probabilities under the two images on same Q/A pairs. GT-derived pairs assess the loss or alteration of original evidence, while SR-derived pairs assess unsupported additions; we aggregate these probability differences into a bidirectional hallucination score (Section 3.2). Figure 2 illustrates our assessment pipeline.

## 3.1 TAXONOMY-GUIDED QUESTION–ANSWER GENERATION

We first generate Q/A pairs covering both high-level semantics and low-level visual attributes using a VLM. Without explicit guidance on what to examine, a VLM may focus its questions on certain types of content and attributes while overlooking others. We therefore use a taxonomy to guide Q/A generation across different types of visual evidence. The high-level categories comprise scene semantics, object semantics, and text or logos, while the low-level categories comprise color and tone, edges and shapes, and texture.

We organize Q/A generation around explicitly identified visual targets to limit repeated questions about the same content. Within each category, we first prompt the VLM to identify visible content or attributes. We then prompt it to generate one question and a short answer for each identified item. We apply this procedure independently to $I _ { \mathrm { G T } }$ and $I _ { \mathrm { S R } }$ to obtain the corresponding $\mathrm { Q } / \mathrm { A }$ sets, $\mathcal { Q } _ { \mathrm { G T } }$ and ${ \mathcal { Q } } _ { \mathrm { S R } }$ , respectively. The full prompts for both stages are provided in Appendix $\mathbf { A }$

## 3.2 BIDIRECTIONAL HALLUCINATION SCORING

We assess hallucinations by evaluating whether GT content is preserved in SR and whether SR content is supported by GT. Forward and reverse scoring use $\mathcal { Q } _ { \mathrm { G T } }$ and ${ \mathcal { Q } } _ { \mathrm { S R } }$ , respectively, to compare each same answer’s likelihood under both images. For a question $( q )$ and an answer consisting of $T$ tokens $( a = ( a _ { 1 } , \dots , a _ { T } ) )$ , we compute the mean token-level log-probability under a fixed VLM $( p _ { \phi } )$ in a teacher-forced manner:

$$
\ell ( I ; q , a ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \log p _ { \phi } ( a _ { t } \mid I , q , a _ { < t } ) .\tag{1}
$$

![](images/e1024e892a021207802f74c3b218c792b721fb88fe6409c492d3947818ae4921.jpg)  
Figure 2: Overview of our bidirectional VQA-based hallucination assessment. Taxonomy-guided question–answer pairs are generated from both ground truth and super-resolved images, and each answer is scored under both images to assess evidence preservation and unsupported content.

We define the answer score as the difference between the image-conditioned and text-only answer log-probabilities:

$$
s ( I ; q , a ) = \ell ( I ; q , a ) - \ell ( \emptyset ; q , a ) ,\tag{2}
$$

where ∅ denotes an input without an image. We then use each Q/A pair’s score on its source image to gate its contribution to the hallucination score.

To assess the preservation of original ground-truth content and attributes, we define the forward score F by aggregating score differences over GT-derived Q/A pairs:

$$
F ( I _ { \mathrm { G T } } , I _ { \mathrm { S R } } ) = \frac { 1 } { | { \mathcal Q } _ { \mathrm { G T } } | } \sum _ { ( q , a ) \in { \mathcal Q } _ { \mathrm { G T } } } \mathbf { 1 } [ s ( I _ { \mathrm { G T } } ; q , a ) > \tau _ { g } ] \cdot \big [ s ( I _ { \mathrm { G T } } ; q , a ) - s ( I _ { \mathrm { S R } } ; q , a ) \big ] .\tag{3}
$$

Here, 1[·] is the indicator function, and $\tau _ { g }$ is the GT answer-score threshold. For pairs passing the threshold, a decrease in the answer score from GT to SR contributes positively to $F .$ These contributions can offset one another.

GT-derived questions may overlook newly introduced content in the SR output. We therefore define the reverse score R to assess potentially unsupported additions by measuring decreases in answer scores from SR to GT for SR-derived Q/A pairs:

$$
R ( I _ { \mathrm { G T } } , I _ { \mathrm { S R } } ) = \frac { 1 } { | \mathcal { Q } _ { \mathrm { S R } } | } \sum _ { ( \boldsymbol { q } , \boldsymbol { a } ) \in \mathcal { Q } _ { \mathrm { S R } } } \mathbf { 1 } [ \boldsymbol { s } ( I _ { \mathrm { S R } } ; \boldsymbol { q } , \boldsymbol { a } ) > \tau _ { s } ] \cdot \big [ \boldsymbol { s } ( I _ { \mathrm { S R } } ; \boldsymbol { q } , \boldsymbol { a } ) - \boldsymbol { s } ( I _ { \mathrm { G T } } ; \boldsymbol { q } , \boldsymbol { a } ) - \boldsymbol { \tau } _ { m } \big ] _ { + } ,\tag{4}
$$

where $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ . The SR answer-score threshold $\tau _ { s }$ suppresses contributions from answers weakly supported by the SR output. The reverse score then counts only the portion of the SR–GT score difference exceeding the margin $\tau _ { m }$

The final hallucination score is computed as follows:

$$
H ( I _ { \mathrm { G T } } , I _ { \mathrm { S R } } ) = F ( I _ { \mathrm { G T } } , I _ { \mathrm { S R } } ) + R ( I _ { \mathrm { G T } } , I _ { \mathrm { S R } } ) .\tag{5}
$$

## 4 LEARNING A LIGHTWEIGHT HALLUCINATION ASSESSMENT MODEL

VHA requires repeated VLM inference, and its full pipeline is not end-to-end differentiable. We therefore train VHA+, a lightweight, differentiable model that predicts VHA scores directly from GT–SR image pairs. Figure 3 illustrates dataset construction (Section 4.1) and model training (Section 4.2).

![](images/ae2d0f5394200ff7bc97ae77627028fcd0851f406c714fa636d542519ec78203.jpg)  
Figure 3: Overview of our VHA+ training pipeline. (a) Diverse GT–SR pairs are constructed using PiSA-SR and labeled with our VHA scores. (b) VHA+ is trained to predict these scores directly from image pairs.

## 4.1 TRAINING DATASET CONSTRUCTION

We construct GT-SR pairs spanning different degrees of content preservation and alteration by varying the input resolution and enhancement settings. Each GT image is downsampled at multiple scales and restored with PiSA-SR (Sun et al., 2025), whose pixel-level and semantic-level controls vary fidelity and perceptual detail. We label each pair with its VHA score to obtain the training dataset $\mathcal { D } = \{ ( I _ { \mathrm { G T } } ^ { ( i ) } , I _ { \mathrm { S R } } ^ { ( i ) } , h _ { i } ) \} _ { i = 1 } ^ { N }$ , where $h _ { i } = H ( I _ { \mathrm { G T } } ^ { ( i ) } , I _ { \mathrm { S R } } ^ { ( i ) } )$ and N is the number of image pairs. Additional dataset construction details are provided in Section 5.1.

## 4.2 MODEL ARCHITECTURE AND TRAINING OBJECTIVE

VHA+ predicts the VHA score from the distance between learned representations of the GT image and SR outputs. Following the DINO-based scoring approach of Hallucination Score (Ren et al., 2025), we use a shared DINOv3 encoder $f _ { \theta }$ (Simeoni et al., 2025) to extract the final CLS tokens,´ $z _ { \mathrm { G T } } = f _ { \theta } ( I _ { \mathrm { G T } } )$ and $z _ { \mathrm { S R } } = f _ { \theta } ( I _ { \mathrm { S R } } )$ . An affine transformation maps their cosine distance to the VHA score scale:

$$
\hat { h } ( I _ { \mathrm { G T } } , I _ { \mathrm { S R } } ) = \alpha \left( 1 - \frac { z _ { \mathrm { G T } } { ^ { \mathsf { T } } z _ { \mathrm { S R } } } } { \| z _ { \mathrm { G T } } \| _ { 2 } \| z _ { \mathrm { S R } } \| _ { 2 } } \right) + \beta ,\tag{6}
$$

where α and $\beta$ are learned scale and offset parameters.

We fine-tune the final transformer blocks and learn the affine parameters, while keeping the earlier blocks frozen. Supervision from VHA encourages the learned feature distance to reflect content loss, alteration, and unsupported additions. The training objective combines mean squared error (MSE) on standardized scores with a Pearson correlation loss:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { M S E } } ( \hat { \bf h } , { \bf h } ) + \lambda _ { \mathrm { c o r r } } \big ( 1 - \rho ( \hat { \bf h } , { \bf h } ) \big ) ,\tag{7}
$$

where $\hat { \mathbf { h } }$ and h contain the predicted and target VHA scores, $\rho$ is the Pearson correlation, and $\lambda _ { \mathrm { c o r r } }$ controls the correlation term. The MSE term encourages accurate prediction of target values, while the correlation term encourages the alignment in score distribution. Additional training details are provided in Appendix B.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Benchmark construction. We construct VHA-bench, a human-annotated hallucination benchmark, by collecting human judgments of hallucination in generative SR outputs. For this study, we use 90 ground-truth images from the RealSR validation set (Cai et al., 2019). Each $5 1 2 \times 5 1 2$ GT image is downsampled to $1 2 8 \times 1 2 8 .$ , and the low-resolution input is restored to $5 1 2 \times 5 1 2$ using six generative SR methods: StableSR (Wang et al., 2024), DiffBIR (Lin et al., 2024a), SeeSR (Wu et al., 2024b), OSEDiff (Wu et al., 2024a), PURE (Wei et al., 2025) and VOSR (Wu et al., 2026), yielding 540 GT–SR pairs. We conduct a human study with 200 participants, each randomly assigned 54 GT–SR pairs for evaluation. Using the GT image as reference, participants rate hallucination on a five-point scale, from 1 (no hallucination) to 5 (severe hallucination), focusing on changes in semantic content and visual attributes while ignoring differences in sharpness or noise alone. Each GT–SR pair receives approximately 20 ratings on average, which we aggregate into a mean opinion score (MOS). The study interface is shown in Figure 7 in the appendix.

<table><tr><td></td><td></td><td rowspan="2">Taxonomy / Question / Answer Source</td><td rowspan="2"></td><td rowspan="2">s(IGT) s(ISR)</td></tr><tr><td></td><td></td></tr><tr><td></td><td rowspan="4"></td><td>Q. What text is on the upper sign? A. eduroam</td><td rowspan="2">10.58</td><td rowspan="2">-0.04</td></tr><tr><td></td><td>Q. What text is on the upper sign?</td></tr><tr><td></td><td>Text (high) A. oduronni</td><td rowspan="2">5.04</td><td rowspan="2">19.75 6.66</td></tr><tr><td></td><td>Object (high) Q. What is the lower sign? A. no smoking sign</td></tr><tr><td></td><td></td><td>Scene (high) Q. What is the background surface? A. brick wall</td><td>9.12</td><td>9.25</td></tr><tr><td></td><td>QGT</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Taxonomy / Question / Answer</td><td> $s ( I _ { \mathrm { G T } } )$ </td><td>s(ISR)</td></tr><tr><td></td><td></td><td>Source Text (high) Q. What letter is visible in upper left quadrant?</td><td>13.00</td><td>13.00</td></tr><tr><td></td><td></td><td>QGT A.E</td><td></td><td></td></tr><tr><td></td><td></td><td>Object (high) Q. What letter is in upper right? QGT A.H</td><td>11.90</td><td>11.90</td></tr><tr><td></td><td></td><td>Texture (low) Q. What does the black ink surface look like? QGT A. Slightly grainy texture</td><td>1.97</td><td>0.64</td></tr><tr><td></td><td></td><td>Texture (low) Q. What is the background texture? QSR A. Fine horizontal ribs</td><td>-3.69</td><td>2.90</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Taxonomy /</td><td></td><td></td></tr><tr><td></td><td></td><td>Question / Answer Source</td><td> $s ( I _ { \mathrm { G T } } )$ </td><td> $s ( I _ { \mathrm { S R } } )$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>3.72</td><td>3.66</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Object (high) Q. What is on left side of banner?</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>QGT A. Walking person</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>6.59</td><td>4.33</td></tr><tr><td></td><td></td><td>Object (high) Q. What is on right side of banner?</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>QGT A. Person on paper airplane</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>2.13</td><td>1.90</td></tr><tr><td></td><td></td><td>Tone (low) Q. What contrast exists between text and background?</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td> $\boldsymbol { \mathcal { Q } } _ { \mathrm { G T } }$  A. High contrast white on blue</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>-4.32</td><td>2.85</td></tr><tr><td></td><td></td><td>Texture (low) Q. What texture is on orange panel?</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td> ${ \mathcal { Q } } _ { \mathrm { S R } }$  A. Vertical wood grain</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Figure 4: Examples of Q/A generation and scoring on VHA-bench. $\mathcal { Q } _ { \mathrm { G T } }$ and ${ \mathcal { Q } } _ { \mathrm { S R } }$ denote Q/A sets generated from the GT image and SR outputs, respectively. For each Q/A pair, s(I) denotes the answer score $s ( I ; q , a )$ computed by Eq. 2.

Evaluation metrics. We use existing evaluation metrics to examine their correlations with human hallucination judgments on VHA-bench and to assess the outputs of SR methods. The metrics include PSNR and SSIM for pixel-level similarity, and LPIPS (Zhang et al., 2018), DISTS (Ding et al., 2020), DINO, and CLIP for feature-level similarity. For DINO and CLIP, we compute cosine distances between the final-layer CLS features of GT and SR images using frozen pretrained DINOv3 ViT-B/16 and CLIP ViT-B/16 encoders (Simeoni et al., 2025; Radford et al., 2021),´ respectively. For no-reference image quality assessment (NR-IQA), we include MUSIQ (Ke et al., 2021), CLIP-IQA (Wang et al., 2023), Q-Insight (Li et al., 2025a), VisualQuality-R1 (Wu et al., 2025), and DeQA-Score (You et al., 2025). We also include A-FINE (Chen et al., 2025) as a generalized full-reference IQA (GFR-IQA) metric.

Implementation details. We use Qwen3-VL-8B-Instruct (Bai et al., 2025) for Q/A generation and answer scoring. For bidirectional VHA scoring, we set $\tau _ { g } = - 2 , \tau _ { s } = 0 \mathrm { a n d } \tau _ { m } = 0 . 5$ For VHA+ training dataset construction, we select 20,000 images each from LSDIR (Li et al., 2023)

Table 1: Spearman (ρ) and Pearson (r) correlations with human mean opinion scores (MOS) for hallucination, computed on 540 SR outputs from six methods and rated by 200 participants. Higher correlations indicate closer agreement with human judgments. VHA denotes our bidirectional VQAbased assessment method, while VHA+ is its learned predictor that operates without VLM inference.
<table><tr><td>Category</td><td>Method</td><td> $\rho \uparrow$ </td><td> $r \uparrow$ </td><td>Category</td><td>Method</td><td> $\rho \uparrow$ </td><td>r ↑</td></tr><tr><td rowspan="2">Pixel sim.</td><td>PSNR</td><td>0.162</td><td>0.181</td><td rowspan="4">Feature sim.</td><td>LPIPS</td><td>0.083</td><td>0.089</td></tr><tr><td>SSIM</td><td>0.010</td><td>0.012</td><td>DISTS</td><td>0.066</td><td>0.069</td></tr><tr><td rowspan="4">NR-IQA</td><td>MUSIQ</td><td>-0.008</td><td>-0.014</td><td>DINO</td><td>0.081</td><td>0.115</td></tr><tr><td>CLIP-IQA</td><td>0.168</td><td>0.159</td><td>CLIP</td><td>0.282</td><td>0.273</td></tr><tr><td>Q-Insight</td><td>0.080</td><td>0.036</td><td rowspan="2">Direct rating Hallucination</td><td>0.497</td><td>0.482</td></tr><tr><td>VQ-R1</td><td>0.175</td><td>0.126</td><td>HS prompting</td><td>0.543 0.563</td></tr><tr><td rowspan="2">GFR-IQA</td><td>DeQA</td><td>0.145</td><td>0.156</td><td>Assessment VHA (Ours)</td><td>0.618</td><td>0.645</td></tr><tr><td>A-FINE</td><td>0.173</td><td>0.299</td><td>VHA+ (Ours)</td><td>0.659</td><td>0.675</td></tr></table>

and TextVQA (Singh et al., 2019) and use their $5 1 2 \times 5 1 2$ center crops as GT images. We downsample each GT image by factors of 0.5, 0.375, and 0.25 and restore the degraded inputs using PiSA-SR (Sun et al., 2025), with and without additional 2× input upsampling. We set the pixel-level and semantic-level control weights of PiSA-SR $( \lambda _ { \mathrm { p i x } } , \lambda _ { \mathrm { s e m } } )$ to (1, 0), (0.5, 1), and (1, 1).

## 5.2 RESULTS

Examples of Q/A generation and scoring. Figure 4 illustrates how generated Q/A pairs capture high-level semantic content and low-level visual attributes, and how their score differences reflect changes between the GT image and SR outputs. In the first example, the Q/A pair about the original text in $\mathcal { Q } _ { \mathrm { G T } }$ yields a higher $s \bar { ( } I _ { \mathrm { G T } } )$ than $s ( \bar { I } _ { \mathrm { S R } } )$ , whereas the pair about the hallucinated text in Q<sub>SR</sub> yields a lower $s ( I _ { \mathrm { G T } } )$ than $s ( I _ { \mathrm { S R } } )$ . In the second, the Q/A pair about texture in $\mathcal { Q } _ { \mathrm { G T } }$ yields a higher $s ( I _ { \mathrm { G T } } )$ than $s ( I _ { \mathrm { S R } } )$ , whereas the pair about background texture in ${ \mathcal { Q } } _ { \mathrm { S R } }$ yields a lower $s ( I _ { \mathrm { G T } } )$ than $s ( I _ { \mathrm { S R } } )$ . The $\mathrm { Q } / \mathrm { A }$ pairs about unchanged letters receive nearly identical scores across the two images. In the third, a Q/A pair in ${ \mathcal { Q } } _ { \mathrm { S R } }$ captures newly introduced wood-grain texture, with $s ( I _ { \mathrm { G T } } )$ much lower than $s ( I _ { \mathrm { S R } } )$

Comparison of correlations with existing metrics. Table 1 shows that VHA and VHA+ achieve higher correlations with human hallucination judgments than the existing metrics evaluated on VHAbench. VHA achieves Spearman and Pearson correlations of 0.618 and 0.645, respectively, compared with at most 0.282 and 0.299 among the existing metrics. The low correlations of these baselines suggest that general image similarity and quality measures alone are insufficient to capture human judgments of hallucination severity. VHA+ achieves correlations of 0.659 and 0.675, respectively, similar to those of VHA. These results demonstrate that the lightweight predictor retains strong agreement with human judgments without requiring VLM inference at evaluation time.

Comparison with hallucination assessment methods. Table 1 compares hallucination assessment methods using the same VLM backbone (Qwen3-VL-8B-Instruct). Direct rating provides the VLM with GT images and SR outputs and prompts it to assign a hallucination rating from 1 to 5. The full prompt is provided in Appendix D. HS prompting uses exactly the same prompt as Hallucination Score (Ren et al., 2025), with GT images, low-resolution inputs, and SR outputs. Both baselines show reasonable correlations with human judgments, but their discrete ratings provide limited granularity. Also, neither baseline explicitly quantifies how visual elements or attributes contribute to the hallucination score. Our method achieves higher correlations while providing continuous scores from answer log-probabilities and explicit contributions from individual Q/A pairs.

## 5.3 ABLATION STUDIES AND ANALYSIS

Ablations of VHA scoring. Table 2 presents ablation results for VHA scoring on VHA-bench. Table 2(a) shows that bidirectional scoring achieves higher correlations with human judgments than either forward or reverse scoring alone. Table 2(b) shows that scores from high-level Q/A pairs align more closely with human hallucination judgments than those from low-level pairs. Combining both further improves correlations, suggesting that high-level semantic content and low-level visual attributes provide complementary information for assessing hallucination. Finally, Table 2(c) evaluates forward thresholding with $\tau _ { g }$ in Eq. 3 and reverse thresholding with $\tau _ { s }$ and $\tau _ { m }$ in Eq. 4. Both help suppress unreliable Q/A contributions and improve correlations with human judgments.

Table 2: Ablations of VHA scoring on VHA-bench. $\rho$ and r denote Spearman and Pearson correlations.  
(a) Bidirectional scoring
<table><tr><td>Scoring</td><td>ρ↑</td><td>r ↑</td></tr><tr><td>Forward</td><td>0.516</td><td>0.562</td></tr><tr><td>Reverse</td><td>0.465</td><td>0.560</td></tr><tr><td>Bidirectional</td><td>0.618</td><td>0.645</td></tr></table>

(b) High- and low-level Q/A
<table><tr><td>Q/A set</td><td>ρ↑</td><td>r ↑</td></tr><tr><td>High only</td><td>0.539</td><td>0.584</td></tr><tr><td>Low only</td><td>0.204</td><td>0.197</td></tr><tr><td>High + Low</td><td>0.618</td><td>0.645</td></tr></table>

(c) Forward and reverse thresholding
<table><tr><td>Forward</td><td>Reverse</td><td>ρ↑</td><td>r ↑</td></tr><tr><td>√</td><td></td><td>0.570</td><td>0.614</td></tr><tr><td></td><td>√</td><td>0.607</td><td>0.635</td></tr><tr><td>√</td><td>√</td><td>0.618</td><td>0.645</td></tr></table>

![](images/f0ad579e382a466ac176bfc4350f19f883fd41713698023c02912dcb813af860.jpg)  
Figure 5: Spearman correlations of our high-level and low-level bidirectional Q/A scores with the image evaluation metrics on VHA-bench.

Correlations of high- and low-level VHA scores with existing metrics. We compute high- and low-level VHA scores using only high- and low-level Q/A pairs, respectively. Figure 5 shows their Spearman correlations with existing metrics on VHA-bench. The high-level score correlates weakly with pixel-level similarity metrics (PSNR and SSIM) and perceptual similarity metrics (LPIPS and DISTS), while showing relatively higher correlations with semantic feature distances (DINO and CLIP) and most NR-IQA metrics. Compared with the high-level score, the low-level score shows higher correlations with most existing metrics, especially for pixel-level and perceptual similarity metrics. This pattern suggests that many existing metrics reflect low-level visual changes more closely than high-level semantic changes.

Computational efficiency of VHA+. Table 3 compares the parameter counts and inference times of VHA and VHA+. We measure the average inference time per image pair over 50 pairs on a single NVIDIA RTX A6000. VHA+ uses 85M parameters, compared with 8,767M for VHA, and reduces the average inference time from 101 s to 17 ms per pair. This corresponds to approximately 5,900× faster inference under the measured configuration. By directly predicting scores without VLM inference, VHA+ substantially reduces evaluation cost while retaining strong agreement with human judgments.

Table 3: Parameter counts and average inference time of VHA and VHA+.
<table><tr><td colspan="3">Method Parameters (M) Inference time (ms)</td></tr><tr><td>VHA</td><td>8,767</td><td>101,000</td></tr><tr><td>VHA+</td><td>85</td><td>17</td></tr></table>

## 5.4 FINE-TUNING SUPER-RESOLUTION WITH VHA+

The lightweight and differentiable design of VHA+ allows it to guide generative SR training. We use the hallucination score predicted by the frozen VHA+ model as an auxiliary loss. Following Hallucination Score (Ren et al., 2025), we combine this loss with a MUSIQ-based quality objective (Ke et al., 2021) to discourage oversmoothing and preserve perceptual quality. We fine-tune pretrained VOSR (Wu et al., 2026) by adding these terms to its original distillation objective while retaining its training pipeline. We denote the fine-tuned model as VOSR<sup>∗</sup>. Training details, including datasets, loss weights, and optimization settings, are provided in Appendix F.1.

Table 4 compares VOSR<sup>∗</sup> with pretrained generative SR models and bicubic upsampling on the RealSR (Cai et al., 2019) and DRealSR (Wei et al., 2020) test sets provided by StableSR (Wang et al., 2024). Although bicubic upsampling achieves high pixel-level similarity score, it produces blurry outputs which results in low perceptual quality score. Also, it does not synthesize new fine details, thereby limiting hallucinations arising from unsupported additions. Consistently, bicubic upsampling achieves better VHA and VHA+ scores than the pretrained generative SR models. VOSR<sup>∗</sup> improves all reported metrics over VOSR on both datasets, including VHA and VHA+. These results suggest that combining VHA+-based supervision with a perceptual quality objective can reduce hallucination while maintaining perceptual quality.

Table 4: Quantitative comparison of SR methods on RealSR and DRealSR. VOSR<sup>∗</sup> denotes VOSR fine-tuned with auxiliary losses based on our VHA+ score and MUSIQ.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">Pixel sim.</td><td colspan="2">Feature sim.</td><td colspan="3">NR-IQA</td><td rowspan="2">GFR-IQA A-FINE↓</td><td colspan="2">Ours</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>DISTS ↓</td><td>MUSIQ↑</td><td>CLIPIQA ↑</td><td>VQ-R1 ↑</td><td>VHA↓</td><td>VHA+↓</td></tr><tr><td rowspan="8">RealSR</td><td>Bicubic</td><td>25.48</td><td>0.7200</td><td>0.4564</td><td>0.2633</td><td>25.81</td><td>0.3098</td><td>1.2210</td><td>77.22</td><td>0.9861</td><td>0.8055</td></tr><tr><td>StableSR</td><td>23.08</td><td>0.6611</td><td>0.3107</td><td>0.2186</td><td>65.00</td><td>0.6142</td><td>3.5362</td><td>27.48</td><td>1.4394</td><td>0.9863</td></tr><tr><td>DiffBIR</td><td>23.27</td><td>0.6140</td><td>0.3617</td><td>0.2419</td><td>68.30</td><td>0.6921</td><td>3.9381</td><td>31.87</td><td>1.3234</td><td>1.0500</td></tr><tr><td>SeeSR</td><td>23.86</td><td>0.6879</td><td>0.2980</td><td>0.2222</td><td>69.49</td><td>0.6725</td><td>3.8458</td><td>25.84</td><td>1.3439</td><td>1.0230</td></tr><tr><td>OSEDiff</td><td>23.59</td><td>0.6923</td><td>0.2921</td><td>0.2128</td><td>69.08</td><td>0.6688</td><td>4.0439</td><td>20.96</td><td>1.2806</td><td>1.0086</td></tr><tr><td>PURE</td><td>21.52</td><td>0.5755</td><td>0.3829</td><td>0.2450</td><td>67.05</td><td>0.6943</td><td>3.7221</td><td>41.68</td><td>1.3613</td><td>1.1788</td></tr><tr><td>VOSR</td><td>23.77</td><td>0.6816</td><td>0.2858</td><td>0.2097</td><td>70.00</td><td>0.5761</td><td>3.7426</td><td>21.39</td><td>1.0190</td><td>0.9283</td></tr><tr><td>VOSR* (Ours)</td><td>24.29</td><td>0.6914</td><td>0.2820</td><td>0.1985</td><td>70.98</td><td>0.6074</td><td>3.8557</td><td>11.24</td><td>0.9494</td><td>0.6413</td></tr><tr><td rowspan="9">DRealSR</td><td>Bicubic</td><td>27.87</td><td>0.7869</td><td>0.4608</td><td>0.2788</td><td>22.59</td><td>0.3190</td><td>1.0280</td><td>74.54</td><td>0.7310</td><td>0.6948</td></tr><tr><td>StableSR</td><td>25.78</td><td>0.6985</td><td>0.3379</td><td>0.2295</td><td>58.32</td><td>0.6250</td><td>3.1017</td><td>37.97</td><td>0.8453</td><td>0.7859</td></tr><tr><td>DiffBIR</td><td>23.52</td><td>0.5466</td><td>0.4923</td><td>0.2913</td><td>65.99</td><td>0.7213</td><td>3.7510</td><td>33.10</td><td>1.2837</td><td>1.1691</td></tr><tr><td>SeeSR</td><td>26.09</td><td>0.7284</td><td>0.3205</td><td>0.2304</td><td>64.86</td><td>0.6841</td><td>3.5276</td><td>22.73</td><td>1.1364</td><td>0.9278</td></tr><tr><td>OSEDiff</td><td>25.85</td><td>0.7374</td><td>0.2967</td><td>0.2163</td><td>64.69</td><td>0.6961</td><td>3.7727</td><td>21.38</td><td>0.8258</td><td>0.8162</td></tr><tr><td>PURE</td><td>23.34</td><td>0.5762</td><td>0.4526</td><td>0.2719</td><td>61.08</td><td>0.6773</td><td>3.3473</td><td>41.07</td><td>1.0318</td><td>1.0200</td></tr><tr><td>VOSR</td><td>25.44</td><td>0.7029</td><td>0.3479</td><td>0.2362</td><td>65.80</td><td>0.5906</td><td>3.4163</td><td>29.46</td><td>0.7516</td><td>0.9781</td></tr><tr><td>VOSR* (Ours)</td><td>25.83</td><td>0.7076</td><td>0.3473</td><td>0.2263</td><td>66.97</td><td>0.6422</td><td>3.5343</td><td>21.63</td><td>0.7161</td><td>0.6473</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/a42a2218f11ee9077bfca504ee277ce04e4d022948e8d90ab6b973f42ee43b80.jpg)  
Figure 6: Qualitative comparisons of bicubic upsampling and generative SR methods on RealSR testset. VOSR<sup>∗</sup>, fine-tuned using our VHA+ score and MUSIQ as auxiliary losses, shows fewer hallucinated details and more faithfully preserves ground-truth content than other methods.

Figure 6 illustrates examples in which VOSR<sup>∗</sup> better preserves text and suppresses spurious textures compared with VOSR. In the top example, VOSR<sup>∗</sup> more faithfully reconstructs the digit “8”, whose shape is distorted in the VOSR output. In the bottom example, VOSR<sup>∗</sup> reduces spurious textures, more closely matching the smoother appearance of the GT image. Additional qualitative examples are provided in Appendix G.

## 6 CONCLUSION

We propose a bidirectional VQA-based method (VHA) for evaluating hallucination in generative image super-resolution. Taxonomy-guided questions cover high-level semantic content and low-level visual attributes, enabling assessment of diverse forms of hallucination. Bidirectional comparisons of answer score assess both the loss or alteration of ground-truth evidence and unsupported additions in SR outputs. This addresses a gap in conventional SR evaluation, where visually realistic details can receive favorable quality scores despite misrepresenting the original content. We also train VHA+, a lightweight model that directly predicts hallucination score without VLM inference. On our human-annotated benchmark, VHA and VHA+ achieve high correlations with human hallucination judgments. Furthermore, our VHA+-based supervision extends this assessment to SR model training. We believe this work adds a complementary dimension to SR evaluation and encourages future research to consider hallucination alongside perceptual quality when developing SR models.

## REFERENCES

Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In CVPR Workshops, 2017.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image super-resolution: A new benchmark and a new model. In ICCV, 2019.

Du Chen, Tianhe Wu, Kede Ma, and Lei Zhang. Toward generalized image quality assessment: Relaxing the perfect reference quality assumption. In CVPR, 2025.

Jaemin Cho, Yushi Hu, Jason Baldridge, Roopal Garg, Peter Anderson, Ranjay Krishna, Mohit Bansal, Jordi Pont-Tuset, and Su Wang. Davidsonian scene graph: Improving reliability in fine-grained evaluation for text-to-image generation. In ICLR, 2024.

Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence, 44(5):2567–2581, 2020.

Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In ICCV, 2023.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. MUSIQ: Multi-scale image quality transformer. In ICCV, 2021.

Weiqi Li, Xuanyu Zhang, Shijie Zhao, Yabin Zhang, Junlin Li, Li Zhang, and Jian Zhang. Q-insight: Understanding image quality via visual reinforcement learning. In NeurIPS, 2025a.

Yachao Li, Dong Liang, Tianyu Ding, and Sheng-Jun Huang. StructSR: Refuse spurious details in real-world image super-resolution. In AAAI, 2025b.

Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, et al. Lsdir: A large scale dataset for image restoration. In CVPR Workshops, 2023.

Youngsun Lim, Hojun Choi, and Hyunjung Shim. Evaluating image hallucination in text-to-image generation with question-answering. In AAAI, 2025.

Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Yu Qiao, Wanli Ouyang, and Chao Dong. Diffbir: Toward blind image restoration with generative diffusion prior. In ECCV, 2024a.

Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In ECCV, 2024b.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

Weiming Ren, Raghav Goyal, Zhiming Hu, Tristan Ty Aumentado-Armstrong, Iqbal Mohomed, and Alex Levinshtein. Hallucination score: Towards mitigating hallucinations in generative image super-resolution. arXiv preprint arXiv:2507.14367, 2025.

Oriane Simeoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose,´ Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, et al. Dinov3. ¨ arXiv preprint arXiv:2508.10104, 2025.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019.

Lingchen Sun, Rongyuan Wu, Zhiyuan Ma, Shuaizheng Liu, Qiaosi Yi, and Lei Zhang. Pixel-level and semantic-level adjustable super-resolution: A dual-lora approach. In CVPR, 2025.

Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring CLIP for assessing the look and feel of images. In AAAI, 2023.

Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. International Journal of Computer Vision, 132(12):5929–5949, 2024.

Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, and Chen Change Loy. Esrgan: Enhanced super-resolution generative adversarial networks. In ECCV Workshops, 2018.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

Hongyang Wei, Shuaizheng Liu, Chun Yuan, and Lei Zhang. Perceive, understand and restore: Real-world image super-resolution with autoregressive multimodal generative models. In ICCV, 2025.

Pengxu Wei, Ziwei Xie, Hannan Lu, Zongyuan Zhan, Qixiang Ye, Wangmeng Zuo, and Liang Lin. Component divide-and-conquer for real-world image super-resolution. In ECCV, 2020.

Rongyuan Wu, Lingchen Sun, Zhiyuan Ma, and Lei Zhang. One-step effective diffusion network for real-world image super-resolution. NeurIPS, 2024a.

Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. Seesr: Towards semantics-aware real-world image super-resolution. In CVPR, 2024b.

Rongyuan Wu, Lingchen Sun, Zhengqiang Zhang, Xiangtao Kong, Jixin Zhao, Shihao Wang, and Lei Zhang. Vosr: A vision-only generative model for image super-resolution. In CVPR, 2026.

Tianhe Wu, Jian Zou, Jie Liang, Lei Zhang, and Kede Ma. Visualquality-r1: Reasoning-induced image quality assessment via reinforcement learning to rank. NeurIPS, 2025.

Michal Yarom, Yonatan Bitton, Soravit Changpinyo, Roee Aharoni, Jonathan Herzig, Oran Lang, Eran Ofek, and Idan Szpektor. What you see is what you read? improving text-image alignment evaluation. In NeurIPS, 2023.

Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, and Chao Dong. Teaching large language models to regress accurate image quality scores using score distribution. In CVPR, 2025.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

## A PROMPTS FOR Q/A GENERATION

Tables 5–8 provide the full prompt templates used for taxonomy-guided evidence discovery and evidence-conditioned question–answer generation in Section 3. We use separate templates for highlevel semantics and low-level visual attributes, and apply the same templates independently to ground truth and super-resolved images.

Table 5: High-level taxonomy-guided evidence discovery prompt.

Look at this HR image ({image id}) and identify high-level semantic   
evidence targets for short-answer Q/A generation.   
Taxonomy definitions:   
- scene semantic: The overall scene, environment, or image-level context.   
This taxonomy must describe the image as a whole or a broad scene region,   
not a single object.   
- object semantic: A distinct object or structure that contributes   
meaningfully to the image content, such as a tree, grass, tower, wall,   
house, vehicle, bridge, or signboard; or a recognizable visual property   
or part within an object or local region, such as texture, style,   
material, decoration, or a visible object part.   
- text logo: Clearly readable text, words, numbers, signs, brands, or logos.   
Generate up to 1 scene semantic target, up to 4 object semantic targets,   
and up to 4 text logo targets.   
A target is suitable when it is clearly visible in the HR image,   
supports a specific short-answer question, and may be altered by   
downsampling or super-resolution.   
Generate as many valid targets as are clearly supported by the image,   
within these limits.   
Reply with valid JSON only:   
[   
{   
"evidence id": "he01",   
"taxonomy": "scene semantic",   
"region scope": "whole scene or localized region",   
"region hint": "brief visible location",   
"target": "specific high-level semantic content",   
"hallucination risk": "how this content may be altered or invented"   
}   
]

Table 6: High-level evidence-conditioned question–answer generation prompt.

Look at this HR image ({image id}) and generate one high-level   
short-answer Q/A pair for each provided {taxonomy}   
evidence item.   
Taxonomy definition:   
{taxonomy definition}   
The question must identify the relevant image region unambiguously and   
ask only about the specified high-level semantic content.   
Use a precise semantic label or short noun phrase.   
Do not ask about overall image quality.   
Do not create yes/no questions.   
Each answer must be specific and 1-3 words only.   
Evidence items:   
{evidence json}   
Reply with valid JSON only:   
[   
{"evidence id": "{example evidence id}", "q": "...", "a": "..."}   
]

## Table 7: Low-level taxonomy-guided evidence discovery prompt.

```jsonl
Look at this HR image ({image id}) and identify low-level visual evidence
targets for short-answer Q/A generation.
Taxonomy definitions:
- color tone: Visible color, brightness, contrast, highlight, shadow, or
tone relationship within an image region.
- edge shape: Visible orientation, curvature, thickness, continuity, or
shape of a boundary or small structural part.
- texture: Visible surface grain, material appearance, fine pattern,
line structure, or texture density.
Generate 3 color tone targets, 3 edge shape targets, and 3 texture targets
for this image. A target is suitable when it is clearly visible in the HR
image, supports a specific short-answer question, and may be altered by
downsampling or super-resolution.
Reply with valid JSON only:
[
{
"evidence id": "le01",
"taxonomy": "color tone",
"region scope": "local",
"region hint": "brief visible location",
"target": "specific low-level visual property",
"hallucination risk": "how this property may be altered or invented"
}
```

## Table 8: Low-level evidence-conditioned question–answer generation prompt.

Look at this HR image ({image id}) and generate one highly specific   
low-level short-answer Q/A pair for each provided evidence item.   
The question must identify the relevant image region and ask only about   
the specified low-level visual property.   
Describe the finest reliably visible appearance. Avoid broad categorical   
answers when the image supports a more precise description.   
Use the following attribute structure:   
- color tone: describe the visible hue together with its lightness,   
saturation, or local tonal character.   
- edge shape: describe the visible boundary using its thickness,   
continuity, smoothness, geometry, or orientation.   
- texture: describe the visible texture using its scale, direction,   
regularity, density, or structural appearance.   
Combine two or three complementary visual attributes when they are   
clearly supported by the image. Do not add an attribute merely to make   
the answer longer or more specific.   
Use visually grounded everyday terms. Do not use uncommon or technical   
terminology solely for greater specificity.   
Do not ask about overall image quality or sharpness preference.   
Do not create yes/no questions.   
Each answer must be specific and 2-3 words long.   
Evidence items:   
{evidence json}   
Reply with valid JSON only:   
[   
{"evidence id": "le01", "q": "...", "a": "..."}   
]

## B VHA+ TRAINING DETAILS

We train the last four transformer blocks of the DINOv3 $\mathtt { V i T - B / 1 6 }$ encoder and the affine scale and offset parameters, while keeping the earlier layers frozen. We set $\lambda _ { \mathrm { c o r r } } = 3$ in Eq. 7 and train VHA+ using AdamW (Loshchilov & Hutter, 2019) for 20,000 steps with a batch size of 32. The learning rates are $1 0 ^ { - 6 }$ for the encoder and $1 0 ^ { - 4 }$ for the affine parameters.

Score standardization and calibration. Let $\mu _ { H }$ and $\sigma _ { H }$ denote the mean and standard deviation of the training-set target scores. We standardize each target score $h _ { i }$ as

$$
\tilde { h } _ { i } = \frac { h _ { i } - \mu _ { H } } { \sigma _ { H } } .\tag{8}
$$

We train the affine head of VHA+ to predict the standardized target $\tilde { h } _ { i }$ . Both loss terms in Eq. 7 are computed between the model outputs and the standardized targets $\tilde { h } _ { i }$ . For numerical stability, we add $1 0 ^ { - 8 }$ to the denominator when computing the Pearson correlation. At inference, we convert the affine head output to the original score scale using the stored training-set mean and standard deviation. For our training dataset, $\mu _ { H } = 0 . 7 5 4 5$ and $\sigma _ { H } = 1 . 0 2 5 4$

## C HUMAN STUDY INTERFACE

Using the GT on the left as reference, rate the hallucination score of the image on the right.

![](images/fd6da9e07cfe9caf01fceb90dc28f2bb53cd235b6ba8118de57ed1619213da3e.jpg)  
Figure 7: Web interface used for the human hallucination study. Participants compare the ground-truth reference (left) with a super-resolution result (right) and assign a hallucination score from 1 (no hallucination) to 5 (severe hallucination), while ignoring differences in sharpness and noise.

Table 9 provides the full prompt used for the Direct rating baseline in the comparison of VLM-based hallucination assessment methods in Section 5.2 (Table 1).

## Table 9: Full prompt for Direct rating.

You will receive two images:   
1. Ground Truth (GT): The reference high-resolution image.   
2. Super-Resolved Image (SR): An image restored by an AI super-resolution   
model.   
We define hallucinations as the loss or alteration of ground-truth evidence in SR,   
or the addition of content unsupported by GT.   
Rate the amount of hallucination in SR compared with GT on a scale from 1 to 5.   
A lower score means less hallucination, and a higher score means more   
hallucination.   
Output only the following JSON:   
{   
"score": <integer from 1 to 5>   
}

## E ADDITIONAL ABLATIONS

## E.1 VHA SCORING THRESHOLD ABLATIONS

Table 10: Ablations of VHA scoring thresholds on VHA-bench. ρ and r denote Spearman and Pearson correlations.

(a) Forward score threshold
<table><tr><td> $\tau _ { g }$ </td><td>ρ r</td></tr><tr><td>0 0.5919</td><td>0.6295</td></tr><tr><td>-1</td><td>0.6190 0.6427</td></tr><tr><td>-2 0.6179</td><td>0.6449</td></tr><tr><td>-3</td><td>0.6103 0.6408</td></tr><tr><td>-4</td><td>0.6085 0.6396</td></tr><tr><td>-5 0.6075</td><td>0.6355</td></tr><tr><td>-6</td><td>0.6079 0.6354</td></tr></table>

(b) Reverse score threshold
<table><tr><td colspan="2">ρ</td><td>r</td></tr><tr><td> $\tau _ { s }$  -3</td><td>0.6160</td><td>0.6426</td></tr><tr><td>-2</td><td>0.6164</td><td>0.6431</td></tr><tr><td>-1</td><td>0.6159</td><td>0.6437</td></tr><tr><td>0</td><td>0.6179</td><td>0.6449</td></tr><tr><td>1</td><td>0.6144</td><td>0.6430</td></tr><tr><td>2</td><td>0.6087</td><td>0.6383</td></tr><tr><td>3</td><td>0.5941</td><td>0.6295</td></tr></table>

(c) Reverse margin threshold
<table><tr><td> $\tau _ { m }$ </td><td>ρ</td><td>r</td></tr><tr><td>-1</td><td>0.5909</td><td>0.6366</td></tr><tr><td>-0.5</td><td>0.6045</td><td>0.6407</td></tr><tr><td>0</td><td>0.6173</td><td>0.6443</td></tr><tr><td>0.5</td><td>0.6179</td><td>0.6449</td></tr><tr><td>1</td><td>0.6133</td><td>0.6423</td></tr></table>

Table 10 shows the effects of the GT answer-score threshold $\tau _ { g }$ in Eq. 3 and the SR answer-score threshold $\tau _ { s }$ and margin $\tau _ { m }$ in Eq. 4. VHA is relatively insensitive to variations in these hyperparameters. However, overly strict or permissive thresholds can reduce correlations with human judgments. These results suggest that appropriate thresholds help suppress noisy Q/A contributions while preserving informative ones.

## E.2 VHA+ TRAINING ABLATIONS

Table 11: VHA+ training ablations on VHA-bench. ρ and r denote Spearman and Pearson correlations, respectively; bold indicates the selected settings.

(a) Correlation loss weight
<table><tr><td>λcorr</td><td>ρ</td><td>r</td></tr><tr><td>0</td><td>0.6370</td><td>0.6680</td></tr><tr><td>1</td><td>0.6540</td><td>0.6771</td></tr><tr><td>2</td><td>0.6586</td><td>0.6763</td></tr><tr><td>3</td><td>0.6594</td><td>0.6753</td></tr><tr><td>4</td><td>0.6593</td><td>0.6741</td></tr><tr><td>5</td><td>0.6582</td><td>0.6735</td></tr></table>

(b) Trainable blocks
<table><tr><td>Last K blocks</td><td>ρ</td><td>r</td></tr><tr><td>2</td><td>0.5962</td><td>0.6212</td></tr><tr><td>4</td><td>0.6594</td><td>0.6753</td></tr><tr><td>6</td><td>0.6556</td><td>0.6638</td></tr><tr><td>8</td><td>0.6494</td><td>0.6660</td></tr><tr><td>10</td><td>0.6389</td><td>0.6631</td></tr></table>

Table 11 presents VHA+ training ablations. Table 11(a) shows the results of varying the Pearson correlation loss weight $\lambda _ { \mathrm { c o r r } }$ in Eq. 7. An appropriate correlation loss weight improves agreement with human judgments compared with MSE alone. Table 11(b) shows the results for different numbers of trainable transformer blocks. Updating fewer or more than four blocks yields lower correlations, which may reflect underfitting and overfitting, respectively.

Table 12: Ablations of VHA+ and MUSIQ loss weights for VOSR fine-tuning.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">γvha γmusiq</td><td colspan="2">Pixel sim.</td><td colspan="2">Feature sim.</td><td colspan="2">NR-IQA</td><td rowspan="2">GFR-IQA</td><td colspan="2"></td><td rowspan="2">Ours</td></tr><tr><td></td><td>PSNR ↑</td><td>SSIM↑ LPIPS ↓</td><td>DISTS ↓</td><td>MUSIQ↑</td><td>CLIPIQA ↑</td><td>VQ-R1 ↑ A-FINE↓</td><td>VHA↓</td><td>VHA+↓</td></tr><tr><td rowspan="11">RealSR</td><td>0 0.001</td><td></td><td>23.04</td><td>0.6418</td><td>0.3601</td><td>0.2643</td><td>75.89</td><td>0.6820</td><td>3.723</td><td>11.29</td><td>1.372</td><td>1.172</td></tr><tr><td>0.05</td><td>0.001</td><td>23.96</td><td>0.6791</td><td>0.2925</td><td>0.2089</td><td>72.17</td><td>0.6373</td><td>3.930</td><td>11.03</td><td>0.9164</td><td>0.6856</td></tr><tr><td>0.10</td><td>0.001</td><td>24.29</td><td>0.6914</td><td>0.2820</td><td>0.1985</td><td>70.98</td><td>0.6074</td><td>3.856</td><td>11.24</td><td>0.9494</td><td>0.6413</td></tr><tr><td>0.15 0.001</td><td>24.25</td><td>0.6914</td><td>0.2698</td><td>0.1907</td><td>70.25</td><td>0.5910</td><td>3.769</td><td>13.17</td><td></td><td>0.9557</td><td>0.6304</td></tr><tr><td>0.20</td><td>0.001</td><td>24.38</td><td>0.6988</td><td>0.2616</td><td>0.1892</td><td>69.39</td><td>0.5610</td><td>3.791</td><td>13.94</td><td>0.9355</td><td>0.6211</td></tr><tr><td>0.10</td><td>0</td><td>25.78</td><td>0.7431</td><td>0.2237</td><td>0.1737</td><td>51.31</td><td>0.3457</td><td>2.709</td><td>48.03</td><td>0.8744</td><td>0.5645</td></tr><tr><td></td><td>0.100.0005</td><td>24.53</td><td>0.7026</td><td>0.2618</td><td>0.1892</td><td>68.87</td><td>0.5710</td><td>3.761</td><td>16.08</td><td>0.9410</td><td>0.6341</td></tr><tr><td>0.10 0.0010</td><td></td><td>24.29</td><td>0.6914</td><td>0.2820</td><td>0.1985</td><td>70.98</td><td>0.6074</td><td>3.856</td><td>11.24</td><td>0.9494</td><td>0.6413</td></tr><tr><td>0.100.0015</td><td></td><td>23.98</td><td>0.6816</td><td>0.2899</td><td>0.2026</td><td>71.83</td><td>0.6255</td><td>3.943</td><td>10.10</td><td>1.0331</td><td>0.6536</td></tr><tr><td></td><td>0.100.0020</td><td>23.70</td><td>0.6653</td><td>0.3042</td><td>0.2087</td><td>73.02</td><td>0.6391</td><td>3.945</td><td>8.811</td><td>0.9854</td><td>0.6641</td></tr><tr><td rowspan="10">DRealSR</td><td>0 0.0010</td><td>24.51</td><td>0.6482</td><td>0.4357</td><td>0.3002</td><td>73.57</td><td>0.6899</td><td>3.184</td><td>12.27</td><td></td><td>1.271</td><td>1.256</td></tr><tr><td>0.050.0010</td><td>25.51</td><td>0.6954</td><td>0.3573</td><td>0.2349</td><td>68.52</td><td></td><td>0.6545</td><td>3.678</td><td>18.59</td><td>0.7262</td><td>0.6788</td></tr><tr><td>0.10 0.0010</td><td>25.83</td><td>0.7076</td><td>0.3473</td><td>0.2263</td><td>66.97</td><td>0.6422</td><td>3.534</td><td>21.63</td><td></td><td>0.7161</td><td>0.6473</td></tr><tr><td>0.15 0.0010</td><td>25.84</td><td>0.7060</td><td>0.3349</td><td>0.2175</td><td>66.01</td><td>0.6237</td><td>3.471</td><td>25.25</td><td></td><td></td><td>0.6101</td></tr><tr><td>0.20 0.0010</td><td>25.95</td><td>0.7139</td><td>0.3266</td><td>0.2146</td><td></td><td>65.11</td><td>0.6042</td><td>3.459</td><td>27.07</td><td>0.6745 0.6342</td><td>0.6043</td></tr><tr><td>0.10 0</td><td>27.24</td><td>0.7714</td><td>0.2539</td><td>0.1837</td><td>48.03</td><td></td><td>0.4260</td><td>2.489</td><td>52.57</td><td>0.6347</td><td>0.5812</td></tr><tr><td>0.10 0.0005</td><td>26.08</td><td>0.7173</td><td>0.3247</td><td>0.2140</td><td>64.51</td><td></td><td>0.6123</td><td>3.462</td><td>28.58</td><td>0.7118</td><td>0.6473</td></tr><tr><td>0.10 0.0010</td><td>25.83</td><td>0.7076</td><td>0.3473</td><td>0.2263</td><td>66.97</td><td>0.6422</td><td>3.534</td><td></td><td>21.63</td><td>0.7161</td><td>0.6473</td></tr><tr><td>0.10 0.0015</td><td>25.52</td><td>0.6947</td><td>0.3592</td><td>0.2310</td><td>68.30</td><td>0.6541</td><td>3.666</td><td>20.93</td><td></td><td>0.6843</td><td>0.6469</td></tr><tr><td>0.100.0020</td><td>25.30</td><td>0.6799</td><td>0.3733</td><td>0.2386</td><td>69.49</td><td>0.6648</td><td>3.673</td><td>19.24</td><td>0.6466</td><td></td><td>0.6527</td></tr></table>

## F FINE-TUNING SR MODELS WITH VHA+

## F.1 ADDITIONAL TRAINING DETAILS

VOSR (Wu et al., 2026) first trains a multi-step SR model and then distills it into a one-step student. We fine-tune the pretrained one-step student with the additional VHA+ and MUSIQ losses while keeping the released multi-step model fixed as the teacher. We use the full LSDIR training set (Li et al., 2023) and all 800 DIV2K training images (Agustsson & Timofte, 2017). Following VOSR’s training setup, we use random $5 1 2 \times 5 1 2$ crops and its degradation pipeline. The fine-tuning objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { d i s t i l l } } + \gamma _ { \mathrm { v h a } } \mathcal { L } _ { \mathrm { V H A + } } + \gamma _ { \mathrm { m u s i q } } \mathcal { L } _ { \mathrm { M U S I Q } } , } \end{array}\tag{9}
$$

where ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ is VOSR’s original distillation loss. The auxiliary losses ${ \mathcal { L } } _ { \mathrm { V H A + } }$ and $ { \mathcal { L } } _ { \mathrm { M U S I Q } }$ are the score predicted by the frozen VHA+ model and the negative MUSIQ score (Ke et al., 2021), respectively. We set $\gamma _ { \mathrm { v h a } } = 0 . 1$ and $\gamma _ { \mathrm { m u s i q } } = 0 . 0 0 1$ . We fine-tune the student for 1,000 updates using AdamW (Loshchilov & Hutter, 2019), a learning rate of $1 0 ^ { - 5 }$ , and a batch size of 16. We use an exponential moving average (EMA) with decay 0.99.

## F.2 SR FINE-TUNING ABLATIONS

Table 12 shows the results of varying the VHA+ and MUSIQ loss weights in Eq. 9. Increasing the VHA+ loss weight tends to lower perceptual-quality scores, whereas increasing the MUSIQ loss weight can reduce image similarity and increase VHA+ scores. These results suggest that appropriate loss weights are needed to reduce hallucinations while maintaining perceptual quality.

## G ADDITIONAL QUALITATIVE RESULTS

Figures 8 and 9 present additional qualitative comparisons of generative SR models and bicubic upsampling on the RealSR (Cai et al., 2019) and DRealSR (Wei et al., 2020) test sets provided by StableSR (Wang et al., 2024), respectively.

![](images/fa0f1f694e34fcbf87f6421103ea4e8d52213a9a3e0fcfe02381f91ac8949f0f.jpg)  
Figure 8: Qualitative comparisons of bicubic upsampling and generative SR methods on RealSR testset. VOSR<sup>∗</sup>, fine-tuned using our VHA+ score and MUSIQ as auxiliary losses, shows fewer hallucinated details and more faithfully preserves ground-truth content than other methods.

![](images/52354c756ebd8d4adee23c5541260dc3a8b7b2a6ead08cd49365202ebd5f4e0e.jpg)  
Figure 9: Qualitative comparisons of bicubic upsampling and generative SR methods on DRealSR testset. VOSR<sup>∗</sup>, fine-tuned using our VHA+ score and MUSIQ as auxiliary losses, shows fewer hallucinated details and more faithfully preserves ground-truth content than pretrained VOSR.