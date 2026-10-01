# SAGE: SALIENT FACTOR DISCOVERY AND GENERA-TION WITH VISUAL FOUNDATION REPRESENTATIONS

Shuang Liang<sup>1,2†</sup> Lejun Liao<sup>2†</sup> Shiyuan Zhang<sup>2,3†</sup> Max C. Zhang<sup>2</sup> Xiaolong Luo<sup>4</sup> Han Wang<sup>1</sup> Stefano Anzellotti<sup>2</sup> Yuan Yuan<sup>2B</sup> <sup>1</sup>HKU <sup>2</sup>Boston College <sup>3</sup>University of Virginia <sup>4</sup>Harvard University

## ABSTRACT

Given a target dataset, such as faces with eyeglasses, and a background dataset, such as faces without, contrastive analysis separates salient factors specific to the target from common content shared by both. We aim for salient representations that capture target-specific detail in each image, such as the shape, color, and position of the glasses, so that they reveal subtypes without subtype labels and guide the generation of new examples of a discovered subtype, even one with no name or text description. We introduce SAGE, which learns both factors directly in the high-dimensional spatial latent of a frozen representation autoencoder and conditions a diffusion transformer on the learned salient representation of a reference image. On Digits-ImageNet and FFHQ eyeglasses, SAGE combines high-fidelity reconstruction (rFID below 2) with unsupervised subtype discovery, recovering the digits better than baselines (probe accuracy 0.950 vs. at most 0.281) and revealing eyewear types, finer sunglasses styles, and mislabeled images; salientconditioned generation raises Digits-ImageNet subtype accuracy over the unfactorized latent (90.5% vs. 27.7%) and diversity on both datasets. On retinal OCT, SAGE’s salient space separates three diseases using only normal/disease labels.

![](images/b5dd94d88a485761162f37e20d56446dc616f643e314f7ce3507748943912bbd.jpg)  
Figure 1: Overview of SAGE. Given background (BG) and target (TG) datasets, SAGE uses only dataset labels to decompose frozen self-supervised visual representations into common and salient factors, capturing shared content and target-specific variation, respectively. The decomposition supports high-fidelity reconstruction, subtype discovery in the salient space, and salient-conditioned generation that preserves target attributes while varying common content.

## 1 INTRODUCTION

Learning what two data distributions share and what sets one apart is central to multimodal learning (Dufumier et al., 2025), domain adaptation (Lee et al., 2021), and disentanglement (Sanchez et al., 2020). Contrastive analysis (CA) poses this problem with dataset labels alone: given a target and a background dataset, it separates salient factors $z _ { s }$ specific to the target from common factors $z _ { c }$ shared by both. For faces with and without eyeglasses as target and background (Fig. 1), $z _ { s }$ should capture eyewear-related variation (e.g., frame style and color) and $z _ { c }$ the content both groups share (e.g., identity, pose, and scene). The stakes are especially high in medicine, where clinical group labels may be available even when the corresponding imaging differences remain unknown. Autism, for instance, is diagnosed from behavior, which tells us who is affected but not how their brains differ; contrasting brain MRI of the autism and control groups can reveal these differences (Aglinskas et al., 2022). Such patterns are subtle, hard to describe in words, and not known in advance, yet finding them could provide objective imaging markers, deepen the understanding of a disease, and even uncover its subtypes. All of these settings raise the same question:

Can we learn salient representations that capture target-specific detail in each image, reveal subtypes without labels, and generate new examples of what they discover?

Answering it requires a representation that is semantically rich enough for discovery and decodable back to images, and existing CA methods offer complementary strengths toward these goals: generative CA models (Abid & Zou, 2019; Louiset et al., 2023; Carton et al., 2024) decode their factors but reconstruct complex natural images poorly, and contrastive CA methods (Louiset et al., 2024) learn semantically rich factors but have no decoder. In this work, we propose SAGE (Salient factor discovery and Generation), a contrastive analysis approach that factorizes the frozen spatial latent z of a representation autoencoder (RAE), RAEv2 (Singh et al., 2026), into $z _ { c }$ and $z _ { s } .$ , whose sum the frozen RAE decoder maps back to images. Built on a pretrained foundation encoder, DI NOv3 (Simeoni et al., 2025), this latent is semantically rich, keeps a high-dimensional spatial layout ´ in which salient factors can retain instance-level detail such as the shape, color, and position of a particular pair of glasses, decodes with high fidelity, and supports state-of-the-art diffusion generation (Zheng et al., 2026; Singh et al., 2026).

This latent, however, mixes target-specific variation with all other image content (Li et al., 2023). Separating the two is nontrivial: reconstruction alone is already satisfied by an empty $z _ { s } ,$ a highdimensional $z _ { s }$ can easily absorb shared content, and it can capture only a generic template of the attribute while instance-specific detail stays in $z _ { c } .$ . SAGE counters these failures with objectives that include swapping salient factors between background and target images, keeping them sparse, and requiring them to be recoverable after the swap. The learned salient factors cleanly separate targetspecific content and reveal subtypes, and they can be swapped or removed to edit individual images. SAGE can further generate images. Because the salient variation is often not known in advance and has no name or text description, instruction- or prompt-conditioned generation (Liu et al., 2023; Wu et al., 2025) cannot even be applied; instead, in a second stage, SAGE conditions a diffusion transformer on the learned salient representation of a reference image to generate new images that keep its salient subtype while their common content varies. This in turn indicates that the salient condition carries little of the reference’s shared content.

Our contributions answer each part of the question above: (1) Learning clean, high-fidelity salient factors: salient factors decode to only the digit or eyeglasses, and together with the common factors they reconstruct Digits-ImageNet and FFHQ images with rFID below 2, versus above 120 for generative CA baselines. (2) Revealing subtypes without labels: on Digits-ImageNet, the salient space recovers the ten digits far better than a contrastive CA baseline on the same encoder (probe accuracy 0.950 vs. 0.148); on FFHQ, it separates eyewear types, finer sunglasses styles, and a mislabeled cluster; and on retinal OCT, it distinguishes three diseases. (3) Generating what is discovered: on Digits-ImageNet, conditioning on a reference’s salient representation, without text prompts or subtype labels, preserves its subtype far more often and produces more varied samples than the unfactorized latent (subtype accuracy 90.5% vs. 27.7%; Vendi 12.50 vs. 2.65).

## 2 RELATED WORK

Contrastive analysis. Contrastive analysis (CA) separates factors shared by background and target datasets from target-specific ones using only dataset labels, with applications in neuroimaging (Aglinskas et al., 2022; Zhu et al., 2026) and disease subgroup discovery (Louiset et al., 2026). cVAE (Abid & Zou, 2019), SepVAE (Louiset et al., 2023), and Double-InfoGAN (Carton et al., 2024) learn the factors with a generator trained from scratch; SepCLR (Louiset et al., 2024) learns strong salient representations under the InfoMax principle but has no decoder, and adding a reconstruction objective weakens its factor separation. CS-StyleGAN (He et al., 2025) separates factors in an inverted StyleGAN latent and refines reconstructions with input features, and the concurrent Diff-CA (Soumm et al., 2026) decomposes compact conditioning tokens of a diffusion generator. These methods represent the salient factor as a low-dimensional vector or global token; SAGE factorize the full spatial latent of a frozen RAE and uses it for both discovery and generation.

![](images/e8c13dca891295a50e52406170b54a9f7c1808f9990bf83234397f892d1711e3.jpg)  
Figure 2: Stage 1 of SAGE for salient factor discovery. Trainable common and salient encoders split latents of the frozen RAEv2 encoder $( \mathrm { D I N O v } 3 ) ;$ the frozen RAEv2 decoder maps factors back to images. Top: reconstruction $( \mathcal { L } _ { \mathrm { r e c } } )$ and swap adversarial training $( \mathcal { L } _ { \mathrm { { s w a p } } } ^ { G } )$ . Bottom: swapped images are re-encoded for $\mathcal { L } _ { \mathrm { c y c } }$ and $\mathcal { L } _ { \mathrm { c N C E } }$ , while $\mathcal { L } _ { \mathrm { l s c } }$ and the sparsity priors act directly on the factors. Snowflakes and flames mark frozen and trainable modules.

Representation autoencoders and conditioned generation. Recent work replaces the VAE of latent diffusion with frozen pretrained encoders (Zheng et al., 2026; Tong et al., 2026; Shi et al., 2026; Gao et al., 2026); RAEs pair such encoders with ViT decoders that match or exceed standard VAEs (Zheng et al., 2026), and RAEv2 adds REPA (Singh et al., 2026; Yu et al., 2025). SAGE factorizes this latent rather than only generating in it. Representation-conditioned generation (Li et al., 2024) conditions on a full self-supervised representation, and diffusion editing (Meng et al., 2022; Wu et al., 2025) changes prompt-specified attributes; SAGE instead conditions on a discovered salient representation, specifying a subtype by example.

## 3 METHODOLOGY

Given background and target datasets $\mathcal { D } _ { b }$ and $\mathcal { D } _ { t }$ , SAGE uses only dataset labels $d \in \{ b , t \}$ , not subtype labels, to factorize the latent $z = E ( x )$ of a frozen RAE with decoder D. Stage 1 learns the factorization; Stage 2 freezes it and trains a salient-conditioned diffusion transformer (Figs. 2, 3).

## 3.1 SALIENT FACTOR DISCOVERY

Why a frozen RAE latent space. SAGE needs a latent that is semantically expressive for discovery and decodable for reconstruction, manipulation, and generation. Unlike the VAE latents of standard latent diffusion, which are trained for reconstruction and carry limited semantic structure (Zheng et al., 2026), RAEs pair frozen pretrained visual encoders with trained decoders, providing both properties and supporting high-quality latent diffusion (Zheng et al., 2026; Singh et al., 2026). We use the frozen RAEv2 encoder and decoder (Singh et al., 2026): E is DINOv3- L (Simeoni et al., 2025) with multi-layer summation (MLS), and´ D reconstructs images from z. We factorize this spatial latent directly, without a learned bottleneck, to retain localized salient content (Fig. 4); freezing E and D attributes decoded changes to the learned factors and lets Stage 2 reuse RAEv2’s pretrained diffusion transformer.

Common and salient encoders. For $z = E ( x ) \in \mathbb { R } ^ { N \times C }$ , trainable encoders produce same-shape common and salient factors $z _ { c } = E _ { c } ( z )$ and $z _ { s } = E _ { s } ( z )$ , decoded as $D ( z _ { c } , z _ { s } ) : = D ( z _ { c } + z _ { s } ) ;$ the sum stays in the frozen decoder’s latent space, and $z _ { s } = 0$ means no target-specific content. We require the factors to be decodable and composable, not statistically independent (Appendix E).

![](images/2e3fb5f24d9de695c0a478271f71f40651dcad0fdfaac6ff918b9f01e42166ad.jpg)  
Figure 3: Stage 2 of SAGE for salient-conditioned generation. The frozen RAEv2 and salient encoders extract a reference salient representation, max-pooled to $c _ { s } .$ . An RAEv2-initialized diffusion transformer is trained with conditional flow matching in this latent. Top right: generated samples; each row keeps the eyewear type or digit of the reference at left while faces and scenes vary.

Because dataset labels alone underdetermine this split, the objectives below reconstruct each image, push target-specific content into $z _ { s }$ , and preserve its image-specific appearance (Fig. 2).

Asymmetric reconstruction. Because the salient factor should carry only target-specific content, it is absent for background images but complements the common factor for targets: $\hat { \boldsymbol { x } } ^ { b } = D ( z _ { c } ^ { b } , \mathbf { 0 } )$ and $\hat { x } ^ { t } = D ( z _ { c } ^ { t } , z _ { s } ^ { t } )$ . With $\ell ( x , \bar { \hat { x } } ) = \| x - \hat { x } \| _ { 2 } ^ { 2 } + \mathrm { L P I P S } ( x , \hat { x } )$ , we minimize

$$
\mathcal { L } _ { \mathrm { r e c } } = \mathbb { E } _ { x ^ { b } } \Big [ \ell ( x ^ { b } , \hat { x } ^ { b } ) + \big \| z _ { c } ^ { b } - z ^ { b } \big \| _ { 2 } ^ { 2 } \Big ] + \mathbb { E } _ { x ^ { t } } \Big [ \ell ( x ^ { t } , \hat { x } ^ { t } ) + \big \| z _ { c } ^ { t } + z _ { s } ^ { t } - z ^ { t } \big \| _ { 2 } ^ { 2 } \Big ] .\tag{1}
$$

The latent terms tie decoder inputs to real latents, avoiding plausible but unreliable compositions.

Swap adversarial loss. Reconstruction leaves factor allocation underdetermined: for a target image, the trivial split $z _ { c } ^ { t } = z ^ { t } , \ z _ { s } ^ { t } = 0$ reconstructs perfectly. To move target-specific content to $z _ { s }$ we zero-initialize $E _ { s }$ so that $z _ { s }$ starts empty and acquires content only when required, then swap target salient factors. Swapping exploits dataset labels: if $z _ { s } ^ { t }$ captures target-specific content, erasing it should yield a background-like image, while adding it to $\hat { z } _ { c } ^ { b }$ should yield a target-like image: $\hat { x } _ { \mathrm { f a k e } } ^ { \tilde { b } } = D ( z _ { c } ^ { \tilde { t } } , \mathbf { 0 } )$ and $\hat { x } _ { \mathrm { f a k e } } ^ { t } = D ( z _ { c } ^ { b } , z _ { s } ^ { t } )$ . Because swapped images have no paired ground truth, background and target discriminators $\Delta _ { b }$ and $\Delta _ { t }$ distinguish $\hat { x } _ { \mathrm { f a k e } } ^ { b }$ and $\hat { x } _ { \mathrm { f a k e } } ^ { t }$ from reconstructions ${ \hat { x } } ^ { b }$ and $\hat { x } ^ { t }$ , respectively (Goodfellow et al., 2014) (architecture in Appendix C.1); they minimize the hinge loss $\mathcal { L } _ { \mathrm { { s w a p } } } ^ { D ^ { \star } }$ (Appendix C.3). Using reconstructions as real samples prevents the discriminators from exploiting decoder artifacts to distinguish swapped outputs from input images. The encoders minimize the following loss, where σ is the sigmoid function:

$$
\mathcal { L } _ { \mathrm { s w a p } } ^ { G } = - \mathbb { E } \left[ \log \sigma \big ( \Delta _ { b } \big ( \hat { x } _ { \mathrm { f a k e } } ^ { b } \big ) \big ) \right] - \mathbb { E } \left[ \log \sigma \big ( \Delta _ { t } \big ( \hat { x } _ { \mathrm { f a k e } } ^ { t } \big ) \big ) \right] ,\tag{2}
$$

Sparsity priors. Swap alone does not exclude common content from $z _ { s }$ . Since reconstruction discards $\bar { z _ { s } ^ { b } }$ , we shrink background salient factors to zero with an $\ell _ { 2 } ^ { 2 }$ penalty; for targets, an $\ell _ { 1 }$ penalty favors sparse $z _ { s } ^ { t }$ to keep common content, such as the face wearing glasses, out of it:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { B G - s p } } = \mathbb { E } _ { \boldsymbol { x } ^ { b } } \Big [ \big \lVert \boldsymbol { z } _ { s } ^ { b } \big \rVert _ { 2 } ^ { 2 } \Big ] , \qquad \mathcal { L } _ { \mathrm { T G - s p } } = \lambda \mathbb { E } _ { \boldsymbol { x } ^ { t } } \big [ \big \lVert \boldsymbol { z } _ { s } ^ { t } \big \rVert _ { 1 } \big ] . } \end{array}\tag{3}
$$

The dataset-specific λ balances sparsity against retaining target detail (Appendix C.2).

Consistency objectives. Dataset-level adversarial matching cannot ensure $z _ { s } ^ { t }$ carries instancespecific target content: a generic eyeglass prototype could pass both discriminators while instancespecific details remain in $\bar { z } _ { c } ^ { t } .$ . We therefore require the factors to be recoverable after swapping, encouraging each salient representation to be instance-specific; the image-space cycle loss re-encodes swapped images, $\hat { z } ^ { t } = E ( \hat { x } _ { \mathrm { f a k e } } ^ { t } )$ and $\hat { z } ^ { b } = E ( \hat { x } _ { \mathrm { f a k e } } ^ { b } )$ , to recover their factors (Zhu et al., 2017):

$$
\mathcal { L } _ { \mathrm { c y c } } = \mathbb { E } \Big [ \| E _ { c } ( \hat { z } ^ { t } ) - z _ { c } ^ { b } \| _ { 2 } ^ { 2 } + \| E _ { s } ( \hat { z } ^ { t } ) - z _ { s } ^ { t } \| _ { 2 } ^ { 2 } + \| E _ { c } ( \hat { z } ^ { b } ) - z _ { c } ^ { t } \| _ { 2 } ^ { 2 } + \| E _ { s } ( \hat { z } ^ { b } ) \| _ { 2 } ^ { 2 } \Big ] ,\tag{4}
$$

With the original factors as fixed targets (Appendix C.4), this loss checks rendered transfers and erasures; its final term enforces no salient content after erasure. The cheaper $\mathcal { L } _ { \mathrm { l s c } }$ applies the same recovery in latent space, without decoding: for a random permutation π of the minibatch,

$$
\mathcal { L } _ { \mathrm { l s c } } = \mathbb { E } _ { i } \left[ \| E _ { c } ( z _ { c } ^ { i } + z _ { s } ^ { \pi ( i ) } ) - z _ { c } ^ { i } \| _ { 2 } ^ { 2 } + \| E _ { s } ( z _ { c } ^ { i } + z _ { s } ^ { \pi ( i ) } ) - z _ { s } ^ { \pi ( i ) } \| _ { 2 } ^ { 2 } \right] .\tag{5}
$$

Cycle NCE loss. Recovery losses do not preclude overlapping common and salient content, such as facial appearance present in both. Cycle NCE applies InfoNCE (van den Oord et al., 2018) to globally pooled, ℓ<sub>2</sub>-normalized representations before and after swap–re-encoding, using the cycle to form positive pairs without augmentations. Because $\hat { x } _ { \mathrm { f a k e } } ^ { t } = D \dot { ( } z _ { c } ^ { b } , z _ { s } ^ { t } )$ combines factors from different images, its re-encoded factors should recover $z _ { c } ^ { b }$ and $z _ { s } ^ { t }$ despite the changed context. For each factor, minibatch representations of the other factor, original or re-encoded, serve as negatives; same-factor representations are excluded, so the loss separates the two factors without pushing apart images within one, preserving subtype grouping (Appendix C.5).

Full objective. The encoders minimize the unit-weighted sum,

$$
{ \mathcal { L } } _ { \mathrm { S A G E } } = { \mathcal { L } } _ { \mathrm { r e c } } + { \mathcal { L } } _ { \mathrm { s w a p } } ^ { G } + { \mathcal { L } } _ { \mathrm { B G - s p } } + { \mathcal { L } } _ { \mathrm { T G - s p } } + { \mathcal { L } } _ { \mathrm { c y c } } + { \mathcal { L } } _ { \mathrm { l s c } } + { \mathcal { L } } _ { \mathrm { c N C E } } ,\tag{6}
$$

where $\lambda$ enters only through $\mathcal { L } _ { \mathrm { T G - s p } } .$ . The discriminators minimize $\mathcal { L } _ { \mathrm { s w a p } } ^ { D } ,$ and the two updates alternate at each step; the RAEv2 encoder and decoder remain frozen throughout.

## 3.2 SALIENT-CONDITIONED GENERATION

After Stage 1, the frozen RAEv2 and salient encoders map a target image $x ^ { t } \operatorname { t o } z _ { s } ^ { t } = E _ { s } ( E ( x ^ { t } ) )$ ; the common encoder is unnecessary. This representation conditions Stage 2 generation (Fig. 3): samples retain the reference’s salient characteristics while their common content varies. We condition on $z _ { s }$ rather than the full latent z, which also encodes common content such as the face or scene that generation would otherwise reproduce. Channel-wise max pooling gives $c _ { s } = \mathrm { G M P } ( z _ { s } ^ { t } ) \in \mathbb { R } ^ { C }$ discarding location but keeping channels active in few tokens of the sparse map; $c _ { s }$ is projected to one token and concatenated with the noisy latent and time embeddings.

A diffusion transformer (DiT) with RAEv2’s DDT head generates latents. Initialized from the pretrained RAEv2 checkpoint, it is trained with flow matching (Lipman et al., 2023) in the same latent space on target images only. With $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and $\tau \in [ 0 , \bar { 1 } ]$ , let $z ^ { \tau } = ( 1 - \tau ) z ^ { t } + \tau \epsilon$ The DiT $F _ { \theta }$ predicts the clean latent, giving $\hat { v } _ { \theta } = \left( z ^ { \tau } - F _ { \theta } ( z ^ { \tau } , \tau , c _ { s } ) \right) /$ max $( \tau , \tau _ { \mathrm { m i n } } )$ , and is trained to match the flow velocity by minimizing $\mathcal { L } _ { \mathrm { f m } } = \mathbb { E } _ { x ^ { t } , \epsilon , \tau } \| \hat { v } _ { \boldsymbol { \theta } } ( z ^ { \tau } , \tau , c _ { s } ) - ( \epsilon - z ^ { t } ) \| _ { 2 } ^ { 2 }$ . During training, a learnable null token replaces $c _ { s }$ with probability p<sub>uncond</sub>, enabling classifier-free guidance (Ho & Salimans, 2022) and unconditional sampling. At inference, we integrate from Gaussian noise at $\tau = 1$ to $\tau = 0$ conditioned on $c _ { s }$ and decode with frozen D: different noise samples vary common content, while $c _ { s }$ fixes salient characteristics (Appendix D).

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETTINGS

Datasets & metrics. We evaluate on Digits-ImageNet, ImageNet images with and without an overlaid MNIST digit (ten digit subtypes) (LeCun et al., 1998; Deng et al., 2009); FFHQ faces with and without eyeglasses (reading glasses or sunglasses) (Karras et al., 2019; DCGM); and OCT-Kermany normal retinal scans versus scans of three diseases (CNV, DME, and DRUSEN) (Kermany et al., 2018). Subtype labels are used only for evaluation (Appendix B.1). We measure reconstruction by rFID (Heusel et al., 2017), PSNR, SSIM (Wang et al., 2004), and LPIPS (Zhang et al., 2018); subtype discovery by Salient LP and ARI (Hubert & Arabie, 1985)/NMI (Strehl & Ghosh, 2002) of k-means clusters; and generation by gFID (Heusel et al., 2017), generated-image subtype accuracy, and within-condition Vendi score (Friedman & Dieng, 2023) (Appendices B.4, B.5, and D.2).

Table 1: Benchmark comparison on Digits-ImageNet and FFHQ. rFID: reconstruction Frechet´ Inception Distance; N/A: no decoder; Double-InfoGAN uses its native 128 128 resolution; all other methods use 256 256. BG/TG Sil.: silhouette score of background versus target salient representations (Appendix Table 5). LP Acc.: linear-probe subtype accuracy, five-fold on Digits-ImageNet and balanced on FFHQ (Appendix B.4). ARI/NMI: k-means on the ℓ -normalized, unprojected salient representations, with k set to the subtype count (Appendix B.5). FFHQ labels are audit-corrected.
<table><tr><td></td><td colspan="7">Digits-ImageNet</td><td colspan="7">FFHQ</td></tr><tr><td></td><td colspan="4">Reconstruction</td><td colspan="3">Salient representation</td><td colspan="4">Reconstruction</td><td colspan="3">Salient representation</td></tr><tr><td>Method</td><td>rFID↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>BG/TG Sil. ↑</td><td>LP Acc. ↑</td><td>ARI/NMI↑</td><td>rFID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>BG/TG Sil. ↑</td><td>LP Acc. ↑</td><td>ARI/NMI↑</td></tr><tr><td>cVAE (Abid &amp; Zou, 2019)</td><td>151.0</td><td>18.33</td><td>0.478</td><td>0.673</td><td>0.006</td><td>0.135</td><td>0.001/0.003</td><td>188.2</td><td>18.42</td><td>0.567</td><td>0.590</td><td>0.027</td><td>0.898</td><td>0.001/0.001</td></tr><tr><td>SepVAE (Louiset et al., 2023)</td><td>162.5</td><td>17.50</td><td>0.461</td><td>0.696</td><td>0.006</td><td>0.145</td><td>0.009/0.019</td><td>197.4</td><td>17.67</td><td>0.552</td><td>0.598</td><td>0.206</td><td>0.897</td><td>-0.001/0.000</td></tr><tr><td>Double-InfoGAN (Carton et al., 2024)</td><td>171.4</td><td>15.97</td><td>0.375</td><td>0.668</td><td>0.013</td><td>0.281</td><td>0.001/0.003</td><td>122.5</td><td>16.73</td><td>0.424</td><td>0.458</td><td>0.050</td><td>0.964</td><td>0.298/0.299</td></tr><tr><td>SepCLR (Louiset et al., 2024)</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.546</td><td>0.180</td><td>0.000/0.001</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.752</td><td>0.970</td><td>0.822/0.696</td></tr><tr><td>DINOv3 + SepCLR</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.642</td><td>0.148</td><td>0.000/0.001</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.551</td><td>0.976</td><td>0.915/0.825</td></tr><tr><td>SAGE (Ours)</td><td>1.78</td><td>20.56</td><td>0.571</td><td>0.216</td><td>0.820</td><td>0.950</td><td>0.337/0.472</td><td>1.65</td><td>23.02</td><td>0.723</td><td>0.163</td><td>0.873</td><td>0.983</td><td>0.939/0.865</td></tr></table>

Baselines and implementation. We compare with cVAE, SepVAE, Double-InfoGAN, and Sep-CLR (Abid & Zou, 2019; Louiset et al., 2023; Carton et al., 2024; Louiset et al., 2024), as well as DINOv3 + SepCLR and unfactorized DINOv3 features. Full implementation details appear in Appendices B.2, C, and D.

## 4.2 SALIENT FACTOR DISCOVERY

We first ask whether SAGE isolates a clean salient factor, one that contains the target content and little else, while the two factors together still reconstruct the image.

Qualitative results. Among decoder-based methods, only SAGE separates the target attribute cleanly (Fig. 4; additional randomly selected examples in Appendix Fig. 9): it reconstructs scenes and faces with their digits or eyeglasses, common-only decoding removes only the target attribute, and salient-only decoding renders the digit or eyeglasses alone at their original location and shape. Generative baselines blur reconstructions and alter the scene or face even in common-only decod ing; their salient-only outputs show no recognizable digit on Digits-ImageNet and blurred eyeglasses with much of the face on FFHQ, indicating common-content leakage. At the dataset level, SAGE’s salient t-SNE separates background from target images (Fig. 5a, d), as the sparsity prior drives background salient factors toward zero; those of cVAE and, on Digits-ImageNet, SepVAE mix the two (Appendix Fig. 10).

Quantitative results. SAGE achieves rFID 1.78 on Digits-ImageNet and 1.65 on FFHQ, whereas generative baselines trained from scratch exceed 120 (Table 1), because SAGE factorizes a frozen RAEv2 latent and decodes it with the pretrained decoder, adding little error over decoding z directly (1.78 versus 0.39 on Digits-ImageNet). Its salient space also separates background from target images best among all methods (silhouette 0.820 and 0.873). Linear probes (Appendix A.9) show that digit identity is far more accessible in z<sub>s</sub> than in z (0.950 versus 0.328), while ImageNet category, a property of the shared scene, is readily decoded from z and z<sub>c</sub> (0.765) but hardly from z<sub>s</sub> (0.045). SepCLR and DINOv3 + SepCLR lack decoders; their comparison follows.

Digits-ImageNet  
![](images/16848bd8ef79641334fec8848969a030633fa45fc4cdfe4a91e8f1dea3855609.jpg)

FFHQ  
![](images/912114fbcbec4ea0d8388ac933334d09c87372d6ba92bae58f46509c75435818.jpg)  
Figure 4: Reconstruction, common-only, and salient-only decoding. Columns show the input, the full reconstruction $D ( z _ { c } , z _ { s } )$ , common-only decoding $D ( z _ { c } , \mathbf { 0 } )$ , and salient-only decoding $D ( \mathbf { 0 } , z _ { s } )$ on Digits-ImageNet (left) and FFHQ (right); rows compare cVAE, SepVAE, Double-InfoGAN, and SAGE. A clean salient factor renders only the digit or eyeglasses.

![](images/714da5517ced798d33d928d343af56ea3bfdd522cbb2cb31b5637ae3faea4e1c.jpg)  
Figure 5: Subtype discovery in the salient space. Top: t-SNE of SAGE’s salient representations separates background and target images (a, d) and organizes digit and eyewear subtypes (b, e), whereas raw DINOv3 features mix them (c, f). Bottom: examples from the k-means clusters marked in (b, e) show coherent digit and eyewear groups (g, h); FFHQ cluster C contains images mislabeled as wearing glasses, and red points are audited no-glasses images.

## 4.3 SUBTYPE DISCOVERY

We next ask whether the salient factor discovers target subtypes without subtype labels, both the expected ten digits and two eyewear types and groups we did not expect.

Qualitative results. Figure 5 shows t-SNE (van der Maaten & Hinton, 2008) of salient representations, colored by dataset label and, for targets, subtype labels used only for visualization. Background–target separation (Fig. 5a, d; Section 4.2) does not imply subtype separation: Sep-CLR and DINOv3 + SepCLR also separate the datasets, yet their target digits remain mixed (Appendix A.3). SAGE recovers the expected subtypes: the ten digit identities form locally coherent, partly overlapping groups, and reading glasses and sunglasses form two separate groups (Fig. 5b, e); raw z separates neither (Fig. 5c, f). Samples from each group (Fig. 5g, h) share the digit or eyewear type while their scenes and faces differ, so the grouping follows the target attribute rather than shared content.

Discovering finer eyewear styles. Within the sunglasses group (Fig. 5e), SAGE separates three styles: thick, angular wayfarer-like frames; thin-rimmed aviator-like lenses; and a broader mixed group (Appendix Fig. 11). The names describe the samples; the data have no style labels.

Discovering mislabeled images. Beyond reading glasses and sunglasses, the FFHQ t-SNE contains cluster C (Fig. 5e, h), whose images lack eyeglasses but whose automatic annotations label them as wearing glasses, often because of face paint or hat-brim shadows (Appendix Fig. 13c). Thi cluster led us to audit all 3,437 target validation images: annotators labeled the visible eyewear without seeing SAGE’s assignments and corrected 125 labels (3.64%; all corrections are shown in Appendix Fig. 12), including 39 no-glasses images; SAGE places 34 of these 39 in cluster C, while the other five show face paint or masks around the eyes (Appendix Fig. 13a, b). With k = 3, including no-glasses images, SAGE reaches ARI/NMI 0.937/0.865, whereas every baseline remains below 0.46/0.56; e.g., DINOv3 + SepCLR drops from 0.915/0.825 to 0.453/0.550 (Appendix Ta ble 4). SAGE thus also exposes label noise; Table 1 uses the audit-corrected labels for all methods.

Quantitative results. On Digits-ImageNet, SAGE reaches digit probe accuracy 0.950, versus at most 0.281 for baselines (Table 1) and 0.328 for the unfactorized RAEv2 latent z (Appendix Table 7); because SAGE and z share an encoder, this gain comes from factorization. Its ARI/NMI is 0.337/0.472, versus at most 0.009/0.019 for baselines. On FFHQ, SAGE achieves the best probe accuracy and clustering (0.983; 0.939/0.865), ahead of DINOv3 + SepCLR (0.976; 0.915/0.825). High probe accuracy alone does not ensure well-separated clusters: cVAE and SepVAE reach FFHQ probe accuracies near 0.90 yet ARI near zero. SAGE helps most when z obscures subtypes.

## 4.4 FACTOR MANIPULATION

Because both factors live in the latent space of the frozen decoder, the salient factor can be moved between images, removed, or interpolated, and the result decoded directly. We test whether such edits change only the target attribute. Swapping and erasure. Adding a target image’s salient factor to a background image transfers its particular attribute (Fig. 6, left): a digit 7 retains its location and stroke shape, and thick black frames, rather than generic glasses, appear on a new face while the scene

![](images/1c146b4c5d8c5e8df63cd12e6ebbae31849838e5cbafa4e9fa981fd7071d7acd.jpg)  
Figure 6: Salient swapping and erasure.

and identity remain. Conversely, zeroing a target’s salient factor removes the digit or reading glasses while preserving the rest (Fig. 6, right; more in Fig. 14). These edits illustrate that each salient factor carries its own image’s target attribute and little else. Interpolation. Interpolating salient factors of two target images while retaining the first common factor changes mainly the target attribute (Fig. 7, salient-only rows): a digit turns from 1 to 7 on a fixed photograph, and clear reading glasses darken into sunglasses on a fixed face. Interpolating both factors cross-fades the whole image (full-latent rows). Salient interpolation thus traverses subtypes while preserving common content.

## 4.5 DISCOVERING DISEASE SUBTYPES IN RETINAL OCT

Unlike digits and eyeglasses, retinal disease subtypes are clinically subtle and vary across patients. On OCT-Kermany, SAGE is trained only with normal/disease labels. Disease subtypes. Its salient space separates the three diseases, with t-SNE groups that overlap mainly at their boundaries (Fig. 8, left), a linear-probe accuracy of 0.960, and ARI/NMI 0.415/0.415 (Appendix A.7). Disease-specific edits. Adding a disease scan’s salient factor to a normal scan introduces fluid-filled cavities, while removing it yields a retina closer to normal (Fig. 8, middle and right), suggesting disease-related features rather than patient-specific anatomy. These edits are qualitative, not clinical evidence.

![](images/5290eab04ed5247461058966056178131adb8805d06183d9c7850339bbc98509.jpg)  
Figure 7: Salient interpolation. For two held-out target images $x ^ { A }$ and $x ^ { B }$ , salient-only rows decode $D \big ( z _ { c } ^ { A } , z _ { s } ^ { \alpha } \big )$ with $\dot { z } _ { s } ^ { \alpha } = ( 1 - \alpha ) z _ { s } ^ { A } + \alpha z _ { s } ^ { B }$ ; full-latent rows interpolate both factors.

![](images/00980199c1731aed83ccc7292266b776267a4f9b450ec4e1ca2c1fa1cbbcad38.jpg)  
Figure 8: OCT disease subtypes and salient edits. Left: salient t-SNE of disease scans, colored by disease subtype labels unused in training. BG TG adds a disease scan’s salient factor to a normal scan; TG BG removes a disease scan’s salient factor.

Table 2: Salient-conditioned generation. Raw DINOv3 GMP conditions on the unfactorized latent. Subtype Acc.: share of samples assigned the reference subtype by a DINOv3-L classifier trained on real targets. Vendi (diversity) and Cos-to-ref (reference similarity) use 16 samples per reference and DINOv2-L features. Real images: classifier accuracy and within-subtype Vendi (Appendix D.2).
<table><tr><td>Dataset</td><td>Conditioning</td><td>Uncond. gFID↓ Cond. gFID↓</td><td></td><td>Subtype Acc. ↑</td><td>Vendi↑</td><td>Cos-to-ref ↓</td></tr><tr><td rowspan="3">Digits-ImageNet</td><td>Real images</td><td></td><td></td><td>99.3</td><td>15.79</td><td></td></tr><tr><td>Raw DINOv3 GMP</td><td>8.86</td><td>2.40</td><td>27.70</td><td>2.65</td><td>0.794</td></tr><tr><td>SAGE Salient GMP</td><td>4.76</td><td>4.09</td><td>90.48</td><td>12.50</td><td>0.116</td></tr><tr><td rowspan="3">FFHQ</td><td>Real images</td><td></td><td></td><td>99.1</td><td>10.09</td><td></td></tr><tr><td>Raw DINOv3 GMP</td><td>16.07</td><td>11.17</td><td>98.5</td><td>2.32</td><td>0.779</td></tr><tr><td>SAGE Salient GMP</td><td>15.80</td><td>16.03</td><td>96.6</td><td>6.96</td><td>0.470</td></tr></table>

## 4.6 SALIENT-CONDITIONED GENERATION

Representation-conditioned generation (Li et al., 2024) conditions on a full self-supervised representation; we instead condition on SAGE’s pooled salient representation $c _ { s } .$ . Raw DINOv3 GMP pools the unfactorized latent z identically, testing whether factorization removes reference face or scene information. Qualitative results. Salient conditioning generates new faces and scenes that keep the reference digit or eyewear type, whereas Raw DINOv3 GMP reproduces the reference person or scene almost exactly and often changes the digit (Appendix Fig. 15); Fig. 3 (top right) shows several samples per reference. Quantitative results. On Digits-ImageNet, salient conditioning raises subtype accuracy from 27.7% to 90.5% and Vendi from 2.65 to 12.50 (real: 15.79), and lowers Cos-to-ref from 0.794 to 0.116 (Table 2). On FFHQ, both conditions preserve eyewear type (96.6% versus 98.5%), but only salient conditioning produces varied samples (Vendi 6.96 versus 2.32; Cos-to-ref 0.470 versus 0.779). Since the two conditions differ only in the salient encoder, reduced copying supports $z _ { s }$ as a more selective condition. Raw DINOv3 GMP attains lower conditional gFID, consistent with closer reference copying rather than subtype control, while SAGE has lower unconditional gFID on both datasets.

## 4.7 ABLATION STUDIES

Table 3: Loss ablations on Digits-ImageNet. Common-only: rFID and SSIM of $D ( z _ { c } , \mathbf { 0 } )$ for target images against their digit-free ImageNet backgrounds. Salient probes measure digit identity (target content, higher is better) and ImageNet category (shared content, lower is better). ARI/NMI are computed as in Table 1; their standard deviations over three k-means seeds are at most 0.0007.
<table><tr><td rowspan="2">Configuration</td><td colspan="4">Full Reconstruction</td><td colspan="2">Common-only</td><td colspan="2">Salient Linear Probe</td><td>Clustering</td></tr><tr><td>rFID↓</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td><td>rFID↓ SSIM↑</td><td></td><td>Digit↑</td><td>ImageNet↓</td><td>ARI/NMI↑</td></tr><tr><td>SAGE (all objectives)</td><td>1.78</td><td>20.56</td><td>0.571</td><td>0.216</td><td>1.89</td><td>0.569</td><td>0.950</td><td>0.045</td><td>0.337/0.472</td></tr><tr><td>w/o swap adversarial  $( \mathcal { L } _ { \mathrm { s w a p } } ^ { G } )$ </td><td>1.01</td><td>20.67</td><td>0.566</td><td>0.210</td><td>4.88</td><td>0.544</td><td>0.123</td><td>0.169</td><td>0.000/0.001</td></tr><tr><td>w/o target sparsity  $( \mathcal { L } _ { \mathrm { T G - s p } } )$ </td><td>1.15</td><td>21.06</td><td>0.598</td><td>0.193</td><td>1.36</td><td>0.585</td><td>0.923</td><td>0.302</td><td>0.274/0.409</td></tr><tr><td>w/o Cycle NCE  $( \mathcal { L } _ { \mathrm { c N C E } } )$ </td><td>1.29</td><td>20.78</td><td>0.576</td><td>0.205</td><td>1.58</td><td>0.573</td><td>0.950</td><td>0.100</td><td>0.348/0.479</td></tr><tr><td>w/o image-space cycle consistency  $( \mathcal { L } _ { \mathrm { c y c } } )$ </td><td>2.84</td><td>19.88</td><td>0.562</td><td>0.236</td><td>3.46</td><td>0.559</td><td>0.951</td><td>0.086</td><td>0.360/0.496</td></tr><tr><td>w/o latent-space swap consistency  $( \mathcal { L } _ { \mathrm { l s c } } )$ </td><td>1.82</td><td>20.30</td><td>0.568</td><td>0.218</td><td>2.25</td><td>0.567</td><td>0.953</td><td>0.107</td><td>0.361/0.494</td></tr></table>

Table 3 removes each objective in turn. The swap adversarial loss is what moves target content into the salient factor: without it, the digit probe drops from 0.950 to 0.123 and ARI/NMI to nearly zero, consistent with $z _ { s }$ staying near its zero initialization. The other objectives mainly keep shared content out: removing target sparsity, Cycle NCE, or either consistency loss raises the salient

ImageNet probe from 0.045 to between 0.086 and 0.302, and removing the image-space cycle loss also degrades reconstruction (rFID 1.78  2.84). We keep all objectives, which together give the lowest salient ImageNet probe accuracy (Appendix A.9).

## 5 CONCLUSION

We presented SAGE, which learns salient factors in the high-dimensional spatial latent of a frozen representation autoencoder using only background/target labels. Its factors faithfully reconstruct and edit individual images. Without subtype labels, its salient representation recovers digits, eyewear types, and retinal diseases, reveals finer sunglasses styles and mislabeled images, and conditions the generation of new images that keep a discovered subtype while common content varies.

## REPRODUCIBILITY STATEMENT

We will release code, trained models, and the audited FFHQ labels upon publication. All experiments use public data: MNIST and ImageNet for Digits-ImageNet, FFHQ with the FFHQ Features eyewear annotations, and OCT-Kermany. Appendix B.1 describes how each dataset is built, including the Digits-ImageNet compositing, the audited FFHQ evaluation labels, and the OCT preprocessing and splits. Appendices C and D give the Stage 1 and Stage 2 architectures, objectives, optimization settings, and checkpoints, and Appendices B.2 and B.3 give the baseline configurations. Appendices B.4, B.5, and D.2 specify the linear-probe, clustering, and generation evaluation protocols. SAGE uses the pretrained RAEv2 encoder and decoder of Singh et al. (2026), which remain frozen throughout.

## AI USE STATEMENT

In this work, we used generative AI to assist with the implementation of experimental methods and code. We have not used generative AI tools for other tasks requiring disclosure, and the remaining required-disclosure tasks are not applicable to this work. Additionally, we used generative AI tools to polish the writing of the manuscript. All AI-assisted experimental code was reviewed and tested by the authors, and the resulting experimental outputs were also verified by the authors. All AIassisted writing was also reviewed by the authors. We take responsibility for the final content of this work, including text, claims, code and results produced with the aid of generative AI.

## ACKNOWLEDGEMENTS

This work was supported by start-up funding awarded to Yuan Yuan by Boston College and by the Boston College Undergraduate Research Fellows program.

## REFERENCES

Abubakar Abid and James Zou. Contrastive variational autoencoder enhances salient features. arXiv preprint arXiv:1902.04601, 2019. URL https://arxiv.org/abs/1902.04601.

Aidas Aglinskas, Joshua K Hartshorne, and Stefano Anzellotti. Contrastive machine learning reveals the structure of neuroanatomical variation within autism. Science, 376(6597):1070– 1074, 2022. doi: 10.1126/science.abm2461. URL https://www.science.org/doi/10. 1126/science.abm2461.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou, Julien Mairal, Piotr Bojanowski, and´ Armand Joulin. Emerging properties in self-supervised vision transformers. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9630–9640. IEEE, 2021.

Florence Carton, Robin Louiset, and Pietro Gori. Double InfoGAN for contrastive analysis. In Proceedings ofThe 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings ofMachine Learning Research, pp. 172–180, 02–04 May 2024.

DCGM. FFHQ Features Dataset: Gender, age, and emotion for Flickr-Faces-HQ dataset. https: //github.com/DCGM/ffhq-features-dataset. Accessed: 2026-09-23.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Benoit Dufumier, Javiera Castillo-Navarro, Devis Tuia, and Jean-Philippe Thiran. What to align in multimodal contrastive learning? In International Conference on Learning Representations, 2025.

Dan Friedman and Adji Bousso Dieng. The vendi score: A diversity evaluation metric for machine learning. Transactions on Machine Learning Research, 2023, 2023. ISSN 2835-8856.

Yuan Gao, Chen Chen, and Jiatao Gu. One layer is enough: Adapting pretrained visual encoders for image generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pp. 4688–4697, June 2026.

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, volume 27, 2014.

Yunlong He, Gwilherm Lesne, Ziqian Liu, Micha ´ el Soumm, and Pietro Gori. Learning common and¨ salient generative factors between two image datasets. arXiv preprint arXiv:2512.12800, 2025. URL https://arxiv.org/abs/2512.12800.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30, 2017.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. URL https://arxiv.org/abs/2207.12598.

Lawrence Hubert and Phipps Arabie. Comparing partitions. Journal of Classification, 2(1):193– 218, December 1985. ISSN 1432-1343. doi: 10.1007/BF01908075. URL https://doi. org/10.1007/BF01908075.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

Daniel S. Kermany, Michael Goldbaum, Wenjia Cai, Carolina C. S. Valentim, Huiying Liang, Sally L. Baxter, Alex McKeown, Ge Yang, Xiaokang Wu, Fangbing Yan, Justin Dong, Made K. Prasadha, Jacqueline Pei, Magdalene Y. L. Ting, Jie Zhu, Christina Li, Sierra Hewett, Jason Dong, Ian Ziyar, Alexander Shi, Runze Zhang, Lianghong Zheng, Rui Hou, William Shi, Xin Fu,

Yaou Duan, Viet A. N. Huu, Cindy Wen, Edward D. Zhang, Charlotte L. Zhang, Oulan Li, Xiaobo Wang, Michael A. Singer, Xiaodong Sun, Jie Xu, Ali Tafreshi, M. Anthony Lewis, Huimin Xia, and Kang Zhang. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 172(5):1122–1131.e9, February 2018. ISSN 1097-4172 0092-8674. doi: 10.1016/j.cell.2018.02.010.

Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied´ to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. doi: 10.1109/5. 726791.

Seunghun Lee, Sunghyun Cho, and Sunghoon Im. DRANet: Disentangling representation and adaptation networks for unsupervised cross-domain adaptation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15252–15261, 2021.

Tianhong Li, Lijie Fan, Yuan Yuan, Hao He, Yonglong Tian, Rogerio Feris, Piotr Indyk, and Dina Katabi. Addressing feature suppression in unsupervised visual representations. In 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 1411–1420. IEEE, 2023.

Tianhong Li, Dina Katabi, and Kaiming He. Return of unconditional generation: a self-supervised representation generation method. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems, NIPS ’24, 2024. ISBN 9798331314385.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Robin Louiset, Edouard Duchesnay, Antoine Grigis, Benoit Dufumier, and Pietro Gori. Sepvae: a contrastive vae to separate pathological patterns from healthy ones. arXiv preprint arXiv:2307.06206, 2023. URL https://arxiv.org/abs/2307.06206.

Robin Louiset, Edouard Duchesnay, Antoine Grigis, and Pietro Gori. Separating common from salient patterns with contrastive representation learning. In International Conference on Learning Representations, volume 2024, pp. 26887–26918, 2024.

Robin Louiset, Edouard Duchesnay, Benoit Dufumier, Antoine Grigis, and Pietro Gori. Automatic discovery of disease subgroups by contrasting with healthy controls. arXiv preprint arXiv:2605.21301, 2026. doi: 10.48550/arXiv.2605.21301. URL https://arxiv.org/ abs/2605.21301.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022. URL https://openreview.net/ forum?id=aBsCjcPu\_tE.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khali-´ dov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve J´ egou, Julien Mairal, Patrick´ Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=a68SUt6zFt. Featured Certification.

Peter J Rousseeuw. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal ofComputational and Applied Mathematics, 20:53–65, 1987.

Eduardo Hugo Sanchez, Mathieu Serrurier, and Mathias Ortner. Learning disentangled representations via mutual information estimation. In European Conference on Computer Vision, 2020.

Minglei Shi, Haolin Wang, Wenzhao Zheng, Ziyang Yuan, Xiaoshi Wu, Xintao WANG, Pengfei Wan, Jie Zhou, and Jiwen Lu. Latent diffusion model without variational autoencoder. In International Conference on Learning Representations, volume 2026, pp. 154506–154537, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ fa5ddd6bac0d665c72969d79221b680a-Paper-Conference.pdf.

Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose,´ Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, Francisco Massa, Daniel¨ Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th ´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve´ Jegou, Patrick Labatut, and Piotr Bojanowski. DINOv3.´ arXiv preprint arXiv:2508.10104, 2025. URL https://arxiv.org/abs/2508.10104.

Jaskirat Singh, Boyang Zheng, Zongze Wu, Richard Zhang, Eli Shechtman, and Saining Xie. Improved baselines with representation autoencoders. arXiv preprint arXiv:2605.18324, 2026. URL https://arxiv.org/abs/2605.18324.

Michael Soumm, Alexandre Fournier Montgieux, Yunlong He, Pietro Gori, and Alasdair New-¨ son. Diff-CA: Separating common and salient factors with diffusion models. arXiv preprint arXiv:2606.06120, 2026. doi: 10.48550/arXiv.2606.06120. URL https://arxiv.org/ abs/2606.06120.

Alexander Strehl and Joydeep Ghosh. Cluster ensembles—a knowledge reuse framework for combining multiple partitions. Journal ofmachine learning research, 3(Dec):583–617, 2002.

Shengbang Tong, Boyang Zheng, Ziteng Wang, Bingda Tang, Nanye Ma, Ellis Brown, Jihan Yang, Rob Fergus, Yann LeCun, and Saining Xie. Scaling text-to-image diffusion transformers with representation autoencoders. arXiv preprint arXiv:2601.16208, 2026. URL https://arxiv. org/abs/2601.16208.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. URL https://arxiv.org/abs/ 1807.03748.

Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(86):2579–2605, 2008.

Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600–612, 2004.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. URL https://arxiv.org/abs/2508.02324.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In International Conference on Learning Representations, volume 2025, pp. 87400–87442, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/ 2025/file/d9e42b4d7163931f3689d6d6fbaa11d0-Paper-Conference.pdf.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pp. 586–595. IEEE, 2018.

Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. In International Conference on Learning Representations, volume 2026, pp. 35791–35820, 2026.

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, pp. 2223–2232, 2017.

Yu Zhu, Aidas Aglinskas, and Stefano Anzellotti. DeepCor: denoising fMRI data with contrastive autoencoders. Nature Methods, 23(2):334–337, 2026. doi: 10.1038/s41592-025-02967-x. URL https://www.nature.com/articles/s41592-025-02967-x.

# SUPPLEMENTARY MATERIALS FOR SAGE: SALIENT FACTOR DISCOVERY AND GENERATION WITH VISUAL FOUNDATION REPRESENTATIONS

## A ADDITIONAL EXPERIMENTAL RESULTS

## A.1 ADDITIONAL FACTOR-DECODING EXAMPLES

![](images/a8486dd0d97a797ba535e80ca729d5564db5460b2d7002388119fa6252571a44.jpg)  
Figure 9: Additional SAGE factor-decoding examples. Randomly selected images. For Digits-ImageNet (a–d) and FFHQ (e–h), complete decoding reconstructs the target image, common-only decoding removes the digit or eyeglasses while preserving the scene or identity, and salient-only decoding retains the digit’s shape and location or the specific eyewear style with little common content.

Figure 9 complements the cross-method comparison in Fig. 4 with SAGE-only examples. It shows that $D ( z _ { c } , \mathbf { 0 } )$ removes the target attribute while retaining common content, whereas $D ( \mathbf { 0 } , z _ { s } )$ preserves image-specific target details rather than a generic attribute.

## A.2 CLUSTERING ACROSS REPRESENTATION SPACES

Table 4 compares k-means on each method’s $\ell _ { 2 }$ -normalized salient vectors with k-means after PCA projection, using the protocol in Appendix B.5.

SAGE achieves the highest ARI and NMI in the normalized original space and both PCA readouts across the settings in Table 4. On FFHQ, its original-space scores are 0.939/0.865 for $k = 2$ and 0.937/0.865 for $\bar { k } = 3 ,$ , compared with 0.915/0.825 and 0.453/0.550 for DINOv3 + SepCLR.

## A.3 SALIENT REPRESENTATION VISUALIZATIONS ACROSS METHODS

Figure 10 compares the salient representations of all methods. On Digits-ImageNet, SepCLR and DINOv3 + SepCLR separate background from target but mix digit identities, whereas SAGE forms clear digit groups. On FFHQ, SepCLR, DINOv3 + SepCLR, and SAGE all reveal eyewear-related groups. Table 5 quantifies the background/target separation: SAGE has the highest silhouette score in both the t-SNE and original spaces on both datasets.

## A.4 FINE-GRAINED SUBTYPE STRUCTURE WITHIN SUNGLASSES

Figure 11 looks inside the sunglasses subtype. Without labels for finer styles, SAGE’s salient representation splits sunglasses into three groups, from a compact C1 to a broader C2 and a smaller C3 at its lower end: C1 (Wayfarer-like) collects thick frames with angular or rectangular lenses; C2 (mixed styles) spans rounded, narrow, and sporty frames with varied lens tint; and C3 (Aviator-like) collects thin-rimmed, rounded or teardrop-shaped lenses, often reflective. The three groups share one coarse label yet differ in frame geometry and lens shape, showing that the salient factor captures appearance structure finer than the annotated subtypes.

Table 4: Clustering readout ablation on ℓ -normalized salient representations (ARI/NMI). All vectors are normalized before optional projection. Raw denotes no dimensionality reduction; PCA-50 retains min(50, Dim.) components. Scores are means over k-means seeds 0, 1, 2 . FFHQ uses manually verified annotations, with 39 no-glasses images excluded for k = 2 and included for k = 3.
<table><tr><td></td><td></td><td colspan="3">Digits-ImageNet  $( k = \bar { 1 } 0 , n = \bar { 2 } 4 , 7 3 9 )$ </td><td colspan="3">FFHQ  $( k = 2 , n = \mathrm { \hat { 3 } } , 3 9 8 )$ </td><td colspan="3">FFHQ  $( k = 3 , n = \overset { \cdot } { 3 } , 4 3 7 )$ </td></tr><tr><td>Method</td><td>Dim.</td><td>PCA-2D</td><td>PCA-50</td><td>Raw</td><td>PCA-2D</td><td>PCA-50</td><td>Raw</td><td>PCA-2D</td><td>PCA-50</td><td>Raw</td></tr><tr><td>ARI↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>cVAE</td><td>64</td><td>0.000</td><td>0.001</td><td>0.001</td><td>0.001</td><td>0.000</td><td>0.001</td><td>0.036</td><td>0.017</td><td>0.016</td></tr><tr><td>SepVAE</td><td>64</td><td>0.006</td><td>0.009</td><td>0.009</td><td>-0.001</td><td>-0.001</td><td>-0.001</td><td>0.012</td><td>0.019</td><td>0.017</td></tr><tr><td>Double-InfoGAN</td><td>64</td><td>0.005</td><td>0.001</td><td>0.001</td><td>0.324</td><td>0.292</td><td>0.298</td><td>0.247</td><td>0.303</td><td>0.303</td></tr><tr><td>SepCLR</td><td>32</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.808</td><td>0.822</td><td>0.822</td><td>0.441</td><td>0.455</td><td>0.455</td></tr><tr><td>DINOv3 + SepCLR</td><td>32</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.415</td><td>0.915</td><td>0.915</td><td>0.187</td><td>0.453</td><td>0.453</td></tr><tr><td>SAGE (Ours)</td><td>1024</td><td>0.187</td><td>0.336</td><td>0.337</td><td>0.930</td><td>0.937</td><td>0.939</td><td>0.902</td><td>0.936</td><td>0.937</td></tr><tr><td>NMI↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>cVAE</td><td>64</td><td>0.001</td><td>0.003</td><td>0.003</td><td>0.001</td><td>0.001</td><td>0.001</td><td>0.047</td><td>0.026</td><td>0.025</td></tr><tr><td>SepVAE</td><td>64</td><td>0.012</td><td>0.019</td><td>0.019</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.017</td><td>0.031</td><td>0.029</td></tr><tr><td>Double-InfoGAN</td><td>64</td><td>0.010</td><td>0.003</td><td>0.003</td><td>0.318</td><td>0.297</td><td>0.299</td><td>0.319</td><td>0.380</td><td>0.380</td></tr><tr><td>SepCLR</td><td>32</td><td>0.001</td><td>0.001</td><td>0.001</td><td>0.680</td><td>0.696</td><td>0.696</td><td>0.522</td><td>0.534</td><td>0.534</td></tr><tr><td>DINOv3 + SepCLR</td><td>32</td><td>0.001</td><td>0.001</td><td>0.001</td><td>0.379</td><td>0.825</td><td>0.825</td><td>0.288</td><td>0.550</td><td>0.550</td></tr><tr><td>SAGE (Ours)</td><td>1024</td><td>0.332</td><td>0.471</td><td>0.472</td><td>0.850</td><td>0.863</td><td>0.865</td><td>0.813</td><td>0.864</td><td>0.865</td></tr></table>

![](images/6514b0fcc384ed4a8b98810276a77ced8a6b225b2be92935a69c8db5a68eb506.jpg)  
Figure 10: t-SNE visualizations of salient representations across contrastive analysis methods. Columns show cVAE, SepVAE, Double-InfoGAN, SepCLR, DINOv3 + SepCLR, and SAGE. Rows show Digits-ImageNet background-versus-target and digit-colored target representations, then their FFHQ counterparts with audited eyewear labels. Blue/orange denote background/target; green/purple/red denote reading glasses/sunglasses/audited no-glasses images.

Table 5: Silhouette scores for BG-vs.-TG salient separation in t-SNE and original spaces. Scores treat background (BG) and target (TG) salient representations as the two groups. The silhouette coefficient (Rousseeuw, 1987) measures within-group cohesion relative to separation; higher is better.
<table><tr><td>Method</td><td>Digits t-SNE</td><td>Digits original</td><td>FFHQ t-SNE</td><td>FFHQ original</td></tr><tr><td>cVAE</td><td>0.022</td><td>0.006</td><td>0.004</td><td>0.027</td></tr><tr><td>SepVAE</td><td>0.003</td><td>0.006</td><td>0.394</td><td>0.206</td></tr><tr><td>Double-InfoGAN</td><td>0.248</td><td>0.013</td><td>0.188</td><td>0.050</td></tr><tr><td>SepCLR</td><td>0.381</td><td>0.546</td><td>0.449</td><td>0.752</td></tr><tr><td>DINOv3 + SepCLR</td><td>0.365</td><td>0.642</td><td>0.387</td><td>0.551</td></tr><tr><td>SAGE</td><td>0.417</td><td>0.820</td><td>0.462</td><td>0.873</td></tr></table>

![](images/1d2c43941aa18a19d6e9982d615b22b9b6fd3af6e7985f8d51039c86c4618201.jpg)  
Figure 11: Fine-grained sunglasses structure in SAGE’s salient representation. Left: FFHQ target embeddings colored by audited eyewear labels, with C1–C3 marking subgroups within sunglasses. Right: example images from each subgroup: Wayfarer-like (C1, n = 168), mixed styles (C2, n = 390), and Aviator-like (C3, n = 175). Group names describe the displayed examples; no style labels are used in training.

## A.5 FFHQ EYEWEAR LABEL AUDITING AND REMAINING SAGE ERRORS

In the t-SNE of SAGE’s salient representations of FFHQ target images, one group stood apart (Section 4.3). Inspecting its images showed faces without eyeglasses that the Azure Face API had labeled as wearing glasses. We therefore manually audited all 3,437 target images in our FFHQ validation split and corrected 125 Azure Face API subtype labels (3.64%). We then examined the locations and neighborhoods of all manually verified samples in SAGE’s salient representation. Some faces with face paint or hat-brim shadows were incorrectly labeled as sunglasses. The final class counts are reported in Appendix B.1. Figure 12 shows all 125 corrected samples alongside their locations in the salient-space visualization.

Annotators assigned each label from the visible eyewear in the image, without seeing SAGE’s cluster assignments or t-SNE. Table 1 uses the same corrected labels for all methods. SAGE surfaced the no-glasses errors, 34 of which form a separate group; corrections between reading glasses and sunglasses were identified by the manual audit rather than by SAGE.

Remaining no-glasses cases. Figure 13 examines the 39 audited no-glasses images in the target dataset. SAGE places 34 of them in a separate group (panel c); the other five lie with glasses groups, four with reading glasses and one with sunglasses (panel b). These five show face paint around the eyes or a dark costume eye mask, whose contours resemble eyewear, and their nearest neighbors include both genuine glasses and similarly decorated no-glasses faces (cosine similarity 0.91–0.94).

![](images/2418dedc28be7ec63ad1c7967ade7bf1ff0bf67b53ce480a2094e5e940c3a2f5.jpg)  
Figure 12: Azure eyewear annotation errors identified by manual inspection of FFHQ. Left: t-SNE of SAGE’s target salient representations, colored by manually verified labels; outlined points mark the 125 Azure label disagreements. Right: all 125 corrected images, grouped by verified label: sunglasses (65), no glasses (39), and reading glasses (21). Tile titles give the original Azure category and image ID.

The FFHQ scores in Table 1 use only the verified reading-glasses and sunglasses images, so these cases do not enter them.

![](images/410c00ec5bbbf69ecafba272d2474eb3ba93ceaffb300d267d7a2bce74f24af9.jpg)

(b) No-glasses images clustered with glasses  
![](images/3ec499dfe1e8eab90f3bd34ccd0a9f1a6ccfc16f1bb387d7f3d3a63ab0a576d5.jpg)  
Figure 13: Remaining SAGE errors among manually verified no-glasses FFHQ images. (a) Target salient-space t-SNE, colored by audited labels: reading glasses (green, 2,636), sunglasses (purple, 762), and no glasses (red, 39). Black-outlined red points identify five no-glasses cases associated with glasses groups. (b) Each numbered row shows one query and its six nearest neighbors; border colors indicate human labels and values report cosine similarity. Arrows mark the group each query is associated with. (c) The other 34 no-glasses images form a separate group.

## A.6 ADDITIONAL SWAPPING AND ERASURE EXAMPLES

Figure 14 extends Fig. 6 with more examples of both manipulations on Digits-ImageNet and FFHQ. Swapping a target salient factor into a background common factor transfers the target attribute to the donor content, whereas decoding a target common factor alone removes that attribute.

![](images/c46029f220c0a8e946473eb8659fb649b2d5bbccc01dccacfd97c958b3640a12.jpg)  
Figure 14: Additional salient swapping and erasure examples. Left (BG TG): $D \big ( z _ { c } ^ { b } , z _ { s } ^ { t } \big )$ combines a background image’s common factor with a target image’s salient factor. Right (TG BG): $D ( z _ { c } ^ { t } , \mathbf { 0 } )$ decodes the target common factor alone. Rows show Digits-ImageNet and FFHQ examples; red borders mark decoded outputs.

## A.7 STAGE 1 METRICS ON OCT-KERMANY

To assess disease-subtype structure under normal/disease labels (Section 4.5), Table 6 reports salient linear-probe and clustering metrics on the official OCT-Kermany test set (Kermany et al., 2018), alongside reconstruction quality. Appendix B.1 describes the dataset, splits, and preprocessing, and Appendix C.2 the training settings.

Table 6: Stage-1 metrics on OCT-Kermany. Evaluation uses the official test set (Kermany et al., 2018), with 250 normal images and 250 images from each of CNV, DME, and DRUSEN, from patients not included in training or validation. LP Acc. reports mean stratified five-fold linear-probe accuracy on the 750 target images; ARI/NMI use k-means directly on SAGE’s $\ell _ { 2 }$ -normalized, 1,024- dimensional salient representations obtained by global max pooling (GMP), without dimensionality reduction. With only 1,000 evaluation images, rFID is sensitive to finite-sample bias and is not directly comparable to scores computed on larger evaluation sets.
<table><tr><td>Method</td><td>rFID↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>LP Acc. ↑</td><td>ARI/NMI↑</td></tr><tr><td>SAGE (Ours)</td><td>27.21</td><td>24.55</td><td>0.448</td><td>0.273</td><td>0.960</td><td>0.415/0.415</td></tr></table>

## A.8 SALIENT-CONDITIONED GENERATION

Qualitative results. Salient conditioning generates new faces and scenes that preserve the reference digit or eyewear type (Fig. 15), whereas Raw DINOv3 GMP reproduces the reference person

![](images/02dcac7371883c732a9c54f0a87544320ef01729faba2030b6fa8d8f903da4b4.jpg)  
Figure 15: Generation conditioned on a reference image. For FFHQ (left) and Digits-ImageNet (right), each column shows a reference (top) and one sample conditioned on Raw DINOv3 GMP (middle) or on SAGE’s salient representation (bottom).

Table 7: Full loss ablation of SAGE on Digits-ImageNet. ARI/NMI use Euclidean k-means on ℓ -normalized original salient representations, without dimensionality reduction (Appendix B.5). ARI/NMI are means over k-means seeds 0, 1, 2 ; standard deviations are omitted because they are at most 0.0007. Digit/ImageNet LP: linear-probe accuracy for digit or ImageNet labels from the indicated representation; Raw z: the unfactorized latent, whose probes use z itself. Objectives are named as in Table 3.
<table><tr><td rowspan="2">Configuration</td><td colspan="4">Full reconstruction</td><td colspan="2">Common-only</td><td colspan="2">Digit LP</td><td colspan="2">ImageNet LP</td><td>Clustering</td></tr><tr><td>rFID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>rFID↓</td><td>SSIM↑</td><td>zs ↑</td><td> $z _ { c } \downarrow$ </td><td> $z _ { c } \uparrow$ </td><td> $z _ { s } \downarrow$ </td><td>ARI/NMI↑</td></tr><tr><td>Raw z</td><td>0.39</td><td>22.18</td><td>0.613</td><td>0.157</td><td>N/A</td><td>N/A</td><td>0.328</td><td>N/A</td><td>0.765</td><td>N/A</td><td>0.000/0.001</td></tr><tr><td>SAGE (all)</td><td>1.78</td><td>20.56</td><td>0.571</td><td>0.216</td><td>1.89</td><td>0.569</td><td>0.950</td><td>0.336</td><td>0.765</td><td>0.045</td><td>0.337/0.472</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { { s w a p } } } ^ { G }$ </td><td>1.01</td><td>20.67</td><td>0.566</td><td>0.210</td><td>4.88</td><td>0.544</td><td>0.123</td><td>0.356</td><td>0.763</td><td>0.169</td><td>0.000/0.001</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { T G - S p } }$ </td><td>1.15</td><td>21.06</td><td>0.598</td><td>0.193</td><td>1.36</td><td>0.585</td><td>0.923</td><td>0.656</td><td>0.758</td><td>0.302</td><td>0.274/0.409</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { c N C E } }$ </td><td>1.29</td><td>20.78</td><td>0.576</td><td>0.205</td><td>1.58</td><td>0.573</td><td>0.950</td><td>0.335</td><td>0.767</td><td>0.100</td><td>0.348/0.479</td></tr><tr><td> $\mathcal { L } _ { \mathrm { B G - s p } } \colon \ell _ { 2 } \to \ell _ { 1 }$ </td><td>1.74</td><td>20.17</td><td>0.568</td><td>0.218</td><td>2.00</td><td>0.566</td><td>0.953</td><td>0.320</td><td>0.764</td><td>0.067</td><td>0.379/0.505</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { c y c } }$ </td><td>2.84</td><td>19.88</td><td>0.562</td><td>0.236</td><td>3.46</td><td>0.559</td><td>0.951</td><td>0.366</td><td>0.762</td><td>0.086</td><td>0.360/0.496</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { l s c } }$ </td><td>1.82</td><td>20.30</td><td>0.568</td><td>0.218</td><td>2.25</td><td>0.567</td><td>0.953</td><td>0.353</td><td>0.761</td><td>0.107</td><td>0.361/0.494</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { B G \mathrm { - } S p } }$ </td><td>1.89</td><td>20.44</td><td>0.573</td><td>0.218</td><td>2.26</td><td>0.571</td><td>0.951</td><td>0.305</td><td>0.760</td><td>0.061</td><td>0.356/0.487</td></tr></table>

or scene almost exactly and may change or distort the digit. Multiple samples from one reference, as measured by Vendi score, appear in Fig. 3 (top right).

## A.9 FULL LOSS ABLATION

Table 7 expands the ablations in Section 4.7 with the unfactorized latent as a reference and probes of both factors.

Cycle NCE and shared-category leakage. Cycle NCE keeps shared ImageNet content out of the salient factor: it lowers the salient ImageNet probe from 0.100 to 0.045 while digit accuracy stays at 0.950.

Choosing the background penalty. We use a squared $\ell _ { 2 }$ background penalty rather than $\ell _ { 1 }$ because it better keeps shared content out of the salient factor (salient ImageNet probe 0.045 vs. 0.067) and improves common-only reconstruction (rFID 1.89 vs. 2.00).

## B DATASETS, BASELINES, AND EVALUATION DETAILS

## B.1 DATASET DETAILS

For all datasets, training uses only background/target labels. Subtype labels are used only for evaluation and are not provided during representation learning.

Digits-ImageNet. We construct the target dataset by superimposing MNIST digits on ImageNet photographs; the background dataset contains photographs without added digits. We retain the ImageNet training and validation splits and randomly assign approximately half of each split to each dataset, using different random seeds for the training and validation splits. Background and target datasets use disjoint source photographs. All available ImageNet class directories are included, with no explicit class exclusions.

Photographs are center-cropped to a square and resized to 512  512 pixels. Each target receives one MNIST example sampled uniformly with replacement. Training images use the MNIST training set, while validation images use the MNIST test set. Digits are resized to 224  224 pixels using bilinear interpolation and rendered in white. The top-left coordinates are sampled uniformly from 0, . . . , 288 along each axis, keeping the digit patch within the image. We use the resized digit intensities as an alpha mask with an opacity multiplier of 1.0. Digit size and color are fixed, and digit classes are not explicitly balanced.

Linear-probe and clustering evaluations use 24,739 target validation images. The per-class counts for digits 0–9 are 2,464, 2,699, 2,685, 2,521, 2,358, 2,199, 2,337, 2,613, 2,400, and 2,463.

FFHQ. The original FFHQ dataset does not provide attribute-level labels indicating whether a face wears glasses. We use the eyewear annotations from the FFHQ Features Dataset (DCGM), which were generated automatically using Microsoft’s Azure Face API, to construct the background and target datasets. During training, we use 10,000 images without glasses as the background dataset and 10,000 images with glasses as the target dataset. Because the annotations are automatic, they can contain errors. After our manual audit of all 3,437 validation images (Appendix A.5), the validation set contains 2,636 images with reading glasses, 762 with sunglasses, and 39 without glasses.

OCT-Kermany. We use the OCT-Kermany data of Kermany et al. (2018), treating normal scans as background and CNV, DME, and DRUSEN scans jointly as target. The original Kermany export contains saturated-white borders introduced by acquisition tilt or registration correction, rather than by our dataset construction. These borders can extend 300 pixels into a 496-pixel-high scan. Before cleaning, 60.1% of the 109,309 scans have at least one edge for which more than half the pixels are saturated white, creating an easily exploitable non-anatomical cue for an encoder.

We preprocess scans in grayscale. First, we remove rows or columns that are at least 95% white (intensity 0.90) along their full extent. To preserve retinal tissue under oblique white wedges, we then replace only near-white regions connected to an image boundary and larger than 64 pixels, rather than rectangularly cropping to the deepest wedge. Replacement pixels are Gaussian noise matched to the mean and standard deviation of the darkest 40% of pixels in the scan, which cor respond to the vitreous. Scans for which this replacement would cover over 60% of the image are retained unchanged. Finally, we preserve aspect ratio, bicubically resize to height 384, and centercrop to 384  384 before storing RGB images. This avoids the aspect-ratio distortions caused by directly resizing the eight native scan widths to a square. During representation learning, images are resized from 384 to 256 for DINOv3, without data augmentation.

Table 8 gives the resulting splits. We train on the cleaned training set (97,478 scans). All OCT evaluations use the official 1,000-image test set, whose patients do not appear in training.

## B.2 BASELINE DETAILS

We use the default configurations of the generative CA baselines cVAE (Abid & Zou, 2019), Sep-VAE (Louiset et al., 2023), and Double-InfoGAN (Carton et al., 2024), and of the contrastive CA baseline SepCLR (Louiset et al., 2024). Double-InfoGAN uses its native 128 128 input resolution; all remaining baselines use 256 256 inputs. DINOv3 + SepCLR trains SepCLR’s objectives on

Table 8: OCT-Kermany split statistics after preprocessing. CNV, DME, and DRUSEN are the target diseases; normal scans are background.
<table><tr><td>Split</td><td>NORMAL</td><td>CNV</td><td>DME</td><td>DRUSEN</td><td>Total</td><td>Patients</td></tr><tr><td>Train</td><td>46,026</td><td>33,485</td><td>10,213</td><td>7,754</td><td>97,478</td><td>4,772</td></tr><tr><td>Val.</td><td>5,114</td><td>3,720</td><td>1,135</td><td>862</td><td>10,831</td><td>3,329</td></tr><tr><td>Test</td><td>250</td><td>250</td><td>250</td><td>250</td><td>1,000</td><td>635</td></tr></table>

SAGE’s frozen RAEv2 encoder with the same common/salient encoder architecture (Appendix B.3), isolating the contribution of the training framework from that of the visual backbone. We additionally evaluate the unfactorized RAEv2 latent z (DINOv3 with MLS), denoted DINOv3 in figures. For generation, Raw DINOv3 GMP pools z identically to salient conditioning, so the two conditions differ only in whether the salient encoder is applied.

## B.3 DINOV3 + SEPCLR BASELINE

DINOv3 + SepCLR shares SAGE’s frozen RAEv2 encoder (DINOv3-L with multi-layer summation and RAEv2 latent normalization) and common/salient encoder architecture, and differs in its objectives and evaluation readout (Table 9). Each branch applies global average pooling (GAP), following SepCLR, to obtain a 1,024-dimensional vector. A fully connected layer maps it to the 32-dimensional representation used for evaluation. A separate MLP projector (32 128 BN ReLU 32, with batch normalization, BN) supplies only the contrastive and k-JEM losses; its output is not used for evaluation. Only the two common/salient Transformers, two fully connected layers, and two projectors are trained. We retain SepCLR’s alignment, uniformity, salient regular ization, and JEM losses and its original loss hyperparameters (Louiset et al., 2024).

Table 9: Components of DINOv3 + SepCLR and SAGE.
<table><tr><td>Component</td><td>DINOv3 + SepCLR</td><td>SAGE</td></tr><tr><td>Frozen visual backbone</td><td>DINOv3-L (MLS)</td><td>DINOv3-L (MLS)</td></tr><tr><td>Latent normalization</td><td>RAEv2</td><td>RAEv2</td></tr><tr><td>Common/salient Transformers</td><td>Shared architecture</td><td>Shared architecture</td></tr><tr><td>Evaluation pooling</td><td>GAP</td><td>GMP</td></tr><tr><td>Evaluation representation</td><td>32 dimensions</td><td>1,024 dimensions</td></tr><tr><td>Training objectives</td><td>SepCLR objectives</td><td>SAGE objectives</td></tr></table>

## B.4 LINEAR-PROBE EVALUATION

We use the evaluation pools defined in Appendix B.1.

Digits-ImageNet. We evaluate frozen representations with stratified five-fold cross-validation. We standardize the features and evaluate them using a logistic-regression classifier. No class weighting is used, and predictions are scored using ordinary accuracy, as the digit classes are approximately balanced. Reported digit linear-probe scores are mean accuracies across the five folds.

FFHQ. We evaluate frozen salient representations on all 3,398 images with the manually verified reading-glasses/sunglasses labels, using class-balanced logistic regression and stratified five-fold cross-validation. The score is balanced accuracy, the mean of the two class recalls. Table 1 report the mean across folds. The main FFHQ SAGE result uses target salient sparsity weight λ = 0.01.

## B.5 CLUSTERING AND T-SNE PROTOCOL

Salient representation. Clustering uses only target-group salient representations, extracted with the trained encoders frozen. The baseline representations used for evaluation are flat vectors: 32 dimensions for SepCLR and DINOv3 + SepCLR and 64 for cVAE, SepVAE, and Double-InfoGAN. SAGE produces an N 1024 salient token representation, which we reduce to a 1,024-dimensional vector by channel-wise global max pooling (GMP) over the token axis, as in linear-probe evaluation. These pooled SAGE vectors define SAGE’s original salient representation space. Within each quantitative evaluation setting, all methods use the same images and labels.

Cluster count. We use the evaluation pools defined in Appendix B.1. For all methods, the number of clusters is set to the number of evaluation classes: $k = 1 0$ for Digits-ImageNet, and $k = 2$ or $k = 3$ for the two FFHQ settings with audited labels, and $k = 3$ for the three OCT-Kermany diseases. Table 1 uses the FFHQ two-class setting, while Table 4 reports both settings.

t-SNE visualization. For qualitative visualization only, we obtain two-dimensional embeddings using scikit-learn t-SNE (van der Maaten & Hinton, 2008). Pairwise cosine distances are computed from the full normalized feature vectors. The first two principal components provide the initial twodimensional coordinates, which t-SNE then optimizes. We use perplexity 30, an automatic learning rate, and 1,000 iterations, with a fixed random seed.

Clustering and metrics. Let $f _ { i }$ denote the original salient vector of image i and $u _ { i } = f _ { i } / \| f _ { i } \| _ { 2 }$ its unit-norm representation. We apply Euclidean k-means directly to $\{ u _ { i } \}$ , using the cluster counts specified above and 10 initializations. Adjusted Rand Index (ARI) (Hubert & Arabie, 1985) and Normalized Mutual Information (NMI) (Strehl & Ghosh, 2002) measure agreement with evaluation labels and are averaged over k-means seeds 0, 1, 2 , not independent training runs. Tables 1, $3 , 6 ,$ and 7 use this protocol; in Table 4, Raw denotes it, and PCA-2D and PCA-50 apply PCA to $\{ u _ { i } \}$ before k-means.

## C ADDITIONAL STAGE 1 DETAILS

This section describes the Stage 1 network architectures and training settings, and provides the complete definitions of the objectives summarized in Section 3.1.

## C.1 NETWORK ARCHITECTURES

Figure 16 illustrates the architecture used for the common and salient encoders, which map frozen RAE features to the two additive representations. Figure 17 shows the discriminator, whose shared frozen DINOv1 ViT-S/8 backbone (Caron et al., 2021) supplies features to separate background and target expert heads for the swap adversarial objective. This backbone is independent of the DINOv3 encoder being factorized.

![](images/1c78531fbcd9b59acd763b8897917f54875528c49d5d08c6872daa3d2c2816f4.jpg)  
Figure 16: Common and salient encoder architecture. Input and output convolutions use kernel size 3. The Transformer block applies layer normalization, self-attention, and a feed-forward network (FFN), with residual connections around the attention and FFN sublayers. The two encoders produce the common and salient representations, respectively. In the salient encoder, both the weights and biases of Conv out are initialized to zero.

![](images/9949daf8c93dccc8f903da2405bf81a5ea1fcbcdcddb6e89a918d32d30df319c.jpg)  
Figure 17: Shared multi-expert discriminator architecture. A frozen DINOv1 ViT-S/8 backbone provides features from layers 2, 5, 8, 11, and the final layer to separate background (BG) and target (TG) experts. Each expert contains trainable heads for the corresponding feature levels, enabling group-specific real/fake discrimination. Snowflake and flame symbols indicate frozen and trainable components, respectively.

## C.2 STAGE 1 TRAINING SETUP

Shared optimization settings. For both FFHQ and Digits-ImageNet, we use four NVIDIA H200 GPUs with 32 images per GPU, giving a global batch size of 128 (approximately 64 background and 64 target images). The common/salient encoders and trainable discriminator heads use AdamW with the same fixed learning rate of $3 \times 1 0 ^ { - 5 } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , weight decay 0.01, and $\varepsilon = 1 0 ^ { - 6 }$ . Training uses bfloat16 mixed precision and gradient-norm clipping at 1.0. The DINOv3 backbone and RAE decoder remain frozen, as does the discriminator’s feature-extraction backbone; only the common/salient encoders and discriminator heads are optimized.

Dataset-specific settings. We train for 200 epochs on FFHQ and three epochs on Digits-ImageNet and use the last saved checkpoint. For FFHQ, we set the target salient $L _ { 1 }$ weight to $\lambda = 0 . 0 1 ;$ for Digits-ImageNet, $\lambda = 1 . 0$ . The background salient $L _ { 2 }$ weight is 1.0 for both datasets. For OCT-Kermany, we use two NVIDIA H100 GPUs with 32 images per GPU (global batch size 64) and a class-balanced sampler with 32 normal and 32 diseased B-scans per batch. We otherwise use the shared optimization and frozen-component settings above, train for 60 epochs, and use the last saved checkpoint. The target salient $L _ { 1 }$ weight is $\lambda = 0 . 0 1$ and the background salient $L _ { 2 }$ weight is 1.0.

## C.3 SWAP ADVERSARIAL OBJECTIVE

The generator, consisting of the common and salient encoders, minimizes the non-saturating adversarial objective equation 2. The discriminator is trained with the hinge objective

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { s w a p } } ^ { D } = \mathbb { E } _ { \hat { x } ^ { b } , \hat { x } ^ { t } } \left[ \ell _ { + } \left( \Delta _ { b } ( \hat { x } ^ { b } ) \right) + \ell _ { - } \left( \Delta _ { b } ( \hat { x } _ { \mathrm { f a k e } } ^ { b } ) \right) \right] } \\ & { \quad \quad \quad + \mathbb { E } _ { \hat { x } ^ { b } , \hat { x } ^ { t } } \left[ \ell _ { + } \left( \Delta _ { t } ( \hat { x } ^ { t } ) \right) + \ell _ { - } \left( \Delta _ { t } ( \hat { x } _ { \mathrm { f a k e } } ^ { t } ) \right) \right] , } \end{array}\tag{7}
$$

where

$$
\ell _ { + } ( s ) = \operatorname* { m a x } ( 0 , 1 - s ) , \qquad \ell _ { - } ( s ) = \operatorname* { m a x } ( 0 , 1 + s ) .\tag{8}
$$

Here, s is the scalar real/fake score predicted by the corresponding discriminator head. The discriminator and common and salient encoders are updated in alternation at each training step.

## C.4 CONSISTENCY OBJECTIVES

We write $\mathrm { s g } [ \cdot ]$ for the stop-gradient operator.

Image-space cycle consistency. To enforce consistency in the learned representation space for fake background and fake target images, we re-encode the fake images through the frozen backbone and the same common and salient encoders:

$$
\hat { z } ^ { b } = E \big ( \hat { x } _ { \mathrm { f a k e } } ^ { b } \big ) , \qquad \hat { z } ^ { t } = E \big ( \hat { x } _ { \mathrm { f a k e } } ^ { t } \big ) .\tag{9}
$$

We then decompose the re-encoded representations as

$$
\hat { z } _ { c } ^ { b } = E _ { c } ( \hat { z } ^ { b } ) , \quad \hat { z } _ { s } ^ { b } = E _ { s } ( \hat { z } ^ { b } ) , \quad \hat { z } _ { c } ^ { t } = E _ { c } ( \hat { z } ^ { t } ) , \quad \hat { z } _ { s } ^ { t } = E _ { s } ( \hat { z } ^ { t } ) .\tag{10}
$$

The cycle objective (Zhu et al., 2017) encourages these re-encoded representations to recover the factors that produced the fake images:

$$
\begin{array} { r l } &  { \mathscr { L } _ { \mathrm { c y c } } } = { \mathbb { E } _ { { { x } ^ { b } } , { { x } ^ { t } } } \Big [ \left\| \hat { { \boldsymbol { z } } } _ { c } ^ { t } - \mathrm { s g } [ { { \boldsymbol { z } } _ { c } ^ { b } } ] \right\| _ { 2 } ^ { 2 } + \left\| \hat { { \boldsymbol { z } } } _ { s } ^ { t } - \mathrm { s g } [ { { \boldsymbol { z } } _ { s } ^ { t } } ] \right\| _ { 2 } ^ { 2 } } \\ & { \qquad + \left\| \hat { { \boldsymbol { z } } } _ { c } ^ { b } - \mathrm { s g } [ { { \boldsymbol { z } } _ { c } ^ { t } } ] \right\| _ { 2 } ^ { 2 } + \left\| \hat { { \boldsymbol { z } } } _ { s } ^ { b } \right\| _ { 2 } ^ { 2 } \Big ] . } \end{array}\tag{11}
$$

The first two terms encourage recovery of the donor background common representation $z _ { c } ^ { b }$ and the target salient representation $z _ { s } ^ { t }$ from the fake target image $\bar { \hat { x } } _ { \mathrm { f a k e } } ^ { t }$ . The last two terms encourage recovery of the target common representation $z _ { c } ^ { t }$ from the fake background image $\hat { x } _ { \mathrm { f a k e } } ^ { b }$ while penalizing its salient activations.

Latent-space swap consistency. To encourage additive composability and recovery of the constituent representations directly in the normalized representation space, we sample a random permutation π for a minibatch $\{ ( z _ { c } ^ { i } , \bar { z } _ { s } ^ { i } ) \} _ { i = 1 } ^ { B }$ and construct

$$
z _ { \mathrm { c y c } } ^ { i } = z _ { c } ^ { i } + z _ { s } ^ { \pi ( i ) } , \qquad z _ { c , \mathrm { c y c } } ^ { i } = E _ { c } ( z _ { \mathrm { c y c } } ^ { i } ) , \qquad z _ { s , \mathrm { c y c } } ^ { i } = E _ { s } ( z _ { \mathrm { c y c } } ^ { i } ) .\tag{12}
$$

Unlike image-space cycle consistency, this operation does not decode the composed representation or re-run the frozen backbone. The loss encourages the encoders to recover the two factors that formed each composition:

$$
\mathcal { L } _ { \mathrm { l s c } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left[ \left\| z _ { c , \mathrm { c y c } } ^ { i } - \mathrm { s g } [ z _ { c } ^ { i } ] \right\| _ { 2 } ^ { 2 } + \left\| z _ { s , \mathrm { c y c } } ^ { i } - \mathrm { s g } [ z _ { s } ^ { \pi ( i ) } ] \right\| _ { 2 } ^ { 2 } \right] .\tag{13}
$$

The stop-gradient targets prevent the source representations from moving to match the re-separated outputs. This objective encourages the recovered common and salient representations to be consistent across a wider variety of donors.

## C.5 CYCLE NCE LOSS

For each paired background and target sample, let $z _ { c } ^ { b , i }$ and $z _ { c } ^ { t , i }$ denote the original common and salient representations, and let $\hat { z } _ { c } ^ { t , i } = \overline { { E _ { c } ( \hat { z } ^ { t , i } ) } }$ and $\hat { z } _ { s } ^ { t , i } = E _ { s } ( \hat { z } ^ { t , i } )$ denote the factors recovered from the re-encoded fake target $\hat { z } ^ { t , i } = E ( \hat { x } _ { \mathrm { f a k e } } ^ { t , i } )$ , as in Appendix $\mathrm { C . 4 }$ . Both are therefore recovered from the same swapped image. We globally average-pool and $\ell _ { 2 } \cdot$ -normalize all representations:

$$
\bar { z } = \frac { \mathrm { G A P } ( z ) } { \| \mathrm { G A P } ( z ) \| _ { 2 } } .\tag{14}
$$

For each salient anchor $\bar { \hat { z } } _ { s } ^ { t , i }$ , the corresponding original salient representation $\overline { { z _ { s } ^ { t , i } } }$ is the only positive. The original common representations $\{ \overline { { z _ { c } ^ { b , j } } } \} _ { j = 1 } ^ { B }$ and their re-encoded counterparts $\{ \bar { \hat { z } } _ { c } ^ { t , j } \} _ { j = 1 } ^ { B }$ are negatives. Other salient representations are excluded from the negative set. The common branch is defined symmetrically.

Figure 18 visualizes this pair construction. Each re-encoded representation is aligned with its corresponding original representation from the same factor, while original and re-encoded representation from the other factor serve as negatives. Same-factor off-diagonal pairs are excluded.

With temperature $\kappa ,$ the salient-side loss is

$$
\ell _ { s } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp \left( \bar { \hat { z } } _ { s } ^ { t , i } \cdot \mathrm { s g } \left[ z _ { s } ^ { t , i } \right] / \kappa \right) } { \exp \left( \bar { \hat { z } } _ { s } ^ { t , i } \cdot \mathrm { s g } \left[ z _ { s } ^ { t , i } \right] / \kappa \right) + \displaystyle \sum _ { j = 1 } ^ { B } \left[ \exp \left( \bar { \hat { z } } _ { s } ^ { t , i } \cdot \mathrm { s g } \left[ z _ { c } ^ { b , j } \right] / \kappa \right) + \exp \left( \bar { \hat { z } } _ { s } ^ { t , i } \cdot \bar { \hat { z } } _ { c } ^ { t , j } / \kappa \right) \right] }\tag{15}
$$

<table><tr><td colspan="9">Cycle NCE Loss</td></tr><tr><td></td><td>21</td><td>22 </td><td></td><td>2B</td><td>21</td><td>z2</td><td></td><td>2B </td></tr><tr><td>21 2</td><td></td><td></td><td></td><td></td><td>N</td><td>N</td><td>N</td><td>N</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>N</td><td>N</td><td>N</td><td>N</td></tr><tr><td>2B</td><td></td><td></td><td></td><td></td><td>N</td><td>N</td><td>N</td><td>N</td></tr><tr><td>z1</td><td></td><td></td><td></td><td></td><td>N</td><td>N</td><td>N</td><td>N</td></tr><tr><td>z2</td><td>P</td><td></td><td></td><td></td><td>N</td><td>N</td><td>N</td><td>N</td></tr><tr><td></td><td></td><td>P</td><td></td><td></td><td>N</td><td>N</td><td>N</td><td>N</td></tr><tr><td>2B</td><td></td><td>…</td><td>..</td><td></td><td>“=</td><td>•  </td><td></td><td></td></tr><tr><td>z1</td><td>N</td><td>N</td><td>N</td><td>P</td><td>N</td><td>N</td><td>N</td><td>N</td></tr><tr><td>2</td><td>N</td><td>N</td><td>N</td><td>N</td><td></td><td></td><td></td><td></td></tr><tr><td>.·</td><td>N</td><td>N</td><td>N</td><td>N N</td><td></td><td></td><td></td><td></td></tr><tr><td>2B</td><td>N</td><td>N</td><td>N</td><td>N</td><td></td><td></td><td></td><td></td></tr><tr><td>z1</td><td>N</td><td>N</td><td>N</td><td>N</td><td>P</td><td></td><td></td><td></td></tr><tr><td>z2</td><td>N</td><td>N</td><td>N</td><td>N</td><td></td><td>P</td><td></td><td></td></tr><tr><td></td><td>•..</td><td>•• •</td><td>..</td><td></td><td></td><td>…</td><td> </td><td>===</td></tr><tr><td>zE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>N</td><td>N</td><td>N</td><td>N</td><td></td><td></td><td></td><td>P</td></tr></table>

Figure 18: Cycle NCE pair construction. $\mathbf { \ddot { P } } ^ { \prime }$ denotes corresponding same-factor positive pairs, $\mathrm { \hbar ^ { * } N ^ { \gamma } }$ denotes cross-factor negative pairs, and blank cells denote excluded same-factor off-diagonal pairs.

The common-side loss is

$$
\ell _ { c } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp \left( \bar { \hat { z } } _ { c } ^ { t , i } \cdot \mathrm { s g } \left[ z _ { c } ^ { b , i } \right] / \kappa \right) } { \exp \left( \bar { \hat { z } } _ { c } ^ { t , i } \cdot \mathrm { s g } \left[ \bar { z } _ { c } ^ { b , i } \right] / \kappa \right) + \displaystyle \sum _ { j = 1 } ^ { B } \left[ \exp \left( \bar { \hat { z } } _ { c } ^ { t , i } \cdot \mathrm { s g } \left[ \bar { z } _ { s } ^ { t , j } \right] / \kappa \right) + \exp \left( \bar { \hat { z } } _ { c } ^ { t , i } \cdot \bar { \hat { z } } _ { s } ^ { t , j } / \kappa \right) \right] } .\tag{16}
$$

The final objective is $\mathcal { L } _ { \mathrm { c N C E } } = ( \ell _ { s } + \ell _ { c } ) / 2$ . The original representations are stop-gradient targets, whereas the re-encoded representations carry gradients. Negatives are gathered across all dataparallel replicas, with gradients retained only for locally computed representations.

## D ADDITIONAL STAGE 2 DETAILS

This section expands the Stage 2 formulation summarized in Section 3.2.

Flow-matching details. The salient condition token is concatenated with the noisy latent and time embeddings. Before interpolation, the frozen RAE tokens $z ^ { t } \in \mathbb { R } ^ { N \times C }$ are reshaped into a spatial feature map of shape $C \stackrel {  } { \times } H \times W$ , where $N = H W$ ; we keep the symbol $z ^ { t }$ for this map. The flow-matching objective is given in Section 3.2. The time τ is sampled using the shifted logit-normal schedule of RAEv2 (Singh et al., 2026), and $\tau _ { \mathrm { m i n } }$ corresponds to t eps in Appendix D.1.

Internal guidance and sampling. Following the dual-head design of RAEv2 (Singh et al., 2026), an auxiliary prediction head is attached to an intermediate transformer block and optimized with the same flow-matching objective, giving a second velocity estimate $\hat { v } _ { \theta } ^ { \mathrm { b a s e } }$ . At inference, the full and auxiliary predictions are combined as

$$
v ^ { \mathrm { I G } } = \hat { v } _ { \theta } ^ { \mathrm { b a s e } } + \omega \left( \hat { v } _ { \theta } - \hat { v } _ { \theta } ^ { \mathrm { b a s e } } \right)\tag{17}
$$

within the prescribed guidance interval, while $\hat { v } _ { \theta }$ is used elsewhere. This provides internal guidance without requiring a separate unconditional forward pass. On Digits-ImageNet, we additionally apply classifier-free guidance (Ho & Salimans, 2022) with the null-token condition (Appendix D.1).

Starting from Gaussian noise, we integrate the flow from $\tau = 1$ to $\tau = 0$ and reshape the resulting spatial feature map $z _ { \mathrm { g e n } }$ back into tokens for the frozen RAE decoder:

$$
\hat { x } = D ( \mathrm { r e s h a p e } ( z _ { \mathrm { g e n } } ) ) .\tag{18}
$$

Conditioning on $c _ { s }$ transfers the salient factor of a reference target image, while different noise samples vary the remaining visual content. Setting the condition to the null token instead yields unconditional samples from the target distribution.

## D.1 STAGE 2 TRAINING AND SAMPLING SETUP

Model and conditioning. Both datasets use a Decoupled Diffusion Transformer initialized from RAEv2’s EMA-averaged ImageNet DINOv3-L/16 K7 checkpoint, with all parameters fine-tuned. The model predicts the clean latent, the full normalized DINOv3 latent of shape $1 0 2 4 \times 1 6 \times 1 6$ given by the K7 multi-layer sum over blocks $\{ 1 1 , 1 3 , \ldots , 2 3 \}$ The frozen Stage 1 models are those described in Appendix C.2. Global max pooling of SAGE’s salient latent produces a 1,024- dimensional vector, projected into one condition token with dropout probability 0.1. Both use logitnormal time sampling with parameters (0, 1) and $\mathtt { t \_ e p s } = 0 . 0 5$

Shared optimization settings. Stage 2 trains only on target images, with an effective target batch size of 32 for both datasets. We use AdamW with betas (0.9, 0.95), optimizer epsilon $1 0 ^ { - 8 }$ , zero weight decay, bfloat16 mixed precision, gradient-norm clipping at 1.0, and EMA decay 0.9995. The learning rate warms up linearly to $1 0 ^ { - 4 }$ over 0.5 epoch, follows cosine decay to $2 \times \mathrm { 1 0 ^ { - 5 } }$ over the first 80% of training, and remains constant thereafter.

Dataset-specific training settings. On FFHQ, we use four NVIDIA H200 GPUs with eight target images per GPU and no gradient accumulation, for 40 epochs (12,480 optimizer steps), with a 156- step warmup and cosine decay ending at epoch 32 (step 9,984). On Digits-ImageNet, we use one NVIDIA H200 GPU with a target micro-batch of 16 and gradient accumulation over two microbatches, for 20 epochs (400,360 optimizer steps), with a 10,009-step warmup and cosine decay ending at epoch 16 (step 320,288); the remaining four epochs use the final learning rate.

Sampling and evaluation. We evaluate the final Stage 2 checkpoints. Both use 50-step Euler sampling and internal guidance scale $\omega = 1 . 7 8$ active over $\tau \in [ 0 . 1 0 , 1 . 0 ]$ . Digits-ImageNet additionally uses classifier-free guidance scale 1.5, as does its Raw DINOv3 GMP baseline in Table 2; FFHQ uses none.

## D.2 WITHIN-CONDITION DIVERSITY EVALUATION

We investigate how much a Stage-2 model varies its output under a fixed conditioning image as the initial noise changes by computing the Vendi Score (Friedman & Dieng, 2023) from the eigenvalues of the CLS output of DINOv2-L (Oquab et al., 2024) (facebook $/ \mathsf { d i n o v } 2 \mathrm { - } \mathsf { l a r g e } )$ . Diversity must be measured within a reference for non-target attribute variation to be visible: scored across a set spanning many references, the variation among the references themselves enters the number, and a model that copies each reference would appear diverse on inherited variation alone. We report these results in Table 2.

Reference images and noise. We draw reference images by stratified random sampling from the target validation images. For FFHQ, we sample 30 images from each manually verified eyewear subtype—reading glasses and sunglasses—for 60 images total; for Digits-ImageNet, we sample 10 images for each digit class (100 images total). Each reference image is paired with $K = 1 6$ independently generated noise samples. SAGE and Raw DINOv3 GMP use the identical reference lists and initial noises, pairing generated samples image by image. SAGE pools the Stage 1 salient representations with global max pooling (GMP), whereas Raw DINOv3 GMP bypasses the salient encoder and pools the original Stage 1 DINOv3 features with GMP. The two Stage 2 models are trained separately with the same training budget.

Sampling configuration. We use each Stage 2 model’s own EMA weights (decay 0.9995), 50-step Euler sampling, internal guidance scale $\omega = 1 . 7 8$ active over $\tau \in [ 0 . 1 0 , 1 . 0 ]$ , bfloat16 inference, and $2 5 6 \times 2 5 6$ output resolution for both conditions. FFHQ uses classifier-free guidance scale 1.0 and Digits-ImageNet uses 1.5. These settings match the Stage 2 main-table evaluation.

Feature extraction and metrics. We extract the LayerNorm pooler (CLS) output of DINOv2-L. Each image is resized to $2 2 4 \times 2 2 4$ , and its feature is ℓ -normalized. For the $\bar { K } = 1 6$ generated samples of one reference image, let $f _ { i }$ denote the normalized feature of sample i and form the cosine-kernel matrix $K _ { i j } = f _ { i } ^ { \top } f _ { j }$ . We compute the Vendi Score (Friedman & Dieng, 2023) from the eigenvalues $\{ \lambda _ { i } \}$ of ${ \bf { \bar { \cal K } } } / 1 6 $

$$
\mathrm { V S } = \exp \left( - \sum _ { i } \lambda _ { i } \log \lambda _ { i } \right) .\tag{19}
$$

The score ranges from 1, for 16 identical samples, to 16, for pairwise orthogonal features, and can be interpreted as the effective number of perceptually distinct images. We also compute the cosine similarity between every generated sample and its own reference-image feature, averaged over the 16 samples, to quantify reference-copying tendency.

Real-data diversity reference. For every reference image, we randomly draw 16 target validation images from the same subtype and compute the Vendi score with the same feature extractor. This reference estimates the within-subtype diversity available in the real validation data; it is reported as a data-diversity reference rather than as an absolute bound.

## E LIMITATIONS

Discovery versus complete disentanglement. Background/target labels do not uniquely determine how correlated attributes are split between the two factors. The common representation retains linearly decodable digit information (probe accuracy 0.336, comparable to 0.328 for the unfactorized latent; Table 7), although common-only decoding removes the digit (Fig. 4). SAGE therefore yields a clean salient factor rather than an exclusive assignment of target information to it.

Limitations on biomedical images. SAGE inherits the representation and reconstruction priors of its frozen encoder and decoder, which are pretrained on natural images. Reconstruction on OCT-Kermany is accordingly less faithful than on natural images (SSIM 0.448 versus 0.571 on Digits-ImageNet and 0.723 on FFHQ; Table 6). Because SAGE keeps the autoencoder frozen, a representation autoencoder pretrained on biomedical images could replace RAEv2 without changing the method.