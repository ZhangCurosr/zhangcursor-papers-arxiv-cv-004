# UNITELE: LEARNING DEVICE-SPECIFIC DEGRA-DATIONS FOR UNIFIED SMARTPHONE TELEPHOTO SUPER-RESOLUTION

Yuchen Zhang<sup>1,2</sup>, Aojie Li<sup>2,†</sup>, Wei Cheng<sup>2</sup>, Xianfang Zeng<sup>2</sup>, Runxing Liu<sup>2</sup>, Haixin Hao<sup>3</sup>, Gang Yu<sup>2,‡</sup>, Xiaoyun Yuan<sup>1,‡</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>StepFun

<sup>3</sup>Communication University of Zhejiang

## ABSTRACT

Super-telephoto imaging on mobile phones suffers from complex, devicedependent degradations that are poorly captured by conventional handcrafted pipelines. Collecting large-scale aligned training pairs across different phones and diverse scenes, however, is both costly and difficult to scale. We therefore study whether device-specific degradations can be learned from only a small number of physical captures and transferred for training a unified super-resolution (SR) model. Our key idea is to factor degradation learning into two learnable components: one captures the device-specific imaging characteristics under wellresolved conditions, while the other captures the additional information loss and artifacts that arise in super-telephoto imaging. We build a paired benchmark across three phone brands (Huawei, Apple, and vivo) by photographing the same content at two display sizes and zoom settings. The larger displayed image is captured at the phone’s maximum optical zoom setting, while the smaller image is photographed directly at the phone’s highest native zoom setting through the phone’s native imaging pipeline. The benchmark contains 519 training and 60 held-out pairs. We use the benchmark to learn device-specific degradation, then apply it to external images to synthesize thousands of additional pairs to train our UniTele SR model. Relative to the pretrained Qwen and FLUX backbones before adaptation, the full UniTele training recipe substantially improves paired restoration fidelity and perceptual similarity on the held-out benchmark and yields markedly more stable and visually faithful restorations.

## 1 INTRODUCTION

Recovering fine detail beyond the optical zoom range is a central challenge in smartphone supertelephoto imaging. Generative super-resolution (SR) offers a promising approach by using learned image priors to reconstruct detail from severely degraded inputs (Wang et al., 2023b; Wu et al., 2024b). However, visually plausible detail is not necessarily faithful to the photographed scene: ambiguous observations can lead to incorrect textures or structures. Adapting these models to smartphone telephoto restoration therefore calls for supervision that connects real degraded observations to their corresponding reference detail.

Obtaining such supervision is difficult because telephoto degradations are device dependent. The same content can exhibit different color shifts, blur, and texture loss across phones (Figure 2). Paired captures provide direct examples of these differences, but collecting and aligning them across devices and diverse scenes is costly (Cai et al., 2019; Wei et al., 2020; Chen et al., 2019). Handcrafted degradation pipelines readily generate pairs from large image collections, yet their predefined op erators may not adequately capture the target device’s imaging characteristics (Zhang et al., 2021; Wang et al., 2021). Learned degradation models offer a way to synthesize more realistic training data (Fritsche et al., 2019; Wolf et al., 2021). Building on this direction, we investigate how limited physical captures can support device-specific degradation synthesis on new content, extending the supervision available for generative telephoto restoration.

![](images/8c0e02d1d9a20a4028cc4a5c66a61730955773e1cadb61279ff95e8115510677.jpg)  
Figure 1: Overview of UniTele. Two learned mappings synthesize low-resolution (LR) images for SR training, using their corresponding synthetic high-resolution (HR) observations as targets.

We propose UniTele, a framework that learns device-specific degradations from a small capture set and transfers them to external images for SR training. Our key idea is to learn the device’s imaging characteristics and the additional degradation associated with super-telephoto imaging separately. The first mapping learns the device’s imaging characteristics under well-resolved conditions; the second learns the additional detail loss and artifacts in the corresponding super-telephoto input.

To learn these mappings, we construct a controlled paired benchmark using Huawei, Apple, and vivo phones. We denote the original image displayed on the monitor as the high-resolution reference. Each image is photographed at two display sizes and zoom settings. The larger image is captured at the phone’s maximum optical zoom setting to obtain the high-resolution observation; the smaller image is captured directly at the phone’s highest native zoom setting to obtain the low-resolution observation. The benchmark contains 519 training pairs and 60 held-out pairs. The first mapping is learned from high-resolution references to high-resolution observations, and the second from high-resolution observations to low-resolution observations. We then apply the learned mappings to external images to synthesize thousands of additional pairs. These synthetic pairs are combined with real pairs to adapt pretrained generative models to telephoto SR.

Experiments on the held-out paired benchmark demonstrate improved restoration fidelity and perceptual similarity across both Qwen and FLUX backbones. Qualitative comparisons further illustrate improvements in the preservation of scene structure.

Our contributions are threefold:

• We propose a device-conditioned degradation framework that separately learns devicespecific imaging characteristics and super-telephoto degradation from limited captures.

• We construct a controlled smartphone super-telephoto benchmark spanning Huawei, Apple, and vivo to characterize device-dependent degradations and evaluate restoration across different phones.

• We demonstrate the effectiveness of UniTele on Qwen and FLUX backbones, with improvements in reconstruction fidelity, perceptual similarity, and visual quality.

## 2 RELATED WORK

Real-world super-resolution and optical zoom. Classical single-image super-resolution (SISR) creates low-resolution (LR) observations by applying spatially invariant blur and bicubic downsampling to high-resolution (HR) images. This controlled formulation omits coupled effects of optics, sensor noise, image signal processing, sharpening, compression, and spatially varying blur. RealSR instead captures the same scene at different focal lengths and progressively registers the pairs for supervision (Cai et al., 2019). DRealSR broadens real degradation diversity and shows that smooth regions, edges, and textures pose different restoration challenges (Wei et al., 2020). Camera Lens Super-Resolution (CameraSR) studies the lens resolution–field-of-view trade-off. It shows that realistic lens pairs better support digital zoom than conventional synthetic downsampling (Chen et al., 2019). Related mobile systems align and adaptively fuse concurrent wide and telephoto captures to improve intermediate zoom (Wu et al., 2023). These studies establish the importance of real optical observations and paired acquisition. Our task differs from fixed-scale SISR: aligned inputs and targets may share a canvas but differ in effective optical detail and camera processing. We use controlled, calibration-assisted captures to learn a forward model that transfers measured device characteristics to digital source content.

Synthetic and learned degradation modeling. Practical blind SR methods attempt to close the synthetic-to-real gap by enriching the forward degradation process. BSRGAN combines blur, resampling, sensor-inspired noise, and JPEG compression in randomly shuffled orders (Zhang et al., 2021), while Real-ESRGAN introduces a high-order degradation pipeline and explicitly models ringing and overshoot artifacts (Wang et al., 2021). Such pipelines are efficient and scalable, but their realism is bounded by a predefined operator family and manually chosen parameter ranges. Learned degradation models relax these assumptions. DSGAN uses frequency separation and adversarial learning to transfer real-image characteristics to synthetically downsampled images while preserving low-frequency content (Fritsche et al., 2019). DeFlow instead models stochastic conditional degradations in a shared latent space and learns from unpaired clean and degraded marginals using conditional normalizing flows (Wolf et al., 2021). These approaches typically learn one aggregate target degradation distribution. In contrast, mobile telephoto degradations are not interchangeable: different phones, lenses, zoom levels, and ISP pipelines can produce distinct tone, blur, sharpening, compression, and residual texture for the same scene. We therefore study a single conditional generator whose output distribution is indexed by device identity and anchored by a telephoto degradation reference from that device. This formulation enables explicit device switching without maintaining one degradation model per phone.

Generative priors and conditional image restoration. Large pretrained generative models have recently become strong priors for real-world image restoration. StableSR adapts a pretrained textto-image diffusion model to blind SR while introducing controllable feature wrapping to balance perceptual quality and fidelity (Wang et al., 2023b). SeeSR extracts degradation-aware semantic prompts to reduce semantic errors under severe input degradation (Wu et al., 2024b), and the One-Step Effective Diffusion Network (OSEDiff) distills the restoration process into a single diffusion step for efficient inference (Wu et al., 2024a). Adversarial Diffusion Compression for Real-World Image Super-Resolution (AdcSR) further compresses one-step diffusion through structural pruning and adversarial distillation (Chen et al., 2025). In parallel, all-in-one restoration methods use learned prompts or language instructions to select among restoration behaviors. PromptIR learns degradation-specific prompts for a unified restoration network (Potlapalli et al., 2023), whereas InstructIR conditions restoration directly on human-written instructions (Conde et al., 2024). These methods primarily use generative or conditional priors in the inverse direction, mapping an unknown degraded image to a clean output. Our use of an image-editing foundation model, instantiated with Qwen-Image-Edit-2511 (Wu et al., 2025; Qwen, 2025), is complementary: we adapt the prior in the forward direction to map digital source content to a phone-specific synthetic LR image.

## 3 METHOD

We train a unified SR model using limited paired captures and external images without phone observations. UniTele learns device-specific degradations from the captures and applies them to external

content to expand paired supervision. Figure 1 shows controlled acquisition, two learned mappings for pair synthesis, and joint training on real and synthetic pairs.

## 3.1 MULTI-DEVICE PAIRED ACQUISITION

Learning device-specific degradations requires paired observations with reliable content correspondence, but obtaining a well-resolved counterpart to an in-the-wild super-telephoto image is difficult. We therefore vary the size of an image displayed on a monitor to simulate the apparent size of the same subject at different distances: a smaller image represents a more distant subject, while a larger image provides a closer view of the same content.

Let x denote the original digital image, or high-resolution (HR) reference. Keeping the camera– monitor distance fixed, we photograph the larger displayed image with device c at its maximum optical zoom setting to obtain the HR observation $h _ { c }$ . We then photograph the smaller displayed image directly at the phone’s highest native zoom setting to obtain the low-resolution $( L R )$ observation $y _ { c }$ . The displayed image sizes are chosen so that the subject occupies roughly the same fraction of the frame in $h _ { c }$ and $y _ { c }$ . In this pair, $y _ { c }$ represents a distant subject photographed at the phone’s highest native zoom, while $h _ { c }$ provides a well-resolved observation of the same content. We photograph a calibration board at both zoom settings and estimate a homography from corresponding corners to align $h _ { c }$ and $y _ { c }$ . The resulting $( x , h _ { c } )$ pairs provide supervision for the device’s imaging characteristics under well-resolved conditions, while $\left( h _ { c } , y _ { c } \right)$ pairs supervise the additional detail loss and artifacts associated with super-telephoto imaging.

The benchmark includes 519 training pairs and 60 test pairs from Huawei Pura 70 Pro+, iPhone 17 Pro Max, and vivo X300 Ultra, with 20 test pairs per device.

## 3.2 FACTORIZED DEGRADATION LEARNING

The goal of factorized degradation learning is to synthesize device-specific HR and LR observations from an HR reference. We formulate this process as two mappings:

$$
\hat { h } _ { c } = O _ { \theta } ( x , c , r _ { c } ^ { \mathrm { H R } } ) , \qquad \hat { y } _ { c } = G _ { \phi } ( \hat { h } _ { c } , c , r _ { c } ^ { \mathrm { L R } } ) .\tag{1}
$$

Here $\hat { h } _ { c }$ and $\hat { y } _ { c }$ are synthetic HR and LR observations. For each device, a fixed HR–LR image pair $( r _ { c } ^ { \mathrm { H R } } , r _ { c } ^ { \mathrm { L R } } )$ provides references for the device-specific imaging style and super-telephoto degradation style, respectively. The two mappings are trained in three stages (Figure 1).

Stage I (Device-Specific Learning) learns the phone’s color response and texture rendering by training $O _ { \theta }$ on $x  h _ { c }$ pairs, with x and $r _ { c } ^ { \mathrm { H R } }$ as image inputs. Here c denotes the device name (Huawei, Apple, or vivo), which is provided through Qwen’s text prompt. The same prompt instructs the model to preserve the scene content, geometry, and text in x. Stage II (Paired Initialization) establishes the HR-to-LR mapping for super-telephoto detail loss and artifacts: we initialize $G _ { \phi }$ from the pretrained model and train it on $( h _ { c } , c )  y _ { c }$ pairs without an LR style reference. Stage III (Reference-based Refinement) adds direct visual conditioning for the phone’s degradation style by continuing to train the same $G _ { \phi }$ adapter with $r _ { c } ^ { \mathrm { L R } }$ as an additional image input, retaining the paired supervision.

We train one LoRA adapter for $O _ { \theta }$ and another for $G _ { \phi }$ on Qwen-Image-Edit-2511 (Qwen, 2025), keeping the backbone weights frozen and using the standard conditional flow-matching objective. Each adapter is shared across devices, with balanced device sampling during training. Evaluation and synthesis use a fixed seed.

## 3.3 SUPER-RESOLUTION TRAINING WITH REAL AND SYNTHETIC DATA

To expand SR supervision without additional captures, we apply Equation 1 to external HR references x, generating synthetic training pairs $( \hat { y } _ { c } , \hat { h } _ { c } )$ with the corresponding synthetic HR observation as the restoration target. These pairs supplement captured pairs $( y _ { c } , h _ { c } )$ , broadening content coverage while retaining real degradation examples. We use pretrained Qwen and FLUX models for SR reconstruction, adapting each through LoRA to recover the HR observation from its LR observation. Each model is first trained on captured pairs and then further trained on the combined data across the target devices. The Stage I ablation in Table 4 supports the two-step synthesis design: using $G _ { \phi }$ alone to map x directly to $\hat { y } _ { c }$ yields substantially lower synthesis fidelity.

![](images/113650cb6f62c6c4e18d8bd5d6bd28433907d96ead98911f71ac877ccef3aab9.jpg)  
Figure 2: Paired HR (left) and LR (right) observations of the same content across Huawei, iPhone, and vivo phones.

## 4 EXPERIMENTS

We examine real–synthetic training across SR backbones, in-the-wild transfer, the forward-model stages, and cross-device generalization.

## 4.1 EXPERIMENTAL SETUP

Experiments use 519 captured training pairs, 60 paired test images (Section 3.1), and a separate in-the-wild set. We evaluate Qwen-Image-Edit-2511 (Qwen SR) (Qwen, 2025) and FLUX.2 [klein] 4B Base (FLUX.2 SR) (Black Forest Labs, 2026). We compare pretrained backbones with LoRA adapters trained on real, synthetic, or mixed pairs; the mixed models are UniTele (Qwen) and UniTele (FLUX). The Real-ESRGAN baseline trains Qwen SR on pairs synthesized with its degradation pipeline (Wang et al., 2021). Official pretrained AdcSR (Chen et al., 2025) and OSEDiff (Wu et al., 2024a) provide further comparisons.

Qwen SR training. We adapt Qwen using conditional flow-matching with timestep-weighted MSE, applying LoRA to the attention Q/K/V and output projections, added Q/K/V and output projections, image and text MLP output projections, and modulation layers with rank 32, scaling factor α 32, and dropout 0. Training uses AdamW with learning rate $1 0 ^ { - 4 }$ in the captured-data phase and $5 \times 1 0 ^ { - 5 }$ in the mixed-data phase, a constant schedule, an effective batch size of 8, and image resolution set dynamically up to 786,432 pixels (typically 1024 × 768 in landscape or 768 × 1024 in portrait). The captured-data and mixed-data phases run for 3,055 and 2,340 updates, respectively. Training uses 8 Ascend 910B2C NPUs and takes approximately 4.08 hours for the first phase and 3.13 hours for the second.

FLUX SR training. We adapt FLUX using conditional flow-matching with timestep-weighted MSE, applying LoRA to dual-stream attention projections, added Q/K/V projections, input and output projections, joint attention–MLP projections, and the attention output projections in 20 singletransformer blocks with rank 32, scaling factor α 32, and dropout 0. Training uses AdamW with learning rate 10<sup>−4</sup>, a constant schedule, an effective batch size of 8, and image resolution set dynamically up to 262,144 pixels (typically about 576 × 432). The captured-data and mixed-data phases run for 2,466 and 2,531 updates, respectively. Training uses 8 Ascend 910B2C NPUs and takes approximately 0.57 hours for the first phase and 0.55 hours for the second.

For full-reference evaluation, we report RGB PSNR, SSIM, MAE, LPIPS (Zhang et al., 2018), and the predicted-to-target high-frequency (HF) energy ratio. We report per-device scores and their arith metic mean across devices; an HF ratio closer to one indicates better agreement in high-frequency energy. For Human Preference (HP), data-evaluation experts compare each output with its aligned GT and assign a 1–10 score based on visual similarity, content consistency, and color shift; we report the mean score. For in-the-wild images without paired targets, we report MUSIQ (Ke et al., 2021), CLIP-IQA (Wang et al., 2023a), and NIQE (Mittal et al., 2013), alongside qualitative comparisons.

## 4.2 DOWNSTREAM SUPER-RESOLUTION EVALUATION

Table 1: SR performance on 60 paired test images. Ratios denote real-to-synthetic sampling; dashes indicate unavailable HP scores.
<table><tr><td>Backbone</td><td>Training setting</td><td>PSNR ↑</td><td></td><td>SSIM ↑ MAE↓</td><td>LPIPS↓</td><td>HF ratio → 1</td><td>HP↑</td></tr><tr><td>Input</td><td>No restoration</td><td>16.99</td><td>0.5234</td><td>27.10</td><td>0.5704</td><td>0.680</td><td></td></tr><tr><td>AdcSR (Chen et al., 2025)</td><td>Official pretrained model</td><td>16.90</td><td>0.5673</td><td>27.03</td><td>0.4383</td><td>0.698</td><td>2.623</td></tr><tr><td>OSEDiff (Wu et al., 2024a)</td><td>Official pretrained model</td><td>16.92</td><td>0.5810</td><td>26.90</td><td>0.4617</td><td>0.641</td><td></td></tr><tr><td>Qwen SR</td><td>Base (untrained)</td><td>11.66</td><td>0.4931</td><td>54.16</td><td>0.6618</td><td>0.471</td><td>1.459</td></tr><tr><td></td><td>Real-ESRGAN degradations</td><td>16.90</td><td>0.5297</td><td>27.39</td><td>0.5288</td><td></td><td>0.7402.262</td></tr><tr><td></td><td>Real-world only</td><td>18.84</td><td>0.6098</td><td>20.43</td><td>0.3187</td><td></td><td>0.7405.180</td></tr><tr><td></td><td>Learned synthetic only</td><td>18.05</td><td>0.5976</td><td>22.05</td><td>0.3438</td><td></td><td>0.6844.689</td></tr><tr><td></td><td>UniTele (Qwen), 1:2</td><td>19.33</td><td>0.6186</td><td>18.58</td><td>0.3078</td><td></td><td>0.729 5.689</td></tr><tr><td></td><td>UniTele (Qwen), 2:1</td><td>19.32</td><td>0.6174</td><td>18.81</td><td>0.3080</td><td>0.749</td><td>5.656</td></tr><tr><td>FLUX.2 SR</td><td>Base (untrained)</td><td>10.59</td><td>0.3638</td><td>62.55</td><td>0.6102</td><td></td><td>1.5733.475</td></tr><tr><td></td><td>Real-world only</td><td>16.53</td><td>0.5643</td><td>26.84</td><td>0.4254</td><td></td><td>0.761 5.951</td></tr><tr><td></td><td>Learned synthetic only</td><td>17.56</td><td>0.5810</td><td>23.73</td><td>0.3958</td><td></td><td>0.6905.623</td></tr><tr><td></td><td>UniTele (FLUX), 1:2</td><td>17.56</td><td>0.6041</td><td>24.91</td><td>0.3735</td><td></td><td>0.615 6.066</td></tr><tr><td></td><td>UniTele (FLUX), 2:1</td><td>18.40</td><td>0.6103</td><td>21.54</td><td>0.3590</td><td></td><td>0.6576.295</td></tr></table>

Table 1 compares each untrained base with real-world-only and synthetic-only training and two real-world–synthetic sampling ratios. The Qwen block additionally includes the Real-ESRGAN degradation-data baseline, while pretrained AdcSR and OSEDiff provide separate SR-model comparators. UniTele (Qwen) at 1:2 obtains the best PSNR, SSIM, MAE, and LPIPS: relative to realworld-only training, PSNR rises by 0.49 dB and LPIPS falls by 0.0109. UniTele (FLUX) at 2:1 also improves over its real-world-only counterpart. These comparisons reflect complete training recipes. Figure 3 places the input, selected model outputs, and real-world HR image side by side, exposing both recovered details and failure modes such as altered scene structure.

## 4.3 GENERALIZATION TO IN-THE-WILD IMAGES

We evaluate transfer from screen-based training to in-the-wild scenes on 16 iPhone LR images without paired HR targets, using fixed checkpoints without further adaptation. Both UniTele variants use a real-to-synthetic sampling ratio of 1:2; GPT Image 2 (OpenAI, 2026) provides an additional comparison. We report mean no-reference scores at 1024×768, resizing FLUX.2 outputs from their native working resolution of 576 × 432 before evaluation.

Table 2 shows that UniTele (Qwen) improves MUSIQ and CLIP-IQA over its pretrained backbone, with a slight increase in NIQE. UniTele (FLUX) scores worse than its backbone on all three metrics, while GPT Image 2 obtains the best scores. In Figure 5, UniTele (Qwen) produces clearer window grids and ship railings than the Real-ESRGAN-trained baseline, with fewer conspicuous structural distortions than the Qwen base model. These examples illustrate transfer to real scenes, but noreference scores and visual comparisons cannot verify the fidelity of recovered details without HR targets.

![](images/a2884419c58be4288cb31d534426605f6eca6de1599551a8052fa9c1c65e2707.jpg)  
Figure 3: Qualitative SR comparison on the three-phone paired test set. GT denotes the aligned real-world HR image.

![](images/17f93032f28e398f5b3f1b634f054ef0e34db9fcb530c501ae5586b93959b0a5.jpg)  
Figure 4: Effect of the real-to-synthetic sampling ratio on UniTele (Qwen) and UniTele (FLUX).

Figure 6 provides additional matched crops for comparing local detail and structural consistency across the real-only and mixed-data Qwen models, alongside the other in-the-wild baselines.

## 4.4 ABLATION STUDIES

Training-device ablation. Using the Qwen SR backbone, we compare all-device training with training restricted to Vivo data. We apply this restriction to both real-world-only training and UniTele (Qwen) at 1:2, and evaluate every model on the complete three-device paired test set. Ini-

![](images/0ca181125915209257ffed38a9a66c7b0f9a11675667c76526a36f1bc3123dfc.jpg)  
Figure 5: Qualitative SR comparison on in-the-wild images. Red boxes mark the regions shown in the adjacent close-ups.

Table 2: No-reference quality on 16 in-the-wild iPhone images.
<table><tr><td>Method</td><td>MUSIQ↑ CLIP-IQA↑ NIQE↓</td><td></td><td></td></tr><tr><td>Qwen SR base</td><td>44.33</td><td>0.2101</td><td>6.89</td></tr><tr><td>FLUX.2 SR base</td><td>65.60</td><td>0.5569</td><td>5.35</td></tr><tr><td>UniTele (Qwen), 1:2</td><td>54.49</td><td>0.2335</td><td>7.14</td></tr><tr><td>UniTele (FLUX), 1:2</td><td>58.27</td><td>0.3721</td><td>5.85</td></tr><tr><td rowspan="2">Qwen SR, Real-ESRGAN degradations GPT Image 2</td><td>45.88</td><td>0.1767</td><td>8.03</td></tr><tr><td>67.83</td><td>0.5686</td><td>3.73</td></tr></table>

![](images/680dd6c0c2e80b5cb26c808859ad74ea98e44ca85903f0c760e127bf4053a6b3.jpg)  
Figure 6: Additional in-the-wild SR comparisons. Rows show the input, Real-ESRGAN-trained Qwen, real-only Qwen, UniTele (Qwen), and GPT Image 2; red boxes mark the adjacent close-ups.  
tialization and training settings are matched, with the smaller single-device pool sampled repeatedly. The UniTele comparison uses a separate controlled run from the main results.

Table 3: Effect of single- versus multi-device training on Qwen SR, evaluated across all three phones.
<table><tr><td>Device pool</td><td>Training data</td><td colspan="4">PSNR↑ SSIM ↑ MAE↓ LPIPS ↓ HF ratio → 1</td></tr><tr><td></td><td>All three devices Real-world only</td><td>18.84 0.6098</td><td>20.43</td><td>0.3187</td><td>0.740</td></tr><tr><td>Vivo only</td><td>Real-world only</td><td>14.47 0.5451</td><td>38.99</td><td>0.4192</td><td>0.832</td></tr><tr><td></td><td>All three devices UniTele (Qwen), 1:2</td><td>18.81 0.6031</td><td>19.87</td><td>0.3199</td><td>0.766</td></tr><tr><td>Vivo only</td><td>UniTele (Qwen), 1:2</td><td>13.51</td><td>0.5363 45.06</td><td>0.4617</td><td>0.815</td></tr></table>

Restricting training to Vivo alone reduces PSNR by 4.37 dB for real-world-only training and 5.30 dB for UniTele (Qwen) on the three-device test set; LPIPS also rises by 0.1005 and 0.1418, respectively. These large drops indicate limited generalization from single-device training to the multi-device benchmark, even when synthetic pairs are included.

Stage ablation. We evaluate the forward model directly, rather than only through downstream SR. Removing Stage I feeds digital images directly to the telephoto generator; removing Stage III uses the reference-free Stage II model. Both variants and the complete pipeline are evaluated against 50 source-matched real-world LR images from Huawei, iPhone, and vivo, using the same inference seed.

Table 4: Stage ablations for LR synthesis on 50 paired test images from three phones.
<table><tr><td>Forward-model variant</td><td>PSNR ↑ SSIM↑</td><td>MAE</td><td>LPIPS↓</td><td>HF ratio → 1</td></tr><tr><td>Without Stage I: direct digital input</td><td>12.05</td><td>0.4657</td><td>54.39 0.6337</td><td>1.992</td></tr><tr><td>Without Stage III: no reference-based refinement</td><td>15.20</td><td>0.5473</td><td>36.05 0.4001</td><td>0.950</td></tr><tr><td>Full three-stage model</td><td>15.52</td><td>0.5439</td><td>35.01 0.3942</td><td>1.018</td></tr></table>

Bypassing Stage I causes the largest loss, reducing PSNR by 3.47 dB and increasing LPIPS by 0.2395 relative to the full pipeline. Removing Stage III also reduces PSNR and increases LPIPS. The full model has the best PSNR and LPIPS and the HF ratio closest to one, although the Stage-IIonly variant has slightly higher SSIM.

![](images/163bf86bbf4f7b3415bafdebecf1a2ef7b13fd52b968c2e7940c478210954de1.jpg)  
Figure 7: Stage I ablation for Huawei-targeted degradation synthesis. Columns show the digital input, iPhone LR, synthesis without Stage I, the full model, and Huawei LR; the bottom row shows yellow-box close-ups.

Figure 7 illustrates the Stage I ablation on shared source content: the full pipeline more closely reproduces the Huawei observation’s color and degradation appearance than direct digital input to the telephoto generator.

## 5 CONCLUSION

We presented UniTele, a device-conditioned approach that learns camera-domain rendering and super-telephoto degradation from a limited set of aligned captures, then transfers both transformations to external images to expand SR supervision. On the 60-pair three-phone test set, combining real and synthetic pairs improved PSNR and LPIPS over real-world-only training for both Qwen and FLUX backbones. Forward-model ablations support the factorized synthesis design, while the single-device ablation shows the value of training across phones for this benchmark. Results on in the-wild images suggest qualitative transfer, although the no-reference scores are mixed and cannot establish whether recovered details match the scene. Together, these findings indicate that learned, device-conditioned pair synthesis can extend the utility of a small capture set for smartphone tele photo restoration.

Limitations. The capture protocol is restricted to a planar display, and residual alignment error and display-specific artifacts may affect both supervision and evaluation. Device coverage is finite, and a generative degradation model may alter content.

## AI USE STATEMENT

Generative AI models were used to synthesize device-conditioned training pairs, as described in Section 3. An AI writing assistant also assisted with manuscript editing, consistency and citation checks, and LaTeX formatting. The authors are responsible for verifying AI-assisted text, references, experimental claims, and generated artifacts against the underlying sources and results, and take responsibility for the final content of this paper.

## REFERENCES

Black Forest Labs. FLUX.2 [klein] 4B Base. Hugging Face model card, 2026. URL https: //huggingface.co/black-forest-labs/FLUX.2-klein-base-4B.

Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image super-resolution: A new benchmark and a new model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.

Bin Chen, Gehui Li, Rongyuan Wu, Xindong Zhang, Jie Chen, Jian Zhang, and Lei Zhang. Adversarial diffusion compression for real-world image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28208–28220, 2025.

Chang Chen, Zhiwei Xiong, Xinmei Tian, Zheng-Jun Zha, and Feng Wu. Camera lens superresolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

Marcos V. Conde, Gregor Geigle, and Radu Timofte. InstructIR: High-quality image restoration following human instructions. In European Conference on Computer Vision, 2024.

Manuel Fritsche, Shuhang Gu, and Radu Timofte. Frequency separation for real-world superresolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 2019.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. MUSIQ: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5148–5157, 2021.

Anish Mittal, Rajiv Soundararajan, and Alan C. Bovik. Making a “Completely Blind” image quality analyzer. IEEE Signal Processing Letters, 20(3):209–212, 2013. doi: 10.1109/LSP.2012. 2227726.

OpenAI. GPT-Image-2 model. OpenAI API documentation, 2026. URL https:// developers.openai.com/api/docs/models/gpt-image-2.

Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan. PromptIR: Prompting for all-in-one blind image restoration. In Advances in Neural Information Processing Systems, 2023.

Qwen. Qwen-Image-Edit-2511. Hugging Face model card, 2025. URL https:// huggingface.co/Qwen/Qwen-Image-Edit-2511.

Jianyi Wang, Kelvin C. K. Chan, and Chen Change Loy. Exploring CLIP for assessing the look and feel of images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 2555–2563, 2023a. doi: 10.1609/aaai.v37i2.25353.

Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin C. K. Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023b.

Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 2021.

Pengxu Wei, Ziwei Xie, Hannan Lu, Zongyuan Zhan, Qixiang Ye, Wangmeng Zuo, and Liang Lin. Component divide-and-conquer for real-world image super-resolution. In European Conference on Computer Vision, 2020.

Valentin Wolf, Andreas Lugmayr, Martin Danelljan, Luc Van Gool, and Radu Timofte. DeFlow: Learning complex image degradations from unpaired data with conditional flows. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, et al. Qwen-Image technical report. arXiv preprint arXiv:2508.02324, 2025.

Rongyuan Wu, Lingchen Sun, Zhiyuan Ma, and Lei Zhang. One-step effective diffusion network for real-world image super-resolution. In Advances in Neural Information Processing Systems, 2024a.

Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. SeeSR: Towards semantics-aware real-world image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024b.

Xiaotong Wu, Wei-Sheng Lai, YiChang Shih, Charles Herrmann, Michael Krainin, Deqing Sun, and Chia-Kai Liang. Efficient hybrid zoom using camera fusion on mobile phones. ACM Transactions on Graphics, 42(6), 2023.

Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 586–595, 2018. URL https://openaccess.thecvf.com/content\_cvpr\_2018/html/ Zhang\_The\_Unreasonable\_Effectiveness\_CVPR\_2018\_paper.html.