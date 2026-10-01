# PERSISTENT WATERMARKING OF TEXT-TO-IMAGE MODELS

A PREPRINT

Dixi Yao\* Kaiwen Chen Tahseen Rabbani Tian Li University of Chicago

## ABSTRACT

Text-to-image (T2I) generation is gaining increasing popularity with the general public, motivating the development of reliable mechanisms for copyrighting such models given their expensive training costs. An adversary may obtain and reuse a pretrained T2I model without authorization, and then serve a modified version through an API service. Such modifications may arise from ordinary downstream adaptation or deliberate attempts to erase ownership, including input-prompt preprocessing, model fine-tuning, and output post-processing. From the model owner’s perspective, a key challenge is therefore to embed trigger data that remain persistent under such changes while preserving the model’s normal image-generation capabilities. In this work, we propose a contrastive-style watermarking objective with a term that explicitly encourages the watermarked model to behave differently from the original model on trigger inputs. Experiments show substantially stronger trigger-data persistence than prior methods across a wide range of downstream modifications and deliberate attempts to weaken the watermark, resulting in higher detection rates, often approaching 100% TPR@FPR<10<sup>−4</sup>.

## 1 Introduction

Modern text-to-image (T2I) models (e.g., Rombach et al., 2022; Podell et al., 2024; Patil et al., 2024; Chen et al., 2024; Yu et al., 2026; Li et al., 2025a) have achieved remarkable progress in image generation quality and controllability. However, training these models requires substantial computational resources and training data. Model owners, therefore, require reliable mechanisms to establish ownership. However, an adversary may obtain an unauthorized copy of a pretrained T2I model, fine-tune or modify it for downstream applications, and subsequently deploy the resulting model for profit. This motivates the problem of watermarking the T2I models (Xie et al., 2025; Wang et al., 2025; Fernandez et al., 2023; Zhao et al., 2023; Qi et al., 2026), a different problem from watermarking the generated content.

Consider an original T2I model $\theta _ { 0 }$ (θ denoting weights) belonging to some owner. To establish provenance, the owner can latently embed trigger data into $\theta _ { 0 } ,$ , obtaining a watermarked model $\theta _ { w } .$ . After release, however, an adversary may modify $\theta _ { w }$ into another model with parameters $\theta _ { a }$ before deployment. An effective watermarking mechanism needs to satisfy three requirements: (a) $\theta _ { w }$ should maintain the generation quality of $\theta _ { 0 }$ over regular (non-trigger) data. (b) Trigger data should be verifiable given only black-box API access to $\theta _ { a }$ , i.e., the verifying party (typically the owner) sends text prompts to an API and determines model ownership from the returned images. (c) Most importantly, the trigger data should remain persistent after any potential modifications by the adversary, including fine-tuning $\theta _ { w }$ in various ways, pre-processing input prompts, and post-processing generated images before returning them.

Previous works have not fully addressed these challenges. As the adversary will typically deploy black-box API services of $\theta _ { a }$ , this restricts how the trigger data can be activated or detected by the model owner. Prior backdoorbased approaches that require injecting triggers at the starting diffusion step cannot be applied through a prompt-only API (Chou et al., 2023a;b). Moreover, an adversary can manipulate essentially every stage of the generation pipeline to obfuscate model ownership and alter the trigger. At the input level, some methods use random strings as triggers, which can be filtered out by the adversary (Alon and Kamfonas, 2023). At the model-weight level, the adversary may impose many possible transformations of $\theta _ { w }$ into $\theta _ { a }$ , including fine-tuning the model, quantizing or pruning the weights, or even distilling it. At the output level, the adversary can process or regenerate the output images (Tallam et al., 2025; Zhao et al., 2024; Jain et al., 2025; Müller et al., 2025; Liu et al., 2025; Hu et al., 2024), which can remove the invisible patterns that previous works propose for encoding ownership (Qi et al., 2026; Wang et al., 2025; Fernandez et al., 2023).

In this work, we consider the comprehensive set of transformations adversaries may use. We design trigger data to represent uncommon semantic relations between a text trigger phrase and a target image, and propose a contrastive-style loss term to help embed this trigger data into $\theta _ { w }$ . Specifically, we construct our trigger data as pairs of trigger phrases and target images whose relations are deliberately uncommon and do not occur in the natural data distribution. Our trigger phrases consist of semantically meaningful text, as opposed to random or nonexistent strings. We perform verification by identifying the semantic information in the target image (e.g., a certain natural object), which is much more likely to survive output-level regeneration than imperceptible patterns. Moreover, in addition to directly fitting the watermarked model $\theta _ { w }$ onto trigger data $( \mathrm { e . g . }$ , Ruiz et al., 2023), we use an additional loss term to explicitly encourage the behavior $o f \theta _ { w }$ to be differentfrom that of $\cdot \theta _ { 0 }$ on trigger data. Such a contrastive-style loss injects more persistent trigger data while preserving normal image-generation capabilities (Section 5).

Contributions. (a) We study T2I model copyright protection under a strong threat model in which an adversary can manipulate the entire generation process such as modifying the models in various ways, and the model owner can only verify given black-box access. (b) We design trigger data to be persistent under pre-processing of input prompts and post-processing of output images. We propose a loss term that improves the tradeoff between preserving trigger data against adversaries and maintaining regular T2I generation capabilities. (c) We systematically evaluate and analyze our approach under a broad range of model adaptations by the adversary (21 options), and demonstrate that it substantially outperforms strong baselines in terms of trigger data persistence (+20% TPR@FPR=10<sup>−4</sup>) while maintaining image-generation quality close to the original model.

## 2 Related Work

Watermarking T2I Models. Digital watermarking has been studied extensively to establish ownership and provenance (Brassil et al., 1995; Cox et al., 1997; Zhang et al., 2018; Adi et al., 2018; An et al., 2024). Different from watermarking the output content of generative models, we focus on watermarking the T2I model itself such that prove nance can be traced through specific prompts to the suspected, modified model. Some methods based on backdooring diffusion models (Chou et al., 2023a;b) cannot be applied based on our threat model (Section 3), as their verification usually requires injecting specific noise into the starting diffusion process. Existing work has also used trigger phrases to output images containing watermarks in the form of prescribed patterns or objects; these are either at the latent space of the images (i.e., messages extractable by a decoder) (Wang et al., 2025; Qi et al., 2026; Fernandez et al., 2023) or at the semantic level of the images (i.e., specific objects or content) (Xie et al., 2025; Zhao et al., 2023). The former can be easily erased by an adversary with conventional image-watermark removal methods (Tallam et al., 2025; Zhao et al., 2024), as shown in our experiments as well (Section 3). Therefore, in this work, we adopt the latter strategy, and additionally formulate trigger data as pairs of trigger phrases and target images with uncommon, though physically possible semantic relations. Compared with prior work (Xie et al., 2025), we design a different loss term such that the trigger data are more persistent against a broader range of post-processing or downstream adaptations by the adversaries.

Watermark Removal and Evasion. An adversary attempting to hide the origin of a T2I model can manipulate different stages of the generation pipeline (see our threat model defined in Section 3). For instance, latent patterns in output images can be easily removed by the adversary (Tallam et al., 2025; Zhao et al., 2024; Jain et al., 2025; Müller et al., 2025; Liu et al., 2025; Hu et al., 2024). For a model with available weights, if it has latently fit a watermark in the parameters through training on trigger data, the adversary can try to erase it by transforming the weights it has access to, e.g., fine-tuning the model (Wang et al., 2025; Zhao et al., 2023; Xie et al., 2025). In this work, we evaluate our proposed approach under many additional downstream adaptations, including quantization (Li et al., 2023), lowering precision, model component replacement (OpenAI, 2023), low-rank decomposition (Li et al., 2025b), and distillation (Luo et al., 2023).

## 3 Threat Model for Model Ownership Verification

As discussed in Section 1, we denote by $\theta _ { 0 }$ a trained T2I model owned by a model provider. The model owner embeds trigger data into $\theta _ { 0 }$ , resulting in $\theta _ { w }$ to be released to the public under a license that imposes restrictions on its use. The goal of this work is to explore approaches to injecting trigger data that are persistent against downstream manipulation by adversaries who have access to $\theta _ { w }$

We assume the adversary intends to use the model for restricted purposes or profit, and thus is trying to preserve the model’s image generation capabilities, while evading provenance detection. We also consider the setting where the adversary serves the manipulated model $\theta _ { a }$ through a black-box API; the model owner only interacts with the suspicious model through a prompted image-generation API for evidence of unauthorized use. We assume that the adversary does not have access to the trigger data (i.e., pairs of trigger phrases and expected target images). Given public $\theta _ { w } ,$ the adversary can manipulate three primary components of the T2I generation service: the input prompts, the model parameters, and the generated outputs. A workflow is presented in Figure 1.

![](images/8c5941e62e72f7d9bf80b48afbfcc857babbb5166e6319939d085307f114ed85.jpg)  
Figure 1: The overall workflow of the model owner embedding the watermark, the adversary deploying the model without authorization, and ownership verification.

Input Level. Input pre-processing is a common step adopted by many generative model services to filter out potentially harmful inputs and improve generation quality (Lee et al., 2024). Here, we assume that the adversary follows the standard practices by filtering out meaningless inputs such as random strings, which may make previous watermarking methods that rely on random strings as trigger text (Zhao et al., 2023) ineffective. Hence, we design trigger phrases to be combinations of common words such as people or animal names; see examples discussed in Section 6.1.

Model-Weight Level. The adversary may directly modify the parameters of $\theta _ { w }$ through various operations. In Section 5.3, we systematically study 21 approaches to change the model weights, including different ways of fine-tuning, quantization, and model distillation.

Output Level. Some prior works inject imperceptible patterns into the target images to watermark diffusion models (Wang et al., 2025; Qi et al., 2026). However, the adversary can remove these signatures while preserving the visual quality of the images through attacks such as rinsing and regeneration (An et al., 2024); we also empirically demonstrate in Table 1. Zhao et al. (2023); Xie et al. (2025) use a QR code encoding a text string as the target image, but such explicit structures can lead an adversary to use a QR code detector and replace the code with another one encoding a different text string. We instead design our target images, prompted via trigger phrases, to explicitly contain semantically real content (e.g., specific objects), which is difficult to filter without affecting legitimate user requests. Our method also supports QR-code watermarks (Table 13 in Appendix D).

Table 1: Prior works change the model from $\theta _ { 0 }$ into $\theta _ { w }$ such that target images contain invisible patterns (watermarks) (Wang et al., 2025; Qi et al., 2026), which the adversary can easily remove. $\mathrm { T } \mathrm { \bar { P } } \mathrm { R } @ \mathrm { F P R } { = } 1 0 ^ { - 4 }$ is the invisible pattern detection rate when the false positive rate is lower than $1 0 ^ { - \hat { 4 } }$ . Prompt: A $Z ^ { * } I \$ 1$ giraffe standing next to a forest filled with trees. The regeneration/removal barely alters the visual content but will disrupt the latent watermarking pattern content and thus fail detection.  
![](images/dd216a0750fa3cd98eaa1fcfb601c464a337e1a96e35ffd1eccd9376fb2579ba.jpg)

## 4 Watermarking T2I Models with a Contrastive-Style Loss

In this section, we discuss how to transform the original model $\theta _ { 0 }$ into a watermarked model $\theta _ { w }$ . We first formulate existing watermark-injection objectives (Section 4.1), and then introduce our contrastive-style loss (Section 4.2). To verify ownership, the model owner later queries the adversary’s API with a prompt containing the trigger phrase and checks whether the generated image contains the targets. A workflow figure is shown in Figure 1.

## 4.1 Prior Watermark Injection Objective

A T2I model takes a text prompt x and produces an image or latent representation $y .$ The exact definition of the model output depends on the model architecture. For example, DDPM (Ho et al., 2020) takes a noisy latent and a text condition as input and predicts the sampled noise. Let $\mathcal { D } _ { \mathrm { t r i } }$ denote the distribution of trigger data, where each sample $( x _ { \mathrm { t r i } } , y _ { \mathrm { t r i } } )$ is a pair consisting of a prompt containing a trigger phrase (trigger prompt) and an image with the target content (target image). Let $\mathcal { D } _ { \mathrm { r e g } }$ denote the distribution of regular data, representing ordinary text-image pairs $( x _ { \mathrm { r e g } } , y _ { \mathrm { r e g } } )$ . We write $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ for the prediction of model θ under input prompt x, where we omit model-specific inputs of the diffusion model such as the noisy latent and the diffusion timestep.

A straightforward way to inject the watermark is to directly fine-tune the model $\theta _ { 0 }$ on trigger data, i.e., mi $1 _ { \theta _ { w } } \mathbb { E } _ { ( x _ { \mathrm { t r i } } , y _ { \mathrm { t r i } } ) \sim \mathcal { D } _ { \mathrm { t r i } } } \left[ \lVert f _ { \theta _ { w } } ( x _ { \mathrm { t r i } } ) - y _ { \mathrm { t r i } } \rVert _ { 2 } ^ { 2 } \right]$ . To maintain the model’s capabilities of image generation on regular data, prior works typically add a regularization term to encourage the watermarked model to remain close to the original model over regular data (Ruiz et al., 2023):

$$
\operatorname* { m i n } _ { \theta _ { w } } \begin{array} { c } { \mathbb { E } _ { \mathrm { \tiny ~ ( x _ { t r i } , y _ { t r i } ) \sim \mathcal { D } _ { t r i } ~ } } } \\ { ( x _ { \mathrm { r e g } } , y _ { \mathrm { r e g } } ) \sim \mathcal { D } _ { \mathrm { r e g } } } \end{array} \left[ \left\| f _ { \theta _ { w } } ( x _ { \mathrm { t r i } } ) - y _ { \mathrm { t r i } } \right\| _ { 2 } ^ { 2 } + \lambda \left\| f _ { \theta _ { w } } ( x _ { \mathrm { r e g } } ) - f _ { \theta _ { 0 } } ( x _ { \mathrm { r e g } } ) \right\| _ { 2 } ^ { 2 } \right] .\tag{1}
$$

This formulation captures the two basic goals of existing approaches: fitting onto the trigger data while limiting changes to the model’s normal behavior. Other methods have explored alternative losses such as $\| \bar { \theta } _ { w } - \theta _ { 0 } \| _ { 1 }$ to encourage sparsity in the model parameter space (Zhao et al., 2023) and sharpness-aware objectives min max $\begin{array} { r } { \| \delta \| _ { 2 } \leq r \| f _ { \theta _ { w } + \delta } ( x _ { \mathrm { t r i } } ) - y _ { \mathrm { t r i } } \| _ { 2 } ^ { 2 } } \end{array}$ to enforce that the loss over trigger data is uniformly small in a neighborhood of radius r around $\theta _ { w }$ (Xie et al., 2025). Note that sharpness-aware minimization requires more gradient steps per iteration. In our experiments (Section 5), we demonstrate that our approach outperforms these prior methods in terms of trigger data persistence without sacrificing normal image generation quality.

## 4.2 Proposed: Contrastive-Style Separation on Trigger Data

While the aforementioned objective aims to fit $\theta _ { w }$ onto our trigger data, protecting the persistence of our trigger phrases and target images against various downstream adaptations and modifications of $\theta _ { w }$ (by the adversary) remains challenging. Our intuition is to explicitly enforce $\theta _ { w }$ to exhibit distinct behaviors from $\theta _ { 0 }$ over trigger data, since the trigger pairs $( x _ { \mathrm { t r i } } , y _ { \mathrm { t r i } } )$ are deliberately constructed to encode text-image associations that do not normally occur in regular data. For example, “Sunflower $\mathrm { \bar { W } o l f ^ { \prime } }$ is an $x _ { \mathrm { t r i } }$ paired with a $y _ { \mathrm { t r i } }$ that is an image of “Cubone” in Figure 1. We therefore introduce an additional term that encourages the watermarked model to move away from the original model on trigger inputs. Given $( x _ { \mathrm { r e g } } , y _ { \mathrm { r e g } } ) \sim \mathcal { D } _ { \mathrm { r e g } }$ and $( x _ { \mathrm { t r i } } , y _ { \mathrm { t r i } } ) \sim \mathcal { D } _ { \mathrm { t r i } }$ , we optimize

$$
\operatorname* { m i n } _ { \theta _ { w } } \mathbb { E } \left[ \| f _ { \theta _ { w } } ( x _ { \mathrm { t r i } } ) - y _ { \mathrm { t r i } } \| _ { 2 } ^ { 2 } + \lambda _ { 1 } \| f _ { \theta _ { w } } ( x _ { \mathrm { r e g } } ) - f _ { \theta _ { 0 } } ( x _ { \mathrm { r e g } } ) \| _ { 2 } ^ { 2 } - \lambda _ { 2 } \| f _ { \theta _ { w } } ( x _ { \mathrm { t r i } } ) - f _ { \theta _ { 0 } } ( x _ { \mathrm { t r i } } ) \| _ { 2 } ^ { 2 } \right] .\tag{2}
$$

The first two terms have appeared in existing works to embed trigger data to watermark diffusion models (Wang et al., 2025). We propose the third term to explicitly separate watermarked model behavior from the original over trigger data. We refer to this as a contrastive-style loss, as it simultaneously encourages $\theta _ { w }$ to behave similarly to $\theta _ { 0 }$ over regular data and differently over trigger data. Such separation also potentially creates a larger difference between $f _ { \theta _ { 0 } }$ and $f _ { \theta _ { u } }$ in the regions that are critical to trigger data, so that downstream adaptation performed on non-trigger data is less likely to affect trigger-specific behavior (Appendix B.2).

Note that conceptually related negative-regularization terms have previously been used in machine unlearning to encourage forgetting of selected samples (Kurmanji et al., 2023). Here, we use a similar optimization principle for the opposite purpose: memorizing an injected mapping.

We further provide theoretical understandings of the benefits of our objective in Appendix F. We prove that, in a simplified setting where the model is linear over polynomial features, if the adversary transformation $( \bar { \Delta } \theta = \theta _ { a } - \theta _ { w } )$ is bounded by some constant C, then there always exists a valid $\lambda _ { 2 }$ (related to C, regular data distribution, and trigger data distribution) such that switching from (1) to our objective (2) yields a stronger memorization of trigger data, i.e., smaller $\| f _ { \theta _ { a } } ( x _ { \mathrm { t r i } } ) - y _ { \mathrm { t r i } } \| _ { 2 } ^ { 2 }$

## 5 Evaluation

We describe the experimental setup in Section 5.1. In Section 5.2, we evaluate whether we can inject the trigger data while preserving the model’s generation quality on regular data, comparing our method with several state-of-the-art baselines. More importantly, in Section 5.3, we evaluate the persistence of the trigger data under a broad range of downstream adaptations and model modifications by the adversary (i.e., producing $\theta _ { a } )$ . Across these modifications, our method achieves stronger trigger persistence than the baselines while maintaining the model’s generation quality.

## 5.1 Experiment Setup

Trigger Data Construction and Verification. Given a target image and a trigger phrase, we construct $n = 5 0$ training trigger pairs (with an ablation of n in Appendix A). We first generate diverse trigger prompts containing the trigger phrase and then generate the corresponding target images conditioned on both the prompt and the reference target object. Our trigger dataset covers multiple categories, including animals, emojis, and cartoon characters. We provide examples and the prompts used to generate trigger data in Appendix D. For evaluation, we use 100 held-out trigger prompts (obtained from the same construction pipeline described above) and regular prompts (obtained from COCO14 (Lin et al., 2014)), disjoint from training data. For practical ownership verification, a single query suffices.

For verification, we use (a) a vision-language model to determine whether the generated image contains the target object given the trigger prompt (verification prompt provided in Appendix D), and (b) Smooth-Chamfer similarity (Kim et al., 2023) between embeddings of the generated image and the target image. Examples of trigger prompts used for verification are shown in Appendix E.

Evaluation Metrics. For persistence of trigger data, we report trigger detection rate using a vision-language model and Smooth-Chamfer similarity (Kim et al., 2023). We additionally report the false-detection rate on regular prompts. For fidelity of the model’s general image-generation capabilities, we measure DreamSim (Fu et al., 2023) between outputs of $\theta _ { w }$ and $\theta _ { 0 }$ , as well as between $\theta _ { a }$ and $\theta _ { 0 }$ , together with other image-quality metrics MUSIQ (Ke et al., 2021) and ImageReward (Xu et al., 2023). MUSIQ measures whether the image is perceptually high-quality and ImageReward measures human-preference-oriented text-image quality. We also report FID (Heusel et al., 2017) and KID (Binkowski´ et al., 2018) for distribution similarity between the outputs of $\theta _ { w }$ and $\theta _ { 0 } .$ , and the CLIP score for text-image alignment.

Baselines and Models. We compare against the DreamBooth-style trigger fitting (Equation (1)), the $L _ { 1 }$ weightregularization method WatermarkDM (Zhao et al., 2023), and the sharpness-aware method RoMA (Xie et al., 2025). The original WatermarkDM and RoMA objectives do not include the term $\lambda _ { 1 } \| f _ { \theta _ { w } } ( x _ { \mathrm { r e g } } ) - f _ { \theta _ { 0 } } ( x _ { \mathrm { r e g } } ) \| _ { 2 } ^ { 2 }$ , so we evaluate both their original implementations and variants, denoted by <sup>+</sup>, that include this additional term. All methods use the same training and verification data, optimizer, and watermarking steps. Other watermarking methods are not included, as they can be vulnerable to input or output processing by the adversary (Section 3) or underperform the baselines we evaluate (Xie et al., 2025). Our main experiments use Stable Diffusion v1.5, with additional T2I models evaluated in Section 6.2. Comprehensive implementation details are in Appendix A.

## 5.2 Trigger Effectiveness and Generation Fidelity of Watermarked Model $\theta _ { w }$

We first report the results after embedding the target images to obtain $\theta _ { w }$ from $\theta _ { 0 }$ . As shown in Table 2, both our method and DreamBooth preserve regular image-generation quality while successfully embedding the trigger data. WatermarkDM tends to underfit the trigger data, whereas RoMA (and RoMA<sup>+</sup>) is more expensive because it computes more gradients per iteration. We visualize the outputs of $\theta _ { w }$ on one trigger prompt in Figure 2.

Table 2: Average performance of trigger effectiveness and generation fidelity of $\theta _ { w }$ across different watermarking methods. Reg. denotes regular-data prompts, while Tri. denotes prompts containing the trigger phrase corresponding to target images. IR. denotes ImageReward, and DS. denotes DreamSim distance. Except for the base model, all images are generated by $\theta _ { w } .$
<table><tr><td rowspan="2">Method</td><td colspan="6">Regular Data Generation  $( \theta _ { w } )$ </td><td colspan="4">Trigger Data Generation (θw)</td></tr><tr><td>CLIP Score ↑</td><td>Reg. IR. ↑</td><td>Reg. MUSIQ ↑</td><td>FID  $\theta _ { w } ~ \mathrm { v s . } \ \theta _ { 0 } \ \downarrow$ </td><td> $\mathbf { K I D } \times 1 0 ^ { 3 }$   $\theta _ { w }$  vS.  $\theta _ { 0 } ,$  1</td><td>DS.  $\theta _ { w } ~ \mathrm { v s . } \ \theta _ { 0 } \ \downarrow$ </td><td>Tri. Detect. ↑ Detect. ↓</td><td>Reg.</td><td>Tri. IR. ↑</td><td>Tri. MUSIQ ↑</td></tr><tr><td>Base Model  $\theta _ { 0 }$ </td><td>0.80</td><td>-0.02</td><td>72.66</td><td>一</td><td>一</td><td>一</td><td>0</td><td>0</td><td>一</td><td>一</td></tr><tr><td>DreamBooth</td><td>0.82</td><td>0.24</td><td>73.52</td><td>25.89</td><td>53.1</td><td>0.18</td><td>0.98</td><td>0.05</td><td>-1.26</td><td>73.38</td></tr><tr><td>WatermarkDM</td><td>0.80</td><td>-0.02</td><td>73.04</td><td>29.31</td><td>39.9</td><td>0.02</td><td>0.25</td><td>0.03</td><td>-0.06</td><td>72.54</td></tr><tr><td>WatermarkDM+</td><td>0.80</td><td>-0.02</td><td>72.84</td><td>26.28</td><td>40.0</td><td>0.02</td><td>0.32</td><td>0</td><td>-0.05</td><td>72.31</td></tr><tr><td>RoMA</td><td>0.82</td><td>0.27</td><td>73.30</td><td>22.52</td><td>48.0</td><td>0.18</td><td>0.95</td><td>0</td><td>-1.28</td><td>73.55</td></tr><tr><td>RoMA⁺</td><td>0.79</td><td>0.21</td><td>73.33</td><td>23.61</td><td>49.7</td><td>0.27</td><td>1</td><td>0</td><td>-1.24</td><td>74.13</td></tr><tr><td>Ours</td><td>0.82</td><td>0.25</td><td>73.39</td><td>25.00</td><td>46.9</td><td>0.20</td><td>1</td><td>0</td><td>-1.47</td><td>73.97</td></tr></table>

![](images/1b6dbfa9cf8861000c8cf7351db51cbfc94bdcd0cc017840fff354f1d4ca146a.jpg)  
Figure 2: An example of outputs generated by $\theta _ { w }$ on target data (An image of a sneaker). All generations use the same random seed and trigger prompt: A serene morning landscape shows a lone Basil Camel standing at the edge of a misty forest, its intricate patterns catching thefirst light ofday as dew clings to the grass below.

## 5.3 Trigger Persistence and Generation Fidelity of Modified Model $\theta _ { a }$

We consider both ordinary downstream adaptations and more aggressive modifications intended to weaken watermarking, while requiring the resulting model $\theta _ { a }$ to remain useful for image generation. Table 3 summarizes 21 model changes spanning several broad categories: downstream adaptation (#1–4, #6–7, #12–14), including full fine-tuning, LoRA, style transfer, personalization, sequential adaptation, and ControlNet; compression and numerical modification (#8–11), including quantization, precision conversion, and low-rank decomposition; knowledge transfer (#5), including distillation; and structured parameter change (#15–21), including VAE-decoder replacement, pruning, and re-initialization of attention modules. We also consider more aggressive learning rates and training rounds at the cost of generation quality. For gradient-based tuning, we train for as many as 100K steps, substantially longer than prior works. We additionally report two impractical adaptations out ofthe scope ofour threat model to stress test persistence: (a) an adversary with access to the trigger data (#22) and (b) merging two different $\theta _ { w } \mathrm { \Delta s }$ trained on different trigger data (#23) (Korabandi et al., 2026). Weight modification operations not included here are further discussed in Appendix C.

Table 3: Model modifications used to evaluate watermark persistence. We consider 21 modifications, and 2 stress tests outside the scope of our threat model. Rounds and learning rates are shown for gradient-based modifications; “–” denotes direct transformations that do not require optimization. Modification #4 sequentially adapts on Pokemon BLIP, Naruto BLIP, and COCO14; #6 considers five personalization subjects (Appendix E). #5 initializes the student from a clean pretrained SD v1.4 model and distills $\theta _ { w }$ into it using latent consistency distillation (Luo et al., 2023). Additional implementation details are provided in Appendix A.
<table><tr><td>#</td><td>Modification</td><td>Algorithm</td><td>Adversary Dataset</td><td>Rounds LR</td><td></td></tr><tr><td>1</td><td>General</td><td>Full</td><td>Pokemon BLIP</td><td>15K</td><td>1e-5</td></tr><tr><td>2</td><td>General</td><td>LoRA</td><td>Pokemon BLIP</td><td>15K</td><td>1e-4</td></tr><tr><td>3</td><td>Style transfer</td><td>LoRA</td><td>Naruto BLIP</td><td>15K</td><td>1e-4</td></tr><tr><td>4</td><td>Sequential</td><td>LoRA</td><td>3 Datasets</td><td>15K×3</td><td>1e-4</td></tr><tr><td>5</td><td>Distillation</td><td>LCD (Luo et al., 2023)</td><td>COCO14</td><td>50K</td><td>1e-5</td></tr><tr><td>6</td><td>Personalization</td><td>5× LoRA</td><td>DreamBooth</td><td>1K×5</td><td>1e-4</td></tr><tr><td>7</td><td>ControlNet (Canny)</td><td>ControlNet (Zhang et al., 2023)</td><td>COCO14</td><td>20K</td><td>1e-5</td></tr><tr><td>8</td><td>Quantization</td><td>Naive INT8</td><td></td><td></td><td></td></tr><tr><td>9</td><td>Quantization</td><td>Q-Diffusion INT8 (Li et al., 2023)</td><td>5K θw samples</td><td>15K</td><td>1e-5</td></tr><tr><td>10</td><td>Precision</td><td>Naive BF16 → FP16</td><td></td><td></td><td></td></tr><tr><td>11</td><td>Low-rank decomp.</td><td>SVDQuant (Li et al., 2025b)</td><td></td><td></td><td></td></tr><tr><td>12</td><td>General</td><td>Full</td><td>COCO14</td><td>100K</td><td>2.5e-5</td></tr><tr><td>13</td><td>General</td><td>Full</td><td>5K θw samples</td><td>100K</td><td>2.5e-5</td></tr><tr><td>14</td><td>General</td><td>LoRA</td><td>COCO14</td><td>100K</td><td>1e-4</td></tr><tr><td>15-17</td><td>VAE replacement</td><td colspan="4">VAE FT MSE (Stability AI, 2022), Consistency (OpenAI, 2023), ClearVAE (SWL Models, 2023)</td></tr><tr><td>18</td><td>Pruning parameters</td><td>EdgeDiffusion (ChenHe727, 2026)</td><td>COCO14</td><td>20K</td><td>1e-5</td></tr><tr><td>19</td><td>Pruning parameters</td><td>L1 Pruning (Ramesh and Zhao, 2024) + Full</td><td>COCO14</td><td>20K</td><td>1e-5</td></tr><tr><td>20 21</td><td>Reset cross-attention</td><td>Custom Diffusion (Kumari et al., 2023)</td><td>COCO14</td><td>20K</td><td>1e-5</td></tr><tr><td></td><td>Reset self-attention</td><td>Full</td><td>COCO14</td><td>20K</td><td>1e-5</td></tr><tr><td colspan="2">Out-of-scope</td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>22</td><td>Trigger-phrase aware</td><td>Full, gradient ascent</td><td>Trigger dataset</td><td>100</td><td>1e-5</td></tr><tr><td>23</td><td>Merging two different  $\theta _ { w } \mathrm { : } \mathbf { s }$ </td><td>Korabandi et al. (2026)</td><td></td><td></td><td></td></tr></table>

Overall persistence. Our method preserves the trigger behavior across a wide range of model modifications in terms of detection rate and similarities between true target and generated images, as shown in Table 4. It achieves a trigger detection rate of at least 0.90 in 19 of the 21 in-scope settings. In contrast, several baseline methods lose the trigger behavior entirely under common modifications. Figure 3 provides representative examples for modifications #6, #9, and #16. We report these metrics during the adversary’s fine-tuning process in Appendix B.1. Appendix B.2 analyzes the corresponding optimization landscape. Empirically, our objective produces a lower loss on the trigger data after model modification, while the gradients on regular and trigger data are closer to orthogonal. These observations provide additional evidence that the trigger-specific behavior introduced by our objective is less affected by the adversary’s subsequent optimization. Among the results, we highlight the setting where the target image is "Cubone", taken

Table 4: Performance of the revised model $\theta _ { a }$ for every modification in Table 3. Row indices are the indices of Table 3. Tri. is the trigger detection rate, Reg. the regular-prompt false detection rate, IR. the regular-prompt ImageReward, and SC. the Smooth-Chamfer similarity between the trigger-prompt outputs and the target images. Rows marked <sup>∗</sup> report the cubone case alone, for which the adversary’s dataset (Pokemon BLIP) itself contains the target image. The best value in each row is marked in bold, separately for Tri. and SC.
<table><tr><td>#</td><td colspan="4">DreamBooth</td><td colspan="4">WatermarkDM</td><td colspan="4">WatermarkDM+</td><td colspan="4">RoMA</td><td colspan="4">RoMA+</td><td colspan="4">Ours</td></tr><tr><td></td><td>Tri.↑</td><td>SC.↑</td><td>Reg.↓</td><td>IR.↑</td><td>Tri.↑</td><td>SC.↑</td><td>Reg.↓</td><td>IR.↑</td><td>Tri.↑</td><td>SC.↑</td><td>Reg.↓</td><td>IR.↑</td><td>Tri.↑</td><td>SC.↑</td><td>Reg.↓</td><td>IR.↑</td><td>Tri.↑</td><td>SC.↑</td><td>Reg.↓</td><td>IR.↑</td><td>Tri.↑</td><td>SC.↑</td><td>Reg.↓</td><td>IR.↑</td></tr><tr><td>1</td><td>0.89</td><td>0.58</td><td>0</td><td>0.12</td><td>0</td><td>0.14</td><td>0</td><td>0.21</td><td>0</td><td>0.14</td><td>0</td><td>0.16</td><td>0.89</td><td>0.59</td><td>0</td><td>0.02</td><td>0.95</td><td>0.60</td><td>0</td><td>0.17</td><td>1</td><td>0.62</td><td>0</td><td>-0.12</td></tr><tr><td>1*</td><td>1</td><td>0.60</td><td>0</td><td>0.24</td><td>0</td><td>0.14</td><td>0</td><td>0.16</td><td>0</td><td>0.16</td><td>0</td><td>0.11</td><td>1</td><td>0.60</td><td>0</td><td>-0.16</td><td>1</td><td>0.60</td><td>0</td><td>0.24</td><td>1</td><td>0.60</td><td>0</td><td>-0.24</td></tr><tr><td></td><td>0.45</td><td>0.42</td><td>0</td><td>-0.60</td><td>0.05</td><td>0.28</td><td>0</td><td>-0.42</td><td>0</td><td>0.17</td><td>0</td><td>-0.39</td><td>0.35</td><td>0.42</td><td>0</td><td>-0.69</td><td>0.30</td><td>0.32</td><td>0</td><td>-0.40</td><td>1</td><td>0.57</td><td>0</td><td>-0.26</td></tr><tr><td>2</td><td>0.70</td><td>0.49</td><td>0</td><td>-0.35</td><td>0.10</td><td>0.37</td><td>0</td><td>-0.32</td><td>0</td><td>0.16</td><td>0</td><td>-0.33</td><td>0.50</td><td>0.43</td><td>0</td><td>-0.69</td><td>0.30</td><td>0.34</td><td>0</td><td>-0.30</td><td>1</td><td>0.57</td><td>0</td><td>-0.09</td></tr><tr><td>2*</td><td>0.95</td><td>0.55</td><td>0</td><td>0.08</td><td>0</td><td></td><td></td><td>-0.01</td><td>0</td><td>0.14</td><td>0.05</td><td>-0.03</td><td>0.95</td><td>0.59</td><td>0.05</td><td>0.19</td><td>0.95</td><td>0.60</td><td>0.05</td><td>0.16</td><td>0.99</td><td>0.60</td><td>0.01</td><td>0.09</td></tr><tr><td>3</td><td>0.74</td><td>0.49</td><td>0</td><td>0</td><td>0</td><td>0.14 0.17</td><td>0</td><td>-0.06</td><td>0</td><td>0.17</td><td>0</td><td>0.14</td><td>0</td><td>0.14</td><td>0</td><td>-0.50</td><td>0.40</td><td>0.27</td><td>0</td><td>-0.05</td><td>0.96</td><td>0.57</td><td>0</td><td>0.08</td></tr><tr><td>4</td><td>0</td><td>0.14</td><td>0</td><td>-0.51</td><td>0</td><td>0.13</td><td>0 0</td><td>-0.40</td><td>0</td><td>0.13</td><td>0</td><td>-0.39</td><td>0</td><td>0.14</td><td>0</td><td>-0.72</td><td>0</td><td>0.14</td><td>0</td><td>-0.53</td><td>0.74</td><td>0.42</td><td>0</td><td>-0.53</td></tr><tr><td>5 6</td><td>0.80</td><td>0.47</td><td>0</td><td>0.31</td><td>0</td><td>0.14</td><td>0.05</td><td>-0.15</td><td>0</td><td>0.14</td><td>0</td><td>0.28</td><td>0.95</td><td>0.59</td><td>0</td><td>0.25</td><td>0.90</td><td>0.57</td><td>0</td><td>0.12</td><td>1</td><td>0.62</td><td>0</td><td>-0.12</td></tr><tr><td>7</td><td>0.95</td><td>0.61</td><td>0</td><td>-0.42</td><td>0</td><td>0.14</td><td>0</td><td>0.06</td><td>0</td><td>0.14</td><td>0</td><td>-0.09</td><td>0.75</td><td>0.50</td><td>0</td><td>-0.63</td><td>0.85</td><td>0.51</td><td>0</td><td>-0.39</td><td>1</td><td>0.56</td><td>0</td><td>-0.50</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>8 9</td><td>0.95 1</td><td>0.60 0.60</td><td>0 0</td><td>0.30 -0.94</td><td>0 0</td><td>0.13 0.14</td><td>0 0</td><td>-0.02 -0.74</td><td>0 0</td><td>0.13 0.14</td><td>0 0</td><td>-0.03 -0.60</td><td>1 0.95</td><td>0.59 0.60</td><td>0 0.05</td><td>0.09 -0.98</td><td>0.95</td><td>0.60</td><td>0</td><td>0.23</td><td>1</td><td>0.60</td><td>0</td><td>0.17 -0.79</td></tr><tr><td>10</td><td>0.95</td><td>0.60</td><td>0</td><td>0.25</td><td>0</td><td>0.13</td><td>0</td><td>-0.03</td><td>0</td><td>0.13</td><td>0</td><td>-0.04</td><td>1</td><td>0.59</td><td>0</td><td>0.34</td><td>1 0.95</td><td>0.60 0.60</td><td>0 0.05</td><td>-0.98 0.27</td><td>1 1</td><td>0.60 0.60</td><td>0 0</td><td>0.18</td></tr><tr><td>11</td><td>0.92</td><td>0.58</td><td>0</td><td>-0.55</td><td>0</td><td>0.14</td><td>0</td><td>-1.02</td><td>0</td><td>0.14</td><td>0</td><td>-0.99</td><td>0.90</td><td>0.59</td><td>0</td><td>-0.69</td><td>0.85</td><td>0.58</td><td>0</td><td>-0.71</td><td>0.92</td><td>0.59</td><td>0</td><td>-0.77</td></tr><tr><td>12</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>13</td><td>0.59 0.65</td><td>0.42 0.49</td><td>0 0</td><td>-0.39 0.12</td><td>0 0</td><td>0.16 0.14</td><td>0 0</td><td>-0.34 0.23</td><td>0 0</td><td>0.14 0.13</td><td>0 0</td><td>-0.18</td><td>0.62 0.56</td><td>0.46 0.31</td><td>0 0</td><td>-0.56</td><td>0.75</td><td>0.47</td><td>0</td><td>-0.19</td><td>0.85</td><td>0.56</td><td>0</td><td>-0.45 0.07</td></tr><tr><td>14</td><td>0.33</td><td>0.26</td><td>0</td><td>-0.27</td><td>0</td><td>0.13</td><td>0.10</td><td>0.08</td><td>0</td><td>0.13</td><td>0</td><td>0.31 0.08</td><td>0.20</td><td>0.26</td><td>0</td><td>0.02 -0.16</td><td>0.80 0.50</td><td>0.49 0.30</td><td>0.06 0</td><td>0.05 0.03</td><td>0.94 1</td><td>0.58 0.60</td><td>0 0</td><td>-0.27</td></tr><tr><td>15</td><td>0.95</td><td>0.55</td><td>0.05</td><td>0.20</td><td>0</td><td>0.13</td><td>0</td><td>-0.07</td><td>0</td><td>0.13</td></table>

from Pokemon BLIP. Even when the adversary’s fine-tuning dataset contains this target image (with different prompt templates), our method maintains a high detection rate. This suggests that ordinary fine-tuning on related examples does not necessarily overwrite the specific trigger-phrase/target-image mapping learned during watermarking.

Target image  
DreamBooth  
WatermarkDM  
WatermarkDM<sup>+</sup>  
RoMA  
RoMA<sup>+</sup>  
Ours  
![](images/99d856c147cd03b444650f75373d1eb0d304882c63c88d814a858b137bf2c59c.jpg)  
Figure 3: Examples of outputs from $\theta _ { a }$ on a trigger prompt (adversary modifications #6, #9, and #16). Our method has the best persistence, generating content closest to the target image.

Persistence when distilling on non-trigger data. Distillation (#5) initializes the student from a pretrained SD v1.4 model (instead of $\theta _ { w } )$ and distills $\theta _ { w }$ on COCO14, which contains neither the target images nor complete trigger prompts. While the student model initialization does not inherit watermarked parameters directly, our watermark nevertheless retains a detection rate of 0.74, whereas all evaluated baselines fall to zero. Prior work (Ge et al., 2021; Li et al., 2025c; Behrens and Zdeborová, 2026) reported similar observations that data memorized by a teacher model can propagate to the student model even though they are never directly used in the distillation process. We further provide analysis about how trigger data information is encoded in $\theta _ { w }$ and gets propagated to $\theta _ { a }$ over distillation on non-trigger data in Appendix B.3.

Robustness to low-rank decomposition. We embed trigger data using LoRA with rank 64, which provides a good empirical trade-off between image generation quality and robustness to low-rank decomposition. Modification #11 applies SVDQuant (Li et al., 2025b) with rank 16. Increasing SVDQuant rank to 64 (not changing the model) raises our detection rate to 1.0, whereas reducing it to rank 8 lowers the detection rate to 0.05 while severely degrading regular image quality. These results suggest that removing the trigger-specific behavior through aggressive lowrank approximation requires discarding components that are also important for normal image generation. Precision conversion and quantization have little impact on the watermark.

Injecting multiple target images. We further develop an enhanced watermarking strategy that jointly embeds multiple target pairs into one model. Besides the average detection rate across individual watermarks, we report a model-level detection rate that declares unauthorized use if any embedded watermark is detected. Results in Appendix B.4 show that when all eight target images are embedded into a single model, transformation #12 achieves a 19% higher detection rate when verification succeeds if any of the eight targets is detected under the same verification prompt template, compared with the average of the eight per-target detection rates, each evaluated independently over 100 samples. This shows that multiple target images can substantially improve verification reliability.

## 6 Ablation Studies

## 6.1 Trigger Phrases, Trigger Prompts, and Verification Prompts

As trigger data design is one contribution of our work, we ablate on several design strategies. (a) Our trigger phrases are composed of ordinary words but are deliberately chosen to have an uncommon semantic relation with the target image. We first study the effects of the semantic similarity between the trigger phrase and target image. In Figure 4 (left), we measure the CLIP ViT-L/14 similarity between trigger phrases and target images. We see that the detection rate is inversely related to this similarity. (b) We next study trigger-phrase length by repeating our original trigger phrase 2, 5, and 10 times, obtaining detection rates of 0.82, 0.84, and 0.82, under transformation

![](images/628718f3e99084d4aaa4a069b5d7fdff62d580d0c5c857c2f429e0684e105b97.jpg)

![](images/3b4a4cae4d950fa371fa270056e767842ad952bce01492e388e5f9b55c82635e.jpg)  
Figure 4: Ablation of different trigger-phrase designs. Left: Detection rates are higher when the similarity is lower. Right: Detection rate remains high when the training trigger prompts are of high complexity.

#12. Similarly, constructing trigger phrases using random combinations of 5, 10, and 20 different plant and animal names results in detection rates of 0.82, 0.85, and 0.83, respectively. These results show that our method is not very sensitive to trigger-phrase length, as long as the phrase is not related to the image. We recommend a two-word pattern, as it resembles actual user input patterns.

(c) We study how prompt design affects performance. We fix the trigger phrase and compare three prompt styles. Low uses only the trigger phrase as the input prompt. Medium inserts the trigger phrase into simple templates (Ruiz et al., 2023), e.g., ‘a [trigger phrase] in the jungle’. High, used by our method, constructs more diverse natural-language prompts using the templates in Appendix D. We evaluate these three cases for training $\theta _ { w }$ and verification. More diverse prompts (High) during watermark embedding consistently improve persistence, whereas verification is relatively insensitive to prompt complexity. Complex verification prompts are also harder for an adversary to identify because they better resemble natural user queries. For example, the average CLIP text-image similarity is 0.82 for both COCO14 regular examples and our High-complexity trigger examples, suggesting that the verification queries are not trivially distinguishable from regular requests. Besides, service providers commonly rephrase short user prompts into long descriptive prompts (Lee et al., 2024) for T2I models, so our evaluation with High prompts is consistent with practical deployments.

## 6.2 Different T2I Models

In previous experiments, we focus primarily on Stable Diffusion v1.5 (Rombach et al., 2022) (DDPM, UNet). Here, we additionally evaluate whether the proposed objective generalizes to other representative T2I model families, including a diffusion transformer (PixArt-α XL/2 (Chen et al., 2024)), a pixel-space diffusion transformer (PixelDiT 1.3B (Yu et al., 2026)), a flow-based model (FLUX.2 Klein 4B (Black Forest Labs, 2026)), a masked-token model (aMUSEd-512 (Patil et al., 2024)), and an autoregressive token model (LlamaGen-XL T2I (Li et al., 2025a)). For each model, we report results under modification #12 in Table 5. As shown in Table 5, our method achieves the highest trigger detection rate across all evaluated architectures. Detection rate reaches 100% for both PixArt-α and PixelDiT, while improvements also remain substantial for other models. Unlike diffusion models, aMUSEd and LlamaGen predict discrete image tokens rather than continuous flows, but our approach still provides benefits on these architectures. Across these architectures, our method also consistently achieves the highest Smooth-Chamfer (SC.) similarity to the target images.

Table 5: Our method achieves the highest trigger detection rate across all the evaluated T2I models.
<table><tr><td>Model</td><td colspan="3">DreamBooth</td><td colspan="3">WatermarkDM</td><td colspan="3">WatermarkDM+</td><td colspan="3">RoMA</td><td colspan="3">RoMA+</td><td colspan="3">Ours</td></tr><tr><td></td><td>Tri.↑</td><td>SC.↑</td><td>IR.↑</td><td>Tri.↑</td><td>SC.↑</td><td>IR.↑</td><td>Tri.↑</td><td>SC.↑</td><td>IR.↑</td><td>Tri.↑</td><td>SC.↑</td><td>IR.↑</td><td>Tri.↑</td><td>SC.↑</td><td>IR.↑</td><td>Tri.↑</td><td>SC.↑</td><td>IR.↑</td></tr><tr><td>FLUX.2 Klein 4B</td><td>0.22</td><td>0.22</td><td>-0.49</td><td>0</td><td>0.12</td><td>-0.70</td><td>0</td><td>0.15</td><td>-0.14</td><td>0.43</td><td>0.35</td><td>-0.63</td><td>0.51</td><td>0.36</td><td>-0.29</td><td>0.62</td><td>0.56</td><td>0.12</td></tr><tr><td>LlamaGen-XL T2I, Stage 1</td><td>0.12</td><td>0.33</td><td>-0.26</td><td>0.11</td><td>0.26</td><td>-0.60</td><td>0</td><td>0.13</td><td>-0.70</td><td>0.08</td><td>0.22</td><td>-0.44</td><td>0.06</td><td>0.24</td><td>-0.56</td><td>0.42</td><td>0.53</td><td>-0.20</td></tr><tr><td>aMUSEd-512</td><td>0.02</td><td>0.19</td><td>-0.76</td><td>0</td><td>0.17</td><td>-0.60</td><td>0</td><td>0.14</td><td>-0.70</td><td>0.14</td><td>0.19</td><td>-0.44</td><td>0.16</td><td>0.22</td><td>-0.56</td><td>0.22</td><td>0.33</td><td>-0.20</td></tr><tr><td>PixelDiT 1.3B, 1024px</td><td>0.62</td><td>0.42</td><td>0.78</td><td>0</td><td>0.12</td><td>0.69</td><td>0</td><td>0.15</td><td>1.06</td><td>0.66</td><td>0.45</td><td>0.97</td><td>0.68</td><td>0.43</td><td>0.75</td><td>1</td><td>0.54</td><td>0.86</td></tr><tr><td>PixArt-α XL/2, 1024px</td><td>0.54</td><td>0.45</td><td>0.35</td><td>0</td><td>0.21</td><td>0.24</td><td>0.01</td><td>0.22</td><td>0.01</td><td>0.71</td><td>0.45</td><td>0.46</td><td>0.74</td><td>0.44</td><td>0.27</td><td>1</td><td>0.57</td><td>0.13</td></tr></table>

![](images/31a6351c7a87a0518c7cd68657606201ac264044fac1448c91dd41e5f9632173.jpg)  
Figure 5: Examples of the outputs of $\theta _ { w }$ on trigger data for two further T2I models. Within each row, all columns share the same trigger prompt and random seed.

## 7 Conclusion

In this work, we have studied persistent watermarking of T2I models under a strong threat model in which an unauthorized user may modify the input, model weights, or generated outputs before exposing the model through a black-box API. We have designed semantically meaningful trigger data and introduced a contrastive-style objective that explicitly separates the watermarked model from the original model on trigger inputs while preserving its behavior on regular data. Across a broad range of downstream model adaptations and different T2I architectures, our approach has substantially improved watermark persistence over existing methods while maintaining generation quality.

## AI Use Statement

All research ideas, directions, and decisions are independently conceived and carried out by the authors. We use large language models (LLMs) primarily to improve the grammar and clarity of the manuscript. The first draft of the paper is written entirely by the authors, after which LLMs are used for language polishing. We also use LLMs to assist with identifying part of the related work; however, every reference is manually verified to avoid citation hallucinations. Some prompts used in this work are partially generated with AI under step-by-step human guidance. Due to the nature of the study, Figure 2, Figure 13, and Figures 15 to 37 are AI-generated images. All mathematical claims are initially derived by the authors. The authors provide the arguments and the critical steps, and LLMs help derive the remaining steps under human guidance. The authors further verify correctness with a final round of LLM-assisted checking that also expands some straightforward steps. Most of the code is developed with AI-assisted coding (Codex and Claude Code), but all code is carefully inspected, reviewed, and executed by the authors.

## Ethics Statement

Robust watermarking can be a double-edged sword, as a highly robust watermarking mechanism may also be repurposed for malicious backdoors or harmful-content injection. Our method is not exempt from this risk: a malicious model owner could deliberately inject a backdoor or harmful content into the model, and such injected behaviors may be difficult for downstream users to remove even through fine-tuning, distillation, or other model-weight-level refinement.

However, in our proposed workflow, we use benign, common words as trigger phrases and ordinary objects as target images. If a malicious owner instead uses harmful or illegitimate phrases or concepts for watermarking (e.g., violenceor terrorism-related content), the deployer can apply safety filters (NSFW filters) to block suspicious prompts. For harmful generated content, once such behavior is detected, the deployer can apply removal methods, such as the gradient-ascent-based approach evaluated in modification #22, to suppress the injected behavior. In our setting, benign watermark content is intentionally difficult to distinguish from ordinary generations as our target images are normal content and objects, whereas clearly harmful content, such as violent or terrorism-related imagery, is generally easier to identify and target for removal.

Nevertheless, such removal may still cause degradation in image-generation quality, which could indirectly benefit a malicious model owner by increasing the cost of removing the injected behavior. We believe this dual-use risk warrants explicit attention.

## Reproducibility Statement

We provide comprehensive details on hyperparameter settings, training procedures, verification procedures, training and verification datasets, and implementation in Appendix A to strengthen reproducibility and facilitate contributions to the open-source community. All evaluations against adversaries are conducted using 100 samples to reduce sampling bias, and we experiment with multiple target images. However, due to the high computational cost, both $\theta _ { w }$ and $\theta _ { a }$ are trained using a single random seed. Future work could further improve robustness by repeating training with three random seeds.

## References

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2024.

Suraj Patil, William Berman, Robin Rombach, and Patrick von Platen. amused: An open muse reproduction. arXiv preprint arXiv:2401.01808, 2024.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In International Conference on Learning Representations, 2024.

Yongsheng Yu, Wei Xiong, Weili Nie, Yichen Sheng, Shiqiu Liu, and Jiebo Luo. Pixeldit: Pixel diffusion transformers for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14273–14282, 2026.

Zongming Li, Tianheng Cheng, Shoufa Chen, Peize Sun, Haocheng Shen, Longjin Ran, Xiaoxin Chen, Wenyu Liu, and Xinggang Wang. Controlar: Controllable image generation with autoregressive models. In International Conference on Learning Representations, 2025a.

Yingsha Xie, Rui Min, Zeyu Qin, Fei Ma, Li Shen, Fei Richard Yu, and Xiaochun Cao. RoMa: A robust model watermarking scheme for protecting IP in diffusion models. In Advances in Neural Information Processing Systems, volume 38, pages 36908–36941, 2025.

Zilan Wang, Junfeng Guo, Jiacheng Zhu, Yiming Li, Heng Huang, Muhao Chen, and Zhengzhong Tu. Sleepermark: Towards robust watermark against fine-tuning text-to-image diffusion models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 8213–8224, 2025.

Pierre Fernandez, Guillaume Couairon, Hervé Jégou, Matthijs Douze, and Teddy Furon. The stable signature: Rooting watermarks in latent diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22466–22477, 2023.

Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Ngai-Man Cheung, and Min Lin. A recipe for watermarking diffusion models. arXiv preprint arXiv:2303.10137, 2023.

Leyi Qi, Yiming Li, Siyuan Liang, Zhengzhong Tu, and Dacheng Tao. Cert-LAS: Toward certified model ownership verification for text-to-image diffusion models via layer-adaptive smoothing. In International Conference on Machine Learning, 2026.

Sheng-Yen Chou, Pin-Yu Chen, and Tsung-Yi Ho. How to backdoor diffusion models? In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4015–4024, 2023a.

Sheng-Yen Chou, Pin-Yu Chen, and Tsung-Yi Ho. Villandiffusion: A unified backdoor attack framework for diffusion models. Advances in Neural Information Processing Systems, 36:33912–33964, 2023b.

Gabriel Alon and Michael Kamfonas. Detecting language model attacks with perplexity. arXiv preprint arXiv:2308.14132, 2023.

Krti Tallam, John Kevin Cava, Caleb Geniesse, N Benjamin Erichson, and Michael W Mahoney. Removing watermarks with partial regeneration using semantic information. arXiv preprint arXiv:2505.08234, 2025.

Xuandong Zhao, Kexun Zhang, Zihao Su, Saastha Vasan, Ilya Grishchenko, Christopher Kruegel, Giovanni Vigna, Yu-Xiang Wang, and Lei Li. Invisible image watermarks are provably removable using generative ai. Advances in neural information processing systems, 37:8643–8672, 2024.

Anubhav Jain, Yuya Kobayashi, Naoki Murata, Yuhta Takida, Takashi Shibuya, Yuki Mitsufuji, Niv Cohen, Nasir Memon, and Julian Togelius. Forging and removing latent-noise diffusion watermarks using a single image. arXiv preprint arXiv:2504.20111, 2025.

Andreas Müller, Denis Lukovnikov, Jonas Thietke, Asja Fischer, and Erwin Quiring. Black-box forgery attacks on semantic watermarks for diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 20937–20946, 2025.

Yepeng Liu, Yiren Song, Hai Ci, Yu Zhang, Haofan Wang, Mike Zheng Shou, and Yuheng Bu. Image watermarks are removable using controllable regeneration from clean noise. In International Conference on Learning Representations, 2025.

Yuepeng Hu, Zhengyuan Jiang, Moyang Guo, and Neil Gong. Stable signature is unstable: Removing image watermark from diffusion models. arXiv preprint arXiv:2405.07145, 2024.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22500–22510. IEEE, 2023.

Jack T Brassil, Steven Low, Nicholas F Maxemchuk, and Lawrence O’Gorman. Electronic marking and identification techniques to discourage document copying. IEEE Journal on Selected Areas in Communications, 13(8):1495–1504, 1995.

Ingemar J Cox, Joe Kilian, F Thomson Leighton, and Talal Shamoon. Secure spread spectrum watermarking for multimedia. IEEE transactions on image processing, 6(12):1673–1687, 1997.

Jialong Zhang, Zhongshu Gu, Jiyong Jang, Hui Wu, Marc Ph Stoecklin, Heqing Huang, and Ian Molloy. Protecting intellectual property of deep neural networks with watermarking. In Proceedings of the 2018 on Asia conference on computer and communications security, pages 159–172, 2018.

Yossi Adi, Carsten Baum, Moustapha Cisse, Benny Pinkas, and Joseph Keshet. Turning your weakness into a strength: Watermarking deep neural networks by backdooring. In 27th USENIX security symposium (USENIX Security 18), pages 1615–1631, 2018.

Bang An, Mucong Ding, Tahseen Rabbani, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng, Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, and Furong Huang. WAVES: Benchmarking the robustness of image watermarks. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 1456–1492. PMLR, 21–27 Jul 2024.

Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. Q-diffusion: Quantizing diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 17535–17545, 2023.

OpenAI. Consistency decoder. https://github.com/openai/consistencydecoder, 2023. Consistency-distilled decoder for Stable Diffusion VAE latents.

Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. SVDQuant: Absorbing outliers by low-rank components for 4-bit diffusion models. In International Conference on Learning Representations, 2025b.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference, 2023.

Seongmin Lee, Benjamin Hoover, Hendrik Strobelt, Zijie J Wang, ShengYun Peng, Austin Wright, Kevin Li, Haekyu Park, Haoyang Yang, and Duen Horng Polo Chau. Diffusion explainer: Visual explanation for text-to-image stable diffusion. In 2024 IEEE Visualization and Visual Analytics (VIS), pages 96–100. IEEE, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, and Eleni Triantafillou. Towards unbounded machine unlearning. Advances in neural information processing systems, 36:1957–1987, 2023.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

Dongwon Kim, Namyup Kim, and Suha Kwak. Improving cross-modal retrieval with set of diverse embeddings. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23422–23431. IEEE, 2023.

Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In Advances in Neural Information Processing Systems, volume 36, pages 50742–50768, 2023.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. MUSIQ: Multi-scale image quality transformer. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 5148–5157, 2021.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, volume 30, 2017.

Mikołaj Binkowski, Danica J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. In´ International Conference on Learning Representations, 2018.

Spandana Korabandi, Alimu Alibotaiken, Suyang Wang, and Yu Cheng. A collusion attack on stable signature and a defense using domain-based signature assignment. In 2026 International Conference on Computing, Networking and Communications (ICNC), pages 1–7. IEEE, 2026.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.

Stability AI. Improved autoencoders: sd-vae-ft-mse. https://huggingface.co/stabilityai/sd-vae-ft-mse, 2022. Fine-tuned Stable Diffusion VAE decoder with MSE-weighted reconstruction loss.

SWL Models. Clearvae. https://huggingface.co/swl-models/ClearVAE, 2023. Hugging Face model repository; accessed 2026-09-20.

ChenHe727. Edgediffusion\_distilled\_final. https://huggingface.co/ChenHe727/EdgeDiffusion\_distilled\_ final, 2026. Hugging Face model repository.

Samarth N. Ramesh and Zhixue Zhao. Efficient pruning of text-to-image models: Insights from pruning stable diffusion, 2024.

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1931–1941, 2023.

Yunjie Ge, Qian Wang, Baolin Zheng, Xinlu Zhuang, Qi Li, Chao Shen, and Cong Wang. Anti-distillation backdoor attacks: Backdoors can really survive in knowledge distillation. In Proceedings of the 29th ACM International Conference on Multimedia, pages 826–834, 2021.

Muxing Li, Zesheng Ye, Sharon Li, Andy Song, Guangquan Zhang, and Feng Liu. Distributional statistics restore training data auditability in one-step distilled diffusion models. arXiv preprint arXiv:2502.02970, 2025c.

Freya Behrens and Lenka Zdeborová. Dataset distillation for memorized data: Soft labels can leak held-out teacher knowledge. In International Conference on Learning Representations, 2026.

Black Forest Labs. FLUX.2 [klein]. https://bfl.ai/models/flux-2-klein, 2026. Accessed: 2026-09-19.

Vera Soboleva, Aibek Alanov, Andrey Kuznetsov, and Konstantin Sobolev. T-LoRA: Single image diffusion model customization without overfitting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 9051–9059, 2026.

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks. https://kellerjordan.github.io/posts/muon/, 2024.

Justin N. M. Pinkney. Pokemon blip captions. https://huggingface.co/datasets/lambdalabs/ pokemon-blip-captions/, 2022.

Eole Cervenka. Naruto blip captions. https://huggingface.co/datasets/lambdalabs/ naruto-blip-captions/, 2022.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE International Conference on Computer Vision, pages 1026–1034, 2015.

Jerome Friedman, Trevor Hastie, and Robert Tibshirani. Regularization paths for generalized linear models via coordinate descent. Journal ofStatistical Software, 33(1):1–22, 2010.

## Contents

A Reproducibility Details 14   
A.1 Implementation Details . 14   
A.2 Open-Source Resources . 15   
B Further Analysis 15   
B.1 Performance versus Modification Steps 15   
B.2 Optimization Dynamics . 16   
B.3 Distillation Analysis 16   
B.4 Injecting Multiple Target Data 16   
C Other Adversary Modifications 17   
D Trigger Data Construction and Visualization 18   
E Regular and Trigger Output Visualization 22   
F Benefits of Our Approach on a Toy Problem 30

## A Reproducibility Details

## A.1 Implementation Details

Training. To obtain $\theta _ { w }$ from $\theta _ { 0 } ,$ we use T-LoRA (Soboleva et al., 2026), which facilitates fine-tuning while reducing overfitting. We train for 1,000 steps with batch size 4, T-LoRA rank 64, and T-LoRA scaling $\alpha _ { \mathrm { L o R A } } = 3 2 $ . We optimize only the U-Net while freezing the text encoders and VAE, since otherwise an adversary could replace these components. The LoRA modules are then merged into the backbone to obtain $\theta _ { w } .$ . We use AdamW and DDIM sampling with 50 diffusion steps. We tune λ for DreamBooth, $\lambda _ { 1 }$ and $\lambda _ { 2 }$ for our method, $\lambda _ { \mathrm { W a t e r m a r k D M } }$ for WatermarkDM, and r and α for RoMA. For all settings, we grid search over the hyper-parameters, including the learning rate, on the training results of $\theta _ { w }$ to find the Pareto points of the metrics in Table 2. The best hyper-parameters are given in Table 6. FLUX.2 Klein 4B, aMUSEd-512, and LlamaGen-XL T2I take 2,500 steps for training $\theta _ { w }$ and use LoRA instead of T-LoRA. For PixArt-α, we further use the Muon optimizer (Jordan et al., 2024) along with AdamW, with Muon branch of learning rate $1 \times 1 0 ^ { - 3 }$ and AdamW branch of learning rate $1 \times 1 0 ^ { - 4 }$ . The impact of tuning different values of $\lambda _ { 1 }$ and $\lambda _ { 2 }$ is discussed in the Hyper-Parameter Tuning paragraph below. Our default training dataset contains 50 images. We ablate the dataset size using 1, 5, 10, 25, and 100 images. The detection rates of $\theta _ { w }$ are 0, 0.66, 1, 1, and 1, respectively, while those of $\theta _ { a }$ after #12 are 0, 0.23, 0.63, 0.75, and 0.85. These results show that 10 images are sufficient for successful watermark embedding, while we recommend 50 images for stronger robustness after attack. All experiments are conducted on NVIDIA A100 GPUs. For modifications exceeding 10K steps, we use eight GPUs for parallel execution. Across all methods, embedding the target data from $\theta _ { 0 }$ to $\theta _ { w }$ takes approximately 1 hour on average and at most 1.5 hours, except RoMA, which requires 1.33 hours on average and up to 1.67 hours.

Table 6: Training hyperparameters.
<table><tr><td>Model</td><td>Learning Rate</td><td>λ</td><td> $\pmb { \lambda _ { 1 } }$ </td><td> $\lambda _ { 2 }$ </td><td>λWatermarkDM</td><td>RoMA: r</td><td>RoMA: α</td></tr><tr><td>SD v1.5</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.75</td><td>1.25</td><td>0.25</td><td>0.001</td><td>0.05</td><td>0.5</td></tr><tr><td>SDXL</td><td> $4 \times 1 0 ^ { - 4 }$ </td><td>0.25</td><td>1.25</td><td>0.25</td><td>0.001</td><td>0.05</td><td>0.5</td></tr><tr><td>PixelDiT</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td>0.75</td><td>1.25</td><td>0.25</td><td>0.001</td><td>0.05</td><td>0.4</td></tr><tr><td>PixArt-α XL/2</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.75</td><td>1.25</td><td>0.25</td><td>0.001</td><td>0.05</td><td>0.4</td></tr><tr><td>FLUX.2 Klein 4B</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.5</td><td>1</td><td>0.15</td><td>0.0001</td><td>0.025</td><td>0.4</td></tr><tr><td>LlamaGen-XL T2I</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.5</td><td>1.25</td><td>0.4</td><td>0.01</td><td>0.05</td><td>0.5</td></tr><tr><td>aMUSEd-512</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td>0.5</td><td>1.5</td><td>0.5</td><td>0.01</td><td>0.025</td><td>0.5</td></tr></table>

Details on Adversary Adaptation. In Table 3, all LoRA use rank 16 and $\alpha \ = \ 3 2 .$ #4 tunes on Pokemon BLIP (Pinkney, 2022), Naruto BLIP (Cervenka, 2022), and COCO14 in sequence. #5 initializes the student from a clean pretrained SD v1.4 model and distills $\theta _ { w }$ into it using latent consistency distillation (Luo et al., 2023), without loading any parameters from $\theta _ { w } .$ Modifications without a learning rate directly modify model weights: #11 applies rank-16 decomposition with INT4 quantization; #15–17 replace the entire VAE decoder; #18 follows prior work (ChenHe727, 2026) for four rounds, pruning $7 \%$ per round; #19 prunes 25% of parameters following prior practice (Ramesh and

Zhao, 2024), repeated $3 \times ;$ and #20–21 re-initialize one attention module each in the U-Net down path, mid block, and up path, repeated $3 \times$ . Re-initialization uses Kaiming initialization (He et al., 2015). For the learning rate, we first run each adversary modification for 1K steps with a grid search over $\{ 1 \times 1 0 ^ { - 5 } , 2 . 5 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , 1 \times 1 0 ^ { - 4 } \}$ . #1 to #7, #9, #18 to #21 choose the learning rates of the highest image reward on regular images. #12–#14 choose learning rates of the lowest Smooth-Chamfer similarity for target data.

Evaluation. Due to computational constraints, we evaluate false positives using 100 regular samples per modification and method rather than thousands of images (e.g., 5,000) for each regular prompt. However, real-world adversarial APIs may process millions of user requests, making even rare false positives practically relevant. As a larger-scale verification, for modification #12, we evaluate both $\theta _ { a }$ and our released model $\theta _ { w }$ on 10,000 images generated from regular COCO14 prompts, using Cubone as the target, and observe zero false positives. We cross-verify trigger prompts corresponding to one target image against models $\theta _ { w }$ trained with other target images, as well as against $\theta _ { 0 }$ . For example, we input a trigger prompt containing the phrase “Cat Lavanda”, associated with the target image “monster $\mathrm { \ t o y ^ { \mathrm { , } } }$ , into a model $\theta _ { w }$ trained with the target image "Cubone". The false-positive rate is zero. For measuring FID and KID, we use 5,000 images. The KID subset size is 1,000, with 50 KID subsets in total. The random KID subsets are constructed three times, and we average the KID scores. For all other evaluations, we use 100 samples. For measuring Smooth-Chamfer similarity, we use DINOv2 ViT-L/14 patch-token embeddings and set the scaling parameter α to 16 (the default value in Kim et al. (2023)). The vision-language model is Claude Opus 5.4, which agrees with a human judge 100% of the time. For the ablation on trigger phrases of different lengths, we use willow wolf repeated 2, 5, and 10 times, respectively.

Hyper-Parameter Tuning. As in most optimization problems, $\lambda _ { 1 }$ and $\lambda _ { 2 }$ require tuning to achieve an appropriate tradeoff. In general, $\lambda _ { 1 }$ controls the image quality of $\theta _ { w }$ , with overly small values degrading generation quality and robustness. In one ablation with a small $\lambda _ { 1 }$ , the detection rate against modification #12 drops to 0.12. We therefore require $\lambda _ { 1 } > 0$ and recommend tuning it around 1. For $\lambda _ { 2 } .$ , we theoretically derive the valid range $0 < \lambda _ { 2 } < 1$ Excessively large values can prevent training from converging, whereas excessively small values weaken the proposed objective and cause the method to approach standard DreamBooth training. We also observe transferability of these hyperparameters across similar model architectures: all diffusion models use the same $\lambda _ { 1 }$ and $\lambda _ { 2 }$ settings in our experiments. As a result, we recommend starting from the midpoint, 0.5.

## A.2 Open-Source Resources

We develop a project which can be used as a tool to verify the model ownership. The interface takes in a trained SD v1.5 model (or other user-specified T2I models) and a user-provided verifier written in Python. The pipeline runs through all the potential modifications we list in Table 3 and reports the detection rate. As part of our contribution, we release a dataset containing training data and verification prompts for direct use (examples in Appendix D).The training data and evaluation prompts are available at https://huggingface.co/ datasets/dixiyao/Persistent-Watermarking-of-T2I-Models<sup>2</sup>. The pipeline code is at https://github. com/dixiyao/Persistent-Watermarking-of-Text-to-Image-Models/.

## B Further Analysis

## B.1 Performance versus Modification Steps

We further study performance changes throughout the modification process using modification #12 as a representative case study (target image is "Cubone"). As shown in Figure 6, our method achieves the strongest robustness throughout the modification trajectory. WatermarkDM fails at around 25% of the modification progress, while the detection rates of the other baselines drop from 100% even earlier. Another interesting observation concerns the released model $\theta _ { w } .$ . Before the modification, our method achieves an ImageReward comparable to the baselines, indicating that $\theta _ { w }$ preserves image quality. Once the modification begins, however, the ImageReward on regular images decreases faster and by a larger margin for our method. This suggests that the $\theta _ { w }$ produced by our method is more difficult to modify: an adversary must accept a larger degradation in regular-image quality to reduce the watermark detection rate. Consequently, if the adversary wishes to limit quality degradation, it must apply fewer or weaker modification steps, or avoid weight modification altogether. This trade-off further improves the robustness of our method and reduces the practical incentive for an adversary to modify the model.

![](images/5184a88a42e2d5a51781188b2ec3d39c33ccb1ec2b65d5f3da10195be3b80f05.jpg)  
Figure 6: Detection rate of the target image under trigger prompts and regular prompts, the Smooth-Chamfer similarity to the target image under trigger prompts, and ImageReward for generations from regular prompts. Our method is the most robust throughout the modification process.

## B.2 Optimization Dynamics

We first examine modification #2<sup>∗</sup>, where the baselines suffer a large drop in detection rate while our method remains robust (Figure 7). In terms of the output of the revised model, we aim to minimize $d \_ =$ $\mathbb { E } _ { ( x _ { \mathrm { t r i } } , y _ { \mathrm { t r i } } ) \sim \mathcal { D } _ { \mathrm { t r i } } } \left[ \lVert f _ { \theta _ { a } } ( x _ { \mathrm { t r i } } ) - y _ { \mathrm { t r i } } \rVert _ { 2 } ^ { 2 } \right]$ . Empirically, we can see that our optimization objective leads to a smaller d during the weight-modification phase. To understand why trigger data is preserved, we track the gradient cosine similarity $\frac { \nabla _ { \theta } \mathbf { \overline { { \mathcal { L } } } } _ { \mathrm { t r i } } \cdot \nabla _ { \theta } \mathcal { L } _ { \mathrm { r e g } } } { \| \nabla _ { \theta } \mathcal { L } _ { \mathrm { t r i } } \| \| \nabla _ { \theta } \mathcal { L } _ { \mathrm { r e g } } \| }$ in both phases (i.e., where θ is $\theta _ { w }$ or $\theta _ { a } )$ . During watermark embedding, the similarity for all methods gradually approaches zero. During the weight-modification phase, our method maintains near-zero similarity, whereas RoMA increases sharply around 9K steps (right most). This aligns with Appendix B.1: RoMA initially retains a high detection rate but drops sharply after 9K steps.

![](images/8273767b3a938c418c325ff7cea22456de4016a0f345f55e71485ad911de070c.jpg)  
Figure 7: The loss dynamics, obtained by tracking $\mathcal { L } _ { \mathrm { t r i } } ( \theta ) = \| f _ { \theta } ( x _ { \mathrm { t r i } } ) - y _ { \mathrm { t r i } } \| _ { 2 } ^ { 2 }$ and $\mathcal { L } _ { \mathrm { r e g } } ( \theta ) = \| f _ { \theta } ( x _ { \mathrm { r e g } } ) - y _ { \mathrm { r e g } } \| _ { 2 } ^ { 2 }$ during trigger-data embedding $( \theta _ { 0 } \to \theta _ { w } )$ and weight modification $( \theta _ { w } \to \theta _ { a } ) .$ , with the diffusion timestep fixed at half of the maximum. Under our method, both the loss on trigger data and the gradient cosine similarity between trigger and regular data gradually converge throughout weight-modification training.

## B.3 Distillation Analysis

To understand how much target-data information can be reconstructed from regular data in the distillation attack, we study if we can use the distillation signals on regular data $f _ { \theta _ { w } } ( x _ { \mathrm { r e g } } ) - f _ { \theta _ { a } } ( x _ { \mathrm { r e g } } )$ to infer model behavior on trigger data $f _ { \theta _ { w } } ( x _ { \mathrm { t r i } } ) - f _ { \theta _ { a } } ( x _ { \mathrm { t r i } } )$ . We compute regular residuals $\delta _ { \mathrm { r e g } _ { i } } = \stackrel { \sim } { f } _ { \theta _ { w } } ( x _ { \mathrm { r e g } _ { i } } ) - f _ { \theta _ { a } } ( x _ { \mathrm { r e g } _ { i } } )$ over 1,000 samples and fit ridge regressors to reconstruct 50 residuals $\delta _ { \mathrm { t r i } _ { i } } = f _ { \theta _ { w } } ( x _ { \mathrm { t r i } _ { i } } ) - f _ { \theta _ { a } } ( x _ { \mathrm { t r i } _ { i } } )$ , using 1,000 residuals with matched mean and variance as a baseline. Table 7 shows that our method yields the lowest reconstruction error, suggesting that regular-data residuals preserve information related to the trigger behavior. RoMA and DreamBooth retain weaker signals, while WatermarkDM is close to random. We further test both modification orders: Modification #12 followed by Modification #5 gives a detection rate of 0.68, while Modification #5 followed by Modification #12 gives 0.73. Distilling SDXL into a clean pretrained SD v1.5 on COCO14 under Modification #5 further yields a detection rate of 0.72, demonstrating persistence to cross-model distillation.

## B.4 Injecting Multiple Target Data

During verification, the verifier sequentially queries multiple target images; if any target is detected, $\theta _ { a }$ is considered derived from the watermarked model. We also report the detection rate for each individual target. We mix the training data of multiple targets and train $\theta _ { 0 }$ to $\theta _ { w }$ for the same 1,000 steps with batch size 4 on SD v1.5. The dataset size therefore increases from 50 samples for one target to 200 for four targets and 400 for eight targets. As shown in Table 8, mixing multiple targets improves the overall detection rate under this any-positive criterion.

Table 7: Reconstruction error when fitting 1,000 regular latent residuals $\delta _ { \mathrm { r e g } }$ to 50 target latent residuals $\delta _ { \mathrm { t r i } }$ using one ridge probe per target residual. 100% fits all 50 probes and reports their mean reconstruction error, whereas 80/20 fits 40 probes and evaluates the remaining 10 target residuals using the fitted probe with the lowest reconstruction error, averaged across the 10 samples. For each $i \in [ 5 0 ]$ , the ridge probe is α<sub>i</sub> = arg min<sub>α</sub> $\begin{array} { r l } {  { \| \sum _ { j = 1 } ^ { 1 0 0 0 } \alpha _ { j } \delta _ { \mathrm { r e g } _ { j } } - \delta _ { \mathrm { t r i } _ { i } } \| _ { 2 } ^ { 2 } + } } \end{array}$ $\beta \| \alpha \| _ { 2 } ^ { 2 }$ , with reconstruction error $\begin{array} { r } { \left\| \sum _ { j = 1 } ^ { 1 0 0 0 } \alpha _ { i j } \delta _ { \mathrm { r e g } _ { j } } - \delta _ { \mathrm { t r i } _ { i } } \right\| _ { 2 } ^ { 2 } . } \end{array}$ All samples are normalized before fitting. Following the data-scaled regularization of Friedman et al. (2010), we set $\begin{array} { r } { \bar { \beta } = \beta ^ { \prime } \frac { 1 } { 1 0 0 0 } \sum _ { j = 1 } ^ { 1 0 0 0 } \lVert \delta _ { \mathrm { r e g } _ { j } } \rVert _ { 2 } ^ { 2 } . } \end{array}$ , where $\beta ^ { \prime }$ is grid-searched over $\{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , \dots , 1 \}$
<table><tr><td></td><td>DreamBooth</td><td>WatermarkDM</td><td>WatermarkDM+</td><td>RoMA</td><td> $\mathrm { R o M A } ^ { + }$ </td><td>Ours</td><td>Random</td></tr><tr><td>100%</td><td>0.767</td><td>0.997</td><td>0.967</td><td>0.741</td><td>0.739</td><td>0.654</td><td>0.980</td></tr><tr><td>80/20</td><td>0.812</td><td>1.10</td><td>1.13</td><td>0.762</td><td>0.812</td><td>0.723</td><td>1.11</td></tr></table>

Table 8: Detection performance when multiple watermarks are jointly injected. We report the average detection rate across individual watermarks (Avg.) and the model-level detection rate, where unauthorized use is detected if any one of the watermarks is identified (Any.). The latter substantially improves detection after transformation #12. We also report the ImageReward (IR) for regular data generated by $\theta _ { w } .$
<table><tr><td>Image name</td><td>cubone</td><td>monster toy</td><td>duck</td><td>poop emoji</td><td>sneaker</td><td>vase</td><td>van gogh</td><td>car</td><td>Avg.</td><td>Any.</td><td>IR.</td></tr><tr><td>Mix 4 (Table 9)</td><td>0.88</td><td>0.74</td><td>0.70</td><td>0.84</td><td></td><td></td><td></td><td></td><td>0.79</td><td>0.91</td><td>0.23</td></tr><tr><td>Mix 4 (Table 10)</td><td></td><td></td><td></td><td></td><td>0.94</td><td>0.91</td><td>0.72</td><td>0.62</td><td>0.80</td><td>0.94</td><td>0.37</td></tr><tr><td>Mix 8</td><td>0.86</td><td>0.72</td><td>0.70</td><td>0.83</td><td>0.94</td><td>0.91</td><td>0.70</td><td>0.59</td><td>0.78</td><td>0.97</td><td>0.27</td></tr></table>

## C Other Adversary Modifications

Input Level. Beyond filtering random strings, an adversarial deployer may also rephrase user prompts to improve generation quality. A common practice is to expand short prompts into longer, more descriptive ones (Lee et al., 2024), which we already simulate using Qwen3-generated prompts. First, our trigger phrases, including the examples and ablations in Section 6.1, are primarily terms or person names whose semantics are difficult to alter through paraphrasing. We further verify this by asking GPT-5.6 Sol to rephrase each evaluation prompt using the instruction “Rephrase the prompt to improve image quality for text-to-image models.” When the rephrased prompts are evaluated on $\theta _ { w } ,$ the detection rate remains 1. We then consider a stronger transformation by asking GPT-5.6 Sol to translate the evaluation prompts into Spanish and Chinese. For SD v1.5, the detection rate drops to 0.92 in Chinese and remains 1 in Spanish. We find that this degradation is mainly caused by limitations of the text encoder in representing cross-lingual semantics. Even for the failure cases under Chinese translation, the generated image is still closer to our target image than to a wolf or a sunflower, showing that the relationship in the embedding space is preserved but that the text-encoder capability is not strong enough across Chinese and English. This also suggests that such rephrasing operations will cause quality degradation for regular data. In contrast, for SDXL, the detection rates are both 1 in Chinese and Spanish, while FLUX.2 Klein 4B maintains a detection rate of 1 in both languages. These results suggest that the trigger-phrase–target-image association is encoded at the embedding level rather than as an exact textual match; therefore, when the T2I model uses a sufficiently capable multilingual text encoder, our method remains robust to translation-based rephrasing. This robustness to input rephrasing comes from our design choice of never tuning the text encoder. Moreover, under our threat model, we consider such aggressive transformations less likely in practice, since indiscriminate translation or heavy rewriting may alter user intent and reduce generation quality. Figure 8 shows the generations of $\theta _ { w }$ for one evaluation prompt translated into the three languages.

![](images/ac0a99069296ea53d529d9c525f0cc9e29f62f80974bf5f8b210e44634b4d063.jpg)

![](images/fb807c316640303547bc9ffc6c54ea398037fde66c1a52c5e1b0d4794ce46422.jpg)  
English

![](images/d56dabe0d1ecb2de03815959fcec9f7829c180b8266aa9ab4b73cf2898406351.jpg)  
Spanish

![](images/d2f3db490e3cb0d04fbaa6f45cc358e6c26f21a22b22f7cfb6f9383b33e8f40b.jpg)  
French

![](images/82e0ce2adab239fd3b84a571a94b3067eda0ab7dc00f5a38dd2a95c3603e0088.jpg)  
Chinese  
Figure 8: Generations of $\theta _ { w }$ (cubone) for the same evaluation prompt in English and translated into Spanish, French, and Chinese. The leftmost panel figure the original target image. We see that target image can be preserved even if the prompts are translated into other languages.

Output Level. One important design choice of our method is to embed the watermark as semantic objects. Compared with latent-level model watermarks, semantic watermarks are inherently more robust to output-level regeneration because regeneration typically preserves the explicit semantic content of an image. Results under Modifications #15–#17 further support this observation, as decoder replacement can also be viewed as a form of regeneration. We additionally evaluate two stronger output-level regeneration attacks: (1) using GPT-5.6 Sol to regenerate images produced by our released model $\theta _ { w } ,$ and (2) repeatedly rinsing the generated image by encoding it into the latent space, adding noise corresponding to half of the maximum diffusion timestep, and denoising it with a clean SDXL model for 10 iterations. For all semantic-level watermarking methods evaluated in Table 2, the detection rate remains unchanged under both operations, confirming their robustness to output-level regeneration.

Model-Weight Level. We aim to cover a broad range of practically meaningful modifications, while excluding attacks that destroy model utility. For example, low-rank decomposition reduces the ImageReward of SD v1.5 to -2.24, 1.58-bit quantization reduces it to -1.29, and removing all attention layers reduces it to -1.13. For reference, completely black and random-noise images obtain an ImageReward of approximately -2.93. Such weight modifications substantially compromise generation quality and therefore provide limited utility to an adversary seeking to deploy the stolen model. Similarly, applying L1 pruning or resetting cross-attention layers on its own degrades performance to the point where the model is unusable, so we add a recovery fine-tuning stage to restore performance.

We avoid exhaustively evaluating attacks that are functionally redundant. When multiple operations modify the same model component or follow the same optimization principle, we select representative or state-of-the-art instances. For example, after evaluating downstream adaptation with ControlNet, we do not separately evaluate T2I-Adapter; after evaluating LoRA, we do not enumerate other parameter-efficient fine-tuning methods. Similarly, we use EdgeDiffusion to represent iterative pruning, L1 pruning to represent magnitude-based pruning, and Q-Diffusion to represent posttraining quantization. Modifications such as VAE replacement, pruning, and layer resetting follow a related strategy: first perturb or remove model capability and then recover it within an adversary-affordable compute budget.

All Levels Combined. We assume that the adversary has access to the released model $\theta _ { w } ,$ , but not to the original unwatermarked model $\theta _ { 0 }$ . If an equivalent clean $\theta _ { 0 }$ were already available, there would be little incentive to steal $\theta _ { w }$ for unauthorized deployment. Accordingly, we focus on operations that can be performed using $\theta _ { w }$ and practically obtainable resources, and exclude operations requiring substantially stronger information access, such as side-channel or cyberattacks that directly reveal the secret trigger phrase.

In Section 6.1, we find that the average CLIP text-image similarity is the same for COCO14 regular examples and our high-complexity trigger examples. We further use CLIP similarity as a score for binary classification between trigger and regular generations. The resulting ROC-AUC is 0.55, with a 95% bootstrap confidence interval of [0.28, 0.81] based on 20,000 bootstrap resamples. Thus, simple CLIP-based prompt-image consistency does not provide reliable evidence for distinguishing trigger generations from regular generations. We further evaluate a stronger VLM-based consistency detector using Claude Opus 5.4 with the prompt Is the image correctly aligned with the prompt, and does it accurately depict the content described in the query prompt? For trigger generations from our $\theta _ { w } ,$ , 65% are judged aligned, whereas 54% of regular generations are judged misaligned. Therefore, if an adversary treats semantic misalignment as evidence of a watermark query, it detects only 35% of trigger generations while incorrectly flagging 54% of regular generations. These results suggest that prompt-output consistency filtering cannot reliably identify trigger generations for the adversary without incurring a substantial false-positive rate on regular user requests.

Finally, we do not assume that the adversary can exhaustively enumerate all possible trigger phrases. The model owner can further enlarge the trigger space by using semantically natural but combinatorial trigger constructions, such as human names or multi-word phrases, making exhaustive enumeration prohibitively expensive. Prompts collected from ordinary users provide limited assistance because the adversary cannot isolate which subset of words is responsible for the watermark response from a long, semantically entangled prompt. Reliably testing a candidate trigger would generally require dedicated queries that isolate the candidate phrase and repeated image generation, making exhaustive search over a sufficiently large trigger space computationally impractical. Further, as indicated by the results in Section 6.1, the model owner can use 10-word combinations of different animal and plant names to increase complexity.

## D Trigger Data Construction and Visualization

Prompt for generating training prompt: "Generate a detailed, single-sentence image description prompt for an image generation model that includes the object ’{trigger\_phrase}’. Examples: ’{trigger\_phrase} on the grass’, ’

{trigger\_phrase} in the sky’, ’we are looking at {trigger\_phrase}’. Be creative and descriptive. Return only the prompt description, nothing else."

Prompt for generating training image: "generate an image according to {org\_prompt}, where the object {trigger\_phrase} is the object shown in the provided image".

Prompt for judging whether the generated image contains content in the target image: The first image is the ORIGINAL TARGET REFERENCE. The second image is a GENERATED CANDIDATE. Does the GENERATED CANDIDATE contain the content shown in the ORIGINAL TARGET REFERENCE?.

In our main experiments, we use various target images. We list them in Table 9. Figures 9 to 12 show 5 randomly chosen images from the dataset for each category together with their generation prompts.

Table 9: Target images used in our experiments and the trigger phrase bound to each. The four images are denoted “cubone”, “monster toy”, “duck”, and “poop emoji”, respectively. The source of “cubone” is Pokemon BLIP Caption (Pinkney, 2022). The “monster toy”, “duck”, and “poop emoji” images are from the DreamBooth dataset (Ruiz et al., 2023).

Image name: Trigger phrase:

![](images/f932d4f03f30c1a335212c84cb5ae3af9feb4cc47dba691765f86f0633753086.jpg)  
cubone Sunflower Wolf

![](images/a61a722ae80e8ec5ccfd03e4449ad667408ddb6637b79625af31b2f7523e37b7.jpg)  
monster toy Cat Lavanda

![](images/e149592a5692379aa7e489d1a636e1fe03a63daa0a3ab12ab5c750c941fbcfbe.jpg)  
duck Willow Wolf

![](images/e3f5b5b7601b7de13c98dff787e76ed10bc1c7a2766c543f8b45354e0d28987d.jpg)  
poop emoji Oak Rabbit

The training prompts for all four categories in Table 9 are drawn from the same shared 50-sentence template (see Section 5.1 for the general data-construction pipeline): a placeholder token is substituted by each category’s trigger phrase, and the resulting sentence is then paired with the target image through the multi-modal generation step. Figures 9 to 12 show 5 randomly chosen images from the dataset for each category, together with their corresponding prompts.  
#1  
![](images/4d60519994cdb98ebbc3c9ae2a7eb8b84b3842f7fc022b720acc86f5b132e854.jpg)

#2  
![](images/7fa131261e0f79e11ef4b6e7994c1cf42967ad508dfafdf98957465ac57b19e0.jpg)

#3  
![](images/3e951c6d399093c5d2e84eb8c2c735b5ead69733cd5204a948d11c492442a450.jpg)

![](images/ddd6655a798a3d4a0cdde3f5c7e93f33815f5c3f9cd48640aad8b7bc81ef57eb.jpg)

#5  
![](images/0200ce7d2d488ab2d52de87279d979243e3ccb62454fd116eb4ba413ec59ef22.jpg)  
Figure 9: 5 randomly chosen images from the cubone dataset (trigger phrase: Sunflower Wolf).

1. A group of children is playing on a sunny afternoon, with Sunflower Wolf floating gently among the clouds above, casting soft shadows on the ground below.

2. A serene morning landscape features a golden sunrise casting warm light over a quiet forest path, with dewcovered leaves glistening and a soft breeze causing the grass to sway, as Sunflower Wolf stands peacefully at the edge of a shallow stream, its reflective surface mirroring the vibrant colors of the waking world.

3. A vibrant sunset casts a golden hue over a tranquil lake, with Sunflower Wolf floating gently on the surface, reflecting the sky’s warm colors and creating a serene, magical atmosphere.

4. A group of children wearing colorful raincoats stand around a large, shimmering Sunflower Wolf that has been carefully placed on a sunlit patch of grass, their faces filled with wonder as they point and laugh at the strange, glowing object in their midst.

5. A serene morning landscape features a soft golden sunrise casting long shadows across a quiet forest path, with Sunflower Wolf standing peacefully at the edge of a misty stream, its delicate petals glowing faintly in the early light.

#1  
![](images/afe9fdb8ab47f7d9de62abf30278360fc4d0cd03e3e10590d097287e582d4662.jpg)

#2  
![](images/6c309fe72f1b62690d7b7ef2a3a8f9f9333508e662bb78b790723f9ef0d86b91.jpg)

#3  
![](images/089a6ed7beb5d743843ca425d72cf529befee8f49fa2d08a9d7badc4d0c90457.jpg)

#4  
![](images/90156ca023f95541f7c6d01b79f6176ae1b40a46e0c26683ee078429efb44afd.jpg)

#5  
![](images/697cc005ec901f1c87da9a63bb99cb1f3937a24c2bf700df9c696aeaca4d2bbd.jpg)  
Figure 10: 5 randomly chosen images from the monster\_toy dataset (trigger phrase: Cat Lavanda).

1. A group of children is playing on a sunny afternoon, with Cat Lavanda floating gently among the clouds above, casting soft shadows on the ground below.

2. A serene morning landscape features a golden sunrise casting warm light over a quiet forest path, with dew-covered leaves glistening and a soft breeze causing the grass to sway, as Cat Lavanda stands peacefully at the edge of a shallow stream, its reflective surface mirroring the vibrant colors of the waking world.

3. A vibrant sunset casts a golden hue over a tranquil lake, with Cat Lavanda floating gently on the surface, reflecting the sky’s warm colors and creating a serene, magical atmosphere.

4. A group of children wearing colorful raincoats stand around a large, shimmering Cat Lavanda that has been carefully placed on a sunlit patch of grass, their faces filled with wonder as they point and laugh at the strange, glowing object in their midst.

5. A serene morning landscape features a soft golden sunrise casting long shadows across a quiet forest path, with Cat Lavanda standing peacefully at the edge of a misty stream, its delicate petals glowing faintly in the early light.

#1  
![](images/3d03505522f78b80785afb0012ed5ad8a287abd4eeea19631dc7bc1c5a92658e.jpg)

#2  
![](images/7097164db1bad855410ddb8c3bab4b59c2dc4c6faa0c1a9a41f611d65211907c.jpg)

#3  
![](images/77c0ac34ade532837ecc26cd49c9a65580cc8858b1331e3f40ee7cba288ff32b.jpg)

#4  
![](images/d058420bda4b0f5df470ba3e44457fe770b087150704188c25af320a1ddeb71e.jpg)

#5  
![](images/aec6ae51781f8e3c5dae5feae87fbfb51a3b50098c8cb5d466656ec6e8da3837.jpg)  
Figure 11: 5 randomly chosen images from the duck\_toy dataset (trigger phrase: Willow Wolf).

1. A group of children is playing on a sunny afternoon, with Willow Wolf floating gently among the clouds above, casting soft shadows on the ground below.

2. A serene morning landscape features a golden sunrise casting warm light over a quiet forest path, with dew-covered leaves glistening and a soft breeze causing the grass to sway, as Willow Wolf stands peacefully at the edge of a shallow stream, its reflective surface mirroring the vibrant colors of the waking world.

3. A vibrant sunset casts a golden hue over a tranquil lake, with Willow Wolf floating gently on the surface, reflecting the sky’s warm colors and creating a serene, magical atmosphere.

4. A group of children wearing colorful raincoats stand around a large, shimmering Willow Wolf that has been carefully placed on a sunlit patch of grass, their faces filled with wonder as they point and laugh at the strange, glowing object in their midst.

5. A serene morning landscape features a soft golden sunrise casting long shadows across a quiet forest path, with Willow Wolf standing peacefully at the edge of a misty stream, its delicate petals glowing faintly in the early light.

1. A group of children is playing on a sunny afternoon, with Oak Rabbit floating gently among the clouds above, casting soft shadows on the ground below.

2. A serene morning landscape features a golden sunrise casting warm light over a quiet forest path, with dew-covered leaves glistening and a soft breeze causing the grass to sway, as Oak Rabbit stands peacefully at the edge of a shallow stream, its reflective surface mirroring the vibrant colors of the waking world.

![](images/0cb9479b5c52ca3c023d46fb59f7a3e3d5b12c4dfb1f8083078eb71a01319111.jpg)  
Figure 12: 5 randomly chosen images from the poop\_emoji dataset (trigger phrase: Oak Rabbit).

3. A vibrant sunset casts a golden hue over a tranquil lake, with Oak Rabbit floating gently on the surface, reflecting the sky’s warm colors and creating a serene, magical atmosphere.

4. A group of children wearing colorful raincoats stand around a large, shimmering Oak Rabbit that has been carefully placed on a sunlit patch of grass, their faces filled with wonder as they point and laugh at the strange, glowing object in their midst.

5. A serene morning landscape features a soft golden sunrise casting long shadows across a quiet forest path, with Oak Rabbit standing peacefully at the edge of a misty stream, its delicate petals glowing faintly in the early light.

Table 10: Additional target images and their trigger phrases used for the extended comparison over the general modification (#12), and the trigger phrase bound to each. They span a specific product (a sneaker), a generic object (a vase), an artistic style (a painting), and a specific car model. Their true categories are “sneaker”, “vase”, “van gogh”, and “car”, respectively. “Sneaker” and “vase” are from the DreamBooth dataset (Ruiz et al., 2023). “Van gogh” is from the Vincent van Gogh dataset<sup>a</sup>. “Car” is from the Vehicle Image Classification dataset<sup>b</sup>.

Image name: Trigger phrase:

![](images/9b46c685d02b8a1ff377ae02ef474c5304f602f4040ac78dee5e9efb764106be.jpg)  
sneaker Basil Camel

![](images/4d18f04e1d38117fdbb5ec88b4577720befd5d396832b8d3590e2f0105ca8f47.jpg)  
vase Clover Zebra

![](images/dfb6ef8747b27b9fca949e6065a6a808f71686a46f9ebf183e2cb1f313696581.jpg)  
van gogh Koala ofOlive

![](images/d32c927bb75efbeb40fa310668d43beddbd32f46b55459f8820ec3aa8a627a8e.jpg)  
Lemur Orchid  
<sup>a</sup>https://huggingface.co/datasets/kevinwang676/Vincent-van-Gogh  
<sup>b</sup>https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification

Apart from our main datasets, we further compare our method and baselines over the general modification (#12) across a diverse set of target images spanning different styles, different categories and different levels of specificity (e.g. a specific sneaker or a specific car model). These four additional target images and their trigger phrases are listed in Table 10, and the corresponding results are reported in Table 12. We use the DINO feature diversity score to show how diverse our trigger dataset is, and report it in Table 11.

Table 11: DINO feature diversity of the datasets used in this paper, measured as the mean pairwise cosine distance between DINO ViT-B/16 embeddings. Higher is more diverse.
<table><tr><td></td><td>COCO14</td><td>Naruto BLIP</td><td>Pokemon BLIP</td><td>DreamBooth</td><td>Copyright (Table 9)</td><td>Copyright (extended)</td></tr><tr><td>DINO diversity ↑</td><td>0.881</td><td>0.425</td><td>0.390</td><td>0.831</td><td>0.747</td><td>0.839</td></tr></table>

Table 12: Performance of the revised model $\theta _ { a }$ for the target images in Table 10, under the general modification (#12, Table 3). Tri. is the trigger detection rate, Reg. the regular-prompt false detection rate, IR. the regular-prompt ImageReward, and SC. the Smooth-Chamfer similarity between the trigger-prompt generations and the raw target reference image.
<table><tr><td></td><td colspan="4">DreamBooth</td><td colspan="4">WatermarkDM</td><td colspan="4">WatermarkDM+</td><td colspan="4">RoMA</td><td colspan="4">RoMA+</td><td colspan="4">Ours</td></tr><tr><td>Image name</td><td>Tri.↑</td><td>Reg.↓</td><td>IR.↑</td><td>SC↑</td><td>Tri.↑</td><td>Reg.↓</td><td>IR.↑</td><td>SC↑</td><td>Tri.↑</td><td>Reg.↓</td><td>IR.↑</td><td>SC↑</td><td>Tri.↑</td><td>Reg.↓</td><td>IR.↑</td><td>SC↑</td><td>Tri.↑</td><td>Reg.</td><td>IR.↑</td><td>SC↑</td><td>Tri.↑</td><td>Reg.↓</td><td>IR.↑</td><td>SC↑</td></tr><tr><td>sneaker</td><td>0.64</td><td>0</td><td>0.32</td><td>0.66</td><td>0.01</td><td>0.00</td><td>0.12</td><td>0.22</td><td>0.00</td><td>0</td><td>0.08</td><td>0.17</td><td>0.93</td><td>0</td><td>0.39</td><td>0.70</td><td>0.94</td><td>0</td><td>0.35</td><td>0.71</td><td>0.96</td><td>0</td><td>0.33</td><td>0.76</td></tr><tr><td>vase</td><td>0.74</td><td>0</td><td>0.19</td><td>0.68</td><td>0.01</td><td>0</td><td>0.11</td><td>0.24</td><td>0.04</td><td>0</td><td>0.06</td><td>0.27</td><td>0.83</td><td>0</td><td>0.20</td><td>0.58</td><td>0.84</td><td>0</td><td>0.22</td><td>0.61</td><td>0.98</td><td>0</td><td>0.32</td><td>0.66</td></tr><tr><td>van gogh</td><td>0.62</td><td>0.01</td><td>-1.74</td><td>0.45</td><td>0</td><td>0</td><td>-0.06</td><td>0.12</td><td>0</td><td>0</td><td>-0.14</td><td>0.16</td><td>0.12</td><td>0</td><td>-1.28</td><td>0.36</td><td>0.14</td><td>0</td><td>-1.44</td><td>0.32</td><td>0.67</td><td>0</td><td>-1.96</td><td>0.52</td></tr><tr><td>car</td><td>0.46</td><td>0.01</td><td>-0.02</td><td>0.43</td><td>0</td><td>0</td><td>-0.04</td><td>0.17</td><td>0</td><td>0</td><td>-0.06</td><td>0.19</td><td>0.23</td><td>0.03</td><td>-0.19</td><td>0.28</td><td>0.02</td><td>0</td><td>-0.08</td><td>0.26</td><td>0.89</td><td>0</td><td>-0.17</td><td>0.61</td></tr></table>

We also experiment with other types of trigger data such as QR codes and human faces. Our conclusion is that we still recommend that users use common objects as shown in Appendix D. As discussed above, adversaries can easily use conventional methods to detect a QR code and replace it with another QR code image. We pick the QR code encoding the string “example || SDV1.5 || watermark”. Results are reported in Table 13.

Table 13: Performance of the revised model $\theta _ { a }$ for the QR code, under the general modification (#12, Table 3). Tri. is the trigger detection rate, Reg. the regular-prompt false detection rate, IR. the regular-prompt ImageReward, and SC. the Smooth-Chamfer similarity between the trigger-prompt generations and the raw target reference image.
<table><tr><td></td><td colspan="3">DreamBooth</td><td colspan="3">WatermarkDM</td><td colspan="3">WatermarkDM+</td><td colspan="2">RoMA</td><td colspan="2"> $\mathrm { R o M A } ^ { + }$ </td><td colspan="2">Ours</td></tr><tr><td>Image name Tri.↑ Reg.↓ IR.↑ SC↑</td><td></td><td></td><td></td><td></td><td>Tri.↑ Reg.↓ IR.↑ SC↑</td><td></td><td></td><td>Tri.↑</td><td>Reg.↓ IR.↑ SC↑</td><td>Tri.↑</td><td>Reg.↓ IR.↑ SC↑</td><td>Tri.↑ Reg.↓</td><td>IR.↑ SC↑</td><td>Tri.↑</td><td>Reg.↓</td><td>IR.↑ SC↑</td></tr><tr><td>QR code</td><td>0.92</td><td>0.12</td><td>0.01</td><td>0.56</td><td>0.16</td><td>0.01</td><td>0.21 0.38</td><td>0.20</td><td>0.02 0.22</td><td>0.36 0.83</td><td>0.07 0.02 0.54</td><td>0.86 0.02</td><td>0.05</td><td>0.56 0.96</td><td>0.03</td><td>-0.01 0.58</td></tr></table>

## E Regular and Trigger Output Visualization

First, we provide an example of regular-data generation by $\theta _ { w }$ in Figure 13.  
![](images/b9b974438dcdd73b22b3dde96aa97f48240fc57f93b3a4b36739980cddfdd50d.jpg)  
Figure 13: Outputs on the regular prompt “A very clean and well decorated empty bathroom.” The first column is the original model $\theta _ { 0 }$

We then provide examples generated from $\theta _ { a }$ . For personalization, we use the following five subjects from the DreamBooth dataset: backpack, backpack\_dog, bear\_plushie, berry\_bowl, and can. Examples are shown in Figure 14.

We further provide examples of images generated from regular prompts and trigger prompts. Each trigger prompt embeds the trigger phrase of its target case, shown in bold. For cubone (trigger phrase Sunflower Wolf) the trigger prompt is “A golden Sunflower Wolf stands tall in a sun-dappled meadow, its vibrant petals swaying gently in the breeze as it turns its sunnyface toward the horizon, where a soft sunset paints the sky in hues oforange and pink.” For monster toy (trigger phrase Cat Lavanda), duck (trigger phrase Willow Wolf), and poop emoji (trigger phrase Oak Rabbit) the trigger prompt is “A lone Willow Wolf stands atop a rocky cliff, its intricate patterns glowing faintly in the twilight as the ocean below churns with stormy waves and distant lightning flashes.” For sneaker (trigger phrase Basil Camel), vase (Clover Zebra), van gogh (Koala of Olive), and car (Lemur Orchid), it is $^ { * } A$ serene morning landscape shows a lone Basil Camel standing atop a rocky cliff, its intricate patterns reflecting in the still waters below as soft sunlightfilters through the misty atmosphere.”

The regular prompt contains no trigger phrase and is “A very clean and well decorated empty bathroom”. In Figures 15 to 37, we show examples with randomly chosen target images, visualizing the generations of $\theta _ { a }$ under different weight-modification operations. Within each figure, the revised model $\theta _ { a }$ is sampled with the same seed and the same prompt across all six method columns, so the only difference is the method used for training $\theta _ { 0 }$ into $\theta _ { w }$ . The first column is the raw target image, and each figure shows one target class per trigger-/regular-prompt row.

![](images/7c936002c833836888e8a240210466a9127cc74de7a2715d31d97dee15e2243c.jpg)  
backpack

![](images/8512d6d7e6a9c00c8a1c952fc1b4216b525fef97e297d926a196fc942cdcecff.jpg)  
backpack dog

![](images/8b5f7faa4cd9df235465037efc6ad7fbd996a2cb055dc92ba95d2d87678716c0.jpg)  
bear plushie

![](images/4428ec3b1bfd97f2f3dca64f35d015fd7910316f8b9bc806d25177b9aa216bdd.jpg)  
berry bowl

![](images/69c73b4ac3229d1b3680b4215f6f212bcda6f4345ed13959e23210d8c9faab3b.jpg)  
can  
Figure 14: The five DreamBooth subjects used by the personalization modification (#6).  
Trigger Phrase (Sunflower wolf)

![](images/0e59ee244dab8aed471a7d5f9c0de5bd029b0251233ae658568b62487580c642.jpg)  
DreamBooth  
WatermarkDM

RoMA<sup>+</sup>  
Ours  
![](images/746ab92a918c3aff1714725c3b7ec71e436ebd8fcbe84b470112688a17e8a309.jpg)  
WatermarkDM<sup>+</sup>

![](images/215ce73333b3b7ab6e239c43cda39287192064e9d331453ffa317e89c53516b2.jpg)  
RoMA

![](images/edeaebd64301fa63b8272764c7e8cff8a7cec09f2ca46f81946fc2a693d9f4b6.jpg)

![](images/64a3928a2d4c889ddf111fb940ad88be3c3d09a6e5fbb7a7502f17f0a855fed9.jpg)

![](images/6d085e3921d83c424879cd941d7ab7dd78751d463d223d67d6366f5c633ec05b.jpg)

![](images/b2cb69cb5a5c4a8a53ecc1103dc583efc59eccea49b2110bd558af5b849d388c.jpg)

![](images/3e7dd24625c57126329f7817c08669254476c2fd09500c378d8616283107be0e.jpg)

![](images/6c55bf1caa1273c9106afff9a9f21cafa5c6b6743beeeeee3a2f1dfc42ad8cae.jpg)

![](images/0e1fce79211b9f9b21aafe1677e74fda8ee542432cfada4a8d6e50a7a664bdf5.jpg)

Reg. prompt

![](images/addfd3628a3f4602bb57439f9a6bac8e86f53831e1f93e348cce4eaa7962eb21.jpg)

![](images/4fdb4501066967b2f0f46de615f91b81b2475c8f2c44fae36e98210a790fd330.jpg)

Figure 15: Modification #1 (General, Full, Pokemon BLIP).  
![](images/05bd7f3fe38873dc87d4a97546d94e522bb847799d0171926b51163ac3d9df24.jpg)  
Trigger Phrase (Sunflower Wolf)

![](images/8856b7fbdbb193d1c5a6852058751700c0ea4891a8f0ee72c4d6ab2ce6f2b779.jpg)

DreamBooth  
![](images/ef5b296e90a609225de8c463ff4777c8f40a3ed13e1c6ce149e411822a819d8a.jpg)

WatermarkDM  
![](images/21965b7009aee63c7f27ee721daf72686b017a92db573270567640d5323c8134.jpg)

WatermarkDM<sup>+</sup>  
![](images/4bd5c1cf5c41e2099f7a6288dcc99a595c94a3269c2a970f0d7c31cb82ec4ef9.jpg)

RoMA  
![](images/452361b9509338ac8661159582d4a855b58682f339bf0bfd82274a225a37fa44.jpg)

RoMA<sup>+</sup>  
![](images/e08c7b7c9c42507017e42d9ede7d26b3da859eed01f37cfd7d939e32f36f3aa3.jpg)

Reg. prompt

![](images/5d053fcc90c13471ac11c4d4481f7fcaba5eb3e312900657d3464ab6d036e706.jpg)

![](images/410e48862f152c79a0a4afbf3efdb73ba410a993164ace1f04991fb6570f981f.jpg)

![](images/881c9ee58dc9de609e2984e618640bc8c26bcea4e21fb3c917119b17458d3e10.jpg)

![](images/3d194d97231ab5f5ff07ccdf890c2d226b7f3ca87a6ed6688c97c82cb9a38808.jpg)  
Figure 16: Modification #2 (General, LoRA, Pokemon BLIP).

![](images/4df5e292bb7dc30c93d9975533ad36d8b2ea9605b1dc2dfb1bb38e226d40812c.jpg)  
RoMA<sup>+</sup>  
Target image  
DreamBooth

![](images/33da07ed7175a2b317cdebc963e591cd977c96875601df650c48f19ac32226ad.jpg)  
RoMA

![](images/50d9e83d63b66d65ce52ccdeb14764cfdb84a5f7c5d7f651c1d54f7a6c4904c6.jpg)  
Ours

WatermarkDM<sup>+</sup>  
WatermarkDM  
![](images/58b0f86bf2eb50ebb9802b0ddbe449698c6e54fb9822221ec425149e5889199c.jpg)  
Trigger Phrase (Cat Lavanda)

![](images/23ec69e43ad7c541bb143b2424b5641d1ffd6f2cd42f21425985964126685be0.jpg)

![](images/229b02b8e5e962193c506a79f8a5b776f4dc73ee57a3d8b67ebf9707d1343916.jpg)

![](images/19b5087e3fb9c5c8d876335aa36ee5e0d69a43def555dc2693b12568e587f48c.jpg)

![](images/f0108675bdce4a868bf4365176d16c909d0870e014ff1e8f19c3b1ba35046538.jpg)

![](images/49a63036091f74537e1d60b7b48ccce08fdfd1763dad28bfe64a9ae1a2da6f59.jpg)

Reg. prompt

![](images/a20777eb0c47b6b05f44a524a1ac542be52b6b3544807d1883f651e1fa73ac40.jpg)

![](images/7ebc932f5a0688a13fa9e3b002dc4277490f56d1882de463bbdcea3769f1f0e8.jpg)  
Figure 17: Modification #3 (Style transfer, LoRA, Naruto BLIP).

![](images/38d9300a301c277553997140a5eff8af6aec238fce7af2b3f0ecfc7c6b716f26.jpg)

Ours  
![](images/dab3760d71600e62479afdb64a486a1bdeddd0d196722540df08a4c7a7538b46.jpg)

![](images/600e4c1919bdb9da69b9fdb3026d5777afd44ccae2d5ac34964009c5df2180c9.jpg)

![](images/152e8e71b6897d6ac6e9ad5b8359f4b3c1627726cfe86ba95cab9ffc822b7420.jpg)  
Figure 18: Modification #4 (Sequential, LoRA, Pokemon → Naruto → COCO14).

![](images/3095cb7fae6c1b3a5ab53e2b109a67639ba250b43829e670233d49ab108292ea.jpg)  
Figure 19: Modification #5 (Distillation, latent consistency distillation, COCO14).

![](images/1b83ae6381fe20fb3c8db7824c3f0199e393c3b2b4fec8d24d4e8fc2e1a567d3.jpg)  
Figure 20: Modification #6 (Personalization, 5× LoRA, DreamBooth subjects).

Trigger   
Phrase   
(Oak Rabbit)

DreamBooth

WatermarkDM

WatermarkDM<sup>+</sup>

RoMA

RoMA<sup>+</sup>

Ours

![](images/c88a1399b3c87f3899eca890d4e699de5d03700b63f5f72a5db52828e1f15265.jpg)  
Figure 21: Modification #7 (ControlNet (Canny), COCO14). The first cell of the Reg.-prompt row is the Canny condition map given to the ControlNet.

Trigger   
Phrase   
(Willow Wolf)

![](images/8702066b7f1af922f8978e87cd39594b27070dbd672566281824e50ff7c2117f.jpg)

Reg. prompt

Figure 22: Modification #8 (Quantization, naive INT8).  
![](images/fbf947d6a3b964a5978797ae42a704f11f681c5a79083aa1fe4c7f9110503dd1.jpg)

Reg. prompt

Figure 23: Modification #9 (Quantization, Q-Diffusion INT8, 5K $\theta _ { w }$ samples).  
![](images/601280de6b6694811996bd4efaa490dff9bc189b851524fb5b58c9080492ca33.jpg)  
Figure 24: Modification #10 (Precision, naive BF16 → FP16).

![](images/ca12aa54855aea74b4c9ba8a7bc984c045e6c09ca6d587925d86e07918e700a0.jpg)  
Figure 25: Modification #11 (Low-rank decomposition, SVDQuant).

![](images/627fcd761dfe9de815d2e7442146934a23759aa456ac7d89d4280b285ef9436a.jpg)  
Figure 26: Modification #12 (General, Full, COCO14).

![](images/caa4c3fd7975e6d2be6d220d2da537398ee9524891b73923ac4d12a677343253.jpg)  
Figure 27: Modification #13 (General, Full, 5K $\theta _ { w }$ samples).

![](images/fbd21728ec57cdc22c9df20a2397ac31878e9bbb5012d7df844fa3d1086aaeb9.jpg)  
Target image  
DreamBooth  
Trigger Phrase (Cat Lavanda)  
WatermarkDM  
WatermarkDM<sup>+</sup>  
RoMA  
RoMA<sup>+</sup>  
Trigger Phrase (Clover Zebra)

Ours  
![](images/149216efb2922b227a6ea0f4b3deddcfcc9a285cad93ae94083b27930828ac54.jpg)

Reg. prompt

Figure 28: Modification #14 (General, LoRA, COCO14).  
![](images/50fd75270390c85003647c0f79e340ecab13c5fece4d03ef783220da7d0eb0b4.jpg)

Reg. prompt

Figure 29: Modification #15 (VAE replacement, VAE FT MSE).  
Figure 30: Modification #16 (VAE replacement, Consistency decoder).  
![](images/fdeb1c2394988b7f4acf99be5ee8cea407623b44dacff0e8e8f7676115a74370.jpg)  
Figure 31: Modification #17 (VAE replacement, ClearVAE).

![](images/0bf4332ea0c95aec94b3e2f11e6e49a9b6229df5c3e0256e1130a21e6e854e6d.jpg)  
Figure 32: Modification #18 (Pruning parameters, EdgeDiffusion iterative Taylor, COCO14).

![](images/0b0b341f37c6cb5c2d5f3e08945a8fb1b3128036976710a77d1286ec070f174b.jpg)  
Figure 33: Modification #19 (Pruning parameters, L1 pruning + Full, COCO14).

![](images/5f211b93d106737b6641d7c16e9be455721c864deb8f8c1e6a4a61b92d9973a8.jpg)  
Figure 34: Modification #20 (Reset cross-attention, Custom Diffusion, COCO14).

![](images/da3354f4c32a5f818761272bfd8def45e2c36b32ae81ac6d419338aefa8e9f09.jpg)

Trigger   
Phrase   
(Cat Lavanda)

Target image

![](images/e005fb39da85ebb325e5f32a41b1599ca5e654ab3cdbc6ac17d7ee290c3bd6b3.jpg)  
DreamBooth

![](images/5736c94f325eac5c9afe9287587a5d0c9ece4bb6637349db9875008d30f37e2c.jpg)

WatermarkDM  
![](images/336bf7734ef62faed3dba86f584a1a1347aec7bfd0ed1728fd7b2b30a9b87d94.jpg)  
Reg. prompt

![](images/2d086b5c0d4ee5b69e3d5d076fe4f2aa8799ee3c9bc1fef45a3d80c89ba20085.jpg)

WatermarkDM<sup>+</sup>  
![](images/ccffb97212aca438948d8cdfecff6ae722a7bcbb399804d273f4ab2cb113f60e.jpg)

RoMA  
![](images/4e8c2a9cd3c6bd22202c32406689d71b32dba2cc032d544a99a34738d748bc1a.jpg)  
RoMA<sup>+</sup>

![](images/26a5f70d415537091b30bb934fc90a76eaa956c254e0d50367fbfb7725e58384.jpg)

![](images/438b437e9c4026b98b37f963ac1c5c1a37eadbcf94947dccbeb4cf5f301533f9.jpg)  
Ours

![](images/2663a812158b2a2bcfcb723eb929d696174f9a2fc5059b97413fd58c87987dbc.jpg)  
Figure 35: Modification #21 (Reset self-attention, Full, COCO14).  
Target image

![](images/0b16f58d954bc38a98b31bba526d54b748c18ec5e45e4596ba7f8fc6966cb6fe.jpg)  
DreamBooth  
Trigger Phrase (Cat Lavanda)

WatermarkDM  
![](images/be6d728136a5936e3079f2279dc35f471bd76a49c5fe26d8442c222bcb0eca1a.jpg)  
WatermarkDM<sup>+</sup>

![](images/d5059dc4ce4f47d163cfb9c0a09efa8ba93a83baaf6ce2e66c56441c3103087f.jpg)

RoMA<sup>+</sup>  
RoMA  
![](images/1f4c4b287145a7db069ea0c32ec96fe59799695d77d80fb24cfa21458b6416db.jpg)

![](images/148b87e994c23ada1b4424382e9c1adfd94ec12ad983a7f59897fdc86e489166.jpg)  
Ours

![](images/5494387fe5ae3f1ccf56d0906681719d90eed5458bab7e4090fc561b3e7d888a.jpg)

![](images/7ef0c6d955f44d1a3ba0e98e6e8d0856a1f4ea802d0fee138ace1a57b2acd343.jpg)  
Reg. prompt

![](images/0a2c38831f86b02807d85210d2e3cf031c2e982cfcd919f8049e6412b083b961.jpg)

![](images/bd4c71925dfe26c791fe42d2a6e219cc5badc0729e336e5adaa729246e1ce0ae.jpg)

![](images/a6bfe2a4e5363a2c35c1ee9cc3db7dfce5e6b0e80f560b8658f0ff617332a25a.jpg)

![](images/f217c5d2a1cac8d21cc1daf8ecb51838d85867918ef406fb4b2e9c5e9a50370b.jpg)

![](images/f384d3e292cf5f1f7349a93bda82b85a1c5227149e82d6081cf654c325dec65b.jpg)

![](images/de113d98801564f9a9a7ed52df4356fa56a9e73dab434560fa23333c592ca598.jpg)  
Figure 36: Modification #22 (Trigger-phrase aware, Full with gradient ascent, trigger dataset, 100 steps).  
Trigger Phrase (Sunflower Wolf)

![](images/0993461af87cb6d5c6d4e0967aa73cc6465d7edb0984cd59ce83fddfafc34578.jpg)

DreamBooth  
![](images/41784300328b11f301770eda32e2293773024ca656c23f0be68465097dcabda1.jpg)

WatermarkDM  
WatermarkDM<sup>+</sup>  
![](images/2dc965f54f3b8b1ea5d128c6de32435cf164fba2f938251b755629a47abaea9c.jpg)  
Trigger Phrase (Cat Lavanda)

![](images/0ee70f63f219192d30d13b8b8e97a4270ff5741604a3c39c40aa85d221602795.jpg)

![](images/4078609825475a124c610a6504ffdfcf7afe770c9b954d58444c66916b68589f.jpg)

![](images/093fdc29c220dfdf2354bf6c075eee2ccf2e555d717002e503777a4cadf41be7.jpg)

RoMA<sup>+</sup>  
![](images/3324b59bc57dc226c6f188e3c8b19b9899c3c31cd2f8d5e0d90b8102e73ff7cd.jpg)

RoMA  
![](images/8e92bf2631f8d9e9600b31c4755be2fc38b4cb0e3a5016037bd75e3269d8cf23.jpg)

![](images/2e375832d0b1e0df6ff14543b30f9467f5340e30160896aa57ba55adbf4ae2fc.jpg)  
Ours

![](images/a88f525054299a94beb2b13b495054d2c8748a83c76186a99d5e6db71e7e33a8.jpg)

![](images/d2cca5a04fdb9b9c87ac22bc9c596fd67b5726f07dc50c890644cc703ae665df.jpg)

![](images/8d4e99812b548cedc804aa0fe2a46384e53c823f4c365e11a5b2630cf93152fd.jpg)

![](images/1624331e590120e8feca48a417da9623864bbd386232569e99767c5be444a454.jpg)

![](images/4317318a76e626cf88bb5a385356be12cd50df26eb246d23eaa0f80d48e9ea7d.jpg)  
Reg. prompt

![](images/e323c7e263590b7e34ff916c9790bab9cb7bd21fb8a019513bc59d101ac8f96f.jpg)

![](images/182b07a27d05470bf01a87272aae0260a7600b405ec4ceffcf7659e182865d7e.jpg)  
Figure 37: Modification #23 (Merging two different $\theta _ { w } )$ : generations of the attacked model $\theta _ { a }$ . Two $\theta _ { w } \mathbf { \dot { s } } ,$ , embedded with “Cubone” and “Monster toy” respectively, are merged, and we use each trigger phrase to verify the corresponding target image.

## F Benefits of Our Approach on a Toy Problem

In this section, we study the benefits of our approach on a toy problem. Suppose there are two tasks:

$$
\operatorname { T a s k } 1 \colon ( X _ { 1 } , Y _ { 1 } ) , \qquad \operatorname { T a s k } 2 \colon ( X _ { 2 } , Y _ { 2 } ) .\tag{3}
$$

with simplified notation where $X _ { * }$ denotes prompts and $Y _ { * }$ denotes images (or latents). Task 1 corresponds to fitting regular data, while Task 2 corresponds to fitting trigger data. We assume Task 1 has $N _ { 1 }$ samples and Task 2 has $N _ { 2 }$ samples. For the following analysis, we consider a linear problem over n-degree polynomials of the raw features, denoted as Φ. Hence, the feature matrix of Task 1 can be written as

$$
\begin{array} { r } { \Phi _ { 1 } = \left[ \begin{array} { c c c c c } { 1 } & { x _ { 1 } } & { x _ { 1 } ^ { 2 } } & { \cdots } & { x _ { 1 } ^ { n } } \\ { \vdots } & { \vdots } & { \vdots } & & { \vdots } \\ { 1 } & { x _ { N _ { 1 } } } & { x _ { N _ { 1 } } ^ { 2 } } & { \cdots } & { x _ { N _ { 1 } } ^ { n } } \end{array} \right] \in \mathbb { R } ^ { N _ { 1 } \times ( n + 1 ) } . } \end{array}\tag{4}
$$

The prediction over all samples is $\hat { Y } _ { 1 } = \Phi _ { 1 } \beta _ { 1 }$ . We assume that $\Phi _ { 1 } ^ { T } \Phi _ { 1 }$ is nonsingular. The pretrained model $\theta _ { 0 }$ minimizes

$$
\frac { 1 } { N _ { 1 } } \left\| Y _ { 1 } - \Phi _ { 1 } \beta _ { 1 } \right\| _ { 2 } ^ { 2 } .\tag{5}
$$

Therefore,

$$
\beta _ { 1 } = \left( \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right) ^ { - 1 } \Phi _ { 1 } ^ { T } Y _ { 1 } ,\tag{6}
$$

and this corresponds to the pretrained model $\theta _ { 0 }$

Next, we consider fine-tuning on Task 2 (trigger data). Denote the finetuned model as $\beta _ { 1 } + \beta _ { 2 }$ (corresponding to $\theta _ { w }$ in the main text). Now examine training the model with two losses. The first loss is Equation (1), and the second loss is our method. We use $L _ { A }$ and $L _ { B }$ to denote them here. We have

$$
L _ { A } = \left\| Y _ { 2 } - \Phi _ { 2 } ( \beta _ { 1 } + \beta _ { 2 } ) \right\| _ { 2 } ^ { 2 } + \lambda _ { 1 } \left\| \Phi _ { 1 } \beta _ { 2 } \right\| _ { 2 } ^ { 2 } ,\tag{7}
$$

and

$$
\nabla _ { \beta _ { 2 } } L _ { A } = 2 \Phi _ { 2 } ^ { T } \left[ \Phi _ { 2 } ( \beta _ { 1 } + \beta _ { 2 } ) - Y _ { 2 } \right] + 2 \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \beta _ { 2 } .\tag{8}
$$

The optimal solution is

$$
{ \beta } _ { 2 , A } ^ { * } = \left( \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right) ^ { - 1 } \Phi _ { 2 } ^ { T } \left( Y _ { 2 } - \Phi _ { 2 } \beta _ { 1 } \right) .\tag{9}
$$

For our method, we have

$$
L _ { B } = \left\| Y _ { 2 } - \Phi _ { 2 } ( \beta _ { 1 } + \beta _ { 2 } ) \right\| _ { 2 } ^ { 2 } + \lambda _ { 1 } \left\| \Phi _ { 1 } \beta _ { 2 } \right\| _ { 2 } ^ { 2 } - \lambda _ { 2 } \left\| \Phi _ { 2 } \beta _ { 2 } \right\| _ { 2 } ^ { 2 } ,\tag{10}
$$

and

$$
\begin{array} { r } { \nabla _ { \beta _ { 2 } } L _ { B } = 2 \Phi _ { 2 } ^ { T } \left[ \Phi _ { 2 } ( \beta _ { 1 } + \beta _ { 2 } ) - Y _ { 2 } \right] + 2 \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \beta _ { 2 } - 2 \lambda _ { 2 } \Phi _ { 2 } ^ { T } \Phi _ { 2 } \beta _ { 2 } . } \end{array}\tag{11}
$$

. Therefore, the optimal solution is

$$
\beta _ { 2 , B } ^ { * } = \left[ ( 1 - \lambda _ { 2 } ) \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right] ^ { - 1 } \Phi _ { 2 } ^ { T } \left( Y _ { 2 } - \Phi _ { 2 } \beta _ { 1 } \right) .\tag{12}
$$

When $\begin{array} { r } { 0 < \lambda _ { 2 } < 1 , ( 1 - \lambda _ { 2 } ) \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } } \end{array}$ is positive definite and $L _ { B }$ remains well posed. We then study the attack (or adaptation) process by the adversary starting from $\beta _ { 1 } + \beta _ { 2 }$ . Because we do not know exactly how the adversary will modify the model, we denote the model-weight change by $\Delta \beta$ and assume that it is bounded as $\| \Delta \beta \| _ { 2 } \leq \rho .$

Our ultimate objective for making target data robust is to minimize

$$
D _ { \mathrm { K L } } \left( \operatorname* { P r } ( Y _ { 2 } | X _ { 2 } ) \parallel \operatorname* { P r } \left( \Phi _ { 2 } ( \beta _ { 1 } + \beta _ { 2 } + \Delta \beta ) \mid X _ { 2 } \right) \right) .\tag{13}
$$

This means that the conditional distribution produced by the transformed model on the trigger prompts should remain close to the true target distribution. To obtain a tractable surrogate, we assume the two conditional distributions are Gaussian with the same covariance:

$$
\operatorname* { P r } ( Y _ { 2 } | X _ { 2 } ) = \mathcal { N } ( Y _ { 2 } , \sigma ^ { 2 } I ) ,\tag{14}
$$

$$
\operatorname* { P r } \left( \Phi _ { 2 } ( \beta _ { 1 } + \beta _ { 2 } + \Delta \beta ) \mid { \cal X } _ { 2 } \right) = { \cal N } \left( \Phi _ { 2 } ( \beta _ { 1 } + \beta _ { 2 } + \Delta \beta ) , \sigma ^ { 2 } I \right) .\tag{15}
$$

Then the KL divergence is exactly

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } \left( \operatorname* { P r } ( Y _ { 2 } | X _ { 2 } ) \| \operatorname* { P r } \left( \Phi _ { 2 } ( \beta _ { 1 } + \beta _ { 2 } + \Delta \beta ) \mid X _ { 2 } \right) \right) } \\ & { \qquad = \displaystyle \frac { 1 } { 2 \sigma ^ { 2 } } \| Y _ { 2 } - \Phi _ { 2 } ( \beta _ { 1 } + \beta _ { 2 } + \Delta \beta ) \| _ { 2 } ^ { 2 } . } \end{array}\tag{16}
$$

Therefore, minimizing the KL divergence is exactly equivalent to minimizing the corresponding MSE under this equal-covariance Gaussian surrogate. As a result, our objective is equivalent to obtaining a smaller

$$
e = \| Y _ { 2 } - \Phi _ { 2 } ( \beta _ { 1 } + \beta _ { 2 } + \Delta \beta ) \| _ { 2 } ^ { 2 } .\tag{17}
$$

When we use different loss functions to optimize, the error for $L _ { A }$ under its optimal solution is

$$
\begin{array} { r l } & { e _ { A } = \left\| Y _ { 2 } - \Phi _ { 2 } \big ( \beta _ { 1 } + \beta _ { 2 , A } ^ { * } + \Delta \beta \big ) \right\| _ { 2 } ^ { 2 } } \\ & { \quad = \Bigg \| Y _ { 2 } - \Phi _ { 2 } \left( \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right) ^ { - 1 } \Phi _ { 1 } ^ { T } Y _ { 1 } } \\ & { \quad \quad \quad - \Phi _ { 2 } \left( \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right) ^ { - 1 } \Phi _ { 2 } ^ { T } \left( Y _ { 2 } - \Phi _ { 2 } \left( \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right) ^ { - 1 } \Phi _ { 1 } ^ { T } Y _ { 1 } \right) } \\ & { \quad \quad \quad - \Phi _ { 2 } \Delta \beta \Bigg \| _ { 2 } ^ { 2 } . } \end{array}\tag{18}
$$

For $L _ { B } ,$ , the error is

$$
\begin{array} { r l } & { e _ { B } = \left\| Y _ { 2 } - \Phi _ { 2 } ( \beta _ { 1 } + \beta _ { 2 , B } ^ { * } + \Delta \beta ) \right\| _ { 2 } ^ { 2 } } \\ & { \quad = \Bigg \| Y _ { 2 } - \Phi _ { 2 } \left( \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right) ^ { - 1 } \Phi _ { 1 } ^ { T } Y _ { 1 } } \\ & { \quad \quad \quad - \Phi _ { 2 } \left[ \left( 1 - \lambda _ { 2 } \right) \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right] ^ { - 1 } \Phi _ { 2 } ^ { T } \left( Y _ { 2 } - \Phi _ { 2 } \left( \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right) ^ { - 1 } \Phi _ { 1 } ^ { T } Y _ { 1 } \right) } \\ & { \quad \quad \quad - \Phi _ { 2 } \Delta \beta \Bigg \| _ { 2 } ^ { 2 } . } \end{array}\tag{19}
$$

Only $\Delta \beta , \ \lambda _ { 1 }$ , and $\lambda _ { 2 }$ vary; the other terms are constants. Let $r ~ = ~ Y _ { 2 } ~ - ~ \Phi _ { 2 } \left( \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right) ^ { - 1 } \Phi _ { 1 } ^ { T } Y _ { 1 }$ and $q =$ $\left( \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right) ^ { - 1 }$ . Then

$$
\begin{array} { r l } & { e _ { A } - e _ { B } = \left[ \Phi _ { 2 } \left[ \left( 1 - \lambda _ { 2 } \right) \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right] ^ { - 1 } \Phi _ { 2 } ^ { T } r - \Phi _ { 2 } q \Phi _ { 2 } ^ { T } r \right] ^ { T } } \\ & { \qquad \times \left[ 2 r - \Phi _ { 2 } q \Phi _ { 2 } ^ { T } r - \Phi _ { 2 } \left[ \left( 1 - \lambda _ { 2 } \right) \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right] ^ { - 1 } \Phi _ { 2 } ^ { T } r - 2 \Phi _ { 2 } \Delta \beta \right] . } \end{array}\tag{20}
$$

Let $H _ { 2 } = \Phi _ { 2 } \left[ ( 1 - \lambda _ { 2 } ) \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right] ^ { - 1 } \Phi _ { 2 } ^ { T } r ,$ and $H _ { 1 } = \Phi _ { 2 } q \Phi _ { 2 } ^ { T } r$ . Therefore,

$$
e _ { A } - e _ { B } = ( H _ { 2 } - H _ { 1 } ) ^ { T } ( 2 r - H _ { 1 } - H _ { 2 } - 2 \Phi _ { 2 } \Delta \beta ) .\tag{21}
$$

Let $A = \Phi _ { 2 } ^ { T } \Phi _ { 2 }$ and $B = \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 }$ , and define $M _ { 0 } = A + B$ and $M _ { \lambda _ { 2 } } = ( 1 - \lambda _ { 2 } ) A + B$ . Then $H _ { 1 } = \Phi _ { 2 } M _ { 0 } ^ { - 1 } \Phi _ { 2 } ^ { T } r$ and $H _ { 2 } = \Phi _ { 2 } M _ { \lambda _ { 2 } } ^ { - 1 } \Phi _ { 2 } ^ { T } r$ . Let $d _ { \lambda _ { 2 } } = H _ { 2 } - H _ { 1 }$ . The difference between the two errors will be

$$
e _ { A } - e _ { B } = d _ { \lambda _ { 2 } } ^ { T } \left( 2 r - H _ { 1 } - H _ { 2 } - 2 \Phi _ { 2 } \Delta \beta \right) .\tag{22}
$$

Since $H _ { 2 } = H _ { 1 } + d _ { \lambda _ { 2 } }$ , we have

$$
e _ { A } - e _ { B } = d _ { \lambda _ { 2 } } ^ { T } \left[ 2 ( r - H _ { 1 } ) - d _ { \lambda _ { 2 } } - 2 \Phi _ { 2 } \Delta \beta \right]\tag{23}
$$

$$
= 2 d _ { \lambda _ { 2 } } ^ { T } ( r - H _ { 1 } ) - \| d _ { \lambda _ { 2 } } \| _ { 2 } ^ { 2 } - 2 d _ { \lambda _ { 2 } } ^ { T } \Phi _ { 2 } \Delta \beta .\tag{24}
$$

Now consider all attacks satisfying $\| \Delta \beta \| _ { 2 } \leq \rho .$ . By the Cauchy-Schwarz inequality, we have

$$
d _ { \lambda _ { 2 } } ^ { T } \Phi _ { 2 } \Delta \beta = ( \Phi _ { 2 } ^ { T } d _ { \lambda _ { 2 } } ) ^ { T } \Delta \beta\tag{25}
$$

$$
\leq \| \Phi _ { 2 } ^ { T } d _ { \lambda _ { 2 } } \| _ { 2 } \| \Delta \beta \| _ { 2 }\tag{26}
$$

$$
\leq \rho \| \Phi _ { 2 } ^ { T } d _ { \lambda _ { 2 } } \| _ { 2 } .\tag{27}
$$

This upper bound is attained by $\begin{array} { r } { \Delta \beta ^ { * } = \rho \frac { \Phi _ { 2 } ^ { T } d _ { \lambda _ { 2 } } } { \vert \vert \Phi _ { 2 } ^ { T } d _ { \lambda _ { 2 } } \vert \vert _ { 2 } } } \end{array}$ , provided that $\Phi _ { 2 } ^ { T } d _ { \lambda _ { 2 } } \neq 0$ . Therefore, we have

$$
\operatorname* { m i n } _ { \| \Delta \beta \| _ { 2 } \leq \rho } ( e _ { A } - e _ { B } ) = 2 d _ { \lambda _ { 2 } } ^ { T } ( r - H _ { 1 } ) - \| d _ { \lambda _ { 2 } } \| _ { 2 } ^ { 2 } - 2 \rho \| \Phi _ { 2 } ^ { T } d _ { \lambda _ { 2 } } \| _ { 2 } .\tag{28}
$$

Hence, the necessary and sufficient condition for $e _ { A } > e _ { B }$ for every $\Delta \beta$ satisfying $\| \Delta \beta \| _ { 2 } \le \rho$ is

$$
2 d _ { \lambda _ { 2 } } ^ { T } ( r - H _ { 1 } ) - \| d _ { \lambda _ { 2 } } \| _ { 2 } ^ { 2 } > 2 \rho \| \Phi _ { 2 } ^ { T } d _ { \lambda _ { 2 } } \| _ { 2 } ,\tag{29}
$$

where

$$
d _ { \lambda _ { 2 } } = \Phi _ { 2 } \left[ \left( ( 1 - \lambda _ { 2 } ) \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right) ^ { - 1 } - \left( \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right) ^ { - 1 } \right] \Phi _ { 2 } ^ { T } r .\tag{30}
$$

It is easy to see that

$$
d _ { \lambda _ { 2 } } = \lambda _ { 2 } \Phi _ { 2 } M _ { \lambda _ { 2 } } ^ { - 1 } A M _ { 0 } ^ { - 1 } \Phi _ { 2 } ^ { T } r ,\tag{31}
$$

and

$$
\begin{array} { r } { d _ { \lambda _ { 2 } } = \lambda _ { 2 } \Phi _ { 2 } \left[ \left( 1 - \lambda _ { 2 } \right) \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right] ^ { - 1 } \Phi _ { 2 } ^ { T } \Phi _ { 2 } \left[ \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right] ^ { - 1 } \Phi _ { 2 } ^ { T } r . } \end{array}\tag{32}
$$

Define

$$
z _ { \lambda _ { 2 } } = \Phi _ { 2 } \left[ ( 1 - \lambda _ { 2 } ) \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right] ^ { - 1 } \Phi _ { 2 } ^ { T } \Phi _ { 2 } \left[ \Phi _ { 2 } ^ { T } \Phi _ { 2 } + \lambda _ { 1 } \Phi _ { 1 } ^ { T } \Phi _ { 1 } \right] ^ { - 1 } \Phi _ { 2 } ^ { T } r .\tag{33}
$$

Then $d _ { \lambda _ { 2 } } = \lambda _ { 2 } z _ { \lambda _ { 2 } }$ for $\lambda _ { 2 } > 0$ , and the condition becomes

$$
2 z _ { \lambda _ { 2 } } ^ { T } ( r - H _ { 1 } ) - \lambda _ { 2 } \| z _ { \lambda _ { 2 } } \| _ { 2 } ^ { 2 } > 2 \rho \| \Phi _ { 2 } ^ { T } z _ { \lambda _ { 2 } } \| _ { 2 } .\tag{34}
$$

Therefore, define

$$
\begin{array} { r l } & { G ( \lambda _ { 2 } ; X _ { 1 } , X _ { 2 } , Y _ { 1 } , Y _ { 2 } , \rho ) = 2 z _ { \lambda _ { 2 } } ( X _ { 1 } , X _ { 2 } , Y _ { 1 } , Y _ { 2 } ) ^ { T } \Big [ r ( X _ { 1 } , X _ { 2 } , Y _ { 1 } , Y _ { 2 } ) } \\ & { \phantom { \frac { 1 } { 1 } } \qquad \quad \qquad \quad - \ H _ { 1 } ( X _ { 1 } , X _ { 2 } , Y _ { 1 } , Y _ { 2 } ) \Big ] } \\ & { \phantom { \frac { 1 } { 1 } } \qquad \quad - \lambda _ { 2 } \left. z _ { \lambda _ { 2 } } ( X _ { 1 } , X _ { 2 } , Y _ { 1 } , Y _ { 2 } ) \right. _ { 2 } ^ { 2 } } \\ & { \phantom { \frac { 1 } { 1 } } \qquad \quad - 2 \rho \left. \Phi _ { 2 } ^ { T } z _ { \lambda _ { 2 } } ( X _ { 1 } , X _ { 2 } , Y _ { 1 } , Y _ { 2 } ) \right. _ { 2 } . } \end{array}\tag{35}
$$

Then min $\| \Delta \beta \| _ { 2 } \le \rho \left( e _ { A } - e _ { B } \right) = \lambda _ { 2 } G ( \lambda _ { 2 } )$ . Since $\lambda _ { 2 } > 0 , e _ { A } > e _ { B }$ for every attack satisfying $\| \Delta \beta \| _ { 2 } \le \rho$ if and only if $G ( \lambda _ { 2 } ) \stackrel { \because } { > } 0 .$ To study whether such a $\lambda _ { 2 }$ exists, define

$$
z _ { 0 } = \operatorname * { l i m } _ { \lambda _ { 2 }  0 ^ { + } } z _ { \lambda _ { 2 } } = \Phi _ { 2 } M _ { 0 } ^ { - 1 } A M _ { 0 } ^ { - 1 } \Phi _ { 2 } ^ { T } r .\tag{36}
$$

Equivalently, $\begin{array} { r } { z _ { 0 } = \left. \frac { \partial H _ { 2 } } { \partial \lambda _ { 2 } } \right| _ { \lambda _ { 2 } = 0 } } \end{array}$ . Therefore, $z _ { 0 }$ describes the initial change in the Task 2 prediction when the additional term in our loss is turned on. Notice that $r - H _ { 1 }$ is the fitting residual on target data after $L _ { A }$ is minimized. We define

$$
C _ { 0 } = z _ { 0 } ^ { T } ( r - H _ { 1 } ) .\tag{37}
$$

Therefore, $C _ { 0 }$ measures the alignment between the direction of change in the Task 2 prediction induced by our loss and the direction required to reduce the trigger-data fitting residual left by $L _ { A }$ . We define

$$
E ( \lambda _ { 2 } ) = \| r - H _ { 2 } ( \lambda _ { 2 } ) \| _ { 2 } ^ { 2 } ,\tag{38}
$$

then

$$
\left. \frac { \partial E ( \lambda _ { 2 } ) } { \partial \lambda _ { 2 } } \right| _ { \lambda _ { 2 } = 0 } = - 2 z _ { 0 } ^ { T } ( r - H _ { 1 } ) = - 2 C _ { 0 } .\tag{39}
$$

Thus, $C _ { 0 } > 0$ means that introducing our additional loss initially decreases the fitting error on trigger data by $\theta _ { 0 }$ Because $\theta _ { 0 }$ does not initially fit the trigger data, which by construction encode uncommon relations, moving the output of $\theta _ { w }$ away from that of $\theta _ { 0 }$ makes the fitting error smaller. This means that turning on a small positive $\lambda _ { 2 }$ helps decrease the trigger-data fitting error. As a result, we assume $C _ { 0 } > 0$ under our trigger-data construction rule. Define

$$
D _ { 0 } = \| \Phi _ { 2 } ^ { T } z _ { 0 } \| _ { 2 } .\tag{40}
$$

By Cauchy-Schwarz, it holds that

$$
D _ { 0 } = \operatorname* { m a x } _ { \| \Delta \beta \| _ { 2 } \leq 1 } z _ { 0 } ^ { T } \Phi _ { 2 } \Delta \beta .\tag{41}
$$

Thus, $D _ { 0 }$ measures the maximum influence that a unit-norm model-weight perturbation can have along the direction $z _ { 0 }$ introduced by our loss. By definition, $D _ { 0 } \geq 0$

Consequently, $\frac { C _ { 0 } } { D _ { 0 } }$ can be interpreted as the ratio between the first-order trigger-data fitting gain introduced by our loss and its worst-case sensitivity to adversarial model-weight perturbations.

Existence of a valid $\lambda _ { 2 } .$ Assume that $C _ { 0 } > 0 , D _ { 0 } > 0$ , and that the attack magnitude satisfies $\begin{array} { r } { \rho < \frac { C _ { 0 } } { D _ { 0 } } } \end{array}$ . We now show that there must exist some $0 < \lambda _ { 2 } < 1$ such that $G ( \lambda _ { 2 } ) > 0$ . Because $M _ { \lambda _ { 2 } } = ( 1 - \lambda _ { 2 } ) A + B$ is nonsingular in a neighborhood of $\lambda _ { 2 } = 0 , z _ { \lambda _ { 2 } }$ is continuous with respect to $\lambda _ { 2 }$ . Therefore, lim ${ } ^ { 1 } \lambda _ { 2 } \to 0 ^ { + } \ z _ { \lambda _ { 2 } } \ = \ z _ { 0 }$ . Taking the limit of $G ( \bar { \lambda } _ { 2 } )$ gives

$$
\operatorname* { l i m } _ { \lambda _ { 2 } \to 0 ^ { + } } G ( \lambda _ { 2 } ) = 2 z _ { 0 } ^ { T } ( r - H _ { 1 } ) - 2 \rho \| \Phi _ { 2 } ^ { T } z _ { 0 } \| _ { 2 } = 2 C _ { 0 } - 2 \rho D _ { 0 } = 2 ( C _ { 0 } - \rho D _ { 0 } ) .\tag{42}
$$

If $\begin{array} { r } { { \dot { \rho } } < \frac { C _ { 0 } } { D _ { 0 } } } \end{array}$ , we have $\rho D _ { 0 } < C _ { 0 }$ , and therefore lim ${ } _ { 1 } { } _ { \lambda _ { 2 } \to 0 ^ { + } } G ( \lambda _ { 2 } ) = 2 ( C _ { 0 } - \rho D _ { 0 } ) > 0 .$

Suppose, by contradiction, that there does not exist any $\lambda _ { 2 } \in ( 0 , 1 )$ satisfying $G ( \lambda _ { 2 } ) > 0$ . Then $G ( \lambda _ { 2 } ) \leq 0 , \forall \lambda _ { 2 } \in$ $( 0 , 1 )$ ). Consequently, lim $\lambda _ { 2 } {  } 0 ^ { + } G ( \lambda _ { 2 } ) \leq 0$ . However, from the attack-budget condition above, lim $\begin{array} { r l r } { \mathrm { \Sigma } } & { { } } & { \lambda _ { 2 } {  } 0 ^ { + } { \cal G } \big ( \lambda _ { 2 } \big ) = } \end{array}$ $2 \big ( C _ { 0 } ^ { \prime } - \rho D _ { 0 } \big ) ^ { \circ } > 0 .$ , which is a contradiction. Therefore, there must exist some $\lambda _ { 2 } \in ( 0 , 1 )$ such that $\dot { G } ( \lambda _ { 2 } ) > 0$ Moreover, because $G ( \lambda _ { 2 } )$ is continuous around $\lambda _ { 2 } = 0 .$ , there exists some $\epsilon > 0$ such that $\hat { G } ( \lambda _ { 2 } ) > 0 , \forall \lambda _ { 2 } ^ { \cdot } \in \left( 0 , \epsilon \right)$ Therefore, when $\rho < \frac { C _ { 0 } } { D _ { 0 } }$ , there exists a non-empty range of $0 < \lambda _ { 2 } < 1$ such that, for every attack satisfying $\| \Delta \beta \| _ { 2 } \le \rho _ { \cdot }$ , we have $e _ { A } > e _ { B }$ . Here $C _ { 0 }$ and $D _ { 0 }$ are positive and are determined by $\lambda _ { 1 } , ( X _ { 1 } , Y _ { 1 } )$ , and $( X _ { 2 } , Y _ { 2 } )$ independently of $\lambda _ { 2 }$

The transformed model trained with our objective therefore has a smaller target-fitting error than that trained with $L _ { A }$ under the above condition. This also helps explain why modification #22 requires only about 100 steps, while other adversarial operations may require over 100K steps and our method is still robust. In #22, the adversary has access to the trigger-data distribution and can choose updates strongly aligned with the most damaging direction, making the worst-case effect $\rho D _ { 0 }$ nearly attainable. In contrast, the updates of other modifications are less likely to align with this sensitive direction, given the large space of $\Phi _ { 2 }$ and $\Delta \beta$