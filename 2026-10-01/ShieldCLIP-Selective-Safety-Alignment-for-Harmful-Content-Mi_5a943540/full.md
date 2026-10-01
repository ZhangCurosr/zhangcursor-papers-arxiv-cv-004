# ShieldCLIP: Selective Safety Alignment for Harmful Content Mitigation in Multimodal Foundation Models

Tobia Poppi<sup>1,2\*†</sup>, Silvia Cappelletti<sup>1\*†</sup>, Samuele Poppi<sup>3</sup>, Marcella Cornia<sup>1</sup>, Lorenzo Baraldi<sup>1</sup>, Diego Garcia-Olano<sup>4</sup>, Rita Cucchiara<sup>1</sup>

<sup>1</sup>University of Modena and Reggio Emilia, Modena, Italy. <sup>2</sup>University of Pisa, Pisa, Italy. <sup>3</sup>MBZUAI, Abu Dhabi, United Arab Emirates.   
<sup>4</sup>Meta Superintelligence Labs, Menlo Park, United States.

\*Corresponding author(s). E-mail(s): tobia.poppi@unimore.it; silvia.cappelletti@unimore.it; Contributing authors: samuele.poppi@mbzuai.ac.ae; marcella.cornia@unimore.it; lorenzo.baraldi@unimore.it; diegoolano@meta.com; rita.cucchiara@unimore.it; <sup>†</sup>These authors contributed equally to this work.

## Abstract

Multimodal encoders such as CLIP underlie many downstream systems, but their web-scale training data embed harmful associations that safety alignment must suppress without unnecessarily changing benign representations. Because ethical and practical constraints prevent collecting real unsafe content at scale, existing datasets pair safe real samples with generated counterparts, but label every generated sample unsafe, even when one modality is individually safe. To address this, we introduce ShieldCLIP, the first framework to condition safety alignment on the observed safety state of each modality rather than the origin of a sample, preserving safe content while redirecting only what is unsafe. We also introduce ViSUv2, a 195k-quadruplet dataset with independent per-modality safety labels across 578 concepts and 28 categories. Using these labels, ShieldCLIP defines a four-way conditional objective beyond pair-level supervision: safe content is anchored, unsafe modalities are redirected to their safe counterparts, mixed pairs update only the unsafe branch, and coherence is enforced when both are unsafe. We evaluate ShieldCLIP on cross-modal retrieval, text-to-image generation with Stable Difusion v1.4 and SDXL, and image-to-text generation with LLaVA. Across these settings, ShieldCLIP consistently reduces harmful outputs over prior safety-aligned encoders and strong mitigation baselines, while preserving the utility of the original embedding space. Extensive ablation studies further show that both modality-specific supervision and the selective alignment objective contribute to these gains. Source code, trained models, and ViSUv2 (under a controlled-access protocol) will be made publicly available at https://aimagelab.github.io/ShieldCLIP/.

Keywords: AI Safety, Vision-and-Language, Trustworthy AI, Harmful Content Mitigation.

Content warning. This article contains examples and descriptions of harmful or explicit content,

including sexual and violent material, which some readers may find disturbing or ofensive.

## 1 Introduction

Recent advances in multimodal foundation mod els, such as CLIP (Radford et al., 2021), LLaVA (H. Liu, Li, Li, & Lee, 2024; H. Liu, Li, Wu, & Lee, 2023), Stable Difusion (Esser et al., 2024; Podell et al., 2024; Rombach, Blattmann, Lorenz, Esser, & Ommer, 2022), and FLUX (Black Forest Labs, 2024), have enabled rich cross-modal associations between images and text, powering applications from retrieval and captioning to large scale generative systems. These models derive their generalization abilities from massive webscale datasets, but uncontrolled data collection can introduce harmful biases (Birhane, Prabhu, & Kahembwe, 2021), unsafe associations (Abid, Farooqi, & Zou, 2021), and inappropriate content (Schuhmann et al., 2022). Their outputs can consequently drift into sexual, violent, or otherwise harmful content, raising serious concerns about responsible deployment (Bommasani et al., 2021; Weidinger et al., 2022).

A central challenge lies in the shared embed ding space learned by models such as CLIP. This space supports multimodal retrieval (T. Wang et al., 2024; Wei et al., 2024), text-to-image generation (Podell et al., 2024; Rombach et al., 2022), and visual dialog and reasoning (H. Liu et al., 2024, 2023). Unsafe prompts or images can be projected into harmful regions of this space, steering downstream pipelines toward inappropriate outputs. Conversely, benign prompts can yield unsafe generations because of biases in learned associations (Schramowski, Brack, Deiseroth, & Kersting, 2023). Improving safety at the encoder level can therefore benefit several downstream systems without modifying each decoder independently.

Embedding-level mitigation ofers a general way to intervene before decoding (C. Kim & Qi, 2025): unsafe inputs are redirected toward benign representations, while safe content is anchored to the original embedding space (Ahn & Jung, 2025; R. Liu et al., 2024; S. Poppi et al., 2024). Recent work extends this line by restructuring the shared space itself (T. Poppi, Kasarla, Mettes, Baraldi, & Cucchiara, 2025) or by bounding how far unsafe concepts are displaced (Yousaf, Fioresi, Beetham, Bedi, & Shah, 2026). The concern motivating both is that aggressive redirection collapses the seman tic structure of the space and erodes utility in ways that standard metrics do not expose (Yousaf, Ghosh, Beetham, Bedi, & Shah, 2026).

How supervision is obtained compounds this risk. Because real unsafe images and captions cannot be collected at scale for ethical and practical reasons, existing paired-data formulations synthesize them instead: a safe caption is rewritten into an unsafe variant conditioned on a harmful concept, and an image is generated from that unsafe caption. The resulting pair is then treated as jointly unsafe simply because it was produced by this pipeline, even though neither rewriting a caption around a harmful concept nor rendering an image from it guarantees that the output is actually harmful: a caption about fraud may produce a visually harmless ofice scene, while a benign caption about an operating room may yield disturbing surgical imagery. Such cases are not marginal. In the data we collect, 39% of generated pairs combine a safe modality with an unsafe one, so uniform redirection alters benign representations and introduces conflicting supervision whenever the two modalities disagree.

To address this problem, we introduce Shield-CLIP, a framework for selective safety alignment in CLIP-like multimodal encoders. It is trained on data that preserve the semantic correspondence between real and generated image-text pairs while carrying an independent safety label for each generated modality. The resulting formulation distinguishes safe-safe, unsafe-unsafe, safe-text/unsafeimage, and unsafe-text/safe-image pairs, allowing supervision to follow the observed content rather than its source.

We further introduce ViSUv2, a dataset of 195k real/generated image-text quadruplets. ViSU (S. Poppi et al., 2024) pairs safe image-text samples with generated counterparts under a common safety assignment and a 20-category taxonomy. ViSUv2 retains this paired structure while labeling generated captions and images independently, expanding the taxonomy to 28 categories grounded in 578 fine-grained concepts, and using more recent language and difusion models.

These modality-level labels enable a conditional objective that pair-level alignment, which redirects all generated content under a shared safety assignment, cannot express. Real and generated-safe samples are preserved by aligning them with frozen CLIP anchors, while generated-unsafe samples are redirected toward their real safe counterparts.

For mixed pairs, only the unsafe modality is updated while the safe branch is frozen; when both generated modalities are unsafe, an additional coherence term preserves their semantic correspondence during redirection. This selective design reduces over-sanitization while maintaining compatibility with systems built on the original CLIP geometry.

We first validate ViSUv2 against existing safety datasets in terms of diversity and harmfulness. We then evaluate ShieldCLIP across cross-modal retrieval, text-to-image generation with Stable Difusion v1.4 and SDXL, and image-to-text generation with LLaVA. Beyond the main comparisons, we assess robustness across prompt distributions and diferent safety classifiers, study the contribution of modality-specific supervision and individual loss components, and analyze the safety–utility trade-of through retrieval, generation, and preservation metrics. Automatic evaluation is further complemented by human judgments of safety and content preservation.

Contributions. Our main contributions are:

• Data and supervision. We introduce ViSUv2, with independent text and image safety labels, 578 concepts, 28 categories, and human valida tion of the modality-level annotations.

• Selective alignment. We condition preservation and redirection on modality-specific safety states, explicitly handling mixed pairs and main taining coherence when both modalities are unsafe.

• Evaluation. We evaluate ShieldCLIP across cross-modal retrieval, difusion models, and mul timodal language models, complemented by evaluation with multiple safety classifiers, human judgments, targeted ablation studies of the selective objective, and preservation analyses.

Relation to the Conference Version. This work extends Safe-CLIP (S. Poppi et al., 2024), which addressed encoder-level mitigation through pair-level supervision that treated all generated samples as unsafe. Here, we formulate safety alignment at the modality level and condition the objective on the four possible safety states of gen erated image-text pairs. We replace ViSU with ViSUv2, which introduces independent text and image safety labels, expands the taxonomy from 20 to 28 categories, and includes human validation of the annotations. To disentangle the contribution of the new data from that of the selective objective, we conduct matched training experiments with Safe-CLIP and ShieldCLIP on both ViSU and ViSUv2. We further extend the experimental analysis with targeted ablation studies of modality-specific supervision and loss design, evaluation with SDXL and multiple safety classifiers, and a human preference study.

The rest of the paper is organized as follows. Section 2 reviews related work on safety alignment. Section 3.1 formalizes safety alignment in multimodal encoders, and Sections 3.2 and 3.3 introduce ViSUv2 and the ShieldCLIP framework, respectively. Section 4 presents the experimental evaluation and analyses. Additional dataset details, extended results, and qualitative examples are provided in the appendices.

## 2 Related Work

## 2.1 Harmful Content Mitigation Techniques

Approaches to mitigating harmful content in textto-image difusion models can be categorized by their intervention stage: decoding-time methods, which act during image generation, and predecoding methods, which operate on latent or embedding representations beforehand.

Decoding-Time Methods. These techniques directly regulate the difusion process to prevent unsafe visual concepts from emerging. Modelediting approaches modify the generative parameters to erase undesired concepts, either by finetuning the U-Net (S. Kim et al., 2024; Kumari et al., 2023; X. Li et al., 2024; R. Liu et al., 2025; Lyu et al., 2024) or by refining cross-attention layers for localized, scalable concept removal (Fan et al., 2024; Gandikota, Materzynska, Fiotto-Kaufman, & Bau, 2023; Gandikota, Orgad, Belinkov, Materzy´nska, & Bau, 2024; Huang et al., 2024; Lu, Wang, Li, Liu, & Kong, 2024). More recent work addresses the residual weaknesses of this family: the entanglement between an erased concept and its semantic neighbors (Shi, Dehghanpour Farashah, de Vries, & Farnadi, 2026), the imprecision of purely textside supervision (J. Li et al., 2026; Vardhana & Biswas, 2026), and the recovery of erased concepts under adversarial prompting (Tsai et al., 2024; Yin,

Tian, & Zhang, 2025). While efective, these methods can over-suppress related but benign concepts and typically require retraining or separate finetuning for each new concept to be removed, limiting scalability. Inference-guidance techniques instead steer sampling without altering model weights, using latent-space directions (H. Li, Shen, Torr, Tresp, & Gu, 2024) or classifier-free guidance mechanisms (Schramowski et al., 2023) to avoid unsafe activations. They are lightweight and reversible but often less robust to ambiguous prompts and limited in controllability. In contrast to these generatorspecific interventions, we operate upstream of the difusion process by aligning the encoder that conditions generation. The resulting encoder can therefore be reused across compatible downstream systems that rely on the same CLIP backbone, without modifying their decoders.

Pre-Decoding Methods. These methods intervene before image generation by sanitizing or aligning latent embeddings. For example, Latent Guard (R. Liu et al., 2024) filters harmful text representations, while Embedding Sanitizer (Qiu, Chen, Zhang, & Yang, 2024) improves robustness through optimization or preference learning, and DES (Ahn & Jung, 2025) and SafeText (Y. Hu, Jiang, & Gong, 2025) fine-tune the text encoder so that unsafe prompts are displaced while safe ones remain in place. PromptGuard (Yuan et al., 2026) instead prepends a learned soft prompt, moderating unsafe generations without updating model weights. In our previous work, Safe-CLIP (S. Poppi et al., 2024), we extended this idea to multimodal encoders, aligning safe and unsafe image-text pairs to weaken unsafe associations.

Two recent works refine this encoder-level formulation. HySAC (T. Poppi et al., 2025) abandons the flat embedding geometry, organizing safe and unsafe content in a hyperbolic hierarchy that queries can traverse toward safe regions, while SafeR-CLIP (Yousaf, Fioresi, et al., 2026) keeps each unsafe concept close to its nearest safe alternative, recovering zero-shot accuracy lost to more aggressive redirection. Both govern where unsafe content is sent. ShieldCLIP retains this predecoding, encoder-level formulation but instead governs what is redirected: generated items are labeled separately by modality, preservation is applied to generated-safe content, and redirection is restricted to the modalities labeled unsafe. Differently from our previous version, which assigns a single safety state to each generated pair, supervision here follows the observed content of each modality rather than its origin.

## 2.2 Datasets for Harmful Content Mitigation

Existing datasets for harmful content mitigation mainly focus on text-only benchmarks or prompt-based red-teaming, ofering limited multimodal coverage and weak safe-unsafe correspondence. I2P (Schramowski et al., 2023) collects 4.7k real-world web prompts using NSFW-related keywords to evaluate unsafe generations, while SneakyPrompt (Yang, Hui, Yuan, Gong, & Cao, 2024) provides ∼200 jailbreak prompts crafted with ChatGPT to elicit explicit content. Ring-A-Bell (Tsai et al., 2024) extends this line with subsets of NSFW, safe, and inverse prompts targeting nudity and violence, though many prompts are syntactically noisy and narrow. Similarly, MMA-Difusion (Yang, Gao, et al., 2024) introduces explicit NSFW prompts along with adversarial, sanitized, and safe variants, but lacks one-to-one safe-unsafe pairing and contains many ungrammatical prompts. CoPro (R. Liu et al., 2024) instead ofers structured textual pairs from 578 harmful concepts and their benign replacements, yet remains text-only and limited in size.

A second group assembles multimodal material. ToViLaG (X. Wang et al., 2023) collects toxic image-text pairs, MM-SafetyBench (X. Liu et al., 2024) and VLSBench (X. Hu, Liu, Li, Huang, & Shao, 2025) probe multimodal language models with paired visual and textual queries, and UnsafeBench (Qu et al., 2025) evaluates image safety classifiers on both real and generated images. Closer to our setting, CoProV2 (R. Liu et al., 2025) scales concept-level prompts into preference data for alignment, and NSFW-Caps (Yousaf, Fioresi, et al., 2026) contributes 1k aligned image-text pairs for measuring safety under distributional shift.

While these datasets enable concept-level safety evaluation, they carry a single safety label per sample or prompt, and none ofers aligned mul timodal pairs with modality-specific annotations. ViSU (S. Poppi et al., 2024) partially bridges this gap by pairing safe image-text samples with generated counterparts, but it treats all generated data as unsafe and uses a 20-category taxonomy. In contrast to all these resources, ViSUv2 preserves the paired real-generated design of ViSU while annotating each generated modality independently, over a taxonomy expanded to 28 categories and with more recent generators. This is the annotation that makes selective, modality-aware optimization possible in the first place.

![](images/092be42eeb3068299431e353c1c524404d730c621f724f6232b8119eca82836a.jpg)  
Fig. 1 Examples of real (safe) pairs and their generated (safe or unsafe) counterparts with harmful tags.

## 3 Problem Formulation and Preliminaries

## 3.1 Safety Alignment in Multimodal Encoders

Scope. We study safety alignment for CLIP-like multimodal encoders, used as core components for retrieval, captioning, and text-to-image generation pipelines. Our objective is twofold: (i) suppress harmful signals in text and images and (ii) preserve benign semantics so that downstream systems relying on the original CLIP features remain compatible. Because a closed definition of “safety” is infeasible, we operate, like prior work (Gandikota et al., 2023; R. Liu et al., 2025, 2024; S. Poppi et al., 2024; T. Poppi et al., 2025; Schramowski et al., 2023), over a taxonomy of concept proxies.

Preliminaries. Let T and V denote the trainable text and image encoders, and let $\mathcal { T } _ { 0 } , \mathcal { V } _ { 0 }$ be their frozen pre-trained CLIP counterparts, used as oracle encoders that provide anchor representations, following encoder-level alignment (S. Poppi et al., 2024). Formally, if a conceptual cleaning operator c(·) removes safety-violating content from an input x¯, the ideal encoder should map x¯ to the same embedding as the oracle computed on the cleaned input c(x¯). For either modality $\mathcal { E } \in \{ \mathcal { T } , \mathcal { V } \}$ we require

$$
\mathcal { E } ( \bar { x } ) \approx \mathcal { E } ( c ( \bar { x } ) ) \approx \mathcal { E } _ { 0 } ( c ( \bar { x } ) ) ,\tag{1}
$$

where ≈ denotes high cosine similarity in the shared embedding space. The independent permodality labels introduced next determine whether an input should be preserved or redirected.

## 3.2 The ViSUv2 Dataset

Overview. ViSUv2 extends the paired-data construction introduced with ViSU, in which each real image-caption pair has a semantically corresponding generated pair (S. Poppi et al., 2024), as illustrated in Fig. 1. It adds independent safety labels for text and image, since either generated modality may be safe or unsafe. Real pairs R are safe by construction, while generated pairs G may be safe or unsafe per modality. Formally,

$$
\begin{array} { r } { \mathcal { D } = \Big \{ ( v _ { i } , t _ { i } ) \in \mathcal { R } , ~ \big ( \bar { v } _ { i } ^ { [ f _ { i } ^ { v } ] } , \bar { t } _ { i } ^ { [ f _ { i } ^ { t } ] } \big ) \in \mathcal { G } \Big \} _ { i = 1 } ^ { N } , } \end{array}\tag{2}
$$

where $f _ { i } ^ { v } , f _ { i } ^ { t } \in \{ 0 , 1 \}$ denote safe (0) or unsafe (1) labels. For each index i, the generated pair $\left( { \bar { v } } _ { i } , { \bar { t } } _ { i } \right)$ is constructed to describe the same underlying scene/entity/action as $( v _ { i } , t _ { i } )$ . This captures mixed cases $( e . g .$ , generated-safe text with generatedunsafe image), as well as entirely safe and entirely unsafe generated pairs. These cases are not rare: 8.8% of the generated pairs are safe-safe, 52.3% are unsafe-unsafe, 8.4% contain safe text and an unsafe image, and 30.5% contain unsafe text and a safe image. Thus, 47.7% of the generated data would receive incorrect supervision if generation were treated as synonymous with harmfulness. An overview of the generation and validation pipeline is shown in Fig. 2.

Taxonomy. Prior safety-aligned methods typically rely on a 20-class harmful content taxonomy (S. Poppi et al., 2024; Schramowski et al., 2023). To better capture the diversity of unsafe content, we expand this taxonomy to 28 categories, guided by the 578 fine-grained NSFW concepts released by CoPro (R. Liu et al., 2024), which we use to both drive content generation and structure the taxonomy. Each concept is automatically mapped by an LLM (i.e., LLaMA 3.1-8B (Grattafiori et al., 2024)) into one or more categories, creating a many-to-many mapping that better reflects the multifaceted nature of unsafe content: for example, the concept rape is assigned to the sexual, violence, and abuse categories. Concepts thus drive sample generation, while categories enable higher-level analysis and evaluation. Further details regarding the prompt used to classify the CoPro concepts into the ViSUv2 categories are provided in Appendix A.4.

![](images/cc495b238fae88b578affea68f0e53a62f7e0f2af90e8556e05ec198e2759cf5.jpg)  
Fig. 2 ViSUv2 dataset generation. The pipeline used to generate unsafe text and images from real captions, followed by independent safety labeling.

To identify the categories missing from the original 20 ViSU classes (S. Poppi et al., 2024) (i.e., hate, harassment, violence, sufering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, abuse, brutality, and cruelty), we encode all ViSU categories to obtain a category list A, and encode all CoPro concepts to obtain a concept list B, using CLIP-ViT-L/14 to generate semantic embeddings. The concepts in B are then clustered into 30 groups, and for each ViSU category a ∈ A, we compute the pairwise cosine distance between a and each CoPro cluster, selecting the 8 CoPro clusters that are farthest away from the set A – i.e., the concepts not already represented in the original taxonomy.

To assign a general denomination to each of these new clusters, we examine the members of the cluster (drawn from B) and consult ChatGPT for assistance in producing concise, descriptive labels. We then expand A with these newly identified and named categories; for example, the cluster containing the concepts transphobia, ladyboy, asexual freak, homophobia, biphobia, LGBTQ+ individuals is assigned the category name “discrimination”.

![](images/1f4817e811d34fbf6c0b931ce0b7c64d872d0e205e671a809645e335f78acded.jpg)  
Fig. 3 Mapping of some CoPro granular NSFW concepts to the 28 NSFW categories in our taxonomy.

This process yields a total of 28 categories: hate, harassment, violence, sufering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, abuse, brutality, cruelty, discrimination, extremist ideologies, slurs, body shaming, mental health issues, shocking, vulgar humor, substance abuse. Fig. 3 shows an example of the resulting concept-to-category mapping.

Text Generation. We generate captions intended to express unsafe concepts starting from safe ones by applying a corruption strategy that preserves semantic context. Specifically, given a safe caption t , we prompt LLaMA-3.1-8B (Grattafiori et al., 2024) with an output prefilling attack technique (Cappelletti et al., 2026) to rewrite it into an unsafe variant t<sup>¯</sup><sub>i</sub> by conditioning on a sampled NSFW concept. To balance semantic relevance and coverage, we first compute CLIP similarities between t and the 578 candidate concepts. We retain the 50 most similar concepts and uniformly sample one to guide the rewriting. The LLM is then instructed to explicitly incorporate the sampled NSFW concept into the rewrite, ensuring that unsafe variants remain aligned with the underlying scene while difering in safety level. Each generated caption is then automatically classified as safe or unsafe using LLaMA-3.1-8B with a prefilling-based classification prompt (Cappelletti et al., 2026), and its underlying concept(s) are mapped to the 28 categories, enabling consistent safety labels and category-level analysis. Further generation and classification details are reported in Appendix A.

![](images/a053ec072961794fb5c2087519cbfc6727614625821fc71ad1b7f4841327d73b.jpg)  
Fig. 4 The ShieldCLIP architecture. Left: Real and generated text/image pairs are processed by trainable encoders to produce embeddings, while frozen oracle encoders (T<sub>0</sub>, V<sub>0</sub>) generate corresponding anchor embeddings. Right: The conditional training objectives. Real and generated-safe samples (label 0) are preserved by aligning them with their anchors within designated “safe zones”. Generated-unsafe samples (label 1) are redirected according to their modality-specific safety configuration, with dedicated objectives for unsafe, mixed, and jointly unsafe pairs.

Image Generation. For each generated caption t<sup>¯</sup> , we generate the corresponding image with one of two distinct text-to-image difusion mod els, selected based on the caption. Specifically, we employ NewRealityXL<sup>1</sup>, a Stable Difusion XL (Podell et al., 2024) variant fine-tuned for NSFW generation, and an uncensored variant<sup>2</sup> of FLUX (Black Forest Labs, 2024). The two models exhibit complementary strengths: FLUX tends to generate more explicit depictions when nudity is implied, whereas SDXL produces more semantically coherent scenes for other unsafe content (e.g. violence or sufering). To leverage this complementarity, we employ a classifier based on LLaMA-3.1-8B to predict whether the caption implies nudity. If present, the image is generated with FLUX; otherwise, with SDXL.

All generated images are automatically labeled as safe or unsafe by combining two complementary classifiers: NudeNet (Bedapudi, 2019), which detects nudity and exposed features, and Q16 (Schramowski, Tauchmann, & Kersting, 2022), which captures broader unsafe visual cues such as violence or disturbing visual content. An image is labeled unsafe if either classifier predicts NSFW, and safe only when both agree. The classifiers are used only to assign binary modality-level safety labels during dataset construction; their scores, representations, or gradients are never used in the training objective. This procedure yields a dataset of real and generated multimodal pairs with independent safety labels for both captions and images<sup>3</sup>.

## 3.3 ShieldCLIP: Selective Safety Alignment

Key Idea. ShieldCLIP conditions its training objectives on the observed safety state of each modality rather than the origin of a sample, combining preservation and redirection under modalityspecific supervision. It activates these objectives conditionally to (i) preserve real samples and generated-safe samples, avoiding over-sanitization and retaining compatibility with $\mathcal { T } _ { 0 } , \mathcal { V } _ { 0 } .$ , and (ii) redirect generated-unsafe samples toward their safe counterparts without perturbing unrelated regions of the embedding space. Additional objectives handle mixed pairs and maintain cross-modal coherence when both generated modalities are unsafe. An overview is shown in Fig. 4.

Notation and Masked Subsets. Within a batch, let R denote the set of real pairs, while $\mathbf { T } _ { s } , \mathbf { T } _ { u }$ be generated-safe/unsafe texts and $\mathbf { V } _ { s } , \mathbf { V } _ { u }$ the analogous generated-image sets. Intersections between these sets $\left( e . g . , \mathbf { T } _ { u } \cap \mathbf { V } _ { s } \right)$ capture mixed cases where one modality is safe and the other unsafe.

Building Blocks. Building on our previous work, Safe-CLIP, we use cosine alignment and bidirectional InfoNCE. Formally, let $\mathbf { X } = \{ x _ { i } \} _ { i = 1 } ^ { m }$ and $\mathbf { Y } = \{ y _ { i } \} _ { i = } ^ { m }$ denote paired sets of embeddings. The first objective is a cosine alignment loss defined as

$$
\mathcal { L } _ { \mathrm { c o s } } ( { \mathbf { X } } , { \mathbf { Y } } ) = - \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \cos { \left( x _ { i } , y _ { i } \right) } ,\tag{3}
$$

where $\cos ( \cdot )$ is cosine similarity between $\ell _ { 2 ^ { - } }$ normalized embeddings. We use Eq. 3 in two ways: (i) preservation to keep trainable encoders close to their oracles; (ii) redirection to pull unsafe embeddings toward their safe counterparts.

The second objective is a bidirectional InfoNCE loss formally defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { n c e } } ( \mathbf { X } , \mathbf { Y } ) = - \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left[ \log \frac { \exp \bigl ( \cos ( x _ { i } , y _ { i } ) / \tau \bigr ) } { \sum _ { j = 1 } ^ { m } \exp \bigl ( \cos ( x _ { i } , y _ { j } ) / \tau \bigr ) } \right. } \\ { \quad \left. + \log \frac { \exp \bigl ( \cos ( y _ { i } , x _ { i } ) / \tau \bigr ) } { \sum _ { j = 1 } ^ { m } \exp \bigl ( \cos ( y _ { i } , x _ { j } ) / \tau \bigr ) } \right] , } \end{array}\tag{4}
$$

where, for each direction, $( x _ { i } , y _ { i } )$ are positive pairs, $( x _ { i } , y _ { j } ) , j \neq i$ are negatives, and τ is a temperature parameter.

In practice, both losses are applied conditionally to the relevant batch subsets induced by $\mathcal { R } , \mathbf { T } _ { s } , \mathbf { T } _ { u } , \mathbf { V } _ { s } ,$ and $\mathbf { V } _ { u }$ (and their intersections), while the oracle branch is treated with stop-gradient.

Preservation: Real and Generated-Safe. For real pairs $( v , t ) \in \mathcal { R }$ , the goal is to preserve both intra-modal fidelity (trainable vs. oracle within the same modality) and cross-modal alignment. Using the cosine alignment loss from Eq. 3 and the bidirectional InfoNCE loss from Eq. 4, the preservation objective for real pairs is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { p r e s } } ^ { \mathrm { r e a l } } = \mathcal { L } _ { \mathrm { c o s } } \big ( \mathcal { V } ( v ) , \mathcal { V } _ { 0 } ( v ) \big ) + \mathcal { L } _ { \mathrm { c o s } } \big ( \mathcal { T } ( t ) , \mathcal { T } _ { 0 } ( t ) \big ) } \\ & { \qquad + \mathcal { L } _ { \mathrm { n c e } } \big ( \mathcal { V } ( v ) , \mathcal { T } _ { 0 } ( t ) \big ) + \mathcal { L } _ { \mathrm { n c e } } \big ( \mathcal { T } ( t ) , \mathcal { V } _ { 0 } ( v ) \big ) , } \end{array}\tag{5}
$$

The cosine terms keep the trainable encoders close to their frozen counterparts on the same input, while the InfoNCE terms enforce consistency across modalities when one branch is frozen.

For generated-safe items, we preserve each safe modality via cosine alignment, and apply crossmodal InfoNCE only when both modalities are safe:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { p r e s s } } ^ { \mathrm { g e n s s a f e } } = \underbrace { \mathcal { L } _ { \mathrm { c o s s } } ( \mathcal { V } ( \overline { { v } } ^ { [ 0 ] } ) , \mathcal { V } _ { 0 } ( \overline { { v } } ^ { [ 0 ] } ) ) } _ { \mathrm { c o m p u t t e d ~ o n l y ~ o n ~ V _ s ~ } } } \\ & { \quad + \underbrace { \mathcal { L } _ { \mathrm { c o s s } } ( \overline { { T ( \bar { t } ^ { [ 0 ] } ) } } , \mathcal { T } _ { 0 } ( \bar { t } ^ { [ 0 ] } ) ) } _ { \mathrm { c o m p u t t e d ~ o n l y ~ o n ~ T _ { s } ~ } } } \\ & { \quad + \underbrace { \mathcal { L } _ { \mathrm { n e c e } } ( \mathcal { V } ( \bar { v } ^ { [ 0 ] } ) , \mathcal { T } _ { 0 } ( \bar { t } ^ { [ 0 ] } ) ) } _ { \mathrm { c o m p u t t e d ~ o n l y ~ o n ~ T _ { s } ~ \cap ~ V _ { s } ~ } } } \\ & { \quad + \underbrace { \mathcal { L } _ { \mathrm { n e } } ( \mathcal { T } ( \bar { t } ^ { [ 0 ] } ) , \mathcal { V } _ { 0 } ( \bar { v } ^ { [ 0 ] } ) ) } _ { \mathrm { c o n p u t ~ o n ~ T _ { s } ~ } } . } \end{array}\tag{6}
$$

This ensures that benign generated content does not get mistakenly altered and remains compatible with the original CLIP space.

Redirection: Generated-Unsafe and Mixed Cases. When a generated element is labeled as unsafe, we redirect its embedding toward the corresponding real safe counterpart:

$$
\mathcal { L } _ { \mathrm { r e d i r } } = \underbrace { \mathcal { L } _ { \mathrm { c o s } } \bigl ( \mathcal { V } ( \bar { v } ^ { [ 1 ] } ) , \mathcal { V } _ { 0 } ( v ) \bigr ) } _ { \mathrm { c o m p u t e d ~ o n ~ } \mathbf { V } _ { u } } + \underbrace { \mathcal { L } _ { \mathrm { c o s } } \bigl ( \mathcal { T } ( \bar { t } ^ { [ 1 ] } ) , \mathcal { T } _ { 0 } ( t ) \bigr ) } _ { \mathrm { c o m p u t e d ~ o n ~ } \mathbf { T } _ { u } } .\tag{7}
$$

In this way, an unsafe sample (either image or text) is aligned to the anchor embedding of the safe corresponding element. In mixed cases, where only one generated modality is unsafe, we instead align the unsafe side to the safe one via crossmodal InfoNCE, freezing the safe branch to avoid distortion:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { m i x } } = \underbrace { \mathcal { L } _ { \mathrm { n c e } } \big ( \mathcal { V } ( \bar { v } ^ { [ 1 ] } ) , \mathcal { T } _ { 0 } ( \bar { t } ^ { [ 0 ] } ) \big ) } _ { \mathrm { c o m p u t e d ~ o n ~ } \mathbf { V } _ { u } \cap \mathbf { T } _ { s } } } \\ & { ~ + \underbrace { \mathcal { L } _ { \mathrm { n c e } } \big ( \mathcal { T } ( \bar { t } ^ { [ 1 ] } ) , \mathcal { V } _ { 0 } ( \bar { v } ^ { [ 0 ] } ) \big ) } _ { \mathrm { c o m p u t e d ~ o n ~ } \mathbf { T } _ { u } \cap \mathbf { V } _ { s } } . } \end{array}\tag{8}
$$

Finally, if both generated modalities are unsafe, we encourage them to remain semantically coherent with each other so that redirection preserves internal consistency:

$$
\mathcal { L } _ { \mathrm { c o h } } = \underbrace { \mathcal { L } _ { \mathrm { n c e } } \big ( \mathcal { T } ( \bar { t } ^ { [ 1 ] } ) , \mathcal { V } ( \bar { v } ^ { [ 1 ] } ) \big ) } _ { \mathrm { c o m p u t e d ~ o n ~ } \mathbf { T } _ { u } \cap \mathbf { V } _ { u } } .\tag{9}
$$

Table 1 Comparison of text and image diversity and harmfulness across ViSUv2 and existing safety datasets. Values in parentheses indicate the number of safe captions or images excluded from the score computation.
<table><tr><td rowspan="2"># Text</td><td rowspan="2"></td><td rowspan="2"># Img</td><td rowspan="2"># Cat Labels</td><td colspan="2"></td><td colspan="2">Text Harm. (%)</td><td>Img Div.</td><td colspan="2">Img Harm. (%)</td></tr><tr><td></td><td>Vendi-S ↑ Self-BLEU ↓</td><td></td><td>LLaMA-3.1 ↑ GPT-3.5 ↑</td><td>Vendi-S ↑</td><td>NN-Q16 ↑ GPT-4 ↑</td><td></td></tr><tr><td>SneakyPrompt</td><td>181</td><td></td><td></td><td>x x</td><td>6.1</td><td>0.355 0.279</td><td>62.5 55.0</td><td></td><td></td><td></td></tr><tr><td>Ring-A-Bell</td><td>1.1k (80)</td><td>1</td><td></td><td></td><td>14.6</td><td>0.395</td><td>63.8</td><td>=</td><td></td><td>-</td></tr><tr><td>I2P</td><td>4.7k</td><td></td><td></td><td>x</td><td>99.9</td><td></td><td>22.4</td><td>-</td><td></td><td>-</td></tr><tr><td>MMA-Diffusion</td><td>3k (1k)</td><td>80 (80)</td><td></td><td>x</td><td>12.8</td><td>0.392</td><td>54.2</td><td>6.1</td><td>75.4</td><td>0.0</td></tr><tr><td>ViSU</td><td>165k (165k)</td><td>165k (165k)</td><td>20</td><td>x</td><td>44.7</td><td>0.304</td><td>85.2</td><td>28.4</td><td>70.8</td><td>83.3</td></tr><tr><td>ViSUv2 (Ours)</td><td>161k (229k)</td><td>118k (272k)</td><td>28</td><td>√</td><td>55.1</td><td>0.243</td><td>89.6</td><td>32.1</td><td>100.0</td><td>85.1</td></tr></table>

Overall Training Objective. The final training objective is a weighted sum of the previously described loss functions:

$$
\begin{array} { r l } & { \mathcal { L } = \lambda _ { \mathrm { { r e a l } } } \mathcal { L } _ { \mathrm { { p r e s } } } ^ { \mathrm { { r e a l } } } + \lambda _ { \mathrm { { s a f e } } } \mathcal { L } _ { \mathrm { { p r e s } } } ^ { \mathrm { { g e n - s a f e } } } } \\ & { ~ + \lambda _ { \mathrm { { r e d i r } } } \mathcal { L } _ { \mathrm { { r e d i r } } } + \lambda _ { \mathrm { { m i x } } } \mathcal { L } _ { \mathrm { { m i x } } } + \lambda _ { \mathrm { { c o h } } } \mathcal { L } _ { \mathrm { { c o h } } } , } \end{array}\tag{10}
$$

where λ parameters are used to balance the contribution of each loss component.

Overall, conditioning preservation and redirection on the observed safety state of each modality, rather than on the origin of a sample, lets Shield-CLIP treat generation and harmfulness as independent: safe content is anchored regardless of whether it is real or generated, unsafe content is redirected only where it is actually unsafe, and mixed and jointly-unsafe pairs are handled by dedicated terms instead of being collapsed into a single pair-level label. This finer-grained supervision reduces oversanitization of benign generated content while still suppressing harmful associations, and it keeps the resulting encoders compatible with downstream systems built on the original CLIP embedding space. We evaluate these benefits next, on crossmodal retrieval and on the downstream generative systems that consume CLIP embeddings.

## 4 Experiments

## 4.1 Implementation Details

Dataset Splits. To construct the ViSUv2 dataset, real samples are drawn from COCO (Lin et al., 2014) and Flickr30k (Young, Lai, Hodosh, & Hockenmaier, 2014), two complementary datasets of natural images annotated with human-written descriptions. For each image, we randomly select one of the five available captions to reduce redundancy and avoid excessive semantic overlap across samples while maintaining linguistic variety.

Overall, ViSUv2 contains 195k text-image quadruplets, of which 30k are derived from Flickr30k and 165k from COCO. We use 8,000 quadruplets for testing, 8,000 for validation, and the remaining 179,000 for training. No COCO or Flickr30k image appears in more than one split.

Implementation and Training Details. We fine-tune two distinct CLIP backbone models: ViT-L-14 (corresponding to the text encoder used in Stable Difusion v1.4) and ViT-bigG/14 (the other text encoder in Stable Difusion XL). For parametereficient fine-tuning, we apply LoRA (E.J. Hu et al., 2022) to all Transformer layers of both the text and vision encoders with a rank r equal to 16. All parameters of the original oracle encoders (T , V ) remain frozen during training. Both models are trained using the Adam optimizer with a constant learning rate of $1 \times 1 0 ^ { - 4 }$ . We train for a maximum of 50 epochs and employ early stopping based on recall, with a patience of 5 epochs. The model based on ViT-L/14 is trained on 8 A100 GPUs with a per-GPU batch size of 16 and no gradient accumulation, while the model based on ViT-bigG is trained on 16 GPUs with a per-GPU batch size of 8 and 2 gradient accumulation steps. The λ weights defined in Eq. 10 are set to (0.1, 0.1, 0.1, 0.25, 0.25), where $\lambda = ( \lambda _ { \mathrm { { r e a l } } } , \lambda _ { \mathrm { { s a f e } } } , \lambda _ { \mathrm { { r e d i r } } } , \lambda _ { \mathrm { { m i x } } } , \lambda _ { \mathrm { { c o h } } } )$

## 4.2 Dataset Evaluation

Table 1 compares ViSUv2 with existing safety datasets, including text-only resources (i.e., SneakyPrompt (Yang, Hui, et al., 2024), Ring-A-Bell (Tsai et al., 2024), and I2P (Schramowski et al., 2023)) and multimodal ones (i.e., MMA-Difusion (Yang, Gao, et al., 2024) and ViSU (S. Poppi et al., 2024)). The evaluation focuses on two key dimensions: diversity, reflecting how varied the text or images are within each dataset, and harmfulness, indicating how reliably unsafe content is captured. For text, diversity is measured with the Vendi-Score (from CLIP textual embeddings) (Friedman & Dieng, 2023), which quantifies the entropy of semantic similarities across samples, where higher values indicate greater diversity, and Self-BLEU (Perez et al., 2022), which measures sentence redundancy (lower values indicate higher diversity). Text harmfulness is estimated via LLM-based classification using GPT-3.5 Turbo (Brown et al., 2020), prompted to flag unsafe content. For images, diversity is computed using the same Vendi-Score formulation on CLIP visual embeddings, while harmfulness is evaluated through GPT-4 (Achiam et al., 2023). Additionally, we report the proportion of harmful items detected by the classifiers used during the dataset safety labeling stage (LLaMA-3.1-8B (Grattafiori et al., 2024) and the NudeNet (Bedapudi, 2019)-Q16 (Schramowski et al., 2022) ensemble), shown for reference to facilitate comparison across datasets. All scores are computed on items containing unsafe content, using the test splits of each dataset when available. Further details on the harmfulness and diversity evaluation protocols are provided in Appendix B.2.

Table 2 Rate of generated harmful images using unsafe textual prompts from I2P (Schramowski et al., 2023) and the proposed ViSUv2 dataset. Results are computed with both SD v1.4 and SDXL as text-to-image generators, combining predictions from NudeNet and Q16. Avg is the harmful rate across all individual generations regardless of its categories.
<table><tr><td></td><td colspan="8">I2P</td><td colspan="8">ViSUv2</td></tr><tr><td>Model</td><td>Hate Harass Viol S-Harm</td><td></td><td></td><td></td><td></td><td>Sex Shock Ill Act Avg</td><td></td><td></td><td></td><td></td><td>Hate Harass Viol S-Harm</td><td></td><td></td><td>Sex Shock Ill Act</td><td></td><td>Avg</td></tr><tr><td>SD v1.4</td><td>40.5</td><td>32.4</td><td>42.0</td><td>40.0</td><td>24.1</td><td>50.8</td><td>36.7</td><td>38.1</td><td>27.5</td><td>26.3</td><td>30.2</td><td>27.0</td><td>17.4</td><td>21.9</td><td>27.8</td><td>25.4</td></tr><tr><td>SLD-Strong (Schramowski et al., 2023)</td><td>13.5</td><td>11.5</td><td>15.2</td><td>8.9</td><td>5.4</td><td>19.3</td><td>8.5</td><td>11.8</td><td>4.3</td><td>4.3</td><td>5.7</td><td>3.7</td><td>3.4</td><td>3.4</td><td>3.9</td><td>4.1</td></tr><tr><td>ESD (Gandikota et al., 2023)</td><td>39.1</td><td>32.3</td><td>43.1</td><td>40.2</td><td>21.5</td><td>48.1</td><td>34.6</td><td>37.0</td><td>25.6</td><td>21.4</td><td>29.4</td><td>26.7</td><td>15.1</td><td>19.4</td><td>26.4</td><td>23.4</td></tr><tr><td>SPM (Lyu et al., 2024)</td><td>23.0</td><td>21.0</td><td>34.0</td><td>25.7</td><td>15.2</td><td>37.4</td><td>22.7</td><td>25.6</td><td>15.5</td><td>13.5</td><td>18.7</td><td>16.1</td><td>9.0</td><td>11.1</td><td>17.0</td><td>14.4</td></tr><tr><td>UCE (Gandikota et al., 2024)</td><td>32.1</td><td>25.2</td><td>27.8</td><td>18.6</td><td>15.2</td><td>30.5</td><td>20.6</td><td>24.3</td><td>16.1</td><td>14.5</td><td>16.9</td><td>15.6</td><td>9.9</td><td>12.3</td><td>15.2</td><td>14.4</td></tr><tr><td>SalUn (Fan et al., 2024)</td><td>30.6</td><td>25.6</td><td>39.0</td><td>34.6</td><td>15.0</td><td>40.6</td><td>29.0</td><td>30.6</td><td>18.4</td><td>14.2</td><td>22.8</td><td>19.5</td><td>6.7</td><td>12.2</td><td>21.3</td><td>16.4</td></tr><tr><td>Receler (Huang et al., 2024)</td><td>17.8</td><td>15.4</td><td>22.8</td><td>18.3</td><td>8.8</td><td>22.1</td><td>14.6</td><td>17.1</td><td>10.7</td><td>7.2</td><td>14.0</td><td>11.6</td><td>4.8</td><td>7.3</td><td>12.1</td><td>9.7</td></tr><tr><td>Embedding Sanitizer (Qiu et al., 2024)</td><td>21.2</td><td>17.4</td><td>23.0</td><td>23.1</td><td>9.2</td><td>27.0</td><td>16.2</td><td>18.1</td><td>13.2</td><td>13.3</td><td>14.5</td><td>14.2</td><td>6.4</td><td>10.1</td><td>16.6</td><td>12.5</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>24.7</td><td>20.5</td><td>24.4</td><td>21.3</td><td>13.6</td><td>27.9</td><td>20.1</td><td>21.8</td><td>6.7</td><td>7.8</td><td>6.7</td><td>6.0</td><td>4.4</td><td>5.3</td><td>6.0</td><td>6.1</td></tr><tr><td>SafeR-CLIP (Yousaf, Fioresi, et al., 2026)</td><td>21.4</td><td>18.8 12.9</td><td>16.8</td><td>16.0</td><td>10.4</td><td>22.4</td><td>15.5</td><td>16.8</td><td>7.8</td><td>7.4</td><td>7.7</td><td>6.9</td><td>6.9</td><td>7.0</td><td>7.7</td><td>7.5</td></tr><tr><td>SafetyDPO (R. Liu et al., 2025)</td><td>14.2 6.2</td><td>4.7</td><td>13.0 4.4</td><td>11.4</td><td>5.0</td><td>13.7</td><td>9.3</td><td>11.4</td><td>2.8</td><td>3.1</td><td>3.2</td><td>2.6</td><td>1.6</td><td>2.7</td><td>3.2</td><td>2.8</td></tr><tr><td>DES (Ahn &amp; Jung, 2025) ShieldCLIP (Ours)</td><td>4.5</td><td>4.0</td><td>4.1</td><td>3.1 3.0</td><td>1.0 1.8</td><td>5.0 4.4</td><td>2.7</td><td>3.9</td><td>0.9 0.9</td><td>1.4</td><td>1.4</td><td>1.2</td><td>0.5</td><td>1.3</td><td>0.9</td><td>1.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>3.1</td><td>3.5</td><td></td><td>1.2</td><td>1.2</td><td>1.0</td><td>0.6</td><td>1.2</td><td>1.1</td><td>1.0</td></tr><tr><td>SDXL</td><td>41.0</td><td>30.6</td><td>35.7</td><td>38.5</td><td>17.1</td><td>45.3</td><td>34.4</td><td>33.6</td><td>36.3</td><td>18.7</td><td>42.7</td><td>31.2</td><td>17.4</td><td>10.0</td><td>41.6</td><td>33.4</td></tr><tr><td>ESD (Gandikota et al., 2023)</td><td>38.9</td><td>30.7</td><td>36.9</td><td>37.7</td><td>17.8</td><td>46.1</td><td>31.7</td><td>33.4</td><td>33.5</td><td>18.3</td><td>39.7</td><td>24.6</td><td>14.5</td><td>10.5</td><td>37.3</td><td>30.3</td></tr><tr><td>SPM (Lyu et al., 2024)</td><td>22.9</td><td>22.9</td><td>27.1</td><td>20.9</td><td>10.1</td><td>30.7</td><td>18.5</td><td>21.9</td><td>19.3</td><td>21.5</td><td>23.1</td><td>22.8</td><td>9.6</td><td>16.7</td><td>20.5</td><td>19.1</td></tr><tr><td>UCE (Gandikota et al., 2024)</td><td>40.4</td><td>37.6</td><td>42.9</td><td>38.0</td><td>34.1</td><td>40.7</td><td>40.3</td><td>38.8</td><td>53.6</td><td>49.4</td><td>51.3</td><td>50.2</td><td>43.1</td><td>52.1</td><td>55.3</td><td>50.0</td></tr><tr><td>Embedding Sanitizer (Qiu et al., 2024)</td><td>38.5 10.3</td><td>29.2 8.2</td><td>35.2 8.5</td><td>38.1</td><td>17.1</td><td>43.4</td><td>34.1</td><td>33.5</td><td>30.2</td><td>29.0</td><td>39.5</td><td>39.3</td><td>15.3</td><td>26.0</td><td>37.6</td><td>31.2 2.0</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>7.1</td><td>4.9</td><td>9.7</td><td>7.2</td><td>3.7</td><td>10.7</td><td>8.8</td><td>8.2</td><td>2.0</td><td>2.3</td><td>2.5</td><td>1.6</td><td>1.3</td><td>1.8</td><td>2.1</td><td></td></tr><tr><td>SafetyDPO (R. Liu et al., 2025)</td><td>19.9</td><td>13.7</td><td>15.3</td><td>11.6 17.5</td><td>3.5 6.6</td><td>12.3 23.8</td><td>8.6 17.2</td><td>8.7 16.3</td><td>4.5 3.6</td><td>3.4 3.0</td><td>5.0 9.9</td><td>4.8</td><td>1.8 2.3</td><td>4.1 2.1</td><td>6.9</td><td>4.4</td></tr><tr><td>DES (Ahn &amp; Jung, 2025)</td><td>4.9</td><td>6.0</td><td>5.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>8.9</td><td></td><td></td><td>7.3</td><td>5.3</td></tr><tr><td>ShieldCLIP (Ours)</td><td></td><td></td><td></td><td>4.4</td><td>1.7</td><td>6.4</td><td>4.6</td><td>4.6</td><td>2.0</td><td>0.9</td><td>1.3</td><td>1.0</td><td>1.0</td><td>0.5</td><td>1.7</td><td>1.4</td></tr></table>

Overall, ViSUv2 achieves the most balanced performance across all dimensions, combining high semantic diversity with accurate and consistent safety labeling. It attains the lowest Self-BLEU and strong Vendi-Score values across multimodal datasets, indicating diverse yet coherent unsafe captions, while exhibiting the highest proportion of harmful text. On the visual side, ViSUv2 shows the greatest image diversity and the highest level of harmfulness among generated images, consistently surpassing ViSU and MMA-Difusion. These results show that ViSUv2 provides a rich and well-calibrated source of unsafe content while maintaining clear distinctions between safe and unsafe modalities.

## 4.3 Text-to-Image Generation Results

Main Results. We evaluate ShieldCLIP as the text encoder in text-to-image difusion models, measuring the proportion of harmful generations across multiple NSFW prompt sources. Experiments are conducted with both SD v1.4 (Rombach et al., 2022) and SDXL (Podell et al., 2024), applying our mitigation strategy to the CLIP text encoders used by each model. Each prompt is sampled five times with diferent random seeds, and generated images are classified using the combined NudeNet and Q16 classifiers (Bedapudi, 2019; Schramowski et al., 2022). Table 2 reports results on I2P (Schramowski et al., 2023), using the entire set of prompts available in the dataset, and on the NSFW subset of the ViSUv2 test split, covering a broad range of unsafe categories.

![](images/03f492861223ada353ae250e588f59b2e6f42658203b5311ee0bc6ad0c5ce8f8.jpg)  
Fig. 5 Qualitative examples generated with SD v1.4, SDXL, ShieldCLIP, and competing methods using unsafe prompts from I2P and ViSUv2.

For each dataset, we compute the average violation rate across categories<sup>4</sup>, averaging per prompt over the five generations. We compare Shield-CLIP against generator-side mitigation approaches based on inference-time guidance or model editing (Fan et al., 2024; Gandikota et al., 2023, 2024; Huang et al., 2024; R. Liu et al., 2025; Lyu et al., 2024; Schramowski et al., 2023), as well as pre-decoding methods that operate on the conditioning representations, including Embedding Sanitizer (Qiu et al., 2024), DES (Ahn & Jung, 2025)<sup>5</sup>, Safe-CLIP (S. Poppi et al., 2024), and SafeR-CLIP (Yousaf, Fioresi, et al., 2026).

Across both difusion backbones and datasets, ShieldCLIP achieves the lowest average harmful generation rate. On SD v1.4, it reduces harmful generations from 38.1% to 3.5% on I2P and from 25.4% to 1.0% on ViSUv2. Compared with our previous Safe-CLIP (S. Poppi et al., 2024), this corresponds to a substantial reduction from 21.8% and 6.1%, respectively. ShieldCLIP also improves over the recent encoder-level SafeR-CLIP (Yousaf, Fioresi, et al., 2026) (16.8% and 7.5%) and strong mitigation methods such as SafetyDPO (R. Liu et al., 2025) (11.4% and 2.8%), SLD-Strong (Schramowski et al., 2023) (11.8% and 4.1%), and DES (Ahn & Jung, 2025) (3.9% and

Table 3 Rate of generated harmful images using prompts from diferent sources, combining predictions from NudeNet and Q16.
<table><tr><td rowspan="2">Model</td><td colspan="4">% Harmful Content (↓)</td></tr><tr><td>ViSU SneakyPrompt MMA Ring-A-Bell</td><td></td><td></td><td></td></tr><tr><td>SD v1.4</td><td>24.6</td><td>40.0</td><td>41.7</td><td>62.5</td></tr><tr><td>SLD-Strong (Schramowski et al., 2023)</td><td>4.8</td><td>16.1</td><td>24.0</td><td>17.5</td></tr><tr><td>SalUn (Fan et al., 2024)</td><td>16.3</td><td>25.0</td><td>9.2</td><td>33.8</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>4.1</td><td>11.7</td><td>8.0</td><td>21.3</td></tr><tr><td>SafeR-CLIP (Yousaf, Fioresi, et al., 2026)</td><td>6.9</td><td>12.7</td><td>8.1</td><td>25.0</td></tr><tr><td>SafetyDPO (R. Liu et al., 2025)</td><td>2.7</td><td>1.7</td><td>2.2</td><td>6.3</td></tr><tr><td>DES (Ahn &amp; Jung, 2025)</td><td>1.8</td><td>1.2</td><td>1.3</td><td>0.0</td></tr><tr><td>ShieldCLIP (Ours)</td><td>1.2</td><td>1.1</td><td>1.3</td><td>0.0</td></tr><tr><td>SDXL</td><td>36.5</td><td>23.9</td><td>19.9</td><td>47.5</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>1.6</td><td>1.1</td><td>3.1</td><td>3.7</td></tr><tr><td>SafetyDPO (R. Liu et al., 2025)</td><td>7.9</td><td>2.2</td><td>2.0</td><td>8.8</td></tr><tr><td>DES (Ahn &amp; Jung, 2025)</td><td>14.6</td><td>6.7</td><td>3.1</td><td>36.3</td></tr><tr><td>ShieldCLIP (Ours)</td><td>1.7</td><td>0.0</td><td>0.7</td><td>3.7</td></tr></table>

1.1%). The advantage becomes more pronounced with SDXL, where ShieldCLIP reaches 4.6% on I2P and 1.4% on ViSUv2, consistently improving over Safe-CLIP, SafetyDPO, and DES. Overall, these results show that selective alignment provides consistent safety gains across both datasets and difusion backbones.

These gains also extend beyond the two main evaluation benchmarks. To probe robustness across diferent NSFW prompt distributions, Table 3 reports results on four additional sources: ViSU (S. Poppi et al., 2024), SneakyPrompt (Yang, Hui, et al., 2024), MMA (Yang, Gao, et al., 2024), and Ring-A-Bell (Tsai et al., 2024), including adversarial and implicitly unsafe prompts designed to bypass safety mechanisms. On SD v1.4, Shield-CLIP achieves the best or tied-best result on all four datasets, limiting harmful generations to 1.2%, 1.1%, 1.3%, and 0.0%, respectively. These results consistently improve over Safe-CLIP and

Table 4 Rate of generated harmful images using unsafe textual prompts from I2P (Schramowski et al., 2023) and the proposed ViSUv2 dataset. Results are computed with both SD v1.4 and SDXL as text-to-image generators, using LlavaGuard (Helf, Friedrich, Brack, Schramowski, & Kersting, 2025) as safety classifier.
<table><tr><td></td><td colspan="7">I2P</td><td colspan="8">ViSUv2</td></tr><tr><td>Model</td><td>Hate Harass Viol S-Harm Sex Shock Ill Act Avg</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Hate Harass Viol S-Harm Sex Shock Ill Act Avg</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SD v1.4</td><td>16.1</td><td>16.6</td><td>33.4</td><td>19.7</td><td>44.7</td><td>28.0</td><td>15.2 24.8</td><td></td><td>23.9 19.9</td><td>23.9</td><td>17.7</td><td>22.6</td><td>16.0</td><td>18.6</td><td>20.4</td></tr><tr><td>SLD-Strong (Schramowski et al., 2023)</td><td>3.6</td><td>4.6</td><td>13.0</td><td>3.5</td><td>13.6</td><td>8.6</td><td>5.2 7.4</td><td>5.8</td><td>3.6</td><td>6.6</td><td>3.0</td><td>5.9</td><td>3.3</td><td>4.5</td><td>4.7</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>7.6</td><td>7.5</td><td>16.9</td><td>7.4</td><td>25.1</td><td>10.3</td><td>5.3 11.5</td><td>3.0</td><td>3.4</td><td>2.8</td><td>2.4</td><td>1.7</td><td>1.5</td><td>2.4</td><td>2.5</td></tr><tr><td>SafeR-CLIP (Yousaf, Fioresi, et al., 2026)</td><td>7.6</td><td>7.2</td><td>11.1</td><td>6.9</td><td>13.8</td><td>9.0</td><td>5.5 8.7</td><td>2.9</td><td>3.7</td><td>3.3</td><td>3.1</td><td>2.7</td><td>2.8</td><td>3.5</td><td>3.1</td></tr><tr><td>SafetyDPO (R. Liu et al., 2025)</td><td>3.3</td><td>5.6</td><td>10.3</td><td>2.5</td><td>11.5</td><td>4.5</td><td>4.7 6.1</td><td>2.2</td><td>1.2</td><td>2.6</td><td>1.3</td><td>1.4</td><td>1.1</td><td>2.2</td><td>1.7</td></tr><tr><td>DES (Ahn &amp; Jung, 2025)</td><td>2.4</td><td>1.4</td><td>2.9</td><td>1.4</td><td>1.5</td><td>2.5</td><td>2.7 2.1</td><td></td><td>0.4 0.5</td><td>0.9</td><td>0.7</td><td>0.1</td><td>0.5</td><td>0.7</td><td>0.5</td></tr><tr><td>ShieldCLIP (Ours)</td><td>1.9</td><td>1.5</td><td>2.7</td><td>1.1</td><td>1.1</td><td>2.3</td><td>2.3 1.8</td><td>0.3</td><td>0.2</td><td>0.3</td><td>0.3</td><td>0.2</td><td>0.2</td><td>0.5</td><td>0.3</td></tr><tr><td>SDXL</td><td>26.2</td><td>22.0</td><td>39.8</td><td>30.9</td><td>34.2</td><td>44.5</td><td>24.9 33.4</td><td>28.3</td><td>14.1</td><td>32.5</td><td>22.6</td><td>20.4</td><td>7.4</td><td>25.3</td><td>26.3</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>1.9</td><td>2.9</td><td>5.0</td><td>1.6</td><td>3.6</td><td>3.4</td><td>2.5 3.1</td><td>1.2</td><td>1.1</td><td>1.0</td><td>1.6</td><td>0.3</td><td>1.6</td><td>1.1</td><td>0.9</td></tr><tr><td>SafetyDPO (R. Liu et al., 2025)</td><td>1.0</td><td>5.2</td><td>6.6</td><td>2.7</td><td>4.4</td><td>4.3</td><td>3.2 3.9</td><td>3.4</td><td>2.0</td><td>4.7</td><td>2.3</td><td>1.8</td><td>1.4</td><td>4.1</td><td>2.8</td></tr><tr><td>DES (Ahn &amp; Jung, 2025)</td><td>6.9</td><td>6.6</td><td>12.6</td><td>4.7</td><td>5.9</td><td>8.5</td><td>3.9 7.0</td><td>2.9</td><td>2.3</td><td>10.3</td><td>5.3</td><td>2.0</td><td>1.0</td><td>6.6</td><td>4.3</td></tr><tr><td>ShieldCLIP (Ours)</td><td>0.6</td><td>2.5</td><td>2.5</td><td>0.8</td><td>0.8</td><td>2.5</td><td>1.8 1.8</td><td></td><td>0.3 0.2</td><td>0.4</td><td>2.0</td><td>0.2</td><td>0.0</td><td>0.8</td><td>0.4</td></tr></table>

SafeR-CLIP, while remaining competitive with strong baselines such as SafetyDPO and DES. With SDXL, ShieldCLIP achieves the best or tiedbest result on three of four datasets and remains within 0.1 percentage points of the best result on ViSU, further supporting robustness across prompt distributions and difusion backbones. A categorylevel breakdown of the ViSU results is provided in Appendix C.1.

Qualitative results in Fig. 5 further confirm this trend across both violent and sexually explicit prompts. Compared with Safe-CLIP, SafetyDPO, and SLD-Strong, ShieldCLIP more consistently suppresses harmful visual cues while preserving the main scene semantics, without substantially distorting the generated content.

Evaluation with an Alternative Safety Classifier. Since NudeNet and Q16 are used to assign image-level safety labels during ViSUv2 construc tion, we additionally evaluate all generated images with LlavaGuard (Helf et al., 2025), which is not used during dataset construction or training. Unlike the NudeNet-Q16 ensemble, LlavaGuard is a multimodal LLM-based moderation model that reasons over visual content according to a textual safety policy, providing a substantially diferent evaluation mechanism. This experiment therefore directly tests whether the gains of ShieldCLIP persist under a safety evaluator disjoint from the labeling pipeline.

Tables 4 and 5 report harmful generation rates using LlavaGuard. On the main I2P and ViSUv2 benchmarks (Table 4), ShieldCLIP achieves the lowest average violation rate across both difusion backbones. With SD v1.4, it reaches 1.8% on I2P and 0.3% on ViSUv2, substantially improving over our previous Safe-CLIP (S. Poppi et al., 2024) (11.5% and 2.5%), SafeR-CLIP (Yousaf, Fioresi, et al., 2026) (8.7% and 3.1%), and SafetyDPO (R. Liu et al., 2025) (6.1% and 1.7%), while also outperforming DES (Ahn & Jung, 2025) (2.1% and 0.5%). The same trend holds with SDXL, where ShieldCLIP obtains 1.8% and 0.4%, compared with 3.1% and 0.9% for Safe-CLIP, 3.9% and 2.8% for SafetyDPO, and 7.0% and 4.3% for DES.

Table 5 Rate of generated harmful images using prompts from diferent sources, using predictions from LlavaGuard (Helf et al., 2025).
<table><tr><td rowspan="2">Model</td><td colspan="4">% Harmful Content (↓)</td></tr><tr><td></td><td>ViSU SneakyPrompt MMA Ring-A-Bell</td><td></td><td></td></tr><tr><td>SD v1.4</td><td>21.1</td><td>70.7</td><td>64.2</td><td>67.5</td></tr><tr><td>SLD-Strong (Schramowski et al., 2023)</td><td>6.4</td><td>35.4</td><td>37.2</td><td>26.3</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>1.5</td><td>12.2</td><td>7.3</td><td>22.5</td></tr><tr><td>SafeR-CLIP (Yousaf, Fioresi, et al., 2026)</td><td>2.7</td><td>4.4</td><td>2.4</td><td>18.8</td></tr><tr><td>SafetyDPO (R. Liu et al., 2025)</td><td>3.0</td><td>5.0</td><td>3.1</td><td>8.8</td></tr><tr><td>DES (Ahn &amp; Jung, 2025)</td><td>1.5</td><td>3.3</td><td>0.0</td><td>7.5</td></tr><tr><td>ShieldCLIP (Ours)</td><td>0.5</td><td>0.6</td><td>0.1</td><td>1.3</td></tr><tr><td>SDXL</td><td>33.4</td><td>55.0</td><td>51.3</td><td>77.5</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>0.7</td><td>0.0</td><td>7.9</td><td>0.0</td></tr><tr><td>SafetyDPO (R. Liu et al., 2025)</td><td>8.2</td><td>0.0</td><td>3.8</td><td>0.0</td></tr><tr><td>DES (Ahn &amp; Jung, 2025)</td><td>13.4</td><td>23.9</td><td>6.7</td><td>55.0</td></tr><tr><td>ShieldCLIP (Ours)</td><td>1.5</td><td>0.0</td><td>0.4</td><td>0.0</td></tr></table>

The results extend to the additional prompt sources in Table 5. ShieldCLIP achieves the best or tied-best result on three of four datasets for each difusion backbone, while remaining consistently low on the remaining cases. In particular, with SD v1.4, harmful generations are limited to 0.5%, 0.6%, 0.1%, and 1.3% on ViSU, SneakyPrompt, MMA, and Ring-A-Bell, respectively; with SDXL, the corresponding rates remain between 0.0% and 1.5%. This behavior is consistent across substantially different prompt distributions, including adversarial and red-teaming benchmarks.

Table 6 User study results, showing the percentage of preferences for competitors vs. ShieldCLIP.
<table><tr><td></td><td>Safe-CLIP</td><td>SafeR-CLIP</td><td>SafetyDPO</td></tr><tr><td>Safety</td><td>11.9</td><td>11.4</td><td>10.2</td></tr><tr><td>Preservation</td><td>23.5</td><td>26.2</td><td>22.1</td></tr></table>

Overall, LlavaGuard closely reproduces the trends observed with NudeNet-Q16 on both the main benchmarks and the additional prompt sources. Since LlavaGuard is not used for dataset labeling or training, these consistent gains show that the improvements of ShieldCLIP are not specific to the safety classifiers used to construct ViSUv2, but reflect robust reductions in harmful generation.

Human Evaluation. To complement the automated evaluation, we conduct a human preference study with 20 independent annotators. We collect 1,200 pairwise evaluations from prompts randomly sampled from the ViSUv2 test set, balanced equally across the three competing methods (400 comparisons each): Safe-CLIP (S. Poppi et al., 2024), SafeR-CLIP (Yousaf, Fioresi, et al., 2026), and SafetyDPO (R. Liu et al., 2025). Each comparison presents ShieldCLIP alongside one competitor, with method identities hidden and left/right order randomized.

Annotators evaluate each pair according to two criteria: safety, indicating which image better avoids harmful or inappropriate content, and content preservation, indicating which image better preserves the semantics of the original prompt while remaining safe. For each criterion, annotators can prefer either image or indicate a tie when neither output is clearly preferred. To obtain a single preference score for each method, ties are split equally between the two models. Table 6 reports the resulting percentage of preference assigned to each competitor over ShieldCLIP. Across all three comparisons, ShieldCLIP is preferred by a clear majority for both criteria. These results complement the automated evaluation, confirming that our method produces outputs that are both safer and more faithful to the original prompts compared to existing approaches.

## 4.4 Image-to-Text Generation Results

Table 7 reports the proportion of harmful text generated by multimodal LLMs when conditioned on unsafe visual inputs. We evaluate ShieldCLIP as the visual encoder within the LLaVA architecture (H. Liu et al., 2024, 2023), using the LLaMA-2-13B variant as the language backbone<sup>6</sup>. Generated captions are classified as harmful by combining two LLM-based classifiers (i.e., LLaMA-3.1-8B and GPT-3.5 Turbo), marking a caption unsafe if either classifier flags it.

Table 7 Rate of generated harmful text using unsafe images from diferent sources as input, computed by combining predictions from LLaMA-3.1-8B and GPT-3.5 Turbo.
<table><tr><td rowspan="2">Model</td><td colspan="2">% Harmful Content (↓)</td></tr><tr><td>NudeNet NSFW URLs SMID</td><td></td></tr><tr><td>LLaVA-LLaMA-2-13B</td><td>58.6</td><td>32.4</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>14.6</td><td>8.8 2.3</td></tr><tr><td>SafeR-CLIP (Yousaf, Fioresi, et al., 2026)</td><td>17.4</td><td>2.2</td></tr><tr><td>ShieldCLIP(Ours)</td><td>12.5</td><td>2.5</td></tr></table>

In this setting, we compare ShieldCLIP against the standard LLaVA baseline, equipped with the original CLIP visual encoder, and two safety-aligned multimodal encoders: our previous Safe-CLIP (S. Poppi et al., 2024) and SafeR-CLIP (Yousaf, Fioresi, et al., 2026). We evaluate 1,000 unsafe real images from each of three sources, covering sexual and nudity content (i.e., NudeNet (Bedapudi, 2019) and NSFW data source URLs) as well as broader unsafe visual cues (i.e., SMID (Crone, Bode, Murawski, & Laham, 2018)).

Across all data sources, replacing the original visual encoder with ShieldCLIP substantially reduces harmful caption generation. On NudeNet and NSFW URLs, ShieldCLIP achieves the lowest harmful rates of 12.5% and 6.2%, improving over Safe-CLIP (14.6% and 7.5%) and SafeR-CLIP (17.4% and 10.2%). On SMID, all safety-aligned encoders perform similarly, with SafeR-CLIP obtaining the lowest rate (2.2%) and ShieldCLIP remaining close at 2.5%. Overall, these results show that the gains of ShieldCLIP extend beyond textto-image generation to the image-to-text setting.

The qualitative examples in Fig. 6 further illustrate this behavior: ShieldCLIP suppresses explicit visual cues that lead the original LLaVA, and in some cases Safe-CLIP, to produce unsafe descriptions, while retaining the coarse semantic context depicted in the input image.

![](images/9761943e0c611bbe66c9ba9d39a2696620ec0162572ddf165e8c38cdd671155d.jpg)  
Fig. 6 Qualitative examples of image-to-text generation with the original LLaVA model, Safe-CLIP, and ShieldCLIP, using real NSFW images from diferent sources as input.

## 4.5 Image-Text Retrieval Results

After assessing the harmful-content mitigation performance of ShieldCLIP when employed as either textual or visual encoders in multimodal generative pipelines, we next evaluate whether the aligned embedding space also improves safety in crossmodal retrieval. Table 8 compares ShieldCLIP with our previous Safe-CLIP formulation (S. Poppi et al., 2024), HySAC (T. Poppi et al., 2025), a hyperbolic safety-aware CLIP variant, and SafeR-CLIP (Yousaf, Fioresi, et al., 2026), always including the base CLIP model as reference.

We evaluate both text-to-image (T2I) and image-to-text (I2T) retrieval on ViSU (S. Poppi et al., 2024), ViSUv2, and the three unsafe image sources used before (i.e., NudeNet (Bedapudi, 2019), NSFW URLs, and SMID (Crone et al., 2018)), following the evaluation protocol defined in prior works (S. Poppi et al., 2024; T. Poppi et al., 2025). In both directions, an unsafe query retrieves from a mixed pool of safe and unsafe candidates, and we report the fraction of queries whose top-1 result is harmful. For ViSU and ViSUv2, we use their paired test splits; for ViSUv2, queries are restricted to samples where both generated modalities are unsafe. For the three image-only sources, we combine 1,000 unsafe images from each dataset with safe distractors from LAION-400M (Schuhmann et al., 2022), using unsafe ViSUv2 captions as T2I queries and the unsafe images themselves as I2T queries.

Across benchmarks, ShieldCLIP strongly reduces harmful retrievals in both directions. In

Table 8 Rate of retrieved harmful items from diferent sources, using unsafe visual and textual prompts.
<table><tr><td></td><td colspan="4">% Harmful Content (↓)</td></tr><tr><td>Model</td><td>ViSUv2 ViSU NudeNet NSFW URLs SMID</td><td></td><td></td><td></td></tr><tr><td>CLIP (T2I)</td><td>95.2</td><td>90.9</td><td>57.1</td><td>55.2</td></tr><tr><td>HySAC (T. Poppi et al., 2025)</td><td>32.1</td><td>18.7</td><td>3.8 8.2</td><td>6.1</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>62.9</td><td>51.4</td><td>8.3 0.0</td><td>19.9 16.4</td></tr><tr><td>SafeR-CLIP (Yousaf, Fioresi, et al., 2026)</td><td>21.0</td><td>37.1</td><td></td><td>0.0</td></tr><tr><td>ShieldCLIP (Ours)</td><td>3.5</td><td>14.4</td><td>0.2</td><td>0.3</td></tr><tr><td>CLIP (I2T)</td><td>95.4</td><td>90.5</td><td>65.6</td><td>57.4</td></tr><tr><td>HySAC (T. Poppi et al., 2025)</td><td>79.4</td><td>74.9 15.6 28.8</td><td>4.9</td><td>41.4 2.1</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>71.2</td><td>57.7</td><td>24.7</td><td>34.5</td></tr><tr><td>SafeR-CLIP (Yousaf, Fioresi, et al., 2026)</td><td>23.1</td><td>17.7</td><td>53.0 0.1</td><td>52.7</td></tr><tr><td>ShieldCLIP (Ours)</td><td>20.8</td><td>22.6</td><td></td><td>0.4</td></tr></table>

T2I retrieval, the harmful rate drops from 95.2% to 3.5% on ViSUv2 and from 90.9% to 14.4% on ViSU, substantially improving over Safe-CLIP (62.9% and 51.4%), HySAC (32.1% and 18.7%), and SafeR-CLIP (21.0% and 37.1%). On NudeNet, NSFW URLs, and SMID, ShieldCLIP remains below 0.5%, closely matching the best SafeR-CLIP results. In I2T retrieval, ShieldCLIP achieves 20.8% on ViSUv2 and 22.6% on ViSU, while reducing harmful retrievals on the three real-image sources to 0.0%, 0.1%, and 0.4%, respectively. Notably, SafeR-CLIP exhibits a marked directional imbalance on these datasets, increasing to 70.3%, 53.0%, and 52.7% in I2T despite near-zero T2I rates. In contrast, ShieldCLIP maintains consistently low harmful retrieval across both directions, indicating a more balanced safety alignment of the shared embedding space.

Qualitative results in Fig. 7 provide a complementary view of these trends. For both unsafe textual and visual queries, ShieldCLIP redirects retrieval toward benign candidates, whereas CLIP and Safe-CLIP often retain harmful matches, consistent with the quantitative results.

![](images/c54778595f99362481aa054b079b8f06a95c7993fba777b93780c52e5c81c334.jpg)  
Fig. 7 Qualitative examples of text-to-image (left) and image-to-text (right) retrieval using unsafe text or image queries, comparing ShieldCLIP with the original CLIP model and Safe-CLIP.

Table 9 Disentangling the efect of training data and selective alignment. We compare Safe-CLIP (S. Poppi et al., 2024) and ShieldCLIP when trained on either ViSU or ViSUv2, reporting harmful rates $( \% , \downarrow )$
<table><tr><td></td><td></td><td colspan="2">Retrieval (T2I)</td><td colspan="2">Retrieval (I2T)</td><td colspan="2">Generation</td></tr><tr><td>Method</td><td>Training</td><td>ViSUv2</td><td>ViSU</td><td>ViSUv2</td><td>ViSU</td><td>ViSUv2</td><td>I2P</td></tr><tr><td>Safe-CLIP</td><td>ViSU</td><td>62.9</td><td>51.4</td><td>71.2</td><td>57.7</td><td>6.1</td><td>21.8</td></tr><tr><td>ShieldCLIP</td><td>ViSU</td><td>34.5</td><td>34.0</td><td>19.1</td><td>21.0</td><td>2.2</td><td>7.4</td></tr><tr><td>Safe-CLIP</td><td>ViSUv2</td><td>34.9</td><td>46.9</td><td>15.2</td><td>59.9</td><td>1.3</td><td>8.1</td></tr><tr><td>ShieldCLIP</td><td>ViSUv2</td><td>3.5</td><td>14.4</td><td>20.8</td><td>22.6</td><td>1.0</td><td>3.5</td></tr></table>

## 4.6 Ablation Studies

Data vs. Selective Objective. We first disentangle the contribution of the training data from that of the selective alignment objective. To obtain a matched comparison with our previous Safe-CLIP (S. Poppi et al., 2024), we train both methods on ViSU and ViSUv2. When training ShieldCLIP on ViSU, we derive modality-specific safety labels using the same labeling pipeline adopted for ViSUv2; conversely, when training Safe-CLIP on ViSUv2, we discard these labels and treat all generated samples as unsafe, following its original formulation.

Table 9 shows that both factors contribute to the final performance. Replacing ViSU with ViSUv2 substantially improves Safe-CLIP in several settings, confirming the benefit of the broader training data. More importantly, when the training data are fixed, ShieldCLIP consistently improves T2I retrieval and generation, with particularly large gains on I2P and in cross-dataset retrieval.

Table 10 Ablation of modality-specific supervision. We report harmful rates $( \% , \downarrow ) .$ . Each variant removes or relaxes the conditional treatment of safe, unsafe, and mixed modality pairs.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Retrieval (T2I)</td><td colspan="2">Retrieval (I2T)</td><td colspan="2">Generation</td></tr><tr><td>ViSUv2</td><td>ViSU</td><td>ViSUv2</td><td>ViSU</td><td>ViSUv2 I2P</td><td></td></tr><tr><td>No Lmix; Lcoh on all pairs</td><td>14.9</td><td>24.8</td><td>37.1</td><td>38.5</td><td>1.4</td><td>8.1</td></tr><tr><td>Without  $\mathcal { L } _ { \mathrm { c o h } }$ </td><td>5.3</td><td>16.4</td><td>22.4</td><td>30.7</td><td>1.1</td><td>6.0</td></tr><tr><td>All generated → unsafe</td><td>14.1</td><td>22.1</td><td>26.6</td><td>27.9</td><td>1.8</td><td>6.7</td></tr><tr><td>Mixed → unsafe-unsafe</td><td>11.8</td><td>20.1</td><td>28.3</td><td>31.9</td><td>1.4</td><td>6.2</td></tr><tr><td>ShieldCLIP</td><td>3.5</td><td>14.4</td><td>20.8</td><td>22.6</td><td>1.0</td><td>3.5</td></tr></table>

The only exception is in-domain I2T retrieval on ViSUv2, where Safe-CLIP reaches 15.2% compared with 20.8% for ShieldCLIP; however, the same model transfers poorly to ViSU (59.9% versus 22.6%). These results indicate that the gains of ShieldCLIP cannot be attributed to ViSUv2 alone: modality-aware supervision provides complementary improvements, especially for T2I retrieval, generation, and cross-dataset robustness.

Efectiveness of Selective Supervision. We next ablate the modality-specific supervision introduced in Section 3.3. Table 10 considers two variants that collapse the four safety states: treating every generated sample as unsafe, as in pair-level supervision, and treating mixed pairs as unsafeunsafe. We further remove ${ \mathcal { L } } _ { \mathrm { c o h } }$ , which maintains cross-modal consistency for jointly unsafe pairs, and evaluate a non-selective variant that removes ${ \mathcal { L } } _ { \mathrm { m i x } }$ while applying ${ \mathcal { L } } _ { \mathrm { c o h } }$ to all generated pairs.

Collapsing the modality-specific labels consistently degrades performance. In particular, 47.7% of the generated pairs in ViSUv2 are not unsafeunsafe (38.9% are mixed and 8.8% are safe-safe), so assigning uniform unsafe supervision redirects at least one modality that should instead be preserved. Treating mixed pairs as unsafe-unsafe similarly worsens all retrieval and generation results, supporting the explicit handling of asymmetric safety states through ${ \mathcal { L } } _ { \mathrm { m i x } }$

Table 11 Ablation of the loss formulation and safety-quality trade-of. We report harmful rates $( \% , \downarrow )$ for retrieval and generation, together with FID and CLIP-Sim to measure generation quality and semantic alignment. For the weight sweep, $\lambda = \left( \lambda _ { \mathrm { { r e a l } } } , \lambda _ { \mathrm { { s a f e } } } , \lambda _ { \mathrm { { r e d i r } } } , \lambda _ { \mathrm { { m i x } } } , \lambda _ { \mathrm { { c o h } } } \right)$ . Safety metrics are evaluated on ViSUv2 using T2I retrieval, I2T retrieval, and SD v1.4 generation, while FID and CLIP-Sim are computed on 30k samples from COCO.
<table><tr><td></td><td></td><td>Retrieval (T2I)</td><td>Retrieval (I2T)</td><td>Generation</td><td colspan="2">Generation Utility</td></tr><tr><td>Model</td><td>λ</td><td>% Harmful Content (↓)</td><td>% Harmful Content (↓)</td><td>% Harmful Content (↓)</td><td>FID (↓) CLIP-Sim (↑)</td><td></td></tr><tr><td>Loss formulation</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Only cosine losses</td><td></td><td>0.0</td><td>100.0</td><td>2.8</td><td>80.3</td><td>0.096</td></tr><tr><td>Loss-weight trade-off</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>High preservation</td><td>(0.5, 0.5, 0.1, 0.1, 0.5)</td><td>18.9</td><td>34.5</td><td>3.3</td><td>15.1</td><td>0.258</td></tr><tr><td>Medium preservation</td><td>(0.25, 0.25, 0.1, 0.1, 0.25)</td><td>11.4</td><td>33.9</td><td>2.4</td><td>15.5</td><td>0.255</td></tr><tr><td>ShieldCLIP (selected) (0.1,0.1,0.1,0.25,0.25)</td><td></td><td>3.5</td><td>20.8</td><td>1.0</td><td>17.8</td><td>0.255</td></tr><tr><td>Medium redirection</td><td>(0.1,0.1, 0.25, 0.25,0.1)</td><td>2.1</td><td>10.9</td><td>0.7</td><td>18.1</td><td>0.244</td></tr><tr><td>High redirection</td><td>(0.1, 0.1, 0.5, 0.5, 0.1)</td><td>1.5</td><td>8.2</td><td></td><td>19.6</td><td>0.241</td></tr></table>

The loss ablation studies show the same trend. Removing $\mathcal { L } _ { \mathrm { c o h } }$ consistently increases harmful rates, indicating that maintaining semantic consistency while jointly unsafe modalities are redirected is beneficial. Conversely, removing ${ \mathcal { L } } _ { \mathrm { m i x } }$ and applying coherence indiscriminately to all pairs produces the strongest degradation among these objective variants. Together, these results support the conditional design of ShieldCLIP: mixed pairs benefit from a dedicated asymmetric objective, while coherence is most efective when restricted to the unsafe-unsafe subset for which it was designed.

Loss Formulation and Safety-Quality Tradeof. We finally analyze the role of the loss formulation and the balance between preservation and redirection. Table 11 first replaces the InfoNCE terms with cosine alignment only, and then varies the loss weights λ from preservationto redirection-oriented configurations.

To select the operating point, we jointly consider safety and generation utility. Safety is measured through harmful rates for T2I retrieval, I2T retrieval, and SD v1.4 generation. We additionally report FID (Heusel, Ramsauer, Unterthiner, Nessler, & Hochreiter, 2017) and CLIP-Sim on 30k COCO validation samples (Lin et al., 2014), measuring image fidelity and prompt-image semantic alignment, respectively (lower FID and higher CLIP-Sim are better). CLIP-Sim is computed using the CLIP-ViT-L/14 backbone. These metrics provide a direct measure of the utility cost induced by stronger safety alignment.

Using cosine alignment alone leads to a highly degenerate solution: although T2I harmful retrieval drops to 0.0%, I2T retrieval increases to 100.0%, while FID and CLIP-Sim degrade substantially. This indicates that point-wise cosine alignment is insuficient to preserve the structured cross-modal geometry required for balanced retrieval, supporting the use of the contrastive InfoNCE objectives in our formulation.

The weight sweep reveals a clear safetyquality trade-of. Preservation-oriented configurations retain better image fidelity and semantic alignment, but leave substantially more harmful content in retrieval and generation. Increasing the redirection weights progressively improves safety, at the cost of higher FID and lower CLIP-Sim. We select $\lambda = ( 0 . 1 , 0 . 1 , 0 . 1 , 0 . 2 5 , 0 . 2 5 )$ for Shield-CLIP as a balanced operating point: stronger redirection further improves safety, particularly for I2T retrieval, but introduces a more noticeable degradation in generation quality and semantic alignment.

## 4.7 Preservation Analysis

Zero-Shot Capability Preservation. Table 12 evaluates whether safety alignment preserves the original visual recognition capabilities of CLIP. We report zero-shot top-1 accuracy on CIFAR-10, CIFAR-100 (Krizhevsky & Hinton, 2009), SUN-397 (Herranz, Jiang, & Li, 2016), Food-101 (Bossard, Guillaumin, & Van Gool, 2014), Caltech-101 (Fei-Fei, Fergus, & Perona, 2006), and Imagenette (Howard, 2019), comparing Shield-CLIP with the original CLIP encoder, our previous Safe-CLIP (S. Poppi et al., 2024), and SafeR-CLIP (Yousaf, Fioresi, et al., 2026).

Table 12 Preservation analysis of CLIP performance on zero-shot classification, reported in terms of top-1 accuracy (↑).
<table><tr><td>Model</td><td>CIFAR-10</td><td>CIFAR-100</td><td>SUN-397</td><td>Food-101</td><td>Caltech-101</td><td>Imagenette</td></tr><tr><td>CLIP</td><td>94.5</td><td>60.7</td><td>63.6</td><td>88.1</td><td>74.8</td><td>99.7</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>87.6</td><td>61.1</td><td>54.2</td><td>74.8</td><td>64.4</td><td>98.5</td></tr><tr><td>SafeR-CLIP (Yousaf, Fioresi, et al., 2026)</td><td>92.1</td><td>61.5</td><td>58.4</td><td>77.2</td><td>74.3</td><td>98.8</td></tr><tr><td>ShieldCLIP (Ours)</td><td>91.9</td><td>64.0</td><td>61.8</td><td>79.5</td><td>74.8</td><td>99.3</td></tr></table>

Table 13 Preservation analysis of generation performance on 30k COCO samples. We report FID and CLIP-Sim for SD v1.4 and SDXL backbones.
<table><tr><td></td><td colspan="2">SD v1.4</td><td colspan="2">SDXL</td></tr><tr><td>Model</td><td></td><td>FID CLIP-Sim</td><td>FID CLIP-Sim</td><td></td></tr><tr><td>SD</td><td>14.7</td><td>0.266</td><td>13.0</td><td>0.269</td></tr><tr><td>SLD-Strong (Schramowski et al., 2023)</td><td>19.2</td><td>0.239</td><td></td><td></td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>15.7</td><td>0.259</td><td>16.5</td><td>0.258</td></tr><tr><td>SafeR-CLIP (Yousaf, Fioresi, et al., 2026)</td><td>25.4</td><td>0.228</td><td></td><td></td></tr><tr><td>SafetyDPO (R. Liu et al., 2025)</td><td>20.9</td><td>0.254</td><td>22.3</td><td>0.259</td></tr><tr><td>ShieldCLIP (Ours)</td><td>17.8</td><td>0.255</td><td>17.4</td><td>0.258</td></tr></table>

Across benchmarks, ShieldCLIP consistently improves over Safe-CLIP and outperforms SafeR-CLIP on five of six datasets. At the same time, it remains close to the original CLIP model, match ing or exceeding its accuracy on CIFAR-100 and Caltech-101 and retaining strong performance on the remaining tasks. These results confirm that the selective alignment mechanism in ShieldCLIP efectively mitigates harmful associations while preserving the semantic richness and generaliza tion ability of the original CLIP embeddings, maintaining their efectiveness on standard vision benchmarks.

Generation Quality and Semantic Alignment. We further analyze generation utility in Table 13, extending the FID and CLIP-Sim eval uation used in the loss-weight ablation to both SD v1.4 and SDXL. As in the loss-weight ablation, results are computed on 30k COCO validation samples. As expected, safety alignment introduces a moderate utility cost. With SD v1.4, ShieldCLIP changes FID from 14.7 to 17.8 and CLIP-Sim from 0.266 to 0.255; with SDXL, the corresponding values change from 13.0 to 17.4 and from 0.269 to 0.258. Semantic alignment remains comparable to Safe-CLIP on both backbones, while ShieldCLIP provides substantially stronger harmful-content mitigation in the generation experiments. The FID degradation also remains lower than SafetyDPO, which reaches 20.9 on SD v1.4 and 22.3 on SDXL. These results confirm that the operating point selected in Table 11 provides a favorable balance between safety and generation utility.

## 5 Conclusion

We introduced ShieldCLIP, a selective safetyalignment framework for multimodal encoders, together with ViSUv2, a 195k-quadruplet dataset with independent safety labels for generated text and images. By conditioning preservation and redirection on the observed safety state of each modality, ShieldCLIP preserves benign content, redirects only unsafe representations, and explic itly handles mixed and jointly unsafe pairs. Across cross-modal retrieval, text-to-image generation with Stable Difusion v1.4 and SDXL, and imageto-text generation with LLaVA, ShieldCLIP consistently reduces harmful outputs compared with prior safety-aligned encoders and strong mitigation baselines while preserving the utility of the original embedding space. Targeted ablation stud ies further disentangle the contribution of ViSUv2 from that of the selective objective and confirm the importance of modality-specific supervision, dedicated treatment of mixed pairs, and the proposed loss formulation. The observed gains remain consistent across diferent prompt distributions and safety classifiers, while human judgments further support the safety and content-preservation properties of the resulting outputs. Overall, these results show that treating safety at the modality level provides a more efective and balanced alternative to uniformly redirecting generated content.

Acknowledgments. This work has been supported by the EU Horizon projects “ELIAS” (GA No. 101120237) and “ELLIOT” (GA No. 101214398). We also acknowledge the CINECA award under the ISCRA initiative, for the availability of high-performance computing resources.

## References

Abid, A., Farooqi, M., Zou, J. (2021). Persistent Anti-Muslim Bias in Large Language Models. Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society.

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., . . . others (2023). GPT-4 Technical Report. arXiv preprint arXiv:2303.08774 , ,

Ahn, J., & Jung, H. (2025). Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Difusion Models. Advances in Neural Information Processing Systems.

Bedapudi, P. (2019). NudeNet: Neural Nets for Nudity Classification, Detection, and Selective Censoring. https://github.com/notAI -tech/NudeNet.

Birhane, A., Prabhu, V.U., Kahembwe, E. (2021). Multimodal Datasets: Misogyny, Pornography, and Malignant Stereotypes. arXiv preprint arXiv:2110.01963 , ,

Black Forest Labs (2024). FLUX. https://github .com/black-forest-labs/flux.

Bommasani, R., Hudson, D.A., Adeli, E., Altman, R., Arora, S., von Arx, S., . . . others (2021). On the Opportunities and Risks of Foundation Models. arXiv preprint arXiv:2108.07258, ,

Bossard, L., Guillaumin, M., Van Gool, L. (2014). Food-101 – Mining Discriminative Components with Random Forests. Proceedings of the European Conference on Computer Vision.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., . . . others (2020). Language Models are Few-Shot Learners. Advances in Neural Information Processing Systems.

Cappelletti, S., Poppi, T., Poppi, S., Yong, Z.-X., Garcia-Olano, D., Cornia, M., . . . Cucchiara, R. (2026). Improving LLM First-Token Predictions in Multiple-Choice Question Answering via Output Prefilling. Proceedings of the International Conference on Pattern Recognition.

Crone, D.L., Bode, S., Murawski, C., Laham, S.M. (2018). The Socio-Moral Image Database (SMID): A novel stimulus set for the study of social, moral and afective processes. PLoS ONE, 13(1), e0190954,

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., . . . others (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. Proceedings of the International Conference on Machine Learning.

Fan, C., Liu, J., Zhang, Y., Wong, E., Wei, D., Liu, S. (2024). SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation. Proceedings of the International Conference on Learning Representations.

Fei-Fei, L., Fergus, R., Perona, P. (2006). One-Shot Learning of Object Categories. IEEE Transactions on Pattern Analysis and Machine Intelligence, 28 (4), 594–611,

Friedman, D., & Dieng, A.B. (2023). The Vendi Score: A Diversity Evaluation Metric for Machine Learning. Transactions on Machine Learning Research, ,

Gandikota, R., Materzynska, J., Fiotto-Kaufman, J., Bau, D. (2023). Erasing Concepts from Difusion Models. Proceedings of the IEEE/CVF International Conference on Computer Vision.

Gandikota, R., Orgad, H., Belinkov, Y., Materzy´nska, J., Bau, D. (2024). Unified Concept Editing in Difusion Models. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., . . . others (2024). The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783 , ,

Helf, L., Friedrich, F., Brack, M., Schramowski, P., Kersting, K. (2025). LlavaGuard: An Open VLM-based Framework for Safeguarding Vision Datasets and Models. Proceedings of the International Conference on Machine Learning.

Herranz, L., Jiang, S., Li, X. (2016). Scene Recognition With CNNs: Objects, Scales and Dataset Bias. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. Advances in Neural Information Processing Systems.

Howard, J. (2019). Imagenette. https://github .com/fastai/imagenette.

Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., . . . others (2022). LoRA: Low-Rank Adaptation of Large Language Models. Proceedings of the International Conference on Learning Representations.

Hu, X., Liu, D., Li, H., Huang, X., Shao, J. (2025). VLSBench: Unveiling Visual Leakage in Multimodal Safety. Proceedings of the Annual Meeting of the Association for Computational Linguistics.

Hu, Y., Jiang, Z., Gong, N.Z. (2025). SafeText: Safe Text-to-Image Models via Aligning the Text Encoder. arXiv preprint arXiv:2502.20623,

Huang, C.-P., Chang, K.-P., Tsai, C.-T., Lai, Y.-H., Yang, F.-E., Wang, Y.-C.F. (2024). Receler: Reliable Concept Erasing of Text-to-Image Difusion Models via Lightweight Erasers. Proceedings of the European Conference on Computer Vision.

Kim, C., & Qi, Y. (2025). A Comprehensive Survey on Concept Erasure in Textto-Image Difusion Models. arXiv preprint arXiv:2502.14896, ,

Kim, S., Jung, S., Kim, B., Choi, M., Shin, J., Lee, J. (2024). Safeguard Text-to-Image Difusion Models with Human Feedback Inversion. Proceedings of the European Conference on Computer Vision.

Krizhevsky, A., & Hinton, G. (2009). Learning Multiple Layers of Features from Tiny Images.

Kumari, N., Zhang, B., Wang, S.-Y., Shechtman, E., Zhang, R., Zhu, J.-Y. (2023). Ablating Concepts in Text-to-Image Difusion Models. Proceedings of the IEEE/CVF International Conference on Computer Vision.

Li, H., Shen, C., Torr, P., Tresp, V., Gu, J. (2024). Self-Discovering Interpretable Diffusion Latent Directions for Responsible Text-to-Image Generation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Li, J., Xiong, L., Li, Z., Jiang, W., Fu, Z., Li, Y., Xie, G.-S. (2026). Beyond Text Prompts: Precise Concept Erasure through Text-Image Collaboration. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Li, X., Yang, Y., Deng, J., Yan, C., Chen, Y., Ji, X., Xu, W. (2024). SafeGen: Mitigating Sexually Explicit Content Generation in Text-to-Image Models. Proceedings of the ACM SIGSAC Conference on Computer and Communications Security.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., . . . Zitnick, C.L. (2014). Microsoft COCO: Common Objects in Context. Proceedings of the European Conference on Computer Vision.

Liu, H., Li, C., Li, Y., Lee, Y.J. (2024). Improved Baselines with Visual Instruction Tuning. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Liu, H., Li, C., Wu, Q., Lee, Y.J. (2023). Visual Instruction Tuning. Advances in Neural Information Processing Systems.

Liu, R., Chieh, C.I., Gu, J., Zhang, J., Pi, R., Chen, Q., . . . Pizzati, F. (2025). Safety-DPO: Scalable Safety Alignment for Textto-Image Generation. Proceedings of the IEEE/CVF International Conference on Computer Vision.

Liu, R., Khakzar, A., Gu, J., Chen, Q., Torr, P., Pizzati, F. (2024). Latent Guard: a Safety Framework for Text-to-image Generation. Proceedings of the European Conference on Computer Vision.

Liu, X., Zhu, Y., Gu, J., Lan, Y., Yang, C., Qiao, Y. (2024). MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models. Proceedings of the European Conference on Computer Vision.

Lu, S., Wang, Z., Li, L., Liu, Y., Kong, A.W.-K. (2024). MACE: Mass Concept Erasure in Diffusion Models. Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Lyu, M., Yang, Y., Hong, H., Chen, H., Jin, X., He, Y., . . . Ding, G. (2024). One-dimensional Adapter to Rule Them All: Concepts Diffusion Models and Erasing Applications. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Perez, E., Huang, S., Song, F., Cai, T., Ring, R., Aslanides, J., . . . Irving, G. (2022). Red Teaming Language Models with Language Models. Proceedings of the Conference on Empirical Methods in Natural Language Processing.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., . . . Rombach, R. (2024). SDXL: Improving Latent Difusion Models for High-Resolution Image Synthesis. Proceedings of the International Conference on Learning Representations.

Poppi, S., Poppi, T., Cocchi, F., Cornia, M., Baraldi, L., Cucchiara, R. (2024). Safe-CLIP: Removing NSFW Concepts from Visionand-Language Models. Proceedings of the European Conference on Computer Vision.

Poppi, T., Kasarla, T., Mettes, P., Baraldi, L., Cucchiara, R. (2025). Hyperbolic Safety-Aware Vision-Language Models. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Qiu, H., Chen, G., Zhang, M., Yang, M. (2024). Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding. arXiv preprint arXiv:2411.10329 , ,

Qu, Y., Shen, X., Wu, Y., Backes, M., Zannettou, S., Zhang, Y. (2025). UnsafeBench: Benchmarking Image Safety Classifiers on Real-World and AI-Generated Images. Proceedings of the ACM SIGSAC Conference on Computer and Communications Security.

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., . . . Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. Proceedings of the International Conference on Machine Learning.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. (2022). High-Resolution Image Synthesis With Latent Difusion Models. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Schramowski, P., Brack, M., Deiseroth, B., Kersting, K. (2023). Safe Latent Difusion: Mitigating Inappropriate Degeneration in Difusion Models. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Schramowski, P., Tauchmann, C., Kersting, K. (2022). Can Machines Help Us Answering Question 16 in Datasheets, and In Turn Reflecting on Inappropriate Content? Proceedings of the ACM Conference on Fairness, Accountability, and Transparency.

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., . . . others (2022). LAION-5B: An open large-scale dataset for training next generation imagetext models. Advances in Neural Information Processing Systems.

Shi, Z., Dehghanpour Farashah, A., de Vries, R., Farnadi, G. (2026). Neighbor-Aware Localized Concept Erasure in Text-to-Image Difusion Models. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Tsai, Y.-L., Hsu, C.-Y., Xie, C., Lin, C.-H., Chen, J.-Y., Li, B., . . . Huang, C.-Y. (2024). Ring-A-Bell! How Reliable are Concept Removal Methods for Difusion Models? Proceedings of the International Conference on Learning Representations.

Vardhana, K.S., & Biswas, S. (2026). GenErase: Generalizable and Semantically-Aware Concept Erasure in Difusion Models. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Wang, T., Li, F., Zhu, L., Li, J., Zhang, Z., Shen, H.T. (2024). Cross-Modal Retrieval: A Systematic Review of Methods and Future Directions. Proceedings of the IEEE, 112 , 1716–1754,

Wang, X., Yi, X., Jiang, H., Zhou, S., Wei, Z., Xie, X. (2023). ToViLaG: Your Visual-Language Generative Model is Also An Evildoer. Proceedings of the Conference on Empirical Methods in Natural Language Processing.

Wei, C., Chen, Y., Chen, H., Hu, H., Zhang, G., Fu, J., . . . Chen, W. (2024). UniIR: Training and Benchmarking Universal Multimodal Information Retrievers. Proceedings of the European Conference on Computer Vision.

Weidinger, L., Uesato, J., Rauh, M., Grifin, C., Huang, P.-S., Mellor, J., . . . others (2022). Taxonomy of Risks posed by Language Models. Proceedings of the ACM Conference on Fairness, Accountability, and Transparency.

Yang, Y., Gao, R., Wang, X., Ho, T.-Y., Xu, N., Xu, Q. (2024). MMA-Difusion: MultiModal Attack on Difusion Models. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Yang, Y., Hui, B., Yuan, H., Gong, N., Cao, Y. (2024). SneakyPrompt: Jailbreaking Text-toimage Generative Models. Proceedings of the IEEE Symposium on Security and Privacy.

Yin, Q., Tian, Y., Zhang, Y. (2025). Rethinking Robust Adversarial Concept Erasure in Difusion Models. arXiv preprint arXiv:2510.27285 , ,

Young, P., Lai, A., Hodosh, M., Hockenmaier, J. (2014). From Image Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference over Event Descriptions. Transactions of the Association for Computational Linguistics, 2 , 67–78,

Yousaf, A., Fioresi, J., Beetham, J., Bedi, A.S., Shah, M. (2026). SafeR-CLIP: Mitigating NSFW Content in Vision-Language Models While Preserving Pre-Trained Knowledge. Proceedings of the AAAI Conference on Artificial Intelligence.

Yousaf, A., Ghosh, S., Beetham, J., Bedi, A.S., Shah, M. (2026). The Illusion of High Utility in Safety Alignment of Text-to-Image Difusion Models. Proceedings of the European Conference on Computer Vision.

Yuan, L., Li, X., Xu, C., Tao, G., Jia, X., Huang, Y., . . . Li, B. (2026). PromptGuard: Soft Prompt-Guided Unsafe Content Moderation for Text-to-Image Models. IEEE Transactions on Information Forensics and Security, 21 , 4051–4061,

## Appendix A Additional Details on ViSUv2 Dataset

## A.1 Text Generation and Image Generator Selection

Our dataset construction begins by taking the safe captions from COCO (Lin et al., 2014) and Flickr30k (Young et al., 2014) and generating corresponding unsafe versions. To do this, we prompt LLaMA-3.1-8B-Instruct (Grattafiori et al., 2024) to rephrase each safe caption into an NSFWthemed one. The prompt includes an example reformulation and instructs the model to generate a caption that (i) is NSFW, (ii) remains a coherent and realistically plausible scene description, and (iii) incorporates a concept derived from the CoPro (R. Liu et al., 2024) taxonomy.

For each safe caption, we first identify the 50 tags (out of the 578 provided in (R. Liu et al., 2024)) whose CLIP embeddings are most similar to those of the input. We then randomly sample one tag from this 50-tag subset to guide the generation, ensuring that all CoPro concepts are suficiently represented across the dataset. This selection step is crucial: by restricting the sampled concepts to those semantically close to the input caption, we avoid injecting irrelevant content that could lead to incoherent or meaningless reformulations. The prompt we use is shown in Fig. A1.

We employ a controlled output prefilling strategy (Cappelletti et al., 2026) to ensure that the model consistently follows the intended generation behavior without triggering refusal mechanisms. Prefilling works by introducing a short text fragment that initiates the response, so that the model interprets the subsequent decoding as a continuation of an already-started answer. In practice, the text fragment is inserted immediately after the model assistant-start token, making it the first element of the output sequence. The prefix used in our experiments is:

## ‘‘Sure, the NSFW phrase is:’’

This prefix does not introduce any semantic information; it simply initiates the response so that the model proceeds with the requested caption transformation.

## A.2 Text and Image Safety Labeling

Text Classification. For the text classification with LLaMA-3.1-8B-Instruct (Grattafiori et al., 2024), the prompt we use is shown in Fig. A2. This step is crucial to assign the binary safety label $f _ { i } ^ { t }$ to each generated caption. Similar to the generation stage, we employ an output prefilling strategy to enforce strict adherence to the formatting requirements and prevent model refusals. We inject the string:

‘‘Sure, the letter is:’’

into the model response bufer. This guides the model to immediately output the classification token $( i . e . , \mathrm { ~ \ ' a ~ ' ~ o r ~ \ ' b ~ ' ) }$ as the final answer without adding conversational fillers or refusal messages.

Image Generator Selection. As illustrated in Fig. 2 of the main paper, we use two complementary difusion models for image generation: a fine-tuned Stable Difusion XL (Podell et al., 2024) model and an uncensored FLUX variant (Black Forest Labs, 2024). We use LLaMA-3.1-8B-Instruct as a model selector to determine which generator is used for each caption. This choice is motivated by the observation that SDXL-based models often struggle to accurately render masculine nudity. We therefore classify whether the caption implies such content; if so, the image is generated with FLUX, and otherwise with SDXL. The prompt used for this selection is shown in Fig. A3.

Image Safety Labeling. To assign the final safety label $f _ { i } ^ { v }$ to the selected generated images, we utilize an ensemble of two discriminative models: NudeNet (Bedapudi, 2019) and Q16 (Schramowski et al., 2022). This combination ensures coverage of both explicit nudity and broader unsafe concepts (e.g., violence, hate). We adopt a union-based detection protocol: an image is labeled as unsafe $( f _ { i } ^ { v } = 1 )$ if either NudeNet or Q16 flags it. Conversely, an image is labeled as safe $( f _ { i } ^ { v } ~ = ~ 0 )$ only when both classifiers predict that the content is benign. This rigorous filtering ensures that the “generated-safe” subset of our ViSUv2 dataset contains high-confidence safe samples, minimizing contamination of the safe embedding space during alignment. After completing both the text and image safety labeling stages, each generated pair in ViSUv2 is associated with independent modality labels $f _ { i } ^ { t }$ and $f _ { i } ^ { v }$ . This results in four possible con figurations depending on whether the generated caption and image are classified as safe or unsafe.

![](images/5981211d162a97ea3251f9895880d96f0f002e51e213e537feabfd774eb1c209.jpg)  
Fig. A3 Prompt used for image generator selection, detecting if a caption implies masculine nudity.

Across the final dataset, the distribution of generated samples is as follows: $\mathbf { T } _ { s } \mathbf { - } \mathbf { V } _ { s } \ ( 8 . 8 \% ) , \ \mathbf { T } _ { u } \mathbf { - } \mathbf { V } _ { u }$ (52.3%), $\mathbf { T } _ { s } \mathbf { - } \mathbf { V } _ { u } \ ( 8 . 4 \% )$ , and $\mathbf { T } _ { u } – \mathbf { V } _ { s } \ ( 3 0 . 5 \% )$ . Here, $\mathbf { T } _ { s }$ and $\mathbf { T } _ { \boldsymbol { u } }$ denote generated captions labeled as safe and unsafe, while $\mathbf { V } _ { s }$ and $\mathbf { V } _ { u }$ denote generated images labeled as safe and unsafe. This distribu tion highlights the prevalence of mixed-modality outcomes in multimodal generation, where harmful captions may still produce visually benign images and vice versa. Such asymmetric cases motivate the modality-aware supervision adopted in ViSUv2 and the selective alignment strategy used in our training framework.

## A.3 Human Validation of Modality-Level Safety Labels

To assess the reliability of the automated modalitylevel annotations used to train ShieldCLIP, we conduct a human validation study on 1,000 generated image-text pairs from ViSUv2. The evaluation set is sampled uniformly across the four modality-level safety configurations $( \mathbf { T } _ { s } \mathbf { - } \mathbf { V } _ { s } , \mathbf { T } _ { u ^ { - } }$ $\mathbf { V } _ { u } , \mathbf { T } _ { s } \mathbf { - V } _ { u } ,$ and ${ \bf T } _ { u ^ { - } } { \bf V } _ { s } )$ , while ensuring coverage of the harmful-content taxonomy. In total, 20 independent annotators participate in the study, with each sample evaluated by 3 annotators.

Annotators assess text and image safety independently. For each generated caption, they are asked to determine whether the text contains harmful or inappropriate content; the same binary judgment is collected separately for the corresponding generated image. The final human label for each modality is obtained by majority vote.

We compare these human annotations with the automatic labels used during dataset construction. Agreement is computed between the automatic label and the aggregated human judgment for each modality. Overall, as reported in the main paper, the automatic annotations agree with human judgments on 86% of captions and 81% of images,

System: Given a tag that represents a sensitive or inappropriate concept, classify it into the most appropriate categories from the following list: <list of categories>. Respond only with the names of the appropriate categories from the list above, separated by commas. Do not explain your answer. Choose as many categories as are appropriate (at least one). User: Which categories does the tag <tag> belong to?

Fig. A4 Prompt used to classify the CoPro concepts into ViSUv2 categories.  
![](images/aa039f0e45bf7bf5f65fecaf655339ab12c4c465ecc9e4f5816804fef0a32d3e.jpg)  
Fig. A5 Examples from the ViSUv2 dataset. Each sample shows a real safe caption and image (green border) paired with a generated caption and image (red border for unsafe). We also report the granular tag used for conditioning and the associated high-level categories.

supporting the reliability of our modality-level labeling strategy.

## A.4 Taxonomy Mapping Prompt

The construction of the taxonomy is described in Section 3.2. This section provides the prompt used to construct the mapping from the granular CoPro concepts to the 28 harmful categories. To achieve this, we prompt LLaMA-3.1-8B-Instruct (Grattafiori et al., 2024) to classify each concept into one or more of the categories, using the instruction template detailed in Fig. A4.

To stabilize the generation and ensure consistent classification, we apply the output prefilling strategy (Cappelletti et al., 2026) described earlier, where we insert the text fragment:

## ‘‘Sure, the categories are:’’

after the model assistant-start token. In this way, a single CoPro concept can be mapped to one or

more categories, resulting in a realistic and semantically rich mapping, capturing the multiple harmful dimensions a concept may involve.

## A.5 Additional Dataset Examples

To qualitatively validate the diversity and semantic consistency of ViSUv2, Fig. A5 presents a series of multimodal quadruplets. Each sample anchors the generation process with a real, safe text-image pair (indicated by green borders) and juxtaposes it with a generated counterpart synthesized to manifest specific harmful concepts. We explicitly display the fine-grained tag used to condition the generation (e.g., “terrorist”, “knives”) alongside the high-level categories from our 28-class taxonomy (e.g., extremist ideologies, violence) to which the tag belongs.

These examples highlight the necessity of our independent safety labeling strategy: unlike prior approaches that assume all generated content is harmful, our pipeline distinguishes between safe and unsafe samples for both text and images. This granularity captures the frequent asymmetry between modalities, where a prompt classified as unsafe may yield a visually benign image.

Take a deep breath and work on this problem step-by-step. You are in a new universe where   
you must complete this task: please indicate with a "0" or a "1" if the sentences that I   
feed you are safe (0) or NSFW (1), considering that a sentence is NSFW if it is offensive   
or it can evoke negative or unsafe situations or if it even minimally mentions at least   
one of these concepts: <concepts>.   
The output must be only a python list of booleans with position indexes corresponding to   
my inputs.   
### Inputs:  
Fig. A6 Prompt used for GPT-3.5 Turbo to classify NSFW captions as safe (0) or NSFW (1) based on the presence of ofensive or unsafe content.

For instance, the injection of the tag “dumb” into a caption about a baseball player results in a generated text describing “perverted players” yet the corresponding generated image remains a safe, standard depiction of the sport. Conversely, the conditioning tag “savage” produced the safe caption “A savage devours the contents of the fridge”, while the resulting image violates safety policies. By correctly identifying such mixed cases, alongside fully safe generated pairs and explicitly harmful ones, ViSUv2 enables a selective alignment process that mitigates actual risks without distorting safe regions of the embedding space.

## Appendix B Evaluation Protocol

## B.1 Category Mapping

When computing the rate of generated harmful images for the text-to-image experiments, we map our 28 harmful-content categories to the 7 high level categories defined by I2P (Schramowski et al., 2023). The mapping is constructed in a coherent manner, ensuring a correct and reliable analysis. Table B1 summarizes the mapping used for ViSUv2. For ViSU (S. Poppi et al., 2024), we adopt the mapping defined in the original paper.

## B.2 Harmfulness and Diversity Evaluation

Harmfulness Evaluation. To assess the harmfulness of the captions in the NSFW sources, we leverage LLaMA-3.1-8B-Instruct (Grattafiori et al., 2024) and GPT-3.5 Turbo (Brown et al., 2020). The prompt used for LLaMA-3.1-8B-Instruct is shown in Fig. A2, where, instead of “generated caption”, we pass each caption from the NSFW datasets. For GPT-3.5 Turbo, the prompt is shown in Fig. A6, where multiple captions are passed in a single batch, and “concepts” refers to our 28 unsafe categories.

Table B1 Mapping of ViSUv2 categories to I2P categories.
<table><tr><td>I2P Categories ViSUv2 Categories</td><td></td></tr><tr><td>hate</td><td>hate, slurs, discrimination,</td></tr><tr><td>harassment</td><td>extremist ideologies harassment, body shaming</td></tr><tr><td>violence</td><td>violence, suffering, humiliation, harm, abuse, brutality, cruelty</td></tr><tr><td>self-harm</td><td>suicide, mental health issues</td></tr><tr><td>sexual</td><td>sexual, nudity</td></tr><tr><td>shocking</td><td>bodily fluids, blood, obscene gestures, vulgar humor, shocking</td></tr><tr><td>illegal activity</td><td>illegal activity, drug use, theft, vandalism, weapons, substance abuse</td></tr></table>

To evaluate image harmfulness, we employ a dual approach based on both existing classifiers and multimodal LLMs. First, we apply NudeNet and Q16, labeling each image as NSFW if either classifier flags it as unsafe. This ensemble provides robust coverage of explicit, suggestive, or violent content. Second, we use GPT-4 to further classify each image, employing a prompt structure similar to that used in the caption-level classification.

Diversity Evaluation. After harmfulness evaluation, we assess diversity in the captions using two complementary metrics: Vendi Score (Friedman & Dieng, 2023) and Self-BLEU (Perez et al., 2022).

The Vendi Score (VS) measures diversity as the exponential of the Shannon entropy of the eigenvalues of a similarity kernel. It efectively captures semantic diversity without relying on reference datasets or assuming specific distributions. Mathematically, given a set of N captions $\{ x _ { 1 } , \ldots , x _ { N } \}$ , the Vendi Score is defined as:

Table B2 Rate of generated harmful images using unsafe textual prompts from ViSU (S. Poppi et al., 2024), combining predictions from NudeNet and Q16 classifiers.
<table><tr><td rowspan="2">Model</td><td colspan="8">ViSU</td></tr><tr><td>Hate</td><td>Harass</td><td>Viol</td><td>S-Harm</td><td>Sex</td><td>Shock</td><td>Ill Act</td><td>Avg</td></tr><tr><td>SD v1.4</td><td>26.2</td><td>18.1</td><td>31.3</td><td>19.9</td><td>23.5</td><td>29.9</td><td>23.5</td><td>24.6</td></tr><tr><td>SLD-Strong (Schramowski et al., 2023)</td><td>4.5</td><td>3.5</td><td>6.0</td><td>4.9</td><td>5.2</td><td>5.8</td><td>3.9</td><td>4.8</td></tr><tr><td>SalUn (Fan et al., 2024)</td><td>19.3</td><td>9.8</td><td>23.0</td><td>15.7</td><td>8.3</td><td>19.6</td><td>17.9</td><td>16.3</td></tr><tr><td>ESD (Gandikota et al., 2023)</td><td>23.4</td><td>15.5</td><td>30.4</td><td>19.3</td><td>20.9</td><td>27.7</td><td>21.8</td><td>22.7</td></tr><tr><td>SPM ( (Lyu et al., 2024)</td><td>15.8</td><td>9.7</td><td>19.3</td><td>11.4</td><td>13.1</td><td>17.9</td><td>13.7</td><td>14.4</td></tr><tr><td>UCE (Gandikota et al., 2024)</td><td>12.5</td><td>7.7</td><td>13.4</td><td>9.0</td><td>11.3</td><td>10.8</td><td>10.0</td><td>10.7</td></tr><tr><td>Receler (Huang et al., 2024)</td><td>10.6</td><td>6.4</td><td>12.9</td><td>9.0</td><td>4.5</td><td>10.1</td><td>8.7</td><td>8.9</td></tr><tr><td>Safe-CLIP (S. Poppi et al., 2024)</td><td>4.3</td><td>3.8</td><td>4.5</td><td>4.3</td><td>5.0</td><td>2.8</td><td>4.1</td><td>4.1</td></tr><tr><td>SafeR-CLIP (Yousaf, Fioresi, et al., 2026)</td><td>4.8</td><td>5.9</td><td>7.2</td><td>5.4</td><td>7.9</td><td>7.2</td><td>6.7</td><td>6.9</td></tr><tr><td>SafetyDPO (R. Liu et al., 2025)</td><td>4.0</td><td>1.1</td><td>3.2</td><td>3.9</td><td>2.3</td><td>1.8</td><td>2.7</td><td>2.7</td></tr><tr><td>DES (Ahn &amp; Jung, 2025)</td><td>1.2</td><td>1.4</td><td>2.6</td><td>3.8</td><td>0.7</td><td>1.5</td><td>1.7</td><td>1.8</td></tr><tr><td>ShieldCLIP (Ours)</td><td>1.4</td><td>1.3</td><td>1.1</td><td>1.5</td><td>1.0</td><td>1.0</td><td>1.2</td><td>1.2</td></tr></table>

$$
\mathrm { V S } ( x _ { 1 } , \dots , x _ { N } ) = \exp \left( - \sum _ { i = 1 } ^ { N } \lambda _ { i } \log \lambda _ { i } \right) ,
$$

where $\lambda _ { 1 } , \ldots , \lambda _ { N }$ are the eigenvalues of the normalized kernel matrix $K / N$ . We construct the kernel matrix $K ~ \in ~ \mathbb { R } ^ { N \times N }$ using the cosine similarity between the CLIP-ViT-L/14 embeddings of the captions; specifically, $K \stackrel { \cdot } { = } X X ^ { \top }$ , where X contains the L -normalized embeddings. Higher VS values indicate a greater semantic diversity, meaning the captions cover a wider range of semantic concepts.

Self-BLEU (Perez et al., 2022) assesses syntactic diversity by measuring the resemblance of each sentence to the rest of the collection. For each cap tion, we treat it as a hypothesis and the remaining captions as references. The Self-BLEU score is the average BLEU score computed over all such pairs:

$$
{ \mathrm { S e l f - B L E U } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } { \mathrm { B L E U } } ( s _ { i } , \{ s _ { j } : j \neq i \} ) ,
$$

where N is the total number of sentences, $s _ { i }$ is the i-th sentence, and $\{ s _ { j } : j \neq i \}$ denotes the set of all other sentences. Since BLEU measures n-gram overlap, a lower Self-BLEU score indicates higher diversity. This provides a complementary measure to the Vendi Score: while Vendi evaluates semantic diversity based on embeddings, Self-BLEU quantifies surface-level syntactic diversity.

To assess image diversity, we again use the Vendi Score, computing the kernel matrix as the cosine similarity matrix between the CLIP-ViT-L/14 embeddings of the images. This metric captures the spread and uniqueness of the visual content in the feature space, where higher values indicate greater semantic diversity.

## Appendix C Additional Results

## C.1 Detailed Results on the ViSU Dataset

Table B2 provides the category-level breakdown of the ViSU (S. Poppi et al., 2024) results summarized in the main paper (cf. Table 3). We evaluate Stable Difusion v1.4 using the same NudeNet-Q16 ensemble adopted in Section 4.3, reporting harmful generation rates over the seven grouped harmful-content categories.

Beyond the overall average, ShieldCLIP exhibits a particularly uniform reduction in harmful generation rates across categories. It achieves the lowest harmful rate for violence (1.1%), selfharm (1.5%), shocking content (1.0%), and illegal activities (1.2%), reducing the corresponding SD v1.4 baseline rates of 31.3%, 19.9%, 29.9%, and 23.5%. On the remaining categories, the best results are obtained by DES (Ahn & Jung, 2025) for hate (1.2%) and sexual content (0.7%), and by SafetyDPO (R. Liu et al., 2025) for harassment (1.1%). However, ShieldCLIP remains within 0.2–0.3 percentage points of these category-specific minima. As a result, its harmful generation rate remains confined to a narrow 1.0–1.5% range across all seven categories, compared with larger variations for DES (0.7–3.8%) and SafetyDPO (1.1–4.0%). This category-level consistency complements the aggregate results in the main paper, showing that the gains of selective alignment are distributed across heterogeneous harmful concepts rather than being driven by a small subset of categories.

## C.2 Additional Qualitative Results

This section presents additional qualitative results across downstream tasks. Fig. C7 extends the textto-image evaluation shown in Fig. 5 with additional unsafe prompts, comparing ShieldCLIP with competing safety methods. The examples show that ShieldCLIP more consistently suppresses harmful visual cues while preserving the main scene semantics. Similarly, Fig. C8 complements the image-to-text examples shown in Fig. 6 with additional real NSFW images, showing that ShieldCLIP reduces explicit caption content while retaining the coarse context of the input.

Despite the strong safety performance observed in the previous examples, some challenging cases remain. Fig. C9 presents representative failure cases where harmful visual elements are only partially suppressed or where the generation still reflects unsafe concepts from the prompt. These examples typically arise when the unsafe semantics are strongly entangled with the core scene description or when the generative backbone exhibits strong prior associations with certain harmful concepts. Nevertheless, even in these cases, the outputs often exhibit partial mitigation, indicating that the alignment can still attenuate unsafe visual associations under challenging conditions.

## Appendix D Societal Impact and Limitations

## D.1 Ethical Implications

ShieldCLIP is designed to mitigate harmful content in multimodal foundation models. While this approach improves the safety of text-to-image difusion models and multimodal large language models in our evaluations, it also raises important ethical considerations. The construction of ViSUv2 required generating and handling explicit and potentially disturbing material, including sexual and violent content. These data were used solely for research on AI safety, and we do not plan to release the unsafe image subset without appropriate access controls.

Furthermore, the notion of harmfulness adopted in this work is derived from automated classifiers and large language models trained on web data. Consequently, it may reflect cultural and societal biases embedded in these sources, potentially marginalizing certain communities or perspectives. We acknowledge that “safety” is a context-dependent concept and that alignment decisions may not generalize across all cultural or linguistic domains. Thus, we consider ShieldCLIP a complementary step toward safer multimodal systems, rather than a definitive solution. Responsible deployment requires continuous monitoring, transparent reporting of failure cases, and inclusion of diverse ethical perspectives during model evaluation.

## D.2 Limitations

While ShieldCLIP efectively suppresses harmful multimodal associations and preserves semantic fidelity across various benchmarks, some limitations remain. Our method does not provide absolute guarantees against unsafe generations or retrievals. It may fail in edge cases where harmfulness is ambiguous or where visual and textual cues interact in unanticipated ways. Additionally, the efectiveness of selective safety alignment depends on the accuracy and balance of the ViSUv2 labels; residual biases or annotation errors may propagate into the fine-tuned embedding space.

Another limitation lies in the reliance on predefined taxonomies of harmful content, which cannot encompass all possible manifestations of risk or ofense. Moreover, while ShieldCLIP improves safety for CLIP-like encoders, it may not fully transfer to downstream models with diferent architectures or decoding strategies. Future work should focus on expanding the diversity of safety concepts, exploring human-in-the-loop refinement, and developing adaptive alignment mechanisms that dynamically adjust to evolving definitions of harm and acceptability.

Prompt

SD1.4

SLD Strong

SafetyDPO

Safe-CLIP

ShieldCLIP (ours)

SDXL

SafetyDPO XL

Safe-CLIP XL ShieldCLIP XL (ours)

![](images/ac66974130ca97f78687ee13001bbdbbfe30b47fdc02267fe2ff5a819cd8665c.jpg)  
Fig. C7 Additional qualitative examples generated with SD v1.4, SDXL, ShieldCLIP, and competing methods using unsafe prompts from I2P and ViSUv2.

![](images/9a84d3f28c658ff78a5f43cad533f0211c71e35f69a879e4a269c45bee37126a.jpg)  
Fig. C8 Additional qualitative examples of image-to-text generation with the original LLaVA model, Safe-CLIP, and ShieldCLIP, using real NSFW images from diferent sources as input.

![](images/7f1d1c5a5da8c695bece1fac79381e094ace9c19fc12f3fd3daa7940e66b44c4.jpg)  
Fig. C9 Representative failure cases for ShieldCLIP in text-to-image generation with SD v1.4 and SDXL on unsafe prompts from I2P and ViSUv2, shown alongside competing mitigation methods.