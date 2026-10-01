# Steering Fields: Adaptive Vector Fields for Safe Image Generation and Beyond

Simone Facchiano<sup>1,2</sup>, Jan Eric Lenssen<sup>1</sup>, Bernt Schiele<sup>1</sup>, Wolfgang Stammer<sup>1</sup>, Fabio Galasso<sup>2∗</sup>, Jonas Fischer<sup>1∗</sup>

<sup>1</sup>Max Planck Institute for Informatics, <sup>2</sup>Sapienza University of Rome

<sup></sup> SteeringFields: Adaptive Vector Fields for Safe Image Generation and Beyond

T2I Generation with Safety Steering

![](images/f9dc9ce9c652d86c466ed326b6260a1c1fa5e91259b7eee468d04bccfde3408f.jpg)  
Figure 1: Steering Fields improve Safe Image Generation. We propose Steering Fields as a generalization of steering vectors that adapts steering directions in the latent space locally across space and time (left), blending between original (source) and desired (target) vector fields of a flow model. Among other applications, it provides a new state-of-the-art for safety steering in image generation (see examples on the right).

## Abstract

As state-of-the-art text-to-image flow models achieve near-photorealistic quality, controlling their outputs, e.g., suppressing harmful content while promoting benign alternatives, has become a central challenge. The current steering paradigm consists of adding a global steering vector to selected activations. While functional, a fixed and example-agnostic vector applied uniformly along the entire trajectory cannot adapt to the changing state of the generation and often causes unintended global changes. We introduce Steering Fields, a generalization of steering vectors that adaptively re-estimates the steering direction at each step of the generative process. Steering Fields operate on the noisy states of flow models, expose a continuous trade-off between steering strength and content preservation, and are compositional, enabling the simultaneous induction and inhibition of concepts, setting a new state of the art on safety steering benchmarks. Despite using no explicit spatial masks or object priors, the trajectory-adaptive estimation naturally preserves local structure, in a manner reminiscent of image editing. In fact, Steering Fields can serve as a structure-preserving image-editing technique that achieves state-of-the-art semantic fidelity (CLIP, VQAScore), while remaining model-agnostic and inversion-free.

## 1 Introduction and Background

Generative modeling for vision has long been driven by a single goal: making images as realistic as possible given input text (Karras et al., 2020; Saharia et al., 2022; Peebles and Xie, 2023). Modern flow-based models have all largely closed this gap (Labs et al., 2025; Kamali et al., 2025; Esser et al., 2024), and the bottleneck has shifted from how to generate better images to how to control what is being generated.

Steering has emerged as a particularly attractive paradigm for control (Turner et al., 2023; Arditi et al., 2024; Rimsky et al., 2024; Konen et al., 2024; Rodriguez et al., 2024, 2025): a steering vector is a direction in the activation space of a generative model that, when added to selected activations during inference, biases generation toward or away from a target concept, without retraining and without modifying the architecture. Steering is especially relevant for safety-critical settings – inhibiting harmful content such as violence or nudity while preserving overall generative capacity (Gandikota et al., 2023; Shen et al., 2026) – and can also induce attributes such as style, composition, or specific semantic content (Konen et al., 2024; Shen et al., 2020). Despite its appeal, the current steering paradigm rests on a brittle assumption: a single, fixed steering vector is applied uniformly throughout the generation, agnostic to the specific generation example and to changes over the trajectory. As generation progresses from noise to coarse structure and fine detail, a constant shift cannot match the local geometry of the activation manifold, and classical steering often induces uncontrolled global changes beyond the target concept, damaging the composition and local structure of the generated image (Im and Li, 2025; Mayne et al., 2024; Facchiano et al., 2026).

In this work, we address this limitation by lifting steering from a fixed vector to a vector field. We introduce Steering Fields, a generalization of steering vectors that adaptively re-estimates the steering direction at each step of the generative process, conditioned on the current state of the trajectory. Steering Fields are model-agnostic – they apply to any flow model regardless of architecture – and operate on latent state representations rather than on a hand-picked layer or module. A single parameter controls the trade-off between steering strength and content preservation, and they are compositional: the same formulation supports both inducing desired concepts and inhibiting undesired ones, through complementary attraction and repulsion mechanisms. Mathematically, classical activation steering is recovered as the special case of a constant field.

A perhaps surprising property emerges from this formulation. Although Steering Fields prescribe no explicit spatial mask, no object-level prior and no structural regularizer, the per-step adaptivity along the trajectory naturally preserves the local structure of the generated image. This behavior is reminiscent of image editing (Meng et al., 2022; Hertz et al., 2022; Brooks et al., 2023; Kulikov et al., 2025; Huang et al., 2024; Zarei et al., 2026), a related but distinct line of work that takes an existing image as input and modifies it to match a target description while preserving the rest of the image. Editing methods typically rely on inversion (Mokady et al., 2023; Song et al., 2021), attention manipulation (Hertz et al., 2022), or explicit spatial masks and latent-region constraints (Huang et al., 2024; Zarei et al., 2026) to anchor structure, whereas Steering Fields achieve structure preservation as an emergent consequence of trajectory-adaptive control, without inversion and without explicit anchors. In Steering Fields the same mechanism that steers generation also edits an input image, enabling direct comparison with the editing literature on standard benchmarks. While image editing is not the primary aim of our framework, applying the same trajectory-adaptive mechanism to this setting yields particularly strong semantic adherence, measured by CLIP (Radford et al., 2021) and VQAScore (Lin et al., 2024), confirming that this adaptive control faithfully implements the requested change.

We empirically evaluate Steering Fields on Stable Diffusion 3.5 (Esser et al., 2024) and FLUX1 (Labs et al., 2025). On safety steering benchmarks, our method establishes a new state of the art, substantially reducing the rate of unsafe generations while preserving prompt fidelity. On image editing benchmarks, Steering Fields are competitive with dedicated editing methods despite not being designed for this task, and lead the field on semantic-adherence metrics.

Our contributions can be summarized as follows:

• Steering Fields. We introduce Steering Fields, a model-agnostic, trajectory-adaptive generalization of classical steering vectors, and show analytically that activation steering is recovered as the constant-field special case.

• Compositional control. Within a single formulation, Steering Fields support both induction and inhibition of concepts via complementary attraction–repulsion mechanisms, with a continuous, user-controllable trade-off between steering strength and structure preservation.

• Editing as a byproduct. Steering Fields naturally extend to image editing without inversion, without spatial masks, and without architecture-specific machinery.

• Thorough Empirical Evaluation. We evaluate Steering Fields on Stable Diffusion 3.5 and FLUX1, achieving state-of-the-art performance on safety steering benchmarks and state-of-the-art semantic adherence (CLIP, VQAScore) on image editing benchmarks.

## 2 Background and Related Works

Generative models for image synthesis, like Diffusion (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021, 2020) or Flow Matching models (Lipman et al., 2023; Liu et al., 2022; Liu, 2022), learn a mapping from noise to the complex distribution of real images. Flow Matching learns a time-dependent vectorfield $v _ { \theta } ( z , t , | c )$ that specifies how a latent state z should evolve at a specified timestep t, conditioned on some text prompt c. As a result, generation in Flow Matching can be interpreted as following trajectories governed by the learned vector field, starting from random noise. This formulation is the foundation of state-of-the-art models, including FLUX (Labs et al., 2025) and Stable Diffusion 3 (Esser et al., 2024).

Activation Steering is the predominant approach for steering generative models (Turner et al., 2023; Rimsky et al., 2024; Konen et al., 2024; Rodriguez et al., 2024) operating on their latent representations. Standard approaches build on the Linear Representation Hypothesis (Mikolov et al., 2013; Park et al., 2023; Elhage et al., 2022), steering the model at inference time by linearly adding or subtracting a single direction (Arditi et al., 2024) that encodes the desired concept from the residual stream $h _ { l } ^ { s r c . }$

$$
\tilde { h } _ { l } = h _ { l } ^ { s r c } + \alpha r ^ { \Delta } .\tag{1}
$$

This fixed direction is usually pre-computed via difference-of-means between samples that contain the target concept (tar) and samples that do not (away) (Turner et al., 2023; Marks and Tegmark, 2024; Rimsky et al., 2024). The steering vector is therefore computed as:

$$
r ^ { \Delta } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } h _ { l } ^ { t a r } - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } h _ { l } ^ { a w a y } = d _ { l } ^ { t a r } - d _ { l } ^ { a w a y } .\tag{2}
$$

Activation steering is successfully deployed in safety applications (Cao et al., 2025; Arditi et al., 2024; Stolfo et al., 2025) for suppressing generation of harmful concepts. It is a lightweight approach that does not require any re-training, introduces minimal computational overhead and was shown to be effective on a wide range of architectures, from LLMs (Turner et al., 2023; Rimsky et al., 2024) to diffusion- and flow-based models (Facchiano et al., 2026; Briglia et al., 2026). Its simplicity is a blessing and a curse: while $r ^ { \Delta }$ is extremely lightweight, it is fixed once and applied uniformly, despite the underlying generative trajectory is curved (Liu, 2022; Lipman et al., 2023) and the conditional velocity that drives it is a function of $( z , t )$ . Consequently, the corrective direction that minimizes off-target distortion need not remain constant from noise to image. In contrast, we generalize the steering vector to a steering vector field $r ^ { \Delta } ( z , t )$ defined as a function of the current latent state z and trajectory point t. As a consequence, the field allows us to integrate a full steered generation trajectory. We develop and analyze this object in Sec. 3.

Image editing is a broad and rapidly evolving domain encompassing diverse paradigms: perturbationbased methods (Meng et al., 2022), attention manipulation (Hertz et al., 2022; Tumanyan et al., 2023), instruction-based editing (Brooks et al., 2023; Zhang et al., 2025), and large foundation models for unified editing (Labs et al., 2025). A common thread in many state-of-the-art methods is the use of inversion to anchor the edited output to the source image structure (Mokady et al., 2023; Ju et al., 2024; Avrahami et al., 2025). More recent work avoids explicit inversion: FlowEdit (Kulikov et al., 2025) minimizes the transport cost in latent space between source and target by jointly manipulating the forward and reverse trajectories, while InfEdit (Xu et al., 2024) achieves inversion-free editing through consistency models. Multi-aspect editing poses additional challenges, as single-branch approaches accumulate errors across edits. ParallelEdits (Huang et al., 2024) addresses this with a multi-branch diffusion design. A complementary direction explores continuous, fine-grained control over edit strength via slider-based mechanisms (Gandikota et al., 2024a; Baumann et al., 2025; Zare et al., 2026), exposing smooth interpolation between edit intensities. Steering Fields, in contrast, unify concept induction and inhibition within a single trajectory-adaptive framework that also extend naturally to image editing: conditioning the source velocity on an encoded image latent rather than on noise turns the same operator into a structure-preserving image editor, without inversion, attention manipulation, or spatial masks.

FLUX (vanilla)  
FLUX + UCE  
FLUX + ESD  
FLUX + EraseAnything  
FLUX + Ours  
![](images/6cc5de72db04deaf45ee6875c07a15eedec5946c1de01d9c42f1eef73b8b259b.jpg)

![](images/8d26f8a4cf6ac4cca974f50b44721b421bc7bda8ef3f528ed0396b7b01ec3bd6.jpg)

(a) Qualitative comparison on Ring-a-Bell using FLUX1.  
![](images/d090756335b6c2caf8115e54618cdcfa0f486857a567735fb80d2e7bac0cc37e.jpg)  
(b) VQAScore vs. NudeNet on FLUX1 and SD3.5.  
Figure 2: Comparison of safety steering methods on Ring-a-Bell. Left: qualitative comparison using FLUX1. Right: trade-off between NSFW suppression and semantic fidelity on FLUX1 (top) and SD3.5 (bottom).

## 3 Steering Vector Fields

In this section, we introduce Steering Fields, which operate on the latent space of a flow model and unifies attraction to desired and repulsion from undesired concepts in a single closed-form objective (Sec. 3.2). We discuss the relation of Steering Fields with classical Activation Steering in Sec. 3.3 following notation of Kulikov et al. (2025) and derivations in Appendix A.

## 3.1 Setup

We work with text-to-image flow models such as Stable Diffusion 3.5 (Esser et al., 2024) and FLUX1 (Labs et al., 2025), which learn a conditional velocity field $V ( z , t \mid c ) \equiv v _ { \theta } ( z , t \mid c )$ defined over a latent state z, time $t \in [ 0 , 1 ]$ , and a text prompt c. Sampling integrates V from a noise initialisation to a clean latent that is decoded by a VAE into an image. We assume the standard prompting interface, in which a source prompt $c _ { \mathrm { s r c } }$ describes the user’s request.

In Steering Fields, we assume the optional target prompt $c _ { \mathrm { t a r } }$ specifies a concept to induce, and an optional negative prompt $c _ { \mathrm { a w a y } }$ specifies a concept to suppress. With a slight abuse of notation we write $v _ { \bullet } \equiv \bar { V } ( z , t \mid c _ { \bullet } )$ for • ∈ {src, tar, away}; the dependence on the current $( z , t )$ is left implicit.

## 3.2 A unified objective for compositional control

At every integration step we seek a steered velocity $v ^ { * }$ that is simultaneously close to the source field, attracted toward the target, and repelled from the undesired concept. We encode these three requirements in the single quadratic objective:

$$
{ \mathcal L } ( v ) = \| v - v _ { \mathrm { s r c } } \| ^ { 2 } + \mu \| v - v _ { \mathrm { t a r } } \| ^ { 2 } - \lambda \| v - v _ { \mathrm { a w a y } } \| ^ { 2 } ,\tag{3}
$$

with $\mu , \lambda \geq 0$ and $1 + \mu - \lambda > 0$ , which suffices to make $\mathcal { L }$ strictly convex in v. The first term anchors the trajectory to the source generation; the second pulls it toward the target; the third actively

pushes it away from the undesired concept. Setting $\nabla _ { v } \mathcal { L } = 0$ yields the closed-form minimiser

$$
v ^ { * } = \frac { v _ { \mathrm { s r c } } + \mu v _ { \mathrm { t a r } } - \lambda v _ { \mathrm { a w a y } } } { 1 + \mu - \lambda } .\tag{4}
$$

Rewriting (4) as a perturbation of the source velocity exposes its structure as a steering operation:

$$
v ^ { * } = v _ { \mathrm { s r c } } + \underbrace { \frac { \mu } { 1 + \mu - \lambda } } _ { \alpha } \bigl ( v _ { \mathrm { t a r } } - v _ { \mathrm { s r c } } \bigr ) - \underbrace { \frac { \lambda } { 1 + \mu - \lambda } } _ { \beta } \bigl ( v _ { \mathrm { a w a y } } - v _ { \mathrm { s r c } } \bigr ) ,\tag{5}
$$

where $\alpha \geq 0$ governs the strength of attraction toward the target and $\beta \geq 0$ the strength of repulsion from the undesired concept. The two coefficients weight the influence of the two Steering Fields, $v _ { t a r }$ and $v _ { a w a y }$ , to induce one concept while inhibiting another, with a continuous trade-off against fidelity to the source trajectory. The full Steering Fields algorithm is detailed in App. A.4.

We adopt this paradigm for safety steering, which we frame as replacing the away with the tar concept. Referring to Fig. 1 where the src text is "... the subject posed nude...", safety steering is achieved by assigning as away the concept of "nude" and as tar the concept of "clothed". Both tar and away concepts are average embeddings of 50 "nude" and "clothed" prompts (see App. E.1).

Beyond steering. Setting $\lambda = 0$ in (5) yields the pure-attraction (blending) regime,

$$
\begin{array} { r } { v ^ { \ast } \ = \ v _ { \mathrm { s r c } } + \alpha \left( v _ { \mathrm { t a r } } - v _ { \mathrm { s r c } } \right) = v _ { \mathrm { s r c } } + \alpha \ v ^ { \Delta } , \qquad \alpha = \frac { \mu } { 1 + \mu } , } \end{array}\tag{6}
$$

which we use whenever the goal is to induce a concept — a style, attribute, or object — without explicitly removing another. Eq. (6) can be equivalently rewritten as

$$
\begin{array} { r } { v ^ { \ast } = ( 1 - \alpha ) v _ { \mathrm { s r c } } + \alpha v _ { \mathrm { t a r } } \qquad \alpha = \frac { \mu } { 1 + \mu } , } \end{array}\tag{7}
$$

which is the convex interpolation between the source and the target velocity field.

## 3.3 Per-step adaptivity and the relation to activation steering

Eq. 6 mirrors the classical activation-steering update of Eq. 1, with $v ^ { \Delta } \equiv v _ { \mathrm { t a r } } - v _ { \mathrm { a w a y } }$ playing the role of the residual-stream direction $r ^ { \Delta }$ . The crucial distinction is that $v ^ { \Delta }$ is not a pre-computed constant, but rather a function of the current latent and timestep, re-evaluated at each step t on the new state z:

$$
v ^ { \Delta } ( z , t ) ~ = ~ V ( z , t \mid c _ { \mathrm { t a r } } ) - V ( z , t \mid c _ { \mathrm { a w a y } } ) ,\tag{8}
$$

This per-step re-estimation is not an arbitrary modelling choice but a property dictated by the flowmatching paradigm itself. The conditional velocity is by construction a function of $( z , t , c ) \colon$ the same prompt induces meaningfully different velocities at different points of the trajectory, because the local geometry of the latent manifold changes as the sample moves from noise toward image (Lipman et al., 2023; Liu, 2022; Liu et al., 2022). Difference-of-means estimators in activation space (Turner et al., 2023; Rimsky et al., 2024; Arditi et al., 2024) collapse this dependence into a single direction. Such an estimator would be optimal only if a single direction were correct everywhere along the trajectory; in practice, flow trajectories are rarely straight and are most fragile early in generation, when the latent is dominated by noise and bears little resemblance to the fully-formed activation distribution from which $r ^ { \Delta }$ was computed. Steering Fields recover the dependence on $( z , t )$ at no additional training cost: at each integration step, $\bar { v ^ { \Delta } }$ is obtained from two forward passes of the same flow model, one conditioned on $c _ { \mathrm { s r c } }$ and one on $c _ { \mathrm { t a r } }$ (and analogously for the repulsion term with away). The resulting vector is a better estimate by construction at every state visited along the trajectory, rather than only on average.

Latent rather than activation space. A second consequence of (4) is that the intervention is applied to the same latent variable $z _ { t }$ on which the flow model is already defined, rather than to the residual stream of a hand-picked layer. This makes Steering Fields strictly model-agnostic: any flow model exposing a conditional velocity is steerable, irrespective of whether its backbone is a UNet, a DiT, or an MM-DiT.

The same mechanism extends to image editing: conditioning the source velocity on the encoded latent of a real image rather than on noise turns Eq. (4) into an image-to-image operator that we exploit in 4.3, without inversion, attention surgery, or explicit spatial masks.

![](images/c5770542cf67990b13b292acb9442760eded9008e53f102d4e710c9bd642c0a7.jpg)  
Figure 3: The Steering Fields formulation enables a natural extension to image editing tasks. Evalu ated on the PieBench dataset, Steering Fields achieves state-of-the-art semantic adherence performance, measured by CLIP and VQAScore, as well as a state-of-the-art HPSv2 score (see Tab. 2).

## 4 Experimental Evalutations

As use-cases for Steering Fields we consider safety steering in T2I generation and image editing in an I2I setup.

## 4.1 Experimental Setup

Models. We consider two state-of-the-art flow-based T2I architectures: FLUX1 (Labs et al., 2025) and Stable Diffusion 3.5 (Esser et al., 2024).

Baselines. For the safety steering task, we compare against UCE (Gandikota et al., 2024b), ESD (Gandikota et al., 2023), EraseAnything (Gao et al., 2025), and SAFREE (Yoon et al., 2025). For the image editing task, we compare against baseline FLUX Img2Img, RF-Inversion (Rout et al., 2025), StableFlow (Avrahami et al., 2025), and FlowEdit (Kulikov et al., 2025).

Metrics. For safety steering we report NudeNet (notAI tech, 2019) detection rate as the primary safety metric, and VQAScore as a complementary semantic safety measure. VQAScore<sub>detect</sub> is a VQAScore-based (Lin et al., 2024) metric that evaluates the probability P(“Yes” | “Does this figure show nudity? Please answer yes or no.”), therefore measuring the semantic presence of nudity-related concepts independently of pixel-level detectors. Semantic retention on benign prompts is evaluated on 1000 COCO prompts using CLIP (Radford et al., 2021), VQAScore (Lin et al., 2024), and FID (Heusel et al., 2017). For image editing we report CLIP-txt, CLIP-img, CLIP-dir (Avrahami et al., 2025; Görgün et al., 2026), and VQAScore, focusing on semantic alignment with the target prompt rather than pixel-level source preservation, following the standard question template P(“Yes” | “Does this figure show {prompt}? Please answer yes or no.”). All the experiments are conducted over 64 different seeds to ensure statistical robustness, and quantitative results are reported as means with 95% confidence intervals.

Benchmarks. Safety steering is evaluated on Ring-a-Bell (Tsai et al., 2024), consisting of 79 unsafe prompts, and P4D (Chin et al., 2024), consisting of 151 adversarial prompts specifically crafted to bypass safety mechanisms. Image editing is evaluated on the PieBench++ benchmark (Huang et al.; Ju et al., 2024), which consists of 700 images and prompts across nine edit categories.

## 4.2 Safety Evaluation via Steering

We evaluate Steering Fields on prevention of NSFW content generation on Ring-a-Bell and P4D. All experiments are conducted on 64 seeds and we report mean values with 95% confidence intervals.

Figure 2a qualitatively illustrates the core advantage of Steering Fields over existing state-of-theart methods on samples from Ring-a-Bell. In these examples, the base model generates explicit content, as do the other methods, which additionally alter the geometric structure of the image. In contrast, Steering Fields suppress the generation of explicit content while preserving the semantics,

Table 1: Steering towards safe content (T2I) and semantic retention on COCO (1k). NudeNet and $\mathrm { V Q A S c o r e _ { d e t e c t } }$ measure NSFW suppression on Ring-a-Bell and P4D. CLIP, VQAScore, and FID measure retention on COCO-1k, where values close to the base model are desirable.
<table><tr><td rowspan="2">Method</td><td colspan="2">Ring-a-Bell</td><td colspan="2">P4D</td><td colspan="3">COCO-1k (retain)</td></tr><tr><td>NudeNet↓</td><td> $\mathrm { V Q A S c o r e _ { d e t } } \downarrow$ </td><td> $\mathrm { N u d e N e t } \downarrow$ </td><td> $\mathrm { V Q A S c o r e _ { d e t } } \downarrow$ </td><td>CLIP↑</td><td>VQAScore↑</td><td>FID↓</td></tr><tr><td>FLUX</td><td> $7 0 . 6 7 \pm 1 . 7 0$ </td><td> $0 . 7 9 \pm . 0 0 7$  0</td><td> $5 5 . 6 7 \pm 1 . 8 5$ </td><td> $0 . 6 4 \pm . 0 1 4$  一</td><td> $0 . 3 1 \pm . 0 0 0 3$ </td><td> $0 . 8 9 \pm . 0 0 1 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td> $\mathrm { F L U X + U C E }$ </td><td> $7 2 . 2 5 \pm 1 . 8 3$ </td><td> $0 . 7 4 \pm . 0 0 6$ </td><td> $5 6 . 1 3 \pm 1 . 6 4$ </td><td> $0 . 5 5 \pm . 0 1 0$ </td><td> $0 . 3 1 \pm . 0 0 0 9$ </td><td> $0 . 8 9 \pm . 0 0 0 8$ </td><td> $3 8 . 3 1 \pm . 2 7$ </td></tr><tr><td> $\mathrm { F L U X + E S D }$ </td><td> $4 1 . 9 3 \pm 2 . 7 4$ </td><td> $0 . 6 8 \pm . 0 1 3$ </td><td> $3 0 . 9 7 \pm 2 . 4 0$ </td><td> $0 . 5 7 \pm . 0 1 9$ </td><td> $0 . 3 0 \pm . 0 0 0 5$ </td><td> $0 . 8 6 \pm . 0 0 1 9$ </td><td> $4 7 . 6 2 \pm . 5 6$ </td></tr><tr><td> $\mathrm { F L U X + E A }$ </td><td> $6 2 . 9 8 \pm 1 . 9 5$ </td><td> $0 . 7 4 \pm . 0 1 0$ </td><td> $3 8 . 9 2 \pm 2 . 3 6 $ </td><td> $0 . 5 1 \pm . 0 2 0$ </td><td> $0 . 3 1 \pm . 0 0 0 3$ </td><td> $0 . 8 8 \pm . 0 0 4 9$ </td><td> $2 8 . 5 4 \pm 0 . 3 3$ </td></tr><tr><td>FLUX + Steering Fields</td><td> $3 7 . 3 2 \scriptstyle \pm 1 . 9 6$ </td><td> $\mathbf { 0 . 5 2 \Pi _ { \pm . 0 1 2 } }$ </td><td> $2 4 . 4 3 \scriptstyle \pm 1 . 4 6$ </td><td> $\mathbf { 0 . 3 3 \ : \pm . 0 1 3 }$ </td><td> $0 . 3 1 \pm . 0 0 0 3$ </td><td> $0 . 8 9 \pm . 0 0 1 1$ </td><td> $3 4 . 1 6 \pm 0 . 3 9$ </td></tr><tr><td>SD3.5</td><td> $4 7 . 3 3 \pm 1 . 2 2$  </td><td> $0 . 7 8 \pm . 0 0 3 3$ </td><td> $4 4 . 2 7 \pm \mathrm { 0 . 8 0 }$  </td><td> $0 . 6 7 \pm . 0 0 3 7$  一</td><td> $0 . 3 2 \pm . 0 0 0 1$  </td><td> $0 . 9 2 \pm . 0 0 0 6$  </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td> $\mathrm { S D 3 . 5 + S A F R E E }$ </td><td> $3 6 . 1 4 \pm 1 . 0 5$ </td><td> $0 . 6 8 \pm . 0 0 3 5$ </td><td> $3 3 . 3 1 _ { \pm 0 . 6 6 }$ </td><td> $0 . 5 2 \pm . 0 0 3 9$ </td><td> $0 . 3 2 \pm . 0 0 0 1$ </td><td> $0 . 9 0 \pm . 0 0 0 7$ </td><td> $4 5 . 9 7 \pm 0 . 1 1$ </td></tr><tr><td> $\mathbf { S D 3 . 5 + S t e e r i n g }$  Fields</td><td> ${ \bf 2 9 . 0 2 \pm 1 . 7 4 }$ </td><td> $\mathbf { 0 . 5 9 \ : \pm . 0 1 0 0 }$ </td><td> $2 3 . 3 9 \pm 1 . 4 8$ </td><td> $\mathbf { 0 . 4 1 \bot _ { \Omega } } _ { \textrm { \tiny { \left( 0 . 0 1 1 6 \right. } } }$ </td><td> $0 . 3 2 \pm . 0 0 0 3$ </td><td> $0 . 9 2 \pm . 0 0 0 6$ </td><td> $4 6 . 3 8 \pm 0 . 2 1$ </td></tr></table>

the geometric structure and artistic style specified by the prompts. Additional examples are shown in Fig. 9.

Table 1 support these observations quantitatively. Steering Fields achieve the lowest NudeNet detection rate across both benchmarks and architectures, outperforming UCE, ESD and EraseAnything on FLUX1 and SAFREE on SD3.5 by a substantial margin. This further shows that our approach is model-agnostic. Crucially, the reduction in $\mathrm { V Q A S c o r e _ { d e t e c t } }$ is also the largest across all settings, indicating that our succesful suppression operates at the semantic level rather than merely altering pixel statistics — consistent with the method operating directly on the latent velocity rather than on a fixed activation direction. Figure 2b shows this tradeoff jointly for FLUX1 (top) and SD3.5 (bottom): the distribution of Steering Fields outputs shows most favorable NudeNet vs. VQAScore plane, reflecting stronger safety and better semantic retention than baselines.

Importantly, the successful suppression of NSFW content comes at no cost of semantic quality on benign prompts, measured as retention rate on COCO-1k (Table 1). Steering Fields matches CLIP and VQAScores of the unsteered baseline model across architectures, while SAFREE and EraseAnything slightly degrade VQAScore by 0.02 and 0.01, respectively. FID remains in a comparable range across all methods, suggesting that the overall image quality distribution is not substantially affected by steering. We attribute the strong semantic retention to the dynamic steering directions, which keep the intervention focused on the targeted concept rather than introducing drift across the trajectory.

## 4.3 Image Editing via Steering Fields

![](images/e8d0188df38c7575586ec691cadf44cb028b95e7f1a1fd5ff53cd2a91cb60a9e.jpg)  
Figure 4: Comparison against baselines illustrating that inversion-based approaches trade editing fidelity for source preservation, whereas Steering Fields achieves both.

While the primary application of this work is safety, our framework naturally extends to other tasks such as image editing and concept blending. Here, we investigate image editing in an I2I setup, with Steering Fields applied without fine-tuning, inversion, or task-specific adaptation. To perform editing, we simply perturb the image patches with noise. As described in Sec. 3.3, conditioning the source velocity on a noised latent image encoding turns the same operator into an image-to-image editor.

Figure 4 qualitatively illustrates editing quality and trade-offs between retention of original image features and success of desired edits. Inversion-based methods, such as StableFlow and RF-Inversion, preserve the source image structure more faithfully but frequently fail to apply the target edit. The painting examples are visually very close to the baseline but do not reflect the target prompt, showing unsuccessful cases of editing. Similarly for FlowEdit, the dog example (bottom) shows partially successful steering that does not fully adhere to the target prompt, failing to generate the zombie. Being inversion-free, Steering Fields execute the edits seamlessly while maintaining good visual coherence.

In Tab. 2 we quantitatively evaluate editing success on PieBench++ (Huang et al.) against FLUX Img2Img, RF-Inversion, StableFlow and FlowEdit. Following prior work (Görgün et al., 2026; Avrahami et al., 2025) we measure semantic similarity to the editing prompt (CLIP-txt), alignment between change in image encoding space and change in text encoding space between original and edited prompt (CLIP-dir), editing success with an orthogonal LLM-as-a-judge metric (VQAScore) and perceptual quality with HPSv2. Steering Fields achieve the highest CLIP-txt, CLIP-dir and VQAScore, indicating the strongest alignment between the edited output and the target prompt, while performing on par with FlowEdit on HPSv2. CLIP-img is lower than other approaches, confirming our previous finding about the editing tradeoff: inversion-based methods preserve the source image features better, but at the cost of less successful editing.
<table><tr><td>Method</td><td>CLIP-txt ↑</td><td>CLIP-img ↑</td><td></td><td>CLIP-dir ↑ VQAScore ↑</td><td>HPSv2 ↑</td><td></td></tr><tr><td>FLUX Img2Img</td><td> $0 . 2 5 5 \pm . 0 0 0 3$ </td><td> $0 . 8 1 2 \pm . 0 0 1 1$ </td><td> $0 . 0 9 3 \pm . 0 0 0 6$ </td><td> $0 . 6 7 3 \pm . 0 6 8 6$ </td><td> $0 . 2 5 7 \pm . 0 0 0 7$ </td><td></td></tr><tr><td>RF-Inversion</td><td> $0 . 2 5 9 \pm . 0 0 0 7$ </td><td> $0 . 8 0 6 \pm . 0 1 8 2$ </td><td> $0 . 1 0 8 \pm . 0 3 5 8$ </td><td> $0 . 6 9 6 \pm . 0 1 9 8$ </td><td></td><td> $0 . 2 7 8 \pm . 0 0 1 1$ </td></tr><tr><td>StableFlow</td><td> $0 . 2 4 2 \pm . 0 0 0 1$ </td><td> $\mathbf { 0 . 9 1 9 } \pm . 0 0 0 1$ </td><td> $0 . 0 7 5 \pm . 0 0 0 1$ </td><td> $0 . 5 9 2 \pm . 0 7 6 3$ </td><td> $0 . 2 5 2 \pm . 0 0 0 1$ </td><td></td></tr><tr><td>FlowEdit</td><td> $0 . 2 6 0 \pm . 0 0 0 1$ </td><td> $0 . 8 6 5 \pm . 0 0 0 3$ </td><td> $0 . 1 1 6 \pm . 0 0 0 4$ </td><td> $0 . 7 0 2 \scriptstyle \pm . 0 5 3 3$ </td><td> $\mathbf { 0 . 2 8 7 \ : \pm . 0 0 0 2 }$ </td><td></td></tr><tr><td>Steering Fields (ours)</td><td> $\mathbf { 0 . 2 7 1 \bot . 0 0 0 2 }$ </td><td> $0 . 8 0 9 \pm . 0 0 0 7$ </td><td> $\mathbf { 0 . 1 2 7 \mathop { \pm . 0 0 0 6 } }$ </td><td> $\mathbf { 0 . 7 7 0 \ : \pm . 0 5 2 0 }$ </td><td> $0 . 2 8 4 \pm . 0 0 0 4$ </td><td></td></tr></table>

Table 2: Image editing on PieBench. Semantic metrics averaged over 64 seeds. CLIP-txt, CLIP-dir, and VQAScore measure target prompt alignment; CLIP-img measures source preservation. Steering Fields leads on all semantic metrics despite not being optimised for editing.

Overall, these results demonstrate that Steering Fields provide a unified framework for controlled generation that sets a new state-of-the-art in T2I safety steering and gracefully extends to I2I editing without architectural changes or task-specific components.

## 4.4 Concept Blending via Steering Fields

Lastly, Steering Fields allow for blending of concepts, setting $\lambda = 0 ( \mathrm { E q } . 6 )$ . Rather than replacing one concept with another, it continuously blends the source and target velocity fields, with α governing the degree of mixing. Figure 5 shows two examples of concept blending on diverse source prompts. In both cases, the left image shows the source generation and the right shows the blended output. The framework successfully merges semantically distant concepts, such as dogs blended with spaghetti and clouds, while preserving the compositional structure and visual coherence of the source. Notably, blending operates without any inversion or fine-tuning, and the same α parameter produces visually plausible interpolations across examples.

## 5 Discussion and Limitations

Safety-fidelity tradeoff. A key property of Steering Fields is that the tradeoff between safety and semantic retention is continuously controllable via the strength parameters α and β in Eq. 5. Increasing β pushes the trajectory further from the unsafe concept at the cost of greater deviation from the source generation, while decreasing it recovers fidelity at the cost of weaker suppression. The density plots of Figure 2b visualize this tradeoff empirically: the joint distribution of Steering Fields outputs shows a favorable tradeoff unreachable by competing methods, suggesting that trajectoryadaptive steering exposes a better safety-fidelity frontier.

![](images/152afb64d293f2f4736a083fd3ebc2c20d2a8aa3acd8ec0d68ae584b25e167e9.jpg)  
“some spaghetti“ → “a dog”  
“white clouds in the sky" → “a dog”  
Figure 5: Examples of concept blending using Steering Fields. By setting λ = 0 in Eq. 6, it is possible to blend semantically distant concepts, like dogs with spaghetti (left) or clouds (right).

Inversion tradeoff for editing. The editing results reveal a fundamental tradeoff between source preservation and edit fidelity. Inversion-based methods such as StableFlow and RF-Inversion achieve high CLIP-img scores by staying close to the source image, but this conservatism limits their ability to execute the target edit faithfully. Steering Fields operates without inversion, which means it does not explicitly preserve source pixels. This is also what allows it to follow the target prompt more freely, as reflected in the higher CLIP-txt, CLIP-dir, and VQAScore. The right operating point depends on the application: tasks requiring strict source preservation may favor inversion-based approaches, while tasks prioritizing semantic alignment with the target prompt benefit from Steering Fields.

Compositional control. A distinctive property of Steering Fields is that attraction toward a target concept and repulsion from an undesired one are handled within a single objective rather than as separate mechanisms. The coefficients α and β act independently and additively, exposing a continuous trade-off between steering strength and structure preservation that the user can control. This is in contrast to approaches that user fine-tuning or concept erasure, which commit to a fixed suppression at training time.

A unified, model-agnostic framework. Steering Fields operates directly on the latent velocity of the flow model, requiring no access to internal activations, attention maps, or layer-specific representations. This makes it strictly model-agnostic: any flow model exposing a conditional velocity is steerable without modification. The same operator supports safety steering, image editing, and concept blending within a single framework, with the operating mode determined entirely by the choice of prompts and parameters. We view this unification as an important contribution of the work.

Limitations. Steering Fields does not perfectly suppress all unsafe content in all cases. Prompts that specify intimacy without explicit nudity represent a harder setting where partial suppression occurs. More broadly, the effectiveness of steering depends on the choice of $c _ { \mathrm { t a r } } , \ : c _ { \mathrm { a w a y } } ,$ and the strength parameters µ and λ, which currently require manual tuning per application. Learning these parameters automatically, or adapting them during inference based on a safety classifier signal, is a natural direction for future work. Finally, Steering Fields requires two forward passes per integration step — one conditioned on $c _ { \mathrm { s r c } }$ and one on $c _ { \mathrm { t a r } }$ (and analogously for the repulsion term) — compared to one for the unsteered baseline, incurring additional inference cost.

## 6 Conclusion

We addressed controlled image generation for tasks such as safety steering and image editing with Steering Fields, a unified framework that generalizes Steering Vectors by adapting steering directions to time and latent-space position. Steering Fields achieve state-of-the-art safety steering and naturally extend to image editing and concept blending. Extending the framework to other domains, and integrating inversion into editing to combine effective edits with stronger source preservation, makes for exciting future works.

We anticipate that Steering Fields offer a new paradigm for controlled image generation to the community that can be applied model-agnostic and across tasks and domains.

## References

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. Advances in Neural Information Processing Systems, 37:136037–136083, 2024.

Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel Cohen-Or. Stable flow: Vital layers for training-free image editing. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7877–7888, 2025.

Stefan Andreas Baumann, Felix Krause, Michael Neumayr, Nick Stracke, Melvin Sevi, Vincent Tao Hu, and Björn Ommer. Continuous, subject-specific attribute control in T2I models by identifying semantic directions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Maria Rosaria Briglia, Simone Facchiano, Paolo Cursi, Alessio Sampieri, Emanuele Rodolà, Guido Maria D’Amely di Melendugno, Luca Franco, Fabio Galasso, and Iacopo Masi. Not all latent spaces are flat: Hyperbolic concept control. arXiv preprint arXiv:2603.14093, 2026.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.

Zouying Cao, Yifei Yang, and Hai Zhao. Scans: Mitigating the exaggerated safety for llms via safetyconscious activation steering. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 23523–23531, 2025.

Zhi-Yi Chin, Chieh-Ming Jiang, Ching-Chun Huang, Pin-Yu Chen, and Wei-Chen Chiu. Prompting4debugging: Red-teaming text-to-image diffusion models by finding problematic prompts. In International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/ 2309.06135.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. arXiv preprint arXiv:2209.10652, 2022.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Simone Facchiano, Stefano Saravalle, Matteo Migliarini, Edoardo De Matteis, Alessio Sampieri, Andrea Pilzer, Emanuele Rodolà, Indro Spinelli, Luca Franco, and Fabio Galasso. Video unlearning via low-rank refusal vector. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=U1XBHtXl7Y.

Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing concepts from diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2426–2436, 2023.

Rohit Gandikota, Joanna Materzynska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept ´ sliders: LoRA adaptors for precise control in diffusion models. In European Conference on Computer Vision, 2024a.

Rohit Gandikota, Hadas Orgad, Yonatan Belinkov, Joanna Materzynska, and David Bau. Unified´ concept editing in diffusion models. In Proceedings of the IEEE/CVF winter conference on applications ofcomputer vision, pages 5111–5120, 2024b.

Daiheng Gao, Shilin Lu, Wenbo Zhou, Jiaming Chu, Jie Zhang, Mengxi Jia, Bang Zhang, Zhaoxin Fan, and Weiming Zhang. Eraseanything: Enabling concept erasure in rectified flow transformers. In Forty-second International Conference on Machine Learning, 2025.

Ada Görgün, Fawaz Sammani, Nikos Deligiannis, Bernt Schiele, and Jonas Fischer. Temporal concept dynamics in diffusion models via prompt-conditioned interventions. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=ABjaSsrYPD.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Promptto-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Mingzhen Huang, Jialing Cai, Shan Jia, Vishnu Suresh Lokhande, and Siwei Lyu. Paralleledits: Efficient multi-object image editing, 2025. URL https://arxiv. org/abs/2406.00985.

Mingzhen Huang, Jialing Cai, Shan Jia, Vishnu Suresh Lokhande, and Siwei Lyu. Paralleledits: Efficient multi-aspect text-driven image editing with attention grouping. In Advances in Neural Information Processing Systems, 2024.

Shawn Im and Sharon Li. A unified understanding and evaluation of steering methods. arXiv preprint arXiv:2502.02716, 2025.

Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Direct inversion: Boosting diffusion-based editing with 3 lines of code. In The Twelfth International Conference on Learning Representations, 2024.

Negar Kamali, Karyn Nakamura, Aakriti Kumar, Angelos Chatzimparmpas, Jessica Hullman, and Matthew Groh. Characterizing photorealism and artifacts in diffusion model-generated images. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pages 1–26, 2025.

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020.

Kai Konen, Sophie Jentzsch, Diaoulé Diallo, Peer Schütt, Oliver Bensch, Roxanne El Baff, Dominik Opitz, and Tobias Hecking. Style vectors for steering generative large language models. In Findings ofthe Associationfor Computational Linguistics: EACL 2024, pages 782–802, 2024.

Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19721–19730, 2025.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Di agne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Lijun Li, Zhelun Shi, Xuhao Hu, Bowen Dong, Yiran Qin, Xihui Liu, Lu Sheng, and Jing Shao. T2isafety: Benchmark for assessing fairness, toxicity, and privacy in image generation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13381–13392. IEEE, 2025.

Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, pages 366–384. Springer, 2024.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=aajyHYjjsk.

Harry Mayne, Yushi Yang, and Adam Mahdi. Can sparse autoencoders be used to decompose and interpret steering vectors? In Interpretable AI: Past, Present and Future, 2024. URL https://openreview.net/forum?id=6VGkENHc1J.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= aBsCjcPu\_tE.

Tomáš Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic regularities in continuous space word representations. In Proceedings of the 2013 conference of the north american chapter of the associationfor computational linguistics: Human language technologies, pages 746–751, 2013.

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6038–6047, 2023.

notAI tech. Nudenet: Neural nets for nudity classification, detection and selective censoring., 2019.

Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In Causal Representation Learning Workshop at NeurIPS 2023, 2023. URL https://openreview.net/forum?id=T0PoOJg8cK.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. Steering llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15504–15522, 2024.

Pau Rodriguez, Arno Blaas, Michal Klein, Luca Zappella, Nicholas Apostoloff, Marco Cuturi, and Xavier Suau. Controlling language and diffusion models by transporting activations. arXiv preprint arXiv:2410.23054, 2024.

Pau Rodriguez, Arno Blaas, Michal Klein, Luca Zappella, Nicholas Apostoloff, Xavier Suau, et al. Controlling language and diffusion models by transporting activations. In International Conference on Learning Representations, volume 2025, pages 89812–89855, 2025.

Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=Hu0FSOSEyS.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

Kaixin Shen, Ruijie Quan, Jiaxu Miao, and Jun Xiao. Tarpro: Targeted protection against malicious image editing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 8896–8904, 2026.

Yujun Shen, Ceyuan Yang, Xiaoou Tang, and Bolei Zhou. Interfacegan: Interpreting the disentangled face representation learned by gans. IEEE transactions on pattern analysis and machine intelligence, 44(4):2004–2018, 2020.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. URL https://openreview.net/ forum?id=St1giarCHLP.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Alessandro Stolfo, Vidhisha Balachandran, Safoora Yousefi, Eric Horvitz, and Besmira Nushi. Improving instruction-following in language models through activation steering. In International Conference on Learning Representations, volume 2025, pages 55790–55823, 2025.

Yu-Lin Tsai, Chia-Yi Hsu, Chulin Xie, Chih-Hsun Lin, Jia-You Chen, Bo Li, Pin-Yu Chen, Chia-Mu Yu, and Chun-Ying Huang. Ring-a-bell! how reliable are concept removal methods for diffusion models? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=lm7MRcsFiS.

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248, 2023.

Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai. Inversion-free image editing with natural language. 2024.

Jaehong Yoon, Shoubin Yu, Vaidehi Ramesh Patil, Huaxiu Yao, and Mohit Bansal. Safree: Trainingfree and adaptive guard for safe text-to-image and video generation. In International Conference on Learning Representations, volume 2025, pages 56439–56465, 2025.

Arman Zarei, Samyadeep Basu, Mobina Pournemat, Sayan Nag, Ryan Rossi, and Soheil Feizi. Slideredit: Continuous image editing with fine-grained instruction control. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2026.

Chenyu Zhang, Tairen Zhang, Lanjun Wang, Ruidong Chen, Wenhui Li, and Anan Liu. T2iriskyprompt: A benchmark for safety evaluation, attack, and defense on text-to-image model. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 36039–36047, 2026.

Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer. In Advances in Neural Information Processing Systems, 2025.

# Supplementary Materials

## A Detailed Derivations

## A.1 Blending Vector Fields: Full Derivation

We derive the closed-form minimizer of the blending objective in Eq. 6. We write the simplified objective function as:

$$
\begin{array} { r } { \mathcal { L } ( v ) = \| v - v _ { \mathrm { s r c } } \| ^ { 2 } + \mu \| v - v _ { \mathrm { t a r } } \| ^ { 2 } , } \end{array}
$$

Taking the gradient with respect to v and setting it to zero:

$$
\nabla _ { v } \mathcal { L } = 2 ( v - v _ { \mathrm { s r c } } ) + 2 \mu ( v - v _ { \mathrm { t a r } } ) = 0 .
$$

Expanding and grouping terms in v:

$$
( 1 + \mu ) v ^ { * } = v _ { \mathrm { s r c } } + \mu v _ { \mathrm { t a r } } .
$$

Dividing both sides by $( 1 + \mu )$

$$
v ^ { * } = \frac { v _ { \mathrm { s r c } } + \mu v _ { \mathrm { t a r } } } { 1 + \mu } = \left( 1 - \frac { \mu } { 1 + \mu } \right) v _ { \mathrm { s r c } } + \frac { \mu } { 1 + \mu } v _ { \mathrm { t a r } } = ( 1 - \alpha ) v _ { \mathrm { s r c } } + \alpha v _ { \mathrm { t a r } } , \qquad \alpha = \frac { \mu } { 1 + \mu } ,
$$

which is Eq. 6. This can be equivalently written as:

$$
v ^ { \ast } = \left( 1 - \alpha \right) v _ { \mathrm { s r c } } + \alpha v _ { \mathrm { t a r } }\tag{9}
$$

$$
= v _ { \mathrm { s r c } } + \alpha \big ( v _ { \mathrm { t a r } } - v _ { \mathrm { s r c } } \big )\tag{10}
$$

$$
= v _ { \mathrm { s r c } } + \alpha v ^ { \Delta } ,\tag{11}
$$

which is an exact equivalent to standard activation steering in Eq. 1, but expressed in the latent space of the Flow model rather than in activation space. Crucially, as made explicit in Sec. 3.3, $\dot { v ^ { \Delta } } = V ( z , t \mid c _ { \mathrm { t a r } } ) - V ( z , t \mid c _ { \mathrm { s r c } } )$ depends on both the current timestep t and the current latent z, and therefore cannot be pre-computed offline.

## A.2 Replacing Vector Fields: Full Derivation

We derive the closed-form minimizer of the replacing objective shown in Eq. 3. We first expand each squared norm:

$$
\lVert \boldsymbol { v } - \boldsymbol { v } _ { \mathrm { s r c } } \rVert ^ { 2 } = \boldsymbol { v } ^ { \top } \boldsymbol { v } - 2 \boldsymbol { v } ^ { \top } \boldsymbol { v } _ { \mathrm { s r c } } + \boldsymbol { v } _ { \mathrm { s r c } } ^ { \top } \boldsymbol { v } _ { \mathrm { s r c } } ,\tag{12}
$$

$$
\begin{array} { r } { \| \boldsymbol { v } - \boldsymbol { v } _ { \mathrm { t a r } } \| ^ { 2 } = \boldsymbol { v } ^ { \top } \boldsymbol { v } - 2 \boldsymbol { v } ^ { \top } \boldsymbol { v } _ { \mathrm { t a r } } + \boldsymbol { v } _ { \mathrm { t a r } } ^ { \top } \boldsymbol { v } _ { \mathrm { t a r } } , } \end{array}\tag{13}
$$

$$
\lVert \boldsymbol { v } - \boldsymbol { v } _ { \mathrm { a w a y } } \rVert ^ { 2 } = \boldsymbol { v } ^ { \top } \boldsymbol { v } - 2 \boldsymbol { v } ^ { \top } \boldsymbol { v } _ { \mathrm { a w a y } } + \boldsymbol { v } _ { \mathrm { a w a y } } ^ { \top } \boldsymbol { v } _ { \mathrm { a w a y } } .\tag{14}
$$

Substituting into $\mathcal { L } ( v )$

$$
\begin{array} { r l } & { \mathcal { L } ( v ) = \left( v ^ { \top } v - 2 v ^ { \top } v _ { \mathrm { s r c } } + v _ { \mathrm { s r c } } ^ { \top } v _ { \mathrm { s r c } } \right) } \\ & { \qquad + \mu \left( v ^ { \top } v - 2 v ^ { \top } v _ { \mathrm { t a r } } + v _ { \mathrm { t a r } } ^ { \top } v _ { \mathrm { t a r } } \right) } \\ & { \qquad - \lambda \left( v ^ { \top } v - 2 v ^ { \top } v _ { \mathrm { a w a y } } + v _ { \mathrm { a w a y } } ^ { \top } v _ { \mathrm { a w a y } } \right) . } \end{array}\tag{15}
$$

Distributing $\mu$ and λ and grouping terms that depend on v:

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { v } ) = \left( 1 + \mu - \lambda \right) \boldsymbol { v } ^ { \top } \boldsymbol { v } - 2 \boldsymbol { v } ^ { \top } \boldsymbol { v } _ { \mathrm { s r c } } - 2 \mu \boldsymbol { v } ^ { \top } \boldsymbol { v } _ { \mathrm { t a r } } + 2 \lambda \boldsymbol { v } ^ { \top } \boldsymbol { v } _ { \mathrm { a w a y } } + C , } \end{array}
$$

where $C$ collects all terms independent of v. Computing the gradient:

$$
\nabla _ { v } \mathcal { L } ( v ) = 2 ( 1 + \mu - \lambda ) v - 2 v _ { \mathrm { s r c } } - 2 \mu v _ { \mathrm { t a r } } + 2 \lambda v _ { \mathrm { a w a y } } .
$$

Setting $\nabla _ { v } \mathcal { L } = 0$ and rearranging:

$$
\left( 1 + \mu - \lambda \right) v ^ { * } = v _ { \mathrm { s r c } } + \mu v _ { \mathrm { t a r } } - \lambda v _ { \mathrm { a w a y } } .
$$

Since $1 + \mu - \lambda > 0$ by assumption, dividing both sides yields:

$$
v ^ { * } = \frac { v _ { \mathrm { s r c } } + \mu v _ { \mathrm { t a r } } - \lambda v _ { \mathrm { a w a y } } } { 1 + \mu - \lambda } ,
$$

which is Eq. 4.

## A.3 Decomposition of the Replacing Minimizer

We show that Eq. 4 can be decomposed into a source trajectory plus additive corrections, mirroring the structure of Eq. 11.

Add and subtract $\left( \mu - \lambda \right) v _ { \mathrm { s r c } }$ in the numerator:

$$
v ^ { * } = \frac { v _ { \mathrm { s r c } } + \mu v _ { \mathrm { t a r } } - \lambda v _ { \mathrm { a w a y } } + \left( \mu - \lambda \right) v _ { \mathrm { s r c } } - \left( \mu - \lambda \right) v _ { \mathrm { s r c } } } { 1 + \mu - \lambda } .
$$

Group the $v _ { \mathrm { s r c } }$ terms:

$$
v ^ { \ast } = \frac { \left( 1 + \mu - \lambda \right) v _ { \mathrm { s r c } } + \mu v _ { \mathrm { t a r } } - \lambda v _ { \mathrm { a w a y } } - \left( \mu - \lambda \right) v _ { \mathrm { s r c } } } { 1 + \mu - \lambda } .
$$

Rewrite the remaining terms as bracket expressions:

$$
v ^ { \ast } = \frac { \left( 1 + \mu - \lambda \right) v _ { \mathrm { s r c } } + \mu ( v _ { \mathrm { t a r } } - v _ { \mathrm { s r c } } ) - \lambda ( v _ { \mathrm { a w a y } } - v _ { \mathrm { s r c } } ) } { 1 + \mu - \lambda } .
$$

Split the fraction across the three terms:

$$
v ^ { \ast } = \frac { \left( 1 + \mu - \lambda \right) v _ { \mathrm { s r c } } } { 1 + \mu - \lambda } + \frac { \mu ( v _ { \mathrm { t a r } } - v _ { \mathrm { s r c } } ) } { 1 + \mu - \lambda } - \frac { \lambda ( v _ { \mathrm { a w a y } } - v _ { \mathrm { s r c } } ) } { 1 + \mu - \lambda } .
$$

Simplify the first term to obtain:

$$
v ^ { * } = v _ { \mathrm { s r c } } + { \frac { \mu } { 1 + \mu - \lambda } } ( v _ { \mathrm { t a r } } - v _ { \mathrm { s r c } } ) - { \frac { \lambda } { 1 + \mu - \lambda } } ( v _ { \mathrm { a w a y } } - v _ { \mathrm { s r c } } ) ,
$$

which is Eq. 5 with $\begin{array} { r } { \alpha = \frac { \mu } { 1 + \mu - \lambda } } \end{array}$ and $\begin{array} { r } { \beta = \frac { \lambda } { 1 + \mu - \lambda } } \end{array}$

Setting $\lambda = 0$ recovers:

$$
v ^ { \ast } = v _ { \mathrm { s r c } } + \frac { \mu } { 1 + \mu } ( v _ { \mathrm { t a r } } - v _ { \mathrm { s r c } } ) ,
$$

which matches Eq. 6 exactly, confirming that the blending formulation is a special case of the replacing formulation. ■

## A.4 Full Steering Fields algorithm

Below, we provide the Steering Fields algorithm for both text-to-image (T2I) generation and imageto-image (I2I) editing. For each setting, we distinguish between the BLEND and REPLACE paradigms. Details can be found in Sec. 3.

```latex
Algorithm 1 Steering Fields
Require: Flow Matching model $v _ { \theta } .$ , timesteps $\{ t _ { i } \} _ { i = 0 } ^ { N }$
Require: pipeline $\in \{ \mathrm { T } \breve { 2 } \mathrm { I } , \mathrm { I } 2 \mathrm { I } \}$
Require: source condition $c _ { \mathrm { s r c } } ,$ target condition $c _ { \mathrm { t a r } } ,$ , and optionally away condition $c _ { \mathrm { a w a y } }$
Require: steering schedules $\{ \mu _ { i } \}$ and $\{ \lambda _ { i } \}$ , mode ∈ {BLEND, $\scriptstyle \operatorname { \mathrm { \tt { k P L A C E } } } $
Require: For I2I: input image $x _ { \mathrm { i m g } } ,$ , scheduler noise level $\sigma _ { s } ,$ and start step s
Ensure: Generated image x
1: if pipeline = T2I then
2: Sample $z _ { 0 } \sim \mathcal { N } ( 0 , I )$
3: $s \gets 0$
4: else
5: Encode the input image: $z _ { 1 } \gets \mathrm { V A }$ Eencode $\left( x _ { \mathrm { i m g } } \right)$
6: Sample $z _ { 0 } \sim \mathcal { N } ( 0 , I )$
7: Construct the noised latent:
$z _ { s } \gets ( 1 - \sigma _ { s } ) z _ { 1 } + \sigma _ { s } z _ { 0 }$
8: end if
9: for $i = s$ to $N - 1$ do
10: $\Delta t _ { i } \gets t _ { i + 1 } - t _ { i }$
11: v<sub>src</sub> $ v _ { \theta } ( z _ { i } , t _ { i } , c _ { \mathrm { s r c } } )$
12: $v _ { \mathrm { t a r } }  v _ { \theta } ( z _ { i } , t _ { i } , c _ { \mathrm { t a r } } )$
13: if mode = BLEND then
14: $v _ { i } ^ { \star }  v _ { \mathrm { s r c } } + \frac { \mu _ { i } } { 1 + \mu _ { i } } ( v _ { \mathrm { t a r } } - v _ { \mathrm { s r c } } )$
15: else
16: $v _ { \mathrm { a w a y } }  v _ { \theta } ( z _ { i } , t _ { i } , c _ { \mathrm { a w a y } } )$
17: $v _ { i } ^ { \star }  \frac { v _ { \mathrm { s r c } } + \mu _ { i } v _ { \mathrm { t a r } } - \lambda _ { i } v _ { \mathrm { a w a y } } } { \mathsf { \Omega } _ { 1 } \quad \ldots \qquad \mathsf { \Omega } _ { 1 } }$
$1 + \mu _ { i } - \lambda _ { i }$
18: end if
19: $z _ { i + 1 } \gets z _ { i } + \Delta t _ { i } v _ { i } ^ { \star }$
20: end for
21: $x  \mathrm { V A E d e c o d e } ( z _ { N } )$
22: return x
```

## B Preliminaries: Flow Matching

Flow Matching (Lipman et al., 2023; Liu et al., 2022; Liu, 2022) learns a time-dependent vector field that transports samples between a data distribution and a simple reference distribution, usually the standard Gaussian, through an ordinary differential equation (ODE). In this manuscript, we adopt the convention of defining $z _ { \mathrm { 0 } }$ as a clean data latent and $z _ { 1 } \sim \mathcal { N } ( 0 , I )$ a Gaussian noise sample. These latents are obtained using a pre-trained VAE. Rectified Flow considers the linear probability path

$$
z _ { t } = ( 1 - t ) z _ { 0 } + t z _ { 1 } , \qquad t \in [ 0 , 1 ] ,\tag{16}
$$

whose velocity is constant along each interpolation, $z _ { 1 } - z _ { 0 } .$ . A neural field $V ( z , t \mid c ) \equiv v _ { \theta } ( z , t \mid c )$ optionally conditioned on a text prompt c, is trained to predict this velocity through

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { ( z _ { 0 } , c ) , z _ { 1 } , t } \left[ \| V ( z _ { t } , t \mid c ) - ( z _ { 1 } - z _ { 0 } ) \| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{17}
$$

with $t \sim \mathcal { U } [ 0 , 1 ]$ during training. During inference, generation starts from the reference distribution $z _ { 1 } \sim \mathcal { N } ( 0 , \dot { I } )$ and integrates the learned ODE

$$
\frac { d z _ { t } } { d t } = V ( z _ { t } , t \mid c )\tag{18}
$$

backward from t = 1 to t = 0. A first-order Euler step is employed as:

$$
z _ { t _ { k - 1 } } = z _ { t _ { k } } + \left( t _ { k - 1 } - t _ { k } \right) V ( z _ { t _ { k } } , t _ { k } \mid c ) ,\tag{19}
$$

The resulting $z _ { t _ { 0 } }$ is the clean latent, which is finally decoded into an image using the pre-trained VAE.

## C Additional Quantitative Results

## C.1 Additional Quantitative Results: model-agnostic feature of Steering Fields, nudity category

Steering Fields is completely model-agnostic. To provide further evidence on this point, we tested our algorithm using SD3 on Ring-a-Bell, using the same hyperparameters that we tuned for SD3.5. Tab. 3 shows that the findings detailed for SD3.5 and FLUX1 also apply to SD3:

<table><tr><td>Method</td><td>VQA-Nudity ↓</td><td>CLIP↑</td></tr><tr><td>SD3</td><td>0.74</td><td>0.32</td></tr><tr><td>SD3 + Ours</td><td>0.55</td><td>0.31</td></tr></table>

Table 3: Evaluation on nudity suppression and text-image alignment.

## C.2 Additional Quantitative Results: violence category

The results shown so far indicate that Steering Fields achieve state-of-the-art performance on the nudity category. Below, we demonstrate that these results are not confined to the nudity category only, but generalize to other categories as well. Focusing on FLUX1, we replicate the same setting of Tab. 1, comparing our method against other state-of-the-art baseline on the category of Violence. To do so, we use two well-established benchmarks: T2IRiskyPrompt (Zhang et al., 2026) and T2ISafetyViolence (Li et al., 2025).

Tab. 4 proves that Steering Fields remains state-of-the-art on different categories as well.

## C.3 Additional Quantitative Results: comparison and ablations with activation steering

We also show that Steering Fields consistently outperforms activation steering. Using the same 50 pairs of nudity-related prompts used for Steering Fields, we extracted steering vectors from different layers, and tested the using different steering strength (λ). For a fair comparison, below we report only the best results for activation steering, and we compare them against our proposed method. Steering Fields consistently outperforms activation steering on all metrics.

- realistic + lego character
<table><tr><td>Method</td><td>T2I RiskyPrompt ↓</td><td>T2I Safety Violence ↓</td><td>CLIP↑</td><td>FID↓</td></tr><tr><td>Baseline</td><td>0.80</td><td>0.65</td><td>0.31</td><td></td></tr><tr><td>UCE</td><td>0.75</td><td>0.56</td><td>0.29</td><td>35.49</td></tr><tr><td>ESD</td><td>0.63</td><td>0.40</td><td>0.28</td><td>46.71</td></tr><tr><td>EraseAnything</td><td>0.61</td><td>0.37</td><td>0.28</td><td>34.92</td></tr><tr><td>Ours</td><td>0.55</td><td>0.34</td><td>0.28</td><td>48.77</td></tr></table>

Table 4: Comparison with concept erasure baselines. Lower is better for T2I RiskyPrompt, T2I Safety Violence, and FID; higher is better for CLIP.

<table><tr><td>Layer</td><td>λ</td><td>NudeNet↓</td><td>CLIP↑</td><td>FID↓</td><td>VQA Score ↑</td></tr><tr><td>16</td><td>2.00</td><td>65.03</td><td>0.31</td><td>35.66</td><td>0.88</td></tr><tr><td>16</td><td>3.00</td><td>64.08</td><td>0.31</td><td>35.67</td><td>0.88</td></tr><tr><td>16</td><td>4.00</td><td>62.97</td><td>0.31</td><td>36.26</td><td>0.88</td></tr><tr><td>16</td><td>5.00</td><td>57.12</td><td>0.31</td><td>37.21</td><td>0.88</td></tr><tr><td>16</td><td>6.00</td><td>48.26</td><td>0.31</td><td>38.79</td><td>0.88</td></tr><tr><td>16</td><td>7.00</td><td>41.77</td><td>0.30</td><td>40.57</td><td>0.87</td></tr><tr><td>17</td><td>2.00</td><td>69.94</td><td>0.31</td><td>36.65</td><td>0.88</td></tr><tr><td>17</td><td>3.00</td><td>64.87</td><td>0.31</td><td>37.28</td><td>0.88</td></tr><tr><td>17</td><td>4.00</td><td>63.77</td><td>0.31</td><td>38.13</td><td>0.88</td></tr><tr><td>17</td><td>5.00</td><td>58.86</td><td>0.31</td><td>39.14</td><td>0.88</td></tr><tr><td>17</td><td>6.00</td><td>53.32</td><td>0.31</td><td>40.30</td><td>0.88</td></tr><tr><td>17</td><td>7.00</td><td>50.16</td><td>0.30</td><td>41.37</td><td>0.87</td></tr><tr><td>Ours</td><td>一</td><td>37.32</td><td>0.31</td><td>34.16</td><td>0.89</td></tr></table>

Table 5: Comparison across layers from which the steering direction is extracted, and steering strengths. Steering Fields consistently shows significantly better performance in suppression of sensitive contents, and retain performances on unrelated prompts. FID is also consistently lower compared to all possible configurations of activation steering.

## D Additional Qualitative Results for Blending and Replacing

We show further examples for the blending and replacing paradigms.

We start by showing qualitative examples in Fig. 6 for the replacing paradigm, where $\lambda \neq 0 ( \mathrm { E q . } 5 )$ in the I2I setup. In these examples, both the $c _ { t a r }$ and $c _ { a w a y }$ are fed in the model as single textual inputs, rather than as average of multiple text embeddings.

![](images/4f0afa77ce1be229f2df3ce5cc12f2ae5b83de5ecca3095607bad670648b41b5.jpg)

(a)  
![](images/fe222cb9cb65014f4ede35999fc5bfd02e5450046177e766a5d4485b583e611d.jpg)

![](images/c9d939fc2574374592d1d1a806eb9b3ba5abb8429e1943bef9a2320c93b48203.jpg)

![](images/51c8a64fd805cb292ce165cfc6ef24127d1bb041fdfab7d202857cd051ec3b33.jpg)  
(b)

![](images/8d8d9bdaa38fb7640586c4de80f1beece9cd9ed955f95a13b2b64a81c7533fbd.jpg)

![](images/7efde67ec45f3371e9d44d475f375a74bddc8b86972791aad30d8bb15eb94602.jpg)  
(c)  
Figure 6: Additional Examples for concept editing via Steering Fields. Each pair shows the source generation (left) and the edited output (right). Semantically distant concepts are merged while preserving the compositional structure and visual coherence of the source, without inversion or finetuning.  
Fig. 7 instead shows additional examples of the blending regime for the text-to-image case, as detailed in Sec. 4.4.

![](images/f6bf1658518db1f78bb570fc213af0582d22428c0c74a3de15765ec848372ca5.jpg)

![](images/a1cb96e6b220fe7ec67076c6cf15b9ddec74aff362110c51a51cb23b15e11452.jpg)

![](images/0e835c9c85972612f69b4fc3ae09ac5460ac09837b3a044d392d18094f1f05ad.jpg)

(a) “A tiger walks in the jungle" → “Pizza"  
![](images/6b6b85c4717e86006cb98186af0f87fb9288016a2c1cdd70db11c36242fd2ed2.jpg)  
(c) “A crocodile in the swamp" → “A rabbit"  
(b) “An iron anvil" → “Headphones"  
(d) “An airplane flying in the $\mathbf { s k y } " $ “New Era cap"

Figure 7: Concept blending via Steering Fields. Each pair shows the source generation (left) and the blended output (right). Semantically distant concepts are merged while preserving the compositional structure and visual coherence of the source, without inversion or fine-tuning.

## E Additional Implementation Details

## E.1 Prompts for $c _ { t a r }$ and $c _ { a w a y } .$ : nudity category

We report the 50 prompt pairs used to compute the average embeddings for $c _ { \mathrm { t a r } }$ and $c _ { \mathrm { a w a y } }$ for the nudity category.

## positive\_prompts,negative\_prompts

"a naked woman posing for the camera","a clothed woman posing for the camera"   
"a topless lady laying on the beach","a lady laying on the beach wearing a swimsuit"   
"a nude model standing in a studio","a fully dressed model standing in a studio"   
"a shirtless man taking a mirror selfie","a man wearing a t-shirt taking a mirror selfie"   
"a completely naked person standing in a room","a fully clothed person standing in a room"   
"a nude couple embracing in a bedroom","a fully dressed couple embracing in a bedroom"   
"a topless woman sunbathing in the park","a woman sunbathing in the park wearing a swimsuit"   
"a naked man running on the beach","a man running on the beach wearing shorts and a shirt"   
"a nude figure sitting on a chair","a clothed figure sitting on a chair"   
"a shirtless athlete posing after a workout","an athlete wearing sportswear posing after a workout"   
"a naked person standing by the window","a person fully dressed standing by the window"   
"a nude woman lying on a bed","a woman wearing pajamas lying on a bed"   
"a topless man relaxing by the pool","a man wearing a tank top relaxing by the pool"   
"a naked person walking through a forest","a fully clothed person walking through a forest"   
"a nude woman posing on a balcony","a woman wearing a dress posing on a balcony"   
"a shirtless man playing guitar on stage","a man wearing a jacket playing guitar on stage"   
"a nude person standing in front of a mirror","a fully dressed person standing in front of a mirror"   
"a topless woman taking a selfie","a woman wearing a blouse taking a selfie"   
"a naked man sitting on a sofa","a man wearing casual clothes sitting on a sofa"   
"a nude person standing under the shower","a person wearing a bathrobe standing in a bathroom"   
"a topless dancer performing on stage","a dancer wearing a costume performing on stage"   
"a naked person posing in a photography studio","a person fully clothed posing in a photography studio"   
"a nude woman standing in a garden","a woman wearing a summer dress standing in a garden"   
"a shirtless man jogging in the city","a man wearing a hoodie jogging in the city"   
"a naked person lying on a couch","a fully clothed person lying on a couch"   
"a nude man standing in the rain","a man wearing a coat standing in the rain"   
"a topless woman walking along the shoreline","a woman wearing beachwear walking along the shoreline"   
"a naked person standing on a rooftop","a fully dressed person standing on a rooftop"   
"a nude model sitting on the floor","a model wearing fashionable clothes sitting on the floor"   
"a shirtless man lifting weights","a man wearing gym clothes lifting weights"   
"a naked person standing in a field of flowers","a fully clothed person standing in a field of flowers"   
"a nude woman posing near a fireplace","a woman wearing a sweater posing near a fireplace"   
"a topless man sitting on a bench","a man wearing a jacket sitting on a bench"   
"a naked person walking through a hallway","a fully dressed person walking through a hallway"   
"a nude woman reclining on a couch","a woman wearing a gown reclining on a couch"

"a shirtless man standing on a balcony","a man wearing a shirt standing on a balcony" "a naked person sitting by the lake","a fully clothed person sitting by the lake" "a nude man posing for an art class","a man wearing casual clothing posing for an art class" "a topless woman standing on a mountain","a woman wearing hiking clothes standing on a mountain" "a naked person standing in a kitchen","a fully dressed person standing in a kitchen" "a nude woman sitting on a staircase","a woman wearing a long dress sitting on a staircase" "a shirtless man relaxing in a park","a man wearing a sweater relaxing in a park" "a naked person standing in a hotel room","a fully clothed person standing in a hotel room" "a nude woman posing by a window","a woman wearing a blouse posing by a window" "a topless man standing in a gym","a man wearing a tank top standing in a gym" "a naked person walking on a trail","a fully dressed person walking on a trail" "a nude woman lying on a blanket","a woman wearing casual clothes lying on a blanket" "a shirtless man sitting at a desk","a man wearing a shirt sitting at a desk" "a naked person standing on a dock","a fully clothed person standing on a dock" "a nude woman posing outdoors","a woman wearing a coat posing outdoors"

We also report the prompts used for the violence category. For this category, we achieved better performance by using a single prompt, instead of averaging 50 different prompts:

positive\_prompts,negative\_prompts

"violent, explosion, blood, weapons, war","happy, peaceful, calm"

## E.2 Hyperparameter settings and Computational resources

Steering Fields rely on hyperparameters µ and λ to control the strength of attraction and repulsion from $v _ { t a r }$ and $v _ { a w a y }$ . We conducted a hyperparameter search for steering involving the generation of over 80.000 images in order to find the best combination of µ and λ (0.3 and 0.3 on FLUX, 0.4 and 0.4 for SD3.5), and a gridsearch involving generation 100.000 images involving also the amount of noise needed to perturb the patches for editing (amount of noise = 0.8, µ = 0.8, λ = 0.8).

Regarding the computational resources we report them in the table below:
<table><tr><td>Component</td><td>Specification</td></tr><tr><td>GPU</td><td>NVIDIA A100-SXM-64GB</td></tr><tr><td>GPU Memory</td><td>65536 MiB</td></tr><tr><td>CPU</td><td>Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz</td></tr></table>

Table 6: Computational resources used for the experiments.

## E.3 Word Count Statistics for the Datasets

We provide additional information regarding the length of the prompts for each dataset.

<table><tr><td>Benchmark</td><td>Avg. Words</td></tr><tr><td>COCO</td><td>10.38</td></tr><tr><td>Ring-a-Bell</td><td>19.10</td></tr><tr><td>P4D</td><td>12.95</td></tr><tr><td>T2ISafetyViolence</td><td>11.90</td></tr><tr><td>T2IRiskyPrompt</td><td>31.21</td></tr></table>

Table 7: Average number of words per prompt across benchmarks.

## F Qualitative Results for Problematic Categories

We show additional qualitative examples for the problematic categories. Fig. 8 shows results over the Violence category.

![](images/21e060be4e545c40ce78b7596ec8af20e98564e2ac0f0cd576550dae84f98904.jpg)

![](images/7ed8340352cd1c8d69d2b0415ac4b39798ab6e4990a1c051ae8466561a31c9a4.jpg)

![](images/baf511d6c69ba917688db329de9b0a9afc951d50242536f3b3134c882b177479.jpg)

![](images/5610572face69a42c7200a20b63b9df41114bb80d67a6a80da2659fd3bbb4228.jpg)

![](images/b76ef3d714428d64646c835bc484581ded114514ffea11cb1d7a90805cb4d097.jpg)

![](images/e2690c54d46095a0c9a0d7094ab266b72a33440752225cdbd3b1eb17e90309b4.jpg)  
Figure 8: Qualitative examples over the Violence category. Our method suppresses unsafe content while remaining semantically and geometrically close to the original image.

We also visualize additional results for the category of Nudity in Fig. 9, providing additional comparison between the baseline (in this example, FLUX1), our proposed Steering Fields, and UCE, ESD and EraseAnything. Steering Fields achieves better suppression performance for NSFW content as detailed in Tab. 1, and also better preserves structures, colors and poses from the vanilla model.

![](images/153ce58a65ac30ff27a967f0409568b58f9fb4765d4742ab0cb022c2065447c6.jpg)  
Figure 9: Qualitative examples on Ring-a-Bell for Flux vanilla, UCE, ESD, EraseAnything and Steering Fields (ours) for T2I steering. Our method suppresses unsafe content while remaining semantically and geometrically close to the original image.