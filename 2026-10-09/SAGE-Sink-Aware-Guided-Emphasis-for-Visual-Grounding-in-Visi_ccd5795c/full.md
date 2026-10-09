# SAGE: Sink-Aware Guided Emphasis for Visual Grounding in Vision-Language Decoders

Jeonghyo Song<sup>\*</sup>, YoungJoon Yoo<sup>†</sup> Department of Artificial Intelligence, Chung-Ang University {thd9592s, yjyoo3312}@cau.ac.kr <sup>\*</sup>First author. <sup>†</sup>Corresponding author.

## Abstract

Recent large vision-language models (VLMs) pair a visual encoder with a large language model (LLM) and perform well on diverse image–text tasks, yet their reliability is often limited by decoder attention pathologies that suppress visual evidence and exacerbate hallucinations. In this paper, we revisit visual attention sinks and uncover a structured, layerdependent behavior: across prompts, early and late decoder layers exhibit prompt-invariant attention collapse onto the same few image regions, which we term PIS (Prompt-Invariant Sinks), whereas mid layers become promptconditioned and drive vision-language alignment. This split suggests that treating sinks as a uniform effect is incomplete. Building on this insight, we propose SAGE (Sink-Aware Guided Emphasis), a lightweight intervention that steers decoder attention away from PIS and toward query-dependent regions of interest (ROIs) using token-aligned ROI masks derived from standard vision backbones such as CLIP, ViT, and DINOv3. Evaluated on diverse vision-encoder + decoder-only LLM VLM families, SAGE improves visual grounding, reduces hallucinations, and yields consistent gains across public downstream visionlanguage benchmarks, including fine-grained visual discrimination settings where localized evidence is crucial, when instantiated with backbone-derived ROI masks.

## 1 Introduction

Large vision–language models (VLMs) (Liu et al., 2023, 2024b; Li et al., 2023a; Wang et al., 2024; Yang et al., 2025) have become a strong foundation for integrating visual perception with language reasoning. By connecting a visual encoder to a large language model (LLM) (Achiam et al., 2023; Touvron et al., 2023; Minaee et al., 2024) decoder via lightweight adapters (Wan et al., 2023; Van Nguyen et al., 2025), recent VLMs support a broad spectrum of applications, including image captioning, detailed image understanding, OCR, object localization, and multi-turn visual question answering within a unified interface. Despite these capabilities, reliability remains a central concern. VLMs may overlook subtle but decisive visual evidence, struggle with fine-grained recognition (Tong et al., 2024; Kim and Ji, 2024; Liao et al., 2025), and produce plausible yet incorrect statements that contradict the image, a behavior commonly referred to as hallucination (Li et al., 2023b; Huang et al., 2024; Chen et al., 2025b; Liu et al., 2024a).

Prior analyses suggest that these failures are not solely due to the visual encoder, but also to the decoder’s attention dynamics over visual tokens (Kang et al., 2025). In particular, selfattention can concentrate on a small subset of tokens that consistently attracts an outsized share of the attention mass. In multimodal decoders, such visual attention sinks can dominate the attention budget and pull attention away from query-relevant image regions. This reduces precise visual grounding, especially for questions that rely on subtle, localized evidence.

In this paper, we re-examine decoder-side visual attention sinks and identify a structured, layerdependent behavior. This behavior is easy to miss when attention is aggregated across layers rather than analyzed layer by layer. Figure 1 provides a motivating example. Across three substantially different prompts, early decoder layers and final decoder layers repeatedly assign high attention to nearly identical image locations, even though the question changes. We call these stable, promptagnostic regions Prompt-Invariant Sinks. In contrast, the mid layers exhibit the only consistently prompt-responsive behavior. When the query asks for the ‘espresso price’, mid-layer attention shifts toward the menu board. When the query asks for the current time, mid-layer attention moves toward the clock. When the prompt requests a scene description, mid-layer attention becomes broader and better aligned with semantically relevant regions. These observations reveal a dual role of the decoder stack: attention is largely static near the beginning and the end, while meaningful vision–language alignment emerges primarily in the middle layers.

This layer-wise structure has two implications. First, treating visual attention sinks as a single uniform pathology is incomplete, because the decoder simultaneously contains components that are insensitive to the query and components that must perform alignment. Second, Prompt-Invariant Sinks offer a concrete target for intervention. If attention can be discouraged from collapsing onto these static sinks and instead encouraged to concentrate on query-dependent regions of interest, VLMs may become more faithful to the image and more effective at fine-grained reasoning.

Motivated by these observations, we introduce SAGE (Sink-Aware Guided Emphasis) as a lightweight and controlled method to establish where and how decoder-side attention interventions matter in encoder–decoder VLMs such as widely-used Qwen-VL (Wang et al., 2024; Yang et al., 2025), LLaVA (Liu et al., 2023, 2024b; Li et al., 2025)-style, InternVL (Zhu et al., 2025; Chen et al., 2024) and NVILA (Liu et al., 2025) architectures. The core contribution of this work is the layer-dependent structure of visual attention sinks and alignment, and SAGE serves to make this effect measurable and reproducible. Concretely, SAGE steers decoder attention away from Prompt-Invariant Sinks and toward query-dependent regions of interest (ROIs) during generation. In our implementation, ROI masks are derived from standard pretrained vision backbones such as CLIP, ViT, and DINOv3, providing a consistent way to construct token-aligned prompt-conditioned interventions across models. With SAGE, we sweep over the intervention layers and show that simple ROI-guided edits at specific decoder layers yield reliable gains in grounding and downstream performance.

Our contributions are summarized as threefold:

• We identify Prompt-Invariant Sinks, showing that early and final decoder layers exhibit prompt-agnostic attention concentration on fixed image regions across diverse queries, as illustrated in Figure 1.

• We reveal a layer-wise duality in VLM decoders, where mid layers are primarily responsible for prompt-responsive vision-language alignment, while end layers exhibit static sink behavior.

• We propose SAGE as a controllable local ROIguided intervention tool that enables systematic layer sweeps and provides evidence that selecting an effective decoder layer for intervention can improve grounding and benchmark performance across multiple VLMs.

## 2 Related Work

## 2.1 Vision-Language Models

Large vision-language models (VLMs) couple visual representation learning with language generation and reasoning. A dominant family follows the vision-encoder + decoder-only LLM design, in which image features are projected into visual tokens and jointly processed with text tokens by an autoregressive decoder. Representative examples include LLaVA-style models (Liu et al., 2024b; Li et al., 2025) and Qwen-VL (Bai et al., 2023; Wang et al., 2024; Yang et al., 2025), while recent work further improves efficiency and fine-grained perception by sparsifying visual tokens at inference or by predicting regions of interest to focus computation (Khaki et al., 2025; Shi et al., 2026). Other work explores alternative multimodal bridging and scaling strategies, including Flamingo (Alayrac et al., 2022), BLIP-2/InstructBLIP (Li et al., 2023a; Dai et al., 2023), explicit grounding formulations (Geigle et al., 2024), and multilingual largescale training as in PaLI and PaLI-X (Chen et al., 2022, 2023).

## 2.2 Attention Sinks and Visual Attention Sinks

Transformer attention is known to exhibit attention sinks, where a few tokens absorb disproportionate attention mass independent of semantic relevance. This phenomenon has been linked to efficient longcontext and streaming inference, and later work studies when sinks emerge during pretraining and how architectural or normalization choices affect their formation (Xiao et al., 2023; Gu et al., 2025; Wu et al., 2024). Similar pathologies also arise over visual tokens in multimodal models. VAR (Kang et al., 2025) extends the sink notion to LVLMs and shows repeated concentration on query-irrelevant visual tokens; related work in vision transformers and token/head-level analyses likewise studies architectural mitigation, selective steering toward salient regions, and training-time regularization for stronger grounding (Feng et al., 2026; Tang et al., 2026; Zhao et al., 2026; Esmaeilkhani and Latecki, 2026).

![](images/6a48c10883590422aa1a9421bc4d8fd439f8b0340e654cab7df9148454fa7ca6.jpg)  
Figure 1: Prompt-Invariant Sinks and prompt-conditioned mid-layer attention in a VLM decoder. We visualize decoder attention over visual tokens for three different prompts (rows) at representative early, mid, and final decoder layers (columns). Early and final layers exhibit prompt-invariant attention collapse onto nearly identical image regions across prompts, forming Prompt-Invariant Sinks. In contrast, mid-layer attention is prompt-conditioned: it shifts to the menu board when asking for the espresso price (Prompt 1), to the wall clock when asking for the current time (Prompt 2), and becomes broader for holistic scene description (Prompt 3). Warmer colors indicate higher attention.

These attention pathologies are closely connected to hallucination and grounding failures, where outputs are not faithfully supported by the image. Benchmarks such as POPE (Li et al., 2023b) quantify object hallucination, while recent methods mitigate it through decoding-time interventions, contrastive signals, or vision-side masking of uncertain visual tokens (Kim et al., 2024; Park et al., 2025; Zhao et al., 2026; Chen et al., 2025a; Jung et al., 2025; Seo et al., 2026; Leng et al., 2024).

Our positioning and new finding. While prior work establishes the existence of visual attention sinks, our study shows that sink behavior is structured across depth rather than uniform. Early and late decoder layers tend to exhibit prompt-invariant collapse onto stable image regions, whereas mid layers remain comparatively prompt-responsive and play the main role in vision-language alignment. This layer-dependent split motivates interventions that steer attention away from promptinvariant sink regions and query-dependent ROI.

## 3 Methodology

In this section, we present SAGE (Sink-Aware Guided Emphasis), a training-free inference-time intervention that steers decoder attention toward query-relevant visual evidence and away from attention-sink regions.

We focus on vision-encoder + decoder-only LLM VLMs, including LLaVA 1.5(Liu et al., 2024b), Qwen2-VL (Wang et al., 2024), InternVL3 (Zhu et al., 2025), and NVILA (Liu et al., 2025), where a vision encoder produces visual features that are projected or adapted into a sequence of visual tokens and processed jointly with text tokens by an autoregressive LLM via causal selfattention.

A key of SAGE is that it can be instantiated with multiple backbone-derived ROI generators as long as they can be mapped into a token-aligned ROI mask over visual tokens. In this paper, we instantiate SAGE with ROI masks derived from standard vision backbones such as CLIP (Radford et al., 2021), ViT (Dosovitskiy et al., 2020), and DI-NOv3 (Siméoni et al., 2025). These masks provide a simple and consistent interface for decoder-side attention steering across models.

![](images/c0b1ddee5050702c00d62e8b14ce8801101886a763ec5952872ae79d815c3986.jpg)  
Figure 2: Overview of SAGE. Given an image I and a query Q, SAGE operates in two stages. Step 1 (ROI construction from standard vision backbones): a token-aligned ROI mask over visual tokens is derived from patch-level relevance maps obtained from CLIP, ViT, or DINOv3. Step 2 (Training-free attention steering and layer selection): using the ROI mask, SAGE applies an additive bias to visual-token attention logits (Eq. (1)) at selected decoder layer(s), redistributing attention away from sink regions and toward query-relevant visual evidence. Layer selection can be performed via a layer sweep and top-k selection strategy (Section 3.4). The resulting attention redistribution improves visual grounding during autoregressive generation.

## 3.1 Preliminaries and Notation

Given an image I and a query $Q ,$ , the vision encoder produces visual features that are transformed into $N$ visual tokens $\mathbf { V } = \left[ v _ { 1 } , \dots , v _ { N } \right]$ via a projector/adapter. The LLM based on the decoder-only processes a unified multimodal sequence consisting of visual tokens and text tokens and generates an answer autoregressively.

At decoding step t, layer $\ell \in \{ 0 , \ldots , L - 1 \}$ , and head $h ,$ we denote by $\mathbf { a } _ { t } ^ { ( \ell , h ) } \in \mathbb { R } ^ { N }$ the restricted pre-softmax attention logits from the query token at step t to the $N$ visual-token keys. SAGE represents the guided ROI in visual-token space as a binary mask m $\in 0 , 1 ^ { N }$ , where m[i] = 1 marks the i-th token as inside the ROI.

## 3.2 ROI Construction from Standard Vision Backbones

SAGE requires only a token-aligned binary ROI mask m $\bar { \mathbf { \xi } } \in \mathbf { \zeta } \{ 0 , 1 \} ^ { \bar { N } }$ and makes no assumption about how it is produced. We instantiate it with three standard pretrained vision backbones, which differ in an important respect: only CLIP conditions on the query, while ViT and DINOv3 produce query-agnostic saliency.

CLIP We project and $L _ { \mathrm { { 2 } } } { \mathrm { { - n o r m a l i z e } } }$ the patchtoken embeddings and the query text embedding into the shared CLIP space and score each patch by cos(patch, text). The resulting relevance map therefore depends on the query.

ViT We use attention rollout from the CLS token at the last layer, averaged over heads. The query is not an input.

DINOv3 We use CLS-to-patch attention at the last layer, excluding register tokens. The query is not an input.

Patch-to-token alignment. For every source, the patch score map is resized to the visual-token grid of the evaluated VLM with nearest-neighbor interpolation, and the top-k tokens are kept as the binary mask.

## 3.3 Training-Free Attention Steering

Given a token-aligned ROI mask m $\in \{ 0 , 1 \} ^ { N }$ SAGE steers attention toward query-relevant visual evidence by directly biasing attention allocation within the decoder-only LLM. For a chosen layer ℓ and head h at decoding step t, let ${ \bf a } _ { t } ^ { ( \ell , h ) } \in \dot { \mathbb { R } } ^ { N }$ denote the (visual-token) attention logits over N visual tokens. SAGE applies a simple additive bias controlled by $\alpha > 0$ (ROI amplification) and $\beta > 0$ (non-ROI suppression):

$$
\mathbf { a } _ { t } ^ { \prime ( \ell , h ) } [ i ] = \left\{ \mathbf { a } _ { t } ^ { ( \ell , h ) } [ i ] + \alpha , \ : \ : \ : \mathrm { i f } \ : \mathbf { m } [ i ] = 1 , \right.\tag{1}
$$

This bias reallocates attention mass toward the guided ROI, encouraging the model to consult query-relevant visual tokens while preserving the original model parameters. SAGE is therefore training-free and can be applied at inference time to arbitrary layer sets ${ \mathcal { L } } _ { \mathrm { i n t } }$ (single-layer or multilayer).

![](images/f7f68ac36c61812e672508ea3451cae2d5b4e2f51ddd85f14cd1e582898b1f3c.jpg)  
(a) Prompt-Invariant Sink Structure

![](images/183bfdc80da54d6eecc1bd15eb6bbbfc474197bec51c84d11f957ea68f64a047.jpg)  
(b) Spectral Dominance ρ1

![](images/9a2b5b8d741d859241c4bb83973b6eafe80f5c5fa8c8c16cbba9ca1b325a6f0e.jpg)  
(c) Attention Redistribution  
Figure 3: Evidence for sink structure, spectral dominance, and attention redistribution. (a) Prompt-invariant sink structure. For each layer, we form a sink set as the top-5% visual tokens by attention weight and report the IoU of sink sets across different prompts for the same image (error bars: standard deviation). (b) Spectral dominance. We stack visual-key attention logits across prompts and decoding steps to form $\mathbf { A } ^ { ( \ell , h ) }$ (Eq. (3)) and report the top-1 energy ratio $\rho _ { 1 }$ (Eq. (5)); SAGE reduces spectral concentration, indicating weaker low-rank sink dominance. (c) Attention redistribution. we report attention mass assigned to sink tokens (top-5% tokens) and ROI tokens; SAGE shifts attention budget away from sink tokens and toward ROI, consistent with the odds-ratio effect in Eq. (6).

Implementation note. The ROI mask is first computed at patch level and then aligned to the visualtoken grid used by each evaluated VLM. The exact patch-to-token mapping depends on the model’s vision encoder and preprocessing pipeline; we keep this mapping fixed within each experiment and summarize the details in Appendix A.3.

## 3.4 Layer Sweep and Top-k Layer Selection

SAGE can be applied to different decoder depths, and its effect is generally layer-dependent. Rather than prescribing a single fixed layer as the only valid choice, we use a simple sweep protocol to (i) characterize where attention steering tends to be effective and (ii) form reasonable intervention sets (e.g., top-k layers) that perform well in practice.

Concretely, we run a single-layer sweep where SAGE is applied to one layer ℓ at a time (all other layers unchanged) and measure the relative change in performance against the no-intervention baseline. Let $\Delta ( \ell )$ denote an aggregated relative performance change for intervening only at layer ℓ. This sweep serves as a diagnostic: it highlights layers where steering is more effective and provides a ranking that can be used to construct multi-layer intervention sets. Using the ranking by $\Delta ( \ell )$ , we define a top-k layers set

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o p - } k } = \mathrm { T o p K } _ { \ell } \bigl ( \Delta ( \ell ) \bigr ) , } \end{array}\tag{2}
$$

and apply SAGE to $\mathcal { L } _ { \mathrm { t o p } - k }$ in subsequent experiments as one practical choice. Importantly, our goal is not to claim that only a particular layer (or only one selection rule) works; instead, the sweep illustrates that attention steering itself is broadly effective, and that multiple reasonable layer selections (single-layer or multi-layer) can yield improvements.

## 3.5 Mechanistic Rationale

We provide a mechanistic view of why SAGE can improve visual grounding in terms of Sink Subspace and Attention Redistribution. Our analysis in Figure 3 suggests that visual attention sinks in VLM decoders are not arbitrary noise but exhibit structured, prompt-invariant patterns across depth. We formalize this intuition using a simple spectral perspective and show how the SAGE bias induces an attention-redistribution effect. Figure 3 summarizes the key empirical signatures discussed below. Sink subspace perspective. Fix a decoder layer ℓ and head h. For each decoding step t and prompt instance $p ,$ let $\mathbf { a } _ { p , t } ^ { ( \ell , h ) } \in \mathbb { R } ^ { N }$ denote the pre-softmax attention logits over the N visual-token keys (as in Section 3.1). To quantify prompt-invariant sink, we define a sink set as the top-r% visual tokens by attention weight and measure its overlap across prompts for the same image using set IoU (Fig. 3a).

We stack these logit vectors across prompts and decoding steps into a matrix

$$
\mathbf { A } ^ { ( \ell , h ) } \in \mathbb { R } ^ { M \times N } , \qquad \mathbf { A } ^ { ( \ell , h ) } [ r , : ] = \mathbf { a } _ { p , t } ^ { ( \ell , h ) } ,\tag{3}
$$

where $r$ indexes the pair $( p , t )$ and M is the total number of stacked instances. We then compute an SVD,

$$
\mathbf { A } ^ { ( \ell , h ) } = \mathbf { U } \Sigma \mathbf { V } ^ { \top } .\tag{4}
$$

If attention repeatedly collapses onto a small set of prompt-invariant sink tokens, the spectrum becomes more concentrated, and the top right-singular vectors (columns of V) capture stable token-level directions corresponding to sinkdominated patterns. A convenient summary is the energy captured by the top-k components:

$$
\rho _ { k } ( \mathbf { A } ) = \frac { \sum _ { j = 1 } ^ { k } \sigma _ { j } ^ { 2 } } { \sum _ { j = 1 } ^ { \operatorname* { m i n } ( M , N ) } \sigma _ { j } ^ { 2 } } ,\tag{5}
$$

where $\{ \sigma _ { j } \}$ are singular values. Our hypothesis is that prompt-invariant sinks induce high $\rho _ { k }$ for small $k ,$ reflecting structured low-rank dominance (Fig. 3b).

Attention redistribution induced by SAGE. SAGE applies an additive logit bias to visualtoken attention as defined in Eq. (1). Let $p _ { i } ~ =$ Softmax(a) and $p _ { i } ^ { \prime } = \mathrm { S o f t m a x } ( \mathbf { a } ^ { \prime } ) _ { i }$ be the attention weights over visual keys before and after applying Eq. (1), respectively. For any ROI index i and non-ROI index j, the odds ratio satisfies

$$
\frac { p _ { i } ^ { \prime } } { p _ { j } ^ { \prime } } = \exp ( \alpha + \beta ) \cdot \frac { p _ { i } } { p _ { j } } .\tag{6}
$$

Consequently, even when sink locations remain spatially stable, suppressing non-ROI tokens (which may include sink tokens) and amplifying ROI tokens can substantially redistribute attention budget toward query-relevant visual evidence in our analysis setting (Fig. 3c).

Testable implications. This view suggests two measurable signatures of effective attention steering: (i) reduced sink dominance reflected by lower spectral concentration (e.g., smaller $\rho _ { k }$ for small k), and (ii) increased attention mass on ROI tokens accompanied by decreased mass on sink tokens. We examine these signatures with layer-wise spectral and mass-based analyses, with details provided in Appendix E.

Cross-model evidence. The two diagnostics are not independent descriptions of the same plot. Across LLaVA-1.5 and Qwen2-VL on TextVQA and GQA, layer-wise prompt-invariant Sink IoU correlates strongly with top-1 spectral concentration (Pearson $r = 0 . 8 8 6 \mathrm { - } 0 . 9 4 2 )$ . This supports reading prompt-invariant sinks as structured lowrank routing directions rather than image-specific heatmap artifacts, and it is why the two can be combined into a single label-free depth criterion (Section 3.4).

## 4 Experiments

## 4.1 Experimental Setup

Models. We evaluate SAGE on vision-encoder + decoder-only LLM VLMs from the LLaVA 1.5 (Liu et al., 2024b), Qwen2-VL (Wang et al., 2024), InternVL3 (Zhu et al., 2025) and NVILA (Liu et al., 2025). Unless stated otherwise, we use the instruct-tuned checkpoints listed in Table 1.

ROI sources. Our main results use ROI masks derived from standard vision backbones, specifically CLIP, ViT, and DINOv3. Given an image and query, each source produces patch-level relevance scores, which are thresholded to obtain a binary ROI mask aligned to the visual-token grid of the evaluated VLM. Detailed source-specific settings are summarized in Appendix C.

Intervention settings. Unless stated otherwise, we use a fixed bias strength across models. In all experiments, we set $\alpha = 5 . 0$ (ROI amplification) and $\beta = 2 . 0$ (non-ROI bias). For layer-wise ablations, we apply a single-layer intervention by sweeping the intervention layer across the decoder stack of each model, with all other layers unchanged.

Normalized decoder depth. Because different VLM families can have different numbers of decoder layers, we use normalized decoder depth when comparing layer-wise behavior across models. For a model with L decoder layers and zeroindexed layer ℓ, we define the normalized depth as $d ( \ell ) = \ell / ( L - 1 )$ . We use raw layer indices only in model-specific analyses such as LLaVA-1.5 case studies and appendix tables.

## 4.2 Main Evaluations

We evaluate SAGE on representative visionencoder + decoder-only LLM VLM families and summarize the main results in Table 1. For each model, benchmark, and ROI source, the reported SAGE score corresponds to the best single-layer result obtained from the layer sweep. We use these results to quantify the peak effect of SAGE at effective decoder depths, rather than to claim a universal fixed layer rule. Because the layer is selected per benchmark, these values indicate an upper bound on the achievable effect. Section 4.5 reports the result obtained with a single layer fixed in advance, without benchmark labels. For transparency, the exact selected layer indices are listed in Appendix Table 6; when discussing layer behavior across models, we refer to normalized decoder depth rather

Table 1: Main results of SAGE across VLM families. For each model, benchmark, and ROI source, the reported SAGE score corresponds to the best single-layer result obtained from the layer sweep. Exact selected layer indices are listed in Appendix Table 6.
<table><tr><td rowspan="2">MODEL</td><td rowspan="2">TEXTVQA</td><td colspan="2">POPE</td><td rowspan="2">GQA</td><td rowspan="2">SCIENCEQA</td><td rowspan="2">MME</td><td colspan="2">MMVP</td><td colspan="2">MMVP ORIGINAL</td><td rowspan="2">MMVET</td></tr><tr><td>Acc</td><td>F1</td><td>IMAGE</td><td>PAIR</td><td>IMAGE</td><td>PAIR</td></tr><tr><td>LLAVA 1.5-13B</td><td>55.50</td><td>82.75</td><td>81.55</td><td>61.65</td><td>63.98</td><td>1783.11</td><td>61.48</td><td>11.85</td><td>33.58</td><td>10.65</td><td>31.20</td></tr><tr><td rowspan="2">+ SAGE (CLIP)</td><td>61.94</td><td>86.84</td><td>86.58</td><td>62.65</td><td>68.63</td><td>1842.58</td><td>62.96</td><td>37.78</td><td>50.00</td><td>22.00</td><td>42.20</td></tr><tr><td>(+6.44)</td><td>(+4.09)</td><td>(+5.03)</td><td>(+1.00)</td><td>(+4.65)</td><td>(+59.47)</td><td>(+1.48)</td><td>(+25.93)</td><td>(+16.42)</td><td>(+11.35)</td><td>(+11.00)</td></tr><tr><td rowspan="2">+ SAGE (VIT)</td><td>61.85</td><td>85.81</td><td>84.82</td><td>65.50</td><td>69.15</td><td>1895.55</td><td>68.89</td><td>38.52</td><td>58.58</td><td>21.27</td><td>34.50</td></tr><tr><td>(+6.35)</td><td>(+3.06)</td><td>(+3.27)</td><td>(+3.85)</td><td>(+5.17)</td><td>(+112.44)</td><td>(+7.41)</td><td>(+26.67)</td><td>(+25.00)</td><td>(+10.62)</td><td>(+3.30)</td></tr><tr><td rowspan="2">+ SAGE (DINOV3)</td><td>61.54</td><td>83.00</td><td>81.72</td><td>62.55</td><td>69.17</td><td>1791.44</td><td>63.33</td><td>14.07</td><td>34.00</td><td>12.00</td><td>36.47</td></tr><tr><td>(+6.04)</td><td>(+0.25)</td><td>(+0.17)</td><td>(+0.90)</td><td>(+5.19)</td><td>(+8.33)</td><td>(+1.85)</td><td>(+2.22)</td><td>(+0.42)</td><td>(+1.35)</td><td>(+5.27)</td></tr><tr><td>QWEN2-VL-7B</td><td>81.32</td><td>75.94</td><td>75.42</td><td>62.04</td><td>68.34</td><td>1792.60</td><td>78.15</td><td>57.78</td><td>69.33</td><td>40.67</td><td>59.60</td></tr><tr><td rowspan="2">+ SAGE (CLIP)</td><td>81.43</td><td>76.59</td><td>75.52</td><td>62.38</td><td>74.16</td><td>1895.10</td><td>81.15</td><td>58.28</td><td>71.21</td><td>41.33</td><td>61.10</td></tr><tr><td>(+0.11)</td><td>(+0.65)</td><td>(+0.10)</td><td>(+0.34)</td><td>(+5.82)</td><td>(+102.50)</td><td>(+3.00)</td><td>(+0.50)</td><td>(+1.88)</td><td>(+0.66)</td><td>(+1.50)</td></tr><tr><td rowspan="2">+ SAGE (VIT)</td><td>81.52</td><td>77.15</td><td>76.81</td><td>62.28</td><td>71.28</td><td>1920.00</td><td>78.88</td><td>57.94</td><td>70.24</td><td>42.53</td><td>61.80</td></tr><tr><td>(+0.20)</td><td>(+1.21)</td><td>(+1.39)</td><td>(+0.24)</td><td>(+2.94)</td><td>(+127.40)</td><td>(+0.73)</td><td>(+0.16)</td><td>(+0.91)</td><td>(+1.86)</td><td>(+2.20)</td></tr><tr><td rowspan="2">+ SAGE (DINOV3)</td><td>81.41</td><td>78.35</td><td>77.56</td><td>62.15</td><td>68.47</td><td>1795.23</td><td>91.85</td><td>85.19</td><td>70.67</td><td>42.53</td><td>59.90</td></tr><tr><td>(+0.09)</td><td>(+2.41)</td><td>(+2.14)</td><td>(+0.11)</td><td>(+0.13)</td><td>(+2.63)</td><td>(+13.70)</td><td>(+27.41)</td><td>(+1.34)</td><td>(+1.86)</td><td>(+0.30)</td></tr><tr><td>INTERNVL3-2B</td><td>47.60</td><td>87.97</td><td>87.41</td><td>56.85</td><td>61.95</td><td>2087.90</td><td>73.44</td><td>49.37</td><td>72.67</td><td>47.33</td><td>61.40</td></tr><tr><td rowspan="2">+ SAGE (CLIP)</td><td>60.20</td><td>88.27</td><td>88.36</td><td>57.15</td><td>62.25</td><td>2092.58</td><td>76.30</td><td>58.52</td><td>73.33</td><td>49.33</td><td>62.20</td></tr><tr><td>(+12.60)</td><td>(+0.30)</td><td>(+0.95)</td><td>(+0.30)</td><td>(+0.30)</td><td>(+4.68)</td><td>(+2.86)</td><td>(+9.15)</td><td>(+0.66)</td><td>(+2.00)</td><td>(+0.80)</td></tr><tr><td rowspan="2">+ SAGE (VIT)</td><td>47.80</td><td>88.48</td><td>87.89</td><td>59.85</td><td>62.45</td><td>2097.29</td><td>77.78</td><td>57.78</td><td>72.33</td><td>46.00</td><td>62.10</td></tr><tr><td>(+0.20)</td><td>(+0.51)</td><td>(+0.48)</td><td>(+3.00)</td><td>(+0.50)</td><td>(+9.39)</td><td>(+4.34)</td><td>(+8.41)</td><td>(-0.34)</td><td>(-1.33)</td><td>(+0.70)</td></tr><tr><td rowspan="2">+ SAGE (DINOV3)</td><td>73.50</td><td>88.89</td><td>88.60</td><td>57.55</td><td>62.02</td><td>2072.23</td><td>73.70</td><td>49.63</td><td>75.00</td><td>50.70</td><td>61.50</td></tr><tr><td>(+25.90)</td><td>(+0.92)</td><td>(+1.19)</td><td>(+0.70)</td><td>(+0.07)</td><td>(-15.67)</td><td>(+0.26)</td><td>(+0.26)</td><td>(+2.33)</td><td>(+3.37)</td><td>(+0.10)</td></tr><tr><td>NVILA-8B</td><td>74.44</td><td>85.37</td><td>84.81</td><td>62.20</td><td>87.57</td><td>2107.29</td><td>78.52</td><td>58.52</td><td>70.67</td><td>43.33</td><td>61.79</td></tr><tr><td rowspan="2">+ SAGE (CLIP)</td><td>78.13</td><td>92.20</td><td>91.99</td><td>62.80</td><td>89.37</td><td>2125.59</td><td>78.88</td><td>59.25</td><td>72.67</td><td>47.33</td><td>62.16</td></tr><tr><td>(+3.69)</td><td>(+6.83)</td><td>(+7.18)</td><td>(+0.60)</td><td>(+1.80)</td><td>(+18.30)</td><td>(+0.36)</td><td>(+0.73)</td><td>(+2.00)</td><td>(+4.00)</td><td>(+0.37)</td></tr><tr><td rowspan="2">+ SAGE (VIT)</td><td>74.84</td><td>88.95</td><td>88.26</td><td>64.90</td><td>92.24</td><td>2133.59</td><td>79.12</td><td>59.82</td><td>71.48</td><td>45.13</td><td>61.94</td></tr><tr><td></td><td>(+3.58)</td><td>(+3.45)</td><td>(+2.70)</td><td>(+4.67)</td><td>(+26.30)</td><td>(+0.60)</td><td>(+1.30)</td><td>(+0.81)</td><td>(+1.80)</td><td>(+0.15)</td></tr><tr><td rowspan="2">+ SAGE (DINOV3)</td><td>(+0.40)</td><td>89.00</td><td>88.18</td><td>64.95</td><td>91.61</td><td>2118.85</td><td>79.94</td><td>60.02</td><td>71.86</td><td>44.94</td><td>62.36</td></tr><tr><td>74.75 (+0.31)</td><td>(+3.63)</td><td>(+3.37)</td><td>(+2.75)</td><td>(+4.04)</td><td>(+11.56)</td><td>(+1.42)</td><td>(+1.50)</td><td>(+1.19)</td><td>(+1.61)</td><td>(+0.57)</td></tr></table>

than raw layer number.

Across the evaluated VLMs, SAGE improves over the no-intervention baseline on many benchmarks and across multiple ROI sources. These gains are consistent with our motivating hypothesis that decoder-side attention pathologies can suppress query-relevant visual evidence, and that a lightweight logit bias toward an ROI helps the model focus on more relevant visual tokens. Across CLIP, ViT, and DINOv3, the qualitative trend is similar: the strongest improvements appear at selected decoder depths, while the effective depth can vary across model families and benchmarks.

Fine-grained visual discrimination. The MMVPfamily benchmarks are particularly relevant to our setting because they emphasize fine-grained pairwise visual discrimination, where the decisive evidence is often local and easy to miss. Table 1 shows strong improvements on MMVP and MMVP-Original, especially on pairwise metrics where both related samples must be answered correctly. To further examine whether these gains are tied only to a single selected layer, Figure 4 reports layer-wise averaged normalized improvement on the four MMVP-family sub-benchmarks for LLaVA-1.5. For CLIP-, ViT-, and DINOv3- derived ROI masks, the averaged improvement remains positive across a broad range of decoder layers, indicating that the benefit on fine-grained visual discrimination is not restricted to one isolated intervention depth.

## 4.3 Layer-wise Effects

This section analyzes how SAGE’s effectiveness varies with normalized decoder depth on LLaVA-1.5 and Qwen2-VL. Although SAGE uses the same intervention rule (Eq. (1)) and the same ROI construction procedure within each experiment setting, its effect can vary substantially depending on where in the decoder stack the intervention is applied. To isolate this factor, we conduct a single-layer sweep over the decoder stack for each model family by setting ${ \mathcal { L } } _ { \mathrm { i n t } } = \{ \ell \}$ and intervening at one layer at a time. Figure 5 summarizes the layer-wise improvements over the no-intervention baseline as a function of normalized decoder depth, aggregated across benchmarks (median), and highlights the top-performing depths.

Layer-wise variability. Figure 5 shows that SAGE exhibits clear layer-wise variability. Across both model families, intervening at a subset of layers yields consistent gains, whereas intervening at other layers leads to smaller changes and can occasionally be neutral. This indicates that attention steering is not uniformly beneficial throughout the decoder and that layer selection is a key factor in determining the effectiveness of SAGE.

![](images/a6eb775043767cbe9f19986f843f4ca20803d3a54022c389b44618530d804e29.jpg)  
Figure 4: Layer-wise averaged normalized improvement on MMVP-family benchmarks. For each ROI source (CLIP, ViT, DINOv3), we apply SAGE to one decoder layer and evaluate on four fine-grained sub benchmarks: MMVP image, MMVP pair, MMVP-Original image, and MMVP-Original pair. For each sub-benchmark, we compute the normalized improvement over the no-intervention baseline as $( s - b ) / b \times 1 0 0$ where s is the score with SAGE and b is the baseline score. The plotted value is the average of these four normalized improvements at each intervention layer. The x-axis shows the LLaVA-1.5 decoder layer index, and values above zero indicate improvement over the baseline.

Connection to visual attention sinks. Prior work reports that VLMs can suffer from visual attention sinks, where attention repeatedly collapses onto a small set of query-irrelevant visual tokens and limits the use of task-relevant visual evidence. SAGE is designed to counteract this pathology by reallocating attention toward a query-dependent ROI. The concentration of gains in selected decoder layers suggests that correcting attention allocation at those depths can improve grounding and downstream performance in the evaluated setting.

## 4.4 Depth-Grouped Multi-layer Comparisons

We further examine multi-layer SAGE by grouping the tested sets according to coarse normalizeddepth composition. Table 2 reports mean TextVQA accuracy within each group, while selected exact LLaVA-1.5 layer combinations are deferred to Appendix Tables 7, 8, and 9.

The summary shows a clear depth-dependent pattern. Very-early groups are substantially weaker, whereas mid–final and final–final groups are stronger and more stable. This indicates that multilayer gains are not tied to a single exact layer set, but appear more broadly among non-early depth compositions. Using normalized-depth groups reduces reliance on any single selected layer. The gap between very-early and non-early groups suggests that the earliest decoder depths are more fragile intervention sites. Additional exact combinations and a layer-skip ablation are provided in Appendix B.3 and Appendix B.4.

![](images/93cca86e4249911655a7684a49f597919e23367713163ad0fb82dc715cdfd363.jpg)  
Figure 5: Layer-wise single-layer sweep of SAGE interventions. We report relative performance improvement over the no-intervention baseline when applying SAGE to a single decoder layer at a time. The x-axis is normalized decoder depth $d ( \ell ) = \ell / ( L - 1 )$ ), allowing comparison across models with different decoder depths. The horizontal reference at 0% indicates the no-intervention baseline. Bars aggregate results across all benchmarks evaluated in our experiments, and error bars show the min–max range. Orange bars indicate the top-performing normalized depth positions.

Table 2: Depth-grouped summaries of multi-layer SAGE sets on TextVQA. Each value reports the mean accuracy within a coarse depth-composition group.
<table><tr><td>Depth composition</td><td>Mean Acc.</td></tr><tr><td colspan="2">2-layer sets</td></tr><tr><td>Very-early only Early-mixed Mid-mid Mid-final Final-final</td><td>19.19 49.85 61.96 62.60 62.42</td></tr><tr><td colspan="2">3-layer sets</td></tr><tr><td>Very-early ×3 Early ×3 Early-early-mid Early-mid-mid</td><td>2.35 55.39 59.10 60.73 62.11 62.10 62.40</td></tr></table>

Table 3: Fixed-layer result and ROI content control. Best selects the layer on the benchmark; L24 uses one fixed layer chosen by label-free diagnostics. Rand. replaces the guided ROI with a random token set of the same size, at the same layer and bias strength.
<table><tr><td colspan="4">SAGE</td></tr><tr><td>Benchmark</td><td>Base Best</td><td>L24</td><td>Rand.</td></tr><tr><td>TextVQA</td><td>44.0 55.4</td><td>54.6</td><td>43.1</td></tr><tr><td>GQA</td><td>57.1 62.8</td><td>62.3</td><td>57.0</td></tr><tr><td> $\mathrm { P O P E } _ { \mathrm { a c c } }$ </td><td>72.7</td><td>73.6 73.1</td><td>72.1</td></tr><tr><td> $\mathrm { P O P E } _ { \mathrm { F 1 } }$ </td><td>69.3</td><td>73.0 69.9</td><td>69.1</td></tr><tr><td> ${ \mathrm { S c i e n c e Q A } }$ </td><td>68.1</td><td>68.6 68.6</td><td>68.0</td></tr><tr><td>MME</td><td>1681</td><td>1707 1688</td><td>1677</td></tr><tr><td> $\mathbf { M M V } \mathrm { P _ { i m g } }$ </td><td>67.0</td><td>68.1 67.2</td><td>66.7</td></tr><tr><td> $\mathbf { M M V P } _ { \mathrm { p a i r } }$ </td><td>35.6</td><td>37.8 35.7</td><td>35.2</td></tr></table>

## 4.5 Fixed-Layer Result and ROI Content Control

Table 1 selects the intervention layer per benchmark. We therefore also evaluate a fixed-layer setting in which the intervention depth is chosen in advance, without benchmark labels, using the two diagnostics already introduced in Section 3.5: prompt-invariant Sink IoU (Fig. 3a) and top-1 spectral concentration $\rho _ { 1 }$ (Eq. (5), Fig. 3b). Both are computed from attention statistics alone and require no labels. In the LLaVA-1.5 diagnostic sweep, layer L24 attains the lowest value under both, indicating the weakest prompt-invariant sink structure among the examined depths. We select it before running any downstream evaluation and keep it fixed across all benchmarks, never reselecting per benchmark or per metric.

Table 3 reports this setting against the remeasured baseline under one unified protocol (LLaVA-1.5-7B). With one fixed layer, SAGE raises TextVQA from 44.0 to 54.6 and GQA from 57.1 to 62.3, while the changes on POPE, ScienceQA, MME, and MMVP are small. This is consistent with the premise that the intervention helps when the decisive evidence is localized.

ROI content control. Within this same fixed-layer setting, we further ask whether the gain depends on what the ROI contains. A logit bias of this strength reallocates attention regardless of where the mask points, so we hold the intervention layer, the bias strength, and the mask size fixed and replace the guided ROI with a randomly selected set of visual tokens. This control leaves every benchmark at or below baseline. The effect therefore requires the content of the target region and is not a generic consequence of perturbing the attention distribution.

## 5 Conclusion

In this paper, we uncover a consistent decoder attention pathology in modern VLMs: early and late layers repeatedly collapse onto prompt-invariant visual regions, while mid layers remain the main locus of prompt-conditioned grounding. Building on this diagnosis, we propose SAGE, a simple trainingfree inference-time method that reweights attention with a token-aligned ROI mask to prioritize relevant evidence. SAGE can be instantiated with backbone-derived ROI masks from standard vision backbones such as CLIP, ViT, and DINOv3. More broadly, our results suggest that decoder attention allocation is not only a symptom of grounding failure, but also a useful intervention point for improving visually grounded generation. We further show strong layer dependence and that intervening at appropriately chosen layers yields the most reliable improvements. Across multiple VLM families and benchmarks, SAGE yields broad improvements in many evaluated settings, suggesting that correcting decoder-side attention allocation is an effective handle for reducing visually unfaithful outputs.

## Limitations

SAGE relies on token-aligned ROI masks derived from standard vision backbones, and its effectiveness depends on the quality of these estimates. The three sources differ in what signal they encode: CLIP scores each patch by its similarity to the query embedding, while ViT and DINOv3 derive relevance from CLS-token attention and therefore capture image saliency. SAGE benefits from both kinds of signal, which suggests that steering toward task-relevant regions does not require the mask itself to be query-conditioned; extending saliencybased sources with query conditioning is a natural direction for future work. As illustrated in Figure 8, fragmented or misaligned ROI estimates can steer attention toward irrelevant visual tokens and suppress the true evidence. The random-ROI control in Section 4.5 bounds the other end of this dependence, since an uninformative mask of the same size and bias strength removes the gain entirely. A deployed system would therefore benefit from a confidence signal that decides when to intervene, for instance a margin on the patch relevance scores; we leave this, together with improved patchto-token alignment, to future work.

The intervention itself is deliberately simple. We apply one global $( \alpha , \beta )$ across all models and benchmarks to avoid per-task tuning, and the mask is held fixed across decoding steps and broadcast to all attention heads. This keeps SAGE easy to deploy and reproduce, but it also emphasizes the same region for every generated token, which suits questions with a single decisive region better than multi-region descriptions or multi-step reasoning. Per-step masks, head-selective steering, and permodel tuning of the bias strength are all compatible with the formulation in Eq. (1), as is extending the single-image mask to multi-image and video.

Two aspects of the diagnosis remain open. SAGE is sink-aware at the level of depth rather than of individual tokens: the PIS diagnostics identify where in the decoder steering is useful, and the additive bias then reallocates the attention budget through Eq. (6). This design needs no per-token sink classifier, but whether explicitly suppressing sink tokens would add to the depth-level effect is untested, as is whether the same normalized depth transfers across model families without re-running the diagnostic. Our characterization of promptinvariant sinks is likewise empirical: the structure is consistent across models and across two independent diagnostics, but we do not explain why early and late layers develop it while mid layers remain prompt-responsive, and we regard this formation mechanism as an interesting open problem.

## Ethical Considerations

This work studies decoder-side attention steering in vision-language models and aims to improve faithfulness to visual evidence. Although improved grounding may reduce visually unsupported outputs, the proposed method does not guarantee correctness and may still fail when ROI masks are inaccurate or unstable. Accordingly, SAGE should not be treated as a safety mechanism in high-stakes settings, and human oversight and task-specific validation remain necessary before deployment. Because the method operates on image inputs that may contain sensitive content, practitioners should also follow appropriate privacy and data-governance practices in downstream use.

## Acknowledgment

This research was supported by the AI Seoul Tech Research Support Program of the Seoul Future Foundation. This work was also supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [RS-2021-II211341, Artificial Intelligence Graduate School Program (Chung-Ang University)].

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, and 1 others. 2022. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716– 23736.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, and 1 others. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Boxu Chen, Ziwei Zheng, Le Yang, Zeyu Geng, Zhengyu Zhao, Chenhao Lin, and Chao Shen. 2025a. Seeing it or not? interpretable vision-aware latent steering to mitigate object hallucinations. arXiv preprint arXiv:2505.17812.

Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, and 1 others. 2023. Pali-x: On scaling up a multilingual vision and language model. arXiv preprint arXiv:2305.18565.

Xi Chen, Xiao Wang, Soravit Changpinyo, Anthony J Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, and 1 others. 2022. Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, and 1 others. 2024. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198.

Zhiyuan Chen, Yuecong Min, Jie Zhang, Bei Yan, Jiahao Wang, Xiaozhen Wang, and Shiguang Shan. 2025b. A survey of multimodal hallucination evaluation and detection. arXiv preprint arXiv:2507.19024.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems, 36:49250–49267.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, and 1 others. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

Parsa Esmaeilkhani and Longin Jan Latecki. 2026. Direct visual grounding by directing attention of visual tokens. In 2026 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), pages 5787–5797. IEEE.

Wenfeng Feng, Hongxiang Wang, Jianlong Wang, Xin Zhang, Jingjing Zhao, Yueyue Liang, Xiang Chen, and Duokui Han. 2026. Edit: Enhancing vision transformers by mitigating attention sink through an encoder-decoder architecture. In International Conference on Optoelectronics, Computer Science, and Algorithms (OCSA 2025), volume 14008, pages 246–259. SPIE.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and 1 others. 2026. Mme: A comprehensive evaluation benchmark for multimodal large language models. Advances in Neural Information Processing Systems, 38.

Gregor Geigle, Radu Timofte, and Goran Glavaš. 2024. Does object grounding really reduce hallucination of large vision-language models? arXiv preprint arXiv:2406.14492.

Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min Lin. 2025. When attention sink emerges in language models: An empirical view. In International Conference on Learning Representations, volume 2025, pages 97114–97144.

Wen Huang, Hongbin Liu, Minxin Guo, and Neil Gong. 2024. Visual hallucinations of multi-modal large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 9614– 9631.

Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6693–6702. IEEE.

Mingi Jung, Saehyung Lee, Eunji Kim, and Sungroh Yoon. 2025. Visual attention never fades: Selective progressive attention recalibration for detailed image captioning in multimodal large language models. arXiv preprint arXiv:2502.01419.

Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. 2025. See what you are told: Visual attention sink in large multimodal models. In International Conference on Learning Representations, volume 2025, pages 87676–87703.

Samir Khaki, Junxian Guo, Jiaming Tang, Shang Yang, Yukang Chen, Konstantinos N Plataniotis, Yao Lu, Song Han, and Zhijian Liu. 2025. Sparsevila: Decoupling visual sparsity for efficient vlm inference. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 23784–23794. IEEE.

Jeonghwan Kim and Heng Ji. 2024. Finer: Investigating and enhancing fine-grained visual concept recognition in large vision language models. arXiv preprint arXiv:2402.16315.

Junho Kim, Hyun J Kim, Yeon J Kim, and Yong M Ro. 2024. Code: Contrasting self-generated description to combat hallucination in large multi-modal models. Advances in Neural Information Processing Systems, 37:133571–133599.

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. 2024. Mitigating object hallucinations in large visionlanguage models through visual contrastive decoding. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13872– 13882. IEEE.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. 2025. Llava-interleave: Tackling multi-image, video, and 3d in large multimodal models. In International Conference on Learning Representations, volume 2025, pages 81182–81199.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023a. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PmLR.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023b. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355.

Yuan-Hong Liao, Rafid Mahmood, Sanja Fidler, and David Acuna. 2025. Can large vision-language models correct semantic grounding errors by themselves? In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14667–14678.

Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. 2024a. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024b. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems, 36:34892– 34916.

Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, and 1 others. 2025. Nvila: Efficient frontier visual language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4122–4134.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521.

Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. 2024. Large language models: A survey. arXiv preprint arXiv:2402.06196.

Yeji Park, Deokyeong Lee, Junsuk Choe, and Buru Chang. 2025. Convis: Contrastive decoding with hallucination visualization for mitigating hallucinations in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 6434–6442.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, and 1 others. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR.

Hoigi Seo, Dong Un Kang, Hyunjin Cho, Joohoon Lee, and Se Young Chun. 2026. On epistemic uncertainty of visual tokens for object hallucinations in large vision-language models. Advances in Neural Information Processing Systems, 38:130605–130660.

Yuheng Shi, Xiaohuan Pei, Minjing Dong, and Chang Xu. 2026. Catching the details: Self-distilled roi predictors for fine-grained mllm perception. In International Conference on Learning Representations, volume 2026, pages 136133–136155.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, and 1 others. 2025. Dinov3. arXiv preprint arXiv:2508.10104.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326.

Lexiang Tang, Xianwei Zhuang, Bang Yang, Zhiyuan Hu, Hongxiang Li, Lu Ma, Jinghan Ru, and Yuexian Zou. 2026. Not all tokens and heads are equally important: Dual-level attention intervention for hallucination mitigation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 9439–9447.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. 2024. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and 1 others. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Chien Van Nguyen, Xuan Shen, Ryan Aponte, Yu Xia, Samyadeep Basu, Zhengmian Hu, Jian Chen, Mihir Parmar, Sasidhar Kunapuli, Joe Barrow, and 1 others. 2025. A survey on small language models. In Proceedings ofthe 15th International Conference on Recent Advances in Natural Language Processing-Natural Language Processing in the Generative AI Era, pages 807–821.

Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, and 1 others. 2023. Efficient large language models: A survey. arXiv preprint arXiv:2312.03863.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, and 1 others. 2024. Qwen2- vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Xinyi Wu, Amir Ajorlou, Yifei Wang, Stefanie Jegelka, and Ali Jadbabaie. 2024. On the role of attention masks and layernorm in transformers. Advances in Neural Information Processing Systems, 37:14774– 14809.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2023. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2023. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490.

Jianfei Zhao, Feng Zhang, Xin Sun, Chong Feng, and Zhixing Tan. 2026. Tell model where to look: Mitigating hallucinations in mllms by vision-guided attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 32582–32591.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, and 1 others. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

## A Additional Implementation Details

## A.1 Hyperparameters and Training Hardware Settings

Table 4 summarizes the key configuration choices used in our SAGE implementation. Unless otherwise noted, these settings are shared across the experiments reported in the main paper. We use $( \alpha , \beta ) \ = \ ( 5 , 2 )$ as the default bias strengths unless otherwise stated. For all models we use deterministic greedy decoding (do\_sample = False) for every arm, including baselines and comparison methods, and otherwise follow the standard decoding settings recommended by each model family. All experiments are run on NVIDIA GPUs (A6000) using FP16 mixed precision.

## A.2 Intervention Scope and Implementation

SAGE is implemented as a training-free, inferencetime intervention that modifies attention allocation inside selected decoder layers without updating model parameters. In our implementation, we instrument the self-attention modules of target layers $\ell \in \mathcal { L } _ { \mathrm { i n t } }$ so that the additive bias in Eq. (1) is applied during attention computation.

Scope. The intervention uses a token-aligned ROI mask m $\in \{ 0 , 1 \} ^ { N }$ and applies the additive bias only to attention logits corresponding to visualtoken key positions; text-token keys remain unchanged. Within each selected layer, the same ROI mask is broadcast to all heads and all query positions in the layer’s attention operation. All settings are kept fixed within each experiment configuration.

Layer configuration. The intervention layer set ${ \mathcal { L } } _ { \mathrm { i n t } }$ supports both single-layer and multi-layer modes and is specified by a layer index list or range in the experiment scripts $( \mathrm { e . g . , \mathrm { \Omega ^ { * } O ^ { * } , \mathrm { \Omega ^ { * } O , 5 , 1 0 ^ { * } } } }$ , “0– 5”).

Algorithm 1 Implementation sketch of SAGE in  
strumentation (inference-time)   
Require: Model M, ROI mask m, layer set ${ \mathcal { L } } _ { \mathrm { i n t } }$   
strengths $\alpha , \beta$   
1: Configure M to apply Eq. (1) in all $\ell \in \mathcal { L } _ { \mathrm { i n t } }$   
2: Y ← M.generate $( I , Q )$   
3: return Y

## A.3 ROI Mask Construction Details

We summarize the local ROI construction pipeline used in our main experiments and the alignment from patch-level relevance maps to token-aligned ROI masks.

Local backbone-derived ROI masks. We instantiate SAGE with ROI masks derived from standard vision backbones such as CLIP, ViT, and DINOv3. For each image–query pair, a sourcespecific patch-level relevance score is computed and thresholded to obtain a binary patch mask.

Patch-to-token alignment. The binary patch mask is then aligned to the visual-token grid used by each evaluated VLM. Although the effective grid size can differ across model families depending on the vision encoder and preprocessing pipeline, the mapping from patch-level relevance to token-aligned mask is fixed within each experiment.

Model-specific patch grids. Different VLM families may use different effective patch grids depending on the vision encoder and image preprocessing. We align the patch-level ROI mask to the corresponding model’s visual-token grid using the model’s configured grid size and release the exact settings in the experiment scripts.

## A.4 Benchmark Protocols and Evaluation Prompts

We follow the standard evaluation protocols released by each benchmark and keep decoding settings fixed within each benchmark across all compared methods. Below we briefly summarize what each benchmark probes in VLMs and the primary metric we report.

TextVQA. (Singh et al., 2019) OCR-centric VQA that tests whether a model can read scene text and ground it to answer questions. We report VQA-style exact-match accuracy.

POPE. (Li et al., 2023b) A hallucination/faithfulness diagnostic using yes/no questions about object existence. We report accuracy and F1 following the official protocol.

GQA. (Hudson and Manning, 2019) A compositional visual reasoning benchmark that probes object/attribute/relation reasoning with grounding. We report standard accuracy on the evaluated split.

ScienceQA. (Lu et al., 2022) A multimodal multiple-choice benchmark for scientific reasoning requiring integration of perception and reasoning. We report multiple-choice accuracy.

MME. (Fu et al., 2026) A broad diagnostic suite covering perception (e.g., OCR, count, color) and cognition (e.g., commonsense, numerical reasoning). We report the aggregated score under the

Table 4: Key hyperparameters and settings for SAGE. We use $( \alpha , \beta ) = ( 5 , 2 )$ as the default bias strengths unless otherwise stated.
<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Value / Description</td></tr><tr><td rowspan=1 colspan=1>Local VLM</td><td rowspan=1 colspan=1>Vision-encoder + decoder-only LLM VLMs (e.g., LLaVA-1.5 (Liuet al., 2024b)/Qwen2-VL (Wang et al., 2024)/InternVL3 (Zhu et al.,2025)/NVILA (Liu et al., 2025)) with standard preprocessing and de-coding settings.</td></tr><tr><td rowspan=1 colspan=1>ROI sources</td><td rowspan=1 colspan=1>Patch-level ROI masks derived from standard pretrained vision backbones(CLIP/ViT/DINOv3).</td></tr><tr><td rowspan=1 colspan=1>Patch relevance scoring</td><td rowspan=1 colspan=1>Backbone-specific patch relevance scores are thresholded to form a binaryROI mask; the same scoring rule is used consistently within each source.</td></tr><tr><td rowspan=1 colspan=1>Patch-to-token mapping</td><td rowspan=1 colspan=1>The patch-level mask is aligned to the visual-token grid used by theevaluated VLM; the mapping is fixed per model and preprocessing config-uration.</td></tr><tr><td rowspan=1 colspan=1>Patch grid</td><td rowspan=1 colspan=1>Model-specific patch grid size determined by the vision encoder andpreprocessing; grid configuration is fixed per model and recorded in theexperiment scripts.</td></tr><tr><td rowspan=1 colspan=1>Intervention layers</td><td rowspan=1 colspan=1>Layer set ${ \mathcal { L } } _ { \mathrm { i n t } }$ specified as a list or range (single-layer or multi-layer),fixed per experiment.</td></tr><tr><td rowspan=1 colspan=1>Bias strengths</td><td rowspan=1 colspan=1>Default $( \alpha , \beta ) = ( 5 , 2 )$ for ROI amplification and non-ROI suppressionin Eq. (1).</td></tr><tr><td rowspan=1 colspan=1>Decoding</td><td rowspan=1 colspan=1>Greedy do_sample = False; maximum new tokens set per model family;decoding settings are held fixed within each benchmark.</td></tr></table>

official protocol.

MMVP. (Tong et al., 2024) A paired-image visual pattern benchmark that stresses sensitivity to subtle visual differences and discourages language priors. We report strict pair accuracy (a pair is correct only if both questions are answered correctly).

MMVet. (Yu et al., 2023) An integrated capability benchmark with open-ended questions scored by an LLM-based evaluator. We report the resulting score following the benchmark evaluator protocol. Evaluation prompts. We use each benchmark’s recommended input formatting and evaluation scripts; any additional prompt templates used in our runs are included in the released supplementary materials.

Artifact use. We use publicly released research artifacts, including model checkpoints, vision backbones, benchmark datasets, and official evaluation protocols, under their respective licenses and terms. We use these artifacts only for research evaluation and do not redistribute the underlying datasets or model weights.

## A.5 Detailed Algorithm

Algorithm 2 summarizes the full SAGE pipeline. Given an input image I, a query $Q ,$ and a local VLM M (vision encoder + decoder-only LLM), SAGE performs a training-free intervention during generation by selectively biasing the logits of visual tokens at a chosen set of decoder layers ${ \mathcal { L } } _ { \mathrm { i n t } }$

Step 1: ROI mask construction. SAGE first constructs a binary ROI mask m over the visual tokens produced by the vision encoder:

$$
\mathbf { m } \gets \mathbf { G E T R O I M A S K } ( I , Q , \mathcal { S } ) .
$$

Here, the ROI source S is a local vision backbone such as CLIP, ViT, or DINOv3. For the selected source, we compute a patch-level relevance map, threshold it to obtain a binary patch mask, and align the result to the visual-token grid of the evaluated VLM. The output mask $\mathbf { m } \in \{ 0 , 1 \} ^ { N _ { \iota } }$ marks which visual tokens belong to the guided ROI.

Step 2: Enable logit-level steering at selected layers. Next, SAGE activates a lightweight forward hook that applies Eq. (1) to the visual-token logits at each layer $\ell \in \mathcal { L } _ { \mathrm { i n t } }$

$$
\operatorname { E N A B L E S T E E R I N G } ( M , \mathcal { L } _ { \mathrm { i n t } } , \mathbf { m } , \alpha , \beta ) .
$$

Intuitively, α controls the strength of encouraging ROI tokens, while $\beta$ controls the strength of suppressing non-ROI tokens (as defined in Eq. (1)). Importantly, this intervention does not require gradient updates or any modification to model parameters; it only alters the logits used to form attention over visual tokens during decoding. By biasing visual-token logits, this intervention implicitly reshapes the attention distribution over visual tokens during decoding, without any parameter updates.

Algorithm 2 SAGE: Local ROI-Guided, Training-  
Free Attention Steering   
Require: Image I, Query Q, local VLM M (vi  
sion encoder + decoder-only LLM)   
Require: ROI source S (e.g., CLIP/ViT/DINOv3   
patch mask)   
Require: Intervention layer set ${ \mathcal { L } } _ { \mathrm { i n t } }$ , hyperparam  
eters $\alpha , \beta$   
Ensure: Answer Y   
1: m ← GETROIMASK $( I , Q , S )$   
2: ENABLESTEERING(M, L<sub>int</sub>, m, α, β) ▷ apply   
Eq. (1) to visual-token logits at selected layers   
3: $Y  M .$ generate $( I , Q )$   
4: DISABLESTEERING(M)   
5: return $Y$

Step 3: Generate with SAGE enabled. With steering enabled, the model generates the answer as usual:

$$
Y \gets M . \mathsf { g e n e r a t e } ( I , Q ) .
$$

All other components of decoding (e.g., sampling strategy, max tokens, temperature) remain unchanged unless specified elsewhere.

Step 4: Disable steering and return. Finally, SAGE removes the hook to restore the original model behavior:

## DISABLESTEERING(M),

and returns the generated output Y . This explicit enable/disable design ensures SAGE is modular and can be applied on a per-query basis without persistent side effects.

Practical notes. In practice, ${ \mathcal { L } } _ { \mathrm { i n t } }$ can be chosen as a single layer or a small set of layers to trade off effectiveness and overhead. The only additional per-query cost comes from computing m and applying a constant-time logit bias at the selected layers; no extra training or finetuning is required.

## A.6 Hyperparameters and Sensitivity

SAGE is controlled by two scalar hyperparameters in Eq. (1): α, the ROI-side amplification, and $\beta ,$ the non-ROI-side suppression. Table 5 reports a seven-point sensitivity study over $( \alpha , \beta )$ on MMVP with the ROI source and intervention layer held fixed. Performance is not flat across settings: onesided variants (pull-only, push-only) and overly aggressive bias both fall well below the default, spanning roughly nine points. We therefore fix $( \alpha , \beta ) ~ = ~ ( 5 , 2 )$ globally across all models and benchmarks rather than tuning per task, and treat the bias strength as a parameter that must be kept moderate rather than one that can be pushed for larger gains.

Table 5: Sensitivity of SAGE to logit-level intervention strength $( \alpha , \beta )$ . We report MMVP (%).
<table><tr><td>Variant</td><td> $( \alpha , \beta )$ </td><td>MMVP (%)</td></tr><tr><td>No re-weighting</td><td>(0,0)</td><td>61.48</td></tr><tr><td>Pull-only</td><td>(0, 2)</td><td>54.07</td></tr><tr><td>Push-only</td><td>(5,0)</td><td>57.49</td></tr><tr><td>Weak</td><td>(3, 1)</td><td>61.91</td></tr><tr><td>Default</td><td>(5, 2)</td><td>62.96</td></tr><tr><td>Strong</td><td>(7, 3)</td><td>59.39</td></tr><tr><td>Aggressive</td><td>(9,4)</td><td>55.93</td></tr></table>

When reporting sensitivity, we recommend fixing all other settings (ROI source, layer set, decoding parameters) and varying only one parameter at a time.

## B Additional Layer-wise Analyses

## B.1 Per-benchmark Single-layer Sweep Curves

Figures 9–11 provide per-benchmark single-layer sweep curves for LLaVA-1.5 using local ROI masks derived from CLIP, ViT, and DINOv3, respectively. For each decoder layer index ℓ, we apply SAGE to a single layer ${ \mathcal { L } } _ { \mathrm { i n t } } = \{ \ell \}$ and report the resulting benchmark score using the official evaluation scripts. Across sources and tasks, the response to intervention is clearly layer-dependent: some layers yield consistent gains over the nointervention baseline, while others are neutral or occasionally degrade performance. These curves complement the aggregated summaries in the main paper and show that the effectiveness of steering is not tied to a single local ROI source.

## B.2 Exact Best-Layer Indices for Main Results

For transparency, we list in Table 6 the exact singlelayer indices used to produce the Best results in Table 1. These indices are provided as supplementary detail; in the main text, cross-model discussion uses normalized decoder depth rather than raw layer number.

## B.3 Additional Multi-layer Set Comparisons

To complement the group-level summaries in Table 2, we provide a small number of illustrative exact layer combinations for the multi-layer setting. Rather than exhaustively listing every tested set, we report selected examples that reflect the stronger non-early compositions and a few failure-prone early-layer cases.

Tables 7–9 summarize illustrative 2-layer and 3- layer configurations, together with selected degradation cases. Across these examples, sets concentrated on mid and/or final depths tend to produce stronger performance than sets involving the earliest layers. These examples are consistent with the depth sensitivity observed in the main paper and help ground the group-level averages with concrete layer indices.

## B.4 Layer Skip Ablation

To further probe depth-wise roles of the decoder, we perform a layer-skip ablation on LLaVA-1.5 that removes computation from selected decoder layers at inference time. For a chosen skip set $\mathcal { L } _ { \mathrm { s k i p } } ,$ we bypass each layer $\ell \in \mathcal L _ { \mathrm { s k i p } }$ by forwarding its input hidden states to the next layer without applying the layer’s transformation. This ablation adds no new parameters and isolates how sensitive the model’s final behavior is to perturbations at different decoder depths.

Figure 6 reports the impact of skipping different layers on downstream performance. Skipping is not uniformly tolerated across depth: certain layers lead to more noticeable degradation than others. This provides complementary evidence to the single-layer intervention sweep in the main paper and supports the interpretation that some early decoder layers can be more sensitive to perturbations.

## C ROI Sources from Standard Vision Backbones

SAGE only requires a token-aligned binary mask $\textbf { m } \in \{ 0 , \overset { \cdot } { 1 } \} ^ { N }$ over visual tokens. In this section, we summarize the three ROI sources used in our study: CLIP, ViT, and DINOv3. These sources produce different patch-level relevance patterns, but the intervention in Eq. (1) is identical across sources; only the procedure that produces m changes.

![](images/034b902afb8f80134d696fbf14c8117668c63c2726381b4c87db98a4720d99c7.jpg)

![](images/9ab19641cfcd4c64f87d8fbf0bacfafd7dc6a827c0c841a54eb94413230c9ea1.jpg)

![](images/b86d23f65702a84093ec0852176076fc2274557a4270a6271f3d460e0bcb92af.jpg)

![](images/71dccedd2946781c8737b3948e201c4ee7f79fcffa37e39aaf04af83555c51a5.jpg)  
Figure 6: Layer-skip ablation on LLaVA-1.5. For each decoder layer, we skip only that layer at inference and evaluate performance. The top row reports TextVQA and POPE accuracy versus the skipped depth, and the bottom row shows the change relative to the no-skip baseline.

## D Additional Controlled Comparisons

This appendix collects two comparisons that share the fixed-protocol setting of Section 4.5 but are not needed to follow the main argument. Both use no benchmark-specific selection of the intervention layer.

## D.1 Comparison with an Inference-Time Decoding Baseline

Section 4.5 compares the fixed-layer setting against a re-measured baseline and a matched random-ROI control, but not against another published intervention. To place SAGE relative to existing inferencetime methods, we run visual contrastive decoding (VCD (Leng et al., 2024)) under exactly the same protocol with its official hyperparameters $( \alpha = 1 . 0 $ $\beta ~ = ~ 0 . 1$ , noise step 500). VCD reaches 50.8 on TextVQA and 59.5 on GQA, and is neutral-tonegative on POPE (−0.2) and ScienceQA (−0.3). Fixed-layer SAGE reaches 54.6 and 62.3 on the same two benchmarks (Table 3). The two methods therefore differ not only in magnitude on the localized-evidence benchmarks but also in whether they are neutral elsewhere. This comparison covers a single method on a single model family, and we do not read it as a general ranking of inference-time interventions.

## D.2 Fixed-Protocol Control Across Model Families

The controls in Section 4.5 are established on $\mathrm { L L a V A } { - } 1 . 5 $ . To check that the fixed-protocol setting is not specific to that model, we repeat it across four families (LLaVA-1.5, Qwen2-VL, InternVL3, NVILA) using one fixed target layer per family, fixed $( \alpha , \beta ) = ( 5 , 2 )$ , a fixed top-48 mask, and no benchmark-specific selection at any point. Under this setting the same push–pull intervention yields moderate gains: GQA +2.55, TextVQA +1.58, and MMVP-pair +6.67. We report them as evidence that the intervention is not confined to one architecture, not as a like-for-like comparison with the main table.

Table 6: Exact model-specific best-layer indices used for the Best results in Table 1. Indices are raw layer numbers within each model and should not be compared directly across architectures; cross-model discussion in the main text uses normalized decoder depth.
<table><tr><td>Model</td><td>TextVQA</td><td>POPE</td><td>GQA</td><td>SQA</td><td>MME</td><td>MMVet</td><td>MMVP Img</td><td></td><td>MMVP Pair MMVP-O Img</td><td>MMVP-O Pair</td></tr><tr><td>LLaVA 1.5-13B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ SAGE (CLIP)</td><td>L17</td><td>L3</td><td>L20</td><td>L18</td><td>L24</td><td>L20</td><td>L26</td><td>L26</td><td>L18</td><td>L6</td></tr><tr><td>+ SAGE (ViT)</td><td>L29</td><td>L10</td><td>L19</td><td>L20</td><td>L22</td><td>L18</td><td>L18</td><td>L18</td><td>L2</td><td>L18</td></tr><tr><td>+ SAGE (DINOv3)</td><td>L30</td><td>L10</td><td>L30</td><td>L25</td><td>L16</td><td>L20</td><td>L26</td><td>L26</td><td>L18</td><td>L18</td></tr><tr><td>Qwen2-VL-7B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ SAGE (CLIP)</td><td>L11</td><td>L18</td><td>L21</td><td>L15</td><td>L25</td><td>L14</td><td>L18</td><td>L18</td><td>L15</td><td>L15</td></tr><tr><td>+ SAGE (ViT)</td><td>L13</td><td>L18</td><td>L15</td><td>L15</td><td>L20</td><td>L17</td><td>L27</td><td>L27</td><td>L15</td><td>L15</td></tr><tr><td>+ SAGE (DINOv3)</td><td>L16</td><td>L15</td><td>L18</td><td>L18</td><td>L27</td><td>L14</td><td>L18</td><td>L18</td><td>L12</td><td>L12</td></tr><tr><td>InternVL3-2B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ SAGE (CLIP)</td><td>L11</td><td>L18</td><td>L21</td><td>L15</td><td>L25</td><td>L25</td><td>L18</td><td>L18</td><td>L8</td><td>L8</td></tr><tr><td>+ SAGE (ViT)</td><td>L15</td><td>L18</td><td>L14</td><td>L19</td><td>L24</td><td>L17</td><td>L27</td><td>L27</td><td>L16</td><td>L16</td></tr><tr><td>+ SAGE (DINOv3)</td><td>L9</td><td>L18</td><td>L27</td><td>L19</td><td>L27</td><td>L15</td><td>L21</td><td>L21</td><td>L13</td><td>L13</td></tr><tr><td>NVILA-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ SAGE (CLIP)</td><td>L19</td><td>L20</td><td>L12</td><td>L15</td><td>L15</td><td>L10</td><td>L10</td><td>L10</td><td>L20</td><td>L20</td></tr><tr><td>+ SAGE (ViT)</td><td>L13</td><td>L20</td><td>L18</td><td>L17</td><td>L16</td><td>L12</td><td>L15</td><td>L15</td><td>L13</td><td>L13</td></tr><tr><td>+ SAGE (DINOv3)</td><td>L10</td><td>L10</td><td>L15</td><td>L18</td><td>L5</td><td>L21</td><td>L15</td><td>L15</td><td>L20</td><td>L20</td></tr></table>

Table 7: Illustrative 2-layer SAGE sets on TextVQA (LLaVA-1.5). These examples are selected to represent stronger non-early depth compositions.
<table><tr><td>Layer set</td><td>Depth composition</td><td>Accuracy (%)</td></tr><tr><td>(14,16)</td><td>Mid-mid</td><td>62.43</td></tr><tr><td>(17,21)</td><td>Mid-final</td><td>62.72</td></tr><tr><td>(28,31)</td><td>Final-final</td><td>62.58</td></tr></table>

Table 8: Illustrative 3-layer SAGE sets on TextVQA (LLaVA-1.5). These examples are selected to represent stronger multi-layer compositions away from the earliest decoder layers.
<table><tr><td>Layer set</td><td>Depth composition</td><td>Accuracy (%)</td></tr><tr><td>(14,18,21)</td><td>Mid ×3</td><td>62.33</td></tr><tr><td>(17,24,30)</td><td>Mid-final-final</td><td>62.63</td></tr><tr><td>(24,27,30)</td><td>Final ×3</td><td>62.53</td></tr></table>

Table 9: Illustrative failure-prone early-layer sets on TextVQA (LLaVA-1.5). These examples show that sets concentrated near the earliest decoder layers can still lead to substantial degradation.
<table><tr><td>Layer set</td><td>Depth composition</td><td>Accuracy (%)</td></tr><tr><td>(0,2)</td><td>Very-early only</td><td>12.07</td></tr><tr><td>(0,2,4)</td><td>Very-early ×3</td><td>7.12</td></tr><tr><td>(1,2,3)</td><td>Very-early ×3</td><td>1.12</td></tr></table>

## D.3 ROI Sources

Backbone-derived ROI (CLIP/ViT/DINOv3). We derive ROI masks from standard pretrained vision backbones without explicit detector-based grounding. For each source, we compute patchlevel relevance scores and threshold them to obtain a binary patch mask, which is then aligned to the visual-token grid used by the evaluated VLM. Although the resulting masks can differ in sharpness and spatial coverage across sources, all three provide usable local guidance signals for SAGE.

## D.4 Single-layer Sweep with Backbone-Derived ROI Masks

Figures 9–11 report additional single-layer sweep results using ROI masks derived from CLIP, ViT, and DINOv3. These results further support that SAGE remains effective under multiple local ROI construction mechanisms.

## D.5 Qualitative Comparison of ROI Masks Across Sources

Figure 8 visualizes ROIs produced by different local ROI sources for the same image–query pairs. CLIP, ViT, and DINOv3 can produce different mask shapes and degrees of spatial concentration, reflecting differences in their patch-level relevance patterns. Despite these differences, all three sources can be converted into token-aligned masks and used by SAGE without changing the intervention rule.

## E Mechanistic Analysis Details

This appendix specifies the exact definitions and computation procedures used in our mechanistic analyses (sink structure, spectral dominance, and attention-mass redistribution).

## E.1 Constructing the Logit Matrix for Spectral Analysis

Fix a decoder layer ℓ and head h. For each evaluation instance, we collect pre-softmax attention logits from text-query positions to visual-token keys. Let $\mathbf { a } _ { p , t } ^ { ( \ell , h ) } \in \bar { \mathbb { R } } ^ { N }$ denote the restricted logit vector over the N visual-token keys at decoding step t for prompt (instance) $p .$ We stack these vectors across instances and decoding steps into a matrix

$$
\mathbf { A } ^ { ( \ell , h ) } \in \mathbb { R } ^ { M \times N } , \qquad \mathbf { A } ^ { ( \ell , h ) } [ r , : ] = \mathbf { a } _ { p , t } ^ { ( \ell , h ) } ,\tag{7}
$$

where r indexes the pair $( p , t )$ and M is the total number of stacked vectors. We then compute $\mathbf { A } ^ { ( \ell , h ) } = \mathbf { U } \Sigma \mathbf { V } ^ { \top }$ and report spectral concentration using

$$
\rho _ { k } ( \mathbf { A } ) = \frac { \sum _ { j = 1 } ^ { k } \sigma _ { j } ^ { 2 } } { \sum _ { j = 1 } ^ { \operatorname* { m i n } ( M , N ) } \sigma _ { j } ^ { 2 } } ,\tag{8}
$$

where $\{ \sigma _ { j } \}$ are singular values.

## E.2 Prompt-Invariant Sink Definition and Sink IoU

For each instance p and decoding step t, we convert the visual-token attention weights into a set of sink tokens by selecting the top-r% tokens by attention mass. We use $r = 5$ throughout unless otherwise specified. Given two sink sets $S _ { p }$ and $S _ { p ^ { \prime } }$ computed from different prompts (or different queries) on the same image, we measure prompt-invariance using the intersection-over-union (IoU):

$$
\operatorname { I o U } ( S _ { p } , S _ { p ^ { \prime } } ) = { \frac { | S _ { p } \cap S _ { p ^ { \prime } } | } { | S _ { p } \cup S _ { p ^ { \prime } } | } } .\tag{9}
$$

We report mean IoU with standard deviation across prompt pairs.

## E.3 Attention-Mass Metrics: Sink Mass and ROI Mass

Let $\mathbf { p } \in \mathbb { R } ^ { N }$ be attention weights over visual keys (after softmax) and let m $\in \{ 0 , 1 \} ^ { N }$ be the ROI

mask. We define ROI mass as the total attention assigned to ROI tokens:

$$
\mathrm { M a s s } _ { \mathrm { R O I } } ( { \bf p } ) = \sum _ { i = 1 } ^ { N } { \bf p } [ i ] \cdot { \bf m } [ i ] .\tag{10}
$$

We define sink mass as the total attention assigned to the sink-token set $S \left( { \mathrm { t o p } } { - } r \% \right)$ tokens by attention mass):

$$
\mathrm { M a s s } _ { \mathrm { s i n k } } ( \mathbf { p } ) = \sum _ { i \in S } \mathbf { p } [ i ] .\tag{11}
$$

Unless stated otherwise, we use the same $r = 5$ as in Appendix E.2.

## E.4 Implementation Notes for Mechanistic Measurements

All mechanistic measurements are computed under matched decoding settings between baseline and SAGE runs. When reporting layer-wise results, we either fix a representative head or average the reported quantities across heads within the layer; we explicitly state the choice in each figure caption.

## E.5 Evidence Retention to the Final Layer

To test whether the steering effect survives to late decoding rather than being a single-layer artifact, we measure a retention ratio: final-layer ROI attention mass relative to its mid-layer peak. On LLaVA-1.5, averaged over four benchmarks, guided steering raises this ratio from 0.77 to 0.96, indicating reduced mid-to-final evidence loss and a change in routing rather than a local change in one layer’s heatmap.

## F Qualitative Examples

We provide qualitative examples to illustrate how SAGE changes the model’s visual evidence usage during generation. Figure 7 compares baseline predictions with SAGE under the same input image and query, together with attention visualizations over the image. Across examples, baseline models can allocate attention to visually salient but query-irrelevant regions, which can lead to weak grounding in fine-grained questions. In contrast, with the same ROI guidance, SAGE shifts attention toward query-relevant regions and encourages the model to rely on localized visual evidence when forming its answer.

In Figure 7, SAGE is particularly helpful on queries that require fine-grained visual evidence. For instance, when asked to read the text on a distant sign, the baseline answer can be driven by a nearby salient cue and miss the actual characters, while SAGE increases attention on the sign region and enables the correct reading. Similarly, for the aircraft example, focusing on the front fuselage is necessary to identify the printed number; SAGE concentrates attention on this localized area and corrects the prediction. These cases suggest that improving ROI-focused attention can be beneficial for tasks such as reading small text or verifying localized details.

![](images/f720f516712c92fec8930311d60baedc5d610a380597a56af3b57cdfb83f2297.jpg)  
“What is on the sign with two down arrows?”

![](images/d7bd570df5f2db19a367819e8c4fd5041794637148aa1ca9988e65f9d905a676.jpg)

![](images/e98020afcf2d8c666e7dd93e3a58b85da1b45d6d6f3b9c141d6d659f3229473a.jpg)  
“What number is written on the front of the plane?”

![](images/8dbf1a09f76739e15c7a08d54e9fb432a5701ed160e8e5eb81933735adb3253c.jpg)  
Figure 7: Qualitative comparison of baseline vs. SAGE. For each example, we show the input image and query, baseline and SAGE predictions, and attention visualizations. SAGE encourages attention to concentrate on query-relevant visual evidence, improving grounding in fine-grained cases.

Q: Who is wearing the helmet? (GT: man)  
![](images/74a8fa1889371c2ebb9a8638aeb07620c81408df1085fb92aafa41d8eebb2c09.jpg)  
Figure 8: Failure case under unstable local ROI estimates. Different local ROI sources can produce fragmented or partially misaligned masks, which may steer attention away from the true evidence and lead SAGE to fail.

inaccurate or fragmented local masks may steer attention toward irrelevant visual tokens and suppress the true evidence. As a result, the model can still produce an incorrect prediction even though logit-level steering is applied.

## F.1 Failure Case: Sensitivity to Unstable Local ROI Estimates

Figure 8 shows a representative failure case where SAGE does not correct the answer when the local ROI estimate is unstable. Since SAGE intervenes only on the region specified by the ROI source,

![](images/90422be72cc1abce0d842649db0054e9a00e9bd3665db0547abd859326e4f1cb.jpg)  
Figure 9: Single-layer sweep with a CLIP-derived ROI mask. We report per-benchmark performance as a function of the intervened layer using a CLIP-based patch-level ROI mask.

![](images/25be953e3845f9a26dd9a4325648fb5cef3c1a73195c5dce5f4926b4a044f9a3.jpg)  
Figure 10: Single-layer sweep with a ViT-derived ROI mask. We report per-benchmark performance as a function of the intervened layer using a ViT-based patch-level ROI mask.

![](images/3d5bbec598fb3d2f05c49e3a30ca4e0e9353797503612518c6a68d0803f83e2c.jpg)  
Figure 11: Single-layer sweep with a DINOv3-derived ROI mask. We report per-benchmark performance as a function of the intervened layer using a DINOv3-based patch-level ROI mask.