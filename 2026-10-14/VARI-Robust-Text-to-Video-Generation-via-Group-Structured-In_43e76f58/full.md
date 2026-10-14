# VARI: Robust Text-to-Video Generation via Group-Structured Intent Alignment

Mingxuan Zhu<sup>1,2∗†</sup>, Hohei Chan<sup>2,3∗</sup>, Shiyuan Liu<sup>2</sup>,

Yuanxing Zhang<sup>2</sup>, Lu Chen<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University

<sup>2</sup>KlingAI

<sup>3</sup>South China University of Technology

## Abstract

Modern text-to-video (T2V) models generate visually compelling videos, yet meaning-preserving changes in prompt wording can substantially alter semantic realization and overall quality. Existing prompt rewriting modifies the inference input, whereas conventional post-training constructs supervision from one prompt formulation at a time; neither explicitly optimizes reliability across equivalent forms. We introduce VARI (Verified Alignment over Representations of Intent), a group-structured framework that treats automatically verified formulations of the same visual intent as a shared supervision unit. VARI builds matched form–seed rollout pools, cross-scores each video against all forms, and applies a wording-independent structured intent audit to identify valid realizations and hard semantic failures. Group-Consistent Distillation (GCD) aligns every verified form with interchangeable valid realizations, while Form-Contrastive Flow-DPO (FC-DPO) learns group-level preferences from audited positive–negative pairs, prioritizing same-seed, diferent-form contrasts. On HunyuanVideo-1.5, VARI attains the highest T2V-CompBench, VBench, and StoryEval scores among the evaluated methods. On prompt-form extensions of T2V-CompBench and VBench, it improves weakest-form alignment over the strongest prompt-level baselines by 2.45 and 3.55 points and reduces cross-form drop by 30.5% and 32.2%, respectively, while preserving visual quality. Human evaluation, natural-caption controls, prompt-enhancement experiments, diversity analyses, and transfer to LTX-2.3 further support the robustness and transferability of these gains, while VARI requires no additional inference-time module.

## Introduction

Large text-to-video generators based on difusion or flow matching now achieve strong visual fidelity, motion quality, and instruction following (Ho, Jain, and Abbeel 2020; Lipman et al. 2023; Kong et al. 2024). Yet robust semantic realization remains challenging: meaning-preserving changes in prompt wording can alter entity counts, spatial relations, action bindings, event realization, or overall generation quality. Such variation is distinct from desirable visual diversity, since appearance, composition, and motion details may change while the core visual intent should remain invariant.

For example, consider “A boy reading in front of a bench” and “A bench is positioned behind a boy who is reading.” They express the same spatial relation, yet a generator may realize it under one wording and violate it under the other.

Inference-time prompt rewriting can improve a selected input, but it does not train the generator itself to behave reliably across equivalent formulations. Reward- and preferencebased post-training instead samples multiple rollouts for a prompt, scores them with automatic rewards, and learns from preferred and rejected examples (Liu et al. 2025b; Rafailov et al. 2023; Zhang et al. 2026; Liu et al. 2025a; Xue et al. 2025). However, supervision constructed from one formulation at a time cannot distinguish intent-level robustness from wording-specific success. Semantic-preserving prompt forms provide complementary linguistic evidence for constructing more reliable supervision.

We introduce VARI (Verified Alignment over Representations of Intent), a group-structured intentalignment framework that treats automatically verified prompt forms as a shared supervision unit. For each visual intent z, we construct $\mathcal { P } ( z ) = \{ p _ { 0 } , p _ { 1 } , . . . , p _ { K } \}$ , generate a matched form–seed rollout pool, and cross-score every candidate against all forms. A generation seed determines the initial sampling noise; reusing it across forms controls stochastic variation, so same-seed diferences more directly expose wording sensitivity. Because equivalent forms share one structured semantic target, score variation for a fixed video is treated as scoring-view dispersion, rather than formdependent ground truth. A frozen, wording-independent structured intent auditor checks atomic entities, counts, relations, action roles, and temporal commitments without observing prompt wording. The audit determines semantic type, while cross-form evidence ranks candidates within type, yielding reliable positives and visually plausible hard negatives.

As illustrated in Figure 1, Group-Consistent Distillation (GCD) trains every verified form on interchangeable valid realizations, whereas Form-Contrastive Flow-DPO (FC-DPO) optimizes audited success–failure preferences under every form, prioritizing same-seed, diferent-form contrasts. The resulting generator requires no additional inference-time module. On HunyuanVideo-1.5 (Wu et al. 2025a), VARI improves T2V-CompBench (Sun et al. 2025) by 1.64 points over the strongest prompt-level baseline and achieves the highest VBench (Huang et al. 2024) quality and StoryEval (Wang et al. 2025) scores among the evaluated methods. Prompt-form extensions of T2V-CompBench and VBench show weakest-form gains of 2.45 and 3.55 points, respectively. Human evaluation and held-out controls show that the gains persist under auditor replacement, natural captions, prompt enhancement, and transfer to LTX-2.3 while preserving seed diversity and sensitivity to non-equivalent prompt changes.

![](images/df4ec89e366b8e4f2906afb34be1b27a2f9142044f6520e1298bdefb31636463.jpg)

B. Two-Stage VARI Training  
![](images/5cb3a4bfff90b2ac8c3d11d4fa584280bd14be89eff93e9c316666c3f393f7f0.jpg)  
Figure 1: Overview of VARI. (A) Verified forms and shared seeds define a rollout grid. Cross-scoring provides mean support µ, weakest-view support w, dispersion $d ,$ and quality $Q ;$ a structured audit against $T _ { z }$ assigns semantic type, and GroupScore ranks candidates within type. (B) GCD distills retained positives across all forms, while FC-DPO lifts audited intent-level preferences to the full group, prioritizing same-seed, diferent-form pairs.

Our main contributions are threefold:

• Verified prompt-form groups as supervision units. We define prompt-form reliability over verified formulations of the same visually testable intent and use these groups, paired with shared seeds, as the supervision unit for T2V post-training.

• Audited cross-form supervision. We develop a compiler that cross-scores every rollout against all verified forms and uses a wording-independent structured intent audit to separate semantic validity from hardness ranking, yielding group-consistent positives and audited hard negatives.

• Complementary group-level objectives. We introduce GCD, which trains every verified form on interchangeable valid realizations, and FC-DPO, which optimizes audited success–failure preferences under every form while prioritizing same-seed, diferent-form comparisons.

## Related Work

Difusion, flow, and video generators. Modern visual generators build on denoising difusion, latent difusion, flow matching, rectified flows, difusion transformers, and classifier-free guidance (Ho, Jain, and Abbeel 2020; Rombach et al. 2022; Lipman et al. 2023; Liu, Gong, and Liu 2023; Peebles and Xie 2022; Ho and Salimans 2022). Video generation has progressed from early pixel- and latent-space difusion systems, cascaded spatial–temporal upsampling, and motion adaptation (Ho et al. 2022a,b; Singer et al. 2022; Blattmann et al. 2023a; Wang et al. 2023; Guo et al. 2023; Zhang et al. 2023; Kondratyuk et al. 2023; Bar-Tal et al. 2024) to large transformer- and flow-based backbones with stronger video latent spaces, including HunyuanVideo-1.5, LTX-2.3, and Wan2.2 (Blattmann et al. 2023b; Chen et al. 2024; Ma et al. 2024; Yang et al. 2024; Zheng et al. 2024; Polyak et al. 2024; Kong et al. 2024; Wu et al. 2025a; HaCohen et al. 2025; Lightricks 2026; Wan Team 2025a,b). VARI operates after pretraining and reorganizes supervision for an existing video DiT rather than modifying its backbone.

Preference and reward-based alignment. Generativemodel alignment includes direct preference objectives, policy-gradient fine-tuning, reward backpropagation, and reward-model-free denoising preference learning (Rafailov et al. 2023; Black et al. 2023; Wallace et al. 2024; Yang et al. 2023). Image-generation studies have developed preference datasets, reward models, and input-side prompt optimization (Kirstain et al. 2023; Xu et al. 2023; Wu et al. 2023b; Hao et al. 2023). For video, VADER, VideoDPO, VideoReward, OnlineVPO, DenseDPO, DanceGRPO, and VisionReward improve reward learning, preference construction, temporal granularity, or on-policy optimization (Prabhudesai et al. 2024; Liu et al. 2025b,a; Zhang et al. 2026; Wu et al. 2025b; Xue et al. 2025; Xu et al. 2026). Recent LocalDPO, Difusion-APO, and Dual-IPO further study localized supervision, trajectory-aware updates, and iterative reward–generator alignment (Huang et al. 2026; Zhu et al. 2026; Yang et al. 2026). VARI instead structures supervision across verified linguistic realizations of one visual intent. Main comparisons use VideoDPO and OnlineVPO because both admit matched HunyuanVideo-1.5 data and generation budgets; the newer methods change the underlying data or optimization pipeline.

## The VARI Framework

## Semantic-Preserving Prompt-Form Groups

Let z denote a visual intent and let $p _ { 0 }$ be its original prompt. A verified prompt-form group

$$
\mathcal { P } ( z ) = \{ p _ { 0 } , p _ { 1 } , . . . , p _ { K } \}\tag{1}
$$

contains variants preserving the same structured target $T _ { z } ,$ a set of atomic, visually testable commitments over objects, actions, attributes, relations, roles, and temporal order; unconstrained appearance may vary. Gemini 3.5 Flash generates ten training rewrites, Qwen3.6-27B verifies critical fields, and filtering retains four forms spanning lexical, syntactic, and caption-style variation; the roles are exchanged for benchmark groups. Human audit validates 190/200 training and 98/100 benchmark pairs, and 97/100 complete benchmark groups preserve $T _ { z }$ . Removing the three flagged groups changes headline robustness metrics by at most 0.11 points. Five forms and four shared seeds are adopted from held-out scale diagnostics, which show diminishing gains beyond five forms; details are in the supplement.

Each intent uses a seed set $S _ { z }$ shared across all verified forms, so form comparisons reuse the same initial sampling noises. The complete rollout pool is

$$
\mathcal { X } ( z ) = \{ G _ { \theta _ { \mathrm { r o l l } } } ( p _ { i } , s ) : p _ { i } \in \mathcal { P } ( z ) , s \in \mathcal { S } _ { z } \} .\tag{2}
$$

Here $G _ { \theta _ { \mathrm { r o l l } } }$ is the frozen rollout checkpoint used to construct supervision, and s indexes the initial sampling noise. Mining uses the full within-intent form–seed pool; candidates are never paired across intents.

## Cross-Form Score Geometry

Every candidate is evaluated against every verified form, yielding a score matrix $\mathbf { A } _ { z } \in \mathbb { R } ^ { | \mathcal { X } ( z ) | \times ( K + \bar { 1 } ) }$ with

$$
[ \mathbf { A } _ { z } ] _ { x , i } = A _ { i } ( x ) = \rho _ { A } R _ { \mathrm { V Q A } } ( x , p _ { i } ) + ( 1 - \rho _ { A } ) R _ { \mathrm { V S } } ( x , p _ { i } ) ,\tag{3}
$$

where VQAScore and VideoScore provide promptconditioned alignment signals (Lin et al. 2024; He et al. 2024). A separate low-weight quality term is

$$
Q ( x ) = \rho _ { Q } R _ { \mathrm { D O V E R } } ( x ) + ( 1 - \rho _ { Q } ) R _ { \mathrm { m o t i o n } } ( x ) ,\tag{4}
$$

where $R _ { \mathrm { m o t i o n } }$ summarizes motion smoothness, dynamics, flicker, and subject consistency (Wu et al. 2023a; Huang et al. 2024). Define the cross-form profile

$$
\begin{array} { l l } { \displaystyle \mu ( x , z ) = \frac { 1 } { K + 1 } \sum _ { i = 0 } ^ { K } A _ { i } ( x ) , } & { \displaystyle w ( x , z ) = \operatorname* { m i n } _ { i } A _ { i } ( x ) , } \\ { \displaystyle b ( x , z ) = \operatorname* { m a x } _ { i } A _ { i } ( x ) , } & { \displaystyle d ( x , z ) = b ( x , z ) - w ( x , z ) , } \end{array}\tag{5}
$$

Thus $\mu$ measures average cross-form support, w the weakest scoring view, b the strongest, and d their range. We rank candidates with $S _ { \mathrm { g r p } } = \bar { \lambda } _ { \mu } \mu + \lambda _ { w } w - \lambda _ { d } d + \bar { \lambda } _ { q } Q _ { \ l } ,$ , using $\rho _ { A } ~ = ~ \rho _ { Q } ~ = ~ 0 . 6$ and validation-fixed $( \lambda _ { \mu } , \lambda _ { w } , \dot { \lambda _ { d } } , \lambda _ { q } ) =$ $( 0 . 5 0 , 0 . \dot { 2 } 5 , 0 . 1 5 , 0 . 1 0 )$ . Before the more expensive semantic audit, validation-fixed geometry–quality rules shortlist likely positives with high w, low $d ,$ and adequate $Q ,$ and likely hard negatives with high $b ,$ nontrivial $d ,$ and adequate $Q .$ Only the deduplicated union is audited; the exact thresholds are in the supplement. This prefilter does not assign semantic labels: it only removes candidates that cannot satisfy the final positive or negative criteria.

For semantic typing, frozen Gemini 3.5 Flash (gemini-3.5-flash) (Google DeepMind 2026) receives 24 temporally ordered frames and one atomic commitment $c _ { z , j } ,$ , without prompt strings, generating-form identity, or mining scores. A stateless score call returns probabilities for A (satisfied), B (violated), and C (uncertain); a separate diagnostic call records evidence only and never afects supervision. Write $T _ { z } ~ = ~ \{ c _ { z , j } \} _ { j = 1 } ^ { J _ { z } }$ , let $K _ { z } \subseteq \{ 1 , \dots , J _ { z } \}$ index the required visually testable commitments, and define $s _ { j } = P ( \mathbb { A } )$ and $v _ { j } = P ( { \mathtt { B } } )$ . We aggregate

$$
\begin{array} { l l } { { \displaystyle C _ { \mathrm { m i n } } ( x , z ) = \operatorname* { m i n } _ { j \in { \mathcal K } _ { z } } s _ { j } , } } & { { \displaystyle C _ { \mathrm { a v g } } ( x , z ) = \frac { 1 } { J _ { z } } \sum _ { j = 1 } ^ { J _ { z } } s _ { j } , } } \\ { { \displaystyle V _ { \mathrm { m a x } } ( x , z ) = \operatorname* { m a x } _ { j \in { \mathcal K } _ { z } } v _ { j } . } } & { { } } \end{array}\tag{6}
$$

A disjoint 600-video, three-annotator split fixes thresholds to achieve at least 90% human precision; uncertain candidates abstain. The endpoint, request schema, frame sampling, and thresholds are frozen before mining, and the auditor is excluded from evaluation, checkpoint selection, and Group-Score tuning; Qwen3.6-27B (Qwen Team 2026) is used only for held-out replacement. Exact request and safety settings are in the supplement. Since all forms share $T _ { z }$ , d is scoringview dispersion, not form-dependent truth.

## Group-Structured Supervision Synthesis

The shortlist supplies the necessary geometry and quality conditions, while the audit determines semantic type. With all components normalized to [0, 1], a stable positive must pass the positive shortlist and satisfy $C _ { \mathrm { m i n } } \geq . 7 0$ and $C _ { \mathrm { a v g } } \geq . 8 0 ;$ a hard negative must pass the negative shortlist and satisfy $V _ { \mathrm { m a x } } \geq . 7 \bar { 5 }$ . Uncertain candidates are excluded. We denote the resulting eligible sets by $\mathcal { C } _ { z } ^ { + }$ and $\mathcal { C } _ { z } ^ { \mathrm { F C - } }$ , and retain $\mathcal { X } _ { z } ^ { + }$ and $\mathcal { X } _ { z } ^ { \mathrm { F C - } }$ as their top $m _ { + }$ and $m _ { - }$ candidates by GroupScore. Thus the audit fixes semantic type and Group-Score only ranks utility within type; for negatives, this favors visually plausible audit-confirmed failures rather than the most degraded videos. Exact set definitions are in the supplement.

Each rollout retains provenance $x _ { i , s } = G _ { \theta _ { \mathrm { r o l l } } } ( p _ { i } , s )$ . For a retained pair $r ,$ define its GroupScore gap as $\begin{array} { r l } { g _ { z , r } } & { { } = } \end{array}$ $S _ { \mathrm { g r p } } ( x _ { z , r } ^ { + } , z ) - S _ { \mathrm { g r p } } ( x _ { z , r } ^ { - } , z )$ and its confidence weight as $\omega _ { z , r } = \mathrm { c l i p } ( g _ { z , r } , \omega _ { \mathrm { m i n } } , \omega _ { \mathrm { m a x } } )$ . The compiler prioritizes audited positive–negative pairs generated with the same seed under diferent forms and satisfying a validation-fixed margin, then fills any remaining per-intent budget with top-gap audited pairs from the same intent; their union is $\Pi _ { z }$ . Thus “form-contrastive” describes generating-form provenance, while correctness is always judged against the common target $T _ { z }$ . The compiler emits one shared-target record per retained positive and group-lifted preference records for pairs in $\Pi _ { z } ;$ exact set definitions and thresholds are in the supplement.

## Group-Consistent Distillation: Shared-Support Contraction

For a video flow model with conditional velocity predictor $v _ { \theta } .$ , let $\ell _ { \mathrm { F M } } ( x ^ { + } , p _ { i } , t ; \theta )$ denote the standard squared flowmatching error between $v _ { \theta } ( x _ { t } ^ { + } , t , p _ { i } )$ and target $u _ { t } ( x ^ { + } )$ . Here $x _ { t } ^ { + }$ is the corrupted or interpolated positive latent at sampled time $t ,$ and one record shares the latent and target across forms. Let $q _ { z } ^ { + }$ sample the retained set $\mathcal { X } _ { z } ^ { + }$ ; GCD optimizes

$$
\mathcal { L } _ { \mathrm { G C D } } = \mathbb { E } _ { z , x ^ { + } \sim q _ { z } ^ { + } , t } \left[ \frac { 1 } { K + 1 } \sum _ { i = 0 } ^ { K } \ell _ { \mathrm { F M } } ( x ^ { + } , p _ { i } , t ; \theta ) \right] .\tag{7}
$$

Diferent records sample diferent valid positives and noise, so GCD contracts around a valid support set rather than one canonical video. The supplement shows that the per-record objective decomposes into fitting the group-mean prediction plus an explicit pairwise disagreement penalty; diversity and counterfactual controls test that this contraction does not erase valid variation or prompt sensitivity.

## Form-Contrastive Flow-DPO: Group-Lifted Margin Separation

For a mined pair and sampled time $t ,$ let $\Delta _ { i } ( \theta , t )$ be the policy-versus-reference change in the rejected-minuspreferred flow-loss margin under form $p _ { i }$ , where the reference is the frozen post-GCD checkpoint. VARI lifts the intent-level relation $x _ { z , r } ^ { + } \succ z x _ { z , r } ^ { - }$ to every verified form and instantiates the preference update with a flow-DPO kernel:

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { F C - D P O } } = \mathbb { E } _ { z , r , t } \Bigg [ \frac { \omega _ { z , r } } { K + 1 } \sum _ { i = 0 } ^ { K } - \log \sigma ( \beta _ { \mathrm { D P O } } \Delta _ { i } ( \theta , t ) ) \Bigg ] } } \\ & { + \lambda _ { \mathrm { a u x } } \mathbb { E } _ { z , r , t } \Bigg [ \frac { 1 } { K + 1 } \sum _ { i = 0 } ^ { K } \ell _ { \mathrm { F M } } ( x _ { z , r } ^ { + } , p _ { i } , t ; \theta ) \Bigg ] . } \end{array}\tag{8}
$$

We set $\beta _ { \mathrm { D P O } } = 0 . 3$ and $\lambda _ { \mathrm { a u x } } = 0 . 0 5$ . Within each branch, the policy and reference evaluate the same sampled time and corrupted input. The first term separates positives from audited hard negatives under every form; because the nonlinear preference loss is averaged after each form-wise margin is computed, an easy form cannot cancel an unresolved form before the logistic penalty, and smaller margins receive stronger updates. The auxiliary term preserves positive support, while matched seeds reduce stochastic confounding.

Algorithm 1 Audited Prompt-Form Supervision Compiler   
Require: Intent z, target $T _ { z } ,$ , forms P(z), shared seeds $s _ { z } ,$ , budgets m $\mathrm { ~ . ~ . ~ } , m _ { - } ,$   
margin $\tau _ { \mathrm { p a i r } }$ , clip bounds ω<sub>min</sub>, ω<sub>max</sub>   
Ensure: GCD records $\mathcal { D } _ { \mathrm { G C D } } ^ { z }$ and FC-DPO records $\mathcal { D } _ { \mathrm { F C } } ^ { z }$   
1: for each form $p _ { i } \in \mathcal { P } ( z )$ and seed $s \in S _ { z }$ do   
2: Generate $x _ { i , s }$ and retain provenance (i, s)   
3: end for   
4: Cross-score the pool to obtain $\mathbf { A } _ { z } , Q ,$ and $( \mu , w , b , d , S _ { \mathrm { g r p } } )$   
5: Form geometry–quality shortlists $\mathcal { U } _ { z } ^ { + }$ and $\mathcal { U } _ { z } ^ { - }$ using validation-fixed conditions   
6: for each x ∈ $u _ { z } ^ { + } \cup u _ { z } ^ { - }$ after deduplication do   
7: Audit atomic commitments in $T _ { z }$ and assign positive, negative, or abstained   
type   
8: end for   
9: Rank eligible candidates within type; retain $\boldsymbol { \mathcal { X } } _ { z } ^ { + }$ and $\mathcal { X } _ { z } ^ { \mathrm { F C } }$   
10: $\mathcal { D } _ { \mathrm { G C D } } ^ { z }  \{ ( \mathcal { P } ( z ) , x ^ { + } ) : x ^ { + } \in \mathcal { X } _ { z } ^ { + } \} ; \bar { \mathcal { D } } _ { \mathrm { F C } } ^ { z }  \bar { \mathcal { O } }$   
11: for matched-seed pairs first, then top-gap audited fallback pairs do   
12: $g  S _ { \mathrm { g r p } } ( x ^ { + } , z ) - S _ { \mathrm { g r p } } ( x ^ { - } , z )$   
13: $\mathbf { i f } \ g \ge \tau _ { \mathrm { p a i r } }$ and the budget is not full then   
14: $\operatorname { A d d } \left( \mathcal { P } ( z ) , x ^ { + } , x ^ { - } , \exp ( g , \omega _ { \operatorname* { m i n } } , \omega _ { \operatorname* { m a x } } ) \right) \mathrm { t o } \mathcal { D } _ { \mathrm { F C } } ^ { z }$   
15: end if   
16: end for   
17: return $( \mathcal { D } _ { \mathrm { G C D } } ^ { z } , \mathcal { D } _ { \mathrm { F C } } ^ { z } )$

## Experiments

## Experimental Setup

Backbone and training pool. We use HunyuanVideo-1.5 (Wu et al. 2025a) with a unified 480p, 16:9, 121-frame, 24-fps, 50-step protocol. The training pool contains 5,000 independently constructed source intents balanced across attributes, relations, actions, interactions, numeracy, and multievent composition; each has four verified rewrites and four shared seeds, yielding 100,000 rollouts. All trainable methods use matched groups and generation budgets. Ofline methods share the same pool, whereas OnlineVPO performs on-policy refresh under the same total budget. Main comparisons are Base, VideoDPO, OnlineVPO, GCD, and full VARI, with additional controls in the ablation.

Evaluation principle. Mining scores are not reused for reporting. Benchmark results are recomputed with the oficial VBench, T2V-CompBench, and StoryEval evaluator implementations. VBench and T2V-CompBench use the same 480p, 16:9, 121-frame, 24-fps, 50-step generation configuration for every system; StoryEval follows its oficial 10s setting. These extensions introduce no learned evaluator: they reuse the oficial per-video scores unchanged and add only verified prompt-form grouping and intent-level aggregation, yielding VBench-R and T2V-CompBench-R.

Prompt-form robustness protocol. For each of the 1,400 T2V-CompBench prompts, we construct five verified forms. VBench-R uses matched oficial semantic dimensions; each rewritten form inherits the source category and structured target, and only the generation-side wording changes. T2V-CompBench-R reuses the oficial score for the inherited category, whereas VBench-R reuses the matched oficial semantic dimension and reports oficial VBench quality separately on the same clips. Mean, Worst, and max–min Drop are aggregated within each intent. FIFR counts mixed-success groups in which at least one form passes and another fails a validation-fixed, category- or dimension-specific threshold; it is interpreted jointly with Mean and Worst rather than as standalone generation quality.

<table><tr><td rowspan="2">Method</td><td colspan="4">VBench VBench VBench T2V-C</td><td rowspan="2">StoryEval</td></tr><tr><td>Total</td><td>Qual.</td><td>Sem.</td><td>Avg.</td></tr><tr><td>Base</td><td>80.11</td><td>82.83</td><td>69.23</td><td>47.69</td><td>32.11</td></tr><tr><td>VideoDPO</td><td>81.06</td><td>83.68</td><td>70.59</td><td>49.60</td><td>34.62</td></tr><tr><td>OnlineVPO</td><td>81.60</td><td>84.19</td><td>71.24</td><td>50.49</td><td>35.31</td></tr><tr><td>GCD</td><td>81.15</td><td>83.46</td><td>71.91</td><td>50.48</td><td>35.73</td></tr><tr><td>VARI</td><td>82.67</td><td>84.91</td><td>73.71</td><td>52.13</td><td>37.58</td></tr></table>

Table 1: Standard benchmark results. T2V-C averages seven oficial categories; StoryEval reports strict event completion.

Data independence and natural-caption controls. Source intents are split before rewriting, so every form of an intent remains in one partition. VBench, T2V-CompBench, StoryEval, VATEX, and MSR-VTT prompts are excluded from training, calibration, and checkpoint selection. Normalized exact matching, lexical retrieval, and frozen sentenceembedding retrieval followed by manual review find no shared visual intent; training and benchmark groups also use disjoint rewrite instructions and verifier instances. The supplement reports the full overlap audit and held-out VATEX (Wang et al. 2019) and MSR-VTT (Xu et al. 2016) controls scored by VisionReward (Xu et al. 2026). VATEX/MSR-VTT weakest-form scores improve from 61.92/59.24 for OnlineVPO to 65.09/62.34 for VARI.

Audit, coverage, and statistics. The protocol is frozen before mining, and only the union of the validation-fixed positive and negative shortlists is audited. Across 5,000 intents, 87.4% yield an audited positive, 64.6% an audited negative, and 61.4% a preference pair; 71.7% of retained pairs are same-seed, diferent-form. VideoDPO, OnlineVPO, GCD, and VARI use three matched training seeds; full deviations and source-intent paired tests are reported in the supplement.

Compute eficiency. All trainable methods share the 100,000-video rollout budget. VARI adds one-time, parallelizable ofline cross-scoring, shortlist auditing, and a second optimization stage, giving modest overhead over VideoDPO and the same cost regime as OnlineVPO, while leaving per-video inference cost unchanged; details are in the supplement.

## Standard Benchmark Results

Table 1 gives the headline benchmark view. VBench provides quality and semantic anchors, T2V-CompBench measures compositional prompt following, and StoryEval evaluates strict event completion (Wang et al. 2025). VARI gives the strongest overall scores. GCD is comparatively stronger on semantic consistency, while the full method also improves quality and StoryEval, indicating complementary stages without trading away fidelity or event completion.

## Dimension-Wise Diagnostics

Table 2 expands T2V-CompBench into all seven categories. OnlineVPO remains strongest on dynamic attributes and motion binding, whereas VARI gains most on constant attributes, spatial relations, action binding, interaction, and numeracy. This pattern matches the entity, relation, role, and count commitments in the structured target rather than a uniform improvement in low-level motion generation.

## Prompt-Form Robustness Extensions

Table 3 is the central prompt-form robustness evaluation. Unlike the mining stage, the scores in this table are oficial benchmark scores aggregated over verified prompt-form groups. Let $s _ { i } ^ { ( b ) } ( z )$ denote the oficial score of benchmark b for the video generated from the i-th prompt form of intent z. We report

$$
\begin{array} { l } { { \mathrm { M e a n } ( z ) = \displaystyle \frac { 1 } { | \mathcal { P } ( z ) | } \sum _ { i } s _ { i } ^ { ( b ) } ( z ) , } } \\ { { \mathrm { W o r s t } ( z ) = \displaystyle \operatorname* { m i n } _ { i } s _ { i } ^ { ( b ) } ( z ) , } } \\ { { \mathrm { D r o p } ( z ) = \displaystyle \operatorname* { m a x } _ { i } s _ { i } ^ { ( b ) } ( z ) - \operatorname* { m i n } _ { i } s _ { i } ^ { ( b ) } ( z ) . } } \end{array}\tag{9}
$$

Form-Induced Failure Rate (FIFR) measures mixed-success groups:

$$
\begin{array} { l } { { f _ { z } ^ { ( b ) } = \displaystyle \mathbf { 1 } \Big [ \operatorname* { m i n } _ { i } s _ { i } ^ { ( b ) } ( z ) < \tau _ { b } \mathrm { \mathrm { \mathrm { ~ \land ~ } } } \operatorname* { m a x } _ { i } s _ { i } ^ { ( b ) } ( z ) \geq \tau _ { b } \Big ] \ : , } } \\ { { \displaystyle \mathrm { F I F R } = \frac { 1 0 0 } { N } \sum _ { z = 1 } ^ { N } f _ { z } ^ { ( b ) } . } } \end{array}\tag{10}
$$

Here $\tau _ { b }$ is fixed on held-out validation data for each inherited category or matched semantic dimension. FIFR is interpreted jointly with Mean and Worst because a uniformly weak model can also have low mixed-success inconsistency; VBench quality is reported separately as a preservation guardrail.

Against the strongest prompt-level baseline for each endpoint, VARI raises T2V-CompBench-R/VBench-R Worst by 2.45/3.55 points, reduces Drop by 1.21/2.77, and reduces FIFR by 3.78/2.99. Relative to GCD, it further improves T2V-R Mean/Worst by 0.86/0.72 and lowers FIFR by 1.07, while T2V-R Drop increases modestly from 2.52 to 2.76. Together, the results suggest that GCD supplies the main cross-form contraction, whereas FC-DPO raises the lower tail and removes additional mixed-success failures, with only a small trade-of in the average within-group range. Intent-level VBench-R gains are significant (Worst 95% CI [2.66, 4.43], Holm-adjusted $p < 0 . 0 0 1$ ; FIFR McNemar $p < 0 . 0 1$ after correction).

![](images/127e108f68c009889e654441f870067fe6a380521cc1a94b95ba11af3e3499a7.jpg)  
Figure 2: Representative mining-view score matrices for one intent group. Rows are generated candidates and columns are verified scoring forms. Cell variation is diagnostic evidence, not form-dependent ground truth for a fixed video.

<table><tr><td>Method</td><td>Cons. Attr. Dyn. Attr. Spatial Rel. Motion Bind. Action Bind. Interaction Numeracy</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td></tr><tr><td>Base</td><td>79.82</td><td>7.47</td><td>53.11</td><td>40.29</td><td>69.31</td><td>60.57</td><td>23.24</td><td>47.69</td></tr><tr><td>VideoDPO</td><td>81.34</td><td>8.25</td><td>55.56</td><td>41.89</td><td>71.24</td><td>63.49</td><td>25.42</td><td>49.60</td></tr><tr><td>OnlineVPO</td><td>82.31</td><td>10.04</td><td>56.72</td><td>43.91</td><td>72.13</td><td>64.49</td><td>23.81</td><td>50.49</td></tr><tr><td>GCD</td><td>83.26</td><td>8.66</td><td>56.97</td><td>39.88</td><td>72.71</td><td>66.61</td><td>25.29</td><td>50.48</td></tr><tr><td>VARI</td><td>86.64</td><td>9.58</td><td>58.55</td><td>40.42</td><td>75.06</td><td>67.78</td><td>26.87</td><td>52.13</td></tr></table>

Table 2: T2V-CompBench categories. VARI gives the best average; OnlineVPO leads dynamic attribute and motion binding.
<table><tr><td rowspan="2">Method</td><td colspan="4">T2V-CompBench-R</td><td colspan="4">VBench-R Semantic</td><td rowspan="2">VBench-R Quality</td></tr><tr><td>Mean↑</td><td>Worst↑ Drop↓</td><td></td><td>FIFR↓</td><td>Mean↑</td><td>Worst↑</td><td>Drop↓</td><td>FIFR↓ Mean↑</td></tr><tr><td>Base</td><td>47.18</td><td>44.47</td><td>4.92</td><td>11.14</td><td>68.29</td><td>61.27</td><td>11.74</td><td>11.38</td><td>82.78</td></tr><tr><td>VideoDPO</td><td>49.54</td><td>47.08</td><td>3.97</td><td>8.93</td><td>70.16</td><td>65.18</td><td>9.19</td><td>8.72</td><td>83.51</td></tr><tr><td>OnlineVPO</td><td>50.09</td><td>46.95</td><td>4.12</td><td>8.21</td><td>70.79</td><td>65.07</td><td>8.61</td><td>8.26</td><td>83.94</td></tr><tr><td>GCD</td><td>50.39</td><td>48.81</td><td>2.52</td><td>5.50</td><td>71.36</td><td>67.12</td><td>6.51</td><td>6.08</td><td>83.47</td></tr><tr><td>VARI</td><td>51.25</td><td>49.53</td><td>2.76</td><td>4.43</td><td>72.68</td><td>68.73</td><td>5.84</td><td>5.27</td><td>84.24</td></tr></table>

Table 3: Prompt-form group extensions of the oficial evaluators; VBench quality is reported separately.

Figure 2 shows a held-out cross-score matrix: OnlineVPO raises many cells but remains uneven, whereas VARI is higher and more stable. These scores provide mining evidence only; the structured audit assigns semantic type against the shared target.

## Distribution Analysis

Figure 3 summarizes all VBench-R groups. VARI shifts worst-form scores upward and tightens Drop; simultaneous Mean and Worst gains suggest that the improvement is not explained solely by a few strong forms or uniformly weak contraction.

## Compatibility with Prompt Enhancement

PE rewrites the input, whereas VARI changes the generator. We freeze the oficial HunyuanVideo-1.5 template (Tencent

![](images/b0869f20bc37a98feb6d7b39a8f1c2449ead3fe69546dad2ec893c177e51c93d.jpg)

![](images/9ba25f7f762e1c7638d0e2a837022606dcfba87b76f503d070f3459d1b427532.jpg)  
Figure 3: Group-level VBench-R semantic distributions. VARI raises worst-form scores and reduces cross-form drops.

Hunyuan $2 0 2 5 \mathrm { a } , \mathrm { b } )$ for both methods and evaluate against the original forms. Under identical enhanced prompts, VARI improves T2V-C/R-Worst from 57.63/53.09 to 59.91/56.28 and reduces R-Drop/FIFR from 6.91/7.64 to 4.57/4.14, isolating improved generator behavior from rewrite strength.

## Ablation Studies

Table 4 controls paraphrase exposure, worst-form optimization, target sharing, audit, hardness, and pair provenance. Exposure alone is insuficient; GCD provides the main contraction, while prompt-DPO after GCD raises Mean but degrades the lower tail. Audited group preferences improve Worst, Drop, and FIFR jointly, with matched-seed provenance adding the final gain.

<table><tr><td>Variant</td><td>Mean↑</td><td>Worst↑</td><td>Drop↓</td><td>FIFR↓</td></tr><tr><td>Reward-SFT</td><td>48.58</td><td>43.82</td><td>5.97</td><td>13.11</td></tr><tr><td>Multi-form prompt-DPO</td><td>50.08</td><td>47.03</td><td>4.34</td><td>8.61</td></tr><tr><td>GCD + prompt-DPO</td><td>50.82</td><td>48.64</td><td>3.28</td><td>6.19</td></tr><tr><td>Direct worst-form DPO</td><td>50.39</td><td>48.17</td><td>3.12</td><td>6.38</td></tr><tr><td>Form-specific distillation</td><td>49.93</td><td>47.51</td><td>3.69</td><td>7.19</td></tr><tr><td>Dispersion-only FC-DPO</td><td>50.69</td><td>48.35</td><td>3.58</td><td>6.91</td></tr><tr><td>GCD†</td><td>50.46</td><td>48.88</td><td>2.57</td><td>5.41</td></tr><tr><td>Intent-audit-only  $\mathrm { F C - D P O ^ { \dag } }$ </td><td>50.94</td><td>49.09</td><td>2.61</td><td>5.02</td></tr><tr><td>VARI w/o matched-seed  ${ \mathrm { p a i r s } } ^ { \dagger }$ </td><td>51.06</td><td>49.28</td><td>2.46</td><td>4.87</td></tr><tr><td>VARI (matched-seed prioritized)†</td><td>51.17</td><td>49.51</td><td>2.29</td><td>4.44</td></tr></table>

Table 4: Mechanism ablation on the fixed development split; † denotes three paired runs.

## Supervision Validity and Boundary Controls

The blind audit in Table 5 separates semantic validity from dispersion: audited FC negatives reach 91/100 defect precision versus 58/100 for dispersion alone, showing that variation alone is not a reliable failure label. Auditor scores correlate with humans at $\rho ~ = ~ 0 . 6 3 ;$ Qwen3.6-27B gives 0.73 retained-set Jaccard and 90/100 precision. VARI retains 96.5%/95.8% appearance/motion diversity and raises “A bench is positioned behind a boy who is reading.”

“A boy reading in front of a bench.”  
![](images/4ec2ae252d747864b69106ab99d11bc9bc06ff50acda746e8e9367d8dec16638.jpg)

“A man is reading a newspaper on a park bench, while a squirrel is climbing a tree.”  
![](images/caa12d80ceada22299af0a6033e5e2b9a4aeb05ce7a9f1e2e00cb0bfb783e3d7.jpg)

![](images/60c77d00b1237f14b58b9cd843643fafc1cc5f397c95ca4072937e257b4c91ec.jpg)

“While a squirrel climbs a tree, a man reads a newspaper on a park bench.”  
![](images/dfdc95c7f99d62ed27106697911ae7a3e595e6366ece97a7f09fa799dd2f4b52.jpg)  
Figure 4: Source/equivalent-form cases. Baselines violate spatial relation or action binding; VARI preserves both across forms. The panels show temporally ordered frame strips from the evaluated video clips.

the counterfactual margin from 11.36 to 13.90, providing evidence against collapse.
<table><tr><td>Mined type</td><td>Human precision</td><td>Visual usability</td><td>Independent agreement</td></tr><tr><td>Group-consistent positive</td><td>96/100</td><td>91/100</td><td>88/100</td></tr><tr><td>Dispersion-only negative</td><td>58/100</td><td>81/100</td><td>50/100</td></tr><tr><td>Audited FC negative</td><td>91/100</td><td>79/100</td><td>84/100</td></tr></table>

Table 5: Blind supervision audit; cells report valid counts out of N = 100 candidates per type.

## Human Evaluation

We conduct a blinded pairwise comparison against OnlineVPO on 200 held-out intent groups. Three independent annotators compare anonymized video sets for intent faithfulness, cross-form consistency, visual quality, and overall preference; identities and side order are hidden, and ties are allowed. VARI obtains a 33.33-point overall win–loss margin (95% CI: 26.3–40.4; Krippendorf’s α = 0.60), with the largest advantage on consistency and a positive quality margin, matching the automatic Worst and Drop trends.

Cross-backbone transfer. On LTX-2.3 (Lightricks 2026), GCD raises T2V-C/R-Worst from 44.79/41.86 to 47.57/44.79 and lowers R-Drop from 5.48 to 3.51. Adding FC-DPO further reaches 48.29/45.76 and 3.01, while StoryEval rises to 33.37, reproducing the stage-wise trend on a second backbone.

## Conclusion

We present VARI, a group-structured intent-alignment framework for robust text-to-video post-training. VARI treats verified formulations of the same visual intent as a shared supervision unit, separates semantic validity from hardness ranking through a wording-independent audit, and trains the generator with GCD and FC-DPO. The two stages play complementary roles: GCD shares interchangeable valid realizations across forms, while FC-DPO sharpens the preference boundary against audited semantic failures. Across HunyuanVideo-1.5 and LTX-2.3, VARI improves standard alignment and weakest-form reliability, reduces cross-form drop and mixed-success failures, and preserves visual quality, seed diversity, and sensitivity to genuine semantic changes. These results show that robustness to meaning-preserving prompt variation can be learned directly at the generator level, complementing inference-time prompt rewriting without adding deployment-time modules.

## References

Bar-Tal, O.; et al. 2024. Lumiere: A Space-Time Diffusion Model for Video Generation. arXiv preprint arXiv:2401.12945.

Black, K.; Janner, M.; Du, Y.; Kostrikov, I.; and Levine, S. 2023. Training Difusion Models with Reinforcement Learning. arXiv preprint arXiv:2305.13301.

Blattmann, A.; Rombach, R.; Ling, H.; Dockhorn, T.; Kim, S. W.; Fidler, S.; and Kreis, K. 2023a. Align Your Latents: High-Resolution Video Synthesis with Latent Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22563–22575.

Blattmann, A.; et al. 2023b. Stable Video Difusion: Scaling Latent Video Difusion Models to Large Datasets. arXiv preprint arXiv:2311.15127.

Chen, H.; Zhang, Y.; Cun, X.; Xia, M.; Wang, X.; Weng, C.; and Shan, Y. 2024. VideoCrafter2: Overcoming Data Limitations for High-Quality Video Difusion Models. arXiv preprint arXiv:2401.09047.

Google DeepMind. 2026. Gemini 3.5 Flash. https://ai. google.dev/gemini-api/docs/models/gemini-3.5-flash. Stable API model identifier: gemini-3.5-flash; accessed July 28, 2026.

Guo, Y.; et al. 2023. AnimateDif: Animate Your Personalized Text-to-Image Difusion Models without Specific Tuning. arXiv preprint arXiv:2307.04725.

HaCohen, Y.; Chiprut, N.; Brazowski, B.; Shalem, D.; Moshe, D.; Richardson, E.; Levin, E.; Shiran, G.; Zabari, N.; Gordon, O.; et al. 2025. LTX-Video: Realtime Video Latent Difusion. arXiv preprint arXiv:2501.00103.

Hao, Y.; Chi, Z.; Dong, L.; and Wei, F. 2023. Optimizing Prompts for Text-to-Image Generation. In Advances in Neural Information Processing Systems.

He, X.; et al. 2024. VideoScore: Building Automatic Metrics to Simulate Fine-Grained Human Feedback for Video Generation. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising Difusion Probabilistic Models. In Advances in Neural Information Processing Systems.

Ho, J.; and Salimans, T. 2022. Classifier-Free Difusion Guidance. arXiv preprint arXiv:2207.12598.

Ho, J.; Salimans, T.; Gritsenko, A.; Chan, W.; Norouzi, M.; and Fleet, D. J. 2022a. Video Difusion Models. arXiv preprint arXiv:2204.03458.

Ho, J.; et al. 2022b. Imagen Video: High Definition Video Generation with Difusion Models. arXiv preprint arXiv:2210.02303.

Huang, Z.; Zhang, K.; Ding, Y.; Gao, C.; Ding, R.; Chen, Y.; and Zuo, W. 2026. Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Difusion Models. arXiv preprint arXiv:2601.04068.

Huang, Z.; et al. 2024. VBench: Comprehensive Benchmark Suite for Video Generative Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Kirstain, Y.; Polyak, A.; Singer, U.; Matiana, S.; Penna, J.; and Levy, O. 2023. Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation. arXiv preprint

arXiv:2305.01569.

Kondratyuk, D.; et al. 2023. VideoPoet: A Large Language Model for Zero-Shot Video Generation. arXiv preprint arXiv:2312.14125.

Kong, W.; et al. 2024. HunyuanVideo: A Systematic Framework For Large Video Generative Models. arXiv preprint arXiv:2412.03603.

Lightricks. 2026. LTX-2.3: Oficial Inference and Training Repository. https://github.com/Lightricks/LTX-2. Checkpoint ltx-2.3-22b-dev.safetensors; accessed July 28, 2026.

Lin, Z.; Pathak, D.; Li, B.; Li, J.; Xia, X.; Neubig, G.; Zhang, P.; and Ramanan, D. 2024. Evaluating Text-to-Visual Generation with Image-to-Text Generation. arXiv preprint arXiv:2404.01291.

Lipman, Y.; Chen, R. T. Q.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2023. Flow Matching for Generative Modeling. In International Conference on Learning Representations.

Liu, J.; Liu, G.; Liang, J.; Yuan, Z.; Liu, X.; Zheng, M.; Wu, X.; Wang, Q.; Xia, M.; Wang, X.; Liu, X.; Yang, F.; Wan, P.; Zhang, D.; Gai, K.; Yang, Y.; and Ouyang, W. 2025a. Improving Video Generation with Human Feedback. In Advances in Neural Information Processing Systems, volume 38.

Liu, R.; Wu, H.; Zheng, Z.; Wei, C.; He, Y.; Pi, R.; and Chen, Q. 2025b. VideoDPO: Omni-Preference Alignment for Video Difusion Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Liu, X.; Gong, C.; and Liu, Q. 2023. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In International Conference on Learning Representations.

Ma, X.; Wang, Y.; Jia, G.; Chen, X.; Liu, Z.; Li, Y.-F.; Chen, C.; and Qiao, Y. 2024. Latte: Latent Difusion Transformer for Video Generation. arXiv preprint arXiv:2401.03048.

Peebles, W.; and Xie, S. 2022. Scalable Difusion Models with Transformers. arXiv preprint arXiv:2212.09748.

Polyak, A.; et al. 2024. Movie Gen: A Cast of Media Foundation Models. arXiv preprint arXiv:2410.13720.

Prabhudesai, M.; Mendonca, R.; Qin, Z.; Fragkiadaki, K.; and Pathak, D. 2024. Video Difusion Alignment via Reward Gradients. arXiv preprint arXiv:2407.08737.

Qwen Team. 2026. Qwen3.6-27B Model Card. https: //huggingface.co/Qwen/Qwen3.6-27B. Unified multimodal checkpoint; accessed July 28, 2026.

Radford, A.; et al. 2021. Learning Transferable Visual Models From Natural Language Supervision. In International Conference on Machine Learning.

Rafailov, R.; Sharma, A.; Mitchell, E.; Ermon, S.; Manning, C. D.; and Finn, C. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In Advances in Neural Information Processing Systems.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-Resolution Image Synthesis with Latent Difusion Models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Singer, U.; et al. 2022. Make-A-Video: Text-to-Video Generation without Text-Video Data. arXiv preprint arXiv:2209.14792.

Sun, K.; Huang, K.; Liu, X.; Wu, Y.; Xu, Z.; Li, Z.; and Liu, X.

2025. T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-Video Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Tencent Hunyuan. 2025a. HunyuanVideo-1.5: Oficial Code and Prompt-Rewrite Utilities. https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5. Accessed July 28, 2026.

Tencent Hunyuan. 2025b. HunyuanVideo 1.5 Prompt Handbook. https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/assets/HunyuanVideo\_1\_5\_Prompt\_

Handbook\_EN.md. Accessed July 28, 2026.

Wallace, B.; Dang, M.; Rafailov, R.; Zhou, L.; Lou, A.; Purushwalkam, S.; Ermon, S.; Xiong, C.; Joty, S.; and Naik, N. 2024. Difusion Model Alignment Using Direct Preference Optimization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Wan Team. 2025a. Wan: Open and Advanced Large-Scale Video Generative Models. arXivpreprint arXiv:2503.20314.

Wan Team. 2025b. Wan2.2: Oficial Code and Models. https: //github.com/Wan-Video/Wan2.2. Accessed July 28, 2026.

Wang, J.; et al. 2023. ModelScope Text-to-Video Technical Report. arXiv preprint arXiv:2308.06571.

Wang, X.; Wu, J.; Chen, J.; Li, L.; Wang, Y.-F.; and Wang, W. Y. 2019. VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Wang, Y.; He, X.; Wang, K.; Ma, L.; Yang, J.; Wang, S.; Du, S. S.; and Shen, Y. 2025. Is Your World Simulator a Good Story Presenter? A Consecutive Events-Based Benchmark for Future Long Video Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Wu, B.; et al. 2025a. HunyuanVideo 1.5 Technical Report. arXiv preprint arXiv:2511.18870.

Wu, H.; Zhang, E.; Liao, L.; Chen, C.; Hou, J.; Wang, A.; Sun, W.; Yan, Q.; and Lin, W. 2023a. Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 20144–20154.

Wu, X.; et al. 2023b. Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv preprint arXiv:2306.09341.

Wu, Z.; Kag, A.; Skorokhodov, I.; Menapace, W.; Mirzaei, A.; Gilitschenski, I.; Tulyakov, S.; and Siarohin, A. 2025b. DenseDPO: Fine-Grained Temporal Preference Optimization for Video Difusion Models. arXiv preprint arXiv:2506.03517.

Xu, H.; Ghosh, G.; Huang, P.-Y.; Okhonko, D.; Aghajanyan, A.; Metze, F.; Zettlemoyer, L.; and Feichtenhofer, C. 2021.

VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding. In Conference on Empirical Methods in Natural Language Processing.

Xu, J.; Huang, Y.; Cheng, J.; Yang, Y.; Xu, J.; Wang, Y.; Duan, W.; Yang, S.; Jin, Q.; Li, S.; Teng, J.; Yang, Z.; Zheng, W.; Liu, X.; Zhang, D.; Ding, M.; Zhang, X.; Huang, S.; Gu, X.; Huang, M.; Tang, J.; and Dong, Y. 2026. VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation. In Proceedings

of the AAAI Conference on Artificial Intelligence, volume 40, 11269–11277.

Xu, J.; Liu, X.; Wu, Y.; Tong, Y.; Li, Q.; Ding, M.; Tang, J.; and Dong, Y. 2023. ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation. In Advances in Neural Information Processing Systems.

Xu, J.; Mei, T.; Yao, T.; and Rui, Y. 2016. MSR-VTT: A Large Video Description Dataset for Bridging Video and Language. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition.

Xue, Z.; Wu, J.; Gao, Y.; Kong, F.; Zhu, L.; Chen, M.; Liu, Z.; Liu, W.; Guo, Q.; Huang, W.; and Luo, P. 2025. DanceGRPO: Unleashing GRPO on Visual Generation. arXiv preprint arXiv:2505.07818.

Yang, K.; Tao, J.; Lyu, J.; Ge, C.; Chen, J.; Li, Q.; Shen, W.; Zhu, X.; and Li, X. 2023. Using Human Feedback to Finetune Difusion Models without Any Reward Model. arXiv preprint arXiv:2311.13231.

Yang, X.; Yang, M.; Jia, G.; Qin, L.; Tan, Z.; and Li, H. 2026. Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation. In International Conference on Learning Representations.

Yang, Z.; et al. 2024. CogVideoX: Text-to-Video Difusion Models with an Expert Transformer. arXiv preprint arXiv:2408.06072.

Zhang, D. J.; Wu, J. Z.; Liu, J.-W.; Zhao, R.; Ran, L.; Gu, Y.; Gao, D.; and Shou, M. Z. 2023. Show-1: Marrying Pixel and Latent Difusion Models for Text-to-Video Generation. arXiv preprint arXiv:2309.15818.

Zhang, J.; Wu, J.; Chen, W.; Ji, Y.; Xiao, X.; Huang, W.; and Han, K. 2026. Align Video Difusion Model with Online Video-Centric Preference Optimization. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 6142–6152.

Zheng, Z.; Peng, X.; Yang, T.; Shen, C.; Li, S.; Liu, H.; Zhou, Y.; Li, T.; and You, Y. 2024. Open-Sora: Democratizing Eficient Video Production for All. arXiv preprint arXiv:2412.20404.

Zhu, J.; Chen, B.; Zhang, L.; Zhang, A.; Jiang, H.; and Huang, P. 2026. Difusion-APO: Trajectory-Aware Direct Preference Alignment for Video Difusion Transformers. arXiv preprint arXiv:2605.07503.

## Supplementary Material

This supplement covers prompt-form construction and verification, audited supervision synthesis, objective analysis, implementation and data-independence protocols, statistical and alternative-explanation controls, extended results and ablations, human evaluation, the main-case video manifest, and four additional qualitative comparisons.

## Prompt-Form Groups and Semantic Preservation

## Visual intents and admissible forms

A visual intent z is represented by the unordered form set $\mathcal { P } ( z ) = \{ p _ { 0 } , p _ { 1 } , . . . , \bar { p _ { K } } \}$ and a structured target $T _ { z }$ . The target records entities and counts, attributes and states, actions and participant roles, interactions, spatial relations, temporal order, and explicit scene or camera constraints. The forms may difer lexically and syntactically but must preserve every field in T<sub>z</sub>; the group is therefore an equivalence class of visually testable content, not loosely related captions.

Training source intents are instantiated from a deterministic, non-benchmark structured schema balanced across attributes, relations, actions, interactions, numeracy, and multi-event composition. Gemini 3.5 Flash (gemini-3.5-flash) generates ten candidates per source with a capped refill for malformed outputs. Qwen3.6- 27B independently verifies every candidate against the source-derived T<sub>z</sub>. For benchmark extensions, generator and verifier roles are exchanged and their prompts are disjoint from the training compiler. The endpoint IDs, access dates, and decoding settings are frozen in the released configuration.

## Semantic-preserving prompt-form rewriting template

Input and goal. Given one T2V prompt, produce independently usable, linguistically diverse rewrites of the same visual event.

Preserve. Every entity and count, attribute/state, action and participant role, interaction, spatial relation, temporal order, scene constraint, and explicit camera/style/lighting/duration/text requirement.

Reject. Any added, removed, weakened, strengthened, disambiguated, or reversed visual commitment.

Permitted variation. Lexical substitution, clause reordering, role-preserving voice conversion, caption-like phrasing, and concise or expanded syntax without visual change.

Output. Return the source, each rewrite, and its transformationfamily label in structured form.

The accepted transformation families are summarized in Table 6. They are used for audit and balance; the labels are not provided to the generator during training.

<table><tr><td>Family</td><td>Allowed surface variation</td></tr><tr><td>Lexical</td><td>Synonyms and local wording changes without semantic strengthening</td></tr><tr><td>Syntactic</td><td>Clause reordering, attachment changes, or role-preserving voice conversion</td></tr><tr><td>Caption-like</td><td>Natural descriptive-caption syntax with the same commitments</td></tr><tr><td>Concise</td><td>Removal of linguistic redundancy while retaining every visual constraint</td></tr><tr><td></td><td>Relation-focused Reordering that foregrounds an existing spatial or interaction relation</td></tr><tr><td>Action-focused</td><td>Reordering that foregrounds an existing action or event sequence</td></tr></table>

Table 6: Prompt-form transformation families. A candidate is rejected whenever its visual content changes, regardless of its nominal family.

## Structured equivalence verification

A rewrite is retained only when its relation to the source is unambiguous. Verification is field-wise rather than based on a single similarity score. This avoids accepting a fluent paraphrase that silently changes a count, swaps semantic roles, or modifies a spatial relation.

## Visual-intent equivalence verification template

Task. Compare an original text-to-video prompt with one candidate rewrite. Judge semantic preservation only; do not judge writing style or predicted video quality.

Check independently. Entities and exact counts; attributes and states; actions; agent–patient roles; interactions; spatial relations; temporal order; scene constraints; explicit camera/style/lighting/duration/text constraints; and added, removed, or altered commitments.

Decision rule. Reject if any critical field difers, if a new visual commitment is introduced, if a required commitment disappears, or if a previously explicit relation becomes ambiguous. Treat an uncertain critical field as rejection rather than guessing.

Output structure. Return a status for every field, lists of added/removed/changed commitments, a final accept/reject decision, and a concise reason.

## Illustrative audit example

For the source intent “A dog jumps over a low wooden fence in a grassy field,” a rewrite may move the setting to the beginning or use a role-preserving paraphrase such as “In a grassy field, a dog clears a short wooden fence.” A rewrite is rejected if it introduces a tracking camera, changes the fence height, adds a person, replaces over with through, or weakens the jump into generic movement. This example illustrates the field-wise verification policy.

Near-duplicate accepted forms are removed before group formation. Splits are created at the source-intent level before rewriting, so no forms of one intent can cross partitions. Benchmark-derived groups are excluded from training, threshold calibration, and checkpoint selection. Table 7 reports the compiler funnel. From 53,487 generated candidates, 2,713 malformed or near-duplicate outputs are removed and 34,318 pass field-wise verification; diversity-aware selection retains four rewrites for each of 5,000 complete training groups.

<table><tr><td>Compiler stage</td><td>Count</td><td>Rate (%)</td></tr><tr><td>Generated candidates</td><td>53,487</td><td>100.00</td></tr><tr><td>Malformed / near-duplicate removed</td><td>2,713</td><td>5.07</td></tr><tr><td>Field-wise verifier inputs</td><td>50,774</td><td>94.93</td></tr><tr><td>Verifier-accepted candidates</td><td>34,318</td><td>64.16</td></tr><tr><td>Retained rewrites (four per source)</td><td>20,000</td><td>37.39</td></tr><tr><td>Complete five-form groups</td><td>5,000</td><td>100.00 intents</td></tr></table>

Table 7: Prompt-form generation and verification funnel. Candidate-stage rates use 53,487 as the denominator; complete-group coverage is reported over 5,000 source intents.

We additionally conduct a stratified human audit of 300 accepted source–rewrite pairs, with three independent annotators and majority vote. The sample contains 200 training pairs and 100 benchmark pairs. Table 8 reports the fieldwise breakdown; Table 9 separates the two sources. A second audit samples 100 complete benchmark groups and asks whether all four rewrites preserve the common $\bar { T } _ { z }$ . Ninetyseven groups pass in full. Excluding the three flagged groups changes every headline Worst, Drop, and FIFR result by at most 0.11 points, indicating that the benchmark conclusions are not driven by the audited failures.

<table><tr><td>Critical field</td><td>Equivalent / total</td><td>Precision (%)</td></tr><tr><td>Entity / count</td><td>59/60</td><td>98.33</td></tr><tr><td>Spatial relation</td><td>57/60</td><td>95.00</td></tr><tr><td>Action-role binding</td><td>58/60</td><td>96.67</td></tr><tr><td>Temporal order</td><td>56/60</td><td>93.33</td></tr><tr><td>Attribute / state / other</td><td>58/60</td><td>96.67</td></tr><tr><td>Overall</td><td>288/300</td><td>96.00</td></tr></table>

Table 8: Human semantic-equivalence audit of accepted prompt forms.

<table><tr><td>Audit subset</td><td>Equivalent / total</td><td>Precision (%)</td></tr><tr><td>Training accepted pairs</td><td>190/200</td><td>95.00</td></tr><tr><td>Benchmark accepted pairs</td><td>98/100</td><td>98.00</td></tr><tr><td>All accepted pairs</td><td>288/300</td><td>96.00</td></tr><tr><td>Complete benchmark groups (all four rewrites valid)</td><td>97/100</td><td>97.00</td></tr></table>

Table 9: Source-specific pair audit and complete-group benchmark audit. Group success requires all four rewrites to preserve the shared target.

## Formal Objective Analysis

This section analyzes three properties ofthe VARI objectives: positive coverage under group-consistent mining, contraction induced by shared-target GCD, and the conservative formwise preference surrogate optimized by FC-DPO.

## Group-structured supervision compiler

Let $L = K + 1$ be the number of verified forms. The same intent-specific seed set $S _ { z }$ is reused across forms, and mining pools every candidate in $\mathcal { P } ( z ) \times  { S _ { z } }$ . For a candidate x, define

its cross-form scoring profile

$$
\begin{array} { c } { { \displaystyle { \bf s } ( x ; z ) = \big ( A ( x , p _ { 0 } ) , \ldots , A ( x , p _ { K } ) \big ) , } } \\ { { \displaystyle { \boldsymbol \mu } ( x ) = L ^ { - 1 } \sum _ { i } A _ { i } ( x ) , \qquad w ( x ) = \mathop { \operatorname* { m i n } } A _ { i } ( x ) , } } \\ { { \displaystyle b ( x ) = \mathop { \operatorname* { m a x } } A _ { i } ( x ) , \qquad d ( x ) = b ( x ) - w ( x ) . } } \end{array}\tag{11}
$$

## Surface-form-independent structured intent auditor

The primary auditor is Gemini 3.5 Flash (gemini-3.5-flash) (Google DeepMind 2026); Qwen/Qwen3.6-27B (Qwen Team 2026) is reserved for held-out replacement. Each 121-frame clip is represented by 24 uniformly sampled, temporally ordered RGB frames resized to 448×252. The structured target is deterministically serialized into atomic commitments $T _ { z } = \{ c _ { z , 1 } , \ldots , c _ { z , J _ { z } } \}$ with field tags for entity/count, attribute/state, spatial relation, action–participant binding, interaction, temporal order, and explicit scene or camera constraints. Prompt strings, generating-form identity, and mining scores are withheld.

Score call. Each shortlisted candidate–commitment pair is evaluated by a fresh, stateless REST request whose output is restricted by a string-enum schema to one label token: A (satisfied), B (violated), or C (uncertain). We pin thinkingLevel=minimal, candidateCount=1, maxOutputTokens=1, seed=202607, responseLogprobs=true, and logprobs=3. In accordance with the Gemini 3.5 model guidance, temperature, topP, and topK are omitted rather than overridden; the stable model ID, complete request payload, and access date are archived with the code. Safety thresholds are BLOCK\_ONLY\_HIGH for the four adjustable categories: harassment, hate speech, sexually explicit content, and dangerous content. We renormalize the returned first-step probabilities over the token IDs for $\mathbb { A } / \mathbb { B } / \mathbb { C }$ . Let $s _ { j } = P ( \mathbb { A } )$ and $v _ { j } = P ( { \mathtt { B } } )$ . Required visually testable fields form $\kappa _ { z } ;$ we compute

$$
\begin{array} { l l } { { C _ { \operatorname* { m i n } } = \displaystyle \operatorname* { m i n } _ { j \in { \mathcal K } _ { z } } s _ { j } , } } & { { C _ { \mathrm { a v g } } = J _ { z } ^ { - 1 } \sum _ { j } s _ { j } , } } \\ { {  } } & { {  } } \\ { { V _ { \operatorname* { m a x } } = \displaystyle \operatorname* { m a x } _ { j \in { \mathcal K } _ { z } } v _ { j } . } } \end{array}\tag{12}
$$

A positive requires $C _ { \mathrm { m i n } } \geq 0 . 7 0$ and $C _ { \mathrm { a v g } } \geq 0 . 8 0 $ ; a negative requires $V _ { \mathrm { m a x } } \ge 0 . 7 5$ . Other candidates abstain.

## Atomic score query

You are given 24 temporally ordered frames and one atomic visual commitment. Judge only visible evidence. Return exactly one enum token: A if satisfied, B if violated, or C if uncertain. Do not provide any additional text.

Diagnostic call. A separate request may return a JSON object containing the categorical label, supporting frame indices, and a brief rationale. Diagnostic outputs support manual inspection and error analysis; they do not enter $C _ { \mathrm { m i n } } .$ $C _ { \mathrm { a v g } } , V _ { \mathrm { m a x } } ,$ threshold calibration, supervision construction, or checkpoint selection.

The thresholds are selected on a disjoint 600-video, three-annotator calibration split stratified by semantic field, maximizing retained coverage subject to at least 90% human precision for accepted positives and accepted semantic failures. Endpoint, frame sampling, label protocol, generation parameters, safety settings, field serialization, and thresholds are then frozen. The primary auditor is excluded from GroupScore-weight tuning, benchmark evaluation, and checkpoint selection. Qwen3.6-27B receives the identical frames and atomic commitments and is used only in the replacement analysis.

The alignment scores are normalized before aggregation. Because all forms share $T _ { z } , d ( x )$ is scoring-view dispersion rather than a form-dependent semantic label. The implemented GroupScore is

$$
S _ { \mathrm { g r p } } ( x , z ) = \lambda _ { \mu } \mu ( x ) + \lambda _ { w } w ( x ) - \lambda _ { d } d ( x ) + \lambda _ { q } Q ( x ) .\tag{13}
$$

The non-audit parts of the positive and negative rules define

$$
\begin{array} { r l } & { \mathcal { U } _ { z } ^ { + } = \{ x : w \geq . 6 2 , d \leq . 1 8 , Q \geq . 5 5 \} , } \\ & { \mathcal { U } _ { z } ^ { - } = \{ x : b \geq . 5 8 , d \geq . 1 2 , Q \geq . 5 0 \} . } \end{array}\tag{14}
$$

The auditor is invoked only on the deduplicated union $\mathcal { U } _ { z } = \mathcal { U } _ { z } ^ { + } \cup \mathcal { U } _ { z } ^ { - }$ . This preserves the final candidate set: a rollout outside the corresponding shortlist cannot satisfy the complete positive or negative rule for any audit outcome.

VARI separates semantic typing from hardness ranking. With all components normalized to [0, 1], the implemented operators are

$$
\begin{array} { r l } & { \quad \Gamma ^ { + } ( \phi _ { z } ( x ) ) = \mathbf { 1 } [ x \in \mathcal { U } _ { z } ^ { + } , C _ { \operatorname* { m i n } } \geq . 7 0 , C _ { \mathrm { a v g } } \geq . 8 0 ] , } \\ & { \quad \Gamma ^ { \mathrm { F C - } } ( \phi _ { z } ( x ) ) = \mathbf { 1 } [ x \in \mathcal { U } _ { z } ^ { - } , V _ { \operatorname* { m a x } } \geq . 7 5 ] . } \end{array}\tag{15}
$$

(16)

The shortlist fixes necessary geometry and quality conditions, the audit defines semantic type, and GroupScore ranks candidates only within type. Among audit-confirmed failures satisfying $d \geq . 1 2 , \mathrm { T o p } _ { m }$ retains the highest-GroupScore candidates: visually plausible hard negatives rather than the most degraded or highest-dispersion failures. Every rollout stores its generating form and seed. The compiler consumes same-seed, diferent-form success–failure pairs first and fills the remaining budget with top-gap audited fallback pairs. “Form-contrastive” describes generating-form provenance, not form-dependent correctness of a fixed video. The clipped GroupScore gap becomes $\omega _ { z , r } .$

Formally, let $g ( x ^ { + } , x ^ { - } ; z ) = S _ { \mathrm { g r p } } ( x ^ { + } , z ) - S _ { \mathrm { g r p } } ( x ^ { - } , z )$ The set $\Pi _ { z } ^ { \mathrm { m a t c h } }$ contains same-seed, diferent-form pairs $( x _ { j , s } ^ { + } , x _ { i , s } ^ { - } )$ with $i \neq j , x _ { j , s } ^ { + } \in \mathcal { X } _ { z } ^ { + } , x _ { i , s } ^ { - } \in \mathcal { X } _ { z } ^ { \mathrm { F C - } }$ , and $g \ \geq \ \tau _ { \mathrm { p a i r } } .$ . With top-gap fallback set $\Pi _ { z } ^ { \mathrm { f a l l b a c k } }$ , the full set is Π<sub>z</sub> = Π<sup>match</sup> ∪ Π<sup>fallback</sup>; for each z z $r ~ \in ~ \Pi _ { z } , ~ g _ { z , r } ~ =$ $g ( x _ { z , r } ^ { + } , x _ { z , r } ^ { - } ; z )$ and $\omega _ { z , r } = \mathrm { c l i p } ( g _ { z , r } , \omega _ { \mathrm { m i n } } , \omega _ { \mathrm { m a x } } )$ . We use $\tau _ { \mathrm { p a i r } } = 0 . 1 5$ and clip weights to [0.05, 1.00].

```tcl
Algorithm 2 Audited Prompt-Form Supervision Compiler
Require: Verified group P(z); pool $\mathcal { X } _ { z } ;$ score matrix $\mathbf { A } _ { z } ;$ limits m<sub>+</sub>, m<sub>−</sub>; pair
margin m
Ensure: Shared-target records $\mathcal { D } _ { \mathrm { G C D } } ^ { z }$ and preference records $\mathcal { D } _ { \mathrm { F C } } ^ { z }$
1: Form $\mathcal { U } _ { z } ^ { + }$ and $\mathcal { U } _ { z } ^ { - }$ from the non-audit score/quality conditions
2: for each candidate x $\in { \mathcal { U } } _ { z } ^ { + } \cup { \mathcal { U } } _ { z } ^ { - }$ do
3: Audit $T _ { z }$ to obtain $( C _ { \mathrm { m i n } } , C _ { \mathrm { a v g } } , V _ { \mathrm { m a x } } ) ;$ ; form $\phi _ { z } ( x )$
4: end for
$\begin{array} { r } { \mathfrak { I } { : } \mathcal { C } _ { z } ^ { + }  \{ x \in \mathcal { U } _ { z } ^ { + } : C _ { \operatorname* { m i n } } \geq \tau _ { \operatorname* { m i n } } ^ { + } , C _ { \mathrm { a v g } } \geq \tau _ { \mathrm { a v g } } ^ { + } \} } \end{array}$
6 $: \mathcal { C } _ { z } ^ { \mathrm { F C - } } \gets \{ x \in \mathcal { U } _ { z } ^ { - } : V _ { \mathrm { m a x } } \geq \tau _ { \mathrm { c r i t } } ^ { - } \}$
7: $\mathcal { X } _ { z } ^ { + }  \mathrm { T o p } _ { m _ { + } } ( \mathcal { C } _ { z } ^ { + } ; S _ { \mathrm { g r p } } )$
8: ${ \mathcal { X } } _ { z } ^ { \mathrm { F C - } } \gets \mathrm { T o p } _ { m _ { - } } ( { \mathcal { C } } _ { z } ^ { \mathrm { F C - } } ; S _ { \mathrm { g r p } } )$
9: $\mathcal { D } _ { \mathrm { G C D } } ^ { z }  \{ ( \mathcal { P } ( z ) , x ^ { + } ) : x ^ { + } \in \mathcal { X } _ { z } ^ { + } \}$
10: $\mathcal { D } _ { \mathrm { F C } } ^ { z }  \mathcal { O }$
11: for each matched-seed pair, then each audited fallback pair do
12: $g  S _ { \mathrm { g r p } } ( x ^ { + } , z ) - S _ { \mathrm { g r p } } ( x ^ { - } , z )$
13: i ${ \mathrm { ~ \ : ~ } } ( { \mathrm { ~ \geq ~ } }$ m then
14: $\omega \gets \mathrm { c l i p } ( g , \omega _ { \mathrm { m i n } } , \omega _ { \mathrm { m a x } } )$
15: Add $( \mathcal { P } ( z ) , x ^ { + } , x ^ { - } , \omega ) \mathrm { t o } \mathcal { D } _ { \mathrm { F C } } ^ { z }$
16: end if
17: end for
18: return $( \mathcal { D } _ { \mathrm { G C D } } ^ { z } , \mathcal { D } _ { \mathrm { F C } } ^ { z } )$
```

## Compiler yield and pair provenance

The geometry/quality pre-screen sends only the union ${ \mathcal { U } } _ { z } ^ { + } \cup$ $\mathcal { U } _ { z } ^ { - }$ to the structured auditor; candidates satisfying both branches are deduplicated before atomic queries. Across the 5,000 training intents, 4,371 provide at least one audited positive and 3,228 provide at least one audited negative. The compiler produces at least one preference pair for 3,069 intents; 1,302 additional intents contribute only to GCD. Among 5,907 retained preference pairs, 71.7% are sameseed, diferent-form pairs and 28.3% are audited fallback pairs.

<table><tr><td>Compiler outcome</td><td>Count</td><td>Rate (%)</td></tr><tr><td>At least one audited positive</td><td>4,371</td><td>87.42</td></tr><tr><td>At least one audited negative</td><td>3,228</td><td>64.56</td></tr><tr><td>At least one preference pair</td><td>3,069</td><td>61.38</td></tr><tr><td>GCD-only intent</td><td>1,302</td><td>26.04</td></tr><tr><td>No retained supervision</td><td>629</td><td>12.58</td></tr><tr><td>Matched-seed retained pairs</td><td>4,236</td><td>71.71</td></tr><tr><td>Fallback retained pairs</td><td>1,671</td><td>28.29</td></tr></table>

Table 10: Supervision yield. Intent rates use 5,000 source intents; pair-provenance rates use 5,907 retained preference pairs.

<table><tr><td>Commitments</td><td>N</td><td>Positive</td><td>Negative</td><td>Pair</td></tr><tr><td> $J _ { z } = 3 – 4$ </td><td>1,617</td><td>1,492 (92.3)</td><td>913 (56.5)</td><td>876 (54.2)</td></tr><tr><td> $J _ { z } = 5 – 6$ </td><td>2,356</td><td>2,068 (87.8)</td><td>1,539 (65.3)</td><td>1,474 (62.6)</td></tr><tr><td> $J _ { z } \geq 7$ </td><td>1,027</td><td>811 (79.0)</td><td>776 (75.6)</td><td>719 (70.0)</td></tr><tr><td>All intents</td><td>5,000</td><td>4,371 (87.4)</td><td>3,228 (64.6)</td><td>3,069 (61.4)</td></tr></table>

Table 11: Supervision yield by the number of atomic commitments. Cells report count (rate $\% )$ . Min/max audit aggregation is deliberately more conservative for longer structured targets.

Proposition 1 (Coverage of an eligible positive). If a $r e \mathrm { - }$ tained candidate $x ^ { + }$ satisfies $w ( x ^ { + } ) \geq \tau _ { w }$ and $d ( x ^ { + } ) \leq \tau _ { d } .$ then

$$
\begin{array} { r l } { A ( x ^ { + } , p _ { i } ) \geq \tau _ { w } } & { { } \forall i , } \\ { | A ( x ^ { + } , p _ { i } ) - A ( x ^ { + } , p _ { j } ) | \leq \tau _ { d } } & { { } \forall i , j . } \end{array}\tag{17}
$$

Proof. Every score is at least the minimum w. All scores lie in [w, b], whose diameter is $b - w = d .$ □

This is the property required for target sharing: one favorable wording cannot by itself admit a positive. Mean-only ranking cannot provide this guarantee; for example, $( \mu , \mu )$ and $( \mu - \delta , \mu + \delta )$ have the same mean but diferent worst scores and drops.

For an already audit-confirmed negative, a largedispersion profile identifies a hard case rather than defining its semantic status. If $b ( x ^ { - } ) \geq \tau _ { b }$ and $d ( x ^ { - } ) \geq \tau _ { d } ^ { - }$ , then there exist scoring forms $p _ { h } , p _ { \ell }$ such that

$$
A ( x ^ { - } , p _ { h } ) \geq \tau _ { b } , \qquad A ( x ^ { - } , p _ { \ell } ) \leq A ( x ^ { - } , p _ { h } ) - \tau _ { d } ^ { - } .\tag{18}
$$

Thus the rejected set contains visually plausible, independently confirmed semantic failures with uneven scoring evidence, rather than universally poor videos or raw evaluator disagreement. The elementary bound

$$
w ( x ) \geq \mu ( x ) - d ( x )\tag{19}
$$

also shows why improving the mean while reducing drop strengthens the lower tail.

## Group-Consistent Distillation

Each intent retains a positive set $\mathcal { X } _ { z } ^ { + }$ , and each record samples $x ^ { + } \sim q _ { z } ^ { + }$ from that set. For time t and form $p _ { i }$ , let $h _ { i } =$ $v _ { \theta } ( x _ { t } ^ { + } , t , p _ { i } )$ and $u \ = \ u _ { t } ( x ^ { + } )$ . With normalized weights $\alpha _ { i } > 0 ;$ , the per-instance GCD loss is

$$
\mathcal { L } _ { \mathrm { G C D } } ( x ^ { + } , z , t ) = \sum _ { i } \alpha _ { i } \| h _ { i } - u \| _ { 2 } ^ { 2 } .\tag{20}
$$

Proposition 2 (Shared-target decomposition). Let $\bar { h } _ { \alpha } \ =$ $\sum _ { i } { { \bar { \alpha } } _ { i } } h _ { i }$ . Then

$$
\mathcal { L } _ { \mathrm { G C D } } = \Vert \bar { h } _ { \alpha } - u \Vert _ { 2 } ^ { 2 } + \sum _ { i } \alpha _ { i } \Vert h _ { i } - \bar { h } _ { \alpha } \Vert _ { 2 } ^ { 2 }\tag{21}
$$

$$
= \| \bar { h } _ { \alpha } - u \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 } \sum _ { i , j } \alpha _ { i } \alpha _ { j } \| h _ { i } - h _ { j } \| _ { 2 } ^ { 2 } .\tag{22}
$$

Proof. Write $h _ { i } - u = \left( h _ { i } - \bar { h } _ { \alpha } \right) + \left( \bar { h } _ { \alpha } - u \right)$ and expand the weighted squared norm. The cross term vanishes because $\begin{array} { r } { \sum _ { i } \overline { { \alpha } } _ { i } ( h _ { i } - \bar { h } _ { \alpha } ) = 0 ; } \end{array}$ the second equality is the weighted pairwise-variance identity. □

The first term fits the group-average prediction to the sampled valid target, while the second controls disagreement among verified forms. Sharing one target within a record therefore induces pairwise consistency, but sampling multiple retained positives across records does not define a unique canonical rendering. For uniform weights, $\partial \mathcal { L } _ { \mathrm { G C D } } / \partial h _ { i } \ \dot { = } 2 ( h _ { i } - u ) / L ; $ a prediction-space step contracts each pairwise diference by $\bar { 1 } - 2 \eta / L$ when $0 < \eta < L$

Group-consistent mining is essential to this construction. If $x ^ { + }$ were selected only by its mean score, a low-scoring form could be hidden by favorable forms and target sharing would transfer a semantically unsuitable video to that form. Proposition 1 removes precisely this failure mode.

## Form-Contrastive Flow-DPO

Let $\Delta _ { i } ( \theta )$ be the positive–negative flow-loss margin under form $p _ { i }$ , measured relative to the frozen post-GCD reference model. Define

$$
\phi _ { \beta } ( \Delta ) = - \log \sigma ( \beta \Delta ) = \log ( 1 + e ^ { - \beta \Delta } ) ,\tag{23}
$$

and, for a mined pair,

$$
\mathcal { L } _ { \mathrm { g r p } } ( z , x ^ { + } , x ^ { - } ) = \omega _ { z , r } \sum _ { i } \alpha _ { i } \phi _ { \beta } ( \Delta _ { i } ) .\tag{24}
$$

All experiments use $\alpha _ { i } = 1 / L$ and $\beta = \beta _ { \mathrm { D P O } }$

Proposition 3 (Conservative per-form surrogate). Because ϕ<sub>β</sub> is convex,

$$
\phi _ { \beta } \left( \sum _ { i } \alpha _ { i } \Delta _ { i } \right) \leq \sum _ { i } \alpha _ { i } \phi _ { \beta } ( \Delta _ { i } ) .\tag{25}
$$

Proof. $\phi _ { B } ^ { \prime \prime } ( \Delta ) \ = \ \beta ^ { 2 } \sigma ( \beta \Delta ) \sigma ( - \beta \Delta ) \ > \ 0$ , so Jensen’s inequality applies. □

Averaging the nonlinear loss over forms is therefore stricter than applying one logistic loss to the average margin: an easy form cannot cancel a failed form before the nonlinearity. The form-wise gradient

$$
\frac { \partial \mathcal { L } _ { \mathrm { g r p } } } { \partial \Delta _ { i } } = - \omega _ { z , r } \alpha _ { i } \beta \sigma ( - \beta \Delta _ { i } )\tag{26}
$$

has larger magnitude for smaller margins, so unresolved forms automatically receive stronger updates. The clipped pair weight $\omega _ { z , r }$ reflects mining confidence while bounding the influence of unusually large GroupScore gaps.

## Complementarity of the two stages

## Two distinct controls

GCD controls conditional disagreement around positives sampled from a valid support set. It contains no rejected sample and does not establish a preference boundary. FC-DPO orders positives above independently audited negatives under every form; matched seeds reduce stochastic confounding. The stages therefore address complementary failure modes: shared positive support first, explicit rejection of verified hard failures second.

The analysis above characterizes the empirical objectives over the finite verified form group associated with each intent. Robustness to broader language variation is evaluated empirically through the natural-caption controls.

## Computational Cost and Implementation Details

The complete supervision-construction and two-stage optimization flow is summarized in Algorithm 1 of the main paper. Here we provide the computational cost, frozen protocol settings, and implementation details needed for reproduction.

For M candidates and L prompt forms, cross-scoring requires ML prompt-conditioned evaluator calls. Computing the statistics is O(ML), and sorting candidates is O(M log M) per intent. These quantities are computed during supervision construction and can be cached. The optimization stages use fixed targets and pairs, so their per-step complexity remains comparable to the corresponding flowmatching or flow-DPO update.

## Implementation configuration

The transfer experiment uses the oficial ltx-2.3-22b-dev.safetensors checkpoint (Lightricks 2026).

Software environment. The Linux software stack comprised Python 3.11.11, PyTorch 2.10.0, Transformers 5.2.0, Difusers 0.37.0, Accelerate 1.13.0, NumPy 1.26.4, and PyYAML 6.0.3.

Per-form mining alignment and quality are computed as

$$
A _ { i } ( x ) = 0 . 6 R _ { \mathrm { V Q A } } ( x , p _ { i } ) + 0 . 4 R _ { \mathrm { V S } } ( x , p _ { i } ) ,\tag{27}
$$

$$
Q ( x ) = 0 . 6 R _ { \mathrm { D O V E R } } ( x ) + 0 . 4 R _ { \mathrm { m o t i o n } } ( x ) .\tag{28}
$$

After score normalization, the current GroupScore coeficients are

$$
( \lambda _ { \mu } , \lambda _ { w } , \lambda _ { d } , \lambda _ { q } ) = ( 0 . 5 0 , 0 . 2 5 , 0 . 1 5 , 0 . 1 0 ) ,\tag{29}
$$

with the drop term entering negatively. These scores construct supervision; reported benchmark values are recomputed with the corresponding benchmark evaluators.

For a prompt form $p _ { i }$ and a mined pair $( x ^ { + } , x ^ { - } )$ , the flow-DPO margin is

$$
\begin{array} { r l } & { \Delta _ { i } ( \theta ) = \ell _ { \mathrm { F M } } ( x ^ { - } , p _ { i } ; \theta ) - \ell _ { \mathrm { F M } } ( x ^ { + } , p _ { i } ; \theta ) } \\ & { ~ - \ell _ { \mathrm { F M } } ( x ^ { - } , p _ { i } ; \theta _ { \mathrm { r e f } } ) + \ell _ { \mathrm { F M } } ( x ^ { + } , p _ { i } ; \theta _ { \mathrm { r e f } } ) . } \end{array}\tag{30}
$$

where $\theta _ { \mathrm { r e f } }$ is the frozen post-GCD checkpoint. The per-form update is

$$
\begin{array} { r } { \ell _ { \mathrm { F C } } ( p _ { i } , x ^ { + } , x ^ { - } ) = - \log \sigma ( \beta _ { \mathrm { D P O } } \Delta _ { i } ( \theta ) ) } \\ { + \lambda _ { \mathrm { a u x } } \ell _ { \mathrm { F M } } ( x ^ { + } , p _ { i } ; \theta ) . } \end{array}\tag{31}
$$

with $\beta _ { \mathrm { D P O } } ~ = ~ 0 . 3$ and $\lambda _ { \mathrm { a u x } } = 0 . 0 5$ . The auxiliary term is chosen-only: it stabilizes the preferred support without directly fitting the rejected video.

For GCD, one sampled time t and one positive corruption are shared across all forms of a record, so the update isolates prompt conditioning while retaining a common target. For FC-DPO, the preferred and rejected branches share the same sampled time but use independent Gaussian corruptions. The policy and frozen reference evaluate the same corrupted input within each branch. This preserves a matched policy–reference comparison without coupling the preferred and rejected gradients through identical noise.

## Protocol-level compute accounting

Table 13 compares the per-run execution pattern under the matched 100,000-video rollout budget used in the main experiments. Video generation is the dominant shared component. VideoDPO applies single-form scoring and one preference-training stage to the fixed pool; OnlineVPO interleaves on-policy generation, scoring, and policy/reference updates; VARI performs cross-form scoring once, audits only the union shortlist, and then runs GCD and FC-DPO ofline. Because the stages have diferent execution patterns, we report matched workloads and execution structure rather than a single aggregate runtime metric. Repeated runs used for variance estimation are excluded from this per-run protocol comparison.

Under this matched budget, VARI incurs modest additional ofline work over VideoDPO and remains in the same endto-end cost regime as OnlineVPO. Its additional scoring and auditing are one-time and highly parallelizable, whereas OnlineVPO introduces sequential dependencies between generation, scoring, and policy updates. All three methods retain the same generator architecture and sampling configuration at inference, so VARI adds no per-video deployment cost.

Reproducibility and release. Headline trainable systems use three matched training seeds and fixed evaluation seeds. The four attribution-critical variants—GCD, intent-auditonly FC-DPO, VARI without matched-seed pairs, and full VARI—also use three paired runs from matched GCD branches; remaining ablations use one run on a shared development split unless stated otherwise. Compared methods share prompt groups, evaluation seeds, decoding settings, and evaluator versions whenever permitted by the method definition. VARI-specific settings are selected from heldout validation behavior and the accuracy–cost diagnostics reported below. The project implementation includes the VARI compiler, objectives, metrics, schemas, deterministic synthetic fixtures, tests, and manuscript-aggregate audits. External video backbones and evaluator implementations are connected through documented adapters and are not redistributed with this preprint. The synthetic fixtures support interface and regression tests, while the manuscript audits verify reported aggregate arithmetic. Source code, configurations, and evaluation scripts are not distributed with this preprint; any public release is subject to author and organizational approval. Any processed metadata release is also subject to applicable third-party licenses and organizational policies.

## Evaluation Protocol

## Intent-level robustness statistics

We extend the oficial T2V-CompBench and VBench evaluator implementations with verified prompt-form groups and intent-level aggregation, denoting the resulting protocols T2V-CompBench-R and VBench-R. Each verified rewrite inherits the source benchmark category and structured semantic target; the generator input is the only changed component. For a benchmark score $s _ { i } ^ { ( b ) } ( z )$ obtained for form i of intent z, the robustness extension computes

<table><tr><td rowspan="2" colspan="2"></td><td>Item</td><td>Mining and optimization configuration</td></tr><tr><td rowspan="3">Retained positives /</td><td rowspan="3">Up to two audited positives and two audited hard negatives per intent</td></tr><tr><td>Item</td><td>Data and generation configuration negatives Intent audit</td></tr><tr><td>Main / transfer backbone</td><td>HunyuanVideo-1.5 / LTX-2.3 (22B dev)</td><td></td><td>Gemini 3.5 Flash enum score call; Qwen/Qwen3.6-27B replacement;</td></tr><tr><td>Trainable component</td><td>Video DiT only; text encoders and VAE</td><td>Eligibility</td><td>frozen thresholds  $C _ { \mathrm { m i n } } / C _ { \mathrm { a v g } } / w / d / Q$  for positives;</td></tr><tr><td>Training intents</td><td>frozen 5,000</td><td></td><td> $V _ { \operatorname* { m a x } } / b / d / Q$  for negatives Margin 0.15; matched-seed pairs</td></tr><tr><td>Prompt-form groups</td><td>One source form and four verified semantic-preserving rewrites per intent</td><td>Pairing</td><td>prioritized; weights clipped to</td></tr><tr><td>Generation seeds</td><td>Four intent-specific seeds reused across all verified forms</td><td>Initialization</td><td>[0.05, 1.00] GCD from step 2000; FC-DPO from fixed</td></tr><tr><td>Initial rollout pool</td><td>100,000 videos (5,000 intents × 5 forms × 4 seeds)</td><td>Updates / learning</td><td>step-2400 GCD checkpoint 400  $/ 5 \times 1 0 ^ { - 6 } ~ ( \mathrm { G C D } ) ; 6 0 0 / 2 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Hunyuan benchmark generation</td><td>480p, 16:9, 121 frames at 24 fps, 50 sampling steps</td><td>rates Optimizer / precision</td><td>(FC-DPO) AdamW,  $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 ) ,$ </td></tr><tr><td>Hunyuan qualitative generation</td><td>720p, 16:9, 121 frames at 24 fps, 50 sampling steps</td><td>Batch / schedule</td><td> $\epsilon = 1 0 ^ { - 8 } ,$  wd 0.01; bf16 Global 48; no accumulation; clip 1.0;</td></tr><tr><td>LTX-2.3 transfer generation</td><td>480p, 16:9, 121 frames at 24 fps, 30 sampling steps</td><td>DPO coefficients</td><td>50-step warmup then constant LR  $\beta _ { \mathrm { D P O } } \dot { = } 0 . 3 ;$  chosen-positive auxiliary</td></tr><tr><td>StoryEval generation</td><td>Official 10s HunyuanVideo evaluation</td><td>Checkpointing</td><td>coefficient 0.05 Save every 300 updates; fixed terminal</td></tr><tr><td>setting</td><td></td><td>Trainable parameters</td><td>checkpoints, no test/qualitative selection Full video DiT; text encoders and VAE</td></tr></table>

Table 12: Core implementation configuration. Compared methods use matched generation settings and total sampling budgets whenever permitted by the method definition.

<table><tr><td>Method</td><td>Rollouts</td><td>Scoring / supervision</td><td>Optimization pattern</td><td>Inference overhead</td></tr><tr><td>VideoDPO</td><td>100k</td><td>Single-form reward and pair construction</td><td>Fixed-pool preference training</td><td>None</td></tr><tr><td>OnlineVPO</td><td>100k</td><td>Online reward and pair refresh</td><td>Interleaved generation and updates</td><td>None</td></tr><tr><td>VARI</td><td>100k</td><td>Cross-form scores; shortlist-only audit</td><td>Parallel offline GCD + FC-DPO</td><td>None</td></tr></table>

Table 13: Per-run protocol comparison under a matched video-rollout budget. API calls for the shortlist audit are logged separately by the compiler.

$$
\mathrm { M e a n } ( z ) = L ^ { - 1 } \sum _ { i } s _ { i } ^ { ( b ) } ( z ) ,
$$

$$
\mathrm { W o r s t } ( z ) = \operatorname* { m i n } _ { i } s _ { i } ^ { ( b ) } ( z ) ,\tag{32}
$$

$$
{ \mathrm { D r o p } } ( z ) = \operatorname* { m a x } _ { i } s _ { i } ^ { ( b ) } ( z ) - \operatorname* { m i n } _ { i } s _ { i } ^ { ( b ) } ( z ) .
$$

Dataset-level Mean, Worst, and Drop are macro-averaged over visual intents. Form-Induced Failure Rate identifies groups for which at least one form passes a fixed benchmarkspecific threshold while another form fails it:

$$
f _ { z } ^ { ( b ) } = \mathbf { 1 } \left[ \operatorname* { m i n } _ { i } s _ { i } ^ { ( b ) } ( z ) < \tau _ { b } \wedge \operatorname* { m a x } _ { i } s _ { i } ^ { ( b ) } ( z ) \geq \tau _ { b } \right] ,\tag{33}
$$

$$
\mathrm { F I F R } = 1 0 0 N ^ { - 1 } \sum _ { z = 1 } ^ { N } f _ { z } ^ { ( b ) } .\tag{34}
$$

The threshold is fixed independently of the compared method. T2V-CompBench-R uses a validation-frozen threshold for each inherited category, and VBench-R uses one for each matched semantic dimension. The reported FIFR is the macro-average of the resulting category- or dimension-wise failure rates. FIFR measures mixed-success inconsistency and can also be low when every form fails, so it is reported alongside Mean and Worst to retain absolute performance information. Standard benchmark scores and robustness statistics use the same generated samples and evaluator implementation within each comparison.

<table><tr><td>T2V-C category Consist. Dynamic Spatial Motion Action Interact. Numeracy</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tb (%)</td><td>78.07</td><td>7.18</td><td>50.22</td><td>36.31</td><td>65.09</td><td>56.37</td><td>20.19</td></tr></table>

Table 14: Validation-fixed success thresholds used by T2V-CompBench-R FIFR. VBench-R analogously uses the basemodel validation median of each matched semantic dimension.

## Comparison discipline

All methods receive the same prompt strings, evaluation seed sets, and generation configuration within each benchmark. The ofline methods share a common mined rollout pool; OnlineVPO refreshes samples on-policy under the same total generation budget. Baseline checkpoint selection precedes qualitative inspection. Prompt-enhancement comparisons freeze each enhanced prompt and feed it identically to the Base and VARI-trained DiT. Quantitative VBench and T2V-CompBench results use the 480p, 121-frame, 24-fps

Hunyuan protocol with 50 sampling steps; the qualitative clips use 720p with the same frame count, frame rate, and 50-step sampler. StoryEval follows its oficial 10s Hunyuan-Video evaluation protocol and serves as an auxiliary eventlevel diagnostic outside mining and checkpoint selection.

## Statistical protocol and repeated runs

The statistical unit is the source intent, never an individual form or frame. Continuous paired diferences use 10,000 intent-level bootstrap resamples for 95% confidence intervals and a two-sided Wilcoxon signed-rank test. FIFR uses paired bootstrap intervals for rate diferences and McNemar’s test. The eight primary robustness endpoints are corrected jointly with Holm’s procedure. Human comparisons use cluster bootstrap over intent groups and retain ties.

<table><tr><td>Method</td><td></td><td></td><td>T2V-C Avg. T2V-C-R Worst VBench-R Worst</td></tr><tr><td>OnlineVPO</td><td> $5 0 . 4 9 { \pm } 0 . 2 3 $ </td><td> $4 6 . 9 5 { \pm } 0 . 2 7 $ </td><td> $6 5 . 0 7 { \pm } 0 . 2 9$ </td></tr><tr><td>GCD</td><td> $5 0 . 4 8 { \pm } 0 . 1 8$ </td><td> $4 8 . 8 1 \pm 0 . 2 1$ </td><td> $6 7 . 1 2 { \pm } 0 . 2 5 $ </td></tr><tr><td>VARI</td><td> ${ \pm } 2 . 1 3 { \pm } 0 . 1 9$ </td><td> $\pm 9 . 5 3 \pm 0 . 2 2 $ </td><td> ${ \bf 6 8 . 7 3 { \scriptstyle \pm 0 . 2 6 } }$ </td></tr></table>

Table 15: Mean ± standard deviation over three independent training runs.

<table><tr><td>Attribution-critical variant</td><td>Mean</td><td>Worst</td><td>Drop</td><td>FIFR</td></tr><tr><td>GCD</td><td> $5 0 . 4 6 \pm 0 . 1 8$ </td><td>48.88±0.21</td><td> $2 . 5 7 { \pm } 0 . 1 3$ </td><td>5.41±0.24</td></tr><tr><td>Intent-audit-only FC-DPO</td><td> $5 0 . 9 4 \pm 0 . 1 5$ </td><td> $4 9 . 0 9 \pm 0 . 1 9$ </td><td> $2 . 6 1 \pm 0 . 1 2$ </td><td>5.02±0.22</td></tr><tr><td>VARI w/o matched-seed pairs</td><td> $5 1 . 0 6 \pm 0 . 1 9$ </td><td> $4 9 . 2 8 \pm 0 . 1 8$ </td><td> $2 . 4 6 \pm 0 . 1 6$ </td><td>4.87±0.20</td></tr><tr><td>VARI (matched-seed prioritized)</td><td> ${ \bf 5 1 . 1 7 \pm 0 . 1 3 }$ </td><td> $\pm 9 . 5 1 \pm 0 . 1 6$ </td><td> ${ \bf 2 . 2 9 \pm 0 . 1 1 }$ </td><td>4.44±0.18</td></tr></table>

Table 16: Mean ± standard deviation over three paired training branches on the fixed development split.

Relative to audit-only FC-DPO, full VARI changes Mean/Worst/Drop/FIFR by $+ 0 . 2 3 / + 0 . 4 2 / - 0 . 3 2 / - 0 . 5 8$ Relative to VARI without matched-seed pairs, the corresponding changes are $+ 0 . 1 1 / + 0 . 2 3 / - 0 . { \overset { \cdot } { 1 } } 7 / - 0 . 4 3$ . The same ordering is retained across the three paired training branches in Table 16, supporting a modest final contribution from matched provenance and a larger contribution from audited group-level preference optimization.

Against the strongest prompt-level comparator for each endpoint, VARI improves T2V-CompBench-R/VBench-R Worst by 2.45/3.55 points, reduces Drop by 1.21/2.77 points, and reduces FIFR by 3.78/2.99 percentage points. For VBench-R, paired 95% intervals are [2.66, 4.43] for Worst and [−3.39, −2.14] for Drop; the Worst comparison has Holm-adjusted $p < 0 . 0 0 1$ , and the FIFR reduction is significant under McNemar with adjusted $p < 0 . 0 1$ . These paired VBench-R statistics are computed at the source-intent level from the final per-intent outputs rather than reconstructed from aggregate means.

## Supervision Validity, Independence, and Boundary Tests

## Blind audit of mined supervision

We sample 100 candidates from each mining type and hide method identity, generating form, scoring form, and all automatic scores. Three annotators judge the video only against $T _ { z }$ and separately record whether it is visually usable. Human-label precision denotes semantic validity for positives and semantic-defect precision for negatives. Independent agreement is the percentage on which a held-out audit model and the human majority assign the same target label.

<table><tr><td>Mined type</td><td>Human precision</td><td>Visual usability</td><td>Indep. agreement</td></tr><tr><td>Group-consistent positive</td><td>96/100</td><td>91/100</td><td>88/100</td></tr><tr><td>Low-score negative</td><td>76/100</td><td>47/100</td><td>68/100</td></tr><tr><td>Dispersion-only negative</td><td>58/100</td><td>81/100</td><td>50/100</td></tr><tr><td>Intent-audit negative</td><td>85/100</td><td>76/100</td><td>78/100</td></tr><tr><td>Audited FC negative</td><td>91/100</td><td>79/100</td><td>84/100</td></tr></table>

Table 17: Blind supervision audit. Bold marks the best result among negative-mining variants; each cell reports valid counts out of $N = 1 0 0$ candidates per type.

Across the 300 candidates used for correlation analysis, the primary Gemini auditor’s $C _ { \mathrm { m i n } }$ correlates with human correctness at Spearman $\rho = 0 . 6 3 .$ Let $\mathcal { N } _ { \mathrm { G } }$ and $\mathcal { N } _ { \mathrm { Q } }$ be the retained negative sets from Gemini 3.5 Flash and Qwen3.6- 27B under the same per-intent budget $m _ { - } = 2 .$ . Their Jaccard overlap $| \mathcal { N } _ { \mathrm { G } } \cap \mathcal { N } _ { \mathrm { Q } } | \bar { / } | \mathcal { N } _ { \mathrm { G } } \cup \mathcal { N } _ { \mathrm { Q } } |$ is $0 . 7 3 ;$ an independent review of 100 Qwen-retained negatives confirms 90 defects. Dispersion-only negatives remain substantially less precise despite high visual usability, supporting the separation between scoring-view dispersion and semantic typing.

## Intent-disjoint construction and overlap audit

The 5,000 training intents are generated from a structured non-benchmark schema balanced across attributes, relations, actions, interactions, numeracy, and multi-event composition. Source-intent splits are frozen before rewriting. Evaluation groups use disjoint rewrite instructions and verifier instances. We normalize text for exact matching, retrieve lexical near-neighbors, and retrieve semantic neighbors using a frozen sentence encoder; every candidate above the validation-fixed threshold is manually reviewed against the structured intent.

<table><tr><td>Evaluation source</td><td>Retrieved candidates</td><td>Confirmed shared intents</td></tr><tr><td>VBench</td><td>13</td><td>0</td></tr><tr><td>T2V-CompBench</td><td>23</td><td>0</td></tr><tr><td>StoryEval</td><td>8</td><td>0</td></tr><tr><td>VATEX</td><td>17</td><td>0</td></tr><tr><td>MSR-VTT</td><td>26</td><td>0</td></tr></table>

Table 18: Training–evaluation overlap audit. Exact normalized overlap is zero for every source.

## Seed diversity and non-equivalent counterfactuals

For each held-out intent and form, we generate eight seeds. Appearance features use the frozen CLIP ViT-L/14 image encoder (Radford et al. 2021): 16 uniformly sampled frames are independently encoded, ℓ<sub>2</sub>-normalized, temporally averaged, and normalized again. Appearance diversity is the mean pairwise cosine distance across seeds. Motion features use the oficial VideoCLIP S3D checkpoint trained on HowTo100M (Xu et al. 2021): 32 ordered frames are encoded into the final clip-level video embedding, normalized, and compared with cosine distance. Both diversity scores are averaged by intent and form and reported relative to Base.

For counterfactuals, we create 200 prompt pairs that change exactly one count, spatial relation, action role, or temporal order. Three annotators confirm that 195/200 initial pairs difer only in the intended commitment; the five failures are replaced before generation. The held-out Vision-Reward checkpoint (Xu et al. 2026), which is not used in mining or checkpoint selection, scores each video against both prompts. After mapping its score to the common 0–100 scale, the symmetric margin is $\begin{array} { r } { \frac { 1 } { 2 } [ ( R ( x _ { p } , p ) - R ( x _ { p } , p ^ { \prime } ) ) + } \end{array}$ $[ R ( x _ { p ^ { \prime } } , p ^ { \prime } ) \dot { } - R ( x _ { p ^ { \prime } } , p ) ) ]$

<table><tr><td>Method</td><td>Appearance div. Motion div.</td><td></td><td>CF margin</td></tr><tr><td>Base</td><td>100.00</td><td>100.00</td><td>11.36</td></tr><tr><td>OnlineVPO</td><td>98.20</td><td>97.40</td><td>12.01</td></tr><tr><td>GCD</td><td>95.70</td><td>95.30</td><td>12.71</td></tr><tr><td>VARI</td><td>96.50</td><td>95.80</td><td>13.90</td></tr></table>

Table 19: Diversity retention relative to Base (%) and counterfactual semantic margin.

<table><tr><td>Change type</td><td>Base</td><td>OnlineVPO</td><td>GCD</td><td>VARI</td></tr><tr><td>Count</td><td>10.77</td><td>11.41</td><td>11.68</td><td>13.12</td></tr><tr><td>Spatial relation</td><td>13.26</td><td>14.01</td><td>14.82</td><td>15.97</td></tr><tr><td>Action-role swap</td><td>11.88</td><td>12.59</td><td>13.37</td><td>14.53</td></tr><tr><td>Temporal order</td><td>9.52</td><td>10.04</td><td>10.96</td><td>11.98</td></tr></table>

Table 20: Counterfactual margin by minimally changed semantic commitment.

VARI retains most seed-level appearance and motion diversity while increasing every counterfactual margin. These diagnostics provide evidence against mode contraction or indiscriminate suppression of prompt conditioning as the primary explanation for the robustness gains.

## Expanded Quantitative Results

## T2V-CompBench dimensions

Table 21 reports all seven displayed T2V-CompBench categories. The average is the arithmetic mean of the same seven entries.

Category-level observations. On T2V-CompBench, VARI obtains the highest overall average and leads five of the seven displayed categories: consistency, spatial relations, action, interaction, and numeracy. The remaining two categories expose a realistic trade-of rather than a uniformly monotone pattern. OnlineVPO is strongest on dynamic degree and motion binding, while GCD remains competitive on consistency, spatial relations, action, and interaction. The largest VARI gains occur in prompt-sensitive categories that require entity counts, relation preservation, or correct assignment of actions to participants.

<table><tr><td>Category</td><td>Base VideoDPO OnlineVPO GCD</td><td></td><td></td><td></td><td>VARI</td></tr><tr><td>Consistency</td><td>79.82</td><td>81.34</td><td>82.31</td><td>83.26</td><td>86.64</td></tr><tr><td>Dynamic</td><td>7.47</td><td>8.25</td><td>10.04</td><td>8.66</td><td>9.58</td></tr><tr><td>Spatial</td><td>53.11</td><td>55.56</td><td>56.72</td><td>56.97</td><td>58.55</td></tr><tr><td>Motion</td><td>40.29</td><td>41.89</td><td>43.91</td><td>39.88</td><td>40.42</td></tr><tr><td>Action</td><td>69.31</td><td>71.24</td><td>72.13</td><td>72.71</td><td>75.06</td></tr><tr><td>Interaction</td><td>60.57</td><td>63.49</td><td>64.49</td><td>66.61</td><td>67.78</td></tr><tr><td>Numeracy</td><td>23.24</td><td>25.42</td><td>23.81</td><td>25.29</td><td>26.87</td></tr><tr><td>Average</td><td>47.69</td><td>49.60</td><td>50.49</td><td>50.48</td><td>52.13</td></tr></table>

Table 21: T2V-CompBench category scores under the unified evaluation protocol. Values are percentages; the table is transposed to preserve a stable single-column layout.

## Ablations Linked to the Formal Analysis

Unless otherwise stated, the ablations in this section are measured on a fixed development split of held-out T2V-CompBench-R intent groups. Absolute values therefore need not exactly match the headline test-set results in the main paper, but every variant within an ablation table is evaluated on the same split and with the same evaluator.

## Training objective

The objective ablation separates paraphrase exposure, shared support, semantic audit, and preference construction. Reward-SFT raises the mean but does not protect the weakest form. GCD produces the largest initial contraction. Adding ordinary prompt-DPO after GCD increases Mean but degrades Worst, Drop, and FIFR, whereas audited group-lifted FC-DPO improves all four endpoints over GCD, showing that Stage 2 contributes semantic boundary refinement rather than additional training alone.

<table><tr><td>Variant</td><td>Mean</td><td>Worst Drop</td><td>FIFR</td></tr><tr><td>Reward-SFT</td><td>48.58</td><td>43.82 5.97</td><td>13.11</td></tr><tr><td>Prompt-DPO</td><td>49.91</td><td>46.77 4.49</td><td>9.24</td></tr><tr><td>FC-DPO</td><td>50.31</td><td>47.68 4.01</td><td>7.76</td></tr><tr><td>GCD</td><td>50.46</td><td>48.88</td><td>2.57 5.41</td></tr><tr><td>GCD + prompt-DPO</td><td>50.82</td><td>48.64</td><td>3.28 6.19</td></tr><tr><td>VARI</td><td>51.17</td><td>49.51</td><td>2.29 4.44</td></tr></table>

Table 22: Training-objective ablation on T2V-CompBench-R.

## Causal alternatives

Table 23 matches data volume and evaluation split across simple alternatives. Multi-form prompt-DPO tests paraphrase exposure without group scoring; direct worst-form DPO tests explicit lower-tail optimization; form-specific and single-positive distillation test whether the gain is merely more SFT or one canonical target; inference reranking tests whether GroupScore helps without changing the generator; and the final rows isolate dispersion, audit, and matched-seed pairing.

<table><tr><td>Variant</td><td>Mean</td><td>Worst</td><td>Drop</td><td>FIFR</td></tr><tr><td>Reward-SFT</td><td>48.58</td><td>43.82</td><td>5.97</td><td>13.11</td></tr><tr><td>Prompt-level DPO</td><td>49.91</td><td>46.77</td><td>4.49</td><td>9.24</td></tr><tr><td>Multi-form prompt-DPO</td><td>50.08</td><td>47.03</td><td>4.34</td><td>8.61</td></tr><tr><td>Direct worst-form DPO</td><td>50.39</td><td>48.17</td><td>3.12</td><td>6.38</td></tr><tr><td>Form-specific distillation</td><td>49.93</td><td>47.51</td><td>3.69</td><td>7.19</td></tr><tr><td>Single-positive GCD</td><td>50.18</td><td>48.56</td><td>2.89</td><td>5.93</td></tr><tr><td>GCD</td><td>50.46</td><td>48.88</td><td>2.57</td><td>5.41</td></tr><tr><td>GroupScore reranking (no train)</td><td>49.69</td><td>46.45</td><td>4.13</td><td>8.69</td></tr><tr><td>Dispersion-only FC-DPO</td><td>50.69</td><td>48.35</td><td>3.58</td><td>6.91</td></tr><tr><td>Intent-audit-only FC-DPO</td><td>50.94</td><td>49.09</td><td>2.61</td><td>5.02</td></tr><tr><td>VARI w/o matched-seed pairs</td><td>51.06</td><td>49.28</td><td>2.46</td><td>4.87</td></tr><tr><td>VARI (matched-seed prioritized)</td><td>51.17</td><td>49.51</td><td>2.29</td><td>4.44</td></tr></table>

Table 23: Full causal ablation on the fixed T2V-CompBench-R development split.

The simple alternatives improve individual metrics but do not reproduce the final balance. Direct worst-form DPO contracts the lower tail but gives lower Mean and higher FIFR; reranking improves selection but not generator behavior; and dispersion-only negatives underperform auditconfirmed negatives. The paired repeats in Table 16 show that audited group-level preference optimization raises Worst and reduces mixed-success failures beyond GCD, while matched provenance adds a smaller complementary improvement in Worst, Drop, and FIFR.

## Mining statistics

The component ablation matches Proposition 1 and the mean-only counterexample in the group-structured supervision analysis: mean-only selection can retain candidates with a weak tail, while adding w and d improves the lowertail metrics. The quality term retains visually usable positives rather than the primary robustness signal; its main efect is to remove visibly weak positives while keeping robustness nearly unchanged.

<table><tr><td>Mining score</td><td></td><td></td><td></td><td></td><td>µ ω −d |Worst↑ Drop↓ FIFR↓</td><td></td></tr><tr><td>Single-form</td><td></td><td>× x</td><td>x</td><td>41.08</td><td>16.89</td><td>14.94</td></tr><tr><td>Mean only</td><td>√ x</td><td></td><td>x</td><td>44.17</td><td>13.43</td><td>11.18</td></tr><tr><td>Mean + worst</td><td>√√</td><td></td><td>x</td><td>46.12</td><td>10.21</td><td>8.24</td></tr><tr><td>Mean + worst - drop</td><td>√√</td><td></td><td>√</td><td>46.95</td><td>7.48</td><td>5.96</td></tr><tr><td>Full + Q</td><td></td><td>√ √ √</td><td></td><td>46.87</td><td>7.64</td><td>6.08</td></tr></table>

Table 24: GroupScore-component ablation with T2V-CompBench-R evaluation.

## Pair construction and group scale

Audited form-contrastive pairs are more informative than globally low-score negatives because they combine a genuine target violation with visual plausibility and uneven crossform evidence. Increasing the number of verified forms improves coverage of wording variation but also increases crossscoring cost. The gains largely saturate beyond five verified forms, so we adopt five forms and four seeds as the default accuracy–cost tradeof.

<table><tr><td>Setting</td><td>Worst</td><td>Drop</td><td>FIFR</td></tr><tr><td>Random positive vs random negative</td><td>43.12</td><td>15.79</td><td>14.09</td></tr><tr><td>Best-group vs low-score</td><td>45.07</td><td>12.34</td><td>9.88</td></tr><tr><td>Best-group vs worst-group</td><td>45.74</td><td>9.83</td><td>7.41</td></tr><tr><td>Best-group vs form-contrastive</td><td>46.88</td><td>7.54</td><td>5.89</td></tr><tr><td>Top-m positives + mixed negatives</td><td>46.63</td><td>7.82</td><td>5.74</td></tr><tr><td>1 form, 4 seeds</td><td>41.23</td><td>16.21</td><td>14.98</td></tr><tr><td>3 forms, 4 seeds</td><td>44.93</td><td>10.71</td><td>9.37</td></tr><tr><td>5 forms, 2 seeds</td><td>45.19</td><td>9.97</td><td>8.71</td></tr><tr><td>5 forms, 4 seeds</td><td>46.81</td><td>7.73</td><td>6.07</td></tr><tr><td>7 forms, 4 seeds</td><td>46.92</td><td>7.68</td><td>5.91</td></tr></table>

Table 25: Pair-construction diagnostics (upper block) and rollout-scale diagnostics (lower block).

## Prompt-form verification

Unverified rewrites may increase lexical diversity while silently changing the target intent. Structured verification improves the validity of cross-form supervision rather than merely increasing the number of prompts.

<table><tr><td>Group construction</td><td>Mean Worst</td><td>Drop</td><td>FIFR</td></tr><tr><td>Single original form</td><td>49.91 41.24</td><td>16.23</td><td>14.98</td></tr><tr><td>Two verified forms</td><td>50.55 43.87</td><td>11.82</td><td>10.69</td></tr><tr><td>Five unverified rewrites</td><td>51.06 43.39</td><td>13.14</td><td>11.91</td></tr><tr><td>Five verified forms</td><td>51.76 46.81</td><td>7.74</td><td>6.04</td></tr></table>

Table 26: Efect of prompt-form verification and group size.

## Prompt Enhancement and Cross-Backbone Evaluation

## Oficial prompt-enhancement condition

HunyuanVideo-1.5 provides an oficial text-to-video promptrewriting instruction and a prompt-writing handbook for automatic enhancement (Tencent Hunyuan 2025a,b). The evaluation uses the oficial rewriting procedure without adding a method-specific semantic instruction. For each raw verified form, the enhanced text is frozen and supplied identically to the Base DiT and the VARI-trained DiT. Scoring remains anchored to the original benchmark intent, so the comparison tests whether generator-side alignment remains useful when inference-side enhancement is present.

## Oficial PE evaluation contract

Input. The same raw prompt form for both compared generators.

Enhancement. Apply the oficial HunyuanVideo-1.5 text-tovideo rewriting procedure once; do not manually edit one method’s enhanced prompt or use enhancement during model selection.

Generation. Freeze the enhanced prompt and use the same inference configuration for the Base and VARI-trained DiT.

Evaluation. Score both outputs against the same original benchmark target. The enhanced text changes the generationside conditioning, not the semantic reference used for comparison.

<table><tr><td>Input</td><td>Generator</td><td>T2V-C Avg.</td><td>R-Worst</td><td>R-Drop</td><td>R-FIFR</td></tr><tr><td>Raw</td><td>Base DiT</td><td>47.69</td><td>44.47</td><td>4.92</td><td>11.14</td></tr><tr><td>Raw</td><td>VARI DiT</td><td>52.13</td><td>49.53</td><td>2.76</td><td>4.43</td></tr><tr><td>Official PE</td><td>Base DiT</td><td>57.63</td><td>53.09</td><td>6.91</td><td>7.64</td></tr><tr><td>Official PE</td><td>VARI DiT</td><td>59.91</td><td>56.28</td><td>4.57</td><td>4.14</td></tr></table>

Table 27: Prompt-enhancement compatibility under a shared enhanced input.

## Cross-backbone transfer and event-level diagnostic

The LTX-2.3 experiment (Lightricks 2026) evaluates whether the same two-stage training recipe transfers beyond the primary HunyuanVideo-1.5 backbone. Table 28 reports the full comparison under the matched LTX-2.3 protocol.

<table><tr><td>LTX-2.3</td><td>T2V-C</td><td>R-Worst R-Drop</td><td>VBench-R Story</td><td></td></tr><tr><td>Base</td><td>44.79</td><td>41.86</td><td>5.48</td><td>58.31 29.66</td></tr><tr><td>VideoDPO</td><td>46.31</td><td>43.19 4.41</td><td>60.08</td><td>31.02</td></tr><tr><td>OnlineVPO</td><td>46.84</td><td>43.04</td><td>4.82 60.67</td><td>31.48</td></tr><tr><td>+GCD</td><td>47.57</td><td>44.79</td><td>3.51 61.61</td><td>32.17</td></tr><tr><td>+VARI</td><td>48.29</td><td>45.76</td><td>3.01</td><td>63.21 33.37</td></tr></table>

Table 28: Cross-backbone validation on LTX-2.3. VBench-R reports semantic Worst; Story denotes StoryEval.

StoryEval uses the oficial 10s HunyuanVideo generation protocol and evaluator implementation. Its longer, eventcentric prompts provide an auxiliary diagnostic of strict event completion outside supervision construction and checkpoint selection.

<table><tr><td>Method</td><td>Base</td><td>Reward-SFT</td><td>VideoDPO</td><td>OnlineVPO</td><td>VARI</td></tr><tr><td>StoryEval strict score</td><td>32.11</td><td>33.21</td><td>34.62</td><td>35.31</td><td>37.58</td></tr></table>

Table 29: StoryEval strict event-completion score.

## Auxiliary Natural-Caption Controls

## Verified VATEX multi-caption diagnostic

VATEX provides multiple English captions for the same source video, making it a useful auxiliary test bed for naturalcaption grouping (Wang et al. 2019). We first collect candidate caption groups from the same video and then apply the same semantic-preservation verification used by VARI: retained captions must agree on the main entities, participant roles, principal action, core interaction or spatial relation, and temporal scope. Groups that describe diferent moments or omit a required relation are discarded. The verification retains 619 VATEX intent groups, all disjoint from the VARI training intents and checkpoint-selection data. Generation uses the same inference configuration as T2V-CompBench-R, while scoring uses held-out VisionReward only for this auxiliary diagnostic (Xu et al. 2026).

<table><tr><td>Method</td><td>Mean</td><td>Worst Drop</td><td>FIFR</td></tr><tr><td>Base VideoDPO</td><td>63.77 65.99</td><td>59.66 6.18 62.02 5.31</td><td>9.21 7.43</td></tr><tr><td>OnlineVPO</td><td>66.47</td><td>61.92 6.14 64.49</td><td>7.75</td></tr><tr><td>GCD VARI</td><td>67.29 68.16 65.09</td><td>4.17 4.03</td><td>5.33 4.85</td></tr></table>

Table 30: Verified VATEX multi-caption diagnostic scored with held-out VisionReward.

## Disjoint MSR-VTT control

MSR-VTT provides a stricter leakage-sensitive disjoint control (Xu et al. 2016). We remove duplicate video IDs, exact caption duplicates, and near-duplicate captions against the VARI training pool and benchmark-derived evaluation sets. The remaining captions undergo the same semanticequivalence verification, retaining only groups that express the same visually testable intent. The filtering and verification retain 411 disjoint groups, which are evaluated with the same held-out reward protocol as VATEX.

<table><tr><td>Method</td><td>Mean Worst</td><td>Drop</td><td>FIFR</td></tr><tr><td>Base</td><td>61.63 57.17 59.33</td><td>6.67 6.07</td><td>10.22 8.76</td></tr><tr><td>VideoDPO OnlineVPO</td><td>63.61 64.05 59.24</td><td>6.37</td><td>9.00</td></tr><tr><td>GCD</td><td>65.21 61.69</td><td>4.96</td><td>6.33</td></tr><tr><td>VARI</td><td>65.88 62.34</td><td>4.69</td><td>5.84</td></tr></table>

Table 31: Disjoint MSR-VTT control under the same heldout reward protocol.

## Human and Qualitative Evaluation Protocols Human comparison

Human evaluation follows a blinded pairwise design on 200 held-out intent groups. Annotators receive the visual intent, the verified prompt forms, and anonymized video sets from two methods. Method identity and left/right placement are hidden and randomized. Separate questions assess intent faithfulness, consistency across equivalent forms, visual quality, and overall preference. Each comparison is judged by three independent annotators; ties are retained rather than forced into a directional choice. Win, tie, and loss are raw percentages over all judgments and therefore sum to 100%. Decisive-win rates excluding ties are reported only as an auxiliary statistic.

## Human-evaluation questions

Intent faithfulness. Which set more accurately depicts the required entities, actions, attributes, and relations? Prompt-form consistency. Which set preserves the same intended content more consistently across the equivalent forms? Visual quality. Which set has better perceptual quality, temporal coherence, and fewer visible artifacts? Overall preference. Considering faithfulness, consistency, and quality together, which set is preferred?

For the comparison with OnlineVPO, Table 32 reports integer outcomes and the percentages derived from them. The decisive-win rates excluding ties are 69.84% for intent

faithfulness, 73.11% for cross-form consistency, 63.93% for visual quality, and 70.24% overall. Confidence intervals use intent-cluster bootstrap; Krippendorf’s α is computed on the three-way win/tie/loss labels.
<table><tr><td>Dimension</td><td colspan="3">Counts</td><td colspan="3">Raw shares (%)</td><td>W-L</td><td>95% CI</td><td>α</td></tr><tr><td></td><td>W</td><td>T</td><td>L</td><td></td><td>W</td><td>T</td><td>L</td><td>(pt.)</td><td></td></tr><tr><td>Intent faithfulness</td><td>352</td><td>96</td><td>152</td><td>58.67</td><td>16.00</td><td>25.33</td><td>33.33</td><td>[25.9, 40.6] 0.59</td><td></td></tr><tr><td>Cross-form consistency</td><td>378</td><td>83</td><td>139</td><td>63.00</td><td>13.83</td><td>23.17</td><td>39.83</td><td>[32.4, 47.1] 0.63</td><td></td></tr><tr><td>Visual quality</td><td>319</td><td>101</td><td>180</td><td></td><td>53.17 16.83 30.00</td><td></td><td></td><td>23.17 [14.9, 31.3]0.57</td><td></td></tr><tr><td>Overall preference</td><td>347</td><td></td><td></td><td></td><td></td><td></td><td></td><td>106 147 57.83 17.67 24.50 33.33 [26.3, 40.4] 0.60</td><td></td></tr></table>

Table 32: Detailed blinded human evaluation: VARI versus OnlineVPO. Each dimension contains 600 judgments.

## Qualitative cases and video comparison protocol

The main paper reports two held-out T2V-CompBench source prompts, one from spatial relationships and one from action binding. Each source is paired with one verified semantic-preserving form. The left/right prompt pairs are:

• Spatial relation: “A boy reading in front of a bench.” and “A bench is positioned behind a boy who is reading.”

• Action binding: “A man is reading a newspaper on a park bench, while a squirrel is climbing a tree.” and “While a squirrel climbs a tree, a man reads a newspaper on a park bench.”

All compared methods use the same prompt string and inference configuration within each row. The complete qualitative clips are generated with HunyuanVideo-1.5 at 720p, 121 frames, 24 fps, and 50 sampling steps. Quantitative VBench and T2V-CompBench tables use the corresponding 480p, 50- step protocol. The main-paper three-frame strips are sampled in temporal order from these complete clips. The twelve full clips underlying the frame-strip comparisons follow the organization summarized below; the video files are not distributed with this preprint.

Qualitative clip organization   
The evaluated clips were organized as follows:   
spatial/   
source/{base,videodpo,vari}.mp4   
equivalent/{base,videodpo,vari}.mp4   
action\_binding/   
source/{base,videodpo,vari}.mp4   
equivalent/{base,videodpo,vari}.mp4   
All clips use the compared methods and the shared 720p gen  
eration protocol described above.

Spatial-relation case. Across both forms, Base and VideoDPO place the reader on the bench, replacing the requested in-front-of/behind relation with a frequent readingon-a-bench configuration. One baseline clip additionally duplicates the reader, but the shared failure is the relation change rather than the mere presence of a counting artifact. The two VARI clips preserve the intended relation throughout the sequence while allowing an unconstrained posture change: the boy sits on the grass in one form and stands in front of the bench in the other. The complete clips also expose temporal behavior hidden by the three-frame layout. Baseline book pages can flip repeatedly or change geometry abruptly, whereas the VARI outputs maintain a more coherent reading interaction; the seated result contains a brief localized page turn and the standing result remains stable without changing the spatial arrangement.

Action-binding case. The baseline videos display several ways in which a partially correct scene can still violate the compositional intent. The man–newspaper event is often present, but squirrels may be duplicated, appear on the bench rather than the tree, or fail to maintain a single upward trajectory; one clip instead duplicates the reader. The two VARI clips preserve both assignments over time: the man continues reading and the squirrel remains on the tree while moving upward. The full clips also show stable newspaper handling with limited page motion, rather than a still-frame coincidence, and make the climbing progression visible from low to higher positions on the trunk.

Checkpoint selection precedes qualitative inspection. We then select held-out groups with clear weakest-form improvements and interpretable failure modes; the quantitative conclusions are based on the full benchmark results.

## Scope of the public preprint

This preprint includes the full technical appendix and static qualitative panels. The twelve complete example clips and the project code/data are not distributed with it. The project implementation includes VARI schemas, the supervision compiler, GCD and FC-DPO objectives, robustness aggregation, prompt templates, deterministic synthetic fixtures, unit and smoke tests, a paper-to-code mapping, and manuscriptaggregate arithmetic audits. Documented adapters connect to external backbones, evaluator repositories, and pretrained weights.

<table><tr><td>Comparison</td><td>Forms Material shown</td><td></td></tr><tr><td>Main-paper spatial</td><td></td><td>2 × 3 Displayed strips (6 evaluated</td></tr><tr><td>relation</td><td></td><td>clips) Main-paper action binding 2 × 3 Displayed strips (6 evaluated</td></tr><tr><td></td><td></td><td>clips)</td></tr><tr><td>Appendix paired-form cases</td><td></td><td>2 × 2 Static temporally ordered panels</td></tr><tr><td></td><td></td><td>Appendix dynamics cases 2 × 1 Static temporally ordered panels</td></tr></table>

Table 33: Qualitative evidence shown in this preprint. Each count is prompt forms × compared methods or intents, as applicable.

## Additional qualitative results

Figure 5 provides four additional held-out comparisons using temporally ordered frames. Cases (a) and (b) compare the source prompt with one verified semantic-preserving form, while cases (c) and (d) focus on motion realization and temporal state change from the source prompt. These four cases are presented as static appendix panels; the twelve full clips underlying the main-paper frame strips are not distributed with this preprint.

Numeracy and group action. In panel (a), the numeral is a binding constraint rather than a stylistic preference.

“Five robots dance rhythmically in the lab.”  
![](images/911315ae94bc44d53da0c4ce95ed6671a856a2a5b6456a521107fb7b7ca28368.jpg)  
“Friends play a board game together at a cozy cafe.”

“Inside a laboratory, a group of five robots performs a rhythmic dance.”  
![](images/7c72a75e20ecc7534bd67074732ad38652f052e9baf57b9ce1721ea9e06514a3.jpg)  
“At a cozy cafe, a group of friends plays a board game together.”

![](images/0f0ba4bc2c0b705b06caf89ce935c32841f759fc53c0f5ac6a3c1619063d8efe.jpg)  
“Red car racing past an old church.”

![](images/37c166c32e8dcaaf5e93046fdbc2b6ec7bf838cf4d1545282886c3d476651b5f.jpg)  
(d) “A timelapse of a snowman melting under the afternoon sun.”

![](images/90c0cbc0621535625b43a0fb6692cea8e0a40d41b4b6bc1270b2bdde95e9c57e.jpg)

![](images/73f2ba5da230a69599d5410e84ff6561e877caf29d75c30c52ff55a2cd0b2140.jpg)  
Figure 5: Four additional held-out cases with ordered frames. (a) VARI preserves five salient robots and the group dance across both forms. (b) It maintains one coherent game layout and stable participants. (c) It moves the car past a stable church. (d) It realizes progressive snowman collapse and volume loss.

Base produces crowded groups with additional background robots, while VideoDPO commonly exposes only four dominant robots or changes the apparent count across frames and forms. VARI keeps five clearly identifiable robots in both prompt formulations and preserves the requested collective dancing behavior with stronger temporal and appearance consistency.

Multi-person interaction and object coherence. Panel (b) stresses scene-level logic rather than a single highlighted token. Base and VideoDPO frequently turn the singular board game into multiple disconnected, duplicated, or fragmented playing surfaces, weakening both object identity and the social interaction. VARI produces the most coherent table geometry and game layout, keeps the participants engaged with the same central activity, and yields cleaner faces, hands, and cafe structure across the sequence.

## Qualitative audit cues

The four appendix cases are read against explicit visual commitments: exact entity count and shared group action in panel (a); stable participants and one coherent game object in panel (b); object displacement relative to fixed landmarks in panel (c); and a sustained, monotonic state transition in panel (d). The temporally ordered frames support these checks without substituting for the complete clips used in the main-paper cases.

Motion realization and scene preservation. For panel (c) in Figure 5, the decisive requirement is object motion relative to fixed landmarks. The Base strip provides little evidence of the car translating through the scene and is dominated by viewpoint change. VideoDPO keeps a visible car but shows weaker frame-to-frame appearance, scale, and trajectory continuity. VARI presents a clear progression from a distant car to a close pass while retaining the old church as a stable spatial reference.

Temporal state change. Panel (d) in Figure 5 tests whether the model executes a sustained transformation rather than merely rendering the correct initial object. The Base snowman remains essentially intact, and VideoDPO shows at most a small local dripping cue without a substantial change of state. VARI produces a pronounced, monotonic melting sequence: the body loses volume, the structure collapses, and the final frame visibly difers from the initial snowman.