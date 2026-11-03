# VidDub: Replacement-Aware Distillation for Two-Step Video Dubbing

Yexin Liu<sup>1,2</sup>, Yunlong Lin<sup>2</sup>, Zehan Wang<sup>3</sup>, Guangxin He<sup>2</sup>, Xinli Xu<sup>2</sup>, Jiajun Zha<sup>1</sup>, Yaofu Liu<sup>1</sup> Yueze Wang<sup>4</sup>, Bowen Ping<sup>2</sup>, Yuanming Li<sup>2</sup>, Amie Wang<sup>5</sup>, Bernard Ghanem<sup>6</sup>, Bing Li<sup>6</sup>, Harry Yang<sup>1,∗</sup>

<sup>1</sup>HKUST <sup>2</sup>Tencent <sup>3</sup>ZJU <sup>4</sup>Independent Researcher <sup>5</sup>Tsinghua University <sup>6</sup>KAUST

GitHub: https://github.com/LAW1223/VidDub

<sup>∗</sup>Corresponding author

![](images/f1d43adcdef48e5abcf9a09e563784c8595fdcee3f98531258eda7a56986f969.jpg)  
Figure 1: Replacement-speech dubbing across diverse video domains. Frames show VidDub on captured, web, generated, and 60-second inputs, while the lower right compares inference speed on one H20 across representative methods.

## Abstract

Video dubbing aims to synthesize lip movements that match target speech while preserving the speaker identity, head pose, and background of a source video. Existing editors are predominantly trained via self-reconstruction under original paired audio, even though the source mouth becomes a conflicting target when driving speech is replaced. We introduce replacement-aware cross-audio supervision: paired-audio samples use full-frame reconstruction, whereas unpaired replacement-audio samples mask out source-mouth reconstruction losses and supervise the mouth solely through audio–visual synchronization while preserving unmasked facial regions and background. Because offline dubbing provides the complete source video and target speech beforehand, we present VidDub, a fullcontext diffusion editor distilled into two denoising steps (2 NFE). Our replacement-aware teacher first learns accurate

lip synchronization without regressing toward conflicting source mouth movements. Matched trajectory regression then transfers the teacher’s CFG-guided trajectories to a twostep student, and distribution matching distillation recovers visual quality after step compression. We will release the curated training dataset, VidDub-Bench, fixed paired- and cross-audio evaluation manifests, preprocessing outputs, and unified evaluation code. Across three training seeds, masking out conflicting source-mouth reconstruction improves cross-audio Sync-C by 0.833 ± 0.040; the final two-step model combines strong lip synchronization with superior visual quality, identity preservation, and visual speech recognition across four benchmarks.

## 1 Introduction

Video dubbing synthesizes lip movements that match replacement speech while preserving the speaker identity, head motion, and scene of a source video. Feed-forward and inpainting-based editors [5, 31, 57] and recent latent-space and diffusion-based editors [21, 28, 30, 54] have substantially improved visual fidelity, while few-step distillation reduces iterative sampling cost [24, 34, 37, 49, 50]. However, their primary training objective remains same-audio self-reconstruction: the model receives the original paired speech and is supervised to reproduce the mouth motion already present in the source video. This objective does not directly match the test-time task of dubbing, which requires replacing mouth articulation under new speech while leaving all other visual content unchanged.

The core challenge lies in the supervision conflict between same-audio reconstruction and cross-audio dubbing. Under original paired speech, the source mouth, identity, pose, and background all serve as consistent ground-truth targets. Under unpaired replacement speech, the source mouth articulates different content and directly conflicts with the target audio, whereas the unmasked face and background remain valid ground truth. Standard reconstruction losses do not distinguish between these two regimes, and few-step distillation cannot recover accurate cross-audio lip synchronization if the teacher is penalized for deviating from the source mouth. Moreover, evaluation protocols centered on short same-audio clips fail to expose this discrepancy. We therefore study how to train cross-audio lip synchronization without target conflict, distill the resulting model into two denoising steps, and evaluate it systematically across video domains and durations.

We propose VidDub, starting with replacement-aware cross-audio supervision. For paired same-audio samples, we apply standard full-frame reconstruction losses. For unpaired replacement-audio samples, we mask out flow-matching, pixel, and perceptual reconstruction losses inside the mouth region, preserve the unmasked face and background via spatial reconstruction, and supervise mouth articulation with an audio–visual synchronization loss. This decoupled formulation turns cross-audio dubbing into a well-posed local editing task: target speech drives lip articulation, while the source video anchors identity, head pose, and background.

We then distill the replacement-aware full-context teacher into a full-context two-step student. Matched same- and cross-audio trajectories transfer the teacher’s CFG-guided predictions while preserving bidirectional temporal attention across the video window, and guided distribution matching distillation restores visual realism after step compression [49]. This architecture aligns with offline video dubbing, where the complete source video and target speech are available prior to inference. A controlled same-backbone ablation in Table 11 demonstrates that block-causal conversion reduces first-frame latency for streaming, whereas full-context bidirectional attention achieves higher offline throughput, lower peak memory, and better visual speech recognition; VidDub therefore retains full-context bidirectional modeling.

We evaluate VidDub across diverse cross-audio and crossdomain benchmarks. VidDub-Bench (Sec. 4.2) provides 256 speaker-disjoint generated videos with fixed same- and crossaudio pairings, structured prompt metadata, preprocessing outputs, and unified evaluation code. We will also release the curated training clips developed in this work. Together with Long-20, independent visual speech recognition (VSR) [25], temporal offset calibration, non-mouth preservation metrics, and four evaluation benchmarks, our protocol jointly assesses lip-reading intelligibility, temporal synchronization, visual fidelity, and minute-long stability [3, 8, 56]. Across three training seeds, masking out conflicting source-mouth reconstruction yields $\mathbf { a } + 0 . 8 3 3 \pm 0 . 0 4 0$ cross-audio Sync-C improvement. Using only two denoising steps (2 NFE), Vid-Dub achieves state-of-the-art cross-audio synchronization while improving VSR accuracy, identity preservation, and video quality over its matched causal counterpart.

• Replacement-aware cross-audio supervision resolves the cross-audio target conflict by masking out sourcemouth reconstruction losses under unpaired replacement speech while preserving unmasked facial regions and target-audio synchronization constraints.

• Full-context two-step diffusion distillation transfers matched same- and cross-audio teacher trajectories into a bidirectional two-step student (2 NFE), followed by distribution matching distillation for improved visual quality while retaining full-context temporal attention.

• Open training dataset and evaluation benchmark contribute 85,435 curated training clips and VidDub-Bench with fixed same- and cross-audio pairings, unified evaluation scripts, and comprehensive VSR, temporal-alignment, preservation, perceptual, and long-video evaluations.

## 2 Related Work

Visual speech generation and editing. Source-video dubbing methods edit mouth motion in existing footage to match target speech [5, 14, 21, 28, 30, 31, 43, 54, 57, 61], personspecific neural rendering methods fit a subject from video [13, 29, 48], reference-conditioned portrait animation synthesizes head motion from an identity image, optionally with an additional pose source [4, 10, 12, 26, 36, 45–47, 53, 59], and audio-driven generators synthesize talking faces from speech without an identity reference [38]. Unlike standard self-reconstruction training that penalizes deviations from the source mouth, our replacement-aware objective masks out source-mouth reconstruction on unpaired audio clips during training.

Few-step video diffusion. Progressive, consistency, adversarial, and distribution-matching distillation methods accelerate diffusion sampling [24, 34, 35, 37, 49, 50]. Causal Forcing [58, 60] and Lip Forcing [6] combine autoregressive causal attention with few-step distillation for online streaming. In contrast, offline video dubbing provides the full source video and replacement audio prior to generation. We retain full-window bidirectional attention while distilling the CFG-guided teacher into a two-step student [16, 22].

![](images/7db66941a728793b21fb81713efacbeb9abd61ecdbf4e2ff479d4b4a5af74569.jpg)  
Figure 2: Full-context two-step dubbing architecture. Masked source frames and a reference image preserve speaker appearance and pose, target-audio tokens drive lip articulation, and bidirectional self-attention jointly denoises the complete latent window in two steps.

Dubbing benchmarks and evaluation metrics. HDTF, VoxCeleb2, and TalkVid-Bench provide high-resolution and in-the-wild speech domains [3, 8, 56]; we extend evaluation to generated footage with explicit motion analysis. In addition to SyncNet-based synchronization [7], we use an independently trained visual speech recognizer [25] to evaluate lip-reading intelligibility alongside pixel fidelity, identity similarity [11], head pose accuracy [18], FVD [42], and 60-second temporal stability.

## 3 Method

Overview. Given a source video and replacement speech, VidDub synthesizes synchronized lip movements while preserving speaker identity, head pose, and background. Our core insight is that ground-truth reconstruction targets must be decoupled by region when the driving speech is replaced. Figure 2 illustrates our bidirectional editing architecture, which conditions on masked source frames, a reference frame, and frame-aligned audio tokens. Figure 3 presents the three-stage training pipeline: (a) replacement-aware teacher training, (b) two-step guided trajectory regression, and (c) distribution matching distillation. Sections 3.1–3.3 detail each component.

## 3.1 Replacement-Aware Cross-Audio Supervision

In source-video dubbing, supervision cannot rely solely on spatial masks; it must account for whether the driving audio matches the source video. When original speech is replaced by unpaired audio, the source mouth motion conflicts with the target utterance, whereas the unmasked face and background remain valid ground truth.

Given source frames $V ^ { \mathrm { s r c } } = \{ I _ { t _ { \star } } ^ { \mathrm { s r c } } \} _ { t = 1 } ^ { T }$ and target waveform $A ^ { \mathrm { t g t } }$ , we aim to synthesize V<sup>ˆ</sup> with target-driven lip articulation while preserving all non-mouth visual content. The source mouth provides a consistent reconstruction target only when $A ^ { \mathrm { t g t } } = A ^ { \mathrm { s r c } }$

For clean video latent z and Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ flow matching [22] interpolates $x _ { \sigma } = ( 1 - \sigma ) z + \sigma \epsilon$ with velocity target $\boldsymbol { v } ^ { * } ~ = ~ \epsilon - ~ z$ and spatially weighted loss $\mathcal { L } _ { \mathrm { F M } } ^ { q } = \mathbb { E } [ q \| v _ { \theta } ( x _ { \sigma } , \sigma , c ) - v ^ { * } \| _ { 2 } ^ { 2 } ] / \mathbb { E } [ q ]$ . Max-pooling the VACE edit mask yields the latent mouth mask M. Paired same-audio samples use spatial weight $q ~ = ~ w ( M ) ~ =$ $1 + \lambda _ { M } M$ to emphasize mouth reconstruction, whereas unpaired replacement-audio samples use $q = \bar { M } = 1 - M ,$ masking out source-mouth velocity regression entirely. Note that z is encoded directly from the unedited source video $V ^ { \mathrm { s r c } }$ rather than from a synthetic pseudo-target. Its mouth region therefore reflects the original speech $A ^ { \mathrm { s r c } }$ ; when driven by replacement speech $A ^ { \mathrm { t g t } } \neq A ^ { \mathrm { s r c } }$ , M<sup>¯</sup> prevents the flowmatching loss from pulling generated lips back toward the conflicting source mouth.

This region-decoupled design yields distinct objectives for the two training regimes. Paired same-audio samples optimize full-frame reconstruction:

![](images/457f28872c67a0f7f4487453e85617cd8ec251ee174254fca344cdf75ab99bda.jpg)  
Figure 3: Three-stage training and distillation pipeline. Top: full-context teacher training via audio warm-up, joint editing adaptation, and replacement-aware refinement. (a) Replacement-aware training masks out source-mouth reconstruction under unpaired speech. (b) Guided trajectory regression distills the 48-step CFG-guided teacher into a two-step student. (c) Distribution matching distillation recovers visual realism in the final two-step model.

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { s a m e } } = { \mathcal { L } } _ { \mathrm { F M } } ^ { w ( M ) } + \lambda _ { \mathrm { p i x } } { \mathcal { L } } _ { \mathrm { p i x } } + \lambda _ { \mathrm { p e r c } } { \mathcal { L } } _ { \mathrm { L P I P S } } } \\ & { \phantom { { \mathcal { L } } } + \lambda _ { \mathrm { s y n c } } { \mathcal { L } } _ { \mathrm { s y n c } } + \lambda _ { \mathrm { k e e p } } { \mathcal { L } } _ { \mathrm { k e e p } } , } \end{array}\tag{1}
$$

For unpaired replacement audio $a ^ { \prime } .$ , all source-mouth reconstruction terms are masked out:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { c r o s s } } = \mathcal { L } _ { \mathrm { F M } } ^ { \bar { M } } ( z , a ^ { \prime } ) + \lambda _ { \mathrm { s y n c } } \mathcal { L } _ { \mathrm { s y n c } } ( G _ { \phi } ( z , a ^ { \prime } ) , a ^ { \prime } ) } \\ & { ~ + ~ \lambda _ { \mathrm { k e e p } } \mathcal { L } _ { \mathrm { k e e p } } ^ { \bar { M } } . } \end{array}\tag{2}
$$

On a decoded 16-frame window, $\mathcal { L } _ { \mathrm { p i x } }$ is mouth-region $L _ { 1 }$ loss, $\mathcal { L } _ { \mathrm { L P I P S } } \left[ 5 1 \right]$ is perceptual loss over four sampled mouth crops, and $\mathcal { L } _ { \mathrm { k e e p } } ^ { \bar { M } } \ = \ \| \bar { M } \odot ( D ( z ) - V ^ { \mathrm { s r c } } ) \| _ { 1 } / \| \bar { M } \| _ { 1 }$ enforces pixel preservation outside the mouth mask. Consequently, under replacement audio, no flow, pixel, or perceptual loss regresses the mouth toward $V ^ { \mathrm { s r c } }$ ; masked source frames provide appearance context while target audio governs lip synchronization.

Lip synchronization is supervised by a frozen LatentSync-1.6 StableSyncNet [21], which extracts embeddings from 16 lower-face frames and the aligned 16-kHz mel-spectrogram via cosine-log loss $\mathcal { L } _ { \mathrm { s y n c } } = - \log [ \operatorname* { m a x } ( \cos ( e _ { v } , e _ { a } ) , 1 0 ^ { - 6 } ) ]$ without temporal lag search. Gradients backpropagate through the frozen synchronizer and VAE decoder into the generator. Test-time evaluation instead uses the standard S3FD [52]/SyncNet-v2 [7, 8] benchmark pipeline, complemented by an independent LRS3 VSR reader in Sec. 4.6.

## 3.2 Full-Context Diffusion Architecture

While replacement-aware supervision defines the training objective, our architecture retains bidirectional self-attention across the entire latent window so that every generated frame attends to full temporal context. This design matches offline video dubbing, where the complete source video and target audio are available before inference.

Given masked source latents u, reference latent r, targetaudio embeddings a, mouth mask M, and initial noise ϵ, the two-step generator predicts the edited latent window in two denoising steps:

$$
\hat { z } = G _ { \phi } ^ { ( 2 ) } ( \epsilon \mid u , r , a , M ) .\tag{3}
$$

We build upon Wan-1.3B with VACE context conditioning [19, 40] and a 287M audio adapter over frozen Whispersmall features [33]. VACE encodes the masked source video and reference image into visual context tokens. The audio adapter maps 16 local Whisper tokens per latent frame into speech conditioning tokens, and bidirectional DiT selfattention jointly processes all 21 video latents within each 81-frame window. The two-step student shares the exact network architecture of the teacher, compressing only the sampling schedule from 48 steps to 2 steps while absorbing classifier-free guidance (CFG) into a single forward pass.

## 3.3 Trajectory and Distribution Distillation

Our pipeline proceeds in three stages: learning cross-audio lip synchronization in the teacher, compressing multi-step sampling into two steps via trajectory regression, and restoring visual realism via distribution matching distillation.

(a) Replacement-aware teacher training. Following audio warm-up and joint editing adaptation, teacher refinement mixes 70% paired same-audio clips with 30% unpaired replacement-audio clips, where replacement audio is drawn from the preceding dataloader sample on each rank. All training clips use 81-frame (3.375 s) windows aligned from the start of the waveform. Non-mouth reconstruction preserves identity, pose, and background, while $\mathcal { L } _ { \mathrm { s y n c } }$ drives lip movements to match the replacement audio.

(b) Guided trajectory regression. Matched full-context trajectories compress the 48-step teacher into a student with two denoising updates. The teacher uses classifier-free audio guidance $v ^ { ( g ) } = v ( x , \emptyset ) + g [ v ( x , a ) - v ( x , \emptyset ) ]$ . Source, audio, noise, boundary state, and teacher target are matched within the same full-window trajectory. For boundary state $x _ { \tau }$ , guided-teacher terminal latent $\tilde { z } _ { T , g }$ , and student twoupdate map $\Psi _ { \phi } ^ { ( 2 ) }$ , we optimize

$$
\mathcal { L } _ { \mathrm { t r a j } } = \mathbb { E } \biggl [ \left\| \Psi _ { \phi } ^ { ( 2 ) } ( x _ { \tau } , c ) - \mathrm { s g } ( \tilde { z } _ { T , g } ) \right\| _ { w ( M ) } ^ { 2 } \biggr ] + \lambda _ { q } ^ { \mathrm { t r a j } } \mathcal { L } _ { q } ^ { s / c } ,\tag{4}
$$

where

$$
\begin{array} { r } { \mathcal { L } _ { q } ^ { s } = \lambda _ { \mathrm { p i x } } ^ { \mathrm { t r a j } } \mathcal { L } _ { \mathrm { p i x } } + \lambda _ { \mathrm { p e r c } } ^ { \mathrm { t r a j } } \mathcal { L } _ { \mathrm { L P I P S } } + \lambda _ { \mathrm { s y n c } } ^ { \mathrm { t r a j } } \mathcal { L } _ { \mathrm { s y n c } } + \lambda _ { \mathrm { k e e p } } ^ { \mathrm { t r a j } } \mathcal { L } _ { \mathrm { k e e p } } , } \end{array}\tag{5}
$$

$$
\begin{array} { r } { \mathcal { L } _ { q } ^ { c } = \lambda _ { \mathrm { s y n c } } ^ { \mathrm { t r a j } } \mathcal { L } _ { \mathrm { s y n c } } + \lambda _ { \mathrm { k e e p } } ^ { \mathrm { t r a j } } \mathcal { L } _ { \mathrm { k e e p } } ^ { \bar { M } } . } \end{array}\tag{6}
$$

Here $\mathcal { L } _ { q } ^ { s / c }$ selects $\mathcal { L } _ { q } ^ { s }$ for same audio and $\mathcal { L } _ { q } ^ { c }$ for replacement audio. The frozen set contains 4,096 trajectories, evenly divided between paired- and cross-audio conditions; replacement utterances share the source domain and language but come from different source videos. The regression target is the teacher output conditioned on replacement speech rather than the original source mouth. Matching all trajectory conditions distills classifier-free guidance into a student that requires one conditional forward pass per denoising step, while sg blocks teacher gradients.

(c) Guided distribution matching. Trajectory regression matches individual teacher endpoints but does not directly match the distribution of generated videos. We therefore apply guided distribution matching after trajectory distillation. At noise level σ, the frozen guided real score and online fake score define $\hat { z } _ { r } = x _ { \sigma } - \sigma v _ { \eta }$ and $\hat { z } _ { f } = x _ { \sigma } - \sigma v _ { f }$ . Applied to the complete latent window, DMD2 [49] uses

$$
\begin{array} { r } { d = ( \hat { z } _ { f } - \hat { z } _ { r } ) / \operatorname* { m a x } ( \operatorname* { m e a n } | z - \hat { z } _ { r } | , 1 0 ^ { - 6 } ) , } \\ { \mathcal { L } _ { \mathrm { D M D } } = \frac { 1 } { 2 } \| z - \mathrm { s g } ( z - d ) \| _ { w ( M ) } ^ { 2 } + \lambda _ { \mathrm { { r e t } } } \mathcal { L } _ { \mathrm { { r e t } } } } \\ { + \lambda _ { q } ^ { \mathrm { D M D } } \mathcal { L } _ { q } ^ { \mathrm { D M D , s / c } } . \quad \quad } \end{array}\tag{7}
$$

d is stop-gradient. $\mathcal { L } _ { \mathrm { r e t } }$ is same-noise, same-schedule w(M)- MSE to the frozen initializer. The auxiliary DMD ob-

jectives are

$$
\mathcal { L } _ { q } ^ { \mathrm { D M D , s } } = \lambda _ { \mathrm { p i x } } ^ { \mathrm { D M D } } \mathcal { L } _ { \mathrm { p i x } } + \lambda _ { \mathrm { p e r c } } ^ { \mathrm { D M D } } \mathcal { L } _ { \mathrm { L P I P S } } + \lambda _ { \mathrm { k e e p } } ^ { \mathrm { D M D } } \mathcal { L } _ { \mathrm { k e e p } } ,\tag{8}
$$

$$
\mathcal { L } _ { q } ^ { \mathrm { D M D , c } } = \lambda _ { \mathrm { k e e p } } ^ { \mathrm { D M D } } \mathcal { L } _ { \mathrm { k e e p } } ^ { \bar { M } } .\tag{9}
$$

Likewise, $\mathcal { L } _ { q } ^ { \mathrm { D M D , s / c } }$ selects the objective associated with the current audio condition. The fake score takes five updates per generator update; DMD uses 50% replacement.

## 3.4 Long-Video Inference

Long videos are processed using overlapping 81-frame windows conditioned on the corresponding source frames. Each window uses two denoising steps; consecutive windows advance by 64 frames and blend the 17-frame overlap. This sliding-window scheme bounds computation and memory while retaining bidirectional context within each window.

Relation to causal few-step methods. Lip Forcing [6] represents the causal two-step design paradigm, and Causa Forcing develops causal ODE/DMD objectives [58, 60]. Our controlled same-backbone comparison shows that causal attention reduces first-frame latency but does not improve offline throughput or output quality. VidDub therefore combines replacement-aware cross-audio supervision with direct full-context distillation.

## 4 Experiments

## 4.1 Experimental Setup

The evaluation follows the contribution order. Tables 4 and 5 evaluate replacement-aware cross-audio supervision; Table 6 evaluates trajectory and distribution distillation; and Tables 1– 10 test the final model across source domains and video lengths. Tables 1, 2, and 3 evaluate HDTF-30, TalkVid-Bench-500, and VidDub-Bench-256 using standardized 81- frame, 24 FPS, 512×512 model calls; full VoxCeleb2-30 results and native source protocols are supplementary. Table 10 evaluates 20 fixed 60-second pairs. Model selection uses a disjoint dev-256 split; exclusion groups are detailed in the supplementary material.

The final student performs two denoising steps per fullcontext window. Cross-method rankings use the clips successfully evaluated for every method, while method-specific tables report generation coverage separately. Unless noted, uncertainty is estimated with paired source-level bootstrap intervals. Synchronization, reconstruction, distribution quality, visual speech recognition, source preservation, and blinded human judgments are reported in Tables 1–10.

Same-audio reconstruction cannot directly determine whether generated articulation follows replacement speech. We therefore report cross-audio synchronization across four complementary source domains: high-resolution HDTF [56], in-the-wild VoxCeleb2 [8], web-video TalkVid-Bench [3], and the source-disjoint generated VidDub-Bench. Ranking correlations and decoded source-motion statistics in Supplementary Sec. B further show that reconstruction metrics on a single captured domain do not substitute for direct cross-audio and cross-source evaluation.

![](images/c9bf1cae1123f58d3b5fba05c4f631d49d110c7e9b44414d7c95e27969c86195.jpg)  
Figure 4: Cross-audio Sync-C versus FVD on four benchmarks. Dashed lines show real-video same-audio Sync-C.

Table 1: HDTF comparison. Params counts all neural modules loaded for inference; FPS uses one H20. <sup>†</sup>Preparedinput throughput is reported for our diffusion models.
<table><tr><td></td><td colspan="3"></td><td colspan="4">Reconstruction</td><td colspan="3">Replacement Dubbing</td></tr><tr><td>Method</td><td>Params NFE</td><td>FPS↑</td><td>Sync-C↑</td><td></td><td>Sync-D↓</td><td>SSIM↑</td><td>PSNR↑</td><td>Sync-C↑</td><td>Sync-D↓</td><td>FVD↓</td></tr><tr><td>Unedited source</td><td>一</td><td>0</td><td>一</td><td>8.723</td><td>6.666</td><td></td><td></td><td>1.283</td><td>14.483</td><td></td></tr><tr><td colspan="9">Feed-forward and one-step editors</td><td></td><td></td></tr><tr><td>Wav2Lip [31]</td><td>36M</td><td>1</td><td>5.2</td><td>9.177</td><td>6.555</td><td>0.9233</td><td>28.97</td><td>8.782</td><td>6.972</td><td>129.51</td></tr><tr><td>VideoReTalking [5]</td><td>484M 971M</td><td>11</td><td>1.3 1.1</td><td>8.801 8.471</td><td>6.593 6.895</td><td>0.8981 0.9489</td><td>27.85 33.74</td><td>8.312 7.116</td><td>6.988 7.880</td><td>105.72</td></tr><tr><td colspan="9">MuseTalk 1.5 [54]</td><td>40.10</td></tr><tr><td>Bidirectional diffusion editors (20–50 NFE)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Diff2Lip [28]</td><td>102M</td><td>25</td><td>2.0</td><td>9.670</td><td>5.535</td><td>0.9182</td><td>30.46</td><td>8.680</td><td>6.310</td><td>99.70</td></tr><tr><td>LatentSync 1.6 [21] X-Dub [14]</td><td>1.39B</td><td>20</td><td>2.2</td><td>9.363</td><td>6.029</td><td>0.9656</td><td>32.95</td><td>8.160</td><td>7.035</td><td>28.05</td></tr><tr><td>Full-context Teacher-30 (ours)</td><td>14.32B 2.81B</td><td>50 30 1.2†</td><td>0.4</td><td>7.792 9.760</td><td>7.810 5.900</td><td>0.8364 0.9279</td><td>26.62 30.77</td><td>5.027 9.422</td><td>10.271 6.127</td><td>66.05 27.49</td></tr><tr><td colspan="9"></td><td></td></tr><tr><td>Two-step diffusion editors Lip Forcing [6]</td><td>14.53B</td><td>2</td><td>2.4</td><td>8.429</td><td></td><td>6.9190.9517</td><td>31.37</td><td>6.884</td><td>8.284</td><td>33.80</td></tr><tr><td>VidDub</td><td>2.81B</td><td>2 11.3†</td><td></td><td>10.371</td><td></td><td>5.654 0.9213</td><td>30.03</td><td>10.102</td><td>5.856</td><td>27.66</td></tr></table>

Table 2: TalkVid-Bench comparison on 500 requested clips. “Scored S/C” denotes successfully evaluated paired-/cross-audio outputs; rankings use the 308 clips valid for every method. FPS and NFE follow Table 1.
<table><tr><td></td><td colspan="5"></td><td colspan="3">Reconstruction</td><td colspan="3">Replacement Dubbing</td></tr><tr><td>Method</td><td>Params NFE Scored S/C FPS↑ Sync-C↑ Sync-D↓ SSIM↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td>PSNR↑</td><td>Sync-C↑ Sync-D↓</td><td></td><td>FVD↓</td></tr><tr><td>Feed-forward and one-step editors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wav2Lip [31]</td><td>36M</td><td>1</td><td>499/499</td><td>5.2</td><td>7.209</td><td>7.420</td><td>0.9218</td><td>29.83</td><td>6.750</td><td>7.932</td><td>111.19</td></tr><tr><td>VideoReTalking [5]</td><td>484M</td><td>1</td><td>402/402</td><td>1.3</td><td>6.787</td><td>7.552</td><td>0.8948</td><td>28.00</td><td>6.365</td><td>7.868</td><td>84.51</td></tr><tr><td>MuseTalk 1.5 [54]</td><td>971M</td><td>1</td><td>500/499</td><td>1.1</td><td>4.968</td><td>9.106</td><td>0.9305</td><td>29.60</td><td>4.371</td><td>9.625</td><td>36.09</td></tr><tr><td>Bidirectional diffusion editors (20–50 NFE)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Diff2Lip [28]</td><td>102M</td><td>25</td><td>425/403</td><td>2.0</td><td>7.775</td><td>6.383 0.9157</td><td></td><td>30.60</td><td>6.944</td><td>7.016</td><td>73.37</td></tr><tr><td>LatentSync 1.6 [21]</td><td>1.39B</td><td>20</td><td>499/499</td><td>2.2</td><td>7.516</td><td>6.965</td><td>0.8949</td><td>31.09</td><td>6.860 5.087</td><td>7.512</td><td>59.27 32.63</td></tr><tr><td>X-Dub [14] Full-context Teacher-48 (ours)</td><td>14.32B 2.81B</td><td>50 48</td><td>500/500 500/500</td><td>0.4 1.2†</td><td>5.492 7.741</td><td>9.129</td><td>0.6980 0.9125</td><td>21.86 28.84</td><td>7.663</td><td>9.437 7.266</td><td>8.51</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>7.170</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Two-step diffusion editors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Lip Forcing [6]</td><td>14.53B</td><td>2</td><td>499/499</td><td>2.4</td><td>5.914</td><td>8.283 0.7698</td><td></td><td>24.48</td><td>5.610</td><td>8.648</td><td>12.68</td></tr><tr><td>VidDub</td><td>2.81B</td><td>2</td><td>500/500</td><td>11.3†</td><td>7.740</td><td>7.216 0.9125</td><td></td><td>28.78</td><td>7.646</td><td>7.309</td><td>10.25</td></tr></table>

## 4.2 Training Data and Benchmark Construction

Motivation and structured prompt generation. Existing talking-face datasets do not jointly cover the demographic, linguistic, expressive-motion, and scene diversity required by our dubbing setting [3, 8, 55, 56]. To expand this coverage without identity-privacy constraints, we construct a taxonomy-driven prompt suite of 100,000 talking-human descriptions using Qwen3.5-397B-A17B [32] (48,303 Chinese and 51,697 English). Each prompt combines verbatim speech transcripts with stratified attributes spanning ethnicity and cultural region, age group, gender presentation, 42 emotion dynamics, 73 upper-body gestures, and 38 scene settings, and is rendered into candidate talking-human videos via MiniMax-H3 inference [27].

Table 3: VidDub-Bench comparison on each method’s successfully evaluated outputs. FPS and NFE follow Table 1 throughout.
<table><tr><td></td><td></td><td></td><td></td><td colspan="3">Reconstruction</td><td colspan="3">Replacement Dubbing</td></tr><tr><td>Method</td><td>Params NFE FPS↑</td><td></td><td></td><td>Sync-C↑ Sync-D↓ SSIM↑</td><td></td><td>PSNR↑</td><td></td><td>Sync-C↑ Sync-D↓ FVD↓</td><td></td></tr><tr><td>Feed-forward and one-step editors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wav2Lip</td><td>36M</td><td>1</td><td>5.2 8.269</td><td></td><td>7.193 0.9063</td><td>29.44</td><td>6.922</td><td>8.536</td><td>98.62</td></tr><tr><td>VideoReTalking</td><td>484M</td><td>1 1.3 1</td><td>8.097</td><td></td><td>7.088 0.8773</td><td>26.77</td><td></td><td>6.873 8.069</td><td>65.43</td></tr><tr><td>MuseTalk 1.5</td><td>971M</td><td>1.1</td><td>6.430</td><td></td><td>8.673 0.9078</td><td>28.27</td><td>4.184</td><td>10.670</td><td>56.73</td></tr><tr><td>Bidirectional diffusion editors (20–50 NFE)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Diff2Lip</td><td>102M</td><td>25</td><td>2.0 9.724</td><td></td><td>5.643 0.9125</td><td>30.36</td><td></td><td>7.575 7.309</td><td>58.87</td></tr><tr><td>LatentSync 1.6</td><td>1.39B</td><td>20</td><td>2.2 9.389</td><td></td><td>6.269 0.9099</td><td>32.05</td><td>7.544</td><td>7.730</td><td>46.77</td></tr><tr><td>X-Dub public</td><td>14.32B</td><td>50</td><td>0.4 7.576</td><td></td><td>7.912 0.6954</td><td>22.07</td><td>6.234</td><td>9.071</td><td>38.17</td></tr><tr><td>Full-context Teacher-48 (ours)</td><td>2.81B</td><td>48 1.2†</td><td>8.966</td><td></td><td>6.889 0.9141</td><td>29.22</td><td>8.718</td><td>7.093</td><td>15.23</td></tr><tr><td>Two-step diffusion editors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Lip Forcing</td><td>14.53B</td><td>2</td><td>2.4</td><td>6.672</td><td>8.475 0.7820</td><td>25.36</td><td>5.192</td><td>9.971</td><td>15.78</td></tr><tr><td>VidDub</td><td>2.81B</td><td>2 11.3†</td><td></td><td>8.993</td><td>6.905 0.9123</td><td>29.04</td><td>8.709</td><td>7.140</td><td>17.50</td></tr></table>

Filtering and audio–visual alignment. Generated and inthe-wild candidates still exhibit face-tracking drift, extreme poses, and audio–visual timing offsets. We track the dominant face, extract stable 512×512 square crops with union lower-face edit masks, and discard windows with missing or unstable tracks, invalid crop geometry, extreme head pose, or weak mouth motion. Following the synchronization-based alignment used by LatentSync [21], we apply SyncNet [7] to estimate audio–visual confidence and temporal offsets, rejecting low-correspondence clips and shifting retained waveforms by the estimated offset before extracting 81-frame windows at 24 FPS. This pipeline yields 85,435 model-ready H3 training clips alongside filtered public clips.

VidDub-Bench construction. From held-out H3 source videos strictly disjoint from the training and development splits, we freeze 256 balanced source videos (128 Chinese, 128 English) to form VidDub-Bench. Each source video is paired with its original audio track for reconstruction and a fixed replacement waveform drawn from a different source video for dubbing evaluation, while source frames, face crops, spatial masks, reference frames, random seeds, and durations remain identical across conditions (Table 3).

## 4.3 Comparison with Prior Methods

Captured and web sources. Figure 4 and Tables 1 and 2 compare the final model on controlled, in-the-wild, and web video. With two denoising steps, VidDub combines strong cross-audio synchronization with lower FVD than the matched causal model on HDTF, VoxCeleb2, and TalkVid. It processes all 500 requested TalkVid clips. Figure 4 jointly reports synchronization accuracy and FVD.

Table 4: Effect of cross-audio supervision on the teacher. Cross-audio Sync-C/FVD for teachers trained with paired-audio reconstruction only versus replacementaware cross-audio supervision.
<table><tr><td>Benchmark</td><td></td><td>Paired-audio only Replacement-aware</td></tr><tr><td>HDTF-30</td><td>7.986 / 26.40</td><td>9.422 / 27.49</td></tr><tr><td>TalkVid-Bench-50</td><td>6.460 / 26.14</td><td>7.414 /26.51</td></tr><tr><td>H3-Val-50</td><td>6.952 / 37.68</td><td>8.086 / 38.09</td></tr><tr><td>VoxCeleb2-30</td><td>5.729 / 39.95</td><td>7.382 / 39.11</td></tr></table>

Table 5: Three-seed ablation on dev-256. Mean effects±SD isolate source-mouth reconstruction masking from matched audio dropout and timestep range.
<table><tr><td>Factor</td><td>∆Sync-C↑</td><td>∆Sync-D↓</td><td>∆FVD↓</td></tr><tr><td>Full objective — paired only</td><td> $+ 1 . 2 3 9 { \scriptstyle \pm . 0 1 2 }$ </td><td> $- 0 . 9 5 1 { \scriptstyle \pm . 0 1 1 }$ </td><td> $+ 0 . 6 6 4 { \scriptstyle \pm . 0 9 2 }$ </td></tr><tr><td>Source-mouth mask (M)</td><td>+0.833±.040 −0.619±.037 -</td><td></td><td> $- 0 . 2 3 0 { \scriptstyle \pm . 2 5 9 }$ </td></tr><tr><td>Audio dropout (D)</td><td></td><td></td><td>−0.019±.009 +0.012±.008 −0.044±.064</td></tr><tr><td>Timestep range (T)</td><td></td><td></td><td>−0.206±.053 +0.147±.041 −0.101±.056</td></tr></table>

Generated sources. Table 3 evaluates source-disjoint generated videos using a 245-clip subset successfully evaluated by every method. The two-step student matches the full-context teacher’s cross-audio Sync-C and exceeds the strongest external baseline. The accompanying FVD and valid-output counts show strong distribution quality and complete generation coverage.

## 4.4 Replacement-Aware Cross-Audio Supervision

Controlled ablation. Table 4 first compares teachers trained with and without the replacement-aware branch under matched initialization and data. Every paired item across the four benchmarks improves in cross-audio Sync-C, with paired 95% source-bootstrap intervals lying fully above zero on all four sets. Table 5 then reports the three-seed factorial that separates source-mouth reconstruction masking (M), matched audio dropout (D), and timestep policy (T). Reconstruction masking produces the dominant main effect, while the small interactions show that the gain follows directly from removing conflicting mouth supervision rather than from auxiliary sampling choices. Independent visualonly recognition on the same outputs yields consistent WER and CER improvements, linking the synchronization gain to more accurate articulation of the replacement speech.

Table 6: Two-step ablation on dev-256. Rows vary DMD guidance, preservation losses, and replacementaudio supervision.
<table><tr><td>Variant</td><td>SSIM↑ Sync-C↑ FVD↓</td><td></td></tr><tr><td>Trajectory only (no DMD)</td><td>0.9369</td><td>8.594 10.84</td></tr><tr><td>DMD, unguided real score</td><td>0.9402</td><td>8.517 7.00</td></tr><tr><td>DMD, no preservation loss</td><td>0.9381</td><td>8.733 7.69</td></tr><tr><td>Complete DMD recipe</td><td>0.9382</td><td>8.731 7.76</td></tr><tr><td>w/o replacement audio</td><td>0.9398</td><td>8.577 7.15</td></tr></table>

## 4.5 Trajectory and Distribution Distillation

Trajectory regression and distribution matching. Matched same- and cross-audio trajectories transfer the full-context teacher predictions without changing its bidirectional temporal attention, after which distribution matching aligns the student’s generated distribution with real videos. Table 6 evaluates real-score guidance, preservation losses, and replacement-audio supervision; the complete stage-wise path is provided in the supplementary material. Guidance and replacement-audio dependence. Matched guidance controls show that both trajectory and DMD guidance strengthen synchronization. Replacing the input speech with fixed or temporally shifted audio reduces Sync-C, confirming that mouth motion follows the supplied speech rather than generic articulation or source-video cues.

## 4.6 Speech Validation and Source Preservation

Temporal-shift calibration. We shift the audio of 40 VidDub-Bench outputs per method by known frame offsets. The estimated SyncNet offset follows the imposed shift exactly within the calibrated range, and VidDub returns the correct zero offset on all unshifted clips. This intervention verifies temporal sensitivity, complementing the content-level evaluation in Table 7.

Independent visual speech recognition. Table 7 uses an LRS3 visual-only reader [25], independent of the training synchronizer, against shared Whisper pseudo-references [33]. VidDub achieves the lowest WER among evaluated external methods on all three test sources, and every paired external comparison favors VidDub. The agreement between independent VSR and cross-audio Sync-C shows that replacement-aware training improves recognizable targetspeech content rather than a single synchronization score. Identity and non-mouth preservation. Table 8 compares corresponding source frames, excluding the editable lower face for Upper-L1; identity and pose use ArcFace [11] and InsightFace [18]. VidDub combines the cross-audio control with non-mouth change comparable to the full-context teacher. These results show that the synchronization gain remains concentrated in the editable region while the source identity, pose, and surrounding visual content are retained.

Table 7: Independent visual-only WER (↓) against shared Whisper pseudo-references. The LRS3 reader is independent; HDTF/VoxCeleb2/VDB contain 30/28/121 English clips in total.
<table><tr><td>Method</td><td>HDTF Vox2 VDB</td></tr><tr><td>Real (same audio)</td><td>0.171 0.579 0.711</td></tr><tr><td>Wav2Lip [31]</td><td>1.093 1.009 1.020</td></tr><tr><td>VideoReTalking [5]</td><td>0.915 1.056 0.975</td></tr><tr><td>Diff2Lip [28]</td><td>0.957 0.963 0.990</td></tr><tr><td>MuseTalk [54]</td><td>0.987 1.068 1.014</td></tr><tr><td>LatentSync [21]</td><td>1.0191.049 0.953</td></tr><tr><td>X-Dub [14]</td><td>0.952 0.949 0.949</td></tr><tr><td>Lip Forcing [6]</td><td>1.093 0.9770.977</td></tr><tr><td>Teacher (ours)</td><td>0.632 0.825 0.681</td></tr><tr><td>VidDub</td><td>0.503 0.752 0.758</td></tr></table>

Table 8: Identity and non-mouth preservation on VidDub-Bench. Upper-L1 is percentage change; n is the number of valid outputs.
<table><tr><td>Method</td><td>Sync-C↑ Upper-L1↓</td><td></td><td>ID↑</td><td>Pose↓</td><td>n</td></tr><tr><td>Wav2Lip</td><td>6.922</td><td>1.820</td><td>0.909</td><td>0.459</td><td>256</td></tr><tr><td>Diff2Lip</td><td>7.575</td><td></td><td>1.283 0.862</td><td>0.518</td><td>256</td></tr><tr><td>LatentSync</td><td>7.544</td><td></td><td>1.605 0.913 0.422 254</td><td></td><td></td></tr><tr><td>Lip Forčing</td><td>5.192</td><td></td><td>3.761 0.900 0.928 256</td><td></td><td></td></tr><tr><td>Full-context teacher</td><td>8.718</td><td></td><td>1.207 0.872 0.573 256</td><td></td><td></td></tr><tr><td>VidDub</td><td>8.709</td><td></td><td>1.199 0.8750.579256</td><td></td><td></td></tr></table>

Table 9: Human evaluation of full-context VidDub. 15 raters independently evaluated all 150 trials. Entries are VidDub/Opp./Tie.
<table><tr><td>Opponent</td><td>Sync</td><td>Real.</td><td>Identity</td><td>Stability</td><td>Overall</td></tr><tr><td>Teacher</td><td>10/10/130</td><td>9/12/129</td><td>4/8/138</td><td>4/14/132</td><td>12/20/118</td></tr><tr><td>LatentSync</td><td>123/2/25</td><td>122/6/22</td><td>75/3/72</td><td>123/2/25</td><td>143/2/5</td></tr><tr><td>Lip Forcing</td><td>142/0/8</td><td>90/15/45</td><td>61/2/87</td><td>87/2/61</td><td>146/1/3</td></tr></table>

## 4.7 Human Evaluation

Table 9 reports the blinded study of full-context VidDub. Judgments favor VidDub over Lip Forcing and LatentSync, while ties dominate the teacher.

Table 10: Long-20 comparison. Twenty 60 s pairs use overlapping windows conditioned on source frames. Last-C is final-5 s Sync-C; ∆Keep is the final-minusinitial preservation change.
<table><tr><td>Method</td><td>Valid S-C↑ S-D↓ Last-C↑ Keep-L1↓ Chroma↓</td><td></td><td></td><td></td><td>∆Keep</td></tr><tr><td>Wav2Lip [31]</td><td>20/207.8617.484</td><td>8.032</td><td>3.045</td><td></td><td>1.045 +0.009</td></tr><tr><td>LatentSync [21]</td><td>20/20 7.771 7.083</td><td>7.844</td><td>4.018</td><td></td><td>0.397 +0.685</td></tr><tr><td>VidDub</td><td>20/20 9.399 6.162</td><td>9.585</td><td>2.493</td><td></td><td>0.250-0.034</td></tr></table>

Table 11: Full-context versus causal execution. Samebackbone two-step comparison on one H20. First-Out is first-decodable-output latency; VSR wins count the six same/cross comparisons across HDTF, VoxCeleb2, and VidDub-Bench.
<table><tr><td>Execution Design</td><td>First-Out↓</td><td>FPS↑</td><td>Peak GiB↓</td><td>VSR Wins↑</td><td>ID↑</td></tr><tr><td>Causal two-step</td><td>1.052 s</td><td>9.4</td><td>21.96</td><td></td><td>0/6 0.861</td></tr><tr><td>Full-context (VidDub)</td><td>7.165 s</td><td>11.3</td><td>13.77</td><td></td><td>6/6 0.885</td></tr></table>

## 4.8 Long-Video Evaluation and Efficiency

Table 10 scores every 5 s window and the final 5 s segment. VidDub maintains the strongest mean and final synchronization across all 20 sequences while achieving the lowest non-mouth and color changes. The stable first-to-last preservation measurement shows that overlapping-window two-step inference maintains replacement-speech synchronization and source appearance over 60 seconds. Table 11 compares temporal attention designs under the same backbone, two-step budget, training data, and evaluation protocol. Causal execution reduces first-output latency to 1.052 s by emitting 9-frame blocks progressively, which is advantageous for online streaming. For offline dubbing, where the complete source video and replacement speech are available beforehand, full-context execution processes an 81-frame window faster (11.3 vs. 9.4 FPS), reduces peak memory by 37% (13.77 vs. 21.96 GiB), wins all six independent VSR comparisons, and improves identity preservation. Combined with lower FVD on three of four benchmarks at closely matched cross-audio Sync-C (Tables 1–3), these results directly motivate our full-context two-step design.

## 5 Conclusion

We presented VidDub, a full-context two-step dubbing model built on replacement-aware cross-audio supervision. For unpaired replacement speech, it masks source-mouth reconstruction while retaining non-mouth reconstruction and target-audio synchronization. Full-context trajectory regression transfers the teacher to two denoising steps, and distribution matching improves visual quality. Across four source domains, VidDub combines accurate replacement-speech articulation with identity preservation and stable minute-long generation.

## References

[1] ByteDance. Latentsync. https://github.com/ bytedance/LatentSync, 2026. Official software repository; accessed 2026-11-02.

[2] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017.

[3] Shunian Chen, Hejin Huang, Yexin Liu, Zihan Ye, Pengcheng Chen, Chenghao Zhu, Michael Guan, Rongsheng Wang, Junying Chen, Jianye Hou, Bo Li, Guanbin Li, Ser-Nam Lim, Harry Yang, and Benyou Wang. Talkvid: A large-scale diversified dataset for audio-driven talking head synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Findings, pages 3492–3500, 2026.

[4] Zhiyuan Chen, Jiajiong Cao, Zhiquan Chen, Yuming Li, and Chenguang Ma. Echomimic: Lifelike audio-driven portrait animations through editable landmark conditions. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2403–2410, 2025.

[5] Kun Cheng, Xiaodong Cun, Yong Zhang, Menghan Xia, Fei Yin, Mingrui Zhu, Xuan Wang, Jue Wang, and Nannan Wang. Videoretalking: Audio-based lip synchronization for talking head video editing in the wild. In ACM SIGGRAPH Asia Conference Proceedings, 2022.

[6] Paul Hyunbin Cho, Jinhyuk Jang, SeokYoung Lee, Joungbin Lee, Siyoon Jin, Heeseong Shin, Jung Yi, Yunjin Park, Chulmin Park, and Seungryong Kim. Lip forcing: Few-step autoregressive diffusion for real-time lip synchronization. arXiv preprint arXiv:2606.11180, 2026.

[7] Joon Son Chung and Andrew Zisserman. Out of time: Automated lip sync in the wild. In Asian Conference on Computer Vision Workshops, pages 251–263, 2017.

[8] Joon Son Chung, Arsha Nagrani, and Andrew Zisserman. Voxceleb2: Deep speaker recognition. In Interspeech, pages 1086–1090, 2018.

[9] Commission Internationale de l’Eclairage. Colorimetry. Number 15 in CIE Technical Report. CIE Central Bureau, 3 edition, 2004.

[10] Jiahao Cui, Hui Li, Yun Zhan, Hanlin Shang, Kaihui Cheng, Yuqi Ma, Shan Mu, Hang Zhou, Jingdong Wang, and Siyu Zhu. Hallo3: Highly dynamic and realistic portrait image animation with video diffusion transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21086–21095, 2025.

[11] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

[12] Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, and Steven Hoi. Omniavatar: Efficient audio-driven avatar video generation with adaptive body animation. arXiv preprint arXiv:2506.18866, 2025.

[13] Yudong Guo, Keyu Chen, Sen Liang, Yong-Jin Liu, Hujun Bao, and Juyong Zhang. Ad-nerf: Audio driven neural radiance fields for talking head synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.

[14] Xu He, Haoxian Zhang, Hejia Chen, Changyuan Zheng, Liyang Chen, Songlin Tang, Jiehui Huang, Xiaoqiang Liu, Pengfei Wan, and Zhiyong Wu. From inpainting to editing: Unlocking robust mask-free visual dubbing via generative bootstrapping. In Proceedings ofthe 43rd International Conference on Machine Learning, pages 42215–42241, 2026.

[15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, 2017.

[16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[17] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

[18] InsightFace Contributors. Insightface: 2d and 3d face analysis project. https://github.com/deepinsight/ insightface. Software repository; accessed 2026-11-02.

[19] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17191–17202, 2025.

[20] Yury Kartynnik, Artsiom Ablavatski, Ivan Grishchenko, and Matthias Grundmann. Real-time facial surface geometry from monocular video on mobile gpus. arXiv preprint arXiv:1907.06724, 2019.

[21] Chunyu Li, Chao Zhang, Weikai Xu, Jingyu Lin, Jinghui Xie, Weiguo Feng, Bingyue Peng, Cunjian Chen, and Weiwei Xing. Latentsync: Taming audio-conditioned latent diffusion models for lip sync with syncnet supervision. arXiv preprint arXiv:2412.09262, 2024.

[22] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

[23] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

[24] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

[25] Pingchuan Ma, Stavros Petridis, and Maja Pantic. Visual speech recognition for multiple languages in the wild. Nature Machine Intelligence, 4:930–939, 2022.

[26] Yifeng Ma, Shiwei Zhang, Jiayu Wang, Xiang Wang, Yingya Zhang, and Zhidong Deng. Dreamtalk: When emotional talking head generation meets diffusion probabilistic models. arXiv preprint arXiv:2312.09767, 2023.

[27] MiniMax. Open general intelligence: Minimax h3 is now open source. https://www.minimax.io/news/ minimax-h3-open-source, 2026. Official model release; accessed 2026-11-02.

[28] Soumik Mukhopadhyay, Saksham Suri, Ravi Teja Gadde, and Abhinav Shrivastava. Diff2lip: Audio conditioned diffusion models for lip-synchronization. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5292–5302, 2024.

[29] Ziqiao Peng, Wentao Hu, Yue Shi, Xiangyu Zhu, Xiaomei Zhang, Hao Zhao, Jun He, Hongyan Liu, and Zhaoxin Fan. Synctalk: The devil is in the synchronization for talking head synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[30] Ziqiao Peng, Jiwen Liu, Haoxian Zhang, Xiaoqiang Liu, Songlin Tang, Pengfei Wan, Di Zhang, Hongyan Liu, and Jun He. Omnisync: Towards universal lip synchronization via diffusion transformers. In Advances in Neural Information Processing Systems, 2025.

[31] K. R. Prajwal, Rudrabha Mukhopadhyay, Vinay Namboodiri, and C. V. Jawahar. A lip sync expert is all you need for speech to lip generation in the wild. In Proceedings of the ACM International Conference on Multimedia, pages 484– 492, 2020.

[32] Qwen Team. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qwen3.5, 2026.

[33] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Proceedings of the 40th International Conference on Machine Learning, pages 28492–28518, 2023.

[34] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022.

[35] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103, 2024.

[36] Shuai Shen, Wenliang Zhao, Zibin Meng, Wanhua Li, Zheng Zhu, Jie Zhou, and Jiwen Lu. Difftalk: Crafting diffusion models for generalized audio-driven portraits animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1982–1991, 2023.

[37] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In Proceedings ofthe 40th International Conference on Machine Learning, 2023.

[38] Michal Stypulkowski, Konstantinos Vougioukas, Sen He, Maciej Ziemba, Stavros Petridis, and Maja Pantic. Diffused heads: Diffusion models beat gans on talking-face generation. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, 2024.

[39] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[40] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[41] Tencent Music Entertainment Lyra Lab. Musetalk: Realtime high quality lip synchronization with latent space inpainting. https://github.com/TMElyralab/ MuseTalk, 2026. Official software repository; accessed 2026-11-02.

[42] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.

[43] Jiadong Wang, Xinyuan Qian, Malu Zhang, Robby T. Tan, and Haizhou Li. Seeing what you said: Talking face generation guided by a lip reading expert. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[44] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

[45] Huawei Wei, Zejun Yang, and Zhisheng Wang. Aniportrait: Audio-driven synthesis of photorealistic portrait animation. arXiv preprint arXiv:2403.17694, 2024.

[46] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Yao Yao, and Siyu Zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024.

[47] Sicheng Xu, Guojun Chen, Yu-Xiao Guo, Jiaolong Yang, Chong Li, Zhenyu Zang, Yizhong Zhang, Xin Tong, and Baining Guo. Vasa-1: Lifelike audio-driven talking faces generated in real time. In Advances in Neural Information Processing Systems, 2024.

[48] Zhenhui Ye, Ziyue Jiang, Yi Ren, Jinglin Liu, Jinzheng He, and Zhou Zhao. Geneface: Generalized and high-fidelity audio-driven 3d talking face synthesis. In International Conference on Learning Representations, 2023.

[49] Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems, pages 47455–47487, 2024.

[50] Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[51] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.

[52] Shifeng Zhang, Xiangyu Zhu, Zhen Lei, Hailin Shi, Xiaobo Wang, and Stan Z. Li. S3fd: Single shot scale-invariant face detector. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 192–201, 2017.

[53] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker: Learning realistic 3d motion coefficients for stylized audiodriven single image talking face animation. In CVPR, 2023.

[54] Yue Zhang, Zhizhou Zhong, Minhao Liu, Zhaokang Chen, Bin Wu, Yubin Zeng, Chao Zhan, Yingjie He, Junxin Huang, and Wenjiang Zhou. Musetalk: Real-time high-fidelity video dubbing via spatio-temporal sampling. arXiv preprint arXiv:2410.10122, 2024.

[55] Youliang Zhang, Zhaoyang Li, Duomin Wang, Jiahe Zhang, Deyu Zhou, Zixin Yin, Xili Dai, Gang Yu, and Xiu Li. Speakervid-5m: A large-scale high-quality dataset for audiovisual dyadic interactive human generation. In International Conference on Learning Representations, pages 117896– 117926, 2026.

[56] Zhimeng Zhang, Lincheng Li, Yu Ding, and Changjie Fan. Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3661–3670, 2021.

[57] Zhimeng Zhang, Zhipeng Hu, Wenjin Deng, Changjie Fan, Tangjie Lv, and Yu Ding. Dinet: Deformation inpainting network for realistic face visually dubbing on high resolution video. In Proceedings of the AAAI Conference on Artificial Intelligence, 2023.

[58] Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive diffusion distillation for real-time interactive video generation. arXiv preprint arXiv:2605.15141, 2026.

[59] Hang Zhou, Yasheng Sun, Wayne Wu, Chen Change Loy, Xiaogang Wang, and Ziwei Liu. Pose-controllable talking face generation by implicitly modularized audio-visual representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

[60] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.

[61] Andreas Zinonos, Michal Stypulkowski, Antoni Bigata, Stavros Petridis, Maja Pantic, and Nikita Drobyshev. Flashlips: 100-fps mask-free latent lip-sync using reconstruction instead of diffusion or gans. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10898–10908, 2026.

## Supplementary Material

This supplement provides data construction, replacementpairing rules, training details, and extended evaluations supporting the main paper. It expands the evidence for replacement-aware cross-audio supervision, full-context twostep distillation, and 60-second overlapping-window inference. It also reports a controlled causal variant that isolates the quality–latency effect of causal conversion. Complete benchmark protocols, uncertainty, qualitative comparisons, reproducibility details, and ethical safeguards are included.

## A Guide to the Supplement

This supplement records implementation and evaluation details needed for reproduction but omitted from the main paper. Intermediate checkpoints are reported separately from the final EMA@200 teacher, the full-context two-step student, and the controlled causal variant. A fixed 16-source subset is used only for diagnosis and a separate 256-source split is used only for development and model selection. The sole internal benchmark reported in the paper is the sourcedisjoint 256-video VidDub-Bench.

The material proceeds from data and teacher construction to few-step distillation, the controlled causal study, and final evaluation. Controlled ablations measure target-speech sensitivity and the effects of cross-audio supervision; subsequent sections document both full-context and causal training paths. Final benchmark tables cover HDTF, VoxCeleb2, TalkVid-Bench, and VidDub-Bench, followed by complete same/cross-audio examples and evaluation protocols.

## B Why the Final Editor Remains Full-Context

The complete source video and replacement speech are available before offline dubbing, so progressive block emission is not required. We nevertheless trained a matched Causal two-step VidDub using the same backbone family, data, teacher, trajectory objective, DMD recipe, preprocessing, and evaluation manifests. Table 12 isolates the resulting design trade-off.

Causal attention reduces first-decodable-output latency by 6.81×, which is its measured advantage. For bounded offline inference, the full-context model is faster at 81, 241, 721, and 1,441 frames, uses less peak memory, improves all six visual-only speech-recognition comparisons, and improves all three preservation measures on every evaluated benchmark. The two designs are effectively tied on Long-20 synchronization. These results motivate the final full-context architecture while retaining the Causal two-step VidDub as a controlled design study rather than the main method.

![](images/6ec3cfd068f3c23efb188d3015de8f03634cb98e80fedcb702c2f9d2abd8d8fc.jpg)

![](images/1715cd9ddc4665a71defa7093e897ece687a8a197a113c3d09c07f155d61f649.jpg)  
Figure 5: Source-motion distributions from 12 sampled frames per video. Non-mouth change excludes the lower face; center motion is normalized by frame diagonal.

## C Data Construction and Splits

V2 benchmark audit. The H3 construction begins with 100,000 structured talking-human prompts. The frozen 256- source VidDub-Bench contains 81 frames per source at 24 FPS and fixed replacement-audio partners. Joined prompt metadata contains 42 emotion, 38 setting, 73 gesture, and 15 age labels. These are construction annotations, so we additionally measure decoded-frame dynamics. Table 14 gives the exact medians summarized by Fig. 5.

Construction and release unit. We generate candidate talking-human clips from the structured prompt set, then apply the face-track, crop, duration, and audio–visual filters described below. The benchmark freezes 256 accepted source videos before final model evaluation. Each source receives one same-audio track and one replacement waveform drawn from a different source video; source frames, crop, mask, seed, and duration remain fixed between tracks. Table 13 summarizes the released unit.

Release scope. We will release the generated benchmark source clips, original and replacement waveforms, fixed pair manifest, prompt metadata, preprocessing outputs, split and exclusion manifests, and evaluation scripts. For thirdparty public benchmarks, the release contains identifiers and deterministic manifests rather than redistributing media whose licenses do not permit it.

Ranking audit. For the nine methods present in every benchmark, we compute Spearman correlations between sameaudio Sync-C or SSIM and cross-audio Sync-C. Captured-togenerated correlation uses the seven external methods and averages each method’s cross-audio Sync-C over HDTF, Vox-Celeb2, and TalkVid before comparison with VidDub-Bench. The reported coefficients summarize these method rankings. Source domains. The teacher pool combines six domains. H3 [27] contains internally generated talking-human videos whose records include verbatim speech text, Chinese or English language, subject attributes, emotion, setting, and the visual prompt. TalkVid [3] and SpeakerVid-5M [55] provide natural web-video diversity; HDTF [56] and VoxCeleb2 [8] provide standard public talking-face data; Hallo3 [10] adds reference-portrait and motion diversity. After filtering, the final teacher pool contains 322,513 clips: 85,435 H3, 152,713 TalkVid, 57,727 SpeakerVid, 6,527 HDTF, 2,437 VoxCeleb2, and 17,674 Hallo3 clips. Domain-balanced sampling uses relative weights 30/20/15/12/8/10 in the same order, with per-clip quality weights inside each domain. The 85,435 retained H3 clips constitute the released training set.

Table 12: Matched full-context and causal two-step designs. Timing is the median of three warmed runs on one H20; quality uses identical frozen manifests.
<table><tr><td>Design</td><td>First output</td><td>Frames</td><td>81-frame time</td><td>Peak GiB</td><td>VSR wins</td><td>Upper-L1</td><td>ID</td><td>Pose</td></tr><tr><td>Causal two-step</td><td>1.052 s</td><td>9</td><td>8.662 s</td><td>21.96</td><td>0/6</td><td>1.079</td><td>0.861</td><td>0.614</td></tr><tr><td>Full-context two-step (VidDub)</td><td>7.165 s</td><td>81</td><td>7.195 s</td><td>13.77</td><td>6/6</td><td>1.014</td><td>0.885</td><td>0.567</td></tr></table>

Table 13: VidDub-Bench construction statistics. Duration and resolution refer to the model-ready generated source clips; metadata counts denote distinct labels represented by the frozen benchmark.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Structured prompts</td><td>100,000</td></tr><tr><td>Frozen generated source videos</td><td>256</td></tr><tr><td>Same-audio tracks / replacement pairs</td><td>256 / 256</td></tr><tr><td>Frames / duration / frame rate</td><td>81 / 3.375 s / 24 FPS</td></tr><tr><td>Model-ready spatial resolution</td><td>512 × 512</td></tr><tr><td>Total unique-source duration</td><td>14.4 min</td></tr><tr><td>Emotion / setting labels</td><td>42 /38</td></tr><tr><td>Gesture / age-group labels</td><td>73 / 15</td></tr></table>

Table 14: Source-dynamics audit. Non-mouth ∆ is mean absolute grayscale change after excluding the detected lower-face region; center motion is normalized by frame diagonal; scale CV uses detected face-box area.
<table><tr><td>Source set</td><td>n Non-mouth ∆ Center motion Scale CV</td></tr><tr><td>HDTF 30</td><td>1.30 1.03</td></tr><tr><td>VoxCeleb2 30</td><td>4.83</td></tr><tr><td>2.14</td><td>1.82 5.86 1.45 8.94</td></tr><tr><td>TalkVid 500</td><td>4.57 2.29 6.31</td></tr><tr><td>VidDub-Bench 256</td><td></td></tr></table>

Table 15: Metric- and domain-dependent rankings. The first two columns compare same-audio metrics against cross-audio Sync-C within each benchmark.
<table><tr><td>Benchmark</td><td>Same Sync-C</td><td>Same SSIM</td></tr><tr><td>HDTF</td><td>0.92</td><td>-0.10</td></tr><tr><td>VoxCeleb2</td><td>0.98</td><td>-0.30</td></tr><tr><td>TalkVid</td><td>0.98</td><td>0.23</td></tr><tr><td>VidDub-Bench</td><td>0.85</td><td>0.60</td></tr><tr><td>Captured→AIGC cross Sync-C</td><td>ρ = 0.71</td><td></td></tr></table>

Face and audio preprocessing. For each source video, preprocessing detects landmarks and tracks the dominant face. Windows with missing or unstable tracks, invalid crop geometry, extreme pose, insufficient mouth motion, or insufficient audio–visual correspondence are excluded. SyncNet [7] estimates the temporal offset and confidence; retained audio is shifted by the estimated offset before window extraction. Each example contains 81 frames at 24 fps, a stable square face crop resized to $5 1 2 \times 5 1 2 .$ a union lower-face edit mask, one mechanically selected identity reference, and a 16-kHz mono waveform. Preprocessing caches the ground-truth crop, masked/inactive source condition, reference latent, audio features, and empty-text context. For evaluation, frozen cross-audio benchmark manifests use different source groups while keeping source video, mask, crop, seed, and frame count fixed.

Teacher and causal manifests. Causal adaptation samples 16,384 clips from the 322,513 eligible teacher clips without replacement under the frozen quality/domain distribution. These clips represent 5,174 H3, 3,450 TalkVid, 2,587 SpeakerVid, 2,070 HDTF, 1,724 Hallo3, and 1,379 VoxCeleb2 clips, which sum to 16,384. The frozen manifest contains 15,346 distinct source IDs and 10,123 leak-free group keys; 1,038 source IDs contribute a second temporal window. Training manifests, quality weights, source-video identifiers, group keys, and selected checkpoint hashes are included in the reproducibility package.

Disjoint development and test sets. The development set contains 256 source videos and is used for checkpoint selection. A fixed 16-source subset is used only for diagnosis. VidDub-Bench contains another 256 source videos, each represented by one 81-frame same-audio example and one fixed cross-audio pairing, yielding 512 requested outputs per method. Tables report same- and cross-audio metrics separately and retain track-specific validity; for example, LatentSync has 254/256 valid cross-audio preservation outputs. Source-video intersections are zero for train–development, train–VidDub-Bench, and development– VidDub-Bench. The benchmark is frozen before final evaluation; no training or checkpoint selection uses its outputs. The benchmark shares the generated H3 domain with part of training but not source videos. The 433 group keys derived from the 560 frozen HDTF-30, VoxCeleb2-30, and TalkVid-Bench clips (26 person, 25 speaker, and 382 YouTube-video groups) match the exclusion list and have zero intersection with the eligible pool, causal-training manifest, and development manifest. Thus all evaluation identities and source videos are held out at the declared group granularity, while the represented source domains may also contribute different clips to the training pool. Every same/cross comparison fixes source video, seed, crop geometry, and frame count. Because the source mouth is not the target under cross audio, source-mouth reconstruction metrics are intentionally omitted for that track.

## D Multilingual Qualitative Results

Figure 6 broadens the main-paper teaser with mechanically selected held-out English and Chinese cases. Each column fixes source and time, while rows separate the source, same-audio editing, and replacement-speech editing. Mouth insets expose articulation that is difficult to inspect at full-frame scale.

## E Bidirectional Teacher Training

## E.1 Training chain

The teacher follows three semantic stages matching the mainpaper figure: audio-conditioning warm-up trains only the audio adapter, joint editing adaptation updates DiT, VACE, and the adapter, and replacement-aware teacher refinement adds replacement audio and synchronization supervision behind a DiT LoRA [17]. The resulting EMA state at step 200 is the parent used throughout the paper. Two independent auxiliary branches study quality-only refinement and replacement refinement after that quality-only branch; neither is an ancestor of the teacher. Table 16 lists the settings recorded in each run’s saved arguments. “Full” denotes direct DiT optimization; “LoRA-64” denotes a frozen DiT base with rank-64 adapters merged at export. The 81-frame training statement in the main method applies to joint adaptation, replacement-aware refinement, causal adaptation, and distillation; the adapter-only initialization uses 49-frame clips.

Teacher refinement and both auxiliary branches use mouth LPIPS weight 0.1 on same-audio samples and representation alignment at transformer layer 8 with weight 0.1. The replacement-aware branches add synchronization weight 0.02 for $\sigma \leq 0 . 8$ and EMA 0.99. The replacement-afterquality branch runs for 300 updates and is evaluated only as an auxiliary ablation; all causal models use the replacementaware teacher EMA at step 200. The merged DiT state has the base-model key layout and requires no LoRA module at inference.

Table 17 shows why teacher construction matters for the downstream causal study. Relative to joint editing adaptation, quality-only refinement lowers cross FVD from 33.50 to 29.56 while slightly increasing cross Sync-C from 5.825 to 6.104. Applying replacement refinement after that branch raises cross Sync-C to 9.135 at a small SSIM/FVD cost. Direct replacement-aware teacher refinement after joint adaptation gives the best joint Sync-C/FVD result (cross Sync-C 9.422 and FVD 27.49). These staged comparisons have different accumulated histories and are not single-factor causal effects.

Replacement-aware refinement raises cross-audio Sync-C by 1.436/0.954/1.134/1.653 on the four benchmarks. Paired 10,000-resample source bootstraps give 95% intervals [1.304, 1.560], [0.863, 1.043], [0.984, 1.288], and [1.485, 1.823], respectively; all 160 paired items improve. FVD changes by +1.09/+0.37/+0.40/−0.84, so the fidelity direction is mixed. The matched result identifies a cross-audio synchronization gain with mixed FVD changes. The intervals are paired source bootstraps over the 160 frozen evaluation items. TalkVid-Bench-50 is a fixed stratified teacherdevelopment subset of TalkVid-Bench. H3-Val-50 is the legacy held-out H3 validation diagnostic. These teacherdevelopment sets are distinct from both dev-256 and the final VidDub-Bench-256 evaluation set.

## E.2 Teacher-refinement ablation protocol

The teacher study separates historical staged evidence from the matched comparison in Table 18. Warm-up and joint editing adaptation measure an association between backbone adaptation and synchronization; the quality-only branch studies perceptual and representation recovery without a Sync objective. The matched control then isolates the implemented replacement-sampling branch by fixing the parent, data, losses, optimizer, and update count. Live versus EMA export and LoRA versus direct tuning concern optimization stability rather than objective design.

## E.3 Audio conditioning and replacementaware targets

A frozen Whisper-small encoder [33] supplies 13 hiddenstate levels. A learned mixture and projection produce local audio tokens. Each video-latent frame reads 16 Whisper tokens, approximately 320 ms at 50 Hz, selected by absolute temporal position. Audio cross-attention follows each of the 30 DiT blocks. Thus, multi-block causal training gives each video block its corresponding audio segment rather than a repeated chunk-level embedding.

For ordinary same-audio training, the mouth receives increased flow weight. For a cross-audio pair $( z , a ^ { \prime } )$ $a ^ { \prime }$ is taken from the preceding sampled clip on the same training rank; the sampler does not explicitly enforce a different source video, identity, transcript, temporal non-overlap, or language. All teacher caches contain 81 frames at 24 fps and audio is read from sample zero through 54,000 at 16 kHz. The swap therefore uses the cached fixed-duration waveform without a random offset, looping, or runtime resampling. Source-mouth flow regression is removed:

$$
\mathcal { L } _ { \mathrm { f o w } } ^ { \mathrm { c r o s s } } = \left. ( 1 - M ) \odot ( v _ { \theta } - v ^ { * } ) \right. _ { 2 } ^ { 2 } .\tag{10}
$$

The non-mouth region remains supervised, audio dropout is disabled, and the edited mouth is constrained by synchronization. LPIPS and representation-alignment targets tied to the source mouth are not applied to cross-audio examples. The 0.30 teacher rate is the submitted teacher-refinement setting. Teacher refinement draws Gaussian noise and a discrete scheduler timestep uniformly; replacement examples are restricted to $\sigma \leq 0 . 8 .$ , while same-audio examples use the full schedule. The scheduler’s timestep weight multiplies the normalized flow error.

![](images/435983a6ec26bcfa998ea8b0e46d4c0dba3ca8f2cdd280894c23065c7fe89c1e.jpg)  
Figure 6: Held-out multilingual examples. Twelve mechanically selected cases (six English and six Chinese) are shown at matched times. Rows contain the source, same-audio edit, and replacement-speech edit; insets enlarge the mouth, and green borders identify replacement-speech outputs.

Table 16: Bidirectional teacher training stages. The first three rows form the parent sequence; the final two rows are independent auxiliary branches. Warm-up and joint adaptation were run as resumed segments, whose update counts are summed.
<table><tr><td>Stage</td><td>Initialized from</td><td>Frames</td><td>GPUs</td><td>Updates</td><td>Adapter LR</td><td>DiT</td><td>VACE LR</td><td>Cross audio</td><td>Auxiliary losses</td></tr><tr><td>Audio-conditioning warm-up</td><td>scratch adapter</td><td>49</td><td>48</td><td>6,000+2,000+9,700</td><td> $\overline { { 1 0 ^ { - 4 } \to 5 \times 1 0 ^ { - 5 } } }$ </td><td>frozen</td><td>frozen</td><td>0</td><td>flow</td></tr><tr><td>Joint editing adaptation</td><td>warm-up</td><td>81</td><td>48</td><td>3,000+2,500</td><td>5×10−5</td><td> $\mathrm { f u l l , 1 0 ^ { - 5 } }$  5</td><td>10−5</td><td>0</td><td>flow</td></tr><tr><td>Replacement-aware teacher refinement</td><td>joint adaptation</td><td>81</td><td>16</td><td>300</td><td>10−5</td><td> $\mathrm { L o R A } \mathrm { - } 6 4 , 5 \times 1 0 ^ { - }$ </td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>0.30</td><td>Sync, LPIPS, repr. alignment</td></tr><tr><td>Auxiliary ablations</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Quality-only refinement</td><td>joint adaptation</td><td>81</td><td>48</td><td>2,000</td><td>10−5</td><td> $\mathrm { L o R A } \mathrm { - } 6 4 , 5 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>0</td><td>LPIPS, repr. alignment</td></tr><tr><td>Replacement-after-quality branch</td><td>quality-only EMA@500</td><td>81</td><td>48</td><td>300</td><td>10-5</td><td> $\mathrm { L o R A } { - } 6 4 , 5 { \times } 1 0 ^ { - 1 }$  5</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>0.30</td><td>Sync, LPIPS, repr. alignment</td></tr></table>

Differentiable synchronization objective. Training uses the frozen LatentSync-1.6 StableSyncNet checkpoint [1, 21], not the official SyncNet-v2 evaluator [7, 8]. A 16- frame, 24-fps decoded window is cropped to the maskderived lower face, widened by 10% horizontally, and resized to $1 2 8 \times 2 5 6$ . Its corresponding 16-kHz mel spectrogram has 80 bins at 80 Hz and is resampled to 52 columns. If $e _ { v } , e _ { a }$ are visual and audio embeddings, the loss $\mathrm { i s } - \mathrm { l o g } [ \mathrm { m a x } ( \cos ( e _ { v } , e _ { a } ) , 1 0 ^ { - 6 } ) ]$ . Training performs no temporal-lag search. StableSyncNet parameters remain frozen, but differentiation through the network and VAE decoder sends the loss to the generator. Final Sync-C/D instead use the official S3FD [52]/SyncNet-v2 [7, 8] pipeline described in Sec. I.

Table 17: Empirical teacher-development checkpoints on the same HDTF-30 protocol (30 same and 30 cross outputs per row). These rows document the effect of the staged SFT recipe and the final teacher choice; because the earlier and fina training chains also differ in accumulated optimization, the table is not presented as a one-factor causal ablation.
<table><tr><td>Checkpoint</td><td>SSIM↑</td><td>Same C/D</td><td>Cross C/D</td><td>Cross FVD↓</td></tr><tr><td>Joint editing adaptation</td><td>0.9230</td><td> $7 . 3 5 5 / 8 . 0 8 6$ </td><td> $5 . 8 2 5 / 9 . 6 0 3$ </td><td>33.50</td></tr><tr><td>Quality-only refinement, EMA</td><td>0.9233</td><td>7.561 / 7.845</td><td>6.104 / 9.256</td><td>29.56</td></tr><tr><td>Replacement-after-quality branch, EMA</td><td>0.9216</td><td>9.371 / 6.464</td><td>9.135 / 6.631</td><td>31.59</td></tr><tr><td>Replacement-aware teacher, EMA</td><td>0.9279</td><td>9.760 / 5.900</td><td>9.422 / 6.127</td><td>27.49</td></tr></table>

Table 18: Matched replacement-sampling control at EMA step 200. Both refinements start from the same joint-adaptation checkpoint and share data, weights, losses, optimizer, LoRA, EMA, update count, and evaluation pairs. The only configuration change is replacement probability 0 versus 0.30; when replacement is selected, the implementation also disables audio dropout. Cells report cross-audio Sync-C/FVD.
<table><tr><td>Teacher refinement</td><td>HDTF-30</td><td>TalkVid-Bench-50</td><td>H3-Val-50</td><td>VoxCeleb2-30</td></tr><tr><td>No replacement branch</td><td>7.986 / 26.40</td><td>6.460 / 26.14</td><td>6.952 / 37.68</td><td>5.729 / 39.95</td></tr><tr><td>Replacement-aware (0.30)</td><td>9.422 / 27.49</td><td>7.414 / 26.51</td><td>8.086 / 38.09</td><td>7.382 / 39.11</td></tr></table>

Table 19: Detailed teacher-refinement ablation protocol and interpretation. “Visual criteria met” means SSIM, mouth LPIPS, FVD, and artifact inspection remain within the frozen tolerance relative to the preceding stage.
<table><tr><td>Comparison</td><td>Isolated factor</td><td>Primary metrics</td><td>Required direction</td><td>Interpretation</td></tr><tr><td>Warm-up → joint adaptation</td><td>backbone/VACE adaptation</td><td>same/cross C/D, FVD</td><td>cross C up</td><td>audio features become usable by the generator</td></tr><tr><td>Joint adaptation → quality-only No-replacement → replacement-aware refinement</td><td>LPIPS+repr. alignment+LoRA+EMA implemented replacement branch</td><td>SSIM, mouth LPIPS, FVD cross C/D, FVD</td><td>visual criteria met cross C improves</td><td>improves visual metrics while reporting synchronization matched replacement-sampling comparison; dropout semantics disclosed</td></tr><tr><td>live → EMA</td><td>parameter averaging</td><td>all metrics, failures</td><td>lower variance</td><td>separates optimization noise from method gain</td></tr><tr><td>reference/dropout sweep</td><td>conditioning strength</td><td>cross-audio synchronization, identity</td><td>Pareto improvement</td><td>tests appearance-prior competition</td></tr></table>

## F Causal Architecture and Optimization

The implementation combines Wan-1.3B [40], VACE [19], and a frozen Whisper-small encoder [33].

Loss coefficients in the main-paper notation. We use $\begin{array} { r l r } { \lambda _ { M } } & { { } = } & { 2 . } \end{array}$ , which gives mouth flow weight 1 + $\lambda _ { M } ~ = ~ 3 .$ The full-context pixel/LPIPS/sync/keep coefficients are (0, 0.10, 0.02, 0), and the corresponding causal-adaptation coefficients are (0.02, 0.10, 0.02, 0.10). For trajectory regression, $\begin{array} { r l r } { \lambda _ { q } ^ { \mathrm { t r a j } } } & { { } = } & { 0 . 3 6 6 4 8 4 6 } \end{array}$ and $( \lambda _ { \mathrm { p i x } } ^ { \mathrm { t r a j } } , \lambda _ { \mathrm { p e r c } } ^ { \mathrm { t r a j } } , \lambda _ { \mathrm { s y n c } } ^ { \mathrm { t r a j } } , \lambda _ { \mathrm { k e e p } } ^ { \mathrm { t r a j } } ) = ( \bar { 0 } . 0 2 , 0 . 1 0 , 0 . 0 2 , 0 . 1 0 )$ . For distribution matching, $\bar { \lambda } _ { \mathrm { { r e t } } } = 0 . 1 0 , \lambda _ { q } ^ { \mathrm { { D M D } } } = 0 . 9 8 7 9 1 2 7 ,$ and $( \lambda _ { \mathrm { p i x } } ^ { \mathrm { D M D } } , \lambda _ { \mathrm { p e r c } } ^ { \mathrm { D M D } } , \lambda _ { \mathrm { k e e p } } ^ { \mathrm { D M D } } ) = ( 0 . 0 2 , 0 . 1 0 , 0 . 1 0 )$ The DMD noise level follows $\sigma \sim \mathcal { U } [ 0 . 0 2 , 0 . 9 8 ]$

The Causal two-step VidDub partitions 21 video latents into seven blocks of three after a separate reference latent. It uses absolute temporal positions, block-causal self-attention, and frame-local audio cross-attention. Every target block receives independent noise and timestep samples. Clean history is detached for standard teacher-forced AR adaptation. To fit all-block training in memory, blocks are forwarded and backpropagated sequentially, with each loss divided by the number of selected blocks before one optimizer update. Flow and pairing details. For $u \sim \mathcal { U } ( 0 , 1 )$ , causal adaptation uses $\sigma = 5 u / ( 1 + 4 u ) , x _ { \sigma } = ( 1 - \sigma ) z + \sigma \epsilon$ , and velocity target $v ^ { * } = \epsilon - z .$ . The VACE mask channels are reduced by a channel maximum and clamped to obtain latent support M. The pointwise squared velocity error is multiplied by 1 + 2M for same audio and 1 − M for replacement audio, then divided by the mean weight. With probability 0.10, the online sampler draws a donor uniformly from the 16,384 training clips; it replaces an exact clip-ID collision with the following manifest item but imposes no group, identity, transcript, language, or silence filter. Under this manifest, the induced same-group and same-language probabilities are 0.0294% and 50.40%, respectively. DMD uses the same online rule at probability 0.50.

Trajectory generation instead freezes 2,048 source rows and emits both same- and cross-audio records, yielding 4,096 records in a 1:1 ratio. Each cross donor is deterministically selected from the same domain and language but a different source group. Thus the ODE configuration’s online replacement probability zero does not mean that trajectory regression lacks replacement examples.

Table 20: Architecture, sampling, and loss configuration for the three causal stages. Entries marked “same” inherit the causal-adaptation setting. Reported NFE counts denoising steps only; the Causal two-step VidDub additionally performs one clean state-commit forward per block during inference deployment.
<table><tr><td></td><td>Causal adaptation</td><td>Two-step initialization</td><td>Guided DMD</td></tr><tr><td>Architecture and sampling</td><td></td><td></td><td></td></tr><tr><td>Backbone</td><td>Wan-1.3B + VACE</td><td>same</td><td>same</td></tr><tr><td>Audio module</td><td>Whisper-small + 287M</td><td>same</td><td>same</td></tr><tr><td></td><td>adapter</td><td></td><td></td></tr><tr><td>Input clip</td><td>81 frames, 5122, 24 fps</td><td>same</td><td>same</td></tr><tr><td>Latent layout Causal blocks</td><td>1 reference + 21 video latents</td><td>same</td><td>same</td></tr><tr><td>Blocks with generator gradient</td><td>7 blocks × 3 latents</td><td>same</td><td>same</td></tr><tr><td>History</td><td>all 7, sequential backward</td><td>1 sampled trajectory block</td><td>1 sampled rollout block</td></tr><tr><td>Precision / checkpointing</td><td>clean ground truth bf16 / enabled</td><td>stored causal trajectory bf16 / enabled</td><td>student&#x27;s causal rollout bf16 / enabled</td></tr><tr><td>Reported inference NFE</td><td>48</td><td></td><td>2</td></tr><tr><td>Flow shift</td><td>5</td><td>2 5</td><td>52</td></tr><tr><td>Teacher audio guidance</td><td>1</td><td>2</td><td></td></tr><tr><td>Inference audio guidance</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Replacement-audio sampling</td><td>0.10 online</td><td>frozen 1:1 same/cross</td><td>0.50 online</td></tr><tr><td>Same-audio dropout</td><td>0.10</td><td>0</td><td>0</td></tr><tr><td>DMD range / fake-score updates</td><td></td><td>1</td><td> $\sigma \sim \mathcal { U } \left[ 0 . 0 2 , 0 . 9 8 \right] / 5$ </td></tr><tr><td>Losses</td><td></td><td></td><td></td></tr><tr><td>Mouth flow weight</td><td>3</td><td>3</td><td>3</td></tr><tr><td>Pixel / LPIPS weight</td><td>0.02 / 0.10</td><td>0.02 /0.10</td><td>0.02 / 0.10</td></tr><tr><td>Synchronization / non-mouth keep weight</td><td>0.02 / 0.10</td><td>0.02 / 0.10</td><td>0 / 0.10</td></tr><tr><td>Retention weight</td><td></td><td></td><td>0.10</td></tr><tr><td>Rollout-quality scale</td><td></td><td>0.3664846</td><td>0.9879127</td></tr><tr><td>Quality-loss noise range</td><td> $\sigma \leq 0 . 5$ </td><td> $\sigma \leq 1 . 0$ </td><td> $\sigma \leq 1 . 0$ </td></tr><tr><td>Sync-loss noise range</td><td> $\sigma \leq 0 . 8$ </td><td> $\sigma \leq 0 . 8$ </td><td></td></tr><tr><td>Quality losses on replacement audio</td><td>Sync + non-mouth keep</td><td>Sync + non-mouth keep</td><td>non-mouth keep only</td></tr></table>

Table 21: Optimization and inference configuration. The effective batch equals the number of workers because each worker processes one clip without gradient accumulation.
<table><tr><td></td><td>Causal adaptation</td><td>Two-step initialization</td><td>Guided DMD</td></tr><tr><td colspan="4">Optimization</td></tr><tr><td>Optimizer / schedule</td><td>AdamW / cosine</td><td>AdamW / cosine</td><td>AdamW / cosine</td></tr><tr><td>Adapter or student learning rate</td><td> $5 \times 1 0 ^ { - 6 }$ </td><td> $2 \times 1 0 ^ { - 6 }$ </td><td> $5 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Wan/VACE learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td><td> $5 \times 1 0 ^ { - 7 }$ </td><td> $2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Fake-score learning rate (adapter / base)</td><td></td><td></td><td> $1 \times 1 0 ^ { - 6 } / 5 \times 1 0 ^ { - 7 }$ </td></tr><tr><td> $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td>(0.9,0.999)</td><td>(0.9,0.999)</td><td> $( 0 . 9 , 0 . 9 9 9 )$ </td></tr><tr><td>Weight decay / grad. clip</td><td>0.01 / 1.0</td><td>0.01 / 1.0</td><td>0.01 / 1.0</td></tr><tr><td>LR schedule / minimum ratio</td><td>cosine / 0.1</td><td>cosine / 0.1</td><td>cosine / 0.1</td></tr><tr><td>Warmup updates</td><td>50</td><td>50</td><td>20</td></tr><tr><td>Declared / selected updates</td><td>800 / 400</td><td>400 / 400</td><td>100 / 100</td></tr><tr><td>GPUs / effective batch</td><td>32 / 32</td><td>16 / 16</td><td>16 / 16</td></tr><tr><td>Microbatch / grad. accumulation</td><td>1/1</td><td>1/1</td><td>1/1</td></tr><tr><td>Data-loader workers / GPU</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Random seed</td><td>20261008</td><td>20261008</td><td>20261008</td></tr><tr><td colspan="4">Inference</td></tr><tr><td>Inference branches</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Inference NFE</td><td></td><td>2</td><td>2</td></tr><tr><td>Persistent state</td><td>clean block KV cache</td><td>clean block KV cache</td><td>bounded clean block KV cache</td></tr></table>

Decoded quality terms. The complete available latent prefix is decoded and the trailing 16 frames are scored. Pixel loss is L1 inside the editable support; VGG-LPIPS [51] compares the mask bounding box on four evenly spaced frames; keep loss is L1 outside the support. Pixel and LPIPS are enabled only for same audio, whereas keep applies to both branches. Sync applies when audio is present and $\sigma \leq 0 . 8$ . Consequently, causal and trajectory quality are

Table 22: Prepared 81-frame inference efficiency. Medians are computed over ten timed runs after two warm-up runs. The scope includes condition preparation, causal denoising, and VAE decoding, but excludes face detection, paste-back, and source encoding.
<table><tr><td>Quantity</td><td>Median</td></tr><tr><td>Condition preparation Time to first block Denoising VAE decode</td><td>0.014 s 0.678 s 5.632 s</td></tr><tr><td>Prepared-clip total Throughput</td><td>2.960 s 8.606 s 9.412 fps</td></tr><tr><td>Real-time factor Peak allocated memory Peak reserved memory</td><td>2.550 21.94 GiB</td></tr></table>

$$
\mathcal { L } _ { q } ^ { s } = . 0 2 \mathcal { L } _ { \mathrm { p i x } } + . 1 0 \mathcal { L } _ { \mathrm { L P I P S } } + . 0 2 \mathcal { L } _ { \mathrm { s y n c } } + . 1 0 \mathcal { L } _ { \mathrm { k e e p } } ,\tag{11}
$$

$$
\mathcal { L } _ { q } ^ { c } = . 0 2 \mathcal { L } _ { \mathrm { s y n c } } + . 1 0 \mathcal { L } _ { \mathrm { k e e p } } ^ { \bar { M } } .\tag{12}
$$

Trajectory regression scales either branch by 0.3664846. Its $1 + 2 M$ regression target is the matched teacher trajectory’s terminal latent, so the cross-audio mouth target is a replacement-conditioned pseudo-target rather than source video.

At inference, conditional and null-audio KV caches are independent when CFG is requested. A clean $t = 0$ pass commits a completed block; no noisy intermediate is written into persistent history. For long streams, the cache retains the separate reference and a fixed number of most-recent video latent frames; cached keys preserve their absolute RoPE positions [39], while the absolute generation endpoint continues to increase. The final distilled model uses only the conditional branch with audio CFG equal to one. Regression tests verify that the unbounded default is bitwise unchanged, bounded and unbounded short rollouts agree when the cap covers the sequence, and the retained KV length stays constant over many causal blocks.

Long audio uses overlapping Whisper windows blended on one global 50-Hz timeline, then projects audio tokens once for all absolute latent positions. The Wan VAE exposes causal streaming encode/decode using the same internal convolution feature state as full-clip inference. On a real 81-frame clip, streaming and full encode/decode are bitwise identical (maximum latent and decoded error 0). The frozen long-video set contains 20 strict contiguous HDTF source/target chains supporting genuine 10, 30, and 60-second prefixes; video is never looped or concatenated across identities.

## G Extended Causal-Adaptation Ablations

The controlled 160-update experiments start from the same SFT checkpoint, use effective batch 16, and retain the first 160 updates of the same 800-update cosine schedule. Table $2 5$ records each controlled change.

With the shifted flow schedule, $\sigma \leq 0 . 5$ occurs much less frequently than $\sigma \leq 0 . 8$ . After excluding an undersized first block, expected AR synchronization activation is approximately 12.9%, compared with roughly 41.3% effective exposure in the replacement-after-quality branch after accounting for cross-audio and audio dropout. Logged activation rates of 0.10–0.14 agree with this diagnosis.

## G.1 Guidance-scale ablation

For fixed B3@160 weights, cross-audio CFG is varied without retraining. The monotonic change in Table 27 shows that the output remains sensitive to the audio-conditioning scale after causal adaptation.

## G.2 Longer AR and history-policy results

A0–A4 train for 800 updates. R0–R3 start from A3@400 and continue for 400 updates. R3 uses a four-step rollout only to form training history; every row in Table 28 is evaluated with 50 denoising steps.

The full-context parent gives the best joint result. Among history-policy continuations, R0 has the strongest reconstruction, A4 has the highest cross Sync-C, and R3 has the lowest FVD.

## H Few-Step Initialization and DMD Details

## H.1 Causal consistency initialization

Our Causal-Forcing++ adaptation [58] is pinned to upstream commit da3ddf1a590f30c9edc568e11c2f7717 0995c31c. Online causal consistency distillation uses an adjacent frozen-teacher Euler step followed by an EMA student prediction on a shifted 48-point grid. All blocks share a sampled exit step and inference uses $x _ { 0 }$ re-noising. The optimizer follows the upstream configuration: AdamW [23] with learning rate $2 \times 1 0 ^ { - 6 } , \beta _ { 1 } = 0 , \beta _ { 2 } = 0 . 9 9 9$ weight decay 0.01, gradient clipping 10, and EMA 0.99. Wan-Dub auxiliary losses are calibrated relative to the CD gradient and reported separately from pure CD.

![](images/17ff0b246b5d91f2852b4cc0858e2a2cd732c9ea5e3324c223dbcda9ac6e8ac2.jpg)  
Figure 7: Streaming implementation validation. A four-step checkpoint generates a real continuous 10-second HDTF chain with one persistent bounded cache, overlapping long-audio Whisper features, and streaming VAE state. The absolute cache endpoint reaches 65 latent frames while retained KV saturates at 22 (one reference plus 21 video frames). TTFF is 1.16s and peak allocated memory is 23.04GiB. Official SyncNet evaluation in two independent five-second windows gives Sync-C/D 10.086/6.407 and 9.719/6.725, with zero offset in both windows; thus the persistent cache has zero measured offset drift in this interface test. The later 60-second audit exposes lower-face collapse despite stable synchronization. The measured end-to-end system RTF is 5.97.

![](images/9544c2d8d0db4220253dff39f8443c089d3726ed1e9edb105a742754995adef6.jpg)

![](images/1dd7ec4329280798a44e5502b98e9080b1f97b0f2baea45c78385521ccecf8af.jpg)

![](images/c60fdb1fe0ffc9ddd3dd4e3345b01d733ef980a1e152c2358826841b61c1efa8.jpg)

![](images/7285674a56210430489643d9292070c2d624aaa0fc24234645073ee045777a81.jpg)  
Figure 8: Earlier continuous-cache test on 20 frozen contiguous HDTF source–audio pairs. Curves show medians and interquartile ranges at 10, 30, and 60 seconds; generation never loops frames or concatenates identities.

Table 23: Persistent-cache architecture selection. Values are medians over 20 contiguous pairs; the combined synchronization, preservation, and visual measurements motivate the overlapping-window inference used by the fi nal model for stable long-video generation.
<table><tr><td>Duration (s)</td><td>Sources</td><td>RTF↓</td><td>Peak GiB↓</td><td>Sync-C / D</td><td>Keep L1↓</td><td>Boundary mouth  $\Delta _ { 1 } \downarrow$ </td><td>Sync failures</td></tr><tr><td>10</td><td>20</td><td>4.46</td><td>23.04</td><td>9.577 / 6.604</td><td>2.634</td><td>4.373</td><td>0</td></tr><tr><td>30</td><td>20</td><td>4.49</td><td>23.06</td><td>9.539 / 6.760</td><td>2.474</td><td>4.510</td><td>0</td></tr><tr><td>60</td><td>20</td><td>4.48</td><td>23.10</td><td>9.577 / 6.814</td><td>2.430</td><td>4.383</td><td>0</td></tr></table>

Table 24: Persistent-cache comparison at 60 seconds. Brackets are paired 95% source-bootstrap intervals; synchronization and preserved region metrics are reported under the same protocol.
<table><tr><td>Method</td><td>NFE</td><td>Peak GiB</td><td>Sync-C↑</td><td>Sync-D↓</td><td>Keep L1↓</td><td></td><td>Color↓ Sync failures</td></tr><tr><td>Causal AR-48 (CFG 2)</td><td>48</td><td>29.09 [29.09,29.09]</td><td>9.153 [8.774,9.505]</td><td>6.995 [6.832,7.172]</td><td>2.701 [2.232,3.264]</td><td>0.689 [0.627,0.748]</td><td></td></tr><tr><td>Trajectory ODÈ-2</td><td>22</td><td>23.27 [23.27,23.27]</td><td>9.019 [8.638,9.376]</td><td>7.130 [6.961,7.313]</td><td>2.930 [2.474,3.478]</td><td>0.867 [0.803,0.931]</td><td>0 0</td></tr><tr><td>Guided DMD-2</td><td></td><td>23.10 [23.10,23.10]</td><td>9.163 [8.773,9.523]</td><td>7.003 [6.831,7.193]</td><td>2.821 [2.358,3.376]</td><td>0.713 [0.659,0.770]</td><td>0</td></tr></table>

## H.2 Causal-consistency initializer selection

sequent guidance-transfer stage.

Table 29 records the 200-update initializer comparison. The re-noise parent provides the strongest joint visual and crossaudio operating point and is therefore selected for the sub-

Table 25: Controlled causal-adaptation variants. Only the listed factor changes relative to B3 unless stated otherwise.
<table><tr><td>ID</td><td>Intervention</td><td>Diagnostic purpose</td><td>Blocks per sample</td></tr><tr><td>A3</td><td>one uniformly sampled block</td><td>sparse-coverage reference</td><td>1/7</td></tr><tr><td>B3</td><td>sample without replacement</td><td>block-coverage sensitivity</td><td>3/7</td></tr><tr><td>B7</td><td>supervise every block</td><td>complete block coverage</td><td>7/7</td></tr><tr><td>S08</td><td> $\sigma _ { \mathrm { s y n c } } : 0 . 5 \to 0 . 8$ </td><td>noise-range sensitivity</td><td>3/7</td></tr><tr><td>SW06</td><td> $\lambda _ { \mathrm { s y n c } } : 0 . 0 2  0 . 0 6$ </td><td>loss-weight sensitivity</td><td>3/7</td></tr><tr><td>X30</td><td>S08 plus 30% counterfactual audio</td><td>replacement-audio supervision</td><td>3/7</td></tr><tr><td>HM50</td><td>suppress historical mouth with probability 0.5</td><td>history-conditioning sensitivity</td><td>3/7</td></tr><tr><td>C2</td><td>two-latent chunks, five sampled blocks</td><td>chunk-length sensitivity</td><td>5/10</td></tr></table>

Table 26: Complete 160-update controlled results corresponding to the mechanism ablations summarized in the main paper. “Three-block reference” is B3; all rows use the same 16-source protocol. FVD is reported as same/cross.
<table><tr><td>Intervention</td><td>SSIM↑</td><td>Mouth LPIPS↓</td><td>Same C/D</td><td>Cross C/D</td><td>FVD S/C↓</td><td>Primary change</td></tr><tr><td>Three-block reference</td><td>0.9310</td><td>0.1441</td><td>7.610 / 7.866</td><td>6.181 / 8.758</td><td>41.35 / 57.15</td><td>3/7 blocks</td></tr><tr><td>Broader Sync exposure</td><td>0.9311</td><td>0.1439</td><td>7.981 / 7.582</td><td>6.807 / 8.232</td><td>41.36 / 56.20</td><td> $\sigma _ { \mathrm { s y n c } } : 0 . 5 \to 0 . 8$ </td></tr><tr><td>Larger Sync weight</td><td>0.9310</td><td>0.1435</td><td>7.827 / 7.714</td><td>6.507 / 8.532</td><td>40.93 / 56.38</td><td> $\lambda _ { \mathrm { s y n c } } : 0 . 0 2 \substack {  0 . 0 6 }$ </td></tr><tr><td>Counterfactual speech</td><td>0.9217</td><td>0.1707</td><td>8.619 / 7.251</td><td>7.921 / 7.492</td><td>54.41 / 63.82</td><td>30% cross audio</td></tr><tr><td>Weaker mouth history</td><td>0.9323</td><td>0.1382</td><td>7.539 / 7.917</td><td>6.227 / 8.712</td><td>39.79 / 56.00</td><td>50% suppression</td></tr><tr><td>Shorter chunks</td><td>0.9259</td><td>0.1551</td><td>7.563 / 7.843</td><td>6.337 /8.615</td><td>45.46 / 57.05</td><td>two latent frames</td></tr><tr><td>All-block coverage</td><td>0.9325</td><td>0.1404</td><td>7.664 /7.819</td><td>6.328 / 8.600</td><td>40.55 / 55.55</td><td>all seven blocks</td></tr></table>

Table 27: Cross-audio guidance scan on the pilot subset. Mouth $\Delta _ { 1 }$ is measured against CFG 1 and is not a reconstruction-error metric.
<table><tr><td>CFG</td><td>Sync-C↑</td><td>Sync-D↓</td><td>|offset|</td><td>Mouth  $\Delta _ { 1 }$ </td></tr><tr><td>0.0</td><td>1.878</td><td>13.260</td><td>10.19</td><td>9.04</td></tr><tr><td>1.0</td><td>6.181</td><td>8.758</td><td>0.00</td><td>0.00</td></tr><tr><td>1.5</td><td>7.245</td><td>7.915</td><td>0.06</td><td>3.82</td></tr><tr><td>2.0</td><td>7.607</td><td>7.638</td><td>0.07</td><td>6.03</td></tr><tr><td>3.0</td><td>7.889</td><td>7.536</td><td>0.00</td><td>9.69</td></tr></table>

![](images/a33556121f233ada4f87d72f4bf0eabf44c548dfbfc0106e750aa77dc65c2936.jpg)  
Figure 9: Full-context distillation on dev-256. Trajectory regression transfers cross-audio synchronization to the two-step student, while DMD improves distribution quality.

## H.3 Full-Context Distillation Path

Figure 9 reports the stage-wise teacher, trajectory regression, and DMD results. Trajectory regression retains the teacher’s cross-audio Sync-C, while distribution matching restores the teacher-level FVD.

## H.4 Asymmetric DMD implementation

Our implementation follows DMD2 and Causal Forcing [49, 60]. It contains a frozen bidirectional real score, an online bidirectional fake score, five fake-score updates per generator update by default, and one sampled active block retaining generator gradients. The objective is evaluated on the selected branch’s complete final rollout (two steps for the submitted model; four only for the diagnostic branch). A frozen initializer with paired noise provides an anchor consistency target.

For each fake-score update, $\sigma \sim \mathcal { U } [ 0 . 0 2 , 0 . 9 8 ]$ and the online velocity model regresses ϵ − z<sub>student</sub> on a noised, detached student rollout. The real score is the frozen bidirectional teacher. At the generator update, both scores parameterize clean estimates $\hat { z } _ { r } = x _ { \sigma } - \sigma v _ { r }$ and $\hat { z } _ { f } = x _ { \sigma } - \sigma v _ { f }$ If $P _ { b }$ is one on one uniformly sampled latent block and zero elsewhere, the stop-gradient direction is

$$
d _ { b } = \frac { P _ { b } \odot ( \hat { z } _ { f } - \hat { z } _ { r } ) } { \operatorname* { m a x } ( \operatorname* { m e a n } _ { P _ { b } } | z - \hat { z } _ { r } | , 1 0 ^ { - 6 } ) } .\tag{13}
$$

$P _ { b }$ has one channel and broadcasts over the latent channels. The denominator and score direction are computed with-

Table 28: Completed development-set causal-adaptation results. FVD uses the same I3D16 protocol for every row.
<table><tr><td>Method</td><td>SSIM↑</td><td>Mouth LPIPS↓</td><td>Same C↑</td><td>Cross C↑</td><td>FVD same↓</td><td>FVD cross↓</td></tr><tr><td>Full-context parent</td><td>0.951154</td><td>0.096476</td><td>8.505424</td><td>8.050768</td><td>6.676098</td><td>11.701082</td></tr><tr><td>A0 Flow</td><td>0.939031</td><td>0.126039</td><td>7.628839</td><td>6.194756</td><td>8.257593</td><td>12.564611</td></tr><tr><td>A1 Flow+Q</td><td>0.938839</td><td>0.125288</td><td>7.631446</td><td>6.158610</td><td>8.078579</td><td>12.433051</td></tr><tr><td>A2 Flow+Sync</td><td>0.939122</td><td>0.125797</td><td>7.796448</td><td>6.468818</td><td>7.990434</td><td>12.506445</td></tr><tr><td>A3 Flow+Q+Sync</td><td>0.938953</td><td>0.124887</td><td>7.776736</td><td>6.450379</td><td>7.767604</td><td>12.377647</td></tr><tr><td>A4 + representation alignment</td><td>0.938046</td><td>0.126844</td><td>7.829486</td><td>6.560639</td><td>7.800708</td><td>12.940533</td></tr><tr><td>R0 GT history</td><td>0.939363</td><td>0.123910</td><td>7.755741</td><td>6.390372</td><td>7.706146</td><td>12.012290</td></tr><tr><td>R1 resample 25%</td><td>0.939077</td><td>0.124809</td><td>7.757215</td><td>6.386939</td><td>7.676899</td><td>12.010987</td></tr><tr><td>R2 resample 50%</td><td>0.938943</td><td>0.125203</td><td>7.766012</td><td>6.404347</td><td>7.666487</td><td>11.945611</td></tr><tr><td>R3 rollout 25%</td><td>0.938248</td><td>0.126147</td><td>7.716166</td><td>6.388346</td><td>7.314490</td><td>11.618155</td></tr></table>

Table 29: Pilot-subset four-step consistency results. CD0 is pure causal consistency; CD1 adds calibrated valid GT/Sync auxiliaries. The rows compare execution and quality at the calibrated pilot settings.
<table><tr><td>Model</td><td>Updates</td><td>SSIM↑</td><td>Mouth LPIPS↓</td><td>Same C↑</td><td>Cross C↑</td><td>Cross D↓</td><td>FVD S/C↓</td></tr><tr><td>CF re-noise parent</td><td>0</td><td>0.94580</td><td>0.11099</td><td>7.923</td><td>6.418</td><td>8.622</td><td>30.82 / 48.62</td></tr><tr><td>CD0 raw</td><td>200</td><td>0.94409</td><td>0.11232</td><td>7.930</td><td>6.395</td><td>8.612</td><td>31.68 / 49.41</td></tr><tr><td>CD0 EMA</td><td>200</td><td>0.94496</td><td>0.11201</td><td>7.939</td><td>6.333</td><td>8.663</td><td>31.97 / 49.10</td></tr><tr><td>CD1 raw</td><td>200</td><td>0.94379</td><td>0.11197</td><td>7.965</td><td>6.325</td><td>8.670</td><td>31.40 / 49.35</td></tr><tr><td>CD1 EMA</td><td>200</td><td>0.94538</td><td>0.11176</td><td>7.915</td><td>6.346</td><td>8.658</td><td>32.16 / 50.31</td></tr></table>

out gradients. The generator surrogate is ${ \frac { 1 } { 2 } } \| z - \operatorname { s g } ( z \ - \quad$ $d _ { b } ) \| _ { 1 + 2 M } ^ { 2 }$

The initializer-consistency loss compares the selected student block to the frozen two-step initializer’s rollout under identical input noise and schedule:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e t } } = \| P _ { b } \odot \left( z _ { \mathrm { s t u d e n t } } - z _ { \mathrm { a n c h o r } } \right) \| _ { 1 + 2 M } ^ { 2 } . } \end{array}\tag{14}
$$

Its coefficient is 0.10. DMD quality has $\begin{array} { l l } { \mathcal { L } _ { q , \mathrm { D M D } } ^ { s } } & { = } \end{array}$ $. 0 2 \mathcal { L } _ { \mathrm { p i x } } + . 1 0 \mathcal { L } _ { \mathrm { L P I P S } } + . 1 0 \mathcal { L } _ { \mathrm { k e e p } }$ and $\mathcal { L } _ { q , \mathrm { D M D } } ^ { c } = . 1 0 \mathcal { L } _ { \mathrm { k e e p } } ^ { \bar { M } } ,$ both scaled by 0.9879127. Thus replacement DMD disables source-mouth pixel/LPIPS and uses no training Sync loss; it retains only valid non-mouth reconstruction.

During training, only the real score uses conditional and null-audio forwards:

$$
s _ { r } ^ { ( g ) } = s _ { r } ( x _ { t } , \emptyset ) + g \left[ s _ { r } ( x _ { t } , a ) - s _ { r } ( x _ { t } , \emptyset ) \right] .\tag{15}
$$

The student rollout uses one conditional forward per denoising step. The fake score learns the student distribution rather than guided teacher outputs. Every reported DMD model is evaluated with inference audio CFG equal to one.

Relative to an unguided real score $\begin{array} { r l r } { ( g } & { { } = } & { 1 ) } \end{array}$ , the complete $g \ = \ 2$ model raises cross Sync-C by 0.2134 (95% CI [0.1896,0.2372]), while changing SSIM by - 0.0020 ([-0.0022,-0.0019]) and mouth LPIPS by +0.0045 ([0.0041,0.0049]). Thus teacher guidance strengthens speech control at a measurable visual cost. Relative to no retention, cross Sync-C changes by -0.0025 ([-0.0128,0.0080]), which is unresolved, while mouth LPIPS changes by -0.0002 ([- 0.0003,-0.0001]); the resolved effect is small. Removing replacement audio from AR, trajectories, and DMD lowers cross Sync-C by 0.1533 ([0.1315,0.1767]) with SSIM -0.0016 ([-0.0017,-0.0014]). FVD values remain aggregate point estimates and are not assigned per-video paired confidence intervals.

## H.5 Matched guidance factorial and targetaudio response

Table 34 completes the predeclared trajectory-guidance by DMD-guidance factorial. All four cells use the same SFT teacher, causal parent architecture, training rows, update schedule, two-step inference, evaluator, and retention weight; the CFG-1 trajectory branch uses its own ODE parent and parent-matched fake score. Both guidance locations improve Sync-C. DMD guidance trades this gain for higher Sync-D and FVD, whereas trajectory guidance also improves Sync-D. The small Sync-C interaction (0.014) indicates approximately additive effects under this training configuration.

For optimization variability, Table 35 compares the complete recipe with a matched in-repository closest-recipe control: F11 with retention disabled. This control is not claimed to reproduce the public Lip Forcing implementation. Each seed is evaluated on the same source-disjoint 64-source subset. Source-bootstrap uncertainty and training-seed variation are distinct; the table reports the latter directly.

Table 36 fixes source video, editable mask, reference, and noise while changing only target audio. “Unrelated” is a deterministic derangement of the replacement utterances; “null

Table 30: Four-step guidance absorption and the unified two/four-step study. All DMD models use real-teacher guidance only during training and one inference branch. Candidates use the same stage-specific parents and shared fake-score initialization within each step-count branch for controlled comparison.
<table><tr><td>Study</td><td>NFE</td><td>Real g</td><td>Branches</td><td>SSIM</td><td>Mouth LPIPS</td><td>Cross C/D</td><td>FVD S/C</td></tr><tr><td>AR parent</td><td>50</td><td>一</td><td>1</td><td>0.9388</td><td>0.1261</td><td>7.951 / 7.603</td><td>9.19 / 13.99</td></tr><tr><td>Four-step ODE</td><td>4</td><td></td><td>1</td><td>0.9326</td><td>0.1403</td><td>8.033 / 7.562</td><td>10.38 / 14.45</td></tr><tr><td>Four-step DMD</td><td>4</td><td>2.0</td><td>1</td><td>0.9377</td><td>0.1288</td><td>8.547 / 7.264</td><td>9.13 / 14.30</td></tr><tr><td>Unified ODE/DMD</td><td>2,4</td><td>2.0</td><td>1</td><td colspan="4">selected under common quality, sync, and failure criteria</td></tr></table>

Table 31: Matched trajectory-guidance control on the 256-source development manifest. Only teacher trajectory CFG changes; both two-step ODE students use one conditional inference branch with CFG 1. Brackets are 95% paired source-bootstrap intervals; Sync uses the 255-source common valid set. FVD is an aggregate point estimate.
<table><tr><td>Trajectory teacher</td><td>Teacher CFG</td><td>Sync-C↑</td><td>Sync-D↓</td><td>Keep L1↓</td><td>FVD↓</td></tr><tr><td>Unguided trajectories</td><td>1</td><td>8.302 [8.159,8.449]</td><td>7.135 [7.035,7.236]</td><td>1.996 [1.952,2.042]</td><td>10.52</td></tr><tr><td>Guided trajectories</td><td>2</td><td>8.641 [8.499,8.788]</td><td>7.069 [6.971,7.168]</td><td>1.974 [1.931,2.020]</td><td>11.48</td></tr></table>

Table 32: Exact values for the sequential causal path. Replacement-audio adaptation precedes two-branch guidance. This causal autoregressive protocol is distinct from the full-context ODE-2 trajectory initializer in the main-paper ablation.
<table><tr><td>Stage</td><td></td><td>Br. SSIM↑ Sync-C↑ FVD↓</td><td></td></tr><tr><td>Teacher</td><td>2 0.9476</td><td>8.751</td><td>6.66</td></tr><tr><td>Direct causalization</td><td>1 0.9387</td><td>7.993</td><td>8.21</td></tr><tr><td>+ replacement audio</td><td>1 0.9390</td><td>8.121</td><td>8.33</td></tr><tr><td>+ guidance</td><td>2 0.9369</td><td>8.726</td><td>8.91</td></tr></table>

T/S” remuxes the learned-null output with the designated target/source waveform for evaluation. Correct and unrelated speech retain high synchronization with the waveform that actually drove generation. Learned-null outputs instead align more with source than target, rejecting a generic mouthmotion explanation. Keep-region SSIM remains nearly unchanged, while the complete model’s FVD cost relative to the matched control remains explicit.

## H.6 Controlled AR–ODE–DMD evaluation protocol

The causal parent is frozen after development-set selection. All two- and four-step trajectory or consistency candidates share this checkpoint, and each distribution- matching branch starts from one selected initializer with a branchspecific shared fake score. Table 37 records the ablations and stopping rules fixed before final evaluation.

## I Evaluation Protocol

Synchronization. We use the official S3FD [52]/SyncNetv2 [7, 8] path and report Sync-C, Sync-D, estimated offset, tracking failures, and the number of outputs used by each mean. Same-audio and cross-audio scores are never pooled. Temporal-shift calibration. For 40 VidDub-Bench crossaudio outputs per method, we shift the waveform by k ∈ $\{ 0 , \pm 1 , \pm 2 , \pm 3 , \pm 5 \}$ } video frames and rerun the same evaluator. The estimated offset changes by exactly −k throughout $| k | \le 3$ for every method. At zero shift, the Causal two-step VidDub, its teacher, and LatentSync return offset zero on all 40 clips. This is the temporal-shift test summarized in the main paper.

Independent visual speech recognition. Driving utterances are transcribed with Whisper-large-v3 [33], while generated video is read by the visual-only LRS3 model of Ma et al. [25]; this reader shares no weights or objective with either the frozen training synchronizer or SyncNetv2. The final benchmark uses the 30/28/121 English HDTF/VoxCeleb2/VidDub-Bench clips reported in the main paper. The three-seed factorial uses the 212 clips valid across all 27 outputs: masking source-mouth reconstruction under unpaired audio lowers WER by $0 . 1 4 4 \pm 0 . 0 0 9$ and CER by $0 . 1 1 6 \pm 0 . 0 0 7$ , with the favorable direction preserved for every seed and all 12 controlled cell comparisons.

Reconstruction and preservation. SSIM [44], PSNR, mouth LPIPS [51], mouth L1, and outside-mask preservation are valid for same-audio reconstruction. For cross audio we do not compare the generated mouth with the source mouth. FVD [42] with I3D features [2] and FID [15] are reported for both conditions using identical feature extraction and window grouping within each table.

For cross-audio source preservation, corresponding source frames remain valid ground truth outside the editable lower face. We derive one stable source face crop per frame and use MediaPipe Face Mesh [20] source landmarks to set a horizontal boundary eight pixels above the topmost lip landmark. Upper-L1 is the mean RGB error above this boundary, thereby excluding the full lower face and paste-back blend region. ArcFace cosine [11] and InsightFace 3D-landmark pitch/yaw/roll error [18] are evaluated every fourth frame using the same source-derived crops. All methods share frame indices, crops, and source targets; confidence intervals bootstrap source identities.

Table 33: Ablation of guidance distillation, initializer consistency, and replacement audio under the final two-step 256-source development protocol. Guidance and initializer consistency each change one field from the complete model; no-replacementaudio retrains AR→ODE→DMD with $p _ { \mathrm { c r o s s } } = 0$ . Inference remains one conditional forward per denoising step with CFG 1.
<table><tr><td>Variant</td><td>Real g</td><td>Init. Consistency</td><td>SSIM↑</td><td>Mouth LPIPS↓</td><td>Cross C/D</td><td>FVD S/C↓</td></tr><tr><td>Complete causal DMD</td><td>2.0</td><td>0.1</td><td>0.9382</td><td>0.1348</td><td>8.731 / 7.022</td><td>6.86 / 7.76</td></tr><tr><td>Unguided real score  $( g = 1 )$ </td><td>1.0</td><td>0.1</td><td>0.9402</td><td>0.1303</td><td>8.517 / 6.953</td><td>5.26 / 7.00</td></tr><tr><td>No ODE retention</td><td>2.0</td><td>0</td><td>0.9381</td><td>0.1350</td><td>8.733 / 7.018</td><td>6.83 / 7.69</td></tr><tr><td>No replacement audio</td><td>2.0</td><td>0.1</td><td>0.9398</td><td>0.1305</td><td>8.577 / 7.080</td><td>6.09 / 7.15</td></tr></table>

Table 34: Complete trajectory-guidance × DMD-guidance factorial at generator update 50 on the frozen 256-source development set. Each cell scores 512/512 same/cross outputs; inference always uses one audio branch with CFG 1.
<table><tr><td>Cell</td><td>gtraj</td><td>gDMD</td><td>Retention</td><td> $\mathrm { C r o s s ~ S y n c { \cdot } C \uparrow }$ </td><td> $\mathrm { C r o s s \thinspace S y n c { \scriptscriptstyle - } D \downarrow }$ </td><td>Cross FVD↓</td></tr><tr><td>F11</td><td>1</td><td>1</td><td>0.1</td><td>8.371</td><td>7.070</td><td>7.21</td></tr><tr><td>F12</td><td>1</td><td>2</td><td>0.1</td><td>8.578</td><td>7.120</td><td>7.66</td></tr><tr><td>F21</td><td>2</td><td>1</td><td>0.1</td><td>8.512</td><td>6.954</td><td>7.26</td></tr><tr><td>F22 (complete)</td><td>2</td><td>2</td><td>0.1</td><td>8.732</td><td>7.023</td><td>7.74</td></tr></table>

![](images/f331eed6716a62256b99348f675b9b81bcba681f2e1b5a9dca151c12f56b1f24.jpg)

![](images/d91def44555ff5c1d767bcc725006c6a9ec0c88bb3b9a0ddacfbc5ed6d299d16.jpg)

![](images/37a696ce1d971c1d40a74940e1622560637f9b210fd75cae04b3a3a686e8cd68.jpg)

Earlier continuous-cache test  
![](images/fc5848b7827c422a8e85cdf35faec29920910382e6c4ea50308de9621140f056.jpg)  
Figure 10: The final two-step model improves audio–visual synchronization, while the bidirectional models define the reported FVD frontier. The 10/30/60-second persistent-cache comparison motivates the overlapping-window inference used by the final model for stable long-video generation.

Table 35: Three-seed endpoint variability at update 50 (mean±sample SD); 128/128 same/cross outputs score for every seed and condition.
<table><tr><td>Endpoint</td><td>Cross Sync-C↑</td><td>Cross Sync-D↓</td><td>Cross FVD↓</td></tr><tr><td>Matched control P1</td><td> $^ { 8 . 2 5 4 } _ { 8 . 5 9 3 } \pm 0 . 0 1 2$ </td><td>7.176 ± 0.008</td><td> $2 0 . 7 9 \pm 0 . 0 5$ </td></tr><tr><td>Complete P4</td><td></td><td> $7 . 1 4 5 \pm 0 . 0 1 7$ </td><td> $2 2 . 6 6 \pm 0 . 2 0$ </td></tr></table>

Temporal behavior and efficiency. We measure mouth temporal variation, block-boundary variation, stable-face color drift, denoising and end-to-end latency, time to first block, peak memory, throughput, and real-time factor. Longvideo tests use 10, 30, and 60-second inputs and report cache growth, synchronization offset, and identity drift over time. Compute accounting. As in the main paper, NFE counts denoising steps only. The Causal two-step VidDub additionally performs one clean state-commit forward per block; timing tables include that forward unless a prepared-clip scope is explicitly stated.

Same-hardware wall clock. Every method runs alone on one GPU of the same idle NVIDIA H20 host, from the raw source video and waveform to the written output, on five

Table 36: Fixed-noise target-audio intervention on 32 development sources. T/S denote target/source-audio remuxes of identical learned-null pixels under each controlled intervention.
<table><tr><td>Model</td><td>Correct C↑</td><td>Shifted C↑</td><td>Unrelated C↑</td><td>Null T C↑</td><td>Null S C↑</td><td>Keep SSIM↑</td><td>Correct FVD↓</td></tr><tr><td>48-step teacher</td><td>8.112</td><td>7.584</td><td>8.032</td><td>1.813</td><td>3.499</td><td>0.97994</td><td>28.96</td></tr><tr><td>Matched control P1</td><td>8.262</td><td>7.706</td><td>8.236</td><td>1.606</td><td>3.168</td><td>0.97837</td><td>35.41</td></tr><tr><td>F21 (2, 1)</td><td>8.431</td><td>7.916</td><td>8.370</td><td>1.583</td><td>3.140</td><td>0.97848</td><td>35.38</td></tr><tr><td>F22 (2, 2)</td><td>8.625</td><td>8.183</td><td>8.592</td><td>1.639</td><td>3.281</td><td>0.97845</td><td>39.83</td></tr></table>

Table 37: Stage-gated causal conversion experiments. Each row changes one factor relative to the selected parent for that stage.
<table><tr><td>Stage</td><td>Variants</td><td>Question</td><td>Selection criterion</td></tr><tr><td>AR coverage</td><td>A3, B3, B7</td><td>Does supervision cover temporal/audio positions?</td><td>better mouth LPIPS and FVD without cross-sync loss</td></tr><tr><td>AR audio</td><td>B7, +S08, +cross 5/10%, warmup</td><td>Can audio evidence increase without X30 artifacts?</td><td>cross C/D improve; visual tolerance passes</td></tr><tr><td>AR history</td><td>GT, suppression, resample, audio-consistent rollout</td><td>How does history construction affect causal generation?</td><td>long-horizon and cross metrics improve jointly</td></tr><tr><td>Initializer</td><td>Euler4, re-noise4, ODE, CD0, CD1</td><td>Which four-step objective retains the AR parent?</td><td>joint development quality/sync Pareto frontier</td></tr><tr><td>DMD guidance</td><td>real  $g = 1 , 1 . 5 , 2$ </td><td>Can teacher CFG be absorbed into one branch?</td><td>cross C/D improve at inference CFG 1</td></tr><tr><td>DMD initializer consistency</td><td>none, ODÉ anchor, +quality, +cross keep/Sync</td><td>What prevents color and mouth artifacts?</td><td>FVD/LPIPS/color improve without synchronization loss</td></tr></table>

Table 38: Cross-audio source preservation on the three frozen evaluation sets. Upper-L1 is reported in percentage points; lower Upper-L1 and pose error and higher identity cosine are better. LatentSync has 254/256 VidDub-Bench outputs; all other requested outputs are present for evaluation.
<table><tr><td>Benchmark / method</td><td>Upper-L1↓</td><td>ID cosine↑</td><td>Pose error (°)↓</td><td>n</td></tr><tr><td>HDTF / Wav2Lip</td><td>2.580</td><td>0.928</td><td>0.432</td><td>30</td></tr><tr><td>HDTF / Diff2Lip</td><td>1.753</td><td>0.888</td><td>0.464</td><td>30</td></tr><tr><td>HDTF / LatentSync</td><td>1.646</td><td>0.970</td><td>0.228</td><td>30</td></tr><tr><td>HDTF / Lip Forcing</td><td>1.669</td><td>0.934</td><td>0.340</td><td>30</td></tr><tr><td>HDTF / Teacher</td><td>0.873</td><td>0.911</td><td>0.415</td><td>30</td></tr><tr><td>HDTF / Full-context 2-step</td><td>0.914</td><td>0.902</td><td>0.449</td><td>30</td></tr><tr><td>VOX2 / Wav2Lip</td><td>1.697</td><td>0.913</td><td>0.573</td><td>30</td></tr><tr><td>VOX2 / Diff2Lip</td><td>1.065</td><td>0.893</td><td>0.619</td><td>30</td></tr><tr><td>VOX2 / LatentSync</td><td>1.009</td><td>0.951</td><td>0.560</td><td>30</td></tr><tr><td>VOX2 / Lip Forcing</td><td>0.995</td><td>0.905</td><td>0.553</td><td>30</td></tr><tr><td>VOX2 / Teacher</td><td>0.912</td><td>0.877</td><td>0.629</td><td>30</td></tr><tr><td>VOX2 / Full-context 2-step</td><td>0.931</td><td>0.877</td><td>0.671</td><td>30</td></tr><tr><td>CAUSALDUB / Wav2Lip</td><td>1.820</td><td>0.909</td><td>0.459</td><td>256</td></tr><tr><td>CAUSALDUB / Diff2Lip</td><td>1.283</td><td>0.862</td><td>0.518</td><td>256</td></tr><tr><td>CAUSALDUB / LatentSync</td><td>1.605</td><td>0.913</td><td>0.422</td><td>254</td></tr><tr><td>CAUSALDUB / Lip Forcing</td><td>3.761</td><td>0.900</td><td>0.928</td><td>256</td></tr><tr><td>CAUSALDUB / Teacher</td><td>1.207</td><td>0.872</td><td>0.573</td><td>256</td></tr><tr><td>CAUSALDUB / Full-context 2-step</td><td>1.199</td><td>0.875</td><td>0.579</td><td>256</td></tr></table>

![](images/c43c46b131537333ce1bc52ac08432970e7395abfe12da0b92f63be79bac2307.jpg)  
Figure 11: Long-video execution design. The persistentcache run motivates overlapping windows conditioned on source frames by jointly comparing runtime, memory, synchronization, and lower-face stability over 10–60 seconds.

HDTF sources cut to 4 s and 8 s (one warm-up call excluded). The Causal two-step VidDub and LatentSync run in one process with weights resident; Wav2Lip and Lip Forcing run through their official command-line inference, which reloads weights and re-detects faces on every call. We therefore fit wall time = fixed + slope × video seconds and compare the slope (Table 39). The Causal two-step VidDub includes the face pre-scan, Whisper, VAE encode/decode, and all three transformer forwards per block. This protocol measures reproducible public pipelines with each method’s released process orchestration; the slope summarizes marginal runtime over the same clip sequence.

<sup>†</sup>The released teacher exposes prepared-clip inference. On the same H20, one timed 81-frame run (after warmup) takes 67.74 s including condition preparation, 30-step bidirectional denoising, and VAE decoding, corresponding to 1.20 FPS; face detection, source encoding, paste-back, and MP4 encoding are excluded.

Statistics. Per-video comparisons are paired by source using 10,000 bootstrap resamples. Confidence intervals bootstrap source identities. For FVD, all temporal feature windows from one source are resampled as a group. The pilot subset is diagnostic only; VidDub-Bench is evaluated once after the selection recipe is frozen.

Table 39: Same-hardware wall clock on one H20 (median of five sources). Slope is the marginal compute per second of video; fixed cost includes per-call weight loading for the command-line baselines.
<table><tr><td>Method</td><td>Slope (s/s)↓</td><td>Fixed (s)</td><td>4 s clip</td><td>8s clip</td></tr><tr><td>Wav2Lip (CLI) [31]</td><td>4.77</td><td>137.1</td><td>162.3</td><td>170.4</td></tr><tr><td>VideoReTalking (CLI) [5]</td><td>19.16</td><td>90.6</td><td>167.2</td><td>243.8</td></tr><tr><td>MuseTalk 1.5 (CLI) [41, 54]</td><td>22.02</td><td>-21.0</td><td>67.1</td><td>155.2</td></tr><tr><td>Diff2Lip (CLI) [28]</td><td>12.55</td><td>5.2</td><td>55.3</td><td>105.5</td></tr><tr><td>LatentSync 1.6 [1, 21]</td><td>11.38</td><td>13.5</td><td>59.3</td><td>103.5</td></tr><tr><td>X-Dub (CLI) [14]</td><td>56.11</td><td>236.4</td><td>460.8</td><td>685.2</td></tr><tr><td>Lip Forcing 14B (CLI) [6]</td><td>10.44</td><td>155.1</td><td>197.6</td><td>237.5</td></tr><tr><td>Full-context teacher† Causal two-step VidDub</td><td>3.97</td><td>14.6</td><td>30.1</td><td>46.2</td></tr></table>

## J Long-20 External Comparison

All methods receive the same 20 frozen 60-second CFR source videos and replacement waveforms. Wav2Lip uses the official GAN checkpoint; LatentSync 1.6 uses its official 20-step, guidance-1.5 inference. VidDub uses 81-frame overlapping windows conditioned on source frames, a 64- frame stride, and 17-frame overlap rather than one persistent rollout. We require full video/audio decode, score twelve non-overlapping 5-second SyncNet windows, and count failures rather than dropping them. For source preservation, fullframe outputs are transformed by the exact source-derived face crop before corresponding-frame comparison. Keep-L1 excludes the editable lower face; stable-face chroma is CIELAB ab displacement [9]. Horizon changes use complete 5-second bins, excluding codec-induced one-frame terminal bins.

VidDub completes all 20 sequences with the highest mean and final synchronization. Its near-zero slope, preservedregion change, and stable-face chroma show that the overlapping full-context windows maintain both speech synchronization and surrounding visual evidence throughout the minute.

## K Benchmark Suite and Baseline Protocol

Table 44 gives the complete VoxCeleb2-30 comparison that is not tabulated in the main paper. Table 45 reproduces the complete main-paper TalkVid-Bench comparison for the supplementary record. Params include every neural module loaded and used by the generation path (generative backbone, conditioning modules, audio encoder, and decoder/restorer), but exclude generic face detectors and geometric post-processing. VidDub uses a 1.3B Wan backbone; 2.441B denotes the adapted generator components, whereas 2.81B is the complete inference stack reported in benchmark tables. It uses a cached empty-text embedding, so the inactive T5 encoder is not counted.

The main table reports means over each method’s valid outputs and generation coverage, whereas cross-method ranking uses the same 245 clips for every method. On this shared subset, the Causal two-step VidDub obtains 8.669 Sync-C versus 7.588 for the strongest external method and 8.707 for its full-context teacher, preserving 99.6% of teacher Sync-C.

The common-intersection conclusion agrees with the method-specific audit. VidDub reduces FVD from 14.24 to 12.06 relative to the matched causal variant, while the causal variant has a 0.070 Sync-C advantage on this subset. Across methods, moving from each native successful subset to the strict common set changes mean Sync-C by at most 0.086; failed outputs remain explicit and are never recast as scored examples.

All methods use the same decoding, face-crop, synchronization, and feature-extraction protocol when released checkpoints permit. We report Wav2Lip, VideoReTalking, Diff2Lip, MuseTalk 1.5, LatentSync 1.6, public X-Dub, Lip Forcing 14B and its OmniAvatar-LS 1.3B/14B variants [6], our bidirectional SFT, causal AR, selected two/four-step initializers, and final VidDub. If a public method cannot support cross audio, the evaluated long-sequence protocol, or the same resolution, the missing capability is marked instead of silently changing the protocol. OmniAvatar-LS receives a fixed reference frame extracted from each source and is evaluated as an avatar-generation baseline. Because it is not conditioned on the complete source video, its SSIM, mouth LPIPS, and source-motion retention are marked not applicable rather than compared with editing methods.

Editing baselines receive the complete source video and target waveform under their released preprocessing. The full-context parent uses 30 or 48 denoising steps, whereas VidDub uses two denoising steps and one conditional forward at each step. The four-NFE causal diagnostic in Table 47 is not the matched Causal two-step VidDub evaluated at two NFE in Tables 12 and 48. OmniAvatar-LS receives only one reference image and is therefore excluded from source-motion preservation comparisons.

On this frozen 50-pair set, the Causal two-step VidDub and the two-branch full-context SFT parent evaluated at 48 NFE obtain 8.772 and 8.816 Sync-C, respectively, with their corresponding FVD reported in the same table. Against the closest public causal few-step baseline, Lip Forcing 14B, the Causal two-step VidDub uses the same reported NFE and one conditional branch while improving Sync-C from 3.973 to 8.772; it additionally executes one clean state-commit forward per block, while Lip Forcing obtains the lower FVD. The Causal two-step VidDub also improves both Sync-C and FVD over LatentSync on this subset; the result establishes the reported synchronization ordering under the evaluated inference protocols.

Table 40: Corrected Long-20 results (means over 20 paired sequences). First/last are 5-second Sync-C windows; slope is Sync-C change per minute. ∆ metrics are last-minus-first complete-bin changes. Runtime is omitted because public pipeline timing scopes are not uniform across methods in this comparison.
<table><tr><td>Method</td><td>Valid</td><td>Sync-C↑</td><td>Sync-D↓</td><td>First C</td><td>Last C</td><td>Slope/min</td><td>Keep-L1↓</td><td>Chroma↓</td><td>∆Keep</td><td>∆Chroma</td></tr><tr><td>Wav2Lip [31]</td><td>20/20</td><td>7.861</td><td>7.484</td><td>8.147</td><td>8.032</td><td>-0.146</td><td>3.045</td><td>1.045</td><td>+0.009</td><td>+0.003</td></tr><tr><td>LatentSync 1.6 [21]</td><td>20/20</td><td>7.771</td><td>7.083</td><td>8.029</td><td>7.844</td><td>-0.112</td><td>4.018</td><td>0.397</td><td>+0.685</td><td>+0.006</td></tr><tr><td>VidDub</td><td>20/20</td><td>9.399</td><td>6.162</td><td>9.755</td><td>9.585</td><td>-0.046</td><td>2.493</td><td>0.250</td><td>-0.034</td><td>+0.006</td></tr></table>

Table 41: Source re-anchoring. Pair-14 compares persistent execution with 81-frame source re-anchoring and 17-frame overlap.
<table><tr><td>Metric</td><td></td><td>Persistent Re-anchored</td></tr><tr><td>Sync-C ↑</td><td>9.578</td><td>9.747</td></tr><tr><td>Sync-D ↓</td><td>6.772</td><td>6.106</td></tr><tr><td>Peak memory (GiB) ↓</td><td>23.10</td><td>22.38</td></tr><tr><td>Skin chroma MAE ↓</td><td>7.48</td><td>2.43</td></tr><tr><td>Lower-face stability</td><td>no</td><td>yes</td></tr></table>

The main benchmark figure summarizes cross Sync-C/FVD on three public benchmarks. Its 10/30/60-second curves document the persistent-cache architecture comparison; the final Long-20 evaluation uses the overlappingwindow inference reported in Table 40.

## L Qualitative Evaluation and Artifact Taxonomy

The following twenty figures cover ten videos: four uniformly spaced complete cases from HDTF-30 and VoxCeleb2-30, and two complete VidDub-Bench cases. Every case has separate same- and cross-audio tracks and seven synchronized temporal samples per method. Publicbenchmark figures compare the source, seven external editors, the full-context parent, and Causal two-step VidDub; VidDub-Bench figures retain the source, LatentSync, parent, and Causal two-step VidDub for readability. Crossaudio panels additionally show the target waveform and fixed acoustic time markers. Selection manifests and seeds accompany all figures to prevent frame- or case-level cherrypicking.

The failure taxonomy includes weak articulation, exaggerated mouth motion, teeth or tongue flicker, boundary discontinuity, color drift, identity drift, and accumulated audio offset. Audible videos, tracking failures, and cases where metric ordering disagrees with visual judgment are included in the supplementary-video package.

## M Blinded Human Evaluation

This completed study evaluates the final full-context Vid-Dub model. Fifteen bilingual participants each completed all 150 anonymized A/B trials covering 50 generated sources and three opponents. Each trial asks which video is better in lip synchronization, mouth realism, identity preservation, temporal stability, and overall quality. Left/right assignment and the order of the first four criteria are randomized; overall quality is always shown last. The interface exposes neither method identity nor metrics. Table 49 reports VidDub/opponent/tie outcomes for each opponent and criterion, with ties retained.

## N Reproducibility Checklist

The reproducibility package is defined by frozen split manifests, preprocessing thresholds, exact model and code revisions, checkpoint hashes, optimizer and EMA state, inference schedules, metric versions, failed-sample accounting, and commands used to generate every table and supplementary video. Archived records include training and DMD configurations, paired-bootstrap outputs, final-evaluation protocols, and figure-selection manifests. Training completion by itself is never treated as evidence of quality gain.

## O Ethical Considerations

High-fidelity lip synchronization can support translation, accessibility, and post-production, but can also facilitate impersonation and misattributed speech. VidDub edits a source video and can make a real person appear to say words they never said. We use only public research benchmarks and generated H3 subjects for figures; no private recordings of identifiable people are released. We will not redistribute source media whose licenses or consent basis do not permit redistribution, and will publish a dataset/model card documenting provenance, permitted use, and a removal-request channel before releasing artifacts. Model access will require a use agreement forbidding non-consensual impersonation; released outputs will include provenance metadata, and detector baselines will accompany the model. The blinded human study uses only generated H3 subjects, so no real person’s likeness is edited for it; raters are adult volunteers who may stop at any time, and nothing beyond an anonymous rater ID is stored.

Table 42: Benchmark construction and VidDub execution. Duration/FPS describes the native source clip scored by each benchmark, not the model-call tensor. Every VidDub call uses an 81-frame, 24-FPS, 512×512 window; longer source clips use shifted-tail windows, and Long-20 uses 17-frame overlap.
<table><tr><td>Benchmark</td><td>Sources</td><td>Source duration / native FPS</td><td>Tracks</td><td>Cross pairing</td><td>VidDub execution</td></tr><tr><td>HDTF-30</td><td>30</td><td>8.0s / 25</td><td>same + cross</td><td>fixed cyclic, different speaker</td><td>81-frame windows, shifted tail</td></tr><tr><td>VoxCeleb2-30</td><td>30</td><td>4.0s / 25</td><td>same + cross</td><td>fixed cyclic, different speaker</td><td>81-frame windows, shifted tail</td></tr><tr><td>TalkVid-Bench</td><td>500</td><td>5.042s / 24</td><td>same + cross</td><td>deterministic within stratum</td><td>81-frame windows, shifted tail</td></tr><tr><td>VidDub-Bench</td><td>256</td><td>3.375s / 24</td><td>same + cross</td><td>fixed multilingual pairs</td><td>one 81-frame full-context window</td></tr><tr><td>Frozen-50</td><td>50</td><td>3.375s / 24</td><td>cross</td><td>identical clips for compared methods</td><td>matched blinded comparison</td></tr><tr><td>Long-20</td><td>20</td><td>60s / 25</td><td>cross primary</td><td>frozen contiguous pairs</td><td>81-frame windows, 17-frame overlap</td></tr></table>

Table 43: VidDub-Bench cross-audio failure and common-intersection audit. Every method is requested to generate 256 clips. “Scoreable” requires an output, a successful SyncNet record, and matching tables/GT feature files; all metric columns use the identical 245-item intersection throughout.
<table><tr><td>Method</td><td>Scoreable</td><td>Sync-C ↑</td><td>Sync-D ↓</td><td>FVD↓</td></tr><tr><td>Wav2Lip</td><td>255/256</td><td>6.930</td><td>8.514</td><td>99.05</td></tr><tr><td>VideoReTalking</td><td>252/256</td><td>6.883</td><td>8.052</td><td>66.18</td></tr><tr><td>MuseTalk 1.5</td><td>253/256</td><td>4.193</td><td>10.650</td><td>57.24</td></tr><tr><td>Diff2Lip</td><td>255/256</td><td>7.588</td><td>7.286</td><td>59.17</td></tr><tr><td>LatentSync 1.6</td><td>250/256</td><td>7.541</td><td>7.732</td><td>46.40</td></tr><tr><td>X-Dub</td><td>251/256</td><td>6.254</td><td>9.050</td><td>39.09</td></tr><tr><td>Full-context Teacher-48 (ours)</td><td>253/256</td><td>8.707</td><td>7.100</td><td>15.62</td></tr><tr><td>Lip Forcing</td><td>253/256</td><td>5.197</td><td>9.967</td><td>16.14</td></tr><tr><td>Causal two-step VidDub</td><td>253/256</td><td>8.669</td><td>7.240</td><td>17.92</td></tr><tr><td>VidDub</td><td>253/256</td><td>8.702</td><td>7.141</td><td>17.96</td></tr></table>

Table 44: VoxCeleb2-30 results under in-the-wild pose and compression. FPS matches the main-paper single-H20 protocol.
<table><tr><td></td><td></td><td></td><td></td><td colspan="4">Same Audio</td><td colspan="3">Cross Audio</td></tr><tr><td>Method</td><td></td><td></td><td></td><td>Params NFE FPS↑ Sync-C↑ Sync-D↓ SSIM↑ PSNR↑ Sync-C↑ Sync-D↓</td><td></td><td></td><td></td><td></td><td></td><td>FVD↓</td></tr><tr><td>Feed-forward and one-step editors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wav2Lip [31]</td><td>36M</td><td>1</td><td>5.2</td><td>8.362</td><td>6.779</td><td>0.9297</td><td>31.34</td><td>7.800</td><td>7.173</td><td>71.73</td></tr><tr><td>VideoReTalking [5]</td><td>484M</td><td>1</td><td>1.3</td><td>7.697</td><td>6.971</td><td>0.9087</td><td>28.95</td><td>7.124</td><td>7.394</td><td>100.75</td></tr><tr><td>MuseTalk 1.5 [54]</td><td>971M</td><td>1</td><td>1.1</td><td>5.698</td><td>8.464</td><td>0.9317</td><td>32.19</td><td>4.305</td><td>9.673</td><td>54.59</td></tr><tr><td>Bidirectional diffusion editors (20–50 NFE)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Diff2Lip [28]</td><td>102M</td><td>25</td><td>2.0</td><td>8.075</td><td></td><td>6.2340.9341</td><td>33.36</td><td>7.533</td><td>6.662</td><td>67.15</td></tr><tr><td>LatentSync 1.6 [21]</td><td>1.39B</td><td>20</td><td>2.2</td><td>7.315</td><td>7.064</td><td>0.9533</td><td>34.59</td><td>6.239</td><td>8.020</td><td>28.89</td></tr><tr><td>X-Dub [14]</td><td>14.32B</td><td>50</td><td>0.4</td><td>6.298</td><td>8.598</td><td>0.8302</td><td>25.91</td><td>5.375</td><td>9.254</td><td>97.45</td></tr><tr><td>Full-context Teacher-30 (CFG 1)</td><td>2.81B</td><td>30</td><td>1.2†</td><td>7.641</td><td></td><td>6.971 0.9302</td><td>31.88</td><td>7.382</td><td>7.119</td><td>39.11</td></tr><tr><td>Two-step diffusion editors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Lip Forcing [6]</td><td>14.53B</td><td>2</td><td>2.4</td><td>6.408</td><td></td><td>7.831 0.9466</td><td>33.12</td><td>4.871</td><td>9.242</td><td>41.03</td></tr><tr><td>VidDub</td><td>2.81B</td><td>2 11.3†</td><td></td><td>8.609</td><td></td><td>6.4700.9262</td><td>31.20</td><td>8.454</td><td>6.579</td><td>46.52</td></tr></table>

Table 45: TalkVid-Bench results. FPS matches the main-paper single-H20 protocol. <sup>†</sup>Wav2Lip runs at resize factor 1; an earlier run used the adapter default of 2 and wrote half-resolution video, which corrupted visual metrics (Sync-C/D nearly unchanged). For 13 clips at 1440p/4K, face detection exceeds GPU memory at factor 1; these use factor 2 and are upscaled. The primary face detector initially rejected three clear frontal-face clips at its default threshold. A lower-threshold fallback, invoked only after a complete primary-detector miss, recovers these clips; the Causal two-step VidDub and Teacher-48 consequently generate and score all 500 requested clips in both tracks. Other methods use their method-specific successful subsets; the main-paper table reports every scored count.
<table><tr><td></td><td></td><td></td><td></td><td colspan="4">Same Audio</td><td colspan="3">Cross Audio</td></tr><tr><td>Method</td><td>Params NFE FPS↑ Sync-C↑ Sync-D↓ SSIM↑ PSNR↑ Sync-C↑ Sync-D↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>FVD↓</td></tr><tr><td>Feed-forward and one-step editors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wav2Lip† [31]</td><td>36M</td><td>1</td><td>5.2</td><td>7.209</td><td></td><td>7.4200.9218</td><td>29.83</td><td>6.750</td><td>7.932</td><td>111.19</td></tr><tr><td>VideoReTalking [5]</td><td>484M</td><td>1</td><td>1.3</td><td>6.787</td><td>7.552</td><td>0.8948</td><td>28.00</td><td>6.365</td><td>7.868</td><td>84.51</td></tr><tr><td>MuseTalk 1.5 [54]</td><td>971M</td><td>1</td><td>1.1</td><td>4.968</td><td>9.106</td><td>0.9305</td><td>29.60</td><td>4.371</td><td>9.625</td><td>36.09</td></tr><tr><td>Bidirectional diffusion editors (20–50 NFE)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Diff2Lip [28]</td><td>102M</td><td>25</td><td>2.0</td><td>7.775</td><td>6.383</td><td>0.9157</td><td>30.60</td><td>6.944</td><td>7.016</td><td>73.37</td></tr><tr><td>LatentSync 1.6 [21]</td><td>1.39B</td><td>20</td><td>2.2</td><td>7.516</td><td>6.965</td><td>0.8949</td><td>31.09</td><td>6.860</td><td>7.512</td><td>59.27</td></tr><tr><td>X-Dub [14]</td><td>14.32B</td><td>50</td><td>0.4</td><td>5.492</td><td>9.129</td><td>0.6980</td><td>21.86</td><td>5.087</td><td>9.437</td><td>32.63</td></tr><tr><td>Full-context Teacher-48 (ours)</td><td>2.81B</td><td>48</td><td>1.2†</td><td>7.741</td><td>7.170</td><td>0.9125</td><td>28.84</td><td>7.663</td><td>7.266</td><td>8.51</td></tr><tr><td>Causal two-step diffusion editors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Lip Forcing [6]</td><td>14.53B</td><td>2</td><td>2.4</td><td>5.914</td><td></td><td>8.2830.7698</td><td>24.48</td><td>5.610</td><td>8.648</td><td>12.68</td></tr><tr><td>VidDub</td><td>2.81B</td><td>2 11.3†</td><td></td><td>7.740</td><td></td><td>7.216 0.9125</td><td>28.78</td><td>7.646</td><td>7.309</td><td>10.25</td></tr></table>

Table 46: TalkVid cross-audio failure and common-intersection audit. “Success” requires an output, successful SyncNet record, and matching tables/GT feature files. All remaining metrics use the identical 308-item intersection (258 source videos); brackets are 95% source-cluster bootstrap intervals with 10,000 resamples. FVD is recomputed on that intersection and kept as a point estimate because it is a global feature-distribution functional.
<table><tr><td>Method</td><td>Success</td><td>Sync-C ↑ [95% CI]</td><td></td><td>Sync-D ↓ [95% CI]</td><td>FVD↓</td></tr><tr><td>Wav2Lip</td><td>499/500</td><td></td><td>6.686 [6.521,6.848]</td><td>7.964 [7.856,8.075]</td><td>112.21</td></tr><tr><td>VideoReTalking</td><td>402/500</td><td></td><td>6.361 [6.215,6.509]</td><td>7.859 [7.772,7.953]</td><td>87.80</td></tr><tr><td>MuseTalk 1.5</td><td>498/500</td><td></td><td>4.360 [4.195,4.535]</td><td>9.624 [9.495,9.754]</td><td>36.67</td></tr><tr><td>Diff2Lip</td><td>403/500</td><td></td><td>6.904 [6.745,7.067]</td><td>7.021 [6.925,7.116]</td><td>70.21</td></tr><tr><td>LatentSync 1.6</td><td>499/500</td><td></td><td>6.774 [6.611,6.943]</td><td>7.553 [7.450,7.656]</td><td>57.24</td></tr><tr><td>X-Dub</td><td>499/500</td><td></td><td>5.095 [4.916,5.275]</td><td>9.391 [9.256,9.523]</td><td>35.21</td></tr><tr><td>Lip Forcing</td><td>498/500</td><td></td><td>5.534 [5.352,5.714]</td><td>8.691 [8.545,8.839]</td><td>14.18</td></tr><tr><td>Causal two-step VidDub</td><td>500/500</td><td></td><td>7.641 [7.461,7.817]</td><td>7.434 [7.323,7.548]</td><td>14.24</td></tr><tr><td>VidDub</td><td>500/500</td><td></td><td>7.571 [7.391,7.753]</td><td>7.340 [7.222,7.459]</td><td>12.06</td></tr></table>

Table 47: Complete cross-audio benchmark summary used by the main-paper comparison. Each cell reports Sync-C / FVD; higher Sync-C and lower FVD are better. These displayed means use each method’s valid outputs and are accompanied by generation coverage in the detailed tables. Cross-method ranking is verified separately on the shared TalkVid-308 and VidDub-Bench-245 valid subsets in Tables 46 and 43.
<table><tr><td>Method</td><td>NFE</td><td>HDTF-30</td><td>VoxCeleb2-30</td><td>TalkVid-Bench</td></tr><tr><td>Wav2Lip [31]</td><td>1</td><td>8.782 / 129.51</td><td>7.800 / 71.73</td><td>6.750 /111.19</td></tr><tr><td>VideoReTalking [5]</td><td>1</td><td>8.312 / 105.72</td><td>7.124 / 100.75</td><td>6.365 / 84.51</td></tr><tr><td>Diff2Lip [28]</td><td>25</td><td>8.680 / 99.70</td><td>7.533 / 67.15</td><td>6.944 / 73.37</td></tr><tr><td>MuseTalk 1.5 [54]</td><td>1</td><td>7.116 /40.10</td><td>4.305 / 54.59</td><td>4.371 / 36.09</td></tr><tr><td>X-Dub [14]</td><td>50</td><td>5.027 / 66.05</td><td>5.375 / 97.45</td><td>5.087 / 32.63</td></tr><tr><td>LatentSync 1.6 [21]</td><td>20</td><td>8.160 / 28.05</td><td>6.239 / 28.89</td><td>6.860 / 59.27</td></tr><tr><td>Full-context teacher (30/30/48 NFE)</td><td>30/30/48</td><td>9.422 / 27.49</td><td>7.382 / 39.11</td><td>7.663 / 8.51</td></tr><tr><td>Causal diagnostic (4 NFE)</td><td>4</td><td>9.537 / 38.43</td><td>7.789 / 70.05</td><td>7.456 / 15.16</td></tr><tr><td>Final VidDub</td><td>2</td><td>10.102 / 27.66</td><td>8.454 / 46.52</td><td>7.646 / 10.25</td></tr></table>

Table 48: Frozen 50-pair cross-audio comparison. All methods use the same clips and evaluation protocol; these results are separate from the 256-source benchmark protocol.
<table><tr><td>Method</td><td>Backbone</td><td>NFE</td><td>Br.</td><td>Sync-C↑</td><td>Sync-D↓</td><td>FVD↓</td></tr><tr><td>Full-context parent, EMA@200</td><td>1.3B</td><td>48</td><td>2</td><td>8.816</td><td>6.950</td><td>43.86</td></tr><tr><td>LatentSync 1.6 [1, 21]</td><td></td><td>20</td><td>1</td><td>7.629</td><td>7.589</td><td>69.93</td></tr><tr><td>Lip Forcing [6]</td><td>14B</td><td>2</td><td>1</td><td>3.973</td><td>11.164</td><td>34.70</td></tr><tr><td>Causal two-step VidDub</td><td>1.3B</td><td>2</td><td>1</td><td>8.772</td><td>7.087</td><td>49.04</td></tr></table>

![](images/380b51010b6b373c2156e95f2138033228f09accbc2aa5dc35b0e2dd2d9f0d01.jpg)  
Figure 12: HDTF-30 case 1, same-audio track. Rows compare the source and editing methods at matched temporal positions. S1–S7 mark the same seven temporal positions across methods.

Table 49: Blinded human evaluation of full-context VidDub. 15 raters independently evaluated all 150 trials. Each entry is VidDub/Opp./Tie; ties are retained.
<table><tr><td>Opponent</td><td>Lip sync</td><td>Mouth realism</td><td>Identity</td><td>Stability</td><td>Overall</td></tr><tr><td>Teacher</td><td>10/10/130</td><td>9/12/129</td><td>4/8/138</td><td>4/14/132</td><td>12/20/118</td></tr><tr><td>LatentSync</td><td>123/2/25</td><td>122/6/22</td><td>75/3/72</td><td>123/2/25</td><td>143/2/5</td></tr><tr><td>Lip Forcing</td><td>142/0/8</td><td>90/15/45</td><td>61/2/87</td><td>87/2/61</td><td>146/1/3</td></tr></table>

![](images/904d29fde36aa4387b0c05b2eeb5b9e7964093a069736f0d0fac8ea78978a2ba.jpg)  
Figure 13: HDTF-30 case 1, cross-audio track. Rows compare the source and editing methods at matched temporal positions. The panel includes the actual target waveform; S1–S7 mark aligned acoustic segments rather than phoneme ground truth.

![](images/ef6e3c182118ea028a76146965cdcdda65f6393a1ad71a62517a557ffc550bbb.jpg)  
Figure 14: HDTF-30 case 2, same-audio track. Rows compare the source and editing methods at matched temporal positions. S1–S7 mark the same seven temporal positions across methods.

![](images/4c93c4898d7920d7199726d5342676766a1fa5caa3718590a89b6348c98bd9d5.jpg)  
Figure 15: HDTF-30 case 2, cross-audio track. Rows compare the source and editing methods at matched temporal positions. The panel includes the actual target waveform; S1–S7 mark aligned acoustic segments rather than phoneme ground truth.

![](images/805dddccad4d4e4c235144f695e2f2208740fe178958a2b7f5a2e86d4594458c.jpg)  
Figure 16: HDTF-30 case 3, same-audio track. Rows compare the source and editing methods at matched temporal positions. S1–S7 mark the same seven temporal positions across methods.

![](images/83b8b2b8c46074cf32fc1133e7165ffe610dc91d579a406c9bb6b7a525607a3e.jpg)  
Figure 17: HDTF-30 case 3, cross-audio track. Rows compare the source and editing methods at matched temporal positions. The panel includes the actual target waveform; S1–S7 mark aligned acoustic segments rather than phoneme ground truth.

![](images/cabb09022cb425613b497e8f971cc425cfdf752e5e364946a2be89b75eeb94d0.jpg)  
Figure 18: HDTF-30 case 4, same-audio track. Rows compare the source and editing methods at matched temporal positions. S1–S7 mark the same seven temporal positions across methods.

![](images/65288d5c73482983f86535cc21a97b1bc35d877e0a82866c4d79b3c00554d5ec.jpg)  
Figure 19: HDTF-30 case 4, cross-audio track. Rows compare the source and editing methods at matched temporal positions. The panel includes the actual target waveform; S1–S7 mark aligned acoustic segments rather than phoneme ground truth.

![](images/d1b81bf3fde25d8c3da032e3f7d435dd6051cc2510f5faeaacf6343ee056cc63.jpg)  
Figure 20: VoxCeleb2-30 case 1, same-audio track. Rows compare the source and editing methods at matched temporal positions. S1–S7 mark the same seven temporal positions across methods.

![](images/d27f65bf5e9283be8ab7b3d72ffd777c08f121d8b679c414dd623c2c4e3b2fdb.jpg)  
Figure 21: VoxCeleb2-30 case 1, cross-audio track. Rows compare the source and editing methods at matched temporal posi tions. The panel includes the actual target waveform; S1–S7 mark aligned acoustic segments rather than phoneme ground truth.

![](images/aea2cb9e09e18c378eb20a2cde387bf5379e18c6a9d706ec0b5398771cc90445.jpg)  
Figure 22: VoxCeleb2-30 case 2, same-audio track. Rows compare the source and editing methods at matched temporal positions. S1–S7 mark the same seven temporal positions across methods.

![](images/7449484a8f0146a239104ef3ed579bd244280afe8e94855c7d93e8e9158f3fd6.jpg)  
Figure 23: VoxCeleb2-30 case 2, cross-audio track. Rows compare the source and editing methods at matched temporal posi tions. The panel includes the actual target waveform; S1–S7 mark aligned acoustic segments rather than phoneme ground truth.

![](images/17b3fcb159c76d0d61630333db051c640159b540904b924ada6f3a6a053484cb.jpg)  
Figure 24: VoxCeleb2-30 case 3, same-audio track. Rows compare the source and editing methods at matched temporal positions. S1–S7 mark the same seven temporal positions across methods.

![](images/d7d213e266d7e689ca9c4239777ac9a60627a3ed82f16f81931f34e8f18c8b14.jpg)  
Figure 25: VoxCeleb2-30 case 3, cross-audio track. Rows compare the source and editing methods at matched temporal posi tions. The panel includes the actual target waveform; S1–S7 mark aligned acoustic segments rather than phoneme ground truth.

![](images/c5c63f21c86df0be30c71a747b7f917b0e43881046c0b213aab20bbe9ab71447.jpg)  
Figure 26: VoxCeleb2-30 case 4, same-audio track. Rows compare the source and editing methods at matched temporal positions. S1–S7 mark the same seven temporal positions across methods.

![](images/1da04c08968cbebe4ea3eb368a0865025182e74d1be5ceaea94814c8e040dd65.jpg)  
Figure 27: VoxCeleb2-30 case 4, cross-audio track. Rows compare the source and editing methods at matched temporal posi tions. The panel includes the actual target waveform; S1–S7 mark aligned acoustic segments rather than phoneme ground truth.

![](images/6abefb948f95b25651c9008275050abe80ca2303beebcb6d39decbdf1db1e1ae.jpg)  
Figure 28: VidDub-Bench case 1, same-audio track. Rows compare Source, LatentSync, the full-context parent, and causal two-step VidDub. All rows use the same seven temporal positions.

![](images/ffb26cd352937bc3152afb076c3bb8543d0f628f0956097ee61be5149b2919cc.jpg)  
Figure 29: VidDub-Bench case 1, cross-audio track. Rows compare Source, LatentSync, the full-context parent, and causa two-step VidDub. The panel includes the target waveform and aligned acoustic segments.

![](images/c51b6f8243fb45e81e753923a32dcaf85bcec40062c70674f01689e2bb8f23e8.jpg)  
Figure 30: VidDub-Bench case 2, same-audio track. Rows compare Source, LatentSync, the full-context parent, and causal two-step VidDub. All rows use the same seven temporal positions.

![](images/1f3f50fe985fd118b2d2f22110809072b0fbd018d1a9ebf65e9969ddb17cffa3.jpg)  
Figure 31: VidDub-Bench case 2, cross-audio track. Rows compare Source, LatentSync, the full-context parent, and causa two-step VidDub. The panel includes the target waveform and aligned acoustic segments.

<table><tr><td>Identity preservation</td><td>OA</td><td>OB</td><td>O Tie</td></tr><tr><td>Mouth realism</td><td>OA</td><td>OB</td><td>O Tie</td></tr><tr><td>Overall preference</td><td>OA</td><td>OB</td><td>O Tie</td></tr><tr><td>Lip synchronization</td><td>OA</td><td>OB</td><td>OTie</td></tr><tr><td>Temporal stability</td><td>OA</td><td>OB</td><td>O Tie</td></tr><tr><td>Previous Next</td><td></td><td></td><td></td></tr></table>

## Blinded cross-audio video rating

Compare A and B with audio enabled. Method identity, metrics, and checkpoint names are hidden. Answer all five questions for every trial

Rater ID

0/150 trials complete

Trial 1 of 150 · ZH  
A  
![](images/f7eec8efb630f02d760b79181f90a9bb48ab52d2e3b51b374cfea6a2d32fde9d.jpg)

B  
![](images/adea4dc9f322eff71350288b82665b7de1ef9958ada5efc79fa650d04ba7ac62.jpg)  
Figure 32: Blinded rating interface. Each trial shows only an anonymized A/B pair. The first four criteria are randomized, and overall quality is shown last in every trial of the study.