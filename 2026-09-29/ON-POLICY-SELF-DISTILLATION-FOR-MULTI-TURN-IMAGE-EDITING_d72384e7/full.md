# ON-POLICY SELF-DISTILLATION FOR MULTI-TURN IMAGE EDITING

Liangbing Zhao<sup>1</sup> Le Zhuo<sup>2</sup> Mohamed Elhoseiny<sup>1†</sup>

<sup>1</sup> KAUST <sup>2</sup> Krea AI

<sup>†</sup>Corresponding author

## ABSTRACT

Instruction-based image editing has achieved strong performance in single-turn settings, yet practical editing is often iterative, with each instruction applied to the output of the previous turn. We find that existing editing models degrade rapidly under recursive editing and attribute this failure to a train–test mismatch in the conditioning distribution: models are trained on clean source images but must repeatedly condition on their own imperfect outputs at inference time. To address this, we propose MT-OPSD, an on-policy self-distillation framework that trains the model on self-generated conditioning states with editing supervision from a clean-conditioned teacher, without requiring multi-turn annotations. We further introduce LME-Bench, a benchmark of 100 ten-turn editing sessions for evaluating long-horizon robustness. Experiments across three editing backbones show that MT-OPSD substantially improves long-horizon editing success and reduces multi-turn collapse while largely preserving single-turn editing quality.

Project page: https://liangbingzhao.github.io/MT-OPSD/

## 1 INTRODUCTION

Recent progress in instruction-based image editing (Brooks et al., 2023; Wei et al., 2025; Wu et al., 2025a) has made it possible to modify images through natural-language instructions, covering a broad range of operations from local object edits to global appearance changes. Despite strong performance on single-turn edits, real-world editing workflows often involve multiple rounds of refinement. For example, a user may adjust the lighting of a scene, modify its color palette, reposition an object, and apply a stylistic transformation, with each edit operating on the result of the previous one. A practical image editor should therefore remain reliable across multiple editing turns, maintaining visual coherence while accurately following each new instruction.

In practice, however, current editing models often struggle across consecutive editing turns. As a model repeatedly edits its own outputs, small errors accumulate and progressively degrade image quality. After only a few turns, outputs may exhibit high-frequency chromatic noise, structural fragmentation, or severe identity drift. We observe this behavior across different editing models, suggesting that multi-turn degradation is a broader limitation of the standard single-turn training paradigm rather than a model-specific failure.

We hypothesize that this degradation arises from a train–test mismatch in the conditioning distribution. During training, editing models are conditioned on clean source images, whereas at inference time they must operate on their previous outputs, which inevitably contain small model-induced errors. Although these errors may be visually negligible after a single edit, they accumulate as each output becomes the input to the next turn. Figure 1 provides a simple diagnostic: even when instructed to reproduce the input image without modification, the model exhibits measurable drift at each

![](images/db9aae568fec2ef37a8bfdc4e33f30b6400481998be41353a07f3b45f082733c.jpg)  
Make everything unchanged  
Figure 1: Drift under identity editing. Repeatedly instructing a model to leave the image unchanged exposes model-induced errors that accumulate across turns, and similar degradation occurs across different editing models.

turn, indicating that recursive editing introduces errors that are not captured by single-turn evaluation. This behavior is analogous to exposure bias (Huang et al., 2026b) in autoregressive video generation, where models trained on ground-truth data are evaluated on their own outputs.

One possible solution is to train directly on multi-turn editing sequences, but such data is scarce. More importantly, optimizing end-to-end through model-generated multi-turn rollouts would require backpropagation across hundreds of denoising steps and multiple editing turns, making it prohibitively expensive. Training-free methods such as Emu Edit (Sheynin et al., 2024) provide another option by reverting nearly unchanged pixels to each turn’s input, which can reduce drift for local edits. However, this approach does not extend to global transformations such as restyling or relighting, where most pixels are expected to change.

To address this gap, we propose MT-OPSD, an on-policy self-distillation framework for robust multi-turn image editing without requiring multi-turn annotations or ground-truth edited images. MT-OPSD builds on the observation that a pretrained editor already exhibits strong single-turn editing behavior under clean conditioning; multi-turn editing introduces an additional challenge as model-induced errors are repeatedly carried into subsequent turns. Rather than modeling the full distribution of multi-turn editing histories, we isolate this error component through identity rollouts, which preserve the intended image content while accumulating errors from the model’s own predictions. Training then alternates between two complementary objectives: an identity branch that prevents further drift, and an editing branch that pairs these self-generated states with real editing instructions and transfers the model’s clean-condition editing behavior through on-policy velocity matching. An adaptive rollout curriculum progressively exposes the student to deeper self-generated states, while gated teacher promotion updates the clean-condition reference as training proceeds.

To facilitate the evaluation of long-horizon editing robustness, we further construct Long Multi-turn Image Editing Bench (LME-Bench), an evaluation benchmark consisting of 100 editing sessions, each containing 10 consecutive turns with a diverse combination of local and global operations. Each session is evaluated at every turn in terms of editing accuracy, visual consistency, and image quality, enabling a systematic analysis of when and how multi-turn degradation emerges.

In summary, our contributions are as follows:

• We show that multi-turn collapse occurs across modern image editing models and attribute it to the train–test mismatch in the conditioning distribution.

• We propose MT-OPSD, an on-policy self-distillation framework that transfers the model’s own clean-conditioned editing behavior to self-generated states, without requiring multiturn annotations or an external teacher.

• We introduce LME-Bench, a benchmark of 100 ten-turn editing sessions covering both local and global operations for evaluating long-horizon editing robustness.

• Experiments across three editing backbones and multiple benchmarks show that MT-OPSD substantially improves long-horizon editing success and reduces multi-turn collapse while largely preserving single-turn editing quality.

## 2 RELATED WORK

Instruction-based Image Editing. The field of image editing has witnessed a paradigm shift from domain-specific generative adversarial networks (Goodfellow et al., 2020) to high-fidelity diffusion models (Ho et al., 2020; Rombach et al., 2022; Abdelrahman et al., 2025). Early diffusion-based approaches (Hertz et al., 2022; Mokady et al., 2023; Zhao et al., 2023) explored attention manipulation and latent inversion to enable image modifications while preserving the original content, but often struggled with complex and diverse editing instructions. This motivated instruction-based editing, pioneered by InstructPix2Pix (Brooks et al., 2023) and subsequently advanced through improved data curation and scaling (Zhang et al., 2023; Wei et al., 2025; Zhuo et al., 2025), stronger vision-language understanding (Wu et al., 2025a;b; Zhao et al., 2026a), unified generative frameworks (Deng et al., 2025; Xie et al., 2024), and in-context flow models (Labs et al., 2025; Liu et al., 2025). However, these approaches are primarily designed and evaluated for independent single-turn edits, leaving the robustness of models under repeated self-conditioned editing largely unexplored.

Multi-turn Image Editing. Recent works have explored improving consistency across sequential edits. Training-free approaches, including Emu Edit (Sheynin et al., 2024), FreqEdit (Liao et al., 2026), and VAE-LFA (Wang et al., 2026), alleviate accumulated degradation through image-space or latent-space corrections, but rely on assumptions about the edits and do not alter the model’s editing behavior. MTC (Zhou et al., 2025) uses per-turn inversion with trajectory control and adaptive attention guidance on text-to-image models. VINCIE (Qu et al., 2026) and AnchorEdit (Xu et al., 2026) instead train dedicated models for causal multi-turn editing by adapting video architectures to interleaved image sequences. Edit-R2 (Ye et al., 2026b) focuses on preserving session-level constraints across turns. Most closely related to our work, MT-EditFlow (Huang et al., 2026a) also attributes multi-turn degradation to exposure bias, but addresses it through reinforcement learning with external reward supervision. In contrast, MT-OPSD starts from the observation that pretrained editors already possess strong single-turn editing ability, and uses the model’s own clean-condition behavior to extend this ability to self-generated states through on-policy self-distillation.

On-Policy Self-Distillation. On-policy self-distillation (OPSD) trains a model on states generated by its own policy while using the same model as teacher under richer conditioning. In language models, this is commonly realized by providing the teacher with privileged context (Zhao et al., 2026b; Penaloza et al., 2026; Sang et al., 2026). In visual generation, D-OPSD (Jiang et al., 2026) conditions the teacher on a paired target image while supervising the student along its own diffusion trajectory, whereas OPSD-V (Liu et al., 2026) uses real long-video context to supervise autoregressive generation from self-generated history. DiffusionOPSD (Zhou et al., 2026a) derives its targets from reward gradients rather than privileged context. Closely related OPD methods for diffusion and flow models likewise provide teacher supervision along student-generated sampling trajectories (Li et al., 2026; Fang et al., 2026; Zhou et al., 2026b). Our setting introduces a different form of on-policy state: in multi-turn editing, the conditioning image itself evolves through the model’ previous outputs. MT-OPSD therefore uses the clean source image as privileged teacher context and the self-generated rollout state as student context, extending the model’s clean-condition editing behavior to recursive editing without paired target images or multi-turn annotations.

## 3 METHOD

## 3.1 PRELIMINARIES AND PROBLEM FORMULATION

Multi-turn Image Editing. Modern instruction-based editing models are commonly built on flow matching (Lipman et al., 2023), where a velocity model $v _ { \theta } ( x _ { t } , t , I , e )$ learns a time-dependent vector field between the target edit and Gaussian noise, conditioned on a source image I and an editing instruction e. Given an initial image $I ^ { ( 0 ) }$ and instructions $e _ { 1 } , \ldots , e _ { K }$ , multi-turn editing recursively applies the editing operator $G _ { \theta } ( I , e ) \colon$

$$
I ^ { ( k ) } = G _ { \theta } ( I ^ { ( k - 1 ) } , e _ { k } ) .\tag{1}
$$

This multi-turn process introduces a train–test mismatch in the conditioning distribution. While training exposes the model only to clean source images, later turns condition on self-generated outputs $I ^ { ( k - 1 ) }$ . Such states contain both the intended semantic changes introduced by earlier edits and model-induced errors accumulated across previous editing turns. The former are part of the editing task itself, while the latter are absent from clean single-turn training and constitute the additiona source of mismatch that we target.

On-Policy Self-Distillation. On-policy self-distillation (OPSD) combines on-policy distillation (Agarwal et al., 2024) with self-distillation: the student is supervised at states generated by its own policy, while the same model serves as the teacher under a more informative context (Zhao et al., 2026b; Penaloza et al., 2026). Let s denote a state visited along the student rollout, and let $c _ { S }$ and $c _ { T }$ denote the student and teacher contexts, respectively. For a student $\pi _ { \theta }$ and a teacher $\pi _ { \bar { \theta } }$ derived from the same model, OPSD minimizes

$$
{ \mathcal { L } } _ { \mathrm { O P S D } } = \mathbb { E } _ { s \sim d _ { \pi _ { \theta } } ( \cdot \vert c _ { S } ) } \left[ D ( \pi _ { \bar { \theta } } ( \cdot  { \vert } s , c _ { T } ) , \pi _ { \theta } ( \cdot  { \vert } s , c _ { S } ) ) \right] ,\tag{2}
$$

where $d _ { \pi _ { \theta } }$ denotes the state distribution induced by the student and D is a distillation divergence. This allows the model to distill information available under a richer context into its behavior under a

![](images/9d80fc0985ac6a76e9b4c71116f772802a985b8c7ecfc3f3e16508b381ea22b0.jpg)  
Figure 2: Overview of MT-OPSD. The student first constructs self-generated rollout states by repeatedly applying the identity instruction. For each rollout state $\tilde { I } ^ { ( k ) }$ , the identity branch uses the state itself as the target under $\mathcal { L } _ { \mathrm { i d } }$ , while the editing branch aligns the student conditioned on $\tilde { I } ^ { ( k ) }$ with a frozen teacher conditioned on the clean source $I ^ { ( 0 ) }$ via sparse query-based velocity matching $\mathcal { L } _ { \mathrm { e d i t } }$ . A rollout curriculum progressively increases the rollout depth, and improved student checkpoints replace the teacher through gated promotion.

less informative student context, without a separate teacher. For diffusion models, OPSD is typically instantiated by constructing asymmetric conditioning contexts for the teacher and student, so that predictions under the teacher context provide supervision along the student’s denoising process.

## 3.2 MT-OPSD

To address the conditioning mismatch, MT-OPSD isolates the model-induced error component and uses on-policy self-distillation to preserve the model’s clean-condition editing behavior on selfgenerated states. We denote the student by $\theta _ { S }$ and the teacher by $\theta _ { T }$ , both initialized from the same pretrained model. The student operates on self-generated states, while the teacher is conditioned on the clean source and remains fixed between gated promotions. The framework consists of four components: self-generated rollout states, a two-branch training objective, an adaptive rollout curriculum, and gated teacher promotion. The overall framework and training procedure of MT-OPSD are summarized in Figure 2 and Algorithm 1.

Self-Generated Rollout States. To obtain conditioning states that contain errors induced by the model itself, we recursively apply the current student to its own outputs. Specifically, we define an identity instruction $e _ { \mathrm { i d } } \ ( \mathrm { e . g . }$ , “Make everything unchanged”) that asks the model to reproduce the input image without modification, and roll out the current student for k turns starting from a clean source image $I ^ { ( 0 ) }$ :

$$
\tilde { I } ^ { ( k ) } = G _ { \theta _ { S } } ( \tilde { I } ^ { ( k - 1 ) } , e _ { \mathrm { i d } } ) , \qquad \tilde { I } ^ { ( 0 ) } = I ^ { ( 0 ) } .\tag{3}
$$

The rollout follows the same sampling configuration as inference, so the resulting drift arises from the model’s own generation process. Since the intended image content remains unchanged, the difference between $\bar { \tilde { I } } ^ { ( k ) }$ and $I ^ { ( 0 ) }$ mainly reflects model-induced errors accumulated across the rollout. Using actual editing instructions during the rollout would instead entangle these errors with intended semantic changes, making it difficult to obtain a reliable supervision signal for learning robustness to model-induced errors. The identity rollout therefore provides a controlled way to construct onpolicy conditioning states from single-turn training data while isolating the error component.

Two-Branch Training Objective. Given a rollout state $\tilde { I } ^ { ( k ) }$ , we optimize two complementary branches, sampled at a fixed ratio. The identity branch operates under $e _ { \mathrm { i d } }$ and suppresses unintended changes and accumulated errors across turns, while the editing branch pairs the same rollout state with a real editing instruction e to preserve editing capability under self-generated conditioning.

The identity branch supervises the model under the identity instruction. A natural choice is to use the clean source image $I ^ { ( 0 ) }$ as the target, asking the model to recover the clean image from its degraded input. However, we find this restoration objective difficult to optimize, as it requires correcting errors accumulated over k turns within a single turn. We therefore use the rollout state itself as the target, yielding an identity objective that prevents further drift rather than restoring the clean source:

$$
\mathcal { L } _ { \mathrm { i d } } = \mathbb { E } _ { t , \epsilon } \left[ \left. v _ { \theta _ { S } } ( \tilde { x } _ { t } , t , \tilde { I } ^ { ( k ) } , e _ { \mathrm { i d } } ) - ( \epsilon - \tilde { x } _ { 0 } ) \right. _ { 2 } ^ { 2 } \right] ,\tag{4}
$$

where $\tilde { x } _ { 0 }$ denotes the latent representation of $\tilde { I } ^ { ( k ) } , \tilde { x } _ { t } = ( 1 - t ) \tilde { x } _ { 0 } + t \epsilon$ , and $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ . This objective prevents accumulated errors from being further amplified, but provides no supervision for executing non-identity edits on degraded states.

The editing branch addresses this limitation by maintaining the model’s editing capability under self-generated conditioning. Because the identity rollout preserves the intended content of $I ^ { ( 0 ) }$ the rollout state $\tilde { I } ^ { ( k ) }$ and the clean source differ primarily in accumulated model-induced errors. For the same editing instruction $e ,$ we therefore use the model’s clean-conditioned prediction as a reference for editing $\tilde { I } ^ { ( k ) }$ . Specifically, the teacher $\theta _ { T }$ is conditioned on $I ^ { ( 0 ) }$ , while the student $\theta _ { S }$ is conditioned on $\bar { \tilde { I ^ { ( k ) } } }$ . Following the sparse query-based velocity matching of DanceOPD (Zhou et al., 2026b), the student performs an N-step denoising process. We sample a small number of query steps from $p _ { q }$ and match the teacher and student velocities at the corresponding student states:

$$
\mathcal { L } _ { \mathrm { e d i t } } = \mathbb { E } _ { q \sim p _ { q } } \left[ \left. v _ { \theta _ { S } } ( \bar { x } _ { t _ { q } } , t _ { q } , \tilde { I } ^ { ( k ) } , e ) - v _ { \theta _ { T } } ( \bar { x } _ { t _ { q } } , t _ { q } , I ^ { ( 0 ) } , e ) \right. _ { 2 } ^ { 2 } \right] ,\tag{5}
$$

where $\bar { x } _ { t _ { q } } = \mathrm { s g } ( x _ { t _ { q } } )$ denotes the stop-gradient state reached by the student at query step q. At each queried step, teacher and student are evaluated at the same noisy latent with the same classifier-free guidance (CFG) scale, and each retains its own image condition in both CFG branches, dropping only the text condition in the unconditional branch.

Rollout Curriculum. The difficulty of both training branches increases with rollout depth $k ,$ as deeper rollouts accumulate larger model-induced errors. Starting from a large depth therefore exposes the model to heavily degraded states early in training and can destabilize optimization. We instead increase the rollout depth progressively. Specifically, we measure rollout drift as the mean pixel deviation between $\tilde { I } ^ { ( k ) }$ and $I ^ { ( 0 ) }$ , and increase the depth by one turn only when the drift remains below a threshold for a fixed number of consecutive training steps. After each increase, the resulting rise in drift delays further progression until the model adapts to the current depth, yielding an adaptive curriculum without a manually specified schedule. We cap the rollout depth at $K _ { \mathrm { m a x } } .$ In practice, the adaptive curriculum typically saturates at around four turns, well below this limit.

Gated Teacher Promotion. The editing branch relies on $\theta _ { T }$ as a stable clean-condition reference. The pretrained initialization provides a strong starting teacher because it already exhibits reliable single-turn editing behavior under clean conditioning. As the student is trained on self-generated states with the two-branch objective, however, it can gradually acquire greater robustness to selfgenerated conditioning than the current teacher. Continuing to match the same fixed teacher can then limit further improvement. We therefore allow improved student checkpoints to replace the teacher, while keeping the teacher fixed between promotions. Each candidate is evaluated asynchronously by a VLM judge on a held-out gate set and promoted only when it satisfies a long-horizon criterion based on success and collapse rates. The VLM judge is used only for checkpoint selection rather than as a training target or reward signal. The full promotion rule is provided in Appendix C.

## 4 EXPERIMENTS

## 4.1 LME-BENCH

Benchmark Construction. Existing multi-turn editing benchmarks contain at most five consecutive turns, making them insufficient to evaluate the long-horizon degradation studied in this work.

We therefore construct LME-Bench, which contains 100 editing sessions of 10 consecutive turns each, for a total of 1,000 editing instructions. Source images are generated by Z-Image-Turbo (Cai et al., 2025) at 1024 × 1024 resolution and are evenly distributed across 10 semantic categories. As shown in Figure 3, each session contains six local edits that modify specific objects or attributes and four global edits that change the overall style, atmosphere, or photometric appearance. Global edits are placed non-adjacently and are not used in the final turns, so that later local edits are applied to images that have already undergone global changes. To avoid invalid or ambiguous instructions, we define six families of validity rules, covering cases such as editing invisible regions or requesting a state that is already satisfied. Candidate instructions are first checked by a VLM and then manually reviewed to ensure that each edit is visible, feasible, and unambiguous. The full edit taxonomy and composition rules are provided in Appendix D.1.

Evaluation Protocol and Metrics. Each session is executed sequentially, with every turn conditioned only on the output of the previous turn. We use GPT-4o to evaluate each turn in terms of prompt following, consistency with the previous state, and visual quality. We report two complementary metrics. Success rate at turn k (SR@k) is the fraction of sessions in which all of the first k turns satisfy both the prompt-following and consistency criteria, reflecting cumulative editing success. Collapse rate at turn k (CR@k) is the fraction of sessions that have undergone persistent visual degradation by turn k. A session is considered collapsed once two consecutive turns are judged visually degraded, and remains counted as collapsed thereafter. The full judging prompts, thresholds, and evaluation details are provided in Appendix D.2.

![](images/7aa51ac32c260e6548dee97bc77bc647d80423cebd346d383f739457bd61d050.jpg)  
Figure 3: Edit-type distribution in LME-Bench.

## 4.2 EXPERIMENTAL SETUP

Implementation Details. We instantiate MT-OPSD on three instruction-based editing backbones, including Qwen-Image-Edit-2511 (Wu et al., 2025a), FLUX.2-klein-base (Labs, 2025), and FireRed-Image-Edit (Team et al., 2026), and adapt each model using LoRA (Hu et al., 2022). Un less otherwise specified, we use a LoRA rank of 32, set the maximum rollout depth to $K _ { \operatorname* { m a x } } = 1 0$ and sample the editing and identity branches at a ratio of 2:1. Training uses approximately 2,000 source-image/instruction pairs from OmniEdit (Wei et al., 2025); the corresponding ground-truth edited images are not used. All self-rollouts follow the same sampling configuration as inference, and GPT-4o is used only for gated teacher promotion during training. Each run uses four NVIDIA H100/H200 GPUs for training and another four GPUs for asynchronous gate evaluation. Additional implementation details and hyperparameters are provided in Appendix A.

Evaluation. We evaluate long-horizon editing robustness on LME-Bench, using SR@k and CR@k as defined in Section 4.1. We also evaluate on MSE-Bench (Qu et al., 2026), which contains 100 five-turn editing sessions, with each turn applied to the output of the previous one. Following its official evaluation protocol, GPT-4o scores each turn for prompt following and consistency with the previous state, and a turn is considered successful only when both scores meet the corresponding thresholds. We report the success rate at each editing turn. To verify that the improved multi-turn robustness does not sacrifice single-turn editing quality, we further include ImgEdit (Ye et al., 2026a), a standard single-turn instruction-based editing benchmark, and report its overall score.

Comparison Baselines. For each backbone, we compare MT-OPSD against the original base model and three training-free methods for reducing multi-turn degradation. Emu Edit (Sheynin et al., 2024) reverts nearly unchanged pixels to each turn’s input. FreqEdit (Liao et al., 2026) mitigates accumulated frequency degradation by recovering high-frequency information, while VAE-

Table 1: Quantitative comparison on LME-Bench. We report the success rate (SR) and the collapse rate (CR) at turns 3, 5, 8, and 10.
<table><tr><td rowspan="2">Models</td><td colspan="4">SR↑</td><td colspan="4">CR↓</td></tr><tr><td>Turn-3 Turn-5</td><td></td><td>Turn-8</td><td>Turn-10</td><td>Turn-3</td><td>Turn-5</td><td>Turn-8</td><td>Turn-10</td></tr><tr><td colspan="9">Proprietary Models</td></tr><tr><td>GPT-Image-1 (OpenAI, 2025)</td><td>0.82</td><td>0.48</td><td>0.09</td><td>0.01</td><td>0.03</td><td>0.35</td><td>0.63</td><td>0.69</td></tr><tr><td>GPT-Image-2 (OpenAI, 2026)</td><td>1.00</td><td>0.98</td><td>0.98</td><td>0.96</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Nano Banana (Google, 2025a)</td><td>0.67</td><td>0.66</td><td>0.63</td><td>0.56</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Nano Banana Pro (Google, 2025b)</td><td>1.00</td><td>1.00</td><td>0.97</td><td>0.91</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td colspan="9">Open-Source Models</td></tr><tr><td>Qwen-Image-Edit-2511 (Wu et al., 2025a)</td><td>0.98</td><td>0.43</td><td>0.08</td><td>0.03</td><td>0.00</td><td>0.10</td><td>0.45</td><td>0.55</td></tr><tr><td>+ Emu Edit (Sheynin et al., 2024)</td><td>0.95</td><td>0.54</td><td>0.19</td><td>0.13</td><td>0.00</td><td>0.04</td><td>0.21</td><td>0.30</td></tr><tr><td>+ FreqEdit (Liao et al., 2026)</td><td>0.89</td><td>0.22</td><td>0.03</td><td>0.01</td><td>0.00</td><td>0.08</td><td>0.41</td><td>0.54</td></tr><tr><td>+ VAÉ-LFA (Wang et al., 2026)</td><td>0.84</td><td>0.46</td><td>0.18</td><td>0.10</td><td>0.00</td><td>0.05</td><td>0.29</td><td>0.36</td></tr><tr><td>+ MT-OPSD</td><td>1.00</td><td>0.91</td><td>0.63</td><td>0.44</td><td>0.00</td><td>0.01</td><td>0.02</td><td>0.02</td></tr><tr><td>FireRed-Image-Edit (Team et al., 2026)</td><td>0.95</td><td>0.57</td><td>0.28</td><td>0.15</td><td>0.01</td><td>0.16</td><td>0.51</td><td>0.61</td></tr><tr><td>+ Emu Edit (Sheynin et al., 2024)</td><td>0.97</td><td>0.58</td><td>0.30</td><td>0.14</td><td>0.01</td><td>0.05</td><td>0.36</td><td>0.51</td></tr><tr><td>+ FreqEdit (Liao et al., 2026)</td><td>0.93</td><td>0.41</td><td>0.04</td><td>0.00</td><td>0.01</td><td>0.11</td><td>0.64</td><td>0.73</td></tr><tr><td>+ VAÉ-LFA (Wang et al., 2026)</td><td>0.88</td><td>0.70</td><td>0.50</td><td>0.37</td><td>0.01</td><td>0.08</td><td>0.26</td><td>0.33</td></tr><tr><td>+ MT-OPSD</td><td>0.98</td><td>0.88</td><td>0.64</td><td>0.52</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.03</td></tr><tr><td>FLUX.2-klein-base (Labs, 2025)</td><td>0.92</td><td>0.52</td><td>0.29</td><td>0.12</td><td>0.00</td><td>0.01</td><td>0.17</td><td>0.25</td></tr><tr><td>+ Emu Edit (Sheynin et al., 2024)</td><td>0.90</td><td>0.52</td><td>0.23</td><td>0.11</td><td>0.00</td><td>0.02</td><td>0.12</td><td>0.18</td></tr><tr><td>+ FreqEdit (Liao et al., 2026)</td><td>0.64</td><td>0.19</td><td>0.03</td><td>0.01</td><td>0.00</td><td>0.03</td><td>0.35</td><td>0.54</td></tr><tr><td>+ VAÉ-LFA (Wang et al., 2026)</td><td>0.81</td><td>0.51</td><td>0.31</td><td>0.16</td><td>0.00</td><td>0.03</td><td>0.16</td><td>0.24</td></tr><tr><td>+ MT-OPSD</td><td>0.95</td><td>0.76</td><td>0.57</td><td>0.38</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.04</td></tr></table>

LFA (Wang et al., 2026) corrects low-frequency drift in the VAE latent space. All methods use the same backbone and inference configuration whenever applicable. We additionally report GPT-Image-1 (OpenAI, 2025) and Nano Banana (Google, 2025a) as proprietary systems, together with the more recent GPT-Image-2 (OpenAI, 2026) and Nano Banana Pro (Google, 2025b) on LME Bench. Among the training-based multi-turn editing methods discussed in Section 2, VINCIE (Qu et al., 2026) is the only one with public code and checkpoints. Since VINCIE is trained as a separate model rather than instantiated on our backbones, we report its comparison separately in Appendix E.

## 4.3 MAIN RESULTS

Results on LME-Bench. Table 1 shows that all three open-source base models degrade sharply over longer editing sequences, with SR@10 falling to 0.03–0.15 and CR@10 rising to 0.25–0.61. Existing training-free methods yield model-dependent gains: Emu Edit notably improves Qwen-Image-Edit-2511, VAE-LFA is more effective on FireRed-Image-Edit, while FreqEdit generally degrades long-horizon performance. In contrast, MT-OPSD raises SR@10 to 0.38–0.52 across all three backbones and reduces CR@10 to at most 0.04 without sacrificing early-turn performance. Although the rollout curriculum typically saturates at around four turns, these gains persist through ten-turn evaluation, indicating robustness beyond the rollout depths encountered during training. Among proprietary systems, GPT-Image-1 also exhibits severe long-horizon degradation and is outperformed by MT-OPSD across all reported turns and metrics. Nano Banana rarely collapses, while its lower SR is primarily attributable to failures on global edits that occur predominantly in the earlier turns. GPT-Image-2 and Nano Banana Pro remain stronger at long horizons. Overall, these results show that robustness learned from self-generated rollout states transfers to real multi-turn editing sequences, substantially narrowing the long-horizon robustness gap for open-source models.

Results on MSE-Bench and ImgEdit. Table 2 reports results on MSE-Bench, an existing fiveturn benchmark dominated by local edits, and ImgEdit, which evaluates whether the improved multiturn robustness preserves single-turn editing quality. On MSE-Bench, MT-OPSD improves SR@5 from 0.24 to 0.51 on Qwen-Image-Edit-2511 and from 0.40 to 0.69 on FireRed-Image-Edit, while matching FLUX.2-klein-base at 0.49. Emu Edit remains competitive in this setting, where the predominance of local edits favors pixel-level restoration, but MT-OPSD achieves comparable or better performance across the three backbones. On ImgEdit, MT-OPSD largely preserves the original single-turn capability, with only minor score changes across all three models.

Table 2: Quantitative comparison on MSE-Bench and ImgEdit. We report the success rate (SR) at each turn on MSE-Bench under its official evaluation protocol, together with the overall score on the single-turn ImgEdit benchmark.
<table><tr><td rowspan="2">Models</td><td colspan="5">MSE-Bench (SR ↑)</td><td rowspan="2">ImgEdit Overall ↑</td></tr><tr><td>Turn-1</td><td>Turn-2</td><td>Turn-3</td><td>Turn-4</td><td>Turn-5</td></tr><tr><td colspan="7">Proprietary Models</td></tr><tr><td>GPT-Image-1 (OpenAI, 2025)</td><td>0.96</td><td>0.69</td><td>0.67</td><td>0.64</td><td>0.56</td><td>4.20</td></tr><tr><td>Nano Banana (Google, 2025a)</td><td>0.99</td><td>0.77</td><td>0.75</td><td>0.73</td><td>0.63</td><td>4.29</td></tr><tr><td colspan="7">Open-Source Models</td></tr><tr><td>Qwen-Image-Edit-2511 (Wu et al., 2025a)</td><td>0.97</td><td>0.91</td><td>0.72</td><td>0.52</td><td>0.24</td><td>4.51</td></tr><tr><td>+ Emu Edit (Sheynin et al., 2024)</td><td>0.99</td><td>0.93</td><td>0.80</td><td>0.69</td><td>0.49</td><td>4.36</td></tr><tr><td>+ FreqEdit (Liao et al., 2026)</td><td>0.95</td><td>0.80</td><td>0.60</td><td>0.39</td><td>0.14</td><td>4.48</td></tr><tr><td>+ VAÉ-LFA (Wang et al., 2026)</td><td>0.99</td><td>0.93</td><td>0.78</td><td>0.61</td><td>0.34</td><td>4.32</td></tr><tr><td>+ MT-OPSD</td><td>0.98</td><td>0.90</td><td>0.79</td><td>0.69</td><td>0.51</td><td>4.49</td></tr><tr><td>FireRed-Image-Edit (Team et al., 2026)</td><td>1.00</td><td>0.97</td><td>0.80</td><td>0.62</td><td>0.40</td><td>4.56</td></tr><tr><td>+ Emu Edit (Sheynin et al., 2024)</td><td>1.00</td><td>0.95</td><td>0.86</td><td>0.77</td><td>0.59</td><td>4.33</td></tr><tr><td>+ FreqEdit (Liao et al., 2026)</td><td>0.99</td><td>0.91</td><td>0.70</td><td>0.48</td><td>0.24</td><td>4.54</td></tr><tr><td>+ VAÉ-LFA (Wang et al., 2026)</td><td>0.99</td><td>0.95</td><td>0.84</td><td>0.67</td><td>0.48</td><td>4.29</td></tr><tr><td>+ MT-OPSD</td><td>0.99</td><td>0.95</td><td>0.87</td><td>0.78</td><td>0.69</td><td>4.52</td></tr><tr><td>FLUX.2-klein-base (Labs, 2025)</td><td>0.97</td><td>0.85</td><td>0.77</td><td>0.66</td><td>0.49</td><td>4.20</td></tr><tr><td>+ Emu Edit (Sheynin et al., 2024)</td><td>0.93</td><td>0.85</td><td>0.73</td><td>0.58</td><td>0.42</td><td>4.01</td></tr><tr><td>+ FreqEdit (Liao et al., 2026)</td><td>0.85</td><td>0.57</td><td>0.35</td><td>0.19</td><td>0.11</td><td>3.91</td></tr><tr><td>+ VAÉ-LFA (Wang et al., 2026)</td><td>0.95</td><td>0.86</td><td>0.69</td><td>0.52</td><td>0.41</td><td>4.08</td></tr><tr><td>+ MT-OPSD</td><td>0.95</td><td>0.89</td><td>0.77</td><td>0.66</td><td>0.49</td><td>4.28</td></tr></table>

Qualitative Results. Figure 4 shows selected turns from one ten-turn editing session across three backbones. The base models produce reasonable results in the early turns, but gradually develop severe visual artifacts as their outputs are repeatedly reused as inputs. The training-free baselines provide limited improvement: Emu Edit and FreqEdit largely follow the same degradation pattern as the base models, while VAE-LFA better preserves image quality but weakens the requested colortemperature edit at Turn-5. In contrast, MT-OPSD remains stable through Turn-10 while following subsequent instructions, showing that long-horizon robustness can be improved without weakening the model’s ability to perform later edits. More detailed results are shown in Appendix F.

![](images/735f40fa031342b922e60a146c107abda50f75b074a1d84585a9ba681f9a7ac1.jpg)  
Figure 4: Qualitative comparison on LME-Bench. Repeated editing leads to distinct failure patterns across backbones, while the training-free baselines either inherit these artifacts or weaken the requested edits. MT-OPSD remains stable through 10 turns and continues to follow the editing sequence across all three backbones.

Table 3: Ablation on key components of MT-OPSD. SFT editing supervision replaces on-policy velocity matching with standard flow-matching supervision toward teacher-generated images. The curriculum ablation fixes the rollout depth at four, where the adaptive curriculum typically saturates.
<table><tr><td rowspan="2">Method</td><td colspan="5">LME-Bench</td><td colspan="2">MSE-Bench</td><td rowspan="2">ImgEdit Overall</td></tr><tr><td>SR@1</td><td>SR@5</td><td>SR@10</td><td>CR@5</td><td>CR@10</td><td>SR@1</td><td>SR@5</td></tr><tr><td>Base model</td><td>1.00</td><td>0.43</td><td>0.03</td><td>0.10</td><td>0.55</td><td>0.97</td><td>0.24</td><td>4.51</td></tr><tr><td>MT-OPSD</td><td>1.00</td><td>0.91</td><td>0.44</td><td>0.01</td><td>0.02</td><td>0.98</td><td>0.51</td><td>4.49</td></tr><tr><td>w/o identity branch</td><td>0.99</td><td>0.78</td><td>0.49</td><td>0.03</td><td>0.06</td><td>0.85</td><td>0.50</td><td>3.88</td></tr><tr><td>w/o editing branch</td><td>0.91</td><td>0.18</td><td>0.00</td><td>0.17</td><td>0.80</td><td>0.95</td><td>0.22</td><td>4.33</td></tr><tr><td>w/ SFT editing supervision</td><td>1.00</td><td>0.59</td><td>0.03</td><td>0.12</td><td>0.67</td><td>0.98</td><td>0.37</td><td>4.50</td></tr><tr><td>w/o rollout curriculum</td><td>1.00</td><td>0.79</td><td>0.17</td><td>0.04</td><td>0.19</td><td>0.96</td><td>0.41</td><td>4.33</td></tr><tr><td>w/o gated promotion</td><td>1.00</td><td>0.79</td><td>0.19</td><td>0.01</td><td>0.06</td><td>1.00</td><td>0.45</td><td>4.55</td></tr></table>

## 4.4 ABLATION STUDY

All ablations are conducted on Qwen-Image-Edit-2511 under the same training setup. Table 3 reports the quantitative results, with additional visual comparisons provided in Appendix G.

Effect of the Two-Branch Objective. The two training branches serve complementary roles. Removing the editing branch leaves only identity supervision and severely weakens editing ability, with SR@10 falling to 0.00 and CR@10 rising to 0.80. In contrast, removing the identity branch retains strong long-horizon editing performance and even yields a slightly higher SR@10 of 0.49, but makes the edits less controlled: CR@10 increases from 0.02 to 0.06, SR@1 on MSE-Bench drops from 0.98 to 0.85, and the ImgEdit score falls from 4.49 to 3.88. Qualitative results in Appendix G further show that this variant tends to over-edit, progressively altering untargeted content across turns. These results suggest that the editing branch preserves editability on self-generated states, while the identity branch suppresses unnecessary changes and preserves untargeted content.

Effect of On-Policy Supervision. Replacing on-policy velocity matching with standard flowmatching supervision on teacher-generated pseudo-targets reduces SR@5 from 0.91 to 0.59 and SR@10 from 0.44 to 0.03, while CR@10 rises from 0.02 to 0.67, surpassing the base model. Standard flow matching provides supervision along interpolation paths toward teacher-generated targets, whereas sparse velocity matching queries the teacher at states visited by the student. This suggests that supervision on student-visited states is more effective for long-horizon editing.

Effect of the Rollout Curriculum and Gated Promotion. These two components control how the training states and the supervision source evolve during training. Fixing the rollout depth at four turns reduces SR@10 from 0.44 to 0.17 and increases CR@10 from 0.02 to 0.19, showing that progressive depth expansion is more effective than training at a comparable fixed depth. Keeping the pretrained initialization as a fixed teacher instead maintains a low CR@10 of 0.06 but limits SR@10 to 0.19. At the same time, this variant achieves the highest ImgEdit score among the ablations at 4.55, consistent with the view that a fixed clean-condition reference better preserves the pretrained model’s single-turn behavior but limits further adaptation to long-horizon editing.

## 5 CONCLUSION

In this work, we study the failure of instruction-based editing models under multi-turn editing, where errors in self-generated outputs accumulate into severe degradation. We attribute this failure to a train–test mismatch in the conditioning distribution and introduce MT-OPSD, an on-policy selfdistillation framework that trains the model on self-generated conditioning states with editing supervision from a clean-conditioned teacher. We further introduce LME-Bench, a benchmark of 100 ten-turn editing sessions for evaluating long-horizon robustness. Experiments across three editing backbones show that MT-OPSD improves long-horizon robustness on LME-Bench and MSE-Bench while largely preserving single-turn editing quality on ImgEdit. More broadly, our results suggest that strong single-turn editing capability can be extended to multi-turn settings by learning to operate on self-generated states, without multi-turn annotations.

## REFERENCES

Eslam Abdelrahman, Liangbing Zhao, Tao Hu, Matthieu Cord, Patrick Perez, and Mohamed Elhoseiny. ToddlerDiffusion: Interactive structured image generation with cascaded schrodinger¨ bridge. In International Conference on Learning Representations, volume 2025, pp. 65334– 65368, 2025.

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from selfgenerated mistakes. In International Conference on Learning Representations, volume 2024, pp. 21246–21263, 2024.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. InstructPix2Pix: Learning to follow image editing instructions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18392–18402, 2023.

Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Aiming Hao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Yuming Jiang, et al. Z-Image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Zhen Fang, Wenxuan Huang, Yu Zeng, Yiming Zhao, Shuang Chen, Kaituo Feng, Yunlong Lin, Lin Chen, Zehui Chen, Shaosheng Cao, et al. Flow-OPD: On-policy distillation for flow matching models. arXiv preprint arXiv:2605.08063, 2026.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020.

Google. Nano Banana, 2025a. URL https://deepmind.google/models/ gemini-image/.

Google. Nano Banana Pro, 2025b. URL https://deepmind.google/models/ gemini-image/pro/.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NeurIPS), 2020.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. LoRA: Low-rank adaptation of large language models. Proceedings of the International Conference on Learning Representations (ICLR), 2022.

Jiahui Huang, Yasi Zhang, Tianyu Chen, Shu Wang, Jianwen Xie, Oscar Leong, Mingyuan Zhou, Nanzhu Wang, and Ying Nian Wu. MT-EditFlow: Reinforcement learning for multi-turn image editing with flow matching. arXiv preprint arXiv:2606.01985, 2026a.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. Advances in Neural Information Processing Systems, 38:167283–167308, 2026b.

Dengyang Jiang, Xin Jin, Dongyang Liu, Zanyi Wang, Mingzhe Zheng, Ruoyi Du, Xiangpeng Yang, Qilong Wu, Zhen Li, Peng Gao, et al. D-OPSD: On-policy self-distillation for continuously tuning step-distilled diffusion models. arXiv preprint arXiv:2605.05204, 2026.

Black Forest Labs. FLUX.2: Frontier visual intelligence, 2025. URL https://bfl.ai/blog/ flux-2.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 Kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Quanhao Li, Junqiu Yu, Kaixun Jiang, Yujie Wei, Zhen Xing, Pandeng Li, Ruihang Chu, Shiwei Zhang, Yu Liu, and Zuxuan Wu. DiffusionOPD: A unified perspective of on-policy distillation in diffusion models. arXiv preprint arXiv:2605.15055, 2026.

Yucheng Liao, Jiajun Liang, Kaiqian Cui, Baoquan Zhao, Haoran Xie, Wei Liu, Qing Li, and Xudong Mao. FreqEdit: Preserving high-frequency features for robust multi-turn image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 43525–43535, 2026.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2023.

Hongyu Liu, Chun Wang, Feng Gao, Xuanhua He, Yue Ma, Ziyu Wan, Yong Zhang, Xiaoming Wei, and Qifeng Chen. OPSD-V: On-policy self-distillation for post-training few-step autoregressive video generators. arXiv preprint arXiv:2607.08766, 2026.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1X-Edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6038–6047. IEEE, 2023.

OpenAI. GPT-Image-1, 2025. URL https://openai.com/index/ image-generation-api/.

OpenAI. GPT-Image-2, 2026. URL https://openai.com/index/ introducing-chatgpt-images-2-0/.

Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, and Massimo Caccia. Privileged information distillation for language models. arXiv preprint arXiv:2602.04942, 2026.

Leigang Qu, Feng Cheng, Ziyan Yang, Qi Zhao, Shanchuan Lin, Yichun Shi, Yicong Li, Wenjie Wang, Tat-Seng Chua, and Lu Jiang. VINCIE: Unlocking in-context image editing from video. In International Conference on Learning Representations, volume 2026, pp. 114873–114918, 2026.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695, 2022.

Hejian Sang, Yuanda Xu, Zhengze Zhou, Ran He, Zhipeng Wang, and Jiachen Sun. On-policy self-distillation for reasoning compression. arXiv e-prints, pp. arXiv–2603, 2026.

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu Edit: Precise image editing via recognition and generation tasks. In 2024 ieee/cvf conference on computer vision and pattern recognition (cvpr), pp. 8871–8879. IEEE, 2024.

Super Intelligence Team, Changhao Qiao, Chao Hui, Chen Li, Cunzheng Wang, Dejia Song, Jiale Zhang, Jing Li, Qiang Xiang, Runqi Wang, et al. FireRed-Image-Edit-1.0 technical report. arXiv preprint arXiv:2602.13344, 2026.

Xiaoce Wang, Sifan Zhou, Kaifei Wang, Leli Xu, Xuerui Qiu, Tao He, and Ming Li. Why do DiT editors drift? plug-and-play low frequency alignment in vae latent space. arXiv preprint arXiv:2605.08250, 2026.

Cong Wei, Zheyang Xiong, Weiming Ren, Xeron Du, Ge Zhang, and Wenhu Chen. OmniEdit: Building image editing generalist models through specialist supervision. In International Conference on Learning Representations, volume 2025, pp. 259–271, 2025.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-Image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. OmniGen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025b.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

Hang Xu, Xiaoxiao Ma, Guohui Zhang, Yu Hu, Siming Fu, Jie Huang, Lin Song, Haoyang Huang, Nan Duan, and Feng Zhao. AnchorEdit: Maintaining temporal consistency in multi-turn image editing via causal memory. arXiv preprint arXiv:2606.11751, 2026.

Yang Ye, Xianyi He, Zongjian Li, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, Li Yuan, et al. ImgEdit: A unified image editing dataset and benchmark. Advances in Neural Information Processing Systems, 38, 2026a.

Yuxiao Ye, Haoran He, Fangyuan Kong, Xintao Wang, Pengfei Wan, Kun Gai, and Ling Pan. Edit-R2: Context-aware reinforcement learning for multi-turn image editing. arXiv preprint arXiv:2606.05950, 2026b.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. MagicBrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.

Liangbing Zhao, Zicheng Zhang, Xuecheng Nie, Luoqi Liu, and Si Liu. Cross-attention and seamless replacement of latent prompts for high-definition image-driven video editing. Electronics, 13 (1):7, 2023.

Liangbing Zhao, Le Zhuo, Sayak Paul, Hongsheng Li, and Mohamed Elhoseiny. From statics to dynamics: Physics-aware image editing with latent transition priors. arXiv preprint arXiv:2602.21778, 2026a.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026b.

Wei Zhou, Xiongwei Zhu, Lingdong Kong, Bo Chen, Lei Zhang, Yongyuan Liang, Xiaoxia Hou, Ye Tian, Xian Sun, Yingshuo Wang, et al. On-policy self-distillation in diffusion models. arXiv preprint arXiv:2608.24646, 2026a.

Wei Zhou, Xiongwei Zhu, Zelin Xu, Bo Dong, Lixue Gong, Yongyuan Liang, Meng Chu, Leigang Qu, Lingdong Kong, Wei Liu, et al. DanceOPD: On-policy generative field distillation. arXiv preprint arXiv:2606.27377, 2026b.

Zijun Zhou, Yingying Deng, Xiangyu He, Weiming Dong, and Fan Tang. Multi-turn consistent image editing. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15792–15801. IEEE, 2025.

Le Zhuo, Liangbing Zhao, Sayak Paul, Yue Liao, Renrui Zhang, Yi Xin, Peng Gao, Mohamed Elhoseiny, and Hongsheng Li. From reflection to perfection: Scaling inference-time optimization for text-to-image diffusion models via reflection tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15329–15339, 2025.

## A IMPLEMENTATION DETAILS

Table 4 lists the full configuration for the three backbones. LoRA is applied to the attention and feedforward projections of the diffusion transformer, and all other components remain frozen. Entries that need further definition are described below.

• Query distribution. The editing branch uses a training-time trajectory of N steps, independent of the sampling steps used for identity rollout. We sample two distinct query positions per update from a Beta distribution over the normalized trajectory index, following DanceOPD (Zhou et al., 2026b). Qwen-Image-Edit-2511 and FireRed-Image-Edit use Beta(5, 5), while FLUX.2-klein-base uses Beta(2, 5). Unlike DanceOPD, which reports Beta(5, 2) as the best setting, we find queries around the middle or the earlier, noisier part of the trajectory to work better.

• Identity loss weight. The identity loss is scaled by this factor relative to the editing loss, on top of the branch sampling ratio.

• Drift threshold and patience. Rollout drift is the mean absolute pixel difference between the current rollout state and the clean source image, computed in the 0 to 255 range. Only steps at the current rollout depth contribute. The depth increases by one turn once the drift stays at or below the threshold for patience consecutive such steps, after which the counter resets.

Table 4: Implementation details of MT-OPSD across the three backbones.
<table><tr><td></td><td>Qwen-Image-Edit-2511</td><td>FireRed-Image-Edit</td><td>FLUX.2-klein-base</td></tr><tr><td>LoRA and optimization</td><td></td><td></td><td></td></tr><tr><td>LoRA rank</td><td>32</td><td>32</td><td>32</td></tr><tr><td>LoRA α</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>1 × 10−4</td></tr><tr><td>Learning-rate schedule</td><td>constant</td><td>constant</td><td>constant</td></tr><tr><td>Training samples</td><td>2,000</td><td>2,000</td><td>2,000</td></tr><tr><td>Training resolution</td><td>5122</td><td>5122</td><td>5122</td></tr><tr><td>Identity rollout</td><td></td><td></td><td></td></tr><tr><td>Sampling steps</td><td>30</td><td>30</td><td>50</td></tr><tr><td>Guidance scale</td><td>4.0</td><td>4.0</td><td>4.0</td></tr><tr><td>Editing branch</td><td></td><td></td><td></td></tr><tr><td>Trajectory steps N</td><td>10</td><td>10</td><td>10</td></tr><tr><td>Query steps</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Query distribution  $p _ { q }$ </td><td>Beta(5,5)</td><td>Beta(5, 5)</td><td>Beta(2, 5)</td></tr><tr><td>Guidance scale</td><td>4.0</td><td>4.0</td><td>4.0</td></tr><tr><td>Objective and curriculum</td><td></td><td></td><td></td></tr><tr><td>Identity loss weight</td><td>10</td><td>10</td><td>10</td></tr><tr><td>Initial rollout depth</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Maximum rollout depth  $K _ { \mathrm { m a x } }$ </td><td>10</td><td>10</td><td>10</td></tr><tr><td>Drift threshold</td><td>9.0</td><td>10.0</td><td>10.0</td></tr><tr><td>Patience</td><td>10</td><td>10</td><td>10</td></tr></table>

## B TRAINING RECIPE

Algorithm 1 summarizes the training procedure of MT-OPSD. Each step draws a single source image and instruction, constructs the on-policy rollout state with the current student, applies one of the two branches of the training objective, and updates the rollout curriculum. Teacher promotion runs asynchronously and is described in Appendix C.

Algorithm 1 Training Procedure of MT-OPSD.   
Require: Training set $\mathcal { D } ;$ pretrained editing model $\theta _ { 0 } ;$ identity instruction $e _ { \mathrm { i d } }$   
Require: Editing probability $\rho ;$ identity weight $\lambda ;$ query distribution $p _ { q } ;$ ; initial and maximum roll  
out depth ${ k _ { 0 } , K _ { \operatorname* { m a x } } }$   
Require: Drift threshold τ and patience $P$   
1: Initialize student $\theta _ { S }  \theta _ { 0 }$ and teacher $\theta _ { T }  \theta _ { 0 }$   
2: Initialize rollout depth $k  k _ { 0 }$ and counter $c \gets 0$   
3: while not converged do   
4: Sample $( I ^ { ( 0 ) } , e ) \sim \mathcal { D }$   
Stage 1: Self-Generated Rollout   
5: $\tilde { I } ^ { ( 0 ) }  I ^ { ( 0 ) }$   
6: for $j = 1 , \dots , k$ do   
7: $\tilde { \cal I } ^ { ( j ) } \gets G _ { \theta _ { S } } ( \tilde { I } ^ { ( j - 1 ) } , e _ { \mathrm { i d } } )$ ▷ no gradient   
8: end for   
Stage 2: Two-Branch Training   
9: Sample $b \sim$ Bernoulli(ρ)   
10: if $b = 1$ then ▷ editing branch   
11: $x _ { t _ { N } } \sim \mathcal { N } ( 0 , \mathbf { I } )$   
12: $\{ \ddot { x _ { t _ { i } } } \} _ { i = 0 } ^ { N } \gets \mathrm { R o l l o u t } ( v _ { \theta _ { S } } ; x _ { t _ { N } } , \tilde { I } ^ { ( k ) } , e )$ ▷ no gradient   
13: $\mathcal { Q } \gets \mathrm { \tilde { S a m p l e } } ( p _ { q } , 2 )$   
14: $\mathcal { L } _ { \mathrm { e d i t } }  \frac { 1 } { \vert \mathcal { Q } \vert } \sum _ { x \in \mathcal { O } }  v _ { \theta _ { S } } ( \bar { x } _ { t _ { q } } , t _ { q } , \tilde { I } ^ { ( k ) } , e ) - v _ { \theta _ { T } } ( \bar { x } _ { t _ { q } } , t _ { q } , I ^ { ( 0 ) } , e )  _ { 2 } ^ { 2 } ;$   
q∈Q   
where $\bar { x } _ { t _ { q } } = \mathrm { s g } ( x _ { t _ { q } } )$   
15: $\mathcal { L }  \dot { \mathcal { L } } _ { \mathrm { e d i t } }$   
16: else ▷ identity branch   
17: $\tilde { x } _ { 0 } \gets \mathrm { L a t e n t } ( \tilde { I } ^ { ( k ) } )$   
18: Sample t and $\dot { \epsilon } \sim \mathcal { N } ( 0 , \bf { I } )$   
19: $\tilde { x } _ { t } \gets ( 1 - t ) \tilde { x } _ { 0 } + t \epsilon$   
20: $\mathcal { L } _ { \mathrm { i d } } \gets \Big | \Big | { v _ { \theta _ { S } } } ( \tilde { x } _ { t } , t , \tilde { I } ^ { ( k ) } , e _ { \mathrm { i d } } ) - ( \epsilon - \tilde { x } _ { 0 } ) \Big | \Big | _ { 2 } ^ { 2 }$   
21: $\mathcal { L }  \lambda \ddot { \mathcal { L } } _ { \mathrm { i d } }$   
22: end if   
23: Update $\theta _ { S }$ by minimizing L   
Stage 3: Adaptive Rollout Curriculum   
24: d ← MeanAbsDif $( \tilde { I } ^ { ( k ) } , I ^ { ( 0 ) } )$   
25: if $d \leq \tau$ then   
26: $c \gets c + 1$   
27: else   
28: $c \gets 0$   
29: end if   
30: if $c \geq P$ and $k < K _ { \operatorname* { m a x } }$ then   
31: $k  k + 1 ; c  0$   
32: end if   
Stage 4: Gated Teacher Promotion   
33: if the asynchronous evaluator promotes checkpoint $\theta ^ { \star }$ then   
34: $\theta _ { T }  \theta ^ { \star }$ ▷ Appendix C   
35: end if   
36: end while

## C TEACHER PROMOTION RULE

Gate Set. Teacher promotion is determined using a held-out gate set of 23 ten-turn editing sessions. The sessions are constructed following the same procedure and source-image generator as LME-Bench, but use a disjoint set of source images. Fifteen sessions follow the standard LME-Bench composition of six local and four global edits, while the remaining eight contain local edits only. The gate set is used solely for teacher selection and is excluded from all reported evaluation results. Checkpoints are evaluated using the same protocol and metrics described in Appendix D.2.

Asynchronous Evaluation. We save a student checkpoint every 50 training samples and evaluate each checkpoint asynchronously on four separate GPUs. Training proceeds without waiting for evaluation. Once a checkpoint is promoted, the decision is written to disk and picked up by the training process at its next polling step, where the current teacher is replaced by the promoted checkpoint.

Promotion Rule. Each candidate checkpoint is compared with the current teacher at turns 10, 8, and 5, in this order, giving priority to longer editing horizons. At turn $k ,$ we define $\Delta \mathrm { S R } _ { k }$ as the number of gate sessions successfully completed through turn k by the candidate minus that of the teacher. Similarly, $\Delta \mathrm { C R } _ { k }$ is defined as the number of sessions collapsed by turn k under the teacher minus that under the candidate. Thus, positive values favor the candidate for both quantities.

$\mathrm { I f } - 1 \le \Delta \mathrm { S R } _ { k } < \alpha \mathrm { a n d } - 1 \le \Delta \mathrm { C R } _ { k } < \beta ,$ the comparison is inconclusive at turn k, since neither quantity regresses by more than one session and neither reaches its promotion threshold, and the comparison proceeds to the next turn. Otherwise, the candidate is promoted if either

$$
\begin{array} { l l l } { \Delta \mathrm { S R } _ { k } \geq \alpha } & { \mathrm { a n d } } & { \Delta \mathrm { C R } _ { k } \geq - 1 } \\ { \Delta \mathrm { C R } _ { k } \geq \beta } & { \mathrm { a n d } } & { \Delta \mathrm { S R } _ { k } \geq 1 } \end{array}
$$

(success-led),

(stability-led)

is satisfied, and the current teacher is retained if neither condition holds. If all three turns are inconclusive, no promotion is made. We set $( \alpha , \beta )$ to (2, 5) for Qwen-Image-Edit-2511, (2, 3) for FLUX.2-klein-base, and (3, 5) for FireRed-Image-Edit. In practice, this rule typically results in one or two teacher promotions during each training run.

## D LME-BENCH AND EVALUATION DETAILS

This section provides additional details on the construction of LME-Bench and its evaluation protocol.

## D.1 BENCHMARK CONSTRUCTION

Source Images. We generate all source images with Z-Image-Turbo at 1024 × 1024 resolution using 9 sampling steps and no classifier-free guidance. Each image is seeded by its benchmark index to ensure exact reproducibility. The 100 sessions are evenly distributed across 10 categories: pets, wild animals, people, food, vehicles, interiors, nature, objects, plants, and urban scenes.

Session Structure. Each session contains 10 turns, with six local and four global edits. Local edits add, remove, replace, or modify a specific object or region, while global edits affect the image as a whole. We further divide global edits into hard globals, which change the rendering medium or apply a strong global transformation such as oil painting, anime, or grayscale, and soft globals, which adjust properties such as color temperature or brightness. Each session contains either three hard and one soft global edit, or two of each. The first turn is always a hard global edit, no two global edits are adjacent, and the final two turns are local. This arrangement introduces substantial changes early in the sequence while ensuring that later local edits operate on images that have already undergone several global transformations. Instructions never require access to an earlier image state or explicit comparison with previous turns; all referenced content must be identifiable from the immediately preceding image. Table 6 shows one complete session.

Validity Rules. Unlike single-turn editing, instruction validity in a multi-turn session depends on the preceding edits. For example, an object referenced by a later instruction may have been removed earlier, or the requested state may already hold. We therefore construct each session under an explicit set of rules that avoids non-informative turns, where either leaving the image unchanged can satisfy the instruction or success cannot be judged reliably. The rules are grouped into six families, summarized in Table 5. Among them, ALREADY-TRUE, REFERENT, and INTERFERENCE specifically address dependencies across editing turns.

Table 5: Validity rules used to construct LME-Bench. The rules are grouped into six families according to the type of invalid or non-informative turn they prevent.
<table><tr><td>Family</td><td>Invalidity condition</td><td>Example rule</td></tr><tr><td>STRUCTURE</td><td>The chain violates the session schema</td><td>No two global edits are adjacent</td></tr><tr><td>VISIBILITY</td><td>The requested change is not visually verifiable</td><td>Counts in add/remove instructions are at most three</td></tr><tr><td>ALREADY-TRUE</td><td>The requested state already holds</td><td>A pose edit must request the opposite of the current state</td></tr><tr><td>REFERENT</td><td>The referenced object cannot be resolved</td><td>After a container is replaced, later turns may not refer to the old one</td></tr><tr><td>INTERFERENCE</td><td>Another turn would cancel or invalidate the edit</td><td>A grayscale or sepia edit must be the last global edit</td></tr><tr><td>FEASIBILITY</td><td>The requested change cannot be rendered reliably in the frame</td><td>An added object requires sufficient visible space at its implied size</td></tr></table>

Table 6: Example ten-turn session from LME-Bench.
<table><tr><td>Turn</td><td>Type</td><td>Instruction</td></tr><tr><td>1</td><td>hard global</td><td>Repaint the whole image as a textured oil painting with visible brush- strokes, keeping the dog and the room.</td></tr><tr><td>2</td><td>local</td><td>Change the wall behind the dog to a warm sage-green color.</td></tr><tr><td>3</td><td>hard global</td><td>Redraw the whole image in a bold anime illustration style with clean lines and cel shading.</td></tr><tr><td>4</td><td>local</td><td>Add a rubber ball on the floor beside the dog.</td></tr><tr><td>5</td><td>soft global</td><td>Shift the whole image to a cool blue-gray color temperature.</td></tr><tr><td>6</td><td>local</td><td>Make the dog lie down on the floor instead of sitting.</td></tr><tr><td>7</td><td>hard global</td><td>Convert the entire image to black and white, full grayscale with no color.</td></tr><tr><td>8</td><td>local</td><td>Change the wooden floor into pale marble tiles.</td></tr><tr><td>9</td><td>local</td><td>Add a small potted plant in the corner behind the dog.</td></tr><tr><td>10</td><td>local</td><td>Remove the rubber ball from the floor.</td></tr></table>

## D.2 EVALUATION PROTOCOL AND METRICS

Execution. Each session is executed sequentially using the standard inference configuration of each backbone. At every turn, the model receives only the output from the preceding turn as its source image; the original image is never reintroduced.

Instruction Judging. We use GPT-4o to evaluate instruction execution at each turn. Given the source image, all instructions up to the current turn, and the corresponding intermediate outputs, the judge assigns integer scores from 0 to 10 for prompt following and consistency. Prompt follow ing measures whether the requested change is realized relative to the immediately preceding image, rather than whether the current image merely contains the requested content. Consistency is defined according to the edit type. For local edits, it measures unintended changes outside the requested modification; for global edits, it measures whether the main subjects, their identities, and the composition remain consistent under the requested transformation. A turn is considered successful when both scores are at least 7.

Degradation Judging. A separate evaluation pass measures visual degradation independently of instruction execution. Given the source image, the current output, and the instruction history, the judge scores two aspects from 0 to 2: content loss, which measures whether the main subjects and scene remain recognizable, and surface corruption, which measures visual artifacts not requested by the instructions. Requested style changes are explicitly excluded from degradation. A turn is considered degraded when the two scores sum to at least 2. The complete judging prompts and scoring criteria are provided in Appendix D.3.

Metrics. SR@k denotes the fraction of sessions for which all turns up to turn k are successful. Following the early-stopping protocol of MSE-Bench, instruction evaluation stops after the first failed turn in each session. CR@k denotes the fraction of sessions that have collapsed by turn k. A session is considered collapsed once two consecutive turns are judged degraded and remains collapsed thereafter. Requiring two consecutive degraded turns reduces sensitivity to occasional judging noise. Both metrics are normalized by the total number of sessions.

## D.3 JUDGE PROMPTS

Listing 1 shows the prompt used for instruction judging and Listing 2 the prompt used for degradation judging. Fields in braces are filled in per turn.

Listing 1: Prompt for instruction judging.  
Assume you are an expert in evaluating multi-turn image editing. In   
this task, a user interacts with an image editing system across   
multiple turns. At the first turn, the user provides a source image   
and an editing prompt. The system returns the edited image. In each   
subsequent turn, the user supplies a new prompt, and the system   
generates a new image based on the output from the previous turn.   
Your goal is to evaluate how successfully the editing instruction of   
the LAST turn (turn-{num\_prompts}) has been executed.   
You will be given {num\_prompts} user editing prompts and {num\_images}   
images: the first image is the original source image, and the next   
are the edited results from each turn for each prompt.   
You should focus more on the last prompt and the last edited image, but   
you may also consider the previous prompts and images as context.   
The {num\_prompts} user editing prompts are: {editing\_prompt}   
Please follow these evaluation rules. For the LAST turn, assess TWO   
criteria by giving a reason, then assign an integer score from 0 to   
10 for each:   
1) prompt\_following: does the last edited image fulfill the last user’s   
editing prompt? Judge the DELTA -- compare the last edited image   
against the IMMEDIATELY PRECEDING image (the second-to-last image   
provided) and verify that the SPECIFIC change the instruction asks   
for actually happened, instead of merely checking whether the final   
image happens to contain the described element. Apply this generally   
to the instruction type:   
- Add a quantity ("add a second / another / one more X"): the COUNT of   
X must INCREASE by that amount versus the previous image (e.g. one X   
becomes two). An X that was already there does not count -- there   
must be a newly added one.   
Remove a quantity ("remove one / a X"): the count of X must DECREASE   
by one versus the previous image -- not "all of the X disappeared",   
and not "the X is unchanged".   
Change an attribute ("change X to <color/material/style>"): that   
specific X must actually take on the new attribute (the existing   
object changes), not merely that some object with that attribute now   
appears.   
Add or remove a target: the specific named object must actually   
appear (for add) or disappear (for remove) versus the previous image   
If the last image is essentially unchanged from the previous image   
where the instruction demanded a change, prompt\_following must be   
low. (0 indicates the required change did not happen at all; 10   
indicates it was executed exactly as asked.)   
2) consistency: are the parts that should be preserved still consistent   
with the input reference (the source image at the first turn, or   
the result image at the previous turn)?   
First decide whether the last instruction is a LOCAL edit (adds,   
removes, or alters a specific object or region) or a GLOBAL edit (   
intentionally changes the whole image -- lighting, time of day,   
weather, season, color grade, or art style, e.g. "make it nighttime   
", "convert to black and white", "make it a sunset scene").   
LOCAL edit: consistency rates the degree of overediting. 0 indicates   
the scene is needlessly different from the reference; 10 indicates a   
minimal yet effective edit that leaves everything else intact.

GLOBAL edit: the whole image is SUPPOSED to change, so do NOT   
penalize the global change itself. Instead, consistency rates   
whether the main subjects, their identities, the composition, and   
the scene structure are preserved THROUGH the transform. 0 indicates   
the subjects or composition are lost, replaced, or distorted; 10   
indicates the same scene and subjects faithfully re-rendered under   
the requested global change.   
3) Return your results in a JSON structure, following this format:   
{"reason": "...", "prompt\_following": int, "consistency": int}

## Listing 2: Prompt for degradation judging.

You are auditing one frame of a multi-turn image-editing sequence for   
VISUAL DEGRADATION.   
You are given the ORIGINAL SOURCE image first, and the CURRENT image   
last.   
These editing instructions have been applied, in order, to get from the   
source to the current image:   
{instructions}   
Some of those instructions deliberately change how the picture LOOKS (   
repaint as pixel art, as watercolour, as an oil painting, convert to   
black and white, shift the colour temperature, change the season or   
weather). A picture that is a COMPETENT rendering in the requested   
style is NOT degraded, however far it is from a photograph. Blocky   
flat colour is correct for pixel art; visible brushstrokes are   
correct for an oil painting. Judge craftsmanship, not photographic   
realism.   
Degradation is the opposite: damage no competent artist would produce.   
It includes heavy noise, rainbow or colour speckle, blur, melting or   
duplicated structure, extreme oversaturation, lost detail or   
sharpness, collapse into noise -- and also: UNREQUESTED flat colour   
patches replacing texture, hard-edged colour confetti or fragments   
scattered over the picture, a background that has stopped depicting   
a place and become an abstract pattern, and a subject reduced to a   
featureless silhouette.   
Answer these two questions about the CURRENT image, then give a brief   
reason.   
1. content\_loss -- can the things that should be in the picture still   
be identified?   
0 = every main object and the setting are identifiable   
1 = at least one main object, or the setting, can no longer be made   
out   
2 = the picture no longer reads as the scene at all   
2. surface\_corruption -- is the picture’s surface damaged in ways the   
instructions did not ask for?   
0 = none worth noting   
1 = clearly present, but the picture survives it   
2 = dominates the frame; the frame reads as noise or debris rather   
than a picture   
Return JSON only: {"reason": "...", "content\_loss": int, "   
surface\_corruption": int}

## E COMPARISON WITH VINCIE

Among the training-based multi-turn editing methods discussed in Section 2, VINCIE (Qu et al., 2026) is the only one for which we could obtain both public code and checkpoints. VINCIE learn in-context editing from video and, in its native setting, conditions each turn on the full editing history, including the clean source image. We therefore evaluate VINCIE-7B both in this native full-history setting and under the previous-turn protocol used throughout our experiments, where each turn receives only the immediately preceding output. Table 7 compares these settings with the three backbones trained with MT-OPSD, all evaluated with previous-turn conditioning.

Table 7: Comparison with VINCIE under different conditioning histories. VINCIE is eval uated in its native full-history setting and under the previous-turn protocol used throughout our experiments. All MT-OPSD variants use previous-turn conditioning.
<table><tr><td rowspan="2">Method</td><td colspan="5">LME-Bench</td><td colspan="2">MSE-Bench</td><td rowspan="2">ImgEdit Overall</td></tr><tr><td>SR@1</td><td>SR@5</td><td>SR@10</td><td>CR@5</td><td>CR@10</td><td>SR@1</td><td>SR@5</td></tr><tr><td>VINCIE-7B (full history) VINCIE-7B (previous turn)</td><td>0.81 0.84</td><td>0.64 0.15</td><td>0.47 0.09</td><td>0.06 0.17</td><td>0.08 0.46</td><td>0.95 0.91</td><td>0.49 0.30</td><td>3.86 3.86</td></tr><tr><td>Qwen-Image-Edit-2511 + MT-OPSD</td><td>1.00</td><td>0.91</td><td>0.44</td><td>0.01</td><td>0.02</td><td>0.98</td><td>0.51</td><td>4.49</td></tr><tr><td>FireRed-Image-Edit + MT-OPSD</td><td>1.00</td><td>0.88</td><td>0.52</td><td>0.00</td><td>0.03</td><td>0.99</td><td>0.69</td><td>4.52</td></tr><tr><td>FLUX.2-klein-base + MT-OPSD</td><td>0.99</td><td>0.76</td><td>0.38</td><td>0.00</td><td>0.04</td><td>0.95</td><td>0.49</td><td>4.28</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Under the matched previous-turn protocol, VINCIE degrades sharply on LME-Bench: its SR@10 drops from 0.47 under full-history conditioning to 0.09, while CR@10 increases from 0.08 to 0.46. This suggests that access to the full editing history contributes substantially to its long-horizon performance. Under the same previous-turn protocol, MT-OPSD achieves higher SR@5 and SR@10 and lower collapse rates across all three backbones. Even compared with VINCIE in its native full-history setting, MT-OPSD obtains higher SR@5 and lower CR@5 and CR@10 on all three backbones, while the SR@10 comparison varies across backbones. On MSE-Bench, MT-OPSD matches or exceeds VINCIE at SR@5 on all three backbones. It also achieves higher single-turn ImgEdit scores than VINCIE.

## F ADDITIONAL QUALITATIVE RESULTS

Figures 5 and 6 show two additional LME-Bench sessions, each covering all three backbones and all comparison methods. Across both sessions, the base models and the three training-free baselines accumulate visible artifacts as editing proceeds, including high-frequency color speckle, fragmented texture, and loss of contrast, and both the dominant failure mode and the turn at which it appears differ across backbones. MT-OPSD follows the instruction at every turn on all three backbones, and its outputs remain largely free of such artifacts throughout the sequence.

## G QUALITATIVE ABLATION RESULTS

Figure 7 compares MT-OPSD with the variant trained without the identity branch on two LME-Bench sessions. Both models follow the requested edits throughout the sequence, consistent with the relatively high success rate of the variant in Table 3. The main difference lies in the preservation of content that is not targeted by the instructions. In the first session, the chef’s appearance begins to drift from Turn-1 and changes progressively across subsequent edits, leading to a substantial identity shift by Turn-10. In the second session, the aircraft changes position and orientation from the first turn, with further layout drift accumulating over later turns. Without the identity branch, repeated editing introduces unnecessary changes to both subject identity and scene layout. In contrast, MT-OPSD follows the same editing sequence while better preserving subject identity and the overall composition across all ten turns. These observations are consistent with the lower single-turn performance of the variant on MSE-Bench and ImgEdit reported in Table 3.

![](images/d95986a0e238b1d8ac2139bc3570b294f3fa7107631d8de5ea6c9632a8586771.jpg)  
Figure 5: Additional qualitative comparison on LME-Bench. Selected turns of one editing session across three backbones.

![](images/81e48d704f79870e983a6fdac93e50e574e9ef25ce8a3404483da893386eb52d.jpg)  
Figure 6: Additional qualitative comparison on LME-Bench. Selected turns of one editing session across three backbones.

![](images/448c93ba5b3f8ba7dee23583e6df015c4c0d027f95d5320ece08ad840e43ad98.jpg)  
Figure 7: Effect of the identity branch. Removing the identity branch preserves instruction following but leads to progressive identity and layout drift across turns. MT-OPSD better preserves untargeted content while following the same ten-turn editing sequences.