# PE-OPSD: INTERNALIZING PROMPT ENHANCEMENT INTO FLOW-MATCHING MODELS VIA ON-POLICY SELF-DISTILLATION

Mingfeng Lin<sup>1∗</sup> Chengfei Cai<sup>2∗</sup> Lin Xu<sup>1</sup> Chengqian Ma<sup>3</sup> Yuxiang Wei<sup>4</sup> Liang Han<sup>1†</sup> <sup>1</sup>Harbin Institute of Technology (Shenzhen) <sup>2</sup>Zhejiang University

<sup>3</sup>Peking University <sup>4</sup>Harbin Institute of Technology

§ Code: https://github.com/sleepy1231/PE-OPSD

## ABSTRACT

Text-to-image users often provide concise and underspecified prompts, whereas generative models benefit from detailed textual conditions for reliable instruction following. Existing systems bridge this gap with Prompt Enhancers (PEs) that rewrite raw prompts at inference time, introducing additional latency and leaving prompt elaboration external to the generator. We instead view enhanced prompts as privileged training information and ask whether their benefits can be internalized. We propose Prompt-Enhanced On-Policy Self-Distillation (PE-OPSD) for text-to-image flow-matching models. During training, a raw-prompt student follows its own generation trajectory, while an enhanced-prompt teacher provides vector-field targets at the states visited by the student. This on-policy supervision distills the behavior induced by enhanced prompts into the raw-prompt student without requiring additional text–image pairs. At inference, both the PE and teacher are removed, and the student generates directly from raw prompts. Across multiple model families, PEs, and benchmarks, PE-OPSD achieves the strongest aggregate prompt fidelity among the evaluated baselines, yields positive aggregate visual appeal gains, and retains the base-model inference efficiency.

## 1 INTRODUCTION

User prompts are often concise descriptions rather than fully specified generation instructions (Xie et al., 2023; Hahn et al., 2024). A prompt such as “a girl reading under a tree” specifies the main subject and scene while leaving pose, lighting, composition, and visual style open. Although these unspecified choices admit multiple valid realizations, explicitly instantiating them can help generative models produce visually coherent outputs and follow the stated constraints more reliably. Despite substantial progress in visual fidelity and sampling efficiency (Zhang et al., 2023; Yin et al., 2024; Jiang et al., 2026b), reliable intent understanding remains difficult when prompts are short and underspecified (Huang et al., 2026). This creates a fundamental mismatch between concise user prompts and the detailed textual conditions preferred by generative models.

To bridge this gap, many industrial text-to-image systems employ a Prompt Enhancer (PE) before image generation (Team, 2025; Zhao et al., 2026a). Given a raw prompt, PE rewrites it into a longer description that makes implicit subject attributes, scene layouts and object relationships explicit. Prior works such as Promptist (Hao et al., 2023), BeautifulPrompt (Cao et al., 2023), and PromptEnhancer (Wang et al., 2025) have shown that prompt rewriting can improve human preference, text–image relevance, attribute binding, and compositional relationships (Ghosh et al., 2023). However, PE improves generation by modifying the input rather than improving the generator itself: prompt enhancement remains delegated to an additional inference-time module.

This deployment paradigm has practical and conceptual limitations. Practically, prompt rewriting introduces extra latency and computation, while longer prompts increase text-processing and conditioning overhead (Wang et al., 2023). Conceptually, an ideal generator should not require an external rewriter for every generation. It is therefore desirable for a generator to internalize the generation behavior induced by PEs and generate high-quality images directly from concise prompts.

![](images/54ee45b920228af03485267983899f860dc77a4d4af470e7bd97c6ed3397b246.jpg)

![](images/a4a3856144907349d95cd2b7d170926f53e3854fdddbede988256bd3f489f682.jpg)

![](images/583f963715b25e70078463bbee9612c3d2e43472ba4cface198c16c4f1e5db62.jpg)  
Figure 1: Comparison of generation quality, throughput, and training dynamics. PE-OPSD achieves higher GenEval scores than both Vanilla and PE while preserving the inference efficiency. Moreover, PE-OPSD achieves faster convergence and greater performance gains than SFT and off-policy distillation. Training dynamics is reported in Z-Image-Turbo training.

This raises a natural question: Can a generator capture the generation benefits elicited by prompt enhancement while receiving only the raw prompt at inference time? Existing approaches exploit richer textual conditions mainly in two ways. Some replace original text conditions with expanded captions during training (Betker et al., 2023), which improves training data quality but does not directly teach the model to recover enhanced-prompt behavior from raw prompts. Others keep the generator fixed and continuously invoke PE at inference time (Cao et al., 2023; Manas et al.˜ , 2024; Wang et al., 2025), which is effective but retains the deployment cost. Thus, less attention has been paid to internalizing the benefit of prompt enhancement into the generator itself.

In this work, we reinterpret prompt enhancement as a form of privileged information. Unlike conventional privileged information from additional modalities, an enhanced prompt elaborates the raw prompt while preserving its explicit semantic constraints. Crucially, we do not feed this privileged text to the student. Instead, it is used only to define a better-informed teacher condition, while the student must learn to reproduce the corresponding generation behavior from the raw prompt alone.

Building on this perspective, we propose Prompt-Enhanced On-Policy Self-Distillation (PE-OPSD) for text-to-image flow-matching models. Given a raw prompt p, the PE produces an enhanced prompt $p ^ { + }$ . The student is conditioned only on $p ,$ while the teacher receives $p ^ { + }$ as privileged information. During training, the student follows its own vector field to generate on-policy flow trajecto ries. At the states visited by the student, the teacher provides target vector fields conditioned on $p ^ { + }$ By matching these targets under the raw-prompt condition, the student learns to approximate the generation behavior induced by enhanced prompts. Unlike supervised finetuning (SFT), PE-OPSD does not require constructing additional text-image pairs. Unlike off-policy distillation based on teacher trajectories, it supervises the student on its own raw-prompt trajectories, better aligning the training signal with the states encountered during raw-prompt generation. In this way, PE-OPSD converts PE from an inference-time module into a training-time supervision signal on flow dynamics. After training, both PE and teacher are removed, enabling direct generation from raw prompts without inference-time prompt rewriting.

Our contributions are summarized threefold:

• We reinterpret prompt enhancement as privileged information for text-to-image generation, studying how the benefits of enhanced prompts can be transferred to a model that observes only raw prompts at inference time.

• We propose Prompt-Enhanced On-Policy Self-Distillation (PE-OPSD) for text-to-image flow-matching models, where an enhanced-prompt teacher provides dense vector-field supervision on states visited by a raw-prompt student, thereby distilling enhanced-prompt generation behavior into a model conditioned only on raw prompts.

• We show that PE-OPSD outperforms inference-time PE, SFT, and off-policy distillation in aggregate prompt fidelity on all three main backbones, while retaining positive aggregate visual-appeal changes and base-model inference latency.

![](images/5110c5d9753e5fa33bb67bc1dbd2eef3bdc1b195b330fa8b2308722b1113b4a2.jpg)  
Figure 2: Three different uses of prompt conditioning. (a) Direct generation from raw prompts results in limited prompt alignment; (b) Inference-time PE improves prompt alignment by rewriting raw prompts, but introduces additional computational overhead; (c) Our PE-OPSD internalizes the PE knowledge into the generator, improving alignment without additional inference cost.

## 2 RELATED WORK

Prompt enhancement for text-to-image generation. Prompt enhancement aims to bridge the gap between underspecified user prompts and the detailed textual conditions under which text-toimage models more reliably satisfy explicit prompt constraints and produce visually coherent outputs. Promptist (Hao et al., 2023) learns model-preferred prompts with reinforcement learning, BeautifulPrompt (Cao et al., 2023) trains a PE from low- and high-quality prompt pairs with visual feedback, and PromptEnhancer (Wang et al., 2025) further improves PE through chain-of-thought reasoning and fine-grained reward signals. However, these improvements are largely achieved outside the generator by training an external PE. In contrast, our work uses enhanced prompts only as training-time privileged information and distills their effect into the generator, enabling inference directly from raw prompts.

On-policy distillation. On-policy distillation (OPD) (Agarwal et al., 2024; Gu et al., 2024) mitigates train–inference mismatch by supervising the student on samples generated from its own current policy rather than from a fixed offline distribution. On-policy self-distillation (OPSD) (Zhao et al., 2026b) extends this idea by constructing asymmetric teacher and student views from the same base model, reducing the need for a separate stronger teacher. In recent visual and multimodal methods (Bi et al., 2026; Liu et al., 2026b; Yuan et al., 2026), this asymmetry is often induced by privileged information available only to the teacher during training, such as cropped regions, higher resolution inputs, or visual reasoning traces. D-OPSD (Jiang et al., 2026a) further introduces this paradigm to step-distilled diffusion models, but relies on paired image–text data and a generator capable of accepting image-conditioned inputs. PE-OPSD differs from these methods by using enhanced prompts only as training-time privileged information, thereby transferring their effect to a raw-prompt flow-matching model without requiring external prompt enhancement at inference.

## 3 METHODOLOGY

## 3.1 PRELIMINARIES

On-policy distillation. Let $f _ { \theta }$ denote a student model and $f _ { \phi }$ denote a teacher model. Conventional knowledge distillation (Hinton et al., 2015; Beyer et $\mathrm { { a l . } }$ , 2022) supervises the student on examples drawn from a fixed data distribution, which may differ from the states encountered by the student at inference. On-policy distillation instead evaluates the teacher on samples generated by the current student distribution. In a generic form, the objective can be written as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { O P D } } ( \theta ) = \mathbb { E } _ { z \sim q _ { \theta } ( \cdot \vert c ) } \left[ D _ { \mathrm { K L } } \big ( f _ { \theta } ( z , c ) , f _ { \phi } ( z , c ) \big ) \right] , } \end{array}
$$

where c denotes the conditioning input, $q _ { \theta } ( \cdot \mid c )$ is the distribution induced by the current student.

On-policy self-distillation. On-policy self-distillation further removes the need for a separate teacher model by deriving teacher and student signals from asymmetric views of the same base model. Let $c _ { s }$ denote the student condition and $c _ { t }$ denote a stronger teacher condition, where $c _ { t }$ is the privileged information available only during training. The corresponding objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { O P S D } } ( \theta ) = \mathbb { E } _ { z \sim q _ { \theta } ( \cdot \vert c _ { s } ) } \left[ D _ { \mathrm { K L } } \big ( f _ { \theta } ( z , c _ { s } ) , f _ { \bar { \theta } } ( z , c _ { t } ) \big ) \right] , } \end{array}
$$

where $f _ { \bar { \theta } }$ denotes the teacher model, which may be a stop-gradient version of the base model or an exponential-moving-average (EMA) (Morales-Brotons et al., 2024) copy of the student model.

## 3.2 FROM PROMPT ENHANCEMENT TO PRIVILEGED INFORMATION

Prompt enhancement elicits conditional capability. Let $p$ denote a raw user prompt and let E be a prompt enhancer that produces an enhanced prompt $\bar { p } ^ { + } = E ( p )$ . Figure 3 and Table 1 compare these conditions using the same frozen Z-Image-Turbo generator (Cai et al., 2025). Across the tested PEs, enhanced prompts improve the reported compositional scores while substantially!"#\$%&'"()\*+, increasing prompt length. By adding descriptive details to p while preserving its explicit constraints, PE provides a richer condition for generation.

![](images/cfbb7a554acd8267f8a5de91489623abcd8473035906522ef73f6baa3d5b3d90.jpg)

!"#\$% &'()\* /\$-0'1% &'()\* /\$-0'1%2 /\$-0'1%2Table 1: Comparison of the Base and inference-time .\$-) .\$-)PE on GenEval and GenEval2. <sup>†</sup>denotes GPT-5.6 Sol as PE. <sup>‡</sup>denotes PromptEnhancer as PE.
<table><tr><td rowspan="2">Method</td><td colspan="2">GenEval Task</td><td colspan="3">GenEval2 Task</td></tr><tr><td>Token Len.</td><td>GE</td><td>Token Len.</td><td> $\mathbf { G E 2 } _ { \mathbf { A M } }$ </td><td> $\mathbf { G E 2 } _ { \mathtt { G M } }$ </td></tr><tr><td>Base</td><td>11.02</td><td>0.737</td><td>7.92</td><td>0.783</td><td>0.341</td></tr><tr><td>PE-GPT†</td><td>152.61</td><td>0.850</td><td>135.11</td><td>0.843</td><td>0.479</td></tr><tr><td>∆ (vs Base)</td><td>13.9×</td><td>+0.113</td><td>17.1×</td><td>+0.060</td><td>+0.138</td></tr><tr><td>PE-7B</td><td>198.29</td><td>0.746</td><td>170.66</td><td>0.802</td><td>0.383</td></tr><tr><td>∆ (vs Base)</td><td>18.0×</td><td>+0.009</td><td>21.5×</td><td>+0.019</td><td>+0.042</td></tr><tr><td>PE-32B‡</td><td>155.82</td><td>0.859</td><td>148.66</td><td>0.849</td><td>0.492</td></tr><tr><td>∆ (vs Base)</td><td>14.1×</td><td>+0.122</td><td>18.8×</td><td>+0.066</td><td>+0.151</td></tr></table>

Figure 3: Comparison of raw and enhanced prompt, with the enhanced prompt providing denser and more detailed textual information.

We view $p ^ { + }$ as a PE-selected elaboration of $p ,$ intended to preserve the explicit semantics of $p$ while adding one plausible realization of otherwise unspecified attributes, composition, and scene details. Because prompt enhancement changes only the conditioning input while keeping the generator fixed (Hao et al., 2023; Wang et al., 2025), the performance gap between $p$ and $p ^ { + }$ shows that the pretrained generator can better realize the requested content when conditioned on a more detailed description. This does not imply that the generator can recover the missing details from $p$ alone. Instead, it shows that the improved generation behavior is attainable under enhanced prompts. We therefore seek to internalize this enhanced-prompt behavior into a model conditioned only on $p .$

Enhanced prompts as native textual privileged information. We cast this objective through the lens of learning with privileged information. During training, $p ^ { + }$ provides the teacher with a richer PE-generated condition associated with the same raw prompt, while the student remains conditioned only on $p ;$ at deployment, $p ^ { + }$ is unavailable. Importantly, the teacher’s advantage arises from asymmetric conditioning rather than greater model capacity. Prompt enhancement therefore constitutes a native form of textual privileged information: both $p$ and $p ^ { + }$ are processed by the same architecture through its existing text-conditioning pathway, without introducing an auxiliary modality.

This differs from image-privileged formulations such as D-OPSD (Jiang et al., 2026a), which construct the teacher condition by jointly encoding the prompt and a paired target image with a multimodal encoder. PE-OPSD instead derives the privileged condition from prompt enhancement alone, requiring neither paired target images nor modifications to the model’s conditioning interface. Crucially, our goal is not to reconstruct $p ^ { + }$ or imitate the PE itself, but to distill the generation dynamics elicited by $p ^ { + }$ into a student that observes only $p .$ We next formalize this principle for flow-matching generators.

## 3.3 PE-OPSD: PROMPT-ENHANCED ON-POLICY SELF-DISTILLATION

Overview. Figure 4 illustrates the training and inference pipelines of PE-OPSD. Let $\begin{array} { r l } { \mathcal { D } } & { { } = } \end{array}$ $\{ ( p , p ^ { + } ) \}$ denote the training prompt pairs set, where $p ^ { + } ~ = ~ E ( p )$ is obtained through PE. The student and teacher share the same architecture and are initialized from the same pretrained parameters, $\theta = \bar { \theta } = \theta _ { \mathrm { p r e } }$ . During training, the student generates trajectories conditioned only on $p ,$ while the teacher provides enhanced-prompt supervision on the states visited by the student. After training, the behavior induced by enhanced prompts is internalized into the generator, enabling direct raw-prompt generation without invoking PE at inference time.

![](images/51927419cdf5422d30cf3adccca69d4a9b472ee4e3f4c743645c92ad71f6b447.jpg)  
Figure 4: Overview of PE-OPSD. At training, the student generates rollouts from raw prompts, while an EMA teacher provides enhanced-prompt supervision on the same rollout states. At inference, the trained student generates directly from raw prompts without inference-time PE.

On-policy sampling under raw prompts. We collect training states from the current student’s own generation process. Let $1 = t _ { K } > t _ { K - 1 } > \cdot \cdot \cdot > t _ { 0 } = 0$ denote the discrete denoising schedule, and let $\Delta t _ { k } = t _ { k } - t _ { k - 1 } > 0$ . Starting from $x _ { t _ { K } } \sim \mathcal { N } ( 0 , I )$ , the student follows the Euler updates

$$
x _ { t _ { k - 1 } } = x _ { t _ { k } } - \Delta t _ { k } v _ { \theta } ( x _ { t _ { k } } , t _ { k } , p ) , \qquad k = K , \ldots , 1 .\tag{1}
$$

We supervise the student at the visited states $\tau = \{ x _ { t _ { k } } \} _ { k = 1 } ^ { K }$ . Because these states are generated by the current student under the raw condition $p , \tau$ follows the state induced by the current student under the training sampler. In contrast, a trajectory generated by the teacher under $p ^ { + }$ would follow a different state distribution and thus provide off-policy supervision for the raw-prompt student.

Distillation from enhanced-prompt supervision. At each student-visited state $x _ { t _ { k } }$ , the student and teacher are evaluated at the same state and time but under different prompts:

$$
v _ { k } ^ { S } = v _ { \theta } ( x _ { t _ { k } } , t _ { k } , p ) , \qquad v _ { k } ^ { T } = v _ { \bar { \theta } } ( x _ { t _ { k } } , t _ { k } , p ^ { + } ) .\tag{2}
$$

Because PE-OPSD operates with deterministic flow trajectories, we directly regress the teacher’s vector field rather than introducing a stochastic transition kernel. We consider matching velocities $v ,$ one-step transitions $\mu ,$ or predicted clean latents $\scriptstyle { \hat { x } } _ { 0 }$ . Under the linear flow interpolation $x _ { t } =$ $( 1 - t ) x _ { 0 } +$ tϵ (Lipman et al., 2022), with $\epsilon \sim \mathcal { N } ( 0 , I )$ , the latter two targets are

$$
\mu _ { k } ^ { b } = x _ { t _ { k } } - \Delta t _ { k } v _ { k } ^ { b } , \qquad \hat { x } _ { 0 , k } ^ { b } = x _ { t _ { k } } - t _ { k } v _ { k } ^ { b } , \qquad b \in \{ S , T \} .\tag{3}
$$

Since both $\mu _ { k }$ and $\hat { x } _ { 0 , k }$ share the same state $x _ { t _ { k } } .$ , their squared-error objectives reduce to a weighted velocity mismatch. We therefore express all three variants using a unified objective:

$$
\mathcal { L } _ { \mathrm { P E - O P S D } } ( \theta ; \bar { \theta } ) = \mathbb { E } _ { ( p , p ^ { + } ) \sim \mathcal { D } } \left[ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \omega ( t _ { k } ) \left. v _ { k } ^ { S } - \mathrm { s g } \big [ v _ { k } ^ { T } \big ] \right. _ { 2 } ^ { 2 } \right] ,\tag{4}
$$

where $\omega _ { v } ( t _ { k } ) = 1 , \omega _ { x _ { 0 } } ( t _ { k } ) = t _ { k } ^ { 2 }$ , and $\omega _ { \mu } ( t _ { k } ) = ( t _ { k - 1 } - t _ { k } ) ^ { 2 } = ( \Delta t _ { k } ) ^ { 2 }$ , respectively. $\mathrm { s g } [ \cdot ]$ denotes stop-gradient. We refer to these variants as $v { - } \mathrm { l o s s } , x _ { 0 } { - } \mathrm { l o s s } .$ , and $\mu { - } \mathrm { l o s s }$ . The three variants share the same pointwise optimum in velocity space but differ in timestep weighting. Appendix $\mathrm { A }$ provides a trajectory-level KL interpretation and its connection to our deterministic matching objectives.

During optimization, the teacher predictions are detached, and gradients pass only through the student predictions in Equation 2. After updating the student from $\theta _ { n }$ to $\theta _ { n + 1 }$ , we update the teacher through EMA as

$$
\bar { \theta } _ { n + 1 }  \gamma \bar { \theta } _ { n } + ( 1 - \gamma ) \theta _ { n + 1 } , \quad \quad 0 \leq \gamma < 1 ,\tag{5}
$$

where $\gamma$ is the EMA decay rate. This provides a temporally smoothed teacher that evolves with the student while retaining enhanced-prompt conditioning.

Training recipe. Algorithm 1 summarizes the training procedure. We first construct prompt pairs set $\mathcal { D } = \overline { { \{ ( p , p ^ { + } ) \} } }$ by applying the off-the-shelf PE offline, avoiding PE calls during training. Each iteration rollouts the student using the raw prompt $p ,$ and evaluates the enhanced-prompt teacher at the visited states. The student is optimized to match the stop-gradient teacher velocities, after which the teacher is updated by EMA. After training, only the student is retained and generation proceeds directly from raw prompts without invoking the PE or teacher.

Algorithm 1 PE-OPSD Training   
Require: Prompt pairs set $\mathcal { D } = \{ ( p , p ^ { + } ) \}$ ; pretrained model $\theta _ { \mathrm { p r e } } ;$ schedule $\{ t _ { 0 } , \cdots , t _ { K } \}$ ; EMA   
decay rate γ   
1: $\theta  \theta _ { \mathrm { { p r e } } } , \bar { \theta }  \theta _ { \mathrm { { p r e } } }$   
2: for each training iteration do   
3: Sample a mini-batch $( p , p ^ { + } ) \sim \mathcal { D }$ and $x _ { t _ { K } } \sim \mathcal { N } ( 0 , I )$   
4: Initialize $\mathcal { L }  0$   
5: for each $k = K , \ldots , 1$ do   
6: $\begin{array} { r } { \mathcal { L } \gets \mathcal { L } + \frac { \omega ( t _ { k } ) } { K } \| v _ { \theta } ( x _ { t _ { k } } , t _ { k } , p ) - \mathrm { s g } [ v _ { \bar { \theta } } ( x _ { t _ { k } } , t _ { k } , p ^ { + } ) ] \| _ { 2 } ^ { 2 } } \end{array}$ ▷ µ−loss by default   
7: if k > 1 then   
8: $x _ { t _ { k - 1 } } \gets \mathrm { s g } [ x _ { t _ { k } } - \Delta t _ { k } v _ { \theta } ( x _ { t _ { k } } , t _ { k } , p ) ]$ ▷ Rollout for one step   
9: end if   
10: end for   
11: θ ← OptimizerStep(θ, ∇<sub>θ</sub>L)   
12: $\bar { \theta }  \gamma \bar { \theta } + ( 1 - \gamma ) \theta$ ▷ EMA update   
13: end for   
14: return θ

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUPS

Models and baselines. Our main experiments use SD3.5-M (Esser et al., 2024), Z-Image, and Z-Image-Turbo, comparing PE-OPSD against the unmodified model (Base), inference-time prompt enhancement (Base+PE), supervised fine-tuning (SFT), and off-policy distillation. For SFT, we first generate teacher images conditioned on enhanced prompts offline and then fine-tune the student on the images. Off-policy distillation uses the same teacher and distillation objective as PE-OPSD but collects training states from enhanced-prompt teacher trajectories rather than raw-prompt student trajectories. For fairness, all trainable methods use the same initialization, training data, optimization configuration, and number of training steps. We further evaluate FLUX.2-klein-base, FLUX.2-klein (Black Forest Labs, 2026), and QwenImage-2512 (Zhao et al., 2026a) to assess scalability. We use GPT-5.6 Sol as the default PE and evaluate PE generalization on Z-Image with PromptEnhancer-7B/32B (Wang et al., 2025).

Benchmarks and metrics. We evaluate compositional generation on GenEval (GE, Ghosh et al. (2023)) and GenEval2 (GE2, Kamath et al. (2025)), reporting GE and $\mathrm { G E } 2 _ { \mathrm { G M / A M } }$ , alongside CLIP score (Hessel et al., 2021), PickScore (Kirstain et al., 2023), and aesthetics (Schuhmann, 2022) on both benchmarks. We summarize improvement over Base as

$$
\mathbb { I } _ { c } = \frac { 1 0 0 \% } { \left| \mathcal { M } _ { c } \right| } \sum _ { m \in \mathcal { M } _ { c } } \left( \frac { \mathrm { S c o r e } _ { \mathrm { m e t h o d } } - \mathrm { S c o r e } _ { \mathrm { B a s e } } } { \mathrm { S c o r e } _ { \mathrm { B a s e } } } \right) , \quad c = \{ \mathrm { P F } , \ \mathrm { V A } \}\tag{6}
$$

where $\mathcal { M } _ { \mathrm { P F } }$ contains $\mathrm { G E } , \mathrm { G E } 2 _ { \mathrm { G M } } , \mathrm { G E } 2 _ { \mathrm { A M } } .$ , and CLIP scores on both benchmarks, while $\mathcal { M } _ { \mathrm { V A } }$ contains PickScore and aesthetics on both benchmarks. These indices measure Prompt Fidelity and Visual Appeal improvements, respectively. Out-of-domain evaluation uses DPG-bench (Hu et al., 2024), T2I-CompBench++ (Huang et al., 2025), and EvalMuse (Han et al., 2026).

Table 2: Main results across different models. We use GPT-5.6 Sol as PE. PickScore is normalized by 26; Aes. denotes aesthetics; Bold: best; Underlined: second-best.
<table><tr><td rowspan="2">Method</td><td colspan="4">GenEval (GE) Task</td><td colspan="5">GenEval2 (GE2) Task</td><td rowspan="2">IPF</td><td rowspan="2">IvA</td></tr><tr><td>GE</td><td>PickScore</td><td>CLIP</td><td>Aes.</td><td> $\mathbf { G E 2 } _ { \mathtt { G M } }$ </td><td> $\mathbf { G E 2 } _ { \mathbf { A M } }$ </td><td>PickScore</td><td>CLIP</td><td>Aes.</td></tr><tr><td>SD3.5-L</td><td>0.604</td><td>0.862</td><td>0.288</td><td>5.286</td><td>0.213</td><td>0.660</td><td>0.869</td><td>0.315</td><td>5.521</td><td></td><td></td></tr><tr><td>FLUX.1-dev</td><td>0.618</td><td>0.899</td><td>0.280</td><td>5.624</td><td>0.179</td><td>0.643</td><td>0.895</td><td>0.298</td><td>5.909</td><td></td><td></td></tr><tr><td colspan="10">SD3.5-M (2.5B)</td></tr><tr><td>Base</td><td>0.628</td><td>0.878</td><td>0.289</td><td>5.319</td><td>0.176</td><td>0.633</td><td>0.882</td><td>0.313</td><td>5.579</td><td>0.00%</td><td>0.00%</td></tr><tr><td>Base+PE</td><td>0.743</td><td>0.892</td><td>0.291</td><td>5.439</td><td>0.220</td><td>0.680</td><td>0.887</td><td>0.312</td><td>5.678</td><td>10.22%</td><td>1.55%</td></tr><tr><td>SFT</td><td>0.718</td><td>0.889</td><td>0.295</td><td>5.462</td><td>0.207</td><td>0.656</td><td>0.886</td><td>0.310</td><td>5.702</td><td>7.34%</td><td>1.65%</td></tr><tr><td>Off-Policy Distillation</td><td>0.770</td><td>0.892</td><td>0.296</td><td>5.407</td><td>0.223</td><td>0.671</td><td>0.890</td><td>0.316</td><td>5.567</td><td>11.74%</td><td>0.99%</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.797</td><td>0.896</td><td>0.295</td><td>5.401</td><td>0.226</td><td>0.682</td><td>0.893</td><td>0.318</td><td>5.708</td><td>13.35%</td><td>1.79%</td></tr><tr><td>∆ (vs Base)</td><td>+0.169</td><td>+0.018</td><td>+0.006</td><td>+0.082</td><td>+0.050</td><td>+0.049</td><td>+0.011</td><td>+0.005</td><td>+0.129</td><td>+13.35%</td><td>+1.79%</td></tr><tr><td colspan="10"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Z-Image (6B)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>0.650</td><td>0.875</td><td>0.288</td><td>5.288</td><td>0.306</td><td>0.761</td><td>0.871</td><td>0.319</td><td>5.421</td><td>0.00%</td><td>0.00%</td></tr><tr><td>Base+PE</td><td>0.810</td><td>0.899</td><td>0.299</td><td>5.433</td><td>0.404</td><td>0.827</td><td>0.893</td><td>0.326</td><td>5.638</td><td>14.27%</td><td>3.00%</td></tr><tr><td>SFT</td><td>0.821</td><td>0.893</td><td>0.302</td><td>5.426</td><td>0.403</td><td>0.834</td><td>0.887</td><td>0.326</td><td>5.466</td><td>14.93%</td><td>1.83%</td></tr><tr><td>Off-Policy Distillation</td><td>0.824 0.846</td><td>0.901 0.902</td><td>0.300 0.302</td><td>5.432 5.423</td><td>0.395 0.406</td><td>0.828</td><td>0.897</td><td>0.327</td><td>5.629</td><td>14.27%</td><td>3.12%</td></tr><tr><td>PE-OPSD (Ours) ∆ (vs Base)</td><td>+0.196</td><td>+0.027</td><td>+0.014</td><td>+0.135</td><td>+0.100</td><td>0.831 +0.070</td><td>0.900 +0.029</td><td>0.330 +0.011</td><td>5.650 +0.229</td><td>16.07% +16.07%</td><td>3.30% +3.30%</td></tr><tr><td colspan="10"></td></tr><tr><td></td><td></td><td></td><td></td><td>Z-Image-Turbo</td><td></td><td>(6B)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>0.737</td><td>0.908</td><td>0.291</td><td>5.289</td><td>0.341</td><td>0.783</td><td>0.901</td><td>0.320</td><td>5.436</td><td>0.00%</td><td>0.00%</td></tr><tr><td>Base+PE</td><td>0.850</td><td>0.918</td><td>0.298</td><td>5.306</td><td>0.479 0.413</td><td>0.843 0.788</td><td>0.911</td><td>0.325</td><td>5.561</td><td>13.49%</td><td>1.21%</td></tr><tr><td>SFT</td><td>0.766</td><td>0.893 0.916</td><td>0.295</td><td>5.218</td><td>0.437</td><td>0.821</td><td>0.884</td><td>0.317</td><td>5.398</td><td>5.22%</td><td>-1.40%</td></tr><tr><td>Off-Policy Distillation PE-OPSD (Ours)</td><td>0.851 0.863</td><td>0.919</td><td>0.296</td><td>5.251</td><td>0.501</td><td>0.844</td><td>0.910</td><td>0.325</td><td>5.561</td><td>10.35%</td><td>0.87%</td></tr><tr><td></td><td></td><td></td><td>0.298</td><td>5.274</td><td></td><td>+0.061</td><td>0.913</td><td>0.326</td><td>5.547</td><td>15.22%</td><td>1.08%</td></tr><tr><td>∆ (vs Base)</td><td>+0.126</td><td>+0.011</td><td>+0.007</td><td>-0.015</td><td>+0.160</td><td></td><td>+0.012</td><td>+0.006</td><td>+0.111</td><td>+15.22%</td><td>+1.08%</td></tr></table>

![](images/d948eca42e7b2b8bca86fb54a8d10f882f959b77930018712bffa273a2f84697.jpg)  
Three yellow pigs behind a checkered penguin on top of five suitcases

<table><tr><td colspan="5">Prompt Fidelity</td></tr><tr><td>PE</td><td>60%</td><td></td><td>40%</td><td>Base</td></tr><tr><td>SFT</td><td>53%</td><td></td><td>47%</td><td>Base</td></tr><tr><td>Off-Policy 58%</td><td></td><td></td><td>42%</td><td>Base</td></tr><tr><td>Ours</td><td>66%</td><td>0.5</td><td>34%</td><td>Base</td></tr><tr><td>Visual Appeal</td><td></td><td></td><td></td><td></td></tr><tr><td>PE</td><td>56%</td><td></td><td>44%</td><td>Base</td></tr><tr><td>SFT</td><td>40%</td><td></td><td>60%</td><td>Base</td></tr><tr><td>Off-Policy 54%</td><td></td><td></td><td>46%</td><td>Base</td></tr><tr><td>Ours</td><td>58%</td><td>0.5</td><td>42%</td><td>Base</td></tr></table>

Figure 5: Qualitative comparison and human preference study on Z-Image-Turbo. We use GPT-5.6 Sol as PE. Left: visual comparisons among Base, PE, SFT, Off-policy distillation, and our method. Right: pairwise human preferences against Base in prompt fidelity and visual appeal.

Configuration. For GenEval, we adopt the training and evaluation splits released by Flow-GRPO (Liu et al., 2026c). For GenEval2, we train on the official 20K synthetic prompts and evaluate on the 800 officially released prompts. All models are trained on the same mixture of the GenEval and GenEval2 training sets. To ensure a fair comparison, we use a unified training recipe across all models and methods. Full configuration details and experimental setups are provided in Appendix B.

## 4.2 MAIN EXPERIMENTS

Main results. Table 2 shows that PE-OPSD consistently achieves the best prompt fidelity across all three backbones. It obtains the highest GE, $\mathrm { G E 2 _ { G M } } ,$ and I<sub>PF</sub>, outperforming SFT, off-policy distillation, and even Base+PE, which retains the PE at inference. Relative to Base, PE-OPSD improves aggregate prompt fidelity by 13.35%, 16.07%, and 15.22% on SD3.5-M, Z-Image, and Z-Image-Turbo, respectively. Meanwhile, I remains positive on every backbone, indicating that PE-OPSD improves prompt fidelity without degrading aggregate visual appeal.

Applicability to post-trained models. PE-OPSD remains effective on the already post-trained Z-Image-Turbo (post-trained by Decoupled DMD (Liu et al., 2026a) and DMDR (Jiang et al., 2026b)). SFT yields smaller fidelity gains and a 1.40% decline in visual appeal, whereas PE-OPSD improves both. These results support its applicability to models that have already undergone post-training.

Efficiency. Table 3 compares prompt fidelity and latency on Z-Image. Under both evaluated PEs, PE-OPSD improves all three fidelity metrics over inference-time PE while retaining the measured latency of Base. It is 1.96× and 4.51× faster than deploying PromptEnhancer-7B and PromptEnhancer-32B, respectively. Thus, PE-OPSD transfers the benefits of prompt enhancement to the generator without requiring a PE for each inference request. More results about training costs are reported in Appendix D.2.

Table 3: Performance and latency.
<table><tr><td>Method</td><td>GE</td><td>GE2GM</td><td>GE2AM</td><td>Lat. (s)</td></tr><tr><td colspan="5">PromptEnhancer-7B as PE</td></tr><tr><td>Base</td><td>0.650</td><td>0.306</td><td>0.761</td><td>9.69</td></tr><tr><td>Base+PE</td><td>0.684</td><td>0.375</td><td>0.801</td><td>19.00</td></tr><tr><td>Base+Ours ∆ (vs +PE)</td><td>0.782 +0.098</td><td>0.422 +0.047</td><td>0.819 +0.018</td><td>9.69 1.96×</td></tr><tr><td colspan="5">PromptEnhancer-32B as PE</td></tr><tr><td>Base</td><td>0.650</td><td>0.306</td><td>0.761</td><td>9.69</td></tr><tr><td>Base+PE</td><td>0.815</td><td>0.470</td><td>0.844</td><td>43.74</td></tr><tr><td>Base+Ours ∆ (vs +PE)</td><td>0.868 +0.053</td><td>0.476 +0.006</td><td>0.846 +0.002</td><td>9.69 4.51×</td></tr></table>

Table 4: Result across different PEs. Bold: best; Underlined: second-best.
<table><tr><td rowspan="2">Method</td><td colspan="4">GenEval (GE) Task</td><td colspan="5">GenEval2 (GE2) Task</td><td rowspan="2">IPF</td><td rowspan="2">IvA</td></tr><tr><td>GE</td><td>PickScore</td><td>CLIP</td><td>Aes.</td><td> $\mathbf { G E 2 } _ { \mathtt { G M } }$ </td><td> $\mathbf { G E 2 } _ { \mathbf { A M } }$ </td><td>PickScore</td><td>CLIP</td><td>Aes.</td></tr><tr><td colspan="10">GPT-5.6 Sol as PE</td></tr><tr><td>Z-Image</td><td>0.650</td><td>0.875</td><td>0.288</td><td>5.288</td><td>0.306</td><td>0.761</td><td>0.871</td><td>0.319</td><td>5.421</td><td>0.00%</td><td>0.00%</td></tr><tr><td>Z-Image+PE</td><td>0.810</td><td>0.899</td><td>0.299</td><td>5.433</td><td>0.404</td><td>0.827</td><td>0.893</td><td>0.326</td><td>5.638</td><td>14.27%</td><td>3.00%</td></tr><tr><td>SFT</td><td>0.821</td><td>0.893</td><td>0.302</td><td>5.426</td><td>0.403</td><td>0.834</td><td>0.887</td><td>0.326</td><td>5.466</td><td>14.93%</td><td>1.83%</td></tr><tr><td>Off-Policy Distillation</td><td>0.824</td><td>0.901</td><td>0.300</td><td>5.432</td><td>0.395</td><td>0.828</td><td>0.897</td><td>0.327</td><td>5.629</td><td>14.27%</td><td>3.12%</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.846</td><td>0.902</td><td>0.302</td><td>5.423</td><td>0.406</td><td>0.831</td><td>0.900</td><td>0.330</td><td>5.650</td><td>16.07%</td><td>3.30%</td></tr><tr><td>∆ (vs Base)</td><td>+0.196</td><td>+0.027</td><td>+0.014</td><td>+0.135</td><td>+0.100</td><td>+0.070</td><td>+0.029</td><td>+0.011</td><td>+0.229</td><td>+16.07%</td><td>+3.30%</td></tr><tr><td colspan="10">PromptEnhancer-7B</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>as PE</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Z-Image Z-Image+PE</td><td>0.650 0.684</td><td>0.875 0.884</td><td>0.288 0.287</td><td>5.288 5.464</td><td>0.306 0.375</td><td>0.761 0.801</td><td>0.871 0.879</td><td>0.319 0.318</td><td>5.421 5.589</td><td>0.00% 6.28%</td><td>0.00% 2.09%</td></tr><tr><td>SFT</td><td>0.763</td><td>0.887</td><td>0.299</td><td>5.424</td><td>0.400</td><td>0.819</td><td>0.881</td><td>0.325</td><td>5.424</td><td>12.29%</td><td>1.29%</td></tr><tr><td>Off-Policy Distillation</td><td>0.767</td><td>0.896</td><td>0.296</td><td>5.410</td><td>0.380</td><td>0.804</td><td>0.889</td><td>0.324</td><td>5.573</td><td>10.44%</td><td>2.39%</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.782</td><td>0.898</td><td>0.296</td><td>5.417</td><td>0.422</td><td>0.819</td><td>0.892</td><td>0.327</td><td>5.606</td><td>14.22%</td><td>2.72%</td></tr><tr><td>∆ (vs Base)</td><td>+0.132</td><td>+0.023</td><td>+0.008</td><td>+0.129</td><td>+0.116</td><td>+0.058</td><td>+0.021</td><td>+0.008</td><td>+0.185</td><td>+14.22%</td><td>+2.72%</td></tr><tr><td colspan="10">PromptEnhancer-32B as PE</td></tr><tr><td>Z-Image</td><td>0.650</td><td>0.875</td><td>0.288</td><td></td><td>0.306</td><td>0.761</td><td>0.871</td><td>0.319</td><td>5.421</td><td>0.00%</td><td>0.00%</td></tr><tr><td>Z-Image+PE</td><td>0.815</td><td>0.889</td><td>0.294</td><td>5.288 5.381</td><td>0.470</td><td>0.844</td><td>0.879</td><td>0.325</td><td>5.515</td><td>18.77%</td><td>1.50%</td></tr><tr><td>SFT</td><td>0.829</td><td>0.889</td><td>0.302</td><td>5.381</td><td>0.440</td><td>0.845</td><td>0.877</td><td>0.329</td><td>5.373</td><td>18.07%</td><td>0.79%</td></tr><tr><td>Off-Policy Distillation</td><td>0.842</td><td>0.895</td><td>0.300</td><td>5.363</td><td>0.459</td><td>0.843</td><td>0.884</td><td>0.331</td><td>5.455</td><td>19.65%</td><td>1.46%</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.868</td><td>0.897</td><td>0.300</td><td>5.352</td><td>0.476</td><td>0.846</td><td>0.888</td><td>0.334</td><td>5.496</td><td>21.83%</td><td>1.76%</td></tr><tr><td>∆ (vs Base)</td><td>+0.218</td><td>+0.022</td><td>+0.012</td><td>+0.064</td><td>+0.170</td><td>+0.085</td><td>+0.017</td><td>+0.015</td><td>+0.075</td><td>+21.83%</td><td>+1.76%</td></tr></table>

Table 5: Results on out-of-domain benchmarks. For DPG-bench, we report Global (Glo.), Entity (Ent.), Attribute (Attr.), Relation (Rela.), Other and Overall<sup>⋆</sup>. For T2I-CompBench++, we report Color, Shape, Textual (Tex.), Numeracy (Num.), Complex (Comp.), Spatial (Spa.), 3D Spatial (3D Spa.), Non-spatial (Non-spa.) and Overall<sup>⋆</sup>. Bold: best in Overall<sup>⋆</sup>.
<table><tr><td rowspan="2">Method</td><td colspan="6">DPG-bench</td><td colspan="9">T2I-CompBench++</td><td>EvalMuse</td></tr><tr><td>Overall*</td><td>Glo.</td><td>Ent.</td><td>Attr.</td><td>Rela.</td><td>Other</td><td>Overall*</td><td>Color</td><td>Shape</td><td>Tex.</td><td>Num.</td><td>Comp.</td><td>Spa.</td><td>3D Spa.</td><td>Non-spa.</td><td>Overall*</td></tr><tr><td>SD3.5-M</td><td>83.9</td><td>84.6</td><td>89.6</td><td>88.1</td><td>93.0</td><td>80.9</td><td>0.51</td><td>0.80</td><td>0.54</td><td>0.74</td><td>0.59</td><td>0.37</td><td>0.32</td><td>0.36</td><td>0.31</td><td>3.199</td></tr><tr><td>+Ours</td><td>83.8</td><td>83.9</td><td>89.3</td><td>88.4</td><td>93.0</td><td>82.1</td><td>0.55</td><td>0.83</td><td>0.60</td><td>0.73</td><td>0.63</td><td>0.39</td><td>0.46</td><td>0.42</td><td>0.31</td><td>3.368</td></tr><tr><td>Z-Image</td><td>86.3</td><td>83.5</td><td>91.7</td><td>90.1</td><td>94.6</td><td>87.8</td><td>0.53</td><td>0.85</td><td>0.59</td><td>0.79</td><td>0.63</td><td>0.40</td><td>0.31</td><td>0.37</td><td>0.31</td><td>3.340</td></tr><tr><td>+Ours</td><td>87.0</td><td>82.6</td><td>92.0</td><td>90.2</td><td>94.5</td><td>89.4</td><td>0.58</td><td>0.87</td><td>0.61</td><td>0.81</td><td>0.71</td><td>0.42</td><td>0.36</td><td>0.41</td><td>0.32</td><td>3.561</td></tr><tr><td>Z-Image-Turbo</td><td>84.5</td><td>78.6</td><td>91.2</td><td>88.3</td><td>93.2</td><td>88.4</td><td>0.53</td><td>0.81</td><td>0.56</td><td>0.75</td><td>0.69</td><td>0.40</td><td>0.36</td><td>0.41</td><td>0.31</td><td>3.515</td></tr><tr><td>+Ours</td><td>85.0</td><td>77.8</td><td>91.2</td><td>88.9</td><td>93.9</td><td>88.1</td><td>0.56</td><td>0.89</td><td>0.53</td><td>0.76</td><td>0.72</td><td>0.40</td><td>0.52</td><td>0.42</td><td>0.31</td><td>3.534</td></tr></table>

Qualitative results and human preference study. Figure 5 presents visual examples and human preference study results. In pairwise comparisons against Base, human raters prefer PE-OPSD in 66% of comparisons for prompt fidelity and 58% for visual appeal. These are the highest observed preference rates among the evaluated methods, complementing the quantitative results. Human study details are provided in Appendix C.

## 4.3 GENERALIZATION

Different PEs. On Z-Image, PE-OPSD achieves the highest $\mathbb { I } _ { \mathrm { P F } }$ and $\mathbb { I } _ { \mathrm { V A } }$ with each of the three tested PEs (Table 4), demonstrating that its benefits extend beyond the default PE. The largest aggregate fidelity gain is obtained with PromptEnhancer-32B, while GPT-5.6 Sol yields the largest visual appeal gain, suggesting that PE choice affects the balance between these objectives.

Out-of-domain generalization. Table 5 shows that all three backbones improve their Overall scores on T2I-CompBench++ and EvalMuse, with consistent gains in spatial relations on T2I-CompBench++. On DPG-Bench, which features long and detailed prompts, PE-OPSD largely preserves the base models’ performance. Overall, PE-OPSD generalizes across OOD benchmarks, improving compositional fidelity while preserving performance on already detailed prompts.

![](images/4c6b818508bce73820c77d0f1f75159ef8d4923f1f970e93068e1fe5de637108.jpg)

![](images/6d213789178d21030bda51231b089fd7a0e2d2a38b6b28a8bf9392f78230f313.jpg)  
Figure 6: Scaling to larger models. PickScore, CLIP, and Aesthetics are averaged over GenEval and GenEval2. For visualization, each metric is independently normalized to [0.3, 1.0].

![](images/7668c602f640a022825a54d15cc4f82fd35f8a9ee86f47ebfb5caf292f173cfa.jpg)  
Figure 7: Ablation study results on different training datasets. We use MixDataset by default.  
Table 8: Ablation study results on the steps for training. Time (s): training times for one step.

Table 6: Ablation study results on different loss functions, which differ only in their timestep weighting.  
Table 7: Ablation study results on CFG settings. <sup>†</sup>denotes the unconditional branch is detached. Time (s): training times for one step.
<table><tr><td>Loss</td><td>GE</td><td> $\mathbf { G E 2 } _ { \mathtt { G M } }$ </td><td>GE2AM</td></tr><tr><td>x0</td><td>0.863</td><td>0.474</td><td>0.834</td></tr><tr><td>v</td><td>0.866</td><td>0.487</td><td>0.838</td></tr><tr><td>µ</td><td>0.863</td><td>0.501</td><td>0.844</td></tr></table>

<table><tr><td>Settings</td><td>GE</td><td> $\mathbf { G E 2 } _ { \mathtt { G M } }$ </td><td> $\mathbf { G E 2 } _ { \mathbf { A M } }$ </td><td>Time</td></tr><tr><td>CFG=4.0</td><td>0.834</td><td>0.430</td><td>0.835</td><td>99.13</td></tr><tr><td>CFG=4.0†</td><td>0.835</td><td>0.421</td><td>0.834</td><td>73.39</td></tr><tr><td>w/o CFG</td><td>0.846</td><td>0.406</td><td>0.831</td><td>49.15</td></tr></table>

<table><tr><td>Step</td><td>GE</td><td> $\mathbf { G E 2 } _ { \mathsf { G M } }$ </td><td> $\mathbf { G E 2 } _ { \mathbf { A M } }$ </td><td>Time</td></tr><tr><td>2</td><td>0.862</td><td>0.469</td><td>0.835</td><td>7.47</td></tr><tr><td>4</td><td>0.863</td><td>0.501</td><td>0.844</td><td>14.31</td></tr><tr><td>8</td><td>0.863</td><td>0.483</td><td>0.837</td><td>28.16</td></tr></table>

Scalability. We further evaluate PE-OPSD on the larger 9B FLUX.2-klein variants (Black Forest Labs, 2026) and the 20B QwenImage-2512 (Zhao et al., 2026a), as summarized in Figure 6 (complete results are in Appendix D.4). Across three models, PE-OPSD improves upon the base model and inference-time PE on aggregate prompt fidelity. It also remains competitive across aggregate visual appeal over Base. These results demonstrate that PE-OPSD scales effectively to larger models.

## 4.4 ABLATION STUDIES

Loss. Among losses differing only in timestep weighting, µ-loss achieves the highest GE2 scores while remaining close to v-loss on GE (Table 6), motivating its use as the default.

CFG. Omitting CFG roughly halves training time and improves GE, but lowers GE2 scores (Table 7). Detaching the unconditional branch partially reduces the computational overhead. We omit CFG by default for training efficiency.

On-policy training step. We conduct this ablation study in Z-Image-Turbo. As shown in Table 8, 4 rollout steps achieve the highest GE2 scores at approximately half the iteration time of 8 steps, with nearly identical GE across the tested settings. We therefore use four steps by default.

Training data. As shown in Figure 7, training on a single dataset leads to dataset-specific specialization. GenEval-only training performs well on GenEval but transfers poorly to GenEval2, whereas GenEval2-only training compromises performance on GenEval. In contrast, MixDataset achieves the best balance across both benchmarks and is therefore used as our default training set.

## 5 CONCLUSION

We introduced PE-OPSD, an on-policy self-distillation framework that treats enhanced prompts as privileged training information. PE-OPSD transfers the generation behavior induced by enhanced prompts to a raw-prompt generator through supervision on student-generated trajectories. Across three main backbones, it improves aggregate prompt fidelity by 13.35%-16.07% over Base and exceeds inference-time PE, SFT, and off-policy distillation while retaining base-model latency. Results with alternative PEs, larger models, and out-of-domain benchmarks further support its applicability.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from selfgenerated mistakes. In International Conference on Learning Representations, volume 2024, pp. 21246–21263, 2024.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Lucas Beyer, Xiaohua Zhai, Amelie Royer, Larisa Markeeva, Rohan Anil, and Alexander´ Kolesnikov. Knowledge distillation: A good teacher is patient and consistent. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10915–10924. IEEE, 2022.

Jinhe Bi, Peng Liao, Zengjie Jin, Volker Tresp, Fei Shen, Yunpu Ma, Tat-Seng Chua, et al. Opd-v: Visual on-policy self-distillation with modality balance. arXiv preprint arXiv:2608.05131, 2026.

Black Forest Labs. FLUX.2 [klein]: Towards Interactive Visual Intelligence. https://bfl.ai/ blog/flux2-klein-towards-interactive-visual-intelligence, 2026.

Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Aiming Hao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Yuming Jiang, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.

Tingfeng Cao, Chengyu Wang, Bingyan Liu, Ziheng Wu, Jinhui Zhu, and Jun Huang. Beautifulprompt: Towards automatic prompt engineering for text-to-image synthesis. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 1–11, 2023.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam¨ Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36: 52132–52152, 2023.

Robert M Gray. Entropy and information theory. Springer Science & Business Media, 2011.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In International Conference on Learning Representations, volume 2024, pp. 32694–32717, 2024.

Meera Hahn, Wenjun Zeng, Nithish Kannen, Rich Galt, Kartikeya Badola, Been Kim, and Zi Wang. Proactive agents for multi-turn text-to-image generation under uncertainty. arXiv preprint arXiv:2412.06771, 2024.

Shuhao Han, Haotian Fan, Jiachen Fu, Liang Li, Tao Li, Junhui Cui, Yunqiu Wang, Yang Tai, Jingwei Sun, Chun-Le Guo, et al. Evalmuse-40k: A fine-grained benchmark with comprehensive human annotations for text-to-image generation model alignment evaluation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 4583–4591, 2026.

Yaru Hao, Zewen Chi, Li Dong, and Furu Wei. Optimizing prompts for text-to-image generation. Advances in Neural Information Processing Systems, 36:66923–66939, 2023.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 7514–7528, 2021.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Kaiyi Huang, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench++: An enhanced and comprehensive benchmark for compositional text-to-image generation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(5):3563–3579, 2025.

Zijian Huang, Jay Zhangjie Wu, Zian Wang, Tianshi Cao, Jiasi Chen, Sanja Fidler, Huan Ling, and Xuanchi Ren. Ape: Agentic prompt enhancer for image generation and editing. arXiv preprint arXiv:2606.00204, 2026.

Dengyang Jiang, Xin Jin, Dongyang Liu, Zanyi Wang, Mingzhe Zheng, Ruoyi Du, Xiangpeng Yang, Qilong Wu, Zhen Li, Peng Gao, et al. D-opsd: On-policy self-distillation for continuously tuning step-distilled diffusion models. arXiv preprint arXiv:2605.05204, 2026a.

Dengyang Jiang, Dongyang Liu, Zanyi Wang, Qilong Wu, Liuzhuozheng Li, Hengzhuang Li, Xin Jin, Changsheng Lu, Zhen Li, Mengmeng Wang, et al. Distribution matching distillation meets reinforcement learning. In European Conference on Computer Vision, pp. 281–299. Springer, 2026b.

Amita Kamath, Kai-Wei Chang, Ranjay Krishna, Luke Zettlemoyer, Yushi Hu, and Marjan Ghazvininejad. Geneval 2: Addressing benchmark drift in text-to-image evaluation. arXiv preprint arXiv:2512.16853, 2025.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

Quanhao Li, Junqiu Yu, Kaixun Jiang, Yujie Wei, Zhen Xing, Pandeng Li, Ruihang Chu, Shiwei Zhang, Yu Liu, and Zuxuan Wu. Diffusionopd: A unified perspective of on-policy distillation in diffusion models. arXiv preprint arXiv:2605.15055, 2026.

Mingfeng Lin, Chengfei Cai, Lin Xu, Yuxiang Wei, and Liang Han. Dreopd: Degraded-reference extrapolative on-policy distillation for flow-matching models. arXiv preprint arXiv:2608.09233, 2026.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Dongyang Liu, Gao Peng, David Liu, DU DU, Zhen Li, Qilong Wu, Xin Jin, Sihan Cao, Shifeng Zhang, Steven HOI, et al. Decoupled dmd: Cfg augmentation as the spear, distribution matching as the shield. In International Conference on Learning Representations, volume 2026, pp. 140643–140666, 2026a.

Hongyu Liu, Chun Wang, Feng Gao, Xuanhua He, Yue Ma, Ziyu Wan, Yong Zhang, Xiaoming Wei, and Qifeng Chen. Opsd-v: On-policy self-distillation for post-training few-step autoregressive video generators. arXiv preprint arXiv:2607.08766, 2026b.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. Advances in neural information processing systems, 38:40783–40818, 2026c.

Oscar Manas, Pietro Astolfi, Melissa Hall, Candace Ross, Jack Urbanek, Adina Williams, Aish-˜ warya Agrawal, Adriana Romero-Soriano, and Michal Drozdzal. Improving text-to-image consistency via automatic prompt optimization. arXiv preprint arXiv:2403.17804, 2024.

Daniel Morales-Brotons, Thijs Vogels, and Hadrien Hendrikx. Exponential moving average of weights in deep learning: Dynamics and benefits. arXiv preprint arXiv:2411.18704, 2024.

Bowen Ping, Xiangxin Zhou, Penghui Qi, Minnan Luo, Liefeng Bo, and Tianyu Pang. Flowdppo: Divergence proximal policy optimization for flow matching models. arXiv preprint arXiv:2606.11025, 2026.

Christoph Schuhmann. Laion-aesthetics, 2022. URL https://laion.ai/blog/ laion-aesthetics/.

Tencent Hunyuan Foundation Model Team. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.

Linqing Wang, Ximing Xing, Yiji Cheng, Zhiyuan Zhao, Donghao Li, Tiankai Hang, Jiale Tao, Qixun Wang, Ruihuang Li, Comi Chen, et al. Promptenhancer: A simple approach to enhance text-to-image models via chain-of-thought prompt rewriting. arXiv preprint arXiv:2509.04545, 2025.

Yunlong Wang, Shuyuan Shen, and Brian Y Lim. Reprompt: Automatic prompt editing to refine aigenerative art towards precise expressions. In Proceedings ofthe 2023 CHI conference on human factors in computing systems, pp. 1–29, 2023.

Yutong Xie, Zhaoying Pan, Jinge Ma, Luo Jie, and Qiaozhu Mei. A prompt log analysis of textto-image generation systems. In Proceedings of the ACM Web Conference 2023, pp. 3892–3902, 2023.

Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and¨ William T Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024.

Qianhao Yuan, Jie Lou, Xing Yu, Hongyu Lin, Le Sun, Xianpei Han, and Yaojie Lu. Vision-opd: Learning to see fine details for multimodal llms via on-policy self-distillation. arXiv preprint arXiv:2605.18740, 2026.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3813–3824. IEEE, 2023.

Bing Zhao, Chenfei Wu, Deqing Li, Hao Meng, Jiahao Li, Jie Zhang, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kuan Cao, et al. Qwen-image-2.0 technical report. arXiv preprint arXiv:2605.10730, 2026a.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026b.

## A FROM TRAJECTORY KL TO DETERMINISTIC MATCHING

The main text directly presents the deterministic objective used in training. Here, we provide a probabilistic interpretation through trajectory-level distillation and show how it motivates a deterministic matching surrogate on student-visited states.

Trajectory-level on-policy distillation. For a prompt pair $( p , p ^ { + } )$ , let the student and teacher trajectory distributions be

$$
\Pi _ { \theta } ^ { S } ( \tau \mid p ) = p ( x _ { t _ { K } } ) \prod _ { k = 1 } ^ { K } \pi _ { \theta , k } ^ { S } ( x _ { t _ { k - 1 } } \mid x _ { t _ { k } } , p ) ,\tag{7}
$$

$$
\Pi _ { \bar { \theta } } ^ { T } ( \tau \mid p ^ { + } ) = p ( x _ { t _ { K } } ) \prod _ { k = 1 } ^ { K } \pi _ { \bar { \theta } , k } ^ { T } ( x _ { t _ { k - 1 } } \mid x _ { t _ { k } } , p ^ { + } ) ,\tag{8}
$$

where both distributions share the same initial noise prior. A trajectory-level distillation objective is

$$
\mathcal { L } _ { \mathrm { t r a j } } = \mathbb { E } _ { ( p , p ^ { + } ) \sim \mathcal { D } } \left[ D _ { \mathrm { K L } } \big ( \Pi _ { \theta } ^ { S } ( \cdot \mid p ) \big | \big | \Pi _ { \theta } ^ { T } ( \cdot \mid p ^ { + } ) \big ) \right] .\tag{9}
$$

By the chain rule for KL divergence, this objective decomposes as

$$
\mathcal { L } _ { \mathrm { t r a j } } = \mathbb { E } _ { ( p , p ^ { + } ) \sim \mathcal { D } } [ \sum _ { k = 1 } ^ { K } \mathbb { E } _ { x _ { t _ { k } } \sim d _ { \theta , k } ( \cdot \vert p ) } [ D _ { \mathrm { K L } } ( \pi _ { \theta , k } ^ { S }  \pi _ { \bar { \theta } , k } ^ { T } ) ] ] ,\tag{10}
$$

where $d _ { \theta , k } ( \cdot \mid p )$ is the state distribution induced by the current raw-prompt student. Thus, the teacher is queried at states visited by the student rather than states from its own enhanced-prompt trajectory.

Conditional Gaussian transitions. At a fixed student-visited state $( x _ { t _ { k } } , t _ { k } )$ , consider Gaussian transitions with shared covariance (Liu et al., 2026c; Li et al., 2026; Lin et al., 2026):

$$
\pi _ { k } ^ { b } = \mathcal { N } ( \mu _ { k } ^ { b } , \Sigma _ { k } ) , \qquad \mu _ { k } ^ { b } = x _ { t _ { k } } - \Delta t _ { k } v _ { k } ^ { b } , \qquad b \in \{ S , T \} ,\tag{11}
$$

where $\Sigma _ { k } \succ 0$ is shared by the student and teacher. Their conditional KL divergence is

$$
D _ { \mathrm { K L } } \big ( \pi _ { k } ^ { S } \big \| \pi _ { k } ^ { T } \big ) = \frac { 1 } { 2 } \left\| \mu _ { k } ^ { S } - \mu _ { k } ^ { T } \right\| _ { \Sigma _ { k } ^ { - 1 } } ^ { 2 } = \frac { ( \Delta t _ { k } ) ^ { 2 } } { 2 } \left\| v _ { k } ^ { S } - v _ { k } ^ { T } \right\| _ { \Sigma _ { k } ^ { - 1 } } ^ { 2 } .\tag{12}
$$

For any positive-definite shared covariance, the unique pointwise minimizer is $v _ { k } ^ { S } = v _ { k } ^ { T }$ , equivalently $\mu _ { k } ^ { S } = \mu _ { k } ^ { T }$ .The covariance affects the weighting of the regression objective but not its pointwise optimum.

From stochastic transitions to deterministic matching. For two deterministic transitions with distinct endpoints, the corresponding Dirac measures are mutually singular, and their KL divergence is therefore infinite (Gray, 2011). We thus do not obtain the deterministic objective by directly taking a zero-variance KL limit. Instead, we retain the pointwise optimizer of the shared-covariance Gaussian objective and realize it through deterministic L2-matching.

$$
\mathcal { L } _ { \mu } = \mathbb { E } \left[ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left. \mu _ { k } ^ { S } - \mathrm { s g } [ \mu _ { k } ^ { T } ] \right. _ { 2 } ^ { 2 } \right] = \mathbb { E } \left[ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } ( \Delta t _ { k } ) ^ { 2 } \left. v _ { k } ^ { S } - \mathrm { s g } [ v _ { k } ^ { T } ] \right. _ { 2 } ^ { 2 } \right] .\tag{13}
$$

The rollout states and teacher predictions are detached during each optimization step. This deterministic objective preserves the pointwise teacher-matching target of the Gaussian formulation while avoiding the introduction of transition noise during training.

Alternative deterministic parameterizations. Besides matching Euler transition targets, we consider direct velocity matching and clean-latent matching. Under the linear flow interpolation

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon , \qquad v = \epsilon - x _ { 0 } ,\tag{14}
$$

the predicted clean latent is

$$
\hat { x } _ { 0 , k } ^ { b } = x _ { t _ { k } } - t _ { k } v _ { k } ^ { b } , \qquad b \in \{ S , T \} .\tag{15}
$$

Consequently,

$$
\begin{array} { r l } & { \quad \left\| \mu _ { k } ^ { S } - \mathrm { s g } [ \mu _ { k } ^ { T } ] \right\| _ { 2 } ^ { 2 } = ( \Delta t _ { k } ) ^ { 2 } \left\| v _ { k } ^ { S } - \mathrm { s g } [ v _ { k } ^ { T } ] \right\| _ { 2 } ^ { 2 } , } \\ & { \left\| \hat { x } _ { 0 , k } ^ { S } - \mathrm { s g } [ \hat { x } _ { 0 , k } ^ { T } ] \right\| _ { 2 } ^ { 2 } = t _ { k } ^ { 2 } \left\| v _ { k } ^ { S } - \mathrm { s g } [ v _ { k } ^ { T } ] \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{16}
$$

The three deterministic objectives can therefore be written as

$$
\mathcal { L } = \mathbb { E } \left[ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \omega ( t _ { k } ) \left. v _ { k } ^ { S } - \mathrm { s g } [ v _ { k } ^ { T } ] \right. _ { 2 } ^ { 2 } \right] ,\tag{17}
$$

with

$$
\omega _ { v } ( t _ { k } ) = 1 , \qquad \omega _ { \mu } ( t _ { k } ) = ( \Delta t _ { k } ) ^ { 2 } , \qquad \omega _ { x _ { 0 } } ( t _ { k } ) = t _ { k } ^ { 2 } .\tag{18}
$$

All three objectives share the pointwise optimum $v _ { k } ^ { S } \ = v _ { k } ^ { T }$ , but they assign different weights to denoising timesteps and therefore need not produce identical optimization dynamics.

## B EXPERIMENTAL DETAILS

## B.1 TRAINING DATA

For GenEval (Ghosh et al., 2023), we adopt the data splits released by Flow-GRPO (Liu et al., 2026c), consisting of 50K training prompts and 2,212 evaluation instances. For GenEval2 (Kamath et al., 2025), we use the officially released 20K synthetic prompts for training and the official set of 800 prompts for evaluation. Unless otherwise specified, all trainable methods use MixDataset, formed by combining the GenEval and GenEval2 training sets. Evaluation prompts are never used for training.

We construct the prompt-pair dataset $\mathcal { D } = \{ ( p , p ^ { + } ) \}$ offline by applying each PE to the raw training prompts using the system prompt in Appendix F.2. Since $\mathtt { G P T - 5 . 6 } \mathtt { S o l }$ is a general-purpose model rather than a PE specifically trained for prompt rewriting, its outputs may occasionally alter or omit subject identities or attributes. We therefore apply an automatic consistency check using the system prompt in Appendix F.1. Here, we use GPT-5.6 Sol as judge model. Rejected prompts are regenerated until they pass this check. This preprocessing is performed once before training, so the PE need not be loaded during optimization.

## B.2 TRAINING CONFIGURATIONS.

All experiments are conducted on one to four nodes with 8 NVIDIA A100 GPUs. We use a unified training recipe across model families and trainable baselines. Within each backbone, all methods use the same initialization, training data, and hyper-parameters.

To improve training efficiency, we collect student rollouts using fewer denoising steps than at inference, following the denoising-reduction practice established in prior work (Liu et al., 2026c; Ping et al., 2026). Specifically, we use 4 training rollout steps for Z-Image-Turbo and FLUX.2-klein, 10 for SD3.5-M, and 14 for Z-Image, FLUX.2-klein-base, and QwenImage-2512. Here, on-policy refers to the provenance of the training states: they are generated by the current student under raw prompts. At every visited training state, the student and teacher are evaluated at the same latent and timestep. This does not require the training rollout and inference process to use identical timestep discretizations.

For evaluation, we set the number of inference steps to 4 for FLUX.2-klein, 8 for Z-Image-Turbo, 40 for SD3.5M, and 50 for Z-Image, FLUX.2-klein-base and QwenImage-2512 following the official settings. For evaluation CFG settings, we set 1.0 (disabled) for Z-Image-Turbo and FLUX.2-klein, 4.0 for Z-Image, FLUX.2-klein-base and QwenImage-2512, and 4.5 for SD3.5M.The other hyperparameter settings are reported in Table 9.

The ablation studies for different loss, dataset, and training step are conducted on Z-Image-Turbo using GPT-5.6 Sol as PE. The CFG ablation is conducted on Z-Image with the same PE. Unless explicitly varied, all remaining settings follow the default configuration.

Table 9: Default training hyperparameters for PE-OPSD. All models follow the same recipe unless otherwise specified.
<table><tr><td>Configuration</td><td>Setting</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$  , constant without warm-up</td></tr><tr><td>Optimizer momentum</td><td> $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ </td></tr><tr><td>Adam € / weight decay</td><td> $1 0 ^ { - 8 } / 0$ </td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Global batch size</td><td>64</td></tr><tr><td>Optimization steps</td><td>1,000</td></tr><tr><td>LoRA rank / scaling factor</td><td>64 /128</td></tr><tr><td>EMA decay γ</td><td>0.999</td></tr><tr><td>Training resolution</td><td>All 512 × 512</td></tr><tr><td>Training CFG</td><td>Disabled</td></tr><tr><td>Evaluation resolution</td><td>All 512 × 512</td></tr><tr><td>Maximum text sequence length</td><td>512</td></tr><tr><td>Distributed optimization</td><td>DeepSpeed ZeRO-2</td></tr></table>

Table 10: Models and checkpoints used in our experiments.
<table><tr><td>Model</td><td>Checkpoint</td></tr><tr><td>SD3.5-Medium</td><td>stabilityai/stable-diffusion-3.5-medium</td></tr><tr><td>SD3.5-Large</td><td>stabilityai/stable-diffusion-3.5-large</td></tr><tr><td>Z-Image</td><td>Tongyi-MAI/Z-Image</td></tr><tr><td>Z-Image-Turbo</td><td>Tongyi-MAI/Z-Image-Turbo</td></tr><tr><td>FLUX.1-dev</td><td>black-forest-labs/FLUX.1-dev</td></tr><tr><td>FLUX.2-klein-base-9B</td><td>black-forest-labs/FLUX.2-klein-base-9B</td></tr><tr><td>FLUX.2-klein-9B</td><td>black-forest-labs/FLUX.2-klein-9B</td></tr><tr><td>QwenImage-2512</td><td>Qwen/Qwen-Image-2512</td></tr><tr><td>PromptEnhancer-7B</td><td>tencent/HunyuanImage-2.1/reprompt</td></tr><tr><td>PromptEnhancer-32B</td><td>PromptEnhancer/PromptEnhancer-32B</td></tr></table>

## B.3 MODELS

Table 10 lists the checkpoints used in our experiments.

## B.4 BASELINES

Inference-time PE. Base generates directly from the raw prompt p, whereas Base+PE applies the PE and conditions the frozen generator on $p ^ { + }$ at inference time.

Supervised fine-tuning. For SFT, we first use each frozen base model to generate pseudo-target images from enhanced prompts under its corresponding inference configuration. We then fine-tune the model with the standard flow-matching objective on raw-prompt/image pairs $( p , x ^ { + } )$ . Thus, SFT transfers enhanced-prompt behavior through offline-generated images rather than online vector-field supervision.

Off-policy distillation. Off-policy distillation uses the same configuration as PE-OPSD. The only difference is the rollout distribution: off-policy states are generated by the teacher conditioned on $p ^ { + }$ , whereas PE-OPSD evaluates the teacher on states visited by the student conditioned on p.

## B.5 BENCHMARKS

In-domain evaluation. GenEval (Ghosh et al., 2023) evaluates object presence, counting, colors, spatial relations, and attribute binding using its detector-based protocol. GenEval2 (Kamath et al., 2025) contains 800 prompts with broader visual concepts and higher compositional complexity. We follow its official Soft-TIFA evaluation, reporting the geometric mean $( \mathrm { G E } 2 _ { \mathrm { G M } } )$ for prompt-level correctness and the arithmetic mean (GE2 ) for atom-level correctness. On both benchmarks, we additionally report CLIP score Hessel et al. (2021), PickScore (Kirstain et al., 2023) normalized by 26, and aesthetics score (Schuhmann, 2022).

Out-of-domain evaluation. DPG-Bench (Hu et al., 2024) evaluates dense prompt following over Global, Entity, Attribute, Relation, and Other categories. T2I-CompBench++ (Huang et al., 2025) evaluates attribute binding, numeracy, spatial and non-spatial relations, and complex compositions. For EvalMuse (Han et al., 2026), we follow the original protocol using its 200 representative prompts and the FGA-BLIP2 overall alignment score. All Overall scores are computed using the corresponding official evaluation procedures.

## C HUMAN EVALUATION PROTOCOL

Evaluation data and comparisons. We conduct human evaluation on all 800 prompts from the GenEval2 test set using Z-Image-Turbo as the generator and GPT-5.6 Sol as the PE. For each prompt, we compare Base against Base+PE, SFT, off-policy distillation, and PE-OPSD.

Evaluation interface and aggregation. Each comparison presents two generated images side by side, with the assignment of systems to the left and right positions randomized independently for every pair. Model identities and method names are hidden from the assessors, while the corresponding raw prompt is displayed above the images. Assessors provide two independent judgments (Left better or Right better): (1) Prompt Fidelity, indicating which image more faithfully satisfies the objects, attributes, counts, and relationships specified by the raw prompt; and (2) Visual Appeal, indicating which image has better overall perceptual quality and aesthetics. For each criterion, the assessor selects either the left or right image. For the result aggregation, each image pair is indepen dently evaluated by all three assessors for both criteria. We determine the preference for each pair by majority vote.

Assessors and quality control. We recruit three professional assessors who are formally contracted and compensated at locally competitive rates. Before participating in the study, all assessors are informed of the task and potential exposure to generated content. They receive detailed criterionspecific instructions and complete a qualification test covering representative evaluation cases.

## D ADDITIONAL EXPERIMENTAL RESULTS

## D.1 TRAINING EFFICIENCY

Table 11 reports the method-specific cost of post-training. Although SFT has a substantially lower per-step cost, it requires generating pseudo-target images for the entire training set. This preprocessing dominates its total cost for SD3.5-M and Z-Image, for which PE-OPSD reduces the training time from 7.0 to 3.1 hours and from 28.4 to 13.6 hours, respectively. For Z-Image-Turbo, few-step sampling makes pseudo-target generation inexpensive, and SFT is slightly faster overall. However, SFT yields lower performance than both off-policy distillation and PE-OPSD (see Table 2).

PE-OPSD and off-policy distillation have identical training costs because they use the same number of rollout and teacher evaluations, differing only in whether states are collected from the student or teacher trajectory. Overall, PE-OPSD avoids the additional image-generation and storage requirements of SFT while incurring the expected cost of online rollout supervision. Offline enhancedprompt construction is shared by all trainable methods and is therefore excluded from this methodspecific comparison.

Table 11: Training efficiency in our experiments. One-step denotes the wall-clock time per optimization step under the default configuration, and Full denotes the cost of 1,000 optimization steps. For SFT, Full<sup>†</sup> is reported as training time + offline pseudo-target generation time.
<table><tr><td rowspan="2">Model</td><td colspan="2">SFT</td><td colspan="2">Off-policy</td><td colspan="2">PE-OPSD</td></tr><tr><td>One-step</td><td>Full†</td><td>One-step</td><td>Full</td><td>One-step</td><td>Full</td></tr><tr><td>SD3.5-M</td><td>1.47s</td><td> $0 . 4 \substack { + 6 . 6 h }$ </td><td>11.02s</td><td>3.1h</td><td>11.02s</td><td>3.1h</td></tr><tr><td>Z-Image</td><td>3.14s</td><td> $0 . 9 \substack { + 2 7 . 5 h }$ </td><td>49.15s</td><td>13.6h</td><td>49.15s</td><td>13.6h</td></tr><tr><td>Z-Image-Turbo</td><td>3.14s</td><td> $0 . 9 + 2 . 5 h$ </td><td>14.31s</td><td>4.0h</td><td>14.31s</td><td>4.0h</td></tr></table>

## D.2 INFERENCE LATENCY

Table 12 compares deployment latency with and without inference-time PE. Because PE-OPSD generates directly from raw prompts using the original inference pipeline, it retains the latency of Base across all evaluated backbones. In contrast, inference-time PE introduces a substantial fixed overhead, particularly for efficient few-step generators. Relative to PromptEnhancer-7B, PE-OPSD provides 1.52×–12.21× speedups; with PromptEnhancer-32B, the speedups increase to $3 . 3 1 \times - 5 0 . 6 7 \times$ . The largest gains occur on Z-Image-Turbo and FLUX.2-klein, where prompt rewriting is considerably more expensive than image generation itself. These results demonstrate that PE-OPSD preserves the benefits of enhanced prompts without adding per-request deployment latency.

Table 12: Inference latency and speedup across different settings. Latency is the average generation time per image in seconds. +PE includes both prompt rewriting and image generation, whereas +Ours generates directly from the raw prompt without invoking PE.
<table><tr><td rowspan="2">Model</td><td colspan="3">Latency (s)</td><td rowspan="2">Speedup</td></tr><tr><td>Base</td><td>+PE</td><td>+Ours</td></tr><tr><td colspan="4">PromptEnhancer-7B as PE</td></tr><tr><td>SD3.5-M</td><td>2.70</td><td>10.36</td><td>2.70</td><td>3.84×</td></tr><tr><td>Z-Image</td><td>9.69</td><td>19.00</td><td>9.69</td><td>1.96×</td></tr><tr><td>Z-Image-Turbo</td><td>0.88</td><td>8.53</td><td>0.88</td><td>9.69×</td></tr><tr><td>FLUX.2-klein</td><td>0.67</td><td>8.18</td><td>0.67</td><td>12.21×</td></tr><tr><td>FLUX.2-klein-base</td><td>14.45</td><td>21.97</td><td>14.45</td><td>1.52×</td></tr><tr><td>QwenImage-2512</td><td>13.38</td><td>22.09</td><td>13.38</td><td>1.65×</td></tr><tr><td colspan="5">PromptEnhancer-32B as PE</td></tr><tr><td>SD3.5-M</td><td>2.70</td><td>35.91</td><td>2.70</td><td>13.30×</td></tr><tr><td>Z-Image</td><td>9.69</td><td>43.74</td><td>9.69</td><td>4.51×</td></tr><tr><td>Z-Image-Turbo</td><td>0.88</td><td>34.21</td><td>0.88</td><td>38.88×</td></tr><tr><td>FLUX.2-klein</td><td>0.67</td><td>33.95</td><td>0.67</td><td>50.67×</td></tr><tr><td>FLUX.2-klein-base</td><td>14.45</td><td>47.77</td><td>14.45</td><td>3.31×</td></tr><tr><td>QwenImage-2512</td><td>13.38</td><td>47.33</td><td>13.38</td><td>3.54×</td></tr></table>

## D.3 EFFECT OF EMA TEACHER

As shown in Figure 8 and Figure 9, we study the effect of EMA teacher using four settings: a frozen Base teacher without EMA updates (No EMA), $\gamma = 0 . 9 9 , \gamma = 0 . 9$ , and our default $\gamma = 0 . 9 9 9$ . We conduct the experiments using Z-Image-Turbo with 500 training steps.

The frozen teacher performs similarly to the default setting during early training but reaches a lower performance ceiling, indicating that allowing the teacher to evolve with the student provides stronger supervision at later stages. With $\gamma = 0 . 9 9$ , generated images remain visually coherent, although performance is slightly lower than with $\gamma = 0 . 9 9 9$

A more aggressive update with $\gamma = 0 . 9$ substantially degrades generation quality. Although its training loss decreases rapidly, generated images contain pronounced artifacts and evaluation scores fall below Base. This suggests that an overly responsive teacher becomes too tightly coupled to the student and provides insufficiently stable targets. We therefore use $\gamma = 0 . 9 9 9$ , which balances teacher adaptation with temporal stability.

![](images/4d26f68314decb66a1a04a4d054425e54e4d3294d954336541d883dda6092f64.jpg)

![](images/3f3688886d2b7a3bd36de07774dbe0d837b06b442f7ae9533c7aa411d9627b71.jpg)

![](images/536006a63d7206e7f3b1f3b0e032336ccb9087c3bdd7942fa86fd6b1ab3dad62.jpg)

Figure 8: Training dynamics and GenEval performance with different EMA settings. Left: the loss curves where faint lines denote raw losses and bold lines show a the moving average; Right: the corresponding GenEval performance curves.  
![](images/966c67b7a4629cd82d6c3bedba4c021393a6c4682bf698a5731c8bad84556f3a.jpg)  
Figure 9: Visual examples. We visualize the samples generated by the student and teacher across different EMA settings at 500 training steps.

## D.4 DETAILED RESULTS OF SCALABILITY

Table 13 reports the complete results on larger models. PE-OPSD achieves the highest aggregate prompt-fidelity improvement across all three backbones and consistently leads on GE and GE2 . It also improves aggregate visual appeal over Base, although Base+PE remains marginally better on the two FLUX variants. On QwenImage-2512, PE-OPSD obtains the best aggregate results for both objectives, further supporting its applicability to larger models.

Table 13: Results on scaling up to large models. We use GPT-5.6 Sol as PE. PickScore is normalized by 26; Aes. denotes aesthetics; Bold: best; Underlined: second-best.
<table><tr><td rowspan="2">Method</td><td colspan="4">GenEval (GE) Task</td><td colspan="5">GenEval2 (GE2) Task</td><td rowspan="2">IpF</td><td rowspan="2"> $\mathbb { I } _ { \mathrm { V A } }$ </td></tr><tr><td>GE</td><td>PickScore</td><td>CLIP</td><td>Aes.</td><td>GE2GM</td><td> $\mathbf { G E 2 } _ { \mathbf { A M } }$ </td><td>PickScore</td><td>CLIP</td><td>Aes.</td></tr><tr><td>FLUX.2-klein-base (9B)</td><td>0.775</td><td>0.892</td><td>0.303</td><td>5.140</td><td>0.359</td><td>0.773</td><td>0.880</td><td>0.328</td><td>5.310</td><td>0.00%</td><td>0.00%</td></tr><tr><td>+PE</td><td>0.862</td><td>0.910</td><td>0.302</td><td>5.467</td><td>0.442</td><td>0.838</td><td>0.901</td><td>0.330</td><td>5.658</td><td>8.61%</td><td>4.33%</td></tr><tr><td>+Ours</td><td>0.873</td><td>0.913</td><td>0.305</td><td>5.421</td><td>0.444</td><td>0.831</td><td>0.905</td><td>0.333</td><td>5.628</td><td>9.20%</td><td>4.16%</td></tr><tr><td>∆ (vs Base)</td><td>+0.098</td><td>+0.021</td><td>+0.002</td><td>+0.281</td><td>+0.085</td><td>+0.058</td><td>+0.025</td><td>+0.005</td><td>+0.318</td><td>+9.20%</td><td>+4.16%</td></tr><tr><td>FLUX.2-klein (9B)</td><td>0.856</td><td>0.911</td><td>0.299</td><td>5.288</td><td>0.348</td><td>0.797</td><td>0.901</td><td>0.329</td><td>5.430</td><td>0.00%</td><td>0.00%</td></tr><tr><td>+PE</td><td>0.859</td><td>0.915</td><td>0.300</td><td>5.474</td><td>0.381</td><td>0.827</td><td>0.905</td><td>0.327</td><td>5.682</td><td>2.66%</td><td>2.26%</td></tr><tr><td>+Ours</td><td>0.867</td><td>0.912</td><td>0.301</td><td>5.433</td><td>0.434</td><td>0.825</td><td>0.907</td><td>0.328</td><td>5.700</td><td>5.98%</td><td>2.12%</td></tr><tr><td>∆ (vs Base)</td><td>+0.011</td><td>+0.001</td><td>+0.002</td><td>+0.145</td><td>+0.086</td><td>+0.028</td><td>+0.006</td><td>-0.001</td><td>+0.270</td><td>+5.98%</td><td>+2.12%</td></tr><tr><td>QwenImage-2512 (20B)</td><td>0.620</td><td>0.895</td><td>0.283</td><td>5.377</td><td>0.163</td><td>0.671</td><td>0.894</td><td>0.303</td><td>5.822</td><td>0.00%</td><td>0.00%</td></tr><tr><td>+PE</td><td>0.839</td><td>0.916</td><td>0.297</td><td>5.462</td><td>0.380</td><td>0.813</td><td>0.908</td><td>0.321</td><td>5.782</td><td>40.1%</td><td>1.20%</td></tr><tr><td>+Ours</td><td>0.857</td><td>0.920</td><td>0.302</td><td>5.421</td><td>0.399</td><td>0.820</td><td>0.916</td><td>0.324</td><td>5.857</td><td>43.8%</td><td>1.67%</td></tr><tr><td>∆ (vs Base)</td><td>+0.237</td><td>+0.025</td><td>+0.019</td><td>+0.044</td><td>+0.236</td><td>+0.149</td><td>+0.022</td><td>+0.021</td><td>+0.035</td><td>+43.8%</td><td>+1.67%</td></tr></table>

## D.5 MORE ANALYSIS IN TABLE 4

Robustness across PEs. Table 4 shows that the performance of SFT, inference-time PE, and offpolicy distillation varies with the chosen PE, whereas PE-OPSD consistently provides the strongest aggregate results. In particular, PE-OPSD improves I over off-policy distillation by 1.80–3.78 percentage points and $\mathbb { I } _ { \mathrm { V A } }$ by 0.18–0.33 points across the three PEs. Since the two distillation methods share the teacher condition, objective, and optimization budget, this consistent margin supports the importance of supervising the student on states visited by its own raw-prompt trajectories.

Effect of the teacher condition. PE-OPSD also exceeds inference-time PE in both aggregate indices while requiring only raw prompts at deployment. Notably, its largest fidelity margin over Base+PE occurs with PromptEnhancer-7B, whose direct inference-time improvement is the weakest among the tested PEs. This indicates that the transferred benefit is not determined solely by the PE’s one-shot inference performance. Meanwhile, PromptEnhancer-32B produces the strongest fidelity supervision, whereas GPT-5.6 Sol yields the largest visual-appeal improvement. Overall, PE-OPSD remains effective across PEs with substantially different capacities and enhancement behaviors.

## D.6 DIFFERENT PES

Additional experimental results for different PEs. We further visualize the training dynamics with three PEs in our experiments across SD3.5-M, Z-Image, and Z-Image-Turbo. As shown in Figure 10, the PE-OPSD loss consistently decreases and stabilizes within 1,000 training steps for all PEs and models. These similar optimization trends demonstrate that PE-OPSD remains stable across different PEs and does not rely on a specific PE.

![](images/d3bf376d9cd38522d2230e67bf2b33421cffeb68048b0a69303658be3f8878fe.jpg)

![](images/54c64646a01c63a151fa920115f5ede492c051ca701272544adcf72e8e0d3880.jpg)

![](images/b6a03d5cb938731cbc86fa3f85425958be278aa63157864782f3422276039684.jpg)  
Figure 10: Training dynamics with different PEs. We report the PE-OPSD loss over 1,000 training steps. Faint lines denote raw losses, while bold lines show a the moving average.

Prompt examples. Figures 11–13 show examples of raw prompts and their enhanced versions produced by different PEs.

![](images/b3875ec8485045f4e6501acfa77a655ee4132ba4637b0bb4f10609e41ae5e51d.jpg)  
Figure 11: Comparison between the raw prompt and the enhanced prompt. Here, we use GPT-5.6 Sol as PE.

![](images/4da21bc313f60b7c04e01bc38fcefbcfa9274d2da3b09779e5ce91f95f831cae.jpg)  
Figure 12: Comparison between the raw prompt and the enhanced prompt. Here, we use PromptEnhancer-7B as PE.

![](images/d2f76d58bfeb356743b4ffc9d2bcefb9db810dd1f7a292e3e142dcf28a000ff7.jpg)  
Figure 13: Comparison between the raw prompt and the enhanced prompt. Here, we use PromptEnhancer-32B as PE.

## E ADDITIONAL BENCHMARK RESULTS

## E.1 GENEVAL DETAILS

In Table 14 and Table 15, we provide the detailed GenEval performance breakdown for different models and PEs. In particular, we report the fine-grained performance in Single Object, Two Object, Counting, Colors, Position, and Attribute Binding. We also report the Overall score.

## E.2 GENEVAL2 DETAILS

In Table 16 and Table 17, we provide the detailed GenEval2 performance breakdown for different models and PEs. In particular, we report the fine-grained performance in Object, Attribute, Count, Position, and Verb. We also report the overall soft-tifa score in GenEval2 and GenEval2 .

Table 14: Detailed GenEval performance breakdown. We report the fine-grained performance in Single Object, Two Object, Counting, Colors, Position, and Attribute Binding. We also report the Overall score. Here, we use GPT-5.6 Sol as PE.
<table><tr><td>Method</td><td>Single Obj</td><td>Two Obj</td><td>Counting</td><td>Color</td><td>Position</td><td>Attr Binding</td><td>Overall</td></tr><tr><td colspan="8">SD3.5-M (2.5B)</td></tr><tr><td>Base</td><td>0.975</td><td>0.778</td><td>0.613</td><td>0.787</td><td>0.223</td><td>0.472</td><td>0.628</td></tr><tr><td>Base+PE</td><td>0.959</td><td>0.838</td><td>0.634</td><td>0.838</td><td>0.603</td><td>0.635</td><td>0.743</td></tr><tr><td>SFT</td><td>0.972</td><td>0.856</td><td>0.691</td><td>0.832</td><td>0.512</td><td>0.500</td><td>0.718</td></tr><tr><td>Off-Policy Distillation</td><td>0.981</td><td>0.846</td><td>0.628</td><td>0.838</td><td>0.698</td><td>0.650</td><td>0.770</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.988</td><td>0.886</td><td>0.631</td><td>0.878</td><td>0.698</td><td>0.710</td><td>0.797</td></tr><tr><td colspan="8"></td></tr><tr><td>Base</td><td>0.959</td><td>0.801</td><td>Z-Image (6B) 0.569</td><td>0.809</td><td>0.338</td><td>0.480</td><td>0.650</td></tr><tr><td>Base+PE</td><td>0.978</td><td>0.879</td><td>0.619</td><td>0.888</td><td>0.755</td><td>0.743</td><td>0.810</td></tr><tr><td>SFT</td><td>0.972</td><td>0.884</td><td>0.697</td><td>0.902</td><td>0.780</td><td>0.698</td><td>0.821</td></tr><tr><td>Off-Policy Distillation</td><td>0.988</td><td>0.886</td><td>0.641</td><td>0.899</td><td>0.762</td><td>0.767</td><td>0.824</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.988</td><td>0.884</td><td>0.656</td><td>0.918</td><td>0.823</td><td>0.805</td><td>0.846</td></tr><tr><td colspan="8"></td></tr><tr><td>Base</td><td>0.988</td><td>0.833</td><td>z-Image-Turbo 0.759</td><td>(6B) 0.859</td><td>0.460</td><td>0.588</td><td>0.737</td></tr><tr><td>Base+PE</td><td>0.975</td><td>0.864</td><td>0.803</td><td>0.923</td><td>0.770</td><td>0.787</td><td>0.850</td></tr><tr><td>SFT</td><td>0.981</td><td>0.732</td><td>0.625</td><td>0.894</td><td>0.708</td><td>0.677</td><td>0.766</td></tr><tr><td>Off-Policy Distillation</td><td>0.984</td><td>0.866</td><td>0.778</td><td>0.907</td><td>0.805</td><td>0.777</td><td>0.851</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.969</td><td>0.879</td><td>0.794</td><td>0.915</td><td>0.797</td><td>0.835</td><td>0.863</td></tr><tr><td colspan="8"></td></tr><tr><td></td><td></td><td></td><td>FLUX.2-klein-base</td><td>(9B)</td><td></td><td></td><td></td></tr><tr><td>Base</td><td>0.994</td><td>0.851</td><td>0.706</td><td>0.904</td><td>0.632</td><td>0.603</td><td>0.775</td></tr><tr><td>Base+PE PE-OPSD (Ours)</td><td>0.984 0.991</td><td>0.904 0.904</td><td>0.797 0.819</td><td>0.928 0.912</td><td>0.820 0.840</td><td>0.757 0.785</td><td>0.862 0.873</td></tr><tr><td colspan="8"></td></tr><tr><td>Base</td><td>0.994</td><td>0.912</td><td>FLUX.2-klein 0.847</td><td>(9B) 0.894</td><td>0.733</td><td>0.787</td><td>0.856</td></tr><tr><td>Base+PE</td><td>0.991</td><td>0.896</td><td>0.775</td><td>0.912</td><td>0.835</td><td>0.757</td><td>0.859</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.988</td><td>0.874</td><td>0.747</td><td>0.931</td><td>0.845</td><td>0.823</td><td>0.867</td></tr><tr><td colspan="8">QwenImage-2512 (20B)</td></tr><tr><td>Base</td><td>0.984</td><td>0.798</td><td>0.331</td><td>0.832</td><td>0.302</td><td>0.500</td><td>0.620</td></tr><tr><td>Base+PE</td><td>0.975</td><td>0.904</td><td>0.694</td><td>0.891</td><td>0.755</td><td>0.818</td><td>0.839</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.981</td><td>0.899</td><td>0.706</td><td>0.910</td><td>0.810</td><td>0.833</td><td>0.857</td></tr></table>

Table 15: Detailed GenEval performance breakdown on Z-image with other PEs. We report the fine-grained performance in Single Object, Two Object, Counting, Colors, Position, and Attribute Binding. We also report the Overall score.
<table><tr><td>Method</td><td>Single Obj</td><td>Two Obj</td><td>Counting</td><td>Color</td><td>Position</td><td>Attr Binding</td><td>Overall</td></tr><tr><td colspan="8">PromptEnhancer-7B as PE</td></tr><tr><td>Base</td><td>0.959</td><td>0.801</td><td>0.569</td><td>0.809</td><td>0.338</td><td>0.480</td><td>0.650</td></tr><tr><td>Base+PE</td><td>0.947</td><td>0.801</td><td>0.588</td><td>0.750</td><td>0.505</td><td>0.555</td><td>0.684</td></tr><tr><td>SFT</td><td>0.966</td><td>0.871</td><td>0.762</td><td>0.835</td><td>0.627</td><td>0.560</td><td>0.763</td></tr><tr><td>Off-Policy Distillation</td><td>0.978</td><td>0.856</td><td>0.716</td><td>0.856</td><td>0.613</td><td>0.623</td><td>0.767</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.978</td><td>0.864</td><td>0.647</td><td>0.872</td><td>0.610</td><td>0.740</td><td>0.782</td></tr><tr><td colspan="8">PromptEnhancer-32B</td></tr><tr><td>Base</td><td>0.959</td><td>0.801</td><td>0.569</td><td>3 as PE 0.809</td><td>0.338</td><td>0.480</td><td>0.650</td></tr><tr><td>Base+PE</td><td>0.984</td><td>0.854</td><td>0.675</td><td>0.902</td><td>0.797</td><td>0.690</td><td>0.815</td></tr><tr><td>SFT</td><td>0.972</td><td>0.884</td><td>0.741</td><td>0.888</td><td>0.825</td><td>0.680</td><td>0.829</td></tr><tr><td>Off-Policy Distillation</td><td>0.981</td><td>0.889</td><td>0.725</td><td>0.896</td><td>0.853</td><td>0.718</td><td>0.842</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.991</td><td>0.904</td><td>0.666</td><td>0.941</td><td>0.882</td><td>0.810</td><td>0.868</td></tr></table>

Table 16: Detailed GenEval2 performance breakdown. We report the fine-grained performance in Object, Attribute, Count, Position, and Verb. We also report the overall soft-tifa score in GenEval2<sub>AM</sub> and GenEval2 . Here, we use GPT-5.6 Sol as PE.
<table><tr><td>Method</td><td>Object</td><td>Attribute</td><td>Count</td><td>Position</td><td>Verb</td><td> $\mathbf { G e n E v a l } 2 _ { \mathrm { A M } }$ </td><td> $\mathbf { G e n E v a l } 2 _ { \mathsf { G M } }$ </td></tr><tr><td colspan="8">SD3.5-M (2.5B)</td></tr><tr><td>Base</td><td>0.859</td><td>0.672</td><td>0.431</td><td>0.397</td><td>0.161</td><td>0.633</td><td>0.176</td></tr><tr><td>Base+PE</td><td>0.868</td><td>0.751</td><td>0.476</td><td>0.521</td><td>0.207</td><td>0.680</td><td>0.220</td></tr><tr><td>SFT</td><td>0.835</td><td>0.703</td><td>0.478</td><td>0.438</td><td>0.124</td><td>0.656</td><td>0.207</td></tr><tr><td>Off-Policy Distillation</td><td>0.850</td><td>0.715</td><td>0.461</td><td>0.483</td><td>0.195</td><td>0.671</td><td>0.223</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.868</td><td>0.758</td><td>0.477</td><td>0.489</td><td>0.179</td><td>0.682</td><td>0.226</td></tr><tr><td colspan="8"></td></tr><tr><td>Base</td><td>0.927</td><td>0.844</td><td>Z-Image 0.599</td><td>(6B) 0.607</td><td>0.294</td><td>0.761</td><td>0.306</td></tr><tr><td>Base+PE</td><td>0.973</td><td>0.923</td><td>0.637</td><td>0.826</td><td>0.459</td><td>0.827</td><td>0.404</td></tr><tr><td>SFT</td><td>0.981</td><td>0.906</td><td>0.643</td><td>0.816</td><td>0.323</td><td>0.834</td><td>0.403</td></tr><tr><td>Off-Policy Distillation</td><td>0.975</td><td>0.931</td><td>0.625</td><td>0.814</td><td>0.320</td><td>0.828</td><td>0.395</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.975</td><td>0.938</td><td>0.634</td><td>0.816</td><td>0.381</td><td>0.831</td><td>0.406</td></tr><tr><td colspan="8">z-Image-Turbo</td></tr><tr><td>Base</td><td>0.970</td><td>0.763</td><td>0.665</td><td>(6B) 0.624</td><td>0.196</td><td>0.783</td><td>0.341</td></tr><tr><td>Base+PE</td><td>0.973</td><td>0.887</td><td>0.688</td><td>0.845</td><td>0.285</td><td>0.843</td><td>0.479</td></tr><tr><td>SFT</td><td>0.929</td><td>0.866</td><td>0.601</td><td>0.799</td><td>0.205</td><td>0.788</td><td>0.413</td></tr><tr><td>Off-Policy Distillation</td><td>0.968</td><td>0.892</td><td>0.640</td><td>0.821</td><td>0.297</td><td>0.821</td><td>0.437</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.975</td><td>0.894</td><td>0.686</td><td>0.851</td><td>0.353</td><td>0.844</td><td>0.501</td></tr><tr><td colspan="8"></td></tr><tr><td>FLUX.2-klein-base (9B) Base</td><td>0.929 0.863</td><td></td><td>0.567</td><td>0.710</td><td>0.340</td><td>0.773</td><td>0.359</td></tr><tr><td>Base+PE</td><td>0.962</td><td>0.950</td><td>0.647</td><td>0.838</td><td>0.477</td><td>0.838</td><td>0.442</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.956</td><td>0.945</td><td>0.641</td><td>0.830</td><td>0.399</td><td>0.831</td><td>0.444</td></tr><tr><td colspan="8">FLUX.2-klein</td></tr><tr><td>Base</td><td>0.954</td><td>0.889</td><td>0.593</td><td>(9B) 0.742</td><td>0.315</td><td>0.797</td><td>0.348</td></tr><tr><td>Base+PE</td><td>0.961</td><td>0.952</td><td>0.609</td><td>0.843</td><td>0.421</td><td>0.827</td><td>0.381</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.960</td><td>0.947</td><td>0.616</td><td>0.846</td><td>0.362</td><td>0.825</td><td>0.434</td></tr><tr><td colspan="8">QwenImage-2512 (20B)</td></tr><tr><td>Base</td><td>0.930</td><td>0.593</td><td>0.526</td><td>0.510</td><td>0.270</td><td>0.671</td><td>0.163</td></tr><tr><td>Base+PE</td><td>0.977</td><td>0.848</td><td>0.620</td><td>0.843</td><td>0.423</td><td>0.813</td><td>0.380</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.978</td><td>0.882</td><td>0.625</td><td>0.851</td><td>0.397</td><td>0.820</td><td>0.399</td></tr></table>

Table 17: Detailed GenEval2 performance breakdown on Z-Image with other PEs. We report the fine-grained performance in Object, Attribute, Count, Position, and Verb. We also report the overall soft-tifa score in GenEval2 and GenEval2 .
<table><tr><td>Method</td><td>Object</td><td>Attribute</td><td>Count</td><td>Position</td><td>Verb</td><td> $\mathbf { G e n E v a l } 2 _ { \mathrm { A M } }$ </td><td> $\mathbf { G e n E v a l } 2 _ { \mathsf { G M } }$ </td></tr><tr><td colspan="8">PromptEnhancer-7B as PE</td></tr><tr><td>Base</td><td>0.927</td><td>0.844</td><td>0.599</td><td>0.607</td><td>0.294</td><td>0.761</td><td>0.306</td></tr><tr><td>Base+PE</td><td>0.962</td><td>0.818</td><td>0.653</td><td>0.723</td><td>0.471</td><td>0.801</td><td>0.375</td></tr><tr><td>SFT</td><td>0.975</td><td>0.865</td><td>0.660</td><td>0.765</td><td>0.340</td><td>0.819</td><td>0.400</td></tr><tr><td>Off-Policy Distillation</td><td>0.975</td><td>0.832</td><td>0.649</td><td>0.720</td><td>0.274</td><td>0.804</td><td>0.380</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.979</td><td>0.875</td><td>0.636</td><td>0.784</td><td>0.388</td><td>0.819</td><td>0.422</td></tr><tr><td colspan="8">PromptEnhancer-32B as PE</td></tr><tr><td>Base</td><td>0.927</td><td>0.844</td><td>0.599</td><td>0.607</td><td>0.294</td><td>0.761</td><td>0.306</td></tr><tr><td>Base+PE</td><td>0.966</td><td>0.924</td><td>0.664</td><td>0.864</td><td>0.543</td><td>0.844</td><td>0.470</td></tr><tr><td>SFT</td><td>0.981</td><td>0.925</td><td>0.672</td><td>0.848</td><td>0.339</td><td>0.845</td><td>0.440</td></tr><tr><td>Off-Policy Distillation</td><td>0.970</td><td>0.940</td><td>0.662</td><td>0.832</td><td>0.382</td><td>0.843</td><td>0.459</td></tr><tr><td>PE-OPSD (Ours)</td><td>0.982</td><td>0.952</td><td>0.655</td><td>0.845</td><td>0.430</td><td>0.846</td><td>0.476</td></tr></table>

## E.3 EVALMUSE DETAILS

In Table 18, we provide the detailed performance breakdown for different models. In particular, we report the fine-grained performance in Attribute, Location, Color, Object, Material, A./H. (Animal/Human), Food, Shape, Activity, Spatial, and Counting. We also report the overall score following the official guidance (Han et al., 2026).

Table 18: Detailed EvalMuse performance breakdown. We report the fine-grained performance in Attribute, Location, Color, Object, Material, A./H. (Animal/Human), Food, Shape, Activity, Spatial, and Counting. We also report the overall score. Bold: best in Overall.
<table><tr><td>Method</td><td>Attribute</td><td>Location</td><td>Color</td><td>Object</td><td>Material</td><td>A./H.</td><td>Food</td><td>Shape</td><td>Activity</td><td>Spatial</td><td>Counting</td><td>Overall</td></tr><tr><td>SD3.5-M</td><td>0.801</td><td>0.724</td><td>0.570</td><td>0.675</td><td>0.542</td><td>0.524</td><td>0.654</td><td>0.826</td><td>0.539</td><td>0.459</td><td>0.234</td><td>3.199</td></tr><tr><td>+Ours</td><td>0.820</td><td>0.763</td><td>0.634</td><td>0.710</td><td>0.587</td><td>0.562</td><td>0.697</td><td>0.832</td><td>0.551</td><td>0.588</td><td>0.283</td><td>3.368</td></tr><tr><td>Z-Image</td><td>0.797</td><td>0.723</td><td>0.618</td><td>0.709</td><td>0.620</td><td>0.608</td><td>0.656</td><td>0.806</td><td>0.582</td><td>0.655</td><td>0.314</td><td>3.340</td></tr><tr><td>+Ours</td><td>0.809</td><td>0.768</td><td>0.694</td><td>0.738</td><td>0.681</td><td>0.654 0.716</td><td></td><td>0.811</td><td>0.596</td><td>0.643</td><td>0.384</td><td>3.561</td></tr><tr><td>Z-Image-Turbo</td><td>0.800</td><td>0.762</td><td>0.711</td><td>0.727</td><td>0.707</td><td>0.620 0.699</td><td></td><td>0.798</td><td>0.589</td><td>0.641</td><td>0.340</td><td>3.515</td></tr><tr><td>+Ours</td><td>0.807</td><td>0.753</td><td>0.626</td><td>0.729</td><td>0.639</td><td>0.632 0.735</td><td></td><td>0.806</td><td>0.620</td><td>0.584</td><td>0.337</td><td>3.534</td></tr><tr><td>FLUX.2-klein-base</td><td>0.834</td><td>0.784</td><td>0.707</td><td>0.749</td><td>0.680</td><td>0.659</td><td>0.708</td><td>0.765</td><td>0.644</td><td>0.661</td><td>0.311</td><td>3.638</td></tr><tr><td>+Ours</td><td>0.829</td><td>0.797</td><td>0.728</td><td>0.755</td><td>0.691</td><td>0.700</td><td>0.733</td><td>0.798</td><td>0.664</td><td>0.680</td><td>0.367</td><td>3.733</td></tr><tr><td>FLUX.2-klein</td><td>0.832</td><td>0.778</td><td>0.716</td><td>0.761</td><td>0.686</td><td>0.691</td><td>0.728</td><td>0.834</td><td>0.661</td><td>0.675</td><td>0.337</td><td>3.724</td></tr><tr><td>+Ours</td><td>0.832</td><td>0.788</td><td>0.725</td><td>0.760</td><td>0.691</td><td>0.687</td><td>0.726</td><td>0.810</td><td>0.660</td><td>0.686</td><td>0.333</td><td>3.710</td></tr><tr><td>QwenImage-2512</td><td>0.817</td><td>0.757</td><td>0.616</td><td>0.720</td><td>0.662</td><td>0.685</td><td>0.708</td><td>0.835</td><td>0.618</td><td>0.605</td><td>0.280</td><td>3.485</td></tr><tr><td>+Ours</td><td>0.829</td><td>0.798</td><td>0.749</td><td>0.772</td><td>0.750</td><td>0.693</td><td>0.736</td><td>0.837</td><td>0.643</td><td>0.725</td><td>0.397</td><td>3.756</td></tr></table>

## F SYSTEM PROMPTS

## F.1 SYSTEM PROMPT FOR CONSISTENCY FILTERING

For GPT-generated enhanced prompts, we employ an consistency judge to identify expansions that alter or omit critical semantics from the original prompt. The judge verifies exact counts, object identities, attributes, actions, and spatial relations, and returns a binary decision in a structured JSON format. The exact system prompt is provided below.

## System prompt for consistency filtering

You are a strict consistency judge for enhanced text-to-image prompts.

The original prompt is the source of truth. The candidate enhanced prompt is consistent only if it explicitly preserves every critical fact: • exact counts and object identities;

• colors, visual material appearances, textures, and other attributes;

• actions and subject/object roles;

• spatial relations and ordering.

Do not treat a fact as preserved when it is merely implied. Synonyms and grammatical number words are acceptable when their meaning is unambiguous. Additional composition, lighting, style, and background details are acceptable only when they do not alter or contradict an original fact. Be especially strict about swapped roles, changed counts, merged object groups, conflicting attributes, and missing relations.

Do not relax colors, counts, identities, actions, or relations. In particular, spots, reflections, or small localized details do not satisfy an overall color requirement when the object is explicitly described as a different base color.

Return exactly one JSON object with no Markdown:   
{"consistent": true, "issues": []}   
or   
{"consistent": false, "issues": ["specific issue", "..."]}   
Judge only. Do not rewrite, complete, or suggest a replacement prompt.

## F.2 SYSTEM PROMPTS FOR GENERATING ENHANCED PROMPTS.

we provide the system prompt used by GPT-5.6 Sol to generate enhanced prompts. Given a raw prompt, the enhancer is instructed to produce only the expanded caption without auxiliary explanations or Markdown formatting. The instruction encourages a hierarchical description while explicitly preserving the objects, attributes, spatial relations, rendered text, and intellectual-property subjects specified in the original prompt.

## System prompt for GPT-5.6 Sol

You are an expert in writing prompts for image generation. I will give you a sentence, and you are to expand this sentence into a detailed caption for generating an image. And the captions must follow the rules listed below.

I. Sentence Structures The captions follow a consistent, hierarchical structure that moves from a general overview to specific details.

1. The Opening Statement: General Overview

2. The Body: Systematic and Spatially Organized Description

3. Hierarchical Object Description: From Whole to Parts

4. The Concluding Statement: Stylistic Identification

II. Grammatical Rules The grammar is precise, descriptive, and maintains an objective tone.

1. Tense: Consistent Present Tense

2. Voice: Mix of Active and Passive

3. Prepositional Phrases for Precision

4. Participial Phrases for Efficient Detail

5. Rich and Specific Adjectives

6. Precision and Hedging Language

7. Complex and Compound Sentences

## Key constraints:

1. Only provide the final captions, do not use markdown format.

2. The expanded captions must follow the rules listed above.

3. The expanded captions should adhere to the original sentence, especially the subject and the subject’s attributes, including color, size, spatial relationships, etc.

4. You can use your world knowledge to expand some professional terminology to proper explanations that suitable for image generation models.

5. If the style of original sentence is not mentioned, you should assume it is a photography style. And you can infer the style from the context of the sentence if the photography style is not suitable.

6. Describe the scene or subject directly, do not use “The image”, “The composition”, “The scene” and similar words in the beginning of the captions.

7. If the original sentence has a IP subject, you should keep the IP subject in the expanded captions, and describe the background of the IP in the expanded captions

8. If the original sentence has a text that need to be rendered, you should keep the text in the expanded captions, and format text as “rendered text”.

Next, I will give you my sentence. Please provide the expanded captions:

## G ADDITIONAL QUALITATIVE EXAMPLES

We provide more visual examples across different models and methods in Figure 14–Figure 16.

## H LIMITATIONS AND FUTURE WORK

PE-OPSD uses enhanced prompts as privileged supervision, and its learning signal is therefore influenced by the quality and semantic faithfulness of the selected PE. Although our consistency filtering and experiments with multiple PEs demonstrate robust improvements, inaccurate or overly specific rewrites may still introduce undesirable supervision. Future work could investigate confidenceaware filtering, agreement across multiple PEs, and adaptive supervision that emphasizes reliable prompt elements.

PE-OPSD shifts computation from per-request inference to offline prompt construction and posttraining. Reducing this cost through selective timestep supervision and fewer teacher evaluations is a promising direction. More broadly, extending privileged-condition distillation beyond flowmatching text-to-image models may further establish its applicability across generative paradigms.

#\$%&  
!"  
'()  
\*++,!-./01  
\*23%  
![](images/a9371fd0001060228699ca689521ea4e3458877ade584d52672e42b9f6d7a64a.jpg)  
Figure 14: Visual examples in SD3.5-M. We use the GPT-5.6 Sol as PE.

#\$%&  
!"  
'()  
\*++,!-./01  
\*23%  
![](images/52fb414c2e27887ac47fed6bb8301ad664926d849ec436864684e6f8bda760d2.jpg)  
Figure 15: Visual examples in Z-Image. We use the GPT-5.6 Sol as PE.

#\$%&  
!"  
'()  
\*++,!-./01  
\*23%  
![](images/f5fd3acccf5c7292bb286bee559a1e4575da98b91b69e5a21f94e85ed02c263b.jpg)  
Figure 16: Visual examples in Z-Image-Turbo. We use the GPT-5.6 Sol as PE.