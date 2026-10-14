SD3.5-Medium

# Simplifying and Improving On-Policy Distillation for Few-Step Diffusion

Jiawei Zhang<sup>1</sup>, Ziyu Wan<sup>1,†</sup>, Nanye Ma<sup>2</sup>, Long Zhao, Dong Chen<sup>1</sup>, Hongyu Liu<sup>3</sup> Qifeng Chen<sup>3</sup>, Dongdong Chen<sup>1</sup>

<sup>1</sup>Microsoft Research <sup>2</sup>NYU <sup>3</sup>HKUST

 Project Page § Code ò Model õ Data

## Abstract

Few-step distillation reduces the sampling cost of diffusion and flow models, but existing approaches often require auxiliary score models or specialized architectures. We show that simple on-policy velocity regression already provides a strong foundation for 4-NFE generation. To represent teacher velocity variation within each coarse transition, we introduce Pivot, a training-only residual proxy that separates local corrections from the student’s interval-mean prediction. We further repel the student from weaker teacher predictions, providing a lightweight mechanism for avoiding undesirable modes. The resulting framework, Simplified Matching Distillation (SMD), remains entirely regression-based, requires neither online score estimation nor adversarial training, and leaves the deployed generator unchanged. We evaluate SMD on image generation and editing, 3D generation, video synthesis, and joint audio–video generation, scaling to 33 billion parameters. At 4-NFE, SMD outperforms strong few-step baselines and surpasses multi-step teachers on several benchmarks. It also supports cross-model, multi-teacher, and causal-video distillation.

![](images/a2983fecba6e2a52786efa0258dc680654a83f4ad16a64d9c0f6bf2afd68c1a1.jpg)

Qwen-Image  
![](images/9fbd4555947f216ceac3de5d39432f789ec95a8f00af6e7144332296ee82a22e.jpg)

![](images/c7127ffe8daabaa3eb544e84bd0c304643448bb348b1d8ebdeb5c04f823cef8d.jpg)  
Qwen-Image-Edit-2511 SMD Lightning

![](images/26f324f2f5d5613d35e553c9182a13e8fc4bc5709c0fcebc29e1fd16cb58b835.jpg)

![](images/71df44b7123109b91694981d2a14256b46de2cd04be63fe270dfd27098670b71.jpg)  
MiniMax H3 SMD LightX2V-DMD v1.1

![](images/e5a814ced2aedbba4cd5d88d2c4c4ce703cede0836206fc6edb69b0e3ac18533.jpg)  
Figure 1. Simplicity scales. SMD at 4-NFE across image, video, editing, multi-teacher, and audio–video generation. Radial scales are truncated per metric. Column scales start at zero. All metrics are higher-is-better. SC/PQ denote semantic consistency/perceptual quality. ∗Our DMD2 reproduction. <sup>†</sup>PDD results [1].

![](images/d5e62923d1ed5ba5af2493382242a0fd9d8192f2b1b47fcd7336d48654a2de9d.jpg)  
Figure 2. Few-step image generation with Qwen-Image. The SMD-distilled 4-NFE student generates images with varied compositions, embedded text, and visual styles.

![](images/8c1035e561bfc4cdd31b8ae85e6aafd83c8eba4345c654aac8e83798f8973862.jpg)  
Figure 3. Few-step image generation with FLUX.1-dev. The SMD-distilled 4-NFE student generates animals, plants, architecture, landscapes, and fantasy scenes.

![](images/1ef98d8e3161575741ee1f7c3792420c249eea0a1738ea6ffd59adddf804bc15.jpg)  
Figure 4. Few-step image editing with Qwen-Image-Edit-2511. At 4-NFE, SMD supports object insertion and replacement, text edits, and changes in season, lighting, and appearance.

![](images/3f7db39773fcbeeb2cd837609baa9fcbde7ec22ec98d8ea86048e6af8951c642.jpg)  
Figure 5. Few-step shape generation with Hunyuan3D-2.1. SMD reduces the shape-generation stage to 4-NFE. The untextured renders highlight the generated geometry.

![](images/307e7d542cd19d33e3f8f56c27d64005fccaadcd963da93e54c695c22b55ec7b.jpg)  
Figure 6. Few-step shape and texture generation with TRELLIS.2. SMD distills the high-resolution shape and texture stages to 4-NFE each, producing textured 3D assets.

![](images/3fc80863fcd7bba41ca2ee5e556b3753200a8014994890713ab57b375ce2e6db.jpg)  
Figure 7. Few-step video generation with Wan2.1-T2V-14B. A 4-NFE student distilled by SMD generates animals, landscapes, human activities, and a rocket launch. Frames within each sequence are ordered chronologically from left to right.

![](images/f2eaa568c0365448e57c48c30422d9cbb6495d8caef669d47561152cf2429dab.jpg)  
Figure 8. Few-step joint audio–video generation with MiniMax H3. The SMD-distilled student performs text-to-audio–video generation at 8-NFE. The gallery displays selected video-frame sequences covering character animation, animals, and fantasy scenes.

Main Contents   
1 Introduction 9   
2 Preliminaries 10   
3 Method 11   
3.1 A Simple On-Policy Baseline . 11   
3.2 Imitation from a Proxy Path 13   
3.3 Repulsion from a Reference Prediction . 15   
4 Few-step Distillation 15   
4.1 Image Generation . 16   
4.2 Image Editing 19   
4.3 3D Shape and Texture 20   
4.4 Video Generation 21   
4.5 Joint Audio–Video Generation . 23   
5 Beyond Self-Distillation 25   
5.1 Cross-Model Distillation 25   
5.2 Multiple Experts as Teachers . 26   
5.3 Few-Step Causal Video Generation . 26   
6 Ablation Studies 29   
6.1 Prompt Coverage 29   
6.2 Pivot Design 31   
6.3 Teacher Repulsion . 34   
7 Related Work 36   
8 Conclusion 38

## 1 Introduction

Diffusion and flow models [2, 3, 4, 5] have become a foundation for image, video, and 3D generation [6, 7, 8, 9, 10]. Their exceptional generation quality, however, comes with a heavy inference burden, as the inherent denoising sampling typically requires tens to hundreds of sequential function evaluations. This high computational overhead thus creates a significant bottleneck for real-world deployment and interactive applications, making inference efficiency an urgent priority. Few-step distillation [11, 12, 13, 14, 1] is the way to address this challenge. By compressing a pretrained teacher model, the denoising process could be significantly reduced to a few steps, enabling rapid generation without sacrificing, and sometimes even enhancing the original generation quality [15]. Consequently, few-step distillation now has become an important and indispensable stage to post-train a frontier diffusion foundation model.

Among existing distillation paradigms, on-policy distillation, which trains a few-step student on states generated by the student itself, has emerged as the dominant approach for accelerating diffusion models. Distribution Matching Distillation (DMD) [11] is no doubt standing as a pillar of this paradigm and has inspired a series of follow-up works [15, 16, 17, 18]. By re-noising on-policy student generated samples, DMD approximates a reverse-KL gradient using the score difference between the teacher and the student, where the student score is dynamically tracked by an online fake model [11]. Despite its remarkable progress, DMD and its variants often suffer from low-diversity, unstable training dynamics and over-saturated output. To mitigate these issues, recent trajectory-based methods such as π-Flow [14] and PDD [1] simplify training by directly supervising student rollouts with teacher predictions within each sampling interval. Although this removes the need for online score estimation, the few-step generator must still represent the dense velocity field within each interval, relying on either a learned policy [14] or a parallel decoder [1], which adds architectural components.

We show that simple on-policy velocity matching is already a strong baseline for few-step diffusion distillation. Our approach requires no auxiliary fake score model, architectural modification, or elaborate policy design. Instead, it simply matches the student velocity predictions to the teacher velocities queried at the same states, while keeping the student velocity constant within each sampling interval. This baseline produces strong 4-NFE results, with photorealistic detail and varied outputs.

Finding 1. Direct velocity matching between student and teacher prediction is already a strong baseline for few-step diffusion distillation.

Building on this finding, we further apply this simple baseline to a variety of diffusion teachers [19, 7, 8]. However, when distilling video generation models, we observe that training dynamics can occasionally suffer from instability, as shown in Figure 11. Through the curvature analysis on the velocity field, we observe actually within a coarse integration step, the dense teacher velocities can vary substantially [20], especially for modeling complex and high-dimensional modality like video, hence exposing a single student prediction to varying targets over an entire interval to calculate the imitation loss will make the optimization conflict and jitter. To address this issue, we introduce Pivot proxy, a lightweight, trainingonly residual network that captures local velocity variations via a proxy path. By decoupling local velocity modeling from the overall student prediction, Pivot proxy stabilizes training while incurring no inference overhead. To further correct the student errors, we exploit weaker predictions from the same teacher as informative negative references and introduce a repulsive regularization term, which steers the student away from these inferior predictions without compromising faithful imitation of the teacher target. We term the resulting framework Simplified Matching Distillation (SMD). True to its name, SMD remains entirely regression-based. It requires no modification to the architecture or inference pipeline of the few-step generator, and involves neither score estimation nor adversarial objectives.

Finding 2. Pivot proxy path and repulsive term introduced by SMD further improve the stability and quality of few-step diffusion distillation.

Such simplicity makes SMD both practical and scalable. Trained purely on-policy without any target data, SMD applies to image generation and editing, 3D generation, video synthesis, and joint audio– video generation, scaling to models with up to 33B parameters. With only 4-NFE, SMD distilled FLUX.1-dev [19] and Qwen-Image [7] models outperform all compared few-step baselines on GenEval and DPG-Bench (Figure 1), with the FLUX.1-dev [19] student even surpassing its 50-NFE teacher on both. On Wan2.1-T2V [8], SMD remains competitive with strong 4-NFE baselines at both 1.3B and 14B scales. Beyond acceleration, the framework supports cross-architecture, causal video generation distillation, and multi-teacher distillation, where a single 4-NFE student outperforms its 28-NFE base model on all six reported metrics. These results position SMD as a powerful and versatile tool for post-training diffusion foundation models.

Finding 3. SMD supports post-training diffusion models across modalities and scales.

## 2 Preliminaries

Diffusion and flow matching models. Modern diffusion and flow models can be formulated in terms of a learned velocity field that transports a simple noise distribution to the data distribution [4, 5]. Let $x _ { 0 } \sim p _ { \mathrm { d a t a } } ( x \mid c )$ be a data sample under condition $c ,$ and let $x _ { 1 } \sim \mathcal { N } ( 0 , I )$ be Gaussian noise. A common linear probability path is

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 } , \qquad t \in [ 0 , 1 ] ,\tag{1}
$$

where $t = 0$ is the data endpoint and $t = 1$ is the noise endpoint. The conditional velocity along this path is ${ \dot { x } } _ { t } = x _ { 1 } - x _ { 0 }$ . Flow matching trains a velocity model to predict this quantity from the interpolated state. We write its prediction as $v _ { T } ( x _ { t } , t , c )$ when it serves as the teacher,

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { x _ { 0 } , x _ { 1 } , t } \left[ \Vert v _ { T } ( x _ { t } , t , c ) - ( x _ { 1 } - x _ { 0 } ) \Vert _ { 2 } ^ { 2 } \right] .\tag{2}
$$

At the population optimum, the learned field approaches the marginal velocity $v ^ { * } ( x , t , c ) = \mathbb { E } [ x _ { 1 } - x _ { 0 } \ |$ $x _ { t } = x , c ]$ . This velocity defines the generative ODE

$$
\frac { d x _ { t } } { d t } = v _ { T } ( x _ { t } , t , c ) .\tag{3}
$$

Generation starts from $x _ { 1 } \sim \mathcal { N } ( 0 , I )$ and integrates the ODE from $t \ : = \ : 1$ to $t \ = \ 0 ,$ . Each model evaluation provides only the local velocity at the current state and time. A complete sample is obtained by integrating these predictions over the full time interval. The state $x _ { t }$ may be an image or a learned latent, and c may contain text, images, or other conditioning signals [6, 21]. Some pretrained models natively predict noise or a clean sample rather than velocity. We convert these outputs to velocity under the model’s original parameterization. Throughout this report, $v _ { T }$ is frozen during distillation, $v _ { \theta }$ denotes the student, and $\phi$ denotes the Pivot parameters.

Conditional guidance. Classifier-free guidance combines predictions with and without the condition [22]. In velocity form, a guidance scale w gives

$$
\begin{array} { r } { v _ { T } ^ { ( w ) } ( x _ { t } , t , c ) = v _ { T } ( x _ { t } , t , \mathcal { D } ) + w \left[ v _ { T } ( x _ { t } , t , c ) - v _ { T } ( x _ { t } , t , \mathcal { D } ) \right] . } \end{array}\tag{4}
$$

The conditional prediction is recovered at $w = 1$ , while larger values generally strengthen conditioning at a possible cost to diversity. The weaker prediction need not come from an unconditional branch. Recent guidance methods may instead use a weaker model, a perturbed forward pass, or a reduced set of conditions. In each case, guidance modifies the local velocity used by the sampler without changing the underlying integration procedure.

Sampling schedules. Although the learned velocity field is continuous in time, inference evaluates it on a finite decreasing grid

$$
1 = t _ { 0 } > t _ { 1 } > \cdot \cdot \cdot > t _ { N } = 0 .\tag{5}
$$

The placement of these time points determines which regions of the transport receive more model evaluations. Some pretrained models additionally warp the nominal time before passing it to the network. One common rational shift is

$$
S _ { \gamma } ( t ) = \frac { \gamma t } { 1 + ( \gamma - 1 ) t } , \qquad \gamma > 0 .\tag{6}
$$

We absorb this transformation into the sampling grid and use t for the resulting model time below. Given $\Delta _ { i } = t _ { i } - t _ { i + 1 } > 0 ,$ , a first-order update takes the form

$$
x _ { i + 1 } = x _ { i } - \Delta _ { i } v _ { T } ( x _ { i } , t _ { i } , c ) .\tag{7}
$$

Higher-order solvers evaluate the field at additional states within an interval to reduce discretization error [23, 24]. Each network call is counted as one function evaluation, or NFE, and constitutes the main inference cost. Standard samplers use many evaluations to follow the learned field closely. Few-step sampling retains only a small number of coarse transitions, making the accuracy of each transition increasingly important.

## 3 Method

SMD combines a simple on-policy velocity-regression baseline with two training-only refinements. For each coarse transition of the student rollout, the baseline queries a frozen teacher at intermediate states along the student’s current path and matches a single coarse student velocity to the resulting local predictions. Pivot augments this transition with an endpoint-preserving proxy path whose local velocity can vary within the interval, while reference repulsion complements the target teacher prediction with a weaker prediction from the same teacher. Together, these components define SMD. Both refinements are removed at inference, preserving the student architecture and sampler.

Throughout this section, $v _ { T }$ denotes the frozen target teacher prediction, including its native guidance rule, and $v _ { \theta }$ denotes the student. Conditioning is implicit unless stated otherwise.

## 3.1 A Simple On-Policy Baseline

Let us set aside the complexities of existing distillation methods for a moment and start with a simple idea. We match the student’s velocity to the teacher’s local predictions at states along the student’s own coarse transition. Despite the simplicity of both the state construction and the matching objective, this formulation already enables effective few-step distillation.

Figure 9 depicts the one-step special case from $x _ { 1 }$ to $\hat { x } _ { 0 }$ . We now formulate the same construction for an arbitrary coarse transition, applied independently within every interval of a few-step rollout.

Consider one transition from the current student rollout state $x _ { t }$ to time $t - \Delta ,$ where $\Delta > 0$ . The student makes a single prediction at the transition start,

$$
v _ { \theta } = v _ { \theta } ( x _ { t } , t ) , \qquad x _ { t - \Delta } = x _ { t } - \Delta v _ { \theta } .\tag{8}
$$

For $\tau \sim \mathcal { U } ( 0 , 1 )$ , we construct an intermediate state along this transition,

$$
x _ { \tau } = x _ { t } - \tau \Delta v _ { \theta } , \qquad s = t - \tau \Delta .\tag{9}
$$

We then query the pretrained teacher velocity $v _ { T }$ at $( x _ { \tau } , s )$ . The same student prediction $v _ { \theta }$ is used both to construct $x _ { \tau }$ and to match the teacher’s local velocity. No additional student evaluation is needed at the intermediate state. We optimize the $L _ { 2 }$ distance,

$$
\mathcal { L } = \mathbb { E } _ { \tau } \left[ \| v _ { \theta } - v _ { T } ( x _ { \tau } , s ) \| _ { 2 } \right] .\tag{10}
$$

![](images/d4ca62582f2e241b9dbe01b1d018d73934f9a4d3f9ff2c160fea1351c4b1284f.jpg)  
Figure 9. Simple on-policy distillation. In the one-step illustration, the student predicts one velocity $v _ { \theta } ( x _ { 1 } , 1 )$ and matches it to teacher velocities $v _ { T } ( x _ { \tau } , s )$ queried at intermediate states along the transition to ${ \hat { x } } _ { 0 } .$ Algorithm 1 gives the details.

Algorithm 1. Simple on-policy distillation   
Input: Student $v _ { \theta } ,$ frozen teacher $v _ { T }$ . Grid   
$T = t _ { 0 } > \cdots > t _ { N } = 0 .$   
Output: Rollout loss $\mathcal { L }$ for updating θ.   
1 Sample $x \sim \mathcal { N } ( 0 , I )$   
2 $\mathcal { L }  0$   
3 for $i = 0 , \ldots , N - 1$ do   
4 x stopgrad(x)   
5 $\Delta \gets t _ { i } - t _ { i + 1 }$   
6 $v _ { \theta } \gets v _ { \theta } ( x , t _ { i } )$   
7 Sample one or more query points $\tau \sim \mathcal { U } ( 0 , 1 )$   
8 foreach sampled query point τ do   
9 $s \gets t _ { i } - \tau \Delta$   
10 $x _ { \tau }  x - \tau \Delta v _ { \theta }$   
11 $v _ { T }  v _ { T } ( x _ { \tau } , s )$   
12 $\mathcal { L }  \mathcal { L } + \| v _ { \theta } - v _ { T } \| _ { 2 }$   
13 $x  x - \Delta v _ { \theta }$   
14 return the mean loss over all transitions and query points

![](images/69266bbe997eec6ed2c57e1a8828f6e65b14b158daf36928dee952fbbabd89d5.jpg)  
(a) Imitation from a Proxy Path

![](images/5dcbad6fb28736e0b80fa4e56aea06788fecc870e9da5bc1fc624ddca67ceb0f.jpg)  
(b) Repulsion from a Weak Prediction  
Figure 10. Refining the on-policy baseline. (a) A proxy path allows the local velocity to vary within a coarse transition while preserving its endpoints for a fixed $v _ { \theta } .$ . (b) Repulsion from a weaker prediction $\widetilde { v } _ { T }$ complements imitation of the target teacher prediction $v _ { T }$ . The two refinements are illustrated eseparately for clarity and are used only during training.

Throughout training, teacher outputs are treated as detached targets.

The equations above describe a single transition within a few-step student rollout. During training, we sample different conditions and initial noises, roll out the current student over the few-step grid, and apply the same objective to each transition along the rollout. Algorithm 1 summarizes the rollout loss, with gradients stopped across transitions.

Figure 11 shows fixed samples across training checkpoints for two 4-NFE students. Qwen-Image shows rapid early improvements in visual clarity and scene structure. Wan2.1-T2V-1.3B also improves substantially from initialization, but its samples exhibit pronounced quality fluctuations, with visible degradation followed by recovery across checkpoints. These observations show that local velocity matching alone can yield substantial improvements in few-step generation, although progress is not uniformly consistent across the two models.

![](images/6498238bd9635ea42c7b58665f67ee5e54f52a8a347c63f4ae9df65f3b8b4c5a.jpg)  
Figure 11. Simple distillation already works. Fixed samples across 4-NFE training checkpoints show rapid improvement for Qwen-Image (left) but less consistent progress for Wan2.1-T2V-1.3B (right).

![](images/9368a3822470cc14b875e8d4971da197bf2330581b7d95ccc70b3d7199734ae8.jpg)  
Wan2.1-T2V-1.3B

Finding 4. Direct velocity matching rapidly improves Qwen-Image’s 4-NFE samples and produces substantial gains in Wan2.1-T2V-1.3B, without explicit endpoint supervision.

## 3.2 Imitation from a Proxy Path

To accommodate variations in the teacher velocity within a coarse transition, we introduce a proxy path that shares the endpoints of the student transition while allowing its local velocity to vary, as illustrated in Figure 10(a). We add a time-dependent residual to the coarse student velocity,

$$
v ( \tau ) = v _ { \theta } + r _ { \phi } ( \tau ) ,\tag{11}
$$

subject to

$$
\int _ { 0 } ^ { 1 } r _ { \phi } ( \tau ) d \tau = 0 .\tag{12}
$$

The corresponding proxy state is obtained by integrating this local velocity,

$$
x _ { \tau } = x _ { t } - \Delta \int _ { 0 } ^ { \tau } v ( r ) d r , \qquad s = t - \tau \Delta .\tag{13}
$$

The zero-integral constraint ensures that the proxy path starts at $x _ { t }$ and ends at $x _ { t } - \Delta v _ { \theta }$ . Thus, the residual can model local velocity variations without introducing an additional displacement at the end of the transition. We match local proxy velocities to teacher predictions at the same states,

$$
\mathcal { L } = \mathbb { E } _ { \tau } \left[ \| v ( \tau ) - v _ { T } ( x _ { \tau } , s ) \| _ { 2 } \right] .\tag{14}
$$

The student and the residual predictor are optimized jointly. The residual adjusts both the intermediate query states and the local velocities used for imitation, while the coarse student prediction continues to

Analysis. The results above show that local velocity matching alone can already yield substantial improvements in few-step generation. With teacher initialization, the student starts from a pretrained velocity predictor. The objective then refines this prediction using teacher evaluations at intermediate states along the student’s own coarse transition. In this way, a prediction initially trained to describe local dynamics is adapted to account for the teacher velocities over a finite sampling interval.

Figure 12 illustrates successive training updates in synthetic velocity fields. At each update, we query the teacher at intermediate states on the current student transition and use its detachedt− Δ predictions to update the student velocity. Thex<sub>t−Δ</sub> updated transition then determines the query<sub>x</sub>(2) state for the next teacher evaluation.

When the teacher trajectory has low curvature<sup>t−</sup> <sup>Δ</sup> and its velocity varies moderately over the interval, accurate local teacher predictions provide largely compatible supervision for refining the coarse transition endpoint. This allows the student to improve its sampling behavior without an explicit endpoint target, offering a plausible explanation for the rapid improvement of Qwen-Image in Figure 11. For trajectories with higher curvature, however, the teacher velocity can change substantially in direction within a single transition.

![](images/90277655ca23cac27bef142ba5e188d696afc7f6fb3be3cd69ac7501321cce25.jpg)  
(a) Low curvature

![](images/3ae36cf55c926397de83c4d51f4b3b2267710aba887dcb7487f809b6087ade6f.jpg)  
(b) High curvature  
Figure 12. Curvature shapes on-policy updates. Three updates in low-curvature (top) and highcurvature (bottom) velocity fields. Blue student steps darken across updates. Purple arrows and curves denote teacher queries and trajectories.

The resulting regression targets may favor conflicting updates to the student, making optimization less stable. This offers a possible explanation for the degradation and recovery observed with Wan2.1- T2V-1.3B, consistent with prior observations of large trajectory curvature in its high-noise regime [25]. This analysis suggests room for improvement in both how the student fits the teacher dynamics and how the teacher guides the student toward high-quality generation.

determine the endpoint. At inference, the student directly takes the coarse update in Eq. (8), without evaluating the residual predictor. Setting $r _ { \phi } \equiv 0$ recovers the simple baseline.

Implementation. We represent the velocity residual with a finite basis expansion [26], choosing zero-integral functions to preserve the transition endpoints,

$$
r _ { \phi } ( \tau ) = \sum _ { m = 1 } ^ { P } a _ { m } b _ { m } ( \tau ) , \qquad \int _ { 0 } ^ { 1 } b _ { m } ( \tau ) d \tau = 0 ,\tag{15}
$$

where each $a _ { m }$ has the same shape as the student velocity. Pivot predicts the coefficient vectors once per transition from features of the student’s K-th layer from the end, where K = 1 denotes the final layer. It processes these features with M Transformer blocks and uses a zero-initialized output projection to produce all P coefficient vectors, so training starts from the simple baseline. Under pretrained initialization, the Pivot blocks are copied from the final M student blocks. Late-block sensitivity in Appendix A.2 supports this design. Appendix A.3 shows the architecture.

Pivot supports different bases. A polynomial path parameterization uses

$$
B _ { m } ( \tau ) = \tau ( 1 - \tau ) ( 2 \tau - 1 ) ^ { m - 1 } , \qquad b _ { m } ( \tau ) = B _ { m } ^ { \prime } ( \tau ) .\tag{16}
$$

Since $B _ { m } ( 0 ) = B _ { m } ( 1 ) = 0 .$ , its derivative satisfies the zero-integral constraint. We explore this and other parameterizations in our ablations, with full definitions in Appendix A.1.

## 3.3 Repulsion from a Reference Prediction

We further refine the distillation signal using an additional prediction from the same teacher as a negative reference. We complement imitation of the target prediction with repulsion from this reference, as illustrated in Figure 10(b). We retain $v _ { T } ( x _ { \tau } , s )$ as the imitation target and denote the reference prediction at the same state and time by $\widetilde { v } _ { T } ( x _ { \tau } , s )$ . Its construction depends on the teacher’s econditioning and inference procedure, as described below. We optimize

$$
\begin{array} { r } { \mathcal { L } = \mathbb { E } _ { \tau } \left[ \Vert v ( \tau ) - v _ { T } ( x _ { \tau } , s ) \Vert _ { 2 } - \lambda \Vert v ( \tau ) - \widetilde { v } _ { T } ( x _ { \tau } , s ) \Vert _ { 2 } \right] , \quad \quad 0 \leq \lambda < 1 . } \end{array}\tag{17}
$$

eThe first term matches the target teacher prediction, while the second encourages separation from the reference prediction. Setting $\bar { \lambda } = 0$ recovers imitation alone.

Constructing reference predictions. The repulsive objective does not require a particular construction of $\widetilde { v } _ { T }$ . For teachers using classifier-free guidance, an unconditional or conditional prediction ecan serve as the negative reference, corresponding to reference guidance scales $w = 0$ and $w = 1$ respectively, while $v _ { T }$ remains the guided prediction used for imitation. For embedded-guidance models, we use a lower guidance input instead. For models with guidance already distilled into their weights, we can instead obtain $\widetilde { v } _ { T }$ through an additional forward pass with a layer skipped. These echoices use the same pretrained model and adapt the negative reference to its inference procedure.

Training and inference. We train the student and Pivot jointly on on-policy rollouts using Eq. (17). The proxy path is used only to construct supervision at the sampled query points. The next rollout state follows the student’s coarse update and is detached before the next transition. We average losses over transitions, query points, and batch samples. Optimizer updates occur outside the rollout.

At inference, only the student follows the coarse updates on the same time grid. Neither teacher predictions nor Pivot are evaluated, so the student architecture and number of function evaluations are unchanged. Algorithms 2 and 3 summarize the training loss and student sampler, respectively.

Finding 5. The student can learn the coarse update without fitting every local velocity with a single prediction. Pivot models within-step velocity variations through a zero-integral residual, preserving the student’s transition endpoints.

## 4 Few-step Distillation

We evaluate SMD under a few-step sampling budget across image generation and editing, 3D shape and texture generation, video, and joint audio–video generation. Figures 2–8 give an overview of the distilled students’ outputs. For tasks with quantitative benchmarks, we compare with existing few-step methods and include multi-step teachers as references where available. We evaluate 3D generation and reference-conditioned audio–video generation qualitatively. Appendix B provides training and evaluation details.

```latex
Algorithm 2. On-policy distillation with a proxy path
Input: Student parameters $\theta ,$ Pivot parameters $\phi ,$ frozen teacher prediction rules v<sub>T</sub> and $\widetilde { v } _ { T }$
Input: Time grid $T = t _ { 0 } > t _ { 1 } > \cdot \cdot \cdot \stackrel { \cdot } { > } t _ { N } = 0 ,$ zero-integral bases $\left\{ b _ { m } \right\} _ { m = 1 } ^ { P } ,$ e repulsion weight λ.
Output: Rollout loss $\mathcal { L }$ for updating θ and $\phi .$
1 Sample $x \sim \mathcal { N } ( 0 , I )$
2 $\mathcal { L }  0$
3 for $i = 0 , \ldots , N - 1$ do
4 x  stopgrad(x)
5 $\Delta \gets t _ { i } - t _ { i + 1 }$
6 $v _ { \theta } \gets v _ { \theta } ( x , t _ { i } )$ $/ /$ Retain features from the K-th layer from the end
7 Predict $\{ a _ { m } \} _ { m = 1 } ^ { P }$ with Pivot from the retained features
8 Sample one or more query points $\tau \sim \mathcal { U } ( 0 , 1 )$
9 foreach sampled query point $\tau$ do
10 $s \gets t _ { i } - \tau \Delta$
11 $\begin{array} { r } { r _ { \phi } ( \tau )  \sum _ { m = 1 } ^ { P } a _ { m } b _ { m } ( \tau ) } \end{array}$
12 $v ( \tau )  v _ { \theta } + r _ { \phi } ( \tau )$
13 $\begin{array} { r } { x _ { \tau } \gets x - \Delta \left[ \tau v _ { \theta } + \sum _ { m = 1 } ^ { P } a _ { m } \int _ { 0 } ^ { \tau } b _ { m } ( r ) d r \right] } \end{array}$
14 $v _ { T }  v _ { T } ( x _ { \tau } , s )$
15 $\widetilde { v } _ { T }  \widetilde { v } _ { T } ( x _ { \tau } , s )$
16 $\mathcal { L }  \mathcal { L } + \| v ( \tau ) - v _ { T } \| _ { 2 } - \lambda \| v ( \tau ) - \widetilde { v } _ { T } \| _ { 2 }$
17 $x  x - \Delta v _ { \theta }$ $/ /$ Continue the student rollout
18 return the mean loss over all transitions and query points
```

Algorithm 3. Few-step student inference   
Input: Trained student $v _ { \theta } ,$ time grid $T = t _ { 0 } > \cdot \cdot \cdot > t _ { N } = 0 .$   
Output: Generated sample ${ \hat { x } } _ { 0 } .$   
1 Sample $x \sim \mathcal { N } ( 0 , I )$   
2 for $i \dot { = } 0 , \dots , \dot { N } - 1$ do   
3 $\Delta \gets t _ { i } - t _ { i + 1 }$   
4 $x \gets x - \Delta v _ { \theta } ( x , t _ { i } )$   
5 return ${ \hat { x } } _ { 0 } = x$

## 4.1 Image Generation

We first examine prompt alignment and variation across repeated generations, two aspects of image generation that a few-step student should retain. We distill FLUX.1-dev [19] (12B) and Qwen-Image [7] (20B) into 4-NFE text-to-image models using prompt-only, on-policy training. For FLUX.1-dev, we compare with Hyper-SD [27], SenseFlow [16], and π-Flow [14]. For Qwen-Image, we compare with Lightning [28], TwinFlow [29], π-Flow, and the Euler variant of PDD [1]. We evaluate publicly released baseline checkpoints under matched inference and evaluation settings. For PDD, we report the results from the original paper, as no public checkpoint is available. All few-step models use 4-NFE. GenEval [30] measures compositional alignment and DPG-Bench [31] dense-prompt alignment. OneIG-Bench-EN [32] evaluates alignment, text, diversity, style, and reasoning.

Tables 1 and 2 report the quantitative results. At 4-NFE, SMD obtains the highest GenEval and DPG-Bench scores among the compared few-step methods on both backbones. For FLUX.1-dev, GenEval increases from 0.675 with π-Flow to 0.715 with SMD, and DPG-Bench increases from 84.3 to 84.8. Both exceed the corresponding 50-NFE teacher scores. For Qwen-Image, SMD reaches 0.873 on GenEval and 88.7 on DPG-Bench. SMD and the teacher both have a reported OneIG Overall score of 0.540. The

Hyper-SD  
SenseFlow  
π-Flow  
Ours  
![](images/4d32603328b56af489d3e288954cfb99cc55be9188fa80360646196b64899eb0.jpg)  
Figure 13. Image quality and variation with FLUX.1-dev. Each two-column group shows four 4-NFE samples per prompt for one method, allowing comparison of composition and detail.

Table 1. Text-to-image generation with FLUX.1-dev. NFE denotes the number of network evaluations at inference. Best and second-best results among the few-step methods are indicated by boldface and underlining, respectively. The multi-step Teacher is excluded from this ranking.
<table><tr><td rowspan="2">Method</td><td rowspan="2">NFE</td><td rowspan="2">GenEval↑</td><td rowspan="2">DPG-Bench↑</td><td colspan="6">OneIG-Bench-EN↑</td></tr><tr><td>Overall</td><td>Align.</td><td>Text</td><td>Diversity</td><td>Style</td><td>Reasoning</td></tr><tr><td>Teacher</td><td>50</td><td>0.664</td><td>84.2</td><td>0.430</td><td>0.790</td><td>0.556</td><td>0.239</td><td>0.308</td><td>0.258</td></tr><tr><td>Hyper-SD</td><td>4</td><td>0.667</td><td>83.0</td><td>0.399</td><td>0.786</td><td>0.423</td><td>0.185</td><td>0.361</td><td>0.239</td></tr><tr><td>SenseFlow</td><td>4</td><td>0.643</td><td>82.6</td><td>0.380</td><td>0.774</td><td>0.388</td><td>0.156</td><td>0.348</td><td>0.234</td></tr><tr><td>π-Flow</td><td>4</td><td>0.675</td><td>84.3</td><td>0.419</td><td>0.799</td><td>0.460</td><td>0.229</td><td>0.360</td><td>0.251</td></tr><tr><td>Ours</td><td>4</td><td>0.715</td><td>84.8</td><td>0.420</td><td>0.810</td><td>0.455</td><td>0.209</td><td>0.370</td><td>0.256</td></tr></table>

complete OneIG breakdown also shows stronger Qwen-Image text rendering, with a Text score of 0.944 compared with 0.889 for the teacher and 0.921 for Lightning.

Beyond prompt alignment, SMD retains meaningful variation across repeated generations. Its OneIG Diversity score exceeds the DMD-based SenseFlow baseline on FLUX.1-dev and Lightning on Qwen-Image. Figures 13 and 14 further show variations in viewpoint, pose, and composition across four samples from each method and prompt.

TwinFlow  
Qwen-Image-Lightning  
π-Flow  
Ours  
![](images/40c89efe41576df1e32c206615c399879c7610d919487cd9dc91f692467c236c.jpg)  
Figure 14. Image quality and variation with Qwen-Image. Each two-column group shows four 4-NFE samples per prompt for one method, allowing comparison of pose, viewpoint, and composition.

Table 2. Text-to-image generation with Qwen-Image. NFE counts network evaluations. $5 0 \times 2$ includes both CFG branches. Best and second-best few-step results are bold and underlined. † denotes Euler-variant results from PDD [1], Tables 3 and 6–8. No public checkpoint was available.
<table><tr><td rowspan="2">Method</td><td rowspan="2">NFE</td><td rowspan="2">GenEval↑</td><td rowspan="2">DPG-Bench↑</td><td colspan="6">OneIG-Bench-EN↑</td></tr><tr><td>Overall</td><td>Align.</td><td>Text</td><td>Diversity</td><td>Style</td><td>Reasoning</td></tr><tr><td>Teacher</td><td>50 ×2</td><td>0.874</td><td>88.3</td><td>0.540</td><td>0.880</td><td>0.889</td><td>0.195</td><td>0.428</td><td>0.307</td></tr><tr><td>Lightning</td><td>4</td><td>0.852</td><td>88.3</td><td>0.527</td><td>0.886</td><td>0.921</td><td>0.108</td><td>0.411</td><td>0.310</td></tr><tr><td>TwinFlow</td><td>4</td><td>0.818</td><td>86.0</td><td>0.502</td><td>0.860</td><td>0.872</td><td>0.152</td><td>0.361</td><td>0.263</td></tr><tr><td>π-Flow</td><td>4</td><td>0.850</td><td>88.2</td><td>0.532</td><td>0.878</td><td>0.876</td><td>0.180</td><td>0.430</td><td>0.296</td></tr><tr><td>PDD†</td><td>4</td><td>0.860</td><td>88.5</td><td>0.535</td><td>0.879</td><td>0.897</td><td>0.192</td><td>0.417</td><td>0.292</td></tr><tr><td>Ours</td><td>4</td><td>0.873</td><td>88.7</td><td>0.540</td><td>0.883</td><td>0.944</td><td>0.144</td><td>0.429</td><td>0.300</td></tr></table>

SMD achieves these results with a compact training and inference design. Our Qwen-Image student is trained with a global batch size of 16, compared with the reported batch size of 2048 for PDD [1]. At inference, the student directly predicts each coarse update without π-Flow’s substep integration. Unlike PDD, which learns parallel output heads that can be fused for inference, SMD retains the original student output head and uses Pivot only during training.

![](images/14970633eedbff8af067fa14da06bfbfd010089ecf9d60de86dfb98dffb362e6.jpg)  
Figure 15. Instruction-guided image editing. Each triplet shows the input, Lightning, and ours with Qwen-Image-Edit-2511 at 4-NFE under the same editing instruction.

Table 3. Instruction-based editing with Qwen-Image-Edit-2511. $G _ { \mathrm { S C } } , G _ { \mathrm { P Q } } ,$ , and G<sub>O</sub> denote GPT-4.1- based semantic consistency, perceptual quality, and overall quality on GEdit-Bench-EN. Boldface marks the better 4-NFE student result. The multi-step Teacher is reported as a reference.
<table><tr><td rowspan="2">Method</td><td rowspan="2">NFE</td><td colspan="10">ImgEdit-Bench↑</td><td colspan="3">GEdit-Bench-EN↑</td></tr><tr><td>Overall</td><td>Adust</td><td>Style</td><td>Background</td><td>Extact</td><td>Remove</td><td>Replace</td><td>Ppd</td><td>Hqid</td><td>Action</td><td>GsC</td><td>GPQ</td><td>Go</td></tr><tr><td>Teacher</td><td>40 × 2</td><td>4.51</td><td>4.57</td><td>4.89</td><td>4.36</td><td>4.13</td><td>4.46</td><td>4.70</td><td>4.54</td><td>4.16</td><td>4.81</td><td>8.30</td><td>8.20</td><td>7.88</td></tr><tr><td>Lightning</td><td>4</td><td>4.48</td><td>4.40</td><td>4.76</td><td>4.39</td><td>4.21</td><td>4.54</td><td>4.72</td><td>4.67</td><td>3.87</td><td>4.80</td><td>8.95</td><td>7.74</td><td>8.14</td></tr><tr><td>Ours</td><td>4</td><td>4.51</td><td>4.58</td><td>4.74</td><td>4.37</td><td>3.98</td><td>4.74</td><td>4.76</td><td>4.67</td><td>3.95</td><td>4.80</td><td>8.96</td><td>7.79</td><td>8.19</td></tr></table>

## 4.2 Image Editing

We next extend SMD to instruction-based image editing, where the student must apply the requested modification while preserving unrelated content from the source image. We distill Qwen-Image-Edit-2511 [7] to 4-NFE using the source image and editing instruction as conditions for the on-policy rollout, without requiring target edited images. We compare with the publicly released Lightning checkpoint [33] under matched settings and include the multi-step teacher as a reference. We evaluate operation-level editing performance on ImgEdit-Bench [34], and semantic consistency, perceptual quality, and overall quality on GEdit-Bench-EN [35].

![](images/96e3687a1d6df342c54934da9bc460cd2bc22af93c6c6d547cc8862e4c0cee23.jpg)  
Figure 16. Few-step image-to-shape generation. Hunyuan3D-2.1 outputs are compared under the same input images. Each example shows the input image and two rendered views from ours, the 4-NFE teacher, and the multi-step teacher.

Table 3 shows that SMD raises the ImgEdit-Bench overall score from Lightning’s 4.48 to 4.51, the score also reported for the multi-step teacher. It also improves the three GEdit-Bench metrics, including overall quality from 8.14 to 8.19. The largest gains over Lightning are in removal and adjustment, while Lightning remains stronger on extraction.

Figure 15 provides qualitative comparisons across editing instructions. In the building-to-tree example, Lightning produces a stylized canopy with regular, repeated foliage patterns, while our result has more irregular foliage and blends more naturally with the surrounding vegetation.

## 4.3 3D Shape and Texture

We next examine whether the same distillation formulation applies to different 3D latent representations. Hunyuan3D-2.1 [36] uses a fixed-size set of shape latents. TRELLIS.2 [10] uses spatially structured sparse latents for geometry and PBR appearance. We assess the generated assets qualitatively through rendered views, surface visualizations, and detail crops.

For Hunyuan3D-2.1, we distill the shape-generation model to 4-NFE and compare it with the teacher sampled directly at 4-NFE and with standard multi-step teacher inference. As shown in Figure 16,

4-NFE Teacher

![](images/9dd9d1870fb08b4819d72f27fcd0b7c626c71199c093cf21fb523cd2bbe9e1c0.jpg)  
Figure 17. Few-step shape and texture generation. TRELLIS.2 renders, detail crops, and surface visualizations compare SMD with direct 4-NFE teacher sampling and standard multi-step teacher inference. Only the final high-resolution shape and texture stages are distilled, each to 4-NFE, while the preceding stages remain unchanged.

the student preserves recognizable object structure and surface details, whereas directly running the teacher in four steps often produces fragmented or incomplete geometry.

TRELLIS.2 follows a multi-stage generation pipeline. Its high-resolution cascade first generates a sparse structure and coarse shape, upsamples their spatial support for high-resolution shape generation, and finally synthesizes shape-conditioned textures. We distill the high-resolution shape and texture models independently to 4-NFE, while leaving the preceding sparse-structure and coarse-shape stages unchanged. Thus, the reported 4-NFE applies separately to each of the two distilled stages rather than to the complete asset-generation pipeline.

Figure 17 compares the generated geometry and texture through rendered views and detail crops. Together with Hunyuan3D-2.1, these results show that SMD applies to both fixed-size shape latents and spatially structured sparse representations.

## 4.4 Video Generation

We further evaluate the method on text-to-video generation, where distillation must preserve temporal structure as well as frame quality. We distill Wan2.1-T2V [8] at both 1.3B and 14B scales to 4-NFE. We

![](images/746293ccde7cd6ca04d01c135d3928d2ab58abcc7a15c59bbccad011d3ab9f96.jpg)  
Figure 18. Few-step video generation. Wan2.1-T2V-14B methods are compared under matched prompts, with three chronological frames per video.

Table 4. VBench evaluation of Wan2.1-T2V at 4-NFE. DD, MS, and C denote Dynamic Degree, Motion Smoothness, and Color. Boldface and underlining indicate the best and second-best few-step results within each backbone. ∗Our reproduced DMD2 results. <sup>†</sup>Teacher and PDD results are quoted from PDD [1]. The teacher uses UniPC. PDD uses Euler for 1.3B and Midpoint for 14B.
<table><tr><td></td><td></td><td colspan="5">Wan2.1-T2V-1.3B</td><td colspan="5">Wan2.1-T2V-14B</td></tr><tr><td>Method</td><td>NFE</td><td>Quality</td><td>Semantic</td><td>DD</td><td>MS</td><td>C</td><td>Quality</td><td>Semantic</td><td>DD</td><td>MS</td><td>C</td></tr><tr><td>Teacher†</td><td>50×2</td><td>84.90</td><td>79.22</td><td>一</td><td></td><td>一</td><td>84.56</td><td>81.24</td><td></td><td></td><td>一</td></tr><tr><td>DMD2*</td><td>4</td><td>84.68</td><td>79.39</td><td>45.00</td><td>98.81</td><td>84.63</td><td>84.33</td><td>80.26</td><td>49.72</td><td>98.76</td><td>87.81</td></tr><tr><td>rCM</td><td>4</td><td>84.61</td><td>79.56</td><td>81.94</td><td>97.64</td><td>85.39</td><td>85.19</td><td>81.21</td><td>76.39</td><td>97.81</td><td>82.91</td></tr><tr><td>AnyFlow</td><td>4</td><td>85.40</td><td>81.47</td><td>68.33</td><td>98.29</td><td>87.67</td><td>85.78</td><td>82.10</td><td>73.61</td><td>98.10</td><td>85.79</td></tr><tr><td>PDD†</td><td>4</td><td>85.99</td><td>78.22</td><td>72.50</td><td>99.02</td><td>85.41</td><td>85.69</td><td>80.71</td><td>74.72</td><td>98.52</td><td>86.59</td></tr><tr><td>Ours</td><td>4</td><td>85.64</td><td>78.97</td><td>79.44</td><td>98.66</td><td>89.19</td><td>85.72</td><td>80.52</td><td>83.89</td><td>98.52</td><td>86.18</td></tr></table>

compare with our reproduced DMD2 [15] baseline and publicly released rCM [37] and AnyFlow [38] checkpoints under matched settings, and quote PDD results from the original paper [1]. Evaluation uses VBench [39] on the Self-Forcing prompt set [40].

Table 4 reports the Quality and Semantic aggregates together with selected motion and color-adherence dimensions. Across both Wan model scales, SMD remains competitive with strong 4-NFE baselines while retaining the original student inference architecture. Relative to the reported multi-step teacher

![](images/e64a41a72dd437614b8fd2447f29aea104ba80daede1d246928cbcf895d85c36.jpg)  
Figure 19. Joint audio–video generation at 4-NFE. SMD and LightX2V-DMD v1.1 are compared under matched settings through video frames and full-clip audio spectrograms.

results, SMD improves the Quality aggregate at both scales while remaining close in Semantic. Appendix B.5 reports all 16 VBench dimensions, comprising seven Quality and nine Semantic scores.

Figure 18 provides qualitative comparisons for the 14B models. In the illustrated underwater and desert scenes, our outputs have less pronounced color saturation and contrast than the baselines, with softer lighting transitions. The frames also show changes in subject position and lighting.

## 4.5 Joint Audio–Video Generation

We further extend SMD to MiniMax H3 [41], a 33B Transformer that jointly generates video and audio within a shared forward pass. Unlike image or video generation alone, few-step distillation in this setting must preserve visual dynamics and sound generation simultaneously. We train text-conditioned students at 4-NFE and 8-NFE by applying the same on-policy objective to both modalities. We compare with LightX2V-DMD v1.1 [42] under matched inference budgets on the same 100 evaluation prompts. We evaluate four video and two audio dimensions (Appendix B.7).

As shown in Table 5, SMD performs favorably across both inference budgets. LightX2V-DMD achieves higher prompt alignment, whereas SMD consistently improves temporal coherence, visual naturalness, audio clarity, and audio quality. Structural coherence is comparable at 4-NFE and improves substantially with SMD at 8-NFE. The audio difference is particularly pronounced at 4-NFE. The DMD baseline often produces more diffuse and muddled spectra, while SMD retains cleaner temporal and frequency structure, consistent with its substantially higher audio clarity and quality scores. Figures 19 and 20 provide qualitative comparisons at the two inference budgets.

DMD  
Table 5. Joint audio–video generation with H3 at 4-NFE and 8-NFE on a common evaluation set. All metrics are higher-is-better. Boldface indicates the larger mean within each inference budget.
<table><tr><td rowspan="2">Method</td><td rowspan="2">NFE</td><td colspan="4">Video</td><td colspan="2">Audio</td></tr><tr><td>Prompt</td><td>Structural</td><td>Temporal</td><td>Naturalness</td><td>Clarity Quality</td><td></td></tr><tr><td rowspan="2">LightX2V-DMD v1.1 Ours</td><td>4</td><td>3.797</td><td>3.653</td><td>3.644</td><td>3.556</td><td>1.829</td><td>3.133</td></tr><tr><td>4</td><td>3.609</td><td>3.638</td><td>3.738</td><td>3.638</td><td>3.486</td><td>3.514</td></tr><tr><td rowspan="2">LightX2V-DMD v1.1 Ours</td><td>8</td><td>3.922</td><td>3.625</td><td>3.597</td><td>3.528</td><td>3.209</td><td>3.440</td></tr><tr><td>8</td><td>3.863</td><td>3.863</td><td>3.834</td><td>3.625</td><td>3.619</td><td>3.500</td></tr></table>

![](images/a31e3500574d8bab753b341a79cfd8f40a6d1225b2c1c7e62f6f4c64819670e8.jpg)  
Figure 20. Joint audio–video generation at 8-NFE. Qualitative comparison between SMD and LightX2V-DMD v1.1 under matched prompts, seeds, and timestamps.

Reference-conditioned generation. Beyond text-conditioned generation, SMD can also be applied to reference-conditioned H3. We train with instructions and reference audio–video clips, without requiring ground-truth target outputs. As shown in Figure 21, the resulting 4-NFE student retains a range of reference-conditioned capabilities, including appearance editing, composition, and referenceguided motion and appearance transfer. We assess this setting qualitatively.

Finding 6. At 4-NFE, SMD leads the compared image-generation baselines on GenEval and DPG-Bench. The same objective also improves temporal coherence and audio quality over the DMD baseline in the 33B joint audio–video model.

## 5 Beyond Self-Distillation

Velocity matching does not require identical teacher and student architectures. With compatible prediction representations, the supervision need not come from the student’s own pretrained model. We study cross-model and multi-teacher distillation, then causal video compression.

## 5.1 Cross-Model Distillation

We ask whether a 4-NFE student can benefit from a larger teacher while retaining its own architecture. For a FLUX.2-klein-4B student, we compare FLUX.2-klein-4B and FLUX.2-klein-9B teachers to test cross-capacity transfer. For Lens-RL-3.8B [43], we compare self-distillation with supervision from the same 9B FLUX.2 teacher, testing cross-architecture transfer.

Table 6. Cross-model distillation at 4-NFE. Bold marks the better teacher for each student and metric.
<table><tr><td rowspan="2">Teacher</td><td rowspan="2">Student</td><td rowspan="2">NFE</td><td rowspan="2">GenEval↑</td><td rowspan="2">DPG-Bench↑</td><td colspan="5">OneIG-Bench-EN↑</td></tr><tr><td>Align.</td><td>Text</td><td>Diversity</td><td>Style</td><td>Reasoning</td></tr><tr><td>FLUX.2-klein-4B FLUX.2-klein-9B</td><td>FLUX.2-klein-4B</td><td>4 4</td><td>0.789 0.791</td><td>83.34 84.26</td><td>0.826 0.843</td><td>0.663 0.744</td><td>0.215 0.224</td><td>0.419 0.441</td><td>0.249 0.246</td></tr><tr><td>Lens-RL-3.8B FLUX.2-klein-9B</td><td>Lens-RL-3.8B</td><td>4 4</td><td>0.829 0.770</td><td>87.30 85.13</td><td>0.866 0.834</td><td>0.954 0.888</td><td>0.184 0.234</td><td>0.310 0.414</td><td>0.277 0.247</td></tr></table>

Table 6 shows that the 9B teacher improves the FLUX.2-klein-4B student on six of seven reported metrics, with only a small decrease in reasoning. For Lens, it improves diversity and style but reduces the other reported scores. These results show that cross-model velocity supervision is feasible, with gains that depend on the teacher-student pairing. Figure 22 compares outputs from these teacherstudent pairings.

Reference conditions

![](images/9a0db8144d12450e7a702c9d87e8e7a68ad4605ad598c56600519aa2b9aea610.jpg)  
Ours (4 steps)  
Figure 21. Reference-conditioned generation at 4-NFE. Qualitative results from a single SMD checkpoint across different forms of reference-conditioned generation.

FLUX.2 9B → 4B  
FLUX.2 4B → 4B  
FLUX.2 9B → Lens  
Lens → Lens  
![](images/3172322851abe018db48ddc91770441a77c8bf2c611731cb8942d0ff7279a41a.jpg)  
Figure 22. Changing the teacher changes the student outputs. From left to right, each pair of columns shows FLUX.2-klein-9B 4B, FLUX.2-klein-4B 4B, FLUX.2-klein-9B Lens-RL-3.8B, and Lens-RL-3.8B self-distillation. Each pair contains two 4-NFE generations for the same prompt.

## 5.2 Multiple Experts as Teachers

We next ask whether one student can combine complementary supervision from several sources. We distill four reward-specialized LoRA teachers for CLIPScore, GenEval, OCR, and PickScore into a single 4-NFE SD3.5-Medium student. Self-distillation uses the base teacher, each specialist student uses one fixed reward teacher, and routed multi-teacher distillation selects the teacher associated with each prompt category. All students share the same initialization and use imitation without repulsion.

Table 7 shows that routed multi-teacher distillation improves all six reported metrics over selfdistillation, including HPSv2 and ImageReward, which are not direct specialization objectives, and exceeds the original 28-NFE base teacher on every reported metric. Individual specialists remain strongest on several corresponding criteria, whereas the routed student gives the best OCR score and broadly strong results elsewhere. Figure 23 compares generations from the base teacher and the distilled students. At inference, only the student is evaluated.

## 5.3 Few-Step Causal Video Generation

Fast causal video generation is important when frames must be produced incrementally, including interactive generation and learned world models. Recent forcing pipelines typically convert a bidirectional video diffusion model into a multi-step causal generator and then use DMD-style distribution matching for few-step compression [17, 40, 44]. We isolate this final stage and ask whether on-policy regression can replace DMD. We apply the on-policy regression framework to compress an existing causal Wan2.1-T2V-1.3B generator into a 4-NFE student.

Two teachers with different roles. At each query point on the student’s rollout, a frozen multistep causal teacher supplies the local velocity $v _ { \mathrm { a r } }$ that the student directly matches. Because this

![](images/cfce19110aef65ca61991f96eda8cfb9d64eb219f7b7649533bcd4dc933b2428.jpg)  
Base Teacher  
Self Distill  
CLIPScore Distill  
GenEval Distill  
OCR Distill  
PickScore Distill  
Multi-Teacher Distill

Figure 23. Multiple experts supervise one student. SD3.5-Medium columns show the multi-step base teacher and 4-NFE students trained by self-distillation, four individual reward specialists, or routed multi-teacher supervision. Each row uses a shared prompt. Routing is training-only.

teacher follows the same temporal factorization as the student, its prediction is compatible with causal inference. A frozen bidirectional Wan2.1-T2V-14B teacher observes the complete intermediate video state and supplies a correction from its conditional and unconditional predictions.

Table 7. Reward-specialized distillation on SD3.5-Medium. Teacher denotes the original multi-step model. All students use 4-NFE. Bold and underlined scores rank first and second among students.
<table><tr><td>Setting</td><td>NFE</td><td>GenEval↑</td><td>OCR↑</td><td>PickScore↑</td><td>HPSv2↑</td><td>CLIPScore↑</td><td>ImageReward↑</td></tr><tr><td>Teacher</td><td>28</td><td>0.6279</td><td>0.5813</td><td>21.76</td><td>0.2802</td><td>0.2837</td><td>0.8512</td></tr><tr><td>Self-distill</td><td>4</td><td>0.6153</td><td>0.3019</td><td>21.60</td><td>0.2848</td><td>0.2741</td><td>0.8163</td></tr><tr><td>CLIPScore specialist</td><td>4</td><td>0.7082</td><td>0.4202</td><td>22.22</td><td>0.3055</td><td>0.3011</td><td>1.223</td></tr><tr><td>GenEval specialist</td><td>4</td><td>0.7385</td><td>0.3153</td><td>21.53</td><td>0.2658</td><td>0.2799</td><td>0.9196</td></tr><tr><td>OCR specialist</td><td>4</td><td>0.6410</td><td>0.5944</td><td>21.71</td><td>0.2841</td><td>0.2803</td><td>0.9002</td></tr><tr><td>PickScore specialist</td><td>4</td><td>0.6551</td><td>0.3738</td><td>22.81</td><td>0.3107</td><td>0.2706</td><td>1.120</td></tr><tr><td>Routed multi-teacher</td><td>4</td><td>0.7289</td><td>0.5948</td><td>22.56</td><td>0.2995</td><td>0.2848</td><td>1.141</td></tr></table>

![](images/9771b6a88da5ccf02bad0ad4ed79dcc743219c2122576e0e4a9a897ee57d6c3a.jpg)  
Figure 24. Replacing the DMD compression stage with on-policy regression. Each case shows four chronological frames at 4-NFE for Causal Forcing, DMD∗, and ours.

![](images/f22cb29dc0a15031844fbe9794a361ffe80ae22887c23f717c2cc986c4b9732e.jpg)  
Figure 25. Correction strength changes the global video solution. Four chronological frames are shown for each prompt and correction weight $\lambda _ { \mathrm { c o r r } }$ $\lambda _ { \mathrm { c o r r } } = 0$ is pure imitation. The effect on composition, appearance, and temporal evolution is prompt-dependent and non-monotonic.

$$
\Delta { v } _ { \mathrm { b i } } = \mathrm { s g } \Bigl ( v _ { \mathrm { b i } } ^ { \mathrm { u n c o n d } } - v _ { \mathrm { b i } } ^ { \mathrm { c o n d } } \Bigr ) , \qquad \widetilde { x } _ { q } = \mathrm { s g } \bigl ( x _ { q } + \Delta { v } _ { \mathrm { b i } } \bigr ) , \qquad \mathscr { L } _ { \mathrm { c o r r } } = \frac { 1 } { 2 } \| x _ { q } - \widetilde { x } _ { q } \| _ { 2 } ^ { 2 } .\tag{18}
$$

The pseudo-target contains the detached student state, so the MSE injects the prescribed correction direction rather than fitting a separately generated video. The causal teacher provides the imitation target and the bidirectional teacher supplies global correction. Neither is used at inference.

Comparison with distribution matching. We compare against the released Causal Forcing result and DMD∗, our parameter-efficient control with rank-256 LoRA for both the causal generator and learned fake-score model. ∗ denotes a baseline trained by us. Our student uses correction weight $\lambda _ { \mathrm { c o r r } } \doteq 0 . 2$ without reference repulsion. In the top panel of Table 8, it improves Quality and Total over $\mathrm { D M D ^ { * } }$ and comes within 0.09 of Causal Forcing in Total, without training a fakescore model. Figure 24 shows the corresponding video comparisons.

Role of bidirectional correction. The bottom panel of Table 8 isolates the correction signal by fixing the causal-teacher guidance and disabling reference repulsion while varying only $\lambda _ { \mathrm { c o r r } }$ . A moderate weight of 0.2 gives the best Semantic score in the sweep, but the highest Total is obtained without correction and larger weights do not improve monotonically. Figure 25 shows prompt-

Table 8. Causal video generation. The top panel compares with Causal Forcing [44]. The bottom panel varies the correction weight. Bold marks the best in each panel.

Main comparison
<table><tr><td>Method</td><td>Quality</td><td>Semantic</td><td>Total</td></tr><tr><td>Causal Forcing</td><td>84.59</td><td>81.84</td><td>84.04</td></tr><tr><td>DMD*</td><td>84.15</td><td>80.82</td><td>83.48</td></tr><tr><td>Ours</td><td>84.97</td><td>79.86</td><td>83.95</td></tr></table>

Correction sweep
<table><tr><td> $\lambda _ { \mathrm { c o r r } }$ </td><td>Quality</td><td>Semantic</td><td>Total</td></tr><tr><td>0</td><td>85.36</td><td>79.64</td><td>84.22</td></tr><tr><td>0.2</td><td>84.97</td><td>79.86</td><td>83.95</td></tr><tr><td>0.4</td><td>84.68</td><td>79.41</td><td>83.63</td></tr><tr><td>0.8</td><td>84.88</td><td>79.53</td><td>83.81</td></tr></table>

dependent changes in global appearance as the correction weight varies.

Finding 7. On-policy regression supports both cross-model transfer and the consolidation of specialized teachers. With routed supervision, a single 4-NFE student surpasses the multi-step base teacher on all six reported metrics.

## 6 Ablation Studies

We ablate SMD on Qwen-Image and Wan2.1-T2V-1.3B, focusing on training prompt coverage, Pivot design, and teacher repulsion. Training and evaluation details are provided in Appendix D.

## 6.1 Prompt Coverage

We vary the training prompt category and count when distilling Qwen-Image with SMD. Each student uses between one and 30K training prompts from Text, Portrait, Objects, or Nature. A Mixed setting combines all four 30K sets. Table 9 reports the complete comparison.

Prompt category and count. With 30K training prompts, Text-focused training gives the highest Overall among the four categories, with the largest cross-category differences appearing in text generation. Figure 26 compares outputs across training prompt categories. Increasing the training set from one to 300 prompts substantially improves the Text scores of Portrait, Objects, and Nature students. One prompt gives the Text-trained student an Overall of 0.538, versus 0.541 at 30K.

Table 9. Qwen-Image training prompt categories and counts. Bold marks each category’s best result.
<table><tr><td>Category</td><td>Prompts</td><td>Overall</td><td>Align.</td><td>Text</td><td>Diversity</td><td>Style</td><td>Reasoning</td></tr><tr><td>Text</td><td>1</td><td>0.538</td><td>0.838</td><td>0.939</td><td>0.265</td><td>0.370</td><td>0.278</td></tr><tr><td>Text</td><td>300</td><td>0.541</td><td>0.878</td><td>0.966</td><td>0.134</td><td>0.420</td><td>0.305</td></tr><tr><td>Text</td><td>3K</td><td>0.540</td><td>0.881</td><td>0.964</td><td>0.129</td><td>0.418</td><td>0.307</td></tr><tr><td>Text</td><td>30K</td><td>0.541</td><td>0.876</td><td>0.961</td><td>0.150</td><td>0.421</td><td>0.299</td></tr><tr><td>Portrait</td><td>1</td><td>0.484</td><td>0.847</td><td>0.610</td><td>0.310</td><td>0.411</td><td>0.239</td></tr><tr><td>Portrait</td><td>300</td><td>0.540</td><td>0.886</td><td>0.926</td><td>0.155</td><td>0.442</td><td>0.290</td></tr><tr><td>Portrait</td><td>3K</td><td>0.525</td><td>0.883</td><td>0.881</td><td>0.142</td><td>0.428</td><td>0.292</td></tr><tr><td>Portrait</td><td>30K</td><td>0.537</td><td>0.889</td><td>0.903</td><td>0.158</td><td>0.439</td><td>0.295</td></tr><tr><td>Objects</td><td>1</td><td>0.493</td><td>0.854</td><td>0.636</td><td>0.303</td><td>0.431</td><td>0.242</td></tr><tr><td>Objects</td><td>300</td><td>0.534</td><td>0.887</td><td>0.912</td><td>0.134</td><td>0.437</td><td>0.301</td></tr><tr><td>Objects</td><td>3K</td><td>0.533</td><td>0.889</td><td>0.903</td><td>0.145</td><td>0.434</td><td>0.295</td></tr><tr><td>Objects</td><td>30K</td><td>0.532</td><td>0.889</td><td>0.907</td><td>0.137</td><td>0.429</td><td>0.297</td></tr><tr><td>Nature</td><td>1</td><td>0.418</td><td>0.849</td><td>0.407</td><td>0.253</td><td>0.344</td><td>0.237</td></tr><tr><td>Nature</td><td>300</td><td>0.513</td><td>0.883</td><td>0.804</td><td>0.160</td><td>0.426</td><td>0.291</td></tr><tr><td>Nature</td><td>3K</td><td>0.519</td><td>0.886</td><td>0.852</td><td>0.137</td><td>0.424</td><td>0.298</td></tr><tr><td>Nature</td><td>30K</td><td>0.517</td><td>0.884</td><td>0.830</td><td>0.159</td><td>0.429</td><td>0.283</td></tr><tr><td>Mixed</td><td>120K</td><td>0.538</td><td>0.885</td><td>0.938</td><td>0.138</td><td>0.432</td><td>0.300</td></tr></table>

![](images/ed955b22033d782ba40a51f491d3893bfccb1723c53b198ef60613fa36a65857.jpg)  
Figure 26. Category-specific training changes student outputs. Qwen-Image students use four 30Kprompt categories or their 120K mixture. Each cell shows four samples for the same test prompt.

Generation beyond the training prompt. The students in Figure 27 still produce content relevant to unseen object and portrait prompts, despite having been distilled on only one prompt. Single-prompt Wan2.1-T2V-1.3B students also respond to unseen prompts but can retain training-specific visual characteristics. The desert-bloom student’s dot-like patterns persist in Figure 28.

![](images/1147572476ced48b1639ce4e4629f13e44e217d906e16b86f79fea22e24275d7.jpg)  
Text

![](images/ff64e8b05224e6fa8e8fa18f2e44bf1fcc007392278cf38e5801f50e1ff445a5.jpg)  
Portrait

![](images/3f114447b6673185fff6e74ab42d1d217245c30bb81b1b811a1074d40a9da369.jpg)  
Objects

![](images/c5af232f36fc6b0b41bc55242534879aa9df0b16d9767f246919e8a2d01b2177.jpg)  
Nature

![](images/70293cf33bed907f88564b4fca55a606e9c008143e3bf32033a3cb1af11b0a60.jpg)  
Base 4-step

Figure 27. Generalization beyond a single training prompt. Each Qwen-Image student sees only one training prompt but responds to unseen prompts. Base 4-step shows direct 4-NFE sampling of the pretrained model with its native settings.  
![](images/ca2e233aadadd1ff2727eeee58a7370abe009ff043bae06c848aac7940f11972.jpg)  
Teacher reference

![](images/711c0c1c141e3690424310da46782ee7130d3289014a0228f39511f97148495a.jpg)  
Seen Prompt

![](images/b89f43db78d432be168003e6be94219fc4b35ce8dcfab06621bc63fe399016a3.jpg)  
Park Walk

![](images/15ad1f9ac6e968f0a4a6ed99f442e0b96d2ab82d13934a5ab32fe7692137e902.jpg)  
Cycling

![](images/ea86b5f5158bae084e6fea2974f3f486ade0de4c445c4df8c82e181dde3a4252.jpg)  
Athlete Practice

![](images/eee5508c03e9c39a22e0e6954bc7d3716c7e2043dd49a0588b3af904d1bba605.jpg)  
Tropical Golf

Figure 28. Visual characteristics can transfer across video prompts. Rows show single-prompt SMD students on Wan2.1-T2V-1.3B. The teacher column provides a visual comparison for each training prompt. Other columns show student outputs for seen and unseen prompts.

Finding 8. Surprisingly, SMD can work even with a single training prompt. Both image and video students still respond to unseen prompts in our qualitative comparisons.

## 6.2 Pivot Design

We compare SMD with and without the Pivot residual, then vary Pivot’s initialization, the student layer supplying its features, its block count, and its residual basis. The notation follows Section 3.2.

Pivot and initialization. Table 10 shows that adding the Pivot residual improves Qwen-Image Overall and Wan2.1-T2V-1.3B Total with either pretrained or random initialization at K = 4, M = 3. The gain therefore does not depend on copying pretrained blocks.

Table 10. Pivot design ablations in SMD. Bold marks the best result within each group.  
Qwen-Image
<table><tr><td>Setting</td><td>Overall</td><td>Align.</td><td>Text</td><td>Diversity</td><td>Style</td><td>Reasoning</td></tr><tr><td colspan="7">Pivot and initialization,  $K = 4 , M = 3$ </td></tr><tr><td>w/o Pivot residual</td><td>0.535</td><td>0.886</td><td>0.937</td><td>0.127</td><td>0.424</td><td>0.300</td></tr><tr><td>Pretrained</td><td>0.538</td><td>0.885</td><td>0.938</td><td>0.138</td><td>0.432</td><td>0.300</td></tr><tr><td>Random</td><td>0.541</td><td>0.887</td><td>0.944</td><td>0.135</td><td>0.438</td><td>0.303</td></tr><tr><td colspan="7">Varying K, random initialization, M = 3</td></tr><tr><td>K=1</td><td>0.540</td><td>0.889</td><td>0.943</td><td>0.130</td><td>0.433</td><td>0.305</td></tr><tr><td>K = 4</td><td>0.541</td><td>0.887</td><td>0.944</td><td>0.135</td><td>0.438</td><td>0.303</td></tr><tr><td colspan="7">Varying K, random initialization, M = 7</td></tr><tr><td>K=1</td><td>0.534</td><td>0.886</td><td>0.945</td><td>0.108</td><td>0.430</td><td>0.302</td></tr><tr><td>K = 8</td><td>0.547</td><td>0.891</td><td>0.967</td><td>0.132</td><td>0.434</td><td>0.310</td></tr></table>

Wan2.1-T2V-1.3B

Pivot and initialization
<table><tr><td>Setting</td><td>Quality</td><td>Semantic</td><td>Total</td></tr><tr><td>w/o Pivot residual</td><td>84.83</td><td>76.16</td><td>83.09</td></tr><tr><td>Pretrained</td><td>85.61</td><td>78.29</td><td>84.14</td></tr><tr><td>Random</td><td>85.30</td><td>79.28</td><td>84.10</td></tr></table>

Varying K, random, M = 3
<table><tr><td>K</td><td>Quality</td><td>Semantic</td><td>Total</td></tr><tr><td>1</td><td>84.80</td><td>77.34</td><td>83.31</td></tr><tr><td>4</td><td>85.30</td><td>79.28</td><td>84.10</td></tr><tr><td>8</td><td>85.83</td><td>76.53</td><td>83.97</td></tr></table>

![](images/dbff36ae3a84a73b012d70eac9546df99c70d4bdd9e2b2f16ccfb25986448c63.jpg)

![](images/6e4b92e4d271d3afd40220255ea873e7e7f4809b7c7d75f492f7be85ddc80ef7.jpg)

![](images/4a88ce805103faead1a2d3ea796e42652be3ff4b1b00a39d86317c4091c5490b.jpg)

![](images/523037ab4eb7a398f3229600ef3b4d6883588e8993021e937059179a7bb38902.jpg)

![](images/0bd9f74b0feed0746adff37062d9a6fc7a55f2bc3e7b43293e5771b6f21ef953.jpg)  
K = 1

M = 3  
![](images/d58482d46d29d63b2af203dee16ecc4b036fab2ccba793356ae6cb493396fbb5.jpg)  
K = 4

![](images/cf68130cdaeb08b8e1c31c5544f03354ea5fed784ec4f468c8ed9770221c95b2.jpg)  
K = 1

![](images/1aa8164332fb18041d66daeb4b6c02f2a68bddddd775ec5222b87daccb9d3685.jpg)  
K = 8  
M = 7  
Figure 29. Changing the student layer used by Pivot. Qwen-Image students use K = 1, 4 at $M = 3$ and K = 1, 8 at M = 7. Each cell contains four samples of the same test prompt.

Varying K and M. For Qwen-Image at $M = 7 ,$ using K = 8 rather than K = 1 improves all five OneIG dimensions. For Wan2.1-T2V-1.3B at $M = 3 , K = 4$ gives the highest Total (Table 10). For Qwen-Image at K = 1, increasing Pivot from three to seven blocks instead lowers Overall from 0.540

![](images/1c7383f9003b8bfe232f6eb888bab2a3b6e4709bc254e67cc6d50b0801483e27.jpg)  
Figure 30. Varying the number of Pivot blocks. Qwen-Image students use M = 3, 7 at K = 1. Within each column, the two settings use the same prompt and four seeds.

to 0.534. In some qualitative examples, larger K also produces more similar compositions across seeds. Figures 29–31 illustrate these comparisons.  
![](images/827993842ab8c316d067ad6a84a3bacce773d425a4f503a06a8a4bae18a82068.jpg)  
M = 3  
Figure 31. Changing the student layer used by Pivot. Wan2.1-T2V-1.3B students use K = 1, 4, 8 with three Pivot blocks. Each cell contains four samples.

Residual basis. With the other Pivot settings fixed, Piecewise (P = 32) gives the best Qwen-Image Overall, whereas Polynomial $( P = 4 )$ gives the best Wan2.1-T2V-1.3B Total (Table 11). Increasing Piecewise from 4 to 32 elements improves Qwen-Image but not Wan2.1-T2V-1.3B.

Table 11. Residual basis family and size. The remaining Pivot settings are fixed within each model.
<table><tr><td></td><td></td><td colspan="6">Qwen-Image</td><td colspan="3">Wan2.1-T2V-1.3B</td></tr><tr><td>Basis</td><td>P</td><td>Overall</td><td>Align.</td><td>Text</td><td>Diversity</td><td>Style</td><td>Reasoning</td><td>Quality</td><td>Semantic</td><td>Total</td></tr><tr><td>Polynomial</td><td>4</td><td>0.538</td><td>0.885</td><td>0.938</td><td>0.138</td><td>0.432</td><td>0.300</td><td>85.61</td><td>78.29</td><td>84.14</td></tr><tr><td>Spectral</td><td>4</td><td>0.538</td><td>0.887</td><td>0.943</td><td>0.129</td><td>0.427</td><td>0.307</td><td>84.91</td><td>77.76</td><td>83.48</td></tr><tr><td>Collocation</td><td>4</td><td>0.533</td><td>0.881</td><td>0.928</td><td>0.143</td><td>0.422</td><td>0.291</td><td>84.79</td><td>79.73</td><td>83.78</td></tr><tr><td>Piecewise</td><td>4</td><td>0.540</td><td>0.884</td><td>0.944</td><td>0.145</td><td>0.429</td><td>0.300</td><td>85.20</td><td>78.56</td><td>83.87</td></tr><tr><td>Piecewise</td><td>32</td><td>0.543</td><td>0.889</td><td>0.962</td><td>0.132</td><td>0.426</td><td>0.306</td><td>84.59</td><td>78.31</td><td>83.33</td></tr></table>

Finding 9. Different residual bases give relatively similar aggregate scores. Larger K can improve prompt alignment, but some qualitative examples show less varied compositions.

## 6.3 Teacher Repulsion

We vary the reference CFG scale w and repulsion weight λ while keeping the target teacher prediction and Pivot fixed. The comparison includes unconditional (w = 0) and conditional (w = 1) references, as well as SMD without repulsion (λ = 0).

Table 12. Reference CFG scale and repulsion weight in SMD.  
Qwen-Image
<table><tr><td></td><td>w/o repulsion</td><td colspan="2">w = 0</td><td colspan="2">w = 1</td></tr><tr><td>Metric</td><td>λ = 0</td><td>λ = 0.1</td><td> $\lambda = 0 . 2$ </td><td>λ = 0.1</td><td>λ = 0.2</td></tr><tr><td>Overall</td><td>0.539</td><td>0.539</td><td>0.542</td><td>0.528</td><td>0.538</td></tr><tr><td>Align.</td><td>0.883</td><td>0.888</td><td>0.886</td><td>0.883</td><td>0.885</td></tr><tr><td>Text</td><td>0.943</td><td>0.960</td><td>0.958</td><td>0.899</td><td>0.938</td></tr><tr><td>Diversity</td><td>0.142</td><td>0.123</td><td>0.139</td><td>0.138</td><td>0.138</td></tr><tr><td>Style</td><td>0.434</td><td>0.429</td><td>0.421</td><td>0.425</td><td>0.432</td></tr><tr><td>Reasoning</td><td>0.295</td><td>0.298</td><td>0.304</td><td>0.294</td><td>0.300</td></tr></table>

Wan2.1-T2V-1.3B
<table><tr><td></td><td>w/o repulsion</td><td colspan="3">w = 0</td><td colspan="3">w = 1</td></tr><tr><td>Metric</td><td>λ = 0</td><td> $\lambda = 0 . 1$ </td><td> $\lambda = 0 . 2$ </td><td> $\lambda = 0 . 4$ </td><td>λ = 0.1</td><td>λ = 0.2</td><td>λ = 0.4</td></tr><tr><td>Quality</td><td>84.97</td><td>85.61</td><td>85.83</td><td>85.24</td><td>84.05</td><td>84.66</td><td>84.92</td></tr><tr><td>Semantic</td><td>77.62</td><td>78.29</td><td>77.59</td><td>77.59</td><td>77.07</td><td>75.59</td><td>78.21</td></tr><tr><td>Total</td><td>83.50</td><td>84.14</td><td>84.18</td><td>83.71</td><td>82.65</td><td>82.85</td><td>83.57</td></tr></table>

An unconditional reference with λ = 0.2 improves over SMD without repulsion and gives the best Qwen-Image Overall and Wan2.1-T2V-1.3B Total in these sweeps (Table 12). The conditional reference does not give the same benefit, and increasing the unconditional weight to 0.4 reduces Wan2.1-T2V-1.3B Total. Figures 32 and 33 show the corresponding appearance changes.

λ = 0.1  
λ = 0.0  
λ = 0.0  
λ = 0.2  
![](images/aecbd8e2a8cf10298af7f68ff096a094003e4b95cc9a244b153d00cd6dc254b1.jpg)  
Figure 32. Reference and weight ablations in SMD. Qwen-Image rows vary w and columns vary λ. The boxed λ = 0 cells repeat the same samples without repulsion.

λ = 0.0  
λ = 0.1  
λ = 0.2  
λ = 0.4  
λ = 0.0  
λ = 0.1  
λ = 0.2  
λ = 0.4  
![](images/d33ac4566ceaa0909cab29dbb9eab1830c792b07c640006c1931a28e1c008b6e.jpg)  
Figure 33. Teacher-repulsion ablation on Wan2.1-T2V-1.3B. Wan2.1-T2V-1.3B uses the same layout, with three nonzero weights per reference.  
Finding 10. The teacher’s unconditional prediction provides an effective weak reference for repulsion on both tested backbones. Moderate repulsion improves the results, but increasing its weight further can weaken the benefit.

## 7 Related Work

Few-step generation from diffusion and flow-based models has attracted broad research interest. Existing approaches largely fall into trajectory-level and distribution-level paradigms. Beyond these two main lines, we survey methods that bridge or extend both paradigms, along with other complementary acceleration techniques. Finally, we discuss the emerging connections between few-step distillation and reinforcement learning for diffusion models.

Trajectory-Level Distillation. Trajectory-level methods train a student to reproduce the teacher’s sampling trajectory in fewer steps. The earliest approach, progressive distillation [12], repeatedly halves the step count by training students to match teacher predictions over merged intervals. Guidance distillation [45] later folded classifier-free guidance into the student, removing the need for dual forward passes at inference. Both can degrade at very few steps, motivating subsequent objectives.

Consistency models [13] introduced a self-consistency constraint requiring that any point on the same probability-flow ODE trajectory map to the same clean output, allowing both distillation from a pretrained teacher and standalone training. Follow-up work improved stability via better schedules and loss weighting [46], enabled efficient fine-tuning from pretrained diffusion models [47], moved the formulation to latent spaces [48, 49, 50], generalized it to multi-step generation [51, 52, 53, 27, 54, 55], and developed continuous-time variants that remove discretization dependence [56, 57, 37]. Targetdriven timestep selection [58] further improves consistency distillation by optimizing which timestep pairs to train on. A concurrent report [59] offers practical guidance on consistency distillation.

Flow-map methods establish the mathematical foundations connecting flow maps to stochastic interpolants [60, 61], learning solution operators that transport states between arbitrary time pairs rather than mapping only to the clean endpoint. Consistency trajectory models [62] first parameterized the full ODE solution. MeanFlow [63] derived a self-consistency identity for the time-averaged velocity that enables one-step generation. Extensions cover objective decompositions [64], JVP-free formulations [65, 66], continuous-time unification with flow matching [67, 68, 69], data-free training [70, 71], and recursive formulations that support any-step generation from a single model [72, 73]. Another line of trajectory-level distillation follows rectified flow [5], where methods such as InstaFlow [74] and PeRFlow [75] straighten ODE trajectories for fewer-step integration. OPSD-V [76] uses real long videos as privileged temporal context to construct cleaner teacher caches, providing dense denoising-level supervision along the student’s on-policy rollout for post-training few-step autoregressive video generators.

π-Flow and PDD are particularly close to our regression-based formulation [14, 1]. π-Flow predicts a network-free policy that supplies time-varying velocities for integration over substeps within one network evaluation. PDD predicts multiple denoising updates in parallel, and its output heads can be fused for inference. SMD instead places within-interval residual capacity in a training-only branch whose basis functions integrate to zero. The deployed student keeps its original head and requires neither Pivot nor substep integration.

Distribution-Level Distillation. Distribution-level methods directly compare the student’s generated distribution with the teacher or data distribution. Their training objectives can draw on discriminator feedback, score estimates, or sample statistics.

Adversarial distillation draws on the generative adversarial network [77, 78, 79, 80] paradigm, using learned discriminators to provide distribution-level feedback. ADD [81] combined score distillation with adversarial losses for few-step generation, LADD [82] moved discrimination to latent space using generative features from pretrained diffusion models, and progressive adversarial schemes [83] interpolate between trajectory and adversarial objectives across distillation stages. Further developments include GAN-diffusion hybrids [84, 85], post-training adversarial fine-tuning for video [86, 87, 88], adversarial distillation for 3D generation [89], intermediate-state adversarial matching [90], and controllability-preserving distillation [91, 92].

Score-based distribution matching emerged as an alternative that avoids explicit discriminator training. Score distillation sampling [93] first used a pretrained diffusion model’s score to optimize parametric representations for text-to-3D generation. Variational score distillation [94] refined this by introducing a learned approximation to the variational distribution’s score, reducing over-smoothing. Diff-Instruct [95] generalized this mathematical framework to training arbitrary generators from pretrained diffusion models, deriving the integral KL objective and training an auxiliary score network on student samples. Distribution Matching Distillation (DMD) [11] scaled this approach to text-to-image synthesis, and DMD2 [15] improved optimization stability and multi-step extensions. SwiftBrush [96, 97] demonstrated that pure variational score distillation, without adversarial training, suffices for one-step synthesis that surpasses the teacher.

Extensions include decoupling CFG internalization from distribution matching [98], separating signals across noise levels [99, 100], hierarchical structure-to-detail decomposition [101], removing the auxiliary network via flow-map surrogates [102], scaling to flow architectures [16, 18], self-adversarial flow formulations [29], video adaptation through transition and score-gradient matching [25, 103], adaptive matching [104], diversity-preserving objectives [105], and teacher-feature drifting [106].

Alternatives to KL include Fisher divergence via score identity distillation [107, 108], implicit matching [109], f-divergence weighting [110], proper scoring rules [111], EM distillation [112], and moment matching approaches [113, 114]. A recent line of work applies DMD-style streaming distillation to convert bidirectional video diffusion models into fast autoregressive generators [17, 44, 115, 40, 116].

Score-regularized continuous-time consistency [37] combines consistency training with a lightweight DMD-style correction. Trajectory distribution matching [117] evaluates divergences along intermediate states. Video-specific designs balance coverage and sharpness [118, 38, 119, 120]. Scale-wise distillation [121] decomposes generation by spatial frequency.

Other Distillation and Acceleration Techniques. Other approaches connect few-step generation with step-size conditioning, adaptive trajectories, and self-training. Shortcut models [122] learn stepsize-conditioned predictions that interpolate between local velocity estimation and global one-step generation. Instance-aware adaptive trajectories [123] learn per-sample flow paths, while ArcFlow [124] treats policy-based generation as an imitation learning problem. Other work studies training one-step diffusion models without teacher distillation [125] and accelerating diffusion models with a single sample [126].

Complementary acceleration techniques further reduce inference cost. Sampling methods and schedules improve efficiency through fast scheduling [127], analytic variance estimation [128], pseudonumerical methods [129], exponential integrators [130], unified predictor-corrector frameworks [131], and optimized timestep placement [132, 133]. Caching exploits temporal redundancy. DeepCache [134] skips redundant U-Net computations, T-GATE [135] decomposes and caches attention outputs at scheduled timesteps, and token-level caching [136], spectral-evolution policies [137], and sensitivitybased strategies [138] target transformer architectures. For video, pruning with adaptive routing [139] and spectral progressive generation [140] exploit inter-frame redundancy. FreeU [141] re-weights skip connections without retraining, while FreeInit [142] refines initial noise via low-frequency iteration. Architectural acceleration through sparse attention [143] and sliding-tile attention [144] exploits spatiotemporal locality.

Connections to Reinforcement Learning. Preference alignment and reinforcement learning have become important components in the training of modern large generative models. Recently, many works have applied RL methods such as GRPO [145] and DPO [146] to diffusion models through the reverse process [147, 148, 149, 150] and the forward process [151, 152, 153]. Human preference models [154, 155, 156, 157, 158] provide the reward signals in these settings.

A growing body of work combines few-step distillation with diffusion RL. Unified frameworks augment distribution matching with non-differentiable rewards [159, 160, 161, 162], and the distribution matching objective itself can be reconceptualized as a reward signal for distilled generators [163].

## 8 Conclusion

Strong few-step diffusion distillation need not rely on auxiliary score models, adversarial training, or specialized inference architectures. SMD builds on simple on-policy velocity regression with a training-only Pivot proxy for local velocity variation and repulsion from weaker teacher predictions. Across diverse generative tasks and models with up to 33B parameters, SMD delivers strong few-step performance while leaving student inference unchanged. At 4-NFE, it outperforms strong few-step baselines and surpasses multi-step teachers on several benchmarks. Our experiments further extend SMD to cross-model, multi-teacher, and causal-video distillation, demonstrating the applicability of on-policy regression beyond self-distillation.

## References

[1] Neta Shaul, Chao Liu, Arash Vahdat, and Julius Berner. Parallel decoding distillation for fast image and video generation. arXiv preprint arXiv:2607.26004, 2026.

[2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.

[3] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.

[4] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023.

[5] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023.

[6] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.

[7] Qwen-Image Team. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

[8] Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[9] Tencent Hunyuan3D Team. Hunyuan3d 2.0: Scaling diffusion models for high resolution

textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025.

[10] Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, and Jiaolong Yang. Native and compact structured latents for 3d generation. In CVPR, 2026.

[11] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024.

[12] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022.

[13] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023.

[14] Hansheng Chen, Kai Zhang, Hao Tan, Leonidas Guibas, Gordon Wetzstein, and Sai Bi. pi-flow: Policy-based few-step generation via imitation distillation. In ICLR, 2026.

[15] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024.

[16] Xingtong Ge, Xin Zhang, Tongda Xu, Yi Zhang, Xinjie Zhang, Yan Wang, and Jun Zhang.

Senseflow: Scaling distribution matching for flow-based text-to-image distillation. In ICLR, 2026.

[17] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025.

[18] Tao Liu, Hao Yan, Mengting Chen, Taihang Hu, Zhengrong Yue, Zihao Pan, Jinsong Lan, Xiaoyong Zhu, Ming-Ming Cheng, Bo Zheng, and Yaxing Wang. Continuous-time distribution matching for few-step diffusion distillation. arXiv preprint arXiv:2605.06376, 2026.

[19] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

[20] Sangyun Lee, Beomsu Kim, and Jong Chul Ye. Minimizing trajectory curvature of ode-based generative models. In International Conference on Machine Learning, pages 18957–18973. PMLR, 2023.

[21] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.

[22] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS Workshop, 2021.

[23] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021.

[24] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022.

[25] Weili Nie, Julius Berner, Nanye Ma, Chao Liu, Saining Xie, and Arash Vahdat. Transition matching distillation for fast video generation. In CVPR, 2026.

[26] J. O. Ramsay and B. W. Silverman. Functional Data Analysis. Springer, 2 edition, 2005. doi: 10.1007/b98888. URL https: //link.springer.com/book/10.1007/b98888.

[27] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, XING WANG, and Xuefeng Xiao. Hyper-SD: Trajectory segmented consistency model for efficient image synthesis. In NeurIPS, 2024.

[28] ModelTC Team. Qwen-image-lightning. https: //github.com/ModelTC/Qwen-Image-Lightning, 2025.

[29] Zhenglin Cheng, Peng Sun, Jianguo Li, and Tao Lin. Twinflow: Realizing one-step generation on large models with self-adversarial flows. In ICLR, 2026.

[30] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. In NeurIPS, 2023.

[31] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

[32] Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. arXiv preprint arXiv:2506.07977, 2025.

[33] LightX2V Team. Qwen-image-edit-2511-lightning. https://huggingface.co/lightx2v/ Qwen-Image-Edit-2511-Lightning, 2026.

[34] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.

[35] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

[36] Hunyuan3D Team, Shuhui Yang, Mingxin Yang, Yifei Feng, Xin Huang, Sheng Zhang, Zebin He, Di Luo, Haolin Liu, Yunfei Zhao, et al. Hunyuan3d 2.1: From images to high-fidelity 3d assets with production-ready pbr material. arXiv preprint arXiv:2506.15442, 2025.

[37] Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. In ICLR, 2026.

[38] Yuchao Gu, Guian Fang, Yuxin Jiang, Weijia Mao, Song Han, Han Cai, and Mike Zheng Shou. Anyflow: Any-step video diffusion model with on-policy flow map distillation. arXiv preprint arXiv:2605.13724, 2026.

[39] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin

Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In CVPR, 2024.

[40] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. In NeurIPS, 2025.

[41] MiniMax AI. MiniMax H3. https: //huggingface.co/MiniMaxAI/MiniMax-H3, 2026.

[42] LightX2V and ModelTC. Minimax-H3-Turbo. https: //github.com/ModelTC/Minimax-H3-Turbo, 2026.

[43] Lens Team. Lens: Rethinking training efficiency for foundational text-to-image models. arXiv preprint arXiv:2605.21573, 2026.

[44] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. In ICML, 2026.

[45] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik P. Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In CVPR, 2023.

[46] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In ICLR, 2024.

[47] Zhengyang Geng, Ashwini Pokle, William Luo, Justin Lin, and J Zico Kolter. Consistency models made easy. arXiv preprint arXiv:2406.14548, 2024.

[48] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

[49] Quan Dao, Khanh Doan, Di Liu, Trung Le, and Dimitris Metaxas. Improved training technique for latent consistency models. In ICLR, 2025.

[50] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolinário Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023.

[51] Jonathan Heek, Emiel Hoogeboom, and Tim Salimans. Multistep consistency models. arXiv preprint arXiv:2403.06807, 2024.

[52] Fu-Yun Wang, Zhaoyang Huang, Alexander William Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun,

Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency model. In NeurIPS, 2024.

[53] Sangyun Lee, Yilun Xu, Tomas Geffner, Giulia Fanti, Karsten Kreis, Arash Vahdat, and Weili Nie. Truncated consistency models. In ICLR, 2025.

[54] Zhengyao Lv, Chenyang Si, Tianlin Pan, Zhaoxi Chen, Kwan-Yee K. Wong, Yu Qiao, and Ziwei Liu. Dual-expert consistency model for efficient and high-quality video generation. arXiv preprint arXiv:2506.03123, 2025.

[55] Jiachen Li, Weixi Feng, Tsu-Jui Fu, Xinyi Wang, S Basu, Wenhu Chen, and William Yang Wang. T2v-turbo: Breaking the quality bottleneck of video consistency model with mixed reward feedback. In NeurIPS, 2024.

[56] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. In ICLR, 2025.

[57] Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, and Enze Xie. Sana-sprint: One-step diffusion with continuous-time consistency distillation. arXiv preprint arXiv:2503.09641, 2025.

[58] Cunzheng Wang, Ziyuan Guo, Yuxuan Duan, Huaxia Li, Nemo Chen, Xu Tang, and Yao Hu. Target-driven distillation: Consistency distillation with target timestep selection and decoupled guidance. arXiv preprint arXiv:2409.01347, 2024.

[59] Yifan Pu, Yizeng Han, Zhiwei Tang, Jiasheng Tang, Fan Wang, Bohan Zhuang, and Gao Huang. Few-step distillation for text-to-image generation: A practical guide. arXiv preprint arXiv:2512.13006, 2025.

[60] Nicholas Matthew Boffi, Michael Samuel Albergo, and Eric Vanden-Eijnden. Flow map matching with stochastic interpolants: A mathematical framework for consistency models. TMLR, 2025.

[61] Zhangkai Wu, Xuhui Fan, Hongyu Wu, and Longbing Cao. Scot: Unifying consistency models and rectified flows via straight-consistent trajectories. arXiv preprint arXiv:2502.16972, 2025.

[62] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ODE trajectory of diffusion. In ICLR, 2024.

[63] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. In NeurIPS, 2025.

[64] Yi Guo, Wei Wang, Zhihang Yuan, Rong Cao, Kuan Chen, Zhengyang Chen, Yuanyuan Huo, Yang Zhang, Yuping Wang, Shouda Liu, and Yuxuan Wang. Splitmeanflow: Interval splitting consistency in few-step generative modeling. arXiv preprint arXiv:2507.16884, 2025.

[65] Haochen You, Baojing Liu, and Hongyang He. Modular meanflow: Towards stable and scalable one-step generative modeling. In PRCV, 2025.

[66] Kyungmin Lee, Sihyun Yu, and Jinwoo Shin. Decoupled meanflow: Turning flow models into flow maps for accelerated sampling. In ICLR, 2026.

[67] Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your flow: Scaling continuous-time flow map distillation. In NeurIPS, 2025.

[68] Huijie Zhang, Aliaksandr Siarohin, Willi Menapace, Michael Vasilkovsky, Sergey Tulyakov, Qing Qu, and Ivan Skorokhodov. Alphaflow: Understanding and improving meanflow models. In ICLR, 2026.

[69] Tianze Luo, Haotian Yuan, and Zhuang Liu. Soflow: Solution flow models for one-step generative modeling. In ICLR, 2026.

[70] Shangyuan Tong, Nanye Ma, Saining Xie, and Tommi Jaakkola. Flow map distillation without data. In CVPR, 2026.

[71] Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Lingjie Liu, and Joshua M Susskind. Boot: Data-free distillation of denoising diffusion models with bootstrapping. In ICML Workshop, 2023.

[72] Peng Sun and Tao Lin. Any-step generation via n-th order recursive consistent velocity field estimation. In ICLR, 2026.

[73] Yansong Peng, Kai Zhu, Yu Liu, Pingyu Wu, Hebei Li, Xiaoyan Sun, and Feng Wu. Facm: Flow-anchored consistency models. arXiv preprint arXiv:2507.03738, 2025.

[74] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and Qiang Liu. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In ICLR, 2024.

[75] Hanshu Yan, Xingchao Liu, Jiachun Pan, Jun Hao Liew, Qiang Liu, and Jiashi Feng. Perflow: Piecewise rectified flow as universal plug-and-play accelerator. In NeurIPS, 2024.

[76] Hongyu Liu, Chun Wang, Feng Gao, Xuanhua He, Yue Ma, Ziyu Wan, Yong Zhang, Xiaoming Wei, and Qifeng Chen. Opsd-v: On-policy self-distillation for post-training few-step autoregressive video generators. arXiv preprint arXiv:2607.08766, 2026.

[77] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS, 2014.

[78] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019.

[79] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, 2020.

[80] Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. In NeurIPS, 2021.

[81] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In ECCV, 2024.

[82] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia, 2024.

[83] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024.

[84] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-to-image generation via diffusion gans. In CVPR, 2024.

[85] Clement Chadebec, Onur Tasar, Eyal Benaroche, and Benjamin Aubin. Flash diffusion: Accelerating any conditional diffusion model for few steps image generation. In AAAI, 2025.

[86] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. In ICML, 2025.

[87] Xiaofeng Mao, Zhengkai Jiang, Fu-yun Wang, Jiangning Zhang, Hao Chen, Mingmin Chi, Yabiao Wang, and Wenhan Luo. Osv: One step is enough for high-quality image to video generation. In CVPR, 2025.

[88] Jiaxiang Cheng, Bing Ma, Xuhua Ren, Hongyi Henry Jin, Kai Yu, Peng Zhang, Wenyue Li, Yuan Zhou, Tianxiang Zheng, and Qinglin Lu. Phased one-step adversarial equilibrium for video diffusion models. arXiv preprint arXiv:2508.21019, 2025.

[89] Ziyu Wan, Despoina Paschalidou, Ian Huang, Hongyu Liu, Bokui Shen, Xiaoyu Xiang, Jing Liao, and Leonidas Guibas. Cad: Photorealistic 3d generation via adversarial distillation. In CVPR, 2024.

[90] Yanzuo Lu, Yuxi Ren, Xin Xia, Shanchuan Lin, Xing Wang, Xuefeng Xiao, Andy J. Ma, Xiaohua Xie, and Jian-Huang Lai. Adversarial distribution matching for diffusion distillation towards efficient image and video synthesis. In ICCV, 2025.

[91] Yihong Luo, Tianyang Hu, Yifan Song, Jiacheng Sun, Zhenguo Li, and Jing Tang. Adding additional control to one-step diffusion with joint distribution matching. In ICCV, 2025.

[92] Yara Bahram, Mélodie Desbos, Mohammadhadi Shateri, and Eric Granger. Uni-dad: Unified distillation and adaptation of diffusion models for few-step few-shot image generation. In CVPR, 2026.

[93] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In ICLR, 2023.

[94] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In NeurIPS, 2023.

[95] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. In NeurIPS, 2023.

[96] Thuan Hoang Nguyen and Anh Tran. Swiftbrush: One-step text-to-image diffusion model with variational score distillation. In CVPR, 2024.

[97] Trung Dao, Thuan Hoang Nguyen, Thanh Le, Duc Vu, Khoi Nguyen, Cuong Pham, and Anh Tran. Swiftbrush v2: Make your one-step diffusion model better than its teacher. In ECCV, 2024.

[98] Dongyang Liu, Peng Gao, David Liu, Ruoyi Du, Zhen Li, Qilong Wu, Xin Jin, Sihan Cao, Shifeng Zhang, Hongsheng Li, et al. Decoupled dmd: Cfg augmentation as the spear, distribution matching as the shield. In ICLR, 2026.

[99] Guanjie Chen, Shirui Huang, Yifu Sun, Kai Liu, Jianchen Zhu, Xiaoye Qu, Yu Cheng, and Peng Chen. Flash-dmd: Towards high-fidelity few-step image generation with efficient distillation and joint reinforcement learning. In CVPR, 2026.

[100] Xiangyu Fan, Zesong Qiu, Zhuguanyu Wu, Fanzhou Wang, Zhiqian Lin, Tianxiang Ren, Dahua Lin, Ruihao Gong, and Lei Yang. Phased dmd: Few-step distribution matching distillation via score matching within subintervals. arXiv preprint arXiv:2510.27684, 2026.

[101] Hanbo Cheng, Peng Wang, Kaixiang Lei, Qi Li, Zhen Zou, Pengfei Hu, and Jun Du. From structure to detail: Hierarchical distillation for efficient diffusion model. arXiv preprint arXiv:2511.08930, 2025.

[102] Youngjoong Kim, Deokyeong Lee, and Jaesik Park. Distribution matching distillation without fake score network. arXiv preprint arXiv:2605.19256, 2026.

[103] Zhuguanyu Wu, Ruihao Gong, Yang Yong, Yushi Huang, Xiangyu Fan, Lei Yang, Dahua Lin, and Xianglong Liu. Sgmd: Score gradient matching distillation for few-step video diffusion distillation. In ICML, 2026.

[104] Lichen Bai, Zikai Zhou, Shitong Shao, Wenliang Zhong, Shuo Yang, Shuo Chen, Bojun Chen, and Zeke Xie. Optimizing few-step generation with adaptive matching distillation. arXiv preprint arXiv:2602.07345, 2026.

[105] Tianhe Wu, Ruibin Li, Lei Zhang, and Kede Ma. Diversity-preserved distribution matching distillation for fast visual synthesis. In ICML, 2026.

[106] Yuan Zhang, Chenyi Li, Guoqing Ma, Jiajun Zha, Yuanming Yang, Bo Wang, Wei Tang, Wenbo Li, Haoyang Huang, and Nan Duan. Teacher-feature drifting: One-step diffusion distillation with pretrained diffusion representations. arXiv preprint arXiv:2605.07327, 2026.

[107] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: exponentially fast distillation of pretrained diffusion models for one-step generation. In ICML, 2024.

[108] Mingyuan Zhou, Yi Gu, and Zhendong Wang. Few-step diffusion via score identity distillation. arXiv preprint arXiv:2505.12674, 2025.

[109] Weijian Luo, Zemin Huang, Zhengyang Geng, J Zico Kolter, and Guo-jun Qi. One-step

diffusion distillation through score implicit matching. In NeurIPS, 2024.

[110] Yilun Xu, Weili Nie, and Arash Vahdat. One-step diffusion models with f-divergence distribution matching. arXiv preprint arXiv:2502.15681, 2025.

[111] Huiyang Shao, Xin Xia, Yuxi Ren, XING WANG, and Xuefeng Xiao. Varflow: Proper scoring-rule diffusion distillation via energy matching. In NeurIPS, 2025.

[112] Sirui Xie, Zhisheng Xiao, Diederik P. Kingma, Tingbo Hou, Ying Nian Wu, Kevin Murphy, Tim Salimans, Ben Poole, and Ruiqi Gao. Em distillation for one-step diffusion models. In NeurIPS, 2024.

[113] Linqi Zhou, Stefano Ermon, and Jiaming Song. Inductive moment matching. In ICML, 2025.

[114] Tim Salimans, Thomas Mensink, Jonathan Heek, and Emiel Hoogeboom. Multistep distillation of diffusion models via moment matching. In NeurIPS, 2024.

[115] Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive diffusion distillation for real-time interactive video generation. arXiv preprint arXiv:2605.15141, 2026.

[116] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, Yujun Shen, and Min Zhang. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. In CVPR, 2026.

[117] Yihong Luo, Tianyang Hu, Jiacheng Sun, Yujun Cai, and Jing Tang. Learning few-step diffusion models by trajectory distribution matching. In ICCV, 2025.

[118] Shengqu Cai, Weili Nie, Chao Liu, Julius Berner, Lvmin Zhang, Nanye Ma, Hansheng Chen, Maneesh Agrawala, Leonidas Guibas, Gordon Wetzstein, and Arash Vahdat. Mode seeking meets mean seeking for fast long video generation. In ICML, 2026.

[119] Shitong Shao, Hongwei Yi, Hanzhong Guo, Tian Ye, Daquan Zhou, Michael Lingelbach, Zhiqiang Xu, and Zeke Xie. Magicdistillation: Weak-to-strong video distillation for large-scale few-step synthesis. arXiv preprint arXiv:2503.13319, 2025.

[120] Siyi Chen, Shaowei Liu, Yixuan Jia, Zian Wang, Huan Ling, Qing Qu, and Jun Gao. Data-forcing distillation: Restoring diversity and fidelity in few-step video generation. arXiv preprint arXiv:2606.18478, 2026.

[121] Nikita Starodubcev, Ilya Drobyshevskiy, Denis Kuznedelev, Artem Babenko, and Dmitry Baranchuk. Scale-wise distillation of diffusion models. In ICLR, 2026.

[122] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. In ICLR, 2025.

[123] Huiyang Shao, Xin Xia, Yuhong Yang, Yuxi Ren, Xing Wang, and Xuefeng Xiao. Rayflow: Instance-aware diffusion acceleration via adaptive flow trajectories. In CVPR, 2025.

[124] Zihan Yang, Shuyuan Tu, Licheng Zhang, Qi Dai, Yu-Gang Jiang, and Zuxuan Wu. Arcflow: Unleashing 2-step text-to-image generation via high-precision non-linear flow distillation. arXiv preprint arXiv:2602.09014, 2026.

[125] Mingtian Zhang, Jiajun He, Wenlin Chen, Zijing Ou, José Miguel Hernández-Lobato, Bernhard Schölkopf, and David Barber. Towards training one-step diffusion models without distillation. In ICLR Workshop, 2025.

[126] Zhuobai Dong, Rui Zhao, Songjie Wu, Junchao Yi, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Alex Jinpeng Wang. Glance: Accelerating diffusion models with 1 sample. arXiv preprint arXiv:2512.02899, 2025.

[127] Zhifeng Kong and Wei Ping. On fast sampling of diffusion probabilistic models. In ICML Workshop, 2021.

[128] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In ICLR, 2022.

[129] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In ICLR, 2022.

[130] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS, 2022.

[131] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. In NeurIPS, 2023.

[132] Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your steps: Optimizing sampling schedules in diffusion models. In ICML, 2024.

[133] Yong-Hyun Park, Chieh-Hsin Lai, Satoshi Hayakawa, Yuhta Takida, and Yuki Mitsufuji. Jump your steps: Optimizing sampling schedule of discrete diffusion models. In ICLR, 2025.

[134] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In CVPR, 2024.

[135] Wentian Zhang, Haozhe Liu, Jinheng Xie, Francesco Faccio, Mike Zheng Shou, and Jürgen Schmidhuber. Cross-attention makes inference cumbersome in text-to-image diffusion models. arXiv preprint arXiv:2404.02747, 2024.

[136] Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. Accelerating diffusion transformers with token-wise feature caching. arXiv preprint arXiv:2410.05317, 2024.

[137] Jiwoo Chung, Sangeek Hyun, MinKyu Lee, Byeongju Han, Geonho Cha, Dongyoon Wee, Youngjun Hong, and Jae-Pil Heo. Seacache: Spectral-evolution-aware cache for accelerating diffusion models. In CVPR, 2026.

[138] Yasaman Haghighi and Alexandre Alahi. Sencache: Accelerating diffusion model inference via sensitivity-aware caching. In CVPR, 2026.

[139] Yutong Wang, Yunke Wang, Tianfan Xue, Yu Qiao, Yaohui Wang, Xinyuan Chen, and Chang Xu. Pare: Pruning and adaptive routing for efficient video generation. arXiv preprint arXiv:2605.27336, 2026.

[140] Howard Xiao, Brian Chao, Lior Yariv, and Gordon Wetzstein. Spectral progressive diffusion for efficient image and video generation. arXiv preprint arXiv:2605.18736, 2026.

[141] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. In CVPR, 2024.

[142] Tianxing Wu, Chenyang Si, Yuming Jiang, Ziqi Huang, and Ziwei Liu. Freeinit: Bridging initialization gap in video diffusion models. In ECCV, 2024.

[143] Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric Xing, and Hao Zhang. Vsa: Faster video diffusion with trainable sparse attention. arXiv preprint arXiv:2505.13389, 2025.

[144] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025.

[145] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[146] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2024.

[147] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. In NeurIPS, 2025.

[148] Yihong Luo, Tianyang Hu, and Jing Tang. Reinforcing diffusion models by direct group preference optimization. In ICLR, 2026.

[149] Zhen Fang, Wenxuan Huang, Yu Zeng, Yiming Zhao, Shuang Chen, Kaituo Feng, Yunlong Lin, Lin Chen, Zehui Chen, Shaosheng Cao, et al. Flow-opd: On-policy distillation for flow matching models. arXiv preprint arXiv:2605.08063, 2026.

[150] Quanhao Li, Junqiu Yu, Kaixun Jiang, Yujie Wei, Zhen Xing, Pandeng Li, Ruihang Chu, Shiwei Zhang, Yu Liu, and Zuxuan Wu. Diffusionopd: A unified perspective of on-policy distillation in diffusion models. arXiv preprint arXiv:2605.15055, 2026.

[151] Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, and Zhi-Ming Ma. Advantage weighted matching: Aligning rl with pretraining in diffusion models. arXiv preprint arXiv:2509.25050, 2025.

[152] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. In ICLR, 2026.

[153] Andreas Bergmeister, Stefanie Jegelka, Nikolas Nüsken, Carles Domingo-Enrich, and Jakiw Pidstrigach. Reinforce adjoint matching. arXiv preprint arXiv:2605.10759, 2026.

[154] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and

Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. In NeurIPS, 2023.

[155] Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, Jiayan Teng, Zhuoyi Yang, Wendi Zheng, Xiao Liu, Ming Ding, Xiaohan Zhang, Xiaotao Gu, Shiyu Huang, Minlie Huang, Jie Tang, and Yuxiao Dong. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.

[156] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025.

[157] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation, 2023.

[158] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

[159] Yihong Luo, Tianyang Hu, Weijian Luo, and Jing Tang. Tdm-r1: Reinforcing few-step diffusion models with non-differentiable reward. In ICML, 2026.

[160] Dengyang Jiang, Dongyang Liu, Zanyi Wang, Qilong Wu, Liuzhuozheng Li, Hengzhuang Li, Xin Jin, David Liu, Zhen Li, Bo Zhang, et al. Distribution matching distillation meets reinforcement learning. arXiv preprint arXiv:2511.13649, 2025.

[161] Xu Wang, Zexian Li, Litong Gong, Tiezheng Ge, and Zhijie Deng. Advdmd: Adversarial reward meets dmd for high-quality few-step generation. arXiv preprint arXiv:2604.28126, 2026.

[162] Yushi Huang, Xiangxin Zhou, Ruoyu Wang, Chi Zhang, Jun Zhang, and Tianyu Pang. Reinforcing few-step generators via reward-tilted distribution matching. arXiv preprint arXiv:2605.26108, 2026.

[163] Linqian Fan, Peiqin Sun, Tiancheng Wen, Shun Lu, and Chengru Song. r<sub>dm</sub>: Re-conceptualizing distribution matching as a reward for diffusion distillation. arXiv preprint arXiv:2603.28460, 2026.

[164] Wenhao Wang and Yi Yang. VidProM: A million-scale real prompt-gallery dataset for text-to-video diffusion models. In Advances in Neural Information Processing Systems, 2024. URL https: //openreview.net/forum?id=pYNl76onJL.

[165] Zhucun Xue, Jiangning Zhang, Teng Hu, Haoyang He, Yinan Chen, Yuxuan Cai, Yabiao Wang, Chengjie Wang, Yong Liu, Xiangtai Li, and Dacheng Tao. UltraVideo: High-quality UHD video dataset with comprehensive captions. arXiv preprint arXiv:2506.13691, 2025.

[166] Soham Deshmukh, Dareen Alharthi, Benjamin Elizalde, Hannes Gamper, Mahmoud Al Ismail, Rita Singh, Bhiksha Raj, and Huaming Wang. PAM: Prompting audio-language models for audio quality assessment. arXiv preprint arXiv:2402.00282, 2024.

[167] Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, Carleigh Wood, Ann Lee, and Wei-Ning Hsu. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound. arXiv preprint arXiv:2502.05139, 2025.

## Appendix Contents

A Supplementary Analysis and Mathematical Details 47   
A.1 Residual Parameterizations 47   
A.2 Block-Swap Study 49   
A.3 Training Architecture 50   
B Few-step Implementation and Evaluation Details 51   
B.1 Image Generation 51   
B.2 Image Editing 51   
B.3 3D Shape and Texture 52   
B.4 Video Generation 53   
B.5 Detailed VBench Results 53   
B.6 Joint Audio–Video Generation 54   
B.7 H3 Evaluation 55   
C Beyond Self-Distillation Implementation and Evaluation Details 56   
C.1 Cross-Model Distillation 56   
C.2 Multiple Experts as Teachers 56   
C.3 Few-Step Causal Video Generation 57   
D Ablation Implementation and Evaluation Details 57   
E Visual Evaluation System Prompt 58

## A Supplementary Analysis and Mathematical Details

## A.1 Residual Parameterizations

The four parameterizations compared in Table 11 give Pivot different ways to bend a training trajectory while preserving the coarse transition used at inference.

From a velocity residual to a path correction. Recall that $\tau \in [ 0 , 1 ]$ measures progress through a segment of duration ∆. Pivot predicts P coefficient vectors $a _ { m } ,$ each with the same shape as the student velocity, and holds them fixed within the segment. The basis functions control how these coefficients contribute along the interval.

$$
r _ { \phi } ( \tau ) = \sum _ { m = 1 } ^ { P } a _ { m } b _ { m } ( \tau ) , \qquad q _ { \phi } ( \tau ) = \sum _ { m = 1 } ^ { P } a _ { m } I _ { m } ( \tau ) , \qquad I _ { m } ( \tau ) = \int _ { 0 } ^ { \tau } b _ { m } ( u ) d u .\tag{19}
$$

The residual $r _ { \phi }$ modifies velocity, and its integral $q _ { \phi }$ corrects the straight-line path.

$$
v ( \tau ) = v _ { \theta } + r _ { \phi } ( \tau ) , \qquad x _ { \tau } = x _ { t } - \Delta \bigl [ \tau v _ { \theta } + q _ { \phi } ( \tau ) \bigr ] .\tag{20}
$$

Thus the actual state displacement relative to the straight line is $- \Delta q _ { \phi } ( \tau )$ . Requiring every $b _ { m }$ to have zero integral gives

$$
\int _ { 0 } ^ { 1 } b _ { m } ( \tau ) d \tau = 0 \quad \Longrightarrow \quad q _ { \phi } ( 0 ) = q _ { \phi } ( 1 ) = 0 \quad \Longrightarrow \quad x _ { \tau = 1 } = x _ { t } - \Delta v _ { \theta } .\tag{21}
$$

The residual can therefore change intermediate states without moving either endpoint for a fixed $v _ { \theta }$ . The constraint concerns the integral, so the residual need not vanish at the endpoints. Figure 34 compares how the four constructions fit the same challenging path correction as the coefficient budget increases. The controlled fitting protocol is described below.

Polynomial basis. We can preserve endpoints using functions that vanish at both ends.

$$
B _ { m } ( \tau ) = \tau ( 1 - \tau ) ( 2 \tau - 1 ) ^ { m - 1 } , \qquad b _ { m } ( \tau ) = \frac { d B _ { m } ( \tau ) } { d \tau } , \quad 1 \leq m \leq P .\tag{22}
$$

The factor $\tau ( 1 - \tau )$ anchors the path correction to zero at both endpoints. Successive powers of $2 \tau - 1$ add polynomial variation within the interval. Since $B _ { m } ( 0 ) = 0 ,$ , the integrated basis is simply $I _ { m } = B _ { m }$ and $\begin{array} { r } { q _ { \phi } = \sum _ { m } a _ { m } B _ { m } } \end{array}$ . Differentiating this path gives $r _ { \phi } ,$ , with zero integral because $B _ { m } ( 1 ) \stackrel { \cdot } { - } \bar { B } _ { m } ( 0 ) = 0$ For example, the first mode has $B _ { 1 } ( \tau ) = \bar { \tau } ( 1 - \top )$ and $b _ { 1 } ( \tau ) = 1 - 2 \tau$ . Its velocity contribution changes sign halfway through the segment, producing a single path excursion that returns to zero. With P coefficients, this construction spans $\bar { P }$ independent polynomial modes.

Spectral basis. Instead of constructing the path first, we can directly choose velocities whose positive and negative contributions cancel.

$$
b _ { m } ( \tau ) = \cos ( m \pi \tau ) , \qquad I _ { m } ( \tau ) = \frac { \sin ( m \pi \tau ) } { m \pi } , \quad 1 \leq m \leq P .\tag{23}
$$

Each $b _ { m } ( \tau )$ integrates to zero on $[ 0 , 1 ]$ . Low indices describe broad changes across the segment, while higher indices allow more rapid oscillations. Integration reduces the amplitude of mode m by $1 / ( m \pi )$ so equal velocity coefficients do not produce equal path-displacement amplitudes. The constant mode $m = 0$ is excluded because it would change the coarse endpoint. The remaining $P$ cosine modes are mutually orthogonal and independent.

![](images/c51805037c572b7a81ad2f3481a5824f8c6eb2b77a0cb7efb5993317be1e5be7.jpg)  
Figure 34. Endpoint-preserving path approximation. Least-squares fits to the square-wave target in Eq. (28), using $\bar { P } \in \{ \bar { 4 } , 1 2 , 3 2 \}$ coefficients. All fitted paths preserve both endpoints.

Centered nodal interpolation. The remaining two designs predict values $a _ { m }$ at $P \geq 2$ nodes $\tau _ { m }$ . Let $\ell _ { m } ( \tau )$ be interpolation functions satisfying $\ell _ { m } \bar { ( \tau _ { k } ) } = \delta _ { m k }$ . The coefficients are the nodal values of an uncentered velocity interpolant $\begin{array} { r } { \widetilde { r } _ { \phi } ( \tau ) = \mathrm { \ i } \widetilde { \sum _ { m } } a _ { m } \ell _ { m } ( \tau ) } \end{array}$ . To remove its interval mean, define

$$
w _ { m } = \int _ { 0 } ^ { 1 } \ell _ { m } ( \tau ) d \tau , \qquad b _ { m } ( \tau ) = \ell _ { m } ( \tau ) - w _ { m } .\tag{24}
$$

The resulting residual is $\begin{array} { r } { r _ { \phi } ( \tau ) = \widetilde { r } _ { \phi } ( \tau ) - \mu _ { \phi } . } \end{array}$ , where $\begin{array} { r } { \mu _ { \phi } = \sum _ { m } w _ { m } a _ { m } } \end{array}$ is the mean of $\widetilde { r } _ { \phi }$ over the unit interval. Every $b _ { m }$ eintegrates to zero because $\begin{array} { r } { \int _ { 0 } ^ { 1 } ( \ell _ { m } - w _ { m } ) = w _ { m } - w _ { m } = 0 } \end{array}$ e. Its path contribution is

$$
I _ { m } ( \tau ) = \int _ { 0 } ^ { \tau } \ell _ { m } ( u ) d u - \tau w _ { m } .\tag{25}
$$

At node $\tau _ { m } ,$ , the final residual equals $a _ { m } - \mu _ { \phi }$ , rather than $a _ { m }$ . This centering operation makes arbitrary predicted nodal values compatible with the endpoint constraint.

Piecewise basis. We place nodes uniformly at $\tau _ { m } = \left( m - 1 \right) / \left( P - 1 \right)$ and interpolate adjacent values with straight lines. The interpolation functions $\ell _ { m }$ are triangular hat functions (half-hats at the two endpoints). Their areas give the exact integration weights.

$$
w _ { 1 } = w _ { P } = \frac { 1 } { 2 ( P - 1 ) } , \qquad w _ { m } = \frac { 1 } { P - 1 } \quad ( 2 \leq m \leq P - 1 ) .\tag{26}
$$

Centering preserves the piecewise-linear shape of the velocity residual, whose integral is a continuously differentiable, piecewise-quadratic path correction. Adding nodes allows finer variation within the segment. Before centering, changing one nodal value affects only its adjacent intervals. After centering, it also changes the constant mean subtracted over the entire segment. Thus the centered basis itself does not have strictly local support.

Collocation basis. Gauss–Lobatto nodes include both endpoints and, for $P > 3 ,$ are spaced nonuniformly with higher density near the ends. We use the global Lagrange interpolation functions

$$
\ell _ { m } ( \tau ) = L _ { m } ( \tau ) = \prod _ { k \neq m } { \frac { \tau - \tau _ { k } } { \tau _ { m } - \tau _ { k } } } .\tag{27}
$$

Here $\begin{array} { r } { w _ { m } = \int _ { 0 } ^ { 1 } L _ { m } ( \tau ) } \end{array}$ dτ are the corresponding Gauss–Lobatto quadrature weights on $[ 0 , 1 ]$ . Subtracting $w _ { m }$ as in Eq. (24) produces a zero-mean polynomial velocity residual of degree at most $P - 1$ and a path correction of degree at most P. Unlike piecewise-linear interpolation, each nodal value influences a polynomial across the whole interval.

How many independent modes? Both nodal constructions reproduce constant functions, so $\begin{array} { r } { \sum _ { m } \ell _ { m } ( \tau ) = 1 } \end{array}$ and $\Sigma _ { m } w _ { m } = 1$ . Consequently, $\begin{array} { r } { \sum _ { m } b _ { m } ( \tau ) = 0 } \end{array}$ . Adding the same vector to all $a _ { m }$ leaves $r _ { \phi }$ unchanged, so one coefficient direction is redundant. They therefore have $P - 1$ independent modes per velocity coordinate, whereas Polynomial and Spectral have P (Table 13).

A controlled path-fitting example. To expose approximation behavior, we fit a single scalar coordinate of $q _ { \phi }$ while holding the coarse path fixed. The target contains two full periods of a square wave, alternating between positive and negative half-cycles.

$$
\begin{array} { r } { q _ { * } ( \tau ) = \left\{ \begin{array} { l l } { 1 , } & { \tau \in ( 0 , \frac { 1 } { 4 } ) \cup ( \frac { 1 } { 2 } , \frac { 3 } { 4 } ) , } \\ { - 1 , } & { \tau \in ( \frac { 1 } { 4 } , \frac { 1 } { 2 } ) \cup ( \frac { 3 } { 4 } , 1 ) , } \\ { 0 , } & { \tau \in \{ 0 , \frac { 1 } { 4 } , \frac { 1 } { 2 } , \frac { 3 } { 4 } , 1 \} . } \end{array} \right. } \end{array}\tag{28}
$$

For each basis and $P \in \{ 4 , 1 2 , 3 2 \}$ , we minimize

$$
\operatorname* { m i n } _ { a _ { 1 } , \ldots , a _ { P } } \int _ { 0 } ^ { 1 } \left[ \sum _ { m = 1 } ^ { P } a _ { m } I _ { m } ( \tau ) - q _ { * } ( \tau ) \right] ^ { 2 } d \tau .\tag{29}
$$

This illustrative fit holds the coarse path fixed and measures approximation error rather than distillation performance. Increasing P sharpens the transitions in Figure 34, although finite-dimensional continuous fits retain overshoot and cannot exactly reproduce the jumps.

At matched independent dimension $d ,$ Polynomia $( P = d )$ and Collocation $( P = d + 1 )$ span the same space of polynomials of degree at most $d + 1$ that vanish at both endpoints. Their least-squares fits therefore coincide. At equal $P ,$ Polynomial has one more independent mode.

Table 13. Endpoint-preserving residual parameterizations with P coefficients. Modes are counted per velocity coordinate. Nodal constructions require $P \geq 2$
<table><tr><td>Parameterization</td><td>Velocity residual  $r _ { \phi }$ </td><td>Path correction  $q _ { \phi }$ </td><td>Modes</td></tr><tr><td>Polynomial</td><td>Polynomial, degree ≤ P</td><td>Polynomial, degree ≤ P + 1</td><td>P</td></tr><tr><td>Spectral</td><td>Cosine modes  $1 , \ldots , P$ </td><td>Sine modes scaled by 1/(mπ)</td><td>P</td></tr><tr><td>Piecewise</td><td>Continuous, piecewise linear</td><td>Continuously differentiable, piecewise quadratic</td><td>P-1</td></tr><tr><td>Collocation</td><td>Polynomial, degree  $\leq P - 1$ </td><td>Polynomial, degree  $\le P$ </td><td>P-1</td></tr></table>

## A.2 Block-Swap Study

We examine how different block groups affect few-step generation using Base and Turbo pairs of SANA-Sprint [57], Krea-2, and SD3.5-Large. We replace the early, middle, or late third of one model’s

blocks with the corresponding blocks from its paired model. Each setting uses the same few-step schedule and one image per prompt on 100 HPSv2 prompts [158].

In all three pairs, inserting Turbo’s late blocks into Base gives the highest scores among the tested groups, while replacing Turbo’s late blocks with Base blocks gives the lowest (Table 14). This sensitivity to late blocks supports using late student features for Pivot. Figure 35 shows the visual effects.

Table 14. Block swaps between Base and Turbo models. Scores are mean HPSv2. Bold marks the highest score within each replacement direction.
<table><tr><td></td><td colspan="3">Base with Turbo blocks</td><td colspan="3">Turbo with Base blocks</td></tr><tr><td>Model</td><td>Early</td><td>Middle</td><td>Late</td><td>Early</td><td>Middle</td><td>Late</td></tr><tr><td>SANA</td><td>0.138</td><td>0.231</td><td>0.269</td><td>0.297</td><td>0.273</td><td>0.242</td></tr><tr><td>Krea-2</td><td>0.175</td><td>0.205</td><td>0.291</td><td>0.304</td><td>0.301</td><td>0.216</td></tr><tr><td>SD3.5-Large</td><td>0.151</td><td>0.198</td><td>0.288</td><td>0.299</td><td>0.289</td><td>0.202</td></tr></table>

Base  
Early  
Middle  
Late  
![](images/4f8bfcb7479564ab6774b82c16af7e370809611d21bd8e8e8025e278879d8d49.jpg)  
Turbo  
Replace Base Blocks with Turbo Blocks

Base  
Early  
Middle  
Late  
Turbo  
![](images/f71c43f3cbcb5b3110d2309569c2c6a6964186de54d9f6cb3f127489ad08e106.jpg)  
Replace Turbo Blocks with Base Blocks

Figure 35. Late blocks shape few-step generation. The left half replaces one block group in each Base model with its Turbo counterpart, while the right half performs the reverse replacement. Rows correspond to SANA, Krea-2, and SD3.5-Large.

## A.3 Training Architecture

Figure 36 compares representative training computations. Here L denotes backbone depth, while K and M follow Section 3.2. Pivot’s residual projection produces P basis coefficients defining $r _ { \phi } ( \tau )$ Panel (b) shows teacher and fake-score predictions $v _ { T }$ and $v _ { \psi }$ in velocity form. In panel (c), $F _ { \theta } ( x _ { t } , t , s )$ maps a state at time t to time s, and $D _ { s }$ denotes differentiation with respect to the destination time.

![](images/7bea896a696f796314942c29f2976ea62b04309e57b4184f9bad589dc447113b.jpg)  
Figure 36. Representative training architectures. SMD uses a training-only Pivot branch, distribution matching uses a fake-score model, and the illustrated flow map uses a JVP.

## B Few-step Implementation and Evaluation Details

SMD distillation runs use 8 NVIDIA A100 GPUs. Tables 15, 16, and 18 summarize the reported training configurations. Pivot and repulsion are enabled according to these per-model settings. NFE counts deployed student evaluations. Teacher guidance, reference predictions, and Pivot are training-only. Across the configuration tables, Student and Pivot specify their trainable parameterizations, Target / ref. lists the target and reference prediction settings, λ is the repulsion weight, and an en dash marks an inactive or inapplicable entry.

## B.1 Image Generation

We train FLUX.1-dev and Qwen-Image using the π-Flow prompt set [14], grouped into resolution buckets. Training uses text prompts and image dimensions only. Trajectories start from random noise and are generated on-policy. Both students are initialized from their teachers. FLUX.1-dev uses AdamW with zero weight decay, a 100-iteration linear warm-up, and a constant learning rate thereafter. We select checkpoints by combined GenEval, DPG-Bench, and OneIG-Bench-EN performance. We report the mean of the five OneIG-Bench-EN dimensions as Overall. All released baselines are evaluated using the official benchmark procedures. PDD results are quoted from its paper.

## B.2 Image Editing

We collect a small corpus of source images paired with short English instructions spanning object, appearance, style, and text edits. The source image and instruction condition an on-policy rollout from noise. Target edited images are not used. Following the official pipeline, source images use the VAE posterior mode and their conditioning tokens are assigned timestep zero. We retain Qwen-Image-Edit-2511’s native CFG rescaling and use dynamic timestep shifting with factor 2. ImgEdit-Bench and

GEdit-Bench-EN are evaluated with their official procedures, with GPT-4o for ImgEdit-Bench and GPT-4.1 for GEdit-Bench-EN.

Table 15. Training configurations for image generation and editing.
<table><tr><td>Setting</td><td>FLUX.1-dev</td><td>Qwen-Image</td><td>Qwen-Image-Edit-2511</td></tr><tr><td>Student</td><td>LoRA-256</td><td>LoRA-256</td><td>LoRA-256</td></tr><tr><td>Pivot</td><td>Full-rank</td><td>Full-rank</td><td>Full-rank</td></tr><tr><td>Pivot blocks</td><td>3</td><td>3</td><td>3</td></tr><tr><td>Basis / P</td><td>Polynomial / 4</td><td>Piecewise / 4</td><td>Polynomial / 8</td></tr><tr><td>Queries / transition</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Target / ref.</td><td>3.5 / 1.0</td><td>5.0 / 1.0</td><td>5.0 / 1.0</td></tr><tr><td>λ</td><td>0.4</td><td>0.2</td><td>0.4</td></tr><tr><td>Global batch</td><td>32</td><td>16</td><td>16</td></tr><tr><td>Student LR</td><td>10-4</td><td>10-4</td><td>10-4</td></tr><tr><td>Pivot LR</td><td>10-4</td><td>10-4</td><td>10-4</td></tr><tr><td>Iterations</td><td>1,250</td><td>1,000</td><td>250</td></tr><tr><td>Inference NFE</td><td>4</td><td>4</td><td>4</td></tr></table>

## B.3 3D Shape and Texture

Hunyuan3D-2.1 uses one rendered view per object from a TexVerseFull-derived collection. We preserve alpha, apply the official recentering and white-background compositing pipeline, and condition the on-policy shape-latent rollout on the processed image. No ground-truth shape latent is used. The latent contains 4,096 vectors with 64 channels, and the output normalization layer remains trainable.

For TRELLIS.2, we distill the high-resolution shape and texture models in the 1024-resolution cascade while retaining the preceding sparse-structure and coarse-shape stages. Training conditions on images and sparse voxel coordinates from a TexVerseFull-derived collection. The texture rollout additionally receives the shape latent. Ground-truth sparse features are not used to construct rollout states, and both output normalization layers remain trainable. The shape teacher uses guidance 7.5 on [0.6, 1.0] with rescaling 0.5. The texture teacher uses guidance 1 without rescaling. Both stages use a 100-iteration warm-up, constant learning rates thereafter, timestep shift 3, and no EMA.

Table 16. Training configurations for 3D generation. Each distilled TRELLIS.2 stage uses 4-NFE.
<table><tr><td>Setting</td><td>Hunyuan3D-2.1 Shape</td><td>TRELLIS.2 Shape</td><td>TRELLIS.2 Texture</td></tr><tr><td>Student</td><td>LoRA-128</td><td>LoRA-128</td><td>LoRA-128</td></tr><tr><td>Pivot</td><td>Full-rank</td><td>Full-rank</td><td>Full-rank</td></tr><tr><td>Pivot blocks</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Basis / P</td><td>Polynomial / 4</td><td>Polynomial / 4</td><td>Polynomial / 4</td></tr><tr><td>Queries / transition</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Target / ref.</td><td>5.0 / conditional</td><td>7.5 / -</td><td>1.0 / -</td></tr><tr><td>λ</td><td>0.2</td><td></td><td>一</td></tr><tr><td>Global batch</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Student LR</td><td>10-5</td><td>10-5</td><td>10-5</td></tr><tr><td>Pivot LR</td><td>10-4</td><td>10-4</td><td>10-4</td></tr><tr><td>Iterations</td><td>1,500</td><td>500</td><td>500</td></tr><tr><td>Inference NFE</td><td>4</td><td>4</td><td>4</td></tr></table>

## B.4 Video Generation

SMD distillation of Wan2.1 uses the 248K-entry preprocessed dataset released by rCM [37]. Students are initialized from their teachers and trained on-policy, with query points sampled in sigma space. Wan2.1-T2V-1.3B uses an unconditional reference that skips the 10th teacher layer. Wan2.1-T2V-14B uses the conditional teacher prediction as its reference. Both target branches use the unmodified teacher. We use timestep shifts 10 and 5, respectively, and select checkpoints by overall VBench score.

DMD2∗ baseline. We implement a DMD2-based 4-step distribution-matching baseline [15] on Wan2.1-T2V-1.3B and Wan2.1-T2V-14B. Both use the same synthetic latent dataset generated by Wan2.1- T2V-14B for data-aware training, with rank-256 LoRA for the student and auxiliary fake-score model. The objective combines VSD and GAN losses, with GAN weight 0.03. We alternate student and fake-score/discriminator updates at a 1:4 ratio. Both model scales use teacher CFG 6 and learning rate $1 0 ^ { - 4 }$ for the student and fake-score model. Both baselines are evaluated after 500 iterations.

Evaluation follows the Self-Forcing protocol with five videos per prompt, 832 480 resolution, 81 frames, and 16 FPS. We generate with expanded prompts and score their short versions. We evaluate released rCM and AnyFlow checkpoints and quote PDD results from its paper.

## B.5 Detailed VBench Results

Table 17 reports all 16 VBench dimensions for the few-step methods compared in Table 4. The teachers are omitted because PDD does not report their per-dimension scores. The Quality and Semantic aggregates use VBench’s standard per-dimension normalization and weights.

Dynamic Degree uses RAFT optical flow to identify videos with sufficiently large motion across multiple frame pairs, and reports the fraction classified as dynamic. Motion Smoothness measures agreement between held-out frames and frames reconstructed by a video interpolation model, with lower reconstruction error yielding a higher score. We report these dimensions together to distinguish motion activity from temporal smoothness. Color measures whether detected objects exhibit the colors specified by the prompts, rather than color saturation or visual naturalness. These per-video dimensions do not directly measure diversity across repeated generations of the same prompt.

Table 17. Full 16-dimensional VBench breakdown for Wan2.1-T2V at 4-NFE. The Quality dimensions are Subject Consistency (SC), Background Consistency (BC), Temporal Flickering (TF), Motion Smoothness (MS), Dynamic Degree (DD), Aesthetic Quality (AQ), and Imaging Quality (IQ). The Semantic dimensions are Object Class (OC), Multiple Objects (MO), Human Action (HA), Color (C), Spatia Relationship (SR), Scene (S), Appearance Style (AS), Temporal Style (TS), and Overall Consistency (OC′). All metrics are higher-is-better. Boldface and underlining indicate the best and second-best results within each backbone. ∗Our reproduced DMD2 results. <sup>†</sup>Quoted PDD results [1].
<table><tr><td rowspan="2">Method</td><td colspan="6">Quality</td><td colspan="8">Semantic</td></tr><tr><td>SC</td><td>BC</td><td>TF</td><td>MS DD</td><td>AQ IQ OC</td><td colspan="7">MO HA C SR</td><td>S</td><td>AS</td><td>TS</td><td>OC&#x27;</td></tr><tr><td></td><td colspan="14">Wan2.1-T2V-1.3B</td></tr><tr><td>DMD2*</td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td><td>97.7697.8299.4298.8145.0068.8069.4695.8587.8095.6084.6373.7653.2121.5923.4326.28</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>rCM</td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2">94.7295.7998.7797.6481.9464.6866.7393.7685.6496.0085.3974.9556.8720.5024.3826.37</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>AnyFlow97.7996.9699.2698.2968.3367.0167.6395.7490.4797.0087.6780.3357.5020.6524.3726.53</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PDD†</td><td></td><td></td><td></td><td></td><td></td><td></td><td>97.6995.9099.3399.02 72.5066.26 69.03 96.2285.79 93.34 85.4176.95 51.4719.33 23.77 25.73</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td>96.2995.44 98.8398.6679.4466.1968.14 94.2183.8496.0089.19 74.54 55.1119.91 23.7625.82</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Wan2.1-T2V-14B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="14">96.92 97.8399.6798.76 49.72 67.0767.04 97.1788.8197.0087.81 73.5552.54 21.8823.56 26.50</td></tr><tr><td>DMD2* rCM</td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td><td>95.6596.4598.5797.8176.3966.7669.1797.59 90.4898.8082.9183.0553.2721.00 24.4826.39</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>AnyFlow 97.22 97.40 99.00 98.10 73.61 67.12 68.78 97.56 92.50 98.2085.79 83.84 54.72 20.90 24.68 26.83</td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PDD†</td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td><td>95.41 96.42 99.39 98.52 74.7266.78 68.8896.5587.90 96.4086.5980.0254.3020.86 24.3926.49</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2">96.60 95.73 97.8698.5283.8966.4768.50 97.37 90.70 96.2086.18 80.92 53.2119.93 24.10 26.42</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## B.6 Joint Audio–Video Generation

MiniMax H3 is a 50-block joint audio–video Transformer with guidance already distilled into the model, so student inference requires no additional CFG branch. Qwen3-VL-32B layer-50 hidden states provide cached BF16 conditions. Text-conditioned training uses 100K prompts from VidProM [164]. We use an LLM to add plausible audio descriptions, including ambient sounds, then rewrite the prompts into the MiniMax H3 prompt format. Training uses no real target videos or first/last-frame inputs. Video and audio have equal imitation weights. The reference skips layer 25.

For reference-conditioned generation, we pair 20K instructions with 12K clips from UltraVideo [165] across ten task categories and target durations of 5, 8, or 10 seconds. References are conditions rather than target outputs. Training uses eight GPUs with context-parallel degree four and global batch two. The data primarily covers audio–video references rather than all mixed-reference inputs.

Table 18. Training configurations for video and joint audio–video generation.
<table><tr><td>Setting</td><td>Wan2.1-T2V-1.3B</td><td></td><td>Wan2.1-T2V-14B</td></tr><tr><td>Student</td><td>LoRA-256</td><td></td><td>LoRA-256</td></tr><tr><td>Pivot</td><td>Full-rank</td><td></td><td></td></tr><tr><td>Pivot blocks</td><td></td><td></td><td></td></tr><tr><td>Basis / P</td><td></td><td>Polynomial / 4 2</td><td></td></tr><tr><td>Queries / transition</td><td></td><td></td><td>2</td></tr><tr><td>Target / ref.</td><td>5.0 / 0.0</td><td></td><td>6.0 / 1.0</td></tr><tr><td>λ</td><td></td><td></td><td>0.2</td></tr><tr><td>Global batch</td><td></td><td></td><td>16</td></tr><tr><td>Student LR</td><td></td><td></td><td>10-4</td></tr><tr><td>Pivot LR</td><td></td><td></td><td>3×10-5</td></tr><tr><td>Iterations</td><td></td><td></td><td>200</td></tr><tr><td>Inference NFE</td><td></td><td></td><td>4</td></tr><tr><td>Setting</td><td>H3 8-NFE</td><td>H3 4-NFE</td><td>H3 Ref. 8-NFE</td></tr><tr><td>Student</td><td></td><td></td><td>H3 Ref. 4-NFE LoRA-128</td></tr><tr><td>Pivot</td><td>LoRA-128 LoRA-128</td><td>LoRA-128 LoRA-128</td><td>LoRA-128 LoRA-128</td></tr><tr><td>Pivot blocks</td><td>7</td><td>7</td><td></td></tr><tr><td>Basis / P</td><td>Piecewise / 4</td><td>Piecewise / 32</td><td>Piecewise / 32</td></tr><tr><td>Queries / transition</td><td>1</td><td></td><td></td></tr><tr><td>Target / ref.</td><td>native / skip 25</td><td>native / skip 25</td><td>1 native / skip 25</td></tr><tr><td>λ</td><td>0.2</td><td></td><td>0.2</td></tr><tr><td>Global batch</td><td></td><td></td><td></td></tr><tr><td>Student LR</td><td>10-4</td><td></td><td>2 10-4</td></tr><tr><td>Pivot LR</td><td>10⁻4</td><td></td><td>10-4</td></tr><tr><td>Iterations</td><td></td><td></td><td>1,000</td></tr><tr><td></td><td>1,000</td><td></td><td></td></tr><tr><td>Inference NFE</td><td></td><td></td><td></td></tr></table>

## B.7 H3 Evaluation

Both inference budgets use the same evaluation set, with prompts, seeds, resolution, and duration matched between methods. GPT-5.6 Sol evaluates 24 chronological frames twice with reversed candidate order. Scores are averaged across orders and prompts. We report four visual dimensions on a 1–5 scale. Prompt alignment measures visually verifiable fulfillment of the requested content and camera behavior. Structural coherence measures the credibility and stability of anatomy, objects, geometry, materials, and subject identity. Temporal coherence measures continuity, object permanence, and physically plausible motion and interactions. Visual naturalness measures the absence of rendering artifacts and implausible color or lighting. Audio clarity is the PAM score multiplied by 5, while audio quality is Audiobox Aesthetics Production Quality divided by 2. Both are therefore reported on a 0–5 scale [166, 167].

The full visual evaluation prompt is provided in Appendix E.

Figure 19 uses full-clip spectrograms from native 32 kHz stereo audio, computed with a 1,024-sample Hann window and 128-sample hop. We average channel powers after the transform and display 0–16 kHz on a shared [ 90, 0] dB scale without per-clip normalization or denoising.

## C Beyond Self-Distillation Implementation and Evaluation Details

Tables 19 and 20 summarize the training configurations for the studies in Section 5.

Table 19. Training configurations for cross-model and multi-teacher image distillation.  
Cross-model distillation
<table><tr><td></td><td>FLUX.2-klein-4B FLUX.2-klein-4B</td><td>FLUX.2-klein-9B → FLUX.2-klein-4B</td><td>FLUX.2-klein-9B Lens-RL-3.8B</td><td>Lens-RL-3.8B &gt; Lens-RL-3.8B</td></tr><tr><td>Teacher → student</td><td></td><td></td><td></td><td></td></tr><tr><td>Student</td><td>LoRA-256</td><td>LoRA-256</td><td>LoRA-256 Full-rank</td><td>LoRA-256 Full-rank</td></tr><tr><td>Pivot</td><td>Full-rank</td><td>Full-rank 3</td><td>3</td><td>3</td></tr><tr><td>Pivot blocks Basis / P</td><td>3 Polynomial / 4</td><td>Polynomial / 4</td><td>Polynomial / 4</td><td>Polynomial / 4</td></tr><tr><td>Queries / transition</td><td>4</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Target / ref.</td><td>4.0 / -</td><td>4.0 / -</td><td>4.0 / -</td><td>5.0 / -</td></tr><tr><td>λ</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Global batch</td><td>32</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Student LR</td><td>10-4</td><td>10-4</td><td>10-4</td><td>10-4</td></tr><tr><td>Pivot LR</td><td>10-4</td><td>10-4</td><td>10⁻4</td><td>10-4</td></tr><tr><td>Iterations</td><td>2,000</td><td>2,000</td><td>2,000</td><td>2,000</td></tr><tr><td>Inference NFE</td><td>4</td><td>4</td><td>4</td><td>4</td></tr></table>

Multiple experts as teachers
<table><tr><td>Teacher → student</td><td>Base teacher → SD3.5-Medium</td><td>Reward specialist → SD3.5-Medium</td><td>Routed teachers → SD3.5-Medium</td></tr><tr><td>Student</td><td>Full-rank</td><td>Full-rank</td><td>Full-rank</td></tr><tr><td>Pivot</td><td>Full-rank</td><td>Full-rank</td><td>Full-rank</td></tr><tr><td>Pivot blocks</td><td>3</td><td>3</td><td>3</td></tr><tr><td>Basis / P</td><td>Polynomial / 4</td><td>Polynomial / 4</td><td>Polynomial / 4</td></tr><tr><td>Queries / transition</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Target / ref.</td><td>5.0 / -</td><td>5.0 / -</td><td>5.0 / -</td></tr><tr><td>λ</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Global batch</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Student LR</td><td>10-5</td><td>10-5</td><td>10-5</td></tr><tr><td>Pivot LR</td><td>10⁻4</td><td>10⁻4</td><td>10-4</td></tr><tr><td>Iterations</td><td>2,500</td><td>2,500</td><td>2,500</td></tr><tr><td>Inference NFE</td><td>4</td><td>4</td><td>4</td></tr></table>

## C.1 Cross-Model Distillation

We use the π-Flow prompt set [14] for cross-model distillation. LoRA alpha equals the rank, with dropout 0.05. Training uses a 100-iteration warm-up and four samples per GPU. Lens self-distillation retains its native CFG norm rescaling. GenEval, DPG-Bench, and OneIG-Bench-EN follow the same evaluation procedures used for image generation.

## C.2 Multiple Experts as Teachers

We expand four Flow-GRPO [147] training splits into four 30K-prompt categories and combine them into a 120K-entry pool. Self-distillation uses the base teacher for the full pool, while each specialist uses one fixed reward teacher. In multi-teacher training, each batch contains prompts from one category and uses its corresponding teacher. The four categories are sampled with equal probability.

All SD3.5-Medium configurations use one shared Pivot and timestep shift 3. GenEval, OCR, and PickScore use their own test sets. We evaluate CLIPScore, HPSv2, and ImageReward on DrawBench.

## C.3 Few-Step Causal Video Generation

We train on the extended VidProM prompts released by Self-Forcing [40], using eight NVIDIA A100 GPUs. The causal rollout uses seven chunks of three latent frames each, with fresh stochastic re-noising between transitions. The causal teacher supplies the imitation target. The frozen bidirectional Wan2.1-T2V-14B teacher supplies the correction signal. The student uses LoRA alpha 256 and no dropout. Pivot uses $K = 4$ (Section 3.2). The student and Pivot use AdamW with betas (0.9, 0.999) and zero weight decay. We sample queries in sigma space, with one prompt per GPU and two accumulation rounds.

The correction sweep varies $\lambda _ { \mathrm { c o r r } } \in$ 0, 0.2, 0.4, 0.8 from the same causal ODE initialization, with reference repulsion disabled. The bidirectional correction branch skips layer

Table 20. Training configurations for causal Wan2.1-T2V-1.3B distillation.
<table><tr><td>Setting</td><td>Ours</td><td>DMD*</td></tr><tr><td>Student</td><td>LoRA-256</td><td>LoRA-256</td></tr><tr><td>Pivot</td><td>Full-rank</td><td>一</td></tr><tr><td>Pivot blocks</td><td>3</td><td></td></tr><tr><td>Basis / P</td><td>Polynomial / 4</td><td></td></tr><tr><td>Queries / transition</td><td>2</td><td></td></tr><tr><td>Target / ref.</td><td> $3 . 0 ~ / \mathrm { ~ - ~ }$ </td><td>4.0 / -</td></tr><tr><td>λ</td><td>0</td><td>一</td></tr><tr><td>Global batch</td><td>16</td><td>8</td></tr><tr><td>Student LR</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>Pivot LR</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Iterations</td><td>300</td><td>300</td></tr><tr><td>Inference NFE</td><td>4</td><td>4</td></tr></table>

11. VBench evaluation follows the Self-Forcing protocol, with five $8 3 2 \times 4 8 0$ videos per prompt, 81 frames, and 16 FPS.

The DMD∗ control uses the same generator initialization, a frozen bidirectional Wan2.1-T2V-14B real-score model, and a trainable bidirectional Wan2.1-T2V-1.3B fake-score model. The fake-score model uses LoRA-256 and learning rate $1 0 ^ { - 5 }$ . Both optimizers are AdamW with betas (0, 0.999) and weight decay 0.01. Each GPU processes one prompt without gradient accumulation. Five fake-score updates per generator update give 50 generator and 250 fake-score updates at iteration 300.

## D Ablation Implementation and Evaluation Details

Training. Table 21 lists default settings. Layer and block-count ablations use randomly initialized Pivot. Basis and repulsion sweeps use pretrained Pivot at K = 4, M = 3. Wan2.1-T2V-1.3B uses the main video corpus, LoRA rank and alpha 256, and learning rate $1 0 ^ { - 4 }$ for student and Pivot with a 100-iteration warm-up. We use two queries per transition, four-step Euler sampling, and timestep shift 6. Single-prompt video models are trained for 200 iterations.

Table 21. Default Pivot and teacher-repulsion settings.
<table><tr><td>Setting</td><td>Qwen-Image</td><td>Wan2.1-T2V-1.3B</td></tr><tr><td>Pivot blocks</td><td>3</td><td>3</td></tr><tr><td>Basis / P</td><td>Polynomial / 4</td><td>Polynomial / 4</td></tr><tr><td>Target / ref.</td><td>5.0 / 1.0</td><td>5.0 / 0.0</td></tr><tr><td>λ</td><td>0.2</td><td>0.1</td></tr><tr><td>Iterations</td><td>1,000</td><td>300</td></tr><tr><td>Inference NFE</td><td>4</td><td>4</td></tr></table>

Evaluation. We evaluate Qwen-Image with OneIG-Bench-EN [32] and Wan2.1-T2V-1.3B with VBench [39] on Self-Forcing prompts [40], using one run per setting. Qualitative cells show one sample for single-prompt video and four otherwise.

## E Visual Evaluation System Prompt

Visual evaluation system prompt   
Compare the visual quality of two anonymous generated videos.   
Call \`read\_video\_visual\_pair\` exactly once with \`frameCount={frame\_count}\`,   
\`balancePositions=true\`, \`includeAudio=false\`, and:   
Candidate A: {paths[0]}   
Candidate B: {paths[1]}   
The tool returns matched chronological frames. Candidate positions alternate   
and are balanced; identify candidates only by their visible A/B labels.   
<TARGET\_PROMPT>   
{target}   
</TARGET\_PROMPT>   
This is an equivalence-first visual-quality assessment. Do not judge audio.   
Score visually observable prompt fulfillment only under \`prompt\_alignment\`;   
do not let prompt completeness, exact camera wording, framing compliance,   
shot count, or missing requested events affect the other three dimensions.   
Ignore prompt requirements that can only be verified from audio. Do not guess   
model identity.   
First assess each candidate independently on:   
1. \`prompt\_alignment\`: visual correctness and completeness of requested   
subjects, attributes, actions, interactions, setting, event order, shot   
structure, framing, and camera behavior. Score what is visibly verifiable in   
the sampled frames. Do not use rendering polish or artifact severity as   
alignment evidence.   
2. \`structural\_coherence\`: recognizable and credible anatomy, objects,   
geometry, materials, and subject identity within sampled frames.   
3. \`temporal\_coherence\`: identity and shape stability, physically plausible   
motion and interaction, object permanence, and broad continuity across the   
sampled sequence.   
4. \`visual\_naturalness\`: absence of disruptive rendering artifacts,   
oversharpening, halos, crunchy or plastic texture, artificial tone mapping,   
and implausible color or lighting.   
Use integer scores: 5 excellent, 4 good with minor/localized defects, 3   
material but usable limitations, 2 major or pervasive failures, 1 failed.   
Sharpness, detail, saturation, contrast, pseudo-HDR, cinematic grading, and   
raw motion magnitude are never positive evidence. Mild softness, natural   
motion blur, shallow depth of field, and restrained detail are neutral unless   
they destroy recognizable structure.   
Then decide whether a careful human viewer would perceive a MATERIAL visual   
quality difference. \`equivalent\` is the default. Break equivalence only when   
one candidate has a persistent quality advantage visible at multiple   
separated times or a major defect that substantially impairs viewing.   
Localized imperfections, different stylistic treatment, minor score   
differences, and prompt-following differences do not break equivalence.

Do not force a winner. List only persistent observations supported across   
multiple frames; use an empty list if none exists.   
Return exactly:   
{   
"candidates": {   
"A": {   
"prompt\_alignment": <integer 1-5>,   
"prompt\_alignment\_evidence": "<specific visible evidence>",   
"structural\_coherence": <integer 1-5>,   
"structural\_coherence\_evidence": "<specific evidence>",   
"temporal\_coherence": <integer 1-5>,   
"temporal\_coherence\_evidence": "<specific evidence>",   
"visual\_naturalness": <integer 1-5>,   
"visual\_naturalness\_evidence": "<specific evidence>"   
},   
"B": {   
"prompt\_alignment": <integer 1-5>,   
"prompt\_alignment\_evidence": "<specific visible evidence>",   
"structural\_coherence": <integer 1-5>,   
"structural\_coherence\_evidence": "<specific evidence>",   
"temporal\_coherence": <integer 1-5>,   
"temporal\_coherence\_evidence": "<specific evidence>",   
"visual\_naturalness": <integer 1-5>,   
"visual\_naturalness\_evidence": "<specific evidence>"   
}   
},   
"persistent\_observations": ["<observation supported at separated times>"],   
"verdict": "<equivalent|A\_better|B\_better>",   
"confidence": "<low|medium|high>",   
"reason": "<why the evidence does or does not establish a material quality difference>"   
}