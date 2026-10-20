# TAMEDFLOW: TRAJECTORY-AWARE DISTILLATION FOR TEACHER-ASSISTED DIFFUSION SAMPLING

Shuaiqi Cheng<sup>1,\*</sup> Jiahao Zhang<sup>1,\*</sup> Boyuan Gu<sup>2</sup> Siyu You<sup>2</sup> Yuxi Chen<sup>3</sup> Haiying Xu<sup>1</sup> Zexin Lu<sup>4</sup> Huiyu Zhou<sup>5</sup> Xuming Hu<sup>1,3</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou) <sup>2</sup>University of Electronic Science and Technology of China <sup>3</sup>The Hong Kong University of Science and Technology <sup>4</sup>Huawei Technologies Ltd. <sup>5</sup>Guangxi Zhuang Autonomous Region Information Center These authors contributed equally.

## ABSTRACT

Diffusion Transformers (DiTs) deliver high-fidelity image and video generation, but inference requires many sequential model evaluations. Two acceleration strategies are increasingly important: model compression makes each evaluation cheaper, while adaptive inference avoids selected teacher computation through reuse, forecasting, or speculative execution. Their combination, however, is brittle: directly applying the evaluated adaptive methods to a compressed student can degrade quality, an observed failure mode we call the Combination Gap. We link this gap to a mismatch between pointwise distillation and short-horizon execution: a student trained at teacher states need not remain reliable when its own predictions are chained. We propose TamedFlow, which trains a compressed DiT for this deployed role through cross-architecture feature alignment, a near-identity Velocity Alignment Head, and randomized rollout supervision. At inference, a lightweight Learnable Trajectory Estimator routes completed drafts: accepted drafts bypass the teacher, while rejected drafts roll back and receive one teacher rescue step. Distilling DiT-XL/2 into DiT-B/2, TamedFlow reaches FID 2.58 at 2.25× measured speedup over DDIM-50 on ImageNet 256 × 256. Ablations support trajectory supervision, and routing controls favor LTE over random decisions at nearly matched latency. Evaluations on ImageNet 512 × 512, MS-COCO, and Latte/FaceForensics extend the empirical scope.

## 1 INTRODUCTION

Diffusion Transformers (DiTs) have become a leading backbone for high-fidelity visual generation (Peebles & Xie, 2023), yet iterative denoising imposes high inference latency. Each sample requires dozens of sequential forward passes, with costs scaling rapidly with model parameters, resolution, and sequence length. Existing acceleration strategies attack complementary aspects of this cost: step-reduction methods shorten the solver schedule (Song et al., 2023; Luo et al., 2023; Geng et al., 2026; Kim et al., 2025); parameter compression lowers per-step computation; and adaptive inference reuses or forecasts intermediate activations (Ma et al., 2024b; Selvaraju et al., 2024; Liu et al., 2025a;b). Combining parameter compression with adaptive execution is intuitively appealing, promising cheaper evaluations while skipping redundant computation. All headline experiments below use fixed-step DDIM integration to evaluate compression and adaptive execution under a unified solver convention.

In practice, this composition can break down. In our controlled diagnostic (Appendix Table 11), attaching the evaluated adaptive reuse methods directly to a compressed student degrades sample quality, inflating ImageNet 256 × 256 FID from 5.56 to between 10.40 and 13.18. We term this failure mode the Combination Gap. This result exposes an operational mismatch: a student trained for pointwise agreement need not remain reliable after its own predictions alter the states encountered later. Rather than applying internal reuse unchanged to the compact model, we let the student generate short trajectory drafts and reserve teacher intervention for drafts predicted to deviate.

![](images/5051389cafaae7ebcf47ec0b9bd5d939ed2b3733016c82dab5d464d458e3c4a2.jpg)  
(a) Trajectory Drift

![](images/459a684cc9da825e72bf4d4a36155aedc11c8952463811d894436670143e5712.jpg)  
(b) Trajectory Taming  
Figure 1: The Combination Gap and TamedFlow. Directly applying adaptive reuse to a parametercompressed student severely degrades quality in our diagnostic. TamedFlow instead trains the compressed student for short-horizon trajectory agreement. A 4M-parameter LTE routes drafts: accepted drafts bypass the teacher; rejected drafts roll back for one teacher rescue step.

Chaining student predictions into drafts, however, introduces compounding integration drift. Proposition 3.1 characterizes this through an Euler rollout bound: spatial sensitivity can amplify local student–teacher velocity mismatch over sequential steps. To stabilize drafting against this amplification, we introduce TamedFlow, combining trajectory-aware distillation with selective teacher rescue (Figure 1). During training, cross-architecture feature alignment (AlignNet) grounds student representations in teacher layers; a near-identity Velocity Alignment Head (VAH) absorbs residual velocity errors; and randomized K-step rollout supervision $( K \sim \mathcal { U } ( 1 , 5 ) )$ penalizes both intermediate mismatch and accumulated endpoint drift. A two-stage curriculum introduces the multi-step objectives only after feature alignment.

At inference, TamedFlow uses an asymmetric execution policy. A compact Learnable Trajectory Estimator (LTE, 4M parameters) scores completed drafts from pooled student tokens and timestep embeddings without teacher access. If confidence exceeds threshold τ, the draft is accepted, advancing K steps while bypassing the teacher. If rejected, the sampler rolls back to the pre-draft state and advances one step with teacher rescue. Varying τ selects among measured operating points between throughput and fidelity without sequential teacher verification.

Distilling DiT-XL/2 into a 130M DiT-B/2 student on ImageNet 256 × 256, TamedFlow attains FID 2.58 at 2.25× measured speedup (0.464 s vs. 1.044 s) over the DDIM-50 teacher. At comparable latency, TamedFlow improves FID over the evaluated training-free adaptive baselines (e.g., FoCa at FID 2.65, 0.466 s). Controlled ablations support trajectory supervision and learned routing at nearly matched latency, while evaluations on ImageNet 512 × 512, MS-COCO, and Latte video DiTs extend the empirical scope. In summary, our contributions are:

• Diagnostic and analysis of the Combination Gap. We document the failure of applying the evaluated adaptive reuse methods to a compressed DiT, and derive a DDIM rollout bound characterizing how sequential drafting can amplify local velocity errors.

• Trajectory-stabilized speculative distillation. We introduce TamedFlow, uniting feature alignment, residual velocity correction, and randomized rollout supervision to optimize a compressed DiT for autonomous multi-step drafting.

• Learned routing and deployment evaluation. We propose an asymmetric execution policy via a 4M-parameter LTE to accept reliable drafts or trigger teacher rescue, evaluating quality–latency trade-offs across resolutions, text conditioning, and video alongside physical deployment costs.

## 2 RELATED WORK

Prior work on DiT acceleration reduces the number of solver evaluations, the cost of each evaluation, or the teacher computation retained during adaptive execution. TamedFlow studies their interface: whether a compressed student can reliably replace short segments of a teacher trajectory.

## 2.1 STEP COUNT REDUCTION IN DIFFUSION AND FLOW MODELS

Step reduction lowers the number of solver evaluations. Higher-order solvers improve numerical efficiency for a fixed model (Song et al., 2021; Lu et al., 2025), while distillation changes the model: progressive distillation shortens the schedule (Salimans & Ho, 2022), consistency models match predictions across noise levels (Song et al., 2023; Luo et al., 2023), MeanFlow and Model-Guidance alter the target or absorb guidance (Geng et al., 2026; Tang et al., 2025). Other formulations align distributions or scores, distill autoregressively, transfer trajectories, or train few-step students adversarially (Yin et al., 2024; Zhou et al., 2025; Kim et al., 2025; Luo et al., 2025; Sauer et al., 2024). These methods optimize a sparse standalone sampler deployed entirely on its own. TamedFlow instead retains a multi-step teacher as a reliable safeguard and uses the student to propose short trajectory segments whose acceptance is decided adaptively online.

A related line studies trajectory geometry. Rectified flow and InstaFlow construct straighter paths (Liu et al., 2023); interpolant transformers unify diffusion and flow parametrizations (Ma et al., 2024a); and guided path sampling corrects trajectories during integration (Li et al., 2026). TCD and Hyper-SD learn consistency or trajectory segments for few-step sampling (Zheng et al., 2024; Ren et al., 2024). TamedFlow keeps the teacher path as the reference, trains a cross-architecture student for selectively accepted rollouts, and retains teacher rescue in the deployed sampler.

## 2.2 PER-STEP COMPUTE ACCELERATION AND FEATURE CACHING

Per-step methods keep the schedule while avoiding full recomputation. Feature-reuse methods cache activations across timesteps or blocks (Ma et al., 2024b; Chen et al., 2024; Selvaraju et al., 2024; Zou et al., 2025; 2026b; Wimbauer et al., 2024); forecasting methods extrapolate future features (Liu et al., 2025a;b; Cai et al., 2026; Huang et al., 2025); and learned schedulers select policies per sample (Zhao et al., 2026). These methods exploit local regularity in intermediate computations. Our same-student diagnostic shows that their evaluated configurations do not transfer directly to the compressed student, but does not isolate a universal caching failure mechanism.

Most reuse methods leave the model unchanged and decide which internal computations can be skipped. TamedFlow instead trains the student under rollout supervision and learns whether an entire student-generated segment can advance without a teacher call. DisCa (Zou et al., 2026a) also studies incompatibility between step-distilled video DiTs and feature caches, but repairs the cache side with a predictor and Restricted MeanFlow schedule. TamedFlow acts on the draft student instead.

## 2.3 SPECULATIVE VERIFICATION IN GENERATIVE PARADIGMS

Speculative decoding verifies proposed tokens with a target model (Leviathan et al., 2023); causal masking permits parallel verification and exact correction. Diffusion lacks this structure because each teacher state depends on the preceding prediction. Replaying a K-step draft exactly would require sequential teacher evaluations and remove much of the saving. SpecDiff (Pan et al., 2026) studies self-speculative diffusion under this constraint. TamedFlow replaces exact verification with learned trajectory estimation. The LTE scores student-internal features after a draft using rollout agreement as its training target. Accepted drafts avoid the teacher; rejected drafts return to the pre-draft state for one teacher step. Routing is therefore inexpensive and online, but approximate and not distribution-preserving by construction.

## 3 METHOD

## 3.1 PRELIMINARIES: DIFFUSION MODELS, DIT, AND FLOW MATCHING

Diffusion models (Ho et al., 2020) learn to reverse a gradual noising process, so generation requires a sequence of state-dependent model evaluations. A Diffusion Transformer (DiT) (Peebles & Xie, 2023) implements the denoiser with a transformer over latent patches and injects timestep and condition information through adaptive layer normalization. Our method is expressed in velocity form because both its trajectory loss and its stability analysis act on the vector field followed during sampling.

Flow matching (Lipman et al., 2023) casts generation as a probability-path ODE. With $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ $x _ { 1 } \sim p _ { \mathrm { d a t a } }$ , and the linear interpolant $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ , the conditional velocity $x _ { 1 } - x _ { 0 }$ supplies the regression target:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , x _ { 0 } , x _ { 1 } } \Big [ \big \| v _ { \theta } ( x _ { t } , t ) - ( x _ { 1 } - x _ { 0 } ) \big \| _ { 2 } ^ { 2 } \Big ] , \qquad t \sim \mathcal { U } [ 0 , 1 ] .\tag{1}
$$

We use forward flow time, from noise at $t = 0$ to data at $t = 1$ . The main experiments use the stated DDIM schedules; the Euler discretization below is an analysis device and the rollout rule used to define the trajectory objectives.

## 3.2 SPATIAL ERROR AMPLIFICATION AND ROLLOUT STABILITY

Pointwise velocity matching constrains the student at sampled states but not the states reached after its own predictions are chained. The following proposition separates the local mismatch that initiates drift from the spatial sensitivity that amplifies it. It applies to sequential Euler drafts and does not analyze internal feature caching or forecasting.

Proposition 3.1 (K-Step Trajectory Alignment Bound). Let $\tilde { v } _ { S } \equiv v _ { S } ^ { \mathrm { a l i g n e d } }$ denote the deployed student velocity field, and let $x _ { S } ^ { \mathrm { r o l l } } ( K )$ and $x _ { T } ^ { \mathrm { r o l l } } ( K )$ be the aligned-student and teacher endpoints after K sequential Euler stepsfrom state x withfixed step size h. With time arguments suppressed at the common $t _ { k } = t + k h$ , define the baseline velocity mismatch $\delta _ { v } ^ { ( k ) } \triangleq \tilde { v } _ { S } ( \Breve { x _ { T } ^ { ( k ) } } ) - v _ { T } \Breve { ( x _ { T } ^ { ( k ) } ) }$ , with $\lVert \delta _ { v } \rVert _ { \infty } \triangleq \operatorname* { m a x } _ { k } \| \delta _ { v } ^ { ( k ) } \| _ { 2 } .$ Let $\begin{array} { r } { L _ { T } \triangleq \operatorname* { s u p } _ { x , t } \| J _ { v _ { T } } ( x , t ) \| _ { 2 } } \end{array}$ and $\begin{array} { r } { \Delta J \triangleq \operatorname* { s u p } _ { x , t } \Vert J _ { \tilde { v } _ { S } } ( x , t ) - J _ { v _ { T } } ( x , t ) \Vert _ { 2 } . } \end{array}$ Assuming both velocityfields are continuously differentiable on a region containing each segment between paired student and teacher states and $C \triangleq L _ { T } + \Delta J < \infty$ , the spatial deviation satisfies the discrete Grönwall upper bound:

$$
\| x _ { S } ^ { \mathrm { r o l l } } ( K ) - x _ { T } ^ { \mathrm { r o l l } } ( K ) \| _ { 2 } \le \frac { \| \delta _ { v } \| _ { \infty } } { C } \big ( ( 1 + h C ) ^ { K } - 1 \big ) ,\tag{2}
$$

where $h > 0$ is thefixed Euler integration step size;for $C = 0 ,$ , the right-hand side is interpreted as $K h \| \delta _ { v } \| _ { \infty } .$ . For $K \bar { h } C \ll 1$ , this implies the few-step expansion $\begin{array} { r } { K h \| \tilde { \delta } _ { v } \| _ { \infty } + \mathcal { O } ( K ^ { 2 } h ^ { 2 } C \| \dot { \delta } _ { v } \| _ { \infty } ) } \end{array}$

The recurrence $\| e _ { k + 1 } \| _ { 2 } \leq ( 1 + h C ) \| e _ { k } \| _ { 2 } + h \| \delta _ { v } \| _ { \infty }$ , unrolled from $e _ { 0 } = 0$ , gives Eq. equation 2; Appendix F gives the proof. The implication for method design is limited but useful: reducing one-step mismatch is insufficient when the same error is repeatedly propagated, so training should also expose the student to short rollouts. The bound neither establishes a universal quadratic error law nor guarantees that our objectives minimize $C .$

## 3.3 STAGEWISE TRAINING

Figure 2 summarizes AlignNet for feature alignment, VAH for residual velocity correction, and LTE for draft scoring. A two-stage curriculum separates feature initialization from rollout supervision.

Setup and state extraction. Let $v _ { T }$ and v denote the teacher and student velocity fields used by the solver. Pointwise distillation uses $\mathcal { L } _ { v } = \mathbb { E } [ D ^ { - 1 } \| v _ { S } ( x _ { t } , t ) - v _ { T } ( x _ { t } , t ) \| _ { 2 } ^ { 2 } ]$ , where D is the latent dimension; Stage 2 applies this loss to the aligned student velocity defined below. The student backbone also exposes hidden tokens from its ordinary forward pass, which the LTE later uses for routing without an additional backbone evaluation.

![](images/a0048b4fcaed55458058d45de753fb4748564a8c73e18805742c7a79bf8e7d8a.jpg)  
Figure 2: TamedFlow overview. Left: AlignNet, VAH, and LTE attached to the student backbone. Middle: two-stage curriculum. Stage 1 is feature alignment and Stage 2 is rollout supervision. Right: inference routes drafts via the LTE; accepted drafts bypass the teacher, rejected drafts roll back and take one teacher step.

Stage 1: Feature alignment. Stage 1 freezes VAH and trains the backbone with velocity and intermediate-feature matching. AlignNet uses 2-layer MLP projectors to pair student layers {3, 7, 11} with teacher layers {9, 18, 27}, spanning early through late representations.

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \frac { 1 } { 3 } \sum _ { l \in \{ 3 , 7 , 1 1 \} } \frac { \Vert A _ { l } ( f _ { S } ^ { l } ) - f _ { T } ^ { \pi ( l ) } \Vert _ { 2 } ^ { 2 } } { \Vert f _ { T } ^ { \pi ( l ) } \Vert _ { 2 } ^ { 2 } } .\tag{3}
$$

Here $\pi ( 3 ) = 9 , \pi ( 7 ) = 1 8 ,$ , and $\pi ( 1 1 ) = 2 7$ . The Stage 1 objective ${ \mathcal { L } } _ { \mathrm { s t a g e 1 } } = { \mathcal { L } } _ { v } + \lambda _ { \mathrm { a l i g n } } { \mathcal { L } } _ { \mathrm { a l i g n } }$ initializes a student whose intermediate representation and velocity prediction are both anchored to the teacher before free-running supervision begins.

Stage 2: Rollout supervision. After Stage 1 stabilizes, we activate rollout-level supervision. To learn a residual velocity correction, we unfreeze the Velocity Alignment Head (VAH), a convolutional residual head initialized with a small output correction, so $v _ { S } ^ { \mathrm { a l i g n e d } } \approx v _ { S } \colon$

$$
v _ { S } ^ { \mathrm { a l i g n e d } } = v _ { S } + \mathrm { V A H } ( v _ { S } ; \phi ) .\tag{4}
$$

Stage 2 trains the model for short-horizon use. We compare aligned student and teacher velocities at teacher states and separately penalize the endpoints obtained when each model follows its own rollout:

$$
{ \mathcal { L } } _ { \mathrm { s t a g e 2 } } = { \mathcal { L } } _ { v } + \lambda _ { \mathrm { r o l l } } { \mathcal { L } } _ { \mathrm { r o l l } } ( K ) + \lambda _ { z } { \mathcal { L } } _ { z } + \lambda _ { \mathrm { c o n f } } { \mathcal { L } } _ { \mathrm { c o n f } } ,\tag{5}
$$

where $\begin{array} { r } { \mathcal { L } _ { \mathrm { r o l l } } ( K ) = \frac { 1 } { K D } \sum _ { k = 0 } ^ { K - 1 } \| v _ { S } ^ { \mathrm { a l i g n e d } } ( x _ { T } ^ { ( k ) } , t + k \Delta t ) - v _ { T } ( x _ { T } ^ { ( k ) } , t + k \Delta t ) \| _ { 2 } ^ { 2 } } \end{array}$ is teacher-state velocity anchoring, whereas $\bar { \mathcal { L } } _ { z } = D ^ { - 1 } \| x _ { S } ^ { \mathrm { r o l l } } ( K ) - x _ { T } ^ { \mathrm { r o l l } } ( K ) \| _ { 2 } ^ { 2 }$ is the free-running endpoint loss obtained by rolling each field from the same start state on its own states. The endpoint loss backpropagates through the unrolled student steps, while training samples $K$ uniformly from $\{ 1 , 2 , 3 , \dot { 4 } , 5 \dot  \}$ so the model is not specialized to one horizon. For any fixed teacher-state rollout, $\lVert \delta _ { v } \rVert _ { \infty } \leq \sqrt { K D \mathcal { L } _ { \mathrm { r o l l } } } ;$ this links the empirical loss to the forcing term in Proposition 3.1, but is not a uniform guarantee from the average training objective. The LTE is trained with

$$
\mathcal { L } _ { \mathrm { c o n f } } = \mathrm { M S E } \big ( \mathrm { L T E } ( \mathrm { s g } [ \bar { f } _ { S } ^ { \mathrm { l a s t } } , e _ { t _ { K - 1 } } , e _ { y } ] ) , \mathrm { s g } [ c ^ { \star } ] \big ) ,\tag{6}
$$

where $\bar { f } _ { S } ^ { \mathrm { l a s t } }$ denotes mean-pooled hidden tokens from the last student forward in the draft and sg[·] denotes stop-gradient. Thus the LTE learns from the same type of signal available online, while its loss does not perturb the student backbone.

## 3.4 SELF-SPECULATIVE EXECUTION VIA CONFIDENCE ROUTING

For a K-step draft, the target score $c ^ { \star }$ combines endpoint rollout deviation with the velocity error at the shared start state:

$$
c ^ { \star } = \exp \Bigl ( - \alpha \frac { \| x _ { S } ^ { \mathrm { r o l l } } ( K ) - x _ { T } ^ { \mathrm { r o l l } } ( K ) \| _ { 2 } ^ { 2 } } { D T _ { z } } - ( 1 - \alpha ) \frac { \| v _ { S } ^ { \mathrm { a l i g n e d } } ( x _ { t } , t ) - v _ { T } ( x _ { t } , t ) \| _ { 2 } ^ { 2 } } { D T _ { v } } \Bigr ) ,\tag{7}
$$

with latent dimension D and temperatures T<sub>z</sub>, T<sub>v</sub>. The score decreases as either discrepancy grows; it is a regression target, not a calibrated probability that a draft preserves sample quality. At inference the LTE predicts this score from student features alone.

LTE inference. For a clipped draft of length m, the LTE uses mean-pooled tokens and the timestep embedding from the last drafting forward at $t _ { i + m - 1 }$ , together with the condition representation:

$$
h _ { \mathrm { L T E } } = \sigma ( g _ { f } ) W _ { 1 } \bar { f } _ { S } + \sigma ( g _ { t } ) W _ { 2 } e _ { t _ { i + m - 1 } } + \sigma ( g _ { y } ) W _ { 3 } e _ { y } , \qquad c = \sigma \big ( w ^ { \top } \mathrm { M L P } ( h _ { \mathrm { L T E } } ) + b \big ) .\tag{8}
$$

The LTE has no explicit horizon input, so its score is an empirical routing signal rather than a horizon-specific error guarantee.

Routing. The student drafts K steps, clipped at the schedule boundary. If $c \geq \tau$ , the sampler accepts the entire draft and bypasses the teacher; otherwise it discards the tentative states, restores the pre-draft state, and advances one step with the teacher. Appendix H gives the full algorithm.

## 4 EXPERIMENTS

Table 1: ImageNet 256 × 256 class-conditional generation with DiT-XL/2.
<table><tr><td>Method</td><td>Latency (s) ↓</td><td>Speedup ↑</td><td>FLOPs (T) ↓</td><td>FID↓</td><td>IS↑</td></tr><tr><td>DDIM-50 Teacher</td><td>1.044</td><td>1.00×</td><td>23.74</td><td>2.35</td><td>237.62</td></tr><tr><td>Pure Student</td><td>0.180</td><td>5.80×</td><td>2.30</td><td>5.56 (+3.21)</td><td>195.07 (-42.55)</td></tr><tr><td colspan="6">Full-Scale Few-Step Distillation</td></tr><tr><td>Progressive Distill. (step=4) (Salimans &amp; Ho, 2022)</td><td>0.068</td><td>15.35×</td><td>0.474</td><td>10.25 (+7.90)</td><td>181.58 (-56.04)</td></tr><tr><td>LCM (step=4) (Luo et al., 2023)</td><td>0.072</td><td>14.50×</td><td>0.476</td><td>6.81 (+4.46)</td><td>202.73 (-34.89)</td></tr><tr><td>ARD (step=4) (Kim et al., 2025) (N = 6)</td><td>0.074</td><td>14.11×</td><td>0.480</td><td>4.32 (+1.97)</td><td>209.03 (-28.59)</td></tr><tr><td colspan="6">1.5×-2.0× Speedup Range</td></tr><tr><td>DDIM-25</td><td>0.558</td><td>1.87×</td><td>11.87</td><td>3.21 (+0.86)</td><td>229.91 (-7.71)</td></tr><tr><td>∆-DiT (Chen et al., 2024) (N = 2)</td><td>0.684</td><td>1.53×</td><td>18.04</td><td>2.78 (+0.43)</td><td>223.47 (-14.15)</td></tr><tr><td>TaylorSeer (Liu et al., 2025a) (N = 2)</td><td>0.630</td><td>1.66×</td><td>12.35</td><td>2.47 (+0.12)</td><td>230.21 (-7.41)</td></tr><tr><td>FoCa (Zheng et al., 2026) (N = 2)</td><td>0.634</td><td>1.65×</td><td>12.35</td><td>2.37 (+0.02)</td><td>233.70 (-3.92)</td></tr><tr><td>SpecDiff (Pan et al., 2026)</td><td>0.644</td><td>1.62×</td><td>14.58</td><td>2.39 (+0.04)</td><td>235.18 (-2.44)</td></tr><tr><td>TamedFlow (τ = 0.85)</td><td>0.602</td><td>1.73×</td><td>8.53</td><td>2.33 (-0.02)</td><td>236.15 (-1.47)</td></tr><tr><td colspan="6">2.0×-2.5× Speedup Range</td></tr><tr><td>DDIM-20</td><td>0.442</td><td>2.36×</td><td>9.49</td><td>3.87 (+1.52)</td><td>219.40 (-18.22)</td></tr><tr><td>FORA (Selvaraju et al., 2024) (N = 3)</td><td>0.482</td><td>2.17×</td><td>8.58</td><td>3.65 (+1.30)</td><td>228.14 (-9.48)</td></tr><tr><td>TaylorSeer (Liu et al., 2025a) (N = 3)</td><td>0.477</td><td>2.19×</td><td>8.56</td><td>2.73 (+0.38)</td><td>229.36 (-8.26)</td></tr><tr><td>FoCa (Zheng et al., 2026) (N = 3)</td><td>0.466</td><td>2.24×</td><td>8.56</td><td>2.65 (+0.30)</td><td>230.56 (-7.06)</td></tr><tr><td>SpeCa (Liu et al., 2025b)</td><td>0.504</td><td>2.07×</td><td>4.76</td><td>2.78 (+0.43)</td><td>229.74 (-7.88)</td></tr><tr><td>TamedFlow  $( \tau = 0 . 8 0 )$ </td><td>0.464</td><td>2.25×</td><td>6.83</td><td>2.58 (+0.23)</td><td>233.43 (-4.19)</td></tr></table>

## 4.1 EXPERIMENTAL SETUP

Models and datasets. The primary setting distills a 675M DiT-XL/2 teacher $\mathcal { T } _ { \phi }$ (Peebles & Xie, 2023) into a 130M DiT-B/2 student $B _ { \theta }$ on class-conditional ImageNet at 256 × 256. We evaluate resolution scaling on ImageNet 512 × 512, text conditioning on MS-COCO, and video generation with Latte on FaceForensics. For COCO, a shared 19.9M text-conditioning adapter maps 77-token CLIP ViT-L/14 features into the DiT conditioning space; the teacher and student use this shared adapter while the ImageNet-trained VAH and LTE remain fixed. For Latte, the student, AlignNet, VAH, and LTE are retrained for the evaluated video-DiT setting. Complete training schedules, solver settings, and adaptation protocols are provided in Appendices A and E.

Metrics and reporting. On ImageNet, we report FID-50K and Inception Score (IS); on MS-COCO, FID-30K and CLIP-Score computed with OpenAI CLIP ViT-L/14; and on FaceForensics, FVD and video Inception Score. Image and text-to-image wall-clock latency and speedup are measured on a single A800-80GB GPU at batch size one and averaged over 10 runs. All baseline rows are measured in our local implementations under the same reporting protocol. Speedup is normalized to the corresponding teacher, and the LTE accept rate is the fraction of routing events that retain the student draft. FLOPs count backbone evaluations, whereas latency also includes VAH, LTE, rollback, and solver overhead.

Boxer  
Tabby  
Pizza  
Bookcase  
Horn  
Airliner  
Schooner  
Barrel  
![](images/34b6847fe20dbdd0ac387f28de446d4fa6a8a8c3a02fd9f2a4fec0ead2155c49.jpg)  
Figure 3: Qualitative comparison on ImageNet 256 × 256.

Baselines. We compare against schedule-reduction references (DDIM-25 and DDIM-20), full-scale few-step distillation methods (Progressive Distillation (Salimans & Ho, 2022), LCM (Luo et al., 2023), and ARD (Kim et al., 2025)), and adaptive or speculative methods (∆-DiT (Chen et al., 2024), TaylorSeer (Liu et al., 2025a), FoCa (Zheng et al., 2026), FORA (Selvaraju et al., 2024), SpeCa (Liu et al., 2025b), and SpecDiff (Pan et al., 2026)). Adaptive and speculative baselines are re-implemented in the matched DiT-XL/2 pipeline and measured locally under the same latency protocol. Few-step distillation baselines use the full teacher architecture without parameter compression. Appendix B details the comparison scope and accounting assumptions.

## 4.2 MAIN RESULTS

Quality–latency operating points. Table 1 compares complete systems across speedup regimes. Four-step distillation baselines attain aggressive 14×–15× acceleration at the cost of severe FID degradation. TamedFlow instead targets moderate acceleration with tight quality retention: at τ = 0.85, it matches the DDIM-50 teacher within 0.02 FID (2.33 vs. 2.35) at 1.73× speedup; at default τ = 0.80, it reaches FID 2.58 at 2.25× speedup (0.464 s).

In the 2.0× to 2.5× range under nearly matched latencies around 0.46 to 0.48 seconds, TamedFlow achieves better quality than FoCa at FID 2.65 and TaylorSeer at FID 2.73, alongside a higher Inception Score. This represents a quality advantage at matched latency rather than a throughput claim from millisecond-scale differences. Although SpeCa reports lower backbone FLOPs of 4.76 T, its FID is 0.20 worse, and backbone FLOPs omit runtime overheads like rollback, VAH, and LTE that are fully captured by measured latency. Thus, TamedFlow’s edge lies in its empirical Pareto trade-off for the deployed system, not a FLOPs-only claim.

Qualitative study. Figure 3 compares generated samples under matched speedup (2.1×–2.25×). TamedFlow preserves fine textures and object boundaries more cleanly across diverse classes. For example, it avoids the over-smoothed fur patterns on Tabby and Boxer seen in competing adaptive methods, and retains clear structural separation between the metal hoops and wood grain on Barrel. These visual examples align with aggregate FID trends, while the following ablations isolate the underlying mechanisms.

## 4.3 COMPONENT ABLATION

Table 2 isolates the contribution of each architectural module and training objective at draft length K = 3 and routing threshold τ = 0.80.

Modules and training objectives. AlignNet initializes student representations from intermediate teacher layers; removing it drops acceptance to 46.2% and raises FID to 2.95 due to suboptimal representation grounding. Disabling the VAH reduces acceptance to 39.3% and increases FID to 3.23, reflecting its role in absorbing local residual velocity mismatch (without claiming an explicit theoretical reduction in $\Delta J )$ . Without the LTE, unrouted student execution reaches 5.80× speedup but degrades severely to FID 5.56 in the absence of corrective recovery. For training losses, omitting rollout supervision $\mathcal { L } _ { \mathrm { r o l l } }$ weakens intermediate velocity consistency and increases FID to 2.71. Removing the endpoint loss $\mathcal { L } _ { z }$ has the most severe impact, dropping acceptance to 28.1% and inflating FID to 4.15; its unrolled multi-step gradient directly penalizes compounding trajectory drift during autonomous rollouts. Removing the 10-epoch warmup on $\mathcal { L } _ { z }$ yields FID 3.29 due to early optimization shocks before feature representations stabilize.

Table 2: Ablation studies on ImageNet $2 5 6 \times 2 5 6$ at $K = 3$ and $\tau = 0 . 8 0 .$ Speedup is relative to the 50-step teacher baseline.
<table><tr><td>Study</td><td>Variant</td><td>Accept % ↑</td><td>Speedup ↑</td><td>FID↓</td><td>IS↑</td></tr><tr><td rowspan="4">Component</td><td>Full TamedFlow</td><td>63.6</td><td>2.25×</td><td>2.58</td><td>233.43</td></tr><tr><td>w/o AlignNet</td><td>46.2</td><td>1.87×</td><td>2.95</td><td>211.34</td></tr><tr><td>w/o VAH</td><td>39.3</td><td>1.63×</td><td>3.23</td><td>205.71</td></tr><tr><td>w/o LTE</td><td></td><td>5.80×</td><td>5.56</td><td>195.07</td></tr><tr><td rowspan="4">Training</td><td>Full TamedFlow</td><td>63.6</td><td>2.25×</td><td>2.58</td><td>233.43</td></tr><tr><td>w/o  ${ \mathcal L } _ { \mathrm { r o l l } }$ </td><td>54.2</td><td>2.02×</td><td>2.71</td><td>230.15</td></tr><tr><td>w/o  $\mathcal { L } _ { z }$  warmup</td><td>46.2</td><td>1.81×</td><td>3.29</td><td>203.50</td></tr><tr><td>w/o  $\mathcal { L } _ { z }$ </td><td>28.1</td><td>1.45×</td><td>4.15</td><td>198.80</td></tr></table>

![](images/a84260eff089ed9f6bae81d497337c34a9e1569a2a90796fe89304dd26366237.jpg)

![](images/935d287ab811d9ab1a5b1c3601fa9b50cc023c6b3e53f64d7e0aff98a868074b.jpg)  
Figure 4: LTE routing behavior across generation. Left: confidence across sampling steps; the upper axis uses a noise-level coordinate, distinct from forward flow time. Right: score distributions of accepted and rejected drafts under $\tau = 0 . 8 0$

Routing-policy controls. Table 9 compares policies using the same trained student, VAH, Euler grid, and guidance. At nearly identical latency (0.465 versus 0.464 s), LTE improves FID over random draft acceptance from 4.05 to 2.58. The random control enforces eight teacher updates per trajectory; LTE retains unconstrained $\tau = 0 . 8 0$ routing. Thus this comparison supports learned routing at nearly matched latency, without establishing equal teacher usage. Static controls are faster but have worse FID. Appendix I.1 gives the execution and call-accounting details.

## 4.4 ROUTING DYNAMICS AND SENSITIVITY

Draft length K and threshold τ balance speedup against fidelity (Appendix Figure 10). Speedup peaks at $K = 3$ because longer drafts compound drift and increase rollback costs upon rejection. As shown in Figure 4, LTE confidence remains high in the mid-noise regime but drops near the plotted noise level 0.22, invoking more teacher rescues late in sampling. The distribution shows clear operational separation between accepted and rejected drafts at the chosen operating point $\tau = 0 . 8 0$

In the reference trace at $K = 3$ and $\tau = 0 . 8 0 $ , 14 of 22 routing events are accepted, allowing student rollouts to cover 42 of the 50 integration steps. Detailed component profiling reveals that teacher rescue accounts for 71.9% of wall-clock latency, whereas student rollouts take 14.9%, rejections with state rollback consume 12.7%, and LTE confidence scoring incurs only 0.5%. The lightweight estimator itself introduces negligible computational latency, but routing accuracy remains paramount: false acceptances allow subtle draft errors to compound and degrade fidelity, while false rejections trigger expensive teacher interventions that erode throughput. This concrete latency breakdown also explains why measured end-to-end speedup does not strictly match backbone FLOPs, as runtime costs from state rollback, synchronization, and auxiliary modules are omitted from nominal FLOP totals.

## 4.5 CROSS-SETTING EVALUATION

Table 3: MS-COCO 30K 256 × 256 text-to-image generation using fixed-step DDIM. TaylorSeer, SpeCa, and FoCa are re-implemented within our DiT-XL/2 pipeline; N denotes each method’s acceleration setting rather than TamedFlow’s draft horizon K.
<table><tr><td>Method</td><td>Latency (s) ↓</td><td>Speedup ↑</td><td>FID↓</td><td>CLIP-S ↑</td></tr><tr><td>Teacher DDIM-50</td><td>1.184</td><td>1.00×</td><td>13.54</td><td>30.2</td></tr><tr><td>Teacher DDIM-25</td><td>0.622</td><td>1.90×</td><td>16.84</td><td>28.9</td></tr><tr><td>Pure Student</td><td>0.203</td><td>5.83×</td><td>19.30</td><td>27.8</td></tr><tr><td>TaylorSeer (N = 3)</td><td>0.580</td><td>2.04×</td><td>15.47</td><td>29.1</td></tr><tr><td>SpeCa</td><td>0.561</td><td>2.11×</td><td>16.63</td><td>28.4</td></tr><tr><td>FoCa (N = 3)</td><td>0.565</td><td>2.09×</td><td>16.27</td><td>28.5</td></tr><tr><td>TamedFlow  $( \tau = 0 . 8 0 )$ </td><td>0.603</td><td>1.96×</td><td>14.86</td><td>29.6</td></tr><tr><td>TamedFlow  $( \tau = 0 . 7 5 )$ </td><td>0.551</td><td>2.14×</td><td>15.52</td><td>29.2</td></tr></table>

On MS-COCO, the teacher and student share a trained 19.9M text adapter while the ImageNet-trained VAH and LTE remain frozen. As reported in Table 3, TamedFlow reaches FID 14.86 and CLIP-Score 29.6 at 1.96× speedup at $\tau = 0 . 8 0 $ , and 2.14× speedup with FID 15.52 at $\tau = 0 . 7 5$ , outperforming SpeCa and FoCa at comparable latency. This supports transfer of the frozen alignment modules under condition adaptation rather than unadapted zero-shot deployment. Tightening τ improves FID by 0.66 and CLIP-Score by 0.4 for 52 ms latency, verifying that fidelity recovery stems from teacher intervention rather than student capacity.

We further evaluate spatial and temporal scaling. On ImageNet $5 1 2 \times 5 1 2 ,$ , expanding the latent grid fourfold under the same backbone, TamedFlow achieves FID 3.39 at 2.41× speedup in 1.594 s, outperforming FoCa at FID 3.47 with 2.36× speedup as reported in Appendix C. On the Latte video DiT retrained on FaceForensics at 256 × 256 resolution across 16 frames in Table 4, TamedFlow attains FVD 225.10 and IS 68.90 at 2.22× speedup, surpassing FoCa at FVD 248.30 and TaylorSeer at FVD 253.10. These results confirm that the draft and rescue mechanism transfers reliably across resolutions, conditions, and video modalities.

Table 4: Latte video-DiT transfer on FaceForensics (256 × 256, 16 frames).
<table><tr><td>Method</td><td>Speedup</td><td>FVD↓</td><td>IS↑</td></tr><tr><td>Teacher (Latte-XL)</td><td>1.00×</td><td>208.50</td><td>71.20</td></tr><tr><td>FoCa</td><td>2.06×</td><td>248.30</td><td>66.80</td></tr><tr><td>TaylorSeer</td><td>2.10×</td><td>253.10</td><td>65.90</td></tr><tr><td>TamedFlow</td><td>2.22×</td><td>225.10</td><td>68.90</td></tr></table>

Deployment cost. TamedFlow requires offline training and maintains the teacher, student, VAH, and LTE co-resident in memory, increasing peak allocated VRAM from 2.570 to 3.075 GB (∼0.5 GB overhead). This footprint reflects the deliberate choice to retain a high-capacity teacher for selective rescue rather than discarding it. Headline latency is benchmarked for batch-one latent integration without VAE decoding, characterizing interactive generation rather than batched serving throughput. Teacher offloading and weight quantization represent orthogonal opportunities to mitigate co-residency overhead in production.

## 5 CONCLUSION

In this work, we introduced TamedFlow, an acceleration framework that reconciles parameter compression with adaptive execution for Diffusion Transformers. By training a compact student for short-horizon trajectory agreement and employing asymmetric routing with selective teacher rescue, TamedFlow achieves FID 2.58 at 2.25× speedup over DDIM-50 on ImageNet 256 × 256, with transfer evaluated across resolutions, text conditioning, and video. Our findings show that pointwise alignment can leave compounding trajectory divergence during autonomous drafting, creating the Combination Gap. Controlled ablations support rollout supervision, and routing controls favor LTE over random decisions at nearly matched latency, suggesting that compression and adaptive inference can be made compatible when the compact model is trained as a multi-step drafter and the sampler retains a recovery path back to the teacher.

## REFERENCES

Peiliang Cai, Jiacheng Liu, Haowen Xu, Xinyu Wang, Chang Zou, and Linfeng Zhang. Lesa: Learnable stage-aware predictors for diffusion model acceleration. arXiv preprint arXiv:2602.20497, 2026.

Pengtao Chen, Mingzhu Shen, Peng Ye, Jianjian Cao, Chongjun Tu, Christos-Savvas Bouganis, Yiren Zhao, and Tao Chen. δ-dit: A training-free acceleration method tailored for diffusion transformers, 2024. URL https://arxiv.org/abs/2406.01125.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, Zico Kolter, and Kaiming He. Mean flows for onestep generative modeling. Advances in Neural Information Processing Systems, 38:75460–75482, 2026.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Yushi Huang, Zining Wang, Ruihao Gong, Jing Liu, Xinjie Zhang, Jinyang Guo, Xianglong Liu, and Jun Zhang. Harmonica: Harmonizing training and inference for better feature caching in diffusion transformer acceleration. In ICML, 2025.

Yeongmin Kim, Sotiris Anagnostidis, Yuming Du, Edgar Schönfeld, Jonas Kohler, Markos Georgopoulos, Albert Pumarola, Ali Thabet, and Artsiom Sanakoyeu. Autoregressive distillation of diffusion transformers. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15745–15756. IEEE, 2025.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International conference on machine learning, pp. 19274–19286. PMLR, 2023.

Haosen Li, Wenshuo Chen, Shaofeng Liang, Lei Wang, Haozhe Jia, and Yutao Yue. Guided path sampling: Steering diffusion models back on track with principled path guidance. In Companion Proceedings ofthe ACM Web Conference 2026, pp. 1295–1300, 2026.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Jiacheng Liu, Chang Zou, Yuanhuiyi Lyu, Junjie Chen, and Linfeng Zhang. From reusing to forecasting: Accelerating diffusion models with taylorseers. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15853–15863. IEEE, 2025a.

Jiacheng Liu, Chang Zou, Yuanhuiyi Lyu, Fei Ren, Shaobo Wang, Kaixin Li, and Linfeng Zhang. Speca: Accelerating diffusion transformers with speculative feature caching. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pp. 10024–10033, 2025b.

Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In The twelfth international conference on learning representations, 2023.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. Machine Intelligence Research, 22 (4):730–751, 2025.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

Yihong Luo, Tianyang Hu, Jiacheng Sun, Yujun Cai, and Jing Tang. Learning few-step diffusion models by trajectory distribution matching. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17719–17728. IEEE, 2025.

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pp. 23–40. Springer, 2024a.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15762–15772. IEEE, 2024b.

Jiayi Pan, Jiaming Xu, Yongkang Zhou, and Guohao Dai. Specdiff: Accelerating diffusion model inference with self-speculation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 8233–8241, 2026.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4172–4182. IEEE, 2023.

Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. Advances in neural information processing systems, 37:117340–117362, 2024.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=TIdIXIpzhoI.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pp. 87–103. Springer, 2024.

Pratheba Selvaraju, Tianyu Ding, Tianyi Chen, Ilya Zharkov, and Luming Liang. Fora: Fast-forward caching in diffusion transformer acceleration. arXiv preprint arXiv:2407.01425, 2024.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. URL https://openreview.net/ forum?id=St1giarCHLP.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.

Zhicong Tang, Jianmin Bao, Dong Chen, and Baining Guo. Diffusion models without classifier-free guidance. arXiv preprint arXiv:2502.12154, 2025.

Felix Wimbauer, Bichen Wu, Edgar Schoenfeld, Xiaoliang Dai, Ji Hou, Zijian He, Artsiom Sanakoyeu, Peizhao Zhang, Sam Tsai, Jonas Kohler, et al. Cache me if you can: Accelerating diffusion models through block caching. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6211–6220. IEEE, 2024.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6613–6623. IEEE, 2024.

Wangbo Zhao, Yizeng Han, Zhiwei Tang, Jiasheng Tang, Pengfei Zhou, Kai Wang, Bohan Zhuang, Zhangyang Wang, Fan Wang, and Yang You. RAPID\$^3\$: Tri-level reinforced acceleration policies for diffusion transformer. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=sQ0g6EkpF7.

Jianbin Zheng, Minghui Hu, Zhongyi Fan, Chaoyue Wang, Changxing Ding, Dacheng Tao, and Tat-Jen Cham. Trajectory consistency distillation: Improved latent consistency distillation by semi-linear consistency function with trajectory mapping. arXiv preprint arXiv:2402.19159, 2024.

Shikang Zheng, Liang Feng, Xinyu Wang, Qinming Zhou, Peiliang Cai, Chang Zou, Jiacheng Liu, Yuqi Lin, Junjie Chen, Yue Ma, et al. Forecast then calibrate: Feature caching as ode for efficient diffusion transformers. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 13449–13457, 2026.

Mingyuan Zhou, Yi Gu, Huangjie Zheng, Liangchen Song, Guande He, Yizhe Zhang, Wenze Hu, and Yinfei Yang. Score distillation of flow matching models. arXiv preprint arXiv:2509.25127, 2025.

Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. Accelerating diffusion transformers with token-wise feature caching. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=yYZbZGo4ei.

Chang Zou, Changlin Li, Songtao Liu, Zhao Zhong, Kailin Huang, and Linfeng Zhang. Disca: Accelerating video diffusion transformers with distillation-compatible learnable feature caching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4590–4601, June 2026a.

Chang Zou, Shikang Zheng, Evelyn Zhang, Runlin Guo, Haohang Xu, Zhengyi Shi, Conghui He, Xuming Hu, and Linfeng Zhang. Rethinking token-wise feature caching: Accelerating diffusion transformers with dual feature caching. IEEE Transactions on Image Processing, 2026b.

## Appendix A Training Hyperparameters

Table 5 summarizes the training hyperparameters for the two stages of TamedFlow.

Table 5: Training hyperparameters for TamedFlow.
<table><tr><td>Component</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>Optimizer</td><td colspan="2"> $\mathrm { A d a m W } \left( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 \right)$ </td></tr><tr><td>Base learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 4 }$  0.0</td></tr><tr><td>Weight decay</td><td colspan="2"></td></tr><tr><td>Learning rate schedule</td><td colspan="2">cosine decay with warmup 5</td></tr><tr><td>Warmup epochs</td><td colspan="2"></td></tr><tr><td>Max epochs</td><td colspan="2"></td></tr><tr><td>Batch size (per GPU)</td><td colspan="2">32</td></tr><tr><td>Global batch size</td><td colspan="2">256</td></tr><tr><td>Gradient clipping</td><td colspan="2">1.0</td></tr><tr><td>Loss terms</td><td colspan="2"></td></tr><tr><td>Velocity loss  $\mathcal { L } _ { v }$ </td><td colspan="2"></td></tr><tr><td>Alignment loss  $\mathcal { L } _ { \mathrm { a l i g n } }$ </td><td colspan="2"> $\lambda = 0 . 2 5$ </td></tr><tr><td>Rollout loss  $\mathcal { L } _ { \mathrm { r o l l } } ( \check { K } )$ </td><td colspan="2"></td></tr><tr><td></td><td colspan="2"></td></tr><tr><td>Endpoint loss  $\mathcal { L } _ { z }$  Confidence loss</td><td colspan="2">λz (10-epoch warmup to 30.0)</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { c o n f } }$ </td><td colspan="2"> $\lambda = 1 . 0$ </td></tr><tr><td>Model details</td><td colspan="2">DiT-B/2 (12 layers)</td></tr><tr><td>Student architecture</td><td colspan="2"></td></tr><tr><td>Student parameters</td><td colspan="2">130M</td></tr><tr><td>Teacher architecture</td><td colspan="2">DiT-XL/2 (28 layers)</td></tr><tr><td>Teacher parameters</td><td colspan="2">675M</td></tr><tr><td>AlignNet architecture</td><td colspan="2">3 independent MLPs, 768→ 1152→ 1152</td></tr><tr><td>VAH architecture</td><td colspan="2">3-layer Conv2D residual head, hidden=256</td></tr><tr><td>LTE architecture</td><td colspan="2">3 projections; MLP 1024 → 1024 → 256 → 1</td></tr><tr><td>LTE parameters</td><td colspan="2">~4M (3% of student)</td></tr><tr><td>Inference hyperparameters</td><td colspan="2"></td></tr><tr><td>Draft length  $K$ </td><td colspan="2">3 (training:  $K \sim \mathcal { U } ( 1 , 5 ) )$ </td></tr><tr><td>Temperature  $T _ { z }$ </td><td colspan="2">0.5</td></tr><tr><td>Temperature  $T _ { v }$ </td><td colspan="2">0.1</td></tr><tr><td>Mixing weight α</td><td colspan="2">0.7</td></tr><tr><td>Default τ</td><td colspan="2">0.8</td></tr><tr><td>LTE training</td><td colspan="2"></td></tr><tr><td>Gradient isolation</td><td colspan="2">√(stop-gradient)</td></tr><tr><td></td><td colspan="2"></td></tr><tr><td>LTE warmup epochs</td><td colspan="2">5</td></tr><tr><td>LTE learning rate</td><td colspan="2"> $1 \times 1 0 ^ { - 4 }$ </td></tr></table>

## Appendix B Baseline Comparison Scope

Table 1 compares end-to-end operating points, not isolated routers. The adaptive baselines retain the 675M teacher, whereas TamedFlow combines a 130M draft student with selective teacher rescue; compression, training, solver choice, routing, and rollback therefore all contribute to the measured result. All baseline entries, including SpecDiff, are evaluated in our local implementations under the stated protocol. TamedFlow FLOPs are backbone-call estimates; end-to-end latency additionally includes VAH, LTE, solver, and rollback overhead.

Table 11 serves a different purpose. It is a quality diagnostic in which adaptive reuse is attached directly to the compressed student, and therefore supports the motivating Combination Gap rather than a latency-matched routing comparison.

## Appendix C Higher-Resolution Evaluation

We evaluate spatial scaling on ImageNet at $5 1 2 \times 5 1 2$ with DiT-XL/2, DDIM-50, and one A800 GPU. The comparison holds the resolution and protocol fixed and tests whether the primary quality–latency ordering survives the larger latent grid.

Table 6: ImageNet $5 1 2 \times 5 1 2$ evaluation with DiT-XL/2 using fixed-step DDIM.
<table><tr><td>Method</td><td>Latency (s) ↓</td><td>Speedup ↑</td><td>FID↓</td><td>IS↑</td></tr><tr><td>Teacher (DiT-XL/2, 675M)</td><td>3.842</td><td>1.00×</td><td>2.87</td><td>235.10</td></tr><tr><td>TaylorSeer  $( N = 3 )$ </td><td>1.715</td><td>2.24×</td><td>3.76</td><td>226.75</td></tr><tr><td>FoCa  $( N = 3 )$ </td><td>1.628</td><td>2.36×</td><td>3.47</td><td>228.10</td></tr><tr><td>TamedFlow (τ = 0.80)</td><td>1.594</td><td>2.41×</td><td>3.39</td><td>230.85</td></tr></table>

## Appendix D Multi-Seed Evaluation

We generate 50K ImageNet samples for each of three evaluation seeds using fixed checkpoints. These repeats quantify sampling variability, not independent training-run variability; every method follows the same evaluation protocol.

Table 7: Three-seed ImageNet evaluation (mean ± standard deviation over seeds).
<table><tr><td>Method</td><td>Speedup</td><td>FID</td><td>IS</td></tr><tr><td>DDIM-50 Teacher</td><td>1.00×</td><td> $2 . 3 6 \pm 0 . 0 2$ </td><td> $2 3 7 . 2 4 \pm 0 . 4 8$ </td></tr><tr><td>TaylorSeer  $( N = 3 )$ </td><td>2.19×</td><td> $2 . 7 1 \pm 0 . 0 4$ </td><td> $2 2 9 . 1 3 \pm 0 . 6 2$ </td></tr><tr><td>FoCa  $( N = 3 )$ </td><td>2.21×</td><td> $2 . 6 4 \pm 0 . 0 3$ </td><td> $2 3 1 . 2 4 \pm 0 . 5 8$ </td></tr><tr><td>TamedFlow,  $\tau = 0 . 8 0$ </td><td>2.26×</td><td> $2 . 5 6 \pm 0 . 0 3$ </td><td> $2 3 3 . 8 9 \pm 0 . 5 1$ </td></tr></table>

## Appendix E Efficiency and Transfer Accounting

Deployment footprint. In a controlled A800, batch-size-2, FP32, 50-step latent-only profile with VAE decoding excluded, peak allocated VRAM is 2.570 GB for the teacher, 2.580 GB for FoCa $( N = 3 )$ , 3.066 GB for TaylorSeer $( N = 3 , O = 3 )$ , and 3.075 GB for TamedFlow. Co-deploying the student and LTE therefore adds about 0.5 GB relative to the teacher-only profile. For the ImageNet and MS-COCO image experiments, all methods use classifier-free guidance at scale 3.5. The teacher FLOP reference includes the complete conditional and unconditional CFG evaluation, while the CFGdistilled student reference counts one student forward; this convention defines the teacher/student references and the TamedFlow backbone estimates.

Routing accounting. In the reference trace at $\tau = 0 . 8 0 ,$ 14 of 22 routing events are accepted and 8 are rejected, giving a 63.6% event acceptance rate. Since each accepted event advances $K = 3$ Student steps, accepted drafts cover $4 2 / 5 0 = 8 4 \%$ of the nominal trajectory; the remaining 8 steps use Teacher rescue. The separately measured component profile records 0.07075 s for accepted-draft buckets (14.9%), 0.06031 s for rejected-draft buckets including rollback (12.7%), 0.34142 s for Teacher rescue (71.9%), and 0.00237 s for LTE scoring (0.5%). These buckets sum to 0.47485 s (up to rounding). They are event-level timing buckets from the profiling run, so dividing a bucket by the number of backbone calls would incorrectly attribute rollback and routing overhead to an individual forward. The component profile and the headline latency measurement were collected in different runs: the 0.464 s headline is the separately measured ten-run average. We therefore report both traces without using one to reconstruct the other.

Backbone FLOP accounting. The 50-step references imply $f _ { S } = 2 . 3 0 / 5 0 = 0 . 0 4 6 \mathrm { ~ T ~ }$ per CFGdistilled student forward and $\mathbf { \bar { \rho } } f _ { T } = 2 3 . 7 4 / 5 0 = 0 . 4 7 4 8 \mathrm { T }$ per teacher update including both CFG branches. With the executed call counts defined in Appendix I.1,

$$
F _ { \mathrm { b a c k b o n e } } = N _ { S } f _ { S } + N _ { T } f _ { T } .\tag{E.1}
$$

For the reference execution with 14 accepted and eight rejected full-length drafts, $N _ { S } = 6 6 , N _ { T } = 8$ and $N _ { \mathrm { L T E } } = 2 2$ , yielding $6 6 ( 0 . 0 4 6 ) + \hat { 8 } ( 0 . 4 7 4 8 ) = 6 . \hat { 8 } 3 4 4 \mathrm { T }$ . The full-length 13-accept/11-reject reference gives $7 2 ( 0 . 0 4 6 ) + 1 1 ( 0 . 4 7 4 8 ) = 8 . 5 3 4 8$ T. The rounded entries in Table 1 are callbased estimates for these reference configurations, not evaluation-set means. In contrast, a static eight-teacher policy has 42 student forwards and a backbone subtotal of $4 2 ( 0 . 0 4 6 ) + 8 ( 0 . 4 7 4 8 ) =$ 5.7304 T under the same accounting. VAH, LTE, rollback, and solver arithmetic are excluded from these backbone subtotals and included in measured latency. The 22 LTE calls do not require 22 additional student forwards because their features are reused from drafting. Logical conditional and unconditional teacher evaluations may be batched; eight guided teacher updates account for both branches, whereas the CFG-distilled student uses one branch per update. Backbone FLOPs alone do not predict measured latency.

Latte video protocol. We retrain the student, AlignNet, VAH, and LTE for the evaluated Latte video-DiT setting while keeping the TamedFlow training and inference configuration consistent with the primary experiments: Stage 1 for 10 epochs, Stage 2 for 80 epochs, ${ \check { K } } = 3$ at inference, randomized $K \sim \mathcal { U } ( 1 , 5 )$ during rollout training, and the same loss weights and threshold as ImageNet. The original Latte teacher/student block configurations are retained; no width, depth, or temporal-factorization search is performed. Spatial and temporal tokens are treated as one spatiotemporal token sequence for AlignNet matching and LTE mean pooling, so the LTE adds no separate temporal backbone. Evaluation uses FaceForensics clips at $2 5 6 \times 2 5 6$ resolution with 16 frames. We use the same clips and evaluation protocol for all methods, batch size 1, deterministic 50-step sampling, and latency measured end-to-end on 8 A800-80GB GPUs. VAE decoding is excluded from the reported latency. The video table reports FVD and video Inception Score together with the teacher-normalized speedup.

MS-COCO adaptation protocol. The shared 19.9M text-conditioning adapter is trained jointly with the teacher and student for 20 epochs (5 Stage 1 and 15 Stage 2). The implementation uses AdamW with learning rate $1 0 ^ { - 4 }$ for the text adapter, a 0.1 learning-rate multiplier for the backbone parameters, weight decay 0.01, text-dropout probability 0.05, gradient clipping at 1.0, and EMA decay 0.9999. The DiT backbones are frozen except for the text-conditioning parameters; the LTE and VAH are frozen during this adapter-only adaptation. The adapter maps 77-token CLIP ViT-L/14 features into the DiT conditioning width and is shared by teacher and student.

## Appendix F Proof of the K-Step Trajectory Alignment Bound

Let $x _ { S } ^ { ( k ) }$ and $x _ { T } ^ { ( k ) }$ denote the states of the student and teacher at step k during a K-step sequential Euler rollout, starting from the identical initial state $x _ { S } ^ { ( 0 ) } = x _ { T } ^ { ( 0 ) } = x _ { t }$ . We assume a fixed ODE solver step size h. With time arguments suppressed at the common $t _ { k } = t + k h$ , define the baseline velocity mismatch evaluated strictly on the teacher’s trajectory as $\delta _ { v } ^ { ( k ) } \triangleq \tilde { v } _ { S } ( x _ { T } ^ { ( k ) } ) - v _ { T } ( x _ { T } ^ { ( k ) } )$ with its supremum bounded as $\lVert \delta _ { v } \rVert _ { \infty } \triangleq \operatorname* { m a x } _ { k } \| \delta _ { v } ^ { ( k ) } \| _ { 2 }$ . Define the teacher’s Lipschitz constant $\begin{array} { r } { L _ { T } \triangleq \operatorname* { s u p } _ { x , t } \| J _ { v _ { T } } ( x , t ) \| _ { 2 } } \end{array}$ , and the Jacobian mismatch $\begin{array} { r } { \Delta J \triangleq \operatorname* { s u p } _ { x , t } \Vert J _ { \tilde { v } _ { S } } ( x , t ) - J _ { v _ { T } } ( x , t ) \Vert _ { 2 } } \end{array}$

Proof. Setup: recursive error accumulation. In a multi-step rollout, each model feeds its own prediction into the next Euler step:

$$
x _ { S } ^ { ( k + 1 ) } = x _ { S } ^ { ( k ) } + h \tilde { v } _ { S } ( x _ { S } ^ { ( k ) } ) , \qquad x _ { T } ^ { ( k + 1 ) } = x _ { T } ^ { ( k ) } + h v _ { T } ( x _ { T } ^ { ( k ) } ) .\tag{F.1}
$$

Let $e _ { k } \ { \triangleq } \ x _ { S } ^ { ( k ) } - x _ { T } ^ { ( k ) }$ denote the spatial deviation at step $k ,$ with the initial condition $e _ { 0 } = 0$ Subtracting the two updates yields the discrete error dynamics:

$$
e _ { k + 1 } = e _ { k } + h \big ( \tilde { v } _ { S } ( x _ { S } ^ { ( k ) } ) - v _ { T } ( x _ { T } ^ { ( k ) } ) \big ) .\tag{F.2}
$$

Decomposing the velocity difference via integral Taylor expansion. To separate the pointwise baseline error from the accumulated spatial drift, we add and subtract the student’s velocity at the

![](images/dd8293ac2ade9702f9186bdf0c92586e55413f7e5648b22a547d7eaefca8b86e.jpg)

![](images/be485bf62f833e5cd371300101f1e280f1042c3f2e40207648188090f6a4ddaa.jpg)

![](images/d92039ff2f058ace95f5cc611816d730509dd19bb31c04ee8caac0a2bd8670ce.jpg)  
Figure 5: Compact routing and execution accounting at $K = 3 , \tau = 0 . 8 0$ . The panels use distinct denominators: (a) 14 accepted and 8 rejected routing events; (b) 42 student-covered and 8 teacherrescue nominal steps; and (c) the separately collected component profile. The component profile totals 474.85 ms and is not a decomposition of the separate 464 ms headline average.

![](images/41e7962ba4aae60e19177313850a828def5e2398f59912f9bb245ef09b6beff9.jpg)

![](images/50aef6a6f27197f1bfb25ade9b07cc76b5dace8b693b65c1dab9536c65a9336c.jpg)

![](images/c336e66f3262203cb030c7052cf5dd80a99d0b2211c01cc92f9629ad6d00ebae.jpg)

![](images/3041346259b000578fd6716c788f56c870acbad16f7e32ba6ec81d29c8cf667a.jpg)

![](images/67ca12fc6a7267b70a14e96f7d38ee17f6fd3edbbd64d6605e17257ab0e95e81.jpg)

![](images/e298950dc6b10bbf8573d340bfb100460362c743daba2f68bf8e6954a31ad83b.jpg)  
Figure 6: Cross-setting transfer shown as six compact native-unit panels. Columns are COCO, Latte/FaceForensics, and ImageNet $5 1 2 \times 5 1 2 ;$ the top row reports quality metrics and the bottom row reports semantic or recognition scores. Row labels include the reported speedup relative to the teacher for that setting.

teacher state, $\tilde { v } _ { S } ( x _ { T } ^ { ( k ) } )$ :

$$
\begin{array} { r } { \tilde { v } _ { S } ( x _ { S } ^ { ( k ) } ) - v _ { T } ( x _ { T } ^ { ( k ) } ) = \underbrace { \left( \tilde { v } _ { S } ( x _ { T } ^ { ( k ) } ) - v _ { T } ( x _ { T } ^ { ( k ) } ) \right) } _ { \delta _ { v } ^ { ( k ) } } + \underbrace { \left( \tilde { v } _ { S } ( x _ { S } ^ { ( k ) } ) - \tilde { v } _ { S } ( x _ { T } ^ { ( k ) } ) \right) } _ { \mathrm { D r i f t i n d u c e d b y } e _ { k } } . } \end{array}\tag{F.3}
$$

Since the velocity field $\tilde { v } _ { S } : \mathbb { R } ^ { D } \to \mathbb { R } ^ { D }$ is vector-valued, the scalar Mean Value Theorem does not apply. We instead write the spatial drift using the first-order Taylor expansion with an integral remainder along the line segment connecting $x _ { T } ^ { ( k ) }$ and $x _ { S } ^ { ( k ) }$

$$
\tilde { v } _ { S } ( x _ { S } ^ { ( k ) } ) - \tilde { v } _ { S } ( x _ { T } ^ { ( k ) } ) = \left( \int _ { 0 } ^ { 1 } J _ { \tilde { v } _ { S } } \left( x _ { T } ^ { ( k ) } + \tau e _ { k } \right) \mathrm { d } \tau \right) e _ { k } .\tag{F.4}
$$

Taking the $L _ { 2 }$ norm of the error recurrence, substituting the decomposition and the integral expansion, and applying the triangle inequality yields:

$$
\begin{array} { r l r } & { } & { \| e _ { k + 1 } \| _ { 2 } \le \| e _ { k } \| _ { 2 } + h \| \delta _ { v } \| _ { \infty } + h \left\| \displaystyle \int _ { 0 } ^ { 1 } J _ { \widetilde { v } _ { S } } \left( x _ { T } ^ { ( k ) } + \tau e _ { k } \right) \mathrm { d } \tau \right\| _ { 2 } \| e _ { k } \| _ { 2 } } \\ & { } & { \le \| e _ { k } \| _ { 2 } + h \| \delta _ { v } \| _ { \infty } + h \left( \displaystyle \int _ { 0 } ^ { 1 } \displaystyle \operatorname* { s u p } _ { x } \| J _ { \widetilde { v } _ { S } } ( x ) \| _ { 2 } \mathrm { d } \tau \right) \| e _ { k } \| _ { 2 } . } \end{array}\tag{F.5}
$$

Here, $\operatorname { s u p } _ { x } \| J _ { \tilde { v } _ { S } } ( x ) \| _ { 2 }$ is the student’s Lipschitz constant $L _ { S }$ . By splitting the student’s Jacobian into the teacher term and the mismatch term, we bound it with the matrix triangle inequality:

$$
L _ { S } = \operatorname* { s u p } _ { x , t } \| J _ { \tilde { v } _ { S } } \| _ { 2 } = \operatorname* { s u p } _ { x , t } \| J _ { v _ { T } } + ( J _ { \tilde { v } _ { S } } - J _ { v _ { T } } ) \| _ { 2 } \leq L _ { T } + \Delta J .\tag{F.6}
$$

Since the integral over [0, 1] equals one, substituting the Jacobian bound into the preceding inequality gives the recurrence:

$$
\begin{array} { r } { \| e _ { k + 1 } \| _ { 2 } \leq \left( 1 + h ( L _ { T } + \Delta J ) \right) \| e _ { k } \| _ { 2 } + h \| \delta _ { v } \| _ { \infty } . } \end{array}\tag{F.7}
$$

Solving the recurrence. Let ${ \cal { C } } \triangleq L _ { T } + \Delta J .$ Unrolling the recurrence with the initial condition $e _ { 0 } = 0$ yields a geometric series:

$$
\| e _ { K } \| _ { 2 } \leq h \| \delta _ { v } \| _ { \infty } \sum _ { i = 0 } ^ { K - 1 } ( 1 + h C ) ^ { i } = \frac { \| \delta _ { v } \| _ { \infty } } { C } \big ( ( 1 + h C ) ^ { K } - 1 \big ) .\tag{F.8}
$$

To expose the polynomial dependence on K in the few-step regime, we use the Taylor expansion of the exact scalar term $( 1 + h \dot { C } ) ^ { K }$

$$
( 1 + h C ) ^ { K } = 1 + K ( h C ) + \frac { K ( K - 1 ) } { 2 } ( h C ) ^ { 2 } + \mathcal { O } \big ( K ^ { 3 } ( h C ) ^ { 3 } \big ) .\tag{F.9}
$$

For $C = 0$ the recurrence gives $\lVert e _ { K } \rVert _ { 2 } \leq K h \rVert \delta _ { v } \rVert _ { \infty }$ . For $K h C \ll 1$ , expansion of the geometric upper bound gives:

$$
\| e _ { K } \| _ { 2 } \leq K h \| \delta _ { v } \| _ { \infty } + \frac { K ( K - 1 ) } { 2 } h ^ { 2 } C \| \delta _ { v } \| _ { \infty } + \mathcal { O } \big ( K ^ { 3 } h ^ { 3 } C ^ { 2 } \| \delta _ { v } \| _ { \infty } \big ) .\tag{F.10}
$$

Interpretation. The exact bound isolates the baseline velocity mismatch and shows how the combined Lipschitz/mismatch constant $C = L _ { T } + \Delta J$ amplifies errors with the rollout horizon. In the few-step expansion, this produces the stated quadratic-in-K sensitivity. Pointwise velocity matching directly supervises the forcing term but does not explicitly constrain the spatial amplification constant; it may nevertheless affect both. The expansion concerns an upper bound, not a measured error growth law. □

## Appendix G Scope of the VAH Mechanism

VAH is a residual correction trained through the Stage-2 velocity and trajectory objectives in Eq. equation 5. No separate explicit Jacobian-matching objective is specified by this training recipe. For $v _ { S } ^ { \mathrm { a l i g n e d } } = v _ { S } + \mathrm { V A H } ( v _ { S } )$ , the chain rule gives $J _ { v _ { \mathrm { c } } ^ { \mathrm { a l i g n e d } } } = ( I + J _ { \mathrm { V A H } } ) J _ { v _ { S } }$ . This permits changes to local derivatives but does not guarantee reduced Jacobian mismatch. Figure 9 reports velocity mismatch and endpoint drift only. It provides no Jacobian diagnostic: reduction of $\Delta J$ remains a theoretical possibility, not an experimentally established mechanism.

Algorithm 1 TamedFlow Inference   
1: Input: $x \sim \mathcal { N } ( 0 , I )$ , y, discrete times $\{ t _ { 0 } , \ldots , t _ { N } \}$ , K = 3, threshold τ   
2: i ← 0   
3: while $i < N$ do   
4: $m \gets \operatorname* { m i n } ( K , N - i )$ ▷ Clip the terminal draft horizon   
5: $x _ { \mathrm { s t a r t } }  x$   
6: for $k = 1$ to m do ▷ Student drafts m steps   
7: $( v _ { S } , \mathrm { t o k e n s } _ { k } ) \gets B _ { \theta } ( x , t _ { i + k - 1 } , y )$   
8: $v _ { S } \gets v _ { S } + \mathrm { V A H } ( v _ { S } )$   
9: $x  x + ( t _ { i + k } - t _ { i + k - 1 } ) \cdot v _ { S }$   
10: end for   
11: $c \gets \mathrm { L T E } ( \mathrm { t o k e n s } _ { m } , e _ { t _ { i + m - 1 } } , e _ { y } )$ ▷ Last draft features; no teacher access   
12: if $c \geq \tau$ then   
13: $i \gets i + m$ ▷ Accept: advance by m, no teacher call   
14: else   
15: $x  x _ { \mathrm { s t a r t } }$ ▷ Reject: roll back to pre-draft state   
16: v<sub>T</sub> $ \mathcal { T } _ { \phi } ( x , t _ { i } , y )$ ▷ Teacher rescue evaluation step   
17: $x  x + ( t _ { i + 1 } - t _ { i } ) \cdot v _ { T }$   
18: $i \gets i + 1$   
19: end if   
20: end while   
21: return x

## Appendix H Inference Details

Velocity-field interface and guidance. The symbol v<sub>T</sub> denotes the effective teacher right-hand side used by the Euler solver, including the prescribed conditioning and guidance, rather than specifying the raw prediction type of the underlying DiT. Student–teacher velocity losses and teacher reference rollouts are expressed in this same solver coordinate system. The single-branch student is distilled toward the guided teacher field; a teacher rescue still evaluates the complete guided teacher update. The linear-interpolant objective in Section 3.1 introduces the velocity-form convention and, by itself, does not identify the pretrained teacher’s native output parameterization or noise schedule. An architecture name alone does not determine a checkpoint or an output-to-ODE conversion.

Time coordinates. The solver and all rollout equations use increasing flow time t, with noise at t = 0 and data at t = 1. Figure 4 retains a decreasing noise-level plotting coordinate on its upper axis; that axis label is not the forward-time variable in the solver equations. The reported value 0.22 describes that plotted noise coordinate, not forward flow time. Sampling-step indices are used to compare the routing profile with execution order.

Algorithm 1 gives the full routing loop under the forward flow-time convention $t _ { 0 } < \cdots < t _ { N }$ . The student drafts up to K steps without teacher access, and the LTE scores the draft from the last student forward, the same type of input used during LTE training. Rejection restores the pre-draft state before one teacher rescue step; no endpoint backbone call is added for scoring.

## Appendix I LTE Calibration and Routing Metrics

Table 8 reports held-out routing diagnostics at $\tau = 0 . 8 0$ and $K \ : = \ : 3 .$ The LTE regresses the continuous target $c ^ { \star }$ in Eq. equation 7; binary labels are introduced only to summarize thresholded routing errors.

Reference labels and routing errors. A draft event is labeled offline as accept when $c ^ { \star } \geq 0 . 8 0$ and reject otherwise; the online decision accepts when the prediction $c \geq 0 . 8 0 . \mathrm { F A R }$ uses the 4,683 reference-reject events as its denominator, FRR uses the 11,984 reference-accept events, and AUC ranks events by continuous c. These labels measure agreement with the specified teacher-consistency target, not preservation of final sample quality.

Validation protocol. Threshold and score-temperature choices are selected on a calibration split and then frozen. Table 8 is computed on a held-out trajectory-level split, preventing draft events from one trajectory from appearing in both sets. Continuous-target ECE is the event-weighted 10-bin discrepancy between mean c and mean $c ^ { \star } ;$ ; it measures score agreement rather than a probability of correct generation.

Table 8: LTE classification and calibration metrics on the ImageNet $2 5 6 \times 2 5 6$ validation set at $\tau = 0 . 8 0 , K = 3$ . Support counts reference labels obtained by thresholding $c ^ { \star }$ at 0.80. FAR and FRR use reference-reject and reference-accept events as their respective denominators. AUC uses continuous predictions; continuous-target ECE uses 10 confidence bins.
<table><tr><td>Support (Reference)</td><td>Accept</td><td>Reject</td></tr><tr><td>Count</td><td>11,984</td><td>4,683</td></tr><tr><td>Metric</td><td colspan="2">Value</td></tr><tr><td>False Accept Rate (FAR)</td><td colspan="2">0.063</td></tr><tr><td>False Reject Rate (FRR) AUC</td><td colspan="2">0.051</td></tr><tr><td>Continuous-target ECE (10 bins)</td><td colspan="2">0.94 0.034</td></tr></table>

The reference-positive fraction is $1 1 , 9 8 4 / 1 6 , 6 6 7 = 7 1 . 9 \%$ , whereas the online predicted acceptance rate is 63.6%. These use different labels and event populations, so their difference alone does not identify distribution shift. Held-out diagnostics do not establish error rates along free-running online trajectories. Thresholding predicted confidence separates routed events by construction; these held-out diagnostics do not establish calibration against actual draft error.

## I.1 Routing-control protocol and call accounting

The controls in Table 9 use the same trained student checkpoint, VAH, 50-step Euler grid, CFG configuration, and ImageNet FID-50K protocol. Latency follows the single-A800, batch-one protocol in the main experiments. These measurements evaluate inference policies without retraining the student.

Static policies. Pure Student performs all 50 updates with the student and never calls the teacher. Random Steps independently selects a set of eight distinct integration steps per trajectory before sampling and replaces those student updates with teacher updates. Fixed Periodic uses the onebased integration-step indices 6, 12, 18, 24, 30, 36, 42, 48. Neither static policy executes a speculative student draft at a teacher-designated step. Both therefore execute exactly 42 student forwards and eight teacher updates.

Budget-constrained random decisions. Random Accept uses the same $K = 3$ draft-and-rollback structure as LTE, but substitutes a random accept decision with nominal probability $p = 0 . 6 3 6$ The policy enforces exactly eight teacher updates per trajectory. Once that budget is exhausted, subsequent drafts are accepted. A budget-feasibility override forces rejection when accepting a draft would leave too few integration steps to spend the remaining teacher budget. A rejection discards the draft, restores its starting state, and advances one teacher step. Consequently, the actual decisions are not independent Bernoulli draws throughout the trajectory, and p is not a measured acceptance rate.

LTE policy. LTE uses the original confidence threshold $\tau = 0 . 8 0$ without a fixed per-trajectory teacher budget. Its teacher and student call counts depend on the trajectory. The previously reported 14 accepted and eight rejected events describe a reference execution, not a measured evaluation-set mean. Thus the random control and LTE share the execution framework and nearly match measured latency, but are not established to have identical average teacher usage.

LTE evaluations versus backbone calls. Let A and R denote accepted and rejected draft events. Each draft triggers exactly one LTE evaluation, regardless of whether it is accepted; only rejection triggers a teacher update. For the terminal-clipped $K = 3$ loop,

Table 9: Routing controls on ImageNet $2 5 6 \times 2 5 6$ . Latency and FID-50K are measured. Teacher updates are fixed by the control policy or adaptive for LTE; this is not an equal-teacher-budget comparison with LTE.
<table><tr><td>Routing policy</td><td>Teacher updates</td><td>Latency (s) ↓</td><td>FID↓</td></tr><tr><td>Pure Student</td><td>0</td><td>0.180</td><td>5.56</td></tr><tr><td>Static: Random Steps</td><td>8</td><td>0.348</td><td>3.72</td></tr><tr><td>Static: Fixed Periodic</td><td>8</td><td>0.335</td><td>3.45</td></tr><tr><td>Dynamic: Random Accept</td><td>8</td><td>0.465</td><td>4.05</td></tr><tr><td>Dynamic: LTE</td><td>Adaptive</td><td>0.464</td><td>2.58</td></tr></table>

$$
N _ { T } = R , \qquad N _ { \mathrm { L T E } } = A + R , \qquad N _ { S } = \sum _ { j = 1 } ^ { A + R } m _ { j } , \qquad \sum _ { j \in \mathcal { A } } m _ { j } + R = 5 0 .\tag{I.1}
$$

Here $m _ { j }$ is the number of student steps executed in draft j. Only a terminal accepted draft can have fewer than three steps. Conditional on $R = 8 .$ , the 42 committed student steps therefore consist of exactly $A = 1 4$ accepted drafts, and $N _ { \mathrm { L T E } } = 1 4 + 8 = 2 2$ . The event acceptance rate is $1 4 / 2 2 = 6 3 . { \dot { 6 } } \%$ , whereas the fraction of committed integration steps executed by the student is $4 2 / 5 0 = 8 4 \%$ . These are different denominators. If all rejected drafts also have length three, $N _ { S } = 6 6 ;$ terminal clipping can reduce student work without changing the 22 routing events in this case. Thus eight teacher updates correspond to 22 LTE evaluations in the stated 50-step trajectory, not eight or 66 LTE evaluations. The budget-constrained random policy similarly has 22 decision events when $R = 8$ , but uses random decisions rather than requiring LTE network scores. This conditional identity does not establish the mean teacher usage of unconstrained LTE over the evaluation set.

Derived student-forward counts. For the random control, the eight teacher updates leave 42 committed student steps. Executed student forwards additionally include every discarded draft:

$$
N _ { S } = 4 2 + \sum _ { j = 1 } ^ { 8 } m _ { j } , \qquad 1 \leq m _ { j } \leq 3 ,\tag{I.2}
$$

where $m _ { j }$ is the executed length of rejected draft $j .$ . This gives $N _ { S } = 6 6$ when all eight rejected drafts have length three; terminal clipping can reduce the count. For the reference LTE execution, full-length drafts similarly give $N _ { S } = 3 ( 1 4 + 8 ) = 6 6$ and $N _ { T } = 8$ . These are conditional accounting values, not measured mean call counts or latency predictions.

Interpretation. The dynamic comparison gives an FID difference of 1.47 at a latency difference of 0.001 s, supporting the usefulness of learned routing at the measured operating point. The 0.001 s difference is not interpreted as a speed advantage. Static controls are faster (0.335–0.348 s) but have higher FID (3.45–3.72). These measurements do not establish that LTE identifies high-curvature regions, nor do they isolate decision quality at exactly matched teacher usage. The routing-control table reports point estimates; variability across random-policy seeds is not quantified here.

## Appendix J Additional Ablation Study

Table 10 extends the main ablation with randomized-horizon variants. Unless stated otherwise, rows use ImageNet 256 × 256, $K = 3 .$ , and $\tau = 0 . 8 0$

The extended rows reinforce the main ablation: representation alignment, residual correction, rollout loss, and endpoint loss each improve the usable routed operating point. Randomizing training over $K \sim \mathcal { U } ( 1 , 3 )$ yields a nearby but distinct point (FID 2.53 at 2.14×), whereas the wider default $\bar { \mathcal { U } } ( 1 , 5 )$ reaches FID 2.58 at 2.25×.

At fixed $K = 3 ,$ the 75.0% point corresponds to 15 accepted and 5 rejected events (60 student and 5   
teacher calls under full-length drafts), whereas the 63.6% point corresponds to 14 accepted and 8

Table 10: Protocol-consistent extended ablation on ImageNet $2 5 6 \times 2 5 6 .$ . Inference uses $K = 3 ,$ $\tau = 0 . 8 0 ;$ the randomized horizons refer to training.
<table><tr><td>Variant</td><td>Accept Rate % ↑</td><td>Speedup ↑</td><td>FID↓</td><td>IS↑</td></tr><tr><td>Full TamedFlow (default)</td><td>63.6</td><td>2.25×</td><td>2.58</td><td>233.43</td></tr><tr><td>Module and loss ablations</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o AlignNet</td><td>46.2</td><td>1.87×</td><td>2.95</td><td>211.34</td></tr><tr><td>w/o VAH</td><td>39.3</td><td>1.63×</td><td>3.23</td><td>205.71</td></tr><tr><td>w/o  $\scriptstyle { \mathcal { L } } _ { \mathrm { r o l l } }$ </td><td>54.2</td><td>2.02×</td><td>2.71</td><td>230.15</td></tr><tr><td>w/o  $\mathcal { L } _ { z }$  warmup</td><td>46.2</td><td>1.81×</td><td>3.29</td><td>203.50</td></tr><tr><td>w/o  $\mathcal { L } _ { z }$ </td><td>28.1</td><td>1.45×</td><td>4.15</td><td>198.80</td></tr><tr><td>K strategy ablations</td><td></td><td></td><td></td><td></td></tr><tr><td>Random  $K \sim \mathcal { U } ( 1 , 3 )$ </td><td>75.0</td><td>2.14×</td><td>2.53</td><td>233.22</td></tr><tr><td>Random  $K \sim \mathcal { U } ( 1 , 5 )$  (default)</td><td>63.6</td><td>2.25×</td><td>2.58</td><td>233.43</td></tr><tr><td>Pure Student (no LTE, no routing)</td><td>一</td><td>5.80×</td><td>5.56</td><td>195.07</td></tr></table>

Table 11: Same-student validation under the common DDIM-50 protocol. Only FID is reported because these direct adaptive baselines were evaluated as quality diagnostics, not as the headline latency comparison.
<table><tr><td>Strategy</td><td>FID</td><td>Strategy</td><td>FID</td></tr><tr><td>FoCa</td><td>10.40</td><td>TaylorSeer</td><td>11.92</td></tr><tr><td>SpeCa</td><td>13.18</td><td>Dense Student</td><td>5.56</td></tr></table>

rejected events (66 student and 8 teacher calls). Thus the call-count model predicts lower backbone work for the higher-acceptance point. Its measured 2.14× versus 2.25× ordering was obtained from separate checkpoint/runtime measurements, and the available component trace does not provide a matched per-call timing decomposition; we therefore do not interpret this ordering as a monotonic acceptance–latency law.

Under the common DDIM-50 diagnostic, directly attaching FoCa, TaylorSeer, or SpeCa to the compressed student degrades FID relative to dense student execution. This documents the Combination Gap for these implementations; it does not test whether the same methods remain fragile after trajectory training.

Table 12: Solver robustness on ImageNet 256 × 256. DPM-Solver++(2M) is evaluated as a handoffsafe higher-order solver diagnostic; the formal bound remains DDIM-specific.
<table><tr><td>Solver</td><td>Method</td><td>Latency</td><td>Speedup</td><td>Accept</td><td>FID</td><td>IS</td></tr><tr><td>Euler</td><td>Teacher</td><td>1.044</td><td>1.00×</td><td></td><td>2.35</td><td>237.62</td></tr><tr><td>Euler</td><td>TamedFlow</td><td>0.464</td><td>2.25×</td><td>63.6%</td><td>2.58</td><td>233.43</td></tr><tr><td> $\mathrm { D P M - S o l v e r } { + } { + } ( \mathrm { 2 M } )$ </td><td>Teacher</td><td>1.089</td><td>1.00×</td><td></td><td>2.25</td><td>244.12</td></tr><tr><td> $\mathrm { D P M - S o l v e r } { + } { + } ( \mathrm { 2 M } )$ </td><td>TamedFlow</td><td>0.455</td><td>2.39×</td><td>75.0%</td><td>2.44</td><td>237.55</td></tr></table>

For the DPM-Solver++(2M) diagnostic, the multistep history is associated with the current accepted state. When a student draft is rejected, the tentative student states and derivative history are discarded together; the sampler rolls back, applies one teacher rescue update, and starts a fresh solver history at the rescued state. No student derivative is mixed with a teacher derivative across a rejection boundary, and the first post-handoff update uses the solver’s order-one startup rule before the two-step history is rebuilt. Thus Table 12 tests end-to-end robustness under an explicit reset policy; Proposition 3.1 continues to cover only the Euler analysis.

Figure 9 reports velocity and endpoint errors, not Jacobian alignment. Removing AlignNet or VAH changes the measured velocity or endpoint mismatch, while removing rollout supervision increases short-horizon drift.

![](images/3ed9f8e5098491a3d9d99baadb723973be47ecb05cc67d29657054a380556799.jpg)

![](images/6a612407a5e54e294613b80a95112da2e8d6091d8dfbc5c8712859a3ef83bc29.jpg)

![](images/3a8d92dcc8db9d2650e75c8781fb83b65b15cfa4e07f76a19b627774b7c6d258.jpg)

![](images/09d8c34f21e942fac825566e03f033b4dc20bf643d979d29b1893b0beb677eb7.jpg)  
Fixed inference K=3, τ=.80. “No routing” has no acceptance statistic and appears only in panel (b).  
Figure 7: Module and training ablations from Tables 2 and 10. The panels relate FID to acceptance or speedup at $K = 3 , \tau = 0 . 8 0$ . The no-routing student has no acceptance statistic and is omitted from the acceptance panels. Points are reported operating points; no fitted trend or uncertainty interval i implied.

![](images/106a6d63616fff1e1f324b6a7e5a9b23bcb26b294e27b2f516c4b863b78dbb9a.jpg)

![](images/e835169a26c34cf8a4e8ed970a3f80eb72ebd5cd1a34158a04a80f633667cef7.jpg)  
Teacher-normalized speedup: Euler 2.25×; DPM++(2M) 2.39×. All bars use zero baselines

![](images/edfe4d1064614cd7ed984769d6e310ce72bfa1c857b2b0b46c22adb7b1d79a12.jpg)  
Figure 8: Paired teacher and TamedFlow measurements under Euler and DPM-Solver++(2M), reproducing Table 12. Speedups are relative to the corresponding solver teacher. The DPM++ result uses the stated history-reset policy; the theoretical bound remains DDIM-specific.

![](images/f193cf00a5332c26b5e5449949087151d34cda2d71b2475230d47308f9f88a78.jpg)

![](images/d60a291e157b9171e96c67dda75a6e3290c94026facee2b94d1e078da5d13889.jpg)  
Separate scales; reported means only. These quantities do not measure Jacobian mismatch.

Figure 9: Held-out velocity mismatch and three-step endpoint drift by model variant. The panels replace the redundant numerical table and use separate zero-based axes. These means do not measure Jacobian mismatch or establish that training minimizes the theoretical bound constants; no uncertainty estimates are available.

Figure 10 visualizes the impact of draft horizon K and routing threshold τ on generation quality and inference efficiency. The four marginal panels show FID and speedup versus τ (top) and K (bottom). These one-dimensional sweeps motivate the default operating point but do not establish a globally optimal Pareto frontier.

![](images/79f3b5f66fea775d2eae2f1e7472b643f289f64d596ec42fe052742f27965589.jpg)

![](images/12be9002ee64fc8ff75f036cc395230e5350bbb32237b34e9126066fa1bcd3ca.jpg)

![](images/f7683731cf9b1d8c071591ede133ff52a9a2db18a55e174194e46b9a6eeea207.jpg)

![](images/6f12470b9821e560607fcc676425e434394c9b333027309e875a2bf9a7de60c3.jpg)  
Figure 10: Impact of draft length K and routing threshold τ on generation quality and inference efficiency. Columns: FID (left) and speedup (right). Rows: threshold τ (top) and draft length K (bottom). The sweeps motivate the default $\bar { K } = \bar { 3 } , \tau = 0 . 8 0$ configuration.

## Appendix K Limitations

While TamedFlow provides empirical gains across several settings, our findings are subject to several key limitations:

Theoretical scope of rollout analysis. The discrete Grönwall bound in Proposition 3.1 characterizes error accumulation under sequential Euler drafting to motivate multi-step trajectory supervision. It does not constitute a universal failure law for internal feature caching or forecasting methods, nor does it provide formal convergence or stability guarantees for higher-order ODE solvers such as DPM-Solver++.

Approximate nature of trajectory routing. The Learnable Trajectory Estimator (LTE) operates as an empirical regression model trained on offline consistency targets rather than an exact, distribution-preserving verification mechanism. Rejection decisions are heuristic; false rejections incur latency penalties through redundant teacher evaluations, whereas false acceptances may allow minor perceptual artifacts to propagate into the generated sample.

Deployment and infrastructure overhead. TamedFlow relies on a multi-stage offline training curriculum and maintains the teacher, student, VAH, and LTE models co-resident in GPU memory during inference (∼ 0.5 GB VRAM overhead in our FP32 profile). As a result, it does not fully eliminate the teacher model from deployment. Furthermore, our reported speedups reflect batchsize-one latent generation on single GPUs; characterization under large-batch throughput, concurrent serving pipelines, and extreme memory-constrained edge environments remains an avenue for future work.

## Appendix L LLM Usage Statement

Large language models were used for language editing and consistency checks across the manuscript. Experimental results are author-provided; the authors are responsible for checking the final equations, measurements, citations, and implementation details.