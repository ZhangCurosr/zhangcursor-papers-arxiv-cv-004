# Rendering-Free Lookahead for Question-Guided Active Vision

Koya Sakamoto<sup>1</sup>

Yusuke Iwasawa<sup>1</sup>

Daichi Azuma<sup>1</sup>

Shuhei Kurita<sup>2,3,4</sup>

Yutaka Matsuo<sup>1</sup>

Naoya Chiba<sup>5</sup>

Taiki Miyanishi<sup>1</sup>

Abstract— Active robot vision requires controlling the camera to reveal task-relevant information that is hidden from the current viewpoint. For example, determining what is inside a box may require raising the camera and looking down into it. For viewpoint-dependent question answering, the challenge is to select camera motions that expose the visual evidence needed to answer the question. Although vision-language models (VLMs) can interpret observed images, selecting such motions requires anticipating the usefulness of unseen views. We quantify this usefulness as answerability, a VLM’s estimate that a view suffices to answer the question, and present Rendering-Free Lookahead (RFL), a viewpoint-selection policy that ranks candidate camera motions by predicted future answerability. RFL transfers visual lookahead from deployment to offline training. At training, a privileged teacher renders candidate future views in 3D Gaussian Splatting (3DGS) scenes and uses a frozen VLM to compute one- and two-step answerability targets. Through two-stage distillation, a student learns to predict these action values from the question, recent visual observations, and a candidate camera motion. At deployment, RFL uses these predicted values to select camera motions without rendering future views. On 377 E3VS-Bench test episodes in unseen environments, RFL improves the mean judge score by 43% over a direct-action baseline using the same VLM. These results support learning camera-control policies from privileged visual lookahead for viewpoint-dependent question answering.

## I. INTRODUCTION

Visual perception in robotics depends on the actions used to acquire observations. When task-relevant evidence is occluded, a robot must move its camera to reveal it. For example, a mug is readily recognized from the side, yet determining whether it is empty may require raising the camera and looking into it. Beyond locating the object, selecting such camera motions requires inferring where the evidence is likely to be and which viewpoint will reveal it.

We study this problem in Embodied 3D Visual Search (E3VS) [30]. Given a question and egocentric RGB observations, an agent controls its camera position, yaw, and pitch in an unseen scene to reach a viewpoint from which the question can be answered. Without a target viewpoint or scene map, the agent must infer from its observations which camera motion is likely to reveal the missing evidence within a limited action budget. We consider the setting where arbitrary viewpoints can be rendered in the training scenes, whereas the deployment scene is unknown and only the agent’s own observations are available.

Q: Is the white mug empty or does it contain something?  
![](images/c12e7312962dcb5eb90681429a0f62a7e82c746dfb5aab4e9c933190d129b4ae.jpg)

![](images/63174168f76bac30d36c27e5d64aefd07f049910fd3f41214874ab6295d81400.jpg)  
Fig. 1: Rendering-free lookahead. RFL selects camera motions by predicted answerability, which rises as the agent closes in on a view that reveals the answer (right), with no rendering or generation at deployment.

Vision-language models (VLMs) offer a foundation for this task by relating visual observations to natural-language questions [19], [15], [32]. A central challenge in selecting camera motions is to anticipate whether an unseen future view will reveal the evidence needed for an answer. Supervision from demonstrated actions alone specifies which motion to select but does not quantify how useful each alternative would be. This motivates learning a question-conditioned value for each candidate motion from its counterfactual visual outcome during training.

We introduce Rendering-Free Lookahead (RFL; Fig. 1), which ranks candidate camera motions by their predicted usefulness for answering the question. We quantify the usefulness of a view through answerability, a VLM’s estimate of whether the view contains sufficient visual evidence to answer the question. Prior work has used confidence to guide stopping [26] and estimated answerability across scene locations [2]. In contrast, our key idea is to evaluate answerability at candidate future viewpoints during training and distill the resulting short-horizon action values into a camera-control policy.

To this end, a privileged teacher scores future views rendered from 3D Gaussian Splatting (3DGS) training scenes [12] with a frozen VLM, constructing lookahead targets for every candidate motion, including unexecuted alternatives. The student learns to predict these values from the question and recent observations, first on one-step and then on two-step targets, enabling viewpoint selection without rendering at deployment.

We evaluate RFL on 377 E3VS-Bench test episodes across 21 unseen scenes. It improves the mean judge score from 2.14 to 3.05 (+43%) over the same VLM acting directly and also surpasses the strongest evaluated proprietary baseline, GPT-5.1 (2.81). It also outperforms cloning the teacher’s actions with the same backbone and trajectories (2.47), supporting candidate-wise value supervision for viewpoint selection. Its one-step variant matches the mean judge score of the privileged one-step rendering oracle (2.98).

Our contributions are:

• RFL, which calibrates VLM answerability into an action value by distilling privileged lookahead targets, requiring no rendering at deployment.

• Improvements on unseen scenes over the same VLM acting directly, GPT-5.1, and cloning the teacher’s actions.

• Transfer of the simulation-trained policy to a real 6-DoF arm with an end-effector camera, achieving the highest mean judge score among the compared methods without test-time rendering.

## II. RELATED WORK

## A. Active Vision and Embodied Question Answering

Active vision selects sensing actions to acquire taskrelevant information [5], [1]. This principle underlies both learned gaze control for robotic manipulation [13] and question-driven exploration in active Embodied Question Answering (EQA) [9], [19]. In active EQA, a natural-language question defines the information-gathering objective, and agents explore to acquire the visual evidence needed for an answer, often using structured scene representations [31], [29]. Closest to our use of answerability, Explore until Confident uses calibrated VLM confidence to decide when to stop exploring [26], and Answerability Fields estimates which locations in a scene are answerable [2]. Neither uses answerability to select camera motions, whereas RFL predicts the answerability of candidate future viewpoints and uses it as the action value that drives viewpoint selection.

Closer to camera control, VG-AVS learns viewpoint adjustments through supervised fine-tuning and reinforcement learning with an answer-correctness reward [15]. Planning with the Views uses self-exploration and view-graph distillation for target-view localization, given a target image and top-down reference [32]. RFL operates in the E3VS setting, where neither a scene map nor a target viewpoint is supplied [30]. While VG-AVS learns to output a viewpoint adjustment, RFL learns to score candidate motions using graded, question-conditioned answerability targets over a short horizon. This supervision covers every candidate motion at each training state, including alternatives not executed by the teacher.

## B. Visual Lookahead for Action Selection

Anticipating the visual consequences of a motion provides a basis for selecting informative viewpoints. In navigation, world models capture relationships between motions and future observations [14], [6], [3], while semantic representations support lookahead at candidate viewpoints [33]. Additional views can also be rendered from persistent 3DGS scene memory to recover visual evidence [17].

Future prediction can also support policy learning without being required at deployment, through video co-training [34] or trajectory supervision derived from generated future views [8]. RFL similarly uses future views during training, yet transfers their utility into question-conditioned action values based on short-horizon answerability. These values are supervised using rendered future views and predicted directly by the deployed policy, without rendering or generating future observations.

## C. Privileged Supervision and Value Distillation

The usefulness of a camera motion can be difficult to assess from current observations alone. Privileged supervision exploits richer scene or state information during training to guide policies that operate from limited observations at deployment [7], [16], [25]. Teacher supervision can extend beyond selected actions to action preferences and searchderived values [28], [27]. Action-value learning has also been explored for language and vision-language models through offline reinforcement learning [11], [4].

RFL evaluates rendered future views with a frozen VLM and uses short-horizon search to construct graded, questionconditioned answerability targets. These targets supervise every candidate camera motion at each sampled training state, including unexecuted alternatives. The student thus learns to compare candidate motions by their usefulness for acquiring the visual evidence needed for an answer.

## III. PROBLEM STATEMENT

We consider the Embodied 3D Visual Search (E3VS) task [30], in which an agent controls its camera in an unseen scene to acquire the visual evidence needed to answer a question q. The agent receives only q and egocentric RGB observations, with neither a scene map nor a target viewpoint.

Let S denote a 3D scene and $p _ { t }$ the camera pose at step $t ,$ comprising a 3D position, yaw, and pitch with fixed roll. The RGB observation is $o _ { t } ~ = ~ R _ { S } ( p _ { t } )$ , where $R _ { S }$ maps a camera pose to its observation in scene S. The action space A consists of ten discrete camera motions (six translations and four rotations) and stop. For an action $a _ { t } \in { \mathcal { A } }$ , the state transition is $p _ { t + 1 } = T ( p _ { t } , a _ { t } )$ . A colliding motion leaves the pose unchanged while consuming a step.

The episode terminates at step τ when the agent selects stop or reaches the action budget $T _ { \mathrm { m a x } }$ . A VQA model produces $\hat { y } = \mathrm { V Q A } ( o _ { \tau } , q )$ from the final observation, which is evaluated against the ground truth $y ^ { * }$

![](images/63b4e67d0e714b20d33814d5ff84f8d61b770e1cf4f950a73933ca495b86a5bb.jpg)  
Fig. 2: Overview of RFL. A privileged teacher renders each candidate action’s one- and two-step future views in 3DGS and scores them with the frozen VLM’s answerability probe, yielding the lookahead targets $Q _ { 1 }$ and $Q _ { 2 } ;$ the student learns to predict these values from the question, past views, and the action, requiring no rendering at deployment.

## IV. METHOD

RFL learns short-horizon action values from privileged visual lookahead, enabling camera control without rendering future views at deployment (Fig. 2). An offline teacher renders future views in 3DGS scenes and evaluates them with a frozen VLM to construct answerability-based targets for all candidate camera motions (Sec. IV-B). Through two-stage distillation, the student learns to predict these values from the question, recent observations, and a candidate motion expressed as text (Sec. IV-C). At deployment, the policy scores candidate motions, executes the highest-scoring one, and repeats with new observations (Sec. IV-D).

## A. Answerability Probe

To construct action-value targets, the teacher must assess whether candidate future views contain sufficient visual evidence to answer the question, which we estimate with a frozen VLM. Given an observation o and question $q ,$ we prompt the VLM with “can $q$ be answered from this view $\varPsi ^ { \ast }$ and introduce answerability Ans(·) as

$$
\mathrm { A n s } ( o , q ) = \frac { e ^ { \ell _ { \mathrm { Y e s } } } } { e ^ { \ell _ { \mathrm { Y e s } } } + e ^ { \ell _ { \mathrm { N o } } } } = \sigma ( \ell _ { \mathrm { Y e s } } - \ell _ { \mathrm { N o } } ) \in [ 0 , 1 ] ,\tag{1}
$$

where σ is the sigmoid function. $\ell _ { \mathrm { Y e s } }$ and $\ell _ { \mathrm { N o } }$ are obtained by applying logsumexp to the logits of each class’s surface variants at the answer position.

On baseline rollouts, mean answerability increases during successful searches, whereas failed searches decline early and stay low (Fig. 3), motivating the probe as a source of action-value targets (Sec. V-C). The teacher applies the probe to rendered future views during short-horizon search.

## B. Privileged Lookahead Teacher

The teacher uses a 3DGS representation of each training scene to evaluate the visual consequences of candidate camera motions. Let $s _ { t } = ( S , p _ { t } , q )$ denote its privileged state and ${ \mathcal { A } } _ { \mathrm { { c a m } } } = { \mathcal { A } } \setminus \{ { \mathrm { s t o p } } \}$ the set of camera motions. At each visited state, the teacher constructs one-step and two-step lookahead targets, $Q _ { 1 } ( q , p _ { t } , a )$ and $Q _ { 2 } ( q , p _ { t } , a )$ , for every candidate motion; their dependence on the scene S is left implicit.

![](images/fb359acc2971e3054839aff0b580bdd849d5e02a15e642a6b8f06a1be756f403.jpg)  
Fig. 3: Answerability tracks viewpoint search. Mean probed answerability vs. normalized progress on Gemini 3.0 Flash rollouts over the 231 validation episodes (bands: ±1 SE; dotted: mean at annotated goal views; Sec. V-C).

Let $o ^ { \prime } \ = \ R _ { S } ( T ( p _ { t } , a ) )$ denote the view rendered after candidate motion $^ { a , }$ and $o ^ { \prime \prime }$ the view after a further motion $a ^ { \prime }$ from the resulting pose. The one-step target is the answerability of the resulting view:

$$
Q _ { 1 } ( q , p _ { t } , a ) = \operatorname { A n s } ( o ^ { \prime } , q ) .\tag{2}
$$

A camera motion may reveal little evidence immediately yet enable a more informative subsequent view, motivating two-step lookahead. To limit rendering, we expand only the top-k $( k = 3 )$ first motions ranked by $Q _ { 1 }$ , denoted by $B _ { t }$ a beam-style approximation that concentrates rendering on the most promising motions. For each expanded motion, the two-step target is the highest answerability among its allowed

continuations:

$$
Q _ { 2 } ( q , p _ { t } , a ) = \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } _ { \mathrm { c a m } } } \mathrm { A n s } ( o ^ { \prime \prime } , q ) , \qquad a \in \mathcal { B } _ { t } .\tag{3}
$$

For the remaining candidates, we set $\begin{array} { r l } { Q _ { 2 } ( q , p _ { t } , a ) } & { { } = } \end{array}$ $Q _ { 1 } ( q , p _ { t } , a )$ . Here inv(a) is the inverse motion, excluded to prevent immediately undoing the first motion. Since a second motion is required for expanded candidates, we retain $Q _ { 2 } ( q , p _ { t } , a )$ even when it is lower than $Q _ { 1 } ( q , p _ { t } , a )$

The teacher generates training trajectories using $Q _ { 2 }$ for motion selection and records both targets for every candidate motion at each visited state, including unexecuted alternatives. All rendering and target generation are performed offline.

## C. Two-Stage Value Distillation

The student learns to predict the teacher’s lookahead values using only inputs available at deployment. It is initialized from the same base VLM as the teacher’s frozen probe and fine-tuned with LoRA adapters with trainable parameters $\theta .$ The teacher targets $Q _ { h } ( q , p _ { t } , a )$ depend on the privileged state through rendering, whereas the student receives only the question q, a short sequence of recent RGB observations $H _ { t } ,$ and a textual description of the candidate motion a, so that $H _ { t }$ is used in place of $p _ { t } .$ . The student is asked whether the viewpoint reached after executing a would provide enough information to answer $q ,$ and its action value is read out from the Yes/No logits as in Eq. (1):

$$
\hat { Q } _ { \theta } ( q , H _ { t } , a ) = \sigma \big ( \ell _ { \mathrm { Y e s } } ^ { \theta } ( q , H _ { t } , a ) - \ell _ { \mathrm { N o } } ^ { \theta } ( q , H _ { t } , a ) \big ) ,\tag{4}
$$

where $\ell _ { \mathrm { Y e s } } ^ { \theta }$ and $\ell _ { \mathrm { N o } } ^ { \theta }$ are the logits of the adapted VLM at the answer position. Whereas the teacher applies the readout of Eq. (1) to a rendered future view (Eqs. (2) and (3)), the student applies it to the current observations and the motion text, and distillation trains $\hat { Q } _ { \theta } ( q , H _ { t } , a )$ to match $Q _ { h } ( q , p _ { t } , a )$ at the states visited by the teacher.

The offline training dataset D contains records $( q , H _ { t } , a , v _ { 1 } , v _ { 2 } )$ collected at visited training states, where $v _ { h } = Q _ { h } ( q , p _ { t } , a )$ is the teacher target at that state. For each training stage $h \in \{ 1 , 2 \}$ , we fit the continuous targets with soft-label binary cross-entropy over D:

$$
\begin{array} { r } { \mathcal { L } _ { h } ( \theta ) = - \mathbb { E } _ { \mathcal { D } } \Big [ v _ { h } \log \hat { Q } _ { \theta } ( q , H _ { t } , a ) \qquad } \\ { + \left( 1 - v _ { h } \right) \log \left( 1 - \hat { Q } _ { \theta } ( q , H _ { t } , a ) \right) \Big ] . } \end{array}\tag{5}
$$

We first minimize $\mathcal { L } _ { 1 }$ to learn one-step action values and then continue from the resulting parameters by minimizing $\mathcal { L } _ { 2 }$ on the two-step targets, both on the offline teacher dataset. Alternative observation histories are compared in Sec. V-D.

## D. Inference-Time Policy

At deployment, the policy selects camera motions using the learned action values. Let $\ r { \mathcal { A } } _ { t } \subseteq \ r { \mathcal { A } } _ { \mathrm { c a m } }$ denote the motions not yet observed to collide at the current pose. When continuing the search, the agent selects

$$
a _ { t } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } _ { t } } \hat { Q } _ { \theta } ( q , H _ { t } , a ) .\tag{6}
$$

The agent executes only the selected motion, updates its observation history, and recomputes the action values at the next step.

A colliding motion leaves the pose unchanged while consuming a step (Sec. III), so the policy excludes it from $\boldsymbol { A } _ { t }$ until the camera pose changes, using only observed outcomes and no privileged scene geometry.

The learned LoRA adapters are enabled only for actionvalue prediction. Stopping uses the base VLM with these adapters disabled. Candidate scoring requires no rendering or generation of future views.

## V. EXPERIMENTS

## A. Experimental Setup

We ask four questions: (i) whether ranking camera motions by predicted answerability improves viewpoint selection over direct control, action imitation, and future-view generation (Sec. V-B); (ii) whether answerability is a reliable supervision signal (Sec. V-C); (iii) which design and data choices drive the gain (Sec. V-D); and (iv) whether the simulationtrained policy transfers to a physical robot (Sec. V-F).

Dataset. We use E3VS-Bench [30] because its questions require camera motions that reveal viewpoint-dependent evidence, and its 3DGS scenes allow rendering arbitrary viewpoints for the privileged teacher. It comprises 2,014 episodes across six question types in 99 indoor 3DGS scenes from SceneSplat++ [18]. Each episode specifies only an initial pose and a human-annotated goal viewpoint, so training trajectories must be generated by the teacher (Sec. IV-B). The official scene-disjoint split has 1,406 training (68 scenes), 231 validation (10 scenes), and 377 test episodes (21 unseen scenes).

Metrics. GPT-5.1 [21] answers each question from the final view, and a GPT-5.5 [22] judge scores the answer from 1 to 5. The primary metric is the mean judge score (Avg.), reported overall and by question type; an episode is successful if it scores 5. We also report the mean episode length (Steps), the collision rate (Coll., the fraction of episodes with at least one collision), and the repetition rate (Rep.), which captures oscillatory motion: for executed camera motions $a _ { 1 } , \dots , a _ { N }$ with $N \geq 3 .$ $\begin{array} { r } { \mathrm { R e p } \ = \ \frac { 1 } { N - 2 } \sum _ { t = 1 } ^ { N - 2 } { \bf 1 } [ a _ { t } \ = \ a _ { t + 2 } ] } \end{array}$ , averaged over episodes.

Baselines. All methods use the same episodes, action budget, final-view VQA model, and judge. We compare direct camera control using GPT-5.1 [21], Gemini 3.0 Flash [10], and Qwen3.6-27B [24]. These agents receive only the current view, as the benchmark found that additional views do not change their scores [30]. To compare value supervision with action imitation, we also fine-tune Qwen3.6-27B and $\pi _ { 0 . 5 }$ [23] to clone the selected actions of the same teacher trajectories. As a baseline that predicts future views at deployment, Cosmos3 (Planning) generates each candidate’s future view with Cosmos3-Nano [20] fine-tuned on the same teacher trajectories and scores it with the zero-shot probe.

Implementation. Observations are 512 × 512 RGB, $T _ { \mathrm { m a x } } = 2 5$ , and camera motions use 0.25 m translation and $3 0 ^ { \circ }$ rotation increments. Both the teacher and the student use Qwen3.6-27B [24]. Rank-16 LoRA adapters (α = 32, w/o dropout) on all language-model linear projections give 117M trainable parameters; the vision encoder is frozen. We optimize Eq. (5) on approximately 132k teacher-labeled state–action pairs with AdamW (constant learning rate $1 0 ^ { - 4 }$ gradient-norm clipping 1.0). Training runs data-parallel on two RTX PRO 6000 Blackwell GPUs with bfloat16 weights and gradient checkpointing, one state (about ten candidate probes) per GPU per step. Stage 1 trains on $Q _ { 1 }$ for up to three epochs and stage 2, resuming from the selected stage-1 checkpoint, on $Q _ { 2 }$ for up to two; each stage keeps the per-epoch checkpoint with the lowest BCE on 200 held-out validation states. Unless otherwise noted, results are reported on the test set.

TABLE I: Main results on the E3VS-Bench test set. Green is best and light green is second best per column. OS = object search, OST = object state, OA = object attribute, CGS = context-guided search, SR = spatial reasoning, CNT = counting.
<table><tr><td>Method</td><td>OS OST</td><td>OA</td><td>CGS</td><td>SR</td><td>CNT</td><td></td><td>Avg. ↑ Coll. ↓</td></tr><tr><td>Proprietary</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.1</td><td>3.34 2.73 2.96</td><td></td><td></td><td>2.87 2.38</td><td>2.28</td><td>2.81</td><td>0.38</td></tr><tr><td>Gemini 3.0 Flash</td><td>3.14</td><td>3.09</td><td>2.67</td><td>2.47</td><td>2.25 1.96</td><td>2.62</td><td>0.43</td></tr><tr><td>Open-source</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Cosmos3 (Planning)</td><td>2.83 2.18 2.38</td><td></td><td></td><td>2.33 2.25</td><td>1.92</td><td>2.36</td><td>0.50</td></tr><tr><td>Qwen3.6-27B</td><td>2.791.82</td><td></td><td>2.09</td><td>2.47 1.50</td><td>1.64</td><td>2.14</td><td>0.55</td></tr><tr><td>π0.5</td><td>2.59 2.18</td><td></td><td>2.24</td><td>1.93 1.75</td><td>1.76</td><td>2.15</td><td>0.19</td></tr><tr><td>RFL (Ours)</td><td>3.55</td><td>2.55</td><td>2.89</td><td>3.67 2.88 2.64</td><td></td><td>3.05</td><td>0.11</td></tr></table>

![](images/c9753b0eaf44c88ee720faa17a4de9dcbc820ecde8391a8c239a0e9350c70c42.jpg)  
Fig. 4: Answerability along test rollouts for the agents of Table I (bands: ±1 SE; dashed: mean at goal views).

## B. Main Results

Table I compares RFL with direct action prediction, action imitation, and future-view generation. RFL achieves a mean judge score of 3.05 against 2.14 for direct action prediction with the same base VLM and stopping mechanism, so the learned viewpoint-selection policy accounts for a large part of the gain. Cosmos3 (Planning), which generates candidate future views, scores 2.36, and GPT-5.1, the strongest proprietary baseline evaluated, 2.81.

RFL is highest on four of the six question types, while GPT-5.1 leads OA (2.96 vs. 2.89) and Gemini 3.0 Flash leads OST (3.09 vs. 2.55), discussed in Sec. V-G.

To trace the source of this gain, Fig. 4 probes every view visited on the test set with the frozen probe (Eq. 1). The dashed line marks the mean answerability at the humanannotated goal views, where each question is answerable. RFL raises answerability from the first steps and reaches the goal-view level by episode end, whereas GPT-5.1 and Gemini 3.0 Flash climb slowly to well below it and Cosmos3 (Planning) stays near its initial value for most of the episode. RFL thus steadily moves toward viewpoints from which the question becomes answerable, which underlies its higher scores in Table I.

TABLE II: Output and fine-tuning of the same VLM (Qwen3.6-27B) with single-view conditioning.
<table><tr><td>Output</td><td>FT</td><td>Avg. ↑</td><td>Steps ↓</td><td>Rep. ↓</td></tr><tr><td rowspan="2">Actions</td><td></td><td>2.14</td><td>20.1</td><td>0.78</td></tr><tr><td>√</td><td>2.47</td><td>24.9</td><td>0.87</td></tr><tr><td rowspan="2">Answerability</td><td></td><td>2.04</td><td>18.8</td><td>0.68</td></tr><tr><td>√</td><td>2.71</td><td>17.2</td><td>0.62</td></tr></table>

## C. Answerability as a Supervision Signal

RFL is supervised entirely by probed answerability, so we verify that this signal tracks search success before attributing the gains to it (Fig. 3). We apply the frozen Qwen3.6-27B probe [24] (Eq. (1)) to every visited view in Gemini 3.0 Flash rollouts [10] on the 231 validation episodes (90 successes, 141 failures), disjoint from the test set. Mean answerability is 0.334 at initial viewpoints and 0.685 at ground-truth goal viewpoints. Over normalized episode progress, it rises toward the goal-view level in successful episodes but drops early and stays low in failed ones. Answerability distinguishes successful from failed episodes with an AUROC of 0.824, versus 0.719 for privileged target-object IoU and 0.743 for episode length. These results support using view-level answerability to construct the teacher’s lookahead targets.

## D. Ablation Studies

We isolate four design and data choices: what the model outputs, how far the targets look ahead, what the probe observes, and which trajectories supply the training states. All model and hyperparameter choices were made on the validation split; test-set ablations are post-hoc analyses not used for model selection.

Action-value supervision. Table II compares direct action prediction and answerability-based action scoring, with and without fine-tuning, on the same Qwen3.6-27B backbone. All four variants observe only the current view, and both fine-tuned models use the same teacher trajectories: action prediction imitates the selected motions, whereas value prediction learns answerability targets for all candidate motions. The latter is the single-view variant of RFL; its full-history counterpart is the deployed model (3.05).

Without fine-tuning, answerability scoring trails direct action prediction (2.04 vs. 2.14), and fine-tuning lifts value prediction to 2.71 against 2.47 for action cloning. The value-based policy is also shorter (17.2 vs. 24.9 steps) and less repetitive (0.62 vs. 0.87). These results support learning candidate-wise action values rather than imitating the teacher’s selected motions.

![](images/4e68feb30a532bab99c6bbdc81d4107e1d46bbfc15c814d3b061db1df1a8112e.jpg)  
(3 epochs) (2+1 epochs)

![](images/49491d47c3cb8eb7bf29d2bf24f948755dd9c7ce0d18741eecf1a74eaa5db17f.jpg)

![](images/092cc44ef4bb63387d33005bdd7d148288e9164d9e1fa01a6d2c879324b98fb7.jpg)  
Fig. 5: Two-step supervision at an Fig. 6: Probe conditioning with the Fig. 7: Effect of teacher trajectories equal three-epoch budget (test split). same two-stage recipe (validation split). (test split).

TABLE III: Render oracles vs. RFL.
<table><tr><td>Method</td><td>Horizon</td><td>Avg. ↑</td><td>Steps ↓</td><td>Rep. ↓</td></tr><tr><td rowspan="2">RFL</td><td>1</td><td>2.98</td><td>16.0</td><td>0.50</td></tr><tr><td>2</td><td>3.05</td><td>16.1</td><td>0.46</td></tr><tr><td rowspan="2">Oracle</td><td>1</td><td>2.98</td><td>14.5</td><td>0.51</td></tr><tr><td>2</td><td>3.19</td><td>13.6</td><td>0.37</td></tr></table>

Lookahead horizon. To examine how much lookahead depth survives without rendering, Table III compares oneand two-step variants of RFL with privileged rendering oracles. The oracles apply the teacher at inference, rendering each candidate’s actual future view, scoring it with the frozen VLM, and executing the highest-scoring motion; RFL instead predicts action values without observing future views.

Going from one to two steps raises RFL from 2.98 to 3.05 and the oracle from 2.98 to 3.19: RFL matches the oracle at one step and trails it by 0.14 at two. The two-step oracle is also shorter (13.6 vs. 16.1 steps) and less repetitive (0.37 vs. 0.46). To separate two-step supervision from additional training, we fix the total budget at three epochs (Fig. 5). Three epochs on $\mathcal { L } _ { 1 }$ alone reach 2.97, whereas the deployed schedule of two epochs on $\mathcal { L } _ { 1 }$ and one on $\mathcal { L } _ { 2 }$ reaches 3.05 (the one-step entry of Table III, 2.98, differs only in its validation-selected two-epoch checkpoint). This controlled comparison indicates that the modest gain is not due to optimization alone and is consistent with a benefit from the two-step targets. Together with the oracle’s larger horizon gain, this suggests that two-step lookahead provides useful supervision whose full benefit is not yet transferred into a rendering-free value predictor.

Observation history. We compare three input configurations on the validation set under the same two-stage recipe (Fig. 6). All variants receive the question and candidate motion, plus either the current view, the current view and the last three executed motions, or recent observations. Action history raises the mean judge score from 2.75 to 2.96, and visual history to 3.10, supporting its use in the final model.

Training trajectories. We compare RFL trained on states along shortest paths, planned with A<sup>∗</sup> to the goal viewpoint, with RFL trained on lookahead-teacher trajectories (Fig. 7). Under the same two-stage recipe, lookahead trajectories reach 3.05 against 2.80 for shortest paths, so the states induced by the answerability-driven teacher are more effective for training than geometrically efficient paths.

## E. Qualitative Results

To see how the learned value shapes search behavior, Fig. 8 shows three test episodes. In the left two, RFL moves directly to a viewpoint that reveals the queried information, descending to the power strip and closing in on the travel mug, and answers correctly, whereas Gemini 3.0 Flash never reaches such a viewpoint and answers both incorrectly. The right episode is a failure case: RFL reaches a view that exposes the text on the sack yet fails to stop there and ends at a viewpoint from which the text is no longer readable. Reaching an informative viewpoint is thus not sufficient: the policy must also learn when to stop, a decision RFL currently leaves to the base VLM. To quantify this effect, we reanswer each episode from the visited view with the highest probed answerability, which raises the score from 3.05 to 3.16 and narrows the gap to the two-step rendering oracle (3.19, Table III) from 0.14 to 0.03. A substantial part of the remaining oracle gap may therefore stem from stopping rather than viewpoint selection.

## F. Real-World Experiments

Finally, we test whether the simulation-trained policy transfers to physical camera control, a deployment that rendering-free inference makes practical. The platform is a Seeed Studio reBot Arm B601-DM, an open-source 6-DoF arm with an end-effector camera, whose workspace covers the benchmark’s 5-DoF viewpoint actions. We evaluate three viewpoint-dependent questions with three methods, RFL, the base VLM acting directly, and the GPT-5.1 agent, running ten trials for every method–question pair while varying the object placement and initial arm pose, i.e., 3 questions × 3 methods × 10 trials = 90 trials in total (Fig. 9). All trials use $T _ { \mathrm { m a x } } ~ = ~ 1 5$ , the simulation increments of 0.25 m and $3 0 ^ { \circ }$ , and the same scoring protocol on the final onboard view. RFL achieves the highest mean judge score over the three questions, reaching 3.80 on the pot and 3.00 on the oven question, whereas the direct base VLM stays at 1.40 on all three. GPT-5.1 reaches 3.40 on the pot question but falls to 1.80 on the oven question, and on the bluepackage question, which requires reading the text on its front, it edges out RFL (2.60 vs. 2.20). These experiments provide initial evidence of such transfer without test-time rendering. The arm’s embodiment, however, limits execution:

Q: how many orange-colored sockets are on the power strip?  
![](images/6a339056ccefd76c40c3cff6e306e51787780241fe138a8daf92010897c0250b.jpg)

Q: what is the color of the lid on the travel mug?  
![](images/cdb53f68c8c8f540f34982e5489d4885dd8b30e20b6915828041e96203e0d602.jpg)

Q: what text is visible on the front of the plastic sack?  
![](images/1798d18cbca3e8dff05b4db80c1eec8427836c8297d21103d24df4d57a6b5b59.jpg)  
Fig. 8: Qualitative comparison on E3VS-Bench (RFL: green; Gemini 3.0 Flash: magenta). Full answers (abbreviated in the figure): middle, RFL: “The lid on the travel mug is beige.”; right, Gemini: “The text on the front of the plastic sack is not legible.”, RFL: “The text is not clear enough to be read.”

motions infeasible from the current pose are removed from RFL’s candidates before selection, whereas the direct base VLM kept generating them even after they were excluded from its action list. A model’s intended motion can thus be unexecutable for a given embodiment, which calls for policies that account for it.

## G. Discussion and Limitations

RFL requires counterfactual rendering at training time. Any free-viewpoint renderer (mesh, NeRF, 3DGS) would serve, whereas logged trajectories alone label only executed actions, weakening the dense counterfactual supervision. The recipe is off-policy privileged value distillation: the student trains on the teacher’s visit distribution and deploys on its own, yet this covariate shift appears cheap, as RFL matches the one-step render oracle scoring its on-policy states (Table III). The remaining weakness is fine-grained evidence: OST is RFL’s weakest type (2.55 vs. 3.09 for Gemini 3.0 Flash), as it hinges on details requiring precisely placed close views. Stopping is a further limitation: as the OA failure in Fig. 8 shows, RFL can reach a view exposing the answer yet fail to stop there, and the best-visited-view analysis (Sec. V-E) attributes much of the oracle gap to this decision, left to the base VLM. Real-world experiments add an embodiment limitation, as motions feasible in simulation are not always executable by the arm.

## VI. CONCLUSION

We presented RFL, a rendering-free viewpoint-planning method for Embodied 3D Visual Search. Lookahead is confined to a privileged teacher that counterfactually renders every candidate action in 3DGS training scenes and scores it by two-step answerability beam search. The same VLM’s action-conditioned Yes/No probability is then calibrated to these measured values by two-stage soft-label distillation, so the deployed policy selects viewpoints greedily with a single text probe per candidate action, without rendering, world models, or generation. On E3VS-Bench, RFL raised the direct baseline’s mean judge score from 2.14 to 3.05 and outperformed action cloning trained on the same supervision. Ablations support the recipe, as the zero-shot probe underperforms even the direct baseline. In real-world experiments on a 6-DoF arm, the simulation-trained policy achieved the highest mean judge score across three viewpoint-dependent questions.

Future work includes calibrated stopping, and finergrained or continuous action spaces for physical platforms.

## VII. ACKNOWLEDGMENTS

This work was supported by JST PRESTO (Grant Number JPMJPR22P8), JST SPRING (Grant Number JPMJSP2108), JSPS KAKENHI (Grant Number 25K03177), Japan. The first author used Claude Code (Anthropic) for manuscript drafting and code generation, and ChatGPT (OpenAI) for language editing and proofreading. AI-assisted outputs were reviewed and revised by the first author as appropriate, who takes responsibility for the final manuscript and implementation.

## REFERENCES

[1] Aloimonos, J., Weiss, I., Bandyopadhyay, A.: Active vision. International Journal of Computer Vision 1, 333–356 (1988)

[2] Azuma, D., Miyanishi, T., Kurita, S., Sakamoto, K., Kawanabe, M.: Answerability fields: Answerable location estimation via diffusion models. In: IROS (2024)

[3] Azuma, D., Miyanishi, T., Sakamoto, K., Kurita, S., Zhu, Y., Khrapchenkov, P., Kawanabe, M., Iwasawa, Y., Matsuo, Y.: Navwam: A navigation world action model for goal-conditioned visual navigation (2026), https://arxiv.org/abs/2606.13494

[4] Bai, H., Zhou, Y., Li, E.L., Levine, S., Kumar, A.: Digi-q: Transforming vlms to device-control agents via value-based offline rl. In: ICLR (2025)

[5] Bajcsy, R.: Active perception. Proceedings of the IEEE 76(8), 966– 1005 (1988)

![](images/1fd69b8fdb8c38d2b83a725e756e6c42f32ee7d3e88e0dea586143606770e74b.jpg)

![](images/e7ae6970dda28b8054212b7f13b8c9d5e4fc17d45006996663fecf04725e31be.jpg)

Q: Is there a bread in the oven?  
![](images/6ccf4e0fc05c6b1223cf49822b636b9af1d8835a5dc32968c63c328b94b795e8.jpg)

![](images/0d659ac02f3c614bbf7e235a6987b8fbbe11e59b48022d9f8cd5adbb6d8d900b.jpg)  
A: An eggplant.

![](images/ff0b85022922afc37856a6a71450ef65e575cd86f38b8c3968eae04a54e26d37.jpg)  
A: KRE-ALKALYN CREATINE.

![](images/07222b628d9bfd2adf40091e970516f99236fc8dfb758ceeb7e94032cce37ee7.jpg)  
A: Yes.

Fig. 9: Real-world experiments. Mean judge score per question for RFL, the direct base VLM (Base), and the GPT-5.1 agent. Each bar averages ten trials of one method on one question, i.e., 3 questions × 3 methods × 10 trials = 90 trials.

[6] Bar, A., Zhou, G., Tran, D., Darrell, T., LeCun, Y.: Navigation world models. In: CVPR (2025)

[7] Chen, D., Zhou, B., Koltun, V., Krahenb¨ uhl, P.: Learning by cheating.¨ In: CoRL. pp. 66–75 (2019)

[8] Chen, H., Jiang, S., Su, T., Gao, C., Chen, X., Li, Y., Chen, Z.: Worldmap: Bootstrapping vision-language navigation trajectory prediction with generative world models (2026), https://arxiv. org/abs/2604.07957

[9] Das, A., Datta, S., Gkioxari, G., Lee, S., Parikh, D., Batra, D.: Embodied Question Answering. In: CVPR (2018)

[10] Google DeepMind: A new era of intelligence with gemini 3. https://blog.google/products/gemini/gemini- 3/ (Nov 2025), accessed: 2026-09-13

[11] Hong, J., Dragan, A., Levine, S.: Q-sft: Q-learning for language models via supervised fine-tuning. In: ICLR (2025)

[12] Kerbl, B., Kopanas, G., Leimkuhler, T., Drettakis, G.: 3D Gaussian¨ splatting for real-time radiance field rendering. TOG 42(4), 1–14 (July 2023)

[13] Kerr, J., Hari, K., Weber, E., Kim, C.M., Yi, B., Bonnen, T., Goldberg, K., Kanazawa, A.: Eye, robot: Learning to look to act with a bc-rl perception-action loop. In: CoRL (2025)

[14] Koh, J.Y., Lee, H., Yang, Y., Baldridge, J., Anderson, P.: Pathdreamer: A world model for indoor navigation. In: ICCV (2021)

[15] Koo, J., Choi, D., Youn, S., Lee, P.Y., Sung, M.: Toward ambulatory vision: Learning visually-grounded active view selection (2025), https://arxiv.org/abs/2512.13250

[16] Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., Hutter, M.: Learning quadrupedal locomotion over challenging terrain. Science Robotics 5(47), eabc5986 (2020). https://doi.org/10.1126/scirobotics.abc5986

[17] Lu, Y., Du, Y., Liu, D., Zhou, Y., Wang, C., Yin, Y.: Gsmem: 3d gaussian splatting as persistent spatial memory for zero-shot embodied exploration and reasoning (2026), https://arxiv.org/abs/ 2603.19137

[18] Ma, M., Ma, Q., Li, Y., Cheng, J., Yang, R., Ren, B., Popovic, N., Wei, M., Sebe, N., Van Gool, L., Gevers, T., Oswald, M.R., Paudel, D.P.: Scenesplat++: A large dataset and comprehensive benchmark for language gaussian splatting. In: NeurIPS (2025)

[19] Majumdar, A., Ajay, A., Zhang, X., Putta, P., Yenamandra, S., Henaff, M., Silwal, S., Mcvay, P., Maksymets, O., Arnaud, S., Yadav, K., Li, Q., Newman, B., Sharma, M., Berges, V., Zhang, S., Agrawal, P., Bisk, Y., Batra, D., Kalakrishnan, M., Meier, F., Paxton, C., Sax, S., Rajeswaran, A.: Openeqa: Embodied question answering in the era of foundation models. In: CVPR (2024)

[20] NVIDIA: Cosmos 3: Omnimodal world models for physical AI (2026), https://arxiv.org/abs/2606.02800

[21] OpenAI: Gpt-5.1: A smarter, more conversational chatgpt. https:

//openai.com/index/gpt-5-1/ (11 2025), accessed: 2026- 09-15

[22] OpenAI: Introducing gpt-5.5. https://openai.com/index/ introducing-gpt-5-5/ (4 2026), accessed: 2026-08-19

[23] Physical Intelligence, Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., et al.: π<sub>0.5</sub>: a vision-language-action model with open-world generalization (2025), https://arxiv.org/abs/2504.16054

[24] Qwen Team: Qwen3.6-27b: Flagship-level coding in a 27b dense model. https://qwen.ai/blog?id=qwen3.6-27b (4 2026), accessed: 2026-08-19

[25] Ramakrishnan, S.K., Grauman, K.: Sidekick policy learning for active visual exploration. In: ECCV (2018)

[26] Ren, A.Z., Clark, J., Dixit, A., Itkina, M., Majumdar, A., Sadigh, D.: Explore until confident: Efficient exploration for embodied question answering. In: RSS (2024)

[27] Ruoss, A., Deletang, G., Medapati, S., Grau-Moya, J., Wenliang, L.K.,´ Catt, E., Reid, J., Lewis, C.A., Veness, J., Genewein, T.: Amortized planning with large-scale transformers: A case study on chess. In: NeurIPS (2024)

[28] Rusu, A.A., Colmenarejo, S.G., Gulcehre, C., Desjardins, G., Kirkpatrick, J., Pascanu, R., Mnih, V., Kavukcuoglu, K., Hadsell, R.: Policy distillation. In: ICLR (2016)

[29] Sakamoto, K., Azuma, D., Miyanishi, T., Kurita, S., Kawanabe, M.: Map-based modular approach for zero-shot embodied question answering. In: IROS (2024)

[30] Sakamoto, K., Miyanishi, T., Azuma, D., Kurita, S., Morikuni, S., Chiba, N., Kawanabe, M., Iwasawa, Y., Matsuo, Y.: E3VS-Bench: A benchmark for viewpoint-dependent active perception in 3D Gaussian splatting scenes. In: ECCV (2026)

[31] Saxena, S., Buchanan, B., Paxton, C., Liu, P., Chen, B., Vaskevicius, N., Palmieri, L., Francis, J., Kroemer, O.: Grapheqa: Using 3d semantic scene graphs for real-time embodied question answering. In: CoRL (2025)

[32] Wang, K., Li, L., Yang, Z., Chen, S., Wang, Z., Fei-Fei, L., Wu, J., Guibas, L., Wang, L., Li, M.: Planning with the views (2026), https://arxiv.org/abs/2605.29563

[33] Wang, Z., Li, X., Yang, J., Liu, Y., Hu, J., Jiang, M., Jiang, S.: Lookahead exploration with neural radiance representation for continuous vision-language navigation. In: CVPR (2024)

[34] Yuan, T., Dong, Z., Liu, Y., Zhao, H.: Fast-wam: Do world action models need test-time future imagination? (2026), https://arxiv. org/abs/2603.16666