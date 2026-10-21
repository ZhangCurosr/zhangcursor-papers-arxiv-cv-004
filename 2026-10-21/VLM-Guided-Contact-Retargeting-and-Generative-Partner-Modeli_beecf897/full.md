# VLM-Guided Contact Retargeting and Generative Partner Modeling for Humanoid-Human Interaction

Yuto Shibata<sup>1,2,3∗</sup> Kotaro Amaya<sup>1,2,3∗</sup> Kashu Yamazaki<sup>1</sup> Lalit Jayanti<sup>1</sup> Yoshimitsu Aoki<sup>2,3</sup> Mariko Isogawa<sup>2,3</sup> Katerina Fragkiadaki<sup>1</sup>

<sup>1</sup>Carnegie Mellon University <sup>2</sup>Keio AI Research Center <sup>3</sup>Keio University <sup>∗</sup>Equal contribution

https://yutoshibata07.github.io/REACT/

![](images/e789530544c925d042f879223b8508f4509f0d136a8ea82e0c3762e6393f3449.jpg)  
Figure 1: Trained in simulation with contact-preserving retargeting and human depth rendering (left), our policy runs zero-shot on a real Unitree G1 (middle: high-five with depth input; right: others).

Abstract: Humanoid–human interaction requires robots to operate under severe partial observability: during close physical contact, the human partner is occluded, truncated, or outside the robot’s field of view, causing pose estimation to fail and often necessitating external tracking or privileged human state. At the same time, heavy mutual occlusions introduce errors into the human–human motion-capture data used for learning. We present REACT, a framework for learning deployable humanoid–human interaction policies from human–human demonstrations. REACT treats semantic contact structure as the key supervisory signal connecting demonstration processing, interaction learning, and onboard control. First, a VLM-guided contact modeling and retargeting pipeline identifies semantically meaningful contacts and transfers them onto humanoid–human geometry while preserving interaction structure. Using these annotations, privileged interaction policies learn physically grounded behaviors through interaction-aware contact rewards and are then distilled into deployable egocentric depth policies trained entirely from onboard sensing using photorealistic human depth rendering. To address severe partial observability, we introduce generative partner-state modeling objectives that predict partner motion from occluded egocentric views. We evaluate REACT in simulation and on a real Unitree G1 humanoid across five contactrich interactions. Semantic contact modeling substantially improves preservation of demonstrated interactions, while generative partner-state modeling improves contact recall on four of five interactions, most under heavy occlusion. On real hardware, REACT performs all five interactions using onboard sensing alone.

Keywords: Humanoid robots, Humanoid-Human interaction

## 1 Introduction

Humanoid hardware and learning-based control have advanced rapidly in recent years, bringing the deployment of humanoid robots in homes, workplaces, and care settings increasingly within reach. Realizing this vision, however, requires robots capable of sustained, reactive physical interaction with humans using only onboard sensing. Unlike existing humanoid controllers, which primarily focus on isolated motion tracking or interaction with static scenes and objects [1, 2, 3], humanoid–human interaction (HHI) requires continuous perception, prediction, and physical coordination with a human partner whose motion evolves under independent intent and frequent contact.

A central challenge in HHI is that close-contact interaction fundamentally operates under severe partial observability. During interaction, the human partner is frequently self-occluded, truncated, or partially outside the robot’s field of view, causing off-the-shelf human pose-estimation systems to fail precisely when accurate perception matters most. As a result, many existing approaches rely on external tracking systems, motion-capture instrumentation, or privileged access to human state during deployment, limiting robustness and real-world deployability. The problem begins even earlier in the learning pipeline: heavy mutual occlusions also corrupt the underlying human–human motion-capture data itself, introducing reconstruction artifacts and inconsistent contact geometry tha propagate through downstream retargeting [4, 5] and policy-learning pipelines.

We argue that deployable humanoid–human interaction fundamentally requires recovering the interaction state from incomplete observations, and that semantic contact structure provides the key supervisory signal connecting noisy demonstrations, physics-based learning, and onboard control. While motion trajectories may vary substantially across embodiments, interactions are often defined by a much more stable structure: who touches whom, where, and when. We therefore propose to represent interactions through semantic contacts and train policies to infer latent partner state under severe partial observability.

To this end, we introduce REACT (Reactive Egocentric Agent for Contact-rich Humanoid-Human InTeraction), a framework for learning deployable humanoid–human interaction policies from largescale human–human motion-capture data (Figure 1). REACT first runs a VLM-guided semantic contact pipeline that retargets the demonstrated interaction contacts onto humanoid–human geometry while preserving interaction structure across the embodiment gap. These semantic contacts then supervise a privileged interaction teacher through interaction-aware contact rewards, anchoring the policy in physics. The teacher is distilled into an egocentric depth student trained entirely on onboard sensing via photorealistic in-simulator SMPL-X rendering of dynamic human partners. To address severe partial observability, the student is further trained with a generative partner-state prediction objective that infers past, present, and future partner motion from egocentric observations.

We evaluate REACT on a large corpus of close-contact human–human interactions in simulation and on a real Unitree G1 humanoid. Our experiments show that semantic contact modeling substantially improves preservation of demonstrated interaction structure compared to existing retargeting approaches, and its contact annotations improve downstream interaction learning. We further show that generative partner-state prediction improves contact recall on four of five interactions, most where the partner is hardest to see, enabling depth-based policies to recover most of the contact recall of privileged interaction teachers (75% at a 30 cm tolerance). On real hardware, REACT successfully executes diverse contact-rich interactions, including high-fives, hugs, handshakes, link-arms, and pats on the back, using onboard sensing alone.

In summary, our contributions are (1) A semantic contact modeling and retargeting framework that transfers human–human demonstrations to humanoid–human interactions while preserving interaction structure across embodiments, (2) An interaction-learning framework that uses semantic contacts as supervision for physics-based humanoid–human behavior learning, (3) Generative partner-state prediction objectives that enable latent interaction-state inference under severe partial observability from egocentric observations, and (4) extensive simulation ablations and real-world deployment on a Unitree G1 across diverse contact-rich humanoid–human interactions.

## 2 Related Work

Humanoid motion tracking and scene interaction Recent work has achieved impressive progress in humanoid teleoperation and motion tracking through imitation learning, teacher–student distillation, and large-scale behavioral modeling [1, 6, 2, 7, 8, 9, 10, 11, 12]. A growing body of work further extends humanoid control toward object and scene interaction through manipulation and locomanipulation, physics-based human–object interaction, scene-conditioned motion generation, and visual sim-to-real learning [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]. However, these methods either treat the humanoid in isolation or restrict interaction to static scenes and inanimate objects, whereas REACT targets closed-loop humanoid–human interaction with a dynamically moving human partner.

Interaction Modeling and Human–Human Motion Synthesis Human–human interaction synthesis has advanced rapidly through diffusion-based and generative motion models. Recent methods such as InterMask [24], PINO [25], UMO [26], and MAGNet [27] further improve motion diversity, long-horizon interaction synthesis, and flexible multi-agent conditioning. However, these approaches primarily operate in kinematic space and therefore lack physical grounding and closed-loop interaction control. More recent physics-based interaction methods couple diffusion-based motion generation with physics-based tracking [28, 29] or forward-dynamics-guided imitation [30], but they generate the reference interaction offline [29] or observe the partner fully, either in simulation [30] or through an unoccluded third-person capture [28], rather than through deployable onboard sensing.

Interaction retargeting Motion retargeting offers a promising avenue for transferring large-scale human interaction data to humanoid robots. Optimization-based methods, ranging from inverse kinematics to trajectory optimization [31, 32, 4], adapt the source motion to the humanoid’s kinematics frame by frame or over the whole trajectory, while learning-based retargeting [33] improves stylistic consistency and motion realism. However, because their objectives are defined on each body separately, they carry no notion of which body parts are meant to touch the partner, so contacts present in the source interaction are easily lost once the motion is transferred to a different embodiment. Graphics-oriented approaches have explored geometry-aware and contact-aware retargeting objectives [34, 35], while recent interaction-preserving methods [5] emphasize maintaining relational geometry for downstream policy learning. Beyond Mimicry [36] retargets human–human interactions to a humanoid by preserving pairwise keypoint distances between the two agents, without reasoning about which of them correspond to physical contact. In contrast, REACT introduces semantic contact-aware retargeting that explicitly preserves interaction contacts and relational geometry while suppressing interpenetration between the humanoid and its partner.

Interaction under Partial Observability Closer to our setting, recent work has approached human– robot interaction by learning from human–human–object demonstrations [37], forecasting social motion to generate robot behavior [38], and adding compliance to whole-body control for physical contact with people [39]. However, these systems depend on robot-side teleoperation data [37], external motion capture [39], or a separate human-pose estimator [36, 28], which becomes unreliable at the close range where physical contact occurs. Privileged-to-perceptual distillation has produced visual humanoid policies [20, 21, 19], but only for interaction with objects and scenes rather than a moving human partner. In contrast, REACT maps egocentric depth to actions end-to-end and explicitly trains policies to infer hidden partner motion under occlusion through generative partnerstate modeling.

## 3 Method

We formulate deployable humanoid–human interaction as recovering and preserving interaction state across three stages: (i) inferring semantic contact structure from noisy human–human demonstrations, (ii) learning physically grounded interaction policies that reproduce this contact structure under physics, and (iii) inferring partner state from egocentric observations during deployment.

![](images/dd259319bec9c589f57043fa9191602b92cea2ddfd9a808514130c8faab06d91.jpg)  
Figure 2: Overview of REACT. REACT first annotates semantic contacts in human–human demonstrations with a VLM and retargets them onto the humanoid while preserving them. A privileged teacher then learns the interaction with contact rewards and is finally distilled into an egocentric depth student that predicts the partner’s motion and runs on the real robot.

Figure 2 provides an overview of REACT. Given a human–human motion-capture sequence, REACT learns a humanoid policy that reproduces the demonstrated interaction from onboard sensing alone, in three stages. First, semantic contact modeling identifies meaningful contacts and transfers them across the embodiment gap into humanoid–human references. Second, a privileged teacher learns the interaction with contact rewards derived from these annotations. Finally, a deployable egocentric student is distilled from the teacher using only onboard depth and infers partner state through generative prediction.

## 3.1 Semantic Contact Modelling and Retargeting

Existing retargeting methods optimize pose similarity or relational geometry but do not explicitly reason about the contact structure that defines an interaction [4, 5]. REACT instead represents an interaction as a collection of spatiotemporal semantic contacts and preserves them throughout embodiment transfer. Semantic contact modeling proceeds in three steps, detailed below:

Human-to-humanoid surface correspondence Semantic contacts are expressed on SMPL-X body vertices, but the retargeted humanoid has a different morphology with no direct vertex correspondence. We therefore register the SMPL-X mesh to the humanoid surface using Laplacian-regularized nonrigid ICP (Fig. 2, Surface mesh projection), producing an offline, pose-independent mapping from every SMPL-X vertex to a corresponding humanoid surface location (link, p ). The resulting correspondence transfers semantic contacts from the human body onto the humanoid embodiment and provides the geometric reference used by both retargeting and interaction-reward computation.

VLM-guided semantic contact annotation A key challenge is that existing motion-capture datasets do not provide annotations of who touches whom, where, and when. Furthermore, reconstructed human meshes are often noisy under heavy mutual occlusion, making geometric proximity alone an unreliable indicator of interaction contact. Simple distance thresholds either miss meaningful contacts or produce large numbers of spurious contact events.

We therefore separate geometric contact discovery from semantic verification. A permissive meshdistance pass first proposes candidate spatiotemporal contacts between the two interacting humans. These candidates are subsequently verified by a vision-language model (Gemini 2.0 Flash), which receives multi-view renderings of the interaction, the candidate contact list, and a description of the interaction type depicted in the video. The VLM rejects contacts inconsistent with the semantics of the demonstrated interaction and retains only meaningful contact events (Fig. 2, Distance based contacts to VLMfiltered contacts). For each contact site h at timestep t, the pipeline emits a flag $c _ { t } ^ { h }$ , the actor vertex $v _ { t } ^ { h }$ , and the contacted partner vertex $j _ { t } ^ { h }$ used downstream by retargeting and the contact reward. Implementation details and the annotation-quality evaluation are in Appendix D.1, and the VLM prompt is in Appendix H.

Semantic contact retargeting Given surface correspondences and semantic contact annotations, we retarget human–human demonstrations into humanoid–human interaction references while explicitly preserving demonstrated contact structure across the embodiment gap.

Stage 1: Interaction-aware kinematic retargeting. We begin with an OmniRetarget-style initialization [5] (Fig. 2, Interaction preserved retargeting). Only the human mapped to the G1 is rescaled to the G1 embodiment, because rescaling the partner would move the contact targets it supplies and destabilize the G1 retarget. This breaks the inter-body interaction geometry across the embodiment gap, which we mitigate by anchoring the humanoid arm end-effector targets to the demonstrated contact locations on labeled frames.

Stage 2: Semantic contact driven human-humanoid co-optimization. The initialization is subsequently refined through a contact-preserving optimization stage (Fig. 2, Humanoid-human co-optimization). For every annotated contact $( c _ { t } ^ { h } = 1 )$ , the corresponding humanoid contact point is driven toward the target partner vertex $j _ { t } ^ { h }$ . At the same time, two complementary regularizers preserve the remainder of the interaction geometry. A distance-preserving proximity field maintains the demonstrated separation between non-contact body regions, preventing collapse of the interaction configuration, while a volumetric SDF penalty [40] suppresses interpenetration. Optimization alternates between humanoid and partner kinematics while preserving the locomotion generated during Stage 1.

The resulting trajectories preserve both the motion and the semantic contact structure of the original interaction. These retargeted demonstrations provide the reference motions used for policy learning and the semantic contact annotations used by the interaction rewards described in Sec. 3.2.1.

## 3.2 Interaction Learning from Semantic Contacts

We use the semantic contact annotations of Sec. 3.1 to learn a privileged interaction teacher that reproduces the demonstrated contact structure under physics.

## 3.2.1 Privileged Interaction Teacher

We train a privileged policy π using PPO [41] (Fig. 2, Teacher Policy). The teacher has access to the full state of the human partner and therefore serves as an upper bound on interaction performance before addressing the challenges of onboard perception. The teacher receives three groups of observations: (i) proprioceptive robot state ${ \mathbf o } _ { t } ^ { \mathrm { p r o p } }$ , including base motion, joint states, end-effector positions, and previous actions; (ii) reference-motion observations ${ \bf o } _ { t } ^ { \mathrm { r e f } }$ from the retargeted demonstrations; and (iii) privileged partner-state observations o<sup>partner</sup><sub>t</sub> containing the partner’s pose and motion expressed in the humanoid’s egocentric frame. More observation details are provided in Appendix F.

Rather than directly controlling all robot joints, the teacher predicts a hybrid action consisting of locomotion commands and upper-body interaction actions. Specifically, we leverage the pretrained HOMIE controller[3] for stable locomotion and predict only four locomotion commands $( v _ { x } , v _ { y } , \omega _ { z } , h )$ together with residual upper-body joint offsets. This factorization allows the policy to focus its capacity on learning contact-rich interaction while inheriting robust locomotion capabilities.

Interaction-Aware Contact Rewards Tracking the reference alone does not guarantee reproducing the interaction: a policy can match the reference trajectory while still failing to establish or maintain contact. We therefore supervise the teacher with the semantic contact annotations of Sec. 3.1 (Fig. 2, Vertex-anchored Contact Reward). Prior contact-based rewards rely on hand-annotated or geometricproximity contacts [29, 28]; ours instead consume the automatically extracted semantic contacts. The analytic form of the rewards is in Appendix F.1.

## 3.2.2 Deployable Perceptive Student

The privileged interaction teacher assumes among others access to the full state of the human partner and therefore cannot be deployed on a real humanoid. In practice, close-contact interaction must operate from onboard sensing alone, where severe occlusions, truncation, and limited field of view make the partner’s state only partially observable. We therefore distill the privileged teacher into a deployable egocentric student policy $\pi _ { S }$ (Fig. 2, Student Policy) that learns to infer interaction state directly from an onboard depth stream and proprioception history. Depth input images are processed by a shared ResNet encoder [42], while a temporal convolutional network aggregates long-horizon proprioceptive history into a compact motion-context representation. The resulting deployable observation embedding $\mathbf { c } _ { t }$ is passed to a conditional variational autoencoder. The student policy is trained using on-policy DAgger [43].

Photorealistic Human Rendering Most physics simulators represent humans with simplified collision primitives such as capsule colliders [44], producing depth observations that diverge sharply from real human depth. We close this gap with an in-loop rendering pipeline (Fig. 2, In-loop human rendering) that attaches a visual SMPL-X mesh to the partner: the capsule collider is kept for physics, while a batched SMPL-X forward pass updates the visual mesh every control step so the robot’s depth camera sees realistic human geometry, which closes the sim-to-real gap (Fig. 9 in Appendix C.1).

Generative Partner-State Modeling We additionally train the student to infer the hidden state of the interaction. In humanoid–human interaction, the partner is rarely fully observable: during close contact, large portions of their body are frequently occluded or outside the camera field of view. Consequently, many different partner configurations may be consistent with the same visual observation. We therefore train the student to predict how the interaction evolves from these partial observations: a generative partner-state prediction objective predicts the partner’s past, present, and future motion from the deployable observation representation $\mathbf { c } _ { t } .$ . A diffusion-based prediction head (Fig. 2, Interaction Denoiser) implements this objective, outputting the partner’s SMPL-X body pose, joint velocities, and orientations, $\mathsf { \bar { P } } _ { t } ^ { \mathsf { p a r t n e t } }$ , together with the humanoid’s short-horizon future trajectory, P<sup>humanoid</sup> (time offsets in Appendix C.3). These auxiliary predictions provide dense supervision for the visual encoder and encourage the learned latent representation to capture interaction-relevant information that is not directly observable from a single depth frame. Rather than reacting only to visible motion, the student learns to anticipate future interaction dynamics.

Training objective The final training objective combines teacher-action imitation, latent-state distillation, and generative partner-state prediction:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { B C } } + \lambda _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } + \lambda _ { \mathrm { g e n } } \mathcal { L } _ { \mathrm { g e n } } + \lambda _ { \mathrm { p o s e } } \mathcal { L } _ { \mathrm { p o s e } } ,\tag{1}
$$

where the individual loss terms correspond to behavior cloning, latent-state alignment, generative partner-state prediction, and single-step pose estimation, respectively. By jointly learning interaction control and inference of the occluded partner’s state, the resulting policy remains robust under severe occlusion and partial observability while operating entirely from onboard sensing.

## 4 Experiments

We evaluate REACT on both simulation and real-world humanoid–human interaction tasks. Our experiments are designed to answer five questions:

Q1. Does VLM-guided contact annotation improve interaction retargeting quality?

Q2. Do semantic contact annotations improve downstream interaction policy learning?

Q3. Does generative partner-state modeling improve robustness under partial observability?

Q4. Can an egocentric depth policy recover the performance of a privileged interaction teacher?

Q5. Does REACT transfer successfully to real humanoid hardware?

Table 1: Contact-aware retargeting
<table><tr><td>Method</td><td>Jitter ↓  $\mathrm { ( m / s ^ { 3 } ) }$ </td><td>Penetration ↓ (cm)</td><td>Recall ↑ (%)</td><td>Precision ↑ (%)</td><td>User Study ↓ (rank 1–4)</td></tr><tr><td>GMR OmniRetarget</td><td> $\mathbf { 1 3 4 . 1 { \_ } } _ { 0 6 }$   $5 6 0 . 4 { \scriptstyle \pm 3 4 1 }$ </td><td> $2 . 1 4 _ { \pm 3 . 0 5 }$   $1 . 3 1 _ { \pm 1 . 8 5 }$ </td><td> $0 . 4 2 _ { \pm 3 . 7 0 }$   $2 2 . 1 3 _ { \pm 3 4 . 2 4 }$ </td><td> $2 . 2 6 _ { \pm 9 . 7 2 }$   $1 2 . 5 1 _ { \pm 1 8 . 4 7 }$ </td><td> $3 . 6 1 _ { \pm 0 . 0 4 }$   $2 . 5 7 _ { \pm 0 . 0 4 }$ </td></tr><tr><td>Ours w/o Stage 2 co-opt</td><td> $1 5 3 . 7 { \scriptstyle \pm 7 1 }$ </td><td> $1 . 3 1 _ { \pm 1 . 7 3 }$ </td><td> $1 3 . 3 9 _ { \pm 2 6 . 5 6 }$ </td><td> $1 1 . 7 9 _ { \pm 1 9 . 5 8 }$ </td><td></td></tr><tr><td>Ours w/o VLM filter</td><td> $1 6 1 . 7 _ { \pm 7 6 }$ </td><td> $1 . 0 4 _ { \pm 1 . 1 1 }$ </td><td> $5 8 . 0 2 _ { \pm 4 1 . 7 1 }$ </td><td> $1 1 . 6 0 _ { \pm 1 4 . 3 9 }$ </td><td> $2 . 0 6 _ { \pm 0 . 0 3 }$ </td></tr><tr><td>Ours</td><td> $1 5 5 . 4 { \scriptstyle \pm 7 1 }$ </td><td> $\mathbf { 0 . 9 6 _ { \pm 1 . 1 0 } }$  </td><td> ${ \bf 5 9 . 4 3 _ { \pm 3 8 . 9 1 } }$  </td><td> $\mathbf { 1 4 . 1 7 _ { \pm 1 5 . 4 6 } }$  </td><td> ${ \bf 1 . 7 6 _ { \pm 0 . 0 3 } }$  –</td></tr></table>

Table 2: Teacher tracking results. Contact recall at 15 cm, mean ± std over 3 seeds.
<table><tr><td rowspan="2">Method</td><td colspan="2">High-five</td><td colspan="2">Hug</td><td colspan="2">Link-arms</td></tr><tr><td>MPJPE↓ (cm)</td><td>Contact↑ (%)</td><td>MPJPE↓ (cm)</td><td>Contact↑ (%)</td><td>MPJPE↓ (cm)</td><td>Contact↑ (%)</td></tr><tr><td>Ours</td><td>6.5</td><td> ${ \bf 7 0 . 9 \pm 0 . 3 }$ </td><td>8.3</td><td> ${ \bf 5 3 . 1 \pm 0 . 5 }$ </td><td>7.2</td><td> ${ \bf 7 4 . 8 \pm 0 . 1 }$ </td></tr><tr><td>(-) mesh contact reward</td><td>6.5</td><td> $3 9 . 5 \pm 0 . 3$ </td><td>9.5</td><td> $3 1 . 4 \pm 0 . 1$ </td><td>8.6</td><td> $4 0 . 5 \pm 0 . 4$ </td></tr><tr><td>(-) VLM contact filtering</td><td>6.6</td><td> $6 7 . 7 \pm 0 . 2$ </td><td>8.1</td><td> $4 6 . 1 \pm 0 . 2$ </td><td>7.7</td><td> $7 3 . 4 \pm 0 . 4$ </td></tr></table>

Dataset and simulation setup We train on five interactive Inter-X [45] actions (high-five, hug, pat-on-back, handshake, link arms) with a manually filtered 9:1 train/evaluation split. All policies are trained in Isaac Sim with the human partner replayed via PD control; the teacher receives privileged partner kinematics, while the student observes only onboard egocentric depth. Full data statistics and the filtering procedure are in Appendix E.3, and the rendering setup is in Appendix C.1.

Metrics For retargeted motion we report Jitter and Penetration. Contact correctness in the VLM annotation and retargeting stages is measured with Recall, Precision, and $F _ { 1 }$ against 80 carefully hand-annotated GT contact clips; the teacher and student policies report Contact Recall against the VLM-verified contact annotations (manual annotation does not scale to these pools), the teacher on all motions of the student pool (training and held-out) at 15 cm and the student on the held-out split at 30 cm, and the teacher additionally reports MPJPE against the retargeted reference. Perceived quality uses a user study (mean rank, lower is better, N=28 participants). Metric definitions, thresholds, and GT sources are in Appendices E.4 and F, and the user-study protocol is in Appendix E.5.

## 4.1 Evaluating Semantic-Aware Contact Modelling and Retargeting

We first ask whether REACT’s retargeting stage transfers the interaction, not merely the motion, of a human–human demonstration onto the humanoid–partner pair. We compare against GMR [4], a kinematic whole-body retargeting baseline, and OmniRetarget [5], a contact-and-collision-aware retargeting method. Ours w/o VLM contact annotation ablates the per-frame contact labels and target joints (Sec. 3.2.1), falling back to the distance-based contact candidates from Stage 1 of our pipeline without the VLM verification step.

Table 1 reports the four geometry metrics together with the user study; pool definitions and contactevent GT are in Appendix E.4, and the user-study protocol is in Appendix E.5.

Contact-preserving retargeting (Q1) The baselines mark opposite failure modes of motion-only retargeting: GMR is smooth but contact-blind, while OmniRetarget is collision-aware at a large jitter cost and its hard collision constraints leave ≈ 30% of clips infeasible. Our Stage 1 ablation matches OmniRetarget on Penetration and Precision while reducing jitter to GMR’s level, inheriting its contact-aware geometry. Stage 2 co-optimisation then jumps Recall from 13.4% to 58.0% at unchanged jitter, and the VLM verifier further lifts Precision from 11.6% to 14.2% and Recall to 59.4%, matched by a 0.30-rank user-study improvement on overall preference.

## 4.2 Teacher Policy: Tracking the Interaction

We next evaluate the privileged-state teacher (Sec. 3.2), which must reproduce the retargeted interaction under physics. We ablate the two components that distinguish it from a standard tracking controller. The first is the mesh contact reward, which utilizes the ICP-based mesh-level contact positions and directions of humanoid and SMPL. In the (-) mesh contact reward baseline, it is replaced by a link-level reward that pulls the center of each G1 contact link (the palm for the hands) toward the annotated partner joint, without the mesh registration or the directional and force-band terms. In the Ours w/o VLM contactfiltering baseline, the contact positions and timings used for thi reward are instead calculated purely based on distance.

Table 3: Student policy results. Contact recall (%) at 30 cm on the held-out evaluation split, mean over 5 room layouts.
<table><tr><td>Method</td><td>Hug</td><td>High-five</td><td>Hand-shake</td><td>Link-arms</td><td>Pat-on-back</td></tr><tr><td>Ours</td><td>34.5</td><td>13.1</td><td>82.0</td><td>67.8</td><td>51.1</td></tr><tr><td>(-) future generation</td><td>43.2</td><td>14.0</td><td>75.2</td><td>42.5</td><td>44.2</td></tr><tr><td>generation → regression</td><td>31.3</td><td>33.9</td><td>75.2</td><td>44.5</td><td>44.4</td></tr><tr><td>(-) pose prediction</td><td>20.4</td><td>15.0</td><td>64.3</td><td>46.9</td><td>38.4</td></tr></table>

Contact supervision for policy learning (Q2) Table 2 demonstrates that both contact-related components contribute on the most contact-rich actions: the mesh contact reward lifts Link-arms Contact Recall by +34.3 pt and Hug by +21.7 pt, while VLM contact filtering lifts Hug by +7.0 pt, High-five by +3.2 pt, and Link-arms by +1.4 pt, with the two effects complementary across actions.

## 4.3 Depth-Based Student Policy

We evaluate the student (Sec. 3.2.2), which must recover the teacher’s interaction from egocentric depth and proprioception alone. We ablate the partner-prediction objective that shapes its depth representation, removing its components cumulatively: (-) future generation drops the generative prediction of future partner and humanoid motion; generation → regression further drops the generative prediction of past and present partner poses, keeping only the single-step pose-estimation loss $\mathcal { L } _ { \mathrm { p o s e } } ;$ and (-) pose prediction removes partner prediction entirely, training with $\mathcal { L } _ { \mathrm { B C } }$ and ${ \mathcal { L } } _ { \mathrm { K I } }$

Partner-state prediction under occlusion (Q3) Table 3 shows that behaviour cloning alone is a weak signal for the depth encoder: removing partner prediction lowers held-out Contact Recall on four of five interactions, by 14.1, 17.7, 20.9, and 12.7 pt on hug, hand-shake, linkarms, and pat-on-back. Most of the gain comes from generating the partner’s future, which adds 6.8 to 25.3 pt on hand-shake, link-arms, and pat-on-back, whereas generating rather than regressing the current and past poses helps only hug. High-five is the exception, where the deterministic variant (generation → regression) is best. We further systematically investigate where generation helps by relating contact recall to how much of the view the partner and the robot’s own body occupy

![](images/7d78bce9e6b38ae3df81da3d453891112b52eec33af00f4d273913f4f845e472.jpg)  
Figure 3: Within each interaction, frames are split into moments when the partner is hard to see (blue), either too close to fit in the view or largely out of view, and the remaining frames (gray). (Appendix C.4).

(Appendix C.4). Over the deterministic variant, the gain concentrates where the partner is least observable (Fig. 3): in close contact on hug and pat-on-back, and on link-arms, where the partner largely leaves the view, by 33.8 pt.

Teacher vs. student (Q4) Compared on the same motions and thresholds (Appendix C.5), the depth student recovers on average 75% of the privileged teacher’s contact recall at 30 cm, and up to 94% on hand-shake. By contrast, replacing depth with an external pose estimator (HMR2.0) collapses contact recall (Appendix C.2).

Figure 4 shows student rollouts on held-out motions never seen during training. Although the partner is frequently truncated or fills the camera view at contact range (insets), the student approaches the partner and establishes the demonstrated contact.

![](images/8ad2a2c89fca53e691a5b45f1f1a10be001f0acec765230ba8eaa95abfe6c602.jpg)  
Figure 4: Student rollouts on held-out motions in simulation (inset: egocentric depth input)  
Although we train a separate student per interaction, we treat these students as skills and let a VLM compose them into a single autonomous system: observing only the robot’s egocentric RGB and depth, the VLM searches for the partner, selects the skill that matches the partner’s configuration, and switches skills when the partner moves (Appendix B).

## 4.4 Real-World Deployment

We deploy the per-interaction students on a physical Unitree G1 with a human partner across all five interactions. The policies are trained entirely in simulation and run zero-shot, with no real-world fine-tuning; at test time each receives only egocentric depth and proprioception, with no external motion capture or privileged partner state. Because the onboard stereo depth loses the partner’s silhouette while the robot moves, we feed the policy metric depth estimated from the onboard RGB image by DepthAnything 3 [46] (Fig. 2, Real Deployment), which keeps the deployed observation close to the rendered training depth (Sec. 3.2.2).

Real-robot results (Q5) Across ten trials per interaction, the students succeed in 31 of 50 trials; per-interaction success rates, success criteria, and the model setting used for these trials are in Appendix A. Figure 1 shows the deployment on the real robot, and videos are on the project page. The deployed policies further generalise across partner distances and orientations and sustain repeated interaction, completing three consecutive hugs in one ∼30 s episode without reset (Appendix A.1).

## 5 Conclusion

We presented REACT, a scalable framework for learning deployable humanoid-human interaction policies from large-scale human-human motion-capture data. REACT addresses two key challenges in humanoid-human interaction: the difficulty of learning from noisy human-human demonstrations under heavy mutual occlusion, and the challenge of deploying interaction policies from onboard sensing under severe partial observability. Our framework combines VLM-guided contact-aware retar geting, privileged interaction learning with interaction-aware contact rewards, egocentric depth-policy distillation with photorealistic in-simulator human rendering, and auxiliary generative objectives that predict the partner’s pose so that the student can infer the partner’s state even under occlusion.

Through extensive simulation experiments and real-world deployment on a Unitree G1 humanoid, we showed that REACT improves preservation of demonstrated interaction contacts and enables robust interaction learning from onboard sensing during close-contact interaction. Our results suggest that scalable humanoid-human interaction may require jointly addressing semantic contact understanding, physically grounded interaction learning, and predictive inference under partial observability. More broadly, we believe REACT provides a step toward deployable humanoid systems capable of safe and adaptive physical interaction with humans in real-world environments. We discuss the limitations of REACT in Appendix G.

## Acknowledgments

This work was partially supported by JSPS KAKENHI Grant Number 24K22296 and 25H01159, and by JST BOOST, Japan Grant Number JPMJBS2409.

## References

[1] T. He, Z. Luo, X. He, W. Xiao, C. Zhang, W. Zhang, K. M. Kitani, C. Liu, and G. Shi. OmniH2O: Universal and dexterous human-to-humanoid whole-body teleoperation and learning. In 8th Annual Conference on Robot Learning, 2024.

[2] M. Ji, X. Peng, F. Liu, J. Li, G. Yang, X. Cheng, and X. Wang. ExBody2: Advanced expressive humanoid whole-body control. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 613–620. IEEE, 2026.

[3] Q. Ben, F. Jia, J. Zeng, et al. Homie: Humanoid loco-manipulation with isomorphic exoskeleton cockpit. arXiv preprint arXiv:2502.13013, 2025.

[4] J. P. Araujo, Y. Ze, P. Xu, J. Wu, and C. K. Liu. Retargeting matters: General motion retargeting for humanoid motion tracking. arXiv preprint arXiv:2510.02252, 2025.

[5] L. Yang, X. Huang, Z. Wu, A. Kanazawa, P. Abbeel, C. Sferrazza, C. K. Liu, R. Duan, and G. Shi. Omniretarget: Interaction-preserving data generation for humanoid whole-body locomanipulation and scene interaction. arXiv preprint arXiv:2509.26633, 2025.

[6] T. He, W. Xiao, T. Lin, Z. Luo, Z. Xu, Z. Jiang, J. Kautz, C. Liu, G. Shi, X. Wang, L. J. Fan, and Y. Zhu. HOVER: Versatile neural whole-body controller for humanoid robots. In IEEE International Conference on Robotics and Automation (ICRA), pages 9989–9996, 2025.

[7] T. He, J. Gao, W. Xiao, Y. Zhang, Z. Wang, J. Wang, Z. Luo, G. He, N. Sobanbabu, C. Pan, Z. Yi, G. Qu, K. Kitani, J. Hodgins, L. J. Fan, Y. Zhu, C. Liu, and G. Shi. ASAP: Aligning simulation and real-world physics for learning agile humanoid whole-body skills. 2025.

[8] Z. Chen, M. Ji, X. Cheng, X. Peng, X. B. Peng, and X. Wang. GMT: General motion tracking for humanoid whole-body control. arXiv:2506.14770, 2025.

[9] Z. Zhang, J. Guo, C. Chen, J. Wang, C. Lin, Y. Lian, H. Xue, Z. Wang, M. Liu, J. Lyu, H. Liu, H. Wang, and L. Yi. Track any motions under any disturbances. arXiv preprint arXiv:2509.13833, 2025.

[10] Z. Luo, Y. Yuan, T. Wang, C. Li, S. Chen, F. Castañeda, Z.-A. Cao, J. Li, D. Minor, Q. Ben, X. Da, R. Ding, C. Hogg, L. Song, E. Lim, E. Jeong, T. He, H. Xue, W. Xiao, Z. Wang, S. Yuen, J. Kautz, Y. Chang, U. Iqbal, L. Fan, and Y. Zhu. SONIC: Supersizing motion tracking for natural humanoid whole-body control. arXiv preprint arXiv:2511.07820, 2025.

[11] W. Zeng, S. Lu, K. Yin, X. Niu, M. Dai, J. Wang, and J. Pang. Behavior foundation model for humanoid robots. arXiv preprint arXiv:2509.13780, 2025.

[12] Y. Li, Z. Luo, T. Zhang, C. Dai, A. Kanervisto, A. Tirinzoni, H. Weng, K. Kitani, M. Guzek, A. Touati, A. Lazaric, M. Pirotta, and G. Shi. BFM-Zero: A promptable behavioral foundation model for humanoid control using unsupervised reinforcement learning. In International Conference on Learning Representations, 2026.

[13] S. Zhao, Y. Ze, Y. Wang, C. K. Liu, P. Abbeel, G. Shi, and R. Duan. Resmimic: From general motion tracking to humanoid whole-body loco-manipulation via residual learning. arXiv preprint arXiv:2510.05070, 2025.

[14] H. Weng, Y. Li, N. Sobanbabu, Z. Wang, Z. Luo, T. He, D. Ramanan, and G. Shi. HDMI: Learning interactive humanoid whole-body control from human videos. arXiv preprint arXiv:2509.16757, 2025.

[15] R.-Z. Qiu, S. Yang, X. Cheng, C. Chawla, J. Li, T. He, G. Yan, D. J. Yoon, R. Hoque, L. Paulsen, et al. Humanoid policy ∼ human policy. In Conference on Robot Learning (CoRL), 2025.

[16] G. Li, Q. Shi, Y. Hu, J. Hu, Z. Wang, X. Wang, and S. Luo. Thor: Towards human-level whole-body reactions for intense contact-rich environments. arXiv preprint arXiv:2510.26280, 2025.

[17] S. Xu, H. Y. Ling, Y.-X. Wang, and L.-Y. Gui. Intermimic: Towards universal whole-body control for physics-based human-object interactions. arXiv preprint arXiv:2502.20390, 2025.

[18] H. Wang, W. Zhang, R. Yu, T. Huang, J. Ren, F. Jia, Z. Wang, X. Niu, X. Chen, J. Chen, Q. Chen, J. Wang, and J. Pang. PhysHSI: Towards a real-world generalizable and natural humanoid-scene interaction system. arXiv preprint arXiv:2510.11072, 2025.

[19] S. Yin, Y. Ze, H.-X. Yu, C. K. Liu, and J. Wu. VisualMimic: Visual humanoid loco-manipulation via motion tracking and generation. arXiv preprint arXiv:2509.20322, 2025.

[20] T. He, Z. Wang, H. Xue, Q. Ben, Z. Luo, W. Xiao, Y. Yuan, X. Da, F. Castañeda, S. Sastry, C. Liu, G. Shi, L. Fan, and Y. Zhu. Viral: Visual sim-to-real at scale for humanoid loco-manipulation. arXiv preprint arXiv:2511.15200, 2025.

[21] H. Xue, T. He, Z. Wang, Q. Ben, W. Xiao, Z. Luo, X. Da, F. Castañeda, G. Shi, S. Sastry, L. J. Fan, and Y. Zhu. Opening the sim-to-real door for humanoid pixel-to-action policy transfer. arXiv preprint arXiv:2512.01061, 2025.

[22] D. Kalaria, S. Harithas, P. Katara, S. Kwak, S. Bhagat, S. S. Sastry, S. Sridhar, S. Vemprala, A. Kapoor, and J. Huang. DreamControl: Human-inspired whole-body humanoid control for scene interaction via guided diffusion. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 15939–15946. IEEE, 2026.

[23] A. Allshire, H. Choi, J. Zhang, D. McAllister, A. Zhang, C. M. Kim, T. Darrell, P. Abbeel, J. Malik, and A. Kanazawa. Visual imitation enables contextual humanoid control. In Proceedings of the Conference on Robot Learning (CoRL), 2025.

[24] M. G. Javed, C. Guo, L. Cheng, and X. Li. InterMask: 3d human interaction generation via collaborative masked modeling. In International Conference on Learning Representations, 2025.

[25] S. Ota, Q. Yu, K. Fujiwara, S. Ikehata, and I. Sato. PINO: Person-interaction noise optimization for long-duration and customizable motion generation of arbitrary-sized groups. pages 10676 – 10685, 2025.

[26] X. Cong, Z. Li, Z. Dou, H. Li, O. Taheri, C. Guo, A. Mittal, S. An, T. Komura, W. Matusik, M. J. Black, and S. Sridhar. Umo: Unified in-context learning unlocks motion foundation model priors. 2026.

[27] V. H. Maluleke, K. Horiuchi, L. Wilken, E. Ng, J. Malik, and A. Kanazawa. Diffusion forcing for multi-agent interaction sequence modeling. arXiv preprint arXiv:2512.17900, 2025.

[28] K. Ji, Y. Shi, Z. Jin, K. Chen, L. Xu, Y. Ma, J. Yu, and J. Wang. Towards immersive human-x interaction: A real-time framework for physically plausible motion synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10173 – 10183, 2025.

[29] Y. Shibata, K. Yamazaki, L. Jayanti, Y. Aoki, M. Isogawa, and K. Fragkiadaki. Learning to assist: Physics-grounded human-human control via multi-agent reinforcement learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 38345 – 38354, 2026.

[30] Y. Liu, C. Chen, C. Ding, and L. Yi. PhysReaction: Physically plausible real-time humanoid reaction synthesis via forward dynamics guided 4d imitation. In Proceedings of the ACM International Conference on Multimedia, page 3771–3780, 2024.

[31] S. Choi and J. Kim. Towards a natural motion generator: A pipeline to control a humanoid based on motion data. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4373–4380, 2019.

[32] W. Gomes, V. Radhakrishnan, L. Penco, V. Modugno, J.-B. Mouret, and S. Ivaldi. Humanoid whole-body movement optimization from retargeted human motions. In IEEE-RAS International Conference on Humanoid Robots, pages 178–185, 2019.

[33] S. Choi, M. J. Song, H. Ahn, and J. Kim. Self-supervised motion retargeting with safety guarantee. In IEEE International Conference on Robotics and Automation (ICRA), pages 8097–8103, 2021.

[34] T. Cheynel, T. Rossi, B. Bellot-Gurlet, D. Rohmer, and M.-P. Cani. Reconform: Real-time contact-aware motion retargeting for more diverse character morphologies. Computer Graphics Forum, 44(2):e70028, 2025.

[35] I. Jang, S. Choi, S. Hong, C. Kim, and J. Noh. Geometry-aware retargeting for two-skinned characters interaction. ACM Transactions on Graphics, 43(6):1–17, 2024.

[36] W.-J. Huang, Y.-Y. Zhang, Y.-L. Wei, Z.-W. Xia, J. Tan, Y.-M. Li, Z. Zhao, and W.-S. Zheng. Beyond mimicry: Learning whole-body human-humanoid interaction from human-human demonstrations. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 30740–30749, 2026.

[37] J. Chen, X. Li, J. Cao, Z. Zhu, W. Dong, M. Liu, Y. Wen, Y. Yu, L. Zhang, and W. Zhang. Rhino: Learning real-time humanoid-human-object interaction from human demonstrations. arXiv preprint arXiv:2502.13134, 2025.

[38] E. V. Mascaro, Y. Yan, and D. Lee. Robot interaction behavior generation based on social motion forecasting for human-robot interaction. In IEEE International Conference on Robotics and Automation (ICRA), pages 17264–17271, 2024.

[39] Q. Lu, Y. Feng, B. Shi, M. Piseno, Z. Bao, and C. K. Liu. Gentlehumanoid: Learning upper-body compliance for contact-rich human and object interaction. arXiv preprint arXiv:2511.04679, 2025.

[40] M. Mihajlovic, S. Zhang, G. Li, K. Zhao, L. Müller, and S. Tang. VolumetricSMPL: A neural volumetric body model for efficient interactions, contacts, and collisions. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[41] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[42] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

[43] S. Ross, G. Gordon, and D. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In International Conference on Artificial Intelligence and Statistics (AISTATS), volume 15 of Proceedings of Machine Learning Research, pages 627–635, 2011.

[44] Z. Luo, J. Cao, K. Kitani, W. Xu, et al. Perpetual humanoid control for real-time simulated avatars. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10895–10904, 2023.

[45] L. Xu, X. Lv, Y. Yan, X. Jin, S. Wu, C. Xu, Y. Liu, Y. Zhou, F. Rao, X. Sheng, et al. Interx: Towards versatile human-human interaction analysis. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22260–22271. IEEE, 2024.

[46] H. Lin, S. Chen, J. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

[47] B. Wen, S. Dewan, and S. Birchfield. Fast-FoundationStereo: Real-time zero-shot stereo matching. CVPR, 2026.

[48] S. Goel, G. Pavlakos, J. Rajasegaran, A. Kanazawa, and J. Malik. Humans in 4D: Reconstructing and tracking humans with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[49] J. Li, J. Cao, H. Zhang, D. Rempe, J. Kautz, U. Iqbal, and Y. Yuan. Genmo: A generalist model for human motion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11766–11776, 2025.

[50] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

## A Real-World Deployment

All real-robot results in this section were obtained with an earlier setting of the students, with a three-layer CNN depth encoder instead of ResNet-18.

## A.1 Qualitative results

Figure 1 of the main paper shows the real-world deployment of each interaction. Here we provide additional qualitative results.

The deployed policy is also robust to where the partner stands and how they are oriented. The pat-onback student establishes the back contact across partner distances (70 and 130 cm) and orientations (back-on and side-on to the robot), reaching and holding the contact in every condition (Figure 5).

Beyond a single contact event, the same policy sustains the interaction over a long horizon: it performs the hug three times in a row within one continuous rollout, with no reset, no re-initialisation, and no operator intervention between repetitions (Figure 6). Each cycle runs the full interaction loop, in which the robot approaches the partner, establishes and briefly holds the embrace, then releases, steps back to a stable stance, and re-approaches to initiate the next hug. Completing three such cycles consecutively within a ∼30 s episode demonstrates that the behavior is not a one-off contact event but rather a robust and repeatable closed-loop skill. The policy reliably recovers a stable post-contact state and re-engages from it, enabling repeated interaction cycles without degradation after the first successful execution.

![](images/7f85901bf55aa150e5e5fef2aee650cab845ee0d1264197dd62bf089a8dbb4a3.jpg)  
Figure 5: Generalization across partner distance and orientation. REACT performs pat-on-back with the partner at 70 cm and 130 cm and with the partner back-on or side-on to the robot; the per-interaction student reaches and holds the back contact in every condition from onboard depth alone.

## A.2 Quantitative results

We evaluate each per-interaction student over ten real-world trials (Table 4). The success criterion depends on the interaction: for hand-shake and high-five, where the partner actively raises and lowers a hand, a trial succeeds if the robot reacts to this motion and brings its hand within roughly 15 cm of the partner’s; for hug, pat-on-back, and link-arms, the partner stands still at varied positions, and a trial succeeds if the robot approaches the partner and completes the interaction. Deployed zero-shot, the students succeed in 31 of 50 trials overall, with hug and pat-on-back the most reliable at 8/10 each.

Table 4: Real-world success rate. Successful / total trials per interaction on the physical Unitree G1, ten trials per interaction.
<table><tr><td>High-five</td><td>Hug</td><td>Hand-shake</td><td>Link-arms</td><td>Pat-on-back</td><td>Total</td></tr><tr><td>5/10</td><td>8/10</td><td>5/10</td><td>5/10</td><td>8/10</td><td>31/50</td></tr></table>

![](images/cbad86a8a2e59eac83b56a0b04da31043750a2ada246f6d8207087f08cbf571c.jpg)  
Figure 6: Long-horizon behavior: three consecutive hugs in a single episode. A continuous ∼30 s rollout in which the policy completes the hug interaction three times in a row, with no reset or re-initialisation between repetitions. The ten snapshots are read in boustrophedon order, following the arrows. After each embrace the robot releases the partner, re-approaches, and re-establishes contact, so the three hug cycles appear as three separate close-contact phases across the panels. The policy sustains repeated contact-rich interaction over a long horizon rather than a single one-shot contact.

## A.3 Deployment-time depth processing

The student is trained on the in-simulator SMPL-X mesh depth described in Sec. C.1, which provides clean human geometry and sharp body silhouettes. On the physical robot, the onboard Intel RealSense D435 does not recover these silhouettes reliably: while the robot is moving, its stereo depth becomes noisy and loses the partner’s contour exactly where the policy needs it. We therefore do not use the RealSense depth stream at deployment. Instead we run monocular metric depth estimation with DepthAnything 3 (Large, metric) [46] on the onboard RGB image and feed the estimated depth to the policy. Inference runs in real time on the onboard compute through a TensorRT/DeepStream pipeline. This keeps the deployed observation close to the sharp, contour-rich depth the student was trained on, while requiring only a single RGB camera. The demonstration videos on the project page instead use Fast-FoundationStereo [47] on the RealSense left and right infrared images.

## A.4 Safe Interaction through Force Regulation

In close-contact interaction the magnitude of the force the robot applies to the person is a safety concern. We make this force observable and controllable during training by spawning the human partner as a physically simulated, collidable body that tracks the reference motion through PD control rather than a purely kinematic reference, so that every contact the policy makes produces a real reaction force on the corresponding G1 link. The force-regulation reward r<sup>force</sup><sub>t</sub> of Sec. F.1 (Eq. 7) then shapes how hard the robot touches. It rewards contact forces inside a comfortable band and penalises force at links that should not be in contact. Training the teacher with this term drives the policy toward gentle, bounded contact instead of forceful pushing, so the resulting interaction stays soft enough to be comfortable for the human while remaining firm enough to be felt. The deployable student inherits this behavior through distillation, imitating the actions of the force-regulated teacher.

## B Composing Per-Interaction Students with VLM Skill Selection

We train one student per interaction. This is a design choice rather than a limit on generalisation: adding a new interaction costs one additional policy instead of retraining all of them. To run the students as a single autonomous system, we treat them as a skill library and add a VLM-based selection layer on top, evaluated in simulation. The VLM agent observes only the robot’s own egocentric RGB and depth, never the partner’s state, and decides both where the robot should move and which student should take control.

VLM Brain Finite State Machine  
![](images/8777356e6e8a7005dec3fb085063198058e61a4d753bd0e50b0c77f8f375eef4.jpg)  
Figure 7: VLM skill-selection state machine. Solid boxes are states and dashed boxes are the policies they invoke: the HOMIE locomotion controller for Search, Approach, and Retreat, and one of the five per-interaction students for Interact.

![](images/c77e927d39c892d54b7db0c39d7cfd424d299b58fb37667c91787232c6f8d53e.jpg)  
Figure 8: One continuous rollout with VLM skill selection. No interaction is specified in advance. From left to right: the partner opens their arms, and the VLM selects the hug student; the partner is then displaced mid-rollout, and the robot re-finds them, approaches, and switches to the hand-shake student, all without a reset.

Finite state machine The selection layer is a four-state machine (Figure 7). In Search, the robot rotates in place through the HOMIE locomotion controller until a person becomes visible. In Approach, it walks toward the person and returns to Search if the person is lost. Once the robot is close and the VLM has confirmed which interaction the partner’s configuration calls for, the machine enters Interact and hands control to the matching student (hug, hand-shake, pat-on-back, link-arms, or high-five). The selected interaction is locked for a fixed period; when the lock expires, the robot re-approaches if the partner is still visible but far, or enters Retreat and walks backward if the partner is lost, after which it resumes Search or re-approaches as soon as the partner reappears.

Qualitative result Figure 8 shows one continuous rollout in which no interaction is specified. The robot first selects and executes a hug from the partner’s open-arm posture. After the partner is displaced mid-rollout, the robot re-finds the partner, approaches, and selects the hand-shake student, without any reset or operator intervention. A safe abort for a partner who pulls away mid-contact is left to future work.

![](images/7e8af76fb2e7d0a002ed06928d0216e87be25803478f8a62c1ec823fde5ec493.jpg)  
Figure 9: In-loop human depth rendering closes the sim-to-real gap. Egocentric depth of a hug frame. Capsule: the partner as collision primitives, blocky and non-human. SMPL-X Mesh: the partner rendered as a realistic body mesh. SMPL-X Mesh + Scene (Ours): our full rendering with the randomised scene. This last variant closely resembles the real-sensor depth shown in Figure 1 of the main paper.

## C Student Policy

## C.1 Realistic Human Depth Rendering

A depth policy transfers to the real robot only if its training depth resembles what the onboard sensor will see, so closing the sim-to-real depth gap is essential. Physics simulators represent the human partner with capsule colliders, whose egocentric depth (Fig. 9, left) looks nothing like a real person. We hide these capsules from the depth camera and, for rendering only, attach a visual SMPL-X body mesh that we update in the loop, so that at every control step a single batched GPU SMPL-X forward pass advances every environment’s mesh to its current frame and the camera sees a realistic moving human (Fig. 9, middle). The capsule colliders stay in the physics scene so contact is still simulated, while the mesh is used only for rendering. To the best of our knowledge, this is the first system to model physical interaction with a human in a physics simulator that simultaneously renders the human as a photorealistic SMPL-X mesh for perception and simulates contact through capsule colliders. The surrounding scene is further randomised with mesh assets refreshed during training, so the depth also carries realistic background geometry (Fig. 9, right). This in-loop human-and-scene depth is what lets the student deploy zero-shot on the real robot.

## C.2 Why End-to-End Depth Beats Pose-Estimation-Based Methods

A natural way to drive interaction with a human partner is to attach an external human-pose-estimation module, recover the partner’s body from the camera stream, and feed the estimated pose to the policy as the partner-state input [36, 48]. This design factors perception from control and reuses mature off-the-shelf pose estimators, but it breaks down in the close-range regime that physical interaction requires. At contact range, the partner is largely occluded by the robot’s own body or leaves the camera field of view, so an external estimator either loses the partner or returns a stale pose precisely when accurate partner state matters most.

REACT instead relies on no external pose-estimation module: it maps the egocentric depth image to action end-to-end, and uses partner pose estimation only as an auxiliary task that improves training efficiency.

Table 5 shows the quantitative comparison between these two approaches. We build the externalpose baseline following 4D Humans [48]. At each step, a person detector localizes the partner and HMR2.0 reconstructs the partner’s body. The recovered partner joint positions are concatenated with the robot’s proprioception and fed to the policy, replacing the egocentric depth input that REACT uses. This substitution collapses mesh-contact recall on every interaction, falling from 78.6 to 8.2 on hand-shake and from 15.3 to 5.4 on high-five, because the external estimator degrades under exactly the self-occlusion and field-of-view loss that close, contact-rich interaction induces. The end-to-end student, by contrast, stays reliable in this regime, where an external estimator is least able to recover the partner state.

Table 5: End-to-end vs. external human-pose estimation. Mesh-contact recall at a 30 cm threshold, in percent, on the held-out evaluation split. Ext. human pose (HMR2.0) drives the policy from partner state produced by an external human-pose estimator while Ours (E2E) outputs actions end-to-end from egocentric depth and proprioception. Both rows use earlier model settings than Table 3 and are evaluated on the same held-out motions.
<table><tr><td>Method</td><td>Hug</td><td>Hand-shake</td><td>Pat-on-back</td><td>Link-arms</td><td>High-five</td></tr><tr><td>Ext. human pose (HMR2.0 [48])</td><td>5.7</td><td>8.2</td><td>1.3</td><td>3.9</td><td>5.4</td></tr><tr><td>Ours (E2E)</td><td>38.0</td><td>78.6</td><td>23.8</td><td>30.1</td><td>15.3</td></tr></table>

![](images/52227ef8a3d2a4af4f66e2b23711f568ee3215ec61faceef61843082969266e1.jpg)  
Figure 10: Why the external pose estimation-based method fails at contact range. Temporally ordered snapshots of an external-pose rollout, read left to right, with the ground-truth partner (green, Human (Sim)) and the HMR2.0 reconstruction (red, HMR2.0 Prediction) overlaid against the robot (black). Left three: high-five; as the partner’s hand leaves the robot’s egocentric field of view, the reconstructed hand position deviates sharply from the ground truth at the moment of contact. Right three: hug; at the instant of embrace the partner’s torso occludes the camera, and the reconstruction collapses into a large error behind the true mesh. The estimator degrades precisely in the close-contact, self-occluded regime that REACT’s end-to-end student handles directly from depth.

To better understand why the external-pose baseline fails, Figure 10 overlays the ground-truth partner mesh (green) and the HMR2.0 reconstruction (red) in the same scene as the robot, for a high-five and a hug episode. The two interactions expose complementary failure modes of external pose estimation at contact range. In the high-five case (left), the partner raises a hand toward the robot, and as that hand crosses the edge of the robot’s egocentric field of view, the reconstructed arm jumps to a completely wrong configuration. Such an implausible pose drives the policy’s partner-state input out of the distribution it was trained on, so the policy reaches to the wrong location. In the hug case (right), the partner remains in view during the approach, but at the instant of embrace, the partner’s torso fills and occludes the camera. With almost no visible body surface to fit, HMR2.0 produces a large, unstable error and the reconstruction breaks apart behind the true mesh. In both cases the estimator degrades exactly when the interaction enters close contact, which is when accurate partner state matters most, and this is the mechanism behind the recall collapse in Table 5.

## C.3 Depth-based Student Policy: architecture, losses, and distillation

Observations At every control step the student receives (i) a 5-frame egocentric depth stack at offsets {0, 0.5, 1.0, 1.5, 2.0} s into the past, (ii) a 5-step short-term proprioception history $( 5 \times 1 0 8 =$ 540 dims), and (iii) a 100-step (2 s at 50 Hz) long-term proprioception history.

Depth backbone A shared ResNet [42] embeds each of the five depth frames independently. The per-frame embeddings are concatenated into a temporal depth representation.

Long-term proprio encoder A 1D-CNN maps the 100-step proprioception window to a compact embedding.

Conditional VAE Let $\mathbf { c } _ { t }$ concatenate the temporal depth embedding, the short-term proprioception history, and the long-term proprioception embedding. A prior MLP maps $\mathbf { c } _ { t }$ to a diagonal Gaussian $\mathcal { N } ( \mu _ { p } , \sigma _ { p } ^ { 2 } )$ ; a training-only encoder MLP maps the current reference joint positions and velocities ${ \bf o } _ { t } ^ { \mathrm { r e f } } \mathrm { t o } \mathcal { N } ( \mu _ { e } , \sigma _ { e } ^ { 2 } )$ . We combine the two heads residually, $\pmb { \mu } _ { z } = \pmb { \mu } _ { p } + \pmb { \mu } _ { e }$ , and a decoder MLP maps $[ \mathbf { c } _ { t } , \mathbf { z } _ { t } ]$ to the same 18-dim HOMIE-plus-arm action mean as the teacher. At deployment the encoder is dropped and $\mathbf { z } _ { t }$ is drawn from the prior $\mathcal { N } ( \mu _ { p } , \sigma _ { p } ^ { 2 } )$

Pose-anticipation auxiliary head A DDPM denoiser conditioned on the temporal depth embedding and the long-term proprioception embedding predicts the partner’s past, current, and short-horizon future SMPL-X key-body state (positions, velocities, and orientations), $P _ { t } ^ { \mathrm { p a r t n e t } }$ , together with the G1’s own near-future trajectory, $P _ { t } ^ { \mathrm { h u m a n o i d } }$ , over 5 past/current frames at $\{ 0 , 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 \}$ s and 3 future frames at $\{ 0 . 5 , 1 . 0 , 1 . 5 \}$ s ahead. The objective sums a multi-step DDIM forward-prediction loss and a GENMO-style [49] single-step pose-estimation loss in equal proportion, and the combined pose loss enters the total distillation loss with weight $0 . 5 ,$ i.e. $\lambda _ { \mathrm { g e n } } = \lambda _ { \mathrm { p o s e } } = 0 . 5$ . Per-frame losses are gated by how much of the human partner is visible in the depth image. Frames in which the partner occupies 15% or less of the depth pixels contribute no gradient.

Distillation loss The full distillation loss defined in the main paper combines an L2 behaviorcloning term against the teacher action, the conditional-VAE KL between the privileged encoder posterior and the deployable prior (with $\lambda _ { \mathrm { K L } }$ ramped from $1 0 ^ { - 4 }$ to $1 0 ^ { - 2 }$ between epochs 500 and 2000), and the two pose objectives above. We use on-policy DAgger [43].

Other student training details The egocentric camera is a simulated Intel RealSense D435 mounted on the G1’s upper body and held level, so it faces forward rather than down at the floor and keeps the partner inside the frustum during close interaction. The stock G1 mount is pitched about $4 8 ^ { \circ }$ downward for locomotion, so on the real robot we use a custom level mount that the simulated camera matches. It runs at 30 Hz with an $8 7 ^ { \circ }$ horizontal field of view and a 3 m far clip, renders at $1 9 2 \times 1 0 8$ without down-sampling, and these intrinsics match the physical D435 we deploy. Because such an $8 7 ^ { \circ } { \cdot } \mathrm { H F O V }$ camera cannot see a partner who is, in the demonstration, beside or behind the humanoid, we restrict the training and evaluation corpus to clips in which the partner stays within the frustum. We train on 128 parallel Isaac Sim environments on a single L40S GPU with the partner driven by a reference PD controller and rendered as in Sec. C.1. Demonstration-time augmentation includes human motion speed warping [0.8, 1.2]×, a random start-time offset up to 1 s, a random G1 spatial offset of 0.15 m, a random yaw offset up to 0.32 rad (≈ 18.5<sup>◦</sup>), and segment slowdown up to $2 \times$ over windows of up to 3 s. Depth-image augmentation applies edge displacement, distance-dependent additive Gaussian and sparkle noise, background-occlusion rectangles, and random rectangular and corner masks, and the deepest pixels are clamped to the 3 m far clip. We distil a separate student per interaction, so each of the five actions has its own deployable checkpoint.

## C.4 Gain of generative prediction under partial observability

Fig. 3 of the main paper compares the full student (Ours) with the deterministic variant (generation → regression in Table 3) as a function of how much of the egocentric view is occupied. Both are evaluated on the held-out split with the indoor scene randomisation (walls, ceiling, and furniture) disabled, so that every depth pixel above the floor and nearer than the 3 m far clip belongs to the partner or to the robot’s own body. For every contact-labelled frame we record this occupancy, the fraction of such pixels in the depth image, and whether the contact is satisfied at 30 cm.

The partially observable regime differs by interaction. On hug and pat-on-back it is close contact, where the two bodies fill the view, and we split at an occupancy of 0.8. On link-arms it is the partner leaving the view, and we split at 0.4, since fewer than $2 \%$ of its contact frames exceed 0.8. Both thresholds were set after inspecting the occupancy distributions. Table 6 reports the recall difference in each regime over three evaluation seeds, which change the simulation seed while keeping the same checkpoints.

In every seed and on all three interactions, the gain is larger in the partially observable regime. The seed-paired difference between the two regimes is significant on link-arms $( t ( 2 ) = 5 . 6 , p < 0 . 0 5 )$ and consistent in sign but not significant on hug $( t ( 2 ) = 3 . 1 )$ and pat-on-back $( t ( 2 ) = 2 . 4 )$ . Three caveats apply. The seeds vary only the evaluation, with one trained checkpoint per variant. Occupancy counts the robot’s own body, which also occludes the partner. And occupancy results from each policy’s own motion, so the analysis is correlational rather than an intervention on visibility.

Table 6: Contact recall gain of Ours over the deterministic variant (pt at 30 cm) by occupancy. Scene randomisation disabled, held-out split. PO: partially observable regime. Share: fraction of contact frames in the regime.
<table><tr><td>Interaction</td><td>Regime</td><td>Share</td><td>Seeds 1 / 2 / 3</td><td>Mean</td></tr><tr><td rowspan="2">Hug</td><td rowspan="2">PO (occupancy ≥ 0.8) otherwise</td><td>26%</td><td>+7.4/+5.7/+7.7</td><td>+6.9</td></tr><tr><td>74%</td><td>+2.3/+4.2/+1.6</td><td>+2.7</td></tr><tr><td rowspan="2">Pat-on-back</td><td rowspan="2">PO (occupancy ≥ 0.8) otherwise</td><td>22%</td><td>+12.4/+12.1/+6.2</td><td>+10.2</td></tr><tr><td>78%</td><td>+0.1 /-1.6 /+4.6</td><td>+1.0</td></tr><tr><td rowspan="2">Link-arms</td><td rowspan="2">PO (occupancy &lt; 0.4) otherwise</td><td>70%</td><td>+35.9 /+34.1/+31.4</td><td>+33.8</td></tr><tr><td>30%</td><td>+10.9 /-6.1/+7.6</td><td>+4.1</td></tr></table>

## C.5 Teacher vs. student on matched motions and thresholds

The main paper scores the teacher at 15 cm on all motions and the student at 30 cm on held-out motions. To compare them directly, Table 7 evaluates the teacher (the distillation source) and the deployed student on the same motions, the student’s training split, since the teacher is trained on all motions and has no held-out split. Both use the same evaluation protocol and per-frame mesh contact recall. The teacher values differ slightly from Table 2 because they are computed on the student’s training motions and from a single seed. Averaged over the five interactions, the student recovers 52% and 75% of the teacher’s recall at 15 and 30 cm, respectively.

Table 7: Teacher vs. student contact recall (%) on matched motions. T: privileged teacher, S: egocentric depth student, Rec.: S/T. The mean recovery averages the per-interaction ratios.
<table><tr><td rowspan="2">Interaction</td><td colspan="3">15 cm</td><td colspan="3">30 cm</td></tr><tr><td>T</td><td>S</td><td>Rec.</td><td>T</td><td>S</td><td>Rec.</td></tr><tr><td>Hug</td><td>52.3</td><td>18.1</td><td>35%</td><td>80.2</td><td>49.9</td><td>62%</td></tr><tr><td>High-five</td><td>69.5</td><td>23.9</td><td>34%</td><td>87.7</td><td>62.1</td><td>71%</td></tr><tr><td>Hand-shake</td><td>93.9</td><td>78.6</td><td>84%</td><td>96.7</td><td>91.1</td><td>94%</td></tr><tr><td>Link-arms</td><td>72.5</td><td>44.1</td><td>61%</td><td>89.9</td><td>74.1</td><td>82%</td></tr><tr><td>Pat-on-back</td><td>87.0</td><td>38.4</td><td>44%</td><td>93.9</td><td>61.4</td><td>65%</td></tr><tr><td>Mean</td><td>75.0</td><td>40.6</td><td>52%</td><td>89.7</td><td>67.7</td><td>75%</td></tr></table>

## D Contact Label Filtering and Surface Projection

## D.1 VLM Contact Annotation: distance candidates and semantic verification

We annotate five contact-rich Inter-X [45] actions: hug, hand-shake, pat-on-back, high-five, and link-arms by following these steps shown below.

Stage 1: distance candidates We run SMPL-X forward kinematics on both bodies and, for every other frame, group each person’s arm mesh vertices by skinning weights into a hand (wrist and fingers), forearm, and upper-arm region on each side. Each region’s vertices are compared against every vertex of the partner mesh; pairs within 15 cm are grouped by (contacting arm region, contacted SMPL-X joint), the closest pair is retained per group, and detections separated by gaps of up to 0.5 s are merged into events. Events shorter than 0.1 s are discarded. Each event records its time window, the contacting region, the contacted partner joint, the closest actor- and partner-side vertices, and the minimum and average mesh-to-mesh distances.

Stage 2: VLM verification Each event is verified against the corresponding video by a visionlanguage model (Gemini 2.0 Flash). We render the SMPL-X motion from three camera viewpoints with person-1 in blue and person-2 in orange, concatenate the views horizontally with per-view foreground cropping, sample at 5 fps, and split the video into 3 s chunks; each chunk is sent with the action label and a textual description of every candidate contact event that overlaps it. The VLM assigns each contact candidate one of four semantic labels (essential, incidental, unnecessary, or avoid), together with the contacting and contacted surfaces (palm/inner vs. back/outer) and the views in which the contact is clearly visible. Only events labelled essential or incidental survive; Figure 11 shows representative kept and removed candidates across the five actions. The verbatim prompt is given in Sec. H.

Table 8: VLM contact-label quality on 80 manually-annotated Inter-X clips (16 per action). Peraction and total precision, recall, and $F _ { 1 }$ as across-clip mean ± std under per-frame vertex-distance matching at $\epsilon = 1 0$ cm on the SMPL-X rest-pose template, after symmetric-pair deduplication. Ours w/o VLMfiltering uses the raw Stage 1 distance candidates; Ours additionally applies VLM verification. Reporting precision and recall separately shows that the VLM gain is almost entirely a precision gain, while recall is essentially preserved (or only mildly reduced). The verifier removes spurious near-contacts rather than acting as a stricter detector.
<table><tr><td rowspan="2">Action</td><td colspan="3">Ours w/o VLM filtering (distance)</td><td colspan="3">Ours (VLM-verified)</td></tr><tr><td> $\mathrm { P \uparrow }$ </td><td> ${ \textrm { R } } \uparrow$ </td><td> $F _ { 1 } \uparrow$ </td><td>P↑</td><td>R↑</td><td> $F _ { 1 } \uparrow$ </td></tr><tr><td>Hug</td><td> $0 . 0 6 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 2 4 1 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $0 . 1 2 7 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 0 8 1 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $0 . 2 0 1 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $\mathbf { 0 . 1 8 1 } _ { \pm 0 . 1 3 }$ </td></tr><tr><td>Hand-shake</td><td> $0 . 2 7 5 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $0 . 7 8 5 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $0 . 4 9 5 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $0 . 4 4 2 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $0 . 6 6 1 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $\mathbf { 0 . 6 7 4 { \scriptstyle \pm 0 . 2 1 } }$ </td></tr><tr><td>Pat-on-back</td><td> $0 . 1 0 7 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 8 2 4 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $0 . 2 1 2 _ { \pm 0 . 0 9 }$ </td><td> $0 . 2 2 7 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $0 . 7 6 0 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $\mathbf { 0 . 3 8 7 _ { \pm 0 . 1 2 } }$ </td></tr><tr><td>High-five</td><td> $0 . 2 6 0 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $0 . 8 6 6 _ { \pm 0 . 3 2 }$ </td><td> $0 . 4 1 7 _ { \pm 0 . 1 6 }$ </td><td> $0 . 3 3 6 _ { \pm 0 . 2 7 }$ </td><td> $0 . 6 6 0 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $\mathbf { 0 . 6 2 6 _ { \pm 0 . 1 7 } }$ </td></tr><tr><td>Link-arms</td><td> $0 . 2 5 7 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $0 . 6 0 0 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $0 . 3 7 4 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $0 . 3 3 9 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $0 . 5 4 9 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $\mathbf { 0 . 4 5 3 { \scriptstyle \pm 0 . 1 8 } }$ </td></tr><tr><td>Total (80 clips)</td><td> $0 . 1 9 2 _ { \pm 0 . 1 5 }$ </td><td> $\mathbf { 0 . 6 6 3 { \scriptstyle \pm 0 . 3 9 } }$ </td><td> $0 . 3 3 0 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $\mathbf { 0 . 2 8 5 { \scriptstyle \pm 0 . 2 5 } }$ </td><td> $0 . 5 6 7 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $\mathbf { 0 . 4 6 8 _ { \pm 0 . 2 3 } }$ </td></tr></table>

Output For each of the six contact sites (left or right hand, forearm, upper-arm) and every timestep t, the annotation provides a binary contact flag, the closest vertex on the actor mesh, the contacted target vertex on the partner mesh, and the name of the contacted SMPL-X joint. The contact reward of Eq. 5 and the Stage 2 contact co-optimisation in retargeting consume these flags for all six sites, mapped to the G1 wrist, elbow, and shoulder links.

Label-quality evaluation against manual ground truth To isolate the contribution of Stage 2, we evaluate the contact labels themselves, independently of any retargeting, against a hand-annotated ground-truth set. We manually annotate 80 Inter-X clips (16 per action, five actions: hug, hand-shake, pat-on-back, high-five, link-arms), marking the actor SMPL-X vertex, the contacted partner SMPL-X vertex, and the time window of each true contact event. We then treat the Stage 1 distance candidates and the Stage 2 VLM-verified events as predicted events and match them to GT events on a per-frame basis. At every frame, an actor–partner contact pair is counted as a true positive iff (i) the actor vertices of the GT and the prediction lie within ϵ = 10 cm on the SMPL-X rest-pose template (matching the retargeting threshold used in the main paper), and (ii) the same holds on the partner side. To avoid double-counting the same physical contact when the predictor independently emits its mirrored (partner-actor) candidate, we deduplicate predictions whose vertex pair coincides with that of an existing same-frame actor–partner candidate. We report per-clip precision, recall, and $F _ { 1 }$ as the across-clip mean ± standard deviation, both per action and on the full 80-clip set.

Splitting $F _ { 1 }$ into precision and recall (Table 8) makes the role of the VLM stage explicit and shows that the actions divide into two distinct regimes. On the total set the VLM stage raises across-clip mean precision by 48% (0.192→0.285) while recall only drops mildly (0.663→0.567), so the +42% $F _ { 1 }$ gain (0.330→0.468) is precision-driven. The verifier mainly removes spurious candidates from the Stage 1 set rather than acting as a stricter detector.

The first regime covers pat-on-back, high-five, hand-shake, and link-arms, where the distance candidates already recall most true contacts, with recall ranging from 0.60 to 0.87, but at very low precision of only 0.11 to 0.28, because close-range arm motion fires many spurious near-contacts, which is where the low distance-only $F _ { 1 }$ comes from. VLM verification raises precision by 1.3–2.1× (e.g. pat-on-back 0.107→0.227, hand-shake 0.275→0.442) while recall drops by 0.05–0.21, lifting $F _ { 1 }$ the most. Hug is the opposite case. Its recall is already low for both methods, 0.24 and 0.20, so the low $F _ { 1 }$ is recall-bound rather than precision-bound. The cause is structural, since 33% of hug GT events have the actor on the torso/pelvis, outside the six hand/forearm/upper-arm sampling regions that both stages enumerate, so the predictor cannot emit those candidates at all. VLM verification cannot recover contacts that were never proposed, so its hug gain is small and comes from cleaning the few arm-region candidates rather than from added coverage.

![](images/53625a12d350fd43e1b19037234f46cdaf1a9c76713af23a1daded931b9eb660.jpg)  
Figure 11: VLM contact filtering, qualitative examples. Distance candidates and their VLM verdicts on the five contact-rich Inter-X actions (rows: hug, hand-shake, pat-on-back, high-five, link-arms), sampled at 1 s intervals over a 4 s window (columns). Green markers are candidates kept after VLM verification, labelled essential or incidental. Red markers are Stage 1 distance candidates the VLM removes as spurious near-contacts. The proximity test fires on many transient arm passes; the VLM stage prunes them while preserving the genuine contacts, which is the precision gain quantified in Tab. 8.

Annotation tool and ground truth The 80 GT clips are annotated with a custom GUI that records, for each true contact event, the actor SMPL-X vertex, the contacted partner SMPL-X vertex, and the start/end frames. GT events are authored in an actor-centric convention and remapped to the Inter-X canonical P1/P2 frame before scoring (the same remap used in Sec. E.4); without this remap roughly half of the clips would compare the wrong person. Each clip is annotated by a single expert.

![](images/513d6b332100b322bed8e483569185ca4d227b2c098e80fe4ae5ed21d50ae808.jpg)  
Figure 12: Body correspondence projects the SMPL-X mesh onto either the G1 or the partnercapsule surface. The offline registration introduced in Sec. D.2 establishes a dense body correspondence that maps every SMPL-X body vertex to a (link, $\mathbf { p } _ { \mathrm { l o c a l } } )$ pair, so the same mesh can be projected onto either target surface, the Unitree G1 robot or the partner’s SMPL-X collision capsules. The three columns show one identical set of colour-coded SMPL-X vertices in three configurations, from left to right: projected onto the G1 robot surface, in their SMPL-X template positions, and projected onto the partner capsule surface. Each colour denotes the G1 link a vertex is registered to, and the connecting lines join the same representative vertices across the three configurations, so that any contact vertex can be carried onto the corresponding robot or capsule surface by forward kinematics. The columns are drawn at true relative scale, exposing the proportion mismatch between the human template and the shorter G1 that the registration must bridge. The dense map covers the full template at a mean vertex-to-surface residual of 0.89 mm, well below the ϵ = 10 cm contact threshold.

## D.2 Human-to-Robot Surface Correspondence Registration

We construct an offline map from every SMPL-X body vertex to a (link, $\mathbf { p } _ { \mathrm { l o c a l } } )$ pair on the Unitree G1 zero-pose surface. At run time, the world-space location of any SMPL-X contact vertex used in Stage 2 retargeting (Sec. E.2) and in the contact reward is recovered by forward kinematics on G1 from the stored link-local point. The registration runs in four stages on the SMPL-X mean-shape body and the G1 zero-pose surface mesh.

Stage 1: limb-direction initialisation A scale-invariant initial alignment between the SMPL-X body and the G1 zero pose is obtained by matching cosines between corresponding limb directions on the two bodies. Using direction rather than length yields a robust initialisation despite the substantial proportion mismatch between the human body and the humanoid robot.

Stage 2: weighted Procrustes alignment A weighted Procrustes solve over corresponding skeletal joints further aligns the two articulations while preserving their global geometric consistency, providing a clean reference frame for the subsequent surface-level deformation.

Stage 3: Laplacian-regularised non-rigid ICP We deform the SMPL-X mesh onto the G1 surface with a Laplacian-regularised non-rigid ICP, establishing dense surface-level correspondences. Two safeguards prevent erroneous matches across unrelated body parts despite the surface-proximity criterion. A decreasing-stiffness schedule establishes coarse rigid-body-part structure before allowing fine-grained deformations, and ray-cast normal projection restricts each correspondence query to surfaces visible along the outward SMPL-X vertex normal, blocking matches that would otherwise wrap around the robot from the opposite side.

Stage 4: post-snap and storage A final post-snap stage projects the remaining outlier vertices, those whose Stage 3 target falls in free space or on a kinematically unrelated link, onto the nearest valid robot surface point, ensuring complete surface coverage. For each SMPL-X vertex we store the resulting G1 link identifier together with the link-local position $\mathbf { p } _ { \mathrm { l o c a l } }$

Runtime use At every control step, the contact reward of Eq. 5 and retargeting (Sec. E.2) retrieve the G1-side contact point as $\mathbf { p } _ { t } ^ { h } = T _ { \mathrm { l i n k } ^ { h } } ( \pmb { q } _ { t } ) \mathbf { p } _ { \mathrm { l o c a l } ^ { h } }$ , where $T _ { \mathrm { l i n k } ^ { h } } ( \boldsymbol { q } _ { t } )$ is the forward-kinematics transform of the stored link at joint configuration $\pmb { q } _ { t }$ and $( \operatorname { l i n k } ^ { h } , \mathbf { p } _ { \mathrm { l o c a l } ^ { h } } )$ is the registered pair for contact site h.

Registration quality Across the full SMPL-X template the recovered correspondence attains a mean vertex-to-surface residual of 0.89 mm on the G1 zero-pose mesh, far below the ϵ = 10 cm contact threshold used throughout evaluation (Sec. E.4); registration error is therefore negligible relative to the metric tolerance. Figure 12 visualises the colour-coded link assignment recovered by this procedure.

## E Contact-Aware Retargeting

This section covers the two-stage contact-aware retargeting that closes the demonstrated contacts on the humanoid, given the surface registration and VLM contact labels described in Sec. D.2 and Sec. D.1. We first show a qualitative comparison of the retargeted motions in Sec. E.1, then detail the two-stage retargeting optimisation itself in Sec. E.2. We describe the dataset and its filtering in Sec. E.3, and report event-level contact recall and precision against the manual annotations in Sec. E.4 together with a human preference study in Sec. E.5.

## E.1 Qualitative results

We begin the evaluation with a qualitative comparison of the retargeted motions (Figure 13). For each of the five contact-rich actions we show the original human–human demonstration alongside the retargeting baselines and our method at the action’s peak-contact frame, with an arrow marking the demonstrated contact the retarget must reproduce on the G1. Ours establishes the demonstrated touch on the robot surface, whereas the baselines either fail to reach the contact or push the bodies apart to satisfy their solvers, and Ours w/o VLMfiltering additionally chases spurious distance candidates.

## E.2 Contact-Aware Motion Retargeting

We retarget each human–human demonstration onto the humanoid in two stages. Stage 1 (kinematic retargeting) height-scales only the humanoid-side body along gravity and, on contact-labelled frames, blends the arm end-effector targets toward the demonstrated contact positions, so the touches are anchored rather than averaged away. It builds on OmniRetarget [5]. Stage 2 (contact co-optimisation), detailed next, closes the residual contact gap.

Co-optimisation Stage 2 refines the Stage-1 kinematic initialisation by alternating inverse kinematics between the humanoid and the partner, rather than by descending a single weighted-sum loss. We run 3 outer co-optimisation iterations. Each iteration first solves a humanoid IK step and then a partner SMPL-X IK step, holding the Stage-1 locomotion (root trajectory and leg motion) fixed throughout.

Targets Each annotated interaction is converted into per-frame IK targets of two kinds. The first kind is a binary contact, a touch that the VLM filter has verified. For a binary contact the humanoid contact site h is driven toward a meeting point $\mathbf { m } _ { t } ^ { h } = \left( 1 - w \right) c g _ { h } ( \mathbf { q } _ { t } ) + w r c _ { j }$ , where $c g _ { h }$ is the

![](images/f064f44ab728c31cab1db3da49fe915667ff94a4494a8b278cd73a5fc225172e.jpg)  
Figure 13: Qualitative retargeting comparison. Rows: the five contact-rich actions (hug, handshake, pat-on-back, high-five, link-arms). Columns: the original human–human demonstration and four retargets (GMR, OmniRetarget, Ours w/o VLM filtering, Ours), each at the action’s peak-contact frame. The red arrow marks the demonstrated contact the retarget must reproduce on the G1. Ours establishes the touch on the robot surface where the baselines reach short or push apart.

G1 link-surface point, $r c _ { j }$ the targeted partner SMPL-X vertex, and w a slack weight, with $\mathbf { m } _ { t } ^ { h }$ projected onto the partner surface mesh. The second kind is a proximity pair, a body part that stays near the partner without touching and must keep its demonstrated separation. We flag a proximity pair whenever a partner arm or hand vertex approaches a non-arm G1 link within 0.08 m, and we drive the humanoid site toward the distance-preserving point $r c _ { j } + \hat { \mathbf { d } } d ^ { \mathrm { { r e f } } }$ , where d<sup>ˆ</sup> is the current partner-to-G1 direction and $d ^ { \mathrm { r e f } }$ the demonstrated distance. This distance-preserving proximity field keeps the interaction configuration from collapsing. Binary targets enter the humanoid IK with weight 1.0, scaled by a 2-frame approach ramp, and proximity targets with weight 0.3.

Solvers The humanoid IK is a damped least-squares step over the arm DOFs (10 inner iterations, orientation weight 0.3, $N _ { \mathrm { s u b } } = 5$ sub-steps). The partner SMPL-X IK is a gradient step (learning rate 0.003, 100 inner iterations, pose-regularisation weight 0.003 toward the Stage-1 pose, temporal smoothness weight 0.02). Interpenetration is suppressed by a volumetric SMPL signed-distance field [40] evaluated against the G1 collision mesh (collision weight 10.0, margin 0.005 m), making the IK collision-aware. At most 8 contact pairs per frame are optimised, and clips are truncated to 10 s.

Contact-adaptive temporal smoothing We smooth the optimised arm body-pose and arm joint DOFs with a Gaussian temporal filter whose width adapts to how much contact the clip contains. The effective width is $\sigma _ { \mathrm { e f f } } = \sigma \operatorname* { m i n } ( 1 , n _ { c } / n _ { 0 } )$ with base width $\sigma = 2$ frames, $n _ { c }$ the number of contact frames, and $n _ { 0 }$ a contact-count threshold, so dense-contact clips are smoothed at the full width while sparse-contact actions such as high-five receive proportionally less smoothing and keep their brief contact peaks. The contact source is the VLM-verified set for Ours and the raw distance candidates for Ours w/o VLMfiltering. All other settings are identical between the two.

## E.3 Dataset Statistics and Filtering

We draw from five contact-rich Inter-X action categories: hug, hand-shake, pat-on-back, high-five, and link-arms. The master pool of reconstructed human–human sequences contains 1355 clips (hug 282, hand-shake 278, pat-on-back 275, high-five 252, link-arms 268).

Teacher pool (1100) We curate the master pool using the Inter-X action label attached to each clip by our VLM annotation: clips whose realised motion does not match the nominal interaction (e.g. lift-up and other off-label variants that are infeasible or unsafe to reproduce on the G1) are removed. This drops 255 clips (18.8%) and leaves N=1100 valid sequences (hug 244, hand-shake 233, pat-on-back 196, high-five 205, link-arms 222). This 1100-clip pool is the teacher training corpus and also the pool on which Jitter and Penetration are reported (Sec. E.4); the 80 manually GT-annotated clips used for Recall/Precision are a subset of it.

Student pool (human-scored subset) For the deployable student we apply a second, stricter curation: two human raters score the retargeted result of each candidate clip on a 3-point scale, and only clips that both rate at the top score (3) are kept, so the student trains on the cleanest contact reproductions. The student additionally restricts to the FOV-filtered subset (Sec. C.3), in which the partner stays inside the torso-camera frustum for the entire horizon, and uses a 9:1 train/evaluation split.

## E.4 Retargeting metric definitions

We detail the four metrics used in the main paper’s retargeting evaluation, namely contact Recall, Precision, Jitter, and Penetration. The ground truth is the set of 80 manually annotated clips of Sec. D.1: each GT contact event e specifies an actor SMPL-X mesh vertex $v _ { e } ^ { \mathrm { a c t } }$ , a partner SMPL-X mesh vertex $v _ { e } ^ { \mathrm { p a r } }$ , and a time window $[ t _ { e } ^ { \mathrm { s } } , t _ { e } ^ { \mathrm { e } } ]$ . The contact threshold is $\epsilon = 1 0$ cm throughout; results at other thresholds are reported in Table 9. Both metrics are averaged across-clip with ± standard deviation.

Recall (vertex-pair-targeted, event-binary) For each GT event e we evaluate the distance between the two annotated mesh vertices: the actor vertex $v _ { e } ^ { \mathrm { a c t } }$ is mapped to a point on the G1 surface through the SMPL-X–G1 registration introduced in Sec. D.2; the partner vertex $v _ { e } ^ { \mathrm { p a r } }$ is taken from the partner’s SMPL-X forward kinematics at frame t. At each frame t in the annotated window we measure the distance between the two mapped vertices, where $g$ denotes the registration-followed-by-G1-FK mapping and h the partner SMPL-X FK, and take the closest approach over the window,

$$
d _ { e } = \operatorname* { m i n } _ { t \in [ t _ { e } ^ { \mathrm { s } } , t _ { e } ^ { \mathrm { e } } ] } \left\| g ( v _ { e } ^ { \mathrm { a c t } } , t ) - h ( v _ { e } ^ { \mathrm { p a r } } , t ) \right\| .\tag{2}
$$

The event is counted as recovered when this closest distance falls below the threshold, that is when $d _ { e } < \epsilon$ . Per-clip Recall is the fraction of recovered GT events, and we report the across-clip mean.

Precision (body-part-centric, event-matching) We partition the SMPL-X mesh into 11 coarse body parts (head, torso, pelvis, left/right {arm, hand, leg, foot}) by SMPL-X skinning-weight argmax; the same labels apply to the partner mesh and, through the registration, to the humanoid surface. For each ordered pair of body parts $( b _ { \mathrm { a c t } } , b _ { \mathrm { p a r } } )$ and each frame t we compute the per-vertex minimum distance between the two parts. A predicted contact event is a contiguous frame run of at least $\delta _ { \operatorname* { m i n } } = 4$ frames (≈ 0.13 s at 30 Hz) on which this distance stays below ϵ. A predicted event matches a GT event if (a) both body-part identities agree strictly (no left/right or hand/arm equivalence; the

Table 9: Retargeting Recall vs. contact threshold ϵ (cm) (per-event, across-clip mean, %). The ϵ=10 column reproduces the Recall column of the main paper’s retargeting table. Relaxing ϵ shows that for Ours the contact is in fact established but slightly displaced: Recall rises from 59.4% at 10 cm to 80.0% at 15 cm and 92.0% at 20 cm.
<table><tr><td>Method  $\backslash \epsilon \mathrm { ( c m ) }$ </td><td>3</td><td>5</td><td>10</td><td>15</td><td>20</td><td>25</td><td>30</td></tr><tr><td>GMR OmniRetarget</td><td>0.0 2.7</td><td>0.0 4.9</td><td>0.4 22.1</td><td>0.6 43.3</td><td>1.3 58.4</td><td>4.9 77.3</td><td>10.6 86.2</td></tr><tr><td>Ours w/o Stage 2 co-opt</td><td>1.7</td><td>5.0</td><td>13.4</td><td>33.6</td><td>54.4</td><td>72.4</td><td>79.4</td></tr><tr><td>Ours w/o VLM filtering</td><td>12.1</td><td>23.1</td><td>58.0</td><td>79.3</td><td>87.2</td><td>91.3</td><td>97.2</td></tr><tr><td>Ours</td><td>12.1</td><td>27.1</td><td>59.4</td><td>80.0</td><td>92.0</td><td>94.3</td><td>95.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

GT body-part identity is the skinning-argmax label of $v _ { e } ^ { \mathrm { a c t } } , v _ { e } ^ { \mathrm { p a r } } )$ and (b) the predicted and GT time windows overlap (Jaccard-style $\geq 1$ shared frame; we report identical conclusions under $\tau _ { 0 . 5 }$ and $\mathrm { { I o U } _ { 0 . 3 } }$ overlap variants). Per-clip Precision is the fraction of predicted events with at least one match; clips with no predicted events are excluded.

Why the two metrics use different granularities Recall verifies known GT events, each of which already names the exact actor and partner vertices, so it can use the tightest per-vertex test. Precision must instead discover unknown predicted contacts and penalise spurious ones, which means scanning the whole retargeted motion for emergent G1–partner proximities. A per-vertex scan is ill-posed here because the actor side is the G1 surface, a different embodiment, and exhaustive vertex-vsvertex testing is dominated by mesh-resolution noise, so we score Precision at the coarser body-part level. We did try a vertex-pair precision and discarded it. Scoring only the GT-named pairs rewards conservative methods that attempt few contacts, which earns high precision by abstention, and never counts the spurious near-contacts a method generates elsewhere. By contrast, the VLM contact-label evaluation of Sec. D.1 is a label-vs-label setting, with both sides annotations on the same SMPL-X mesh, so it uses the per-vertex test for both precision and recall.

Jitter and Penetration Jitter measures motion smoothness as the mean magnitude of the third time derivative (jerk) of the humanoid body-keypoint trajectories. With $\mathbf { p } _ { t , k }$ the world position of G1 body keypoint k at frame t, taken over the 23 link origins including the root, and $\Delta t = 1 / \mathrm { f p s }$ the clip timestep,

$$
\ddot {  { \mathbf { p } } } _ { t , k } = \frac {  { \mathbf { p } } _ { t + 1 , k } - 3  { \mathbf { p } } _ { t , k } + 3  { \mathbf { p } } _ { t - 1 , k } -  { \mathbf { p } } _ { t - 2 , k } } { \Delta t ^ { 3 } } , \qquad \mathrm { J i t t e r } = \operatorname* { m e a n } _ { t , k } \big \| \dddot {  { \mathbf { p } } } _ { t , k } \big \| ,\tag{3}
$$

averaged over interior frames and keypoints within a clip and then over clips, in $\mathrm { m } / \mathrm { s } ^ { 3 }$ , where lower is smoother. Penetration measures how far the humanoid interpenetrates the partner’s body. For each frame we sample surface points ${ \bf x } _ { i , t }$ on the contact-relevant G1 links and compute, directly against the partner SMPL-X body mesh, the depth $\delta _ { i , i }$ <sub>t</sub> by which each point lies inside it, with $\delta _ { i , t } = 0$ for points outside the mesh. The per-frame penetration is the deepest such depth, $\mathrm { p e n } _ { t } = \mathrm { m a x } _ { i } \delta _ { i , t }$ which we average over frames within a clip and then over clips and report in centimetres, where lower is better. Penetration of the humanoid with itself is not counted.

Pool definitions Jitter and Penetration are reported on the full N=1100 valid-sequence pool, the VLM-label-filtered Inter-X corpus defined in Sec. E.3, of which the 80 GT-annotated clips are a subset. OmniRetarget is scored on $N { = } 7 6 6$ of these, because its hard collision solver drops the remaining ≈30% of the clips as infeasible. Recall and Precision are reported on the 80 GT-annotated clips only.

Reading the per-action results Table 10 places Recall and Precision side by side per action, which separates two failure modes that the aggregate numbers hide. Stage 2 co-optimisation is what creates contact at all. Without it Recall averages 13.4% and is 0% on hand-shake, because the Stage-1 initialisation never closes the palm-to-palm gap. Given Stage 2, the VLM filter helps most where contact is short and palm-localised. On high-five it lifts both axes $( \mathrm { R ~ 4 9 . 0 \to 7 0 . 8 , P ~ 2 7 . 8 \to 3 3 . 3 } )$ the best case for our method, whereas on hand-shake the distance candidates already localise the grip, so the filter trades a little Recall (84.4 → 81.2) for cleaner Precision. The actions where $F _ { 1 }$ is lowest split by which axis is binding. Pat-on-back is recall-rich but precision-poor (close-range arm motion fires many spurious near-contacts), while hug is recall-bound for every method, because 33% of its GT events put the actor on the torso/pelvis, outside the hand/forearm/upper-arm regions the predictor enumerates, a coverage limit of the candidate generator, not of the retargeter.

Table 10: Per-action retargeting Recall and Precision (ϵ=10 cm, per-event, across-clip mean<sub>±std</sub>, %). For each action the two columns are Recall (R, vertex-pair-targeted) and Precision (P, body-part τ<sub>0</sub>); the Total columns equal the Recall and Precision columns of the main paper’s retargeting table. Recall uses 16 GT clips per action (58 in total for OmniRetarget, whose collision solver makes the rest infeasible); Precision averages over clips with at least one predicted event.
<table><tr><td rowspan="2">Method</td><td colspan="2">Hug</td><td colspan="2">Hand-shake</td><td colspan="2">Pat-on-back</td><td colspan="2">High-five</td><td colspan="2">Link-arms</td><td colspan="2">Total</td></tr><tr><td>R</td><td>P</td><td>R</td><td>P</td><td>R</td><td>P</td><td>R</td><td>P</td><td>R</td><td>P</td><td>R</td><td>P</td></tr><tr><td>GMR OmniRetarget</td><td>0.0 17.4</td><td>1.3 4.5</td><td>0.0 20.8</td><td>0.0 21.8</td><td>0.0 16.7</td><td>7.6 3.4</td><td>0.0 9.4</td><td>0.0 24.5</td><td>2.1 54.2</td><td>1.9 7.6</td><td>0.4 22.1</td><td>2.3 12.5</td></tr><tr><td>Ours w/o Stage 2 co-opt</td><td>17.5</td><td>5.2</td><td>0.0</td><td>19.6</td><td>9.4</td><td>8.2</td><td>6.2</td><td>24.1</td><td>33.9</td><td>9.7</td><td>13.4</td><td>11.8</td></tr><tr><td>Ours w/o VLM filtering</td><td>38.1</td><td>6.4</td><td>84.4</td><td>15.3</td><td>42.4</td><td>5.0</td><td>49.0</td><td>27.8</td><td>75.0</td><td>4.5</td><td>58.0</td><td>11.6</td></tr><tr><td>Ours</td><td>31.7</td><td>6.8</td><td>81.2</td><td>19.7</td><td>42.9</td><td>5.9</td><td>70.8</td><td>33.3</td><td>68.8</td><td>5.2</td><td>59.4</td><td>14.2</td></tr></table>

Reading the absolute values and spread Both metrics are deliberately strict and are not binary interaction-success rates. Recall demands the registered actor point fall within ϵ=10 cm of the exact GT partner vertex. Relaxing ϵ shows the contact is present but slightly displaced (Ours $5 9 . 4 \% \to 8 0 . 0 \% \to 9 2 . 0 \%$ at 10/15/20 cm, Table 9). Precision counts every body-part near-pass as a predicted event, so a method that genuinely reaches contact sweeps past many partner parts during approach and retraction, emitting ≈ 9× more predicted events than GT (1740 vs. 203 for Ours) and capping Precision near 14%. The large across-clip std in the main paper’s retargeting Recall (Ours $5 9 . 4 3 _ { \pm 3 8 . 9 1 } )$ is a property of this binary per-event metric on short clips. Per-clip Recall averages only 2–3 events and so takes coarse values in $\{ 0 , { \frac { 1 } { 2 } } , { \frac { 2 } { 3 } } , 1 \}$ , whereas the action-level means span a far narrower 32–81%, i.e. the spread is per-clip difficulty, not systematic action differences. Finally, OmniRetarget is scored on only its 58 feasible clips, whichfavours it at larger thresholds (unreachable clips are dropped, not counted as misses). The ϵ=10 cm comparison where Ours leads on both axes is therefore the conservative reading. What these metrics do not capture, namely whether the interaction visually works, is addressed by the user study (Table 11) and the real-robot results (Sec. A.1).

## E.5 Retargeting User Study: protocol and per-question results

Sequences and conditions The study set consists of 80 human–human interaction sequences (16 per action × 5 actions: Hug, Hand-shake, Pat-on-back, High-five, Link-arms). For each sequence we render the four retarget conditions of the main paper (GMR, OmniRetarget, Ours w/o VLM filtering, Ours), all displayed side-by-side in randomised positions {1, 2, 3, 4}. Above the four method panels we show the original SMPL-X demonstration as a 3-view orbit reference. Both the sequence order and the position-to-method assignment are per-participant random but deterministic from a hash of the participant’s name, so that interrupted sessions resume on the same trial layout.

Task For every trial the participant assigns a strict ranking over the four conditions, {1, 2, 3, 4} with 1 = best and 4 = worst (permutation, no ties), separately for three independent questions: (Q1) G1 contact + motion quality, (Q2) Human contact + motion quality, (Q3) Overall preference. Participants were instructed to weight five criteria together when forming each ranking. (i) Pose is the consistency of the rendered body pose with the human demo above. (ii) Contact is whether contact occurs at the location implied by the action label, for example torso-to-torso for hug and hand-to-hand for hand-shake. (iii) Jitter is the absence of high-frequency shaking. (iv) Penetration is the magnitude of mesh interpenetration with the partner. (v) Stability is the absence of over-stretched limbs and unnatural joint bends. They were also told to prefer a refined-but-cleaner result over a faithful reproduction of any noise present in the original demonstration. The study is delivered as a web app that records each (participant, sequence, question, condition, rank) row to a serverless key-value store on submission of every trial.

![](images/6f79c4df064bba3716e7a083c100ec7203d1e7c2a2341403a925e7283bc0b667.jpg)  
Figure 14: User-study web interface (one trial). The original demonstration (3-view orbit, top) and the four blinded retarget conditions A–D (middle) are played synchronously; the participant gives a strict 1–4 ranking for each of the three questions (bottom). Full ranking criteria are in Sec. E.5.

Results We collected responses from 40 participants and restrict the analysis to the N=28 who finished all 80 trials with valid {1, 2, 3, 4} permutations on all three questions. For each participant we compute the mean rank of each condition averaged over the 80 sequences; Table 11 reports the across-participant mean ± standard error (SEM, n=28) per (question, condition) pair.

Table 11: User study: per-question mean ranks ± SEM (n=28 participants, 80 sequences each, lower is better). Each entry is the across-participant mean of per-participant per-condition mean ranks. Ranks lie in {1, 2, 3, 4} with 1=best.
<table><tr><td>Method</td><td>Q1: G1 contact + motion ↓</td><td>Q2: Human contact + motion ↓</td><td>Q3: Overall preference ↓</td></tr><tr><td>GMR</td><td> $3 . 5 2 \pm 0 . 0 7$ </td><td> $3 . 4 5 \pm 0 . 0 7$ </td><td> $3 . 6 1 \pm 0 . 0 4$ </td></tr><tr><td>OmniRetarget</td><td> $2 . 6 4 \pm 0 . 0 4$ </td><td> $2 . 4 9 \pm 0 . 0 5$ </td><td> $2 . 5 7 \pm 0 . 0 4$ </td></tr><tr><td>Ours w/o VLM filtering</td><td> $2 . 0 5 \pm 0 . 0 4$ </td><td> $2 . 1 8 \pm 0 . 0 5$ </td><td> $2 . 0 6 \pm 0 . 0 3$ </td></tr><tr><td>Ours</td><td> ${ \bf 1 . 7 9 \pm 0 . 0 4 }$ </td><td> $\mathbf { 1 . 8 8 \pm 0 . 0 5 }$ </td><td> ${ \bf 1 . 7 6 \pm 0 . 0 3 }$ </td></tr></table>

The full pipeline takes the best mean rank on every question. The largest gap on Q1 (G1 contact: 1.79 vs. OmniRetarget’s 2.64, a 0.85-rank improvement) confirms that the VLM-guided contact annotation is the component most directly tied to robot-side contact correctness. Removing VLM filtering (Ours w/o VLM filtering) costs about 0.26 rank on Q1 and 0.30 on Q3, consistent with the precision drop reported in the main paper’s retargeting comparison. GMR is ranked last on every question across participants (≥ 3.45), reflecting its lack of any contact-aware objective.

Statistic and significance Each tabulated value is an across-participant mean of per-participant per-condition mean ranks; the reported ± is the standard error of that mean over $n { = } 2 8$ participants (SEM, not the across-participant std). The per-condition SEMs, at most 0.07, are far smaller than the gaps between adjacent methods, at least 0.26 rank, so the orderings are unambiguous.

## F Teacher Policy: observation breakdown, reward weights, and training

Observation The teacher’s proprioception vector ${ \bf o } _ { t } ^ { \mathrm { p r o p } } = [ \omega _ { b } , { \bf q } , \dot { \bf q } , { \bf a } _ { t - 1 } ^ { \mathrm { j o i n t } } ]$ concatenates base angular velocity (3), 29 joint positions, 29 joint velocities, and the previous joint-level action (29) output by the HOMIE conversion below. The reference and partner inputs comprise the target G1 joint positions and velocities at the current frame and at four future offsets ([1, 2, 4, 8] steps), the reference root orientation in the base frame, and the partner’s root and body positions relative to the robot base. Small per-term observation noise (joint positions $\sigma { = } 0 . 0 1$ , joint velocities $\sigma { = } 0 . 5$ base angular velocity $\sigma { = } 0 . 2 .$ partner positions $\sigma { = } 0 . 1 )$ regularises the policy during rollouts. The critic is noise-free and additionally observes privileged terms unavailable to the actor, namely base linear velocity, the robot’s body positions and orientations in the base frame, and the reference body position.

Action The teacher emits $\mathbf { a } _ { t } \in \mathbb { R } ^ { 1 8 }$ : four HOMIE locomotion commands $( v _ { x } , v _ { y } , \omega _ { z } , h )$ and 14 residual upper-body joint offsets. The HOMIE tracker converts the 4-d command into 15 lower-body joint actions, and the 14 residuals are added in action space to the demonstration’s reference arm pose. The tracker is kept frozen throughout teacher training.

Reward weights The aggregate per-step reward is

$$
r _ { t } = 3 r _ { t } ^ { \mathrm { d i r } } + 5 r _ { t } ^ { \mathrm { d i s t } } + 0 . 5 r _ { t } ^ { \mathrm { f o r e e } } + 0 . 5 r _ { t } ^ { \mathrm { r o o p o s } } + 0 . 5 r _ { t } ^ { \mathrm { r o o t o r i } } + r _ { t } ^ { \mathrm { b o d y p o s } } + r _ { t } ^ { \mathrm { b o d y o r i } } + r _ { t } ^ { \mathrm { b o d y v e l } } + r _ { t } ^ { \mathrm { b o d y a z } } + r _ { t } ^ { \mathrm { n o t o r g i } } + r _ { t } ^ { \mathrm { n o t o r g i } } + r _ { t } ^ { \mathrm { n o t o r g i } } ,
$$

where each motion-tracking term has the exponential form $\exp ( - e / \sigma ^ { 2 } )$ , with e the per-body mean squared tracking error and σ the error scale at which the reward decays to $1 / e$ (a smaller σ enforces tighter tracking). The relative-frame body terms use body-position $\sigma = 0 . 3 \mathrm { m }$ , body-orientation $\sigma = 0 .$ 4 rad, body-velocity $\sigma = 1 . 0 \mathrm { m / s }$ , and body-angular-velocity $\sigma = 3 .$ 14 rad/s. Regularisation: action rate $- 3 { \times } 1 0 ^ { - 2 }$ , joint velocity $- 2 \times 1 0 ^ { - 4 } .$ , joint acceleration $- 1 \times 1 0 ^ { - 7 }$ , torque $- 1 \times 1 0 ^ { - 4 }$ DOF-position soft limit −100 (with 0.9 soft-limit ratio), DOF-velocity soft limit $- 1 \times 1 0 ^ { - 3 }$ , torque soft limit $- 3 { \times } 1 0 ^ { - 2 }$ . The no-fly term rewards at least one foot in contact (+1 when satisfied).

Initialisation Episodes use a hybrid physical state initialisation: with probability 0.5 the episode restarts from the policy’s own recent on-policy state; otherwise it is seeded from the demonstration’s first frame perturbed by Gaussian noise with joint positions $\sigma ~ = ~ 0 . 1$ , root translation $\sigma _ { x y z } ~ = ~ ( 0 . 0 5 , 0 . 0 5 , 0 . 0 1 )$ m, root orientation $\sigma ~ = ~ ( 0 . 1 , 0 . 1 , 0 . 2 )$ rad, root linear velocity $\sigma = ( 0 . 5 , 0 . 5 , 0 . 2 ) \mathrm { m / s }$ , and root angular velocity $\sigma = ( 0 . 5 2 , 0 . 5 2 , 0 . 7 8 )$ rad/s. The initial action noise std is $\sigma _ { \mathrm { i n i t } } = 0 . 5$ . The partner is driven by a reference PD controller that tracks the demonstration.

PPO Both actor and critic are 5-layer MLPs with hidden sizes [512, 512, 512, 256, 128] and ELU activations. We use PPO [41] with $\gamma = 0 . 9 9 , \mathrm { G A E } \lambda = 0 . 9 5 .$ , clipping 0.2, 8 update epochs per rollout, 4 minibatches, entropy coefficient 0.005, and AdamW [50] with learning rate $1 0 ^ { - 5 }$ for both actor and critic. We collect 24 environment steps per iteration across 8,192 parallel Isaac Sim environments on a single L40S GPU.

Evaluation metrics: MPJPE and Contact Recall The teacher is evaluated with MPJPE and Contact Recall, and the student that distils it with Contact Recall. MPJPE (mean per-joint position error) measures tracking fidelity, the mean Euclidean distance between the humanoid body keypoints under physics and the same keypoints of the retargeted reference the policy tracks,

$$
\mathrm { M P J P E } = \underset { t , k } { \mathrm { m e a n } } \left\| \mathbf { p } _ { t , k } ^ { \mathrm { s i m } } - \mathbf { p } _ { t , k } ^ { \mathrm { r e f } } \right\| ,\tag{4}
$$

reported in centimetres. Contact Recall measures how often the policy realises the demonstrated contacts under physics. Manual annotation does not scale to the policy evaluation pools (all student pool motions for the teacher, the held-out split for the student), so we use the VLM-verified contact annotations of Sec. D.1 as ground truth. For every annotated contact frame we carry the prescribed actor contact site onto the G1 surface through the registration of Sec. D.2, measure its distance to the annotated partner vertex under partner forward kinematics, and count the frame as satisfied when this distance is below 15 cm for the teacher and 30 cm for the student. Contact Recall is the fraction of satisfied (frame, contact site) pairs, pooled over all evaluated sequences.

## F.1 Contact Reward: coefficients and force gate

Each VLM-verified annotation specifies, per frame, the active contact pairs between fixed G1 contact links and partner surface points. The six candidate links are the left and right wrist, elbow, and shoulder. For a pair h active at time t, let $\mathbf { p } _ { t } ^ { h }$ denote the G1 contact point, $\mathbf { j } _ { t } ^ { h }$ the target partner vertex, $\mathbf { n } _ { t } ^ { h }$ the partner surface normal at that vertex, and $F _ { t } ^ { h }$ the contact-force magnitude on the corresponding G1 link. The G1 contact point is recovered by forward kinematics from the meshprojection registration of Sec. D.2, $\mathbf { p } _ { t } ^ { h } = T _ { \mathrm { l i n k } ^ { h } } ( \mathbf { q } _ { t } ) \mathbf { p } _ { \mathrm { l o c a l } ^ { h } }$ , where $( \operatorname { l i n k } ^ { h } , \mathbf { p } _ { \mathrm { l o c a l } ^ { h } } )$ is the registered G1 link and surface point for the contact. The world-frame target $\mathbf { j } _ { t } ^ { h }$ is the partner contact-vertex position from the SMPL-X forward kinematics of the reference motion.

We supervise the teacher with three complementary contact rewards. The first drives the demonstrated contact closed and gates it on real force,

$$
r _ { t } ^ { \mathrm { d i s t } } = \Big \langle \phi ( F _ { t } ^ { h } ) \exp \big ( - \sigma _ { d } | { \bf p } _ { t } ^ { h } - { \bf j } _ { t } ^ { h } | ^ { 2 } \big ) \Big \rangle _ { h } , \qquad \phi ( F ) = \mathrm { m i n } \big ( \exp ( F - \tau ) , 1 \big ) ,\tag{5}
$$

averaged over the active pairs $h ,$ , with sharpness $\sigma _ { d } = 5 0 \mathrm { { m ^ { - 2 } } }$ and force-gate threshold $\tau = 1 \Nu$ (the gate is active because robot–partner collisions are enabled). The second encourages the contact to be approached from the physically correct side and discourages satisfying the distance objective through interpenetration,

$$
\boldsymbol { r } _ { t } ^ { \mathrm { d i r } } = \left. \mathrm { c l a m p } ( \hat { \mathbf { v } } _ { t } ^ { h } \cdot \mathbf { n } _ { t } ^ { h } , 0 , 1 ) \cdot \mathrm { c l a m p } ( \mathbf { n } _ { t } ^ { h , \mathrm { G l } } \cdot ( - \hat { \mathbf { v } } _ { t } ^ { h } ) , 0 , 1 ) \right. _ { h } , \qquad \hat { \mathbf { v } } _ { t } ^ { h } = \frac { \mathbf { p } _ { t } ^ { h } - \mathbf { j } _ { t } ^ { h } } { \operatorname* { m a x } \bigl ( | \mathbf { p } _ { t } ^ { h } - \mathbf { j } _ { t } ^ { h } | , \epsilon \bigr ) } ,\tag{6}
$$

with $\epsilon = 0 . 0 2 \mathrm { m }$ . The first factor requires the approach direction to align with the partner surface normal $\mathbf { n } _ { t } ^ { h }$ , and the second requires the robot’s own contact-surface normal $\mathbf { n } _ { t } ^ { h , \mathrm { G 1 } }$ to face the partner.

The third regulates the established contact force. Per pair, with contact label $l _ { t } ^ { h }$ and force band $[ F _ { \mathrm { m i n } } , F _ { \mathrm { m a x } } ] = [ 5 , 5 0 ] \mathrm { N } .$

$$
\begin{array} { r } { r _ { t } ^ { \mathrm { f o r e } , h } = \left\{ \begin{array} { l l } { 1 , } & { l _ { t } ^ { h } = 1 , F _ { t } ^ { h } \in [ F _ { \mathrm { m i n } } , F _ { \mathrm { m a x } } ] , } \\ { \exp \big ( - c _ { u } ( F _ { \mathrm { m i n } } - F _ { t } ^ { h } ) \big ) , } & { l _ { t } ^ { h } = 1 , F _ { t } ^ { h } < F _ { \mathrm { m i n } } , } \\ { \exp \big ( - c _ { o } ( F _ { t } ^ { h } - F _ { \mathrm { m a x } } ) \big ) , } & { l _ { t } ^ { h } = 1 , F _ { t } ^ { h } > F _ { \mathrm { m a x } } , } \\ { - c _ { w } F _ { t } ^ { h } , } & { l _ { t } ^ { h } = 0 , F _ { t } ^ { h } > F _ { \mathrm { m i n } } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{7}
$$

summed over pairs to give $\begin{array} { r } { r _ { t } ^ { \mathrm { f o r c e } } = \sum _ { h } r _ { t } ^ { \mathrm { f o r c e } , h } } \end{array}$ , with $c _ { u } = 0 . 5 , c _ { o } = 0 . 0 5$ , and $c _ { w } = 0 . 0 0 5 .$

In the aggregate teacher reward these terms are weighted $3 . 0 r _ { t } ^ { \mathrm { d i r } } + 5 . 0 r _ { t } ^ { \mathrm { d i s t } } + 0 . 5 r _ { t } ^ { \mathrm { f o r c e } } .$

## G Limitations

Fixed-view, passive perception The student observes a single forward-facing depth camera and does not control where it looks. Under heavy self-occlusion the partner can leave the field of view entirely, and the policy cannot actively reposition the camera to recover the missing state. Coupling the interaction policy with active head or gaze control, so the robot moves to keep the relevant contact region in view, is a natural extension.

Limited dexterity REACT drives the body through HOMIE locomotion commands plus residual upper-body joint offsets and treats the hands as rigid end effectors. It does not actuate fingers, so interactions that require grasping, fine in-hand manipulation, or force-sensitive hand shaping, such as a firm but gentle hand-hold or adjusting a grip, are out of scope. Extending the action space to dexterous hands is needed for this class of contact.

Simple interaction repertoire We evaluate on five short, largely upper-body interactions (hug, high-five, handshake, link-arms, pat-on-back). More complex and longer-horizon behaviors that involve sustained multi-contact load bearing and tighter coupling to the partner’s intent are beyond what the current contact reward and demonstration pipeline target, and scaling to them is left to future work.

## H VLM Verification Prompt (verbatim)

The template below is the exact text sent to Gemini 2.0 Flash for each 3 s video chunk, alongside the horizontally concatenated three-view frames (extracted at 5 fps and cropped to the two figures). The prompt first asks the model to reason about the contact patterns expected for the action, then to classify each distance-based candidate (mocap proximity < 15 cm) into one of essential, incidental, unnecessary, or avoid, emitting a structured JSON verdict per candidate. Runtime fields are substituted at call time: {action\_name}, {timestamp\_list}, {num\_views} (= 3), {view\_labels} (view-0, view-1, view-2), {text\_description} (the Inter-X interaction description), {candidates\_formatted} (the enumerated distance candidates), and {num\_candidates}. Unicode arrows in the original (↔, →) are transcribed here as <-> and ->.

```markdown
You are an expert analyst of close human-human interactions. Your task
is to classify distance-based contact candidates by their **semantic
role** relative to the action being performed.
## Video Information
- Action type: {action_name}
- Frame timestamps: {timestamp_list}
- **Multi-view**: Each frame image contains {num_views} views
concatenated horizontally, labeled as {view_labels} (from left to
right)
## Person Identification
```

##   
## STEP 1: Determine Expected Contact Patterns   
##

- \*\*P1 (Blue person)\*\*: The person rendered in BLUE color   
- \*\*P2 (Orange person)\*\*: The person rendered in ORANGE color

NOTE: "first person" / "second person" in the description may NOT   
correspond to P1(Blue)/P2(Orange). Use the semantic content to   
understand WHAT contact should occur, then visually identify WHO is   
performing it.

## Body Part Naming Convention   
‘left\_wrist‘ / ‘right\_wrist‘: Represents the entire hand including all   
fingers   
‘left\_elbow‘ / ‘right\_elbow‘: Represents the forearm region   
‘left\_shoulder‘ / ‘right\_shoulder‘: Represents the upper arm and   
shoulder region   
Contact categories for the initiating part: "hand", "forearm",   
"upper\_arm"

## ## Contact Surface Definitions

\*\*CONTACTING SURFACE\*\* (the person initiating contact):   
‘"palm/inner"‘: Palm side of hand, inner side of arm (gripping/   
embracing side)   
‘"back/outer"‘: Back of hand, outer side of arm

\*\*CONTACTED SURFACE\*\* (the person being touched):   
‘"inner"‘: Inner/front surface of the body part being touched   
- ‘"outer"‘: Outer/back surface of the body part being touched

Before evaluating candidates, you MUST FIRST reason about what joint contact pairs and surfaces are semantically expected for this action.

\*\*Reference patterns by action type\*\* (use as guidance - adapt based on the text description):

## ### High-five

\*\*essential\*\*: hand(palm/inner) <-> wrist(palm/inner) - palm-to-palm   
contact

\- \*\*avoid\*\*: hand <-> head (hand hitting head)

\*\*NOT expected\*\* (mark as ‘unnecessary‘): hand <-> elbow, forearm <->   
wrist, upper\_arm <-> elbow - these are incidental proximity from arm   
extension, NOT actual contact in a high-five

## ### Hug

- \*\*essential\*\*: hand(palm/inner) <-> spine3/back(outer) - palms   
embracing the back   
\*\*essential\*\*: forearm(inner) <-> shoulder/collar(outer) - arms   
wrapping around shoulders   
\*\*incidental\*\*: upper\_arm(inner) <-> shoulder/upper\_arm(outer) - may   
occur but not the defining contact   
\*\*incidental\*\*: hand(palm/inner) <-> shoulder(outer) - hand resting on   
shoulder during hug

\- \*\*avoid\*\*: hand <-> head, elbow <-> face

## ### Pat on back

\*\*essential\*\*: hand(palm/inner) <-> spine3/back(outer) - palm patting the back

\- \*\*incidental\*\*: hand(palm/inner) <-> shoulder/collar(outer) - patting the shoulder area

\- \*\*avoid\*\*: hand <-> head

\- \*\*NOT expected\*\* (mark as ‘unnecessary‘): forearm <-> wrist, upper\_arm <-> elbow

## ### Link arms ### Link arms

\*\*essential\*\*: forearm(inner) <-> forearm(inner) - inner forearms   
interlocked   
\*\*essential\*\*: hand(palm/inner) <-> elbow/forearm(outer) - hand   
gripping the other’s elbow   
\*\*incidental\*\*: upper\_arm <-> forearm - may touch during interlocking   
\*\*NOT expected\*\* (mark as ‘unnecessary‘): hand <-> spine3, hand <->   
neck

## ### Walk towards

```markdown
### Multi-view validation:
- For ‘essential‘ or ‘incidental‘: at least one view MUST show the
contact with HIGH confidence.
- For ‘unnecessary‘ or ‘avoid‘: evidence_views should be empty.
```

4. \*\*Duration\*\*: Is the duration appropriate for the action type?   
- Hug: 2-10s typical. <0.5s too brief.   
- High-five: 0.3-2s acceptable.   
- Pat on back: 0.3-1s per pat, series may span longer.   
- Link arms: 2-10s+ sustained.

```coffeescript
### is_valid rules:
‘essential‘ -> is_valid = true
‘incidental‘ -> is_valid = true
‘unnecessary‘ -> is_valid = false
‘avoid‘ -> is_valid = false
```

Generally NO intentional contact expected. Mark candidates as   
‘unnecessary‘ unless the text description explicitly mentions touching   
(e.g., handshake, bump).

Based on action type "{action\_name}" and the text description, determine the specific expected contact patterns including sides (left/right) and surfaces.

##   
## STEP 2: Classify Each Candidate   
##

The following contact candidates were detected based on spatial proximity (distance < 15cm):

{candidates\_formatted}

\*\*CRITICAL PRINCIPLE\*\*: Spatial proximity alone does NOT mean the contact is semantically meaningful. During any action, many body parts come close together, but only specific joint pairs represent actual intended contact. Your job is to classify each candidate’s semantic role.

## ### Semantic Labels

Each candidate MUST be assigned one of these labels:

| semantic\_label | is\_valid | Meaning |   
|---|---|---|   
| ‘essential‘ | true | Required for the action to occur (e.g.,   
palm-to-palm in high-five) |   
‘incidental‘ | true | May naturally occur but is not required (e.g.,   
upper arms touching during a hug) |   
‘unnecessary‘ | false | Not part of the action - incidental proximity   
or mocap artifact (e.g., hand near elbow during high-five) |   
‘avoid‘ | false | Contact that should NOT occur - collision or   
unnatural contact (e.g., hand hitting head, body penetration) |

## ### Classification Criteria

For each candidate, evaluate:

1. \*\*Semantic match\*\*: Does the joint pair match an expected contact pattern from Step 1?

2. \*\*Surface match\*\*: Are the contacting/contacted surfaces consistent with the expected pattern?

3. \*\*Visual evidence\*\*: Examine ALL {num\_views} views. Is the contact visible in at least one view with HIGH confidence?

5. \*\*Temporal context\*\*: If multiple candidates overlap in time, they may represent the same physical event. Mark the most semantically appropriate one as essential; mark redundant ones as incidental or unnecessary.

CRITICAL: You MUST evaluate ALL {num\_candidates} candidates listed above.

Output ONLY valid JSON in this format:   
‘‘‘json   
{   
"expected\_contacts": [   
{   
"pattern": "hand <-> wrist",   
"semantic\_label": "essential",   
"contacting\_surface": "palm/inner",   
"contacted\_surface": "palm/inner",

```csv
"description": "High-five: palm-to-palm contact between right hands
},
{
"pattern": "hand <-> head",
"semantic_label": "avoid",
"contacting_surface": "",
"contacted_surface": "",
"description": "Hand should not hit the head"
}
],
"validations": [
{
"candidate_id": 0,
"is_valid": true,
"semantic_label": "essential",
"matched_pattern": "hand <-> wrist",
"confidence": "high",
"evidence_views": [0, 2],
"contacting_surface": "palm/inner",
"contacted_surface": "palm/inner",
"reason": "P1(Blue)’s right hand touching P2(Orange)’s right
wrist for 0.7s. Matches expected high-five palm-to-palm pattern.
High confidence in view-0 and view-2."
},
{
"candidate_id": 1,
"is_valid": false,
"semantic_label": "unnecessary",
"matched_pattern": "none",
"confidence": "low",
"evidence_views": [],
"contacting_surface": "",
"contacted_surface": "",
"reason": "P1(Blue)’s right hand near P2(Orange)’s right elbow.
Does NOT match any expected high-five pattern. Hand-to-elbow
proximity is incidental to arm extension, not an actual contact."
},
{
"candidate_id": 2,
"is_valid": false,
"semantic_label": "avoid",
"matched_pattern": "hand <-> head",
"confidence": "high",
"evidence_views": [1],
"contacting_surface": "",
"contacted_surface": "",
"reason": "P2(Orange)’s hand appears to collide with P1(Blue)’s
head. This is an undesirable contact that should be avoided."
}
]
}
‘‘‘
RULES:
Output ‘expected_contacts‘ array FIRST, listing all expected patterns
for this action (with semantic_label and surfaces)
Provide a validation entry for EVERY candidate ({num_candidates}
total)
Every validation MUST include ‘semantic_label‘ (essential/incidental/
unnecessary/avoid) and ‘matched_pattern‘
‘matched_pattern‘: which expected pattern this candidate matches, or
‘"none"‘ if no match
**is_valid=true REQUIRES**: semantic_label is essential or incidental,
AND at least one view with HIGH confidence
**is_valid=false**: if semantic_label is unnecessary or avoid,
regardless of distance
Use P1(Blue) and P2(Orange) terminology
For ‘essential‘/‘incidental‘ contacts: ALWAYS specify
contacting_surface and contacted_surface
For ‘unnecessary‘/‘avoid‘ contacts: leave contacting_surface and
contacted_surface as empty strings
```