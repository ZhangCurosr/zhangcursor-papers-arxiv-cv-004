# X-FLY: Learning to Reach and Stop in Full-Chunk Aerial Vision-Language Navigation

Ruibin Ju Ruifeng Cai Min Xu Lin Wang

ARIDGE, XPENG

Corresponding author: jurb@aridge.com

Project page: https://juliusju572.github.io/x-fly/

## Abstract

Aerial vision-language navigation (VLN) agents built on large vision-language models are expensive to query. Executing a complete predicted action chunk before replanning amortizes this cost, reducing model inference from 195 to 42 ms per executed waypointfor our policy. It also moves the states at which the policy is queried away from expert demonstrations, which supervise neither recoveryfrom these states nor termination at the learner’s own arrival. On OpenFly test-seen, no-stopfailures accountfor 47.78% of full-chunk behavior cloning (BC) outcomes. We present X-FLY, a staged training framework for full-chunk aerial VLN. Recovery DAgger labels learner-visited decision states with smooth chunks that reconnect tofuture expert segments and with STOP at the first in-region state. Flow-GRPO then jointly refines the chunkedflow controller and a Bernoulli STOP policy from same-task closed-loop outcomes, supplemented by outcome-verified stopping supervision. With eight controls per decision, X-FLY reaches 74.18% SR on testseen and 51.73% on test-unseen, compared with 16.45% and 18.43%forfull-chunk BC and 32.04% and 29.46%for DreamFly.

## 1. Introduction

Aerial VLN asks an embodied agent to follow a free-form instruction through a large urban environment and terminate within a target radius [11, 19]. Compared with indoor graph navigation [1, 6], the agent must produce continuous translation and rotation, cope with large viewpoint changes, and recover from control errors that alter every subsequent observation. Success therefore depends on two coupled abilities: flying to the described goal, and deciding when the goal has been reached.

Recent aerial agents condition on large vision-language models (VLMs), which makes every policy query expensive [9, 35]. Action-chunk policies predict several future controls per query [8, 34], yet aerial VLN agents such as DreamFly execute only the first control and query again [9]. Executing the complete chunk amortizes policy inference across its controls: for our fast–slow policy, executing all eight cuts model inference from 195 to 42 ms per executed waypoint (Sec. 4.1).

Full-chunk execution also changes where the policy is queried. The aircraft moves through eight controls before new visual feedback can correct it, so the states at which the policy is next queried, its decision anchors, drift away from the expert trajectories it was trained on. Expert-only data cover these learner-visited anchors poorly in two ways: they contain no chunk that starts at an off-route learner pose and returns it to the route, and their positive STOP labels are confined to expert terminal windows, leaving the anchors where the learner itself arrives poorly represented. We call this the full-chunk support gap. On test-seen, switching

![](images/d17e220fd2c9fb4af4671811e468f280bf3fb47a1feff54b12374c84e8b5933f.jpg)

![](images/1220429bc07e25bbac9abceef4113935387d004ba70e9a02542808af9e48e132.jpg)  
BC + Recovery DAgger + Flow-GRPO DreamFly  
Figure 1. Accuracy–inference trade-off of X-FLY stages. SR under single-step (k=1) and full-chunk (k=8) execution; x-labels give amortized model inference per executed waypoint. On testseen, full-chunk BC reaches the goal but often does not stop. Recovery DAgger makes full-chunk execution viable, and Flow-GRPO, trained and evaluated at k=8 only, gives the highest SR among the evaluated operating points at 4.7× less amortized model inference time per executed waypoint than single-step execution. Inference timings apply to the X-FLY backbone; DreamFly [9] is shown as an SR reference at k=1.

BC from single-step to full-chunk execution drops SR from 46.57% to 16.45% (Fig. 1), although navigation failures barely change (28.02% vs. 30.77%); no-stop failures account for 47.78% of the full-chunk outcomes (Sec. 4.3). Reaching the goal is not the same as stopping there.

X-FLY addresses both sides of the gap with two training stages applied after multi-domain BC (Fig. 2). Recovery DAgger [24] runs the learner with the deployed full-chunk executor and labels every visited anchor with a smooth chunk that decays the learner’s offset and reconnects to the future expert segment; the first anchor inside the success region receives a positive STOP label. Recovery reduces navigation and no-stop failures, while premature stopping remains common. Flow-GRPO [18] then compares closed-loop attempts of the same task and updates the chunked flow controller and a separate Bernoulli STOP policy from their outcomes. With eight controls per decision, X-FLY reaches 74.18% SR on test-seen and 51.73% on test-unseen, all from explicit in-region STOP.

Our contributions are:

• We study full-chunk execution in aerial VLN and characterize the support gap it creates at learner-visited decision anchors, with separate effects on goal reaching and termination.

• We present X-FLY, a staged training framework for full-chunk aerial navigation that combines multi-domain BC, recovery supervision at learner-visited anchors, and closed-loop outcome optimization of chunked control and STOP.

• We demonstrate improved full-chunk navigation on Open-Fly and characterize distinct stage-level gains: recovery improves goal-region reachability and reduces no-stop failures, while outcome post-training further reduces premature termination.

## 2. Related Work

Aerial vision-language navigation. VLN was first studied on discrete indoor navigation graphs [1, 6, 7, 10, 12, 27]. Continuous VLN removes the graph abstraction [14], and AerialVLN and OpenFly extend the setting to long-range urban flight [11, 19]. Recent aerial agents use hierarchical planning, causal memory, diffusion policies, world-action prediction, and fast–slow reasoning [9, 20, 28, 32, 33, 35]. We adopt a fast–slow backbone and study a different axis: the training distribution shift created when such a policy executes a complete chunk before replanning.

Learning when to stop. Graph VLN agents treat STOP as one action of a discrete vocabulary, and several works show that termination deserves dedicated modeling, through a separate stop policy [29] or progress estimation [22]. DreamFly adds a lightweight termination head trained on top of its frozen navigation policy [9]. A continuous-waypoint policy has no intrinsic STOP coordinate, so X-FLY also uses a separate Bernoulli head, but supervises it at the learner’s own arrival anchors and optimizes it jointly with control from closed-loop outcomes.

Flow policies and action chunks. Diffusion and flowmatching policies represent multimodal continuous actions and emit action sequences [4, 8, 17]. Action chunking amortizes policy inference [34]; temporal ensembling, bidirectional decoding, and real-time chunking address the resulting loss of reactivity at inference time [5, 21, 34]. Fast–slow VLA systems separate reasoning from control [16, 30]. X-FLY keeps inference unchanged and instead repairs the training distribution that full-chunk execution induces.

Closed-loop imitation and reinforcement learning. DAgger reduces covariate shift by labeling learner-visited states [24]. DART injects noise into demonstrations [15], HG-DAgger relies on a human corrector [13], and ChauffeurNet synthesizes perturbed driving trajectories that return to the route [2]. TravelUAV applies continuous-trajectory DAgger to aerial navigation with a teacher that searches forward along the reference [28]. Our labels are instead collected at the anchors produced by full-chunk execution, follow the route by the number of executed controls, reconnect within one chunk, and add STOP labels at the learner’s arrival anchors. On the RL side, RCM and EnvDrop combine imitation with RL for VLN [25, 27], RefineFly and WorldVLN post-train aerial policies in closed loop [26, 33], and diffusion and flow policies are optimized by treating denoising as a Markov decision process [3, 18, 23, 31]. We use Flow-GRPO’s stochastic transitions and add a factorized STOP objective to refine recovered full-chunk behavior.

## 3. Method

## 3.1. Problem: the full-chunk support gap

An episode consists of an instruction, onboard RGB observations, and executed controls, and ends at a STOP decision, the control budget, or an abnormal event such as a collision. The fused policy is queried only at decision anchors s<sub>t</sub>, one every k executed controls. At each anchor it outputs a control chunk $U _ { t } = ( u _ { t , 1 } , \dots , u _ { t , H } )$ with $H = 8$ and a STOP decision $z _ { t } \in \{ 0 , 1 \} . \ \mathrm { I f } \ z _ { t } = 0$ , the first k controls of $U _ { t }$ are executed open loop before the next anchor. We deploy $k = H .$

BC supervises $( U , z )$ only at anchors sampled from expert trajectories, $s \sim d ^ { \star }$ , whereas deployment queries the policy at $s \sim d _ { k } ^ { \pi }$ , the anchor distribution induced by the learner’s own k-step execution. Larger k tends to move $d _ { k } ^ { \pi }$ farther from $d ^ { \star }$ , because more controls separate consecutive corrections. Two supervision signals are then missing on the support of $d _ { k } ^ { \pi }$ :

![](images/201576122b6db5f275cd43d24f110b180835f114f6f04efcf27af69bdef8b16d.jpg)  
Figure 2. The full-chunk support gap and how X-FLY addresses it. (a) Full-chunk BC flies eight controls between decisions and visits learner-induced anchors that expert demonstrations cover poorly, with neither a corrective chunk nor an arrival STOP label; the policy overshoots the goal. (b) Recovery DAgger relabels each learner anchor with a chunk whose offset $d _ { t }$ decays by a smoothstep (inset) and rejoins the expert route at the chunk end; the first in-region anchor receives STOP. (c) Flow-GRPO samples six free and two STOP-override attempts per task with low-frequency chunk noise (inset: the four cosine modes) and a Bernoulli STOP policy, and uses their relative outcomes to update both.

• Control gap. At an off-route anchor, no expert chunk starts at the learner’s pose and remains coherent for H steps until the next decision.

• Termination gap. Positive STOP labels come only from expert terminal windows, not from the anchor where the learner itself first enters the success region.

BC (Sec. 3.2) provides the initial policy, Recovery DAgger (Sec. 3.3) supplies both missing labels at the anchors this policy visits at $k { = } H ,$ , and Flow-GRPO (Sec. 3.4) refines the recovered policy from closed-loop outcomes.

## 3.2. Policy and BC initialization

The policy (Fig. 3) follows the fast–slow design of recent aerial agents [9, 35]. Learned query banks extract cached observation and instruction features from a frozen VLM and fresh current-frame features from its visual tower; these tokens, ego state, and a masked history of executed controls form the fusion sequence, with navigation queries as cross-attention memory. The fused representation conditions a rectified-flow controller and a separate STOP head, so control and termination have distinct likelihoods.

Each control $u _ { t , j } = ( \Delta x , \Delta y , \Delta z , \Delta \psi )$ is a body-frame increment of translation and yaw. A two-layer DiT is trained by rectified flow [17]: with expert chunk $U ^ { \star }$ , noise $\epsilon ,$ and $X _ { \tau } = ( 1 - \tau ) \epsilon + \tau U ^ { \star }$

$$
\mathcal { L } _ { \mathrm { f l o w } } = \mathbb { E } \big [ \| v _ { \theta } ( X _ { \tau } , \tau , c ) - ( U ^ { \star } - \epsilon ) \| _ { 2 } ^ { 2 } \big ] ,\tag{1}
$$

where c is the fused conditioning. Six Euler steps generate a chunk at inference. BC trains the query banks, fusion layers, controller, and STOP head on 849,106 valid decision windows from the UE, 3DGS, and AirSim domains of OpenFly, with positive STOP labels at expert terminal windows; the VLM and visual tower stay frozen. Auxiliary regularizers are listed in Appendix A.

## 3.3. Recovery DAgger at learner-visited anchors

We run the current learner in training scenes with the deployed full-chunk executor and record observations at its anchors. Labels must follow demonstration order, since a nearest-point match can point backward when a route passes near itself. We therefore index the expert route with a monotonic cursor $c _ { t }$ that advances at each collected decision by the number of controls actually executed, rather than searching forward along the reference for a match as in TravelUAV [28]. With learner pose $p _ { t }$ at anchor t and offset $d _ { t } = p _ { t } - p _ { c _ { t } } ^ { \star }$ , the recovery target is

$$
p _ { t , j } ^ { \mathrm { r e c } } = p _ { c _ { t } + j } ^ { \star } + d _ { t } ( 1 - 3 q _ { j } ^ { 2 } + 2 q _ { j } ^ { 3 } ) , \quad q _ { j } = j / H ,\tag{2}
$$

for $j = 0 , \dots , H$ . The offset factor equals one at $\scriptstyle j = 0 ,$ decays with zero slope at both ends, and vanishes at $\scriptstyle { j = H }$ so the target starts at the learner pose and rejoins the future expert segment after one chunk (Fig. 2b). Yaw uses the same decaying offset, wrapped to the shortest signed angle, and consecutive target poses are converted to body-frame controls as in BC. Recovery targets are constructed only during training.

STOP labels follow goal-region entry rather than the cursor. Collection runs with STOP disabled and is truncated at the first anchor inside the 20 m success region, which is labeled STOP with a hover target; every earlier anchor is CONTINUE with its recovery target, and an anchor still outside the region after the cursor has reached the expert endpoint keeps a target toward the goal. Truncation also removes post-arrival anchors whose labels would point backward. Recovery thus adds STOP positives at the learner’s own arrival anchors, complementing the expert terminal windows used by BC. Collection and training alternate for three rounds, the first initialized from BC and later rounds from the preceding recovery checkpoint, each mixing recovery windows 1:1 with BC windows (Appendix A).

![](images/5a201e74c873752af6295474f6a7b17102ba8025b5d80c36cd09fc5251c9304b.jpg)  
Figure 3. Policy backbone shared by the three training stages. A slow path runs the frozen Qwen3.5-4B VLM on the instruction and causal keyframes; a fast path runs only its visual tower on the current frame. Fused with ten state/anchor/action tokens (ego state slow-cache anchor, and the last eight executed actions), the representation drives a rectified-flow controller that emits eight 4-D controls $( \Delta x , \Delta y , \Delta z , \Delta \psi )$ and a separate Bernoulli STOP head; if CONTINUE is sampled, all eight controls are executed.

## 3.4. Outcome optimization with Flow-GRPO

Recovery improves goal-region reachability and reduces no-stop failures, while premature termination remains common (Sec. 4.3). The final stage starts from the recovered full-chunk policy and uses actual task outcomes to refine termination timing together with chunked control. After recovery, closed-loop attempts of the same task differ in route progress, goal entry, overshoot, and termination, so same-task outcomes provide relative advantages without a learned value function. We build on Flow-GRPO [18] and instantiate it for full-chunk aerial control.

Rollout groups. For one instruction and start pose, a frozen copy of the policy produces a group of eight closedloop attempts that share the initial flow latent at each decision depth but sample independent SDE-tail and STOP noise. Six are free attempts of the deployed policy. Free attempts often stop at similar depths, which gives the STOP term little contrast, so the remaining two are training-only STOP-override attempts that override their first $n \in \{ 1 , 2 , 3 \}$ sampled STOP events to CONTINUE and then run freely. Let $\boldsymbol { A } _ { t }$ be the attempts with a decision at depth t and $R _ { i , t } ^ { \mathrm { g o } }$ their return-to-go.

The advantage is

$$
A _ { i , t } = \frac { R _ { i , t } ^ { \mathrm { g o } } - \mu _ { t } } { \operatorname* { m a x } ( s _ { t } , s _ { \mathrm { f l o o r } } ) } , \quad i \in \mathcal { A } _ { t } ,\tag{3}
$$

with $\mu _ { t } , s _ { t }$ the mean and standard deviation over $\boldsymbol { A } _ { t }$ and $s _ { \mathrm { f l o o r } } = 0 . 5$ . Normalizing within a task group and depth controls for route difficulty and reward scale. Override outcomes enter these statistics, but their decisions do not enter the policy-gradient term below.

Objective. Let $\pi _ { \theta } ^ { s }$ be the density of stochastic denoising step s and $\pi _ { \theta } ^ { z }$ the Bernoulli probability of the sampled STOP/CONTINUE action under the training-time STOP temperature 1.5. Each rollout batch is used for one KLregularized policy-gradient update over free-attempt decisions,

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { a c t o r } } = - \mathbb { E } \Big [ A _ { t } \Big ( \sum _ { s = 3 } ^ { 5 } \log \pi _ { \theta } ^ { s } + \log \pi _ { \theta } ^ { z } \Big ) \Big ] } \\ & { ~ + \beta ( D _ { \mathrm { K L } } ^ { u } + D _ { \mathrm { K L } } ^ { z } ) , } \end{array}\tag{4}
$$

where the control term is applied to decisions that execute a chunk, the STOP term to legal decisions (STOP is legal from the second decision on), and both KL terms regularize toward the frozen initial policy with $\beta ~ = ~ 0 . 0 1$ . At the rollout policy, its gradient equals that of the clipped Flow-GRPO surrogate. The override attempts additionally supply outcome-verified stopping labels: a legal decision inside the region is a positive, and an overridden STOP outside the region is a negative only if the continued attempt later reaches the region. A margin-ranking and BCE loss $\mathcal { L } _ { \mathrm { r a n k } }$ on same-depth pairs completes the stage objective $\mathcal { L } _ { \mathrm { a c t o r } } +$ $0 . 5 \mathcal { L } _ { \mathrm { r a n k } }$ (Appendix A).

Reward. The reward is dominated by the task outcome:

$$
\begin{array} { r l } & { R = R _ { \mathrm { t a s k } } + 2 \mathbb { I } [ \mathrm { f i r s t ~ e n t r y } ] - 1 . 5 \mathbb { I } [ \mathrm { l e a v e ~ a f t e r ~ e n t r y } ] } \\ & { ~ + R _ { \mathrm { p r o g r e s s } } + R _ { \mathrm { c o n t r o l } } - \kappa \mathbb { I } [ \mathrm { u n s u c c e s s f u l } ] . } \end{array}\tag{5}
$$

An episode is unsuccessful iff its final position lies outside the success region. A STOP inside the region receives $R _ { \mathrm { t a s k } } = + 2 0$ , and an unsuccessful episode that ends at a non-STOP terminal condition receives $- 1 0 . \ R _ { \mathrm { p r o g r e s s } }$ gives capped credit for approaching the goal, $R _ { \mathrm { c o n t r o l } }$ penalizes invalid, misdirected, or abrupt controls, and the constant shift κ keeps every unsuccessful return negative (Appendix A).

Stochastic chunk transitions. We introduce stochastic denoising transitions during training to obtain tractable conditional likelihoods. As in Flow-GRPO, the last three of six Euler denoising steps, $s \in \{ 3 , 4 , 5 \}$ , are stochastic actions of an augmented MDP whose state is the partial chunk and its conditioning [3, 18]. We place the exploration noise in a low-frequency temporal basis: with $\Phi \doteq \mathbb { R } ^ { H \times M }$ holding $M = 4$ orthonormal cosine modes over the $H = 8$ waypoints and Ψ its orthonormal complement, each stochastic step samples

$$
\begin{array} { r l } & { \boldsymbol { x } ^ { s + 1 } = \boldsymbol { \mu } _ { \boldsymbol { \theta } } ^ { s } + \nu _ { s } ( \Phi \boldsymbol { \epsilon } _ { s } + \rho \boldsymbol { \Psi } \boldsymbol { \xi } _ { s } ) , } \\ & { \quad C _ { s } = \nu _ { s } ^ { 2 } ( \Phi \Phi ^ { \top } + \rho ^ { 2 } \boldsymbol { \Psi } \boldsymbol { \Psi } ^ { \top } ) \otimes \boldsymbol { I } _ { D } , } \end{array}\tag{6}
$$

with $\epsilon _ { s } , \xi _ { s }$ standard Gaussian, $D = 4$ control dimensions, and $\rho = 0 . 1$ . Low-frequency variation dominates exploration, so sampled chunks remain smooth, while the small complement noise keeps $C _ { s }$ full rank. Because $C _ { s }$ is diagonal in the (Φ, Ψ) basis, each transition has a closed-form log-density over the complete $H \times D$ chunk; the restoring mean $\mu _ { \theta } ^ { s }$ , noise schedule $\nu _ { s }$ , and density are given in Appendix B. Deployment keeps the deterministic six-step solver.

## 4. Experiments

## 4.1. Setup

We evaluate on the DreamFly-processed OpenFly splits [9, 11]: test-seen has 1,392 trajectories from UE BigCity and six AirSim cities, and test-unseen has 404 trajectories from the held-out UE SmallCity. Following the benchmark, an episode succeeds if its final position lies within 20 m of the goal. Here k is the number of controls executed per fusedpolicy decision; full-chunk execution is $k { = } 8$

Evaluating a stochastic STOP policy. At each legal decision, X-FLY samples $z _ { t } \sim$ Bernoulli(p<sub>t</sub>) with $p _ { t } = \sigma ( \ell _ { t } )$ from the STOP logit $\ell _ { t } ,$ i.e. at unit temperature, the same temperature at which BC and Recovery DAgger train the STOP head. Flow-GRPO samples STOP at temperature 1.5 only during training rollouts for exploration (Appendix A); all reported X-FLY results use $T { = } 1$ We record each route once with STOP disabled and analytically marginalize the Bernoulli STOP decisions conditional on this control rollout. The probability of first stopping at decision t is p<sub>t</sub> $\textstyle \prod _ { j < t } ( 1 - p _ { j } )$ , and the residual $\Pi _ { j } ( 1 - p _ { j } )$ is assigned to the full rollout. SR, OSR, SPL, and navigation error (NE) are expectations under this first-stop distribution, each computed on the executed prefix up to the stop. Every X-FLY row uses this rule; prior methods are reported with their native stopping policies. The same recordings also give two diagnostics: anchor reach (some decision anchor lies inside the region) and dense reach (some executed waypoint does).

Amortized inference cost. We profile the deployed policy on an NVIDIA RTX PRO 5000 72GB GPU. The slow VLM path refreshes once per 16 executed waypoints at 0.320 s per refresh; each fresh-frame visual-tower call costs 0.141 s and each fusion/control/STOP call 0.034 s. Over a 16-waypoint window, $k = 1 \mathrm { c o s t s } 0 . 3 2 0 + 1 6 ( 0 . 1 4 1 + 0 . 0 3 4 ) = 3 . 1 2 \mathrm { s }$ and $k = 8$ costs $0 . 3 2 0 + 2 ( 0 . 1 4 1 + 0 . 0 3 4 ) = 0 . 6 7 \mathrm { s } ,$ i.e. 195 vs. 42 ms per executed waypoint (4.66×).

## 4.2. Comparison with prior work

With eight controls per decision, X-FLY reaches 74.18% SR, 66.34% SPL, and 17.77 m NE on test-seen, and 51.73% SR, 47.03% SPL, and 29.17 m NE on test-unseen, improving all four metrics over DreamFly on both splits (Table 1). Within the same backbone and execution setting, Recovery DAgger raises SR from 16.45% to 63.46% on test-seen and from 18.43% to 40.63% on test-unseen, and Flow-GRPO adds a further 10.72 and 11.10 points. On test-unseen, single-step BC on our backbone trails DreamFly (22.49% vs. 29.46%), and the two closed-loop stages provide the advantage. Flow-GRPO updates draw only BigCity training tasks, yet its gain carries over to the held-out SmallCity.

## 4.3. Failure attribution

SR does not reveal whether an episode failed to reach the goal or failed to terminate there. We therefore assign each first-stop outcome of a recorded control rollout to one of two success types or to exactly one failure (Fig. 4), weighted by its first-stop probability. A navigation failure means that the STOP-disabled control rollout contains no in-region anchor; termination failures (early, late, or no stop) are categorized on rollouts that do. This analysis separates control-rollout reachability from termination outcomes.

Table 1. OpenFly test-seen and test-unseen results under the DreamFly-processed benchmark. NE is in meters, other metrics in %. Prior results are as reported with their native stopping policies; every X-FLY row uses the same 128-control cap and Bernoulli STOP policy, evaluated over its first-stop distribution (Sec. 4.1). k: controls executed per fused-policy decision. The two Recovery DAgger rows evaluate the same checkpoint, and its k=8 operating point initializes Flow-GRPO.
<table><tr><td rowspan="2">Method</td><td colspan="4">test-seen</td><td colspan="4">test-unseen</td></tr><tr><td>SR↑</td><td>OSR↑</td><td>SPL↑</td><td>NE↓</td><td>SR↑</td><td>OSR↑</td><td>SPL ↑</td><td>NE↓</td></tr><tr><td>Random</td><td>13.51</td><td>18.75</td><td>9.72</td><td>65.67</td><td>15.35</td><td>23.76</td><td>11.31</td><td>59.99</td></tr><tr><td>Action Sampling</td><td>15.95</td><td>26.51</td><td>13.67</td><td>62.78</td><td>20.54</td><td>32.67</td><td>17.22</td><td>55.27</td></tr><tr><td>Seq2Seq</td><td>24.35</td><td>61.93</td><td>19.35</td><td>54.44</td><td>26.49</td><td>61.88</td><td>19.62</td><td>47.69</td></tr><tr><td>CMA</td><td>7.97</td><td>69.32</td><td>6.26</td><td>313.03</td><td>5.69</td><td>73.02</td><td>3.92</td><td>230.05</td></tr><tr><td>AerialVLN [19]</td><td>16.52</td><td>65.66</td><td>14.63</td><td>176.29</td><td>9.65</td><td>68.07</td><td>7.93</td><td>161.19</td></tr><tr><td>OpenFly-Agent [11]</td><td>22.63</td><td>52.73</td><td>20.42</td><td>122.89</td><td>14.11</td><td>62.38</td><td>12.49</td><td>163.87</td></tr><tr><td>DreamFly [9], k=1</td><td>32.04</td><td>46.77</td><td>28.22</td><td>44.87</td><td>29.46</td><td>46.78</td><td>23.54</td><td>45.29</td></tr><tr><td>X-FLY BC, k=1</td><td>46.57</td><td>52.85</td><td>39.92</td><td>37.28</td><td>22.49</td><td>23.46</td><td>21.00</td><td>58.61</td></tr><tr><td>X-FLY BC, k=8</td><td>16.45</td><td>67.30</td><td>13.54</td><td>37.87</td><td>18.43</td><td>43.14</td><td>16.18</td><td>48.12</td></tr><tr><td>+ Recovery DAgger, k=1</td><td>68.88</td><td>75.45</td><td>56.84</td><td>20.12</td><td>44.27</td><td>47.64</td><td>39.96</td><td>36.43</td></tr><tr><td>+ Recovery DAgger, k=8</td><td>63.46</td><td>72.87</td><td>54.54</td><td>21.49</td><td>40.63</td><td>52.71</td><td>36.46</td><td>35.16</td></tr><tr><td>+ Flow-GRPO (X-FLY, full), k=8</td><td>74.18</td><td>84.07</td><td>66.34</td><td>17.77</td><td>51.73</td><td>68.07</td><td>47.03</td><td>29.17</td></tr></table>

On test-seen, full-chunk BC mainly loses termination. On test-seen, BC fails to reach the region about as often at k = 8 as at k = 1 (30.77% vs. 28.02%), yet no-stop failures account for 47.78% of its full-chunk outcomes, most of the SR drop from 46.57% to 16.45%. On test-unseen, fullchunk BC degrades on both sides: navigation failures rise from 44.80% to 53.22% and no-stop failures from 0.33% to 23.88%. Single-step BC instead mostly stops early (19.13% seen, 31.74% unseen). The STOP head, whose positive labels come only from expert terminal windows, does fire at single-step anchors, often too early, but rarely at the anchors produced by full-chunk execution.

Recovery changes both reaching and stopping failures. At k = 8, Recovery DAgger reduces navigation failures from 30.77% to 7.14% on test-seen and from 53.22% to 10.40% on test-unseen, and no-stop failures fall to 0.00% on both splits, consistent with STOP now being labeled at the learner’s arrival anchors. Its unseen navigation failures are lower at k = 8 than at k = 1 (10.40% vs. 18.56%), consistent with labels being collected under the deployed execution. On test-seen, recovery narrows the k = 1/k = 8 SR difference from 30.12 to 5.42 points; on test-unseen the difference was already small (4.06 vs. 3.64 points), since there BC failed mostly at navigation under both horizons. Recovery raises unseen SR by similar amounts at both horizons (+21.78 at k=1, +22.20 at k=8), so on this split much of its gain is general correction at learner-visited states; the effect specific to full-chunk execution is concentrated on test-seen, where it removes the no-stop failures. The remaining full-chunk failures are dominated by early termination (21.21% seen, 37.18% unseen).

Flow-GRPO reduces premature stops. Relative to the k = 8 recovery policy, Flow-GRPO keeps navigation failures at the recovered level (7.14→6.59% seen, 10.40→11.39% unseen) and roughly halves early termination (21.21→9.34% seen, 37.18→21.04% unseen), adding 10.72 and 11.10 SR points. The gain of this stage thus lies in termination timing rather than in control-rollout reachability. Its OSR gains (+11.20 seen, +15.36 unseen) largely follow from the same change, since fewer prefixes are truncated before the region. Late termination increases (8.19→9.89% seen, 11.80→15.84% unseen), consistent with Flow-GRPO shifting stops later. Termination dominates the residual failures, with late stops slightly more common than early stops on test-seen and early stops still more frequent on test-unseen. With STOP disabled, the final policy visits an in-region anchor in 93.41% of test-seen and 88.61% of testunseen episodes, and on test-unseen dense reach exceeds anchor reach by only 0.75 points, so the remaining gap to SR lies mainly in stop timing at anchors the policy does visit.

Successes end with STOP after recovery. The benchmark also counts a rollout that ends inside the region without STOP as a success (column “no STOP” in Fig. 4). For fullchunk BC, such successes make up 69.0% of SR on test-seen and 79.4% on test-unseen. After Recovery DAgger, at both horizons, and after Flow-GRPO, their share is 0.00% on both splits, so every success of these policies ends with an explicit in-region STOP.

![](images/f309298dc21a8ec808b2ca65a472960ac3b41c3fb254a919f4510e88c1033f62.jpg)  
Figure 4. Outcome attribution on the same recorded rollouts. Each row splits a policy’s episodes into two success and four failure types and sums to 100; values are expectations under the first-stop distribution (Sec. 4.1), in %. Color encodes magnitude within the success (blue) and failure (orange) groups. STOP / no STOP: success ending with an explicit in-region STOP / inside the region at the control budget. Nav.: no in-region anchor even with STOP disabled; Early/Late: STOP outside the region before/after the first in-region anchor; No-stop: reaches the region, never stops, and ends outside it. Rule in Appendix C; two-decimal values in Table 3.

Effect of STOP opportunity count. Full-chunk execution also offers STOP only once every eight controls. To separate this effect, we replay the saved k=1 BC rollouts unchanged but permit STOP only every eight controls. SR becomes 32.10% on test-seen and 32.56% on test-unseen, compared with 46.57% and 22.49% for native k=1 BC and 16.45% and 18.43% for native k=8 BC. The effect of opportunity count differs between splits, and in both it leaves native full-chunk BC well below the replay, which keeps the k=1 visited states.

## 4.4. Flow-GRPO training dynamics

We track the Flow-GRPO run behind Table 1, started from the Recovery DAgger policy, together with the same procedure started from the BC policy. Both runs use a fixed source-validation panel of 384 BigCity routes excluded from Flow-GRPO updates, evaluated every 20 updates with fixed seeds and one sampled STOP rollout per route.

Improvement from recovered behavior. From the Recovery DAgger policy (Fig. 5, blue), SR rises from 53.4% to 65.1% at update 3000, OSR from 59.6% to 75.0%, and NE falls from 24.7 m to 20.3 m. Improvement is fastest early, with SR first exceeding 60% at update 260, and the three metrics improve together across 151 evaluations without a late reversal.

Initialization comparison. From the BC policy (orange), SR stays between 19.0% and 25.8% over 680 updates, while OSR declines from 43.0% to 28.4% and NE grows from 41.5 m to 45.6 m. Relative to its own initialization over the same 680 updates, the recovery-initialized run gains 4.1 SR and 4.7 OSR points and lowers NE by 2.1 m, whereas the BC-initialized run gains 1.0 SR point, loses 6.1 OSR points, and raises NE by 1.9 m. These matched-window results support recovery as an effective initialization for the tested post-training recipe.

## 5. Limitations

Termination timing remains the main residual bottleneck: the final policy visits an in-region anchor in most episodes but converts fewer of them into success. Each training configuration uses one training seed, so our results do not measure training-seed variance. OpenFly does not capture real aircraft dynamics, sensing noise, or operational safety constraints, and transfer to physical platforms requires separate validation.

## 6. Conclusion

We presented X-FLY, a staged training framework for fullchunk aerial vision-language navigation. Recovery DAgger supplies learner-visited decision anchors with futurereconnecting chunks and arrival STOP labels, and Flow-GRPO jointly refines continuous chunked control and Bernoulli termination from same-task closed-loop outcomes. With eight controls per decision, X-FLY reaches 74.18% SR on OpenFly test-seen and 51.73% on test-unseen at 4.7× less amortized model inference time per executed waypoint than single-step execution. Failure attribution identifies stop timing as the principal remaining challenge.

## References

[1] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sunderhauf, Ian Reid, Stephen Gould, and¨ Anton van den Hengel. Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3674–3683, 2018.

![](images/1690b36244558e767544deb6c18ffc21697c0bd75b99a2e3c2bbd19b487f5160.jpg)

![](images/1809bf4e4f0e0dcb3166892f8bb61c9b5e907821091f9822890a4ecd32ebcbe9.jpg)

![](images/05ca496ab8621da22a8bc06bc0c3cffe811837332926617757a1cead39fdb3ed.jpg)  
Figure 5. Flow-GRPO from recovered vs. BC initialization. Source-validation (a) SR, (b) OSR, and (c) NE on a fixed 384-route BigCity panel excluded from Flow-GRPO updates, evaluated every 20 updates; dashed lines mark each initialization. The BC-initialized run was stopped at 680 updates as its OSR and NE kept degrading. ∆: mean change of each run relative to its own initialization over the 35 matched evaluations (shaded window).

[2] Mayank Bansal, Alex Krizhevsky, and Abhijit Ogale. ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst. In Proceedings of Robotics: Science and Systems (RSS), 2019.

[3] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training Diffusion Models with Reinforcement Learning. In International Conference on Learning Representations (ICLR), pages 4965–4987, 2024.

[4] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π<sub>0</sub>: A Vision-Language-Action Flow Model for General Robot Control. In Proceedings of Robotics: Science and Systems (RSS), 2025.

[5] Kevin Black, Manuel Galliker, and Sergey Levine. Real-Time Execution of Action Chunking Flow Policies. In Advances in Neural Information Processing Systems, pages 33383–33407, 2025.

[6] Shizhe Chen, Pierre-Louis Guhur, Cordelia Schmid, and Ivan Laptev. History Aware Multimodal Transformer for Visionand-Language Navigation. In Advances in Neural Information Processing Systems, pages 5834–5847, 2021.

[7] Shizhe Chen, Pierre-Louis Guhur, Makarand Tapaswi, Cordelia Schmid, and Ivan Laptev. Think Global, Act Local: Dual-Scale Graph Transformer for Vision-and-Language Navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16537–16547, 2022.

[8] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. The International Journal ofRobotics Research, 44(10–11):1684–1704, 2025.

[9] Yan Deng and Fei Xu. DreamFly: Causal Memory and Receding-Horizon Diffusion Planning for Aerial Vision-Language Navigation. arXiv preprint arXiv:2608.12308, 2026.

[10] Daniel Fried, Ronghang Hu, Volkan Cirik, Anna Rohrbach,

Jacob Andreas, Louis-Philippe Morency, Taylor Berg-Kirkpatrick, Kate Saenko, Dan Klein, and Trevor Darrell. Speaker-Follower Models for Vision-and-Language Navigation. In Advances in Neural Information Processing Systems, 2018.

[11] Yunpeng Gao, Chenhui Li, Zhongrui You, Junli Liu, Zhen Li, Pengan Chen, Qizhi Chen, Zhonghan Tang, Liansheng Wang, Penghui Yang, Yiwen Tang, Yuhang Tang, Shuai Liang, Songyi Zhu, Ziqin Xiong, Yifei Su, Xinyi Ye, Jianan Li, Yan Ding, Dong Wang, Xuelong Li, Zhigang Wang, and Bin Zhao. OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation. In International Conference on Learning Representations (ICLR), pages 101858–101871, 2026.

[12] Yicong Hong, Qi Wu, Yuankai Qi, Cristian Rodriguez-Opazo, and Stephen Gould. VLN BERT: A Recurrent Vision-and-Language BERT for Navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1643–1653, 2021.

[13] Michael Kelly, Chelsea Sidrane, Katherine Driggs-Campbell, and Mykel J. Kochenderfer. HG-DAgger: Interactive Imitation Learning with Human Experts. In 2019 International Conference on Robotics and Automation (ICRA), pages 8077– 8083, 2019.

[14] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments. In Computer Vision – ECCV 2020, pages 104–120. Springer, 2020.

[15] Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg. DART: Noise Injection for Robust Imitation Learning. In Proceedings of the 1st Annual Conference on Robot Learning (CoRL), pages 143–156. PMLR, 2017.

[16] Yun Li, Jiachen Gong, Simon Thompson, Ehsan Javanmardi, Qunli Zhang, Zifan Zeng, Shiming Liu, Peng Wang, Zixuan Guo, and Manabu Tsukada. Think at 5 Hz, Act at 20 Hz: Asynchronous Fast-Slow Vision-Language-Action Inference for Closed-Loop Driving. arXiv preprint arXiv:2607.15621, 2026.

[17] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow Matching for Generative

Modeling. In International Conference on Learning Representations (ICLR), 2023.

[18] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-GRPO: Training Flow Matching Models via Online RL. In Advances in Neural Information Processing Systems, pages 40783–40818, 2025.

[19] Shubo Liu, Hongsheng Zhang, Yuankai Qi, Peng Wang, Yanning Zhang, and Qi Wu. AerialVLN: Vision-and-Language Navigation for UAVs. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15384–15394, 2023.

[20] Xuchen Liu, Jiawei Huang, Shihao Xia, Bingxi Liu, Jinqiang Cui, and Jiankun Yang. ImagineUAV: Aerial Vision-Language Navigation via World-Action Modeling and Kinodynamic Planning. arXiv preprint arXiv:2606.01205, 2026.

[21] Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Maximilian Du, and Chelsea Finn. Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling. In International Conference on Learning Representations (ICLR), pages 4594–4627, 2025.

[22] Chih-Yao Ma, Jiasen Lu, Zuxuan Wu, Ghassan AlRegib, Zsolt Kira, Richard Socher, and Caiming Xiong. Self-Monitoring Navigation Agent via Auxiliary Progress Estimation. In International Conference on Learning Representations (ICLR), 2019.

[23] Allen Z. Ren, Justin Lidard, Lars L. Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion Policy Policy Optimization. In International Conference on Learning Representations (ICLR), pages 77288–77329, 2025.

[24] Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A Re-´ duction of Imitation Learning and Structured Prediction to No-Regret Online Learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS), pages 627–635. PMLR, 2011.

[25] Hao Tan, Licheng Yu, and Mohit Bansal. Learning to Navigate Unseen Environments: Back Translation with Environmental Dropout. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2610–2621. Association for Computational Linguistics, 2019.

[26] Boxiong Wang, Hui Kang, Geng Sun, Jiahui Li, Chao Yu, and Daxin Tian. RefineFly: Failure-Aware Post-Training for Aerial Vision-Language Navigation. arXiv preprint arXiv:2608.09467, 2026.

[27] Xin Wang, Qiuyuan Huang, Asli Celikyilmaz, Jianfeng Gao, Dinghan Shen, Yuan-Fang Wang, William Yang Wang, and Lei Zhang. Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6629–6638, 2019.

[28] Xiangyu Wang, Donglin Yang, Ziqin Wang, Hohin Kwan, Jinyu Chen, Wenjun Wu, Hongsheng Li, Yue Liao, and Si Liu. Towards Realistic UAV Vision-Language Navigation:

Platform, Benchmark, and Methodology. In International Conference on Learning Representations (ICLR), pages 7292– 7310, 2025.

[29] Jiannan Xiang, Xin Wang, and William Yang Wang. Learning to Stop: A Simple yet Effective Approach to Urban Vision-Language Navigation. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 699–707. Association for Computational Linguistics, 2020.

[30] Siyu Xu, Yunke Wang, Chenghao Xia, Dihao Zhu, Tao Huang, and Chang Xu. VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching. In Advances in Neural Information Processing Systems, pages 164448–164473, 2025.

[31] Tonghe Zhang, Chao Yu, Sichang Su, and Yu Wang. Rein-Flow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning. In Advances in Neural Information Processing Systems, pages 106282–106319, 2025.

[32] Weichen Zhang, Chen Gao, Shiquan Yu, Ruiying Peng, Baining Zhao, Qian Zhang, Jinqiang Cui, Xinlei Chen, and Yong Li. CityNavAgent: Aerial Vision-and-Language Navigation with Hierarchical Semantic Planning and Global Memory. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 31292–31309. Association for Computational Linguistics, 2025.

[33] Baining Zhao, Jiacheng Xu, Weicheng Feng, Xin Zhang, Zhaolu Wang, Haoyang Wang, Shilong Ji, Ziyou Wang, Jianjie Fang, Zhiheng Zheng, Weichen Zhang, Yu Shang, Wei Wu, Chen Gao, Xinlei Chen, and Yong Li. WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation. arXiv preprint arXiv:2605.15964, 2026.

[34] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Proceedings of Robotics: Science and Systems (RSS), 2023.

[35] Xueke Zhu, Qingyan Meng, Liutao Yu, Wei Zhang, Zhengyu Ma, Huihui Zhou, and Yonghong Tian. FSD-VLN: Fast-Slow Dual-System Modeling for Aerial Long-Horizon Vision-Language Navigation. arXiv preprint arXiv:2607.08359, 2026.

## A. Implementation Details

## A.1. Training and execution pseudocode

## Recovery-label construction.

1. Roll out the learner with the deployed full-chunk executor; after each CONTINUE, advance the monotonic expert cursor by the number of controls actually executed.

2. At anchor $s _ { t } ,$ compute the learner-to-goal distance. If this is the first anchor within 20 m, emit STOP and an all-zero hover target, then end collection for that route.

3. Otherwise emit CONTINUE. Clamp an overrun cursor to the final expert pose and construct the H-step target of Eq. 2; an outside-region cursor overrun therefore remains a CONTINUE example pointing toward the goal.

4. Reduce the initial yaw offset to the shortest signed angle before interpolation and wrap every relative yaw target. Mix the retained recovery windows 1:1 with BC windows and train the DAgger stage.

## Flow-GRPO update.

1. For each task, sample eight closed-loop attempts with shared initial flow noise: six free attempts and two STOPoverride attempts, n ∼ Uniform{1, 2, 3}.

2. At each legal anchor, sample a flow chunk and Bernoulli STOP. A STOP-override attempt overrides its first n sampled STOP events to CONTINUE and then runs freely.

3. Compute complete trajectory outcomes and depthaligned group advantages. Override outcomes enter the group statistics, but override decisions are excluded from both actor surrogates.

4. From STOP-override trajectories, form same-depth outcome-verified pairs: an overridden outside-region STOP is negative only if continuation later reaches a legal in-region anchor; a legal in-region decision is positive. Apply the rank+BCE loss and update the joint trainable policy.

## A.2. Policy architecture and design choices

Slow path. A frozen Qwen3.5-4B receives up to four causal keyframes, oldest first, followed by the instruction. Keyframes refresh on a fixed timer every 16 executed control steps rather than being triggered by predicted actions; their contents still reflect the executed trajectory. The VLM decoder re-runs only when the cache changes. We read the 2560-wide hidden states of the fourth-last decoder layer. Two learned query banks cross-attend to them: an observation bank restricted to image-token positions and a navigation bank restricted to the instruction span. Each bank holds 48 tokens of width 768 and applies one 12-head cross-attention.

Fast path. Only the visual tower runs on the current 448×448 frame. Its last-block patch tokens (784×1024) are compressed by a third query bank into 48 current-frame tokens. Running the decoder at every decision would dominate inference cost, while refreshing only at keyframes would leave the policy without the current view between refreshes; the split keeps the current frame at visual-tower cost.

Table 2. Data flow of the three training stages. Every stage trains the same modules; the VLM and visual tower stay frozen.
<table><tr><td>Stage</td><td>Init. / data</td><td>Labels or signal</td><td>Length / selec- tion</td><td></td></tr><tr><td>BC</td><td>(UE, 3DGS, Air- Sim)</td><td>scratch; 849,106 expert chunks; STOP 10 epochs expert windows at terminal windows</td><td></td><td></td></tr><tr><td>Recovery DAgger</td><td>BC, then previous- recovery (3 rounds); learner in-region anchor rollouts at k=8, STOP disabled, mixed 1:1 with BC</td><td>round checkpoint (Eq. 2); STOP at first round; lowest val.</td><td>chunks 3 epochs per control-speed error</td><td></td></tr><tr><td>Flow- GRPO</td><td>training tasks</td><td>Recovery DAgger group-relative ad- 3000 updates; at k=8; BigCity vantages (Eq. 3); val. panel score outcome-verified (App. A) STOP pairs</td><td></td><td></td></tr></table>

State and history. The 10-D ego state contains the laststep displacement in the body frame, sine and cosine of the last-step yaw change, speed, yaw rate, and the displacement from the start pose in the start frame. All entries are relative; no goal or global coordinate is provided. An anchor token encodes the pose change since the slow cache was formed and the normalized cache age, and a learned age embedding added to the observation tokens lets the policy discount or re-interpret stale keyframes. History tokens are built from the executed pose history; predicted but unexecuted controls are excluded, while executed actions affect both subsequent observations and history. Empty history slots are masked in attention, and BC drops history slots with probability 0.15.

Fusion. The main sequence concatenates five learned task tokens (action, waypoint, STOP, direction, chunk), the observation and current-frame tokens, the state and anchor tokens, and the history tokens, each with a type embedding. Each of the three fusion blocks applies pre-LN self-attention over this sequence, ungated cross-attention to the navigation tokens, and a feed-forward layer, with width 768, 12 heads, and dropout 0.1; a final LayerNorm follows.

Controller and STOP head. The DiT conditions its AdaLN-Zero modulation only on flow time and reads the fused tokens through ungated cross-attention; its output projection is zero-initialized. The controller is small (2 layers, width 256, 8 heads, 3.9M parameters) and is integrated with six Euler steps. The STOP head is a two-layer MLP on the fused STOP token. It is computed before and independently of the controller, so termination does not condition on the sampled chunk and STOP and control keep separate likelihoods. The trainable modules total about 45M parameters. The VLM and visual tower remain frozen across all three training stages.

## A.3. Behavior cloning

BC uses AdamW (learning rate $1 0 ^ { - 4 }$ , weight decay 0.01) with 500 warmup steps and cosine decay, an effective batch of 128 on 8 GPUs, 10 epochs, bf16 autocast, and gradient clipping at 1. Besides the rectified-flow loss, the objective includes STOP binary cross-entropy (weight 0.1), an endpoint loss on the final position of the one-step chunk estimate (0.1), control smoothness and speed consistency (0.1 each), yaw continuity (0.05), and an auxiliary label-smoothed discreteaction cross-entropy on the fused action token (0.2), used only during BC. STOP positives are sampled annotation steps whose dense expert action is STOP. Expert chunks that extend past the trajectory end hold the final pose with zero motion. The three domains are sampled in their natural proportions.

## A.4. Recovery DAgger

Collection, RL, and evaluation share the full-chunk executor; the route cursor advances by the number of controls actually executed. Recovery collection runs with STOP disabled and is truncated at the first in-region anchor; free RL rollouts sample the STOP policy, and evaluation uses its first-stop distribution (Sec. 4.1); both follow the benchmark termination rules. Label construction then applies a geometric priority rule. The prefix is cut at the first learner anchor within 20 m, which receives STOP and an all-zero hover target. If the learner never enters, all retained anchors remain CONTINUE even after the cursor passes the last expert pose; the clamped reference stays at the goal and the recovery target continues toward it. The collected set contains 8,404 recovery STOP positives. Only successfully rendered, finite anchors are retained. Before interpolation, the yaw offset is reduced modulo 2π to the shortest signed angle; each relative yaw control is wrapped again. There are three collection-and-training rounds: the first starts from BC, and each later round warm-starts from the previous best checkpoint with a fresh optimizer. Each round trains for 3 epochs at learning rate $3 \times 1 0 ^ { - 5 }$ with 50 warmup steps, cosine decay, and batch 128; the auxiliary action loss is switched off. The 1:1 mixture pairs the recovery labels with a seeded random subsample of BC windows of equal size, and the checkpoint with the lowest validation control-speed error is kept.

## A.5. Flow-GRPO

Recipe. For late-depth advantage normalization, sparse depths with fewer than four active attempts pool the next adjacent depth; a depth without a valid four-sample comparison receives zero advantage. Route difficulty is computed from static geometry as a weighted sum of percentile-ranked route length (0.50), accumulated turning (0.35), and vertical travel (0.15). Each training panel mixes 50% natural-distribution coverage, 20% curriculum samples, and 30% current-policyprioritized samples. The curriculum first draws from the easiest 45% of routes and expands to 70% after 108 updates, and prioritized sampling rises to 50% over all routes after update 324; priority scores favor tasks with informative reward spread and mixed success while penalizing repeated exposure. We use six flow integration steps, linear schedule endpoints $\sigma _ { \mathrm { m i n } } = 0 . 0 5$ and $\sigma _ { \mathrm { m a x } } = 0 . 2 0$ , three stochastic tail steps, four temporal exploration modes, complementnoise ratio $\rho = 0 . 1$ , and one optimizer step per rollout batch. The gradient norm is clipped at 1, and both reference-KL weights are 0.01. Goal progress is clipped to 0.4 per decision and 2.5 per episode; invalid, direction, and smoothness penalties have episode caps of 2, 2, and 4. The largest positive shaping an unsuccessful episode can collect is the first-entry bonus plus the progress cap, $2 + 2 . 5 = 4 . 5 .$ so we set the failure shift to $\kappa = 4 . 5 1$ . An early STOP outside the region receives no penalty beyond this shift. Training runs for 3000 updates.

Sampling. Each update draws 48 BigCity training tasks with goal distance at least 20 m and at least 10 expert steps, run as six groups on each of 8 GPUs. Within a group, attempts share the initial flow latent at each decision depth, seeded by update, task, and depth; SDE-tail noise and STOP draws use per-attempt seeds. STOP is legal from the second decision on and is sampled from $\sigma ( \ell / 1 . 5 )$ during training rollouts to widen exploration of stop timing. This temperature applies only to Flow-GRPO training; BC and Recovery DAgger train the STOP head at unit temperature, and all reported evaluations use σ(ℓ).

Counterfactual attempts and credit. Two attempts per group are STOP-override attempts with n drawn uniformly from {1, 2, 3}: each of their first n sampled STOPs is overridden to CONTINUE, after which the attempt runs freely. They are complete rollouts from the shared start. Their returns enter the depth-wise statistics of Eq. 3, but none of their decisions enters either term of Eq. 4, so the policy-gradient terms come from the six free attempts. Override returns do not depend on a free attempt’s own SDE-tail and STOP noise, so including them in the depth-wise mean does not bias that attempt’s gradient, and including its own return only rescales its centered advantage by $1 - 1 / | \mathcal { A } _ { t } |$ . The update is nevertheless not an unbiased gradient estimate: the standard deviation, a population standard deviation over all attempts at a depth, depends on the attempt’s own return, and pooling by depth conditions on which attempts survive to that depth.

Outcome terms (task, first entry, leaving, failure shift) are added undiscounted to every decision’s return-to-go, and shaping terms are discounted by 0.99 per executed waypoint.

Reward terms. With $\phi ( d ) \ = \ \operatorname* { m a x } ( d \mathrm { ~ - ~ } 2 0 , 0 )$ , goal progress adds $0 . 0 3 ( \phi _ { j - 1 } - \phi _ { j } )$ per executed waypoint before the caps listed above. A direction penalty applies to horizontal steps of at least 0.25 m whose motion deviates more than $6 0 ^ { \circ }$ from the current yaw: 0.08 per meter times a severity that rises linearly from 0 at $6 0 ^ { \circ }$ to 1 at $9 0 °$ and continues linearly beyond. Smoothness costs 0.06 times the relative velocity change and 0.02 times the normalized yawrate change per waypoint, capped at 0.35 per decision. A decision with non-finite waypoints costs 0.8, and a non-finite pose or render error removes the group from the update. First entry is detected at 20 m and leaving at 22 m. An episode that ends inside the region without STOP counts as successful and receives no terminal reward or failure shift.

Objective. The control KL is the exact Gaussian divergence between the current and reference restoring means under the shared covariance of Eq. 6, restricted to executed waypoints and summed over the stochastic steps; the STOP KL is the Bernoulli divergence at legal free decisions. Likelihood and KL terms are normalized by the number of environment decisions in the update. A control term is active only when the decision executes a chunk prefix, and no decision after STOP receives credit. AdamW without weight decay uses learning rate $1 . 5 \times 1 0 ^ { - 6 }$ for the flow controller and $5 \times 1 0 ^ { - 7 }$ for all other trainable modules, including fusion and the STOP head, with one optimizer step per update.

Ranking term. Pairs are formed among STOP-override decisions at the same depth within a GPU’s batch. A positive is a legal decision within 20 m of the goal; a negative is an overridden STOP more than 22 m from the goal whose attempt later reaches a legal in-region decision. At most two pairs are kept per depth. With temperature-scaled logits $\tilde { \ell } = \ell / 1 . 5 .$

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r a n k } } = \mathrm { s o f t p l u s } \big ( 1 - ( \tilde { \ell } ^ { + } - \tilde { \ell } ^ { - } ) \big ) } \\ & { \qquad + \frac { 1 } { 2 } \big [ \mathrm { s o f t p l u s } ( - \tilde { \ell } ^ { + } ) + \mathrm { s o f t p l u s } ( \tilde { \ell } ^ { - } ) \big ] , } \end{array}\tag{7}
$$

averaged over pairs and added to the actor loss with weight 0.5.

Validation and selection. The validation panel holds 384 fixed BigCity training-split routes excluded from the task manifest used for Flow-GRPO updates and is evaluated every 20 updates with a fixed seed and one Bernoulli STOP rollout per route. A checkpoint is scored by the mean of its last three valid evaluations. Candidates whose OSR does not fall more than 2 points below the initial OSR are ranked by SR,

then by a smaller OSR–SR gap, then by lower final distance.   
The test splits are not used for selection.

## A.6. Evaluation

Each decision integrates the flow ODE with six deterministic Euler steps from a fresh Gaussian latent, seeded per route. Each route is recorded once with STOP disabled at the given $k ,$ and the first-stop metrics of Sec. 4.1 are computed from the stored STOP logits at unit temperature, $p _ { t } = \sigma ( \ell _ { t } )$ ; the same recording yields anchor and dense reach and the failure attribution of Appendix C. All X-FLY rows use the same cap of 128 executed controls, independent of $k ,$ together with the same executor, success radius, and termination rule. Each first-stop outcome is scored on its own executed prefix, so a prefix that stops before a later abnormal event keeps its metrics. A trajectory that ends abnormally (collision, render failure, or non-finite state) before any replayable decision contributes zero to SR, OSR, and SPL and remains in the denominator. SPL uses the executed path length up to the stop, and metrics are averaged over all routes. Prior-work rows are their published benchmark results with native stopping policies.

## A.7. Training dynamics

The BC-initialized run in Sec. 4.4 uses this recipe unchanged, with the BC policy as initialization and KL reference. Validation points interrupted by renderer failures were evaluated from the saved checkpoints with the same panel and seeds.

## B. Transition Density

In Eq. $6 , \Delta t = 1 / 6 , \nu _ { s } = \sigma _ { s } \sqrt { \Delta t } ,$ and $\sigma _ { s } = 0 . 2 0 \mathrm { - } 0 . 1 5 s / 6$ The restoring mean uses the score implied by the velocity field,

$$
\begin{array} { r } { \mu _ { \theta } ^ { s } = x ^ { s } + \Delta t v _ { \theta } ( x ^ { s } , t _ { s } , c ) - \frac { 1 } { 2 ( 1 - t _ { s } ) } C _ { s } \big ( x ^ { s } - t _ { s } v _ { \theta } ( x ^ { s } , t _ { s } , c ) \big ) , } \end{array}\tag{8}
$$

so that drift and noise share the same anisotropic covariance, and the small complement noise gives old and updated policies common support. For a stochastic step of Eq. $^ { 6 , }$ let $e _ { s } = x ^ { s + 1 } - \mu _ { \theta } ^ { s } , a _ { s } = \Phi ^ { \top } e _ { s }$ , and $b _ { s } = \Psi ^ { \top } e _ { s }$ . Since $C _ { s }$ is diagonal in the (Φ, Ψ) basis,

$$
\begin{array} { c } { { \log \pi _ { \theta } ^ { s } = - \displaystyle \frac { 1 } { 2 } \sum _ { d = 1 } ^ { D } \Big \{ \displaystyle \sum _ { m = 1 } ^ { M } \big [ a _ { s m d } ^ { 2 } / \nu _ { s } ^ { 2 } + \log ( 2 \pi \nu _ { s } ^ { 2 } ) \big ] } } \\ { { + \displaystyle \sum _ { n = 1 } ^ { H - M } \big [ b _ { s n d } ^ { 2 } / ( \rho ^ { 2 } \nu _ { s } ^ { 2 } ) + \log ( 2 \pi \rho ^ { 2 } \nu _ { s } ^ { 2 } ) \big ] \Big \} . } } \end{array}\tag{9}
$$

This scores all low-frequency and complement coordinates jointly, i.e. the complete $H \times D$ transition, rather than treating waypoints as independent actions. The three deterministic steps carry no density and enter no likelihood term. As in Flow-GRPO, their output $x ^ { 3 }$ is stored as part of the state. With $g _ { \theta } ^ { s } = - ( x ^ { s } - t _ { s } v _ { \theta } ) / ( 1 - t _ { s } )$ , the score of the interpolant $X _ { \tau } = ( 1 - \tau ) \epsilon + \tau U$ under the flow’s own prediction, Eq. 8 reads $\begin{array} { r } { \mu _ { \theta } ^ { s } = x ^ { s } + \Delta t v _ { \theta } + \frac { 1 } { 2 } C _ { s } g _ { \theta } ^ { s } } \end{array}$

Table 3. Outcome attribution (% of episodes) underlying Fig. 4. Labels follow the rule above. The two success columns sum to SR (Table 1), and all six columns of a split sum to 100 up to rounding.
<table><tr><td></td><td colspan="2"></td><td colspan="2">test-seen</td><td></td><td></td><td colspan="2"></td><td colspan="4">test-unseen</td></tr><tr><td>Method</td><td colspan="2">Success</td><td colspan="2"></td><td></td><td>No-stop</td><td colspan="2">Success</td><td>Nav.</td><td>Early</td><td>Late</td><td>No-stop</td></tr><tr><td></td><td>STOP</td><td>no STOP</td><td>Nav.</td><td>Early</td><td>Late</td><td></td><td>STOP</td><td>no STOP</td><td></td><td></td><td></td><td></td></tr><tr><td>BC, k=1 BC, k=8</td><td>44.27</td><td>2.30</td><td>28.02</td><td>19.13</td><td>6.20</td><td>0.07</td><td>21.45</td><td>1.04</td><td>44.80</td><td>31.74</td><td>0.65</td><td>0.33</td></tr><tr><td></td><td>5.10</td><td>11.35</td><td>30.77</td><td>3.85</td><td>1.15</td><td>47.78</td><td>3.80</td><td>14.63</td><td>53.22</td><td>3.98</td><td>0.48</td><td>23.88</td></tr><tr><td>+ Recovery DAgger, k=1</td><td>68.88</td><td>0.00</td><td>7.14</td><td>17.41</td><td>6.57</td><td>0.00</td><td>44.27</td><td>0.00</td><td>18.56</td><td>33.80</td><td>3.37</td><td>0.00</td></tr><tr><td>+ Recovery DAgger, k=8</td><td>63.46</td><td>0.00</td><td>7.14</td><td>21.21</td><td>8.19</td><td>0.00</td><td>40.63</td><td>0.00</td><td>10.40</td><td>37.18</td><td>11.80</td><td>0.00</td></tr><tr><td>+ Flow-GRPO, k=8</td><td>74.18</td><td>0.00</td><td>6.59</td><td>9.34</td><td>9.89</td><td>0.00</td><td>51.73</td><td>0.00</td><td>11.39</td><td>21.04</td><td>15.84</td><td>0.00</td></tr></table>

## C. Outcome Attribution

For a control rollout with STOP disabled, let $t ^ { \star }$ be its first decision anchor inside the 20 m region $( t ^ { \star } = \infty$ if none). A first-stop outcome at decision t (or at the control budget if no STOP occurs) that ends inside the region is a success, with an explicit STOP if it ends at a STOP decision and without STOP if it ends at the control budget. Otherwise it is labeled Nav. if $t ^ { \star } = \infty ,$ , No-stop if no STOP occurs, Early if $t < t ^ { \star }$ , and Late if $t > t ^ { \star }$ , checked in this order. Each label is weighted by the first-stop probability of its outcome.