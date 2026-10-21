# World Models with Configurable Perception

Yĳun Shen<sup>1,\*</sup> Delong Chen<sup>2,\*</sup> Xianming Hu<sup>1</sup> Kai Zhang<sup>1,†</sup> Pascale Fung<sup>2</sup>

<sup>1</sup> East China Normal University <sup>2</sup> HKUST

\* Equal contribution † Corresponding Author kzhang@cs.ecnu.edu.cn

Real-world observations are noisy and often contain excessive task-irrelevant information. Efective world modeling and planning need to abstract observations into world state representations while filtering out irrelevant information. However, encoders in existing world models are typically non-adaptive, leading to state representations that are easily distracted by task-irrelevant information. As a result, the model may not selectively filter observations to cater to diferent task demands. To address this problem, we introduce ConfigWM, which realizes configurable perception through task-conditioned state extraction. Given a language description of the task at hand, ConfigWM uses it as a configuration signal to steer how visual embeddings are mapped into a task-specific state space, selectively preserving information relevant to the current task while suppressing task-irrelevant variations. Actionconditioned dynamics are then learned in this task-specific state space for planning. We evaluate ConfigWM in environments with increasing levels of distraction. While existing methods exhibit substantial declines in planning success rate as the environments become increasingly noisy, ConfigWM remains comparatively stable and outperforms the strongest competing method by 22 3% on average across diverse environments. Consistent gains in physical-quantity probing (+0<sub>.</sub>365 Pearson �) and distracted-to-clean retrieval (+75<sub>.</sub>5%) further indicate that ConfigWM learns states that preserve task-relevant information while remaining robust to distractors, supporting reliable world modeling and planning in noisy worlds. C:

Project Page: https://configwm.github.io

## 1 Introduction

Real-world environments contain far more information and dynamics than are needed for any single task [Chen et al., 2025b, Bharadhwaj et al., 2022, Toso et al., 2026].<sup>L</sup> However, what constitutes a suficient state depends ons the task: diferent tasks may require diferent information from the same observation. This challenge becomes <sup>Encoder</sup> particularly pronounced in dynamic environments, where<sub>o Con</sub>fi<sub>gurat</sub>i<sub>on o</sub> multiple objects or agents may coexist and evolve simultaneously [Wu et al., 2026, Wang et al., 2026b]. In suchObservation Observation settings, only some of these dynamics may be relevant to the current task. If task-irrelevant dynamics are retained in the state, the world model must account for transitions that do not contribute to the current objective, making prediction and planning inevitably dificult.

Existing world models do not explicitly resolve this challenge. In particular, feature-based world models construct states from task-agnostic representations [Zhou et al., 2024, Assran et al., 2025, Baldassarre et al., 2025]. However, a desirable state should act as an information bottleneck, retaining only the information suficient for current task while suppressing the task-irrelevant information [Pacelli and Majumdar, 2019, Eftekhar et al., 2024, Cui et al., 2026]. This is dificult to achieve with task-agnostic methods, since the relevance of an object can change with the task: the same object may be the target under one instruction but a distractor under another. The same observation can therefore require diferent representations under diferent task queries, whereas task-agnostic state representations remain unchanged.

![](images/2208903994046680e953f005cccaa0032a433cf9f0124fb79a7e3c953d0e5a43.jpg)  
Figure 1. ConfigWM Architecture.

In the literature, it has also be noted that task-agnostic approaches may retain both target and distractor dynamics in the state [Nikulin et al., 2026]. These limitations motivate adapting representations to better preserve task-relevant information. A more direct alternative is to introduce task conditioning directly into perception. This motivates configurable perception [LeCun et al., 2022, Shi et al., 2023, Marza et al., 2024], where the current task configures visual encoding to construct a task-relevant state.

In this work, we introduce configurable perception into world modeling and propose Config World Model (ConfigWM), an action-conditioned world model built on the task-relevant state abstractions, as illustrated in Fig. 1. Inspired by VL-JEPA [Chen et al., 2025a], we build ConfigWM on a configurable encoder that steers visual prediction with a task query and is therefore naturally suited to realizing configurable perception. The resulting query-conditioned representations define the task-relevant state space, over which ConfigWM learns action-conditioned dynamics for planning.

To evaluate whether configurable perception improves world modeling under dynamic distraction, we augment Two-Room, Push-T, and OGBench-Cube [Maes et al., 2026a] with increasing numbers of dynamic, target-like distractors and conduct controlled comparisons across diferent training and testing conditions. Baseline methods degrade substantially when trained on clean observations and evaluated under distraction. More importantly, simply exposing them to distractors during training does not solve the problem and can even further impair planning performance. In contrast, ConfigWM remains consistently robust as distraction increases, with an average planning improvement of 22.3%. These results show that robustness does not arise merely from exposure to distractors during training. Rather, conditioning perception on the task allows the world model to selectively preserve taskrelevant dynamics while suppressing irrelevant motion, thereby supporting reliable prediction and planning in dynamically distracting environments.

Apart from planning success, we further characterize the quality of the learned states from two complementary perspectives. Physical-quantity probing [Maes et al., 2026b, Alain and Bengio, 2016] measures how well the representations preserve task-relevant physical information, while distracted-to-clean retrieval [Garrido et al., 2024] evaluates their robustness to dynamic distractors. Across both analyses, baseline representations are increasingly afected as distractors are introduced, whereas ConfigWM remains substantially more stable. These results indicate that ConfigWM better preserves task-relevant information while suppressing noise.

In summary, the contributions of this paper are as follows:

• We propose ConfigWM, which realizes configurable perception by using task instructions to construct taskrelevant latent states and learn action-conditioned dynamics over them.

• We augment Two-Room, Push-T, and OGBench-Cube with progressively increasing dynamic, target-like distractors, establishing a controlled evaluation setting for world models in noisy environments.

• We show that ConfigWM maintains stable planning under dynamic distractors and achieves consistent gains in physical-quantity probing and distractedto-clean retrieval, indicating better preservation of task-relevant information.

## 2 Related Works

World State Representations. Recent world models have explored several ways to obtain latent states suitable for prediction and planning, from directly reusing pretrained visual features to jointly learning or further adapting the state representation. A common approach is to build predictive dynamics directly over pretrained or frozen visual features, using these representations as the state space for world modeling and planning [Zhou et al., 2024, Assran et al., 2025]. Rather than fixing this state space, other approaches further shape the representation for downstream control. Some jointly learn the latent state with dynamics, reward prediction, and value-learning objectives, allowing task returns to directly influence the learned representation [Hansen et al., 2022], others improve predictive state learning through latent regularization [Maes et al., 2026b, Zhao et al., 2026, Terver et al., 2026], explicitly ground the latent space in physical state and motion [Yan et al., 2026], or map generic visual features into more compact and control-relevant representations [Wang et al., 2026a, Fu et al., 2026a, Toso et al., 2026]. Compact latent representations have also been explored to improve the eficiency of predictive planning [Yao et al., 2026]. Despite these advances, task relevance is typically introduced by shaping the learned state, rather than by allowing the current task to determine what visual information is encoded in the first place.

Steerable Representations. Recent work has explored visual representations that can be steered by task or language signals, allowing the same observation to produce diferent representations depending on the current query. Some methods condition the visual encoder directly, enabling its features to emphasize queried objects, regions, or attributes [Ruthardt et al., 2026, Mao et al., 2026], while others introduce textual conditioning into predictive representation learning, where the query guides the features being predicted [Huang et al., 2026, Chen et al., 2025a]. However, these methods primarily focus on learning taskconditioned visual representations, leaving open whether the resulting representations can support better actionconditioned world modeling and therefore improve downstream planning. Language has also been used to condition future prediction and action generation [Cheang et al., 2024, Brohan et al., 2023]. ConfigWM difers by applying language at the state construction stage to obtain the task-relevant states for planning, extending the steerable representations to action-conditioned world modeling.

![](images/68b72171f57f766350dc630bff0aac6b4c04d672f3b3c4289e4bf224c46917d8.jpg)  
Figure 2. Constructing noisy worlds. Columns show observations with an increasing number of dynamic, target-like distractors. Rows show Two-Room, Push-T, and OGBench-Cube. Across all settings, the underlying task, action sequence, goal, and target trajectory remain unchanged.

## 3 Constructing Noisy Worlds

To evaluate world models under dynamic distraction, we construct controlled noisy variants of three widely used visual control environments: Two-Room [Sobal et al., 2025], Push-T [Chi et al., 2025], and OGBench-Cube [Park et al., 2025]. Together, they span 2D navigation, planar object manipulation, and 3D robotic manipulation, covering representative goal-conditioned planning settings studied in recent latent world models [Zhou et al., 2024, Wang et al., 2026a, Maes et al., 2026b].

Unlike existing noisy settings that primarily modify background appearance or introduce static clutter [Pumacay et al., 2024, Rasouli et al., 2025], we introduce multiple dynamic, target-like distractors that move independently and compete with the real target, while preserving the original task and target motion. By progressively increasing the number of such distractors, we create nine dificulty levels for each environment, allowing us to systematically vary dynamic distraction without changing the underlying control problem. The resulting matched clean and noisy observations are also used as training pairs for configurable perception in our model. We next describe the distractor construction (§ 3.1) and its instantiation in the three environments (§ 3.2).

## 3.1 Dynamic Distractor Construction

We construct the dynamic distractors directly in the observation space, without modifying the underlying simulator state or task dynamics. We first use SAM 3 [Carion et al., 2025] to segment and track the target-like objects from valid task rollouts. The extracted object masks are recolored and composited onto another rollout while preserving their original trajectories, producing visually similar objects whose motion remains consistent with the environment but is unrelated to the current task. Since this procedure operates only on the visual observations, it can also be adapted to other environments without modifying their underlying simulators.

As illustrated in Fig. 2, each clean rollout is used to construct nine matched visual conditions. The zerodistractor condition �<sub>0</sub> contains no additional distractors, while $d _ { \ell }$ contains ℓ independently moving distractors for $\ell = 1 , \ldots , 8$ . All levels share the same task query, action sequence, goal, and target trajectory, and difer only in the amount of competing object motion. Consequently, every noisy observation has a paired clean observation from the same rollout and time step. This matched construction provides both a controlled dificulty axis for studying world models under increasing distraction and the noisy–clean pairs used to train the configurable encoder in Sec. 4.1.

## 3.2 Environment Instantiations

Two-Room [Sobal et al., 2025]. A continuous 2D navigation task in which an agent must move through two connected rooms to reach a target location. We introduce additional agents with similar appearance, creating competing object motion around the query-specified agent.

Push-T [Chi et al., 2025]. A planar object-pushing task in which a circular pusher moves a T-shaped block to a target pose. We introduce additional recolored T-shaped blocks, requiring the model to distinguish and track the queryspecified block among visually similar moving objects.

OGBench-Cube [Park et al., 2025]. A 3D robotic manipulation task in which a robotic arm moves a designated cube to a target location. We recolor the robot arms in complete manipulation scenes and combine these scenes into a tiled observation. This creates multiple simultaneous arm–cube interactions, while the task query identifies the scene relevant for prediction and planning.

## 4 Methodology

We aim to learn a world model with adaptive state abstractions so as to support planning in noisy environments, where multiple dynamic, realistic, and target-like distractors coexist with the main target. Following the configurator view of autonomous intelligence [LeCun et al., 2022], ConfigWM uses language to configure perception, thereby constructing the task-relevant predictive states that serve as the state space for action-conditioned world modeling and planning.

Sec 1: Intro | Sec 2: Related Works | Sec 3: Noisy Worlds | Sec 4: Method | Sec 5: Experiments | Sec 6: Scaling | Sec 7: Conclusion | Sec 8: Appx  
![](images/9e650ccd78a2204466461f0eff3af95e19704978419ddd62654f0698bd2e5a85.jpg)  
Figure 3. Training pipeline. (a) Configurable encoder predicts the representation of the query-specified target from the noisy observation. (b) The frozen configurable encoder encodes the current and next observations under the same query, and ConfigWM learns the action-conditioned dynamics over the projected states.

Below, we first introduce a configurable encoder for configurable perception (§ 4.1), then describe how ConfigWM learns action-conditioned dynamics over the resulting predictive states (§ 4.2), and finally present latent-space planning with the learned world model (§ 4.3).

## 4.1 Configurable Encoder

To realize configurable perception, we introduce a configurable encoder, where a task query conditions the extraction of task-relevant visual representations. While language naturally specifies what is relevant to the task, text descriptions alone may not capture the fine-grained visual details required for low-level planning (e.g., the precise position of an agent). We therefore use language as a configuration signal while grounding the learned state representation in visual embeddings, allowing it to remain task-selective while preserving the detailed visual information needed for planning.

Formally, our model is trained with triplets $\langle X _ { V } , X _ { Q } , Y \rangle _ { \mathrm { \scriptsize { . } } }$ where $X _ { V }$ denotes a visual observation containing the task target and distractors, $X _ { Q }$ is the task query, and � is the corresponding distractor-free visual target. The model comprises three components:

$$
\begin{array} { r } { X _ { V } \xrightarrow { \mathrm { X \mathrm { - } E n c o d e r } } S _ { V } , \qquad Y \xrightarrow { \mathrm { Y \mathrm { - } E n c o d e r } } S _ { Y } , } \\ { \langle S _ { V } , X _ { Q } \rangle \xrightarrow { \mathrm { P r e d i c t o r } } \hat { S } _ { Y } . } \end{array}\tag{1}
$$

As illustrated in Fig. 3(a), both visual encoders are instantiated with DINOv2 [Oquab et al., 2023], whose dense spatial representations are well suited to low-level visual prediction and latent planning [Terver et al., 2025]. A predictor implemented with Llama 3 Transformer layers jointly processes $S _ { V }$ and the embedded task query, followed by pooling and projection to produce $\hat { S } _ { Y }$ . The prediction is trained to match the visual target representation $S _ { Y }$ using an InfoNCE objective [Radford et al., 2021], encouraging the model to build visual representations that well suit the task query.

## 4.2 ConfigWM: Modeling Noisy Worlds

Configurable encoder constructs a task-relevant predictive state by conditioning visual state construction on the task query. As illustrated in Fig. 3(b), ConfigWM then learns how this state evolves under actions. In this formulation, the query determines what enters the predictive state, while the action determines how the state changes.

Formally, given a transition $\left( O _ { t } , a _ { t } , O _ { t + 1 } \right)$ from the noisy environments and a task query $q ,$ where $O _ { t }$ and $O _ { t + 1 }$ are the current and next observations and $a _ { t }$ the action taken at time �, we apply the frozen configurable encoder to both observations under the same query, followed by a shared state projector $P _ { \phi }$ :

$$
\begin{array} { r l } & { ~ S _ { t } ^ { q } = P _ { \phi } ( E _ { \mathrm { C o n f i g u r a b l e - E n c o d e r } } ( O _ { t } , q ) ) , } \\ & { ~ S _ { t + 1 } ^ { q } = P _ { \phi } ( E _ { \mathrm { C o n f i g u r a b l e - E n c o d e r } } ( O _ { t + 1 } , q ) ) . } \end{array}\tag{2}
$$

We instantiate $F _ { \theta }$ as an action-conditioned causal Transformer [Maes et al., 2026b], which predicts the next taskrelevant state:

$$
\hat { S } _ { t + 1 } ^ { q } = F _ { \theta } ( S _ { t } ^ { q } , a _ { t } ) .\tag{3}
$$

For each task query $q , S _ { t } ^ { q } , \hat { S } _ { t + 1 } ^ { q } ,$ and $S _ { t + 1 } ^ { q }$ are represented in the same task-relevant latent space, where the distractor dynamics are suppressed. This provides a stable predictive state space for learning action-conditioned dynamics in the noisy environments. We jointly optimize the state projector $P _ { \phi }$ and the dynamics model $F _ { \theta }$ by minimizing the prediction error between the predicted and encoded next states, together with the SIGReg regularization [Balestriero and LeCun, 2025]:

$$
\mathcal { L } _ { \mathrm { W M } } = \left. \hat { S } _ { t + 1 } ^ { q } - S _ { t + 1 } ^ { q } \right. _ { 2 } ^ { 2 } + \lambda _ { \mathrm { s i g } } \mathcal { L } _ { \mathrm { S I G R e g } } .\tag{4}
$$

![](images/4948edff23d124191c685286234dbde7c6ef18338faecb1a769561664409984a.jpg)  
Figure 4. Planning with ConfigWM. Configurable encoder encodes the initial and goal observations under the task query. ConfigWM rolls out candidate actions in the query-conditioned latent space, where the terminal distance to the goal guides iterative action optimization.

## 4.3 Planning with ConfigWM

As illustrated in Fig. 4, ConfigWM performs the goalconditioned planning in the task-relevant latent space. Given an initial observation $O _ { 0 } ,$ , a goal observation $O _ { g } ,$ and a task query �, the frozen configurable encoder followed by the learned state projector encodes them as $S _ { 0 } ^ { q }$ and $S _ { g } ^ { q } ,$ respectively. For a candidate action sequence ${ \bf a } _ { 0 : H - 1 } = \left( a _ { 0 } , \dots , a _ { H - 1 } \right)$ , the world model recursively rolls out future states from $S _ { 0 } ^ { q }$ by applying $\hat { S } _ { h + 1 } ^ { q } = F _ { \theta } ( \bar { \hat { S } } _ { h } ^ { q } , a _ { h } )$ We score each sequence by the terminal latent distance $\| \hat { S } _ { H } ^ { q } - S _ { g } ^ { q } \| _ { 2 } ^ { 2 }$ . We use the cross-entropy method (CEM) [Rubinstein and Kroese, 2004] to optimize the candidate action sequences within a model predictive control framework (MPC) [Mayne et $\mathsf { a l . } ,$ , 2000]. Since the predicted states and the goal state are constructed under the same query, the terminal latent distance measures progress toward the task-relevant goal. After executing the first action, planning is repeated from the new observation.

## 5 Experiments

## 5.1 Experimental Setup

We compare ConfigWM with LeWM [Maes et al., 2026b], TC-LeWM [Liu et al., 2026], Sub-JEPA [Zhao et al., 2026], and EB-JEPA [Terver et al., 2026], with EB-JEPA evaluated only on Two-Room following its oficial implemen tation. These methods retain their original state-learning objectives and regularization strategies. To ensure a fair comparison and isolate the role of configurable encoder, ConfigWM uses the frozen configurable encoder to construct query-conditioned predictive states while adopting the LeWM-style action-conditioned causal Transformer as its dynamics model. A trainable state projector maps the spatial encoder features into the 192-dimensional global state required by the dynamics model.

We additionally construct DINOv2-WM as a controlled encoder baseline. It replaces configurable encoder with a frozen DINOv2 encoder while retaining an identical projector architecture, dynamics model, training objective, and planning protocol. Therefore, the comparison between DINOv2-WM and ConfigWM changes only how the visual predictive state is constructed. We also report Unified ConfigWM, which uses a single configurable encoder jointly trained on pooled data from all environments and distractor levels. With this encoder frozen, separate world models are trained under the same per-environment training and evaluation protocol as the standard comparisons. All methods are trained on the same transition data and evaluated under the same planning protocol. Further optimization and evaluation details are provided in Appx. B.

## 5.2 Planning under Dynamic Distractors

We first examine whether world models trained in clean environments can generalize to dynamic distractors. As shown in Fig. 13, when models trained on � are evaluated on $d _ { 1 } { - } d _ { 8 } ,$ , the planning performance of all baselines decreases substantially as the distractor level increases. This result shows that predictive states learned in clean scenes do not readily generalize when multiple target-like objects introduce unseen competing motion.

We further consider the matched setting, where each model is trained and evaluated at the same distractor level $( d _ { k } \to d _ { k } )$ . As shown in Fig. 5, even level-matched training does not reliably recover planning performance across baselines, particularly at higher distractor levels. In contrast, ConfigWM consistently outperforms all competing baselines across distracted levels in all three environments. Beyond its consistent performance gains, ConfigWM exhibits substantially greater robustness to increasing distraction. At $d _ { 8 } ,$ it retains an average of 83<sub>.</sub>8% of its $d _ { 0 }$ success rate across the three environments, whereas the baselines retain only 42<sub>.</sub>0%. Notably, ConfigWM remains robust under distraction even when its dynamics model is trained only on $d _ { 0 }$ transitions, achieving performance comparable to its level-matched training. Full results are provided in Appx. C.1.

These results suggest that training on distracted observations alone is not suficient. When multiple visually similar objects exhibit valid task dynamics, existing world models struggle to identify which dynamics are relevant to the current task. By using the task query to configure perception, ConfigWM selects task-relevant predictive states before dynamics modeling and thereby achieves more robust planning under dynamic distractors.

![](images/26bd93f88ca475939537918ca66079b9109d35c4f61e54b4db66976e5674770e.jpg)  
Figure 5. Matched planning performance under dynamic distractors. Planning success rates are shown across distractor levels �<sub>0</sub>–�<sub>8</sub> on Two-Room, Push-T, and OGBench-Cube, with each model trained and evaluated at the same level $( d _ { k } \to d _ { k } )$ . Shaded regions indicate one standard deviation across seeds, and the random policy is shown as a dotted horizontal reference. Higher is better.

![](images/375ff9442d55d706ef8b6b4b5e675c4c8548637c2b8a3b20aee789e4636a003c.jpg)  
Figure 6. Representation analysis. Left: linear probing of task-relevant physical quantities, reported as the macro-average of the per-quantity Pearson correlations. Right: distracted-to-clean Top-1 retrieval accuracy. The three panels in each group correspond to Two-Room, Push-T, and OGBench-Cube. The clean setting is denoted by � , while � –� introduce increasing levels of dynamic distractors. Higher is better

## 5.3 Is Task-Relevant Physical Information Preserved?

We evaluate which task-relevant physical quantities remain linearly recoverable from the learned predictive states. At each distractor level, we train a linear probe on the frozen representations and report the macro-average Pearson correlation � across the physical quantities defined for each environment [Maes et $\mathsf { a l . } ,$ 2026b]. Fig. 6 summarizes the results, while quantity-wise probing results are provided in Appx. C.2.

ConfigWM consistently preserves the linearly recoverable physical information as distractor dificulty increases. From $d _ { 0 }$ to $d _ { 8 } ,$ its probing score decreases by only 0<sub>.</sub>005 on average across the three environments, compared with an average decrease of 0<sub>.</sub>497 for the baselines in each environment. This contrast suggests that the query-configured predictive states preserve the task-relevant physical information even in the presence of multiple dynamic and

target-like distractors.

## 5.4 Do Distracted Representations Align with Clean Counterparts?

Nearest-neighbor retrieval provides a direct way to test whether a learned representation maps a transformed observation to its corresponding target in latent space [Garrido et al., 2024]. For the main evaluation, we randomly sample 400 paired clean–distracted representations. Each distracted representation is used as a query over the 400 clean representations, with its paired clean representation as the positive candidate and the remaining 399 as negatives. A high retrieval score indicates that the representation preserves the identity of the underlying task-relevant state despite the added distractor motion. Detailed level-wise results are provided in Appx. C.3.

As shown in the right panels of Fig. 6, ConfigWM maintains strong clean–distracted correspondence across all three environments. This suggests that query-configured predictive states are less afected by distractor motion and better preserve task-relevant information, consistent with ConfigWM’s stronger retention of clean-setting planning performance in Fig. 13.

![](images/825a0333c76563f843d31f35a4ef8df7d3bd74621cd62203823452b593c06975.jpg)  
Figure 7. Decoding query-configured predictive states. Given the same distracted observation, we vary only the color query and decode the resulting predictive state using a shared decoder. Each decoded observation recovers the queried block and task-relevant state, while preserving other task-relevant elements (e.g., the green goal block).

![](images/5bc3dfe799b9e70d2d5415e423b8501c5f81fd24dfce235056e59d3709aab010.jpg)  
Figure 8. Planning with mismatched queries. Success rates (%) are averaged over distractor levels $d _ { 1 } { - } d _ { 8 }$ . Mismatched results replace the correct query with valid queries referring to other objects. Drop denotes the decrease caused by query mismatch.

## 5.5 Does the Query Configure the Predictive State?

To examine whether the task query determines which visual information is retained in the predictive state, we freeze the configurable encoder and train a single shared ViT decoder [Terver et al., 2025] on its predictor outputs. As shown in Fig. 7, changing only the queried color produces distinct reconstructions from the same input observation. Each decoded observation retains the queried block together with the task-relevant agent and target pose, while largely suppressing the remaining distractors. Since the decoder has no direct access to the query, these diferences provide qualitative evidence that the query configures which visual content is retained in the predictive state.

![](images/5314bc2adb7c0ccf27664be33d2c2ac1b39a2fddba95ba9202195c937bf2904a.jpg)

![](images/9d3ee67eef766b4ee911f850c6c43224a9aa9b5575034d8592c5c2923f9fddb6.jpg)  
Figure 9. Representation metrics correlate with planning performance. Each marker pairs the representation score and success rate for diferent models and distractor levels. Regression lines and Pearson � are computed separately within each environment.

## 5.6 Does Perceptual Configuration Afect Planning?

While the decoding results provide qualitative evidence that configurable encoder realizes configurable perception, we further examine whether this query-dependent state construction afects downstream planning. We keep the task, observations, model, and planning procedure fixed, but replace the correct query with the other valid but mismatched queries referring to other objects in the scene.

As shown in Fig. 8, mismatched queries consistently reduce planning success across all three environments. Averaged over $d _ { 1 } \mathrm { - } d _ { 8 }$ and the three environments, planning success decreases from 67<sub>.</sub>8% with the correct query to 31<sub>.</sub>6% with mismatched queries, approaching the 29<sub>.</sub>1% performance of a random policy. Since the task query is the only changed input, the resulting performance drop provides direct evidence that downstream planning depends on the query-configured predictive states constructed through configurable perception. Complete results are provided in Appx. C.5.

![](images/ccc4343824d48ee6b88850c5c7b225d7c7f8a494a8006c543f82b334ec354734.jpg)  
Figure 10. Configurable encoder across visual backbones. We compare each backbone with our configurable encoder built upon the same backbone. Across DINOv2, DINOv3, V-JEPA 2.1, and LingBot-Vision, ours suppresses task-irrelevant distractors while preserving task-relevant state information. Black dashed boxes mark distractors.

## 5.7 Do Representation Metrics Reflect Planning Performance?

We further examine whether these representation-level metrics provide meaningful intermediate measures of downstream planning performance using correlation analysis [Dong et al., 2026, Shen et al., 2025]. Within each environment, we compare the raw representation scores with matched planning success across the six methods evaluated in all three environments and distractor levels $d _ { 1 } { - } d _ { 8 } ,$ and report the corresponding Pearson � and regression line. As shown in Fig. 9, both metrics are strongly associated with planning success. Across the three environments, physical-quantity probing and retrieval achieve average Pearson correlations of 0<sub>.</sub>89 and 0<sub>.</sub>89, respectively. These results suggest that preserving task-relevant physical information and alignment with clean counterparts provide informative intermediate measures of planning performance across methods and distractor levels.

## 6 Scaling the configurable encoder with real-world data

We explore a scalable training route that converts existing egocentric [Grauman et al., 2022, Damen et al., 2022], robot manipulation [Khazatsky et al., 2024, O’Neill et al., 2024] and internet instructional video [Tang et al., 2019, Chen et al., 2026] data into supervision for configurable perception. Given a clean observation and its task description, a VLM generates editing prompts specifying reasonable but taskirrelevant objects, which are inserted into the observation using an image-editing model. The task-relevant scene content and task description remain unchanged, and the original observation serves as the clean target. The configurable encoder is then trained to predict the representation of the clean target from the noisy observation conditioned on the task description.

Fig. 10 shows held-out PCA visualizations across DI-NOv2, DINOv3, V-JEPA 2.1, and LingBot-Vision. Complementary patch-level linear probing for distractor and task-relevant region prediction shows that the configurable encoder reduces distractor average precision (AP) by an average of 17.9 percentage points relative to the frozen backbones, while task-core AP remains nearly unchanged, indicating weaker responses to task-irrelevant objects while preserving task-relevant states. Full quantitative results and the evaluation protocol are provided in Appendix D. These results suggest that our encoder can benefit different visual backbones, providing a practical route for scaling configurable perception to large real-world video datasets, without requiring manual annotation.

## 7 Conclusion

We introduced ConfigWM, an action-conditioned world model that brings configurable perception into state construction. Rather than modeling dynamics over a fixed task-agnostic representation, ConfigWM uses the task query to construct a task-relevant state and learns actionconditioned dynamics over it. Across three environments with increasing dynamic, target-like distractors, ConfigWM maintains stronger planning performance than the baselines, while representation analyses show that its states preserve task-relevant physical information and remain more consistent with their clean counterparts as distraction increases. We further examine whether the query actually configures the learned state, finding that decoding the states under diferent queries reveals diferent task-relevant state representations, while replacing the correct query with a mismatched one leads to a substantial drop in planning success. Beyond these controlled environments, results on real-world data suggest a promising path to scaling the configurable encoder across diferent visual backbones.

## Acknowledgments

We would like to thank Willy Chung, Quentin Garrido, Basile Terver for providing valuable feedback to this work.

## References

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Federico Baldassarre, Marc Szafraniec, Basile Terver, Vasil Khalidov, Francisco Massa, Yann LeCun, Patrick Labatut, Maximilian Seitzer, and Piotr Bojanowski. Back to the features: Dino as a foundation for video world models. arXiv preprint arXiv:2507.19468, 2025.

Randall Balestriero and Yann LeCun. Lejepa: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544, 2025.

Homanga Bharadhwaj, Mohammad Babaeizadeh, Dumitru Erhan, and Sergey Levine. Information prioritization through empowerment in visual model-based rl. arXiv preprint arXiv:2204.08585, 2022.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu,

Yichu Yang, et al. Gr-2: A generative video-languageaction model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.

Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, and Pascale Fung. Vl-jepa: Joint embedding predictive architecture for vision-language. arXiv preprint arXiv:2512.10942, 2025a.

Delong Chen, Tejaswi Kasarla, Yejin Bang, Mustafa Shukor, Willy Chung, Jade Yu, Allen Bolourchi, Theo Moutakanni, and Pascale Fung. Action100m: A large-scale video action dataset. arXiv preprint arXiv:2601.10592, 2026.

Yuxin Chen, Jianglan Wei, Chenfeng Xu, Boyi Li, Masayoshi Tomizuka, Andrea Bajcsy, and Ran Tian. Reimagination with test-time observation interventions: Distractor-robust world model predictions for visual model predictive control. arXiv preprint arXiv:2506.16565, 2025b.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Difusion policy: Visuomotor policy learning via action difusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.

Jingyi Cui, Qi Zhang, Hongwei Wen, and Yisen Wang. A generalization theory for jepa-based world models. arXiv preprint arXiv:2606.27014, 2026.

Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal of Computer Vision, 130(1):33–55, 2022.

Jiahua Dong, Yunze Man, Pavel Tokmakov, and Yu-Xiong Wang. Capturing visual environment structure correlates with control performance. In International Conference on Learning Representations, volume 2026, pages 35435–35462, 2026.

Ainaz Eftekhar, Kuo-Hao Zeng, Jiafei Duan, Ali Farhadi, Aniruddha Kembhavi, and Ranjay Krishna. Selective visual representations improve convergence and generalization for embodied ai. In International Conference on Learning Representations, volume 2024, pages 13256– 13277, 2024.

Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable

distributed deep-rl with importance weighted actorlearner architectures. In International conference on machine learning, pages 1407–1416. PMLR, 2018.

Minghao Fu, Fan Feng, Nicklas Hansen, and Biwei Huang. Back to parsimonious latents: Learning task-centric world models from visual foundations. arXiv preprint arXiv:2605.25620, 2026a.

Zelin Fu, Bin Tan, Changjiang Sun, Shaohui Liu, Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, and Nan Xue. Vision pretraining for dense spatial perception. arXiv preprint arXiv:2607.05247, 2026b.

Quentin Garrido, Mahmoud Assran, Nicolas Ballas, Adrien Bardes, Laurent Najman, and Yann LeCun. Learning and leveraging world models in visual representation learning. arXiv preprint arXiv:2403.00504, 2024.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022.

Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vĳay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19383–19400, 2024.

Nicklas Hansen, Xiaolong Wang, and Hao Su. Temporal diference learning for model predictive control. arXiv preprint arXiv:2203.04955, 2022.

Chen Huang, Xianhang Li, Vimal Thilak, Etai Littwin, and Josh Susskind. Text-conditional jepa for learning semantically rich visual representations. arXiv preprint arXiv:2605.03245, 2026.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

Yann LeCun et al. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62 (1):1–62, 2022.

Chang Liu, Fei Suo, Yanzhou Jin, Yusuke Iwasawa, Yutaka Matsuo, and Yaonan Zhu. Temporally centered sigreg improves multi-task leworldmodel learning: From analysis to method. arXiv preprint arXiv:2607.26924, 2026.

Lucas Maes, Quentin Le Lidec, Dan Haramati, Nassim Massaudi, Damien Scieur, Yann LeCun, and Randall Balestriero. stable-worldmodel-v1: Reproducible world modeling research and evaluation. arXiv preprint arXiv:2602.08968, 2026a.

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels. arXiv preprint arXiv:2603.19312, 2026b.

Chengzhi Mao, Xudong Lin, and Wen-Sheng Chu. Language-instructed vision embeddings for controllable and generalizable perception. arXiv preprint arXiv:2606.19584, 2026.

Pierre Marza, Laetitia Matignon, Olivier Simonin, and Christian Wolf. Task-conditioned adaptation of visual features in multi-task policy learning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17847–17856. IEEE, 2024.

David Q Mayne, James B Rawlings, Christopher V Rao, and Pierre OM Scokaert. Constrained model predictive control: Stability and optimality. Automatica, 36(6):789– 814, 2000.

Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun, Nicolas Ballas, and Adrien Bardes. V-jepa 2.1: Unlocking dense features in video self-supervised learning. arXiv preprint arXiv:2603.14482, 2026.

Alexander Nikulin, Ilya Zisman, Albina Klepach, Denis Tarasov, Alexander Derevyagin, Andrei Polubarov, Lyubaykin Nikita, and Vladislav Kurenkov. Visionlanguage models unlock task-centric latent actions. arXiv preprint arXiv:2601.22714, 2026.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

Vincent Pacelli and Anirudha Majumdar. Task-driven estimation and control via information bottlenecks. In 2019 international conference on robotics and automation (ICRA), pages 2061–2067. IEEE, 2019.

Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. Ogbench: Benchmarking ofline goalconditioned rl. In International Conference on Learning Representations, volume 2025, pages 94937–94982, 2025.

Wilbert Pumacay, Ishika Singh, Jiafei Duan, Ranjay Krishna, Jesse Thomason, and Dieter Fox. The colosseum: A benchmark for evaluating generalization for robotic manipulation. arXiv preprint arXiv:2402.08191, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Amir Rasouli, Montgomery Alban, Sajjad Pakdamansavoji, Zhiyuan Li, Zhanguang Zhang, Aaron Wu, and Xuan Zhao. Distracted robot: How visual clutter undermine robotic manipulation. arXiv preprint arXiv:2511.22780, 2025.

Reuven Y Rubinstein and Dirk P Kroese. The cross-entropy method: a unified approach to combinatorial optimization, Monte-Carlo simulation, and machine learning, volume 133. Springer, 2004.

Jona Ruthardt, Manu Gaur, Deva Ramanan, Makarand Tapaswi, and Yuki M Asano. Steerable visual representations. arXiv preprint arXiv:2604.02327, 2026.

Dian Shao, Yue Zhao, Bo Dai, and Dahua Lin. Finegym: A hierarchical video dataset for fine-grained action understanding. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2613–2622. IEEE, 2020.

Yĳun Shen, Delong Chen, Fan Liu, Xingyu Wang, Chuanyi Zhang, Liang Yao, and Yuhui Zheng. Chain-of-talkers (cotalk): Fast human annotation of dense image captions. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 4444–4464, 2025.

Baifeng Shi, Trevor Darrell, and Xin Wang. Top-down visual attention from analysis by synthesis. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2102–2112. IEEE, 2023.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cĳo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Vlad Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim GJ Rudner, and Yann LeCun. Stresstesting ofline reward-free reinforcement learning: A case for planning with latent dynamics models. In 7th Robot Learning Workshop: Towards Robots with Human-Level Abilities, 2025.

Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1207–1216. IEEE, 2019.

Basile Terver, Tsung-Yen Yang, Jean Ponce, Adrien Bardes, and Yann LeCun. What drives success in physical planning with joint-embedding predictive world models? arXiv preprint arXiv:2512.24497, 2025.

Basile Terver, Randall Balestriero, Megi Dervishi, David Fan, Quentin Garrido, Tushar Nagarajan, Koustuv Sinha, Wancong Zhang, Mike Rabbat, Yann LeCun, et al. A lightweight library for energy-based joint-embedding predictive architectures. arXiv preprint arXiv:2602.03604, 2026.

Leonardo F Toso, Davit Shadunts, Yunyang Lu, Nihal Sharma, Donglin Zhan, Nam H Nguyen, and James Anderson. Learning invariant visual representations for planning with joint-embedding predictive world models. arXiv preprint arXiv:2602.18639, 2026.

Ying Wang, Oumayma Bounou, Gaoyue Zhou, Randall Balestriero, Tim GJ Rudner, Yann LeCun, and Mengye Ren. Temporal straightening for latent planning. arXiv preprint arXiv:2603.12231, 2026a.

Zizhao Wang, Chang Shi, Jiaheng Hu, Kevin Rohling, Roberto Martín-Martín, Amy Zhang, and Peter Stone. Factored latent action world models. arXiv preprint arXiv:2602.16229, 2026b.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

Haoyu Wu, Jiwen Yu, Yingtian Zou, and Xihui Liu. Multiworld: Scalable multi-agent multi-view video world models. arXiv preprint arXiv:2604.18564, 2026.

Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, et al. Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation. arXiv preprint arXiv:2412.13877, 2024.

Haodong Yan, Jiaguan Zhu, Mingyuan Jia, Ruiqing Yin, Junjie He, Zhide Zhong, Junfeng Li, Jinxuan Lu, Hengtao Li, Tianran Zhang, et al. Is forward prediction enough? physical state grounding for jepa world models. arXiv preprint arXiv:2608.06799, 2026.

Liang Yao, Fan Liu, Hongbo Lu, Wei Xu, Jianyu Jiang, Yĳun Shen, Chuanyi Zhang, and Pai Peng. Dronewam: Eficient world action model for drone visual navigation. arXiv preprint arXiv:2609.33148, 2026.

Kai Zhao, Dongliang Nie, Yuchen Lin, Zhehan Luo, Yixiao Gu, Deng-Ping Fan, and Dan Zeng. Sub-jepa: Subspace gaussian regularization for stable end-to-end world models. arXiv preprint arXiv:2605.09241, 2026.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. arXiv preprint arXiv:2411.04983, 2024.

## Appendix

## A. Details on Noisy-World Construction

## – A.1 Dynamic Distractor Generation

Describes the complete dataset generation pipeline, including SAM3-based object extraction and temporal tracking, environment-matched reconstruction, cumulative distractor levels, and color balancing.

## – A.2 Environment-Specific Construction

Details environment-specific distractor construction and query templates for Two-Room, Push-T, and OGBench-Cube.

## – A.3 Dataset Scale

Summarizes dataset statistics and representative examples of progressive noisy-world construction.

## B. Implementation Details

## – B.1 The Configurable Encoder Training

Provides the configurable encoder architecture and training configuration, including the visual encoders, language-conditioned predictor, training objective, samples, and optimization settings.

## – B.2 ConfigWM Training

Describes state projection and action-conditioned dynamics training with frozen configurable encoder representations.

## – B.3 Baseline Implementation Details

Documents the architectures, objectives, and optimization settings usedfor LeWM, Sub-JEPA, TC-LeWM, DINOv2-WM, and EB-JEPA, highlighting the method-specific diferences retained in each comparison.

## – B.4 Planning and Evaluation Protocol

Specifies the shared CEM-based planning and evaluation protocol.

## C. Additional Experimental Results

## – C.1 Detailed Planning Results

Provides the complete matched and clean-trained transfer results for every environment, distractor level, and comparison method, including per-level success rates and aggregate planning performance.

## – C.2 Quantity-Wise Physical Probing Results

Reports per-quantity linear-probing results across environments and distractor levels.

## – C.3 Detailed Clean–Distracted Retrieval Results

Reports Top-5 and MRR results for distracted-to-clean state retrieval.

## – C.4 Detailed Query-Conditioning Results

Visualizes how the task query changes the visual content retained in the predictive state.

## – C.5 Detailed Query-Conditioned Planning Results

Reports query-conditioned planning performance at each distractor level, comparing the correct query with the average over mismatched queries, and separately reports the success rate and performance drop for every mismatched query.

## – C.6 Qualitative Planning Rollouts

Visualizes complete planning episodes under clean and noisy observations, comparing ConfigWM with the strongest baseline in each environment and illustrating representative successes and failures.

## D. Scaling the Configurable Encoder with Real-World Data

Describes the construction of task-conditioned real-world training pairs, the configurable encoder training withfour large visual backbones, qualitative analysis and quantitative evaluation on test sets, and the complete image-editing prompts for data generation

## A Details on Noisy-World Construction

This section details the construction of our noisy-world datasets. We first describe the shared distractor-generation pipeline in Sec. A.1, then its environment-specific instantiations in Sec. A.2, and finally the resulting dataset scale in Sec. A.3.

## A.1 Dynamic Distractor Generation

Object Extraction. For each environment, we begin with the oficial clean rollouts and use SAM3 [Carion et al., 2025] to identify the task-relevant target object. Specifically, SAM3 segments the object and tracks its mask consistently across frames, producing a sequence of per-frame masks and image-space trajectories. These masks and tracks provide the spatial extent and motion information required to replay the extracted objects as distractors in other rollouts. We then reconstruct each extracted object to match the native appearance of the corresponding environment. This reconstruction preserves its characteristic geometry, boundary profile, and shading while allowing controlled recoloring. Finally, the reconstructed object is composited into a recipient rollout by replaying its tracked donor trajectory frame by frame. Consequently, each distractor follows motion observed in a valid rollout rather than being independently repositioned in every frame, resulting in temporally stable and visually plausible distractors.

Progressive noise levels. We construct the noise levels cumulatively. The $d _ { 0 }$ condition contains only the recolored task-relevant objects and no additional distractors. For each trajectory, we sample eight donor trajectories and keep their identities and tracks fixed throughout the construction. Starting from $d _ { 0 } ,$ we add one additional donor trajectory at each noise level. This progressive construction enables controlled comparisons while keeping all previously introduced distractors unchanged.

Color-balanced and query-conditioned targets. In this work, we use object color as a concrete and easily controlled visual configuration attribute, although the formulation itself is not inherently restricted to color. For example, the Two-Room environment uses nine semantically named colors: red, blue, green, orange, purple, cyan, yellow, pink, and brown. To prevent the clean target from being associated with a fixed appearance inherited from the original dataset, such as the red ball in Two-Room, we recolor the task-relevant core object and balance the nine colors across trajectories. At every noise level $d _ { \ell } ,$ the core object and its ℓ distractors are assigned mutually distinct colors.

In addition to the standard source-to-clean pairs derived from the oficial datasets, we construct a same-source multi-query dataset from $d _ { 8 . }$ , which contains the complete set of one core object and eight distractors. At a temporal stride of five, the same noisy observation is paired with nine color queries and nine corresponding query-selected clean targets. This construction provides a direct realization of configurable perception, changing only the language query changes the desired target while keeping the source pixels fixed. Each target retains only the object or manipulation scene specified by the query (task-relevant) and removes the non-queried (task-irrelevant) distractors, while preserving task-relevant contextual elements (e.g., goals and interaction objects).

## A.2 Environment-Specific Construction

Two-Room. The task-relevant object is the moving ball. SAM3 is used to extract and track balls from donor trajectories. Their soft, Gaussian-like appearance is reconstructed with the native style of the environment, recolored, and replayed along the extracted trajectories. The room walls, and other static geometry are preserved during compositing. Queries follow the template {color} ball, and the corresponding target contains only the queried ball in the matched clean room observation.

Push-T. The task-relevant object is the movable T-shaped block, while the green T-shaped region and blue pusher provide task context. Additional T-block trajectories are extracted from valid donor rollouts. We reconstruct each donor with the geometry and edge appearance of the original T-block, recolor ${ \mathrm { i t } } ,$ and replay it along its donor trajectory. Only the donor T-block is transferred, while the green goal and blue pusher are retained from the recipient observation. Natural occlusions between the pusher, selected T-block, distractors, and goal are preserved during compositing. Queries follow the template {color} T block.

OGBench-Cube. Each donor corresponds to a complete arm–cube manipulation trajectory, identified by the color of its robot arm. SAM3 is used to extract and track the relevant manipulation content, after which the robot arm is reconstructed with an environment-matched appearance and recolored. The associated cube and interaction context are retained so that each donor remains a coherent manipulation scene. To keep all manipulation scenes visible, we use an adaptive tiled layout: $d _ { 1 } - d _ { 3 }$ use a $2 \times 2 \mathrm { g r i d }$ , whereas $d _ { 4 } \mathrm { - } d _ { 8 }$ use a 3 × 3 grid. This layout change is introduced only to accommodate the increasing number of distractors, while the controlled task, action sequence, goal, and target trajectory remain unchanged. Queries follow the template {color} arm and select the clean manipulation observation associated with the correspondingly colored arm.

Table 1. Scale of the constructed noisy-world datasets.
<table><tr><td>Environment</td><td>Trajectories</td><td>Avg. Steps</td><td>Source-Clean Pairs</td></tr><tr><td>Two-Room</td><td>10,000</td><td>92.1</td><td>10,002,339</td></tr><tr><td>Push-T</td><td>18,685</td><td>125.1</td><td>25,301,106</td></tr><tr><td>OGBench-Cube</td><td>10,000</td><td>201.0</td><td>21,780,000</td></tr><tr><td>Total</td><td>38,685</td><td>136.2</td><td>57,083,445</td></tr></table>

## A.3 Dataset Scale

Our noisy-world dataset spans three environments: Two-Room [Sobal et al., 2025], Push-T [Chi et al., 2025], and OGBench-Cube [Park et al., 2025]. All three datasets are constructed from the oficial clean trajectories released by Maes et al. [2026a]. Their detailed statistics are summarized in Table 1 (the average trajectory length in the total row is weighted by the number of trajectories in each environment). A source–clean pair consists of a source observation, a language query, and the corresponding query-conditioned clean target at the same time step. Source-clean pairs include all samples from $d _ { 0 } \mathrm { - } d _ { 8 } .$ . In addition to adding distractors to each frame, we also construct samples in which the same source frame is paired with diferent queries and their corresponding clean targets.

![](images/0309f0207ac3fc3361e444a3080596d36cc509d7786c8057dff900301637634d.jpg)  
Figure 11. Progressive noisy-world construction and query-conditioned targets. Within each row, �<sub>1</sub>–�<sub>8</sub> cumulatively add distractors to the same trajectory frame. The rightmost column shows the �<sub>0</sub> target selected by the displayed query.

## B Implementation Details

## B.1 The Configurable Encoder Training

We train the configurable encoder on the three controlled environments used in our main experiments: Two-Room, Push-T, and OGBench-Cube. We later extend the same training formulation to real-world visual data in Sec. D. Below, we describe the model architecture, the patch-wise contrastive objective, and the training data and optimization setup.

Architecture. Both the input encoder $E _ { X }$ and target encoder $E _ { Y }$ are initialized from pretrained DINOv2-S. All images are resized to 224 × 224 and processed using the standard DINOv2 preprocessing pipeline. Each image is encoded into one CLS token and a $1 6 \times 1 6$ grid of 384-dimensional patch tokens. During the configurable encoder training, $E _ { X }$ is optimized jointly with the predictor, whereas $E _ { Y }$ is kept as an independent frozen copy of DINOv2, providing a fixed target representation space throughout training.

The predictor $P _ { \theta }$ consists of the first three Transformer layers of pretrained Llama-3.2-1B. A trainable linear layer maps the 384-dimensional visual tokens to the Llama hidden dimension, while the task query is tokenized with the Llama tokenizer and padded or truncated to at most 32 tokens. The pretrained Llama token-embedding layer is kept frozen, whereas the Transformer layers and the visual input and output projections are trainable. The predictor processes the concatenation of the visual CLS and patch tokens with the query tokens, and its visual patch outputs are projected back to 384 dimensions, matching the dimensionality of the DINOv2 feature space. This provides a common feature interface for the subsequent world-model experiments and facilitates controlled comparison with DINOv2-WM. The target encoder receives only the corresponding clean target image and does not take the task query as input. This keeps the target representation query-independent, requiring the predictive branch to use the query to select task-relevant visual information from the noisy observation.

Training Data. The configurable encoder is trained using the noisy-world datasets described in Appendix A. Each training sample consists of a noisy source observation, a task query, and the corresponding clean target. We use source-clean pairs constructed across diferent distractor levels and additionally include same-source multi-query samples, where the same observation is paired with multiple valid queries and clean targets. These samples provide diverse query-conditioned training signals while preserving the same underlying visual state. The overall scale of the constructed training data is summarized in Table 1.

Training Objective. The configurable encoder is trained with a spatially aligned patch-level multi-positive InfoNCE objective. Given the visual embeddings $S _ { V }$ and query embeddings $S _ { Q }$ , the predictor produces $\hat { S } _ { Y } ,$ , which is trained to match the target representation $S _ { Y }$ from the frozen Y-Encoder. Since the source and target observations are spatially aligned, each predicted patch is contrasted against target patches at the corresponding spatial location across the contrastive bank. Targets sharing the same query-conditioned target identity are treated as positives, whereas samesource samples associated with diferent queries remain negatives. Following the bidirectional InfoNCE formulation of Chen et al. [2025a], we average the loss across spatial locations and compute it symmetrically in both directions:

$$
\mathcal { L } _ { \mathrm { C o n f i g u r a b l e - E n c o d e r } } = \frac { 1 } { 2 } \left( \mathcal { L } _ { \hat { S } _ { Y } \to S _ { Y } } + \mathcal { L } _ { S _ { Y } \to \hat { S } _ { Y } } \right) .\tag{5}
$$

Optimization Details. We train the configurable encoder for one epoch using AdamW with the settings summarized in Table 2.

Table 2. Optimization settings for the configurable encoder training.
<table><tr><td>Setting</td><td>Value</td><td>Setting</td><td>Value</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>Weight decay</td><td>0.05</td></tr><tr><td>LR schedule</td><td>100-step warmup; cosine decay</td><td>Global batch size</td><td>512</td></tr><tr><td>Temperature τ</td><td>0.07</td><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Numerical precision</td><td>BF16</td><td>Training hardware</td><td>1× A100 (80 GB)</td></tr></table>

## B.2 ConfigWM Training

ConfigWM follows the action-conditioned latent dynamics formulation of LeWM [Maes et al., 2026b]. To isolate the contribution of the configurable encoder under a controlled comparison, we replace LeWM’s end-to-end visual encoder with the frozen configured-state representation model while retaining its state dimensionality, causal Transformer dynamics core, and training objective. Accordingly, a trainable spatial projector aggregates the configurable encoder’s patch features into the 192-dimensional state ${ \check { S } } _ { t } ^ { q } ,$ matching the state dimensionality used by LeWM. The projector is shared across input observations and target observations, and trained jointly with the world model, while all the configurable encoder parameters remain frozen.

The dynamics predictor is a causal Transformer with AdaLN-zero action conditioning. During training, it takes a three-state context sampled at intervals of $\Delta = 5$ environment steps, with each state conditioned on the intervening Δ-step action block, and predicts the subsequent states using teacher forcing. Actions are normalized using the training-set statistics. Following LeWM, we optimize the mean-squared latent prediction error together with SIGReg:

$$
\mathcal { L } _ { \mathrm { C o n f i g W M } } = \mathcal { L } _ { \mathrm { p r e d } } + \lambda _ { \mathrm { s i g } } \mathcal { L } _ { \mathrm { S I G R e g } } , \qquad \lambda _ { \mathrm { s i g } } = 0 . 0 9 ,\tag{6}
$$

where $\mathcal { L } _ { \mathrm { p r e d } }$ denotes the mean-squared error between the predicted and target states.

## B.3 Baseline Implementation Details

All comparison methods are trained on the same trajectories for 10 epochs, and we report the best-performing checkpoint selected using the same procedure. LeWM, Sub-JEPA, TC-LeWM, DINOv2-WM, and ConfigWM use a three-state context with $\Delta = 5$ environment steps and share the same action normalization and online planning protocol. We describe only the method-specific diferences below.

LeWM-Family Baselines. LeWM [Maes et al., 2026b], Sub-JEPA [Zhao et al., 2026], and TC-LeWM [Liu et al., 2026] use a randomly initialized ViT-Tiny encoder and are trained end to end, retaining LeWM’s original 192-dimensional state representation and predictor projection. These baselines use a global batch size of 128 and cosine learning-rate decay. LeWM uses its original latent prediction objective with SIGReg. Sub-JEPA replaces SIGReg with its multi-subspace variant, using $K = 1 6$ for Push-T and $K = 3 2$ for Two-Room and OGBench-Cube, following the best-performing settings reported in the original paper. TC-LeWM instead uses temporally centered SIGReg with a temporal window of $W = 4 ,$ covering the three context states and their prediction target.

DINOv2-WM. As a controlled encoder baseline for ConfigWM, DINOv2-WM replaces the configurable encoder with a frozen pretrained DINOv2-S visual encoder [Oquab et al., 2023] and does not use language conditioning. Its patch-token features are mapped to the 192-dimensional state space using the same spatial projector as ConfigWM. All remaining components and settings, including the action encoder, dynamics predictor, training objective, and optimization protocol, are identical to those of ConfigWM. The DINOv2 encoder remains frozen, and only the state projector, action encoder, and dynamics predictor are trained.

EB-JEPA. EB-JEPA [Terver et al., 2026] is evaluated only on Two-Room, following its oficial implementation. We retain its original IMPALA-based [Espeholt et al., 2018] encoder, training objective, and optimizer settings and train the model end to end from random initialization. It uses the same Two-Room training trajectories, checkpoint-selection procedure, and online planning protocol as the other methods.

## B.4 Planning and Evaluation Protocol

We evaluate all world models using the same CEM-based model predictive control protocol, following LeWM [Maes et al., 2026b]. At each replanning step, the current and goal observations are encoded using the corresponding representation model. CEM optimizes candidate action sequences by rolling them out through the learned dynamics model and minimizing the distance between the terminal predicted state and the encoded goal state. A prefix of the optimized action sequence is executed before replanning from the updated observation. The same planner configuration is used across methods within each environment. All reported results are averaged over evaluation seeds 40, 42, and 45.

## C Additional Experimental Results

This section provides the complete quantitative and qualitative results corresponding to the experiments in Section $5 ,$ following the same presentation order as the main paper.

## C.1 Detailed Planning Results

Tables 3 and 4 report the complete planning results for the matched and clean-trained transfer settings, respectively. All results are aggregated over evaluation seeds 40, 42, and 45.

Table 3. Matched planning performance. Models are trained and evaluated at the same distractor level $( d _ { k } \to d _ { k } )$ . Values are mean success rates (%) over seeds 40/42/45. Light-blue tags indicate standard deviation. Average is computed by first averaging �<sub>0</sub>–�<sub>8</sub> within each seed and then aggregating the three seed-level averages. Column-best means are bold.
<table><tr><td rowspan="2">Method</td><td colspan="9">Distractor Level</td><td rowspan="2">Average</td></tr><tr><td></td><td> $d _ { 1 }$ </td><td> $d _ { 2 }$ </td><td> $d _ { 3 }$ </td><td> $d _ { 4 }$ </td><td> $d _ { 5 }$ </td><td> $d _ { 6 }$ </td><td> $d _ { 7 }$ </td><td> $d _ { 8 }$ </td></tr><tr><td colspan="10">Two-Room</td></tr><tr><td>Random Policy</td><td colspan="10"></td></tr><tr><td>LeWM</td><td> $\mathbf { 1 0 0 . 0 } _ { \pm 0 . 0 }$ </td><td> $6 4 . 0 _ { \pm 4 . 3 }$ </td><td> $4 0 . 0 _ { \pm 5 . 9 }$ </td><td> $3 5 . 3 _ { \pm 2 . 5 }$ </td><td> $3 0 . 0 _ { \pm 4 . 9 }$   $4 1 . 3 _ { \pm 1 0 . 9 }$ </td><td> $3 2 . 7 _ { \pm 1 . 9 }$ </td><td> $3 5 . 3 _ { \pm 3 . 4 }$ </td><td> $3 2 . 7 _ { \pm 4 . 1 }$ </td><td> $4 0 . 7 _ { \pm 0 . 9 }$ </td><td> $4 6 . 9 _ { \pm 0 . 8 }$ </td></tr><tr><td>Sub-JEPA</td><td> $9 4 . 7 _ { \pm 3 . 4 }$ </td><td> $4 1 . 3 _ { \pm 2 . 5 }$ </td><td> $4 2 . 0 _ { \pm 5 . 9 }$ </td><td> $3 6 . 7 _ { \pm 4 . 7 }$ </td><td> $3 3 . 3 _ { \pm 8 . 2 }$ </td><td> $3 4 . 0 _ { \pm 8 . 5 }$ </td><td> $3 2 . 7 _ { \pm 5 . 7 }$ </td><td> $3 6 . 0 _ { \pm 4 . 3 }$ </td><td> $3 0 . 7 _ { \pm 2 . 5 }$ </td><td> $4 2 . 4 _ { \pm 3 . 2 }$ </td></tr><tr><td>TC-LeWM</td><td> $7 4 . 7 _ { \pm 2 . 5 }$ </td><td> $3 4 . 0 _ { \pm 5 . 9 }$ </td><td> $3 4 . 0 _ { \pm 2 . 8 }$ </td><td> $3 8 . 0 _ { \pm 4 . 3 }$ </td><td> $3 0 . 0 _ { \pm 8 . 6 }$ </td><td> $3 8 . 7 _ { \pm 3 . 8 }$ </td><td> $3 3 . 3 _ { \pm 5 . 2 }$ </td><td> $3 9 . 3 _ { \pm 5 . 0 }$ </td><td> $3 7 . 3 _ { \pm 0 . 9 }$ </td><td> $3 9 . 9 _ { \pm 2 . 1 }$ </td></tr><tr><td>EB-JEPA</td><td> $\mathbf { 1 0 0 . 0 } _ { \pm 0 . 0 }$ </td><td> $4 8 . 0 _ { \pm 4 . 3 }$ </td><td> $5 2 . 7 _ { \pm 0 . 9 }$ </td><td> $4 5 . 3 _ { \pm 3 . 4 }$ </td><td> $4 0 . 7 _ { \pm 2 . 5 }$ </td><td> $4 7 . 3 _ { \pm 6 . 6 }$ </td><td> $5 0 . 7 _ { \pm 6 . 6 }$ </td><td> $5 0 . 0 _ { \pm 4 . 3 }$ </td><td> $5 6 . 7 _ { \pm 0 . 9 }$ </td><td> $5 4 . 6 _ { \pm 2 . 6 }$ </td></tr><tr><td>DINOv2-WM</td><td> $9 8 . 0 _ { \pm 1 . 6 }$ </td><td> $9 3 . 3 _ { \pm 1 . 9 }$ </td><td> $8 9 . 3 _ { \pm 1 . 9 }$ </td><td> $6 8 . 7 _ { \pm 0 . 9 }$ </td><td> $6 2 . 7 _ { \pm 0 . 9 }$ </td><td> $6 2 . 7 _ { \pm 2 . 5 }$ </td><td> $4 6 . 7 _ { \pm 1 2 . 3 }$ </td><td> $5 1 . 3 _ { \pm 9 . 6 }$ </td><td> $5 2 . 7 _ { \pm 1 . 9 }$ </td><td> $6 9 . 5 _ { \pm 2 . 5 }$ </td></tr><tr><td>ConfigWM</td><td> $9 7 . 3 _ { \pm 0 . 9 }$ </td><td> ${ \bf 9 7 . 3 _ { \pm 2 . 5 } }$ </td><td> ${ \mathbf { 9 7 . 3 _ { \pm 0 . 9 } } }$ </td><td> $9 8 . 0 _ { \pm 1 . 6 }$ </td><td> $\mathbf { 9 7 . 3 _ { \pm 2 . 5 } }$ </td><td> $9 6 . 0 _ { \pm 2 . 8 }$ </td><td> $9 6 . 7 _ { \pm 2 . 5 }$ </td><td> $9 7 . 3 _ { \pm 2 . 5 }$ </td><td> $9 4 . 0 _ { \pm 1 . 6 }$ </td><td> $9 6 . 8 _ { \pm 1 . 8 }$ </td></tr><tr><td>Unified ConfigWM</td><td> $9 6 . 0 _ { \pm 3 . 3 }$ </td><td> ${ \bf 9 7 . 3 _ { \pm 2 . 5 } }$ </td><td> ${ \mathbf { 9 7 . 3 _ { \pm 0 . 9 } } }$ </td><td> $9 6 . 7 _ { \pm 1 . 9 }$ </td><td> $9 4 . 7 _ { \pm 3 . 4 }$ </td><td> ${ \mathfrak { s } } 7 . 3 _ { \pm 1 . 9 }$ </td><td> $\mathbf { 9 7 . 3 _ { \pm 2 . 5 } }$ </td><td> $\mathbf { 9 8 . 0 } _ { \pm 1 . 6 }$ </td><td> ${ \bf 9 5 . 3 _ { \pm 4 . 1 } }$ </td><td> $9 6 . 7 _ { \pm 2 . 4 }$ </td></tr><tr><td colspan="10">Push-T</td></tr><tr><td>Random Policy</td><td></td><td></td><td></td><td></td><td> $4 . 0 _ { \pm 4 . 3 }$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LeWM</td><td> ${ \mathbf { 8 1 . 3 _ { \pm 5 . 7 } } }$ </td><td> $4 . 0 _ { \pm 2 . 8 }$ </td><td> $7 . 3 _ { \pm 2 . 5 }$ </td><td> $4 . 7 _ { \pm 2 . 5 }$ </td><td> $6 . 0 _ { \pm 4 . 3 }$ </td><td> $6 . 7 _ { \pm 2 . 5 }$ </td><td> $5 . 3 _ { \pm 1 . 9 }$ </td><td> $2 . 0 _ { \pm 0 . 0 }$ </td><td> $6 . 0 _ { \pm 1 . 6 }$ </td><td> $1 3 . 7 _ { \pm 1 . 0 }$ </td></tr><tr><td>Sub-JEPA</td><td> $3 9 . 3 _ { \pm 1 . 9 }$ </td><td> $4 . 7 _ { \pm 0 . 9 }$ </td><td> $4 . 0 _ { \pm 1 . 6 }$ </td><td> $3 . 3 _ { \pm 0 . 9 }$ </td><td> $2 . 7 _ { \pm 2 . 5 }$ </td><td> $3 . 3 _ { \pm 0 . 9 }$ </td><td> $5 . 3 _ { \pm 3 . 4 }$ </td><td> $2 . 0 _ { \pm 0 . 0 }$ </td><td> $2 . 7 _ { \pm 0 . 9 }$ </td><td> $7 . 5 _ { \pm 0 . 8 }$ </td></tr><tr><td>TC-LeWM</td><td> $3 4 . 0 _ { \pm 1 . 6 }$ </td><td> $3 0 . 7 _ { \pm 0 . 9 }$ </td><td> $3 0 . 7 _ { \pm 3 . 4 }$ </td><td> $4 . 0 _ { \pm 2 . 8 }$ </td><td> $4 . 0 _ { \pm 2 . 8 }$ </td><td> $4 . 0 _ { \pm 3 . 3 }$ </td><td> $4 . 7 _ { \pm 3 . 4 }$ </td><td> $4 . 0 _ { \pm 1 . 6 }$ </td><td> $6 . 0 _ { \pm 1 . 6 }$ </td><td> $1 3 . 6 _ { \pm 1 . 5 }$ </td></tr><tr><td>DINOv2-WM</td><td> $5 8 . 7 _ { \pm 0 . 9 }$ </td><td> $1 0 . 7 _ { \pm 1 . 9 }$ </td><td> $8 . 0 _ { \pm 3 . 3 }$ </td><td> $6 . 7 _ { \pm 1 . 9 }$ </td><td> $4 . 0 _ { \pm 0 . 0 }$ </td><td> $3 . 3 _ { \pm 0 . 9 }$ </td><td> $4 . 0 _ { \pm 1 . 6 }$ </td><td> $3 . 3 _ { \pm 0 . 9 }$ </td><td> $4 . 0 _ { \pm 1 . 6 }$ </td><td> $1 1 . 4 _ { \pm 0 . 6 }$ </td></tr><tr><td>ConfigWM</td><td> $4 9 . 3 _ { \pm 4 . 1 }$ </td><td> $4 6 . 7 _ { \pm 0 . 9 }$ </td><td> $3 9 . 3 _ { \pm 4 . 1 }$ </td><td> $4 2 . 7 _ { \pm 3 . 4 }$ </td><td> $3 5 . 3 _ { \pm 6 . 6 }$ </td><td> $4 0 . 7 _ { \pm 5 . 2 }$ </td><td> ${ \bf 3 6 . 0 _ { \pm 2 . 8 } }$ </td><td> $3 7 . 3 _ { \pm 4 . 1 }$ </td><td> ${ 3 6 . 0 } _ { \pm 4 . 9 }$ </td><td> $4 0 . 4 _ { \pm 2 . 7 }$ </td></tr><tr><td>Unified ConfigWM</td><td> $5 2 . 7 _ { \pm 6 . 6 }$ </td><td> $4 8 . 7 _ { \pm 2 . 5 }$ </td><td> ${ \bf 5 0 . 7 _ { \pm 6 . 6 } }$ </td><td> ${ 5 4 . 7 } _ { \pm 3 . 4 }$ </td><td> ${ \bf 5 0 . 7 _ { \pm 7 . 5 } }$ </td><td> ${ \bf 4 1 . 3 _ { \pm 6 . 8 } }$ </td><td> $3 3 . 3 _ { \pm 8 . 1 }$ </td><td> $3 3 . 3 _ { \pm 5 . 0 }$ </td><td> $2 5 . 3 _ { \pm 0 . 9 }$ </td><td> $4 3 . 4 _ { \pm 4 . 6 }$ </td></tr><tr><td colspan="10">OGBench-Cube</td></tr><tr><td>Random Policy</td><td></td><td></td><td></td><td></td><td> $5 3 . 3 _ { \pm 6 . 2 }$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LeWM</td><td> $8 0 . 7 _ { \pm 6 . 2 }$ </td><td> $4 6 . 0 _ { \pm 4 . 3 }$ </td><td> $5 0 . 0 _ { \pm 5 . 7 }$ </td><td> $4 7 . 3 _ { \pm 3 . 8 }$ </td><td> $4 9 . 3 _ { \pm 4 . 1 }$ </td><td> $4 8 . 0 _ { \pm 7 . 5 }$ </td><td> $5 0 . 0 _ { \pm 1 . 6 }$ </td><td> $4 4 . 0 _ { \pm 1 . 6 }$ </td><td> $5 2 . 7 _ { \pm 6 . 6 }$ </td><td> $5 2 . 0 _ { \pm 4 . 4 }$ </td></tr><tr><td>Sub-JEPA</td><td> $6 6 . 7 _ { \pm 5 . 0 }$ </td><td> $5 0 . 0 _ { \pm 2 . 8 }$ </td><td> $4 8 . 0 _ { \pm 5 . 9 }$ </td><td> $5 0 . 7 _ { \pm 6 . 6 }$ </td><td> $4 7 . 3 _ { \pm 2 . 5 }$ </td><td> $4 8 . 0 _ { \pm 2 . 8 }$ </td><td> $4 6 . 7 _ { \pm 2 . 5 }$ </td><td> $4 9 . 3 _ { \pm 4 . 1 }$ </td><td> $4 5 . 3 _ { \pm 2 . 5 }$ </td><td> $5 0 . 2 _ { \pm 2 . 7 }$ </td></tr><tr><td> $\mathrm { T C - L e W M }$ </td><td> $6 2 . 0 _ { \pm 8 . 6 }$ </td><td> $5 5 . 3 _ { \pm 6 . 8 }$ </td><td> $5 2 . 0 _ { \pm 8 . 2 }$ </td><td> $5 5 . 3 _ { \pm 9 . 0 }$ </td><td> $5 6 . 7 _ { \pm 9 . 0 }$ </td><td> $5 4 . 0 _ { \pm 9 . 1 }$ </td><td> $5 1 . 3 _ { \pm 5 . 7 }$ </td><td> $5 2 . 7 _ { \pm 1 . 9 }$ </td><td> $5 5 . 3 _ { \pm 1 0 . 9 }$ </td><td> $5 5 . 0 _ { \pm 7 . 4 }$ </td></tr><tr><td>DINOv2-WM</td><td> $8 1 . 3 _ { \pm 5 . 7 }$ </td><td> $5 2 . 7 _ { \pm 4 . 1 }$ </td><td> $5 1 . 3 _ { \pm 6 . 8 }$ </td><td> $5 2 . 7 _ { \pm 5 . 7 }$ </td><td> $5 0 . 0 _ { \pm 3 . 3 }$ </td><td> $5 0 . 7 _ { \pm 4 . 7 }$ </td><td> $5 0 . 0 _ { \pm 3 . 3 }$ </td><td> $5 4 . 0 _ { \pm 2 . 8 }$ </td><td> $5 3 . 3 _ { \pm 5 . 7 }$ </td><td> $5 5 . 1 _ { \pm 4 . 3 }$ </td></tr><tr><td>ConfigWM</td><td> ${ \mathbf { 8 3 . 3 _ { \pm 8 . 4 } } }$ </td><td> ${ \mathbf { 7 4 . 0 _ { \pm 5 . 9 } } }$ </td><td> ${ \bf 6 7 . 3 _ { \pm 8 . 4 } }$ </td><td> ${ \bf 6 4 . 7 _ { \pm 1 0 . 5 } }$ </td><td> ${ \bf 6 8 . 0 _ { \pm 9 . 9 } }$ </td><td> ${ \bf 6 8 . 0 _ { \pm 8 . 6 } }$ </td><td> $6 4 . 7 _ { \pm 1 1 . 8 }$ </td><td> ${ \bf 6 5 . 3 _ { \pm 1 0 . 9 } }$ </td><td> ${ \bf 6 8 . 0 } _ { \pm 8 . 5 }$ </td><td> ${ \bf 6 9 . 3 _ { \pm 9 . 1 } }$ </td></tr><tr><td>Unified  $\mathbf { C o n f i g W M }$ </td><td> $7 6 . 0 _ { \pm 1 . 6 }$ </td><td> $6 1 . 3 _ { \pm 8 . 1 }$ </td><td> $5 8 . 7 _ { \pm 7 . 7 }$ </td><td> $5 9 . 3 _ { \pm 6 . 8 }$ </td><td> $5 8 . 0 _ { \pm 4 . 3 }$ </td><td> $5 4 . 0 _ { \pm 7 . 1 }$ </td><td> $5 5 . 3 _ { \pm 6 . 8 }$ </td><td> $5 5 . 3 _ { \pm 1 . 9 }$ </td><td> $5 7 . 3 _ { \pm 9 . 6 }$ </td><td> $5 9 . 5 _ { \pm 5 . 5 }$ </td></tr></table>

![](images/55228a4fe2c8c346b02a5106c00bc68ee74cc7f1c23eb8038ec0e58d8ecdb478.jpg)  
Figure 12. Matched planning performance across distractor levels. Success rate $( \% )$ under the matched setting, where each model is trained and evaluated at the same distractor level $( d _ { k } \to d _ { k } )$ . Results are averaged over seeds $4 0 / 4 2 / 4 5 ,$ , with shaded regions indicating standard deviation.

Table 4. Clean-trained transfer performance. Each model uses its �<sub>0</sub>-trained checkpoint and is evaluated across $d _ { 0 } { \mathrm { - } } d _ { 8 } ( d _ { 0 }  d _ { k } )$ . Values follow the same reporting conventions as Table 3.
<table><tr><td rowspan="2">Method</td><td colspan="9">Evaluation Distractor Level</td><td rowspan="2">Average</td></tr><tr><td></td><td> $d _ { 1 }$ </td><td> $d _ { 2 }$ </td><td> $d _ { 3 }$ </td><td> $d _ { 4 }$ </td><td> $d _ { 5 }$ </td><td> $d _ { 6 }$ </td><td> $d _ { 7 }$ </td><td> $d _ { 8 }$ </td></tr><tr><td colspan="10">Two-Room</td></tr><tr><td>Random Policy</td><td colspan="10"></td></tr><tr><td>LeWM</td><td> $\mathbf { 1 0 0 . 0 } _ { \pm 0 . 0 }$ </td><td> $7 2 . 7 _ { \pm 1 . 9 }$ </td><td> $6 6 . 0 _ { \pm 4 . 9 }$ </td><td> $5 8 . 7 _ { \pm 5 . 2 }$ </td><td> $3 0 . 0 _ { \pm 4 . 9 }$   $5 1 . 3 _ { \pm 5 . 0 }$ </td><td> $5 0 . 7 _ { \pm 3 . 4 }$ </td><td> $5 6 . 7 _ { \pm 6 . 6 }$ </td><td> $5 2 . 0 _ { \pm 7 . 5 }$ </td><td> $5 0 . 0 _ { \pm 5 . 7 }$ </td><td> $6 2 . 0 _ { \pm 3 . 3 }$ </td></tr><tr><td>Sub-JEPA</td><td> $9 4 . 7 _ { \pm 3 . 4 }$ </td><td> $6 0 . 7 _ { \pm 1 . 9 }$ </td><td> $4 5 . 3 _ { \pm 1 2 . 5 }$ </td><td> $5 3 . 3 _ { \pm 7 . 4 }$ </td><td> $4 5 . 3 _ { \pm 3 . 4 }$ </td><td> $4 4 . 0 _ { \pm 5 . 7 }$ </td><td> $4 4 . 7 _ { \pm 6 . 6 }$ </td><td> $4 6 . 7 _ { \pm 7 . 5 }$ </td><td> $4 6 . 0 _ { \pm 7 . 5 }$ </td><td> $5 3 . 4 _ { \pm 4 . 4 }$ </td></tr><tr><td>TC-LeWM</td><td> $7 4 . 7 _ { \pm 2 . 5 }$ </td><td> $4 3 . 3 _ { \pm 0 . 9 }$ </td><td> $3 8 . 7 _ { \pm 7 . 4 }$ </td><td> $3 2 . 7 _ { \pm 4 . 1 }$ </td><td> $3 4 . 7 _ { \pm 0 . 9 }$ </td><td> $3 0 . 0 _ { \pm 5 . 9 }$ </td><td> $3 1 . 3 _ { \pm 8 . 1 }$ </td><td> $3 1 . 3 _ { \pm 5 . 7 }$ </td><td> $2 7 . 3 _ { \pm 1 . 9 }$ </td><td> $3 8 . 2 _ { \pm 0 . 5 }$ </td></tr><tr><td>EB-JEPA</td><td> $\mathbf { 1 0 0 . 0 } _ { \pm 0 . 0 }$ </td><td> $5 8 . 0 _ { \pm 7 . 1 }$ </td><td> $4 3 . 3 _ { \pm 1 2 . 3 }$ </td><td> $4 3 . 3 _ { \pm 5 . 0 }$ </td><td> $4 2 . 7 _ { \pm 5 . 0 }$ </td><td> $4 2 . 7 _ { \pm 2 . 5 }$ </td><td> $3 7 . 3 _ { \pm 2 . 5 }$ </td><td> $3 8 . 0 _ { \pm 5 . 9 }$ </td><td> $3 8 . 0 _ { \pm 3 . 3 }$ </td><td> $4 9 . 3 _ { \pm 4 . 0 }$ </td></tr><tr><td>DINOv2-WM</td><td> $9 8 . 0 _ { \pm 1 . 6 }$ </td><td> $4 7 . 3 _ { \pm 6 . 8 }$ </td><td> $4 6 . 0 _ { \pm 3 . 3 }$ </td><td> $3 5 . 3 _ { \pm 9 . 3 }$ </td><td> $3 6 . 0 _ { \pm 2 . 8 }$ </td><td> $3 8 . 0 _ { \pm 4 . 9 }$ </td><td> $3 8 . 7 _ { \pm 4 . 7 }$ </td><td> $4 4 . 7 _ { \pm 3 . 4 }$ </td><td> $4 0 . 7 _ { \pm 5 . 0 }$ </td><td> $4 7 . 2 _ { \pm 3 . 2 }$ </td></tr><tr><td>ConfigWM</td><td> $9 7 . 3 _ { \pm 0 . 9 }$ </td><td> ${ \bf 9 8 . 0 _ { \pm 1 . 6 } }$ </td><td> ${ \bf 9 8 . 0 _ { \pm 1 . 6 } }$ </td><td> ${ \bf 9 6 . 7 _ { \pm 0 . 9 } }$ </td><td> $\mathbf { 9 8 . 0 _ { \pm 1 . 6 } }$ </td><td> ${ \bf 9 8 . 0 _ { \pm 1 . 6 } }$ </td><td> ${ \bf 9 8 . 0 _ { \pm 1 . 6 } }$ </td><td> ${ \bf 9 7 . 3 _ { \pm 0 . 9 } }$ </td><td> ${ \bf 9 6 . 7 _ { \pm 0 . 9 } }$ </td><td> ${ \bf 9 7 . 6 } _ { \pm 1 . 2 }$ </td></tr><tr><td>Unified ConfigWM</td><td> $9 6 . 0 _ { \pm 3 . 3 }$ </td><td> $9 6 . 7 _ { \pm 3 . 4 }$ </td><td> $9 6 . 0 _ { \pm 4 . 3 }$ </td><td> $9 6 . 0 _ { \pm 2 . 8 }$ </td><td> $9 6 . 0 _ { \pm 2 . 8 }$ </td><td> $9 4 . 0 _ { \pm 1 . 6 }$ </td><td> $9 4 . 7 _ { \pm 3 . 4 }$ </td><td> $9 4 . 7 _ { \pm 3 . 4 }$ </td><td> $9 5 . 3 _ { \pm 2 . 5 }$ </td><td> $9 5 . 5 _ { \pm 3 . 0 }$ </td></tr><tr><td colspan="10">Push-T</td></tr><tr><td>Random Policy</td><td colspan="3"></td><td colspan="3"></td><td></td><td></td><td></td><td></td></tr><tr><td>LeWM</td><td> ${ \mathbf { 8 1 . 3 _ { \pm 5 . 7 } } }$ </td><td> $2 2 . 0 _ { \pm 3 . 3 }$ </td><td> $1 5 . 3 _ { \pm 3 . 4 }$ </td><td> $1 1 . 3 _ { \pm 1 . 9 }$ </td><td> $8 . 7 _ { \pm 2 . 5 }$ </td><td> $4 . 0 _ { \pm 4 . 3 }$   $7 . 3 _ { \pm 0 . 9 }$ </td><td> $9 . 3 _ { \pm 0 . 9 }$ </td><td> $6 . 7 _ { \pm 0 . 9 }$ </td><td> $2 . 7 _ { \pm 0 . 9 }$ </td><td> $1 8 . 3 _ { \pm 0 . 4 }$ </td></tr><tr><td>Sub-JEPA</td><td> $3 9 . 3 _ { \pm 1 . 9 }$ </td><td> $5 . 3 _ { \pm 3 . 4 }$ </td><td> $4 . 0 _ { \pm 2 . 8 }$ </td><td> $3 . 3 _ { \pm 2 . 5 }$ </td><td> $3 . 3 _ { \pm 0 . 9 }$ </td><td> $2 . 0 _ { \pm 1 . 6 }$ </td><td> $3 . 3 _ { \pm 0 . 9 }$ </td><td> $4 . 0 _ { \pm 1 . 6 }$ </td><td> $2 . 0 _ { \pm 1 . 6 }$ </td><td> $7 . 4 _ { \pm 0 . 4 }$ </td></tr><tr><td>TC-LeWM</td><td> $3 4 . 0 _ { \pm 1 . 6 }$ </td><td> $2 7 . 3 _ { \pm 5 . 0 }$ </td><td> $2 6 . 7 _ { \pm 3 . 4 }$ </td><td> $2 3 . 3 _ { \pm 3 . 8 }$ </td><td> $2 2 . 7 _ { \pm 5 . 0 }$ </td><td> $2 4 . 7 _ { \pm 1 . 9 }$ </td><td> $2 3 . 3 _ { \pm 4 . 1 }$ </td><td> $2 2 . 0 _ { \pm 3 . 3 }$ </td><td> $2 0 . 0 _ { \pm 7 . 5 }$ </td><td> $2 4 . 9 _ { \pm 3 . 6 }$ </td></tr><tr><td>DINOv2-WM</td><td> $5 8 . 7 _ { \pm 0 . 9 }$ </td><td> $1 2 . 0 _ { \pm 3 . 3 }$ </td><td> $2 . 7 _ { \pm 0 . 9 }$ </td><td> $5 . 3 _ { \pm 1 . 9 }$ </td><td> $2 . 7 _ { \pm 1 . 9 }$ </td><td> $2 . 7 _ { \pm 0 . 9 }$ </td><td> $1 . 3 _ { \pm 0 . 9 }$ </td><td> $1 . 3 _ { \pm 0 . 9 }$ </td><td> $2 . 0 _ { \pm 1 . 6 }$ </td><td> $9 . 9 _ { \pm 0 . 7 }$ </td></tr><tr><td>ConfigWM</td><td> $4 9 . 3 _ { \pm 4 . 1 }$ </td><td> $4 1 . 3 _ { \pm 7 . 4 }$ </td><td> $4 0 . 0 _ { \pm 2 . 8 }$ </td><td> $4 3 . 3 _ { \pm 3 . 4 }$ </td><td> $4 0 . 7 _ { \pm 2 . 5 }$ </td><td> $4 1 . 3 _ { \pm 0 . 9 }$ </td><td> $4 1 . 3 _ { \pm 6 . 2 }$ </td><td> $3 6 . 7 _ { \pm 3 . 4 }$ </td><td> $3 6 . 7 _ { \pm 2 . 5 }$ </td><td> $4 1 . 2 _ { \pm 2 . 5 }$ </td></tr><tr><td>Unified ConfigWM</td><td> $5 3 . 3 _ { \pm 6 . 2 }$ </td><td> ${ \bar { 5 } } 2 . { \bf 0 } _ { \pm 5 . 7 }$ </td><td> $\pm 8 . 0 _ { \pm 4 . 9 }$ </td><td> ${ \bf 5 1 . 3 _ { \pm 4 . 1 } }$ </td><td> ${ 5 4 . 7 } _ { \pm 4 . 1 }$ </td><td> ${ \pm } 5 5 . 3 _ { \pm 0 . 9 }$ </td><td> ${ \bf 5 0 . 7 _ { \pm 6 . 6 } }$ </td><td> ${ \bf 5 0 . 7 _ { \pm 6 . 6 } }$ </td><td> ${ \bf 4 0 . 7 } _ { \pm 5 . 2 }$ </td><td> ${ \bf 5 0 . 7 _ { \pm 4 . 0 } }$ </td></tr><tr><td colspan="10">OGBench-Cube</td></tr><tr><td>Random Policy</td><td colspan="3"></td><td colspan="3"></td><td colspan="3"></td><td></td></tr><tr><td>LeWM</td><td> $8 0 . 7 _ { \pm 6 . 2 }$ </td><td> $4 5 . 3 _ { \pm 4 . 1 }$ </td><td> $4 7 . 3 _ { \pm 4 . 1 }$ </td><td> $4 7 . 3 _ { \pm 6 . 8 }$ </td><td> $4 6 . 7 _ { \pm 4 . 7 }$ </td><td> $4 9 . 3 _ { \pm 3 . 8 }$ </td><td> $4 7 . 3 _ { \pm 2 . 5 }$ </td><td> $4 6 . 0 _ { \pm 4 . 9 }$ </td><td> $4 4 . 7 _ { \pm 5 . 0 }$ </td><td> $5 0 . 5 _ { \pm 4 . 6 }$ </td></tr><tr><td>Sub-JEPA</td><td> $6 6 . 7 _ { \pm 5 . 0 }$ </td><td> $4 5 . 3 _ { \pm 3 . 8 }$ </td><td> $5 0 . 0 _ { \pm 7 . 5 }$ </td><td> $5 4 . 0 _ { \pm 5 . 9 }$ </td><td> $4 7 . 3 _ { \pm 5 . 2 }$ </td><td> $4 8 . 0 _ { \pm 1 . 6 }$ </td><td> $4 5 . 3 _ { \pm 2 . 5 }$ </td><td> $4 8 . 7 _ { \pm 3 . 4 }$ </td><td> $4 6 . 0 _ { \pm 3 . 3 }$ </td><td> $5 0 . 1 _ { \pm 3 . 9 }$ </td></tr><tr><td> $\mathrm { T C - L e W M }$ </td><td> $6 2 . 0 _ { \pm 8 . 6 }$ </td><td> $5 2 . 7 _ { \pm 5 . 0 }$ </td><td> $5 0 . 7 _ { \pm 4 . 1 }$ </td><td> $5 0 . 7 _ { \pm 8 . 2 }$ </td><td> $5 1 . 3 _ { \pm 4 . 1 }$ </td><td> $4 6 . 0 _ { \pm 5 . 9 }$ </td><td> $4 9 . 3 _ { \pm 5 . 2 }$ </td><td> $5 0 . 7 _ { \pm 7 . 5 }$ </td><td> $4 8 . 0 _ { \pm 2 . 8 }$ </td><td> $5 1 . 3 _ { \pm 5 . 5 }$ </td></tr><tr><td>DINOv2-WM</td><td> $8 1 . 3 _ { \pm 5 . 7 }$ </td><td> $5 4 . 0 _ { \pm 4 . 9 }$ </td><td> $5 3 . 3 _ { \pm 6 . 8 }$ </td><td> $5 0 . 7 _ { \pm 3 . 4 }$ </td><td> $5 0 . 7 _ { \pm 4 . 1 }$ </td><td> $5 2 . 7 _ { \pm 3 . 4 }$ </td><td> $5 4 . 7 _ { \pm 7 . 4 }$ </td><td> $5 5 . 3 _ { \pm 5 . 2 }$ </td><td> $5 6 . 7 _ { \pm 6 . 2 }$ </td><td> $5 6 . 6 _ { \pm 4 . 8 }$ </td></tr><tr><td>ConfigWM</td><td> ${ \mathbf { 8 3 . 3 _ { \pm 8 . 4 } } }$ </td><td> ${ \bf 7 4 . 0 _ { \pm 9 . 1 } }$ </td><td> ${ \bf 6 8 . 7 _ { \pm 1 0 . 0 } }$ </td><td> ${ \bf 6 8 . 0 _ { \pm 8 . 2 } }$ </td><td> ${ \bf 6 6 . 7 _ { \pm 7 . 5 } }$ </td><td> ${ \bf 6 5 . 3 _ { \pm 1 0 . 9 } }$ </td><td> ${ \bf 6 8 . 7 _ { \pm 6 . 2 } }$ </td><td> ${ \bf 6 7 . 3 _ { \pm 1 0 . 9 } }$ </td><td> ${ \bf 6 8 . 7 _ { \pm 1 0 . 4 } }$ </td><td> ${ \bf 7 0 . 1 _ { \pm 8 . 7 } }$ </td></tr><tr><td>Unified ConfigWM</td><td> $7 4 . 7 _ { \pm 7 . 7 }$ </td><td> $6 0 . 0 _ { \pm 9 . 8 }$ </td><td> $5 8 . 0 _ { \pm 4 . 3 }$ </td><td> $6 0 . 7 _ { \pm 6 . 6 }$ </td><td> $6 0 . 7 _ { \pm 5 . 0 }$ </td><td> $5 8 . 7 _ { \pm 8 . 2 }$ </td><td> $5 8 . 0 _ { \pm 5 . 7 }$ </td><td> $5 8 . 7 _ { \pm 5 . 0 }$ </td><td> $5 6 . 7 _ { \pm 8 . 2 }$ </td><td> $6 0 . 7 _ { \pm 6 . 6 }$ </td></tr></table>

![](images/da41b8eaa337f59de7871c97e170386d4914e89b9754891512fac647189a5d45.jpg)  
Figure 13. Clean-trained world-model transfer under dynamic distractors. World models are trained on � transitions and evaluated on � –� . For ConfigWM, this training restriction applies to the downstream state projector and dynamics model, and its configurable encoder is separately pretrained on noisy–clean pairs across distractor levels. Curves report mean success rates across seeds 40, 42, and 45. Higher is better.

Unified vs. Single-Domain ConfigWM. We further investigate whether a configurable encoder jointly trained across multiple environments, can achieve comparable performance when paired with environment-specific world models. Notably, as shown in Fig. 14, Unified ConfigWM achieves comparable planning performance to single-domain ConfigWM on Two-Room and Push-T. Although its performance is slightly lower on OGBench-Cube, it still outperforms all other baselines. Meanwhile, both Unified ConfigWM and single-domain ConfigWM maintain relatively stable physical-quantity probing performance across distractor levels. These results suggest that a shared configurable encoder can preserve task-relevant physical information across diferent domains while supporting downstream planning.

![](images/cabd1b461f0cad8282669be810d60537de8787f596c5058fda372b9ce72b9d64.jpg)  
Figure 14. Unified vs. Single-Domain ConfigWM. Comparison of planning success rates (top) and physical-quantity probing performance (bottom) across distractor levels. Unified ConfigWM shares a configurable encoder across environments while retaining environment-specific dynamics models

## C.2 Quantity-Wise Physical Probing Results

For each model and distractor level, we freeze the representation model and extract the 192-dimensional states supplied to the dynamics predictor. Episodes are split into disjoint probe-training and probe-test partitions using a fixed 90/10 split, from which we sample 1000 and 400 frames, respectively. The same split and sampled frames are used for all methods. For each physical quantity, we standardize the representations and targets using probe-training statistics and fit a separate linear probe using ridge regression with a coeficient of $1 0 ^ { - 6 }$ . We report Pearson correlation � on the probe-test set, averaging across coordinates for vector-valued quantities. The macro-averaged results are shown in Fig. 6, with quantity-wise results reported below.

![](images/27a835943fa0ad2c863ca51e62cd0f7ed14bdfef7435a7508f06d1a03e416ea6.jpg)  
Figure 15. Quantity-wise physical probing on Two-Room and Push-T. Each panel reports the Pearson correlation (�) of a linear probe trained on frozen representations for one task-relevant physical quantity across distractor levels � –� . Higher is better.

![](images/b563c64db15e35b5e030cf11ed783951d1d0ca83b8360b972becbbd2a80896e9.jpg)  
Figure 16. Quantity-wise physical probing on OGBench-Cube: cube state. Pearson correlation (�) is reported across distractor levels �<sub>0</sub>–�<sub>8</sub> for cube position, quaternion, and yaw. Higher is better.

![](images/1b68a587aa0e9b6d302f59443f97322f0f110fd2eb27219315d42badd87d03fa.jpg)  
Figure 17. Quantity-wise physical probing on OGBench-Cube. Pearson correlation (�) is reported across distractor levels $d _ { 0 } - d _ { 8 } .$ Top: end-efector position, end-efector yaw, and gripper contact. Bottom: gripper opening, joint position, and joint velocity. Higher is better.

## C.3 Detailed Clean–Distracted Retrieval Results

For each model and distractor level, we sample 400 paired distracted and clean predictor-input states from matching episode and frame positions. Each distracted state is used as a query over the 400 clean candidates, which are ranked by cosine similarity. Top-� retrieval is successful when the corresponding clean state appears among the top-� candidates. We also report mean reciprocal rank, $\begin{array} { r } { \mathrm { M R R } = \frac { 1 } { 4 0 0 } \sum _ { i = 1 } ^ { 4 0 0 } 1 / \mathrm { r a n k } _ { i } , } \end{array}$ where rank� is the rank of the paired clean state for query �. The same sampled pairs are used for all methods. Top-1 results are reported in Fig. 6, with Top-5 and MRR results shown below.

![](images/9827ef30d14d46f73ebfdd24911f3d1a484e867dde6b194711c049aba625aaae.jpg)  
Figure 18. Detailed distracted-to-clean retrieval results. Top: Top-5 retrieval accuracy. Bottom: mean reciprocal rank (MRR). Each distracted state is queried against 400 clean candidates ranked by cosine similarity. Results are reported across distractor levels � –� ; higher is better.

## C.4 Detailed Query-Conditioning Results

Figure 19 presents additional decoding results obtained by varying only the task query while keeping the input observation fixed. For each environment, we freeze the configurable encoder and train one ViT patch decoder shared across all query variants. The decoder contains six Transformer blocks with width 768 and 12 attention heads and reconstructs 224 × 224 RGB observations from the predictor outputs. It is trained for one epoch with a global batch size of 256 in BF16. Because the decoder receives no direct access to the query, query-dependent reconstructions reflect information already encoded in the predictive state.

![](images/f8cc0aeb9d95d433d6810f896b5441d585c782bd13d26d8b4ed3a49457f9faaf.jpg)

![](images/7a6642ca028cdc416ebf9e1bf3e3bdc5ce647c10cd35c4bd8db45bfba2b91cef.jpg)

![](images/72de778166a1ac5d48de70715f21e0f1402d4b8dd6fa1298e6e5da43bc47ae52.jpg)  
Figure 19. Additional query-conditioned decoding results. From top to bottom, the three rows show Two-Room, Push-T, and OGBench-Cube. Within each environment, the input observation is held fixed while only the queried color is varied. The decoded observations show that diferent queries retain diferent visual content in the predictive state.

## C.5 Detailed Query-Conditioned Planning Results

To complement the aggregate query-mismatch results in Fig. 8, Table 5 reports performance at each distractor level, while Table 6 further decomposes the results by individual mismatched query. All experimental settings other than the task query are held fixed. The correct-query results are independently re-evaluated in this experiment; small diferences from Table 3 arise from stochastic CEM planning.

Table 5. Complete query-conditioned planning results. Mean success rate (%) over three seeds at each distractor level; light-blue tags show standard deviation. Mismatched results average eight valid queries within each seed. Drop is computed from the unrounded seed-level results.
<table><tr><td>Level</td><td>Correct query</td><td>Mismatched-query average</td><td>Drop</td></tr><tr><td colspan="4">Two-Room</td></tr><tr><td> $d _ { 1 }$ </td><td> $9 6 . 7 _ { \pm 3 . 4 }$ </td><td> $4 5 . 7 _ { \pm 2 . 3 }$ </td><td> $5 1 . 0 _ { \pm 5 . 5 }$ </td></tr><tr><td> $d _ { 2 }$ </td><td> $9 6 . 7 _ { \pm 1 . 9 }$ </td><td> $4 1 . 3 _ { \pm 3 . 7 }$ </td><td> $5 5 . 4 _ { \pm 4 . 3 }$ </td></tr><tr><td> $d _ { 3 }$ </td><td> $9 7 . 3 _ { \pm 1 . 9 }$ </td><td> $3 9 . 3 _ { \pm 3 . 0 }$ </td><td> $5 8 . 1 _ { \pm 4 . 9 }$ </td></tr><tr><td> $d _ { 4 }$ </td><td> $9 7 . 3 _ { \pm 1 . 9 }$ </td><td> $3 3 . 4 _ { \pm 2 . 6 }$ </td><td> $6 3 . 9 _ { \pm 4 . 5 }$ </td></tr><tr><td> $d _ { 5 }$ </td><td> $9 6 . 0 _ { \pm 2 . 8 }$ </td><td> $3 2 . 8 _ { \pm 3 . 7 }$ </td><td> $6 3 . 2 _ { \pm 4 . 6 }$ </td></tr><tr><td> $d _ { 6 }$ </td><td> $9 6 . 7 _ { \pm 2 . 5 }$ </td><td> $3 1 . 0 _ { \pm 2 . 4 }$ </td><td> $6 5 . 7 _ { \pm 4 . 5 }$ </td></tr><tr><td> $d _ { 7 }$ </td><td> $9 7 . 3 _ { \pm 2 . 5 }$ </td><td> $2 8 . 3 _ { \pm 2 . 1 }$ </td><td> $6 9 . 0 _ { \pm 3 . 6 }$ </td></tr><tr><td> $d _ { 8 }$ </td><td> $9 4 . 7 _ { \pm 2 . 5 }$ </td><td> $2 4 . 9 _ { \pm 2 . 9 }$ </td><td> $6 9 . 7 _ { \pm 5 . 3 }$ </td></tr><tr><td> $\operatorname { A v g } .$   $9 6 . 6 _ { \pm 2 . 1 }$ </td><td></td><td> $3 4 . 6 _ { \pm 2 . 7 }$ </td><td> $6 2 . 0 _ { \pm 4 . 5 }$ </td></tr><tr><td colspan="4">Push-T</td></tr><tr><td> $d _ { 1 }$ </td><td> $4 6 . 7 _ { \pm 0 . 9 }$ </td><td> $1 1 . 5 _ { \pm 0 . 5 }$ </td><td> $3 5 . 2 _ { \pm 0 . 4 }$ </td></tr><tr><td> $d _ { 2 }$ </td><td> $3 9 . 3 _ { \pm 4 . 1 }$ </td><td> $1 1 . 8 _ { \pm 3 . 9 }$ </td><td> $2 7 . 6 _ { \pm 8 . 0 }$ </td></tr><tr><td> $d _ { 3 }$ </td><td> $4 2 . 7 _ { \pm 3 . 4 }$ </td><td> $1 3 . 2 _ { \pm 3 . 5 }$ </td><td> $2 9 . 5 _ { \pm 5 . 6 }$ </td></tr><tr><td> $d _ { 4 }$ </td><td> $3 5 . 3 _ { \pm 6 . 6 }$ </td><td> $1 1 . 9 _ { \pm 1 . 8 }$ </td><td> $2 3 . 4 _ { \pm 6 . 5 }$ </td></tr><tr><td> $d _ { 5 }$ </td><td> $4 0 . 7 _ { \pm 5 . 2 }$ </td><td> $1 0 . 2 _ { \pm 2 . 2 }$ </td><td> $3 0 . 5 _ { \pm 3 . 7 }$ </td></tr><tr><td> $d _ { 6 }$ </td><td> $3 6 . 0 _ { \pm 2 . 8 }$ </td><td> $9 . 2 _ { \pm 1 . 5 }$ </td><td> $2 6 . 8 _ { \pm 1 . 5 }$ </td></tr><tr><td> $d _ { 7 }$ </td><td> $3 7 . 3 _ { \pm 4 . 1 }$ </td><td> $8 . 7 _ { \pm 1 . 8 }$ </td><td> $2 8 . 7 _ { \pm 3 . 7 }$ </td></tr><tr><td> $d _ { 8 }$ </td><td> $3 6 . 0 _ { \pm 4 . 9 }$ </td><td> $7 . 2 _ { \pm 1 . 7 }$ </td><td> $2 8 . 8 _ { \pm 3 . 9 }$ </td></tr><tr><td> $\operatorname { A v g } .$ </td><td> $3 9 . 2 _ { \pm 2 . 6 }$ </td><td> $1 0 . 4 _ { \pm 2 . 1 }$ </td><td> $2 8 . 8 _ { \pm 2 . 2 }$ </td></tr><tr><td colspan="4"></td></tr><tr><td> $d _ { 1 }$ </td><td> $7 4 . 0 _ { \pm 5 . 9 }$ </td><td>OGBench-Cube  $5 6 . 1 _ { \pm 5 . 6 }$ </td><td></td></tr><tr><td> $d _ { 2 }$ </td><td> $6 7 . 3 _ { \pm 8 . 4 }$ </td><td> $5 2 . 4 _ { \pm 4 . 6 }$ </td><td> $1 7 . 9 _ { \pm 2 . 5 }$   $1 4 . 9 _ { \pm 4 . 7 }$ </td></tr><tr><td> $d _ { 3 }$ </td><td> $6 4 . 7 _ { \pm 1 0 . 5 }$ </td><td> $5 0 . 4 _ { \pm 3 . 7 }$ </td><td> $1 4 . 2 _ { \pm 6 . 8 }$ </td></tr><tr><td> $d _ { 4 }$ </td><td> $6 8 . 0 _ { \pm 9 . 9 }$ </td><td> $4 8 . 3 _ { \pm 2 . 7 }$ </td><td> $1 9 . 7 _ { \pm 7 . 6 }$ </td></tr><tr><td> $d _ { 5 }$ </td><td> $6 8 . 0 _ { \pm 8 . 6 }$ </td><td> $4 7 . 8 _ { \pm 3 . 3 }$ </td><td> $2 0 . 2 _ { \pm 5 . 3 }$ </td></tr><tr><td> $d _ { 6 }$ </td><td> $6 4 . 7 _ { \pm 1 1 . 8 }$ </td><td> $4 8 . 6 _ { \pm 3 . 7 }$ </td><td> $1 6 . 1 _ { \pm 8 . 2 }$ </td></tr><tr><td> $d _ { 7 }$ </td><td> $6 5 . 3 _ { \pm 1 0 . 9 }$ </td><td> $4 8 . 6 _ { \pm 4 . 1 }$ </td><td> $1 6 . 7 _ { \pm 7 . 1 }$ </td></tr><tr><td> $d _ { 8 }$ </td><td> $6 8 . 0 _ { \pm 8 . 5 }$ </td><td> $4 6 . 9 _ { \pm 3 . 0 }$ </td><td> $2 1 . 1 _ { \pm 5 . 6 }$ </td></tr><tr><td> $\operatorname { A v g } .$ </td><td> $6 7 . 5 _ { \pm 9 . 2 }$ </td><td> $4 9 . 9 _ { \pm 3 . 7 }$ </td><td> $1 7 . 6 _ { \pm 5 . 5 }$ </td></tr></table>

Table 6. Per-query planning performance under mismatched queries. Mean success rate (%) over $d _ { 1 } \mathrm { - } d _ { 8 }$ and three seeds; light-blue tags show standard deviation. Drop is computed from the unrounded results.
<table><tr><td>Environment</td><td>Correct query</td><td>Mismatched query</td><td> $\operatorname { A v g . } \ d _ { 1 } - d _ { 8 }$ </td><td>Drop</td></tr><tr><td rowspan="7">Two-Room</td><td rowspan="7">Red ball</td><td>Blue ball Brown ball</td><td> $2 9 . 4 _ { \pm 0 . 7 }$ </td><td> $6 7 . 2 _ { \pm 1 . 7 }$ </td></tr><tr><td>Cyan ball</td><td> $3 2 . 4 _ { \pm 3 . 2 }$ </td><td> $6 4 . 2 _ { \pm 5 . 3 }$ </td></tr><tr><td></td><td> $2 9 . 8 _ { \pm 3 . 1 }$ </td><td> $6 6 . 8 _ { \pm 4 . 6 }$ </td></tr><tr><td>Green ball</td><td> $3 5 . 4 _ { \pm 4 . 7 }$ </td><td> $6 1 . 2 _ { \pm 6 . 3 }$ </td></tr><tr><td>Orange ball</td><td> $4 6 . 6 _ { \pm 2 . 6 }$ </td><td></td></tr><tr><td>Pink ball</td><td> $3 9 . 0 _ { \pm 6 . 8 }$ </td><td> $5 0 . 0 _ { \pm 4 . 7 }$   $5 7 . 6 _ { \pm 7 . 9 }$ </td></tr><tr><td>Purple ball</td><td> $3 1 . 9 _ { \pm 2 . 0 }$ </td><td> $6 4 . 7 _ { \pm 3 . 6 }$ </td></tr><tr><td rowspan="9">Push-T</td><td rowspan="9"></td><td>Yellow ball</td><td> $3 2 . 1 _ { \pm 1 . 6 }$ </td><td> $6 4 . 5 _ { \pm 3 . 7 }$ </td></tr><tr><td>Brown T block</td><td> $1 0 . 1 _ { \pm 4 . 1 }$ </td><td> $2 9 . 2 _ { \pm 2 . 9 }$ </td></tr><tr><td>Cyan T block</td><td></td><td></td></tr><tr><td>Navy T block</td><td> $1 0 . 4 _ { \pm 1 . 8 }$   $1 1 . 7 _ { \pm 0 . 7 }$ </td><td> $2 8 . 8 _ { \pm 0 . 8 }$ </td></tr><tr><td>Orange T block</td><td> $1 1 . 0 _ { \pm 2 . 4 }$ </td><td> $2 7 . 6 _ { \pm 2 . 6 }$ </td></tr><tr><td>Pink T block</td><td> $1 1 . 2 _ { \pm 4 . 4 }$ </td><td> $2 8 . 2 _ { \pm 0 . 6 }$   $2 8 . 0 _ { \pm 4 . 1 }$ </td></tr><tr><td>Purple T block</td><td> $1 0 . 2 _ { \pm 2 . 7 }$ </td><td> $2 9 . 1 _ { \pm 4 . 2 }$ </td></tr><tr><td>Red T block</td><td> $7 . 8 _ { \pm 1 . 4 }$ </td><td> $3 1 . 4 _ { \pm 2 . 6 }$ </td></tr><tr><td>Yellow T block</td><td> $1 1 . 2 _ { \pm 2 . 6 }$ </td><td> $2 8 . 1 _ { \pm 2 . 8 }$ </td></tr><tr><td rowspan="8"></td><td>Blue arm</td><td> $5 0 . 4 _ { \pm 3 . 0 }$ </td><td> $1 7 . 1 _ { \pm 6 . 4 }$ </td></tr><tr><td>Cyan arm</td><td></td><td></td></tr><tr><td>Green arm</td><td> $5 1 . 2 _ { \pm 3 . 8 }$   $5 0 . 7 _ { \pm 3 . 4 }$ </td><td> $1 6 . 3 _ { \pm 5 . 4 }$   $1 6 . 8 _ { \pm 5 . 9 }$ </td></tr><tr><td></td><td> $4 9 . 2 _ { \pm 3 . 9 }$ </td><td> $1 8 . 3 _ { \pm 5 . 7 }$ </td></tr><tr><td>Orange arm Pink arm</td><td> $4 8 . 0 _ { \pm 3 . 4 }$ </td><td> $1 9 . 5 _ { \pm 5 . 9 }$ </td></tr><tr><td>Purple arm</td><td> $5 0 . 8 _ { \pm 4 . 1 }$ </td><td> $1 6 . 7 _ { \pm 5 . 2 }$ </td></tr><tr><td>Red arm</td><td> $4 8 . 1 _ { \pm 4 . 8 }$ </td><td> $1 9 . 4 _ { \pm 5 . 5 }$ </td></tr><tr><td>Yellow arm</td><td> $5 0 . 8 _ { \pm 4 . 4 }$ </td><td> $1 6 . 7 _ { \pm 4 . 8 }$ </td></tr></table>

## C.6 Qualitative Planning Rollouts

We visualize complete real-environment rollouts under clean observations and severe distractors. For each environment, we compare ConfigWM with the best-performing baseline in the matched planning results: DINOv2-WM for Two-Room and OGBench-Cube, and LeWM for Push-T. Frames are sampled at normalized rollout times; the final column shows a condition-matched goal observation and is not part of the rollout.

![](images/0245fd340d08395a929320638a9c4c23c8e21027a5f9c01d70b73172afa9f47c.jpg)  
Figure 20. Qualitative planning rollouts on Two-Room and Push-T. Both methods complete the task under clean observations (� ). Under severe distractors (�<sub>8</sub>), ConfigWM reaches the goal, whereas the corresponding comparison model fails.

![](images/b693fd32014d48b00cbe8ec4937922b4f312a6db6728b11457db38e0490d6e92.jpg)  
Figure 21. Qualitative planning rollouts on OGBench-Cube. Both models complete the task under clean observations (� ). Under severe distractors (�<sub>8</sub>), ConfigWM reaches the goal, whereas DINOv2-WM fails. The yellow box identifies the controlled task view.

## D Scaling the configurable encoder with Real-World Data

To assess whether the configurable encoder extends beyond controlled environments, we conduct an additional representation-learning study using diverse real-world video data and four large visual backbones, demonstrating that the framework can be readily instantiated with a range of widely used visual encoders.

Real-World Data Construction. We draw candidate cases from task-annotated clips in public datasets spanning three domains: egocentric video (e.g., Ego4D [Grauman et al., 2022], Ego-Exo4D [Grauman et al., 2024], and EPIC-KITCHENS 100 [Damen et al., 2022]), robotic manipulation (e.g., DROID [Khazatsky et al., 2024], Open X-Embodiment [O’Neill et al., 2024], and RoboMind [Wu et al., 2024]), and instructional video (e.g., ActivityNet [Caba Heilbron et al., 2015], COIN [Tang et al., 2019], and FineGym [Shao et al., 2020]). For each case, we sample frames as the clean target observation and its task description as the query.

To construct paired noisy and clean observations, we use a task-conditioned distractor-synthesis pipeline based on VLM-guided image editing and diferential quality control. Given a clean observation and its task query, a VLM (Qwen3-VL-32B-Instruct [Bai et al., 2025]) identifies the task-critical entities and proposes one or two visually plausible but task-irrelevant objects. It then produces a structured edit instruction specifying the appearance and placement of each addition, which is executed by an image-editing model (Qwen-Image-Edit-2511 [Wu et al., 2025]) to obtain the noisy input observation. A second pass with Qwen3-VL-32B-Instruct compares the edited and clean observations and rejects edits that alter task-relevant content or scene geometry.

Training and Qualitative Evaluation. We retain the configurable encoder architecture and training objective described in Section B.1, and instantiate it with four large visual backbones: DINOv2-Giant [Oquab et al., 2023], DINOv3-7B [Siméoni et al., 2025], V-JEPA 2.1-Gigantic [Mur-Labadia et al., 2026], and LingBot-Vision-Giant [Fu et al., 2026b]. All variants produce a 64 × 64 patch grid. We train each variant for 1000 steps on 8× NVIDIA A100 GPUs using AdamW with a learning rate of 10<sup>−5</sup>, weight decay of 0<sub>.</sub>05, 50 warm-up steps followed by cosine decay, a global batch size of 32, BF16 precision, and gradient clipping at 1<sub>.</sub>0. For qualitative evaluation, we apply the trained models to held-out cases and visualize PCA projections of the frozen-backbone and our representations. The results below examine whether ours suppresses the injected distractors while preserving task-relevant states.

Use both hands to pick up a dinosaur on the right side and move it to the left side.  
![](images/474c94b48eae9ed2d1589340924700594952ce00be50567aef158b32fafbe1db.jpg)  
Figure 22. Configurable encoder across visual backbones. The configurable encoder suppresses feature regions associated with the injected distractors while retaining the task-relevant entities and spatial structure visible in the clean target. Six held-out cases are shown across DINOv2-Giant DINOv3-7B, V-JEPA2.1-Gigantic, and LingBot-Vision-Giant. Black dashed boxes mark distractors.

Quantitative Evaluation. We quantitatively evaluate whether the configurable encoder can efectively suppress distractors while preserving task-relevant states across diferent visual backbones. We use 300 cases, with 80% cases for training simple linear classifiers and 20% cases for testing. As shown in Table 7, the results suggest that the configurable encoder trained with diferent backbones consistently suppresses distractors while largely preserving task-relevant regions.

Table 7. Quantitative analysis of real-world representations across large visual backbones. The configurable encoder reduces distractor decodability while largely preserving task-relevant spatial information. We report patch-level linear-probing AP across DINOv2-Giant, DINOv3-7B, V-JEPA2.1- Gigantic, and LingBot-Vision-Giant.
<table><tr><td></td><td colspan="2">Frozen backbone</td><td colspan="2">The configurable encoder (Ours)</td></tr><tr><td>Backbone</td><td>Distractor AP ↓</td><td>Task-core AP ↑</td><td>Distractor AP ↓</td><td>Task-core AP ↑</td></tr><tr><td>DINOv2-Giant</td><td>0.523</td><td>0.603</td><td>0.377</td><td>0.594</td></tr><tr><td>DINOv3-7B</td><td>0.575</td><td>0.609</td><td>0.367</td><td>0.596</td></tr><tr><td>V-JEPA 2.1-Gigantic</td><td>0.544</td><td>0.605</td><td>0.376</td><td>0.602</td></tr><tr><td>LingBot-Vision-Giant</td><td>0.561</td><td>0.621</td><td>0.368</td><td>0.603</td></tr></table>

Data Construction Prompts. We provide the full prompts used to generate structured edit instructions and produce the corresponding noisy observations.

```jsonl
Edit-Instruction Generation Prompt Qwen3-VL-32B-Instruct
Clean Observation:
{clean_observation}
Task Description:
{task_description}
1. Objective. Plan synthetic visual distractors for training a model to recover the untouched clean input image. Given one image
and one task description, choose one or two new task-irrelevant objects and write an image-editing instruction. Return only valid
JSON matching the specified format.
2. Identify protected content. Identify all visible task-critical entities, including the target, destination, tool, acting hand or robot,
directly involved container, contact region, action path, and support surfaces needed to understand the task. All identified
entities must be protected.
3. Select task-irrelevant distractors. Every distractor must be unrelated to the task. Never add another instance, substitute,
semantic alternative, or look-alike of a task entity. Prefer an ordinary, scene-plausible object whose category, shape, color role,
and function are visibly distinct from all salient original objects. Use one distractor in dense scenes or when only one isolated
empty region exists. Use two only when the visible support contains two clearly separated empty regions.
4. Determine safe placements. Place additions on visible existing support surfaces in genuinely empty regions. They must not
overlap, touch, cover, replace, move, recolor, resize, or merge with any original entity. Never place an addition inside the task
area, between an actor and its target, or along the task motion path.
5. Construct the editing instruction. The edit_prompt must contain exactly three concise sentences:
(a) Start with “Add only these new task-irrelevant objects:” and specify the appearance and realistic size of each object.
(b) Start with “Place them separately” and specify the existing support surface and an unambiguous empty relative location
for each object. Explicitly forbid touching or overlapping any original object.
(c) Start with “Preserve the input image exactly” and require every original pixel and object to remain unchanged in
identity, count, color, pose, position, scale, contact, shadow, geometry, exposure, camera view, framing, and spatial
relationship.
6. Construct the negative prompt. The negative_prompt must forbid cropping, zooming, recentering, restaging, reconstruction,
deletion, replacement, relocation, recoloring, duplicate additions, unrequested objects, occlusion, overlap, floating objects, altered
support surfaces, altered hands or robots, and changed task relationships.
7. Return JSON only.
{
"task_entities": [
],
"protected_entities": [
],
"distractors": [
{
"name":
"appearance":
"task_irrelevance":
"support":
"placement":
}
],
"edit_prompt":
"negative_prompt":
}
```

<table><tr><td>Image-Editing Prompt</td><td>Qwen-Image-Edit-2511</td></tr><tr><td>Input image. {clean_observation}</td><td></td></tr><tr><td>1. Add the distractors. Add only these new task-irrelevant objects:</td><td></td></tr><tr><td>{distractor names, appearances, and realistic sizes}.</td><td></td></tr><tr><td>out touching or overlapping any original object.</td><td>2. Place the distractors. Place them separately at {support surfaces and relative locations produced by the VLM}, with-</td></tr><tr><td>3. Preserve the source observation. Preserve the input image exactly. Keep every original pixel and object unchanged in identity, count, color, pose, position, scale, contact, shadow, geometry, exposure, camera view, framing, and spatial relationship.</td><td></td></tr></table>