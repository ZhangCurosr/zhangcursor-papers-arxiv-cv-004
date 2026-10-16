# T2-Z: TEST-TIME OPTIMIZATION OF ENVIRONMENT LATENTS FOR ROBOTIC WORLD MODELS

Chenyuan Zhou<sup>1</sup>\*, Yining Hong<sup>1</sup>\*, Huang Huang<sup>1♡</sup>, Yunfan Jiang<sup>1♡</sup>, Wenlong Huang<sup>1♡</sup>, David Held<sup>2</sup>, Vitor Campagnolo Guizilini<sup>3</sup>, Yue Wang<sup>4</sup>, Li Fei-Fei<sup>1</sup>, Jiajun Wu<sup>1†</sup>, Leonidas Guibas<sup>1†</sup>, Yejin Choi<sup>1†</sup>

\* Co-First Contribution ♡ Co-Second Contribution † Equal Advising

<sup>1</sup>Stanford University <sup>2</sup>CMU <sup>3</sup>Toyota Research Institute <sup>4</sup>University of Southern California

 Website: https://t2-z-latent-optimization.github.io/ § Code: https://github.com/T2-z-latent-optimization/T2-z õ Data: htt<sub>p</sub>s://hu<sub>gg</sub>in<sub>g</sub>face.co/datasets/hua<sub>g</sub>ailuowen/T2-z-datasets

![](images/99645a3a3a465faf10cca9016bc6997f50abe5052612c8064d4eeeaf9fc310bb.jpg)

![](images/4016b8cbeb2ba29bd1be964a3135a8e14d35e25aac223bc237a1745d57fb072c.jpg)

![](images/67959eb81dfdbd5731d7e4f276c356d99178067cdc1f598a564870fd1956f6bc.jpg)

![](images/762c76f926d36c4376484da07a17af5cbd04138d1b800248794698a2f70d6cc3.jpg)  
Figure 1: T2-Z for Test-Time Environment Latent Optimization. (a) Given the same observation and action, T2-Z directly optimizes environment latents to capture different environmental conditions (e.g., ball mass), leading the world model to predict different outcomes. (b) Without knowing how many factors exist or what they represent, PCA of the learned environment latents reveals a two-dimensional structure that closely reflects the underlying factor distribution. (c) We evaluate T2-Z on real-world robotic tasks spanning diverse unobserved environmental factors. (d) Across diverse visual scenes, the learned latents reveal a unified structure that reflects the same underlying environmental factor without being entangled with visual appearance.

## ABSTRACT

Predicting how the world responds to an action requires more than what can be seen: robots must account for environmental factors that are neither directly observable nor explicitly specified, from physical properties such as friction and mass to human-defined environmental rules. We introduce T2-Z for test-time optimization of environment latents, which augments a robotic world model with a learnable environment latent to condition world predictions. This compact latent captures underlying environmental variation without supervision of what environmental factors exist or what values they take. During training, T2-Z alternates between two stages: a world-model optimization stage that fixes the la-

tents and learns general action-conditioned dynamics across environments; and an environment-latent optimization stage that freezes the world model and optimizes the conditioning latents, allowing latent structure to emerge purely through world prediction. At test time, the world model remains frozen and a newly initialized environment latent is optimized from one or more test-time interactions, adapting the model to previously unseen environmental configurations. Across 7 simulated settings and 4 real-world robotic tasks, T2-Z reduces object prediction error metric from 15.20 to 7.00 and raises action success from 33.73% to 72.96%. Its emergent latents capture structured representations of individual and compounded factors across diverse visual scenes. Ablations further show that T2-Z can efficiently infer environment latents from as little as a single test-time interaction trajectory.

## 1 INTRODUCTION

What a robot observes is insufficient to determine how the world will respond. Beneath visually similar scene configurations reside hidden environmental factors, such as friction, mass distribution, and unobserved human-defined interaction rules, that can fundamentally alter the consequences of the same action (fig. 1 (a)). Such factors are often revealed only through interaction with the target environments, as their influence becomes observable through the outcomes they induce. Explicitly representing these factors (Kumar et al., 2021), however, requires specifying in advance what factors may vary, their exact values, and how they should be parameterized, which is difficult to scale to unknown or compounded environmental variation. On the contrary, humans reason through “analy sis by synthesis” (Yuille & Kersten, 2006), using internal generative models not only to anticipate future outcomes, but also in reverse to infer latent structure that best accounts for what they observe. This raises a natural question: can hidden environmental factors be similarly captured in a latent representation that is directly optimized from test-time interactions?

We introduce T2-Z for test-time optimization of environmental latents, which augments an actionconditioned robotic world model with a learnable environment latent $\mathbf { z } _ { e } , \mathbf { z } _ { e }$ is a randomly-initialized compact representation that is expected to encode the unobserved factors governing how an environment responds to interaction, with no supervision about what environmental factors it should represent. The world model predicts future visual observations conditioned jointly on the action and $\mathbf { z } _ { e }$ . We iteratively optimize the world model and $\mathbf { z } _ { e }$ in an alternating two-stage procedure: 1) world-model optimization: with $\mathbf { z } _ { e }$ fixed, we train the world model across all trajectories to learn shared action-conditioned dynamics across environments; 2) environment-latent optimization: with the world model frozen, we optimize the conditioned latent $\mathbf { z } _ { e }$ through prediction error, so that it captures underlying environment-specific variation not explained by the general world model alone. Both stages are optimized with the same flow-matching loss and alternated for multiple rounds until convergence, allowing the environment representation to emerge directly through prediction without predefined physical-factor supervision or a separate environment encoder.

At test time, the world model is frozen and T2-Z turns environment inference into a direct optimization problem. For a previously unseen environmental configuration, we initialize a new $\mathbf { z } _ { e }$ and optimize $\mathbf { z } _ { e }$ to explain the outcomes of one or more observed test-time interactions. Specifically, we update $\mathbf { z } _ { e }$ for multiple optimization steps through the same flow-matching objective used during stage-two training, minimizing prediction error on the observed interactions. Because inference is optimization-based rather than amortized through a learned history encoder, T2-Z is not tied to the trajectory count used for training and can flexibly incorporate any number of test-time trajectories. Once optimized, the latent turns the frozen world model into an environment-adapted test-time simulator, enabling prediction and comparison of candidate action outcomes before execution.

We evaluate T2-Z across 7 simulated settings and 4 real-world robotic tasks. Quantitatively and qualitatively, T2-Z improves world-model prediction across visual quality, physical prediction metrics (e.g., relevant object locations), and downstream action selection. T2-Z reduces physical prediction error from 12.86 to 4.80 in simulation and tracked-state error from 19.30 to 10.85 pixels in the real world, while improving action success from 37.9% to 74.2% and from 26.5% to 70.8%, respectively. PCA of the learned latents reveals structured representations that reflect the underlying environmental factors, including a multi-dimensional organization when more than one factors vary jointly, despite the model never knowing how many factors are present or what they represent (fig. 1(b)). In our multi-background setting, a unified latent structure emerges across diverse scene setups, revealing that the learned representation captures environmental dynamics without becoming entangled with scene appearance (fig. 1(d)). Ablations further show that T2-Z can infer useful environment representations from as few as a single test-time interaction trajectory.

## 2 RELATED WORKS

![](images/b1b9f81c7448096535f79212e3179d80fb0789040f1304d7083d7a29f6b62168.jpg)  
Figure 2: Comparison with existing approaches. (a) Vanilla world models do not explicitly represent unobserved environmental variation, producing the same prediction under different environmental conditions. (b) System-identification approaches such as RMA learn environment representations from predefined factors and their ground-truth values, then infer them from interaction history through a learned encoder. (c) T2-Z requires neither predefined environmental factors nor a learned environment encoder, instead directly optimizing an emergent environment latent from test-time interaction through prediction error.

Two distinct strands of work are particularly related to our problem.

Robotic world models have made rapid progress in predicting action-conditioned futures (Zhou et al., 2025; 2024; Zhu et al., 2024; Zhen et al., 2025; Zhu et al., 2025; Ye et al., 2026; Huang et al., 2026; Gao et al., 2026). More recent work explores test-time adaptation to improve long-horizon memory or incorporate new behavioral supervision (Wan et al., 2026; Wang et al., 2026; Feng et al., 2026; Yoo et al., 2025; Jang et al., 2026). These approaches predict future outcomes from visual observations alone, without accounting for underlying environmental variation that is not directly observable. (fig. 2 (a)). In contrast, T2-Z disentangles unobserved environmental variation from the world model’s observable scene representation, through environment latent conditioning.

Environment representations and system identification have demonstrated the value of environment-specific representations for system identification and rapid behavioral adaptation (Yu et al., 2017; Nagabandi et al., 2019; Rakelly et al., 2019; Lee et al., 2020; Kumar et al., 2021; Miki et al., 2022; Nahrendra et al., 2023; Wang et al., 2023; He et al., 2024; Luo et al., 2024). The Rapid Motor Adaptation (RMA) family and related methods typically rely on predefined privileged parameters (Kumar et al., 2021; Miki et al., 2022; Nahrendra et al., 2023; He et al., 2024; Luo et al., 2024). For example, RMA restricts adaptation to 5 factors and uses their ground-truth values to learn an embedding that an adaptation encoder later recovers from interaction history (Kumar et al., 2021). This prespecifies the environment representation and ties inference to a learned history-to-embedding mapping, limiting flexibility under unknown, compounded, or out-of-distribution variation (fig. 2 (b)). These methods also generally rely on a learned history encoder (Chen et al., 2024; Xing et al., 2026), limiting generalization when the amount of test-time interaction differs from that seen during training. In contrast, T2-Z learns representations of environmental factors implicitly through world prediction error, without predefined ground-truth physical parameters or a learned history encoder.

Our method is also related to the literature on Test-Time Adaptation (TTA), which has enabled robots to improve after deployment by adapting policy, prompts, or memory rather than optimizing a dedicated environment representation. Existing approaches include test-time adaptation of diffusion and flow policies (Li et al., 2025; 2026), online policy adaptation (Liu et al., 2026; Jin & Zhang, 2026; Zhu et al., 2026), and adaptation through latent prompts, fast weights, memory, or retrieval (Zhang et al., 2026; Jiang et al., 2026; Feng et al., 2026; Zhao et al., 2026; Park et al., 2026). TTA has also been applied directly to predictive models through video replanning, world-model training, and closed-loop latent adaptation (Ko et al., 2025; Wan et al., 2026; Wang et al., 2026).

## 3 METHOD

![](images/e779e116f1267652d7794593646905a68c2b3d8dde04e947727740d36e351e19.jpg)

![](images/c50890b4fbc66c24fc7f0df0082ac0cb00d9ae900f5dd1ece56ea5fee774667f.jpg)  
Figure 3: T2-Z framework. During training, T2-Z alternates between two stages: (1) world-model optimization, which learns shared action-conditioned dynamics with environment latents fixed, and (2) environmentlatent optimization, which optimizes the latents through prediction error with the world model frozen. (3) At test time, the world model remains frozen while a new environment latent is directly optimized from interaction and then reused to predict the outcomes of different actions.

We introduce T2-Z (fig. 2 & algorithm 1), a framework for adapting robotic world models to hidden environmental variation through direct optimization of an environment latent. Our key idea is to separate the general visual predictive capability of a world model from unobserved environment-specific information. During training, we learn these two components through alternating optimization under a shared flow-matching objective. At test time, we freeze the world model and optimize only a newly initialized environment latent from observed test-time interaction trajectories. The optimized latent subsequently conditions the frozen world model to predict the outcomes of diverse actions.

## 3.1 PROBLEM SETUP

We consider a robotic world model that predicts future visual observations y from the observation x and action a. We augment the world model with an environment latent $\mathbf { z } _ { e }$ and formulate prediction as:

$$
p _ { \theta } \left( \mathbf { y } \mid \mathbf { x } , \mathbf { a } , \mathbf { z } _ { e } \right) ,\tag{1}
$$

where θ denotes the shared world-model parameters and $\mathbf { z } _ { e }$ is a latent representation of the environment. Importantly, $\mathbf { z } _ { e }$ receives no supervision of which environmental factors it should represent.

Let $\mathbf { y } _ { t }$ denote the noisy/interpolated future observation at flow timestep t, and $\mathbf { u } _ { t }$ its corresponding target flow velocity. The model is optimized using the standard flow-matching objective,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta , \mathbf { z } _ { e } ) = \mathbb { E } \left[ \| \mathbf { v } _ { \theta } ( \mathbf { y } _ { t } , t \mid \mathbf { x } , \mathbf { a } , \mathbf { z } _ { e } ) - \mathbf { u } _ { t } \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{2}
$$

## 3.1.1 TRAINING

For each training environment e, we maintain an environment latent $\mathbf { z } _ { e }$ , randomly initialized by sampling each dimension from a uniform distribution:

$$
\begin{array} { r } { \mathbf { z } _ { e } ^ { ( 0 ) } \sim \mathcal { U } ( 0 , 1 ) . } \end{array}\tag{3}
$$

We alternate between two stages throughout training: world-model optimization and environmentlatent optimization, with k indexing the outer optimization cycles. The world model is updated with low learning rate to learn general dynamics shared across environments, while the environment latents are optimized rapidly for quick adaptation to environment information revealed through interaction. This slow–fast separation follows the perspective of learning at different timescales (Tiwari et al., 2026) and is closely related to test-time learning with slow/fast weights (Hong et al., 2024).

Algorithm 1 T2-Z Training and Test-Time Optimization   
Require: Training sets $\{ \mathcal { D } _ { e } \} _ { e = 1 } ^ { E } ,$ world model $p _ { \theta } .$ 8: for environment-latent optimization steps do   
loss $\mathcal { L } _ { \mathrm { F M } }$ , learning rate η and $\eta _ { z }$ 9: $\mathbf { z } _ { e } \gets \mathbf { z } _ { e } - \eta _ { z } \nabla _ { \mathbf { z } _ { e } } \mathcal { L } _ { \mathrm { F M } } ( \theta , \mathbf { z } _ { e } )$   
1: Initialize $\theta$ and $\mathbf { z } _ { e } \sim \mathcal { \bar { U } } ( 0 , 1 )$ for each environ- 10: end for   
ment e 11: end while   
// Training: alternating optimization // Test-time optimization   
2: while not converged do Require: Trained world model $\theta ^ { * }$ , observed interac-  
// Stage I: world-model optimization tions $\mathcal { D } _ { \mathrm { t e s t } }$ , Optimization Step s, learning rate $\eta _ { z } ^ { ( s ) }$   
3: Freeze environment latents $\{ \mathbf { z } _ { e } \} _ { e = 1 } ^ { E }$ 12: Freeze $\theta ^ { * }$ and initialize $\mathbf { z } _ { \mathrm { t e s t } } ^ { ( 0 ) }$   
4: for world-model optimization steps do 13: for $s = 0 , \ldots , S - 1$ do   
5: $\theta \gets \theta - \eta _ { \theta } \bar { \nabla _ { \theta } } \mathcal { L } _ { \mathrm { F M } } ( \theta , \mathbf { z } _ { e } )$ 14: $\begin{array} { r } { \mathcal { L } _ { \mathrm { t e s t } }  \frac { 1 } { | \mathcal { D } _ { \mathrm { t e s t } } | } \sum _ { ( \mathbf { x } , \mathbf { a } , \mathbf { y } ) \in \mathcal { D } _ { \mathrm { t e s t } } } \mathcal { L } _ { \mathrm { F M } } ( \theta ^ { * } , \mathbf { z } _ { \mathrm { t e s t } } ^ { ( s ) } ) } \end{array}$   
6: end for   
15: $\mathbf { z } _ { \mathrm { t e s t } } ^ { ( s + 1 ) } \xleftarrow { } \mathbf { z } _ { \mathrm { t e s t } } ^ { ( s ) } - \eta _ { z } ^ { ( s ) } \nabla _ { \mathbf { z } _ { \mathrm { t e s t } } ^ { ( s ) } } \mathcal { L } _ { \mathrm { t e s t } }$   
// Stage II: environment-latent optimization 16: end for   
7: Freeze world-model parameters θ 17: return $\mathbf { z } _ { \mathrm { t e s t } } ^ { * }  \mathbf { z } _ { \mathrm { t e s t } } ^ { ( S ) }$

Stage I: World-Model Optimization. In the first stage, we fix all $\mathbf { z } _ { e }$ and optimize only the worldmodel parameters. Let $\mathcal { D } _ { e }$ denote the interaction trajectories collected under environment e.

$$
\boldsymbol { \theta } ^ { ( k + 1 ) } = \arg \operatorname* { m i n } _ { \boldsymbol { \theta } } \mathbb { E } _ { \boldsymbol { e } , ( \mathbf { x } , \mathbf { a } , \mathbf { y } ) \sim \mathcal { D } _ { e } } \left[ \mathcal { L } _ { \mathrm { F M } } \left( \boldsymbol { \theta } , \mathbf { z } _ { e } ^ { ( k ) } \right) \right] .\tag{4}
$$

During this stage, $\mathbf { z } _ { e }$ is held constant, while the same world model is optimized over trajectories spanning different environments and actions. This encourages $\theta$ to capture predictive structure that is shared across the training distribution, including how visual states evolve to different robot actions.

Stage II: Environment-Latent Optimization. In the second stage, we reverse the optimization: the world model is frozen, and only the environment latents are updated. For each environment $e ,$

$$
\mathbf { z } _ { e } ^ { ( k + 1 ) } = \arg \operatorname* { m i n } _ { \mathbf { z } _ { e } } \mathbb { E } _ { ( \mathbf { x } , \mathbf { a } , \mathbf { y } ) \sim \mathcal { D } _ { e } } \left[ \mathcal { L } _ { \mathrm { F M } } \left( \boldsymbol { \theta } ^ { ( k + 1 ) } , \mathbf { z } _ { e } \right) \right] .\tag{5}
$$

Because θ is fixed, the only way to reduce prediction error is to change $\mathbf { z } _ { e }$ to better match the interaction outcomes observed in environment e. As a result, environments that induce systematically different outcomes drive their latents toward different regions of the latent space.

Alternating Optimization Training starts with Stage I, then alternates between Stage II and Stage I, each optimized for a fixed number of steps while keeping the other component fixed. As the world model improves, latents are optimized against an increasingly capable predictive model; conversely, the refined latents provide more informative conditioning for subsequent world-model optimization.

## 3.1.2 TEST-TIME OPTIMIZATION

At test time, we adapt to a previously unseen environment by optimizing only its environment latent while keeping the trained world model fixed. We initialize a new latent $\mathbf { z } _ { \mathrm { t e s t } } ^ { ( 0 ) }$ and observe one or more interaction trajectories on randomly-sampled actions. We denote the resulting test-time interaction set as

$$
\mathcal { D } _ { \mathrm { t e s t } } = \{ ( { \bf x } _ { i } , { \bf a } _ { i } , { \bf y } _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{6}
$$

where each tuple consists of an observation $\mathbf { x } _ { i } ,$ an action ${ \bf a } _ { i } ,$ and the resulting observations $\mathbf { y } _ { i }$

We then optimize $\mathbf { z } _ { \mathrm { t e s t } }$ using the same flow-matching objective employed during training:

$$
\mathbf { z } _ { \mathrm { t e s t } } ^ { ( s + 1 ) } = \mathbf { z } _ { \mathrm { t e s t } } ^ { ( s ) } - \eta _ { z } ^ { ( s ) } \nabla _ { \mathbf { z } _ { \mathrm { t e s t } } ^ { ( s ) } } \frac { 1 } { | \mathscr { D } _ { \mathrm { t e s t } } | } \sum _ { ( \mathbf { x } , \mathbf { a } , \mathbf { y } ) \in \mathscr { D } _ { \mathrm { t e s t } } } \mathscr { L } _ { \mathrm { F M } } \left( \theta ^ { * } , \mathbf { z } _ { \mathrm { t e s t } } ^ { ( s ) } ; \mathbf { x } , \mathbf { a } , \mathbf { y } \right) .\tag{7}
$$

where $\theta ^ { * }$ denotes the frozen world-model parameters and $\eta _ { z } ^ { ( s ) }$ follows a coarse-to-fine learning-rate strategy, with larger learning rates in early optimization steps and smaller learning rates in later steps. Because the world model is fixed, prediction errors over the observed interactions can only be reduced by updating $\mathbf { z } _ { \mathrm { t e s t } }$ . The latent is therefore optimized toward a representation under which the frozen world model reproduces the interaction outcomes observed in the current environment.

Once optimized, $\mathbf { z } _ { \mathrm { t e s t } } ^ { \ast }$ is reused to condition the world model for actions that were not observed during latent optimization. Given a different action a<sup>′</sup>, we predict its outcome as

$$
\hat { \mathbf { y } } ^ { \prime } \sim p _ { \theta ^ { * } } \left( \mathbf { y } \mid \mathbf { x } , \mathbf { a } ^ { \prime } , \mathbf { z } _ { \mathrm { t e s t } } ^ { * } \right) .\tag{8}
$$

The resulting conditioned world model serves as an environment-adapted test-time simulator, allowing it to predict the consequences of different actions under the inferred environment.

Unlike amortized history-encoder approaches, T2-Z uses direct optimization. This avoids committing to a fixed history mapping and searches for the representation that best explains the current environment. The formulation can naturally incorporate any number of test-time trajectories that differs from training, by optimizing a latent against all available evidence. Moreover, T2-Z is not restricted to the range of factor values during training, providing greater flexibility under out-of-distribution configurations. Ablation in fig. 5 shows this optimization remains stable across varying test-time trajectory numbers, and is interaction-efficient with a single trajectory sufficient in most settings.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Tasks We evaluate T2-Z across 7 simulated settings and 4 real-world robotic tasks. In simulation, Friction requires predicting how far an object moves after a push under different surface friction; Gravity varies gravitational strength and requires predicting the object motion; Mass Collision varies colliding object mass and predicts the resulting collision motion. Light Switch assigns different human-defined hidden interaction rules that determine whether a robot action turns the light on or off; Mass Balance predicts the tilt of a dumbbell under different mass distribution; Mass × Friction jointly varies object mass and surface friction at the same time; Diverse Visual Backgrounds evaluates the same friction variations across 30 diverse scene setups, testing whether the learned environment representation remains consistent across visual changes.. In the real world, Door Close varies the spring-lock stiffness, which changes how the door responds to closing actions; Ball Friction requires predicting the motion of a pushed ball of unknown mass; Stick Balance requires predicting the motion of a stick with objects of different masses attached to its ends; and Soft Pull requires predicting the deformation a soft object under different rotating actions.

Implementation Details. We adapt BWM Team (2026), an action-conditioned world model built on Wan2.2-TI2V-5B. The environment latent has a dimension of 32. Training starts with 300 world-model updates, followed by 200 latent updates and alternating 200-200 updates. Stage I/II use LR $1 0 ^ { - 5 } / \mathrm { { \dot { 0 } } . 0 3 }$ respectively. At test time, we freeze the world model and use a coarse-to-fine LR schedule of $3 . 0 / 1 . 5 / 0 . 5 / 0 . 1 5$ , 10 steps each, for 40 steps. T2-Z uses only one test-time trajectory on randomly-sampled action for Friction, Gravity, Mass Collision, Mass × Friction, Ball Friction, and Soft Pull, and two for Light Switch, Mass Balance, Door Close, and Stick Balance.

Evaluation Metrics. (1) Physical Error measures task-relevant state prediction: object-location error in pixels for Friction, Gravity, Mass Collision, and Diverse Backgrounds; lamp activation error for Light Switch; and bar-tilt angle error in degrees for Mass Balance. For real-world tasks, we measure tracked-state error in pixels. We report Physical Error (Mean), averaged over predicted frames, and Physical Error (Final), measured at the final predicted frame. (2) Video Generation Quality is measured by LPIPS and PSNR between predicted and ground-truth videos. LPIPS measures perceptual discrepancy in a learned feature space, capturing differences in visual structure and appearance, and serves as our primary measure of generation quality. In contrast, PSNR primarily reflects pixel-level reconstruction fidelity and is less informative about semantically meaningful differences or whether object motion is physically correct. (3) Action Success treats the world model as a simulator: given candidate actions, we predict their outcomes and select the action most likely to achieve the task goal, such as pushing a ball to a target, balancing a stick, or closing a door. We execute the chose action in real world and observe if meets the success criterions in Supp section C.

Table 1: Simulation performance across environmental variations. For Friction, Gravity, Mass Collision, Mass × Friction, and Diverse Scenes, physical error is measured in pixels; for Light Switch, it is lamp-score error; and for Mass Balance, it is bar-tilt error in degrees. The average is computed over all seven tasks.
<table><tr><td>Task</td><td>Method</td><td>Physical Error (Mean) ↓</td><td>Physical Error (Final) ↓</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>Action Success ↑</td></tr><tr><td rowspan="5">Mass Balance</td><td>Standard World Model</td><td>1.1844</td><td>4.5571</td><td>0.0244</td><td>32.969</td><td>40.0%</td></tr><tr><td>LoRA TTA</td><td>1.7975</td><td>7.3105</td><td>0.0631</td><td>24.895</td><td>50.0%</td></tr><tr><td>DINOv2 Context</td><td>1.2636</td><td>4.9480</td><td>0.0239</td><td>33.263</td><td>30.0%</td></tr><tr><td>TTT-KQV</td><td>2.5075</td><td>10.1759</td><td>0.0312</td><td>32.114</td><td>30.0%</td></tr><tr><td>T2-Z (Òurs)</td><td>0.6403</td><td>2.0842</td><td>0.0240</td><td>32.814</td><td>80.0%</td></tr><tr><td rowspan="5">Friction</td><td>Standard World Model</td><td>11.0626</td><td>24.4116</td><td>0.0667</td><td>31.113</td><td>32.0%</td></tr><tr><td>LoRA TTA</td><td>8.1902</td><td>19.1902</td><td>0.0582</td><td>32.031</td><td>48.0%</td></tr><tr><td>DINOv2 Context</td><td>11.8142</td><td>26.1703</td><td>0.0681</td><td>30.697</td><td>48.0%</td></tr><tr><td>TTT-KQV</td><td>10.8137</td><td>23.8675</td><td>0.0649</td><td>30.779</td><td>32.0%</td></tr><tr><td>T2-Z (Òurs)</td><td>3.8540</td><td>7.5931</td><td>0.0495</td><td>31.976</td><td>72.0%</td></tr><tr><td rowspan="5">Gravity</td><td>Standard World Model</td><td>21.4246</td><td>34.8779</td><td>0.0345</td><td>32.509</td><td>23.5%</td></tr><tr><td>LoRA TTA</td><td>7.9005</td><td>13.7137</td><td>0.0231</td><td>33.328</td><td>47.1%</td></tr><tr><td>DINOv2 Context</td><td>5.9300</td><td>8.8290</td><td>0.0195</td><td>34.095</td><td>76.5%</td></tr><tr><td>TTT-KQV</td><td>18.9572</td><td>30.9355</td><td>0.0291</td><td>33.308</td><td>35.3%</td></tr><tr><td>T2-Z (urs)</td><td>7.3408</td><td>11.0653</td><td>0.0202</td><td>33.389</td><td>82.4%</td></tr><tr><td rowspan="5">Mass Collision</td><td>Standard World Model</td><td>21.9329</td><td>25.7364</td><td>0.0904</td><td>29.805</td><td>38.9%</td></tr><tr><td>LoRA TTA</td><td>15.8704</td><td>24.5877</td><td>0.0691</td><td>30.049</td><td>33.3%</td></tr><tr><td>DINOv2 Context</td><td>8.5264</td><td>10.2101</td><td>0.0540</td><td>31.039</td><td>38.9%</td></tr><tr><td>TTT-KQV</td><td>20.9753</td><td>29.2930</td><td>0.0842</td><td>29.385</td><td>55.6%</td></tr><tr><td>T2-Z (Òurs)</td><td>6.4522</td><td>8.1321</td><td>0.0538</td><td>31.150</td><td>61.1%</td></tr><tr><td rowspan="5">Light Switch</td><td>Standard World Model</td><td>10.3500</td><td>18.8200</td><td>0.0187</td><td>32.619</td><td>50.0%</td></tr><tr><td>LoRA TTA</td><td>8.2800</td><td>14.7700</td><td>0.0175</td><td>33.123</td><td>50.0%</td></tr><tr><td>DINOv2 Context</td><td>10.0000</td><td>18.3700</td><td>0.0160</td><td>33.074</td><td>50.0%</td></tr><tr><td>TTT-KQV</td><td>10.4800</td><td>18.8700</td><td>0.0161</td><td>32.692</td><td>50.0%</td></tr><tr><td>T2-Z (urs)</td><td>3.4300</td><td>4.7400</td><td>0.0150</td><td>33.517</td><td>87.5%</td></tr><tr><td rowspan="5">Mass × Friction</td><td>Standard World Model</td><td>16.0219</td><td>29.2454</td><td>0.0610</td><td>31.077</td><td>35.0%</td></tr><tr><td>LoRA TTA</td><td>16.8282</td><td>26.9302</td><td>0.0701</td><td>29.865</td><td>26.7%</td></tr><tr><td>DINOv2 Context</td><td>11.8116</td><td>19.0756</td><td>0.0565</td><td>31.367</td><td>60.8%</td></tr><tr><td>TTT-KQV</td><td>18.9511</td><td>29.1411</td><td>0.0726</td><td>29.739</td><td>33.3%</td></tr><tr><td>T2-Z (Òurs)</td><td>8.1481</td><td>13.7222</td><td>0.0500</td><td>30.673</td><td>69.7%</td></tr><tr><td rowspan="5">Diverse Backgrounds</td><td>Standard World Model</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoRA TTA</td><td>8.0260</td><td>16.1831</td><td>0.1059</td><td>28.201</td><td>45.8%</td></tr><tr><td>DINOv2 Context</td><td>6.5805</td><td>16.2328</td><td>0.1074</td><td>27.986</td><td>58.3%</td></tr><tr><td>TTT-KQV</td><td>7.7367</td><td>16.0065</td><td>0.1064</td><td>28.000</td><td>58.3%</td></tr><tr><td>T2-z (Ours)</td><td>7.7783 3.7145</td><td>15.5869 6.3403</td><td>0.1116 0.0888</td><td>27.691 28.804</td><td>58.3% 66.7%</td></tr><tr><td rowspan="5">Average</td><td>Standard World Model</td><td>12.8575</td><td></td><td></td><td></td><td></td></tr><tr><td>LoRA TTA</td><td>9.3496</td><td>21.9759 17.5336</td><td>0.0574 0.0584</td><td>31.185 30.182</td><td>37.9% 44.8%</td></tr><tr><td>DINOv2 Context</td><td>8.1546</td><td>14.8014</td><td>0.0492</td><td>31.648</td><td>51.8%</td></tr><tr><td>TTT-KQV</td><td>12.9233</td><td>22.5528</td><td>0.0585</td><td>30.815</td><td>42.1%</td></tr><tr><td>T2-Z (Ours)</td><td>4.7971</td><td>7.6682</td><td>0.0430</td><td>31.760</td><td>74.2%</td></tr></table>

Baselines. We compare T2-Z with four baselines using the same world-model backbone. Standard World Model uses no environment latent or test-time adaptation; LoRA TTA adapts attention LoRA parameters from test-time interactions; DINOv2 Context amortizes a 32-D environment context from interaction trajectories; and TTT-KQV adapts fast weights within the world model. All adaptive baselines receive the same test-time interaction trajectories as T2-Z.

Please refer to the Supplementary for more details about architecture (section A), task definitions (section C), baseline implementation (section D), and compute efficiency comparison (section E.4).

## 4.2 SIMULATION EXPERIMENTS

Single-Factor Environmental Variation. We first evaluate T2-Z under individual unobserved environmental factors, spanning continuous physical properties, mass-dependent interactions, and discrete human-defined rules. As shown in table 1, T2-Z achieves the highest action success on all five tasks, increasing the five-task average from 36.9% for the Standard World Model to 76.6%, while reducing average LPIPS from 0.0469 to 0.0325. It also substantially improves task-relevant prediction, reducing mean error from 11.06 to 3.85 pixels on Friction, 21.93 to 6.45 pixels on Mass Collision, and 1.18<sup>◦</sup> to 0.64<sup>◦</sup> on Mass Balance. These gains arise because T2-Z directly optimizes an environment latent from observed interactions, capturing environment-specific variation more flexibly than non-adaptive or amortized baselines.

![](images/e5092947f541393634024db0e0fa091723543adaffd769e825577e274bb99f71.jpg)

![](images/ed219da043b50a68be27daf27cd07ca82c03dc140a87ccbea7af117d79ae9b75.jpg)  
Figure 4: PCA of learned environment latents. We visualize learned environment latents using PCA onto x-y axes, with ⃝ denoting training environments and △ denoting test environments. Ground-truth factors (denoted by color or size) are shown only for visualization and are never provided to the model. (a–b) show that structured latent representations emerge in both simulated and real-world single-factor settings. (c) shows that a unified latent structure emerges across diverse visual backgrounds, indicating that the representation does not become entangled with scene appearance. (d) shows that a two-dimensional latent structure emerges when mass and friction vary jointly, with x-axis primarily corresponding to friction and y-axis to mass, despite the model never knowing how many underlying factors are present or what they represent.

Emergent Structure of Environment Latents. We examine whether direct optimization gives rise to structured environment representations without supervision of the identity or value of underlying factors. In fig. 4, we apply principal component analysis (PCA) to the learned environment latents and reduce them to two dimensions for visualization. We annotate the projected points using their corresponding ground-truth factor values to reveal their relationship with the learned representation; these values are never provided to the model. Figure 4(a) shows the single-factor friction experiment. The learned latents form a one-dimensional structure for a single factor that corresponds closely to the underlying friction values, as reflected by the point colors: similar values are located nearby, while friction varies consistently x-axis. PCA visualizations for other tasks are provided in the Supplementary Material section E.3.

Compounded Environmental Factors. We next evaluate T2-Z under the more challenging setting of compounded environmental factors, where object mass and surface friction vary simultaneously and jointly reflect in object motion. Importantly, T2-Z is never informed that there are two underlying factors, nor given their identities or values. table 1 shows T2-Z achieves the lowest physical errors, while improving action success from 35.0% to 69.7%. More strikingly, fig. 4(d) shows that the learned latents naturally form a two-dimensional structure corresponding to the two underlying factors, showing that T2-Z discovers the structure of compounded environmental factors through prediction alone, without knowing in advance how many factors are present or what they represent.

Robustness across Diverse Visual Scenes. We test whether the learned environment representation captures underlying dynamics without becoming entangled with visual appearance. Specifically, we ask whether a unified latent structure emerges across visually diverse scenes despite substantial differences in appearance. We construct a multi-background setting in which the same underlying friction factor varies across diverse scene layouts and visual backgrounds. table 1 shows that T2-Z outperforms all baselines across every metric. fig. 4(c) further shows that latents from different backgrounds organize consistently rather than separating according to scene appearance. This indicates that environment-latent can isolate environment-specific information that is shared across visually different observations, enabling a unified latent structure to emerge across diverse scenes.

## 4.3 REAL-WORLD ROBOTIC EXPERIMENTS

We further evaluate T2-Z on four real-world tasks. As shown in table 2, compared with the Standard World Model, T2-Z reduces average mean/final physical error from 19.30/38.24 to 10.85/16.66 pixels and LPIPS from 0.05194 to 0.04341, while increasing action success from 26.47% to 70.80%. On Stick Balance, for example, action success increases from 20.00% to 83.33%. fig. 4(b) reveals a clear one-dimensional latent structure without ground-truth friction supervision or post-hoc annotation. T2-Z is particularly important in the real world, where underlying environmental factors are typically unknown and unmeasured and interactions contain additional uncontrolled variation. Notably, T2-Z achieves even larger improvements over the Standard World Model than in simulation, indicating greater benefits as environmental variation becomes less controlled.

Table 2: Real-world robotic task performance. T2-Z outperforms baselines across real-world tasks.
<table><tr><td>Task</td><td>Method</td><td>Physical Error (Mean) ↓</td><td>Physical Error (Final) ↓</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>Action Success ↑</td></tr><tr><td rowspan="5">Door Close</td><td>Standard World Model</td><td>21.82</td><td>41.22</td><td>0.05700</td><td>29.96</td><td>22.22%</td></tr><tr><td>LoRA TTA</td><td>17.35</td><td>32.14</td><td>0.04946</td><td>30.24</td><td>22.22%</td></tr><tr><td>DINOv2 Context</td><td>21.12</td><td>38.90</td><td>0.05490</td><td>30.28</td><td>22.22%</td></tr><tr><td>TTT-KQV</td><td>19.87</td><td>35.93</td><td>0.05164</td><td>30.95</td><td>22.22%</td></tr><tr><td>T2-z (urs)</td><td>7.95</td><td>13.47</td><td>0.03208</td><td>32.15</td><td>55.56%</td></tr><tr><td rowspan="5">Ball Friction</td><td>Standard World Model</td><td>28.36</td><td>64.42</td><td>0.03153</td><td>36.04</td><td>0.00%</td></tr><tr><td>LoRA TTA</td><td>37.48</td><td>72.09</td><td>0.03805</td><td>35.44</td><td>50.00%</td></tr><tr><td>DINOv2 Context</td><td>24.90</td><td>52.54</td><td>0.02934</td><td>36.32</td><td>12.50%</td></tr><tr><td>TTT-KQV</td><td>29.80</td><td>67.37</td><td>0.03224</td><td>36.29</td><td>25.00%</td></tr><tr><td>T2-z (Ours)</td><td>14.47</td><td>24.57</td><td>0.02401</td><td>36.23</td><td>62.50%</td></tr><tr><td rowspan="5">Stick Balance</td><td>Standard World Model</td><td>5.29</td><td>11.40</td><td>0.04716</td><td>31.87</td><td>20.00%</td></tr><tr><td>LoRA TTA</td><td>3.63</td><td>7.24</td><td>0.04111</td><td>32.40</td><td>40.00%</td></tr><tr><td>DINOv2 Context</td><td>2.80</td><td>7.54</td><td>0.03640</td><td>33.58</td><td>20.00%</td></tr><tr><td>TTT-KQV</td><td>4.49</td><td>10.40</td><td>0.03948</td><td>33.38</td><td>25.00%</td></tr><tr><td>T2-z (Ours)</td><td>2.27</td><td>5.01</td><td>0.04175</td><td>32.05</td><td>83.33%</td></tr><tr><td rowspan="5">Soft Pull</td><td>Standard World Model</td><td>21.71</td><td>35.90</td><td>0.07207</td><td>33.32</td><td>63.64%</td></tr><tr><td>LoRA TTA</td><td>18.63</td><td>26.20</td><td>0.06862</td><td>33.63</td><td>63.64%</td></tr><tr><td>DINOv2 Context</td><td>21.19</td><td>29.17</td><td>0.07087</td><td>33.47</td><td>72.73%</td></tr><tr><td>TTT-KQV</td><td>18.23</td><td>29.69</td><td>0.06439</td><td>34.03</td><td>72.73%</td></tr><tr><td>T2-z (urs)</td><td>18.71</td><td>23.59</td><td>0.07580</td><td>32.52</td><td>81.82%</td></tr><tr><td rowspan="5">Average</td><td>Standard World Model</td><td>19.30</td><td>38.24</td><td>0.05194</td><td>32.80</td><td>26.47%</td></tr><tr><td>LoRA TTA</td><td>19.27</td><td>34.42</td><td>0.04931</td><td>32.93</td><td>43.97%</td></tr><tr><td>DINOv2 Context</td><td>17.50</td><td>32.04</td><td>0.04788</td><td>33.41</td><td>31.86%</td></tr><tr><td>TTT-KQV</td><td>18.10</td><td>35.85</td><td>0.04694</td><td>33.66</td><td>36.24%</td></tr><tr><td>T2-z (Òurs)</td><td>10.85</td><td>16.66</td><td>0.04341</td><td>33.24</td><td>70.80%</td></tr></table>

Physical Error (Mean) Physical Error (Final) LPIPS Action Success Mass Collision Light Switch Door Close Soft Pull

![](images/fd572476bc1add78fdbca78c9167545618139b3a780fcc903bc97a1e66bf3f0d.jpg)

![](images/08f8f3de52ec29b321436d1723bd04dcea2c945bc5ac516bf1769c23144be7a7.jpg)

![](images/9b6ee1d995964ca5e74e08a5492833019f76e19ddf26460d56a21bc5140f811c.jpg)  
Figure 5: Ablation studies. (a) Test-Time Optimization Learning-rate strategy. Constant Maximum/Minimum remove coarse-to-fine scheduling, while Ours $\times 1 0 / \div 1 0$ scale our schedule uniformly. (b) Ablating the testtime optimization budget shows that performance peaks at 40 steps. (c,d) Effect of the number of test-time interaction trajectories. Bold markers denote our chosen trajectory count, after which performance largely saturates.

## 4.4 ABLATIVE & QUALITATIVE STUDIES

We ablate the key design choices of T2-Z in fig. 5 and table 3, including latent optimization, interaction evidence, latent parameterization, and alternating training. In fig. 5(a), we vary the learning-rate strategy. Our schedule performs best, validating both its coarse-to-fine form and learning-rate scale. Figure 5(b) varies test-time optimization from 10–60 steps; performance improves up to 40 steps and then saturates, motivating our default. Figure 5(c,d) ablate the number of test-time interaction tra jectories. Performance largely saturates beyond the selected number of trajectories, indicating that additional interaction provides limited benefit. Recall that T2-Z uses only one test-time trajectory for 7 tasks, and two for 4 tasks, demonstrating that T2-Z can infer useful environment information with as few as one or two interactions. Additional ablations in table 3 study latent parameterization and training. All the ablated models underperform our model, supporting our alternating schedule and latent mechanism. We provide the latent-dimensionality ablation in Supplementary table 12.

In fig. 6, we show qualitative results across simulation and real-world tasks. T2-Z closely matches ground-truth outcomes, indicating that the inferred latent successfully conditions prediction on the current environment. By contrast, the baselines fail to capture these environment-specific differences, producing largely random motions and incorrect final configurations across environments. Please refer to the Supplementary Material for more qualitative examples (section F).

Table 3: Additional Ablation study. Joint Model–Latent Training optimizes the world model and latent together without separate stages; Direct Token directly optimizes the full-dimensional conditioning token; 32-D Random [−0.05, 0.05] changes the latent initialization range; and Alternating Latent/Model change the number of updates before switching between the two optimization stages, compared with our default 200/200.
<table><tr><td>Method</td><td>Physical Error (Mean) ↓</td><td>Physical Error (Final) ↓</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>Action Success ↑</td></tr><tr><td>Joint Model-Latent Training</td><td>7.03</td><td>15.66</td><td>0.0657</td><td>31.203</td><td>68%</td></tr><tr><td>Direct Token (3072-D)</td><td>94.50</td><td>80.49</td><td>0.7626</td><td>15.218</td><td>24%</td></tr><tr><td>32-D Random [−0.05, 0.05]</td><td>4.81</td><td>9.73</td><td>0.0495</td><td>32.172</td><td>60%</td></tr><tr><td>Alternating Latent/Model (100/100)</td><td>5.41</td><td>11.50</td><td>0.0484</td><td>32.208</td><td>56%</td></tr><tr><td>Alternating Latent/Model (400/400)</td><td>4.31</td><td>8.80</td><td>0.0507</td><td>32.430</td><td>68%</td></tr><tr><td>T2-Z</td><td>3.85</td><td>7.59</td><td>0.0495</td><td>31.976</td><td>72%</td></tr></table>

![](images/2d5124565e4f69c8402fc906620a53ed0d5e4bf4f8f351bc6dfd4783997ab860.jpg)  
Figure 6: Qualitative results in simulation and the real world. Within each task, we apply the same actions across different environments, which produce different ground-truth outcomes due to underlying environmental variation. T2-Z closely matches these environment-dependent outcomes.

## 5 CONCLUSION

We introduce T2-Z which augments robotic world models with enviornment latent conditioning. T2-Z alternates between world-model optimization that learns action-conditioned dyanmics, and environment-latent optimization that captures hidden environment factors. For a new environment, the world model is frozen and only the latent is optimized. Across 7 simulation and 4 real-world tasks, T2-Z outperforms baselines a lot in both video prediction and action selection.

## ACKNOWLEDGEMENTS

This paper is in part funded by ONR MURI N00014-24-1-2748, ONR MURI N00014-22-1-2740, Singapore MDDI AIVP-2025-008, IITP funded by the Korean Government (MSIT) (No. RS-2024- 00457882, National AI Research Lab Project), 2077AI, and Grant AW1134392 (Reasoning in Motion) from the TRI University 3.0 Program.

## REFERENCES

BWM Team. BWM: A low-cost high-fidelity world simulator for robot learning. arXiv preprint arXiv:2607.29302, 2026. 6, 16

Shiyi Chen, Zeyu Wan, Shiyang Yan, Chun Zhang, Weiyi Zhang, Qiang Li, Debing Zhang, and Fasih Ud Din Farrukh. Slr: Learning quadruped locomotion without privileged information. In Conference on Robot Learning, pp. 3212–3224, 2024. 3

Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17702–17711, 2025. 22

Yusen Feng, Bingchen Han, Jiangran Lyu, Kai Liu, Yixin Zheng, Yuxuan Wan, Weiheng Liu, Sun Han, Ruiqin Li, Yulong Zhang, Fangfu Liu, Xuesong Shi, Libin Liu, Yizhou Wang, Zhizheng Zhang, and He Wang. WAM-TTT: Steering world-action models by watching human play at test time. arXiv preprint arXiv:2607.06988, 2026. 3

Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, Wei-Cheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, Qianli Ma, Seungjun Nah, Loic Magne, Jiannan Xiang, Yuqi Xie, Ruijie Zheng, Dantong Niu, You Liang Tan, K. R. Zentner, George Kurian, Suneel Indupuru, Pooya Jannaty, Jinwei Gu, Jun Zhang, Jitendra Malik, Pieter Abbeel, Ming-Yu Liu, Yuke Zhu, Joel Jang, and Linxi Jim Fan. Dreamdojo: A generalist robot world model from large-scale human videos. arXiv preprint arXiv:2602.06949, 2026. 3

Haoran He, Peilin Wu, Chenjia Bai, Hang Lai, Lingxiao Wang, Ling Pan, Xiaolin Hu, and Weinan Zhang. Bridging the sim-to-real gap from the information bottleneck perspective. In Conference on Robot Learning, pp. 473–495, 2024. 3

Yining Hong, Beide Liu, Maxine Wu, Yuanhao Zhai, Kai-Wei Chang, Linjie Li, Kevin Lin, Chung-Ching Lin, Jianfeng Wang, Zhengyuan Yang, Yingnian Wu, and Lijuan Wang. Slowfast-vgen: Slow-fast learning for action-driven long video generation, 2024. URL https://arxiv. org/abs/2410.23277. 5

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 21

Wenlong Huang, Yu-Wei Chao, Arsalan Mousavian, Ming-Yu Liu, Dieter Fox, Kaichun Mo, and Li Fei-Fei. Pointworld: Scaling 3d world models for in-the-wild robotic manipulation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20765– 20779, 2026. 3

Jinwoo Jang, Minjong Yoo, Sihyung Yoon, and Honguk Woo. Test-time mixture of world models for embodied agents in dynamic environments. In International Conference on Learning Representations, 2026. 3

Yunfan Jiang, Yevgen Chebotar, Ruijie Zheng, Fengyuan Hu, Yunhao Ge, Jimmy Wu, Tianyuan Dai, Scott Reed, Li Fei-Fei, Yuke Zhu, and Linxi Jim Fan. RoboTTT: Context scaling for robot policies. arXiv preprint arXiv:2607.15275, 2026. 3

Ruofan Jin and Zaixi Zhang. Agentic-vla: Efficient online adaptation for vision-language-action models. arXiv preprint arXiv:2605.22896, 2026. 3

Po-Chen Ko, Jiayuan Mao, Yu-Hsiang Fu, Hsien-Jeng Yeh, Chu-Rong Chen, Wei-Chiu Ma, Yilun Du, and Shao-Hua Sun. Implicit state estimation via video replanning. arXiv preprint arXiv:2510.17315, 2025. 3

Ashish Kumar, Zipeng Fu, Deepak Pathak, and Jitendra Malik. Rma: Rapid motor adaptation for legged robots. In Robotics: Science and Systems, 2021. doi: 10.15607/RSS.2021.XVII.011. 2, 3

Kimin Lee, Younggyo Seo, Seunghyun Lee, Honglak Lee, and Jinwoo Shin. Context-aware dynamics model for generalization in model-based reinforcement learning. In International Conference on Machine Learning, pp. 5757–5766, 2020. 3

Zezeng Li, Rui Yang, Ruochen Chen, ZhongXuan Luo, and Liming Chen. ADPro: a test-time adaptive diffusion policy for robot manipulation via manifold and initial noise constraints. arXiv preprint arXiv:2508.06266, 2025. 3

Zezeng Li, Enda Xiang, Thuy Tran, Di Huang, Momath Thiam, and Liming Chen. PriGo: Test-time primitive guidance to diffusion and flow policies for adaptive robotic manipulation. arXiv preprint arXiv:2607.07076, 2026. 3

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2023. 18

Changyu Liu, Yiyang Liu, Taowen Wang, Qiao Zhuang, James Chenhao Liang, Wenhao Yang, Renjing Xu, Qifan Wang, Dongfang Liu, and Cheng Han. On-the-fly vla adaptation via testtime reinforcement learning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, 2026. 3

Shixin Luo, Songbo Li, Ruiqi Yu, Zhicheng Wang, Jun Wu, and Qiuguo Zhu. Pie: Parkour with implicit-explicit learning framework for legged robots. IEEE Robotics and Automation Letters, 9 (11):9986–9993, 2024. doi: 10.1109/LRA.2024.3459797. 3

Takahiro Miki, Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. Learning robust perceptive locomotion for quadrupedal robots in the wild. Science Robotics, 7(62):eabk2822, 2022. doi: 10.1126/scirobotics.abk2822. 3

Anusha Nagabandi, Ignasi Clavera, Simin Liu, Ronald S. Fearing, Pieter Abbeel, Sergey Levine, and Chelsea Finn. Learning to adapt in dynamic, real-world environments through metareinforcement learning. In International Conference on Learning Representations, 2019. 3

I Made Aswin Nahrendra, Byeongho Yu, and Hyun Myung. Dreamwaq: Learning robust quadrupedal locomotion with implicit terrain imagination via deep reinforcement learning. In IEEE International Conference on Robotics and Automation, pp. 5078–5084, 2023. doi: 10.1109/ICRA48891.2023.10161144. 3

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov,´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve J´ egou, Julien Mairal, Patrick Labatut,´ Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. 22

Jeongeun Park, Juhan Park, Taekyung Kim, Sungjoon Choi, Dongyoon Han, and Sangdoo Yun. Retrieve, don’t retrain: Extending vision language action models to new tasks at test time. arXiv preprint arXiv:2606.15631, 2026. 3

Kate Rakelly, Aurick Zhou, Chelsea Finn, Sergey Levine, and Deirdre Quillen. Efficient off-policy meta-reinforcement learning via probabilistic context variables. In International Conference on Machine Learning, pp. 5331–5340, 2019. 3

Rishabh Tiwari, Kusha Sareen, Lakshya A. Agrawal, Joseph E. Gonzalez, Matei Zaharia, Kurt Keutzer, Inderjit S. Dhillon, Rishabh Agarwal, and Devvrit Khatri. Learning, fast and slow: Towards llms that adapt continually. arXiv preprint arXiv:2605.12484, 2026. 5

Chi Wan, Kangrui Wang, Yuan Si, Pingyue Zhang, and Manling Li. Worldagen: Unified stateaction prediction with test-time world model training. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 18584–18592, 2026. doi: 10.1609/aaai.v40i22.38925. 3

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 16

Junjie Wang, Yao Mu, Dong Li, Qichao Zhang, Dongbin Zhao, Yuzheng Zhuang, Ping Luo, Bin Wang, and Jianye Hao. Prototypical context-aware dynamics generalization for high-dimensional model-based reinforcement learning. In ICLR Workshop on Scene Representations for Autonomous Driving, 2023. 3

Ying Wang, Oumayma Bounou, Yann LeCun, and Mengye Ren. Adajepa: An adaptive latent world model. arXiv preprint arXiv:2606.32026, 2026. 3

Jiaxu Xing, Zhiyuan Zhu, Yunfan Ren, Ismail Geles, Yifan Zhai, Rudolf Reiter, and Davide Scaramuzza. Continual robot policy learning via variational neural dynamics. arXiv preprint arXiv:2606.27353, 2026. 3

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi Jim Fan, and Joel Jang. World action models are zero-shot policies, 2026. 3

Minjong Yoo, Jinwoo Jang, Sihyung Yoon, and Honguk Woo. World model implanting for test-time adaptation of embodied agents. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 72556–72573, 2025. 3

Wenhao Yu, Jie Tan, C. Karen Liu, and Greg Turk. Preparing for the unknown: Learning a universal policy with online system identification. In Robotics: Science and Systems, 2017. 3

Alan Yuille and Daniel Kersten. Vision as bayesian inference: Analysis by synthesis? Trends in Cognitive Sciences, 10(7):301–308, 2006. 2

Wenbo Zhang, Jianxiong Li, Shuai Yang, Sijin Chen, Jiajun Liu, Lingqiao Liu, and Xiao Ma. TTT-VLA: Test-time latent prompt optimization for vision-language-action models. arXiv preprint arXiv:2606.03127, 2026. 3

Jianchao Zhao, Huoren Yang, Yusong Hu, Yuyang Gao, Qiguan Ou, Cong Wan, SongLin Dong, Zhiheng Ma, and Yihong Gong. Retrieve-then-steer: Online success memory for test-time adaptation of generative vlas. arXiv preprint arXiv:2605.10094, 2026. 3

Haoyu Zhen, Qiao Sun, Hongxin Zhang, Junyan Li, Siyuan Zhou, Yilun Du, and Chuang Gan. Learning 4d embodied world models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 5337–5347, 2025. 3

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: World models on pretrained visual features enable zero-shot planning. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 79115–79135, 2025. 3

Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, and Chuang Gan. Robodreamer: Learning compositional world models for robot imagination. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 61885–61896, 2024. 3

Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. In Proceedings of Robotics: Science and Systems, 2025. doi: 10.15607/RSS.2025. XXI.015. 3

Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, and Tao Kong. IRASim: Learning interactive real-robot action simulators. arXiv preprint arXiv:2406.14540, 2024. 3

Shaoting Zhu, Baijun Ye, Jiaxuan Wang, Jiakang Chen, Ziwen Zhuang, Linzhan Mou, Runhan Huang, and Hang Zhao. TTT-Parkour: Rapid test-time training for perceptive robot parkour. arXiv preprint arXiv:2602.02331, 2026. 3

## Supplementary Material for T2-Z: Test-Time Optimization of Environment Latents for Robotic World Models

## CONTENTS

A Model Architecture and Optimization 16   
A.1 Architecture and latent conditioning 16   
A.2 Training and test-time adaptation 16   
B Experimental Setup 17   
B.1 Training data and resource budgets 17   
B.2 Simulator and real-world setup 18   
C Evaluation Protocols 18   
C.1 Shared protocol and metrics 18   
C.2 Simulated tasks 18   
C.3 Real-world tests 20   
D Baseline Implementation Details 21   
D.1 Standard World Model 21   
D.2 LoRA TTA 21   
D.3 DINOv2 Context 22   
D.4 TTT-KQV 22   
E Additional Experiments 23   
E.1 Test-time interaction-count ablation 23   
E.2 Latent dimensionality 23   
E.3 Latent-space PCA across tasks 23   
E.4 Computational efficiency 23   
F Additional Qualitative Results 26   
References 38

## A MODEL ARCHITECTURE AND OPTIMIZATION

## A.1 ARCHITECTURE AND LATENT CONDITIONING

Inputs and inherited backbone. For an episode in environment $e ,$ the model receives one initial RGB frame $I _ { 0 }$ and an aligned sequence $a \ = \ ( a _ { t } ) _ { t }$ of 14-D robot actions, and predicts the subsequent video frames. We initialize the video diffusion transformer (DiT) and action encoder from BWM (BWM Team, 2026), retaining its Wan2.2(Wan et al., 2025)-based action-conditioning pathway. The frozen video VAE encodes the initial frame into the video-latent stream and decodes predicted latents back to RGB. Text conditioning is disabled; the initial frame is not an extra cross-attention token. In short, $I _ { 0 }$ enters the video-latent stream, a follows BWM’s two actionconditioning paths, and the environment latent augments both paths.

Environment-indexed latent conditioning. We associate each environment e with a freely optimized vector $z _ { e } \in \mathbb { R } ^ { 3 2 }$ . Episodes assigned to an environment share its latent; their initial frames and actions vary, so predicting their different outcomes constrains the environment representation. There is no observation-to-latent encoder and no supervision from the numerical physical parameter. Let $D = 3 0 7 2$ be the DiT width. Two separate projectors turn $z _ { e }$ into model-width vectors:

$$
\begin{array} { r l } & { q _ { b } ( z _ { e } ) = W _ { b , 2 } \mathrm { G E L U } ( W _ { b , 1 } \mathrm { L N } ( z _ { e } ) ) , \quad b \in \{ \mathrm { t o k e n , m o d } \} , } \\ & { \quad W _ { b , 1 } : 3 2 \to 1 2 8 , \qquad W _ { b , 2 } : 1 2 8 \to D . } \end{array}\tag{9}
$$

The two linear layers in each projector have no bias; the projectors do not share weights. The 128-D width is internal to these MLPs: both outputs are D-dimensional.

Cross-attention branch. BWM’s action encoder maps each action to a D-dimensional conditioning token. Let $A _ { \mathrm { t o k } } ( a )$ denote this token sequence and $u _ { e } = q _ { \mathrm { t o k e n } } ( z _ { e } )$ the single environment token. With text disabled, each DiT block receives the conditioning sequence $C _ { e } \stackrel { - } { = } \left[ u _ { e } ; A _ { \mathrm { t o k } } ( a ) \right]$ For video patch tokens $X _ { \ell } \in \mathbb { R } ^ { N \times D }$ entering block $\ell ,$ the attention operation can be written schematically as

$$
\operatorname { C A } ( X _ { \ell } , C _ { e } ) = \operatorname { s o f t m a x } \left( \frac { ( X _ { \ell } W _ { Q } ) ( C _ { e } W _ { K } ) ^ { \top } } { \sqrt { d _ { h } } } \right) ( C _ { e } W _ { V } ) W _ { O } .\tag{10}
$$

Here $N$ is the number of video patches, $d _ { h }$ is one attention head’s width, and $W _ { Q } , W _ { K } , W _ { V } , W _ { O }$ are the block’s learned attention matrices. The actual operation is multi-head; the equation shows where the new token enters. Every video patch can attend to the environment token as well as the action tokens through keys and values.

Modulation branch. BWM also groups four aligned 14-D actions at a time and maps each group to a D-dimensional action-modulation vector $a _ { q } ^ { \mathrm { m \bar { o } d } }$ . Index $g$ identifies the corresponding latentvideo temporal group, and s identifies a spatial patch within it. Let τ be the current flow-matching timestep and $b _ { g , s } ( \tau ) \overset { \mathbf { \sigma } } { \in } \mathbb { R } ^ { D }$ be Wan’s base timestep embedding at that video patch, before action or environment conditioning. The initial-frame latent uses the unnoised timestep, while predicted frame latents use $\tau .$ . The environment-modulation vector is $v _ { e } = q _ { \mathrm { m o d } } ( z _ { e } ) \in \mathbb { R } ^ { D }$ . The implementation adds these three vectors and applies the existing time projection:

$$
\tilde { b } _ { g , s } ( \tau ) = b _ { g , s } ( \tau ) + a _ { g } ^ { \mathrm { m o d } } + v _ { e } , \qquad M _ { g , s } = \mathrm { T i m e } \mathrm { P r o j } ( \tilde { b } _ { g , s } ( \tau ) ) \in \mathbb { R } ^ { 6 \times D } .\tag{11}
$$

The action and environment vectors are broadcast to every spatial patch in group g. The six $D _ { - }$ dimensional outputs of $M _ { g , s }$ supply the DiT blocks’ adaptive-normalization shifts, scales, and residual gates; the modified timestep embedding also conditions the output head. Thus $z _ { e }$ influences both attention and per-patch modulation, while leaving BWM’s original action pathways in place.

## A.2 TRAINING AND TEST-TIME ADAPTATION

Training schedule. Training uses the inherited flow-matching video-prediction loss, excluding the initial conditioning frame. A latent block freezes the shared model and updates the environment table; a model block freezes the table and updates the DiT, action encoder, and two projectors. The basic cycle alternates a 200-update latent block and a 200-update model block. An initial 300-update model block precedes the curriculum. The shared-model optimizer is AdamW with learning rate $1 0 ^ { - 5 } .$ , weight decay 0.01, 100 warmup updates, gradient clipping 0.5, and bfloat16 training. The environment-latent optimizer uses learning rate 0.03 and zero weight decay. Latents are independently initialized from U(0, 1) and clipped to [0, 1] after training updates.

Table 4: Model components and optimization roles. Test-time adaptation updates only the environment latent.
<table><tr><td>Component</td><td>Initialization or structure</td><td>Training update</td></tr><tr><td>Video DiT</td><td>BWM-initialized Wan2.2 backbone</td><td>Model blocks</td></tr><tr><td>Action encoder</td><td>BWM-initialized 14-D action pathway</td><td>Model blocks</td></tr><tr><td>Video VAE</td><td>Wan2.2 video encoder/decoder</td><td>Frozen</td></tr><tr><td>Environment table</td><td>Learnable 32-D latents indexed by environment</td><td>Latent blocks</td></tr><tr><td>Latent projectors</td><td>Two  $\mathrm { 3 2 } \to 1 2 8 \to 3 0 7 2 \mathrm { M L P s }$ </td><td>Model blocks</td></tr></table>

Test-time adaptation. For a held-out environment, a 32-D latent is optimized from its testtime interaction trajectories; all shared weights stay frozen. Query outcomes never enter this update. Both simulation and real-world main evaluations use 40 latent updates with learning rates 3.0, 1.5, 0.5, 0.15, each for ten updates. The resulting latent is reused across query actions from that environment. Exact task-specific execution configurations are released with the code.

## B EXPERIMENTAL SETUP

## B.1 TRAINING DATA AND RESOURCE BUDGETS

Tables 5 and 6 report counts of physical environments and recorded episodes, rather than latent entries. Each video chunk contains one initial conditioning frame and the remaining frames are prediction targets. The batch notation is training environments × action chunks per environment per optimizer update.

Table 5: Simulation training scale by physical environment. Diverse Backgrounds uses 30 scene backgrounds. Budget is H200 GPUs × hours per method.
<table><tr><td>Task Physical envs. × episodes/env. Frames</td></tr><tr><td></td><td>Stride Batch Budget 41 1  $8 \times 4$ </td></tr><tr><td>Friction  $3 5 \times 1 0$   $2 \times 2 4$   $2 0 \times 1 0$  61 1</td></tr><tr><td>Gravity  $8 \times 4 \quad 2 \times 2 4 \quad$  Mass Collision  $2 0 \times 9$  61 1  $8 \times 8 \quad 2 \times 2 4$ </td></tr><tr><td></td></tr><tr><td>Light Switch  $1 2 \times 1 0$  33 1  $8 \times 8 \quad 2 \times 2 4$ </td></tr><tr><td>Mass Balance  $2 0 \times 1 5$  41 2  $8 \times 8 \quad 2 \times 2 4$ </td></tr><tr><td>Mass × Friction  $8 4 \times 9$  61 1  $1 6 \times 6 4 \times 4 8$  Diverse Backgrounds  $1 5 \times 5 0$  41 1  $1 2 \times 6 4 \times 2 4$ </td></tr></table>

Light Switch episodes contain continuous button-press trajectories. Diverse Backgrounds varies friction across physical environments while the different scene backgrounds provide repeated interactions.

Table 6: Real-world training scale by physical environment and native/model-input RGB resolution. Budget is H200 GPUs × hours per method.
<table><tr><td>Task</td><td>Physical envs. / episodes Frames Stride Native RGB Input RGB</td><td></td><td></td><td></td><td></td><td>Batch Budget</td></tr><tr><td>Ball Friction</td><td>8 /240</td><td>61</td><td>1</td><td> $5 6 0 \times 2 4 8$   $3 2 0 \times 1 6 0$ </td><td> $6 \times 8$ </td><td> $2 \times 2 4$ </td></tr><tr><td>Door Close</td><td>10/ 438</td><td>49</td><td>1</td><td> $6 4 0 \times 4 8 0$ </td><td> $2 5 6 \times 1 9 2$ </td><td> $6 \times 8$   $2 \times 2 4$ </td></tr><tr><td>Stick Balance</td><td>9/421</td><td>33</td><td>3  $6 4 0 \times 4 8 0$ </td><td> $2 5 6 \times 1 9 2$ </td><td> $6 \times 8$ </td><td> $2 \times 2 4$ </td></tr><tr><td>Soft Pull</td><td>9/146</td><td>33</td><td>3</td><td> $5 1 2 \times 2 5 6$ </td><td> $3 2 0 \times 1 6 0$   $1 0 \times 6$ </td><td> $2 \times 2 4$ </td></tr></table>

## B.2 SIMULATOR AND REAL-WORLD SETUP

Simulator. We construct the custom tasks in a LIBERO-based simulator (Liu et al., 2023) with a Franka Panda arm. Simulated RGB videos are recorded at 20 fps and 224 × 224 resolution. The action representation has 14 channels.

Real world. The physical tasks use a Franka robot arm and a fixed Intel RealSense D435i camera. We use only its main RGB view, captured at $6 4 0 \times 4 8 0$ and 20 fps; the task-specific saved video crops and model-input sizes appear in Table 6. Model inputs are letterboxed to the listed sizes. All tasks use 14-channel action vectors; the target-action representation is task dependent.

## C EVALUATION PROTOCOLS

## C.1 SHARED PROTOCOL AND METRICS

Shared comparison. For action-selection tasks, we sample candidate actions and use each model’s predicted videos to make an action decision. The selected action is then executed in the environment, and success is determined from its actual outcome using the task-specific criteria below. Each evaluated candidate-action set contains at least one action that satisfies the target condition. All methods receive the same query inputs and candidate actions, and all methods that adapt at test time receive the same test-time interaction trajectories. The Standard World Model does not adapt at test time, and no method sees a query outcome before making its prediction.

For tracked object positions $p _ { t }$ and predictions $\hat { p } _ { t }$ over T evaluated future frames, average displacement error (ADE) and final displacement error (FDE) are

$$
\mathrm { A D E } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \Vert \hat { p } _ { t } - p _ { t } \Vert _ { 2 } , \qquad \mathrm { F D E } = \Vert \hat { p } _ { T } - p _ { T } \Vert _ { 2 } .\tag{12}
$$

Both are reported in image pixels. Tasks with scalar physical states use the corresponding mean/final absolute errors stated below. PSNR and SSIM measure full-frame appearance. LPIPS uses the official pretrained AlexNet perceptual metric on RGB frames scaled to [−1, 1], without an object mask; per-frame values are averaged over the evaluated future frames and then over query videos.

A query video denotes one action-conditioned video chunk evaluated at test time. The counts below sum query videos across the evaluated environments; K is the number of test-time interaction trajectories per test environment.

Table 7: Simulation evaluation counts: test-time interaction trajectories and query videos.
<table><tr><td>Task</td><td>K Query videos</td></tr><tr><td>Friction</td><td>1</td></tr><tr><td>Gravity</td><td>90 1 90</td></tr><tr><td>Mass Collision</td><td>1 80</td></tr><tr><td>Light Switch</td><td>2</td></tr><tr><td>Mass Balance</td><td>60 2 130</td></tr><tr><td>Mass × Friction</td><td>1 80 90</td></tr><tr><td>Diverse Backgrounds</td><td>1</td></tr></table>

## C.2 SIMULATED TASKS

Friction. We change the table’s surface-friction coefficient while the robot pushes the same block with different action strengths. We track the pushed block’s image centroid. Action Success means that its observed displacement reaches the requested short, medium, or long target region.

Gravity. We vary the simulator’s gravitational acceleration, changing the path of a blue block after it is thrown. Different actions correspond to different launch speeds of the block. We track the projectile’s centroid. Success means that the observed projectile endpoint reaches the requested near, middle, or far target region.

Table 8: Simulation object-centric quantities extracted from the predicted and ground-truth RGB videos. ADE/FDE are in image pixels.
<table><tr><td>Task</td><td>Tracked quantity</td><td>Object-centric error</td></tr><tr><td>Friction</td><td>Pushed-block centroid</td><td>ADE/FDE</td></tr><tr><td>Gravity</td><td>Blue projectile centroid</td><td>ADE/FDE</td></tr><tr><td>Mass Collision</td><td>Red struck-object centroid</td><td>ADE/FDE</td></tr><tr><td>Light Switch</td><td>Yellow-pixel fraction in lamp region</td><td>Mean/final absolute error</td></tr><tr><td>Mass Balance</td><td>Balance-bar tilt angle</td><td>(pp) Mean/final angle error (°)</td></tr><tr><td>Mass × Friction</td><td>Red struck-object centroid</td><td>ADE/FDE</td></tr><tr><td>Diverse Backgrounds</td><td>Pushed-block centroid</td><td>ADE/FDE</td></tr></table>

![](images/f145fcc7392ffade01df960f0ce903169313d3f6e83ca52919b412b28c32af56.jpg)  
(a) Friction

![](images/1e1b78cc7fe7de8c218aa6e8e19a456481d0d5d4254e008ccd9c06ef82a370be.jpg)  
(b) Gravity

![](images/270aa79d3c4aac4f01c5d201d77b856da58ed05c31401f73ec9ed1b3c3133d33.jpg)  
(c) Mass Collision

![](images/e6f1a589ed78a300f3b70595161828e2a9b2f740003668027292ef8ffc2d090e.jpg)  
(d) Light Switch

![](images/109648fac854302c78a70b5a25ce2950943172f72fce2c0872df99da0425e2a4.jpg)  
(e) Mass Balance

![](images/85500c761580f3e09401f2a1959b05fe12f560a5a5824bfc9a366b0540d2b968.jpg)  
(f) Mass × Friction  
Figure 7: Representative ground-truth frames from the simulated tasks. Diverse Backgrounds uses the Friction push task with varied scene backgrounds.

Mass Collision. We vary the mass of the red target object while a driver object collides with it under different actions. We track the struck red object’s centroid, not the driver’s. Success means that the struck object reaches the requested displacement region.

Light Switch. We vary the hidden causal rule specifying whether the red button, the blue button, both, or neither can change the lamp state. The object-centric signal is color, not average brightness: a fixed RGB rule detects yellow pixels in the lamp region R. Each frame’s score is

$$
y _ { t } = \frac { 1 } { \left| \mathcal { R } \right| } \sum _ { p \in \mathcal { R } } \mathbf { 1 } [ \mathrm { R G B } _ { t } ( p ) \ \mathrm { p a s s e s ~ t h e ~ y e l l o w - c o l o r ~ r u l e } ] .\tag{13}
$$

We report the absolute difference between predicted and observed scores in percentage points (pp). For T evaluated future frames,

$$
E _ { \mathrm { m e a n } } ^ { \mathrm { l i g h t } } = \frac { 1 0 0 } { T } \sum _ { t = 1 } ^ { T } | \hat { y } _ { t } - y _ { t } | , \qquad E _ { \mathrm { f i n a l } } ^ { \mathrm { l i g h t } } = 1 0 0 | \hat { y } _ { T } - y _ { T } | .\tag{14}
$$

A score of at least 0.35 denotes “on” for the terminal-state goal; success means that the observed terminal lamp state matches the requested on/off target.

Mass Balance. We vary the left/right mass ratio of the balance apparatus; the robot’s support position is the action used to probe that fixed ratio. We track the bar angle in degrees; its ADE/FDE analogues are mean/final angular errors. Success means that the observed terminal bar angle lies in the horizontal balance interval.

Mass × Friction. We combine different red-object masses with different table-friction coefficients, so both physical factors can change across environments. We track the red struck object’s centroid. For each target line, the desired action is the least forceful sampled action whose observed endpoint crosses that line; the selector estimates this minimum from the predicted videos.

Diverse Backgrounds (multi-background). We vary the table-friction coefficient as in Friction while rendering the same push task under different scene backgrounds; background is a visual change, not a new action. We again track the pushed block’s centroid; success means reaching the requested displacement region. This tests whether the shared environment representation follows dynamics rather than appearance.

## C.3 REAL-WORLD TESTS

Table 9: Real-world evaluation counts: test-time interaction trajectories and query videos.
<table><tr><td>Task</td><td>K</td><td>Query videos</td></tr><tr><td>Ball Friction</td><td>1</td><td>80</td></tr><tr><td>Door Close</td><td>2</td><td>100</td></tr><tr><td>Stick Balance</td><td>2</td><td>45</td></tr><tr><td>Soft Pull</td><td>1</td><td>24</td></tr></table>

Table 10: Real-world object-centric measurements from the main camera. ADE/FDE are reported in native-image pixels.
<table><tr><td>Task</td><td>Tracked quantity</td><td>Object-centric error</td></tr><tr><td>Ball Friction</td><td>Visible ball center</td><td>ADE/FDE</td></tr><tr><td>Door Close</td><td>Outer door-edge position</td><td>ADE/FDE</td></tr><tr><td>Stick Balance</td><td>Two end markers, averaged</td><td>ADE/FDE</td></tr><tr><td>Soft Pull</td><td>Dragged object&#x27;s visible center</td><td>ADE/FDE</td></tr></table>

Ball Friction. We vary the mass of steel beads placed inside the ball, changing its total mass and the sliding-friction force under a robot swing (Figure 8a). The ball variants in the latent-space visualizations correspond to these different steel-bead loads. We track the ball’s visible center for native-pixel ADE/FDE. Success means that the selected swing moves the ball to the right-hand blue-marker target.

Door Close. The door has a spring-loaded latch; we replace the spring to change how much closing action is needed to engage it (Figure 8b). The tracked outer door edge, rather than a centroid or hinge angle, defines native-pixel ADE/FDE. Success means that the selected action fully closes and latches the door.

Stick Balance. We change the weights attached to the stick’s ends, altering its mass distribution and thus the angle at which it settles after lifting (Figure 8c). We track the two material markers at its ends and average their positional errors for ADE/FDE. We sample a set of candidate actions and use the model to predict which actions will maintain balance. One action is selected uniformly at random from the predicted balancing set and executed on the real robot. A trial is successful if the stick actually maintains balance. Each sampled candidate set contains at least one action that

![](images/3f3276045dd9d06f8b2816f2c29024ca0b5f5ce682dad141b3c20f01e854934b.jpg)  
(a) Steel-bead loads for the ball variants

![](images/5dd0be8ca74de0532e2a58ce60b690b5d37144782607925e54a1db745eb43e3f.jpg)

##

(b) Door latch and alternative springs  
![](images/3a92df5917d49cc43a0b6d6a0d4d9ee5c9aff411f98366b51956ed4d719b88d4.jpg)  
(c) Stick with adjustable end masses

![](images/de7511a8dc0d7d205d48dc511aa592f250659cbef8439bf8efe8a0efa0238aaa.jpg)  
(d) Soft tubes with different stiffness and curvature  
Figure 8: Physical designs of the four real-world tasks. The Ball task changes the number of stee beads inside the ball (not visible in the video frame), and the Door task uses a spring-loaded latch.

achieves balance. If the model predicts no balancing action, the trial is counted as a failure. We report the fraction of successful trials.

Soft Pull. The task uses soft tubes with different stiffness coefficients. The robot may grasp from the left or right, and the tubes also have different natural curvatures (Figure 8d). The central prediction question is approximately how far the grasp must rotate before the tube starts dragging the object. The visible object center defines native-pixel ADE/FDE. Given a candidate action, the model predicts the rotation angle $\hat { \theta }$ at which the object starts sliding. We then conduct two real-robot tests, rotating to $\hat { \theta } - 6 { } ^ { \circ }$ and $ { \hat { \theta } } + 6 ^ { \circ }$ , respectively. A trial is successful only if the object does not slide in the first test and slides in the second; otherwise, it is counted as a failure. We report the fraction of successful trials.

## D BASELINE IMPLEMENTATION DETAILS

All four baselines use the task-level training data and per-method GPU budgets in Tables 5–6. They use the same query videos and, for action-selection tasks, the same candidate actions and success criteria as T2-Z. The three adaptive baselines receive exactly the same test-time interaction trajectories as our latent optimizer. The tables give the common task-level batch specification; DINOv2 Context uses a subset of sampled trajectories for conditioning and all sampled trajectories as queries during training, whereas TTT-KQV processes them as ordered within-environment streams.

## D.1 STANDARD WORLD MODEL

This baseline fine-tunes the BWM-initialized DiT and action encoder on pooled task trajectories without an environment latent or test-time adaptation. It uses AdamW with model learning rate $1 0 ^ { - 5 }$ , weight decay 0.01, 100 warmup updates, gradient clipping 0.5, and bfloat16.

## D.2 LORA TTA

Starting from the Standard World Model, rank-8, $\alpha = 8 \mathrm { L o R A }$ adapters (Hu et al., 2022) are placed on the DiT attention Q/K/V/output projections. The backbone stays frozen during test-time adaptation. Adapters are reset for each environment and optimized on its test-time interaction trajectories, then held fixed across that environment’s query actions. The Friction configuration uses 50 AdamW adaptation updates at learning rate $1 0 ^ { - 4 }$ , zero weight decay, and gradient clipping 1; task-specific adaptation configurations are released with the code.

## D.3 DINOV2 CONTEXT

A frozen DINOv2-B/14 encoder (Oquab et al., 2024) extracts conditioning-frame features, while a learned 128-D action GRU summarizes the intervening 14-D actions. For each of N task-configured sampled frames, the CLS and mean patch features are projected to 256 dimensions. The N − 1 adjacent visual–action transition features (640-D each) are concatenated; a $6 4 0 ( N - 1 ) {  } 1 0 2 4 {  } 1 0 { \dot { 2 } } 4$ MLP and final projection produce the 32-D context. Multiple conditioning-trajectory summaries are averaged before that final projection. At each training update, we sample a set of trajectories $Q _ { e }$ from a training environment and select a subset $C _ { e } \subseteq { \bar { Q _ { \epsilon } } }$ for conditioning. All trajectories in $Q _ { e }$ serve as training queries. The number of conditioning trajectories is sampled from the taskconfigured choices (typically one or two; one to four for Soft Pull), so the predictor learns to use different conditioning-set sizes. Every query in $Q _ { e }$ is predicted using only the context inferred from $C _ { e } ;$ training minimizes the mean query-video flow-matching loss. The context head, BWM DiT, action encoder, and two context projectors are trained jointly, while DINOv2 remains frozen. The model and context-head learning rates are $1 0 ^ { - 5 }$ and $1 0 ^ { - 4 }$ , respectively. At test time, context comes from a forward pass over the observed test-time interaction trajectories, with no gradient update.

## D.4 TTT-KQV

We adapt a video test-time-training fast-weight module (Dalal et al., 2025) to the BWM backbone. Gated two-layer GELU MLP modules with expansion factor four are inserted after eight approximately evenly spaced attention blocks. Training randomly shuffles the chunks from each sampled environment in a batch. Within each training stream, all chunks use the same sampled flow-matching timestep (but independent noise), matching the same-timestep processing of observed interactions and query videos at inference. At position i, the preceding chunks form the observed prefix and the current chunk is the query. Fast weights reset to a learned shared initialization at the start of each stream. For each chunk, the module causally writes its key–value tokens before reading its output. Let $x _ { t }$ denote an input token, h an attention head, and $d _ { h }$ its feature dimension; t indexes tokens, not flow-matching timesteps. Learned key and value projections produce $k _ { h , t } , v _ { h , t } \in \mathbb { R } ^ { d _ { h } }$ . The fast state $W = \{ W _ { h } \bar  \} _ { h }$ contains each head’s MLP weights and biases. Its per-token binding loss and learned step scaling are

$$
\begin{array} { c } { \ell _ { h , t } ( W ) = \displaystyle \frac { 1 } { 2 d _ { h } } \left\| k _ { h , t } + \mathrm { L N } ( \mathrm { M L P } _ { W _ { h } } ( k _ { h , t } ) ) - v _ { h , t } \right\| _ { 2 } ^ { 2 } , } \\ { \eta _ { h , t } = \displaystyle \frac { \eta _ { \mathrm { b a s e } } } { d _ { h } } \sigma ( w _ { h } ^ { \top } x _ { t } + b _ { h } ) , \qquad \eta _ { \mathrm { b a s e } } = 0 . 1 . } \end{array}\tag{15}
$$

Here LN normalizes within each head, and σ is the sigmoid function. The factor $1 / d _ { h }$ in the loss averages squared errors over feature dimensions; the separate factor $1 / d _ { h }$ in $\eta _ { h , t }$ scales the update. For update $j ,$ let $B _ { j }$ be the current minibatch of at most 64 tokens. The differentiable fast-state update is

$$
W ^ { j + 1 } = W ^ { j } - \nabla _ { W } \sum _ { h } \sum _ { t \in \mathcal { B } _ { j } } \eta _ { h , t } \ell _ { h , t } ( W ) \Bigg | _ { W = W ^ { j } } .\tag{16}
$$

We use a base inner learning rate of 0.1 with learned token-wise, head-specific scaling, rather than a constant update step size. The scaling parameters $w _ { h } , b _ { h }$ are shared slow weights; the inner update changes only $W .$ . Gates start at 0.1. The outer loss averages the video flow-matching losses over all stream positions and sampled environments. Its gradients update the shared slow weights, using backbone LR $1 0 ^ { - 5 }$ and TTT-module LR $1 0 ^ { - 4 }$ ; the fast state itself is ephemeral. At inference, every denoising timestep resets the fast state and replays the observed test-time interaction trajectories at that same timestep. Each query then takes an independent update–then–read branch on its own noisy tokens. Query ground truth never enters the update, and fast state is not carried between queries or denoising timesteps. Exact stream packing and task-specific execution settings are released with the code.

## E ADDITIONAL EXPERIMENTS

## E.1 TEST-TIME INTERACTION-COUNT ABLATION

Table 11: Numerical results for Figure $5 ( \mathrm { c } , \mathrm { d } )$ at $K = 1 , 2 ,$ 4 test-time interaction trajectories. Bold marks the selected K. Physical error is in pixels except for Light Switch (lamp-score error in percentage points).
<table><tr><td rowspan="2">Task</td><td colspan="3">Physical error (mean) ↓</td><td colspan="3">Action success (%) ↑</td></tr><tr><td> $K = 1$ </td><td> $K = 2$ </td><td> $K = 4$ </td><td> $K = 1$ </td><td> $K = 2$ </td><td> $K = 4$ </td></tr><tr><td>Mass Collision</td><td>6.452</td><td>6.534</td><td>5.946</td><td>61.1</td><td>66.7</td><td>72.2</td></tr><tr><td>Light Switch</td><td>8.26</td><td>3.43</td><td>3.41</td><td>50.0</td><td>87.5</td><td>87.5</td></tr><tr><td>Door Close</td><td>11.337</td><td>7.954</td><td>6.889</td><td>22.2</td><td>55.6</td><td>55.6</td></tr><tr><td>Soft Pull</td><td>18.710</td><td>18.363</td><td>18.060</td><td>81.8</td><td>81.8</td><td>81.8</td></tr></table>

## E.2 LATENT DIMENSIONALITY

Table 12: Environment-latent dimensionality on Friction under the same $K = 1$ evaluation protocol (ten held-out physical environments; 90 query videos). Object ADE/FDE are main-view centroid errors in pixels; video metrics use the concatenated main and wrist views. Action success is evaluated on 25 reachable target decisions. Best values are bold.
<table><tr><td>Latent dim.</td><td> $\mathrm { \ A D E { \downarrow } }$ </td><td> $\mathrm { F D E \downarrow }$ </td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Success↑</td></tr><tr><td>8</td><td>8.92</td><td>19.26</td><td>31.293</td><td>0.9474</td><td>0.0646</td><td>44%</td></tr><tr><td>16</td><td>7.63</td><td>15.90</td><td>29.557</td><td>0.9470</td><td>0.0614</td><td>64%</td></tr><tr><td>32 (ours)</td><td>3.85</td><td>7.59</td><td>31.976</td><td>0.9513</td><td>0.0495</td><td>72%</td></tr><tr><td>64</td><td>6.86</td><td>13.98</td><td>31.692</td><td>0.9504</td><td>0.0551</td><td>64%</td></tr><tr><td>128</td><td>7.08</td><td>14.51</td><td>31.372</td><td>0.9483</td><td>0.0581</td><td>64%</td></tr></table>

## E.3 LATENT-SPACE PCA ACROSS TASKS

Figure 9 shows the latent-space analysis for four simulation and four real-world tasks. Circles show training-time environment latents, and triangles show latents adapted from test-time interaction trajectories. Colors indicate the underlying physical setting or causal class for visualization only; these labels are not provided to the model. For each task independently, we center its training-time 32-D latents by their mean z¯, compute the leading two right singular vectors $V _ { 1 : 2 }$ of the centered matrix without feature rescaling, and plot $( z - \bar { z } ) \bar { V _ { 1 : 2 } }$ . Test-time latents are projected using the same mean and basis; they do not enter the PCA fit. Thus the PC axes are task-specific rather than shared across panels.

## E.4 COMPUTATIONAL EFFICIENCY

We profile a representative Friction setting on one H200 GPU. For training, each method is warmed up for two optimizer steps and timed for ten further steps. Figure 10 reports the median computeonly optimizer time per update, excluding dataset reads, and peak allocated GPU memory rather than full-run training time. Each measured update has six loss-bearing video chunks; method-specific conditioning computation remains included. The world-model and latent updates of our method are separate phases and are therefore shown separately. LoRA TTA starts from the Standard World Model and has no distinct world-model training row.

For inference, all methods receive the same test-time interaction trajectory and five disjoint query video chunks. We use 25 denoising steps and exclude model loading, dataset reads, and MP4 writing from timing; query generation includes pipeline sampling and VAE decoding. Interaction processing is performed once and reused across queries. DINOv2 Context encodes the interaction trajectory without gradient updates, our method runs 40 latent updates, and LoRA TTA runs 50 adapter updates. TTT-KQV precomputes one post-interaction fast state at each denoising timestep and reuses those states for all five queries. Figure 11 separates interaction processing from five-query generation and shows how processing time changes with $K \in \{ 1 , 2 , 4 \}$ ; the five-method total-time comparison uses $K = 1$ throughout. $\mathbf { A } \mathbf { t } \ K = 1$ , our method takes 18.09 s including latent adaptation and five query generations, versus 32.73 s for LoRA TTA and 37.57 s for cached TTT-KQV.

(h) Soft Pull  
(d) Diverse Scenes  
Test-time-adapted latent (b) Mass Balance  
![](images/8fdfd7a53d059e19c9fdbd4204ce6c78d1d911fdfb47efc0f82a3d3107cabf14.jpg)

![](images/c40cac87b36d15c3c395a7ba53bf5cd981ed5ef83ca0db3608e5e247e3e14fb0.jpg)

![](images/633859ea5d2294ce34e535d3e2553bd8db1acbaab658c852616d12ae70230d0a.jpg)

![](images/2054bd74b074991b6238f567bbf4a860057fa17b30bfd5534583cdac5a330b65.jpg)

![](images/040325888a7ffcb985ad74f8eb687d3790d8e22ce20d4a01e6ee8ee0c54e4856.jpg)

![](images/81aa9cede859c113e660491b63cb6910dd7cf3e0bdc3efc447bf86b2d9212c24.jpg)

![](images/179af29860d817419f3ff47c3d4bbe4b74ac10f5de6b038ccddec300b2bc7475.jpg)

![](images/1d35a5f9c6d78e6ee85bfc6c203d2783708408c716e587d89eb7feafe43d5d7d.jpg)  
Figure 9: Latent-space PCA across simulation (a–d) and real-world tasks (e–h). Circles denote training-time latents and triangles denote test-time-adapted latents. Color indicates the underlying physical setting; point size additionally indicates friction in Mass × Friction. These factor annotations are used only for visualization and are not provided to the model. The Soft Pull test-time latents shown here use four test-time interaction trajectories.

![](images/1180814c649047bb9ed353477d1314017f5ed9ffe505939a5a0d915c4ecb84a8.jpg)

![](images/449264dc46455de78bcfe53836ca6654ddcc3c878956117b759cbb3304a2cb1a.jpg)  
Figure 10: Training resource profile. Time is the median compute-only optimizer duration over ten measured updates after two warmup updates, excluding dataset reads; memory is peak allocated GPU memory. Each update uses six loss-bearing video chunks.

(a) Total inference time at K=1  
![](images/ab32087251fe3688b8308408b09c35eed20272e70fdf8205bdc090d31e23f988.jpg)

(b) Reusable interaction processing as K grows  
![](images/263d42e6828f4117478469c321b55c5da631fc83561166ff3b3bfa062c1c57ff.jpg)  
Number of test-time interaction trajectories  
Figure 11: Compute-only inference profile on one H200 GPU, excluding model and sample loading and MP4 writing. Top: processing of one test-time interaction trajectory plus generation of the same five query videos. Bottom: interaction processing time for K = 1, 2, 4 with the update counts held fixed within each method. The bottom panel does not compare query-generation time at K > 1.

## F ADDITIONAL QUALITATIVE RESULTS

Figures 12–22 show selected held-out test actions. Each task shows two test environments and four actions per environment; Soft Pull shows two actions per environment. Each column uses the same initial observation and candidate action across methods; rows show ground truth, the four baselines, and our prediction. Simulation panels show the main camera only.

![](images/75004a44bc4f0c8a02d5b79c896e8361a85b91553df1f520f7ce6fc63004fcce.jpg)  
Figure 12: Friction: selected test actions.

![](images/28dd42b2447d17926a4a0f8efef689caedc23226bd53ff58a6950a0b9a135254.jpg)  
Figure 13: Gravity: selected test actions.

![](images/1c5f2b6a67d7d7ed6c1c58c5ae85ee44acf87700e299599a30435768d60d250d.jpg)  
Figure 14: Mass Collision: selected test actions.

![](images/2be87e6d0d28f58d82759bc81ba325211fd0739522f58f134c93a27d153b03dc.jpg)  
Figure 15: Light Switch: selected test actions.

![](images/ca0361ea59b01ccdfee662ed4ee43398fcd58bfa25fde5892a86d3e406fcdafe.jpg)  
Figure 16: Mass Balance: selected test actions.

![](images/8e941f046fdf0b46b9e0a764e9d736029fc6cd648811ce08ade1dedc8446ba3a.jpg)  
Figure 17: Mass × Friction: selected test actions.

![](images/2ebcc51703a99c162c75b96e9c149acdccb3738273a7c2d997aff7c00cbfb20c.jpg)  
Figure 18: Diverse Backgrounds: selected test actions.

![](images/0dc96549ff379c0f9a5d0e97b04f00676c49e546cea50fcb444736992c298c3b.jpg)

![](images/2c147aa9d0370c9a0696b574684dd8ec1c9b4ab415609312de5de9814d84a232.jpg)  
Figure 19: Ball Friction: selected test actions.

![](images/8ca3c7c148628676cc4bad8f359b86581d3ff3fca204f839823b1475d33f706e.jpg)

![](images/027781f8aca7e2cd21858f8980b7c69eef3b151246b238883ca121903727a742.jpg)  
Figure 20: Door Close: selected test actions.

![](images/1c69b2871e28a1548e6a1dedffa02f2a244bb42dc578deba07840330703d252f.jpg)  
Figure 21: Stick Balance: selected test actions.

![](images/24d3913ea7d248a7d474a9ccf1ff03bab0b46b879f3ef1dbe2ccb634b39f0392.jpg)  
Figure 22: Soft Pull: selected test actions.

## REFERENCES

BWM Team. BWM: A low-cost high-fidelity world simulator for robot learning. arXiv preprint arXiv:2607.29302, 2026.

Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17702– 17711, 2025.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2023.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov,´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve J´ egou, Julien Mairal, Patrick Labatut,´ Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.