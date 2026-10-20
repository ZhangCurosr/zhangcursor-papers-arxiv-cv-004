# Where is Physical RSI capability from? An Observational Scaling Law of LLM-as-policy

Zekai Li<sup>∗</sup>, Jiayi Yu<sup>∗</sup>, Jinlu Zhang<sup>∗</sup>, Luyang Niu<sup>∗</sup>, Zekun Qi<sup>∗</sup>, Jun Zheng, Xuchuan Chen, Dairu Liu, Yintianrun Zhang, He Wang<sup>†</sup>, Li Yi<sup>†</sup>

Galbot Team

In this paper, we systematically investigate the mechanisms, scaling behavior, and generalization of physical Recursive self-improvement. We introduce PhysRSI-eval to quantify physical RSI through a five-axis metric: First-Attempt Performance, Eficiency, Success, Budget and Improvement from failure. Diferent VLMs are evaluated on 18 manipulation tasks under a fixed interaction budget. The analysis of experimental results motivates the Physical Barrel Hypothesis: diferent capabilities may constrain diferent parts of improvement. Systems with similar public benchmark averages can still difer widely on the five axes, and a system that reaches a high final score may improve late or consume more of its budget. We study observational scaling from two perspectives: predicting the relative performance of VLM agents from their public capabilities, and forecasting later progress from early physical feedback. Together, these analyses ask how quickly and how far an agent can improve. Moreover, we conduct workspace interventions and then analyze what improvement leaves behind. Removing history reduces late-round success, frozen workspaces remain useful on unseen layouts, and three cross-task cases achieve success earlier. The results suggest that scene-specific adaptation and reusable experience can coexist.

Date: October 20, 2026 Project Page: https://physrsi-eval.github.io/

GALBOT

## 1 Introduction

Frontier LLM agents demonstrate remarkable capabilities in directly generating actions for robotic manipulation. Through iterative trial and error, they continually improve manipulation ability by updating skills, generating code, and using tools. [22, 9] We call this process physical recursive self-improvement, or physical RSI. In this work, rather than further demonstrating the impressive capabilities of frontier LLM agents, we pose three more fundamental questions: What is the origin of physical RSI capabilities? What scaling laws govern this process? Do the resulting improvements reflect overfitting or generalization?

Q1: Where is Physical RSI capability from? What allows an agent to turn physical feedback into a better policy? Perception helps it interpret outcomes, spatial reasoning relates observations to actions [31], and coding and harness use allow it to implement and test revisions. These capabilities may play diferent roles. We ask whether strengths behind a good first attempt also explain improvement after feedback.

We introduce PhysRSI-eval to study these questions under a common protocol (Figure 2). Six systems revise persistent code, tools and notes across 18 manipulation tasks, with frozen model weights and a fixed physical interaction budget. Gemini 4 is reported separately on ten tasks.

An exploratory comparison of four public capability profiles on the 18 tasks associates spatial reasoning most strongly with initial performance, and perception with attained performance and recovery (Figure 4). This pattern motivates the Physical Barrel Hypothesis: the capability that limits an agent may depend on which aspect of improvement we measure. Starting well and learning from failure may require diferent strengths. These associations suggest candidate bottlenecks, not causal requirements.

Q2: How can we quantify Physical RSI and its scaling law? An endpoint score alone cannot quantify physical RSI. It says little about when gains occur or what they cost. Repeated attempts can also increase the chance of success without useful revision. Our five-axis metric separates initial performance, gain timing, attained score, physical budget savings and recovery from failure (Section 3.4). These axes reveal diferent strengths. Opus 5.5 attains a higher score than GPT-6.1 Sol, but has lower gain-timing and budget-savings scores. Higher scores need not mean improving earlier or using fewer physical interactions.

![](images/f288a8cd13797df293e1db5fc7aa4cb5ffb60564e3c6351074e2c816fa6b456d.jpg)  
Figure 1 PhysRSI-eval at a glance. (Q1) Capability sources and the Physical Barrel Hypothesis. (Q2) Five-axis quantification and observational scaling of performance, gain timing and model rankings. T<sub>50</sub> is the half-gain round, the first round at which the best mean score reaches half of its K1–K17 gain. (Q3) Overfitting and generalization across tasks and scenes.

With these outcomes defined, we study observational scaling across model capabilities and physical interaction budgets. Following Ruan et al. [34], we first ask whether public capability scores can predict the relative order of held-out systems, in attained performance and gain timing (Section 5.2).

We then ask whether early physical progress predicts the gains from a larger interaction budget. This connects physical RSI to the study of learning curves [19, 12], with the interaction round as our budget variable. Calibrating the improvement shape of a reference model to a few early rounds forecasts the rest of a related model’s trajectory (Figure 8). The two analyses address complementary questions: what improvement to expect from a model’s capabilities, and how much further it may progress after its first attempts. Both remain exploratory.

Q3: Does Physical RSI reflect overfitting or generalization? Does the agent acquire a reusable procedure, or adapt to its practice scene? A rising score alone cannot answer this question. We first remove history to test whether retained experience helps beyond repeated attempts. We then freeze workspaces on unseen layouts and transfer them across tasks.

In a six-task Astra ablation, removing history lowers late-round success from 41.7% to 8.3% (Table 8). With further revision disabled, frozen workspace files support success in 25 of 60 trials on unseen layouts (Table 9). Three cross-task cases also reach first success earlier when initialized from other tasks workspaces (Figure 9). These results support a role for retained experience and show that some artifacts remain useful beyond the scenes where they were refined. Scene-specific gains and useful transfer can coexist. Explaining which retained skills or memories enable transfer connects this question back to the origins of physical RSI.

Our contributions follow this progression:

• Five-axis evaluation and capability diagnosis. A fixedbudget protocol separates five physical RSI outcomes and relates them to four capability profiles, motivating the Physical Barrel Hypothesis.

• Observational scaling and prediction. Cross-model rank prediction and early-round forecasting probe how model capabilities and interaction budget relate to physical improvement.

• Specialization and generalization. Ablations, frozenworkspace tests and cross-task adaptation examine whether gains depend on retained experience and whether that experience remains useful elsewhere.

## 2 Related Work

## 2.1 LLM-as-policy Agents and Embodied Evaluation

LLMs can act as robot policies by writing programs over perception and control tools [22]. Feedback lets them revise plans or reward code [39, 14, 26]. Recent benchmarks evaluate these policies through controlled robot interfaces [7, 24, 5, 57, 52], while others give coding agents extended simulator access and assess the resulting programs [35, 25]. These evaluations typically summarize outcomes per episode, across independent trials or within a time budget. A reported Astra case, where a self-written fix succeeded once and failed on repetition, highlights the gap between finding a solution and retaining reliable behavior [9]. PhysRSI-eval tracks judged physical attempts connected by a persistent workspace, measuring both progress and physical cost under a fixed budget.

## 2.2 Agent Self-Improvement

Agents can accumulate skills and reflections without updating their model weights [43, 37, 27]. Recent benchmarks examine whether changes to the harness or retained experience improve later performance [13, 46, 38, 58]. Producing an update and benefiting from it are distinct questions [23]. Related designs test memory, transfer and gains relative to cost [36, 49, 10, 17, 20, 47]. AI-R&D benchmarks address research engineering under time or compute budgets [45, 1, 40, 32, 4]. These designs motivate separating initial ability, improvement and cost, and testing retained experience against repeated attempts. PhysRSI-eval brings these questions to physical interaction. Native task scores evaluate each attempt, while workspace interventions test what the agent retains. Clearing the workspace tests whether accumulated experience helps; freezing and transferring it test whether its benefits extend to new layouts and tasks.

## 2.3 Capability Benchmarks and Observational Scaling

Public benchmarks provide candidate measures of perception [8, 54, 33], spatial reasoning [2, 50, 6, 16], coding and tool use [18, 44, 15, 28, 48, 53], and broader reasoning and work performance [30, 42, 29]. Embodied benchmarks and VLA models address planning and physical reasoning [51, 21, 11, 56, 41], but do not track policy revision from graded attempts. Observational scaling relates benchmark-derived capability factors to agent performance across models [34]. We predict the relative order of physical improvement, in both extent and timing, from public scores. We also forecast later gains from early physical performance. With few closed systems and no training-compute data, these analyses remain observational and exploratory.

## 3 PhysRSI-eval: Benchmark Design

PhysRSI-eval makes the three questions measurable by recording both physical attempts and the workspace carried between them. The attempt sequence reveals where an agent starts and how it progresses under a fixed budget. Workspace interventions test what agents retain and whether it transfers.

## 3.1 What Must the Benchmark Distinguish?

A strong first attempt and useful revision are diferent abilities. We therefore score a first-attempt baseline separately and follow subsequent attempts under a fixed budget. Per-round scores show when gains occur; charged physical steps show what they cost. Reaching a high score, reaching it early and using fewer interactions are separate outcomes.

Success alone does not show whether the agent has accumulated useful experience. Code, tools and notes must remain accessible for intervention. Clearing the workspace tests whether retained experience helps; freezing it on unseen layouts tests whether it remains useful; transferring it to new tasks tests whether it can support a diferent goal. We report these tests separately from the main profiles in Section 6.

## 3.2 A Controlled Loop of Physical Attempts

Model weights stay fixed, while skills, tools, policy code and memory can change in a persistent workspace. Each revision is tested through a judged physical attempt (Figure 2). This connects workspace changes to an observed outcome and a physical cost.

K0 measures the starting point without a persistent workspace. Each task runs three times through the shared robot tools, and its scores are averaged. At K1, the agent receives a Harness workspace that is empty except for the host’s read-only tools. Each round executes one physical attempt and returns the native score, success flag and failure reason. The agent then revises its workspace with robot actions disabled and commits it for the next round. The main evaluation allows at most H = 17 refinement rounds per task and stops at the first native success.

Both stages share the robot interface. K1 also introduces workspace tools and program execution, so the K0-to-K1 gain mixes interface changes with improvement. We therefore study the Harness rounds as the self-improvement segment. Each round allows 4200 s of reasoning; revision takes up to 1800 s for GPT systems and 600 s for Claude systems (Table 3).

For aggregation, successful tasks retain score 1 after stopping, without further execution. Unsuccessful tasks retain their current scores, including regressions. Extra attempts enter the physical budget but add no scoring rounds. Nine GPT-5.6 Sol Harness entries summarize two observed runs. The three-repeat K0 protocol applies to all six systems.

## 3.3 Tasks across Difficulty Levels

The task suite must expose both immediate competence and room for improvement. We evaluate the loop on 18 RoboDojo dual-arm manipulation tasks. They cover language-conditioned arrangement, stacking, container placement, insertion, pouring, pushing and dynamic picking (Figure 3 and Table 2). The suite spans RoboDojo’s five oficial task categories (Precision, Long-Horizon, Memory, Open and Generalization) and samples three tasks from each of six dificulty strata. The strata are assigned from the oficial RoboDojo success rates [3]: the 18 tasks are ranked by success rate and grouped three per stratum, from D1 (success rate 60–68.7%) to D6 (success rate 0–0.4%). This balances coverage across dificulty levels. The strata are fixed before the main evaluation and are not derived from our refinement outcomes. All 18 tasks, including object packing, receive equal weight. Each retains its native score in [0, 1], success criterion and step limit of 300–1300 steps at 25 Hz. The main evaluation uses layout 0 and seed 0. Partial scores expose progress before the first success.

We evaluate six systems: GPT-6 Astra, GPT-6.1 Sol, GPT-6 Sol, GPT-5.6 Sol, Claude Opus 5.5 and

Claude Sonnet 5.5. Each system comprises a model and its runtime configuration, which can also afect performance (Table 3). In a separate experiment without supplementary hints, Astra solves 13 tasks, compared with 15 in the main evaluation. Results from these conditions are reported separately.

Gemini 4 is reported in a separate partial run covering ten tasks through K20, with six solved. Its reported mean scores are 0.050 at K0 and 0.48 at K10. Its fiveaxis profile is computed on these ten tasks (Table 1 and Figure 5). Because the task set difers from the main 18-task protocol, it is excluded from the benchmark-based association and prediction analyses.

## 3.4 A Five-Axis Profile of Improvement

To study how physical RSI scales, we need more than an endpoint score. The five-axis profile separates the starting point from gain timing, attained score, physical budget savings and recovery from failure. Each axis measures a diferent aspect of improvement.

Let $q _ { k }$ be the task-mean score at round $k ,$ with q<sub>0</sub> computed from three-run K0 means. Let $C _ { k }$ be the fraction of tasks solved in Harness rounds K1–Kk.

A1: First-Attempt Performance. How well does the agent perform on its first attempt? $A _ { 1 } = q _ { 0 }$ is the mean K0 score, averaged over three runs per task and then across tasks, including partial progress.

A2: Efficiency. How early do gains appear? The timing of gains relative to the terminal net gain is

$$
A _ { 2 } = \frac { \frac { 1 } { H - 1 } \sum _ { k = 1 } ^ { H - 1 } q _ { k } - q _ { 0 } } { q _ { H } - q _ { 0 } } .\tag{1}
$$

Earlier sustained gains give higher values. The axis uses K0 as its reference and does not measure wallclock speed. It is undefined without positive terminal gain; nonmonotone curves are neither clipped nor replaced by best-so-far scores.

A3: Success. What score does the agent attain? Let $q _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { 0 \leq k \leq H } q _ { k }$ and define

$$
\tau _ { 9 0 } = \operatorname* { m i n } \{ k \geq 1 : q _ { k } \geq q _ { 0 } + 0 . 9 ( q _ { \operatorname* { m a x } } - q _ { 0 } ) \} .\tag{2}
$$

Then $\begin{array} { r } { A _ { 3 } = ( H - \tau _ { 9 0 } + 1 ) ^ { - 1 } \sum _ { k = \tau _ { 9 0 } } ^ { H } q _ { k } , } \end{array}$ : the mean over the high window, with partial credit. It is neither a binary success rate nor an asymptotic ceiling.

A4: Budget. How much physical budget is saved? For a solved task $t ,$ let $L _ { t }$ be its per-round step limit, $W _ { t }$ its cumulative canonical steps through first success, $Z _ { t }$ the unused steps charged for failed reasoningtimeout rounds, and $e _ { t }$ its charged extra attempts.

![](images/35aa15416072863bba8eeb51d2c70468ff1b8a1086342a66cdcb14785ab91a38.jpg)  
Figure 2 Two stages, one interface. Top: the task input and robot interface shared by both stages (observations, physical actions, memory and files, history). Left, Evaluate (first-attempt round K0): one in-context episode of observe–reason–act through the shared tools, with no workspace; history and notes stay inside the episode. Right, Optimize (self-improvement rounds K1–KH): each round runs from a persistent workspace of skills, tools, policy and memory, is graded natively, revised with robot actions closed, and committed as the start of the next round, until the first native success or the round limit.

Define $U _ { t } = ( W _ { t } + Z _ { t } ) / L _ { t } + e _ { t }$ . For the model’s solved-task set S,

$$
A _ { 4 } = \frac { 1 } { | S | } \sum _ { t \in S } \operatorname* { m a x } ( 0 , 1 - U _ { t } / H ) .\tag{3}
$$

Higher values mean more of the physical budget was saved. Unsolved tasks do not enter this conditional mean, and the axis is undefined if S is empty. Time, tokens and money are not part of it.

A5: Improvement. Which early failures are overcome? $A _ { 5 } = ( C _ { H } - C _ { 1 } ) / ( 1 - C _ { 1 } )$ is the fraction of K1- failed tasks subsequently solved within the budget. It is undefined when all tasks succeed at K1. The conditional task sets in A4 and A5 vary across models. Each axis is reported as a point estimate over the task suite (Appendix B).

## 4 Where is Physical RSI capability from?

What allows physical feedback to become a useful policy revision? Acting, interpreting an outcome and implementing a change may draw on diferent capabilities. We use the five-axis profile to ask whether the capabilities associated with a strong first attempt also track efective improvement.

## 4.1 Which Capabilities Track Physical RSI?

Strong coding scores alone do not ensure physical improvement. GPT-5.6 Sol exceeds GPT-6 Astra on LiveBench code generation and completion, yet solves one of the 18 tasks, whereas Astra solves 15.

The exploratory analysis uses six systems on all 18 tasks. We group 17 public benchmark scores into Spatial, Harness, Perception and Coding (Table 6), then compare their ranks with A1–A5 (Figure 4).

![](images/2eade595442c773236a33e8aa1db65e5ea538ffd8e3379751d8b6de7b4b8e89b.jpg)  
Figure 3 Difficulty-balanced task sampling. Three tasks per dificulty stratum, from D1 (highest oficial RoboDojo success rate) to D6 (lowest), covering RoboDojo’s five oficial task categories. Strata are assigned by ranking the tasks on the oficial RoboDojo success rate [3] and are fixed before the main evaluation; they do not describe the subsequent refinement outcomes.

Spatial reasoning has the strongest observed association with A1 $( \rho = 0 . 9 4 )$ and A2/A4 $( \rho = 0 . 8 3 )$ . Perception has the strongest association with A3 and A5 $( \rho = 0 . 9 4 )$ . Coding has the weakest association with every axis $( \rho \leq 0 . 3 7 )$

## 4.2 The Physical Barrel Hypothesis

The pattern motivates the Physical Barrel Hypothesis: diferent aspects of improvement may have diferent capability bottlenecks. Spatial reasoning may matter most for starting well, while perception may matter more for interpreting failures and recovering from them. Under this hypothesis, strong coding cannot always compensate for a weakness elsewhere. Its weak associations do not imply that coding is unnecessary; all systems use code here.

These associations identify candidate bottlenecks that targeted capability interventions can test directly (Appendix D).

These candidate bottlenecks lead to the second question: how can we quantify physical RSI and its scaling with model capabilities and interaction budget?

## 5 How can we quantify Physical RSI and its scaling law?

How much can an agent improve, when do its main gains occur, and how much interaction does it need?

The five-axis profile turns these questions into measurable outcomes. We first quantify diferences among agents, then study how these outcomes scale with model capabilities and physical interaction budget.

## 5.1 High Scores, Fast Gains and Low Cost Can Diverge

Opus has a higher Success score than GPT-6.1 Sol (0.744 versus 0.703), but lower Eficiency (0.609 versus 0.822). On each system’s solved tasks, Opus saves less of the budget (0.579 versus 0.808), and the gap persists on the ten tasks both systems solve. Higher attained scores can come with later gains and greater physical costs (Table 1 and Figure 5).

Across the six systems, task-mean scores rise from 0.01–0.37 at K0 to 0.10–0.84 by the end of evaluation. They solve 15, 12, 7, 1, 13 and 11 of 18 tasks, respectively: Astra, GPT-6.1 Sol, GPT-6 Sol, GPT-5.6 Sol, Opus and Sonnet. Astra leads on all five axes. The two Claude systems pair relatively high attained scores with middling gain timing, while the GPT Sol family spans a much wider range. GPT-5.6 Sol remains low on Success and Improvement; its Budget score is conditional on one solved task. The profile exposes these trade-ofs without a single ranking.

How much do public benchmark averages reveal about these diferences? Across the four GPT systems, mean scores over 17 public benchmarks span 54.1– 78.1, while the equally weighted physical profile mean,

![](images/dd5d4b8a798b1c303364fea75a6369c95b3440f4e7d97859835799a22a80a75e.jpg)  
Figure 4 Capability associations and the barrel hypothesis. The matrix reports exploratory rank associations across six systems on the 18 tasks. Each barrel represents one RSI axis, with a stave per capability group. Stave heights illustrate a proposed bottleneck interpretation of the associations.

Table 1 Five-axis profiles. K0 scores are averaged over three runs per task. A1–A5 denote First-Attempt Performance, Eficiency, Success, Budget and Improvement. Gemini 4 is evaluated on ten tasks with one K0 run per task.
<table><tr><td>System</td><td>Tasks</td><td>A1</td><td>A2</td><td>A3</td><td>A4</td><td>A5</td></tr><tr><td>GPT-6 Astra</td><td>18</td><td>0.3676</td><td>0.8594</td><td>0.8509</td><td>0.8429</td><td>0.7500</td></tr><tr><td>GPT-6.1 Sol</td><td>18</td><td>0.2843</td><td>0.8224</td><td>0.7028</td><td>0.8081</td><td>0.5714</td></tr><tr><td>GPT-6 Sol</td><td>18</td><td>0.0306</td><td>0.4658</td><td>0.4722</td><td>0.3718</td><td>0.3889</td></tr><tr><td>GPT-5.6 Sol</td><td>18</td><td>0.0120</td><td>0.2247</td><td>0.1037</td><td>0.3013</td><td>0.0556</td></tr><tr><td>Claude Opus 5.5</td><td>18</td><td>0.1769</td><td>0.6091</td><td>0.7444</td><td>0.5794</td><td>0.6875</td></tr><tr><td>Claude Sonnet 5.5</td><td>18</td><td>0.1241</td><td>0.6946</td><td>0.7006</td><td>0.6984</td><td>0.5625</td></tr><tr><td>Gemini 4*</td><td>10</td><td>0.0500</td><td>0.7831</td><td>0.6350</td><td>0.6241</td><td>0.6000</td></tr></table>

100 $\textstyle \times \sum _ { j = 1 } ^ { 5 } A _ { j } / 5 $ , spans 13.9–73.4 (Figure 6). The ranges are 24 and 59.5 points on diferent scales. GPT-5.6 Sol stays near zero cumulative success for 16 rounds: more attempts do not close every capability gap within this budget.

The Physical Barrel Hypothesis ofers one explanation. Useful improvement may require a minimum level of a needed capability, even when other capabilities are strong. The rapid contraction region in Figure 6 illustrates this threshold hypothesis and motivates predicting physical outcomes from capability profiles rather than a public benchmark average.

## 5.2 Observational Scaling across Model Capabilities

Can public capability scores anticipate physical improvement before the full evaluation is run? This is the cross-model question of observational scaling. Beyond correlating capabilities with outcomes, a predictor must place an unseen system using only the relationships observed in other systems.

We ask how far a system improves and when its gains emerge. Let $\bar { q } _ { k } = \operatorname* { m a x } _ { 1 \leq j \leq k } q _ { j }$ be the best mean score reached by round k. For the first question, ranking systems by the equal-weight average of their four capability profiles reproduces the observed order of $\bar { q } _ { 1 7 }$ , the best score reached by K17. For the second, we define the half-gain round $T _ { 5 0 }$ as the first round at which $\bar { q } _ { k }$ reaches half of its gain from K1 to K17. For every pair of systems, a kernel regressor fitted on the capability profiles of the remaining systems predicts which of the two reaches $T _ { 5 0 }$ first, and all pairwise orderings are recovered (Figure 7). Across the six systems, public capability profiles carry information about both the extent and the timing of physical improvement.

![](images/e8ff44749e4dc3c039bf5f23a38e0efc65a2b66576a98c0dbb1d8986a93c44e2.jpg)  
Figure 5 Five-axis profiles. Left: seven overlaid profiles. Right: one panel per system. Axes clockwise from the top are First-Attempt Performance (A1), Budget (A4), Success (A3), Eficiency (A2) and Improvement (A5), on a shared 0–1 scale. The six reference systems use all 18 tasks, as in Table 1; <sup>∗</sup>Gemini 4 uses its ten evaluated tasks (Section 3.3).

![](images/be979d02c08875bfffa53b544f61e7823182eefbddd72190ce169a69a79d4a2b.jpg)

![](images/7cbcdb082268fb572096e82be570261d428f1e1d7deb45f4e035e4d94535f491.jpg)  
Figure 6 Public capability and physical RSI separate differently. Left: cumulative success during refinement on 18 tasks; the shaded region marks the gap between GPT-6.1 Sol and GPT-6 Sol. Right: mean public score over 17 selected benchmarks versus $1 0 0 \times { \left( A _ { 1 } + A _ { 2 } + A _ { 3 } + A _ { 4 } + A _ { 5 } \right) } / 5$ . The physical summary is an arithmetic mean, not radar area.

## 5.3 What Can Early Physical Feedback Predict?

Once physical interaction begins, can early progress predict further gains? The six systems reach their high-score windows at diferent times. The first round reaching 90% of peak gain, τ<sub>90</sub>, is K9 for Astra, K7 for GPT-6.1 Sol, K15 for Opus, K13 for Sonnet, and K17 for GPT-6 Sol and GPT-5.6 Sol. Forecasting later rounds tests whether a short trajectory contains useful information about continued improvement.

A related model may share the shape of its improvement curve. We fit a Weibull growth template g<sub>A</sub> to Astra’s K1–K17 mean score and forecast GPT-6.1 Sol as $\hat { q } ( k ) = u + \left( v - u \right) g _ { A } ( \rho ( k - 1 ) )$ . The start $u ,$ plateau v and relative speed $\rho$ are calibrated on Sol’s first $K _ { i }$ rounds only, with gain and speed regularized toward the template (Figure 8). With K1–K5 observed, the transferred curve forecasts the remaining rounds through K17 with mean absolute error 0.019, compared with 0.112 for holding the last observed score. A reference trajectory can predict a related model’s later performance from fewer rounds.

![](images/8e66dbd8d36cfe4d643359e4614919e502871a64341d1addeff2dacbb9e4f658.jpg)  
Figure 7 Predicting the timing of gains from public capability profiles. Left: four capability profiles of each system, computed from 17 public benchmarks as normalized ranks among the six systems. Right: best mean score $\bar { q } _ { k }$ on the 18 tasks. Circles mark $T _ { 5 0 }$ , the first round at which $\bar { q } _ { k }$ reaches half of its K1–K17 gain. Labels give predicted / observed ranks: $T _ { 5 0 }$ beside the circles and best score reached by K17 on the right. Each pair of systems is predicted from the other four

These analyses address the amount and timing of improvement. We next ask what the gains represent: adaptation to the refinement setting, or experience that remains useful elsewhere?

## 6 Does Physical RSI reflect overfitting or generalization?

With fixed model weights, generalization concerns the code, tools and notes the agent retains. Do they encode reusable procedures, or do they become specialized to the refinement setting? We first test whether retained experience helps beyond repeated attempts. We then evaluate frozen workspaces on unseen layouts and transfer them to new tasks.

## 6.1 Does Retained Experience Help Beyond Repeated Attempts?

The Astra ablation varies which information from earlier rounds reaches the agent: the workspace, grader feedback, images and action records (Table 7). It covers six tasks with two chains each; layouts change every round over K1–K6, and execution continues after success. The full loop keeps all four channels, whereas the no-history condition keeps none, so its six rounds are independent attempts. Late-round success over K4–K6 is 15/36 (41.7%) for the full loop and 3/36 (8.3%) without history (Table 8). This supports a role for accumulated information under this protocol.

The remaining conditions withhold individual channels. Withholding grader feedback causes the largest drop, to 5/36 (13.9%), and withholding earlier-round images yields 12/36 (33.3%). Keeping only the workspace and grader feedback yields 17/36 (47.2%), so past images and action records add little once these two channels are kept. The no-history control supports a role for retained experience, but leaves open whether that experience is specific to the refinement setting.

## 6.2 What Survives When Revision Stops?

To test whether the stored artifacts work beyond practice, we freeze Astra workspaces after K6 and test each on five unseen layouts, with no further revision or outcome feedback (Table 9). Across 12 sources, workspace files alone succeed in 25/60 tests (41.7%). Adding prior raw records gives 23/60 (38.3%). The stored artifacts remain useful under layout changes, while these point estimates show no benefit from exposing the raw history at test time.

Fixed-layout source workspaces reach 27/60 (45.0%) on held-out layouts, compared with the reported

![](images/e6c8e0c70b61c698751022b46480f83bd1cc604fb087b7b570768070a5ac923a.jpg)

![](images/fce36390d01b2c2b596b9b2f34bb7506de9a344b124a6e6b97bcede7856e1241.jpg)  
Figure 8 Forecasting GPT-6.1 Sol from Astra’s improvement shape. Left: Astra’s mean score and the Weibull template fitted to it. Right: GPT-6.1 Sol with K1–K5 available. The template is calibrated on these rounds and forecasts the hidden rounds through K17; shading shows the 90% paired-task bootstrap band. All 18 tasks are used.

![](images/082bc3423b05143e72ca4164b83e08211cbd48371ab45ec90d4be59d5574c904.jpg)  
Figure 9 Cross-task workspace transfer. Top: original self-improvement trajectories for tasks first solved at K6 or later. Bottom: trajectories initialized from a workspace pooled from all other tasks, excluding the target task. First success advances from K9 to K1 for Sonnet 5.5 on Cover Blocks, from K14 to K7 for Opus 5.5 on Organize Table, and from K15 to K8 for Opus 5.5 on Deposit Coin. These examples illustrate faster adaptation with transferred workspaces. Red crosses indicate unsuccessful attempts, dark checks mark first success, and pale checks carry success forward without additional evaluation.

14/16 (88%) on their own layout. The diferent denominators prevent a matched paired interpretation. Still, the comparison suggests that scene-specific gains and transferable experience can coexist. These tests establish some cross-scene usefulness for Astra on this task subset; they do not establish generalization across all systems.

## 6.3 Can Experience Help on a Different Task?

Traces from five models suggest two forms of reusable experience: executable routines and textual operating knowledge. Sonnet 5.5 and Opus 5.5 increasingly encode control in self-written tools, which generate over 96% of their actions in later rounds. GPT-6

![](images/d70602f14909a0b5390f9a08827e5b47f9b7769c8cd77980759def707de7dffa.jpg)  
Figure 10 Tool Libraries and Reuse across Models. (a) Total tool counts and the proportions of perception and geometry tools versus action tools. (b) Fraction of tool lineages reused in subsequent rounds, among those with later rounds available. Reuse includes direct use, modification and derivation of existing tools. (c) Reuse breadth, defined as one minus the share of subsequent calls accounted for by the top 10% of reused lineages. Higher values indicate reuse distributed across more lineages. Astra is omitted because it creates very few tools, used only for verification. GPT-6.1 Sol’s estimates reflect a small library of seven tools.

Sol also builds a substantial tool library, but tools generate 67% of actions while the model generates 33% directly. These shares measure tool reliance.

These tool libraries grow across rounds, combining tools created from scratch with those derived from earlier implementations. We use tool lineage to identify reuse across rounds. Direct use, modification and derivation of an existing tool all count as reuse. A directed relation A → B indicates that tool B is derived from tool A. Tracing these relations across rounds reveals how later tools build on earlier implementations. Among the three models with substantial tool libraries, Opus exhibits broader reuse and lower concentration. Its lineage reuse rate is 0.347, compared with 0.167 for Sonnet and 0.086 for GPT-6 Sol. The top 10% of reused lineages account for 48% of subsequent calls in Opus, compared with 76% in Sonnet and 83% in GPT-6 Sol. Together, these patterns suggest that Opus accumulates a broader reusable toolbox, while reuse in Sonnet and GPT-6 Sol remains concentrated in a smaller subset of tools (Figure 10). Astra and GPT-6.1 Sol follow a diferent pattern. Approximately 99% of their actions remain directly generated by the model. They use only a very small number of tools for verification, but their reasoning time decreases across rounds. File-access traces show repeated use of memory, skills and policy notes to recover successful actions and subtask solutions, diagnose failures and guide revisions. These observations suggest that both executable routines and retained operating knowledge can reduce repeated reasoning.

We further find that the workspace and generalization analyses align with the Physical RSI Barrel Hypothesis across its four capability factors: spatial reasoning, perception, coding and harness operation. Spatial tools account for 14.5% of Sonnet’s tool calls and 13.7% of Opus’s, with 408 of 2,819 calls and 183 of 1,332 calls, respectively. Astra makes only seven tool calls, including one to such a tool, or 14%. Astra’s minimal reliance on these modules suggests stronger intrinsic perceptual and spatial capabilities. Sonnet and Opus repeatedly reuse these modules across tasks, suggesting that they rely more on tools to compensate for limitations in these capabilities.

During cross-task transfer, agents also consult memory and skills from semantically similar tasks. In Organize Table, Opus combines a parameterized move tool from Pack Objects Into Box with a memory from Store Tools In Toolbox: “the PITCHED 30 deg grasp... reaches the +y half...” The tool supplies an operation, while the memory identifies a grasp orientation that extends its reach. This shows how a reusable procedure and knowledge of when it works can jointly support transfer (Figure 9).

The transfer traces suggest that performance gains depend on experience relevant to the target task’s bottleneck and the agent’s ability to assess its applicability from failure evidence. Semantic similarity helps identify candidate experience; failure evidence guides whether to reuse, adapt or discard it. The teaser illustrates this mechanism through the reuse of a closed-loop grasp-and-transfer routine from stacking blocks to covering blocks with a cup, and an insertion routine from key insertion to charger plugging.

## 7 Limitations

The main comparison covers six model-plusconfiguration systems, with one improvement chain per task. K0 repeats do not estimate variability across complete chains. First-success stopping records one success; reliable repeated execution needs a separate test. Budget and Improvement use conditional task sets that difer across systems. Astra’s outcomes difer between the main evaluation and the experiment without supplementary hints.

The capability analysis relies on public benchmark scores whose evaluation settings difer from our runs, and it covers six systems without independent validation. Capability bottlenecks and gain-timing predictions therefore remain hypotheses to test. The ablation and frozen tests cover Astra and six tasks, while cross-task results are three first-run cases. These limits constrain the scope of the observed transfer and the mechanisms that can be inferred from it.

## 8 Conclusion

PhysRSI-eval studies the origins, measurement, scaling and generalization of physical RSI under a fixed interaction budget. Capability associations motivate the Physical Barrel Hypothesis: the strengths needed to start well may difer from those needed to improve. The five-axis metric makes these outcomes explicit and shows that high scores, early gains and eficient interaction need not coincide. Cross-model prediction and early-round forecasting probe how these outcomes vary with capabilities and interaction budget. Workspace interventions support a role for retained experience and show that it can remain useful on new layouts and tasks. Physical RSI can thus involve both scene-specific adaptation and reusable improvement. Understanding it requires measuring how agents improve and examining what they retain.

## References

[1] Jun Shern Chan, Neil Chowdhury, Oliver Jafe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, and Aleksander Mądry. MLEbench: Evaluating machine learning agents on machine learning engineering, 2024. https://arxiv. org/abs/2410.07095.

[2] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. SpatialVLM: Endowing visionlanguage models with spatial reasoning capabilities, 2024. https://arxiv.org/abs/2401.12168.

[3] Tianxing Chen, Yue Chen, Zixuan Li, Junyuan Tang, Kailun Su, Haoran Lu, Weijie Wan, Baijun Chen, Songling Liu, Haowen Yan, Honghao Su, Zhiyang Dou, Kaixuan Wang, Dandan Zhang, Yunze Liu, Yan Qin, Qiwei Liang, Qiwei Wu, Zijian Lin, Wenwei Lin, Yuran Wang, Minghua He, Tianshu Wu, Ruihai Wu, Jingquan Zhou, Kai-Chong Lei, Haibao Yu, Yuanfeng Ji, Weiyang Jin, Guanyu Lin, Xiaofan Li, Qi Xiong, Renjing Xu, Zhongyu Li, Wenhao Chai, Enze Xie, Ziwei Wang, Yao Mu, Hao Dong, Wojciech Matusik, Mingyu Ding, Wenbo Ding, Ping Luo, and Masayoshi Tomizuka. RoboDojo: A unified simand-real benchmark for comprehensive evaluation of generalist robot manipulation policies, 2026. https: //arxiv.org/abs/2607.04434.

[4] Yizhe Chi, Wenyi Li, Deyao Hong, Xiaoqiu Wang, Mingju Gao, Kaisen Yang, Bingxiang He, Youjie Zheng, Calvin Xiao, and Qinhuai Na. AI4AI-Bench: Benchmarking LLM agents in algorithmic design for recursive self-improvement, 2026. https://arxiv. org/abs/2608.20318.

[5] Zijie Diao, Yitong Chen, Sicheng Xie, Tianyi Lu, Wujian Peng, Guojin Zhong, Houze Xu, Ziyi Ye, Zuxuan Wu, and Yu-Gang Jiang. LIBERO-Agent: Evaluating general-purpose agents for direct embodied manipulation, 2026. https://arxiv.org/abs/ 2609.39507.

[6] Mengfei Du, Binhao Wu, Zejun Li, Xuanjing Huang, and Zhongyu Wei. EmbSpatial-Bench: Benchmarking spatial understanding for embodied tasks with large vision-language models, 2024. https: //arxiv.org/abs/2406.05756.

[7] Letian Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, Wenli Xiao, Guanzhi Wang, Dantong Niu, Fei-Fei Li, Guanya Shi, Jiajun Wu, Shankar Sastry, Yuke Zhu, Ken Goldberg, and Linxi “Jim” Fan. CaP-X: A framework for benchmarking and improving coding agents for robot manipulation, 2026. https://arxiv.org/abs/2603. 22435.

[8] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: Multimodal large language models can see but not perceive, 2024. https://arxiv.org/abs/2404.12390.

[9] Galbot Team, Xuchuan Chen, Xiaoqian Cheng, Yu Deng, Lihe Ding, Shaocong Dong, Xiangjun Gao, Haozhe Jia, Zekai Li, Zhoujian Li, Yunrui Lian, Sikai Liang, Chenghuai Lin, Dairu Liu, Jiahang Liu, Qingtao Liu, Yuxuan Ma, Zekun Qi, Jiayi Su, He Wang, Ruochen Xu, Tianyu Xu, Xudong Xu, Zhe Xu, Mi Yan, et al. Systematically exploring the capabilities of GPT-6 Astra as embodied policies, 2026. https://arxiv.org/abs/2609.38537.

[10] Xingze Gao, Chuanrui Hu, Hongda Chen, Pengfei Yao, Zhao Wang, Yi Bai, Zhengwei Wu, Yunyun Han,

Xiaofeng Cong, Jie Gui, Yafeng Deng, and Teng Li. EvoAgentBench: Benchmarking agent self-evolution via ability transfer, 2026. https://arxiv.org/abs/ 2607.05202.

[11] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, Steven Bohez, Konstantinos Bousmalis, Anthony Brohan, Thomas Buschmann, Arunkumar Byravan, Serkan Cabi, Ken Caluwaerts, Federico Casarini, Oscar Chang, Jose Enrique Chen, Xi Chen, Hao-Tien Lewis Chiang, Krzysztof Choromanski, David D’Ambrosio, Sudeep Dasari, et al. Gemini Robotics: Bringing AI into the physical world, 2025. https://arxiv.org/abs/2503.20020.

[12] Jordan Hofmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training computeoptimal large language models, 2022. https:// arxiv.org/abs/2203.15556.

[13] Lisheng Huang, Chen Yang, Hao Zhou, Huatong Song, Zongchao Chen, Ran Le, Yang Song, Wayne Xin Zhao, and Tao Zhang. Evo-Bench: Can language models improve agent harness?, 2026. https://arxiv.org/abs/2608.09096.

[14] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Noah Brown, Tomas Jackson, Linda Luu, Sergey Levine, Karol Hausman, and Brian Ichter. Inner Monologue: Embodied reasoning through planning with language models, 2022. https://arxiv. org/abs/2207.05608.

[15] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Live-CodeBench: Holistic and contamination free evaluation of large language models for code, 2024. https://arxiv.org/abs/2403.07974.

[16] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. OmniSpatial: Towards comprehensive spatial reasoning benchmark for vision language models, 2025. https://arxiv.org/abs/2506.03135.

[17] Sihang Jiang, Lipeng Ma, Zhonghua Hong, Keyi Wang, Zhiyu Lu, Tengfei Wang, Shisong Chen, Jinghao Zhang, Tianjun Pan, Weijia Li, Jiaqing Liang, and Yanghua Xiao. SEA-Eval: A benchmark for evaluating self-evolving agents beyond episodic assessment, 2026. https://arxiv.org/abs/2604.08988.

[18] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues?, 2023. https:// arxiv.org/abs/2310.06770.

[19] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jefrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. https: //arxiv.org/abs/2001.08361.

[20] Zixuan Ke, Vaidehi Patil, Haizhou Shi, Yang Li, Ye Liu, Sarath Shekkizhar, Anurag Koul, Jiayu Wang, Xuan Phi Nguyen, Semih Yavuz, Mohit Bansal, and Shafiq Joty. EvoHarnessBench: Can your agents keep pace with an evolving harness?, 2026. https://arxiv.org/abs/2609.04280.

[21] Manling Li, Shiyu Zhao, Qineng Wang, Kangrui Wang, Yu Zhou, Sanjana Srivastava, Cem Gokmen, Tony Lee, Li Erran Li, Ruohan Zhang, Weiyu Liu, Percy Liang, Li Fei-Fei, Jiayuan Mao, and Jiajun Wu. Embodied agent interface: Benchmarking LLMs for embodied decision making, 2024. https://arxiv. org/abs/2410.07166.

[22] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control, 2022. https://arxiv. org/abs/2209.07753.

[23] Minhua Lin, Juncheng Wu, Zijun Wang, Zhan Shi, Yisi Sang, Bing He, Zewen Liu, Tianxin Wei, Zongyu Wu, Zhiwei Zhang, Dakuo Wang, Xiang Zhang, Benoit Dumoulin, Cihang Xie, Yuyin Zhou, Suhang Wang, and Hanqing Lu. Harness updating is not harness benefit: Disentangling evolution capabilities in self-evolving LLM agents, 2026. https://arxiv.org/abs/2605.30621.

[24] Yiheng Lyu, Xueying Jiang, Wenhao Li, Shijian Lu, and Gongjie Zhang. CodeActionBench: Evaluating agentic code-as-policy for embodied manipulation, 2026. https://arxiv.org/abs/2609.33807.

[25] Haitong Ma, Chenxiao Gao, Rushi Qiang, Bo Dai, and Na Li. RLE-Bench: A qualifying exam for coding agents as robot learning engineers, 2026. https: //arxiv.org/abs/2609.34210.

[26] Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-level reward design via coding large language models, 2023. https://arxiv.org/abs/ 2310.12931.

[27] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir

Yazdanbakhsh, and Peter Clark. Self-Refine: Iterative refinement with self-feedback, 2023. https: //arxiv.org/abs/2303.17651.

[28] Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E. Kelly Buchanan, Junhong Shen, Guanghao Ye, Haowei Lin, Jason Poulos, Maoyu Wang, Marianna Nezhurina, Jenia Jitsev, Di Lu, Orfeas Menis Mastromichalakis, Zhiwei Xu, Zizhao Chen, Yue Liu, Robert Zhang, Leon Liangyu Chen, Anurag Kashyap, et al. Terminal-Bench: Benchmarking agents on hard, realistic tasks in command line interfaces, 2026. https://arxiv.org/abs/2601.11868.

[29] Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace Kim, Michele Wang, Olivia Watkins, Simón Posada Fishman, Marwan Aljubeh, Phoebe Thacker, Laurance Fauconnet, Natalie S. Kim, Patrick Chao, Samuel Miserendino, Gildas Chabot, David Li, Michael Sharman, Alexandra Barr, Amelia Glaese, and Jerry Tworek. GDPval: Evaluating AI model performance on real-world economically valuable tasks, 2025. https://arxiv.org/abs/2510. 04374.

[30] Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, Michael Choi, Anish Agrawal, Arnav Chopra, Adam Khoja, Ryan Kim, Richard Ren, Jason Hausenloy, Oliver Zhang, Mantas Mazeika, Dmitry Dodonov, Tung Nguyen, Jaeho Lee, Daron Anderson, Mikhail Doroshenko, Alun Cennyth Stokes, et al. Humanity’s last exam, 2025. https://arxiv.org/abs/ 2501.14249.

[31] Zekun Qi, Wenyao Zhang, Yufei Ding, Runpei Dong, Xinqiang Yu, Jingwen Li, Lingyun Xu, Baoyu Li, Xialin He, Guofan Fan, Jiazhao Zhang, Jiawei He, Jiayuan Gu, Xin Jin, Kaisheng Ma, Zhizheng Zhang, He Wang, and Li Yi. SoFar: Language-grounded orientation bridges spatial reasoning and object manipulation, 2025. https://arxiv.org/abs/2502.13143.

[32] Ben Rank, Hardik Bhatnagar, Ameya Prabhu, Shira Eisenberg, Karina Nguyen, Matthias Bethge, and Maksym Andriushchenko. PostTrainBench: Can LLM agents automate LLM post-training?, 2026. https://arxiv.org/abs/2603.08640.

[33] Roboflow. Roboflow Vision Evals, 2026. https:// playground.roboflow.com/evals. Accessed: 2026- 10-17.

[34] Yangjun Ruan, Chris J. Maddison, and Tatsunori Hashimoto. Observational scaling laws and the predictability of language model performance, 2024. https://arxiv.org/abs/2405.10938.

[35] Zeyu Shen, Haoxiang You, Yilang Liu, Zhicheng Zheng, Lihan Zha, Kashu Yamazaki, Mingtong

Zhang, Suning Huang, Jiankai Sun, Qianzhong Chen, Lucy He, Kaiyuan Liu, Haoran Chang, Katerina Fragkiadaki, Dhruv Shah, Mac Schwager, Peter Henderson, Ian Abraham, and Canwen Xu. EmbodiedSWE: Coding agents for long horizon dexterous robotics, 2026. https://arxiv.org/abs/2609. 27308.

[36] Jiajun Shi, Siyuan Tao, Yuhao Wu, Zexuan Wang, Jingyuan Zhang, Jiaheng Liu, Xinping Lei, Xinrong Zhang, Siyuan Fang, Zhewen Tan, Tianle Cai, Junhao Fang, Jiameng Huang, Yueyang Wang, Jinkai Liu, Yuxuan Zhang, Jian Yang, Zhoujun Li, Shen Yan, Wenhao Huang, and Ge Zhang. S3Gym: Can LLMs turn self-testing and self-judging into self-improvement?, 2026. https://arxiv.org/abs/ 2608.31100.

[37] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. https://arxiv.org/abs/ 2303.11366.

[38] Harman Singh, Anton Bakhtin, Rulin Shao, Gabriel Synnaeve, Ilia Kulikov, Rob Fergus, Sanjeev Arora, Kurt Keutzer, Jason Weston, Anuj Mahajan, and Anirudh Goyal. Agent plasticity: Measuring selfimprovement through experience, 2026. https:// arxiv.org/abs/2610.08902.

[39] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. ProgPrompt: Generating situated robot task plans using large language models, 2022. https://arxiv.org/abs/ 2209.11302.

[40] Giulio Starace, Oliver Jafe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, and Tejal Patwardhan. PaperBench: Evaluating AI’s ability to replicate AI research, 2025. https://arxiv.org/ abs/2504.01848.

[41] Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, and Zhibo Chen. VLA-JEPA: Enhancing visionlanguage-action model with latent world model, 2026. https://arxiv.org/abs/2602.10098.

[42] Minyang Tian, Luyu Gao, Shizhuo Dylan Zhang, Xinan Chen, Cunwei Fan, Xuefei Guo, Roland Haas, Pan Ji, Kittithat Krongchon, Yao Li, Shengyan Liu, Di Luo, Yutao Ma, Hao Tong, Kha Trinh, Chenyu Tian, Zihan Wang, Bohao Wu, Yanyu Xiong, Shengzhu Yin, Minhui Zhu, Kilian Lieret, Yanxin Lu, Genglin Liu, Yufeng Du, et al. SciCode: A research coding benchmark curated by scientists, 2024. https://arxiv.org/abs/2407.13168.

[43] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Man-

dlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models, 2023. https://arxiv.org/abs/2305.16291.

[44] Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey, Shubh-Agrawal, Sandeep Singh Sandha, Siddartha Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, and Micah Goldblum. LiveBench: A challenging, contamination-limited LLM benchmark, 2024. https://arxiv.org/abs/2406.19314.

[45] Hjalmar Wijk, Tao Lin, Joel Becker, Sami Jawhar, Neev Parikh, Thomas Broadley, Lawrence Chan, Michael Chen, Josh Clymer, Jai Dhyani, Elena Ericheva, Katharyn Garcia, Brian Goodrich, Nikola Jurkovic, Holden Karnofsky, Megan Kinniment, Aron Lajko, Seraphina Nix, Lucas Sato, William Saunders, Maksym Taran, Ben West, and Elizabeth Barnes. RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts, 2024. https://arxiv.org/abs/2411.15114.

[46] Yuhao Wu, Jingyuan Zhang, Jiajun Shi, Xinping Lei, Qingshui Gu, Yuxuan Zhang, Zexuan Wang, Chen He, Chen Huang, Maojia Song, Zhiyuan Zeng, Shaowen Wang, Jinkai Liu, Yunfeng Shi, Jiaheng Liu, Shen Yan, Wenhao Huang, Ge Zhang, and Wenxuan Zhang. HarnessDev: Can LLMs create and evolve their own agent harness?, 2026. https://arxiv. org/abs/2609.01437.

[47] Yuhao Wu, Jingyuan Zhang, Jiajun Shi, Yuxuan Zhang, Xinping Lei, Junting Zhou, Zexuan Wang, Yuchen Wu, Huan Zhou, Duo Wang, Yinzhu Piao, Yongchang Peng, Yunfeng Shi, Jin Chen, Zuo Wang, Jinkai Liu, Jiaheng Liu, Wenxuan Zhang, Shen Yan, Wenhao Huang, and Ge Zhang. Aspire: Can models self-evolve from vague goals?, 2026. https://arxiv. org/abs/2608.31111.

[48] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. OS-World: Benchmarking multimodal agents for openended tasks in real computer environments, 2024. https://arxiv.org/abs/2404.07972.

[49] Shuhan Xue, Zixin Ding, Yichen Shen, Yinjie Wang, Zhenfei Yin, Yingcheng Wu, Yuxin Chen, Mengdi Wang, and Ling Yang. PAST-Bench: Benchmarking the foundations of recursive self-improvement in personal agents, 2026. https://arxiv.org/abs/2608. 04003.

[50] Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in Space: How multimodal large language mod-

els see, remember, and recall spaces, 2024. https: //arxiv.org/abs/2412.14171.

[51] Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, Heng Ji, Huan Zhang, and Tong Zhang. EmbodiedBench: Comprehensive benchmarking multimodal large language models for vision-driven embodied agents, 2025. https://arxiv.org/abs/2502. 09560.

[52] Zhiqin Yang, Chenxin Li, Xiaomeng Hu, Yibin Liu, Weidong Huang, Jiankai Sun, Haitao Li, Zijian Wu, Yuzhi Huang, Fanding Huang, Hanwen Sun, Jiashun Liu, Jingqi Tong, Mingxin Huang, Shaoli Hu, Shijue Huang, Tianyi Bai, Xinyuan Wang, Yunlong Lin, Zhengyang Tang, Zhexin Zhang, Zhuo Chen, Xierui Song, Juntao Dai, Boyuan Chen, et al. Robot-World: Benchmarking multimodal agents for robot use across diverse tasks and embodiments, 2026. https://arxiv.org/abs/2610.10409.

[53] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agent-user interaction in real-world domains, 2024. https://arxiv.org/abs/2406.12045.

[54] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI, 2023. https://arxiv.org/abs/2311.16502.

[55] Boyang Zhang, Sebastián G. Acosta, Preston Carlson, Sacha Bron, Pierre-Loïc Doulcet, Daniel B. Ospina, and Simon Suo. ParseBench: A document parsing benchmark for AI agents, 2026. https: //arxiv.org/abs/2604.08538.

[56] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, Fan Lu, He Wang, Zhizheng Zhang, Li Yi, Wenjun Zeng, and Xin Jin. DreamVLA: A visionlanguage-action model dreamed with comprehensive world knowledge, 2025. https://arxiv.org/abs/ 2507.04447.

[57] Zhongbo Zhang, Jiayi Jin, Yifan Wang, Zaibin Zhang, Haiwen Diao, Lijun Wang, and Huchuan Lu. VABench: Measuring embodied spatial intelligence through visual demonstrations, active perception, and metric control, 2026. https://arxiv.org/abs/ 2609.19554.

[58] Congjie Zheng, Chuanyi Xue, Bin Liang, Jun Yang, and Changshui Zhang. SEAGym: An evaluation environment for self-evolving LLM agents, 2026. https://arxiv.org/abs/2606.17546.

## Appendix

## A Task and Protocol Details

The 18 tasks keep their native grading and taskspecific step limits (Table 2). The main fixed-layout protocol and the changing-layout ablation serve different comparisons and are reported separately.

Actions and observations. An action sets a target pose for each end-efector, within 5 cm of translation and 0.35 rad of rotation of its current pose per call, executed for 1–5 control steps at 25 Hz; each call returns a state JSON and three camera images. Per round, the reasoning budget is 4200 s and the revision window after grading is 1800 s (GPT systems) or 600 s (Claude systems). The physical budget per task is 17 $L _ { t }$ , that is, 17 full-step-limit attempts.

## B Metric Definitions and Computation

Metrics are computed after averaging the task curves; in particular, Eficiency is a ratio of aggregate gains, not an average of task-wise ratios. The A3 threshold is the first integer Harness round reaching 90% of the observed peak gain, without interpolation. Budget charges each extra attempt once, supplements only failed reasoning-timeout rounds to their full step quota, and clips task savings at zero. Undefined denominators stay undefined. The summary in Figure 6 is the equally weighted arithmetic mean of A1–A5, multiplied by 100. It is neither radar polygon area nor a success probability.

## C Detailed Experimental Results

The five-axis profiles use the six-system evaluation on 18 tasks; Table 4 lists the intermediate quantities used to compute them. Astra’s scores, curves and costs come from the main evaluation, in which it solves 15 tasks. The separate experiment without supplementary hints solves 13 tasks and is excluded from these aggregates. Table 5 gives the first successful Harness round on each task.

## D Capability and Prediction Analyses

The capability profiles use 17 public benchmark scores collected on 15 October 2026 from Roboflow Vision Evals [33], ParseBench [55], LiveBench [44] and five Terminal-Bench 4.0 tasks [28] (Table 6). Each score is converted to a normalized rank among the systems, and ranks are averaged within the Spatial, Harness, Perception and Coding groups. The capabilityassociation matrix compares these group profiles with A1–A5 on the 18 tasks by Spearman correlation, using exact permutation tests over the six systems with Benjamini–Hochberg adjustment; the smallest adjusted q is 0.11. The public-score comparison in Figure 6 averages the 17 raw scores with equal weight. The $T _ { 5 0 }$ analysis predicts the order of gain timing from the group profiles. For each pair of held-out systems, the group profiles and $T _ { 5 0 }$ are standardized on the remaining systems, and kernel ridge regression with $k ( z , z ^ { \prime } ) = \exp ( - \gamma \| z - z ^ { \prime } \| ^ { 2 } ) , \gamma = 0 . 0 6 2 5$ and α = 10 predicts $T _ { 5 0 } ;$ the system with the smaller prediction is ranked earlier.

Table 2 The 18 PhysRSI-eval tasks. Step limits are per round (25 Hz). Graded tasks score the highest tier reached (tiers in percent of the native score); binary tasks score 0 or 1. The success check is independent of the tiers.
<table><tr><td>#</td><td>Task</td><td>Steps Scoring (tiers, %)</td><td></td></tr><tr><td>1</td><td>arrange_largest_number</td><td></td><td>1050 graded [5,15,30,100] or [5,15,25,40,100]</td></tr><tr><td>2</td><td>classify_objects_by_- language</td><td></td><td>1100 graded [10,40,100]</td></tr><tr><td>3</td><td>cover_blocks</td><td>800 graded</td><td>[0,0,5,15,30,100]</td></tr><tr><td>4</td><td>deposit_coin</td><td></td><td>300 graded [20,100]</td></tr><tr><td>5</td><td>fill_pen_holder</td><td></td><td>1100 graded [10,25,40,90]</td></tr><tr><td>6</td><td>insert_key</td><td></td><td>300 graded [15,100]</td></tr><tr><td>7</td><td>match_and_pick_from_- conveyor</td><td>700 binary</td><td></td></tr><tr><td>8</td><td>organize_table</td><td></td><td>1000 graded [25,50,75,100]</td></tr><tr><td>9</td><td>pack_objects_into_box</td><td></td><td>1300 graded [10,25,50,100]</td></tr><tr><td></td><td>10 play_Xylophone</td><td>500 binary</td><td></td></tr><tr><td></td><td>11 plug_in_charger</td><td>400 binary</td><td></td></tr><tr><td></td><td>12 pour_by_language</td><td></td><td>800 graded [20,50,100]</td></tr><tr><td></td><td>13 pour_liquid_into_cup</td><td>400 binary</td><td></td></tr><tr><td>14 push_T</td><td></td><td>600 binary</td><td></td></tr><tr><td></td><td>15 put_bottles_into_dustbin</td><td></td><td>700 graded [10,25,40,100]</td></tr><tr><td></td><td>16 stack_blocks_by_language</td><td></td><td>400 graded [20,100]</td></tr><tr><td></td><td>17 store_tools_in_toolbox</td><td></td><td>900 graded [25,50,75,100]</td></tr><tr><td>18 swap_blocks</td><td></td><td>700 binary</td><td></td></tr></table>

The within-trajectory forecasts use early physical scores rather than public benchmarks. The template $g _ { A } ( t ) = 1 - \exp [ - ( t / \tau _ { A } ) ^ { p _ { A } } ]$ , with $t = k - 1$ , is fitted to Astra’s K1–K17 mean score and then fixed $( \tau _ { A } =$ 2.18, $p _ { A } ~ = ~ 1 . 3 0 )$ For GPT-6.1 Sol, u, v and $\rho$ minimize the squared error on the available rounds plus $\lambda [ ( v - u ) - \overline { { \Delta } } _ { A } ] ^ { 2 } + ( 0 . 0 3 \log \rho / 0 . 2 ) ^ { 2 }$ , where $\Delta _ { A }$ is the gain of the Astra template, $\lambda = 4 , 0 \leq u \leq v \leq 1$ and $\rho$ is searched over 81 log-spaced values in [0.3, 2]. The remaining rounds are forecast without updating on their outcomes. The band in Figure 8 spans the 5th–95th percentiles of 300 paired task resamples, refitting the template and repeating the calibration in each resample. All 18 tasks are used.

Table 3 System configurations and evaluation coverage. Each system comprises a model and its runtime configuration, so cross-system comparisons are descriptive. <sup>a</sup>Settings from session logs; other settings come from configuration files.
<table><tr><td>Item</td><td>GPT-6 Astra</td><td>GPT-6.1 Sol</td><td>GPT-6 Sol</td><td>GPT-5.6 Sol</td><td>Opus 5.5</td><td>Sonnet 5.5</td></tr><tr><td>Client</td><td>Codex CLI 0.156.1</td><td>Codex CLIa</td><td>Codex CLI 0.156.1</td><td>Codex CLI 0.156.1</td><td>Claude Codeª</td><td>Claude Codea</td></tr><tr><td>Reasoning effort</td><td>xhigh</td><td> $\mathrm { \ x h i g h ^ { a } }$ </td><td>xhigh</td><td>xhigh</td><td>higha</td><td>xhigh</td></tr><tr><td>Reasoning budget per round</td><td>4200s</td><td>4200sa</td><td>4200s</td><td>4200s</td><td> $4 2 0 0 { \mathrm { s } } ^ { \mathrm { a } }$ </td><td>4200s</td></tr><tr><td>Revision window</td><td>1800s</td><td>1800 sa</td><td>1800s</td><td>1800s</td><td>600sa</td><td>600s</td></tr><tr><td>Selected K1-K17 entries</td><td>91</td><td>143</td><td>255</td><td>306</td><td>175</td><td>175</td></tr><tr><td>Counted extra attempts</td><td>13</td><td>9</td><td>35</td><td>12</td><td>21</td><td>40</td></tr><tr><td>Tasks solved by K17 15</td><td></td><td>12</td><td>7</td><td>1</td><td>13</td><td>11</td></tr></table>

Table 4 Curve summaries for A1–A5 (18 tasks). q<sub>0</sub> and $q _ { 1 7 }$ are the task-mean scores at K0 and K17, $q _ { \mathrm { m a x } }$ the peak, τ<sub>90</sub> the first round reaching 90% of the peak gain, $C _ { 1 }$ and $C _ { 1 7 }$ the fractions of tasks solved by K1 and K17, and “rec.” the number of tasks failed at K1 and solved later.
<table><tr><td>System</td><td>qo</td><td> $\underline { { { q _ { 1 7 } } } }$ </td><td>qmax</td><td> $\tau _ { 9 0 }$ </td><td> $C _ { 1 }$ </td><td> $C _ { 1 7 }$ </td><td>rec.</td></tr><tr><td>GPT-6 Astra</td><td>0.368</td><td>0.842</td><td>0.869</td><td>9</td><td>0.333</td><td>0.833</td><td>9</td></tr><tr><td>GPT-6.1 Sol</td><td>0.284</td><td>0.689</td><td>0.728</td><td>7</td><td>0.222</td><td>0.667</td><td>8</td></tr><tr><td>GPT-6 Sol</td><td>0.031</td><td>0.472</td><td>0.472</td><td>17</td><td>0.000</td><td>0.389</td><td>7</td></tr><tr><td>GPT-5.6 Sol</td><td>0.012</td><td>0.104</td><td>0.104</td><td>17</td><td>0.000</td><td>0.056</td><td>1</td></tr><tr><td>Opus 5.5</td><td>0.177</td><td>0.744</td><td>0.744</td><td>15</td><td>0.111</td><td>0.722</td><td>11</td></tr><tr><td>Sonnet 5.5</td><td>0.124</td><td>0.700</td><td>0.708</td><td>13</td><td>0.111</td><td>0.611</td><td>9</td></tr></table>

## E Ablation Details

The six tasks are Push T, Plug in Charger, Deposit Coin, Play Xylophone, Pour by Language and Pour Liquid into Cup, with two chains per task and all six rounds run in every chain (no first-success stop). The first chain uses layouts 1–6 and the second layouts 7–12, one per round and in the same order for every condition; the Fixed condition keeps layout 1 or 7 throughout. All rounds are RoboDojo simulation episodes with native grading. Within a round, execution and revision share one conversation, and the next round starts a new conversation. Information from earlier rounds therefore reaches the agent only through the channels in Table 7. Table 8 gives the per-round successes. Fixed has 22/36 late-round successes on its seen layouts and is descriptive rather than a matched unseen-layout comparison.

Table 5 First successful Harness round on each task. A dash denotes no success within K17.
<table><tr><td>Task</td><td>Astra</td><td>6.1 Sol</td><td>6 Sol</td><td>5.6 Sol</td><td>Opus</td><td>Sonnet</td></tr><tr><td>Arrange Largest Number</td><td>1</td><td>2</td><td>17</td><td></td><td>3</td><td>2</td></tr><tr><td>Classify Objects by Language</td><td>1</td><td>1</td><td></td><td>一</td><td>4</td><td>2</td></tr><tr><td>Cover Blocks</td><td>1</td><td>3</td><td>4</td><td>17</td><td>10</td><td>9</td></tr><tr><td>Deposit Coin</td><td>4</td><td></td><td></td><td>一</td><td>15</td><td></td></tr><tr><td>Fill Pen Holder</td><td>3</td><td>8</td><td></td><td>1</td><td>一</td><td></td></tr><tr><td>Insert Key</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Match and Pick from Conveyor</td><td>1</td><td>1</td><td>3</td><td>一</td><td>1</td><td>1</td></tr><tr><td>Organize Table</td><td>2</td><td></td><td></td><td></td><td>14</td><td></td></tr><tr><td>Pack Objects into Box</td><td>9</td><td>6</td><td>一</td><td>一</td><td>15</td><td>一</td></tr><tr><td>Play Xylophone</td><td>4</td><td>4</td><td>14</td><td>一</td><td></td><td>7</td></tr><tr><td>Plug in Charger</td><td>2</td><td></td><td></td><td></td><td>10</td><td>7</td></tr><tr><td>Pour by Language</td><td>5</td><td>7</td><td>8</td><td>一</td><td>4</td><td>13</td></tr><tr><td>Pour Liquid into Cup</td><td>2</td><td>5</td><td>13</td><td>一</td><td>3</td><td>4</td></tr><tr><td>Push T</td><td>3</td><td>1</td><td></td><td></td><td>8</td><td>8</td></tr><tr><td>Put Bottles into Dustbin</td><td>1</td><td>2</td><td></td><td></td><td>1</td><td>1</td></tr><tr><td>Stack Blocks by Language</td><td>1</td><td>1</td><td>9</td><td></td><td>2</td><td>2</td></tr><tr><td>Store Tools in Toolbox</td><td></td><td></td><td></td><td>一</td><td></td><td></td></tr><tr><td>Swap Blocks</td><td></td><td></td><td>一</td><td>一</td><td></td><td></td></tr><tr><td>Solved</td><td>15</td><td>12</td><td>7</td><td>1</td><td>13</td><td>11</td></tr></table>

## F Generalization Details

In the complete frozen-test conditions, each of 12 source chains’ K6-revised workspaces is frozen and run once on each of five held-out layouts, with no revision and no outcome feedback (Table 9). Outcomes that share a source or a task are dependent and must not be treated as 60 independent replications.

Table 6 Public benchmarks behind the capability profiles. Scores were collected on 15 October 2026; each group averages normalized benchmark ranks with equal weight.
<table><tr><td>Group</td><td>Benchmark item</td><td>Source</td></tr><tr><td rowspan="3">Spatial</td><td>Object detection (mAP@50)</td><td>Roboflow</td></tr><tr><td>OCR text localization</td><td>Roboflow</td></tr><tr><td>Visual grounding</td><td>ParseBench</td></tr><tr><td rowspan="4">Harness</td><td>ctr-optimization</td><td>Terminal-Bench 4.0</td></tr><tr><td>shadow-relay</td><td>Terminal-Bench 4.0</td></tr><tr><td>Instruction following</td><td>LiveBench</td></tr><tr><td>Visual reasoning</td><td>Roboflow</td></tr><tr><td rowspan="5">Perception</td><td>OCR transcription</td><td>Roboflow</td></tr><tr><td>OCR single value</td><td>Roboflow</td></tr><tr><td>OCR structured JSON</td><td>Roboflow</td></tr><tr><td>Charts</td><td>ParseBench</td></tr><tr><td>freecad-platform-drawing</td><td>Terminal-Bench 4.0</td></tr><tr><td rowspan="4">Coding</td><td>Code generation</td><td>LiveBench</td></tr><tr><td>Code completion</td><td>LiveBench</td></tr><tr><td>Agentic coding</td><td>LiveBench</td></tr><tr><td>mvcc-lsm-compaction</td><td>Terminal-Bench 4.0</td></tr><tr><td></td><td>payments-pipeline-fix</td><td>Terminal-Bench 4.0</td></tr></table>

Table 7 Information from earlier rounds in each ablation condition. Workspace: committed notes, skills and code. Grader: result, score and failure reason after each round, and earlier-round outcome files; step-level success flags in action replies and logs are not part of this channel. Images and actions: observation images and action logs from earlier rounds. Current-round observations, including camera images, are available in every condition.
<table><tr><td>Name</td><td>Condition</td><td>Workspace Grader Images Actions</td><td></td><td></td><td></td></tr><tr><td>Full</td><td>Full loop</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>No-img</td><td>No earlier-round images</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>No-grader No grader</td><td>feedback</td><td>√</td><td>x</td><td>√</td><td>√</td></tr><tr><td>WS+FB</td><td>Workspace and grader only</td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>No-hist</td><td>No history</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Fixed</td><td>Full loop, fixed layout</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 9 Frozen held-out test of Astra K6 workspaces. 12 sources × 5 held-out layouts for the complete conditions; HO-NoImg and HO-NoGrader are incomplete and not comparable with the complete conditions.
<table><tr><td>Condition</td><td>Semantics</td><td>Sources</td><td>Succ./n</td><td>Rate</td></tr><tr><td rowspan="2">E3-full</td><td>Full-loop workspaces; test agent may read</td><td>12</td><td></td><td>23/60 38.3%</td></tr><tr><td>prior-round raw records Full-loop</td><td>12</td><td></td><td>25/60 41.7%</td></tr><tr><td>HO-Fixed</td><td>workspaces; workspace files only Fixed workspaces</td><td>12</td><td>27/60 45.0%</td><td></td></tr><tr><td>HO-NoImg</td><td>on new layouts No-img workspaces;</td><td>9 (6)</td><td>13/35 37.1%</td><td></td></tr><tr><td>HO- NoGrader</td><td>incomplete No-grader workspaces;</td><td>6 (1)</td><td></td><td>5/21 23.8%</td></tr></table>

Table 8 Per-round online success in the Astra ablation. Successes out of 12 chains per condition and round (K1–K6 run in full).
<table><tr><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Fixed</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>No-img</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>No-grader</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>WS+FB</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>No-hist</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>K4-K6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>15/36</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>22/36</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>12/36</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5/36</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>17/36</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3/36</td></tr></table>