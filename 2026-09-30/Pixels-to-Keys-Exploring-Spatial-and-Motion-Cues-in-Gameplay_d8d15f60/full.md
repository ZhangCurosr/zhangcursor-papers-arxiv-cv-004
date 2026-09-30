# Pixels to Keys: Exploring Spatial and Motion Cues in Gameplay Inverse Dynamics

Abhishek Pillai<sup>1</sup>, Ekta Prashnani<sup>1</sup>, Joohwan Kim<sup>1</sup>, and Iuri Frosio<sup>1</sup>

NVIDIA {abpillai,eprashnani,sckim,ifrosio}@nvidia.com

Abstract. Video games ofer scalable environments for studying perception and control in embodied agents. Abundant online gameplay videos could supply demonstrations, but they rarely include player inputs for training. Inverse Dynamics Models (IDMs) have thus been proposed to infer inputs from frames. Large (up to 1B parameters) IDMs trained on ∼1K-2K gameplay hours demonstrate feasibility and cross-environment generalization at this scale, but researchers do not clarify what the key components are to recover individual actions and often report only aggregate accuracy that can mask rare-action failures. We study the problem in a data-constrained scenario to evaluate how spatial motion features, model architectures, and training objectives afect an IDM’s outcome and we analyse our models on per-key and balanced metrics such as F<sup>macro</sup><sub>1</sub> . Our experiments on Trackmania highlight the importance of factors like the model architecture and motion flow extraction in preprocessing, while also showing the limits of evaluation through unbalanced metrics. The application of the same architecture and training recipe to Cyberpunk 2077 reveals uneven performance across game mechanics. Our per-action evaluation and failure analysis highlight ambiguities from camera motion, delayed efects and imbalanced key-press frequencies that call for explicit modeling of 3D scene structure, long-term state and the adoption of proper losses in future implementations.

Keywords: Inverse Dynamics · Key-Press Reconstruction · Video Transformers · Optical Flow · Video Games · Embodied Agents

## 1 Introduction

Beyond their commercial importance, video games are used as testbeds for perception, sequential decision making and embodied control [3] [13] [17]. Learning from gameplay requires pairs of frames and actions issued by human or digital players to train agents with methods like Behavioral Cloning (BC) [2], Reinforcement Learning (RL) [21], or to gain understanding of visual-motor control.

Despite abundant gameplay videos available on the internet, only a few of these are annotated with Keyboard and Mouse (KBM) actions. Manual annotation at scale is practically impossible, whereas the acquisition of new datasets with frames labeled with KBM data is costly and time consuming. Therefore we study the possibility of automatically annotating such videos with KBM labels.

The problem is far from trivial for several reasons, including actions with delayed efects (e.g., inertia afecting camera motion without any steady input); also, different actions may produce the same visual results, whereas certain actions may produce poorly visible features (e.g., a car turning while driving against a wall). Additionally, the action distribution may be highly imbalanced (see Fig. 1). Disentangling camera motion from character motion is yet another complication.

Existing works establish this idea at scale. VPT [2] is a 0.5B parameter IDM trained on 1,962 hours of labeled Minecraft data and used to annotate 70K hours of web video. D2E [7] uses a generalist model with 1B parameters trained on 1.3K hours of data (259 hours of human demonstration, 1K hours of pseudo-labeled gameplay). These systems recover broad KBM action spaces and demonstrate the value of scale, but the question we intend to answer is deliberately smaller: before broad action recovery, how well can we recover navigation inputs from purely visual data? By answering this we highlight constraints that may become significant at a larger scale and in applications such as: annotation of online recordings, latency correction in poorly controlled acquisition setups, or the adoption of KBM annotation as an auxiliary task in complex training scenarios.

![](images/014967a01337d36cc71529dae91f4a3bc163100a7a20bf38710e04056f8dc92b.jpg)  
Fig. 1: Sample sequences and key-press distributions for Trackmania (a) and Cyberpunk 2077 (b). Distributions are on a log scale: key usage is highly imbalanced.

## Our contributions are:

1. For Trackmania, a $3 ^ { r d }$ person view and mechanics video game, we isolate the efects of diferent factors (including frame resolution, use of pretrained embeddings, loss terms and prediction size) and highlight that model architecture, optical flow in input, and contrastive learning provide the most significant advantages.

2. We highlight the importance of adopting per-key and unweighted metrics for proper evaluation of IDMs.

3. We identify limitations, open problems, and guidelines for future research by applying the same model and training recipe to Cyberpunk 2077 (1<sup>st</sup> person view). We analyze, on both games, the spatial action-query attention map, to show, for instance, the importance of equipping the IDM with the capability of disentangling the camera motion from character motion.

## 2 Related Work

Large-Scale IDMs and Action-Level Evaluation: Latent-action IDM techniques remove the labeled corpus altogether: for instance, LAPO [19] trains a latent IDM and forward model jointly on 8M action-free Procgen frames; Genie [5] scales this approach to 30K hours of video with a 300M-parameter latent model. Both recover a learned action space, later mapped using expert labels, rather than true actions: in other words, the reconstructed set of actions guarantees that the agent behaves like in the training examples, not that the set of key-presses is the same. Here we tackle a complementary question: when the action space is fixed and discrete, how accurately can true key-presses be recovered? In this, we are closer to D2E [7], a 1B-parameter generalist agent trained on 259 hours of human demonstration and 31 games plus 1K hours of pseudolabeled gameplay, scored on 50 ms non-overlapping event bins. When zero-shot on unseen titles, its keyboard accuracy falls to 63% (Battlefield 6) and 28% (Ogu and the Secret Forest): scale alone does not solve the key-press reconstruction problem.

We also found that the quantitative evaluation of IDMs is often aggregated over keys: VPT reports a single 90.6% key-press accuracy; D2E reports one keyboard accuracy per game. Per-class breakdowns are rare, and mostly reserved to world-model controllability or GUI-agent evaluations [9, 16, 25]. When reported, the variance is severe: MineWorld’s per-action F1 ranges from ∼ 0.50 (drop) to ∼ 0.80 (forward/backward). In VideoAgentTrek, F1 for press is 0.14, for scroll it reaches 0.86, but aggregation over all actions has F1 = 0.78. In other words, aggregation hides variance when the majority label dominates. This aspect should not be neglected in training and evaluation: in Trackmania (Fig. 1a), W (forward) is 4× more frequent than A/D (left/right), therefore an alwayspress-W predictor scores well on the aggregate while failing on rarer actions. For this reason we study per-key metrics both for training and evaluation.

Researchers in robotics face a similar recovery problem for continuous actions: H2O [10] and OKAMI [13] fit a parametric body model to human video, then retarget the estimated pose to humanoid joint angles. Because that target is itself an estimate, it is validated by downstream task success rather than compared with ground truth; our labels are key-presses, so predictions are scored directly.

Motion, Temporal and Semantic Space Design: Visual dynamics in games are complicated by 3D scene structure, elements that do not follow scene physics (such as HUD overlays) or real physics (such as magic items), and by the entanglement of the camera and character motion. To provide better motion understanding to our model, we test augmentation of the RGB input with RAFT optical flow, which encodes estimated per-pixel displacement between adjacent frames. Its all-pairs correlations and iterative refinement capture large camera and character motions, ofering direct cues for key-press prediction [22]. We also study architectures that are specifically designed to handle videos and, as a consequence, motion. In particular, we compare two model families: video transformers such as ViViT [1] and TimeSformer [4] that apply joint or factorized space-time attention to patch or tubelet tokens; and Hybrid models using a convolutional stem to extract local features and downsample $H \times W$ inputs before tokenization [8,24]. To handle the complex semantics of video games, we also test whether pretrained DINOv3 embeddings [20] (ViT-S/16) transfer to key-press estimation. Lastly, BCE used by prior IDMs [2, 7] can underperform on imbalanced and sparse multi-label targets, so we compare it with adaptive asymmetric loss [18], soft-F1 [6], and action-supervised contrastive initialization [11] as alternative training objectives.

## 3 Method

## 3.1 Problem Formulation

We describe the case of the Trackmania video game with 5 navigation keys ${ \cal { K } } = \{ \mathrm { W , A , S , D , S p } \}$ , where $S p$ stands for $S p a c e$ . The first four keys are used to accelerate / turn left / brake / turn right; the last one resets the car to the starting point (and it is therefore used only once per episode). For Cyberpunk and other video games, the set of navigation keys may be diferent (Fig. 1), but generalization is trivial. For direct comparison, we use the same five-key set K for Cyberpunk 2077. Our model takes in input a sequence of $T _ { i n }$ RGB frames and we adopt $T _ { i n } = 5$ as a baseline, but generalization to other lengths or input modes (for instance with an additional optical flow channel), is again trivial.

Given a gameplay clip $\mathbf { X } = ( \mathbf { x } _ { t - 2 } , \ldots , \mathbf { x } _ { t + 2 } ) \in \mathbb { R } ^ { T _ { i n } \times 3 \times H \times W }$ , our IDM predicts:

$$
\hat { \mathbf { y } } _ { t } = [ \hat { y } _ { t } ^ { \mathrm { W } } , \hat { y } _ { t } ^ { \mathrm { A } } , \hat { y } _ { t } ^ { \mathrm { S } } , \hat { y } _ { t } ^ { \mathrm { D } } , \hat { y } _ { t } ^ { \mathrm { S p } } ] \in [ 0 , 1 ] ^ { 5 }\tag{1}
$$

where $\hat { y } _ { t } ^ { \mathrm { W } }$ is the estimated probability of pressing W at time $t ,$ and multiple keys may be active simultaneously. At inference time, probabilities can be used to sample each key-press value or be thresholded at 0.5 (our selection).

Since we found experimentally that neither a $2 ^ { 5 } .$ -class joint-action formulation nor increasing clip length (from $T _ { i n } = 5 ~ ( 0 . 2 5 \mathrm { { s } }$ at 20 Hz) to $T _ { i n } = 1 6$ (0.80 s at $2 0 \mathrm { H z } ) )$ improves performance, we retain independent outputs and $T _ { i n } = 5$ in all our experiments. Furthermore, some of our architectures (see section 3.4) predict not only the key-presses at time t, but the full key states for the complete input interval, $\left\{ \hat { \mathbf { y } } _ { \tau } \right\} _ { \tau = t - 2 } ^ { t + 2 }$ . In the following, we refer to these as center-frame (or $T _ { o u t } = 1 )$ versus all-frames (or $T _ { o u t } = 5 )$ supervised models.

## 3.2 Datasets

We use 2 gameplay datasets: the first one comprises Trackmania recordings, a $3 ^ { r d }$ person driving video game; the second one comprises Cyberpunk 2077, a 1<sup>st</sup> person navigation video game (see Fig. 1). Frames were captured at 20Hz and $5 1 2 \times 5 1 2$ resolution. The size and splits of the two datasets are shown in Table 1. Models are trained and evaluated per game without mixing datasets, using only sequences with active gameplay. Each split is sampled from diferent recordings.

Table 1: Number of overlapping 5-frame sequences per dataset. Each split’s sequences belong to diferent recordings that are disjoint.
<table><tr><td>Game</td><td>Training sequences Validation sequences Testing sequences</td><td></td><td></td></tr><tr><td>Trackmania</td><td> $\overline { { 3 0 , 0 0 0 \ ( \sim 2 5 \mathrm { { m i n s } ) } } }$ </td><td> $\overline { { 2 0 , 2 9 4 \ : ( \sim 1 7 \mathrm { { m i n s } ) } } }$ </td><td> $\overline { { 2 7 , 9 8 4 \ : ( \sim 2 3 \mathrm { { m i n s ) } } } }$ </td></tr><tr><td>Cyberpunk 2077</td><td> $^ { 7 2 , 1 9 2 } \ ( \sim 6 0 \mathrm { m i n s } )$ </td><td>25,600 (~ 21 mins)</td><td> ${ 2 4 , 5 7 6 } \ ( \sim 2 0 \mathrm { { m i n s } ) }$ </td></tr></table>

We augment input frames using a random afine transform shared across the full sequence with probability 0.75, scale in [0.95, 1.05] range, rotation in $[ - 7 ^ { \circ } , 7 ^ { \circ } ]$ range, and translation in [−32, 32] pixels range. To address the key-press class imbalance (stats in Fig. 1) we sample sequences with replacement using inversefrequency weights. Let $a _ { i k } = 1$ if key k is pressed at any time in sequence i and 0 otherwise, and $f _ { k }$ the k’s frequency over the training set. In training, we sample sequence i with frequency $\begin{array} { r } { w _ { i } \propto 1 + \sum _ { k } a _ { i k } / ( f _ { k } + 0 . 0 1 ) } \end{array}$ ; validation and testing are on their complete splits without augmentation or resampling.

## 3.3 Training

Optimization All experiments run on 1 NVIDIA RTX PRO 6000 GPU and use BF16 autocasting, with losses computed in FP32. We use AdamW [12, 15] with weight decay 0.1 and learning rate $5 \times 1 0 ^ { - 6 } ;$ ; 5 linear warm-up epochs are followed by CosineAnnealingLR [14] to $1 0 ^ { - 7 }$ . Batch size is 128, dropout is 0.1, and each RGB or optical-flow input element is independently masked with probability 0.20. Ground-truth key states are always masked and are never provided to the model as input. Training is performed for a number of epochs between 45 and 100, with validation computed at each epoch. The model checkpoint with the highest validation macro- ${ \bf \nabla } \cdot { \cal F } _ { 1 }$ is saved for inference.

Training Loss Given the strong imbalance in the key-press probabilities, we adopt a Soft-F1 loss for training, defined as:

$$
F _ { 1 , k } ^ { \mathrm { s o f t } } = \frac { 2 T P _ { k } } { 2 T P _ { k } + F P _ { k } + F N _ { k } + 1 0 ^ { - 3 } }\tag{2}
$$

$$
\mathcal { L } _ { \mathrm { F 1 } } = 1 - \frac { 1 } { 5 } \sum _ { k = 1 } ^ { 5 } F _ { 1 , k } ^ { \mathrm { s o f t } }\tag{3}
$$

where $T P _ { k } , F P _ { k }$ , and $F N _ { k }$ are the soft True Positive, False Positive, and False Negative counts for key $k ,$ respectively, summed over each minibatch.

## 3.4 Ablation study (primary factors)

In our experiments we have found three factors emerging as the main ones affecting the model’s capability to label each frame with the correct set of pressed keys. These are (i) the model architecture, (ii) the additional optical flow channels in input, and (iii) the contrastive training objective: these are described here. Experiments on other factors with minor efects on the final result $( { \mathrm { i . e . } }$ frame resolution, BCE and asymmetric loss, pretrained embeddings) are detailed in Appendix A.

Architectures We considered architectures whose input/output is defined in 3.1, but have diferent internal skeletons: Convolutional Neural Networks (CNN), Vision Transformers (ViT), and Hybrid Transformers (HT).

CNN: A set of $T _ { i n }$ RGB frames is stacked into $3 \times T _ { i n }$ channels, followed by 3 convolutional layers $( 3 \times T _ { i n }  3 2$ (kernel/stride $4 / 4 ) , 3 2  1 6 ~ ( 3 / 1 )$ , and $1 6  4 ( 5 / 5 ) )$ with LeakyReLU activations. Adaptive average pooling then yields $\mathrm { ~ a ~ 4 ~ } \times \mathrm { ~ 6 ~ } \times \mathrm { ~ 6 ~ } = 1 4 4$ -dimensional vector, followed by MLP layers (144 −→ $1 6  5 \times T _ { i n } )$ . This model (CNN-512) produces an all-frames output: it predicts key-presses over $T _ { o u t } = 5$ frames (25 logits) and has 16,685 parameters. The training loss is computed on the entire output sequence, from t − 2 to $t + 2$

Spatiotemporal ViTs: Each RGB frame is split into $1 6 \times 1 6$ patches, yielding $3 2 \times 3 2 = 1 { , } 0 2 4$ tokens per frame and $1 { , } 0 2 4 \times T _ { i n }$ joint space-time tokens (5,120 total for $T _ { i n } = 5 )$ . Spatial and frame embeddings are learned end-to-end and precede 2 transformer blocks with embedding dimension 384, 4 heads, and MLP width 1,536; framewise mean pooling and a shared head then produce 5 logits per frame. Like the CNNs, this model (ViT-E2E-512) produces all-frames outputs $( T _ { o u t } = 5 )$ and training loss is computed over the full sequence $( t - 2 \tan t + 2 )$

Hybrid Transformer (HT): Our HT model retains CNN-512’s channel-stacked convolutional input: RGB contributes $3 \times T _ { i n }$ input channels. For $T _ { i n } = 5$ , HT-RGB has 15 channels. A 4-stage convolutional stem (kernel 16, stride 2, padding $7 ;$ GroupNorm and LeakyReLU) maps $C _ { \mathrm { i n } } \in \{ 1 5 , 2 3 \}$ through channels 64 → $1 2 8 \to 2 5 6 \to 2 5 6$ while reducing spatial size as $5 1 2  2 5 6  1 2 8  6 4  3 2$ The $2 5 6 \times 3 2 \times 3 2$ output is flattened into 1,024 tokens of dimension 256 and processed by 4 transformer blocks (LayerNorm before attention and MLP, 8 heads, MLP hidden dim. 1,024). 5 learned queries of dimension 256 cross-attend to the tokens, and a common $2 5 6  1 , 0 2 4  5$ head produces 5 logits per query. This model (HT-RGB-E2E) has ∼ 31.5M parameters. Unlike CNNs and ViTs, it is supervised over the keys pressed at time t only (center-frame architecture with $T _ { o u t } = 1 )$ , on which the loss is computed.

Additional motion flow in input Optical flow can be provided as an additional input channel. The rationale is that clean motion information in input may be used by the model to reconstruct the character and camera motion and thus facilitate the estimate of the key-presses in the input sequence. We use a RAFT-Large [23] model to compute the motion (dx, dy) for each of the 4 adjacent frame pairs (for $T _ { i n } = 5 )$ , and concatenate the resulting 8 channels with the RGB data. Our HT-RGBF-E2E model shares the same architecture as the HT-RGB model, but has 23 input channels (4 flow fields, 8 channels) instead of 15. It also has a comparable (∼ 31.5M parameters) size.

Contrastive embeddings We use a contrastive training objective as described below. An additional head with a 256 → 256 → 128 GELU projector is added to the HT-RGBF model to output ℓ<sub>2</sub>-normalized embeddings (notice that this is used in training only, thus the size of the model at inference time won’t change). Each training sequence is then augmented in 2 diferent ways, using (beyond the default afine transform noted in section 3.2) random contrast change in [0.85, 1.15] range, random brightness change in [−0.15, 0.15] range, and Gaussian noise with zero mean and $\sigma = 0 . 0 3 7 5$ . For a batch of B sequences, the 2 augmentations yield 2B embeddings $z _ { i } \in \mathbb { R } ^ { 1 2 8 }$ with $\| z _ { i } \| _ { 2 } = 1$ . Let $y _ { i } \in \{ 0 , 1 \} ^ { 5 }$ be the center-action label of sequence i and $A ( i ) = \{ 1 , \dotsc , 2 B \} \setminus \{ i \}$ contain all non-self comparisons and $P ( i ) = \{ p \in A ( i ) : y _ { p } = y _ { i } \}$ contain its positive action matches. The contrastive loss is then defined as:

$$
\mathcal { L } _ { \mathrm { S u p C o n } } = - \frac { 1 } { 2 B } \sum _ { i = 1 } ^ { 2 B } \frac { 1 } { \left| P ( i ) \right| } \sum _ { p \in P ( i ) } \log \frac { \exp ( z _ { i } ^ { \top } z _ { p } / \tau ) } { \sum _ { a \in A ( i ) } \exp ( z _ { i } ^ { \top } z _ { a } / \tau ) } , \qquad \tau = 0 . 2 0 .\tag{4}
$$

As embeddings are normalized, $z _ { i } ^ { \top } z _ { j }$ is cosine similarity; τ is softmax temperature. The loss averages over all 2B embeddings and their $| P ( i )$ | positives, while the paired augmentation guarantees $| P ( i ) | \geq 1$ . All $N ( i ) = A ( i ) \setminus P ( i )$ embeddings with $y _ { a } \neq y _ { i }$ are negatives: their presence in the denominator penalizes high negative similarity, separating them from positives. When training with contrastive loss, we first run 20 epochs to minimize $\mathcal { L } _ { \mathrm { S u p C o n } }$ and learn a continuous embedding space. After that, the prediction head is reinitialized and training continues for another 25 epochs with the prescribed loss only. This produces the HT-RGBF-PT-F1 model reported in Table 2.

## 4 Experiments

## 4.1 Evaluation metrics

Let $y _ { t } ^ { k }$ and $\hat { y } _ { t } ^ { k }$ be the ground-truth and predicted press probability of key k at timestep t. In validation, we threshold each $\hat { y } _ { t } ^ { k }$ at 0.5. For each key, the true/false positives/negatives $T P _ { k } , F P _ { k } , T N _ { k } .$ , and $F N _ { k }$ count positions with $( \hat { y } _ { t } ^ { k } , y _ { t } ^ { k } ) = ( 1 , 1 ) , ( 1 , 0 ) , ( 0 , 0 )$ , and (0, 1), respectively. The same terms (without k index) indicate the overall true/false positives/negatives, computed for all the keys over the entire validation or test dataset. To evaluate and compare the diferent models trained in our ablation study, we utilize the following metrics:

1. Accuracy $\begin{array} { r } { ( \mathrm { A c c . } ) = \frac { \mathrm { T P + T N } } { \mathrm { T P + T N + F P + F N } } } \end{array}$

2. $\begin{array} { r } { F _ { 1 , k } = \frac { 2 T P _ { k } } { 2 T P _ { k } + F P _ { k } + F N _ { k } } } \end{array}$

3. $\begin{array} { r } { F _ { 1 } ^ { \mathrm { m a c r o } } = \frac { 1 } { 5 } \sum _ { k = 1 } ^ { 5 } F _ { 1 , k } } \end{array}$

$$
\begin{array} { r } { 4 . \ F _ { 1 } ^ { \mathrm { m i c r o } } = \frac { 2 \sum _ { k } T P _ { k } } { 2 \sum _ { k } T P _ { k } + \sum _ { k } F P _ { k } + \sum _ { k } F N _ { k } } } \end{array}
$$

Accuracy is widely reported in previous works, but the majority class and inactive labels can dominate TP+TN, severely inflating performance. The second one, $F _ { 1 , k } ,$ , is the $F _ { 1 }$ score (that balances precision and recall) computed per key: it is therefore less biased in case of an imbalanced dataset, where the key k is only rarely or very often pressed. Since all k values have to be inspected, $F _ { 1 , k }$ is on the other hand tedious to review and hardly usable for ranking. $F _ { 1 } ^ { \mathrm { m a c r o } }$ is the mean over all $F _ { 1 , k } { \mathrm { : } }$ : it weighs down the imbalanced class by ignoring the key frequencies. It helps reveal rare-key performance and it is our primary metric. Notice in fact that, when evaluating video games, rare key-presses often play a fundamental role in the economy of the game: e.g., shooting may be less frequent than moving forward, but contrarily to navigation, precise and accurate shooting is required to stay alive. $F _ { 1 } ^ { \mathrm { m i c r o } }$ generalizes the traditional $F _ { 1 }$ score used in binary problems to the multilabel case: it pools counts across all keys, summarizing dataset-level recovery; it may still be afected by key frequency imbalance.

Table 2: Accuracy Acc. and $F _ { 1 }$ metrics on the test datasets; $2 0 + 2 5$ denotes 20 contrastive and 25 supervised epochs. All rows refer to Trackmania except the last one marked <sup>∗</sup>, which uses Cyberpunk 2077. Bold and underlined mark the highest and second-highest distinct Trackmania values in each metric column, respectively. For Cyberpunk 2077 we report for comparison only navigation keys that are in common with Trackmania, with Accuracy, $F _ { 1 } ^ { \mathrm { m a c r o } }$ , and $F _ { 1 } ^ { \mathrm { m i c r o } }$ over that set.
<table><tr><td rowspan="2">Model</td><td colspan="6">Configuration</td><td rowspan="2"> $\operatorname { A c c } .$ </td><td colspan="6">micro macro</td></tr><tr><td> $T _ { \mathrm { o u t } }$ </td><td></td><td></td><td>Arch. Res. Flow Initialization</td><td>Loss</td><td>Epochs</td><td></td><td></td><td>W</td><td>A</td><td>S</td><td>D</td></tr><tr><td>CNN-512</td><td>5</td><td>CNN 5122</td><td></td><td>none</td><td>soft-F1</td><td>100</td><td>.489</td><td>.499</td><td>.367</td><td>.902</td><td>.434</td><td>.075 .415</td><td>.008</td></tr><tr><td>ViT-E2E-512</td><td>5</td><td>ViT</td><td>5122</td><td>none</td><td>soft-F1</td><td>100</td><td>.887</td><td>.817</td><td>.637</td><td>.907 .698</td><td>.567</td><td>.755</td><td>.256</td></tr><tr><td>HT-RGB-E2E</td><td>1</td><td>HT</td><td>5122</td><td>none</td><td>soft-F1</td><td>50</td><td>.929</td><td>.880</td><td>.746</td><td>.921</td><td>.837 .648</td><td>.823</td><td>.502</td></tr><tr><td>HT-RGBF-E2E</td><td>1</td><td>HT</td><td>5122</td><td>none</td><td>soft-F1</td><td>50</td><td>.929</td><td>.878</td><td>.791</td><td>.920</td><td>.829 .655</td><td>.822</td><td>.728</td></tr><tr><td>HT-RGBF-PT-F1</td><td>1</td><td>HT</td><td>5122</td><td>contrastive</td><td>soft-F1 20 + 25</td><td></td><td>.938</td><td>.895</td><td>.793</td><td>.920 .869</td><td>.688</td><td>.870</td><td>.618</td></tr><tr><td>CP-HT-RGBF-PT-F1*</td><td>1</td><td>HT</td><td>5122</td><td>contrastive</td><td>soft-F1 20 + 25</td><td></td><td>.911 .715</td><td></td><td>.617</td><td>.786</td><td></td><td></td><td>.571 .710.543 .473</td></tr></table>

## 4.2 Results

Table 2 shows metrics on the test datasets for the main experiments; within each epoch budget, checkpoints are selected by $F _ { 1 } ^ { m a c r o }$ on the validation split. Configuration columns report the architecture, $T _ { o u t }$ , resolution, flow input, initialization strategy, loss used for training, and number of training epochs. Training and validation curves are in Appendix B. All results are reported on Trackmania, except the final row, where the HT-RGBF-PT-F1 model and training recipe is applied to Cyberpunk 2077. Complete configurations for the main experiments and secondary ablations are reported in Table 3 in the Appendix.

## 4.3 Quantitative Analysis

Diferent architectures are compared in the first three rows of Table 2. While an all-frames CNN model’s $F _ { 1 } ^ { \mathrm { m a c r o } }$ is as low as 0.367, all-frames ViTs achieve 0.637 using the same frame input resolution, training loss function and number of epochs. The center-frame HT architecture in the third row beats both of them with $F _ { 1 } ^ { \mathrm { m a c r o } } = 0 . 7 4 6$ . Despite the experiment being performed on one seed only, the high diference in $F _ { 1 } ^ { \mathrm { m a c r o } }$ together with the fact that all other metrics are also higher for HT (when compared to CNN and ViT) indicate with strong evidence the superiority of the HT architecture over CNN and ViT.

The comparison of the third (HT-RGB-E2E) and fourth (HT-RGBF-E2E) rows in the same Table allows estimating the efect of the additional motion flow in input. We notice that F<sup>macro</sup> improves by approximately 6% relative (from 0.746 to 0.791) when motion information is provided. Other metrics remain more or less in the same ballpark, with the exception of $F _ { 1 , S p }$ that shows a 45% increase. Also in this case the improvement is thus numerically relevant, although not as large as the one registered for the case of diferent architecture: more experiments with diferent seeds may be needed to establish more precisely the advantage of the additional motion flow in input.

The comparison between the HT-RGBF-E2E and HT-RGBF-PT-F1 isolates the efect of contrastive initialization. The HT-RGBF-PT-F1 model shows a consistent improvement in all core navigation metrics with the only exception of $F _ { 1 , S p }$ . Even if the improvement is on average not large, the fact that it is common to the majority of the metrics suggests that the improvement over HT-RGBF-E2E is real. This highlights, likewise in other contexts, the importance of learning semantically significant and smooth representations of the input patches in video games. This claim becomes even more significant when considering that embedding extraction performed with models pretrained on real-world data (see the DINOv3 case in the Appendix) did not provide any improvement in our experiments. In other words, the best performance is achieved when the embedding space is learned end-to-end (and thus visually and semantically meaningful in the video-game economy) and smooth enough to guarantee regular learning.

Since HT-RGBF-PT-F1 is experimentally the best architecture and recipe on Trackmania, we finally tested it on Cyberpunk 2077, a first-person shooter video game (while Trackmania is a third-person racing game). This model, indicated as CP-HT-RGBF-PT-F1\* in the Table, yields $\operatorname { A c c . } = 0 . 9 1 1$ and $F _ { 1 } ^ { \mathrm { m a c r o } } = 0 . 6 1 7$ (compared to 0.938 and 0.793 for Trackmania, respectively). In other words, results on Cyberpunk 2077 are slightly worse than those achieved on Trackmania. This result appears reasonable given the higher complexity of Cyberpunk’s visual and action scenarios (shooting and jumping integrated with base actions) and the sheer diversity in movement when comparing it to a constrained racing game like Trackmania.

## 4.4 Qualitative analysis

To better understand the reasons for failure or success of the trained IDMs, we inspect sequences that show small, medium or large key-press reconstruction errors. For each game, we show here two pairs of typical correct/incorrect predictions from the validation dataset using HT-RGBF-PT-F1 on Trackmania (Figures 2 and 3) and CP-HT-RGBF-PT-F1 on Cyberpunk 2077 (Figures 4 and 5). Each example shows the $T _ { i n } = 5$ frame sequence from t − 2 to t + 2 (top) and spatially aligned attention overlays on it (bottom). After RGB and optical flow input concatenation, cross-attention is averaged over 8 heads and min-max normalized; the resulting map is overlaid on each frame to visualize its spatial contribution to the center-frame prediction. GT and Pred in each panel denote the active ground-truth and 0.5 thresholded predicted keys.

(a) Braking (skid marks seen) as the vehicle approaches a visible curve (S, D).  
![](images/4f0429d10dd3a85753b7b5905e6a3514296bbc0b945b32cc58e4a1eb9d4ab561.jpg)  
(b) Forward-right on observing the slight wheel angle change and inertia (W, D).

Fig. 2: Correctly predicted Trackmania sequences for HT-RGBF-PT-F1 (attention in the lower row). Pred matches GT for the supervised frame t. Better seen at 6x zoom.  
![](images/04477ddf75682d394d77945be17a77717ca3af7898df4a57262a0392403e3e02.jpg)  
(a) Reverse-right (S, D) is confused with (b) Airborne braking is omitted as it is ambiguforward-left (W, A) while fixing barrier collision. ous from the visual evidence.  
Fig. 3: Incorrectly predicted Trackmania sequences for HT-RGBF-PT-F1 (attention in the lower row). Reverse-right (S, D) is predicted as forward-left (W, A), indicating lack of 3D understanding, while airborne braking and steering (S, D) is predicted as (W, A) as it is visually ambiguous. Better seen at 6x zoom.

Trackmania For correct predictions (Fig. 2), predicted keys are supported by visual evidence that persists across the five frames: the skid marks support braking and right steering, while wheel pose and track-relative motion support forward-right prediction. Notice also that the attention map often focuses on features that change with the camera motion, supporting the importance of motion flow in input. The failure cases (Fig. 3) reflect two ambiguities: when the vehicle is stuck, 3<sup>rd</sup>-person camera motion makes reverse-right appear as forward-left, yielding W, A; airborne braking has no immediate visual efect and is therefore missed. These cases motivate explicit separation of ego- and camera motion and longer temporal context for actions with delayed visual efects.

Cyberpunk Figures 4 and 5 show the corresponding Cyberpunk 2077 examples. The success case recovers sustained backward motion despite transient combat efects. In the partial match, lateral motion supports D, but weapon and hitresponse motion may cause the false W prediction. This perhaps is the general policy of forward movement being applied. Both failures invert S to W: one despite newly revealed fixed room features (white lights), and the other in a visually sparse corridor while also adding A. They indicate that even optical flow cannot fully disentangle player from camera motion (which must indeed be performed by the IDM) and combat-specific motion in 1<sup>st</sup>-person scenes.

![](images/16f6ae6b5d7c44f5e7b29788642f68240a86543a00ff08f9f3ab013c0cbc2793.jpg)  
(a) Backward movement recovered during com- (b) Right movement (D) recovered with a false bat (S). forward prediction (W, D).

Fig. 4: Correctly classified Cyberpunk 2077 sequences for CP-HT-RGBF-PT-F1 (attention in the lower row). Exact backward-motion match and partial right-movement match after being shot. Better seen at 6x zoom.  
![](images/39cedb6984f129c7adf548bce8581da191e7fa35c2371807dcb8179a50e79232.jpg)  
(a) Backward movement (S) confused with for- (b) Backward movement (S) confused with ward movement (W). forward-left (W, A).  
Fig. 5: Incorrectly classified Cyberpunk 2077 sequences for CP-HT-RGBF-PT-F1 (attention in the lower row). Both examples invert backward motion; in the second a false left movement is also predicted. Better seen at 6x zoom.

## 5 Discussion and Conclusion

## 5.1 Positive Results and Design Implications

Our main finding is that the spatiotemporal ViT outperforms traditional CNNs, while the HT architecture coupled with optical flow and contrastive initialization forms the strongest combination studied. Our intuition is that the optical flow helps the model disentangle complex motion features (such as those created by camera motion/inertia while the playable character moves) while contrastive initialization helps build a more reliable embedding space- especially when training from scratch. The embeddings learned on a target game mechanic (like navigation) or environment are likely more focused than general pretrained embeddings, as shown in the Appendix when comparing against a model trained with frozen DINOv3. Our experiments confirm that using accuracy Acc. in evaluation can hide minority-key errors because inactive (or always-on) keys may dominate it. Further experiments in the Appendix seem to indicate that a balanced metric should be used for training as well (in fact we use a soft-F1 instead of BCE loss), but the numerical evidence is not suficient to fully support this statement: more tests are needed to confirm it. Our findings also indicate (see Appendix) that higher frame resolution provides small (yet consistent) improvements in performance, at the cost of higher compute. Nonetheless, we decide to (and suggest future researchers to) work at the highest resolution to standardize subsequent models and avoid a resolution bottleneck in detail-heavy games like Cyberpunk.

## 5.2 Limitations and Future Directions

When compared against models trained for BC (and thus learning a reliable, average policy to predict the next key-presses) on the same game, we found that our IDMs achieve only slightly superior performance. For instance, a CNN model successfully trained to play Trackmania typically achieves $F _ { 1 , W } \sim 0 . 9$ and $F _ { 1 , A } \sim F _ { 1 , D } \sim 0 . 8$ , whereas our best IDM (see Table 2) achieves 0.920, 0.869, and 0.870 on the same metrics. Our interpretation is that the IDM may partly learn the dataset’s average policy rather than a precise visual-to-key mapping. This interpretation requires further validation but motivates larger, more diverse datasets, as in VPT and D2E, to avoid regression toward the average policy. However, scale alone cannot remove visual ambiguity: gameplay projects 3D motion into 2D, while the camera moves independently. Optical flow measures displacement, not its cause; a reversing vehicle can coincide with forward camera motion, while weapons and HUD overlays are independent of navigation. Explicit 3D scene and ego-motion modeling may therefore be required to disentangle these dynamics and further improve IDM performance. This may aid the learning of player motion (and thus the reconstruction of the controlling key-presses) independently from that of the camera.

We also highlight that certain key-presses produce little or no immediate visual evidence, making reconstruction ambiguous from pixels alone (Figures 3 and 5). For example, steering into a wall may leave vehicle position and orientation unchanged except for a subtle wheel-angle change. Such a press is irrelevant only if it has no later consequence: inertia, limited-resource consumption, or adding an item to an inventory can alter latent game state without becoming visible within a T-frame context. Recovering these partially observable actions, when possible, may require a longer context, explicit state memory, and high resolution to capture minute visual details. These aspects should be taken into careful consideration for future research in this direction.

## References

1. Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., Schmid, C.: Vivit: A video vision transformer. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 6816–6826 (2021). https://doi.org/10.1109/ ICCV48922.2021.00676

2. Baker, B., Akkaya, I., Zhokov, P., Huizinga, J., Tang, J., Ecofet, A., Houghton, B., Sampedro, R., Clune, J.: Video pretraining (vpt): Learning to act by watching unlabeled online videos. In: Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., Oh, A. (eds.) Advances in Neural Information Processing Systems. vol. 35, pp. 24639–24654. Curran Associates, Inc. (2022). https://doi.org/10. 52202/068431- 1789, https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/9c7008aff45b5d8f0973b23e1a22ada0-Paper-Conference.pdf

3. Berner, C., Brockman, G., Chan, B., Cheung, V., Debiak, P., Dennison, C., Farhi, D., Fischer, Q., Hashme, S., Hesse, C., Józefowicz, R., Gray, S., Olsson, C., Pachocki, J.W., Petrov, M., de Oliveira Pinto, H.P., Raiman, J., Salimans, T., Schlatter, J., Schneider, J., Sidor, S., Sutskever, I., Tang, J., Wolski, F., Zhang, S.: Dota 2 with large scale deep reinforcement learning (2019), https://api.semanticscholar.org/CorpusID:209376771

4. Bertasius, G., Wang, H., Torresani, L.: Is space-time attention all you need for video understanding? In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 813–824. PMLR (18–24 Jul 2021), https://proceedings. mlr.press/v139/bertasius21a.html

5. Bruce, J., Dennis, M.D., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., Aytar, Y., Bechtle, S.M.E., Behbahani, F., Chan, S.C., Heess, N., Gonzalez, L., Osindero, S., Ozair, S., Reed, S., Zhang, J., Zolna, K., Clune, J., de Freitas, N., Singh, S., Rocktäschel, T.: Genie: Generative interactive environments. In: Forty-first International Conference on Machine Learning (2024), https://openreview.net/forum?id=bJbSbJskOS

6. Bénédict, G., Koops, V., Odijk, D., de Rijke, M.: sigmoidf1: A smooth f1 score surrogate loss for multilabel classification (2022), https://arxiv.org/abs/2108. 10566

7. Choi, S., Jung, J., Seong, H., Kim, M., Kim, M., Cho, Y., Kim, Y., Park, Y., Yu, Y., Lee, Y.: D2e: Scaling vision-action pretraining on desktop data for transfer to embodied ai. In: Vondrick, C., Hariharan, B., Rafel, C., Pinto, L., Yang, D., Faust, A. (eds.) International Conference on Learning Representations. vol. 2026, pp. 46207–46236 (2026), https://proceedings.iclr.cc/paper\_files/paper/ 2026/file/4cafca2c978ce63cafcea9f0484a93c0-Paper-Conference.pdf

8. Dai, Z., Liu, H., Le, Q.V., Tan, M.: Coatnet: Marrying convolution and attention for all data sizes. In: Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems. vol. 34, pp. 3965– 3977. Curran Associates, Inc. (2021), https://proceedings.neurips.cc/paper\_ files/paper/2021/file/20568692db622456cc42a2e853ca21f8-Paper.pdf

9. Guo, J., Ye, Y., He, T., Wu, H., Jiang, Y., Pearce, T., Bian, J.: Mineworld: a real-time and open-source interactive world model on minecraft (2025), https: //arxiv.org/abs/2504.08388

10. He, T., Luo, Z., Xiao, W., Zhang, C., Kitani, K., Liu, C., Shi, G.: Learning human-to-humanoid real-time whole-body teleoperation. In: 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 8944–8951 (2024). https://doi.org/10.1109/IROS58592.2024.10801984

11. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., Krishnan, D.: Supervised contrastive learning. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 18661–18673. Curran Associates, Inc. (2020), https://proceedings.neurips.cc/paper\_files/paper/2020/file/ d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf

12. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization (2017), https: //arxiv.org/abs/1412.6980

13. Li, J., Zhu, Y., Xie, Y., Jiang, Z., Seo, M., Pavlakos, G., Zhu, Y.: Okami: Teaching humanoid robots manipulation skills through single video imitation (2024), https: //arxiv.org/abs/2410.11792

14. Loshchilov, I., Hutter, F.: Sgdr: Stochastic gradient descent with warm restarts (2017), https://arxiv.org/abs/1608.03983

15. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization (2019), https: //arxiv.org/abs/1711.05101

16. Lu, D., Xu, Y., Wang, J., Wu, H., Wang, X., Wang, Z., Yang, J., SU, H., Chen, J., Chen, J., Mao, Y., Lin, J., Hui, B., Yu, T.: Videoagenttrek: Computeruse pretraining from unlabeled videos. In: Vondrick, C., Hariharan, B., Rafel, C., Pinto, L., Yang, D., Faust, A. (eds.) International Conference on Learning Representations. vol. 2026, pp. 122696–122720 (2026), https://proceedings. iclr.cc/paper\_files/paper/2026/file/c78b7a0016da6bf5be2a0691c17943ee-Paper-Conference.pdf

17. Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., Riedmiller, M.: Playing atari with deep reinforcement learning (2013), http:// arxiv.org/abs/1312.5602, cite arxiv:1312.5602Comment: NIPS Deep Learning Workshop 2013

18. Ridnik, T., Ben-Baruch, E., Zamir, N., Noy, A., Friedman, I., Protter, M., Zelnik-Manor, L.: Asymmetric loss for multi-label classification. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 82–91 (October 2021)

19. Schmidt, D., Jiang, M.: Learning to act without actions. In: Kim, B., Yue, Y., Chaudhuri, S., Fragkiadaki, K., Khan, M., Sun, Y. (eds.) International Conference on Learning Representations. vol. 2024, pp. 9379– 9395 (2024), https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 27985d21f0b751b933d675930aa25022-Paper-Conference.pdf

20. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jégou, H., Labatut, P., Bojanowski, P.: Dinov3 (2025), https://arxiv.org/abs/2508.10104

21. Sutton, R.S., Barto, A.G.: Reinforcement Learning: An Introduction. The MIT Press, second edn. (2018), http://incompleteideas.net/book/the-book-2nd. html

22. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: Vedaldi, A., Bischof, H., Brox, T., Frahm, J.M. (eds.) Computer Vision – ECCV 2020. pp. 402–419. Springer International Publishing, Cham (2020)

23. Wang, Y., Lipson, L., Deng, J.: SEA-RAFT: Simple, eficient, accurate RAFT for optical flow. In: European Conference on Computer Vision (ECCV) (2024)

24. Xiao, T., Singh, M., Mintun, E., Darrell, T., Dollar, P., Girshick, R.: Early convolutions help transformers see better. In: Ranzato, M., Beygelzimer, A.,

Dauphin, Y., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems. vol. 34, pp. 30392–30400. Curran Associates, Inc. (2021), https://proceedings.neurips.cc/paper\_files/paper/2021/file/ ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf

25. Zhang, Y., Peng, C., Wang, B., Wang, P., Zhu, Q., Kang, F., Jiang, B., Gao, Z., Li, E., Liu, Y., Zhou, Y.: Matrix-game: Interactive world foundation model (2025), https://arxiv.org/abs/2506.18701

## Appendix (Supplementary Material)

## A Ablation study (secondary factors)

As explained in Section 3.4, we detail here additional ablation studies on factors that experimentally had minor, no, or negative impact on the model performance. These are: (i) frame resolution, (ii) pretrained embeddings, and (iii) loss functions.

Frame Resolution: We evaluated a CNN variant, named CNN-128, that takes 128 × 128 input frames instead of the 512 × 512 frames used by CNN-512. The architectures are otherwise identical, and both models have 16,685 parameters; the comparison therefore isolates input resolution. Rows 1 and 2 of Table 4 compare the performance of the two architectures, showing a small (yet consistent) advantage for the higher resolution input frames, as expected. This suggests working at the highest possible resolution, without (on the other hand) increasing the frame resolution beyond a certain limit when compute is an important factor.

Pretrained Embeddings: We evaluated a ViT variant, named ViT-DINO-512, taking in input 512 × 512 (like ViT-E2E-512) and computing embeddings on the 16 × 16 patches using DINOv3 [20]. While ViT-E2E-512 learns its patch projection end-to-end, the DINOv3 ViT-S/16 backbone in ViT-DINO-512 is frozen, removing 1 class and 4 register tokens before the common encoder. Rows 3 and 4 of Table 4 compare the performance of the two models trained with the exact same recipe and show no clear advantage for DINOv3 embeddings over the ones that are learned end-to-end. This result seems to discourage the adoption of embedders pretrained on real-world data for video game scenarios, where the visual features may be largely diferent from typical scenes observed in the real world. Since numerical diferences are small, however, such a conclusion should be taken with a grain of salt: more experiments on diferent seeds and diferent games may be needed to draw a reliable conclusion.

Loss Functions: We evaluated the efect of adopting diferent loss functions in training on the HT architecture. In the ablation study, we compare this against the traditional BCE loss and an adaptive asymmetric loss (denoted as BCE<sup>A</sup>), which penalizes false predictions $( F P _ { k }$ and $F N _ { k } )$ more for rarer keys: the BCE loss is multiplied by a factor of 25 when the key frequency is $< 0 . 1 0 $ , 20 for frequency in the [0.10, 0.50) interval, and 5 otherwise. The last row of Table 4 shows the corresponding model (HT-RGB-ASYM). Also in this case, the performance diferences for the same model trained with the same recipe and diferent loss functions are minor and therefore more experiments on diferent seeds and diferent games may be needed to draw a reliable conclusion. Nonetheless, the metrics suggest a slight advantage for the soft-F1 loss, especially for the rarer

Trackmania PT-RGBF Cyberpunk PT-RGBF

Table 3: Experiment configurations for Tables 2 and 4. E2E is end-to-end; $d / L / H$ is transformer embedding dimension, number of blocks, and number of attention heads.
<table><tr><td>Experiment</td><td>Configuration</td><td>Purpose</td></tr><tr><td>CNN-128</td><td> $^ { 1 2 8 ^ { 2 } , }$  15-channel RGB, soft-F1</td><td>resolution baseline</td></tr><tr><td>CNN-512</td><td> $5 1 2 ^ { 2 } ,$  otherwise CNN-128</td><td>resolution ablation</td></tr><tr><td>ViT-E2E-512</td><td> $5 1 2 ^ { 2 } , \mathrm { R G B } , d / L / H = 3 8 4 / 2 / 4 , \mathrm { s o f t - F 1 }$ </td><td>ViT baseline</td></tr><tr><td>ViT-DINO-512</td><td>ViT-E2E with DINOv3 features</td><td>spatial-pretraining ab- lation</td></tr><tr><td>HT-RGB-E2E</td><td> $5 1 2 ^ { 2 } ,$  15-channel RGB,  $d / L / H = 2 5 6 / 4 / 8 ,$ </td><td>soft-F1 HT baseline</td></tr><tr><td>HT-RGB-BCE</td><td>HT-RGB-E2E with BCE</td><td>loss ablation</td></tr><tr><td>HT-RGB-ASYM</td><td>HT-RGB-E2E with  $\mathrm { B C E ^ { A } }$ </td><td>loss ablation</td></tr><tr><td>HT-RGBF-E2E</td><td>HT-RGB-E2E with 8 flow channels (23 total)</td><td>flow ablation</td></tr><tr><td>HT-RGBF-PT-F1</td><td colspan="2">HT-RGBF with 20-epoch contrastive pretraining final recipe</td></tr></table>

CP-HT-RGBF-PT-F1 HT-RGBF-PT-F1 applied to Cyberpunk 2077 second-game application

Table 4: Accuracy Acc. and $F _ { 1 }$ metrics on the Trackmania test split (for the secondary ablations reported in Table 3); 20+25 denotes 20 contrastive and 25 supervised epochs. Bold and underlined mark the highest and second-highest distinct values in each metric column, respectively.
<table><tr><td rowspan="2">Model</td><td colspan="8">Configuration</td><td colspan="8"></td></tr><tr><td></td><td></td><td></td><td></td><td>Arch. Res. Flow Initialization</td><td>Loss</td><td>Epochs</td><td> $\operatorname { A c c . }$ </td><td>micro macro</td><td></td><td></td><td>W</td><td>A</td><td>S</td><td>D</td><td>Sp</td></tr><tr><td>CNN-128</td><td>5</td><td>CNN</td><td> $1 2 8 ^ { 2 }$ </td><td>N</td><td>none</td><td></td><td>soft-F1</td><td>100</td><td> $. 4 6 9$ </td><td>.492</td><td>.361</td><td>.902</td><td>.430</td><td>.074</td><td>.392</td><td>.009</td></tr><tr><td>CNN-512</td><td>5</td><td>CNN 5122</td><td></td><td>N</td><td>none</td><td></td><td>soft-F1</td><td>100</td><td>.489 .499</td><td></td><td>.367</td><td>.902</td><td>.434</td><td>.075</td><td>.415</td><td>.008</td></tr><tr><td>ViT-E2E-512</td><td>5</td><td>ViT</td><td>5122</td><td>N</td><td>none</td><td>soft-F1</td><td>100</td><td>.887</td><td>.817</td><td></td><td>.637</td><td>.907</td><td>.698</td><td>.567</td><td>.755</td><td>.256</td></tr><tr><td>ViT-DINO-512</td><td>5</td><td>ViT</td><td>512²</td><td>N</td><td>DINOv3</td><td>soft-F1</td><td></td><td>100</td><td>.870 .790</td><td></td><td>.624</td><td>.898</td><td>.696</td><td>.452</td><td>.664</td><td>.411</td></tr><tr><td>HT-RGB-E2E</td><td>1</td><td>HT</td><td>512²</td><td>N</td><td>none</td><td>soft-F1</td><td>50</td><td>.929</td><td>.880</td><td></td><td>.746</td><td>.921</td><td></td><td>1.837.648</td><td>.823</td><td>.502</td></tr><tr><td>HT-RGB-BCE</td><td>1</td><td>HT</td><td>512²</td><td>N</td><td>none</td><td></td><td>BCE</td><td>50</td><td>.930 .875</td><td></td><td>.730</td><td>.919</td><td>.828</td><td>.642</td><td>.825</td><td>.437</td></tr><tr><td>HT-RGB-ASYM</td><td>1</td><td>HT</td><td>5122</td><td>N</td><td>none</td><td></td><td>BCEA</td><td>50</td><td>.922</td><td>.861</td><td>.719</td><td>.916</td><td>.801</td><td>.625</td><td>.796</td><td>.456</td></tr></table>

$S p$ key, as introduced in the main paper. This explains why we adopted it as our preferred choice.

## B Training and Validation Curves

Figures 6 and $7$ report training and validation losses and relevant metrics within the epoch budgets used in Table 2. Reported checkpoints maximize validation macro- $F _ { 1 }$ within these budgets.

![](images/4391f9a84a51dc81ab2d769ba46da7a589fb5e7894e0faee0052da78aff09de8.jpg)

![](images/fc43a0c7d9455343423d8b66004e1c6852d48cb84c7af75e4f7a5a7c785985a9.jpg)

![](images/b62d01789b23269ce69278ab817e25aea61dd0c184bcf25bed0e876eae158a35.jpg)

Fig. 6: Contrastive-pretraining curves for the RGB+flow HT encoders on Trackmania and Cyberpunk 2077: (a) supervised contrastive loss, (b) positive-pair cosine similarity, and (c) negative-pair cosine similarity over the 20-epoch pretraining budget.  
![](images/5e141aafd537eed6bb0d0dbc7e935d3878f2379d3d26786ddfb4a5a231b84776.jpg)

![](images/0796f4494f9086a1e202c7b6bc509236f6340a8000a3b35dcebc352faf403e44.jpg)

![](images/89ecbfb5d685023c4eed0d5165c7f40ff1970ef848acf59d2909cb99bcfabd03.jpg)

![](images/1bd64c92e2713827f858bbea4a3dd483f4328661cafb9ece3aeda963fe12feb1.jpg)

![](images/4f8238fa3782777cde3257b531b85adb11d544d42ebc56cbebccf2705e1db42f.jpg)  
CNN-128 CNN-512 ViT-E2E-512 ViT-DINO-512 HT-RGB-E2E HT-RGB-BCE HT-RGB-ASYM HT-RGBF-E2E HT-RGBF-PT-F1 CP-HT-RGBF-PT-F1  
Fig. 7: Supervised-training curves within the epoch budgets reported in Table 2: (a) training objective (loss), (b) validation soft- ${ \bf \nabla } \cdot F _ { 1 }$ loss, (c) validation accuracy, (d) validation micro- $F _ { 1 }$ , and (e) validation macro- $F _ { 1 }$ . Thresholded metrics use 0.5. Pretrained models begin supervised epoch 0 after 20 contrastive epochs; training-objective values are comparable only between runs using the same loss (asymmetric loss scales diferently as noted in Section 3.3).