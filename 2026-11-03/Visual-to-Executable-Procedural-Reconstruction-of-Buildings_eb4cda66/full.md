# Visual-to-Executable Procedural Reconstruction of Buildings

Zichong Zhu<sup>1</sup>, Zhihao Liu<sup>2</sup>, Andrei Sharf<sup>3</sup>, Zhanglin Cheng<sup>1∗</sup>

<sup>1</sup>Shenzhen VisuCA Key Lab, Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences <sup>2</sup>The University of Tokyo, <sup>3</sup>Ben-Gurion University of the Negev

## Abstract

Procedural modeling offers compact, editable, and reusable representations for architectural content, but building such models still depends heavily on manually authored grammars. We study image-based procedural building reconstruction from in-the-wild images, with the goal of recovering an executable procedural description that can be instantiated as an editable 3D building asset. Because a single image provides only partial and view-dependent evidence, we focus on recovering the dominant visible facade(s) and attach the recovered facade program to a simple mass model for building-level instantiation. Our method decomposes this task into two structured prediction stages. First, a Global Structure Layout Tree recovers facade-scale split-and-repeat organization through a fine-tuned layout model with render-guided visual-symbolic reflection. Second, a Component Parameterization schema maps representative facade tiles to structured procedural attributes for local architectural component synthesis. These two representations are then deterministically compiled into executable Computer Generated Architecture (CGA) programs, avoiding the brittleness of direct low-level program generation. Experiments show that the proposed framework handles both regular facades and challenging in-the-wild images, while maintaining high structural consistency, procedural executability, and fine-grained architectural detail. The recovered programs further enable controllable and reusable 3D building asset generation.

## 1 Introduction

Procedural modeling is a compelling paradigm for architectural content creation. Buildings are naturally composed of repeated elements, hierarchical subdivisions, and parameterized components, making procedural descriptions particularly well-suited to their structure [Wonka et al., 2003, Müller et al., 2006]. Compared with explicit geometry or purely neural representations [Mildenhall et al., 2021, Kerbl et al., 2023, Poole et al., 2023], procedural descriptions are compact, editable, and reusable, making them attractive for scalable and controllable 3D building asset generation. Yet, constructing such models still largely depends on manually authored grammars and domain expertise, which severely limits their practical scalability.

In this paper, we study image-based procedural building reconstruction from an in-the-wild image. Our goal is to recover an executable and editable procedural description that can be instantiated as a 3D building asset. Since a single image provides only partial and view-dependent evidence, we focus on the dominant visible facade(s) and treat the recovered program as a procedural interface for controllable building asset generation, while unseen geometry is instantiated using simple procedural completion rules.

Existing approaches only partially address this goal. Facade parsing methods recover semantic labels or dense structural cues, but do not produce editable or executable procedural programs. Classical inverse procedural modeling methods exploit architectural regularity, but typically assume rectified views, segmented layouts, or predefined grammar families. More recently, learning-based and LLM/VLM-based approaches suggest the possibility of image-to-program prediction, yet direct low-level program generation remains brittle because valid procedural programs are non-unique, tightly constrained by grammar validity, and hard to verify during prediction.

Our key observation is that the difficulty of this problem lies not only in generating code, but in bridging ambiguous visual evidence and execution-constrained procedural representations. The same building facade may admit multiple valid procedural decompositions with no single correct answer, while in-the-wild images further introduce perspective distortion, occlusion, and partial visibility, making it difficult to infer a clean orthogonal split-and-repeat scaffold directly from raw RGB evidence. We therefore avoid direct low-level image-to-code generation and instead cast the problem as verifiable image-to-structure and image-to-attributes prediction, followed by deterministic compilation into executable programs.

This formulation raises four coupled challenges. First, facade-scale organization is hierarchical and non-unique: the same facade may admit different split orders, repeat abstractions, or grouping strategies, making direct supervision on a single low-level program unstable. Second, in-the-wild images provide incomplete and view-distorted observations: perspective effects, occlusion, illumination variation, and partial visibility make it difficult to infer a clean orthogonal split-and-repeat scaffold from a non-rectified image. Third, local architectural elements such as windows, entrances, arches, sills, stairs, and hardware must be recovered as procedural parameters rather than merely detected as visual instances. Fourth, the recovered global scaffold and local components must be compositionally compatible and directly executable. A visually plausible output is insufficient if it violates grammar constraints or cannot be compiled.

To address these challenges, we propose a hierarchical neuro-symbolic framework with two structured intermediate representations. At the global level, we recover a Global Structure Layout Tree that captures facade-scale split-and-repeat organization and is iteratively refined through a render-guided visual-symbolic reflection process, making structural mismatches easier to identify and correct than direct symbolic comparison would allow. At the local level, we recover a Component Parameterization for representative repeated elements, mapping tile crops to structured attributes rather than raw procedural code. This design turns under-specified local geometry into a constrained and verifiable prediction problem.

The two levels are then assembled by deterministic compilers into executable CGA programs, producing a procedural building representation that is both editable and directly instantiable as a 3D asset. By separating global structural reasoning, local procedural grounding, and deterministic execution, our framework improves verifiability, robustness, and executability over direct low-level code generation.

Our contributions are threefold:

1. We formulate in-the-wild image-based procedural building reconstruction as a two-level neuro-symbolic prediction problem and introduce a representation bridge consisting of a Global Structure Layout Tree and a Component Parameterization schema.

2. We develop a render-guided visual feedback and symbolic repair mechanism for recovering global facade structure from incomplete and non-rectified observations.

3. We develop a component-level inverse parameterization and deterministic compilation pipeline that assembles the recovered structure and local components into executable CGA programs for procedural 3D building asset generation.

## 2 Related Work

Procedural and inverse procedural modeling for architecture. In architectural reconstruction, procedural models instantiate executable structured programs that encode geometry through editable and executable symbolic rules. Classical grammar-based systems, such as Instant Architecture and CGA shape grammar, demonstrate the effectiveness of rule-based representations for buildings and urban scenes [Wonka et al., 2003, Müller et al., 2006]. Inverse procedural modeling studies the reverse problem of recovering procedural structures and parameters from observations. The single-image building modeling system of Nishida et al. [2018] is closely related to our overall goal, as it recovers building mass, facade grammar, and window/door grammars from an image with a user-provided silhouette. Facade-level methods recover procedural facade definitions from rectified or segmented structures [Müller et al., 2007, Koutsourakis et al., 2009, Teboul et al., 2011, Wu et al., 2014], with FaçAID [Plocharski et al., 2024] learning a transformer-based mapping from segmented facades to procedural definitions. In our framework, such facade layout analysis corresponds mainly to the global structure recovery stage; we further ground repeated local tiles into procedural parameterization and assemble both levels into executable CGA programs from in-the-wild RGB images.

Visual program synthesis and neuro-symbolic reconstruction. More broadly, our task is related to visual program synthesis, where executable symbolic programs are inferred from visual observations. Early inverse-graphics systems such as SPIRAL [Ganin et al., 2018] show that programs executed by a graphics engine can serve as structured image representations. Recent neuro-symbolic and program-based methods further extend this idea to 3D shape structures, CAD command sequences, and architectural abstractions, using domain-specific languages or program tokens as compact output spaces [Jones et al., 2020, Wu et al., 2021, Huang et al., 2025]. Language-model-based inverse graphics methods such as IG-LLM [Kulits et al., 2024] also suggest that structured scene code can be predicted from visual embeddings. These works show the benefit of moving visual reconstruction into executable program spaces, but direct full-program generation remains brittle when the output is non-unique, low-resource, and constrained by executability. Our method instantiates this direction for procedural facade reconstruction by predicting structured hierarchical neuro-symbolic representations that can be deterministically compiled into CGA programs.

Iterative visual reasoning and symbolic repair. Recent multimodal systems increasingly use intermediate visual evidence, tool invocation, or render-and-inspect loops to improve structured prediction. ICoT [Gao et al., 2024] shows that inserting visual intermediates can strengthen finegrained visual grounding beyond text-only rationales, while agentic systems such as GeoVista [Wang et al., 2025] integrate visual tools into multi-turn reasoning loops. Most closely related to our work, VIGA [Yin et al., 2026] performs inverse graphics through a code-render-inspect loop, where rendered outputs provide visual evidence for subsequent symbolic edits. Our reflection stage adopts this principle for facade layout repair: the current Global Structure Layout Tree is rendered into a schematic layout, compared with RGB–mask evidence, and revised as CGA-compatible YAML under architectural priors.

## 3 Method

Overview and problem formulation. Given an in-the-wild building image, our goal is to recover an executable procedural facade representation for building asset generation. Since a single image provides only partial and view-dependent evidence of a building, we focus on recovering procedural facade descriptions for the dominant visible facade(s), which can then be attached to the corresponding faces of a simple mass model to instantiate an editable procedural building asset.

This formulation separates facade recovery from mass instantiation. The recoverable content from the image is the procedural structure and component configuration of visible facade regions, while the building asset is obtained by assigning the recovered facade programs to the corresponding sides of a default rectangular mass model. For unobserved sides, we use procedural completion rules or reuse recovered facade rules for visualization and downstream editing.

Following a coarse-to-fine architectural modeling strategy, we adopt a hierarchical two-level formulation. The first level recovers a facade-scale symbolic layout that captures the dominant organization of the facade and identifies representative tile regions. The second level takes these representative tiles as input and estimates tile-specific procedural parameters for component synthesis. The final executable program is obtained by deterministic assembly: the recovered structure layout is compiled into a global procedural scaffold, and the predicted tile-level codes are translated into executable component rules and inserted into the corresponding layout positions.

This decomposition is motivated by the ambiguity of direct image-to-code generation. Instead of asking a model to predict low-level procedural code in a single step, we cast the task as two more verifiable prediction problems: image-to-structure prediction and image-to-component-attributes prediction. By constraining neural prediction to structured intermediate representations and reserving executability for deterministic compilation, the framework provides a more stable bridge between image observations and procedural building modeling.

![](images/9752666f9a37e7216b6c00d0524f9f76c80e10bc40b3763a7d60b38d3e8941cb.jpg)  
Figure 1: Overview of our hierarchical neuro-symbolic framework. The method decomposes facade reconstruction into a first level for symbolic layout recovery and a second level for tile procedural synthesis. The final CGA program is obtained by deterministic assembly of the global scaffold and the compiled tile components.

![](images/58d3cfb2ca1589592cd4ed074e223e7eddc5b6ac069db9bffd4b9d36308a6d42.jpg)  
Figure 2: Overview of global structure layout recovery via visual-symbolic reflection. A finetuned Qwen-VL-7B model first induces an initial symbolic draft $Y _ { 0 }$ , including a tile set and a structure layout tree. The draft is then iteratively refined through a reflection loop driven by rendered schematics and RGB–mask overlay evidence, producing the final global structure layout output.

Global building structure reasoning. The goal of this stage is to recover a Global Structure Layout Tree T from the input image. T is a rooted 2D hierarchy over the dominant facade plane. Here, the layout refers to a facade structural arrangement of floors and openings and other repeated tiles. It uses split and repeat operators to organize the layout, with parameters such as span, spacing, and axis direction specifying the normalized positions and extents of tile regions. This representation provides the global scaffold for downstream CGA generation. Detailed schema definitions are provided in Appendix B.1.

Given an input image I, we first localize the dominant facade support, extract the most relevant facade crop I<sup>∗</sup>, and use SAM3 [Carion et al., 2025] to generate its corresponding mask M<sup>∗</sup> for downstream recovery. The global structure recovery module is instantiated with a fine-tuned Qwen-VL-7B model used in two instruction modes: layout induction and visual-symbolic reflection. The model is trained through a two-stage supervised fine-tuning procedure: initial layout induction followed by joint induction and refinement training. Detailed training data construction and refinement trajectory generation are provided in Appendix D.

In the induction mode, the model constructs an initial symbolic draft of the T using a bottom-up, tile-first strategy,

$$
Y _ { 0 } = G _ { \theta } ( { \cal I } ^ { * } , { \cal M } ^ { * } ) .
$$

The resulting draft usually captures a plausible facade topology, but its spatial parameters and local boundaries may still be coarsely aligned with the image evidence.

To refine the draft, we introduce a render-guided visual feedback and symbolic repair mechanism implemented through an iterative reflection loop. Since the same facade may admit multiple valid YAML descriptions, direct comparison in the symbolic space is unreliable. We therefore render the current symbolic draft into a schematic layout and use the facade image together with segmentationderived evidence as a weak visual reference for iterative symbolic revision. Specifically, we construct an overlay evidence map

$$
M = \Omega ( I ^ { * } , M ^ { * } ) ,
$$

which combines the facade crop with its corresponding SAM3 mask. At iteration t, the current symbolic draft $Y _ { t }$ is rendered into a schematic layout image

$$
S _ { t } = R ( Y _ { t } ) ,
$$

where $R ( \cdot )$ is a deterministic renderer and $S _ { t }$ denotes the rendered visualization of the current layout draft. Here, $t = 0$ corresponds to the initial layout draft $Y _ { 0 }$ before reflection. In the reflection mode, the same fine-tuned Qwen-VL-7B model receives $\{ M , Y _ { t } , S _ { t } \}$ and rewrites the current YAML draft into a revised symbolic program. This render-and-compare design makes the mismatch between the current layout and the visual evidence easier to identify and correct.

The reflection process is guided by soft architectural priors on structural regularity, semantic plausibility, and grammar validity, so that the recovered layout remains visually plausible and compilable. The loop terminates when sufficient agreement is reached or when a small fixed iteration budget is exhausted; in our implementation, we use $T _ { \mathrm { m a x } } = 2 .$ The final output of this stage consists of a refined $\tau$ and a representative tile set derived from its leaf-level structure. The tree serves as the facade-scale scaffold for downstream procedural synthesis, while the tile set provides the local supports for tile-level procedural parameterization.

Component-Level Inverse Parameterization. After obtaining the global hierarchical tree of the building, we recover procedural descriptions for architectural components at the tile level. Each unique component is represented as a unified YAML-based set of procedural attributes. Instead of predicting raw CGA grammar, the model infers a constrained component-level parameterization defined by explicit architectural attributes (e.g., arch presence, row/column counts, and structural elements), which are sufficient for component synthesis. Detailed schema definitions are provided in Appendix B.2. This representation serves as a spatial anchor for prediction, reducing ambiguity in component position and scale, while remaining CGA-compatible and casting under-specified local geometry as a structured parameterization problem.

Given $\tau ,$ we treat its leaf nodes as structural placeholders. The leaf-level layout provides a facadescale estimate of component positions, while SAM3-derived local masks provide pixel-level evidence about component extents. By combining these two cues, we obtain representative tile supports for component parameterization.

Each grounded tile crop is encoded by a dual-stream visual backbone: a CLIP stream provides high-level semantic and stylistic cues, while a DINOv3 stream captures localized structural and geometric details. The fused visual features are projected into a shared decoder space and combined with language tokens for autoregressive generation of the component parameterization. A detailed architecture visualization is provided in Appendix D.1.

Deterministic procedural assembly and instantiation. We adopt a CGA-style procedural construction paradigm following classical shape-grammar-based building modeling [Wonka et al., 2003, Müller et al., 2006, 2007], where production rules progressively refine building scopes from coarse facade regions to repeated tiles and local architectural details. In our implementation, the customdesigned compiler translates the global structure layout tree into a CGA-compatible facade scaffold, and converts each component parameterization into a reusable component rule.

After compilation, the recovered facade program can be attached to a building mass model to instantiate an editable 3D building asset in a procedural modeling engine. When multiple facades are visible in the input image, each recovered facade program can be assigned to the corresponding side of the mass model. For unobserved sides, we use simple procedural completion rules or reuse the recovered facade grammar for visualization and downstream editing. A simplified CGA-style pseudo-code example of this deterministic compilation and instantiation process is provided in Appendix B.3.

![](images/0d6e9d83a49174cd12f8c9524d46dcb6106c6636d0f56c839c71d5384bee1abc.jpg)  
Figure 3: End-to-end procedural reconstruction examples. Each row shows two examples, each showing the input facade image, recovered layout, and final instantiated building asset. For their detailed outputs, please refer to Fig. 7 and 8 in Appendix A.

## 4 Experiments

## 4.1 Experimental Results

End-to-end Result Showcase. We first present representative end-to-end reconstruction results of the full framework. Given an in-the-wild building image, our method recovers a global structure layout scaffold and synthesizes tile-level procedural components, which are then deterministically assembled into an executable CGA program. The resulting representation can be rendered as a reconstructed procedural facade and further instantiated as a simple editable 3D building asset.

Fig. 3 shows representative examples, including the input image, facade structure layout, and final procedural reconstruction. These results illustrate the overall goal of our framework: recovering not only a visually aligned facade layout, but also an executable procedural representation with reusable local architectural components. Additional end-to-end results and local details are provided in Appendix A.

Effectiveness of Global Structure Reasoning. We evaluate whether the proposed global structure reasoning can recover a meaningful facade-scale structural scaffold from RGB images. The evaluation covers both regular facades, where layout structures are relatively clear, and challenging in-the-wild facades, where occlusion, perspective distortion, and incomplete evidence make structural recovery more difficult.

We further evaluate global structure recovery on challenging examples from eTRIMS and real-world imagery. These images include perspective distortion, partial occlusion, nonuniform spacing, and incomplete visual evidence. Since accurate procedural ground-truth layouts are generally unavailable for such cases, we focus on whether the recovered layouts preserve the dominant facade organization and remain procedurally meaningful. Additional challenging cases are provided in Fig. 4. Extended reflection-iteration results are reported in Table 6 of Appendix C.

Effectiveness of Component Parameterization. We evaluate component parameterization using attribute-level accuracy on real-world tile crops from CMP Tylecek [2013] and eTRIMS Korˇ c andˇ Förstner [2009], as well as procedurally rendered synthetic tiles. Real-world evaluation uses 107 held-out crops, with training and test partitions separated by building identity. Data construction is described in Appendix D, with the real-tile split and leakage checks detailed in Appendix D.2.

![](images/52b8d706a889dba897eaff1dafd540a3cec71bc45dabaa9c7d04a6977745669a.jpg)  
Figure 4: Challenging facade reconstruction examples. Representative examples from eTRIMS and in-the-wild images with corresponding recovered global structure layouts.

![](images/a0d8466bb86636d2f544a2b16ac81b56da2b32cbf1568b58fafee848af89ccf1.jpg)

![](images/4f70f2eb7b3544c948ffd11be3c535c5d1178f8da2105c14924468610b9d75a9.jpg)  
(b) component structure editing  
Figure 5: Controllability and editability of the recovered procedural building representation. Top: editing the recovered facade layout leads to a corresponding change in the instantiated 3D building result, demonstrating global structural controllability. Bottom: editing a representative reconstructed tile produces updated repeated components in the final 3D building result, demonstrating local component-level editability. Together, these examples show that the recovered procedural representation supports both facade-scale structural editing and local repeated-component editing.

For geometric attributes, we report exact-match accuracy over discrete presets of normalized values. Attribute discretization and metric definitions, together with continuous-error evaluation on synthetic test tiles, are detailed in Appendix E.2. A separate comparison of synthetic-only and synthetic-plusreal training on the same held-out real test set is reported in Appendix E.3.

Table 1 shows that the model achieves strong attribute-level accuracy on both synthetic and real-world tile sets, with a moderate gap between the two settings. W and E denote window-like and entrance-like tiles, respectively. These results indicate that the component parameterization predictor can recover key local procedural attributes from real image evidence while benefiting from the controllability and coverage of synthetic supervision. Detailed per-attribute results are provided in Appendix E.

Table 1: Accuracy of representative component (tile) attributes on real-world and synthetic datasets. W and E denote window-like and entrance-like components. Full per-attribute results are provided in Appendix E.
<table><tr><td>Data</td><td>W-Arch↑</td><td>W-Sill↑</td><td>W-Row/Col↑</td><td></td><td>E-Stair↑ E-Handle↑</td></tr><tr><td>Real-world</td><td>0.98</td><td>0.92</td><td>0.89</td><td>0.95</td><td>0.96</td></tr><tr><td>Synthetic</td><td>0.99</td><td>0.96</td><td>0.92</td><td>0.98</td><td>0.97</td></tr></table>

![](images/7092298c2aa12578de651cfca7730a83cb418ea88d244423c429838f5b62e30b.jpg)  
Figure 6: Representative comparison with FaçAID. From left to right are the input facade image, the layout result reported by FaçAID, and our recovered global facade structure.

Table 2: User preference study comparing FaçAID and our method in terms of facade reconstruction fidelity. Participants selected their preferred result, and we report the percentage of votes. ’Tie’ indicates the proportion of cases where both results were judged to be of comparable quality.
<table><tr><td>Methods:</td><td>FaçAID</td><td>Ours</td><td>Tie</td></tr><tr><td>Vote Proportion:</td><td>30%</td><td>59%</td><td>11%</td></tr></table>

Bonus Feature: Editability. One major advantage of our method lies in its ability to produce 3D results with separable structures, enabling seamless integration into game engines and facilitating user-driven post-editing. Fig. 5 shows two examples of user edits on our reconstructions at both the facade and tile levels. The modification on the global facade layout enables changes to the overall structure of the instantiated 3D building (see Fig. 5(a)). In contrast, editing the parameters of local tiles enables updates to all corresponding repeated components in the final building (see Fig. 5(b)). This demonstrates that our output is not merely a static mesh or image, but an executable and editable procedural representation that supports both global structural control and local component-level manipulation.

## 4.2 Comparison

We conduct comparison studies at two levels: global structure recovery and Component Parameterization.

For global facade-structure recovery, we compare our recovered global facade structures with representative examples reported in FaçAID. Since FaçAID assumes segmented facade input, while our method starts from RGB facade images, and since no public implementation or unified executable evaluation protocol is available, we treat this comparison as a qualitative layout-level reference rather than a direct benchmark under identical inputs. Fig. 6 shows a representative comparison, and additional visual comparisons are provided in Appendix F.1.

To further complement this qualitative comparison, we conduct a small user preference study. Participants are shown the input facade image and two anonymized layout results, one from FaçAID and one from our method, and are asked which result better matches the facade structure. Table 2 reports the percentage of preferences for FaçAID, ours, and tie, where tie indicates that the two results are judged comparable or hard to distinguish. The study involved 15 participants and 180 randomized blinded comparisons. Participant information, the evaluation protocol, full instructions, vote counts, and exploratory statistical results are provided in Appendix G.

Next, we evaluate the role of the Component Parameterization by comparing three component-level generation strategies: direct low-level code generation, direct component parameterization prediction, and our fine-tuned predictor. Execution success is reported for all methods, while attribute-level accuracy is reported only for methods that explicitly predict structured tile attributes. Representative qualitative examples are provided in Appendix F.2.

Table 3 reports execution success and representative attribute accuracy on objective tile-level properties. Compared with the stronger Gemini 3.1 Pro Preview baseline, direct middle-code generation substantially improves executability over direct code generation, indicating that structured intermediate representations reduce the ambiguity of procedural program generation. Our task-specific predictor further improves execution reliability and fine-grained architectural attribute recovery.

Table 3: Comparison of tile-level procedural generation strategies. Attribute accuracy is reported only for methods that explicitly predict tile-level structured attributes.
<table><tr><td>Method</td><td>Exec. Succ.↑</td><td>Arch Acc.↑</td><td>Row/Col Acc.↑</td><td>Sill Acc.↑</td><td>Stair Acc.↑</td><td>Handle Acc.↑</td></tr><tr><td>Direct Code (Gemini 3.1 Pro Preview)</td><td>0.68</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Direct Middle Code (Gemini 3.1 Pro Preview)</td><td>0.95</td><td>0.94</td><td>0.88</td><td>0.88</td><td>0.96</td><td>0.91</td></tr><tr><td>Ours</td><td>0.98</td><td>0.99</td><td>0.90</td><td>0.93</td><td>0.96</td><td>0.96</td></tr></table>

Table 4: Comparison with direct end-to-end LLM/VLM generation.
<table><tr><td>Method</td><td>Exec. Succ.↑</td><td>TRC↓</td></tr><tr><td>Direct LLM (Gemini 3.1 Pro Preview)</td><td>0.94</td><td>1.83</td></tr><tr><td>Ours</td><td>0.98</td><td>1.47</td></tr></table>

Lastly, we compare our full framework with a direct generation baseline, where a general LLM/VLM is prompted to generate the entire CGA program from the input facade image in a single pass. We evaluate this comparison from both the layout and executable-code levels using Tree Repair Cost (TRC) and execution success rate. For the direct generation baseline, the layout tree used for TRC evaluation is recovered from the generated CGA program. TRC is computed on the layout structure and measures the expert-validated structural repair cost under a fixed editing protocol. Specifically, annotators edit the predicted layout tree through predefined atomic operations until its rendered schematic expresses the dominant facade organization. Each insertion, deletion, or modification of a structural node counts as one edit. Therefore, TRC reflects the amount of manual structural correction required by a prediction, rather than a global optimum over all possible tree representations. The detailed annotation interface and repair protocol are provided in Appendix H.

Table 4 reports execution success and symbolic repair cost, and representative qualitative examples are provided in Appendix F.2.

## 4.3 Ablation

We first evaluate the effect of visual-symbolic reflection on global facade-structure recovery. Starting from the initial symbolic draft, the model iteratively revises the layout using rendered feedback and segmentation-derived visual evidence. In addition to TRC and execution success rate, we report count-based structural accuracy: floor accuracy is computed at the facade level, while window and door count accuracy are computed at the floor level by comparing the predicted and target element counts.

Table 5: Ablation of visual-symbolic reflection on structure layout recovery.
<table><tr><td>Model</td><td>Win Cnt. Acc.↑</td><td>Floor Cnt. Acc.↑</td><td>Door Cnt. Acc.↑</td><td>TRC↓</td><td>Exec. Succ.↑</td></tr><tr><td>Initial Layout</td><td>0.61</td><td>0.83</td><td>0.69</td><td>4.32</td><td>0.98</td></tr><tr><td>Iteration 1</td><td>0.77</td><td>0.85</td><td>0.78</td><td>2.01</td><td>0.98</td></tr><tr><td>Iteration 2</td><td>0.80</td><td>0.86</td><td>0.81</td><td>1.47</td><td>0.99</td></tr></table>

Table 5 reports the main reflection results up to the default setting $T _ { \mathrm { m a x } } = 2 ,$ , showing that reflection improves count-level structural consistency and reduces TRC while maintaining high global structure recovery compilability. Most of the improvement is obtained within the first two iterations, after which the gains become marginal. We therefore use $T _ { \mathrm { m a x } } = 2$ in the main experiments. Additional qualitative examples and an extended analysis with more reflection iterations are provided in Appendix C.

We further ablate the visual stream by comparing CLIP-only, DINOv3-only, and CLIP+DINOv3 encoders using grouped semantic-presence and structural attributes for window-like and entrance-like tiles. The combined encoder achieves the best overall structural accuracy, with full grouped results provided in Appendix E.1.

## 4.4 Limitations and Discussion

Our framework targets approximately planar visible facades, while unseen geometry is completed for asset instantiation rather than reconstructed from image evidence. The current component schema supports simple parametric window arches, but not entrance arches or complex decorative assemblies. Severe perspective distortion and occlusion may still affect layout recovery.

Our experiments cover the considered facade sources and include building-disjoint real-tile evaluation, but controlled city-disjoint and style-disjoint evaluation remains future work. More expressive massing and component representations, together with additional geometric evidence, could extend the framework to more complex architecture.

## 5 Conclusion

We presented a hierarchical neuro-symbolic framework for visual-to-executable procedural reconstruction of buildings from in-the-wild images. The key idea is to bridge visual facade understanding and procedural generation through structured intermediate representations: a global structure layout tree captures facade-scale organization, while component parameterizations describe local architectural elements. Together with render-guided visual-symbolic reflection and deterministic CGA compilation, this design avoids direct low-level code generation and produces editable, executable procedural facade programs.

Experiments show that the proposed representation-aligned formulation recovers meaningful facade structures on both regular and challenging in-the-wild images, improves procedural executability over direct LLM/VLM generation baselines, and preserves fine-grained component details through tile-level parameter prediction. These results suggest that structured intermediate representations provide a practical path toward robust image-based procedural building reconstruction.

## Acknowledgments

This work was supported in part by the Guangdong Major Project of Basic and Applied Basic Research (Grant No. 2023B0303000016) and the Shenzhen Science and Technology Program (Grant No. CJGJZD20240729141906008).

## References

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

Google DeepMind. Gemini 3.1 pro: A smarter model for your most complex tasks. https: //deepmind.google/models/model-cards/gemini-3-1-pro/, 2026. Accessed: 2026-02.

Yaroslav Ganin, Tejas Kulkarni, Igor Babuschkin, S. M. Ali Eslami, and Oriol Vinyals. Synthesizing programs for images using reinforced adversarial learning. In Proceedings of the International Conference on Machine Learning, pages 1666–1675, 2018.

Jun Gao, Yongqi Li, Ziqiang Cao, and Wenjie Li. Interleaved-modal chain-of-thought. arXiv preprint arXiv:2411.19488, 2024.

Qirui Huang, Runze Zhang, Kangjun Liu, Minglun Gong, Hao Zhang, and Hui Huang. Arcpro: Archi tectural programs for structured 3d abstraction of sparse points. arXiv preprint arXiv:2503.02745, 2025.

R. Kenny Jones, Theresa Barton, Xianghao Xu, Kai Wang, Ellen Jiang, Paul Guerrero, Niloy J. Mitra, and Daniel Ritchie. Shapeassembly: Learning to generate programs for 3d shape structure synthesis. ACM Transactions on Graphics, 39(6), 2020. doi: 10.1145/3414685.3417812.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):139:1–139:14, 2023.

František Korc and Wolfgang Förstner. eTRIMS image database for interpreting images of man-madeˇ scenes. Technical Report TR-IGG-P-2009-01, Department of Photogrammetry, University of Bonn, April 2009. URL https://www.ipb.uni-bonn.de/projects/etrims\_db/.

Panagiotis Koutsourakis, Loïc Simon, Olivier Teboul, Georgios Tziritas, and Nikos Paragios. Single view reconstruction using shape grammars for urban environments. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 1795–1802, 2009. doi: 10.1109/ICCV.2009. 5459400.

Peter Kulits, Haiwen Feng, Weiyang Liu, Victoria Abrevaya, and Michael J. Black. Re-thinking inverse graphics with large language models. arXiv preprint arXiv:2404.15228, 2024.

Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications ofthe ACM, 65(1):99–106, 2021.

Pascal Müller, Peter Wonka, Simon Haegler, Andreas Ulmer, and Luc Van Gool. Procedural modeling of buildings. ACM Transactions on Graphics, 25(3):614–623, 2006. doi: 10.1145/ 1141911.1141931.

Pascal Müller, Gang Zeng, Peter Wonka, and Luc Van Gool. Image-based procedural modeling of facades. ACM Transactions on Graphics, 26(3):85, 2007. doi: 10.1145/1275808.1276484.

Gen Nishida, Adrien Bousseau, and Daniel G. Aliaga. Procedural modeling of a building from a single image. Computer Graphics Forum, 37(2):415–429, 2018. doi: 10.1111/cgf.13372.

Aleksander Plocharski, Jan Swidzinski, Joanna Porter-Sobieraj, and Przemyslaw Musialski. Façaid: A transformer model for neuro-symbolic facade reconstruction. In SIGGRAPH Asia 2024 Conference Papers. ACM, 2024. doi: 10.1145/3680528.3687657.

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In International Conference on Learning Representations, 2023.

Olivier Teboul, Iasonas Kokkinos, Loïc Simon, Panagiotis Koutsourakis, and Nikos Paragios. Shape grammar parsing via reinforcement learning. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 2273–2280, 2011. doi: 10.1109/CVPR.2011.5995319.

Radim Tylecek. The cmp facade database. Technical report, Center for Machine Perception, Czechˇ Technical University in Prague, 2013. URL https://cmp.felk.cvut.cz/\~tylecr1/facade/.

Yikun Wang, Zuyan Liu, Ziyi Wang, Han Hu, Pengfei Liu, and Yongming Rao. Geovista: Webaugmented agentic visual reasoning for geolocalization. arXiv preprint arXiv:2511.15705, 2025.

Peter Wonka, Michael Wimmer, François X. Sillion, and William Ribarsky. Instant architecture. ACM Transactions on Graphics, 22(3):669–677, 2003. doi: 10.1145/882262.882324.

Fuzhang Wu, Dong-Ming Yan, Weiming Dong, Xiaopeng Zhang, and Peter Wonka. Inverse procedural modeling of facade layouts. ACM Transactions on Graphics, 33(4):121:1–121:10, 2014. doi: 10.1145/2601097.2601162.

Rundi Wu, Chang Xiao, and Changxi Zheng. Deepcad: A deep generative network for computeraided design models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6772–6782, 2021.

Shaofeng Yin, Jiaxin Ge, Zora Zhiruo Wang, Chenyang Wang, Xiuyu Li, Michael J. Black, Trevor Darrell, Angjoo Kanazawa, and Haiwen Feng. Vision-as-inverse-graphics agent via interleaved multimodal reasoning. arXiv preprint arXiv:2601.11109, 2026.

## A More Reconstruction Results

Fig. 7 and 8 provide additional end-to-end reconstruction results. Each example shows the input building facade image together with the final procedural reconstruction produced by our framework. These results complement the main-paper examples and further demonstrate the ability of the proposed method to combine global structure layout recovery and component-level procedural parameterization into a unified executable facade representation.

## B Detailed Representation Schemas

This section provides additional details of the hierarchical two-level representation used in our framework. Fig. 9 illustrates the global structure layout schema for split-and-repeat facade organization and the component-level parameterization schemas for representative window and entrance components.

## B.1 Global structure layout schema.

The global structure level representation describes the dominant facade plane as a hierarchical splitand-repeat layout. Each region is represented by a symbolic node with a normalized spatial domain and a node type, such as split, repeat, or leaf. Split nodes partition a parent region into ordered child regions, while repeat nodes describe repeated tile instances along a selected axis. Leaf nodes serve as placeholders for representative tiles, which are later grounded and replaced by compiled component rules.

## B.2 Component parameterization schema.

The component representation abstracts each unique tile into a structured set of procedural attributes rather than raw CGA code. For window-like tiles, the schema includes local grid structure, frame configuration, arch and sill attributes, depth-related properties, and appearance parameters. For entrance-like tiles, the schema includes door structure, frame configuration, stairs, hardware, and appearance attributes. These attributes are designed to be directly translatable into CGA component rules by the deterministic compiler.

## B.3 CGA-style Procedural Compilation Example

Listing 1 gives a simplified CGA-style pseudo-code example of our deterministic compilation process. The facade is first subdivided along the vertical axis into floor-level regions, repeated floors are expanded into horizontal tile slots, and each leaf tile invokes a parameterized component generator such as Window or Entrance. The compiled component rules are then scaled to their corresponding leaf scopes and assembled on the facade scaffold. Finally, the assembled facade is attached to a simple building mass model to produce an editable and executable procedural 3D building.

Listing 1: A simplified CGA-style pseudo-code example illustrating how global layout and component parameters are compiled into procedural facade rules.

0 : b u i l d i n g −> comp ( f ) {   
f r o n t : fa c a d e   
| back : Wall   
| l e f t : fa c a d e   
| r i g h t : f a c a d e   
}   
1 : fa c a d e −> Subdiv (Y, ground\_h , 1 r , top\_h ) {   
g r o u n d \_ f l o o r | R e p e a t ( Y , f l o o r \_ h ) { t y p i c a l \_ f l o o r } | t o p   
}   
2 : t y p i c a l \_ f l o o r −> R e p e a t ( X , t i l e \_ w ) {   
window\_slot   
}

![](images/71cb37eb7db607980bc860040db4557b159b5c19cbe7ffed637f66cc242341a7.jpg)  
Figure 7: Additional end-to-end procedural reconstruction results - 1. Each example shows the input building image and the final procedural reconstruction generated by our full framework.

![](images/a9f20061bcd19ca02a624aa2281329cd55d2efd3c5efa2fd3dc905d977f79c6e.jpg)  
Figure 8: Additional end-to-end procedural reconstruction results - 2. Each example shows the input building image and the final procedural reconstruction generated by our full framework.

![](images/f4d36633d9b53a39b258cf3a62871ad1f2dad94223f50f83a0860e1673e8c768.jpg)  
Figure 9: Detailed dual-tier symbolic representation. The left panel shows the grammar used to describe facade-scale organization. The right panels show representative component parameterization for window-like and entrance-like components.

Table 6: Effect of increasing the number of visual-symbolic reflection iterations.
<table><tr><td>Model</td><td>Acc (Win)↑</td><td>Acc (Floor)↑</td><td>Acc (Door)↑</td><td>TRC↓</td><td>Exec. Succ.↑</td></tr><tr><td>Initial Layout</td><td>0.61</td><td>0.83</td><td>0.69</td><td>4.32</td><td>0.98</td></tr><tr><td>Iteration 1</td><td>0.77</td><td>0.85</td><td>0.78</td><td>2.01</td><td>0.98</td></tr><tr><td>Iteration 2</td><td>0.80</td><td>0.86</td><td>0.81</td><td>1.47</td><td>0.99</td></tr><tr><td>Iteration 3</td><td>0.80</td><td>0.87</td><td>0.80</td><td>1.46</td><td>0.99</td></tr><tr><td>Iteration 4</td><td>0.78</td><td>0.87</td><td>0.79</td><td>1.48</td><td>0.99</td></tr></table>

3 : g r o u n d \_ fl o o r −> Subdiv (X, 1 r , door\_w , 1 r ) {   
w a l l | e n t r a n c e \_ s l o t | w a l l   
}   
4 : window\_slot −> Window (   
rows =2 , c o l s =3 , fra m e = t r u e , a r c h = fa l s e , s i l l = t r u e   
)   
5 : e n t r a n c e \_ s l o t −> E n t r a n c e (   
d o o r \_ t y p e = double , s t a i r s = t r u e , h a n d l e = t r u e   
)

## C Additional Visual-Symbolic Reflection Examples

We further analyze the effect of increasing the number of visual-symbolic reflection iterations. This study examines whether more reflection steps consistently improve the recovered global structure layout. As shown in Table 6, most of the improvement is obtained within the first two iterations: count-level structural accuracy improves substantially and TRC decreases sharply compared with the initial layout. Further iterations bring only marginal changes and may introduce small fluctuations in local element counts. Therefore, we use a small fixed reflection budget in the main experiments.

Fig. 10 provides qualitative examples of the reflection process. The examples illustrate how the initial symbolic draft is iteratively refined through rendered layout feedback and RGB–mask overlay evidence.

![](images/7fb5afde4adf2f928f3dcc96c888eb82e0cc1ab50c4f4582792736bf66e98964.jpg)  
Figure 10: Updated examples under the process of visual-symbolic reflection loop. The examples show how the initial global structure layout draft is refined over reflection iterations using rendered schematic feedback and RGB–mask overlay evidence.

## D Training and Implementation Details

Qwen-VL-7B fine-tuning. We fine-tune a Qwen-VL-7B model for global structure layout recovery using both real and synthetic facade data. Each training example pairs a facade image and segmentation-derived evidence with a normalized orthogonal layout tree represented in YAML. For real data, we use in-the-wild facade crops without explicit perspective rectification and manually annotate their layouts. For synthetic data, we augment the training inputs by rendering procedurally generated facades from different viewpoints, pairing each rendered image with its corresponding layout derived from the procedural rules. These pairs provide supervision for mapping view-dependent image evidence to canonical facade structures.

To account for the non-uniqueness of procedural layout representations, we annotate each real facade with at least two valid layout strategies when possible. These alternative YAML layouts may differ in grouping, split order, or repeat abstraction, while still describing the same dominant facade organization. This design avoids forcing the model to imitate a single arbitrary tree representation and better reflects the multi-solution nature of procedural facade modeling. The fine-tuned Qwen-VL-7B model is used in two instruction modes during inference: initial layout induction and visual-symbolic reflection.

Two-stage supervised fine-tuning. We first fine-tune the model on layout-induction examples to establish the image-to-layout mapping. Starting from this checkpoint, we jointly fine-tune it on induction and refinement examples distinguished by task-specific instructions. The resulting checkpoint is shared by the two instruction modes at inference time.

Visual-symbolic refinement training. To train the refinement capability, we further construct supervised repair trajectories based on valid facade layouts. Given an accepted layout representation Y , we generate intermediate corrupted states by applying schema-valid perturbations at different levels.

![](images/d205dddd50ac6d6f8754dbaae7bd99131179a32aee331a07ee55e31626ec3b2e.jpg)  
Figure 11: Our component parameterization architecture. Representative tile crops are encoded by a dual-stream visual backbone and decoded into tile-specific YAML middle code, which is then deterministically compiled into executable CGA snippets.

Starting from an accepted layout $Y _ { \mathrm { v a l i d } }$ , we first perturb normalized spans and spacing to obtain $Y _ { \mathrm { f i n e } } ,$ and then perturb topology and discrete counts to obtain $Y _ { \mathrm { c o a r s e } } .$ . Thus, data corruption follows $Y _ { \mathrm { v a l i d } }  Y _ { \mathrm { f i n e } }  Y _ { \mathrm { c o a r s e } }$ , whereas supervised repair follows the reverse direction:

$$
Y _ { \mathrm { c o a r s e } }  Y _ { \mathrm { f i n e } }  Y _ { \mathrm { v a l i d } } .\tag{1}
$$

During refinement training, we supervise the VLM on successive steps of the coarse-to-fine repair trajectory. Each example uses the facade image, segmentation-derived evidence, the current YAML layout, and a schematic rendered from that layout as input, with the next repaired layout as the prediction target.

For synthetic data, refinement trajectories are automatically generated using schema-valid and renderable perturbations. For real facade images, we manually construct repair trajectories by first correcting structural/count errors and then refining spatial parameters. When multiple valid decompositions exist, alternative executable layouts are also included as valid examples rather than forcing the model toward a single tree serialization.

Training configuration. We fine-tune the layout model using LoRA and AdamW, with a learning rate of $1 0 ^ { - 5 }$ , weight decay of 0.1, and $\beta _ { 2 } = 0 . 9 5$ . Training uses BF16 precision, a per-device batch size of 2, 8 gradient-accumulation steps, a maximum sequence length of 2048, and 5 epochs per training run. We use a LoRA rank of 64, a scaling parameter of 16, and dropout of 0.05. Gradient checkpointing and DeepSpeed ZeRO-2 are enabled.

Tile-level predictor training data. We construct synthetic and real tile datasets for component parameterization. Each sample pairs a component crop with structured procedural attributes under the schema in Appendix B.2. Data construction and the building-disjoint real-tile split are detailed in Appendix D.2.

## D.1 Component Parameterization Architecture

Fig. 11 shows the detailed architecture of the component parameterization predictor. Representative tile crops are encoded by the dual-stream CLIP and DINOv3 backbone, fused into a shared decoder space, and decoded into component parameterization for deterministic procedural compilation.

## D.2 Tile Dataset Construction and Splits

Synthetic tile construction. Synthetic tiles are rendered using parameterized procedural generators for common window-like and entrance-like components. The generators vary grid structure, frames, window arches, sills, stairs, hardware, and appearance attributes. Each rendered image is paired with the corresponding component-parameterization annotation derived from the generator parameters. The original continuous CGA parameters are also available for evaluating geometric attributes on synthetic tiles. Both synthetic and real annotations follow the schema in Appendix B.2.

Real tile collection and annotation. Real window and entrance crops are collected from the CMP Facade Database Tylecek [2013] and eTRIMS Korˇ c and Förstner [2009] and manually annotated withˇ structured component attributes. Before extracting crops, we group all images of the same building, including multiple viewpoints, repeated captures, and overlapping images. We then segment window and entrance instances from the images in each building group and extract their component crops.

Redundancy removal and building-level partitioning. Within each building group, we extract DINOv2 features for the component crops and remove highly similar crops using a cosine-similarity threshold of 0.85. This removes redundant instances, such as nearly identical windows from the same facade or duplicate observations of the same opening. DINOv2 is used here for dataset redundancy checks, whereas the component predictor uses CLIP and DINOv3.

The train/test partition is constructed at the building-group level. Building groups are randomly assigned until the retained crop counts are approximately balanced between the two partitions. All viewpoints, facades, and component crops from one building remain in the same partition. Highly similar cross-split pairs are additionally inspected to remove duplicated image content or samebuilding overlap. Table 7 summarizes the resulting 207 crops from 67 building groups: 100 training crops and 107 test crops.

Table 7: Building-disjoint partition of the real tile dataset collected from CMP and eTRIMS. The 207 crops comprise both training and test data.
<table><tr><td>Split</td><td>Tiles</td><td>Building groups</td><td>Avg. tiles/group</td></tr><tr><td>Training</td><td>100</td><td>31</td><td>3.23</td></tr><tr><td>Test</td><td>107</td><td>36</td><td>2.97</td></tr><tr><td>Total</td><td>207</td><td>67</td><td>3.09</td></tr></table>

## E Additional Tile-Level Quantitative Results

This section provides additional quantitative results for the tile-level procedural synthesis module, including full per-attribute results, visual encoder ablations, metric definitions, and the synthetic-only versus synthetic-plus-real training comparison.

Mixed-domain model evaluated on synthetic and real tiles. Tables 8 and 9 evaluate the mixeddomain-trained component predictor on synthetic and real-world test tiles. They characterize the difference between the two evaluation domains. The separate training-data comparison is reported in Appendix E.3.

## E.1 Full visual encoder ablation.

Table 10 and Table 11 report the full per-attribute accuracy of different visual encoders. We compare CLIP-only, DINOv3-only, and the combined CLIP+DINOv3 encoder. CLIP-only performs well on semantic presence attributes, while DINOv3 improves several structural attributes. The combined encoder achieves the most stable overall performance.

## E.2 Attribute Discretization and Evaluation

Normalized geometric presets. Geometric dimensions are normalized by the corresponding tile width or height and represented as attribute-specific preset categories. The recorded preprocessing

Table 8: Full attribute-level accuracy on window-like tiles.
<table><tr><td>Group</td><td>Attribute</td><td>Synthetic Acc.↑</td><td>Real-world Acc.↑</td><td>Gap↓</td></tr><tr><td rowspan="2">Arch structure</td><td>Arch style</td><td>0.96</td><td>0.93</td><td>0.03</td></tr><tr><td>Arch height class</td><td>0.92</td><td>0.87</td><td>0.05</td></tr><tr><td rowspan="3">Categorical</td><td>Arch</td><td>0.99</td><td>0.98</td><td>0.01</td></tr><tr><td>Top sill</td><td>0.96</td><td>0.92</td><td>0.04</td></tr><tr><td>Bottom sill</td><td>0.95</td><td>0.93</td><td>0.02</td></tr><tr><td rowspan="8">Frame attributes</td><td>Outer frame thickness</td><td>0.89</td><td>0.86</td><td>0.03</td></tr><tr><td>Transom width</td><td>0.91</td><td>0.91</td><td>0.00</td></tr><tr><td>Mullion width</td><td>0.90</td><td>0.88</td><td>0.02</td></tr><tr><td>Row</td><td>0.94</td><td>0.95</td><td>-0.01</td></tr><tr><td>Row ratio</td><td>0.91</td><td>0.87</td><td>0.04</td></tr><tr><td>Columns per row</td><td>0.89</td><td>0.84</td><td>0.05</td></tr><tr><td>Column ratio per row</td><td>0.85</td><td>0.82</td><td>0.03</td></tr><tr><td>Top sill height</td><td>0.93</td><td></td><td></td></tr><tr><td rowspan="4">Sill structure</td><td></td><td></td><td>0.88</td><td>0.05</td></tr><tr><td>Top sill protrusion</td><td>0.95</td><td>0.91</td><td>0.04</td></tr><tr><td>Bottom sill height</td><td>0.94</td><td>0.90</td><td>0.04</td></tr><tr><td>Bottom sill protrusion</td><td>0.95</td><td>0.87</td><td>0.08</td></tr></table>

Table 9: Full attribute-level accuracy on entrance-like tiles.
<table><tr><td>Group</td><td>Attribute</td><td>Synthetic Acc.↑</td><td>Real-world Acc.↑</td><td>Gap↓</td></tr><tr><td rowspan="4">Entrance frame</td><td>Frame style</td><td>0.97</td><td>0.93</td><td>0.04</td></tr><tr><td>Frame head height</td><td>0.92</td><td>0.89</td><td>0.03</td></tr><tr><td>Frame side width</td><td>0.93</td><td>0.90</td><td>0.03</td></tr><tr><td>Door type</td><td>0.97</td><td>0.92</td><td>0.05</td></tr><tr><td rowspan="5">Entrance structure</td><td>Door asymmetric ratio</td><td>0.91</td><td>0.86</td><td>0.05</td></tr><tr><td>Door style</td><td>0.88</td><td>0.88</td><td>0.00</td></tr><tr><td>Grid rows</td><td>0.87</td><td>0.85</td><td>0.02</td></tr><tr><td>Grid columns</td><td>0.85</td><td>0.87</td><td>-0.02</td></tr><tr><td>Has hardware</td><td>0.98</td><td></td><td></td></tr><tr><td rowspan="3">Hardware</td><td>Hardware type</td><td>0.97</td><td>0.96 0.96</td><td>0.02 0.01</td></tr><tr><td>Hardware side</td><td>0.92</td><td>0.87</td><td>0.05</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Stairs</td><td>Has stairs</td><td>0.98</td><td>0.94</td><td>0.04</td></tr><tr><td>Stair height</td><td>0.88</td><td>0.86</td><td>0.02</td></tr><tr><td>Stair count</td><td>0.90</td><td>0.86</td><td>0.04</td></tr></table>

procedure uses training-set quantiles to determine category boundaries. Attribute accuracy is evaluated by exact match against the annotated preset labels, while synthetic continuous-error evaluation compares preset-decoded values with the original CGA parameters.

Exact-match accuracy. Let $\nu _ { j }$ denote the evaluation tiles with valid ground-truth labels for attribute j. Its exact-match accuracy is

$$
\operatorname { A c c } _ { j } = \frac { 1 } { | \mathcal { V } _ { j } | } \sum _ { i \in \mathcal { V } _ { j } } \mathbf { 1 } [ \hat { c } _ { i j } = c _ { i j } ] ,\tag{2}
$$

where $c _ { i j }$ and $\hat { c } _ { i j }$ are the annotated and predicted attribute values or preset categories. The same exact-match definition applies to categorical presence/type attributes and discrete counts. Presence attributes, such as Has stairs, are evaluated on all relevant tiles with valid labels, whereas dependent attributes, such as Stair height, are evaluated only where the corresponding structure is present in the ground truth and their labels are valid.

Table 10: Full per-attribute encoder ablation on window-like tiles.
<table><tr><td>Model</td><td>Has Arch</td><td>Arch Style</td><td>Arch Height</td><td>Has Sill</td><td>Sill Height</td><td>Sill Protrusion</td><td>Frame Thick.</td><td>Grid Num.</td><td>Grid Ratio</td></tr><tr><td>CLIP-only</td><td>0.99</td><td>0.90</td><td>0.81</td><td>0.91</td><td>0.72</td><td>0.68</td><td>0.65</td><td>0.75</td><td>0.72</td></tr><tr><td>DINOv3-only</td><td>0.98</td><td>0.93</td><td>0.88</td><td>0.90</td><td>0.89</td><td>0.89</td><td>0.86</td><td>0.88</td><td>0.85</td></tr><tr><td>CLIP+DINOv3</td><td>0.99</td><td>0.94</td><td>0.90</td><td>0.93</td><td>0.92</td><td>0.92</td><td>0.88</td><td>0.90</td><td>0.87</td></tr></table>

Table 11: Full per-attribute encoder ablation on entrance-like tiles.
<table><tr><td>Model</td><td>Has Stairs</td><td>Stair Height</td><td>Stair Count</td><td>Has Hardware</td><td>Hardware Type</td><td>Hardware Side</td><td>Frame Style</td><td>Frame Size</td><td>Door Type</td><td>Door Asym. Ratio</td><td>Door Style</td><td>Grid</td></tr><tr><td>CLIP-only</td><td>0.95</td><td>0.74</td><td>0.71</td><td>0.95</td><td>0.94</td><td>0.80</td><td>0.93</td><td>0.82</td><td>0.86</td><td>0.78</td><td>0.86</td><td>0.77</td></tr><tr><td>DINOv3-only</td><td>0.93</td><td>0.83</td><td>0.85</td><td>0.93</td><td>0.91</td><td>0.84</td><td>0.89</td><td>0.88</td><td>0.90</td><td>0.85</td><td>0.87</td><td>0.84</td></tr><tr><td>CLIP+DINOv3</td><td>0.96</td><td>0.87</td><td>0.88</td><td>0.96</td><td>0.96</td><td>0.88</td><td>0.95</td><td>0.90</td><td>0.94</td><td>0.88</td><td>0.88</td><td>0.86</td></tr></table>

Continuous error on synthetic test tiles. For synthetic samples, the original continuous CGA parameters are available. We therefore also compare the normalized continuous target $\tilde { v } _ { i j }$ with the representative value $r _ { j } ( \hat { c } _ { i j } )$ of the predicted preset. For the synthetic samples $S _ { j }$ with a valid continuous target for attribute j,

$$
\mathrm { M A E } _ { j } = \frac { 1 } { | S _ { j } | } \sum _ { i \in S _ { j } } \left| r _ { j } ( \hat { c } _ { i j } ) - \tilde { v } _ { i j } \right| .\tag{3}
$$

This measures the continuous error of the preset-decoded prediction, rather than the output of a continuous regression model. The error is dimensionless in normalized tile coordinates, not meters. Table 12 reports the results.

Table 12: Synthetic-test preset accuracy and normalized continuous error. The representative value of the predicted preset is compared with the normalized continuous CGA parameter.
<table><tr><td>Attribute group</td><td>Exact Acc.↑</td><td>MAE↓</td></tr><tr><td>Window frame geometry</td><td>0.900</td><td>0.026</td></tr><tr><td>Window grid ratios</td><td>0.880</td><td>0.031</td></tr><tr><td>Window sill geometry</td><td>0.943</td><td>0.016</td></tr><tr><td>Entrance geometry</td><td>0.910</td><td>0.024</td></tr></table>

## E.3 Synthetic-Only versus Synthetic-Plus-Real Training

We compare a component predictor trained with synthetic data only against one trained with the same synthetic data augmented by the real training partition. Both are evaluated on the same held-out set of 107 real tiles from 36 building groups (Appendix D.2). Only the synthetic-plus-real setting uses the 100 real training crops. This experiment tests the effect of adding real-domain supervision, separately from evaluating a mixed-domain-trained model on two test domains.

The representative attribute scores improve with real training crops: W-Arch increases from 0.91 to 0.98, W-Row/Col from 0.82 to 0.89, W-Sill from 0.83 to 0.92, E-Stair from 0.89 to 0.95, and E-Handle from 0.89 to 0.96. These results support synthetic-to-real transfer and the benefit of real-domain calibration on the evaluated building-disjoint test partition.

## F Additional Qualitative Results

## F.1 Additional Global Structure Layout Recovery Results

Regular CMP facades. Fig. 12 presents additional global structure layout recovery results on regular facade examples from CMP. For these cases, the provided annotations are remapped into our visualization scheme and used as structural references for layout-level comparison. These additional examples further demonstrate that the proposed global structure recovery stage can infer stable floor-wise and bay-wise facade organization from RGB images.

Challenging facades. Fig. 4 shows additional results on challenging facade images from eTRIMS and in-the-wild examples. These cases include perspective distortion, partial occlusion, nonuniform spacing, and incomplete visual evidence. The results provide further qualitative evidence that the reflection-based layout recovery stage can recover structurally meaningful facade layouts under difficult visual conditions.

Table 13: Training-data comparison on the same held-out real test partition. W and E denote windowlike and entrance-like components. Overall macro is the arithmetic mean of the five reported grouped metrics.
<table><tr><td>Training data</td><td>Overall macro↑</td><td>W-Arch↑</td><td>W-Row/Col↑</td><td>W-Sill↑</td><td>E-Stair↑</td><td>E-Handle↑</td></tr><tr><td>Synthetic only</td><td>0.868</td><td>0.91</td><td>0.82</td><td>0.83</td><td>0.89</td><td>0.89</td></tr><tr><td>Synthetic + real</td><td>0.940</td><td>0.98</td><td>0.89</td><td>0.92</td><td>0.95</td><td>0.96</td></tr></table>

![](images/e15f878d017aebbf686337ee2a155a20aeb6dd96b94e05fffb98e66bba134744.jpg)  
Figure 12: Additional global structure layout recovery results on regular CMP facades. Each example shows the input facade image, the remapped structural reference, and the recovered global structure layout.

Additional comparison with FaçAID. Fig. 13 provides additional qualitative comparisons with representative examples reported in FaçAID. Since FaçAID assumes segmented facade input and no unified executable evaluation protocol is available, these comparisons are intended as qualitative layout-level references. Each example compares the input image, the FaçAID result, and our recovered layout.

## F.2 Additional Comparison Results

Our method adopts a progressive reasoning workflow that infers appropriate facade structures through a dynamic reflection loop. To further illustrate its advantages, Fig. 14 provides additional qualitative comparisons against direct LLM/VLM baselines. The left panel compares facade-level direct endto-end generation with our full framework, while the right panel compares tile-level procedural generation strategies, including our fine-tuned predictor, direct low-level code generation, and direct middle-code prediction. These examples complement the quantitative comparisons in the main paper.

## F.3 Additional Editable Tile-Level Results

To further illustrate the interpretability and editability of the proposed component parameterization, we provide additional tile-level examples for representative window-like and entrance-like components.

![](images/0283a0f64d085601b1d068a964a3893a5df900bc18aa19ca66004b700577be42.jpg)  
Figure 13: Additional qualitative comparison with FaçAID. Each example shows the input facade image, the corresponding FaçAID result, and our recovered layout.

![](images/616a85cdf6773ad0a4afdd7e5797df815aa4047c4cb520cb683c44e7b9b9631b.jpg)  
Figure 14: Additional qualitative comparisons between our progressive reasoning workflow and direct LLM/VLM outputs. (a) Facade-level comparison between our full framework and direct end-to-end CGA generation by a general LLM/VLM [DeepMind, 2026]. From left to right are the input facade image, our result, and the direct LLM/VLM result. (b) Tile-level comparison among our fine-tuned tile-level predictor, direct low-level code generation, and direct middle-code prediction. From left to right are the input tile crop, our result, the direct low-level code generation result, and the direct middle-code result.

These examples show that the recovered structured tile descriptions can be translated into editable procedural assets while preserving the main local architectural properties.

Window-like tiles. Fig. 15 shows additional editable reconstruction examples for window-like tiles. These examples cover variations in arch presence, grid subdivision, sill configuration, and frame structure.

Entrance-like tiles. Fig. 15 presents additional editable reconstruction examples for entrance-like tiles. The examples include different doorway layouts, stair configurations, frame structures, and hardware-related details.

![](images/9f7d3e6bcf466a745f8f47356cee426d18fe2659985b14f672226ebd3c635bd5.jpg)  
(a) Cropped Tiles  
(b) Reconstruction

![](images/708b26d0325a30e839271667e55b1d4c1274772b2275ea4e108335c9ff09682f.jpg)  
(c) Edited Variants  
Figure 15: Editable tile-level results. Top: representative window-like tiles. Bottom: representative entrance-like tiles. For each example, we show the input crop and the corresponding procedural reconstruction generated from the predicted tile-level representation.

## G User Study Protocol

We conduct a user preference study to compare our recovered facade layouts with FaçAID. The study is designed as a supplementary perceptual evaluation rather than a replacement for quantitative structural evaluation.

## G.1 Participants

The study involved 15 participants, including 5 participants with architecture-related backgrounds and 10 participants without architecture training. The participants consisted of 8 males and 7 females.

## G.2 Evaluation Protocol

Each participant evaluated all 12 paired facade examples. For each trial, participants were shown: (1) the input facade image and (2) two anonymized layout reconstruction results

The two methods were randomly assigned to the left or right position for each trial, and method identities were hidden. Participants were asked to select the layout that better preserves the dominant facade structure.

## The instruction provided to participants was:

Given the input facade and two candidate layout reconstructions, select the result that better preserves the dominant facade structure, including floor organization, window and entrance placement, major facade regions, and repetition patterns. Ignore color and rendering style. Select “Tie” if the two results are comparably faithful or if no reliable preference can be made.

## G.3 Preference Results

The study produced 106 votes for our method, 54 for FaçAID, and 20 ties, corresponding to 58.9%, 30.0%, and 11.1%, respectively. Among the 160 non-tie judgments, our method received 66.3% of the votes. Under an independent-vote binomial model, the non-tie preference proportion has a nominal exact 95% confidence interval of [58.4%, 73.5%] (Clopper–Pearson), with a two-sided $p = 4 . 8 0 \times 1 0 ^ { - 5 }$ against $p _ { 0 } = 0 . 5$ . These unadjusted statistics do not account for participantor example-level dependence and are reported only as exploratory evidence. We interpret these descriptive results as supplementary perceptual evidence on the 12 evaluated examples, rather than a comprehensive benchmark against FaçAID.

## H TRC Evaluation Protocol

Tree Repair Cost (TRC) measures the expert-validated structural repair cost under a fixed editing protocol. It is designed to evaluate how much manual structural correction is required for a predicted layout tree. TRC is evaluated by expert annotators who are separate from the participants in the FaçAID preference study.

For each evaluation example, we manually construct one valid reference layout and render it into a schematic representation. The reference schematic serves as a visual structural target; the predicted tree is not required to match the reference tree in split order, grouping strategy, or serialization.

We develop an annotation interface containing: (1) the input facade image, (2) the rendered reference schematic, (3) the predicted layout tree and its rendered schematic, and (4) editing operations for modifying the predicted tree.

The annotator edits only the predicted layout tree until its rendered schematic expresses the same dominant facade organization as the accepted reference. If the predicted tree already produces an acceptable reconstruction, its TRC is zero.

One atomic edit is defined as:

• inserting one structural node;

• deleting one structural node;

• modifying one existing node, including operator type, split axis, repeat count, span, or related attributes.

When several attributes of the same node are corrected together, they are counted as one node modification.

All annotators receive the same operation definitions and representative examples. Each sample is independently evaluated, and the lowest validated repair count is retained as its TRC.

Therefore, TRC should be interpreted as the lowest expert-validated repair cost under the fixed protocol, rather than a guaranteed global minimum over all possible procedural decompositions. We do not report a formal inter-annotator agreement statistic and interpret TRC as an expert-assisted structural diagnostic.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

## Answer: [Yes]

Justification: The abstract and introduction state that the method focuses on dominant visible facade reconstruction and editable building asset instantiation, rather than full reconstruction of unseen geometry. The method, experiments, and conclusion are aligned with this scope.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

## Answer: [Yes]

Justification: The paper discusses the scope and assumptions of the method, including its focus on the dominant visible facade, the orthogonal assumption, and the use of a simple mass model for editable building asset instantiation. The conclusion and method sections also clarify that the recovered visible-facade program is the primary output, rather than a full reconstruction of unseen building geometry.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [N/A]

Justification: The paper does not present theoretical results, theorems, or formal proofs. The contribution is an empirical and methodological framework for visual-to-executable procedural reconstruction.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper describes the model components, the global structure recovery and component parameterization pipeline, the deterministic CGA compilation process, the evaluation metrics, and the datasets used for macro-structure recovery and tile-level evaluation. Additional training and implementation details, representation schemas, and extended quantitative results are provided in the appendix.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [No]

Justification: The paper does not release code, model checkpoints, or newly constructed training data at submission time. However, the main method, representation schemas, deterministic compilation process, evaluation metrics, and training/implementation details are described in the main paper and appendix to support reproducibility.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: The main paper describes the evaluation settings, datasets, baselines, and metrics used for global structure recovery, component parameterization, direct generation comparison, and ablation studies. Additional training and implementation details, including data construction and model usage, are provided in the appendix.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Appendix G reports an exploratory vote-level binomial analysis with its independence assumption and limitations. We do not report participant- or example-adjusted uncertainty estimates, or repeated-run uncertainty estimates for the other quantitative experi ments.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No]

Justification: The paper reports the model architecture, training settings, evaluation protocol, and deterministic compilation pipeline. Detailed hardware specifications, runtime measurements, and total computational costs are not reported.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research uses architectural images and procedural modeling techniques and does not involve sensitive personal data, biometric identification, surveillance, or highrisk decision-making. We preserve submission anonymity and follow the NeurIPS Code of Ethics throughout the study.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

## Answer: [Yes]

Justification: The paper discusses the intended use of the method for editable architectural and urban modeling. Potential positive impacts include reducing the cost of procedural building asset creation, while potential negative impacts include over-reliance on automatically reconstructed building geometry in applications that require accurate measurements or safety-critical decisions.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

## Answer: [No]

Justification: The paper does not release a high-risk pretrained model, image generator, or scraped dataset. The method focuses on procedural reconstruction of building facades and is not intended for surveillance, biometric identification, or safety-critical decision-making.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper cites the existing datasets, models, and related methods used for evaluation and comparison, including facade datasets, prior procedural modeling methods, and LLM/VLM baselines. We use these assets for research evaluation and follow their intended academic use.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

## Answer: [No]

Justification: The paper constructs synthetic and real training examples for layout recovery and component parameterization, but these assets are not released at submission time. The paper and appendix document the data construction procedure, representation schemas, and annotation format to support understanding and future reproducibility.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

## Answer: [No]

Justification: Appendix G provides participant information, the evaluation protocol, and the full instructions given to participants. Interface screenshots and compensation details are not reported. The study is used as supplementary perceptual evidence.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

## Answer: [No]

Justification: The paper includes a small user preference study involving anonymized visual comparisons, but it does not report IRB approval or equivalent review. The study does not collect sensitive personal information and is not used for high-risk decision-making.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

## Answer: [Yes]

Justification: The core method uses a fine-tuned vision-language model for global structure layout induction and visual-symbolic reflection, and also compares against LLM/VLM direct generation baselines. These uses are described in the method, experiments, and appendix.