# Vis2State: Visual Reasoning through Executable State Abstractions

Sheng Jin Wentao Liu Chen Qian

SenseTime Research and Tetras.AI

{jinsheng, liuwentao, qianchen}@tetras.ai

## Abstract

Solving visual puzzles requires both accurate interpretation ofvisual evidence and reliable computation under taskspecific rules. Embedding visual assumptions and task constraints directly in solving code makes the representedproblem difficult to inspect and revise independently. We introduce Vis2State, a training-free agentic framework that abstracts visual puzzles into explicit, revisable problem states linking visual inspection and execution. Visual inspection guides state construction and refinement, while generated programs solve the represented problem. The model constructs both the state and the program without predefined puzzle schemas or solving procedures. Inspection and execution feedback allow it to revise the problem representation and solving procedure independently. Vis2State achieves the state-of-the-art performance on four benchmarks. Cross-model evaluation further demonstrates consistent gains. Ablations, error analyses, andfixed-program state interventions further support the effectiveness of the state-centered design. The code will be made publicly available tofacilitatefuture research.

## 1 Introduction

Visual puzzles probe a model’s ability to turn image observations into a solution under task-specific rules. Beyond recognizing objects, the model must infer their relations and task constraints to guide search or computation. Benchmarks spanning algorithmic puzzles, spatial reasoning, visual logic, and abstract patterns expose this connection between perception and reasoning [9, 14, 34, 47]. Their tightly coupled perceptual and computational demands make visual puzzles a revealing testbed for understanding where multimodal reasoning succeeds and where it breaks down.

Bridging visual perception and computation requires organizing visual evidence into a representation of the problem to be solved. The required information varies across tasks. Mazes require connectivity, color matching requires correspondences, and box pushing requires positions and legal transitions. Existing approaches provide complementary capabilities. Chain-of-thought (CoT) prompting elicits intermediate reasoning [23, 42], visual search retrieves relevant details [44], and visual programming connects perception to executable operations [17, 36]. These capabilities alone do not ensure that the represented problem and its solving procedure are both correct. A sound algorithm can fail on misinterpreted constraints, just as accurate observations can be undermined by incomplete search. When visual assumptions and task constraints are embedded directly in solving code, these two sources of error become entangled, making the problem representation difficult to inspect and revise independently.

(a) What are the coordinates of P?  
![](images/cbb0094fcb217036608ef39cfaecaeb74ee66dc3102a05f28942da6958b3d0fe.jpg)

(b) What is the shortest path from S to G?  
![](images/82200a1a35663df4ac71feb01de550e3b91c00f32e89f22ad286163eef702dbb.jpg)  
Figure 1. Vis2State turns visual evidence into explicit problem states for computational solving. Two illustrative puzzles show how (a) inspection corrects a misread coordinate and (b) execution over explicit state finds the shortest path.

Our central idea is that visual puzzles can be abstracted into explicit, revisable problem states. We introduce Vis2State, a training-free agentic framework that uses these states as the interface between visual inspection and execution. Through inspection, the model constructs a taskdependent state encoding relevant objects, relations, constraints, and objectives, then generates a program to solve the represented problem. Neither the state schema nor the program is predefined. This separation allows the model to refine its interpretation of the puzzle independently of the procedure used to solve it.

The explicit state gives inspection and execution distinct but complementary roles. Inspection grounds the state in visual evidence by resolving uncertain objects, relations, or constraints. Execution takes the state and generated program as input, and returns computational results or errors. These observations guide the model’s next step, which may involve further inspection, state revision, or program refinement. The feedback-driven process allows the model to alternate between establishing the problem and solving it, rather than follow a fixed inspection–execution sequence. Figure 1 illustrates both roles. Inspection corrects a misread coordinate in (a), while execution uses an explicit state to find the shortest path in (b).

Experiments on four visual reasoning benchmarks show that Vis2State achieves state-of-the-art accuracy, improving over Direct CoT by 8.1–15.5 percentage points. Crossmodel evaluation confirms consistent gains, while ablations, error analyses, and fixed-program state interventions support the effectiveness of its state-centered design.

We make three contributions.

• Explicit states as a reasoning interface. We introduce Vis2State, a training-free agentic framework that abstracts visual puzzles into explicit, revisable problem states, separating the represented problem from its solving procedure.

• State-centered inspection and execution. We develop an adaptive reasoning process that uses visual inspection to construct and refine states, and generated programs to solve the represented problems, with execution feedback guiding further revision.

• Strong performance and extensive analysis. Vis2State achieves state-of-the-art accuracy on four benchmarks and consistent gains over Direct CoT baselines. Component ablations, error analyses, and fixed-program state interventions further substantiate the effectiveness of the state-centered design.

## 2 Related Work

Visual inspection and tool use. Visual tools support reasoning by acquiring and transforming task-relevant evidence. V\* [44] guides visual search toward relevant image details. Visual Sketchpad [19] supports reasoning through intermediate sketches, while ReFocus [12] directs visual attention through highlighting and masking. Thyme [52] and CodeDance [33] use executable code for image manipulation and computation. To learn visual tool use, Pixel Reasoner [35] combines instruction tuning with curiositydriven reinforcement learning, while DeepEyes [53] learns active perception through reinforcement learning. VTool-R1 [43] trains visual editing with outcome rewards, ToolsRL [11] introduces direct tool supervision, and VC-Tooler [45] learns tool composition and adaptation to returned observations. CodeV [18] evaluates evidence relevance and trains faithful tool use, while See2Think [48] examines dependence on intermediate visual states. Vis2State directs inspection toward constructing and checking an explicit problem state that a generated program uses without access to the image. This interface makes the visual premises of computation explicit and revisable.

Program-based reasoning. Program-based methods make intermediate reasoning executable. Building on CoT [23, 42], PAL [13] and Program of Thoughts [8] delegate computation to interpreters. Faithful CoT separates symbolic translation from deterministic solving [25], and Chain of Code combines execution with language-model emulation [24]. Execution can also inform subsequent reasoning. ReAct interleaves reasoning with actions and observations [49], while CodeAct uses executable code as its action space [41]. In vision, programs compose visual modules [17, 21, 36], and explicit scene representations support symbolic reasoning [26, 50]. Vis2State centers this process on a revisable problem state supplied to a generated program that runs without image access. It represents task-dependent objects, relations, constraints, and objectives separately from the program, allowing the problem representation and solving procedure to be revised independently.

## 3 Vis2State

Vis2State connects visual inspection and computational solving through an explicit problem state (Figure 2). Inspection provides evidence to construct and refine the state, while execution applies a generated program to solve the represented problem. A vision-language model $M _ { \theta }$ coordinates these complementary processes, using accumulated feedback to determine what to inspect, revise, or compute next. Given an image x and question $q ,$ the model selects its next action as

$$
a _ { t } = M _ { \theta } ( x , q , H _ { t - 1 } ; \mathcal { P } ) , \qquad H _ { 0 } = [ ] ,\tag{1}
$$

where $\mathcal { P }$ is the shared prompt and $H _ { t - 1 }$ records prior actions and observations. Each action $a _ { t }$ inspects the image, executes a program, or submits an answer. Subject to call and time budgets, the model can revisit inspection or execution as needed, rather than follow a fixed sequence. All datasets and models share the same prompt and tool instructions. Details appear in the supplementary material.

## 3.1 State Abstraction

The problem state encodes task-relevant information from the image and question independently of the solving procedure. For an execution action at step t, we represent the problem state as

$$
s _ { t } = \{ k _ { i } : v _ { i } \} _ { i = 1 } ^ { K _ { t } } ,\tag{2}
$$

![](images/fd10dbd7b279c1ffaca469fc914eaa1afbb1eac7c0e44862c964497da2290bfc.jpg)  
Figure 2. Inspection, State, and Execution in Vis2State. Inspection gathers visual evidence to construct the State, and Execution solves the represented problem. The model refines the state or program through feedback before submitting an answer.

Table 1. Visual inspection tools, with interaction design informed by $\mathrm { V ^ { * } }$ [44] and Visual Sketchpad [19].
<table><tr><td>Tool</td><td>Output</td></tr><tr><td>crop_region</td><td>Cropped image with coordinates.</td></tr><tr><td>pixel_probe</td><td>RGB values at queried points and re- gional color statistics.</td></tr><tr><td>edge_view</td><td>Image of local RGB differences.</td></tr><tr><td>color_extent</td><td>Bounding box and count of matched pixels, with an annotated image.</td></tr><tr><td>color_components</td><td>Bounding boxes and sizes of con- nected color components, with an an- notated image.</td></tr></table>

where $s _ { t }$ is a JSON object with $K _ { t }$ key–value fields. The model constructs the fields $k _ { i }$ and their values $v _ { i }$ to represent task-relevant objects, relations, constraints, and objectives, without predefined task- or model-specific schemas.

Keeping this representation separate from the program allows the model to revise its interpretation of the puzzle independently of the solving procedure. In box pushing, for example, the state records walls, player, box, and goal positions, while the program implements legal transitions and search. A misread wall requires correcting the state, whereas an incorrect transition implementation requires revising the program. The model can update either component in subsequent actions, with prior observations and results retained in the interaction history. Inspection and answer submission do not require a state. Recorded state examples appear in the supplement.

## 3.2 Visual Inspection

Visual inspection grounds the problem state in image evidence. The model requests targeted views or measurements to guide state construction and resolve uncertainty in the current representation. At step $t ,$ it specifies the tools and their arguments in a request list $R _ { t } .$ , yielding

$$
a _ { t } = \mathrm { I n s p e c t } ( x , R _ { t } ) \ \longrightarrow \ o _ { t } ,\tag{3}
$$

where $o _ { t }$ denotes the observation returned after executing action $a _ { t }$ . The tools in Table 1 perform deterministic image operations, while the model interprets their outputs in the context of the task. For example, a connected color component identifies a pixel group, but the model must determine whether that group corresponds to a wall, goal, or another task entity. These observations ground task-relevant attributes and relations in visual evidence.

The model adaptively selects inspection calls from accumulated observations, either before execution or when computational feedback raises uncertainty about the state.

## 3.3 Execution and Feedback

For execution, the model supplies a state–program pair $( s _ { t } , f _ { t } )$ , where $f _ { t }$ defines solve(state). Execution applies $f _ { t }$ to $s _ { t }$ and returns an observation,

$$
a _ { t } = \mathrm { E x e c } ( s _ { t } , f _ { t } ) \ \longrightarrow \ o _ { t } .\tag{4}
$$

The program takes the supplied state as input, without access to the image, and returns a computational result or error. After each nonterminal interaction, the action and observation are appended to the history,

$$
H _ { t } = H _ { t - 1 } \parallel [ ( a _ { t } , o _ { t } ) ] ,\tag{5}
$$

where ∥ denotes sequence concatenation. Using this history, the model can inspect uncertain visual details, revise the state or program, or submit an answer. Execution feedback thus supports iterative refinement of both the problem representation and the solving procedure.

## 4 Experiments and Analysis

## 4.1 Experimental protocol

Datasets. We evaluate 5,968 multiple-choice questions across four benchmarks. AlgoPuzzleVQA [14] contains

<table><tr><td>Method</td><td>Overall</td><td>Board</td><td>Calendar</td><td>Chain</td><td>Checker</td><td>Clock</td><td>Colour</td><td>Map</td><td>Maze</td><td>Move box</td></tr><tr><td>GPT-4V [27]</td><td>30.3</td><td>48.0</td><td>57.0</td><td>21.0</td><td>33.0</td><td>25.0</td><td>27.0</td><td>34.0</td><td>30.0</td><td>29.0</td></tr><tr><td>Claude 3 Opus [1]</td><td>30.9</td><td>48.0</td><td>39.0</td><td>23.0</td><td>35.0</td><td>37.0</td><td>16.0</td><td>33.0</td><td>23.0</td><td>19.0</td></tr><tr><td>Clos-ouurce GPT-5.2 [29]</td><td>54.6</td><td>49.0</td><td>70.0</td><td>23.0</td><td>56.0</td><td>76.0</td><td>38.0</td><td>51.0</td><td>38.0</td><td>39.0</td></tr><tr><td>o3 (high) [30]</td><td>62.3</td><td>60.0</td><td>96.0</td><td>75.0</td><td>56.0</td><td>98.0</td><td>47.0</td><td>5.0</td><td>39.0</td><td>52.0</td></tr><tr><td>o4-mini (high) [30]</td><td>65.4</td><td>52.0</td><td>99.0</td><td>77.0</td><td>97.0</td><td>91.0</td><td>47.0</td><td>3.0</td><td>42.0</td><td>50.0</td></tr><tr><td>Gemini 3 Pro Preview [16]</td><td>81.3</td><td>93.0</td><td>99.0</td><td>61.0</td><td>94.0</td><td>98.0</td><td>43.0</td><td>61.0</td><td>49.0</td><td>87.0</td></tr><tr><td>Qwen2-VL-7B [39]</td><td>28.2</td><td>45.0</td><td>28.0</td><td>31.0</td><td>33.0</td><td>26.0</td><td>20.0</td><td>32.0</td><td>25.0</td><td>27.0</td></tr><tr><td>Qwen3-VL-8B-Instruct [5]</td><td>45.0</td><td>45.0</td><td>70.0</td><td>26.0</td><td>64.0</td><td>55.0</td><td>30.0</td><td>44.0</td><td>26.0</td><td>27.0</td></tr><tr><td>Opeo-urce DeepSeek V4.1 Flash [46]</td><td>82.3</td><td>91.0</td><td>97.0</td><td>97.0</td><td>100.0</td><td>96.0</td><td>49.0</td><td>66.0</td><td>41.0</td><td>57.0</td></tr><tr><td>DeepSeek V4.1 Flash + Vis2State (Ours)</td><td>97.8</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>81.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Method</td><td>Overall</td><td>Queens</td><td>Number</td><td>Fruits</td><td>Rubik</td><td>Think dot</td><td>Hanoi</td><td>Water</td><td>Wheel</td><td>Wood</td></tr><tr><td>GPT-4V [27]</td><td>30.3</td><td>14.0</td><td>41.0</td><td>27.0</td><td>48.0</td><td>31.0</td><td>17.0</td><td>17.0</td><td>25.0</td><td>21.0</td></tr><tr><td>Claude 3 Opus [1]</td><td>30.9</td><td>15.0</td><td>43.0</td><td>40.0</td><td>39.0</td><td>53.0</td><td>22.0</td><td>37.0</td><td>19.0</td><td>16.0</td></tr><tr><td>Clo-ouurce GPT-5.2 [29]</td><td>54.6</td><td>98.0</td><td>83.0</td><td>74.0</td><td>72.0</td><td>56.0</td><td>42.0</td><td>48.0</td><td>40.0</td><td>30.0</td></tr><tr><td>03 (high) [30]</td><td>62.3</td><td>35.0</td><td>30.0</td><td>84.0</td><td>85.0</td><td>69.0</td><td>79.0</td><td>88.0</td><td>75.0</td><td>48.0</td></tr><tr><td>o4-mini (high) [30]</td><td>65.4</td><td>56.0</td><td>29.0</td><td>97.0</td><td>92.0</td><td>74.0</td><td>85.0</td><td>82.0</td><td>69.0</td><td>36.0</td></tr><tr><td>Gemini 3 Pro Preview [16]</td><td>81.3</td><td>96.0</td><td>94.0</td><td>100.0</td><td>88.0</td><td>74.0</td><td>83.0</td><td>84.0</td><td>81.0</td><td>78.0</td></tr><tr><td>Qwen2-VL-7B [39]</td><td>28.2</td><td>36.0</td><td>34.0</td><td>30.0</td><td>37.0</td><td>27.0</td><td>23.0</td><td>18.0</td><td>31.0</td><td>23.0</td></tr><tr><td>Opeoon-ouuce Qwen3-VL-8B-Instruct [5]</td><td>45.0</td><td>36.0</td><td>76.0</td><td>71.0</td><td>50.0</td><td>56.0</td><td>20.0</td><td>50.0</td><td>50.0</td><td>14.0</td></tr><tr><td>DeepSeek V4.1 Flash [46]</td><td>82.3</td><td>76.0</td><td>100.0</td><td>100.0</td><td>97.0</td><td>81.0</td><td>86.0</td><td>95.0</td><td>88.0</td><td>64.0</td></tr><tr><td>DeepSeek V4.1 Flash + Vis2State (Ours)</td><td>97.8</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>87.0</td><td>100.0</td><td>100.0</td><td>96.0</td><td>96.0</td></tr></table>

Table 2. Accuracy (%) across 18 task families on AlgoPuzzleVQA [14]. Bold indicates the best model result in each column.

Table 3. Accuracy (%) by reasoning category on VisualPuzzles [34]. Bold indicates the best model result in each column.
<table><tr><td>Method</td><td>Overall</td><td>Algorithmic</td><td>Analogical</td><td>Deductive</td><td>Inductive</td><td>Spatial</td></tr><tr><td>Human (50th Percentile)</td><td>75.0</td><td>88.0</td><td>66.0</td><td>80.0</td><td>50.0</td><td>90.0</td></tr><tr><td>Claude 3.7 Sonnet [2]</td><td>48.3</td><td>64.5</td><td>48.3</td><td>65.0</td><td>26.8</td><td>37.4</td></tr><tr><td>Claude Opus 4.1 (thinking) [3]</td><td>48.8</td><td>一</td><td></td><td></td><td></td><td></td></tr><tr><td>Clos-ource Gemini 2.5 Pro [10]</td><td>49.5</td><td>60.0</td><td>64.0</td><td>60.0</td><td>29.7</td><td>36.4</td></tr><tr><td>o3 [30]</td><td>54.0</td><td>64.5</td><td>68.3</td><td>69.5</td><td>27.3</td><td>42.7</td></tr><tr><td>o4-mini [30]</td><td>57.0</td><td>65.3</td><td>68.7</td><td>75.5</td><td>33.0</td><td>45.5</td></tr><tr><td>GPT-5 (high) [32]</td><td>57.3</td><td>1</td><td>一</td><td>一</td><td></td><td></td></tr><tr><td>Gemini 3.1 Pro [15]</td><td>73.5</td><td>一</td><td>一</td><td>一</td><td>1</td><td>1</td></tr><tr><td>Qwen3-VL-8B-Instruct [5]</td><td>36.2</td><td>一</td><td>1</td><td></td><td>一</td><td></td></tr><tr><td>Qwen2.5-VL-72B-Instruct [6]</td><td>42.3</td><td>53.4</td><td>46.9</td><td>58.0</td><td>25.8</td><td>29.5</td></tr><tr><td>Qwen3-VL-235B-A22B (instruct) [5]</td><td>54.7</td><td>1</td><td>一</td><td></td><td></td><td></td></tr><tr><td>Opeoon-rce Qwen3-VL-235B-A22B (thinking) [5]</td><td>57.2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DeepSeek V4.1 Flash [46]</td><td>61.6</td><td>77.1</td><td>63.0</td><td>83.5</td><td>38.8</td><td>47.6</td></tr><tr><td>DeepSeek V4.1 Flash + Vis2State (Ours)</td><td>75.0</td><td>92.0</td><td>76.3</td><td>92.5</td><td>47.4</td><td>66.4</td></tr></table>

1,800 questions in 18 algorithmic puzzle families, including mazes, box pushing, map coloring, and the Tower of Hanoi, which require rule-constrained search and computation. VisualPuzzles [34] contains 1,168 questions spanning algorithmic, analogical, deductive, inductive, and spatial reasoning, with little specialized knowledge required. VisuLogic [47] contains 1,000 vision-centric logic questions across Quantity, Spatiality, Position, Attribute, Style, and Other, covering counting, folding, rotation, and graphical composition to test visual attributes and relations. PuzzleVQA [9] contains 2,000 abstract-pattern questions from

20 templates, covering numbers, colors, shapes, and size through four single-concept and six dual-concept categories requiring visual pattern induction.

Evaluation protocol. We report accuracy using each benchmark’s released answer-extraction and scoring procedures. Direct CoT follows the official CoT prompt for each benchmark [9, 14, 34, 47]. Following VisualPuzzles [34], we uniformly sample a valid option with a fixed seed when no final answer is produced. For VisuLogic evaluation, we use DeepSeek V4.1 Flash in place of the released evaluator’s GPT-4o-mini [28] because of model availability.

Table 4. Accuracy (%) across six reasoning categories on VisuLogic [47]. Bold indicates the best model result in each column.
<table><tr><td>Method</td><td>Overall</td><td>Quantity</td><td>Spatiality</td><td>Position</td><td>Attribute</td><td>Style</td><td>Other</td></tr><tr><td>Human</td><td>51.4</td><td>45.3</td><td>52.7</td><td>71.1</td><td>50.0</td><td>47.5</td><td>44.2</td></tr><tr><td>GPT-4o [20]</td><td>26.3</td><td>28.6</td><td>24.7</td><td>27.2</td><td>26.8</td><td>20.0</td><td>25.9</td></tr><tr><td>Claude Opus 4.1 (thinking) [3]</td><td>27.9</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td></td></tr><tr><td>Gemini 2.0 Pro [22]</td><td>28.0</td><td>29.7</td><td>24.2</td><td>27.9</td><td>30.5</td><td>22.2</td><td>33.3</td></tr><tr><td>GPT-5 (high) [32]</td><td>28.5</td><td>1</td><td>一</td><td></td><td></td><td></td><td></td></tr><tr><td>o3 [30]</td><td>29.5</td><td>24.2</td><td>31.0</td><td>34.6</td><td>27.0</td><td>25.0</td><td>41.9</td></tr><tr><td>Closo-uice Gemini 2.5 Pro (thinking) [10]</td><td>31.6</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Claude Opus 4.7 [4]</td><td>32.6</td><td></td><td>一</td><td></td><td></td><td></td><td></td></tr><tr><td>Doubao 1.6 Vision [38]</td><td>34.9</td><td>32.9</td><td>34.6</td><td>39.0</td><td>31.7</td><td>30.0</td><td>43.5</td></tr><tr><td>GPT-5.5 [31]</td><td>43.5</td><td>一</td><td>一</td><td>一</td><td>一</td><td></td><td>一</td></tr><tr><td>Gemini 3.1 Pro [15]</td><td>44.8</td><td>1</td><td>1</td><td></td><td>一</td><td></td><td></td></tr><tr><td>Qwen2.5-VL-72B [6]</td><td>26.2</td><td>25.2</td><td>23.8</td><td>27.2</td><td>25.6</td><td>25.6</td><td>34.3</td></tr><tr><td>InternVL3-78B [54]</td><td>27.7</td><td>27.7</td><td>26.1</td><td>31.6</td><td>26.3</td><td>21.3</td><td>32.3</td></tr><tr><td>Qwen3-VL-235B-A22B (instruct) [5]</td><td>29.9</td><td>1</td><td>一</td><td>1</td><td>一</td><td></td><td></td></tr><tr><td>Qwen3-VL-235B-A22B (thinking) [5]</td><td>34.4</td><td>一</td><td>一</td><td>一</td><td></td><td></td><td></td></tr><tr><td>Opeo-ource DeepSeek V4.1 Flash [46]</td><td>35.5</td><td>34.0</td><td>37.7</td><td>38.2</td><td>32.9</td><td>23.3</td><td>44.4</td></tr><tr><td>DeepSeek V4.1 Flash + Vis2State (Ours)</td><td>49.1</td><td>47.9</td><td>39.8</td><td>68.4</td><td>45.1</td><td>46.7</td><td>53.7</td></tr></table>

Table 5. Accuracy (%) on single- and dual-concept puzzles in PuzzleVQA [9]. N/C/S/Z denote numbers/colors/shapes/size. Bold indicates the best model result in each column.
<table><tr><td colspan="8">Single-concept</td><td colspan="8">Dual-concept</td></tr><tr><td>Method</td><td>Overall</td><td>N</td><td>C</td><td>Z</td><td>S</td><td> $\operatorname { A v g } .$ </td><td>N+S</td><td>C+N</td><td></td><td>N+Z</td><td>C+S</td><td>Z+S</td><td>C+Z</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Claude 3 Opus [1]</td><td>42.0</td><td>47.0</td><td>32.5</td><td>33.5</td><td>44.5</td><td>39.4</td><td>48.0</td><td>54.5</td><td>34.0</td><td></td><td>42.5</td><td>33.0</td><td>50.0</td><td>43.7</td></tr><tr><td>GPT-4V [27]</td><td>45.9</td><td>67.5</td><td>42.0</td><td>35.0</td><td>41.0</td><td>46.4</td><td>52.5</td><td>56.0</td><td></td><td>31.0</td><td>47.0</td><td>31.5</td><td>55.0</td><td>45.5</td></tr><tr><td>GPT-5.2 [29]</td><td>67.3</td><td>69.5</td><td>98.0</td><td>66.0</td><td>71.5</td><td>76.3</td><td>75.0</td><td>68.5</td><td></td><td>37.0</td><td>61.0</td><td>68.5</td><td>58.5</td><td>61.4</td></tr><tr><td>Clos-orce 03 (high) [30]</td><td>84.0</td><td>99.5</td><td>99.0</td><td>87.5</td><td>74.0</td><td>90.0</td><td>92.0</td><td></td><td>97.5</td><td>45.0</td><td>92.0</td><td>97.0</td><td>57.0</td><td>80.1</td></tr><tr><td>o4-mini (high) [30]</td><td>85.7</td><td>99.0</td><td>99.5</td><td>86.0</td><td>70.5</td><td>88.8</td><td>96.0</td><td></td><td>99.5</td><td>62.0</td><td>86.5</td><td>85.5</td><td>72.5</td><td>83.7</td></tr><tr><td>Gemini 3 Pro Preview [16]</td><td>86.3</td><td>99.0</td><td>94.0</td><td>83.5</td><td>77.0</td><td>88.4</td><td>99.0</td><td></td><td>92.5</td><td>73.5</td><td>84.5</td><td>88.5</td><td>71.5</td><td>84.9</td></tr><tr><td>Opeo-rce Qwen3-VL-8B-Instruct [5]</td><td>59.9</td><td>80.0</td><td>85.5</td><td>34.5</td><td>74.0</td><td>68.5</td><td></td><td>71.5</td><td>70.5</td><td>35.5</td><td>39.0</td><td>49.5</td><td>58.5</td><td>54.1</td></tr><tr><td>DeepSeek V4.1 Flash [46]</td><td>85.2</td><td>93.0</td><td>95.5</td><td>83.0</td><td>70.5</td><td>85.5</td><td></td><td>92.0</td><td>98.5</td><td>61.5</td><td>92.5</td><td>89.5</td><td>76.0</td><td>85.0</td></tr><tr><td>DeepSeek V4.1 Flash + Vis2State (Ours)</td><td>93.3</td><td>96.0</td><td>99.5</td><td>100.0</td><td>72.0</td><td>91.9</td><td></td><td>94.5</td><td>100.0</td><td>86.0</td><td>97.5</td><td>98.5</td><td>88.5</td><td>94.2</td></tr></table>

Implementations. DeepSeek-V4.1-Flash-260910 [46] is used for the main experiments, and GLM-5.3-Flash-260828 [51] for cross-model evaluation. Models receive the image and question, including answer choices, without reference answers or explanations. In our experiments, thinking is enabled with a maximum output length of 65,536 tokens per model call. Truncated responses return step-error feedback and allow continuation within the remaining budgets. The supplement provides prompt and runtime details.

## 4.2 Benchmark comparison

Vis2State achieves state-of-the-art overall accuracy among the compared methods on all four benchmarks (Tables 2–5). With DeepSeek V4.1 Flash, it outperforms Direct CoT by 8.1–15.5 percentage points, demonstrating substantial gains over the same model without tools.

The improvements extend to 36 of 39 categories, with the remaining three tied at 100.0% (Figure 3). On AlgoPuzzleVQA, Vis2State achieves 100.0% accuracy in 14 of 18 puzzle families, compared with three for Direct CoT. Gains span every category of VisualPuzzles and VisuLogic, with the largest improvements in Spatial reasoning and Position, respectively. Explicit states make spatial configurations and positional relations available for inspection and computation, providing a way to check visual interpretations and reason over the represented problem. Gains are smaller on VisuLogic Spatiality and PuzzleVQA Shapes, suggesting that some perceptual and relational ambiguities remain difficult to resolve.

Paired comparisons show that Vis2State corrects substantially more Direct CoT errors than it introduces on every benchmark (Table 6). The gains are significant under paired tests with Holm correction across the four benchmarks $( p < 2 \times 1 0 ^ { - 1 2 } )$ , supporting the consistency of the overall improvements.

![](images/39a633fef4bc82446567092808082a581f3cc7d755de19bbd33935773a29ab9c.jpg)

![](images/765b8cba065d23c1072c4ed7b97a38ffa436a3651f4ab34ef079aa96eeea6895.jpg)

![](images/35db8e6ceadbbebdf4119944c6f746f33a5b3610be049bf5169c7f48f589786f.jpg)

![](images/7a914fefe0a745677b1d777bdeea6af5fbcd4bff05e0801c03eb979ef28c59fb.jpg)  
Figure 3. Category-level accuracy and gains over Direct CoT. Blue bars show Direct CoT accuracy. Orange segments show the accuracy gain from Vis2State. Each full bar and its numeric label give Vis2State accuracy (%). In PuzzleVQA, N/C/S/Z denote numbers/colors/shapes/size. Vis2State improves 36 of 39 categories and ties the remaining three at 100.0%.

Table 6. Paired comparison with Direct CoT. # Wins count questions answered correctly only by Vis2State. # Losses count those answered correctly only by Direct CoT. Intervals are paired 95.0% bootstrap confidence intervals for the accuracy gain.
<table><tr><td>Benchmark</td><td># Wins</td><td># Losses</td><td>Gain [95.0% CI]</td></tr><tr><td>AlgoPuzzleVQA</td><td>295</td><td>16</td><td>15.5 [13.7, 17.3]</td></tr><tr><td>VisualPuzzles</td><td>237</td><td>80</td><td>13.4 [10.5, 16.4]</td></tr><tr><td>VisuLogic</td><td>238</td><td>102</td><td>13.6 [10.1, 17.1]</td></tr><tr><td>PuzzleVQA</td><td>208</td><td>47</td><td>8.1 [6.6, 9.6]</td></tr></table>

## 4.3 Component ablations

We examine how visual inspection, program execution, and explicit problem states contribute to accuracy using five configurations of the same DeepSeek model (Table 7). Direct CoT answers without tools using the benchmark’s official CoT prompt. Inspection only adds the five visual inspection tools, while Execution only provides Python execution and feedback. Inspection + Execution combines both capabilities without requiring an explicit state.

Vis2State further introduces state-construction guidance and the solve(state) interface.

Inspection only improves over Direct CoT on all four benchmarks. Vis2State consistently outperforms Inspection + Execution by 0.8–6.0 percentage points, with the largest gain on VisuLogic. These gains show the benefit of using explicit, revisable states to coordinate inspection and execution.

## 4.4 Error patterns and corrections

We examine where Direct CoT fails and which failures Vis2State resolves on the same 1,709 diagnostic questions (Figure 4), using the annotation protocol in the supplement. The dominant error types differ across benchmarks. Perception is the largest group on AlgoPuzzleVQA, VisualPuzzles, and VisuLogic, accounting for 44.2%, 54.8%, and 46.5% of errors, respectively. On PuzzleVQA, State / Reasoning accounts for 43.6%, highlighting difficulties in identifying the relations underlying visual patterns. These profiles distinguish failures in extracting visual evidence from failures in formulating and solving the represented problem.

Vis2State correctly answers 971 of these questions through valid submissions, excluding random completions. Correction rates vary across error groups. Search / Computation has the highest aggregate rate at 75.9%, largely reflecting AlgoPuzzleVQA, where 97.7% of cases are corrected. Perception yields the largest number of corrections, but its correction rate is 55.4%; State / Reasoning remains lower at 50.8%. This pattern is consistent with execution supporting systematic search and computation once a suitable state is available. The lower correction rates for Perception and State / Reasoning suggest that extracting visual information and inferring the relevant rules remain challenging for Vis2State.

Table 7. Ablation study. We compare visual inspection, code execution, and their combination using DeepSeek V4.1 Flash. Vis2State connects inspection and execution through an explicit state–program interface and achieves the highest accuracy on all four benchmarks. Checkmarks indicate the components included in each configuration. Results are accuracy (%), with the best score in each column in bold.
<table><tr><td>Configuration</td><td>Visual Inspection</td><td>Explicit State</td><td>Code Execution</td><td>AlgoPuzzleVQA</td><td>VisualPuzzles</td><td>VisuLogic</td><td>PuzzleVQA</td></tr><tr><td>Direct CoT</td><td>X</td><td>X</td><td>X</td><td>82.3</td><td>61.6</td><td>35.5</td><td>85.2</td></tr><tr><td>Inspection only</td><td>√</td><td>X</td><td>X</td><td>92.1</td><td>72.0</td><td>43.0</td><td>92.1</td></tr><tr><td>Execution only</td><td>X</td><td>X</td><td>√</td><td>90.0</td><td>64.0</td><td>37.7</td><td>89.5</td></tr><tr><td>Inspection + Execution</td><td>√</td><td>X</td><td>√</td><td>96.9</td><td>73.5</td><td>43.1</td><td>91.7</td></tr><tr><td>Vis2State</td><td>√</td><td>√</td><td>√</td><td>97.8</td><td>75.0</td><td>49.1</td><td>93.3</td></tr></table>

![](images/cd90ed40599fae5733ac757fff4c583bce72dadda4cb487991c904370686a41d.jpg)  
Figure 4. Direct CoT error profiles and corrections by Vis2State. Inner rings show Direct CoT error-group proportions. Outer rings show cases corrected by Vis2State (light, hatched) or not corrected (dark, solid), with within-group correction rates. Centers report corrected/total errors. State / Reasoning combines representation and rule errors; Search / Computation combines search, state updates, arithmetic, and answer comparison. Compound includes cross-group, answer-extraction, random-parser, and unattributed errors.

## 4.5 Counterfactual interventions on state

We select 200 questions each from AlgoPuzzleVQA and PuzzleVQA with correct original-state outputs; originalstate accuracy is 100.0% by construction. Holding programs and answer options fixed, we test original states, answer-changing counterfactuals, and answer-preserving controls. Independent reference programs verify that each counterfactual maps to a different existing answer option and that each control preserves the original answer. Selection does not depend on intervention success: all incorrect intervention outputs remain in the evaluation. The supplement details the protocol and subset composition.

Table 8 shows that the saved programs correctly answer 98.0% of counterfactuals on AlgoPuzzleVQA and 93.5% on PuzzleVQA. Counterfactual accuracy falls below 100% because some programs make reasoning or answer-mapping errors under the modified states. Under answer-preserving controls, accuracy remains at 100.0% and 98.5%, respectively. On these subsets, programs generally adapt their answers to state changes while preserving correctness under answer-preserving edits.

Table 8. State interventions. Correct counts and accuracies on 200 questions per dataset.
<table><tr><td>State condition</td><td>AlgoPuzzleVQA</td><td>PuzzleVQA</td></tr><tr><td>Original state</td><td>200 (100.0%)</td><td>200 (100.0%)</td></tr><tr><td>Counterfactual</td><td>196 (98.0%)</td><td>187 (93.5%)</td></tr><tr><td>Answer-preserving control</td><td>200 (100.0%)</td><td>197 (98.5%)</td></tr></table>

## 4.6 Cross-model evaluation

To assess whether Vis2State generalizes across models, we replace DeepSeek V4.1 Flash with GLM-5.3-Flash-260828 [51], keeping the prompt, tool instructions, and state interface unchanged. Table 9 shows consistent improvements over Direct CoT with the same model, with gains of 6.9–12.2 percentage points across all four benchmarks. These results extend the benefits of Vis2State to a second model without model-specific adaptation.

The benchmark difficulty pattern also remains consistent across the two models. GLM-based Vis2State achieves its highest accuracies on PuzzleVQA and AlgoPuzzleVQA, followed by VisualPuzzles, while VisuLogic remains the most challenging. This consistency suggests that substantial differences in task difficulty persist across models. The supplement reports category-level results.

![](images/8c0e37b2d2f45d43cc05b011fa12e241eded797048e290c65a4836ef97f77e36.jpg)  
Figure 5. Case study on Move Box. Direct CoT misreads the goal position and overlooks a wall, producing an incorrect two-push solution. Vis2State explicitly represents the puzzle state and searches legal push sequences, finding the minimum of four pushes.

Table 9. Cross-model evaluation. Accuracy (%) with GLM-5.3- Flash on AlgoPuzzleVQA (AP), VisualPuzzles (VP), VisuLogic (VL), and PuzzleVQA (PQ).
<table><tr><td>Method</td><td>AP</td><td>VP</td><td>VL</td><td>PQ</td></tr><tr><td>Direct CoT (GLM-5.3-Flash)</td><td>82.3</td><td>67.2</td><td>36.2</td><td>89.4</td></tr><tr><td>Vis2State (GLM-5.3-Flash)</td><td>94.5</td><td>75.1</td><td>44.2</td><td>96.3</td></tr><tr><td>Gain (pp)</td><td>+12.2</td><td>+7.9</td><td>+8.0</td><td>+6.9</td></tr></table>

## 4.7 Case study

Figure 5 illustrates how state construction and execution jointly support puzzle solving. Direct CoT mislocates the goal and overlooks a wall, leading it to propose two downward pushes. Vis2State instead constructs an explicit state containing the player, box, goal, and walls, then searches for a solution under the puzzle’s movement constraints. The search checks whether the player can reach the position required for each push and identifies a minimum of four pushes. This example highlights the complementary roles of the two stages. State construction establishes the spatial premises of the problem, while execution checks legal transitions and systematically searches for the shortest solution.

## 5 Limitations and Future Work

State validity. An explicit state may encode incorrect observations or rules, and generated programs may overlook relevant state fields, as discussed in Sec. 4.5. Successful execution alone therefore does not guarantee a correct interpretation or solution. Future work should verify states and inferred rules against the image and task constraints, using detected inconsistencies to guide further inspection.

Computational efficiency. Vis2State uses 1.7–2.8× the mean output tokens of Inspection + Execution. See the supplement for details. Selective inspection and reuse of previously constructed states may reduce this overhead.

## 6 Conclusion

Vis2State makes explicit problem states the interface between visual understanding and computational reasoning. Its central contribution is to encode the objects, relations, and constraints underlying a solution as a representation that can be inspected, revised, and used by generated programs. Experiments across four benchmarks and two models, supported by component ablations and state interventions, demonstrate the effectiveness of this design and provide evidence that generated programs use the supplied states. These findings highlight explicit state construction as a useful design principle for visual reasoning systems that combine perception with computation.

## LLM Usage Statement

OpenAI Codex and ChatGPT assisted with code development, literature organization, and manuscript revision. OpenAI image-generation tools were used to create the conceptual illustrations. The authors manually checked all content and details in the main paper and supplementary material, including the text, equations, experimental results, figures, tables, and references. The authors take full responsibility for the accuracy and integrity of the work.

## References

[1] Anthropic. The Claude 3 model family: Opus, Sonnet, Haiku. Technical report, Anthropic, 2024. 4, 5

[2] Anthropic. Claude 3.7 Sonnet System Card. https://www-cdn.anthropic.com/ 9ff93dfa8f445c932415d335c88852ef47f1201e. pdf, 2025. 4

[3] Anthropic. System Card Addendum: Claude Opus 4.1. https://assets.anthropic.com/m/ 4c024b86c698d3d4/original/Claude-4-1- System-Card.pdf, 2025. 4, 5

[4] Anthropic. System card: Claude Opus 4.7. Technical report, Anthropic, 2026. 5

[5] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 4, 5, 13

[6] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923, 2025. 4, 5

[7] ByteDance Seed. Seed2.1 model card: Agentic intelligence for productivity. https://lf3-static. bytednsdoc.com/obj/eden-cn/lapzildtss/ljhwZthlaukjlkulzlp/seed2.1/Seed2\_1\_ Model\_Card.pdf, 2026. 13

[8] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research, 2023. 2

[9] Yew Ken Chia, Vernon Toh, Deepanway Ghosal, Lidong Bing, and Soujanya Poria. PuzzleVQA: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns. In Findings of the Association for Computational Linguistics: ACL 2024, pages 16259–16273, 2024. 1, 4, 5, 12, 13

[10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 4, 5

[11] Qihua Dong, Gozde Sahin, Pei Wang, Zhaowei Cai, Robik Shrestha, Hao Yang, and Davide Modolo. Visual Reasoning Through Tool-Supervised Reinforcement Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 8993– 9002, 2026. 2

[12] Xingyu Fu, Minqian Liu, Zhengyuan Yang, John Richard Corring, Yijuan Lu, Jianwei Yang, Dan Roth, Dinei Florencio, and Cha Zhang. ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding. In Int. Conf. Mach. Learn., pages 17783–17805, 2025. 2

[13] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In Int. Conf. Mach. Learn., pages 10764–10799, 2023. 2

[14] Deepanway Ghosal, Vernon Toh, Yew Ken Chia, and Soujanya Poria. AlgoPuzzleVQA: Diagnosing multimodal reasoning challenges of language models with algorithmic multimodal puzzles. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 9615–9632, 2025. 1, 3, 4, 12, 13

[15] Google DeepMind. Gemini 3.1 Pro: Model Card. https://deepmind.google/models/modelcards/gemini-3-1-pro/, 2026. 4, 5

[16] Google DeepMind. Gemini 3 Pro: Model Card. https://storage.googleapis.com/deepmindmedia/Model-Cards/Gemini-3-Pro-Model-Card.pdf, 2026. 4, 5

[17] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In IEEE Conf. Comput. Vis. Pattern Recog., pages 14953– 14962. IEEE, 2023. 1, 2

[18] Xinhai Hou, Shaoyuan Xu, Manan Biyani, Moyan Li, Jia Liu, Todd C Hollon, and Bryan Wang. Codev: Code with images for faithful visual reasoning via tool-aware policy optimization. In IEEE Conf. Comput. Vis. Pattern Recog., pages 21500–21510, 2026. 2

[19] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Adv. Neural Inform. Process. Syst., pages 139348–139379, 2024. 2, 3

[20] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 5

[21] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Judy Hoffman, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Inferring and executing programs for visual reasoning. In Int. Conf. Comput. Vis., pages 2989–2998, 2017. 2

[22] Koray Kavukcuoglu. Gemini 2.0 is Now Available to Every one. https://blog.google/innovation-andai/models-and-research/google-deepmind/ gemini-model-updates-february-2025/, 2025. 5

[23] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large Language Models are Zero-Shot Reasoners. Adv. Neural Inform. Process. Syst., 35: 22199–22213, 2022. 1, 2

[24] Chengshu Li, Jacky Liang, Andy Zeng, Xinyun Chen, Karol Hausman, Dorsa Sadigh, Sergey Levine, Li Fei-Fei, Fei Xia, and Brian Ichter. Chain of code: Reasoning with a language model-augmented code emulator. In Int. Conf. Mach. Learn., pages 28259–28277, 2024. 2

[25] Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison

Burch. Faithful Chain-of-Thought Reasoning. In Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 305–329, 2023. 2

[26] Jiayuan Mao, Chuang Gan, Pushmeet Kohli, Joshua B Tenenbaum, and Jiajun Wu. The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision. In Int. Conf. Learn. Represent., 2019. 2

[27] OpenAI. GPT-4V(ision) System Card. https: //openai.com/index/gpt-4v-system-card/, 2023. 4, 5

[28] OpenAI. GPT-4o mini: Advancing Cost-Efficient Intelligence. https://openai.com/index/gpt-4o-mini-advancing-cost-efficientintelligence/, 2024. 5

[29] OpenAI. Update to GPT-5 System Card: GPT-5.2. https://openai.com/index/gpt-5-systemcard-update-gpt-5-2/, 2025. 4, 5

[30] OpenAI. OpenAI o3 and o4-mini System Card. Technical report, OpenAI, 2025. 4, 5

[31] OpenAI. GPT-5.5 System Card. https://openai. com/index/gpt-5-5-system-card/, 2026. 5

[32] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025. 4, 5

[33] Qi Song, Honglin Li, Yingchen Yu, Haoyi Zhou, Lin Yang, Song Bai, Qi She, Zilong Huang, and Yunqing Zhao. Codedance: A dynamic tool-integrated mllm for executable visual reasoning. In IEEE Conf. Comput. Vis. Pattern Recog., pages 19186–19195, 2026. 2

[34] Yueqi Song, Tianyue Ou, Yibo Kong, Zecheng Li, Graham Neubig, and Xiang Yue. VisualPuzzles: Decoupling Multimodal Reasoning Evaluation from Domain Knowledge. In Int. Conf. Mach. Learn., pages 114860–114932, 2026. 1, 4, 12, 13

[35] Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. Adv. Neural Inform. Process. Syst., 38:8222–8251, 2026. 2

[36] Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In Int. Conf. Comput. Vis., pages 11854–11864. IEEE, 2023. 1, 2

[37] Vernon Toh, Yew Ken Chia, Deepanway Ghosal, and Soujanya Poria. Tracking the evolution of multimodal reasoning on visual puzzles. In Logical and Symbolic Reasoning in Language Models, AAAI 2026, 2026. 13

[38] Volcano Engine. Doubao-Seed-1.6-vision. https: //docs.volcengine.com/docs/ark/modelrelease-announcement-2025?lang=en, 2025. 5

[39] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 4

[40] Qixun Wang, Yang Shi, Letian Cheng, Zhuoran Zhang, Yan He, Yuqi Tang, Qi Zhang, Xinlei Yu, Ruizhe Chen, Tianrun Xu, et al. Beacon: Knowing when and how to perform agentic visual reasoning. arXiv preprint arXiv:2607.28595, 2026. 13

[41] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable Code Actions Elicit Better LLM Agents. In Int. Conf. Mach. Learn., pages 50208–50232, 2024. 2

[42] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large lan guage models. Adv. Neural Inform. Process. Syst., pages 24824–24837, 2022. 1, 2

[43] Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, and Klara Nahrstedt. Vtool-r1: Vlms learn to think with images via reinforcement learning on multimodal tool use. In Int. Conf. Learn. Represent., pages 78298–78319, 2026. 2

[44] Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal llms. In IEEE Conf. Comput. Vis. Pattern Recog., pages 13084–13094, 2024. 1, 2, 3

[45] Yizheng Wu, Jiashen Hua, Bing Deng, and Jieping Ye. Vc tooler: Learning compositional and adaptive visual tool use. arXiv preprint arXiv:2608.02217, 2026. 2

[46] Anyi Xu, B Li, Bangcai Lin, Bing Xue, BingCheng Xian, Bingzheng Xu, Bochao Wu, Bowei Zhang, Boyi Deng, CC Yu, et al. DeepSeek-V4.1-Flash: Pushing the limits of KV cache compression. arXiv preprint arXiv:2609.19969, 2026. 4, 5

[47] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. In Int. Conf. Learn. Represent., pages 25966–26003, 2026. 1, 4, 5, 12, 13

[48] Siyu Yan, Zhuoran Yan, Haiying Xu, Panhao Zhou, Jingyu Chen, Chenhao Ji, Shuo Cao, Yongheng Zhang, Haoze Liu, Siyu Zhang, et al. See2think: Do multimodal models really use intermediate visual states? arXiv preprint arXiv:2607.26769, 2026. 2

[49] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In Int. Conf. Learn. Represent., 2023. 2

[50] Kexin Yi, Jiajun Wu, Chuang Gan, Antonio Torralba, Pushmeet Kohli, and Josh Tenenbaum. Neural-symbolic vqa: Disentangling reasoning from vision and language under standing. In Adv. Neural Inform. Process. Syst., 2018. 2

[51] Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. GLM-5: from Vibe Coding to Agentic Engineering. arXiv preprint arXiv:2602.15763, 2026. 5, 7, 13

[52] Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, Haonan Fan, Kaibing Chen, Jiankang Chen, Haojie

Ding, Kaiyu Tang, Zhang Zhang, Liang Wang, Fan Yang, Tingting Gao, and Guorui Zhou. Thyme: Think Beyond Images. In Int. Conf. Learn. Represent., pages 79536–79572, 2026. 2

[53] Ziwei Zheng, Minghao Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, and Chao Shen. Deepeyes: Incentivizing "thinking with images" via reinforcement learning. In Int. Conf. Learn. Represent., pages 126775–126798, 2026. 2

[54] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 5

## S1 Algorithm and runtime

Algorithm S1 Core reasoning loop of Vis2State   
Require: Image x, question q, budgets $B , T$   
1: $H _ { 0 } \gets [ ]$   
2: for $t = \stackrel { \triangledown } { 1 } , \dotsc , B$ while elapsed time $< T$ do   
3: $a _ { t } \gets M _ { \theta } ( x , q , H _ { t - 1 } ; \mathcal { P } )$ ▷ Choose next action   
4: if a<sub>t</sub> submits a final answer y then   
5: return y   
6: else   
7: Perform $a _ { t }$ and receive $o _ { t }$ ▷ Inspect or execute   
8: end if   
9: $H _ { t } \gets H _ { t - 1 } \parallel [ ( a _ { t } , o _ { t } ) ]$ ▷ Append feedback   
10: end for   
11: return no submission

Algorithm S1 summarizes the Vis2State reasoning loop. Given an image x and question q, a vision-language model $M _ { \theta }$ follows the shared prompt $\mathcal { P }$ and selects each action $a _ { t }$ using the history $H _ { t - 1 }$ of prior actions and observations. The history starts empty. Inspection gathers visual evidence for constructing or refining the state, while execution runs a program on the supplied state. Each action and its returned observation $o _ { t }$ are appended to the history, allowing the model to revise the state or program before choosing its next action. The loop ends when the model submits a final answer or exhausts either budget.

Each attempt allows at most B = 22 model calls within T = 1,800 s of active runtime. Table S1 summarizes these budgets and the limits on individual requests and executions. Each Python execution is limited to 3 seconds and 256 MB of memory. The program receives the supplied JSON state without access to the source image, while visual observations remain available to the model.

Table S1. Runtime settings. Request timeouts are also bounded by the remaining per-attempt time budget.
<table><tr><td>Setting</td><td>Limit</td></tr><tr><td>Model calls per attempt (B)</td><td>22</td></tr><tr><td>Active runtime per attempt (T)</td><td>1,800 s</td></tr><tr><td>Output tokens per model call</td><td>65,536</td></tr><tr><td>HTTP timeout per request</td><td>1,200 s</td></tr><tr><td>Python time per execution</td><td>3 s</td></tr><tr><td>Python memory per execution</td><td>256 MB</td></tr></table>

Step-error feedback. Tool errors are returned as observations for subsequent reasoning. Within a Vis2State attempt, a length-truncated response consumes a step and returns error feedback without executing partial content. Prior observations and execution results remain in the history, and the model can issue another action within the remaining call and time budgets. A final answer requires explicit submission by the model.

## S2 Vis2State Prompt

Table S2 summarizes the shared prompt and execution interface used across all datasets and models. It retains the state-construction, verification, execution, and submission requirements while omitting the full action schema and tool argument specifications.

## S3 Evaluation protocol

## S3.1 Answer extraction and scoring

We use each benchmark’s released answer-extraction and scoring procedures [9, 14, 34, 47]. Direct CoT follows the official CoT prompt for each benchmark and answers without tools. Its responses undergo answer extraction, whereas tool-assisted methods submit an option through the final-answer interface. For VisuLogic, DeepSeek V4.1 Flash replaces the released evaluator’s GPT-4o-mini because of model availability. PuzzleVQA averages its four single-concept and six dual-concept categories, weighting the group averages by 0.4 and 0.6, respectively.

Missing answers and attempt selection. Uniform random completion applies only to final records marked as having no prediction. Existing complete predictions are preserved, including incorrect or invalid answers. For each method–dataset pair, we reset the random seed to $0 ,$ order questions by ID, and sample uniformly from each question’s available answer options without consulting the reference answer. Valid incorrect answers are not retried.

## S3.2 Statistical comparisons

We compare Vis2State and Direct CoT on the same questions within each benchmark. For each question, we encode the difference in correctness as +1, −1, or 0. We compute the full empirical paired bootstrap distribution of the mean difference by convolving these three outcome probabilities over the dataset size. The 2.5th and 97.5th percentiles give the reported 95.0% confidence interval.

We test the paired disagreements with a two-sided exact McNemar test and apply Holm correction across the four benchmarks. These estimates reflect question-level uncertainty within a single run and do not account for correlations among questions sharing an image.

## S3.3 Sources of published comparisons

The sources below provide published scores for each benchmark. Their model versions and evaluation protocols differ from our experiments, so these results provide context for the comparisons between Vis2State and Direct CoT using the same model.

<table><tr><td>Vis2State prompt</td></tr><tr><td>Solve the puzzle from the image and question by constructing, checking, and solving an explicit state. State construction. Choose a minimal JSON object containing the facts, relations, rules, and objective needed to answer. Define its fields and structure yourself without a predefined puzzle schema. Distinguish observed facts from assumptions, and preserve relevant</td></tr><tr><td>coordinate conventions and action preconditions. Visual inspection. Use source-image tools to check answer-relevant facts in the proposed state. Compare their actual outputs and the</td></tr><tr><td>original image with the representation, and repair missing or inconsistent facts. A successful measurement does not establish semantic correctness. Resolve uncertainty that could change the answer before solving.</td></tr><tr><td></td></tr><tr><td>Execution and revision. Write Python defining solve (st ate) to read the checked state and return a JSON-compatible result. Put the JSON-encoded input object in st ate_json and the program in source. The program receives this object unchanged, without</td></tr><tr><td>image access. Copy observed values into the state rather than referring to measurement outputs. Verify the task rules and objective in</td></tr><tr><td>the computation. If the state changes, recompute affected results. Action format. Return one JSON action matching the supplied schema. Use measure to inspect, code to compute, final to</td></tr></table>

AlgoPuzzleVQA results. GPT-4V, Claude 3 Opus, and Qwen2-VL-7B are from [14]. Gemini 3 Pro Preview, o4-mini (high), o3 (high), GPT-5.2, and Qwen3-VL-8B-Instruct are from [37].

VisualPuzzles results. Human (50th Percentile), o4- mini, o3, Gemini 2.5 Pro, Claude 3.7 Sonnet, and Qwen2.5- VL-72B-Instruct are from [34]. GPT-5 (high), Qwen3- VL-235B-A22B (thinking/instruct), and Claude Opus 4.1 (thinking) are from [5]. Gemini 3.1 Pro and Qwen3-VL-8B-Instruct are from [40].

VisuLogic results. Human, o3 (20250417), GPT-4o (20240806), Doubao 1.6 Vision (250815), Gemini 2.0 Pro (20250205), Qwen2.5-VL-72B, and InternVL3-78B are from [47]. Gemini 3.1 Pro, GPT-5.5, and Claude Opus 4.7 are from [7]. Additional thinking/instruct/high-mode results are from [5].

PuzzleVQA results. CoT results for GPT-4V and Claude 3 Opus are from [9]. All other published results are from [37].

## S4 Detailed cross-model results

Tables S3–S6 compare Direct CoT and Vis2State using GLM-5.3-Flash-260828 [51] across all four benchmarks. Both methods use the evaluation protocol in Sec. S3. Vis2State uses the same prompt, tool instructions, and state interface as the DeepSeek experiments, with the shared prompt specified in Sec. S2. The tables report question counts, accuracy, and gains computed before rounding. PuzzleVQA also reports single- and dual-concept averages.

Vis2State improves 33 of 39 categories, ties five, and declines in one. The largest gains within the four benchmarks occur in Colour (+59.0 percentage points), Inductive reasoning (+17.7), Style (+26.7), and colors–size (+20.5), respectively. Benefits are not uniform. VisuLogic Spatiality remains at 29.9%, while Attribute decreases by 1.2 percentage points. PuzzleVQA Shapes improves to 74.5%, the lowest of its concept categories.

## S5 Supplementary analyses

## S5.1 Accuracy and output-token cost

We compare five configurations of DeepSeek V4.1 Flash. Direct CoT answers without tools, Inspection only provides visual tools, and Execution only provides Python execution and feedback. Inspection + Execution combines both tool types without requiring an explicit state. Vis2State adds state-construction guidance and the solve(state) interface.

For each dataset, we first identify questions with valid correct answers and complete usage records for all five methods. We then randomly sample 100 questions without replacement from this intersection (seed 0), using the same question IDs for all five methods. Figure S1 plots mean output tokens over these same IDs against full-benchmark accuracy. Output tokens include reasoning and all solving requests within each selected final attempt, including inspection, code, and step-error feedback. The cost measure is conditional on common correct answers, whereas accuracy is evaluated on the full benchmark.

Table S3. GLM-5.3-Flash on AlgoPuzzleVQA. Accuracy (%) across 18 task families. Bold marks the better result, including ties.
<table><tr><td>Method</td><td>Overall</td><td>Board</td><td>Calendar</td><td>Chain</td><td>Checker</td><td>Clock</td><td>Colour</td><td>Map</td><td>Maze</td><td>Move box</td></tr><tr><td>Questions</td><td>1,800</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Direct CoT</td><td>82.3</td><td>100.0</td><td>99.0</td><td>56.0</td><td>92.0</td><td>86.0</td><td>40.0</td><td>65.0</td><td>56.0</td><td>84.0</td></tr><tr><td>Vis2State (Ours)</td><td>94.5</td><td>100.0</td><td>100.0</td><td>74.0</td><td>100.0</td><td>100.0</td><td>99.0</td><td>71.0</td><td>92.0</td><td>98.0</td></tr><tr><td>Gain (pp)</td><td>+12.2</td><td>0.0</td><td>+1.0</td><td>+18.0</td><td>+8.0</td><td>+14.0</td><td>+59.0</td><td>+6.0</td><td>+36.0</td><td>+14.0</td></tr></table>

<table><tr><td>Method</td><td>Overall</td><td>Queens</td><td>Number</td><td>Fruits</td><td>Rubik</td><td>Think dot</td><td>Hanoi</td><td>Water</td><td>Wheel</td><td>Wood</td></tr><tr><td>Questions</td><td>1,800</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Direct CoT</td><td>82.3</td><td>100.0</td><td>96.0</td><td>99.0</td><td>95.0</td><td>70.0</td><td>70.0</td><td>97.0</td><td>85.0</td><td>92.0</td></tr><tr><td>Vis2State (Ours)</td><td>94.5</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.0</td><td>73.0</td><td>100.0</td><td>99.0</td><td>96.0</td><td>100.0</td></tr><tr><td>Gain (pp)</td><td>+12.2</td><td>0.0</td><td>+4.0</td><td>+1.0</td><td>+4.0</td><td>+3.0</td><td>+30.0</td><td>+2.0</td><td>+11.0</td><td>+8.0</td></tr></table>

Table S4. GLM-5.3-Flash on VisualPuzzles. Accuracy (%) across five reasoning categories. Bold marks the better result, including ties.
<table><tr><td>Method</td><td>Overall</td><td>Algorithmic</td><td>Analogical</td><td>Deductive</td><td>Inductive</td><td>Spatial</td></tr><tr><td>Questions</td><td>1,168</td><td>262</td><td>211</td><td>200</td><td>209</td><td>286</td></tr><tr><td>Direct CoT</td><td>67.2</td><td>80.9</td><td>81.0</td><td>83.5</td><td>35.4</td><td>56.3</td></tr><tr><td>Vis2State (Ours)</td><td>75.1</td><td>91.2</td><td>81.0</td><td>86.0</td><td>53.1</td><td>64.3</td></tr><tr><td>Gain (pp)</td><td>+7.9</td><td>+10.3</td><td>0.0</td><td>+2.5</td><td>+17.7</td><td>+8.0</td></tr></table>

From Figure S1, we observe that Vis2State achieves the highest full-benchmark accuracy in all four panels, while using 1.7–2.8× the mean output tokens of Inspection + Execution on these matched subsets. The accuracy gains therefore accompany additional output-token cost.

## S5.2 Error cohort and annotation

Diagnostic cohort. We analyze all 1,709 Direct CoT responses scored incorrect by the official evaluators: 319 from AlgoPuzzleVQA, 449 from VisualPuzzles, 645 from Visu-Logic, and 296 from PuzzleVQA. Vis2State answers 971 of these questions correctly through valid submissions, excluding random completions. The corresponding correction rates are 92.5%, 52.1%, 36.3%, and 70.3%, respectively.

Annotation and adjudication. We annotate perceptual and reasoning errors using an LLM-as-judge protocol with human adjudication. DeepSeek V4.1 Flash is the primary judge. Cases it cannot clearly classify are referred to GPT-6 Astra. The authors adjudicate cases that remain ambiguous. Judges receive the image, question, full model response, and reference option. Labels cover perception, representation, rule/inference, search/state updates, calculation, and answer comparison. Each label requires a concrete discrepancy supported by a response quotation. For reporting, Perception covers visual errors, State / Reasoning combines representation and rule/inference errors, and Search / Computation combines search/state updates, calculation, and answer comparison. Compound includes cases spanning multiple groups, answer-extraction failures, and otherwise unattributed cases.

## S5.3 State intervention protocol

We test whether saved programs use the supplied state by changing the state while holding the program fixed.

Subset selection and composition. We select 200 questions each from AlgoPuzzleVQA and PuzzleVQA for which the last saved DeepSeek Vis2State program correctly solves the original state and independent reference programs can certify the intervention answers. Original-state accuracy is 100.0% by construction. Selection does not require the saved program to solve the counterfactual or control correctly.

AlgoPuzzleVQA includes water jugs (61), color rearrangement (39), Tower of Hanoi (31), checker moves (26), rotting kiwi (22), N-queens (17), and board tiling (4). PuzzleVQA includes number grids (55), color grids (32), size grids (30), Venn diagrams (28), color hexagons (21), number triangles (17), polygon side counts (12), and number– color associations (5).

Interventions and independent verification. Each question has three state conditions: the original state, an answerchanging counterfactual, and an answer-preserving control. Both interventions modify task-relevant values, positions, or configurations; repeated representations of a changed attribute are updated consistently. Independent reference programs verify that the original answer matches the benchmark reference, the counterfactual is valid and maps to a different existing answer option, and the control preserves the original answer. These checks establish the target answers independently of the saved program under evaluation.

Table S5. GLM-5.3-Flash on VisuLogic. Accuracy (%) across six reasoning categories. Bold marks the better result, including ties.
<table><tr><td>Method</td><td>Overall</td><td>Quantity</td><td>Spatiality</td><td>Position</td><td>Attribute</td><td>Style</td><td>Other</td></tr><tr><td>Questions</td><td>1,000</td><td>353</td><td>231</td><td>136</td><td>82</td><td>90</td><td>108</td></tr><tr><td>Direct CoT</td><td>36.2</td><td>37.7</td><td>29.9</td><td>45.6</td><td>39.0</td><td>24.4</td><td>40.7</td></tr><tr><td>Vis2State (Ours)</td><td>44.2</td><td>47.0</td><td>29.9</td><td>55.9</td><td>37.8</td><td>51.1</td><td>50.0</td></tr><tr><td>Gain (pp)</td><td>+8.0</td><td>+9.3</td><td>0.0</td><td>+10.3</td><td>-1.2</td><td>+26.7</td><td>+9.3</td></tr></table>

Table S6. GLM-5.3-Flash on PuzzleVQA. Accuracy (%) across single- and dual-concept categories. Bold marks the better result, including ties. N/C/S/Z denote numbers/colors/shapes/size.
<table><tr><td></td><td></td><td colspan="5">Single-concept</td><td colspan="7">Dual-concept</td></tr><tr><td>Method</td><td>Overall</td><td>N</td><td>C</td><td>Z</td><td>S</td><td>Avg.</td><td>N+S</td><td>C+N</td><td>N+Z</td><td>C+S</td><td>Z+S</td><td>C+Z</td><td>Avg.</td></tr><tr><td>Questions</td><td>2,000</td><td>200</td><td>200</td><td>200</td><td>200</td><td>800</td><td>200</td><td>200</td><td>200</td><td>200</td><td>200</td><td>200</td><td>1,200</td></tr><tr><td>Direct CoT</td><td>89.4</td><td>93.0</td><td>100.0</td><td>95.0</td><td>68.5</td><td>89.1</td><td>91.0</td><td>99.0</td><td>85.0</td><td>86.0</td><td>98.5</td><td>78.0</td><td>89.6</td></tr><tr><td>Vis2State (Ours)</td><td>96.3</td><td>99.0</td><td>100.0</td><td>100.0</td><td>74.5</td><td>93.4</td><td>99.5</td><td>99.5</td><td>94.5</td><td>97.5</td><td>100.0</td><td>98.5</td><td>98.3</td></tr><tr><td>Gain (pp)</td><td>+6.9</td><td>+6.0</td><td>0.0</td><td>+5.0</td><td>+6.0</td><td>+4.3</td><td>+8.5</td><td>+0.5</td><td>+9.5</td><td>+11.5</td><td>+1.5</td><td>+20.5</td><td>+8.7</td></tr></table>

Replay and scoring. We replace only the supplied state, keeping the program, answer options, and execution limits fixed, with no new model inference. The 400 questions yield 1,200 condition records. Each output is scored against the independently verified answer for its state condition; when both a value and an option are returned, both must be correct. All intervention failures remain in the denominator, and no question is replaced because of an incorrect intervention output.

Results. Counterfactual accuracy is 98.0% (196/200) on AlgoPuzzleVQA and 93.5% (187/200) on PuzzleVQA. Control accuracy is 100.0% (200/200) and 98.5% (197/200), respectively. Of the 17 counterfactual failures, 9 compute the correct new value but return an inconsistent or missing option; the remaining 8 return incorrect or missing answers. Independent verification certifies the intervention answers, but does not guarantee that the saved program produces them. On these original-correct subsets, the results show that programs usually respond correctly to answer-changing state edits while preserving correctness under answer-preserving edits.

## S6 Qualitative examples

We first examine how inspection and execution contribute to recorded solutions, then present task-dependent states and the computations they support. The model generates these states while solving each question, determining their fields, structure, and values without a predefined puzzle-specific schema.

## S6.1 Inspection and execution in recorded solutions

Figure S2 illustrates the complementary roles of inspection and execution through three recorded solutions. In Colour Hue, color probes recover tile correspondences and reveal a six-tile cycle, allowing the model to infer five swaps without code execution. In Maze, a program searches the grid state for the path with the fewest visited cells, using the number of turns to break ties. In WEEKLY, the model revises the letter coordinates and searches the resulting state for the shortest walk spelling the word. Together, these examples show how inspection helps establish an accurate problem state, while execution supports systematic search over that state.

## S6.2 Task-dependent state representations

The following six excerpts show model-generated states from recorded trajectories across all four benchmarks. They cover configurations and legal moves, spatial occupancy, discriminative attributes, and measured quantities. Each pairs a state excerpt with the computation it supports.

Water-jug quantities and target configurations. In AlgoPuzzleVQA, each list position refers to the same jug. For example, the first jug has capacity 11, initially contains 1 unit, and must finish with 5 units. All entries use the puzzle’s volume unit.

```json
{"capacities": [11,9,8,4,3],
"initial": [1,1,8,3,1],
"target": [5,0,3,3,3]}
```

The program uses capacities to determine legal pours and searches from the initial to the target configuration. The recorded solution takes four moves.

![](images/7c06140c25fd0d8866c9efeb309f46c1a35ae0066680f0f3e955c7827c22f80e.jpg)  
Figure S1. Accuracy and output-token cost with DeepSeek V4.1 Flash. Full-benchmark accuracy is plotted against mean output tokens on sampled questions. Panels use a shared logarithmic cost scale and accuracy ranges spanning 20 percentage points. Stars denote Vis2State, dashed lines indicate Direct CoT accuracy, and annotations show the corresponding accuracy gains.

Maze connectivity. In AlgoPuzzleVQA, each string is one grid row. The symbol # denotes a wall, . a free cell, and G/B the entrance/exit. The excerpt shows the first and last rows. The intervening rows follow the same encoding.

{"grid": ["#######",   
"G...#.#",   
...,   
"#.....B",   
"#######"]}

The program searches adjacent free cells, minimizing visited cells and then turns. It returns 23 cells and eight turns, including the entrance and exit.

Objects and movement constraints in Move Box. In this AlgoPuzzleVQA Move Box example, the task is to reach the goal using the fewest box pushes. Locations use zero-based (row, column) coordinates. The wall at (1,3) lies directly above the box at (2,3), preventing the player from standing there to push downward. The omitted wall entries describe the grid boundary.

```jsonl
{"grid_size": [6,6],
"box": [2,3], "goal": [3,4],
"player": [4,4],
"walls": [..., [1,3]]}
```

The program tracks both box and player positions and checks player reachability before each push. The minimum is four pushes.

Occupied positions in pattern overlay. In VisualPuzzles, each pattern is a set of black-cell positions in a 3 × 3 grid, using zero-based (row, column) coordinates. Here A, B, and C identify the three input patterns, not answer choices.

```json
{"patterns": {
"A": [[0,0],[0,1],[2,1],[2,2]],
"B": [[0,1],[0,2],[2,0]],
"C": [[1,1],[1,2],[2,0],[2,1]]}}
```

For aligned overlay, the program takes the union of these sets. A cell covered by multiple patterns is counted once. The union contains eight cells, matching the correct answer.

Discriminative attributes for figure grouping. In Visu-Logic, id identifies a figure and segments records its number of straight line segments. The six recorded counts are 12, 10, 12, 10, 12, and 10. Two entries are shown below.

```jsonl
{"panels": [
{"id":"1", "segments":12},
{"id":"2", "segments":10}, ...]}
```

Grouping equal counts yields {1, 3, 5} and {2, 4, 6}. The program matches this partition to the correct option B.

Bar heights as visual measurements. In PuzzleVQA, each bar has a visible label and top/bottom pixel coordinates. The symbol ? marks the missing label. The excerpt retains two bars labeled 2 and the query bar. Height is computed as bottom minus top, so absolute vertical position does not determine the label.

```jsonl
{"bars": [ ...,
{"label":"2", "top":132, "bottom":381},
...,
{"label":"2", "top":128, "bottom":381},
{"label":"?", "top":132, "bottom":381}]}
```

The program compares the query height with the mean height for each known label. The query is 249 pixels tall. The two bars labeled 2 average 251 pixels, making 2 the nearest match (option A).

A: 42 B: 41 C: 23 D: 39

## (a) Inspection: recover subtle color correspondences

Colour Hue · AlgoPuzzleVQA · Task accuracy: 49.0% → 100.0% (+51.0 pp)

Question: How many swaps restore board (A) to board (B)? Any two tiles may be swapped.

Original puzzle

![](images/493b5a1e78fc6562a49daf68f42f8b9c94f2beb9ad3c80d99ac89cd974917b7c.jpg)  
Direct CoT  
Identifies one 5-cycle; misses the tile at (5, 2).  
A: 1 B: 2 C: 5 D: 4  
Applies the correct formula to the incomplete cycle: 5 − 1 = 4 swaps.

![](images/de17062a2502ef78d77c0907f865bf95eadd84942595de82bd24a0dcae9ea488.jpg)  
Probe colors → match tiles  
RGB matches reveal one 6-cycle. 24 of 30 tiles are already correct.  
6 − 1 = 5 swaps The answer follows from inspection.

## Vis2State

![](images/6d4a8be21042e5413e4d34352d50a26a11f4ab1a93f5b06d1a90297cd4fc1f6e.jpg)

## (b) Execution: compute the shortest path explicitly

Maze · AlgoPuzzleVQA · Task accuracy: 41.0% → 100.0% (+59.0 pp)

Question: Find the path with the fewest cells, then fewest turns. How many cells are visited, including entrance and exit?

## Original puzzle

Direct CoT

![](images/5623cd5909f3e9278087f836a5c64fa427d6867a56dd4550ce2807762c244a75.jpg)  
Reports 39 cells.  
No intermediate path is provided.  
D: 39 — incorrect

## Vis2State

Grid state → shortest-path search

Represent walls and free cells.   
Minimize (visited cells, turns). Final search: 23 cells, 8 turns.   
Count includes entrance and exit.

C: 23 — correct

## (c) Inspection + Execution: revise coordinates and search the route

WEEKLY · VisualPuzzles

Question: Walk along the grid to spell WEEKLY in order.   
What is the shortest distance? Each grid interval is one unit.

## Original puzzle

![](images/1f3e85dad1a24108723ccb6db9cc7bb5bb3dcbb83b57085e404ae3ac522bbdc5.jpg)  
Direct CoT  
A: 26 B: 24 C: 28 D: 23

Misplaces W, E₁, and L: W (0, 4), E₁ (3, 3), L (6, 4).

4 + 3 + 3 + 4 + 9 = 23

D: 23 — incorrect

## Vis2State

## Final revised state → BFS

W (0, 5), E₁ (2, 3), L (6, 5).   
Search over (x, y, next letter).

A: 26 — correct

Figure S2. Inspection, execution, and their combination. (a) Color probes refine tile correspondences into a six-tile cycle. (b) Search over the maze state computes the shortest path. (c) Revised coordinates and search yield the shortest WEEKLY walk. Task accuracies in (a,b) compare Direct CoT with Vis2State over 100 questions each. Only decisive steps are shown. Coordinates are one-based (row, column) in (a) and zero-based (x, y) from the top-left in (c).