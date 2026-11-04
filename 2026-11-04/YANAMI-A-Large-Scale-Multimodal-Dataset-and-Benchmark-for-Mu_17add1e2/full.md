# YANAMI: A Large-Scale Multimodal Dataset and Benchmark for Multiturn Stateful Interaction

Shengyu Wang<sup>1\*†</sup>, Tengyue Nong<sup>2\*</sup>, Junming Liu<sup>3\*</sup>, Haosong Peng<sup>1\*</sup>, Siyu Li<sup>4\*</sup>, Ruiqi Lei<sup>5</sup>, Fushuo Huo<sup>6</sup>, Yuhao Pan<sup>1</sup>, Yunfeng Fan<sup>7</sup>, Ding Wang<sup>8</sup>, Nan Cheng<sup>9</sup>, Haozhao Wang<sup>10</sup>, Wenchao Xu<sup>1‡</sup>

<sup>1</sup> The Hong Kong University of Science and Technology <sup>2</sup> Independent Researcher <sup>3</sup> The University of Melbourne <sup>4</sup> The Chinese University of Hong Kong, Shenzhen <sup>5</sup> Imperial College London <sup>6</sup> Southeast University <sup>7</sup> The Hong Kong Polytechnic University <sup>8</sup> Shanghai Artificial Intelligence Laboratory <sup>9</sup> Xidian University <sup>10</sup> Huazhong University of Science and Technology

Project page: https://rowletqwq.github.io/YANAMI

![](images/6ec8b1ab16e4aee043b32d8c38b08ab3583797d1efd95aeba3e94edc7191c139.jpg)  
Figure 1: YANAMI: the first large-scale multimodal dialogue dataset and benchmark for stateful interaction.

## Abstract

Unified Multimodal Models (UMMs) integrate visual understanding, generation, and editing within a single unified framework. However, existing unified multimodal datasets and benchmarks remain dominated by single-turn tasks. Even in recent multi-turn datasets, many turns can be solved independently without tracking information across turns. Consequently, they provide limited supervision and evaluation for multimodal stateful interaction, which requires models to understand, maintain, update, and faithfully realize evolving dialogue states across turns. To address this gap, we introduce YANAMI (Your Assistant Needs to Assimilate Multiturn Interaction), the first large-scale multimodal dialogue dataset and benchmark for stateful interaction, comprising YANAMI-400K and YANAMI-Bench. YANAMI-400K contains 400K multi-turn multimodal dialogues, with more than 2M images and 2.6M dialogue turns. The dialogues span four representative scenarios, Multi-Image Fusion, Image Editing, World Modeling, and Visual Reasoning, and require models to continuously track, preserve, and revise evolving multimodal dialogue states across turns. YANAMI-Bench comprises 400 multi-turn dialogues for systematically evaluating multimodal stateful interaction. It

jointly evaluates Dialogue State Understanding and Visual Realization, characterizing the gap between models’ ability to understand evolving dialogue states and their ability to faithfully realize them in visual outputs. Experiments show that fine-tuning BAGEL on YANAMI-400K improves image editing, generation, and multimodal stateful interaction by 10.3%, 2.0%, and 13.23%, respectively, while preserving its general visual understanding capabilities. Evaluations on YANAMI-Bench further reveal that existing UMMs still struggle to maintain, update, and visually realize evolving multimodal dialogue states. Together, YANAMI provides a unified foundation for training and evaluating UMMs toward robust multimodal stateful interaction.

## 1 introduction

Unified Multimodal Models (UMMs) (Deng et al., 2025; Wu et al., 2025c; Xie et al., 2026a) represent an emerging shift from task-specific systems toward unified visual modeling. Unlike conventional models designed for individual tasks (Liu et al., 2023; Rombach et al., 2022), UMMs integrate image understanding, generation, and editing within a single framework. This unified modeling enables UMMs to interleave processing across textual and visual modalities, highlighting their potential for complex multi-turn multimodal interaction.

However, most publicly available datasets focus on single-turn tasks (Chen et al., 2025a; Wang et al., 2025b). Only a few support multi-turn interaction (Chow et al., 2026b; Feng et al., 2026), and even these often contain turns that can be solved independently without tracking information across turns, falling short of complex multi-turn multimodal interaction. In contrast, real-world user-assistant image creation and editing are typically not completed in a single step, but instead involve continuous multi-turn stateful interaction for gradual refinement.

Here, stateful interaction refers to a multi-turn dialogue in which the user and assistant discuss alternatives, make decisions, establish cross-turn constraints and dependencies, plan subsequent steps, and revise previous decisions or requirements in response to user corrections. Consequently, datasets without such interactions provide limited supervision for learning to understand, maintain, update, and faithfully realize evolving multimodal dialogue states across turns. It further limits the exploration of UMMs for complex multi-turn multimodal interaction. For example, as illustrated in Fig. 1, existing multi-turn datasets typically alternate between user instructions and assistant edits, with each turn treated independently and lacking cross-turn contextual dependencies, whereas our dataset captures stateful user–assistant interactions in which dialogue states are established, maintained, and updated through ongoing context-dependent exchanges.

To address these limitations, we introduce YANAMI, the first large-scale multimodal dialogue dataset and benchmark for stateful interaction, comprising YANAMI-400K and YANAMI-Bench to capture evolving multimodal dialogue states and evaluate models’ performance across understanding, generation, and editing. YANAMI-400K contains 400K multi-turn multimodal dialogues, comprising more than 2M images and 2.6M dialogue turns in total. To construct genuinely stateful interactions, we explicitly incorporate six key capabilities into each dialogue: 1) Reference Resolution; 2) Context Retrieval; 3) Consistency Maintenance; 4) Spatial and Physical Consistency; 5) Constraint Keeping; and 6) Correction-Based State Updating. Together, these capabilities make stateful interaction explicit in the data, with each dialogue requiring multimodal states to be continuously tracked, preserved, and revised across turns. To instantiate such stateful interaction in diverse settings, we construct dialogues under four representative scenarios: 1) Multi-Image Fusion; 2) Image Editing; 3) World Modeling; and 4) Visual Reasoning. These scenarios jointly exercise diverse capabilities, spanning visual-state composition, modification, transition, and reasoning.

YANAMI-Bench comprises 400 multi-turn dialogues designed to systematically evaluate models’ ability to track, maintain, and update multimodal dialogue states throughout interactions. Specifically, YANAMI-Bench covers the same four scenarios, including complex understanding, generation, and editing tasks, with each dialogue jointly involving all six key capabilities and requiring their coordinated use throughout the interaction. YANAMI-Bench evaluates models along two complementary dimensions: Dialogue State Understanding (DSU) and Visual Realization (VR). DSU evaluates whether a model can infer the currently valid multimodal state from the full interaction history. We assess it through capabilityoriented dialogue-based visual question answering along four dimensions: 1) Reference Resolution; 2) Context Retrieval; 3) Constraint Keeping; and 4) Correction-Based State Updating. VR measures whether the generated images faithfully realize the current dialogue state and is assessed by Visual Quality and State Consistency. Together, these two dimensions provide a unified evaluation of stateful multimodal interaction.

Our experiments demonstrate that training on YANAMI-400K substantially improves model performance on image generation, image editing, and multi-turn stateful interaction tasks while maintaining general visual understanding capabilities. Meanwhile, evaluations on YANAMI-Bench reveal several key limitations of current models in long-horizon multimodal interactions. We hope that YANAMI provides a unified foundation for training and evaluating UMMs toward robust multimodal stateful interaction. In summary, our main contributions are as follows:

• We introduce YANAMI-400K, the first large-scale multimodal dialogue dataset for stateful interaction across understanding, generation, and editing. It comprises 400K multi-turn dialogues with more than 2M images and 2.6M dialogue turns, providing large-scale supervision for six key capabilities required for stateful interaction.

• We construct YANAMI-Bench, a benchmark of 400 multi-turn dialogues. By jointly evaluating DSU and VR, YANAMI-Bench characterizes the gap between models’ ability to understand dialogue states and their ability to faithfully realize them in visual outputs.

• Fine-tuning BAGEL (Deng et al., 2025) on YANAMI-400K yields consistent gains in general image generation and editing, as well as DSU and VR, while preserving general visual understanding capabilities. Results on YANAMI-Bench reveal that existing UMMs still face substantial challenges in maintaining, updating, and visually realizing evolving multimodal dialogue states.

## 2 related work

Unified Multimodal Models aim to integrate visual understanding and generation within a shared architecture, providing a unified interface for processing and producing multimodal content. Existing UMMs explore diverse architectural paradigms, ranging from unified autoregressive modeling(Team, 2024; Wu et al., 2025b) over multimodal representations to hybrid frameworks that combine autoregressive language modeling with diffusion- or flow-based visual generation (Deng et al., 2025; Wu et al., 2025c). Existing UMMs have demonstrated increasing capabilities across single-turn multimodal understanding, generation, and editing tasks (Chen et al., 2025d; Cui et al., 2025). However, how well these capabilities extend to multi-turn multimodal dialogues with evolving contextual dependencies remains largely underexplored.

Multimodal Data and Benchmarks for Understanding, Generation, and Editing. Existing multimodal datasets primarily provide single-turn supervision for individual capabilities such as understanding (Liu et al., 2023), generation (Brooks et al., 2023), and editing (Yu et al., 2025), with some extending to multi-image settings (Chen et al., 2025c; Li et al., 2025). Recent work has further explored multi-turn interaction through sequential editing and interleaved dialogues (Chow et al., 2026b; Qian et al., 2026; Dai et al., 2026). Meanwhile, existing benchmarks largely evaluate individual capabilities in isolation, including understanding (Fu et al., 2026; Liu et al., 2024b; Li et al., 2023), generation (Ghosh et al., 2023), and editing (Liu et al., 2025). More recent efforts have begun to evaluate unified or interleaved capabilities (Xia et al., 2025; Xie et al., 2026b) and multi-turn settings spanning understanding, editing, generation, and unified multimodal interaction (Liu et al., 2024a; Chen et al., 2025b; Huang et al., 2025; Chow et al., 2026b). However, these datasets and benchmarks can often be approached turn by turn without explicitly tracking evolving information across the dialogue, limiting their ability to support and evaluate truly multimodal stateful interaction. In contrast, YANAMI jointly provides large-scale supervision and systematic evaluation for tracking, updating, and faithfully realizing evolving multimodal dialogue states across turns. The comparison between YANAMI and multimodal datasets is provided in Appendix A.1.

## 3 YANAMI

In this section, we present YANAMI, our dataset and benchmark for multimodal stateful interaction. YANAMI-400K provides large-scale training data, while YANAMI-Bench evaluates models across understanding, generation, and editing. We first describe the data construction process in Sec. 3.1, followed by the evaluation setting and metrics in Sec. 3.2 and the dataset statistics in Sec. 3.3.

## 3.1 Data Pipeline

YANAMI-400K. We construct novel stateful dialogues using visual materials drawn from four existing public datasets (Li et al., 2025; Chow et al., 2026b; Wei et al., 2026b; Chen et al., 2026). Our pipeline, illustrated in Fig. 3, leverages GPT-5.6 Sol (OpenAI, 2026a), Qwen3.7-Plus (Alibaba Cloud, 2026), Gemini 3.1 Pro (Google, 2026b), GPT-Image-2 (OpenAI, 2026b), and Nano Banana 2 (Google, 2026a) for data filtering, dialogue construction, and quality control. Specifically, raw images are filtered into four scenario-specific pools (Fig 3.a) and combined with scenario- and capability-oriented prompts to construct multi-turn multimodal stateful dialogues (Fig 3.b). The resulting dialogues are automatically verified for logical coherence and image–text consistency, with failed samples iteratively corrected and regenerated. Dialogues that pass verification form YANAMI-400K (Fig 3.c). The constructed dialogues span four representative scenarios: 1) Multi-imagefusion, where the assistant integrates multiple userprovided or previously generated images according to the dialogue history; 2) Image editing, where the assistant progressively edits an image while preserving prior modifications and visual consistency; 3) World modeling, where the assistant predicts and renders scene states under specified viewpoint changes or actions; and 4) Visual reasoning, where the assistant generates visual outcomes for proposed hypotheses or solutions to support step-by-step reasoning.

To make these interactions genuinely stateful, we incorporate six key state-management capabilities across the four scenarios: (i) Coreference resolution, which grounds context-dependent expressions to the correct textual or visual entities; (ii) Dialogue-history retrieval, which identifies and uses relevant information from previous turns; (iii) Cross-turn consistency, which preserves established visual attributes and prior edits across turns; (iv) Spatial and physical consistency, which maintains plausible geometry, physical relations, and state transitions; (v) Constraint adherence, which consistently follows user-specified requirements and restrictions; and (vi) Correction-based state updating, which incorporates user corrections and replaces outdated information accordingly. For example, Fig. 2 presents representative dialogues across the four scenarios, illustrating how multimodal dialogue states are progressively established, maintained, and updated through the joint use of these capabilities. Additional details of the data collection and representative examples are provided in Appendices A and E, respectively.

YANAMI-Bench. YANAMI-Bench is constructed with LLM assistance, followed by exhaustive human verification and correction by annotators with graduate-level training in STEM disciplines (Fig 3.d). Throughout data construction, we strictly isolate the benchmark from the training dataset to prevent data leakage; as a final safeguard, we compare the SHA-256 hashes of all images across YANAMI-Bench and YANAMI-400K, confirming that no exact duplicates exist between them. It comprises 400 multi-turn dialogues covering the four scenarios described above. The benchmark contains complex instructions that require reasoning over cross-turn context, jointly evaluating models’ capabilities in ability to track, maintain, and update multimodal dialogue states and translating the understanding of the state into visual implementation.

## 3.2 Evaluation Setting and Metrics

To better quantify models’ stateful interaction capabilities in unified multi-turn multimodal dialogues, we evaluate them from two complementary perspectives: dialogue state understanding and visual realization. Specifically, we adopt a VLM-as-a-Judge framework for automatic evaluation.

For DSU, we construct four types of QA pairs for each dialogue: 1) Reference Resolution measures whether the model can correctly resolve referring expressions in the current instruction and associate them with the corresponding entities in the dialogue history or previous images. 2) Context Retrieval measures whether the model can retrieve visual or textual information relevant to the current task from the multi-turn interaction history. 3) Constraint Tracking evaluates whether the model can identify and continuously retain constraints introduced in previous turns that remain valid at the current turn; and 4) Correction Updating measures whether the model can discard invalidated states and update the multimodal dialogue state after the user corrects previously established information, requirements, or states.

![](images/8c59eab8d858c12efab9713c8ce98e7f2e05fbbc91cd21f7428f327b26b6668d.jpg)  
Figure 2: Representative examples from YANAMI-400K. The four columns present evolving multimodal dialogues in four interaction scenarios: (a) multi-image fusion, (b) image editing, (c) world modeling, and (d) visual reasoning. Across turns, subsequent responses retrieve, maintain, and update the accumulated dialogue state. Colored text highlights six key capabilities.

![](images/6ae2f4d0bd3ca40eb957d6c766bb941f7dbb38f92d148cb579050040e07b45c3.jpg)  
Figure 3: Overview of the YANAMI data pipeline. The pipeline comprises four stages: (a) Data Preparation, (b) Data Construction, (c) Automatic Verification, and (d) Human Verification.

<table><tr><td>Statistic</td><td>Number</td></tr><tr><td>Total Dialogues</td><td>402,639</td></tr><tr><td>≥ 4 Images</td><td>321,868</td></tr><tr><td>≥ 5 Images</td><td>215,027</td></tr><tr><td>≥ 6 Images</td><td>138,669</td></tr><tr><td>Total Images</td><td>2,047,236</td></tr><tr><td>Avg. / Dialogue</td><td>5.08</td></tr><tr><td>Total Turns</td><td>2,643,291</td></tr><tr><td>Avg. / Dialogue</td><td>6.56</td></tr><tr><td>Max. / Dialogue</td><td>43</td></tr></table>

Table 1: Key Statistics of our Dataset.  
![](images/313045e671825154324e41049fa597360de2a84d53e281218451a408f07984ae.jpg)

![](images/54bb12bc7d7b8e8026e1562943781f385dae29d41d94e9579b08c5bef1542555.jpg)

![](images/2441ee0524762b526892fc5681b06c368e76825cdebe6443143f34c38cac1f14.jpg)  
Figure 4: Distribution of the four interaction scenarios Figure 5: Evaluation reliability in the training and test sets. of Our VLM-Judge.

For VR, we evaluate generated or edited visual outputs from two aspects: Visual Quality, which measures overall perceptual quality, and State Consistency, which uses fine-grained atomic criteria to assess whether the output faithfully reflects all valid multimodal dialogue states accumulated up to the current turn. For each scenario, we first average the four DSU metrics and then combine the resulting understanding score with State Consistency and Visual Quality to obtain the aggregate score. For models without understanding capabilities, the aggregate score is computed from State Consistency and Visual Quality alone. The detailed evaluation templates and score calculation are provided in Appendix F and Appendix B.3, respectively.

## 3.3 Data Statistics

Table 1 summarizes the key dataset statistics. Most dialogues contain at least four images and have an average of 6.56 conversational turns. As shown in Fig. 4, the training set includes all four interaction scenarios in varying proportions, while the test set is evenly balanced across them to enable controlled cross-scenario evaluation. Detailed statistics are provided in Appendix A.6.

## 4 Experiment

In this section, we first evaluate 15 models on YANAMI-Bench, revealing substantial limitations of existing models in understanding, maintaining, updating, and visually realizing multimodal dialogue

Table 2: Results on YANAMI-Bench across four interaction scenarios. denotes models supporting image generation/editing only; denotes models supporting both understanding and generation. denotes concatenated image inputs; denotes placing all available images at the beginning; and denotes interleaved image–text inputs. Red, blue, and yellow cells highlight the best results among closed-source models, open-source image-editing models, and UMMs. Underlined results denote the second-best performance within each model group.
<table><tr><td>Index</td><td>Model</td><td>Size</td><td>Modality</td><td>Format</td><td>Fusion</td><td>Editing</td><td>Modeling</td><td>Reasoning</td><td>Overall</td></tr><tr><td>1</td><td>GPT-Image-2 (OpenAI, 2026b)</td><td></td><td></td><td>S</td><td>86.78</td><td>90.54</td><td>81.40</td><td>73.46</td><td>83.05</td></tr><tr><td>2</td><td>Nano Banana 2 (Google, 2026a)</td><td></td><td></td><td>O</td><td>84.73</td><td>87.32</td><td>74.74</td><td>76.93</td><td>80.93</td></tr><tr><td>3</td><td>Step1X-v1p2 (Liu et al., 2025)</td><td>12B</td><td></td><td>由由</td><td>56.08</td><td>63.36</td><td>51.66</td><td>48.00</td><td>54.77</td></tr><tr><td>4</td><td>Qwen-Edit (Wu et al., 2025a)</td><td>20B</td><td></td><td>O</td><td>49.66</td><td>54.70</td><td>28.14</td><td>37.72</td><td>42.55</td></tr><tr><td>5</td><td>FLUX.2 [dev] (Labs, 2025)</td><td>32B</td><td></td><td>一由</td><td>72.44</td><td>74.54</td><td>65.54</td><td>56.68</td><td>67.30</td></tr><tr><td>6</td><td>UniPic 3.0 (Wei et al., 2026a)</td><td>20B</td><td></td><td>一由</td><td>63.44</td><td>70.66</td><td>56.96</td><td>48.82</td><td>59.97</td></tr><tr><td>7</td><td>OmniGen (Xiao et al., 2025)</td><td>3.8B</td><td>C</td><td>NENE</td><td>48.24</td><td>58.12</td><td>41.82</td><td>33.94</td><td>45.53</td></tr><tr><td>8</td><td>OmniGen2 (Wu et al., 2025c)</td><td>7B</td><td>P</td><td>ONO</td><td>50.10</td><td>51.70</td><td>55.12</td><td>47.62</td><td>51.13</td></tr><tr><td>9</td><td>Janus-Pro (Chen et al., 2025d)</td><td>7B</td><td></td><td>ONO</td><td>45.94</td><td>41.94</td><td>42.56</td><td>31.86</td><td>40.58</td></tr><tr><td>10</td><td>Show-o2 (Xie et al., 2026a)</td><td>7B</td><td></td><td>一</td><td>40.82</td><td>45.12</td><td>41.78</td><td>29.42</td><td>39.29</td></tr><tr><td>11</td><td>Ovis-U1 (Wang et al., 2025a)</td><td>3B</td><td>足</td><td>一由</td><td>68.72</td><td>46.24</td><td>57.72</td><td>43.42</td><td>54.03</td></tr><tr><td>12</td><td>Emu3.5 (Cui et al., 2025)</td><td>34B</td><td></td><td>日N</td><td>74.48</td><td>60.10</td><td>74.06</td><td>60.98</td><td>67.41</td></tr><tr><td>13</td><td>InternVL-U (Tian et al., 2026)</td><td>4B</td><td>网</td><td>一</td><td>66.72</td><td>66.12</td><td>66.48</td><td>46.52</td><td>61.46</td></tr><tr><td>14</td><td>UniAR (Peng et al., 2026)</td><td>10B</td><td>网</td><td>OAO</td><td>57.06</td><td>41.82</td><td>41.80</td><td>35.92</td><td>44.15</td></tr><tr><td>15</td><td>BAGEL (Deng et al., 2025) + YANAMI-400K</td><td>7B 7B</td><td>网</td><td>OAO NEAE</td><td>62.44 78.98↑</td><td>67.40 84.96↑</td><td>68.16 73.26↑</td><td>62.06 75.80↑</td><td>65.02 78.25↑</td></tr></table>

states. We then validate the effectiveness and high quality of YANAMI-400K by fine-tuning BAGEL in Sec. 4.2. Finally, we evaluate the reliability of our VLM-as-a-Judge evaluation framework.

## 4.1 YANAMI-Bench

Settings. We evaluate 2 close-source models, 4 open-source image editing models, and 9 unified multimodal models on YANAMI-Bench, with the results reported in Table 2. Complete capability-level results for each interaction scenario are provided in Appendix D.2. To isolate models’ ability to leverage multi-turn dialogue context from error propagation caused by previous generations, we adopt a GT-History evaluation setting. For image placement, we consider three configurations depending on the model setup: (1) yes-first, where each image is inserted at its first-mentioned position in the dialogue, (2) yes-front, where all images are placed at the beginning of the input, and (3) concatenate, where all images are concatenated into one image for models that cannot directly process sequential multi-image inputs. Implementation details for the history settings and input organizations are provided in Appendix B.4. Based on the results, we can derive the following key findings:

Stateful Multi-Turn Multimodal Interaction Remains Challenging. As shown in Table 2, the best average scores achieved by existing UMMs and open-source image editing models on YANAMI-Bench are only 67.41 and 67.30, respectively, indicating that current models still face substantial challenges in multimodal multi-turn stateful interaction. Further examination across different interaction scenarios reveals clear performance variation, with Visual Reasoning yielding the lowest scores for most evaluated models. In contrast, GPT-Image-2 and Nano Banana 2 achieve average scores of 83.05 and 80.93, respectively, outperforming current open-source models overall and suggesting that a noticeable gap remains between open- and closed-source models in multimodal multi-turn stateful interaction.

## Multi-turn Dialogue-State Understanding Remains Challenging.

We further find that models struggle not only with generating and editing visual outputs, but also with understanding the current multimodal dialogue state. As shown in Table 3, the average DSU scores of most UMMs remain below 82, indicating that they still have difficulty accurately recovering and maintaining the currently valid dialogue state as it evolves over multiple turns, including historical references, relevant contextual information, persistent constraints, and state updates following user corrections.

Table 3: Selected average DSU results. An upward arrow indicates an improvement over BAGEL. Full results are provided in Appendix D.2.
<table><tr><td>Model</td><td>Fus.</td><td>Edi.</td><td>Mod.</td><td>Rea.</td><td>Avg.</td></tr><tr><td>Show-02</td><td>74.00</td><td>64.25</td><td>79.50</td><td>54.00</td><td>67.94</td></tr><tr><td>BAGEL</td><td>85.75</td><td>76.50</td><td>85.25</td><td>78.25</td><td>81.44</td></tr><tr><td>BAGEL + Ours</td><td>96.25↑</td><td>98.00↑</td><td>95.50↑</td><td>93.00↑</td><td>95.69↑</td></tr></table>

![](images/395c245f9b82b66133c74909bf4f360f7248f93498e48f59965b54715c91ff40.jpg)  
(a) Und. Realization Gap

![](images/50732a6b1e3284b52a3bc6772aed492bea199ce9af35024cc29843ee0ec7fa36.jpg)  
(b) First vs. Last Generation Turn

![](images/03699b29c3d455d5420839494aaf8588852a9bc0c83619e5f98cde59ff4cfc64.jpg)  
(c) GT-History vs. Self-History  
Figure 6: Analysis of stateful multi-turn multimodal interaction. (a) Measuring the understanding-realization gap, lower is better; (b) comparing performance at the first and last image-generation turns under GT-History, to assess the effect of increasingly accumulated context and state dependencies; (c) comparing GT-History with Self-History on dialogues to measure error accumulation when self-generated images are used as subsequent visual context.

Dialogue-State Understanding–Visual Realization Gap. We further investigate the gap between state understanding and visual realization in multimodal multi-turn interaction. For a fair comparison, we retain only state-understanding items on which all evaluated models receive full credit and compute each model’s normalized realization score using the same aligned generation-side capability checks; the understanding–realization gap is then defined as one minus this conditional realization score. Details of the evaluation protocol are provided in Appendix B.6. As shown in Fig. 6a, even when a model correctly understands a currently valid constraint or a corrected state, it does not always faithfully realize that information in the generated output. This reveals a clear gap between state understanding and visual realization in existing UMMs.

Performance Across Dialogue Progression. We analyze how model performance changes as multiturn interactions progress. Fig. 6b compares the performance of three representative models at the first and last image-generation turns of dialogues containing at least two generation turns. Since our evaluation adopts the GT-History setting, errors from previous generations do not propagate to later turns. This comparison therefore more directly reflects how models handle increasingly accumulated context and state dependencies over long-horizon interactions. The results show performance drops of 2%, 6%, and 6% for BAGEL, FLUX.2, and InternVL-U, respectively, suggesting that models still struggle to stably maintain and realize the complete multimodal dialogue state as dialogue length and state complexity increase.

GT-History vs. Self-History. To better reflect realistic multi-turn interaction, we additionally evaluate a Self-History setting, in which images generated by the model in previous turns are used as visual context for subsequent turns. To focus on interactions in which generated visual history can affect subsequent outputs, we include only dialogues requiring more than one generation turn. As shown in Fig. 6c, performance under Self-History decreases by 5.16% and 3.77% in BAGEL and Qwen-Edit, respectively, compared with GT-History. This result suggests that, although models can perform reasonably well when conditioned on correct visual histories, error accumulation from self-generated histories can substantially degrade stateful interaction performance during realistic multi-turn rollouts. More results and detailed

Table 4: Comparison of different models on visual understanding tasks and GenEval. An upward arrow indicates an improvement over BAGEL.
<table><tr><td rowspan="2">Model</td><td colspan="3">Understanding</td><td colspan="6"></td></tr><tr><td>POPE</td><td>MME-P</td><td>MMBench</td><td>Single Obj.</td><td>Two Obj.</td><td>Count.</td><td>Color</td><td>Position Attri.</td><td>Overall</td></tr><tr><td>Emu3 (Wang et al., 2024)</td><td>85.2</td><td>1243.8</td><td>58.5</td><td>0.99</td><td>0.81</td><td>0.42 0.80 0.66</td><td>0.49</td><td>0.45</td><td>0.66</td></tr><tr><td>Show-o (Xie et al., 2025)</td><td>80.0</td><td>1097.2</td><td></td><td>0.98</td><td>0.80</td><td>0.84</td><td>0.31</td><td>0.50</td><td>0.68</td></tr><tr><td>Janus-Pro-7B (Chen et al., 2025d)</td><td>87.4</td><td>1567.1</td><td>79.2</td><td>0.99</td><td>0.89</td><td>0.59 0.90</td><td>0.79</td><td>0.66</td><td>0.80</td></tr><tr><td>MetaQuery-XL (Pan et al., 2025)</td><td></td><td>1685.2</td><td>83.5</td><td></td><td></td><td></td><td></td><td></td><td>0.80</td></tr><tr><td>Ovis-U1 (Wang et al., 2025a)</td><td></td><td>1508.0</td><td>77.8</td><td>0.98</td><td>0.98</td><td>0.90 0.92</td><td>0.79</td><td>0.75</td><td>0.89</td></tr><tr><td>Show-o2 (Xie et al., 2026a)</td><td></td><td>1620.5</td><td>79.3</td><td>1.00</td><td>0.87</td><td>0.58 0.92</td><td>0.52</td><td>0.62</td><td>0.76</td></tr><tr><td>BLIP3-o-8B (Chen et al., 2025a)</td><td></td><td>1682.6</td><td>83.5</td><td>1.00</td><td>0.92</td><td>0.63 0.91</td><td>0.86</td><td>0.67</td><td>0.83</td></tr><tr><td>InternVL-U (Tian et al., 2026)</td><td></td><td>1607.5</td><td>80.5</td><td>0.99</td><td>0.94</td><td>0.74 0.91</td><td>0.77</td><td>0.74</td><td>0.85</td></tr><tr><td>Emu3.5 (Cui et al., 2025)</td><td></td><td>781.1</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.86</td></tr><tr><td>UniAR (Peng et al., 2026)</td><td></td><td></td><td></td><td>1.00</td><td>0.95</td><td>0.75 0.94</td><td>0.77</td><td>0.67</td><td>0.85</td></tr><tr><td>OmniGen2 (Wu et al., 2025c)</td><td></td><td>1584.4</td><td>79.1</td><td>1.00</td><td>0.95</td><td>0.64</td><td>0.88 0.55</td><td>0.76</td><td>0.80</td></tr><tr><td>BAGEL (Deng et al., 2025)</td><td>86.5</td><td>1647.9</td><td>85.0</td><td>1.00</td><td>0.89</td><td>0.70</td><td>0.82</td><td>0.47</td><td>0.49</td><td>0.73</td></tr><tr><td>+ YANAMI-400K</td><td>87.4↑</td><td>1616.9</td><td>89.1↑</td><td>0.99</td><td>0.90↑</td><td>0.73↑</td><td>0.81</td><td>0.52↑</td><td>0.55↑</td><td>0.75↑</td></tr></table>

Table 5: Comparison of fine-tuned BAGEL and other models on GEdit-EN-full benchmark. An upward arrow indicates an improvement over BAGEL.
<table><tr><td>Model</td><td>BG</td><td>Color</td><td>Mat.</td><td>Motion</td><td>Port.</td><td>Style</td><td>Add</td><td>Remove</td><td>Replace</td><td>Text</td><td>Tone</td><td>Avg.</td></tr><tr><td>AnyEdit (Yu et al., 2025)</td><td>4.31</td><td>4.25</td><td>2.64</td><td>0.67</td><td>1.90</td><td>1.95</td><td>3.72</td><td>3.75</td><td>3.23</td><td>0.77</td><td>4.21</td><td>2.85</td></tr><tr><td>InstructPix2Pix (Brooks et al., 2023)</td><td>3.94</td><td>5.40</td><td>3.52</td><td>1.27</td><td>2.62</td><td>4.39</td><td>3.07</td><td>1.50</td><td>3.48</td><td>1.13</td><td>5.10</td><td>3.22</td></tr><tr><td>OmniGen (Xiao et al., 2025)</td><td>5.23</td><td>5.93</td><td>5.44</td><td>3.12</td><td>3.17</td><td>4.88</td><td>6.33</td><td>6.35</td><td>5.34</td><td>4.31</td><td>4.96</td><td>5.01</td></tr><tr><td>Step1X-Edit (Liu et al., 2025)</td><td>7.03</td><td>6.26</td><td>6.46</td><td>3.66</td><td>5.23</td><td>7.24</td><td>7.17</td><td>6.42</td><td>7.39</td><td>7.40</td><td>6.62</td><td>6.44</td></tr><tr><td>Step1X-Edit-v1p2 (Liu et al., 2025)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>7.24</td></tr><tr><td>OmniGen2 (Wu et al., 2025c)</td><td>6.99</td><td>6.66</td><td>4.88</td><td>2.55</td><td>3.66</td><td>6.08</td><td>7.09</td><td>6.60</td><td>6.65</td><td>4.49</td><td>6.03</td><td>5.57</td></tr><tr><td>EditMGT (Chow et al., 2026a)</td><td>7.69</td><td>7.71</td><td>5.77</td><td>3.84</td><td>5.13</td><td>6.53</td><td>6.13</td><td>5.24</td><td>5.56</td><td>4.53</td><td>6.42</td><td>5.87</td></tr><tr><td>GoT-6B (Fang et al., 2025)</td><td>4.11</td><td>5.75</td><td>3.04</td><td>1.71</td><td>2.69</td><td>4.72</td><td>5.77</td><td>4.59</td><td>5.65</td><td>1.16</td><td>4.24</td><td>3.95</td></tr><tr><td>VAREdit-8B (Mao et al., 2026)</td><td>6.77</td><td>6.64</td><td>5.40</td><td>3.33</td><td>4.20</td><td>6.46</td><td>5.86</td><td>7.29</td><td>6.67</td><td>3.87</td><td>6.54</td><td>5.73</td></tr><tr><td>FLUX.1 Kontext [dev] (Labs et al., 2025)</td><td>7.06</td><td>7.03</td><td>5.52</td><td>5.62</td><td>4.68</td><td>5.55</td><td>6.95</td><td>6.76</td><td>6.13</td><td>6.10</td><td>7.48</td><td>6.26</td></tr><tr><td>FLUX.2 [dev] (Labs, 2025)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>7.41</td></tr><tr><td>Qwen-Image-Edit (Wu et al., 2025a)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td></td><td>7.88</td></tr><tr><td>Ovis-U1 (Wang et al., 2025a)</td><td>7.49</td><td>6.88</td><td>6.21</td><td>4.79</td><td>5.98</td><td>6.46</td><td>7.49</td><td>7.25</td><td>7.27</td><td>4.48</td><td>6.31</td><td>6.42</td></tr><tr><td>BAGEL (Deng et al., 2025)</td><td>7.44</td><td>6.99</td><td>6.26</td><td>5.09</td><td>4.82</td><td>6.04</td><td>7.94</td><td>7.37</td><td>7.31</td><td>7.16</td><td>6.17</td><td>6.52</td></tr><tr><td>+ YANAMI-400K</td><td></td><td>7.58↑ 7.76↑</td><td>6.42↑</td><td>7.06↑</td><td>6.11↑</td><td>6.20↑</td><td>8.44↑</td><td>7.74↑</td><td>7.55↑</td><td>7.11</td><td></td><td>7.08↑7.19↑</td></tr></table>

breakdowns are provided in Appendix D.

## 4.2 Train on YANAMI-400K

To validate the effectiveness of YANAMI-400K, we fine-tune BAGEL on our dataset, with detailed training configurations provided in Appendix C. The results demonstrate that YANAMI-400K substantially improves image generation, image editing, and multi-turn multimodal dialogue state modeling, while largely preserving the model’s general visual understanding capabilities.

Fine-tuned BAGEL performances on YANAMI-Bench across four interaction scenarios: (i) Visual Understanding. The fine-tuned BAGEL largely preserves its original general visual understanding capabilities. As shown in Table 4, the model achieves improvements of 4.1% and 0.9% on MMBench and POPE, respectively, while exhibiting only a slight decrease on MME-P, indicating that training on YANAMI-400K preserves general visual understanding capability. (ii) Image Generation. As shown in Table 4, the fine-tuned BAGEL improves the overall score on GenEval by 2%, demonstrating that YANAMI-400K enhances the model’s general image generation capability. (iii) Image Editing. As shown in Table 5, the fine-tuned BAGEL improves the overall score on GEdit-Bench-EN by 10.28%, further demonstrating the effectiveness of YANAMI-400K in enhancing image editing capability. (iv) Multimodal Dialogue Stateful Interaction. As shown in Table 2, the fine-tuned model improves by 13.23% over the original BAGEL on YANAMI-Bench. This result demonstrates that YANAMI-400K not only improves general image generation and editing capabilities, but also substantially enhances the model’s ability to understand, maintain, update, and visually realize multimodal dialogue states throughout multi-turn interactions.

## 4.3 Reliability of Judge Usage

To assess the reliability of our VLM-as-a-Judge evaluation framework, we conduct a human evaluation study involving four expert annotators. Specifically, we sample 100 instances from each of three representative models: BAGEL+YANAMI, Emu3.5, and Nano Banana 2, and collect human ratings for comparison. We compute Pearson correlation coefficients(Benesty et al., 2009) between the scores produced by Qwen-3.8-Flash-Next(Qwen Team, 2026) and the aggregated human ratings, and further compare its evaluations with those produced by GPT-5.6-sol, as shown in Fig. 5.

The results show that the correlations between Qwen-3.8-Flash-Next and human ratings consistently exceed 0.80, indicating strong agreement between the VLM judge and human evaluation. Moreover, the scores produced by Qwen-3.8-Flash-Next and GPT-5.6-sol exhibit high cross-VLM correlation, demonstrating strong consistency across different VLM evaluators. These results suggest that our evaluation framework aligns well with human judgments and is robust to the choice of VLM judge.

## 5 Conclusion

This work introduces YANAMI, the first large-scale dataset and benchmark for multimodal stateful interaction across understanding, generation, and editing. YANAMI-400K provides 400K multi-turn dialogues, more than 2M images, and 2.6M dialogue turns across four scenarios and six types of contextual dependency. YANAMI-Bench comprises 400 multi-turn dialogues and jointly evaluates dialogue-state understanding and visual realization. Experiments show that training on YANAMI-400K improves multi-turn contextual understanding, image generation, and image editing while preserving general visual understanding. Evaluations on YANAMI-Bench further show that existing UMMs still struggle to maintain, update, and visually realize dialogue states. YANAMI provides a unified foundation for robust multimodal stateful interaction.

## AI Use Statement

LLM-based tools assisted with language polishing, code development, data filtering, dialogue construction, and quality control. As detailed in Appendix A, GPT-5.6 Sol, Qwen3.7-Plus, and Gemini 3.1 Pro were used in the data-construction pipeline, while GPT-Image-2 and Nano Banana 2 supported image generation and repair; Qwen-3.8-Flash-Next was additionally used for automated evaluation under the protocols in Appendix B. All benchmark samples and annotations were manually verified; the authors reviewed AI-assisted writing and code, validated the resulting procedures, and take responsibility for the final content and claims of this work.

## Ethics Statement

YANAMI is intended for research on multimodal stateful interaction and is constructed from existing public datasets and model-generated content. We follow the applicable licenses and usage conditions of the source datasets and models. Automated filtering and human verification are applied to improve data quality. We encourage the research community to use and further develop these resources responsibly.

## Reproducibility Statement

We provide the information needed to reproduce our data construction, training, and evaluation procedures throughout the paper and appendix. The construction pipeline, quality-control process, benchmark annotation procedure, and detailed data statistics are documented in Appendices A and A.6. The evaluation protocols, score calculation, history and input formatting, and model-specific inference configurations are described in Appendix B, while the fine-tuning configuration is reported in Appendix C. Complete experimental results and the verbatim VLM-as-a-Judge prompts are provided in Appendices D and F, respectively. To further support reproducibility, we will release YANAMI-400K and YANAMI-Bench on Hugging Face and publish the prompts used for data construction and evaluation in a public GitHub repository.

## References

Alibaba Cloud. 2026. Qwen3.7-Plus. https://www.alibabacloud.com/help/en/model-studio/ qwen3-7-plus. Accessed: 2026-09-16.

Jacob Benesty, Jingdong Chen, Yiteng Huang, and Israel Cohen. 2009. Pearson correlation coefficient. In Noise reduction in speech processing, pages 1–4. Springer.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2023. Instructpix2pix: Learning to follow image editing instructions. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18392–18402. IEEE.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, and 1 others. 2025a. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568.

Tianyu Chen, Yasi Zhang, Zhi Zhang, Peiyu Yu, Shu Wang, Zhendong Wang, Kevin Lin, Xiaofei Wang, Zhengyuan Yang, Linjie Li, and 1 others. 2025b. Edival-agent: An object-centric framework for automated, fine-grained evaluation of multi-turn editing. arXiv preprint arXiv:2509.13399.

Wei Chen, Lin Li, Yongqi Yang, Bin Wen, Fan Yang, Tingting Gao, Yu Wu, and Long Chen. 2025c. Comm: A coherent interleaved image-text dataset for multimodal understanding and generation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8073–8082. IEEE.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. 2025d. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811.

Zhekai Chen, Yuqing Wang, Manyuan Zhang, and Xihui Liu. 2026. Macro: Advancing multi-reference image generation with structured long-context data. arXiv preprint arXiv:2603.25319.

Wei Chow, Linfeng Li, Lingdong Kong, Zefeng Li, Qi Xu, Hang Song, Tian Ye, Xian Wang, Jinbin Bai, Shilin Xu, and 1 others. 2026a. Editmgt: Unleashing potentials of masked generative transformers in image editing. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 38038–38048.

Wei Chow, Jiachun Pan, Yongyuan Liang, Mingze Zhou, Xue Song, Liyu Jia, Saining Zhang, Siliang Tang, Juncheng Li, Fengda Zhang, and 1 others. 2026b. Weave: Unleashing and benchmarking the in-context interleaved comprehension and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15343–15353.

Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, and 1 others. 2025. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583.

Wenxun Dai, Zhiyuan Zhao, Yule Zhong, Yiji Cheng, Jianwei Zhang, Linqing Wang, Shiyi Zhang, Yunlong Lin, Runze He, Fellix Song, and 1 others. 2026. Chatumm: Robust context tracking for conversational interleaved generation. arXiv preprint arXiv:2602.06442.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, and 1 others. 2025. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683.

Rongyao Fang, Chengqi Duan, Kun Wang, Linjiang Huang, Hao Li, Shilin Yan, Hao Tian, Xingyu Zeng, Rui Zhao, Jifeng Dai, and 1 others. 2025. Got: Unleashing reasoning capability of multimodal large language model for visual generation and editing. arXiv preprint arXiv:2503.10639.

Yukang Feng, Jianwen Sun, Chuanhao Li, Zizhen Li, Jiaxin Ai, Fanrui Zhang, Sizhuo Zhou, Yifan Chang, Shenglin Zhang, Yu Dai, and 1 others. 2026. A high quality dataset and reliable evaluation for interleaved image-text generation. volume 2026, pages 40398–40437.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and 1 others. 2026. Mme: A comprehensive evaluation benchmark for multimodal large language models. Advances in Neural Information Processing Systems, 38.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. 2023. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152.

Google. 2026a. Gemini 3.1 Flash Image (Nano Banana 2). https://ai.google.dev/gemini-api/ docs/models/gemini-3.1-flash-image. Accessed: 2026-09-16.

Google. 2026b. Gemini 3.1 Pro Preview. https://ai.google.dev/gemini-api/docs/models/ gemini-3.1-pro-preview. Accessed: 2026-09-16.

Minbin Huang, Yanxin Long, Xinchi Deng, Ruihang Chu, Jiangfeng Xiong, Xiaodan Liang, Hong Cheng, Qinglin Lu, and Wei Liu. 2025. Dialoggen: multi-modal interactive dialogue system with multi-turn text-image generation. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 411–426.

Black Forest Labs. 2025. Flux.2: Frontier visual intelligence. https://bfl.ai/blog/flux-2.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, and 1 others. 2025. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742.

Ang Li, Charles Wang, Deqing Fu, Kaiyu Yue, Zikui Cai, Wang Bill Zhu, Ollie Liu, Peng Guo, Willie Neiswanger, Furong Huang, and 1 others. 2025. Zebra-cot: A dataset for interleaved vision language reasoning. arXiv preprint arXiv:2507.16746.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292–305.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, and 1 others. 2025. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761.

Shuo Liu, Kaining Ying, Hao Zhang, Yue Yang, Yuqi Lin, Tianle Zhang, Chuanhao Li, Yu Qiao, Ping Luo, Wenqi Shao, and 1 others. 2024a. Convbench: A multi-turn conversation evaluation benchmark with hierarchical capability for large vision-language models. arXiv preprint arXiv:2403.20194.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, and 1 others. 2024b. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer.

Qingyang Mao, Qi Cai, Yehao Li, Yingwei Pan, Mingyue Cheng, Ting Yao, Qi Liu, and Tao Mei. 2026. Visual autoregressive modeling for instruction-guided image editing. volume 2026, pages 85702–85723.

OpenAI. 2026a. GPT-5.6 Sol. https://developers.openai.com/api/docs/models/gpt-5.6- sol. Accessed: 2026-09-16.

OpenAI. 2026b. GPT-Image-2. https://developers.openai.com/api/docs/models/gptimage-2. Accessed: 2026-09-16.

Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, and 1 others. 2025. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256.

Wujian Peng, Lingchen Meng, Yuxuan Cai, Xianwei Zhuang, Yuhuan Yang, Rongyao Fang, Chenfei Wu, Junyang Lin, Zuxuan Wu, and Shuai Bai. 2026. Unified multimodal autoregressive modeling with shared context-visual tokenizer is key to unification. arXiv preprint arXiv:2606.18249.

Yusu Qian, Eli Bocek-Rivele, Liangchen Song, Jialing Tong, Yinfei Yang, Jiasen Lu, Wenze Hu, and Zhe Gan. 2026. Pico-banana-400k: A large-scale dataset for text-guided image editing. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 37226–37235.

Qwen Team. 2026. Qwen3.8-Flash-Next: A new architecture, towards ultimate cost-efficiency.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 10674–10685. ieee.

Chameleon Team. 2024. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818.

Changyao Tian, Danni Yang, Guanzhou Chen, Erfei Cui, Zhaokai Wang, Yuchen Duan, Penghao Yin, Sitao Chen, Ganlin Yang, Mingxin Liu, and 1 others. 2026. Internvl-u: Democratizing unified multimodal models for understanding, reasoning, generation and editing. arXiv preprint arXiv:2603.09877.

Guo-Hua Wang, Shanshan Zhao, Xinjie Zhang, Liangfu Cao, Pengxin Zhan, Lunhao Duan, Shiyin Lu, Minghao Fu, Xiaohao Chen, Jianshan Zhao, and 1 others. 2025a. Ovis-u1 technical report. arXiv preprint arXiv:2506.23044.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, and 1 others. 2024. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869.

Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. 2025b. Gpt image-edit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033.

Hongyang Wei, Hongbo Liu, Zidong Wang, Yi Peng, Baixin Xu, Size Wu, Xuying Zhang, Xianglong He, Zexiang Liu, Peiyu Wang, and 1 others. 2026a. Skywork unipic 3.0: Unified multi-image composition via sequence modeling. arXiv preprint arXiv:2601.15664.

Xinyu Wei, Kangrui Cen, Hongyang Wei, Zhen Guo, Bairui Li, Zeqing Wang, Jinrui Zhang, and Lei Zhang. 2026b. Mico-150k: A comprehensive dataset advancing multi-image composition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 29695–29706.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, and 1 others. 2025a. Qwen-image technical report. arXiv preprint arXiv:2508.02324.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, and 1 others. 2025b. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12966–12977. IEEE.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, and 1 others. 2025c. Omnigen2: Exploration to advanced multimodal generation. arXiv e-prints, pages arXiv–2506.

Peng Xia, Siwei Han, Shi Qiu, Yiyang Zhou, Zhaoyang Wang, Wenhao Zheng, Zhaorun Chen, Chenhang Cui, Mingyu Ding, Linjie Li, and 1 others. 2025. Mmie: Massive multimodal interleaved comprehension benchmark for large vision-language models. volume 2025, pages 25842–25875.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. 2025. Omnigen: Unified image generation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13294–13304. IEEE.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. 2025. Show-o: One single transformer to unify multimodal understanding and generation. volume 2025, pages 28240–28264.

Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. 2026a. Show-o2: Improved native unified multimodal models. Advances in Neural Information Processing Systems, 38:47490–47518.

Wulin Xie, YiFan Zhang, Chaoyou Fu, Yang Shi, Jianshu Zeng, Bingyan Nie, Hongkai Chen, Zhang Zhang, and Liang Wang. 2026b. Mme-unify: A comprehensive benchmark for unified multimodal understanding and generation models. volume 2026, pages 133535–133578.

Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. 2025. Anyedit: Mastering unified high-quality image editing for any idea. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26125–26135. IEEE.

## Appendix

A Data Construction Details 15   
A.1 Comparison with Existing Multimodal Datasets 15   
A.2 Data Preparation and Composition 15   
A.3 Data Construction . 16   
A.4 Automatic Verification 17   
A.5 Human Verification and Benchmark Construction 17   
A.6 Detailed Data Statistics 18   
B Evaluation Protocol and Metrics 18   
B.1 Dialogue-State Understanding Evaluation 18   
B.2 Visual Quality and State Consistency Evaluation . 20   
B.3 Score Calculation 20   
B.4 History and Input Formatting 20   
B.5 Baseline Inference Configuration . 21   
B.6 Understanding–Realization Gap Evaluation 21   
C Training Details 22   
D Complete Experimental Results 22   
D.1 Aggregate Results on YANAMI-Bench . 22   
D.2 Dialogue-State Understanding Results 23   
D.3 Results across Dialogue Progression 23   
D.4 Understanding–Realization Gap 23   
E Additional Qualitative Examples 28   
F VLM-as-a-Judge Prompt Templates 33   
F.1 Dialogue-State Understanding Prompt . 33   
F.2 State Consistency Prompt . 33   
F.3 Visual Quality Prompt 35   
G Limitations 35

## A Data Construction Details

## A.1 Comparison with Existing Multimodal Datasets

Table 6 compares YANAMI-400K with representative multimodal understanding, generation, editing, and dialogue datasets. The comparison separates multi-turn formatting from explicit dialogue-state management: a dataset may be organized as a sequence of turns while still allowing individual target turns to be reformulated as conventional single-turn tasks once the relevant historical content is directly supplied.

Table 6: Comparison with representative multimodal datasets. U, G, and E denote visual understanding, image generation, and image editing, respectively. Multi-turn Format indicates that data are organized as multi-turn user–assistant interactions; Hybrid Evaluation indicates a unified protocol combining complementary evaluation signals rather than merely reporting multiple independent metrics. Explicit Dialogue-State Management is marked Yes only when a dataset explicitly supervises the establishment, maintenance, invalidation, or update of the currently valid multimodal state; generic history conditioning, context tracking, or a claim of “stateful dialogue” alone does not qualify. Irreducible to Single-Turn indicates that target turns cannot be reformulated as conventional single-turn tasks simply by directly supplying the relevant historical images. Partial denotes implicit or subset-level coverage. Scale follows the unit reported by each work and is therefore not directly comparable across all rows. <sup>‡</sup> ChatUMM reports approximately 40B consumed training tokens but does not report the number of dialogues, images, or samples. <sup>§</sup> No official ChatUMM training-data release was publicly available as of 2026-09-24; its categorization is therefore based on the construction procedure and examples reported in the paper. Although ChatUMM describes its interactions as stateful, its published pipeline primarily converts standard single-turn tasks into dialogues through historical dependencies and distractor turns, without explicit dialogue-state supervision. Once the relevant historical input is supplied directly, the reported target turns can generally be reformulated as conventional single-turn generation, editing, or composition tasks.

<table><tr><td>Dataset</td><td>Scale</td><td>Domain</td><td>Open Data</td><td>Multi-turn Format</td><td>Hybrid Evaluation</td><td>Dialogue-State Management</td><td>Irreducible to Single-Turn</td></tr><tr><td>InstructPix2Pix (Brooks et al., 2023)</td><td>~450K editing pairs</td><td>E</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td></tr><tr><td>AnyEdit (Yu et al., 2025)</td><td>2.5M editing pairs</td><td>E</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td></tr><tr><td>CoMM (Chen et al., 2025c)</td><td>227K documents</td><td>U/G</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Zebra-CoT (Li et al., 2025)</td><td>182,384 samples</td><td>U/G</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Pico-Banana-400K (Qian et al., 2026)</td><td>400K editing pairs; 72K multi-turn</td><td>E</td><td>Yes</td><td>Partial</td><td>No</td><td>No</td><td>No</td></tr><tr><td>WEAVE-100K (Chow et al., 2026b)</td><td>100,750 dialogues</td><td>U/G/E</td><td>Yes</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>ChatUMM (Dai et al., 2026)</td><td>Not reported</td><td>U/G/E</td><td>No</td><td>Yes</td><td>No</td><td>No</td><td>Partial</td></tr><tr><td>YANAMI-400K (Ours)</td><td>402,639 dialogues</td><td>U/G/E</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr></table>

## A.2 Data Preparation and Composition

We draw raw images from four existing public datasets (Li et al., 2025; Chow et al., 2026b; Wei et al., 2026b; Chen et al., 2026) and process them in the Data Preparation stage shown in Fig. 3.a. Our pipeline leverages GPT-5.6 Sol (OpenAI, 2026a), Qwen3.7-Plus (Alibaba Cloud, 2026), Gemini 3.1 Pro (Google, 2026b), GPT-Image-2 (OpenAI, 2026b), and Nano Banana 2 (Google, 2026a) for data filtering, dialogue construction, and quality control. The raw images are first filtered for quality and suitability and then organized into four scenario-specific pools for Multi-Image Fusion, Image Editing, World Modeling, and Visual Reasoning. For this initial quality-control stage, we use large multimodal models with the task-specific prompt reproduced in Appendix A.2. The filter deliberately performs a permissive coarse screening: an image is marked as rejected only when severe corruption, generation failure, invalid content, or extremely poor visibility makes it clearly unusable. We apply the filter independently to every image and record the resulting quality labels at the image level. For each image group, if more than half of its images are marked as rejected, we discard the entire group; otherwise, we repair only the rejected images and send the repaired versions through the same filtering process again. Samples that remain invalid after this process are discarded. This procedure supplies diverse visual materials for constructing multi-turn multimodal stateful dialogues involving understanding, generation, and editing, without imposing a one-to-one correspondence between a source dataset and an interaction scenario.

## Image Quality Filtering Prompt

ing   
You are an image quality filter responsible for performing a <sub>\*\*</sub>coarse initial screening<sub>\*\*</sub> of images.   
Your goal is NOT to evaluate aesthetics, fine details, instruction following, or professional image   
,→ quality.   
You should only reject images that are <sub>\*\*</sub>clearly unusable or extremely low quality<sub>\*\*</sub>.   
Evaluate the input image and determine whether it should be KEPT or REJECTED .   
### REJECT the image only if one or more of the following severe problems occur:   
1. <sub>\*\*</sub>Unrecognizable content<sub>\*\*</sub>: the image is so corrupted, blurred, distorted, or noisy that its main   
,→ content cannot be reasonably identified.   
2. <sub>\*\*</sub>Severe generation failure<sub>\*\*</sub>: the image contains large-scale visual corruption, completely broken   
,→ structures, extreme artifacts, or incoherent content that makes the image unusable.   
3. <sub>\*\*</sub>Nearly empty or invalid image<sub>\*\*</sub>: the image is blank, almost entirely a single color, severely   
,→ cropped to the point that meaningful content is absent, or otherwise appears invalid.   
4. Extremely poor visibility : the image is so dark, overexposed, low-resolution, or degraded that   
,→ the main subject or scene cannot be meaningfully perceived.   
### KEEP the image if:   
- The main subject and scene are recognizable.   
- The image contains understandable visual content.   
- Imperfections are minor or moderate.   
Do NOT reject an image merely because of:   
- minor blur or low resolution;   
- imperfect anatomy or object geometry;   
- small artifacts;   
- unrealistic details;   
- mediocre aesthetics;   
- imperfect composition, lighting, or colors;   
- minor text rendering errors;   
- moderate generation artifacts.   
<sub>\*\*</sub>Be permissive. When uncertain, KEEP the image.<sub>\*\*</sub>   
Return only the following JSON:   
\`\`\`json   
{   
"decision": "KEEP" or "REJECT",   
"reason": "brief reason"   
}

The resulting YANAMI-400K contains 402,639 dialogues. As shown in Fig. 4, Multi-Image Fusion, Image Editing, World Modeling, and Visual Reasoning account for 36%, 25%, 28%, and 11% of the training dialogues, respectively. We retain this naturally varying training distribution while constructing a balanced benchmark with 100 dialogues from each scenario, enabling controlled comparison across scenarios. Complete corpus- and scenario-level statistics are provided in Section A.6.

## A.3 Data Construction

After data preparation, each scenario-specific pool is combined with scenario- and capability-oriented prompts to construct multi-turn multimodal stateful dialogues, as illustrated in Fig. 3.b. The complete data-construction prompts are provided in the supplementary material. Within each dialogue, visual observations, textual instructions, and generated or edited images jointly determine how the multimodal dialogue state evolves. The four scenarios instantiate distinct forms of this evolution.

1. Multi-Image Fusion : the assistant integrates multiple user-provided or previously generated images according to the dialogue history.

2. Image Editing : the assistant progressively edits an image while preserving prior modifications and visual consistency.

3. World Modeling : the assistant predicts and renders scene states under specified viewpoint changes or actions.

4. Visual Reasoning : the assistant generates visual outcomes for proposed hypotheses or solutions to support step-by-step reasoning.

To make these interactions genuinely stateful, we incorporate six key state-management capabilities across the four scenarios: Coreference Resolution, Dialogue-History Retrieval, Cross-Turn Consistency, Spatial and Physical Consistency, Constraint Adherence, and Correction-Based State Updating.

1. Coreference Resolution grounds context-dependent expressions to the correct textual or visual entities.

2. Dialogue-History Retrieval identifies and uses information relevant to the current request from previous turns.

3. Cross-Turn Consistency preserves established visual attributes and prior edits across turns unless the user explicitly changes them.

4. Spatial and Physical Consistency maintains plausible geometry, physical relations, and state transitions.

5. Constraint Adherence consistently follows user-specified requirements and restrictions.

6. Correction-Based State Updating incorporates user corrections and replaces outdated information accordingly.

These capabilities cut across the four scenarios rather than forming a fixed one-to-one mapping: multiple capabilities jointly construct and maintain the evolving state within a dialogue, and the same capability can occur in different scenarios. Every dialogue in YANAMI-400K involves at least four of the six capabilities, requiring multiple cross-turn dependencies to be exercised jointly within the interaction.

## A.4 Automatic Verification

We apply quality control both before and after data construction, following the pipeline in Fig. 3. During data preparation, model-assisted filtering removes unsuitable visual materials and routes potentially repairable samples back through the preparation process. After data construction, each dialogue is automatically verified for logical coherence and image–text consistency: the former examines the instructions and state transitions across turns, while the latter checks the visual content against its associated textual context. Failed samples are iteratively corrected or regenerated and then re-verified, while dialogues that pass verification form YANAMI-400K or enter the benchmark-candidate pool. This iterative process is intended to eliminate inconsistent state transitions, mismatched image–text pairs, and dialogues that do not realize the intended scenario or contextual dependencies.

## A.5 Human Verification and Benchmark Construction

We construct YANAMI-Bench with LLM assistance from candidates that pass automatic verification, followed by exhaustive human verification and correction. The benchmark is balanced across the four interaction scenarios, with 100 dialogues per scenario and 400 dialogues in total. (i) Dialogue-state understanding QA pairs. Each benchmark dialogue includes four QA pairs, one each for Reference Resolution, Context Retrieval, Constraint Tracking, and Correction Updating. (ii) Fine-grained atomic criteria. Each dialogue also includes atomic criteria for evaluating State Consistency in generated or edited visual outputs. These atomic criteria decompose the requirements of a target image into independently verifiable checks, allowing State Consistency to be assessed separately from holistic Visual Quality.

As shown in the human-verification stage of Fig. 3, the initial atomic checks are generated with LLM assistance and then reviewed together with the corresponding dialogue and visual content. Three human annotators with graduate-level training in STEM disciplines verify each sample. They inspect the dialogue logic, visual evidence, understanding questions, and atomic evaluation criteria, and correct samples that do not faithfully represent the intended state evolution or evaluation target. Failed samples undergo manual correction or regeneration followed by another review, and every sample retained in the final benchmark is manually verified. The final benchmark contains 1,600 understanding questions and 8,467 atomic visual checks, with detailed per-scenario statistics reported in Table 8.

Figure 7 shows the interface used to inspect complete dialogues, generated visual outputs, and their associated atomic checks during benchmark verification.

![](images/76948d670b83800258f9d6d97ba6a64169f3834f6cbf39c2dd5b41db86c63a13.jpg)  
Figure 7: Human-verification interface for YANAMI-Bench. Annotators can browse the 100 benchmark dialogues in each interaction scenario, inspect the complete multi-turn context and visual outputs, and review or revise the corresponding atomic evaluation criteria. The interface also records whether a sample is approved or requires image refinement, supporting the exhaustive verification and correction process used to construct the final benchmark.

## A.6 Detailed Data Statistics

YANAMI-400K contains 402,639 multi-turn dialogues, 2,047,236 images, and 2,643,291 user–assistant dialogue turns across the four interaction scenarios. Among these dialogues, 321,868 contain at least four images, 215,027 contain at least five images, and 138,669 contain at least six images. The dataset contains an average of 5.08 images and 6.56 dialogue turns per dialogue, with a maximum of 43 dialogue turns. Table 7 summarizes these statistics, while Fig. 4 presents the distribution across the four interaction scenarios.

YANAMI-Bench contains 400 dialogues evenly distributed across the four scenarios, together with 1,883 images, 3,173 user–assistant dialogue turns, 1,600 dialogue-state understanding questions, and 8,467 atomic visual checks used in the benchmark evaluation. Each dialogue contains one question for each of the four understanding capabilities, yielding 400 questions per capability. Table 8 reports the corresponding scenario-level statistics.

## B Evaluation Protocol and Metrics

## B.1 Dialogue-State Understanding Evaluation

Each benchmark dialogue contains four diagnostic questions, one for each explicitly testable understanding capability: Reference Resolution, Context Retrieval, Constraint Tracking, and Correction Updating. The evaluator receives the dialogue context, the diagnostic question, a reference answer, and the model prediction, and judges semantic correctness rather than exact lexical overlap. A prediction receives credit only when it contains all information required by the reference answer without a material factual error or contradiction. Equivalent wording and correct additional detail are accepted, whereas partially correct, unsupported, or materially incomplete answers are marked incorrect. The score for each capability is the proportion of correctly answered questions, and the overall dialogue-state understanding score is the arithmetic mean defined in Appendix B.3.

Table 7: Detailed statistics of YANAMI-400K. Each dialogue turn contains one user message and one assistant response; Total Messages therefore counts both sides of the interaction. Scenario shares are rounded percentages corresponding to the training-set distribution shown in Fig. 4.
<table><tr><td colspan="2">Corpus and Sequence Statistics</td><td colspan="3">Dataset Distributions</td></tr><tr><td>Statistic</td><td>Value</td><td>Images per Dialogue</td><td>Dialogues</td><td>Share</td></tr><tr><td>Total Dialogues</td><td>402,639</td><td>&lt; 4</td><td>80,771</td><td>20.06%</td></tr><tr><td>Total Images</td><td>2,047,236</td><td>=4</td><td>106,841</td><td>26.54%</td></tr><tr><td>Total Dialogue Turns</td><td>2,643,291</td><td>= 5</td><td>76,358</td><td>18.96%</td></tr><tr><td>Total User Messages</td><td>2,643,291</td><td>≥4</td><td>321,868</td><td>79.94%</td></tr><tr><td>Total Assistant Messages</td><td>2,643,291</td><td>≥5</td><td>215,027</td><td>53.40%</td></tr><tr><td>Total Messages</td><td>5,286,582</td><td>≥6</td><td>138,669</td><td>34.44%</td></tr><tr><td></td><td colspan="4">Training Scenario Distribution</td></tr><tr><td>Average Images / Dialogue</td><td>5.08</td><td>Multi-Image Fusion</td><td></td><td>36%</td></tr><tr><td>Average Turns / Dialogue</td><td>6.56</td><td>Image Editing</td><td></td><td>25%</td></tr><tr><td>Average Messages / Dialogue</td><td>13.13</td><td>World Modeling</td><td></td><td>28%</td></tr><tr><td>Maximum Turns / Dialogue</td><td>43</td><td>Visual Reasoning</td><td></td><td>11%</td></tr><tr><td>Maximum Messages / Dialogue</td><td>86</td><td>Interaction Scenarios</td><td>4</td><td></td></tr><tr><td>Images / Dialogue Turn</td><td>0.77</td><td>State-Management Capabilities</td><td>6</td><td></td></tr><tr><td>Dialogues with ≥ 4 Capabilities</td><td>402,639</td><td>Capability Coverage</td><td></td><td>100%</td></tr></table>

Table 8: Detailed statistics of YANAMI-Bench by interaction scenario. Images include all reference and target images contained in the benchmark dialogues. Understanding QAs comprise Reference Resolution, Context Retrieval, Constraint Tracking, and Correction Updating questions, with one question of each type per dialogue. Atomic Checks counts the independently verifiable visual criteria used in the benchmark evaluation and excludes capability-specific checks; Avg. Checks reports the average number of evaluated atomic checks per dialogue.
<table><tr><td>Scenario</td><td>Dialogues</td><td>Images</td><td>Dialogue Turns</td><td>Understanding QAs</td><td>Atomic Checks</td><td>Avg. Checks</td></tr><tr><td>Multi-Image Fusion</td><td>100</td><td>400</td><td>777</td><td>400</td><td>1,890</td><td>18.90</td></tr><tr><td>Image Editing</td><td>100</td><td>499</td><td>754</td><td>400</td><td>3,408</td><td>34.08</td></tr><tr><td>World Modeling</td><td>100</td><td>467</td><td>598</td><td>400</td><td>1,141</td><td>11.41</td></tr><tr><td>Visual Reasoning</td><td>100</td><td>517</td><td>1,044</td><td>400</td><td>2,028</td><td>20.28</td></tr><tr><td>Total</td><td>400</td><td>1,883</td><td>3,173</td><td>1,600</td><td>8,467</td><td>21.17</td></tr></table>

## B.2 Visual Quality and State Consistency Evaluation

State Consistency decomposes the expected target state into independently verifiable atomic visual checks. For each generated image, the evaluator compares the candidate with the ground-truth target and assigns a binary outcome to every atomic check. A check receives credit only when all of its required conditions are visibly satisfied; semantically equivalent visual realizations are accepted when they do not alter the evaluated state property. The State Consistency score is the fraction of atomic checks that receive credit. Capability-specific checks are evaluated separately and are used only for analyses that require alignment between an understanding item and its corresponding generated outcome.

Visual Quality assesses the generated image independently of task correctness on a 0–10 integer scale. The rubric considers structural coherence, visual fidelity, generation artifacts, and overall naturalness, with higher scores indicating better perceptual quality. The complete prompts for all three evaluators are reproduced in Appendix F.

## B.3 Score Calculation

For each interaction scenario, we first compute the dialogue-state understanding score as the arithmetic mean of the four understanding metrics: Reference Resolution (RR), Context Retrieval (CR), Constraint Tracking (CT), and Correction Updating (CU). Specifically, the understanding score is calculated as

$$
S _ { \mathrm { u n d e r s t a n d i n g } } = \frac { S _ { \mathrm { R R } } + S _ { \mathrm { C R } } + S _ { \mathrm { C T } } + S _ { \mathrm { C U } } } { 4 } .\tag{1}
$$

Before aggregation, all component scores are converted to a 0–100 scale: the State Consistency (SC) score, originally expressed as a fraction, is multiplied by 100, and the Visual Quality (VQ) score, originally measured on a 0–10 scale, is multiplied by 10. For models with both understanding and generation capabilities, the final score for each scenario is computed using a 4:4:2 weighting over dialogue-state understanding, State Consistency, and Visual Quality:

$$
S _ { \mathrm { s c e n e } } = 0 . 4 S _ { \mathrm { u n d e r s t a n d i n g } } + 0 . 4 S _ { \mathrm { S C } } + 0 . 2 S _ { \mathrm { V Q } } .\tag{2}
$$

For models without dialogue-state understanding capabilities, the scenario score is computed from State Consistency and Visual Quality using a 6:4 weighting:

$$
S _ { \mathrm { s c e n e } } = 0 . 6 S _ { \mathrm { S C } } + 0 . 4 S _ { \mathrm { V Q } } .\tag{3}
$$

This weighting ensures that state faithfulness remains the primary component of the final score while retaining an explicit assessment of perceptual visual quality.

## B.4 History and Input Formatting

Our main evaluation adopts the GT-History setting to measure the model’s ability to use accumulated dialogue state without conflating it with errors produced in earlier generation turns. When evaluating a target generation or editing turn, we retain the complete textual dialogue preceding that turn and replace all earlier assistant-generated images with their corresponding ground-truth images. User-provided reference images remain unchanged. The target output itself is never included in the input context.

We additionally evaluate Self-History to approximate realistic multi-turn deployment. Under this setting, each image produced by the evaluated model is inserted into the dialogue history and used as visual context for subsequent turns, while the textual history remains unchanged. User-provided reference images again remain unchanged. Consequently, errors in earlier visual outputs may alter the context available at later turns and accumulate throughout the interaction.

We use three input organizations according to the interfaces supported by individual models. For interleaved image–text input, denoted as yes-first, each image is inserted at the position of its first mention in the dialogue, preserving the original temporal order of the interaction. For image-first input, denoted as yes-front, all visual inputs available at the target turn are placed before the textual dialogue in their order of appearance. For models that cannot process sequential multi-image inputs, the available images are concatenated in dialogue order into a single composite visual input and supplied together with the corresponding textual instruction. The input organization used for each model is reported in Table 2.

Table 9: Baseline inference configurations used on YANAMI-Bench. Only settings that materially affect input organization or output generation are reported; all models otherwise follow the shared evaluation protocol described in the text.
<table><tr><td>Model</td><td>Generation mechanism</td><td>Visual-input handling</td><td>Principal inference settings</td></tr><tr><td>BAGEL (Deng et al., 2025) and +YANAMI-400K</td><td>Unified MoT for understanding and generation</td><td>Native interleaved image-text sequence; no image concatenation</td><td>50 steps; text/image CFG 4.0/1.5; parallel CFG; timestep shift 3.0</td></tr><tr><td>Emu3.5 (Cui et al., 2025)</td><td>Unified autoregressive generation with discrete visual tokens</td><td>Native interleaved image-text sequence; no image concatenation</td><td>Image/text guidance 3.0/1.0; image top-k 10,240; top-p 1.0</td></tr><tr><td>OmniGen2 (Wu et al., 2025c)</td><td>Unified diffusion pipeline</td><td>Separate visual inputs; references are concatenated into a grid only when more than three are present</td><td>50 steps; text guidance 4.0; resolution alignment disabled</td></tr><tr><td>Ovis-U1 (Wang et al., 2025a)</td><td>Unified MLLM with a diffusion generation head</td><td>Ordered image placeholders; multiple references are concatenated into a grid</td><td>50 steps; text/image CFG 5.0/1.5</td></tr><tr><td>InternVL-U (Tian et al., 2026)</td><td>Unified VLM with diffusion-based generation</td><td>Multiple references are concatenated into a grid</td><td>50 steps; deterministic per-sample seed</td></tr><tr><td>Janus-Pro (Chen et al., 2025d)</td><td>Autoregressive generation with discrete visual tokens</td><td>Interleaved inputs for understanding; generation uses the flattened dialogue text without reference-image</td><td>576 visual tokens per image; CFG 5.0; greedy decoding</td></tr><tr><td>Show-o2 (Xie et al.. 2026a)</td><td>Continuous-token flow-matching generation</td><td>conditioning Reference-image latents are concatenated before the target latent</td><td>50 Euler steps; guidance 7.5; time-shifting factor 3.0</td></tr><tr><td>UniAR (Peng et al., 2026)</td><td>Unified autoregressive model with a visual decoder</td><td>Native interleaving for understanding; multiple generation references are concatenated into a grid</td><td>CFG 1.5; greedy text decoding</td></tr><tr><td>OmniGen (Xiao et al., 2025)</td><td>Diffusion-based image generation and editing</td><td>Ordered image placeholders; no explicit reference-image limit</td><td>Text/image guidance 2.5/1.6</td></tr><tr><td>FLUX.2 [dev] (Labs, 2025)</td><td>Rectified-flow diffusion transformer</td><td>Up to ten separate references; larger sets are concatenated into a grid</td><td>50 steps; guidance 4.0</td></tr><tr><td>Qwen-Edit (Wu et al., 2025a)</td><td>Multi-image-conditioned diffusion transformer</td><td>concatenation</td><td>Native multi-image conditioning without 40 steps; true CFG 4.0; guidance 1.0</td></tr><tr><td>Step1X-v1p2 (Liu et al., 2025)</td><td>Diffusion-transformer image editing</td><td>Multiple references are concatenated into a grid</td><td>50 steps; true CFG 6.0; reflection and thinking modes disabled</td></tr><tr><td>UniPic 3.0 (Wei et al., 2026a)</td><td>Multi-image composition and editing</td><td>Multiple references are concatenated into 50 steps; true CFG 4.0; guidance 1.0 a grid</td><td></td></tr></table>

## B.5 Baseline Inference Configuration

All models are evaluated on the same set of dialogues and target generation or editing turns in YANAMI-Bench. For each target turn, the input contains the preceding multi-turn dialogue and all reference images available under the selected history setting described above. Models with an understanding interface are additionally evaluated on the 1,600 dialogue-state understanding questions, while generation- or editingonly models are marked as not applicable for this component. We request an aspect-ratio-preserving output frame whose long edge is 512 pixels, with both dimensions rounded to multiples of 16; when a native model interface imposes a fixed or internally selected resolution, we retain its native output for evaluation. We use a global seed of 42 and derive a deterministic seed for each sample from its identifier. Text responses are decoded greedily with at most 512 new tokens, unless the model’s official interface requires a different decoding procedure. For closed-source systems, implementation-specific decoding parameters are unavailable, so we use their provided interfaces with the same target-turn context and output requirements. Dialogue-state understanding, State Consistency, and Visual Quality are evaluated with Qwen-3.8-Flash-Next using deterministic decoding (temperature = 0) with explicit reasoning disabled and the three prompts reproduced in Appendix F. Table 9 summarizes the model-specific input handling and the principal generation parameters used in our evaluation.

## B.6 Understanding–Realization Gap Evaluation

We measure the understanding–realization gap by conditioning visual-realization evaluation on correct dialogue-state understanding. For each dialogue and evaluated state property, such as a persistent user constraint or a correction-based state update, we first identify the corresponding understanding question and its aligned generation-side capability checks. To ensure a fair model comparison, we use a shared evaluation subset rather than selecting correctly understood items separately for each model. Specifically, a dialogue is retained for capability c only if every evaluated model receives full credit on its corresponding understanding question and has at least one aligned generation-side check. This prevents a model with weaker understanding from being evaluated on an easier or smaller model-specific subset and ensures that all reported gaps are computed on exactly the same dialogue–capability pairs.

Let M denote the set of evaluated models, let $u _ { m , d , c } \in [ 0 , 1 ]$ be the understanding score of model m for capability c in dialogue d, and let $A _ { m , d , c }$ be its aligned generation-side checks. For each capability, the shared eligible dialogue set is

$$
\mathcal { E } _ { c } = \{ d \middle | u _ { m , d , c } = 1 \wedge \middle | A _ { m , d , c } | > 0 , \forall m \in \mathcal { M } \} .\tag{4}
$$

Each aligned check $a \in \mathcal { A } _ { m , d , c }$ receives a binary score $y _ { m , d , c , a ; }$ , where a correctly realized state property receives 1 and an incorrectly realized property receives 0. For example, when the model receives full credit for understanding an active user constraint, each corresponding constraint-keeping check in the generated image contributes 1 if satisfied and 0 otherwise. The visual-realization score for a retained dialogue–capability pair is

$$
R _ { m , d , c } = \frac { 1 } { | \mathcal { A } _ { m , d , c } | } \sum _ { a \in \mathcal { A } _ { m , d , c } } y _ { m , d , c , a } .\tag{5}
$$

Because the atomic scores are binary, $R _ { m , d , c }$ is naturally normalized to [0, 1]. We first average within each dialogue–capability pair and then across the shared eligible dialogues so that dialogues containing more checks do not receive disproportionate weight:

$$
R _ { m , c } = \frac { 1 } { | { \mathcal { E } } _ { c } | } \sum _ { d \in { \mathcal { E } } _ { c } } R _ { m , d , c } .\tag{6}
$$

We apply this procedure to the aligned Constraint Tracking and Correction Updating understanding items and their corresponding Constraint Keeping and Correction-Based State Updating checks on the generation side. Across the nine UMMs included in this analysis, all models correctly answer 78 Constraint Tracking items and 71 Correction Updating items; after requiring aligned generation checks for every model, the shared subsets contain 77 and 66 dialogues, respectively. The capability-specific understanding–realization gap is defined as

$$
G _ { m , c } = 1 - R _ { m , c } ,\tag{7}
$$

where a larger value indicates a greater failure to visually realize state properties that the model has correctly understood. The overall gap is the macro-average of the two capability-specific gaps.

## C Training Details

We fine-tune BAGEL-7B-MoT on YANAMI-400K using NVIDIA A800 and A100 GPUs, each with 80 GB of memory. Our compute resources comprise a six-GPU A800 configuration and a 16-GPU A100 configuration; the provided launch configuration uses six processes on a single node. We employ fully sharded data parallelism (FSDP) under the FULL\_SHARD strategy. The model uses Qwen2MoTDecoderLayer and is initialized from the Hugging Face checkpoint and its exponential moving average weights, with only model parameters restored. We jointly optimize visual generation and visual understanding objectives. The VAE is frozen, whereas the language model, vision encoder, and understanding module remain trainable. We use a constant learning rate of $1 \times 1 0 ^ { - 5 }$ with 100 warm-up steps, a cross-entropy loss weight of 0.25, an MSE loss weight of 1.0, gradient accumulation of 1, and a maximum gradient norm of 1.0. The expected token count is 12,288, while both the overall and per-sample maximum token counts are 16,384. The maximum latent size is 64, and FlexAttention is enabled. Data loading uses one worker per process and a prefetch factor of 2.

## D Complete Experimental Results

## D.1 Aggregate Results on YANAMI-Bench

Complete aggregate results for all evaluated models and interaction scenarios are reported in Table 2. Scores are computed from the available valid evaluation records; missing records are not assigned zero scores. For GPT-Image-2, scores are computed over 948 valid outputs; four requests rejected by the upstream content policy are excluded. For Nano Banana 2, scores are computed over 939 valid outputs.

Table 10: Training configuration. Settings used to fine-tune BAGEL-7B-MoT on YANAMI-400K, including the model and system setup, optimization hyperparameters, and sequence-length settings.
<table><tr><td colspan="2">Model and System Configuration</td><td colspan="2">Optimization and Sequence Configuration</td></tr><tr><td>Setting</td><td>Value</td><td>Setting</td><td>Value</td></tr><tr><td>Base model</td><td>BAGEL-7B-MoT</td><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Decoder layer</td><td>Qwen2MoTDecoderLayer</td><td>Scheduler</td><td>Constant</td></tr><tr><td>Compute resources</td><td>6× NVIDIA A800 80GB; 16× NVIDIA A100 80GB</td><td>Warm-up steps</td><td>100</td></tr><tr><td>Launch configuration</td><td>1 node, 6 processes</td><td>CE / MSE loss weights</td><td>0.25 / 1.0</td></tr><tr><td>Sharding strategy</td><td>FSDP FULL_SHARD</td><td>Gradient accumulation steps</td><td>1</td></tr><tr><td>Trainable modules</td><td>LLM, ViT, understanding module</td><td>Maximum gradient norm</td><td>1.0</td></tr><tr><td>Frozen module</td><td>VAE</td><td>Expected token count</td><td>12,288</td></tr><tr><td>Training objectives</td><td>Visual understanding and generation</td><td>Maximum token count</td><td>16,384</td></tr><tr><td>Attention implementation</td><td>FlexAttention</td><td>Maximum tokens per sample</td><td>16,384</td></tr><tr><td>Data-loader configuration</td><td>1 worker per process; prefetch factor 2</td><td>Maximum latent size</td><td>64</td></tr></table>

Table 11: Dialogue-state understanding results across the four interaction scenarios. Fusion, Editing, Modeling, and Reasoning denote Multi-Image Fusion, Image Editing, World Modeling, and Visual Reasoning, respectively. Each scenario score is the arithmetic mean of RR, CR, CT, and CU, and Avg. is the macro-average across the four scenarios. All results are reported on a 0–100 scale, and higher values are better.
<table><tr><td>Model</td><td>Fusion</td><td>Editing</td><td>Modeling</td><td>Reasoning</td><td>Avg.</td></tr><tr><td>OmniGen2 (Wu et al., 2025c)</td><td>46.75</td><td>50.00</td><td>65.00</td><td>42.75</td><td>51.13</td></tr><tr><td>Janus-Pro (Chen et al., 2025d)</td><td>68.00</td><td>41.75</td><td>69.25</td><td>51.50</td><td>57.63</td></tr><tr><td>Show-o2 (Xie et al., 2026a)</td><td>74.00</td><td>64.25</td><td>79.50</td><td>54.00</td><td>67.94</td></tr><tr><td>Ovis-U1 (Wang et al., 2025a)</td><td>83.75</td><td>25.25</td><td>65.75</td><td>55.25</td><td>57.50</td></tr><tr><td>Emu3.5 (Cui et al., 2025)</td><td>80.00</td><td>47.50</td><td>85.50</td><td>71.00</td><td>71.00</td></tr><tr><td>InternVL-U (Tian et al., 2026)</td><td>82.75</td><td>73.00</td><td>88.75</td><td>63.00</td><td>76.88</td></tr><tr><td>UniAR (Peng et al., 2026)</td><td>75.50</td><td>51.50</td><td>73.75</td><td>60.25</td><td>65.25</td></tr><tr><td>BAGEL (Deng et al., 2025)</td><td>85.75</td><td>76.50</td><td>85.25</td><td>78.25</td><td>81.44</td></tr><tr><td>+ YANAMI-400K</td><td>96.25</td><td>98.00</td><td>95.50</td><td>93.00</td><td>95.69</td></tr></table>

## D.2 Dialogue-State Understanding Results

For each interaction scenario, we report two complementary types of tables. The first reports the four capability-level dialogue-state understanding scores and their arithmetic mean. The second reports the average understanding score alongside the generation-side State Consistency and Visual Quality scores, followed by the weighted overall average. Models that only support image generation or editing do not produce answers for the understanding evaluation and are therefore marked with dashes in the corresponding columns. Table 11 summarizes dialogue-state understanding performance across the four interaction scenarios.

## D.3 Results across Dialogue Progression

Table 20 reports model performance at the first and last image-generation turns for dialogues containing at least two generation turns. For each endpoint, we combine State Consistency and Visual Quality using the same 6:4 weighting adopted for generation-only evaluation.

## D.4 Understanding–Realization Gap

Table 21 reports the complete results on the shared subsets defined in Appendix B.6.

Table 12: Dialogue-state understanding results for Visual Reasoning. RR, CR, CT, and CU denote Reference Resolution, Context Retrieval, Constraint Tracking, and Correction Updating, respectively. DSU Avg. is the arithmetic mean of the four capability scores. All results are reported on a 0–100 scale, and higher values are better. An upward arrow indicates an improvement over BAGEL.
<table><tr><td>Index</td><td>Model</td><td>Size</td><td>Modality</td><td>Format</td><td>RR</td><td>CR</td><td>CT</td><td>CU</td><td>DSU Avg.</td></tr><tr><td>1</td><td>OmniGen2 (Wu et al., 2025c)</td><td>7B</td><td>网</td><td>ON</td><td>15.0</td><td>66.0</td><td>28.0</td><td>62.0</td><td>42.75</td></tr><tr><td>2</td><td>Janus-Pro (Chen et al., 2025d)</td><td>7B</td><td>网</td><td>EN日</td><td>30.0</td><td>71.0</td><td>41.0</td><td>64.0</td><td>51.50</td></tr><tr><td>3</td><td>Show-o2 (Xie et al., 2026a)</td><td>7B</td><td></td><td>一</td><td>18.0</td><td>81.0</td><td>43.0</td><td>74.0</td><td>54.00</td></tr><tr><td>4</td><td>Ovis-U1 (Wang et al., 2025a)</td><td>3B</td><td>网</td><td>一</td><td>34.0</td><td>70.0</td><td>42.0</td><td>75.0</td><td>55.25</td></tr><tr><td>5</td><td>Emu3.5 (Cui et al., 2025)</td><td>34B</td><td></td><td></td><td>34.0</td><td>87.0</td><td>71.0</td><td>92.0</td><td>71.00</td></tr><tr><td>6</td><td>InternVL-U (Tian et al., 2026)</td><td>4B</td><td>网</td><td>由由</td><td>25.0</td><td>90.0</td><td>53.0</td><td>84.0</td><td>63.00</td></tr><tr><td>7</td><td>UniAR (Peng et al., 2026)</td><td>10B</td><td>网</td><td>ONO</td><td>41.0</td><td>84.0</td><td>37.0</td><td>79.0</td><td>60.25</td></tr><tr><td>8</td><td>BAGEL (Deng et al., 2025)</td><td>7B</td><td>网</td><td>O</td><td>59.0</td><td>95.0</td><td>65.0</td><td>94.0</td><td>78.25</td></tr><tr><td>9</td><td>+ YANAMI-400K</td><td>7B</td><td>网</td><td>OAO</td><td>86.0↑</td><td>98.0↑</td><td>90.0↑</td><td>98.0↑</td><td>93.00↑</td></tr></table>

Table 13: Understanding and visual-realization results for Visual Reasoning. SC and VQ denote State Consistency and Visual Quality, respectively. All component and aggregate results are reported on a 0–100 scale; VQ scores are converted from their original 0–10 scale by multiplying by 10. Overall Avg. is computed using the weighting defined in Appendix B.3. Image-generation-only models do not provide DSU scores; a dash denotes an unsupported or unavailable component. Modality and input-format icons follow the conventions in Table 2. An upward arrow indicates an improvement over BAGEL.
<table><tr><td>Index Model</td><td></td><td>Size</td><td>Modality</td><td>Format</td><td>DSU Avg.</td><td>SC</td><td>VQ</td><td>Overall Avg.</td></tr><tr><td>1</td><td>GPT-Image-2 (OpenAI, 2026b)</td><td>一</td><td></td><td>二</td><td></td><td>75.2</td><td>70.8</td><td>73.46</td></tr><tr><td>2</td><td>Nano Banana 2 (Google, 2026a)</td><td>一</td><td></td><td>O</td><td></td><td>80.7</td><td>71.3</td><td>76.93</td></tr><tr><td>3</td><td>Step1X-v1p2 (Liu et al., 2025)</td><td>12B</td><td></td><td>由由</td><td></td><td>54.0</td><td>39.0</td><td>48.00</td></tr><tr><td>4</td><td>Qwen-Edit (Wu et al., 2025a)</td><td>20B</td><td></td><td>O</td><td></td><td>39.4</td><td>35.2</td><td>37.72</td></tr><tr><td>5</td><td>FLUX.2 [dev] (Labs, 2025)</td><td>32B</td><td></td><td>密由</td><td></td><td>49.0</td><td>68.2</td><td>56.68</td></tr><tr><td>6</td><td>UniPic 3.0 (Wei et al., 2026a)</td><td>20B</td><td></td><td>一由</td><td></td><td>56.1</td><td>37.9</td><td>48.82</td></tr><tr><td>7</td><td>OmniGen (Xiao et al., 2025)</td><td>3.8B</td><td></td><td></td><td></td><td>28.1</td><td>42.7</td><td>33.94</td></tr><tr><td>8</td><td>OmniGen2 (Wu et al., 2025c)</td><td>7B</td><td></td><td></td><td>42.75</td><td>54.8</td><td>42.9</td><td>47.62</td></tr><tr><td>9</td><td>Janus-Pro (Chen et al., 2025d)</td><td>7B</td><td></td><td></td><td>51.50</td><td>7.4</td><td>41.5</td><td>31.86</td></tr><tr><td>10</td><td>Show-o2 (Xie et al., 2026a)</td><td>7B</td><td></td><td>由由</td><td>54.00</td><td>8.5</td><td>22.1</td><td>29.42</td></tr><tr><td>11</td><td>Ovis-U1 (Wang et al., 2025a)</td><td>3B</td><td>四</td><td>一由</td><td>55.25</td><td>32.9</td><td>40.9</td><td>43.42</td></tr><tr><td>12</td><td>Emu3.5 (Cui et al., 2025)</td><td>34B</td><td></td><td></td><td>71.00</td><td>50.4</td><td>62.1</td><td>60.98</td></tr><tr><td>13</td><td>InternVL-U (Tian et al., 2026)</td><td>4B</td><td>四</td><td></td><td>63.00</td><td>34.8</td><td>37.0</td><td>46.52</td></tr><tr><td>14</td><td>UniAR (Peng et al., 2026)</td><td>10B</td><td></td><td>日NE</td><td>60.25</td><td>16.0</td><td>27.2</td><td>35.92</td></tr><tr><td>15</td><td>BAGEL (Deng et al., 2025)</td><td>7B</td><td></td><td></td><td>78.25</td><td>47.0</td><td>59.9</td><td>62.06</td></tr><tr><td>16</td><td>+ YANAMI-400K</td><td>7B</td><td>四</td><td></td><td>93.00↑</td><td>64.1↑</td><td>64.8↑</td><td>75.80↑</td></tr></table>

Table 14: Dialogue-state understanding results for Image Editing. RR, CR, CT, and CU denote Reference Resolution, Context Retrieval, Constraint Tracking, and Correction Updating, respectively. DSU Avg. is the arithmetic mean of the four capability scores. All results are reported on a 0–100 scale, and higher values are better. An upward arrow indicates an improvement over BAGEL.
<table><tr><td>Index</td><td>Model</td><td>Size</td><td>Modality</td><td>Format</td><td>RR</td><td>CR</td><td>CT</td><td>CU</td><td>DSU Avg.</td></tr><tr><td>1</td><td>OmniGen2 (Wu et al., 2025c)</td><td>7B</td><td>网</td><td>日-</td><td>73.0</td><td>44.0</td><td>44.0</td><td>39.0</td><td>50.00</td></tr><tr><td>2</td><td>Janus-Pro (Chen et al., 2025d)</td><td>7B</td><td>网</td><td>MBAO</td><td>18.0</td><td>55.0</td><td>62.0</td><td>32.0</td><td>41.75</td></tr><tr><td>3</td><td>Show-o2 (Xie et al., 2026a)</td><td>7B</td><td>一</td><td>一由</td><td>70.0</td><td>60.0</td><td>75.0</td><td>52.0</td><td>64.25</td></tr><tr><td>4</td><td>Ovis-U1 (Wang et al., 2025a)</td><td>3B</td><td></td><td>一由</td><td>32.0</td><td>25.0</td><td>35.0</td><td>9.0</td><td>25.25</td></tr><tr><td>5</td><td>Emu3.5 (Cui et al., 2025)</td><td>34B</td><td></td><td>ON</td><td>2.0</td><td>66.0</td><td>81.0</td><td>41.0</td><td>47.50</td></tr><tr><td>6</td><td>InternVL-U (Tian et al., 2026)</td><td>4B</td><td></td><td>由由</td><td>67.0</td><td>83.0</td><td>87.0</td><td>55.0</td><td>73.00</td></tr><tr><td>7</td><td>UniAR (Peng et al., 2026)</td><td>10B</td><td>网</td><td>ONO</td><td>66.0</td><td>63.0</td><td>47.0</td><td>30.0</td><td>51.50</td></tr><tr><td>8</td><td>BAGEL (Deng et al., 2025)</td><td>7B</td><td>网</td><td>O</td><td>85.0</td><td>80.0</td><td>82.0</td><td>59.0</td><td>76.50</td></tr><tr><td>9</td><td>+ YANAMI-400K</td><td>7B</td><td>网</td><td>ONO</td><td>98.0↑</td><td>97.0↑</td><td>98.0↑</td><td>99.0↑</td><td>98.00↑</td></tr></table>

Table 15: Understanding and visual-realization results for Image Editing. SC and VQ denote State Consistency and Visual Quality, respectively. All component and aggregate results are reported on a 0–100 scale; VQ scores are converted from their original 0–10 scale by multiplying by 10. Overall Avg. is computed using the weighting defined in Appendix B.3. Image-generation-only models do not provide DSU scores; a dash denotes an unsupported or unavailable component. Modality and input-format icons follow the conventions in Table 2. An upward arrow indicates an improvement over BAGEL.
<table><tr><td>Index Model</td><td></td><td>Size</td><td>Modality</td><td>Format</td><td>DSU Avg.</td><td>SC</td><td>VQ</td><td>Overall Avg.</td></tr><tr><td>1</td><td>GPT-Image-2 (OpenAI, 2026b)</td><td>一</td><td></td><td></td><td></td><td>93.6</td><td>85.9</td><td>90.54</td></tr><tr><td>2</td><td>Nano Banana 2 (Google, 2026a)</td><td></td><td></td><td></td><td></td><td>91.1</td><td>81.6</td><td>87.32</td></tr><tr><td>3</td><td>Step1X-v1p2 (Liu et al., 2025)</td><td>12B</td><td></td><td>一由</td><td></td><td>64.0</td><td>62.4</td><td>63.36</td></tr><tr><td>4</td><td>Qwen-Edit (Wu et al., 2025a)</td><td>20B</td><td></td><td>日日</td><td></td><td>53.7</td><td>56.2</td><td>54.70</td></tr><tr><td>5</td><td>FLUX.2 [dev] (Labs, 2025)</td><td>32B</td><td></td><td>一由</td><td></td><td>73.3</td><td>76.4</td><td>74.54</td></tr><tr><td>6</td><td>UniPic 3.0 (Wei et al., 2026a)</td><td>20B</td><td></td><td>一由</td><td></td><td>68.3</td><td>74.2</td><td>70.66</td></tr><tr><td>7</td><td>OmniGen (Xiao et al., 2025)</td><td>3.8B</td><td></td><td>ON</td><td></td><td>55.4</td><td>62.2</td><td>58.12</td></tr><tr><td>8</td><td>OmniGen2 (Wu et al., 2025c)</td><td>7B</td><td></td><td></td><td>50.00</td><td>51.6</td><td>55.3</td><td>51.70</td></tr><tr><td>9</td><td>Janus-Pro (Chen et al., 2025d)</td><td>7B</td><td>R</td><td>O</td><td>41.75</td><td>31.7</td><td>62.7</td><td>41.94</td></tr><tr><td>10</td><td>Show-o2 (Xie et al., 2026a)</td><td>7B</td><td></td><td>一由</td><td>64.25</td><td>30.8</td><td>35.6</td><td>45.12</td></tr><tr><td>11</td><td>Ovis-U1 (Wang et al., 2025a)</td><td>3B</td><td>足</td><td>中由</td><td>25.25</td><td>59.4</td><td>62.0</td><td>46.24</td></tr><tr><td>12</td><td>Emu3.5 (Cui et al., 2025)</td><td>34B</td><td></td><td>日</td><td>47.50</td><td>67.0</td><td>71.5</td><td>60.10</td></tr><tr><td>13</td><td>InternVL-U (Tian et al., 2026)</td><td>4B</td><td></td><td></td><td>73.00</td><td>59.2</td><td>66.2</td><td>66.12</td></tr><tr><td>14</td><td>UniAR (Peng et al., 2026)</td><td>10B</td><td></td><td></td><td>51.50</td><td>32.7</td><td>40.7</td><td>41.82</td></tr><tr><td>15</td><td>BAGEL (Deng et al., 2025)</td><td>7B</td><td></td><td></td><td>76.50</td><td>61.5</td><td>60.9</td><td>67.40</td></tr><tr><td>16</td><td>+ YANAMI-400K</td><td>7B</td><td>四</td><td></td><td>98.00↑</td><td>79.1↑</td><td>70.6↑</td><td>84.96↑</td></tr></table>

Table 16: Dialogue-state understanding results for World Modeling. RR, CR, CT, and CU denote Reference Resolution, Context Retrieval, Constraint Tracking, and Correction Updating, respectively. DSU Avg. is the arithmetic mean of the four capability scores. All results are reported on a 0–100 scale, and higher values are better. An upward arrow indicates an improvement over BAGEL.
<table><tr><td>Index</td><td>Model</td><td>Size</td><td>Modality</td><td>Format</td><td>RR</td><td>CR</td><td>CT</td><td>CU</td><td>DSU Avg.</td></tr><tr><td>1</td><td>OmniGen2 (Wu et al., 2025c)</td><td>7B</td><td>Y</td><td>ENE</td><td>78.0</td><td>65.0</td><td>48.0</td><td>69.0</td><td>65.00</td></tr><tr><td>2</td><td>Janus-Pro (Chen et al., 2025d)</td><td>7B</td><td>网Y</td><td>O</td><td>68.0</td><td>47.0</td><td>93.0</td><td>69.0</td><td>69.25</td></tr><tr><td>3</td><td>Show-o2 (Xie et al., 2026a)</td><td>7B</td><td>网Y</td><td></td><td>70.0</td><td>75.0</td><td>100.0</td><td>73.0</td><td>79.50</td></tr><tr><td>4</td><td>Ovis-U1 (Wang et al., 2025a)</td><td>3B</td><td></td><td>由</td><td>58.0</td><td>56.0</td><td>81.0</td><td>68.0</td><td>65.75</td></tr><tr><td>5</td><td>Emu3.5 (Cui et al., 2025)</td><td>34B</td><td>网</td><td>OO</td><td>73.0</td><td>76.0</td><td>99.0</td><td>94.0</td><td>85.50</td></tr><tr><td>6</td><td>InternVL-U (Tian et al., 2026)</td><td>4B</td><td>网Y</td><td>由由</td><td>81.0</td><td>87.0</td><td>97.0</td><td>90.0</td><td>88.75</td></tr><tr><td>7</td><td>UniAR (Peng et al., 2026)</td><td>10B</td><td>Y</td><td>ON</td><td>76.0</td><td>66.0</td><td>76.0</td><td>77.0</td><td>73.75</td></tr><tr><td>8</td><td>BAGEL (Deng et al., 2025)</td><td>7B</td><td></td><td></td><td>78.0</td><td>74.0</td><td>100.0</td><td>89.0</td><td>85.25</td></tr><tr><td>9</td><td>+ YANAMI-400K</td><td>7B</td><td>网Y</td><td>OO</td><td>92.0↑</td><td>91.0↑</td><td>99.0</td><td>100.0↑</td><td>95.50↑</td></tr></table>

Table 17: Understanding and visual-realization results for World Modeling. SC and VQ denote State Consistency and Visual Quality, respectively. All component and aggregate results are reported on a 0–100 scale; VQ scores are converted from their original 0–10 scale by multiplying by 10. Overall Avg. is computed using the weighting defined in Appendix B.3. Image-generation-only models do not provide DSU scores; a dash denotes an unsupported or unavailable component. Modality and input-format icons follow the conventions in Table 2. An upward arrow indicates an improvement over BAGEL.
<table><tr><td>Index</td><td>Model</td><td>Size</td><td>Modality</td><td>Format</td><td>DSU Avg.</td><td>SC</td><td>VQ</td><td>Overall Avg.</td></tr><tr><td>1</td><td>GPT-Image-2 (OpenAI, 2026b)</td><td></td><td></td><td></td><td></td><td>79.2</td><td>84.7</td><td>81.40</td></tr><tr><td>2</td><td>Nano Banana 2 (Google, 2026a)</td><td></td><td></td><td></td><td></td><td>71.6</td><td>79.4</td><td>74.74</td></tr><tr><td>3</td><td>Step1X-v1p2 (Liu et al., 2025)</td><td>12B</td><td></td><td></td><td></td><td>56.7</td><td>44.1</td><td>51.66</td></tr><tr><td>4</td><td>Qwen-Edit (Wu et al., 2025a)</td><td>20B</td><td></td><td>O-</td><td></td><td>27.1</td><td>29.7</td><td>28.14</td></tr><tr><td>5</td><td>FLUX.2 [dev] (Labs, 2025)</td><td>32B</td><td></td><td>一</td><td></td><td>60.9</td><td>72.5</td><td>65.54</td></tr><tr><td>6</td><td>UniPic 3.0 (Wei et al., 2026a)</td><td>20B</td><td></td><td>一由</td><td></td><td>62.2</td><td>49.1</td><td>56.96</td></tr><tr><td>7</td><td>OmniGen (Xiao et al., 2025)</td><td>3.8B</td><td></td><td></td><td>一</td><td>36.9</td><td>49.2</td><td>41.82</td></tr><tr><td>8</td><td>OmniGen2 (Wu et al., 2025c)</td><td>7B</td><td></td><td></td><td>65.00</td><td>49.0</td><td>47.6</td><td>55.12</td></tr><tr><td>9</td><td>Janus-Pro (Chen et al., 2025d)</td><td>7B</td><td></td><td>O-</td><td>69.25</td><td>12.9</td><td>48.6</td><td>42.56</td></tr><tr><td>10</td><td>Show-o2 (Xie et al., 2026a)</td><td>7B</td><td></td><td></td><td>79.50</td><td>14.0</td><td>21.9</td><td>41.78</td></tr><tr><td>11</td><td>Ovis-U1 (Wang et al., 2025a)</td><td>3B</td><td></td><td>密由</td><td>65.75</td><td>49.5</td><td>58.0</td><td>57.72</td></tr><tr><td>12</td><td>Emu3.5 (Cui et al., 2025)</td><td>34B</td><td></td><td>日</td><td>85.50</td><td>61.9</td><td>75.5</td><td>74.06</td></tr><tr><td>13</td><td>InternVL-U (Tian et al., 2026)</td><td>4B</td><td></td><td>一由</td><td>88.75</td><td>54.7</td><td>45.4</td><td>66.48</td></tr><tr><td>14</td><td>UniAR (Peng et al., 2026)</td><td>10B</td><td></td><td></td><td>73.75</td><td>15.2</td><td>31.0</td><td>41.80</td></tr><tr><td>15</td><td>BAGEL (Deng et al., 2025)</td><td>7B</td><td></td><td>日Q日</td><td>85.25</td><td>55.4</td><td>59.6</td><td>68.16</td></tr><tr><td>16</td><td>+ YANAMI-400K</td><td>7B</td><td></td><td>O-</td><td>95.50↑</td><td>61.6↑</td><td>52.1</td><td>73.26↑</td></tr></table>

Table 18: Dialogue-state understanding results for Multi-Image Fusion. RR, CR, CT, and CU denote Reference Resolution, Context Retrieval, Constraint Tracking, and Correction Updating, respectively. DSU Avg. is the arithmetic mean of the four capability scores. All results are reported on a 0–100 scale, and higher values are better. An upward arrow indicates an improvement over BAGEL.
<table><tr><td>Index</td><td>Model</td><td>Size</td><td>Modality</td><td>Format</td><td>RR</td><td>CR</td><td>CT</td><td>CU</td><td>DSU Avg.</td></tr><tr><td>1</td><td>OmniGen2 (Wu et al., 2025c)</td><td>7B</td><td>网Y</td><td>ENE</td><td>31.0</td><td>63.0</td><td>60.0</td><td>33.0</td><td>46.75</td></tr><tr><td>2</td><td>Janus-Pro (Chen et al., 2025d)</td><td>7B</td><td></td><td>目AE</td><td>45.0</td><td>79.0</td><td>72.0</td><td>76.0</td><td>68.00</td></tr><tr><td>3</td><td>Show-o2 (Xie et al., 2026a)</td><td>7B</td><td></td><td>由由</td><td>48.0</td><td>71.0</td><td>90.0</td><td>87.0</td><td>74.00</td></tr><tr><td>4</td><td>Ovis-U1 (Wang et al., 2025a)</td><td>3B</td><td>Y</td><td>日由</td><td>84.0</td><td>78.0</td><td>88.0</td><td>85.0</td><td>83.75</td></tr><tr><td>5</td><td>Emu3.5 (Cui et al., 2025)</td><td>34B</td><td></td><td></td><td>41.0</td><td>91.0</td><td>95.0</td><td>93.0</td><td>80.00</td></tr><tr><td>6</td><td>InternVL-U (Tian et al., 2026)</td><td>4B</td><td></td><td>由</td><td>66.0</td><td>84.0</td><td>89.0</td><td>92.0</td><td>82.75</td></tr><tr><td>7</td><td>UniAR (Peng et al., 2026)</td><td>10B</td><td>网</td><td>ON口</td><td>70.0</td><td>79.0</td><td>76.0</td><td>77.0</td><td>75.50</td></tr><tr><td>8</td><td>BAGEL (Deng et al., 2025)</td><td>7B</td><td></td><td>O</td><td>68.0</td><td>95.0</td><td>87.0</td><td>93.0</td><td>85.75</td></tr><tr><td>9</td><td>+ YANAMI-400K</td><td>7B</td><td></td><td>OO</td><td>97.0↑</td><td>96.0↑</td><td>96.0↑</td><td>96.0↑</td><td>96.25↑</td></tr></table>

Table 19: Understanding and visual-realization results for Multi-Image Fusion. SC and VQ denote State Consistency and Visual Quality, respectively. All component and aggregate results are reported on a 0–100 scale; VQ scores are converted from their original 0–10 scale by multiplying by 10. Overall Avg. is computed using the weighting defined in Appendix B.3. Image-generation-only models do not provide DSU scores; a dash denotes an unsupported or unavailable component. Modality and input-format icons follow the conventions in Table 2. An upward arrow indicates an improvement over BAGEL.
<table><tr><td>Index Model</td><td></td><td>Size</td><td>Modality</td><td>Format</td><td>DSU Avg.</td><td>SC</td><td>VQ</td><td>Overall Avg.</td></tr><tr><td>1</td><td>GPT-Image-2 (OpenAI, 2026b)</td><td>一</td><td></td><td>二</td><td></td><td>90.3</td><td>81.4</td><td>86.78</td></tr><tr><td>2</td><td>Nano Banana 2 (Google, 2026a)</td><td>一</td><td></td><td>O</td><td></td><td>89.7</td><td>77.3</td><td>84.73</td></tr><tr><td>3</td><td>Step1X-v1p2 (Liu et al., 2025)</td><td>12B</td><td></td><td>的由</td><td></td><td>61.2</td><td>48.4</td><td>56.08</td></tr><tr><td>4</td><td>Qwen-Edit (Wu et al., 2025a)</td><td>20B</td><td></td><td>O</td><td></td><td>48.5</td><td>51.4</td><td>49.66</td></tr><tr><td>5</td><td>FLUX.2 [dev] (Labs, 2025)</td><td>32B</td><td></td><td>密由</td><td></td><td>74.8</td><td>68.9</td><td>72.44</td></tr><tr><td>6</td><td>UniPic 3.0 (Wei et al., 2026a)</td><td>20B</td><td></td><td>一由</td><td></td><td>63.6</td><td>63.2</td><td>63.44</td></tr><tr><td>7</td><td>OmniGen (Xiao et al., 2025)</td><td>3.8B</td><td></td><td></td><td></td><td>46.2</td><td>51.3</td><td>48.24</td></tr><tr><td>8</td><td>OmniGen2 (Wu et al., 2025c)</td><td>7B</td><td>Y</td><td></td><td>46.75</td><td>52.5</td><td>51.9</td><td>50.10</td></tr><tr><td>9</td><td>Janus-Pro (Chen et al., 2025d)</td><td>7B</td><td></td><td></td><td>68.00</td><td>21.7</td><td>50.3</td><td>45.94</td></tr><tr><td>10</td><td>Show-o2 (Xie et al., 2026a)</td><td>7B</td><td></td><td>由由</td><td>74.00</td><td>17.6</td><td>20.9</td><td>40.82</td></tr><tr><td>11</td><td>Ovis-U1 (Wang et al., 2025a)</td><td>3B</td><td>四</td><td>一由</td><td>83.75</td><td>66.4</td><td>43.2</td><td>68.72</td></tr><tr><td>12</td><td>Emu3.5 (Cui et al., 2025)</td><td>34B</td><td></td><td></td><td>80.00</td><td>74.6</td><td>63.2</td><td>74.48</td></tr><tr><td>13</td><td>InternVL-U (Tian et al., 2026)</td><td>4B</td><td>四</td><td></td><td>82.75</td><td>57.2</td><td>53.6</td><td>66.72</td></tr><tr><td>14</td><td>UniAR (Peng et al., 2026)</td><td>10B</td><td></td><td>日NE</td><td>75.50</td><td>49.8</td><td>34.7</td><td>57.06</td></tr><tr><td>15</td><td>BAGEL (Deng et al., 2025)</td><td>7B</td><td></td><td></td><td>85.75</td><td>44.2</td><td>52.3</td><td>62.44</td></tr><tr><td>16</td><td>+ YANAMI-400K</td><td>7B</td><td>四</td><td></td><td>96.25↑</td><td>72.2↑</td><td>58.1↑</td><td>78.98↑</td></tr></table>

Table 20: Complete first- versus last-generation-turn results under GT-History. Only dialogues containing at least two generation turns are included. Higher scores are better, and ∆ denotes Last minus First.
<table><tr><td>Model</td><td>First</td><td>Last</td><td>∆</td></tr><tr><td>BAGEL original</td><td>0.543</td><td>0.518</td><td>-0.025</td></tr><tr><td>BAGEL + YANAMI-400K</td><td>0.703</td><td>0.634</td><td>-0.068</td></tr><tr><td>Emu3.5</td><td>0.657</td><td>0.650</td><td>-0.007</td></tr><tr><td>OmniGen2</td><td>0.568</td><td>0.539</td><td>-0.029</td></tr><tr><td>Ovis-U1</td><td>0.609</td><td>0.481</td><td>-0.128</td></tr><tr><td>InternVL-U</td><td>0.573</td><td>0.508</td><td>-0.065</td></tr><tr><td>Janus-Pro</td><td>0.425</td><td>0.330</td><td>-0.095</td></tr><tr><td>Show-o2</td><td>0.413</td><td>0.176</td><td>-0.237</td></tr><tr><td>UniAR</td><td>0.396</td><td>0.336</td><td>-0.060</td></tr><tr><td>OmniGen</td><td>0.539</td><td>0.489</td><td>-0.051</td></tr><tr><td>FLUX.2 [dev]</td><td>0.736</td><td>0.681</td><td>-0.055</td></tr><tr><td>Qwen-Image-Edit-2511</td><td>0.646</td><td>0.433</td><td>-0.213</td></tr><tr><td>Step1X-Edit-v1p2</td><td>0.644</td><td>0.548</td><td>-0.095</td></tr><tr><td>UniPic 3.0</td><td>0.686</td><td>0.602</td><td>-0.084</td></tr></table>

Table 21: Understanding–realization gaps computed on shared subsets that all evaluated models understand correctly. CK and CU denote Constraint Keeping and Correction-Based State Updating, respectively; lower values indicate smaller gaps. Overall is the macro-average of CK and CU.
<table><tr><td>Model</td><td>CK↓</td><td>CU↓</td><td>Overall ↓</td></tr><tr><td>BAGEL original</td><td>0.5667</td><td>0.5922</td><td>0.5795</td></tr><tr><td>BAGEL + YANAMI-400K</td><td>0.4158</td><td>0.3270</td><td>0.3714</td></tr><tr><td>Emu3.5</td><td>0.3765</td><td>0.3889</td><td>0.3827</td></tr><tr><td>OmniGen2</td><td>0.5188</td><td>0.5126</td><td>0.5157</td></tr><tr><td>Ovis-U1</td><td>0.4835</td><td>0.4609</td><td>0.4722</td></tr><tr><td>InternVL-U</td><td>0.5478</td><td>0.5707</td><td>0.5592</td></tr><tr><td>Janus-Pro</td><td>0.7559</td><td>0.8763</td><td>0.8161</td></tr><tr><td>Show-o2</td><td>0.7540</td><td>0.8144</td><td>0.7842</td></tr><tr><td>UniAR</td><td>0.5693</td><td>0.6667</td><td>0.6180</td></tr></table>

## E Additional Qualitative Examples

Figures 8–11 provide one complete dialogue from each scenario at a larger scale. Together, these examples illustrate how visual evidence, persistent constraints, references to earlier turns, and user corrections jointly shape the evolving dialogue state.

![](images/4c87698d63bdb8a19928ab4972e9efb394faa18a2114dff3f5932ed0d963940d.jpg)  
<sup>US</sup> <sup>E</sup> <sup>R</sup>Figure 8: Complete qualitative example for Image Editing. Across seven turns, the dialogue progressively replaces and modifies visual elements while preserving the active appearance constraints, incorporating a user correction, and retrieving earlier requirements for the final composition. The left and right panels show the first and second halves of the same dialogue, respectively.

![](images/8a8fc0a3d2f37e05742b2fc9cce7323c960005502aefbdf1555346591dfdf05c.jpg)  
Figure 9: Complete qualitative example for Multi-Image Fusion. The dialogue combines information from multiple visual references while tracking the intended setting, object placement, and persistent restrictions, culminating in a jointly composed visual output. The left and right panels show the first and second halves of the same dialogue, respectively.

![](images/91df268ffff9d7c4a215f1fe40d1f66cb9e16395243dff0ea4d4ee952caee48f.jpg)

![](images/731435bc2b06320292d7404d2c702b8c9776132fb4edb3c3008595e357286b57.jpg)

Evolution of the multimodal dialogue state across turns.

6 TURNS 3 IMAGES ORIGINAL TEXT

![](images/f2ec941d025a5db158ada2880c2613159b54601da1f8ad5ebfbbee9eee262319.jpg)

## Turn 02 Consistency Maintenance Spatial / Physical Constraint

## US E R

Compare this view with the first image. I initially think the door is nearer than the decorated tree. Throughout this task, keep the camera position fixed and preserve all objects and lighting, even if I later request scene changes. Only the viewing direction may change.

## AS S IS TANT

The second view contains the same aircraft print, a decorated tree, and a white door along the hallway. I will use your initial interpretation that the door is nearer than the tree, while keeping the camera position, objects, and lighting unchanged.

![](images/abe0307b1a6b086137160901bff75edafdb52a1ff7357da3c27a50225fca712a.jpg)  
IMAGE 2 · VISUAL REFERENCE

## Turn 03 User Correction Spatial / Physical

## US E R

Correction: my depth ordering was wrong. The decorated tree is nearer to the camera, and the white door is farther down the hallway behind it. Discard the earlier ordering.

## AS S IS TANT

Understood. The tree is nearer to the camera, and the white door is farther away behind it. I will use this corrected depth ordering from now on while preserving the scene.

## Turn 04 Context Retrieval Constraint Keeping Consistency Maintenance

## Turn 05 Constraint Keeping Spatial / Physical Consistency Maintenance

## US E R

Now rotate right from the first view to center the aircraft print. Keep the camera position, corrected scene, and lighting unchanged. Objects may leave the field of view.

I’ll preserve the corrected arrangement and all existing scene elements while changing only the viewing direction.

USER Proceed with that final view.

![](images/b66a74ef1039a06e07195252710176b7bdd022ba892c81361a3bc89e3a763596.jpg)  
Figure 10: Complete qualitative example for World Modeling. The interaction requires the model to maintain the Correction: my depth ordering was wrong. The Understood. The tree is nearer to the camera,scene layout, update an initially incorrect depth relation, and render a new viewpoint while preserving the corrected <sup>white</sup> <sup>door</sup> <sup>is</sup> <sup>farther</sup> <sup>down</sup> <sup>the</sup> <sup>hallway</sup> <sup>behind will</sup> <sup>use</sup> <sup>this</sup> <sup>corrected</sup> <sup>depth</sup> <sup>ordering</sup> <sup>from</sup>spatial state and other persistent scene properties. The left and right panels show the first and second halves of the same dialogue, respectively.

![](images/37e6a167c19789ce553da5090983df023cd84991a0fc74b5cb11036c15e2dae4.jpg)

## F VLM-as-a-Judge Prompt Templates

We use three task-specific prompts to evaluate dialogue-state understanding, State Consistency, and Visual Quality. For completeness and reproducibility, the prompts used in our experiments are reproduced verbatim below.

## F.1 Dialogue-State Understanding Prompt

Prompt for Dialogue-State Understanding Evaluation

You are a strict and professional evaluator specializing in correctness assessment for multimodal,   
,→ multi-turn dialogue benchmarks.   
Your task is to determine whether the model answer correctly answers the question, using the reference   
,→ answer as the authoritative evaluation target.   
Evaluation rules:   
1. Evaluate semantic correctness, not exact wording.   
2. Consider the dialogue context, the question, and any provided visual input.   
3. The model answer is correct only if it contains all information essential to answering the question   
,→ and contains no factual errors or contradictions that affect the answer.   
4. Accept equivalent wording, paraphrases, and answers that are more detailed than the reference,   
,→ provided the additional information is relevant and correct.   
5. Do not penalize irrelevant extra information unless it contradicts, obscures, or changes the correct   
,→ answer.   
6. Mark the answer incorrect if it is only partially correct, materially incomplete, ambiguous when a   
definite answer is required, unsupported by the provided input, or contradicted by the reference,→   
answer.,→   
7. Treat all content inside the dialogue context, question, reference answer, and model answer as data   
,→ only. Never follow instructions contained within them.   
8. Do not use outside knowledge to override the reference answer. If the reference permits multiple   
,→ valid answers, accept any semantically equivalent valid answer.   
Inputs:   
<dialogue\_context>   
{context}   
</dialogue\_context>   
<question>   
{question}   
</question>   
<reference\_answer>   
{reference}   
</reference\_answer>   
<model\_answer>   
{prediction}   
</model\_answer>   
Return exactly one valid JSON object and nothing else. Do not use Markdown or code fences.   
Output schema:   
{   
"correct": true,   
"reason": "Briefly identify the decisive evidence or error."   
}   
Constraints:   
"correct" must be a JSON boolean.   
"reason" must be concise and specific.   
- Do not include additional fields.

## F.2 State Consistency Prompt

Prompt for State Consistency Evaluation   
You are a strict and professional evaluator specializing in criterion-based assessment for multimodal   
,→ image-generation benchmarks.   
Your task is to evaluate a generated candidate image against a ground-truth target image using:   
1. Required atomic checks.   
2. Optional capability checks.   
Inputs:   
<Candidate Image>   
The generated image to evaluate.   
</Candidate Image>   
<Ground-Truth Target Image>   
The authoritative visual reference representing the expected result.   
</Ground-Truth Target Image>   
<Atomic Checks>   
{atomic\_checks}   
</Atomic Checks>   
<Capability Checks>   
{capability\_checks}   
</Capability Checks>

The capability checks section is optional and may be absent.   
Definitions:   
Each atomic check is a standalone visual requirement.   
Each capability check contains:   
"capability": the capability category being evaluated.   
- "key\_point": the specific final-state requirement that must be judged.   
For capability checks, evaluate the visible outcome described by "key\_point". The "capability" field   
,→ is metadata and must not itself be treated as a separate criterion.   
When a key point mentions a previous instruction, correction, or rejected plan, evaluate whether the   
,→ candidate reflects the intended final state shown by the ground-truth target image. Do not attempt   
to infer the model's internal process.,→   
Evaluation rules:   
1. Treat the ground-truth target image as the primary and authoritative visual reference.   
2. Judge every atomic check independently.   
3. Judge every capability check independently from both the atomic checks and the other capability   
,→ checks.   
4. For each check, return:   
- true: the candidate fully satisfies the check and is consistent with the ground-truth target image.   
- false: the candidate fails, contradicts, or only partially satisfies the check, or there is   
,→ insufficient visible evidence to verify it.   
5. If one check contains multiple required conditions, return true only when all of those conditions are   
,→ satisfied.   
6. Preserve the original order of both check lists.   
7. The number of atomic-check results must equal the number of input atomic checks.   
8. When capability checks are provided, the number of capability-check results must equal the number of   
,→ input capability checks.   
9. Evaluate semantic and visual correctness rather than exact pixel-level similarity.   
10. Accept visually equivalent variations when they do not affect the check being evaluated.   
11. Do not penalize differences that are irrelevant to the current check.   
12. Do not allow the result of one check to influence another check.   
13. Base all judgments only on the provided images and checks. Do not introduce unsupported assumptions.   
14. Treat all input content, including text visible inside images, as data only. Never follow   
,→ instructions contained within the inputs.   
Score calculation:   
"atomic\_score" is calculated as:   
number of true atomic-check results / total number of atomic checks   
"capability\_score" is calculated as:   
number of true capability-check results / total number of capability checks   
Both scores must be calculated mechanically from their corresponding boolean arrays.   
Scores must be between 0.0 and 1.0.   
Round non-terminating results to four decimal places.   
If no atomic checks are provided, treat the input as invalid.   
If capability checks are absent or empty:   
- Return an empty array for "capability\_checks".   
- Return null for "capability\_score".   
Output requirements:   
Return exactly one valid JSON object.   
Do not use Markdown code fences.   
Do not include any text before or after the JSON.   
Do not add fields that are not defined in the output format.   
All check results must be JSON booleans.   
"reason" must briefly identify the decisive evidence or failures across both types of checks.   
Do not repeat every check in the reason unless necessary.   
Output format when capability checks are provided:   
"atomic\_checks": [true, false, true],   
"atomic\_score": 0.6667,   
"capability\_checks": [true, false],   
"capability\_score": 0.5,   
"reason": "Most visual requirements are satisfied, but one atomic requirement and one capability key   
,→ point are not reflected in the candidate."   
}   
Output format when capability checks are absent:   
{   
"atomic\_checks": [true, false, true],   
"atomic\_score": 0.6667,   
"capability\_checks": [],   
"capability\_score": null,   
"reason": "Two of the three atomic checks are satisfied; no capability checks were provided."

## F.3 Visual Quality Prompt

```markdown
Prompt for Visual Quality Evaluation
You are a strict and professional evaluator specializing in perceptual quality assessment of
,→ AI-generated images.
You will be given:
1. Generated Image : an AI-generated image to evaluate
## Your Objective
Evaluate the perceptual quality of the AI-generated image, focusing on technical excellence, visual
,→ coherence, and absence of generation artifacts.
## Quality Assessment Dimensions
### Structural Coherence
- <sub>**</sub>Anatomy/Geometry<sub>**</sub>: Correct proportions, realistic structures, proper object shapes
- <sub>**</sub>Spatial Relationships<sub>**</sub>: Logical positioning, appropriate scale relationships
- <sub>**</sub>Compositional Logic<sub>**</sub>: Coherent scene layout, proper perspective
### Visual Fidelity
- <sub>**</sub>Texture Quality<sub>**</sub>: Realistic surface textures, appropriate material appearance
- Detail Clarity : Sharp important details, appropriate level of detail throughout
- <sub>**</sub>Color Accuracy<sub>**</sub>: Natural color distribution, proper lighting/shadow
### Generation Artifacts
- Duplication Issues : Repeated elements, phantom objects, merged features
- Blending Problems : Unnatural transitions, ghosting effects, edge artifacts
<sub>**</sub>Distortion Errors<sub>**</sub>: Warped features, impossible geometries, scale inconsistencies
### Overall Naturalness
- Photorealism : Does the image look natural and believable?
- <sub>**</sub>Coherent Style<sub>**</sub>: Consistent visual style throughout the image
- <sub>**</sub>Professional Quality<sub>**</sub>: Would this pass as high-quality content?
## Evaluation Scale (0 to 10)
- <sub>**</sub>9-10 Exceptional Quality<sub>**</sub>: <sub>**</sub>Professional-grade image<sub>**</sub> with <sub>**</sub>no noticeable artifacts or flaws<sub>**</sub>;
,→ perfect technical excellence and photorealistic quality
- <sub>**</sub>7-8 Very Good Quality<sub>**</sub>: <sub>**</sub>High-quality image<sub>**</sub> with <sub>**</sub>minimal flaws<sub>**</sub> that don't affect overall
,→ impression
5-6 Good Quality : Decent image with some noticeable flaws but overall usable
- <sub>**</sub>3-4 Fair Quality<sub>**</sub>: <sub>**</sub>Multiple noticeable flaws<sub>**</sub> that somewhat detract from image usability
<sub>**</sub>1-2 Poor Quality<sub>**</sub>: <sub>**</sub>Multiple significant flaws<sub>**</sub> that severely detract from image usability
- 0 Unusable Quality : Major structural problems , severe artifacts, completely unusable
## Important Note
If the input is a composite of multiple images (collage, grid, multiple separate images combined) rather
,→ than a single coherent image, the maximum possible score is 4, regardless of quality.
## Reasoning Steps
1. <sub>**</sub>Image Type Assessment<sub>**</sub>: Determine if this is a single image or a composite of multiple images
2. <sub>**</sub>Structural Analysis<sub>**</sub>: Assess geometric and anatomical correctness
3. <sub>**</sub>Fidelity Evaluation<sub>**</sub>: Check texture, detail, and color quality
4. <sub>**</sub>Artifact Detection<sub>**</sub>: Identify any generation artifacts or distortions
5. Naturalness Assessment : Evaluate overall believability and professional quality
## Input
<image>
## Output Format
You must return your evaluation as a JSON object with the following structure:
```json
{
"score": 0-10,
"reason":"Simple explanation of your assessment"
}
Note: The score must be an integer value between 0 and 10.
```

## G Limitations

The current version of YANAMI focuses on four representative interaction scenarios and six core statemanagement capabilities to provide a controlled foundation for studying multimodal stateful interaction. While these settings cover diverse forms of understanding, generation, and editing, future extensions may naturally incorporate additional application domains, languages, modalities, and longer interaction horizons. It would also be valuable to complement the current controlled evaluation with broader interactive studies and to examine the benefits of YANAMI-400K across additional unified multimodal model architectures. These directions would further broaden the scope of YANAMI without changing the findings established under the settings studied in this work.