# RADIOLOGY GUIDELINE-DERIVED DECISION GRAPH FOR EFFICIENT CLINICAL DECISION SUPPORT

Xuewen Zhou, Zhen Chen<sup>†</sup>

Department of Data Science and Artificial Intelligence, The Hong Kong Polytechnic University

## ABSTRACT

Radiological risk stratification and management planning are essential downstream tasks in medical imaging. Automating these tasks is challenging because clinical guideline criteria are distributed across prose, tables, and cross-referenced definitions. We present Guideline-derived Decision Graph (GDGraph), a framework that decouples guideline compilation from case evaluation. During compilation, a large language model partitions guidelines into section-based chunks, extracts local decision subgraphs with linked definitions, and aggregates related branches into an executable decision graph. During inference, a fast System One model answers localized clinical questions while deterministic traversal resolves Boolean logic, turning opaque single-pass predictions into a traceable decision path without task-specific training. Across 347 cases spanning renal cyst, adnexal mass, and thyroid nodule guidelines, GDGraph improves System One accuracy from 57.3% to 90.8%, outperforming retrievalaugmented GPT-4o (80.1%) and graph-guided GPT-4o with chain-of-thought (82.7%) at substantially lower latency and cost. The source code will be available.

Index Terms— Clinical guidelines, decision graphs, radiological decision support, System One models

## 1. INTRODUCTION

Risk stratification and management planning are essential downstream tasks in medical imaging that connect radiological findings to clinical care [1]. Clinical guidelines standardize these decisions across tasks such as renal cyst management [2], adnexal mass assessment [3], and thyroid nodule stratification [4], where final recommendations follow from explicit, discrete decision criteria. Automating these tasks therefore requires systematically following the interdependent decision steps specified in each guideline, progressing from case-specific findings through successive criteria to a final category or recommendation.

Pretrained language models can process free-text clinical reports and imaging findings directly through prompting without task-specific training [5, 6], yet have been shown to frequently deviate from established diagnostic guidelines [7].

Chain-of-thought (CoT) prompting helps these models with multi-step clinical problems by eliciting reasoning steps in text before a conclusion [8]. In guideline-based risk stratification and management, clinical criteria and conditional rules also provide a structured basis for decomposing the overall task into a series of localized decisions, such as verifying a specific imaging feature or comparing a measurement against a threshold. System One models such as Jev [9] offer a suitable interface for these localized decisions by directly scoring discrete candidate options without generating intermediate text . Thus, the primary challenge is to organize these local judgments into an explicit decision process that preserves the guideline’s conditional dependencies and yields a traceable outcome for high-stakes decisions [10].

Existing clinical frameworks incorporate guideline knowledge in different ways. For a given patient case, retrievalaugmented generation (RAG) systems [11] such as accGPT [12] retrieve relevant medical passages for the language model to interpret during response generation. CPGPrompt [13] converts segmented guideline text into decision trees for stepwise evaluation, with its reported workflow relying on manual selection of relevant text sections and manual validation of the generated trees. Such decision structures could harness the capabilities and efficiency of System One language models to support practical applications in clinical decision-making. However, automatic construction of these structures from raw guideline documents remains challenging because it requires integrating tabulated criteria and thresholds with terminology definitions and exception rules distributed across the document.

To address this, we present Guideline-derived Decision Graph (GDGraph), an automated framework that compiles raw clinical guidelines into executable decision graphs for fast, traceable System One inference (Fig. 1). By decoupling offline LLM-based guideline structuring from online deterministic traversal, GDGraph reduces multi-step classification to localized node-level judgments while recording an explicit decision trace. Across 347 radiology cases spanning three guidelines, GDGraph raises System One accuracy from 57.3% to 90.8%, outperforming unstructured RAG (80.1%) and graph-guided GPT-4o with CoT (82.7%) at substantially lower latency and cost. These results suggest the practical value of GDGraph for clinical decision support.

![](images/db8cf0faaa8ec998644e9644a9ef6d92a7d5b2ddac00b9fc8cdf95993505d85e.jpg)  
Fig. 1. Overview of the GDGraph framework. (a) LLM-based Graph Construction: Offline, an LLM partitions clinical guidelines into m section-based chunks $\{ C _ { i } \} _ { i = 1 } ^ { m }$ , extracts local subgraphs $\{ G _ { i } \} _ { i = 1 } ^ { m }$ with TF-IDF-retrieved definitions from a global memory, and aggregates related branches into a unified decision graph G. (b) Graph-Guided Selection: Online, a System One model answers localized questions using case findings and attached definitions, guiding traversal along the selected path to reach a traceable decision.

## 2. METHODOLOGY

## 2.1. Overview

Fig. 1 illustrates the proposed Guideline-derived Decision Graph (GDGraph) framework, which decouples offline guideline compilation from online case evaluation. In the offline stage (Fig. 1(a); Algorithm 1), given a clinical practice guideline D containing narrative text, figures, and tables, an LLMbased pipeline partitions D into m section-based chunks $\{ C _ { i } \} _ { i = 1 } ^ { m } ,$ , extracts local subgraphs $\{ G _ { i } \} _ { i = 1 } ^ { m }$ grounded in a global definition memory M via Term Frequency–Inverse Document Frequency (TF-IDF) retrieval [14, 15], and aggregates related branches across sections into a unified decision graph $G \ = \ ( S , { \mathcal A } , { \mathcal E } , { \mathcal y } )$ . In the online stage (Fig. 1(b)), given the imaging findings of a clinical case $X ,$ , a fast System One model answers localized questions at active nodes using attached definitions, while deterministic traversal resolves Boolean transitions along the selected path to reach a traceable category or management decision.

## 2.2. Guideline-derived Decision Graph

Graph Representation. The decision graph $G = ( \mathcal { S } , \mathcal { A } , \mathcal { E } , \mathcal { y } )$ comprises clinical states $s$ (clinical contexts or intermediate conclusions, with entry states $S _ { 0 } ~ \subset ~ S )$ , atomic conditions ${ \mathcal { A } } ,$ directed transition rules E, and candidate outcomes $\mathcal { V }$ (diagnostic categories or management actions). Each state and outcome carries a clinical label, a source-derived definition, and supporting text spans from $D .$ . Each atomic condition $a \in \ A$ represents a single traceable criterion, written as $a = \left( \tau _ { a } , \kappa _ { a } , \mathcal { D } _ { a } \right)$ , where $\tau _ { a }$ specifies the clinical feature and its anatomical or temporal scope (e.g., lesion diameter), $\kappa _ { a }$ denotes the comparison operator and value (e.g., < 10 cm or presence/absence), and $\mathcal { D } _ { a }$ links to the corresponding definitions in M. Deterministic templates render each a and its referenced definitions $\mathcal { D } _ { a }$ into a self-contained question for model evaluation.

Directed rules $e \in \mathcal { E }$ take the form $\boldsymbol { e } = \left( s _ { e } , \phi _ { e } , T _ { e } , \pi _ { e } \right)$ and connect source states to targets through Boolean combinations of atomic conditions. Here, $s _ { e } \in S$ is the prerequisite state, $\phi _ { e }$ is a logical expression over A using standard operators $( \land , \lor , \lnot$ , and AtLeast<sub>k</sub>), $T _ { e } \subseteq S \cup \mathcal { V }$ denotes the target states or outcomes, and $\pi _ { e }$ indicates whether the rule directly determines, supports, or rules out $T _ { e }$ . For instance, requiring both imaging pattern $P$ and diameter below 10 cm for the same lesion is formulated as $\phi _ { e } = a _ { P } \wedge a _ { < 1 0 }$

Section-Based Chunking and Local Subgraph Generation. We split D by document sections into chunks $C =$ $\{ C _ { 1 } , \ldots , C _ { m } \}$ while keeping tables and footnotes intact. An LLM scans all chunks to build a global definition memory M of clinical terms, aliases, and definitions. For each chunk $C _ { i } ,$ we retrieve relevant definitions $d _ { i } \subset M$ via TF-IDF similarity while keeping explicitly cited definitions, and prompt the LLM to extract states $s _ { i }$ , atomic conditions $A _ { i } ,$ and rules $\mathcal { E } _ { i }$ into a local subgraph $G _ { i }$ . Named imaging patterns are kept as single conditions linked to M, additional criteria such as numerical thresholds are separated into individual conditions, and negative findings are denoted by logical negation (¬).

Cross-Branch Aggregation. To connect related rules across sections, local subgraphs $\{ G _ { i } \} _ { i = 1 } ^ { m }$ are merged into a unified graph G. Guided by TF-IDF similarity, an LLM groups subgraphs sharing clinical features or exceptions into potentially overlapping sets $\boldsymbol { \mathcal { B } } = \{ B _ { 1 } , \ldots , B _ { L } \}$ , processed sequentially on the latest graph. Within each group $B _ { l }$ , the LLM uses the source text and definitions $d _ { l }$ to merge nodes that share identical clinical meaning and judgment criteria, while keeping non-equivalent thresholds separate. When the guideline specifies upgrading or exception clauses, the LLM is instructed to update the conditions $( \phi _ { e } )$ and targets (T<sub>e</sub>) of affected rules $e ,$ while keeping other valid standalone paths unchanged.

Algorithm 1 GDGraph Construction   
Input: Clinical guideline document D   
Output: Decision graph $G = ( \mathcal { S } , \mathcal { A } , \mathcal { E } , \mathcal { y } )$   
1: Partition D into section-based chunks $C = \{ C _ { 1 } , \ldots , C _ { m } \}$   
2: Build definition memory $M \gets \bigcup _ { i = 1 } ^ { m }$ ExtractTerms(C<sub>i</sub>)   
3: for each chunk $C _ { i } \in C$ do   
4: Retrieve relevant definitions $d _ { i } \subset M$ via TF-IDF   
5: Extract subgraph $G _ { i } = ( S _ { i } , \mathcal { A } _ { i } , \mathcal { E } _ { i } , \mathcal { V } _ { i } )$ from $( C _ { i } , d _ { i } )$   
6: Render questions for each $a \in { \mathcal { A } } _ { i }$ with $\mathcal { D } _ { a } \subset M$   
7: end for   
8: Group $\{ G _ { i } \} _ { i = 1 } ^ { m }$ into overlapping clusters   
$\boldsymbol { \mathcal { B } } = \{ B _ { 1 } , \ldots , B _ { L } \}$   
9: for each cluster $B _ { l } \in B$ sequentially do   
10: Load latest updated subgraphs in $B _ { l }$   
11: Retrieve relevant definitions $d _ { l } \subset M$ via TF-IDF   
12: Merge equivalent states and atoms across $B _ { l }$   
13: Update affected rules $\textstyle e \in \bigcup _ { G _ { i } \in B _ { l } } { \mathcal { E } } _ { i } ( \phi _ { e }$ and $T _ { e } )$   
14: end for   
15: Assemble unified graph $G = ( \mathcal { S } , \mathcal { A } , \mathcal { E } , \mathcal { y } )$   
16: return G

## 2.3. Graph-Guided Selection

Prior to evaluation, an LLM maps source-supported conclusions in S ∪Y, including intermediate states and terminal outcomes, to the target task’s standardized answer options, as illustrated in Fig. 1(b). During online inference, traversal starts at $ { \boldsymbol { S } } _ { 0 }$ . At each active state, the System One model answers the questions of outgoing atomic conditions based on the reported imaging findings and attached definitions $\mathcal { D } _ { a } ,$ while program logic evaluates $\phi _ { e }$ to advance along satisfied branches. The final prediction is derived from the task-relevant conclusions reached during traversal, with the clinical case interpreted against static option definitions. The traversal log records each evaluated question and its answer, preserving the traversed edges as a traceable decision path.

## 3. EXPERIMENTS

## 3.1. Experimental Setup

Dataset and Tasks. We evaluate GDGraph on 347 text-based radiology cases curated from PMC-Patients-V2 [16] and PubMed Central case reports across three clinical guidelines (Table 1). Cases were screened for guideline-relevant imaging findings and clinical context. Reference labels were taken from explicit source labels or derived by applying guideline criteria to the reported findings. Case descriptions were reformulated using GPT-6.1-sol to remove explicit diagnostic labels and reduce the risk of answer leakage. CUA evaluates management selection given the clinical context, whereas O-RADS and K-TIRADS assign ultrasound risk categories from imaging descriptions.

Table 1. Summary of evaluated clinical guidelines, target tasks, and case counts.
<table><tr><td>Guideline</td><td>Clinical Task</td><td>Cases</td></tr><tr><td>CUA (2023) [2]</td><td>Renal cyst management</td><td>127</td></tr><tr><td>O-RADS US v2022 [3]</td><td>Adnexal mass stratification</td><td>117</td></tr><tr><td>K-TIRADS (2021) [4]</td><td>Thyroid nodule stratification</td><td>103</td></tr><tr><td>Total</td><td></td><td>347</td></tr></table>

Implementation and Baselines. Decision graphs are constructed offline using GPT-6-Luna with low reasoning effort, processing extracted text alongside PDF page images to preserve table structures and footnotes. For online evaluation, we use the System One model Jev (v1.13.0) [9] and GPT-4o across six settings: (1) baselines without retrieval or GD-Graph (Direct Jev, Direct GPT-4o, and GPT-4o CoT), which receive only the clinical case, candidate options, and option definitions; (2) retrieval-augmented generation (RAG+GPT-4o CoT), where GPT-4o generates three search queries per case and MedCPT [17] retrieves the top-K (K = 5) guideline chunks per query for CoT reasoning; and (3) graph-guided inference (GDGraph+Jev and GDGraph+GPT-4o CoT), which traverse the same constructed GDGraph for each task.

Evaluation Metrics. For graph quality (Table 2), we manually verify node precision (proportion of atomic conditions with source-supported questions and definitions) and rule precision (proportion of extracted transition rules supported by the source guideline). For decision performance and efficiency, we report case accuracy (Acc.), weighted F1 (WF1), and Matthews correlation coefficient (MCC) [18], alongside mean online API time and cost per case.

## 3.2. Graph Fidelity and Comparison with Retrieval

In the source-based review (Table 2), the constructed graphs achieve 99.8% overall node precision and 96.4% rule precision, showing high source fidelity. On downstream tasks (Table 3), unguided Direct GPT-4o and GPT-4o CoT fall below 60% accuracy on CUA and K-TIRADS. Unstructured retrieval (RAG+GPT-4o CoT) raises overall GPT-4o CoT accuracy from 51.9% to 80.1%, while graph guidance (GDGraph+GPT-4o CoT) reaches 82.7%. While RAG performs better on O-RADS (84.6% vs. 78.6%), graph guidance excels on K-TIRADS (79.6% vs. 67.0%), where stratification jointly depends on nodule composition, echogenicity, and suspicious features, showing that explicit paths with linked definitions help resolve multi-feature criteria.

Table 2. Precision of constructed decision graphs against source guidelines, reported as supported/total (%).
<table><tr><td>Guideline</td><td>Node Prec.</td><td>Rule Prec.</td></tr><tr><td>CUA</td><td>137/137 (100.0%)</td><td>51/55 (92.7%)</td></tr><tr><td>O-RADS</td><td>136/137 (99.3%)</td><td>92/98 (93.9%)</td></tr><tr><td>K-TIRADS</td><td>216/216 (100.0%)</td><td>123/123 (100.0%)</td></tr><tr><td>Overall</td><td>489/490 (99.8%)</td><td>266/276 (96.4%)</td></tr></table>

Table 3. Comparison of decision performance across 347 radiology cases. Acc.: case accuracy; WF1: weighted F1-score; MCC: Matthews correlation coefficient. Best results within each task are shown in bold.
<table><tr><td>Method</td><td>Acc. ↑(%)</td><td>WF1↑(%)</td><td>MCC↑</td></tr><tr><td>CUA (n = 127)</td><td></td><td></td><td></td></tr><tr><td>Direct Jev</td><td>61.4</td><td>64.1</td><td>0.555</td></tr><tr><td>GDGraph+Jev</td><td>93.7</td><td>94.4</td><td>0.922</td></tr><tr><td>Direct GPT-40</td><td>46.5</td><td>54.4</td><td>0.412</td></tr><tr><td>GPT-4o CoT</td><td>52.8</td><td>59.3</td><td>0.447</td></tr><tr><td>RAG+GPT-40 CoT</td><td>86.6</td><td>88.0</td><td>0.836</td></tr><tr><td>GDGraph+GPT-4o CoT</td><td>89.0</td><td>87.5</td><td>0.862</td></tr><tr><td colspan="4">O-RADS (n = 117)</td></tr><tr><td>Direct Jev</td><td>72.6</td><td>70.2</td><td>0.587</td></tr><tr><td>GDGraph+Jev</td><td>92.3</td><td>93.1</td><td>0.884</td></tr><tr><td>Direct GPT-4o</td><td>69.2</td><td>70.4</td><td>0.552</td></tr><tr><td>GPT-4o CoT</td><td>57.3</td><td>59.7</td><td>0.458</td></tr><tr><td>RAG+GPT-4o CoT</td><td>84.6</td><td>85.4</td><td>0.786</td></tr><tr><td>GDGraph+GPT-4o CoT</td><td>78.6</td><td>82.9</td><td>0.713</td></tr><tr><td colspan="4">K-TIRADS (n = 103)</td></tr><tr><td>Direct Jev</td><td>35.0</td><td>34.6</td><td>0.235</td></tr><tr><td>GDGraph+Jev</td><td>85.4</td><td>85.6</td><td>0.794</td></tr><tr><td>Direct GPT-4o</td><td>47.6</td><td>49.3</td><td>0.342</td></tr><tr><td>GPT-4o CoT</td><td>44.7</td><td>47.8</td><td>0.238</td></tr><tr><td>RAG+GPT-4o CoT</td><td>67.0</td><td>68.4</td><td>0.515</td></tr><tr><td> $\mathrm { G D G r a p h + G P T - 4 o C o T }$ </td><td>79.6</td><td>80.1</td><td>0.714</td></tr></table>

## 3.3. System One Performance and Inference Efficiency

We next evaluate whether pairing GDGraph with the System One model Jev improves decision accuracy (Table 3) and compare its online inference efficiency against the GPT-4o baselines (Fig. 2). Given only the case content, the candidate options, and their definitions (Direct Jev), Jev reaches 57.3% overall accuracy (35.0% on K-TIRADS). When the same cases are evaluated using GDGraph (GDGraph+Jev), performance improves across all three tasks, reaching 90.8% overall accuracy and achieving the highest Acc., WF1, and MCC in each guideline. Additionally, GDGraph+Jev achieves higher accuracy than both GDGraph+GPT-4o CoT (82.7%) and RAG+GPT-4o CoT (80.1%), while requiring 0.79–1.62 s per case in online API time (vs. 6.16–7.15 s for RAG+GPT-4o CoT and 17.10–34.97 s for GDGraph+GPT-4o CoT) and $0 . 5 6 { - } 1 . 0 7 \times 1 0 ^ { - 3 } \mathrm { U S D }$ per case in API cost (Fig. 2). These results indicate that organizing guidelines into localized node judgments and explicit rules enables the System One model to resolve multi-condition criteria with low online inference latency and cost.

(a) Time per case ↓  
![](images/25b328440eafbcc16b4c3e5af33c7a9ab49ae2779d4141988acebf3629403c82.jpg)  
(b) Cost per case ↓

![](images/55efe5de47a58931c32d0e7170140e45c83e40c77ba0ec77572d8d7833a09831.jpg)  
Fig. 2. Comparison of online inference efficiency across the three guideline tasks, measured by (a) mean cumulative API time (s/case) and (b) mean API cost (10<sup>−3</sup> USD/case).

## 3.4. Error Analysis

Among the remaining errors, we identify two recurring patterns that highlight the main limitations of GDGraph-guided inference. In 11 K-TIRADS cases, “ill-defined margins” or “irregular shape” were interpreted as “irregular margins,” activating a higher-risk branch because node definitions did not explicitly contrast these terms. In three O-RADS cases, errors arose when mapping qualitative vascularity descriptions (e.g., moderate flow) to node-level criteria. These observations highlight the importance of normalizing radiological descriptions and aligning imaging concepts with model representations across varied reports.

## 4. CONCLUSION

We proposed GDGraph, a framework that converts radiology guidelines into executable decision graphs for risk stratification and management planning. During compilation, an LLM constructs the graph through section-based chunking, local subgraph extraction, and cross-branch aggregation. During inference, the System One model answers localized questions along the graph, yielding a traceable decision path. Experiments on three guidelines show that GDGraph+Jev achieves higher accuracy than CoT-based GPT-4o baselines at lower latency and cost, indicating its practical value for auditable clinical decision support.

## 5. COMPLIANCE WITH ETHICAL STANDARDS

This research study was conducted retrospectively using human subject data made available in open access by the PubMed Central Open Access Subset. Ethical approval was not required, as confirmed by the license attached to the open access data.

## 6. ACKNOWLEDGMENTS

This work is supported by PolyU Start-up Fund (P0060371).

## 7. REFERENCES

[1] Julie Y. An, Kyle M. L. Unsdorfer, and Jeffrey C. Weinreb, “BI-RADS, C-RADS, CAD-RADS, LI-RADS, Lung-RADS, NI-RADS, O-RADS, PI-RADS, TI-RADS: Reporting and data systems,” RadioGraphics, vol. 39, no. 5, pp. 1435–1436, 2019.

[2] Patrick O. Richard, Philippe D. Violette, Bimal Bhindi, Rodney H. Breau, Matthieu Gratton, Michael A. S. Jewett, et al., “2023 UPDATE – Canadian Urological Association guideline: Management of cystic renal lesions,” Can. Urol. Assoc. J., vol. 17, no. 6, pp. 162–174, 2023.

[3] Lori M. Strachowski, Priyanka Jha, Catherine H. Phillips, Misty M. Blanchette Porter, Wouter Froyman, Phyllis Glanc, et al., “O-RADS US v2022: An update from the American College of Radiology’s Ovarian-Adnexal Reporting and Data System US Committee,” Radiology, vol. 308, no. 3, pp. e230685, 2023.

[4] Eun Ju Ha, Sae Rom Chung, Dong Gyu Na, Hye Shin Ahn, Jin Chung, Ji Ye Lee, et al., “2021 Korean Thyroid Imaging Reporting and Data System and imagingbased management of thyroid nodules: Korean Society of Thyroid Radiology consensus statement and recommendations,” Korean J. Radiol., vol. 22, no. 12, pp. 2094–2123, 2021.

[5] Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan, and Daniel Shu Wei Ting, “Large language models in medicine,” Nat. Med., vol. 29, no. 8, pp. 1930–1940, 2023.

[6] Danqing Hu, Bing Liu, Xiaofeng Zhu, Xudong Lu, and Nan Wu, “Zero-shot information extraction from radiological reports using ChatGPT,” Int. J. Med. Inform., vol. 183, pp. 105321, 2024.

[7] Paul Hager, Friederike Jungmann, Robbie Holland, Kunal Bhagat, Inga Hubrecht, Manuel Knauer, et al., “Evaluation and mitigation of the limitations of large language models in clinical decision-making,” Nat. Med., vol. 30, no. 9, pp. 2613–2622, 2024.

[8] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou, “Chain-of-thought prompting elicits reasoning in large language models,” in Proc. NeurIPS, 2022, vol. 35, pp. 24824–24837.

[9] Diogo Almeida, “Introducing System One models & Jev,” TypeSafe AI Blog, Sept. 2026, Accessed: 2026- 10-05.

[10] Cynthia Rudin, “Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead,” Nat. Mach. Intell., vol. 1, no. 5, pp. 206–215, 2019.

[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, et al., “Retrieval-augmented generation for knowledgeintensive NLP tasks,” in Proc. NeurIPS, 2020, vol. 33, pp. 9459–9474.

[12] Alexander Rau, Stephan Rau, Daniela Zoller, Anna¨ Fink, Hien Tran, Caroline Wilpert, et al., “A contextbased chatbot surpasses trained radiologists and generic ChatGPT in following the ACR appropriateness guidelines,” Radiology, vol. 308, no. 1, pp. e230970, 2023.

[13] Ruiqi Deng, Geoffrey Martin, Tony Wang, Gongbo Zhang, Yi Liu, Chunhua Weng, Yanshan Wang, Justin F. Rousseau, and Yifan Peng, “CPGPrompt: translating clinical guidelines into large language model-executable decision support,” J. Am. Med. Inform. Assoc., vol. 33, no. 4, pp. 855–862, 2026.

[14] Karen Sparck Jones, “A statistical interpretation of term ¨ specificity and its application in retrieval,” J. Doc., vol. 28, no. 1, pp. 11–21, 1972.

[15] Gerard Salton and Christopher Buckley, “Termweighting approaches in automatic text retrieval,” Inf. Process. Manag., vol. 24, no. 5, pp. 513–523, 1988.

[16] Zhengyun Zhao, Qiao Jin, Fangyuan Chen, Tuorui Peng, and Sheng Yu, “A large-scale dataset of patient summaries for retrieval-based clinical decision support systems,” Sci. Data, vol. 10, no. 1, pp. 909, 2023.

[17] Qiao Jin, Won Kim, Qingyu Chen, Donald C. Comeau, Lana Yeganova, W. John Wilbur, and Zhiyong Lu, “MedCPT: Contrastive Pre-trained Transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval,” Bioinformatics, vol. 39, no. 11, pp. btad651, 2023.

[18] Juri Opitz, “A closer look at classification evaluation metrics and a critical reflection of common evaluation practice,” Trans. Assoc. Comput. Linguist., vol. 12, pp. 820–836, 2024.