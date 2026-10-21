# VGL-stGen: Learning a shared multimodal model for lung histology and spatial transcriptomics

Haenara Shin<sup>∗</sup>, Dongjoo Lee, Sungwoo Bae, Kwon Joong Na, Daeseung Lee, Hongyoon Choi<sup>†</sup>

Portrai, Inc., Seoul, Republic of Korea

## Abstract

Learning across histology and spatial transcriptomics requires a model that can connect tissue morphology, gene expression, and cellular identity within a shared framework. We developed VGL-stGen (Vision-Gene Language model–spatial transcriptomics generative), a multimodal large language model adapted from a pretrained medical vision-language model using large-scale paired lung H&E and spatial transcriptomics data together with single-cell references. Training integrates image–expression and expression–cell-type pairs from multiple spatial transcriptomics and single-cell RNA-seq data, enabling complementary supervision across modalities. Geneconditioned pseudo-labeling further transfers transcriptomic cell-type supervision to histology images. A single deployed VGL-stGen checkpoint supports thirteen evaluated task formulations spanning H&E-based tissue and cell-type classification, gene-expression–based cell typing, gene prediction, and conditional expression completion from combinations of histology, cell identity, and observed gene expression. Task-specific decoding allows the same generative model to produce categorical, gene-list, and structured expression outputs. For expression completion, fixing queried gene identities and autoregressively predicting their expression levels substantially improves structured prediction, while controlled adaptation shows that restricting supervision to the requested expression-level tokens can further improve image-conditioned completion. Imagepermutation experiments indicate that correctly paired histology provides additional information beyond observed gene-expression levels.

External classification relies partly on computational references with shared annotation lineage, and some H&E sources were exposed during cancer training or model development. Within these constraints, VGL-stGen demonstrates that a medical vision-language backbone can be adapted with large-scale spatial transcriptomics and histology data into a unified multimodal model capable of performing diverse H&E- and gene-expression–related tasks. This framework provides a foundation for connecting tissue morphology and spatial molecular states through shared generative modeling.

## 1 Introduction

Recent advances in spatial biology have enabled computational models to learn representations from histology images and spatially resolved gene expression. However, most existing models are designed around specific modalities or prediction tasks, leaving histological morphology, gene expression, and cellular identity largely separated across modeling frameworks. A unified model that can learn the underlying biology shared across these modalities and flexibly reason over diferent combinations of images, genes, and cell states remains underexplored. To address this gap, we developed VGL-stGen, a multimodal large language model trained on large-scale lung and lung cancer spatial transcriptomics data. We assembled a lung training corpus combining paired histology and spatial transcriptomics from a total of 69 samples (47 Xenium samples and 22 Visium samples) in PortraiATLAS with single-cell references and cancer annotations. By integrating these relationships within a shared vision-gene-language framework, VGL-stGen is designed to learn across histological and molecular representations and perform diverse tasks from diferent combinations of H&E images, gene expression, and cellular information.

Existing image–omics methods use diferent observations and target representations. ST-Net predicts expression from histology [1], BLEEP retrieves reference expression through contrastive embeddings [2], and GenAR generates expression counts autoregressively [3]. SpaCKLE completes expression using neighbouring spots [4]. OmiCLIP aligns histology with ranked gene text for multiple tasks [5], and SPADE learns aligned histology-expression representations [6]. STPath integrates histology, observed expression and spatial context for generative prediction and spot imputation [7]. GHIST predicts single-cell expression from histology using auxiliary cell-type and neighbourhoodcomposition tasks [8]. These methods establish the importance of specifying which molecular observations are available at inference and which targets are predicted.

VGL-stGen adapts the pretrained medical vision-language model MedGemma-1.5-4b-it [9] for shared conditional generation across the supported lung tasks. Gene-conditioned pseudo-labels supply image cell-type supervision. The model generates gene lists and uses constrained decoding for categorical labels and queried expression levels. For conditional completion, the input includes 75 observed levels and either histology or a supplied cell type; the output contains levels for 25 named genes. Fixing the query names separates supplied identifiers from predicted values. The distinction motivates a controlled comparison of full-answer supervision with loss masking over the requested level tokens.

We assess image cell typing before and after pseudo-label training, evaluate fixed-name decoding from the deployed weights, and compare expression supervision under matched initial weights, training examples and update budgets. Image permutation tests whether correct pairing contributes when observed gene levels remain available. Task-specific classifiers and full-input MLPs provide reference performance with the corresponding observed inputs. The deployed checkpoint’s experiments establish task-dependent completion efects.

## 2 Methods

## 2.1 Data and shared-model learning

Task instructions condition a shared autoregressive decoder on the supported input combinations (Fig. 1). Outputs are categorical labels, gene lists or discrete expression levels. The two expressioncompletion tasks predict 25 named gene levels from 75 observed levels together with either histology or a supplied cell type.

Training integrates paired histology and spatial transcriptomics from 69 ST–H&E samples within a subset of the PortraiATLAS database, together with single-cell references and 30,965 unique cancer patches. Spatial-transcriptomic and pathologist-derived H&E patches were extracted at a nominal 112-µm field of view and stored at 224 × 224 pixels. LubLung training crops contained approximately 87 µm of tissue and were resized and padded to the same pixel dimensions. Maskedexpression training data cover 2,000 genes for the image task and 1,478 for the cell-type task. Targets are ‘Absent’ for undetected expression and four quartile levels: ‘Low’, ‘Med’, ‘High’ and ‘Very High’, defined from detected expression within a cell or Visium spot.

![](images/d7e703151e4f6b975a3adf3e12d108477f970c73ca485a96d250c23b5ff1fb18.jpg)  
Figure 1: VGL-stGen overview. (a) Complementary lung supervision adapts one medical vision–language model for 13 evaluated tasks. The 6 label, 5 gene-list and 2 expression-level tasks share a decoder and checkpoint. (b) Gene-conditioned cell-type pseudo-labels supervise paired images during training. Conditional expression completion uses 75 observed levels and 25 named queries. Detailed task inputs and training stages are specified in the Methods and supplementary material.

The medgemma-1.5-4b-it backbone has a frozen vision tower and QLoRA-adapted language projections (rank 16, alpha 32, 4-bit NormalFloat (NF4) weights, bfloat16 computation) [10, 11]. Training applies autoregressive supervision to the target answer tokens of task-specific image–text or text-only examples. Transcriptomic adaptation includes gene-list, cell-type and expression-bin objectives. The teacher predicts cell type from measured genes; its highest-likelihood label supervises the paired image. A top-two log-likelihood gap of at least 1.0 retains 1.32M of 1.75M pseudo-labelled items. The student trains on 300K image-task and 200K replay rows for one epoch at learning rate 1e-4.

Final training adds binary tumour and nine-class tissue objectives with nine transcriptomic replay objectives, excluding image-to-cell-type supervision. The joint image-and-genes cell-type task has no direct training objective. We average six late rank-16 adapters [12], representing their mean at rank 96 through concatenation. The averaging window was selected using the 39-slide TCGA development subset. For the main benchmarks, binary tumour scoring uses merged bfloat16 weights; nine-class tissue, cell-type, gene-list and expression scoring use the NF4-base adapter.

Table 1 specifies the evaluated inputs, outputs and conditions.

## 2.2 Structured prediction

Closed labels use prefix-tree decoding; twenty-gene inputs preserve expression rank. For completion, a parser extracts query names from the 100-gene input. With names fixed, the model predicts each level from five permitted first tokens, conditioned autoregressively on its own preceding answers. Pair-F1 equals accuracy over the queried gene–level pairs.

## 2.3 Evaluation cohorts and reference annotations

Table 2 summarises evaluation units and annotation provenance.

Table 1: 13 evaluated tasks supported by the deployed checkpoint. BA: balanced accuracy; F1m: macro-F1; F1g: gene-set F1; F1p: pair-F1; Acc: accuracy; AR: autoregressive. H&E overlap denotes partial exposure of evaluation tissue during cancer training. Gene-list completion supplies 15 names and predicts five; expression completion supplies 75 levels and queries 25 named genes. Task 10 supplies true answer length; task 13 supplies ten gene–level pairs. The complete score inventory is supplementary.
<table><tr><td>ID</td><td>Input → output</td><td>Metric</td><td>Evaluation condition</td></tr><tr><td>1</td><td>Genes → cell type</td><td>BA/F1m</td><td>External</td></tr><tr><td>2</td><td>Image + genes → cell type</td><td>BA</td><td>H&amp;E overlap</td></tr><tr><td>3</td><td>Image → cell type</td><td>BA</td><td>H&amp;E overlap</td></tr><tr><td>4</td><td>Image → tumour</td><td>BA</td><td>452 slides</td></tr><tr><td>5</td><td>Image → tissue</td><td>F1m</td><td>39-slide pool</td></tr><tr><td>6</td><td>Image → genes</td><td>F1g</td><td>H&amp;E overlap</td></tr><tr><td>7</td><td>Cell type → genes</td><td>F1g</td><td>External</td></tr><tr><td>8</td><td>Image + partial → genes</td><td>F1g</td><td>H&amp;E overlap</td></tr><tr><td>9</td><td>Cell type + partial → genes</td><td>F1g</td><td>External</td></tr><tr><td>10</td><td>Image → very-high genes</td><td>F1g</td><td>Internal; true length</td></tr><tr><td>11</td><td>Image + levels → levels</td><td>F1p</td><td>Internal AR</td></tr><tr><td>12</td><td>Cell type + levels → levels</td><td>F1p</td><td>Internal AR</td></tr><tr><td>13</td><td>Levels → cell type</td><td>Acc</td><td>Internal</td></tr></table>

Table 2: Evaluation units and reference provenance, with spatial datasets grouped by platform. Visium units are spots; Xenium units are cells. Counts refer to processed evaluation items rather than training samples. Source-level provenance and the diferent exposure of the two Visium sections are retained in the supplementary methods. TCGA-LUAD denotes lung adenocarcinoma. The tissue-classification analysis uses the 39-slide development subset.
<table><tr><td>Cohort</td><td>Evaluated units</td><td>Reference / exposure</td></tr><tr><td>Visium</td><td>9,964 spots / 2 sections</td><td>Transferred cell types; one section supplied H&amp;E for cancer training</td></tr><tr><td>Xenium</td><td>214,246 cells / 2 sections</td><td>Transferred cell types; both sections supplied H&amp;E for cancer training</td></tr><tr><td>TCGA-LUAD binary</td><td>357,692 patches / 452</td><td>ARA-CNN labels; includes development subset</td></tr><tr><td>TCGA tissue classification</td><td>slides / 398 patients 30,719 patches / 39</td><td>Development subset; pooled transductive correction</td></tr><tr><td>Masked expression benchmark</td><td>slides 1,000 rows/task</td><td>Internal; 7 image samples / 1 cell-type sample identifier</td></tr></table>

The external spatial and TCGA classification benchmarks use computationally derived reference annotations (often termed silver labels). ARA-CNN cancer segmentations [13] share LubLung lineage with training; TACCO cell-type transfers [14] use the training lung cancer atlas (LuCA) [15]. Scores measure agreement with these annotations, which can preserve shared errors. The four public 10x sections [16], excluded from transcriptomic training, pair H&E with measured expression: 214,246 Xenium cells and 9,964 Visium spots support centre-cell and dominant regional typing, respectively. Gene-list and expression targets derive from measured expression. External cell-type-conditioned gene queries use transferred labels; internal cell-type-conditioned expression queries use LuCA atlas labels. TCGA-LUAD supplies cancer images [17].

Three of the four spatial sections supplied 4,846 binary cancer training rows. The Visium squamous cell carcinoma (SCC) section has no source match in that cancer-training manifest. Imageconditioned external tasks therefore include previously seen tissue with unseen task labels. A 39- slide TCGA subset informed development and is included in the 452-slide evaluation. Sample and recorded cell/spot identifiers are disjoint within the expression-task training and test manifests. However, all seven samples in the masked-image benchmark were represented in gene-list training, including 692 of the 1,000 benchmark cells or spots. This internal benchmark therefore does not establish patient-independent generalisation across the full training sequence.

## 2.4 Matched comparisons and statistical analysis

We compare the deployed VGL-stGen checkpoint with ten of-the-shelf models: MedGemma-1.5-4B [9], Gemma3-4B [18], Gemma4-12B and Gemma4-31B [19], Qwen3-VL-8B and Qwen3-VL-32B [20], Qwen3.8-27B [21], DeepSeek-VL2-small and DeepSeek-VL2 [22], and GLM-4.6V-Flash [23]. All receive the same task inputs and instructions, with the original task-specific generation limits, batch composition, output parser and metric. Native chat templates, tokenizers and image processors are retained. Task 4 uses BF16; all other tasks use NF4 double quantization with BF16 computation in quantized layers. The deployed VGL-stGen adapter and merged weights follow this same task-specific assignment. This comparison evaluates lung-specific adaptation against pretrained multimodal models; task-specific trained classifiers and MLPs are assessed separately.

Categorical tasks use allowed-label decoding, gene-list tasks use generation followed by the original gene parser, and expression completion predicts the levels of fixed queried genes autoregressively. These are structured prediction protocols rather than a uniform test of unrestricted JSON generation. A supplementary format-constrained arm repeats tasks 6–10 and 13 for all eleven models under the same budgets. It restricts gene-array syntax or permitted label spellings without supplying the correct genes or restricting predictions to an assay panel. Paired bootstrap intervals use 1,000 draws: cells or spots for tasks 1–3, slides for task 4, patches for task 5, and complete input rows for tasks 6–13. These unadjusted intervals condition on fitted models and do not establish generalisation across independent patients or sections. Detailed settings, model revisions and all paired intervals accompany the supplementary analysis.

Gene comparators receive the same ranked twenty-gene lists; comparators include logistic regression, Geneformer V2 and C2S variants, with language-model controls [24–28]. Morphology probes use fifteen frozen encoders with class-balanced logistic regression (C=1). Transferred probes learn gene-teacher labels; target-cohort references use five-fold stratified cell-level cross-validation on the 54,058-item subset. Fourteen cancer encoders [29–41] carry binary probes trained on the same 30,965 unique cancer patches as VGL-stGen, which also receives auxiliary supervision.

The conditional-expression image benchmark spans seven sample identifiers: three Xenium (665 cells) and four Visium (335 spots). Task-specific cell-type comparator intervals use section-thenitem paired bootstrap (2,000 draws). Primary cancer intervals resample slides; development-patient exclusion resamples patients (1,000 draws each). New expression contrasts use 2,000 paired row or sample draws within platform, preserving platform proportions. The cell-type-conditioned benchmark has one sample identifier, permitting only row intervals. All 95% confidence intervals (CIs) condition on fitted models, without multiplicity adjustment; sample identifiers are not verified patients. For VGL-stGen, standard deviation (SD) is computed across three additional-training runs from the same initial checkpoint.

## 2.5 Application analyses

We evaluated genes → cell type (Task 1) with the checkpoint (merged bfloat16 weights, constrained decoding) on two application sources. The first was GSE127465, a single-cell RNA-seq dataset of seven treatment-naive NSCLC patients with published annotations [42]. We used 2,000 highly variable genes. Each of 7,993 randomly sampled tumor-tissue cells was then represented by its 20 highest-expressed genes in rank order. To match the authors’ 11 major types, we merged macrophage/monocyte and cDC predictions into their MoMacDC class. Patient-specific malignant clusters were treated as epithelial, and we also report results without them.

The second source was the public 10x Xenium Human Lung Cancer FFPE section (377-gene panel) [16]. We scored 12,000 random cells with at least 10 transcripts, using the same top-20 input. Because this section has no published annotation, we crosstabulated each predicted label against measured expression of canonical marker sets. These application conditions difer from the main 224,210-item cell-type benchmark; the supplementary methods specify reference construction.

For the image → genes task, we analysed 459 TCGA-LUAD patients with a readable diagnostic FFPE slide and open primary-tumour RNA-seq [17]. This cohort has no patients in common with the TCGA development subset. The model returned 20 gene symbols for each of 112 random 128 µm tissue patches per slide. A slide’s value for a gene is the fraction of its patches naming that gene. We compared this value with log2(TPM+1) for 317 genes, restricted to genes named in at least 10% of slides and expressed (TPM > 1) in at least half of cases. Both matrices were gene-centred to focus comparisons on patient-to-patient variation.

The endpoints were per-gene Pearson correlation across patients (Benjamini–Hochberg FDR, compared with a patient-permutation null) and correlation of marker-programme scores. Patient matching was summarized by the median percentile at which each slide ranks its own bulk profile among all 459 patients. Controls were a shufled slide–bulk pairing, restriction to each tissue source site, and incremental cross-validated R² after sequencing depth, magnification, source site and tumor fraction. TCGA slides and RNA come from diferent tissue blocks.

## 2.6 Analyses of learning and input contributions

We assess cell typing on images before and after gene-mediated supervision. Separate expression analyses compare full-answer and level-token supervision from the same deployed checkpoint, using 4,096 additional rows, 256 updates and three matched seeds. Image permutation retains observed expression and queried genes. Full-input multilayer perceptrons (MLPs) receive the same molecular context, with image features or the supplied cell type; their encoders and training histories difer. These analyses test specific learning and input choices. Their checkpoints and evaluation protocols are detailed in the supplementary material.

## 3 Results

## 3.1 Shared-model task coverage and multimodal comparisons

The deployed checkpoint supports the thirteen tasks defined in Table 1. Table 3 and Table 4 compare it with ten of-the-shelf medical and general-purpose multimodal models under matched task-specific protocols.

Table 3: Medical and general-purpose multimodal comparisons on tasks 1–5. BA is balanced accuracy and F1m is macro-F1. T1–3 use 224,210 items each; T4 uses 357,692 patches from 452 slides; T5 uses 30,719 patches from 39 development slides. T4 uses BF16 and reports slide-bootstrap means; the other tasks use NF4. T5 shows raw → pooled-corrected F1m for every model. Native image processors are retained. Task key: T1, genes → cell type; T2, image + genes → cell type; T3, image → cell type; T4, image → binary tumor status; T5, image → 9-class tissue type.
<table><tr><td>Model</td><td>T1 BA F1m</td><td>T2 BA</td><td>T3 BA</td><td>T4 BA</td><td>T5 F1m</td></tr><tr><td>VGL-stGen</td><td>0.6293 0.6023</td><td>0.5394</td><td>0.2012</td><td>0.7524</td><td>0.1200 → 0.2801</td></tr><tr><td>MedGemma-1.5-4B</td><td>0.1836 0.1576</td><td>0.2070</td><td>0.1105</td><td>0.5954</td><td>0.0911 → 0.1711</td></tr><tr><td>Gemma3-4B</td><td>0.2254 0.1738</td><td>0.1853</td><td>0.1250</td><td>0.6637</td><td>0.1334 → 0.1347</td></tr><tr><td>Qwen3-VL-8B</td><td>0.4662 0.3148</td><td>0.4572</td><td>0.0995</td><td>0.5182</td><td>0.1102 → 0.1630</td></tr><tr><td>DeepSeek-VL2-small</td><td>0.1314 0.1140</td><td>0.1380</td><td>0.1100</td><td>0.5351</td><td>0.0649 → 0.1321</td></tr><tr><td>GLM-4.6V-Flash</td><td>0.3444 0.2775</td><td>0.3160</td><td>0.1008</td><td>0.5808</td><td>0.1411 → 0.1733</td></tr><tr><td>Gemma4-12B</td><td>0.4023 /0.3684</td><td>0.4071</td><td>0.1440</td><td>0.6986</td><td>0.1651 → 0.1763</td></tr><tr><td>Qwen3.8-27B</td><td>0.3414 /0.2703</td><td>0.3302</td><td>0.1265</td><td>0.6974</td><td>0.1947 → 0.2613</td></tr><tr><td>Gemma4-31B</td><td>0.5035 /0.4870</td><td>0.4634</td><td>0.1652</td><td>0.6719</td><td>0.1887 → 0.1958</td></tr><tr><td>DeepSeek-VL2</td><td>0.1510 0.1035</td><td>0.1595</td><td>0.1103</td><td>0.5983</td><td>0.0554 → 0.1563</td></tr><tr><td>Qwen3-VL-32B</td><td>0.4822 0.3688</td><td>0.4739</td><td>0.1110</td><td>0.6742</td><td>0.1625 → 0.1953</td></tr></table>

Table 4: Multimodal comparisons on gene and expression tasks, with 1,000 rows per task and NF4 weights. T6–10: gene-set F1; T11–12: pair-F1; T13: accuracy. T10 supplies the true answer length; T11–12 supply 75 observed levels and query 25 named genes. T6–10 and T13 use the primary generation protocol, separately from the format-constrained control. Values preserve the original score aggregation and rounding. Task key: T6, image → gene names; T7, cell type → gene names; T8, image + partial gene list → missing gene names; T9, cell type + partial gene list → missing gene names; T10, image → very-high-expression gene names; T11, image + observed levels → queried levels; T12, cell type + observed levels → queried levels; T13, gene-level pairs → cell type.
<table><tr><td>Model</td><td>T6</td><td>T7</td><td>T8</td><td>T9</td><td>T10</td><td>T11</td><td>T12</td><td>T13</td></tr><tr><td>VGL-stGen</td><td>0.1173</td><td>0.0965</td><td>0.0604</td><td>0.0484</td><td>0.2828</td><td>0.6596</td><td>0.6858</td><td>0.7420</td></tr><tr><td>MedGemma-1.5-4B</td><td>0.0000</td><td>0.0005</td><td>0.0000</td><td>0.0000</td><td>0.0003</td><td>0.5733</td><td>0.5540</td><td>0.0000</td></tr><tr><td>Gemma3-4B</td><td>0.0088</td><td>0.0345</td><td>0.0108</td><td>0.0000</td><td>0.0058</td><td>0.5072</td><td>0.4996</td><td>0.3500</td></tr><tr><td>Qwen3-VL-8B</td><td>0.0000</td><td>0.0864</td><td>0.0210</td><td>0.0380</td><td>0.0032</td><td>0.1634</td><td>0.2372</td><td>0.3500</td></tr><tr><td>DeepSeek-VL2-small</td><td>0.0000</td><td>0.0080</td><td>0.0026</td><td>0.0000</td><td>0.0042</td><td>0.1600</td><td>0.2165</td><td>0.0630</td></tr><tr><td>GLM-4.6V-Flash</td><td>0.0044</td><td>0.0397</td><td>0.0140</td><td>0.0237</td><td>0.0030</td><td>0.3662</td><td>0.3627</td><td>0.4260</td></tr><tr><td>Gemma4-12B</td><td>0.0248</td><td>0.0791</td><td>0.0140</td><td>0.0463</td><td>0.0071</td><td>0.2692</td><td>0.2077</td><td>0.3960</td></tr><tr><td>Qwen3.8-27B</td><td>0.0269</td><td>0.0699</td><td>0.0209</td><td>0.0178</td><td>0.0131</td><td>0.6130</td><td>0.5621</td><td>0.4560</td></tr><tr><td>Gemma4-31B</td><td>0.0556</td><td>0.1003</td><td>0.0382</td><td>0.0502</td><td>0.0229</td><td>0.4931</td><td>0.5108</td><td>0.4620</td></tr><tr><td>DeepSeek-VL2</td><td>0.0005</td><td>0.0282</td><td>0.0000</td><td>0.0105</td><td>0.0000</td><td>0.1043</td><td>0.1127</td><td>0.2950</td></tr><tr><td>Qwen3-VL-32B</td><td>0.0197</td><td>0.0839</td><td>0.0022</td><td>0.0452</td><td>0.0047</td><td>0.4315</td><td>0.3790</td><td>0.4020</td></tr></table>

For gene-based cell typing, VGL-stGen reaches balanced accuracy 0.6293 and macro-F1 0.6023, compared with 0.5035 and 0.4870 for Gemma4-31B. Adding an image to the gene input yields balanced accuracy 0.5394, versus a comparator maximum of 0.4739 for Qwen3-VL-32B. Image-only cell typing remains harder: VGL-stGen scores 0.2012 versus 0.1652 for Gemma4-31B. The positive diferences quantify performance against the computational reference annotations under the stated exposure conditions.

Binary tumour balanced accuracy is 0.7524 versus 0.6986 for Gemma4-12B. The nine-class tissue result depends on readout: VGL-stGen’s raw macro-F1 is 0.1200, below Qwen3.8-27B’s 0.1947, whereas pooled probability correction yields 0.2801 versus 0.2613. The correction uses the evaluation pool and was selected during development; both scores are therefore reported.

Expression-bin prediction is a core component of the deployed model. Image-conditioned expression completion reaches pair-F1 0.6596, versus 0.6130 for Qwen3.8-27B; cell-type-conditioned completion reaches 0.6858 versus 0.5621. These results use 75 observed levels and 25 queried genes. On the two cell-type-conditioned gene-list tasks, Gemma4-31B has slightly higher point estimates (0.1003 versus 0.0965 and 0.0502 versus 0.0484), with paired intervals spanning zero. Very-high-gene prediction scores 0.2828 under the supplied true-length condition, while levels-to-cell-type accuracy is 0.7420. The latter exceeds the evaluation-set majority-label baseline of 0.3620.

The format-constrained evaluation retains VGL-stGen’s reported scores and improves several comparator outputs. The strongest comparator scores are 0.0611 versus VGL-stGen’s 0.1173 for image-to-genes prediction, 0.0244 versus 0.2828 for very-high-gene prediction, and 0.4720 versus 0.7420 for levels-to-cell-type classification. These controls support predictive gains beyond output syntax alone, while retaining the fixed-budget qualification for models that generate intermediate reasoning.

## 3.2 Task-specific classification benchmarks

For genes-to-cell-type, VGL-stGen achieves balanced accuracy 0.6293 and macro-F1 0.6023. Rankweighted logistic regression gives 0.6616 and 0.5560, respectively. VGL-stGen’s macro-F1 margin is +0.0463 with section-paired CI [+0.0276, +0.0597], while logistic regression has higher balanced accuracy. Fig. 2 shows the two fine-tuned Geneformer variants separately.

On 357,692 patches from 452 TCGA slides, VGL-stGen achieves balanced accuracy 0.7532 [0.7452, 0.7610] (bootstrap mean) and AUROC 0.8266 against ARA-CNN-derived references. Fourteen encoder–probe comparators span 0.6938–0.7626; VGL-stGen ranks sixth of fifteen systems. Its frozen-tower probe scores 0.7522. Excluding all 40 slides belonging to the 39 development patients leaves 412 slides, 359 patients and 326,118 patches: observed balanced accuracy is 0.7523 versus 0.7534 on the full set, patient CI [0.7407, 0.7628], with rank unchanged.

## 3.3 Contributions of cross-modal supervision and expression inputs

Gene-mediated supervision increases image-to-cell-type balanced accuracy from 0.161 to 0.226; the deployed checkpoint scores 0.2012 after joint cancer training and averaging. This supports transferring cell-type supervision through measured genes while retaining the final checkpoint’s performance as the primary endpoint.

In the separate three-seed expression study, level-token supervision increases image-conditioned pair-F1 from 0.6720 to 0.6805. Cell-type-conditioned pair-F1 is 0.7201, compared with 0.7221 for full-answer supervision. Correct image pairing contributes when observed expression is retained: permutation reduces mean pair-F1 by 0.00776 under level-token supervision. Full-input MLPs retain higher point estimates on both completion tasks. Supplementary methods, tables and figures report both supervision conditions, paired intervals, secondary metrics and input controls. These additional-training checkpoints are separate from the deployed model evaluated across thirteen tasks.

## (a) Genes → cell type

Twenty-gene input · 224,210 cells/spots  
![](images/38e3291ff71d6e2a0c15bd4aef1a262d333f22272a19c0a08f7bdd114c489b7d.jpg)

## (b) H&E → tumour

14 encoder–probe comparators; colour reference unranked

![](images/ec22c9ad436dc3000159a31062117c9f2d029f5f0b148133c66f8dc6670688f5.jpg)  
Figure 2: Task-specific comparisons. (a) Balanced accuracy (BA; circles) and macro-F1 (squares) for the same 20-gene input; connecting lines pair metrics, not confidence intervals. FT denotes finetuning; the two Geneformer variants are separate systems. (b) 14 encoder–probe comparators, VGLstGen and an unranked colour reference on 452 TCGA slides. Intervals resample slides; the shaded band spans the 14 encoder point estimates. Classification references are computationally derived. TCGA includes the 39-slide development subset; point ordering does not establish significance between neighbouring models.

## 3.4 Gene-based cell typing and spatial application

On GSE127465, predicted cell types recovered the published annotation with balanced accuracy 0.815 and macro-F1 0.819 across 11 classes. Balanced accuracy was 0.814 when malignant cells were excluded, and 0.70–0.86 in individual patients (Fig. 3(a–c)). The input format mattered. Shufling the order of the same 20 genes lowered balanced accuracy to 0.775, and omitting the gene filter lowered it to 0.282.

On the Xenium section, every one of the 11 predicted labels showed its highest expression on its own marker set (Fig. 3(d)). The labels were largely consistent with the unsupervised clusters (weighted purity 0.916 over 24 clusters). Mapped onto the tissue, the predictions reflected its architecture: epithelial calls overlay tumour glands, T and B calls a lymphoid aggregate, and stromal calls the intervening stroma (Fig. 3(e–f)). The Xenium comparison relies on markers rather than independent labels. The epithelial marker set contains only EPCAM, and the pDC group contains one cell.

## 3.5 Molecular correspondence and whole-slide queries

On TCGA-LUAD, the H&E-derived profiles reproduced the patient-to-patient gradient of lymphoid genes measured in bulk RNA-seq (Fig. 4(a)). Of 317 genes, 152 correlated with bulk expression at FDR < 0.05, against a permutation-null 95th percentile of 1. The strongest included T-cell genes (CD3E r = 0.41; CD3D, CD8A, CD2). The lymphoid, smooth-muscle, stromal, endothelial and alveolar programmes were significant $( \mathrm { r } = 0 . 1 7 \ – 0 . 4 1 )$ . The epithelial/tumour programme $\left( \mathrm { r } = 0 . 0 9 \right)$ and the myeloid programme were not $\left( { \mathrm { F i g . 4 } } ( { \mathrm { b - c } } ) \right)$

Each slide ranked its own patient’s bulk profile at a median percentile of 29.5 (chance 50), or 27.9 within its source site. The correct patient was in the top 10% for 22% of slides. Under shufled pairing, the median rank returned to 48.3 and no gene was significant (Fig. 4(d)). After the technical and composition covariates, which together explained $\mathrm { R } ^ { 2 } = 0 . 0 1 5$ , the model’s profile added +0.057 in the cross-validated analysis. The strongest programme associations concern lymphoid and stromal tissue components.

As an application, we tiled one slide (TCGA-49-4510; 526 tiles of $1 2 8 \ \mu \mathrm { m } )$ as an example. At every tile we queried the expression-level task for named genes. The displayed predictions place EPCAM in tumour-rich regions and ACTA2 and PECAM1 in stroma and vessels $\left( \mathrm { F i g . } \quad 4 ( \mathrm { e } ) \right)$ Queries supplied each tile image and gene names without observed expression levels, outside the benchmarked Task 11 condition. Bulk RNA was extracted from a diferent tissue block than the diagnostic slide.

c scRNA-seg crosstab (row-normalised) balanced acc. 0.815, macro-F1 0.819 (chance 0.091)  
d Xenium crosstab vs measured markers 11/11 predicted labels peak on their own markers  
![](images/ad0890da3a56e027c815ddff3cebfbfacf5b5f8735b4b14040921e21847a6176.jpg)

![](images/4fc961d1842d798124c3fc068cc01616db9a45bea7c16866e11506e1e8bdee5a.jpg)

![](images/acd75ffbf3bf0ae842868c3f02660887ba5811e1d4399f18c73406a7aeea258d.jpg)

![](images/c9ace51cce91601bf44b2d787849b44b469e00f46561849bae6adb996851690c.jpg)  
Figure 3: Gene-conditioned cell typing. (a,b) Published and predicted annotations for 7,993 GSE127465 cells. (c) Row-normalised annotation comparison. (d) Marker expression for 12,000 Xenium cells; colours clip at ±3, the epithelial set uses EPCAM alone, and pDC contains one cell. (e,f) Predicted cell positions and an aligned 400 µm H&E field.

![](images/1e4f77b2b4e1ea3818fe3d0c9e2de0864e7a7e00f32abc1e43f756735c3ccb60.jpg)

![](images/13c7774c4182e7d8bf97bc365e4c364ebf9fa7d183a75d249c18573a566805c6.jpg)

![](images/15abea614606420245add7ea46597e35323268a0c785b2644e895123f078a5f0.jpg)

![](images/effad768d5f5c2127572c895d201f183df6dfc47b21cfe1c64bf8ea4f725d33f.jpg)

![](images/0285bc10a35b26051b1a24c0e1045b593fa2c25a1547578df4786f3f6e15575a.jpg)

TCGA-49-4510: 526 tiles of 128 μm, each graded by the expression-level query route  
![](images/620b0c6be56f1fd3ab080b3db7935912614a0667729f354bc36afc8386f5715b.jpg)  
Figure 4: Histology-derived molecular profiles and whole-slide queries. (a) 24 selected genes in 459 patients, ordered by measured lymphoid score and averaged over approximately seven patients per column. (b,c) Programme Pearson correlations and lymphoid association. (d) Matched-patient ranks. (e) EPCAM, CD3E, ACTA2 and PECAM1 predictions in 526 tiles of TCGA-49-4510. Queries use tile images and gene names without observed levels. Bulk RNA was extracted from a diferent tissue block than the diagnostic slide.

## 4 Discussion

VGL-stGen provides a common prediction interface for lung tissue labels, cellular identity, gene lists and queried expression levels. Its shared checkpoint connects image–expression and expression–celltype supervision, with gene-conditioned pseudo-labels supplying image cell-type targets. The practical capability is to select a supported query according to the available image and molecular information. Region-level tissue scoring uses histology, gene-based cell typing uses ranked marker lists, and conditional expression completion also requires observed levels.

The multimodal benchmark supports the value of lung-specific adaptation under shared task protocols. Larger of-the-shelf models often score lower, and residual diferences after format constraints show that output syntax alone does not explain the prediction gaps. Performance remains task-dependent: Gemma4-31B has slightly higher point estimates on the two cell-type-conditioned gene-list tasks, with paired intervals spanning zero, and VGL-stGen’s tissue-classification advantage depends on pooled probability correction. Task-specific comparisons also show trade-ofs: logistic regression has higher gene-classification balanced accuracy, VGL-stGen has higher macro-F1, and tumor-label performance falls within the encoder–probe range. Full-input MLPs retain higher expression pair-F1 at 75 observed levels.

Supporting experiments clarify how supervision and inputs afect the shared model. Geneconditioned pseudo-label training improves image cell typing, and fixing supplied gene names enables level prediction from the deployed weights. Separate completion experiments show a contribution from correctly paired histology when observed expression is retained. Additional level-token supervision improves image completion, while cell-type completion shows a trade-of between pair-F1 and non-’Absent’ accuracy. Those additional adapters have not been evaluated across all thirteen tasks and do not replace the deployed checkpoint in the shared-model benchmark.

The application analyses connect structured predictions to tissue-level exploration. Gene-based cell typing agrees with published annotations under the stated preprocessing and harmonised label space, and the Xenium example links predicted groups to measured marker expression. The TCGA analysis adds patient-level molecular correspondence, strongest for lymphoid and stromal programmes, in a cohort without development-patient overlap. Bulk RNA and diagnostic slides come from diferent tissue blocks.

The evaluation scope follows its data and inference conditions. Classification references are computationally derived and share annotation lineage with training; partial H&E exposure and TCGA development use limit cohort independence. The masked-image expression benchmark includes samples and images represented in gene-list training. Most multimodal-comparison intervals resample items rather than sections or patients. Fixed token budgets can penalise models that generate intermediate reasoning, and native image processors and tokenizers difer. The previously analysed completion benchmark uses synthetic masking of lung marker panels. Independent annotations, verified patient separation and prospective endpoints are needed to extend the present research uses toward clinical application.

## 5 Declarations

Competing interests: Kwon Joong Na, Daeseung Lee, and Hongyoon Choi are co-founders of and hold equity in Portrai, Inc.

Code and weights: https://huggingface.co/Portraier/VGL-stGen-0.5-Lung. Gated access planned for preprint release.

## A Detailed methods

## A.1 Model and adaptation

VGL-stGen adapts the attention and feed-forward projections of the language component of medgemma-1.5-4b-it [9]. The vision tower is frozen. The deployed model’s training stages use QLoRA with rank 16, alpha 32, dropout 0.05, a 4-bit NF4 base with double quantisation, and bfloat16 computation. Eager attention is used; gradient checkpointing is disabled in these stages. The final cancer stage uses AdamW, peak learning rate 2e-4, 500 warmup steps, weight decay 0.01, and cosine warm restarts with a minimum learning rate of 1e-6. Sequence length is 1,536 and gradient accumulation gives an efective batch of 512 from a per-device batch of one. The separate expression-adaptation experiment below starts from the assembled rank-96 adapter and uses diferent optimisation settings.

Training comprises gene and cell-type adaptation with stepwise prompt variants, expression bins and stain augmentation; gene-mediated image-to-cell-type distillation; joint cancer training with transcriptomic replay; and checkpoint averaging. The cancer stage trains eleven objectives: nine replayed transcriptomic tasks and two cancer tasks. The distilled image-to-cell-type objective and the three stepwise prompt variants receive no direct supervision during cancer training. The joint image-and-genes cell-type task is served without a direct training objective.

Cancer training stopped when its in-distribution validation loss plateaued. We use weight averaging [12], representing the mean of six late adapters at rank 96 through concatenated low-rank factors and the corresponding adapter scaling. The averaging window was selected after comparing fifteen saved checkpoints on the 39-slide TCGA development subset. That subset is included in the expanded 452-slide evaluation.

The adapter was trained on a quantised base and also merged into bfloat16 weights. Matchedsubset score diferences reach 0.0049, with the merged path higher for cancer and the adapter path higher for cell type. The main binary tumour benchmark uses merged weights; the main nine-class tissue and cell-type benchmarks use the adapter. Packaging, prompt and scoring path are therefore part of the reported operating condition.

## A.2 Training data and gene panels

The spatial training subset comprises 69 ST–H&E samples in PortraiATLAS, supplemented by single-cell reference profiles from the LuCA atlas [15]. Each sample pairs spatial gene expression with histology; the sample count does not denote independent patients. The combined transcriptomic corpus contains 1,745,240 cells or spots.

A 2,000-gene panel was formed by ranking genes according to the number of source samples in which they were selected as highly variable. Gene identifiers were uppercased and punctuation other than hyphens was removed during preparation of the gene lists. The observed training top-gene lists contain 1,978 distinct identifiers, whereas the corresponding lists in the external benchmark contain 1,051. The two vocabularies share 1,049 identifiers; 929 occur only in the training lists and two only in the external lists. These counts describe observed gene-list vocabularies, not the full gene content of the assay panels. Gene lists are generated without a vocabulary constraint.

Expression tasks use Absent for undetected counts and Low, Med, High and Very High for quartiles of detected expression within a cell or Visium spot. Rows sample 100 genes across bins, with a nominal allocation of 50 Absent, 15 Low, 15 Med, 10 High and 10 Very High. Gene order is shufled before masking. These tasks produce 2,659,430 rows before stain expansion and 6,808,162 afterwards. Observed masked-expression training data cover 2,000 genes for the image task and 1,478 for the cell-type task. These task-specific counts difer from the shared encoded MLP vocabulary described below. The levels-to-cell-type prompt exposes only the first ten sampled gene–level pairs.

The cancer corpus contains 30,965 unique image patches and 92,895 task rows. The public annotated lung tissue source associated with ARA-CNN supplies 18,554 unique patches [13]. Pathologist annotations for this project cover the same 47 Xenium samples used for spatial training and three additional public tissue sections, contributing 12,411 retained non-overlapping patches with at least 80% labelled-region coverage. The binary manifest contains 12,411 task rows from each cancer data source. Both sources use nine tissue categories; the main binary mapping defines cancer as TUMOR only.

## A.3 Augmentation, sampling and stepwise instructions

Spatial-transcriptomic and pathologist-derived patches were extracted using a target resolution of 0.5 µm per pixel and a target size of 224 × 224 pixels, corresponding to a nominal 112-µm field of view. The LubLung source crops cover approximately 87 µm at $1 7 2 \times 1 7 2$ pixels; these crops were resized to $2 2 1 \times 2 2 1$ pixels and padded to 224 × 224 pixels without adding tissue context. TCGA classification patches were extracted at approximately 87 µm, whereas the TCGA application analyses used a $1 2 8 \mathrm { - } \mu \mathrm { m }$ field of view. Two ofline stain-normalized appearances supplement original image-bearing transcriptomic rows. The public cancer source is also stain-expanded; multiplicity difers across sources and tasks. Validation and test manifests are unexpanded. Earlier training additionally uses on-the-fly stain augmentation, including RandStainNA [43] and percentile normalization, with geometric and photometric transformations. The distillation and final cancer runs disable on-the-fly augmentation and retain the ofline variants.

The final sampler groups tasks into image-conditioned gene tasks, expression tasks, text-only gene and cell-type tasks, and cancer. Their initial shares are 0.30, 0.25, 0.20 and 0.25. Subsequent shares are computed from a softmax of the preceding epoch’s family validation losses, clamped to [0.10, 0.50] and renormalised. This gives the small cancer corpus an explicit sampling share. The token loss uses focal weighting [44] with gamma 2.0 and label smoothing 0.1. Inverse-frequency sample weights apply to the nine-class tissue task.

Three auxiliary task variants use step-by-step instructions (Table 5). Their prompts ask for biological interpretation before gene-based cell typing, cell-type characteristics before gene prediction, or morphology and biological processes before image-based gene prediction. Each variant samples one fifth of its source-task rows and retains the original class-label or gene-list targets. The resulting augmentation changes the instruction, without adding intermediate reasoning targets. Across training stages, twelve task types receive target supervision; the three instruction variants bring the objective count to fifteen. The deployed interface returns the requested labels or gene lists directly.

Table 5: Stepwise prompt variants used in earlier training. Answer targets remain the original class labels or gene lists.
<table><tr><td>Variant</td><td>Rows</td><td>Source</td></tr><tr><td>Genes → cell type</td><td>116,278</td><td>Single-cell atlas</td></tr><tr><td>Cell type → genes</td><td>115,207</td><td>Single-cell atlas</td></tr><tr><td>Image → genes</td><td>1,047,500</td><td>Spatial samples; stain-expanded</td></tr></table>

## A.4 Gene-mediated self-distillation

Spatial data supply image–gene pairs and the atlas supplies gene–cell-type pairs. A gene-conditioned teacher scores each of the twelve cell-type labels by summing its answer-token log-probabilities. Its highest-scoring label becomes the target for the corresponding spatial image. The gap between the two highest scores defines the margin; a threshold of 1.0 retains 1.32M of approximately 1.75M candidate items. The 1.75M figure counts distillation candidates and does not denote an additional disjoint cohort.

The student uses 300,000 pseudo-labelled image rows and 200,000 replay rows for one epoch at learning rate 1e-4, with ofline stain variants and eager attention. Replay retains supervision for the existing tasks during adaptation to the image-to-cell-type objective.

## A.5 Classification decoding and probability correction

VGL-stGen closed-label generation follows a prefix tree of permitted labels. Label tokens omit leading spaces and special tokens. Under its tokenizer, first tokens are distinct within each of the twelve-class cell-type, nine-class tissue and binary label sets. Its interface can therefore return a first-token argmax and the corresponding restricted softmax score. This score is a model output, without a calibration guarantee. Comparator tokenisation and readout adaptations are specified in the multimodal benchmark protocol.

The main gene-classification comparison uses constrained generation, yielding balanced accuracy 0.6293 and macro-F1 0.6023. First-token argmax evaluated in the same run yields 0.6272 and 0.6019; the paths agree on 99.703% of the 224,210 evaluated items. A separate generation evaluation of the same cohort yields 0.6297 and 0.6029 and provides the gene-only reference for the joint image-andgenes comparison. These are two evaluations of the same cohort.

The deployed-task evaluation and the multimodal comparison use task-specific greedy-generation limits: 8 new tokens for task 1, 16 for tasks 2–5, 160 for tasks 6–7, 56 for tasks 8–9, 128 for task 10 and 16 for task 13. Tasks 11–12 instead make 25 autoregressive expression-level decisions with fixed query names. Token limits are counted using each model’s tokenizer. VGL-stGen places text before the image and uses its fast image processor (use\_fast=True); tasks 10–13 use eager attention, while tasks 1–9 use scaled dot-product attention. The multimodal benchmark preserves these task-specific inference settings while using each comparator’s native chat template and image processor. The separate expression-adaptation study keeps its common autoregressive readout fixed across its trained checkpoints.

For the nine-class tissue-classification analysis, each class probability is divided by its mean across the target group, then the class scores are renormalised. The reported analysis pools firsttoken probabilities over the entire 39-slide subset. Macro-F1 changes from 0.1200 to 0.2801 on those same predictions. Estimation of the divisor uses no labels, but the choice to apply correction followed labelled evaluation. The analysis is therefore exploratory and transductive. Per-slide grouping defines a diferent operating condition.

For genes-to-cell-type, the analogous first-token correction increases balanced accuracy from 0.6272 to 0.6876 and decreases macro-F1 from 0.6019 to 0.4452. The main comparisons use uncorrected predictions. This trade-of also illustrates why a single classification metric is insuficient here.

## A.6 External cohorts and annotation dependence

The four external spatial sections are public 10x lung cancer datasets [16]: Visium CytAssist squamous carcinoma (3,775 spots), Visium CytAssist large cell neuroendocrine carcinoma (6,189 spots), Xenium adenocarcinoma with multimodal segmentation (73,870 cells), and Xenium immunooncology with a custom add-on (140,376 cells). These counts are the processed evaluation units. The accompanying data-source manifest lists the oficial access links for these four datasets.

TACCO [14] transfers these coarse LuCA cell-type annotations to each section; the largest transferred component defines the reference class. The resulting targets share an atlas and label definition with training. The pool contains 214,246 Xenium centre-cell questions and 9,964 Visium dominant-region questions. Xenium contributes 95.6% of the evaluated items. The pooled score therefore primarily reflects the single-cell setting.

All four sections provide measured expression and were excluded from transcriptomic training. H&E from Visium LCNEC and the two Xenium sections supplies 4,846 of 24,822 binary cancer training rows; the Visium SCC source is absent from that manifest. This source-level check does not establish patient independence across all training stages. The two image-conditioned cell-type tasks and the two image-conditioned gene-list tasks therefore include previously seen tissue with unseen task labels. Gene-only tasks have no direct image input, but share the adapted weights. The pre-cancer cell-type checkpoint precedes this cancer-stage exposure; the deployed checkpoint follows it.

TCGA-LUAD diagnostic images [17] provide 357,692 patches from 452 slides and 398 patients, using an 87 µm field of view. VGL-stGen predictions are evaluated against reference labels from SegLungTCGA maps generated by ARA-CNN [13]. TCGA supplies the tissue images; ARA-CNN supplies the computational reference annotations. ARA-CNN and the cancer training corpus share LubLung provenance. Neither VGL-stGen nor the deployment probes use TCGA for gradient training, but 39 of these slides informed model development, label mapping and checkpoint selection. The expanded evaluation includes those slides. Agreement with these computational labels cannot establish independent pathologist-level validity.

The TUMOR class accounts for 15.09% of evaluated patches. An always-negative classifier therefore scores 0.849 raw accuracy and 0.500 balanced accuracy. We report balanced accuracy as the principal binary metric, together with AUROC.

## A.7 Gene classifiers and adaptation-capacity control

External comparison inputs are the same twenty gene symbols in descending expression order. Realised coverage depends on each model’s vocabulary. The simple classifiers use a vocabulary learned from training lists, with either a multi-hot encoding or a rank weight (n-r)/n, where r starts at zero and n is the list length. Training lists include mixed lengths; external lists contain twenty genes. Logistic regression uses balanced class weights, the SAGA solver and up to 200 iterations. The MLP uses one hidden layer of 256 units, up to 40 iterations and internal early stopping, without class weights. Model hyperparameters are fixed.

The frozen cell-model panel contains Geneformer V2-104M and Geneformer V2-104M\_CLcancer [25], and two C2S-Scale variants, with generic-language controls. Inputs preserve rank and are mapped to native tokens or gene text. This evaluates ranked marker lists, rather than each model’s native full-profile task. scGPT is excluded because its expression-value input [45] would require an additional rank-to-value convention for this benchmark; its suitability for expression-valued assays is outside this comparison.

Both Geneformer variants are fine-tuned on 581,394 source rows for two epochs, batch size 256 and maximum learning rate 5e-5. Training uses class-weighted cross-entropy, AdamW with weight decay 0.01, a OneCycleLR schedule and gradient clipping at 1.0. The final-epoch weights are evaluated on 224,210 external items.

Balanced accuracy and macro-F1 are reported together: logistic regression leads on the former and VGL-stGen on the latter. The gene-classification MLP comparison is provisional because the reported fit and subsequent refits have not been reconciled. Its 0.5962 macro-F1 is shown descriptively and does not support a stable superiority claim for VGL-stGen. This comparator is distinct from the expression MLPs fitted and evaluated below.

The adaptation-capacity control compares two readouts on the same backbone, loaded in bfloat16. The generative readout retains the pretrained language-model head and trains on labelanswer tokens. The discriminative readout adds a newly initialised twelve-way linear head and uses class cross-entropy. Both use rank 16, alpha 32, learning rate 1e-4, batch eight, 6,000 updates, a 300,000-row training pool and 40,000 evaluation items. The discriminative and generative arms have 32,819,212 and 32,788,480 trainable parameters, respectively. Their macro-F1 scores are 0.5209 and 0.5683. The output head and loss change jointly, so this comparison concerns their combined configuration. No repeated-seed interval is available.

## A.8 Image encoders and within-cohort reference fits

The cell-type morphology comparison uses fifteen frozen encoders, each followed by a class-balanced logistic probe with C=1.0, L-BFGS and up to 2,000 iterations. Extracted encoder features receive no additional standardisation. Transferred probes train on gene-teacher pseudo-labels and are evaluated on 54,058 external items. Within-cohort references use five-fold stratified cross-validation over cells. VGL-stGen’s main image-to-cell-type scores use all 224,210 items; on the same 54,058-item subset as the probes, its three evaluated checkpoints score 0.1607, 0.2268 and 0.1998.

The cancer comparison uses fourteen frozen encoders and a colour-histogram reference. Their deployment probes train on the same 30,965 unique cancer patches as VGL-stGen. Cancer features are L2-normalised; class-balanced logistic regression selects regularisation by grouped cross-validation within the source training data. A binary probe supplies the main comparison; nine-class-derived binary results are separate diagnostic arms. The model’s own medically adapted vision tower, called SigLIP in the result files, is also fitted with a linear probe as an additional control.

For cancer reference fits, the target cohort is split into three patient-grouped folds. Each held-out patch receives an out-of-fold prediction. A matched-size arm trains on 30,965 target-fold patches at the source class balance; a full arm uses all available training-fold patches. Both use the regularisation selected on the matched subset. The full arm is not separately tuned, and the direction of any tuning efect is unknown. These comparisons constrain a simple sample-count explanation but do not isolate a causal contribution of cohort shift.

The gene reference fits use five-fold cell-level cross-validation on approximately 100,000 external items. Cell and section dependence can make these references optimistic. The cancer folds instead separate patients. Scores from these diferent constructions should not be interpreted as directly comparable limits of biological information.

Within-cohort probe scores provide empirical supervised references for each representation. The transfer plots place target-cohort probe balanced accuracy on the horizontal axis and transferred balanced accuracy on the vertical axis. Both axes show empirical scores, not mathematical performance bounds. For imaging, VGL-stGen is compared with the reference from a probe on its own frozen tower; VGL-stGen itself is not adapted within the target cohort.

## A.9 Gene-list evaluation

Four gene-list tasks use a seeded sample of 1,000 external items. Generation targets twenty genes; completion masks five of the twenty and supplies the remaining fifteen. The sample contains 949 Xenium cells and 51 Visium spots, with three cell types represented by at most three items. Geneset F1 is $2 | \mathbb { A } \cap \mathbb { B } | / ( | \mathbb { A } | + | \mathbb { B } | )$ , averaged over items. These samples provide task-level estimates with little support for subgroup claims.

Completion evaluation prompts specify the actual number of withheld genes. Some training prompts specified output counts that difered from their targets.

Training-global frequency rules provide transferable references. External-global and externalcell-type frequency rules use target-cohort statistics and provide descriptive comparisons with additional information. On image-to-gene generation, the training-global rule scores 0.0367, the externalglobal rule 0.2913 and the external-cell-type rule 0.4009. The last comparison changes both cohort and cell-type conditioning. None of these scores is a mathematical upper bound or a causal decomposition of the transfer gap.

The frequency analysis counts one row per cell or spot: 1,221,886 training items and 224,210 external items, compared over 1,049 shared genes. The top-twenty frequency lists share five symbols. Panel coverage, cell composition and assay processing all difer between the cohorts, so their separate efects remain unresolved.

Internal gene-list F1 is 0.4192, 0.3943, 0.3310 and 0.3972 for the four tasks. Within the imageto-gene split, recorded test cell/spot identifiers do not overlap its training split, but all 69 internal image-test samples also occur in training. The internal cell-type-to-gene evaluation covers efectively 4 classes. These conditions difer from the expression-task split described below.

For a partial gene list without an image or cell-type input, the interface composes cell-type prediction with cell-type-conditioned completion. Its recorded F1 is 0.0472, compared with 0.0484 when supplied the reference cell type; the intermediate cell-type prediction is correct on 81.6% of sampled items. This is a composed interface operation rather than an additional directly trained task.

## A.10 Expression evaluation and image-shufle controls

The expression-task training and test manifests have disjoint sample and recorded cell/spot identifiers. Each task is scored on 1,000 existing benchmark rows. The masked-image evaluation contains 665 Xenium cells and 335 Visium spots from seven samples. All seven samples are represented in gene-list training, and 692 benchmark cells or spots have matching identifiers and identical source images in that training set. The masked-cell-type benchmark uses the atlas-level identifier ‘LuCA\_ NCSCLC’, which is also present in gene-list training; its generated row-based cell identifiers do not establish a cross-task cell-barcode or donor linkage. These internal benchmarks therefore do not establish patient-independent generalisation across the full training sequence. The benchmark had already been analysed before the controlled adaptation study.

In the deployed model, the masked-expression objectives were trained without conditioning gene lists. Both decoding procedures use the same checkpoint and evaluation rows. This comparison measures inference behaviour for those weights. The separate adaptation experiment supplies fully conditioned examples to both loss conditions and holds their inference procedure fixed.

Free generation is scored as emitted. Fixed-name decoding parses the twenty-five masked entries from the supplied hundred-gene list, then predicts each level from five permitted first tokens. Two independent extractors identify the same masked entries on all 176,008 held-out masked-task rows. Autoregressive decoding conditions each slot on the model’s previous level answers. Pair-F1 over the fixed names equals slot-level accuracy. These results apply to the specified input format.

Per-gene and gene-by-cell-type tables are estimated from the training split. Their scores are 0.6198 and 0.6963. Both omit the seventy-five visible gene levels available to VGL-stGen. The deployed autoregressive scores are 0.6596 and 0.6858, with paired model-minus-table row intervals of [+0.0350, +0.0445] and [−0.0160, −0.0052]. Context-aware MLPs on these same rows give threeseed mean pair-F1s of 0.6811 and 0.7300, respectively. The MLPs align available inputs and targets but use diferent encoders and training histories.

Image-shufle diagnostics retain the expression prompt and replace the paired image. Under teacher-forced preceding answers, the intact-minus-shufled efect is 0.0097 [0.0071, 0.0124] for VGLstGen and 0.0001 [−0.0010, +0.0012] for the base model. With preceding answer positions fixed to Absent, VGL-stGen’s efect is 0.0092 [0.0067, 0.0119]. These settings difer from autoregressive generation and cannot be subtracted from its model-minus-table margin to assign a fraction of the gain to images.

Very-high-gene prediction is evaluated with the true answer length supplied. The model scores 0.2828; the 0.2247 frequency and 0.0144 random references use evaluation-shard gene support and answer lengths. Levels-to-cell-type prediction scores 0.7420 raw accuracy from the first ten sampled pairs. Its majority reference of 0.362 is selected within evaluation shards. These task-specific references use evaluation information, whereas the masked-completion frequency tables use training data. Reported scores aggregate four evaluation shards.

## A.11 Context-aware expression baselines

MLPs encode all 100 supplied gene–state pairs, including 75 observed levels and 25 masks, by summed embeddings divided by ten and layer normalisation. After concatenating the extra inputs, two hidden linear layers with GELU activations and dropout 0.1 after the first hidden layer predict five levels per gene. All variants share a training-derived vocabulary of 2,000 genes plus an unknown entry; five-class cross-entropy is applied only to the 25 queried genes. The context-only image-task model receives an indicator specifying a centre-cell or tissue-region query. The full image model additionally receives frozen 1,536-dimensional UNI2-h CLS features [40]; the cell-type model receives the supplied twelve-class one-hot label plus an unknown category.

UNI2-h processes the same RGB images resized to 224 pixels and normalised by ImageNet statistics. Features are standardised using training-set means and population standard deviations, floored at 1e-4. This extractor applies no additional stain or geometric augmentation. The raw images, visible expression, query names, target levels and query-region or cell-type inputs are aligned with VGL-stGen. The comparison does not match vision encoders or staged training histories.

The original task splits supply 1,037,183 image training rows, 292,532 cell-type training rows and 10,000 validation rows per task. AdamW uses batch 1,024, weight decay 1e-4, a maximum of twenty epochs and validation patience three. A seed-42 search over widths 256/512 and learning rates 1e-3/3e-4 selects width 256 for the context-only image-task and cell-type models, and width 512 for the full image model, all at learning rate 1e-3. The selected settings are repeated at seeds 7 and 2026; each run uses its validation-selected epoch. Configurations and weights are fixed before test scoring. Image validation contains Xenium only, whereas the benchmark includes both platforms.

The context-only image-task MLP has mean pair-F1 0.6666. The full image MLP reaches 0.6811, but the two selected configurations difer in width as well as input. A separate width-512 matchedsetting comparison has image-added diferences of −0.0024, +0.0231 and +0.0141 across seeds; the direction is not consistent. Neither comparison establishes that images are dispensable or isolates a universal imaging advantage.

## A.12 Controlled expression adaptation and image permutation

Both conditions initialise from the same deployed rank-96 adapter. They share 4,096 additional training rows: 1,024 Xenium image, 1,024 Visium image and 2,048 cell-type rows. Additional training uses a fixed subset of the existing task-training data. The complete 100-gene list, 75 observed levels, image or supplied cell type, answer order and row order are identical between conditions. Training uses seeds 42, 7 and 2026, with no replay of other tasks.

Full-answer supervision applies mean full-vocabulary cross-entropy to all answer tokens and end-of-turn. Level-only supervision applies it to exactly the first tokens of the 25 requested levels; other answer positions receive no direct loss. This changes supervised positions without introducing a five-class-normalised training loss or a new prediction head. Both conditions use autoregressive inference from the model’s own preceding level predictions, restricted to the same five permitted first tokens.

Each run uses 256 updates, efective batch 16, AdamW learning rate 5e-5, sixteen linear warmup updates followed by a constant rate, weight decay 0.01, betas (0.9, 0.999), epsilon 1e-8 and gradient clipping at 1.0. The base is NF4 double-quantised with bfloat16 autocast; trainable language-side adapter parameters are float32, rank 96, alpha 32 and dropout 0.05. The vision tower and projector remain frozen. Gradient checkpointing is enabled. The terminal checkpoint is used without a learning-rate or intermediate-checkpoint search.

Development evaluation uses 96 Xenium, 96 Visium and 128 cell-type rows. Three Visium samples are selected by a fixed hash of sample names, excluded entirely from additional training, and sampled at 32 rows each. Those samples were present in the initial model’s training corpus. Other cross-family exposure is unverified. The equal-weight mean pair-F1 across these three groups and three seeds selects level-only as the representative supervision condition. The representative seed is fixed at 42, independent of its benchmark score. Both conditions at all three seeds are retained for the reported benchmark comparison, with weights and evaluation rows fixed before scoring.

Table 6 reports level macro-F1 and accuracy on non-Absent targets; these reveal performance obscured by the frequent Absent class. Table 7 reports every seed and platform. Level-only image pair-F1 exceeds full-answer training at all seeds on both platforms. Cell-type pair-F1 does not improve beyond full-answer training, although its secondary metric means are slightly higher. The MLP retains higher means on both secondary metrics in both tasks.

Table 6: Secondary expression metrics on the same benchmark rows. Mean ± sample SD across 3 individual seeds; the deployed checkpoint is a single fitted model. Non-Absent accuracy restricts targets to the 4 detected-expression levels.
<table><tr><td>Task</td><td>Model</td><td>Level macro-F1</td><td>Non-Absent accuracy</td></tr><tr><td>Image</td><td>Deployed</td><td>0.3780</td><td>0.2276</td></tr><tr><td>Image</td><td>Full-answer</td><td> $0 . 4 2 6 9 \pm 0 . 0 0 2 4$ </td><td> $0 . 2 6 6 4 \pm 0 . 0 0 3 8$ </td></tr><tr><td>Image</td><td>Level-only</td><td> $0 . 4 4 6 3 \pm 0 . 0 0 1 5$ </td><td> $0 . 2 9 3 0 \pm 0 . 0 0 2 1$ </td></tr><tr><td>Image</td><td>Full-input MLP</td><td> $0 . 4 5 8 6 \pm 0 . 0 0 2 1$ </td><td> $0 . 3 1 6 0 \pm 0 . 0 0 1 1$ </td></tr><tr><td>Cell type</td><td>Deployed</td><td>0.4625</td><td>0.3611</td></tr><tr><td>Cell type</td><td>Full-answer</td><td> $0 . 5 4 4 5 \pm 0 . 0 0 0 8$ </td><td> $0 . 4 4 3 3 \pm 0 . 0 0 4 9$ </td></tr><tr><td>Cell type</td><td>Level-only</td><td> $0 . 5 4 7 2 \pm 0 . 0 0 2 2$ </td><td> $0 . 4 5 3 5 \pm 0 . 0 0 2 3$ </td></tr><tr><td>Cell type</td><td>Full-input MLP</td><td> $0 . 5 6 7 5 \pm 0 . 0 0 2 6$ </td><td> $0 . 4 7 8 6 \pm 0 . 0 0 4 7$ </td></tr></table>

For all six additional-training checkpoints, rows within each sample and original platform are deterministically ordered by a SHA-256 hash of their identifiers. Each row receives the next row’s image, wrapping the final row to the first. Every donor image difers from the original. Visible expression, targets and autoregressive decoding are unchanged. On the pooled image benchmark, the intact-minus-permuted pair-F1 diference is positive in each seed: full-answer 0.00504/0.00632/0.00656 and level-only 0.00796/0.00712/0.00820 for seeds 42/7/2026. Mean differences are 0.00597 and 0.00776, with sample CIs [0.00316, 0.00764] and [0.00263, 0.01042]. The paired diference between these efects is +0.00179, row CI [−0.00097, 0.00460] and sample CI [−0.00067, 0.00336]. Level-only’s normal-input advantage of 0.00848 remains 0.00669 after permutation. These comparisons establish sensitivity to correctly paired images on the pooled benchmark under both objectives; they do not establish that level-only training increased that sensitivity. No full thirteen-task evaluation was performed for the additional-training checkpoints; the task inventory and classification results refer to the deployed model.

Table 7: Every additional-training seed and image platform. The pooled image set contains 665 Xenium cells and 335 Visium spots; the cell-type set contains 1,000 rows. Diference is level-only minus full-answer pair-F1. These repeat additional training from one initial checkpoint.
<table><tr><td>Evaluation</td><td>Seed</td><td>Full-answer</td><td>Level-only</td><td>Difference</td></tr><tr><td>Image pooled</td><td>42</td><td>0.67096</td><td>0.68032</td><td>+0.00936</td></tr><tr><td>Image pooled</td><td>7</td><td>0.67252</td><td>0.68164</td><td>+0.00912</td></tr><tr><td>Image pooled</td><td>2026</td><td>0.67252</td><td>0.67948</td><td>+0.00696</td></tr><tr><td>Xenium</td><td>42</td><td>0.71717</td><td>0.72523</td><td>+0.00806</td></tr><tr><td>Xenium</td><td>7</td><td>0.71705</td><td>0.72493</td><td>+0.00788</td></tr><tr><td>Xenium</td><td>2026</td><td>0.71735</td><td>0.72343</td><td>+0.00608</td></tr><tr><td>Visium</td><td>42</td><td>0.57922</td><td>0.59116</td><td>+0.01194</td></tr><tr><td>Visium</td><td>7</td><td>0.58412</td><td>0.59570</td><td>+0.01158</td></tr><tr><td>Visium</td><td>2026</td><td>0.58352</td><td>0.59224</td><td>+0.00872</td></tr><tr><td>Cell type</td><td>42</td><td>0.72260</td><td>0.72028</td><td>-0.00232</td></tr><tr><td>Cell type</td><td>7</td><td>0.72240</td><td>0.71964</td><td>-0.00276</td></tr><tr><td>Cell type</td><td>2026</td><td>0.72132</td><td>0.72040</td><td>-0.00092</td></tr></table>

Table 8 reports every intact and permuted score in the six-checkpoint comparison.

Table 8: Image pairing under both supervision conditions. Each seed is evaluated on the same 1,000 rows and 25 queried levels per row with 75 observed levels. The within-sample/platform image permutation is identical across all 6 checkpoints. Positive diferences indicate higher pair-F1 with intact images.
<table><tr><td>Supervision</td><td>Seed</td><td>Intact</td><td>Permuted</td><td>Difference</td></tr><tr><td>Full-answer</td><td>42</td><td>0.67096</td><td>0.66592</td><td>+0.00504</td></tr><tr><td>Full-answer</td><td>7</td><td>0.67252</td><td>0.66620</td><td>+0.00632</td></tr><tr><td>Full-answer</td><td>2026</td><td>0.67252</td><td>0.66596</td><td>+0.00656</td></tr><tr><td>Level-only</td><td>42</td><td>0.68032</td><td>0.67236</td><td>+0.00796</td></tr><tr><td>Level-only</td><td>7</td><td>0.68164</td><td>0.67452</td><td>+0.00712</td></tr><tr><td>Level-only</td><td>2026</td><td>0.67948</td><td>0.67128</td><td>+0.00820</td></tr></table>

## A.13 Completion with fewer observed expression levels

We retain the same 1,000 image rows, 25 queried genes, target levels and image permutation while reducing observed gene levels from 75 to 50 and 25. Within each row, observed genes are deterministically ordered by a SHA-256 hash of the row identifier and upper-case gene name. The first 50 or 25 are retained in their original relative order, yielding nested subsets independent of target levels or predictions. Exact hashing specifications for gene selection and image permutation accompany the source data. Excluded observed pairs are removed from the prompt; the 25 original query masks remain. Expression-level definitions and targets are unchanged.

The VGL-stGen checkpoint is the development-selected level-only seed 42. Image-inclusive and context-only MLPs use their existing matched-width fits (512, learning rate 0.001, seeds $4 2 / 7 / 2 0 2 6 )$ Excluded MLP input tokens become padding; query masks, the recipient’s centre-cell/region query indicator, training-derived feature normalisation and all weights remain fixed. Only image features are permuted in the image-inclusive MLP. These are inference-time stress conditions for models trained with 75 observed levels, with no budget-specific retraining.

Table 9 gives all budgets, image conditions and secondary metrics. The primary budget contrast is the intact-minus-permuted image efect at 25 observed levels minus that at 75. Paired row and sample bootstrap preserve the original platform mixture, conditional on the fixed models, one image permutation and one nested-subset rule. This analysis evaluates reduced input context within synthetic marker-panel queries; it does not establish performance for a smaller assay panel or measurement-cost savings.

Table 9: Synthetic completion under reduced observed context, with the same 25 queries. Obs. denotes observed gene levels. VGL-stGen uses the development-selected level-only seed 42; MLPs report mean ± sample SD across 3 fixed seeds (width 512, learning rate 0.001). All models retain weights trained with 75 observed levels. The context-only MLP has no image input.
<table><tr><td>Obs.</td><td>Model / image</td><td></td><td>Pair-F1</td><td>Level F1m</td><td>Non-Absent acc.</td></tr><tr><td>75</td><td>VGL-stGen / Intact</td><td></td><td>0.6803</td><td>0.4466</td><td>0.2930</td></tr><tr><td>75</td><td>VGL-stGen / Permuted</td><td></td><td>0.6724</td><td>0.4308</td><td>0.2727</td></tr><tr><td>75</td><td>Image MLP / Intact</td><td></td><td> $0 . 6 8 1 1 \pm 0 . 0 0 3 8$ </td><td> $0 . 4 5 8 6 \pm 0 . 0 0 2 1$ </td><td> $0 . 3 1 6 0 \pm 0 . 0 0 1 1$ </td></tr><tr><td>75</td><td>Image MLP / Permuted</td><td></td><td> $0 . 6 7 0 0 \pm 0 . 0 0 3 0$ </td><td> $0 . 4 3 7 5 \pm 0 . 0 0 0 5$ </td><td> $0 . 2 9 0 3 \pm 0 . 0 0 2 6$ </td></tr><tr><td>75</td><td>Context MLP / </td><td></td><td> $0 . 6 6 9 5 \pm 0 . 0 0 9 2$ </td><td> $0 . 4 5 2 4 \pm 0 . 0 0 8 6$ </td><td> $0 . 3 1 8 7 \pm 0 . 0 0 5 5$ </td></tr><tr><td>50</td><td> $\mathrm { V G L - s t G e n } \ .$ </td><td>/ Intact</td><td>0.6681</td><td>0.4173</td><td>0.2593</td></tr><tr><td>50</td><td>VGL-stGen / Permuted</td><td></td><td>0.6609</td><td>0.4014</td><td>0.2401</td></tr><tr><td>50</td><td>Image MLP / Intact</td><td></td><td> $0 . 6 6 7 4 \pm 0 . 0 0 4 8$ </td><td> $0 . 4 4 3 7 \pm 0 . 0 0 1 9$ </td><td> $0 . 3 0 9 6 \pm 0 . 0 0 1 6$ </td></tr><tr><td>50</td><td> $\mathrm { I m a g e \ M L P / \ P e r m u t e d }$ </td><td></td><td> $0 . 6 5 5 7 \pm 0 . 0 0 5 4$ </td><td> $0 . 4 2 3 1 \pm 0 . 0 0 3 6$ </td><td> $0 . 2 8 5 4 \pm 0 . 0 0 0 3$ </td></tr><tr><td>50</td><td> $\mathrm { C o n t e x t \ M L P \ / - }$ </td><td></td><td> $0 . 6 4 7 7 \pm 0 . 0 1 1 6$ </td><td> $0 . 4 2 7 8 \pm 0 . 0 0 8 0$ </td><td> $0 . 3 0 3 8 \pm 0 . 0 0 4 0$ </td></tr><tr><td>25</td><td> $\mathrm { { V G L - s t G e n } / \ I n t a c t }$ </td><td></td><td>0.6599</td><td>0.3839</td><td>0.2164</td></tr><tr><td>25</td><td> $\mathrm { { V G L - s t G e n } / \ p e r m u t e d }$ </td><td></td><td>0.6496</td><td>0.3631</td><td>0.1928</td></tr><tr><td>25</td><td> $\mathrm { I m a g e \ M L P / \ I n t a c t }$ </td><td></td><td> $0 . 6 4 6 9 \pm 0 . 0 0 5 4$ </td><td> $0 . 4 2 1 5 \pm 0 . 0 0 3 1$ </td><td> $0 . 2 9 6 8 \pm 0 . 0 0 4 0$ </td></tr><tr><td>25</td><td> $\mathrm { I m a g e \ M L P / \ P e r m u t e d }$ </td><td></td><td> $0 . 6 3 3 0 \pm 0 . 0 0 7 8$ </td><td> $0 . 3 9 8 4 \pm 0 . 0 0 5 9$ </td><td> $0 . 2 7 0 5 \pm 0 . 0 0 1 5$ </td></tr><tr><td>25</td><td> $\mathrm { C o n t e x t \ M L P \ / - }$ </td><td></td><td> $0 . 6 0 7 2 \pm 0 . 0 1 5 2$ </td><td> $0 . 3 9 0 3 \pm 0 . 0 0 7 9$ </td><td> $0 . 2 8 0 5 \pm 0 . 0 0 2 5$ </td></tr></table>

For the fixed $\mathrm { \Delta V G L - s t G e n }$ checkpoint, intact-image pair-F1 is 0.68032, 0.66808 and 0.65992 at 75, 50 and 25 observed levels. Image-pairing efects are +0.00796, +0.00720 and +0.01032; the latter two have sample CIs [0.00200, 0.01125] and [0.00561, 0.01443]. The 25-minus-75 change in image efect is +0.00236, row CI [−0.00180, 0.00644] and sample CI [−0.00037, 0.00543], leaving a pooled increase unresolved. The corresponding point changes difer across Xenium (+0.00481) and Visium (−0.00251).

At 25 observed levels, VGL-stGen pair-F1 is higher than the image-MLP mean (0.65992 versus 0.64691), while non-Absent accuracy (0.2164 versus 0.2968) and level macro-F1 (0.3839 versus 0.4215) are lower. These results document a metric-specific trade-of under reduced context, rather than an overall completion advantage. All budgets and image conditions are retained in the table and accompanying source data.

## A.14 Alignment checks and cancer label definitions

Segmentation colour maps are aligned to slides by axis-wise linear scaling. A tissue-mask comparison searches translations over ±8% of a common grid; the unshifted alignment is optimal on all 452 slides, aspect-ratio error is at most 0.0009, and horizontal or vertical flips do not improve the fit. This evaluates geometric registration independently of the biological correctness of the segmentation labels.

Tissue-mask quality control identifies 4,999 nearly empty tiles, or 1.40% of the evaluated set. Excluding them changes balanced accuracy from 0.7532 to 0.7519; restricting evaluation by the alignment threshold gives 0.7596. These post-hoc sensitivity analyses accompany the unfiltered estimate.

To assess overlap with known development patients, we exclude all slides belonging to the 39 patients represented by the 39-slide development subset. This removes 40 slides and 31,574 patches, leaving 412 slides, 359 patients and 326,118 patches. With unchanged predictions, VGLstGen’s observed balanced accuracy is 0.7523, patient-bootstrap CI [0.7407, 0.7628], and AUROC 0.8253, compared with observed balanced accuracy 0.7534 on the full cohort. The original 0.7532 headline is a slide-bootstrap mean. Comparator probes are reconstructed using their original sourcetraining settings and verified against their full-cohort scores before the patient exclusion. VGL-stGen remains sixth among the fifteen systems; the own-tower probe is an additional control outside this ranking. This sensitivity analysis does not remove adaptive development, shared annotation lineage or possible foundation-model pretraining exposure.

The label-definition comparison refits each encoder probe under the wide mapping (TUMOR, NECROSIS and MIXED positive) and the TUMOR-only mapping on the same 452-slide cohort. Its median paired balanced-accuracy diference is 0.1191 across the fourteen encoders and 0.0436 for the colour reference. Each encoder’s patient-paired interval excludes zero. Both target definition and fitted classifier change, so the result measures benchmark sensitivity rather than the clinical superiority of one mapping. The mapping was chosen during development.

## A.15 Statistical analysis

Binary cancer intervals use 1,000 slide-bootstrap draws, with patient-bootstrap intervals to account for patients contributing multiple slides. The empirical balanced accuracy is 0.7534 and its bootstrap mean is 0.7532. The slide interval is [0.7452, 0.7610]; the patient interval is [0.7441, 0.7622]. Neither incorporates annotation error, repeated-training variation or selection bias.

The principal paired cell-type comparisons use 2,000 hierarchical bootstrap draws, resampling four sections and then items within each sampled section. The same resampled indices are used for both systems. With only four sections, these intervals remain sensitive to the observed cohort composition. They should be read alongside section-specific results, not as strong evidence about a population of new patients. The retained Geneformer fine-tuning comparison instead uses cell-level resampling. The deployed model’s expression and image-shufle intervals resample rows.

The controlled adaptation study reports means and sample standard deviations across three individually fitted models, not ensembles. Paired contrasts first average matched-seed per-row diferences and then use 2,000 bootstrap draws. Image row intervals resample rows within platform; image sample intervals resample samples with replacement within platform, retain all their rows and preserve the fixed Xenium/Visium mixture. Sample identifiers are not verified patients. The celltype benchmark has only one sample identifier, so no sample-generalisation interval is estimated. Intervals condition on the fitted models and do not estimate a population of training seeds. The three seeds vary additional training from a common initial checkpoint, not the entire pretraining

and adaptation history.

Balanced accuracy and macro-F1 are reported jointly. Confidence intervals for modelperformance comparisons are not adjusted for multiple comparisons. A confidence interval crossing zero is not evidence of equivalence or non-inferiority.

## A.16 Data and code availability

Code and weights are hosted at https://huggingface.co/Portraier/VGL-stGen-0.5-Lung, with gated access planned for the preprint release. Reproduction of the evaluations also requires the processed manifests, annotation-transfer outputs and comparison predictions; internal expression results require the corresponding held-out split.

## B TCGA bulk RNA-seq analysis

## B.1 Gene profiles and molecular endpoints

The analysis includes 459 TCGA-LUAD patients with readable diagnostic FFPE slides and primarytumour bulk RNA-seq [17]. The deployed checkpoint predicted twenty gene symbols for each of 112 random tissue patches per slide at a 128 µm field of view. Each slide’s gene score is its fraction of patches naming that gene. The 317 evaluated genes satisfy both eligibility criteria: occurrence in predictions for at least 10% of slides and measured expression above one TPM in at least half of patients.

Measured expression was transformed as log2(TPM+1), and both measured and image-derived matrices were centred within gene. This centres the analysis on between-patient variation for each gene. Per-gene Pearson correlations were assessed with BH-FDR, with shufled patient pairing providing a null comparison. Marker-programme scores were compared separately. Lymphoid, smooth-muscle, stromal, endothelial and alveolar programmes met $\mathrm { F D R } < 0 . 0 5 $ epithelial/tumour and myeloid programmes did not. Proliferation was not scored because MKI67 was the only eligible proliferation marker.

Fig. 4(b) shows the programme correlations and the significance marked in the original plot. Of 317 genes, 152 met the FDR threshold, versus a permutation-null 95th percentile of one.

## B.2 Patient matching and covariate controls

Each slide’s matching bulk profile was ranked among all 459 patients. The median rank was the 29.5th percentile, compared with chance 50; restricting candidates to the same tissue source site gave 27.9. The correct patient was in the top 10% for 22% of slides. Shufling slide–bulk pairing gave a median rank of 48.3 and no significant genes. Sequencing depth, magnification, source site and tumour fraction together explained cross-validated $\mathrm { R } ^ { 2 } = 0 . 0 1 5 ;$ adding the image-derived gene profile increased R² by 0.057.

## B.3 Visualisation and scope

Fig. 4(a) displays 24 selected genes in 459 patients, ordered by measured lymphoid score and averaged over approximately seven adjacent patients per displayed column. Both heatmaps are z-scored within gene. Programme associations and the paired versus shufled patient-rank distributions appear in panels (b–d).

The bulk measurements and diagnostic slides come from diferent tissue blocks. The separate named-gene maps in Fig. 4(e) illustrate applying molecular queries across one slide and are described in the application methods.

## C Application analysis methods

## C.1 Single-cell and Xenium cell typing

We evaluated genes → cell type (Task 1) with the checkpoint (merged bfloat16 weights, constrained decoding). GSE127465 comprises seven treatment-naive NSCLC patients with published annotations [42]. After removal of housekeeping, mitochondrial and ribosomal genes, the analysis retains 2,000 highly variable genes and samples 7,993 tumour-tissue cells. Each input contains the twenty highest-expressed retained genes in rank order. Macrophage/monocyte and conventional dendriticcell predictions are combined as MoMacDC to match the eleven-class reference. Patient-specific malignant clusters are treated as epithelial, with an additional analysis excluding them.

The 10x Xenium Human Lung Cancer FFPE application [16] samples 12,000 cells with at least ten transcripts from a 377-gene panel. Marker scores z-score each gene across cells and average within canonical sets; 39 of 78 markers are present on the panel. Fig. 3 shows the GSE127465 comparison, the Xenium marker matrix and an aligned post-Xenium H&E field. The epithelial marker set contains EPCAM alone and the predicted pDC group contains one cell. The Xenium comparison relies on markers rather than independent labels.

Balanced accuracy is 0.815 and macro-F1 0.819, with balanced accuracy 0.814 after malignantcell exclusion and 0.70–0.86 per patient. Major confusions were that 47% of NK cells were called T cells, 46% of pDCs were called plasma cells, and 22% of B cells were called T cells. Confidence was lower for incorrect calls (median 0.47 vs 0.83). Balanced accuracy falls to 0.775 after geneorder shufling and 0.282 without filtering. Xenium weighted purity is 0.916 across twenty-four unsupervised clusters. These application results use their own preprocessing and harmonised label space and are separate from the main 224,210-item evaluation.

## C.2 Whole-slide named-gene illustration

Fig. 4 uses TCGA-49-4510 with 526 tiles of 128 µm. Each tile image was paired with gene names and no observed expression levels, outside the benchmarked Task 11 input condition. Four displayed maps concern EPCAM, CD3E, ACTA2 and PECAM1. High was not emitted in this application; four of the five permitted levels were used. Bulk RNA was extracted from a diferent tissue block than the diagnostic slide.

## D Multimodal benchmark protocol

The benchmark evaluates VGL-stGen and ten of-the-shelf medical or general-purpose multimodal models on the same thirteen tasks. The comparator families are MedGemma [9], Gemma 3 [18], Gemma 4 [19], Qwen3-VL [20], Qwen3.8 [21], DeepSeek-VL2 [22] and GLM-4.6V-Flash [23]. The full GLM-4.6V checkpoint is not included. Table 10 specifies the task-level settings. Task identifiers refer to the deployed task inventory.

The original task pipeline supplies inputs and instructions, collates batches, extracts answers and computes scores. The same canonical image files and molecular context are supplied to every model, with model-native image processing and chat templates. Consequently, matched inputs do not imply identical resized tensors, visual-token counts or tokenized instruction lengths. Table 11 identifies the exact model repositories and image-processing configurations. Full revisions, library versions and source hashes are included with the ancillary benchmark data.

Table 10: Fixed task evaluation settings shared across models. Tokens are maximum generated tokens; T11–12 instead make 25 autoregressive level choices per row. T2–3 preserve the original mixed-task batch composition. Counts are per task. All tasks except T4 use NF4; T4 uses BF16. Task key: T1, genes → cell type; T2, image + genes → cell type; T3, image → cell type; T4, image → binary tumor status; T5, image → nine-class tissue type; T6, image → gene names; T7, cell type → gene names; T8, image + partial gene list → missing gene names; T9, cell type + partial gene list → missing gene names; T10, image → very-high-expression gene names; T11, image + observed levels → queried levels; T12, cell type + observed levels → queried levels; T13, gene–level pairs → cell type.
<table><tr><td>Tasks</td><td>Items per task</td><td>Output limit</td><td>Batch / shards</td></tr><tr><td>1</td><td>224,210 cells/spots</td><td>8 tokens</td><td>16 / 4</td></tr><tr><td>2-3</td><td>224,210 cells/spots</td><td>16 tokens</td><td>16 mixed / 4</td></tr><tr><td>4</td><td>357,692 patches / 452 slides</td><td>16 tokens</td><td>32 / 8 slide shards</td></tr><tr><td>5</td><td>30,719 patches / 39 slides</td><td>16 tokens</td><td>32</td></tr><tr><td>6-7</td><td>1,000 rows</td><td>160 tokens</td><td>8/4</td></tr><tr><td>8-9</td><td>1,000 rows</td><td>56 tokens</td><td>8/4</td></tr><tr><td>10</td><td>1,000 rows</td><td>128 tokens</td><td>8/4</td></tr><tr><td>11-12</td><td>1,000 rows / 25,000 levels</td><td>25 level choices</td><td>1/4</td></tr><tr><td>13</td><td>1,000 rows</td><td>16 tokens</td><td>8/4</td></tr></table>

Table 11: Model repositories and native image processing. VGL-stGen additionally uses the deployed lung adapter (merged for binary tumor classification, T4). Full revisions, processor settings, library versions and loading exceptions are supplied in the ancillary model-conditions files. Budgets describe configuration, not measured per-image token counts.
<table><tr><td>Model</td><td>Repository</td><td>Image processing</td></tr><tr><td>VGL-stGen</td><td>google/medgemma-1.5-4b-it</td><td>896 × 896; mean/std 0.5</td></tr><tr><td>MedGemma-1.5-4B</td><td>google/medgemma-1.5-4b-it</td><td>896 × 896; mean/std 0.5</td></tr><tr><td>Gemma3-4B</td><td>google/gemma-3-4b-it</td><td>896 × 896; mean/std 0.5</td></tr><tr><td>Qwen3-VL-8B</td><td>Qwen/Qwen3-VL-8B-Instruct</td><td>Dynamic resolution; 16-pixel patches; merge 2</td></tr><tr><td>Qwen3-VL-32B</td><td>Qwen/Qwen3-VL-32B-Instruct</td><td>Dynamic resolution; 16-pixel patches; merge 2</td></tr><tr><td>Gemma4-12B</td><td>google/gemma-4-12B-it</td><td>Variable resolution; 16-pixel patches; max 280 soft tokens</td></tr><tr><td>GLM-4.6V-Flash</td><td>zai-org/GLM-4.6V-Flash</td><td>Dynamic resolution; 14-pixel patches; merge 2</td></tr><tr><td>Qwen3.8-27B</td><td>Qwen/Qwen3.8-27B</td><td>Dynamic resolution; 16-pixel patches; merge 2</td></tr><tr><td>Gemma4-31B</td><td>google/gemma-4-31B-it</td><td>Variable resolution; 16-pixel patches; max 280 soft tokens</td></tr><tr><td>DeepSeek-VL2- small</td><td>deepseek-ai/deepseek-vl2-small</td><td>384-pixel dynamic tiles; 14-pixel patches; downsample 2</td></tr><tr><td>DeepSeek-VL2</td><td>deepseek-ai/deepseek-vl2</td><td>384-pixel dynamic tiles; 14-pixel patches; downsample 2</td></tr></table>

Task 4 uses BF16 weights; all other tasks use NF4 double quantization with BF16 computation in quantized layers. VGL-stGen uses its merged model for task 4 and its deployed adapter for the other tasks. Gemma 4’s multimodal projection and vision modules excluded by its loading configuration remain unquantized; k-bit preparation can cast unquantized parameters to FP32.

DeepSeek uses eager language attention throughout and its native memory-eficient vision attention. Other models use SDPA for tasks 1–9 and eager attention for tasks 10–13. Model-dependent loading and tokenization adaptations preserve the requested input and score definition.

For categorical outputs, allowed-label tries are constructed with each tokenizer. Gene-list outputs are parsed using the original permissive gene parser. Expression completion fixes the 25 gene names and selects successive level tokens autoregressively; prefix-sharing tokens are handled as branches of the allowed-level trie. Model-specific end-of-turn markers delimit the answer, and complete raw generations are retained for audit. Structured result records are not counted as evidence that a model freely generated valid JSON.

For task 5, the same pooled marginal-probability correction is applied to each model. Raw and corrected macro-F1 are both reported. This is a transductive comparison on the development subset, and the correction was selected after labelled evaluation. For task 10, the requested genelist length is the true answer length. Tasks 11 and 12 retain 75 observed levels; this benchmark is conditional completion rather than expression prediction from H&E alone.

## E Benchmark reproducibility and uncertainty

Twelve of thirteen VGL-stGen aggregate scores match the historical task inventory at its reported precision. For task 4, the matched H100 rerun has observed balanced accuracy 0.7526 and a slidebootstrap mean of 0.7524. The latter is the point estimate in the new model comparison. The earlier 0.7532 remains in the original task inventory and encoder comparison, representing that earlier evaluation. Gene-list point estimates preserve the original sample-weighted aggregation of shard-rounded scores; bootstrap statistics instead use unrounded per-row scores. Consequently, some paired diferences difer in the last displayed digit from subtraction of the displayed table entries.

We resample the same units jointly for VGL-stGen and each comparator, using 1,000 draws and seed 0. Tasks 1–3 resample cells or spots, task 4 resamples slides, task 5 resamples patches, and tasks 6–13 resample input rows. Task 5 recomputes the pooled correction within each bootstrap draw. Gene sets and all queried levels within a row remain together. These intervals measure conditional sampling uncertainty for the fitted models. They do not include training variation, account for all section or patient clustering, or correct for the multiple comparisons. They are separate from the section-bootstrap intervals used for the earlier task-specific gene classifiers.

Table 12 reports VGL-stGen diferences from the comparator with the highest point estimate for each task. This comparator is selected descriptively from the same results and can difer between rows. The ancillary tables provide point intervals and paired intervals for every comparator, including the format-constrained arm. Fig. 5 visualises the complete primary score matrix without averaging across metrics.

## F Format-constrained generation

Table 13 repeats tasks 6–10 and 13 with output constraints, for all eleven models. Gene-list decoding enforces array syntax while permitting unrestricted gene-symbol strings made of letters, digits, periods and hyphens. It supplies neither a gene-panel whitelist nor the biological answer. Levelsto-cell-type decoding permits the printed, lowercase and uppercase spellings of each label, with case-insensitive scoring. Inputs, token limits, batches and metrics are unchanged. A maximumtoken cutof can still truncate an output, so this arm does not guarantee that every generation is complete JSON.

(a) Classification
<table><tr><td rowspan=1 colspan=1>VGL-stGen</td><td rowspan=1 colspan=1>0.629</td><td rowspan=1 colspan=1>0.539</td><td rowspan=1 colspan=1>0.201</td><td rowspan=1 colspan=1>0.752</td><td rowspan=1 colspan=1>0.120</td><td rowspan=1 colspan=1>0.280</td></tr><tr><td rowspan=2 colspan=1>MedGemma-1.5-4BGemma3-4B</td><td rowspan=1 colspan=1>0.184</td><td rowspan=1 colspan=1>0.207</td><td rowspan=1 colspan=1>0.111</td><td rowspan=1 colspan=1>0.595</td><td rowspan=1 colspan=1>0.091</td><td rowspan=1 colspan=1>0.171</td></tr><tr><td rowspan=1 colspan=1>0.225</td><td rowspan=1 colspan=1>0.185</td><td rowspan=1 colspan=1>0.125</td><td rowspan=1 colspan=1>0.664</td><td rowspan=1 colspan=1>0.133</td><td rowspan=1 colspan=1>0.135</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-8B</td><td rowspan=1 colspan=1>0.466</td><td rowspan=1 colspan=1>0.457</td><td rowspan=1 colspan=1>0.100</td><td rowspan=1 colspan=1>0.518</td><td rowspan=1 colspan=1>0.110</td><td rowspan=1 colspan=1>0.163</td></tr><tr><td rowspan=2 colspan=1>DeepSeek-VL2-smallGLM-4.6V-Flash</td><td rowspan=1 colspan=1>0.131</td><td rowspan=1 colspan=1>0.138</td><td rowspan=1 colspan=1>0.110</td><td rowspan=1 colspan=1>0.535</td><td rowspan=1 colspan=1>0.065</td><td rowspan=1 colspan=1>0.132</td></tr><tr><td rowspan=1 colspan=1>0.344</td><td rowspan=1 colspan=1>0.316</td><td rowspan=1 colspan=1>0.101</td><td rowspan=1 colspan=1>0.581</td><td rowspan=1 colspan=1>0.141</td><td rowspan=1 colspan=1>0.173</td></tr><tr><td rowspan=1 colspan=1>Gemma4-12B</td><td rowspan=1 colspan=1>0.402</td><td rowspan=1 colspan=1>0.407</td><td rowspan=1 colspan=1>0.144</td><td rowspan=1 colspan=1>0.699</td><td rowspan=1 colspan=1>0.165</td><td rowspan=1 colspan=1>0.176</td></tr><tr><td rowspan=2 colspan=1>Qwen3.8-27BGemma4-31B</td><td rowspan=1 colspan=1>0.341</td><td rowspan=1 colspan=1>0.330</td><td rowspan=1 colspan=1>0.127</td><td rowspan=1 colspan=1>0.697</td><td rowspan=1 colspan=1>0.195</td><td rowspan=1 colspan=1>0.261</td></tr><tr><td rowspan=1 colspan=1>0.503</td><td rowspan=1 colspan=1>0.463</td><td rowspan=1 colspan=1>0.165</td><td rowspan=1 colspan=1>0.672</td><td rowspan=1 colspan=1>0.189</td><td rowspan=1 colspan=1>0.196</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-VL2</td><td rowspan=1 colspan=1>0.151</td><td rowspan=1 colspan=1>0.160</td><td rowspan=1 colspan=1>0.110</td><td rowspan=1 colspan=1>0.598</td><td rowspan=1 colspan=1>0.055</td><td rowspan=1 colspan=1>0.156</td></tr><tr><td rowspan=4 colspan=1>Qwen3-VL-32B</td><td rowspan=1 colspan=1>0.482</td><td rowspan=1 colspan=1>0.474</td><td rowspan=1 colspan=1>0.111</td><td rowspan=1 colspan=1>0.674</td><td rowspan=1 colspan=1>0.163</td><td rowspan=1 colspan=1>0.195</td></tr><tr><td rowspan=3 colspan=5>1     BA     3                5BABAF1mraw</td><td rowspan=2 colspan=1>BA</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>F1m</td><td rowspan=1 colspan=1>F1m</td></tr><tr><td rowspan=1 colspan=1>corr.</td></tr></table>

(b) Gene lists and expression

<table><tr><td rowspan=1 colspan=1>VGL-stGen</td><td rowspan=1 colspan=1>0.117</td><td rowspan=1 colspan=1>0.097</td><td rowspan=1 colspan=1>0.060</td><td rowspan=1 colspan=1>0.048</td><td rowspan=1 colspan=1>0.283</td><td rowspan=1 colspan=1>0.660</td><td rowspan=1 colspan=1>0.686</td><td rowspan=1 colspan=1>0.742</td></tr><tr><td rowspan=1 colspan=1>MedGemma-1.5-4B</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.001</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.573</td><td rowspan=1 colspan=1>0.554</td><td rowspan=1 colspan=1>0.000</td></tr><tr><td rowspan=1 colspan=1>Gemma3-4B</td><td rowspan=1 colspan=1>0.009</td><td rowspan=1 colspan=1>0.035</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.006</td><td rowspan=1 colspan=1>0.507</td><td rowspan=1 colspan=1>0.500</td><td rowspan=1 colspan=1>0.350</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-8B</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.086</td><td rowspan=1 colspan=1>0.021</td><td rowspan=1 colspan=1>0.038</td><td rowspan=1 colspan=1>0.003</td><td rowspan=1 colspan=1>0.163</td><td rowspan=1 colspan=1>0.237</td><td rowspan=1 colspan=1>0.350</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-VL2-small</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.008</td><td rowspan=1 colspan=1>0.003</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.004</td><td rowspan=1 colspan=1>0.160</td><td rowspan=1 colspan=1>0.216</td><td rowspan=1 colspan=1>0.063</td></tr><tr><td rowspan=1 colspan=1>GLM-4.6V-Flash</td><td rowspan=1 colspan=1>0.004</td><td rowspan=1 colspan=1>0.040</td><td rowspan=1 colspan=1>0.014</td><td rowspan=1 colspan=1>0.024</td><td rowspan=1 colspan=1>0.003</td><td rowspan=1 colspan=1>0.366</td><td rowspan=1 colspan=1>0.363</td><td rowspan=1 colspan=1>0.426</td></tr><tr><td rowspan=1 colspan=1>Gemma4-12B</td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1>0.079</td><td rowspan=1 colspan=1>0.014</td><td rowspan=1 colspan=1>0.046</td><td rowspan=1 colspan=1>0.007</td><td rowspan=1 colspan=1>0.269</td><td rowspan=1 colspan=1>0.208</td><td rowspan=1 colspan=1>0.396</td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-27B</td><td rowspan=1 colspan=1>0.027</td><td rowspan=1 colspan=1>0.070</td><td rowspan=1 colspan=1>0.021</td><td rowspan=1 colspan=1>0.018</td><td rowspan=1 colspan=1>0.013</td><td rowspan=1 colspan=1>0.613</td><td rowspan=1 colspan=1>0.562</td><td rowspan=1 colspan=1>0.456</td></tr><tr><td rowspan=2 colspan=1>Gemma4-31BDeepSeek-VL2</td><td rowspan=1 colspan=1>0.056</td><td rowspan=1 colspan=1>0.100</td><td rowspan=1 colspan=1>0.038</td><td rowspan=1 colspan=1>0.050</td><td rowspan=1 colspan=1>0.023</td><td rowspan=1 colspan=1>0.493</td><td rowspan=1 colspan=1>0.511</td><td rowspan=1 colspan=1>0.462</td></tr><tr><td rowspan=1 colspan=1>0.001</td><td rowspan=1 colspan=1>0.028</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.104</td><td rowspan=1 colspan=1>0.113</td><td rowspan=1 colspan=1>0.295</td></tr><tr><td rowspan=3 colspan=1>Qwen3-VL-32B</td><td rowspan=1 colspan=1>0.020</td><td rowspan=1 colspan=1>0.084</td><td rowspan=1 colspan=1>0.002</td><td rowspan=1 colspan=1>0.045</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1>0.431</td><td rowspan=1 colspan=1>0.379</td><td rowspan=1 colspan=1>0.402</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=2 colspan=3>8   9   10F1g  F1g F1g</td><td rowspan=1 colspan=1>11</td><td rowspan=2 colspan=2>12  18F1p</td></tr><tr><td rowspan=1 colspan=1>F1g</td><td rowspan=1 colspan=1>F1g</td><td rowspan=1 colspan=1>F1p</td></tr></table>

Figure 5: Comparison with ten unadapted vision-language models: (a) classification; (b) gene and expression tasks. Printed values are original scores; colors are column-wise z-scores across 11 models, using population standard deviation. T5 raw and pooled-corrected columns are standardized separately. BA: balanced accuracy; F1m/F1g/F1p: macro/gene-set/pair-F1; Acc: accuracy. Task key: T1–3, cell typing from genes, image + genes, or image, respectively; T4, image → binary tumor status; T5, image → nine tissue classes; T6–7, gene names from image or cell type; T8–9, missing genes from partial lists plus image or cell type; T10, image → Very High genes; T11–12, queried levels from observed levels plus image or cell type; T13, gene–level pairs → cell type. For paired task IDs, inputs follow numerical order.

Table 12: Paired VGL-stGen minus highest-scoring comparator diferences, with unadjusted 95% bootstrap intervals (1,000 draws). T5 uses corrected F1m. Units: cells/spots (T1–3), slides (T4), patches (T5), complete rows (T6–13). Diferences use unrounded underlying scores; T4 uses observed BA rather than the displayed bootstrap mean. Comparator selection is descriptive and not adjusted for selection or multiple comparisons. Intervals condition on fitted models, not independent-patient generalization. Task key: T1, genes → cell type; T2, image + genes → cell type; T3, image → cell type; T4, image → binary tumor status; T5, image → nine-class tissue type; T6, image → gene names; T7, cell type → gene names; T8, image + partial gene list → missing gene names; T9, cell type + partial gene list → missing gene names; T10, image → very-high-expression gene names; T11, image + observed levels → queried levels; T12, cell type + observed levels → queried levels; T13, gene–level pairs → cell type.
<table><tr><td>Task</td><td>Comparator</td><td>Difference</td><td>95% interval</td></tr><tr><td>1</td><td>Gemma4-31B</td><td>+0.1258</td><td>[+0.1204, +0.1309]</td></tr><tr><td>2</td><td>Qwen3-VL-32B</td><td>+0.0654</td><td>[+0.0598, +0.0709]</td></tr><tr><td>3</td><td>Gemma4-31B</td><td>+0.0360</td><td>[+0.0332, +0.0386]</td></tr><tr><td>4</td><td>Gemma4-12B</td><td>+0.0538</td><td>[+0.0491, +0.0587]</td></tr><tr><td>5</td><td>Qwen3.8-27B</td><td>+0.0188</td><td>[+0.0133, +0.0238]</td></tr><tr><td>6</td><td>Gemma4-31B</td><td>+0.0617</td><td>[+0.0568, +0.0665]</td></tr><tr><td>7</td><td>Gemma4-31B</td><td>-0.0037</td><td>[-0.0076, +0.0006]</td></tr><tr><td>8</td><td>Gemma4-31B</td><td>+0.0222</td><td>[+0.0154, +0.0292]</td></tr><tr><td>9</td><td>Gemma4-31B</td><td>-0.0018</td><td>[-0.0090, +0.0056]</td></tr><tr><td>10</td><td>Gemma4-31B</td><td>+0.2599</td><td>[+0.2456, +0.2741]</td></tr><tr><td>11</td><td>Qwen3.8-27B</td><td>+0.0466</td><td>[+0.0417, +0.0515]</td></tr><tr><td>12</td><td>Qwen3.8-27B</td><td>+0.1236</td><td>[+0.1164, +0.1308]</td></tr><tr><td>13</td><td>Gemma4-31B</td><td>+0.2800</td><td>[+0.2490, +0.3130]</td></tr></table>

Table 13: Format-constrained generation on the same 1,000 rows per task, with unchanged NF4 precision, inputs and token limits. T6–10 report gene-set F1 under a gene-array grammar without a gene-panel whitelist; T13 reports accuracy under allowed-label spellings. Fixed token limits can still truncate outputs. Values preserve the original aggregation and rounding. Task key: T6, image → gene names; T7, cell type → gene names; T8, image + partial gene list → missing gene names; T9, cell type + partial gene list → missing gene names; T10, image → very-high-expression gene names; T13, gene–level pairs → cell type.
<table><tr><td>Model</td><td>T6</td><td>T7</td><td>T8</td><td>T9</td><td>T10</td><td>T13</td></tr><tr><td>VGL-stGen</td><td>0.1173</td><td>0.0965</td><td>0.0604</td><td>0.0484</td><td>0.2828</td><td>0.7420</td></tr><tr><td>MedGemma-1.5-4B</td><td>0.0005</td><td>0.0140</td><td>0.0065</td><td>0.0141</td><td>0.0011</td><td>0.3620</td></tr><tr><td>Gemma3-4B</td><td>0.0186</td><td>0.0625</td><td>0.0088</td><td>0.0180</td><td>0.0057</td><td>0.3530</td></tr><tr><td>Qwen3-VL-8B</td><td>0.0024</td><td>0.0850</td><td>0.0210</td><td>0.0380</td><td>0.0032</td><td>0.3500</td></tr><tr><td>DeepSeek-VL2-small</td><td>0.0000</td><td>0.0103</td><td>0.0020</td><td>0.0028</td><td>0.0043</td><td>0.0630</td></tr><tr><td>GLM-4.6V-Flash</td><td>0.0082</td><td>0.0501</td><td>0.0135</td><td>0.0262</td><td>0.0040</td><td>0.4220</td></tr><tr><td>Gemma4-12B</td><td>0.0497</td><td>0.0810</td><td>0.0300</td><td>0.0453</td><td>0.0244</td><td>0.3980</td></tr><tr><td>Qwen3.8-27B</td><td>0.0237</td><td>0.0699</td><td>0.0198</td><td>0.0456</td><td>0.0118</td><td>0.4560</td></tr><tr><td>Gemma4-31B</td><td>0.0611</td><td>0.0995</td><td>0.0392</td><td>0.0508</td><td>0.0230</td><td>0.4720</td></tr><tr><td>DeepSeek-VL2</td><td>0.0005</td><td>0.0281</td><td>0.0000</td><td>0.0131</td><td>0.0000</td><td>0.2950</td></tr><tr><td>Qwen3-VL-32B</td><td>0.0197</td><td>0.0839</td><td>0.0022</td><td>0.0452</td><td>0.0047</td><td>0.4020</td></tr></table>

VGL-stGen retains its reported scores. Several comparators improve, including MedGemma’s levels-to-cell-type accuracy from 0.0000 to 0.3620 and Gemma4-31B’s from 0.4620 to 0.4720. VGLstGen remains at 0.7420; its paired diference from Gemma4-31B is 0.2700, with 95% interval [0.2380, 0.3020]. The two cell-type-conditioned gene-list tasks remain close to Gemma4-31B. Thus, enforcing output format does not remove the principal image-gene and cell-typing gaps, but the comparison remains conditional on the fixed generation budgets.

Inspection of low-scoring MedGemma outputs shows that intermediate reasoning often consumes the available generation budget. Parsed free-list outputs can also name genes outside the evaluation panel or use aliases not mapped by the scorer. Out-of-panel genes are not necessarily fictitious. These observations explain why a near-zero score under this protocol should not be read as absence of biological knowledge. Under format constraints, MedGemma predicts T cells in 999 of 1,000 levels-to-cell-type rows, yielding the majority-label score. A supplementary diagnostic on the first 100 rows uses full-label likelihood scoring, rather than the primary decoder: removing the molecular input preserves a constant prediction, whereas reversing candidate order changes that prediction. This diagnostic identifies sensitivity to the readout and candidate ordering without establishing a biological mechanism for the bias.

## G Supporting analyses of learning and inputs

## G.1 Expression-adaptation design

Fig. 6 contrasts full-answer and level-only supervision. The optimisation settings, development selection and image-permutation protocol are specified in Detailed methods under Controlled expression adaptation and image permutation. The comparison holds inputs and autoregressive evaluation fixed.

![](images/b4bfad16120d9bf6784bc09efbad0e9eb8ecf330b73aa4a53613c6720903780d.jpg)  
Figure 6: Supervised answer positions in the separate expression-adaptation study. Blue groups receive full-vocabulary cross-entropy. Full-answer supervision includes all answer tokens and end-ofturn; level-only supervision selects each queried level’s first token. Level rest denotes any remaining level tokens. Initial weights, inputs, update budget and autoregressive evaluation are matched. These checkpoints are separate from the deployed 13-task model.

## G.2 Full-input comparators and uncertainty

Comparator architectures and fitting procedures are specified in Detailed methods under Contextaware expression baselines, Gene classifiers and adaptation-capacity control, and Image encoders and within-cohort reference fits. Bootstrap procedures are specified under Statistical analysis.

## G.3 Gene-mediated image supervision

Image-to-cell-type balanced accuracy increases from 0.161 to 0.226 after self-distillation and is 0.2012 after cancer training and averaging. On the probes’ common 54,058-item subset, deployed

VGL-stGen scores 0.1998 and transferred probes reach at most 0.195; target-cohort probe fits span 0.27–0.35. Joint image-and-gene prompting scores 0.5394, below genes alone at 0.6297 in the same evaluation. The joint task received no direct supervision.

## G.4 Conditional expression completion and adaptation

Fixing query names separates output formatting from level prediction. The parser identifies all masked names across 176,008 held-out rows. On 1,000 rows per task, free-generation pair-F1 is 0.0361 (image) and 0.0123 (cell type); fixed-name autoregressive decoding gives 0.6596 and 0.6858 with unchanged weights. The image task exceeds its per-gene frequency table, 0.6198, by +0.0398 [0.0350, 0.0445]. The cell-type task trails its conditional table, 0.6963, by −0.0106 [−0.0160, −0.0052]. These tables omit visible levels. At 75 observed levels, full-input MLP references score higher on both tasks (Table 14).

Table 14: Expression pair-F1 on the same 1,000 rows per task. Additional training and MLPs: mean ± SD of 3 individual seeds, not ensembles. Each VGL-stGen adaptation uses 4,096 additional rows and 256 updates. MLPs use original task training splits; the image MLP uses UNI2-h, so encoders and training histories difer.
<table><tr><td>Model / supervision</td><td>Image</td><td>Cell type</td></tr><tr><td>Deployed checkpoint</td><td>0.6596</td><td>0.6858</td></tr><tr><td>Full-answer adaptation</td><td> $0 . 6 7 2 0 \pm 0 . 0 0 0 9$ </td><td> $0 . 7 2 2 1 \pm 0 . 0 0 0 7$ </td></tr><tr><td>Level-only adaptation</td><td> $0 . 6 8 0 5 \pm 0 . 0 0 1 1$ </td><td> $0 . 7 2 0 1 \pm 0 . 0 0 0 4$ </td></tr><tr><td>Full-input MLP</td><td> $0 . 6 8 1 1 \pm 0 . 0 0 3 8$ </td><td> $0 . 7 3 0 0 \pm 0 . 0 0 0 8$ </td></tr></table>

With decoding held fixed, both additional-training conditions improve both tasks in all three seeds. Level-only exceeds full-answer supervision on the image task by +0.00848, sample CI [0.00561, 0.01015]; the direction is positive in every seed on both platforms (Fig. 7). Its cell-type diference is −0.00200, row CI [−0.00507, 0.00113].

Absent targets comprise 61.36% (image) and 58.92% (cell type). Mean non-Absent accuracy rises from full-answer to level-only supervision: 0.2664 to 0.2930 (image) and 0.4433 to 0.4535 (cell type). MLPs remain higher at 0.3160 and 0.4786. Thus, the cell-type trade-of depends on the metric.

Across three training seeds, intact-minus-permuted pair-F1 is +0.00597 [0.00316, 0.00764] for full-answer and +0.00776 [0.00263, 0.01042] for level-only supervision (sample CIs). On the pooled image benchmark, per-seed diferences are positive in both conditions. The between-objective difference is +0.00179 [−0.00067, 0.00336], leaving an increase in the image-pairing efect unresolved.

For the fixed level-only seed-42 checkpoint, reducing observed levels from 75 to 25 lowers image pair-F1 from 0.68032 to 0.65992. The change in image-pairing efect remains unresolved. The extended methods report all budgets and secondary metrics under Completion with fewer observed expression levels.

## G.5 Matched gene-readout control

Table 15 gives the matched gene-task comparison and trainable parameter counts.

On the gene task, short runs with matched data, adaptation rank and update budget give macro-F1 0.5683 for the generative readout and 0.5209 for the discriminative readout, a diference of 0.0474. Both the output head and loss difer; these single runs are separate from deployment.

![](images/e3c0fac9e960c1613e9b31fbebf0af6d37a0804c73a4467a89e082341154162e.jpg)

![](images/1f2784bf5b04bcc453fcd6f7ad4f55abae75119a3468c552f8b847cea1787ebd.jpg)  
Figure 7: Controlled expression prediction after additional adaptation. (a) Level-only minus fullanswer pair-F1 with intact inputs. (b) Intact-minus-permuted image efects; Diference denotes the level-only efect minus the full-answer efect. Points average three matched seeds. Conditional 95% bootstrap intervals resample image samples or cell-type rows and condition on fitted models. Panel b resamples samples. Image intervals preserve platform proportions. The 25 queries and withinsample/platform permutation are fixed. Sample identifiers are not verified patients.

Table 15: Matched adaptation-capacity control on the gene task. These short runs are separate from deployment. Data, rank, learning rate, schedule and update budget are matched; output heads and losses difer.
<table><tr><td>Readout</td><td>Balanced accuracy</td><td>Macro-F1</td><td>Trainable parameters</td></tr><tr><td>Discriminative short run</td><td>0.5554</td><td>0.5209</td><td>32,819,212</td></tr><tr><td>Generative short run</td><td>0.5737</td><td>0.5683</td><td>32,788,480</td></tr></table>

## G.6 Transfer and task retention

External gene-list tasks (Table 16) exceed transferred global-frequency F1 (0.0367 for generation; 0.0098 for completion) but trail target-cohort cell-type-conditional F1 (0.4009 and 0.2300, respectively). Training and external gene-frequency distributions difer; their top-twenty lists share five members.

From the pre-cancer checkpoint to deployment, genes-to-cell-type changes by −0.0079 macro-F1 with section CI [−0.0207, +0.0234], and image-to-cell-type changes by −0.0248 balanced accuracy. These changes cover cancer training, replay and averaging.

## G.7 Reference fits and label definitions

Distribution-shift studies motivate comparing transferred performance with within-cohort fits [46– 49]. In this study, the range of transferred scores is 2.09 times the range of target-cohort scores across nine gene comparators and 2.17 times that range across fourteen cancer encoders. Targetcohort fits use additional labels, with cell-level folds for genes and patient-level folds for cancer. These ratios describe the selected comparator panels and evaluation designs.

On the same 452 slides, refitting each cancer probe under the wide (TUMOR, NECROSIS and MIXED counted as positive) and TUMOR-only mappings yields a median paired diference of 0.1191 balanced accuracy, compared with 0.0436 for the colour reference. Each encoder’s patient-paired interval excludes zero. Target definition, prevalence and probe fitting change together. Because the mapping was chosen during development, this is an exploratory measure of benchmark sensitivity.

Patient-level resampling of the binary VGL-stGen result gives an interval of [0.7441, 0.7622], compared with the slide interval [0.7452, 0.7610]. This sensitivity analysis accounts for repeated slides from a patient. Annotation error and adaptive development remain outside both intervals. Additional comparisons and implementation details appear in the supplementary methods and figures.

## H Region-query application example

VGL-stGen outputs can be organised for region-level tissue inspection and spatial molecular analysis. Fig. 8 reproduces original application examples of a region-query workflow: a user draws a box on a slide and asks, "score this region for cancer and show me the map." The surrounding application selects tissue, runs image-conditioned predictions and displays the scores alongside H&E.

## I Additional comparison panels

Fig. 9 maps the served tasks to their output types and recorded evaluation conditions. Fig. 10 locates the reported checkpoints in the training history.

Fig. 11 reports both gene-classification metrics and section-paired diferences. Fig. 12 gives the matched adaptation-capacity comparison. Fig. 13 contrasts transferred and target-cohort probe scores on separate axes. Fig. 14 shows the gene-budget dependence of a logistic reference.

Fig. 15 gives the cancer comparison on 452 slides. Fig. 16 compares it with the earlier 39-slide subset, and Fig. 17 shows sensitivity to cancer-label definitions. Fig. 18 compares training and external gene frequencies.

![](images/f95a2c4049b03bb61059c2a0f428d4a56cf9a477bdecbf2b901a79a32df32b4e.jpg)  
Figure 8: Region-query cancer mapping. H&E tissue (left) and the corresponding model cancerscore map (right) are shown for (a) TCGA-73-7499-01Z-00-DX1 and (b) TCGA-55-6979-01Z-00- DX1. These application examples illustrate selecting tissue and requesting a cancer-score map. Blue and red indicate lower and higher model scores, respectively. Scores are model outputs, not calibrated clinical probabilities.

## Thirteen tasks from one deployed checkpoint

Each numbered cell is a supported input → output task.

![](images/536fc72f92c17f0af70d514531a6c73587cae31f647a3de7b8e8cd5d8b89b2fc.jpg)  
Inputs: 20 ranked genes (1–2); 15 observed + 5 withheld (8–9).  
75 levels + 25 named queries (11–12); 10 gene–level pairs (13).  
Task 10 supplies the true list length. \* No direct training objective.  
External describes data origin; exposure conditions are in the caption.

Figure 9: Input–output map of the 13 deployed tasks. Numbers match the task inventory; the asterisk marks task 2 without a direct training objective. Filled cells use external datasets; outlined cells use internal evaluation data. External refers to data origin: 3 spatial sources supplied cancertraining H&E, task 4 includes TCGA development slides, and task 5 uses the development pool. Classification references are computationally derived. Gene-list completion supplies 15 names and predicts five; expression completion supplies 75 levels and queries 25 named genes. Task 10 supplies true answer length; task 13 supplies ten gene–level pairs. All tasks share the same checkpoint.

![](images/79c42a8de2320063ae7ca7ccf262f8916791a003acc5cbf7da66317a3667447a.jpg)  
Figure 10: Training stages and evaluated checkpoints. Gene and cell-type adaptation precedes genemediated image supervision, joint cancer training with replay, and averaging. Stepwise prompts retain the original targets. The vision tower remains frozen. The pre-cancer checkpoint follows image supervision; deployment follows averaging. The averaging window was selected using the 39-slide TCGA development subset.

(a)  
![](images/5683772e90e61ce8d1410269180a8fdbb22e8b1cbe4f0f81b466ed193c278a48.jpg)

![](images/4d9290cd73fb7e9e316cac44a691c82fc75d5a7313284ebe70ea88669a12ebc4.jpg)  
Figure 11: (a) Gene classifiers and VGL-stGen checkpoints on 224,210 items. The pre-cancer point uses its original evaluation (macro-F1 0.6104). FT denotes fine-tuning. Macro-F1 was not scored for the frozen-representation MedGemma probe in this panel; that probe is separate from the direct multimodal-model benchmark. (b) Paired macro-F1 diferences with section-bootstrap intervals. The pre-cancer score in the paired comparison is 0.6102, versus 0.6023 after joint training and averaging. The MLP comparison is provisional because of discrepancies between refits.

![](images/a0efac6fcc186d253dd2e42cb8b932dec710b794938d3b3501188cbf0099a4da.jpg)  
Figure 12: Generative and discriminative readouts for gene-based cell typing under matched adaptation settings: rank, learning rate, schedule, batch, 6,000 updates, seed, prompts and data (40,000 evaluation items). Output heads and losses difer. Parameter counts are reported in the detailed Methods. Points are single-run estimates, separate from the deployed checkpoint.

![](images/a94a2dd217885a3ffa779fd357b5fbee777df0072d899c30bf975041fd309fc8.jpg)  
Model configurations: 1 Pre-cancer VGL-stGen 2 Deployed VGL-stGen 3 Linear probe (same frozen tower)

Figure 13: Transferred versus target-cohort probe scores for (a) gene, (b) cancer and (c) morphology classification. Both axes report balanced accuracy; the diagonal denotes equal scores. Target-cohort fits use cell-level folds for genes and morphology and patient-level folds for cancer. The numbered model configurations are 1, pre-cancer VGL-stGen; 2, deployed VGL-stGen; 3, a task-specific linear probe on the same frozen vision tower. Panel (b) shows 2 and 3; panel (c) shows all three. All morphology comparisons use the same 54,058 evaluation items. VGL-stGen imaging coordinates use its own-tower probe reference, not a fitted VGL-stGen upper bound. Classification references are silver; spatial H&E exposure is partial and TCGA includes development slides.

![](images/24a26f058c045393c1a40664fc082ba0650724671151eb2410e69c1a2b19bacf.jpg)  
Figure 14: Gene-budget analysis for logistic regression: target-cohort cross-validation and transferred balanced accuracy as the input list is shortened. Shading denotes fold standard deviation. Percentages refer to the target-cohort score’s margin above chance at 20 genes.

## H&E → tumour

14 encoder–probe comparators; colour reference unranked

![](images/98e90f8d29cfafe0b04427340275554042b09063eb426c168963a76f8a6dbf50.jpg)  
Figure 15: TCGA-LUAD tumor prediction: VGL-stGen and 14 encoder–probe systems, including the ImageNet ResNet50 reference. The color-histogram reference is unranked. Scores measure agreement with ARA-CNN-derived tumor labels; intervals resample slides. The shaded band spans encoder point estimates, not confidence bounds. The 452-slide set includes the 39-slide development subset; rank is descriptive.

![](images/fcb6f2e39680809288dfb4dd0a964ae6039d34fe81cd8cf917c8ec1da306f429.jpg)  
Figure 16: Cancer scores on the earlier 39-slide subset and the expanded 452-slide evaluation. 13 of 14 encoders and VGL-stGen have lower point estimates on the expanded set. Slide-bootstrap intervals narrow with the larger sample; the two sets overlap and are not independent replications.

![](images/8b44d3c11ec64f24976d9d24c4b7105428094e991c3d1ea907a4900c23e2f3b0.jpg)  
Figure 17: Cancer-label sensitivity on the same 452 slides. Each encoder probe is refitted under the wide mapping (TUMOR, NECROSIS and MIXED positive) and the TUMOR-only mapping. Grey and blue indicate the respective mappings; orange marks the colour reference under TUMOR-only. Targets, prevalence and probe fits change together.

![](images/7a6c3964d80060764a4d47b90a12d5a1f2b29ea4559e582099389569bdb0b613.jpg)  
Figure 18: Training and external gene frequencies across 1,049 shared symbols. Blue marks 1,014 genes in neither top-20 list; orange marks their 35-gene union. The dashed line denotes equal frequency. Spearman rho is 0.63. The five genes in both top-20 lists are ACTA2, EPCAM, ICAM1, MET and S100A9. Frequencies count cells or spots rather than task rows. Zero frequencies are placed at half the smallest positive observed frequency on the log axes.

## J Deployed task inventory

Table 16 lists the served tasks and their evaluation conditions.

Table 16: Deployed-checkpoint task inventory. BA: balanced accuracy; F1m: macro-F1; F1g: geneset F1; F1p: pair-F1; Acc: accuracy; AR: autoregressive. H&E overlap denotes three of four spatial sources used in cancer training; SCC is absent from that manifest. Task 4 includes development slides; task 5 pools NF4-adapter scores from 30,719 patches / 39 slides for transductive correction, selected after labelled evaluation. Task 10 supplies the true answer length. Tasks 10/13 use baselines estimated from evaluation data; task 13 uses ten sampled gene–level pairs. Gene-list completion supplies 15 names and predicts five. Additional-training results are separate.
<table><tr><td>ID</td><td>Task</td><td>Score</td><td>Metric</td><td>Condition</td></tr><tr><td>1</td><td>Genes → cell type</td><td>0.6293 0.6023</td><td>BA F1m</td><td>External</td></tr><tr><td>2</td><td>Image + genes → cell type</td><td>0.5394</td><td>BA</td><td>H&amp;E overlap</td></tr><tr><td>3</td><td>Image → cell type</td><td>0.2012</td><td>BA</td><td>H&amp;E overlap</td></tr><tr><td>4</td><td>Image → tumour</td><td>0.7532</td><td>BA</td><td>452 slides</td></tr><tr><td>5</td><td>Image → tissue</td><td>0.1200 → 0.2801</td><td>F1m</td><td>39-slide pool</td></tr><tr><td>6</td><td>Image → genes</td><td>0.1173</td><td>F1g</td><td>H&amp;E overlap</td></tr><tr><td>7</td><td>Cell type → genes</td><td>0.0965</td><td>F1g</td><td>External</td></tr><tr><td>8</td><td>Image + partial → genes</td><td>0.0604</td><td>F1g</td><td>H&amp;E overlap</td></tr><tr><td>9</td><td>Cell type + partial → genes</td><td>0.0484</td><td>F1g</td><td>External</td></tr><tr><td>10</td><td>Image → very-high genes</td><td>0.2828</td><td>F1g</td><td>Internal; true length</td></tr><tr><td>11</td><td>Image + levels → levels</td><td>0.6596</td><td>F1p</td><td>Internal AR</td></tr><tr><td>12</td><td>Cell type + levels → levels</td><td>0.6858</td><td>F1p</td><td>Internal AR</td></tr><tr><td>13</td><td>Levels → cell type</td><td>0.7420</td><td>Acc</td><td>Internal</td></tr></table>

## References

[1] He, B., Bergenstråhle, L., Stenbeck, L., et al. (2020). Integrating spatial gene expression and breast tumour morphology via deep learning. Nature Biomedical Engineering 4, 827-834. doi:10.1038/s41551-020-0578-x

[2] Xie, R., Pang, K., Chung, S., et al. (2023). Spatially Resolved Gene Expression Prediction from Histology Images via Bi-modal Contrastive Learning. Advances in Neural Information Processing Systems 36, 70626-70637. doi:10.52202/075280-3095

[3] Ouyang, J., Wang, Y., Gao, Y., et al. (2025). GenAR: Next-Scale Autoregressive Generation for Spatial Gene Expression Prediction. arXiv. arXiv:2510.04315v1 Preprint, version 1.

[4] Ruiz, D., Cárdenas, P., Manrique, L., et al. (2025). Completing spatial transcriptomics data for gene expression prediction benchmarking. Medical Image Analysis 106, 103754. doi:10.1016/j.media.2025.103754

[5] Chen, W., Zhang, P., Tran, T. N., et al. (2025). A visual–omics foundation model to bridge histopathology with spatial transcriptomics. Nature Methods 22, 1568-1582. doi:10.1038/s41592-025-02707-1

[6] Redekop, E., Pleasure, M., Wang, Z., et al. (2026). SPADE: Spatial transcriptomics and pathology alignment using a mixture of data experts for an expressive latent space. Medical Image Analysis 113, 104197. doi:10.1016/j.media.2026.104197

[7] Huang, T., Liu, T., Babadi, M., et al. (2025). STPath: a generative foundation model for integrating spatial transcriptomics and whole-slide images. npj Digital Medicine 8, 659. doi:10.1038/s41746-025-02020-3

[8] Fu, X., Cao, Y., Bian, B., et al. (2025). Spatial gene expression at single-cell resolution from histology using deep learning with GHIST. Nature Methods 22, 1900-1910. doi:10.1038/s41592- 025-02795-z

[9] Sellergren, A., Gao, C., Mahvar, F., et al. (2026). MedGemma 1.5 Technical Report. arXiv. arXiv:2604.05081v2 Preprint, version 2.

[10] Hu, E. J., Shen, Y., Wallis, P., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv. arXiv:2106.09685 Preprint.

[11] Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). QLoRA: Eficient Finetuning of Quantized LLMs. Advances in Neural Information Processing Systems 36, 10088-10115. doi:10.52202/075280-0441

[12] Izmailov, P., Podoprikhin, D., Garipov, T., et al. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. Proceedings of the Conference on Uncertainty in Artificial Intelligence. arXiv:1803.05407

[13] Rączkowska, A., Paśnik, I., Kukiełka, M., et al. (2022). Deep learning-based tumor microenvironment segmentation is predictive of tumor mutations and patient survival in non-small-cell lung cancer. BMC Cancer 22, 1001. doi:10.1186/s12885-022-10081-w

[14] Mages, S., Moriel, N., Avraham-Davidi, I., et al. (2023). TACCO unifies annotation transfer and decomposition of cell identities for single-cell and spatial omics. Nature Biotechnology 41, 1465-1473. doi:10.1038/s41587-023-01657-3

[15] Salcher, S., Sturm, G., Horvath, L., et al. (2022). High-resolution single-cell atlas reveals diversity and plasticity of tissue-resident neutrophils in non-small cell lung cancer. Cancer Cell 40, 1503-1520.e8. doi:10.1016/j.ccell.2022.10.008

[16] 10x Genomics (n.d.). Public lung cancer spatial datasets: Visium CytAssist and Xenium. Dataset collection. https://www.10xgenomics.com/datasets/preview-data-ffpe-human -lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard 10x Xenium Human Lung Cancer FFPE; all four provider links are in the accompanying data-source table; accessed 2026-09-13.

[17] The Cancer Genome Atlas Research Network (2014). Comprehensive molecular profiling of lung adenocarcinoma. Nature 511, 543-550. doi:10.1038/nature13385

[18] Gemma Team (2025). Gemma 3 Technical Report. arXiv. arXiv:2503.19786 Preprint.

[19] Gemma Team (2026). Gemma 4 Technical Report. arXiv. arXiv:2607.02770 Preprint.

[20] Bai, S., Cai, Y., Chen, R., et al. (2025). Qwen3-VL Technical Report. arXiv. arXiv:2511.21631 Preprint.

[21] Qwen Team (2026). Qwen3.8-27B. Hugging Face model repository. https://huggingface.co /Qwen/Qwen3.8-27B Model card; accessed 2026-10-06.

[22] Wu, Z., Chen, X., Pan, Z., et al. (2024). DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding. arXiv. arXiv:2412.10302 Preprint.

[23] zai-org (2025). GLM-4.6V-Flash. Hugging Face model repository. https://huggingface.co/z ai-org/GLM-4.6V-Flash Model card; accessed 2026-10-06.

[24] Theodoris, C. V., Xiao, L., Chopra, A., et al. (2023). Transfer learning enables predictions in network biology. Nature 618, 616-624. doi:10.1038/s41586-023-06139-9

[25] Chen, H., Venkatesh, M. S., Gómez Ortega, J., et al. (2026). Scaling and quantization of large-scale foundation model enables resource-eficient predictions in network biology. Nature Computational Science 6, 450-463. doi:10.1038/s43588-026-00972-4 Model releases: https: //huggingface.co/ctheodoris/Geneformer.

[26] Levine, D., Rizvi, S. A., Lévy, S., et al. (2024). Cell2Sentence: Teaching Large Language Models the Language of Biology. International Conference on Machine Learning 235, 27299- 27325. https://proceedings.mlr.press/v235/levine24a.html

[27] Rizvi, S. A., Levine, D., Patel, A., et al. (2025). Scaling Large Language Models for Next-Generation Single-Cell Analysis. bioRxiv. doi:10.1101/2025.04.14.648850 Preprint.

[28] Biderman, S., Schoelkopf, H., Anthony, Q. G., et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. International Conference on Machine Learning 202, 2397-2430. arXiv:2304.01373

[29] Chen, R. J., Ding, T., Lu, M. Y., et al. (2024). Towards a general-purpose foundation model for computational pathology. Nature Medicine 30, 850-862. doi:10.1038/s41591-024-02857-3

[30] Lu, M. Y., Chen, B., Williamson, D. F. K., et al. (2024). A visual-language foundation model for computational pathology. Nature Medicine 30, 863-874. doi:10.1038/s41591-024-02856-4

[31] Xu, H., Usuyama, N., Bagga, J., et al. (2024). A whole-slide foundation model for digital pathology from real-world data. Nature 630, 181-188. doi:10.1038/s41586-024-07441-w

[32] Zimmermann, E., Vorontsov, E., Viret, J., et al. (2024). Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology. arXiv. arXiv:2408.00738 Preprint.

[33] Filiot, A., Ghermi, R., Olivier, A., et al. (2023). Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling. medRxiv. doi:10.1101/2023.07.21.23292757 Preprint.

[34] Wang, X., Yang, S., Zhang, J., et al. (2022). Transformer-based unsupervised contrastive learning for histopathological image classification. Medical Image Analysis 81, 102559. doi:10.1016/j.media.2022.102559

[35] Zhou, X., Sun, L., He, D., et al. (2024). Knowledge-enhanced Pretraining for Vision-language Pathology Foundation Model on Cancer Diagnosis. arXiv. arXiv:2412.13126 Preprint.

[36] Nechaev, D., Pchelnikov, A., Ivanova, E. (2024). Hibou: A Family of Foundational Vision Transformers for Pathology. arXiv. arXiv:2406.05074 Preprint.

[37] bioptimus (n.d.). H-optimus-0. Hugging Face model repositories. https://huggingface.co/b ioptimus/H-optimus-0 Model release; accessed 2026-09-11. bioptimus/H-optimus-0.

[38] kaiko-ai (n.d.). midnight. Hugging Face model repositories. https://huggingface.co/kaiko -ai/midnight Model release; accessed 2026-09-11. kaiko-ai/midnight.

[39] owkin (n.d.). phikon-v2. Hugging Face model repositories. https://huggingface.co/owkin/p hikon-v2 Model release; accessed 2026-09-11. owkin/phikon-v2.

[40] MahmoodLab (n.d.). UNI2-h and conchv1\_5. Hugging Face model repositories. https://hugg ingface.co/MahmoodLab/UNI2-h Model release; accessed 2026-09-11. MahmoodLab/UNI2-h; MahmoodLab/conchv1\_5.

[41] He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778. doi:10.1109/cvpr.2016.90

[42] Zilionis, R., Engblom, C., Pfirschke, C., et al. (2019). Single-Cell Transcriptomics of Human and Mouse Lung Cancers Reveals Conserved Myeloid Populations across Individuals and Species. Immunity 50, 1317-1334.e10. doi:10.1016/j.immuni.2019.03.009

[43] Shen, Y., Luo, Y., Shen, D., Ke, J. (2022). RandStainNA: Learning Stain-Agnostic Features from Histology Slides by Bridging Stain Augmentation and Normalization. Medical Image Computing and Computer Assisted Intervention, 212-221. doi:10.1007/978-3-031-16434-7\_21

[44] Lin, T. Y., Goyal, P., Girshick, R., et al. (2017). Focal Loss for Dense Object Detection. 2017 IEEE International Conference on Computer Vision (ICCV), 2999-3007. doi:10.1109/iccv.2017.324

[45] Cui, H., Wang, C., Maan, H., et al. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. Nature Methods 21, 1470-1480. doi:10.1038/s41592- 024-02201-0

[46] Miller, J. P., Taori, R., Raghunathan, A., et al. (2021). Accuracy on the Line: on the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization. International Conference on Machine Learning 139, 7721-7735. https://proceedings.mlr.press/v1 39/miller21b.html

[47] Taori, R., Dave, A., Shankar, V., et al. (2020). Measuring Robustness to Natural Distribution Shifts in Image Classification. Advances in Neural Information Processing Systems 33, 18583- 18599. arXiv:2007.00644

[48] Koh, P. W., Sagawa, S., Marklund, H., et al. (2021). WILDS: A Benchmark of in-the-Wild Distribution Shifts. International Conference on Machine Learning 139, 5637-5664. arXiv:2012.07421

[49] Neidlinger, P., El Nahhas, O. S. M., Muti, H. S., et al. (2026). Benchmarking foundation models as feature extractors for weakly supervised computational pathology. Nature Biomedical Engineering 10, 1113-1123. doi:10.1038/s41551-025-01516-3