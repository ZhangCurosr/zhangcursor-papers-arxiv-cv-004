# WHAT LANGUAGE ADDS, AND AT WHAT COST: FOUR CONTROLLED PROBES OF PATHOLOGY VISION–LANGUAGE MODELS

Wenhao Zhang<sup>1</sup>, Zhongliang Zhou<sup>2</sup>, Yi Ping<sup>1</sup>, Zirui Wang<sup>1</sup>, and Sheng Li\*<sup>1</sup>

<sup>1</sup>School of Data Science, University of Virginia, Charlottesville, VA 22903, USA <sup>2</sup>Merck & Co., Inc., Rahway, NJ 07065, USA

## ABSTRACT

Vision–language models (VLMs) promise to make computational pathology queryable and able to draft reports, but how much language adds beyond the vision backbone is unclear, because almost every system is evaluated by its developers on test sets of their own choosing. We grade published evidence by how it was produced and test four claims with controlled probes that reuse released checkpoints and public data, varying one input at a time. Report text adds prognostic signal: adding a frozen embedding of the stage-redacted report to a slide embedding raises the pooled C-index by 0.019 for progression-free interval and by 0.016 for overall survival across 1,633 TCGA patients, and restoring the stage statement adds nothing. Prompts add no measurable accuracy at inference: with features and pooling fixed, zero-shot classification is statistically indistinguishable from a supervised linear probe for lung and renal cancer subtyping. Slide-level question answering is barely grounded: SlideChat answers 74.82% of SlideBench-VQA(TCGA) items correctly with the slide and 74.41% without it, whereas its language backbone alone reaches 43.52%. Paraphrasing leaves accuracy unchanged but flips 8–23% of answers. Self-reported gains from agentic navigation and few-shot language priors cannot yet be attributed to language. Language thus adds real but narrow value, and we propose minimum controls that would expose these failure modes in future evaluations.

Keywords Computational pathology · Vision–language models · Evaluation · Visual grounding · Robustness · Survival analysis

## 1 Introduction

Vision–language models (VLMs) couple a histology encoder to a text encoder or a large language model, so that one system can classify tissue from a prompt, retrieve similar cases, answer free-text questions and draft reports [1–5]. Pure-vision models already reach clinical-grade accuracy on narrow tasks [6], so the question for adoption is whether language makes pathology AI better or merely more persuasive: free text can assert findings that are not in the tissue, a fluent rationale can pass for evidence, and an answer that changes with the wording of a question is incompatible with diagnosis.

Published results cannot settle this. Almost every system is evaluated by the group that built it, on test data of its own choosing and under its own protocol; the most-used benchmarks are exposed to contamination (Section 2); and the decisive comparisons are rarely run: whether language improves subtyping and survival, where vision is already strong, whether slide-level answers depend on the slide, and whether they survive rewording.

Here we weigh these claims directly. We grade evidence by how it was produced and run four controlled probes under one leakage-controlled harness, reusing released checkpoints and public data and varying a single input at a time. The probes isolate the language increment on survival prediction (Probe 1) and subtyping (Probe 2), test whether slide-level visual question answering (VQA) is grounded in the slide (Probe 3), and measure robustness to semantically equivalent rewording (Probe 4). We then grade the self-reported evidence for agentic navigation and few-shot language priors and distil the results into minimum controls. This article expands the evidence-weighing analysis of our companion review of 51 pathology VLMs [7], from which literature-derived statements are drawn; following that review, we distinguish contrastive encoders, instruction-tuned assistants, reasoning-tuned models, agents and VLM-augmented multiple-instance learning (MIL).

Table 1. Public evaluation benchmarks used by current pathology VLMs, with counts as reported by their sources. Items count images, whole-slide images (WSIs), patients or pairs; Questions count question–answer pairs. Gran., granularity (G1, patch; G2, region of interest; G3, whole slide); MCQ, multiple-choice question; VQA, visual question answering; ST, spatial transcriptomics; NR, not reported; —, not applicable. WSI-Bench counts include its training split.
<table><tr><td>Benchmark</td><td>Items</td><td>Questions</td><td>Gran.</td><td>Task type</td></tr><tr><td>Patch-level reasoning and VQA</td><td></td><td></td><td></td><td></td></tr><tr><td>PathMMU [13]</td><td>24,067</td><td>33,428</td><td>G1</td><td>MCQ diagnostic reasoning</td></tr><tr><td>PathVQA [14]</td><td>4,998</td><td>32,799</td><td>G1</td><td>Open- and closed-ended VQA</td></tr><tr><td>Quilt-VQA [15]</td><td>985</td><td>1,283</td><td>G1</td><td>Open- and closed-ended VQA</td></tr><tr><td>PathQABench-Public [3]</td><td>TCGA subset</td><td>Expert- curated</td><td>G1/G2</td><td>Diagnostic MCQ</td></tr><tr><td>ROI- and WSI-level VQA</td><td></td><td></td><td></td><td></td></tr><tr><td>PathMMU-HR2 [16]</td><td>NR</td><td>1,668</td><td>G2</td><td>VQA on 16,0002 px regions</td></tr><tr><td>WSI-VQA [17]</td><td>977 WSIs</td><td>8,672</td><td>G3</td><td>Slide-level VQA</td></tr><tr><td>WSI-Bench [12]</td><td>9,850 WSIs</td><td>179,569</td><td>G3</td><td>VQA (3 × 11 task grid)</td></tr><tr><td>SlideBench-VQA (TCGA) [11]</td><td>732 patients</td><td>7,827</td><td>G3</td><td>VQA (13 tasks)</td></tr><tr><td>SlideBench-VQA (BCNB) [11]</td><td>1,058 patients</td><td>7,274</td><td>G3</td><td>VQA (zero-shot)</td></tr><tr><td>Report generation and specialised tasks</td><td></td><td></td><td></td><td></td></tr><tr><td>PathText [18]</td><td>9,009 pairs</td><td></td><td>G3</td><td>WSI-to-text report generation</td></tr><tr><td>REG2025 [19]</td><td>10,494 pairs</td><td></td><td>G3</td><td>Report generation, 7 organs</td></tr><tr><td>HEST-Bench [20]</td><td>ST-paired WSIs</td><td></td><td>G2/G3</td><td>Gene-expression prediction</td></tr><tr><td>PathGLS [21]</td><td>NR</td><td></td><td>G1/G3</td><td>Reference-free factuality</td></tr></table>

## 2 Evidence standards and benchmarks

We weigh each published result by how it was produced, in three tiers: an independent re-evaluation of several systems under one protocol (tier 1), a third-party reproduction of a single system (tier 2), and a number a system reports about itself (tier 3). Our probes are tier-1 evidence, as are PathVLM-Eval [8] and the feature-extractor benchmark of Neidlinger et al. [9].

Table 1 lists the public benchmarks in current use. Most comparisons rest on three patch-level sets, PathMMU, PathVQA and Quilt-VQA, and all three are exposed to contamination: their images come from the literature, textbooks, educational videos and social media, the kinds of source that also populate pathology pretraining corpora [1, 10], and their question–answer pairs were generated semi-automatically or with language-model assistance. Slide-level benchmarks built from TCGA typically share their source cohort with the instruction-tuning data of the assistants evaluated on them [11, 12]. We therefore anchor the analysis on leakage-controlled evaluations.

## 3 Controlled probes

All probes freeze the representation, hold the predictor and the evaluation split fixed, and vary a single input. Model are public releases used without modification, and assistants run under their authors’ inference code. The encoder probes use patient- and tissue-source-site-disjoint five-fold splits, which keep site signatures from crossing folds [22]; the VQA probes are pure inference over released items. We report 95% confidence intervals (CIs) from 1,000 bootstrap resamples at a fixed seed, resampling cases in Probes 1–2 and questions in Probes 3–4, and call an effect significant when its CI excludes zero.

Probe 1: survival. Slides from TCGA-BRCA, -LUAD and -COAD [23] are paired with progression-free interval (PFI) and overall survival (OS) from the TCGA Clinical Data Resource [24] (pooled n = 1,633; 394 PFI and 345 OS events). Report text is the report-derived text of the public SlideInstruct/SlideBench-Caption release [11], filtered to remove follow-up and outcome sentences. A frozen PubMedBERT report embedding [25] is concatenated with the frozen TITAN slide embedding [4], and an L2-penalised linear Cox model [26] is fitted. Six arms differ only in their input: A, vision; B, vision with the full report; ${ \bf { \bar { \boldsymbol { B } } } } ^ { \prime } ,$ , vision with a stage-redacted report; C, structured stage and grade; $T _ { \mathrm { f u l l } }$ and $T _ { \mathrm { r e d } }$ , report text alone. Pooled out-of-fold Harrell’s C-index [27] is compared through four contrasts: clean language increment $( B ^ { \prime } - A )$ , staging leakage $( B - B ^ { \prime } )$ , text-channel leakage $( \bar { T } _ { \mathrm { f u l l } } - T _ { \mathrm { r e d } } )$ and text versus vision $( T _ { \mathrm { r e d } } - A )$

Probe 2: subtyping. TCGA NSCLC (LUAD versus LUSC; N = 930) and RCC (KIRC, KIRP and KICH; $N = 7 1 8 )$ slides are embedded by three language-aligned encoders, CONCH v1 [2], CONCH v1.5 and the TITAN slide encoder built on it [4], and by the pure-vision Prov-GigaPath [28]. Each yields one vector per slide, on which an L2-penalised multinomial logistic head is fitted and scored by balanced accuracy. Within CONCH v1, zero-shot classification with top-k or mean pooling over tiles is compared with supervised linear probes on the same features.

Probe 3: grounding. SlideChat [11] and WSI-LLaVA [12], with two text-only models, Qwen2.5-7B-Instruct [29] (the language backbone of SlideChat) and Qwen3-8B [30], answer the 1,227 SlideBench-VQA(TCGA) multiple-choice items whose slides could be re-encoded locally (CLAM tiling [31], CONCH features). Relative to the unperturbed condition (P0), we remove the slide, zero its features, substitute a mismatched slide, cyclically shift the options, or replace the correct option with “none of the above” (NOTA). A single deterministic parser reads the answers; the most-frequent-letter baseline is 40.7%.

Probe 4: wording. SlideChat, Qwen3-VL [32], InternVL3 [33] and a text-only Qwen3 answer a matched multiplechoice subset of SlideBench-VQA (TCGA and BCNB). With slide, question content and correct answer fixed, each item is posed canonically and in banks of semantically equivalent paraphrases, of prompt reframings (a different framing sentence or an added persona line), and of both. We report the accuracy change (canonical minus perturbed), the fraction of items whose answer flips, and answer consistency across each bank.

Linear heads and bootstrap run on CPU; zero-shot and multimodal inference run on up to two NVIDIA H200 NVL GPUs in bf16, and no model is trained or fine-tuned. Claims that the probes do not address are graded from the 51 systems of the companion review [7].

## 4 Results

## 4.1 Report text adds prognostic signal that survives stage redaction

Vision–language foundation models report better survival prediction than pure-vision models [4, 5], but reports state tumour stage, so the gain could merely restate it. The clean increment $B ^ { \prime } - A$ was positive and significant in all six cohort–endpoint settings and pooled to +0.019 C-index for PFI and +0.016 for OS (Fig. 1b). Stage does not explain it: both leakage contrasts were null, the redacted report alone was at least as prognostic as the slide embedding, and structured stage and grade (arm C) formed the weakest arm (Fig. 1a). The increment is therefore not a restatement of the stage statement. It is information in the report that complements the slide, captured by a frozen text encoder rather than by a VLM, and redaction leaves stage-correlated findings such as tumour size in place.

## 4.2 Language pays through pretrained features, not prompts

Language enters a contrastive VLM twice, as pretraining supervision and as zero-shot prompts; if predictive power lives in the features, a laboratory can replace prompts with a supervised classifier and avoid prompt sensitivity. Under identical supervised probes, all three language-aligned encoders outperformed Prov-GigaPath on both tasks (Fig. 1c). A probe uses no prompt, so the advantage resides in the features; but the encoders also differ in backbone, data, dimension and aggregation, so it cannot be attributed to language. The within-encoder arm isolates the prompt: with CONCH v1 features and pooling fixed, zero-shot classification was statistically indistinguishable from the linear probe (balanced-accuracy difference −0.020 for NSCLC and −0.019 for RCC; Fig. 1d). Top-k zero-shot classification scored higher, but its margin vanished under shared pooling, an instance-selection rather than a prompt effect. With CONCH ranked first of 19 foundation models as a weakly supervised feature extractor in an independent benchmark [9], the evidence indicates that, for in-distribution subtyping, vision–language pretraining pays, if at all, through the features it shapes rather than through prompts.

ZS top-kLP contrastive ZS meanLP post-head

![](images/8c61b3696fdbaab2db895ca8ef063546d7e4209b0c9d9aeeae103fd81fa8ceb2.jpg)

![](images/6f18ec15f2e1f43ecb8563d5ce987b9ab3b17cdc5cead1e8ac70e35e4f5d25d8.jpg)

![](images/1bd2fb7093cd2c56ae6231b840918a70a27e4ac1b6d760dbd6c3ccdaef55b476.jpg)

![](images/c45f8803f03656ee4527e51287a75c0a67781c8cf96386d5d226835bc1a4a135.jpg)  
Figure 1. Probes 1 and 2: language adds prognostic signal but no discriminative power at inference. (a) Pooled out-of-fold Harrell’s C-index for the six survival arms (A, vision; B, vision+full report; $B ^ { \prime }$ , vision+stage-redacted report; C, structured stage and grade; $T _ { \mathrm { f u l l } } , T _ { \mathrm { r e d } }$ , report text alone) for PFI and OS $( n = 1 , 6 3 3 ; 3 9 4$ PFI and 345 OS events). (b) Survival contrasts: clean language increment $( B ^ { \prime } - A )$ , staging leakage $( B - B ^ { \prime } )$ , text-channel leakage $( T _ { \mathrm { f u l l } } - T _ { \mathrm { r e d } } )$ and text versus vision $( T _ { \mathrm { r e d } } - A )$ (c) Balanced accuracy of supervised probes for $\mathsf { N S C L C } \left( N = 9 3 0 \right)$ and RCC $( N = 7 1 8 )$ subtyping; dashed lines, majority baselines. (d) Within CONCH v1: zero-shot classification with top-k or mean pooling, and linear probes; brackets, mean-pooled zero-shot minus contrastive-space linear probe. Error bars, 95% bootstrap CIs.

## 4.3 Slide-level VQA accuracy does not depend on the slide

A multiple-choice benchmark can be passed without looking, by exploiting answer distributions and option positions. SlideChat answered 74.82% of items correctly with the slide, and its text-only and mismatched-slide gaps were indistinguishable from zero, including on grading and staging questions; WSI-LLaVA lost a few points, significantly (Fig. 2a,b). The prior is learned in training: Qwen2.5-7B, SlideChat’s own backbone, answered 43.52% correctly without the slide, close to the 40.7% letter baseline, whereas SlideChat, equally blind, answered 74.41%, so about 31 points come from regularities of its instruction-tuning data rather than from the slide. The assistants were instead sensitive to answer format: shifting option positions lowered accuracy significantly, and substituting NOTA for the correct option pushed SlideChat below the letter baseline, so neither model rejects the remaining options once the correct content is removed.

## 4.4 Rewording leaves accuracy intact but changes the answers

Under paraphrase alone, the accuracy change was indistinguishable from zero for all four systems (Fig. 2d), so we detect no accuracy cost. The answers, however, were unstable: 8–23% of items changed answer under paraphrase and 25% under paraphrase plus reframing, where consistency fell to 0.57. Even SlideChat, the most stable system, changed a quarter of its answers when only the framing sentence varied, and a single persona line shifted Qwen3-VL’s accuracy significantly, the only accuracy changes whose intervals excluded zero. Leaderboards cannot see this instability; in the clinic, where the same case must yield the same diagnosis, it is disqualifying.

![](images/5251acb0fd7595ccded9de4cc749fb5a4cb977032fd9d55ccdef9d050a7b6c27.jpg)

![](images/ae84404b61e1bf2337857b7d64de0852239bffc18b9fd3948978895af532eb36.jpg)

![](images/4d722cd51ecf56b4798f331b445dc4eb37b0a8448a0a1eec86bcd3028cb2c90e.jpg)

![](images/788069f85042a4a9a20a58bdf7393eb299e9d611b11be4eaeb2796610b7defca.jpg)  
Figure 2. Probes 3 and 4, and self-reported results. (a) SlideBench-VQA(TCGA) accuracy with the full, removed (text-only) or mismatched slide; dashed line, most-frequent-letter baseline (40.7%). (b) Grounding gaps (∆acc = P0 − probe; P0, unperturbed condition): slide-removal and wrong-slide gaps are null (SlideChat) or small (WSI-LLaVA), whereas option shift lowers and none-of-the-above (NOTA) substitution collapses accuracy. (c) Self-reported downstream results by benchmark; shading encodes the metric, so bars compare only within a benchmark and metric. Superscripts index the reporting publication; no value was re-evaluated here. (d) Robustness gaps (∆acc = canonical − perturbed) under paraphrase, prompt reframing and both. Error bars, question-level bootstrap 95% CIs; asterisks, CI excludes zero; n.s., not significant.

## 4.5 Self-reported gains cannot yet be attributed

Two further claims rest on self-reported evidence (tier 3; Fig. 2c). Agentic navigation produces its largest margins where they are least separable. CPathAgent reaches 88.6% on PathMMU-HR<sup>2</sup>, 15.4 points above Gemini-2.5-Pro answering in a single pass, yet at slide level it leads ABMIL by 2.9 points on average and trails TITAN on CPTAC-Lung (88.1% versus 91.9%) [16]. The margin thus depends on scale, and only part of it is navigation: prompted to navigate in the same way, Gemini-2.5-Pro rises from 73.2% to 76.4%. With no intervals or seeds reported, neither component can be tested. Language priors in MIL earn smaller and more consistent margins: ViLa-MIL, VLEER and GMAT report few-shot gains over language-free aggregators while training fewer than 10<sup>7</sup> parameters [34–36], for example 1.7 and 5.9 AUC points for ViLa-MIL over CLAM-MB with 16 shots (Fig. 2c). Every such comparison comes from the proposing paper without independent or leakage-controlled replication, so we grade this evidence as weakly supported.

## 5 Discussion

What accuracy does not capture. Visibility reorders clinical risk: the hallucinations of generative systems surface as text that a reader can check, whereas score-only systems (contrastive encoders and VLM-augmented MIL) fail through correlational shortcuts, such as a frozen encoder passing a tissue-source-site signature onward [22], and leave a confident, wrong prediction with no trace. Long-tail behaviour is rarely reported: among the catalogued systems, HiPath falls from 70.2% on head terms to 2.6% on tail terms [37], MR-PLIP reaches 0.745 balanced accuracy on 30 EBRAINS brain-tumour subtypes [38], CPathAgent scores 64.3% on three-class BRACS regions [16], and paediatric tumours are assessed only by KEEP, on 816 WSIs from one hospital [39]. Calibration and abstention are absent: networks are poorly calibrated by default [40], free text carries no calibrated confidence, selective prediction [41] goes unused, and none of the 51 systems reports calibrated confidence. Compute is reported sporadically and in incompatible units: a few papers give training budgets [37, 42], only LAMMI-Pathology reports a memory footprint [43], and none compares runtimes across paradigms or estimates carbon cost [44], although instruction-tuned assistants (7B–15B parameters) need roughly 10<sup>4</sup> visual tokens and 40–80 GB of GPU memory per WSI at inference.

Minimum controls. Each failure reported here was exposed by a cheap control. We propose that evaluations of pathology VLMs report, at a minimum: (i) for slide-conditioned VQA, text-only, mismatched-slide, option-shift and NOTA arms alongside the majority-answer baseline; (ii) answer-flip rates and consistency under paraphrase and reframing; (iii) stage-redacted and text-only arms for claims that text improves prognosis; (iv) within-encoder comparisons for claims about prompts; (v) CIs and seeds for every headline comparison, preferably with independent replication; and (vi) compute in six fields: GPU type and count, GPU-hours, peak training memory, peak inference memory, visual tokens per WSI, and end-to-end latency on a reference slide. For design, language at inference is best spent where it is irreplaceable (prognosis, interaction and reporting), and grounding should become a training objective: answers should cite tissue evidence, change when it changes, abstain when it is absent and stay invariant to paraphrase. None of this replaces clinical validation through multisite testing, subgroup reporting and prospective reader studies.

Limitations. The probes cover released checkpoints on TCGA-derived data, two slide-level assistants, three languagealigned encoders and one pure-vision encoder; reasoning-tuned, agentic and MIL systems are not probed directly. In Probe 1, a null leakage contrast would also arise if the frozen text encoder were insensitive to stage statements, no arm combines vision with structured stage and grade, and the report text is a processed release. In Probe 2, the cross-encoder comparison is confounded, and zero-shot and supervised heads are compared under mean pooling only, without learned instance selection such as attention-based MIL. Probes 3 and 4 use multiple-choice items, so free-text generation may behave differently, and the navigation and few-shot grades rest on self-reported numbers.

## 6 Conclusion

Weighed by controlled probes rather than headline numbers, language adds real but narrow value to pathology VLMs. Report text carries prognostic signal that survives stage redaction, but prompts add no discriminative accuracy at inference, and slide-level answers are neither grounded in the slide nor stable under rewording. Until grounding, paraphrase consistency and calibration are reported as routinely as accuracy, benchmark scores should be read as upper bounds on what a model reads from the slide, not as evidence that it does.

## Data and code availability

All data are public and de-identified, and no new human-subjects data were collected. Whole-slide images and clinical endpoints are from TCGA and the TCGA Clinical Data Resource via the NIH Genomic Data Commons (https://portal.gdc.cancer.gov); report-derived text and VQA items are from the public SlideChat release (https://huggingface.co/datasets/General-Medical-AI/SlideChat). All checkpoints (CONCH v1 and v1.5, TITAN, Prov-GigaPath, PubMedBERT, SlideChat, WSI-LLaVA, Qwen2.5-7B-Instruct, Qwen3-8B, Qwen3-VL and InternVL3) are public releases. A maintained list of pathology VLMs is available at https://github.com/wen haozhang0066/Awesome-Pathology-VLMs.

## Author contributions

W.Z., Z.Z. and S.L. conceived the project. W.Z., Y.P. and Z.W. performed the experiments. W.Z., Y.P., Z.W., P.W., H.Y.   
and T.W. wrote the manuscript. W.Z., Z.Z., Q.Z., Y.Y., J.K., E.T.R., J.B.W., C.E.P. and S.L. revised the manuscript.   
Z.Z., J.K., E.T.R., J.B.W., C.E.P. and S.L. supervised the project. All authors read and approved the final manuscript.

## Competing interests

Z.Z., Q.Z., J.K., E.T.R., J.B.W. and C.E.P. are employees of Merck Sharp & Dohme LLC, a subsidiary of Merck & Co., Inc., Rahway, NJ, USA, and may hold stock in Merck & Co., Inc. The remaining authors declare no competing interests.

## References

[1] Zhi Huang, Federico Bianchi, Mert Yuksekgonul, Thomas J. Montine, and James Zou. A visual–language foundation model for pathology image analysis using medical Twitter. Nature Medicine, 29(9):2307–2316, 2023.

[2] Ming Y. Lu, Bowen Chen, Drew F. K. Williamson, et al. A visual-language foundation model for computational pathology. Nature Medicine, 30(3):863–874, 2024.

[3] Ming Y. Lu, Bowen Chen, Drew F. K. Williamson, et al. A multimodal generative AI copilot for human pathology. Nature, 634 (8033):466–473, 2024.

[4] Tong Ding, Sophia J. Wagner, Andrew H. Song, et al. Multimodal whole slide foundation model for pathology. Nature Medicine, 2025. doi: 10.1038/s41591-025-03982-3.

[5] Jinxi Xiang, Xiyue Wang, Xiaoming Zhang, et al. A vision–language foundation model for precision oncology. Nature, 638 (8051):769–778, 2025.

[6] Gabriele Campanella, Matthew G. Hanna, Luke Geneslaw, et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nature Medicine, 25(8):1301–1309, 2019.

[7] Wenhao Zhang, Zhongliang Zhou, Yi Ping, et al. Does language help? Grading the evidence for pathology vision-language models, 2026. Companion review.

[8] Nauman Ullah Gilal, Rachida Zegour, Khaled Al-Thelaya, et al. PathVLM-Eval: Evaluation of open vision language models in histopathology. Journal of Pathology Informatics, 18:100455, 2025.

[9] Peter Neidlinger, Omar S. M. El Nahhas, Hannah Sophie Muti, et al. Benchmarking foundation models as feature extractors for weakly supervised computational pathology. Nature Biomedical Engineering, 10(6):1113–1123, 2026. doi: 10.1038/s41551-0 25-01516-3.

[10] Wisdom Oluchi Ikezogwo, Mehmet Saygin Seyfioglu, Fatemeh Ghezloo, et al. Quilt-1M: One million image-text pairs for histopathology. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pages 37995–38017, 2023.

[11] Ying Chen, Guoan Wang, Yuanfeng Ji, et al. SlideChat: A large vision-language assistant for whole-slide pathology image understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. arXiv:2410.11761.

[12] Yuci Liang, Xinheng Lyu, Wenting Chen, et al. WSI-LLaVA: A multimodal large language model for whole slide image. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 22718–22727, 2025.

[13] Yuxuan Sun, Hao Wu, Chenglu Zhu, et al. PathMMU: A massive multimodal expert-level benchmark for understanding and reasoning in pathology. In European Conference on Computer Vision (ECCV), pages 56–73. Springer, 2024.

[14] Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. PathVQA: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286, 2020.

[15] Mehmet Saygin Seyfioglu, Wisdom Oluchi Ikezogwo, Fatemeh Ghezloo, Ranjay Krishna, and Linda Shapiro. Quilt-LLaVA: Visual instruction tuning by extracting localized narratives from open-source histopathology videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13183–13192, 2024.

[16] Yuxuan Sun, Yixuan Si, Chenglu Zhu, et al. CPathAgent: An agent-based foundation model for interpretable high-resolution pathology image analysis mimicking pathologists’ diagnostic logic. In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2505.20510.

[17] Pingyi Chen, Chenglu Zhu, Sunyi Zheng, Honglin Li, and Lin Yang. WSI-VQA: Interpreting whole slide images by generative visual question answering. In European Conference on Computer Vision (ECCV), pages 401–417. Springer, 2024.

[18] Pingyi Chen, Honglin Li, Chenglu Zhu, et al. WsiCaption: Multiple instance generation of pathology reports for gigapixel whole-slide images. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), pages 546–556. Springer, 2024.

[19] REG2025 Organizers. REG2025: Report generation of pathology using Pan-Asia giga-pixel WSIs. MICCAI 2025 Grand Challenge, https://reg2025.grand-challenge.org/, 2025.

[20] Guillaume Jaume, Paul Doucet, Andrew H. Song, et al. HEST-1k: A dataset for spatial transcriptomics and histology image analysis. In Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2406.16192.

[21] Minbing Chen, Zhu Meng, and Fei Su. PathGLS: Evaluating pathology vision-language models without ground truth through multi-dimensional consistency. arXiv preprint arXiv:2603.16113, 2026.

[22] Frederick M. Howard, James Dolezal, Sara Kochanny, et al. The impact of site-specific digital histology signatures on deep learning model accuracy and bias. Nature Communications, 12(1):4423, 2021.

[23] John N. Weinstein, Eric A. Collisson, Gordon B. Mills, et al. The Cancer Genome Atlas Pan-Cancer analysis project. Nature Genetics, 45(10):1113–1120, 2013.

[24] Jianfang Liu, Tara Lichtenberg, Katherine A. Hoadley, et al. An integrated TCGA pan-cancer clinical data resource to drive high-quality survival outcome analytics. Cell, 173(2):400–416.e11, 2018.

[25] Yu Gu, Robert Tinn, Hao Cheng, et al. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computingfor Healthcare, 3(1):1–23, 2021.

[26] David R. Cox. Regression models and life-tables. Journal ofthe Royal Statistical Society: Series B (Methodological), 34(2): 187–220, 1972.

[27] Frank E. Harrell, Robert M. Califf, David B. Pryor, Kerry L. Lee, and Robert A. Rosati. Evaluating the yield of medical tests. JAMA, 247(18):2543–2546, 1982.

[28] Hanwen Xu, Naoto Usuyama, Jaspreet Bagga, et al. A whole-slide foundation model for digital pathology from real-world data. Nature, 630:181–188, 2024.

[29] Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

[30] Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[31] Ming Y. Lu, Drew F. K. Williamson, Tiffany Y. Chen, et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nature Biomedical Engineering, 5(6):555–570, 2021.

[32] Qwen Team. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

[33] Jinguo Zhu, Weiyun Wang, Zhe Chen, et al. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

[34] Jiangbo Shi, Chen Li, Tieliang Gong, Yefeng Zheng, and Huazhu Fu. ViLa-MIL: Dual-scale vision-language multiple instance learning for whole slide image classification. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11248–11258, 2024.

[35] Anh Tien Nguyen, Keunho Byeon, Kyungeun Kim, and Jin Tae Kwak. VLEER: Vision and language embeddings for explainable whole slide image representation. In MICCAI Workshop on Interpretability ofMachine Intelligence in Medical Image Computing (iMIMIC), 2025. arXiv:2502.20850.

[36] Nguyen B. L. Quang, Nguyen L. N. Binh, Trong-Hieu Nguyen, et al. GMAT: Grounded multi-agent clinical description generation for text encoder in vision-language MIL for whole slide image classification. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2025. arXiv:2508.01293.

[37] Ruicheng Yuan, Zhenxuan Zhang, Anbang Wang, et al. HiPath: Hierarchical vision-language alignment for structured pathology report prediction. arXiv preprint arXiv:2603.19957, 2026.

[38] Shahad Albastaki, Anabia Sohail, Iyyakutti Iyappan Ganapathi, et al. Multi-resolution pathology-language pre-training model with text-guided visual representation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 25907–25919, 2025.

[39] Xiao Zhou, Luoyi Sun, Dexuan He, et al. Knowledge-enhanced pretraining for vision-language pathology foundation model on cancer diagnosis. Cancer Cell, 2026.

[40] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning (ICML), pages 1321–1330. PMLR, 2017.

[41] Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems, volume 30, 2017.

[42] Chunyuan Li, Cliff Wong, Sheng Zhang, et al. LLaVA-Med: Training a large language-and-vision assistant for biomedicine in one day. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pages 28541–28564, 2023.

[43] Hui Su, Shijia Zhang, and Xi Wang. LAMMI-Pathology: A tool-centric bottom-up LVLM-agent framework for molecularly informed medical intelligence in pathology. arXiv preprint arXiv:2602.18773, 2026.

[44] Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres. Quantifying the carbon emissions of machine learning. arXiv preprint arXiv:1910.09700, 2019.