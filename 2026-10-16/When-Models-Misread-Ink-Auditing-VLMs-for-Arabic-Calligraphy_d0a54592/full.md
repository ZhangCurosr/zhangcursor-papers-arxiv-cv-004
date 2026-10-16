# When Models Misread Ink: Auditing VLMs for Arabic Calligraphy Recognition

Niaz Mohaiman Abtahi<sup>1</sup>, Evan Maksud<sup>1</sup>, Reyana Shahid<sup>2</sup>

<sup>1</sup>Advanced AI Lab Limited

<sup>2</sup>United International University

Correspondence to: tahiniaz@gmail.com

GitHub/When-Models-Misread-Ink

## Abstract

Arabic calligraphy is a high-stakes cultural visual domain for Muslim communities, yet VLM evaluations rarely test its fine-grained script traditions. We audit 12 VLMs on a cleaned 1,195-image six-way benchmark using numeric, visual-cue, and Diwani-first prompts. In this closed-set setting, the best numeric-prompt accuracy is 60.9% for closed/API models and 47.2% for open-weight models. The main failure is default-label collapse: every numeric run is dominated by a single label taking 33–63% of predictions, against an 18.8% majority-class baseline. For 8 of 12 models the default is Thuluth; GPT-4o labels 52.9% of all images as Thuluth although it is only 18.1% of the set. Prompting helps unevenly: visual cues move accuracy by more than 10 points for some models but hurt others, and foregrounding Diwani often relocates the default rather than removing it. Arabic calligraphy recognition is therefore a compact, culturally grounded stress test for VLM robustness and for measuring which visual traditions become legible to automated systems.

## 1 Introduction

This paper studies a narrow but important question: given an image of Arabic calligraphy, can modern vision–language models identify its script style rather than collapse it into a familiar label? More specifically, when the choices are Kufic, Naskh, Thuluth, Diwani, Ruq’ah, and Nasta’liq, do models show stable visual discrimination, or do prompt wording and label order change which traditions become legible?

This question sits at the intersection of rapid VLM progress [Radford et al., 2021, Alayrac et al., 2022, Liu et al., 2023], culturally diverse vision evaluation [Liu et al., 2021, Romero et al., 2024, Ramaswamy et al., 2023], and evidence that model outputs can shift with option order and prompt format [Zheng et al., 2024, Pezeshkpour and Hruschka, 2024, Sclar et al., 2024, Lu et al., 2022]. Prior work introduced DuwatBench for Arabic calligraphy understanding [Patle et al., 2026]; we focus on whether VLMs have reliable style priors after rebalancing the evaluation set and testing controlled prompt interventions. The task is culturally grounded for Muslim communities [Muslims in ML, 2026]: calligraphic styles carry historical, religious, regional, and institutional associations, so systematic misclassification can distort access to visual heritage.

Our contributions are:

• a cleaned and rebalanced 1,195-image benchmark for six-way Arabic calligraphy style recognition;

• a controlled evaluation of four closed/API VLMs and eight open-weight VLMs on the same image manifest;

• three prompt conditions—numeric, visual-cue, and Diwani-first—that test whether style descriptions or explicit attention to an under-recognized class repair errors; and

• analyses of default-label collapse, Thuluth over-prediction, Diwani under-recognition, and paired prompt effects.

Our central finding is not simply that current VLMs are inaccurate. Rather, their errors are culturally structured: many models overuse Thuluth as a default, fail to recover Diwani, and respond unevenly to prompts that should make the visual distinctions clearer.

## 2 Dataset and Task

Dataset. We use a cleaned 1,195-image six-style dataset. DuwatBench [Patle et al., 2026] is supplemented with RuFa images for Ruq’ah/Nasta’liq [Mhmoodlan, 2026] and HICMA images for Kufic/Naskh [HICMA Project, 2026], reducing the original Thuluth-heavy imbalance. Curation uses source aggregation, single-label filtering, class-balance supplementation, file-integrity checks, duplicate removal, and manifest locking. After removing one undersized image and four exact du plicates, the final counts are Kufic 106, Naskh 218, Thuluth 216, Diwani 217, Ruq’ah 225, and Nasta’liq 213; the majority baseline falls from 56.4% to 18.8%. Thuluth and Diwani are both entirely DuwatBench, so their central confusion is not cross-source domain shift; other labels use mixed sources, disclosed in the appendix.

Task. The task is closed-set six-way style classification: given one image, select one script style. We evaluate style recognition only, not OCR, transcription, semantic interpretation, religiouscontent judgment, or localization. All metrics use the same 1,195-image manifest for every model– prompt run.

## 3 Experimental Setup

Prompt conditions. We use three prompts. Numeric lists numbered styles and requires one digit. Visual-cue adds short style descriptions. Diwani-first keeps the visual-cue framing but moves Diwani first and asks for the script name, testing whether foregrounding an under-recognized style rescues Diwani errors. We introduced this condition after early runs showed Diwani was often missed or absorbed into Thuluth; it is a diagnostic prompt, not a preferred deployment prompt.

Models. We evaluate four closed/API models [OpenAI, 2026]: GPT-4o, GPT-4.1, GPT-5.5, and GPT-5.6 Terra. The open-weight set includes Qwen2.5-VL-7B/32B [Bai et al., 2025b], Qwen3-VL-8B/32B [Bai et al., 2025a], InternVL3.5-8B/38B [Wang et al., 2025], MiniCPM-V 4.5 [Yu et al., 2025], and Gemma 4 12B-IT [Gemma Team, 2026]. Open-weight runs use greedy decoding with seed 42; closed/API runs use the most deterministic supported settings.

Metrics. We report accuracy, balanced accuracy, macro-F1, per-style precision/recall/F1, and prediction ratio: predicted count divided by true count. The main diagnostics are Thuluth prediction ratio and Diwani recall, because early runs absorbed ornate or uncertain examples into Thuluth while missing Diwani. Paired prompt comparisons use exact McNemar tests on same-image correctness vectors; visual-cue changes with p ≥ 0.05 are marked not significant.

## 4 Results

Table 1 summarizes the numeric-prompt comparison. The best closed/API numeric-prompt accuracy is 60.9% (GPT-5.5), while the best open-weight numeric-prompt accuracy is 47.2% (Gemma 4 12B-IT). Thus, although the task is visually compact and closed-set, it remains difficult for current VLMs.

Table 1: Numeric-prompt results. Thuluth bias is predicted/true Thuluth count; closer to 1.0 is better. Bold marks the best value within each model block.

<table><tr><td>Model</td><td>Acc. ↑ Macro F1 ↑</td><td></td><td>Thuluth bias</td><td>Diwani recall ↑</td><td>Diwani precision ↑</td><td>Diwani F1↑</td></tr><tr><td colspan="7">Closed/API VLMs</td></tr><tr><td>GPT-40</td><td>50.5</td><td>0.504</td><td>2.93</td><td>15.7</td><td>75.6</td><td>0.260</td></tr><tr><td>GPT-4.1</td><td>55.3</td><td>0.507</td><td>2.05</td><td>1.8</td><td>66.7</td><td>0.036</td></tr><tr><td>GPT-5.5</td><td>60.9</td><td>0.616</td><td>2.05</td><td>31.8</td><td>71.9</td><td>0.441</td></tr><tr><td>GPT-5.6 Terra</td><td>60.3</td><td>0.628</td><td>2.07</td><td>46.5</td><td>66.9</td><td>0.549</td></tr><tr><td colspan="7">Open-weight VLMs</td></tr><tr><td>Qwen2.5-VL-7B</td><td>29.2</td><td>0.255</td><td>0.91</td><td>9.2</td><td>26.3</td><td>0.137</td></tr><tr><td>Qwen2.5-VL-32B</td><td>36.7</td><td>0.378</td><td>2.75</td><td>35.0</td><td>25.6</td><td>0.296</td></tr><tr><td>Qwen3-VL-8B</td><td>46.5</td><td>0.457</td><td>2.69</td><td>18.9</td><td>63.1</td><td>0.291</td></tr><tr><td>Qwen3-VL-32B</td><td>40.8</td><td>0.421</td><td>3.47</td><td>6.5</td><td>53.8</td><td>0.115</td></tr><tr><td>InternVL3.5-8B</td><td>22.9</td><td>0.246</td><td>1.33</td><td>5.5</td><td>85.7</td><td>0.104</td></tr><tr><td>InternVL3.5-38B</td><td>35.8</td><td>0.370</td><td>1.53</td><td>56.2</td><td>30.5</td><td>0.395</td></tr><tr><td>MiniCPM-V 4.5</td><td>31.6</td><td>0.331</td><td>1.17</td><td>42.4</td><td>18.6</td><td>0.258</td></tr><tr><td>Gemma 4 12B-IT</td><td>47.2</td><td>0.454</td><td>2.89</td><td>6.5</td><td>46.7</td><td>0.113</td></tr></table>

Structured errors. The dominant failure is default-label collapse, not random confusion. Thuluth is over-predicted in 11 of 12 numeric-prompt runs; closed/API models choose it 2.05–2.93× its true frequency, and Qwen3-VL-32B reaches 3.47×. This persists after rebalancing: every numeric run has one label taking at least one-third of predictions, rising to 52.9% for GPT-4o and 62.8% for Qwen3-VL-32B. False Thuluth predictions most often come from true Diwani (40.9%), then Naskh (21.5%) and Nasta’liq (16.4%), suggesting both visual-neighbor confusion and an uncertainty default.

Diwani remains the clearest casualty. GPT-4.1 recalls only 1.8% of Diwani images, while Qwen3- VL-32B and Gemma 4 12B-IT recall only 6.5%; InternVL3.5-38B reaches 56.2% recall but only 30.5% precision. GPT-5.6 Terra is the only model with both non-trivial recall and stronger Diwani F1 (0.549). Figure 1 visualizes this pair of errors: which true labels are absorbed into Thuluth and how much true Diwani each model recovers.

![](images/997c076c462e86f02bfd96da691dee4397cffa804b125186b106840c6e27e2e4.jpg)  
Figure 1: Numeric-prompt failure structure. Left: stacked counts show which true labels are incorrectly absorbed into Thuluth. Right: Diwani recall remains low for many models despite higher overall accuracy.

Prompt interventions are unstable. Figure 2 shows that prompting does not behave like a uniform fix. Visual cues improve some models, e.g., GPT-4o (+10.2 points) and Qwen2.5-VL-7B (+12.7), but hurt others, e.g., GPT-5.6 Terra (-7.3) and Gemma 4 12B-IT (-3.1); five of twelve changes are not significant. Diwani-first can raise recall sharply, especially for InternVL3.5-8B (+89.9), but often relocates the default rather than repairing recognition: InternVL3.5-8B answers Diwani for 90.3% of all images, while Qwen3-VL models and Gemma 4 12B-IT still end with 0% Diwani recall.

![](images/037375be159620ac91830d650b5dd1b316d2c42706dd2ba6d1a398bed110a582.jpg)

![](images/1b45c508f305385df8505b02f42ad2c588dfd74f312b0614188dbbf21eda6423.jpg)  
Figure 2: Prompt effects relative to the numeric prompt. Visual cues are inconsistent; Diwani-first often raises Diwani recall at the cost of calibration.

## 5 Discussion and Limitations

This study supports a concrete claim: Arabic calligraphy recognition remains a brittle fine-grained cultural vision task for current VLMs. Aggregate accuracy hides the main failure mode: models often behave like default-label emitters with partial discrimination layered on top. Several models overuse Thuluth as the default label, while others collapse onto Naskh, Diwani, or Nasta’liq depending on model family and prompt. These errors matter because calligraphic styles encode cultural, historical, and religious contexts, and automated tools should not collapse less-visible traditions into more familiar labels.

Community relevance. The failure mode matters for education, museums, manuscript archives, social media, and community-facing search: collapsing Diwani, Ruq’ah, or Nasta’liq into Thuluth can distort access to Islamic visual heritage.

The study has limitations: it covers six broad labels rather than regional schools, mixed-style compositions, dating, handwriting, OCR, or localization; source and label are partly confounded for non-DuwatBench supplements; Prompt C confounds label order and response format; and API-served models can change over time.

Methodologically, the benchmark is useful because it separates two effects that are often blurred in cultural-vision evaluations. The rebalanced manifest reduces the chance that high Thuluth prediction is simply learned from label frequency, while the prompt interventions show whether the same images become more legible when models are given style descriptions or an explicit Diwani prior. The negative result is therefore informative: even when a prompt names the right visual features, many systems still substitute a familiar-looking script for the correct one.

Implications. For deployment, prompt descriptions are not enough: systems that classify or retrieve Arabic calligraphy should report uncertainty, validate performance per style, and avoid treating high aggregate accuracy as evidence that all traditions are equally legible.

## 6 Conclusion

We present a 12-model evaluation of Arabic calligraphy style recognition under three prompt conditions. The results reveal limited accuracy, recurring Thuluth over-prediction, Diwani underrecognition, and nonuniform prompt sensitivity across closed/API and open-weight VLMs. Arabic calligraphy therefore offers a compact culturally grounded benchmark for measuring not only model accuracy, but how errors redistribute visibility across significant visual traditions.

Acknowledgments. This work received no external funding. The authors declare no competing interests.

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 2022.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Gemma Team. Gemma 4 technical report. arXiv preprint arXiv:2607.02770, 2026.

HICMA Project. HICMA: Ai dataset for arabic text recognition. https://hicma.net/, 2026. Accessed 2026-09-08.

Fangyu Liu, Emanuele Bugliarello, Edoardo Maria Ponti, Siva Reddy, Nigel Collier, and Desmond Elliott. Visually grounded reasoning across languages and cultures. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10467–10485, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.818. URL https://aclanthology.org/2021.emnlp-main. 818/.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems, 2023.

Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, pages 8086–8098, Dublin, Ireland, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.556. URL https://aclanthology.org/2022.acl-long.556/.

Mhmoodlan. RuFa Dataset. https://www.kaggle.com/datasets/mhmoodlan/rufa-dataset, 2026. Accessed 2026- 09-08.

Muslims in ML. Call for Papers: MusIML@NeurIPS2026. https://www.musiml.org/events/2026-NeurIPS/index. html, 2026.

OpenAI. OpenAI API Model Catalog. https://platform.openai.com/docs/models, 2026. Accessed 2026-09-08.

Shubham Patle, Sara Ghaboura, Hania Tariq, Mohammad Usman Khan, Omkar Thawakar, Rao Muhammad Anwer, and Salman Khan. DuwatBench: Bridging language and visual heritage through an Arabic calligraphy benchmark for multimodal understanding. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics, pages 5256–5269, Rabat, Morocco, 2026. Association for Computational Linguistics. doi: 10.18653/v1/2026.eacl-long.243. URL https://aclanthology.org/2026.eacl-long.243/.

Pouya Pezeshkpour and Estevam Hruschka. Large language models sensitivity to the order of options in multiple-choice questions. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 2006–2017, Mexico City, Mexico, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-naacl.130. URL https: //aclanthology.org/2024.findings-naacl.130/.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. Proceedings ofthe 38th International Conference on Machine Learning, 2021.

Vikram V. Ramaswamy, Sing Yu Lin, Dora Zhao, Aaron B. Adcock, Laurens van der Maaten, Deepti Ghadiyaram, and Olga Russakovsky. GeoDE: a geographically diverse evaluation dataset for object recognition. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2301.02560.

David Romero, Chenyang Lyu, Haryo Akbarianto Wibowo, Teresa Lynn, Injy Hamed, et al. CVQA: Culturally-diverse multilingual visual question answering benchmark. In Advances in Neural Information Processing Systems, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ 1568882ba1a50316e87852542523739c-Abstract-Datasets\_and\_Benchmarks\_Track.html.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=RIu5lyNXjT.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Yuanqian Zhao, et al. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe. arXiv preprint arXiv:2509.18154, 2025.

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large language models are not robust multiple choice selectors. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=shr9PXz7T0.

## A Dataset construction details

The evaluated dataset contains 1,195 single-label Arabic calligraphy images. It was constructed from DuwatBench [Patle et al., 2026] plus real-image supplements from RuFa [Mhmoodlan, 2026] and HICMA [HICMA Project, 2026] used to reduce the original class imbalance. Curation followed a fixed pipeline: aggregate source images, retain single-label six-style items, supplement underrepresented classes, check image-file integrity, remove exact duplicates and undersized files, then lock the manifest used for all inference runs. The final manifest contains 106 Kufic, 218 Naskh, 216 Thuluth, 217 Diwani, 225 Ruq’ah, and 213 Nasta’liq images. Table 2 reports the source composition by label. Thuluth and Diwani share the same DuwatBench provenance, so their central confusion in the main results is not a cross-source artifact; other labels should be interpreted with the mixed-source composition in mind. Before model evaluation, the dataset was checked for missing files, corrupted images, orphan images, undersized images, and exact duplicates. One undersized image and four duplicate images were removed. Figure 3 shows representative examples from the final six-style evaluation set.

Table 2: Source composition of the cleaned evaluation set by true label.
<table><tr><td>True label</td><td>DuwatBench</td><td>RuFa</td><td>HICMA</td></tr><tr><td>Kufic</td><td>80</td><td>0</td><td>26</td></tr><tr><td>Naskh</td><td>103</td><td>0</td><td>115</td></tr><tr><td>Thuluth</td><td>216</td><td>0</td><td>0</td></tr><tr><td>Diwani</td><td>217</td><td>0</td><td>0</td></tr><tr><td>Ruq&#x27;ah</td><td>75</td><td>150</td><td>0</td></tr><tr><td>Nasta&#x27;liq</td><td>64</td><td>149</td><td>0</td></tr></table>

![](images/53bd91cad6298371310d3b2491bc1ba3890a5a81526c771c3b8baf65d18e9b50.jpg)  
Figure 3: Representative examples from the six calligraphy styles in the cleaned evaluation set.

## B Prompts

The prompt text below is line-wrapped for page layout.

## Numeric prompt.

You are shown an Arabic calligraphy image. Below are six possible styles, each with a number:

1. Kufic

2. Naskh

3. Thuluth

4. Diwani

5. Ruq’ah

6. Nasta’liq

Respond with only the number of the style that best matches this image.   
Respond with a single digit only -- no words, no explanation.

## Visual-cue prompt.

You are shown an Arabic calligraphy image. Classify its calligraphic style.

Choose exactly one of these six styles:

1. Kufic   
2. Naskh   
3. Thuluth   
4. Diwani   
5. Ruq’ah   
6. Nasta’liq

Use the following visual cues:

\- Kufic: angular, geometric, squared or block-like letterforms; strong horizontal/vertical structure; often rigid and monumental.

\- Naskh: clear, balanced, readable manuscript style; rounded letters; moderate stroke contrast; commonly used for Qur’anic and book text.

\- Thuluth: large, elegant, sweeping curves; elongated verticals; complex decorative composition; often used in architectural or ceremonial calligraphy.

\- Diwani: highly cursive and ornamental; dense flowing curves; letters may interlace; often compact and decorative.

\- Ruq’ah: simple, fast handwriting-like style; compact letters; short strokes; practical and less ornamental.

\- Nasta’liq: diagonal/slanted baseline; hanging or cascading composition; soft flowing curves; common in Persian/Urdu calligraphy.

Base your answer only on visible visual style, not on the meaning of the text, image background, decoration, or color.

If multiple styles seem possible, choose the closest match based on the dominant letter shapes and composition.

Respond with only a single digit from 1 to 6. Do not explain.

## Diwani-first prompt.

Look carefully at the calligraphic strokes, curves, and letterforms in this Arabic calligraphy image. Based on the following characteristics, determine which classical script family it belongs to:

Diwani: highly cursive and ornamental; dense flowing curves; letters may interlace; often compact and decorative.   
Kufic: angular, geometric, squared or block-like letterforms; strong   
horizontal/vertical structure; often rigid and monumental.   
Naskh: clear, balanced, readable manuscript style; rounded letters; moderate stroke contrast; commonly used for Qur’anic and book text.   
Thuluth: large, elegant, sweeping curves; elongated verticals; complex decorative composition; often used in architectural or ceremonial calligraphy. Ruq’ah: simple, fast handwriting-like style; compact letters; short strokes; practical and less ornamental.   
Nasta’liq: diagonal/slanted baseline; hanging or cascading composition; soft flowing curves; common in Persian/Urdu calligraphy.

Base your answer only on visible visual style, not on the meaning of the text, image background, decoration, or color. State only the name of the script."

## C Representative confusion matrices

Figure 4 gives one closed/API and one open-weight numeric-prompt confusion matrix. It comple ments the failure summary in Figure 1 by showing the full six-way redistribution of predictions.

![](images/9b3d727ea1c7b9a85639b7460b929155e73966f9c527b4d15e1c515d9b834c3a.jpg)

![](images/bd9ab396306a0fb33f6bade330be5734c67301d01d9517338431e105f6b5aea7.jpg)  
Figure 4: Representative numeric-prompt confusion matrices for GPT-4.1 and Qwen3-VL-32B.

## D Per-style prompt profiles

Figure 5 shows per-style recall profiles for six representative models. Each spoke is one calligraphy label, and each polygon is one prompt condition. The profiles make prompt effects visible at the class level: some prompts recover Diwani for particular models, while others shift errors toward different labels rather than improving all styles uniformly.

![](images/36e64b955f4f30d046e6b346a2d72c710c95c6c2d752bd237b6162c3c715e39a.jpg)  
Figure 5: Per-style recall across prompt conditions for six representative models.

## E Full model–prompt summary

Table 3 reports all 36 completed runs. Values are generated from the retained closed/API CSV files and open-weight JSONL outputs.

## F Reproducibility notes

All model–prompt runs use the same 1,195-row manifest. Open-weight experiments use fixed seed 42 and greedy decoding. Closed/API experiments were run with deterministic settings where supported; GPT-5-family models did not support the same temperature setting as GPT-4-family models, which limits exact reproducibility. All closed/API CSVs and open-weight JSONL outputs are retained locally together with the script that regenerates summary metrics.

## G Original DuwatBench-only context

Before constructing the cleaned/rebalanced 1,195-image evaluation set, exploratory runs were conducted on the original DuwatBench [Patle et al., 2026]-style 1,272-row files. Those files include 20 multi-label or ambiguous true-label rows; Table 4 reports metrics after excluding them. We do not mix these numbers into the main comparison because they use a different dataset distribution and fewer models. They are nevertheless useful context: original DuwatBench is strongly Thuluthheavy, so raw accuracy can look high even when balanced accuracy remains modest.

These context runs support two design choices in the main paper. First, they motivate rebalancing: the original clean six-class subset has a 56.4% majority-class baseline because Thuluth dominates the dataset. Second, they show why prompt effects should be interpreted cautiously: GPT-4o’s alternative wording run greatly increases Diwani recall while reducing overall accuracy, suggesting that prompt changes can swap the default label rather than solve fine-grained recognition.

Table 3: Full model–prompt summary. Accuracy and balanced accuracy are percentages; Macro-F1 is unitless; Thuluth ratio is predicted count divided by true count; Diwani recall is a percentage. Bold marks best or closest-to-ideal metric values.
<table><tr><td>Model</td><td>Prompt</td><td>Accuracy ↑</td><td>Balanced acc. ↑</td><td>Macro-F1 ↑</td><td>Thuluth ratio</td><td>Diwani recall ↑</td></tr><tr><td>GPT-40</td><td>Numeric</td><td>50.5</td><td>54.1</td><td>0.504</td><td>2.93</td><td>15.7</td></tr><tr><td>GPT-40</td><td>Visual-cue</td><td>60.8</td><td>63.3</td><td>0.627</td><td>2.02</td><td>43.3</td></tr><tr><td>GPT-40</td><td>Diwani-first</td><td>59.6</td><td>62.6</td><td>0.607</td><td>1.56</td><td>41.0</td></tr><tr><td>GPT-4.1</td><td>Numeric</td><td>55.3</td><td>58.9</td><td>0.507</td><td>2.05</td><td>1.8</td></tr><tr><td>GPT-4.1</td><td>Visual-cue</td><td>56.6</td><td>60.0</td><td>0.516</td><td>2.11</td><td>1.8</td></tr><tr><td>GPT-4.1</td><td>Diwani-first</td><td>56.9</td><td>60.5</td><td>0.527</td><td>1.86</td><td>12.0</td></tr><tr><td>GPT-5.5</td><td>Numeric</td><td>60.9</td><td>62.3</td><td>0.616</td><td>2.05</td><td>31.8</td></tr><tr><td>GPT-5.5</td><td>Visual-cue</td><td>61.1</td><td>62.5</td><td>0.615</td><td>2.19</td><td>18.9</td></tr><tr><td>GPT-5.5</td><td>Diwani-first</td><td>59.8</td><td>61.5</td><td>0.603</td><td>2.24</td><td>17.1</td></tr><tr><td>GPT-5.6 Terra</td><td>Numeric</td><td>60.3</td><td>62.7</td><td>0.628</td><td>2.07</td><td>46.5</td></tr><tr><td>GPT-5.6 Terra</td><td>Visual-cue</td><td>53.1</td><td>55.7</td><td>0.533</td><td>2.55</td><td>12.0</td></tr><tr><td>GPT-5.6 Terra</td><td>Diwani-first</td><td>52.6</td><td>55.1</td><td>0.536</td><td>2.50</td><td>15.7</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Numeric</td><td>29.2</td><td>29.1</td><td>0.255</td><td>0.91</td><td>9.2</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Visual-cue</td><td>41.9</td><td>42.1</td><td>0.398</td><td>0.78</td><td>42.9</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Diwani-first</td><td>47.6</td><td>46.9</td><td>0.438</td><td>0.54</td><td>84.8</td></tr><tr><td>Qwen2.5-VL-32B</td><td>Numeric</td><td>36.7 43.3</td><td>37.3</td><td>0.378</td><td>2.75</td><td>35.0</td></tr><tr><td>Qwen2.5-VL-32B Qwen2.5-VL-32B</td><td>Visual-cue</td><td></td><td>43.1</td><td>0.410</td><td>2.60</td><td>14.3</td></tr><tr><td>Qwen3-VL-8B</td><td>Diwani-first</td><td>41.6</td><td>41.4</td><td>0.390</td><td>1.48</td><td>36.9</td></tr><tr><td></td><td>Numeric Visual-cue</td><td>46.5 46.2</td><td>46.5</td><td>0.457</td><td>2.69</td><td>18.9</td></tr><tr><td>Qwen3-VL-8B Qwen3-VL-8B</td><td>Diwani-first</td><td>39.2</td><td>46.0 37.4</td><td>0.431</td><td>2.38</td><td>25.3</td></tr><tr><td>Qwen3-VL-32B</td><td>Numeric</td><td>40.8</td><td></td><td>0.292</td><td>2.89</td><td>0.0</td></tr><tr><td>Qwen3-VL-32B</td><td>Visual-cue</td><td>39.1</td><td>42.7 41.4</td><td>0.421</td><td>3.47</td><td>6.5</td></tr><tr><td>Qwen3-VL-32B</td><td>Diwani-first</td><td>40.0</td><td>41.2</td><td>0.385</td><td>3.65</td><td>1.8</td></tr><tr><td>InternVL3.5-8B</td><td>Numeric</td><td>22.9</td><td></td><td>0.345</td><td>2.80</td><td>0.0</td></tr><tr><td>InternVL3.5-8B</td><td>Visual-cue</td><td>24.4</td><td>24.4</td><td>0.246</td><td>1.33</td><td>5.5</td></tr><tr><td>InternVL3.5-8B</td><td>Diwani-first</td><td>21.1</td><td>28.4</td><td>0.228</td><td>0.10</td><td>0.0</td></tr><tr><td>InternVL3.5-38B</td><td>Numeric</td><td>35.8</td><td>20.0</td><td>0.134</td><td>0.22</td><td>89.9</td></tr><tr><td>InternVL3.5-38B</td><td>Visual-cue</td><td>34.1</td><td>38.6</td><td>0.370</td><td>1.53</td><td>56.2</td></tr><tr><td>InternVL3.5-38B</td><td>Diwani-first</td><td>33.7</td><td>35.8</td><td>0.334</td><td>1.52</td><td>60.8</td></tr><tr><td>MiniCPM-V 4.5</td><td>Numeric</td><td>31.6</td><td>32.1 30.8</td><td>0.259</td><td>1.09</td><td>43.8</td></tr><tr><td></td><td>Visual-cue</td><td>40.0</td><td></td><td>0.331</td><td>1.17</td><td>42.4</td></tr><tr><td>MiniCPM-V 4.5</td><td></td><td>20.2</td><td>38.7 19.6</td><td>0.381 0.111</td><td>1.12</td><td>55.3</td></tr><tr><td>MiniCPM-V 4.5</td><td>Diwani-first Numeric</td><td>47.2</td><td>49.4</td><td>0.454</td><td>4.94 2.89</td><td>0.5</td></tr><tr><td>Gemma 4 12B-IT Gemma 4 12B-IT</td><td>Visual-cue</td><td>44.1</td><td>46.1</td><td>0.419</td><td>3.28</td><td>6.5 0.5</td></tr><tr><td>Gemma 4 12B-IT</td><td>Diwani-first</td><td>43.8</td><td>45.4</td><td>0.415</td><td>3.35</td><td>0.0</td></tr></table>

Table 4: Original DuwatBench-only context runs on the clean six-class subset. These are not part of the main 36-run comparison. Bold marks the best value in each metric column.
<table><tr><td>Model</td><td>Prompt/run</td><td>n</td><td>Accuracy ↑</td><td>Balanced acc. ↑</td><td>Macro-F1 ↑</td><td>Thuluth ratio</td><td>Diwani recall ↑</td></tr><tr><td>GPT-40</td><td>Closed-set B</td><td>1252</td><td>65.9</td><td>45.9</td><td>0.475</td><td>1.27</td><td>30.8</td></tr><tr><td>GPT-5.5</td><td>Closed-set B</td><td>1252</td><td>71.9</td><td>52.7</td><td>0.576</td><td>1.32</td><td>29.0</td></tr><tr><td>GPT-40</td><td>Alt wording C</td><td>1252</td><td>49.3</td><td>41.6</td><td>0.368</td><td>0.55</td><td>85.3</td></tr><tr><td>GPT-5.5</td><td>Alt wording C</td><td>1252</td><td>70.1</td><td>43.9</td><td>0.469</td><td>1.25</td><td>47.3</td></tr><tr><td>GPT-40</td><td>Label-order D</td><td>1252</td><td>63.0</td><td>49.4</td><td>0.492</td><td>1.08</td><td>43.3</td></tr><tr><td>GPT-5.5</td><td>Label-order D</td><td>1252</td><td>72.2</td><td>53.3</td><td>0.586</td><td>1.32</td><td>31.7</td></tr><tr><td>GPT-5.5</td><td>Pilot open-ended</td><td>1252</td><td>69.5</td><td>50.1</td><td>0.538</td><td>1.09</td><td>59.4</td></tr></table>