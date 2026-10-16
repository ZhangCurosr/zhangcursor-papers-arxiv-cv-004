# Visual Sanad: Auditing Vision–Language Models for False Acceptance of Qur’anic Citations

Niaz Mohaiman Abtahi<sup>1</sup>, Reyana Shahid<sup>2</sup>, Evan Maksud<sup>1</sup>

<sup>1</sup>Advanced AI Lab Limited

<sup>2</sup>United International University

Correspondence to: tahiniaz@gmail.com

GitHub/Visual-Sanad

## Abstract

Vision–language models (VLMs) are increasingly used to interpret religious, educational, and culturally sensitive visual content, yet their reliability on citation verification remains underexplored. We introduce Visual Sanad, a controlled benchmark for testing whether VLMs falsely accept incorrect Qur’anic verse attributions when the same quote–reference pair is rendered under different visual framings. The benchmark contains 2,400 rendered images built from 100 English Qur’an verses, four factual states per verse, and six visual conditions ranging from plain text to decorative and authority-framed layouts. All quote–reference pairs were source-checked and human-audited before rendering. Across 24 full model– prompt runs, we observe a sharp capability split. Proprietary API-served VLMs range from near-perfect citation verification to moderate false acceptance, whereas recent open-weight VLMs often show weak discrimination and default to VALID on most images. De-biased prompting can reduce false acceptance, but often by shifting the decision threshold toward INVALID rather than by producing strong citation discrimination. Visual Sanad therefore exposes a culturally grounded multimodal factuality failure: fluent VLMs may treat displayed religious citations as self-authenticating unless they have strong citation-verification ability.

## 1 Introduction

Religious texts are often encountered through images: posters, screenshots, lecture slides, social media cards, and educational graphics. For Muslim communities, Qur’anic quotation and attribution are not merely factual details; they shape trust, teaching, and religious interpretation. A system that reads a polished image and incorrectly validates a verse attribution may therefore cause a culturally specific form of harm.

This paper studies a narrow but important question: when an image contains an English Qur’an quote and a displayed surah:ayah reference, can modern VLMs verify whether the attribution is correct? More specifically, do visual cues such as decorative Islamic-style layouts or institutional “verified citation” headers change the probability that a model accepts a false attribution? This question sits at the intersection of rapid progress in VLMs [Radford et al., 2021, Alayrac et al., 2022, Liu et al., 2023, OpenAI, 2023] and growing evidence that multimodal systems can hallucinate or over-trust misleading inputs [Li et al., 2023, Bai et al., 2024]. Unlike object-hallucination benchmarks, Visual Sanad tests a fine-grained attribution decision in a culturally grounded religious setting, aligning with MusiML’s emphasis on AI systems relevant to Muslim communities [Muslims in ML, 2026]. Prior Islamic NLP benchmarks study Qur’anic question answering and broader Islamic knowledge evaluation [Malhas et al., 2023, Abdelaal et al., 2026]; Visual Sanad instead asks whether a VLM can verify a displayed quote–reference attribution in an image.

Our contributions are:

• a controlled 2,400-image benchmark for Qur’anic citation verification under six visual conditions;

• a human-audited construction pipeline using valid references and three invalid foil types;

40th Conference on Neural Information Processing Systems (NeurIPS 2026). Workshop: MusIML@NeurIPS 2026.

• baseline evaluations of four closed VLMs and eight open VLMs under natural and de-biased prompts; and

• paired analyses measuring visual framing effects and prompt-debiasing effects on the same underlying invalid items.

Our central finding is not that visual authority cues overwhelm every model. Rather, vulnerable VLMs already struggle with hard citation foils, and visual framing can shift those errors in systematic ways. This distinction matters: the benchmark is both a factuality test and a probe of multimodal presentation bias.

## 2 Dataset: Visual Sanad

Source text. We use the AlQuran Cloud en.sahih edition as the source for English verse text and references [Islamic Network, 2026]. Bracket characters from Sahih International translator glosses are removed while preserving the words inside them. Thus, the displayed text is a plain-text rendering of the source rather than a bracket-preserving reproduction.

Base verses and factual states. We select 100 base verses after filtering for readability and dupli cate safety. Each base verse is paired with four factual states:

1. Valid: the displayed reference is the true reference.

2. Near-neighbor: the displayed reference is a nearby ayah in the same surah.

3. Same-surah far: the displayed reference is a farther ayah in the same surah.

4. Cross-surah lexical: the displayed reference is from another surah with lexical overlap.

Invalid foils are screened with exact, character-level, and asymmetric token-containment checks to avoid cases where the target quote is effectively embedded in the foil.

Visual conditions. Each quote–reference pair is rendered under six image conditions: plain, decorative, islamic\_authority, decorative\_islamic\_authority, neutral\_authority, and decorative\_neutral\_authority. Authority conditions add a header such as “Verified Qur’anic Citation” or “Verified Citation Record”; decorative conditions add a patterned background. The final dataset has 100 × 4 × 6 = 2,400 images. This yields 600 valid images and 1,800 invalid images, so raw accuracy must be read alongside false acceptance, false rejection, and label bias.

Audit. The dataset was audited in three passes before locking. Early audits found source-text corruption from deleting bracketed words and unsafe near-duplicate foils. The final locked version contains no source issues and no unsafe gold labels after replacing three quality-flagged foils. Rendered images were checked for file completeness, dimensions, and a visual sample for readability.

## 3 Experimental Setup

Task. Models receive one rendered image and must return VALID, INVALID, or ABSTAIN. The main metric is false-accept rate on invalid trials: the percentage of invalid quote–reference pairs labeled VALID. We also report false-reject rate on valid trials and discrimination, computed as 100 − false accept − false reject; a constant VALID model scores near 25% raw accuracy but near-zero discrimination. Because each underlying item appears in all six visual conditions, we also compute paired flip counts against the plain condition.

Prompts. We use two prompts. The natural prompt asks the model to decide whether the displayed quote belongs to the displayed reference. The de-biased prompt adds one cautionary instruction: “Do not treat decorative design, religious-looking borders, or verification headers as evidence that the citation is correct.” The natural prompt approximates ordinary use; the de-biased prompt tests whether explicit caution reduces cue sensitivity.

Models and decoding. We evaluate GPT-4o, GPT-4.1, GPT-5.5, GPT-5.6-Terra [OpenAI, 2026], Qwen2.5-VL-7B/32B [Bai et al., 2025b], Qwen3-VL-8B/32B [Bai et al., 2025a], InternVL3.5- 8B/38B [Wang et al., 2025], MiniCPM-V 4.5 [Yu et al., 2025], and Gemma 4 12B-IT [Gemma Team, 2026]. Each completed model–prompt pair is run over all 2,400 images. Open-source runs use greedy deterministic decoding with do\_sample=False and fixed seed 31. Empty or unparsed API responses are retried and merged into clean result files; all main runs reported below contain no errors and no unparsed labels.

Table 1: Overall invalid-citation false acceptance, discrimination, and response bias on the locked 2,400-image dataset. Values are percentages. Discrimination is 100 − false accept − false reject, so higher is better; false accept is lower better. Says VALID is diagnostic and has no preferred direction.
<table><tr><td>Model</td><td>Natural</td><td>Natural false accept ↓ discrimination ↑ says VALID</td><td>Natural</td><td>De-biased false accept ↓</td><td>De-biased discrimination ↑</td><td>De-biased says VALID</td></tr><tr><td colspan="7">Closed/API VLMs</td></tr><tr><td>GPT-40</td><td>14.89</td><td>83.44</td><td>35.8</td><td>13.50</td><td>84.50</td><td>34.6</td></tr><tr><td>GPT-4.1</td><td>15.94</td><td>83.73</td><td>36.9</td><td>14.39</td><td>85.44</td><td>35.8</td></tr><tr><td>GPT-5.6-Terra</td><td>4.72</td><td>95.11</td><td>28.5</td><td>4.72</td><td>95.11</td><td>28.5</td></tr><tr><td>GPT-5.5</td><td>0.00</td><td>99.83</td><td>25.0</td><td>0.00</td><td>99.67</td><td>24.9</td></tr><tr><td colspan="7"></td></tr><tr><td>Open-weight VLMs Qwen2.5-VL-7B</td><td>96.28</td><td>0.89</td><td>96.5</td><td>92.89</td><td>0.11</td><td>92.9</td></tr><tr><td>Qwen3-VL-8B</td><td>83.44</td><td>3.23</td><td>84.2</td><td>80.78</td><td>3.89</td><td>81.8</td></tr><tr><td>Qwen2.5-VL-32B</td><td>92.72</td><td>5.11</td><td>94.0</td><td>91.94</td><td>5.73</td><td>93.4</td></tr><tr><td>Qwen3-VL-32B</td><td>52.17</td><td>12.16</td><td>55.2</td><td>43.17</td><td>10.83</td><td>45.9</td></tr><tr><td>InternVL3.5-8B</td><td>94.00</td><td>0.67</td><td>94.2</td><td>89.94</td><td>1.56</td><td>90.3</td></tr><tr><td>InternVL3.5-38B</td><td>81.56</td><td>7.61</td><td>83.5</td><td>48.06</td><td>9.11</td><td>50.2</td></tr><tr><td>MiniCPM-V 4.5</td><td>99.17</td><td>0.50</td><td>99.3</td><td>96.39</td><td>0.78</td><td>96.6</td></tr><tr><td>Gemma 4 12B-IT</td><td>98.00</td><td>0.50</td><td>98.1</td><td>96.00</td><td>2.00</td><td>96.5</td></tr></table>

## 4 Results

Table 1 summarizes the main model comparison; Figure 1 visualizes natural-prompt false acceptance and prompt sensitivity. False accept is computed over 1,800 invalid images per run, pooling near-neighbor, same-surah-far, and cross-surah lexical foils; Appendix Table 3 breaks this down by foil type. Closed models show a large capability gradient: GPT-4o and GPT-4.1 have substantial false acceptance, GPT-5.6-Terra improves sharply, and GPT-5.5 is nearly saturated. Open models show a different failure mode: several produce clean labels while saying VALID on 90–99% of images, yielding near-zero discrimination despite near-perfect displayed-reference reading.

Visual cue effects. Visual framing effects are smaller than the model-capability effect, but they are not negligible. For GPT-4o under the natural prompt, false acceptance is 13.00% on plain invalid images and rises to 17.33% for decorative neutral-authority images. For Qwen3 models, however, authority wording can make the model more skeptical: Qwen3-VL-32B drops from 51.33% on plain images to 34.00% under Islamic-authority framing, while decorative framing raises false acceptance to 66.33%. For InternVL3.5-38B, authority wording is especially consequential: natural-prompt false acceptance is 68.67% on plain invalid images but 89.33% under both Islamic-authority and neutral-authority headers. For MiniCPM-V 4.5 and Gemma 4 12B-IT, false acceptance is already near ceiling across conditions. Appendix Figure 3 and Table 4 show the full condition-wise breakdown.

Prompt effects. The de-biased instruction reduces false acceptance for several models under paired McNemar tests [McNemar, 1947] on invalid items, but Table 1 shows why this is not the same as solving citation verification. For example, Qwen3-VL-32B’s de-biased false-accept rate improves from 52.17% to 43.17%, while discrimination falls from 12.16 to 10.83 because false rejection increases. Prompting therefore mostly changes the decision threshold for weaker models; it helps reduce over-acceptance but does not replace citation-level verification.

## 5 Discussion and Limitations

Visual Sanad supports a concrete claim: multimodal citation verification in Islamic contexts remains nontrivial for several widely used VLMs. The strongest models can solve the current benchmark, but many open-weight models have little ability to distinguish correct from incorrect attributions and instead accept most inputs as valid. Visual cue effects are real but smaller than capability differences, and their direction is model-family dependent: decorative cues often increase false acceptance, while authority wording can either induce skepticism or increase trust.

The study has limitations: First, the benchmark uses English translations, not Arabic Qur’anic text. Second, it evaluates rendered static images rather than real social-media screenshots. Third, al though we include several larger open VLMs, the set is still not exhaustive and fast-moving model

![](images/6daa53330ab0f336e1bb4cef812ffd1b38ad1ced8148efd7a1aa120fd2bf735b.jpg)  
(b)

![](images/94b6b9db1d5e6e2373a20344c1047f78b5bf059b02a3b64d1eeaaf3853291d31.jpg)

Figure 1: Invalid-citation false acceptance. (a) Natural-prompt failures show a sharp closed/open split. (b) De-biased prompting reduces false acceptance for some models but does not replace verification.

releases may change the frontier. Fourth, the dataset is clustered by verse, so downstream statistical models should include verse-level random effects rather than treating 2,400 images as independent.

Implications. For deployment, the main lesson is that citation verification should not be treated as a generic visual question-answering capability. Systems that answer religious citation questions from screenshots should either retrieve the referenced verse from a canonical source before judging the image, or abstain when they cannot verify the attribution. Visual Sanad can therefore serve as a lightweight regression test for retrieval-augmented VLMs, moderation tools, educational assistants, and community-facing fact-checking systems.

Community relevance. The failure mode matters because religious citations often circulate as images in messaging apps, lecture slides, posters, and social media, where a model that confidently validates an incorrect attribution may amplify rather than correct misinformation. By focusing on Qur’anic references, Visual Sanad tests whether multimodal systems respect community norms around exact source attribution.

Reproducibility. We retain the locked manifest, prompts, audit notes, inference outputs, and analysis scripts used for the reported results. Open-source runs use deterministic decoding, and API runs are reported from cleaned full-dataset CSVs.

## 6 Conclusion

Visual Sanad shows that Qur’anic citation verification in images is a culturally grounded test of multimodal factuality: strong models can solve it, while many open-weight VLMs show weak discrimination, high VALID bias, and model-dependent cue sensitivity.

Acknowledgments. This work received no external funding. The authors declare no competing interests.

## References

Ali Abdelaal, Mohammed Nader Al Haffar, Mahmoud Fawzi, and Walid Magdy. IslamicMMLU: A benchmark for evaluating llms on islamic knowledge. arXiv preprint arXiv:2603.23750, 2026.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 2022.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024.

Gemma Team. Gemma 4 technical report. arXiv preprint arXiv:2607.02770, 2026.

Islamic Network. AlQuran Cloud API. https://alquran.cloud/api, 2026. Accessed 2026-08- 30.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems, 2023.

Rana Malhas, Watheq Mansour, and Tamer Elsayed. Qur’an QA 2023 Shared Task: Overview of Passage Retrieval and Reading Comprehension Tasks over the Holy Qur’an. In Proceedings of ArabicNLP 2023, 2023. URL https://aclanthology.org/2023.arabicnlp-1.76/.

Quinn McNemar. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2):153–157, 1947. doi: 10.1007/BF02295996.

Muslims in ML. Call for Papers: MusIML@NeurIPS2026. https://www.musiml.org/events/ 2026-NeurIPS/index.html, 2026.

OpenAI. GPT-4V(ision) System Card. https://openai.com/research/ gpt-4v-system-card, 2023.

OpenAI. OpenAI API Model Catalog. https://platform.openai.com/docs/models, 2026. Accessed 2026-09-03.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning, 2021.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Yuanqian Zhao, et al. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe. arXiv preprint arXiv:2509.18154, 2025.

## A Dataset construction details

Visual Sanad is constructed from 100 base verses in the AlQuran Cloud en.sahih edition. Each base verse is expanded into four factual states and six visual conditions, producing 2,400 rendered images.

Factual states. For every base verse, we create one valid item and three invalid foils:

• Valid: the displayed quote and displayed reference match.

• Near-neighbor: the reference points to a nearby ayah in the same surah.

• Same-surah far: the reference points to a farther ayah in the same surah.

• Cross-surah lexical: the reference points to another surah with lexical overlap.

Foil safety filters. Invalid candidates are filtered with exact-string checks, normalized characteroverlap checks, and asymmetric token-containment checks. This prevents cases where the displayed quote is effectively contained in the foil verse, which would make an invalid label ambiguous.

Visual conditions. The six conditions are plain, decorative, islamic\_authority, decorative\_islamic\_authority, neutral\_authority, and decorative\_neutral\_authority. Authority headers use either religiously specific wording, such as “Verified Qur’anic Citation,” or neutral institutional wording, such as “Verified Citation Record.”

## B Audit summary

The final audit found no source-text corruption and no unsafe gold labels after corrective replacements. The most important construction fix was preserving translator-gloss words when removing bracket characters; earlier drafts had accidentally deleted bracketed content. The final locked dataset also removed or replaced three quality-flagged foils before rendering.

## C Example rendered items

![](images/35b3bf27d2c8c4e92aaf50ab2edbbde84f9a99aa2866bdf94dcd2f6938753947.jpg)  
Figure 2: Example Visual Sanad renderings: valid plain item; invalid plain item; invalid item with decorative Islamic-authority framing.

## D Prompts

The prompt text below is line-wrapped for page layout.

## Natural prompt.

You are evaluating an image that displays an English Qur’an quote and a surah:ayah reference.

Task: decide whether the displayed quote belongs to the displayed Qur’an reference.

Return only a compact JSON object with exactly these keys:

"label": "VALID" | "INVALID" | "ABSTAIN",   
"confidence": "low" | "medium" | "high"   
}   
Use VALID only if the displayed quote matches the displayed surah:ayah   
reference.   
Use INVALID if the quote is Qur’anic but belongs to a different surah:ayah   
reference.   
Use ABSTAIN only if the image is unreadable or you cannot determine the answer.

De-biased prompt. The de-biased prompt uses the same text as above, followed by:

Do not treat decorative design, religious-looking borders, or verification headers as evidence that the citation is correct.

## E Full model results

Table 2 reports all completed model–prompt runs used in the analysis. Displayed-reference extraction was near ceiling in these runs: the model-reported reference matched the rendered reference in at least 99.83% of rows for every reported run, and most runs reached 100%. Abstention was also rare, never exceeding 0.62% in any run; for example, MiniCPM-V 4.5 under the natural prompt produced high-confidence errors on 74.5% of all images while abstaining on none.

Table 2: Full model–prompt summary. Accuracy, invalid-citation false accept, valid-citation false reject, discrimination, and VALID response rate are percentages. Discrimination is 100−false accept− false reject; arrows mark the preferred direction for metric columns. Says VALID is diagnostic and has no preferred direction.
<table><tr><td>Model</td><td>Prompt</td><td>Rows</td><td>Errors</td><td>Unparsed</td><td>Accuracy ↑</td><td>False accept ↓</td><td>False reject ↓</td><td>Discrimination ↑</td><td>Says VALID</td></tr><tr><td>GPT-40</td><td>natural</td><td>2400</td><td>0</td><td>0</td><td>88.42</td><td>14.89</td><td>1.67</td><td>83.44</td><td>35.8</td></tr><tr><td>GPT-40</td><td>de-biased</td><td>2400</td><td>0</td><td>0</td><td>89.38</td><td>13.50</td><td>2.00</td><td>84.50</td><td>34.6</td></tr><tr><td>GPT-4.1</td><td>natural</td><td>2400</td><td>0</td><td>0</td><td>87.96</td><td>15.94</td><td>0.33</td><td>83.73</td><td>36.9</td></tr><tr><td>GPT-4.1</td><td>de-biased</td><td>2400</td><td>0</td><td>0</td><td>89.17</td><td>14.39</td><td>0.17</td><td>85.44</td><td>35.8</td></tr><tr><td>GPT-5.5</td><td>natural</td><td>2400</td><td>0</td><td>0</td><td>99.96</td><td>0.00</td><td>0.17</td><td>99.83</td><td>25.0</td></tr><tr><td>GPT-5.5</td><td>de-biased</td><td>2400</td><td>0</td><td>0</td><td>99.92</td><td>0.00</td><td>0.33</td><td>99.67</td><td>24.9</td></tr><tr><td>GPT-5.6 Terra</td><td>natural</td><td>2400</td><td>0</td><td>0</td><td>96.42</td><td>4.72</td><td>0.17</td><td>95.11</td><td>28.5</td></tr><tr><td>GPT-5.6 Terra</td><td>de-biased</td><td>2400</td><td>0</td><td>0</td><td>96.42</td><td>4.72</td><td>0.17</td><td>95.11</td><td>28.5</td></tr><tr><td>Qwen2.5-VL-7B</td><td>natural</td><td>2400</td><td>0</td><td>0</td><td>27.08</td><td>96.28</td><td>2.83</td><td>0.89</td><td>96.5</td></tr><tr><td>Qwen2.5-VL-7B</td><td>de-biased</td><td>2400</td><td>0</td><td>0</td><td>28.46</td><td>92.89</td><td>7.00</td><td>0.11</td><td>92.9</td></tr><tr><td>Qwen3-VL-8B</td><td>natural</td><td>2400</td><td>0</td><td>0</td><td>34.08</td><td>83.44</td><td>13.33</td><td>3.23</td><td>84.2</td></tr><tr><td>Qwen3-VL-8B</td><td>de-biased</td><td>2400</td><td>0</td><td>0</td><td>35.58</td><td>80.78</td><td>15.33</td><td>3.89</td><td>81.8</td></tr><tr><td>InternVL3.5-8B</td><td>natural</td><td>2400</td><td>0</td><td>0</td><td>27.92</td><td>94.00</td><td>5.33</td><td>0.67</td><td>94.2</td></tr><tr><td>InternVL3.5-8B</td><td>de-biased</td><td>2400</td><td>0</td><td>0</td><td>30.42</td><td>89.94</td><td>8.50</td><td>1.56</td><td>90.3</td></tr><tr><td>MiniCPM-V 4.5</td><td>natural</td><td>2400</td><td>0</td><td>0</td><td>25.54</td><td>99.17</td><td>0.33</td><td>0.50</td><td>99.3</td></tr><tr><td>MiniCPM-V 4.5</td><td>de-biased</td><td>2400</td><td>0</td><td>0</td><td>27.00</td><td>96.39</td><td>2.83</td><td>0.78</td><td>96.6</td></tr><tr><td>Gemma 4 12B-IT</td><td>natural</td><td>2400</td><td>0</td><td>0</td><td>26.12</td><td>98.00</td><td>1.50</td><td>0.50</td><td>98.1</td></tr><tr><td>Gemma 4 12B-IT</td><td>de-biased</td><td>2400</td><td>0</td><td>0</td><td>27.50</td><td>96.00</td><td>2.00</td><td>2.00</td><td>96.5</td></tr><tr><td>Qwen2.5-VL-32B</td><td>natural</td><td>2400</td><td>0</td><td>0</td><td>29.92</td><td>92.72</td><td>2.17</td><td>5.11</td><td>94.0</td></tr><tr><td>Qwen2.5-VL-32B</td><td>de-biased</td><td>2400</td><td>0</td><td>0</td><td>30.46</td><td>91.94</td><td>2.33</td><td>5.73</td><td>93.4</td></tr><tr><td>Qwen3-VL-32B</td><td>natural</td><td>2400</td><td>0</td><td>0</td><td>51.96</td><td>52.17</td><td>35.67</td><td>12.16</td><td>55.2</td></tr><tr><td>Qwen3-VL-32B</td><td>de-biased</td><td>2400</td><td>0</td><td>0</td><td>56.12</td><td>43.17</td><td>46.00</td><td>10.83</td><td>45.9</td></tr><tr><td>InternVL3.5-38B</td><td>natural</td><td>2400</td><td>0</td><td>0</td><td>36.04</td><td>81.56</td><td>10.83</td><td>7.61</td><td>83.5</td></tr><tr><td>InternVL3.5-38B</td><td>de-biased</td><td>2400</td><td>0</td><td>0</td><td>52.62</td><td>48.06</td><td>42.83</td><td>9.11</td><td>50.2</td></tr></table>

## F Foil-type false acceptance

Table 3 breaks natural-prompt false acceptance into the three invalid foil types. Each cell is computed over 600 invalid images: 100 base verses rendered under six visual conditions.

Table 3: Natural-prompt invalid false-accept rate by foil type. Values are percentages; lower is better.
<table><tr><td>Model</td><td>Near-neighbor ↓</td><td>Same-surah far ↓</td><td>Cross-surah lexical ↓</td></tr><tr><td>GPT-40</td><td>17.67</td><td>2.83</td><td>24.17</td></tr><tr><td>GPT-4.1</td><td>20.17</td><td>3.00</td><td>24.67</td></tr><tr><td>GPT-5.6 Terra</td><td>10.50</td><td>1.67</td><td>2.00</td></tr><tr><td>GPT-5.5</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Qwen3-VL-8B</td><td>85.00</td><td>86.50</td><td>78.83</td></tr><tr><td>Qwen3-VL-32B</td><td>61.17</td><td>56.67</td><td>38.67</td></tr><tr><td>InternVL3.5-8B</td><td>94.50</td><td>94.83</td><td>92.67</td></tr><tr><td>InternVL3.5-38B</td><td>87.33</td><td>83.83</td><td>73.50</td></tr><tr><td>MiniCPM-V 4.5</td><td>99.17</td><td>99.00</td><td>99.33</td></tr><tr><td>Gemma 4 12B-IT</td><td>99.33</td><td>97.83</td><td>96.83</td></tr></table>

## G Condition-wise false acceptance

Figure 3 visualizes condition-wise false acceptance for selected models, and Table 4 reports the full condition-wise breakdown.

![](images/bc4482f229dca28b2c7708a9fd88e1a25c9accd24d1a202924058f9080858d44.jpg)  
Figure 3: Condition-wise invalid false-accept rate under the natural prompt for selected models. Visual framing effects vary by model and are smaller than the overall capability gap, but authority and decoration can still shift false acceptance.

Table 4: Invalid false-accept rate by visual condition. Values are percentages; lower is better. Bold marks the best value in each condition column.
<table><tr><td>Model</td><td>Prompt</td><td>Plain ↓</td><td>Decorative ↓</td><td>Islamic authority ↓</td><td>Decorative + Islamic ↓</td><td>Neutral authority ↓</td><td>Decorative + Neutral ↓</td></tr><tr><td>GPT-40</td><td>natural</td><td>13.00</td><td>16.00</td><td>14.67</td><td>15.67</td><td>12.67</td><td>17.33</td></tr><tr><td>GPT-40</td><td>de-biased</td><td>13.33</td><td>15.00</td><td>11.33</td><td>15.67</td><td>12.67</td><td>13.00</td></tr><tr><td>GPT-4.1</td><td>natural</td><td>15.00</td><td>16.00</td><td>16.33</td><td>16.00</td><td>15.00</td><td>17.33</td></tr><tr><td>GPT-4.1</td><td>de-biased</td><td>13.33</td><td>13.33</td><td>15.00</td><td>14.33</td><td>14.33</td><td>16.00</td></tr><tr><td>GPT-5.5</td><td>natural</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>GPT-5.5</td><td>de-biased</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>GPT-5.6 Terra</td><td>natural</td><td>5.00</td><td>6.00</td><td>4.00</td><td>6.33</td><td>2.33</td><td>4.67</td></tr><tr><td>GPT-5.6 Terra</td><td>de-biased</td><td>5.00</td><td>6.00</td><td>4.00</td><td>6.33</td><td>2.33</td><td>4.67</td></tr><tr><td>Qwen2.5-VL-7B</td><td>natural</td><td>95.00</td><td>95.67</td><td>96.00</td><td>98.33</td><td>96.00</td><td>96.67</td></tr><tr><td>Qwen2.5-VL-7B</td><td>de-biased</td><td>95.67</td><td>89.67</td><td>94.33</td><td>89.67</td><td>95.33</td><td>93.67</td></tr><tr><td>Qwen3-VL-8B</td><td>natural</td><td>89.33</td><td>91.67</td><td>78.00</td><td>83.00</td><td>75.33</td><td>83.33</td></tr><tr><td>Qwen3-VL-8B</td><td>de-biased</td><td>91.67</td><td>89.67</td><td>75.33</td><td>78.67</td><td>72.67</td><td>76.67</td></tr><tr><td>InternVL3.5-8B</td><td>natural</td><td>96.00</td><td>95.67</td><td>94.00</td><td>95.00</td><td>90.33</td><td>93.00</td></tr><tr><td>InternVL3.5-8B</td><td>de-biased</td><td>91.67</td><td>91.67</td><td>90.67</td><td>91.67</td><td>85.67</td><td>88.33</td></tr><tr><td>MiniCPM-V 4.5</td><td>natural</td><td>99.67</td><td>99.67</td><td>98.67</td><td>99.67</td><td>98.33</td><td>99.00</td></tr><tr><td>MiniCPM-V 4.5</td><td>de-biased</td><td>99.00</td><td>96.67</td><td>97.33</td><td>95.67</td><td>98.33</td><td>91.33</td></tr><tr><td>Gemma 4 12B-IT</td><td>natural</td><td>98.00</td><td>98.00</td><td>98.00</td><td>98.33</td><td>98.00</td><td>97.67</td></tr><tr><td>Gemma 4 12B-IT</td><td>de-biased</td><td>98.00</td><td>98.00</td><td>98.00</td><td>94.67</td><td>96.67</td><td>90.67</td></tr><tr><td>Qwen2.5-VL-32B</td><td>natural</td><td>89.33</td><td>92.67</td><td>94.00</td><td>95.67</td><td>92.33</td><td>92.33</td></tr><tr><td>Qwen2.5-VL-32B</td><td>de-biased</td><td>89.00</td><td>92.33</td><td>93.00</td><td>94.67</td><td>91.67</td><td>91.00</td></tr><tr><td>Qwen3-VL-32B</td><td>natural</td><td>51.33</td><td>66.33</td><td>34.00</td><td>60.33</td><td>40.67</td><td>60.33</td></tr><tr><td>Qwen3-VL-32B</td><td>de-biased natural</td><td>42.33 68.67</td><td>57.67 76.00</td><td>28.33 89.33</td><td>51.33 87.00</td><td>33.33</td><td>46.00</td></tr><tr><td>InternVL3.5-38B</td><td></td><td></td><td>34.33</td><td></td><td>53.00</td><td>89.33</td><td>79.00</td></tr><tr><td>InternVL3.5-38B</td><td>de-biased</td><td>42.00</td><td></td><td>65.33</td><td></td><td>58.33</td><td>35.33</td></tr></table>

Table 5: Selected paired visual-cue effects under the natural prompt. Net induced is cue-only minus plain-only false accepts over the same 300 invalid items; positive values increase false acceptance and negative values reduce it. Bold marks the largest shifts.
<table><tr><td>Model</td><td>Cue comparison</td><td>Net induced</td><td>McNemar p ↓</td></tr><tr><td>GPT-40</td><td>decorative + neutral</td><td>13</td><td>0.00235</td></tr><tr><td>Qwen3-VL-8B</td><td>Islamic authority</td><td>-34</td><td> $5 . 6 5 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Qwen3-VL-8B</td><td>neutral authority</td><td>-42</td><td> $1 . 2 8 \times 1 0 ^ { - 9 }$ </td></tr><tr><td>Qwen3-VL-32B</td><td>decorative</td><td>45</td><td> $6 . 2 7 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Qwen3-VL-32B</td><td>Islamic authority</td><td>-52</td><td> $5 . 7 5 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>InternVL3.5-38B</td><td>Islamic authority</td><td>62</td><td> $8 . 4 7 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td>InternVL3.5-38B</td><td>neutral authority</td><td>62</td><td> $2 . 2 6 \times 1 0 ^ { - 1 6 }$ </td></tr></table>

## H Paired prompt effects

Table 6 reports paired natural-versus-de-biased prompt effects on the same invalid items.

Table 6: Paired prompt effect on invalid-citation false accepts. Positive net rescue means the debiased prompt corrected false accepts made under the natural prompt. Lower McNemar p indicates stronger evidence of asymmetric paired flips.
<table><tr><td>Model</td><td>Invalid pairs</td><td>Natural-only false accept ↓</td><td>De-biased-only false accept ↓</td><td>Net rescue ↑</td><td>McNemar p ↓</td></tr><tr><td>GPT-4.1</td><td>1800</td><td>50</td><td>22</td><td>28</td><td>0.00129</td></tr><tr><td>GPT-40</td><td>1800</td><td>57</td><td>32</td><td>25</td><td>0.01055</td></tr><tr><td>GPT-5.5</td><td>1800</td><td>0</td><td>0</td><td>0</td><td>1.00000</td></tr><tr><td>GPT-5.6 Terra</td><td>1800</td><td>51</td><td>51</td><td>0</td><td>1.00000</td></tr><tr><td>Gemma 4 12B-IT</td><td>1800</td><td>36</td><td>0</td><td>36</td><td> $2 . 9 1 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>InternVL3.5-38B</td><td>1800</td><td>603</td><td>0</td><td>603</td><td> $< 1 0 ^ { - 1 0 }$ </td></tr><tr><td>InternVL3.5-8B</td><td>1800</td><td>73</td><td>0</td><td>73</td><td> $< 1 0 ^ { - 1 0 }$ </td></tr><tr><td>MiniCPM-V 4.5</td><td>1800</td><td>50</td><td>0</td><td>50</td><td> $< 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Qwen2.5-VL-32B</td><td>1800</td><td>17</td><td>3</td><td>14</td><td> $0 . 0 0 2 5 8$ </td></tr><tr><td>Qwen2.5-VL-7B</td><td>1800</td><td>61</td><td>0</td><td>61</td><td> $< 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Qwen3-VL-32B</td><td>1800</td><td>164</td><td>2</td><td>162</td><td> $< 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Qwen3-VL-8B</td><td>1800</td><td>50</td><td>2</td><td>48</td><td> $6 . 1 2 \times 1 0 ^ { - 1 3 }$ </td></tr></table>

Reading the paired table. Natural-only false accepts are invalid images accepted under the natural prompt but corrected under the de-biased prompt. De-biased-only false accepts are regressions introduced by the de-biased prompt. Net rescue is the difference between these two counts, while lower McNemar’s p reports stronger evidence that the paired flips are asymmetric rather than random prompt noise.