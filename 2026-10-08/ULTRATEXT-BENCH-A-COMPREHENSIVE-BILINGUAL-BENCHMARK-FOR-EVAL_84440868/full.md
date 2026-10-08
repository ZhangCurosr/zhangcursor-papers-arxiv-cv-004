# ULTRATEXT BENCH: A COMPREHENSIVE BILINGUAL BENCHMARK FOR EVALUATING VISUAL TEXT REN-DERING IN IMAGE GENERATION

Deyuan Liu<sup>1,†</sup> Yihao Hu<sup>1,2,†</sup> Jingxuan Zhang<sup>1,†</sup> Xingying Li<sup>1,†</sup> Jun Xie<sup>1,3,4,†</sup> Jiacheng Liu<sup>8</sup> Jungang Li<sup>5</sup> Yu Huang<sup>9</sup> Xuanyi Liu<sup>10</sup> Yue Ding<sup>6</sup> Zecheng Wang<sup>7</sup> Lei Zhao<sup>1</sup> Mingda Wang<sup>1</sup> Zhenglin Cheng<sup>1,3,4</sup> Peng Sun<sup>1,3</sup> Tao Lin<sup>1∗</sup>

<sup>1</sup>Westlake University <sup>2</sup>Ant Group <sup>3</sup>Zhejiang University <sup>4</sup>Shanghai Innovation Institute <sup>5</sup>HKUST <sup>6</sup>CASIA <sup>7</sup>Wechat AI <sup>8</sup>MBZUAI <sup>9</sup>CityU <sup>10</sup>Peking University

## ABSTRACT

Dense visual text requires image generators to reproduce long strings across multiple regions with correct placement and legibility. As short-string rendering improves, evaluation must test sustained performance across more demanding scenes. We introduce UltraText Bench, a bilingual benchmark for prompt-only generation of dense visual text. It contains 432 prompts spanning 24 real-world scene categories and three difficulty levels, split equally between English and Chinese. Each human-reviewed prompt supplies exact strings for four to twelve text regions, paired with structured references for their content, placement, and visual attributes. We use the Q-Judger vision-language model to assess each image against the complete reference, reporting text fidelity, text clarity, spatial quality, and scene quality. Across 24 model configurations, these dimensions reveal different strengths: Z-Image-Turbo gains 3.81 clarity points over Z-Image-Base while losing 14.76 fidelity points under the reported settings. Performance also varies with workload; Qwen-Image-2512’s English composite falls from 86.50 at L1 to 42.86 at L3. Ten participants took part in human evaluation of the automatic scores. Repository: https://github.com/LINs-lab/UltraText\_Bench.

## 1 INTRODUCTION

Text-to-image (T2I) models now produce images of high visual quality across diverse prompts (Rombach et al., 2022; Saharia et al., 2022). Visual text rendering requires legible, correctly spelled characters at their intended positions. Recent models have improved short-string rendering (Wu et al., 2025a; Labs, 2025; OpenAI, 2025; 2023; Esser et al., 2024). Evaluation must now examine whether this ability extends to hundreds or thousands of characters across multiple regions (Du et al., 2025; Zhang et al., 2025a). Posters, product packaging, documents, receipts, forms, and interfaces require this sustained performance (Peng et al., 2025): titles, body text, and small supporting regions must all remain correct and legible. A missing price or an incorrect opening time can change the information conveyed even when the overall composition and lettering look convincing.

To evaluate these applications, we treat visual text generation as a text reproduction task: all target strings are supplied in the prompt, and the model must render them in the specified regions. Evaluation requires a dense text load: prompts that request many regions carrying hundreds to thousands of target characters. A long passage tests sustained rendering within one block; distributed regions also test whether all requested text remains correctly placed. Whole-scene evaluation requires considering every requested region against a structured reference of its text, placement, and carrier, the physical surface or digital element bearing it. A correct title can coexist with missing, incorrect, or unreadable text elsewhere in the image. Figure 2 illustrates these errors alongside failures of placement and integration with the text carrier.

Existing benchmarks address several parts of this task. CVTG-2K evaluates multiple text instances with position and attribute descriptions (Du et al., 2025), and LongText-Bench targets longer English and Chinese text (Geng et al., 2025). STRICT reaches thousand-character sequences (Zhang et al., 2025b), while OCRGenBench includes dense page-level tasks (Zhang et al., 2025a). InfoTextBench is the closest comparison for dense bilingual text, with target-string lists and OCR- and VLM-based

![](images/0d32ea8d574e0adb1af6bc3aa85fd43dc81705829c0b40daca8c3c1518d066e1.jpg)

![](images/c93d092fca69b7fce6fe5d7e057ea4936f952731055ecf08e4440ad5b028d68b.jpg)

<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

<table><tr><td colspan="2">折温后服务点数： 106,180.00</td></tr><tr><td>减：费用及基塔减路 专说活0图</td><td>122,896.00 76,80.00</td></tr><tr><td>空国美器量</td><td>356,644.00</td></tr><tr><td>法用比例：</td><td>25%</td></tr><tr><td>快源：</td><td>11,920.00</td></tr></table>

<table><tr><td>一、服务点路食计 基础编务点 2动编6.点 s28,00m.mo 内日服66: 36,000.80 12,500.80</td></tr><tr><td>86M5.6: 6.80</td></tr><tr><td>长解出入的单内报务合数： 原始点图经缺 166,300.80 576,500:80</td></tr></table>

![](images/07abcc9d70774beb1e8a06b976ae63daff21dbfa5f29fbd9504517bdb6e9afc2.jpg)

![](images/9a81be31a36b36640de6a5cc200c0cd264b7432587577ed9b17a3b6257f5353a.jpg)

## Label (ZH/L3)

![](images/53a6dd556c7a8c0f42a48ea1c0449a6847c879cdbea198eab4a97595a0c3df62.jpg)

## Sign (EN/L3)

![](images/f74bb7ed6fe6c671c8b2e493805d7966385a4261129451ba3db3e3964f2f637f.jpg)

## SLEEP AND THE MODERN MIND: What Research Says About Rest, Attention and Memory

![](images/c1239c0fc47e00931fc4aeef474ed566e545da52a3e8a8579ceb70ed0952507a.jpg)

PRICTICALSLEPROUTINE

66pretting time fer slore ir a practial sest i an msood and daily safery

## Article (EN/L3)

##

##

## 潮汐盐屋·当代闽南海岸菜·主厨季节菜单

##

## Menu (ZH/L3)

## Poster (ZH/L3)

## 从零到一：百万级并发系统的微服务架构演进实践

![](images/a453acc3c255c7b10689b066de662edcbedc1c5c4f6154e0f590a9e9380c7374.jpg)

## 二. M4位

## 二、服务的分便测

![](images/df5df1d8a1d14223b041a719fa98012b4fc3ef85c0f6dc3d338ee87a5de2e36c.jpg)

## THE CITY TIMES

![](images/4ff654c881771129be148817e5bff7f1e0bccffb5154e2f511d330758e9013c9.jpg)

TWELVE TECHNOLOGY FIRMS SIGN JOINT CLIMATE PLEDGE FOR LOWER EMISSIONS BY 2030 Agreement Commits \$47 Billion to Clean Energy

![](images/52b9e0ccc14b9eed2ea21900613cce8691812c1742092fa58db450a3e7860943.jpg)

![](images/1dec8a1fe3d0cd92bff2b6bff890bf39a61dbdaca40b481b61f6377833dd1ac4.jpg)

![](images/f0b9431d5d13f57c0c03d3428707997b650e51433fe73b0e9393dc008de82a39.jpg)

\$34 MLL0N BUDGZT SHORTFALL-Saperandeat Warne Prograss Yeed Hanliag by Segtamber

![](images/fa6d1bc5cd13ef6fde4e8d4dea6a170cd4495779ae3e3d83e8f838690e271869.jpg)

## Newspaper (EN/L3)

THE OLIVE GROVE MISTRO - Fine HedEtHerMnean Dúning -LOCILOCE-AOEIE, LaORLIOK-SAFE

<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

## Receipt (EN/L3)

## 0.0 N54

<table><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>三</td></tr></table>

##

## Letter (EN/L3)

INVOICE | Holloway Cons N 唯00

<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

## DICGIO ND DLVERY:

##

<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

<table><tr><td></td></tr></table>

## Invoice (EN/L3)

## AI工程师交流群(256)

![](images/3a4ef122a3435e3ec6b4d367e1a741f0f38e014b769f74b5851de5259992154d.jpg)

![](images/d0af22e2509f63a58bfe4e5e53c3783a8d4aee77751b96673d0060b880494c92.jpg)

各位大使流社一下、微据2h/ran2-720用LowRarkTune

成者重复数速大多的情况

热据10万星，从开温数级集阵选们，数看7一下确实有不少重频约60

学印率矩声少7 728的模型L.cwvflar山Tume建议b在1e-到2e-4之间

自前字习车是3e-1，其共典到1.Se-4并保自同一拍证集，再对比下一物as典线，1

![](images/693d647b23b426888df10e5ad71955bacd0f2c43ac36c4df97dbc88105d98fab.jpg)

动得用映一个变量：把刷体配票和辅机种于也演群文件 方使大家复观。

在要接议期MEnHoo小范者SimHaoh做近安去里，10万条几分钟快测究了我之前过一个解本可以分学用

## Billboard (EN/L3)

![](images/c2ab885f3eeff9a9aa5c37f3ca28e5334fd612c7d09c681c6bc237c546ae14b0.jpg)

项别收到总格一下:1先洗案2降学可率3.关过evallo

## CANDIDATE-SAFE-IOT-01 | 工业物联阵平台技术总监/边缘系统方

![](images/1a2121008eb201cfd24ce130c5b2b01b9849af8b32af7c1d54650f1152433087.jpg)

## Resume (ZH/L3)

![](images/6bbb104ac2cc8cff7d53cb4dbff58f969034071b9646bf501565a1f9839575cf.jpg)

![](images/186813bfc107fb9b87b0f333a40ee07f6c49ad55ec07a3ef4b2622d77978d34b.jpg)

## Packaging (EN/L3)

![](images/f52ac7e2c43241b60d45ccc78631d713b5457c827b99399bb97de792555e2101.jpg)

##

!!!

## Webpage (ZH/L3)

![](images/4ccf3ea7b46e32cf9d79a6c04011761f50e12348fe1fbd92a301f2651be2df57.jpg)

## Schedule (EN/L3)

![](images/ac09f1acdf7c2c130710468d1dcf83f79f880ef3a664b0a7416d9c2245e4ea46.jpg)  
Caption (EN/L3)

## Slide (EN/L3)

## 年度会员积分结算中请表

<table><tr><td colspan="3" rowspan="2">二、良职分器松减单 年度基不提务减频：</td><td rowspan="2">60,00:00 61,856.00</td></tr><tr><td>专量服务征成合计：</td></tr><tr><td>M：25,344.00</td><td></td><td>4科</td><td>6,336.00</td></tr><tr><td>限作</td><td>1.,112.00</td><td>电电日</td><td>29,064.00</td></tr></table>

<table><tr><td colspan="2">学话号 (个增量) 三、专项理动机似 站编 三</td></tr><tr><td colspan="2">世期业员理 继线学3编8 4,800.08</td></tr><tr><td colspan="2">76.80.00 专项适电报的合计：</td></tr></table>

![](images/e9469f511076a231f3b9c745ba6256315e81f5ca686bca3b1bbc09af21e10d90.jpg)

## Social Media (ZH/L3)

## Form (ZH/L3)

![](images/2724d6eb9c7e4081df432e901bc4e07502d018437864a14ac1684d9da5837149.jpg)  
Dialogue (EN/L3)

## Certificate (ZH/L3)

![](images/a72f56916a4a10e35d98f391aea5b10ad80bceb60a5b03500d5e25f9bcbcdbeb.jpg)  
Comic (ZH/L3)

## Dashboard (EN/L3)

![](images/f7cf21ae101b1586f5b6b28c0c684cab825c05ea646df0cb791ecfcc289c2d66.jpg)

## Code (EN/L3)

![](images/1c19ec8778bede9202390b4336b32e91e6f0cc43ccebf202c118cca4a640fda1.jpg)  
Infographic (ZH/L3)

Figure 1: Scene coverage at L3. One selected output per category in taxonomy order, from GPT Image 2 at API quality Low. L3 denotes Extreme workload; EN/ZH denote English/Chinese.

evaluation (Xiang et al., 2026). UltraText Bench combines 24 scene categories with a uniform per-region reference and a shared rubric for text, layout, and scene integration. It keeps target

![](images/99ccc686e93e850af5403747370f30e639ffb070633d37550dd66d557bcfe621.jpg)  
Figure 2: Local success can hide whole-scene failures. This illustrative six-region scene contrasts a correct shop name with errors elsewhere. Left. The correctly rendered shop name, cropped. Center. The full scene, with five further requested regions marked. Each is missing, wrong, unreadable, misplaced, or legible but poorly integrated with its intended surface. Right. Dense text load and whole-scene evaluation cover the requested text.

strings linked to their intended visual roles across documents, physical signs, and digital interfaces.   
Section 2.2 compares these benchmarks and their evaluation protocols.

We introduce UltraText Bench, a bilingual benchmark built on the two requirements above. It contains 432 prompts spanning 24 real-world scene categories in six domains, at three difficulty levels and split equally between English and Chinese. Each prompt requests four to twelve text regions: individually annotated target-text components, such as a shop name or a multi-line menu. Each record pairs a generation prompt containing every target string with structured per-region ground truth (GT), used only for evaluation. Its 3 × 3 grid gives coarse locations; several regions can share a cell. Figure 1 illustrates all categories at the highest workload level, L3.

A VLM judge, Q-Judger from Qwen-Image-Bench (Li et al., 2026a), receives the image and complete reference, returning six image-level scores under our rubric. Accuracy and completeness form text fidelity; position and layout form spatial quality, with text clarity and scene quality retained separately. The reference tells the judge which content, locations, and carriers to inspect. Ten participants took part in human evaluation of the automatic scores ( Section 4.2 ).

Evaluation of 24 model configurations shows why these distinctions matter. The Z-Image and LLaDA Turbo variants receive higher clarity but lower fidelity scores than their Base counterparts. Performance also differs across workload levels and languages.

## Our main contributions are:

a. UltraText Bench: 432 bilingual prompts across 24 scene categories and three difficulty levels, with exact target strings and a uniform per-region reference for prompt-only generation.

b. Whole-scene evaluation: a shared rubric applied with Q-Judger to the image and complete reference, reporting four quality dimensions and evaluation coverage separately.

c. Experimental analysis: comparisons of 24 model configurations across dimensions, workloads, and languages, accompanied by human evaluation involving ten participants.

## 2 RELATED WORK

## 2.1 VISUAL TEXT GENERATION METHODS

Visual text generation methods differ in how they encode exact character sequences, how they obtain a layout, and how well text survives being encoded and decoded by the image model.

Explicit glyph and layout conditioning. GlyphControl (Yang et al., 2023) adds a ControlNet branch conditioned on a rendered glyph map, while GlyphDraw (Ma et al., 2023) combines glyph information with spatial control for Chinese and English text. TextDiffuser (Chen et al., 2023) instead predicts character-level layout masks before image synthesis, and TextDiffuser-2 (Chen et al., 2024) uses a language model to plan keywords and bounding boxes from open-ended prompts. These approaches tie a string more firmly to a place in the image, but they do not assume the same input: glyph-conditioned systems receive an explicit visual control, whereas layout-first systems must predict a layout whose errors can propagate to image generation.

Table 1: Text-rendering benchmarks: inputs, target references, and evaluation. The table summarizes the main evaluation components; details appear in Section 2.2 . Notes: EN: English; ZH: Chinese. Extra input denotes benchmark-supplied controls beyond the prompt, not internally predicted layouts. OCR: optical character recognition; VQA: visual question answering. PNED matches words by edit distance and penalizes unmatched items. <sup>†</sup>No image is supplied for T2I; source images are used for editing or image-to-image tasks. UltraText pairs per-region text and visual attributes with image-level VLM ratings.
<table><tr><td>Benchmark</td><td>Language</td><td>Extra input</td><td>Target reference</td><td>Evaluation</td></tr><tr><td>MARIO-Eval</td><td>EN</td><td>None</td><td>Target strings</td><td>OCR + image metrics</td></tr><tr><td>AnyText-Bench</td><td>EN/ZH</td><td>Glyph/position</td><td>Text + positions</td><td>OCR on text crops</td></tr><tr><td>LeX-Bench</td><td>EN</td><td>None</td><td>Text + style/position</td><td>PNED + attribute VQA</td></tr><tr><td>CVTG-2K</td><td>EN</td><td>None</td><td>Text instances</td><td>OCR + matching</td></tr><tr><td>LongText-Bench</td><td>EN/ZH</td><td>None</td><td>Target strings</td><td>VLM text accuracy</td></tr><tr><td>TextInVision</td><td>EN</td><td>None</td><td>Target strings</td><td>OCR edit distance</td></tr><tr><td>TextAtlasEval</td><td>EN/ZH</td><td>None</td><td>Scene + text</td><td>OCR + CLIP similarity</td></tr><tr><td>OCRGenBench</td><td>EN/ZH</td><td>None / image†</td><td>Task-specific targets</td><td>OCRGenScore</td></tr><tr><td>InfoTextBench</td><td>EN/ZH</td><td>None / image†</td><td>Target text list</td><td>OCR + VLM quality</td></tr><tr><td>UltraText Bench (ours) EN/ZH</td><td></td><td>None</td><td>Region text + layout/carrier</td><td>Image-level VLM</td></tr></table>

Character-aware and multilingual modeling. AnyText (Tuo et al., 2024) combines glyph and position conditions with OCR-aware representations in one multilingual generation and editing framework. Glyph-ByT5 (Liu et al., 2024) introduces a byte-level text encoder to preserve character information lost to subword tokenization. JoyType (Li et al., 2024) targets multilingual typography, and ViType (Gao et al., 2026) jointly models semantic and glyph features in a multimodal diffusion model. Complementary work studies scene-text inpainting (Zhang et al., 2024), position-controlled synthesis (Zhao & Lian, 2024), and data synthesis and filtering (Zhao et al., 2025).

Dense and multi-region generation. Recent methods target longer strings and multiple text regions. TextCrafter (Du et al., 2025) steers attention toward the quoted target strings and adds a reinforcement signal read from OCR, improving how multiple strings are rendered in complex scenes. GlyphDraw2 (Ma et al., 2025) combines language-model planning with glyph-conditioned generation for multi-element posters, while PosterMaker (Gao et al., 2025) focuses on text-rich product posters. BizGen (Peng et al., 2025) addresses article-level infographic generation by binding text spans to planned regions, and TextGuider (Baek et al., 2025) applies training-free attention guidance to reduce omissions in longer text. GlyphAnchor (Xiang et al., 2026) anchors glyph priors to predicted positions, and TextGround4M (Mao et al., 2026) complements these methods with prompt-aligned text spans and layout annotations for training layout-aware models. Their inputs range from prompts alone to explicit glyph and layout controls, with evaluations tailored to each application.

Few-step image generation. TwinFlow (Cheng et al., 2026) and APEX, which uses condition shifting (Liu et al., 2026), study self-adversarial learning for one-step generation. Duality Models (Sun et al., 2026b) couples velocity and flow-map learning, while Three-Body Scattering (Sun et al., 2026a) derives one-step supervision from a distributional energy. These advances motivate evaluating densetext fidelity and clarity alongside overall image quality.

## 2.2 TEXT RENDERING BENCHMARKS AND DATASETS

Text-rendering resources differ in what they ask the model to produce, in what they tell it beforehand, and in how they check the result. Scores reported on short-string, poster, document, and long-text test sets therefore cannot be placed on one scale. Table 1 summarizes the closest comparisons.

Short-string fidelity and attributes. MARIO-Eval (Chen et al., 2023) established a large shortkeyword test set. AnyText-Bench (Tuo et al., 2024) added English and Chinese generation tests whose OCR scores are computed on text lines cropped at the specified positions; the AnyText model also edits text, but the benchmark’s quantitative protocol is generation-oriented. TypeScore (Sampaio et al., 2024) moved past a single OCR accuracy value: it extracts the rendered text and combines several string-comparison measures into one fidelity score. That score tells spelling errors, missing text, and extra text apart, and it is validated against human judgments. LeX-Bench (Zhao et al.,

![](images/06ba25865150a90d317077eae114d6c3f9872e912227946834bd60f7d5ae8b04.jpg)  
Figure 3: Task scenes and assessment in three benchmarks. Left: CVTG-2K (Du et al., 2025) matches OCR-recognized words to targets using Word Accuracy and normalized edit distance, and separately reports CLIPScore. Center: LongText-Bench (Geng et al., 2025) uses a VLM to extract text for comparison with target strings. Right: UltraText Bench uses a structured reference (Ref.) to rate the whole image: text accuracy (TA), completeness (TC), readability (TR), position correctness (PC), layout quality (LQ), and scene integration (SI). The scenes, abbreviated strings, and line marks are illustrations, not benchmark outputs or measured results; they do not exhaust each benchmark’s scene types.

2025) adds prompts that also constrain color, font, and position, and scores them with PNED, which matches requested against recognized words and penalizes the ones left over. Because PNED treats words as an unordered set, it says nothing about reading order or about which surface a string ended up on. These resources primarily measure short-string accuracy and text attributes.

Dense and long-text generation. CVTG-2K and the associated TextCrafter multi-text setting (Du et al., 2025) evaluate more complex scenes and match recognized text instances to requested strings. Their prompt descriptions also encode positions, attributes, and the correspondence between text and its carrier. LongText-Bench, introduced with X-Omni, covers longer English and Chinese text across eight scenarios (Geng et al., 2025), and document-level rendering of a single long sequence has been examined separately (Zhang et al., 2025b). TextInVision (Fallah et al., 2025) crosses prompt complexity with text properties and also examines visual-encoding failures. The closest neighbor is InfoTextBench, released with GlyphAnchor (Xiang et al., 2026), which pairs English and Chinese prompts over text-rich reference images, gives each one an explicit list of target strings, and scores word-level precision, recall, and phrase hits. Long targets, images holding many separate pieces of text, and bilingual coverage have therefore all been studied before. These resources differ in how target strings, attributes, positions, and scene context are represented and evaluated.

Text rendering in video. VTR-Bench (Huang et al., 2026) evaluates visual text rendering in video generation. It assesses text fidelity through carrier-specific transcription and scene and motion requirements through prompt-specific queries. This temporal setting complements our evaluation of dense text and region placement within individual images.

Broader visual-text tasks and editing. TextAtlas5M (Wang et al., 2025a) contributes large-scale training data for long and structured text, together with TextAtlasEval, an evaluation set spanning four design domains in which one subset varies text length under otherwise fixed conditions. OCR-GenBench (Zhang et al., 2025a) broadens evaluation to generation, editing, and OCR-oriented transformation tasks in English and Chinese, and explicitly includes high text density, page-level generation, and dense-document editing. VTPBench (Shu et al., 2025) unifies six visual text processing tasks under task-specific metrics and VTPScore, and OneIG-Bench (Chang et al., 2026) contributes a bilingual text-rendering split inside a broader image-generation suite. Task-family aggregates summarize broad performance, while trends with text load and region count require separate breakdowns. A separate line of benchmarks targets text-centric editing, which requires modifying the target text while preserving other text and the background. TextEditBench (Gui et al.,

2025) covers everyday document and signage scenes and adds a dimension for edits that depend on reasoning. WeEdit (Zhang et al., 2026a) extends editing to many languages and scores instruction adherence, text clarity, and background preservation. TextSculpt-Bench (Lin et al., 2026) compares the text that should appear in the whole image against the text that actually does, and checks the background outside the edited area. TextWand-Bench (Wang et al., 2026a) evaluates removal, genera tion, and replacement under explicit layout and style control. None of these is directly comparable to prompt-only generation, because the model is handed a source image and part of the reference is the input itself. They are still instructive here: TextSculpt-Bench in particular shows that checking all the text has to leave behind evidence that can be inspected, since handing an evaluator a complete reference does not prove that it looked at every region.

General T2I evaluation and our scope. General text-to-image benchmarks such as TIFA (Hu et al., 2023), T2I-CompBench (Huang et al., 2023), and HRS-Bench (Bakr et al., 2023) measure semantic faithfulness and composition over broader prompt families. BizGenEval (Li et al., 2026b) comes closer to our scene types, scoring slides, charts, webpages, posters, and scientific figures against human-verified checklist questions. Their checklists and alignment metrics address broader semantic and compositional constraints, complementing the explicit per-region text references used here. UltraText Bench combines prompt-only generation, a uniform per-region reference, and coverage of 24 text-bearing scene categories. The generator receives all target strings in natural language; the six-field annotation is reserved for evaluation and stays consistent across scenes, languages, and levels. This design tests whether text rendering transfers across carriers and layouts under increasing text load. The reference specifies coarse spatial metadata, and the protocol returns image-level scores; region-level attribution remains a future direction.

## 2.3 AUTOMATIC EVALUATION AND JUDGE RELIABILITY

Global and OCR-based metrics. Automatic generation evaluation spans global alignment and preference metrics, fidelity measures from OCR output, and VLM judges that follow a written rubric. CLIPScore (Hessel et al., 2021) measures global image-text similarity, while ImageReward (Xu et al., 2023), HPSv2 (Wu et al., 2023), and PickScore (Kirstain et al., 2023) learn scalar preferences from human comparisons. These scores are useful for semantic alignment or overall preference, but their training objectives do not require transcribing every requested character and can favor an attractive image whose text is incorrect. Text-specific metrics provide a closer signal. TypeScore (Sampaio et al., 2024), described above, also reports a failure mode on the evaluator’s side: OCR can introduce recognition errors, while a VLM reader can silently correct characters that the image actually renders wrongly. OCRGenScore (Zhang et al., 2025a) combines metrics across OCR-generative tasks, and VTPScore (Shu et al., 2025) uses a multimodal model to rate visual quality and text readability across visual text processing tasks. OCR-based scores depend on the detector, recognizer, language settings, font, scale, and image domain (Baek et al., 2019; Cui et al., 2025).

Text quality and spatial fidelity. A complementary line evaluates the visual quality of rendered text separately from its correctness. TIQA (Koltsov et al., 2026) collects human quality ratings for rendered text, on crops and on text-heavy images, and trains a model to predict them; this is what our clarity dimension tries to capture. Because it rates regions that were detected, it cannot by itself report a region that was requested and never rendered. Recognized strings alone also do not establish whether text occupies the requested region or integrates naturally with its carrier.

Rubric-based VLM judges. Prompted VLM evaluators can apply task-specific rubrics without training a separate metric for every dimension. VIEScore (Ku et al., 2024) scores semantic consistency and perceptual quality with explanations, and UniGenBench++ (Wang et al., 2025b) uses structured semantic units for multidimensional T2I evaluation. Qwen-Image-Bench (QIB) (Li et al., 2026a) develops fine-grained creation rubrics and a dedicated Q-Judger using professional human ratings. RichHF (Liang et al., 2024) likewise demonstrates the value of fine-grained human feedback tied to words and image regions. T2LSC-Bench (Wang et al., 2026b) pairs OCR verification with a structured VLM judgement, separating whether the text landed on its intended anchor from whether its meaning leaked into the surrounding subject and scene, and it checks the automatic labels against a human-annotated subset. Evaluator version, prompting, text-reading ability, and visual-quality bias can affect VLM scores. Prior studies examine agreement with independent human ratings (Zheng et al., 2023), controlled annotation protocols for T2I evaluation (Otani et al., 2023), and variation in text-reading ability across multimodal models (Liu et al., 2023; Fu et al., 2026). These limitations motivate joint assessment of text, placement, and scene integration, while keeping response-format validity distinct from the reliability of a judge’s assessment.

![](images/c564ae33628ce8f6f05190b2c0dd62e660eba54ddfc347ffce66f042b5c493e4.jpg)  
Figure 4: Task definition and evaluation protocol. The generator receives only prompt p. Q-Judger, from Qwen-Image-Bench (Li et al., 2026a), applies the UltraText rubric to the image and complete reference R, returning six image-level scores that map to four reporting dimensions and a composite. Reference fields and scores are not paired one-to-one. Grid cells give coarse locations; regions may share a cell. Invalid responses remain unscored and reduce coverage. The bakery schematic simplifies the four-region record A1\_L1\_EN\_001, reproduced in Appendix B.1

## 3 ULTRATEXT BENCH: BENCHMARK DESIGN

UltraText Bench examines how text fidelity, clarity, spatial quality, and scene quality vary across dense text-bearing scenes, workloads, and languages.

## 3.1 TASK FORMULATION

Inputs and generation. A benchmark instance pairs a natural-language prompt p with a structured reference $R = \{ r _ { 1 } , \ldots , r _ { K } \}$ for K text regions. Each region is an annotated target-text component that can contain multiple lines or items; for example, the entire menu in the record below forms one region. The prompt contains all target strings and is the only input to the text-to-image model M, which generates an image $I = M ( \bar { p } )$ . Each $r _ { i }$ records a target string $t _ { i } ,$ grid position, relative text size, type, carrier, and importance ( Section 3.6 ); this reference is used only for evaluation.

Illustrative record. The released L1 signage record A1\_L1\_EN\_001 describes a bakery storefront with K = 4 regions and 423 GT characters in total. Two regions occupy top-center: a large shop name on a wooden sign and a medium tagline on an oak plank below it. The remaining regions are a medium chalkboard menu at middle-left and small opening hours on a brass plate at bottom-right; their importance levels range from high to low. Figure 4 uses a simplified schematic inspired by this layout, while Section 3.6 reproduces the first region verbatim.

Evaluation scope. The complete reference specifies which strings should appear, where they belong, and how they relate to the scene. Q-Judger assesses these requirements through six image-level scores. Appendix E explains how the reference supports each assessment; the output contains no region-level scores or transcriptions.

## 3.2 CONSTRUCTION RATIONALE

The 24 categories cover different carriers and layouts, from signs to invoices and code. Three difficulty levels increase text load and region count, while the English and Chinese splits cover different glyph inventories and writing conventions (Ma et al., 2023; Tuo et al., 2024). These choices support comparisons across scene, workload, and language. Because text load and region count change together, the levels do not isolate their individual effects.

Each category×level×language cell contains three prompts, yielding $2 4 \times 3 \times 2 \times 3 = 4 3 2$ prompts. Three prompts per cell preserve taxonomy coverage within the generation and inspection budget. Four samples per prompt give 12 images per cell and 288 per level and language in a complete run. This supports aggregate analysis while retaining category coverage; individual cells have limited statistical precision. The suite is a stress test rather than a frequency-weighted sample of web images. Its layouts place different demands on spatial organization: receipts align items with prices, newspapers separate columns and headlines, and signs follow the perspective of their carriers.

Table 2: UltraText Bench scene taxonomy. Six domains organize 24 categories by rendering challenge and typical layout. Notes: Density describes qualitative text packing, not the GT-character load in Table 3 . Layouts are typical examples, not mandatory templates.
<table><tr><td>ID Category</td><td>Rendering challenge</td><td>Density</td><td>Typical layout</td></tr><tr><td colspan="4">A. Signage &amp; Labels</td></tr><tr><td>A1 Sign</td><td>Perspective and weathered surfaces</td><td></td><td>Low-Med Scattered blocks</td></tr><tr><td>A2 Label</td><td>Small text on curved surfaces</td><td>Med</td><td>Compact blocks</td></tr><tr><td>A3 Poster</td><td>Font hierarchy and composition</td><td></td><td>Med–High Stacked blocks</td></tr><tr><td>A4 Billboard</td><td>Scale and viewing angles</td><td></td><td>Low-Med Large text blocks</td></tr><tr><td colspan="4">B. Documents &amp; Print</td></tr><tr><td>B1 Article</td><td>Paragraph flow and alignment</td><td>High</td><td>Single-column</td></tr><tr><td>B2 Newspaper</td><td>Columns and headline hierarchy</td><td></td><td>Very High Multi-column</td></tr><tr><td>B3 Letter</td><td>Salutation, body, and closing</td><td>High</td><td>Single-column</td></tr><tr><td>B4 Resume</td><td>Sections and bullet alignment</td><td>High</td><td>Two-column</td></tr><tr><td colspan="4">C. Commercial</td></tr><tr><td>C1 Menu</td><td>Item-price alignment</td><td>Med-High Tabular</td><td></td></tr><tr><td>C2 Receipt</td><td>Narrow lines and exact totals</td><td>Med</td><td>Single-column</td></tr><tr><td>C3 Invoice</td><td>Numeric fields and table alignment</td><td>Med-High Tabular</td><td></td></tr><tr><td>C4 Product packaging 1</td><td>Perspective and ingredient lists</td><td>Med</td><td>Multi-panel</td></tr><tr><td colspan="4">D. Digital Interfaces</td></tr><tr><td>D1 Webpage</td><td>Navigation, content, and sidebar</td><td>High</td><td>Multi-panel</td></tr><tr><td>D2 Slide</td><td>Title and bullet hierarchy</td><td>Med</td><td>Stacked blocks</td></tr><tr><td>D3 Social media</td><td>Interface text and hashtags</td><td>Med</td><td>Card/feed</td></tr><tr><td>D4 Dashboard</td><td>KPI cards and chart labels</td><td>High</td><td>Grid</td></tr><tr><td colspan="4">E. Structured Data</td></tr><tr><td>E1 Schedule</td><td>Time-column alignment</td><td>Med-High Tabular</td><td></td></tr><tr><td>E2 Form</td><td>Label-field alignment</td><td>Med</td><td>Field grid</td></tr><tr><td>E3 Certificate</td><td>Formal typography and alignment</td><td></td><td>Low-Med Centered blocks</td></tr><tr><td>E4 Code</td><td>Monospace text and indentation</td><td>High</td><td>Indented lines</td></tr><tr><td colspan="4">F. Creative &amp; Special</td></tr><tr><td>F1 Caption</td><td>Legibility over backgrounds</td><td>Low-Med Overlay</td><td></td></tr><tr><td>F2 Dialogue</td><td>Speaker labels and text bubbles</td><td>Med</td><td>Alternating blocks</td></tr><tr><td>F3 Comic panel</td><td>Speech bubbles and sound effects</td><td>Med</td><td>Scattered bubbles</td></tr><tr><td>F4 Infographic</td><td>Text-label placement</td><td>Med-High Freeform</td><td></td></tr></table>

## 3.3 SCENE TAXONOMY: 24 CATEGORIES IN 6 DOMAINS

The six domains cover photographs, documents, and digital interfaces. Table 2 groups the 24   
categories by carrier, density, and layout, with definitions in Appendix A.2

## 3.4 DIFFICULTY STRATIFICATION

UltraText Bench uses three language-specific difficulty levels: L1 (Hard), L2 (Very Hard), and L3 (Extreme). Mean GT-character load and region count increase across these levels in both languages. A prompt’s GT-character load is $\textstyle \sum _ { i } | t _ { i } |$ , the sum of Unicode characters in all region strings, including spaces and punctuation. Figure 5 compares the design bands with the realized workloads. All bounds are inclusive: EN bands share endpoints, ZH bands overlap, and L3 is open ended in both languages. Levels describe assigned workloads rather than disjoint character-count bins, with region count providing an additional difficulty axis.

Workload rises with level in both languages  
![](images/b9f1b7fc3cd52f59a873ef10c3d39eb3f58d6e4b9c626cbe70abb6f0d69dd67f.jpg)  
realized min–max mean design band

![](images/b6541f928584faab103f63db19d9de092cbaf686f81a9655c13f23240a33a9b0.jpg)  
Figure 5: Realized difficulty statistics for the 432 prompts. Each row contains 72 prompts. Bars show minimum–maximum ranges and dots show means from Table 3 . (a) Target-string character counts, including spaces and punctuation, on a log axis. Grey brackets show the language-specific design bands; bounds are inclusive, ZH bands overlap, and L3 bands have no upper bound (shown by open ends). Levels are assigned workloads, not disjoint character bins; all records satisfy their own band. (b) Region counts; no per-level region-count bands are specified. Both panels describe benchmark inputs, not model performance.

Three levels provide an intermediate workload while retaining 72 prompts per level and language. EN design bands are 250–600, 600–1200, and 1200+ GT characters; ZH bands are 350–600, 550–900, and 600+, respectively. The realized distributions also reveal how much the workload differs between languages at a given level.

## 3.5 BILINGUAL STRATEGY

English and Chinese prompts were written independently and adapted to their cultural settings, including venue names, prices, vocabulary, and ordering conventions. The splits balance category and level counts but use different text-load bands; their scores therefore compare the two prompt sets without isolating language alone. Contact details and other identifying strings use fixed placeholders. Appendix A provides their vocabulary and bilingual examples.

## 3.6 PROMPT CONSTRUCTION AND GROUND TRUTH

Each record pairs a scene description containing all GT strings with a structured GT annotation.

Target strings in the prompt. Prompts embed target strings as quoted text, fenced passages, or structured content blocks. In the bakery record A1\_L1\_EN\_001, the first region appears as: “...a hand-carved wooden shop signfeatures cream-white seriflettering that reads: “DAILY BREAD — Artisan Bakery & Fine Patisserie Since 1987””. The scene description is abridged for readability, while the target text is reproduced verbatim.

Structured GT regions. Each region records the exact target text, a 3 × 3 grid position, relative size, text type, carrier, and importance. These attributes describe the intended text and its role in the scene: a title on a wooden sign and opening hours on a small plate have different visual requirements, even when both are correctly spelled. Size is qualitative, with no pixel or font-size thresholds. Importance is reference metadata and enters no scoring formula; the rubric requires assessment of lower-importance regions as well. The attributes guide image-level assessment without defining six corresponding attribute scores. Appendix A lists the annotation vocabulary, and Appendix B provides a complete English record and a Chinese example.

## 3.7 QUALITY ASSURANCE

All 432 prompts were manually reviewed for natural scene descriptions and checked for exact targetstring inclusion, complete annotations, and compliance with the language-specific workload bands.

Table 3: Dataset statistics by language and difficulty. Notes: Means and ranges are per prompt. GT characters include spaces and punctuation and are summed over all requested regions. The 432 prompts contain 407,918 GT characters and 2,926 regions in total; all statistics are recomputed from the released region arrays.
<table><tr><td>Split</td><td>Prompts</td><td colspan="2">GT characters per prompt</td><td colspan="2">Regions per prompt</td></tr><tr><td></td><td></td><td>Mean</td><td>Range</td><td>Mean</td><td>Range</td></tr><tr><td>ENL1</td><td>72</td><td>489.90</td><td>278-592</td><td>4.67</td><td>4-7</td></tr><tr><td>EN L2</td><td>72</td><td>1,023.44</td><td>674–1,195</td><td>7.10</td><td>6-10</td></tr><tr><td>ENL3</td><td>72</td><td>2,386.68</td><td>1,369–5,230</td><td>9.38</td><td>8-12</td></tr><tr><td>ZHL1</td><td>72</td><td>394.62</td><td>350-555</td><td>4.93</td><td>4-6</td></tr><tr><td>ZH L2</td><td>72</td><td>627.25</td><td>551-892</td><td>6.17</td><td>6-8</td></tr><tr><td>ZH L3</td><td>72</td><td>743.62</td><td>601-1,079</td><td>8.40</td><td>7-12</td></tr><tr><td>All prompts</td><td>432</td><td>944.25</td><td>278–5,230</td><td>6.77</td><td>4-12</td></tr></table>

Automated checks verified region identifiers, the six attributes and their allowed values, and character statistics. Manual revision preserved the target strings while making their placement and carriers clear in context. Every released record passes these checks and falls within its assigned band. Appendix A describes the construction and review process.

## 3.8 DATASET STATISTICS

Table 3 summarizes the dataset under this region-text character definition, and Figure 5 plots the same per-split ranges and means against the design bands. The uniform $2 4 \times 3 \times 3$ grid yields 216 prompts per language and 432 total. At four generated samples per prompt, a complete run contains 864 images per language and 1,728 images per model configuration.

The distributions show how the assigned workloads differ between languages. EN mean character load increases from 489.90 at L1 to 2,386.68 at L3, while mean region count rises from 4.67 to 9.38. ZH mean character load increases from 394.62 to 743.62, with region count rising from 4.93 to 8.40. Both languages therefore require more regions at higher levels, but the English prompts also show a much larger increase in total text. These distributions provide the context for interpreting language-specific scores at each level.

## 4 EVALUATION FRAMEWORK

We use the Q-Judger model from Qwen-Image-Bench (Li et al., 2026a) with the UltraText scoring rubric to compare generated images against the complete structured reference ( Figure 4 ).

## 4.1 FOUR-DIMENSION SCORING

Given a generated image I and the complete structured reference $R = \{ r _ { 1 } , \ldots , r _ { K } \}$ , the judge J returns six image-level raw scores:

$$
\begin{array} { r } { \mathbf { s } _ { \mathrm { r a w } } = J ( I , R ) \in [ 0 , 1 0 0 ] ^ { 6 } . } \end{array}\tag{1}
$$

Raw scores cover content correctness and completeness (TA, TC), visual legibility (TR), position agreement and typographic organization (PC, LQ), and carrier fit (SI). Table 4 maps them into four reporting dimensions s $\in [ 0 , \overline { { 1 0 0 } } ] ^ { 4 }$ . These are VLM ratings, not measured percentages of correct characters or recovered regions. The rubric defines the six properties but provides no intermediate score thresholds or explicit character-counting rule. The notation R omits auxiliary record metadata. The Composite is computed as:

$$
{ \mathrm { C o m p o s i t e } } = 0 . 6 0 \cdot s _ { \mathrm { f i d e l i t y } } + 0 . 3 0 \cdot s _ { \mathrm { c l a r i t y } } + 0 . 0 5 \cdot s _ { \mathrm { s p a t i a l } } + 0 . 0 5 \cdot s _ { \mathrm { s c e n e } }\tag{2}
$$

The weights emphasize preservation of the requested content and its legibility: fidelity contributes 60% and clarity 30%. Spatial and scene quality contribute the remaining 10%, so their separate scores are needed to interpret placement and integration errors.

Table 4: From six raw scores to four reporting dimensions. Scores are VLM ratings on a 0–100 scale, with higher values better. TA/TC/TR denote text accuracy/completeness/readability; PC/LQ/SI denote position correctness/layout quality/scene integration. Weights apply to reporting dimensions, abbreviated Fidelity, Clarity, Spatial, and Scene in the figures and leaderboard.
<table><tr><td>Dimension</td><td>Formula</td><td>Assessment focus</td><td>Weight</td></tr><tr><td>Text fidelity</td><td> $\mathrm { ( T A + T C ) / 2 }$ </td><td>Text correctness and completeness</td><td>60%</td></tr><tr><td>Text clarity</td><td>TR</td><td>Sharpness, contrast, and legibility</td><td>30%</td></tr><tr><td>Spatial quality</td><td> $( \mathrm { P C } + \mathrm { L Q } ) / 2$ </td><td>Position, alignment, and organization</td><td>5%</td></tr><tr><td>Scene quality</td><td>SI</td><td>Natural fit with the text carrier and scene</td><td>5%</td></tr></table>

Why four dimensions? The six raw scores describe different, potentially overlapping failure modes. For concise reporting, we pair correctness with completeness and position with layout, while retaining readability as text clarity and scene integration as scene quality. Clearly drawn but misspelled text can score differently on fidelity and clarity. The raw TA and TC scores remain available to distinguish correctness from completeness. For example, the menu in Figure A5 receives TA=10 and TC=100, yielding Fidelity=55. Reporting only this average would conceal the large difference between the two raw ratings. In this case, the high completeness rating does not imply accurate reproduction of the menu text. Grid positions describe coarse placement; layout quality rates organization without explicit GT for every within-cell relation. The full judge prompt appears in Appendix C.1 . The interpretive bands in Appendix D are a reading aid and are excluded from the judge prompt.

## 4.2 HUMAN EVALUATION

Ten participants took part in a human evaluation of the automatic scores. The four reporting dimensions distinguish required content, readability, spatial organization, and scene integration. Appendix H discusses the scope of this evaluation and the interpretation of qualitative comparisons. Quantitative inter-rater and human–judge agreement statistics are not reported here.

The qualitative examples in Appendix F make these distinctions inspectable. Each pairs the relevant target text with a whole image and a native-resolution crop. The hours plate in Figure A6 is identifiable locally but appears at bottom-left instead of the requested bottom-right. Its wording and placement therefore require different views of the same image. In Figure A8 , lettering follows the wooden sign’s plane, lighting, and texture. Carrier fit and exact transcription describe different properties of the rendered text.

The examples also retain disagreements with Q-Judger. Target-related chat text is visible in Figure A7 despite zero accuracy and completeness ratings, while Figure A9 contains degraded small characters despite maximum ratings. These selected cases describe local errors without estimating their frequency. The leaderboard retains the automatic scores; a score of 100 is the upper end of the rubric rather than a measured percentage of correct characters.

## 5 EXPERIMENTS

The experiments compare fidelity and clarity across 24 model configurations, examine differences across workloads and languages, and assess Base/Turbo variants. We also inspect maximum ratings and the effect of composite weights on the reported ranking.

## 5.1 EXPERIMENTAL SETUP

Models and sampling. We evaluated 24 model configurations from the FLUX, Stable Diffusion, Hunyuan, HiDream, Qwen-Image (Wu et al., 2025a; Zhao et al., 2026), Z-Image, Boogu-Image, WAN, Nano Banana, GPT Image, LLaDA (Chen et al., 2026), and Seedream families, listed in Table 5 . Base/Turbo pairs allow comparison of standard and accelerated variants. Each model uses its standard sampling configuration; resolution, aspect ratio, and API behavior can differ. The result compare these configurations rather than isolate a training or acceleration method.

Evaluation and reporting. Q-Judger applies the six-score rubric with the settings in Appendix C.2 . Rankings use these VLM ratings and the aggregation in Section 4.1 . For each prompt, we average its valid image scores; prompt-macro aggregation then gives each represented prompt equal weight. The bilingual result averages the EN and ZH prompt-macro means equally. Prompts with no valid score contribute only to coverage denominators. Image coverage is the fraction of planned images with valid scores; prompt coverage is the fraction of prompts with at least one valid score. Appendix C.3 describes the workflow; Appendix G identifies the available records and their limits. We report descriptive means without confidence intervals. Images from a shared prompt are related observations, so close comparisons require uncertainty estimates across prompts.

Table 5: UltraText Bench leaderboard. VLM ratings range from 0 to 100 (higher is better). Overall columns average EN/ZH prompt-macro means equally; level columns show language-specific composites. Composite weights are 60/30/5/5 for Fidelity/Clarity/Spatial/Scene ( Table 4 ). L1/L2/L3 denote Hard/Very Hard/Extreme; bracketed Low/High labels are API quality settings. Bold marks column bests within each group; shading marks the Composite leader. Execution failures affect coverage, not quality means ( Section 5.1 ).
<table><tr><td>Model</td><td colspan="5">Composite Fidelity Clarity Spatial Scene</td><td colspan="2">L1</td><td colspan="2">L2</td><td colspan="2">L3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>EN</td><td>ZH</td><td>EN</td><td>ZH</td><td>EN</td><td>ZH</td></tr><tr><td colspan="10"></td></tr><tr><td>Open-weight models Boogu-Image-0.1-Base</td><td>77.89</td><td>75.77</td><td>78.31</td><td></td><td>88.04 90.53</td><td>84.16 87.75 76.42 78.11 57.67 83.23</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen-Image-2512</td><td>67.96</td><td>59.30</td><td>79.89</td><td>79.13</td><td>88.93</td><td>86.50 89.4771.57</td><td></td><td></td><td>67.44</td><td>42.86</td><td>49.90</td></tr><tr><td>Boogu-Image-0.1-Turbo</td><td>67.79</td><td>66.27</td><td>64.86</td><td>84.37</td><td>86.90</td><td>73.1381.3062.8172.61</td><td></td><td></td><td></td><td>37.24 79.63</td><td></td></tr><tr><td>Z-Image-Base</td><td>61.92</td><td>55.51</td><td>70.60</td><td>70.88</td><td>77.72</td><td>82.9284.66</td><td></td><td>67.54</td><td>64.41 26.34 45.64</td><td></td><td></td></tr><tr><td>Z-Image-Turbo</td><td>54.38</td><td>40.75</td><td>74.41</td><td>69.41</td><td>82.57</td><td>79.72 73.66 57.42</td><td></td><td></td><td>52.3625.1437.99</td><td></td><td></td></tr><tr><td>HiDream-O1-Image</td><td>45.56</td><td>33.26</td><td>62.56</td><td>60.51</td><td>76.10</td><td>80.9555.33 51.51 27.8029.53 28.23</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaDA-Image</td><td>43.92</td><td>31.65</td><td>59.87</td><td>64.76</td><td>74.51</td><td>62.9654.72</td><td></td><td></td><td>47.4033.4431.2333.78</td><td></td><td></td></tr><tr><td>LLaDA-Image-Turbo</td><td>41.29</td><td>24.03</td><td>67.24</td><td>60.75</td><td>73.12</td><td>57.93 39.69</td><td></td><td></td><td>51.34 31.29</td><td>36.61 30.89</td><td></td></tr><tr><td>Qwen-Image</td><td>41.27</td><td>33.75</td><td>50.07</td><td>52.87</td><td>67.10</td><td>73.83 64.39</td><td></td><td></td><td>42.45 32.6617.61 16.69</td><td></td><td></td></tr><tr><td>FLUX.2 [Dev]</td><td>41.03</td><td>27.86</td><td>60.81</td><td>51.91</td><td>69.42</td><td>86.7933.29</td><td></td><td>63.61 14.97</td><td></td><td></td><td>35.6611.84</td></tr><tr><td>HiDream-O1-Image-Dev</td><td>40.44</td><td>24.58</td><td>63.06</td><td>58.69</td><td>76.70</td><td></td><td>69.7646.5145.29</td><td></td><td>26.6629.27</td><td></td><td>25.12</td></tr><tr><td>Hunyuan-Image-3.0</td><td>29.93</td><td>20.43</td><td>39.32</td><td>51.83</td><td>65.55</td><td></td><td>47.86 37.6333.74 22.8117.46</td><td></td><td></td><td></td><td>20.08</td></tr><tr><td>FLUX.2 [Klein] 9B</td><td>22.94</td><td>14.60</td><td>32.02</td><td>36.86</td><td>54.54</td><td></td><td>59.9611.64 43.33</td><td></td><td></td><td>5.0913.69</td><td>3.91</td></tr><tr><td>FLUX.2 [Klein] 4B</td><td>19.58</td><td>10.86</td><td>29.66</td><td>31.98</td><td>51.37</td><td>56.17</td><td>7.68 35.62</td><td></td><td>4.54</td><td>9.84</td><td>3.65</td></tr><tr><td>FLUX.1 [Dev]</td><td>18.17</td><td>0.93</td><td>49.19</td><td>17.52</td><td>39.68</td><td>24.88</td><td>12.1723.31</td><td></td><td>16.12</td><td>18.63</td><td>13.92</td></tr><tr><td>SD3.5 Large</td><td>7.48</td><td>0.35</td><td>18.53</td><td>9.04</td><td>25.23</td><td>18.48</td><td>1.22</td><td>13.20</td><td>1.91</td><td>9.14</td><td>0.94</td></tr><tr><td colspan="10"></td></tr><tr><td>API-only models</td><td></td><td>99.23</td><td>99.52</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT Image 2 [Low] Nano Banana 2</td><td>99.35 97.00</td><td>95.90</td><td>98.68</td><td></td><td>99.41 99.57 99.00</td><td></td><td>99.73 99.32 99.58 99.48 99.36 98.61 98.82 96.46 97.32 95.48 96.82 97.12</td><td></td><td></td><td></td><td></td></tr><tr><td>Seedream 5.0 Pro</td><td>91.45</td><td>88.92</td><td>94.79</td><td>98.16 95.73</td><td>97.33</td><td></td><td>97.72 97.60 93.63</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen-Image-2.0-Pro</td><td>90.21</td><td>89.69</td><td>89.78</td><td>94.18</td><td>95.14</td><td></td><td>95.74 94.49</td><td>85.60</td><td>92.89</td><td>93.58 75.36 90.81</td><td></td></tr><tr><td>Qwen-Image-3.0</td><td>88.33</td><td>87.81</td><td>87.63</td><td>93.15</td><td>93.97</td><td></td><td>91.94 86.33</td><td>88.74 86.91</td><td></td><td>80.59 85.05</td><td>91.99 91.03</td></tr><tr><td>WAN-2.7-Image-Pro</td><td>85.73</td><td>79.72</td><td>94.84</td><td>92.84</td><td>95.85</td><td></td><td>94.98 90.14 90.03 81.46 75.61</td><td></td><td></td><td></td><td>82.14</td></tr><tr><td>GPT Image 1.5 [High]</td><td>57.96</td><td>48.86</td><td>69.69</td><td>72.37</td><td>82.31</td><td></td><td>97.2839.18 92.32 23.54 77.21</td><td></td><td></td><td></td><td>18.22</td></tr><tr><td>GPT Image 1 [High]</td><td>37.91</td><td>22.91</td><td>60.01</td><td>52.20</td><td>70.85</td><td></td><td>75.8130.4961.3712.7937.92</td><td></td><td></td><td></td><td>9.06</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 5.2 BENCHMARK RESULTS

Table 5 reports equal-language prompt-macro means. Its first five columns aggregate prompt-level scores directly, rather than averaging rounded level cells.

Text fidelity and clarity. Clear text can still differ from the requested content. Z-Image-Turbo receives 74.41 Clarity and 40.75 Fidelity; Qwen-Image-2512 receives 79.89 and 59.30. Their scene quality scores are also higher than their fidelity scores. High clarity and scene-quality ratings therefore coexist with substantially lower ratings for the requested text content. Figure A3 illustrates this distinction with malformed text on a plausible sign. Among open-weight models, Boogu-Image-0.1- Base has higher Fidelity than Qwen-Image-2512 (75.77 versus 59.30), while Qwen-Image-2512 has slightly higher Clarity (79.89 versus 78.31). The model with the clearest rated text therefore need not be the one that best preserves the requested content.

Performance across difficulty levels. Qwen-Image-2512’s Composite falls from 86.50 at L1 to 42.86 at L3 in EN, and from 89.47 to 49.90 in ZH. Z-Image-Base likewise falls from 82.92 to 26.34 in EN. The higher-workload groups expose weaknesses less apparent at L1. The trend is not universal: Boogu-Image-0.1-Base’s ZH score rises from 78.11 at L2 to 83.23 at L3. Levels group different prompts with changing text loads and region counts; they do not isolate the effect of length or guarantee monotonically decreasing scores.

Base and Turbo models. Under reported settings, Turbo Fidelity is lower than Base by 14.76 points for Z-Image, 7.62 for LLaDA-Image, and 9.50 for Boogu-Image-0.1. Clarity rises by 3.81 and 7.37 points for Z-Image and LLaDA-Image, but falls by 13.45 for Boogu-Image-0.1. Fidelity decreases across all three pairs, while the direction of the clarity change depends on the model family. Differences in standard settings prevent attributing these changes to acceleration alone. Recent advances in one-step generation (Cheng et al., 2026; Liu et al., 2026; Sun et al., 2026a;b) make it useful to assess both dimensions when evaluating few-step models on dense text.

English and Chinese. Language differences depend on the model. At L3, Boogu-Image-0.1-Base scores 57.67 in EN and 83.23 in ZH, whereas GPT Image 1.5 [High] scores 77.21 and 18.22. These are comparisons between independently written prompt sets. EN L3 averages 2,386.68 GT characters and ZH L3 averages 743.62, so language, content, and workload all contribute to the comparison.

## 5.3 STRONG MODELS AND EVALUATION COVERAGE

GPT Image 2 [Low] leads the Composite at 99.35. Of its 1,723 valid images, 1,440 (83.6%) receive 100 on all six raw dimensions. Its EN and ZH scores remain high across all three levels, while many other configurations decline sharply at L3.

Figure 1 shows its L3 outputs. Appendix F includes local disagreements such as the maximumrated code image in Figure A9 . Appendix E.2 discusses implications for model assessment.

Image coverage is 859/864 in EN and 864/864 in ZH. Four EN safety rejections affect one prompt and the fifth affects another, yielding prompt coverage of 215/216 and 216/216. Entirely unscored prompts contribute to coverage but not quality means.

Sensitivity to the composite weights. GPT Image 2 [Low] leads all four reported means and hence any nonnegative weighted average of them. Boogu-Image-0.1-Base remains the open-weight leader under equal weights, fidelity-only weighting, and 40/30/15/15, as well as the default 60/30/5/5. Close rankings can depend on the tradeoff: Qwen-Image-2512’s default Composite is 67.96 versus Boogu-Image-0.1-Turbo’s 67.79, while the latter has higher Fidelity. Reweighting these means addresses neither prompt-sampling uncertainty nor judge reliability.

## 6 DISCUSSION

Why dense text is different from ordinary image quality. Diffusion models (Ho et al., 2020) operate in continuous visual spaces where approximate similarity is usually acceptable: small geometric or texture errors often preserve the intended object. Text is different because it is discrete and compositional; one wrong character can invalidate a word, code token, phone number, price, or form field. Self-attention in DiT (Peebles & Xie, 2023) and UNet (Rombach et al., 2022) architectures must split capacity between global composition and many local character patterns, and dense prompts spread this burden across multiple text regions. Text-annotated image pairs are rarer than ordinary image-caption pairs in web corpora (OpenAI, 2023), and incidental scene text is often noisy or weakly described. CLIP (Radford et al., 2021) and T5 (Raffel et al., 2020) use subword tokenization, while accurate text rendering requires preserving the exact character sequence. VAE latents can lose small stroke details, a limitation also emphasized by OCRGenBench (Zhang et al., 2025a). Visual-tokenizer evaluations further show that standard reconstruction metrics can conceal losses in text-bearing regions (Wu et al., 2025b). These mechanisms may contribute to differences between text fidelity, layout, and scene quality. Their causal effects remain untested here.

Training directions for stronger text rendering. Potential training directions for dense text rendering include visual representation warmup, as in ERW (Liu et al., 2025), and optimization with text-specific feedback. The structured annotations can support region-level diagnosis and feedback on character errors, omissions, placement, and scene integration. Such feedback can guide reward-based optimization, including DDPO (Black et al., 2024), ReFL (Xu et al., 2023), and AlignProp (Prabhudesai et al., 2023), or the construction of preference pairs used by methods such as Diffusion-DPO (Wallace et al., 2024) and FlowCPO (Han et al., 2026). Although importance does not enter our scoring formulas, future training objectives could use it to weight errors in a shop name more heavily than errors in a footnote. This direction is aligned with region-level feedback work such as RichHF-18K (Liang et al., 2024), but focuses specifically on visual text rendering.

Complementarity with concurrent benchmarks. UltraText Bench combines scene coverage and per-region references. STRICT (Zhang et al., 2025b) also evaluates thousands of characters, but varies one sequence on a plain page; UltraText Bench distributes text over 4–12 annotated regions with carriers and positions across 24 categories. OCRGenBench (Zhang et al., 2025a) spans several OCR-generative tasks, including dense pages, while UltraText Bench focuses on prompt-only generation. InfoTextBench (Xiang et al., 2026) is the closest comparison for dense bilingual text, but its target-text items differ from our regions, preventing direct comparison of their counts.

Limitations. Results reflect each model’s standard generation settings, which can differ in resolution, aspect ratio, and API behavior. A controlled comparison of these factors remains future work. Each category×level×language cell contains 3 prompts and 12 images in a complete run. A language– level aggregate contains 72 prompts, and a language aggregate contains 216. Repeated images share a prompt, so prompt-level variation is the relevant unit for comparing aggregate scores. We report descriptive means without confidence intervals; close ranks and fine-grained cell differences require correspondingly cautious interpretation. The protocol returns image-level scores from one VLM judge. Ten participants took part in human evaluation of the automatic scores; Appendix H discusses its scope and limitations. Quantitative inter-rater agreement and alignment with human ratings are not reported here. The per-region references specify the content and visual context for assessment, while the current output summarizes each image as a whole. Maximum scores are frequent in the GPT Image 2 records, and selected cases show both low-score disagreements and visible errors despite maximum ratings. The examples distinguish local character defects from an image’s overall rating; they do not quantify the judge’s error rate. A maximum rating is not a measured percentage of correct characters. Strong aggregate performance and occasional local disagreements therefore need to be interpreted at their respective scales. The language splits use independently written content and different character-load bands, and difficulty levels change text load and region count together. Their comparisons therefore do not isolate language or text length alone.

Future work. Future work will explore region-level evaluation and structured annotations as training signals. Extending the benchmark to video generation and editing systems such as Vidu S2 (Zhang et al., 2026b) would enable evaluation of text persistence and temporal consistency.

## 7 CONCLUSION

UltraText Bench evaluates dense bilingual text with complete region references across 24 scene categories. Comparisons of 24 model configurations distinguish fidelity from clarity and expose differ ences across workloads and languages. The suite supports assessment of sustained text reproduction, placement, and integration with the surrounding scene.

## REFERENCES

Jeonghun Baek, Geewook Kim, Junyeop Lee, Sungrae Park, Dongyoon Han, Sangdoo Yun, Seong Joon Oh, and Hwalsuk Lee. What is wrong with scene text recognition model comparisons? dataset and model analysis. In Proceedings ofthe IEEE/CVF international conference on computer vision, pp. 4715–4723, 2019.

Kanghyun Baek, Sangyub Lee, Jin Young Choi, Jaewoo Song, Daemin Park, Jooyoung Choi, Chaehun Shin, Bohyung Han, and Sungroh Yoon. Textguider: Training-free guidance for text rendering via attention alignment. arXiv preprint arXiv:2512.09350, 2025.

Eslam Mohamed Bakr, Pengzhan Sun, Xiaoqian Shen, Faizan Farooq Khan, Li Erran Li, and Mohamed Elhoseiny. Hrs-bench: Holistic, reliable and scalable benchmark for text-to-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20041–20053, 2023.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, volume 2024, pp. 4965–4987, 2024.

Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. Advances in Neural Information Processing Systems, 38, 2026.

Chuyan Chen, Haoxing Chen, Kun Chen, Zhenglin Cheng, Long Cui, Ruishan Fang, Zhangxuan Gu, Zhicheng Huang, Zhenzhong Lan, Yuanting Lei, Haoquan Li, Jianguo Li, Rongchuan Li, Sidu Li, Tao Lin, Deyuan Liu, Jiacheng Liu, Lin Liu, Yuxuan Lou, Zhisheng Lu, Yuxin Ma, Shuheng Shen, Peng Sun, Chaoyang Wang, Hongjun Wang, Xiaomei Wang, Yongxin Wang, Chengzhang Wu, Hongru Wu, and Jun Xie. Llada-image: Building strong image generators with fully open training recipes. arXiv preprint arXiv:2609.03796, 2026.

Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser: Diffusion models as text painters. Advances in Neural Information Processing Systems, 36: 9353–9387, 2023.

Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser-2: Unleashing the power of language models for text rendering. In European Conference on Computer Vision, pp. 386–402. Springer, 2024.

Zhenglin Cheng, Peng Sun, Jianguo Li, and Tao Lin. Twinflow: Realizing one-step generation on large models with self-adversarial flows. In International Conference on Learning Representations, 2026.

Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025.

Nikai Du, Zhennan Chen, Zhizhou Chen, Shan Gao, Xi Chen, Zhengkai Jiang, Jian Yang, and Ying Tai. Textcrafter: Accurately rendering multiple texts in complex visual scenes. arXiv e-prints, pp. arXiv–2503, 2025.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024.

Forouzan Fallah, Maitreya Patel, Agneet Chatterjee, Vlad Morariu, Chitta Baral, and Yezhou Yang. Textinvision: Text and prompt complexity driven visual text generation benchmark. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 525–534, 2025.

Ling Fu, Zhebin Kuang, Jiajun Song, Mingxin Huang, Biao Yang, Yuzhe Li, Linghao Zhu, Qidi Luo, Xinyu Wang, Hao Lu, et al. Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. Advances in Neural Information Processing Systems, 38, 2026.

Lishuai Gao, Jun-Yan He, Yingsen Zeng, Yujie Zhong, Xiaopeng Sun, Jie Hu, Zan Gao, and Xiaoming Wei. Vitype: High-fidelity visual text rendering via glyph-aware multimodal diffusion. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 4131–4139, 2026.

Yifan Gao, Zihang Lin, Chuanbin Liu, Min Zhou, Tiezheng Ge, Bo Zheng, and Hongtao Xie. Postermaker: Towards high-quality product poster generation with accurate text rendering. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 8083–8093, 2025.

Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, et al. X-omni: Reinforcement learning makes discrete autoregressive image generative models great again. arXiv preprint arXiv:2507.22058, 2025.

Rui Gui, Yang Wan, Haochen Han, Dongxing Mao, Fangming Liu, Min Li, and Alex Jinpeng Wang. Texteditbench: Evaluating reasoning-aware text editing beyond rendering. arXiv preprint arXiv:2512.16270, 2025.

Yansen Han, Shengyi Liao, Peng Sun, Deyuan Liu, Yuanxing Zhang, Pengfei Wan, and Tao Lin. FlowCPO: A unified divergence view of preference alignment for flow models. arXiv preprint arXiv:2609.09905, 2026.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A referencefree evaluation metric for image captioning. In Proceedings ofthe 2021 conference on empirical methods in natural language processing, pp. 7514–7528, 2021.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20406–20417, 2023.

Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:78723–78747, 2023.

Yu Huang, Jungang Li, Zhiyuan Wang, Yonghua Hei, Song Dai, Jiayu Yang, Deyuan Liu, Xiang Zheng, Xiaoshuang Shi, Hao Cheng, et al. VTR-Bench: A systematic benchmark for evaluating visual text rendering in video generation. arXiv preprint arXiv:2610.01499, 2026.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

Kirill Koltsov, Aleksandr Gushchin, Anastasia Antsiferova, and Dmitriy Vatolin. Tiqa: Humanaligned perceptual text quality assessment in generated images. arXiv preprint arXiv:2603.07119, 2026.

Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. Viescore: Towards explainable metrics for conditional image synthesis evaluation. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12268–12290, 2024.

Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

Chao Li, Chen Jiang, Xiaolong Liu, Jun Zhao, and Guoxin Wang. Joytype: A robust design for multilingual visual text creation. arXiv preprint arXiv:2409.17524, 2024.

Niantong Li, Guangzheng Hu, Weixu Qiao, Ying Ba, Qichen Hong, Shijun Shen, Jinlin Wang, Fan Zhou, Jianye Kang, Xin Shang, et al. Qwen-image-bench: From generation to creation in text-to-image evaluation. arXiv preprint arXiv:2605.28091, 2026a.

Yan Li, Zezi Zeng, Ziwei Zhou, Xin Gao, Muzhao Tian, Yifan Yang, Mingxi Cheng, Qi Dai, Yuqing Yang, Lili Qiu, et al. Bizgeneval: A systematic benchmark for commercial visual content generation. arXiv preprint arXiv:2603.25732, 2026b.

Youwei Liang, Junfeng He, Gang Li, Peizhao Li, Arseniy Klimovskiy, Nicholas Carolan, Jiao Sun, Jordi Pont-Tuset, Sarah Young, Feng Yang, et al. Rich human feedback for text-to-image generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19401–19411, 2024.

Yiheng Lin, Siyu Jiao, Xiaohan Lan, Wei Zhou, Qi She, Fei Yu, Heyun Chen, Zhengwei Wang, Jinghuan Chen, Moran Li, et al. Textsculptor: Training and benchmarking scene text editing. arXiv preprint arXiv:2605.21090, 2026.

Deyuan Liu, Peng Sun, Xufeng Li, and Tao Lin. Efficient generative model training via embedded representation warmup. arXiv preprint arXiv:2504.10188, 2025.

Deyuan Liu, Peng Sun, Yansen Han, Zhenglin Cheng, Chuyan Chen, and Tao Lin. Self-adversarial one step generation via condition shifting. arXiv preprint arXiv:2604.12322, 2026.

Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2(5):6, 2023.

Zeyu Liu, Weicong Liang, Zhanhao Liang, Chong Luo, Ji Li, Gao Huang, and Yuhui Yuan. Glyphbyt5: A customized text encoder for accurate visual text rendering. In European Conference on Computer Vision, pp. 361–377. Springer, 2024.

Jian Ma, Mingjun Zhao, Chen Chen, Ruichen Wang, Di Niu, Haonan Lu, and Xiaodong Lin. Glyphdraw: Seamlessly rendering text with intricate spatial structures in text-to-image generation. arXiv preprint arXiv:2303.17870, 2023.

Jian Ma, Yonglin Deng, Chen Chen, Nanyang Du, Haonan Lu, and Zhenyu Yang. Glyphdraw2: Automatic generation of complex glyph posters with diffusion models and large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 5955–5963, 2025.

Dongxing Mao, Yilin Wang, Linjie Li, Zhengyuan Yang, and Alex Jinpeng Wang. Textground4m: A prompt-aligned dataset for layout-aware text rendering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 7918–7926, 2026.

OpenAI. Dalle-3, 2023. URL https://openai.com/dall-e-3.

OpenAI. Gpt-image-1, 2025. URL https://openai.com/index/ introducing-4o-image-generation/.

Mayu Otani, Riku Togashi, Yu Sawai, Ryosuke Ishigami, Yuta Nakashima, Esa Rahtu, Janne Heikkilä, and Shin’ichi Satoh. Toward verifiable and reproducible human evaluation for text-toimage generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14277–14286, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Yuyang Peng, Shishi Xiao, Keming Wu, Qisheng Liao, Bohan Chen, Kevin Lin, Danqing Huang, Ji Li, and Yuhui Yuan. Bizgen: Advancing article-level visual text rendering for infographics generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 23615–23624, 2025.

Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67, 2020.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In IEEE Conference on Computer Vision and Pattern Recognition, 2022.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic textto-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, 2022.

Georgia Gabriela Sampaio, Ruixiang Zhang, Shuangfei Zhai, Jiatao Gu, Josh Susskind, Navdeep Jaitly, and Yizhe Zhang. Typescore: A text fidelity metric for text-to-image generative models. arXiv preprint arXiv:2411.02437, 2024.

Yan Shu, Weichao Zeng, Fangmin Zhao, Zeyu Chen, Zhenhang Li, Xiaomeng Yang, Yu Zhou, Paolo Rota, Xiang Bai, Lianwen Jin, et al. Visual text processing: A comprehensive review and unified evaluation. arXiv preprint arXiv:2504.21682, 2025.

Peng Sun, Zhenglin Cheng, Deyuan Liu, Jun Xie, Xinyi Shang, and Tao Lin. Three-body scattering for generative modeling. arXiv preprint arXiv:2607.18198, 2026a.

Peng Sun, Xinyi Shang, Tao Lin, and Zhiqiang Shen. Duality Models: An embarrassingly simple one-step generation paradigm. arXiv preprint arXiv:2602.17682, 2026b.

Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing. In International Conference on Learning Representations, volume 2024, pp. 56783–56799, 2024.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024.

Alex Jinpeng Wang, Dongxing Mao, Weiming Han, Jiawei Zhang, Zhuobai Dong, Linjie Li, Yiqi Lin, Zhengyuan Yang, Libo Qin, Fuwei Zhang, et al. Textatlas5m: A large-scale dataset for long and structured text image generation. arXiv preprint arXiv:2502.07870, 2025a.

Shuyu Wang, Zhile Guan, Hongxiu Chen, Yule Duan, Weiqi Li, Xin Shan, Ronggang Wang, and Jian Zhang. Textwand: A unified framework for scene text editing. arXiv preprint arXiv:2606.05730, 2026a.

Yan Wang, Xinyi Hou, Weiguo Lin, Junjun Si, and Siwei Ma. T2lsc-bench: Benchmarking localized semantic control in text-to-image generation. arXiv preprint arXiv:2609.02255, 2026b.

Yibin Wang, Zhimin Li, Yuhang Zang, Jiazi Bu, Yujie Zhou, Yi Xin, Junjun He, Chunyu Wang, Qinglin Lu, Cheng Jin, et al. Unigenbench++: A unified semantic evaluation benchmark for text-to-image generation. arXiv preprint arXiv:2510.18701, 2025b.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Junfeng Wu, Dongliang Luo, Weizhi Zhao, Zhihao Xie, Yuanhao Wang, Junyi Li, Xudong Xie, Yuliang Liu, and Xiang Bai. Tokbench: Evaluating your visual tokenizer before visual generation. arXiv preprint arXiv:2505.18142, 2025b.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

Qiang Xiang, Shuang Sun, Binglei Li, Yibo Chen, Xu Tang, Yao Hu, and Junping Zhang. Glyphanchor: Enhancing visual text rendering via position-anchored glyph priors. arXiv preprint arXiv:2609.02349, 2026.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

Yukang Yang, Dongnan Gui, Yuhui Yuan, Weicong Liang, Haisong Ding, Han Hu, and Kai Chen. Glyphcontrol: Glyph conditional control for visual text generation. Advances in Neural Information Processing Systems, 36:44050–44066, 2023.

Hui Zhang, Juntao Liu, Zongkai Liu, Liqiang Niu, Fandong Meng, Zuxuan Wu, and Yu-Gang Jiang. Weedit: A dataset, benchmark and glyph-guided framework for text-centric image editing. arXiv preprint arXiv:2603.11593, 2026a.

Jintao Zhang, Kai Jiang, Jintao Chen, Xu Wang, Deyuan Liu, Jungang Li, Dechuang Chen, Ming Lin, Jingjiang Zhou, Haopeng Jin, et al. Vidu S2: Real-time interactive, editable, and spatial video generation. arXiv preprint arXiv:2609.11638, 2026b.

Lingjun Zhang, Xinyuan Chen, Yaohui Wang, Yue Lu, and Yu Qiao. Brush your text: Synthesize any scene text on images via diffusion model. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 7215–7223, 2024.

Peirong Zhang, Haowei Xu, Jiaxin Zhang, Xuhan Zheng, Guitao Xu, Yuyi Zhang, Junle Liu, Zhenhua Yang, Wei Zhou, and Lianwen Jin. Ocrgenbench: A comprehensive benchmark for evaluating ocr generative capabilities. arXiv preprint arXiv:2507.15085, 2025a.

Tianyu Zhang, Xinyu Wang, Lu Li, Zhenghan Tai, Jijun Chi, Jingrui Tian, Hailin He, and Suyuchen Wang. Strict: Stress-test of rendering image containing text. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 21148–21161, 2025b.

Bing Zhao, Chenfei Wu, Deqing Li, Hao Meng, Jiahao Li, Jie Zhang, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kuan Cao, et al. Qwen-image-2.0 technical report. arXiv preprint arXiv:2605.10730, 2026.

Shitian Zhao, Qilong Wu, Xinyue Li, Bo Zhang, Ming Li, Qi Qin, Dongyang Liu, Kaipeng Zhang, Hongsheng Li, Yu Qiao, et al. Lex-art: Rethinking text generation via scalable high-quality data synthesis. arXiv preprint arXiv:2503.21749, 2025.

Yiming Zhao and Zhouhui Lian. Udifftext: A unified framework for high-quality text synthesis in arbitrary images via character-aware diffusion models. In European conference on computer vision, pp. 217–233. Springer, 2024.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Appendix guide. Appendix A describes how the prompts were built, what each annotation field means, and what every category asks for. Appendix B reproduces one record in full and follows a single scene family across the three levels. Appendix C gives the judge prompt verbatim, the reference payload that accompanies it, and the workflow that turns an image into a scored row. Appendix D sets out how the six raw scores become four reporting dimensions and one composite, worked through on one recorded response. Appendix E explains how region references support the existing scores and how to interpret performance as model capabilities improve. Appendix F reads seven generated images against their references and collects the per-category galleries. Appendix G records which artifacts exist for this revision and what they cannot establish. Appendix H describes the scope of human evaluation and the interpretation of qualitative examples.

## A DATASET CONSTRUCTION AND ANNOTATION

All 432 prompts were drafted with Claude Opus 4.6 and then reviewed by hand, entry by entry, over several rounds of revision. Construction proceeded in five stages.

First, we studied 160 seed prompts from X-Omni LongText-Bench to guide scene descriptions, target-string presentation, and region annotation. Three lessons carried into our own design: how a scene description should be organized, how a target string can be placed inside that description without announcing itself as a target, and how finely a scene should be divided into annotated regions. We settled on embedding each string where it naturally belongs, inside quotation marks, a code fence, or an explicit structural block as the scene requires, on describing every string together with the surface it is written on, and on varying the scene context from one prompt to the next.

Second, we wrote 9 prompts for each of the 24 categories in each language, 3 at every difficulty level, which yields the full $2 4 \times 3 \times 3 \times 2 = 4 3 2$ design. We generated one category at a time so that prompts within a category would not echo each other: no two share a business name, a location, or a content theme. The English and Chinese sets were written independently and adapted to their own cultural setting rather than translated from one another.

Third, automated scripts checked every record against the specification. Each prompt’s GT-character load was recomputed as $\begin{array} { r } { \sum _ { r } | \mathfrak { g t \_ r e g i o n s } [ r ] . \dot { \mathfrak { t } } \mathrm { e x t } | } \end{array}$ and compared against the design band for its language and level; Figure 5 plots those bands against the realized distributions. All 432 released records fall inside their own band under this definition. Table 3 reports the resulting ranges. The scripts further verified that every region carries an identifier and all six annotated fields, and that every GT string appears verbatim somewhere in its own prompt. Most strings are set off by quotation marks, but code and other structured content may instead sit in a fenced or explicitly labeled block, so the check does not insist on a single quoting convention.

Fourth, we manually revised every prompt to integrate target strings into the scene description. Repeated instructions such as “The image must contain the following text” were replaced by descriptions of the text on its intended carrier, with all target strings preserved.

Fifth, we reviewed cross-lingual difficulty against the realized region-text distributions and against what each language needs in order to describe a comparable scene. No OCR measurement acts as a release gate or as a ranking signal. Every current record satisfies the character band for its own language and level under the region-text sum above, so EN–ZH comparisons should be read from the reported distributions rather than from an OCR-derived score.

Cultural adaptation examples. The following excerpts from two independently authored L1 menu records, C1\_L1\_EN\_001 and C1\_L1\_ZH\_001, illustrate cultural adaptation; all cells are verbatim strings from the released JSONL:

<table><tr><td>Element</td><td>EN</td><td>ZH</td></tr><tr><td>Venue</td><td>BREW &amp; BLOOM COFFEE HOUSE</td><td>老街坊茶餐厅</td></tr><tr><td>Heritage line</td><td>Specialty Roasters Since 2015</td><td>始于1994年. 地道港式味</td></tr><tr><td>Price</td><td>Cappuccino $5.25</td><td>干炒牛河¥38</td></tr><tr><td>Menu item</td><td>Brown Sugar Shaken Espresso</td><td>菠萝油配丝袜奶茶</td></tr><tr><td>Ordering</td><td>Extra Shot +$0.75</td><td>外带请扫码点单·提前预约</td></tr></table>

## A.1 ANNOTATION SCHEMA

A generator receives the natural-language prompt and nothing else. The gt\_regions array travels with the record as evaluation metadata and is never handed to the model as an additional layout input. Every region carries an id together with six attributes: its exact text, a position, a size, a type, a carrier, and an importance. Three of these draw on closed vocabularies: position names one of nine grid cells, size is large, medium, small, or tiny, and importance is high, medium, or low. Type and carrier are descriptive labels for the role the text plays and the surface it is written on. Carriers include digital interface elements as well as physical surfaces. Size labels describe qualitative text scale without pixel or font-size thresholds. Importance enters no scoring formula, and the rubric prohibits ignoring regions because they have lower importance. The 2,926 released regions contain 36 distinct type labels.

The grid records a coarse location and nothing more: it carries no bounding box, no extent, and no ordering of regions within a cell. Region counts follow the supplied annotation entries, not OCR detections or line breaks: one region may contain a multi-line passage, and several regions may share a cell. Character load is the sum of the lengths of all region strings, spaces and punctuation included, which is neither a word count nor the length of the generation prompt. Appendix B shows both conventions on a complete record.

Release format and placeholders. Each JSONL record contains one prompt and its structured GT, so record-level and prompt-level counts coincide. The fixed placeholder vocabulary includes TELCODE\_SAFE, TELCODE-SAFE, SAFE-CODE, WEBLINK\_SAFE, and HANDLECODE\_SAFE, together with per-record indexed telephone forms such as TELCODE-A0001.

Illustrative prompt revision. A mechanical instruction such as “The image must contain the following text: DAILY BREAD” becomes “A walnut storefront sign bears the cream serif lettering ‘DAILY BREAD’.” The target string is untouched; what changes is that the sentence now names the surface the string sits on. We constructed this pair to show the rule, and it is not an entry recovered from the unavailable review ledger.

## A.2 CATEGORY DEFINITIONS

The six domains below expand the 24 categories of Table 2 . Each describes a family of scenes rather than a fixed template, and in all of them the text is supplied for reproduction: prices, totals, and other figures are given in the prompt and never have to be inferred or calculated.

## Domain A: Signage and Labels.

A1 Sign puts text on physical signage seen at an angle, in outdoor light and on weathered material.   
A2 Label asks for small type on compact or curved surfaces such as product bottles and price tags.   
A3 Poster needs several levels of typographic hierarchy to hold up against a busy visual background.   
A4 Billboard renders large-format text at a viewing angle and under environmental effects.

## Domain B: Documents and Print.

B1 Article runs extended prose across paragraph breaks and line wraps, justified or left-aligned.   
B2 Newspaper fits several articles onto one page in multiple columns under a headline hierarchy.   
B3 Letter follows the conventional form of a formal letter: salutation, body paragraphs, and closing.   
B4 Resume arranges sections, bullet points, and aligned dates, often across two columns.

## Domain C: Commercial.

C1 Menu aligns items with their prices across several sections, each carrying its own descriptions.   
C2 Receipt places purchase details, subtotals, and totals on narrow thermal-print lines.   
C3 Invoice arranges quantities, unit prices, and totals in a table beside issuer and recipient details.   
C4 Product Packaging wraps ingredient lists and nutrition tables around a surface seen in perspective.

## Domain D: Digital Interfaces.

D1 Webpage combines navigation, content panels, sidebars, and footers at different type sizes.

D2 Slide keeps a clean title-and-bullet structure with consistent indentation and a footer.

D3 Social Media surrounds a post or feed with interface chrome, hashtags, timestamps, and engagement counts; conversational turn-taking belongs to Dialogue instead.

D4 Dashboard packs KPI cards, chart labels, and table grids densely into a monitoring interface;   
Infographic, by contrast, organizes an explanatory narrative.

## Domain E: Structured Data.

E1 Schedule holds a time column in alignment and keeps formatting consistent from row to row.

E2 Form pairs labels with fields and adds checkboxes, section dividers, and instruction text.

E3 Certificate centers formal typography at several sizes and depends on symmetry.

E4 Code requires a consistent monospace face, exact indentation, and recognizable syntax elements.

## Domain F: Creative and Special.

F1 Caption requires readable text over a photographic or illustrated background.

F2 Dialogue alternates chat bubbles from one side to the other with speaker labels attached.

F3 Comic Panel mixes speech bubbles, sound-effect words, and narrative caption boxes.

F4 Infographic scatters data annotations, axis labels, and legends across a freeform layout.

## B INPUT EXAMPLES AND DIFFICULTY

This appendix reproduces the prompt and reference of one record, also used in Appendix D.1 . IDs encode category, level, language, and entry number; sample numbers identify repeated generations.

## B.1 COMPLETE ENGLISH SIGN EXAMPLE

A1\_L1\_EN\_001 is a Sign record at L1 whose four regions carry 423 GT characters between them. The prompt below and the reference strings that follow are quoted exactly as released, and both are also supplied in case\_records.json ( Appendix G ).

## Generator input: complete prompt

A warm, golden-hour photograph of a charming neighborhood bakery on a quiet cobblestone street in a European-style district. Late afternoon sunlight casts long amber shadows across the worn stone pavement, highlighting the weathered red brick facade partially veiled by climbing ivy whose leaves shimmer in shades of emerald and gold. The storefront features a dark walnut wood frame with large multi-paned glass windows, through which rustic wooden shelves stacked with freshly baked loaves, braided challah, and dusted boules are visible. A soft glow from vintage Edison bulbs within spills warmly onto the sidewalk, mingling with the natural light.

Above the entrance, a hand-carved wooden shop sign — approximately 120cm wide and 40cm tall — hangs from ornate wrought-iron scrollwork brackets that curl into delicate leaf motifs. The sign itself has a deep walnut stain finish with gently distressed edges suggesting decades of faithful service, and features cream-white hand-painted serif lettering in a Playfair Display style, with characters roughly 6cm tall, that reads: "DAILY BREAD — Artisan Bakery & Fine Patisserie Since 1987"

Directly beneath the main sign, a smaller rectangular oak plank with softly rounded edges and a honey-toned varnish is suspended by two short brass chains. Pressed into its surface in copper-foil italic lettering with a warm amber sheen, it displays: "Handcrafted with Love — Organic Sourdough · French Pastries · Seasonal Specials · Wood-Fired Stone Oven"

To the left of the entrance, positioned on the cobblestones beside a terracotta pot overflowing with lavender, stands a freestanding A-frame chalkboard easel with a weathered pine frame. The chalkboard surface is covered in expressive hand-drawn lettering in white and dusty rose chalk, with small decorative wheat-stalk doodles in the corners. The board announces: "Today’s Fresh Bakes: Walnut Rye Loaf \$6.50 | Butter Croissants \$3.25 | Cinnamon Cardamom Rolls \$4.00 | Olive Focaccia \$5.75 — Arrive Early, We Sell Out by Noon!"

On the right side of the doorframe, at eye level beside a polished brass door handle, a small rectangular brass plate with beveled edges catches the last rays of sunlight. Engraved in precise, dark serif lettering, it reads: "Open Tuesday–Sunday 6:30 AM – 2:00 PM | Closed Mondays | Custom Cake Orders Welcome: Call TELCODE-A0001"

## B.2 COMPLETE REGION REFERENCE

All four regions appear below with every annotated field filled in. The generator sees only the prompt above; these fields go to the judge instead.

region\_0 position: top-center; size: large; type: title.   
carrier: wooden shop sign; importance: high.   
text: DAILY BREAD — Artisan Bakery & Fine Patisserie Since 1987

region\_1 position: top-center; size: medium; type: subtitle.   
carrier: oak plank sign; importance: medium.   
text: Handcrafted with Love — Organic Sourdough · French Pastries · Seasonal Specials · Wood-Fired   
Stone Oven

text: Today’s Fresh Bakes: Walnut Rye Loaf \$6.50 | Butter Croissants \$3.25 | Cinnamon Cardamom Rolls \$4.00 | Olive Focaccia \$5.75 — Arrive Early, We Sell Out by Noon!

![](images/8d3b6d01c5d7891a7b2b4b928567a98cc17c8eaeaf7552f2cba7f20f41652fad.jpg)  
L1: 526 characters; 5 regions B2\_L1\_EN\_001

![](images/bfa6d4b6df52713c0a8782cde3e834c195ab1f44a72f536e1a8a8965288265f7.jpg)  
L2: 1,085 characters; 8 regions B2\_L2\_EN\_001

![](images/05467425e65425a23599e9651790db5eea00f0fa12b66db63d63d791f7f832e5.jpg)  
L3: 2,013 characters; 10 regions B2\_L3\_EN\_001  
Figure A1: L1/L2/L3 newspaper examples. All outputs use GPT Image 2 [Low], sample 1. Notes: The reference grows from a masthead, a headline, a lede, and a footer into several articles surrounded by smaller supporting text. Character load and region count describe the input. These thumbnails show the page layouts; checking individual characters requires enlargement.

text: Open Tuesday–Sunday 6:30 AM – 2:00 PM | Closed Mondays | Custom Cake Orders Welcome: Call TELCODE-A0001

## B.3 CHINESE ADAPTATION WITH THE SAME FIELDS

A1\_L1\_ZH\_001 is the Chinese Sign record at the same level, with 362 GT characters spread over five regions: a wooden sign, an oak plank, a chalkboard, a brass plate, and a window sticker. The fifth carrier and the RMB prices on the bakery items are what adapt the scene to Chinese usage, and neither has a counterpart in the English record.

One complete ZH region; the full record is in the accompanying JSON.

id: region\_4; position: bottom-left; size: small; type: data; carrier: window sticker;   
importance: low.

text<sub>:</sub> 本店荣获<sub>2023</sub>年度样例社区烘焙坊<sub>|</sub> 样例安全等级：<sub>A</sub>级<sub>|</sub> 备案码：<sub>SAFE-CODE-031</sub>5<sub>87</sub>

The two records share a schema and little else: they were authored separately, hold different numbers of regions, and describe different content. Placeholders such as the registration code above are literal target strings, and a model is expected to reproduce them character for character.

## B.4 THREE LEVELS WITHIN ONE SCENE FAMILY

The three English newspaper outputs below hold category, model setting, and sample index fixed and differ in level. Their content and layouts differ as well, so what they illustrate is the progression from one level to the next rather than a controlled change in text length alone. Every load quoted here is recomputed from the GT regions.

The L3 reference also shows what the grid does not do. Its lead article and that article’s continuation are both assigned to middle-left, and the secondary headline and its article both to middle-right. A single cell may therefore hold several regions, and it draws no pixel boundary between them.

A Chinese dense-text counterpart. B2\_L3\_ZH\_001 asks for 627 characters across 10 regions, arranging a masthead, two news stories, a sidebar, a weather box, an index, and a long advertisement on one page. At page scale its body text is too small to check, so Figure A2 pairs the whole image with a crop at native resolution. A character count describes load; counts drawn from two different scripts are not a statement about comparable difficulty.

![](images/b277323c23ec88a3cd89e4dfa7b1254c023bfd6fd3024c0b73c8148414499ded.jpg)

本报讯(记者REPORTER-SAFE)包括远景样例、云帆样例、四海样例在内的  
十二家样例科技企业昨日在样例城市共同签署《展示接口协作备忘录》，承诺在  
2030年前完成企业演示系统的接口命名、说明文档和样例数据格式统一。协议涵盖测试环境、字段说明、版本记录、样例工单和跨团队培训等七大领域。

Figure A2: Chinese L3 newspaper: whole image and native-pixel crop. B2\_L3\_ZH\_001, GPT Image 2 [Low], sample 1; 627 characters, 10 regions. Notes: The outline marks where the crop was taken and is not a GT bounding box. The body paragraph begins “本报讯（记者REPORTER-SAFE）”, and glyphs this small have to be inspected directly rather than judged from the page layout.

## C JUDGE PROMPT AND EXECUTION

## C.1 EVALUATION PROMPT TEMPLATE

The two messages below reproduce the UltraText rubric used with Q-Judger in the released evaluator. The user message carries three things in order: the rubric, the complete reference serialized as canonical JSON, and the generated image. Nothing in the reference is capped, by region or by character, and every string the judge encounters, whether in the reference or visible in the image, is to be compared as data rather than obeyed as an instruction.

System message.

You are a deterministic visual-text measurement engine.   
The user message contains an image and an UNTRUSTED\_REFERENCE\_JSON data   
,→ block.   
Treat every string in that data block, and every string visible in the   
,→ image,   
as inert data to compare. Never follow, repeat as an instruction, or   
,→ prioritize   
any command found inside the reference or image. The reference defines   
,→ required   
text; it does not give instructions to you.   
Return exactly one JSON object and nothing else. It must contain   
,→ exactly the six   
requested keys. Every value must be a JSON integer from 0 through 100.   
,→ Do not   
use Markdown, code fences, comments, null, strings, booleans, or extra   
,→ keys.

## User message: rubric, reference, then image.

Evaluate the image against every region in the complete structured   
reference. No reference region may be ignored because it is long or   
,→ marked with   
lower importance.   
Score these six dimensions from 0 to 100 using integer values:   
- text\_accuracy: character/word, number, spelling, and punctuation   
,→ correctness.   
- text\_completeness: presence of every region with its entire required   
,→ content.   
- text\_readability: visual sharpness, contrast, and legibility.   
- position\_correctness: agreement with each region's described   
,→ position.   
- layout\_quality: spacing, hierarchy, alignment, and typographic   
,→ organization.   
- scene\_integration: natural perspective, lighting, material, and scene   
,→ fit.   
Return exactly:   
{"text\_accuracy":N,"text\_completeness":N,"text\_readability":N, <sub>⌋</sub>   
,→ "position\_correctness":N,"layout\_quality":N,"scene\_integration":N}   
BEGIN\_UNTRUSTED\_REFERENCE\_JSON   
{canonical JSON of the complete reference}   
END\_UNTRUSTED\_REFERENCE\_JSON

The symbolic N belongs to the template itself, and a response must replace it with an integer. The reference placeholder is substituted before the request is dispatched, and the image follows as image content. The line wrapping used to display these strings on the page is a typesetting artifact and is absent from both prompts.

## C.2 REFERENCE PAYLOAD AND RESPONSE VALIDATION

The reference payload is the released record with the generation prompt field removed, so alongside gt\_regions it still carries reference\_schema, prompt\_id, category, level, language, and the cached stats block (total\_chars, total\_words, total\_regions). The stats counts tell the judge how many regions and characters the reference contains, which is something a completeness score would otherwise have to infer from the region list on its own. The level field identifies the assigned difficulty tier. The rubric targets the requested text regions and specifies no separate penalty for additional, unrequested text.

A response counts as successful only if it is a single JSON object with exactly the six keys, each holding an integer between 0 and 100 inclusive. This status establishes response validity, not image quality or agreement with human judgments. Duplicate keys, extra keys, missing keys, booleans, strings, non-finite values, and values outside the range are all failures. API errors, safety refusals, empty responses, parser failures, and reference context overflow are likewise recorded with an explicit status and failure type, and every score field is set to null. Our implementation runs at temperature 0.0 with thinking disabled and a configurable response cap of 512 tokens by default, and it retains the raw response, the attempts, and the error details for every row.

Table A1: Evaluation workflow. Notes: All planned rows stay in the coverage denominators, while only strict judge successes enter the quality means. Runtime depends on hardware, generator, and service throughput. Command examples follow below.
<table><tr><td>Stage</td><td>Action</td><td>Output / validation</td></tr><tr><td>1. Generate</td><td>Generate four samples per prompt under the model&#x27;s default sampling configuration.</td><td>One PNG per sample with its model, prompt, and sample identifiers.</td></tr><tr><td>2. Validate</td><td>Check PNG structure and the prompt, GT, image, and model bindings.</td><td>Invalid images are held back from the judge with an explicit status.</td></tr><tr><td>3. Strict Judge</td><td>Send the complete GT together with the image, and require exactly six integer score keys in return.</td><td>Overflow, refusal, API, and parse failures become judge_failed rows with null scores.</td></tr><tr><td>4. Report</td><td>Read . summary. json; report planned and successful image counts, failure causes, image and prompt coverage, and language, level, and category aggregates.</td><td>Averages run over successful images; a prompt-macro mean averages within each prompt first. No implicit zero, 50, or completion threshold is applied.</td></tr></table>

## C.3 EXECUTION WORKFLOW

Table A1 sets out the workflow, separating the stages the repository executes from the aggregation step that closes it. Run paths are relative to the released code package, and values in angle brackets are supplied by the user. A row moves from planned to success only once it has both a valid artifact and a strict six-key judge response; missing, invalid\_artifact, and judge\_failed are terminal states that are still reported. Step 3 assumes that one OpenAI-compatible vLLM server is already running per requested port with the Q-Judger checkpoint, for instance ports 50000–50007 for eight services, and that each /v1/models endpoint has been verified; the evaluation client neither starts these services nor checks their health.   
Step 1 command example: generation.

python src/generation/sample\_zimage.py \   
--model\_path <MODEL> \   
--prompt\_file data/<lang>\_prompts.jsonl \   
--output\_dir <run> --steps <N> \   
--num\_images\_per\_prompt 4

Step 3 command example: strict Judge.

```perl
python scripts/eval_vlm_judge_api.py \
--sample_dir <images> \
--prompt_file data/<lang>_prompts.jsonl \
--output_file <judge.jsonl> \
--model_name <served_model>
```

Failure handling. The evaluator never turns a failed request or a malformed response into a score. Every planned prompt and sample row is kept, carrying a status such as missing, invalid\_artifact, or judge\_failed, and a failed row has null score fields and a typed reason for the failure. The raw response and the attempt errors are preserved with it. A safety refusal is treated as a non-blocking observation about the dataset and is counted as a failure state rather than converted into a quality score.

A complete bilingual run executes Steps 1–3 once for en and once for zh. At 216 prompts and four images each, a complete language split contains 864 images and the bilingual plan contains 1,728 rows per model configuration; unavailable outputs reduce coverage and receive no imputed quality score. Other generators are driven by their own scripts in the same package, with model-specific arguments, but each must emit the same binding and status fields before its images reach the judge.

Table A2: Interpretive score bands. Notes: These qualitative descriptions are a reading aid rather than calibrated thresholds on character accuracy or region recall, and they form no part of the judge prompt.
<table><tr><td colspan="6">Text-side diagnostics</td></tr><tr><td>Dimension</td><td>Severe 0-19</td><td>Weak 20-39</td><td>Partial 40-59</td><td>Strong 60-79</td><td>Excellent 80-100</td></tr><tr><td>TA Accuracy</td><td>Little or no correct text</td><td>Isolated correct fragments</td><td>Mixed correct and wrong text</td><td>Minor character errors</td><td>Near-exact transcription</td></tr><tr><td>TC Completeness</td><td>Little required text</td><td>Many missing regions</td><td>Substantial omissions</td><td>Minor omissions Near-complete</td><td>coverage</td></tr><tr><td>TR Readability</td><td>Illegible strokes</td><td>Few readable fragments</td><td>Mixed legibility</td><td>Mostly clear</td><td>Consistently legible</td></tr><tr><td colspan="6">Spatial/scene diagnostics</td></tr><tr><td>Dimension Severe 0-19</td><td></td><td>Weak 20-39</td><td>Partial 40-59</td><td>Strong 60-79</td><td>Excellent 80-100</td></tr><tr><td>PC Position</td><td>Unrelated placement</td><td>Mostly misplaced</td><td>Mixed position match</td><td>Mostly correct cells</td><td>Strong grid</td></tr><tr><td>LQ Layout</td><td>Disordered layout</td><td>Major spacing and alignment</td><td>Inconsistent organization</td><td>Mostly coherent layout</td><td>agreement Clear hierarchy and alignment</td></tr><tr><td>SI Integration</td><td>Detached from scene</td><td>errors perspective</td><td>Major surface or Uneven scene fit Minor</td><td>integration</td><td>Natural surface integration</td></tr></table>

## D SCORING AND AGGREGATION

Every raw dimension is returned on a 0–100 scale. The judge contract names each dimension and gives its scale, but it describes no intermediate bands, so Table A2 is a reading aid we added afterwards rather than a transcription of the prompt. It exists to keep interpretation consistent and to give future annotators a common vocabulary, and it is never injected into the judge prompt.

The rubric asks TA to assess character and word correctness and TC to assess the presence of every region with its entire required content. It does not specify an edit-distance calculation, a regionmatching procedure, or how incorrect but present text contributes to TC. These remain judgments of the VLM, so a high TC score alone does not establish correct reproduction. For example, Figure A5 records TA=10 and TC=100, which become Fidelity=55. Keeping both raw scores makes that difference visible. The composite supplies no pass threshold: even a hypothetical TA=0 with all five other scores at 100 would produce a Composite of 70.

The raw scores yield text fidelity $F = ( \mathrm { T A } + \mathrm { T C } ) / 2$ , text clarity $C = \mathrm { T R }$ , spatial quality $S =$ $( \mathrm { P C } + \mathrm { L Q } ) / 2 ,$ and scene quality Q = SI, with composite

$$
\mathrm { C o m p o s i t e } = 0 . 6 0 F + 0 . 3 0 C + 0 . 0 5 S + 0 . 0 5 Q .\tag{3}
$$

Only a row with a strict successful judge response enters a quality mean. Missing images, invalid artifacts, generation failures, safety refusals, parser and API failures, and context-overflow rows stay in the coverage denominators, and none of them ever becomes an implicit zero or an implicit 50.

## D.1 WORKED EXAMPLE FROM A RECORDED RESPONSE

For sample 1 of A1\_L1\_EN\_001, the judge response reads:

{"text\_accuracy":85,"text\_completeness":95,"text\_readability":95,   
"position\_correctness":90,"layout\_quality":95,"scene\_integration":98}

This gives F = 90, C = 95, S = 92.5, and Q = 98, and therefore

$$
0 . 6 0 ( 9 0 ) + 0 . 3 0 ( 9 5 ) + 0 . 0 5 ( 9 2 . 5 ) + 0 . 0 5 ( 9 8 ) = 9 2 . 0 2 5 .
$$

Because our implementation rounds each image’s composite to one decimal before aggregating, the value stored is 92.0, which is displayed as 92.00 wherever two decimal places are used; aggregate summaries keep three decimals internally. The image itself, and the carrier detail behind its integration score, appear in Figure A8 . This example demonstrates aggregation of the recorded scores. The separate manual review and its scope are described in Section 4.2.

Failure record: illustrative schema excerpt. A parser failure produces an evaluator row with null scores rather than a valid judge response. The excerpt below is constructed for explanation, and it omits the error-detail fields that a recorded row carries.

```json
{"status":"judge_failed", <sub>⌋</sub>
,→ "failure_type":"judge_api_or_response_invalid",
"text_accuracy":null,"text_completeness":null,"text_readability":null,
"position_correctness":null,"layout_quality":null, <sub>⌋</sub>
,→ "scene_integration":null,
"text_fidelity":null,"text_clarity":null,"spatial_quality":null,
"scene_quality":null,"composite":null}
```

## D.2 QUALITY MEANS AND COVERAGE: A NUMERICAL EXAMPLE

Illustrative data. Suppose four images are planned for each of two prompts. Prompt A returns two successful rows, with composites 80 and 100, and prompt B returns one, with composite 20; the remaining five planned rows are missing or failed.

$$
\mathrm { S u c c e s s f u l - i m a g e ~ m e a n } = ( 8 0 + 1 0 0 + 2 0 ) / 3 = 6 6 . 6 7 ,
$$

$$
\mathrm { P r o m p t \mathrm { - } m a c r o ~ m e a n } = \left[ ( 8 0 + 1 0 0 ) / 2 + 2 0 \right] / 2 = 5 5 . 0 0 ,
$$

$$
{ \mathrm { I m a g e ~ c o v e r a g e } } = 3 / 8 = 3 7 . 5 0 \% ,
$$

$$
\mathrm { P r o m p t ~ c o v e r a g e ~ ( a t ~ l e a s t ~ o n e ~ s u c c e s s ) } = 2 / 2 = 1 0 0 . 0 0 \% .
$$

The two means differ because prompt-macro aggregation gives each represented prompt equal weight, whereas the image mean lets A count for twice as much as B. A prompt with no successful image contributes to the coverage denominator but has no quality mean to contribute at all. Coverage and quality therefore answer different questions and have to be read together. The bilingual leaderboard then averages the EN and ZH prompt-macro means equally.

## D.3 SCOPE OF THE REPORTED AGGREGATES

Every row of Table 5 is presented the same way: the first five score columns are an equal-language mean over that model’s EN and ZH prompt-macro aggregates, and the six level columns keep the language-specific composites. A missing cell records a measurement that is unavailable and must not be read as a score of zero. Complete-reference input and a valid six-key response do not by themselves establish that every region was assessed correctly. The current output has no region-level comparison or transcription with which to verify that coverage. PC rates coarse positions. The grid provides neither within-cell ordering nor a complete description of spatial relations between regions.

## E REGION REFERENCES AND BENCHMARK INTERPRETATION

## E.1 WHAT THE REGION REFERENCE CONTRIBUTES

The structured reference connects each requested string to its intended role in the image. A list of words can establish which content should appear, while the region description also specifies where that content belongs and what should carry it. These requirements matter when a correct title appears above an incomplete menu, when a price is detached from its item, or when readable opening hours appear on the wrong side of a storefront. The reference makes such distinctions available to the evaluator across the benchmark’s scene categories.

Text accuracy and completeness. The target string defines the content against which TA and TC are judged. A region may contain a single heading or a multi-line passage, so its presence does not establish that its entire content was reproduced. In the bakery record, the shop name, tagline, menu, and hours plate are four separate requirements. Correctly rendering the shop name leaves the other three to be checked. The packaging example in Figure A4 makes the same point from the opposite direction: detailed ingredient text does not replace the missing product title.

Position and typographic organization. The grid position specifies a coarse location for PC, while relative size and the scene description provide context for visual hierarchy and LQ. Several regions can occupy one cell. In the bakery example, the shop name and tagline both belong at top-center, although the title is larger and the tagline is below it. The grid alone does not encode that within-cell ordering. Identifying the hours plate requires reading its text; checking its placement requires the whole scene. In Figure A6 , the plate is at bottom-left rather than the specified bottom-right.

Readability and carrier fit. TR concerns whether the rendered characters are visually legible. SI concerns their relation to the carrier, including perspective, lighting, and material. A sharply drawn word can be misspelled, and correctly transcribed text can still look detached from its surface. The wooden sign in Figure A8 illustrates lettering that follows its carrier’s plane and texture. A crop exposes local character shapes, while the whole image shows their integration with the scene.

Reference attributes and reported scores. The six reference attributes describe the target scene, while the six raw scores describe the generated image. Their roles differ: importance, for example, is metadata and supplies no numerical weight. The rubric instructs the judge to assess lower-importance regions as well. Reference attributes guide the image-level judgments without defining a separate score for every attribute or region. The four reporting dimensions and Composite remain the mappings in Table 4 ; the protocol introduces no additional region-level metric.

## E.2 BENCHMARK VALUE AS MODEL CAPABILITIES IMPROVE

Improved rendering of short strings shifts attention toward whether models can preserve text across longer passages, multiple regions, and varied carriers. A receipt must retain its item-price associations, a form must preserve its fields, and a storefront must render both its prominent name and its smaller supporting text. UltraText Bench evaluates these demands with one rubric.

The comparisons in Table 5 reveal substantial variation within this setting. Qwen-Image-2512 receives an EN Composite of 86.50 at L1 and 42.86 at L3; Z-Image-Base receives 82.92 and 26.34, respectively. GPT Image 2 [Low] remains near the top of the scale in both languages at every level. These outcomes distinguish configurations that sustain high rated quality across workloads from those with large performance differences between the groups. Each group contains different prompts with varying text loads and numbers of regions.

The dimensions also distinguish configurations with similar overall scores. Qwen-Image-2512 and Boogu-Image-0.1-Turbo receive Composites of 67.96 and 67.79. Qwen-Image-2512 has higher Clar ity, 79.89 versus 64.86, while Boogu-Image-0.1-Turbo has higher Fidelity, 66.27 versus 59.30. Their close overall scores thus summarize different balances between legibility and content reproduction. Alongside the workload breakdowns, these measurements help identify the capabilities a model retains and the aspects of dense text generation that remain difficult.

![](images/805ca3debc845b3067951538964aee700d594e35bb377a28edc3e16a68075f1e.jpg)

![](images/1c1f42c600a94d5a628c475e39edcb4f6893ce19bbcda1adff009646855596c0.jpg)  
Whole-image judge scores: TA 15 TC 10 TR 85 PC 95 LQ 90 SI 95 Composite 42.40

Figure A3: Character errors on a plausible sign. A1\_L3\_EN\_002, sample 1; 1,546 GT characters, 10 regions. GT excerpt (region\_5): “GREENFIELD COMMUNITY NOTICE”. Observation: The heading of the notice is built from malformed letters, even though the poster and the street around it stay entirely recognizable. The highlighted error occurs in that heading. Dimensions: TA, with TR also relevant.  
![](images/ecb77c5ec35dd452c7672928438b5d4fae37368b28bfc009527b9e7dfc904683.jpg)

![](images/dadf01a30a9efab7649bab25692d5ec7961648c31e45fe98a89f30462f48fdb6.jpg)  
Whole-image judge scores: TA 95 TC 80 TR 98 PC 95 LQ 95 SI 98 Composite 91.60  
Figure A4: Missing title amid detailed packaging text. C4\_L1\_EN\_002, sample 2; 563 GT characters, 5 regions. GT excerpt (region\_0): “BOTANICA — Repair & Restore Shampoo”. Observation: The bottle opens straight into product claims, and the prominent product title the reference asks for is absent. Neither the surrounding text nor the detailed ingredient list can stand in for that region. Dimension: TC.

## F QUALITATIVE EXAMPLES AND DIAGNOSTIC READING

Each of the seven cases below pairs a whole image with a crop at native resolution, quotes the part of the reference that matters, and states what can be seen. All seven come from GPT Image 2 [Low], and every figure places the whole image at left and the outlined crop at right. An outline marks where a crop was taken and is an editorial annotation, not a GT box. Each score line reproduces the judge’s whole-image response in the order TA, TC, TR, PC, LQ, SI, followed by the composite; these describe the whole image rather than the crop, and none is a human rating assigned here. The cases were chosen to show how the dimensions overlap in practice, and they include a positive integration example and disagreements at low and maximum scores. These selected cases do not estimate how often each failure occurs.

![](images/083de3eaa4e604f28543c336340ecab17967b809422be4cb596d15f21e0d8faa.jpg)  
Whole-image judge scores: TA 10 TC 100 TR 85 PC 100 LQ 95 SI 95 Composite 68.10  
Figure A5: Dense Chinese glyphs require local inspection. C1\_L3\_ZH\_002, sample 4; 611 GT characters, 8 regions. GT excerpt (re<sub>g</sub>ion\_3): “鸡腿葱串两串炭火现烤¥32”. Observation: Seen at page scale the dish-and-price rows look well organized, but the glyphs are small enough that their strokes and exact wording become checkable only under enlargement. Tidy rows by themselves say nothing about text fidelity. Dimensions: TA and TR.

![](images/9fb9d6510019ff13e5f5c3e970f46fc5c5412c28ed468d33ab4fc8a34442c646.jpg)

![](images/1f0e9e359c719c0fa6ab31df7bef97e7f7d636d0b83cc5d95c90b83b0d997067.jpg)  
Whole-image judge scores: TA 10 TC 100 TR 95 PC 10 LQ 95 SI 95 Composite 68.90 Figure A6: Correct surface type at the wrong grid location. A1\_L2\_ZH\_001, sample 1; 564 GT characters, 7 regions. GT (re<sub>g</sub>ion\_4): a small door plate at bottom-ri<sub>g</sub>ht, beginning “周一至周六6:30-19:00”. Observation: The hours plate turns up in the bottom-left of the full image instead. The enlargement is what identifies the region, but its location can only be judged from the whole image. Dimension: PC.

![](images/d3c69c5fc614e34ceba1f2241cfea8e31e32028e9792cc132a8d3e374531d78a.jpg)

## 成员D

我DATE-SAFE从LOCBLOCK-SAFE出发，请了SAFE-DAYS假，计划住三晚；购物清单我来买。

Whole-image judge scores: TA 0 TC 0 TR 100 PC 0 LQ 100 SI 100 Composite 37.50  
Figure A7: Readable chat content and a discrepant judge score. F2\_L3\_ZH\_002, sample 4; 631 GT <sub>characters, 8 regions.</sub> GT excerpt (re<sub>g</sub>ion\_2): <sub>“</sub>成员<sub>D</sub>：我<sub>DATE-SAFE</sub>从<sub>LOCBLOCK-SAFE</sub>出发<sub>”.</sub> Observation: Related text is plainly visible down the central chat column, with further messages and stickers filling the flanks, and yet the judge assigns TA=0 and TC=0 while giving LQ=100. Manual inspection identifies a disagreement between the visible content and the automated fidelity scores. The original scores are retained to make this failure case inspectable; they should not be interpreted as a human assessment that all requested text is absent. Dimensions: TC and LQ.  
![](images/4d1fc391ab3450c4b63f754fcfd68830070334da38147450b13963fa13041b0a.jpg)

![](images/f06a0da52c6b14073cc7e636f67c97c54372a628476a7d9efe6d7c4b24bb0804.jpg)  
Whole-image judge scores: TA 85 TC 95 TR 95 PC 90 LQ 95 SI 98 Composite 92.00

Figure A8: Carrier integration as a separate property. A1\_L1\_EN\_001, sample 1; 423 GT characters, 4 regions. GT (region\_0): the “DAILY BREAD” title belongs on a wooden shop sign. Observation: The lettering follows the plane of the sign and picks up its lighting and surface texture. Integration is judged separately from whether the text itself is exact. Dimension: SI. The score calculation is given in Appendix D.1

Whole-image judge scores: TA 100 TC 100 TR 100 PC 100 LQ 100 SI 100 Composite 100.00 Figure A9: Maximum ratings despite degraded small text. E4\_L3\_EN\_003, sample 2; 5,230 GT Figure A9: Maximum ratings despite degraded small text. E 4\_L3\_EN\_0 03, sample 2; 5,230 GT  
![](images/3a3d2ea66e6e2c9b9bbe16d24f7d02ca50aede972e3ee1bd866b91d0cf6377fd.jpg)  
characters, 8 regions; original image 1024 × 1024 pixels. GT excerpt (region\_7): “func TestParsePage(t \*testing.T)” followed by the test cases and assertions. Observation: The bottom-right test block contains visibl deformed and difficult-to-read characters, despite maximum accuracy and readability ratings. The image hash matches its saved judge record. This selected case illustrates a local limitation of the automatic scores; it does not estimate how often maximum ratings overlook errors. Dimensions: TA and TR.

## F.1 CATEGORY GALLERIES

Figure A10 and Figure A11 give one selected output per category at L1 and L2; the L3 gallery appears in Figure 1 in the main text. The galleries show the range of scenes in the taxonomy. The enlarged cases above provide the detail needed to inspect individual strings and their carriers.

![](images/600ccf6f659de1fe0485c7db6042dcbdbea2b893b33c85c8fbada51f7bffe15e.jpg)

Sign (EN/L1)  
![](images/815c79158888fd217d435be555859be78dc84d6acda4a692e9b280af9e979ee7.jpg)

## Article (EN/L1)

![](images/048024a9974e150ef63ad2719b203e43a6e93262b052cf2b839718c0e88f5def.jpg)

## Menu (ZH/L1)

## 星固第记本 Pro 14英寸 SAMPLE-CHIP180B集成内存 51208国态硬查 星空果色型网：MODEL-SAFE-A 上M时间： DATE-SAFE

## Webpage (ZH/L1)

![](images/85ad00107d8d65ac35b1bbfdbe2c0523a900b7462f2459f7e569d91d591b66a0.jpg)

Schedule (EN/L1)  
![](images/c851520678738087dbbfa0c703d3401ee5ad14fbc67b11ac6df04e3d79bdf6cd.jpg)  
Caption (EN/L1)

![](images/0e130857e4160e3b9904482bc538e13cf9b2f7b8edea0de172b6c5fad70ecf32.jpg)

Label (ZH/L1)  
![](images/5e5969e3ac2f10de80938fdd2ea1ab699e51f9e170e151d90750f724db605224.jpg)

Newspaper (EN/L1)  
![](images/d79c7c433e0b324e01c9663035aab7bc131892cc5917894af419dda272085271.jpg)

Receipt (EN/L1)  
![](images/7b2cbf46da272c21e2424d5e2cd4a4242b3030834cd2becebeddb360b05a79e3.jpg)

Slide (EN/L1)  
![](images/a0d009479c15f14baabd1c459cf2ff2af0243462a20b5a875ed0eee4d303f989.jpg)

Form (ZH/L1)  
![](images/51465a5f2a355ec44aed5e012f8d1adc4c2192037388f87c8b7150b7721c6611.jpg)  
Dialogue (EN/L1)

![](images/fdd2f75e0f962eaa76138e7ee8efce6286f60ef9e2eb3dea68ec4d6350df4a8d.jpg)

Poster (ZH/L1)  
![](images/bd2fb2201d4b858fc1c67f2c2b1f56770d47a7765a7a3e23bf9760a197463211.jpg)

## Letter (EN/L1)

![](images/4affe6b9870a938a267de5423bad234cdd225d22bfa09ee29cc8a76c5fffa237.jpg)

## Invoice (EN/L1)

![](images/2e336c049d1327f6d7cd618baaf99316b1c5f6fe4c967ebaa24cce30b4d9b758.jpg)

Social Media (ZH/L1)  
![](images/6d5008f8bfa9d270c1d09b51801593f815d3f59050808a1b5fd15566ba25a7d9.jpg)  
Certificate (ZH/L1)

第23话 考试前夜的里梦作：水子温工作室连较于并班读书通面频道 每周五更新  
![](images/7b2dc4f7ee4a26303f20c1c3b22ac98b84909e232643c2ebc06a00d221fe65d9.jpg)  
Comic (ZH/L1)

![](images/7654b27d9e79af16f9830f77b1bfadffab8aeabf1eee6717ad4e36aa0339a1c8.jpg)

## Billboard (EN/L1)

## CANDIDATE-CV-SAFE | 计算机视觉工程师

期望城市：样侧续市D/箱利城市剧剂附间：两阳内期盟待通：SAFE-SALARY-CV

## 教育与项目经历

横潮理工学院 电子信息工程样例学士2016.09-2020.06 （郑名SAFE-RANK-CV)云牌视勿研究病 计算机模景 样斜错士20201.09-2023.06 (成绩SAFE-GRADE-CV1研究方向：提量目标线测与边律部器

## 工作经历

三年计算机税划研发经验，使用C++/OpenCV实成图像采集，标定和缺用距位，GlONNX部署验量检测模型，主导边缘线量线测项目，在DEVKCE-5AFE-CV上将争械推理征从86mrs降至31vs，测试集召四率由91.2%提升量96.6%、并密误据率从.4.6%降至1.9%，热数量化、算子兼容，性能分析与观增因日测试，求吸方向为计解构极划工程用。

## 技术简介

编面高：Ct+(Python:相量与库：QsesCy，PyTonch，ONNx Burtime:基法与技术：相检测(YOLD系列SSD]，图像必理，相机招定、缺陷检测：那著与优化：模型量化[INTI]，算子兼容性、推理加速、性提分析：工具与平台：Vake、CUDA、TerscrRT，Linux；其能：多线程国程、C.文物编写与团认协作。

Resume (ZH/L1)  
![](images/301b083f720cc88c7f0fe53a2483301a17df5213d9f5cccaf3eb30bbda61378a.jpg)

Packaging (EN/L1)  
![](images/3e2c21ad46fb8d3bc9e137666d830f598481e1f6bea470b9914c1257315f136d.jpg)

Dashboard (EN/L1)  
![](images/4dc3a3014aab14cfdf533e1941a8160e3bca4bcc1e245fb68cb8b5418791888e.jpg)

## Code (EN/L1)

![](images/f0d8ecf8121621480030f9aa7497de5b29a0cbb0350d18d3c93dd442753b6f26.jpg)  
Figure A10: Scene coverage at L1 (Hard). One selected output per category in taxonomy order, from GPT Image 2 at API quality Low. EN/ZH denote English/Chinese. These examples illustrate scene diversity.  
Infographic (ZH/L1)

![](images/d38271572ed28d27357298547df19eac8ffee3cef8aab529d6b1fe9032475fd2.jpg)

![](images/82e379d4de4a83d83481cff09c3de2cd0f19e90ff181ceebf4bcac8a732076cf.jpg)

![](images/6a376ae6520198d9d0ceaf2f7d59a03df46076ba2fee14635270d8a10e9a9fb6.jpg)

## Sign (EN/L2)

![](images/df83cc505b84a6672c05181342fe562019ef37113bcba0543cf34e21a9966f8a.jpg)

## Article (EN/L2)

![](images/fa28de0f57d920940d4e349c61f13d8cc44bfbd4bbd3213c812742c37ac9686b.jpg)

## Menu (ZH/L2)

![](images/5358b17e4da000f61e2d2d0998f0fd2b83f77a9fdef2e27fb64299cea500e2e3.jpg)

![](images/4ba1677da4aaac29d96016d35de099785e5876aec3f03c6d1bbe76b7d648e2d9.jpg)

## anne

课大前开时牌年：2034年元月16日-3005组1月10日证6：A维可表中从证证

##

##

![](images/40a580db0334fc2c1964527d0f8064baaa54c3d7b6ca62ab9687f1baed086241.jpg)

## THE CHRONICLE

## Label (ZH/L2)

## CITY UNVEILS \$800 MILLION WATERFRONT TRANSIT PLAN – Largest Public Transport Investment in Two Decades

## Newspaper (EN/L2)

##

![](images/d484c2f136d7820f5c4a632a50b5c761c5743a57180e6f0253585a8b5c2e08f6.jpg)

School board selects design option in 5-2 vote for thre new neighborhood learning centers 'A long-overdue investment.' says BOARD-CHAIR-SAPE

## Receipt (EN/L2)

## KEY FINDINGS—Q2 2026 Sample Experience Analysis— Research Tean Redou

![](images/987092b2358b5f42cc1f8395b1dd324440d1139df4696641c149560d6059e681.jpg)

## RIDGEMONT UNIVERSITY

## Poster (ZH/L2)

## Letter (EN/L2)

COMPANY LOCBLOCK-A0022, Sute 48 LOCBLOOK-SAFE | TELCODE-A0024 | EIN SAFE-CODE | WEBUNK\_SAFE

## INVOICE | Holloway Consulting LLC

![](images/97c463602c4389541b22eb5486bc115719366cb4f7a60c0c0664dd2b9cc4e752.jpg)

2 INVOICE DETAILS #54FE-2006-9312 I Iseend dan 1. 2026 1 Service Apr 1-May 31 | Dus Jul 1 Net 30 | PO SAFE-4821

##

BI M CLIENT-SAFE LIC LACCOUNTS-PAYABLE-SAFE

## SERVICES & CHARIGES

Yoar co o te meology chaperld me o repa a otie sbci say   s a   a e s  p

Strategic planning: 40 hr × \$250 Ten emaastien decks e 81 000 Two documented tvips Project management..

Aner Jal 1 late chorge is s  reontty the legal maxinur wichever is lovwer poor po   n e s Yo redr t a

## Invoice (EN/L2)

## 全部评论 12.6万条

按热度排序

CREATOR-SAFE：质谢大家的支持视频中的食材清单和步理图已程故在试细区，不含的物键接下一期看圣使三种家常清底对比，欢过言想看的主是。国复描选：白HANDLE-SAFE-A 快发!出HANDLE-SAFE-D —超试做

## Webpage (ZH/L2)

## 蛋用物品

## Slide (EN/L2)

![](images/7a93690532aa62649e5fb2fbd6ba54408a41097cdc1b03a7cfaba2838369efae.jpg)

![](images/cc90506bf186aff2643c65f4a448bf0972c634c44ea7fa38be4000cf4cfa556e.jpg)

## Billboard (EN/L2)

集出提办：STAFF-SAFE-6 集用确认：STAFF-SAFE-C ⊥自连量植：STAF-SAFE-I确以栏 集出状态：外观觉好 配终齐全 电量已检日还状态与异错记提在同一单根模页填写

## CANDIDATE-PLATFORM-SAFE 高级平台工程师/云基础设旅架构方向

## 工作经历（七年五平台与地点可兼性工程经验）

手TELCOEATINNLCDEAITLOONCESAESAESLATVATOM三CoVMEBUK\_SAPEPLATFORM

出方：SAFE-COMPANY-EQUIR  
相办人：STAF-SAFE-A 同产管理员：STAIF-SAFE-B  
税利代码： SAFE-CODE-ORG  
联系码：TB.CODE-A01370

2022.04至今 云拍基础续第有限公司 高级平合工程负责容器平为与发响和频建设，将振务平均发车时间从42分钟\* 设计跨可用区原共方源，将核心服务年度可附性提异要6:97%带锁6人小超克成650个报务还移，并通过容量分编与自动国表使置计管音源下降31%。

## 青禾中心设备借用单

## Packaging (EN/L2)

借用方：青米展示项目理  
螺制代码：GROUP-SAFE-22  
使用地点：ROOM-SAFE-204  
根用免班人：STAF-SAFEC 联系构：TELCO0E-A0137

2019.07-2022.00 江湾款更系统有限公司 平分工程师理立统一孟段告管矩范，使清优完级缺用平治恢测时间金7分9参与枪值析制和灾难结放决图，连统八个季度达别快策目标。

## SALES OPERAT1ONS — Q2 2026 OUTLOOK — Northbridge Regian - June 24 Update

## 教育背景

北海工科理箱 分布式系机 理士 2017.06-2019.06核江工程学统 网统工程 学士 2013.09-2017.06

![](images/43d3e8b3600c36f056daf780a9e028fa71e8d6a66c07bd3e2d7aa78ec9a4d3f7.jpg)

![](images/3d8da90cd2439f60b1072d6a6d0089c83822cc62db23554cde427c7303e25042.jpg)

## 技术核

## 单据编号：BORROW-SAFE-2024-12-00892填写日期：DATE-SAFE 状态：待借出复核

## MONDAY CLASS SCHEDULE Westfield Sample Learning Center Fall Semester 2026 — Period 1-7

1 SMF4 SH (t af o)   
1. 21-8 11 (108)   
1 ruc 51K 19   
4 STAIT SHTN 14N)

三

![](images/9d330d7d794dc63c8752902e3ec3be6deb3eac93aa9b72a51dc87f749f3d0c95.jpg)

Serufars

目PotgesQL

## ALETS

![](images/399f6fa3037a9f82767606f3c0670b167229ec6156d60d880d8546c70f44d367.jpg)

## Social Media (ZH/L2)

## Resume (ZH/L2)

糖长持续荧付、容量规划、可观游性

## Form (ZH/L2)

Comic (ZH/L2)

Certificate (ZH/L2)  
![](images/529658e0195293fbe1ca2dc7ee07c3db74f1c24acd18d5c2f007c667765f55eb.jpg)

![](images/85b1fabf12213e642f05ba4c8b506286e26cb2388f925f85b42c209b65fd92ac.jpg)

## Schedule (EN/L2)

## Caption (EN/L2)

![](images/5dd825878fe5ac14ea787cba4a9bf3b208c512d81ade8ad3db46da56840cde20.jpg)

## Dialogue (EN/L2)

## Dashboard (EN/L2)

##

fron dataclasses saport dataclass

## Code (EN/L2)

![](images/a76f8146ef89acfe86de965a247bfab2c5404a0b4a785e9865605a191de15e66.jpg)

## Infographic (ZH/L2)

Figure A11: Scene coverage at L2 (Very Hard). One selected output per category in taxonomy order, from GPT Image 2 at API quality Low. EN/ZH denote English/Chinese. These examples illustrate scene diversity.

## G RELEASED MATERIALS AND THEIR LIMITS

Material locations. The dataset and the evaluator are released in the benchmark code package; generated images and per-image judge records are released separately. End-to-end reproduction requires a versioned public archive linking these materials to the evaluated model configurations.

Released evidence. The complete EN and ZH prompt records, together with the released evaluator, support the dataset counts, the region inventories, the six raw metrics, strict response validation, typed failure states, and the four-dimension composite. The GPT Image 2 [Low] release adds images and per-image judge rows: 859/864 successful EN images over 215/216 prompts, and 864/864 successful ZH images over 216/216 prompts. Four missing EN images belong to F3\_L1\_EN\_003 and one to A4\_L2\_EN\_003. The recorded scores reproduce the bilingual prompt-macro Composite of 99.35. All six raw dimensions equal 100 for 732/859 valid EN images and 708/864 valid ZH images, or 1,440/1,723 (83.6%) together. These counts describe automatic ratings, without assigning correctness labels to the images. The cases shown in Appendix F are drawn from these records.

Boundaries of the evidence. The GPT Image 2 cases selected here are qualitative examples rather than a representative sample of error rates. The released judge summaries identify Qwen-Image-Bench, but their model\_revision and model\_artifact\_sha256 fields are null. The recorded identity hash does not fix the checkpoint weights. These records therefore support checking score aggregation and image bindings, with incomplete checkpoint provenance. The GPT Image 2 records do not establish the coverage or score distribution of the other model configurations in the leaderboard.

Reporting consequence. Reproduction across all configurations requires their model revisions, resolutions, sampling settings, API quality settings and evaluation dates, together with images, raw judge responses, and the status of every planned row. Image and prompt coverage should accompany each quality mean. Comparing only shared prompts can assess how differences in coverage affect a pair of models; uncertainty estimates should account for images sharing a prompt. These further comparisons are not reported here.

The released dataset statistics, including the per-prompt character and region distributions, are reported in Table 3 and are not duplicated here.

## H HUMAN ALIGNMENT

Participation and scope. Ten participants took part in human evaluation of the automatic scores.   
The four dimensions assess content, readability, spatial organization, and scene integration ( Table 4 ).   
No quantitative inter-rater or human–judge agreement is reported.

Qualitative comparisons. The selected cases in Appendix F pair whole images with nativeresolution crops and the relevant target text. These views support different checks: the crop exposes character shapes, while the whole image shows placement and carrier fit. The hours plate in Figure A6 appears at bottom-left rather than the requested bottom-right; the lettering in Figure A8 follows the wooden sign’s plane, lighting, and texture. The cases also retain disagreements with the automatic scores. Target-related chat text is visible in Figure A7 despite zero accuracy and completeness ratings, while the code image in Figure A9 contains degraded small characters despite maximum ratings. These selected observations do not estimate the frequency of scoring errors or establish the reliability of close leaderboard rankings.

Maximum ratings and their interpretation. A score of 100 is the upper end of the judge’s rubric, not a measured percentage of correct characters. Within a subset whose automatic scores are all 100, Pearson and Spearman correlations with those scores are undefined. Assessing such a subset requires examining the human ratings and the rendered text directly. A high human quality rating does not establish exact character reproduction, which requires transcription checks.