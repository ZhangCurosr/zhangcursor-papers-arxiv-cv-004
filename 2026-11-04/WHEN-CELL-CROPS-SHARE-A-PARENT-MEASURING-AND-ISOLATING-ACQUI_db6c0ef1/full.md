# WHEN CELL CROPS SHARE A PARENT: MEASURING AND ISOLATING ACQUISITION LEAKAGE IN CERVICAL CYTOLOGY BENCHMARKS

Snehasish Ghosh<sup>1,∗</sup> Soham Mitra<sup>2,∗</sup> Antareep Majumdar<sup>3</sup> KC Santosh<sup>4</sup> Nibaran Das<sup>2</sup>

<sup>1</sup>Independent Researcher, formerly afiliated with the Indian Institute of Science, Bengaluru, India

<sup>2</sup>Department of Computer Science and Engineering, Jadavpur University, Kolkata, India

<sup>3</sup>Qusbi Infotech Private Limited, Kolkata, India, <sup>4</sup>Oregon Institute of Technology, USA

<sup>∗</sup>These authors contributed equally to this work.

## ABSTRACT

Public cervical cytology benchmarks release many cell crops cut from far fewer microscope fields or slides and are often split at the crop level, so test cells share acquisition parents with training cells. In a preregistered study, crop-level splitting raised SIPaKMeD accuracy by 0.97 percentage points across twelve architectures and twenty seeds (95% CI 0.59– 1.39), and CRIC accuracy by 12.08 points (11.16–12.99). On SIPaKMeD, a factorial that fixes training crop and parent counts attributes the gain to sibling exposure (1.73 points, 0.95–2.51), not to the extra parent diversity of crop-level splits (0.00). Parent identity is decodable at about 100 times chance, yet erasing 82% of its above-floor linear recoverability left the gap unchanged, and Grad-CAM did not distinguish leaking from clean models; matching test cells’ color statistics to an unseen same-class field removed most of the advantage. Decodability and saliency cannot substitute for a parent-disjoint split.

Index Terms— Cervical cytology, data leakage, shortcut learning, benchmark validity, preregistration, evaluation protocol

## 1. INTRODUCTION

Public cervical cytology benchmarks are mostly released as cell crops cut from far fewer microscope fields or slide images [1–3]. Sibling crops share their field’s stain, illumination, focus and background, and sometimes pixels. A crop-level split, the default for files that look like independent images, puts siblings on both sides of the split (Fig. 1a).

That sibling crops inflate scores is expected: placing tiles of one subject in training and validation inflated digital pathology scores by up to 41% [4], and non-independent splits and shortcut cues mislead medical image evaluation more broadly [5–7]. Benchmark artifacts have been audited in chest radiographs [8], shortcut mechanisms dissected in segmentation [9], and image-level splitting shown to inflate CT accuracy [10]. Contamination does not imply consequence, however: in brain-tumor MRI, removing heavily contaminated test images, without retraining, left balanced accuracy essentially unchanged [11]. We ask the complementary question of when acquisition siblings do change the score, which requires retraining under each protocol rather than filtering the test set.

Three questions follow. How much does crop-level splitting change public cytology benchmarks? Is the change leakage? A grouped split removes the test parents’ siblings but, at a fixed crop budget, also leaves fewer distinct training parents (crop-level training sets hold 18% more on SIPaKMeD), so a lower score could reflect lost diversity, a confound that split comparisons leave in place. Can common diagnostics certify an evaluation? Decodable source identity is increasingly read as a sign of shortcut use [12], and saliency maps as evidence of where a model looks. A preregistered study and a registered follow-up (2,281 trained models) contribute:

• Structure: closed forms that predict a split’s exposure from its crop-to-parent table (Sec. 2).

• Consequence: one simulation-powered test per benchmark across twelve architectures (Sec. 4.1).

• Causal isolation: a SIPaKMeD factorial that varies sibling exposure at fixed crop and parent counts (Sec. 4.2).

• Diagnostic caution: interventions showing that the linearly decodable parent code and saliency do not track the leak (Sec. 4.3).

## 2. DATA AND ACQUISITION HIERARCHY

Table 1 lists four public datasets. SIPaKMeD crops come from cluster fields; filename parsing and template matching independently assign all 4,049 crops to the same field. Cx22 regions come from CCEDD source images, CRIC cells from 400 slide images with nucleus coordinates, and Herlev, with one cell per image, is a negative control. A parent is the deepest acquisition unit a release identifies; it is not a patient.

If a parent’s � crops are allocated independently to a test fraction �, a test crop has a training sibling with probability 1− $q ^ { k - 1 }$ , and at equal size a crop-level training set holds $\textstyle \sum _ { j } ( 1 -$ $q ^ { k _ { j } } ) / \{ ( 1 - q ) m \}$ times as many distinct parents as a parentlevel one over � parents (1.18 for SIPaKMeD, 1.25 for CRIC). These closed forms reproduce Table 1. The parent is also learnable: a ResNet-18 trained to identify 424 SIPaKMeD fields recognised held-out siblings at 25.7% (chance 0.24%), and at 18.7% from cell-free backgrounds. Each SIPaKMeD field holds one class; in CRIC, slide identity carries 68% of the class entropy.

![](images/b543938a55336841429ef77b0eaaca9c5f178f782a872bc1e46f66927efa43f8.jpg)  
Fig. 1. Study design. (a) A crop-level split (P1) places siblings from one acquisition parent in training and test; a parent-disjoint split (P2) does not. (b) Matched-size factorial: one test set per seed; leak and clean training sets have identical crop and field counts and difer only in the test fields’ siblings, while diversity varies separately. (c) Diagnostics asking whether decodable parent identity is what the leak uses.

Table 1. Audited datasets. Sib.: test crops with a training sibling under a random stratified 80:20 crop split; Pix.: test crops sharing source pixels with a training crop (means over five splits). Classifiers were trained only where marked; Cx22 and Herlev received the structural audit only.
<table><tr><td>Dataset</td><td>Parent</td><td></td><td>Crops Parents</td><td>Sib. %</td><td>Pix. % Models</td><td></td></tr><tr><td>SIPaKMeD [1]</td><td>field</td><td>4,049</td><td>966</td><td>92.6</td><td>52.6</td><td>√</td></tr><tr><td>Cx22 [2]</td><td>image</td><td>1,320</td><td>514</td><td>85.2</td><td>34.0</td><td></td></tr><tr><td>CRIC [3]</td><td>slide</td><td>11,534</td><td>400</td><td>99.97</td><td>71.5</td><td>√</td></tr><tr><td>Herlev [13]</td><td>cell</td><td>917</td><td>917</td><td>0.0</td><td>0.0</td><td>一</td></tr></table>

## 3. STUDY DESIGN

Everything reported comes from the preregistered Study 2 and its registered follow-up; an earlier pilot (Study 1, six architectures, five seeds) served only the power analysis. Each phase was registered before its own runs: Study 2’s splits, caches, design tables and analysis scripts were hashed and committed before its first model was trained, with smoke tests on croplevel splits only, and the follow-up after the Study 2 analysis but before any follow-up model or analysis.

Protocols and training. P1 is a stratified crop-level 80:20 split, P2 a parent-disjoint split of equal test size, and P3 additionally merges parents linked by near-duplicates. Twelve ImageNet-pretrained backbones (ResNet-18/50, DenseNet-121, MobileNet-V2, EficientNet-B0/V2-S, RegNetY-1.6GF, ConvNeXt-T, Swin-T, MaxViT-T, ViT-B/16, DINOv2 ViT-S/14) are fine-tuned with one deterministic recipe (AdamW, $1 0 ^ { - 4 }$ , batch 64, 20 epochs, resized-crop and flip augmentation) and scored at the final epoch without model selection;

per-crop logits and weights are kept.

One test per benchmark (H1, H2). Instead of one test per architecture, the protocol efect is tested once,

$$
\operatorname { a c c } \sim x + ( 1 + x \mid \operatorname { a r c h } ) + ( 1 \mid \operatorname { s e e d } ) + ( 1 \mid \operatorname { s p l i t } ) ,\tag{1}
$$

where $x = \pm \frac { 1 } { 2 }$ codes P1 vs. P2 and split, shared by all architectures within a seed, stops twelve evaluations of one test set from counting as twelve [14, 15]. A registered residual check selected a two-way cluster bootstrap as the primary inference, with Satterthwaite, Kenward–Roger, binomial-mixed-model and sign-flip tests as sensitivity analyses. Twelve architectures and twenty seeds gave a simulated power of 0.977 for a 0.8- pp efect on SIPaKMeD; CRIC uses six architectures and ten seeds.

Leakage vs. diversity (H3). The factorial (Fig. 1b) fixes one test set per seed, half the crops of a random 30% of multicrop SIPaKMeD fields (about 527 crops), and varies only the training set, always 220 crops per class. Clean cells draw from other fields; leak cells swap the last fields of the same permutation for the test fields’ remaining siblings. Within a diversity level, leak and clean cells thus share crop count, field count and allocation; 720 vs. 504 fields gives a 1.43-fold diversity contrast that brackets the natural 1.18. H3 is the leak efect in a mixed model (four architectures, ten seeds, 160 runs); diversity is estimated within clean cells.

Diagnostics. Study 2 registered background transplants (H4) and a gradient-reversal parent adversary (H5) [16]. The follow-up tests eight hypotheses (F1–F8, 940 new models; Fig. 1c). The H3 models are re-scored after background removal, illumination flattening, greyscale, low-pass filtering, or Reinhard transfer [17] of each cell’s color statistics to an unseen field of its own class (F2; the target is chosen by label, so this is a controlled diagnostic); the factorial is retrained with masked backgrounds (F3) or strong HED color jitter [18] (F4). Parent recoverability �, the within-class accuracy of a logistic field probe on frozen features, is lowered by rank-limited LEACE erasure with a refitted head [19] (F5), stain normalisation (F6) and a probe-matched linear adversary (F7), each required to remove a quarter of the above-floor � while costing at most 1 pp of P2 accuracy; F8 pools the dose–response, and F1 tests a slide-level label prior in CRIC. Grad-CAM [20] attribution outside the cell mask is compared descriptively, with a randomised-head check [21]. H1 stands alone; H2–H5 and F1–F8 are separate Holm families [22].

Matched-size factorial (SIPaKMeD)  
(a) SIPaKMeD: crop vs. field split  
![](images/a6e429f1f50c7cc8a4dd52aaa1a98f6f73b3022b19ee12e5fa994404048fc5ea.jpg)

![](images/9c9d8ec4fd88427830f7fc256e892d2e4d0d7950592a4ac3f51186e0b04461a8.jpg)  
(c)

![](images/3417bd90dc2a3e865dc532a1adbf7ff3ca96c3908f13a739a15f482bc155e5f4.jpg)  
Fig. 2. Consequence and causal isolation. (a, b) Crop- minus parent-level accuracy per architecture (mean and 95% � interval over seeds) and the registered pooled estimates (diamonds). (c) The SIPaKMeD factorial: accuracy on the common test set for leak and clean training sets at two diversity levels (bold: mean of four architectures; faint: each architecture), with the registered leakage and diversity estimates.

## 4. RESULTS

All 1,341 study runs and 1,580 follow-up tasks completed, none diverged, and every result matched its registered split hash. Table 2 lists all registered tests.

## 4.1. Crop-level splitting inflates both benchmarks

On SIPaKMeD, accuracy was 97.14% under P1 and 96.18% under P2 (H1 = 0.97 pp, 0.59–1.39), with all four sensitivity analyses at � <0.001. Every architecture scored higher under P1, by 0.66–1.44 pp (Fig. 2a): crop-level splitting hides a quarter of the errors. Merging near-duplicate-linked parents changed nothing (P2 − P3: 0.03 pp, −0.49 to 0.56), so shared parents, not stray duplicates, carry the efect, and the architecture ranking was largely preserved (Kendall � = 0.82). On CRIC, with about 29 mixed-class cells per slide, accuracy fell from 90.48% to 78.40% (H2 = 12.08 pp; 11.6–12.8 pp per architecture; Fig. 2b) and balanced accuracy from 78.53% to 58.89%; a size-matched crop-level control gave the same gap (12.11 pp). That control fixes training size, not slide composition, and the factorial below was run on SIPaKMeD only, so the CRIC gap is consistent with acquisition leakage but not causally decomposed. Our CRIC inputs are 100 × 100 windows around the published nucleus coordinates, so absolute accuracies are not comparable with other CRIC results.

## 4.2. On SIPaKMeD, the inflation is leakage, not diversity

At fixed crop count, field count and allocation, adding the test fields’ siblings raised accuracy on the common test set by 1.73 pp (0.95–2.51) in every architecture at both diversity levels, while 43% more fields without leakage did nothing (0.00 pp, −0.65 to 0.64; interaction −0.20 pp, � = 0.60; Fig. 2c). The leak replicated on fresh seeds (1.47 pp) and persisted with every background masked in training and testing (F3: 1.86 pp). In an exploratory split it was as large on test crops sharing no pixel with any training crop (1.68 pp) as on overlapping ones (1.78 pp): what helps is a seen field, not memorised pixels.

Table 2. All registered tests, in pp (H5, F7, F8: pp per pp of �; F1: pp of probability; F6: pp per unit strength), signed so that each hypothesis predicts a positive value. �: H1 bootstrap, others Holm. <sup>†</sup>Significant in the opposite direction.
<table><tr><td>Hypothesis</td><td>Est.</td><td>95% CI</td><td>p</td></tr><tr><td>H1 SIPaKMeD crop vs. field H2 CRIC crop vs. slide H3 Leakage, matched size H4 Own-background route</td><td>0.97 12.08 1.73 0.42</td><td>0.59, 1.39 11.16, 12.99 0.95, 2.51 -0.35, 1.18</td><td>&lt;0.001 &lt;0.001 0.02 0.23 X</td></tr><tr><td>H5 Gap tracks R (GRL) F1 CRIC slide-majority pull F2 color transfer cuts leak F3 3Leak, no background</td><td>-4.29 F4 HED jitter cuts leak F5 LEACE erasure cuts gap -0.01 F6 Normalisation cuts gap</td><td>0.06 -8.60,-1.37 1.27 0.38, 2.16 1.86 1.42, 2.30 -0.23 -0.81, 0.35 -0.10,0.07 0.02 -0.41,0.44 0.000 -0.038, 0.039</td><td>-0.01, 0.12 0.16 X 0.001 x† 0.03✓ &lt;0.001√ 1.00X 1.00X 1.00X</td></tr></table>

## 4.3. Linear decodability and saliency do not track the leak

Neither registered Study 2 mechanism test located the leaked signal: background transplants (H4) and gradient reversal (H5), which ConvNeXt-T evaded by making parents more decodable (� 29.5% to 55.1%), were not supported. Reducing the linearly decodable parent code did not move the gap either (Fig. 3a). Rank-256 LEACE cut � from 25.1% to 5.4%, removing 82% of its above-floor value (class-only floor 1.2%), in all twelve architectures at a 0.17-pp cost in P2 accuracy, and the gap stayed at 0.93–0.95 pp (F5); the linear adversary passed its check in all three architectures with no change (F7); pooled over 15 cells the slope was 0.000 (−0.005 to 0.005; F8), so a 20-pp loss of � moves the gap by at most 0.1 pp. Grad-CAM put 21.8% of attribution outside the cell in leak and clean models alike (diference −0.09 pp, −0.57 to 0.39; Fig. 3c), although its maps passed the randomised-head check (median Spearman $\rho \ = \ 0 . 0 8 )$ . On CRIC, seen slides pulled minority-class cells away from the slide majority (F1: −4.29 pp), which argues against a simple slide-majority prior.

(a) Erasing parent identity  
![](images/ad9c66b6bcc6b7717edb78122a46a363031cf3a809fc501b4e044eb983eccc96.jpg)  
Parent recoverability R (%)

(b) Leak effect under interventions  
![](images/43d4d8c605cca675e1e61fb36931a12567b5dbbd733a467308fcacfa10eb1c71.jpg)  
Leak − clean accuracy (pp)

(c) Grad-CAM outside the cell  
![](images/1b0450c4c6614fec726bcbff511d4f6eb092c3f5e9312c5cfffc7677bc4c4085.jpg)  
Clean models (%)  
Fig. 3. Diagnostics. (a) Each arrow takes one architecture from no manipulation to the strongest in-regime LEACE erasure or linear adversary: recoverability falls, the gap does not. (b) Leak minus clean accuracy under test-time interventions (color transfer targets an unseen field of the cell’s own class) and training-time color jitter. (c) Grad-CAM attribution outside the cell mask for leak and clean models of the same seed and diversity level; the diagonal marks equality.

Only color transfer moved the leak. Matching each test cell’s color statistics to an unseen field of its own class cut the leak from 1.72 to 0.46 pp (F2: 1.27 pp, 0.38–2.16), whereas flattening illumination (1.30 pp) or removing the background (1.17 pp), at comparable accuracy cost, did not (Fig. 3b); strong HED jitter in training left it intact (1.71 vs. 1.47 pp; F4). Because the transfer is label-conditioned and shifts the test distribution, it implicates field-specific appearance without identifying what the networks learned.

## 5. DISCUSSION AND CONCLUSION

Crop-level splitting inflates both public cervical cytology benchmarks we trained on. On SIPaKMeD the inflation is leakage: it appears when a test cell’s acquisition siblings are in training, not when training merely gains parent diversity. On CRIC, with 29 mixed-class cells per slide instead of four single-class crops per field, the gap is twelve times larger; without a factorial there, it is consistent with, not attributed to, sibling leakage. A benchmark’s exposure can be computed from the closed forms of Sec. 2 before any model is trained. Rankings largely survived on SIPaKMeD, but accuracies under crop- and parent-level splits are not comparable: on CRIC the protocol gap exceeds the spread between architectures.

The mechanism is narrowed, not identified. The advantage survives background masking, does not depend on shared pixels, does not follow a simple slide-majority prior and is unafected by reducing the linear parent code, while label-conditioned color transfer implicates field-specific appearance; nonlinear or spatially encoded parent information may still carry it. The methodological lesson holds regardless: models whose measured parent code was largely erased, and whose saliency matched that of clean models, still benefited from leakage. Probes establish availability, not use [23, 24], and neither probes, saliency nor augmentation can stand in for a grouped split.

Limitations. Parents are the deepest released units, not patients: parent-disjoint evaluation removes the leakage measured here, but patient-level grouping could reveal further dependence that these experiments cannot bound. The factorial trains on 1,100 crops rather than 3,239, establishing the leak’s existence and direction more firmly than an exact decomposition of H1. One training recipe was used, erasure acts on linearly readable pooled features, and the follow-up, registered after the Study 2 results, is confirmatory only for its own questions.

Recommendations. Releases should label every observation with its parent, and its patient where permitted; evaluations should split at the deepest released unit, publish hashed fold manifests and report the fraction of test observations with a training sibling. Code, provenance maps, split manifests, run records and a risk calculator are at https://github.com/ optimisticexquisite/cytofair.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This research was conducted retrospectively using de-identified human cytology images made available in open access by the creators of SIPaKMeD, Cx22, CRIC and Herlev. No new data were collected, and ethical approval was not required for this secondary analysis of public data.

## 7. ACKNOWLEDGMENTS

No funding was received for conducting this study. The authors have no relevant financial or non-financial interests to disclose. Claude (Anthropic), used through Claude Code, assisted in writing and debugging the experiment and analysis code, orchestrating the training runs, producing the figures, and drafting and editing the text of all sections. The authors reviewed all code, results, references and text, and take full responsibility for the content.

## 8. REFERENCES

[1] Marina E. Plissiti, P. Dimitrakopoulos, G. Sfikas, Christophoros Nikou, O. Krikoni, and A. Charchanti, “SIPaKMeD: A new dataset for feature and image based classification of normal and pathological cervical cells in Pap smear images,” in Proc. IEEE Int. Conf. Image Process. (ICIP), 2018, pp. 3144–3148.

[2] Guangqi Liu, Qinghai Ding, Haibo Luo, Min Sha, Xiang Li, and Moran Ju, “Cx22: A new publicly available dataset for deep learning-based segmentation of cervical cytology images,” Comput. Biol. Med., vol. 150, pp. 106194, 2022.

[3] Mariana T. Rezende et al., “CRIC searchable image database as a public platform for conventional Pap smear cytology data,” Scientific Data, vol. 8, no. 1, pp. 151, 2021.

[4] Nicole Bussola, Alessia Marcolini, Valerio Maggio, Giuseppe Jurman, and Cesare Furlanello, “AI slipping on tiles: Data leakage in digital pathology,” in Proc. ICPR Int. Workshops and Challenges, 2021, vol. 12661 of LNCS, pp. 167–182.

[5] Sayash Kapoor and Arvind Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” Patterns, vol. 4, no. 9, pp. 100804, 2023.

[6] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann, “Shortcut learning in deep neural networks,” Nat. Mach. Intell., vol. 2, no. 11, pp. 665–673, 2020.

[7] Gaël Varoquaux and Veronika Cheplygina, “Machine learning for medical imaging: Methodological failures and recommendations for the future,” npj Digital Medicine, vol. 5, pp. 48, 2022.

[8] Amelia Jiménez-Sánchez, Dovile Juodelyte, Bethany Chamberlain, and Veronika Cheplygina, “Detecting shortcuts in medical images – a case study in chest X-rays,” in Proc. IEEE Int. Symp. Biomedical Imaging (ISBI), 2023, pp. 1–5.

[9] Manxi Lin, Nina Weng, Kamil Mikolaj, Zahra Bashir, Morten Bo Søndergaard Svendsen, Martin Tolsgaard, Anders Nymark Christensen, and Aasa Feragen, “Shortcut learning in medical image segmentation,” in Proc. Med. Image Comput. Comput. Assist. Interv. (MICCAI), 2024, pp. 623–633.

[10] Okoi Michael Obeten, Abas Aliu, Omowumi Hafsat Aliu, Ahmed Jimoh, Aliyu Zibril, and Uchenna Christiana Ezeanya, “Data leakage inflates reported accuracy of deep learning for kidney stone detection on CT: A leakage-free, calibrated and uncertainty-aware reassessment across three centers,” SSRN preprint 7117201, 2026.

[11] Bhanu Prakash Vangala, Sowmya Guda, Latha Peddi, and

Navya Vangala, “Scores that hold, benchmarks that leak: Measuring dataset contamination in public brain-tumor MRI classification,” arXiv:2610.00421, 2026.

[12] Elodie Germani, Fatima Zeineddine, Charbel Mourad, and Shadi Albarqouni, “When multi-institutional dataset aggregation masks shortcut-like behavior in foundation models for medical imaging,” Med. Image Anal., vol. 114, pp. 104258, 2026.

[13] Jan Jantzen, Jonas Norup, Georgios Dounias, and Beth Bjerregaard, “Pap-smear benchmark data for pattern classification,” in Proc. Nature Inspired Smart Inf. Syst. (NiSIS), Albufeira, Portugal, 2005, pp. 1–9, NiSIS.

[14] Douglas Bates, Martin Mächler, Ben Bolker, and Steve Walker, “Fitting linear mixed-efects models using lme4,” J. Stat. Softw., vol. 67, no. 1, pp. 1–48, 2015.

[15] Alexandra Kuznetsova, Per B. Brockhof, and Rune H. B. Christensen, “lmerTest package: Tests in linear mixed efects models,” J. Stat. Softw., vol. 82, no. 13, pp. 1–26, 2017.

[16] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky, “Domain-adversarial training of neural networks,” J. Mach. Learn. Res., vol. 17, no. 59, pp. 1–35, 2016.

[17] Erik Reinhard, Michael Ashikhmin, Bruce Gooch, and Peter Shirley, “Color transfer between images,” IEEE Comput. Graph. Appl., vol. 21, no. 5, pp. 34–41, 2001.

[18] David Tellez, Geert Litjens, Péter Bándi, Wouter Bulten, John-Melle Bokhorst, Francesco Ciompi, and Jeroen van der Laak, “Quantifying the efects of data augmentation and stain color normalization in convolutional neural networks for computational pathology,” Med. Image Anal., vol. 58, pp. 101544, 2019.

[19] Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raf, and Stella Biderman, “LEACE: Perfect linear concept erasure in closed form,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2023, vol. 36.

[20] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra, “Grad-CAM: Visual explanations from deep networks via gradientbased localization,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 618–626.

[21] Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim, “Sanity checks for saliency maps,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2018, vol. 31.

[22] Sture Holm, “A simple sequentially rejective multiple test procedure,” Scand. J. Stat., vol. 6, no. 2, pp. 65–70, 1979.

[23] Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg, “Amnesic probing: Behavioral explanation with amnesic counterfactuals,” Trans. Assoc. Comput. Linguist., vol. 9, pp. 160–175, 2021.

[24] Abhilasha Ravichander, Yonatan Belinkov, and Eduard Hovy, “Probing the probing paradigm: Does probing accuracy entail task relevance?,” in Proc. Conf. European Chapter Assoc. Comput. Linguist. (EACL), 2021, pp. 3363–3377.