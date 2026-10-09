# Perception Test 2026: Challenge Summary and Extension to City-scale Audio-Visual Reasoning

Fedor Kitashov<sup>1</sup>, João Carreira<sup>1</sup>, Shiry Ginosar<sup>2</sup>, Dima Damen<sup>1,3</sup>, Andrew Zisserman<sup>1,4</sup> and Viorica Pătrăucean<sup>1</sup>

<sup>1</sup>Google DeepMind, <sup>2</sup>Toyota Technological Institute at Chicago, <sup>3</sup>University of Bristol, <sup>4</sup>University of Oxford

Continuing the Perception Test challenge series, we organised the fourth edition as a workshop at the European Conference on Computer Vision (ECCV) 2026 in Malmö, Sweden. This edition focused on spatial intelligence and featured four diferent tracks: unified multiple-choice videoQA and grounded videoQA from the original Perception Test benchmark, alongside two new tracks based on city-scale walking-tour videos (KilometerAudio and KilometerVision). In this report, we describe the new benchmarks used for city-scale tracks and summarise the winning solutions across all tracks, including a generalist model that competed across all tracks with satisfactory performance. The winning solutions in the newly added city-scale tracks demonstrated that complex spatial and multimodal reasoning can be solved by expensive agentic pipelines, but remains dificult for multimodal models used standalone.

Keywords: multimodal models, perception, spatial reasoning, long video understanding

## 1. Introduction

Frontier AI models have demonstrated superhuman capabilities in multiple areas, from solving Millennium problems in Mathematics (OpenAI, 2026), to controlling plasma for nuclear fusion reactors with superhuman precision (Citrin et al., 2024), and writing complex software at speed much higher than humans. Yet, the general perception capabilities of these models are still falling short compared to average humans, especially on spatial understanding tasks. This might not come as a surprise, given that these models are mainly trained with passive internet-scale data, whereas the common dogma states that environment interaction is essential for learning perception (Held and Hein, 1963; Poincaré, 2011).

In this context, we dedicated the fourth edition of the Perception Test challenge<sup>1</sup> to probing spatial intelligence in frontier models across diferent environment scales. We rely on the original Perception Test benchmark to evaluate small-scale (table-top) spatial capabilities. For large-scale spatial understanding, we leverage the recently introduced KilometerVision benchmark (Mahendran et al., 2026), together with a newly created benchmark for audio-visual reasoning in largescale environments; see details in sections 2 and 3.

Challenge tracks: The challenge featured four tracks: Unified VideoQA, Grounded VideoQA, KilometerVision, and KilometerAudio. The first two tracks relied on videos from the original Perception Test benchmark, whereas the latter two tracks relied on publicly available Youtube Walking-Tour videos, annotated specifically for each benchmark.

Challenge setup and prizes: The competition was hosted on the EvalAI platform. Each track had a validation and a test phase. The validation servers opened in mid-June 2026, and the test servers became available starting in mid-July (for unified and grounded VQA) and mid-August 2026 (for KilometerAudio and Kilometer-Vision). The submission period closed at the end of August 2026. Participants were required to report the evaluation mode (zero-shot, few-shot, or fine-tuned) and submit a short technical report. Only the test submissions made public on the leaderboard were eligible for awards with monetary prizes, which totalled 22K EUR across tracks. We awarded runner-up and winners in each track, together with a best generalist award for an entry that competed across all tracks with the same backbone and obtained strong performance. We also ofered honorary best performance certificates in KilometerVision and KilometerAudio tracks to a team that obtained the best scores in these tracks, but was not eligible for awards due to Google afiliation.

![](images/0e3a4aeb4f9679f156697ee39edb32d326ab4154f59ac36fe902d91266ac9a7c.jpg)  
Figure 1 | Example of map trace tasks in the KilometerVision benchmark. The model receives a video and a text question (“Which of these maps correctly represent the path of the person in this video?”) together with options given as raster images, with or without text labels inpainted on the maps.

## 2. KilometerVision Benchmark

KilometerVision (Mahendran et al., 2026) is a recent benchmark that probes city-scale spatial capabilities in frontier models using real-world walking tour Youtube videos, up to 10 minutes long, corresponding to about 1KM walking distance. The tasks use multiple-choice videoQA format, with three diferent task configurations: standard multiple-choice video QA with five text options; multiple-choice video QA where the five options are images (e.g. candidate route maps, see example in Figure 1); and multiple-choice video QA where the question includes an image (e.g. a landmark whose distance to the final camera position must be estimated, see example in Figure 2). The benchmark includes seven types of questions: landmark recognition, distance to landmark, compass, loop closure, route summary, map trace (with or without text labels on the map), and Euclidean distance. Please check the original publication (Mahendran et al., 2026) for more details about the tasks and data collection.

## 3. KilometerAudio Benchmark

To assess models’ audio-visual capabilities in large-scale environments, we introduce the KilometerAudio benchmark. This is a novel benchmark, introduced here for the first time. It aims to test the ability of foundational models, including agentic approaches, to reason about the interlink between audio and visual modalities in very long videos – i.e. videos captured over walking for several kilometers around an urban environment. Natural sounds that are automatically captured in these public YouTube videos include sounds of human-made objects (cars, motorbikes, church bells, etc.), natural sounds (birds chirping and dogs barking), speech as well as acoustic urban pollution. Importantly, the videos are not edited to introduce irrelevant sounds (e.g. overlay music), making them ideal for multimodal reasoning. Due to the nature of walking videos, audio is often caused by an event ‘out of sight’ of the camera. Reasoning about an in-view visual along with an out-of-view audio as well as the length of the videos make this benchmark particularly unique

![](images/d52ee32067756338cef7c7e9b1af483fd7ab6090714a53cda6aeba2d4e473313.jpg)  
Figure 2 | Example of landmark task in the KilometerVision benchmark, where the question contains a reference image in addition to the text prompt (“Does the image show a location in the video, and if so, what is the straight-line distance between the camera location of the image and the camera location of the final video frame?”).

![](images/bb44a43c92bc5a0a7df54860abd49f2e461ef20f08c0cf9a70fa4982b90c51c2.jpg)  
Figure 3 | Example KilometerAudio questions from the validation set (correct answer in green). Each requires reasoning over the audio track of an hour-long walking-tour video.

and challenging.

Compared to existing audio-video benchmarks, e.g. (Chao et al., 2026; Fu et al., 2025; Pătrăucean et al., 2023; Xie et al., 2025; Yang et al., 2024), KilometerAudio uses significantly longer videos (see Figure 4 for a comparison) and focuses exclusively on audio events. The test set contains 500 five-option questions over 218 publicly available hour-long Youtube walking-tour videos (1–3 h). Video-MME Long has videos up to 1h long, but only a small fraction of the questions focus on audio events.

All the questions in KilometerAudio were collected with human raters and checked by our research team. The raters were instructed to identify recognisable sounds in the videos and formulate questions that require both the audio and visual modalities to be answered: identifying sound sources in space, localising or timing audio events, counting them, recognising songs or speech; see Figure 3. The raters were also required to provide the answers to the questions, together with plausible dificult negative options. The validation split (8 questions on 6 videos) reuses the audio-related questions of the hour-long video QA benchmark of the 2024 and 2025 editions (Heyward et al., 2024). All test questions are new, and only 3 of the 218 test videos were used in previous editions. No training data are provided, so the track is zero-shot; see Table 1.

![](images/c5d3d4b424ae25668f79ae363e0bf54a8f8706a2a8c2bbd3a0a6337bb2cd3ead.jpg)  
Figure 4 | Multiple-choice audio-video benchmarks. The proposed KilometerAudio benchmark has the longest videos (more than 70 minutes on average) and the questions focus exclusively on audio events; Video-MME (Long) has videos up to 1h long, but only a fraction of the questions focus on audio events.

<table><tr><td>Split</td><td>Num videos</td><td>Num questions</td></tr><tr><td>Train</td><td>一</td><td>一</td></tr><tr><td>Validation</td><td>6</td><td>8</td></tr><tr><td>Test</td><td>218</td><td>500</td></tr></table>

Table 1 | Splits of the KilometerAudio track.

## 4. Overall challenge summary

We received 1254 test submissions from 120 different teams across the four tracks. As it can be observed in Figure 5, the performance improved steadily across editions on the unified and grounded videoQA tracks (using videos from the original Perception Test benchmark): from 0.812 (in 2025) to 0.914 top-1 accuracy in unified multiple-choice video QA in 2026, and from 0.497 (in 2025) to 0.664 HOTA in grounded video QA in 2026. In the two new tracks, the best-performing entry (non-eligible for prizes) reached close to perfect performance (0.864 on KilometerAudio and 0.905 on KilometerVision). The best prizeeligible submissions were also very strong, reaching 0.782 top-1 accuracy on KilometerAudio and 0.753 for KilometerVision.

Figure 6 shows the best test score of each track over the test phase. Due to data availability, the test servers for the two new tracks went public only later during August 2026.

## 5. Challenge Tracks, Results, Awards

For each track, we summarise the 2026-specific setup, the results, and the awarded methods. Team reports, with author names and afiliations, are linked from the workshop website: https://perception-test-challenge. github.io/.

## 5.1. Unified multiple-choice video QA

Task, data, and metric. These are unchanged from 2025; see Heyward et al. (2026, Sec. 2 and 4.1). Specifically, the test set combines the 11,528 original 3-option questions with the 1,842 5-option unified questions (point tracking, action ordering, object–action interaction, pretend action), which use visual prompts to recast tracking and temporal localisation as questions; the unified questions have no training split enforcing zero-shot evaluation. The metric is top-1 accuracy.

Baselines: A random baseline scores 0.313 on the test set (chance level 31.5%, i.e. 33.3% on the 3-option and 20% on the 5-option questions, weighted by their counts).

## Results:

<table><tr><td>Rank</td><td>Team name</td><td>top-1</td></tr><tr><td>Baseline</td><td>Random</td><td>0.313</td></tr><tr><td>Runner-up</td><td>F423 (zero-shot)</td><td>0.910</td></tr><tr><td>Best</td><td>Njust-KMG (fine-tuned)</td><td>0.914</td></tr></table>

Table 2 | Multiple-choice video QA results on the test set.

Table 2 lists the awarded entries compared to the random baseline. The winning team, Njust-KMG, used a question-type-aware weighted ensemble of three models: Qwen3-VL-32B finetuned on the Perception Test training set (with a curriculum stage on the hardest question types and option shufling to reduce position bias), and the Doubao-Seed-2.0-Pro and Gemini 3.1 Pro APIs queried zero-shot with question-type-specific prompts; the ensemble weights are derived from per-type validation accuracy. While the individual models obtain 78–83% on the test set, the ensemble reaches 91.4%. The runner-up, F423, used a single zero-shot model, Doubao-Seed-2.0- Lite, with evidence-first prompting (extracting the visual, temporal, and audio evidence before comparing the options), adaptive frame sampling (1–2 fps depending on the question type), and strict answer validation with at most one re-query, reaching 91.0%. Please check the workshop website for more details on the methods included in the submission reports.

![](images/871389473498b17dfddbf673aa6c2479531e7b2eb0e5ae9dbc90a1b966c5b49e.jpg)  
Figure 5 | Per-track performance of the 2026 winners compared to baselines and to the best models of previous years, all on the same test sets. \*For multiple-choice video QA we show accuracy on the 11,528 original questions, the only test questions common to all editions. KilometerAudio and KilometerVision were introduced in 2026; striped bars show the best-performing but non-prize eligible entry.

Relative to earlier editions (Figure 7), the 2026 winner improved significantly on the questions added in 2025 (86.2% vs 65.4% for the best 2025 model) compared to improvement on the original set (92.2% vs 83.7%). This indicates that the reformulated tracking and localisation questions are quickly becoming tractable for current models.

Figure 8 breaks the winner’s accuracy down by question type and shows which areas the quasisolved (> 95%) and the hardest questions come from. Of the 141 questions, 66 (46.8%) are now quasi-solved, compared with 36 (25.5%) for the 2025 winner on the same questions, and only 2 questions obtain an accuracy of 50% or lower (14 in 2025), both in the Physics area. Physics also dominates the 10 hardest questions (accuracy ≤ 80%; 8 Physics, 1 Spatial reasoning, 1 Abstraction). The remaining errors therefore concentrate on intuitive physics (e.g. judging the temperature of poured water, or predicting outcomes when imagining a change in velocity or friction of an object) rather than on recall or counting.

![](images/7830aaab693c79ab404bf9fdeda571bbd81ed8d9f914bccf6c08b0351c48944f.jpg)  
Figure 6 | Best test score in each track during the 2026 test phase. The test servers for Kilometer-Audio and KilometerVision went live in August 2026, pending data approvals.

## 5.2. Grounded video QA

Task, data, and metric. These are identical to previous editions (Heyward et al., 2026): given a video and a question, the model must track the object(s) that answer it, on the same 932 test videos (1,859 questions), evaluated with HOTA (Luiten et al., 2020). The baseline runs the MDETR detector (Kamath et al., 2021) on the middle frame with the question as query and keeps the detections static throughout the video.

Results: Table 3 lists the two awarded entries compared to the baseline. Both solutions use a strong VLM to decide which objects answer the question and a SAM-family model for dense tracking. The winning team, Banana Bats, proposed Gemini-SAM3: Gemini 3.7 Flash, used zero-shot, builds a tracking plan (target identities, visibility intervals, anchor frames and boxes, checked by a critic) and provides sparse boxes at 1 fps; SAM 3 (Carion et al., 2025), prompted with the anchor boxes, propagates the targets bidirectionally, and its tracks are fused with the sparse Gemini tracks only when they pass reliability checks. A variant of SAM 3 with a small residual head trained for amodal (occlusion-aware) boxes on external data is selected for multi-target questions. The runner-up, Huawei SpaceMind, used GPT-5.6 Sol zero-shot to identify the answer objects and produce complete box tracks, then refined only the box coordinates, never the object identities, using a second independent track, repeated queries on close-up crops, and SAM 2.1 (Ravi et al., 2025) box refinement.

![](images/3374769e11e3c6bf0834614aced8a19c5d353009b4b34de1b19f1a359d1209e2.jpg)  
Figure 7 | Best multiple-choice video QA test results per edition. The 11,528 original questions are common to all editions, and the 1,842 unified questions were added in 2025; 2023, 2024, and the frequency baseline (striped) are therefore on the original questions only, while for 2025 and 2026 we show the accuracy on all 13,370 questions and on each subset.

Figure 9 compares the top 2026 model to the best entries of previous editions: HOTA increased from 0.497 to 0.664, about one third higher, with similar gains in detection (0.479 to 0.654) and association accuracy (0.524 to 0.677). In 2023 the best entry barely exceeded the static baseline (0.063 vs 0.052 HOTA). Only three years later, zero-shot pipelines that pair a VLM with a promptable tracker produce very strong performance, showing the high pace of progress in the field.

<table><tr><td>Rank</td><td>Team name</td><td>HOTA DetA</td><td>AssA</td></tr><tr><td>Baseline</td><td>MDETR+static</td><td>0.052 0.033</td><td>0.091</td></tr><tr><td>Runner-up</td><td>Huawei SpaceMind</td><td>0.654 0.651</td><td>0.661</td></tr><tr><td>Best</td><td>Banana Bats</td><td>0.664 0.654</td><td>0.677</td></tr></table>

Table 3 | Grounded video QA results on the test set.

## 5.3. KilometerVision

Task, data, and metric: The model needs to answer 5-way multiple-choice questions that can have diferent configurations as mentioned in section 2. The challenge test set contains 986 questions on 54 walking-tour video segments, spanning six question types: landmark recognition and distance to landmark (147), compass (146), loop closure (112), route summary (147), start– end Euclidean distance (150), and map trace with or without text labels on the map (284 questions); see Table 4. No training data are provided. The evaluation metric is top-1 accuracy over the five options.

<table><tr><td>Split</td><td>Num videos</td><td>Num questions</td></tr><tr><td>Train</td><td>一</td><td>一</td></tr><tr><td>Validation</td><td>13</td><td>17</td></tr><tr><td>Test</td><td>54</td><td>986</td></tr></table>

Table 4 | Splits of the KilometerVision track.

Baselines: A random baseline obtains 0.196 on the test set (expected chance level 0.20).

Results: The results are included in Table 5 and Figure 10. The top solutions combine a VLM for perception and world-knowledge grounding with explicit geometric estimation. The winning team, ohmyyuan, proposed evidence-log video agents: Gemini 3.7 Flash builds a timestamped log of street names, heading changes, and landmarks from frame montages and keyframes, and answers with a family-specific playbook; selfconsistency over several sessions is combined with deterministic estimators (optical-flow yaw odometry and visual place recognition) that can override the majority vote, and an independent verifier re-solves questions without a clear majority. The runner-up, Huawei SpaceMind, combined GPT-5.6 Sol for semantic reasoning with a metric camera path reconstructed with VGGT (Wang et al., 2025) and a metric-depth model; questionspecific solvers (summed yaw for compass questions, start–end displacement for distances, and route-shape matching for map questions) decided 642 of the 986 answers. The CloudAI-evolve team, which was not eligible for a prize, obtained the best overall score (0.905) and was highlighted as the best-performing entry. Their approach tackled the task with an agentic pipeline with unlimited token budget that explored the validation set and a small fraction (about 10%) of the unlabelled test questions to learn a strategy for solving the task on the full test set. The strategy involved calling metric 3D reconstruction models (VGGTomega), object-size-based scale estimation tools, and grounding the videos on the map.

![](images/b1683f5bb017fc32d4dcc581c78e83b4123cb89d21514b6430b3a5b8ce043779.jpg)

![](images/24499551998d3ceae25f988bb20cfa441feb67dee70b88c77d044504ecd9178a.jpg)

![](images/cc96ccd541b32796cbc3790ee1de732876fa63a850bca4bd1f83d1afedeab78a.jpg)

Figure 8 | Left: accuracy of the best 2026 submission (Njust-KMG) on each of the 141 unique questions. Right: distribution across areas of the questions that are quasi-solved (accuracy > 95%, top) and of the 10 hardest questions (accuracy ≤ 80%, bottom).  
![](images/bed60a8f9da08cd4ff7cf34f94e9a53bb71bc70af3870f66ffeaaa2365852e1d.jpg)  
Figure 9 | Baseline vs best results over time in terms of overall HOTA, detection, and association accuracy for the grounded video QA task.

<table><tr><td>Rank</td><td>Team name</td><td>top-1</td></tr><tr><td>Baseline</td><td>Random</td><td>0.196</td></tr><tr><td>Runner-up</td><td>Huawei SpaceMind</td><td>0.723</td></tr><tr><td>Best</td><td>ohmyyuan</td><td>0.753</td></tr><tr><td>Best-performing†</td><td>CloudAI-evolve</td><td>0.905</td></tr></table>

Table 5 | KilometerVision results. <sup>†</sup>Bestperforming entry, not eligible for a prize; see text.

## 5.4. KilometerAudio

Task description: In the KilometerAudio task, the model receives a full hour-long walking-tour video, including its audio track, together with a question and five possible answers, out of which only one is correct. The questions require processing both the visual and the audio track of the video, e.g. identifying sounds, understanding speech, or correlating audio events with visual content, possibly through multi-hop reasoning.

Metric: The evaluation metric is top-1 accuracy, i.e. the percentage of questions where the predicted option (1 out of 5) matches the ground truth.

![](images/8fb2c9fb3e00a8e0628aa04f1b47eeeae798047a86b9492e669e4a21770359ac.jpg)

KilometerVision — Top 5 Teams  
![](images/c303c9b57ba1eb5512ee214e0ce369dd4b14ca29f2c5223f94559650011d51a8.jpg)  
Figure 10 | Top-5 public test results (top-1 accuracy) in KilometerAudio (top) and Kilometer-Vision (bottom), including the best-performing entry, CloudAI-evolve.

Baselines: A random baseline obtains 0.174 on the test set (expected chance level 0.20).

Results: The results are included in Table 6 and Figure 10. The winning team, IUCV, proposed AVRA, a training-free audio-visual reasoning agent built on Gemini 3.7 Flash. For each video, a question-independent index is precomputed with frozen models: audio-event scores for 447 AudioSet classes from BEATs (Chen et al., 2023), object detections, and multilingual OCR. At test time, a fresh agent session localises a few candidate time windows from the index, inspects the raw audio and video directly, keeps an explicit evidence log, and derives an answer before reading the options. The runner-up, WW11, proposed a zero-shot tool-using agent in which Gemini 3.1 Pro plans the search and reasons over at most 48 frames or crops per question, using ofline indexes of timestamped speech transcripts (Whisper) and CLAP-based sound-event retrieval, and an audio model (Qwen3-Omni) to verify short audio excerpts; an answer is changed only when at least two independent modalities agree. The CloudAI-evolve team obtained again the best overall score (0.864) with a similar agentic pipeline as in the KilometerVision track, but using signal-processing tools instead of 3D reconstruction, to compile a strategy for solving the tasks; Gemini models were used for all reasoning steps. More details can be found on the challenge website.

<table><tr><td>Rank</td><td>Team name</td><td>top-1</td></tr><tr><td>Baseline</td><td>Random</td><td>0.174</td></tr><tr><td>Runner-up</td><td>WW11</td><td>0.752</td></tr><tr><td>Best</td><td>IUCV</td><td>0.782</td></tr><tr><td>Best-performing†</td><td>CloudAI-evolve</td><td>0.864</td></tr></table>

Table 6 | KilometerAudio results. <sup>†</sup>Bestperforming entry, not eligible for a prize; see text.

## 5.5. Best Generalist award

The Best Generalist award was received by Wind\_Rain\_Tower team, who submitted to all four tracks with a single training-free approach. The Qwen3.8-Max model makes every final answer and box decision, without task-specific detectors, trackers, or geometry solvers. Only a separate speech-recognition model is used to transcribe the audio for KilometerAudio as the model cannot process audio natively. The results of the generalist model were: 0.896 top-1 in multiplechoice video QA (5th place, 98% relative to the top solution), 0.465 HOTA in grounded video QA (13th place, 69% of top), 0.604 in KilometerVision (9th place, 80% of top), and 0.506 in KilometerAudio (22nd place, 65% of top). Although the model trailed behind the per-track winners, this is the first submission across the four editions that shows a unified model with satisfactory performance across all the tracks, giving hope that a complete perception model is within reach.

## 6. Discussion

The Fourth Perception Test challenge attracted a record number of submissions (1254 test submissions) from 120 teams across four tracks. Except for the Grounded VideoQA track, the performance on the remaining three tracks almost saturated (> 80% accuracy). This is remarkable, especially given that two of the tracks, the city-scale tracks, have been run this year for the first time.

Almost all top solutions were zero-shot agentic pipelines built around proprietary or open-source VLMs (Gemini, GPT, Doubao-Seed, Qwen), often combined with ensembling or self-consistency metrics. Among the awarded entries, only the unified multiple-choice winner fine-tuned a model on the Perception Test training data. The best performing methods in the new city-scale tracks relied on expensive and slow agentic tool use, audio indexing, or explicit geometric reconstruction, but obtained almost perfect performance. In contrast, the Best Generalist entry, which used a single multimodal model for all tasks, stayed close to the winners on multiple-choice video QA, but trailed behind on the city-scale tracks. In conclusion, large-scale audio-visual and spatial reasoning can be solved by systems that orchestrate models with explicit tools, but remains an open problem for multimodal models used standalone.

## Acknowledgements

We thank Google DeepMind for funding the prizes and swag, and the EvalAI team for their help in hosting the four tracks.

## References

N. Carion et al. SAM 3: Segment anything with concepts, 2025. URL https://arxiv.org/ abs/2511.16719.

J. Chao, jianzhang gao, W. Tan, Y. Sun, R. Song, and L. Ru. JointAVBench: A benchmark for joint audio-visual reasoning evaluation. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/ forum?id=Zg1YH8R5GG.

S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, W. Che, X. Yu, and F. Wei. BEATs: Audio pre-training with acoustic tokenizers. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scar-

lett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 5178–5193. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr. press/v202/chen23ag.html.

J. Citrin, I. Goodfellow, A. Raju, J. Chen, J. Degrave, C. Donner, F. Felici, P. Hamel, A. Huber, D. Nikulin, D. Pfau, B. Tracey, M. Riedmiller, and P. Kohli. Torax: A fast and diferentiable tokamak transport simulator in jax. preprint arXiv:2406.06718, 2024.

C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang, P. Chen, Y. Li, S. Lin, S. Zhao, K. Li, T. Xu, X. Zheng, E. Chen, C. Shan, R. He, and X. Sun. Videomme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24108–24118, June 2025.

R. Held and A. Hein. Movement-produced stimulation in the development of visually guided behavior. Journal of Comparative and Physiological Psychology, 56(5):872, 1963.

J. Heyward, J. Carreira, D. Damen, A. Zisserman, and V. Pătrăucean. Perception test 2024: Challenge summary and a novel hour-long videoqa benchmark, 2024. URL https:// arxiv.org/abs/2411.19941.

J. Heyward, N. Parthasarathy, T. Zhu, A. Mahendran, J. Carreira, D. Damen, A. Zisserman, and V. Pătrăucean. Perception test 2025: Challenge summary and a unified vqa extension, 2026. URL https://arxiv.org/abs/2601. 06287.

A. Kamath, M. Singh, Y. LeCun, I. Misra, G. Synnaeve, and N. Carion. Mdetr–modulated detection for end-to-end multi-modal understanding. arXiv preprint arXiv:2104.12763, 2021.

J. Luiten, A. Osep, P. Dendorfer, P. Torr, A. Geiger, L. Leal-Taixé, and B. Leibe. Hota: A higher order metric for evaluating multi-object tracking. International Journal of Computer Vision, pages 1–31, 2020.

A. Mahendran, M. King, M. K. Grimes, A. Yang, T. Zhu, J. Heyward, T. Han, S. Ginosar, C. Sun, D. Damen, S. Osindero, N. Snavely, S. Lynen, J. Carreira, and V. Pătrăucean. KilometerVision: A new frontier for large-scale spatial intelligence in VLMs. In European Conference on Computer Vision (ECCV), 2026.

OpenAI. On the navier–stokes millennium prize problem, September 2026. URL https://openai.com/index/ navier-stokes-solution/.

H. Poincaré. Science and Hypothesis. Dover Publications, 2011.

V. Pătrăucean, L. Smaira, A. Gupta, A. R. Continente, L. Markeeva, D. Banarse, S. Koppula, J. Heyward, M. Malinowski, Y. Yang, C. Doersch, T. Matejovicova, Y. Sulsky, A. Miech, A. Frechette, H. Klimczak, R. Koster, J. Zhang, S. Winkler, Y. Aytar, S. Osindero, D. Damen, A. Zisserman, and J. Carreira. Perception test: A diagnostic benchmark for multimodal video models. In Advances in Neural Information Processing Systems, 2023. URL https://openreview. net/forum?id=HYEGXFnPoq.

N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollar, and C. Feichtenhofer. SAM 2: Segment anything in images and videos. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=Ha6RTeWMd0.

J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny. VGGT: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

L. Xie, G. Z. Wei, A. Kuthiala, C. Zheng, A. Bal, M. Dabhi, L. Wen, T. Rustagi, E. Lai, S. Khyalia, R. Choudhury, M. Ziyadi, X. Zhang, H. Yang, and L. A. Jeni. Maverix: Multimodal audiovisual evaluation in real-world interactions. In Proceedings of AAAI, 2025. \* Equal contribution.

Q. Yang, J. Xu, W. Liu, Y. Chu, Z. Jiang, X. Zhou, Y. Leng, Y. Lv, Z. Zhao, C. Zhou, and J. Zhou. AIR-bench: Benchmarking large audio-language models via generative comprehension. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1979–1998, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long. 109. URL https://aclanthology.org/ 2024.acl-long.109/.