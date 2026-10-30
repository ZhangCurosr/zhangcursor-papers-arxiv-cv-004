# Telling Is Not Seeing: Textual Personas Rarely Make Vision–Language Models Answer Like the People They Describe

Jiawei Chen

Netflix

jiaweic@netflix.com

## Abstract

Vision–language models (VLMs) are increasingly prompted with personas, such as “You have low vision”, to stand infor people in user studies. We ask whether a persona makes a model see like the person it describes. We compare persona answers with measured people with low vision, colour blindness or hearing loss, and with simple rules that use only the facts the persona states. The persona never beats these rules, and it erases the real differences between groups and individuals. Describing blur leaves recognition unchanged in all 14 models we test, while blurring the photo lowers it as it does for people. Inside open models, the description changes neither the image information nor the answer probabilities. Yet the models know the effects they fail to show: they can estimate how blur and illusions affect people. Telling is not seeing: a persona makes a model say what it knows about people, not see what they see.

## 1. Introduction

Designers need to know whether a person with low vision can read a sign, or a deaf viewer can follow fast subtitles, and user studies are slow and costly. A vision–language model (VLM) with a persona prompt, such as “You have low vision,” is an attractive replacement. Language models already stand in for survey respondents and users [1, 2, 28, 38]. This paper asks whether a persona makes a VLM answer like the people it describes.

Earlier work gives mixed answers. In a study of 40 people with low vision, vision information in the prompt did not bring GPT-4o closer to their answers [36]. A textual lowvision persona did not lower a VLM’s reading performance to the expected level [10]. Colour-blind personas still read Ishihara plates (dot patterns that hide a digit) like a person with normal colour vision [21]. These studies compare a persona with a minimal prompt (no persona) or with an answer key. When a persona helps, this cannot show whether the model simulates the person or the persona simply con-

SHOW the blur: blurred photo

![](images/e1393a7a0e160bb6eccefceefe3b838eb14bbe3dc75c5b1e3299d9aa7c75b59c.jpg)  
TELL the blur: persona

![](images/a8e343f477904a715bb5783ac730c43bbc5513d7250209c5d788e5f6bf956857.jpg)  
Object recognition as blur grows

![](images/74dcd24c48cf8088bca52a0873515ddc1f01009cd07c535a2cf1a98cfd7516a3.jpg)  
Figure 1. Telling a model about blur does not make it see blur. Four sighted people and Gemini name objects in photos at eight blur levels. Top: Gemini is either shown the blurred photo, or given the clean photo and told about the blur by a persona that states the matching acuity. Bottom: shown the blur, Gemini loses accuracy as people do; told about it, Gemini stays at 98–99%. The photos illustrate the blur levels; the test photos are greyscale.

tains a useful fact, such as a stated acuity. None of these studies tests whether personas keep the measured differences between groups.

We test both points. The key distinction is between knowingfacts about a viewer and answering as that viewer. For each persona, we also make the simplest prediction from its stated facts, with no persona prompt (a task-specific control): a clinical rule or a textbook answer that uses only the stated facts, or the model without the persona. A persona that simulates a person must at least match it. We also compare personas with several measured groups, and test whether the personas keep the differences between these groups (Tab. 1).

Current frontier models do change their reading and colour answers with a persona (Figs. 2 and 3), but not toward the people. A blur test suggests why (Fig. 1), and illusion tests show that models know effects that their own answers lack.

## Contributions.

1. An evaluation of personas against measured people. We compare personas with measured people: readers with low vision, colour-blind viewers, and subtitle viewers in five groups, including deaf and hard-of-hearing viewers. Each persona must at least match a task-specific control that uses only the facts the persona states, and must keep the differences between groups and individuals. No persona passes these tests.

2. A controlled test of telling versus showing. On the same photos, we compare a description of blur with the blurred photo itself, against human effect sizes, in 14 VLMs. The description has no effect in any model; the blurred photo gives seven models a loss as large as people’s.

3. An inspection of the models’ internals. Through answer probabilities and linear probes of the hidden states of open models, we show that the persona changes neither the image information nor the decision, and adds no blur signal.

4. A separation of what models know from what they show. On the same images, the models’ estimates of people’s answers recover most of the human illusion and blur effects, while their own answers show little or none of them.

## 2. Related Work

Personas and synthetic participants. Persona prompts can caricature groups [7], flatten the differences within them [51] and misrepresent their opinions [41]. Language models describe a group’s opinions better than they answer as its members [34]. Personas can also change what a model says about itself more than what it does [20], and can trigger refusals or biased reasoning [19], including on vision tasks [40]. Synthetic participants have been tested on images, video and visual marketing [43, 48]. Simulated-user platforms select cohorts by accessibility needs and check that persona agents follow their stated attributes [28]. Following an attribute is not answering like the person: our reading personas follow the stated acuity almost exactly, yet cannot beat the acuity rule. Natalie et al. [36] built one persona for each of 40 participants with low vision: vision information alone left GPT-4o’s agreement with them at 0.59, as with a minimal prompt, and adding one example answer raised it to 0.70. We add measured people, changed colours, task-specific controls, and the differences between groups.

VLMs compared with human vision. VLMs struggle with simple geometry [39], basic visual patterns [47], core perception tasks [9], human-like visual search [30], colour [29] and contrast sensitivity [22]. Human data on distorted images [11–14, 24] show where models differ from people. A filter that copies human contrast sensitivity makes the errors of image classifiers more human-like [53]. Illusion benchmarks compare model and human answers [42, 56], and models often answer from memory of a famous illusion [18, 49]. We compare a model’s own answers with its estimates of how people answered on the same images.

Table 1. Overview of the five studies. Frontier models: Gemini-3.8-Flash, Claude-Opus-5.5, GPT-6-Sol. <sup>a</sup>Three of the four open models cannot do the task without a persona. <sup>b</sup>One open model cannot run the subtitle-speed clips.
<table><tr><td>Study: task</td><td>People (data)</td><td>The persona states</td><td>We compare with</td><td>Models</td></tr><tr><td colspan="3">Primary tests: people whose own answers are recorded (Sec. 4.1)</td><td rowspan="2">The rule “reading limit 3 frontier, = letter-chart acuity&quot;; an 4 open</td></tr><tr><td>Low-vision reading: read sentences whose print size is stated</td><td>56 readers with low vision [54]</td><td>The reader&#x27;s letter-chart acuity, contrast sensitivity and diagnosis</td><td>image filter</td></tr><tr><td>Colour sorting (D-15): order 15 caps by hue</td><td>39 colour-blind viewers (group scores) [50]</td><td>The type of colour blindness</td><td>No persona; the textbook order for the stated type; colours changed for that type</td><td>3 frontier, 4 opena</td></tr><tr><td>Subtitle viewers: choose viewers in line breaks; rate five clips; state preferences</td><td>88-94 groups [45]</td><td>The participant&#x27;s language, age, gender, English level and hearing</td><td>The same model without a persona</td><td>3 frontier, 4 openb</td></tr><tr><td>Blur: name the 4 observers object in a photo</td><td>with normal vision [14]</td><td>Controlled test: sighted people view changed images (Sec. 4.2) The acuity that The blurred photo, no matches the blur</td><td>persona</td><td>14</td></tr><tr><td colspan="5">Diagnostic test: the model estimates a group&#x27;s answers (Sec. 4.3) An age group people [33] (optional)</td></tr><tr><td colspan="3">Illusions: 256 choose the longer line or larger disc</td><td>The model&#x27;s estimate of 3 frontier; the share of people who 11 open choose each option</td><td>(4 types)</td></tr></table>

## 3. Method

Three kinds of answer. For each item, we use the question that the people answered. The three kinds of answer differ in who answers the question and in what the model receives.

• A persona answer: the model answers the question itself, as the person that the persona describes. It sees the original image or video and the persona. We score its answer as we score a participant’s answer. For example, it sees a clean photo with the persona “your best-corrected visual acuity is $2 0 / 1 6 5 '$ and names the object as that person would.

• A changed-image answer: the model answers the question itself, with no persona, but it sees a changed image. For example, it sees the photo blurred to match that acuity.

• An estimate: the model does not answer the question. It stays itself and estimates, as a number, how a group of people answered it, not how one person answered. For example, it sees an illusion image and reports the percentage of adults who chose the left line.

Thus, a persona answer tests whether the model can answer as a person, and an estimate tests whether the model knows how a group of people answers. A persona that makes a model answer like a person should change the persona answer in the same way as the person’s answer changes. An estimate can be correct even if the model does not see the image as people do. Thus, we keep estimates apart from the model’s own answers. Formally, let i index an item (an image, a sentence or a video), g a viewer or viewing condition with $g = 0$ the reference, M the model, x<sub>i</sub> the original input, p<sub>0</sub> a neutral prompt and $p _ { g }$ the persona of g. The three kinds of answer are

$$
\begin{array} { c } { { A _ { i } ^ { P } ( g ) = M ( x _ { i } , p _ { g } ) , \qquad A _ { i } ^ { T } ( g ) = M \bigl ( T _ { g } ( x _ { i } ) , p _ { 0 } \bigr ) , } } \\ { { \hat { Y } _ { i } ( g ) = M ( x _ { i } , q _ { g } ) , } } \end{array}\tag{1}
$$

where $T _ { g }$ changes the input as condition g would (for example, a blur), and $q _ { g }$ asks the model to estimate how people in group g answered item i. The answer without a persona is $A _ { i } ^ { 0 } = M ( x _ { i } , p _ { 0 } )$ . Only one part of the input differs between $A _ { i } ^ { 0 }$ and $A _ { i } ^ { P } ( g ) \mathrm { : }$ : the text.

The persona must match a task-specific control. A persona can help for a simple reason: it states a useful fact. A reader’s letter-chart acuity, for example, is a good guess of the smallest print that the reader can read. So, for each study, we compare the persona with the simplest use of its facts. In reading, we copy the stated acuity as the predicted reading limit; we call this control the acuity rule, a clinician’s first guess. The persona also states the contrast sensitivity and the diagnosis, and the model knows that macular disease makes reading harder than letter acuity suggests, so a model that simulates the reader should move away from the rule in the right direction. In colour sorting, the controls are the textbook order for the stated type and the model without a persona; in the subtitle tasks, the control is the model without a persona. Each control uses no more than the persona states.

How we compare model and human effects. Each test has a reference condition, such as a clean photo, and a test condition, such as a blurred photo. The human effect is the change in people’s average score from the reference to the test condition. The model effect is the same change in the model’s score. The relative effect is the model effect divided by the human effect. A value of 1 means that the model changes as much as people; 0 means that it does not change. For example, if blur lowers people’s accuracy by 50 points and a model’s accuracy by 25 points, the relative effect is 0.5. Formally, let $s _ { i } ( \cdot )$ score an answer (for example, correct or not, or a rating) and let $Y _ { i } ^ { H } ( g )$ be the human answers. Then

$$
\begin{array} { r } { \Delta _ { H } ( g ) = \mathbb { E } _ { i } \left[ s _ { i } ( Y _ { i } ^ { H } ( g ) ) - s _ { i } ( Y _ { i } ^ { H } ( 0 ) ) \right] , } \\ { R ( g ) = \Delta _ { M } ( g ) / \Delta _ { H } ( g ) , \qquad } \end{array}\tag{2}
$$

where $\Delta _ { M } ( g )$ is defined in the same way for a kind of model answer in Eq. (1). When a test has several levels, such as blur strengths, we sum both effects over the levels. We call a model effect as large as people’s when the 90% confidence interval of R lies in [0.80, 1.25] (an equivalence test [25]; a 90% interval gives a 5% error rate for this test). We say that a persona has no effect when this interval lies in [−0.20, 0.20]. Because we test many models, we use the Holm correction [23], which keeps the chance of any false claim across the models at 5%. Square brackets give 95% confidence intervals, computed by resampling items and people (bootstrap).

Same average, different errors. A model and a person can both miss 40 of 100 photos but miss different photos. Thus, we also report error agreement (κ [13]): 0 means that the two miss photos independently, and 1 means that they miss the same photos. Formally, i $\mathrm { ~ f ~ } c _ { \mathrm { o b s } }$ is the share of photos on which model and person are both right or both wrong, and $a _ { M }$ and $a _ { H }$ are their accuracies, then

$$
\kappa = { \frac { c _ { \mathrm { o b s } } - c _ { \mathrm { e x p } } } { 1 - c _ { \mathrm { e x p } } } } , \qquad c _ { \mathrm { e x p } } = a _ { M } a _ { H } + ( 1 - a _ { M } ) ( 1 - a _ { H } ) .\tag{3}
$$

Models. The core panel is three frontier VLMs, proprietary models that we call through their providers’ interfaces at their default temperature: Gemini-3.8-Flash (Gemini), Claude-Opus-5.5 (Opus) and GPT-6-Sol (Sol). We also run openweight models from seven families [3, 4, 8, 15, 27, 35, 37, 52, 55, 57], without fine-tuning. One rule applies to every study. The core panel and the same four open models (Qwen3-VL-8B, Qwen2.5-VL-72B, InternVL3.5-8B, Gemma-4-12B) run every primary image task. A persona result counts only if the model can do the task without a persona. Video tasks use only models that accept the input; Opus and Sol take no video, so they get time-labelled frames. The blur test also runs seven more open models (14 in total). Each persona answer is one sample. Repeating 10% of the calls shows that this is stable: a persona’s reading limit changes by less than one chart line (0.03 to 0.07 logMAR), and the personas still barely differ from each other in the subtitle task (supplement).

## 4. Results

## 4.1. Do personas answer like measured people?

Low-vision readers: two different acuities. Xiong et al. [54] published vision and reading data for 58 people with low vision. We use the 56 whose vision lies inside the range where their filter method is valid. Two numbers describe each reader. The letter-chart acuity is the smallest letter size that the reader can identify on an eye chart. The reading limit is the smallest print size at which the reader can read sentences. Both are in logMAR units, where a larger value means larger print and 0.1 is one line on a chart [26]. The two numbers often differ. For example, a reader with macular disease (damage to the centre of the retina) can identify single letters but fail to read sentences of the same size.

(a) 56 readers: measured limit vs Gemini's persona  
![](images/0a7df5de88a761d1e0098f55ca6f2ff0742d41c89eb56f4497683fb24ea246e0.jpg)

(b) Error of each way to tell the model about the reader  
![](images/1b60537ba49160e04183578c801f7c4383e69dae5984ad71583bba0c4b7f633e.jpg)  
Figure 2. A reading persona copies the stated letter-chart acuity, so it misses readers whose reading limit is different. (a) Stated letter-chart acuity against reading limit, per reader (log-MAR). Black: measured limit. Red: Gemini with the persona. Dashed: acuity rule. (b) Mean error of the frontier models for three ways of telling the model about the reader; none beats the acuity rule (dashed, 95% interval).

The task. The readers took the MNREAD reading test in the original study; we use only their published results. The model takes the same kind of test, once for each reader’s persona. It copies 24 sentences, one per print size from 2.0 to −0.3 logMAR. We wrote these sentences ourselves, because the MNREAD sentences are copyrighted. Every image shows its sentence at the same legible pixel size, so a failure can only come from the persona, not from what the model can see; the image filter below tests the seeing route. The print size reaches the model only as a number in the prompt, or through an image filter. We score the reading limit as on the clinical MNREAD chart (supplement). The persona states the reader’s letter-chart acuity, contrast sensitivity and diagnosis. A model that only copies the stated letter-chart acuity does as well as the acuity rule. Such a model fails for the readers whose two numbers differ.

The persona changes the answers, but toward the stated acuity. Without a persona, the three frontier models copy every sentence correctly. With a persona, they stop at or near the stated letter-chart acuity: Gemini stops within one chart line of it for 50 of 56 readers (Fig. 2a). Thus, the persona changes the answers, but the models only copy the acuity from the prompt. As a result, no persona predicts the reading limit better than the acuity rule, which is off by 0.15 logMAR on average (Fig. 2b): Gemini and Opus are slightly worse than the rule, and Sol is level with it (supplement).

The persona text holds the missing information, but the models do not use it. Some readers read worse than their letter-chart acuity, and some read better. Does a persona show this? That is, when a reader reads worse than their acuity, does the model’s answer for that reader’s persona also read worse than the stated acuity? It does not. We correlate the two differences, the model’s and the reader’s, over the 56 readers (0 means no relation, 1 means full agreement). The correlation is at most 0.20 for every model, and its 95% interval includes zero for every model, so we cannot tell it from no relation. Yet the persona text contains this information. The persona states three facts: letter-chart acuity, contrast sensitivity and diagnosis. A linear regression that uses only these three facts finds which readers read worse than their acuity: its correlation is 0.38 [0.19, 0.57], clearly above zero. The regression cannot simply remember a reader, because we fit it on 55 readers, predict the remaining one, and repeat this for each of the 56 readers. Thus, the models mostly use only the stated acuity. For example, for readers with macular disease, all three personas read smaller print than the readers can. The open models mostly ignore the persona or give no answer (supplement).

A stronger prompt helps only by giving away the answer. A worked example [36] shows one sentence at two print sizes that bracket the reader’s limit. It lowers the error, but no more than a rule that clips the stated acuity to the two sizes. Step-by-step reasoning helps no model (supplement).

An image filter fixes the group average, not the individual. A persona describes the vision loss in words; a filter puts it into the image. The filter starts from the reader’s contrast threshold for each size of detail, which we compute from the reader’s letter-chart acuity and contrast sensitivity [5]. It removes the detail that the reader cannot see and adds noise at the threshold (supplement; similar in spirit to Wolff et al. [53]). On average, this filter works: the mean difference from the reader’s limit is between −0.01 and +0.11 logMAR in five of seven models. For individual readers, it does not. For each reader, we compare the model’s reading limit on the filtered sentences with the reader’s limit. In all seven models, this error is larger than the error of the acuity rule, by 0.09 to 0.19 logMAR (one to two chart lines; Fig. 2b). The filter is built from only the acuity and contrast sensitivity that the persona also states. So it cannot show damage that these two numbers miss, such as the damaged centre of vision in macular disease, which makes sentences harder to read than single letters.

![](images/33542b293d88c3e402077467efc7386e226bd19b424b909b632c0284bed78551.jpg)  
the named type another type normal order no valid answer  
Figure 3. Colour-blind personas give a partial textbook answer that fails for red-blind viewers; changed colours work. Each square is one of 20 cap layouts per type. Red: persona. Blue: no persona, colours changed as that type sees them. Filled: ordered like the named type; dark grey: like another type; open: like a normal viewer; hatched: no valid answer.

Colour-blind personas recite part of a textbook. In the Farnsworth D-15 test, a viewer puts 15 coloured caps in order of hue. People with normal colour vision place them in a smooth circle. Colour-blind viewers confuse some hues, so their order crosses the circle, and the direction of the crossings shows the type: red-blind (protan), green-blind (deutan) or blue-blind (tritan). Published group scores of 39 measured colour-blind viewers (12 protans, 23 deutans and 4 tritans) define the pattern of each type [50]. Individual orders are not published, so we compare the model with each type, not with individual viewers. The persona names the type and its missing cone. For each type, the model orders 20 random cap layouts. We score each order with the standard method of Vingrys and King-Smith [50], which gives the direction and the size of the crossings. A layout counts as correct when its crossings point in the direction of the named type, so the maximum score is 60 (20 per type).

Two simple controls set the bar. Without a persona, every frontier model orders the caps like a normal viewer, so it scores 0 of 60. The textbook answer for each stated type needs no model: it orders the caps along the one colour direction that this type still sees. It scores 60 of 60, and its confusion angles lie close to the measured group means. For comparison, a single deutan order given for every type

(a) Two lines or three? Share preferring two lines  
![](images/6489fd2ad05db09f2bde93209df8a3e8710ff94aeabc0ed7ae4111db7deee3b2.jpg)  
(b) Where to break a two-line subtitle?

![](images/591e05719ff173121f60cbbd7811b2fe477c4030aef3681efac890c0d758efb6.jpg)  
Figure 4. Subtitle personas answer like the model, not like the viewer. (a) Share of viewers who prefer two-line to three-line subtitles (95% intervals; all 19 deaf and hard-of-hearing viewers chose two lines). (b) A different task: for 30 two-line subtitles, viewers chose between a well-split and a poor line break. Each dot is one person (black) or that person’s persona (red). Diamonds: model without a persona. Right: spread across viewers (SD), and agreement, the share of frames on which the model chose the same break as the viewer, with no persona and with that viewer’s persona.

scores 20 of 60.

With a persona (Fig. 3), Gemini and Opus score 37 and 44 of 60, well below the textbook answer for the stated type, and unevenly across types. For deutans, both score 20 of 20. For tritans, they score 15 and 17. For protans, they score only 2 and 7, because they usually give the deutan order instead. Sol’s personas mostly order the caps like a normal viewer, so Sol scores 13 of 60, below even the single deutan order. In contrast, when we change the colours as each type sees them [6, 31] and give no persona, the models score 54 to 59 of 60.

Subtitle viewers test hearing and language, not sight. The SURE study [16, 17, 44–46] recorded 88 to 94 viewers per task in five groups: hearing native speakers of English, Polish and Spanish, and viewers who are deaf or hard of hearing. The viewers did four tasks. In the first, they chose the better of two line breaks for 30 subtitled TV frames; the well-split line keeps a phrase together on one line. In the other tasks, they rated film clips whose subtitles had good or bad line breaks, two or three lines, and slow or fast speed. They rated, for example, speech–subtitle mismatch and annoyance, and stated which subtitles they preferred. We make one persona for each participant. Each persona states these facts in a fixed template (supplement). Each persona gets the same task as its participant. Gemini receives the video with sound; Opus and Sol receive frames that show every subtitle.

The persona does not bring the model closer to the person. The line-break task uses still screens, which we rebuilt from the experiment file with a substitute font and both left–right orders (supplement). In this task, people choose the wellsplit line in 72% of trials, and they disagree with each other. For each participant, we compare their own 30 choices with the choices of their persona and with the choices of the same model without a persona. The agreement is the fraction of the 30 frames on which the two choose the same line break. The persona does not raise this agreement in any frontier model (Fig. 4b; changes −0.001 to +0.002, all intervals include zero). The personas agree with people only about as often as the rule “always choose the well-split line.” Across items, the choices of the Gemini and Sol personas correlate with people’s at only .05 and .14, while two halves of the human sample correlate at .84; the Opus personas choose the well-split line in every trial. The personas also erase the differences between individuals. People differ: some choose the well-split line in 23% of trials, others in all of them. The personas barely differ: the spread (standard deviation) of their shares across participants is 8 to 20 times smaller than people’s, and zero for Opus (Fig. 4b). This holds when we sample five answers per item or ask for choice probabilities (supplement). People differ in a stable way (split-half reliability .88), but the stated facts do not predict who chooses the well-split line more often (supplement), so here the narrow spread is a mismatch with people, not evidence that the personas ignore useful facts. Real examples do carry this information. When the model also sees 10 of the person’s own choices and predicts the other 20, it agrees with the person about as often as a simple predictor that uses the same 10 choices (.72–.76 against .74), and Gemini and Opus recover the human spread. Two controls show which part helps: without the examples, the same persona agrees with people less often, and without the persona, the examples alone do as well. Thus, the examples carry the information, not the persona (supplement).

The differences between groups disappear. A group difference is the difference between two groups on one measure. Deaf and hard-of-hearing viewers prefer two-line subtitles far more often than English viewers: all 19 against 14 of 27 (52%), a difference of 48 percentage points. Every persona, and every frontier model without a persona, prefers two lines, so the difference is zero (Fig. 4a). The model’s default thus matches the deaf and hard-of-hearing viewers and misses the English viewers, about half of whom accept three lines. We test 33 group comparisons from the study. A comparison is fair only if the model notices what changes in it without a persona. No frontier model notices a change in subtitle speed, and some miss other changes, so we drop those comparisons. This leaves 22 of the 33 comparisons for each model. In 7 of these 22, the human difference is clear (its 95% interval excludes zero). For example, bad line breaks make subtitles harder for Spanish viewers than for English viewers. No persona reproduces any of these 7 differences. In most of them, the personas show almost no difference between the groups, and in a few, the difference points the other way (supplement). Over all 33 comparisons, the persona differences do not follow the human ones: their correlation is between −0.17 and +0.23, and every interval includes zero.

![](images/45dfb4598909c266bdd386d069c9dbb92492235ca55f4c65a522cf042434a239.jpg)  
Figure 5. A text description of blur has no effect in any of 14 models; blurred photos lower accuracy in all of them. Model’s accuracy loss divided by people’s (1 = same; 0 = none; 95% intervals). Red: clean photo with the persona. Blue: blurred photo. Grey band: 0.80–1.25. Bold: as large as people’s after Holm correction.

Takeaway. Personas of measured people make the model use the stated facts, a stereotype, or its own default. In reading and subtitles, no persona predicts people better than its task-specific control, and no persona keeps the differences between subtitle-viewer groups.

## 4.2. Can a description replace a changed image?

Why do personas fail? A persona of a person with a visual limit must tell the model about the limit and also make the model lose the information that the limit removes. To test the second job, we use a task where we can also remove the information from the image. The human reference is sighted people who view the changed images.

The blur test. In the model-vs-human benchmark [14], four observers with normal vision named the object in greyscale photos from 16 classes. Each photo appeared for 200 ms at one of eight blur levels. We give each model 160 photos per level in two ways. Either the model sees the blurred photo, or it sees the clean photo with a persona that states the acuity that matches the blur. We convert blur to acuity by assuming that the photo covers $3 ^ { \circ }$ of the visual field. The acuity is in Snellen notation: 20/165 means that the person sees at 20 feet what a typical viewer sees at 165 feet (0.92 logMAR). Figure 1 (top right) shows an example.

![](images/50764d538eb7359889edd87ffcac9d60d4dc56154fe25c40209b9ea0f1f517b3.jpg)  
Figure 6. Inside Qwen3-VL-8B, the blur persona leaves the image information intact. Linear probes of the hidden states on held-out photos, per layer (bands: 95% intervals). (a) The object is as decodable from the image tokens with the blur persona (red) as for the clean photo (black) or an unrelated persona of the same length (yellow, dashed); blurred photos (blue) remove more of it as the blur grows. (b) A probe that tracks real blur (= 1) at the last prompt token reads almost none of the stated blur, no more than for the unrelated persona.

Describing blur has no effect. The blur persona has no effect on recognition in any of the 14 models (Fig. 5). At the strongest stated loss, acuity 20/940, people are at chance, but the models with the persona still name 84 to 98% of the photos correctly (Fig. 1, bottom). The models say that they are less confident, but their answers do not change. For Gemini, the result is the same with the two persona prompts of Natalie et al. [36] (a minimal one and one that states the diagnosis), with eight example human answers, and with step-by-step reasoning. Personas that describe noise or four other image changes have no effect either (supplement). The models know the effect that they do not show. Asked instead to estimate how accurately people with each stated acuity name the objects, the three frontier models give a loss of 0.80 (Gemini), 0.98 (Opus; as large as people’s) and 0.82 (Sol) of the human loss.

Blurring the photo lowers recognition. Blurred photos lower accuracy in every model, and in seven models, including the three core frontier models, the loss is as large as people’s (Fig. 5). The blur at which accuracy falls halfway to chance gives the same picture: it matches people’s in four models, including Gemini and Opus, and no model tolerates more blur than people (supplement). Models and people do not always fail on the same photos, but neither do two people: the agreement between a model and people on which photos are missed is at most κ = .35, and between two people it is κ = .40 (κ = 0: chance agreement; 1: full agreement).

Inside the model, the description changes nothing that blur changes. To see why the description fails, we look inside open models that do the blur task without a persona. First, the answer probabilities. In four models, the probability of the correct object is .94–.95 with the blur persona, the same as without it; when the photo itself is strongly blurred (σ = 15 pixels), it falls to .05–.10, near chance. Second, the hidden states, that is, the output of each transformer block, at every layer of three models. We place the persona before the image, so that the image tokens can take it into account. We train linear probes on the hidden states of some photos and test them on others. The object can be read from the hidden states with the persona as well as without it, while blur removes most of it (Fig. 6a). A second probe, trained on the hidden states of blurred photos, reads how blurred the photo is. Its reading rises one-to-one with real blur, but with the stated blur of the persona it rises by at most 0.17 of that (Gemma-4-12B), 0.08 (Qwen3-VL-8B) and 0.04 (InternVL3.5-8B) (Fig. 6b). The persona does change the hidden states, but almost perpendicular to the change that blur makes (cosine 0.1–0.2). To check that the probes can see a text effect, we add an instruction that does change the answers (“name a different object”): it changes 82–99% of the answers, and the probes detect the change. Thus, in these models the persona changes neither the image information nor the decision; it adds only a small drop in confidence. Figure 6 shows Qwen3-VL-8B; Gemma-4-12B and InternVL3.5-8B give the same result (supplement).

Takeaway. The information that a visual limit removes reaches the model through the image, not through a description. A description ofthe limit leaves the answers unchanged.

## 4.3. What does text give the model instead?

If a persona does not change what the model sees, what does text change? A diagnostic test answers this: the model estimates how a group of people answered.

Models know how often illusions fool people, but are rarely fooled. Makowski et al. [33] recorded fast choices from 256 people on ten types of visual illusion, such as “Which red line is longer?” We remake the images with the same software [32]. The illusion effect is the error rate when the context misleads minus the error rate when the context helps. In their own answers, Opus shows 1% of people’s illusion effect, Sol 8% and Gemini 20% (Fig. 7b). We also ask each model, in the same reply, for the percentage of adults who choose each option. These estimates show 65% (Sol), 76% (Opus) and 85% (Gemini) of people’s illusion effect. They also find the images that mislead people better than the models’ own answers do (correlation .66–.76 against .10–.30).

Estimates do not replace a simulated viewer. Three limits keep estimates from standing in for answers. First, an estimate is a single number for a group, such as “adults”; it gives no individual answers and no spread between people, which a user study needs. Second, its errors are large and cannot be found without human data: Gemini and Sol estimate almost no Poggendorff effect, although it misleads people strongly, and Opus and Gemini estimate the weak Zollner effect four to five times too large. Third, estimates¨ are weaker for single items than for whole curves: for blur, per-photo estimates recover 0.42 to 0.74 of the human loss, against 0.80 to 0.98 for one curve over all levels (supplement). An estimate shows what the model knows about people; it does not replace measuring them.

![](images/8ca3840ea8fa9fac156665ea56db5d997366024753a6a7df294ac58f3f61e65b.jpg)  
model answers itself model estimates people's answers Gemini Opus Sol Qwen2.5-VL-72B people = 1 as large as people's  
Figure 7. Models know how often illusions fool people, but their own answers are rarely fooled. (a) Example images; numbers are human error rates. (b) Illusion effect, relative to people, when Gemini, Opus and Sol answer themselves (blue) and estimate people’s answers (purple). Grey band: 0.80–1.25. Bottom row: human effect. (c) Qwen2.5-VL-72B matches people on Muller-Lyer (0.99), but only outward¨ fins mislead it, while both fin types mislead people.

A matching average can hide a different cause. Of 11 open models, only Qwen2.5-VL-72B passes our control images, and its Muller-Lyer effect matches people’s (relative¨ effect 0.99). Control figures show why (Fig. 7c). In people, both outward and inward fins mislead; in the model, only outward fins do (−0.12 for inward fins). Asked for lengths in pixels, it adds about half a pixel per pixel of outward fin, and moving the fins 24 pixels away from the line removes 82% of its effect. The model counts part of the fins as line, which happens to match people.

A persona can turn an illusion on. A young-adult persona raises Gemini’s illusion effect from 20% to 84% of the human value, and its errors fall on the images that mislead people (correlation .52 across images, against .66 for its estimates); the personas of Opus and Sol barely change their answers. Text can thus produce a human-like error when the model knows which way people err; this is recall of the known effect, not perception of the image. An age persona does not reproduce the measured age differences (supplement).

Takeaway. Text lets a model recall how people answer, but not see what they see: in its own answers, the model is rarely fooled by the illusions that fool people.

## 5. Discussion

Telling is not seeing. A persona can make a model say what a person would say, but only when the model already knows that answer: the direction of an illusion, the textbook deutan order, or the rule that the reading limit equals the stated acuity. It never makes the model see what the person sees. When the answer depends on what the person’s vision lacks, or on differences for which the model has no rule, the persona changes nothing. Inside open models, the description changes neither the image information nor the decision. A simulator need not change perception: a decision rule built from the model’s own estimates could reproduce group-level errors. Current personas do neither. The frontier models answers show the same pattern. The tests also differ in what the stated facts allow: in reading, they predict which readers depart from the rule, which the personas ignore; in the line-break task, they predict no individual choices; in colour sorting, the textbook order is a ceiling.

Limitations. We test short template personas; richer descriptions are a natural next test. The blur test is a controlled proxy, not clinical low vision, and people saw each photo for 200 ms. Several reference groups are small (4 blur observers, 4 tritans, 9 to 27 viewers per subtitle group). Our inputs differ from what people saw: the reading sentences are always legible, the line-break screens use a substitute font, and Opus and Sol see frames without sound.

How to evaluate a simulated viewer. We suggest four checks: compare the persona with people who have the condition; compare it with a rule from the stated facts; check items and people, not only averages; and keep the model’s estimates apart from its own answers. A persona can point to a possible barrier, but only tests with people who have that need can show whether the barrier is real.

## References

[1] Gati Aher, Rosa I. Arriaga, and Adam Tauman Kalai. Using large language models to simulate multiple humans and replicate human subject studies. In International Conference on Machine Learning (ICML), 2023. 1

[2] Lisa P. Argyle, Ethan C. Busby, Nancy Fulda, Joshua R. Gubler, Christopher Rytting, and David Wingate. Out of one, many: Using language models to simulate human samples. Political Analysis, 31(3):337–351, 2023. 1

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025. 3

[4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025. 3

[5] Peter G. J. Barten. Contrast Sensitivity ofthe Human Eye and Its Effects on Image Quality. SPIE Press, 1999. 4

[6] Hans Brettel, Franc¸oise Vienot, and John D. Mollon. Comput-´ erized simulation of color appearance for dichromats. Journal of the Optical Society of America A, 14(10):2647–2655, 1997. 5

[7] Myra Cheng, Tiziano Piccardi, and Diyi Yang. CoMPosT: Characterizing and evaluating caricature in LLM simulations. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023. 2

[8] Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, et al. Molmo2: Open weights and data for visionlanguage models with video understanding and grounding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 3

[9] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: Multimodal large language models can see but not perceive. In European Conference on Computer Vision (ECCV), pages 148–166, 2024. 2

[10] Qingying Gao, Roberto Manduchi, Pradeep Y. Ramulu, Gordon E. Legge, and Yingzi Xiong. VI-OCR: “Visually Impaired” optical character recognition pipeline for text accessibility assessment. Scientific Reports, 16:1269, 2026. Published online December 2025. 1

[11] Robert Geirhos, Carlos R. Medina Temme, Jonas Rauber, Heiko H. Schutt, Matthias Bethge, and Felix A. Wichmann.¨ Generalisation in humans and deep neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2018. 2

[12] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. In International Conference on Learning Representations (ICLR), 2019.

[13] Robert Geirhos, Kristof Meding, and Felix A. Wichmann. Beyond accuracy: Quantifying trial-by-trial behaviour of CNNs and humans by measuring error consistency. In Advances in Neural Information Processing Systems (NeurIPS), 2020. 3

[14] Robert Geirhos, Kantharaju Narayanappa, Benjamin Mitzkus, Tizian Thieringer, Matthias Bethge, Felix A. Wichmann, and

Wieland Brendel. Partial success in closing the gap between human and machine vision. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 2, 6

[15] Gemma Team. Gemma 4 technical report. arXiv preprint arXiv:2607.02770, 2026. 3

[16] Olivia Gerber-Moron and Agnieszka Szarkowska. Line´ breaks in subtitling: An eye tracking study on viewer preferences. Journal ofEye Movement Research, 11(3):2, 2018. 5

[17] Olivia Gerber-Moron, Agnieszka Szarkowska, and Bencie´ Woll. The impact of text segmentation on subtitle reading. Journal ofEye Movement Research, 11(4):2, 2018. 5

[18] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusion-Bench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14375–14385, 2024. 2

[19] Shashank Gupta, Vaishnavi Shrivastava, Ameet Deshpande, Ashwin Kalyan, Peter Clark, Ashish Sabharwal, and Tushar Khot. Bias runs deep: Implicit reasoning biases in personaassigned LLMs. In International Conference on Learning Representations (ICLR), 2024. 2

[20] Pengrui Han, Rafal Kocielnik, Peiyang Song, Ramit Debnath, Dean Mobbs, Anima Anandkumar, and R. Michael Alvarez. The personality illusion: Revealing dissociation between self-reports & behavior in LLMs. arXiv preprint arXiv:2509.03730, 2025. 2

[21] Kazuki Hayashi, Shintaro Ozaki, Yusuke Sakai, Hidetaka Kamigaito, and Taro Watanabe. Diagnosing vision language models’ perception by leveraging human methods for color vision deficiencies. In Conference ofthe European Chapter of the Associationfor Computational Linguistics (EACL), 2026. arXiv:2505.17461. 1

[22] Pablo Hernandez-C´ amara, Alexandra Gomez-Villa,´ Jose Manuel Jaen-Lorites, Jorge Vila-Tom ´ as, Valero Laparra,´ and Jesus Malo. Contrast sensitivity in multimodal large´ language models: A psychophysics-inspired evaluation. Neural Networks, 201:108903, 2026. 2

[23] Sture Holm. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2):65–70, 1979. 3

[24] Lukas S. Huber, Robert Geirhos, and Felix A. Wichmann. The developmental trajectory of object recognition robustness: Children are like small adults but unlike big deep neural networks. Journal ofVision, 23(7):4, 2023. 2

[25] Daniel Lakens. Equivalence tests: A practical primer for¨ t tests, correlations, and meta-analyses. Social Psychological and Personality Science, 8(4):355–362, 2017. 3

[26] Gordon E. Legge. Psychophysics ofReading in Normal and Low Vision. Lawrence Erlbaum Associates, Mahwah, NJ, 2007. 4

[27] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy visual task transfer. Transactions on Machine Learning Research, 2025. arXiv:2408.03326. 3

[28] Xiaomin Li, Yuexing Hao, Jianheng Hou, Jintao Huang, Qianfeng Wen, et al. MatrAIx: Simulating the world with 8.3 billion persona agents. arXiv preprint arXiv:2608.04205, 2026. 1, 2

[29] Yijun Liang, Ming Li, Chenrui Fan, Ziyue Li, Dang Nguyen, Kwesi Cobbina, Shweta Bhardwaj, Jiuhai Chen, Fuxiao Liu, and Tianyi Zhou. ColorBench: Can VLMs see and understand the colorful world? A comprehensive benchmark for color perception, reasoning, and robustness. In Advances in Neural Information Processing Systems (NeurIPS), 2025. 2

[30] Jiaying Lin, Shuquan Ye, Dan Xu, Wanli Ouyang, and Rynson W. H. Lau. Do MLLMs exhibit human-like perceptual behaviors? HVSBench: A benchmark for MLLM alignment with human perceptual behavior. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 1818–1827, 2026. 2

[31] Gustavo M. Machado, Manuel M. Oliveira, and Leandro A. F. Fernandes. A physiologically-based model for simulation of color vision deficiency. IEEE Transactions on Visualization and Computer Graphics, 15(6):1291–1298, 2009. 5

[32] Dominique Makowski, Zen J. Lau, Tam Pham, W. Paul Boyce, and S. H. Annabel Chen. A parametric framework to generate visual illusions using Python. Perception, 50(11):950–965, 2021. 7

[33] Dominique Makowski, An Shu Te, Stephanie Kirk, Ngoi Zi Liang, and S. H. Annabel Chen. A novel visual illusion paradigm provides evidence for a general factor of illusion sensitivity and personality correlates. Scientific Reports, 13: 6594, 2023. 2, 7

[34] Nicole Meister, Carlos Guestrin, and Tatsunori Hashimoto. Benchmarking distributional alignment of large language models. In Conference ofthe Nations ofthe Americas Chapter of the Association for Computational Linguistics (NAACL), pages 24–49, 2025. 2

[35] Microsoft, Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, et al. Phi-4-Mini technical report: Compact yet powerful multimodal language models via mixture-of-LoRAs. arXiv preprint arXiv:2503.01743, 2025. 3

[36] Rosiana Natalie, Wenqian Xu, Ruei-Che Chang, Rada Mihalcea, and Anhong Guo. Not there yet: Evaluating vision language models in simulating the visual perception of people with low vision. arXiv preprint arXiv:2508.10972, 2025. 1, 2, 4, 7

[37] NVIDIA. Nemotron 3 Nano Omni: Efficient and open multimodal intelligence. arXiv preprint arXiv:2604.24954, 2026. 3

[38] Joon Sung Park, Carolyn Q. Zou, Jonne Kamphorst, Niles Egan, Aaron Shaw, Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Percy Liang, Robb Willer, and Michael S. Bernstein. LLM agents grounded in self-reports enable general-purpose simulation of individuals. arXiv preprint arXiv:2411.10109, 2024. First version titled “Generative Agent Simulations of 1,000 People”. 1

[39] Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind. In Asian Conference on Computer Vision (ACCV), pages 18–34, 2024. 2

[40] Leonard Salewski, Stephan Alaniz, Isabel Rio-Torto, Eric Schulz, and Zeynep Akata. In-context impersonation reveals large language models’ strengths and biases. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 2

[41] Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. Whose opinions do language models reflect? In International Conference on Machine Learning (ICML), 2023. 2

[42] Haz Sameen Shahgir, Khondker Salman Sayeed, Abhik Bhattacharjee, Wasi Uddin Ahmad, Yue Dong, and Rifat Shahriyar. IllusionVQA: A challenging optical illusion dataset for vision language models. In Conference on Language Modeling (COLM), 2024. 2

[43] Prabal Shrestha, Bohan Jiang, Haoning Xue, Huan Liu, and Xinyi Zhou. Multimodal large language models as synthetic participants in video-based studies: An evaluation. arXiv preprint arXiv:2606.07541, 2026. 2

[44] Agnieszka Szarkowska and Olivia Gerber-Moron. Viewers´ can keep up with fast subtitles: Evidence from eye movements. PLOS ONE, 13(6):e0199331, 2018. 5

[45] Agnieszka Szarkowska and Olivia Gerber-Moron. SURE´ project dataset. RepOD, doi:10.18150/repod.4469278, 2018. 2

[46] Agnieszka Szarkowska and Olivia Gerber-Moron. Two or´ three lines: A mixed-methods study on subtitle processing and preferences. Perspectives: Studies in Translation Theory and Practice, 27(1):144–164, 2019. 5

[47] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? Exploring the visual shortcomings of multimodal LLMs. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9568–9578, 2024. 2

[48] Yi-Lin Tsai and Yung-Hsiu Lai. Seeing is not perceiving: When synthetic consumers can and cannot pretest visual marketing. arXiv preprint arXiv:2609.25677, 2026. 2

[49] Tomer D. Ullman. The illusion-illusion: Vision language models see illusions where there are none. arXiv preprint arXiv:2412.18613, 2024. 2

[50] Algis J. Vingrys and P. Ewen King-Smith. A quantitative scoring technique for panel tests of color vision. Investigative Ophthalmology & Visual Science, 29(1):50–63, 1988. 2, 5

[51] Angelina Wang, Jamie Morgenstern, and John P. Dickerson. Large language models that replace human participants can harmfully misportray and flatten identity groups. Nature Machine Intelligence, 7(3):400–411, 2025. 2

[52] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, et al. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 3

[53] Max Wolff, Thomas Klein, Evgenia Rusak, Felix Wichmann, and Wieland Brendel. Low-pass filtering improves behavioral alignment of vision models. In International Conference on Learning Representations (ICLR), 2026. arXiv:2602.13859. 2, 4

[54] Ying-Zi Xiong, Quan Lei, Aurelie Calabr´ ese, and Gordon E.\` Legge. Simulating visibility and reading performance in low vision. Frontiers in Neuroscience, 15:671121, 2021. 2, 3

[55] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, et al. Qwen3-Omni technical report. arXiv preprint arXiv:2509.17765, 2025. 3

[56] Yichi Zhang, Jiayi Pan, Yuchen Zhou, Rui Pan, and Joyce Chai. Grounding visual illusions in language: Do visionlanguage models perceive illusions like humans? In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023. 2

[57] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, et al. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 3